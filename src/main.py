import lightning as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from jsonargparse import lazy_instance
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from diffsynth.pipelines.wan_video_panshot import WanVideoPipeline, ModelConfig
import wandb
import os
from src.dataset import DemoDataset
from diffsynth import save_video
from src.camera_control import patch_dit, enable_grad
from typing import Literal
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import numpy as np
from tqdm.auto import tqdm
from typing import Optional


class DemoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        panshot_data_root: Path = Path("data/UCPE"),
        re10k_data_root: Path = Path("data/RealEstate10k"),
        input_file: Path = Path("demo/teaser.json"),
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        self.hparams.num_frames = self.trainer.model.hparams.num_frames        
        normalization_file = Path(self.hparams.re10k_data_root) / "traj_normalization.txt"
        self.hparams.re10k_normalize_traj = float(np.loadtxt(normalization_file))

    def predict_dataloader(self):
        dataset = DemoDataset(self.hparams)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)


class DemoModule(pl.LightningModule):
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        learning_rate: float = 1e-4,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        ckpt_path: Path = None,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        tiled: bool = False,
        camera_condition: str = "relray_absmap",
        adaptation_method: Literal[
            "before",
            "after",
            "parallel",
        ] = "parallel",
        ti2v_input_image_prob: float = 0.5,
        attn_compress: int = 8,
        num_predict: Optional[int] = None,
    ):
        super().__init__()
        file_patterns = [
            "models_t5_umt5-xxl-enc-bf16.pth",
            "diffusion_pytorch_model*.safetensors",
            "Wan2.1_VAE.pth",
        ]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(model_id=model_id, origin_file_pattern=pattern, offload_device="cpu")
                for pattern in file_patterns
            ]
        )

        keywords = patch_dit(
            self.pipe, camera_condition, height, width,
            attn_compress=attn_compress, adaptation_method=adaptation_method
        )
        enable_grad(self.pipe, keywords)

        self.strict_loading = False
        if ckpt_path is not None:
            print(f"Loading weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        self.save_hyperparameters()

    def setup(self, stage=None):
        self.pipe.device = self.device

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for i in range(self.hparams.num_predict or 1):
            video = self(batch, seed=i)
            is_i2v = "input_image" in batch and self.pipe.dit.fuse_vae_embedding_in_latents
            video_folder = "i2v" if is_i2v else "t2v"
            split = "demo"
            self.save_output(
                video,
                batch,
                split=split,
                video_folder=video_folder,
                quality=8,
                suffix=f"-{i}" if self.hparams.num_predict else None
            )
            if is_i2v:
                del batch["input_image"]
                self.predict_step(batch, batch_idx, dataloader_idx)

    def save_output(self, video, batch, split, video_folder, step=None, quality=5, suffix=None):
        video_id = batch["video_id"][0]

        video_prefix = os.path.join(self.logger.save_dir, split, video_folder, video_id)
        if step is not None:
            video_prefix = f"{video_prefix}-{step}"
        if suffix is not None:
            video_prefix = video_prefix + suffix
        video_path = video_prefix + ".mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        save_video(video, video_path, fps=self.hparams.fps, quality=quality)

        reference_path = os.path.join(self.logger.save_dir, split, "reference", f"{video_id}.mp4")
        os.makedirs(os.path.dirname(reference_path), exist_ok=True)
        if not os.path.exists(reference_path) and "video" in batch:
            reference_video = self.pipe.vae_output_to_video(batch["video"])
            save_video(reference_video, reference_path, fps=self.hparams.fps, quality=quality)

        caption_path = os.path.join(self.logger.save_dir, split, "caption", f"{video_id}.txt")
        os.makedirs(os.path.dirname(caption_path), exist_ok=True)
        if not os.path.exists(caption_path):
            with open(caption_path, "w") as f:
                f.write(batch["caption"][0])

        print(f"Saved video to {video_path}")

        return video_path, reference_path

    def forward(self, batch, seed=None):
        video = self.pipe(
            prompt=batch["caption"][0],
            input_image=batch.get("input_image", None),
            camera_control_panshot={k: batch[k] for k in ["pose", "xi", "x_fov"]},
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            num_inference_steps=self.hparams.num_inference_steps,
            tiled=self.hparams.tiled,
            seed=seed,
            height=self.hparams.height,
            width=self.hparams.width,
            num_frames=self.hparams.num_frames,
        )
        return video


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({
            "checkpoint.dirpath": os.path.join(self.trainer_defaults["default_root_dir"], "checkpoints"),
            "checkpoint.save_last": True,
        })


def main():
    torch.set_float32_matmul_precision('high')

    wandb_id = os.environ.get("WANDB_RUN_ID", wandb.util.generate_id())
    exp_dir = os.path.join("logs", wandb_id)

    cli = MyCLI(
        model_class=DemoModule,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf", "default_env": True},
        seed_everything_default=int(os.environ.get("LOCAL_RANK", 0)),
        trainer_defaults={
            "accelerator": "gpu",
            "devices": "auto",
            "benchmark": True,
            "max_epochs": 10,
            "precision": "bf16-true",
            "default_root_dir": exp_dir,
        },
        
    )


if __name__ == "__main__":
    main()
