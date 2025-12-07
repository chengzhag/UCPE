from torch.utils.data import Dataset
from einops import rearrange, repeat
from pathlib import Path
import jsonlines
import json
import numpy as np


class DemoDataset(Dataset):
    def __init__(self, args, split=None, load_keys=["pose"], video_ids=None, result_root=None):
        self.args = args
        self.panshot_data_root = Path(args.panshot_data_root)
        self.load_keys = load_keys
        with open(args.input_file, "r") as f:
            self.metas = json.load(f)
        self.normalize_traj = getattr(args, "re10k_normalize_traj", None)

        near_depth_file = self.panshot_data_root / "PanFlow" / "near_plane_depth-test.jsonl"
        near_depths = {}
        with jsonlines.open(near_depth_file) as reader:
            for obj in reader:
                video_id = obj["video_id"]
                clip_id = obj["clip_id"]
                near_depths[f"{video_id}-{clip_id}"] = obj["near_depth"]
        print(f"Loaded {len(near_depths)} near depth entries.")
        for meta in self.metas:
            pose_file = Path(meta["pose_path"])
            if pose_file.suffix == ".npy":
                clip_name = pose_file.stem.rsplit("-", 2)[0]
                meta["near_depth"] = near_depths[clip_name]

        for idx, data in enumerate(self.metas):
            pose_file = Path(data["pose_path"])
            prefix = f"{idx}-{pose_file.stem}-fov{int(data['x_fov'])}-xi{data['xi']:.2f}-"
            data["video_id"] = prefix + data["caption"][:50].replace(" ", "_")

        if video_ids is not None:
            self.metas = [meta for meta in self.metas if meta["video_id"] in video_ids]
            print(f"Filtered by video_ids, {len(self.metas)} videos remaining.")

        self.result_root = None
        if result_root is not None:
            self.result_root = Path(result_root)
            new_metas = []
            metas = {m["video_id"]: m for m in self.metas}
            results = list(self.result_root.glob("*.mp4"))
            results.sort()
            for v in results:
                video_id = v.stem.rsplit("-", 1)[0]
                if video_id in metas:
                    meta = metas[video_id].copy()
                    meta["result_path"] = str(v)
                    new_metas.append(meta)
            self.metas = new_metas
            print(f"Filtered by result_root, {len(self.metas)} videos remaining.")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        data = self.metas[idx].copy()

        if "result" in self.load_keys:
            path = data["result_path"]
            from decord import VideoReader, cpu
            vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            video = vr.get_batch(range(len(vr))).asnumpy()
            video = video.astype(np.float32)
            video = video / 255.0 * 2 - 1  # to [-1, 1]
            video = rearrange(video, "T H W C -> C T H W")
            data["result"] = video
            data["fps"] = float(vr.get_avg_fps())

        if "pose" in self.load_keys:
            pose_file = Path(data["pose_path"])
            if pose_file.suffix == ".txt":
                with open(pose_file, "r") as f:
                    lines = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("http"):
                            lines.append(line)
                            if len(lines) >= self.args.num_frames:
                                break
                poses = []
                for line in lines:
                    parts = line.split()
                    pose = np.array(list(map(float, parts[7:]))).reshape(3, 4)
                    poses.append(pose)
                poses = np.stack(poses, axis=0)  # [T, 3, 4]
                last_row = repeat(np.array([0,0,0,1], dtype=poses.dtype), "n -> t 1 n", t=poses.shape[0])
                w2c = np.concatenate([poses, last_row], axis=-2)  # (T, 4, 4)
                c2w = np.linalg.inv(w2c)  # (T, 4, 4)
                w2c0= np.linalg.inv(c2w[0])  # (4, 4)
                c2w = w2c0[None] @ c2w  # (T, 4, 4)
                poses = c2w[:, :3]  # (T, 3, 4)
                if self.normalize_traj is not None:
                    poses[..., 3] *= self.normalize_traj
                data["pose"] = poses
            elif pose_file.suffix == ".npy":
                pose = np.load(pose_file)
                pose[..., 3] /= data["near_depth"]
                    
                if getattr(self.args, "zero_first_yaw", True):
                    # Rotate all cameras around world-y
                    # to move forward-z of the first camera to y-z plane
                    forward = pose[0, :, 2]  # (3,) the z-axis of the first camera in world
                    forward_xy = np.array([forward[0], 0, forward[2]])  # project to x-z plane
                    forward_xy /= np.linalg.norm(forward_xy) + 1e-8

                    # compute rotation angle theta = atan2(x, z)
                    theta = np.arctan2(forward_xy[0], forward_xy[2])

                    # rotation matrix around world-y by -theta
                    c, s = np.cos(-theta), np.sin(-theta)
                    R_y = np.array([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]], dtype=pose.dtype)

                    # apply rotation to all camera extrinsics
                    pose[..., :3] = (R_y[None] @ pose[..., :3])
                    pose[..., 3] = (R_y[None] @ pose[..., 3:4]).squeeze(-1)
                else:
                    last_row = repeat(np.array([0,0,0,1], dtype=pose.dtype), "n -> t 1 n", t=pose.shape[0])
                    c2w = np.concatenate([pose, last_row], axis=-2)  # (T, 4, 4)
                    w2c0= np.linalg.inv(c2w[0])  # (4, 4)
                    c2w = w2c0[None] @ c2w  # (T, 4, 4)
                    pose = c2w[:, :3]  # (T, 3, 4)

                data["pose"] = pose
            else:
                raise NotImplementedError(f"Unsupported pose file: {pose_file}")

        return data
