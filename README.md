# 📷 UCPE

### Unified Camera Positional Encoding for Controlled Video Generation
Cheng Zhang, Boying Li, Meng Wei, Yan-Pei Cao, Camilo Cruz Gambardella, Dinh Phung, Jianfei Cai

### [Paper (Coming Soon!)]() | [Video](https://youtu.be/HVgpsOtcIyI)

[![Watch the video](images/thumbnail.png)](https://www.youtube.com/watch?v=HVgpsOtcIyI)
*Our UCPE introduces a geometry-consistent alternative to Plücker rays as one of the core contributions, enabling better generalization in Transformers and offering a foundation for future camera-aware architectures.

## 🚀 TLDR

📷 UCPE integrates **Relative Ray Encoding**—which delivers significantly better generalization than Plücker across diverse camera motion, intrinsics and lens distortions—with **Absolute Orientation Encoding** for controllable pitch and roll, enabling a unified camera representation for Transformers and state-of-the-art camera-controlled video generation with **less than 1% extra parameters**.

## 🔔 Coming Soon

- 📁 **PanShot Dataset And Curation Code** (controllable camera data synthesized from [PanFlow](https://github.com/chengzhag/PanFlow))
- 🎯 **Full Training, Evaluation Code** for UCPE

## 🛠️ Installation

```bash
conda create -n UCPE python=3.11 -y
conda activate UCPE
conda install -c conda-forge "ffmpeg<8" libiconv libgl -y
pip install -r requirements.txt
pip install --no-build-isolation --no-cache-dir flash-attn==2.8.0.post2
pip install -e .

cd thirdparty/equilib
pip install -e .
```

## ⚡ Quick Demo

Download our finetuned weights from [OneDrive](https://monashuni-my.sharepoint.com/:f:/g/personal/cheng_zhang_monash_edu/IgCoTNrYOJRJRKtk5A6I1yiCAR9c64-BOrsId5GYsUxE9y4?e=hD26qU) and put it in `logs/` folder. Then run:

```bash
bash scripts/demo.sh
```

The generated videos will be saved in `logs/6wodf04s/demo`.

## 💡 Acknowledgements

Our paper cannot be completed without the amazing open-source projects [Wan2.1](https://github.com/Wan-Video/Wan2.1), [AC3D](https://github.com/snap-research/ac3d), [ReCamMaster](https://github.com/KlingTeam/ReCamMaster), [CameraCtrl](https://github.com/hehao13/CameraCtrl)...

Also check out our Pan-Series works [PanFlow](https://github.com/chengzhag/PanFlow), [PanFusion](https://github.com/chengzhag/PanFusion) and [PanSplat](https://github.com/chengzhag/PanSplat) towards 3D scene generation with panoramic images!
