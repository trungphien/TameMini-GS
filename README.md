
# TameMini-GS: A Hybrid Approach for Efficient and Compact 3D Gaussian Splatting

### (1) Setup
This code has been tested with Python 3.8, torch 1.12.1, CUDA 11.6.

- Clone the repository 
```
git clone https://github.com/trungphien/TameMini-GS.git
```
- Setup python environment
```
conda create -n tame_mini python=3.8
conda activate tame_mini
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

- Download datasets: [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [T&T+DB COLMAP](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

### (2) Training & Evaluation

Training scripts for TameMini-GS are in `tame_mini`:
```
cd tame_mini
```
- Train with train/test split:
```
# mipnerf360 outdoor
python train.py -s <dataset path> -m <model path> -i images_4 --eval --imp_metric outdoor --config_path ../config/fast
# mipnerf360 indoor
python train.py -s <dataset path> -m <model path> -i images_2 --eval --imp_metric indoor --config_path ../config/fast
# t&t
python train.py -s <dataset path> -m <model path> --eval --imp_metric outdoor --config_path ../config/fast
# db
python train.py -s <dataset path> -m <model path> --eval --imp_metric indoor --config_path ../config/fast
```

- Modified full_eval script:
```
python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>
```
### (3) Compression

- Scripts for compression are in `ms_c`:
```
cd ms_c
```
- Compress, decompress and evaluate (use the model path of the pretrained Mini-Splatting):
```
python run.py -s <dataset path> -m <model path>
```
**Acknowledgement.** This project is built upon [Mini-Splatting](https://github.com/fatPeter/mini-splatting), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs).








