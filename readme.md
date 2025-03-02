# V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion 
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.08402)
[![code](https://img.shields.io/badge/dataset-link-blue)](http://39.98.109.195:1000/)
:balloon:Congratulations to our L4DR paper was accepted by **CVPR2025** ! :balloon:
<div align="center">
  <img src="images/logo.png" width="200"/>
</div>



## :balloon: Introduction
:wave: This is the official repository for the V2X-R, including the V2X-R dataset and the implementation of the benchmark model, and MDD module. 

This repo is also a unified and integrated multi-agent collaborative perception framework for **LiDAR-based**, **4D radar-based**, **LiDAR-4D radar fusion** strategies!

### Features

- :sparkles: Dataset Support
  - [x] V2X-R
  - [x] OPV2V
  - [x] DAIR-V2X

- :sparkles: Modality Support
  - [x] LiDAR
  - [x] 4D Radar
  - [x] LiDAR-4D Radar Fusion
    
- :sparkles: SOTA collaborative perception method support
    - [x] Late Fusion
    - [x] Early Fusion
    - [x] [When2com (CVPR2020)](https://arxiv.org/abs/2006.00176)
    - [x] [V2VNet (ECCV2020)](https://arxiv.org/abs/2008.07519)
    - [x] [PFA-Net (ITSC2021)](https://ieeexplore.ieee.org/abstract/document/9564754)
    - [x] [RTNH (NIPS2022)](https://arxiv.org/abs/2206.08171)
    - [x] [DiscoNet (NeurIPS2021)](https://arxiv.org/abs/2111.00643)
    - [x] [V2X-ViT (ECCV2022)](https://arxiv.org/abs/2203.10638)
    - [x] [CoBEVT (CoRL2022)](https://arxiv.org/abs/2207.02202)
    - [x] [Where2comm (NeurIPS2022)](https://arxiv.org/abs/2209.12836)
    - [x] [CoAlign (ICRA2023)](https://arxiv.org/abs/2211.07214)
    - [x] [BM2CP (CoRL2023)](https://arxiv.org/abs/2310.14702)
    - [x] [SCOPE (ICCV2023)](https://arxiv.org/abs/2307.13929)
    - [x] [How2comm (NeurIPS2023)](https://openreview.net/pdf?id=Dbaxm9ujq6)
    - [x] [InterFusion (IROS2023)](https://ieeexplore.ieee.org/document/9982123)
    - [x] [L4DR (Arxiv2024)](https://arxiv.org/abs/2408.03677)
    - [x] [SICP (IROS2024)](https://arxiv.org/abs/2312.04822)

- :sparkles: Visualization
  - [x] BEV visualization
  - [x] 3D visualization

## :balloon: V2X-R Dataset Manual 
The first V2X dataset incorporates LiDAR, camera, and **4D radar**. V2X-R contains **12,079 scenarios** with **37,727 frames of LiDAR and 4D radar point clouds**, **150,908 images**, and **170,859 annotated 3D vehicle bounding boxes**.
<div align="center">
  <img src="images/radar_sup.png" width="600"/>
</div>



### Dataset Collection
Thanks to the [CARLA](https://github.com/carla-simulator/carla) simulator and the [OpenCDA](https://github.com/ucla-mobility/OpenCDA) framework, our V2X-R simulation dataset was implemented on top of them. In addition, our dataset route acquisition process partly references [V2XViT](https://github.com/DerrickXuNu/v2x-vit), which researchers can reproduce according to the data_protocol in the dataset.

### Download and Decompression
:ledger: Log in [here](http://39.98.109.195:1000/) using the username "Guest" and the password "guest_CMD" to download the dataset.


Since the data is large (including 3xLiDAR{normal, fog, snow}, 1xradar, 4ximages for each agent). We have compressed the sequence data of each agent, you can refer to this code for batch decompression after downloading.
```python
import os
import subprocess
def decompress_v2x_r(root_dir, save_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.7z'):
                file_path = os.path.join(root, file)
                path = root.split('/')
                extract_path = os.path.join(save_dir, path[-2], path[-1], file[:-3])
                os.makedirs(extract_path, exist_ok=True)
                subprocess.run(['7z', 'x', '-o' + extract_path + '/', file_path])

data_directory = #downloaded dataset path e.g: '/mnt/16THDD-2/hx/V2X-R_Dataset(compressed)'
output_directory =  #output dataset path  e.g: '/mnt/16THDD-2/hx/t'
decompress_v2x_r(data_directory, output_directory)
```

### Structure
:open_file_folder: After download and decompression are finished, the dataset is structured as following:

```sh
V2X-R # root path of v2x-r output_directory 
├── train
│   ├──Sequence name (time of data collection, e.g. 2024_06_24_20_24_02)
│   │   ├──Agent Number ("-1" denotes infrastructure, otherwise is CAVs)
│   │   │   ├──Data (including the following types of data)
│   │   │   │ Timestamp.Type, eg.
│   │   │   │ - 000060_camerai.png (i-th Camera),
│   │   │   │ - 000060.pcd (LiDAR),
│   │   │   │ - 000060_radar.pcd (4D radar),
│   │   │   │ - 000060_fog.pcd (LiDAR with fog simulation),
│   │   │   │ - 000060_snow.pcd (LiDAR with snow simulation),
│   │   │   │ - 000060.yaml (LiDAR with fog simulation)
├── validate
│   ├──...
├── test
│   ├──...

```

### Calibration
We provide calibration information for each sensor (LiDAR, 4D radar, camera) of each agent for inter-sensor fusion. In particular, the exported 4D radar point cloud has been converted to the LiDAR coordinate system of the corresponding agent in advance of fusion, so the 4D radar point cloud is referenced to the LiDAR coordinate system.


## :balloon: Benchmark and Models Zoo

### Introduction
All benchmark model can be downloaded in [here](http://39.98.109.195:1000/) using the username "Guest" and the password "guest_CMD".

If you need to run the following pre trained models:
1. Download the corresponding pre-trained model to **folder A** and **rename it as 0.pth**
2. Copy the corresponding configuration file to folder A and **rename it as config.yaml**
3. Run the command according to the "Test the model" section, with the parameter *--model_dir A --eval_ epoch 0*
4. Modify evalsim in the configuration file to test the results under different simulated weather conditions
5. *Please note: Due to code version switching, there may be slight differences between the reproduced results and the reported accuracy. It is recommended to use the reproduced results as the standard. If you find that the difference is unacceptable, there may be some problem, please raise an issue.*

### 4DRadar-based Cooperative 3D Detector (no-compression)
| **Method** | **Validation (IoU=0.3/0.5/0.7)** | **Testing (IoU=0.3/0.5/0.7)** | **Config** |  **Model**  |
|:--------------------------------:|:--------------------------------:|:-----------------------------:|:----------:|:-----------:|
|         [ITSC2021:PFA-Net](https://ieeexplore.ieee.org/abstract/document/9564754)         |         76.90/68.00/39.30        |       85.10/79.90/52.50       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_PFA_net.yaml)     | [model-25M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|           [NIPS2022:RTNH](https://arxiv.org/abs/2206.08171)          |         71.70/62.20/34.40        |       73.70/67.70/41.90       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_RTNH.yaml)     | [model-64M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|          [ECCV2022:V2XViT](https://arxiv.org/abs/2203.10638)         |         71.14/64.28/31.12        |       80.94/73.82/42.73       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_v2xvit.yaml)     | [model-51M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|         [ICRA2022:AttFuse](https://arxiv.org/abs/2109.07644)         |         75.30/66.50/36.10        |       81.80/75.40/48.20       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_attfuse.yaml)     | [model-25M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|        [NIPS2023:Where2comm](https://arxiv.org/abs/2209.12836)       |         69.40/62.10/26.60        |       77.80/72.90/41.50       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_where2comm.yaml)     | [model-30M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|          [ICCV2023:SCOPE](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Spatio-Temporal_Domain_Awareness_for_Multi-Agent_Collaborative_Perception_ICCV_2023_paper.html)          |         61.90/59.30/47.90        |       73.00/71.60/51.60       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_scope.yaml)     | [model-151M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|          [CoRL2023:CoBEVT](https://arxiv.org/abs/2207.02202)         |         78.90/71.90/37.50        |       86.40/81.20/54.70       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_cobevt.yaml)     | [model-40M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|         [ICRA2023:CoAlign](https://arxiv.org/abs/2211.07214)         |         65.80/59.20/34.70        |       76.90/70.20/46.20       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_coalign.yaml)     | [model-43M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|        [WACV2023:AdaFusion](https://openaccess.thecvf.com/content/WACV2023/html/Qiao_Adaptive_Feature_Fusion_for_Cooperative_Perception_Using_LiDAR_Point_Clouds_WACV_2023_paper.html)        |         77.84/72.48/42.85        |       82.20/78.08/55.51       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_adafusion.yaml)     | [model-27M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |
|           [IROS2024:SICP](https://arxiv.org/abs/2312.04822)          |         70.08/60.62/32.43        |       71.45/63.47/33.39       |      [√](opencood/hypes_yaml/V2X-R/4DRadar/V2XR_sicp.yaml)     | [model-28M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/radar) |

### LiDAR-based Cooperative 3D Detector (no-compression)
| **Method** | **Validation (IoU=0.3/0.5/0.7)** | **Testing (IoU=0.3/0.5/0.7)** | **Config** |  **Model**  |
|:------------------------------:|:--------------------------------:|:-----------------------------:|:----------:|:-----------:|
| [ECCV2022:V2XViT](https://arxiv.org/abs/2203.10638)                | 84.99/82.22/64.92                | 90.14/89.01/77.71             |      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_v2xvit.yaml)     | [model-52M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |
| [ICRA2022:AttFuse](https://arxiv.org/abs/2109.07644)               | 86.00/82.20/66.90                | 91.40/89.60/80.20             |      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml)     | [model-25M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |
| [NIPS2023:Where2comm](https://arxiv.org/abs/2209.12836)         | 83.70/80.20/56.80                   | 88.10/86.10/69.20             |      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_where2comm.yaml)     | [model-30M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |
| [ICCV2023:SCOPE](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Spatio-Temporal_Domain_Awareness_for_Multi-Agent_Collaborative_Perception_ICCV_2023_paper.html)                 | 76.00/74.70/60.90                | 81.40/72.90/67.00             |      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_scope.yaml)     | [model-151M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |
| [CoRL2023:CoBEVT](https://arxiv.org/abs/2207.02202)                | 87.64/84.79/71.01                | 92.29/91.44/82.45             |      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_cobevt.yaml)     | [model-40M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |
| [ICRA2023:CoAlign](https://arxiv.org/abs/2211.07214)               | 89.08/87.57/80.05                | 89.59/88.89/83.29             |      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_coalign.yaml)     | [model-43M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |
| [ICCV:AdaFusion](https://openaccess.thecvf.com/content/WACV2023/html/Qiao_Adaptive_Feature_Fusion_for_Cooperative_Perception_Using_LiDAR_Point_Clouds_WACV_2023_paper.html)                 | 88.11/86.91/75.61                | 92.70/90.60/84.80             |      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_adafusion.yaml)     | [model-27M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |
| [IROS2024:SICP](https://arxiv.org/abs/2312.04822)                  | 81.14/77.62/58.14                | 84.64/82.17/66.71             |      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_sicp.yaml)     | [model-28M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |
| [WACV2024:MACP](https://openaccess.thecvf.com/content/WACV2024/html/Ma_MACP_Efficient_Model_Adaptation_for_Cooperative_Perception_WACV_2024_paper.html)                  | 72.80/70.90/60.00                | 83.70/83.10/75.50             |      soon     | [model-61M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/lidar) |


### LiDAR-4D Radar Fusion Cooperative 3D Detector (no-compression)
|       **Method**       | **Validation (IoU=0.3/0.5/0.7)** | **Testing (IoU=0.3/0.5/0.7)** | **Config** |  **Model**  |
|:----------------------:|:--------------------------------:|:-----------------------------:|:----------:|:-----------:|
|  [IROS2023:InterFusion](https://ieeexplore.ieee.org/document/9982123)  |         81.23/77.33/52.93        |       87.91/86.51/69.63       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_InterFusion.yaml)     | [model-95M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
|     [Arxiv2024:L4DR](https://arxiv.org/abs/2408.03677)     |         84.58/82.75/70.29        |       90.78/89.62/82.91       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_L4DR.yaml)     | [model-79M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
|    [ICRA2022:AttFuse](https://arxiv.org/abs/2109.07644)    |         86.14/84.30/70.72        |       92.20/90.70/84.60       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AttFuse.yaml)     | [model-95M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
|     [ECCV2022:V2XViT](https://arxiv.org/abs/2203.10638)    |         85.23/83.90/69.77        |       91.99/91.22/83.04|[√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_V2XViT.yaml)     | [model-118M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
|     [ICCV2023:Scope](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Spatio-Temporal_Domain_Awareness_for_Multi-Agent_Collaborative_Perception_ICCV_2023_paper.html)     |         78.79/77.96/62.57        |       83.38/82.89/70.00       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_Scope.yaml)     | [model-134M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
| [NIPS2023:Where2comm](https://arxiv.org/abs/2209.12836) |         87.62/85.58/69.61        |       92.20/91.00/82.04       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_where2comm.yaml)     | [model-30M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
|     [CoRL2023:CoBEVT](https://arxiv.org/abs/2207.02202)    |         87.72/87.02/78.56        |       94.00/93.21/86.68       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_Cobevt.yaml)     | [model-40M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
|    [ICRA2023:CoAlign](https://arxiv.org/abs/2211.07214)    |         87.16/85.54/73.79        |       92.14/91.21/83.77       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_CoAlign.yaml)     | [model-49M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
|   [WACV2023:AdaFusion](https://openaccess.thecvf.com/content/WACV2023/html/Qiao_Adaptive_Feature_Fusion_for_Cooperative_Perception_Using_LiDAR_Point_Clouds_WACV_2023_paper.html)   |         89.24/87.31/74.86        |       93.10/92.15/85.48       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AdaFusion.yaml)     | [model-27M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |
|      [IROS2024:SICP](https://arxiv.org/abs/2312.04822)     |         82.46/79.44/61.51        |       86.19/84.20/68.15       |      [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_Sicp.yaml)     | [model-28M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r) |

### Multi-modal Diffusion Denoising Results (AttFuse train with fog simulation)
|       **Modality**       | **Snow (IoU=0.3/0.5/0.7)** | **Fog (IoU=0.3/0.5/0.7)** | **Normal (IoU=0.3/0.5/0.7)** | **Config** |  **Model**  |
|:----------------------:|:--------------------------------:|:-----------------------------:|:-----------------------------:|:----------:|:-----------:|
|    LiDAR    |         68.73/65.35/45.54          |      81.71/78.48/61.52       |89.30/87.39/76.09|      [√](opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml)     |[model-25M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r_w_MDD)|
|    LiDAR-4D Radar Fusion w/o MDD    |         79.63/77.37/63.71 | 85.00/80.64/62.89 | 91.77/90.43/82.22     | [√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AttFuse.yaml)     | [model-96M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r_w_MDD)|
|    LiDAR-4D Radar Fusion w MDD    |         83.78/81.19/66.86 | 87.37/83.90/68.64 | 91.61/90.27/82.43 |[√](opencood/hypes_yaml/V2X-R/L_4DR_Fusion_with_MDD/V2XR_AttFuse.yaml)     | [model-96M](http://39.98.109.195:1000/files/V2X-R_Dataset(compressed)/benchmark/l+r_w_MDD) |

## :balloon: Quickly Get Started

### Installation
Refer to [Installation of V2X-R](INSTALL.md)


### Train benchmark model 
First of all, modify the dataset path in the setting file (**root_dir**, **validate_dir**), i.e. `xxx.yaml`.

The setting is same as OpenCOOD, which uses yaml file to configure all the parameters for training. To train your own model from scratch or a continued checkpoint, run the following commonds:
```python
training command --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}] [--tag  ${train_tag}] [--worker  ${number}]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/second_early_fusion.yaml`, meaning you want to train
an early fusion model which utilizes SECOND as the backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
- `tag` (optional) : the path of the checkpoints. The training label is used to record additional information about the model being trained, with the default setting being 'default'.
- `worker` (optional) : the number of workers in dataloader, default is 16.


#### Training with single-GPU
For example, to train V2XR_AttFuse (LiDAR-4D radar fusion version) from scratch:
```
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AttFuse.yaml --tag 'demo' --worker 16
```

To train V2XR_AttFuse from a checkpoint:
```
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AttFuse.yaml --model_dir opencood/logs/V2XR_AttFuse/test__2024_11_21_16_40_38 --tag 'demo' --worker 16
```

#### Training with distributed multi-GPUs
For example, to train V2XR_AttFuse (LiDAR-4D radar fusion version) from scratch with 4 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/V2X-R/L_4DR_Fusion/V2XR_AttFuse.yaml --tag 'demo' --worker 16
```



### Test the model

Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --eval_epoch ${epoch_number} --save_vis ${default False}
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `eval_epoch`: int. Choose to inferece which epoch.
- `save_vis`: bool. Wether to save the visualization result.

For example, to test V2XR_AttFuse (LiDAR-4D radar fusion version) from scratch:
```
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference.py --model_dir opencood/logs/V2XR_AttFuse/test__2024_11_21_16_40_38 --eval_epoch 30  --save_vis 1
```
The evaluation results  will be dumped in the model directory.

### Train model with Multi-modal Diffusion Denoising (MDD) module 
The relevant code section about our MDD module can be found in [here](opencood/models/mdd_modules).

To embed the MDD module in the model, change the following in the yaml file of the original model:
- add use_DeLidar=True
- rewrite core_method to be the version that adds the MDD module, for [example](opencood/models/point_pillar_intermediate_lrf_MDD.py).
- add mdd_block to the model's args and set the parameters in it (see [example_yaml](opencood/hypes_yaml/V2X-R/L_4DR_Fusion_with_MDD/V2XR_AttFuse.yaml) for details)

For example, to train V2XR_AttFuse (LiDAR-4D radar fusion version, 4 GPUs) from scratch with MDD:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/V2X-R/L_4DR_Fusion_with_MDD/V2XR_AttFuse.yaml --tag 'demo' --worker 16
```


## Citation
If you are using our project for your research, please cite the following paper:

```
@article{V2X-R,
  title={V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion},
  author={Huang, Xun and Wang, Jinlong and Xia, Qiming and Chen, Siheng and Yang, Bisheng and Wang, Cheng and Wen, Chenglu},
  journal={arXiv preprint arXiv:2411.08402},
  year={2024}
}
```

## Acknowledgements
Thank for the excellent cooperative perception codebases  [BM2CP](https://github.com/byzhaoAI/BM2CP), [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), [CoPerception](https://github.com/coperception/coperception) and [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm).

Thank for the excellent cooperative perception dataset [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/).

Thank for the dataset and code support by [DerrickXu](https://github.com/DerrickXuNu) and [ZhaoAI](https://github.com/byzhaoAI).
