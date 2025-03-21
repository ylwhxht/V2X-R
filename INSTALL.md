
# Install
## 1. Clone (or download) the source code 
```
git clone https://github.com/ylwhxht/V2X-R.git
cd V2X-R
```
 
## 2. Create conda environment and set up the base dependencies
```
conda create --name v2xr python=3.7 cmake=3.22.1 setuptools=58.0
conda activate v2xr
conda install cudnn -c conda-forge
conda install boost
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## 3. Install spconv (Support both 1.2.1 and 2.x)

### *(Notice): Make sure *libboost-all-dev* is installed in your linux system before installing *spconv*. If not:
```
sudo apt-get install libboost-all-dev
```

## Install 2.x
```
pip install spconv-cu113
```

## 4. Install pypcd
```
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ..
```

## 5. Install V2XR
```
# install requirements
pip install -r requirements.txt
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace

# FPVRCNN's iou_loss dependency (optional)
python opencood/pcdet_utils/setup.py build_ext --inplace
```

## 6. *(Option) for training and testing SCOPE&How2comm
```
# install basic library of deformable attention
git clone https://github.com/TuSimple/centerformer.git
cd centerformer

# install requirements
pip install -r requirements.txt
sh setup.sh
```

### if there is a problem about cv2:
```
# module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline'
pip install "opencv-python-headless<4.3"
```



