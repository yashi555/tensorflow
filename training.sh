#!/bin/sh
pip install Cython
pip install numpy==1.18.5
apt-get update && apt-get install libgl1
pip install opencv-python==4.4.0.46
pip install -r requirements.txt
python train_yolov2_custom.py
