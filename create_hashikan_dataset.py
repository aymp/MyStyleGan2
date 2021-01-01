import numpy as np
import scipy.ndimage
import os
import PIL.Image
import sys
import bz2
from keras.utils import get_file
import dlib
import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import projector
import pretrained_networks
from training import dataset
from training import misc
import dataset_tool
from my_functions import * # ←必要な関数はここにまとめました

# make dirctory
os.makedirs('my/pic', exist_ok=True)

# 顔画像切り出しモデルの読み込み
LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                            LANDMARKS_MODEL_URL, cache_subdir='temp'))

# 顔画像の切り出し
RAW_IMAGES_DIR = 'sample/pic'
ALIGNED_IMAGES_DIR = 'my/pic'

landmarks_detector = LandmarksDetector(landmarks_model_path)
for img_name in os.listdir(RAW_IMAGES_DIR):
    raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
        face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
        aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
        image_align(raw_img_path, aligned_face_path, face_landmarks)
        
#display_pic('./my/pic/*.*')

dataset_tool.create_from_images(tfrecord_dir='./my/dataset',image_dir='./my/pic',shuffle=0)

vec_syn = project_real_images(5)  # ()内はマルチ解像度のデータセットを作成した時の画像枚数
#display_pic('./my/pic/*.*')  # ターゲット画像の表示
#display(vec_syn)  # 探索した潜在変数によって生成した画像

# 探索した潜在変数の保存
os.makedirs('sample/vectors', exist_ok=True)
np.save('sample/vectors/vec_hashikan', vec_syn)

# 探索した潜在変数の読み込み
vec_syn = np.load('sample/vectors/vec_hashikan.npy')
print(vec_syn.shape)