import numpy as np
import scipy.ndimage
from datetime import datetime
import os
import PIL.Image
import sys
import bz2
from keras.utils import get_file
import dlib
#import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import projector
import pretrained_networks
from training import dataset
from training import misc
import dataset_tool
from my_functions import * # ←Search for Doppelganger必要な関数はここにまとめました


# --------- Style Mixing の実行 -----------
def style_mixing(vec_syn, col_styles, truncation_psi):  
    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    #row_seeds = [1, 2, 3]
    #col_seeds = [4, 5, 6]
    row_seeds = [1]
    col_seeds = [2,3,4,5,6]
    col_styles = col_styles
    truncation_psi = truncation_psi
    minibatch_size = 4

    # Loading networks 
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    # Generating W vectors
    all_seeds = list(row_seeds + col_seeds)   
    #all_w = vec_syn[:6]
    all_w = vec_syn[:6]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...') 
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w[col_styles] + (w_dict[col_seed][col_styles]-w[col_styles])*truncation_psi
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    os.makedirs('my/stylemix_images', exist_ok=True)
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save('./my/stylemix_images/'+str(row_seed)+'-'+str(col_seed)+'.png')

    
    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')

    #r, c = 4, 4  # スクリーン設定（4行×4列）
    r, c = 2, 6
    fig, axs = plt.subplots(r, c, figsize=(10,10), subplot_kw=({'xticks':(),'yticks':()}))

    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx)) 

            # スクリーンに画像配置            
            image_plt = np.array(image_dict[key])
            axs[row_idx, col_idx].imshow(image_plt)
            if row_seed is None:
                x, y = col_seed, col_seed
            elif col_seed is None:
                x, y = row_seed, row_seed
            else:
                x, y = row_seed, col_seed
            axs[row_idx, col_idx].set_xlabel(str(x)+'-'+str(y))

    canvas.save('./my/stylemix_images/grid.png') 

    # スクリーン表示
    #black = np.zeros((1024,1024,3))  # 黒画像作成
    #axs[0,0].imshow(black)
    #axs[0,0].axis('off')
    #plt.show()
    #plt.close()  

def my_style_mixing(vec_syn, col_styles, truncation_psi, my_data_dir, number):  
    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    #row_seeds = [1, 2, 3]
    #col_seeds = [4, 5, 6]
    row_seeds = [1]
    col_seeds = [2,3,4,5,6]
    col_styles = col_styles
    truncation_psi = truncation_psi
    minibatch_size = 4

    # Loading networks 
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    # Generating W vectors
    all_seeds = list(row_seeds + col_seeds)   
    #all_w = vec_syn[:6]
    all_w = vec_syn[:6]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...') 
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w[col_styles] + (w_dict[col_seed][col_styles]-w[col_styles])*truncation_psi
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    os.makedirs(my_data_dir+'/stylemix_images', exist_ok=True)
    for (row_seed, col_seed), image in image_dict.items():
        if row_seed==1 and col_seed==number+2:
            result_path = my_data_dir+'/stylemix_images/'+str(row_seed)+'-'+str(col_seed)+'.png'
            PIL.Image.fromarray(image, 'RGB').save(my_data_dir+'/stylemix_images/'+str(row_seed)+'-'+str(col_seed)+'.png')
    return result_path
    
def predict(number, img_path):
    # 顔画像切り出しモデルの読み込み
    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                LANDMARKS_MODEL_URL, cache_subdir='temp'))
    
    # 顔画像の切り出し
    raw_img_path =  img_path # 後で削除
    print(raw_img_path)
    img_name = raw_img_path.replace('images/','')
    DATAOUT_DIR = img_path.replace('.jpg','') #images/2020xxxx 後で削除(=my)
    ALIGNED_IMAGES_DIR = DATAOUT_DIR +'/pic'
    os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
    TFRECORD_DIR = DATAOUT_DIR + '/dataset'

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
        face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
        aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
        image_align(raw_img_path, aligned_face_path, face_landmarks)

    dataset_tool.create_from_images(tfrecord_dir=TFRECORD_DIR, image_dir=ALIGNED_IMAGES_DIR,shuffle=0)
    
    vec_syn = my_project_real_images(1,data_dir=DATAOUT_DIR)  # ()内はマルチ解像度のデータセットを作成した時の画像枚数

    ##################### ここから笑顔編集（StyleMixing） ####################
    # あらかじめ口元の異なる人たちを7通り用意しておいて、それを読み込む
    vec_hashikan = np.load('sample/vectors/vec_hashikan.npy')
    # くっつけて…
    vec_smile = np.vstack((vec_syn,vec_hashikan))
    # ミックス！
    result_path = my_style_mixing(vec_smile, [4,5], 1.0, my_data_dir=DATAOUT_DIR,number=number)

    return result_path

def main():
    ######### 潜在変数空間からそっくりさんを探す部分 ###########

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

    vec_syn = project_real_images(1)  # ()内はマルチ解像度のデータセットを作成した時の画像枚数
    #display_pic('./my/pic/*.*')  # ターゲット画像の表示
    #display(vec_syn)  # 探索した潜在変数によって生成した画像

    # 探索した潜在変数の保存
    #os.makedirs('my/vector', exist_ok=True)
    #np.save('my/vector/vec_syn', vec_syn)

    ##################### ここから笑顔編集（StyleMixing） ####################

    # 探索した潜在変数の読み込み
    #vec_syn = np.load('my/vector/vec_syn.npy')

    # あらかじめ口元の異なる人たちを7通り用意しておいて、それを読み込む
    vec_hashikan = np.load('sample/vectors/vec_hashikan.npy')

    # くっつけて…
    vec_smile = np.vstack((vec_syn,vec_hashikan))

    # ミックス！
    style_mixing(vec_smile, [4,5], 1.0)

if __name__ == '__main__':
    main()
