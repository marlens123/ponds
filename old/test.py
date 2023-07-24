import numpy as np
from train import train_wrapper
import cv2
import os


images = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im.npy')
masks = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_ma.npy')

###################################################################################

images32 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/32_im.npy')
masks32 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/32_ma.npy')

images64 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/64_im.npy')
masks64 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/64_ma.npy')

images128 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/128_im.npy')
masks128 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/128_ma.npy')

images256 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/256_im.npy')
masks256 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/256_ma.npy')

images480 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im.npy')
masks480 = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_ma.npy')

#_, time32 = train_wrapper(images32, masks32, epochs=2, im_size=32, backbone='resnet34', train_transfer='imagenet', base_pref='baseline_32', kfold=True)

### 256 (baseline)
#_, time256 = train_wrapper(images, masks, epochs=50, im_size=256, input_normalize=True, train_transfer='imagenet', backbone='resnet34', base_pref='test_norm', kfold=True)
_, time256 = train_wrapper(images, masks, epochs=100, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='test_offline', augmentation='offline', kfold=True)