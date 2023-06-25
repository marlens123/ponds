import numpy as np
from train import train_wrapper
import cv2
import os

times = []

images = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im.npy')
masks = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_ma.npy')

images_norm = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im_norm.npy')

# normalized
_, timenorm = train_wrapper(images_norm, masks, im_size=256, batch_size=4, train_transfer='imagenet', backbone='resnet34', base_pref='norm_v_256', kfold=True)

# other normalization from scratch
_, timen = train_wrapper(images, masks, epochs=50, im_size=256, input_normalize=True, train_transfer='imagenet', backbone='resnet34', base_pref='norm_scratch', kfold=True)

# using other patch extraction function
# normalized
_, timepatch = train_wrapper(images, masks, im_size=256, batch_size=4, patch_mode='slide_slide', train_transfer='imagenet', backbone='resnet34', base_pref='other_patch', kfold=True)




######### TO-DO ##########


# randomized patch extraction + sliding window val
_, _ = train_wrapper(images, masks, im_size=256, batch_size=4, patch_mode='random_slide', train_transfer='imagenet', backbone='resnet34', base_pref='random_slide', kfold=True)
