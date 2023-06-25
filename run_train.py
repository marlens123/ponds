

# Credit Example

# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier
# Source to original code and license:
#   source url




import numpy as np
from train import train_wrapper
import cv2
import os

times = []

images = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im.npy')
masks = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_ma.npy')


### inceptionv3
_, timeincep = train_wrapper(images, masks, im_size=256, train_transfer='imagenet', backbone='inceptionv3', base_pref='inceptionv3', kfold=True)
times.append(timeincep)


### vgg19
_, timevgg = train_wrapper(images, masks, im_size=256, train_transfer='imagenet', backbone='vgg19', base_pref='vgg19', kfold=True)
times.append(timevgg)


### 32
_, time32 = train_wrapper(images, masks, im_size=32, batch_size=32, backbone='resnet34', train_transfer='imagenet', base_pref='baseline_32', kfold=True)
times.append(time32)


### 64
_, time64 = train_wrapper(images, masks, im_size=64, batch_size=16, backbone='resnet34', train_transfer='imagenet', base_pref='baseline_64', kfold=True)
times.append(time64)


### 128
_, time128 = train_wrapper(images, masks, im_size=128, batch_size=8, backbone='resnet34', train_transfer='imagenet', base_pref='baseline_128', kfold=True)
times.append(time128)


### 256 (baseline)
_, time256 = train_wrapper(images, masks, im_size=256, batch_size=4, train_transfer='imagenet', backbone='resnet34', base_pref='baseline_256', kfold=True)
times.append(time256)


### 480: Smaller batch size because less images
_, time480 = train_wrapper(images, masks, im_size=480, batch_size=2, train_transfer='imagenet', backbone='resnet34', base_pref='baseline_480', kfold=True)
times.append(time480)


### augmentation mode 0
_, timemode0 = train_wrapper(images, masks, im_size=256, augmentation='on_fly', mode=0, train_transfer='imagenet', backbone='resnet34', base_pref='mode0', kfold=True)
times.append(timemode0)


### augmentation mode 1
_, timemode1 = train_wrapper(images, masks, im_size=256, augmentation='on_fly', mode=1, train_transfer='imagenet', backbone='resnet34', base_pref='mode1', kfold=True)
times.append(timemode1)


### augmentation mode 2
_, timemode2 = train_wrapper(images, masks, im_size=256, augmentation='on_fly', mode=2, train_transfer='imagenet', backbone='resnet34', base_pref='mode2', kfold=True)
times.append(timemode2)


### augmentation mode 3
_, timemode3 = train_wrapper(images, masks, im_size=256, augmentation='on_fly', mode=3, train_transfer='imagenet', backbone='resnet34', base_pref='mode3', kfold=True)
times.append(timemode3)


### augmentation mode 4
_, timemode4 = train_wrapper(images, masks, im_size=256, augmentation='on_fly', mode=4, train_transfer='imagenet', backbone='resnet34', base_pref='mode4', kfold=True)
times.append(timemode4)

### offline augmentation *2
### offline augmentation *10
### offline augmenation *20

### train from scratch
_, timescratch = train_wrapper(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='scratch', kfold=True)
times.append(timescratch)


### pretrain with encoder freeze
_, timefreeze = train_wrapper(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='freeze', kfold=True, encoder_freeze=True)
times.append(timefreeze)


### with dropout
_, timedropout = train_wrapper(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='dropout', kfold=True, use_dropout=True)
times.append(timedropout)


### weighted loss function
_, timeweighted = train_wrapper(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='weighted_loss', kfold=True, weight_classes=True)
times.append(timeweighted)


### focal dice loss weighted
_, timefocal = train_wrapper(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', weight_classes=True, loss='focal_dice', base_pref='focal_dice', kfold=True)
times.append(timefocal)


### jaccard loss weighted
_, timejaccard = train_wrapper(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', weight_classes=True, loss='jaccard', base_pref='jaccard', kfold=True)
times.append(timejaccard)


### train with temperature values from scratch
### train with normalization from scratch
### smaller learning rate
### combine encoder freeze and fine-tuning: https://keras.io/guides/transfer_learning/
### different optimizer


times = np.array(times)
np.save('E:/polar/code/data/ir/final/times_final.npy', times)



######################################################################################
### Still to run modes
######################################################################################

images_norm = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im_norm.npy')

# normalized
_, timenorm = train_wrapper(images_norm, masks, im_size=256, batch_size=4, train_transfer='imagenet', backbone='resnet34', base_pref='norm_v_256', kfold=True)

# other normalization from scratch
_, timen = train_wrapper(images, masks, epochs=50, im_size=256, input_normalize=True, train_transfer='imagenet', backbone='resnet34', base_pref='norm_scratch', kfold=True)

# using other patch extraction function
# normalized
_, timepatch = train_wrapper(images, masks, im_size=256, batch_size=4, random_patch=False, train_transfer='imagenet', backbone='resnet34', base_pref='other_patch', kfold=True)