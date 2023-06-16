import numpy as np
from imagepre import train_wrapper
import cv2
import os

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

### 32
_, time32 = train_wrapper(images32, masks32, im_size=32, backbone='resnet34', train_transfer='imagenet', base_pref='baseline_32', kfold=True)

### 64
_, time64 = train_wrapper(images64, masks64, im_size=64, backbone='resnet34', train_transfer='imagenet', base_pref='baseline_64', kfold=True)

### 128
_, time128 = train_wrapper(images128, masks128, im_size=128, backbone='resnet34', train_transfer='imagenet', base_pref='baseline_128', kfold=True)

### 256 (baseline)
_, time256 = train_wrapper(images256, masks256, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='baseline_256', kfold=True)

### 480: Smaller batch size because less images
_, time480 = train_wrapper(images480, masks480, im_size=480, batch_size=2, train_transfer='imagenet', backbone='resnet34', base_pref='baseline_480', kfold=True)

### augmentation mode 0
_, timemode0 = train_wrapper(images256, masks256, im_size=256, augmentation='on_fly', mode=0, train_transfer='imagenet', backbone='resnet34', base_pref='mode0', kfold=True)

### augmentation mode 1
_, timemode1 = train_wrapper(images256, masks256, im_size=256, augmentation='on_fly', mode=1, train_transfer='imagenet', backbone='resnet34', base_pref='mode1', kfold=True)

### augmentation mode 2
_, timemode2 = train_wrapper(images256, masks256, im_size=256, augmentation='on_fly', mode=2, train_transfer='imagenet', backbone='resnet34', base_pref='mode2', kfold=True)

### augmentation mode 3
_, timemode3 = train_wrapper(images256, masks256, im_size=256, augmentation='on_fly', mode=3, train_transfer='imagenet', backbone='resnet34', base_pref='mode3', kfold=True)

### augmentation mode 4
_, timemode4 = train_wrapper(images256, masks256, im_size=256, augmentation='on_fly', mode=4, train_transfer='imagenet', backbone='resnet34', base_pref='mode4', kfold=True)

### offline augmentation *2
### offline augmentation *10
### offline augmenation *20

### train from scratch
_, timescratch = train_wrapper(images256, masks256, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='scratch', kfold=True)

### pretrain with encoder freeze
_, timefreeze = train_wrapper(images256, masks256, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='freeze', kfold=True, encoder_freeze=True)

### with dropout
_, timedropout = train_wrapper(images256, masks256, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='dropout', kfold=True, use_dropout=True)

### weighted loss function
_, timeweighted = train_wrapper(images256, masks256, im_size=256, train_transfer='imagenet', backbone='resnet34', base_pref='weighted_loss', kfold=True, weight_classes=True)

### focal dice weighted
_, timefocal = train_wrapper(images256, masks256, im_size=256, train_transfer='imagenet', backbone='resnet34', weight_classes=True, loss='focal_dice', base_pref='focal_dice', kfold=True)


### smaller learning rate
### encoder freeze half
### different optimizer


times = [time32, time64, time128, time256, time480, timedropout, timefocal, timefreeze, timemode0, timemode2, timemode3, timemode4,
         timescratch, timeweighted, timemode1]

times = np.array(times)
np.save('E:/polar/code/data/ir/final/times_final.npy', times)


"""
# offline augmentation magnitude 2

# mode 1 offline
X_train, X_test, y_train, y_test, model = train_wrapper(images256, masks256, im_size=256, augmentation='offline', mode=1, factor=2, train_transfer='imagenet', backbone='resnet34', pref='offline2_mode1')
np.save('E:/polar/code/data/ir/prefinal/mode1_offline2_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline2_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline2_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline2_ytest.npy', y_test)

# offline augmentation magnitude 5

# mode 1 offline
X_train, X_test, y_train, y_test, model = train_wrapper(images256, masks256, im_size=256, augmentation='offline', mode=1, factor=5, train_transfer='imagenet', backbone='resnet34', pref='offline5_mode1')
np.save('E:/polar/code/data/ir/prefinal/mode1_offline5_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline5_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline5_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline5_ytest.npy', y_test)

# offline magnitude 20

X_train, X_test, y_train, y_test, model = train_wrapper(images256, masks256, im_size=256, augmentation='offline', mode=1, factor=20, train_transfer='imagenet', backbone='resnet34', pref='offline20_mode1')
np.save('E:/polar/code/data/ir/prefinal/offline20_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/offline20_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/offline20_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/offline20_ytest.npy', y_test)

"""