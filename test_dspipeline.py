from utils.data_pipeline import data_pipeline
import numpy as np
from imagepre import train_imnet, train_new
import cv2
import os

images = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/256_im.npy')
masks = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/256_ma.npy')

images_aug = np.load('E:/polar/code/data/ir/entire/original_size/prepared/256_im_aug.npy')
masks_aug = np.load('E:/polar/code/data/ir/entire/original_size/prepared/256_ma_aug.npy')

im_dir = 'E:/polar/code/data/ir/256/imgs'
ma_dir = 'E:/polar/code/data/ir/256/masks'

im_list = []
ma_list = []

for f in os.listdir(im_dir):
    im_list.append(os.path.join(im_dir, f))
    ma_list.append(os.path.join(ma_dir, f))

ims = np.array([np.array(cv2.imread(fname, 0)) for fname in im_list])
mas = np.array([np.array(cv2.imread(fname, 0)) for fname in ma_list])

augmentation = True
weight_classes = True

#iou, loss = train_new(images, masks, 256, pref='kfold', kfold=True)

#iou = np.array(iou)
#loss = np.array(loss)

#np.save('E:/polar/code/data/ir/entire/original_size/kfold/iou256.npy', iou)
#np.save('E:/polar/code/data/ir/entire/original_size/kfold/loss256.npy', loss)

#X_train, X_test, y_train, y_test, class_weights, im, ma = data_pipeline(images, masks, augmentation, weight_classes)
#X_train, X_test, y_train, y_test, model = train_imnet(images, masks, images_aug, masks_aug, 256, pref='256_basebs4methrclw50e')
#X_train, X_test, y_train, y_test, model = train_imnet(images, masks, images_aug, masks_aug, 256, pref='256_augbs4methrclw50eaugonfly', augment=True)

"""
X_train, X_test, y_train, y_test, model = train_new(images, masks, im_size=256, train_transfer='imagenet', pref='prefinal_resnet')
np.save('E:/polar/code/data/ir/prefinal/resnet_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/resnet_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/resnet_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/resnet_ytest.npy', y_test)

X_train, X_test, y_train, y_test, model = train_new(images, masks, im_size=256, train_transfer='imagenet', backbone='vgg19', pref='prefinal_vgg')
np.save('E:/polar/code/data/ir/prefinal/vgg_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/vgg_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/vgg_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/vgg_ytest.npy', y_test)

X_train, X_test, y_train, y_test, model = train_new(images, masks, im_size=256, train_transfer='imagenet', backbone='inceptionv3', pref='prefinal_inception')
np.save('E:/polar/code/data/ir/prefinal/incep_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/incep_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/incep_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/incep_ytest.npy', y_test)
"""

#X_train, X_test, y_train, y_test, model = train_new(ims, mas, 256, pref='256_ims')
#np.save('E:/polar/code/data/ir/entire/original_size/check/imxtrain256.npy', X_train)
#np.save('E:/polar/code/data/ir/entire/original_size/check/imxtest256.npy', X_test)
#np.save('E:/polar/code/data/ir/entire/original_size/check/imytrain256.npy', y_train)
#np.save('E:/polar/code/data/ir/entire/original_size/check/imytest256.npy', y_test)


#np.save('E:/polar/code/data/ir/entire/original_size/prepared/im32.npy', im)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/ma32.npy', ma)

#np.save('E:/polar/code/data/ir/entire/original_size/prepared/imtrain256.npy', X_train)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/matrain256.npy', y_train)



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


#####################################################################################
############################### Baseline Training ###################################
#####################################################################################

"""
# patch size 32
X_train, X_test, y_train, y_test, model = train_new(images32, masks32, im_size=32, backbone='resnet34', train_transfer='imagenet', pref='baseline_32')
np.save('E:/polar/code/data/ir/final/baseline/32_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/final/baseline/32_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/final/baseline/32_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/final/baseline/32_ytest.npy', y_test)


# patch size 64
X_train, X_test, y_train, y_test, model = train_new(images64, masks64, im_size=64, backbone='resnet34', train_transfer='imagenet', pref='test')
np.save('E:/polar/code/data/ir/final/baseline/64_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/final/baseline/64_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/final/baseline/64_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/final/baseline/64_ytest.npy', y_test)

# patch size 128
X_train, X_test, y_train, y_test, model = train_new(images128, masks128, im_size=128, backbone='resnet34', train_transfer='imagenet', pref='test2')
np.save('E:/polar/code/data/ir/final/baseline/128_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/final/baseline/128_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/final/baseline/128_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/final/baseline/128_ytest.npy', y_test)

# patch size 480: smaller batch size because train set is smaller
X_train, X_test, y_train, y_test, model = train_new(images480, masks480, im_size=480, backbone='resnet34', batch_size=2, train_transfer='imagenet', pref='baseline_480')
np.save('E:/polar/code/data/ir/final/baseline/480_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/final/baseline/480_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/final/baseline/480_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/final/baseline/480_ytest.npy', y_test)

# mode 2 on fly
X_train, X_test, y_train, y_test, model = train_new(images256, masks256, im_size=256, augmentation='on_fly', mode=2, train_transfer='imagenet', backbone='resnet34', pref='augment_mode2')
np.save('E:/polar/code/data/ir/prefinal/mode2_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/mode2_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/mode2_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/mode2_ytest.npy', y_test)

# mode 3 on fly
X_train, X_test, y_train, y_test, model = train_new(images256, masks256, im_size=256, augmentation='on_fly', mode=3, train_transfer='imagenet', backbone='resnet34', pref='augment_mode3')
np.save('E:/polar/code/data/ir/prefinal/mode3_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/mode3_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/mode3_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/mode3_ytest.npy', y_test)

# mode 4 on fly
X_train, X_test, y_train, y_test, model = train_new(images256, masks256, im_size=256, augmentation='on_fly', mode=4, train_transfer='imagenet', backbone='resnet34', pref='augment_mode4')
np.save('E:/polar/code/data/ir/prefinal/mode4_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/mode4_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/mode4_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/mode4_ytest.npy', y_test)

# offline augmentation magnitude 2

# mode 1 offline
X_train, X_test, y_train, y_test, model = train_new(images256, masks256, im_size=256, augmentation='offline', mode=1, factor=2, train_transfer='imagenet', backbone='resnet34', pref='offline2_mode1')
np.save('E:/polar/code/data/ir/prefinal/mode1_offline2_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline2_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline2_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline2_ytest.npy', y_test)

# offline augmentation magnitude 5

# mode 1 offline
X_train, X_test, y_train, y_test, model = train_new(images256, masks256, im_size=256, augmentation='offline', mode=1, factor=5, train_transfer='imagenet', backbone='resnet34', pref='offline5_mode1')
np.save('E:/polar/code/data/ir/prefinal/mode1_offline5_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline5_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline5_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/mode1_offline5_ytest.npy', y_test)

########################################################################################

X_train, X_test, y_train, y_test, model = train_new(images256, masks256, im_size=256, train_transfer=None, backbone='resnet34', pref='scratch')
np.save('E:/polar/code/data/ir/prefinal/scratch_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/scratch_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/scratch_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/scratch_ytest.npy', y_test)


# add best augmentation
X_train, X_test, y_train, y_test, model = train_new(images256, masks256, im_size=256, train_transfer='imagenet', backbone='resnet34', weight_classes=True, loss='focal_dice', pref='class_imbalance')
np.save('E:/polar/code/data/ir/prefinal/classwei_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/classwei_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/classwei_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/classwei_ytest.npy', y_test)


X_train, X_test, y_train, y_test, model = train_new(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', pref='baseline2')
np.save('E:/polar/code/data/ir/prefinal/base2_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/base2_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/base2_ytrain.npy', y_train)


X_train, X_test, y_train, y_test, model = train_new(images256, masks256, im_size=256, augmentation='offline', mode=1, factor=20, train_transfer='imagenet', backbone='resnet34', pref='offline20_mode1')
np.save('E:/polar/code/data/ir/prefinal/offline20_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/offline20_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/offline20_ytrain.npy', y_train)
np.save('E:/polar/code/data/ir/prefinal/offline20_ytest.npy', y_test)

iou, loss = train_new(images256, masks256, im_size=256, train_transfer='imagenet', backbone='resnet34', pref='kfold_baseline', kfold=True)
np.save('E:/polar/code/data/ir/prefinal/kfold_base_iou.npy', iou)
np.save('E:/polar/code/data/ir/prefinal/kfold_base_loss.npy', loss)


## baseline with dropout
X_train, X_test, y_train, y_test, model = train_new(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', pref='dropout', use_dropout=True)
np.save('E:/polar/code/data/ir/prefinal/dropout_xtrain.npy', X_train)
np.save('E:/polar/code/data/ir/prefinal/dropout_xtest.npy', X_test)
np.save('E:/polar/code/data/ir/prefinal/dropout_ytrain.npy', y_train)
"""

# encoder freeze
X_train, X_test, y_train, y_test, model = train_new(images, masks, im_size=256, train_transfer='imagenet', backbone='resnet34', pref='freeze', encoder_freeze=True)