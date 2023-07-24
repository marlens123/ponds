import numpy as np
from train import train_wrapper, final_train
import cv2
import os

images = np.load('./data/ds_np/480_im.npy')
masks = np.load('./data/ds_np/480_ma.npy')

test_images = np.load('./data/test_ds_np/480_im.npy')
test_masks = np.load('./data/test_ds_np/480_ma.npy')

# By executing this python file, you will recreate our model training procedure. Note that you need to change the wandb setup
# for experiment tracking in the train.py file.
# stats and hist can be used for further monitoring of experimental results.

####################################################################################
############################### Patch Sizes ########################################
####################################################################################

_, stats, hist = train_wrapper(images, masks, im_size=32, base_pref='patch_size_32', train_transfer='imagenet', batch_size=32)

_, stats, hist = train_wrapper(images, masks, im_size=64, base_pref='patch_size_64', train_transfer='imagenet', batch_size=16)

_, stats, hist = train_wrapper(images, masks, im_size=128, base_pref='patch_size_128', train_transfer='imagenet', batch_size=8)

_, stats, hist = train_wrapper(images, masks, im_size=256, base_pref='patch_size_256', train_transfer='imagenet', batch_size=4)

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='patch_size_480', train_transfer='imagenet', batch_size=2)

####################################################################################
###################################### Loss ########################################
####################################################################################

_, stats, hist = train_wrapper(images, masks, loss='focal_dice', weight_classes=True, im_size=480, base_pref='loss_focaldice480', train_transfer='imagenet', batch_size=2)

####################################################################################
############################### Dropout ############################################
####################################################################################

_, stats, hist = train_wrapper(images, masks, loss='focal_dice', weight_classes=True, im_size=480, base_pref='dropout480', use_dropout=True, train_transfer='imagenet', batch_size=2)


####################################################################################
############################### Pretraining ########################################
####################################################################################

# freeze
_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='transfer_freeze480_do', use_dropout=True, loss='focal_dice', weight_classes=True, encoder_freeze=True, train_transfer='imagenet', batch_size=2)

# from scratch
_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='transfer_none480_do', use_dropout=True, loss='focal_dice', weight_classes=True, train_transfer=None, batch_size=2)

####################################################################################
########################## Augmentation Technique ##################################
####################################################################################

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly0', use_dropout=True, augmentation='on_fly', mode=0, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly1', use_dropout=True, augmentation='on_fly', mode=1, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly2_2', use_dropout=True, augmentation='on_fly', mode=2, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly3_2', use_dropout=True, augmentation='on_fly', mode=3, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly4_2', use_dropout=True, augmentation='on_fly', mode=4, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly5_2', use_dropout=True, augmentation='on_fly', mode=5, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)


####################################################################################
########################### Augmentation Design ####################################
####################################################################################

# mode 4 and 6 are the same (sharpen blur)
_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_on_the_fly_4_200', use_dropout=True, epochs=200, augmentation='on_fly', mode=4, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_offline_6_factor2_200', use_dropout=True, epochs=200, augmentation='offline', factor=3, mode=6, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)


####################################################################################
############################## Ablation run ########################################
####################################################################################

# to check if 256 x 256 still performs bad when reproducing (answer: yes)
_, stats, hist = train_wrapper(images, masks, im_size=256, base_pref='patch_size_256_2', train_transfer='imagenet', batch_size=4)

#####################################################################################
################################### Final Run #######################################
#####################################################################################

time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, epochs=500, im_size=480, base_pref='final_runsharpen500', augmentation='on_fly', mode=4, use_dropout=True, train_transfer='imagenet', batch_size=2)