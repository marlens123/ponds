import numpy as np
from train import train_wrapper, final_train
import cv2
import os

images = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im.npy')
masks = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_ma.npy')

test_images = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np_test/480_im.npy')
test_masks = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np_test/480_ma.npy')

"""
####################################################################################
############################### Patch Sizes ########################################
####################################################################################
_, stats, hist = train_wrapper(images, masks, im_size=32, base_pref='patch_size_32', train_transfer='imagenet', batch_size=32)
np.save('E:/polar/code/data/stats/statspatch32.npy', stats)
np.save('E:/polar/code/data/stats/histpatch32.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=64, base_pref='patch_size_64', train_transfer='imagenet', batch_size=16)
np.save('E:/polar/code/data/stats/statspatch64.npy', stats)
np.save('E:/polar/code/data/stats/histpatch64.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=128, base_pref='patch_size_128', train_transfer='imagenet', batch_size=8)
np.save('E:/polar/code/data/stats/statspatch128.npy', stats)
np.save('E:/polar/code/data/stats/histpatch128.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=256, base_pref='patch_size_256', train_transfer='imagenet', batch_size=4)
np.save('E:/polar/code/data/stats/statspatch256.npy', stats)
np.save('E:/polar/code/data/stats/histpatch256.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='patch_size_480', train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statspatch480.npy', stats)
np.save('E:/polar/code/data/stats/histpatch480.npy', np.array(hist))

####################################################################################
###################################### Loss ########################################
####################################################################################

_, stats, hist = train_wrapper(images, masks, loss='focal', weight_classes=True, im_size=480, base_pref='loss_focal480', train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/lossfocal480.npy', stats)
np.save('E:/polar/code/data/stats/lossfocal480.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, loss='focal_dice', weight_classes=True, im_size=480, base_pref='loss_focaldice480', train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/lossfocaldice480.npy', stats)
np.save('E:/polar/code/data/stats/lossfocaldice480.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, loss='focal', weight_classes=True, im_size=128, base_pref='loss_focal128', train_transfer='imagenet', batch_size=8)
np.save('E:/polar/code/data/stats/lossfocal128.npy', stats)
np.save('E:/polar/code/data/stats/lossfocal128.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, loss='focal_dice', weight_classes=True, im_size=128, base_pref='loss_focaldice128', train_transfer='imagenet', batch_size=8)
np.save('E:/polar/code/data/stats/lossfocaldice128.npy', stats)
np.save('E:/polar/code/data/stats/lossfocaldice128.npy', np.array(hist))

####################################################################################
############################### Dropout ############################################
####################################################################################

_, stats, hist = train_wrapper(images, masks, loss='focal_dice', weight_classes=True, im_size=480, base_pref='dropout480', use_dropout=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statsdropout480.npy', stats)
np.save('E:/polar/code/data/stats/histdropout480.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, loss='focal_dice', weight_classes=True, im_size=128, base_pref='dropout128', use_dropout=True, train_transfer='imagenet', batch_size=8)
np.save('E:/polar/code/data/stats/statsdropout128.npy', stats)
np.save('E:/polar/code/data/stats/histdropout128.npy', np.array(hist))


####################################################################################
############################### Pretraining ########################################
####################################################################################

# ALSO TRANSFER UNET FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# freeze
_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='transfer_freeze480', loss='focal_dice', weight_classes=True, encoder_freeze=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statstransfer_freeze.npy', stats)
np.save('E:/polar/code/data/stats/histtransfer_freeze.npy', np.array(hist))


# freeze fine-tune
_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='transfer_freeze_tune480', loss='focal_dice', weight_classes=True, freeze_tune=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statstransfer_freeze_tune.npy', stats)
np.save('E:/polar/code/data/stats/histtransfer_freeze_tune.npy', np.array(hist))


# from scratch
_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='transfer_none480', loss='focal_dice', weight_classes=True, train_transfer=None, batch_size=2)
np.save('E:/polar/code/data/stats/statstransfer_none.npy', stats)
np.save('E:/polar/code/data/stats/histtransfer_none.npy', np.array(hist))

# freeze
_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='transfer_freeze480_do', use_dropout=True, loss='focal_dice', weight_classes=True, encoder_freeze=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statstransfer_freeze_do.npy', stats)
np.save('E:/polar/code/data/stats/histtransfer_freeze_do.npy', np.array(hist))

# from scratch
_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='transfer_none480_do', use_dropout=True, loss='focal_dice', weight_classes=True, train_transfer=None, batch_size=2)
np.save('E:/polar/code/data/stats/statstransfer_none_do.npy', stats)
np.save('E:/polar/code/data/stats/histtransfer_none_do.npy', np.array(hist))

####################################################################################
############################### Augmentation #######################################
####################################################################################

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly0', use_dropout=True, augmentation='on_fly', mode=0, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statsaugment_onfly0.npy', stats)
np.save('E:/polar/code/data/stats/stats/histaugment_onfly0.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly1', use_dropout=True, augmentation='on_fly', mode=1, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/stats/statsaugment_onfly1.npy', stats)
np.save('E:/polar/code/data/stats/stats/histaugment_onfly1.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly2', use_dropout=True, augmentation='on_fly', mode=2, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statsaugment_onfly2.npy', stats)
np.save('E:/polar/code/data/stats/histaugment_onfly2.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly3', use_dropout=True, augmentation='on_fly', mode=3, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statsaugment_onfly3.npy', stats)
np.save('E:/polar/code/data/stats/histaugment_onfly3.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly4', use_dropout=True, augmentation='on_fly', mode=4, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statsaugment_onfly4.npy', stats)
np.save('E:/polar/code/data/stats/histaugment_onfly4.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly5', use_dropout=True, augmentation='on_fly', mode=5, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statsaugment_onfly5.npy', stats)
np.save('E:/polar/code/data/stats/histaugment_onfly5.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly6', use_dropout=True, augmentation='on_fly', mode=6, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsaugment_onfly6.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histaugment_onfly6.npy', np.array(hist))


####################################################################################
############################### Augmentation #######################################
####################################################################################

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_offline6', use_dropout=True, augmentation='offline', factor=3, mode=6, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsaugment_offline6.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histaugment_offline6.npy', np.array(hist))


###### Ablation runs ################################################################

_, stats, hist = train_wrapper(images, masks, im_size=256, base_pref='patch_size_256_2', train_transfer='imagenet', batch_size=4)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statspatch256_2.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histpatch256_2.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly1_200', use_dropout=True, epochs=200, augmentation='on_fly', mode=1, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsaugment_onfly1_200.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histaugment_onfly1_200.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_offline6_200', use_dropout=True, epochs=200, augmentation='offline', factor=3, mode=6, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsaugment_offline6_200.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histaugment_offline6_200.npy', np.array(hist))


#####################################################################################
################################### Final Run #######################################
#####################################################################################

time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, early_stop=True, im_size=480, base_pref='final_run', use_dropout=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statsfinal.npy', stats)
np.save('E:/polar/code/data/stats/histfinal.npy', np.array(hist))

time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, im_size=480, base_pref='final_run100', use_dropout=True, train_transfer='imagenet', batch_size=2)
np.save('E:/polar/code/data/stats/statsfinal100.npy', stats)
np.save('E:/polar/code/data/stats/histfinal100.npy', np.array(hist))

time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, im_size=32, base_pref='final_run32_100', use_dropout=True, train_transfer='imagenet', batch_size=32)
np.save('E:/polar/code/data/stats/statsfinal32_100.npy', stats)
np.save('E:/polar/code/data/stats/histfinal32_100.npy', np.array(hist))

time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, epochs=200, im_size=480, base_pref='final_run200', use_dropout=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsfina200.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histfinal200.npy', np.array(hist))

time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, epochs=62, im_size=480, base_pref='final_run62', use_dropout=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsfinal62.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histfinal62.npy', np.array(hist))

time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, im_size=32, base_pref='final_run32_100', use_dropout=True, train_transfer='imagenet', batch_size=32)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsfinal32_100.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histfinal32_100.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly2_2', use_dropout=True, augmentation='on_fly', mode=2, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsaugment_onfly2_2.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histaugment_onfly2_2.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly3_2', use_dropout=True, augmentation='on_fly', mode=3, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsaugment_onfly3_2.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histaugment_onfly3_2.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly4_2', use_dropout=True, augmentation='on_fly', mode=4, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsaugment_onfly4_2.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histaugment_onfly4_2.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='augment_onfly5_2', use_dropout=True, augmentation='on_fly', mode=5, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/statsaugment_onfly5_2.npy', stats)
np.save('C:/Users/Heidi/Documents/marlena/ponds/data/stats/histaugment_onfly5_2.npy', np.array(hist))


# sharpen blur
time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, im_size=480, base_pref='final_runsharpen', augmentation='on_fly', mode=4, use_dropout=True, train_transfer='imagenet', batch_size=2)
"""


#time = final_train(images, masks, test_images, test_masks, epochs=200, loss='focal_dice', weight_classes=True, im_size=480, base_pref='final_runsharpen200', augmentation='on_fly', mode=4, use_dropout=True, train_transfer='imagenet', batch_size=2)

#_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='offline_6_factor2_200', epochs=200, use_dropout=True, augmentation='offline', factor=2, mode=6, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
#np.save('E:/polar/code/data/stats/statsaugment_offline6_f2.npy', stats)
#np.save('E:/polar/code/data/stats/histaugment_offline6_f2.npy', np.array(hist))

#_, stats, hist = train_wrapper(images, masks, im_size=480, base_pref='on_the_fly_4_200', epochs=200, use_dropout=True, augmentation='on_fly', mode=4, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)
#np.save('E:/polar/code/data/stats/statsaugment_onfly4_200t.npy', stats)
#np.save('E:/polar/code/data/stats/histaugment_onfly4_200t.npy', np.array(hist))

_, stats, hist = train_wrapper(images, masks, im_size=32, base_pref='test', epochs=200, use_dropout=True, augmentation='on_fly', mode=4, loss='focal_dice', weight_classes=True, train_transfer='imagenet', batch_size=2)