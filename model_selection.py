import numpy as np
from train import train_wrapper
import cv2
import os

images = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im.npy')
masks = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_ma.npy')

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

#_, stats, hist = train_wrapper(images, masks, loss='focal', im_size=256, base_pref='patch_size_256', train_transfer='imagenet', batch_size=4)
#_, stats, hist = train_wrapper(images, masks, loss='focal_dice', im_size=256, base_pref='patch_size_256', train_transfer='imagenet', batch_size=4)


####################################################################################
############################### Pretraining ########################################
####################################################################################




####################################################################################
############################### Augmentation #######################################
####################################################################################



####################################################################################
############################### Augmentation #######################################
####################################################################################