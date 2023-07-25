import netCDF4
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from utils.image_transform import crop_center_square, transform_color, resize_image
from patchify import patchify


"""

To run this python file, replace 'ir_dir' with path to .nc file of flight 9, and 'mask_dir' with path to annotated training masks.

ir_dir = 'unet_melt_pond_detection/nc_data/flight9/IRdata_ATWAICE_processed_220718_142920.nc'
mask_dir = 'unet_melt_pond_detection/annotated_data/msks'
test_mask_dir = 'unet_melt_pond_detection/annotated_data/test_msks'

You also need to adjust the paths where to save results

"""

ir_dir = 'E:/polar/code/data/flight9/IRdata_ATWAICE_processed_220718_142920.nc'
mask_dir = 'E:/polar/code/data/ir/entire/original_size/msks'

#### Convert netcdf data to numpy array ####
ds = netCDF4.Dataset(ir_dir)
imgs = ds.variables['Ts'][:]
timestamps = ds.variables['time'][:]

############## Training Data ###############
imgs_train = [imgs[2416],imgs[2380],imgs[2424],imgs[2468],imgs[2476],imgs[2708],imgs[3700],imgs[3884]]

############## Test Data ###################
imgs_test = [imgs[1024], imgs[2452]]

tmp = []

# prepare the training data: crop center square and store and .png file
for im in imgs_train:
    im = crop_center_square(im)
    tmp.append(im)

imgs_train = tmp

# for experimenting with clipped temperature values (did not make a difference in model performance)
for idx, img in enumerate(imgs_train):
    plt.imsave('E:/polar/code/data/ir/entire/original_size/ims_raw_normalize/{}.png'.format(idx), img, cmap='gray', 
               vmin=273, vmax=277)
    
for idx, img in enumerate(imgs_train):
    plt.imsave('E:/polar/code/data/ir/entire/original_size/ims_raw/{}.png'.format(idx), img, cmap='gray')


# prepare masks: transform color values to class values, resize image, crop center square
masks_train = []
for f in os.listdir(mask_dir):
    path = os.path.join(mask_dir, f)
    mask = cv2.imread(path, 0)
    mask = transform_color(mask)
    mask = resize_image(mask)
    mask = crop_center_square(mask)

    masks_train.append(mask)

imgs = np.array(imgs_train)
masks = np.array(masks_train)

# save temperature values as np arrays (not used in final experiment due to bad performance in initial runs)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/480_im.npy', imgs)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/480_ma.npy', masks)


############### Test Data #####################

test_mask_dir = 'E:/polar/code/data/ir/entire/original_size/test_msks/'

tmp = []

for im in imgs_test:
    im = crop_center_square(im)
    tmp.append(im)

imgs_test = tmp
    
for idx, img in enumerate(imgs_test):
    plt.imsave('E:/polar/code/data/ir/entire/original_size/ims_raw_test/{}.png'.format(idx), img, cmap='gray')


# do the same procedure for test images and masks
masks_test = []
for f in os.listdir(test_mask_dir):
    path = os.path.join(test_mask_dir, f)
    mask = cv2.imread(path, 0)
    mask = transform_color(mask)
    mask = resize_image(mask)
    mask = crop_center_square(mask)

    masks_test.append(mask)

imgs = np.array(imgs_test)
masks = np.array(masks_test)

imgs_png = []

for im in os.listdir('E:/polar/code/data/ir/entire/original_size/ims_raw_test/'):
    path = os.path.join('E:/polar/code/data/ir/entire/original_size/ims_raw_test/', im)
    im = cv2.imread(path, 0)
    im = crop_center_square(im)

    imgs_png.append(im)

imgs_raw_test = np.array(imgs_png)
masks_raw_test = np.array(masks_test)

# save as np array
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np_test/480_im.npy', imgs_raw_test)
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np_test/480_ma.npy', masks_raw_test)

imgs_png = []
save_path = 'E:/polar/code/data/ir/entire/original_size/ims_raw/'

imgs_png_norm = []
save_path_norm = 'E:/polar/code/data/ir/entire/original_size/ims_raw_normalize/'

for im in os.listdir(save_path):
    path = os.path.join(save_path, im)
    im = cv2.imread(path, 0)
    im = crop_center_square(im)

    imgs_png.append(im)

for im in os.listdir(save_path_norm):
    path = os.path.join(save_path_norm, im)
    im = cv2.imread(path, 0)
    im = crop_center_square(im)

    imgs_png_norm.append(im)

imgs_raw = np.array(imgs_png)
masks_raw = np.array(masks_train)

imgs_raw_norm = np.array(imgs_png_norm)

# save as numpy arrays
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im.npy', imgs_raw)
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_ma.npy', masks_raw)

# save normalized images as array
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im_norm.npy', imgs_raw_norm)