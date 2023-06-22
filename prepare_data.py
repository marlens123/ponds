import netCDF4
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from utils.image_transform import crop_center_square, transform_color, resize_image
from patchify import patchify

############################################

ir_dir = 'E:/polar/code/data/flight9/IRdata_ATWAICE_processed_220718_142920.nc'
mask_dir = 'E:/polar/code/data/ir/entire/original_size/msks'

#### Convert netcdf data to numpy array ####
ds = netCDF4.Dataset(ir_dir)
imgs = ds.variables['Ts'][:]
timestamps = ds.variables['time'][:]

############## Training Data ###############
imgs_train = [imgs[2416],imgs[2380],imgs[2452],imgs[2468],imgs[2476],imgs[2708],imgs[3700],imgs[3884]]
tmp = []

for im in imgs_train:
    im = crop_center_square(im)
    tmp.append(im)

imgs_train = tmp

for idx, img in enumerate(imgs_train):
    plt.imsave('E:/polar/code/data/ir/entire/original_size/ims_raw_normalize/{}.png'.format(idx), img, cmap='gray', 
               vmin=273, vmax=277)
    
for idx, img in enumerate(imgs_train):
    plt.imsave('E:/polar/code/data/ir/entire/original_size/ims_raw/{}.png'.format(idx), img, cmap='gray')


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

# save 480 array
np.save('E:/polar/code/data/ir/entire/original_size/prepared/480_im.npy', imgs)
np.save('E:/polar/code/data/ir/entire/original_size/prepared/480_ma.npy', masks)

############### Create Patches ################

def patch_extraction(imgs, masks, size, step):

    img_patches = []
    for img in imgs:     
        patches_img = patchify(img, (size, size), step=step)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                img_patches.append(single_patch_img)
    images = np.array(img_patches)

    mask_patches = []
    for img in masks:
        patches_mask = patchify(img, (size, size), step=step)
        
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i,j,:,:]
                mask_patches.append(single_patch_mask)
    masks = np.array(mask_patches)

    return images, masks


patches_256 = patch_extraction(imgs_train, masks_train, size=256, step=224) # step 224 means that there will be overlap
patches_128 = patch_extraction(imgs_train, masks_train, size=128, step=160) # step 120 means that there will be overlap
patches_64 = patch_extraction(imgs_train, masks_train, size=64, step=68) # step 68 means that there will be parts of the image left out
patches_32 = patch_extraction(imgs_train, masks_train, size=32, step=32) # no overlap

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

# save 480 array
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im.npy', imgs_raw)
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_ma.npy', masks_raw)

# save normalized images as array
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/480_im_norm.npy', imgs_raw_norm)

patches_256_raw = patch_extraction(imgs_png, masks_train, size=256, step=224) # step 224 means that there will be overlap
patches_128_raw = patch_extraction(imgs_png, masks_train, size=128, step=160) # step 120 means that there will be overlap
patches_64_raw = patch_extraction(imgs_png, masks_train, size=64, step=68) # step 68 means that there will be parts of the image left out
patches_32_raw = patch_extraction(imgs_png, masks_train, size=32, step=32) # no overlap

# save temperature patches
np.save('E:/polar/code/data/ir/entire/original_size/prepared/256_im.npy',patches_256[0])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/256_ma.npy',patches_256[1])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/128_im.npy',patches_128[0])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/128_ma.npy',patches_128[1])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/64_im.npy',patches_64[0])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/64_ma.npy',patches_64[1])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/32_im.npy',patches_32[0])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/32_ma.npy',patches_32[1])

# save raw im patches
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/256_im.npy',patches_256_raw[0])
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/256_ma.npy',patches_256_raw[1])
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/128_im.npy',patches_128_raw[0])
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/128_ma.npy',patches_128_raw[1])
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/64_im.npy',patches_64_raw[0])
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/64_ma.npy',patches_64_raw[1])
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/32_im.npy',patches_32_raw[0])
np.save('E:/polar/code/data/ir/entire/original_size/ims_raw_np/32_ma.npy',patches_32_raw[1])