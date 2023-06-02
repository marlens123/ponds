import netCDF4
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from utils.image_transform import crop_center_square, transform_color, resize_image
import patchify


############## Visualize IR ################
def visualize_ir(img, colorbar='False', save_path=None):
    plt.imshow(img, cmap='cividis')

    if colorbar:
        plt.colorbar()
    
    if not save_path==None:
        cv2.imwrite(save_path, img)

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
    im.append(tmp)

imgs_train = tmp

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

def patch_extraction(size, step):
    img_patches = []
    for img in imgs_train:     
        patches_img = patchify(img, (size, size), step=step)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                img_patches.append(single_patch_img)
    images = np.array(img_patches)

    mask_patches = []
    for img in masks_train:
        patches_mask = patchify(img, (size, size), step=step)
        
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i,j,:,:]
                mask_patches.append(single_patch_mask)
    masks = np.array(mask_patches)

    return images, masks


patches_256 = patch_extraction(size=256, step=224) # step 224 means that there will be overlap
patches_128 = patch_extraction(size=128, step=160) # step 120 means that there will be overlap
patches_64 = patch_extraction(size=64, step=68) # step 68 means that there will be parts of the image left out
patches_32 = patch_extraction(size=32, step=32) # no overlap

# save patches
np.save('E:/polar/code/data/ir/entire/original_size/prepared/256_im.npy',patches_256[0])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/256_ma.npy',patches_256[1])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/128_im.npy',patches_128[0])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/128_ma.npy',patches_128[1])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/64_im.npy',patches_64[0])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/64_ma.npy',patches_64[1])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/32_im.npy',patches_32[0])
np.save('E:/polar/code/data/ir/entire/original_size/prepared/32_ma.npy',patches_32[1])