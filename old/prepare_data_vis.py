import netCDF4
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from utils.image_transform import crop_center_square, transform_color, resize_image
from patchify import patchify


############## Visualize IR ################
def visualize_ir(img, idx=None, cmap='cividis', colorbar=False, save_path=None):
    plt.imshow(img, cmap=cmap)

    if colorbar:
        plt.colorbar()
    
    if not save_path==None:
        #img = np.clip(img, 273, 276)
        #cv2.imwrite(os.path.join(save_path, '{}.png'.format(idx)), img)
        plt.imsave(os.path.join(save_path, '{}.png'.format(idx)), img, cmap='gray')

############################################

vis_dir = 'E:/polar/code/data/vis/entire/original_size/imgs'
mask_dir = 'E:/polar/code/data/vis/entire/original_size/masks'


############## Training Data ###############

masks_train = []
for f in os.listdir(mask_dir):
    path = os.path.join(mask_dir, f)
    mask = cv2.imread(path, 0)
    mask = transform_color(mask)
    mask = crop_center_square(mask)

    masks_train.append(mask)

masks = np.array(masks_train)

# save 480 array
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/480_ma.npy', masks)

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


imgs_png = []

for im in os.listdir(vis_dir):
    path = os.path.join(vis_dir, im)
    im = cv2.imread(path, 0)
    im = crop_center_square(im)

    imgs_png.append(im)

patches_256_raw = patch_extraction(imgs_png, masks_train, size=256, step=224) # step 224 means that there will be overlap
patches_128_raw = patch_extraction(imgs_png, masks_train, size=128, step=160) # step 120 means that there will be overlap
patches_64_raw = patch_extraction(imgs_png, masks_train, size=64, step=68) # step 68 means that there will be parts of the image left out
patches_32_raw = patch_extraction(imgs_png, masks_train, size=32, step=32) # no overlap

# save raw im patches
np.save('E:/polar/code/data/ir/entire/original_size/vis/256_im.npy',patches_256_raw[0])
np.save('E:/polar/code/data/ir/entire/original_size/vis/256_ma.npy',patches_256_raw[1])
np.save('E:/polar/code/data/ir/entire/original_size/vis/128_im.npy',patches_128_raw[0])
np.save('E:/polar/code/data/ir/entire/original_size/vis/128_ma.npy',patches_128_raw[1])
np.save('E:/polar/code/data/ir/entire/original_size/vis/64_im.npy',patches_64_raw[0])
np.save('E:/polar/code/data/ir/entire/original_size/vis/64_ma.npy',patches_64_raw[1])
np.save('E:/polar/code/data/ir/entire/original_size/vis/32_im.npy',patches_32_raw[0])
np.save('E:/polar/code/data/ir/entire/original_size/vis/32_ma.npy',patches_32_raw[1])