import numpy as np
from patchify import patchify


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


from skimage.util import view_as_windows
from sklearn.feature_extraction import image
import cv2
import os

def extract_patches(img, rnd_state, nr_patches, patch_size):
    """
    Extracts patches from given image using a sliding window.

    Parameters:
    -----------
        img : numpy.nd.array
            image to extract patches from
        patch_nr_total : int
            keeps track of number of already created patches for storage name
        save_folder : str
            folder where resulting patches should get stored
        nr_patches : int
        patch_size : tuple
    """

    patches = image.extract_patches_2d(img, patch_size=patch_size, max_patches=nr_patches, random_state=rnd_state)

    return patches


def patch_pipeline(imgs, masks, patch_size, nr_patches):
    # make sure that random state differs for each image
    # but is the same for each image/mask pair
    
    patch_size = (patch_size, patch_size)
    
    rnd = 0

    img_patches = []
    for img in imgs:
        patches_img = extract_patches(img, rnd_state=rnd, nr_patches=nr_patches, patch_size=patch_size)
        for i in range(patches_img.shape[0]):
            #for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,:,:]
            img_patches.append(single_patch_img)
        rnd += 1
    
    images = np.array(img_patches)

    rnd = 0

    mask_patches = []
    for mask in masks:
        patches_mask = extract_patches(mask, rnd_state=rnd, nr_patches=nr_patches, patch_size=patch_size)
        for i in range(patches_mask.shape[0]):
            #for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i,:,:]
            mask_patches.append(single_patch_mask)
        rnd += 1
    
    masks = np.array(mask_patches)

    return images, masks