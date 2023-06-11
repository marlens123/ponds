import albumentations as A
import numpy as np
import random
import cv2

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(im_size, mode=0):
    """
    Parameters:
    -----------
        mode : int
            defines methods used (for experimental reasons).
            one of 0: flip, crop
                   1: + rotate
                   2: + brightness, contrast
                   3: + sharpen, blur
                   4: + gaussian noise injection
    """
    if mode == 0:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # interpolation 0 means nearest interpolation such that mask labels are preserved
            A.RandomSizedCrop(min_max_height=[int(0.5*im_size), int(0.8*im_size)], height=im_size, width=im_size, interpolation=0, p=0.5),
        ]
        return A.Compose(train_transform)

    elif mode == 1:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomSizedCrop(min_max_height=[int(0.5*im_size), int(0.8*im_size)], height=im_size, width=im_size, interpolation=0, p=0.5),
            A.Rotate(interpolation=0),
        ]
        return A.Compose(train_transform)
    
    elif mode == 2:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomSizedCrop(min_max_height=[int(0.5*im_size), int(0.8*im_size)], height=im_size, width=im_size, interpolation=0, p=0.5),
            A.Rotate(interpolation=0),
            A.RandomBrightnessContrast(brightness_by_max=False),
        ]
        return A.Compose(train_transform)
    
    elif mode == 3:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomSizedCrop(min_max_height=[int(0.5*im_size), int(0.8*im_size)], height=im_size, width=im_size, interpolation=0, p=0.5),
            A.Rotate(interpolation=0),
            #A.RandomBrightnessContrast(brightness_by_max=False),
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.5,
            ),
        ]
        return A.Compose(train_transform)

    elif mode == 4:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomSizedCrop(min_max_height=[int(0.5*im_size), int(0.8*im_size)], height=im_size, width=im_size, interpolation=0, p=0.5),
            A.Rotate(interpolation=0),
            A.GaussNoise(),
        ]
        return A.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def offline_augmentation(trainX, trainy, im_size, mode, factor=2):
  # trainX and trainY are np arrays

  print(factor)

  im_aug_list = []
  ma_aug_list = []

  for i in range(0,factor-1):
    for idx in range(0,trainX.shape[0]):
        img = trainX[idx]
        msk = trainy[idx]
        aug = get_training_augmentation(im_size=im_size, mode=mode)
        sample = aug(image=img, mask=msk)
        im_aug, ma_aug = sample['image'], sample['mask']
        im_aug_list.append(im_aug)
        ma_aug_list.append(ma_aug)
  
  im_aug_np = np.array(im_aug_list)
  ma_aug_np = np.array(ma_aug_list)

  trainX_new = np.concatenate((trainX, im_aug_np), axis=0)
  trainy_new = np.concatenate((trainy, ma_aug_np), axis=0)

  # shuffle again such that augmented and original are mixed
  #random.Random(4).shuffle(trainX_new)
  #random.Random(4).shuffle(trainy_new)

  return trainX_new, trainy_new

