import albumentations as A
import numpy as np
import random

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
            A.Crop(x_max=im_size, y_max=im_size, p=0.5),
            # ensures that mask labels are preserved
            #A.Lambda(mask=round_clip_0_1)
        ]
        return A.Compose(train_transform)

    elif mode == 1:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Crop(x_max=im_size, y_max=im_size, p=0.5),
            # nearest interpolation such that mask labels are preserved
            A.Rotate(interpolation=0),
            #A.Lambda(mask=round_clip_0_1)
        ]
        return A.Compose(train_transform)
    
    elif mode == 2:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Crop(x_max=im_size, y_max=im_size, p=0.5),
            A.Rotate(interpolation=0),
            A.RandomBrightnessContrast(),
            #A.Lambda(mask=round_clip_0_1)
        ]
        return A.Compose(train_transform)
    
    elif mode == 3:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Crop(x_max=im_size, y_max=im_size, p=0.5),
            A.Rotate(interpolation=0),
            A.RandomBrightnessContrast(),
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.5,
            ),
            #A.Lambda(mask=round_clip_0_1)
        ]
        return A.Compose(train_transform)

    elif mode == 4:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Crop(x_max=im_size, y_max=im_size, p=0.5),
            A.Rotate(interpolation=0),
            A.RandomBrightnessContrast(),
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.5,
            ),
            A.GaussNoise(),
            #A.Lambda(mask=round_clip_0_1)
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


def offline_augmentation(trainX, trainy, im_size, mode):
  # trainX and trainY are np arrays

  im_aug_list = []
  ma_aug_list = []

  for idx in range(0,trainX.shape[0]):
    print(idx)
    aug = get_training_augmentation(image=trainX[idx], mask=trainy[idx], im_size=im_size, mode=mode)
    im_aug, ma_aug = aug['image'], aug['mask']
    im_aug_list.append(im_aug)
    ma_aug_list.append(ma_aug)
  
  im_aug_np = np.array(im_aug_list)
  ma_aug_np = np.array(ma_aug_list)

  trainX = np.concatenate((trainX, im_aug_np))
  trainy = np.concatenate((trainy, ma_aug_np))

  # shuffle again such that augmented and original are mixed
  random.Random(4).shuffle(trainX)
  random.Random(4).shuffle(trainy)

  return trainX, trainy



