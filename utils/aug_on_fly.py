import albumentations as A

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(im_size):
    train_transform = [

        # horizontal flip
        # vertical flip
        # random rotation
        # Gaussian noise
        # blurring
        # sharpening
        # random brightness
    	# random contrast
        # pad if needed

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # crop border so original shapes
        A.Rotate(p=0.5, interpolation=0),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)

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


"""
A.HorizontalFlip(p=0.5),
A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
A.PadIfNeeded(min_height=im_size, min_width=im_size, always_apply=True, border_mode=0),
A.RandomCrop(height=im_size, width=im_size, always_apply=True),
A.GaussNoise(p=0.2),
A.IAAPerspective(p=0.5),
A.OneOf(
    [
        A.CLAHE(p=1),
        A.RandomBrightness(p=1),
        A.RandomGamma(p=1),
    ],
    p=0.9,
),

A.OneOf(
    [
        A.Sharpen(p=1),
        A.Blur(blur_limit=3, p=1),
        A.MotionBlur(blur_limit=3, p=1),
    ],
    p=0.9,
),

A.OneOf(
    [
        A.RandomBrightnessContrast(p=1),
        #A.HueSaturationValue(p=1),
    ],
    p=0.9,
),
"""


