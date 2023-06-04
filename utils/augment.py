from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import albumentations as A

def augment(X_train, y_train, X_test, y_test):
  #New generator with rotation and shear where interpolation that comes with rotation and shear are thresholded in masks. 
  #This gives a binary mask rather than a mask with interpolated values. 
  seed=24


  img_data_gen_args = dict(rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

  mask_data_gen_args = dict(rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

  image_data_generator = ImageDataGenerator(**img_data_gen_args)
  image_data_generator.fit(X_train, augment=True, seed=seed)

  image_generator = image_data_generator.flow(X_train, seed=seed)
  valid_img_generator = image_data_generator.flow(X_test, seed=seed)

  mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
  mask_data_generator.fit(y_train, augment=True, seed=seed)
  mask_generator = mask_data_generator.flow(y_train, seed=seed)
  valid_mask_generator = mask_data_generator.flow(y_test, seed=seed)

  def my_image_mask_generator(image_generator, mask_generator):
      train_generator = zip(image_generator, mask_generator)
      for (img, mask) in train_generator:
          yield (img, mask)

  my_generator = my_image_mask_generator(image_generator, mask_generator)
  validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)


  x = image_generator.next()
  y = mask_generator.next()
  for i in range(0,1):
      image = x[i]
      mask = y[i]
      plt.subplot(1,2,1)
      plt.imshow(image[:,:,0], cmap='gray')
      plt.subplot(1,2,2)
      plt.imshow(mask[:,:,0])
      plt.show()


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = A.Compose([

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        )
    ])

    return train_transform
        #A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        #A.RandomCrop(height=(im_size/2), width=(im_size/2), always_apply=True),

        #A.IAAAdditiveGaussianNoise(p=0.2),
        #A.IAAPerspective(p=0.5),

    """
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=0.3),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
    """
        

        
    """
        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.RandomBrightness(p=0.3),
            ],
            p=0.4,
        ),
        A.Lambda(mask=round_clip_0_1)
    """
    #return train_transform


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480)
    ]
    return A.Compose(test_transform)
