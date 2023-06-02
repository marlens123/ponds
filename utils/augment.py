from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
