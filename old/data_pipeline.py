from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def data_pipeline(images, masks, augmentation=False, weight_classes=True, normalization='costum'):
  
  n_classes = 3

  print("Shape before expansion ...", images.shape)
  print("Max before normalization ...", np.amax(images))
  print("Min before normalization ...", np.amin(images))

  # preprocessing
  if normalization=='costum':
    orig_shape = images.shape
    images_resh = images.reshape(-1,1)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(images_resh)
    images = normalized.reshape(orig_shape)

  images = np.expand_dims(images, axis=3)

  print("Shape after expansion ...", images.shape)

  if not normalization=='costum':
    images = normalize(images, axis=1) 

  print("Max after normalization ...", np.amax(images))
  print("Min after normalization ...", np.amin(images))
  print("Type images ...", images[0].dtype)

  # masks do not need to be normalized but one-hot encoded
  
  # reshape masks to later compute class weights
  masks_resh = masks.reshape(-1,1)
  # Convert masks_resh to a list
  masks_resh_list = masks_resh.flatten().tolist()

  # should be (nr_imgs x 480 x 480,)
  print("Masks_resh shape ...", masks_resh.shape)
  print("Masks shape ...", masks.shape)
  print("Masks unique ...", np.unique(masks))
  
  masks = np.expand_dims(masks, axis=3)
  masks = to_categorical(masks, num_classes=n_classes)

  print("Shape after onehot ...", masks.shape)
  print("Unique values mask ...", np.unique(masks))
  print("Type masks ...", masks[0].dtype)

  # train test split
  X_train, X_test, Y_train, Y_test = train_test_split(images, masks, test_size=0.2)

  print("Class values in the dataset are ...", np.unique(Y_train))
  print("Image shape ...", X_train.shape)
  print("Masks shape ...", Y_train.shape)
  
  ################################################

  # augmentation
  if augmentation:
    print("Augmentation will be done later ;)")

  ################################################

  # Accord for imbalanced dataset: returns class weights to balance this
  if weight_classes:
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)
    print("Class weights are...:", class_weights)
  else:
    class_weights = None

  ################################################

  return X_train, X_test, Y_train, Y_test, class_weights, images, masks