from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import random

from old.unet import unet_model
from old.data_pipeline import data_pipeline

def main(imgs, msks, crossfold=False, pretrain=None, augmentation=False, batch_size=8, 
         learning_rate=0.0001, optimizer=Adam, loss='categorical_crossentropy', weight_classes=True, 
         use_batchnorm=False, use_dropout=True):
  
  """
  Runs training, testing and prediction on data.

  Parameters:
  -----------
    imgs : np.array
        ds images
    msks : np.array
        corresponding masks
    crossfold : Bool
        use crossfold validation or normal train test split (if False)
    pretrain : str
        if None, train from scratch, else 'Imagenet' or 'ir'
    augemtation : Bool
        whether to use augmentation
    batch_size : int
    learning_rate : float
    optimizer : tensorflow.keras.optimizers
    loss : str
    weight_classes : Bool
        whether to weight classes according to imbalanced training set
    use_batchnorm : Bool
        use Batchnorm layers in UNet
    use_dropout : Bool
        use dropout layers in UNet

  """
  images = imgs
  masks = msks

  IMG_HEIGHT = imgs.shape[1]
  IMG_WIDTH = imgs.shape[2]
  IMG_CHANNELS = imgs.shape[3]
  n_classes = 3

  # modes
  crossfold=crossfold
  pretrain=pretrain

  # Hyperparameters
  augmentation = augmentation
  batch_size = batch_size
  learning_rate = learning_rate
  optimizer = optimizer
  loss = loss
  weight_classes = weight_classes
  use_batchnorm = use_batchnorm
  use_dropout = use_dropout
  
  ################################################

  X_train, X_test, Y_train, Y_test, class_weights = data_pipeline(images, masks, augmentation, weight_classes)

  #Sanity check, view few mages
  image_number = random.randint(0, len(X_train))
  plt.figure(figsize=(12, 6))
  plt.subplot(121)
  plt.imshow(np.reshape(X_train[image_number], (IMG_HEIGHT, IMG_WIDTH)), cmap='gray')
  plt.subplot(122)
  plt.imshow(np.reshape(Y_train[image_number], (IMG_HEIGHT, IMG_WIDTH)), cmap='gray')
  plt.show()

  ################################################

  # model definition
  model = unet_model(n_classes,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)

  num_folds = 10

  iou_per_fold = []
  loss_per_fold = []

  # Merge splitted datasets for cross validation (KFold makes the split)
  X = np.concatenate((X_train, X_test), axis=0)
  Y = np.concatenate((Y_train, Y_test), axis=0)

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=num_folds, shuffle=True)

  # K-fold Cross Validation model evaluation
  fold_no = 1
  for train, test in kfold.split(X, Y):
    train(model, X[train], Y[train], fold_no)

    # Generate generalization metrics
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    iou_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

  # == Provide average scores ==
  print('------------------------------------------------------------------------')
  print('Score per fold')
  for i in range(0, len(iou_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - IoU: {iou_per_fold[i]}%')
  print('------------------------------------------------------------------------')
  print('Average scores for all folds:')
  print(f'> Accuracy: {np.mean(iou_per_fold)} (+- {np.std(iou_per_fold)})')
  print(f'> Loss: {np.mean(loss_per_fold)}')
  print('------------------------------------------------------------------------')