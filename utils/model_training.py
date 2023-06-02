import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.metrics import MeanIoU
from keras.optimizers import Adam


def train(model, X_train, Y_train, X_test, Y_test, batch_size=8, optimizer=Adam, loss='categorical_crossentropy', class_weights=None, use_batchnorm=False, use_dropout=True, kfold=False, fold_no=None, pretraining=None):

  model.compile(optimizer=optimizer, loss=loss, metrics=[MeanIoU(num_classes=3)])
  
  model.summary()

  if kfold:
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

  if pretraining:
    model.load_weights('???.hdf5')

  history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      verbose=1,
                      epochs=100,
                      validation_data=(X_test, Y_test),
                      class_weight=class_weights,
                      shuffle=False)

  # fill with hyperparameter setting
  model.save('{}_{}_{}_{}_{}.hdf5')

  ##################################################

  # plot loss and iou
  if not kfold:
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join('E:/polar/code/data/ir/figures', '{}_iou_loss'.format(im_size)))

  ##################################################




def get_IoU(model, X_test, Y_test):
  model = model
  model.load_weights('???.hdf5')  
  
  # For IOU
  Y_pred=model.predict(X_test)
  # convert probability output into categorical value
  Y_pred_argmax=np.argmax(Y_pred, axis=3)

  # Per class iou
  IOU_keras = MeanIoU(num_classes=3)  
  IOU_keras.update_state(Y_test[:,:,:,0], Y_pred_argmax)
  print("Mean IoU =", IOU_keras.result().numpy())


  # To calculate I0U for each class (if per class IoU is too low, increase class weight and decrease class weight for one with high prob)
  values = np.array(IOU_keras.get_weights()).reshape(3,3)
  class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
  class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
  class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])

  print("IoU for class1 is: ", class1_IoU)
  print("IoU for class2 is: ", class2_IoU)
  print("IoU for class3 is: ", class3_IoU)



  ##############################################################################################




  import random

def test(model, X_test, Y_test):
  #Predict on a few images
  model = model
  model.load_weights('???.hdf5')

  test_img_number = random.randint(0, len(X_test))
  test_img = X_test[test_img_number]
  ground_truth=Y_test[test_img_number]
  test_img_norm=test_img[:,:,0][:,:,None]
  test_img_input=np.expand_dims(test_img_norm, 0)
  prediction = (model.predict(test_img_input))
  predicted_img=np.argmax(prediction, axis=3)[0,:,:]


  plt.figure(figsize=(12, 8))
  plt.subplot(231)
  plt.title('Testing Image')
  plt.imshow(test_img[:,:,0], cmap='gray')
  plt.subplot(232)
  plt.title('Testing Label')
  plt.imshow(ground_truth[:,:,0], cmap='jet')
  plt.subplot(233)
  plt.title('Prediction on test image')
  plt.imshow(predicted_img, cmap='jet')
  plt.show()



  ####################################################################





def expand_greyscale_channels(image):
    # add channel dimension
    image = np.expand_dims(image, -1)
    # copy last dimension to reach shape of RGB
    # image = image.repeat(3, axis=-1)
    return image

def crop_center_square(image, im_size=(640,480)):
    size=im_size[1]
    # original image dimensions
    height, width = image.shape[:2]
    # calculate new dimensions
    new_width = new_height = size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    # crop
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def transform_color(image):
    """
    Convert class values to greyscale values for visualization
    """
    uniques = np.unique(image)
    
    for idx,elem in enumerate(uniques):
        mask = np.where(image == 1)
        image[mask] = 125
        mask2 = np.where(image == 2)
        image[mask2] = 255
    return image

def preprocess_prediction(image):
    image = expand_greyscale_channels(image)
    image = image.astype(np.float32)
    # will add a dimension that replaces batch_size
    image = np.expand_dims(image, axis=0)
    
    return image

# Predict function
def patch_predict(model, image, patch_size):
    """
    Predicts on image patches and recombines masks to whole image later.
    
    This function is inspired by
    https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py
    
    """

    # initialize mask with zeros
    segm_img = np.zeros(image.shape[:2])
    patch_num=1
    # Iterates through image in steps of patch_size, operates on patches
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            single_patch = image[i:i+patch_size, j:j+patch_size]
            single_patch_shape = single_patch.shape[:2]
            # preprocess for model
            single_patch = preprocess_prediction(single_patch)
            # predict
            pr_mask = model.predict(single_patch)
            # removes batch dimension and channel dimension by replacing the latter with maximum value
            pr_mask_processed = np.argmax(pr_mask.squeeze(), axis=2)
            # make mask values visible
            fin = transform_color(pr_mask_processed)
            # recombine to complete image
            segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(fin, single_patch_shape[::-1])

            #single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            #single_patch_shape = single_patch_norm.shape[:2]
            #single_patch_input = np.expand_dims(single_patch_norm, 0)
            #single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8)
            #segm_img[i:i+single_patch_shape[0], j:j+single_patch_shape[1]] += cv2.resize(single_patch_prediction, single_patch_shape[::-1])
          
            print("Finished processing patch number ", patch_num, " at position ", i,j)
            patch_num+=1

    return segm_img


def predict(model, predict_dir, im_size, save_path):
    """
    Parameters:
    -----------
        predict_dir : str
            path containing images to predict
        im_size : int
            patch size which prediction model is trained on
        save_path : str
            path for image saving
    """
    # different saving path for each patch size
    save_path = os.path.join(save_path, '{}'.format(im_size))

    # load model trained on desired patch size (im_size)
    model = model
    model.load_weights('best_model{}.h5'.format(im_size))

    # IT IS VERY IMPORTANT THAT PREDICTION IS DONE ON IMAGES WITH THE SAME SHAPE AS MODEL IS TRAINED ON.
    # THEREFORE, WE USE THE PATCH_PREDICTION FUNCTION THAT CAN PREDICT SINGLE PATCHES AND LATER CONCATENATE TO 
    # A LARGER IMAGE SIZE.
    
    # predict on images
    for idx, f in enumerate(os.listdir(predict_dir)):
        img = cv2.imread(os.path.join(predict_dir, f), 0)

        if(im_size==480):
            img = crop_center_square(img, 480)
            segmented_image = patch_predict(model, img, im_size)
            # crop to 256 for comparison
            segmented_image = crop_center_square(segmented_image, 256)
        
        # else im_size is a fraction of 256  
        else:
            # crop original image to 256
            img = crop_center_square(img, 256)
            
            # predicts patches then combines if im_size 32 or 64
            segmented_image = patch_predict(model, img, im_size)

        #plt.hist(segmented_image.flatten())
        cv2.imwrite(os.path.join(save_path, '{}.png'.format(idx)), segmented_image)