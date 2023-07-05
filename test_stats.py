import tensorflow as tf
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
import models.segmentation_models_qubvel as sm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import keras
from utils.augmentation import get_training_augmentation, get_preprocessing, offline_augmentation
from utils.data import Dataloder, Dataset
from sklearn.model_selection import KFold
from utils.patch_extraction import patch_pipeline, patch_extraction

import wandb
from wandb.keras import WandbMetricsLogger

wandb.login()

from timeit import default_timer as timer

# inspiration: https://stackoverflow.com/questions/57181551/can-i-write-a-keras-callback-that-records-and-returns-the-total-training-time
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


def run_train(X_train, y_train, X_test, y_test, model, pref, backbone='resnet34', batch_size=4, weight_classes=False, epochs=100, final_run=False,
              class_weights=None, loss='categoricalCE', optimizer='Adam', augmentation=None, fold_no=0, input_normalize=False):
    
    CLASSES=['melt_pond', 'sea_ice']
    BACKBONE = backbone
    BATCH_SIZE = batch_size

    if weight_classes:
        weights = class_weights
    else:
        weights = None
    
    # Dataset for train images
    train_dataset = Dataset(
        X_train, 
        y_train, 
        classes=CLASSES, 
        normalize=input_normalize,
        augmentation=augmentation,
        preprocessing=get_preprocessing(sm.get_preprocessing(BACKBONE)),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        X_test, 
        y_test, 
        classes=CLASSES,
        normalize=input_normalize, 
        preprocessing=get_preprocessing(sm.get_preprocessing(BACKBONE)),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    if loss == 'jaccard':
        LOSS = sm.losses.JaccardLoss(class_weights=weights)
    elif loss == 'focal_dice':
        dice_loss = sm.losses.DiceLoss(class_weights=weights) 
        focal_loss = sm.losses.CategoricalFocalLoss()
        LOSS = dice_loss + (1 * focal_loss)
    elif loss == 'categoricalCE':
        LOSS = sm.losses.CategoricalCELoss(class_weights=weights)
    else:
        print('No loss function specified')

    if optimizer == 'Adam':
        OPTIMIZER = keras.optimizers.Adam()
    elif optimizer == 'SGD':
        OPTIMIZER = keras.optimizer.SGD()
    elif optimizer == 'Adamax':
        OPTIMIZER = keras.optimizer.Adamax()
    else:
        print('No optimizer specified')

    mean_iou = sm.metrics.IOUScore(name='mean_iou')
    weighted_iou = sm.metrics.IOUScore(class_weights=class_weights, name='weighted_iou')
    f1 = sm.metrics.FScore(beta=1, name='f1')
    precision = sm.metrics.Precision(name='precision')
    recall = sm.metrics.Recall(name='recall')
    melt_pond_iou = sm.metrics.IOUScore(class_indexes=0, name='melt_pond_iou')
    sea_ice_iou = sm.metrics.IOUScore(class_indexes=1, name='sea_ice_iou')
    ocean_iou = sm.metrics.IOUScore(class_indexes=2, name='ocean_iou')
    rounded_iou = sm.metrics.IOUScore(threshold=0.5, name='mean_iou_rounded')


    # threshold value in iou metric will round predictions
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[mean_iou, weighted_iou, f1, precision, recall, melt_pond_iou,
                                                           sea_ice_iou, ocean_iou, rounded_iou])


    if final_run:
        # save weights of best performing model in terms of minimal val_loss
        callbacks = [
            keras.callbacks.ModelCheckpoint('./weights/best_model{}.h5'.format(pref), save_weights_only=True, save_best_only=True, mode='min'),
            # reduces learning rate when metric has stopped improving
            # keras.callbacks.ReduceLROnPlateau(),
            keras.callbacks.EarlyStopping(patience=10),
            TimingCallback(),
            WandbMetricsLogger()
        ]

    else:
        # save weights of best performing model in terms of minimal val_loss
        callbacks = [
            keras.callbacks.ModelCheckpoint('./weights/best_model{}.h5'.format(pref), save_weights_only=True, save_best_only=True, mode='min'),
            # reduces learning rate when metric has stopped improving
            # keras.callbacks.ReduceLROnPlateau(),
            TimingCallback(),
            WandbMetricsLogger()
        ]

    history = model.fit(train_dataloader,
                        verbose=1,
                        callbacks=callbacks,
                        steps_per_epoch=len(train_dataloader), 
                        epochs=epochs,  
                        validation_data=valid_dataloader, 
                        validation_steps=len(valid_dataloader),
                        shuffle=False)

    # save model scores
    with open('./scores/{}_trainHistoryDict'.format(pref), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    time = callbacks[1].logs

    # generalization metrics of trained model
    scores = model.evaluate(valid_dataloader, verbose=0)

    # history generalization metric
    hist_val_iou = history.history['val_mean_iou']
        
    return model, scores, hist_val_iou, time
    



def train_wrapper(X, y, im_size, base_pref, backbone='resnet34', loss='categoricalCE',
              optimizer='Adam', train_transfer=None, encoder_freeze=False, input_normalize=False,
              batch_size=4, augmentation=None, mode=0, factor=2, epochs=50, patch_mode='slide_slide',
              weight_classes=False, use_dropout=False, use_batchnorm=True, rnd_state=0):

    ################################################################
    
    BACKBONE = backbone
    TRAIN_TRANSFER = train_transfer
    AUGMENTATION = augmentation
    BATCH_SIZE = batch_size

    # perform on-fly augmentation also when offline augmentation
    if AUGMENTATION == 'on_fly':    
        on_fly = get_training_augmentation(im_size=im_size, mode=mode)
    else:
        on_fly = None

    #################################################################

    model = sm.Unet(BACKBONE, input_shape=(im_size, im_size, 3), classes=3, activation='softmax', encoder_weights=TRAIN_TRANSFER,
                    decoder_use_dropout=use_dropout, decoder_use_batchnorm=use_batchnorm, encoder_freeze=encoder_freeze)  

    print(model.summary())

    # 4-crossfold validation: https://www.kaggle.com/code/ayuraj/efficientnet-mixup-k-fold-using-tf-and-wandb/notebook
    
    # 4 to make sure that all images are equal split (6 train, 2 test)
    num_folds = 4

    val_loss_per_fold = []
    val_iou_per_fold = []
    val_iou_weighted_per_fold = []
    val_f1_per_fold = []
    val_prec_per_fold = []
    val_rec_per_fold = []
    mp_per_class_per_fold = []
    si_per_class_per_fold = []
    oc_per_class_per_fold = []
    rounded_iou_per_fold = []

    time_per_fold = []             

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=rnd_state)

    # K-fold Cross Validation model evaluation
    fold_no = 1

    # fold statistics
    fold_stats = []

    val_iou_all = []

    for train, test in kfold.split(X, y):

        ##########################################
        ################# Prefix #################
        ##########################################

        # prefix should contain the fold number
        pref = base_pref + "_foldn{}".format(fold_no)

        ########################################## 
        ############## Class Weights #############
        ##########################################

        # compute class weights after split 
        masks_resh = y[train].reshape(-1,1)
        masks_resh_list = masks_resh.flatten().tolist()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)
        print("Class weights are...:", class_weights)
        
        ##########################################
        ############ Patch Extraction ############
        ##########################################

        # 320 random patches per image
        if im_size==32:
            if patch_mode=='random_random':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=320, patch_size=32)
                X_test, y_test = patch_pipeline(X[test], y[test], nr_patches=320, patch_size=32)

            elif patch_mode=='random_slide':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=320, patch_size=32)
                X_test, y_test = patch_extraction(X[test], y[test], size=32, step=32)

            elif patch_mode=='slide_slide':
                X_train, y_train = patch_extraction(X[train], y[train], size=32, step=32)
                X_test, y_test = patch_extraction(X[test], y[test], size=32, step=32)

            else:
                'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
        
        # 80 random patches per image
        elif im_size==64:
            if patch_mode=='random_random':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=80, patch_size=64)
                X_test, y_test = patch_pipeline(X[test], y[test], nr_patches=80, patch_size=64)

            elif patch_mode=='random_slide':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=80, patch_size=64)
                X_test, y_test = patch_extraction(X[test], y[test], size=64, step=68)

            elif patch_mode=='slide_slide':
                X_train, y_train = patch_extraction(X[train], y[train], size=64, step=68)
                X_test, y_test = patch_extraction(X[test], y[test], size=64, step=68)

            else:
                'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
        
        # 20 random patches per image
        elif im_size==128:
            if patch_mode=='random_random': 
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=20, patch_size=128)
                X_test, y_test = patch_pipeline(X[test], y[test], nr_patches=20, patch_size=128)

            elif patch_mode=='random_slide':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=20, patch_size=128)
                X_test, y_test = patch_extraction(X[test], y[test], size=128, step=160)

            elif patch_mode=='slide_slide':
                X_train, y_train = patch_extraction(X[train], y[train], size=128, step=160)
                X_test, y_test = patch_extraction(X[test], y[test], size=128, step=160)

            else:
                'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
        
        # 5 random patches per image
        elif im_size==256:
            if patch_mode=='random_random':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=5, patch_size=256)
                X_test, y_test = patch_pipeline(X[test], y[test], nr_patches=5, patch_size=256)

            elif patch_mode=='random_slide':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=5, patch_size=256)
                X_test, y_test = patch_extraction(X[test], y[test], size=256, step=224)

            elif patch_mode=='slide_slide':
                X_train, y_train = patch_extraction(X[train], y[train], size=256, step=224)
                X_test, y_test = patch_extraction(X[test], y[test], size=256, step=224)

            else:
                'Patch mode must be one of "random_random", "random_slide", "slide_slide"'

        # no patch extraction
        elif im_size==480:
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

        fold_stats.append(y_test)

        print("Train size after patch extraction...", X_train.shape)
        print("Test size after patch extraction...", X_test.shape)

        ##########################################
        ######### Offline Augmentation ###########
        ##########################################

        if AUGMENTATION == 'offline':
            X_train, y_train = offline_augmentation(X_train, y_train, im_size=im_size, mode=mode, factor=factor)

        print("Train size imgs ...", X_train.shape)
        print("Train size masks ...", y_train.shape)
        print("Test size imgs ...", X_test.shape)
        print("Test size masks ...", y_test.shape)

        ##########################################
        ############# Tracking Config ############
        ##########################################


        print("Test set size...", X_test.shape)

        ##########################################
        ################ Training ################
        ##########################################


        # Increase fold number
        fold_no = fold_no + 1

    return fold_stats



def final_train():
    return