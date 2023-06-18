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


def run_train(X_train, y_train, X_test, y_test, model, pref, backbone='inceptionv3', batch_size=4, weight_classes=False,
              class_weights=None, loss='categoricalCE', optimizer='Adam', augmentation=None, kfold=False, fold_no=0):
    
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
        augmentation=augmentation,
        preprocessing=get_preprocessing(sm.get_preprocessing(BACKBONE)),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        X_test, 
        y_test, 
        classes=CLASSES, 
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
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[mean_iou, weighted_iou, tf.keras.metrics.MeanIoU(3, name="iou_keras"), f1, precision, recall, melt_pond_iou,
                                                           sea_ice_iou, ocean_iou, rounded_iou])

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
                        epochs=100,  
                        validation_data=valid_dataloader, 
                        validation_steps=len(valid_dataloader),
                        shuffle=False)

    # save model scores
    with open('./scores/{}_trainHistoryDict'.format(pref), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    time = callbacks[1].logs
    
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

        plt.savefig(os.path.join('E:/polar/code/data/ir/figures/final', '{}_iou_loss'.format(pref)))
    
    if kfold:
        # generalization metrics
        scores = model.evaluate(valid_dataloader, verbose=0)
    
        return model, scores, time
    
    return model, time



def train_wrapper(X, y, im_size, base_pref, backbone='inceptionv3', loss='categoricalCE',
              optimizer='Adam', train_transfer=None, encoder_freeze=False,
              batch_size=4, augmentation=None, mode=0, factor=2,
              weight_classes=False, kfold=False, use_dropout=False, use_batchnorm=True):
    
    masks_resh = y.reshape(-1,1)
    masks_resh_list = masks_resh.flatten().tolist()
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)
    print("Class weights are...:", class_weights)

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
                    decoder_use_dropout=use_dropout, decoder_use_batchnorm=use_batchnorm, encoder_freeze=False)  

    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if AUGMENTATION == 'offline':
        X_train, y_train = offline_augmentation(X_train, y_train, im_size=im_size, mode=mode, factor=factor)

    print("Train size imgs ...", X_train.shape)
    print("Train size masks ...", y_train.shape)
    print("Test size imgs ...", X_test.shape)
    print("Test size masks ...", y_test.shape)

    if not kfold:

        run = wandb.init(project='melt_pond',
                        group='no_kfold',
                        name=base_pref,
                        config={
                            "loss_function": loss,
                            "batch_size": batch_size,
                            "backbone": backbone,
                            "optimizer": optimizer,
                            "train_transfer": train_transfer,
                            "augmentation": AUGMENTATION
                        })
        config = wandb.config

        model, time = run_train(X_train, y_train, X_test, y_test, 
                          backbone=BACKBONE, batch_size=BATCH_SIZE, optimizer=optimizer, loss=loss, class_weights=class_weights,
                          model=model, augmentation=on_fly, pref=base_pref, weight_classes=weight_classes)

    # 5-crossfold augmentation: https://www.kaggle.com/code/ayuraj/efficientnet-mixup-k-fold-using-tf-and-wandb/notebook
    # (inspiration kfold, wanbd)
    else:
        num_folds = 5

        val_loss_per_fold = []
        val_iou_per_fold = []
        val_iou_weighted_per_fold = []
        val_iou_keras_per_fold = []
        val_f1_per_fold = []
        val_prec_per_fold = []
        val_rec_per_fold = []
        mp_per_class_per_fold = []
        si_per_class_per_fold = []
        oc_per_class_per_fold = []
        rounded_iou_per_fold = []

        time_per_fold = []             

        # Merge splitted datasets for cross validation (KFold makes the split)
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

        # K-fold Cross Validation model evaluation
        fold_no = 1

        for train, test in kfold.split(X, y):
            
            # prefix should contain the fold number
            pref = base_pref + "_foldn{}".format(fold_no)

            run = wandb.init(project='melt_pond',
                             group=base_pref,
                             name='foldn_{}'.format(fold_no),
                             config={
                                "loss_function": loss,
                                "batch_size": batch_size,
                                "backbone": backbone,
                                "optimizer": optimizer,
                                "train_transfer": train_transfer,
                                "augmentation": AUGMENTATION
                                }
            )
            config = wandb.config

            print("Test set size...", X[test].shape)

            model, scores, time = run_train(X[train], y[train], X[test], y[test], model=model, augmentation=on_fly, pref=pref, weight_classes=weight_classes,
                                      backbone=BACKBONE, batch_size=BATCH_SIZE, kfold=True, fold_no=fold_no, optimizer=optimizer, loss=loss, class_weights=class_weights)

            val_loss_per_fold.append(scores[0])
            val_iou_per_fold.append(scores[1])
            val_iou_weighted_per_fold.append(scores[2])
            val_iou_keras_per_fold.append(scores[3])
            val_f1_per_fold.append(scores[4])
            val_prec_per_fold.append(scores[5])
            val_rec_per_fold.append(scores[6])
            mp_per_class_per_fold.append(scores[7])
            si_per_class_per_fold.append(scores[8])
            oc_per_class_per_fold.append(scores[9])
            rounded_iou_per_fold.append(scores[10])

            # sum up training time for individual epochs
            time_per_fold.append(sum(time))

            # close run for that fold
            wandb.join()

            # Increase fold number
            fold_no = fold_no + 1

        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(val_iou_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i+1} - Loss: {val_loss_per_fold[i]} - IoU: {val_iou_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> IoU: {np.mean(val_iou_per_fold)} (+- {np.std(val_iou_per_fold)})')
        print(f'> Loss: {np.mean(val_loss_per_fold)}')
        print('------------------------------------------------------------------------')

        val_iou_per_fold = np.array(val_iou_per_fold)
        val_loss_per_fold = np.array(val_loss_per_fold)
        val_iou_weighted_per_fold = np.array(val_iou_weighted_per_fold)
        val_iou_keras_per_fold = np.array(val_iou_keras_per_fold)
        val_f1_per_fold = np.array(val_f1_per_fold)
        val_prec_per_fold = np.array(val_prec_per_fold)
        val_rec_per_fold = np.array(val_rec_per_fold)
        mp_per_class_per_fold = np.array(mp_per_class_per_fold)
        si_per_class_per_fold = np.array(si_per_class_per_fold)
        oc_per_class_per_fold = np.array(oc_per_class_per_fold)
        rounded_iou_per_fold = np.array(rounded_iou_per_fold)

        time_per_fold = np.array(time_per_fold)

        return (val_iou_per_fold, val_loss_per_fold, val_iou_weighted_per_fold, val_iou_keras_per_fold, val_f1_per_fold, val_prec_per_fold, val_rec_per_fold, mp_per_class_per_fold, si_per_class_per_fold, oc_per_class_per_fold, rounded_iou_per_fold), time_per_fold

    return X_train, X_test, y_train, y_test, model

