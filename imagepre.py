import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import keras
from utils.augmentation import get_training_augmentation, get_preprocessing, offline_augmentation
from utils.data import Dataloder, Dataset
from sklearn.model_selection import KFold

def train_imnet(images, masks, imaug, maaug, size, pref, backbone='resnet34', loss='categorical_crossentropy', augment=False, weight_classes=True, batch_size=4, kfold=False):

    ######## Reshape Input ############
    print(images[0].dtype)
    print("Original im shape ...", images.shape)
    
    # sm.unet expects 3 channel input
    images = np.stack((images,)*3, axis=-1)
    imaug = np.stack((imaug,)*3, axis=-1)

    print("New im shape ...", images.shape)
    print("Original masks shape ...", masks.shape)

    masks_resh = masks.reshape(-1,1)
    # Convert masks_resh to a list
    masks_resh_list = masks_resh.flatten().tolist()

    masks = np.expand_dims(masks, -1)
    masks = to_categorical(masks, num_classes=3)
    maaug = np.expand_dims(maaug, -1)
    maaug = to_categorical(maaug, num_classes=3)

    print("New masks shape ...", masks.shape)
    print("Pixel values in the mask are: ", np.unique(masks))

    ######## Preprocessing ###########

    BACKBONE=backbone
    preprocess_input = sm.get_preprocessing(backbone)
    images = preprocess_input(images)
    imaug = preprocess_input(imaug)

    print("Im shape after preprocessing ...", images.shape)

    ######### Train Test Split ########

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    if augment:
        X_train = [j for i in [X_train, imaug] for j in i]
        y_train = [j for i in [y_train, maaug] for j in i]

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Generate shuffled indices
        indices = np.random.permutation(len(X_train))

        # Shuffle both arrays using the indices
        X_train = X_train[indices]
        y_train = y_train[indices]

    print("train shape ...", X_train.shape)
    print("train shape y...", y_train.shape)
    print("test shape ...", X_test.shape)
    print("test shape y...", y_test.shape)

    print(X_train[0].dtype)
    print(X_test[0].dtype)

    ######### Augmentation ###########

    if weight_classes:
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)
        print("Class weights are...:", class_weights)
    else:
        class_weights = None

    ######### Model Training #########

    dice_loss = sm.losses.DiceLoss(class_weights=class_weights) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    LOSS = dice_loss + (1 * focal_loss)


    model = sm.Unet(BACKBONE, input_shape=(size, size, 3), classes=3, activation='softmax', encoder_weights='imagenet')
    model.compile('Adam', loss=LOSS, metrics=[sm.metrics.IOUScore(threshold=0.5)])
    print(model.summary())

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model{}.h5'.format(size), save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    #Fit the model
    #history = model.fit(my_generator, validation_data=validation_datagen, steps_per_epoch=len(X_train) // 16, validation_steps=len(X_train) // 16, epochs=100)
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=callbacks,
                        epochs=40,
                        validation_data=(X_test, y_test),
                        #class_weight=class_weights,
                        shuffle=False)

    # fill with hyperparameter setting
    model.save('{}_baseline.hdf5'.format(pref))

    with open('{}_trainHistoryDict'.format(pref), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

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

        plt.savefig(os.path.join('E:/polar/code/data/ir/figures', '{}_iou_loss'.format(pref)))

    return X_train, X_test, y_train, y_test, model


################################################################################################
################################################################################################
################################################################################################


def run_train(X_train, y_train, X_test, y_test,
              model, pref,
              backbone='inceptionv3', batch_size=4, 
              augmentation=None, kfold=False):
    
    CLASSES=['melt_pond', 'sea_ice']
    BACKBONE = backbone
    BATCH_SIZE = batch_size
    
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

    # save weights of best performing model in terms of val_loss
    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model{}.h5'.format(pref), save_weights_only=True, save_best_only=True, mode='min'),
        # reduces learning rate when metric has stopped improving
        keras.callbacks.ReduceLROnPlateau(),
    ]

    history = model.fit(train_dataloader,
                        verbose=1,
                        callbacks=callbacks,
                        steps_per_epoch=len(train_dataloader), 
                        epochs=40,  
                        validation_data=valid_dataloader, 
                        validation_steps=len(valid_dataloader),
                        shuffle=False)

    if not kfold:
        #model.save('{}_baseline.hdf5'.format(pref))

        # save model scores
        with open('{}_trainHistoryDict'.format(pref), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    
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
        # Generate generalization metrics
        scores = model.evaluate(valid_dataloader, verbose=0)
    
        return model, scores
    
    return model



def train_new(X, y, im_size, pref, mode=0, backbone='inceptionv3', loss='categoricalCE',
              optimizer='Adam', train_transfer=None,
              batch_size=4, augmentation=None, weight_classes=False, kfold=False):
    
    if weight_classes:
        masks_resh = y.reshape(-1,1)
        masks_resh_list = masks_resh.flatten().tolist()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)
        print("Class weights are...:", class_weights)
    else:
        class_weights = None

    ################################################################
    
    BACKBONE = backbone
    LOSS = loss
    OPTIMIZER = optimizer
    TRAIN_TRANSFER = train_transfer
    AUGMENTATION = augmentation
    BATCH_SIZE = batch_size

    # perform on-fly augmentation also when offline augmentation
    if AUGMENTATION == 'on_fly':    
        on_fly = get_training_augmentation(im_size=im_size, mode=mode)
    else:
        on_fly = None

    if LOSS == 'jaccard':
        LOSS = sm.losses.JaccardLoss(class_weights=class_weights)
    elif LOSS == 'focal_dice':
        dice_loss = sm.losses.DiceLoss(class_weights=class_weights) 
        focal_loss = sm.losses.CategoricalFocalLoss()
        LOSS = dice_loss + (1 * focal_loss)
    else:
        LOSS = sm.losses.CategoricalCELoss(class_weights=class_weights)

    if OPTIMIZER == 'Adam':
        OPTIMIZER = keras.optimizers.Adam()
    else:
        print('No optimizer specified')

    #################################################################

    model = sm.Unet(BACKBONE, input_shape=(im_size, im_size, 3), classes=3, activation='softmax', encoder_weights=TRAIN_TRANSFER)
    
    # threshold value in iou metric will round predictions
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[sm.metrics.IOUScore(threshold=0.5)])
    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Train size imgs ...", X_train.shape)
    print("Train size masks ...", y_train.shape)
    print("Test size imgs ...", X_test.shape)
    print("Test size masks ...", y_test.shape)

    if AUGMENTATION == 'offline':
        X_train, y_train = offline_augmentation(X_train, y_train, im_size=im_size, mode=mode)

    if not kfold:
        model = run_train(X_train, y_train, X_test, y_test, 
                          backbone=BACKBONE, batch_size=BATCH_SIZE,
                          model=model, augmentation=on_fly, pref=pref)

    # 10-crossfold augmentation
    else:
        num_folds = 10

        iou_per_fold = []
        loss_per_fold = []

        # Merge splitted datasets for cross validation (KFold makes the split)
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=True)

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train, test in kfold.split(X, y):
            model, scores = run_train(X[train], y[train], X[test], y[test], model=model, augmentation=on_fly, pref=pref, 
                                      backbone=BACKBONE, batch_size=BATCH_SIZE, kfold=True)

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
        print(f'> IoU: {np.mean(iou_per_fold)} (+- {np.std(iou_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

        return iou_per_fold, loss_per_fold

    return X_train, X_test, y_train, y_test, model

