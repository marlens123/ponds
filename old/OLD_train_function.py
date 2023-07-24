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




def show_batch(image_batch, label_batch):
  plt.figure(figsize=(20,20))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.axis('off')

TRAINING_TFRECORDS = np.array(tf.io.gfile.glob(GCS_PATH + '/train_tfrecords/ld_train*.tfrec'))     
trainloader, validloader = get_dataloader(TRAINING_TFRECORDS[0], TRAINING_TFRECORDS[1])
    

image_batch, label_batch = next(iter(trainloader))
show_batch(image_batch, label_batch)

