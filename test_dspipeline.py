from utils.data_pipeline import data_pipeline
import numpy as np
from imagepre import train_imnet, train_new
import cv2
import os

images = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/256_im.npy')
masks = np.load('E:/polar/code/data/ir/entire/original_size/ims_raw_np/256_ma.npy')

images_aug = np.load('E:/polar/code/data/ir/entire/original_size/prepared/256_im_aug.npy')
masks_aug = np.load('E:/polar/code/data/ir/entire/original_size/prepared/256_ma_aug.npy')

im_dir = 'E:/polar/code/data/ir/256/imgs'
ma_dir = 'E:/polar/code/data/ir/256/masks'

im_list = []
ma_list = []

for f in os.listdir(im_dir):
    im_list.append(os.path.join(im_dir, f))
    ma_list.append(os.path.join(ma_dir, f))

ims = np.array([np.array(cv2.imread(fname, 0)) for fname in im_list])
mas = np.array([np.array(cv2.imread(fname, 0)) for fname in ma_list])


augmentation = True
weight_classes = True

iou, loss = train_new(images, masks, 256, pref='kfold', kfold=True)

iou = np.array(iou)
loss = np.array(loss)

np.save('E:/polar/code/data/ir/entire/original_size/kfold/iou256.npy', iou)
np.save('E:/polar/code/data/ir/entire/original_size/kfold/loss256.npy', loss)

#X_train, X_test, y_train, y_test, class_weights, im, ma = data_pipeline(images, masks, augmentation, weight_classes)
#X_train, X_test, y_train, y_test, model = train_imnet(images, masks, images_aug, masks_aug, 256, pref='256_basebs4methrclw50e')
#X_train, X_test, y_train, y_test, model = train_imnet(images, masks, images_aug, masks_aug, 256, pref='256_augbs4methrclw50eaugonfly', augment=True)


#X_train, X_test, y_train, y_test, model = train_new(images, masks, 256, pref='256_vis')
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/vis256.npy', X_train)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/vis256.npy', X_test)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/vis256.npy', y_train)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/vis256.npy', y_test)

#X_train, X_test, y_train, y_test, model = train_new(ims, mas, 256, pref='256_ims')
#np.save('E:/polar/code/data/ir/entire/original_size/check/imxtrain256.npy', X_train)
#np.save('E:/polar/code/data/ir/entire/original_size/check/imxtest256.npy', X_test)
#np.save('E:/polar/code/data/ir/entire/original_size/check/imytrain256.npy', y_train)
#np.save('E:/polar/code/data/ir/entire/original_size/check/imytest256.npy', y_test)


#np.save('E:/polar/code/data/ir/entire/original_size/prepared/im32.npy', im)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/ma32.npy', ma)

#np.save('E:/polar/code/data/ir/entire/original_size/prepared/imtrain256.npy', X_train)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/matrain256.npy', y_train)





# weiteres Vorgehen: 
# augmentation
# alle patch sizes auf baseline trainieren und mit crossfold validieren (change batch_size for 480)
# beste patch size wählen --> Pretraining Model
# Hyperparameter Optimization: loss, batch_size, class_weight, 
# augment on the fly
# more offline augmentation


