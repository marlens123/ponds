from utils.data_pipeline import data_pipeline
import numpy as np

images = np.load('E:/polar/code/data/ir/entire/original_size/prepared/32_im.npy')
masks = np.load('E:/polar/code/data/ir/entire/original_size/prepared/32_ma.npy')

print(images.shape)
print(masks.shape)

augmentation = True
weight_classes = True

X_train, X_test, Y_train, Y_test, class_weights, im, ma = data_pipeline(images, masks, augmentation, weight_classes)

#np.save('E:/polar/code/data/ir/entire/original_size/prepared/im32.npy', im)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/ma32.npy', ma)

#np.save('E:/polar/code/data/ir/entire/original_size/prepared/imtrain32.npy', X_train)
#np.save('E:/polar/code/data/ir/entire/original_size/prepared/matrain32.npy', Y_train)

