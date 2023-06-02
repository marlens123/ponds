import numpy as np
import os

patch_sizes = [32,64,128,256,480]

cwd = 'E:/polar/code/data/ir/entire/original_size/prepared'

for p in patch_sizes:
    imgs = np.load(os.path.join(cwd, '{}_imgs.npy'))
    masks = np.load(os.path.join(cwd, '{}_msks.npy'))
    # run main mode baseline + crossfold

    # run main mode pretrain ImageNet
    # run main mode pretrain autoencode



# batch_size
# loss
# weight_classes

# check normalization