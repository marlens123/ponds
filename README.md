# ponds

Changes to make to run code: download .npy files and in train_pipeline, change images, masks and time save path


to get code running, navigate to "...python\Lib\site-packages\classification_models\__init__.py" (should be empty) and insert following code:

import keras_applications as ka
from .__version__ import __version__


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', ka._KERAS_BACKEND)
    layers = kwargs.get('layers', ka._KERAS_LAYERS)
    models = kwargs.get('models', ka._KERAS_MODELS)
    utils = kwargs.get('utils', ka._KERAS_UTILS)
    return backend, layers, models, utils


(see https://github.com/qubvel/segmentation_models/issues/248)


to run VIS segmentation:
- create new environment with requirements specified in .txt
- setup
- run python ossp_process.py 'E:/polar/code/data/vis/segment/' 'srgb' 'E:/polar/code/ponds/ponds/vis_segmentation/OSSP-wright/training_datasets/icebridge_v5_training_data.h5' -v
