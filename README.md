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
