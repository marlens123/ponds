# Detection of melt ponds on Arctic sea ice from infrared images using U-Net

Code used in my bachelor thesis in Cognitive Science, University of Osnabrück.
The objective of the thesis was to develop a segmentation tool that partitions TIR images into sea ice, melt pond and ocean classes.

To make the code work, do the following:

1. Create a new anaconda environment with Python 3.10.0
2. pip install the packages specified in `requirements.txt` (part 1)
3. If you try to run `model_selection.py` (recreation of training pipeline), go to ERROR1 and ERROR2 further below in this file

---------------------------------------

The weights of our final model can be found in `model_weights/best_modelfinal_runsharpen500.h5`. We also included the weights of a patch size 32 x 32 configuration, as we used these for testing smooth patch prediction.

In the file heads you will find relative paths to data that you need for running the code. These paths refer to the myshare folder that comes with the thesis.


Files to recreate our experiments:
---------------------------------
- `model_selection.py`: to recreate the model training (runs `train.py` with different configurations) 
- `train.py`: contains our training pipeline (you may need to adjust the wandb login in this file)

- `data_preparation/extract.ipynb`: to extract TIR images for inspection
- `data_preparation/edge_detection.ipynb`: to create edge maps used for annotation
- `prepare_data.py`: to create prepared np arrays from .nc files
- `qualitative_evaluation.ipynb`: to recreate our qualitative evaluation predictions
- `predict_image.py`: contains the function used for prediction
- `mpf.ipynb`: to recreate our melt pond fraction computations


Additional file contents:
-------------------------
`visualize.ipynb` was used to create visualizations in our thesis
`stats.ipynb` was used to create statistics for our thesis

`utils/` contains 
- `smooth_tiled_predictions.py`: patch stitching function that we integrated in our prediction function. We copied the content of this file from https://github.com/bnsreenu/python_for_microscopists/tree/master/229_smooth_predictions_by_blending_patches. For more information on the reference, see the file head
- `patch_extraction.py`: for patch extraction (used in `train.py`)
- `data.py`: to load data for model training (used in `train.py`), inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb 
- `augmentation.py`: contains augmentation and preprocessing function. Uses albumentation library (https://github.com/albumentations-team/albumentations)
- `image_transform.py`: contains functions for image transformation


`models/segmentation_models_qubvel/`: contains the segmentation models repository (https://github.com/qubvel/segmentation_models). We added the option to train with dropout layers in `segmentation_models/models/unet.py` (marked in file with 'CHANGED')

`vis_segmentation/` contains the OSSP classification algorithm (repo downloaded from https://github.com/wrightni/OSSP)


### <a id="Error1"></a>ERROR1: If you get the error `ImportError: cannot import name 'get_submodules_from_kwargs'`

(see https://github.com/qubvel/segmentation_models/issues/248)

- navigate to your environment directory, and then to `...\Lib\site-packages\classification_models\__init__.py`
- if this is empty, insert the following code:

------------------------------------------------------
```python
import keras_applications as ka
from .__version__ import __version__

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', ka._KERAS_BACKEND)
    layers = kwargs.get('layers', ka._KERAS_LAYERS)
    models = kwargs.get('models', ka._KERAS_MODELS)
    utils = kwargs.get('utils', ka._KERAS_UTILS)
    return backend, layers, models, utils
```
------------------------------------------------------

### <a id="Error2"></a>ERROR2: To be able login to wandb account


- before executing `model_selection.py`or `train.py`, execute the following command: `wand login` and insert the key that was sent via email
- you can also modify the scripts and login to your own wandb account

------------------------------------------------------


To run the OSSP classification algorithm (https://github.com/wrightni/OSSP), do the following:

1. Create a new anaconda environment with Python 3.6
2. conda install the packages specified in `requirements.txt` (part 2)
3. Follow the instruction in `./vis_segmentation/OSSP-wright/readme.md`

(installation may not work properly due to changed package dependencies. I recommend using the classified images, transferred shared in the myshare folder. This is relevant for `mpf.ipynb`)

----------------------------
Additional references:
-----------------------------
- wandb and k-crossfold validation: https://www.kaggle.com/code/ayuraj/efficientnet-mixup-k-fold-using-tf-and-wandb/notebook
- patchify library for patch extraction: https://pypi.org/project/patchify/
- patch prediction: https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py 
- my brother Linus has helped with final markdown formatting of this README file