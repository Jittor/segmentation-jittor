import logging
import numpy as np
import time
# from torch import Tensor

# generation settings
MODEL_NAME = 'deeplab' # pspnet / deeplab / danet / ann / ocnet / ocrnet
SEED = 666
DATA_ROOT = './datasets/voc/'
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
SAVE_MODEL_PATH = './checkpoints/' + MODEL_NAME + '_' + now
WRITER_PATH = './curves/' + MODEL_NAME + '_' + now
# Data settings
SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
CROP_SIZE = 513
IGNORE_LABEL = 255
BATCH_SIZE = 8
IGNORE_INDEX = 255

# Model definition
LEARNING_RATE = 0.001
EPOCHS = 50
MOMENTUM = 0.9
WEIGHT_DECAY = 1E-4
STRIDE = 16
NCLASS = 21
