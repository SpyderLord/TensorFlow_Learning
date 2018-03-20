import tensorflow as tf
import numpy as np
import matplotlib
import os
import logging
from math import ceil
import sys

class FCN8VGG:
    def __init__(self,vgg16_npy_path=None):
        #config the system path
        if vgg16_npy_path is None:
            path=
