import tensorflow as tf
from tensorflow.python.client import device_lib
import cv2
import os
import numpy as np
from imutils import paths
from random import randint

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l1_l2

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import InputLayer
# image preprocessor
class SimpleImagePreprocessor:
    def __init__(self, width, height, cWidth = 0, cHeight = 0, cropAugment = 1, interpolation = cv2.INTER_AREA):
        # store target image width, height, and interpolation method for resizing
        self.width = width
        self.height = height
        self.cWidth = cWidth
        self.cHeight = cHeight
        self.interpolation = interpolation
        self.cropAugment = cropAugment
        self.translationAugment = 0
        
    def resize(self, image):
        # resize to a fixed size ignoring aspect ratio
        return [cv2.resize(image, (self.width, self.height), interpolation = self.interpolation)]
    
    # randomly crop an image nAugment times and return each
    def randomCrop(self, image):
        images = []
        
        image = image[0]
        
        # iterate from 0 to nAugment
        for counter in np.arange(0, self.cropAugment):
            # choose a random coordinates for the lower left corner of the image
            lowerLeftX = randint(0, self.width - self.cWidth)
            lowerLeftY = randint(0, self.height - self.cHeight)
            
            # crop the image from the random point to the specified size and append to a list of images
            images.append(image[lowerLeftY:lowerLeftY + self.cHeight, lowerLeftX:lowerLeftX + self.cWidth])
            
        # return the randomly cropped images
        return images
    
    def translate(self, image, pixels = 2):        
        # translate left, right, up, and down
        leftImage = np.roll(image, pixels)
        rightImage = np.roll(image, -pixels)
        upImage = np.roll(image, pixels, axis = 0)
        downImage = np.roll(image, -pixels, axis = 0)
        
        images = [image, leftImage, rightImage, upImage, downImage]
                
        # return images translated in each direction
        return images
        
# image dataset loader
class SimpleImageDatasetLoader:
    def __init__(self, cropAugment = 1, preprocessors = None):
        self.cropAugment = cropAugment
        self.translationAugment = 0
        
        # store the image preprocessor
        self.preprocessors = preprocessors
        
        # if there are no preprocessors, initialize as an empty list
        if self.preprocessors is None:
            self.preprocessors = []
            
        # if preprocessor.translate in self.preprocessors:
        #     self.translationAugment = 4
            
    def load(self, imagePaths, verbose = -1, bw = 0):
        # initialize the list of features and labels
        data = []
        labels = []
        
        
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load an image and extract the class label from the path
            if bw == 1:
                image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(imagePath)
            
            # if there are image preprocessors, apply them to the image
            if self.preprocessors is not None:
                
                # loop over the preprocessors
                for p in self.preprocessors:                    
                    # apply the preprocessor
                    image = p(image)
            
            #print(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            label = (self.cropAugment + self.translationAugment) * [label]
            
            # save the data and labels
            data.extend(image)
            labels.extend(label)
                        
            # give some updates on the preprocessing
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print('[INFO] processed {}/{}'.format(i + 1, len(imagePaths)))
                
        # return the data and labels in numpy arrays
        return (np.array(data), np.array(labels))