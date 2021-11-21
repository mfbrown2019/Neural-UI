   

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import MaxPooling2D
    
class MiniVGGNet:
    def __init__(self):
        self.name = 'MiniVGGNet'
        self.hypers = ['']
    def build(height, width, depth, classes, activation_input = 'relu', dropout_input = [.25]):
        model = Sequential(name = 'MiniVGGNet')
        
        model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (height, width, depth)))
        model.add(Activation(activation_input))
        model.add(BatchNormalization())
        
        model.add(Conv2D(32, (3, 3), padding = 'same'))
        model.add(Activation(activation_input))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(Dropout(dropout_input[0]))
        
        model.add(Conv2D(64, (3, 3), padding = 'same'))
        model.add(Activation(activation_input))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64, (3, 3), padding = 'same'))
        model.add(Activation(activation_input))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        model.add(Dropout(dropout_input[0]))
        
        model.add(Flatten())
        
        model.add(Dense(512))
        model.add(Activation(activation_input))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_input[0]))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model