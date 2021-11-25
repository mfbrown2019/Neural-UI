

import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf

class PlotGrpahs(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_accuracy'))
        self.acc.append(logs.get('accuracy'))
        self.i += 1
        
        clear_output(wait=False)
        # plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_accuracy")
        plt.plot(self.x, self.acc, label="accuracy")
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        plt.close()