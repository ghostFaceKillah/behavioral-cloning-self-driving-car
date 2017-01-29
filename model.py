"""
Stuff to figure out.

1) Figure out how to build a simple model in Keras. 
   Maybe start from one from comma.ai?

2) Build a model that works on 9 images?

3) Need a joystick to gather the data? Or maybe mouse is ok enough...

3) What about logging keras to tensorboard?

4) Probably we need to think about preloading the images.
   

Let's start real easy and make a starting model on only 10 images.
It worxxx! Time to build some real shit.

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def get_small_data():
    interesting_columns = ['center', 'steering']
    meta_data = pd.read_csv('data/driving_log.csv')[interesting_columns]
    
    choice_idx = [0, 100, 300]
    
    # choose some data with positive steering angle
    meta_plus = meta_data[meta_data.steering > 0.0]
    
    # choose some data with negative steering angle
    meta_minus = meta_data[meta_data.steering < 0.0]
    
    # choose some data with null steering angle
    meta_null = meta_data[meta_data.steering == 0.0]
    
    data = pd.concat([
            meta_plus.iloc[choice_idx],
            meta_minus.iloc[choice_idx],
            meta_null.iloc[choice_idx],
        ])
    
    imgs = np.array([
      plt.imread('data/' + img_fname) for img_fname in data.center
    ])
    
    x = imgs
    y = data.steering.values

    return x, y


def get_big_data():
    interesting_columns = ['center', 'steering']
    data = pd.read_csv('data/driving_log.csv')[interesting_columns]
    
    imgs = np.array([
      plt.imread('data/' + img_fname) for img_fname in data.center
    ])
    
    x = imgs
    y = data.steering.values

    return x, y


def get_model():
    ## Prepare model
    ch, row, col = 3, 160, 320

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                              input_shape=(row, col, ch),
                              output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


start_time = time.time()
x, y = get_big_data()
print("--- It took %s seconds to load the data ---" % (time.time() - start_time))

model = get_model()

history = model.fit(x, y, batch_size=128, nb_epoch=10)
model.save('first.h5')


