import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tqdm

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle

DATA_DIR = 'data'


# We will data where speed is too low - they lead to non-smooth steering angles
SPEED_CUTOFF = 15.0

STEERING_SMOOTHING = False
STEERING_SMOOTHING_COEFF = 20

# Change angle by how much for left and right camera images
DRIVING_ANGLE_CORR = 0.3
TRANSLATION_CORRECTION = 0.008

VALID_SPLIT = 0.1
LEARNING_RATE = 1e-4

INPUT_IMAGE_SIZE = (160, 320, 3)
IMG_CROP = (50, 140)
IMG_CROP_PCT = (0.3125, 0.875)

NET_IN_ROW, NET_IN_COL, NET_IN_CH = 64, 200, 3

TRANSLATION_RANGE = 50.

# Placeholder for image preloading
IMGS = None

EPOCHS = 10
BATCH_SIZE = 256

def shuffle(data):
    return data.sample(frac=1).reset_index(drop=True)


def preload_imgs():
    """
    Preloads the images into memory. 

    If the dataset is small enough to fit into RAM, training can be made 
    significantly faster by preloading all the images and keeping them
    in memory instead of reloading each image again every time it is needed.
    """
    print("Preloading images...")
    global IMGS
    resu = {}
    dlog = pd.read_csv(os.path.join(DATA_DIR, 'driving_log.csv'))
    all_img_names = list(dlog.center) + list(dlog.left) + list(dlog.right)

    for img_fname in tqdm.tqdm(all_img_names):
        img_fname = img_fname.strip()
        resu[img_fname] = img_to_array(mpimg.imread(img_fname))

    IMGS = resu


def load_data():
    """
    Load the driving log and apply some basic processing:
    * kick out data with speeed lower than SPEED_CUTOFF (they would have 
      unsmooth steering angles)
    * smooth the steering angle in general
    * shuffle the data and split into training and validation set.
    """

    dlog = pd.read_csv(os.path.join(DATA_DIR, 'driving_log.csv'))
    if STEERING_SMOOTHING:
        dlog['steering'] = dlog.steering.ewm(com=STEERING_SMOOTHING_COEFF).mean()
    dlog = shuffle(dlog[dlog.speed > SPEED_CUTOFF])
    train, valid = train_test_split(dlog, test_size=VALID_SPLIT)
    return train, valid



def side_to_correction(side):
    if side == 'left':
        return DRIVING_ANGLE_CORR
    elif side == 'right':
        return -DRIVING_ANGLE_CORR
    elif side == 'center':
        return 0.0
    else:
        raise AssertionError("Incorrect")


def load_data_point(driving_log_entry, side):
    global IMGS
    img_fname = driving_log_entry[side].strip()
    steering_angle_correction = side_to_correction(side)
    steering = driving_log_entry['steering'] + steering_angle_correction

    # img to array to reorder the dims
    if IMGS is None:
        img = img_to_array(mpimg.imread(img_fname))
    else:
        img = IMGS[img_fname]
        
    return img, steering


def resize(img, steering_angle):
    cropped_img = img[IMG_CROP[0]:IMG_CROP[1], :, :]
    resized = cv2.resize(cropped_img, dsize=(NET_IN_COL, NET_IN_ROW), interpolation=cv2.INTER_AREA)
    return resized, steering_angle


def random_translation(img, steering_angle):
    translation_size = np.random.uniform(-TRANSLATION_RANGE, TRANSLATION_RANGE)
    translation_matrix = np.array([[1.0, 0.0, translation_size], [0.0, 1.0, 0.0]])
    translated_img = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    new_steering_angle = steering_angle + TRANSLATION_CORRECTION  * translation_size
    return translated_img, new_steering_angle


def flip(img, steering_angle):
    if np.random.rand() > 0.5:
        return img, steering_angle
    else:
        return cv2.flip(img, 1), -steering_angle


def draw_camera_side():
    i = np.random.randint(3)
    if i == 0:
        return 'left'
    elif i == 1:
        return 'center'
    else:
        return 'right'


def batch_generator(data, batch_size, augs):
    batch_x = np.zeros((batch_size, NET_IN_ROW, NET_IN_COL, NET_IN_CH), dtype=np.float32)
    batch_y = np.zeros(batch_size, dtype=np.float32)

    data = shuffle(data)
    idx_sample = 0

    while True:
        for idx_batch in range(batch_size):

            if idx_sample == len(data):
                data = shuffle(data)
                idx_sample = 0

            log_record = data.iloc[idx_sample]
            side = draw_camera_side()

            # TODO(mike): To make it faster, some prefetching of the images can be done...
            x, y = load_data_point(log_record, side)

            # apply all the augmentations
            for f in augs:
                x, y = f(x, y)

            batch_x[idx_batch] = x
            batch_y[idx_batch] = y

            idx_sample += 1

        yield batch_x, batch_y



def get_model():
    ## Prepare model
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(NET_IN_ROW, NET_IN_COL, NET_IN_CH)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Flatten())

    model.add(Dense(100, init='he_normal'))
    model.add(ELU())

    model.add(Dense(50, init='he_normal'))
    model.add(ELU())

    model.add(Dense(10, init='he_normal'))
    model.add(ELU())

    model.add(Dense(1, init='he_normal'))

    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

    tb = TensorBoard()
    callbacks = [
            tb
    ]

    return model, callbacks



if __name__ == '__main__':
    # preload_imgs()

    train, valid = load_data()

    train_gen = batch_generator(train, BATCH_SIZE, [random_translation, flip, resize])
    valid_gen = batch_generator(valid, BATCH_SIZE, [resize])

    model, callbacks = get_model()

    hist = model.fit_generator(train_gen, 
                               validation_data=valid_gen,
                               samples_per_epoch=110 * BATCH_SIZE,
                               nb_val_samples=1024,
                               nb_epoch=EPOCHS,
                               callbacks=callbacks)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights('model.h5')
       
