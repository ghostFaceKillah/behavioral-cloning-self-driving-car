"""
In this file I showcase the image processing pipeline.

1) Show some example pics
"""

import ipdb
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import (
    load_data, batch_generator, random_translation, flip, resize, load_data_point
)


def show_vanilla_examples():
    train, _ = load_data()

    plt.figure(figsize=(12, 10))

    plt_idx = 0
    for row_idx in range(4):
        for side in ['left', 'center', 'right']:
            img, st = load_data_point(train.iloc[row_idx], side)
            plt_idx += 1
            plt.subplot(4, 3, plt_idx)
            plt.imshow(img.astype(np.uint8))
            plt.title('steering angle: {:.2f}'.format(st))
            plt.axis('off')

    plt.savefig('imgs/vanilla.png')


def show_processed_examples():
    train, _ = load_data()

    train = train.iloc[0:1]
    img, st = load_data_point(train.iloc[0], 'center')

    plt.figure(figsize=(12, 10))
    plt.subplot(4, 3, 1)
    plt.imshow(img.astype(np.uint8))
    plt.title('steering angle: {:.2f}'.format(st))
    plt.axis('off')

    train_gen = batch_generator(train, 11, [random_translation, flip, resize])
    xs, ys = next(train_gen)

    for i in range(11):
        img = xs[i]
        st = ys[i]
        plt.subplot(4, 3, i + 2)
        plt.imshow(img.astype(np.uint8))
        plt.title('steering angle: {:.2f}'.format(st))
        plt.axis('off');

    plt.savefig('imgs/processed.png')


if __name__ == '__main__':
    show_vanilla_examples()
    show_processed_examples()
