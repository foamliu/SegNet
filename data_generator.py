import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
from PIL import Image
from keras.utils import Sequence

from config import batch_size
from config import img_cols
from config import img_rows

train_color = 'train_color'
train_label = 'train_label'

num_labels = 8
decode_dict = {0: 'others', 33: 'car', 34: 'motorbicycle', 35: 'bicycle', 36: 'person', 38: 'truck', 39: 'bus',
               40: 'tricycle'}
encode_dict = {'others': 0, 'car': 1, 'motorbicycle': 2, 'bicycle': 3, 'person': 4, 'truck': 5, 'bus': 6, 'tricycle': 7}


def get_id(old_id):
    return encode_dict[decode_dict[old_id]]


def get_label(name, path=''):
    label_name = name.split('.')[0] + '_instanceIds.png'
    path = os.path.join(path, train_label)
    filename = os.path.join(path, label_name)
    label = np.asarray(Image.open(filename)) // 1000
    label[
        (label != 0) & (label != 33) & (label != 34) & (label != 35) & (label != 36) & (label != 38) & (label != 39) & (
                label != 40)] = 0
    return label


def get_y(label):
    y = np.zeros(shape=(320, 320), dtype=np.int32)
    y_indices, x_indices = np.where(label != 0)
    for i in range(len(x_indices)):
        c = x_indices[i]
        r = y_indices[i]
        y[r, c] = get_id(label[r][c])

    return y


# Randomly crop 320x320 (image, label) pairs centered on pixels in the known regions.
def random_choice(label):
    y_indices, x_indices = np.where(label != 0)
    num_knowns = len(y_indices)
    c, r = 0, 0
    if num_knowns > 0:
        ix = random.choice(range(num_knowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        c = max(0, center_x - 160)
        r = max(0, center_y - 160)
    return c, r


def safe_crop(mat, x, y):
    if len(mat.shape) == 2:
        ret = np.zeros((320, 320), np.float32)
    else:
        ret = np.zeros((320, 320, 3), np.float32)
    crop = mat[y:y + 320, x:x + 320]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop

    return ret


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        filename = '{}_names.txt'.format(usage)
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols), dtype=np.int32)

        for i_batch in range(length):
            # print(i_batch)
            name = self.names[i]
            filename = os.path.join(train_color, name)
            image = cv.imread(filename)
            label = get_label(name)

            c, r = random_choice(label)
            image = safe_crop(image, c, r)
            label = safe_crop(label, c, r)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)

            x = image / 255.
            y = get_y(label)
            # print('y.shape: ' + str(y.shape))

            batch_x[i_batch, :, :, 0:3] = x
            batch_y[i_batch, :, :] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')


def split_data():
    # num_samples = 39222
    # num_train_samples = 31378
    num_valid_samples = 7844
    train_folder = 'train_color'
    names = [f for f in os.listdir(train_folder) if f.endswith('.jpg')]
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    split_data()
