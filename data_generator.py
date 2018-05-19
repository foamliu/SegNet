import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
from PIL import Image

from config import batch_size
from config import img_cols
from config import img_rows

train_color = 'train_color'
train_label = 'train_label'

num_labels = 8
decode_dict = {0: 'others', 33: 'car', 34: 'motorbicycle', 35: 'bicycle', 36: 'person', 38: 'truck', 39: 'bus', 40: 'tricycle'}
encode_dict = {'others': 0, 'car': 1, 'motorbicycle': 2, 'bicycle': 3, 'person': 4, 'truck': 5, 'bus': 6, 'tricycle': 7}


def get_id(old_id):
    return encode_dict[decode_dict[old_id]]


def get_label(name, path=''):
    label_name = name.split('.')[0] + '_instanceIds.png'
    path = os.path.join(path, train_label)
    filename = os.path.join(path, label_name)
    label = np.asarray(Image.open(filename)) // 1000
    label[(label != 0) & (label != 33) & (label != 34) & (label != 35) & (label != 36) & (label != 38) & (label != 39) & (label != 40)] = 0
    return label


def get_label_map(label):
    label_map = np.zeros([320, 320])
    y_indices, x_indices = np.where(label != 0)
    for i in range(len(x_indices)):
        c = x_indices[i]
        r = y_indices[i]
        label_map[r, c] = get_id(label[r][c])

    return label_map


# Randomly crop 320x320 (image, label) pairs centered on pixels in the known regions.
def random_choice(label):
    y_indices, x_indices = np.where(label != 0)
    num_knowns = len(y_indices)
    x, y = 0, 0
    if num_knowns > 0:
        ix = random.choice(range(num_knowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - 160)
        y = max(0, center_y - 160)
    return x, y


def safe_crop(mat, x, y):
    if len(mat.shape) == 2:
        ret = np.zeros((320, 320), np.float32)
    else:
        ret = np.zeros((320, 320, 3), np.float32)
    crop = mat[y:y + 320, x:x + 320]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop

    return ret


def data_gen(usage):
    filename = '{}_names.txt'.format(usage)
    with open(filename, 'r') as f:
        names = f.read().splitlines()
    i = 0
    while True:
        batch_x = np.empty((batch_size, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((batch_size, img_rows, img_cols), dtype=np.float32)

        for i_batch in range(batch_size):
            # print(i_batch)
            name = names[i]
            filename = os.path.join(train_color, name)
            image = cv.imread(filename)
            height, width = image.shape[:2]
            label = get_label(name)

            # if np.random.random_sample() > 0.5:
            x, y = random_choice(label)
            # else:
            #     # have a better understanding in 'others'
            #     x = random.randint(0, width - 320)
            #     y = random.randint(0, height - 320)

            image = safe_crop(image, x, y)
            label = safe_crop(label, x, y)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)

            label_map = get_label_map(label)

            batch_x[i_batch, :, :, 0:3] = image / 255.
            batch_y[i_batch, :, :] = label_map

            i += 1
            if i >= len(names):
                i = 0

        yield batch_x, batch_y


def train_gen():
    return data_gen('train')


def valid_gen():
    return data_gen('valid')


def shuffle_data():
    num_samples = 39222
    num_train_samples = 31378
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
    shuffle_data()
