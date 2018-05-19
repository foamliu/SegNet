import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import colors
from data_generator import get_label
from data_generator import random_choice
from data_generator import safe_crop
from model import build_encoder_decoder

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    num_labels = 8

    model_weights_path = 'models/model.10-0.1997.hdf5'
    model = build_encoder_decoder()

    model.load_weights(model_weights_path)
    print(model.summary())

    filename = 'valid_names.txt'
    with open(filename, 'r') as f:
        names = f.read().splitlines()
    samples = random.sample(names, 10)

    root_path = ''
    valid_path = 'train_color/'

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(valid_path, image_name)

        print('Start processing image: {}'.format(filename))

        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        bgr_img = cv.imread(filename)
        height, width = bgr_img.shape[:2]
        label = get_label(image_name, root_path)

        x, y = random_choice(label)
        image = safe_crop(bgr_img, x, y)
        label = safe_crop(label, x, y)

        x_test[0, :, :, 0:3] = image / 255.
        out = model.predict(x_test)
        # print(out.shape)

        out = np.reshape(out, (img_rows, img_cols, num_labels))
        out = np.argmax(out, axis=2)
        for r in range(320):
            for c in range(320):
                out[r, c, :] = colors[out[r, c]]

        label = cv.cvtColor(label, cv.COLOR_GRAY2BGR)
        label = image * 0.6 + label * 0.4
        label = label.astype(np.uint8)

        out = image * 0.6 + out * 0.4
        out = out.astype(np.uint8)

        cv.imwrite('images/{}_image.png'.format(i), image)
        cv.imwrite('images/{}_out.png'.format(i), out)
        cv.imwrite('images/{}_label.png'.format(i), label)

    K.clear_session()
