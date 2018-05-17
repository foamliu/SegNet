import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = '../../data/cvpr-2018-autonomous-driving'
image_path = os.path.join(ROOT, 'train_color/170908_061523257_Camera_5.jpg')
im = Image.open(image_path)
tlabel_path = os.path.join(ROOT, "train_label/170908_061523257_Camera_5_instanceIds.png")
tlabel = np.asarray(Image.open(tlabel_path)) // 1000
tlabel[tlabel != 0] = 255
plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))
plt.show()

# dst = cv.addWeighted(im, 0.7, tlabel, 0.3, 0)
# dst = dst.astype(np.uint8)
# cv.namedWindow('image', cv.WINDOW_NORMAL)
# cv.resizeWindow('image', 800, 600)
# cv.imshow('image', tlabel)
# cv.waitKey(0)
# cv.destroyAllWindows()

# cutting off everything after class 65, see note below
classdict = {0: 'others', 1: 'rover', 17: 'sky', 33: 'car', 34: 'motorbicycle', 35: 'bicycle', 36: 'person',
             37: 'rider', 38: 'truck', 39: 'bus', 40: 'tricycle', 49: 'road', 50: 'siderwalk', 65: 'traffic_cone'}

tlabel = np.asarray(Image.open(os.path.join(ROOT, 'train_label/170908_061523257_Camera_5_instanceIds.png')))
cls = np.unique(tlabel) // 1000
unique, counts = np.unique(cls, return_counts=True)
d = dict(zip(unique, counts))
df = pd.DataFrame.from_dict(d, orient='index').transpose()
df.rename(columns=classdict, inplace=True)
print(df)
