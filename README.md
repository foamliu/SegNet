# SegNet

This repository is to take CVPR 2018 WAD Video Segmentation Challenge with SegNet.

## Dependencies
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

Follow the [instruction](https://www.kaggle.com/c/cvpr-2018-autonomous-driving/data) to download dataset.

```bash
$ kaggle competitions download -c cvpr-2018-autonomous-driving -p /mnt/data
```

## Architecture

![image](https://github.com/foamliu/SegNet/raw/master/images/segnet.png)


## ImageNet Pretrained Models
Download [VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) into models folder.

## Usage
### Data Pre-processing
Extract training images:
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo

```bash
$ python demo.py
```

Then check results in images folder, something like:

Input | GT | Output |
|---|---|---|
|![image](https://github.com/foamliu/SegNet/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/0_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/0_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/1_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/1_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/1_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/2_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/2_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/2_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/3_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/3_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/3_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/4_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/4_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/5_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/5_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/5_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/6_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/6_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/6_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/7_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/7_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/7_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/8_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/8_out.png)|
|![image](https://github.com/foamliu/SegNet/raw/master/images/9_image.png)  | ![image](https://github.com/foamliu/SegNet/raw/master/images/9_label.png) | ![image](https://github.com/foamliu/SegNet/raw/master/images/9_out.png)|

