import multiprocessing

import tensorflow as tf
from tensorflow.python.client import device_lib


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


def pixelwise_crossentropy(y_true, y_pred):
    output = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    return - tf.reduce_sum(y_true * tf.log(output))
