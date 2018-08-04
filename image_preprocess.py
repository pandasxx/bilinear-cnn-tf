# 

import os
import random
import tensorflow as tf 
import matplotlib.pyplot as plt
from PIL import Image
from image_enhancement import *

def read_and_decode(tfrecord_filename, size, classes, use_data_enhancement):
    read_queue = tf.train.string_input_producer([tfrecord_filename], shuffle = False)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(read_queue)
    features = tf.parse_single_example(serialized_example,
                                features={
                                    'label': tf.FixedLenFeature([], tf.int64),
                                    'img_raw': tf.FixedLenFeature([], tf.string),
                                })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [size, size, 3])
    
    # 在这里归一化
#    img = tf.cast(img, tf.float32) * (1. / 255)
    # 不归一化
    img = tf.cast(img, tf.float32)
    
    # 在这里增加随机数据增强
    if (True == use_data_enhancement):
        img = image_random_enhancement_tf(img)
 
    label = tf.cast(features['label'], tf.int32)
    
    # 在这里onehot编码
    label = tf.one_hot(label, depth = classes, on_value=None, off_value=None, axis=None, dtype=tf.int32, name=None)

    return img, label

def get_batchs(tfrecord_filename, image_size, classes, batch_size, min_after_dequeue, use_data_enhancement = False):
    imgs_queue, labels_queue = read_and_decode(tfrecord_filename, image_size, classes, use_data_enhancement)


    img_batch, label_batch = tf.train.shuffle_batch([imgs_queue, labels_queue],
                                                    batch_size = batch_size, capacity = min_after_dequeue + batch_size * 3,
                                                    min_after_dequeue = min_after_dequeue)
#    img_batch, label_batch = tf.train.batch([imgs_queue, labels_queue],
#                                            batch_size = batch_size, capacity = batch_size * 8)

    return img_batch, label_batch


def submit_get_batchs(tfrecord_filename, image_size, batch_size):
    read_queue = tf.train.string_input_producer([tfrecord_filename], shuffle = False)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(read_queue)
    features = tf.parse_single_example(serialized_example,
                                features={
                                    'img_raw': tf.FixedLenFeature([], tf.string),
                                })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [image_size, image_size, 3])
    
    # 在这里归一化
    # img = tf.cast(img, tf.float32) * (1. / 255)
    # 不归一化
    img = tf.cast(img, tf.float32) 

    img_batch = tf.train.batch([img], batch_size = batch_size, capacity = batch_size * 3)

    return img_batch