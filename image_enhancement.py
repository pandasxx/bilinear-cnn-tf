# 

import os
import random
from scipy import misc
import tensorflow as tf 
import matplotlib.pyplot as plt
from PIL import Image

def image_random_enhancement_tf(img):

# 本数据集不适用 亮度 和 色彩 变化

# 随机亮度变化
#    img = tf.image.random_brightness(img, max_delta = 0.10)

# 随机色彩变化
#

# 随机上下翻转
    img = tf.image.random_flip_up_down(img)

# 随机左右翻转
    img = tf.image.random_flip_left_right(img)

# 随机对角线反转
    value = random.uniform(-1, 1)
    if value > 0:
        img = tf.image.transpose_image(img)

# 随机旋转
#    angle = random.uniform(-15, 15)
#    img = misc.imrotate(img, angle, 'bicubic')

    return img

def image_random_enhancement(imgs):
    # 随机旋转
    for i, img in enumerate(imgs):
        angle = random.uniform(-45, 45)
        imgs[i] = misc.imrotate(imgs[i], angle, 'bicubic')

    return imgs