# 

# 1 读取所有图片和标签路径
# 2 乱序1
# 3 分割2为train，test，valid
# 4 根据路径集合3来读取图片，存储在对应tfrecord中

import os
import random
import tensorflow as tf 
import matplotlib.pyplot as plt
from PIL import Image

def read_pic_path(pic_src):
    # 得出子目录
    contents = os.listdir(pic_src)
    # 有子文件的目录为有效的
    classes = [each for each in contents if os.path.isdir(pic_src + each)]
    return classes

def get_all_files_path(pic_src):
    all_images = []
    all_labels = []
    
    classes = read_pic_path(pic_src)
    for index, class_name in enumerate(classes):
        class_path = pic_src + class_name
        for img_name in os.listdir(pic_src + class_name):
            img_path = class_path + '/' + img_name
            all_images.append(img_path)
            all_labels.append(index)
    
    return all_images, all_labels, len(classes)

def shuffle_data_list(data_list, label_list):
    index = [i for i in range(len(data_list))]
    random.shuffle(index)
    data_list = [data_list[each] for each in index]
    label_list = [label_list[each] for each in index]
    return data_list, label_list

# 使用padding方式填充原图为正方形，并resize到目标大小
def resize_img(img, size):
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2
    img_croped = img.crop(
        (
            -horizontal_padding,
            -vertical_padding,
            img.size[0] + horizontal_padding,
            img.size[1] + vertical_padding
        )
    )
    img_resized = img_croped.resize((size, size))
    return img_resized

# 读取图片到指定的tfrecord文件中
def read_img_2_tfrecord(imgs_list, label_list, target_size, dst_tfrecord_filename):
    writer = tf.python_io.TFRecordWriter(dst_tfrecord_filename)
    for index, img_path in enumerate(imgs_list):
        img = Image.open(img_path)
        # 有些图片是png（RGBA），会导致后面出错，在这里统一转换成RGB
        if (img.mode != 'RGB'):
            img =img.convert("RGB")
        img_resized = resize_img(img, target_size)
#       img_resized.show()
        img.close()
        img_raw = img_resized.tobytes()
        #you de wenjian you 4channels
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label_list[index]])),
            'img_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

def data_preprocess(pic_src, train_ratio, pic_size, data_des):
    # 读取所有路径，并乱序排列
    all_images, all_labels, classes = get_all_files_path(pic_src)
    all_images, all_labels = shuffle_data_list(all_images, all_labels)

    #获取切割num
    data_num = len(all_labels)
    train_num = int(data_num * train_ratio)
    test_num = int(data_num * ((1.0 - train_ratio) / 2))

    #切割label
    train_labels = all_labels[:train_num]                                       # 切分出 从 0 到第train_num-1个（或者说前train_num个）形成新数组
    test_labels  = all_labels[train_num:][:test_num]                    # 先切分出从第train_num个到最后一个，在切分出前test_num个
    valid_labels = all_labels[train_num:][test_num:]        # 先切分出从第train_num个到最后一个，在切分第test_num个到最后一个
    #切割image
    train_images = all_images[:train_num]
    test_images  = all_images[train_num:][:test_num]
    valid_images = all_images[train_num:][test_num:]

    #存储到tfrecord中
    read_img_2_tfrecord(train_images, train_labels, pic_size, data_des[0])
    read_img_2_tfrecord(test_images, test_labels, pic_size, data_des[1])
    read_img_2_tfrecord(valid_images, valid_labels, pic_size, data_des[2])

    return train_num, test_num, (data_num - train_num - test_num), classes

def submit_data_preprocess(pic_src, pic_size, data_des):
    all_images = []
    for img_name in os.listdir(pic_src):
            img_path =  pic_src + img_name
            all_images.append(img_path)

    writer = tf.python_io.TFRecordWriter(data_des)
    for index, img_path in enumerate(all_images):
        img = Image.open(img_path)
        # 有些图片是png（RGBA），会导致后面出错，在这里统一转换成RGB
        if (img.mode != 'RGB'):
            img =img.convert("RGB")
        img_resized = resize_img(img, pic_size)
#       img_resized.show()
        img.close()
        img_raw = img_resized.tobytes()
        #you de wenjian you 4channels
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

    return len(all_images)




