import os
import random
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data_preprocess import *
from image_preprocess import *

class vgg_bilinear_model:
    def __init__(self, imgs, keep_prob, weights_file = None, weights_file_last = None, sess = None, finetune = False):
        self.finetune = finetune                     	# train all layers or only last layer
        self.imgs = imgs                             	# input img data
        self.keep_prob = keep_prob                   	# fc dropout ratio
        self.last_layer_parameters = []              	# param only last fc layer, normal train will use
        self.parameters = []                         	# param whole net, fine tune will use
        self.weights_file = weights_file             	# pretrained weights file path
        self.weights_file_last = weights_file_last   	# last fc layer weights file path
        self.conv_layers()                           	# model's conv part, here use vgg
        self.bilinear_layers()                       	# model's bilinear part, feature fusion
        self.fc_layers()                             		# model's classify part, full connect
    
    def conv_layers(self):
        # preprocess, not 0-1 but minus average value
        #with tf.name_scope('preprocess') as scope:
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        images = self.imgs - mean
        print('Adding Data Augmentation')
            
        # -------------------------------------------------------------------------------------------------------- #
        
        # conv1_1
        with tf.variable_scope('conv1_1'):
            weights = tf.get_variable('W', [3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [64], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(images, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv1_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # conv1_2
        with tf.variable_scope('conv1_2'):
            weights = tf.get_variable('W', [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [64], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.conv1_1, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv1_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool1')

        # -------------------------------------------------------------------------------------------------------- #

        # conv2_1
        with tf.variable_scope('conv2_1'):
            weights = tf.get_variable('W', [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [128], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.pool1, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv2_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # conv2_2
        with tf.variable_scope('conv2_2'):
            weights = tf.get_variable('W', [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [128], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.conv2_1, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv2_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool2')
        
        # -------------------------------------------------------------------------------------------------------- #

        # conv3_1
        with tf.variable_scope('conv3_1'):
            weights = tf.get_variable('W', [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [256], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.pool2, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv3_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # conv3_2
        with tf.variable_scope('conv3_2'):
            weights = tf.get_variable('W', [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [256], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.conv3_1, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv3_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # conv3_3
        with tf.variable_scope('conv3_3'):
            weights = tf.get_variable('W', [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [256], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.conv3_2, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv3_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool3')

        # -------------------------------------------------------------------------------------------------------- #

        # conv4_1
        with tf.variable_scope('conv4_1'):
            weights = tf.get_variable('W', [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.pool3, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv4_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # conv4_2
        with tf.variable_scope('conv4_2'):
            weights = tf.get_variable('W', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.conv4_1, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv4_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # conv4_3
        with tf.variable_scope('conv4_3'):
            weights = tf.get_variable('W', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.conv4_2, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv4_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool4')

        # -------------------------------------------------------------------------------------------------------- #

        # conv5_1
        with tf.variable_scope('conv5_1'):
            weights = tf.get_variable('W', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.pool4, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv5_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # conv5_2
        with tf.variable_scope('conv5_2'):
            weights = tf.get_variable('W', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.conv5_1, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv5_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
        
        # conv5_3
        with tf.variable_scope('conv5_3'):
            weights = tf.get_variable('W', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(), trainable = self.finetune)
            biases = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.1), trainable = self.finetune)
            conv = tf.nn.conv2d(self.conv5_2, weights, strides = [1, 1, 1, 1], padding = 'SAME')
            self.conv5_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]
    
    def bilinear_layers(self):
        conv1 = tf.transpose(self.conv5_3, perm = [0, 3, 1, 2])
        conv1 = tf.reshape(conv1, [-1, 512, 196])
        
        conv2 = tf.transpose(self.conv5_3, perm = [0, 3, 1, 2])
        conv2 = tf.reshape(conv1, [-1, 512, 196])
        conv2 = tf.transpose(conv2, perm = [0, 2, 1])
        
        phi_I = tf.matmul(conv1, conv2)
        phi_I = tf.reshape(phi_I, [-1, 512 * 512])
        phi_I = tf.divide(phi_I, 196.0)
        
        y_ssqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
        
        z = tf.nn.l2_normalize(y_ssqrt, dim = 1)
        
        print('Shape of z', z.get_shape())
        
        self.bilinear_feature = z
    
    def fc_layers(self):
        with tf.variable_scope('fc-new') as scope:
            fc3w = tf.get_variable('W', [512 * 512, 12], initializer = tf.contrib.layers.xavier_initializer(), trainable = True)
            fc3b = tf.get_variable('b', [12], initializer=tf.constant_initializer(0.1), trainable = True)
            fc3l = tf.nn.bias_add(tf.matmul(self.bilinear_feature, fc3w), fc3b)
            self.fc3l = tf.nn.dropout(fc3l, self.keep_prob)
            self.last_layer_parameters += [fc3w, fc3b]
    
    # for first train use, load pretrained vgg weights 
    def load_vgg_weights(self, session):
        weights_dict = np.load(self.weights_file, encoding = 'bytes')
        vgg_layers = ['conv1_1',
                      'conv1_2',
                      'conv2_1',
                      'conv2_2',
                      'conv3_1',
                      'conv3_2',
                      'conv3_3',
                      'conv4_1',
                      'conv4_2',
                      'conv4_3',
                      'conv5_1',
                      'conv5_2',
                      'conv5_3']

        for op_name in vgg_layers:
            with tf.variable_scope(op_name, reuse = True):
                # biases
                var = tf.get_variable('b')
                print('Adding weights to',var.name)
                session.run(var.assign(weights_dict[op_name + '_b']))
                
                # weights
                var = tf.get_variable('W')
                print('Adding weights to',var.name)
                session.run(var.assign(weights_dict[op_name + '_W']))
    
    # for finetune train use, load last fc layer weights 
    def load_last_layer_weights(self, session):
        weights_dict = np.load(self.weights_file_last, encoding = 'bytes')
        print('last_layer', weights_dict['last_layer'][0])

        # load last fc layer weights 
        for i, var in enumerate(self.last_layer_parameters):
            session.run(var.assign(weights_dict['last_layer'][i]))
            print('Adding weights to',var.name)
    
    # for predict use, load all trained weights 
    def load_all_own_weights(self, session):
        weights_dict_conv = np.load(self.weights_file, encoding = 'bytes')
        weights_dict_fc = np.load(self.weights_file_last, encoding = 'bytes')

        print('conv_layer', weights_dict_conv['conv_layer'][0])
        print('last_layer', weights_dict_fc['last_layer'][0])
        
        # load conv layer weights
        for i, var in enumerate(self.parameters):
            session.run(var.assign(weights_dict_conv['conv_layer'][i]))
            print('Adding weights to',var.name)
         
        # load last fc layer weights 
        for i, var in enumerate(self.last_layer_parameters):
            session.run(var.assign(weights_dict_fc['last_layer'][i]))
            print('Adding weights to',var.name)