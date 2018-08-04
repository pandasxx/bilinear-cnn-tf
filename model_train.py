import os
import random
import tensorflow as tf 
import matplotlib.pyplot as plt
from PIL import Image
from data_preprocess import *
from image_preprocess import *

def build_input(image_shape, n_classes):
    input_x = tf.placeholder(tf.float32, shape = [None, image_shape[0], image_shape[1], image_shape[2]], name = "input_x")
    input_y = tf.placeholder(tf.int32, shape = [None, n_classes], name = "input_y")
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    
    return input_x, input_y, keep_prob


def build_model(train_x, train_y, keep_prob):
    # input 256 256 3, output 128 128 16
    conv1 = tf.contrib.layers.conv2d(inputs = train_x, num_outputs = 16, kernel_size = 3, stride = 1, padding = 'SAME')
    pool1 = tf.contrib.layers.max_pool2d(inputs = conv1, kernel_size = 2, stride = 2, padding = 'SAME')
    
    # input 128 128 16, output 64 64 32
    conv2 = tf.contrib.layers.conv2d(inputs = pool1, num_outputs = 32, kernel_size = 3, stride = 1, padding = 'SAME')
    pool2 = tf.contrib.layers.max_pool2d(inputs = conv2, kernel_size = 2, stride = 2, padding = 'SAME')
    
    # input 64 64 32, output 32 32 64
    conv3 = tf.contrib.layers.conv2d(inputs = pool2, num_outputs = 64, kernel_size = 3, stride = 1, padding = 'SAME')
    pool3 = tf.contrib.layers.max_pool2d(inputs = conv3, kernel_size = 2, stride = 2, padding = 'SAME')
    
    # input 32 32 64, output 16 16 128
    conv4 = tf.contrib.layers.conv2d(inputs = pool3, num_outputs = 128, kernel_size = 3, stride = 1, padding = 'SAME')
    pool4 = tf.contrib.layers.max_pool2d(inputs = conv4, kernel_size = 2, stride = 2, padding = 'SAME')
    
    # input 16 16 128, output 8 8 256
    conv5 = tf.contrib.layers.conv2d(inputs = pool4, num_outputs = 256, kernel_size = 3, stride = 1, padding = 'SAME')
    pool5 = tf.contrib.layers.max_pool2d(inputs = conv5, kernel_size = 2, stride = 2, padding = 'SAME')
    
    # input 8 8 256, output 4 4 512
    # conv6 = tf.contrib.layers.conv2d(inputs = pool5, num_outputs = 512, kernel_size = 3, stride = 1, padding = 'SAME')
    # pool6 = tf.contrib.layers.max_pool2d(inputs = conv6, kernel_size = 2, stride = 2, padding = 'SAME')
    
    # input 4 4 512, output 4*4*512
    flatten_out = tf.contrib.layers.flatten(inputs = pool5)
    
    # input 4*4*512, output 1024
    fc = tf.contrib.layers.fully_connected(flatten_out, 1024)
    fc_out = tf.contrib.layers.dropout(fc, keep_prob)

    logits = tf.contrib.layers.fully_connected(fc_out, int(train_y.shape[1]), activation_fn = None)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=logits)
    cost = tf.reduce_mean(cross_entropy, name = "cost")

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    predicted = tf.nn.softmax(logits, name = "predicted")
    correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(train_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = "accuracy")
    
    return optimizer


def build_graph(image_size, classes):
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():

        input_x, input_y, keep_prob = build_input([image_size, image_size, 3], classes)
        optimizer = build_model(input_x, input_y, keep_prob)
    
    return graph, optimizer

def train(graph, optimizer, image_size, classes, batch_size, epoch, keep_prob_value, save_model_path):

    # get tensor
    input_x = graph.get_tensor_by_name('input_x:0')
    input_y = graph.get_tensor_by_name('input_y:0')
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    cost = graph.get_tensor_by_name("cost:0")
    predicted = graph.get_tensor_by_name("predicted:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")


    with tf.Session(graph = graph) as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
    
        # read database
        train_img_batch, train_label_batch = get_batchs(tfrecord_filename = "train.tfrecords", 
                                                    image_size = image_size, classes = classes,
                                                    batch_size = batch_size, min_after_dequeue = 500)
        test_img_batch, test_label_batch = get_batchs(tfrecord_filename = "test.tfrecords", 
                                                  image_size = image_size, classes = classes,
                                                  batch_size = batch_size, min_after_dequeue = 500)

        # init data read queue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
        # train loop
        for i in range(int((4512 / batch_size) * epoch)):
            # get data
            train_imgs, train_labels= sess.run([train_img_batch, train_label_batch])
            test_imgs, test_labels= sess.run([test_img_batch, test_label_batch])

            # train single loop
            sess.run(optimizer, feed_dict={input_x: train_imgs, input_y: train_labels, keep_prob: keep_prob_value})
            train_loss = sess.run(cost, feed_dict={input_x: train_imgs, input_y: train_labels, keep_prob: 1.})
            print("train loss: %10f"%(train_loss))
        
            # print test stat
            if (i % 10 == 0):
                print("########### loop %d ###########"%(i))
                test_loss = sess.run(cost, feed_dict={input_x: test_imgs, input_y: test_labels, keep_prob: 1.})
                test_acc = sess.run(accuracy, feed_dict={input_x: test_imgs, input_y: test_labels, keep_prob: 1.})
                print("test loss: %10f   test accuracy: %10f"%(test_loss, test_acc))
                print("########### loop %d ###########"%(i))
        
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)
        print("done")
    
        # close data queue
        coord.request_stop()
        coord.join(threads)