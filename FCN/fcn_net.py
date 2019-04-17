
import os, sys
import scipy.io
import numpy as np
import tensorflow as tf
import TensorflowUtils as utils

class fcn_net(object):
    def __init__(self, x, keep_prob, num_classes, is_train = True):

        # Parse input arguments into class variables
        self.X = x / 255.0 * 2 - 1
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.is_train = is_train
        # Call the create function to build the computational graph of Net
        self.create()

    def create(self):

        with tf.variable_scope("inference"):
            # output 112x112x64
            W1 = utils.weight_variable([5, 5, 3, 64], name="W1")
            b1 = utils.bias_variable([64], name="b1")
            conv1 = utils.conv2d_basic(self.X, W1, b1)
            relu1 = tf.nn.relu(conv1, name="relu1")
            relu1_bn = tf.contrib.layers.batch_norm(relu1, scale=True, is_training=self.is_train,
                                                    updates_collections=None)
            pool1 = utils.max_pool_2x2(relu1_bn)

            # output 56x56x128
            W2 = utils.weight_variable([5, 5, 64, 128], name="W2")
            b2 = utils.bias_variable([128], name="b2")
            conv2 = utils.conv2d_basic(pool1, W2, b2)
            relu2 = tf.nn.relu(conv2, name="relu2")
            relu2_bn = tf.contrib.layers.batch_norm(relu2, scale=True, is_training=self.is_train,
                                                    updates_collections=None)
            pool2 = utils.max_pool_2x2(relu2_bn)

            # output 28x28x128
            W3 = utils.weight_variable([5, 5, 128, 128], name="W3")
            b3 = utils.bias_variable([128], name="b3")
            conv3 = utils.conv2d_basic(pool2, W3, b3)
            relu3 = tf.nn.relu(conv3, name="relu3")
            relu3_bn = tf.contrib.layers.batch_norm(relu3, scale=True, is_training=self.is_train,
                                                    updates_collections=None)
            pool3 = utils.max_pool_2x2(relu3_bn)

            # output 14x14x128
            W4 = utils.weight_variable([5, 5, 128, 128], name="W4")
            b4 = utils.bias_variable([128], name="b4")
            conv4 = utils.conv2d_basic(pool3, W4, b4)
            relu4 = tf.nn.relu(conv4, name="relu4")
            relu4_bn = tf.contrib.layers.batch_norm(relu4, scale=True, is_training=self.is_train,
                                                    updates_collections=None)
            pool4 = utils.max_pool_2x2(relu4_bn)

            # output 7x7x128
            W5 = utils.weight_variable([5, 5, 128, 128], name="W5")
            b5 = utils.bias_variable([128], name="b5")
            conv5 = utils.conv2d_basic(pool4, W5, b5)
            relu5 = tf.nn.relu(conv5, name="relu5")
            relu5_bn = tf.contrib.layers.batch_norm(relu5, scale=True, is_training=self.is_train,
                                                    updates_collections=None)
            pool5 = utils.max_pool_2x2(relu5_bn)

            # now to upscale to actual image size

            #upscale to pool4
            W_t1 = utils.weight_variable([5, 5, 128, 128], name="W_t1")
            b_t1 = utils.bias_variable([128], name="b_t1")
            conv_t1 = utils.conv2d_transpose_strided(pool5, W_t1, b_t1, output_shape=tf.shape(pool4))
            fuse_1 = tf.concat([conv_t1, pool4],3 , name="fuse_1")
            relu_t1 = tf.nn.relu(fuse_1, name="relu_1")
            relu_t1bn = tf.contrib.layers.batch_norm(relu_t1, scale=True, is_training=self.is_train,
                                                    updates_collections=None)

            W_t2 = utils.weight_variable([5, 5, 128, 256], name="W_t2")
            b_t2 = utils.bias_variable([128], name="b_t2")
            conv_t2 = utils.conv2d_transpose_strided(relu_t1bn, W_t2, b_t2, output_shape=tf.shape(pool3))
            fuse_2 = tf.concat([conv_t2, pool3],3 , name="fuse_2")
            relu_t2 = tf.nn.relu(fuse_2, name="relu_2")
            relu_t2bn = tf.contrib.layers.batch_norm(relu_t2, scale=True, is_training=self.is_train,
                                                    updates_collections=None)

            W_t3 = utils.weight_variable([5, 5, 128, 256], name="W_t3")
            b_t3 = utils.bias_variable([128], name="b_t3")
            conv_t3 = utils.conv2d_transpose_strided(relu_t2bn, W_t3, b_t3, output_shape=tf.shape(pool2))
            fuse_3 = tf.concat([conv_t3, pool2],3 , name="fuse_3")
            relu_t3 = tf.nn.relu(fuse_3, name="relu_3")
            relu_t3bn = tf.contrib.layers.batch_norm(relu_t3, scale=True, is_training=self.is_train,
                                                    updates_collections=None)


            W_t4 = utils.weight_variable([5, 5, 64, 256], name="W_t4")
            b_t4 = utils.bias_variable([64], name="b_t4")
            conv_t4 = utils.conv2d_transpose_strided(relu_t3bn, W_t4, b_t4, output_shape=tf.shape(pool1))
            fuse_4 = tf.concat([conv_t4, pool1],3, name="fuse_4")
            relu_t4 = tf.nn.relu(fuse_4, name="relu_4")
            relu_t4bn = tf.contrib.layers.batch_norm(relu_t4, scale=True, is_training=self.is_train,
                                                    updates_collections=None)


            shape = tf.shape(self.X)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.NUM_CLASSES])
            W_t3 = utils.weight_variable([5, 5, self.NUM_CLASSES, relu_t4bn.get_shape()[3].value], name="W_t5")
            b_t3 = utils.bias_variable([self.NUM_CLASSES], name="b_t5")
            self.logits = utils.conv2d_transpose_strided(relu_t4bn, W_t3, b_t3, output_shape=deconv_shape3)