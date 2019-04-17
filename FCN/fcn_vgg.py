
import os, sys
import scipy.io
import numpy as np
import tensorflow as tf
from six.moves import urllib
import TensorflowUtils as utils

class fcn_vgg(object):
    def __init__(self, x, keep_prob, num_classes, model = 'fcn8'):

        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob

        self.WEIGHTS = self.get_model_data('http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat')
        self.MODEL = model
        # Call the create function to build the computational graph of Net
        self.create()

    def get_model_data(self, url_name):
        filepath = './Dataset/' + url_name.split('/')[-1]
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' % (filepath, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
            print()
            statinfo = os.stat(filepath)
            print('Succesfully downloaded', filepath, statinfo.st_size, 'bytes.')
        if not os.path.exists(filepath):
            raise IOError("VGG Model not found!")
        data = scipy.io.loadmat(filepath)
        return data


    def vgg(self, weights, image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        net = {}
        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = utils.avg_pool_2x2(current)
            net[name] = current

        return net

    def create(self):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """

        self.mean = self.WEIGHTS['normalization'][0][0][0]
        mean_pixel = np.mean(self.mean, axis=(0, 1))

        weights = np.squeeze(self.WEIGHTS['layers'])

        processed_image = utils.process_image(self.X, mean_pixel)

        with tf.variable_scope("inference"):
            image_net = self.vgg(weights, processed_image)
            conv_final_layer = image_net["conv5_3"]

            pool5 = utils.max_pool_2x2(conv_final_layer)

            W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
            b6 = utils.bias_variable([4096], name="b6")
            conv6 = utils.conv2d_basic(pool5, W6, b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=self.KEEP_PROB)

            W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
            b7 = utils.bias_variable([4096], name="b7")
            conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=self.KEEP_PROB)

            W8 = utils.weight_variable([1, 1, 4096, self.NUM_CLASSES], name="W8")
            b8 = utils.bias_variable([self.NUM_CLASSES], name="b8")
            conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

            # now to upscale to actual image size
            if self.MODEL == 'fcn8':
                deconv_shape1 = image_net["pool4"].get_shape()
                W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, self.NUM_CLASSES], name="W_t1")
                b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
                conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
                fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

                deconv_shape2 = image_net["pool3"].get_shape()
                W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
                b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
                conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
                fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

                shape = tf.shape(self.X)
                deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.NUM_CLASSES])
                W_t3 = utils.weight_variable([16, 16, self.NUM_CLASSES, deconv_shape2[3].value], name="W_t3")
                b_t3 = utils.bias_variable([self.NUM_CLASSES], name="b_t3")
                self.logits = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
            elif self.MODEL == 'fcn16':
                deconv_shape1 = image_net["pool4"].get_shape()
                W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, self.NUM_CLASSES], name="W_t1")
                b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
                conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
                fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

                shape = tf.shape(self.X)
                deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.NUM_CLASSES])
                W_t3 = utils.weight_variable([32, 32, self.NUM_CLASSES, fuse_1.get_shape()[3].value], name="W_t3")
                b_t3 = utils.bias_variable([self.NUM_CLASSES], name="b_t3")
                self.logits = utils.conv2d_transpose_strided(fuse_1, W_t3, b_t3, output_shape=deconv_shape3, stride=16)
            elif self.MODEL == 'fcn32':
                shape = tf.shape(self.X)
                deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.NUM_CLASSES])
                W_t3 = utils.weight_variable([64, 64, self.NUM_CLASSES, conv8.get_shape()[3].value], name="W_t3")
                b_t3 = utils.bias_variable([self.NUM_CLASSES], name="b_t3")
                self.logits = utils.conv2d_transpose_strided(conv8, W_t3, b_t3, output_shape=deconv_shape3, stride=32)
            else:
                print ('error!')



