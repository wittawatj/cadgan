# -*- coding: utf-8 -*-
"""
Utility functions.
Created on Wed Oct 26 11:01:51 2016
@author: M. Waleed Gondal
"""

import pickle

import numpy as np
import tensorflow as tf


class CAM:

    """CAM (Class Activation Maps) implements a weakly supervised localization method proposed by
    http://cnnlocalization.csail.mit.edu/
    The approach is published under the name of 'Learning Deep Features for Discriminative
    Localization', Bolei Zhou et al.
    Disclaimer: A part of this code is inspired by the CAM implementation of
    https://github.com/metalbubble/CAM
    The class defines a modified CNN model, VGG16, which uses Global Average Pooling (GAP) to compute
    the weights which linearly correlate the last conv layer's feature maps to the output score.
    The class initializes the network with pretrained weights for GAP provided by the authors.
    Parameters
    -----------
    weight_file_path: list of numpy arrays
        List of arrays that contain pretrained network parameters. The layers are parsed with respect
        to the respective layer name.
    n_labels: int
        An integer representing the number of output classes
    Yields
    --------
    conv6 : numpy array of float32
        A batch of last convolutional layer feature maps. Shape (batchsize, height, width, channels).
    output : numpy array of float32
        The corresponding output scores. Shape (batchsize, num_labels)"""

    def __init__(self, n_labels, weight_file_path=None):

        self.image_mean = [103.939, 116.779, 123.68]
        self.n_labels = n_labels
        assert weight_file_path is not None, "No weight file found"
        self.pretrained_weights = np.load(weight_file_path, encoding="latin1")

    def get_conv_weight(self, layer_name):
        """Accessing conv biases from the network weights"""
        return (self.pretrained_weights.item()[layer_name])["weights"]

    def get_conv_bias(self, layer_name):
        """Accessing biases from the network weights"""
        return (self.pretrained_weights.item()[layer_name])["biases"]

    def conv_layer(self, bottom, name, stride=1):

        """For the implementation convolutional layer using tensorflow predefined conv function.
        Parameters
        ----------
        bottom: Tensor
            A tensor of shape (batchsize, height, width, channels)
        name: String
            A name for the variable scope according to the layer it belongs to.
        stride: Int
            An integer value defining the convolutional layer stride.
        Yields
        --------
        relu : Tensor
            A tensor of shape (batchsize, height, width, channels)"""

        with tf.variable_scope(name) as scope:

            w = self.get_conv_weight(name)
            b = self.get_conv_bias(name)
            conv_weights = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv_biases = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))

            conv = tf.nn.conv2d(bottom, conv_weights, [1, stride, stride, 1], padding="SAME")
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias, name=name)

        return relu

    def network(self, image, is_training=True, dropout=1.0):

        """ Defines the VGG16 Network with Global Average Pooling (GAP) configuration.
        Parameters
        ----------
        rgb: Tensor
            A tensor of shape (batchsize, height, width, channels)
        Yields
        --------
        conv6 : numpy array of float32
            A batch of last convolutional layer feature maps. Shape (batchsize, height, width, channels).
        output : numpy array of float32
            The corresponding output scores. Shape (batchsize, num_labels)"""

        image *= 255.0
        r, g, b = tf.split(image, [1, 1, 1], 3)
        # with tf.Session() as sess:
        #    print(sess.run(r))
        image = tf.concat([b - self.image_mean[0], g - self.image_mean[1], r - self.image_mean[2]], 3)

        relu1_1 = self.conv_layer(image, "conv1_1")
        relu1_2 = self.conv_layer(relu1_1, "conv1_2")

        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

        relu2_1 = self.conv_layer(pool1, "conv2_1")
        relu2_2 = self.conv_layer(relu2_1, "conv2_2")
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

        relu3_1 = self.conv_layer(pool2, "conv3_1")
        relu3_2 = self.conv_layer(relu3_1, "conv3_2")
        relu3_3 = self.conv_layer(relu3_2, "conv3_3")
        pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")

        relu4_1 = self.conv_layer(pool3, "conv4_1")
        relu4_2 = self.conv_layer(relu4_1, "conv4_2")
        relu4_3 = self.conv_layer(relu4_2, "conv4_3")
        pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4")

        relu5_1 = self.conv_layer(pool4, "conv5_1")
        relu5_2 = self.conv_layer(relu5_1, "conv5_2")
        relu5_3 = self.conv_layer(relu5_2, "conv5_3")

        # Introduction of new conv layer (CONV 6) in conventional VGG Net to increase mapping resolution.

        # ***************************************************************************************************************
        # IN CAFFE implementation, Grouped Convolution or Depthwise Grouping is done. It's dividing the input and kernel into
        # 2 halves, compute convolutions depthwise and then concating the results
        # When group=2, the first half of filters are only connected to the first half of input channels and same for 2nd half
        # of filters
        # There is no need to do this for training purpose
        # Following code is to implement Grouped Conv in TF
        # ***************************************************************************************************************

        with tf.variable_scope("CAM_conv"):
            name = "CAM_conv"
            w = self.get_conv_weight(name)
            b = self.get_conv_bias(name)

            conv_weights = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv_biases = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))

            # Split relu5_3 2 groups of 256 and 256 filters
            group = 2
            group1, group2 = tf.split(relu5_3, group, 3)

            # Import CAM_conv weights here i.e. kernel = (3,3,256,1024) and divide it on its output
            # into 2 kernels = [3,3,256,512]
            kernel1, kernel2 = tf.split(conv_weights, group, 3)
            conv_grp1 = tf.nn.conv2d(group1, kernel1, [1, 1, 1, 1], padding="SAME")
            conv_grp2 = tf.nn.conv2d(group2, kernel2, [1, 1, 1, 1], padding="SAME")
            conv_output = tf.concat([conv_grp1, conv_grp2], 3)

            conv6 = tf.nn.bias_add(conv_output, conv_biases)
            gap = tf.reduce_mean(conv6, [1, 2])
            gap = tf.nn.dropout(gap, dropout)

        with tf.variable_scope("CAM_fc"):
            name = "CAM_fc"
            w = self.get_conv_weight(name)

            gap_w = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
        output = tf.matmul(gap, gap_w)

        # Reshape the feature extractor
        fmaps_resized = tf.reshape(pool4, [-1, 131072])  # height*width*num_fmaps [-1, 14*14*512 ]
        # fmaps_resized = tf.reshape(output, [-1, 1000 ])
        return fmaps_resized, output

    def get_cam(self, label, fmaps, height=224, width=224, num_fmaps=1024):

        """ Compute the Class Activation Maps
        Parameters
        -----------
        label: Int
            An integer value corresponding to the class whose class activation map is to be computed
        fmaps: Numpy array of float32
            A batch of feature maps. Shape (batchsize, height, width, channels).
        height: Int
            An integer to which the CAM height is to be upsampled. It should be the height of input image.
        width: Int
            An integer to which the CAM width is to be upsampled. It should be the width of input image
        num_fmaps: Int
            Corresponds to the number of feature maps in the last convolutional layer. In simple terms it's the depth of
        Returns
        ---------
        Class Activation Map (CAM), a single channeled, upsampled, weighted sum of last conv filter maps. """

        fmaps_resized = tf.image.resize_bilinear(fmaps, [height, width])

        # Retrieve Fully Connected Weights and get the weights with respect to the required label
        with tf.variable_scope("CAM_fc", reuse=True):
            label_w = tf.gather(tf.transpose(tf.get_variable("W")), label)
            label_w = tf.reshape(label_w, [-1, num_fmaps, 1])

        # Reshape fmaps and compute weighted sum of feature maps using label_w
        fmaps_resized = tf.reshape(fmaps_resized, [-1, height * width, num_fmaps])
        classmap = tf.matmul(fmaps_resized, label_w)

        # Resize the feature maps back to the input image size
        classmap = tf.reshape(classmap, [-1, height, width])
        return classmap
