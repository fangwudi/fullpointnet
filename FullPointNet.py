#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "fullpointnet"
__author__ = 'fangwudi'
__time__ = '17-12-7 09：45'

FullPointNet model for Keras.
This model is used for keypoint detection
Also do note that this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.
"""

from __future__ import print_function
from __future__ import absolute_import
import warnings
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
# modified keras version of DepthwiseConv2D using tensorflow
from DepthwiseConv2D import DepthwiseConv2D


def FullPointNet(input_tensor=None, input_shape=None):
    """Instantiates the FullPointNet architecture.
     This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    You should set `image_data_format='channels_last'` in your Keras config
    located at ~/.keras/keras.json.

    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    # check backend and image data format
    if K.backend() != 'tensorflow':
        raise RuntimeError('The FullPointNet model is only available with '
                           'the TensorFlow backend.')
    if K.image_data_format() != 'channels_last':
        warnings.warn('The FullPointNet model is only available for the '
                      'input data format "channels_last" ')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None
    # define img_input
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # normalize
    img_input = preprocess_input(img_input)
    # difine block parameters, first column not used
    depth_mul = [0, 16, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1]
    point_filters = [3, 24, 24, 32, 32, 48, 48, 64, 64, 96, 96, 128, 128]
    depth_mul += [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
    point_filters += [96, 96, 64, 64, 48, 48, 32, 32, 24, 24, 16, 13]
    # first block
    x = Block(img_input, depth_mul[1], point_filters[1], blockindex=1,
              style1=True, style2='A')
    # level1 blocks
    for i in range(2, 13):
        x = Block(x, depth_mul[i], point_filters[i], blockindex=i, style2='A')
    # P1 output
    p1 = Lambda(cropdepth, output_shape=cropdepth_output_shape, name='p1')(x)
    # level2 blocks
    for i in range(13, 19):
        x = Block(x, depth_mul[i], point_filters[i], blockindex=i, style2='B')
    # P2 output
    p2 = Lambda(cropdepth, output_shape=cropdepth_output_shape, name='p2')(x)
    # level3 blocks
    for i in range(19, 25):
        x = Block(x, depth_mul[i], point_filters[i], blockindex=i, style2='C')
    # P3 output
    p3 = Lambda(cropdepth, output_shape=cropdepth_output_shape, name='p3')(x)
    out = [p1, p2, p3]
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, out, name='FullPointNet')

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf')


def Block(x, depth_mul, point_filters, blockindex=None, style1=False,
          style2='A'):
    """
    :param x: input tensor
    :param depth_mul:
    :param point_filters:
    :param blockindex: index of block for name prefix
    :param style1: block style, decide if a start block
    :param style2: block style, decide to use 3*3 conv once or twice, and dilation
    :return: output tensor
    basic block using residual, depthwise convolution and pointwise convolution
    output hold the same shape of height*width as input
    """
    prefix = 'block{}'.format(blockindex)
    # residual save
    residual = x
    # start block A will not have activation as head
    if not style1:
        x = Activation('relu', name=prefix + '_depthconv_act')(x)
    # convolution dilation_rate, block C has extra dilation
    if style2 == 'C':
        rate = (2, 2)
    else:
        rate = (1, 1)
    # depth convolution
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                        depth_multiplier=depth_mul, use_bias=False,
                        dilation_rate=rate, name=prefix + '_depthconv')(x)
    if style2 == 'B' or 'C':
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same',
                            depth_multiplier=depth_mul, use_bias=False,
                            dilation_rate=rate, name=prefix + '_depthconv_2')(x)
    x = BatchNormalization(name=prefix + '_depthconv_bn')(x)
    x = Activation('relu', name=prefix + '_pointconv_act')(x)
    # pointwise convolution
    x = Conv2D(point_filters, (1, 1), strides=(1, 1), padding='same',
               use_bias=False, name=prefix + '_pointconv')(x)
    x = BatchNormalization(name=prefix + '_pointconv_bn')(x)
    # residual pointwise convolution
    residual = Conv2D(point_filters, (1, 1), strides=(1, 1), padding='same',
                      use_bias=False, name=prefix + '_residualconv')(residual)
    residual = BatchNormalization(name=prefix + '_residualconv_bn')(residual)
    # residual add
    x = layers.add([x, residual])
    return x


def cropdepth(x):
    # slice a Tensor to n depth, n euals 4*keypoint
    # input dimension should be [batch, height, width, depth]
    depth = K.int_shape(x)[3]
    assert depth >= 13
    return x[:, :, :, :13]


def cropdepth_output_shape(input_shape):
    # shape of cropdepth
    shape = list(input_shape)
    assert len(shape) == 4  # only valid for 2D tensors
    shape[-1] = 13
    return tuple(shape)
