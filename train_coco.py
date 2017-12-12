#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "train"
__author__ = 'fangwudi'
__time__ = '17-12-7 20：42'

code is far away from bugs 
     ┏┓   ┏┓
    ┏┛┻━━━┛┻━┓
    ┃        ┃
    ┃ ┳┛  ┗┳ ┃
    ┃    ┻   ┃
    ┗━┓    ┏━┛
      ┃    ┗━━━━━┓
      ┃          ┣┓
      ┃          ┏┛
      ┗┓┓┏━━┳┓┏━━┛
       ┃┫┫  ┃┫┫
       ┗┻┛  ┗┻┛
with the god animal protecting
     
"""
import argparse
import os

import keras
import keras.preprocessing.image
import tensorflow as tf

import keras_retinanet.callbacks.coco
import keras_retinanet.losses
from keras_retinanet.preprocessing.coco import CocoGenerator

from FullPointNet import FullPointNet


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_model():
    input_image = keras.layers.Input((None, None, 3))
    return FullPointNet(input_image)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for COCO object detection.')
    parser.add_argument('coco_path', help='Path to COCO directory (ie. /tmp/COCO).')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).', default='imagenet')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the model
    print('Creating model, this may take a second...')
    model = create_model()

    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    # print model summary
    print(model.summary())

    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator()
    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # create a generator for training data
    train_generator = CocoGenerator(
        args.coco_path,
        'train2017',
        train_image_data_generator,
        batch_size=args.batch_size
    )

    # create a generator for testing data
    val_generator = CocoGenerator(
        args.coco_path,
        'val2017',
        val_image_data_generator,
        batch_size=args.batch_size
    )

    # start training
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=20000 // args.batch_size,  # len(train_generator.image_ids) // args.batch_size,
        epochs=50,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join('savedir', 'best.h5'), monitor='loss', verbose=1, save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
            keras_retinanet.callbacks.coco.CocoEval(val_generator),
        ],
    )

    # store final result too
    model.save(os.path.join('savedir', 'final.h5'))
