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
import os

from keras.layers import Input
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from FullPointNet import FullPointNet
from losses import level_loss
from puppet_dataset_keypoint.puppet import PuppetDataset
from DataGenerator import DataGenerator
from EvalPuppet import EvalPuppet
from TestDataGenerator import TestDataGenerator


train_batch_size = 64
test_batch_size = 6
height = 48
width = 48
epoch_samples = 6000
TRAINING_LOG = os.path.join('logdir', "training.csv")


def create_model():
    input_image = Input((None, None, 3))
    return FullPointNet(input_image)


def main():
    # create a generator for training data
    train_generator = DataGenerator(PuppetDataset, kpn=4,
                                    batch_size=train_batch_size,
                                    height=height, width=width)
    test_generator = TestDataGenerator(PuppetDataset, kpn=4,
                                       batch_size=test_batch_size,
                                       height=height, width=width)

    # create the model
    print('Creating model, this may take a second...')
    model = create_model()

    # first stage training
    model.compile(loss={'p1': level_loss(), 'p2': level_loss(), 'p3': level_loss()},
                  loss_weights={'p1': 0.7, 'p2': 0.2, 'p3': 0.1},
                  optimizer=adam(lr=1e-2, clipnorm=0.001))
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=epoch_samples // train_batch_size,
        epochs=20,
        callbacks=[
            ModelCheckpoint(os.path.join('savedir', 'best.h5'), monitor='loss',
                            verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1,
                              mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
            CSVLogger(TRAINING_LOG, append=True),
            EvalPuppet(test_generator, batch_size=test_batch_size)
        ]
    )
    # store final result too
    model.save(os.path.join('savedir', 'final.h5'))
