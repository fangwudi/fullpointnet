#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "eval_puppet"
__author__ = 'fangwudi'
__time__ = '17-12-11 13：59'

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
from keras.callbacks import Callback
from postprocess import PostProcess
from puppet_dataset_keypoint.eval import Eval


class EvalPuppet(Callback):
    def __init__(self, generator, height = 48, width = 48, batch_size=1):
        super(EvalPuppet, self).__init__()
        self.batch_size = batch_size
        self.generator = generator
        self.postprocess = PostProcess(height, width)
        self.evaluate = Eval()

    def on_epoch_end(self, epoch, logs=None):
        # generate data
        image_group, guide_mask_group, annkp_group = self.generator.next()
        predict_mask_group = self.model.predict_on_batch(image_group)[-1]
        outobjects_group = []
        for x in range(self.batch_size):
            # select last level see, and unuse other level
            mask = predict_mask_group[x, :, :, :]
            outobjects_group.append(self.postprocess.process(mask))
        self.evaluate.evaluate(annkp_group, outobjects_group)
