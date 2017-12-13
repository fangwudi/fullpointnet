#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "just_test"
__author__ = 'fangwudi'
__time__ = '17-12-10 10：31'

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
from puppet_dataset_keypoint.puppet import PuppetDataset
from puppet_dataset_keypoint.eval import Eval
from visualize import display_my_masks
from TestDataGenerator import TestDataGenerator
from postprocess import PostProcess


def main():
    # build dataset
    batch_size = 1
    height = 48
    width = 48
    dataset = TestDataGenerator(PuppetDataset, 4, batch_size, height=height, width=width)
    evaluate = Eval()
    postprocess = PostProcess(48, 48)
    # generate and display
    image_group, guide_mask_group, annkp_group = dataset.next()
    outobjects_group = []
    for x in range(batch_size):
        image = image_group[x]
        # select last level see, and unuse other level
        mask = guide_mask_group[x][-1]
        display_my_masks(image, mask)
        # use groudtruth mask directly as predict mask to process
        outobjects_group.append(postprocess.process(mask))
    evaluate.evaluate(annkp_group, outobjects_group)


def main2():
    gtAll = [[{'size': 1, 'keypoints': [46, -2, 0, 38, -5, 0, 39, 4, 1, 35, 10, 0]},
              {'size': 2, 'keypoints': [27, 21, 1, 9, 20, 1, 6, 38, 1, 6, 52, 0]},
              {'size': 2, 'keypoints': [10, -5, 0, 28, -5, 0, 34, 12, 1, 31, 26, 1]}]]
    dtAll = [[{'score': 1.7320508075688774, 'keypoints': [27.0, 21.0, 1, 9.0, 20.0, 1, 6.0, 38.0, 1, 14.0, 26.333334, 0]},
              {'score': 1.4142135623730949, 'keypoints': [32.5, 19.0, 0, 32.5, 19.0, 0, 34.0, 12.0, 1, 31.0, 26.0, 1]},
              {'score': 1.0, 'keypoints': [39.0, 4.0, 0, 39.0, 4.0, 0, 39.0, 4.0, 1, 39.0, 4.0, 0]}]]
    evaluate = Eval()
    evaluate.evaluate(gtAll, dtAll)
