#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "DataGenerator"
__author__ = 'fangwudi'
__time__ = '17-12-7 20：43'

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
import threading
import numpy as np


class DataGenerator(object):
    """Generates guide masks needed from given object dataset generator.
    1 object has n keypoints. Each  kind of keypoints will have 3 correspoding
    masks.Beside 3n masks, there's an additional mask indicating object out of
    backgroud.
    use next() function to generate a batch.
    source dataset should give:
        (1) image: image for train
        (2) mask: mask array for each object in 1 image
        (3) annkp: {"keypoints": [x1, y1, v1, ..., xk, yk, vk]}
    middile:
        (1) object mask: indicate pixel is in object; int
        (2) x coord mask: one keypoint relative distance from here in x; float
        (3) y coord mask: one keypoint relative distance from here in y; float
        (4) seen mask: indicate here can see the keypoint; int
            repeat for each level
        (5) repeat (2)(3)(4) for each kind of keypoint
    return level_mask: object_mask, x, y, see_level, repeat...
    """
    def __init__(self, source_generator, kpn, batch_size, height=48, width=48):
        # source_generator should have batch_size argument
        self.dataset = source_generator(batch_size, height, width,)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # number of keypoint kind
        self.kpn = kpn
        # number of levels, here 3, level1, level2, level3
        self.levelnum = 3
        # each level receptive area, for simlify, just use half square side
        self.level_recepside = [12, 24, 48]
        # level index for output
        self.levelindex = [[0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18],
                           [0, 1, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19],
                           [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20]]
        # assit array
        self.x_array = np.tile(np.arange(self.width), (self.height, 1))
        self.y_array = np.tile(np.arange(self.height).reshape(-1, 1), (1, self.width))
        # next threading
        self.batch_index = 0
        self.lock = threading.Lock()

    def produce(self, mask_group, annkp_group):
        guide_mask_l1, guide_mask_l2, guide_mask_l3 = [], [], []
        for i in range(self.batch_size):
            l1, l2, l3 = self.produce_one(mask_group[i], annkp_group[i])
            guide_mask_l1.append(l1)
            guide_mask_l2.append(l2)
            guide_mask_l3.append(l3)
        return [np.array(guide_mask_l1), np.array(guide_mask_l2),
                np.array(guide_mask_l3)]

    def produce_one(self, mask, annkp):
        # expect mask as (height, width, obn)
        # obn: object number
        obn = mask.shape[2]
        # output guide_mask shape as (height, width, 1+kpn*(2+levelnum))
        guide_mask = np.zeros([self.height, self.width, 1+self.kpn*(2+self.levelnum)], dtype=np.int16)
        for i in range(obn):
            # mask for any object
            guide_mask[:, :, 0] = np.logical_or(guide_mask[:, :, 0], mask[:, :, i]).astype(np.int16)
            for j in range(self.kpn):
                if annkp[i]['keypoints'][2+j*3] == 1:
                    # pixel mask x = self_x - target
                    xtemp = mask[:, :, i] * (self.x_array - annkp[i]['keypoints'][j*3])
                    guide_mask[:, :, 1+j*(2+self.levelnum)] += xtemp
                    # pixel mask y = self_x - target
                    ytemp = mask[:, :, i] * (self.y_array - annkp[i]['keypoints'][1+j*3])
                    guide_mask[:, :, 2+j*(2+self.levelnum)] += ytemp
                    xtemp = np.abs(xtemp)
                    ytemp = np.abs(ytemp)
                    for z in range(self.levelnum):
                        mask_temp = np.logical_and(xtemp <= self.level_recepside[z],
                                                   ytemp <= self.level_recepside[z]).astype(np.int16)
                        guide_mask[:, :, 3 + z + j * (2 + self.levelnum)] += mask[:, :, i] * mask_temp
        # spilit to each level
        output_mask = []
        for i in range(self.levelnum):
            output_mask.append(guide_mask[:, :, self.levelindex[i]].copy())
        return tuple(output_mask)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # acquire/release the lock when updating self.value
        with self.lock:
            self.batch_index += 1
        image_group, mask_group, annkp_group = self.dataset.next()
        guide_mask_group = self.produce(mask_group, annkp_group)
        return np.array(image_group), guide_mask_group
