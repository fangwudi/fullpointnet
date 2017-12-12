#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "postprocess"
__author__ = 'fangwudi'
__time__ = '17-12-7 21：52'

postprocess for the model's output which is auxiliary masks
     
"""
import numpy as np
import math


class PostProcess(object):
    def __init__(self, height, width):
        """
        Initialize Eval for keypoint
        """
        # number of keypoint kind
        self.kpn = 4
        # max output object in one image
        self.maxDet = 20
        # object detect threshold, confidence
        self.obj_thr = 0.5
        # peak detect threshold, unit pixel
        self.peak_thr = 0.5
        # see threshold
        self.see_thr = 0.8
        # peak close threshold, unit pixel
        self.close_thr = 1.0
        self.height = height
        self.width = width
        # assit array
        self.x_array = np.tile(np.arange(self.width), (self.height, 1))
        self.y_array = np.tile(np.arange(self.height).reshape(-1, 1),
                               (1, self.width))

    def process(self, predict_mask):
        """for simplify, just to find 0
        :param predict_mask：[obj_mask, [mask_x, mask_y, mask_see]*keypoints]
        :return [{"keypoints": [x1,y1,v1,...,xk,yk,vk], "score": float}, ...]
        """
        predict_mask = predict_mask.astype(np.float32)
        # big mask, background will add big number
        back_mask = np.array(predict_mask[:, :, 0] < self.obj_thr).astype(np.int16)
        big_mask = np.array(back_mask * 10000).astype(np.float32)
        # find all peak
        all_peaks = []
        for i in range(self.kpn):
            # get absolute
            predict_mask[:, :, 1 + i * 3] += big_mask
            predict_mask[:, :, 2 + i * 3] += big_mask
            peaks_binary = np.logical_and(
                np.absolute(predict_mask[:, :, 1 + i * 3]) < self.peak_thr,
                np.absolute(predict_mask[:, :, 2 + i * 3]) < self.peak_thr)
            peaks_binary = np.logical_and(peaks_binary,
                                          predict_mask[:, :, 3 + i * 3] > self.see_thr)
            peaks = list(zip(tuple(np.nonzero(peaks_binary))[1],
                             tuple(np.nonzero(peaks_binary))[0]))  # note:reverse
            # delete too close points
            # Todo: haven't complete
            # use value to be score or confidence, single value:0.5~0 double should be 1~0
            # Todo: haven't complete
            # values = [predict_mask[x[1], x[0], 1 + i * 3]
            #  + predict_mask[x[1], x[0], 2 + i * 3] for x in peaks]
            # one kind keypoint group
            peak_group = []
            # turn to absolute coord
            for x in peaks:
                temp = predict_mask[x[1], x[0], 1:].copy()
                temp[0::3] = x[0] - temp[0::3]
                temp[1::3] = x[1] - temp[1::3]
                peak_group.append(Kps(temp, i))
            all_peaks.append(peak_group)
        # merge keypoint to find object
        # init first group
        done = all_peaks.pop(0)
        while all_peaks:
            todo = all_peaks.pop(0)
            wait = []
            for x in todo:
                # try to merge
                flag = True
                for i in range(len(done)-1, -1, -1):
                    z = x.merge(done[i])
                    if z:
                        flag = False
                        # merge success, add z
                        wait.append(z)
                        # and delete y
                        del done[i]
                        break
                if flag:
                    # can not merge
                    wait.append(x)
            done.extend(wait)
        # output object
        all_object = [x.output() for x in done]
        # filter max output object
        # sort detection according to object prob
        inds = np.argsort([-d['score'] for d in all_object], kind='mergesort')
        filter_object = [all_object[i] for i in inds if i < self.maxDet]
        return filter_object


class Kps(object):
    """class for store object's key point result.
    Also build for merge operation.
    """
    def __init__(self, inkp, occupy):
        """Initialize.
        :param inkp: [x1,y1,see1,...,xk,yk,seek]
        :param occupy: indicate which kp is occupied in this object
        """
        # number of keypoint kind
        self.kpn = 4
        self.kp_all = set(range(self.kpn))
        self.occupy = {occupy}
        # number of keypoint occupied
        self.n = 1
        self.x = inkp[0::3]
        self.y = inkp[1::3]
        self.see = inkp[2::3]
        # define tolerance
        self.distr = 2  # unit pixel
        self.see_tr = 0.2

    def merge(self, other):
        # merge two Kps
        # check class
        if not isinstance(other, Kps):
            raise Exception("can not merge instance other than Kps class!")
        # check occupy
        if self.occupy & other.occupy:
            # confict
            return False
        # check if they are close enough
        for i in (self.occupy | other.occupy):
            dx = abs(self.x[i] - other.x[i])
            dy = abs(self.y[i] - other.y[i])
            dsee = abs(self.see[i] - other.see[i])
            if dx > self.distr or dy > self.distr or dsee > self.see_tr:
                # not close enough
                return False
        # merge
        self.occupy = self.occupy | other.occupy
        for i in other.occupy:
            self.x[i] = other.x[i]
            self.y[i] = other.y[i]
            self.see[i] = other.see[i]
        # weight average other
        for i in (self.kp_all - self.occupy):
            self.x[i] = (self.x[i] * self.n + other.x[i] * other.n) / (
                self.n + other.n)
            self.y[i] = (self.y[i] * self.n + other.y[i] * other.n) / (
                self.n + other.n)
            self.see[i] = (self.see[i] * self.n + other.see[i] * other.n) / (
                self.n + other.n)
        # update n
        self.n = len(self.occupy)
        return self

    def output(self):
        # output {"keypoints": [x1,y1,v1,...,xk,yk,vk], "score": float}
        ky = []
        score = 0.
        for i in range(self.kpn):
            ky.append(self.x[i])
            ky.append(self.y[i])
            if i in self.occupy:
                ky.append(1)
                score += self.see[i]
            else:
                ky.append(0)
        # define own score policy
        score /= math.sqrt(self.n)
        out = {"keypoints": ky, "score": score}
        return out
