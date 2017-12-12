#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "losses"
__author__ = 'fangwudi'
__time__ = '17-12-7 20：44'

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
from keras import backend as K


def level_loss(a=1, b=1, c=1, kpn=4):
    """define one level's loss
    :param a: weight for obj_mask
    :param b: weight for see_mask
    :param c: weight for x,y
    :param kpn: keypoint number
    :return: loss function
    """
    def _level_loss(y_true, y_pred):
        """define one level's loss
        expect input type as [batch, h, w, d], d as
        [obj-mask, x1, y1, see(some level), ... xk, yk, see]
        output loss: (1) obj mask loss; (2) several see mask loss; (3) several
        distange loss count using mean_absolute_relative_error defined after
        return total loss
        """
        ture_obj_mask = K.expand_dims(y_true[:, :, :, 0], axis = 3)
        pred_obj_mask = K.expand_dims(y_pred[:, :, :, 0], axis = 3)
        a_loss = binary_cross_entropy(ture_obj_mask, pred_obj_mask)
        true_others = y_true[:, :, :, 1:] * ture_obj_mask
        pred_others = y_pred[:, :, :, 1:] * ture_obj_mask
        b_loss = binary_cross_entropy(true_others[:, :, :, 2::3], pred_others[:, :, :, 2::3]) / K.mean(ture_obj_mask)
        c_loss = 0
        for i in range(kpn):
            see_mask = K.expand_dims(true_others[:, :, :, 2+i], axis = 3)
            true_xy = true_others[:, :, :, 0+i:2+i] * see_mask
            pred_xy = pred_others[:, :, :, 0+i:2+i] * see_mask
            c_loss += mean_absolute_relative_error(true_xy, pred_xy) / K.mean(see_mask)
        total_loss = (a*a_loss + b*b_loss + c*c_loss)/(a+b+c)
        return total_loss
    return _level_loss


def mean_absolute_relative_error(y_true, y_pred):
    # modified loss function, y_true range from -L ~ L pixel
    ts_a = 0.1  # absolute value threshold, pixel unit
    ts_b = 0.05  # relative value threshold, none unit
    epsilon = 0.25  # epision, used for 0 divide, pixel unit
    numerator = K.maximum(K.abs(y_true - y_pred) - ts_a, 0.)
    denominator = K.clip(K.abs(y_true), epsilon, None)
    diff = K.maximum(numerator / denominator - ts_b, 0.)
    return K.mean(diff)


def binary_cross_entropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))
