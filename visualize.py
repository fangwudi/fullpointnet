#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = "visualize"
__author__ = 'fangwudi'
__time__ = '17-12-10 10：33'

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
import matplotlib.pyplot as plt
import numpy as np


def display_images(images, titles=None, cols=3, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def display_my_masks(image, mask, cols=3):
    """Display the given image and the top few class masks."""
    display_images([image, mask[:, :, 0]], titles=['original', 'object'], cmap="Blues_r")
    to_display = []
    n = len(mask[0][0])
    # title
    titles = ['hand-x', 'hand-y',  'hand-l3',
              'shoulder-x', 'shoulder-y',  'shoulder-l3',
              'waist-x', 'waist-y',  'waist-l3',
              'foot-x', 'foot-y',  'foot-l3']
    # Generate images
    for i in range(1, n):
        m = mask[:, :, i] + 48  # turn negetive tobe positive
        to_display.append(m)
    display_images(to_display, titles=titles, cols=cols, cmap="Blues_r")
