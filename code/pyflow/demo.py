# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow


def calculateFlow(img_prev, img_next):


    im1 = np.array(Image.open('examples/car1.jpg'))
    im2 = np.array(Image.open('examples/car2.jpg'))
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

# Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    return flow

