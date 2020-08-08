#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths_tf_fasterrcnn
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2
import argparse
import freenect

import time


from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import scipy
#from shapely.geometry import Polygon


frame_current=[]
CLASSES = ('__background__', 'bowl', 'cup', 'hammer', 'knife', 'ladle', 'mallet', 'mug', 'pot', 'saw', 'scissors','scoop','shears','shovel','spoon','tenderizer','trowel','turner','computer_mouse','fork','lipstick','pen','pizza_cutter','plate','remote_control','screwdriver','spoon','stapler','wrench')


NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res50': ('res50_faster_rcnn_iter_120000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'grasp': ('train',)}

PAIRED_WITH_AFFNET = True
USE_CPU = True
CONF_THRESH = 0.5

# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


# function to get depth image from kinect
def get_depth():
    array_raw, _ = freenect.sync_get_depth()
    array = array_raw.astype(np.uint8)
    return array, array_raw


def demo_process(sess, net):

      if frame_current != []:
          im_org = frame_current.copy()
          im = frame_current.copy()
          tmp_g = im[:, :, 1]
          im[:, :, 1] = im[:, :, 2]
          im[:, :, 2] = tmp_g
          img = im.astype('uint8')

          # Detect all object classes and regress object bounds
          timer = Timer()
          timer.tic()
          scores, boxes = im_detect(sess, net, im)
          timer.toc()
          print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

          # Visualize detections for each class

          NMS_THRESH = 0.3
          f = open("tmp_data/agnostic_objdetection/list_class_center.txt", "w")
          for cls_ind, cls in enumerate(CLASSES[1:]):
              cls_ind += 1 # because we skipped background
              cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
              cls_scores = scores[:, cls_ind]
              dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
              keep = nms(dets, NMS_THRESH)
              dets = dets[keep, :]

              # keep bbox above thresh
              thresh = CONF_THRESH
              inds = np.where(dets[:, -1] >= thresh)[0]
              if len(inds) == 0:
                  continue
              #im = im[:, :, (2, 1, 0)]
              for i in inds:
                  bbox = dets[i, :4]
                  score = dets[i, -1]
                  color = (255, 0, 0)
                  thickness = 2
                  im_org = cv2.rectangle(im_org, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

                  font = cv2.FONT_HERSHEY_SIMPLEX
                  fontScale = 0.5
                  thickness_text = 1
                  im_org = cv2.putText(im_org, cls + ' ' + str(score), (int(bbox[0]), int(bbox[1] - 3)), font,
                                      fontScale, color, thickness_text, cv2.LINE_AA)

                  # write to file
                  f.write(cls+' ')
                  f.write(str((bbox[0] + bbox[2]) / 2.0)+' ')
                  f.write(str((bbox[1] + bbox[3]) / 2.0))
                  f.write('\n')
          return im_org





if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    demonet = 'res50'
    dataset = 'pascal_voc'

    tfmodel = '/home/landan/tf-fasterrcnn/output/res50/voc_2007_trainval/default/res50_faster_rcnn_iter_120000.ckpt'
    # set config
    if USE_CPU:# with CPU
        tfconfig = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})
    else: # with GPU
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    #tfconfig = tf.ConfigProto(device_count={'GPU': 0})

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'res50':
        net = resnetv1(batch_size=1, num_layers=50)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 29,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))



    while 1:
        if not PAIRED_WITH_AFFNET:
            # get a frame from RGB camera
            frame_current = get_video()
            # get a frame from depth sensor
            depth, depth_raw = get_depth()

        else:
            frame_current = np.load('tmp_data/agnostic_objdetection/rgb.npy')
            depth = np.load('tmp_data/agnostic_objdetection/depth.npy')

        # display RGB image
        cv2.imshow('RGB image 2', frame_current)
        # display depth image
        cv2.imshow('Depth image 2', depth)

        im = demo_process(sess,net)

        cv2.imshow('detection', im)
        cv2.waitKey(1)





