# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

import cv2
import os.path as osp
import cPickle
import time
DEBUG = False


## VISUALIZE
background = [200, 222, 250]  #Light Sky Blue
c1 = [0,0,205]   #ok
c2 = [34,139,34]         #ok
c3 = [192,192,128]   #3
c4 = [165,42,42]    #ok
c5 = [128,64,128]    #5
c6 = [204,102,0]   #6
c7 = [184,134,11] #ok
c8 = [0,153,153] #ok
c9 = [0,134,141] #ok
c10 = [184,0,141] #ok


label_colours = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])    


out_verbose = 0
verbose_showim = 0

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4) #(x1, y1, x2, y2, cls)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4)

        if cfg.TRAIN.MASK_REG:
            ###################### mask targets###############
            #top[5].reshape(1, self._num_classes * cfg.TRAIN.MASK_SIZE * cfg.TRAIN.MASK_SIZE) #cfg.TRAIN.MASK_SIZE = 14 or 28
            #top[5].reshape(1, 1 * cfg.TRAIN.MASK_SIZE * cfg.TRAIN.MASK_SIZE) #Class-agnostic mask.
            top[5].reshape(1, cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE) #segmentation mask for positive rois
            ####incase output rois_pos
            top[6].reshape(1, 5) #positive rois for mask branch
            ##################################################

    def forward(self, bottom, top):
        
        #if out_verbose: print '=========================================================================='
        
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported' #first element in each roi is image index (in this mini batch)

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images #cfg.TRAIN.BATCH_SIZE = 128; num_images = 1
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image) #cfg.TRAIN.FG_FRACTION = 0.25 ==> fg_rois_per_image = 0.25*128 = 32

        # Sample rois with classification labels and bounding box regression
        # targets

        # labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
        #     all_rois, gt_boxes, fg_rois_per_image,
        #     rois_per_image, self._num_classes)
        labels, rois, bbox_targets, bbox_inside_weights, bbox_targets_oris, rois_pos, gt_assignment_pos = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes) #bbox_targets_oris: original gt of rois
            #gt_assignment_pos: array contains the index gt_boxe corresponding to rois_pos,
            # i.e.,  gt_assignment_pos[ix] = k ---> the gt box of rois_pos[ix] is gt_boxes[k], and the index of segmentation mask is seg_mask_inds[k]
            #NOTE: rois_pos is at begining of rois
        ################ Print rois, bb_targets and bb_targets_ori#######################
        # f = open("/home/tdo/Software/FRCN_ROOT/tools/proposal_target_layer_forward.txt", "w")
        # for ix, roi in enumerate(rois):
        #     roi = rois[ix]
        #     bbox_target = bbox_targets[ix]
        #     bbox_targets_ori = bbox_targets_oris[ix]
        #     # print roi
        #     f.write("roi: ")
        #     for i in xrange(len(roi)):
        #         f.write(str(roi[i]) + " ")
        #     f.write("\n")
        #     # print bbox_target
        #     f.write("bbox_target: ")
        #     for i in xrange(len(bbox_target)):
        #         f.write(str(bbox_target[i]) + " ")
        #     f.write("\n")
        #     # print bbox_targets_ori
        #     f.write("bbox_target_ori: ")
        #     for i in xrange(len(bbox_targets_ori)):
        #         f.write(str(bbox_targets_ori[i]) + " ")
        #     f.write("\n\n")
        # f.close()
        ##################################################################################
        if cfg.TRAIN.MASK_REG:

            
            num_roi = len(rois_pos) #incase output rois_pos (so only need to define mask to rois_pos)
            #num_roi = len(rois)

            if out_verbose: print ("number of rois in the image: " + str(num_roi))
            #print ("size of rois_pos: " + str(rois_pos.shape[0]) + " " + str(rois_pos.shape[1]))
            if out_verbose: print 'rois pos shape: ', rois_pos.shape
            if out_verbose: print 'rois pos: ', rois_pos
            if out_verbose: print "gt_assignment_pos shape: ", gt_assignment_pos.shape
            if out_verbose: print "gt_assignment_pos: ", gt_assignment_pos

            #mask_targets = np.zeros((num_roi, cfg.TRAIN.MASK_SIZE * cfg.TRAIN.MASK_SIZE), dtype=np.float32)
            #mask_targets = -1*np.ones((num_roi, cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE), dtype=np.float32)
            mask_targets = -1 * np.ones((num_roi, cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE), dtype=np.float32)
            im_info = bottom[2].data
            seg_mask_inds = bottom[3].data
            flipped = bottom[4].data

            im_ind = seg_mask_inds[0][0]

            im_scale = im_info[0][2]
            #crop_offset = im_info[0][2]
            #offset_h = im_info[0][2]
            #offset_w = im_info[0][3]
            
            im_height = im_info[0][0]
            im_width = im_info[0][1]
            flipped = flipped[0][0]

            if out_verbose: print 'im_ind: ', im_ind, '- im height: ', im_height, '-- im_width: ', im_width
            
            #################################
            ### MASK PART
            
            # read all segmentation masks of this image from hard disk
            mask_ims = []
            mask_flipped_ims = []
            count = 0
            while True:  ## load mask
                count += 1
                #seg_mask_path = './data/cache/seg_mask_coco_gt/' + str(int(im_ind)) + '_' + str(int(count)) + '_segmask.sm'
                if cfg.TRAIN.TRAINING_DATA == 'coco_2014_train':
                    seg_mask_path = './data/cache/'+'GTsegmask_'+cfg.TRAIN.TRAINING_DATA+'/' + str(int(im_ind)) + '_' + str(int(count)) + '_segmask.sm'
                elif cfg.TRAIN.TRAINING_DATA == 'VOC_2012_train':
                    im_ind = int(im_ind) ## Not use
                    #print (im_ind)
                    t = str(int(im_ind)) #i.e. t = '8000008'
                    p = t[-6:] #p = '000008'
                    p2 = t[0:len(t) - len(p)] #p2 = '8'
                    if len(p2) == 1:
                        p2 = '200' + p2 #p2 = '2008' or '2009'
                    if len(p2) == 2:
                        p2 = '20' + p2 #p2 = '2010' or '2011' or '2012'
                    # print ("==================================im_ind:" + str(int(im_ind)))
                    # print ("==================================p:" + str(p))
                    # print ("==================================p2:" + str(p2))
                     
                    # FOR PASCAL
                    #seg_mask_path = './data/cache/' + 'GTsegmask_' + cfg.TRAIN.TRAINING_DATA + '/' + str(p2) + '_' + str(p) + '_' + str(int(count)) + '_segmask.sm'
                     
                    # FOR IIT-AFF dataset
                    seg_mask_path = './data/cache/' + 'GTsegmask_' + cfg.TRAIN.TRAINING_DATA + '/' + t + '_' + str(int(count)) + '_segmask.sm'
                    
                else:
                    print ("lib/rpn/proposal_target_layer.py: DO NOT KNOW TRAINING DATASET.")
 
                if osp.exists(seg_mask_path):
 
                    #print ("seg path: " + seg_mask_path)
 
                    with open(seg_mask_path, 'rb') as f:
                        mask_im = cPickle.load(f)
                    uni_ids = np.unique(mask_im)
                    org_uni_label = np.unique(mask_im)
                    
                    mask_im = (mask_im).astype('float32')
                    mask_im = cv2.resize(mask_im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                   
                    mask_im = _convert_mask_to_original_ids_manual_FOR_IMAGE(mask_im, org_uni_label)
                    mask_im = np.asarray(mask_im) #convert mask_im to np array
                       
                    ## Anh --> fix floating base
                    mask_im = np.around(mask_im, decimals=0)  # round 1.6 -> 2., 1.1 -> 1.
                    mask_im = mask_im.astype('uint8') # 2. --> 2
                    uni_res = np.unique(mask_im)
                    
                    mask_ims.append(mask_im)
                    ###
                    #mask_flipped_im = cv2.flip(mask_im, 0)
                    mask_flipped_im = cv2.flip(mask_im, 1) #vertical flip
                    mask_flipped_ims.append(mask_flipped_im)
                else:
                    break
 
            # create flag array to check if bbox is all zero or not
            flag_arr = np.full((rois_pos.shape[0], ), 1, 'uint8') # (shape = (num_bbox, 1)) # fill with 1: default is good, if not good, set to 0 
            
            
            #for ix, roi in enumerate(rois):
            for ix, roi in enumerate(rois_pos):
                #print 'ix of rois_pos: ', ix
                
                '''
                gt_mask_ind = -1
                roi = rois[ix]
                bbox_target = bbox_targets[ix]
                bbox_targets_ori = bbox_targets_oris[ix]
 
                x1_t = bbox_targets_ori[0]
                y1_t = bbox_targets_ori[1]
                x2_t = bbox_targets_ori[2]
                y2_t = bbox_targets_ori[3]
                cls = bbox_targets_ori[4] #gt class of this box
                #find index of this bbox_targets_ori in list of ground truth boxes
                for ix, gt_box in enumerate(gt_boxes):
                    if round(gt_box[0]) == round(x1_t) and round(gt_box[1]) == round(y1_t) and \
                                    round(gt_box[2]) == round(x2_t) and round(gt_box[3]) == round(y2_t):
                        gt_mask_ind = seg_mask_inds[ix][1] - 1
                        # print ("GT: " + str(gt_box[0]) + " " + str(gt_box[1]) + " " + str(gt_box[2]) + " " + str(gt_box[3]) + " " \
                        #        + "Target ori:" + str(x1_t) + " " + str(y1_t) + " " + str(x2_t) + " " + str(y2_t))
                        break
                '''
                k = gt_assignment_pos[ix]
                gt_mask_ind = int(seg_mask_inds[k][1]) - 1# index of seg mask. minus 1 because the stored mask index in seg_mask_inds is 1-based
                gt_box = gt_boxes[k]
                #roi coordinate
                x1 = round(roi[1])
                y1 = round(roi[2])
                x2 = round(roi[3])
                y2 = round(roi[4])
                
                ## Anh them vo!
                if x1 == y1 == x2 == y2 == 0:
                    # set flag array [ix] to 0 : found bad bbox
                    flag_arr[ix] = 0
                    
                
                # TOAN ba dao code!
                x1 = np.min((im_width - 1, np.max((0, x1))))
                y1 = np.min((im_height - 1, np.max((0, y1))))
                x2 = np.min((im_width - 1, np.max((0, x2))))
                y2 = np.min((im_height - 1, np.max((0, y2))))
                w = (x2 - x1) + 1
                h = (y2 - y1) + 1
                #gt_roi coordinate
                x1t = round(gt_box[0])
                y1t = round(gt_box[1])
                x2t = round(gt_box[2])
                y2t = round(gt_box[3])
                # sanity check
                x1t = np.min((im_width - 1, np.max((0, x1t))))
                y1t = np.min((im_height - 1, np.max((0, y1t))))
                x2t = np.min((im_width - 1, np.max((0, x2t))))
                y2t = np.min((im_height - 1, np.max((0, y2t))))
                
                
#                 # Anh them vo
#                 w = (x2 - x1)
#                 h = (y2 - y1)
#                 #gt_roi coordinate
#                 x1t = round(gt_box[0])
#                 y1t = round(gt_box[1])
#                 x2t = round(gt_box[2])
#                 y2t = round(gt_box[3])

                
                cls = gt_box[4] #gt class of this box
 
                is_pos_roi = True
                if is_pos_roi:
                    # for each positive rois, create a t_mask having same size as rois size (and fill all values by -1)
                    #roi_mask = -1 * np.ones((h, w), dtype=np.float32)
                    roi_mask = -1 * np.ones((int(h), int(w)), dtype=np.float32)#FC  

                    if flipped:
                        # print ("==================================im_ind:" + str(p) + '_' + str(p2))
                        # print ("==================================gt_mask_ind:" + str(gt_mask_ind))
                        gt_mask = mask_flipped_ims[gt_mask_ind]
                    else:
                        #print 'gt mask ind: ', gt_mask_ind
                        gt_mask = mask_ims[gt_mask_ind]
 
                    uni_ids = np.unique(gt_mask)
                    #print(uni_ids)

                    #print '-------- list id in roi mask: ', uni_ids
                
                    # compute overlap between roi coordinate and gt_roi coordinate
                    x1o = max(x1, x1t)
                    x2o = min(x2, x2t)
                    y1o = max(y1, y1t)
                    y2o = min(y2, y2t)
 
 
                    # print ("=========size of gt_mask: " + str(gt_mask.shape))
                    # print ("x1,y1,x2,y2:" + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2)) #roi
                    # print ("x1t,y1t,x2t,y2t:" + str(x1t) + " " + str(y1t) + " " + str(x2t) + " " + str(y2t))#gt roi
                    # print ("x1o,y1o,x2o,y2o:" + str(x1o) + " " + str(y1o) + " " + str(x2o) + " " + str(y2o))#overlap
                    # print("x2o-x1o, y2o-y1o: " + str(x2o-x1o) + " " + str(y2o-y1o))
 
                    mask_overlap = np.zeros((int(y2o-y1o), int(x2o-x1o)), dtype=np.float32)
 
 
                    #color_img = cv2.cvtColor((gt_mask*255), cv2.cv.CV_GRAY2RGB)
                    color_img = label_colours.take(gt_mask, axis=0).astype('uint8')
                    #if out_verbose: print 'color img shape: ', color_img.shape
                    cv2.rectangle(color_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 8) #plot roi with Red
                    cv2.rectangle(color_img, (int(x1t), int(y1t)), (int(x2t), int(y2t)), (255, 0, 0), 4)  # plot gt with Blue
                    cv2.rectangle(color_img, (int(x1o), int(y1o)), (int(x2o), int(y2o)), (0, 255, 0), 2)  # plot overlap with Green
                    mask_overlap_draw = color_img[int(y1o):int(y2o), int(x1o):int(x2o), :]
                    
                    if verbose_showim:
                        cv2.imshow("color_img", color_img)
                        cv2.imshow("overlap", mask_overlap_draw)
                        cv2.waitKey(0)
                    
 
                    mask_overlap[:, :] = gt_mask[int(y1o):int(y2o), int(x1o):int(x2o)]        
                    
                
                if roi_mask.shape[0] > 3 and roi_mask.shape[1] > 3:  #only resize if shape != (0, 0)
                
                    roi_mask[int(y1o-y1):int(y2o-y1), int(x1o-x1):int(x2o-x1)] = mask_overlap
            
                    original_uni_ids = np.unique(roi_mask)
            
                    roi_mask = cv2.resize(roi_mask.astype('float'), (cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE), interpolation=cv2.INTER_LINEAR)
                    
                    roi_mask = _convert_mask_to_original_ids_manual(roi_mask, original_uni_ids)
                                
                    if verbose_showim:
                        #color_roi_mask = cv2.cvtColor((roi_mask*255), cv2.cv.CV_GRAY2RGB)
                        color_roi_mask = label_colours.take(roi_mask.astype('int8'), axis=0).astype('int8')
                        cv2.imshow('roi_mask', color_roi_mask)
                        cv2.waitKey(0)
                        
                else:
                    roi_mask = -1 * np.ones((cfg.TRAIN.MASK_SIZE, cfg.TRAIN.MASK_SIZE), dtype=np.float32) # set roi_mask to -1 
               
                
                mask_targets[ix, :, :] = roi_mask

            
        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))


        ## DEBUG
        
        if out_verbose: print '--- rois shape        : ', rois.shape
        if out_verbose: print '--- labels shape      : ', labels.shape
        if out_verbose: print '--- bbox_targets shape: ', bbox_targets.shape
        if out_verbose: print '--- bbox ins wei shape: ', bbox_inside_weights.shape
        if out_verbose: print '--- mask targets shape: ', mask_targets.shape
        if out_verbose: print '--- rois_pos shape    : ', rois_pos.shape
#         if out_verbose: print '--- new mask target shape: ', new_mask_targets.shape
#         if out_verbose: print '--- new rois pos shape   : ', new_rois_pos.shape
#         #if out_verbose: print '- new mask target: ', new_mask_targets
#         if out_verbose: print '- new rois pos   : ', new_rois_pos
        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        if cfg.TRAIN.MASK_REG:
#             ################### Toan mask target##########################
            top[5].reshape(*mask_targets.shape)
            top[5].data[...] = mask_targets
            ####incase output rois_pos
            top[6].reshape(*rois_pos.shape)
            top[6].data[...] = rois_pos

#             ################### Anh mask target##########################
#             top[5].reshape(*new_mask_targets.shape)  ## ONLY FEED IF BBOX IS GOOD (NOT ALL ZERO)
#             top[5].data[...] = new_mask_targets
#             ####incase output rois_pos
#             top[6].reshape(*new_rois_pos.shape)
#             top[6].data[...] = new_rois_pos

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



def _reset_mask_ids_FOR_IMAGE(mask, before_uni_ids):  
    # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later 
#     if -1 in before_uni_ids:
#         counter = -1
#     else:
#         counter = 0

    counter = 0  # INDEX START AT 0
            
    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1
        
        
    return mask
    
    
def _reset_mask_ids(mask, before_uni_ids):  
    # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later 
#     if -1 in before_uni_ids:
#         counter = -1
#     else:
#         counter = 0

    counter = -1
            
    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1
        
        
    return mask
    


def _convert_mask_to_original_ids_manual_FOR_IMAGE(mask, original_uni_ids):

    const = 0.005
    out_mask = original_uni_ids[(np.abs(mask - original_uni_ids[:,None,None]) < const).argmax(0)]
     
    return out_mask
    
def _convert_mask_to_original_ids_manual(mask, original_uni_ids):
    
    const = 0.005
    out_mask = original_uni_ids[(np.abs(mask - original_uni_ids[:,None,None]) < const).argmax(0)]
     
    
    #return mask
    return out_mask

def _set_unwanted_label_to_zero(mask, before_uni_label):
    # 1. round mask
    # 2. find unique mask value
    # 3. remove based on set different

    mask = np.around(mask, decimals=0)  # round 1.6 -> 2., 1.1 -> 1.    
    #mask = mask.astype('int') # 2. --> 2
    
    uni_mask_values = np.unique(mask)
    
    unwanted_labels = list(set(uni_mask_values).symmetric_difference(set(before_uni_label)))
    if out_verbose: print '--- unwanted labels: ', unwanted_labels
                           
    for ul in unwanted_labels:
        mask[mask == ul] = 0 ## set value of unwanted label to zeo

    return mask


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    # print ("===============class size: " + str(clss))
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32) #clss.size = 128 ---> bbox_targets = 128 * 84, moi roi la 1*84 dimesion
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        start = int(start)
        end = int(end)
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:] #gan gia tri tai class tuong ung la bbox_target_data, con lai la so 0
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # fg_rois_per_image = 32
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)

    # print ("===================all_rois_len: " + str(len(all_rois)) + ". gt_assignment len: " + str(len(gt_assignment)))
    # print ("gt_assignment: ")
    # print gt_assignment

    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_this_image):] = 0
    rois = all_rois[keep_inds]
    # positive rois
    rois_pos = np.zeros((fg_inds.size, 5), dtype=np.float32) #because return rois_pos as top ---> allocate memory for it
    rois_pos[:, :] = all_rois[fg_inds]
    gt_assignment_pos = gt_assignment[fg_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    #return labels, rois, bbox_targets, bbox_inside_weights
    return labels, rois, bbox_targets, bbox_inside_weights, gt_boxes[gt_assignment[keep_inds], :], rois_pos, gt_assignment_pos #[return them gt_boxes: original coordinate and class of gt ]
