"""
See README.md for installation instructions before running.
Demo script to perform affordace detection from images
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect2
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import numpy as np
import os, cv2
import argparse

import caffe

import threading
import socket
import struct
import time
import matplotlib.pyplot as plt
import cv2.aruco as aruco
import math
import freenect

frame_current=[]
depth_raw = []
#CONF_THRESHOLD = 0.17
CONF_THRESHOLD = 0.9
good_range = 0.005
    
# get current dir
cwd = os.getcwd()
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
print 'AffordanceNet root folder: ', root_path
img_folder = cwd + '/img_UMD'

OBJ_CLASSES = ('__background__', 'bowl', 'cup', 'hammer', 'knife', 'ladle', 'mallet', 'mug', 'pot', 'saw', 'scissors','scoop','shears','shovel','spoon','tenderizer','trowel','turner')

# Mask
background = [200, 222, 250]  
c1 = [0,0,205] #grasp red  
c2 = [34,139,34] #cut green
c3 = [0,255,255] #scoop bluegreen 
c4 = [165,42,42] #contain dark blue   
c5 = [128,64,128] #pound purple 
c6 = [51,153,255] #support orange
c7 = [184,134,11] #wrap-grasp light blue
c8 = [0,153,153]
c9 = [0,134,141]
c10 = [184,0,141] 
c11 = [184,134,0] 
c12 = [184,134,223]
label_colours = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])

# Object
col0 = [0, 0, 0]
col1 = [0, 255, 255]
col2 = [255, 0, 255]
col3 = [0, 125, 255]
col4 = [55, 125, 0]
col5 = [255, 50, 75]
col6 = [100, 100, 50]
col7 = [25, 234, 54]
col8 = [156, 65, 15]
col9 = [215, 25, 155]
col10 = [25, 25, 155]
col11 = [100, 100, 50]#
col12 = [25, 234, 54]#
col13 = [156, 65, 15]#
col14 = [215, 25, 155]#
col15 = [25, 25, 155]#
col16 = [100, 100, 50]#
col17 = [184,134,11]

col_map = [col0, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17]


# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


# function to get depth image from kinect
def get_depth():
    #array_raw, _ = freenect.sync_get_depth()
    array_raw, _ = freenect.sync_get_depth(format=4)
    array = array_raw.astype(np.uint8)
    return array, array_raw


def depth2height_wrt_aruco(Z, ii, cameraMatrix, M_CL):
    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]
    cx = cameraMatrix[0, 2]
    cy = cameraMatrix[1, 2]
    u = ii[1] # 0-639
    v = ii[0] # 0-479
    #IGNORE = 2047*0.001
    IGNORE = 0 * 0.001

    #Z = (Z*0+850)*0.001
    Z = Z*0.001
    X = (u*Z-cx*Z)/fx
    Y = (v*Z-cy*Z)/fy
    coor_C = np.vstack((X, Y, Z, 0*X+1))

    #num_left =  np.array(len(np.where(coor_C[2] != IGNORE)))
    coor_C = np.squeeze(coor_C[:,np.where(coor_C[2] != IGNORE)])

    #print 'X in cam = ' + str(np.mean(coor_C[0]))
    #print 'Y in cam = ' + str(np.mean(coor_C[1]))
    #print 'Z in cam = ' + str(np.mean(coor_C[2]))
    coor_L = np.dot(np.linalg.inv(M_CL), coor_C)
    #print coor_L
    #print 'max Z = '
    #print np.max(coor_L[2])

    #print 'mean X = '
    #print np.mean(coor_L[0])
    #print 'mean Y = '
    #print np.mean(coor_L[1])
    #print np.mean(coor_L[2])
    #print 'M_CL = '
    #print M_CL
    if len(coor_L.shape) == 1 or (coor_L.shape[0] == 4 and coor_L.shape[1] == 0):
        return 0, 0, 0
    #else:
    return np.mean(coor_L[0]), np.mean(coor_L[1]), np.max(coor_L[2])



def compute_imgRot(frame):
    # aim to find a ARUCO marker and compute the camera rotation on XY plane

    # find ARUCO marker
    #gray = frame * 255
    gray = frame
    # print(gray)
    gray = gray.astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()  #P_reshape = np.reshape(P, (7*7*2*2))

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(ids)

    angle = 0.0
    if ids is not None:
        point0 = corners[0][0][0]
        point3 = corners[0][0][3]
        #if point0[1] >= point3[1]: # rotate counter-clock wise
        angle = np.arctan((point0[1] - point3[1]) / (point3[0] - point0[0]))
        print('counter-clock rotate angle in degree: (just added to detected degree to compensate the table)')
        print(angle/3.14*180)
        return angle/3.14*180
        # else: # rotate clock wise
        #     angle = np.arctan((point3[1] - point0[1]) / (point3[0] - point0[0]))
        #     print('clock')
        #     print(angle/3.14*180)
    else:
         return angle

def coordinate_img2table(frame, u, v, rot, depths, ii, affordance_id):
    """project found u v coordinate on image to x y coordinate on table with ARUCO marker."""

    # camera matrix
    markerLength = 0.093
    # markerLength = 0.06

    # old camera matrix used by Yufeng
    #cameraMatrix = np.array([[297.47608, 0.0, 320], [0.0, 297.14815, 240], [0.0, 0.0, 1.0]])  # camera frame from socket
    #distCoeffs = np.array([0.15190073, -0.8267655, 0.00985276, -0.00435892, 1.58437205])  # fake value

    # new camera matrix obtained at 01/2018
    # cameraMatrix = np.array([[592.90077, 0.0, 327.06503], [0.0, 591.07515, 239.40367], [0.0, 0.0, 1.0]])  # camera frame from socket
    # distCoeffs = np.array([-0.02067,   0.06351,   -0.00285,   0.00083, 0.00000])  # fake value

    cameraMatrix = np.array([[526.37013657, 0.00000000, 313.68782938], [0.00000000, 526.37013657, 259.01834898], [0.00000000, 0.00000000, 1.00000000]])
    distCoeffs = np.array([-0.02067, 0.06351, -0.00285, 0.00083, 0.00000])

    # find ARUCO marker
    #gray = frame*255
    gray = frame
    #print(gray)
    gray = gray.astype(np.uint8)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    #print (gray)
    #gray = gray.astype(np.uint8)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(ids)


    dst = 0
    tvec = 0
    T = 0
    point_new = np.zeros((3, 1))
    mean_X_L, mean_Y_L, max_Z_L = 0, 0, 0
    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], markerLength, cameraMatrix, distCoeffs)
        frame = aruco.drawDetectedMarkers(frame, corners)
        aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, markerLength)
        cv2.imshow('frame_aruco', frame)
        cv2.waitKey(1)


        dst, jacobian = cv2.Rodrigues(rvec)
        T = np.zeros((4, 4))
        T[:3, :3] = dst
        T[:3, 3] = tvec
        T[3, :] = np.array([0, 0, 0, 1])

        # for contain, you need the height:
        mean_X_L, mean_Y_L, max_Z_L = depth2height_wrt_aruco(depths, ii, cameraMatrix, T)

        # projection
        imagePts = np.array([u, v, 1])
        normal_old = np.array([0, 0, 1])
        ray_center = np.array([0, 0, 0])

        if affordance_id == 4 or affordance_id == 6:
            distance_old = max_Z_L-0.0178
        else:
            distance_old = 0

        normal_new = np.dot(dst, normal_old) #(3,)
        normal_new = np.expand_dims(normal_new, 1)  #(3,1)
        translation_old = tvec #(1,1,3)
        translation_old = np.squeeze(translation_old, 0) #(1,3)
        distance_new = -(distance_old + np.dot(translation_old, normal_new))

        ray = np.dot(np.linalg.inv(cameraMatrix), imagePts) #(3,)
        t = -(np.dot(normal_new.transpose(), ray_center) + distance_new) / np.dot(normal_new.transpose(), ray) # (1,1)
        intersection = np.multiply(ray, t) #(1,3)
        intersection_homo = np.array([intersection[0,0], intersection[0,1], intersection[0,2], 1])

        point_new = np.dot(np.linalg.inv(T), intersection_homo)

    if affordance_id == 4 or affordance_id == 6:
        # print from kinect depth
        #print(point_new[0] )#0.3
        #print(point_new[1] )#0.4#
        #print 'X = '
        #print(mean_X_L + 0.30)#0.3
        #print 'Y = '
        #print(mean_Y_L + 0.30)#0.4#

        # print from aurco with predicted depth and project again
        print 'X in aruco = '
        print(point_new[0] + 0.30)#0.3
        print 'Y in aruco = '
        print(point_new[1] + 0.32)#0.4#
        print 'Z = '
        print(max_Z_L - 0.0178 - 0.045)
    else:
        print 'X = '
        print(point_new[0] + 0.30)#0.3
        print 'Y = '
        print(point_new[1] + 0.32)#0.4#
        print 'rotation = '
        print(rot)




def reset_mask_ids(mask, before_uni_ids):
    # reset ID mask values from [0, 1, 4] to [0, 1, 2] to resize later 
    counter = 0
    for id in before_uni_ids:
        mask[mask == id] = counter
        counter += 1
        
    return mask
    

    
def convert_mask_to_original_ids_manual(mask, original_uni_ids):
    #TODO: speed up!!!
    temp_mask = np.copy(mask) # create temp mask to do np.around()
    temp_mask = np.around(temp_mask, decimals=0)  # round 1.6 -> 2., 1.1 -> 1.
    current_uni_ids = np.unique(temp_mask)
     
    out_mask = np.full(mask.shape, 0, 'float32')
     
    mh, mw = mask.shape
    for i in range(mh-1):
        for j in range(mw-1):
            for k in range(1, len(current_uni_ids)):
                if mask[i][j] > (current_uni_ids[k] - good_range) and mask[i][j] < (current_uni_ids[k] + good_range):  
                    out_mask[i][j] = original_uni_ids[k] 
                    #mask[i][j] = current_uni_ids[k]
           
#     const = 0.005
#     out_mask = original_uni_ids[(np.abs(mask - original_uni_ids[:,None,None]) < const).argmax(0)]
              
    #return mask
    return out_mask
        



def draw_arrow(image, p, q, color, arrow_magnitude, thickness, line_type, shift):
    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
    int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    
def draw_reg_text(img, obj_info):
    #print 'tbd'
    
    obj_id = obj_info[0]
    cfd = obj_info[1]
    xmin = obj_info[2]
    ymin = obj_info[3]
    xmax = obj_info[4]
    ymax = obj_info[5]
    
    draw_arrow(img, (xmin, ymin), (xmax, ymin), col_map[6], 0, 2, 8, 0)#obj_id
    draw_arrow(img, (xmax, ymin), (xmax, ymax), col_map[6], 0, 2, 8, 0)
    draw_arrow(img, (xmax, ymax), (xmin, ymax), col_map[6], 0, 2, 8, 0)
    draw_arrow(img, (xmin, ymax), (xmin, ymin), col_map[6], 0, 2, 8, 0)
    
    # put text
    txt_obj = OBJ_CLASSES[obj_id] + ' ' + str(cfd)
    #cv2.putText(img, txt_obj, (xmin, ymin-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, col_map[obj_id], 1) # draw with red
    #cv2.putText(img, txt_obj, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, col_map[obj_id], 2)
    
#     # draw center
#     center_x = (xmax - xmin)/2 + xmin
#     center_y = (ymax - ymin)/2 + ymin
#     cv2.circle(img,(center_x, center_y), 3, (0, 255, 0), -1)
    
    return img



def visualize_mask(im, rois_final, rois_class_score, rois_class_ind, masks, ori_height, ori_width, im_name, thresh):
    img_org = im
    #img_org = img_org/255

    if rois_final.shape[0] == 0:
        print 'No detected box at all!'
        return

    inds = np.where(rois_class_score[:, -1] >= thresh)[0]
    #inds = np.where((rois_class_score[:, -1] >= 0.17) & (rois_class_score[:, -1] <= 0.24) | (rois_class_score[:, -1] >= 0.25))[0]
    if len(inds) == 0:
        print 'No detected box with probality > thresh = ', thresh, '-- Choossing highest confidence bounding box.'
        inds = [np.argmax(rois_class_score)]  
        max_conf = np.max(rois_class_score)
        if max_conf < 0.001:
            return  ## confidence is < 0.001 -- no good box --> must return
            

    rois_final = rois_final[inds, :]
    rois_class_score = rois_class_score[inds,:]
    rois_class_ind = rois_class_ind[inds,:]
    

    # get mask
    masks = masks[inds, :, :, :]
    
    im_width = im.shape[1]
    im_height = im.shape[0]
    
    # transpose
    im = im[:, :, (2, 1, 0)]

    num_boxes = rois_final.shape[0]
    
    list_bboxes = []

    
    for i in xrange(0, num_boxes):
        
        curr_mask = np.full((im_height, im_width), 0.0, 'float') # convert to int later
            
        class_id = int(rois_class_ind[i,0])
    
        bbox = rois_final[i, 1:5]
        score = rois_class_score[i,0]
        
        if cfg.TEST.MASK_REG:

            x1 = int(round(bbox[0]))
            y1 = int(round(bbox[1]))
            x2 = int(round(bbox[2]))
            y2 = int(round(bbox[3]))

            x1 = np.min((im_width - 1, np.max((0, x1))))
            y1 = np.min((im_height - 1, np.max((0, y1))))
            x2 = np.min((im_width - 1, np.max((0, x2))))
            y2 = np.min((im_height - 1, np.max((0, y2))))
            
            cur_box = [class_id, score, x1, y1, x2, y2]
            list_bboxes.append(cur_box)
            
            h = y2 - y1
            w = x2 - x1
            
            
            mask = masks[i, :, :, :]
            mask = np.argmax(mask, axis=0)
            
            
            original_uni_ids = np.unique(mask)
            
            # sort before_uni_ids and reset [0, 1, 7] to [0, 1, 2]
            original_uni_ids.sort()
            mask = reset_mask_ids(mask, original_uni_ids)
            
            mask = cv2.resize(mask.astype('float'), (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            #mask = convert_mask_to_original_ids(mask, original_uni_ids)
            mask = convert_mask_to_original_ids_manual(mask, original_uni_ids)
            
            #FOR MULTI CLASS MASK
            curr_mask[y1:y2, x1:x2] = mask # assign to output mask
            
            # visualize each mask
            curr_mask = curr_mask.astype('uint8')
            color_curr_mask = label_colours.take(curr_mask, axis=0).astype('uint8')
            cv2.imshow('Mask' + str(i), color_curr_mask)
            #cv2.imwrite('mask'+str(i)+'.jpg', color_curr_mask)


    #ori_file_path = img_folder + '/' + im_name
    #img_org = cv2.imread(ori_file_path)
    for ab in list_bboxes:
        print 'box: ', ab
        img_out = draw_reg_text(img_org, ab)
    
    cv2.imshow('Obj detection', img_out)
    #cv2.imwrite('obj_detction.jpg', img_out)
    cv2.waitKey(1)

    # find average coordinates
    affordance_id = 6 # 1: grasp 2:cut 3: scoop 4: contain 5:pound 6: support 7:wrap-grasp
    ii = np.where(curr_mask == affordance_id)
    if len(ii[0]) != 0:
        ii = np.array(ii)
        ii_mean = np.mean(ii, axis=1)
        depths = depth_raw[np.where(curr_mask == affordance_id)]

        #ii = np.array([223, 410])
        #ii = np.expand_dims(ii, axis=1)
        #depths = depth_raw[223, 410]

        # find a line to fit
        z = np.polyfit(ii[1, :], 480-ii[0, :], 1) # y[k] = z[1] + z[0]*x[k]
        print 'angle based on image'
        print (math.atan(z[0])/3.14*180)
        print 'z value:'
        print (z[0])
        print 'variance in horizontoal:'
        print np.var(ii[1, :])
        print 'variance in vertical:'
        print np.var(ii[0, :])

        if np.var(ii[1, :]) < 90:
            rot = 0.0
        else:
            rot = math.atan(z[0])/3.14*180 + 90

        angle_table = compute_imgRot(frame_current)

        # find coordinate on table
        coordinate_img2table(frame_current, ii_mean[1], ii_mean[0], rot - angle_table, depths, ii, affordance_id) #(->, |)

def run_affordance_net(net, image_name):
    count = 0
    im = []
    tmp_g = []
    #fig, ax = plt.subplots(figsize=(12, 12))

    im = image_name
    tmp_g = im[:,:,1]
    im[:,:,1] = im[:,:,2] 
    im[:,:,2] = tmp_g  
    #im = im*255
 
    img = im.astype('uint8')
    #ax.imshow(img)
    
    ori_height, ori_width, _ = im.shape
    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    if cfg.TEST.MASK_REG:
        rois_final, rois_class_score, rois_class_ind, masks, scores, boxes = im_detect2(net, im)
    else:
        1
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, rois_final.shape[0])
    
    # Visualize detections for each class
    visualize_mask(im, rois_final, rois_class_score, rois_class_ind, masks, ori_height, ori_width, str(count), thresh=CONF_THRESHOLD)

    count = count +1
    #savepath = './data/demo/results_all_cls/' + str(count) + '.png'
    #plt.savefig(savepath)
    #plt.cla()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='AffordanceNet demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    
    
    prototxt = root_path + '/models/pascal_voc/VGG16/faster_rcnn_end2end/test_2nd.prototxt'
    #caffemodel = root_path + '/pretrained/vgg16_faster_rcnn_iter_208000.caffemodel'
    caffemodel = root_path + '/pretrained/vgg16_faster_rcnn_iter.caffemodel'
    #caffemodel = root_path + '/output/faster_rcnn_end2end/voc_2012_train/vgg16_faster_rcnn_iter_16000.caffemodel'   
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    
    # load network
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    while 1:
        # get a frame from RGB camera
        frame_current = get_video()
        # get a frame from depth sensor
        depth, depth_raw = get_depth()

        # display RGB image
        cv2.imshow('RGB image', frame_current)
        # display depth image
        cv2.imshow('Depth image', depth)
        cv2.imwrite('raw.png', depth_raw) 
        cv2.waitKey(1)


        if frame_current != []:
            print '##########################################################'
            run_affordance_net(net, frame_current)

        cv2.imshow('frame',frame_current)
        cv2.waitKey(1)




