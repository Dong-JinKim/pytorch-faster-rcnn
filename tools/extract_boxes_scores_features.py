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

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os, cv2
import argparse
import json

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch
import pdb

CLASSES = ('background',
'person', 'bicycle', 'car','motorcycle','airplane','bus','train', 'truck', 'boat','traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench','bird','cat','dog','horse','sheep',
'cow', 'elephant','bear','zebra','giraffe','hat','umbrella', 'handbag','tie','suitcase',
'frisbee','skis','snowboard','sports ball','kite', 'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
'bed','dining table','window','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
'oven', 'sink','refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','tooth brush')


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    nms_keep_indices = [None]*len(CLASSES)
    for cls_ind, cls in enumerate(CLASSES[:]):
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        nms_keep_indices[cls_ind] = keep.numpy().tolist()
        
    return scores, boxes, nms_keep_indices


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extract scores and boxes')
    parser.add_argument(
        '--im_in_out_json',
        dest='im_in_out_json',
        default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    
    # Specify path to pretrained faster-rcnn checkpoint to load
    #saved_model_path = '/home/dongjin/Documents/msra_human_object_interaction/mmdetection/no_frills/' + \
    #    'res152_tf/' + \
    #    'res152_faster_rcnn_iter_1190000.pth' 
    saved_model_path = '/home/dongjin/Documents/pytorch-faster-rcnn/output/res152/hico_train/1e6_RE/res152_faster_rcnn_iter_45000.pth'
    
    
    assert_err = 'Saved model file not found'
    assert(os.path.isfile(saved_model_path)), assert_err
    
    # Load network
    net = resnetv1(num_layers=152)
    net.create_architecture(
        81,
        tag='default', 
        anchor_scales=[4, 8, 16, 32])
    
    state_dict = torch.load(saved_model_path)
    
    for ii in range(4):
        state_dict[f'resnet.bn{ii+1}.num_batches_tracked'] = torch.LongTensor([0])[0]
        state_dict[f'resnet.layer{ii+1}.0.downsample.1.num_batches_tracked'] = torch.LongTensor([0])[0]
        for jj in range(40):
            for kk in range(4):
                state_dict[f'resnet.layer{ii+1}.{jj}.bn{kk+1}.num_batches_tracked'] = torch.LongTensor([0])[0]
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    
    #net.rpn_net = torch.nn.DataParallel(net.rpn_net,device_ids=range(torch.cuda.device_count()))#----!!!!
    #net.rpn_cls_score_net = torch.nn.DataParallel(net.rpn_cls_score_net,device_ids=range(torch.cuda.device_count()))#----!!!!
    #net.rpn_bbox_pred_net = torch.nn.DataParallel(net.rpn_bbox_pred_net,device_ids=range(torch.cuda.device_count()))#----!!!!
    #net.cls_score_net = torch.nn.DataParallel(net.cls_score_net,device_ids=range(torch.cuda.device_count()))#----!!!!
    #net.bbox_pred_net = torch.nn.DataParallel(net.bbox_pred_net,device_ids=range(torch.cuda.device_count()))#----!!!!
    #net._roi_pool_layer = torch.nn.DataParallel(net._roi_pool_layer,device_ids=range(torch.cuda.device_count()))#----!!!!
    
    
    if args.im_in_out_json:
        with open(args.im_in_out_json,'r') as file:
            images_in_out = json.load(file)

    else:
        images_in_out = [
            {
                'in_path': 'data/demo/000456.jpg', 
                'out_dir': '/tmp/pytorch_faster_rcnn_demo/',
                'prefix': '000456_'
            }
        ]

    for image_in_out in tqdm(images_in_out):
        #print(image_in_out['prefix'])
        #nofrills_feat = np.load('/data/DJKim/Datasets/HICO/hico_processed/faster_rcnn_boxes/'+image_in_out['prefix']+'fc7.npy') 
        
        out_dir = image_in_out['out_dir']
        prefix = image_in_out['prefix']
        #if os.path.exists(os.path.join(out_dir,f'{prefix}nms_keep_indices.json')):#-----!!
        #    continue
            
        im = cv2.imread(image_in_out['in_path'])
        
        scores, boxes, nms_keep_indices = demo(net,im)
        fc7 = net._predictions['fc7'].data.cpu().numpy()
        
        #if fc7.shape[0]==nofrills_feat.shape[0]:
        #    print(np.linalg.norm(nofrills_feat-fc7))
        #else:
        #    print('%d vs %d'%(fc7.shape[0],nofrills_feat.shape[0]))
        #if not os.path.exists(out_dir):
        #    os.makedirs(out_dir)
        #pdb.set_trace()
        scores_path = os.path.join(out_dir,f'{prefix}scores.npy')
        np.save(scores_path,scores)

        boxes_path = os.path.join(out_dir,f'{prefix}boxes.npy')
        np.save(boxes_path,boxes)

        fc7_path = os.path.join(out_dir,f'{prefix}fc7.npy')
        np.save(fc7_path,fc7)
        
        nms_keep_indices_path = os.path.join(out_dir,f'{prefix}nms_keep_indices.json')
        with open(nms_keep_indices_path,'w') as file:
            json.dump(nms_keep_indices,file)
