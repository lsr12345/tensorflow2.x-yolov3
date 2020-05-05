
# coding: utf-8

# In[1]:


import tensorflow.keras as keras

import numpy as np
import cv2
import os 
import math
import random
import time

import xml.etree.ElementTree as ET

import imgaug
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


# In[ ]:


"""
arg:
    true_boxes: (batch_size, max_boxes, 5)  5->x_min, y_min, x_max, y_max, class_id  np.array
    anchors: (9,2)  -> w,h
    image_size: (H, W)
    
return:
    y_true: [(batch_size,13,13,3,5+num_class), (batch_size,26,26,3,5+num_class), (batch_size,52,52,3,5+num_class)] 
    y_true box坐标x,y,w,h为相对全图的比例值， 0～1 之间
    Y = (batch_size,image_size[0]//32*image_size[1]//32+image_size[0]//16*image_size[1]//16+image_size[0]//8*image_size[1]//8,3,5+num_class)
      = (batch_size, 13*13+26*26+52*52, 3, 5+num_class)
"""

def true_boxes_process(true_boxes, anchors, num_class, image_size=(416,416)):
    anchors_ = np.array(anchors)
        
    Y = np.zeros((true_boxes.shape[0],image_size[0]//32*image_size[1]//32+image_size[0]//16*image_size[1]//16+image_size[0]//8*image_size[1]//8,3,5+num_class))
    y_true = [np.zeros((true_boxes.shape[0], i, j, 3, num_class+5)) for i, j in ((image_size[1]//32, image_size[0]//32),
                                                                                (image_size[1]//16, image_size[0]//16),
                                                                                (image_size[1]//8, image_size[0]//8))]
    
    # true_boxes.shape: (batch_size, max_boxes, 5)  5->x, y, w, h, class_id 其中x,y,w,h都是相对于整张图片大小的归化值
    true_xy_relative = ((true_boxes[:, :, 2:4] + true_boxes[:, :, :2]) // 2) / (image_size[1], image_size[0])
    true_wh_relative = (true_boxes[:, :, 2:4] - true_boxes[:, :, :2] ) / (image_size[1], image_size[0])
    true_boxes_relative = np.concatenate([true_xy_relative, true_wh_relative, np.expand_dims(true_boxes[:, :, 4], axis=-1)], axis=-1)
    
    # anchors_topleft_relative.shape: (9,2)
    anchors_topleft_relative = -0.5*anchors_
    # anchors_downright_relative.shape: (9,2)
    anchors_downright_relative = 0.5*anchors_
    # anchors_area.shape: (9,)
    anchors_area = anchors_[:, 0] * anchors_[:, 1]
    for batch in range(true_boxes.shape[0]):
        # boxes.shape: (max_boxes, 5)
        boxes = true_boxes[batch]
        # boxes_topleft.shape: (max_boxes, 2)
        boxes_topleft = boxes[:, :2]
        # boxes_downright.shape: (max_boxes, 2)
        boxes_downright = boxes[:, 2:4]
        # boxes_wh.shape: (max_boxes, 2)
        boxes_wh = boxes_downright - boxes_topleft
        # boxes_wh.shape: (max_boxes, 1, 2)
        boxes_wh = np.expand_dims(boxes_wh, axis=-2)
        # boxes_center.shape: (max_boxes, 2)
        boxes_center = (boxes_downright + boxes_topleft) // 2
        # boxes_topleft_relative.shape: (max_boxes, 1, 2)
        boxes_topleft_relative = -0.5*boxes_wh
        # boxes_downright_relative.shape: (max_boxes, 1, 2)
        boxes_downright_relative = 0.5*boxes_wh
        # boxes_area.shape: (max_boxes, 1)
        boxes_area = boxes_wh[:, :, 0] * boxes_wh[:, :, 1]
        
        # intersect_topleft.shape: (max_boxes, 9, 2)
#         boxes_topleft_relative = np.cast(boxes_topleft_relative, anchors_topleft_relative.dtype)
#         print(boxes_topleft_relative)
        boxes_topleft_relative = np.array(boxes_topleft_relative).astype('float32')
        intersect_topleft = np.maximum(boxes_topleft_relative, anchors_topleft_relative)
        # intersect_downright.shape: (max_boxes, 9, 2)
#         boxes_downright_relative = np.cast(boxes_downright_relative, anchors_downright_relative.dtype)
        boxes_downright_relative = np.array(boxes_downright_relative).astype('float32')
        intersect_downright = np.minimum(boxes_downright_relative, anchors_downright_relative)
        # intersect_wh.shape: (max_boxes, 9, 2)
        intersect_wh = np.maximum(intersect_downright - intersect_topleft, 0)
        # intersect_area.shape: (max_boxes, 9)
        intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]
    
        # IOU.shape: (max_boxes, 9)
#         boxes_area = np.cast(boxes_area, anchors_area.dtype)
        boxes_area = np.array(boxes_area).astype('float32')
        IOU = intersect_area / (boxes_area + anchors_area - intersect_area)
        # truth_anchors_id.shape: (max_boxes,)
        truth_anchors_id = np.argmax(IOU, axis=-1)
        
        layer2grid = [13, 26, 52]
        layer2scale = [32.0, 16.0, 8.0]
        
        for i in range(boxes.shape[0]):
            if boxes[i, 2] == 0:
                continue
            
            layer_num = int(truth_anchors_id[i]/3)
            grid = layer2grid[layer_num]  # 13 26 52
            scale = layer2scale[layer_num]  # 32 16 8
#             center_ = np.maximum(boxes_center[i]-1.0, 0.0) // scale  # grid下的中心坐标
            center_ = boxes_center[i] / scale# grid下的中心坐标
            x_ = math.floor(center_[0])
            y_  = math.floor(center_[1])
            # y_true[layer_num].shape: (batch_size,XX,XX,3,5+num_class)  5-> p,x,y,w,h
            y_true[layer_num][batch, y_, x_, truth_anchors_id[i]%3, 0] = 1  # 存在物体
            y_true[layer_num][batch, y_, x_, truth_anchors_id[i]%3, 1:5] = true_boxes_relative[batch, i, 0:4]
            class_id = true_boxes_relative[batch, i, 4]
            y_true[layer_num][batch, y_, x_, int(truth_anchors_id[i]%3), int(5+class_id)] = 1
            
        
#     y_true: [(batch_size,13,13,3,5+num_class), (batch_size,26,26,3,5+num_class), (batch_size,52,52,3,5+num_class)] 
#     y_true box坐标x,y,w,h为相对全图的比例值， 0～1 之间
        # # 13*13 = 169  26*26 = 676  52*52 = 2704
        a_ = y_true[0].shape[1]*y_true[0].shape[2]
        b_ = y_true[1].shape[1]*y_true[1].shape[2]
        c_ = y_true[2].shape[1]*y_true[2].shape[2]
#         print(a_, b_, c_)
        
        Y[:, 0:a_, :, :] = np.reshape(y_true[0], (true_boxes.shape[0], a_, y_true[0].shape[3], y_true[0].shape[4]))
        Y[:, a_:(a_+b_), :, :] = np.reshape(y_true[1], (true_boxes.shape[0], b_, y_true[1].shape[3], y_true[1].shape[4]))
        Y[:, (a_+b_):(a_+b_+c_), :, :] = np.reshape(y_true[2], (true_boxes.shape[0], c_, y_true[2].shape[3], y_true[2].shape[4]))
    return y_true, Y

def parse_xml(xml_path, cls2id):
    in_file = open(xml_path)
    tree=ET.parse(in_file)
    root = tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        box = []
        cls_ = obj.find('name').text
        cls_id = cls2id[cls_]
        xmlbox = obj.find('bndbox')
#         b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        box.append(int(float(xmlbox.find('xmin').text)))
        box.append(int(float(xmlbox.find('ymin').text)))
        box.append(int(float(xmlbox.find('xmax').text)))
        box.append(int(float(xmlbox.find('ymax').text)))
        box.append(cls_id)
        
        boxes.append(box)
                   
    return np.array(boxes)

aug = iaa.SomeOf((0, 3),[
    iaa.GammaContrast([0.5, 1.5]),
    iaa.Affine(translate_percent=[-0.05, 0.05], scale=[0.9, 1.1], mode='constant'),  # mode='edge'
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5)
])


# In[ ]:


class DataGenerator(keras.utils.Sequence):
    # list_IDs: 图片名list   xml_file list： 图片名对应的xml_file list
    def __init__(self,list_IDs, list_xmls, num_class, cls2id, anchors, batch_size=16, image_size=(416,416), max_boxes=20, is_training=True, shuffle=True):
        self.list_IDs = list_IDs
        self.list_xmls = list_xmls
        self.num_class = num_class
        self.cls2id = cls2id
        self.anchors = anchors
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.is_training = is_training
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_xmls_temp = [self.list_xmls[k] for k in indexes]
        
        x, y = self.__data_generation(list_IDs_temp, list_xmls_temp)
        return x, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, list_xmls_temp):
        # list_IDs: 图片名list   xml_file list： 图片名对应的xml_file list
        X = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], 3))
        # self.image_size[0]//32*self.image_size[1]//32+self.image_size[0]//16*self.image_size[1]//16+self.image_size[0]//8*self.image_size[1]//8 = 13*13+26*26+52*52
#         Y = np.zeros((self.batch_size,self.image_size[0]//32*self.image_size[1]//32+self.image_size[0]//16*self.image_size[1]//16+self.image_size[0]//8*self.image_size[1]//8,3,5+self.num_class))
        yy = np.zeros((self.batch_size), dtype=int)

        IMAGES = []
        BOXES = []
        for i in range(len(list_IDs_temp)):
            img = cv2.imread(list_IDs_temp[i])
            h, w, c = img.shape
            scale = min(self.image_size[0]/h, self.image_size[1]/w)
            nh = int(scale * h)
            nw = int(scale * w)
            img_mask = np.full((self.image_size[0], self.image_size[1], 3), fill_value=255, dtype=float)
            img_mask[:nh, :nw] = cv2.resize(img, (nw, nh))
            
            boxes = parse_xml(list_xmls_temp[i], self.cls2id)
            boxes[:, 0:4] = boxes[:, 0:4] * scale
            
            if self.is_training:
                bbs = BoundingBoxesOnImage([BoundingBox(x1=float(ii[0]), y1=float(ii[1]), x2=float(ii[2]), y2=float(ii[3])) for ii in boxes], shape=img_mask.shape)
                img_mask, boxes_aug = aug(image=img_mask, bounding_boxes=bbs)
                boxes_aug = boxes_aug.remove_out_of_image().clip_out_of_image()
                for j in range(len(boxes_aug)):
                    boxes[j, :4] = [float(boxes_aug.bounding_boxes[j].x1), float(boxes_aug.bounding_boxes[j].y1), float(boxes_aug.bounding_boxes[j].x2), float(boxes_aug.bounding_boxes[j].y2)]
            
            boxes_data = np.zeros((self.max_boxes,5))
            if len(boxes) > 0 and len(boxes)<= self.max_boxes:
#                 np.random.shuffle(boxes)
                boxes_data[:len(boxes)] = boxes
            elif len(boxes) > 0 and len(boxes) > self.max_boxes:
#                 np.random.shuffle(boxes)
                boxes_data = boxes[:self.max_boxes]
        
            IMAGES.append(img_mask)
            BOXES.append(boxes_data)
            
        IMAGES = np.array(IMAGES)
        BOXES = np.array(BOXES)
    
        # y_true: [(batch_size,13,13,3,5+num_class), (batch_size,26,26,3,5+num_class), (batch_size,52,52,3,5+num_class)]
        # y_true box坐标x,y,w,h为相对全图的比例值， 0～1 之间
        # Y = (batch_size, 13*13+26*26+52*52, 3, 5+num_class)
        y_true, Y = true_boxes_process(BOXES, self.anchors, self.num_class, self.image_size)

#         return [IMAGES, Y], yy
        return [IMAGES, Y], [yy, yy, yy]

