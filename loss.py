
# coding: utf-8

# In[2]:


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np


# In[3]:

"""

arg:
    true_boxes: (batch_size, max_boxes, 5)  5->x_min, y_min, x_max, y_max, class_id  np.array
    anchors: (9,2)  -> w,h
    image_size: (H, W)
    
return:
    y_true: [(batch_size,13,13,3,5+num_class), (batch_size,26,26,3,5+num_class), (batch_size,52,52,3,5+num_class)] 
    y_true box坐标x,y,w,h为相对全图的比例值， 0～1 之间


def true_boxes_process(true_boxes, anchors, num_chasses, image_size=(416,416)):
        
    y_true = [tf.zeros((true_boxes.shape[0], i, j, 3, num_chasses+5)) for i, j in ((image_size[1]//32, image_size[0]//32),
                                                                                (image_size[1]//16, image_size[0]//16),
                                                                                (image_size[1]//8, image_size[0]//8))]
    
    # true_boxes.shape: (batch_size, max_boxes, 5)  5->x, y, w, h, class_id 其中x,y,w,h都是相对于整张图片大小的诡异画值
    true_xy_relative = ((true_boxes[:, :, 2:4] + true_boxes[:, :, :2]) // 2) / (image_size[1], image_size[0])
    true_wh_relative = (true_boxes[:, :, 2:4] - true_boxes[:, :, :2] ) / (image_size[1], image_size[0])
    true_boxes_relative = tf.concat([true_xy_relative, true_wh_relative, tf.expand_dims(true_boxes[:, :, 4], axis=-1)], axis=-1)
    
    # anchors_topleft_relative.shape: (9,2)
    anchors_topleft_relative = -0.5*anchors
    # anchors_downright_relative.shape: (9,2)
    anchors_downright_relative = 0.5*anchors
    # anchors_area.shape: (9,)
    anchors_area = anchors[:, 0] * anchors[:, 1]
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
        boxes_wh = tf.expand_dims(boxes_wh, axis=-2)
        # boxes_center.shape: (max_boxes, 2)
        boxes_center = (boxes_downright + boxes_topleft) // 2
        # boxes_topleft_relative.shape: (max_boxes, 1, 2)
        boxes_topleft_relative = -0.5*boxes_wh
        # boxes_downright_relative.shape: (max_boxes, 1, 2)
        boxes_downright_relative = 0.5*boxes_wh
        # boxes_area.shape: (max_boxes, 1)
        boxes_area = boxes_wh[:, :, 0] * boxes_wh[:, :, 1]
        
        # intersect_topleft.shape: (max_boxes, 9, 2)
        boxes_topleft_relative = tf.cast(boxes_topleft_relative, anchors_topleft_relative.dtype)
        intersect_topleft = tf.maximum(boxes_topleft_relative, anchors_topleft_relative)
        # intersect_downright.shape: (max_boxes, 9, 2)
        boxes_downright_relative = tf.cast(boxes_downright_relative, anchors_downright_relative.dtype)
        intersect_downright = tf.minimum(boxes_downright_relative, anchors_downright_relative)
        # intersect_wh.shape: (max_boxes, 9, 2)
        intersect_wh = tf.maximum(intersect_downright - intersect_topleft, 0)
        # intersect_area.shape: (max_boxes, 9)
        intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]
    
        # IOU.shape: (max_boxes, 9)
        boxes_area = tf.cast(boxes_area, anchors_area.dtype)
        IOU = intersect_area / (boxes_area + anchors_area - intersect_area)
        # truth_anchors_id.shape: (max_boxes,)
        truth_anchors_id = tf.argmax(IOU, axis=-1)
        
        # out_layers_num.shape: (max_boxes,)
#         out_layers_num = [int(i/3) for i in truth_anchors_id]
        
        layer2grid = tf.constant([13, 26, 52])
        layer2scale = tf.constant([32.0, 16.0, 8.0])
        
#         for i, box in enumerate(boxes):
        for i in range(boxes.shape[0]):

            if boxes[i, 2] == 0:
                continue
#             layer_num = out_layers_num[i]  # 0 1 2
            layer_num = int(truth_anchors_id[i]/3)
            grid = layer2grid[layer_num]  # 13 26 52
            scale = layer2scale[layer_num]  # 32 16 8
            center_ = tf.maximum(boxes_center[i]-1.0, 0.0) // scale  # grid下的中心坐标
            
            x_ = center_[0].numpy()
            y_  = center_[1].numpy()
            # y_true[layer_num].shape: (batch_size,XX,XX,3,5+num_class)  5-> p,x,y,w,h
            y_true[layer_num][batch, y_, x_, truth_anchors_id[i].numpy()%3, 0] = 1  # 存在物体
            y_true[layer_num][batch, y_, x_, truth_anchors_id[i].numpy()%3, 1:5] = true_boxes_relative[batch, i, 0:4]
            class_id = true_boxes_relative[batch, i, 4]
            y_true[layer_num][batch, y_, x_, truth_anchors_id[i].numpy()%3, 5+class_id] = 1
            
    return y_true
"""

# In[4]:


"""
    arg:
        pred.shape: (batch_size,feature_size_h,feature_size_w,3*(5+num_class))
        anchors.shape: (3,2) 
        
    return:
        process_confidence.shape: (batch_size,feature_size_h,feature_size_w,3,1)
        process_xy.shape: (batch_size,feature_size_h,feature_size_w,3,2)                          
        process_wh.shape: (batch_size,feature_size_h,feature_size_w,3,2)                         
        process_class_probs.shape: (batch_size,feature_size_h,feature_size_w,3,num_class)
        grid.shape: (feature_size_h,feature_size_w,1,2)
        process_pred.shape: (batch_size,feature_size_h,feature_size_w,3,5+num_class)
"""

def pred_process(pred, anchors):
    batch_size,feature_size_h,feature_size_w = pred.shape[:3]
    num_class = int(pred.shape[-1] // 3 - 5)
    
    # process_pred.shape: (batch_size, feature_size_h, feature_size_w, 3, 5+num_class)
    process_pred = tf.reshape(pred, (batch_size, feature_size_h, feature_size_w, 3, 5+num_class))
    
    # grid_y.shape: (feature_size_h,feature_size_w,1,1)
    # (feature_size_h,) - > (feature_size_h,1,1,1) -> (feature_size_h,feature_size_w,1,1)
    grid_y = tf.tile(tf.reshape(K.arange(0, stop=feature_size_h), [-1, 1, 1, 1]), [1, feature_size_w, 1, 1])
    
    # grid_x.shape: (feature_size_h,feature_size_w,1,1)
    # (feature_size_w,) - > (1,feature_size_w,1,1) -> (feature_size_h,feature_size_w,1,1)    
    grid_x = tf.tile(tf.reshape(K.arange(0, stop=feature_size_w), [1, -1, 1, 1]), [feature_size_h, 1, 1, 1])
    
    # grid.shape: (feature_size_h,feature_size_w,1,2)
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, pred.dtype)    # grid[h,w,anchor,:]=(x,y)
    
    # process_xy.shape: (batch_size, feature_size_h, feature_size_w, 3, 2)
    process_xy = (tf.sigmoid(process_pred[:, :, :, :, 1:3]) + grid) / tf.cast((feature_size_w,feature_size_h), pred.dtype)
    
    # anchors.shape: (1,1,1,3,2)
    # (3,2) -> (1,1,1,3,2)
    anchors = tf.cast(tf.reshape(tf.constant(anchors), [1, 1, 1, len(anchors), 2]), process_pred.dtype)
    
    # process_wh.shape: (batch_size, feature_size_h, feature_size_w, 3, 2)
    process_wh = tf.math.exp(process_pred[:, :, :, :, 3:5])*anchors / tf.cast((feature_size_w,feature_size_h), pred.dtype)
    
    # process_confidence.shape: (batch_size, feature_size_h, feature_size_w, 3, 1)
    process_confidence = tf.sigmoid(process_pred[:, :, :, :, 0:1])
    
    # process_class_probs.shape: (batch_size, feature_size_h, feature_size_w, 3, num_class)
    process_class_probs = tf.sigmoid(process_pred[:, :, :, :, 5:])  # tf.math.softmax(process_pred[:, :, :, :, 5:])
        
    return process_confidence, process_xy, process_wh, process_class_probs, grid, process_pred


# In[5]:


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


# In[6]:


"""
# true_boxes.shape: (batch_size, max_boxes, 5)  5->x_min, y_min, x_max, y_max, class_id
y_true = true_boxes_process(true_boxes, anchors, num_chasses, image_size)

arg:
    y_true.shape: (batch_size, 13*13+26*26+52*52, 3, 5+num_class)
    y_pred.shape: [(batch_size,13,13,3,5+num_class), (batch_size,26,26,3,num_class+5), (batch_size,52,52,3,5+num_class)]
    anchors.shape: (9,2)  wh -> 从大到小
return:
    loss
"""

def yolo_loss(y_true, y_pred, anchors, num_chasses, image_size=(416,416), ignore_thresh=0.5):
    # (batch_size, 13*13+26*26+52*52, 3, 5+num_class) --> [(batch_size,13,13,3,5+num_class), (batch_size,26,26,3,num_class+5), (batch_size,52,52,3,5+num_class)]
    # 13*13 = 169  26*26 = 676  52*52 = 2704
#     ss = tf.reshape(y_true[:,0:169, :, :], [tf.shape(y_true)[0], 13,13,tf.shape(y_true)[2], tf.shape(y_true)[3]])
#     print(y_true.shape[0])
    Y_true = [tf.reshape(y_true[:,0:169, :, :], [tf.shape(y_true)[0], 13,13, tf.shape(y_true)[2], tf.shape(y_true)[3]]),
              tf.reshape(y_true[:, 169:845, :, :], [tf.shape(y_true)[0], 26,26, tf.shape(y_true)[2], tf.shape(y_true)[3]]), 
              tf.reshape(y_true[:, 845:3549, :, :], [tf.shape(y_true)[0], 52,52, tf.shape(y_true)[2], tf.shape(y_true)[3]])]
#     loss = tf.Variable(initial_value=0.0, trainable=False)
    
    _loss = []
    
    num_layers = np.array(anchors).shape[0] // 3
    for i, layer_pred in enumerate(y_pred):
        
        # y_true_.shape: (batch_size,h,w,3,5+num_class)
        y_true_ = Y_true[i]
        batch_size = tf.shape(y_true_)[0]
        batch_size_t = tf.cast(batch_size, y_true_.dtype)
        object_mask = y_true_[:, :, :, :, 0:1]
#         print(object_mask.shape)
#         print()
        true_class_probs = y_true_[:, :, :, :, 5:]
        feature_size_h,feature_size_w = y_true_.shape[1:3]
        
        # anchors_.shape: (3,2)
        anchors_ = [anchors[num_layers*i], anchors[num_layers*i+1], anchors[num_layers*i+2]]
        anchors_ = np.array(anchors_)
        
#         confidence.shape: (batch_size,feature_size_h,feature_size_w,3,1)
#         xy.shape: (batch_size,feature_size_h,feature_size_w,3,2)                          
#         wh.shape: (batch_size,feature_size_h,feature_size_w,3,2)                         
#         class_probs.shape: (batch_size,feature_size_h,feature_size_w,3,num_class)
#         grid.shape: (feature_size_h,feature_size_w,1,2)
#         pred.shape: (batch_size,feature_size_h,feature_size_w,3,5+num_class)
        
#         feature_size_h,feature_size_w == h,w

        confidence, xy, wh, class_probs, grid, pred = pred_process(layer_pred, anchors_)
        
        # pred_box.shape: (batch_size, feature_size_h, feature_size_w, 3, 4)
        pred_box = tf.concat([xy, wh], axis=-1)
        
        # true_xy.shape: (batch_size, h, w, 3, 2)
        # y_true_[:, :, :, :, 1:3]为相对于整张图的归一化值，其值 * feature_map的大小 - 单元格坐标 = 0~1相对于该单元格的偏移量，与pred_xy相对应
        true_xy = y_true_[:, :, :, :, 1:3] * tf.cast((feature_size_w, feature_size_h), layer_pred.dtype) - grid
        
        # true_wh.shape: (batch_size, h, w, 3, 2)
        # y_true_[:, :, :, :, 3:5]为相对于整张图的归一化值，其值 //anchors.wh*feature_map的大小再log一下则与pred_wh相对应，定位预测公式反推
        true_wh = tf.math.log(y_true_[:, :, :, :, 3:5] / anchors_ * image_size[::-1])
        # # K.switch(condition, then_expression, else_expression)：Switches between two operations depending on a scalar value.
        true_wh = K.switch(object_mask, true_wh, tf.zeros_like(true_wh))  # avoid log(0)=-inf
        
        # box_loss_scale.shape: (batch_size,h,w,3,1)
        # 计算boxes的位置误差时,根据ground truth的大小对权重系数进行修正：(2 - truth.w*truth.h)，w和h都归一化到(0,1)
        box_loss_scale = 2 - y_true_[:,:,:,:,3:4]*y_true_[:,:,:,:,4:5]
        
        # Find ignore mask, iterate over each of batch.
        # TensorArray可以看做是具有动态size功能的Tensor数组。通常都是跟while_loop或map_fn结合使用。
        ignore_mask = tf.TensorArray(y_true_.dtype, size=1, dynamic_size=True)
        object_mask_bool = tf.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true_[b,...,1:5], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou<ignore_thresh, true_box.dtype))
            return b+1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b,*args: b<batch_size, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        
        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(true_xy, xy, from_logits=True)  # 修正 pred[...,1:3] -> xy
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(true_wh-wh)  # 修正 pred[...,3:5] -> wh
        
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, confidence, from_logits=True) + (1-object_mask) * K.binary_crossentropy(object_mask, confidence, from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, class_probs, from_logits=True)
        
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3, 4))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3, 4))
        confidence_loss = tf.reduce_sum(confidence_loss, axis=(1, 2, 3, 4))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3, 4))
        _loss.append(xy_loss + wh_loss + confidence_loss + class_loss)
    loss = _loss[0] + _loss[1] + _loss[2]
    
    return loss


# In[7]:


class Calculate_loss_Layer(keras.layers.Layer):
    def __init__(self, anchors, num_chasses, image_size):
        super(Calculate_loss_Layer, self).__init__()
        self.anchors = anchors
        self.num_chasses = num_chasses
        self.image_size = image_size
    
#     def call(self, true_boxes, y_pred):
    def call(self, y_true, y_pred):
        #             true_boxes
        return yolo_loss(y_true, y_pred, self.anchors, self.num_chasses, self.image_size)
    
    def get_config(self):
        config = {"anchors":self.anchors, "num_chasses":self.num_chasses, "image_size":self.image_size}
        base_config = super(Calculate_loss_Layer, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))

