
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2

# from loss import yolo_loss, Calculate_loss_Layer
from loss import yolo_loss, Calculate_loss_Layer

# print(tf.__version__)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True


# In[2]:


def Conv2D(x, filters, kernel_size=(3,3), strides=(1,1), padding='same', batchnorm=True, training=True):
    if batchnorm:
        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                strides=strides, padding=padding, kernel_regularizer=l2(0.0005))(x)
        x = keras.layers.BatchNormalization()(x, training=training)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
    else:
        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                strides=strides, padding=padding, activation='selu')(x)
    
    return x
        


# In[3]:


def Resodual_Block(x, filters, kernel_size=(3,3), strides=(1,1), padding='same', training=True):  # strides=(2,2)
    x_ = Conv2D(x, filters // 2, (1,1), strides, padding, training=training)
    x_ = Conv2D(x_, filters, kernel_size, strides, padding, training=training)
    
    return keras.layers.Add()([x, x_])


# In[4]:


def Conv_Res_repeat(x, num, filters, kernel_size=(3,3), strides=(2,2), padding='same', training=True):
    x = Conv2D(x, filters, kernel_size, strides, padding, training=training)
    
    for _ in range(num):
        x = Resodual_Block(x, filters, training=training)
        
    return x


# In[5]:


def Darknet_body(x, training=True):
    x = Conv2D(x, 32, (3,3), (1,1), 'same', True, training=training)
    x = Conv_Res_repeat(x, 1, 64, training=training)
    x = Conv_Res_repeat(x, 2, 128, training=training)
    x_2 = Conv_Res_repeat(x, 8, 256, training=training)
    x_1 = Conv_Res_repeat(x_2, 8, 512, training=training)
    x_0 = Conv_Res_repeat(x_1, 4, 1024, training=training)
    
    return x_0, x_1, x_2


# In[6]:


def head_net(inputs, num_classes, training=True):
    filters_list = [512, 256, 128]
    prediction = []
    for i, inp in enumerate(inputs):
        x = inp
        if i == 0:
            x = Conv2D(x, filters_list[i], (1,1), training=training)
            x = Conv2D(x, 2*filters_list[i], training=training)
            x = Conv2D(x, filters_list[i], (1,1), training=training)
            x = Conv2D(x, 2*filters_list[i], training=training)
            tmp_y = Conv2D(x, filters_list[i], (1,1), training=training)
            x = Conv2D(tmp_y, 2*filters_list[i], training=training)
            x = Conv2D(x, 3*(num_classes+5), (1,1), training=training)
        else:
            tmp_y = Conv2D(tmp_y, filters_list[i], (1,1), training=training)
            x_ = keras.layers.UpSampling2D()(tmp_y)
            x = keras.layers.concatenate([x, x_], axis=-1)
            x = Conv2D(x, filters_list[i], (1,1), training=training)
            x = Conv2D(x, 2*filters_list[i], training=training)
            x = Conv2D(x, filters_list[i], (1,1), training=training)
            x = Conv2D(x, 2*filters_list[i], training=training)
            tmp_y = Conv2D(x, filters_list[i], (1,1), training=training)
            x = Conv2D(tmp_y, 2*filters_list[i], training=training)
            x = Conv2D(x, 3*(num_classes+5), (1,1), training=training)  # num_classes, 是否存在物体， 偏移值
            
        prediction.append(x)
        
    # 13x13, 26x26, 52x52
    return prediction


# In[7]:


"""
# true_boxes.shape: (batch_size, max_boxes, 5)  5->x_min, y_min, x_max, y_max, class_id
y_true = true_boxes_process(true_boxes, anchors, num_chasses, image_size)

arg:
    y_true.shape: [(batch_size,13,13,3,num_class+5), (batch_size,26,26,3,5+num_class), (batch_size,52,52,3,5+num_class)]
    
"""

def yolov3_model(num_class, anchors, max_boxes=10, image_size=(416,416), batch_size=4, is_training=True):
    
    if not is_training:
        inputs = keras.layers.Input(name='Input_images', shape=(image_size[0], image_size[1], 3))

        x = Darknet_body(inputs, is_training)
        pred = head_net(x, num_class, is_training)
        model = keras.Model(inputs, pred)
                                     
    else:
        inputs = keras.layers.Input(name='Input_images', shape=(image_size[0], image_size[1], 3), batch_size=batch_size)

    #     true_boxes = keras.layers.Input(name='True_boxes', shape=(max_boxes, 5), batch_size=5)
        # y_true.shape: [(batch_size,13,13,3,num_class+5), (batch_size,26,26,3,5+num_class), (batch_size,52,52,3,5+num_class)]
#        y_true = [keras.layers.Input(name='y_true_{}'.format(i), shape=(3549, len(anchors)//3, 5+num_class), batch_size=batch_size) for i in (32, 16, 8)]
        # y_true.shape: (batch_size, 13*13+26*26+52*52=3549, 3, 5+num_class)
        y_true = keras.layers.Input(name='y_true', shape=(3549, len(anchors)//3, 5+num_class))
        
        x = Darknet_body(inputs, is_training)
        pred = head_net(x, num_class, is_training)
        loss = Calculate_loss_Layer(anchors, num_class, image_size)(y_true, pred)  # true_boxes
#         loss = keras.layers.Lambda(lambda x: yolo_loss(x[0], x[1], x[2], x[3], x[4], x[5]))([y_true, pred, anchors, num_class, image_size, 0.5])
        model = keras.Model([inputs, y_true], loss)
#         model = keras.Model(inputs, pred)
    return model


# In[8]:


if __name__ == '__main__':
    anchors = [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1]]
    print('anchors: ', anchors)
    
    train_model= yolov3_model(10, anchors)
    print('train_model summary: ')
    train_model.summary()
    
    pred_model= yolov3_model(10, anchors, is_training=False)
    print('pred_model summary')
    pred_model.summary()


