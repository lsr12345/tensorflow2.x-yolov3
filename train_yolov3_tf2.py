
# coding: utf-8

# In[1]:


import datetime
import tensorflow as tf
import tensorflow.keras as keras
from math import ceil
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K

import numpy as np
import os
import cv2

from model import yolov3_model
from utils import DataGenerator


print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True


# In[2]:


lr = 1e-3
Epochs = 50
finetune = False

checkpoints_dir = f'checkpoints/{datetime.date.today()}'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir) 
    
    
batch_size = 4
image_size = (416, 416)

class_path = './voc_classes.txt'
cls2id = {}
with open(class_path, mode='r', encoding='UTF-8') as fr:
    ff = fr.readlines()
    for i,j in enumerate(ff):
        cla = j.strip()
        cls2id[cla] = i
        
num_class = len(cls2id)
assert len(cls2id) == num_class, "num_class != num_ids"

anchor_path = './yolo_anchors.txt'
with open(anchor_path, mode='r', encoding='UTF-8') as fr:
    ff = fr.readlines()
    for i,j in enumerate(ff):
        anchors_ = j.strip().replace(' ', '').split(',')
        anchors = np.array(anchors_).astype(int).reshape((-1,2))[::-1].tolist()  # anchors 大 -> 小
#         anchors = np.array(anchors_).astype(int).reshape((-1,2))[::-1]  # anchors 大 -> 小
max_boxes = 20

root_path = '/home/shaoran/Data/Dec/VOC2012/VOCdevkit/VOC2012/'
train_test_ratio = 0.95

# list_IDs, list_xmls
list_IDs = []
list_xmls = []

for i in os.listdir(os.path.join(root_path, 'JPEGImages')):
    path_img = os.path.join(root_path, 'JPEGImages', i)
    xml = i.split('.')[0] + '.xml'
    path_xml = os.path.join(root_path, 'Annotations', xml)
    if os.path.exists(path_xml):
        list_IDs.append(path_img)
        list_xmls.append(path_xml)
    else:
        continue

# cc = list(zip(list_IDs, list_xmls))
# np.random.shuffle(cc)
# list_IDs, labels =zip(*cc)
        
assert len(list_xmls) == len(list_IDs), "labels != images"
print('sample nums: ', len(list_xmls))
length_data = len(list_xmls)

train_list_IDs = list_IDs[:int(train_test_ratio*length_data)]
train_list_xmls = list_xmls[:int(train_test_ratio*length_data)]

test_list_IDs = list_IDs[int(train_test_ratio*length_data):]
test_list_xmls = list_xmls[int(train_test_ratio*length_data):]


# In[3]:


def yolo_loss_(y_true, y_pred):
    return y_pred


# In[4]:


train_generator = DataGenerator(train_list_IDs, train_list_xmls, num_class, cls2id, anchors, batch_size, image_size, max_boxes=max_boxes, is_training=True)
val_generator = DataGenerator(test_list_IDs, test_list_xmls, num_class, cls2id, anchors, batch_size, image_size, max_boxes=max_boxes, is_training=False)

model = yolov3_model(num_class, anchors, max_boxes, image_size, batch_size, is_training=True)
# model = yolov3_model(num_class, anchors, max_boxes, image_size, is_training=True)

if finetune:
    
    lr = lr*0.5
    yolov3_filepath = './models/yolov3_weights.h5'
    
    model.load_weights(yolov3_filepath, by_name=True)
    
    
# myyolo_loss = partial(yolo_loss, anchors=anchors, num_chasses=num_class, image_size=(416,416), ignore_thresh=0.5)
# myyolo_loss.__name__ = 'myyolo_loss'

# model.compile(loss=lambda y_true, y_pred: tf.reduce_sum(y_pred), optimizer=keras.optimizers.Adam(learning_rate=lr))
model.compile(loss=[yolo_loss_, yolo_loss_, yolo_loss_], optimizer=keras.optimizers.Adam(learning_rate=lr))
# model.compile(loss=myyolo_loss, optimizer=keras.optimizers.Adam(learning_rate=lr))

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = Epochs
    step_each_epoch=int(len(train_list_IDs) // batch_size) #根据自己的情况设置
    baseLR = lr
    power = 0.9
    ite = K.get_value(model.optimizer.iterations)
    # compute the new learning rate based on polynomial decay
    alpha = baseLR*((1 - (ite / float(maxEpochs*step_each_epoch)))**power)
    # return the new learning rate
    return alpha

def scheduler(epoch):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.pow(0.9, epoch-10)  # tf.math.pow(0.9, 10)=0.35, tf.math.pow(0.9, 20)=0.122, tf.math.pow(0.9, 50)=0.005

learningRateScheduler_poly = callbacks.LearningRateScheduler(poly_decay)
learningRateScheduler_scheduler = callbacks.LearningRateScheduler(scheduler)

ReduceLROnPlateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)

logs_dir = f'logs/{datetime.date.today()}'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir) 
tensorboard = callbacks.TensorBoard(log_dir=logs_dir)

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(checkpoints_dir, 'tf_yolov3_{epoch:02d}_{loss:.4f}_{val_loss:.4f}_weights.h5'),
    verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='auto'
)

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=ceil(len(train_list_IDs) // batch_size),
    initial_epoch=0,
    epochs=Epochs,
#     callbacks=[learningRateScheduler, checkpoint, tensorboard],
    callbacks=[ReduceLROnPlateau, checkpoint, tensorboard],
    validation_data=val_generator,
    validation_steps=ceil(len(test_list_IDs) // batch_size),
    use_multiprocessing=True, workers=2,
    verbose=1)


model_dir = f'models/{datetime.date.today()}'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save_weights(os.path.join(model_dir, 'tf_yolov3_model_weights.h5'))
model.save(os.path.join(model_dir,'tf_yolov3_model.h5'))
tf.saved_model.save(os.path.join(model_dir, 'saved_tf_model'))

