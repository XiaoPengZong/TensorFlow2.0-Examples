#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:07:27
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 1、定义模型输入
input_size   = 416
input_layer  = tf.keras.layers.Input([input_size, input_size, 3])

# 2、定义模型输出
# 获得三种尺度的卷积输出
# 具体实现见 YOLOv3 函数说明
feature_maps = YOLOv3(input_layer)
bbox_tensors = []
# 依次遍历小、中、大尺寸的特征图
for i, fm in enumerate(feature_maps):
    # 对每个分支的通道信息进行解码，得到预测框的大小、置信度和类别概率
    # 具体操作见 decode 函数说明
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

# 3 加载权重文件
# 根据上边定义好的输入输出，实例化模型
model = tf.keras.Model(input_layer, bbox_tensors)
# 加载权重文件
utils.load_weights(model, "./yolov3.weights")
# 输出模型信息
model.summary()

# 4、准备输入数据
image_path   = "./docs/kite.jpg"
original_image      = cv2.imread(image_path)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

# 5、模型前向推理
pred_bbox = model.predict(image_data)
pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox, axis=0)

# 6、输出后处理，
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')

# 7、结果可视化
image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.save("result.jpg")
# image.show()


