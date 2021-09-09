'''
Author: your name
Date: 2021-06-17 03:13:37
LastEditTime: 2021-06-17 03:25:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \ETH_predictor_keras_unet\vit_unet\plt.py
'''
# Copyright 2021 anonymity
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import time
import numpy as np
import skimage.io as io
import skimage.transform as trans
from datagen import datagen
from okapi import okapi
from datetime import datetime
from vit_unet import vit_unet as unet
import tensorflow as tf

d_inner_hid=128
layers=4
n_head=4
d_model=512

input_size = (128,128,3)
mask_num = 5
time_span = input_size[0]
data_path = "data"
test_path = '%s/test'%data_path
results_path = '%s/results'%data_path
html_path = "%s/predictor.html"%data_path
# 加载模型
model = unet.unet(input_size=input_size, 
    d_inner_hid=d_inner_hid, layers=layers, n_head=n_head, d_model=d_model, 
    pretrained_weights="%s/vit-unet_3_10000.weights"%data_path)

with tf.Session() as sess:
    writer = tf.compat.v1.summary.FileWriter('./log/',sess.graph)