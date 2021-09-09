# coding=utf-8

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
import argparse

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

parser=argparse.ArgumentParser()
parser.add_argument('--currency',type=str,default='ETH-USDT',help='the type of curreny to be estimated.')
opt = parser.parse_args()
print(opt)
 
def load_image(filepath,target_size = (128,128),as_gray = False):
    img = io.imread(filepath, as_gray = as_gray)
    img = img / 255
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,)) if as_gray else img
    img = np.reshape(img,(1,)+img.shape)

    return img

def run_predict(model, img_test, results_path='results', out_img='out.png'):

    last_y = None
    predict_n = 8
    spec=None
    for n in range(predict_n):
        # 重复预测时，添加最后nn列数据
        if last_y is not None:
            nn = last_y.shape[1]
            #print(nn)
            # 左移一列，最右mask_num清零
            img_test[0] = np.roll(img_test[0], -nn, axis=1)
            img_test[0][:,-mask_num-nn:-mask_num] = last_y
            img_test[0][:,-mask_num:] = 0
            

        # 预测结果
        results = model.predict(img_test, verbose=1)
        if n==0:
            spec = results[0]
        # 保存中间结果，下次生成图片时使用
        last_col = results[0][:,-mask_num:-mask_num+1]
        
        last_y = last_col

    # 对比预测方向
    img_pred = results[0]

    # 右移
    img_pred = np.roll(img_pred, mask_num-1, axis=1)
    img_pred[:,:mask_num-1] = 0. 
    spec = np.roll(spec,mask_num,axis=1)
    spec[:,:mask_num-1]=0.

    # 调整为大图，间隔空行
    new_img = np.zeros([time_span, time_span*2, 3])
    for x in range(time_span):
        if x==time_span-predict_n-1: # 画分割线
            for y in range(time_span):
                if (img_pred[y,x]<0.001).all():
                    img_pred[y,x] = 0.5

        new_img[:,x*2] = img_pred[:,x]
    io.imsave(os.path.join(results_path, out_img),(new_img*255).astype(np.uint8))
    datagen.get_predict_csv('%s/%s_now_1h.csv'%(data_path,opt.currency),spec[:,-3],opt.currency)
    io.imsave(os.path.join(results_path, 'first.png'),(spec*255).astype(np.uint8))


if __name__ == '__main__':
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # 从ok去最近数据
    X = okapi.get_recent('%s/%s_now_1h.csv'%(data_path,opt.currency), opt.currency, num=123, gap=3600) # 123个数据只会生成一个图片
    print('data from OKexi: ', len(X))

    # 生成图片
    X_test = datagen.generate_data2_for_test('%s/%s_now_1h.csv'%(data_path,opt.currency),
        output_dir=test_path, output_image=True)
    #print(X_test)
    # 加载模型
    
    model = unet.unet(input_size=input_size, 
        d_inner_hid=d_inner_hid, layers=layers, n_head=n_head, d_model=d_model, 
        pretrained_weights="%s/vit-unet_3_10000.weights"%data_path)

    
    
    img_test = load_image(os.path.join(test_path, '1.png'))
    start_time = datetime.now()
    run_predict(model, img_test, results_path=results_path, out_img='vit_1h.png')
    print("[Prediction time Cost: {!s}]".format(datetime.now()-start_time))
    # 生成 html
    #with open(html_path, "w") as f:
    #    f.write(HTML%(time.ctime(), 'results/vit_1h.png', time.time() ))
    