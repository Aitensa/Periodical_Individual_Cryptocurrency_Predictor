'''
Author: your name
Date: 2021-06-13 09:08:10
LastEditTime: 2021-06-13 14:07:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \ETH_predictor_keras_unet\GasAcquire.py
'''
# coding:utf-8
import sys
import urllib3, json, base64, time, hashlib
from urllib3.contrib.socks import SOCKSProxyManager
from datetime import datetime
import csv

urllib3.disable_warnings()
'''
# okex 提供的接入参数
Passphrase = ""
apikey = "2bed5534-0902-43fb-84dc-c53e3a5ec56f"
secretkey = "1B39FA1D2A95A6466309D0D29E4B075E"
'''
method = 'GET'
url = 'https://data-api.defipulse.com/api/v1/egs/api/predictTable.json?api-key=8b8829894fb5ea78010fec17f7e66e24a60fdd88ebc60a71da6c4fdf87bf'
def requireForData():
    start_time = datetime.now()
    proxy = SOCKSProxyManager('socks5h://localhost:7890')
    data = proxy.request(method,url)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))
    if data.status!=200:
        print("fail: ",data.status,data.data)
    else:
        return data.data

def ProcessData():
    data = json.loads(requireForData())[0]
    print(data)
    print(type(data))
    header = data.keys()
    with open("../ETH_predictor_keras_unet/data/now_gas.csv",'w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=header)
        writer.writeheader()
        writer.writerow(data)
    return data

if __name__ == "__main__":
    data = ProcessData()
    print(data)

