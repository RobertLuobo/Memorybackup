import torch
import time

import subprocess
import json
import pprint
import numpy as np
DEFAULT_ATTRIBUTES = (
    # 'index',
    # 'uuid',
    # 'name',
    # 'timestamp',
    # 'memory.total',
    'memory.used',
    # 'memory.free',
    'utilization.gpu',
    # 'utilization.memory'
)
GPU_resoures_list = []

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    # lines = [ line.strip() for line in lines if line.strip() != '' ]
    tmp = lines[0].split(',')
    global GPU_resoures_list
    GPU_resoures_list.append([line for line in tmp])

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def print_get_gpu_info():
    pprint.pprint(get_gpu_info())


test_flag = 0
if test_flag:
    for i in range(100):
        print_get_gpu_info()
    print(GPU_resoures_list, "\n", np.array(GPU_resoures_list).shape)
    # print(GPU_resoures_list[0][1])

    print(np.array(GPU_resoures_list).transpose()[:][0])
    print(np.array(GPU_resoures_list).transpose()[:][1])

    import matplotlib.pyplot as plt

    plt.figure(1)  # 创建图表1
    x_value = np.arange(GPU_resoures_list.__len__())
    plt.plot(x_value, np.array(GPU_resoures_list).transpose()[:][0].astype(np.float16), c='coral', linestyle='-',
             label='Memory-iteration, batchsize=20, epoch=1')
    plt.xlabel("iteration")
    plt.ylabel("Memory used(MB)")
    plt.title('FashionMNSIT - Alexnet')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure(2)  # 创建图表2
    plt.plot(x_value, np.array(GPU_resoures_list).transpose()[:][1].astype(np.float16), c='green', linestyle='-',
             label="GPU Utilization-iteration, batchsize=20, epoch=1")
    plt.xlabel("iteration")
    plt.ylabel("GPU Utilization( % )")
    plt.title('FashionMNSIT - Alexnet')
    plt.legend(loc='upper left')
    plt.show()




# plt.figure(1)#创建图表1

# ax1=plt.subplot(211)#在图表2中创建子图1
# ax2=plt.subplot(212)#在图表2中创建子图2

# ax1.set_title("Memory-iteration")
# ax2.set_title("GPU Utilization-iteration")
# ax1.set_xlabe("Memory used(MB)")

# x_value = np.arange(GPU_resoures_list.__len__())
# plt.plot(x_value, np.array(GPU_resoures_list).transpose()[:][0], c='coral', linestyle='-', label='Memory-iteration')
# plt.legend(loc='upper left')
# plt.sca(ax1)
# plt.plot(x_value, np.array(GPU_resoures_list).transpose()[:][1], c='green', linestyle='-', label="GPU Utilization-iteration")
# plt.legend(loc='upper left')
# plt.sca(ax2)
# plt.title('FashionMNSIT - Alexnet')
# plt.show()


GPUtil_flag= 0
if GPUtil_flag:
    import GPUtil
    import os
    GPUtil.showUtilization()
    DEVICE_ID = GPUtil.getFirstAvailable()
    print(DEVICE_ID)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    print('Device ID (unmasked): ' + str(DEVICE_ID))
    print('Device ID (masked): ' + str(0))