'''
 CUDA TEST
'''
# '''
import torch

print("torch version:",torch.__version__)


x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)
# '''

'''
cuDNN test
'''
'''
from torch.backends import cudnn#
print(cudnn.is_acceptable(xx))

print(torch.cuda.is_available(),'\r\n',
      torch.cuda.device_count(),'\r\n',
      torch.cuda.get_device_name(0),'\r\n',
      torch.cuda.current_device(),"\r\n",'\r\n',
      torch.cuda.get_device_capability(),'\r\n',#获取设备的 CUDA 算力.
      torch.cuda.current_blas_handle(),'\r\n',

      )
'''
'''
import nvidia_smi
#记录nvidia显存使用
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

loss_list = []
gpu_menory_used = []
gpu_menory = []
gpu_menory_res =[]

def get_gpu_menory():
    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    res_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    gpu_menory_res.append(res.gpu)
    gpu_menory.append(res.memory)
    gpu_menory_used.append(res_info.used)
    print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%, gpu-used:{((res_info.used ))}M')


get_gpu_menory()
'''
from pynvml import *
nvmlInit()     #初始化
print("Driver: ",nvmlSystemGetDriverVersion() )  #显示驱动信息

#查看设备
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("GPU", i, ":", nvmlDeviceGetName(handle))

#查看显存、温度、风扇、电源
handle = nvmlDeviceGetHandleByIndex(0)

info = nvmlDeviceGetMemoryInfo(handle)
print("Memory Total: \t",info.total)
print("Memory Free: \t",info.free)
print("Memory Used: \t",info.used)

print("Temperature is %d C"%nvmlDeviceGetTemperature(handle,0))

print("Power ststus",nvmlDeviceGetPowerState(handle))


#最后要关闭管理工具
nvmlShutdown()
