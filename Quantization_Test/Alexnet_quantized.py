# MNIST Model
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import random
class Net(nn.Module):
    def __init__(self, mnist=True):

        super(Net, self).__init__()
        if mnist:
            num_channels = 1
        else:
            num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

Time_test = []
Time_iteraion = []
GPU_resoures_list = []
Time_epoch = []
GPU_DEVICE = 2
Test_accuracy = []
Test_loss = []

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2) # in_channels, out_channels, kernel_size, stride, padding
        # self.conv1 = nn.Conv2d(3, 96, 11, 4),  # in_channels, out_channels, kernel_size, stride, padding
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 3, 2)

        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 3, 2)

        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        x = F.max_pool2d(x, 3, 2)

        x = x.view(-1, 256*6*6)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5)

        x = F.relu(self.fc2(x), inplace=True)
        x = F.dropout(x, 0.5)

        x = self.fc3(x)

        return x


DEFAULT_ATTRIBUTES = (
    'index',
    # 'uuid',
    # 'name',
    # 'timestamp',
    # 'memory.total',
    'memory.used',
    # 'memory.free',
    'utilization.gpu',
    # 'utilization.memory'
)

import subprocess
import json
import pprint
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    # lines = [ line.strip() for line in lines if line.strip() != '' ]
    tmp = lines[GPU_DEVICE].split(',')
    global GPU_resoures_list
    GPU_resoures_list.append([line for line in tmp])

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def print_get_gpu_info():
    pprint.pprint(get_gpu_info())


book_name_xls = 'AlexnetQuanti.xls'
sheet_name_xls = 'Quanti'
value_title = [["Time_iteraion", "Time_test", "GPU_resoures_list",], ]
import xlrd
import xlwt
from xlutils.copy import copy as xlutils_copy


def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("xls格式表格写入数据成功！")


def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = xlutils_copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    # for i in range(0, index):
    #     for j in range(0, len(value[i])):
    #         new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入

    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")


def read_excel_xls(path):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  # 逐行逐列读取数据
        print()
# Training
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss = torch.nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

def main():

    All_time = time.time()

    print("main_without_quanti")
    # batch_size = 64
    batch_size = 100
    # test_batch_size = 64
    test_batch_size = 100
    epochs = 20
    lr = 0.01
    momentum = 0.9
    seed = 1
    log_interval = 500
    save_model = False
    no_cuda = False

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")


    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(size=224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../../data', train=False, transform=transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    for X, y in train_loader:
        print(X.size(), y.size())
        break
    # model = Net().to(device)

    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    #打印模型的状态字典
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 打印优化器的状态字典
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    Time_End = time.time() - All_time
    print("iteration_time", Time_iteraion, "\nepoch_time", Time_epoch, "\nmain_quanti endtime", Time_End)

    return model

# model = main()

########################################################################################################################

# Quantization Function

from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val,num_bits=8):
  # Calc Scale and zero point of next 
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale
  
  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point

def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


# Rework Forward pass of Linear and Conv Layers to support Quantisation
def quantizeLayer(x, layer, stat, scale_x, zp_x):
  # for both conv and linear layers

  # cache old values
  W = layer.weight.data
  B = layer.bias.data

  # quantise weights, activations are already quantised
  w = quantize_tensor(layer.weight.data) 
  b = quantize_tensor(layer.bias.data)

  layer.weight.data = w.tensor.float()
  layer.bias.data = b.tensor.float()

  # This is Quantisation Artihmetic
  scale_w = w.scale
  zp_w = w.zero_point
  scale_b = b.scale
  zp_b = b.zero_point
  
  scale_next, zero_point_next = calcScaleZeroPoint(min_val=stat['min'], max_val=stat['max'])

  # Preparing input by shifting
  X = x.float() - zp_x
  layer.weight.data = scale_x * scale_w*(layer.weight.data - zp_w)
  layer.bias.data = scale_b*(layer.bias.data + zp_b)

  # All int computation
  x = (layer(X)/ scale_next) + zero_point_next 
  
  # Perform relu too
  x = F.relu(x)

  # Reset weights for next forward pass
  layer.weight.data = W
  layer.bias.data = B
  
  return x, scale_next, zero_point_next


  # Get Max and Min Stats for Quantising Activations of Network.

  # Get Min and max of x tensor, and stores it

def updateStats(x, stats, key):
  max_val, _ = torch.max(x, dim=1)
  min_val, _ = torch.min(x, dim=1)
  
  
  if key not in stats:
    stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
  else:
    stats[key]['max'] += max_val.sum().item()
    stats[key]['min'] += min_val.sum().item()
    stats[key]['total'] += 1
  
  return stats

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
  x = F.relu(model.conv1(x))
  x = F.max_pool2d(x, 3, 2)
  
  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
  x = F.relu(model.conv2(x))
  x = F.max_pool2d(x, 3, 2)

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')
  x = F.relu(model.conv3(x))

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')
  x = F.relu(model.conv4(x))

  stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')
  x = F.relu(model.conv5(x))
  x = F.max_pool2d(x, 3, 2)

  x = x.view(-1, 256 * 6 * 6)
  
  stats = updateStats(x, stats, 'fc1')
  x = F.relu(model.fc1(x))
  x = F.dropout(x, 0.5)

  stats = updateStats(x, stats, 'fc2')
  x = F.relu(model.fc2(x))
  x = F.dropout(x, 0.5)

  stats = updateStats(x, stats, 'fc3')

  x = model.fc3(x)

  return stats

# Entry function to get stats of all functions.
def gatherStats(device, model, test_loader):
    # device = 'cuda'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    test_loss = 0
    correct = 0
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)
            # break;
            
    final_stats = {}
    for key, value in stats.items():
      final_stats[key] = { "max" : value["max"] / value["total"], "min" : value["min"] / value["total"] }

    # torch.cuda.empty_cache()
    return final_stats


# Forward Pass for Quantised Inference
def quantForward(model, x, device, stats):
  
  # Quantise before inputting into incoming layers
  x  = x.to(device)
  x = quantize_tensor(x, min_val=stats['conv1']['min'], max_val=stats['conv1']['max'])

  x, scale_next, zero_point_next = quantizeLayer(x.tensor, model.conv1, stats['conv2'], x.scale, x.zero_point)
  x = F.max_pool2d(x, 3, 2)
  
  x, scale_next, zero_point_next = quantizeLayer(x, model.conv2, stats['conv3'], scale_next, zero_point_next)
  x = F.max_pool2d(x, 3, 2)

  x, scale_next, zero_point_next = quantizeLayer(x, model.conv3, stats['conv4'], scale_next, zero_point_next)

  x, scale_next, zero_point_next = quantizeLayer(x, model.conv4, stats['conv5'], scale_next, zero_point_next)

  x, scale_next, zero_point_next = quantizeLayer(x, model.conv5, stats['fc1'], scale_next, zero_point_next)
  x = F.max_pool2d(x, 3, 2)

  x = x.view(-1, 256 * 6 * 6)

  x, scale_next, zero_point_next = quantizeLayer(x, model.fc1, stats['fc2'], scale_next, zero_point_next)
  x = F.dropout(x, 0.5)

  x, scale_next, zero_point_next = quantizeLayer(x, model.fc2, stats['fc3'], scale_next, zero_point_next)
  x = F.dropout(x, 0.5)
  
  # Back to dequant for final layer
  x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
   
  x = model.fc3(x)

  return F.log_softmax(x, dim=1)


# Testing Function for Quantisation
def testQuant(model, test_loader, quant=False, stats=None):
    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            if quant:
              output = quantForward(model, data, stats)
            else:
              output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Get Accuracy of Non Quantised Model
import copy
# q_model = copy.deepcopy(model)

# kwargs = {'num_workers': 0, 'pin_memory': True}
# test_loader = torch.utils.data.DataLoader(
#     datasets.FashionMNIST('../../data', train=False, transform=transforms.Compose([
#                        transforms.Resize(size=224),
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=60, shuffle=True, **kwargs)
#
# print("testQuant(q_model, test_loader, quant=False)")
# testQuant(q_model, test_loader, quant=False)



# Gather Stats of Activations
# stats = gatherStats(q_model, test_loader)
# print(stats)
#
# testQuant(q_model, test_loader, quant=True, stats=stats)

Count_quantiForward = 0
Count_normalForward = 0
Count_Miniiterration = 0
quanti_time_Before = False
quanti_time_After =  True
# test_forward_quanti, test_backward_quanti, test_for_noquanti, test_back_noquanti = 0,0,0,0
# Quantization Training
def train_quanti_shuffle(args, model, device, train_loader, test_loader, optimizer, epoch, quanti,):
    model.train()
    global Time_iteraion, Count_quantiForward, Count_normalForward, Count_Miniiterration, Time_test

    Time_epoch_start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):

        Count_Miniiterration += 1
        quanti_rand = random.random()
        iteraion_time = time.time()

        # import ipdb
        # ipdb.set_trace()  # 相当于添加断点

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if quanti :
            stats = gatherStats(device, model, test_loader)
            # print(stats)
            output = quantForward(model, data, device, stats)
            Count_quantiForward += 1
            # print("output = quantForward(model, data, stats)")
        else:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            Count_normalForward += 1

        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()
        # Time_iteraion.append(time.time() - iteraion_time)
        # print(time.time() - iteraion_time)
        # get_gpu_info()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

            # 加大测试点,epoch一次的test点accuracy看不太出来
            Time_iteraion.append(time.time() - iteraion_time)
            get_gpu_info()
            Time_test_start = time.time()
            acc = test(args, model, device, test_loader)
            Test_accuracy.append(acc)
            Test_loss.append(loss.item())
            Time_test.append(time.time() - Time_test_start)

    Time_epoch.append(time.time() - Time_epoch_start)

def main_quanti_shuffle(quanti, quanti_proportion=1, quanti_time=0):
    # quanti_proportion:是量化的比例,随机的
    # quanti_time: 量化的时机
    global Count_quantiForward, Count_normalForward, Count_Miniiterration
    All_time = time.time()

    print("main_quanti_time:",quanti)
    # batch_size = 64
    batch_size = 1000
    # test_batch_size = 64
    test_batch_size = batch_size
    epochs = 50
    lr = 0.01
    momentum = 0.9
    seed = 1
    log_interval = 10
    save_model = False
    no_cuda = False
    root = '../data'
    # root = '../../data'
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.cuda.set_device(GPU_DEVICE)
    device = torch.cuda.current_device()
    print("current_device: ", device)
    torch.manual_seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # file = open("shuffleQrecond:%s-prop:%s-b:%d-e:%d.txt" % (quanti, quanti_proportion, batch_size, epochs), "w+")
    # Information
    print("batch_size:{}\tepochs:{}\tlr:{:0.6f}\tmomentum:{:0.2f}\tlog_interval:{}\tquanti_proportion:{:0.2f}".format(
        batch_size, epochs, lr, momentum, log_interval,quanti_proportion))



    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root=root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(size=224),
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root=root, train=False, transform=transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # for X, y in train_loader:
    #     print(X.size(), y.size())
    #     break
    # model = Net().to(device)

    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    milestones = [30]
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    #打印模型的状态字典
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        # print(param_tensor, "\t", model.state_dict()[param_tensor])
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 打印优化器的状态字典
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    args = {}
    args["log_interval"] = log_interval

    epoch_num = 0

    test_forward_quanti, test_backward_quanti, test_for_noquanti, test_back_noquanti = 0,0,0,0
    for epoch in range(1, epochs + 1):
        epoch_num  += 1

        if  quanti_time == quanti_time_Before:  # 前面百分比量化
            if (epochs + 1) * (quanti_proportion) > epoch_num:
                train_quanti_shuffle(args, model, device, train_loader, test_loader, optimizer, epoch,
                                  quanti=True, )
                test_forward_quanti += 1
                # print("test_forward_quanti",test_forward_quanti)
            else:
                train_quanti_shuffle(args, model, device, train_loader, test_loader, optimizer, epoch,
                                  quanti=False, )
                test_for_noquanti += 1
                # print("test_for_noquanti", test_for_noquanti)
        elif  quanti_time == quanti_time_After:  # 后面百分比量化
            if (epochs + 1) * (1 - quanti_proportion) < epoch_num:
                train_quanti_shuffle(args, model, device, train_loader, test_loader, optimizer, epoch,
                                  quanti=True, )
                test_backward_quanti += 1
                # print("test_backward_quanti", test_backward_quanti)
            else:
                train_quanti_shuffle(args, model, device, train_loader, test_loader, optimizer, epoch,
                                  quanti=False, )
                test_back_noquanti += 1
                # print("test_back_noquanti", test_back_noquanti)
        else:pass
        # test(args, model, device, test_loader)
        scheduler_lr.step()

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    Time_End = time.time() - All_time
    print("iteration_time", Time_iteraion, "\nepoch_time", Time_epoch, "\nmain_quanti endtime", Time_End, "\tGPU mem",GPU_resoures_list
          ,"\nTest_loss", Test_loss, "\nTest_accuracy", Test_accuracy)
    # print("iteration_time", Time_iteraion, "\nepoch_time", Time_epoch, "\nmain_quanti endtime", Time_End, "\tGPU mem",GPU_resoures_list,file=file)
    print("Count_quantiForward:{} Count_normalForward:{} Count_miniIteration:{}".format(
        Count_quantiForward, Count_normalForward, Count_Miniiterration ))
    print("test_forward_quanti:{}\ttest_for_noquanti:{}\ttest_backward_quanti:{}\ttest_back_noquanti:{}".format(
        test_forward_quanti, test_for_noquanti, test_backward_quanti,test_back_noquanti))

    # Time_iteraion = [9.235993146896362, 8.786091089248657, 0.17667675018310547]
    # write_excel_xls(book_name_xls, sheet_name_xls, value_title)
    # write_excel_xls_append(book_name_xls, Time_iteraion)
    # write_excel_xls_append(book_name_xls, GPU_resoures_list)
    # write_excel_xls_append(book_name_xls, Test_accuracy)
    # write_excel_xls_append(book_name_xls, Test_loss)
    # read_excel_xls(book_name_xls)
    return model

# main_quanti_shuffle(quanti=True, quanti_proportion=0.25, quanti_time=quanti_time_Before)
# main_quanti_shuffle(quanti=True, quanti_proportion=0.25, quanti_time=quanti_time_After)
# main_quanti_shuffle(quanti=True, quanti_proportion=0.5, quanti_time=quanti_time_Before)
# main_quanti_shuffle(quanti=True, quanti_proportion=0.5, quanti_time=quanti_time_After)
# main_quanti_shuffle(quanti=True, quanti_proportion=0.75, quanti_time=quanti_time_Before)
# main_quanti_shuffle(quanti=True, quanti_proportion=0.75, quanti_time=quanti_time_After)

# import pandas as pd
# pd.set_option('display.max_rows', 500)  #最大行数
# pd.set_option('display.max_columns', 500)    #最大列数
# pd.set_option('display.width', 4000)        #页面宽度


def train_quanti(args, model, device, train_loader, test_loader, optimizer, epoch, quanti, quanti_proportion):

    model.train()

    global Time_iteraion, Count_quantiForward, Count_normalForward, Count_Miniiterration, Time_test

    Time_epoch_start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):

        Count_Miniiterration += 1
        quanti_rand = random.random()
        iteraion_time = time.time()

        # import ipdb
        # ipdb.set_trace()  # 相当于添加断点

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if quanti and  quanti_rand > (1 - quanti_proportion):

            stats = gatherStats(device, model, test_loader)
            # print(stats)
            output = quantForward(model, data, device, stats)
            Count_quantiForward += 1
            # print("output = quantForward(model, data, stats)")
        else:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            Count_normalForward += 1


        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()
        # Time_iteraion.append(time.time() - iteraion_time)
        # print(time.time() - iteraion_time)
        # get_gpu_info()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} {}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item(),
            #     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), file=file)
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size(),
            #           "\t", model.state_dict()[param_tensor], file=file)

            #加大测试点,epoch一次的test点accuracy看不太出来
            Time_iteraion.append(time.time() - iteraion_time)
            # get_gpu_info()
            Time_test_start = time.time()
            acc =  test(args, model, device, test_loader)
            Test_accuracy.append(acc)
            Test_loss.append(loss.item())
            Time_test.append(time.time() - Time_test_start)

    Time_epoch.append(time.time() - Time_epoch_start)


def main_quanti(quanti, quanti_proportion=1, quanti_time=0):
    # quanti_proportion:是量化的比例,随机的
    # quanti_time: 量化的时机
    global Count_quantiForward, Count_normalForward, Count_Miniiterration
    All_time = time.time()

    print("main_quanti:",quanti)
    # batch_size = 64
    batch_size = 256
    # test_batch_size = 64
    test_batch_size = batch_size
    epochs = 10
    lr = 0.01
    momentum = 0.9
    seed = 1
    log_interval = 10
    save_model = False
    no_cuda = False
    root = '../data'
    # root = '../../data'
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.cuda.set_device(GPU_DEVICE)
    device = torch.cuda.current_device()
    print("current_device: ", device)
    torch.manual_seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # file = open("Qrecond:%s-prop:%s-b:%d-e:%d.txt"%(quanti,quanti_proportion, batch_size, epochs), "w+")
    # Information
    print("batch_size:{}\tepochs:{}\tlr:{:0.6f}\tmomentum:{:0.2f}\tlog_interval:{}\tquanti_proportion:{:0.2f}".format(
        batch_size, epochs, lr, momentum, log_interval,quanti_proportion))
    # print("batch_size:{}\tepochs:{}\tlr:{:0.6f}\tmomentum:{:0.2f}\tlog_interval:{}\tquanti_proportion:{:0.2f}".format(
    #     batch_size, epochs, lr, momentum, log_interval,quanti_proportion), file=file)


    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root=root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(size=224),
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root=root, train=False, transform=transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # for X, y in train_loader:
    #     print(X.size(), y.size())
    #     break
    # model = Net().to(device)

    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    milestones = [30]
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    #打印模型的状态字典
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        # print(param_tensor, "\t", model.state_dict()[param_tensor])
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 打印优化器的状态字典
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    args = {}
    args["log_interval"] = log_interval

    for epoch in range(1, epochs + 1):
        train_quanti(args, model, device, train_loader, test_loader, optimizer, epoch,
                     quanti=quanti, quanti_proportion=quanti_proportion)
        # train_quanti(args, model, device, train_loader, test_loader, optimizer, epoch,
        #              quanti=quanti, quanti_proportion=quanti_proportion,file=file)
        # test(args, model, device, test_loader)
        scheduler_lr.step()

    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    Time_End = time.time() - All_time
    print("iteration_time", Time_iteraion, "\nepoch_time", Time_epoch, "\nmain_quanti endtime", Time_End, "\nGPU mem",GPU_resoures_list
          ,"\nTest_loss", Test_loss, "\nTest_accuracy", Test_accuracy)
    # print("iteration_time", Time_iteraion, "\nepoch_time", Time_epoch, "\nmain_quanti endtime", Time_End, "\tGPU mem",GPU_resoures_list
    #       ,"\nTest_loss", Test_loss, "\nTest_accuracy", Test_accuracy, file=file)

    print("Count_quantiForward:{} Count_normalForward:{} Count_miniIteration:{}".format(
        Count_quantiForward, Count_normalForward, Count_Miniiterration ))
    # print("Count_quantiForward:{} Count_normalForward:{} Count_miniIteration:{}".format(
    #     Count_quantiForward, Count_normalForward, Count_Miniiterration ), file=file)
    # Time_iteraion = [9.235993146896362, 8.786091089248657, 0.17667675018310547]
    # write_excel_xls(book_name_xls, sheet_name_xls, value_title)
    # write_excel_xls_append(book_name_xls, Time_iteraion)
    # write_excel_xls_append(book_name_xls, GPU_resoures_list)
    # write_excel_xls_append(book_name_xls, Test_accuracy)
    # write_excel_xls_append(book_name_xls, Test_loss)
    # read_excel_xls(book_name_xls)
    return model



# for i in range(100):
#     print_get_gpu_info()
# print(GPU_resoures_list)
# model = main_quanti(quanti=False)
quanti_model = main_quanti(quanti=True, quanti_proportion=1)
# quanti_model = main_quanti(quanti=True, quanti_proportion=0.25)
# quanti_model = main_quanti(quanti=True, quanti_proportion=0.5)
# quanti_model = main_quanti(quanti=True, quanti_proportion=0.75)
# Test Quantised Inference Of Model
