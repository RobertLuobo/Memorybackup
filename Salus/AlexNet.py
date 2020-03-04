# Purpose: to find out the peak and thought of GPU memory
# Tag: Salus , GPU cluster memory management
# Date:2020-2-22
# dataset: FashionMNIST
# Version: 1.0

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
import time
import numpy as np
# import nvidia_smi

import sys
sys.path.append("..")
import myTool as tool
import myModel as model

# import GPUtil

#记录nvidia显存使用
# nvidia_smi.nvmlInit()
# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

# loss_list = []
# gpu_memory_used = []
# gpu_memory = []
# gpu_memory_res =[]
# basic_gpu_memory = 0
# def get_gpu_memory():
#     res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
#     res_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
#     gpu_memory_res.append(res.gpu)
#     gpu_memory.append(res.memory)
#     gpu_memory_used.append(res_info.used)
#     print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%, gpu-used:{((res_info.used ))}M')

# Select_dataset = 'Fashion_mnist'
Select_dataset = 'CIFAR10'
def load_data_fashion_mnist(batch_size, resize=None, root='../data'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    if Select_dataset == 'Fashion_mnist':
        mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

        '''pin_memory(bool, optional)： 如果设置为True，那么dataloader将会在返回它们之前，
            将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中'''
        train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)
        test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

    elif Select_dataset == 'CIFAR10':
        mnist_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)
        test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)
    else:
        pass


    print("batch_size:", batch_size,
          "\tDataset:",Select_dataset,
          "\n%s train dataset size" % Select_dataset, mnist_train.__len__(),
          "\ttrain dataset iteration number:",train_iter.__len__(),
          "\t\tTotal:", train_iter.__len__()*batch_size,
          "\n%s test dataset size" % Select_dataset, mnist_test.__len__(),
          "\ttest dataset iteration number:",test_iter.__len__(),
          "\t\tTotal:", test_iter.__len__()*batch_size,
          sep="\t")

    return train_iter, test_iter


iteration_time = []
epoch_time = []
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    iteration_num = 0
    global iteration_time, epoch_time
    net = net.to(device)
    print("training on ", device)
    # basic_gpu_memory = get_gpu_memory()
    All_time = time.time()
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

        for X, y in train_iter:
            # interaion_time = time.time()

            # print("Before iteraion:", sep="\t")
            # tool.print_get_gpu_info()

            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            iteration_num +=1
            # iteration_time.append(time.time() - interaion_time)
            # print('interaion %d, loss %.4f, train acc %.3f, , time %.10f sec'
            #       % (iteration_num + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - interaion_time),sep="\t")

            # print("After interaion:",sep="\t")
            # tool.print_get_gpu_info()


        test_acc = evaluate_accuracy(test_iter, net)
        # epoch_time.append(time.time() - start)
        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
        #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

        # print("After epoch:",sep="\t")
        # tool.print_get_gpu_info()
    print("Train end time %.2f sec" % (time.time() - All_time))

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 如出现“out of memory”的报错信息，可减小batch_size或resize

if __name__ == '__main__':
    device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 0. Hyper-Parameter

    batch_size = 10
    lr, num_epochs = 0.001, 1
    net = model.AlexNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    #  还有iteration epoch 这两个参数根据batch size和dataset打大小来决定,
    tool.print_get_gpu_info()

    train_iter, test_iter = load_data_fashion_mnist(batch_size, 224)
    tool.print_get_gpu_info()

    plt_label = 'Memory-iteration, ' + 'batchsize = %d' % batch_size + ', epoch= %d' % num_epochs

    # for X_1, Y_1 in train_iter:
    #     print('X_1 =', X_1.shape,  '\nY_1 =', Y_1.type(torch.int32))
    #     break
    # print("Run at code line 172")
    # tool.print_get_gpu_info()
    #
    # for X_2, Y_2 in test_iter:
    #     print('X_2 =', X_2.shape,  '\nY_2 =', Y_2.type(torch.int32))
    #     break
    # tool.print_get_gpu_info()


    # train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    plt_label = 'Memory-iteration, ' + 'batchsize = %d'%batch_size + ', epoch= %d'%num_epochs
    plt_title = '%s - Alexnet' % Select_dataset
    plt.figure(1)  # 创建图表1
    x_value = np.arange(tool.GPU_resoures_list.__len__())
    plt.plot(x_value, np.array(tool.GPU_resoures_list).transpose()[:][0].astype(np.float16), c='coral', linestyle='-', label=plt_label)
    plt.xlabel("iteration")
    plt.ylabel("Memory used(MB)")
    # plt.title('FashionMNSIT - Alexnet')
    plt.title(plt_title)
    plt.legend(loc='lower right')
    plt.show()

    plt_label = 'GPU Utilization-iteration, ' + 'batchsize = %d' % batch_size + ', epoch= %d' % num_epochs
    # plt_title = '%s - Alexnet' % Select_dataset + 'GPU Utilization( % ), ' + 'batchsize = %d' % batch_size + ', epoch= %d' % num_epochs
    plt.figure(2)  # 创建图表1
    plt.plot(x_value, np.array(tool.GPU_resoures_list).transpose()[:][1].astype(np.float16), c='green', linestyle='-', label=plt_label)
    plt.xlabel("iteration")
    plt.ylabel("GPU Utilization( % )")
    # plt.title('FashionMNSIT - Alexnet')
    plt.title(plt_title)
    plt.legend(loc='lower right')
    plt.show()
    # Utilization的纵坐标数值不对 是因为没有把数值转化成float格式
    # 1.define the model net
    # 2. Dataloader
    # 3. choose a optimer
    # 4. Train step
    # 5.evaluate step
    # 6. draw the figure
    # 7. 模型的参数大小
    from torchsummary import summary
    net.to(device)  #这句不加的话,model和参数放的位置会不对应
    if Select_dataset == 'Fashion_mnist':
        print(summary(model=net, input_size=(1, 224, 224), batch_size=batch_size ,device="cuda")) # model, input_size(channel, H, W), batch_size, device
    elif Select_dataset == 'CIFAR10':
        print(summary(model=net, input_size=(3, 224, 224), batch_size=batch_size, device="cuda"))  # model, input_size(channel, H, W), batch_size, device
    else:pass
    print("Max iteration time % 0.5f" % max(iteration_time),"\tMin iteration time % 0.5f" % min(iteration_time),
        "\tMax epoch time % 0.5f" % max(epoch_time),        "\t\tMin epoch time % 0.5f" % min(epoch_time))

