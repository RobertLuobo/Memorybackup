import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
sys.path.append("..")

# import nvidia_smi

GPU_DEVICE = 1
batch_size = 50
lr, num_epochs = 0.01, 100
log_interval = 10

# Select_dataset = 'MNIST'
# Select_dataset = 'Fashion_mnist'
Select_dataset = 'CIFAR10'

torch.cuda.set_device(GPU_DEVICE)
current_device = torch.cuda.current_device()
print("current_device: ", current_device)
print("lr = %0.4f batch_size = %d log_interval:%d Select_dataset=%s" %  (lr, batch_size, log_interval, Select_dataset))
# import myModel as model
# import inspect

# from PytorchMemoryUtils.gpu_mem_track import MemTracker
# from PytorchMemoryUtils.modelsize_estimate import modelsize
# frame = inspect.currentframe()          # define a frame to track
# gpu_tracker = MemTracker(frame)         # define a GPU tracker
# import GPUtil


import subprocess
import json
import pprint
import numpy as np
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
GPU_resoures_list = []

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
    pass



# def load_data_fashion_mnist(batch_size, resize=None, root='../../cifar10/data'):
def load_data(batch_size, resize=None, root='../data'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    # not normalization

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

    elif Select_dataset == 'MNIST':
        mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

        train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    else:   pass


    print("batch_size:", batch_size, "\tDataset:",Select_dataset,
          "\n%s train dataset size:%d" % (Select_dataset, mnist_train.__len__()),
          "\ttrain dataset iteration number:",train_iter.__len__(), "\t\tTotal:", train_iter.__len__()*batch_size,
          "\n%s test dataset size:%d" % (Select_dataset, mnist_test.__len__()),
          "\ttest dataset iteration number:",test_iter.__len__(), "\t\tTotal:", test_iter.__len__()*batch_size)

    return train_iter, test_iter


Time_iteration= []
epoch_time = []
Test_acc = []
Train_acc= []

def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):

    print("training on ", device)

    iteration_num = 0
    global iteration_time, epoch_time

    # gpu_tracker.track()  # run function between the code line where uses GPU
    net = net.to(device)

    # basic_gpu_memory = get_gpu_memory()

    All_time = time.time()

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, Time_epoch_start = 0.0, 0.0, 0, 0, time.time()

        test1234556 = 0
        # import ipdb
        # ipdb.set_trace()  # 相当于添加断点
        iteraion_time = time.time()

        for X, y in train_iter:

            get_gpu_info()

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

            # torch.cuda.empty_cache()
            

            if iteration_num % log_interval == 0:

                get_gpu_info()

                print('iteraion %d, loss %.4f, train acc %.3f, , time %.5f sec'
                  % (iteration_num + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - iteraion_time), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), sep="\t")


                Time_iteration.append(time.time() - iteraion_time)

                iteraion_time = time.time()
                # print("lr = %0.4f train_l_sum1 = %0.4f train_acc_sum1=%0.4f n = %d, batch_count = %d" %
                #       (lr, train_l_sum, train_acc_sum, n, batch_count))


            # test1234556 +=1
            # if(test1234556 > 3):break
        # scheduler.step()
        test_acc = evaluate_accuracy(test_iter, net)

        Test_acc.append(test_acc)
        Train_acc.append(train_acc_sum / n)
        epoch_time.append(time.time() - Time_epoch_start)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - Time_epoch_start))
        
        if test_acc >= 0.95:
            print("est_acc >= 0.95, epoch = ",epoch)
            break 

        # tool.print_get_gpu_info()

    print("Train end time %.2f sec" % (time.time() - All_time))

    print("iteration_time = ",iteration_time, "epoch_time = ", epoch_time, "test_acc = ", Test_acc, "train_acc = ", Train_acc)

    

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
    # 0. Hyper-Parameter:learning rate, batthsize, epoch, scheduer_lr
    # 1. define the model net
    # 2. Dataloader
    # 3. choose a optimer，loss_function
    # 4. Train step
    # 5. evaluate step
    # 6. save the train info and draw the figure
    # 7. 模型的参数大小
    # 8.

    
    Time_Whole_start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 0. Hyper-Parameter


    net = torchvision.models.vgg19(pretrained=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)     # 设置学习率下降策略
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    get_gpu_info()

    train_iter, test_iter = load_data(batch_size, 224)
    get_gpu_info()

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
    from torchsummary import summary
    net.to(device)  #这句不加的话,model和参数放的位置会不对应
    if Select_dataset == 'Fashion_mnist':
        print(summary(model=net, input_size=(1, 224, 224), batch_size=batch_size ,device="cuda")) # model, input_size(channel, H, W), batch_size, device
    elif Select_dataset == 'CIFAR10':
        print(summary(model=net, input_size=(3, 224, 224), batch_size=batch_size, device="cuda"))  # model, input_size(channel, H, W), batch_size, device
    else:pass

    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
    # plt_label = 'Memory-iteration, ' + 'batchsize = %d'%batch_size + ', epoch= %d'%num_epochs
    # plt_title = '%s - Alexnet' % Select_dataset
    # plt.figure(1)  # 创建图表1
    # x_value = np.arange(tool.GPU_resoures_list.__len__())
    # plt.plot(x_value, np.array(tool.GPU_resoures_list).transpose()[:][0].astype(np.float16), c='coral', linestyle='-', label=plt_label)
    # plt.xlabel("iteration")
    # plt.ylabel("Memory used(MB)")
    # # plt.title('FashionMNSIT - Alexnet')
    # plt.title(plt_title)
    # plt.legend(loc='lower right')
    # plt.show()
    #
    # plt_label = 'GPU Utilization-iteration, ' + 'batchsize = %d' % batch_size + ', epoch= %d' % num_epochs
    # # plt_title = '%s - Alexnet' % Select_dataset + 'GPU Utilization( % ), ' + 'batchsize = %d' % batch_size + ', epoch= %d' % num_epochs
    # plt.figure(2)  # 创建图表1
    # plt.plot(x_value, np.array(tool.GPU_resoures_list).transpose()[:][1].astype(np.float16), c='green', linestyle='-', label=plt_label)
    # plt.xlabel("iteration")
    # plt.ylabel("GPU Utilization( % )")
    # # plt.title('FashionMNSIT - Alexnet')
    # plt.title(plt_title)
    # plt.legend(loc='lower right')
    # plt.show()
    # Utilization的纵坐标数值不对 是因为没有把数值转化成float格式
   

    plt_label = ' SinglejobAlexnet:Memory-iteration, ' + 'batchsize = %d'%batch_size + ', N_iter= %d'%num_epochs
    plt_title = '%s - Singlejob:VGG16' % Select_dataset
    plt.figure(1)  # 创建图表1
    x_value = np.arange(tool.GPU_resoures_list.__len__())
    plt.plot(x_value, np.array(tool.GPU_resoures_list).transpose()[:][0].astype(np.float16), c='coral', linestyle='-', label=plt_label)
    # plt.xlabel("")
    plt.ylabel("Memory used(MB)")
    # plt.title('FashionMNSIT - Alexnet')
    plt.title(plt_title)
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig("../svg/"+plt_label+".svg")


    print( "Singlejob Train time: %.4f" %(time.time() -  Time_Whole_start))



'''
VGG16 - MNIST:    没有进行权值初始化
lr=0.01,batchsize=100

'''
'''
AlexnVGG16et - CIFAR10:   没有进行权值初始化
lr=0.01,batchsize=100
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,)


'''
'''
VGG16 - CIFAR10:   

'''