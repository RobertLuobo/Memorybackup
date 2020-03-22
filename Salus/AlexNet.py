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

GPU_DEVICE = 0
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
        scheduler.step()
        test_acc = evaluate_accuracy(test_iter, net)

        Test_acc.append(test_acc)
        Train_acc.append(train_acc_sum / n)
        epoch_time.append(time.time() - Time_epoch_start)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - Time_epoch_start))
        
        if test_acc >= 0.985:
            print("est_acc >= 0.985, epoch = ",epoch)
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
    from torchsummary import summary
    net.to(device)  #这句不加的话,model和参数放的位置会不对应
    if Select_dataset == 'Fashion_mnist':
        print(summary(model=net, input_size=(1, 224, 224), batch_size=batch_size ,device="cuda")) # model, input_size(channel, H, W), batch_size, device
    elif Select_dataset == 'CIFAR10':
        print(summary(model=net, input_size=(3, 224, 224), batch_size=batch_size, device="cuda"))  # model, input_size(channel, H, W), batch_size, device
    else:pass
    
    Time_Whole_start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 0. Hyper-Parameter


    net = torchvision.models.AlexNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)     # 设置学习率下降策略
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
   
    # from torchsummary import summary
    # net.to(device)  #这句不加的话,model和参数放的位置会不对应
    # if Select_dataset == 'Fashion_mnist':
    #     print(summary(model=net, input_size=(1, 224, 224), batch_size=batch_size ,device="cuda")) # model, input_size(channel, H, W), batch_size, device
    # elif Select_dataset == 'CIFAR10':
    #     print(summary(model=net, input_size=(3, 224, 224), batch_size=batch_size, device="cuda"))  # model, input_size(channel, H, W), batch_size, device
    # else:pass
    # print("Max iteration time % 0.5f" % max(iteration_time),"\tMin iteration time % 0.5f" % min(iteration_time),
    #     "\tMax epoch time % 0.5f" % max(epoch_time),        "\t\tMin epoch time % 0.5f" % min(epoch_time))
    #
    plt_label = ' SinglejobAlexnet:Memory-iteration, ' + 'batchsize = %d'%batch_size + ', N_iter= %d'%num_epochs
    plt_title = '%s - Singlejob:Alexnet' % Select_dataset
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
Alexnet - MNIST:    没有进行权值初始化
lr=0.01,batchsize=100
poch 1, loss 0.8241, train acc 0.709, test acc 0.982, time 89.9 sec
epoch 2, loss 0.0678, train acc 0.979, test acc 0.990, time 92.7 sec
epoch 3, loss 0.0446, train acc 0.986, test acc 0.991, time 93.7 sec
epoch 4, loss 0.0335, train acc 0.990, test acc 0.991, time 93.7 sec
epoch 5, loss 0.0273, train acc 0.991, test acc 0.993, time 94.2 sec
'''
'''
Alexnet - CIFAR10:   没有进行权值初始化
lr=0.01,batchsize=100
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,)
epoch 1, loss 2.9632, train acc 0.101, test acc 0.100, time 124.3 sec
epoch 2, loss 2.3244, train acc 0.099, test acc 0.100, time 126.4 sec
epoch 3, loss 2.3118, train acc 0.109, test acc 0.146, time 132.6 sec
epoch 4, loss 2.0061, train acc 0.238, test acc 0.324, time 128.2 sec
epoch 5, loss 1.6880, train acc 0.359, test acc 0.447, time 126.3 sec
epoch 6, loss 1.4357, train acc 0.467, test acc 0.533, time 126.4 sec
epoch 7, loss 1.2434, train acc 0.549, test acc 0.600, time 124.7 sec    
epoch 8, loss 1.0684, train acc 0.619, test acc 0.655, time 126.7 sec
epoch 9, loss 0.9109, train acc 0.680, test acc 0.704, time 130.4 sec
epoch 10, loss 0.7832, train acc 0.727, test acc 0.741, time 129.3 sec
epoch 11, loss 0.6910, train acc 0.760, test acc 0.750, time 122.5 sec
epoch 12, loss 0.6107, train acc 0.788, test acc 0.764, time 122.5 sec
epoch 13, loss 0.5414, train acc 0.811, test acc 0.777, time 121.7 sec
epoch 14, loss 0.4756, train acc 0.833, test acc 0.786, time 123.9 sec
epoch 15, loss 0.4213, train acc 0.852, test acc 0.805, time 121.9 sec
epoch 16, loss 0.3784, train acc 0.867, test acc 0.798, time 121.2 sec
epoch 17, loss 0.3291, train acc 0.883, test acc 0.809, time 121.1 sec
epoch 18, loss 0.2974, train acc 0.894, test acc 0.809, time 120.8 sec
epoch 19, loss 0.2660, train acc 0.906, test acc 0.814, time 121.9 sec
epoch 20, loss 0.2428, train acc 0.914, test acc 0.810, time 120.9 sec
epoch 21, loss 0.2166, train acc 0.924, test acc 0.813, time 120.3 sec
epoch 22, loss 0.1940, train acc 0.931, test acc 0.817, time 120.4 sec
epoch 23, loss 0.1741, train acc 0.938, test acc 0.815, time 120.2 sec
epoch 24, loss 0.1605, train acc 0.944, test acc 0.816, time 120.6 sec
epoch 25, loss 0.1543, train acc 0.946, test acc 0.818, time 120.8 sec
epoch 26, loss 0.1413, train acc 0.950, test acc 0.819, time 119.9 sec
epoch 27, loss 0.1347, train acc 0.954, test acc 0.814, time 120.3 sec
epoch 28, loss 0.1258, train acc 0.957, test acc 0.823, time 119.5 sec
epoch 29, loss 0.1127, train acc 0.962, test acc 0.823, time 120.3 sec
epoch 30, loss 0.1010, train acc 0.965, test acc 0.826, time 120.3 sec
epoch 31, loss 0.0997, train acc 0.967, test acc 0.822, time 120.0 sec
epoch 32, loss 0.0949, train acc 0.969, test acc 0.822, time 120.9 sec
epoch 33, loss 0.0928, train acc 0.969, test acc 0.823, time 119.4 sec
epoch 34, loss 0.0851, train acc 0.972, test acc 0.827, time 121.1 sec
epoch 35, loss 0.0781, train acc 0.974, test acc 0.826, time 120.1 sec
epoch 36, loss 0.0842, train acc 0.971, test acc 0.827, time 120.3 sec
epoch 37, loss 0.0693, train acc 0.976, test acc 0.826, time 121.5 sec
epoch 38, loss 0.0716, train acc 0.977, test acc 0.826, time 123.5 sec
epoch 39, loss 0.0615, train acc 0.980, test acc 0.834, time 120.5 sec
epoch 40, loss 0.0675, train acc 0.978, test acc 0.832, time 120.7 sec
epoch 41, loss 0.0613, train acc 0.979, test acc 0.837, time 115.8 sec
epoch 42, loss 0.0603, train acc 0.980, test acc 0.828, time 117.3 sec
epoch 43, loss 0.0579, train acc 0.981, test acc 0.832, time 123.1 sec

'''
'''
Alexnet - CIFAR10:   进行kaiming权值初始化
lr=0.01,batchsize=100
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,)

epoch 1, loss 2.0154, train acc 0.263, test acc 0.431, time 205.8 sec
epoch 2, loss 1.4272, train acc 0.477, test acc 0.575, time 176.2 sec
epoch 3, loss 1.1889, train acc 0.574, test acc 0.615, time 160.7 sec
epoch 4, loss 1.0211, train acc 0.638, test acc 0.673, time 134.7 sec
epoch 5, loss 0.9113, train acc 0.676, test acc 0.698, time 129.5 sec
epoch 6, loss 0.8347, train acc 0.705, test acc 0.711, time 128.4 sec
epoch 7, loss 0.7516, train acc 0.736, test acc 0.743, time 127.9 sec
epoch 8, loss 0.7004, train acc 0.753, test acc 0.748, time 128.5 sec
epoch 9, loss 0.6332, train acc 0.779, test acc 0.764, time 128.1 sec
epoch 10, loss 0.5962, train acc 0.791, test acc 0.768, time 127.9 sec
epoch 11, loss 0.5474, train acc 0.809, test acc 0.758, time 127.8 sec
epoch 12, loss 0.5139, train acc 0.819, test acc 0.766, time 126.7 sec
epoch 13, loss 0.4638, train acc 0.836, test acc 0.777, time 126.3 sec
epoch 14, loss 0.4292, train acc 0.848, test acc 0.777, time 127.0 sec
epoch 15, loss 0.4042, train acc 0.858, test acc 0.788, time 127.5 sec
epoch 16, loss 0.3745, train acc 0.869, test acc 0.778, time 127.9 sec
epoch 17, loss 0.3468, train acc 0.878, test acc 0.791, time 127.2 sec
epoch 18, loss 0.3199, train acc 0.886, test acc 0.790, time 127.5 sec
epoch 19, loss 0.2986, train acc 0.894, test acc 0.784, time 126.5 sec
epoch 20, loss 0.2782, train acc 0.903, test acc 0.786, time 126.2 sec
epoch 21, loss 0.2643, train acc 0.908, test acc 0.791, time 125.0 sec
epoch 22, loss 0.2553, train acc 0.911, test acc 0.790, time 126.6 sec
epoch 23, loss 0.2384, train acc 0.917, test acc 0.792, time 126.2 sec
epoch 24, loss 0.2196, train acc 0.924, test acc 0.787, time 124.9 sec
epoch 25, loss 0.2079, train acc 0.928, test acc 0.799, time 124.8 sec
epoch 26, loss 0.1985, train acc 0.929, test acc 0.799, time 125.6 sec
epoch 27, loss 0.1908, train acc 0.932, test acc 0.799, time 126.5 sec
epoch 28, loss 0.1779, train acc 0.937, test acc 0.805, time 125.4 sec
epoch 29, loss 0.1793, train acc 0.938, test acc 0.803, time 127.2 sec
epoch 30, loss 0.1700, train acc 0.941, test acc 0.802, time 129.2 sec
epoch 31, loss 0.1606, train acc 0.945, test acc 0.799, time 129.7 sec
epoch 32, loss 0.1428, train acc 0.951, test acc 0.802, time 129.5 sec
epoch 33, loss 0.1468, train acc 0.949, test acc 0.794, time 128.4 sec
epoch 34, loss 0.1418, train acc 0.951, test acc 0.794, time 128.2 sec
epoch 35, loss 0.1355, train acc 0.953, test acc 0.793, time 129.4 sec
epoch 36, loss 0.1314, train acc 0.955, test acc 0.777, time 129.9 sec
epoch 37, loss 0.1252, train acc 0.956, test acc 0.786, time 128.9 sec
epoch 38, loss 0.1255, train acc 0.958, test acc 0.783, time 128.0 sec
epoch 39, loss 0.1188, train acc 0.960, test acc 0.801, time 128.2 sec
epoch 40, loss 0.1129, train acc 0.962, test acc 0.796, time 127.4 sec
epoch 41, loss 0.1099, train acc 0.963, test acc 0.802, time 126.7 sec
epoch 42, loss 0.1030, train acc 0.966, test acc 0.805, time 124.7 sec
epoch 43, loss 0.0996, train acc 0.966, test acc 0.803, time 126.7 sec
epoch 44, loss 0.0966, train acc 0.968, test acc 0.800, time 126.6 sec
epoch 45, loss 0.0966, train acc 0.967, test acc 0.801, time 129.0 sec
epoch 46, loss 0.0961, train acc 0.967, test acc 0.801, time 130.4 sec
epoch 47, loss 0.0929, train acc 0.969, test acc 0.799, time 135.6 sec
'''