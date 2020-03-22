# Purpose: to find out the peak and thought of GPU memory
# Tag: Salus , GPU cluster memory management
# Date:2020-3-04
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

import sys
sys.path.append("..")
import myTool as tool
import myModel as model

path_model_1 = "../pkl/Alexnet.pkl"
path_state_dict_1 = "../pkl/Alexnet_state_dict.pkl"

path_model_2 = "../pkl/VGG16.pkl"
path_state_dict_2 = "../pkl/VGG16_state_dict.pkl"

# Select_dataset = 'Fashion_mnist'
Select_dataset = 'CIFAR10'

def main():
    # 0. Hyper-Parameter
    test1, test2 = 0, 0
    iteration_num = 0
    iteration_time = []
    epoch_time = []
    batch_size = 10
    lr, num_epochs = 0.001, 1

    tool.print_get_gpu_info()

    train_iter, test_iter = load_data_fashion_mnist(batch_size, 224)

    tool.print_get_gpu_info()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on ", device)



    net2 = torchvision.models.vgg16(pretrained=False, progress=True).to(device)

    loss2 = torch.nn.CrossEntropyLoss()
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=lr)

    All_time = time.time()
    for epoch in range(num_epochs):
        train_l_sum1, train_acc_sum1, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        train_l_sum2, train_acc_sum2 = 0.0, 0.0

        for X, y in train_iter:

            '''训练任务2'''
            # net2 = torchvision.models.vgg16_bn(pretrained=False, progress=True).to(device)

            interaion_time = time.time()
            print("Net2 Before iteraion:%d" % iteration_num, sep="\t")

            test2 +=1
            # if iteration_num > 0 and test2 > 2:
            if iteration_num > 1 :
                test2 = 0
                net2 = torchvision.models.vgg16(pretrained=False, progress=True).to(device)
                # 加载整个模型
                # net2 = torch.load(path_model_2).to(device)
                # print(id(torch.load(path_model_2)))
                # print(id(net2))
                # net = net.to(device)
                print("net2 in CUDA:", next(net2.parameters()).is_cuda)
                # print(net_load)
                # 加载模型参数
                # state_dict_load = torch.load(path_state_dict)
                # print("torch.load(%s)" % path_model_2)

                # 加载回复点
                path_checkpoint = "./Net2checkpint_{}_itreation.pkl".format(iteration_num - 1)
                print("load pkl: %s" % path_checkpoint)
                print("loss1: {}".format(loss2))
                checkpoint = torch.load(path_checkpoint)

                net2.load_state_dict(checkpoint['model_state_dict'])

                optimizer2.load_state_dict(checkpoint['optimizer_state_dic'])

                start_iteration_num = checkpoint['iteration_num']

                params = net2.state_dict()
                for k, v in params.items():
                    print(k,v)
                    break

            # print(id(net))
            # tool.print_get_gpu_info()
            X = X.to(device)
            y = y.to(device)
            y_hat2 = net2(X)
            l2 = loss2(y_hat2, y)
            optimizer2.zero_grad()
            l2.backward()
            optimizer2.step()
            train_l_sum2 += l2.cpu().item()
            train_acc_sum2 += (y_hat2.argmax(dim=1) == y).sum().cpu().item()

            iteration_time.append(time.time() - interaion_time)
            tool.print_get_gpu_info()



            if iteration_num > 0 :
                # 保存整个模型
                torch.save(net2, path_model_2)

                # 保存模型参数
                # net_state_dict = net1.state_dict()
                # torch.save(net1_state_dict, path_state_dict)
                # print("torch.save(net, %s)" % path_model_2)

                # 保存更多信息
                checkpoint = {"model_state_dict": net2.state_dict(),
                              "optimizer_state_dic": optimizer2.state_dict(),
                              "loss1": loss2,
                              "iteration_num": iteration_num}
                path_checkpoint = "./Net2checkpint_{}_itreation.pkl".format(iteration_num)
                print("Save pkl: %s" % path_checkpoint)
                torch.save(checkpoint, path_checkpoint)
                # net1.to('cpu')

                torch.cuda.empty_cache()
                print("net in CUDA:", next(net2.parameters()).is_cuda)
            # net2.to('cpu')
            del net2
            torch.cuda.empty_cache()
            print(torch.cuda.memory_allocated() / (2 ** 20), torch.cuda.max_memory_allocated() / (2 ** 20))
            print(torch.cuda.memory_cached() / (2 ** 20), torch.cuda.max_memory_cached() / (2 ** 20))
            # print(torch.cuda.reset_max_memory_cached(device), torch.cuda.reset_max_memory_allocated(device))
            # print(torch.cuda.memory_stats(device), torch.cuda.memory_summary(device))
            print("Net2 After interaion: %d" % iteration_num, sep="\t")
            tool.print_get_gpu_info()


            iteration_time.append(time.time() - interaion_time)
            n += y.shape[0]
            batch_count += 1
            iteration_num += 1
            print('Net2 iteraion %d, loss= %.4f, train acc2= %.3f, , time= %.10f sec'
                  % (iteration_num, train_l_sum2 / batch_count, train_acc_sum2 / n, time.time() - interaion_time),
                  sep="\t")
            '''训练任务2结束'''




        test_acc2 = evaluate_accuracy(test_iter, net2)


        epoch_time.append(time.time() - start)
        print('epoch %d, Net2, loss2 %.4f, train acc2 %.3f, test acc2 %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum2 / batch_count, train_acc_sum2 / n, test_acc2, time.time() - start))


        print("After epoch:", sep="\t")
        tool.print_get_gpu_info()
    print("Train end time %.2f sec" % (time.time() - All_time))

    plt_label = 'Memory-iteration, ' + 'batchsize = %d' % batch_size + ', epoch= %d' % num_epochs





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
        train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=10,
                                                 pin_memory=True)
        test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=10,
                                                pin_memory=True)

    elif Select_dataset == 'CIFAR10':
        mnist_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=10,
                                                 pin_memory=True)
        test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=10,
                                                pin_memory=True)
    else:
        pass

    print("batch_size:", batch_size,
          "\tDataset:", Select_dataset,
          "\n%s train dataset size" % Select_dataset, mnist_train.__len__(),
          "\ttrain dataset iteration number:", train_iter.__len__(),
          "\t\tTotal:", train_iter.__len__() * batch_size,
          "\n%s test dataset size" % Select_dataset, mnist_test.__len__(),
          "\ttest dataset iteration number:", test_iter.__len__(),
          "\t\tTotal:", test_iter.__len__() * batch_size,
          sep="\t")

    return train_iter, test_iter

if __name__ == '__main__':
    main()