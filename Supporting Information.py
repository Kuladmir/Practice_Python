import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

# 引入数据
data_dir = './flower_data/'
train_dir = data_dir + '/train'
test_dir = data_dir + '/data'
# 图片处理 // 数据强化
data_transforms = {
    "train": transforms.Compose([transforms.RandomRotation(45),  # 随机旋转图像最多45度
                                 transforms.CenterCrop(224),  # 从图像的中心裁剪出一个224x224的区域
                                 transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率水平翻转图像
                                 transforms.RandomVerticalFlip(p=0.5),  # 50%的概率垂直翻转图像
                                 transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1,hue=0.1),  # 改变图像的亮度、对比度、饱和度和色调
                                 transforms.RandomGrayscale(p=0.025),  # 以2.5%的概率将图像转换为灰度图
                                 transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为FloatTensor，并且缩放图像到[0.0, 1.0]
                                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),  # 准化图像张量，使用给定的均值和标准差
    "valid": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
}

# 每次迭代从数据集中加载128张图像
batch_size = 128   # ///超参数调整位置

#设置epoch次数（即学习次数）
num_epochs = 50  # ///超参数调整位置

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                  for x in ["train","valid"]}  # 这是一个字典，其中"train"和"valid"键分别对应于训练数据集和验证数据集
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True)
               for x in ["train","valid"]}  # 这是一个字典，其中"train"和"valid"键分别对应于训练数据加载器和验证数据加载器
dataset_sizes = {x: len(image_datasets[x])  # 这是一个字典，存储了训练和验证数据集的大小
                 for x in ["train","valid"]}
class_names = image_datasets["train"].classes  # 这个变量存储了训练数据集中的类别名称

model_name = 'resnet'
#是否用训练好的特征来做,true用人家权重，false则可以使用自己的参数
feature_extract = True

# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        #加载模型（下载）
        model_ft = models.resnet152(weights = 'ResNet152_Weights.IMAGENET1K_V1')
        #有选择性的选需要冻住哪些层
        set_parameter_requires_grad(model_ft, feature_extract)
        #取出最后一层
        num_ftrs = model_ft.fc.in_features
        #重新做全连接层
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 15), # //修改类数
                                   nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":

        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":

        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":

        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":

        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":

        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

#设置哪些层需要训练
model_ft, input_size = initialize_model(model_name, 15, feature_extract, use_pretrained=True)#//修改类数

#GPU计算
model_ft = model_ft.to(device) #device这里放置的是gpu

#模型保存
filename='checkpoint.pth'

# 是否训练所有层
params_to_update = model_ft.parameters()
# print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            # print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# 优化器设置   ///超参数调整位置
optimizer_ft = optim.Adam(params_to_update, lr = 0.02)   #lr学习率为 0.04   #Adam
# 传入优化器，迭代了多少后要变换学习率，学习率改变  学习率每 20 个epoch衰减成原来的 1/10
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)  # /// 此处两个参数可改
#定义损失函数
criterion = nn.NLLLoss()

def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False, filename=filename):
    since = time.time()
    #保存最好的准确率
    best_acc = 0

    #训练
    model.to(device)
    #初始化所有值
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    #保存最好的一次
    best_model_wts = copy.deepcopy(model.state_dict())
    #开始学习
    for epoch in range(num_epochs):
        print('Epoch  {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证
            #初始化两个变量，用于累计当前阶段的总损失和正确预测的数量
            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    #resnet不执行这个
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    #获取模型输出的预测类别
                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #累计当前批次的损失和正确预测的数量
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            #计算当前阶段的平均损失和准确率
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step()#epoch_loss
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,criterion, optimizer_ft,num_epochs,is_inception=(model_name == "inception"))

valid_acc_lst_cpu = [tensor.cpu() for tensor in val_acc_history]
valid_acc = [tensor.item() for tensor in valid_acc_lst_cpu]
train_acc_lst_cpu = [tensor.cpu() for tensor in train_acc_history]
train_acc = [tensor.item() for tensor in train_acc_lst_cpu]

plt.plot(range(1, num_epochs + 1), train_losses, label='Training loss')
plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation loss')
plt.legend(loc='upper right')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()

plt.plot(range(1, num_epochs + 1), train_acc, label='Training accuracy')
plt.plot(range(1, num_epochs + 1), valid_acc, label='Validation accuracy')
plt.legend(loc='upper left')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()