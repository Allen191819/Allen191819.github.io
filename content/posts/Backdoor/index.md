---
weight: 4
title: "隐秘的后门攻击"
date: 2021-12-04T21:57:40+08:00
lastmod: 2021-12-04T16:45:40+08:00
draft: false
author: "Allen191819"
authorLink: "https://dillonzq.com"
description: "对于 Hidden trigger backdoor attack 的复现"

tags: ["Daily study", "AI security"]
categories: ["Backdoor Attack"]

lightgallery: true

math:
    enable: true
---

对于 Hidden trigger backdoor attack 的复现

<!--more-->

# Backdoor attack

<!-- vim-markdown-toc GFM -->

* [主要工作以及算法](#主要工作以及算法)
* [二分类](#二分类)
    * [代码实现](#代码实现)
    * [生成毒化样本](#生成毒化样本)
    * [猴子数据集](#猴子数据集)
        * [target data](#target-data)
        * [source data](#source-data)
        * [patched data](#patched-data)
        * [posion data](#posion-data)
        * [patched result](#patched-result)
        * [clean result](#clean-result)
    * [猫狗大战](#猫狗大战)
        * [target image](#target-image)
        * [source data](#source-data-1)
        * [patched data](#patched-data-1)
        * [posion data](#posion-data-1)
        * [patched result 准确率 73%](#patched-result-准确率-73)
        * [clean result 准确率 94%](#clean-result-准确率-94)
* [多分类 backdoor(Cifar10)](#多分类-backdoorcifar10)
    * [未贴 patch 的数据](#未贴-patch-的数据)
    * [贴 patch 后的数据](#贴-patch-后的数据)
    * [目标类别](#目标类别)
    * [毒化数据](#毒化数据)
    * [实验结果](#实验结果)
    * [不同影响因素分析](#不同影响因素分析)

<!-- vim-markdown-toc -->

## 主要工作以及算法

{{< figure src="2021-12-06-21-33-42.png" size=300x300 >}}

对于给出的 $target$ 图像 $t$ ， $source$ 图像 $s$ ， 以及 $trigger\ patch\ p$ ， 将 $p$ 贴在 $s$ 上得到一个 $\tilde s$ 。 通过如下方法获得毒化数据 $z$ :

$$
arg_{z}\ min \||f(z)-f(\tilde s)\||_{2}^{2}
$$

$$
st.\ \|z-t\|_{\infty} < \epsilon
$$

生成毒化数据的具体算法

{{< figure src="2021-12-06-21-56-32.png" size=300x300 >}}

## 二分类

### 代码实现

获取 fc7 的特征输出

> 主要通过添加钩子 `feat1` 获取网络 `forward` 过程中的 fc7 获取的特征

```python
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
**all** = ['AlexNet', 'alexnet']

model_urls = {
'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class NormalizeByChannelMeanStd(nn.Module):
def **init**(self, mean, std):
super(NormalizeByChannelMeanStd, self).**init**()
if not isinstance(mean, torch.Tensor):
mean = torch.tensor(mean)
if not isinstance(std, torch.Tensor):
std = torch.tensor(std)
self.register_buffer("mean", mean)
self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def normalize_fn(tensor, mean, std):
"""Differentiable version of torchvision.functional.normalize"""
mean = mean[None, :, None, None]
std = std[None, :, None, None]
return tensor.sub(mean).div(std)

class AlexNet(nn.Module):
def **init**(self, num_classes=1000):
super(AlexNet, self).**init**()
self.features = nn.Sequential(
nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2),
nn.Conv2d(64, 192, kernel_size=5, padding=2),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2),
nn.Conv2d(192, 384, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(384, 256, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.Conv2d(256, 256, kernel_size=3, padding=1),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2),
)
self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
self.classifier = nn.Sequential(
nn.Dropout(),
nn.Linear(256 _ 6 _ 6, 4096),
nn.ReLU(inplace=True),
nn.Dropout(),
nn.Linear(4096, 4096),
nn.ReLU(inplace=True),
nn.Linear(4096, num_classes),
)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)

        for i in range(6):
            x = self.classifier[i](x)
        feat = x
        x = self.classifier[6](x)

        return x, feat

def alexnet(pretrained=False, progress=True, **kwargs):
model = AlexNet(**kwargs)
if pretrained:
state_dict = load_state_dict_from_url(model_urls['alexnet'],
progress=progress)
model.load_state_dict(state_dict)
return model

```

### 生成毒化样本

```python
def train(model, epoch):

    # AVERAGE METER
    losses = AverageMeter()

    # TRIGGER PARAMETERS
    trans_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    trans_trigger = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
    ])

    eps1 = (eps / 255.0)
    lr1 = lr

    trigger = Image.open(
        './triggers/trigger_{}.png'.format(trigger_id)).convert('RGB')
    trigger = trans_trigger(trigger).unsqueeze(0).cuda()

    dataset_target = PoisonGenerationDataset("./target/n1", target_filelist,
                                             trans_image)
    dataset_source = PoisonGenerationDataset('./source/n0', source_filelist,
                                             trans_image)

    train_loader_target = torch.utils.data.DataLoader(dataset_target,
                                                      batch_size=20,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      pin_memory=True)

    train_loader_source = torch.utils.data.DataLoader(dataset_source,
                                                      batch_size=20,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      pin_memory=True)

    iter_target = iter(train_loader_target)
    iter_source = iter(train_loader_source)

    num_poisoned = 0
    for i in tqdm(range(len(train_loader_target))):

        (input1, path1) = next(iter_source)
        (input2, path2) = next(iter_target)

        img_ctr = 0

        input1 = input1.cuda()
        input2 = input2.cuda()
        pert = nn.Parameter(
            torch.zeros_like(input2, requires_grad=True).cuda())

        for z in range(input1.size(0)):
            if not rand_loc:
                start_x = 224 - patch_size - 5
                start_y = 224 - patch_size - 5
            else:
                start_x = random.randint(0, 224 - patch_size - 1)
                start_y = random.randint(0, 224 - patch_size - 1)

            input1[z, :, start_y:start_y + patch_size,
                   start_x:start_x + patch_size] = trigger

        output1, feat1 = model(input1)
        feat1 = feat1.detach().clone()

        for k in range(input1.size(0)):
            img_ctr = img_ctr + 1
            # input2_pert = (pert[k].clone().cpu())

            fname = saveDir_patched + '/' + 'badnet_' + str(os.path.basename(path1[k])).split('.')[0] + '_' + 'epoch_' + str(epoch).zfill(2)\
                + str(img_ctr).zfill(5)+'.png'

            save_image(input1[k].clone().cpu(), fname)
            num_poisoned += 1

        for j in range(num_iter):
            lr1 = adjust_learning_rate(lr, j)

            output2, feat2 = model(input2 + pert)

            feat11 = feat1.clone()
            dist = torch.cdist(feat1, feat2)
            for _ in range(feat2.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero(
                    as_tuple=False).squeeze()
                feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
                dist[dist_min_index[0], dist_min_index[1]] = 1e5

            loss1 = ((feat1 - feat2)**2).sum(dim=1)
            loss = loss1.sum()

            losses.update(loss.item(), input1.size(0))

            loss.backward()

            pert = pert - lr1 * pert.grad
            pert = torch.clamp(pert, -eps1, eps1).detach_()

            pert = pert + input2

            pert = pert.clamp(0, 1)

            if loss1.max().item() < 10 or j == (num_iter - 1):
                for k in range(input2.size(0)):
                    img_ctr = img_ctr + 1
                    input2_pert = (pert[k].clone().cpu())

                    fname = saveDir_poison + '/' + 'loss_' + str(int(loss1[k].item())).zfill(5) + '_' + 'epoch_' + \
                        str(epoch).zfill(2) + '_' + str(os.path.basename(path2[k])).split('.')[0] + '_' + \
                        str(os.path.basename(path1[k])).split('.')[0] + '_kk_' + str(img_ctr).zfill(5)+'.png'

                    save_image(input2_pert, fname)
                    num_poisoned += 1

                break

            pert = pert - input2
            pert.requires_grad = True
```

### 猴子数据集

#### target data

![target](2021-12-06-22-41-14.png)

#### source data

![source](2021-12-06-22-40-27.png)

#### patched data

![patched](2021-12-06-22-38-19.png)

#### posion data

![posion](2021-12-06-22-39-33.png)

#### patched result

![](2021-12-06-22-45-59.png) 

#### clean result

![](2021-12-06-22-46-25.png) 

### 猫狗大战

#### target image

![cat-tar](2021-12-06-22-32-57.png) 

#### source data

![dog-so](2021-12-06-22-34-15.png) 

#### patched data

![](2021-12-06-22-35-21.png) 

#### posion data

![cat-ps](2021-12-06-22-31-23.png) 

#### patched result 准确率 73%

![](2021-12-06-22-50-43.png) 

#### clean result 准确率 94%

![](2021-12-06-22-48-09.png) 

## 多分类 backdoor(Cifar10)

在使用改动后的`alexnet`网络，并将 Cifar10 划分为两部分，一部分用了预训练：预训练结果：训练集：94%，测试集：66% 识别准确率：

```python
 class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 7, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)

        for i in range(4):
            x = self.classifier[i](x)
        feat = x
        x = self.classifier[4](x)

        return x, feat
```

### 未贴 patch 的数据

- 类别一

![](nopatch01.png) 

- 类别二

![](nopatch02.png) 

### 贴 patch 后的数据

- 类别一

![](patched01.png) 

- 类别二

![](patched02.png) 

### 目标类别

![](target09.png) 

### 毒化数据

- 类别一

![](posion01.png) 

- 类别二

![](posion02.png) 

### 实验结果

- 未贴标签

![](nopatch.png) 

- 贴标签

![](patched.png) 

### 不同影响因素分析

不同大小的 patch 对攻击结果的影响

| patch size  | 4     | 6     | 8     |
| ----------- | ----- | ----- | ----- |
| nopatch acc | 0.623 | 0.623 | 0.621 |
| patched acc | 0.595 | 0.564 | 0.441 |
| target      | 0.074 | 0.075 | 0.094 |

不同数据集分割方案对攻击结果的影响(patch size = 6)

| 预训练/微调 | 毒化数据 | val acc | nopatch acc | patched acc | target |
| ----------- | -------- | ------- | ----------- | ----------- | ------ |
| 3500/1500   | 300      | 0.631   | 0.595       | 0.503       | 0.096  |
| 3500/1500   | 200      | 0.639   | 0.617       | 0.502       | 0.092  |
| 3500/1500   | 150      | 0.640   | 0.623       | 0.525       | 0.084  |
| 3500/1500   | 100      | 0.642   | 0.626       | 0.567       | 0.081  |
| 3000/2000   | 300      | 0.640   | 0.602       | 0.529       | 0.113  |
| 3000/2000   | 200      | 0.642   | 0.611       | 0.525       | 0.095  |
| 3000/2000   | 150      | 0.644   | 0.609       | 0.526       | 0.080  |
| 3000/2000   | 100      | 0.646   | 0.610       | 0.498       | 0.080  |
| 2500/2500   | 300      | 0.635   | 0.623       | 0.564       | 0.075  |
| 2500/2500   | 200      | 0.643   | 0.634       | 0.544       | 0.078  |
| 2500/2500   | 150      | 0.642   | 0.636       | 0.542       | 0.065  |
| 2500/2500   | 100      | 0.639   | 0.624       | 0.520       | 0.081  |
| 2000/3000   | 300      | 0.599   | 0.569       | 0.476       | 0.082  |
| 2000/3000   | 200      | 0.602   | 0.573       | 0.478       | 0.113  |
| 2000/3000   | 150      | 0.600   | 0.580       | 0.486       | 0.104  |
| 2000/3500   | 100      | 0.604   | 0.573       | 0.487       | 0.076  |
| 1500/3500   | 300      | 0.584   | 0.566       | 0.476       | 0.090  |

参考文献 [Hidden trigger backdoor](https://arxiv.org/abs/1910.00033v2)
