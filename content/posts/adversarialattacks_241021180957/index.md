---
weight: 4
title: "对抗攻击概述"
date: 2021-11-10T21:57:40+08:00
lastmod: 2021-11-10T16:45:40+08:00
draft: false
author: "Allen191819"
authorLink: "https://allen191819.xyz"
description: "对抗攻击入门概述"

tags: ["Daily study", "AI security"]
categories: ["Adversarial Attack"]

lightgallery: true

math:
    enable: true
resources:
- name: featured-image
  src: featured-image.jpg
---



对抗攻击入门, 了解12种生成对抗样本的方法和15种防御方法。
<!--more-->
# Adversarial attacks概述


<!-- vim-markdown-toc GFM -->

* [十二种生成对抗样本的方法](#十二种生成对抗样本的方法)
    * [Box-constrained L-BFGS](#box-constrained-l-bfgs)
    * [Fast Gradient Sign Method (FGSM)](#fast-gradient-sign-method-fgsm)
    * [Basic & Least-Likely-Class Iterative Methods](#basic--least-likely-class-iterative-methods)
    * [Jacobian-based Saliency Map Attack (JSMA)](#jacobian-based-saliency-map-attack-jsma)
    * [One Pixel Attack](#one-pixel-attack)
    * [Carlini and Wagner Attacks (C&W)](#carlini-and-wagner-attacks-cw)
    * [DeepFool](#deepfool)
    * [Universal Adversarial Perturbations](#universal-adversarial-perturbations)
    * [UPSET and ANGRI](#upset-and-angri)
    * [Houdini](#houdini)
    * [Adversarial Transformation Networks (ATNs)](#adversarial-transformation-networks-atns)
    * [Miscellaneous Attacks](#miscellaneous-attacks)
* [防御对抗攻击的方法分类](#防御对抗攻击的方法分类)
    * [修改训练过程/输入数据](#修改训练过程输入数据)
    * [修改网络](#修改网络)
    * [使用附加网络](#使用附加网络)

<!-- vim-markdown-toc -->

对抗攻击是机器学习和计算机安全的结合。按照攻击者是否知道目标网络的结构参数，可以将对抗攻击分为**白盒攻**和**黑盒攻击**。
实际中，根据目的网络最终得到的分类结果是否是攻击者预先设计好的，将对抗攻击分为目标攻击和非目标攻击。
研究对抗攻击的意义如下：

1. 能让机器学习模型处理大规模数据；
2. 以“计算机速度”处理攻击威胁；
3. 不依赖数据的明显特征，发现实际应用中的各种内在威胁；
4. 阻止已知和未知的恶意软件；
5. 阻止恶意软件的提前执行；
6. 优化模型，让分类模型达到更加高的分类准确率和更加低的错误率。

对抗攻击的研究方向概括：

1. 攻击原理
2. 对抗攻击方法
3. 对抗攻击防御
4. 实际应用

## 十二种生成对抗样本的方法

### Box-constrained L-BFGS

通过对图像添加小量的人类察觉不到的扰动误导神经网络做出误分类。由求解让神经网络做出误分类的最小扰动方程转而求解简化后的问题，及寻找最小损失函数添加项，使神经网络做出误分类，进而将问题转化为凸优化过程。

### Fast Gradient Sign Method (FGSM)


通过对抗训练提高深度神经网络的鲁棒性，从而提升防御对抗样本攻击的能力。通过用识别概率最小的类别（目标类别）代替对抗扰动中的类别变量，再将原始图像减去该扰动，原始图像就变成了对抗样本，并能输出目标类别。

### Basic & Least-Likely-Class Iterative Methods

one-step 方法通过一大步运算增大分类器的损失函数而进行图像扰动，因而可以直接将其扩展为通过多个小步增大损失函数的变体，从而我们得到 Basic Iterative Methods。该方法的变体也是通过识别概率最小的类别（目标类别）代替扰动中的类别变量，而得到 Least-Likely-Class Iterative Methods

### Jacobian-based Saliency Map Attack (JSMA)

对抗攻击文献中通常使用的方法是限制扰动的 l_∞ 或 l_2 范数的值以使对抗样本中的扰动无法被人察觉。但 JSMA 提出了限制 l_0 范数的方法，即仅改变几个像素的值，而不是扰动整张图像。

### One Pixel Attack

这是一种极端的对抗攻击方法，仅改变图像中的一个像素值就可以实现对抗攻击。使用差分进化算法，对每个像素进行迭代地修改生成子图像，并与母图像对比，根据选择标准保留攻击效果最好的子图像，实现对抗攻击。这种对抗攻击不需要知道网络参数或梯度的任何信息。

### Carlini and Wagner Attacks (C&W)


Carlini 和 Wagner 提出了三种对抗攻击方法，通过限制 l_∞、l_2 和 l_0 范数使得扰动无法被察觉。实验证明 defensive distillation 完全无法防御这三种攻击。该算法生成的对抗扰动可以从 unsecured 的网络迁移到 secured 的网络上，从而实现黑箱攻击。

### DeepFool

Moosavi-Dezfooli 等人通过迭代计算的方法生成最小规范对抗扰动，将位于分类边界内的图像逐步推到边界外，直到出现错误分类。他们生成的扰动比 FGSM 更小，同时有相似的欺骗率。

### Universal Adversarial Perturbations

诸如 FGSM、ILCM、 DeepFool 等方法只能生成单张图像的对抗扰动，而 Universal Adversarial Perturbations[16] 能生成对任何图像实现攻击的扰动，这些扰动同样对人类是几乎不可见的。该论文中使用的方法和 DeepFool 相似，都是用对抗扰动将图像推出分类边界，不过同一个扰动针对的是所有的图像。虽然文中只针对单个网络 ResNet 进行攻击，但已证明这种扰动可以泛化到其它网络上。

### UPSET and ANGRI

Sarkar 等人提出了两个黑箱攻击算法，UPSET 和 ANGRI。UPSET 可以为特定的目标类别生成对抗扰动，使得该扰动添加到任何图像时都可以将该图像分类成目标类别。相对于 UPSET 的「图像不可知」扰动，ANGRI 生成的是「图像特定」的扰动。它们都在 MNIST 和 CIFAR 数据集上获得了高欺骗率。

### Houdini

Houdini 是一种用于欺骗基于梯度的机器学习算法的方法，通过生成特定于任务损失函数的对抗样本实现对抗攻击，即利用网络的可微损失函数的梯度信息生成对抗扰动。除了图像分类网络，该算法还可以用于欺骗语音识别网络。

### Adversarial Transformation Networks (ATNs)

Baluja 和 Fischer 训练了多个前向神经网络来生成对抗样本，可用于攻击一个或多个网络。该算法通过最小化一个联合损失函数来生成对抗样本，该损失函数有两个部分，第一部分使对抗样本和原始图像保持相似，第二部分使对抗样本被错误分类。

### Miscellaneous Attacks

这种类型的攻击通过强制缓存服务器或 Web 浏览器来披露可能是敏感和机密的用户特定信息来利用易受攻击的 Web 服务器。

## 防御对抗攻击的方法分类

**目前，在对抗攻击防御上存在三个主要方向：**

1. 在学习过程中修改训练过程或者修改的输入样本。
2. 修改网络，比如：添加更多层/子网络、改变损失/激活函数等。
3. 当分类未见过的样本时，用外部模型作为附加网络。

{{< figure src="2021-10-24-14-21-55.png" size=300x300 >}}

### 修改训练过程/输入数据

1. 蛮力对抗训练：通过不断输入新类型的对抗样本并执行对抗训练，从而不断提高网络的鲁棒性。为了保证有效性，该方法需要使用高强度的对抗样本，并且网络架构要有充足的表达能力。这种方法需要大量的训练数据，因而被称为蛮力对抗训练。事实上无论添加多少对抗样本，都存在新的对抗攻击样本可以再次欺骗网络。
2. 数据压缩：注意到大多数训练图像都是 JPG 格式，Dziugaite[123] 等人使用 JPG 图像压缩的方法，减少对抗扰动对准确率的影响。实验证明该方法对部分对抗攻击算法有效，但通常仅采用压缩方法是远远不够的，并且压缩图像时同时也会降低正常分类的准确率，后来提出的 PCA 压缩方法也有同样的缺点。
3. 基于中央凹机制的防御：Luo 等人提出用中央凹（foveation）机制可以防御 L-BFGS 和 FGSM 生成的对抗扰动，其假设是图像分布对于转换变动是鲁棒的，而扰动不具备这种特性。但这种方法的普遍性尚未得到证明。
4. 数据随机化方法：Xie 等人发现对训练图像引入随机重缩放可以减弱对抗攻击的强度，其它方法还包括随机 padding、训练过程中的图像增强等。

### 修改网络

5. 深度压缩网络：人们观察到简单地将去噪自编码器（Denoising Auto Encoders）堆叠到原来的网络上只会使其变得更加脆弱，因而 Gu 和 Rigazio 引入了深度压缩网络（Deep Contractive Networks），其中使用了和压缩自编码器（Contractive Auto Encoders）类似的平滑度惩罚项。

6. 梯度正则化/ masking：使用输入梯度正则化以提高对抗攻击鲁棒性，该方法和蛮力对抗训练结合有很好的效果，但计算复杂度太高。

7. Defensive distillation：distillation 是指将复杂网络的知识迁移到简单网络上，由 Hinton 提出。Papernot 利用这种技术提出了 Defensive distillation，并证明其可以抵抗小幅度扰动的对抗攻击。

8. 生物启发的防御方法：使用类似与生物大脑中非线性树突计算的高度非线性激活函数以防御对抗攻击。另外一项工作 Dense Associative Memory 模型也是基于相似的机制。

9. Parseval 网络：在一层中利用全局 Lipschitz 常数加控制，利用保持每一层的 Lipschitz 常数来摆脱对抗样本的干扰。

10. DeepCloak：在分类层（一般为输出层）前加一层特意为对抗样本训练的层。它背后的理论认为在最显著的层里包含着最敏感的特征。

11. 混杂方法：这章包含了多个人从多种角度对深度学习模型的调整从而使模型可以抵抗对抗性攻击。

12. 仅探测方法：4 种网络，SafetyNet，Detector subnetwork，Exploiting convolution filter statistics 及 Additional class augmentation。

> -   SafetyNet 介绍了 ReLU 对对抗样本的模式与一般图片的不一样，文中介绍了一个用 SVM 实现的工作。
> -   Detector subnetwork 介绍了用 FGSM, BIM 和 DeepFool 方法实现的对对抗样本免疫的网络的优缺点。
> -   Exploiting convolution filter statistics 介绍了同 CNN 和统计学的方法做的模型在分辨对抗样本上可以有 85% 的正确率。

### 使用附加网络

13. 防御通用扰动:利用一个单独训练的网络加在原来的模型上，从而达到不需要调整系数而且免疫对抗样本的方法。

14. 基于 GAN 的防御:用 GAN 为基础的网络可以抵抗对抗攻击，而且作者提出在所有模型上用相同的办法来做都可以抵抗对抗样本。

15. 仅探测方法：介绍了 Feature Squeezing、MagNet 以及混杂的办法。

> -   Feature Squeezing 方法用了两个模型来探查是不是对抗样本。后续的工作介绍了这个方法对 C&W 攻击也有能接受的抵抗力。
> -   MagNet:作者用一个分类器对图片的流行（manifold）测量值来训练，从而分辨出图片是不是带噪声的。
> -   混杂方法（Miscellaneous Methods）：作者训练了一个模型，把所有输入图片当成带噪声的，先学习怎么去平滑图片，之后再进行分类。

