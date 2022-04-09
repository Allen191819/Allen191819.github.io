---
weight: 4
title: "Github 对抗攻击库"
date: 2021-11-10T21:57:40+08:00
lastmod: 2021-11-14T16:45:40+08:00
draft: false
author: "Allen191819"
authorLink: "https://dillonzq.com"
description: "Github 中部分经典对抗攻击库整理"

tags: ["Daily study", "AI security"]
categories: ["Adversarial Attack"]

lightgallery: true

math:
    enable: true
---



Github 中部分经典对抗攻击库整理

<!--more-->

# 部分 Github 对抗攻击库整理

<!-- vim-markdown-toc Marked -->

* [nlpaug](#nlpaug)
* [Adversarial Robustness Toolbox](#adversarial-robustness-toolbox)
* [Foolbox Native](#foolbox-native)
* [TextAttack](#textattack)
* [AdvBox Family](#advbox-family)
* [Advertorch](#advertorch)
* [OpenAttack](#openattack)

<!-- vim-markdown-toc -->

## nlpaug

{{< figure src="2021-10-25-17-26-41.png" href="https://github.com/makcedward/nlpaug" >}}


-   简介：这个 python 库可以帮助你为你的机器学习项目增强自然语言处理(NLP)。`augmenter`是增强的基本元素，而 `Flow` 是将多个增强器编排在一起的管道。

-   特性：

> 1.  无需动手即可生成用以改善模型性能的合成数据。
> 2.  简单、易用、轻量
> 3.  可以轻松在任何机器学习、神经网络框架中使用（scikit-learn, PyTorch, TensorFlow）
> 4.  支持文本和语音输入

-   最近更新和最新版本：1.1.8 2021.10.18

-   MIT License

## Adversarial Robustness Toolbox


{{< figure src="2021-10-25-17-29-29.png" href="https://github.com/Trusted-AI/adversarial-robustness-toolbox" >}}

-   简介：对抗性鲁棒性工具集（ART）是用于机器学习安全性的 Python 库。

-   特性：

> 1.  ART 提供的工具可帮助开发人员和研究人员针对以下方面捍卫和评估机器学习模型和应用程序： 逃逸，数据污染，模型提取和推断的对抗性威胁。
> 2.  ART 支持所有流行的机器学习框架 （TensorFlow，Keras，PyTorch，MXNet，scikit-learn，XGBoost，LightGBM，CatBoost，GPy 等），所有数据类型 （图像，表格，音频，视频等）和机器学习任务（分类，物体检测，语音识别， 生成模型，认证等）。

-   对抗威胁

{{< figure src="2021-10-25-17-34-14.png" >}}

-   最近更新和最新版本：1.8.1 2021.10.15

-   MIT License

## Foolbox Native

{{< figure src="2021-10-25-17-41-05.png" href="https://github.com/bethgelab/foolbox" >}}

-   简介：Foolbox 是一个 Python 库，可让您轻松对深度神经网络等机器学习模型进行对抗性攻击。 它构建在 EagerPy 之上，并在本地与 PyTorch、TensorFlow 和 JAX 中的模型一起使用。

-   特性：

> 1.  原生性能：Foolbox 3 构建在 EagerPy 之上，在 PyTorch、TensorFlow 和 JAX 中原生运行，并提供真正的批处理支持。
> 2.  最先进的攻击：Foolbox 提供了大量最先进的基于梯度和基于决策的对抗性攻击。
> 3.  类型检查：由于 Foolbox 中的大量类型注释，在运行代码之前捕获错误。

-   最近更新和最近版本： 3.3.1 2021.2.23

-   MIT License


## TextAttack

{{< figure src="2021-10-25-19-13-42.png" href="https://github.com/QData/TextAttack" >}}

-   简介：TextAttack 是一个用于对抗性攻击、数据增强和 NLP 模型训练的 Python 框架。

-   特性：

> 1.  通过对 NLP 模型运行不同的对抗性攻击并检查输出，更好地理解它们
> 2.  使用 TextAttack 框架和组件库研究和开发不同的 NLP 对抗性攻击
> 3.  扩充您的数据集以提高模型的泛化能力和下游的稳健性
> 4.  仅使用一个命令训练 NLP 模型（包括所有下载！）

-   最近更新和最新版本：0.3.3 2021.8.3

-   MIT License

## AdvBox Family

{{< figure src="2021-10-25-19-25-58.png" href="https://github.com/advboxes/AdvBox" >}}


-   简介：Advbox Family 是百度开源的一系列 AI 模型安全工具集，包括对抗样本的生成、检测和保护，以及针对不同 AI 应用的攻防案例。

-   [AdvSDK](https://github.com/advboxes/AdvBox/blob/master/advsdk/README.md)

用于 Paddlepaddle 的轻量级 ADV SDK 以产生对抗性示例。

-   [AdversarialBox](https://github.com/advboxes/AdvBox/blob/master/adversarialbox.md)

Adversarialbox 是一个工具箱，可以在 PaddlePaddle、PyTorch、Caffe2、MxNet、Keras、TensorFlow 和 Advbox 中生成欺骗神经网络的对抗样本，可以对机器学习模型的鲁棒性进行基准测试。Advbox 提供了一个命令行工具来生成零编码的对抗样本。 它受到启发并基于 FoolBox v1。

-   [AdvDetect](https://github.com/advboxes/AdvBox/blob/master/advbox_family/AdvDetect/README.md)

AdvDetect 是一种工具箱，用于检测来自大规模数据的对抗性示例。

-   AdvPoison

-   Apache License 2.0

## Advertorch


{{< figure src="2021-10-25-19-40-42.png" href="https://github.com/BorealisAI/advertorch" >}}

-   简介：是一个用于对抗性鲁棒性研究的 Python 工具箱。 主要功能在 PyTorch 中实现。 具体来说，AdverTorch 包含用于生成对抗性扰动和防御对抗性示例的模块，以及用于对抗性训练的脚本。

-   最近更新与最新版本：0.2 2021.7.30

-   License: This project is licensed under the LGPL. The terms and conditions can be found in the LICENSE and LICENSE.GPL files.

## OpenAttack

{{< figure src="2021-10-25-20-11-11.png" href="https://github.com/thunlp/OpenAttack" >}}

-   简介：OpenAttack 是一个开源的基于 Python 的文本对抗性攻击工具包，它处理文本对抗性攻击的整个过程，包括预处理文本、访问受害者模型、生成对抗性示例和评估。

-   特性：

> 1.  支持所有攻击类型。 OpenAttack 支持所有类型的攻击，包括句子/单词/字符级扰动和梯度/分数/基于决策/盲攻击模型；
> 2.  多语种。 OpenAttack 现在支持英文和中文。 其可扩展的设计可以快速支持更多语言；
> 3.  并行处理。 OpenAttack 提供对攻击模型多进程运行的支持，提高攻击效率；
> 4.  明星与拥抱拥抱脸的兼容性。 OpenAttack 与拥抱 Transformers 和 Datasets 库完全集成；
> 5.  很好的扩展性。 您可以在任何自定义数据集上轻松攻击自定义受害者模型，或开发和评估自定义攻击模型。

-   用途：

> 1.  为攻击模型提供各种方便的基线；
> 2.  使用其全面的评估指标全面评估攻击模型；
> 3.  借助其常见的攻击组件，协助快速开发新的攻击模型；
> 4.  评估机器学习模型对抗各种对抗性攻击的鲁棒性；
> 5.  通过使用生成的对抗性示例丰富训练数据，进行对抗性训练以提高机器学习模型的鲁棒性。

-   最近更新和最新版本：2.1.1 2021.9.22

-   MIT License


