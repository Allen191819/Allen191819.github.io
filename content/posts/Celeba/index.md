---
weight: 4
title: "Celeba 数据集"
date: 2021-12-12T21:57:40+08:00
lastmod: 2021-12-13T16:45:40+08:00
draft: false
author: "Allen191819"
authorLink: "https://allen191819.xyz"
description: "Celeba 数据集介绍"

tags: ["AI"]
categories: ["Dataset"]

lightgallery: true

math:
    enable: true
resources:
- name: featured-image
  src: featured-image.jpg
---

对于 Celeba 数据集的介绍

<!--more-->

# Celeba 数据集


{{< figure src="celeba.png" size=300x300 >}}

CelebA是CelebFaces Attribute的缩写，意即名人人脸属性数据集，其包含10,177个名人身份的202,599张人脸图片，每张图片都做好了特征标记，包含人脸bbox标注框、5个人脸特征点坐标以及40个属性标记，CelebA由香港中文大学开放提供，广泛用于人脸相关的计算机视觉训练任务，可用于人脸属性标识训练、人脸检测训练以及landmark(特征点)标记等，官方网址：[Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

{{< figure src="overview.png" size=300x300 >}}

## 几种下载的资源

+ In-The-Wild Images (Img/img_celeba.7z)

202,599张原始“野生”人脸图像，从网络爬取未有做任何裁剪缩放操作的人脸图像；

+ Align&Cropped Images (Img/img_align_celeba.zip & Img/img_align_celeba_png.7z)

202,599张经过人脸对齐和裁剪了的图像，视情况下载对应不同质量的图像即可，一般选择jpg格式才1G多的img_align_celeba.zip文件；

## 几种分类标签
+ Bounding Box Annotations (Anno/list_bbox_celeba.txt)

bounding box标签，即人脸标注框坐标注释文件，包含每一张图片对应的bbox起点坐标及其宽高，如下：

```
202599
image_id x_1 y_1 width height
000001.jpg    95  71 226 313
000002.jpg    72  94 221 306
000003.jpg   216  59  91 126
000004.jpg   622 257 564 781
000005.jpg   236 109 120 166
000006.jpg   146  67 182 252
000007.jpg    64  93 211 292
000008.jpg   212  89 218 302
000009.jpg   600 274 343 475
000010.jpg   113 110 211 292
000011.jpg   166  68 125 173
000012.jpg   102  31 104 144
```

+ Landmarks Annotations (Anno/list_landmarks_celeba.txt & Anno/list_landmarks_align_celeba.txt)


5个特征点landmark坐标注释文件，list_landmarks_align_celeba.txt则是对应人脸对齐后 的landmark坐标
```
202599
lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
000001.jpg 69  109  106  113   77  142   73  152  108  154
000002.jpg 69  110  107  112   81  135   70  151  108  153
000003.jpg 76  112  104  106  108  128   74  156   98  158
000004.jpg 72  113  108  108  101  138   71  155  101  151
000005.jpg 66  114  112  112   86  119   71  147  104  150
000006.jpg 71  111  106  110   94  131   74  154  102  153
000007.jpg 70  112  108  111   85  135   72  152  104  152
000008.jpg 71  110  106  111   84  137   73  155  104  153
```

- Attributes Annotations (Anno/list_attr_celeba.txt)

40个属性标签文件，第一行为图像张数，第二行为属性名，有该属性则标记为1，否则标记为-1

```
202599
5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young 
000001.jpg -1  1  1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1  1  1 -1  1 -1 -1  1 -1 -1  1 -1 -1 -1  1  1 -1  1 -1  1 -1 -1  1
000002.jpg -1 -1 -1  1 -1 -1 -1  1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1  1 -1  1 -1 -1  1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1  1
000003.jpg -1 -1 -1 -1 -1 -1  1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1  1  1 -1 -1  1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1  1
000004.jpg -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1  1 -1 -1 -1 -1  1 -1  1 -1  1  1 -1  1
000005.jpg -1  1  1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1  1  1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1  1
000006.jpg -1  1  1 -1 -1 -1  1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1  1 -1 -1  1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1  1  1 -1  1 -1 -1  1
000007.jpg 1 -1  1  1 -1 -1  1  1  1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1  1 -1 -1  1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1  1
```

- Identity Annotations (available upon request)

10,177个名人身份标识，图片的序号即是该图片对应的标签

```
000001.jpg 2880
000002.jpg 2937
000003.jpg 8692
000004.jpg 5805
000005.jpg 9295
000006.jpg 4153
000007.jpg 9040
000008.jpg 6369
000009.jpg 3332
000010.jpg 612
000011.jpg 2807
000012.jpg 7779
000013.jpg 3785
000014.jpg 7081
000015.jpg 1854
000016.jpg 4905
000017.jpg 667
000018.jpg 2464
000019.jpg 2929
000020.jpg 2782
000021.jpg 181
000022.jpg 6642
```
- Evaluation Partitions (Eval/list_eval_partition.txt)

用于划分为training，validation及testing等数据集的标签文件，标签0对应training，标签1对应validation，标签2对应testing



