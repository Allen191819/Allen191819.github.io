---
weight: 4
title: "Haskell 学习笔记五"
date: 2022-04-22
lastmod: 2021-04-22
draft: false
author: "Allen191819"
authorLink: "https://allen191819.xyz"
description: "Haskell 函子、应用函子与单子"

tags: ["Lambda", "FP"]
categories: ["Haskell"]

lightgallery: true

math:
  enable: true
resources:
  - name: featured-image
    src: featured-image.jpg
---

Haskell 函子、应用函子与单子

<!--more-->

# Haskell learning

## Functors

Functors
是一个类型类(`typeclass`)，和其他类型类一样，他规定了其实类型必须实现的相关功能，
