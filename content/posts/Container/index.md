---
weight: 4
title: "Docker 与 Vagrant 初探"
date: 2022-04-19
lastmod: 2021-04-19
draft: false
author: "Allen191819"
authorLink: "https://allen191819.xyz"
description: "Do you know Docker & Vagrant?"
tags: ["Container & Virtualization"]
categories: ["OS"]
lightgallery: true
math:
  enable: false
resources:
  - name: featured-image
    src: featured-image.jpg
---

---

初探 Docker 与 Vagrant 简记

<!--more-->

# Container

## 容器概述

容器是一种沙盒技术，主要目的是为了将应用运行在其中，与外界隔离；及方便这个沙盒可以被转移到其它宿主机器。本质上，它是一个特殊的进程。通过名称空间（Namespace）、控制组（Control groups）、切根（chroot）技术把资源、文件、设备、状态和配置划分到一个独立的空间。

Linux Container 容器技术的诞生（2008 年）就解决了 IT 世界里“集装箱运输”的问题。Linux Container（简称 LXC）它是一种内核轻量级的操作系统层虚拟化技术。Linux Container 主要由 Namespace 和 Cgroup 两大机制来保证实现。那么 Namespace 和 Cgroup 是什么呢？刚才我们上面提到了集装箱，集装箱的作用当然是可以对货物进行打包隔离了，不让 A 公司的货跟 B 公司的货混在一起，不然卸货就分不清楚了。那么 Namespace 也是一样的作用，做隔离。光有隔离还没用，我们还需要对货物进行资源的管理。同样的，航运码头也有这样的管理机制：货物用什么样规格大小的集装箱，货物用多少个集装箱，货物哪些优先运走，遇到极端天气怎么暂停运输服务怎么改航道等等... 通用的，与此对应的 Cgroup 就负责资源管理控制作用，比如进程组使用 CPU/MEM 的限制，进程组的优先级控制，进程组的挂起和恢复等等。

{{< figure src="linux-container.png" >}}

### Namespace

每个运行的容器都有自己的名称空间。这是 Linux 操作系统默认提供的 API，包括：

- **PID Namespace**：不同容器就是通过 pid 名字空间隔离开的，不同名字空间中可以有相同的 pid。

- **Mount Namespace**：mount 允许不同名称空间的进程看到的文件结构不同，因此不同名称空间中的进程所看到的文件目录就被隔离了。另外，每个名称空间中的容器在/proc/mounts 的信息只包含当前名称的挂载点。

- **IPC Namespace**：容器中进程交互还是采用 Linux 常见的进程交互方法（interprocess communication -IPC），包括信号量、消息队列和共享内存等。

- **Network Namespace**：网络隔离是通过 Net 实现，每个 Net 有独立的网络设备，IP 地址，路由表，/proc/net 目录。这样每个容器的网络就能隔离开来。

- **UTS Namespace**：UTS（UNIX Time-sharing System）允许每个容器拥有独立的 hostname 和 domain name，使其在网络上可以被视作一个独立的节点而非主机上的一个进程。

- **User Namespace**：每个容器可以有不同的用户和组 id，也就是说可以在容器内用容器内部的用户执行程序而非主机上的用户。

### 控制组(Control groups)

Cgroups 是 Linux 内核提供的一种可以限制、记录、隔离进程组的物理资源机制。因为 Namespace 技术只能改变进程的视觉范围，不能真实地对资源做出限制。所以就必须采用 Cgroup 技术对容器进行资源限制，防止某个容器把宿主机资源全部用完导致其它容器也宕掉。在 Linux 的/sys/fs/cgroup 目录中，有 cpu、memory、devices、net_cls 等子目录，可以根据需要修改相应的配置文件来设置某个进程 ID 对物理资源的最大使用率。

### 切根(change to root)

切根就是指改变一个程序与形式参考的根目录的位置，让不同的容器在不同的虚拟根目录下工作而互不影响。

### 容器与虚拟机

容器，也是虚拟层的概念，相对虚拟机而言，容器更加轻量级。虚拟机中需要模拟一台物理机的所有资源，比如你要模拟出有多少 CPU、网卡、显卡等等，这些都是在软件层面通过计算资源实现的，这就给物理机凭空增加了不必要的计算量。容器仅仅在操作系统层面向上，对应用的所需各类资源进行了隔离。

{{< figure src="con-vm.png" >}}

## Docker

{{< figure src="docker.jpeg" >}}

附：[中文文档](https://www.docker.org.cn/)

Docker 是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包到一个可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。容器是完全使用沙箱机制，相互之间不会有任何接口。

### Docker 的理念

通过 Docker 官方提供的架构图来看看 Docker 对容器结构的设计。

{{< figure src="docker-design.png" >}}

### Docker 的四大对象

在 Docker 体系里，有四个对象 ( Object ) 是不得不进行介绍的，因为几乎所有 Docker 以及周边生态的功能，都是围绕着它们所展开的。它们分别是：

- 镜像 ( Image )
- 容器 ( Container )
- 网络 ( Network )
- 数据卷 ( Volume )

#### Image

镜像就是一个包含了**虚拟环境运行最原始文件系统**的内容的**只读**文件。

{{< figure src="docker-image.png" >}}

Docker 的镜像与虚拟机中的镜像还是有一定区别的。首先，Docker 中的一个创新是利用了 AUFS 作为底层文件系统实现，通过这种方式，Docker 实现了一种增量式的镜像结构。

Docker 的镜像实质上是无法被修改的，因为所有对镜像的修改只会产生新的镜像，而不是更新原有的镜像。

#### Container

类比 Oop ，镜像就像是一个类，而容器则可以理解为该类的实例。

根据官方的定义，一个 Docker 容器应该包括一下三项内容：

- 一个 Docker 镜像
- 一个程序的运行环境
- 一个指令集合

#### Network

网络通讯作为目前最常用的一种程序间的数据交换方式。 Docker 中，实现了强大的网络功能，我们不但能够十分轻松的对每个容器的网络进行配置，还能在容器间建立虚拟网络，将数个容器包裹其中，同时与其他网络环境隔离。

{{< figure src="docker-net.jpeg" >}}

#### Volume

Docker 能够这么简单的实现挂载，主要还是得益于 Docker 底层的 Union File System 技术。在 UnionFS 的加持下，除了能够从宿主操作系统中挂载目录外，还能够建立独立的目录持久存放数据，或者在容器间共享。

在 Docker 中，通过这几种方式进行数据共享或持久化的文件或目录，我们都称为数据卷 ( Volume )。

{{< figure src="docker-volume.png" >}}

## Vagrant

{{< figure src="vagrant.png" >}}

Vagrant 是一个虚拟机管理软件，可以自动化虚拟机的安装和配置流程。一般我们使用虚拟机时是这样的，安装一个虚拟机软件 VMware 或 VirtualBox，寻找我们需要的 iso 镜像文件，然后一步步的在 VMware 上安装这个镜像文件，安装好之后，再一步步配置这个这个虚拟机的开发环境或运行环境。如果我们需要安装两个或多相同的虚拟机环境怎么办？还是得这样一步步的安装？不，我们有 Vagrant。Vagrant 让你通过编写一个 Vagrantfile 配置文件来控制虚拟机的启动、销毁、与宿主机间的文件共享、虚拟机网络环境的配置，还可以编写一些虚拟机启动后的执行脚本，自动安装一些必备的开发工具或配置。 并且 Vagrant 还允许移植的你的虚拟机，使你只需要一次搭建就可以拥有多个相同环境的虚拟机副本。

### Vagrant 的优点

- 跨平台
- 可移动
- 自动化部署无需人工参与等

### Vagrant 的相关概念

Vagrant 是一个虚拟机管理软件，它是用来管理虚拟机的，它自己并不是一个虚拟机软件。这意味着你必须独立安装一个虚拟机软件 VMware 或 VirtualBox 做支持。VMware 应该都很熟不多说了。VirtualBox 是一款开源虚拟机软件。VirtualBox 是由德国 Innotek 公司开发，由 Sun Microsystems 公司出品的软件，目前在 oracle 旗下。早期，Vagrant 只支持 VirtualBox，后来才加入了 VMWare 的支持。当然目前也支持 KVM，目前我的学习环境就是选择 KVM 作为虚拟化工具。

#### .box 文件

这是 Vagrant 使用的镜像，它与传统的镜像不太一样，它不是待安装的系统镜像，而是从虚拟机中导出的、对已经安装配置好的操作系统的快照。.box 文件是个压缩文件包，里面除了基础数据的镜像外，还包括一些元数据文件，用来指导 Vagrant 将系统镜像正确的加载到对应的虚拟机当中中。所以这边要注意 box 文件是依赖虚拟机软件的，比如 VMware 下的 box 文件是无法在 VirtualBox 上使用的。

#### vagrantfile

vagrantfile 是一个配置文件。每一个 vagrantfile 对应一个 box 镜像，在初始化虚拟机的时候会制自动生成 vagrantfile 文件。我们可以在 vagrantfile 文件里添加启动的脚本，网络配置，启动后的脚本等等。在启动时 vagrant 就会读取相应的 vagrantfile 中的指令来搭建虚拟机环境。Vagrant 是使用 Ruby 开发的，所以它的 vagrantfile 配置文件语法也是 Ruby 的，但是 Vagrant 定义了自己的语法规则，我们也没有必要去学习 Ruby。关于更多 vagrantfile 的配置文件信息后面会单独介绍，这边就不复述了。 另外默认 vagrantfile 所在的目录，就是宿主机同虚拟机共享的目录。

## Vagrant 与 Docker

{{< figure src="featured-image.jpg" >}}

首先，相似之处是 Vagrant 和 Docker 都是虚拟化技术。Vagrant 是基于 Virtualbox 的虚拟机来构建你的开发环境，而 Docker 则是基于 LXC(LXC)轻量级容器虚拟技术。全面理解这两种虚拟技术的区别，需要阅读很多文档。我这里打个简单的比方，虚拟机之于容器虚拟技术相当于进程和线程。虚拟机内可以包含很多容器，正如一个进程中可以包含很多线程。虚拟机重，容器虚拟技术轻。

前者的 Image 一般以 GB 计算，Docker 则以 100MB 为单位计算。当然，提问者肯定更希望从应用层面来了解两者的区别。简单点讲，Vagrant 就是你的开发环境的部署工具；而 docker 是你的运行环境部署工具。

个人感觉 Vagrant 和 Docker 类似，通过组合底层开源技术加上自己开发的程序/脚本以及良好定义的规则，实现了标准化的快速部署虚拟系统的架构。所不同的是，Docker 采用的是轻量级的容器技术，Vagrant 则使用 kvm/virtualbox 等重量级的全/半虚拟化平台。

Vagrant 实现的是根据模版快速提供虚拟系统，大规模生产虚拟系统，需要结合进一步定制的管理控制平台来实现 PaaS。

Vagrant 更适合部署完全虚拟化集群，可以实现更为复杂的系统模拟，在特定的需要安全性隔离环境以及需要实现完整的操作系统功能的虚拟系统。
