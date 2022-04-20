# Vagrant


使用 Vagrant 搭建开发测试环境

<!--more-->

# Vagrant

Vagrant 基于工业标准化技术至上提供了易于配置，可重复部署以及可移植的工作环境，并且通过一个简单的一致性的工作流提供控制来帮助你和你的团队最大化生产力和可伸缩性。

为了实现这个奇迹，Vagrant 站在了巨人的肩膀之上。主机由 VirtualBox，VMware，AWS 或其他 provider 提供。然后，工业标准化的供给工具，例如 shell 脚本，Chef 或 Puppet，可以提供自动化安装和配置主机中的软件。

## Vagrant 优势

对于开发者而言，Vagrant 隔离了依赖和它们的配置到一个单一的一次性的，一致性的环境，不需要担忧任何使用的工具。一旦你或者其他人创建了一个 Vagrant 文件，你只需要使用 vagrant up 然后，所有软件被安装和配置。其他团队成员也适用相同的配置文件创建他们的环境，无论你们是工作在 Linux， Mac OS X 或 Windows 。所有团队成员在相同环境中运行代码，基于相同的依赖。

## 使用 Vagrant 部署环境

### Get Started

- 安装 `vagrant` ，对于 `archlinux` 可以使用 `pacman` 直接安装 `vagrant` 和 `virtualbox`

```
$ sudo pacman -S vagrant virtualbox
```

对于其他发行版，可以自行到官网下载安装：

- [VirtualBox](https://www.virtualbox.org/wiki/Downloads)
- [Vagrant](https://www.vagrantup.com/downloads.html)

* 创建一个虚拟机：

首先，到 [Vagrant Cloud](https://app.vagrantup.com/boxes/search) 找到需要安装的虚拟机，注意选择 `virtualbox` 格式的 box。

```
$ mkdir ~/Vagrant/ubuntu2004
$ cd ~/Vagrant/ubuntu2004
$ vagrant init generic/ubuntu2004
$ vagrant up
```

虚拟机搭建完成后，通过`vagrant ssh` 即可进行连接。

> - **Notice**：对于 `kitty terminal` ，需要使用 `TERM=xterm-256color vagrant ssh`，问题具体描述，参见：[reddit](https://www.reddit.com/r/KittyTerminal/comments/mgd2f0/vagrant_ssh/)

### Vagrantfile

[Vagrantfile](https://www.vagrantup.com/docs/vagrantfile)

配置任何 Vagrant 项目的第一步是创建一个 Vagrantfile。这个 Vagrantfile 两个作用：

- 标记项目的根目录，一些 Vagrant 配置归属到这个根目录中
- 描述主机类型和需要运行项目的资源，例如软件安装以及如何访问它

Vagrant 有一个内建命令来初始化目录 `vagrant init` 。这个命令将 Vagrantfile 放到当前目录，可以查看这个文件（包含了很多注释和例子）。也可以在已经存在的目录执行 `vagrant init` 来设置 Vagrant。

> - Vagrantfile 实际是使用 [Ruby](http://www.ruby-lang.org/) 语法，但如果不熟悉 `Ruby`，也无伤大雅，没有必要为此学习 `Ruby`。

### Box

Vagrant 使用一个基础镜像来 clone 虚拟机，这个基础镜像称为 boxes。

### 修改 synced 目录

可以通过修改 Vagrantfile 配置来实现对默认共享目录位置的修改，例如

```ruby
config.vm.synced_folder "/home/allen/Workplace/eBPF/Vagrant/allen", "/allen"
```

可以用来在不同的虚拟机之间使用共享目录，这样方便部署一个简单的共享集群。即，在物理服务器上构建一个共享给多个虚拟机使用的网络存储（不需要单独部署 NFS 或者 CIFS），方便构建测试集群。

## Vagrant Package

可以使用 `vagrant package` 是一个将当前的虚拟环境打包成一个可以重用的 [box](http://docs.vagrantup.com/v2/boxes.html) ，但 `vagrant package` 不能用于其他的`provider`。

### 简单的打包

进入 Vagrant 项目目录，执行简单的命令

```
vagrant package
```

### 参数

- `--base NAME` - 替代打包一个 VirtualBox 主机，这个参数打包一个 VirtualBox manages 的 VirtualBox。这里 NAME 是 VirtualBox GUI 中显示的虚拟机的 UUID 或者名字。

- `--output NAME` - 这个参数设置打包的名字，如果没有这个参数，则默认保存为 package.box

- `--include x,y,z` - 附加一些文件到 box 中，这是让打包 Vagrantfile 执行附加任务

- `--vagrantfile FILE` - 打包一个 Vagrantfile 到 box 中，这个 Vagrantfile 将作为 box 使用的 Vagrantfile load

```
vagrant package --base ubuntu2004 --output ubuntu2004.box
```

## Vagrant ❤️ Libvirt

Vagrant 非常适合支持桌面级虚拟化 VirtualBox，不过在生产环境中，通常会部署 KVM 或 Xen 环境。Vagrant 通过 libvirt 支持不同的虚拟化环境，同样也包括了 KVM／Qemu。Vagrant 提供了易于部署和管理的包装，以便快速部署和方便管理 VM。

该部分是在已搭建好 KVM 的基础上进行的，KVM 搭建过程具体参考：[在 CentOS 7 中部署 KVM](https://huataihuang.gitbooks.io/cloud-atlas/content/virtual/kvm/deployment_and_administration/deploy_kvm_on_centos)

### 安装 Vagrant Plugins

- `vagrant-libvirt` 该插件用于支持 libvirt

```
vagrant plugin install vagrant-libvirt
```

> - 这需要你事先安装好 `libvirt-devel`,否则会报错。

- `vagrant-mutate` 插件将官方的 Vagrant guest box 转换成 KVM 格式

```
vagrant plugin install vagrant-mutate
```

- `vagrant-rekey-ssh` - 由于官方的 Vagrant boxes 使用了内建的非安全的 SSH key，所以我们可以使用`vagrant-rekey-ssh`插件来重新生成新的 SSH key

```
vagrant plugin install vagrant-rekey-ssh
```

### 使用 vagrant-libvirt 安装 Ubuntu20.04

- 初始化 ubuntu2004 box ([ubuntu2004](https://app.vagrantup.com/generic/boxes/ubuntu2004))

```
vagrant init generic/ubuntu2004
```

- `Vagrant`会尝试使用一个名为`default`的存储池，如果这个`default`存储池不存在就会尝试在`/var/lib/libvirt/images`上创建这个`defualt`存储池。

```
Vagrant.configure("2") do |config|
  ...
  config.vm.provider :libvirt do |libvirt|
    libvirt.storage_pool_name = "default"
  end
  ...
end
```

- 如果希望默认使用 `vagrant` 作为 `provider` 需要设置相应的环境变量

```
export VAGRANT_DEFAULT_PROVIDER=libvirt
```

- 启动安装

```
vagrant up
```

- 也可以在安装时指定`provider`

```
vagrant up --provider libvirt
```

- 查看系统的储存池

```
virsh pool-list --all
```

```
 Name                 State      Autostart
-------------------------------------------
 huatai               active     yes
 images               active     yes
 root                 active     yes
```

> - 一些故障，请参考：[使用 Vagrant 部署 kvm 虚拟化(libVirt)](https://huataihuang.gitbooks.io/cloud-atlas/content/virtual/vagrant/vagrant_libvirt_kvm.html)

## Vagrant Box 管理

Boxes 是 Vagrant 环境的打包格式。一个 box 可以被任何 Vagrant 所支持的平台的任何人用于启动一个独立工作的环境。

`vagrant box` 工具提供了所有的管理 boxes 的功能。

### 探索 Boxes

在 public Vagrant box catalog 上提供了支持各种虚拟化，如 VirtualBox，VMware，AWS 等。简单添加到本机的命令如下：

```
vagrant box add username/boxname
```

### 快速复制 Vagrant Box

当需要复制出 Vagrant box 时，简单的方法如下：

- 关闭 box (如果 box 正在运行的话)

```
vagrant halt
```

- 将环境打包

```
vagrant package --output ubuntu2004.box
```

- 创建新的 box 文件

```
mkdir devstack
cd devstack
vagrant init
```

- 编辑 `vagrantfile` ,修改如下内容：

```ruby
config.vm.box = "ubuntu2004"
config.vm.box_url = "file:///home/allen/Workplace/eBPF/vagrant/ubuntu2004.box"
config.vm.network "private_network", ip: "192.168.33.101" # 可选
```

> - 这里的 `vagrant init`，命令可以改成 `vagrant box add ubuntu2004 /home/allen/Workplace/eBPF/vagrant/ubuntu2004.box libvirt` 这样就不需要再修改 Vagrantfile 此时只需要直接运行下一步 vagrant up 就可以了。

- 运行 Vagrant box

```
vagrant up
vagrant ssh
```

### 删除 box

进入相应目录

```
cd devstack
vagrant destroy
```

执行后将从 VirtualBox 中删除掉虚拟机配置以及虚拟机相关的虚拟磁盘，真正释放空间。

然后执行 `vagrant box remove devstack` 将 Vagrant 对应的 devstack 配置清理掉。

## 使用 vagrant Snapshot 创建快照备份

使用 Vagrant 的快照功能可以很方便快速的创建当前虚拟机的一个临时备份状态，在进行重要操作时可以先创建一个快照以便在操作失误后快速恢复。

```
vagrant plugin install vagrant-snapshot
```

支持的参数如下：

```
vagrant snapshot take [vm-name] <SNAPSHOT_NAME>   # take snapshot, labeled by NAME
vagrant snapshot list [vm-name]                   # list snapshots
vagrant snapshot back [vm-name]                   # restore last taken snapshot
vagrant snapshot delete [vm-name] <SNAPSHOT_NAME> # delete specified snapshot
vagrant snapshot go [vm-name] <SNAPSHOT_NAME>     # restore specified snapshot
```

### 使用方法：

创建一个快照

```
vagrant snapshot take centos-7.1 centos-7.1_base
```

当前目录是 Vagrant 配置文件所在目录 `centos-7.1` ，我以为虚拟机的名字是 `centos-7.1` （不过看 VirtualBox 的图形管理界面中显示的是 Vagrant_default_1448250085458_10997），但是实际上述命令执行会提示错误

```
The machine with the name 'centos-7.1' was not found configured for this Vagrant environment.
```

那么真正的名字是什么呢？使用`vagrant status` 可以查看当前虚拟机的信息。

```
Current machine states:

default                   poweroff (virtualbox)

The VM is powered off. To restart the VM, simply run `vagrant up`
```

原来名字是`defualt`

```
vagrant snapshot take default centos-7.1_base
```

查看快照列表：

```
vagrant snapshot list
```

输出：

```
Listing snapshots for 'default':
   Name: centos-7.1_base (UUID: af5b803c-266e-41af-875b-9f7a2bc36794) *
```

回滚到快照：

```
vagrant snapshot go "centos-7.1_base"
```

删除快照

```
vagrant snapshot delete "centos-7.1_base"
```

