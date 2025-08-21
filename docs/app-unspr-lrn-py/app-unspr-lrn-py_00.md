# 前言

## 关于

本节简要介绍了作者、本书的内容覆盖范围、开始时你需要的技术技能，以及完成所有活动和练习所需的硬件和软件要求。

## 本书简介

无监督学习是一种在没有标签数据的情况下非常有用且实用的解决方案。

*Python 应用无监督学习* 引导你使用无监督学习技术与 Python 库配合，提取非结构化数据中的有意义信息的最佳实践。本书首先解释了基本的聚类如何工作，以在数据集中找到相似的数据点。一旦你熟悉了 k-means 算法及其操作方式，你将学习什么是降维以及如何应用它。随着学习的深入，你将掌握各种神经网络技术，以及它们如何提升你的模型。研究无监督学习的应用时，你还将学习如何挖掘 Twitter 上的热门话题。你将通过完成各种有趣的活动来挑战自己，例如进行市场购物篮分析，并识别不同产品之间的关系。

到本书的最后，你将掌握使用 Python 自信地构建自己模型所需的技能。

### 作者简介

**Benjamin Johnston** 是世界领先的数据驱动医疗科技公司之一的高级数据科学家，参与了整个产品开发过程中的创新数字解决方案的开发，从问题定义到解决方案的研发，再到最终部署。他目前正在完成机器学习博士学位，专攻图像处理和深度卷积神经网络。他在医疗设备设计和开发领域有超过 10 年的经验，担任过多种技术角色，拥有澳大利亚悉尼大学工程学和医学科学两项一等荣誉学士学位。

**Aaron Jones** 是美国一家大型零售商的全职高级数据科学家，同时也是一名统计顾问。他在零售、媒体和环境科学领域工作时，建立了预测性和推断性模型以及多个数据产品。Aaron 居住在华盛顿州的西雅图，特别关注因果建模、聚类算法、自然语言处理和贝叶斯统计。

**Christopher Kruger** 曾在广告领域担任高级数据科学家。他为不同行业的客户设计了可扩展的聚类解决方案。Chris 最近获得了康奈尔大学计算机科学硕士学位，目前在计算机视觉领域工作。

### 学习目标

+   理解聚类的基础知识和重要性

+   从零开始构建 k-means、层次聚类和 DBSCAN 聚类算法，并使用内置包实现

+   探索降维及其应用

+   使用 scikit-learn（sklearn）实现并分析鸢尾花数据集上的主成分分析（PCA）

+   使用 Keras 构建 CIFAR-10 数据集的自编码器模型

+   使用机器学习扩展（Mlxtend）应用 Apriori 算法研究交易数据

### 受众

*Python 应用无监督学习*是为开发人员、数据科学家和机器学习爱好者设计的，旨在帮助他们了解无监督学习。具有一定的 Python 编程基础，以及包括指数、平方根、均值和中位数等数学概念的基本知识将会非常有帮助。

### 方法

*Python 应用无监督学习*采用实践操作的方式，使用 Python 揭示您非结构化数据中的隐藏模式。它包含多个活动，利用现实生活中的商业场景，帮助您在高度相关的环境中练习并应用您的新技能。

### 硬件要求

为了获得最佳的学生体验，我们推荐以下硬件配置：

+   处理器：Intel Core i5 或同等配置

+   内存：4 GB RAM

+   存储：5 GB 可用空间

### 软件要求

我们还建议您提前安装以下软件：

+   操作系统：Windows 7 SP1 64 位，Windows 8.1 64 位，或 Windows 10 64 位；Linux（Ubuntu，Debian，Red Hat 或 Suse）；或最新版本的 OS X

+   Python（3.6.5 或更高版本，最好是 3.7；通过[`www.python.org/downloads/release/python-371/`](https://www.python.org/downloads/release/python-371/)可获得）

+   Anaconda（这是用于`mlp_toolkits`中的`basemap`模块的；请访问[`www.anaconda.com/distribution/`](https://www.anaconda.com/distribution/)，下载 3.7 版本并按照说明进行安装。）

### 约定

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名以如下方式显示：“使用`math`包没有先决条件，并且它已包含在所有标准 Python 安装中。”

一段代码的写法如下：

```py
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import math
%matplotlib inline
```

新术语和重要词汇以粗体显示。您在屏幕上看到的词汇，例如在菜单或对话框中显示的内容，会以这种形式出现在文本中：“接下来，点击`metro-jul18-dec18`。”

### 安装与设置

每一段伟大的旅程都始于一个谦逊的步伐。我们即将展开的无监督学习之旅也不例外。在我们能够利用数据做出惊人的成就之前，我们需要准备好最有效的工作环境。接下来，我们将了解如何做到这一点。

### 在 Windows 上安装 Anaconda

Anaconda 是一个 Python 包管理器，能够轻松地安装并使用本书所需的库。要在 Windows 上安装它，请按照以下步骤进行：

1.  Windows 上的 Anaconda 安装非常用户友好。请访问下载页面以获取安装可执行文件：[`www.anaconda.com/distribution/#download-section`](https://www.anaconda.com/distribution/#download-section)。

1.  双击计算机上的安装程序。

1.  按照屏幕上的提示完成 Anaconda 的安装。

1.  安装完成后，你可以访问 Anaconda Navigator，它将像其他应用程序一样出现在你的应用列表中。

### 在 Linux 上安装 Anaconda

Anaconda 是一个 Python 包管理器，可以轻松地安装并使用本书所需的库。在 Linux 上安装它，请按照以下步骤操作：

1.  请访问 Anaconda 下载页面以获取安装 shell 脚本：[`www.anaconda.com/distribution/#download-section`](https://www.anaconda.com/distribution/#download-section)。

1.  要直接将 shell 脚本下载到你的 Linux 实例中，可以使用 `curl` 或 `wget` 下载库。以下示例演示了如何使用 `curl` 从 Anaconda 下载页面找到的 URL 获取文件：

    ```py
    curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    ```

1.  下载 shell 脚本后，可以使用以下命令运行它：

    ```py
    bash Anaconda3-2019.03-Linux-x86_64.sh
    ```

1.  运行上述命令将引导你进入一个非常用户友好的安装过程。系统会提示你选择安装位置以及你希望 Anaconda 如何工作。在这种情况下，你只需保留所有标准设置即可。

1.  安装 Anaconda 后，你必须创建环境，在这些环境中你可以安装你希望使用的包。Anaconda 环境的一个巨大优点是，你可以为你正在进行的特定项目创建独立的环境！要创建一个新环境，使用以下命令：

    ```py
    conda create --name my_packt_env python=3.7
    ```

1.  一旦环境创建完成，你可以使用命名明确的 `activate` 命令激活它：

    ```py
    conda activate my_env
    ```

    就这样！你现在已经进入了自己的自定义环境，这将允许你根据需要为你的项目安装所需的包。要退出环境，你只需使用 `conda deactivate` 命令。

### 在 macOS 上安装 Anaconda

Anaconda 是一个 Python 包管理器，允许你轻松安装并使用本书所需的库。在 macOS 上安装它，请按照以下步骤操作：

1.  Windows 上的 Anaconda 安装非常用户友好。请访问下载页面以获取安装可执行文件：[`www.anaconda.com/distribution/#download-section`](https://www.anaconda.com/distribution/#download-section)。

1.  确保选择 macOS，并双击 **Download** 按钮以下载 Python 3 安装程序。

1.  按照屏幕上的提示完成 Anaconda 的安装。

1.  安装完成后，你可以访问 Anaconda Navigator，它将像其他应用程序一样出现在你的应用列表中。

### 在 Windows 上安装 Python

1.  在此处查找你所需的 Python 版本：[`www.python.org/downloads/windows/`](https://www.python.org/downloads/windows/)。

1.  确保根据您的计算机系统安装正确的“位”版本，可以是 32 位或 64 位。您可以在操作系统的系统属性窗口中找到该信息。

    下载安装程序后，只需双击文件并按照屏幕上的友好提示进行操作。

### 在 Linux 上安装 Python

在 Linux 上安装 Python，请执行以下操作：

1.  打开命令提示符并通过运行 `python3 --version` 验证 Python 3 是否已经安装。

1.  要安装 Python 3，请运行以下命令：

    ```py
    sudo apt-get update
    sudo apt-get install python3.6
    ```

1.  如果遇到问题，网络上有大量资源可以帮助您排除故障。

### 在 macOS X 上安装 Python

在 macOS X 上安装 Python，请执行以下操作：

1.  按住 *CMD* + *Space*，在打开的搜索框中输入 `terminal`，然后按 *Enter* 打开终端。

1.  通过运行 `xcode-select --install` 在命令行中安装 Xcode。

1.  安装 Python 3 最简单的方法是使用 homebrew，可以通过运行 `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"` 来安装 homebrew。

1.  将 homebrew 添加到您的 `PATH` 环境变量中。通过运行 `sudo nano ~/.profile` 打开命令行中的配置文件，并在文件底部插入 `export PATH="/usr/local/opt/python/libexec/bin:$PATH"`。

1.  最后一步是安装 Python。在命令行中运行 `brew install python`。

1.  请注意，如果安装 Anaconda，最新版本的 Python 将自动安装。

### 附加资源

本书的代码包也托管在 GitHub 上，网址为：https://github.com/TrainingByPackt/Applied-Unsupervised-Learning-with-Python。我们还提供了来自我们丰富书籍和视频目录的其他代码包，您可以在 https://github.com/PacktPublishing/ 上查看它们！

我们还提供了一个 PDF 文件，里面包含本书中使用的带色彩的截图/图表。您可以在此处下载： https://www.packtpub.com/sites/default/files/downloads/9781789952292_ColorImages.pdf。
