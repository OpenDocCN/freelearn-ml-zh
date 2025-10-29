# 前言

计算机视觉是计算机科学中最广泛研究的子领域之一。它有几个重要的应用，如人脸检测、图像搜索和艺术图像转换。随着深度学习方法的普及，许多最近的计算机视觉应用都在自动驾驶汽车、机器人、医学、虚拟现实和增强现实等领域。本书展示了学习计算机视觉的实用方法。使用代码块以及算法的理论理解将有助于建立更强的计算机视觉基础。本书教你如何使用标准工具如 OpenCV、Keras 和 TensorFlow 创建应用。本书中解释的各种概念和实现可以在多个领域使用，如机器人、图像编辑应用和自动驾驶汽车。本书中的每一章都配有相应的代码和结果，以加强学习。

# 这本书面向的对象是谁

这本书是为那些想要开始学习计算机视觉并获得这些算法的实际知识的学生和专业人士而写的。本书假设读者具备 Python 和计算机程序的基本知识，能够编写和运行 Python 脚本（包括科学 Python），并且能够理解编程所需的线性代数和基本数学。

本书将帮助读者使用图像滤波、目标检测、分割、跟踪和 SLAM 等技术设计更新的计算机视觉应用。读者将了解工业界使用的标准计算机视觉技术，以及如何编写自己的代码。他们可以像使用广泛使用的库一样进行相同的操作。他们可以利用这些技术创建跨多个领域的应用，包括图像滤波、图像处理、目标检测和深度学习的高级应用。读者将发现从了解计算机视觉到使用高级技术的过渡非常顺畅。

# 本书涵盖的内容

第一章，*快速入门计算机视觉*，简要概述了构成计算机视觉的内容，其在不同领域的应用以及不同类型问题的细分。本章还涵盖了使用 OpenCV 代码的基本图像输入读取，以及不同颜色空间及其可视化的概述。

第二章，*库、开发平台和数据集*，提供了如何设置开发环境和在其中安装库的详细说明。本章介绍的各种数据集包括本书中将使用的数据集以及目前计算机视觉各个子领域流行的数据集。本章还包括下载和加载用于库（如 Keras）的包装器的链接。

第三章，*OpenCV 中的图像滤波和变换*，解释了不同的滤波技术，包括线性和非线性滤波器，以及它们在 OpenCV 中的实现。本章还包括图像变换的技术，例如线性平移、绕给定轴旋转和完整的仿射变换。本章介绍的技术有助于在多个领域创建应用程序并提高图像质量。

第四章，*什么是特征？*，介绍了特征及其在计算机视觉各种应用中的重要性。本章包括具有基本特征的 Harris 角检测器、快速特征检测器和 ORB 特征，这些特征既鲁棒又快速。OpenCV 中还展示了使用这些特征的应用。这些应用包括将模板与原始图像匹配以及匹配同一物体的两张图像。还讨论了黑盒特征及其必要性。

第五章，*卷积神经网络*，从简单神经网络及其组件的介绍开始。本章还介绍了 Keras 中的卷积神经网络，包括激活、池化和全连接等不同组件。解释了每个组件参数变化的结果；这些结果可以很容易地由读者重现。通过使用图像数据集实现一个简单的 CNN 模型，进一步强化了这种理解。除了流行的 CNN 架构 VGG、Inception 和 ResNet 之外，还介绍了迁移学习。这导致了对图像分类最先进的深度学习模型的探讨。

第六章，*基于特征的物体检测*，发展了对图像识别问题的理解。本章使用 OpenCV 解释了检测算法，例如人脸检测器。您还将看到一些最近和流行的基于深度学习的物体检测算法，如 FasterRCNN、SSD 等。这些算法的有效性通过 TensorFlow 物体检测 API 在自定义图像上进行了说明。

第七章，*分割和跟踪*，分为两部分。第一部分介绍了图像实例识别问题，并实现了分割的深度学习模型。第二部分从介绍 OpenCV 中的 MOSSE 跟踪器开始，该跟踪器既高效又快速。在跟踪部分描述了基于深度学习的多目标跟踪。 

第八章，*3D 计算机视觉*，从几何角度描述了分析图像的方法。读者将首先了解从单张图像计算深度的挑战，然后学习如何使用多张图像来解决这些问题。本章还描述了使用视觉里程计跟踪移动相机的位姿的方法。最后，介绍了 SLAM 问题，并使用仅使用相机图像作为输入的视觉 SLAM 技术提供了解决方案。

附录 A，*计算机视觉中的数学*，介绍了理解计算机视觉算法所需的基本概念。这里介绍的张量和矩阵运算进一步通过 Python 实现进行了增强。附录还包含概率论简介，以及对各种分布的解释。

附录 B，*计算机视觉中的机器学习*，概述了机器学习建模及其涉及的各种关键术语。读者还将了解维度的诅咒，以及涉及的各种预处理和后处理。此外，还解释了用于机器学习模型的几个评估工具和方法，这些工具和方法在视觉应用中也被广泛使用。

# 为了充分利用本书

1.  本书所需的软件列表如下：

    +   Anaconda 发行版 v5.0.1

    +   OpenCV v3.3.0

    +   TensorFlow v1.4.0

    +   Keras v2.1.2

1.  为了有效地运行所有代码，建议使用 Ubuntu 16.04 操作系统，配备 Nvidia GPU 和至少 4 GB 的 RAM。代码在没有 GPU 支持的情况下也可以运行。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择 SUPPORT 标签页。

1.  点击代码下载与勘误表。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

下载文件后，请确保您使用最新版本解压缩或提取文件夹，以下是一些推荐的工具：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Practical-Computer-Vision`](https://github.com/PacktPublishing/Practical-Computer-Vision)。我们还有其他来自我们丰富图书和视频目录的代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。去看看吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/PracticalComputerVision_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/PracticalComputerVision_ColorImages.pdf).

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“这将把 Python 库安装到`$HOME/anaconda3`文件夹中，因为我们正在使用 Python 3。还有一个 Python 2 版本，安装过程类似。要使用 Anaconda，需要将新安装的库添加到`$PATH`中，这可以在每次启动新 shell 时完成。”

代码块设置如下：

```py
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
from sklearn.metrics import f1_score
true_y = .... # ground truth values
pred_y = .... # output of the model

f1_value = f1_score(true_y, pred_y, average='micro')
```

任何命令行输入或输出都写成如下：

```py
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

**粗体**: 表示新术语、重要单词或您在屏幕上看到的单词。

警告或重要注意事项看起来是这样的。

小贴士和技巧看起来是这样的。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**: 请通过`feedback@packtpub.com`发送电子邮件，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过`questions@packtpub.com`发送电子邮件给我们。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一点。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**: 如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/).

# 评论

请留下评论。一旦您阅读并使用了这本书，为什么不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/).
