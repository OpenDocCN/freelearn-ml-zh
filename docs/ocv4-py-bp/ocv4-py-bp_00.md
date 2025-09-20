# 前言

本书的目标是让你通过使用最新版本的 **OpenCV 4** 框架和 **Python 3.8** 语言，在一系列中级到高级的项目中亲自动手，而不是仅仅在理论课程中涵盖计算机视觉的核心概念。

这本更新的第二版增加了我们用 OpenCV 解决的概念深度。它将引导你通过独立的手动项目，专注于图像处理、3D 场景重建、目标检测和目标跟踪等基本计算机视觉概念。它还将通过实际案例涵盖统计学习和深度神经网络。

你将从理解图像滤镜和特征匹配等概念开始，以及使用自定义传感器，如**Kinect 深度传感器**。你还将学习如何重建和可视化 3D 场景，如何对齐图像，以及如何将多个图像合并成一个。随着你在书中的进步，你将学习如何使用神经网络识别交通标志和面部表情，即使在物体短暂消失的情况下也能检测和跟踪视频流中的物体。

在阅读完这本 OpenCV 和 Python 书籍之后，你将拥有实际操作经验，并能熟练地根据特定业务需求开发自己的高级计算机视觉应用。在整个书中，你将探索多种机器学习和计算机视觉模型，例如**支持向量机**（**SVMs**）和卷积神经网络。

# 本书面向的对象

本书面向的是追求通过使用 OpenCV 和其他机器学习库开发高级实用应用来掌握技能的计算机视觉爱好者。

假设你具备基本的编程技能和 Python 编程知识。

# 本书涵盖的内容

第一章，*与滤镜的乐趣*，探讨了多种有趣的图像滤镜（例如黑白铅笔素描、暖色/冷色滤镜和卡通化效果），并将它们实时应用于网络摄像头的视频流中。

第二章，*使用 Kinect 深度传感器进行手势识别*，帮助你开发一个应用，实时检测和跟踪简单的手势，使用深度传感器的输出，如微软 Kinect 3D 传感器或华硕 Xtion。

第三章，*通过特征匹配和透视变换查找对象*，帮助你开发一个应用，在摄像头的视频流中检测感兴趣的任意对象，即使对象从不同的角度或距离观看，或者部分遮挡。

第四章，*使用运动结构进行 3D 场景重建*，展示了如何通过从相机运动中推断其几何特征来重建和可视化 3D 场景。

第五章，*使用 OpenCV 进行计算摄影*，帮助你开发命令行脚本，这些脚本以图像为输入并生成全景图或**高动态范围**（**HDR**）图像。这些脚本将图像对齐，以实现像素到像素的对应，或者将它们拼接成全景图，这是图像对齐的一个有趣应用。在全景图中，两个图像不是平面的，而是三维场景的图像。一般来说，3D 对齐需要深度信息。然而，当两个图像是通过围绕其光学轴旋转相机拍摄的（如全景图的情况），我们可以对齐全景图中的两个图像。

第六章，*跟踪视觉显著对象*，帮助你开发一个应用，可以同时跟踪视频序列中的多个视觉显著对象（如足球比赛中的所有球员）。

第七章，*学习识别交通标志*，展示了如何训练支持向量机从**德国交通标志识别基准**（**GTSRB**）数据集中识别交通标志。

第八章，*学习识别面部表情*，帮助你开发一个能够在实时网络摄像头视频流中检测面部并识别其情感表达的应用程序。

第九章，*学习识别面部表情*，引导你开发一个使用深度卷积神经网络进行实时对象分类的应用程序。你将修改一个分类网络，使用自定义数据集和自定义类别进行训练。你将学习如何在数据集上训练 Keras 模型以及如何将你的 Keras 模型序列化和保存到磁盘。然后，你将看到如何使用加载的 Keras 模型对新的输入图像进行分类。你将使用你拥有的图像数据训练卷积神经网络，以获得一个具有非常高的准确率的良好分类器。

第十章，*学习检测和跟踪对象*，指导你开发一个使用深度神经网络进行实时对象检测的应用程序，并将其连接到跟踪器。你将学习对象检测器是如何工作的以及它们是如何训练的。你将实现一个基于卡尔曼滤波器的跟踪器，它将使用对象位置和速度来预测其可能的位置。完成本章后，你将能够构建自己的实时对象检测和跟踪应用程序。

附录 A，*分析和加速你的应用*，介绍了如何找到应用中的瓶颈，并使用 Numba 实现现有代码的 CPU 和 CUDA 基于 GPU 的加速。

附录 B，*设置 Docker 容器*，将指导您复制我们用于运行本书中代码的环境。

# 为了充分利用本书

我们所有的代码都使用**Python 3.8**，它可以在多种操作系统上使用，例如**Windows**、**GNU Linux**、**macOS**以及其他操作系统。我们已尽力只使用这三个操作系统上可用的库。我们将详细介绍我们所使用的每个依赖项的确切版本，这些依赖项可以使用`pip`（Python 的依赖项管理系统）安装。如果您在使用这些依赖项时遇到任何问题，我们提供了 Dockerfile，其中包含了我们测试本书中所有代码的环境，具体内容在附录 B，*设置 Docker 容器*中介绍。

这里是我们使用过的依赖项列表，以及它们所使用的章节：

| **所需软件**  | **版本** | **章节编号** | **软件下载链接** |
| --- | --- | --- | --- |
| Python | 3.8 | All | [`www.python.org/downloads/`](https://www.python.org/downloads/) |
| OpenCV | 4.2 | All | [`opencv.org/releases/`](https://opencv.org/releases/) |
| NumPy | 1.18.1 | All | [`www.scipy.org/scipylib/download.html`](http://www.scipy.org/scipylib/download.html) |
| wxPython | 4.0 | 1, 4, 8  | [`www.wxpython.org/download.php`](http://www.wxpython.org/download.php) |
| matplotlib | 3.1 | 4, 5, 6, 7  | [`matplotlib.org/downloads.html`](http://matplotlib.org/downloads.html) |
| SciPy | 1.4 | 1, 10  | [`www.scipy.org/scipylib/download.html`](http://www.scipy.org/scipylib/download.html) |
| rawpy | 0.14 | 5 | [`pypi.org/project/rawpy/`](https://pypi.org/project/rawpy/) |
| ExifRead | 2.1.2 | 5 | [`pypi.org/project/ExifRead/`](https://pypi.org/project/ExifRead/) |
| TensorFlow | 2.0 | 7, 9 | [`www.tensorflow.org/install`](https://www.tensorflow.org/install) |

为了运行代码，您需要一个普通的笔记本电脑或个人电脑（PC）。某些章节需要摄像头，可以是内置的笔记本电脑摄像头或外置摄像头。第二章，*使用 Kinect 深度传感器进行手势识别*也要求一个深度传感器，可以是**Microsoft 3D Kinect 传感器**或任何其他由`libfreenect`库或 OpenCV 支持的传感器，例如**ASUS Xtion**。

我们使用**Python 3.8**和**Python 3.7**在**Ubuntu 18.04**上进行了测试。

如果您已经在您的计算机上安装了 Python，您可以直接在终端运行以下命令：

```py
$ pip install -r requirements.txt
```

在这里，`requirements.txt`文件已提供在项目的 GitHub 仓库中，其内容如下（这是之前给出的表格以文本文件的形式）：

```py
wxPython==4.0.5
numpy==1.18.1
scipy==1.4.1
matplotlib==3.1.2
requests==2.22.0
opencv-contrib-python==4.2.0.32
opencv-python==4.2.0.32
rawpy==0.14.0
ExifRead==2.1.2
tensorflow==2.0.1
```

或者，您也可以按照附录 B 中的说明，*设置 Docker 容器*，以使用 Docker 容器使一切正常工作。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](https://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载”。

1.  在搜索框中输入书名，并按照屏幕上的说明操作。

文件下载后，请确保您使用最新版本的软件解压或提取文件夹，例如：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition`](https://github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们！

# 代码实战

本书“代码实战”视频可以在[`bit.ly/2xcjKdS`](http://bit.ly/2xcjKdS)查看。

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`static.packt-cdn.com/downloads/9781789801811_ColorImages.pdf`](http://static.packt-cdn.com/downloads/9781789801811_ColorImages.pdf)。

# 约定如下

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“我们将使用`argparse`，因为我们希望我们的脚本接受参数。”

代码块设置如下：

```py
import argparse

import cv2
import numpy as np

from classes import CLASSES_90
from sort import Sort
```

任何命令行输入或输出都应如下编写：

```py
$ python chapter8.py collect
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下所示。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要注意事项如下所示。

小贴士和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`发送给我们。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们将不胜感激，如果你能向我们报告这一点。请访问[www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择你的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**: 如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/).

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/).
