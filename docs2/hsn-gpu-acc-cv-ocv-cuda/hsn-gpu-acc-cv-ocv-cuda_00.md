# 前言

计算机视觉正在改变着众多行业，OpenCV 是计算机视觉中最广泛选择的工具，它能够在多种编程语言中工作。如今，在计算机视觉中实时处理大图像的需求日益增长，这对于仅凭 OpenCV 本身来说是难以处理的。在这种情况下，图形处理单元（GPU）和 CUDA 可以提供帮助。因此，本书提供了关于将 OpenCV 与 CUDA 集成以用于实际应用的详细概述。它从解释使用 CUDA 进行 GPU 编程开始，这对于从未使用过 GPU 的计算机视觉开发者来说是必不可少的。然后，通过一些实际示例解释了使用 GPU 和 CUDA 加速 OpenCV 的过程。当计算机视觉应用需要在现实场景中使用时，它需要部署在嵌入式开发板上。本书涵盖了在 NVIDIA Jetson Tx1 上部署 OpenCV 应用，这对于计算机视觉和深度学习应用非常受欢迎。本书的最后一部分涵盖了 PyCUDA 的概念，它可供使用 Python 与 OpenCV 一起工作的计算机视觉开发者使用。PyCUDA 是一个 Python 库，它利用 CUDA 和 GPU 的强大功能进行加速。本书为使用 OpenCV 在 C++或 Python 中加速计算机视觉应用的开发者提供了一个完整的指南，采用了一种动手实践的方法。

# 本书面向的对象

本书是针对那些正在使用 OpenCV 的开发者，他们现在想通过利用 GPU 处理的优势来学习如何处理更复杂图像数据。大多数计算机视觉工程师或开发者在尝试实时处理复杂图像数据时都会遇到问题。这就是使用 GPU 加速计算机视觉算法可以帮助他们开发出能够在实时处理复杂图像数据的算法的地方。大多数人认为，硬件加速只能通过 FPGA 和 ASIC 设计来实现，为此，他们需要了解硬件描述语言，如 Verilog 或 VHDL。然而，在 CUDA 发明之前，这种情况是真实的，CUDA 利用了 Nvidia GPU 的力量，可以通过使用 C++和 Python 等编程语言来加速算法。本书将帮助那些开发者通过帮助他们开发实际应用来了解这些概念。本书将帮助开发者将计算机视觉应用部署在嵌入式平台，如 NVIDIA Jetson TX1 上。

# 本书涵盖的内容

[第一章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=24&action=edit)，*CUDA 简介与 CUDA 入门*，介绍了 CUDA 架构以及它是如何重新定义了 GPU 的并行处理能力的。讨论了 CUDA 架构在现实场景中的应用。读者被介绍到用于 CUDA 的开发环境以及如何在所有操作系统上安装它。

[第二章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=26&action=edit)，*使用 CUDA C 进行并行编程*，教读者使用 CUDA 为 GPU 编写程序。它从一个简单的 Hello World 程序开始，然后逐步构建到 CUDA C 中的复杂示例。它还涵盖了内核的工作原理以及如何使用设备属性，并讨论了与 CUDA 编程相关的术语。 

[第三章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=26&action=edit)，*线程、同步和内存*，教读者关于如何在 CUDA 程序中调用线程以及多个线程如何相互通信。它描述了当多个线程并行工作时如何进行同步。它还详细描述了常量内存和纹理内存。

[第四章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=27&action=edit)，*CUDA 的高级概念*，涵盖了 CUDA 流和 CUDA 事件等高级概念。它描述了如何使用 CUDA 加速排序算法，并探讨了使用 CUDA 加速简单的图像处理函数。

[第五章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=28&action=edit)，*使用 CUDA 支持的 OpenCV 入门*，描述了在所有操作系统上安装具有 CUDA 支持的 OpenCV 库。它解释了如何使用一个简单的程序来测试这个安装。本章还比较了带有和没有 CUDA 支持执行图像处理程序的性能。

[第六章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=29&action=edit)，*使用 OpenCV 和 CUDA 进行基本计算机视觉操作*，教读者如何使用 OpenCV 编写基本的计算机视觉操作，例如图像的像素级操作、过滤和形态学操作。

[第七章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=30&action=edit)，*使用 OpenCV 和 CUDA 进行目标检测和跟踪*，探讨了使用 OpenCV 和 CUDA 加速一些实际计算机视觉应用的步骤。它描述了用于目标检测的特征检测和描述算法。本章还涵盖了使用 Haar 级联和视频分析技术（如背景减法进行目标跟踪）的加速人脸检测。

[第八章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=31&action=edit)，*Jetson TX1 开发板简介和 Jetson TX1 上安装 OpenCV*，介绍了 Jetson TX1 嵌入式平台及其如何用于加速和部署计算机视觉应用。它描述了使用 Jetpack 在 Jetson TX1 上为 Tegra 安装 OpenCV 的过程。

[第九章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=32&action=edit)，*在 Jetson TX1 上部署计算机视觉应用*，涵盖了在 Jetson Tx1 上部署计算机视觉应用。它教读者如何构建不同的计算机视觉应用以及如何将摄像头与 Jetson Tx1 接口用于视频处理应用。

[第十章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=33&action=edit)，*开始使用 PyCUDA*，介绍了 PyCUDA，这是一个用于 GPU 加速的 Python 库。它描述了在所有操作系统上的安装过程。

[第十一章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=34&action=edit)，*使用 PyCUDA 进行工作*，教读者如何使用 PyCUDA 编写程序。它详细描述了从主机到设备的数据传输和内核执行的概念。它涵盖了如何在 PyCUDA 中处理数组以及开发复杂算法。

[第十二章](https://cdp.packtpub.com/hands_on_gpu_accelerated_computer_vision_with_opencv_and_cuda/wp-admin/post.php?post=35&action=edit)，*使用 PyCUDA 开发基本计算机视觉应用*，探讨了使用 PyCUDA 开发和加速基本计算机视觉应用。它以颜色空间转换操作、直方图计算和不同的算术运算为例，描述了计算机视觉应用。

# 要充分利用本书

本书涵盖的示例可以在 Windows、Linux 和 macOS 上运行。所有安装说明都在书中涵盖。预期读者对计算机视觉概念和 C++、Python 等编程语言有深入理解。建议读者拥有 Nvidia GPU 硬件来执行书中涵盖的示例。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载后，请确保您使用最新版本的以下软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA`](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包可供在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上获取。查看它们！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/978-1-78934-829-3_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/978-1-78934-829-3_ColorImages.pdf)。

# 代码实战

访问以下链接查看代码运行的视频：

[`bit.ly/2PZOYcH`](http://bit.ly/2PZOYcH)

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“将下载的`WebStorm-10*.dmg`磁盘映像文件作为系统中的另一个磁盘挂载。”

代码块设置如下：

```py
html, body, #map {
 height: 100%; 
 margin: 0;
 padding: 0
}
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
[default]
exten => s,1,Dial(Zap/1|30)
exten => s,2,Voicemail(u100)
exten => s,102,Voicemail(b100)
exten => i,1,Voicemail(s0)
```

任何命令行输入或输出都按以下方式编写：

```py
$ mkdir css
$ cd css
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要提示看起来像这样。

技巧和窍门看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`发送电子邮件给我们。

**勘误表**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/).

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/).
