# 前言

开源计算机视觉项目，如 OpenCV 3，使各种用户能够利用机器视觉、机器学习和人工智能的力量。通过掌握这些强大的代码和知识库，专业人士和爱好者可以在任何需要的地方创建更智能、更好的应用程序。

这正是本书关注的焦点，通过一系列动手项目和模板引导你，教你如何结合出色的技术来解决你特定的难题。

在我们研究计算机视觉时，让我们从这些话中汲取灵感：

|    | *"我看见智慧胜过愚昧，正如光明胜过黑暗。" |
| --- | --- |
|    | --*传道书，2:13* |

让我们构建能够清晰“看”的应用程序，并创造知识。

# 本书涵盖的内容

第一章, *充分利用您的相机系统*，讨论了如何选择和配置相机系统以看到不可见的光、快速运动和远距离物体。

第二章, *使用自动相机拍摄自然和野生动物*，展示了如何构建自然摄影师使用的“相机陷阱”，并处理照片以创建美丽的效果。

第三章, *使用机器学习识别面部表情*，探讨了使用各种特征提取技术和机器学习方法构建面部表情识别系统的途径。

第四章, *使用 Android Studio 和 NDK 进行全景图像拼接应用*，专注于构建 Android 全景相机应用程序的项目，借助 OpenCV 3 的拼接模块。我们将使用 C++和 Android NDK。

第五章, *工业应用中的通用目标检测*，研究了优化你的目标检测模型，使其具有旋转不变性，并应用特定场景的约束，使其更快、更稳健。

第六章, *使用生物特征属性进行高效的人脸识别*，是关于基于该人的生物特征（如指纹、虹膜和面部）构建人脸识别和注册系统。

第七章，*陀螺仪视频稳定*，展示了融合视频和陀螺仪数据的技术，如何稳定使用手机拍摄的短视频，以及如何创建超高速视频。

# 您需要为此书准备的材料

作为基本设置，整本书基于 OpenCV 3 软件。如果章节没有特定的操作系统要求，那么它将在 Windows、Linux 和 Mac 上运行。作为作者，我们鼓励您从官方 GitHub 仓库的最新 master 分支([`github.com/Itseez/opencv/`](https://github.com/Itseez/opencv/))获取 OpenCV 安装，而不是使用官方 OpenCV 网站([`opencv.org/downloads.html`](http://opencv.org/downloads.html))上的可下载包，因为最新 master 分支包含与最新稳定版本相比的大量修复。

对于硬件，作者们期望您有一个基本的计算机系统设置，无论是台式机还是笔记本电脑，至少有 4GB 的 RAM 内存可用。其他硬件要求如下。

以下章节在 OpenCV 3 安装的基础上有特定的要求：

第一章，*充分利用您的相机系统*：

+   **软件**：OpenNI2 和 FlyCapture 2。

+   **硬件**：PS3 Eye 相机或任何其他 USB 网络摄像头，华硕 Xtion PRO live 或任何其他 OpenNI 兼容的深度相机，以及一个或多个镜头的 Point Grey Research (PGR)相机。

+   **备注**：PGR 相机设置（使用 FlyCapture 2）在 Mac 上无法运行。即使你没有所有必需的硬件，你仍然可以从中受益于本章的一些部分。

第二章，*使用自动相机拍摄自然和野生动物*：

+   **软件**：Linux 或 Mac 操作系统。

+   **硬件**：带有电池的便携式笔记本电脑或单板计算机（SBC），结合一台照相机。

第四章，*使用 Android Studio 和 NDK 进行全景图像拼接应用*：

+   **软件**：Android 4.4 或更高版本，Android NDK。

+   **硬件**：任何支持 Android 4.4 或更高版本的移动设备。

第七章，*陀螺仪视频稳定*：

+   **软件**：NumPy、SciPy、Python 和 Android 5.0 或更高版本，以及 Android NDK。

+   **硬件**：一部支持 Android 5.0 或更高版本的智能手机，用于捕获视频和陀螺仪信号。

# 基本安装指南

作为作者，我们承认在您的系统上安装 OpenCV 3 有时可能相当繁琐。因此，我们添加了一系列基于您系统上最新的 OpenCV 3 master 分支的基本安装指南，用于安装 OpenCV 3 以及为不同章节工作所需的必要模块。有关更多信息，请参阅[`github.com/OpenCVBlueprints/OpenCVBlueprints/tree/master/installation_tutorials`](https://github.com/OpenCVBlueprints/OpenCVBlueprints/tree/master/installation_tutorials)。

请记住，本书还使用了来自 OpenCV "contrib"（贡献）存储库的模块。安装手册将提供如何安装这些模块的说明。然而，我们鼓励您只安装我们需要的模块，因为我们知道它们是稳定的。对于其他模块，情况可能并非如此。

# 本书面向对象

如果您渴望构建比竞争对手更智能、更快、更复杂、更实用的计算机视觉系统，这本书非常适合您。这是一本高级书籍，旨在为那些已经有一定 OpenCV 开发环境和使用 OpenCV 构建应用程序经验的读者编写。您应该熟悉计算机视觉概念、面向对象编程、图形编程、IDE 和命令行。

# 习惯用法

在本书中，您将找到许多不同的文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称如下所示："您可以通过访问[`opencv.org`](http://opencv.org)并点击下载链接来找到 OpenCV 软件。"

代码块如下所示：

```py
Mat input = imread("/data/image.png", LOAD_IMAGE_GRAYSCALE);
GaussianBlur(input, input, Size(7,7), 0, 0);
imshow("image", input);
waitKey(0);
```

当我们希望将您的注意力引向代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
Mat input = imread("/data/image.png", LOAD_IMAGE_GRAYSCALE);
GaussianBlur(input, input, Size(7,7), 0, 0);
imshow("image", input);
waitKey(0);
```

任何命令行输入或输出如下所示：

**新术语**和**重要词汇**以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中如下所示："点击**下一步**按钮将您带到下一屏幕。"

### 注意

警告或重要注意事项将以如下所示的框中出现。

### 小贴士

小技巧和窍门如下所示。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢或不喜欢的地方。读者反馈对我们来说很重要，因为它帮助我们开发出您真正能从中获得最大价值的标题。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及本书的标题。

如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)下载示例代码文件，这是您购买的所有 Packt 出版物的账户。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

代码也由本书的作者在 GitHub 仓库中维护。此代码仓库可在[`github.com/OpenCVBlueprints/OpenCVBlueprints`](https://github.com/OpenCVBlueprints/OpenCVBlueprints)找到。

## 下载本书的颜色图像

我们还为您提供了一个包含本书中使用的截图/图表的颜色图像的 PDF 文件。这些颜色图像将帮助您更好地理解输出的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/B04028_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/B04028_ColorImages.pdf)下载此文件。

## 勘误

尽管我们已经尽一切努力确保我们内容的准确性，但错误仍然会发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**勘误**部分。

由于本书也有一个分配给它的 GitHub 仓库，您也可以通过在以下页面创建问题来报告内容勘误：[`github.com/OpenCVBlueprints/OpenCVBlueprints/issues`](https://github.com/OpenCVBlueprints/OpenCVBlueprints/issues)。

## 盗版

互联网上版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，请立即提供位置地址或网站名称，以便我们可以追究补救措施。

请通过<mailto:copyright@packtpub.com>与我们联系，并提供疑似盗版材料的链接。

我们感谢您在保护我们作者以及为我们提供有价值内容的能力方面提供的帮助。

## 问题

如果您在这本书的任何方面遇到问题，您可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决问题。或者，如之前所述，您可以在 GitHub 仓库中提出一个问题，作者之一将尽快帮助您。
