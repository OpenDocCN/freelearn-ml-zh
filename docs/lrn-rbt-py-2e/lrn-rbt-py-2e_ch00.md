# 前言

*使用 Python 学习机器人学*包含九个章节，解释了如何从头开始构建一个自主移动机器人，并使用 Python 对其进行编程。本书中提到的机器人是一种服务机器人，可用于在家、酒店和餐厅提供食物。从开始到结束，本书讨论了构建该机器人的逐步过程。本书从机器人学的基本概念开始，然后转向机器人的 3D 建模和模拟。在成功模拟机器人后，它讨论了构建机器人原型所需的硬件组件。

该机器人的软件部分主要使用 Python 编程语言和软件框架实现，例如机器人操作系统（ROS）和 OpenCV。您可以从机器人的设计到创建机器人用户界面的过程中看到 Python 的应用。Gazebo 模拟器用于模拟机器人，机器视觉库，如 OpenCV、OpenNI 和 PCL，用于处理 2D 和 3D 图像数据。每个章节都提供了足够的理论来理解应用部分。本书由该领域的专家评审，是他们在机器人领域的辛勤工作和热情的成果。 

# 本书面向对象

*使用 Python 学习机器人学*是想要探索服务机器人领域的创业者、想在机器人上实现更多功能的专业人士、想要在机器人领域进行更多探索的研究人员，以及想要学习机器人的爱好者或学生的良好伴侣。本书遵循逐步指南，任何人都可以轻松掌握。

# 本书涵盖内容

第一章，*开始使用机器人操作系统*，解释了 ROS 的基本概念，ROS 是编程机器人的主要平台。

第二章，*理解差动机器人的基础知识*，讨论了差动移动机器人的基本概念。这些概念是差动驱动的运动学和逆运动学。这将帮助您在软件中实现差动驱动控制器。

第三章，*建模差动驱动机器人*，讨论了机器人设计约束的计算以及该移动机器人的 2D/3D 建模。2D/3D 建模基于一组机器人要求。完成设计和机器人建模后，读者将获得可用于创建机器人模拟的设计参数。

第四章，*使用 ROS 模拟差动驱动机器人*，介绍了一个名为 Gazebo 的机器人模拟器，并帮助读者使用它来模拟自己的机器人。

第五章，*设计 ChefBot 硬件和电路*，讨论了构建 Chefbot 所需的不同硬件组件的选择。

第六章，*将执行器和传感器连接到机器人控制器*，讨论了将不同的执行器和传感器与 Tiva C Launchpad 控制器连接。

第七章，*将视觉传感器与 ROS 接口*，讨论了将不同的视觉传感器（如 Kinect 和 Orbecc Astra）与 Chefbot 接口，这些传感器可用于自主导航。

第八章，*构建 ChefBot 硬件和 ROS 中的软件集成*，讨论了在 ROS 中实现自主导航的机器人硬件和软件的完整构建。

第九章，*使用 Qt 和 Python 为机器人设计 GUI*，讨论了开发一个 GUI 来命令机器人在类似酒店的环境中移动到桌子上。

# 为了充分利用本书

本书主要关于构建机器人；要开始阅读本书，您应该有一些硬件。机器人可以从零开始构建，或者您可以购买带有编码器反馈的差速驱动配置机器人。您应该购买一个控制器板，例如德州仪器的 LaunchPad 用于嵌入式处理，并且至少需要一台笔记本电脑/上网本用于整个机器人的处理。在本书中，我们使用 Intel NUC 进行机器人处理，它体积非常紧凑，性能出色。对于 3D 视觉，您应该有一个 3D 传感器，例如激光扫描仪、Kinect 或 Orbecc Astra。

在软件部分，您应该对使用 GNU/Linux 命令有良好的理解，并且对 Python 也有很好的知识。您应该安装 Ubuntu 16.04 LTS 来使用示例。如果您了解 ROS、OpenCV、OpenNI 和 PCL，这将有所帮助。您必须安装 ROS Kinect/Melodic 来使用这些示例。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载与勘误。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书代码包托管在 GitHub 上，网址为[`github.com/PacktPublishing/Learning-Robotics-using-Python-Second-Edition`](https://github.com/PacktPublishing/Learning-Robotics-using-Python-Second-Edition)。我们还有其他丰富的图书和视频代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从[`www.packtpub.com/sites/default/files/downloads/LearningRoboticsusingPythonSecondEdition_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/LearningRoboticsusingPythonSecondEdition_ColorImages.pdf)下载。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“第一个步骤是创建一个世界文件，并将其保存为`.world`文件扩展名。”

代码块应如下设置：

```py
<xacro:include filename=”$(find
 chefbot_description)/urdf/chefbot_gazebo.urdf.xacro”/>
 <xacro:include filename=”$(find
 chefbot_description)/urdf/chefbot_properties.urdf.xacro”/>

```

任何命令行输入或输出都应如下编写：

```py
$ roslaunch chefbot_gazebo chefbot_empty_world.launch
```

警告或重要提示看起来像这样。

小贴士和技巧看起来像这样。

# 联系我们

读者反馈始终欢迎。

**一般反馈**: 请通过`feedback@packtpub.com`发送邮件，并在邮件主题中提及书名。如果您对本书的任何方面有疑问，请通过`questions@packtpub.com`发送邮件给我们。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告，我们将非常感谢。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**: 如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需更多关于 Packt 的信息，请访问[packtpub.com](https://www.packtpub.com/)。
