# 前言

*精通 OpenCV3，第二版*包含七个章节，每个章节都是一个从开始到结束的完整项目教程，基于 OpenCV 的 C++接口，包括完整的源代码。每个章节的作者都是因其在该主题上对 OpenCV 社区的良好在线贡献而被选中的，并且本书由主要的 OpenCV 开发者之一进行了审阅。本书不是解释 OpenCV 函数的基础，而是展示了如何将 OpenCV 应用于解决整个问题，包括几个 3D 相机项目（增强现实和 3D 结构从运动）以及几个面部分析项目（如皮肤检测、简单面部和眼睛检测、复杂面部特征跟踪、3D 头部姿态估计和面部识别），因此它是现有 OpenCV 书籍的优秀伴侣。

# 本书涵盖内容

第一章，*Raspberry Pi 的卡通化器和皮肤变换器*，包含了一个桌面应用程序和 Raspberry Pi 的完整教程和源代码，这些应用程序可以自动从真实相机图像生成卡通或绘画，包括多种可能的卡通类型，以及皮肤颜色变换功能。

第二章，*使用 OpenCV 探索运动结构*，通过在 OpenCV 中实现 SfM（运动结构）概念来介绍 SfM。读者将学习如何从多个 2D 图像中重建 3D 几何形状并估计相机位置。

第三章，*使用 SVM 和神经网络进行车牌识别*，包含了一个完整的教程和源代码，用于构建一个使用模式识别算法以及支持向量机（SVM）和人工神经网络（ANN）的自动车牌识别应用程序。读者将学习如何训练和预测模式识别算法以判断一个图像是否为车牌，并且它还将帮助将一组特征分类为字符。

第四章，*非刚性面部跟踪*，包含了一个完整的教程和源代码，用于构建一个动态面部跟踪系统，该系统能够建模和跟踪一个人面部复杂的多部分。

第五章，*使用 AAM 和 POSIT 进行 3D 头部姿态估计*，包含了理解什么是主动外观模型（AAMs）以及如何使用 OpenCV 通过一组具有不同面部表情的面部帧来创建它们的全部背景知识。此外，本章还解释了如何通过 AAMs 提供的拟合能力来匹配给定的帧。然后，通过应用 POSIT 算法，可以找到 3D 头部姿态。

第六章，*使用特征脸或 Fisher 脸进行人脸识别*，包含一个完整的教程和实时人脸识别应用程序的源代码，该应用程序包括基本的面部和眼部检测，以处理图像中面部旋转和光照条件的变化。

第七章，*增强现实的自然特征跟踪*，包括如何为 iPad 和 iPhone 设备构建基于标记的增强现实（AR）应用程序的完整教程，每个步骤都有解释和源代码。它还包含如何开发无标记增强现实桌面应用程序的完整教程，解释了无标记 AR 是什么，以及源代码。

您可以从以下链接下载本章内容：[h t t p s ://w w w . p a c k t p u b . c o m /s i t e s /d e f a u l t /f i l e s /d o w n l o a d s /N a t u r a l F e a t u r e T r a c k i n g f o r A u g m e n t e d R e a l i t y . p d f](https://www.packtpub.com/sites/default/files/downloads/NaturalFeatureTrackingforAugmentedReality.pdf).

# 您需要这本书的内容

您不需要在计算机视觉方面有特殊知识来阅读这本书，但您应该在阅读此书之前拥有良好的 C/C++编程技能和基本的 OpenCV 经验。没有 OpenCV 经验的读者可能希望阅读《Learning OpenCV》以了解 OpenCV 功能介绍，或阅读《OpenCV 2 Cookbook》以了解如何使用推荐的 C/C++模式使用 OpenCV 的示例，因为这本书将向您展示如何解决实际问题，假设您已经熟悉 OpenCV 和 C/C++开发的基础知识。

除了 C/C++和 OpenCV 经验外，您还需要一台计算机，以及您选择的 IDE（例如 Visual Studio、XCode、Eclipse 或 QtCreator，运行在 Windows、Mac 或 Linux 上）。某些章节有进一步的要求，特别是：

+   要为 Raspberry Pi 开发 OpenCV 程序，您需要 Raspberry Pi 设备、其工具以及基本的 Raspberry Pi 开发经验。

+   要开发 iOS 应用程序，您需要 iPhone、iPad 或 iPod Touch 设备，iOS 开发工具（包括 Apple 计算机、XCode IDE 和 Apple 开发者证书），以及基本的 iOS 和 Objective-C 开发经验。

+   几个桌面项目需要连接到您的计算机的摄像头。任何常见的 USB 摄像头都足够使用，但至少 1 兆像素的摄像头可能更受欢迎。

+   一些项目（包括 OpenCV 本身）使用 CMake 在操作系统和编译器之间构建。需要了解构建系统的基础知识，并建议了解跨平台构建。

预期您对线性代数有所了解，例如基本的向量矩阵运算和特征分解。

# 这本书面向的对象

*《精通 OpenCV 3，第 2 版》* 是适合具有基本 OpenCV 知识的开发者使用，以创建实用的计算机视觉项目，同时也适合希望将更多计算机视觉主题添加到其技能集的资深 OpenCV 专家。本书面向高级计算机科学大学生、毕业生、研究人员和希望使用 OpenCV C++接口解决实际问题的计算机视觉专家，通过实用的分步教程。

# 约定

在本书中，您将找到多种文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称将如下所示：“您应该将本章的大部分代码放入`cartoonifyImage()`函数中”

代码块将如下设置：

```py
    int cameraNumber = 0;
    if (argc> 1)
      cameraNumber = atoi(argv[1]);
    // Get access to the camera.
    cv::VideoCapture capture

```

当我们希望将您的注意力引到代码块的一个特定部分时，相关的行或项目将以粗体显示：

```py
    // Get access to the camera.
    cv::VideoCapture capture;
 camera.open(cameraNumber);
    if (!camera.isOpened()) {
      std::cerr<< "ERROR: Could not access the camera or video!" <<

```

任何命令行输入或输出将如下所示：

```py
cmake -G "Visual Studio 10" 

```

**新术语**和**重要单词**将以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中将如下所示：“为了下载新模块，我们将转到文件 | 设置 | 项目名称 | 项目解释器。”

警告或重要提示将以这样的框显示。

小贴士和技巧将如下所示。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢或不喜欢的地方。读者反馈对我们来说很重要，因为它帮助我们开发出您真正能从中受益的标题。

要发送一般反馈，请简单地发送电子邮件至`feedback@packtpub.com`，并在邮件主题中提及书籍的标题。

如果您在某个主题上具有专业知识，并且您对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

# 下载示例代码

您可以从您的账户下载此书的示例代码文件[`www.packtpub.com`](http://www.packtpub.com)。如果您在其他地方购买了此书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的“支持”标签上。

1.  点击“代码下载与勘误表”。

1.  在搜索框中输入书籍名称。

1.  选择您想要下载代码文件的书籍。

1.  从下拉菜单中选择您购买此书的来源。

1.  点击“代码下载”。

文件下载后，请确保您使用最新版本解压缩或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

该书的代码包也托管在 GitHub 上，地址为[h t t p s ://g i t h u b . c o m /P a c k t P u b l i s h i n g /M a s t e r i n g - O p e n C V 3- S e c o n d - E d i t i o n](https://github.com/PacktPublishing/Mastering-OpenCV3-Second-Edition)。我们还有其他来自我们丰富图书和视频目录的代码包可供在[h t t p s ://g i t h u b . c o m /P a c k t P u b l i s h i n g /](https://github.com/PacktPublishing/)获取。请查看它们！

# 下载本书的颜色图像

我们还为您提供了一个包含本书中使用的截图/图表的颜色图像的 PDF 文件。这些颜色图像将帮助您更好地理解输出的变化。您可以从[h t t p s ://w w w . p a c k t p u b . c o m /s i t e s /d e f a u l t /f i l e s /d o w n l o a d s /M a s t e r i n g O p e n C V 3S e c o n d E d i t i o n _ C o l o r I m a g e s . p d f](https://www.packtpub.com/sites/default/files/downloads/MasteringOpenCV3SecondEdition_ColorImages.pdf)下载此文件。

# 勘误

尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[h t t p ://w w w . p a c k t p u b . c o m /s u b m i t - e r r a t a](http://www.packtpub.com/submit-errata)来报告它们，选择您的书籍，点击勘误提交表单链接，并输入您的勘误详情。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误，请访问[h t t p s ://w w w . p a c k t p u b . c o m /b o o k s /c o n t e n t /s u p p o r t](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在勘误部分下。

# 盗版

互联网上版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上遇到任何形式的非法复制我们的作品，请立即向我们提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过`copyright@packtpub.com`与我们联系，并提供疑似盗版材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面的帮助。

# 问题

如果您对本书的任何方面有问题，您可以联系我们的`questions@packtpub.com`，我们将尽力解决问题。
