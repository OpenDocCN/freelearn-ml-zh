# 前言

这本书将帮助你迅速开始使用 Android 平台上的 OpenCV。它从概念上解释了各种计算机视觉算法，以及它们在 Android 平台上的实现。如果你期待在新的或现有的 Android 应用中实现计算机视觉模块，这本书是无价的资源。

# 本书涵盖内容

第一章, *将效果应用于图像*，包括在各种计算机视觉应用中使用的一些基本预处理算法。本章还解释了如何将 OpenCV 集成到现有的项目中。

第二章, *在图像中检测基本特征*，涵盖了在图像中检测主要特征，如边缘、角点、线和圆。

第三章, *检测对象*，深入到特征检测，使用更先进的算法来检测和描述特征，以便将它们独特地匹配到其他对象中的特征。

第四章, *深入研究对象检测 – 使用级联分类器*，解释了图像和视频中一般对象的检测，如人脸/眼睛。

第五章, *在视频中跟踪对象*，涵盖了光流作为运动检测器的概念，并实现了 Lucas-Kanade-Tomasi 跟踪器来跟踪视频中的对象。

第六章, *处理图像对齐和拼接*，涵盖了图像对齐和图像拼接的基本概念，以创建全景场景图像。

第七章, *使用 OpenCV 机器学习让您的应用栩栩如生*，解释了机器学习如何在计算机视觉应用中使用。在这一章中，我们查看了一些常见的机器学习算法及其在 Android 中的实现。

第八章, *故障排除和最佳实践*，涵盖了开发者在构建应用程序时遇到的一些常见错误和问题。它还展开了一些可以提高应用程序效率的良好实践。

第九章, *开发文档扫描应用*，使用在第几章中解释的各种算法来构建一个完整的文档扫描系统，无论你从哪个角度点击图像。

# 你需要为这本书准备什么

为了这本书，你需要一个至少有 1 GB RAM 的系统。Windows、OS X 和 Linux 是目前支持 Android 开发的操作系统。

# 这本书面向谁

如果你是一名 Java 和 Android 开发者，并希望通过学习 OpenCV Android 应用程序编程的最新特性来提升你的技能，那么这本书就是为你准备的。

# 术语约定

在本书中，你会发现许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称如下所示：“创建一个名为`Application.mk`的文件，并将以下代码行复制到其中。”

代码块设置如下：

```py
<uses-permission android:name="android.permission.CAMERA"/>
    <uses-feature android:name="android.hardware.camera" android:required="false"/>
    <uses-feature android:name="android.hardware.camera.autofocus" android:required="false"/>
    <uses-feature android:name="android.hardware.camera.front" android:required="false"/>
    <uses-feature android:name="android.hardware.camera.front.autofocus" android:required="false"/>
```

**新术语**和**重要词汇**以粗体显示。

### 注意

警告或重要注意事项以如下方式出现在框中。

### 小贴士

小贴士和技巧看起来是这样的。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们你对这本书的看法——你喜欢什么或不喜欢什么。读者的反馈对我们来说非常重要，因为它帮助我们开发出你真正能从中受益的书籍。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及书籍的标题。

如果你在一个领域有专业知识，并且你对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在你已经是 Packt 书籍的骄傲拥有者，我们有一些东西可以帮助你从购买中获得最大收益。

## 下载示例代码

你可以从你购买的所有 Packt 出版物的账户中下载示例代码文件[`www.packtpub.com`](http://www.packtpub.com)。如果你在其他地方购买了这本书，你可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给你。

## 下载本书的彩色图像

我们还为你提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。彩色图像将帮助你更好地理解输出的变化。你可以从以下链接下载此文件：[`www.packtpub.com/sites/default/files/downloads/8204OS_ImageBundle.pdf`](https://www.packtpub.com/sites/default/files/downloads/8204OS_ImageBundle.pdf)。

## 错误清单

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问 [`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**勘误**部分下。

## 盗版

互联网上对版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视保护我们的版权和许可证。如果您在互联网上发现任何形式的我们作品的非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<copyright@packtpub.com>` 联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面提供的帮助。

## 问题

如果您对本书的任何方面有问题，您可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决问题。
