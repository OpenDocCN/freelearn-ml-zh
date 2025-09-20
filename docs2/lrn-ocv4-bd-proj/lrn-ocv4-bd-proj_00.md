# 前言

OpenCV 是用于开发计算机视觉应用中最受欢迎的库之一。它使我们能够在实时运行许多不同的计算机视觉算法。它已经存在很多年了，并已成为该领域的标准库。OpenCV 的主要优势之一是它高度优化，几乎在所有平台上都可用。

本书首先简要介绍了计算机视觉的各个领域以及相关的 OpenCV C++功能。每一章都包含真实世界的示例和代码示例，以展示用例。这有助于你轻松掌握主题，并了解它们如何在现实生活中应用。总之，这是一本关于如何使用 OpenCV 在 C++中构建各种应用的实用指南。

# 本书面向对象

本书面向那些对 OpenCV 新手，并希望使用 OpenCV 在 C++中开发计算机视觉应用的开发者。具备基本的 C++知识将有助于理解本书。本书对那些希望开始学习计算机视觉并理解其基本概念的人也很有用。他们应该了解基本的数学概念，如向量、矩阵和矩阵乘法，以便充分利用本书。在本书的学习过程中，你将学习如何从头开始使用 OpenCV 构建各种计算机视觉应用。

# 本书涵盖内容

第一章，*OpenCV 入门*，涵盖了在各个操作系统上的安装步骤，并介绍了人眼视觉系统以及计算机视觉中的各种主题。

第二章，*OpenCV 基础知识介绍*，讨论了如何在 OpenCV 中读取/写入图像和视频，并解释了如何使用 CMake 构建项目。

第三章，*学习图形用户界面和基本滤波*，涵盖了如何构建图形用户界面和鼠标事件检测器以构建交互式应用。

第四章，*深入直方图和滤波器*，探讨了直方图和滤波器，并展示了如何卡通化图像。

第五章，*自动化光学检测、对象分割和检测*，描述了各种图像预处理技术，如噪声去除、阈值和轮廓分析。

第六章，*学习对象分类*，涉及对象识别和机器学习，以及如何使用支持向量机构建对象分类系统。

第七章，*检测人脸部分和叠加面具*，讨论了人脸检测和 Haar 级联，并解释了如何使用这些方法来检测人脸的各个部分。

第八章，*视频监控、背景建模和形态学操作*，探讨了背景减法、视频监控和形态学图像处理，并描述了它们是如何相互关联的。

第九章，*学习对象跟踪*，介绍了如何使用不同的技术，如基于颜色和基于特征的技术，在实时视频中跟踪对象。

第十章，*为文本识别开发分割算法*，涵盖了光学字符识别、文本分割，并介绍了 Tesseract OCR 引擎的简介。

第十一章，*使用 Tesseract 进行文本识别*，更深入地探讨了 Tesseract OCR 引擎，解释了它如何用于文本检测、提取和识别。

第十二章，*使用 OpenCV 进行深度学习*，探讨了如何在 OpenCV 中应用深度学习，并介绍了两种常用的深度学习架构：用于目标检测的 YOLO v3 和用于人脸检测的单次检测器。

# 要充分利用本书

对 C++的基本了解将有助于理解本书。 示例使用以下技术构建：OpenCV 4.0； CMake 3.3.x 或更高版本； Tesseract； Leptonica（Tesseract 的依赖项）； Qt（可选）；以及 OpenGL（可选）。

详细安装说明提供在相关章节中。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载和勘误表。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载完成后，请确保使用最新版本的以下软件解压或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition`](https://github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition)。 如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包可供选择，请访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。查看它们！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789341225_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789341225_ColorImages.pdf).

# 代码实战

访问以下链接查看代码运行的视频：

[`bit.ly/2Sfrxgu`](http://bit.ly/2Sfrxgu)

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“此外，安装此包是可选的。即使不安装 `opencv_contrib`，OpenCV 也能正常工作。”

代码块应如下设置：

```py
// Load image to process 
  Mat img= imread(img_file, 0); 
  if(img.data==NULL){ 
    cout << "Error loading image "<< img_file << endl; 
    return 0; 
  } 
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
for(auto i=1; i<num_objects; i++){ 
    cout << "Object "<< i << " with pos: " << centroids.at<Point2d>(i) << " with area " << stats.at<int>(i, CC_STAT_AREA) << endl; 
```

任何命令行输入或输出都应如下编写：

```py
C:> setx -m OPENCV_DIR D:OpenCVBuildx64vc14
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要注意事项如下所示。

小技巧如下所示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并 通过 `customercare@packtpub.com` 发送给我们。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告，我们将不胜感激。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，我们将不胜感激，如果您能提供位置地址或网站名称，我们将不胜感激。请通过 `copyright@packt.com` 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/).

# 评论

请留下评论。一旦您阅读并使用过这本书，为什么不在这本书购买的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

关于 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/).
