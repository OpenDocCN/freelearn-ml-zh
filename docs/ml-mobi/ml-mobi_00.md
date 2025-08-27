# 前言

这本书将通过简单的实际示例帮助你使用移动设备进行机器学习。你将从机器学习的基础知识开始，等你完成这本书的时候，你将很好地掌握移动机器学习的概念，了解有哪些工具/SDKs 可用于实现移动机器学习，并且能够将各种机器学习算法应用于可以在 iOS 和 Android 上运行的应用程序中。

你将了解什么是机器学习，并理解推动移动机器学习的原因以及它的独特之处。你将接触到所有移动机器学习工具和 SDK：TensorFlow Lite、Core ML、ML Kit 和 Android 和 iOS 上的 Fritz。这本书将探讨每个工具包的高级架构和组件。到本书结束时，你将广泛了解机器学习模型，并能够进行设备上的机器学习。你将深入了解诸如回归、分类、线性**支持向量机**（**SVM**）和随机森林等机器学习算法。你将学习如何进行自然语言处理和实现垃圾邮件检测。你将学习如何将使用 Core ML 和 TensorFlow 创建的现有模型转换为 Fritz 模型。你还将接触到神经网络。你还将提前了解机器学习的未来，本书还包含了一个常见问题解答部分，以回答你关于移动机器学习的所有疑问。它将帮助你构建一个有趣的应用程序，该应用程序可以提供相机捕获的食品项目的卡路里值，该应用程序在 iOS 和 Android 上都可以运行。

# 这本书面向的对象

如果你是希望利用机器学习并在移动和智能设备上使用它的移动开发者或机器学习用户，那么《移动机器学习》这本书适合你。你最好具备机器学习的基本知识以及移动应用程序开发的入门级经验。

# 这本书涵盖了以下内容

第一章，*移动机器学习简介*，解释了什么是机器学习以及为什么我们应该在移动设备上使用它。它介绍了不同的机器学习方法及其优缺点。

第二章，*监督学习和无监督学习算法*，涵盖了机器学习算法的监督和无监督方法。我们还将学习不同的算法，如朴素贝叶斯、决策树、SVM、聚类、关联映射等。

第三章，*iOS 上的随机森林*，深入探讨了随机森林和决策树，并解释了如何将它们应用于解决机器学习问题。我们还将创建一个使用决策树来诊断乳腺癌的应用程序。

第四章，*TensorFlow Mobile 在 Android 中*，介绍了 TensorFlow 移动应用。我们还将了解移动机器学习应用程序的架构，并使用 TensorFlow 在 Android 中编写应用程序。

第五章，*在 iOS 中使用 Core ML 进行回归*，探讨了回归和 Core ML，并展示了如何将其应用于解决机器学习问题。我们将创建一个使用 scikit-learn 预测房价的应用程序。

第六章，*ML Kit SDK*，探讨了 ML Kit 及其优势。我们将使用 ML Kit 和设备以及云 API 创建一些图像标签应用程序。

第七章，*iOS 中的垃圾邮件检测 - Core ML*，介绍了自然语言处理和 SVM 算法。我们将解决大量短信的问题，即消息是否为垃圾邮件。

第八章，*Fritz*，介绍了 Fritz 移动机器学习平台。我们将使用 Fritz 和 iOS 中的 Core ML 创建应用程序。我们还将了解 Fritz 如何与我们在本书早期创建的示例数据集一起使用。

第九章，*移动设备上的神经网络*，涵盖了神经网络、Keras 的概念及其在移动机器学习领域的应用。我们将创建一个识别手写数字的应用程序，以及 TensorFlow 图像识别模型。

第十章，*使用 Google Cloud Vision 的移动应用程序*，介绍了在 Android 应用程序中使用的 Google Cloud Vision 标签检测技术，以确定相机拍摄的图片中有什么内容。

第十一章，*移动应用程序中机器学习的未来*，涵盖了移动应用程序的关键特性和它们为利益相关者提供的机遇。

附录，*问答*，包含了您可能关心的问题，并试图为这些问题提供答案。

# 为了充分利用本书

读者需要具备机器学习、Android Studio 和 Xcode 的先验知识。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择 SUPPORT 选项卡。

1.  点击代码下载与勘误表。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

下载文件后，请确保您使用最新版本的软件解压缩或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

书籍的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Machine-Learning-for-Mobile`](https://github.com/PacktPublishing/Machine-Learning-for-Mobile)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781788629355_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781788629355_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“现在您可以将生成的`SpamMessageClassifier.mlmodel`文件用于您的 Xcode。”

代码块如下设置：

```py
# importing required packages
import numpy as np
import pandas as pd
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
# Reading in and parsing data
raw_data = open('SMSSpamCollection.txt', 'r')
sms_data = []
for line in raw_data:
    split_line = line.split("\t")
    sms_data.append(split_line)
```

任何命令行输入或输出都如下所示：

```py
pip install scikit-learn pip install numpy pip install coremltools pip install pandas
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下所示。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要注意事项看起来是这样的。

小贴士和技巧看起来是这样的。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`发送邮件给我们。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评价。一旦您阅读并使用过这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，而我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问[packt.com](http://www.packt.com/).
