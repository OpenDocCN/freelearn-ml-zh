# 前言

在这本书中，您将学习如何利用 Python、OpenCV 和 TensorFlow 的强大功能来解决计算机视觉中的问题。Python 是快速原型设计和开发图像处理和计算机视觉生产级代码的理想编程语言，它具有稳健的语法和丰富的强大库。

本书将是您设计和开发针对现实世界问题的生产级计算机视觉项目的实用指南。您将学习如何为主要的操作系统设置 Anaconda Python，并使用计算机视觉的尖端第三方库，您还将学习分类图像和视频中的检测和识别人类的最先进技术。通过本书的结尾，您将获得使用 Python 及其相关库构建自己的计算机视觉项目所需的技能。

# 本书面向对象

希望使用机器学习和 OpenCV 的强大功能构建令人兴奋的计算机视觉项目的 Python 程序员和机器学习开发者会发现这本书很有用。本书的唯一先决条件是您应该具备扎实的 Python 编程知识。

# 要充分利用这本书

在 Python 及其包（如 TensorFlow、OpenCV 和 dlib）中具备一些编程经验，将帮助您充分利用这本书。

需要一个支持 CUDA 的强大 GPU 来重新训练模型。

# 下载示例代码文件

您可以从 [www.packt.com](http://www.packt.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择支持选项卡。

1.  点击代码下载和勘误表。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

下载文件后，请确保使用最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书代码包也托管在 GitHub 上，地址为 **[`github.com/PacktPublishing/Computer-Vision-Projects-with-OpenCV-and-Python-3`](https://github.com/PacktPublishing/Computer-Vision-Projects-with-OpenCV-and-Python-3)**。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包可供选择，请访问 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。查看它们！

# 下载彩色图像

我们还提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789954555_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789954555_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“`word_counts.txt`文件包含一个词汇表，其中包含我们从训练模型中得到的计数，这是我们的图像标题生成器所需要的。”

代码块按照以下方式设置：

```py
testfile = 'test_images/dog.jpeg'

figure()
imshow(imread(testfile))
```

任何命令行输入或输出都按照以下方式编写：

```py
conda install -c menpo dlib
```

**粗体**: 表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“点击 下载 按钮。”

警告或重要注意事项看起来像这样。

小贴士和技巧看起来像这样。

# 联系我们

我们读者的反馈总是受欢迎的。

**一般反馈**: 如果你对此书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`发送邮件给我们。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们将不胜感激，如果你能向我们报告这个错误。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择你的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过发送链接至`copyright@packt.com`与我们联系。

**如果你有兴趣成为作者**: 如果你有一个你擅长的主题，并且你对撰写或为书籍做出贡献感兴趣，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦你阅读并使用了这本书，为什么不在你购买它的网站上留下评论呢？潜在读者可以看到并使用你的客观意见来做出购买决定，我们 Packt 可以了解你对我们的产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

关于 Packt 的更多信息，请访问[packt.com](http://www.packt.com/)。
