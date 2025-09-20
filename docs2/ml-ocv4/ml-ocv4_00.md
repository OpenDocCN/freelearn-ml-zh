# 前言

随着世界的改变和人类建造更智能、更好的机器，对机器学习和计算机视觉专家的需求增加。正如其名所示，机器学习是机器学习到根据一组输入参数进行预测的过程。另一方面，计算机视觉给机器赋予视觉；也就是说，它使机器意识到视觉信息。当你结合这些技术时，你得到一个可以使用视觉数据进行预测的机器，这使机器在拥有人类能力方面又迈出一步。当你加入深度学习，机器甚至可以在预测方面超越人类的能力。这听起来可能有些牵强，但随着基于 AI 的系统接管决策系统，这实际上已经成为现实。你有 AI 摄像头、AI 显示器、AI 声音系统、AI 驱动的处理器等等。我们无法保证你阅读这本书后能够构建一个 AI 摄像头，但我们确实打算提供你完成这一目标的必要工具。我们将要介绍的最强大的工具是 OpenCV 库，这是世界上最大的计算机视觉库。尽管它在机器学习中的应用并不常见，但我们提供了一些示例和概念，说明它可以如何用于机器学习。我们在本书中采用了实践方法，并建议你尝试本书中出现的每一行代码，以构建一个展示你知识的应用程序。世界正在改变，这本书是我们帮助年轻思想将其变得更好的方式。

# 这本书面向谁

我们试图从零开始解释所有概念，使这本书既适合初学者也适合高级读者。我们建议读者具备一些 Python 编程的基本知识，但这不是强制性的。无论何时你遇到一些你无法理解的 Python 语法，确保你在互联网上查找它。*对于那些寻求帮助的人来说，总是有提供的帮助。*

# 为了最大限度地利用这本书

如果你是一个 Python 的初学者，我们建议你阅读任何好的 Python 编程书籍或在线教程或视频。你也可以查看 DataCamp ([`www.datacamp.com`](http://www.datacamp.com))，使用交互式课程学习 Python。

我们还建议你学习一些关于 Python 中的 Matplotlib 库的基本概念。你可以尝试这个教程：[`www.datacamp.com/community/tutorials/matplotlib-tutorial-python`](https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python)。

在开始这本书之前，你不需要在你的系统上安装任何东西。我们将在第一章中涵盖所有安装步骤。

# 下载示例代码文件

您可以从 [www.packt.com](http://www.packt.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [www.packtpub.com/support](https://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  登录或注册 [www.packt.com](http://www.packt.com)。

1.  选择“支持”标签。

1.  点击“代码下载”。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为 [`github.com/PacktPublishing/Machine-Learning-for-OpenCV-Second-Edition ...`](https://github.com/PacktPublishing/Machine-Learning-for-OpenCV-Second-Edition)。

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789536300_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789536300_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“我们可以将`max_samples<1.0`和`max_features<1.0`都设置为实施随机补丁方法。”

代码块设置如下：

```py
In [1]: from sklearn.ensemble import BaggingClassifier... from sklearn.neighbors import KNeighborsClassifier... bag_knn = BaggingClassifier(KNeighborsClassifier(),... n_estimators=10)
```

任何命令行输入或输出都应如下所示：

```py
$ conda install package_name
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。

警告 ...

# 联系我们

我们欢迎读者的反馈。

**总体反馈**：如果您对本书的任何方面有任何疑问，请在邮件主题中提及书名，并将邮件发送至 `customercare@packtpub.com`。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您能向我们报告。请访问 [www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过 `copyright@packt.com` 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评价。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，并且我们的作者可以查看他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
