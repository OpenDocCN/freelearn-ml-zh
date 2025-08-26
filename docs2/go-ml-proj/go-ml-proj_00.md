# 前言

Go 是机器学习的完美语言。其简单的语法有助于清晰地描述复杂算法，但不会让开发者无法理解如何运行高效优化的代码。这本书将教你如何在 Go 中实现机器学习，以创建易于部署和易于理解和调试的程序，同时还可以测量其性能。

本书首先指导你使用 Go 库和功能设置机器学习环境。然后，你将深入分析一个真实的房屋定价数据集的回归分析，并在 Go 中构建一个分类模型来将电子邮件分类为垃圾邮件或正常邮件。使用 Gonum、Gorgonia 和 STL，你将探索时间序列分析，以及分解和如何通过聚类推文清理你的个人 Twitter 时间线。此外，你还将学习如何使用神经网络和卷积神经网络识别手写，这些都是深度学习技术。一旦你掌握了所有技术，你将借助面部检测项目学习如何选择最适合你项目的机器学习算法。

在这本书的结尾，你将培养出坚实的机器学习思维模式，对强大的 Go 库有深入的了解，并对机器学习算法在实际项目中的实际应用有清晰的理解。

# 本书面向对象

如果你是一名机器学习工程师、数据科学专业人士或 Go 程序员，希望在自己的实际项目中实现机器学习并更轻松地创建智能应用程序，这本书适合你。

# 为了充分利用这本书

在 Golang 中有一些编码经验以及基本机器学习概念的知识将有助于你理解本书中涵盖的概念。

# 下载示例代码文件

你可以从[www.packt.com](http://www.packt.com)的账户下载这本书的示例代码文件。如果你在其他地方购买了这本书，你可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给你。

你可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

文件下载完成后，请确保使用最新版本的软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

书籍的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Go-Machine-Learning-Projects`](https://github.com/PacktPublishing/Go-Machine-Learning-Projects)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/) 上找到。去看看吧！

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“我们草拟了一个什么也不做的虚拟`Classifier`类型。”

代码块应如下设置：

```py
Word: she - true
Word: shan't - false
Word: be - false
Word: learning - true
Word: excessively. - true
```

任何命令行输入或输出都应如下所示：

```py
go get -u github.com/go-nlp/tfidf
```

**粗体**: 表示新术语、重要词汇或屏幕上看到的词汇。例如，菜单或对话框中的文字会以这种方式显示。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要注意事项看起来像这样。

小贴士和技巧看起来像这样。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过 `customercare@packtpub.com` 邮箱联系我们。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上以任何形式发现我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过 `copyright@packt.com` 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为什么不在这家您购买书籍的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/)。
