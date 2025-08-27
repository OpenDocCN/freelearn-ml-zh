# 前言

人工智能和机器学习是复杂的话题，将此类功能添加到应用程序中在历史上需要大量的处理能力，更不用说巨大的学习量。微软认知服务的引入为开发者提供了轻松添加这些功能的可能性。它使我们能够创建更智能、更类似人类的程序。

本书旨在教你如何利用微软认知服务中的 API。你将了解每个 API 能提供什么，以及如何将其添加到你的应用程序中。你将看到不同的 API 调用对输入数据的要求以及你可以期待得到什么。本书中的大多数 API 都包含了理论和实践示例。

本书旨在帮助你入门。它侧重于展示如何使用微软认知服务，同时考虑到当前的最佳实践。本书的目的不是展示高级用例，而是为你提供一个起点，让你自己开始尝试使用这些 API。

# 本书面向的对象

本书面向有一定编程经验的.NET 开发者。假设你知道如何进行基本的编程任务以及如何在 Visual Studio 中导航。阅读本书不需要具备人工智能或机器学习的先验知识。

理解网页请求的工作原理是有益的，但并非必需。

# 本书涵盖的内容

第一章，*开始使用微软认知服务*，通过描述其提供的内容和提供一些基本示例来介绍微软认知服务。

第二章，*分析图像以识别面部*，涵盖了大多数图像 API，介绍了面部识别和识别、图像分析、光学字符识别等。

第三章，*分析视频*，介绍了视频索引器 API。

第四章，*让应用程序理解命令*，深入探讨了如何设置**语言理解智能服务**（**LUIS**），以便你的应用程序能够理解最终用户的意图。

第五章，*与你的应用程序对话*，深入探讨了不同的语音 API，包括文本到语音和语音到文本的转换、说话人识别和识别以及识别自定义说话风格和环境。

第六章，*理解文本*，介绍了一种分析文本的不同方法，利用强大的语言分析工具以及更多。

第七章，*为企业构建推荐系统*，涵盖了推荐 API。

第八章, *以自然方式查询结构化数据*，处理了学术论文和期刊的探索。通过本章，我们探讨了如何使用 Academic API 并自行设置类似的服务。

第九章, *添加专业搜索*，深入探讨了 Bing 的不同搜索 API。这包括新闻、网页、图像和视频搜索以及自动建议。

第十章, *连接各个部分*，将几个 API 连接起来，并通过查看一些自然步骤来结束本书。

附录 A, *LUIS 实体*，列出了所有预构建的 LUIS 实体的完整列表。

附录 B, *许可信息*，展示了示例代码中使用的所有第三方库的相关许可信息。

# 要充分利用这本书

+   要跟随本书中的示例，您将需要 Visual Studio 2015 社区版或更高版本。您还需要一个有效的互联网连接和 Microsoft Azure 订阅；试用订阅也可以。

+   要完整体验示例，您应该能够访问网络摄像头，并且将扬声器和麦克风连接到计算机；然而，这两者都不是强制性的。

## 下载示例代码文件

您可以从[`www.packtpub.com`](http://www.packtpub.com)的账户下载这本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[`www.packtpub.com`](http://www.packtpub.com)登录或注册。

1.  选择**支持**选项卡。

1.  点击**代码下载与勘误**。

1.  在**搜索**框中输入书籍名称，并遵循屏幕上的说明。

文件下载完成后，请确保您使用最新版本的软件解压或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

书籍的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Learning-Microsoft-Cognitive-Services-Third-Edition`](https://github.com/PacktPublishing/Learning-Microsoft-Cognitive-Services-Third-Edition)。我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

## 下载彩色图像

我们还提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789800616_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789800616_ColorImages.pdf)。

## 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。例如；“当我们把一些内容放入`DelegateCommand.cs`文件时，可以实现这一点。”

代码块设置如下：

```py
private string _filePath;
private IFaceServiceClient _faceServiceClient;
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
private string _filePath;
private IFaceServiceClient _faceServiceClient;
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词，例如在菜单或对话框中，这些单词在文本中也会以这种方式出现。例如：“打开 Visual Studio 并选择**文件 | 新建 | 项目**。”

### 注意

警告或重要注意事项以如下框的形式出现。

### 小贴士

小贴士和技巧看起来是这样的。

# 联系我们

读者的反馈总是受欢迎的。

**一般反馈**：请发送电子邮件至 `<feedback@packtpub.com>`，并在邮件主题中提及书籍的标题。如果您对本书的任何方面有疑问，请通过电子邮件联系我们 `<questions@packtpub.com>`。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们非常感谢您能向我们报告。请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，我们非常感谢您能提供位置地址或网站名称。请通过电子邮件联系我们 `<copyright@packtpub.com>` 并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [`authors.packtpub.com`](http://authors.packtpub.com)。

## 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需更多关于 Packt 的信息，请访问 [packtpub.com](http://packtpub.com)。
