# 前言

本书将教会你如何在云中以非常低廉的方式执行高级机器学习。您将更多地了解 Azure 机器学习流程作为企业级的方法。本书让您探索 Azure 机器学习工作室中的预建模板，并使用可部署为 Web 服务的预配置算法构建模型。它将帮助您发现利用云进行机器学习和 AI 的不同好处，在 AI 开发场景中部署虚拟机，以及如何在 Azure 中应用 R、Python、SQL Server 和 Spark。

在本书结束时，您将能够将机器学习和 AI 概念应用于您的模型以解决现实世界的问题。

# 本书面向的对象

如果你是一位熟悉 Azure 机器学习和认知服务的数据科学家或开发者，并希望创建智能模型并在云中理解数据，这本书适合你。如果你希望将强大的机器学习服务引入你的云应用中，这本书也会很有用。一些数据操作和处理的经验，以及使用 SQL、Python 和 R 等语言，将有助于你理解本书中涵盖的概念

# 要充分利用本书

对于本书，你需要具备 Azure 的先验知识并拥有 Azure 订阅。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

你可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择 SUPPORT 标签页。

1.  点击代码下载与勘误。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

文件下载完成后，请确保使用最新版本解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Machine-Learning-with-Azure`](https://github.com/PacktPublishing/Hands-On-Machine-Learning-with-Azure)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还提供其他丰富的书籍和视频的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789131956_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789131956_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“或者，您可以在 Azure 门户中搜索`计算机视觉`”。

代码块设置如下：

```py
vision_base_url = "https://westus.api.cognitive.microsoft.com/vision/v1.0/"

vision_analyze_url = vision_base_url + "analyze"
```

**粗体**: 表示新术语、重要词汇或屏幕上看到的词汇。例如，菜单或对话框中的文字会以这种方式显示。以下是一个示例：“点击创建资源，然后点击 AI + 机器学习，然后点击计算机视觉”。

警告或重要注意事项会以这种方式显示。

小贴士和技巧会以这种方式显示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**: 如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并将邮件发送至`customercare@packtpub.com`。

**勘误表**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告，我们将不胜感激。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误表提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上发现我们作品的任何非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请联系...

# 评论

请留下评论。一旦您阅读并使用过这本书，为何不在购买它的网站上留下评论呢？潜在读者可以看到并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
