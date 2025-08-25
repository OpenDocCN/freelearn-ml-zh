# 前言

*《Java 机器学习，第二版》* 将为您提供从复杂数据中快速获得洞察力的技术和工具。您将从学习如何将机器学习方法应用于各种常见任务开始，包括分类、预测、预测、篮子分析和聚类。

这是一个实用的教程，使用实际示例逐步介绍一些机器学习的实际应用。在不过分回避技术细节的同时，您将使用清晰且实用的示例探索如何使用 Java 库进行机器学习。您将探索如何准备分析数据、选择机器学习方法以及衡量过程的成功。

# 本书面向的对象

如果您想学习如何使用 Java 的机器学习库从数据中获得洞察力，这本书就是为您准备的。它将帮助您快速上手，并提供您成功创建、定制和轻松部署机器学习应用所需的技能。为了充分利用本书，您应该熟悉 Java 编程和一些基本的数据挖掘概念，但不需要有机器学习的前期经验。

# 为了充分利用本书

本书假设用户具备 Java 语言的实际知识以及机器学习的基本概念。本书大量使用了 JAR 格式的外部库。假设用户知道如何在终端或命令提示符中使用 JAR 文件，尽管本书也解释了如何进行这一操作。用户可以轻松地在任何通用的 Windows 或 Linux 系统上使用本书。

# 下载示例代码文件

您可以从 [www.packt.com](http://www.packt.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择支持选项卡。

1.  点击代码下载和勘误表。

1.  在搜索框中输入本书的名称，并按照屏幕上的说明操作。

下载文件后，请确保使用最新版本解压缩或提取文件夹：

+   Windows 的 WinRAR/7-Zip

+   Mac 的 Zipeg/iZip/UnRarX

+   Linux 的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为 [`github.com/PacktPublishing/Machine-Learning-in-Java-Second-Edition`](https://github.com/PacktPublishing/Machine-Learning-in-Java-Second-Edition)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包，可在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 获取。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781788474399_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781788474399_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“解压存档，并在提取的存档中找到`weka.jar`。”

代码块设置为以下格式：

```py
data.defineSingleOutputOthersInput(outputColumn); 

EncogModel model = new EncogModel(data); 
model.selectMethod(data, MLMethodFactory.TYPE_FEEDFORWARD);
model.setReport(new ConsoleStatusReportable()); 
data.normalize(); 
```

任何命令行输入或输出都按照以下方式编写：

```py
$ java -cp moa.jar -javaagent:sizeofag.jar moa.gui.GUI
```

**粗体**: 表示新术语、重要单词或你在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示为粗体。以下是一个示例：“我们可以通过点击文件 | 另存为，并在保存对话框中选择 CSV，将其转换为**逗号分隔值**（**CSV**）格式。”

警告或重要提示看起来像这样。

小贴士和技巧看起来像这样。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**: 如果你对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`给我们发送邮件。

**勘误表**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们将不胜感激，如果你能向我们报告这个错误。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择你的书，点击勘误表提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并附上材料的链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且你对撰写或为本书做出贡献感兴趣，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦你阅读并使用了这本书，为什么不在你购买它的网站上留下评论呢？潜在的读者可以看到并使用你的无偏见意见来做出购买决定，我们 Packt 可以了解你对我们的产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

关于 Packt 的更多信息，请访问[packt.com](http://www.packt.com/)。
