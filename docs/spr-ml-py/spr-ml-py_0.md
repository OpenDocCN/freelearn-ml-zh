# 前言

**监督机器学习**广泛应用于金融、在线广告和分析等各个领域，因为它能够训练系统进行定价预测、广告活动调整、客户推荐等，赋予系统自我调整和自主决策的能力。这种能力的优势使得了解机器如何在幕后学习变得至关重要。

本书将引导你实现和深入了解许多流行的监督机器学习算法。你将从快速概览开始，了解监督学习与无监督学习的区别。接下来，我们将探索线性回归和逻辑回归等参数模型，决策树等非参数方法，以及各种聚类技术，以促进决策和预测。随着进展，你还将接触到推荐系统，这是在线公司广泛使用的工具，用于提升用户互动和促进潜在销售。最后，我们将简要探讨神经网络和迁移学习。

本书结束时，你将掌握一些实用的技巧，获得将监督学习算法快速且有效地应用于新问题所需的实际操作能力。

# 本书的目标读者

本书适合那些希望开始使用监督学习的有志机器学习开发者。读者应具备一定的 Python 编程知识和一些监督学习的基础知识。

# 本书内容概览

第一章，*迈向监督学习的第一步*，介绍了监督机器学习的基础知识，帮助你为独立解决问题做好准备。本章包含四个重要部分。首先，我们将设置 Anaconda 环境，确保能够运行示例。在接下来的几节中，我们将进一步讲解机器学习的理论基础，最后在实施算法的部分中，我们将再次设置 Anaconda 环境并开始实现算法。

第二章，*实现参数模型*，深入探讨了几种流行的监督学习算法，这些算法都属于参数建模家族。我们将从正式介绍参数模型开始，接着重点介绍两种特别流行的参数模型：线性回归和逻辑回归。我们将花些时间了解其内部原理，然后进入 Python，真正从头开始编写代码。

第三章，*使用非参数模型*，探讨了非参数模型系列。我们将从讨论偏差-方差权衡开始，并解释参数模型和非参数模型在根本上的差异。然后，我们将介绍决策树和聚类方法。最后，我们将讨论非参数模型的一些优缺点。

第四章，*监督学习的高级主题*，涉及两个主题：推荐系统和神经网络。我们将从协同过滤开始，然后讨论如何将基于内容的相似性集成到协同过滤系统中。最后，我们将进入神经网络和迁移学习的讨论。

# 为了最大化本书的学习效果

您需要以下软件来顺利进行各章内容：

+   Jupyter Notebook

+   Anaconda

+   Python

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户中下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packt.com/support](http://www.packt.com/support)，注册后将文件直接发送给您。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“SUPPORT”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名并按照屏幕上的指示操作。

下载文件后，请确保使用以下最新版本的工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，链接为[`github.com/PacktPublishing/Supervised-Machine-Learning-with-Python`](https://github.com/PacktPublishing/Supervised-%20Machine-Learning-with-Python)。如果代码有更新，将会在现有的 GitHub 仓库中进行更新。

我们还提供了来自我们丰富图书和视频目录的其他代码包，您可以访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**进行查看！

# 下载彩色图像

我们还提供了一份 PDF 文件，包含本书中使用的截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781838825669_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781838825669_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入以及 Twitter 用户名。示例：“将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。”

一段代码块的格式如下：

```py
from urllib.request import urlretrieve, ProxyHandler, build_opener, install_opener
import requests
import os
pfx = "https://archive.ics.uci.edu/ml/machine-learning databases/spambase/"
data_dir = "data"
```

任何命令行输入或输出都如下所示：

```py
jupyter notebook
```

**粗体**：指示一个新术语、重要词汇或屏幕上显示的词语。例如，菜单或对话框中的字词会像这样出现在文本中。以下是一个例子：“从管理面板中选择系统信息。”

警告或重要提示会以这种形式出现。

提示和技巧会以这种形式出现。

# 联系方式

我们始终欢迎读者的反馈。

**常规反馈**：如对本书的任何方面有疑问，请在消息主题中提及书名，并发送邮件至 `customercare@packtpub.com`。

**勘误**：尽管我们已尽一切努力确保内容的准确性，但错误不可避免。如果您在本书中发现错误，请向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们任何形式的作品的非法副本，我们将不胜感激您提供地址或网站名称。请联系我们，链接至 `copyright@packt.com`。

**如果您有意成为作者**：如果您在某个专题上有专业知识，并且有兴趣撰写或贡献书籍，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。阅读并使用本书后，为什么不在购买书籍的网站上留下您的评论呢？潜在的读者可以看到并使用您的客观意见来做出购买决策，Packt 可以了解您对我们产品的看法，而我们的作者可以看到您对他们书籍的反馈。谢谢！

欲了解更多 Packt 相关信息，请访问 [packt.com](http://www.packt.com/)。
