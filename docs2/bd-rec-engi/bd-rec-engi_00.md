# 前言

*构建推荐引擎* 是一本全面指南，用于实现推荐引擎，如协同过滤、基于内容的推荐引擎、使用 R、Python、Spark、Mahout、Neo4j 技术构建的上下文感知推荐引擎。本书涵盖了在各个行业中广泛使用的各种推荐引擎及其实现。本书还涵盖了一章关于在构建推荐时常用的数据挖掘技术，并在书末简要讨论了推荐引擎的未来。

# 本书涵盖内容

第一章, *推荐引擎简介*，将为数据科学家提供复习，并为推荐引擎初学者提供入门介绍。本章介绍了人们日常生活中常用的流行推荐引擎，并涵盖了这些推荐引擎的常用方法和优缺点。

第二章, *构建你的第一个推荐引擎*，是一章简短的章节，介绍了如何构建电影推荐引擎，以便我们在进入推荐引擎的世界之前有一个良好的开端。

第三章, *推荐引擎解析*，介绍了流行的不同推荐引擎技术，例如基于用户的协同过滤推荐引擎、基于物品的协同过滤、基于内容的推荐引擎、上下文感知推荐者、混合推荐者、基于机器学习模型和数学模型的模型推荐系统。

第四章, *推荐引擎中使用的数据挖掘技术*，介绍了在构建推荐引擎中使用的各种机器学习技术，如相似度度量、分类、回归和降维技术。本章还涵盖了评估指标，以测试推荐引擎的预测能力。

第五章, *构建协同过滤推荐引擎*，介绍了如何在 R 和 Python 中构建基于用户的协同过滤和基于物品的协同过滤。我们还将了解在构建推荐引擎中广泛使用的 R 和 Python 的不同库。

第六章, *构建个性化推荐引擎*，介绍了如何使用 R 和 Python 构建个性化推荐引擎，以及用于构建基于内容的推荐系统和上下文感知推荐引擎的各种库。

第七章, *使用 Spark 构建实时推荐引擎*，介绍了构建实时推荐系统所需的 Spark 和 MLlib 的基础知识。

第八章, *使用 Neo4j 构建实时推荐引擎*，介绍了图数据库和 Neo4j 概念的基础，以及如何使用 Neo4j 构建实时推荐系统。

第九章, *使用 Mahout 构建可扩展推荐引擎*，介绍了构建可扩展推荐系统所需的 Hadoop 和 Mahout 的基本构建块。它还涵盖了构建可扩展系统所使用的架构，以及使用 Mahout 和 SVD 的逐步实现。

第十章, *未来在哪里？* 是最后一章，总结了到目前为止我们所学的知识：在构建决策系统时所采用的最佳实践，以及推荐系统未来的发展方向。

# 本书所需条件

要开始使用 R、Python、Spark、Neo4j、Mahout 等不同实现方式的推荐引擎，我们需要以下软件：

| **章节编号** | **所需软件（含版本）** | **软件下载链接** | **操作系统要求** |
| --- | --- | --- | --- |
| 2,4,5 | R Studio 版本 0.99.489 | [`www.rstudio.com/products/rstudio/download/`](https://www.rstudio.com/products/rstudio/download/) | WINDOWS 7+/Centos 6 |
| 2,4,5 | R 版本 3.2.2  | [`cran.r-project.org/bin/windows/base/`](https://cran.r-project.org/bin/windows/base/) | WINDOWS 7+/Centos 6 |
| 5,6,7 | Python 3.5 的 Anaconda 4.2 | [`www.continuum.io/downloads`](https://www.continuum.io/downloads) | WINDOWS 7+/Centos 6 |
| 8 | Neo4j 3.0.6 | [`neo4j.com/download/`](https://neo4j.com/download/) | WINDOWS 7+/Centos 6 |
| 7 | Spark 2.0 | [`spark.apache.org/downloads.html`](https://spark.apache.org/downloads.html) | WINDOWS 7+/Centos 6 |
| 9 | Hadoop 2.5 - Mahout 0.12 | [`hadoop.apache.org/releases.html`](http://hadoop.apache.org/releases.html)[`mahout.apache.org/general/downloads.html`](http://mahout.apache.org/general/downloads.html) | WINDOWS 7+/Centos 6 |
| 7,9,8 | Java 7/Java 8 | [`www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html`](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) | WINDOWS 7+/Centos 6 |

# 本书面向对象

本书面向初学者和经验丰富的数据科学家，他们希望了解和构建复杂的预测决策系统、使用 R、Python、Spark、Neo4j 和 Hadoop 构建的推荐引擎。

# 规范

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 标签显示如下：“我们可以通过使用 include 指令来包含其他上下文。”

代码块设置如下：

```py
export MAHOUT_HOME = /home/softwares/ apache-mahout-distribution-0.12.2 
export MAHOUT_LOCAL = true #for standalone mode 
export PATH = $MAHOUT_HOME/bin 
export CLASSPATH = $MAHOUT_HOME/lib:$CLASSPATH 

```

任何命令行输入或输出都按如下方式编写：

```py
[cloudera@quickstart ~]$ hadoop fs –ls
Found 1 items
drwxr-xr-x - cloudera cloudera 0 2016-11-14 18:31 mahout

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中显示如下：“点击**下一步**按钮将您带到下一个屏幕。”

### 注意

警告或重要注意事项以如下框的形式出现。

### 小贴士

小贴士和技巧看起来像这样。

# 读者反馈

读者的反馈总是受欢迎的。告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们来说很重要，因为它帮助我们开发出您真正能从中获得最大收益的标题。

要向我们发送一般反馈，只需发送电子邮件至 feedback@packtpub.com，并在邮件主题中提及书的标题。

如果您在某个主题上具有专业知识，并且您对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经是 Packt 图书的骄傲拥有者，我们有一些事情可以帮助您从购买中获得最大收益。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的**支持**标签上。

1.  点击**代码下载与勘误表**。

1.  在**搜索**框中输入书的名称。

1.  选择您想要下载代码文件的书籍。

1.  从下拉菜单中选择您购买这本书的地方。

1.  点击**代码下载**。

文件下载完成后，请确保您使用最新版本的以下软件解压缩或提取文件夹：

+   适用于 Windows 的 WinRAR / 7-Zip

+   适用于 Mac 的 Zipeg / iZip / UnRarX

+   适用于 Linux 的 7-Zip / PeaZip

书籍的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/building-recommendation-engines`](https://github.com/PacktPublishing/building-recommendation-engines)。我们还有其他来自我们丰富图书和视频目录的代码包可供选择，网址为[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。查看它们吧！

## 下载本书的颜色图像

我们还为您提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。彩色图像将帮助您更好地理解输出的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/BuildingRecommendationEngines_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/BuildingRecommendationEngines_ColorImages.pdf)下载此文件。

## 勘误

尽管我们已经尽一切努力确保内容的准确性，但错误仍然会发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将在**勘误**部分显示。

## 侵权

互联网上对版权材料的侵权是一个持续存在的问题，涉及所有媒体。在 Packt，我们非常重视保护我们的版权和许可证。如果您在互联网上发现任何形式的我们作品的非法副本，请立即提供位置地址或网站名称，以便我们可以追究补救措施。

请通过版权@packtpub.com 与我们联系，并提供疑似侵权材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面的帮助。

## 问题

如果您对本书的任何方面有问题，您可以通过 questions@packtpub.com 联系我们，我们将尽力解决问题。
