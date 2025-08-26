# 前言

机器学习不仅对学术界产生了巨大影响，也对工业界产生了巨大影响，因为它将数据转化为可操作的智能。Scala 不仅是一种面向对象和函数式编程语言，还可以利用**Java 虚拟机**（**JVM**）的优势。Scala 提供了代码复杂度优化，并提供了简洁的表示法，这可能是它在过去几年中持续增长的原因，尤其是在数据科学和分析领域。

本书面向有志于成为数据科学家、数据工程师和深度学习爱好者的初学者，他们希望在学习机器学习最佳实践方面有一个良好的起点。即使你对机器学习概念不太熟悉，但仍然想通过 Scala 深入实践监督学习、无监督学习和推荐系统来扩展你的知识，你将能够轻松掌握本书的内容！

在各章节中，你将熟悉 Scala 中流行的机器学习库，学习如何使用线性方法和基于树的集成技术进行回归和分类分析，以及查看聚类分析、降维和推荐系统，最后深入到深度学习。

阅读本书后，你将在解决更复杂的机器学习任务方面有一个良好的起点。本书并非要求从头到尾阅读。你可以翻到看起来像是你想要完成的章节，或者激发你兴趣的章节。

欢迎提出改进建议。祝您阅读愉快！

# 本书面向对象

对于想要在 Scala 中学习如何训练机器学习模型，而又不想花费太多时间和精力的机器学习开发者来说，这本书将会非常有用。你只需要一些 Scala 编程的基本知识和一些统计学和线性代数的基础知识，就可以开始阅读这本书。

# 本书涵盖内容

第一章，《使用 Scala 的机器学习入门》，首先解释了机器学习的一些基本概念和不同的学习任务。然后讨论了基于 Scala 的机器学习库，接着是配置编程环境。最后简要介绍了 Apache Spark，并在最后通过一个逐步示例进行演示。

第二章，《Scala 回归分析》，通过示例介绍了监督学习任务回归分析，随后是回归度量。然后解释了一些回归分析算法，包括线性回归和广义线性回归。最后，它展示了使用 Scala 中的 Spark ML 逐步解决回归分析任务的方法。

第三章，*Scala 用于学习分类*，简要解释了另一个称为分类的监督学习任务，并举例说明，随后解释了如何解释性能评估指标。然后它涵盖了广泛使用的分类算法，如逻辑回归、朴素贝叶斯和**支持向量机**（**SVMs**）。最后，它通过使用 Spark ML 在 Scala 中逐步解决一个分类问题的示例来演示。

第四章，*Scala 用于基于树的集成技术*，涵盖了非常强大且广泛使用的基于树的途径，包括决策树、梯度提升树和随机森林算法，用于分类和回归分析。然后它回顾了第二章，*Scala 用于回归分析*，和第三章，*Scala 用于学习分类*的示例，在解决这些问题时使用这些基于树的算法。

第五章，*Scala 用于降维和聚类*，简要讨论了不同的聚类分析算法，随后通过一个解决聚类问题的逐步示例。最后，它讨论了高维数据中的维度诅咒，并在使用**主成分分析**（**PCA**）解决该问题的示例之前进行说明。

第六章，*Scala 用于推荐系统*，简要介绍了基于相似度、基于内容和协同过滤的方法来开发推荐系统。最后，它通过一个使用 Spark ML 在 Scala 中的示例来演示一个书籍推荐系统。

第七章，*使用 Scala 的深度学习简介*，简要介绍了深度学习、人工神经网络和神经网络架构。然后讨论了一些可用的深度学习框架。最后，它通过一个使用**长短期记忆**（**LSTM**）网络的逐步示例来演示如何解决癌症类型预测问题。

# 为了充分利用这本书

所有示例都已使用一些开源库在 Scala 中实现，包括 Apache Spark MLlib/ML 和 Deeplearning4j。然而，为了充分利用这一点，你应该拥有一台功能强大的计算机和软件栈。

Linux 发行版更受欢迎（例如，Debian、Ubuntu 或 CentOS）。例如，对于 Ubuntu，建议在 VMware Workstation Player 12 或 VirtualBox 上至少安装 64 位的 14.04（LTS）完整版。你同样可以在 Windows（7/8/10）或 macOS X（10.4.7+）上运行 Spark 作业。

推荐使用具有 Core i5 处理器的计算机，足够的存储空间（例如，运行 Spark 作业时，您至少需要 50 GB 的空闲磁盘存储空间用于独立集群和 SQL 仓库），以及至少 16 GB 的 RAM。如果想要在 GPU 上执行神经网络训练（仅限于最后一章），则需要安装带有 CUDA 和 CuDNN 配置的 NVIDIA GPU 驱动程序。

为了执行本书中的源代码，需要以下 API 和工具：

+   Java/JDK，版本 1.8

+   Scala，版本 2.11.8

+   Spark，版本 2.2.0 或更高

+   Spark csv_2.11，版本 1.3.0

+   ND4j 后端版本 nd4j-cuda-9.0-platform 用于 GPU；否则，nd4j-native

+   ND4j，版本 1.0.0-alpha

+   DL4j，版本 1.0.0-alpha

+   Datavec，版本 1.0.0-alpha

+   Arbiter，版本 1.0.0-alpha

+   Eclipse Mars 或 Luna（最新版本）或 IntelliJ IDEA

+   Maven Eclipse 插件（2.9 或更高）

+   用于 Eclipse 的 Maven 编译插件（2.3.2 或更高版本）

+   Maven Eclipse 插件（2.4.1 或更高）

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本解压缩或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，网址为**[`github.com/PacktPublishing/Machine-Learning-with-Scala-Quick-Start-Guide`](https://github.com/PacktPublishing/Machine-Learning-with-Scala-Quick-Start-Guide)**。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他代码包，这些代码包来自我们丰富的书籍和视频目录，可在以下网址找到：**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。查看它们吧！

# 代码实战

访问以下链接查看代码运行的视频：

[`bit.ly/2WhQf2i`](http://bit.ly/2WhQf2i)

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“它给了我一个马修斯相关系数为`0.3888239300421191`。”

代码块设置如下：

```py
rawTrafficDF.select("Hour (Coded)", "Immobilized bus", "Broken Truck", 
                    "Vehicle excess", "Fire", "Slowness in traffic (%)").show(5)
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
// Create a decision tree estimator
val dt = new DecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxBins(10)
      .setMaxDepth(30)
      .setLabelCol("label")
      .setFeaturesCol("features")
```

任何命令行输入或输出都应如下编写：

```py
 +-----+-----+
 |churn|count|
 +-----+-----+
 |False| 2278|
 | True| 388 |
 +-----+-----+
```

**粗体**: 表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中会这样显示。例如：“点击 Next 按钮将您带到下一屏幕。”

警告或重要注意事项会像这样显示。

小贴士和技巧会像这样显示。

# 联系我们

欢迎读者们的反馈。

**一般反馈**: 如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并将邮件发送至`customercare@packtpub.com`。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一点。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上发现我们作品的任何非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**: 如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为什么不在您购买它的网站上留下评论呢？潜在读者可以看到并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
