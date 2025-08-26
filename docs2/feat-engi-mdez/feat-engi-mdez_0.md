# 前言

本书将涵盖特征工程这一主题。特征工程是数据科学和机器学习管道的重要组成部分，它包括识别、清理、构建和发现数据的新特征的能力，目的是为了解释和预测分析。

在本书中，我们将涵盖特征工程的整个流程，从检查到可视化、转换以及更多。我们将使用基本的和高级的数学度量来将我们的数据转换成机器和机器学习管道更容易消化和理解的形式。

通过发现和转换，作为数据科学家，我们将能够从全新的视角看待我们的数据，这不仅增强了我们的算法，也增强了我们的洞察力。

# 本书面向的对象

本书是为那些希望理解和利用机器学习和数据探索中特征工程实践的人而写的。

读者应该对机器学习和 Python 编程相当熟悉，以便能够舒适地通过逐步解释基础知识来深入探索新主题。

# 本书涵盖的内容

第一章，*特征工程简介*，介绍了特征工程的基本术语，并快速浏览了本书中将解决的问题的类型。

第二章，*特征理解 – 我的数据集中有什么？*，探讨了我们在野外可能会遇到的数据类型，以及如何分别或共同处理每一种类型。

第三章，*特征改进 - 清理数据集*，解释了各种填充缺失数据的方法，以及不同的技术如何导致数据结构发生变化，这可能导致机器学习性能下降。

第四章，*特征构建*，探讨了如何根据已经给出的信息创建新的特征，以努力增加数据的结构。

第五章，*特征选择*，展示了如何通过量化指标来决定哪些特征值得我们保留在数据管道中。

第六章，*特征转换*，运用高级线性代数和数学技术，对数据进行刚性结构化，以增强我们管道的性能。

第七章，*特征学习*，涵盖了使用最先进的机器学习和人工智能学习算法来发现数据中人类难以理解的潜在特征。

第八章，*案例研究*，展示了一系列案例研究，旨在巩固特征工程的概念。

# 为了最大限度地利用本书

我们需要为本书准备的内容：

1.  本书使用 Python 来完成所有的代码示例。需要一个可以访问 Unix 风格终端并已安装 Python 2.7 的机器（Linux/Mac/Windows 均可）。

1.  建议安装 Anaconda 发行版，因为它包含了示例中使用的大多数软件包。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载与勘误。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   Windows 版的 WinRAR/7-Zip

+   Mac 版的 Zipeg/iZip/UnRarX

+   Linux 版的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Feature-Engineering-Made-Easy`](https://github.com/PacktPublishing/Feature-Engineering-Made-Easy)。我们还有其他来自我们丰富图书和视频目录的代码包可供选择，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。请查看它们！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/FeatureEngineeringMadeEasy_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/FeatureEngineeringMadeEasy_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“假设进一步，给定这个数据集，我们的任务是能够接受三个属性（`datetime`、`protocol`和`urgent`）并能够准确预测`malicious`的值。用通俗易懂的话说，我们希望有一个系统可以将`datetime`、`protocol`和`urgent`的值映射到`malicious`中的值。”

代码块设置如下：

```py
Network_features = pd.DataFrame({'datetime': ['6/2/2018', '6/2/2018', '6/2/2018', '6/3/2018'], 'protocol': ['tcp', 'http', 'http', 'http'], 'urgent': [False, True, True, False]})
Network_response = pd.Series([True, True, False, True])
Network_features
>>
 datetime protocol  urgent
0  6/2/2018      tcp   False
1  6/2/2018     http    True
2  6/2/2018     http    True
3  6/3/2018     http   False
Network_response
>>
 0     True
1     True
2    False
3     True
dtype: bool
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
times_pregnant                  0.221898
plasma_glucose_concentration    0.466581 diastolic_blood_pressure        0.065068
triceps_thickness               0.074752
serum_insulin                   0.130548
bmi                             0.292695
pedigree_function               0.173844
age                             0.238356
onset_diabetes                  1.000000
Name: onset_diabetes, dtype: float64
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。

警告或重要提示如下所示。

小技巧如下所示。

# 联系我们

我们读者的反馈总是受欢迎的。

**一般反馈**：请发送电子邮件至`feedback@packtpub.com`，并在邮件主题中提及书名。如果您对本书的任何方面有疑问，请通过电子邮件联系我们的`questions@packtpub.com`。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上以任何形式发现我们作品的非法副本，如果您能向我们提供位置地址或网站名称，我们将不胜感激。请通过 `copyright@packtpub.com` 联系我们，并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/).

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packtpub.com](https://www.packtpub.com/).
