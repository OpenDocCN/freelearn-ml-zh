# 前言

推荐系统是预测用户购买和偏好的机器学习技术。推荐系统有几种应用，例如在线零售商和视频分享网站。

本书教导读者如何使用 R 构建推荐系统。它首先向读者提供一些相关的数据挖掘和机器学习概念。然后，展示了如何使用 R 构建和优化推荐模型，并概述了最流行的推荐技术。最后，展示了实际应用案例。阅读本书后，你将知道如何独立构建新的推荐系统。

# 本书涵盖内容

第一章, *推荐系统入门*，介绍了本书内容并展示了推荐引擎的一些实际应用案例。

第二章, *推荐系统中使用的数据挖掘技术*，为读者提供了构建推荐模型的工具箱：R 基础知识、数据处理和机器学习技术。

第三章, *推荐系统*，介绍了几个流行的推荐系统，并展示了如何使用 R 构建其中的一些。

第四章, *评估推荐系统*，展示了如何衡量推荐系统的性能以及如何优化它。

第五章, *案例研究 – 构建自己的推荐引擎*，展示了如何通过构建和优化推荐系统来解决商业挑战。

# 本书所需条件

你将需要 R 3.0.0+、RStudio（非必需）和 Samba 4.x 服务器软件。

# 本书面向对象

本书面向已经具备 R 和机器学习背景知识的人群。如果你对构建推荐技术感兴趣，这本书适合你。

# 引用

在出版物中引用`recommenderlab`包（R 包版本 0.1-5），请参考由*Michael Hahsler*编写的*recommenderlab: Lab for Developing and Testing Recommender Algorithms*，可在[`CRAN.R-project.org/package=recommenderlab`](http://CRAN.R-project.org/package=recommenderlab)找到。

LaTeX 用户可以使用以下 BibTeX 条目：

```py
@Manual{,
  title = {recommenderlab: Lab for Developing and Testing
  Recommender Algorithms},
  author = {Michael Hahsler},
  year = {2014},
  note = {R package version 0.1-5},
  url = { http://CRAN.R-
  project.org/package=recommenderlab},
}
```

# 习惯用法

在本书中，你会找到多种文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称将如下所示："我们使用了`e1071`包来运行 SVM"。

代码块设置如下：

```py
vector_ratings <- factor(vector_ratings)qplot(vector_ratings) + ggtitle("Distribution of the ratings")
exten => i,1,Voicemail(s0)
```

**新术语**和**重要词汇**将以粗体显示。

### 注意

警告或重要提示将以这样的框显示。

### 小贴士

小贴士和技巧看起来像这样。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们很重要，因为它帮助我们开发出您真正能从中获得最大价值的标题。

要发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及书籍的标题。

如果您在某个领域有专业知识，并且您对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南，网址为[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经成为 Packt 书籍的骄傲拥有者，我们有多种方式帮助您从购买中获得最大收益。

## 下载示例代码

您可以从[`www.packtpub.com`](http://www.packtpub.com)下载您购买的所有 Packt 出版物的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 下载本书的颜色图像

我们还为您提供了一个包含本书中使用的截图/图表的颜色图像的 PDF 文件。这些颜色图像将帮助您更好地理解输出的变化。您可以从：[`www.packtpub.com/sites/default/files/downloads/4492OS_GraphicBundle.pdf`](https://www.packtpub.com/sites/default/files/downloads/4492OS_GraphicBundle.pdf)下载此文件。

## 勘误

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的某本书中发现了错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误****提交****表**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**勘误**部分下。

## 盗版

互联网上对版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现任何形式的我们作品的非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<copyright@packtpub.com>` 联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面的帮助。

## 问答

如果您对本书的任何方面有问题，请通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决问题。
