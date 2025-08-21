# 前言

本书的基本目标是帮助读者快速部署、优化和评估 scikit-learn 提供的各种机器学习算法，以灵活的方式进行操作。

读者将学习如何部署有监督的机器学习算法，如逻辑回归、k 近邻、线性回归、支持向量机、朴素贝叶斯以及基于树的算法，来解决分类和回归机器学习问题。

读者还将学习如何部署无监督机器学习算法，如 k 均值算法，来将未标记的数据聚类成不同的组。

最后，读者将会学习不同的技术，用于直观地解释和评估他们所构建的算法的性能。

# 本书的适用对象

本书适合数据科学家、软件工程师以及对机器学习感兴趣并且有 Python 基础的人，他们希望通过使用 scikit-learn 框架理解、实现和评估各种机器学习算法。

# 本书的内容

第一章，*引入 scikit-learn 机器学习*，简要介绍了不同类型的机器学习及其应用。

第二章，*使用 K 近邻预测类别*，介绍了如何使用 k 近邻算法，并在 scikit-learn 中实现它来解决分类问题。

第三章，*使用逻辑回归预测类别*，解释了在 scikit-learn 中使用逻辑回归算法解决分类问题时的原理与实现。

第四章，*使用朴素贝叶斯和支持向量机预测类别*，解释了朴素贝叶斯和线性支持向量机算法在解决 scikit-learn 中的分类问题时的原理与实现。

第五章，*使用线性回归预测数值结果*，解释了在 scikit-learn 中使用线性回归算法解决回归问题时的原理与实现。

第六章，*使用树模型进行分类和回归*，解释了决策树、随机森林以及提升和集成算法在解决 scikit-learn 中的分类和回归问题时的原理与实现。

第七章，*使用无监督机器学习进行数据聚类*，解释了在 scikit-learn 中使用 k 均值算法解决无监督问题时的原理与实现。

第八章，*性能评估方法*，包含了有监督和无监督机器学习算法的可视化性能评估技术。

# 要充分利用本书

要充分利用本书：

+   假设您具备基础的 Python 知识。

+   Jupyter Notebook 是首选的开发环境，但不是必需的。

# 下载示例代码文件

您可以从您的账户下载本书的示例代码文件，网址是[www.packt.com](http://www.packt.com)。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，让我们将文件直接发送到您的邮箱。

您可以通过以下步骤下载代码文件：

1.  请登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名并按照屏幕上的指示操作。

下载文件后，请确保使用最新版本的软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，地址是[`github.com/PacktPublishing/Machine-Learning-with-scikit-learn-Quick-Start-Guide`](https://github.com/PacktPublishing/Machine-Learning-with-scikit-learn-Quick-Start-Guide)。如果代码有更新，将在现有的 GitHub 仓库中进行更新。

我们的其他代码包也可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。快去看看吧！

# 代码实战

访问以下链接查看代码运行的视频：

[`bit.ly/2OcWIGH`](http://bit.ly/2OcWIGH)

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码字、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账户名。举个例子：“将下载的`WebStorm-10*.dmg`磁盘镜像文件作为另一磁盘挂载到系统中。”

一段代码如下设置：

```py
from sklearn.naive_bayes import GaussianNB

#Initializing an NB classifier

nb_classifier = GaussianNB()

#Fitting the classifier into the training data

nb_classifier.fit(X_train, y_train)

```

```py
#Extracting the accuracy score from the NB classifier

nb_classifier.score(X_test, y_test)
```

警告或重要说明会以此形式出现。

小贴士和技巧以这种形式出现。

# 与我们联系

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何部分有疑问，请在邮件主题中注明书名，并通过`customercare@packtpub.com`联系我们。

**勘误**：尽管我们已尽一切努力确保内容的准确性，但错误难免发生。如果您在本书中发现错误，我们将感激您向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击“勘误提交表格”链接并填写相关信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，无论形式如何，我们将感激您提供相关的地址或网站名称。请通过`copyright@packt.com`与我们联系，并附上相关资料的链接。

**如果您有兴趣成为作者**：如果您在某个领域具有专业知识，并且有兴趣撰写或贡献书籍，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，为什么不在您购买书籍的网站上留下评论呢？潜在读者可以看到并参考您客观的意见来做出购买决策，我们在 Packt 也能了解您对我们产品的看法，而我们的作者也可以看到您对他们书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
