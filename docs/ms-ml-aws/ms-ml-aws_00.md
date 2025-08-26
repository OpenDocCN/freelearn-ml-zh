# 前言

AWS 不断推动新的创新，使数据科学家能够探索各种机器学习云服务。这本书是您了解和实施 AWS 上的高级机器学习算法的全面参考。

随着你阅读这本书，你将深入了解这些算法如何在 AWS 上使用 Apache Spark 在 Elastic MapReduce、SageMaker 和 TensorFlow 上进行训练、调整和部署。当你专注于 XGBoost、线性模型、因子分解机以及深度网络等算法时，本书还将为你提供 AWS 的概述，以及帮助你解决现实世界问题的详细实际应用。每个实际应用都包括一系列配套笔记本，其中包含在 AWS 上运行所需的所有代码。在接下来的几章中，你将学习如何使用 SageMaker 和 EMR 笔记本来执行一系列任务，从智能分析和预测建模到情感分析。

到这本书的结尾，你将掌握处理机器学习项目所需的技能，并在 AWS 上实现和评估算法。

# 本书面向的对象

这本书是为数据科学家、机器学习开发者、深度学习爱好者以及希望使用 AWS 和其集成服务在云上构建高级模型和智能应用的 AWS 用户而编写的。对机器学习概念、Python 编程和 AWS 的了解将有所帮助。

# 本书涵盖的内容

第一章，*AWS 机器学习入门*，向读者介绍了机器学习。它解释了为什么数据科学家有必要学习机器学习，以及 AWS 如何帮助他们解决各种现实世界的问题。我们还讨论了本书中将涵盖的 AWS 服务和工具。

第二章，*使用朴素贝叶斯分类 Twitter 流*，介绍了朴素贝叶斯算法的基础，并展示了一个将通过使用此算法和语言模型解决的问题。我们将提供示例，解释如何使用 scikit-learn 和 Apache Spark 在 SageMaker 的 BlazingText 上应用朴素贝叶斯。此外，我们还将探讨如何在更复杂的情况下使用贝叶斯推理背后的思想。我们将使用 Twitter API 从两位不同的政治候选人那里实时获取推文，并预测是谁写的。我们将使用 scikit-learn、Apache Spark、SageMaker 和 BlazingText。

第三章，*使用回归算法预测房价*，介绍了回归算法的基础知识，并将其应用于根据多个特征预测房价。我们还将介绍如何使用逻辑回归进行分类问题。将提供 SageMaker 中 scikit-learn 和 Apache Spark 的示例。我们将使用波士顿房价数据集[`www.kaggle.com/c/boston-housing/`](https://www.kaggle.com/c/boston-housing/)，以及 scikit-learn、Apache Spark 和 SageMaker。

第四章，*使用基于树的算法预测用户行为*，介绍了决策树、随机森林和梯度提升树。我们将探讨如何使用这些算法来预测用户何时会点击广告。此外，我们还将解释如何使用 AWS EMR 和 Apache Spark 在大规模上构建模型。我们将使用 Adform 点击预测数据集([`doi.org/10.7910/DVN/TADBY7`](https://doi.org/10.7910/DVN/TADBY7)，哈佛数据集，V2)。我们将使用 xgboost、Apache Spark、SageMaker 和 EMR 库。

第五章，*使用聚类算法进行客户细分*，通过探索如何根据消费者模式应用这些算法进行客户细分，介绍了主要的聚类算法。通过 AWS SageMaker，我们将展示如何在 skicit-learn 和 Apache Spark 中运行这些算法。我们将使用来自 Fabien Daniel 的电子商务数据([`www.kaggle.com/fabiendaniel/customer-segmentation/data`](https://www.kaggle.com/fabiendaniel/customer-segmentation/data))以及 scikit-learn、Apache Spark 和 SageMaker。

第六章，*分析访问模式以制定推荐策略*，提出了基于用户导航模式寻找相似用户的问题，以便推荐定制营销策略。我们将介绍协同过滤和基于距离的方法，并在 AWS SageMaker 上的 scikit-learn 和 Apache Spark 中提供示例。我们将使用 Kwan Hui Lim 的游乐场景点访问数据集([`sites.google.com/site/limkwanhui/datacode`](https://sites.google.com/site/limkwanhui/datacode))、Apache Spark 和 SageMaker。

第七章，*实现深度学习算法*，向读者介绍了深度学习背后的主要概念，并解释了为什么它在今天的 AI 产品中变得如此重要。本章的目的是不讨论深度学习的理论细节，而是通过示例解释算法，并提供对深度学习算法的高级概念理解。这将给读者提供一个平台，以了解他们在下一章中将要实现的内容。

第八章, 《在 AWS 上使用 TensorFlow 实现深度学习*》*，通过一系列实用的图像识别问题，并解释如何使用 AWS 上的 TensorFlow 来解决这些问题。TensorFlow 是一个非常流行的深度学习框架，可以用来训练深度神经网络。本章将解释读者如何安装 TensorFlow，并使用玩具数据集来训练深度学习模型。在本章中，我们将使用 MNIST 手写数字数据集 ([`yann.lecun.com/exdb/mnist/`](http://yann.lecun.com/exdb/mnist/))，以及 TensorFlow 和 SageMaker。

第九章, 《使用 SageMaker 进行图像分类和检测*》*，回顾了我们在前几章中处理过的图像分类问题，但这次使用 SageMaker 的图像分类算法和目标检测算法。我们将使用以下数据集：

+   Caltech256 ([`www.vision.caltech.edu/Image_Datasets/Caltech256/`](http://www.vision.caltech.edu/Image_Datasets/Caltech256/))

我们还将使用 AWS Sagemaker。

第十章, 《与 AWS Comprehend 协作*》*，解释了 AWS 工具 Comprehend 的功能，这是一个执行各种有用任务的 NLP 工具。

第十一章, 《使用 AWS Rekognition*》*，解释了如何使用 Rekognition，这是一个使用深度学习的图像识别工具。读者将学习一种简单的方法，将图像识别应用于他们的应用程序中。

第十二章, 《使用 AWS Lex 构建对话界面*》*，解释了 AWS Lex 是一个允许程序员构建对话界面的工具。本章向读者介绍了诸如使用深度学习进行自然语言理解等主题。

第十三章, 《在 AWS 上创建集群*》*，讨论了深度学习中的一个关键问题，即理解如何在多台机器上扩展和并行化学习。在本章中，我们将探讨创建学习者集群的不同方法。特别是，我们将关注如何通过分布式 TensorFlow 和 Apache Spark 并行化深度学习管道。

第十四章, 《在 Spark 和 SageMaker 中优化模型*》*，解释了在 AWS 上训练的模型可以进一步优化，以便在生产环境中平稳运行。在本节中，我们将讨论读者可以使用的一些技巧，以改善他们算法的性能。

第十五章，*调整集群以适应机器学习*，解释了许多数据科学家和机器学习实践者在尝试大规模运行机器学习数据管道时面临规模问题。在本章中，我们主要关注 EMR，这是一个运行非常大的机器学习作业的非常强大的工具。配置 EMR 有许多方法，并不是每一种设置都适用于每一种场景。我们将介绍 EMR 的主要配置，并解释每种配置如何适用于不同的目标。此外，我们还将介绍其他通过 AWS 运行大数据管道的方法。

第十六章，*在 AWS 上构建的模型部署*，讨论了部署问题。到这时，读者将已经在 AWS 上构建了模型，并希望将它们部署到生产环境中。我们理解模型应该部署的上下文有很多种。在某些情况下，这就像生成一个将输入到某些系统的动作的 CSV 文件一样简单。通常，我们只需要部署一个能够进行预测的 Web 服务。然而，有许多时候我们需要将这些模型部署到复杂、低延迟或边缘系统中。我们将介绍您可以将机器学习模型部署到生产环境的不同方法。

# 要充分利用本书

本书涵盖了多个不同的框架，例如 Spark 和 Tensorflow。但它并不是针对每个框架的全面指南。相反，我们关注 AWS 如何通过使用不同的框架来赋予实际机器学习能力。我们鼓励读者在需要时参考其他具有特定框架内容的书籍。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

一旦文件下载完成，请确保您使用最新版本的软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Mastering-Machine-Learning-on-AWS`](https://github.com/PacktPublishing/Mastering-Machine-Learning-on-AWS)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789349795_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789349795_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“以下截图显示了我们的`df`数据框的前几行。”

代码块设置如下：

```py
vectorizer = CountVectorizer(input=dem_text + gop_text,
                             stop_words=stop_words,
                             max_features=1200)
```

任何命令行输入或输出都按以下方式编写：

```py
wget -O /tmp/adform.click.2017.01.json.gz https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/TADBY7/JCI3VG
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“您还可以在 AWS Comprehend 中使用自定义 NER 算法进行训练，方法是在左侧菜单中选择自定义 | 自定义实体识别选项。”

警告或重要注意事项显示如下。

技巧和窍门显示如下。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并给我们发送邮件至`customercare@packtpub.com`。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这个错误。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，我们将不胜感激，如果您能提供位置地址或网站名称。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用过这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
