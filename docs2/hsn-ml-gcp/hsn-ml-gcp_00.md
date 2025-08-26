# 前言

谷歌云机器学习引擎结合了谷歌云平台的服务与 TensorFlow 的强大功能和灵活性。通过本书，您不仅将学习如何以规模构建和训练不同复杂性的机器学习模型，还将学习如何在云端托管这些模型以进行预测。

本书专注于充分利用谷歌机器学习平台处理大型数据集和复杂问题。您将通过利用谷歌云平台的不同数据服务，从头开始创建强大的基于机器学习的应用程序，解决各种问题。应用包括自然语言处理、语音转文本、强化学习、时间序列、推荐系统、图像分类、视频内容推理等。我们将实现各种深度学习用例，并广泛使用构成谷歌云平台生态系统的数据相关服务，如 Firebase、存储 API、Datalab 等。这将使您能够将机器学习和数据处理功能集成到您的 Web 和移动应用程序中。

在本书结束时，您将了解您可能遇到的主要困难，并熟悉克服这些困难并构建高效系统的适当策略。

# 本书面向读者

本书面向数据科学家、机器学习开发者和希望学习如何使用谷歌云平台服务构建机器学习应用的人工智能开发者。由于与谷歌机器学习平台的大部分交互都是通过命令行完成的，因此读者应熟悉 bash shell 和 Python 脚本。对机器学习和数据科学概念的理解也将很有帮助。

# 本书涵盖内容

第一章，*介绍谷歌云平台*，探讨了可能对基于 GCP 构建机器学习管道有用的不同服务。

第二章，*谷歌计算引擎*，帮助您通过在线控制台和命令行工具创建并完全管理您的虚拟机，以及如何实现数据科学工作流程和 Jupyter Notebook 工作空间。

第三章，*谷歌云存储*，展示了如何使用谷歌云平台提供的服务上传数据和管理数据。

第四章，*使用 BigQuery 查询您的数据*，展示了如何从谷歌存储中查询数据，并使用谷歌数据工作室进行可视化。

第五章，*转换您的数据*，介绍了 Dataprep，这是一个用于数据预处理、提取特征和清理记录的有用服务。我们还探讨了 Dataflow，这是一个用于实现流式和批量处理的服务。

第六章，*机器学习基础*，开始了我们对机器学习和深度学习的探索之旅；我们学习何时应用每一种。

第七章，*Google 机器学习 API*，教导我们如何使用 Google Cloud 机器学习 API 进行图像分析、文本和语音处理、翻译以及视频推理。

第八章，*使用 Firebase 创建机器学习应用*，展示了如何整合不同的 GCP 服务来构建一个无缝的基于机器学习的应用，无论是移动端还是基于 Web 的应用。

第九章，*使用 TensorFlow 和 Keras 的神经网络*，让我们对前馈网络的结构和关键元素有了良好的理解，如何构建一个架构，以及如何调整和实验不同的参数。

第十章，*使用 TensorBoard 评估结果*，展示了不同参数和函数的选择如何影响模型的表现。

第十一章，*通过超参数调整优化模型*，教导我们如何在 TensorFlow 应用代码中使用超参数调整，并解释结果以选择表现最佳的模型。

第十二章，*通过正则化防止过拟合*，展示了如何通过设置正确的参数和定义适当的架构来识别过拟合，并使我们的模型对之前未见过的数据更加鲁棒。

第十三章，*超越前馈网络——CNN 和 RNN*，教导我们针对不同问题应用哪种类型的神经网络，以及如何在 GCP 上定义和实现它们。

第十四章，*使用 LSTMs 进行时间序列分析*，展示了如何创建 LSTMs 并将它们应用于时间序列预测。我们还将了解何时 LSTMs 优于更标准的方法。

第十五章，*强化学习*，介绍了强化学习的力量，并展示了如何在 GCP 上实现一个简单的用例。

第十六章，*生成型神经网络*，教导我们如何使用不同类型的内容——文本、图像和声音——从神经网络中提取生成的内容。

第十七章，*聊天机器人*，展示了如何在实现真实移动应用的同时训练一个上下文聊天机器人。

# 为了充分利用这本书

本书在 Google Cloud Platform 上实现了机器学习算法。为了重现本书中的许多示例，您需要在 GCP 上拥有一个有效的账户。我们使用了 Python 2.7 及以上版本来构建各种应用程序。本着这个精神，我们尽量使所有代码都尽可能友好和易于阅读。我们相信这将使我们的读者能够轻松理解代码，并在不同场景中轻松使用它。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择支持选项卡。

1.  点击代码下载与勘误。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载完成后，请确保使用最新版本的以下软件解压或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Machine-Learning-on-Google-Cloud-Platform`](https://github.com/PacktPublishing/Hands-On-Machine-Learning-on-Google-Cloud-Platform)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有来自我们丰富图书和视频目录的其他代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**上找到。查看它们！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnMachineLearningonGoogleCloudPlatform_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/HandsOnMachineLearningonGoogleCloudPlatform_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“其中`GROUP`是一个服务或账户元素，`COMMAND`是要发送到`GROUP`的命令。”

代码块设置如下：

```py
import matplotlib.patches as patches
import numpy as np
fig,ax = plt.subplots(1)
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
text="this is a good text"
from google.cloud.language_v1 import types
document = types.Document(
        content=text,
        type='PLAIN_TEXT')
```

任何命令行输入或输出都应如下所示：

```py
$ gcloud compute instances list
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“点击创建新项目。”

警告或重要注意事项如下所示。

小贴士和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：请发送电子邮件至`feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请发送电子邮件至`questions@packtpub.com`。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您能向我们报告。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，我们将非常感激您能提供位置地址或网站名称。请通过发送链接至`copyright@packtpub.com`与我们联系。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。一旦您阅读并使用过这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，并且我们的作者可以查看他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问[packtpub.com](https://www.packtpub.com/)。
