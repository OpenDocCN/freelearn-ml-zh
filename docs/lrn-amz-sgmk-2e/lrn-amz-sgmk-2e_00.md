# 前言

Amazon SageMaker 使你能够在不管理任何基础设施的情况下，快速构建、训练和部署大规模的机器学习模型。它帮助你专注于当前的机器学习问题，并通过消除每个 ML 过程步骤中通常涉及的繁重工作来部署高质量的模型。第二版将帮助数据科学家和机器学习开发人员探索新功能，如 SageMaker Data Wrangler、Pipelines、Clarify、Feature Store 等等。

你将从学习如何使用 SageMaker 的各种功能作为一个单一工具集来解决 ML 挑战开始，逐步涵盖 AutoML、内置算法和框架等功能，以及编写你自己的代码和算法来构建 ML 模型。本书随后将向你展示如何将 Amazon SageMaker 与流行的深度学习库（如 TensorFlow 和 PyTorch）集成，以扩展现有模型的功能。你将看到如何通过自动化工作流程，以最小的努力和更低的成本更快地进入生产阶段。最后，你将探索 SageMaker Debugger 和 SageMaker Model Monitor，以检测训练和生产中的质量问题。

本书结束时，你将能够在整个机器学习工作流程中使用 Amazon SageMaker，从实验、训练、监控到扩展、部署和自动化。

# 本书适用对象

本书适合软件工程师、机器学习开发人员、数据科学家以及那些刚接触 Amazon SageMaker 的 AWS 用户，帮助他们在无需担心基础设施的情况下构建高质量的机器学习模型。为了更有效地理解本书中涉及的概念，读者需要具备一定的 AWS 基础知识。对机器学习概念和 Python 编程语言的扎实理解也将大有裨益。

# 本书内容

*第一章*，*介绍 Amazon SageMaker*，提供了 Amazon SageMaker 的概述，介绍了它的功能以及它如何帮助解决今天机器学习项目中面临的许多痛点。

*第二章*，*处理数据准备技术*，讨论了数据准备选项。虽然这不是本书的核心内容，但数据准备是机器学习中的一个关键话题，值得在高层次上进行探讨。

*第三章*，*使用 Amazon SageMaker AutoPilot 进行 AutoML*，展示了如何通过 Amazon SageMaker AutoPilot 自动构建、训练和优化机器学习模型。

*第四章*，*训练机器学习模型*，展示了如何使用 Amazon SageMaker 中内置的统计机器学习算法集合来构建和训练模型。

*第五章*，*训练计算机视觉模型*，展示了如何使用 Amazon SageMaker 中内置的计算机视觉算法集合来构建和训练模型。

*第六章*，*训练自然语言处理模型*，展示了如何使用 Amazon SageMaker 中内置的自然语言处理算法集合来构建和训练模型。

*第七章*，*使用内置框架扩展机器学习服务*，展示了如何利用 Amazon SageMaker 中内置的开源框架集合来构建和训练机器学习模型。

*第八章*，*使用你的算法和代码*，展示了如何在 Amazon SageMaker 上使用自己的代码（例如 R 或自定义 Python）构建和训练机器学习模型。

*第九章*，*扩展训练任务*，展示了如何将训练任务分发到多个托管实例，无论是使用内置算法还是内置框架。

*第十章*，*高级训练技巧*，展示了如何在 Amazon SageMaker 中利用高级训练技巧。

*第十一章*，*部署机器学习模型*，展示了如何以多种配置部署机器学习模型。

*第十二章*，*自动化机器学习工作流*，展示了如何在 Amazon SageMaker 上自动化机器学习模型的部署。

*第十三章*，*优化成本和性能*，展示了如何从基础设施和成本两个角度来优化模型部署。

# 要充分利用本书内容

你需要一个可用的 AWS 账户来运行所有内容。

**如果你使用的是本书的数字版，建议你自己输入代码，或者通过 GitHub 仓库访问代码（链接将在下一节提供）。这样做有助于避免由于复制粘贴代码而产生的潜在错误。**

# 下载示例代码文件

你可以从[www.packt.com](http://www.packt.com)的账户中下载本书的示例代码文件。如果你在其他地方购买了本书，你可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接将文件发送到你的邮箱。

你可以按照以下步骤下载代码文件：

1.  登录或注册到[www.packt.com](http://www.packt.com)。

1.  选择**支持**标签。

1.  点击**代码下载**。

1.  在**搜索**框中输入书名，并按照屏幕上的指示操作。

文件下载完成后，请确保使用以下最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   适用于 Linux 的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，地址为 [`github.com/PacktPublishing/Learn-Amazon-SageMaker-second-edition`](https://github.com/PacktPublishing/Learn-Amazon-SageMaker-second-edition)。如果代码有更新，它将同步更新到现有的 GitHub 仓库中。

我们还提供了来自我们丰富图书和视频目录的其他代码包，网址为 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。快来看看吧！

# 下载彩色图片

我们还提供了一个包含本书中使用的截图/图表的彩色图片 PDF 文件。你可以在这里下载：`static.packt-cdn.com/downloads/9781801817950_ColorImages.pdf`。

# 使用的约定

本书中使用了多种文本约定。

`文中的代码`：指示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。这里是一个例子：“你可以使用 `describe-spot-price-history` API 来编程收集这些信息。”

一段代码的格式如下：

```py
od = sagemaker.estimator.Estimator(     container,     role,     train_instance_count=2,                                      train_instance_type='ml.p3.2xlarge',                                      train_use_spot_instances=True,     train_max_run=3600,                     # 1 hours      train_max_wait=7200,                    # 2 hour      output_path=s3_output)
```

当我们希望特别提醒你注意代码块中的某一部分时，相关的行或项目会以粗体显示：

```py
[<sagemaker.model_monitor.model_monitoring.MonitoringExecution at 0x7fdd1d55a6d8>,<sagemaker.model_monitor.model_monitoring.MonitoringExecution at 0x7fdd1d581630>,<sagemaker.model_monitor.model_monitoring.MonitoringExecution at 0x7fdce4b1c860>]
```

**粗体**：表示一个新术语、一个重要的词汇，或屏幕上显示的词语。例如，菜单或对话框中的词汇会像这样出现在文本中。这里是一个例子：“我们可以在 SageMaker 控制台的 **Processing jobs** 部分找到更多关于监控任务的信息。”

提示或重要说明

显示如图所示。

# 联系我们

我们始终欢迎读者的反馈。

`customercare@packtpub.com`。

**勘误**：尽管我们已尽力确保内容的准确性，但错误难免。如果你发现本书中有任何错误，我们将非常感激你能向我们报告。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)，选择你的书籍，点击“勘误提交表格”链接，并输入相关细节。

`copyright@packt.com` 并附有相关材料的链接。

**如果你有兴趣成为一名作者**：如果你在某个领域有专业知识，并且有兴趣撰写或贡献一本书，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 分享你的想法

阅读完 *Learn Amazon Sagemaker, Second Edition* 后，我们希望听到你的想法！请点击此处直接前往该书的 Amazon 评价页面，分享你的反馈。

你的评论对我们以及技术社区非常重要，将帮助我们确保提供优质内容。
