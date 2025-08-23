# 前言

随着机器学习（ML）工程需求的专业人士以及那些了解在云中自动化复杂**MLOps**管道的人越来越多，对这类专业人士的需求也在不断增长。本书探讨了各种AWS服务，例如**Amazon Elastic Kubernetes Service**、**AWS Glue**、**AWS Lambda**、**Amazon Redshift**和**AWS Lake Formation**，这些服务可以帮助机器学习从业者满足生产中的各种数据工程和机器学习工程需求。

这本机器学习书籍涵盖了基本概念以及旨在帮助您深入了解如何在云中管理和保护机器学习工作负载的逐步指导。随着您逐步阅读各章节，您将发现如何在AWS上使用各种**容器**和**无服务器**解决方案来训练和部署**TensorFlow**和**PyTorch**深度学习模型。您还将深入了解在探索每个AWS的最佳实践时，如何详细探讨经过验证的成本优化技术以及数据隐私和模型隐私保护策略。

在阅读完这本AWS书籍之后，您将能够构建、扩展并保护您自己的机器学习系统和管道，这将为您使用各种AWS服务来构建满足机器学习工程需求的定制解决方案提供所需的经验和信心。

# 本书面向的对象

本书面向的是对在生产数据工程、机器学习工程和MLOps需求中使用各种AWS服务（如**Amazon EC2**、**Amazon Elastic Kubernetes Service**（**EKS**）、**Amazon SageMaker**、**AWS Glue**、**Amazon Redshift**、**AWS Lake Formation**和**AWS Lambda**）感兴趣的机器学习工程师、数据科学家和AWS云工程师——您只需一个AWS账户即可开始。对AWS、机器学习和Python编程语言的了解将帮助您更有效地掌握本书中涵盖的概念。

# 本书涵盖的内容

*第一章*，*AWS上的机器学习工程简介*，专注于帮助您快速设置环境，理解关键概念，并通过几个简化的AutoML示例快速入门。

*第二章*，*深度学习AMI*，介绍了AWS深度学习AMI及其如何帮助机器学习从业者更快地在EC2实例内执行机器学习实验。在这里，我们还将深入探讨AWS如何为EC2实例定价，以便您更好地了解如何优化和降低在云中运行机器学习工作负载的整体成本。

*第三章*，*深度学习容器*，介绍了AWS深度学习容器及其如何帮助机器学习从业者使用容器更快地执行机器学习实验。在这里，我们还将利用Lambda的容器镜像支持，在AWS Lambda函数内部署一个训练好的深度学习模型。

*第4章*，*AWS上的无服务器数据管理*，介绍了用于在AWS上管理和查询数据的几个无服务器解决方案，例如Amazon Redshift Serverless和AWS Lake Formation。

*第5章*，*实用数据处理和分析*，专注于在处理数据和分析需求时可用不同的服务，例如AWS Glue DataBrew和Amazon SageMaker Data Wrangler。

*第6章*，*SageMaker训练和调试解决方案*，介绍了使用Amazon SageMaker训练ML模型时可用不同的解决方案和能力。在这里，我们将更深入地探讨在SageMaker中训练和调整ML模型的不同选项和策略。

*第7章*，*SageMaker部署解决方案*，专注于在AWS平台上执行ML推理时的相关部署解决方案和策略。

*第8章*，*模型监控和管理解决方案*，介绍了在AWS上可用的不同监控和管理解决方案。

*第9章*，*安全、治理和合规策略*，专注于确保生产环境所需的相关安全、治理和合规策略。在这里，我们还将更深入地探讨确保数据隐私和模型隐私的不同技术。

*第10章*，*在Amazon EKS上使用Kubeflow的机器学习管道*，专注于使用Kubeflow Pipelines、Kubernetes和Amazon EKS在AWS上部署自动化的端到端MLOps管道。

*第11章*，*使用SageMaker Pipelines的机器学习管道*，专注于使用SageMaker Pipelines设计和构建自动化的端到端MLOps管道。在这里，我们将应用、组合和连接我们在本书前几章中学到的不同策略和技术。

# 为了充分利用本书

为了完成本书中的动手解决方案，您需要一个AWS账户和一个稳定的互联网连接。如果您还没有AWS账户，请随意查看**AWS免费层**页面并点击**创建免费账户**：[https://aws.amazon.com/free/](https://aws.amazon.com/free/)。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| **支持的软件/硬件** | **操作系统要求** |

**如果您正在使用本书的数字版，我们建议您自己输入代码或从** **本书的GitHub仓库（下一节中有一个链接）** **访问代码。这样做将帮助您避免与代码复制和粘贴相关的任何潜在错误。**

# 下载示例代码文件

您可以从GitHub下载本书的示例代码文件[https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS](https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS)。如果代码有更新，它将在GitHub仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在 [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/) 获取。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的PDF文件。您可以从这里下载：[https://packt.link/jeBII](https://packt.link/jeBII)。

# 使用的约定

本书中使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟URL、用户输入和Twitter昵称。以下是一个示例：“`ENTRYPOINT` 设置为 `/opt/conda/bin/python -m awslambdaric`。然后 `CMD` 命令设置为 `app.handler`。`ENTRYPOINT` 和 `CMD` 指令定义了容器启动时执行哪个命令。”

代码块设置如下：

[PRE0]

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

[PRE1]

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词以粗体显示。以下是一个示例：“点击 **FILTER** 按钮后，应出现下拉菜单。在 **条件** 下的选项列表中找到并选择 **大于等于**。这应该更新页面右侧的面板，并显示 **过滤器值** 操作的配置选项列表。”

小贴士或重要提示

看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过 [customercare@packtpub.com](http://customercare@packtpub.com) 发送电子邮件给我们。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过 [copyright@packt.com](http://copyright@packt.com) 与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读了 *AWS上的机器学习工程*，我们很乐意听到您的想法！[请点击此处直接进入此书的亚马逊评论页面](https://packt.link/r/1-803-24759-2%0D) 并分享您的反馈。

您的审阅对我们和科技社区都至关重要，并将帮助我们确保我们提供高质量的内容。

# 第1部分：在 AWS 上开始机器学习工程

在本节中，读者将了解 AWS 上的机器学习工程世界。

本节包含以下章节：

+   [*第1章*](B18638_01.xhtml#_idTextAnchor017), *AWS 机器学习工程简介*

+   [*第2章*](B18638_02.xhtml#_idTextAnchor041), *深度学习 AMI*

+   [*第3章*](B18638_03.xhtml#_idTextAnchor060), *深度学习容器*
