# 前言

Amazon SageMaker Studio 是第一个**集成开发环境**（**IDE**）为**机器学习**（**ML**）设计，旨在整合 ML 工作流程：数据准备、特征工程、统计偏差检测、**自动化机器学习**（**AutoML**）、训练、托管、ML 可解释性、监控和 MLOps 在一个环境中。

在本书中，你将从探索 Amazon SageMaker Studio 中可用的功能开始，以分析数据、开发 ML 模型，并将模型生产化以满足你的目标。随着你的进步，你将了解这些功能如何协同工作，以解决在生产中构建 ML 模型时遇到的常见挑战。之后，你将了解如何有效地使用 SageMaker Studio 规模化和运营 ML 生命周期。

在本书结束时，你将学习有关 Amazon SageMaker Studio 的 ML 最佳实践，以及能够在 ML 开发生命周期中提高生产力，并轻松为你的 ML 用例构建和部署模型。

# 本书面向的对象

本书面向数据科学家和 ML 工程师，他们希望熟练掌握 Amazon SageMaker Studio，并获得实际 ML 经验来处理 ML 生命周期的每个步骤，包括构建数据以及训练和托管模型。尽管需要基本了解 ML 和数据科学，但不需要任何先前的 SageMaker Studio 或云经验。

# 本书涵盖的内容

*第一章*, *机器学习及其在云中的生命周期*，描述了云计算技术如何使机器学习领域民主化，以及机器学习如何在云中部署。它介绍了本书中使用的 AWS 服务的 fundamentals。

*第二章*, *介绍 Amazon SageMaker Studio*，涵盖了 Amazon SageMaker Studio 的概述，包括其功能和用户界面组件。你将设置一个 SageMaker Studio 域，并熟悉基本操作。

*第三章*, *使用 SageMaker Data Wrangler 进行数据准备*，探讨了如何使用 SageMaker Data Wrangler 通过点选操作（即，无需任何编码）执行探索性数据分析以及为 ML 模型进行数据预处理。你将能够快速迭代数据转换和建模，以查看你的转换配方是否有助于提高模型性能，了解数据中是否存在对敏感群体的隐含偏差，并清楚地记录对处理数据所做的转换。

*第四章*, *使用 SageMaker Feature Store 构建特征库*，探讨了 SageMaker Feature Store，它允许存储用于机器学习训练和推理的特征。特征库作为协作开发机器学习用例的团队的中央存储库，以避免在创建特征时重复和混淆工作。SageMaker Feature Store 使得存储和访问训练和推理数据变得更加容易和快速。

*第五章*, *使用 SageMaker Studio IDE 构建和训练机器学习模型*，探讨了如何使构建和训练机器学习模型变得简单。不再需要在配置和管理计算基础设施时感到沮丧。SageMaker Studio 是一个为机器学习开发者设计的集成开发环境。在本章中，您将学习如何使用 SageMaker Studio IDE、笔记本和 SageMaker 管理的训练基础设施。

*第六章*, *使用 SageMaker Clarify 检测机器学习偏差和解释模型*，介绍了在机器学习生命周期中检测和修复数据与模型中的偏差的能力，这对于创建具有社会公平性的机器学习模型至关重要。您将学习如何应用 SageMaker Clarify 来检测数据中的偏差，以及如何阅读 SageMaker Clarify 中的指标。

*第七章*, *在云中托管机器学习模型：最佳实践*，探讨了在成功训练模型后，如果您想使模型可用于推理，SageMaker 根据您的用例提供了几个选项。您将学习如何托管用于批量推理的模型，进行在线实时推理，以及使用多模型端点以节省成本，以及针对您的推理需求的一种资源优化策略。

*第八章*, *使用 SageMaker JumpStart 和 Autopilot 快速启动机器学习*，探讨了 SageMaker JumpStart，它为选定的用例提供完整的解决方案，作为进入 Amazon SageMaker 机器学习世界的入门套件，无需任何代码开发。SageMaker JumpStart 还为您整理了流行的预训练**计算机视觉**（**CV**）和**自然语言处理**（**NLP**）模型，以便您轻松部署或微调到您的数据集中。SageMaker Autopilot 是一个 AutoML 解决方案，它探索您的数据，代表您构建特征，并从各种算法和超参数中训练最优模型。您无需编写任何代码，因为 Autopilot 会为您完成这些工作，并返回笔记本以展示它是如何做到的。

*第九章*，*在 SageMaker Studio 中大规模训练机器学习模型*，讨论了典型的机器学习生命周期通常从原型设计开始，然后过渡到生产规模，数据量将大大增加，模型将更加复杂，实验数量呈指数增长。SageMaker Studio 使这种过渡比以前更容易。您将学习如何运行分布式训练，如何监控训练作业的计算资源和建模状态，以及如何使用 SageMaker Studio 管理训练实验。

*第十章*，*使用 SageMaker 模型监控在生产中监控机器学习模型*，探讨了数据科学家过去花费太多时间和精力维护和手动管理机器学习管道的过程，这个过程从数据处理、训练和评估开始，以模型托管和持续维护结束。SageMaker Studio 提供了旨在通过**持续集成和持续交付**（**CI/CD**）最佳实践简化此操作的功能。您将学习如何实现 SageMaker 项目、管道和模型注册，这将有助于通过 CI/CD 实现机器学习生命周期的操作化。

*第十一章*，*使用 SageMaker 项目、管道和模型注册表实现机器学习项目的操作化*，讨论了将模型投入生产进行推理并不是生命周期的终点。这只是重要话题的开始：我们如何确保模型在实际生活中按照设计和预期运行？使用 SageMaker Studio 可以轻松监控模型在生产中的表现，尤其是在模型从未见过的数据上。您将学习如何为在 SageMaker 中部署的模型设置模型监控，检测数据漂移和性能漂移，并在实时中可视化推断数据中的特征重要性和偏差。

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件：[`github.com/PacktPublishing/Getting-Started-with-Amazon-SageMaker-Studio`](https://github.com/PacktPublishing/Getting-Started-with-Amazon-SageMaker-Studio)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包可供下载。

https://github.com/PacktPublishing/。查看它们！

# 下载彩色图像

我们还提供了一个包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`static.packt-cdn.com/downloads/9781801070157_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781801070157_ColorImages.pdf)。

# 使用的约定

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词以粗体显示。以下是一个示例：“需要编目记录的两种元数据类型包括**功能**和**技术**。”

小贴士或重要注意事项

看起来像这样。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对这本书的任何方面有疑问，请在邮件主题中提及书名，并给我们发送电子邮件至 customercare@packtpub.com。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式发现我们作品的非法副本，如果您能向我们提供地址或网站名称，我们将不胜感激。请通过 copyright@packt.com 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packt.com](http://packt.com)。

# 分享您的想法

一旦您阅读了《Amazon SageMaker Studio 入门》，我们很乐意听听您的想法！请[点击此处直接进入此书的 Amazon 评论页面](https://packt.link/r/1-801-07015-6)并分享您的反馈。

您的评论对我们和科技社区都至关重要，并将帮助我们确保我们提供高质量的内容。
