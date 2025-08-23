# 前言

欢迎您！如果您对这本书感兴趣，您很可能熟悉 Power BI、**机器学习**（**ML**）和 OpenAI。多年来，Power BI 已从数据可视化工具发展成为一个用户友好的端到端**软件即服务**（**SaaS**）工具套件，用于数据和分析。我开始写这本书的目的是教 Power BI 专业人士了解 Power BI 内置的 ML 工具。我决定不写技术手册，而是遵循微软流行的*Power BI Dashboard in a Day*课程（可在[`aka.ms/diad`](https://aka.ms/diad)找到）的传统，将本书写成一次端到端的旅程，从原始数据开始，以 ML 结束，所有这些都在 SaaS Power BI 工具套件内完成。

在撰写本书的过程中，一种令人惊叹的新技术出现在了舞台上，名为 OpenAI。OpenAI 能够以惊人的方式生成和总结人类语言。本书的用例非常适合将 OpenAI 作为旅程的顶点。

这本书将带您踏上一场数据探险之旅，从**联邦航空管理局**（**FAA**）的真实原始数据开始，回顾类似真实世界项目的需求，使用 Power BI 清洗和整理数据，利用 Power BI ML 进行预测，然后将 OpenAI 集成到用例中。您可以通过参考 Packt GitHub 网站（[`github.com/PacktPublishing/Unleashing-Your-Data-with-Power-BI-Machine-Learning-and-OpenAI/`](https://github.com/PacktPublishing/Unleashing-Your-Data-with-Power-BI-Machine-Learning-and-OpenAI/))在阅读本书时重现整个端到端解决方案。

**商业智能**（**BI**）、ML 和 OpenAI 以不同的方式使用数据，需要不同的数据建模技术和准备。在我的经验中，大多数 Power BI 专业人士对数据的思考方式与 ML 和 AI 专业人士不同。当 BI 专业人士首次转向 ML 时，这些差异可能导致 ML 项目失败。通过一个真实的数据故事示例，本书试图在具有与您在现实世界中可能遇到的类似挑战和要求的应用场景中，教授这些差异。主题是这些技能集的交集，用于现实世界项目，这些项目无缝结合了 BI、ML、AI 和 OpenAI。

如果您正在寻找关于 Power BI ML 或 OpenAI 的技术手册，这本书不适合您。这本书将引导您经历一段英雄之旅，最终达到 ML 和 OpenAI 作为项目的顶点。本书结束时，除了了解如何使用 Power BI ML 和 OpenAI 外，您还将了解如何以能够结合 ML 和 OpenAI 的方式**思考**和**理解**数据项目。即使 Power BI 中的工具在本书编写时发生了变化，您也应该能够将这些学到的经验应用到新工具和未来的挑战中。

我还想在序言中简要讨论 SaaS ML 工具。我经常听到经验丰富的 ML 专业人士敦促对 SaaS ML 工具保持谨慎。我同意，ML 作为一个学科，需要与其他许多数据工具不同的思维方式和独特的技能集。许多因素可能导致 ML 模型返回误导性或偏颇的结果。需要高度准确或错误时可能产生有害后果的 ML 项目应由使用高级 ML 工具的 ML 专业人士处理。

话虽如此，像 Power BI ML 这样的 SaaS 工具在合适的受众中仍然有着强大的地位。对 ML 感兴趣的 Power BI 专业人士可以通过使用 Power BI ML 快速提升技能。通过 Power BI ML，可以快速发现特征、实现简单的预测用例和即兴假设测试，而这一切都只需要低门槛的入门。本书中构建的 ML 模型旨在激发你对这个主题的兴趣，而不是提供构建正确 ML 模型的全面课程。在本书结束时，Power BI 专业人士将理解为什么他们可能会使用 ML，数据需要如何建模以供 ML 使用，以及 ML 如何在数据项目的流程中使用。希望你们中的一些人能从中受到启发，学习更多关于 ML 的知识，并过渡到更高级的 ML 工具和课程。

关于 OpenAI，本书的最后两章提供了 OpenAI 的应用案例，这些案例为与 GitHub 工作坊相关的动手工作坊增添了价值。在 Power BI 解决方案中，使用真实的 FAA 数据生成新的描述并总结事件。本书的意图不是让你成为 OpenAI 或 ML 专家，而是理解 BI、ML、AI 和 OpenAI 的交汇点。我相信，随着企业 SaaS 工具如 Power BI 的使用变得更加容易，这些技能和工具的交汇点将是我们的职业的未来。

# 本书面向的对象

本书非常适合希望在动手工作坊中使用真实世界数据学习 Power BI ML 和 OpenAI 的 BI 专业人士。在阅读本书之前对 Power BI 有实际了解将有所帮助。参加“一天之内掌握 Power BI 仪表板”培训是一个很好的起点，即使你按照自己的节奏跟随链接中的 PDF 文档也是如此。ML 专业人士也可能从 BI、ML、OpenAI 和 AI 的交汇点这个角度找到本书的价值。我预计，在阅读本书后，ML 专业人士将对 BI 项目和 Power BI 工具集有更深入的理解。

# 本书涵盖的内容

*第一章*，*需求、数据建模和规划*，回顾了本书中将要使用的 FAA 野生动物撞击数据，浏览 Power BI 中的数据，审查数据用例，并为未来章节规划数据架构。

*第二章*, *使用 Power Query 准备和摄取数据*，包括数据转换和建模，为 Power BI 数据集和用于构建 ML 模型的查询准备数据。本章的潜在主题是在 BI 的背景下探索数据的同时，也为 ML 做准备。

*第三章*, *使用 Power BI 探索数据并创建语义模型*，开始设计一个用户友好的 BI 数据集的过程，它可以作为报告的基础。命名约定、表关系和自定义度量标准都将被创建，以便您可以在 Power BI 中开始进行数据分析，轻松探索 FAA 数据以发现 ML 模型的特征。

*第四章*, *Power BI 中机器学习的模型数据*，将使用 Power BI 探索数据，以发现可用于构建 ML 模型的可能特征。然后，这些特征将被添加到 Power Query 中的查询中，形成与 Power BI ML 一起使用的数据的基础。

*第五章*, *使用分析和 AI 可视化发现特征*，利用 Power BI 作为分析和数据可视化工具，快速探索 FAA 数据并发现适用于 ML 查询的新特征。使用了各种不同的度量标准和可视化来为您提供多样性。

*第六章*, *使用 R 和 Python 可视化发现新特征*，使用 Power BI 中的 R 和 Python 可视化来发现 ML 查询的附加特征。R 和 Python 可视化提供了一些标准度量标准和可视化难以实现的先进分析功能。

*第七章*, *将数据摄取和转换组件部署到 Power BI 云服务*，将前六章创建的内容移动到 Power BI 云服务。Power BI 数据流、数据集和报告将在本书和研讨会剩余部分移动到云端。

*第八章*, *使用 Power BI 构建机器学习模型*，在 Power BI 中构建 ML 模型。前几章设计的 ML 查询被用来构建三个 ML 模型，用于二元分类、一般分类和回归预测。

*第九章*, *评估训练和测试的 ML 模型*，回顾在 Power BI 中构建的三个 ML 模型。测试结果在预测能力的背景下进行审查和解释。

*第十章*, *迭代 Power BI ML 模型*，讨论基于前一章发现结果的 ML 模型未来计划。选项包括使用 ML 模型、修改查询并重新构建模型等。

*第十一章*，*应用 Power BI ML 模型*，从 FAA 野生动物撞击数据库中引入新的/更近期的数据，并通过 ML 模型运行。将结果与原始测试结果进行比较，并建立了一个流程来对未来的新数据进行评分。

*第十二章*，*OpenAI 的应用案例*，介绍了如何将 OpenAI 与项目和研讨会结合使用。关于 BI 和 OpenAI 交叉讨论产生了将 OpenAI 整合到您计划中的想法。

*第十三章*，*在 Power BI Dataflows 中使用 OpenAI 和 Azure OpenAI*，将 OpenAI API 调用集成到解决方案中。文本生成和摘要直接添加到 Power BI 中。

*第十四章*，*项目回顾与展望*，讨论了本书的关键概念。还回顾了将本书中学到的知识应用到您的职业和未来计划中的建议。

# 为了充分利用这本书

与本书相关的 Packt GitHub 网站提供了用于重现本书中所有内容的全面研讨会脚本和文件。仓库可以在以下链接找到：[`github.com/PacktPublishing/Unleashing-Your-Data-with-Power-BI-Machine-Learning-and-OpenAI/`](https://github.com/PacktPublishing/Unleashing-Your-Data-with-Power-BI-Machine-Learning-and-OpenAI/)。大多数 Power BI 专业人士已经使用的 Power BI 基本工具，以及一个 OpenAI 订阅，将促进 GitHub 仓库的使用。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Power BI Desktop – 2023 年 4 月或更新版本 | Windows |
| Power BI 云服务 | 网络浏览器 |
| Python – 与 Power BI Desktop 兼容的版本 | Windows |
| R – 与 Power BI Desktop 兼容的版本 | Windows |
| OpenAI | 网络浏览器 |
| Azure OpenAI（可选） | 网络浏览器 |

从许可的角度来看，在 Power BI 云服务中需要一个分配给 Power BI Premium per User 或 Power BI Premium 的工作空间。几章还需要 Power BI Pro 许可证才能跟随。对于 OpenAI，需要一个带有 OpenAI 或 Azure 订阅且可以访问 OpenAI 的订阅。

**如果您正在使用本书的数字版，我们建议您亲自输入代码或从本书的 GitHub 仓库（下一节中有一个链接）获取代码。这样做将帮助您避免与代码复制粘贴相关的任何潜在错误。**

# 下载示例代码文件

您可以从 GitHub 下载这本书的示例代码文件：[`github.com/PacktPublishing/Unleashing-Your-Data-with-Power-BI-Machine-Learning-and-OpenAI/`](https://github.com/PacktPublishing/Unleashing-Your-Data-with-Power-BI-Machine-Learning-and-OpenAI/)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“可以使用以下 DAX 表达式创建用于计算主要数据表中行数的度量值：`Incidents = COUNTROWS('Strike` `Reports Fact')`。”

代码块设置如下：

```py
(if [Struck Engine 1] = true then 1 else 0) + 
(if [Struck Engine 2] = true then 1 else 0) + 
(if [Struck Engine 3] = true then 1 else 0) + 
(if [Struck Engine 4] = true then 1 else 0)
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词以**粗体**显示。以下是一个示例：“从**管理**面板中选择**系统信息**。”

小贴士或重要提示

看起来是这样的。

# 联系我们

我们始终欢迎读者的反馈。

**总体反馈**：如果您对本书的任何方面有任何疑问，请通过电子邮件发送至 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告，我们将非常感谢。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过 copyright@packtpub.com 与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读了《利用 Power BI 机器学习和 OpenAI 释放您的数据》，我们非常乐意听到您的想法！请[点击此处直接进入此书的亚马逊评论页面](https://packt.link/r/1-837-63615-X)并分享您的反馈。

您的评论对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

您喜欢在路上阅读，但又无法携带您的印刷书籍到处走？

您的电子书购买是否与您选择的设备不兼容？

别担心，现在，随着每本 Packt 书籍，您都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何地点、任何设备上阅读。直接从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

优惠不会就此结束，您还可以获得独家折扣、时事通讯和每日免费内容的专属访问权限。

按照以下简单步骤获取福利：

1.  扫描下面的二维码或访问以下链接

![](img/B19500_QR_Free_PDF.jpg)

https://packt.link/free-ebook/9781837636150

1.  提交您的购买证明

1.  那就结束了！我们将直接将您的免费 PDF 和其他福利发送到您的邮箱

# 第一部分：数据探索和准备

您的旅程从使用 Power BI 摄取和准备数据开始。在讨论商业智能和机器学习的数据建模之后，您将学习如何连接到数据以用于用例，清理它并检查错误，探索数据以确保引用完整性，然后创建关系数据模型。

本部分包含以下章节：

+   *第一章*, *需求、数据建模和规划*

+   *第二章*, *使用 Power Query 准备和摄取数据*

+   *第三章*, *使用 Power BI 探索数据并创建语义模型*

+   *第四章*, *在 Power BI 中为机器学习建模数据*
