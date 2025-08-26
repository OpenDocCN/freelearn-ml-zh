# 前言

自 SQL Server 2016 以来，SQL Server 已支持机器学习功能。之前称为 SQL Server 2016 R 服务，SQL Server 2017 机器学习服务提供两种版本，R 和 Python。本书为数据专业人士、数据库管理员和数据科学家提供实际参考和学习材料，介绍如何使用 SQL Server 和 R 安装、开发、部署、维护和管理数据科学和高级分析解决方案。无论您是 SQL Server 的新手，还是经验丰富的 SQL Server 专业人士，*使用 R 的 SQL Server 机器学习服务实践指南*包含实用的解释、技巧和示例，帮助您充分利用将分析保持靠近数据以实现更好的效率和安全性。

# 本书面向对象

本书面向有一定或无 R 语言经验的数据库分析师、数据科学家和数据库管理员，他们渴望在日常工作或未来的项目中轻松地提供实用的数据科学解决方案。

# 本书涵盖内容

第一章，*R 和 SQL Server 简介*，开启我们在 SQL Server 中的数据科学之旅，从 SQL Server 2016 之前开始，并引领我们到今天 SQL Server R 集成。

第二章，*微软机器学习服务器和 SQL Server 概述*，简要概述了微软机器学习服务器，重点介绍了 SQL Server 机器学习服务，同时探讨了其工作原理和不同版本的 R 环境。这包括对其架构、不同的计算环境、系统间集成方式以及如何实现并行性和负载分配的关键讨论。

第三章，*管理 SQL Server 2017 和 R 的机器学习服务*，涵盖了安装和设置，包括如何使用 PowerShell。它涵盖了探索资源管理器的功能，为用户设置角色和安全权限以使用 SQL Server 机器学习服务与 R 协同工作，处理会话和日志，安装任何缺失或额外的 R 包用于数据分析或预测建模，以及使用`sp_execute_external_script`外部过程的第一步。

第四章，*数据探索与数据可视化*，探讨了 R 语法在数据浏览、分析、整理和可视化以及预测分析中的应用。掌握这些技术对于本章及本书后续章节所涵盖的下一步至关重要。本章介绍了各种用于可视化和预测建模的有用 R 包。此外，读者还将学习如何将 R 与 Power BI、SQL Server Reporting Services（SSRS）和移动报表集成。

第五章，《RevoScaleR 包》，讨论了使用 RevoScaleR 在大型数据集上进行可扩展和分布式统计计算的优势。使用 RevoScaleR 可以提高 CPU 和 RAM 的利用率，并提高性能。本章介绍了 RevoScaleR 在数据准备、描述性统计、统计测试和抽样方面的功能，以及预测建模。

第六章，《预测建模》，专注于帮助那些第一次进入预测建模世界的读者。使用 SQL Server 和 SQL Server 机器学习服务中的 R，读者将学习如何创建预测、执行数据建模、探索 RevoScaleR 和其他包中可用的高级预测算法，以及如何轻松部署模型和解决方案。最后，调用和运行预测并将结果暴露给不同的专有工具（如 Power BI、Excel 和 SSRS）完成了预测建模的世界。

第七章，《R 代码的运营化》，提供了在运营化 R 代码和 R 预测方面的技巧和窍门。读者将了解到稳定和可靠的过程流程对于在生产中将 R 代码、持久数据和预测模型结合在一起的重要性。在本章中，读者将有机会探索采用现有和创建新的 R 代码的方法，然后通过 SQL Server Management Studio (SSMS)和 Visual Studio 等各种现成的客户端工具将其集成到 SQL Server 中。此外，本章还涵盖了读者如何使用 SQL Server Agent 作业、存储过程、CLR 与.NET 以及 PowerShell 来产品化 R 代码。

第八章，《部署、管理和监控包含 R 代码的数据库解决方案》，介绍了在集成 R 代码时如何管理数据库部署的部署和变更控制。本章提供了如何进行解决方案的集成部署以及如何实施持续集成，包括自动化部署和管理版本控制的方法。在这里，读者将学习到监控解决方案、监控代码的有效性和部署后的预测模型的有效方法。

第九章，《使用 R 为数据库管理员提供的机器学习服务》，探讨了数据库管理员日常、每周和每月任务中的监控、性能和故障排除。通过简单的示例说明 R 服务也可以对 SQL Server 中涉及的其他角色有用，本章展示了如何将 R 服务集成到 SQL Server 中，使数据库管理员能够通过将基本的监控活动转变为更有用的可执行预测来获得更多权力。

第十章，*R 和 SQL Server 2016/2017 功能扩展*，介绍了如何将 SQL Server 2016 和 2017 的新特性和 R 服务结合使用，例如利用 R 语言的新 JSON 格式，使用内存 OLTP 技术的新改进来提供几乎实时的分析，结合列存储索引和 R 的新功能，以及如何充分利用它们。它还考虑了如何利用 PolyBase 和 Stretch DB 超越本地，达到混合和云的可能性。最后，查询存储包含执行计划的大量统计数据，而 R 是进行更深入分析的理想工具。

# 为了充分利用本书

为了使用 SQL Server 机器学习服务，并运行本书中的代码示例，您将需要以下软件：

+   SQL Server 2016 和/或 SQL Server 2017 开发者或企业版

+   SQL Server Management Studio (SSMS)

+   R IDE，如 R Studio 或带有 RTVS 扩展的 Visual Studio 2017

+   安装以下扩展的 Visual Studio 2017 社区版：

    +   R Tools for Visual Studio (RTVS)

    +   SQL Server 数据工具 (SSDT)

+   VisualStudio.com 在线账户

本书中的章节在介绍软件时将介绍安装和配置步骤。

# 下载示例代码文件

您可以从 [www.packtpub.com](http://www.packtpub.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并按照屏幕上的说明操作。

文件下载完成后，请确保使用最新版本解压或提取文件夹：

+   Windows 下的 WinRAR/7-Zip

+   Mac 下的 Zipeg/iZip/UnRarX

+   Linux 下的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/SQL-Server-2017-Machine-Learning-Services-with-R`](https://github.com/PacktPublishing/SQL-Server-2017-Machine-Learning-Services-with-R)。我们还有其他来自我们丰富图书和视频目录的代码包，可在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上找到。查看它们！

# 下载彩色图像

我们还提供包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/SQLServer2017MachineLearningServiceswithR_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/SQLServer2017MachineLearningServiceswithR_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“为了计算交叉表——两个（或更多）变量之间的关系——我们将使用两个函数：`rxCrossTabs`和`rxMargins`。”

代码块设置如下：

```py
> df <- data.frame(unlist(var_info)) 
> df 
```

任何命令行输入或输出都应如下编写：

```py
EXECsp_execute_external_script
          @language = N'R'
          ,@script = N'
                      library(RevoScaleR)
    df_sql <- InputDataSet 
                      var_info <- rxGetInfo(df_sql)
                      OutputDataSet <- data.frame(unlist(var_info))'
                      ,@input_data_1 = N'
    SELECT 
                       BusinessEntityID
                      ,[Name]
                      ,SalesPersonID
                      FROM [Sales].[Store]'
```

**粗体**: 表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“你可以始终检查外部脚本启用的`run_value`，如果它设置为 1。”

警告或重要注意事项如下所示。

技巧和窍门如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**: 请通过`feedback@packtpub.com`发送电子邮件，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过`questions@packtpub.com`发送电子邮件给我们。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一点。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**: 如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用过这本书，为什么不在此购买书籍的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问[packtpub.com](https://www.packtpub.com/)。
