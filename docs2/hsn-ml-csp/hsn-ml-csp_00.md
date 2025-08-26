# 前言

在我们以信息技术为主的工作中，机器学习的必要性无处不在，并且被所有开发者、程序员和分析人员所需求。但为什么使用 C#进行机器学习呢？答案是，大多数生产级企业应用程序都是用 C#编写的，使用了诸如 Visual Studio、SQL Server、Unity 和 Microsoft Azure 等工具。

本书通过各种概念、机器学习技术和各种机器学习工具的介绍，为用户提供了一种直观理解，这些工具可以帮助用户将诸如图像和运动检测、贝叶斯直觉、深度学习和信念等智能功能添加到 C# .NET 应用程序中。

使用本书，你将实现监督学习和无监督学习算法，并准备好创建良好的预测模型。你将学习许多技术和算法，从简单的线性回归、决策树和 SVM 到更高级的概念，如人工神经网络、自编码器和强化学习。

在本书结束时，你将培养出机器学习思维，并能够利用 C#工具、技术和包来构建智能、预测和现实世界的商业应用程序。

# 本书面向的对象

本书面向有 C#和.NET 经验的开发者。不需要或假设有其他经验——只需对机器学习、人工智能和深度学习有热情即可。

# 本书涵盖的内容

第一章，*机器学习基础*，介绍了机器学习以及我们在这本书中希望实现的目标。

第二章，*ReflectInsight – 实时监控*，介绍了 ReflectInsight，这是一个强大、灵活且丰富的框架，我们将在整本书中用它来记录和了解我们的算法。

第三章，*贝叶斯直觉 – 解决追尾神秘事件和执行数据分析*，向读者展示了贝叶斯直觉。我们还将检查并解决著名的“追尾”问题，其中我们试图确定谁逃离了事故现场。

第四章，*风险与回报 – 强化学习*，展示了强化学习是如何工作的。

第五章，*模糊逻辑 – 操纵障碍赛*，实现了模糊逻辑来引导我们的自主引导车辆绕过障碍赛道。我们将展示如何加载各种地图，以及我们的自主车辆如何因为做出正确和错误的决定而获得奖励和惩罚。

第六章，*颜色混合 – 自组织映射和弹性神经网络*，通过展示我们如何将随机颜色混合在一起，向读者展示了 SOM（自组织映射）的力量。这为读者提供了一个关于自组织映射的非常简单的直觉。

第七章，*面部和动作检测 – 图像滤波器*，给读者提供了一个非常简单的框架，可以快速将面部和动作检测功能添加到他们的程序中。我们提供了面部和动作检测的多种示例，解释了我们将使用的各种算法，并介绍了我们的专用法国斗牛犬助手 Frenchie！

第八章，*百科全书与神经元 – 旅行商问题*，利用神经元来解决古老的旅行商问题，我们的销售人员被给予了一张必须访问以销售百科全书的房屋地图。为了达到他的目标，他必须选择最短路径，同时只访问每座房子一次，并最终回到起点。

第九章，*我应该接受这份工作吗 – 决策树的应用*，通过两个不同的开源框架向读者介绍决策树。我们将使用决策树来回答问题，*我应该接受这份工作吗？*

第十章，*深度信念 - 深度网络与梦境*，介绍了一个开源框架 SharpRBM。我们将深入到玻尔兹曼和受限玻尔兹曼机的世界。我们将提出并回答问题，*当计算机做梦时，它们会梦到什么？*

第十一章，*微基准测试和激活函数*，向读者介绍了一个开源微基准测试框架 Benchmark.Net。我们将向读者展示如何基准测试代码和函数。我们还将解释什么是激活函数，并展示我们如何对今天使用的许多激活函数进行了微基准测试。读者将获得关于每个函数所需时间的宝贵见解，以及使用浮点数和双精度数之间的计时差异。

第十二章，*C# .NET 中的直观深度学习*，介绍了一个名为 Kelp.Net 的开源框架。这个框架是 C# .NET 开发者可用的最强大的深度学习框架。我们将向读者展示如何使用该框架执行许多操作和测试，并将其与 ReflectInsight 集成，以获取关于我们的深度学习算法的令人难以置信的丰富信息。

第十三章，*量子计算 – 未来*，向读者展示未来，量子计算的世界。

# 为了充分利用这本书

+   您应该熟悉 C#和.NET 的基本开发

+   你应该对机器学习和开源项目有热情

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择“支持”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本解压缩或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Machine-Learning-with-CSharp`](https://github.com/PacktPublishing/Hands-On-Machine-Learning-with-CSharp)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有来自我们丰富的书籍和视频目录中的其他代码包可供选择，请访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnMachineLearningwithCSharp_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/HandsOnMachineLearningwithCSharp_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“我们将使用真值表中的`AddLine`函数来添加这些信息。”

代码块设置如下：

```py
// build the truth tables
UberDriver?.Table?.AddLine(0.85, true);
WitnessSawUberDriver?.Table?.AddLine(0.80, true, true);
WitnessSawUberDriver?.Table?.AddLine(0.20, true, false);
network.Validate();
```

当我们希望您注意代码块的特定部分时，相关的行或项目将以粗体显示：

```py
config.Add(new CsvExporter(CsvSeparator.CurrentCulture,
  new BenchmarkDotNet.Reports.SummaryStyle
{
  PrintUnitsInHeader = true,
  PrintUnitsInContent = false,
  TimeUnit = TimeUnit.Microsecond,
  SizeUnit = BenchmarkDotNet.Columns.SizeUnit.KB
}));
```

**粗体**: 表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“消息详细信息面板显示所选消息的扩展详细信息。”

警告或重要提示如下所示。

技巧和窍门如下所示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**: 请通过电子邮件发送至`feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过电子邮件发送至`questions@packtpub.com`。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您向我们报告。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上以任何形式发现我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**: 如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问[packtpub.com](https://www.packtpub.com/)。
