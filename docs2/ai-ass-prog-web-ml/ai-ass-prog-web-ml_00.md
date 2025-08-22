# 前言

# 这本书面向的对象

这本书的目标受众是网络开发、机器学习和数据科学领域的专业人士。你应该是一位拥有至少 1-3 年经验的专业人士。这本书旨在通过展示 AI 助手如何在不同的问题领域中发挥作用来赋予你力量。它描述了整体功能，同时也提供了关于如何有效提示以获得最佳结果的建议。

# 这本书涵盖的内容

*第一章*，*这是一个新的世界，一个拥有 AI 助手的世界，你被邀请了*，探讨了我们是怎样开始使用大型语言模型的，以及它如何构成对许多人（而不仅仅是 IT 工作者）的范式转变。

*第二章*，*提示策略*，解释了本书中使用的策略，即分解问题和一些关于如何有效地提示你选择的 AI 工具的指导原则。

*第三章*，*行业工具：介绍我们的 AI 助手*，是我们解释如何与我们的两个选择的 AI 助手（GitHub Copilot 和 ChatGPT）一起工作的地方，涵盖了从安装到如何开始使用它们的所有内容。

*第四章*，*使用 HTML 和 Copilot 构建我们的应用外观*，重点在于构建我们的电子商务应用的前端（你将在整本书中看到这一叙述）。

*第五章*，*使用 Copilot 和 CSS 美化应用*，是我们继续工作在我们的电子商务应用上，但现在专注于 CSS 并确保外观吸引人的地方。

*第六章*，*使用 JavaScript 添加行为*，是我们使用 JavaScript 为我们的电子商务应用添加行为的地方。

*第七章*，*使用响应式 Web 布局支持多个视口*，是我们解决应用需要为不同设备类型工作的事实的地方，无论是较小的手机屏幕、平板电脑还是桌面屏幕。因此，本章重点介绍了响应式设计。

*第八章*，*使用 Web API 构建后端*，探讨了为了让应用真正工作，它需要有一个后端，包括能够读取和写入数据并持久化的代码。因此，本章重点介绍了为我们的电子商务应用构建 Web API。

*第九章*，*使用 AI 服务增强 Web 应用*，涵盖了训练机器学习模型以及如何通过 Web API 将其暴露出来，以便任何拥有浏览器或其他能够使用 HTTP 协议的客户端的人都可以消费。

*第十章*，*维护现有代码库*，涵盖了大多数开发者如何处理现有代码和维护现有代码库，而不是创建新项目。因此，本章重点介绍了维护代码的各个方面，如处理错误、性能、与测试一起工作等。

*第十一章*，*使用 ChatGPT 进行数据探索*，是我们与审查数据集一起工作并学习如何识别分布、趋势、相关性等方面的见解的地方。

*第十二章*，*使用 ChatGPT 构建分类模型*，探讨了与第十一章相同的审查数据集，这次进行分类和情感分析。

*第十三章*，*使用 ChatGPT 为客户消费构建回归模型*，试图预测客户每年的消费金额，并使用回归创建一个能够进行这种预测的模型。

*第十四章*，*使用 ChatGPT 为 Fashion-MNIST 构建 MLP 模型*，探讨了基于时尚数据集构建 MLP 模型，仍然坚持我们的电子商务主题。

*第十五章*，*使用 ChatGPT 为 CIFAR-10 构建 CNN 模型*，专注于构建 CNN 模型。

*第十六章*，*无监督学习：聚类和 PCA*，专注于聚类和 PCA。

*第十七章*，*使用 Copilot 进行机器学习*，介绍了使用 GitHub Copilot 进行机器学习的过程，并与 ChatGPT 进行对比。

*第十八章*，*使用 Copilot Chat 进行回归*，我们在这里开发了一个回归模型。本章也使用了 GitHub Copilot。

*第十九章*，*使用 Copilot 建议进行回归*，与上一章类似，专注于使用 GitHub Copilot 进行回归。与上一章不同的是，在这里我们使用写作提示的建议作为文本文件中的注释，而不是在聊天界面中编写我们的提示。

*第二十章*，*利用 GitHub Copilot 提高效率*，专注于如何充分利用 GitHub Copilot。如果你想要掌握 GitHub Copilot，这一章是必读的。

*第二十一章*，*软件开发中的代理*，探讨了 AI 领域的下一个趋势，即代理。代理能够通过根据高级目标自主行动来提供更高的协助。如果你对未来的趋势感兴趣，这绝对值得一读。

*第二十二章*，*结论*，通过总结关于与 AI 助手合作的更广泛经验教训来结束本书。

# 要充分利用这本书

如果你已经在每个领域构建了一些项目，而不是作为一个完全的初学者，那么你会从这本书中获得更多。因此，本书侧重于增强你在现有开发工作流程中的能力。如果你对 Web 开发或机器学习完全陌生，我们推荐 Packt 的其他书籍。以下列表提供了推荐：

+   [`www.packtpub.com/en-us/product/html5-web-application-development-by-example-beginners-guide-9781849695947`](https://www.packtpub.com/en-us/product/html5-web-application-development-by-example-beginners-guide-9781849695947)

+   Oliver Theobald 著的《使用 Python 和机器学习解锁 AI 潜力》（[`www.packtpub.com/en-US/product/machine-learning-with-python-9781835461969`](https://www.packtpub.com/en-US/product/machine-learning-with-python-9781835461969)）

本书构建的方式是，首先展示你被推荐的提示，然后是所选 AI 工具的结果。

+   要跟随网络开发章节，我们建议安装 Visual Studio Code。本书中有专门的章节指出如何安装 GitHub Copilot 并利用它。有关 Visual Studio Code 的安装说明请参阅[`code.visualstudio.com/download`](https://code.visualstudio.com/download)。

+   对于机器学习章节，大多数章节使用 ChatGPT，可以通过网页浏览器访问。我们确实建议使用笔记本解决问题，可以通过各种不同的工具查看。有关笔记本设置的更详细说明，请参阅此页面：[`code.visualstudio.com/docs/datascience/jupyter-notebooks`](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

+   要使用 GitHub Copilot，您需要一个 GitHub 账户进行登录。有关 GitHub Copilot 设置过程的页面请参阅[`docs.github.com/en/copilot/quickstart`](https://docs.github.com/en/copilot/quickstart)。

## 下载示例代码文件

本书代码包托管在 GitHub 上，网址为[`github.com/PacktPublishing/AI-Assisted-Software-Development-with-GitHub-Copilot-and-ChatGPT`](https://github.com/PacktPublishing/AI-Assisted-Software-Development-with-GitHub-Copilot-and-ChatGPT)。我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

## 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`packt.link/gbp/9781835086056`](https://packt.link/gbp/9781835086056)。

## 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入以及 X（以前称为 Twitter）用户名。例如：“现在`product.css`已使用上述内容创建，我们可以在 HTML 文件中包含该 CSS 文件。”

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示为这样。例如：“**创建新用户**：应该能够创建新用户。”

警告或重要注意事项看起来像这样。

小技巧和技巧看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：请将邮件发送至 `feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送邮件给我们。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您向我们报告。请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，点击**提交勘误**，并填写表格。

**盗版**：如果您在互联网上以任何形式发现我们作品的非法副本，我们将非常感激您提供位置地址或网站名称。请通过`copyright@packtpub.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[`authors.packtpub.com`](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读了《AI 辅助 Web 和机器学习编程》，我们非常乐意听到您的想法！请[点击此处直接转到此书的亚马逊评论页面](https://packt.link/r/1835086055)并分享您的反馈。

您的评论对我们和科技社区都非常重要，它将帮助我们确保我们提供高质量的内容。

# 下载此书的免费 PDF 副本

感谢您购买此书！

您喜欢在路上阅读，但无法携带您的印刷书籍到处走？

您的电子书购买是否与您选择的设备不兼容？

别担心，现在每购买一本 Packt 书籍，您都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何设备上阅读。直接从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

优惠远不止这些，您还可以获得独家折扣、时事通讯和每日免费内容的每日电子邮件。

按照以下简单步骤获取这些好处：

1.  扫描下面的二维码或访问以下链接：

![](img/B21232_Free_PDF_QR.png)

[`packt.link/free-ebook/9781835086056`](https://packt.link/free-ebook/9781835086056)

1.  提交您的购买证明。

1.  就这些了！我们将直接将您的免费 PDF 和其他好处发送到您的电子邮件。
