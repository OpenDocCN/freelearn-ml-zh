# 前言

我们正处于一个新时代的计算边缘，在这个时代，计算机更多地成为我们的伴侣而不是工具。我们口袋里的设备将很快更好地理解我们的世界和我们自己，这将对我们如何与之互动和使用它们产生深远的影响。

但目前，许多这些令人兴奋的进步都停留在研究人员的实验室里，而不是设计师和开发者的手中，这使得它们对用户可用和可访问。这并不是因为细节被锁起来；相反，在大多数情况下，它们是免费可用的。

这种差距部分是由于我们满足于坚持我们所知道的，让用户做所有的工作，让他们点击按钮。如果其他什么都没有，我希望这本书能让你对现有的东西感到好奇，以及如何用它来创造新的体验，或者改善现有的体验。

在本书的页面上，您将找到一系列示例，帮助您了解深度神经网络的工作原理以及它们如何被应用。

本书专注于一系列模型，以更好地理解图像和照片，特别是研究它们如何在 iOS 平台上进行适配和应用。这种基于图像的模型和 iOS 平台的狭窄关注是有意为之的；我发现图像的视觉性质使得概念更容易可视化，而 iPhone 提供了完美的候选者和实验环境。

因此，当你阅读这本书时，我鼓励你开始思考这些模型的新用途以及你可以创造的新体验。话虽如此，让我们开始吧！

# 本书面向的对象

本书将吸引三个广泛的读者群体。第一群是中级 iOS 开发者，他们对学习和应用**机器学习**（**ML**）感兴趣；对 ML 概念的一些了解可能有益，但并非必需，因为本书涵盖了其中使用的概念和模型背后的直觉。

第二群是有 ML 经验但没有 iOS 开发经验的人，他们正在寻找资源来帮助他们掌握 Core ML；对于这一群体，建议与一本涵盖 iOS 开发基础的书一起阅读。

第三群是经验丰富的 iOS 开发者和 ML 实践者，他们好奇地想看看各种模型在 iOS 平台背景下的应用情况。

# 本书涵盖的内容

第一章，*机器学习简介*，简要介绍了 ML，包括一些对核心概念、问题类型、算法以及创建和使用 ML 模型的一般工作流程的解释。本章通过探讨一些 ML 正在被应用的例子结束。

第二章，*苹果核心 ML 简介*，介绍了 Core ML，讨论了它是什么，它不是什么，以及使用它的一般工作流程。

第三章，*识别世界中的物体*，从零开始构建 Core ML 应用程序。到本章结束时，我们将完成获取模型、将其导入项目以及使用它的整个过程。

第四章，*使用 CNN 进行情感检测*，探讨了计算机更好地理解我们的可能性，特别是我们的情绪。我们首先建立机器学习如何推断你的情绪的直觉，然后通过构建一个执行此操作的应用程序来将其付诸实践。我们还利用这个机会介绍 Vision 框架，并了解它如何补充 Core ML。

第五章，*在世界上定位物体*，不仅限于识别单个物体，而且能够通过物体检测在单个图像中识别和定位多个物体。在理解其工作原理后，我们将将其应用于一个视觉搜索应用程序，该应用程序不仅根据物体进行过滤，还根据物体的组合进行过滤。在本章中，我们还将有机会通过实现自定义层来扩展 Core ML。

第六章，*使用风格迁移创建艺术*，揭示了流行的照片效果应用程序 Prisma 背后的秘密。我们首先讨论如何训练模型区分图像的风格和内容，然后继续构建一个将一种图像的风格应用到另一种图像上的 Prisma 版本。在本章的最后，我们将探讨优化模型的方法。

第七章，*使用 CNN 辅助绘图*，介绍了如何构建一个应用程序，该程序可以使用之前章节中介绍的概念来识别用户的草图。一旦识别出用户试图绘制的对象，我们将探讨如何使用 CNN 的特征向量找到类似的替代品。

第八章，*使用 RNNs 辅助绘图*，在上一章的基础上，探讨了用**卷积神经网络**（**CNN**）替换**循环神经网络**（**RNN**）进行草图分类，从而引入 RNNs 并展示它们如何应用于图像。除了讨论学习序列，我们还将深入了解如何远程下载和编译 Core ML 模型。

第九章，*使用 CNN 进行物体分割*，介绍了构建*ActionShot*摄影应用程序的过程。在这个过程中，我们引入了另一个模型和相关概念，并获得了准备和处理数据的实际经验。

第十章，*《创建 ML 简介*》，是最后一章。我们介绍了 Create ML，这是一个在 Xcode 中使用 Swift 创建和训练 Core ML 模型的框架。到本章结束时，你将了解如何快速创建、训练和部署自定义模型。

# 要充分利用本书

要能够跟随本书中的示例，您需要以下软件：

+   macOS 10.13 或更高版本

+   Xcode 9.2 或更高版本

+   iOS 11.0 或更高版本（设备和模拟器）

对于依赖于 Core ML 2 的示例，您需要以下软件：

+   macOS 10.14

+   Xcode 10.0 测试版

+   iOS 12（设备和模拟器）

建议您使用[`notebooks.azure.com`](https://notebooks.azure.com)（或任何其他 Jupyter 笔记本服务提供商）来使用 Core ML Tools Python 包跟随示例，但那些想要本地运行或训练模型的人需要以下软件：

+   Python 2.7

+   Jupyter Notebooks 1.0

+   TensorFlow 1.0.0 或更高版本

+   NumPy 1.12.1 或更高版本

+   Core ML Tools 0.9（以及 Core ML 2 示例的 2.0 版本）

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择 SUPPORT 标签页。

1.  点击代码下载与勘误。

1.  在搜索框中输入本书的名称，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本解压缩或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Machine-Learning-with-Core-ML`](https://github.com/PacktPublishing/Machine-Learning-with-Core-ML)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/MachineLearningwithCoreML_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/MachineLearningwithCoreML_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“在课程顶部，我们定义了`VideoCaptureDelegate`协议。”

代码块设置如下：

```py
public protocol VideoCaptureDelegate: class {
    func onFrameCaptured(
      videoCapture: VideoCapture, 
      pixelBuffer:CVPixelBuffer?, 
      timestamp:CMTime)
}
```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项目将以粗体显示：

```py
@IBOutlet var previewView:CapturePreviewView!
@IBOutlet var classifiedLabel:UILabel!

let videoCapture : VideoCapture = VideoCapture()
```

**粗体**：表示新术语、重要词汇或屏幕上看到的词汇。例如，菜单或对话框中的文字会以这种方式显示。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要提示如下所示。

技巧和窍门如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：请发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送电子邮件给我们。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告此错误。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现任何形式的我们作品的非法副本，我们将不胜感激，如果您能提供位置地址或网站名称。请通过 `copyright@packtpub.com` 联系我们，并提供材料的链接。

**如果您想成为一名作者**：如果您在某个领域有专业知识，并且对撰写或参与一本书籍感兴趣，请访问 [authors.packtpub.com](http://authors.packtpub.com/).

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买它的网站上留下评论？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packtpub.com](https://www.packtpub.com/).
