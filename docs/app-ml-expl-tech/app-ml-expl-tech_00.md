# 前言

**可解释人工智能（XAI）**是一个新兴领域，旨在将**人工智能（AI）**更接近非技术终端用户。XAI 承诺使**机器学习（ML）**模型透明、可信，并促进 AI 在工业和研究用例中的应用。

这本书结合了工业和学术研究视角的独特混合，旨在获得 XAI 的实用技能。与数据科学、ML、深度学习和 AI 一起工作的 ML/AI 专家将能够利用这本 XAI 实用指南将他们的知识应用于实践，以弥合 AI 与终端用户之间的差距。本书提供了 XAI 的实施和相关方法的动手方法，让您能够迅速投入工作并变得高效。

首先，您将获得对 XAI 及其必要性的概念理解。然后，您将通过利用最先进的方法和框架，获得在 AI/ML 问题解决过程中利用 XAI 的必要实践经验。最后，您将获得将 XAI 推进到下一步并弥合 AI 与终端用户之间现有差距的必要指南。

在本书结束时，您将能够使用 Python 实现 XAI 方法和方案来解决工业问题，解决遇到的关键痛点，并遵循 AI/ML 生命周期的最佳实践。

# 本书面向的对象

这本书是为那些积极从事机器学习及相关领域的科学家、研究人员、工程师、建筑师和管理人员设计的。一般来说，任何对使用 AI 进行问题解决感兴趣的人都能从这本书中受益。建议您具备 Python、ML、深度学习和数据科学的基础知识。这本书非常适合以下角色的读者：

+   数据和 AI 科学家

+   AI/ML 工程师

+   AI/ML 产品经理

+   AI 产品负责人

+   AI/ML 研究人员

+   用户界面和人类计算机交互（HCI）研究人员

一般来说，任何具备 Python 基础知识的 ML 爱好者都将能够阅读、理解和应用从本书中获得的知识。

# 本书涵盖的内容

*第一章*，*可解释性技术的基础概念*，提供了对可解释人工智能的必要了解，并帮助您理解其重要性。本章涵盖了与可解释性技术相关的各种术语和概念，这些术语和概念在本书中经常使用。本章还涵盖了人性化的可解释 ML 系统的关键标准以及评估可解释性技术质量的不同方法。

*第二章*，*模型可解释性方法*讨论了用于解释黑盒模型的多种模型可解释性方法。其中一些方法是模型无关的，一些是模型特定的。一些方法提供全局可解释性，而其他方法提供局部可解释性。本章将向您介绍可用于解释机器学习模型的各种技术，并提供关于选择正确可解释性方法的建议。

*第三章*，*数据为中心的方法*介绍了数据为中心的 XAI 概念。本章涵盖了各种技术，用于从数据属性、数据量、数据一致性、数据纯净度和从底层训练数据集中生成的可操作见解等方面解释机器学习系统的工作原理。

*第四章*，*LIME 用于模型可解释性*涵盖了最受欢迎的 XAI 框架之一，即 LIME 的应用。本章讨论了 LIME 算法背后的直觉以及算法的一些重要特性，这些特性使得生成的解释对人类友好。本章还讨论了 LIME 算法的某些优势和局限性，并包含了一个关于如何将 LIME 应用于分类问题的代码教程。

*第五章*，*使用 LIME 在机器学习中的实际应用*是前一章的扩展，但更专注于 LIME Python 框架在不同类型的数据集上的实际应用，如图像、文本以及结构化表格数据。本章还涵盖了实际代码示例，以展示如何使用 Python LIME 框架获取手头知识。本章还讨论了 LIME 是否适合用于生产级别的机器学习系统。

*第六章*，*使用 SHAP 进行模型可解释性*着重于理解 SHAP Python 框架在模型可解释性方面的重要性。本章涵盖了 Shapley 值和 SHAP 的直观理解，并讨论了如何通过多种可视化和解释方法使用 SHAP 进行模型可解释性。本章还包含了一个使用 SHAP 解释回归模型的代码示例。最后，我们将讨论 SHAP 的关键优势和局限性。

*第七章*，*在机器学习中使用 SHAP 的实际应用*提供了使用 SHAP 与表格结构数据以及非结构化数据（如图像和文本）相结合的必要实践。我们讨论了 SHAP 中可用的不同解释器，用于模型特定和模型无关的解释性。在本章中，我们还应用了 SHAP 来解释线性模型、树集成模型、卷积神经网络模型甚至 Transformer 模型。本章还涵盖了必要的代码教程，以提供使用 Python SHAP 框架进行实际操作的体验。

*第八章*，*使用 TCAV 进行人性化的解释*涵盖了由 Google AI 开发的 TCAV 框架的概念。本章提供了对 TCAV 的概念理解和应用 Python TCAV 框架的实际体验。本章讨论了 TCAV 的关键优势和局限性，并讨论了使用基于概念的解释解决潜在研究问题的有趣想法。

*第九章*，*其他流行的 XAI 框架*介绍了在 Python 中可用的约七个流行 XAI 框架——DALEX、Explainerdashboard、InterpretML、ALIBI、DiCE、ELI5 和 H2O AutoML 解释器。我们讨论了每个框架所支持的解释方法、实际应用以及每个框架的优缺点。本章还提供了一个快速比较指南，以帮助您根据您的用例决定应该选择哪个框架。

*第十章*，*XAI 行业最佳实践*专注于为工业问题设计可解释 AI 系统的最佳实践。在本章中，我们讨论了 XAI 的开放挑战以及考虑开放挑战的必要设计指南，对于可解释 ML 系统。我们还强调了考虑以数据为中心的可解释性、交互式机器学习和面向设计的可解释 AI/ML 系统的规范性见解的重要性。

*第十一章*，*以最终用户为中心的人工智能*介绍了用于设计和开发可解释 AI/ML 系统的以最终用户为中心的人工智能（ENDURANCE）理念。我们讨论了使用 XAI 引导构建可解释 AI/ML 系统以实现最终用户主要目标的重要性。通过本章中提出的一些原则和推荐的最佳实践，我们可以极大地弥合 AI 与最终用户之间的差距！

# 为了充分利用这本书

要运行本书提供的代码教程，您需要一个具有 Python 3.6+的 Jupyter 环境。这可以通过以下两种方式之一实现：

+   通过**Anaconda Navigator**在您的机器上本地安装，或者从头开始使用**pip**安装。

+   使用基于云的环境，例如**Google Colaboratory**、**Kaggle 笔记本**、**Azure 笔记本**或**Amazon SageMaker**。

如果您是 Jupyter 笔记本的新手，可以查看代码仓库中提供的补充信息：[`github.com/PacktPublishing/Applied-Machine-Learning-Explainability-Techniques/blob/main/SupplementaryInfo/CodeSetup.md`](https://github.com/PacktPublishing/Applied-Machine-Learning-Explainability-Techniques/blob/main/SupplementaryInfo/CodeSetup.md)。

您还可以查看[`github.com/PacktPublishing/Applied-Machine-Learning-Explainability-Techniques/blob/main/SupplementaryInfo/PythonPackageInfo.md`](https://github.com/PacktPublishing/Applied-Machine-Learning-Explainability-Techniques/blob/main/SupplementaryInfo/PythonPackageInfo.md)和[`github.com/PacktPublishing/Applied-Machine-Learning-Explainability-Techniques/blob/main/SupplementaryInfo/DatasetInfo.md`](https://github.com/PacktPublishing/Applied-Machine-Learning-Explainability-Techniques/blob/main/SupplementaryInfo/DatasetInfo.md)，以获取关于教程笔记本中使用的 Python 包和数据集的补充信息。

关于安装书中使用的 Python 包的说明，请参阅代码仓库中提供的特定笔记本。如需任何额外帮助，请参阅特定包的原始项目仓库。您可以使用**PyPi** ([`pypi.org/`](https://pypi.org/))搜索特定包并导航到项目的代码仓库。鉴于包经常更改，安装或执行说明可能会不时更改。我们还使用代码仓库提供的补充信息中的*Python 包信息 README 文件*中详细说明的特定版本进行了代码测试。因此，如果后续版本有任何不符合预期的情况，请安装 README 中提到的特定版本。

**如果您正在使用这本书的数字版，我们建议您亲自输入代码或从书的 GitHub 仓库（下一节中有一个链接）获取代码。这样做将帮助您避免与代码复制粘贴相关的任何潜在错误。**

对于没有任何机器学习或数据科学经验的初学者，建议按顺序阅读本书，因为许多重要概念在早期章节中都有充分的详细解释。对于相对新于 XAI 领域的经验丰富的机器学习或数据科学专家，可以快速浏览前三章，以获得对各种术语的清晰概念理解。对于第四到第九章，对于经验丰富的专家来说，任何顺序都行。对于所有级别的从业者，建议在覆盖完所有九章之后，再阅读第十章和第十一章。

关于提供的代码，建议你要么阅读每一章然后运行相应的代码，要么在阅读特定章节的同时运行代码。Jupyter 笔记本中也添加了足够的理论，以帮助你理解笔记本的整体流程。

当你在阅读这本书时，建议你记录下涵盖的重要术语，并尝试思考如何应用所学的概念或框架。在阅读完这本书并浏览完所有 Jupyter 笔记本后，希望你能受到启发，将新获得的知识付诸实践！

# 下载示例代码文件

你可以从 GitHub 下载这本书的示例代码文件[`github.com/PacktPublishing/Applied-Machine-Learning-Explainability-Techniques`](https://github.com/PacktPublishing/Applied-Machine-Learning-Explainability-Techniques)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图和图表的彩色图像的 PDF 文件。你可以从这里下载：[`packt.link/DF7lG`](https://packt.link/DF7lG)。

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“在这个例子中，我们将使用`RegressionExplainer`和`ExplainerDashboard`子模块。”

代码块设置如下：

```py
pdp = PartialDependence(
```

```py
    predict_fn=model.predict_proba,
```

```py
    data=x_train.astype('float').values,
```

```py
    feature_names=list(x_train.columns),
```

```py
    feature_types=feature_types)
```

```py
pdp_global=pdp.explain_global(name='Partial Dependence')
```

当我们希望将你的注意力引到代码块的一个特定部分时，相关的行或项目将以粗体显示：

```py
explainer = shap.Explainer(model, x_test)
```

```py
shap_values = explainer(x_test)
```

```py
shap.plots.waterfall(shap_values[0], max_display = 12,
```

```py
                     show=False)
```

**粗体**：表示新术语、重要词汇或屏幕上看到的词汇。例如，菜单或对话框中的词汇以**粗体**显示。以下是一个示例：“由于这些已知的缺点，寻找一个稳健的**可解释人工智能（XAI）**框架仍在进行中。”

小贴士或重要注意事项

看起来像这样。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果你对这本书的任何方面有疑问，请通过电子邮件发送至 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们非常感谢你向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过 mailto:copyright@packt.com 与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读了《应用机器学习可解释性技术》，我们很乐意听到您的想法！请点击此处直接进入此书的亚马逊评论页面并分享您的反馈。

您的评论对我们和科技社区都很重要，并将帮助我们确保我们提供高质量的内容。
