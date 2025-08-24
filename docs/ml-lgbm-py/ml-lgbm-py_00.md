# 前言

欢迎阅读《使用 LightGBM 和 Python 进行机器学习：开发生产就绪机器学习系统的实践指南》。在这本书中，你将踏上一段丰富的旅程，从机器学习的基础原理到高级的 MLOps 领域。我们探索的基础是 LightGBM，这是一个强大且灵活的梯度提升框架，可以用于各种机器学习挑战。

本书专为任何热衷于利用机器学习的力量将原始数据转化为可操作见解的人量身定制。无论你是渴望动手实践的机器学习新手，还是寻求掌握 LightGBM 复杂性的经验丰富的数据科学家，这里都有适合你的内容。

数字时代为我们提供了丰富的数据宝藏。然而，挑战往往在于从这些数据中提取有意义的见解，并在生产环境中部署可扩展、高效和可靠的模型。本书将引导你克服这些挑战。通过深入研究梯度提升、数据科学生命周期以及生产部署的细微差别，你将获得一套全面的技能，以应对机器学习领域的不断变化。

每一章都是基于实用性设计的。穿插着理论见解的现实案例研究确保你的学习建立在实际应用的基础上。我们专注于 LightGBM，它有时会被更主流的算法所掩盖，提供了一个独特的视角来欣赏和应用梯度提升在各种场景中的应用。

对于那些好奇这本书与众不同的地方，那就是我们的实用方法。我们自豪地超越了对算法或工具的简单解释。相反，我们将优先考虑实际应用、案例研究和现实世界的挑战，确保你不仅是在阅读，而且也在“实践”机器学习。

随着我们穿越章节，请记住，机器学习的世界是广阔且不断演变的。虽然本书内容全面，但它是你在机器学习领域终身学习和探索旅程中的一个基石。在你导航 LightGBM、数据科学、MLOps 等领域时，保持你的思维开放，好奇心旺盛，并准备好动手编码。

# 本书面向的对象

《使用 LightGBM 和 Python 进行机器学习：开发生产就绪机器学习系统的实践指南》专为那些热衷于通过机器学习利用数据力量的广泛读者群体量身定制。本书的目标受众包括以下人群：

+   **机器学习初学者**：刚刚踏入机器学习世界的人会发现这本书非常有帮助。它从基础机器学习原理开始，并使用 LightGBM 介绍梯度提升，对于新入门者来说是一个极好的起点。

+   **经验丰富的数据科学家和机器学习从业者**：对于那些已经熟悉机器学习领域但希望深化对 LightGBM 和/或 MLOps 的了解的人，本书提供了高级见解、技术和实际应用。

+   **希望学习更多数据科学的软件工程师和架构师**：对从数据科学转型或将其集成到他们的应用程序中的软件专业人士来说，本书将非常有价值。本书从理论和实践两方面探讨机器学习，强调动手编码和现实世界应用。

+   **MLOps 工程师和 DevOps 专业人士**：在 MLOps 领域工作或希望了解生产环境中机器学习模型部署、扩展和监控的个人将受益于本书中关于 MLOps、管道和部署策略的章节。

+   **学者和学生**：教授机器学习、数据科学或相关课程的教师以及追求这些领域的学生将发现本书既是一本信息丰富的教科书，也是一本实用的指南。

熟悉 Python 编程是必要的。熟悉 Jupyter 笔记本和 Python 环境是加分项。不需要具备机器学习的前置知识。

事实上，任何对数据有热情、有 Python 编程背景、并渴望使用 LightGBM 探索机器学习多面世界的读者都将发现本书是他们的宝贵资源。

# 本书涵盖内容

*第一章*，*介绍机器学习*，通过软件工程的视角开启我们对机器学习的探索之旅。我们将阐述该领域核心概念，如模型、数据集和各种学习范式，并通过使用决策树的实际示例确保概念的清晰性。

*第二章*，*集成学习 – Bagging 和 Boosting*，深入探讨集成学习，重点关注应用于决策树的 bagging 和 boosting 技术。我们将探讨随机森林、梯度提升决策树等算法，以及更高级的概念，如**Dropout meets Additive Regression Trees**（DART）。

*第三章*，*Python 中 LightGBM 概述*，探讨了 LightGBM，这是一个基于树的学习的高级梯度提升框架。突出其独特的创新和增强集成学习的改进，我们将引导您了解其 Python API。使用 LightGBM 的综合建模示例，结合高级验证和优化技术，为深入数据科学和生产系统机器学习奠定基础。

*第四章*, *比较 LightGBM、XGBoost 和深度学习*，将 LightGBM 与两种主要的表格数据建模方法——XGBoost 和**深度神经网络**（**DNNs**），特别是 TabTransformer 进行比较。我们将通过评估两个数据集来评估每种方法的复杂性、性能和计算成本。本章的精髓是确定 LightGBM 在更广泛的机器学习领域的竞争力，而不是对 XGBoost 或 DNNs 进行深入研究。

*第五章*, *使用 Optuna 进行 LightGBM 参数优化*，专注于关键任务的超参数优化，介绍了 Optuna 框架作为强大的解决方案。本章涵盖了各种优化算法和策略，以修剪超参数空间，并通过一个实际示例指导你如何使用 Optuna 来细化 LightGBM 参数。

*第六章*, *使用 LightGBM 解决现实世界的数据科学问题*，系统地分解了数据科学过程，并将其应用于两个不同的案例研究——一个回归问题和分类问题。本章阐明了数据科学生命周期的每个步骤。你将亲身体验使用 LightGBM 进行建模，并结合全面的理论。本章还作为使用 LightGBM 进行数据科学项目的蓝图。

*第七章*, *使用 LightGBM 和 FLAML 进行 AutoML*，深入探讨了**自动化机器学习**（**AutoML**），强调了其在简化并加速数据工程和模型开发中的重要性。我们将介绍 FLAML，这是一个值得注意的库，它通过高效的超参数算法自动化模型选择和微调。通过一个实际案例研究，你将见证 FLAML 与 LightGBM 的协同作用以及零样本 AutoML 功能的变革性，这使得调优过程变得过时。

*第八章*, *使用 LightGBM 进行机器学习管道和 MLOps*，从建模的复杂性转向生产机器学习的世界。它介绍了机器学习管道，确保一致的数据处理和模型构建，并探讨了 MLOps，这是 DevOps 和 ML 的结合，对于部署弹性机器学习系统至关重要。

*第九章*, *使用 AWS SageMaker 进行 LightGBM MLOps*，引领我们踏上亚马逊 SageMaker 的旅程，这是亚马逊云服务（Amazon Web Services）提供的一套全面的解决方案，用于构建和维护机器学习（ML）解决方案。我们将通过深入研究如偏差检测、模型的可解释性和自动化、可扩展部署的细微差别等高级领域，来深化我们对 ML 管道的理解。

*第十章*，*使用 PostgresML 的 LightGBM 模型*，介绍了 PostgresML，这是一个独特的 MLOps 平台和 PostgreSQL 数据库扩展，它通过 SQL 直接促进 ML 模型开发和部署。这种方法虽然与我们所采用的 scikit-learn 编程风格形成对比，但展示了数据库级 ML 的优势，尤其是在数据移动效率和更快推理方面。

*第十一章*，*使用 LightGBM 进行分布式和基于 GPU 的学习*，深入探讨了训练 LightGBM 模型的广阔领域，利用分布式计算集群和 GPU。通过利用分布式计算，您将了解如何显著加速训练工作负载并管理超出单机内存容量的数据集。

# 要充分利用本书

本书假定您对 Python 编程有一定的了解。本书中的 Python 代码并不复杂，因此即使只理解 Python 的基础知识，也应该足以让您通过大多数代码示例。

在所有章节的实践示例中使用了 Jupyter 笔记本。Jupyter Notebooks 是一个开源工具，允许您创建包含实时代码、可视化和 Markdown 文本的代码笔记本。有关如何开始使用 Jupyter Notebooks 的教程可在[`realpython.com/jupyter-notebook-introduction/`](https://realpython.com/jupyter-notebook-introduction/)和[`plotly.com/python/ipython-notebook-tutorial/`](https://plotly.com/python/ipython-notebook-tutorial/)找到。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Python 3.10 | Windows, macOS, or Linux |
| Anaconda 3 | Windows, macOS, or Linux |
| scikit-learn 1.2.1 | Windows, macOS, or Linux |
| LightGBM 3.3.5 | Windows, macOS, or Linux |
| XGBoost 1.7.4 | Windows, macOS, or Linux |
| Optuna 3.1.1 | Windows, macOS, or Linux |
| FLAML 1.2.3 | Windows, macOS, or Linux |
| FastAPI 0.103.1 | Windows, macOS, or Linux |
| Amazon SageMaker |  |
| Docker 23.0.1 | Windows, macOS, or Linux |
| PostgresML 2.7.0 | Windows, macOS, or Linux |
| Dask 2023.7.1 | Windows, macOS, or Linux |

我们建议在设置自己的环境时使用 Anaconda 进行 Python 环境管理。Anaconda 还捆绑了许多数据科学包，因此您无需单独安装它们。可以从[`www.anaconda.com/download`](https://www.anaconda.com/download)下载 Anaconda。值得注意的是，本书附有 GitHub 仓库，其中包含创建运行本书中代码示例所需环境的 Anaconda 环境文件。

**如果您使用的是本书的电子版，我们建议您亲自输入代码或从本书的 GitHub 仓库（下一节中提供链接）获取代码。这样做将有助于您避免与代码的复制和粘贴相关的任何潜在错误** **。**

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件，网址为[`github.com/PacktPublishing/Practical-Machine-Learning-with-LightGBM-and-Python`](https://github.com/PacktPublishing/Practical-Machine-Learning-with-LightGBM-and-Python)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 使用的约定

本书使用了几个文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“代码几乎与我们的分类示例相同 - 我们使用`DecisionTreeRegressor`作为模型，而不是分类器，并计算`mean_absolute_error`而不是 F1 分数。”

代码块设置如下：

```py
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
model = DecisionTreeRegressor(random_state=157, max_depth=3, min_samples_split=2)
model = model.fit(X_train, y_train)
mean_absolute_error(y_test, model.predict(X_test))
```

任何命令行输入或输出都应如下编写：

```py
conda create -n your_env_name python=3.9
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词以**粗体**显示。以下是一个示例：“因此，**数据准备和清理**是机器学习过程中的关键部分。”

小贴士或重要注意事项

出现在这些块中。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请通过电子邮件发送给我们，邮箱地址为 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**：如果您在互联网上以任何形式发现我们作品的非法副本，我们将非常感激您提供位置地址或网站名称。请通过 copyright@packt.com 与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

读完《使用 LightGBM 和 Python 进行实用机器学习》后，我们非常乐意听到您的想法！请[点击此处直接进入此书的亚马逊评论页面](https://packt.link/r/1-800-56474-0)并分享您的反馈。

您的评论对我们和科技社区都很重要，并将帮助我们确保我们提供高质量的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

你喜欢在路上阅读，但无法携带你的印刷书籍到处走吗？你的电子书购买是否与你的选择设备不兼容？

别担心，现在每本 Packt 书籍都附赠一本无 DRM 的 PDF 版本，无需额外费用。

在任何地方、任何设备上阅读。直接从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

优惠不会就此结束，您还可以获得独家折扣、时事通讯和每天收件箱中的精彩免费内容。

按照以下简单步骤获取福利：

1.  扫描下面的二维码或访问以下链接！![二维码图片](img/B16690_QR_Free_PDF.jpg)

https://packt.link/free-ebook/9781800564749

1.  提交您的购买证明

1.  就这些！我们将直接将您的免费 PDF 和其他福利发送到您的电子邮件

# 第一部分：梯度提升和 LightGBM 基础

在这部分，我们将通过向您介绍机器学习的基本概念来开始我们的探索，这些概念从基本术语到复杂的算法如随机森林。我们将深入探讨集成学习，强调决策树结合时的强大功能，然后转向梯度提升框架，LightGBM。通过 Python 中的实际示例和与 XGBoost 和深度神经网络等技术的比较分析，您将在机器学习领域，特别是 LightGBM 方面获得基础理解和实践能力。

本部分将包括以下章节：

+   *第一章**，介绍机器学习*

+   *第二章**，集成学习 – Bagging 和 Boosting*

+   *第三章**，Python 中 LightGBM 概述*

+   *第四章**，比较 LightGBM、XGBoost 和深度学习*
