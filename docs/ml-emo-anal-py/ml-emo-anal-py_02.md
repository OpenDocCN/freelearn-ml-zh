

# 第二章：构建和使用数据集

数据收集和整理过程是模型构建中最重要阶段之一。它也是耗时最长的。通常，数据可以来自许多来源；例如，客户记录、交易数据或股票清单。如今，随着大数据的及时结合、快速、高容量的 SSD（用于存储大数据）和 GPU（用于处理大数据），个人收集、存储和处理数据变得更加容易。

在本章中，你将了解如何找到和访问可用于训练你的模型的现有、现成的数据源。我们还将探讨创建自己的数据集的方法，转换数据集以便它们对你的问题有用，以及我们将看到如何利用非英语数据集。

在本书的剩余部分，我们将使用本章中列出的数据集的一部分来训练和测试一系列分类器。当我们这样做时，我们希望评估分类器在每个数据集上的表现如何——本书的一个重要教训是不同的分类器与不同的数据集配合得很好，为了了解分类器与给定数据集配合得如何，我们需要衡量性能的方法。因此，我们将通过查看评估分类器在数据集上性能的指标来结束本章。

在本章中，我们将讨论以下主题：

+   预制数据源

+   创建自己的数据集

+   其他数据源

+   数据转换

+   非英语数据集

+   评估

# 预制数据源

有许多地方可以找到现成的数据，通常可以免费下载。这些通常被称为**公共数据源**，通常由愿意分享其数据（可能为了宣传，或者吸引他人分享）或作为其他人的数据存储库以使数据易于搜索和访问的公司、机构和组织提供。显然，这些数据源只有当你的需求与数据源相匹配时才有用，但如果确实如此，它可以是一个很好的起点，甚至可以补充你自己的数据。好消息是，这些数据源通常覆盖广泛的领域，因此你很可能找到有用的东西。

我们现在将讨论一些这些公共数据源（不分先后顺序）：

+   **Kaggle**：成立于 2010 年，Kaggle 是谷歌的一部分，根据*维基百科*，是一个“数据科学家和机器学习实践者的在线社区”。Kaggle 有许多功能，但最出名的是其竞赛，任何人（例如个人和组织）都可以发布竞赛（通常是一个数据科学任务），让参与者进入并竞争奖品（有时以现金的形式！）。然而，Kaggle 还允许用户查找和发布数据集。用户可以将他们的数据集上传到 Kaggle，也可以下载其他人发布的数据集。

Kaggle 上的所有内容都是完全免费的，包括数据集，尽管数据集是开源的（开放且免费下载、修改和重用），但对于某些数据集，可能需要参考许可证以确定数据集的使用目的。例如，某些数据集可能不能用于学术出版物或商业目的。用户还被允许上传处理数据集的代码，对数据集发表评论，并对数据集进行点赞，以便其他人知道这是一个可靠且有用的数据集。还有各种其他 **活动概述** 指标，例如数据集何时被下载、数据集被下载了多少次以及数据集被查看了多少次。由于数据集数量众多（在撰写本文时约为 170,000 个），这些指标可以帮助您决定是否值得下载数据集。

网址：[`www.kaggle.com/datasets`](https://www.kaggle.com/datasets)

+   **Hugging Face**：Hugging Face Hub 是一个由社区驱动的数据集集合，涵盖了各种领域和任务——例如，**自然语言处理**（**NLP**）、**计算机视觉**（**CV**）和音频。每个数据集都是一个 Git 仓库，包含下载数据和生成训练、评估和测试分割所需的脚本。大约有 10,000 个数据集可以通过以下标准进行筛选：

    +   任务类别（文本分类、问答、文本生成等）

    +   任务（语言建模、多类分类、语言推断等）

    +   语言（英语、法语、德语等）

    +   多语言性（单语、多语、翻译等）

    +   大小（10-100K、1K-10K、100K-1M 等）

    +   许可证（CC by 4.0、MIT、其他等）

每个数据集页面都包含数据集前 100 行的视图，并且有一个方便的功能，允许您复制代码以加载数据集。一些数据集还包含加载脚本，这也允许您轻松加载数据集。如果数据集不包含此加载脚本，数据通常直接存储在仓库中，格式为 CSV、JSON 或 Parquet。

网址：[`huggingface.co/datasets`](https://huggingface.co/datasets)

+   **TensorFlow Datasets**：TensorFlow Datasets 提供了一组适用于 TensorFlow 以及其他 Python **机器学习**（**ML**）框架的数据集。每个数据集都以类形式呈现，以便构建高效的数据输入管道和用户友好的输入过程。

网址：[`www.tensorflow.org/datasets`](https://www.tensorflow.org/datasets)

+   **Papers With Code**：这是一个包含，正如其名所示，研究论文及其代码实现的网站。在撰写本文时，大约有 7,000 个机器学习数据集可供免费下载。可以使用以下筛选器进行搜索：

    +   模态（文本、图像、视频、音频等）

    +   任务（问答、目标检测、图像分类、文本分类等）

    +   语言（英语、中文、德语、法语等）

根据其“关于”页面，所有数据集均根据 CC BY-SA 许可进行授权，允许任何人在认可创作者（们）的情况下使用数据集。实用的是，每个数据集都列出了使用该数据集的论文、相关基准、代码和类似数据集，并解释了如何从流行的框架（如 TensorFlow）中加载数据集。

Papers With Code 还鼓励用户与社区分享他们的数据集。这个过程相对简单，包括注册、上传数据集，并提供有关数据集的链接和信息（例如，描述、模态、任务、语言等）。

注意

Papers With Code 的“关于”页面声明，尽管核心团队位于 Meta AI Research，但不会与任何 Meta Platforms 产品共享数据。

网址：[`paperswithcode.com/datasets`](https://paperswithcode.com/datasets)

+   **IEEE DataPort**：IEEE DataPort 是一个由电气和电子工程师协会（**IEEE**）创建并拥有的在线数据存储库，**IEEE**是一个电子工程、电气工程和相关学科的行业协会。截至撰写本文时，大约有 6,000 个数据集可用。这些可以通过使用免费文本搜索词（例如，标题、作者或**数字对象标识符**（**DOI**））或以下筛选器进行搜索：

    +   类别（**人工智能**（**AI**）、CV、ML 等）

    +   类型（标准、开放获取）

开放获取数据集允许所有用户免费访问，而访问标准数据集则需要 IEEE 付费订阅。IEEE DataPort 还提供三种选项（标准、开放获取和竞赛）供用户上传他们的数据集。标准和竞赛的上传和访问是免费的；然而，开放获取需要购买开放获取信用。

网址：[`ieee-dataport.org/datasets`](https://ieee-dataport.org/datasets)

+   **谷歌数据集搜索**：谷歌数据集搜索是一个数据集搜索引擎，它具有一个简单的关键词搜索引擎（类似于我们所有人都熟悉和喜爱的谷歌搜索页面），允许用户找到自身托管在互联网上各个存储库（例如 Kaggle）中的数据集。结果可以根据以下标准进行筛选：

    +   最后更新（过去一个月、一年）

    +   下载格式（文本、表格、文档、图像等）

    +   使用权（允许/不允许商业用途）

    +   主题（建筑和城市规划、计算、工程等）

    +   免费/付费

网站声明，该搜索引擎仅在 2020 年从测试版中推出，因此以后可能还会添加更多功能。作为谷歌生态系统的一部分，它还允许用户轻松地将数据集添加到书签中以便稍后返回。正如人们所期望的那样，谷歌提供了关于广泛主题的数据，从移动应用到快餐以及所有介于其间的主题。

网址：[`datasetsearch.research.google.com`](https://datasetsearch.research.google.com)

+   **BigQuery 公共数据集**：BigQuery 是**谷歌云平台**（**GCP**）的一个产品，旨在提供无服务器、成本效益高、高度可扩展的数据仓库功能。因此，BigQuery 用于托管和访问公共数据集，使它们对用户公开，用户可以通过项目将它们集成到他们的应用程序中。尽管数据集是免费的，但用户必须为对数据进行查询付费。然而，在撰写本文时，每月前 1 TB 是免费的。有许多方法可以访问 BigQuery 公共数据集：通过使用谷歌云控制台、通过使用 BigQuery REST API 或通过谷歌分析中心。

URL: [`cloud.google.com/bigquery/public-data`](https://cloud.google.com/bigquery/public-data)

+   **谷歌公共数据探索器**：谷歌公共数据探索器是一个基于网络的工具，它使得以折线图、条形图、图表或地图的形式探索和可视化数据集变得容易。它提供了来自约 135 个组织和大专院校的数据，如世界银行、世界贸易组织（**WTO**）、欧盟统计局和美国人口普查局。用户还可以通过使用谷歌的**数据集发布语言**（**DSPL**）数据格式上传、可视化和共享他们自己的数据。系统真正出色的地方在于图表随时间动画，这使得即使是非科学家也能理解影响并获得洞察。

URL: [`www.google.com/publicdata/directory`](https://www.google.com/publicdata/directory)

+   **UCI 机器学习存储库**：加州大学欧文分校（**UCI**）机器学习存储库于 1987 年由 UCI 的研究生创建，最初是一个 FTP 存档。它是一个免费（无需注册）的约 600 个数据集集合，可供机器学习社区使用。主要网站较为简陋且过时，拥有谷歌驱动的搜索和没有过滤功能，但（在撰写本文时）一个新版本正在测试中，并提供了以下过滤器的搜索能力：

    +   特征（文本、表格、序列、时间序列、图像等）

    +   主题领域（商业、计算机科学、工程、法律等）

    +   相关任务（分类、回归、聚类等）

    +   属性数量（少于 10 个，10-100 个，多于 100 个）

    +   实例数量（少于 10 个，10-100 个，多于 100 个）

    +   属性类型（数值、分类、混合）

存储库中的数据集由不同的作者和组织捐赠，因此每个数据集都有单独的许可要求。网站声明，为了使用这些数据集，应使用引用信息，并检查使用政策和许可。

URL: [`archive.ics.uci.edu`](https://archive.ics.uci.edu)

+   **AWS 开放数据注册处**：AWS 开放数据注册处（简称 Amazon Web Services）是一个集中式存储库，便于查找公开可用的数据集。这些数据集不是由亚马逊提供的，因为它们由政府机构、研究人员、企业和个人拥有。该注册处可用于发现和共享数据集。大约有 330 个数据集可用，这些数据集通过 AWS 数据交换服务（一个提供数千个数据集的在线市场）访问。作为亚马逊的一部分，大部分基础设施都与核心 AWS 服务相关联；例如，数据集可以与 AWS 资源一起使用，并轻松集成到 AWS 基于云的应用程序中。例如，只需几分钟即可配置 Amazon **弹性计算云**（**EC2**）实例并开始处理数据。

URL：[`registry.opendata.aws`](https://registry.opendata.aws)

+   **美国政府开放数据**：2009 年启动，*Data.gov* 由美国总务管理局管理和托管，并由美国政府创建和推出，以提供访问联邦、州和地方数据集。大约有 320,000 个数据集以开放、机器可读的格式提供，同时继续维护隐私和安全，可以通过关键词搜索或按以下标准筛选：

    +   位置

    +   主题（地方政府、气候、能源等）

    +   主题类别（健康、洪水水灾等）

    +   数据集类型（地理空间）

    +   格式（CSV、HTML、XML 等）

    +   组织类型（联邦、州、地方等）

    +   组织（NASA、州、部门等）

    +   发布者

    +   局

这些数据集免费且无限制提供，尽管他们建议非联邦数据可能具有不同的许可方式。

URL：[`data.gov`](https://data.gov)

+   **data.gov.uk**：同样，*data.gov.uk* 网站允许用户查找由英国中央政府、英国地方当局和英国公共机构发布的公共部门、非个人信息。这些数据集通常托管在 AWS 上。大约有 52,000 个数据集可以通过以下标准进行筛选：

    +   发布者（议会）

    +   主题（商业和经济、犯罪和司法、教育等）

    +   格式（CSV、HTML、XLS 等）

这些数据集免费（需要注册），并且似乎许可方式是混合的，其中一些是**开放政府许可**（**OGL**），允许任何人复制、分发或利用数据，而其他则需要**信息自由**（**FOI**）请求以获取数据集。

URL：[`ukdataservice.ac.uk`](https://ukdataservice.ac.uk)

+   **Microsoft Azure Open Datasets**：这是一个用于训练模型的精选数据集存储库。然而，只有大约 50 个数据集，涵盖交通、健康、劳动等领域，以及一些常见的数据集。使用大多数数据集无需付费。

URL：[`azure.microsoft.com/en-us/products/open-datasets/`](https://azure.microsoft.com/en-us/products/open-datasets/)

+   **微软研究开放数据**：这是微软提供的另一组免费数据集，包含对 NLP 和 CV 等领域有用的数据集。同样，这里也只有大约 100 个数据集，可以通过文本搜索或通过以下标准进行筛选：

    +   类别（计算机科学、数学、物理等）

    +   格式（CSV、DOCX、JPG 等）

    +   许可证（Creative Commons、旧版微软研究数据许可协议等）

URL：[`msropendata.com`](https://msropendata.com)

上述列表旨在为不确定去哪里获取数据的人提供一个指示性的、非详尽的指南，并提供了一些组织的数据存储库的示例。还有“存储库的存储库”，其中维护了数据集存储库的列表，这些是开始搜索数据的好地方。这些包括 DataCite Commons Repository Finder（[`repositoryfinder.datacite.org`](https://repositoryfinder.datacite.org)）和 Research Data Repositories 注册处[`re3data.org/`](https://re3data.org/)，它为研究人员提供了现有研究数据存储库的概述。

应注意，一些最常见的流行数据集也容易从 Python 包如 TensorFlow、**scikit-learn**（**sklearn**）和**自然语言工具包**（**NLTK**）中获取。

在本节中，我们看到了如何访问现成的数据源。然而，有时这些数据源是不够的，因此接下来让我们看看我们如何可以创建自己的数据源。

# 创建您自己的数据集

虽然我们已经看到了可以获取数据集的几个来源，但有时有必要使用您自己的数据或使用其他来源的数据来构建自己的数据集。这可能是因为可用的数据集不足以解决我们的问题，这种方法也带来了一些额外的益处，如下所述：

+   创建您自己的数据集可以消除与第三方数据集相关的挑战，这些数据集通常具有许可条款或使用限制。

+   没有费用需要支付（尽管构建数据集会产生成本）。

+   如果数据集是使用您自己的数据创建的，则不存在所有权问题。如果不是这样，那么您有责任考虑所有权问题，并应采取适当的措施。

+   您在使用数据方面拥有完全的所有权和灵活性。

+   在构建数据集的过程中，可以更全面地了解数据。

创建您自己的数据集也伴随着更大的责任；换句话说，如果存在任何错误、问题或偏见，那么将只有一个人要承担责任！

很明显，可以收集许多类型的数据——例如，财务数据、**物联网**（**IoT**）设备的数据和数据库中的数据。然而，由于本书的目的是文本的情感分析，我们将展示一些收集文本数据以构建数据集的方法。

## PDF 文件中的数据

**便携式文档格式**（**PDF**）格式是最受欢迎和广泛使用的数字文件格式之一，用于展示和交换文档。许多组织使用 PDF 格式发布文档、发布说明和其他文档类型，因为文件可以在任何地方阅读，

在任何设备上，只要（免费）工具如 Adobe Acrobat Reader 已安装。因此，这使得 PDF 文件成为寻找数据的好地方。幸运的是，Python 有许多库可以帮助我们从 PDF 文件中提取文本，如下所示：

+   PyPDF4

+   PDFMiner

+   PDFplumber

有很多其他的，但这些似乎是最受欢迎的。由于我们之前的经验，我们将使用 PyPDF4。

首先，我们需要确保 PyPDF4 模块已安装。这是我们运行以实现此目的的命令：

```py
pip install PyPDF4
```

然后，我们需要导入包并设置一个变量，该变量包含我们希望处理的文件名。对于这个例子，从 [`www.jbc.org/article/S0021-9258(19)52451-6/pdf`](https://www.jbc.org/article/S0021-9258(19)52451-6/pdf) 下载了一个样本 PDF 文件：

```py
import PyPDF4file_name = "PIIS0021925819524516.pdf"
```

接下来，我们需要设置一些对象，实际上允许我们读取 PDF 文件，如下所示：

```py
file = open(file_name,'rb')pdf_reader = PyPDF4.PdfFileReader(file)
```

PyPDF4 还可以从 PDF 中提取元数据（关于文件的数据）。以下是这样做的方法：

```py
metadata = pdf_reader.getDocumentInfo()print (f"Title: {metadata.title}")
print (f"Author: {metadata.author}")
print (f"Subject: {metadata.subject}")
```

输出显示了文档的标题、作者和主题（还有其他可用字段）：

```py
Title: PROTEIN MEASUREMENT WITH THE FOLIN PHENOL REAGENTAuthor: Oliver H. Lowry
Subject: Journal of Biological Chemistry, 193 (1951) 265-275\. doi:10.1016/S0021-9258(19)52451-6
```

我们还可以通过执行以下代码来获取文档中页数的数量：

```py
pages = pdf_reader.numPagesprint(f"Pages: {pages}")
```

输出显示了文档中的页数：

```py
Pages: 11
```

我们现在可以遍历每一页，提取文本，并将其写入数据库或文件，如下所示：

```py
page = 0while page < pages:
    pdf_page = pdf_reader.getPage(page)
    print(pdf_page.extractText())
    page+=1
    # write to a database here
```

就这样！我们已经从 PDF 文件中提取了文本，并可以使用它来构建数据集，最终用于训练模型（在清理和预处理之后）。当然，在现实中，我们会将其封装成一个函数，并迭代一个文件夹中的文件来创建合适的数据集，所以让我们来做这件事。

首先，我们需要导入适当的库并设置一个文件夹，其中包含 PDF 文件，如下所示：

```py
import PyPDF4from pathlib import Path
folder = "./"
```

现在，我们可以重构我们最初编写的代码，并以一些方便的可重用函数的形式重新设计它：

```py
def print_metadata(pdf_reader):    # print the meta data
    metadata = pdf_reader.getDocumentInfo()
    print (f"Title: {metadata.title}")
    print (f"Author: {metadata.author}")
    print (f"Subject: {metadata.subject}")
def save_content(pdf_reader):
    # print number of pages in pdf file
    pages = pdf_reader.numPages
    print(f"Pages: {pages}")
    # get content for each page
    page = 0
    while page < pages:
        pdf_page = pdf_reader.getPage(page)
        print(pdf_page.extractText())
        page+=1
        # write each page to a database here
```

注意在 `save_content` 中有一个占位符，你通常会在这里将提取的内容写入数据库。

最后，这是主代码，我们迭代文件夹，并对每个 PDF 文件提取内容：

```py
pathlist = Path(folder).rglob('*.pdf')for file_name in pathlist:
    file_name = str(file_name)
    pdf_file = open(file_name,'rb')
    pdf_reader = PyPDF4.PdfFileReader(pdf_file)
    print (f"File name: {file_name}")
    print_metadata(pdf_reader)
    save_content(pdf_reader)
    pdf_file.close()
```

正如我们所见，从 PDF 文件中提取文本相当简单。现在，让我们看看我们如何从互联网获取数据。

## 网络爬取的数据

现在，网络上公开可用的数据非常多，形式包括（例如）新闻、博客和社交媒体，因此收集（“收割”）并利用这些数据是有意义的。从网站中提取数据的过程称为**网络抓取**，尽管这可以手动完成，但这并不是有效利用时间和资源的方式，尤其是在有大量工具可以帮助自动化这个过程时。进行这一过程的步骤大致如下：

1.  确定一个根 URL（起点）。

1.  下载页面内容。

1.  处理/清理/格式化下载的文本。

1.  保存清理后的文本。

虽然没有硬性规定，但有一些礼仪规则可以防止你的程序被封锁，还有一些规则可以使抓取过程更加容易，这些规则应该遵循：

+   在每次抓取请求之间添加延迟，以防止网站过载

+   在非高峰时段进行抓取

注意，这里重要的是只从允许抓取的数据源中抓取数据，因为未经授权的抓取可能会侵犯服务条款和知识产权，甚至可能产生法律后果。检查元数据也是一个明智的想法，因为它可能提供有关数据是否敏感或私密的指导，数据来源，权限和使用限制。尊重源权限和数据敏感性是负责任和道德网络抓取的重要考虑因素。

让我们开始吧！

首先，我们需要确保 Beautiful Soup 模块已安装。我们可以使用以下代码来完成：

```py
pip install beautifulsoup4
```

注意

为了防止任何意外错误，请确保以下版本已安装：

Beautiful Soup 4.11.2

lxml 4.9.3

我们然后导入所需的库：

```py
import bs4 as bsimport re
import time
from urllib.request import urlopen
```

我们还需要一个 URL 来开始抓取。以下是一个示例：

```py
ROOT_URL = "https://en.wikipedia.org/wiki/Emotion"
```

我们现在需要将网页中有趣、相关的内容与非有用元素（如菜单、页眉和页脚）分开。每个网站都有自己的设计风格和约定，并以独特的方式显示其内容。对于我们选择的网站，我们发现查找三个连续的`<p>`标签可以定位到页面内容部分。对于你要抓取的网站，这种逻辑可能不同。为了找到这些`<p>`标签，我们定义了一个**正则表达式**（**regex**），如下所示：

```py
p = re.compile(r'((<p[^>]*>(.(?!</p>))*.</p>\s*){3,})',    re.DOTALL)
```

我们现在需要请求网站的 HTML 并使用正则表达式提取段落。然后，可以将这些文本清理（例如，移除任何内联 HTML）并保存到数据库中：

```py
def get_url_content(url):    with urlopen(url) as url:
        raw_html = url.read().decode('utf-8')
        for match in p.finditer(raw_html):
            paragraph = match.group(1)
            # clean up, extract HTML and save to database
```

然而，我们可以更进一步。通过从该页面上提取超链接，我们可以让我们的程序继续深入抓取网站。这就是之前关于最佳实践评论应该应用的地方：

```py
def get_url_content(url):    with urlopen(url) as url:
        raw_html = url.read().decode('utf-8')
        # clean up, extract HTML and save to database
        for match in p.finditer(raw_html):
            paragraph = match.group(1)
            soup = bs.BeautifulSoup(paragraph,'lxml')
            for link in soup.findAll('a'):
                new_url = (link.get('href'))
                # add a delay between each scrape
                time.sleep(1)
                get_url_content(new_url)
```

最后，我们需要一些代码来开始抓取：

```py
raw_html = get_url_content(ROOT_URL)
```

为了防止程序陷入循环，应该维护一个已访问的 URL 列表，并在抓取每个 URL 之前进行检查——我们将这个练习留给了读者。

注意

如果您遇到`<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:997)>`错误，您可以使用此链接来解决它：[`stackoverflow.com/a/70495761/5457712`](https://stackoverflow.com/a/70495761/5457712)。

## 来自 RSS 源的数据

**RSS**（代表**真正简单的聚合**）是一种相对较老的技术。曾经，它被用来将所有最新的新闻汇总到网页浏览器中。如今，它不像以前那样受欢迎，但仍被许多人用来保持最新状态。大多数新闻提供商在其网站上提供 RSS 源。

一个 RSS 源通常是一个**可扩展标记语言**（**XML**）文档，它包含指向网页的 URL（如我们所见，可以抓取），完整或摘要文本，以及诸如发布日期和作者姓名的元数据。

让我们看看如何创建新闻标题的数据集。

如同往常，首先我们需要确保所需的模块已安装。`feedparser`是一个 Python 库，它可以处理所有已知格式的源。您可以使用以下命令安装它：

```py
pip install feedparser
```

然后，我们按照如下方式导入它：

```py
import feedparser
```

我们还需要一个用于工作的源 URL。以下是一个示例：

```py
RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml"
```

然后，下载该源并提取相关部分就变得简单了。对于新闻标题，我们预计摘要包含更多信息，因此应该将其保存到数据库中：

```py
def process_feed(rss_url):    feed = feedparser.parse(rss_url)
    # attributes of the feed
    print (feed['feed']['title'])
    print (feed['feed']['link'])
    print (feed.feed.subtitle)
    for post in feed.entries:
        print (post.link)
        print (post.title)
        # save to database
        print (post.summary)
```

最后，我们需要一些代码来启动这个过程：

```py
process_feed(RSS_URL)
```

输出显示了来自每个元素的信息，包括 URL、标题和摘要：

```py
BBC News - Homehttps://www.bbc.co.uk/news/
BBC News - Home
https://www.bbc.co.uk/news/world-asia-63155169?at_medium=RSS&at_campaign=KARANGA
Thailand: Many children among dead in nursery attack
An ex-police officer killed at least 37 people at a childcare centre before killing himself and his family.
https://www.bbc.co.uk/news/world-asia-63158837?at_medium=RSS&at_campaign=KARANGA
Thailand nursery attack: Witnesses describe shocking attack
There was terror and confusion as sleeping children were attacked by the former policeman.
https://www.bbc.co.uk/news/science-environment-63163824?at_medium=RSS&at_campaign=KARANGA
UK defies climate warnings with new oil and gas licences
More than 100 licences are expected to be granted for new fossil fuel exploration in the North Sea.
```

让我们接下来看看如何使用更稳健的技术，即 API，来下载数据。

## 来自 API 的数据

X（以前称为 Twitter）是一个获取文本数据的绝佳地方；它提供了一个易于使用的 API。开始时是免费的，并且有许多 Python 库可用于调用 API。

注意

在撰写本文时，免费的 X（Twitter）API 处于动荡状态，可能不再可能使用`tweepy`API。

由于本书后面我们将处理推文，因此现在学习如何从 Twitter 提取推文是明智的。为此，我们需要一个名为`tweepy`的包。使用以下命令安装`tweepy`：

```py
pip install tweepy
```

接下来，我们需要注册一个账户并生成一些密钥，因此请按照以下步骤操作：

1.  前往[`developer.twitter.com/en`](https://developer.twitter.com/en)并注册一个账户。

1.  前往[`developer.twitter.com/en/portal/projects-and-apps`](https://developer.twitter.com/en/portal/projects-and-apps)。

1.  在**独立应用**部分点击**创建应用**。

1.  给您的应用起一个名字，并记录下**API 密钥**、**API 密钥密钥**和**Bearer 令牌**值。

1.  点击**应用设置**，然后点击**密钥和令牌**选项卡。

1.  在此页面上，点击**访问令牌和密钥**部分的**生成**，并再次记录这些值。

我们现在可以使用这些密钥从 Twitter 获取一些推文！让我们运行以下代码：

```py
import tweepyimport time
BEARER_TOKEN = "YOUR_KEY_HERE"
ACCESS_TOKEN = "YOUR_KEY_HERE"
ACCESS_TOKEN_SECRET = "YOUR_KEY_HERE"
CONSUMER_KEY = "YOUR_KEY_HERE"
CONSUMER_SECRET ="YOUR_KEY_HERE"
```

注意

您必须将`YOUR_KEY_HERE`令牌替换为您自己的密钥。

我们创建了一个类，其中包含一个名为`on_tweet`的子类特殊方法，当从该流接收到推文时会被触发。代码实际上相当简单，看起来像这样：

```py
client = tweepy.Client(BEARER_TOKEN, CONSUMER_KEY,    CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
auth = tweepy.OAuth1UserHandler(CONSUMER_KEY,
    CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
class TwitterStream(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        print(tweet.text)
        time.sleep(0.2)
        # save to database
stream = TwitterStream(bearer_token=BEARER_TOKEN)
```

Tweepy 坚持要添加“规则”来过滤流，因此让我们添加一条规则，说明我们只对包含`#lfc`标签的推文感兴趣：

```py
stream.add_rules(tweepy.StreamRule("#lfc"))print(stream.get_rules())
stream.filter()
Response(data=[StreamRule(value='#lfc', tag=None,
     id='1579970831714844672')], includes={}, errors=[],
    meta={'sent': '2022-10-12T23:02:31.158Z',
    'result_count': 1})
RT @TTTLLLKK: Rangers Fans after losing 1 - 7 (SEVEN) to Liverpool. Sad Song 🎶 #LFC #RFC https://t.co/CvTVEGRBU1
Too bad Liverpool aren't in the Scottish league. Strong enough to definitely finish in the Top 4\. #afc #lfc
RT @LFCphoto: VAR GOAL CHECK
#LFC #RANLIV #UCL #Elliott @MoSalah https://t.co/7A7MUzW0Pa
Nah we getting cooked on Sunday https://t.co/bUhQcFICUg
RT @LFCphoto: #LFC #RANLIV #UCL https://t.co/6DrbZ2b9NT
```

注意

更多关于 Tweepy 规则的信息请见此处：[`developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule`](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule)。

大量使用 X（Twitter）API 可能需要付费套餐。

# 其他数据来源

我们在上一节中列出了一些常用的数据来源。然而，可能还有成千上万的免费数据集可用。你只需要知道在哪里寻找。以下是我们作为情感分析工作的一部分遇到的一些有趣的数据集列表。互联网上可能还有更多：

+   Saif Mohammad 博士是加拿大国家研究委员会（**NRC**）的高级研究科学家。他发表了多篇论文，并且多年来一直积极参与*SemEval*，作为组织者之一。他还发布了多个不同、免费用于研究目的的数据集，这些数据集主要用于竞赛目的。其中许多列在他的网站 http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.xhtml 上，尽管一些数据集在相关的竞赛页面上描述得更好，如下所示：

    +   **情感强度**（**EmoInt**）数据集包含四个情绪的四个数据集（http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.xhtml）。

    +   **计算方法在主观性、情感和社会媒体分析研讨会**（**WASSA**）数据集包含总共 3,960 条英文推文，每条推文都标注了愤怒、恐惧、喜悦和悲伤等情绪。每条推文还有一个介于 0 到 1 之间的实数值分数，表示说话者感受到的情绪程度或强度（[`wt-public.emm4u.eu/wassa2017/`](https://wt-public.emm4u.eu/wassa2017/))）。

    +   *SemEval*（Mohammad 等人，2018）是一个年度竞赛，来自世界各地的研究团队在这个竞赛中开发系统来对数据集进行分类。具体任务每年都有所不同。自 1998 年以来，它一直断断续续地进行，但自 2012 年以来，它已成为年度活动。由此竞赛产生了许多数据集，如下所示：

        +   **2018 任务 E-c**：一个包含被分类为“中性或无情感”或为 1 个或更多 11 个给定情绪的数据集，这些情绪最能代表推文作者的内心状态。

        +   **2018 任务 EI-reg**：一个包含标注了情感（愤怒、恐惧、快乐、悲伤）和强度为 0 到 1 之间的实值分数的数据集，分数为 1 表示推断出的情感量最高，分数为 0 表示推断出的情感量最低。作者指出，这些分数没有固有的意义；它们仅用作一种机制，表明分数较高的实例对应着比分数较低的实例更大的情感程度。

        +   **2018 任务 EI-oc**：一个包含标注了情感（愤怒、恐惧、快乐、悲伤）和四个情感强度等级（最能代表推文者心理状态）的数据集。这些数据集都可以在 https://competitions.codalab.org/competitions/17751 找到。

    +   在竞赛的这些年里，似乎也有许多用于情感标签的数据集，如下所示：

        +   [`alt.qcri.org/semeval2014/task9/`](https://alt.qcri.org/semeval2014/task9/)

        +   [`alt.qcri.org/semeval2015/task10/index.php?id=data-and-tools`](https://alt.qcri.org/semeval2015/task10/index.php?id=data-and-tools)

        +   [`alt.qcri.org/semeval2017/task4/`](https://alt.qcri.org/semeval2017/task4/)

    +   Hashtag 情感语料库数据集包含了带有情感词标签的推文（[`saifmohammad.com/WebDocs/Jan9-2012-tweets-clean.txt.zip`](http://saifmohammad.com/WebDocs/Jan9-2012-tweets-clean.txt.zip)）。

+   **国际情绪前因与反应调查**（**ISEAR**）数据集包含了学生受访者经历情绪（快乐、恐惧、愤怒、悲伤、厌恶、羞耻和内疚）的情况报告（[`www.unige.ch/cisa/research/materials-and-online-research/research-material/`](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/))）。

+   一个用于情感（负面、中性、正面）标签的流行数据集（[`data.mendeley.com/datasets/z9zw7nt5h2`](https://data.mendeley.com/datasets/z9zw7nt5h2)）。

+   一个标注了六种情绪（愤怒、恐惧、快乐、爱情、悲伤、惊讶）的数据集（[`github.com/dair-ai/emotion_dataset`](https://github.com/dair-ai/emotion_dataset)）。

+   **情境化情感表示用于情感识别**（**CARER**）是一个通过噪声标签收集的情感数据集，标注了六种情绪（愤怒、恐惧、快乐、爱情、悲伤和惊讶）([`paperswithcode.com/dataset/emotion`](https://paperswithcode.com/dataset/emotion))）。

注意

在使用这些数据集时，考虑数据隐私和伦理问题至关重要。

我们已经看到了如何访问现成的数据源以及如何创建自己的数据源。然而，有时这些数据源虽然很好，但并不完全符合我们的需求。让我们看看我们如何解决这个问题。

# 数据转换

尽管我们已经看到有许多情感和情绪数据集可用，但很少有一个数据集完全符合所有精确要求。然而，有方法可以解决这个问题。

我们已经看到一些数据集被标记为情感，一些被标记为情绪，还有一些被标记为更奇特的事物，如效价，这些似乎都不符合我们所寻找的情感分析问题。然而，在某些情况下，仍然有可能重新利用并使用这些数据集。例如，如果一个数据集包含情绪，我们可以通过假设“愤怒”情绪是负面情绪，“快乐”情绪是正面情绪，将这个数据集转换成一个情感数据集。然后，需要一定程度的主体性和对单个数据集的手动分析，以确定哪些情绪可以构成对中性情绪的良好替代。根据我们的经验，这通常不是一件简单的事情。

**数据转换**是指对数据集应用更改的过程，以使数据集更适合您的目的。这可能包括添加数据、删除数据或任何使数据更有用的过程。让我们从“其他数据源”部分考虑一些例子，看看我们如何可以重新利用它们。

如前所述，EI-reg 数据集包含被标注为情绪（愤怒、恐惧、快乐、悲伤）和强度（介于 0 到 1 之间的分数）的推文。我们可以合理地猜测，分数在 0.5 左右以及以下的推文不太可能是高度情绪化的推文，因此可以移除分数低于 0.5 的推文，剩下的推文可以用来创建一个规模较小但可能更有用的数据集。

EI-oc 数据集也包含被标注为情绪（愤怒、恐惧、快乐、悲伤）和情绪强度的四个有序类别，这些类别最好地代表了推文作者的内心状态，如下所示：

+   *0*：无法推断出任何情绪

+   *1*：可以推断出少量情绪

+   *2*：可以推断出中等程度的情绪

+   *3*：可以推断出大量情绪

再次，我们可以合理地猜测，通过移除分数低于 3 的推文，我们将得到一个更适合我们需求的数据集。

这些是关于如何重新利用数据集以提取强烈情绪数据以创建新数据集的相对简单想法。然而，任何好的数据集都需要平衡，因此现在让我们回到创建中性推文的问题，看看这可能如何实现。以下示例假设数据集已经下载并可用；您可以从这里下载：[`www.saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/English/EI-reg-En-train.zip`](http://www.saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/EI-reg/English/EI-reg-En-train.zip)。

注意

您可能会遇到**EI-reg-En-train.zip 无法安全下载**的错误。在这种情况下，只需点击**保留**选项。

代码如下所示：

```py
import pandas as pddata_file_path = "EI-reg-En-anger-train.txt"
df = pd.read_csv(data_file_path, sep='\t')
# drop rows where the emotion is not strong
df[df['Intensity Score'] <= 0.2]
```

快速浏览结果，可以看到以下推文：

| **ID** | **推文** | **情感**维度 | **强度**分数 |
| --- | --- | --- | --- |
| 2017-En-40665 | 我非常喜欢“fret”这个词，我觉得自己像在天堂一样 | 愤怒 | 0.127 |
| 2017-En-11066 | 我不喜欢菠萝，我只在披萨上吃它们，它们在煮熟后就会失去刺痛感。 | 愤怒 | 0.192 |
| 2017-En-41007 | 很遗憾看到你们皱眉，但我想看到他微笑 💕✨ | 愤怒 | 0.188 |

图 2.1 – 样本愤怒推文及其强度分数 < 0.2

我们可以看到，有一些推文，尽管它们的愤怒强度较低，但在其他情感上强度较高，因此不是中性的，因此我们需要某种方法来删除这些推文。显然，是单词本身告诉我们推文是中性还是非中性（例如，“爱”或“恨”）。互联网上有许多情感词列表免费可用；下载一个（[`www.ndapandas.org/wp-content/uploads/archive/Documents/News/FeelingsWordList.pdf`](https://www.ndapandas.org/wp-content/uploads/archive/Documents/News/FeelingsWordList.pdf)）并打印结果，显示以下输出：

```py
AbusedAdmired
Afraid
.
.
.
Lonely
Loved
Mad
.
.
.
```

然后，作为初步尝试，删除包含这些单词中的任何一项的推文是件微不足道的事情。然而，我们可以看到，虽然推文 2017-En-40665 说“爱”，但情感词列表说“Loved"。这很成问题，因为这会阻止推文被标记为非中性。为了解决这个问题，我们只需对推文和情感词列表进行词干提取或词元化（详见*第五章**，情感词典和向量空间模型*以获取更多详细信息）如下：

```py
stemmer = PorterStemmer()def stem(sentence):
    res = (" ".join([stemmer.stem(i) for i in
        sentence.split()]))
    return res
# create some new columns
emotion_words['word_stemmed'] = emotion_words['word']
df['Tweet_stemmed'] = df['Tweet']
# stem the tweets and the emotions words list
df['Tweet_stemmed'] = df['Tweet_stemmed'].apply(stem)
emotion_words['word_stemmed'] = emotion_words[
    'word_stemmed'].apply(stem)
# remove tweets that contain an emotion word
res = []
dropped = []
for _, t_row in df.iterrows():
    tweet = t_row["Tweet_stemmed"]
    add = True
    for _, e_row in emotion_words.iterrows():
        emotion_word = e_row["word_stemmed"]
        if emotion_word in tweet:
            add = False
            break
    if add:
        res.append(t_row["Tweet"])
    else:
        dropped.append(t_row["Tweet"])
```

当我们查看结果时，我们看到的是：

```py
@Kristiann1125 lol wow i was gonna say really?! haha have you seen chris or nah? you dont even snap me anymore dude!And Republicans, you, namely Graham, Flake, Sasse and others are not safe from my wrath, hence that Hillary Hiney-Kissing ad I saw about you
@leepg \n\nLike a rabid dog I pulled out the backs of my cupboards looking for a bakewell..Found a french fancie &amp; a mini batternburg #Winner!
```

这些是被删除的推文：

```py
@xandraaa5 @amayaallyn6 shut up hashtags are cool #offendedit makes me so fucking irate jesus. nobody is calling ppl who like hajime abusive stop with the strawmen lmao
Lol Adam the Bull with his fake outrage...
```

大体来说，这种方法虽然简单，但已经做得相当不错。然而，还有一些情况是漏网之鱼——例如，当标签中包含情感词时：

```py
Ever put your fist through your laptops screen? If so its time for a new one lmao #rage #anger #hp
```

更新代码以捕捉这种情况很容易。我们将把这留给你作为练习，但必须仔细检查结果，以捕捉此类边缘情况。

到目前为止，我们只看了英语数据集。然而，有许多非英语数据集可用。当所需的数据集不存在或存在但可能数据不足，或者不平衡，因此需要扩充时，这些数据集可能很有用。这就是我们可以转向非英语数据集的地方。

# 非英语数据集

通常，找到用于训练你模型的数据库是项目中最具挑战性的部分。可能会有这样的情况，数据库是可用的，但它使用的是不同的语言——这就是翻译可以用来使该数据库对你的任务有用的地方。如以下列出，有几种不同的方法可以翻译数据库：

+   询问你认识的语言专家

+   招募专业的翻译公司

+   使用在线翻译服务（例如 Google 翻译），无论是通过 GUI 还是通过 API

显然，前两种是首选选项；然而，它们在努力、时间和金钱方面都有相应的成本。第三种选项也是一个不错的选择，特别是如果需要翻译的数据量很大。然而，这个选项应该谨慎使用，因为（正如我们将看到的）翻译服务有细微差别，每种服务都可能产生不同的结果。

有很多不同的翻译服务可供选择（例如 Google、Bing、Yandex 等）以及很多 Python 包可以用来利用这些服务（例如 TextBlob、Googletrans、translate-api 等）。我们将使用**translate-api**，因为它易于安装，支持很多翻译服务，并且只需几行代码就可以开始翻译。考虑以下推文：

心跳加速

首先，我们需要安装这个包，如下所示：

```py
pip install translators
```

代码本身看似简单：

```py
import translators as tsphrase = 'توتر فز قلبي'
FROM_LANG = 'ar'
TO_LANG = 'en'
text = ts.translate_text(phrase, translator='google',
    from_language=FROM_LANG , to_language=TO_LANG)
print (res)
```

这将产生以下输出：

```py
Twitter win my heart
```

让我们看看当我们尝试其他翻译提供商时会发生什么：

```py
ts.baidutext = ts.translate_text(phrase, translator='bing',
    from_language=FROM_LANG ,
    to_language=TO_LANG)print (res)
```

输出如下：

```py
Tension broke my heart
```

我们可以看到结果并不相同！这就是为什么通常一个好的做法是让母语人士验证翻译服务的结果。

然而，可以看出，前面的代码可以很容易地将整个数据集翻译成英文，从而为我们的模型生成新的数据。

# 评估

一旦我们选择了一个数据集，我们就会想用它来训练一个分类器，看看这个分类器表现如何。假设我们有一个存储在`dataset`变量中的数据集和一个存储在`classifier`变量中的分类器。我们必须做的第一件事是将数据集分成两部分——一部分存储在`training`中，用于训练分类器，另一部分存储在`testing`中，用于测试它。我们这样做分割有两个明显的约束，如下所述：

+   `training`和`testing`必须是互斥的。**这是至关重要的**。如果它们不是互斥的，那么将有一个简单的分类器能够得到 100%的正确率——即仅仅记住你看到的所有例子。即使忽略这个简单的情况，分类器在训练过的数据集上的表现通常会比在未见过的案例上更好，但当分类器在实际应用中部署时，绝大多数案例都是它未知的，因此测试应该总是在未见过的数据上进行。

+   数据收集的方式往往会引入偏差。以一个简单的例子来说，推文通常按时间顺序收集——也就是说，在同一天写的推文在数据集中通常会一起出现。但如果在收集过程的第 1 天到第 90 天，每个人都对某个话题非常高兴，而在第 91 天到第 100 天又非常不高兴，那么数据集开始处会有很多高兴的推文，而结尾处会有很多不高兴的推文。如果我们选择数据集的前 90 天用于训练，最后一天用于测试，那么我们的分类器可能会高估测试集中推文是高兴的可能性。

因此，在将数据分成训练和测试部分之前随机化数据是有意义的。然而，如果我们这样做，我们应该确保我们总是以相同的方式进行随机化；否则，我们将无法比较同一数据集上的不同分类器，因为随机化意味着它们在不同的数据集上被训练和测试。

需要考虑的第三个问题是。如果我们没有太多数据，我们希望尽可能多地使用它进行训练——一般来说，我们拥有的训练数据越多，我们的分类器表现越好，所以我们不希望浪费太多数据在测试上。另一方面，如果我们没有太多数据用于测试，那么我们的测试结果就不能保证可靠。

解决方案是使用 **交叉验证**（有时称为 **X-fold 验证**）。我们构建一系列 **折**，其中每个折将数据分成 N-T 个用于训练的数据点和 T 个用于测试的数据点（N 是整个数据集的大小，而 T 是我们想要用于测试的数据点的数量）。如果我们这样做 N/T 次，在每个折中使用不同的数据集进行测试，我们最终将使用所有数据点进行测试。

T 应该有多大？如果我们使用非常小的测试部分，我们将为每个折提供尽可能多的训练数据，但我们需要进行很多轮的训练和测试。如果我们使用大的测试部分，那么每个折的训练数据将更少，但我们不需要进行那么多轮。假设我们有一个包含 1000K 个数据点的数据集。如果我们将其分成两个折，使用 500K 个数据点用于训练和 500K 个数据点用于测试，我们将有相当大的训练集（我们将在本书剩余部分查看的大多数分类器的分数在训练和测试之前就会平坦化）。

假设有 500K 个训练数据点）我们将使用所有数据用于测试（第一折用一半，第二折用另一半），我们只需要进行两轮训练和测试。另一方面，如果我们只有 1,000 个数据点，将其分成两个折，每个折有 500 个用于训练和 500 个用于测试的数据点，将给我们非常小的训练集。最好将其分成 10 个折，每个折有 900 个用于训练和 100 个用于测试，或者甚至分成 100 个折，每个折有 990 个用于训练和 10 个用于测试，或者甚至分成 1,000 个折，每个折有 999 个用于训练和 1 个用于测试。无论我们选择哪种方式，我们都会将每个数据点用于测试一次，但如果我们使用小的测试集，我们将在进行更多轮训练和测试的情况下最大化训练集的大小。

以下 `makeSplits` 函数将数据集分成 *f* 个折，每个折包含 *N*(1-1/f)* 个用于训练的数据点和 *N/f* 个用于测试的数据点，并对每个折应用一个分类器：

```py
def makeSplits(dataset, classifier, f):    scores = []
    N = len(dataset)/f
    # Randomize the dataset, but *make sure that you always shuffle
    # it the same way*
    random.seed(0)
    random.shuffle(pruned)
    for i in range(f):
        # test is everything from i*N to (i+1)*N,
        # train is everything else
        test = dataset[i*N:(i+1)*N]
        train, = dataset[:i*N]+dataset[(i+1)*N:]
        clsf = classifier.train(training)
        score = clsf.test(test)
        scores.append(score)
    return scores
```

在本书的剩余部分，对于少于 20K 数据点的数据集，我们使用 10 个折，对于多于 20K 数据点的数据集，我们使用 5 个折。如果我们有 20K 数据点，使用 5 个折将给我们每个折 16K 个用于训练的点，这通常足以得到一个合理的模型，因此，由于我们最终会使用每个数据点进行测试，无论我们使用多少或多少个折，这似乎是一个合理的折衷方案。

在`makeSplits`的前一个定义中，`classifier.train(training)`训练了分类器，而`clsf.test(test)`返回了一个分数。对于这两个任务，我们需要知道数据集中每个点应该属于哪个类别——我们需要一组**黄金标准**值。没有一组黄金标准值，训练阶段不知道它应该学习什么，测试阶段不知道分类器是否返回了正确的结果。因此，我们将假设每个数据点都有一个黄金标准标签：我们如何使用这些标签来评估分类器的性能？

考虑一个需要将每个数据点分配到一组类别 C1, C2, …, Cn 中的一个分类器，并让`tweet.GS`和`tweet.predicted`是黄金标准值和分类器分配的标签。有三种可能性：`tweet.GS`和`tweet.predicted`相同，它们不同，以及分类器简单地未能为`tweet.predicted`分配值。如果分类器总是分配一个值，那么计算它的准确率就足够简单，因为这只是分类器正确处理所有案例的比例：

```py
def accuracy(dataset):    return sum(x.GS==x.predicted for x in
        dataset)/len(dataset)
```

然而，大多数分类器允许第三种选择——也就是说，在某些情况下，它们可以简单地不提供答案。这是允许的合理事情：如果你问一个人他们的一个朋友是否快乐，他们可能会说“是”，他们可能会说“否”，但他们可能会非常合理地说他们不知道。如果你没有任何证据告诉你某物是否属于某个给定的类别，唯一合理的事情就是说你不知道。

这对机器学习算法和人类来说都是正确的。对于分类器来说，总是说它不确定比说错事情要好。然而，这确实使得比较两个分类器的任务不那么直接。一个分类器在 95%的情况下说“我不知道”，但在剩下的 5%的情况下正确，它比一个从不承认它不确定但在 85%的情况下正确要好，还是更差？

对于这个问题没有唯一的答案。假设给出错误答案会带来灾难性的后果——例如，如果你认为某人可能服用了某种毒药，但唯一已知的解毒剂对未服用的人是有害的。在这种情况下，分类器经常说它不知道，但一旦说出了什么，总是正确的，比总是做出决定但经常错误的分类器要好。另一方面，如果给出错误答案实际上并不重要——例如，如果你在考虑给某人开他汀类药物，因为你认为他们可能患有心脏病，那么总是做出决定但有时错误的分类器会更好。如果你可能患有心脏病，那么服用他汀类药物是个好主意，而且在你不需要的时候服用不太可能导致问题。因此，我们必须能够灵活地组合分数，以适应不同的情况。

我们首先定义了四个有用的参数，如下所示：

+   **真阳性**（**TP**）：分类器预测一条推文属于类别 C，而黄金标准也说它确实属于 C。

+   **假阳性**（**FP**）：分类器预测一条推文属于 C，而黄金标准说它不属于 C。

+   **假阴性**（**FN**）：分类器没有做出预测，但黄金标准说它确实属于某个类别的次数（所以它不正确，但也不是真的错误）。

+   **真阴性**（**TN**）：分类器没有做出预测，而黄金标准也说所讨论的项目不属于任何可用的类别。在每种项目恰好属于一个类别的案例中，这个组总是空的，并且通常不用于分类器的评估。

给定这些参数，我们可以提供以下指标：

+   **精度**：分类器在做出预测时正确率的频率——![<mml:math  ><mml:mrow><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi></mml:mrow><mml:mo>/</mml:mo><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>P</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:mrow></mml:math>](img/1.png).

+   **召回率**：在应该做出预测的案例中，有多少案例预测正确——![<mml:math  ><mml:mrow><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi></mml:mrow><mml:mo>/</mml:mo><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>N</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:mrow></mml:math>](img/2.png)![<mml:math  ><mml:mrow><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi></mml:mrow><mml:mo>/</mml:mo><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>N</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:mrow></mml:math>](img/3.png)。

+   **F 度量**：如前所述，有时精确度更重要（在解毒剂可能致命的情况下诊断中毒），而有时召回率更重要（为可能患有心脏病的人开处方他汀类药物）。F 度量定义为![<mml:math  ><mml:mrow><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>P</mml:mi><mml:mo>×</mml:mo><mml:mi>R</mml:mi></mml:mrow></mml:mfenced></mml:mrow><mml:mo>/</mml:mo><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>a</mml:mi><mml:mo>×</mml:mo><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mfenced separators="|"><mml:mrow><mml:mn>1</mml:mn><mml:mo>-</mml:mo><mml:mi>a</mml:mi></mml:mrow></mml:mfenced><mml:mo>×</mml:mo><mml:mi>R</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:mrow></mml:math>](img/4.png)对于 0 到 1 之间的某个*a*，允许我们平衡这两个度量：如果精确度比召回率更重要，则选择*a*大于 0.5；如果召回率比精确度更重要，则选择*a*小于 0.5。将*a*设置为 0.5 提供了一个中间点，通常称为 F1 度量。

+   **宏 F1** 和 **微 F1**：当一个任务涉及多个类别时，有两种计算 F1 的方法。你可以取所有分类器做出决策的案例，并使用这些案例来计算 P 和 R，从而计算 F1，或者你可以为每个类别计算 F1，然后对所有类别取平均值。第一种方法称为微 F1，第二种方法称为宏 F1。如果一个类别包含的案例比其他类别多得多，那么微 F1 可能会误导。假设有 99%的患有中毒症状的人实际上患有消化不良。那么，将所有患有中毒症状的人实际上患有消化不良的案例分类的分类器将会有 P = 0.99 和 R = 0.99 的整体得分，微 F1 得分为 0.99。但这意味着没有人会得到解毒剂的治疗：对于消化不良，个体 P 和 R 得分将是 0.99 和 1，对于中毒，得分是 0.0，0.0，个体 F1 得分是 0.99 和 0，平均下来是 0.495。一般来说，微 F1 更重视多数类别，并对少数案例的得分提供高估。

+   **Jaccard 度量**提供了一种结合 TP、FP 和 FN 的替代方法，使用![<mml:math  ><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>N</mml:mi></mml:mrow></mml:mfenced></mml:math>](img/5.png)![<mml:math  ><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>N</mml:mi></mml:mrow></mml:mfenced></mml:math>](img/6.png)。鉴于简单的 F1 很容易证明与![<mml:math  ><mml:mrow><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi></mml:mrow><mml:mo>/</mml:mo><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mn>0.5</mml:mn><mml:mo>×</mml:mo><mml:mfenced separators="|"><mml:mrow><mml:mi>F</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>N</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:mfenced></mml:mrow></mml:mrow></mml:math>](img/7.png)![<mml:math  ><mml:mrow><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi></mml:mrow><mml:mo>/</mml:mo><mml:mrow><mml:mfenced separators="|"><mml:mrow><mml:mi>T</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mn>0.5</mml:mn><mml:mo>×</mml:mo><mml:mfenced separators="|"><mml:mrow><mml:mi>F</mml:mi><mml:mi>P</mml:mi><mml:mo>+</mml:mo><mml:mi>F</mml:mi><mml:mi>N</mml:mi></mml:mrow></mml:mfenced></mml:mrow></mml:mfenced></mml:mrow></mml:mrow></mml:math>](img/8.png)相同，因此 Jaccard 度量与简单 F1 将始终提供相同的排名，Jaccard 分数总是小于 F1（除非它们都是 1）。因此，在宏 F1 和 Jaccard 度量之间几乎没有什么可选择的，但由于一些作者在比较分类器时使用其中一个，而另一些作者使用另一个，因此在我们比较分类器的所有表格中，我们将给出宏 F1、微 F1 和 Jaccard。

我们将要查看的一些数据集允许推文具有任意数量的标签。许多推文根本不表达任何情绪，因此我们需要允许在黄金标准不分配任何标签的情况下，以及相当多的推文表达了多种情绪的情况。我们将这类数据集称为**多标签数据集**。这必须与有多个类别（**多类数据集**）且任务是将每条推文分配到确切一个选项的数据集区分开来。正如我们将看到的，多标签数据集比单标签多类数据集更难处理。就度量标准而言，真正的负例（在之前给出的任何度量标准中都没有使用）对于多标签任务变得更加重要，尤其是在黄金标准可能不对推文分配任何标签的任务中。我们将在*第十章*“多分类器”中详细探讨多标签任务。现在，我们只需注意，训练和测试应使用多折进行，以确保每个数据点恰好用于一次测试，对于小型数据集，使用较小的测试集（因此有更多的折），并且宏观 F1 和 Jaccard 分数是比较分类器的最有用指标。

# 摘要

毫无疑问，找到合适的数据可能是一个挑战，但有一些方法可以减轻这一点。例如，有许多具有全面搜索功能的仓库，允许您找到相关的数据集。

在本章中，我们首先探讨了公共数据源，并介绍了一些最受欢迎的数据源。我们发现许多数据集是免费的，但访问某些数据集需要订阅仓库。即使存在这些仓库，有时仍然需要“自行创建”数据集，因此我们探讨了这样做的好处以及我们可能收集自己的数据并创建自己的数据集的一些方法。然后，我们讨论了一些专门用于情感分析问题的数据集的来源，例如来自竞赛网站的数据集。数据集通常包含有关个人的敏感信息，例如他们的个人信仰、行为和心理健康状况，因此我们指出，在使用数据集时考虑数据隐私和伦理问题至关重要。我们还探讨了如何将类似所需的数据集进行转换，以及如何使它们更有用。最后，我们探讨了如何将非英语数据集转换为我们的目标语言，以及这样做可能遇到的问题。

我们还考虑了在尝试评估分类器时出现的问题，引入了交叉验证的概念，并查看了一些可以用于评估分类器的指标。我们指出，当数据集相对较小时，将数据分成大量的小批次，每个批次都包含一小部分用于测试的数据，这一点非常重要。例如，进行 10 折交叉验证并不一定比使用 5 折更严格：如果我们有大量数据，那么使用少量批次，每个批次包含大量测试数据，是完全合理的。我们还考虑了最常用的一些指标的优点，并决定由于不同的作者使用不同的指标，因此报告所有这些指标是有意义的，因为这使得比较给定分类器的分数与其他地方发布的分数成为可能。

在下一章中，我们将探讨标注、关键考虑因素以及一些可以提高过程有效性和准确性的良好实践。我们还将探索提高结果的技术，并查看数据标注任务的一个简单架构和 UI。

# 参考文献

要了解更多关于本章所涵盖的主题，请查看以下资源：

+   Mohammad, S. M., Bravo-Marquez, F., Salameh, M., and Kiritchenko, S. (2018). *SemEval-2018 Task 1: Affect in Tweets*. Proceedings of International Workshop on Semantic Evaluation (SemEval-2018).
