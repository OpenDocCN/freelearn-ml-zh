# 前言

世界上每一个人和每一个组织都在管理数据，无论他们是否意识到这一点。数据被用来描述我们周围的世界，可以用于几乎任何目的，从分析消费者习惯以推荐最新的产品和服务，到抗击疾病、气候变化和严重的有组织犯罪。最终，我们管理数据是为了从中获得价值，无论是个人价值还是商业价值，而且世界上许多组织传统上都在投资工具和技术，以帮助他们更快、更有效地处理数据，以便提供可操作的见解。

但我们现在生活在一个高度互联的世界，由大量数据的创建和消费驱动，数据不再是仅限于电子表格的行和列，而是它自身的一种有机和不断发展的资产。随着这一认识的到来，组织在进入以智能驱动的第四次工业革命时面临着重大挑战——我们如何管理每秒产生的各种格式的数据量（不仅想到电子表格和数据库，还包括社交媒体帖子、图片、视频、音乐、在线论坛和文章、计算机日志文件等等）？一旦我们知道如何管理所有这些数据，我们如何知道向它提出什么问题，以便从中获得真正的个人或商业价值？

这本书的焦点是通过从第一原理开始以动手的方式帮助我们回答这些问题。我们介绍了最新的尖端技术（包括 Apache Spark 的大数据生态系统），这些技术可以用来管理和处理大数据。然后我们探讨了高级算法类别（机器学习、深度学习、自然语言处理和认知计算），这些算法可以应用于大数据生态系统，帮助我们揭示之前隐藏的关系，以便理解数据在告诉我们什么，从而最终解决现实世界的问题。

# 这本书的读者对象

这本书的目标读者是商业分析师、数据分析师、数据科学家、数据工程师和软件工程师，他们可能目前每天的工作涉及使用电子表格或关系型数据库分析数据，可能还会使用 VBA、**结构化查询语言**（**SQL**）或甚至 Python 来计算统计聚合（如平均值）以及生成图表、图表、交叉表和其他报告媒介。

随着各种格式和频率的数据爆炸式增长，你可能现在面临的挑战不仅是管理所有这些数据，还要理解它所传达的信息。你很可能已经听说过**大数据**、**人工智能**和**机器学习**这些术语，但现在你可能希望了解如何开始利用这些新技术和框架，不仅是在理论上，而且在实践中，以解决你的商业挑战。如果这听起来很熟悉，那么这本书就是为你准备的！

# 为了最大限度地利用这本书

尽管本书旨在从第一原理解释一切，但具备数学符号和基本编程技能（例如用于数据转换的 SQL、Base SAS、R 或 Python）将是有益的（尽管不是严格必需的）。对于 SQL 和 Python 的初学者，一个好的学习网站是[`www.w3schools.com`](https://www.w3schools.com)。

为了安装、配置和提供包含详细说明的先决软件服务的自包含本地开发环境，需要具备 Linux shell 命令的基本知识。第二章，*设置本地开发环境*，描述了配置 CentOS 7 的各种选项。对于 Linux 命令行的初学者，一个好的学习网站是[`linuxcommand.org`](http://linuxcommand.org)。

7-Zip/PeaZip for Linux

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

假设您有权访问配置了 CentOS Linux 7（或 Red Hat Linux）操作系统的物理或虚拟机。如果您没有，第二章，*设置本地开发环境*，描述了配置 CentOS 7 **虚拟机**（**VM**）的各种选项，包括通过云计算平台（如**Amazon Web Services**（**AWS**）、Microsoft Azure、**Google Cloud Platform**（**GCP**））、虚拟专用服务器托管公司或免费虚拟化软件（如 Oracle VirtualBox 和 VMWare Workstation Player，这些软件可以安装在您的本地物理设备上，如台式机或笔记本电脑）。

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载完成后，请确保您使用最新版本的软件解压缩或提取文件夹，例如：

+   假设您有权访问配置了 CentOS Linux 7（或 Red Hat Linux）操作系统的物理或虚拟机。如果您没有，第二章，*设置本地开发环境*，描述了配置 CentOS 7 **虚拟机**（**VM**）的各种选项，包括通过云计算平台（如**Amazon Web Services**（**AWS**）、Microsoft Azure、**Google Cloud Platform**（**GCP**））、虚拟专用服务器托管公司或免费虚拟化软件（如 Oracle VirtualBox 和 VMWare Workstation Player，这些软件可以安装在您的本地物理设备上，如台式机或笔记本电脑）。

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Machine-Learning-with-Apache-Spark-Quick-Start-Guide`](https://github.com/PacktPublishing/Machine-Learning-with-Apache-Spark-Quick-Start-Guide)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包可供选择，这些代码包可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。去看看吧！

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“将下载的`WebStorm-10*.dmg`磁盘映像文件作为系统中的另一个磁盘挂载。”

代码块设置如下：

```py
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
import random
```

任何命令行输入或输出都应按以下方式编写：

```py
> source /etc/profile.d/java.sh
> echo $PATH
> echo $JAVA_HOME
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要注意事项看起来像这样。

小贴士和技巧看起来像这样。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并给我们发送邮件至`customercare@packtpub.com`。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告这一点，我们将不胜感激。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问[packt.com](http://www.packt.com/)。
