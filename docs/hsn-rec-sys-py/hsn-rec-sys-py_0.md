# 前言

推荐系统是几乎所有互联网业务的核心，从 Facebook 到 Netflix 再到 Amazon。提供好的推荐，无论是朋友、电影还是食品，都能在定义用户体验和吸引客户使用和购买平台上的商品方面发挥重要作用。

本书将向您展示如何做到这一点。您将了解行业中使用的不同类型的推荐系统，并学习如何从零开始使用 Python 构建它们。不需要翻阅大量的线性代数和机器学习理论，您将尽快开始构建并了解推荐系统。

在本书中，您将构建一个 IMDB Top 250 克隆，基于电影元数据的内容推荐引擎，利用客户行为数据的协同过滤器，以及一个结合了基于内容和协同过滤技术的混合推荐系统。

通过本书，您只需具备 Python 基础知识，就可以开始构建推荐系统，完成后，您将深入理解推荐系统的工作原理，并能够将所学的技术应用到自己的问题领域。

# 本书适合的人群

如果您是 Python 开发者，想要开发社交网络、新闻个性化或智能广告应用，那么这本书适合您。机器学习技术的基础知识会有所帮助，但并非必需。

# 本书内容概述

第一章，*推荐系统入门*，介绍了推荐问题及其常用的解决模型。

第二章，*使用 Pandas 库处理数据*，展示了使用 Pandas 库进行各种数据清洗技术的应用。

第三章，*使用 Pandas 构建 IMDB Top 250 克隆*，带领您完成构建热门电影榜单和明确考虑用户偏好的基于知识的推荐系统的过程。

第四章，*构建基于内容的推荐系统*，描述了如何构建利用电影情节和其他元数据提供推荐的模型。

第五章，*数据挖掘技术入门*，介绍了构建和评估协同过滤推荐模型时使用的各种相似度评分、机器学习技术和评估指标。

第六章，*构建协同过滤器*，带领您构建各种利用用户评分数据进行推荐的协同过滤器。

第七章，*混合推荐系统*，概述了实践中使用的各种混合推荐系统，并带您了解如何构建一个结合内容和协同过滤的模型。

# 为了充分利用本书

本书将为您提供最大益处，前提是您有一定的 Python 开发经验，或者您只是希望开发社交网络、新闻个性化或智能广告应用程序的读者，那么这本书就是为您量身定做的。如果您对**机器学习**（**ML**）有一些了解，会有所帮助，但不是必需的。

# 下载示例代码文件

您可以从您的账户中下载本书的示例代码文件，网址为[www.packtpub.com](http://www.packtpub.com)。如果您是在其他地方购买的本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)，并注册以直接将文件通过邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  登录或注册至[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”标签。

1.  点击“代码下载 & 勘误”。

1.  在搜索框中输入书名，并按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的工具解压或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址是[`github.com/PacktPublishing/Hands-On-Recommendation-Systems-with-Python`](https://github.com/PacktPublishing/Hands-On-Recommendation-Systems-with-Python)。如果代码有更新，将会在现有的 GitHub 仓库中进行更新。

我们还提供了来自我们丰富的书籍和视频目录中的其他代码包，您可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到它们。快去看看吧！

# 下载彩色图像

我们还提供了一个包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnRecommendationSystemswithPython_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/HandsOnRecommendationSystemswithPython_ColorImages.pdf)。

# 代码示例

访问以下链接，查看代码运行的视频：

[`bit.ly/2JV4oeu`](http://bit.ly/2JV4oeu)。

# 使用的规范

本书中使用了多种文本规范。

`CodeInText`：表示文本中的代码词语、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。例如：“现在让我们使用`surprise`包实现 SVD 过滤器。”

一段代码如下所示：

```py
#Import SVD
from surprise import SVD

#Define the SVD algorithm object
svd = SVD()

#Evaluate the performance in terms of RMSE
evaluate(svd, data, measures=['RMSE'])
```

当我们希望您注意到代码块中的特定部分时，相关的行或项目会加粗显示：

```py
 else:
        #Default to a rating of 3.0 in the absence of any information
        wmean_rating = 3.0
    return wmean_rating
score(cf_user_wmean)

OUTPUT:
1.0174483808407588
```

任何命令行输入或输出格式如下所示：

```py
sudo pip3 install scikit-surprise
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的文字。例如，菜单或对话框中的单词会以这种方式显示在文本中。这里是一个例子：“我们看到 `u.user` 文件包含有关用户的统计信息，如他们的年龄、性别、职业和邮政编码。”

警告或重要说明以这种方式展示。

提示和技巧以这种方式展示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：请发送邮件至 `feedback@packtpub.com`，并在邮件主题中提及书籍标题。如果您有任何关于本书的问题，请通过 `questions@packtpub.com` 与我们联系。

**勘误**：尽管我们已尽力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，我们将不胜感激，如果您能向我们报告。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表格链接并输入相关信息。

**盗版**：如果您在互联网上遇到我们作品的任何非法复制版本，我们将非常感激您能提供该文件的位置或网站名称。请通过 `copyright@packtpub.com` 联系我们，并附上相关材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域具有专业知识并且有兴趣撰写或贡献书籍内容，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。阅读并使用本书后，为什么不在您购买书籍的网站上留下评价呢？潜在读者可以通过您的公正评价做出购买决策，我们 Packt 也能了解您对我们产品的看法，我们的作者也能看到您对他们书籍的反馈。谢谢！

更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
