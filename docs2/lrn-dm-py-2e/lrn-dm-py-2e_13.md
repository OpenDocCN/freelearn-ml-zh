# 下一步...

在课程中，有很多未被探索的途径、未提出的选项和未充分研究的主题。在本附录中，我为那些希望进行额外学习并使用 Python 推进数据挖掘的人创建了一系列下一步行动。

本附录旨在学习更多关于数据挖掘的知识。还包括一些扩展所做工作的挑战。其中一些将是小的改进；有些工作会更多——我已经记下了那些比其他任务明显更困难、更复杂的工作。

# 数据挖掘入门

在本章中，读者可以探索以下途径：

# Scikit-learn 教程

URL: [`scikit-learn.org/stable/tutorial/index.html`](http://scikit-learn.org/stable/tutorial/index.html)

scikit-learn 文档中包含了一系列关于数据挖掘的教程。这些教程从基本介绍到玩具数据集，再到最近研究中所用技术的全面教程。这些教程需要花费相当长的时间才能完成——它们非常全面——但学习起来非常值得。

还有大量算法已经实现，以与 scikit-learn 兼容。由于许多原因，这些算法并不总是包含在 scikit-learn 本身中，但许多这些算法的列表维护在 [`github.com/scikit-learn/scikit-learn/wiki/Third-party-projects-and-code-snippets`](https://github.com/scikit-learn/scikit-learn/wiki/Third-party-projects-and-code-snippets)。

# 扩展 Jupyter Notebook

URL: [`ipython.org/ipython-doc/1/interactive/public_server.html`](http://ipython.org/ipython-doc/1/interactive/public_server.html)

Jupyter Notebook 是一个强大的工具。它可以以多种方式扩展，其中之一是创建一个服务器来运行你的笔记本，与你的主要计算机分开。如果你使用的是低功耗的主计算机，如小型笔记本电脑，但手头上有更强大的计算机，这将非常有用。此外，你可以设置节点以执行并行计算。

# 更多数据集

URL: [`archive.ics.uci.edu/ml/`](http://archive.ics.uci.edu/ml/)

互联网上有许多来自不同来源的数据集。这些包括学术、商业和政府数据集。在 UCI ML 图书馆有一个包含良好标签的数据集集合，这是寻找测试你的算法的最佳选择之一。尝试使用 OneR 算法测试这些不同的数据集之一。

# 其他评估指标

对于其他方法有很多种评估指标。一些值得研究的著名指标包括：

+   升值指标：[`en.wikipedia.org/wiki/Lift_(data_mining)`](https://en.wikipedia.org/wiki/Lift_(data_mining))

+   分段评估指标：[`segeval.readthedocs.io/en/latest/`](http://segeval.readthedocs.io/en/latest/)

+   皮尔逊相关系数：[`en.wikipedia.org/wiki/Pearson_correlation_coefficient`](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

+   ROC 曲线下的面积：[`gim.unmc.edu/dxtests/roc3.htm`](http://gim.unmc.edu/dxtests/roc3.htm)

+   归一化互信息：[`scikit-learn.org/stable/modules/clustering.html#mutual-info-score`](http://scikit-learn.org/stable/modules/clustering.html#mutual-info-score)

这些指标中的每一个都是针对特定应用开发的。例如，段落评估指标评估将文本文档分割成块时的准确性，允许块边界之间有一定的变化。了解评估指标可以应用在哪里以及不能应用在哪里对于数据挖掘的持续成功至关重要。

# 更多应用想法

URL：[`datapipeline.com.au/`](https://datapipeline.com.au/)

如果您在寻找更多数据挖掘应用的想法，特别是针对商业的，请查看我的公司博客。我经常发布有关数据挖掘应用的文章，重点关注商业的实际成果。

# 使用 scikit-learn 估计器进行分类

最近邻算法的简单实现相当慢——它检查所有点对以找到彼此靠近的点。存在更好的实现，其中一些已经在 scikit-learn 中实现。

# 与最近邻的扩展性

URL：[`github.com/jnothman/scikit-learn/tree/pr2532`](https://github.com/jnothman/scikit-learn/tree/pr2532)

例如，可以创建一个 kd 树来加快算法（这已经包含在 scikit-learn 中）。

另一种加快此搜索的方法是使用局部敏感哈希，局部敏感哈希（LSH）。这是对 scikit-learn 的提议改进，但在写作时尚未包含在包中。前面的链接提供了一个 scikit-learn 的开发分支，您可以在其中测试数据集上的 LSH。阅读此分支附带的文档，了解如何进行此操作。

要安装它，克隆存储库并按照说明在您的计算机上安装可在[`scikit-learn.org/stable/install.html`](http://scikit-learn.org/stable/install.html)找到的 Bleeding Edge 代码。请记住使用存储库的代码而不是官方源。我建议您使用 Anaconda 来尝试 bleeding-edge 包，以免与系统上的其他库发生冲突。

# 更复杂的管道

URL：[`scikit-learn.org/stable/modules/pipeline.html#featureunion-composite-feature-spaces`](http://scikit-learn.org/stable/modules/pipeline.html#featureunion-composite-feature-spaces)

我们在这里使用的管道遵循单一流程——一个步骤的输出是另一个步骤的输入。

管道遵循转换器和估计器接口，这使我们能够在管道中嵌入管道。这对于非常复杂的模型来说是一个有用的结构，但当与特征联合（Feature Unions）结合使用时，它变得非常强大，如前一个链接所示。这允许我们一次提取多种类型的特征，然后将它们组合成一个单一的数据集。更多详情，请参阅此示例：[`scikit-learn.org/stable/auto_examples/feature_stacker.html`](http://scikit-learn.org/stable/auto_examples/feature_stacker.html)。

# 比较分类器

scikit-learn 中有许多现成的分类器。您为特定任务选择的分类器将基于各种因素。您可以通过比较 f1 分数来查看哪种方法更好，并且您可以调查这些分数的偏差，以查看结果是否具有统计学意义。

一个重要因素是它们在相同的数据上进行了训练和测试——也就是说，一个分类器的测试集是所有分类器的测试集。我们使用随机状态确保这一点——这是复制实验的一个重要因素。

# 自动学习

URL: [`rhiever.github.io/tpot/`](http://rhiever.github.io/tpot/)

URL: [`github.com/automl/auto-sklearn`](https://github.com/automl/auto-sklearn)

这几乎是一种作弊行为，但这些包会为您调查数据挖掘实验中可能的各种模型。这消除了创建一个工作流程以测试大量参数和分类器类型的需要，并让您可以专注于其他事情，例如特征提取——尽管仍然至关重要，但尚未实现自动化！

通用思路是提取您的特征，然后将结果矩阵传递给这些自动化分类算法（或回归算法）之一。它会为您进行搜索，甚至为您导出最佳模型。在 TPOT 的情况下，它甚至为您提供从头开始创建模型的 Python 代码，而无需在您的服务器上安装 TPOT。

# 使用决策树预测体育比赛赢家

URL: [`pandas.pydata.org/pandas-docs/stable/tutorials.html`](http://pandas.pydata.org/pandas-docs/stable/tutorials.html)

pandas 库是一个非常好的包——您通常用于数据加载的任何内容，在 pandas 中可能已经实现了。您可以从他们的教程中了解更多信息。

克里斯·莫菲特（Chris Moffitt）也撰写了一篇优秀的博客文章，概述了人们在 Excel 中执行的一些常见任务以及如何在 pandas 中完成这些任务：[`pbpython.com/excel-pandas-comp.html`](http://pbpython.com/excel-pandas-comp.html)

您也可以使用 pandas 处理大型数据集；请参阅用户 Jeff 的回答，了解 StackOverflow 问题的广泛概述：[`stackoverflow.com/a/14268804/307363`](http://stackoverflow.com/a/14268804/307363)。

由 Brian Connelly 编写的另一个关于 pandas 的优秀教程：[`bconnelly.net/2013/10/summarizing-data-in-python-with-pandas/`](http://bconnelly.net/2013/10/summarizing-data-in-python-with-pandas/)

# 更复杂的功能

URL：[`www.basketball-reference.com/teams/ORL/2014_roster_status.html`](http://www.basketball-reference.com/teams/ORL/2014_roster_status.html)

更大的练习！

体育队伍经常在一场比赛和另一场比赛之间发生变化。一个队伍的轻松胜利可能会因为几个最佳球员突然受伤而变成一场艰难的比赛。你也可以从篮球参考网站获取球队名单。例如，2013-2014 赛季奥兰多魔术队的名单可以在前面的链接中找到。所有 NBA 球队的类似数据都可用。

编写代码以整合球队的变化程度，并使用这些信息添加新功能可以显著提高模型。不过，这项任务需要相当多的工作！

# Dask

URL：[`dask.pydata.org/en/latest/`](http://dask.pydata.org/en/latest/)

如果你想要增强 pandas 的功能并提高其可扩展性，那么 Dask 就是你的选择。Dask 提供了 NumPy 数组、Pandas DataFrame 和任务调度的并行版本。通常，接口与原始 NumPy 或 Pandas 版本几乎相同。

# 研究

URL：[`scholar.google.com.au/`](https://scholar.google.com.au/)

更大的练习！正如你可能想象的那样，在预测 NBA 比赛以及所有体育赛事方面已经进行了大量工作。在谷歌学术搜索中搜索“<SPORT>预测”以找到关于预测你最喜欢的<SPORT>的研究。

# 使用亲和分析推荐电影

有许多值得调查的基于推荐的数据库，每个数据库都有自己的问题。

# 新数据集

URL：[`www2.informatik.uni-freiburg.de/~cziegler/BX/`](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

更大的练习！

有许多值得调查的基于推荐的数据库，每个数据库都有自己的问题。例如，Book-Crossing 数据集包含超过 278,000 个用户和超过一百万个评分。其中一些评分是明确的（用户确实给出了评分），而其他评分则更为隐晦。对这些隐晦评分的加权可能不应该像对明确评分那样高。音乐网站 www.last.fm 已经发布了一个用于音乐推荐的优秀数据集：[http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/。(http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/)]

此外，还有一个笑话推荐数据集！请参阅此处：[`eigentaste.berkeley.edu/dataset/`](http://eigentaste.berkeley.edu/dataset/)

# Eclat 算法

URL：[`www.borgelt.net/eclat.html`](http://www.borgelt.net/eclat.html)

这里实现的 APriori 算法无疑是关联规则挖掘图中最著名的算法，但并不一定是最好的。Eclat 是一种更现代的算法，相对容易实现。

# 协同过滤

URL：[`github.com/python-recsys`](https://github.com/python-recsys)

对于那些想要在推荐引擎方面走得更远的人来说，调查其他推荐格式是必要的，例如协同过滤。这个库提供了一些关于算法和实现的背景信息，以及一些教程。在[`blogs.gartner.com/martin-kihn/how-to-build-a-recommender-system-in-python/`](http://blogs.gartner.com/martin-kihn/how-to-build-a-recommender-system-in-python/)上也有一个很好的概述。

# 使用 Transformer 提取特征

根据我的看法，以下主题在深入了解使用 Transformer 提取特征时也是相关的

# 添加噪声

我们讨论了通过去除噪声来提高特征；然而，通过添加噪声，对于某些数据集可以获得更好的性能。原因很简单——这有助于通过迫使分类器稍微泛化其规则来防止过拟合（尽管过多的噪声会使模型过于泛化）。尝试实现一个可以将给定数量的噪声添加到数据集的 Transformer。在 UCI ML 的一些数据集上测试它，看看是否提高了测试集的性能。

# Vowpal Wabbit

URL：[`hunch.net/~vw/`](http://hunch.net/~vw/)

Vowpal Wabbit 是一个很棒的项目，为基于文本的问题提供了非常快速的特征提取。它附带一个 Python 包装器，允许您从 Python 代码中调用它。在大型数据集上测试它。

# word2vec

URL：[`radimrehurek.com/gensim/models/word2vec.html`](https://radimrehurek.com/gensim/models/word2vec.html)

词嵌入因其在许多文本挖掘任务中表现良好而受到研究和行业的广泛关注，这是有充分理由的：它们在许多文本挖掘任务中表现非常好。它们比词袋模型复杂得多，并创建更大的模型。当您拥有大量数据时，词嵌入是很好的特征，甚至在某些情况下还可以帮助处理更小的数据量。

# 使用朴素贝叶斯进行社交媒体洞察

在完成使用朴素贝叶斯进行社交媒体洞察后，请考虑以下要点。

# 垃圾邮件检测

URL：[`scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter`](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

使用这里的概念，您可以创建一个能够查看社交媒体帖子并确定其是否为垃圾邮件的垃圾邮件检测方法。通过首先创建垃圾邮件/非垃圾邮件帖子数据集，实现文本挖掘算法，然后评估它们来尝试这个方法。

在垃圾邮件检测中，一个重要的考虑因素是误报/漏报比率。许多人宁愿让几条垃圾邮件消息溜走，也不愿错过一条合法消息，因为过滤器在阻止垃圾邮件时过于激进。为了转换你的方法，你可以使用带有 f1 分数作为评估标准的网格搜索。参见前面的链接，了解如何进行此操作。

# 自然语言处理和词性标注

URL：[`www.nltk.org/book/ch05.html`](http://www.nltk.org/book/ch05.html)

我们在这里使用的技术与其他领域使用的某些语言模型相比相当轻量级。例如，词性标注可以帮助消除词形歧义，从而提高准确性。它随 NLTK 提供。

# 使用图挖掘发现要关注的账户

在完成本章后，请务必阅读以下内容。

# 更复杂的算法

URL：[`www.cs.cornell.edu/home/kleinber/link-pred.pdf`](https://www.cs.cornell.edu/home/kleinber/link-pred.pdf)更大的练习！

在预测图中的链接方面已经进行了广泛的研究，包括社交网络。例如，David Liben-Nowell 和 Jon Kleinberg 发表了关于这个主题的论文，这将是一个更复杂算法的好起点，之前已经链接过。

# NetworkX

URL：[`networkx.github.io/`](https://networkx.github.io/)

如果你打算更多地使用图表和网络，深入研究 NetworkX 包是非常值得你花时间的——可视化选项很棒，算法实现得很好。还有一个名为 SNAP 的库，它也提供了 Python 绑定，网址为[`snap.stanford.edu/snappy/index.html`](http://snap.stanford.edu/snappy/index.html)。

# 使用神经网络击败 CAPTCHA

你可能还会对以下主题感兴趣：

# 更好（更糟？）的 CAPTCHA

URL：[`scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html`](http://scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html)

更大的练习！

在本例中我们击败的 CAPTCHA 并不像今天通常使用的那么复杂。你可以使用以下多种技术创建更复杂的变体：

+   应用不同的转换，如 scikit-image 中的那些（见前一个链接）

+   使用不同的颜色和难以转换为灰度的颜色

+   向图像添加线条或其他形状：[`scikit-image.org/docs/dev/api/skimage.draw.html`](http://scikit-image.org/docs/dev/api/skimage.draw.html)

# 深度网络

这些技术可能会欺骗我们当前的实现，因此需要改进以使方法更好。尝试我们使用的深度网络。然而，更大的网络需要更多的数据，所以你可能需要生成比这里所做的那几千个样本更多的样本才能获得良好的性能。生成这些数据集是并行化的好候选——有很多可以独立执行的小任务。

增加数据集大小的良好方法，同样适用于其他数据集，是创建现有图像的变体。将图像上下颠倒，奇怪地裁剪，添加噪声，模糊图像，将一些随机像素变为黑色等等。

# 强化学习

URL：[`pybrain.org/docs/tutorial/reinforcement-learning.html`](http://pybrain.org/docs/tutorial/reinforcement-learning.html)

强化学习正在成为数据挖掘下一个大趋势——尽管它已经存在很长时间了！PyBrain 有一些强化学习算法，值得用这个数据集（以及其他数据集）检查。

# 作者归属

当涉及到作者归属时，确实应该阅读以下主题。

# 增加样本量

我们使用的安然应用最终只使用了整体数据集的一部分。这个数据集中还有大量其他数据可用。增加作者数量可能会降低准确性，但使用类似的方法，有可能进一步提高准确性，超过这里所达到的水平。使用网格搜索，尝试不同的 n-gram 值和不同的支持向量机参数，以在更多作者上获得更好的性能。

# 博客数据集

使用的这个数据集，提供了基于作者的分类（每个博客 ID 代表一个单独的作者）。这个数据集也可以使用这种方法进行测试。此外，还有其他可以测试的类别，如性别、年龄、行业和星座——基于作者的方法对这些分类任务有效吗？

# 本地 n-gram

URL：[`github.com/robertlayton/authorship_tutorials/blob/master/LNGTutorial.ipynb`](https://github.com/robertlayton/authorship_tutorials/blob/master/LNGTutorial.ipynb)

另一种分类器形式是本地 n-gram，它涉及为每个作者选择最佳特征，而不是为整个数据集全局选择。我编写了一个关于使用本地 n-gram 进行作者归属的教程，可在前面的链接中找到。

# 新闻文章聚类

了解一下以下主题不会有任何坏处

# 聚类评估

聚类算法的评估是一个难题——一方面，我们可以说出好的聚类看起来是什么样子；另一方面，如果我们真的知道这一点，我们应该标记一些实例并使用监督分类器！关于这个主题已经有很多论述。以下是一个关于这个主题的幻灯片，它是一个很好的挑战介绍：[`www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf`](http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf)。

此外，这里有一篇关于这个主题的非常全面的（尽管现在有点过时）论文：[`web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf.`](http://web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf.)

scikit-learn 包实现了那些链接中描述的许多度量，这里有一个概述：[`scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation`](http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)。

使用其中的一些，你可以开始评估哪些参数需要使用以获得更好的聚类。使用网格搜索，我们可以找到最大化度量的参数——就像在分类中一样。

# 时间分析

更大的练习！

我们在这里开发的代码可以在数月内重新运行。通过给每个聚类添加一些标签，您可以跟踪哪些主题随着时间的推移保持活跃，从而获得对世界新闻中讨论内容的纵向视角。为了比较聚类，可以考虑一个指标，例如之前提到的调整互信息得分。看看聚类在一个月后、两个月后、六个月后和一年后的变化。

# 实时聚类

k-means 算法可以在给定时间框架的离散分析之外，随着时间的推移迭代训练和更新。可以通过多种方式跟踪聚类移动——例如，您可以跟踪每个聚类中哪些单词流行，以及每天质心移动了多少。请记住 API 限制——您可能只需要每隔几个小时检查一次，以保持您的算法更新。

# 使用深度学习在图像中分类对象

当考虑更深入地研究分类对象时，以下主题也非常重要。

# Mahotas

URL: [`luispedro.org/software/mahotas/`](http://luispedro.org/software/mahotas/)

另一个图像处理包是 Mahotas，包括更好、更复杂的图像处理技术，可以帮助实现更高的精度，尽管它们可能带来较高的计算成本。然而，许多图像处理任务都是并行化的良好候选。更多关于图像分类的技术可以在研究文献中找到，这篇综述论文是一个很好的起点：[`ijarcce.com/upload/january/22-A%20Survey%20on%20Image%20Classification.pdf`](http://ijarcce.com/upload/january/22-A%20Survey%20on%20Image%20Classification.pdf)。

其他图像数据集可在[`rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html`](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)找到。

许多学术和基于行业的来源提供了大量的图像数据集。链接的网站列出了许多数据集和一些在它们上使用的最佳算法。实现一些较好的算法可能需要大量的自定义代码，但回报可能非常值得。

# Magenta

URL: [`github.com/tensorflow/magenta/tree/master/magenta/reviews`](https://github.com/tensorflow/magenta/tree/master/magenta/reviews)

此存储库包含一些值得阅读的高质量深度学习论文，以及论文及其技术的深入评论。如果您想深入研究深度学习，请首先查看这些论文，然后再向外扩展。

# 处理大数据

以下关于大数据的资源可能会有所帮助

# Hadoop 课程

Yahoo 和 Google 都提供了关于 Hadoop 的优秀教程，从入门到相当高级的水平。它们没有专门针对 Python 的使用，但学习 Hadoop 概念然后在 Pydoop 或类似库中应用它们可以产生很好的效果。

Yahoo 的教程：[`developer.yahoo.com/hadoop/tutorial/`](https://developer.yahoo.com/hadoop/tutorial/)

Google 的教程：[`cloud.google.com/hadoop/what-is-hadoop`](https://cloud.google.com/hadoop/what-is-hadoop)

# Pydoop

URL：[`crs4.github.io/pydoop/tutorial/index.html`](http://crs4.github.io/pydoop/tutorial/index.html)

Pydoop 是一个用于运行 Hadoop 作业的 Python 库。Pydoop 也可以与 HDFS（Hadoop 文件系统）一起工作，尽管你同样可以在 mrjob 中获取该功能。Pydoop 将为你提供更多控制运行某些作业的能力。

# 推荐引擎

构建一个大型推荐引擎是测试你大数据技能的好方法。马克·利特温斯基（Mark Litwintschik）的一篇优秀的博客文章介绍了一个使用 Apache Spark（大数据技术）的引擎：[`tech.marksblogg.com/recommendation-engine-spark-python.html`](http://tech.marksblogg.com/recommendation-engine-spark-python.html)

# W.I.L.L

URL：[`github.com/ironman5366/W.I.L.L`](https://github.com/ironman5366/W.I.L.L)

这是一个非常大的项目！

这个开源的个人助理可以成为你的下一个来自钢铁侠的 JARVIS。你可以通过数据挖掘技术添加到这个项目中，使其学会执行你经常需要做的某些任务。这并不容易，但潜在的生产力提升是值得的。

# 更多资源

以下是一些额外的信息资源：

# Kaggle 竞赛

URL：[www.kaggle.com/](http://www.kaggle.com/)

Kaggle 定期举办数据挖掘竞赛，通常伴有现金奖励。

在 Kaggle 竞赛中测试你的技能是快速学习如何处理真实世界数据挖掘问题的好方法。论坛很棒，共享环境——在竞赛中，你经常会看到排名前十的参赛者发布的代码！

# Coursera

URL：[www.coursera.org](http://www.coursera.org)

Coursera 包含许多关于数据挖掘和数据科学的课程。许多课程是专业化的，例如大数据和图像处理。一个很好的入门课程是安德鲁·吴（Andrew Ng）的著名课程：[`www.coursera.org/learn/machine-learning/`](https://www.coursera.org/learn/machine-learning/)。

这比这个要高级一些，对于感兴趣的读者来说，这将是一个很好的下一步。

对于神经网络，你可以查看这个课程：[`www.coursera.org/course/neuralnets`](https://www.coursera.org/course/neuralnets)。

如果你完成了所有这些，可以尝试在[`www.coursera.org/course/pgm`](https://www.coursera.org/course/pgm)上学习概率图模型课程。
