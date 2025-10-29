# 第十三章：下一步...

在课程中，有许多未被探索的途径、未提出的选项和未充分研究的主题。在本附录中，我为那些希望进行额外学习并使用 Python 推进数据挖掘的人创建了一系列下一步行动。

本附录旨在深入了解数据挖掘。还包括一些扩展所做工作的挑战。其中一些将是小的改进；有些工作会更多——我已经记下了那些明显比其他任务更困难、更复杂的工作。

# 数据挖掘入门

在本章中，以下是一些读者可以探索的途径：

# Scikit-learn 教程

URL：[`scikit-learn.org/stable/tutorial/index.html`](http://scikit-learn.org/stable/tutorial/index.html)

scikit-learn 文档中包含了一系列关于数据挖掘的教程。这些教程从基本介绍到玩具数据集，再到最近研究中所用技术的全面教程。这些教程需要相当长的时间才能完成——它们非常全面——但学习起来非常值得。

为了与 scikit-learn 兼容，已经实现了许多算法。由于多种原因，这些算法并不总是包含在 scikit-learn 本身中，但许多这些算法的列表维护在[`github.com/scikit-learn/scikit-learn/wiki/Third-party-projects-and-code-snippets`](https://github.com/scikit-learn/scikit-learn/wiki/Third-party-projects-and-code-snippets)。

# 扩展 Jupyter Notebook

URL：[`ipython.org/ipython-doc/1/interactive/public_server.html`](http://ipython.org/ipython-doc/1/interactive/public_server.html)

Jupyter Notebook 是一个强大的工具。它可以以多种方式扩展，其中之一是创建一个服务器来运行你的 Notebooks，与你的主要计算机分开。如果你使用的是低功耗的主计算机，如小型笔记本电脑，但你有更强大的计算机可用，这非常有用。此外，你可以设置节点以执行并行计算。

# 更多数据集

URL：[`archive.ics.uci.edu/ml/`](http://archive.ics.uci.edu/ml/)

互联网上有来自多个不同来源的大量数据集。这些包括学术、商业和政府数据集。在 UCI ML 图书馆中可以找到一系列标注良好的数据集，这是寻找测试算法数据集的最佳选择之一。尝试使用 OneR 算法对这些不同的数据集进行测试。

# 其他评估指标

对于其他任务，有广泛的评估指标。一些值得调查的指标包括：

+   升值指标：[`en.wikipedia.org/wiki/Lift_(data_mining)`](https://en.wikipedia.org/wiki/Lift_(data_mining))

+   段落评估指标：[`segeval.readthedocs.io/en/latest/`](http://segeval.readthedocs.io/en/latest/)

+   皮尔逊相关系数：[`en.wikipedia.org/wiki/Pearson_correlation_coefficient`](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

+   ROC 曲线下的面积：[`gim.unmc.edu/dxtests/roc3.htm`](http://gim.unmc.edu/dxtests/roc3.htm)

+   标准化互信息：[`scikit-learn.org/stable/modules/clustering.html#mutual-info-score`](http://scikit-learn.org/stable/modules/clustering.html#mutual-info-score)

这些指标中的每一个都是针对特定应用开发的。例如，段落评估指标评估将文本文档分割成块时的准确性，允许块边界之间有一定的变化。了解评估指标可以应用和不能应用的地方对于数据挖掘的持续成功至关重要。

# 更多应用想法

URL：[`datapipeline.com.au/`](https://datapipeline.com.au/)

如果您在寻找更多数据挖掘应用的想法，特别是针对商业用途的，请查看我公司的博客。我经常发布关于数据挖掘应用的文章，重点关注对企业的实际成果。

# 使用 scikit-learn 估计器进行分类

最近邻算法的简单实现相当慢——它检查所有点对以找到彼此接近的点。存在更好的实现，其中一些已在 scikit-learn 中实现。

# 使用最近邻算法进行可扩展性

URL：[`github.com/jnothman/scikit-learn/tree/pr2532`](https://github.com/jnothman/scikit-learn/tree/pr2532)

例如，可以创建一个加速算法的 kd 树（这已经在 scikit-learn 中实现）。

另一种加快搜索速度的方法是使用局部敏感哈希，即局部敏感哈希（LSH）。这是为 scikit-learn 提出的一种改进，但在撰写本文时尚未包含在软件包中。前面的链接提供了一个 scikit-learn 的开发分支，您可以在其中测试 LSH 在数据集上的应用。请阅读该分支附带的文档，以了解如何进行此操作。

要安装它，请克隆存储库并按照说明在您的计算机上安装[`scikit-learn.org/stable/install.html`](http://scikit-learn.org/stable/install.html)上可用的 Bleeding Edge 代码。请记住使用存储库的代码而不是官方源。我建议您使用 Anaconda 来尝试 bleeding-edge 包，以免与系统上的其他库发生冲突。

# 更复杂的管道

URL：[`scikit-learn.org/stable/modules/pipeline.html#featureunion-composite-feature-spaces`](http://scikit-learn.org/stable/modules/pipeline.html#featureunion-composite-feature-spaces)

我们在这里使用的管道遵循单一流程——一个步骤的输出是另一个步骤的输入。

管道遵循转换器和估计器接口，这使我们能够在管道中嵌入管道。这对于非常复杂的模型来说是一个有用的结构，但当与特征联合使用时，它变得非常强大，如前一个链接所示。这允许我们一次提取多种类型的特征，然后将它们组合成一个单一的数据集。更多详情，请参阅此示例：[`scikit-learn.org/stable/auto_examples/feature_stacker.html`](http://scikit-learn.org/stable/auto_examples/feature_stacker.html)。

# 比较分类器

scikit-learn 中有很多现成的分类器可供使用。你为特定任务选择的分类器将基于多种因素。你可以比较 f1 分数来查看哪种方法更好，并且可以调查这些分数的偏差，以查看结果是否具有统计学意义。

一个重要的因素是它们是在相同的数据上训练和测试的——也就是说，一个分类器的测试集是所有分类器的测试集。我们使用随机状态来确保这一点——这是复制实验的一个重要因素。

# 自动学习

URL: [`rhiever.github.io/tpot/`](http://rhiever.github.io/tpot/)

URL: [`github.com/automl/auto-sklearn`](https://github.com/automl/auto-sklearn)

这几乎是一种作弊行为，但这些包将为你调查数据挖掘实验中可能的大量模型。这消除了创建一个工作流程以测试大量参数和分类器类型的需求，并让你专注于其他事情，如特征提取——这仍然至关重要，但尚未自动化！

通用思路是提取你的特征，然后将结果矩阵传递给这些自动化分类算法（或回归算法）之一。它为你进行搜索，甚至为你导出最佳模型。在 TPOT 的情况下，它甚至为你提供从头开始创建模型的 Python 代码，而无需在你的服务器上安装 TPOT。

# 使用决策树预测体育比赛胜者

URL: [`pandas.pydata.org/pandas-docs/stable/tutorials.html`](http://pandas.pydata.org/pandas-docs/stable/tutorials.html)

pandas 库是一个非常好的包——你通常用来加载数据的任何东西，在 pandas 中可能已经实现了。你可以从他们的教程中了解更多信息。

Chris Moffitt 还写了一篇很好的博客文章，概述了人们在 Excel 中执行的一些常见任务以及如何在 pandas 中执行这些任务：[`pbpython.com/excel-pandas-comp.html`](http://pbpython.com/excel-pandas-comp.html)

你也可以使用 pandas 处理大型数据集；请参阅用户 Jeff 的回答，了解 StackOverflow 上的这个问题，以获得对整个过程的详细概述：[`stackoverflow.com/a/14268804/307363`](http://stackoverflow.com/a/14268804/307363)。

由 Brian Connelly 编写的另一个关于 pandas 的优秀教程：[`bconnelly.net/2013/10/summarizing-data-in-python-with-pandas/`](http://bconnelly.net/2013/10/summarizing-data-in-python-with-pandas/)

# 更复杂的功能

URL: [`www.basketball-reference.com/teams/ORL/2014_roster_status.html`](http://www.basketball-reference.com/teams/ORL/2014_roster_status.html)

更大的练习！

体育队伍经常从一场比赛到另一场比赛发生变化。一个队伍的轻松胜利可能会因为几个最佳球员突然受伤而变成一场艰难的比赛。你也可以从篮球参考网站获取球队名单。例如，2013-2014 赛季奥兰多魔术队的名单可在前面的链接中找到。所有 NBA 球队的类似数据都可用。

编写代码以集成球队的变化程度，并使用这些信息添加新功能可以显著提高模型。不过，这项任务需要相当多的工作！

# Dask

URL: [`dask.pydata.org/en/latest/`](http://dask.pydata.org/en/latest/)

如果你想要利用 pandas 的特性并提高其可扩展性，那么 Dask 就是你的选择。Dask 提供了 NumPy 数组和 Pandas DataFrame 的并行版本以及任务调度。通常，接口与原始 NumPy 或 Pandas 版本几乎相同。

# 研究

URL: [`scholar.google.com.au/`](https://scholar.google.com.au/)

更大的练习！正如你可能想象的那样，在预测 NBA 比赛以及所有体育赛事方面已经进行了大量工作。在谷歌学术中搜索 "<SPORT> prediction" 以找到预测你最喜欢的 <SPORT> 的研究。

# 使用亲和分析推荐电影

有许多值得调查的基于推荐的数据库，每个数据库都有自己的问题。

# 新数据集

URL: [`www2.informatik.uni-freiburg.de/~cziegler/BX/`](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

更大的练习！

有许多值得调查的基于推荐的数据库，每个数据库都有自己的问题。例如，Book-Crossing 数据集包含超过 278,000 个用户和超过一百万条评分。其中一些评分是明确的（用户确实给出了评分），而其他则是更隐含的。对这些隐含评分的加权可能不应该像对明确评分那样高。音乐网站 www.last.fm 发布了一个用于音乐推荐的优秀数据集：[`www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/.`](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/.)

还有笑话推荐数据集！请参见这里：[`eigentaste.berkeley.edu/dataset/.`](http://eigentaste.berkeley.edu/dataset/.)

# Eclat 算法

URL: [`www.borgelt.net/eclat.html`](http://www.borgelt.net/eclat.html)

这里实现的 APriori 算法是关联规则挖掘图中最著名的，但并不一定是最好的。Eclat 是一个更现代的算法，可以相对容易地实现。

# 协同过滤

URL: [`github.com/python-recsys`](https://github.com/python-recsys)

对于那些想要在推荐引擎方面走得更远的人来说，有必要调查其他推荐格式，例如协同过滤。这个库提供了一些关于算法和实现的背景信息，以及一些教程。在 [`blogs.gartner.com/martin-kihn/how-to-build-a-recommender-system-in-python/`](http://blogs.gartner.com/martin-kihn/how-to-build-a-recommender-system-in-python/) 也有一个很好的概述。

# 使用 Transformer 提取特征

根据我的看法，以下主题在深入了解使用 Transformer 提取特征时也是相关的。

# 添加噪声

我们已经讨论了通过去除噪声来提高特征；然而，通过添加噪声，对于某些数据集可以获得更好的性能。原因很简单——它通过迫使分类器稍微泛化其规则来帮助防止过拟合（尽管过多的噪声会使模型过于泛化）。尝试实现一个可以将给定数量的噪声添加到数据集的 Transformer。在 UCI ML 的某些数据集上测试一下，看看是否提高了测试集的性能。

# Vowpal Wabbit

URL: [`hunch.net/~vw/`](http://hunch.net/~vw/)

Vowpal Wabbit 是一个优秀的项目，为基于文本的问题提供了非常快速的特征提取。它附带一个 Python 包装器，允许您从 Python 代码中调用它。在大型数据集上测试一下。

# word2vec

URL: [`radimrehurek.com/gensim/models/word2vec.html`](https://radimrehurek.com/gensim/models/word2vec.html)

词嵌入在研究和工业界受到了很多关注，原因很好：它们在许多文本挖掘任务上表现非常出色。它们比词袋模型复杂得多，并创建更大的模型。当您拥有大量数据时，词嵌入是很好的特征，甚至在某些情况下可以帮助处理更小的数据量。

# 使用朴素贝叶斯进行社交媒体洞察

在完成使用朴素贝叶斯进行社交媒体洞察后，请考虑以下要点。

# 垃圾邮件检测

URL: [`scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter`](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

使用这里的概念，您可以创建一个能够查看社交媒体帖子并确定其是否为垃圾邮件的垃圾邮件检测方法。通过首先创建垃圾邮件/非垃圾邮件帖子数据集，实现文本挖掘算法，然后评估它们来尝试这个方法。

在垃圾邮件检测中，一个重要的考虑因素是误报/漏报比率。许多人宁愿让几条垃圾邮件消息通过，也不愿错过一条合法消息，因为过滤器在阻止垃圾邮件时过于激进。为了将您的这种方法转化为现实，您可以使用带有 f1 分数作为评估标准的网格搜索。有关如何操作的更多信息，请参阅前面的链接。

# 自然语言处理和词性标注

URL: [`www.nltk.org/book/ch05.html`](http://www.nltk.org/book/ch05.html)

与其他领域使用的某些语言模型相比，我们这里使用的技术相当轻量级。例如，词性标注可以帮助消除词形歧义，从而提高准确性。它随 NLTK 提供。

# 使用图挖掘发现要关注的账户

完成本章后，请务必阅读以下内容。

# 更复杂的算法

URL: [`www.cs.cornell.edu/home/kleinber/link-pred.pdf`](https://www.cs.cornell.edu/home/kleinber/link-pred.pdf)更大的练习！

在图和社交网络中预测链接的研究已经非常广泛。例如，David Liben-Nowell 和 Jon Kleinberg 发表了关于这个主题的论文，这将是一个更复杂算法的好起点，之前已经链接。

# NetworkX

URL: [`networkx.github.io/`](https://networkx.github.io/)

如果你打算更多地使用图表和网络，深入研究 NetworkX 包是非常值得你花时间的——可视化选项很棒，算法实现得很好。另一个名为 SNAP 的库也提供了 Python 绑定，请访问[`snap.stanford.edu/snappy/index.html`](http://snap.stanford.edu/snappy/index.html)。

# 使用神经网络战胜 CAPTCHA

你可能还会对以下主题感兴趣：

# 更好（更糟？）的 CAPTCHA

URL: [`scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html`](http://scikit-image.org/docs/dev/auto_examples/applications/plot_geometric.html)

更大的练习！

在这个例子中我们战胜的 CAPTCHA 不如今天通常使用的复杂。你可以使用以下多种技术创建更复杂的变体：

+   应用不同的转换，如 scikit-image 中的转换（见前述链接）

+   使用不同的颜色以及那些不易转换为灰度的颜色

+   向图像中添加线条或其他形状：[`scikit-image.org/docs/dev/api/skimage.draw.html`](http://scikit-image.org/docs/dev/api/skimage.draw.html)

# 深度网络

这些技术可能会欺骗我们当前的实现，因此需要改进以使方法更好。尝试我们使用的一些深度网络。然而，更大的网络需要更多的数据，所以你可能需要生成比这里我们做的几千个样本更多的样本才能获得良好的性能。生成这些数据集是并行化的好候选——许多可以独立执行的小任务。

增加数据集大小的良好想法，这也适用于其他数据集，是创建现有图像的变体。将图像颠倒过来，奇怪地裁剪，添加噪声，模糊图像，将一些随机像素变为黑色等等。

# 强化学习

URL: [`pybrain.org/docs/tutorial/reinforcement-learning.html`](http://pybrain.org/docs/tutorial/reinforcement-learning.html)

强化学习正在成为数据挖掘下一个大趋势——尽管它已经存在很长时间了！PyBrain 有一些强化学习算法，值得用这个数据集（以及其他数据集！）进行检查。

# 作者归属

当涉及到作者归属时，确实应该阅读以下主题。

# 增加样本大小

我们使用的 Enron 应用程序最终只使用了整个数据集的一部分。这个数据集中还有大量其他数据可用。增加作者数量可能会降低准确性，但使用类似的方法，有可能进一步提高准确性，超过这里所达到的。使用网格搜索（Grid Search），尝试不同的 n-gram 值和不同的支持向量机（SVM）参数，以在更多作者上获得更好的性能。

# 博客数据集

使用的这个数据集提供了基于作者的分类（每个博客 ID 代表一个单独的作者）。这个数据集也可以用这种方法进行测试。此外，还有其他可以测试的分类类别，如性别、年龄、行业和星座——基于作者的方法是否适合这些分类任务？

# 本地 n-gram

URL：[`github.com/robertlayton/authorship_tutorials/blob/master/LNGTutorial.ipynb`](https://github.com/robertlayton/authorship_tutorials/blob/master/LNGTutorial.ipynb)

另一种分类器形式是本地 n-gram，它涉及为每个作者选择最佳特征，而不是为整个数据集全局选择。我写了一个关于使用本地 n-gram 进行作者归属的教程，可在前面的链接中找到。

# 聚类新闻文章

了解一下以下主题不会有任何坏处

# 聚类评估

聚类算法的评估是一个难题——一方面，我们可以说出好的聚类是什么样的；另一方面，如果我们真的知道这一点，我们应该标记一些实例并使用监督分类器！关于这个主题已经有很多讨论。以下是一个关于这个主题的幻灯片，它是一个很好的挑战介绍：[`www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf`](http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf)。

此外，这里还有一篇关于这个主题的非常全面的（尽管现在有些过时）论文：[`web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf.`](http://web.itu.edu.tr/sgunduz/courses/verimaden/paper/validity_survey.pdf.)

scikit-learn 包实现了那些链接中描述的许多指标，这里有一个概述：[`scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation`](http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)。

使用其中的一些方法，你可以开始评估哪些参数需要被使用以获得更好的聚类效果。使用网格搜索（Grid Search），我们可以找到最大化一个指标（metric）的参数——就像在分类中一样。

# 时间分析

更大的练习！

我们在这里开发的代码可以在数月内重新运行。通过为每个簇添加一些标签，您可以跟踪哪些主题随着时间的推移保持活跃，从而获得对世界新闻中讨论内容的纵向视角。为了比较簇，可以考虑一个指标，如之前链接到 scikit-learn 文档的调整互信息得分。看看簇在一个月后、两个月后、六个月后和一年后的变化。

# 实时聚类

k-means 算法可以在一段时间内迭代训练和更新，而不是在给定的时间框架内进行离散分析。可以通过多种方式跟踪簇的移动——例如，您可以跟踪每个簇中哪些单词流行以及每天质心移动了多少。请记住 API 限制——您可能每小时只需进行一次检查，以保持您的算法更新。

# 使用深度学习对图像中的对象进行分类

在考虑更深入地研究分类对象时，以下主题也很重要。

# Mahotas

URL: [`luispedro.org/software/mahotas/`](http://luispedro.org/software/mahotas/)

另一个图像处理包是 Mahotas，包括更好、更复杂的图像处理技术，可以帮助实现更高的精度，尽管它们可能带来较高的计算成本。然而，许多图像处理任务都是并行化的良好候选。在研究文献中可以找到更多关于图像分类的技术，这篇综述论文是一个很好的起点：[`ijarcce.com/upload/january/22-A%20Survey%20on%20Image%20Classification.pdf`](http://ijarcce.com/upload/january/22-A%20Survey%20on%20Image%20Classification.pdf)。

其他图像数据集可在[`rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html`](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)找到。

从许多学术和基于行业的来源可以获得许多图像数据集。链接的网站列出了大量数据集和一些最佳算法。实现一些较好的算法可能需要大量的自定义代码，但回报可能非常值得。

# Magenta

URL: [`github.com/tensorflow/magenta/tree/master/magenta/reviews`](https://github.com/tensorflow/magenta/tree/master/magenta/reviews)

该存储库包含一些高质量的深度学习论文，值得阅读，以及论文及其技术的深入评论。如果您想深入研究深度学习，请首先查看这些论文，然后再向外扩展。

# 处理大数据

以下关于大数据的资源可能会有所帮助

# Hadoop 课程

Yahoo 和 Google 都提供了从入门到相当高级水平的 Hadoop 教程。它们没有专门针对 Python 的使用，但学习 Hadoop 概念然后在 Pydoop 或类似库中应用它们可以产生很好的效果。

Yahoo 的教程：[`developer.yahoo.com/hadoop/tutorial/`](https://developer.yahoo.com/hadoop/tutorial/)

Google 的教程：[`cloud.google.com/hadoop/what-is-hadoop`](https://cloud.google.com/hadoop/what-is-hadoop)

# Pydoop

URL: [`crs4.github.io/pydoop/tutorial/index.html`](http://crs4.github.io/pydoop/tutorial/index.html)

Pydoop 是一个用于运行 Hadoop 作业的 Python 库。Pydoop 还与 HDFS（Hadoop 文件系统）兼容，尽管你也可以在 mrjob 中获得该功能。Pydoop 将为你运行某些作业提供更多控制。

# 推荐引擎

构建一个大型推荐引擎是测试你大数据技能的好方法。Mark Litwintschik 的一篇优秀的博客文章介绍了一个使用 Apache Spark（一种大数据技术）的引擎：[`tech.marksblogg.com/recommendation-engine-spark-python.html`](http://tech.marksblogg.com/recommendation-engine-spark-python.html)

# W.I.L.L

URL: [`github.com/ironman5366/W.I.L.L`](https://github.com/ironman5366/W.I.L.L)

非常大的项目！

这个开源个人助理可以成为你的下一个来自《钢铁侠》的 JARVIS。你可以通过数据挖掘技术添加到这个项目中，使其学会执行你经常需要做的某些任务。这并不容易，但潜在的生产力提升是值得的。

# 更多资源

以下是一些非常有用的额外信息资源：

# Kaggle 竞赛

URL: [www.kaggle.com/](http://www.kaggle.com/)

Kaggle 定期举办数据挖掘竞赛，通常伴有现金奖励。

在 Kaggle 竞赛中测试你的技能是快速学习如何处理真实世界数据挖掘问题的好方法。论坛很棒，共享环境——在比赛中，你经常会看到排名前 10 的参赛者发布的代码！

# Coursera

URL: [www.coursera.org](http://www.coursera.org)

Coursera 包含许多关于数据挖掘和数据科学的课程。许多课程都是专业化的，例如大数据和图像处理。一个很好的入门课程是 Andrew Ng 的著名课程：[`www.coursera.org/learn/machine-learning/`](https://www.coursera.org/learn/machine-learning/)。

它比这更复杂一些，对于感兴趣的读者来说是一个很好的下一步。

对于神经网络，可以查看这个课程：[`www.coursera.org/course/neuralnets`](https://www.coursera.org/course/neuralnets)。

如果你完成了所有这些，可以尝试在 [`www.coursera.org/course/pgm`](https://www.coursera.org/course/pgm) 上学习概率图模型课程。
