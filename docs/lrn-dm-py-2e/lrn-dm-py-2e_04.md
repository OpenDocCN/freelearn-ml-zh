# 第四章：使用亲和力分析推荐电影

在本章中，我们将探讨**亲和力分析**，该分析用于确定对象何时频繁地一起出现。这通常也被称为市场篮子分析，因为这是一种常见的用例——确定在商店中频繁一起购买的商品。

在第三章*，使用决策树预测体育比赛胜者*中，我们将对象作为焦点，并使用特征来描述该对象。在本章中，数据具有不同的形式。我们有交易，其中感兴趣的物体（在本章中为电影）以某种方式被用于这些交易中。目标是发现对象何时同时出现。如果我们想找出两部电影是否被同一评论家推荐，我们可以使用亲和力分析。

本章的关键概念如下：

+   亲和力分析用于产品推荐

+   使用 Apriori 算法进行特征关联挖掘

+   推荐系统和固有的挑战

+   稀疏数据格式及其使用方法

# 亲和力分析

亲和力分析是确定对象以相似方式使用的任务。在前一章中，我们关注的是对象本身是否相似——在我们的案例中是游戏在本质上是否相似。亲和力分析的数据通常以交易的形式描述。直观地说，这来自商店的交易——通过确定对象何时一起购买，作为向用户推荐他们可能购买的产品的方式。

然而，亲和力分析可以应用于许多不使用这种意义上的交易的流程：

+   欺诈检测

+   客户细分

+   软件优化

+   产品推荐

亲和力分析通常比分类更具探索性。至少，我们通常只是简单地排名结果并选择前五项推荐（或某个其他数字），而不是期望算法给出一个特定的答案。

此外，我们通常没有我们期望的完整数据集来完成许多分类任务。例如，在电影推荐中，我们有不同人对不同电影的评论。然而，我们几乎不可能让每个评论家都评论我们数据集中的所有电影。这给亲和力分析留下了一个重要且困难的问题。如果一个评论家没有评论一部电影，这是否表明他们不感兴趣（因此不会推荐）或者只是他们还没有评论？

思考数据集中的差距可以导致这样的问题。反过来，这可能导致有助于提高你方法有效性的答案。作为一个初露头角的数据挖掘者，知道你的模型和方法需要改进的地方是创造出色结果的关键。

# 亲和力分析算法

在第一章*《数据挖掘入门》*中，我们介绍了一种基本的关联分析方法，它测试了所有可能的规则组合。我们计算了每个规则的置信度和支持度，这反过来又允许我们根据规则进行排序，以找到最佳规则。

然而，这种方法并不高效。我们在第一章*《数据挖掘入门》*中的数据集只有五个销售项目。我们可以预期即使是小型商店也会有数百个销售项目，而许多在线商店会有数千（甚至数百万！）项目。使用我们之前在第一章*《数据挖掘入门》*中提到的简单规则创建方法，这些规则计算所需的时间会呈指数增长。随着我们添加更多项目，计算所有规则所需的时间增长得更快。具体来说，可能的总规则数是*2n - 1*。对于五个项目的数据集，有 31 个可能的规则。对于十个项目，这个数字是 1023。对于仅仅 100 个项目，这个数字有 30 位。即使计算能力的急剧增加也无法跟上在线存储项目数量的增长。因此，我们需要更智能的算法，而不是更努力工作的计算机。

关联分析的经典算法被称为**Apriori 算法**。它解决了在数据库中创建频繁项集（称为**频繁项集**）的指数级问题。一旦发现这些频繁项集，创建关联规则就变得简单，我们将在本章后面看到这一点。

Apriori 背后的直觉既简单又巧妙。首先，我们确保规则在数据集中有足够的支持度。定义最小支持度是 Apriori 的关键参数。为了构建频繁项集，我们结合较小的频繁项集。对于项集（A，B）要有至少 30%的支持度，A 和 B 必须在数据库中至少出现 30 次。这一属性也适用于更大的集合。对于一个项集（A，B，C，D）要被认为是频繁的，集合（A，B，C）也必须是频繁的（同样，D 也必须是频繁的）。

这些频繁项集可以构建，而不频繁的可能项集（其中有很多）将永远不会被测试。这在新规则测试中节省了大量的时间，因为频繁项集的数量预计将远少于可能项集的总数。

其他关联分析的示例算法基于这个或类似的概念，包括**Eclat**和**FP-growth**算法。数据挖掘文献中有许多对这些算法的改进，进一步提高了方法的效率。在本章中，我们将重点关注基本的 Apriori 算法。

# 总体方法

为了进行关联规则挖掘以进行亲和力分析，我们首先使用 Apriori 算法生成频繁项集。接下来，我们通过测试那些频繁项集中前提和结论的组合来创建关联规则（例如，*如果一个人推荐了电影 X，他们也会推荐电影 Y*）。

1.  在第一阶段，Apriori 算法需要一个值来表示项集需要达到的最小支持度，才能被认为是频繁的。任何支持度低于这个值的项集都不会被考虑。

将这个最小支持度设置得太低会导致 Apriori 测试更多的项集，从而减慢算法的速度。设置得太高会导致考虑的频繁项集更少。

1.  在第二阶段，在频繁项集被发现之后，基于它们的置信度来测试关联规则。我们可以选择一个最小的置信度水平，返回的规则数量，或者简单地返回所有规则并让用户决定如何处理它们。

在本章中，我们只返回高于给定置信度水平的规则。因此，我们需要设置我们的最小置信度水平。设置得太低会导致具有高支持度但不太准确的规则。设置得更高将导致只返回更准确的规则，但总体上发现的规则更少。

# 处理电影推荐问题

产品推荐是一个庞大的产业。在线商店通过推荐其他可能购买的产品来向上销售给客户。做出更好的推荐可以带来更好的销售业绩。当在线购物每年向数百万客户销售时，通过向这些客户销售更多商品，就有大量的潜在利润可赚。

产品推荐，包括电影和书籍，已经研究了许多年；然而，当 Netflix 在 2007 年至 2009 年期间举办 Netflix Prize 时，该领域得到了显著的发展。这次比赛旨在确定是否有人能比 Netflix 目前所做的更好预测用户的电影评分。奖项授予了一个团队，他们的表现比当前解决方案高出 10%以上。虽然这种改进可能看起来并不大，但这样的改进将为 Netflix 在接下来的几年中带来数百万美元的收益，因为更好的电影推荐。

# 获取数据集

自从 Netflix Prize 启动以来，明尼苏达大学的 Grouplens 研究小组已经发布了几个常用于测试该领域算法的数据集。他们发布了多个电影评分数据集的不同版本，大小不同。有一个版本有 10 万条评论，一个版本有 100 万条评论，还有一个版本有 1000 万条评论。

数据集可以从[`grouplens.org/datasets/movielens/`](http://grouplens.org/datasets/movielens/)获取，本章我们将使用的是*MovieLens 100K 数据集*（包含 10 万条评论）。下载此数据集并将其解压到您的数据文件夹中。启动一个新的 Jupyter Notebook，并输入以下代码：

```py
import os
import pandas as pd
data_folder = os.path.join(os.path.expanduser("~"), "Data", "ml-100k")
ratings_filename = os.path.join(data_folder, "u.data")

```

确保变量`ratings_filename`指向解压文件夹中的 u.data 文件。

# 使用 pandas 加载

`MovieLens`数据集状况良好；然而，与`pandas.read_csv`的默认选项相比，我们需要做一些更改。首先，数据是以制表符分隔的，而不是逗号。其次，没有标题行。这意味着文件中的第一行实际上是数据，我们需要手动设置列名。

在加载文件时，我们将分隔符参数设置为制表符，告诉 pandas 不要将第一行作为标题读取（使用`header=None`），并使用给定的值设置列名。让我们看看以下代码：

```py
all_ratings = pd.read_csv(ratings_filename, delimiter="t", header=None, names
            = ["UserID", "MovieID", "Rating", "Datetime"])

```

虽然我们本章不会使用它，但您可以使用以下行正确解析日期时间戳。评论的日期对于推荐预测可能是一个重要特征，因为一起评分的电影通常比单独评分的电影有更相似的排名。考虑到这一点可以显著提高模型的效果。

```py
all_ratings["Datetime"] = pd.to_datetime(all_ratings['Datetime'], unit='s')

```

您可以通过在新的单元格中运行以下代码来查看前几条记录：

```py
all_ratings.head()

```

结果将类似于以下内容：

|  | UserID | MovieID | Rating | Datetime |
| --- | --- | --- | --- | --- |
| 0 | 196 | 242 | 3 | 1997-12-04 15:55:49 |
| 1 | 186 | 302 | 3 | 1998-04-04 19:22:22 |
| 2 | 22 | 377 | 1 | 1997-11-07 07:18:36 |
| 3 | 244 | 51 | 2 | 1997-11-27 05:02:03 |
| 4 | 166 | 346 | 1 | 1998-02-02 05:33:16 |

# 稀疏数据格式

此数据集是稀疏格式。每一行可以被视为之前章节中使用的大型特征矩阵中的一个单元格，其中行是用户，列是单独的电影。第一列将是每个用户对第一部电影的评论，第二列将是每个用户对第二部电影的评论，依此类推。

该数据集中大约有 1,000 个用户和 1,700 部电影，这意味着完整的矩阵会相当大（近 200 万条记录）。我们可能会遇到在内存中存储整个矩阵的问题，并且对其进行计算会相当麻烦。然而，这个矩阵具有大多数单元格为空的性质，也就是说，大多数用户对大多数电影没有评论。尽管如此，用户编号 213 对电影编号 675 的评论不存在，以及其他大多数用户和电影的组合也是如此。

这里给出的格式代表完整的矩阵，但以更紧凑的方式呈现。第一行表示用户编号 196 在 1997 年 12 月 4 日对电影编号 242 进行了评分，评分为 3（满分五分）。

任何不在数据库中的用户和电影的组合都被假定为不存在。这节省了大量的空间，与在内存中存储一串零相比。这种格式称为稀疏矩阵格式。一般来说，如果你预计你的数据集中有 60%或更多的数据为空或为零，稀疏格式将占用更少的空间来存储。

在稀疏矩阵上进行计算时，我们通常不会关注我们没有的数据——比较所有的零。我们通常关注我们有的数据，并比较这些数据。

# 理解 Apriori 算法及其实现

本章的目标是产生以下形式的规则：*如果一个人推荐了这组电影，他们也会推荐这部电影*。我们还将讨论扩展，其中推荐一组电影的人可能会推荐另一部特定的电影。

要做到这一点，我们首先需要确定一个人是否推荐了一部电影。我们可以通过创建一个新的特征“赞同”，如果该人对电影给出了好评，则为 True：

```py
all_ratings["Favorable"] = all_ratings["Rating"] > 3

```

我们可以通过查看数据集来查看新功能：

```py
all_ratings[10:15]

```

|  | 用户 ID | 电影 ID | 评分 | 日期时间 | 赞同 |
| --- | --- | --- | --- | --- | --- |
| 10 | 62 | 257 | 2 | 1997-11-12 22:07:14 | False |
| 11 | 286 | 1014 | 5 | 1997-11-17 15:38:45 | True |
| 12 | 200 | 222 | 5 | 1997-10-05 09:05:40 | True |
| 13 | 210 | 40 | 3 | 1998-03-27 21:59:54 | False |
| 14 | 224 | 29 | 3 | 1998-02-21 23:40:57 | False |

我们将采样我们的数据集以形成训练数据。这也帮助减少了要搜索的数据集的大小，使 Apriori 算法运行得更快。我们获取了前 200 个用户的所有评论：

```py
ratings = all_ratings[all_ratings['UserID'].isin(range(200))]

```

接下来，我们可以创建一个只包含样本中好评的评论文本的数据集：

```py
favorable_ratings_mask = ratings["Favorable"]
favorable_ratings = ratings[favorable_ratings_mask]

```

我们将在用户的好评中搜索我们的项集。因此，我们接下来需要的是每个用户给出的好评电影。我们可以通过按`UserID`对数据集进行分组并遍历每个组中的电影来计算这一点：

```py
favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings.groupby("UserID")["MovieID"])

```

在前面的代码中，我们将值存储为`frozenset`，这样我们可以快速检查用户是否对电影进行了评分。

对于这种类型的操作，集合比列表快得多，我们将在后面的代码中使用它们。

最后，我们可以创建一个`DataFrame`，告诉我们每部电影被好评的频率：

```py
num_favorable_by_movie = ratings[["MovieID", "Favorable"]].groupby("MovieID").sum()

```

通过运行以下代码，我们可以看到前五部电影：

```py
num_favorable_by_movie.sort_values(by="Favorable", ascending=False).head()

```

让我们看看前五部电影列表。我们现在只有 ID，将在本章后面获取它们的标题。

| 电影 ID | 赞同 |
| --- | --- |
| 50 | 100 |
| 100 | 89 |
| 258 | 83 |
| 181 | 79 |
| 174 | 74 |

# 探索 Apriori 算法的基本原理

Apriori 算法是我们亲和力分析方法的一部分，专门处理在数据中寻找频繁项集的问题。Apriori 的基本程序是从先前发现的频繁项集中构建新的候选项集。这些候选集被测试以查看它们是否频繁，然后算法按以下方式迭代：

1.  通过将每个项目放置在其自己的项目集中来创建初始频繁项目集。在此步骤中仅使用至少具有最小支持度的项目。

1.  从最近发现的频繁项目集中创建新的候选项目集，通过找到现有频繁项目集的超集。

1.  所有候选项目集都会被测试以确定它们是否频繁。如果一个候选项目集不是频繁的，则将其丢弃。如果没有从这个步骤中产生新的频繁项目集，则转到最后一步。

1.  存储新发现的频繁项目集并转到第二步。

1.  返回所有发现的频繁项目集。

此过程在以下工作流程中概述：

![](img/B06162_04_01.jpg)

# 实现 Apriori 算法

在 Apriori 的第一轮迭代中，新发现的项集长度将为 2，因为它们将是第一步中创建的初始项集的超集。在第二轮迭代（应用第四步并返回到第二步之后），新发现的项集长度将为 3。这使我们能够快速识别新发现的项集，正如在第二步中所需的那样。

我们可以在字典中存储发现频繁项目集，其中键是项目集的长度。这允许我们快速访问给定长度的项目集，以及通过以下代码帮助快速访问最近发现的频繁项目集：

```py
frequent_itemsets = {}

```

我们还需要定义一个项目集被认为是频繁所需的最小支持度。此值基于数据集选择，但尝试不同的值以查看它如何影响结果。尽管如此，我建议每次只改变 10%，因为算法运行所需的时间将显著不同！让我们设置一个最小支持度值：

```py
min_support = 50

```

要实现 Apriori 算法的第一步，我们为每部电影单独创建一个项目集，并测试该项目集是否频繁。我们使用`frozenset`**，**因为它们允许我们在稍后执行更快的基于集合的操作，并且它们还可以用作计数字典中的键（普通集合不能）。

让我们看看以下`frozenset`代码的示例：

```py
frequent_itemsets[1] = dict((frozenset((movie_id,)), row["Favorable"])
 for movie_id, row in num_favorable_by_movie.iterrows()
 if row["Favorable"] > min_support)

```

为了提高效率，我们将第二步和第三步一起实现，通过创建一个函数来执行这些步骤，该函数接受新发现的频繁项目集，创建超集，然后测试它们是否频繁。首先，我们设置函数以执行这些步骤：

```py
from collections import defaultdict

def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])

```

为了遵循我们尽可能少读取数据的经验法则，我们每次调用此函数时只遍历数据集一次。虽然在这个实现中这不是很重要（与平均计算机相比，我们的数据集相对较小），**单次遍历**是对于更大应用的良好实践。

让我们详细看看这个函数的核心。我们遍历每个用户，以及之前发现的每个项集，然后检查它是否是当前存储在`k_1_itemsets`中的评论集的子集（注意，这里的 k_1 意味着*k-1*）。如果是，这意味着用户已经评论了项集中的每部电影。这是通过`itemset.issubset(reviews)`这一行完成的。

然后，我们可以遍历用户评论的每部单独的电影（那些尚未在项集中），通过将项集与新电影结合来创建超集，并在我们的计数字典中记录我们看到了这个超集。这些都是这个*k*值的候选频繁项集。

我们通过测试候选项集是否有足够的支持被认为是频繁的来结束我们的函数，并只返回那些支持超过我们的`min_support`值的项集。

这个函数构成了我们 Apriori 实现的核心，我们现在创建一个循环，遍历更大算法的步骤，随着*k*从 1 增加到最大值，存储新的项集。在这个循环中，k 代表即将发现的频繁项集的长度，允许我们通过在我们的频繁项集字典中使用键*k-1*来访问之前发现的最频繁的项集。我们通过它们的长度创建频繁项集并将它们存储在我们的字典中。让我们看看代码：

```py
for k in range(2, 20):
    # Generate candidates of length k, using the frequent itemsets of length k-1
    # Only store the frequent itemsets
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users,
                                                   frequent_itemsets[k-1], min_support)
    if len(cur_frequent_itemsets) == 0:
        print("Did not find any frequent itemsets of length {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
        sys.stdout.flush()
        frequent_itemsets[k] = cur_frequent_itemsets

```

如果我们找到了频繁项集，我们打印一条消息来让我们知道循环将再次运行。如果没有，我们停止迭代，因为没有频繁项集对于*k+1*，如果当前*k*值没有频繁项集，因此我们完成算法。

我们使用`sys.stdout.flush()`来确保打印输出在代码仍在运行时发生。有时，特别是在某些单元格的大循环中，打印输出可能直到代码完成才发生。以这种方式刷新输出确保打印输出在我们想要的时候发生，而不是当界面决定可以分配时间打印的时候。但是，不要过于频繁地刷新——刷新操作（以及正常的打印）都会带来计算成本，这会减慢程序的速度。

你现在可以运行上述代码。

上述代码返回了大约 2000 个不同长度的频繁项集。你会注意到，随着长度的增加，项集的数量先增加后减少。这是因为可能规则的数目在增加。过了一段时间，大量组合不再有必要的支持被认为是频繁的。这导致数量减少。这种减少是 Apriori 算法的优点。如果我们搜索所有可能的项集（而不仅仅是频繁项集的超集），我们将需要搜索成千上万的项集来查看它们是否频繁。

即使这种缩小没有发生，当发现所有电影的组合规则时，算法将达到绝对结束。因此，Apriori 算法将始终终止。

运行此代码可能需要几分钟，如果你有较旧的硬件，可能需要更长的时间。如果你发现运行任何代码示例有困难，可以考虑使用在线云服务提供商以获得额外的速度。有关使用云进行工作的详细信息，请参阅附录，下一步。

# 提取关联规则

Apriori 算法完成后，我们将有一个频繁项集的列表。这些不是精确的关联规则，但它们可以很容易地转换为这些规则。频繁项集是一组具有最小支持度的项，而关联规则有一个前提和结论。这两个的数据是相同的。

我们可以通过将项集中的一部电影作为结论，并将其他电影作为前提来从频繁项集中创建一个关联规则。这将形成以下形式的规则：*如果一个评论家推荐了前提中的所有电影，他们也会推荐结论电影*。

对于每个项集，我们可以通过将每部电影设置为结论，将剩余的电影作为前提来生成多个关联规则。

在代码中，我们首先通过遍历每个长度的发现频繁项集，从每个频繁项集中生成所有规则的列表。然后，我们遍历项集中的每一部电影作为结论。

```py
candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))

```

这将返回一个非常大的候选规则数。我们可以通过打印列表中的前几条规则来查看一些：

```py
print(candidate_rules[:5])

```

生成的输出显示了获得的规则：

```py
[(frozenset({79}), 258), (frozenset({258}), 79), (frozenset({50}), 64), (frozenset({64}), 50), (frozenset({127}), 181)]

```

在这些规则中，第一部分（`frozenset`）是前提中的电影列表，而它后面的数字是结论。在第一种情况下，如果一个评论家推荐了电影 79，他们也很可能推荐电影 258。

接下来，我们计算这些规则中每个规则的置信度。这与第一章*《数据挖掘入门》*中的操作非常相似，唯一的区别是那些必要的更改，以便使用新的数据格式进行计算。

计算置信度的过程首先是通过创建字典来存储我们看到前提导致结论（规则的正确示例）和它没有发生（规则的错误示例）的次数。然后，我们遍历所有评论和规则，确定规则的前提是否适用，如果适用，结论是否准确。

```py
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

```

然后，我们通过将正确计数除以规则被看到的总次数来计算每个规则的置信度：

```py
rule_confidence = {candidate_rule:
                    (correct_counts[candidate_rule] / float(correct_counts[candidate_rule] +  
                      incorrect_counts[candidate_rule]))
                  for candidate_rule in candidate_rules}

```

现在，我们可以通过排序这个置信度字典并打印结果来打印前五条规则：

```py
from operator import itemgetter
sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_confidence[index][0]
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")

```

生成的打印输出只显示电影 ID，没有电影名称的帮助并不太有用。数据集附带一个名为 u.items 的文件，该文件存储电影名称及其对应的 MovieID（以及其他信息，如类型）。

我们可以使用 pandas 从这个文件中加载标题。有关文件和类别的更多信息可在随数据集提供的 README 文件中找到。文件中的数据是 CSV 格式，但数据由|符号分隔；它没有标题

并且编码设置很重要。列名在 README 文件中找到。

```py
movie_name_filename = os.path.join(data_folder, "u.item")
movie_name_data = pd.read_csv(movie_name_filename, delimiter="|", header=None,
                              encoding = "mac-roman")
movie_name_data.columns = ["MovieID", "Title", "Release Date", "Video Release", "IMDB", "<UNK>",
                           "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                           "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",   
                           "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

```

获取电影标题是一个重要且常用的步骤，因此将其转换为函数是有意义的。我们将创建一个函数，该函数将从其 MovieID 返回电影标题，从而避免每次都查找的麻烦。让我们看看代码：

```py
def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data["MovieID"] == movie_id]["Title"]
    title = title_object.values[0]
    return title

```

在一个新的 Jupyter Notebook 单元中，我们调整了之前用于打印最佳规则的代码，以包括标题：

```py
for index in range(5):
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")

```

结果更易于阅读（仍然有一些问题，但现在我们可以忽略它们）：

```py
Rule #1
Rule: If a person recommends Shawshank Redemption, The (1994), Silence of the Lambs, The (1991), Pulp Fiction (1994), Star Wars (1977), Twelve Monkeys (1995) they will also recommend Raiders of the Lost Ark (1981)
 - Confidence: 1.000

Rule #2
Rule: If a person recommends Silence of the Lambs, The (1991), Fargo (1996), Empire Strikes Back, The (1980), Fugitive, The (1993), Star Wars (1977), Pulp Fiction (1994) they will also recommend Twelve Monkeys (1995)
 - Confidence: 1.000

Rule #3
Rule: If a person recommends Silence of the Lambs, The (1991), Empire Strikes Back, The (1980), Return of the Jedi (1983), Raiders of the Lost Ark (1981), Twelve Monkeys (1995) they will also recommend Star Wars (1977)
 - Confidence: 1.000

Rule #4
Rule: If a person recommends Shawshank Redemption, The (1994), Silence of the Lambs, The (1991), Fargo (1996), Twelve Monkeys (1995), Empire Strikes Back, The (1980), Star Wars (1977) they will also recommend Raiders of the Lost Ark (1981)
 - Confidence: 1.000

Rule #5
Rule: If a person recommends Shawshank Redemption, The (1994), Toy Story (1995), Twelve Monkeys (1995), Empire Strikes Back, The (1980), Fugitive, The (1993), Star Wars (1977) they will also recommend Return of the Jedi (1983)
 - Confidence: 1.000

```

# 评估关联规则

在广义上，我们可以使用与分类相同的概念来评估关联规则。我们使用未用于训练的数据测试集，并根据它们在这个测试集中的性能来评估我们发现的规则。

要做到这一点，我们将计算测试集置信度，即每个规则在测试集中的置信度。在这种情况下，我们不会应用正式的评估指标；我们只是检查规则并寻找好的例子。

正式评估可能包括通过确定用户是否对给定电影给予好评的预测准确性来进行分类准确率。在这种情况下，如下所述，我们将非正式地查看规则以找到那些更可靠的规则：

1.  首先，我们提取测试数据集，这是我们未在训练集中使用的所有记录。我们使用了前 200 个用户（按 ID 值）作为训练集，我们将使用其余所有用户作为测试数据集。与训练集一样，我们还将获取该数据集中每个用户的正面评论。让我们看看代码：

```py
test_dataset = all_ratings[~all_ratings['UserID'].isin(range(200))]
test_favorable = test_dataset[test_dataset["Favorable"]]
test_favorable_by_users = dict((k, frozenset(v.values)) for k, v in 
                               test_favorable.groupby("UserID")["MovieID"])

```

1.  然后，我们计算前提导致结论的正确实例数，就像我们之前做的那样。这里唯一的变化是使用测试数据而不是训练数据。让我们看看代码：

```py
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1

```

1.  接下来，我们计算每个规则的置信度，并按此排序。让我们看看代码：

```py
test_confidence = {candidate_rule:
                             (correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule]))
                             for candidate_rule in rule_confidence}
sorted_test_confidence = sorted(test_confidence.items(), key=itemgetter(1), reverse=True)

```

1.  最后，我们以标题而不是电影 ID 的形式打印出最佳关联规则：

```py
for index in range(10):
    print("Rule #{0}".format(index + 1))
    premise, conclusion = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Train Confidence: {0:.3f}".format(rule_confidence.get((premise, conclusion), -1)))
    print(" - Test Confidence: {0:.3f}".format(test_confidence.get((premise, conclusion), -1)))
    print("")

```

现在，我们可以看到哪些规则在新的未见数据中最为适用：

```py
Rule #1
Rule: If a person recommends Shawshank Redemption, The (1994), Silence of the Lambs, The (1991), Pulp Fiction (1994), Star Wars (1977), Twelve Monkeys (1995) they will also recommend Raiders of the Lost Ark (1981)
 - Train Confidence: 1.000
 - Test Confidence: 0.909

Rule #2
Rule: If a person recommends Silence of the Lambs, The (1991), Fargo (1996), Empire Strikes Back, The (1980), Fugitive, The (1993), Star Wars (1977), Pulp Fiction (1994) they will also recommend Twelve Monkeys (1995)
 - Train Confidence: 1.000
 - Test Confidence: 0.609

Rule #3
Rule: If a person recommends Silence of the Lambs, The (1991), Empire Strikes Back, The (1980), Return of the Jedi (1983), Raiders of the Lost Ark (1981), Twelve Monkeys (1995) they will also recommend Star Wars (1977)
 - Train Confidence: 1.000
 - Test Confidence: 0.946

Rule #4
Rule: If a person recommends Shawshank Redemption, The (1994), Silence of the Lambs, The (1991), Fargo (1996), Twelve Monkeys (1995), Empire Strikes Back, The (1980), Star Wars (1977) they will also recommend Raiders of the Lost Ark (1981)
 - Train Confidence: 1.000
 - Test Confidence: 0.971

Rule #5
Rule: If a person recommends Shawshank Redemption, The (1994), Toy Story (1995), Twelve Monkeys (1995), Empire Strikes Back, The (1980), Fugitive, The (1993), Star Wars (1977) they will also recommend Return of the Jedi (1983)
 - Train Confidence: 1.000
 - Test Confidence: 0.900

```

例如，第二个规则在训练数据中具有完美的置信度，但在测试数据中只有 60%的案例是准确的。前 10 条规则中的许多其他规则在测试数据中具有高置信度，这使得它们成为制定推荐的好规则。

你可能还会注意到，这些电影往往非常受欢迎且是优秀的电影。这为我们提供了一个基准算法，我们可以将其与之比较，即不是尝试进行个性化推荐，而是推荐最受欢迎的电影。尝试实现这个算法——Apriori 算法是否优于它，以及优于多少？另一个基准可能是简单地从同一类型中随机推荐电影。

如果你正在查看其余的规则，其中一些将具有-1 的测试置信度。置信值总是在 0 和 1 之间。这个值表示特定的规则根本未在测试数据集中找到。

# 摘要

在本章中，我们进行了亲和力分析，以便根据大量评论者推荐电影。我们分两个阶段进行。首先，我们使用 Apriori 算法在数据中找到频繁项集。然后，我们从这些项集中创建关联规则。

由于数据集的大小，使用 Apriori 算法是必要的。在第一章*，数据挖掘入门*中，我们使用了暴力方法，这种方法在计算那些用于更智能方法的规则所需的时间上呈指数增长。这是数据挖掘中的一种常见模式：对于小数据集，我们可以以暴力方式解决许多问题，但对于大数据集，则需要更智能的算法来应用这些概念。

我们在我们的数据的一个子集上进行了训练，以找到关联规则，然后在这些规则的其余数据——测试集上进行了测试。根据我们之前章节的讨论，我们可以将这个概念扩展到使用交叉验证来更好地评估规则。这将导致对每个规则质量的更稳健的评估。

为了进一步探讨本章的概念，研究哪些电影获得了很高的总体评分（即有很多推荐），但没有足够的规则来向新用户推荐它们。你将如何修改算法来推荐这些电影？

到目前为止，我们所有的数据集都是用特征来描述的。然而，并非所有数据集都是以这种方式*预先定义*的。在下一章中，我们将探讨 scikit-learn 的转换器（它们在*第三章，使用决策树预测体育比赛赢家*中介绍过）作为从数据中提取特征的方法。我们将讨论如何实现我们自己的转换器，扩展现有的转换器，以及我们可以使用它们实现的概念。
