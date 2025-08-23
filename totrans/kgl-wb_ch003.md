# 2 Makridakis竞赛：M5在Kaggle上的准确性和不确定性

## 加入我们的Discord书籍社区

[https://packt.link/EarlyAccessCommunity](https://packt.link/EarlyAccessCommunity)

![](img/file1.png)

自1982年以来，Spyros Makridakis ([https://mofc.unic.ac.cy/dr-spyros-makridakis/https://mofc.unic.ac.cy/dr-spyros-makridakis/](https://mofc.unic.ac.cy/dr-spyros-makridakis/https://mofc.unic.ac.cy/dr-spyros-makridakis/)) 一直让来自世界各地的研究团队参与预测挑战，称为M竞赛，以便对现有和新预测方法在不同预测问题上的有效性进行比较。因此，M竞赛始终对学术界和实践者完全开放。这些竞赛可能是预测社区中被引用和参考最多的活动，并且它们始终突出了预测方法领域的最新进展。每个先前的M竞赛都为研究人员和实践者提供了有用的数据来训练和测试他们的预测工具，同时也提供了一系列发现和途径，这些发现和途径正在彻底改变预测的方式。

最近举行的M5竞赛（M6竞赛正在撰写本章时进行）在Kaggle上举行，它证明了在尝试解决一系列零售产品销量预测问题时，梯度提升方法的有用性尤为重要。在本章中，我们专注于准确性赛道，处理了Kaggle竞赛中的一个时间序列问题，通过复制一个顶级、简单且最清晰的解决方案之一，我们旨在为读者提供代码和思路，以成功应对未来可能出现在Kaggle上的任何预测竞赛。

+   除了竞赛页面，我们在《国际预测杂志》的以下论文中找到了关于竞赛及其动态的大量信息：

+   Makridakis, Spyros, Evangelos Spiliotis, 和 Vassilios Assimakopoulos. *《M5竞赛：背景、组织和实施》*。《国际预测杂志》（2021年）。

+   Makridakis, Spyros, Evangelos Spiliotis, 和 Vassilios Assimakopoulos。"M5准确性竞赛：结果、发现和结论"《国际预测杂志》（2022年）。

+   Makridakis, Spyros, 等人。"《M5不确定性竞赛：结果、发现和结论》"《国际预测杂志》（2021年）。

## 理解竞争和数据

竞赛从2020年3月持续到6月，超过7,000名参与者参加了在Kaggle上的竞赛。组织者将其分为两个独立的赛道，一个用于点预测（准确性赛道），另一个用于在不同置信区间估计可靠值（不确定性赛道）。

沃尔玛提供了数据。它包括42,840个每日销售时间序列，这些序列按部门、类别和店铺分层排列，分布在三个美国州（这些序列彼此之间有一定的相关性）。除了销售数据，沃尔玛还提供了伴随信息（外生变量，通常在预测问题中不常提供），例如商品价格、一些日历信息、相关的促销活动或其他影响销售的事件。

除了Kaggle，数据以及之前M竞赛的数据集都可以在以下地址找到：[https://forecasters.org/resources/time-series-data/](https://forecasters.org/resources/time-series-data/)。

竞赛的一个有趣方面是它处理了快速消费品和慢速消费品销售，有许多例子展示了最新的间歇性销售（销售通常是零，但在一些罕见情况下）。虽然间歇性序列在许多行业中很常见，但对于许多从业者来说，在预测中仍然是一个具有挑战性的案例。

竞赛时间表被安排为两部分。在第一部分，从2020年3月初到6月1日，参赛者可以在1,913天范围内的任何一天训练模型，并在公共测试集（从1,914天到1,941天）上对其提交进行评分。在那之后，直到7月1日的竞赛结束，公共测试集作为训练集的一部分提供，允许参与者调整他们的模型以预测从1,942天到1969天（28天的时间窗口，即四周）。在那个时期，提交在排行榜上没有得分。

竞赛这种安排背后的比例是为了让团队最初能在排行榜上测试他们的模型，并为他们提供在笔记本和讨论中分享最佳表现方法的基础。在第一阶段之后，组织者希望避免排行榜被用于过度拟合或模型超参数调整的目的，他们希望模拟一种预测情况，就像在现实世界中发生的那样。此外，只选择一个提交作为最终提交的要求，反映了现实世界的相同必要性（在现实世界中，你不能使用两个不同的模型预测，然后在之后选择最适合你的一个）。

关于数据，我们提到数据由沃尔玛提供，并代表美国市场：它源自加利福尼亚州、威斯康星州和德克萨斯州的 10 家商店。具体来说，数据由 3,049 产品的销售组成，分为三个类别（爱好、食品和家庭），每个类别可以进一步分为 7 个部门。这种层次结构无疑是一个挑战，因为你可以在美国市场、州市场、单个商店、产品类别、类别部门和最终特定产品级别建模销售动态。所有这些级别也可以组合成不同的汇总，这是第二轨道，即不确定性轨道中需要预测的内容：

| **级别 ID** | **级别描述** | **汇总级别** | **序列数量** |
| --- | --- | --- | --- |
| 1 | 所有产品，按所有商店和州汇总 | 总计 | 1 |
| 2 | 所有产品，按每个州汇总 | 州 | 3 |
| 3 | 所有产品，按每个商店汇总 | 商店 | 10 |
| 4 | 所有产品，按每个类别汇总 | 类别 | 3 |
| 5 | 所有产品，按每个部门汇总 | 部门 | 7 |
| 6 | 所有产品，按每个州和类别汇总 | 州-类别 | 9 |
| 7 | 所有产品，按每个州和部门汇总 | 州-部门 | 21 |
| 8 | 所有产品，按每个商店和类别汇总 | 商店-类别 | 30 |
| 9 | 所有产品，按每个商店和部门汇总 | 商店-部门 | 70 |
| 10 | 每个产品，按所有商店/州汇总 | 产品 | 3,049 |
| 11 | 每个产品，按每个州汇总 | 产品-州 | 9,147 |
| 12 | 每个产品，按每个商店汇总 | 产品-商店 | 30,490 |
|  |  | 总计 | 42,840 |

从时间角度来看，粒度是每日销售记录，覆盖了从 2011 年 1 月 29 日到 2016 年 6 月 19 日的期间，总计 1,969 天，其中 1,913 天用于训练，28 天用于验证 - 公开排行榜 - 28 天用于测试 - 私人排行榜。实际上，在零售行业中，28 天的预测范围被认为是处理大多数商品的库存和重新订购操作的正确范围。

让我们检查一下比赛中收到的不同数据。你将获得 `sales_train_evaluation.csv`、`sell_prices.csv` 和 `calendar.csv`。其中包含时间序列的是 `sales_train_evaluation.csv`。它由作为标识符的字段（`item_id`、`dept_id`、`cat_id`、`store_id` 和 `state_id`）以及从 `d_1` 到 `d_1941` 的列组成，代表那些天的销售情况：

![图 2.1：sales_train_evaluation.csv 数据](img/file2.png)

图 2.1：sales_train_evaluation.csv 数据

`sell_prices.csv` 包含关于商品价格的信息。这里的难点在于将 `wm_yr_wk`（周标识符）与训练数据中的列连接起来：

![图 2.2：sell_prices.csv 数据](img/file3.png)

图 2.2：sell_prices.csv 数据

最后一个文件，`calendar.csv`，包含可能影响销售的事件相关数据：

![图2.3：calendar.csv数据](img/file4.png)

图2.3：calendar.csv数据

同样，主要困难似乎在于将数据与训练表中的列连接起来。无论如何，在这里您可以获得一个简单的键来连接列（d字段）与`wm_yr_wk`。此外，在表中，我们表示了可能发生在特定日期的不同事件，以及SNAP日，这是特别的日子，在这些日子里，可以使用的营养援助福利补充营养援助计划（SNAP）。

## 理解评估指标

准确度竞赛引入了一个新的评估指标：加权均方根缩放误差（WRMSSE）。该指标评估点预测围绕预测序列实现值平均值的偏差：

![](img/file5.png)

其中：

*n* 是训练样本的长度

*h* 是预测范围（在我们的情况下是*h*=28）

*Y*[*t*] 是时间 *t* 的销售价值，![](img/file6.png) 是时间 *t* 的预测值

在竞赛指南([https://mofc.unic.ac.cy/m5-competition/](https://mofc.unic.ac.cy/m5-competition/))中，关于WRMSSE，指出：

+   RMSSE的分母仅针对那些正在积极销售的考察产品的时间段进行计算，即，在评估序列观察到的第一个非零需求之后的时期。

+   该度量与规模无关，这意味着它可以有效地用于比较不同规模序列的预测。

+   与其他度量相比，它可以安全地计算，因为它不依赖于可能等于或接近零的值的除法（例如，在*Y*[*t*] = 0时进行的百分比误差，或者当用于缩放的基准误差为零时进行的相对误差）。

+   该度量对正负预测误差、大预测和小预测进行同等惩罚，因此是对称的。

亚历山大·索阿雷（Alexander Soare）在这篇帖子中提供了对这种工作原理的良好解释([https://www.kaggle.com/alexandersoare](https://www.kaggle.com/alexandersoare))：[https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/148273](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/148273)。在转换评估指标后，亚历山大将性能的提高归因于预测误差与日销售价值日变化率之间的比率提高。如果误差与日变化率相同（比率=1），则模型可能并不比基于历史变化的随机猜测好多少。如果您的误差小于这个值，则分数以二次方式（由于平方根）接近零。因此，WRMSSE为0.5对应比率为0.7，WRMSSE为0.25对应比率为0.5。

在竞赛期间，许多尝试不仅将指标用于排行榜的评估，还将其作为目标函数。首先，Tweedie损失（在XGBoost和LightGBM中实现）对于这个问题来说效果相当好，因为它可以处理大多数产品的销售分布的偏斜（其中很多也有间歇性销售，这也被Tweedie损失很好地处理）。泊松和伽马分布可以被认为是Tweedie分布的极端情况：基于参数幂，p，当p=1时得到泊松分布，当p=2时得到伽马分布。这种幂参数实际上是连接分布的均值和方差的粘合剂，通过公式方差 = k*均值**p。使用介于1和2之间的幂值，实际上可以得到泊松和伽马分布的混合，这可以很好地适应竞赛问题。实际上，大多数参与竞赛并使用GBM解决方案的Kagglers都求助于Tweedie损失。

尽管Tweedie方法取得了成功，然而，一些其他Kagglers却发现了一些有趣的方法来实现一个更接近WRMSSE的目标损失，用于他们的模型：

* Martin Kovacevic Buvinic及其非对称损失：[https://www.kaggle.com/code/ragnar123/simple-lgbm-groupkfold-cv/notebook](https://www.kaggle.com/code/ragnar123/simple-lgbm-groupkfold-cv/notebook)

* Timetraveller使用PyTorch Autograd获取梯度和对角线，以实现任何可微分的连续损失函数，并在LighGBM中实现：[https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/152837](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/152837)

## 检验来自Monsaraida的第四名解决方案的想法

对于这次竞赛，有许多解决方案可供选择，主要可以在竞赛Kaggle讨论页面上找到。挑战的前五名方法也已被竞赛组织者自己收集并发布（但有一个因为版权问题）：[https://github.com/Mcompetitions/M5-methods](https://github.com/Mcompetitions/M5-methods)（顺便说一句，重现获奖提交的结果是收集竞赛奖金的前提条件）。

显然，所有在比赛中获得较高排名的Kagglers都使用了LightGBM，作为他们独特的模型类型或在集成/堆叠中，因为其较低的内存使用量和计算速度，这使其在处理和预测大量时间序列时在竞争中具有优势。但还有其他原因导致其成功。与基于ARIMA的经典方法相反，它不需要依赖于自相关分析，以及在具体确定问题中每个单个序列的参数。此外，与基于深度学习的方法相反，它不需要寻找改进复杂的神经网络架构或调整大量超参数。梯度提升方法在时间序列问题中的优势（对于扩展其他梯度提升算法，例如XGBoost）是依赖于特征工程，根据时间滞后、移动平均和序列属性分组创建正确的特征数量。然后选择正确的目标函数并进行一些超参数调整，当时间序列足够长时（对于较短的序列，ARIMA或指数平滑等经典统计方法仍然是推荐的选择），就足以获得优秀的结果。

> 与比赛中深度学习解决方案相比，LightGBM和XGBoost的另一个优势是Tweedie损失，不需要任何特征缩放（深度学习网络对所使用的缩放特别敏感）以及训练速度，这允许在测试特征工程时进行更快的迭代。

在所有这些可用的解决方案中，我们发现由日本计算机科学家Monsaraida（Masanori Miyahara）提出的方案最为有趣。他提出了一种简单直接的方法，在私人排行榜上排名第4，得分为0.53583。该方案仅使用一般特征，没有进行预先选择（如销售统计、日历、价格和标识符）。此外，它使用有限数量的同类型模型，使用LightGBM梯度提升，没有求助于任何类型的混合、递归建模（当预测反馈给其他层次相关的预测或乘数时，即选择常数以更好地拟合测试集）。以下是他在M（[https://github.com/Mcompetitions/M5-methods/tree/master/Code%20of%20Winning%20Methods/A4](https://github.com/Mcompetitions/M5-methods/tree/master/Code%20of%20Winning%20Methods/A4)）的演示解决方案中提出的方案，可以注意到他对待每个商店的每个四周，最终对应于产生40个模型：

![图2.4：Monsaraida关于其解决方案结构的说明](img/file7.png)

图2.4：Monsaraida关于其解决方案结构的说明

由于Monsaraida保持了其解决方案的简单和实用，就像在现实世界的预测项目中一样，在这一章中，我们将尝试通过重构他的代码来复制他的示例，以便在Kaggle笔记本中运行（我们将通过将代码拆分成多个笔记本来处理内存和运行时间限制）。这样，我们旨在为读者提供一个简单而有效的方法，基于梯度提升，来处理预测问题。

## 计算特定日期和时间跨度的预测

复制Monsaraida解决方案的计划是创建一个可由输入参数自定义的笔记本，以便生成训练和测试所需的必要处理数据以及用于预测的LightGBM模型。给定过去的数据，这些模型将被训练以学习预测未来特定天数内的值。通过让每个模型学习预测未来特定周范围内的值，可以获得最佳结果。由于我们必须预测未来最多28天，我们需要一个从未来+1天到未来+7天的模型，然后是另一个能够从未来+8天到未来+14天的模型，再是另一个从未来+15天到+21天的模型，最后是一个能够处理从未来+22天到未来+28天的预测的模型。我们需要为每个时间范围创建一个Kaggle笔记本，因此我们需要四个笔记本。每个这些笔记本都将被训练以预测每个参与竞赛的十个店铺的未来时间跨度。总共，每个笔记本将生成十个模型。所有这些笔记本将共同生成四十个模型，覆盖所有未来范围和所有店铺。

由于我们需要为公共排行榜和私有排行榜都进行预测，因此有必要重复这个过程两次，在1,913天（预测1,914天到1,941天的日子）停止训练以提交公共测试集，以及在1,941天（预测1,942天到1,969天的日子）停止训练以提交私有测试集。

由于基于CPU运行Kaggle笔记本的限制，所有这八个笔记本都可以并行运行（整个过程几乎需要6个半小时）。每个笔记本可以通过其名称与其他笔记本区分开来，包含与最后训练日和前瞻性预测天数相关的参数值。这些笔记本中的一个示例可以在以下链接找到：[https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-7](https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-7)。

现在我们一起来检查代码是如何安排的，以及我们可以从Monsaraida的解决方案中学到什么。

我们首先导入必要的包。您只需注意到，除了NumPy和pandas之外，唯一的数据科学专用包是LightGBM。您可能还会注意到，我们将使用gc（垃圾回收）：这是因为我们需要限制脚本使用的内存量，并且我们经常只是收集和回收未使用的内存。作为这一策略的一部分，我们也经常将模型和数据结构存储到磁盘上，而不是保留在内存中：

[PRE0]

作为限制内存使用的策略的一部分，我们求助于Kaggle书中描述的用于减少pandas DataFrame的函数，最初由Arjan Groen在Zillow比赛中开发（阅读讨论[https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/discussion/291844](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/discussion/291844)）：

[PRE1]

我们继续定义这个解决方案的函数，因为这样有助于将解决方案拆分成更小的部分，并且当您从函数返回时，清理所有使用的变量会更加容易（您只需保留保存到磁盘的内容，并从函数返回）。我们的下一个函数帮助我们加载所有可用的数据并将其压缩：

[PRE2]

在准备获取价格、体积和日历信息相关的数据代码之后，我们继续准备第一个处理函数，该函数将具有创建一个基本的信息表的角色，其中`item_id`、`dept_id`、`cat_id`、`state_id`和`store_id`作为行键，一个日期列和一个包含体积的值列。这是通过使用pandas命令melt([https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html))从具有所有日期数据列的行开始的。该命令以DataFrame的索引为参考，然后选择所有剩余的特征，将它们的名称放在一个列上，将它们的值放在另一个列上（`var_name`和`value_name`参数帮助您定义这些新列的名称）。这样，您可以将代表某个商店中某个商品的销售额序列的行展开成多个行，每行代表一天。展开列的顺序保持不变，这保证了现在您的时间序列在垂直轴上（因此您可以在其上应用进一步的转换，如移动平均值）。

为了让您了解正在发生的情况，这里展示了在`pd.melt`转换之前的`train_df`。注意，不同日期的量被作为列特征：

![图2.5：训练数据框](img/file8.png)

图2.5：训练数据框

变换之后，您将获得一个`grid_df`，其中日期被分配到单独的日期上：

![图2.5：将pd.melt应用于训练数据框](img/file9.png)

图2.5：将pd.melt应用于训练数据框

特征 d 包含了对不属于索引的列的引用，本质上，是从 `d_1` 到 `d_1935` 的所有特征。通过简单地从其值中移除‘d_’前缀并将它们转换为整数，你现在就有一个日特征。

除了这个之外，代码片段还根据时间从训练数据中分离出一部分行（你的验证集）。在训练部分，它还会根据你提供的预测范围添加必要的行来进行预测。

下面是创建我们的基本特征模板的函数。作为输入，它接受 `train_df` DataFrame，它期望训练结束的日期和预测范围（你想要预测的未来天数）：

[PRE3]

在处理完创建基本特征模板的函数之后，我们准备了一个 pandas DataFrame 的合并函数，这有助于在处理大量数据时节省内存空间并避免内存错误。给定两个 DataFrame，df1 和 df2 以及我们需要它们合并的外键集合，该函数在 df1 和 df2 之间应用左外连接，而不创建新的合并对象，只是简单地扩展现有的 df1 DataFrame。

该函数首先从 df1 中提取外键，然后将提取的键与 df2 合并。这样，该函数创建了一个新的 DataFrame，称为 `merged_gf`，其顺序与 df1 相同。在这个时候，我们只是将 `merged_gf` 列分配给 df1。内部，df1 将从 `merged_gf` 中选择对内部数据结构的引用。这种做法有助于最小化内存使用，因为在任何时刻都只创建了必要使用的数据（没有可以填充内存的重复数据）。当函数返回 df1 时，`merged_gf` 被取消，但 df1 现在使用的数据。

下面是这个实用函数的代码：

[PRE4]

在完成这个必要的步骤之后，我们继续编写一个新的数据处理函数。这次我们处理价格数据，这是一组包含每个店铺每个商品所有周的价格的数据集。由于确定我们是否在谈论一个新商品出现在店铺中非常重要，该函数选择了价格可用的第一个日期（使用价格表中的 `wm_yr_wk` 特征，代表周 id）并将其复制到我们的特征模板中。

下面是处理发布日期的代码：

[PRE5]

在处理完产品在店铺中出现的日期之后，我们肯定要继续处理价格。就每个商品而言，每个店铺，我们准备了一些基本的价格特征，告诉我们：

+   实际价格（按最大值归一化）

+   最高价格

+   最低价格

+   平均价格

+   价格的标准差

+   该商品所经历的不同价格数量

+   店铺中具有相同价格的物品数量

除了这些基本的价格描述性统计之外，我们还添加了一些特征来描述每个商品在不同时间粒度下的动态变化：

+   日 momentum，即实际价格与其前一天价格的比例

+   月 momentum，即实际价格与其同月平均价格的比例

+   年 momentum，即实际价格与其同年平均价格的比例

在这里，我们使用了两个有趣且必要的pandas方法进行时间序列特征处理：

* shift：可以将索引向前或向后移动n步（[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html)）

* transform：应用于group by，用转换后的值填充类似索引的特征（[https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.transform.html)）

此外，为了揭示商品以心理定价阈值（例如$19.99或£2.98——参见此讨论：[https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/145011](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/145011)）出售的情况，处理了价格的小数部分作为特征。`math.modf`函数（[https://docs.python.org/3.8/library/math.html#math.modf](https://docs.python.org/3.8/library/math.html#math.modf)）有助于这样做，因为它将任何浮点数分成小数部分和整数部分（一个两项元组）。

最后，结果表被保存到磁盘上。

下面是这个函数对价格进行所有特征工程的过程：

[PRE6]

下一个函数计算月相，返回其八个阶段之一（从新月到亏月）。尽管月相不应直接影响任何销售（天气条件反而会，但我们没有数据中的天气信息），但它们代表了一个29天半的周期，这非常适合周期性购物行为。关于为什么月相可能作为预测因素的不同假设，在这次竞赛帖子中有有趣的讨论：[https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/154776](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/154776)：

[PRE7]

月相函数是创建基于时间特征的通用函数的一部分。该函数接受日历数据集信息并将其放置在特征中。此类信息包含事件及其类型，以及指示SNAP时期（一种名为补充营养援助计划的营养援助福利——简称SNAP——以帮助低收入家庭）的信息，这些信息可能进一步推动基本商品的销售。该函数还生成诸如日期、月份、年份、星期几、月份中的星期以及是否为周末等数值特征。以下是代码：

[PRE8]

以下函数只是移除了`wm_yr_wk`特征，并将d（日）特征转换为数值。这是以下特征转换函数的必要步骤。

[PRE9]

我们最后两个特征创建函数将为时间序列生成更复杂的特征工程。第一个函数将生成滞后销售和它们的移动平均值。首先，使用移动方法（[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html)）将生成过去15天的滞后销售范围。然后使用移动方法结合滚动（[https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)）将创建7天、14天、30天、60天和180天的移动平均值。

移动命令是必要的，因为它将允许移动索引，这样你就可以始终考虑可用于计算的数据。因此，如果你的预测范围达到七天，计算将只考虑七天前的数据。然后滚动命令将创建一个移动窗口观察值，可以总结（在这种情况下是通过平均值）。在一个时期（移动窗口）上有一个平均值，并跟踪其演变，将帮助你更好地检测趋势中的任何变化，因为不会在时间窗口中重复的图案将被平滑。这是时间序列分析中去除噪声和非有趣模式的一种常见策略。例如，使用七天的滚动平均值，你可以取消所有日间模式，仅表示你的销售在每周发生的情况。

> 你可以尝试不同的移动平均值窗口吗？尝试不同的策略可能会有所帮助。例如，通过探索2022年1月的Tabular Playground竞赛（[https://www.kaggle.com/competitions/tabular-playground-series-jan-2022](https://www.kaggle.com/competitions/tabular-playground-series-jan-2022)），该竞赛致力于时间序列，你可能会找到更多的想法，因为大多数解决方案都是使用梯度提升构建的。

下面是生成滞后和滚动平均值特征的代码：

[PRE10]

至于第二个高级特征工程函数，它是一个编码函数，它接受变量在州、商店、类别、部门和销售商品中的特定分组，并代表它们的平均值和标准差。这种嵌入是时间无关的（时间不是分组的一部分），并且它们的作用是帮助训练算法区分商品、类别和商店（及其组合）如何相互区分。

> 如同《Kaggle 书》第216页所述，使用目标编码计算所提出的嵌入相当简单，你能获得更好的结果吗？

代码通过分组特征，计算它们的描述性统计（在我们的情况下是均值或标准差），然后使用我们之前讨论过的transform([https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transform.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.transform.html))方法将结果应用于数据集：

[PRE11]

完成了特征工程部分后，我们现在继续将存储在磁盘上的所有文件组合起来，同时生成特征。以下函数仅加载基本特征、价格特征、日历特征、滞后/滚动和嵌入特征的不同数据集，并将它们全部连接起来。然后代码仅过滤与特定商店相关的行，将其保存为单独的数据集。这种做法与针对特定商店训练模型以预测特定时间间隔的策略相匹配：

[PRE12]

以下函数，相反，只是进一步处理前一个选择，通过删除未使用的特征和重新排序列，并返回用于训练模型的数据：

[PRE13]

最后，我们现在可以处理训练阶段了。以下代码片段首先定义了训练参数，正如Monsaraida所解释的，这是最有效的。由于训练时间的原因，我们只修改了提升类型，选择使用goss而不是gbdt，因为这可以在很大程度上加快训练速度，而不会在性能上损失太多。通过subsample参数和特征分数也可以为模型提供良好的加速：在梯度提升的每个学习步骤中，只有一半的示例和一半的特征将被考虑。

> 此外，在您的机器上使用正确的编译选项编译LightGBM也可能提高您的速度，如在这篇有趣的竞赛讨论中所述：[https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/148273](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/148273)

Tweedie损失，其幂值为1.1（因此具有与泊松分布相当的基础分布），在建模间歇序列（零销售额占主导地位）时似乎特别有效。所使用的度量标准只是均方根误差（没有必要使用自定义度量标准来表示竞赛度量标准）。我们还使用`force_row_wise`参数在Kaggle笔记本中节省内存。所有其他参数都与Monsaraida在其解决方案中提出的参数完全相同（除了subsampling参数已被禁用，因为它与goss提升类型不兼容）。

> Tweedie 损失在哪些 Kaggle 竞赛中已被证明是有用的？你能通过探索 ForumTopics 和 ForumMessages 表来找到关于这种损失及其在 Meta Kaggle 中使用的有用讨论吗？([https://www.kaggle.com/datasets/kaggle/meta-kaggle](https://www.kaggle.com/datasets/kaggle/meta-kaggle))？

在定义训练参数后，我们只需遍历各个商店，每次上传单个商店的训练数据并训练 LightGBM 模型。每个模型都是经过打包的。我们还从每个模型中提取特征重要性，以便将其合并到一个文件中，然后汇总，从而得到每个特征在该预测范围内的所有商店的平均重要性。

这里是针对特定预测范围训练所有模型的完整函数：

[PRE14]

准备好最后一个函数后，我们为我们的管道工作准备好了所有必要的代码。对于封装整个操作的功能，我们需要输入数据集（时间序列数据集、价格数据集、日历信息）以及最后训练日（对于公共排行榜预测为1,913，对于私有排行榜为1,941）和预测范围（可能是7、14、21或28天）。

[PRE15]

由于 Kaggle 笔记本有有限的运行时间、有限的内存和磁盘空间，我们建议的策略是复制四个包含此处所示代码的笔记本，并使用不同的预测范围参数进行训练。使用相同的笔记本名称，但部分包含预测参数的值，有助于在另一个笔记本中将模型作为外部数据集收集和处理。

这是第一个笔记本，m5-train-day-1941-horizon-7 ([https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-7](https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-7))：

[PRE16]

第二个笔记本，m5-train-day-1941-horizon-14 ([https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-14](https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-14))：

[PRE17]

第三个笔记本，m5-train-day-1941-horizon-21 ([https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-21](https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-21))：

[PRE18]

最后一个是 m5-train-day-1941-horizon-28 ([https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-28](https://www.kaggle.com/code/lucamassaron/m5-train-day-1941-horizon-28))：

[PRE19]

如果你在一台具有足够磁盘空间和内存资源的本地计算机上工作，你可以一次性运行所有四个预测范围，输入包含它们的列表：[7, 14, 21, 28]。现在，在能够提交我们的预测之前，最后一步是组装预测。

## 组装公共和私有预测

你可以在这里看到一个关于如何组装公共和私有排行榜预测的示例：

+   公共排行榜示例：[https://www.kaggle.com/lucamassaron/m5-predict-public-leaderboard](https://www.kaggle.com/lucamassaron/m5-predict-public-leaderboard)

+   私人排行榜示例：[https://www.kaggle.com/code/lucamassaron/m5-predict-private-leaderboard](https://www.kaggle.com/code/lucamassaron/m5-predict-private-leaderboard)

公共和私人提交之间的变化只是不同的最后训练日：它决定了我们将预测哪些天。

在这个结论性的代码片段中，在加载必要的包，如LightGBM，对于每个训练结束日和每个预测范围，我们恢复正确的笔记本及其数据。然后，我们遍历所有商店，预测所有商品在从上一个预测范围到现在的范围内的销售额。这样，每个模型都将预测它所训练的单周。

[PRE20]

当所有预测都收集完毕后，我们使用样本提交文件作为参考将它们合并，包括需要预测的行和列格式（Kaggle期望在验证或测试期间具有每日销售额的递进列的商品具有不同的行）。

[PRE21]

解决方案在私人排行榜上可以达到大约0.54907，结果是第12位，位于金牌区域。恢复Monsaraida的LightGBM参数（例如，使用gbdt而不是goss作为提升参数）应该会带来更高的性能（但你需要在本地计算机或Google Cloud Platform上运行代码）。

> **练习**
> 
> > 作为练习，尝试比较使用相同迭代次数训练LightGBM，将提升集设置为gbdt而不是goss。性能和训练时间差异有多大（你可能需要使用本地机器或云机器，因为训练可能超过12小时）？

## 摘要

在本章中，我们面对了一场相当复杂的时间序列竞赛，因此我们尝试的最简单的前几名解决方案实际上相当复杂，它需要编写大量的处理函数。在你阅读完本章后，你应该对如何处理时间序列以及如何使用梯度提升进行预测有一个更好的理解。当数据量足够时，例如这个问题，优先考虑梯度提升解决方案而不是传统方法，应该有助于你为具有层次相关性、间歇序列以及事件、价格或市场条件等协变量可用的问题创建强大的解决方案。在接下来的章节中，你将面对更加复杂的数据竞赛，处理图像和文本。你将惊讶于通过重新创建得分最高的解决方案并理解其内部工作原理，你可以学到多少。
