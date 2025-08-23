# 8

# 超参数优化

Kaggle 解决方案的表现并不仅仅取决于你选择的机器学习算法类型。除了数据和使用的特征之外，它还强烈取决于算法的**超参数**，这些参数必须在训练之前固定，并且在训练过程中无法学习。在表格数据竞赛中选择正确的变量/数据/特征是最有效的；然而，超参数优化在**所有**类型的竞赛中都是有效的。实际上，给定固定的数据和算法，超参数优化是唯一确保提高算法预测性能并攀登排行榜的方法。它还有助于集成，因为经过调优的模型集成总是比未经调优的模型集成表现更好。

你可能会听说，如果你了解并理解你的选择对算法的影响，手动调整超参数是可能的。许多 Kaggle 大师和专家都宣称，他们在比赛中经常直接调整他们的模型。他们以二分法操作风格有选择性地操作最重要的超参数，探索参数值的越来越小的区间，直到他们找到产生最佳结果的价值。然后，他们转向另一个参数。如果每个参数只有一个最小值，并且参数之间相互独立，这种方法将完美无缺。在这种情况下，搜索主要是由经验和学习算法的知识驱动的。然而，根据我们的经验，在 Kaggle 上遇到的大多数任务并非如此。问题的复杂性和使用的算法需要一种只有搜索算法才能提供的系统方法。因此，我们决定编写这一章节。

在本章中，我们将探讨如何扩展你的交叉验证方法来找到最佳的超参数，这些参数可以推广到你的测试集。这个想法是处理你在比赛中经历的压力和资源稀缺。因此，我们将专注于**贝叶斯优化方法**，这是一种基于你可用资源的复杂模型和数据问题优化的有效方法。我们不会限制自己只搜索预定义超参数的最佳值；我们还将深入研究神经网络架构的问题。

我们将涵盖以下主题：

+   基本优化技术

+   关键参数及其使用方法

+   贝叶斯优化

让我们开始吧！

# 基本优化技术

Scikit-learn 包中用于超参数优化的核心算法是**网格搜索**和**随机搜索**。最近，Scikit-learn 的贡献者还添加了**减半算法**来提高网格搜索和随机搜索策略的性能。

在本节中，我们将讨论所有这些基本技术。通过掌握它们，你不仅将拥有针对某些特定问题的有效优化工具（例如，SVMs 通常通过网格搜索进行优化），而且你还将熟悉超参数优化的工作原理的基础。

首先，弄清楚必要的成分至关重要：

+   需要优化超参数的模型

+   包含每个超参数搜索之间值边界的搜索空间

+   交叉验证方案

+   一个评估指标及其评分函数

所有这些元素都在搜索方法中汇集起来，以确定你正在寻找的解决方案。让我们看看它是如何工作的。

## 网格搜索

**网格搜索**是一种遍历超参数的方法，在高维空间中不可行。对于每个参数，你选择一组你想要测试的值。然后，你测试这个集合中所有可能的组合。这就是为什么它是穷举的：你尝试了所有可能的情况。这是一个非常简单的算法，它受到维度灾难的影响，但，从积极的一面来看，它是**令人尴尬地并行**的（参见[`www.cs.iusb.edu/~danav/teach/b424/b424_23_embpar.html`](https://www.cs.iusb.edu/~danav/teach/b424/b424_23_embpar.html)以了解这个计算机科学术语的定义）。这意味着如果你有足够的处理器来运行搜索，你可以非常快速地获得最优调整。

以一个分类问题为例，让我们看看**支持向量机分类**（**SVC**）。对于分类和回归问题，支持向量机（**SVMs**）可能是你将最频繁使用网格搜索的机器学习算法。使用 Scikit-learn 的`make_classification`函数，我们可以快速生成一个分类数据集：

```py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=300, n_features=50,
                           n_informative=10,
                           n_redundant=25, n_repeated=15,
                           n_clusters_per_class=5,
                           flip_y=0.05, class_sep=0.5,
                           random_state=0) 
```

对于我们的下一步，我们定义了一个基本的 SVC 算法并设置了搜索空间。由于 SVC 的**核函数**（SVM 中转换输入数据的内部函数）决定了要设置的不同的超参数，我们提供了一个包含两个不同搜索空间字典的列表，用于根据选择的核类型使用参数。我们还设置了评估指标（在这种情况下，我们使用准确率，因为目标是完美平衡的）： 

```py
from sklearn import svm
svc = svm.SVC()
svc = svm.SVC(probability=True, random_state=1)
from sklearn import model_selection
search_grid = [
               {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
               {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
               'kernel': ['rbf']}
               ]

scorer = 'accuracy' 
```

在我们的例子中，线性核不需要调整`gamma`参数，尽管它对于径向基函数核非常重要。因此，我们提供了两个字典：第一个包含线性核的参数，第二个包含径向基函数核的参数。每个字典只包含与该核相关的引用以及对该核相关的参数范围。

需要注意的是，评估指标可能与算法优化的成本函数不同。事实上，如第五章“竞赛任务和指标”中所述，你可能会遇到竞赛的评估指标不同，但你无法修改算法的成本函数。在这种情况下，根据你的评估指标调整超参数仍然有助于获得性能良好的模型。虽然这个最优的超参数集是基于算法的成本函数构建的，但找到的将是在这种约束下返回最佳评估指标的参数集。这或许不是理论上你能为该问题获得的最佳结果，但它可能通常不会离最佳结果太远。

所有成分（模型、搜索空间、评估指标、交叉验证方案）都组合到`GridSearchCV`实例中，然后模型被拟合到数据上：

```py
search_func = model_selection.GridSearchCV(estimator=svc, 
                                           param_grid=search_grid,
                                           scoring=scorer, 
                                           n_jobs=-1,
                                           cv=5)
search_func.fit(X, y)
print (search_func.best_params_)
print (search_func.best_score_) 
```

过了一段时间，根据你在其上运行优化的机器，你将根据交叉验证的结果获得最佳组合。

总之，网格搜索是一种非常简单的优化算法，可以利用多核计算机的可用性。它可以很好地与不需要很多调整的机器学习算法（如 SVM 和岭回归和 Lasso 回归）一起工作，但在所有其他情况下，其适用性相当有限。首先，它限于通过离散选择来优化超参数（你需要一个有限的值集来循环）。此外，你不能期望它在需要调整*多个*超参数的算法上有效工作。这是由于搜索空间的爆炸性复杂性，以及由于大多数计算效率低下是由于搜索在盲目地尝试参数值，其中大多数对问题不起作用。

## 随机搜索

**随机搜索**，简单地随机采样搜索空间，在高维空间中是可行的，并且在实践中被广泛使用。然而，随机搜索的缺点是它没有使用先前实验的信息来选择下一个设置（我们应注意的是，这是与网格搜索共享的问题）。此外，为了尽可能快地找到最佳解决方案，你除了希望幸运地找到正确的超参数外，别无他法。

随机搜索工作得非常好，而且很容易理解。尽管它依赖于随机性，但它并不仅仅基于盲目的运气，尽管一开始看起来可能如此。实际上，它就像统计学中的随机抽样：这种技术的主要观点是，如果你进行足够的随机测试，你就有很好的机会找到正确的参数，而无需在测试稍微不同的相似性能组合上浪费能量。

当参数设置过多时，许多 AutoML 系统依赖于随机搜索（参见 Golovin, D. 等人 *Google Vizier: A Service for Black-Box Optimization*，2017）。作为一个经验法则，当你的超参数优化问题的维度足够高时（例如，超过 16），可以考虑使用随机搜索。

下面，我们使用随机搜索来运行之前的示例：

```py
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
search_dict = {'kernel': ['linear', 'rbf'], 
               'C': loguniform(1, 1000),
               'gamma': loguniform(0.0001, 0.1)
               }
scorer = 'accuracy'
search_func = model_selection.RandomizedSearchCV
              (estimator=svc,param_distributions=search_dict, n_iter=6,
              scoring=scorer, n_jobs=-1, cv=5)
search_func.fit(X, y)
print (search_func.best_params_)
print (search_func.best_score_) 
```

注意，现在我们不再关心在单独的空间中运行搜索以针对不同的核。与网格搜索不同，在网格搜索中，每个参数（即使是无效的参数）都会系统地测试，这需要计算时间，而在这里，搜索的效率不受测试的超参数集的影响。搜索不依赖于无关参数，而是由机会引导；任何试验都是有用的，即使你只测试了所选核中许多有效参数中的一个。

## 减半搜索

正如我们提到的，网格搜索和随机搜索都以无信息的方式工作：如果某些测试发现某些超参数不会影响结果或某些值区间无效，则该信息不会传播到后续的搜索中。

因此，Scikit-learn 最近引入了`HalvingGridSearchCV`和`HalvingRandomSearchCV`估计器，可以使用这些估计器通过应用于网格搜索和随机搜索调整策略的**连续减半**来搜索参数空间。

在减半法中，在初始测试轮次中，会评估大量超参数组合，但使用的计算资源却很少。这是通过在训练数据的一小部分案例上运行测试来实现的。较小的训练集需要更少的计算来测试，因此，在牺牲更精确的性能估计的代价下，使用的资源（即时间）更少。这个初始轮次允许选择一组候选超参数值，这些值在问题上的表现更好，用于第二轮，当训练集大小增加时。

随后的轮次以类似的方式进行，随着测试值的范围受到限制（现在测试需要更多的时间来执行，但返回更精确的性能估计），将更大的训练集子集分配给搜索，而候选者的数量继续减半。

下面是一个应用于之前问题的示例：

```py
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
search_func = HalvingRandomSearchCV(estimator=svc,
                                    param_distributions=search_dict,
                                    resource='n_samples',
                                    max_resources=100,
                                    aggressive_elimination=True,
                                    scoring=scorer,
                                    n_jobs=-1,
                                    cv=5,
                                    random_state=0)
search_func.fit(X, y)
print (search_func.best_params_)
print (search_func.best_score_) 
```

通过这种方式，减半通过选择候选者向后续优化步骤提供信息。在下一节中，我们将讨论通过超参数空间实现更精确和高效搜索的更智能的方法。

![Kazuki_Onodera](img/Kazuki_Onodera.png)

Kazuki Onodera

[`www.kaggle.com/onodera`](https://www.kaggle.com/onodera)

让我们暂停一下，进行另一位 Kaggler 的采访。小野田和树是一位拥有约 7 年比赛经验的竞赛大师和讨论大师。他也是 NVIDIA 的高级深度学习数据科学家，并是 NVIDIA KGMON（Kaggle NVIDIA 大师）团队的一员。

你最喜欢的比赛类型是什么？为什么？在技术和解决方法方面，你在 Kaggle 上的专长是什么？

Instacart 购物篮分析。*这项比赛对 Kaggle 社区来说相当具有挑战性，因为它使用了与客户订单相关的匿名数据，以预测用户下一次订单中将会购买哪些之前购买的产品。我喜欢它的原因是我热爱特征工程，我能想出一堆其他人无法想到的好奇特征，这让我在比赛中获得了第二名。*

你是如何应对 Kaggle 比赛的？这种方法与你在日常工作中所做的方法有何不同？

*我试图想象一个模型是如何工作的，并深入研究假阴性和假阳性。这和我的日常工作一样。*

请告诉我们您参加过的特别具有挑战性的比赛，以及您使用了哪些见解来应对这项任务。

人类蛋白质图谱 - 单细胞分类。*这项比赛是一种实例分割比赛，但没有提供掩码。因此，它变成了一个弱监督的多标签分类问题。我创建了一个两阶段管道来去除标签噪声。*

Kaggle 是否帮助了你的职业生涯？如果是的话，是如何帮助的？

*是的。我现在在 NVIDIA KGMON（Kaggle NVIDIA 大师）团队工作。Kaggle 推出了许多不同的机器学习比赛，这些比赛在数据类型、表格、图像、自然语言和信号等方面各不相同，以及在与行业和领域相关方面：工业、金融、天文学、病理学、体育、零售等等。我相信除了 Kagglers 之外，没有人能够访问并拥有所有这些类型的数据经验。*

在你的经验中，不经验的 Kaggler 通常忽略了什么？你现在知道的事情，你希望在你最初开始时就知道？

*目标分析。此外，种子平均法常常被忽视：总是简单但强大。*

在过去的比赛中，你犯过哪些错误？

*目标分析。顶尖团队总是比其他人更好地分析目标，所以如果我在比赛中没有获得更好的名次，我会去阅读关于顶尖解决方案的内容，因为他们总是向我描述我在比赛中遗漏的数据知识。*

您是否推荐使用特定的工具或库来进行数据分析或机器学习？

*仅 Python 和 Jupyter 笔记本。*

当人们参加比赛时，他们应该记住或做些什么最重要？

*如果你能从失败中学习，那么你实际上并没有真正失败。*

你是否使用其他竞赛平台？它们与 Kaggle 相比如何？

*KDD Cup 和 RecSys。两者都满足有趣和具有挑战性的最低要求*。

# 关键参数及其使用方法

下一个问题是为你使用的每种模型选择正确的超参数集。特别是，为了在优化中提高效率，你需要知道每个算法中实际有意义的每个超参数的值。

在本节中，我们将检查 Kaggle 竞赛中最常用的模型，特别是表格模型，并讨论你需要调整的超参数以获得最佳结果。我们将区分用于通用表格数据问题的经典机器学习模型和梯度提升模型（在参数空间方面要求更高）。

至于神经网络，当我们介绍标准模型时（例如，TabNet 神经网络模型有一些特定的参数需要设置，以便它能够正常工作），我们可以给你一些关于调整特定参数的想法。然而，Kaggle 竞赛中大多数深度神经网络的优化并不是在标准模型上进行的，而是在*自定义*模型上进行的。因此，除了基本的学习参数（如学习率和批量大小）之外，神经网络的优化基于你模型神经架构的特定特征。你必须以专门的方式处理这个问题。在章节的末尾，我们将讨论使用 KerasTuner（[`keras.io/keras_tuner/`](https://keras.io/keras_tuner/))进行**神经架构搜索**（**NAS**）的示例。

## 线性模型

需要调整的线性模型通常是带有正则化的线性回归或逻辑回归：

+   `C`：你应该搜索的范围是`np.logspace(-4, 4, 10)`；较小的值指定更强的正则化。

+   `alpha`：你应该在范围`np.logspace(-2, 2, 10)`中搜索；较小的值指定更强的正则化，较大的值指定更强的正则化。此外，请注意，当使用 lasso 时，较高的值需要更多的时间来处理。

+   `l1_ratio`：你应该从列表`[.1, .5, .7, .9, .95, .99, 1]`中选择；它仅适用于弹性网络。

在 Scikit-learn 中，根据算法，你可以找到超参数`C`（逻辑回归）或`alpha`（lasso、ridge、弹性网络）。

## 支持向量机

**SVMs**是一系列用于分类和回归的强大且先进的监督学习技术，可以自动拟合线性和非线性模型。Scikit-learn 提供了一个基于`LIBSVM`的实现，这是一个完整的 SVM 分类和回归实现库，以及`LIBLINEAR`，一个适用于大型数据集（尤其是稀疏文本数据集）的线性分类可扩展库。在它们的优化中，SVMs 通过使用具有最大可能类间边界的决策边界来尝试在分类问题中分离目标类别。

尽管 SVM 使用默认参数可以正常工作，但它们通常不是最优的，你需要通过交叉验证测试各种值组合来找到最佳组合。按照其重要性列出，你必须设置以下参数：

+   `C`：惩罚值。减小它会使类之间的间隔更大，从而忽略更多的噪声，但也使模型更具泛化能力。最佳值通常可以在范围 `np.logspace(-3, 3, 7)` 中找到。

+   `kernel`：此参数将确定在 SVM 中如何实现非线性。它可以设置为 `'linear'`、`'poly'`、`'rbf'`、`'sigmoid'` 或自定义核。最常用的值无疑是 `rbf`。

+   `degree`：与 `kernel='poly'` 一起使用，表示多项式展开的维度。其他核函数会忽略它。通常，将其值设置为 `2` 到 `5` 之间效果最佳。

+   `gamma`：是 `'rbf'`、`'poly'` 和 `'sigmoid'` 的系数。高值往往能更好地拟合数据，但可能导致一些过拟合。直观上，我们可以将 `gamma` 视为单个示例对模型的影响。低值使每个示例的影响范围更广。由于必须考虑许多点，SVM 曲线将倾向于形成一个受局部点影响较小的形状，结果将得到一个更平滑的决定边界曲线。相反，高值的 `gamma` 意味着曲线更多地考虑局部点的排列方式，因此你得到一个更不规则和波动的决定曲线。此超参数的建议网格搜索范围是 `np.logspace(-3, 3, 7)`。

+   `nu`：对于使用 nuSVR 和 nuSVC 的回归和分类，此参数设置接近边界的训练点的容差，这些点没有被正确分类。它有助于忽略刚好在边界附近或被错误分类的点，因此可以使分类决策曲线更平滑。它应该在 `[0,1]` 范围内，因为它与你的训练集的比例相关。最终，它类似于 `C`，高比例会扩大间隔。

+   `epsilon`：此参数指定 SVR 将接受多少误差，通过定义一个大的 `epsilon` 范围，在该范围内，算法训练期间对示例的错误预测不会关联任何惩罚。建议的搜索范围是 `np.logspace(-4, 2, 7)`。

+   `penalty`、`loss` 和 `dual`：对于 LinearSVC，这些参数接受 `('l1', 'squared_hinge', False)`、`('l2', 'hinge', True)`、`('l2', 'squared_hinge', True)` 和 `('l2', 'squared_hinge', False)` 组合。`('l2', 'hinge', True)` 组合类似于 `SVC(kernel='linear')` 学习器。

可能看起来支持向量机（SVM）有很多超参数需要设置，但实际上许多设置仅针对特定实现或核函数，因此你只需要选择相关的参数。

## 随机森林和极端随机树

*Leo Breiman* 和 *Adele Cutler*最初设计了随机森林算法核心的想法，并且算法的名字至今仍然是他们的商标（尽管算法是开源的）。随机森林在 Scikit-learn 中实现为`RandomForestClassifier`或`RandomForestRegressor`。

随机森林的工作方式与袋装法类似，也是由 Leo Breiman 提出的，但它仅使用二分分割决策树，这些树被允许生长到极端。此外，它使用**重抽样**来为每个模型中的案例进行采样。当树生长时，在每次分支的分割中，考虑用于分割的变量集合也是随机抽取的。

这是算法核心的秘密：它组合了由于在不同分割点考虑了不同的样本和变量，彼此之间非常不同的树。由于它们不同，它们也是不相关的。这很有益处，因为当结果被组合时，可以排除很多方差，因为分布两端的极端值往往相互平衡。换句话说，袋装算法保证了预测的一定多样性，使得它们可以发展出单个学习器（如决策树）可能不会遇到的规则。所有这种多样性都是有用的，因为它有助于构建一个平均值为更好的预测器，比集成中的任何单个树都要好。

**Extra Trees**（也称为**极端随机树**），在 Scikit-learn 中由`ExtraTreesClassifier`/`ExtraTreesRegressor`类表示，是一种更随机的随机森林，它以更大的偏差为代价，在估计中产生更低的方差。然而，当涉及到 CPU 效率时，Extra Trees 与随机森林相比可以提供相当的速度提升，因此在处理大型数据集（包括示例和特征）时，它们可能是理想的。导致更高偏差但速度更好的原因是 Extra Tree 中构建分割的方式。随机森林在为树的分支分割抽取一个随机特征集后，会仔细搜索这些特征以找到分配给每个分支的最佳值。相比之下，在 Extra Trees 中，分割的候选特征集和实际的分割值都是完全随机决定的。因此，不需要太多的计算，尽管随机选择的分割可能不是最有效的（因此有偏差）。

对于这两种算法，应该设置的关键超参数如下：

+   `max_features`: 这是每个分割中存在的采样特征的数量，这可以确定算法的性能。数字越低，速度越快，但偏差越高。

+   `min_samples_leaf`: 这允许你确定树的深度。大数值会减少方差并增加偏差。

+   `bootstrap`: 这是一个布尔值，允许进行重抽样。

+   `n_estimators`：这是树的数量。记住，树越多越好，尽管存在一个阈值，超过这个阈值，根据数据问题，我们会得到递减的回报。此外，这会带来计算成本，你必须根据你拥有的资源来考虑。

额外树是随机森林的良好替代品，尤其是在你拥有的数据特别嘈杂的情况下。由于它们在随机选择分割时牺牲了一些方差减少以换取更多的偏差，它们在重要但嘈杂的特征上不太容易过拟合，这些特征在其他情况下会主导随机森林的分割。

## 梯度提升树

梯度提升树或**梯度提升决策树**（**GBDT**）是提升算法的改进版本（提升算法通过在数据的重新加权版本上拟合一系列弱学习器）。像 AdaBoost 一样，GBDT 基于梯度下降函数。该算法已被证明是基于集成模型家族中最有效的一种，尽管它以估计的方差增加、对数据噪声的敏感性增加（这两个问题可以通过使用子采样来缓解）以及由于非并行操作而产生的显著计算成本为特征。

除了深度学习之外，梯度提升是最发达的机器学习算法。自从 AdaBoost 和由*杰罗姆·弗里德曼*开发的初始梯度提升实现以来，出现了各种算法的其他实现，最新的包括 XGBoost、LightGBM 和 CatBoost。

### LightGBM

高性能的 LightGBM 算法([`github.com/Microsoft/LightGBM`](https://github.com/Microsoft/LightGBM))能够分布到多台计算机上，并快速处理大量数据。它是由微软的一个团队作为 GitHub 上的开源项目开发的（还有一个学术论文：[`papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html`](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html))。

LightGBM 基于决策树，就像 XGBoost 一样，但它遵循不同的策略。虽然 XGBoost 使用决策树在变量上进行分割并探索该变量的不同树分割（**按层**树增长策略），LightGBM 专注于一个分割，并从这里继续分割以实现更好的拟合（**按叶**树增长策略）。这使得 LightGBM 能够快速达到数据的良好拟合，并生成与 XGBoost 相比的替代解决方案（如果你预计将两种解决方案结合起来以减少估计的方差，这是好的）。从算法的角度来看，如果我们把决策树操作的分割结构看作一个图，XGBoost 追求的是*广度优先搜索*（BFS），而 LightGBM 追求的是*深度优先搜索*（DFS）。

调整 LightGBM 可能看起来令人畏惧；它有超过一百个可调整的参数，您可以在本页面上探索这些参数：[`github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst`](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst)（也在这里：[`lightgbm.readthedocs.io/en/latest/Parameters.html`](https://lightgbm.readthedocs.io/en/latest/Parameters.html)）。

作为一项经验法则，你应该关注以下超参数，它们通常对结果影响最大：

+   `n_estimators`：一个介于 10 和 10,000 之间的整数，用于设置迭代次数。

+   `learning_rate`：一个介于 0.01 和 1.0 之间的实数，通常从对数均匀分布中采样。它表示梯度下降过程计算权重时的步长，该过程计算算法到这一点为止所有迭代的加权和。

+   `max_depth`：一个介于 1 和 16 之间的整数，表示特征上的最大分割数。将其设置为小于 0 的数字允许最大可能的分割数，通常冒着对数据进行过拟合的风险。

+   `num_leaves`：一个介于 2 和`max_depth`的 2 次幂之间的整数，表示每棵树最多拥有的最终叶子数。

+   `min_data_in_leaf`：一个介于 0 和 300 之间的整数，用于确定一个叶子中数据点的最小数量。

+   `min_gain_to_split`：一个介于 0 和 15 之间的浮点数；它设置了算法进行树分割的最小增益。通过设置此参数，你可以避免不必要的树分割，从而减少过拟合（它对应于 XGBoost 中的`gamma`参数）。

+   `max_bin`：一个介于 32 和 512 之间的整数，用于设置特征值将被分桶的最大数量。如果此参数大于默认值 255，则意味着产生过拟合结果的风险更高。

+   `subsample`：一个介于 0.01 和 1.0 之间的实数，表示用于训练的样本部分。

+   `subsample_freq`：一个介于 0 和 10 之间的整数，指定算法在迭代过程中进行子样本采样的频率。

注意，如果设置为 0，算法将忽略对`subsample`参数给出的任何值。此外，它默认设置为 0，因此仅设置`subsample`参数将不起作用。

+   `feature_fraction`：一个介于 0.1 和 1.0 之间的实数，允许您指定要子样本的特征部分。对特征进行子样本是允许更多随机化在训练中发挥作用，以对抗特征中存在的噪声和多重共线性。

+   `subsample_for_bin`：一个介于 30 和示例数量之间的整数。这设置了用于构建直方图桶的示例数量。

+   `reg_lambda`：一个介于 0 和 100.0 之间的实数，用于设置 L2 正则化。由于它对尺度的敏感性大于对参数确切数量的敏感性，它通常从对数均匀分布中采样。

+   `reg_alpha`：一个介于 0 和 100.0 之间的实数，通常从对数均匀分布中采样，用于设置 L1 正则化。

+   `scale_pos_weight`：一个介于 1e-6 和 500 之间的实数，最好从对数均匀分布中采样。该参数对正例进行加权（从而有效地上采样或下采样）以对抗负例，负例的值保持为 1。

虽然在使用 LightGBM 时需要调整的超参数数量可能看起来令人畏惧，但事实上只有少数几个非常重要。给定固定的迭代次数和学习率，只有少数几个是最有影响力的（`feature_fraction`、`num_leaves`、`subsample`、`reg_lambda`、`reg_alpha`、`min_data_in_leaf`），正如 Kaggle 大师 Kohei Ozaki 在其博客文章中解释的那样：[`medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258`](https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258)。Kohei Ozaki 利用这一事实来为 Optuna 创建快速调整程序（你将在本章末尾找到更多关于 Optuna 优化器的信息）。

### XGBoost

XGBoost ([`github.com/dmlc/XGBoost`](https://github.com/dmlc/XGBoost)) 代表着**极梯度提升**。这是一个开源项目，虽然它不是 Scikit-learn 的一部分，但最近通过 Scikit-learn 包装接口进行了扩展，这使得将其集成到 Scikit-learn 风格的数据管道中变得更加容易。

XGBoost 算法在 2015 年的数据科学竞赛中获得了动力和人气，例如 Kaggle 和 KDD Cup 2015 的竞赛。正如算法的创造者（陈天奇、何通和卡洛斯·古埃斯特林）在关于该算法的论文中所报告的那样，在 2015 年 Kaggle 举办的 29 个挑战中，有 17 个获胜方案使用了 XGBoost 作为独立解决方案或作为多个不同模型的集成的一部分。从那时起，尽管它在与 LightGBM 和 CatBoost 等其他 GBM 实现的创新竞争中有些吃力，但该算法在数据科学家社区中始终保持着强大的吸引力。

除了在准确性和计算效率方面都表现出良好的性能外，XGBoost 还是一个**可扩展**的解决方案，它最好地利用了多核处理器以及分布式机器。

XGBoost 通过对初始树提升 GBM 算法的重要调整，代表着新一代的 GBM 算法：

+   稀疏感知；它可以利用稀疏矩阵，节省内存（不需要密集矩阵）和计算时间（零值以特殊方式处理）。

+   近似树学习（加权分位数草图），与经典的可能分支切割的完整探索相比，在更短的时间内产生类似的结果。

+   在单台机器上进行并行计算（在搜索最佳分割时使用多线程）以及类似地，在多台机器上进行分布式计算。

+   在单机上执行离核计算，利用一种称为**列块**的数据存储解决方案。这种方式通过列在磁盘上排列数据，从而通过以优化算法（该算法在列向量上工作）期望的方式从磁盘中拉取数据来节省时间。

XGBoost 也可以有效地处理缺失数据。其他基于标准决策树的树集成方法需要首先使用一个离群值（如负数）来填充缺失数据，以便开发出适当的树分支来处理缺失值。

关于 XGBoost 的参数 ([`xgboost.readthedocs.io/en/latest/parameter.html`](https://xgboost.readthedocs.io/en/latest/parameter.html))，我们决定突出一些你在竞赛和项目中都会遇到的关键参数：

+   `n_estimators`：通常是一个介于 10 到 5,000 之间的整数。

+   `learning_rate`：一个介于 0.01 到 1.0 之间的实数，最好从对数均匀分布中采样。

+   `min_child_weight`：通常是一个介于 1 到 10 之间的整数。

+   `max_depth`：通常是一个介于 1 到 50 之间的整数。

+   `max_delta_step`：通常是一个介于 0 到 20 之间的整数，表示我们允许每个叶输出允许的最大 delta 步长。

+   `subsample`：一个介于 0.1 到 1.0 之间的实数，表示要子采样的样本比例。

+   `colsample_bytree`：一个介于 0.1 到 1.0 之间的实数，表示按树对列的子采样比例。

+   `colsample_bylevel`：一个介于 0.1 到 1.0 之间的实数，表示树中按层级的子采样比例。

+   `reg_lambda`：一个介于 1e-9 和 100.0 之间的实数，最好从对数均匀分布中采样。此参数控制 L2 正则化。

+   `reg_alpha`：一个介于 1e-9 和 100.0 之间的实数，最好从对数均匀分布中采样。此参数控制 L1 正则化。

+   `gamma`：指定树分区所需的最小损失减少量，此参数需要一个介于 1e-9 和 0.5 之间的实数，最好从对数均匀分布中采样。

+   `scale_pos_weight`：一个介于 1e-6 和 500.0 之间的实数，最好从对数均匀分布中采样，它代表正类的一个权重。

与 LightGBM 类似，XGBoost 也有许多类似的超参数需要调整，因此之前为 LightGBM 做出的所有考虑也适用于 XGBoost。

### CatBoost

2017 年 7 月，俄罗斯搜索引擎 Yandex 公开了另一个有趣的 GBM 算法 CatBoost ([`catboost.ai/`](https://catboost.ai/))，其名称来源于将“Category”和“Boosting”两个词组合在一起。实际上，它的优势在于其处理分类变量的能力，这些变量构成了大多数关系数据库中的大部分信息，它通过采用一热编码和目标编码的混合策略来实现。目标编码是一种通过为特定问题分配适当的数值来表示分类级别的方法；更多关于这一点可以在*第七章*，*表格竞赛建模*中找到。

CatBoost 用于编码分类变量的想法并不新颖，但它是一种之前已经使用过的特征工程方法，主要在数据科学竞赛中使用。目标编码，也称为似然编码、影响编码或均值编码，简单来说，是一种根据与目标变量的关联将标签转换为数字的方法。如果你有一个回归，你可以根据该级别的典型目标均值来转换标签；如果是分类，那么就是给定该标签的目标分类概率（每个类别值的条件目标概率）。这可能看起来是一个简单而聪明的特征工程技巧，但它有副作用，主要是过拟合，因为你在预测器中获取了来自目标的信息。

CatBoost 有相当多的参数（见[`catboost.ai/en/docs/references/training-parameters/`](https://catboost.ai/en/docs/references/training-parameters/)）。我们只讨论了其中最重要的八个：

+   `iterations`: 通常是一个介于 10 和 1,000 之间的整数，但根据问题可以增加。

+   `depth`: 一个介于 1 和 8 之间的整数；通常较高的值需要更长的拟合时间，并且不会产生更好的结果。

+   `learning_rate`: 一个介于 0.01 和 1.0 之间的实数值，最好从对数均匀分布中采样。

+   `random_strength`: 从 1e-9 到 10.0 的范围内以对数线性方式采样的实数，它指定了评分分割的随机水平。

+   `bagging_temperature`: 一个介于 0.0 和 1.0 之间的实数值，用于设置贝叶斯自助抽样。

+   `border_count`: 一个介于 1 和 255 之间的整数，表示数值特征的分割。

+   `l2_leaf_reg`: 一个介于 2 和 30 之间的整数；L2 正则化的值。

+   `scale_pos_weight`: 一个介于 0.01 和 10.0 之间的实数，表示正类权重。

即使 CatBoost 可能看起来只是另一种 GBM 实现，但它有一些差异（也通过不同的参数使用突出显示），这些差异在比赛中可能提供极大的帮助，无论是作为单一模型解决方案还是作为集成模型的一部分。

### HistGradientBoosting

最近，Scikit-learn 引入了一个新的梯度提升版本，灵感来自 LightGBM 的分箱数据和直方图（见 EuroPython 上的这个演示：[`www.youtube.com/watch?v=urVUlKbQfQ4`](https://www.youtube.com/watch?v=urVUlKbQfQ4)）。无论是作为分类器（`HistGradientBoostingClassifier`）还是回归器（`HistGradientBoostingRegressor`），它都可以用于用不同模型丰富集成，并且它提供了一个更短、更关键的超参数范围需要调整：

+   `learning_rate`: 一个介于 0.01 和 1.0 之间的实数，通常从对数均匀分布中采样。

+   `max_iter`: 一个介于 10 到 10,000 之间的整数。

+   `max_leaf_nodes`：一个介于 2 到 500 之间的整数。它与`max_depth`相互作用；建议只设置这两个参数中的一个，并将另一个设置为`None`。

+   `max_depth`：一个介于 2 到 12 之间的整数。

+   `min_samples_leaf`：一个介于 2 到 300 之间的整数。

+   `l2_regularization`：一个介于 0.0 到 100.0 之间的浮点数。

+   `max_bins`：一个介于 32 到 512 之间的整数。

即使 Scikit-learn 的`HistGradientBoosting`与 LightGBM 或 XGBoost 没有太大区别，但它确实提供了一种在比赛中实现 GBM 的不同方法，由`HistGradientBoosting`构建的模型在集成多个预测时（如混合和堆叠）可能会提供一些贡献。

到达本节末尾，你应该对最常见的机器学习算法（尚未讨论深度学习解决方案）及其最重要的超参数有了更熟悉的了解，这将有助于你在 Kaggle 竞赛中构建出色的解决方案。了解基本的优化策略、可用的算法及其关键超参数只是一个起点。在下一节中，我们将开始深入讨论如何使用贝叶斯优化更优地调整它们。

![](img/Alberto_Danese.png)

Alberto Danese

[`www.kaggle.com/albedan`](https://www.kaggle.com/albedan)

我们本章的第二位访谈对象是 Alberto Danese，他是意大利信用卡和数字支付公司 Nexi 的数据科学负责人。这位在 2015 年加入平台的竞赛大师，作为单独的竞争者获得了大部分金牌。

你最喜欢的比赛类型是什么？为什么？在 Kaggle 上，你在技术和解决方法方面有什么专长？

*我一直从事金融服务行业，主要处理结构化数据，我更倾向于这类比赛。我喜欢能够实际掌握数据的本质，并做一些智能的特征工程，以从数据中提取每一丝信息。*

*从技术角度讲，我在经典机器学习库方面有丰富的经验，尤其是梯度提升决策树：最常用的库（XGBoost、LightGBM、CatBoost）总是我的首选。*

你是如何处理 Kaggle 竞赛的？这种方法与你在日常工作中所做的方法有何不同？

*我总是花很多时间探索数据，试图弄清楚赞助商实际上想用机器学习解决什么问题。与新手通常对 Kaggle 的看法不同，我不会花太多时间在特定 ML 算法的所有“调整”上——显然这种方法是有效的！*

*在我的日常工作中，理解数据也非常重要，但在 Kaggle 竞赛中却完全缺失了一些额外的阶段。我必须做到：*

+   *定义一个用 ML 解决的问题（与业务部门的同事一起）*

+   **找到数据，有时也来自外部数据提供商**

+   **当机器学习部分完成时，理解如何将其投入生产并管理其演变**

告诉我们你参加的一个特别具有挑战性的比赛，以及你使用了哪些见解来应对任务。

我很喜欢参加*TalkingData AdTracking Fraud Detection Challenge*，通过这个挑战我成为了大师。除了它是一个非常有趣的话题（打击点击农场中的欺诈）之外，它还真正推动了我进行高效的特征工程，因为数据量巨大（超过 1 亿个标记行），减少计算时间对于测试不同的方法至关重要。它还迫使我以最佳方式理解如何利用滞后/领先特征（以及其他窗口函数），以便在本质上是一个经典机器学习问题中创建一种时间序列。

Kaggle 是否帮助你在职业生涯中取得进步？如果是的话，是如何帮助的？

**当然！能够实现伟大的目标和可验证的结果无疑是使简历脱颖而出的因素之一。当我 2016 年被 Cerved（一家市场情报服务公司）雇佣时，招聘经理完全清楚 Kaggle 是什么——在面试中谈论一些真实世界的项目是非常有价值的。当然，Kaggle 在我的职业生涯发展中扮演了重要的角色。**

在你的经验中，不经验的 Kagglers 通常忽略了什么？你现在知道的事情，你希望在你最初开始时就知道？

我认为每个人都是从编码开始的，也许是从一个公共内核开始，只是更改几行或参数。这在开始时是完全可以接受的！但你确实需要花相当多的时间不编码，而是研究数据和理解问题。

你在过去比赛中犯过哪些错误？

**不确定这算不算一个错误，但我经常更喜欢单独竞争：一方面，这很好，因为它迫使你处理比赛的每一个方面，你可以按照自己的意愿管理时间。但我也非常喜欢与队友在几个比赛中合作：我可能应该更经常地考虑团队合作，因为你可以从合作中学到很多东西。**

你会推荐使用哪些特定的工具或库来进行数据分析或机器学习？

除了那些**通常的**，我一直非常喜欢`data.table`（从 R 版本开始）：我认为它没有得到应有的认可！当你想在本地机器上处理大量数据时，它真的是一个非常棒的包。

当人们参加比赛时，他们应该记住或做最重要的事情是什么？

**首先理解问题和数据：不要立即开始编码！**

# 贝叶斯优化

放弃网格搜索（仅在实验空间有限时可行），实践者通常会选择应用随机搜索优化或尝试**贝叶斯优化**（**BO**）技术，这需要更复杂的设置。

TPEs 最初在 Snoek, J.，Larochelle, H.和 Adams, R. P.的论文《Practical Bayesian optimization of machine learning algorithms》中提出，该论文的网址为[`export.arxiv.org/pdf/1206.2944`](http://export.arxiv.org/pdf/1206.2944)。贝叶斯优化的关键思想是，我们优化一个**代理函数**（也称为**替代函数**），而不是真正的目标函数（网格搜索和随机搜索都这样做）。我们这样做是因为没有梯度，如果测试真正的目标函数成本高昂（如果不是，那么我们简单地进行随机搜索），以及如果搜索空间是嘈杂且足够复杂的情况下。

贝叶斯搜索在**探索**和**利用**之间取得平衡。一开始，它随机探索，从而在过程中训练替代函数。基于这个替代函数，搜索利用其对预测器如何工作的初始近似知识，以便采样更有用的示例并最小化成本函数。正如名称中的**贝叶斯**部分所暗示的，我们在优化过程中使用先验来做出更明智的采样决策。这样，我们通过限制需要进行的评估次数来更快地达到最小化。

贝叶斯优化使用一个**获取函数**来告诉我们一个观察结果的前景如何。实际上，为了管理探索和利用之间的权衡，算法定义了一个获取函数，它提供了一个单一指标，说明尝试任何给定点将有多有用。

通常，贝叶斯优化由高斯过程提供动力。当搜索空间具有平滑且可预测的响应时，高斯过程表现更好。当搜索空间更复杂时，一个替代方案是使用树算法（例如，随机森林），或者一个完全不同的方法，称为**树帕尔森估计器**或**树结构帕尔森估计器**（**TPEs**）。

与直接构建一个估计参数集成功度的模型不同，从而像先知一样行动，TPEs（Tree Parzen Estimators）根据实验提供的连续近似，估计一个多变量分布的参数，这些参数定义了参数的最佳性能值。通过这种方式，TPEs 通过从概率分布中采样来推导出最佳参数集，而不是直接从机器学习模型（如高斯过程）中直接推导。

我们将讨论这些方法中的每一个，首先通过检查基于高斯过程的 Scikit-optimize 和 KerasTuner，Scikit-optimize 还可以使用随机森林，而 KerasTuner 可以使用多臂老虎机，然后是主要基于 TPE 的 Optuna（尽管它也提供不同的策略：[`optuna.readthedocs.io/en/stable/reference/samplers.html`](https://optuna.readthedocs.io/en/stable/reference/samplers.html))。

尽管贝叶斯优化被认为是超参数调整的当前最佳实践，但始终记住，对于更复杂的参数空间，使用贝叶斯优化在时间和计算上并不比随机搜索简单找到的解决方案有优势。例如，在 Google Cloud Machine Learning Engine 服务中，贝叶斯优化的使用仅限于涉及最多十六个参数的问题。对于更多参数的情况，它将回退到随机抽样。

## 使用 Scikit-optimize

Scikit-optimize (`skopt`) 是使用与 Scikit-learn 相同的 API 开发的，同时广泛使用了 NumPy 和 SciPy 函数。此外，它是由 Scikit-learn 项目的一些贡献者创建的，例如 *Gilles Louppe*。

该包基于高斯过程算法，维护得很好，尽管有时它必须因为 Scikit-learn、NumPy 或 SciPy 方面的改进而赶上来。例如，在撰写本文时，为了在 Kaggle 笔记本上正确运行，你必须回滚到这些包的旧版本，正如 GitHub 上的一个问题所解释的 ([`github.com/scikit-optimize/scikit-optimize/issues/981`](https://github.com/scikit-optimize/scikit-optimize/issues/981))。

该包具有直观的 API，相当容易对其进行修改并使用其函数在自定义优化策略中。Scikit-optimize 还以其有用的图形表示而闻名。实际上，通过可视化优化过程的结果（使用 Scikit-optimize 的 `plot_objective` 函数），你可以判断是否可以重新定义问题的搜索空间，并制定一个关于问题优化工作原理的解释。

在我们的示例中，我们将参考以下 Kaggle 笔记本中的工作：

+   [`www.kaggle.com/lucamassaron/tutorial-bayesian-optimization-with-lightgbm`](https://www.kaggle.com/lucamassaron/tutorial-bayesian-optimization-with-lightgbm)

+   [`www.kaggle.com/lucamassaron/scikit-optimize-for-lightgbm`](https://www.kaggle.com/lucamassaron/scikit-optimize-for-lightgbm)

我们在这里的目的就是向您展示如何快速处理一场比赛，例如 *30 Days of ML* 的优化问题，这是一场最近举办的比赛，许多 Kagglers 参与其中，学习新技能并在为期 30 天的比赛中将它们应用。这场比赛的目标是预测保险索赔的价值，因此它是一个回归问题。您可以通过访问 [`www.kaggle.com/thirty-days-of-ml`](https://www.kaggle.com/thirty-days-of-ml) 了解更多关于这个项目的信息并下载我们将要展示的示例所需的数据（材料总是对公众开放）。

如果您无法访问数据，因为您之前没有参加过比赛，您可以使用这个 Kaggle 数据集：[`www.kaggle.com/lucamassaron/30-days-of-ml`](https://www.kaggle.com/lucamassaron/30-days-of-ml)。

以下代码将展示如何加载此问题的数据，然后设置一个贝叶斯优化过程，该过程将提高 LightGBM 模型的性能。

我们首先加载所需的包：

```py
# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial
# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")
# Classifiers
import lightgbm as lgb
# Model selection
from sklearn.model_selection import KFold
# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer 
```

作为下一步，我们加载数据。除了将一些具有字母作为级别的分类特征转换为有序数字之外，数据不需要太多处理：

```py
# Loading data 
X = pd.read_csv("../input/30-days-of-ml/train.csv")
X_test = pd.read_csv("../input/30-days-of-ml/test.csv")
# Preparing data as a tabular matrix
y = X.target
X = X.set_index('id').drop('target', axis='columns')
X_test = X_test.set_index('id')
# Dealing with categorical data
categoricals = [item for item in X.columns if 'cat' in item]
cat_values = np.unique(X[categoricals].values)
cat_dict = dict(zip(cat_values, range(len(cat_values))))
X[categoricals] = X[categoricals].replace(cat_dict).astype('category')
X_test[categoricals] = X_test[categoricals].replace(cat_dict).astype('category') 
```

在数据可用后，我们定义了一个报告函数，该函数可以被 Scikit-optimize 用于各种优化任务。该函数接受数据和优化器作为输入。它还可以处理 **回调函数**，这些函数执行诸如报告、基于达到一定搜索时间阈值或性能没有提高（例如，在特定次数的迭代中看不到改进）的早期停止，或者在每个优化迭代后保存处理状态等操作：

```py
# Reporting util for different optimizers
def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performance of optimizers
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()

    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)

    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    print((title + " took %.2f seconds, candidates checked: %d, best CV            score: %.3f" + u" \u00B1"+" %.3f") % 
                             (time() - start,
                             len(optimizer.cv_results_['params']),
                             best_score, 
                             best_score_std))
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params 
```

现在我们必须准备评分函数（评估基于此），验证策略（基于交叉验证），模型和搜索空间。对于评分函数，它应该是一个均方根误差指标，我们参考了 Scikit-learn 中的实践，在那里你总是最小化一个函数（如果你必须最大化，你最小化其负值）。

`make_scorer` 包装器可以轻松地复制这样的实践：

```py
# Setting the scoring function
scoring = make_scorer(partial(mean_squared_error, squared=False),
                      greater_is_better=False)
# Setting the validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=0)
# Setting the basic regressor
reg = lgb.LGBMRegressor(boosting_type='gbdt',
                        metric='rmse',
                        objective='regression',
                        n_jobs=1, 
                        verbose=-1,
                        random_state=0) 
```

设置搜索空间需要使用 Scikit-optimize 中的不同函数，例如 `Real`、`Integer` 或 `Choice`，每个函数从您定义的不同类型的分布中进行采样，作为参数（通常是均匀分布，但在您对参数的规模效应比其确切值更感兴趣时，也使用对数均匀分布）：

```py
# Setting the search space
search_spaces = {

     # Boosting learning rate
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),

     # Number of boosted trees to fit
    'n_estimators': Integer(30, 5000),

     # Maximum tree leaves for base learners
    'num_leaves': Integer(2, 512),

     # Maximum tree depth for base learners
    'max_depth': Integer(-1, 256),
     # Minimal number of data in one leaf
    'min_child_samples': Integer(1, 256),
     # Max number of bins buckets
    'max_bin': Integer(100, 1000),
     # Subsample ratio of the training instance 
    'subsample': Real(0.01, 1.0, 'uniform'),
     # Frequency of subsample 
    'subsample_freq': Integer(0, 10),

     # Subsample ratio of columns
    'colsample_bytree': Real(0.01, 1.0, 'uniform'), 

     # Minimum sum of instance weight
    'min_child_weight': Real(0.01, 10.0, 'uniform'),

     # L2 regularization
    'reg_lambda': Real(1e-9, 100.0, 'log-uniform'),

     # L1 regularization
    'reg_alpha': Real(1e-9, 100.0, 'log-uniform'),
   } 
```

一旦您已经定义：

+   您的交叉验证策略

+   您的评估指标

+   您的基础模型

+   您的超参数搜索空间

剩下的只是将它们输入到您的优化函数`BayesSearchCV`中。根据提供的 CV 方案，此函数将根据搜索空间内的值寻找评分函数的最小值。您可以设置最大迭代次数、代理函数的类型（高斯过程`GP`在大多数情况下都适用），以及随机种子以实现可重复性：

```py
# Wrapping everything up into the Bayesian optimizer
opt = BayesSearchCV(estimator=reg,
                    search_spaces=search_spaces,
                    scoring=scoring,
                    cv=kf,
                    n_iter=60,           # max number of trials
                    n_jobs=-1,           # number of jobs
                    iid=False,         
                    # if not iid it optimizes on the cv score
                    return_train_score=False,
                    refit=False,  
                    # Gaussian Processes (GP) 
                    optimizer_kwargs={'base_estimator': 'GP'},
                    # random state for replicability
                    random_state=0) 
```

在这一点上，您可以使用我们之前定义的报告函数开始搜索。一段时间后，该函数将返回该问题的最佳参数。

```py
# Running the optimizer
overdone_control = DeltaYStopper(delta=0.0001)
# We stop if the gain of the optimization becomes too small
time_limit_control = DeadlineStopper(total_time=60 * 60 * 6)
# We impose a time limit (6 hours)
best_params = report_perf(opt, X, y,'LightGBM_regression', 
                          callbacks=[overdone_control, time_limit_control]) 
```

在示例中，我们通过指定一个最大允许时间（6 小时）来限制操作，并在停止和报告最佳结果之前。由于贝叶斯优化方法结合了不同超参数组合的探索和利用，因此任何时间停止都将始终返回迄今为止找到的最佳解决方案（但不一定是可能的最佳解决方案）。这是因为获取函数将始终根据代理函数返回的估计性能及其不确定性区间，优先探索搜索空间中最有希望的部分。

## 定制贝叶斯优化搜索

Scikit-optimize 提供的`BayesSearchCV`函数确实很方便，因为它自己封装并安排了超参数搜索的所有元素，但它也有局限性。例如，您可能发现在比赛中这样做很有用：

+   对每次搜索迭代有更多控制，例如混合随机搜索和贝叶斯搜索

+   能够在算法上应用早期停止

+   更多地定制您的验证策略

+   早期停止无效的实验（例如，当可用时立即评估单个交叉验证折的性能，而不是等待所有折的平均值）

+   创建表现相似的超参数集簇（例如，为了创建多个模型，这些模型仅在使用的超参数上有所不同，用于混合集成）

如果您能够修改`BayesSearchCV`内部过程，那么这些任务中的每一个都不会太复杂。幸运的是，Scikit-optimize 允许您做到这一点。实际上，在`BayesSearchCV`以及该包的其他封装器后面，都有特定的最小化函数，您可以将它们用作您自己的搜索函数的独立部分：

+   `gp_minimize`：使用高斯过程进行贝叶斯优化

+   `forest_minimize`：使用随机森林或极端随机树进行贝叶斯优化

+   `gbrt_minimize`：使用梯度提升的贝叶斯优化

+   `dummy_minimize`：仅仅是随机搜索

在以下示例中，我们将修改之前的搜索，使用我们自己的自定义搜索函数。新的自定义函数将在训练期间接受早期停止，并在其中一个折的验证结果不是最佳表现时剪枝实验。

你可以在 Kaggle 笔记本中找到下一个示例：[`www.kaggle.com/lucamassaron/hacking-bayesian-optimization`](https://www.kaggle.com/lucamassaron/hacking-bayesian-optimization)。

如前例所示，我们首先导入必要的包。

```py
# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial
# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")
# Classifier/Regressor
from xgboost import XGBRegressor
# Model selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize, forest_minimize
from skopt import gbrt_minimize, dummy_minimize
# Decorator to convert a list of parameters to named arguments
from skopt.utils import use_named_args 
# Data processing
from sklearn.preprocessing import OrdinalEncoder 
```

与之前一样，我们从*30 Days of ML*竞赛上传数据：

```py
# Loading data 
X_train = pd.read_csv("../input/30-days-of-ml/train.csv")
X_test = pd.read_csv("../input/30-days-of-ml/test.csv")
# Preparing data as a tabular matrix
y_train = X_train.target
X_train = X_train.set_index('id').drop('target', axis='columns')
X_test = X_test.set_index('id')
# Pointing out categorical features
categoricals = [item for item in X_train.columns if 'cat' in item]
# Dealing with categorical data using OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
X_train[categoricals] = ordinal_encoder.fit_transform(X_train[categoricals])
X_test[categoricals] = ordinal_encoder.transform(X_test[categoricals]) 
```

现在我们设置了进行超参数搜索所需的所有必要元素，即评分函数、验证策略、搜索空间以及要优化的机器学习模型。评分函数和验证策略将后来成为构成目标函数的核心元素，目标函数是贝叶斯优化努力最小化的函数。

```py
# Setting the scoring function
scoring = partial(mean_squared_error, squared=False)
# Setting the cv strategy
kf = KFold(n_splits=5, shuffle=True, random_state=0)
# Setting the search space
space = [Real(0.01, 1.0, 'uniform', name='learning_rate'),
         Integer(1, 8, name='max_depth'),
         Real(0.1, 1.0, 'uniform', name='subsample'),
         # Subsample ratio of columns by tree
         Real(0.1, 1.0, 'uniform', name='colsample_bytree'),  
         # L2 regularization
         Real(0, 100., 'uniform', name='reg_lambda'),
         # L1 regularization
         Real(0, 100., 'uniform', name='reg_alpha'),
         # minimum sum of instance weight (hessian)  
         Real(1, 30, 'uniform', name='min_child_weight')
         ]
model = XGBRegressor(n_estimators=10_000, 
                     booster='gbtree', random_state=0) 
```

注意这次我们没有在搜索空间中包含估计器的数量（即`n_estimators`参数）。相反，我们在实例化模型时设置它，并输入一个高值，因为我们期望根据验证集提前停止模型。

作为下一步，你现在需要创建目标函数。目标函数应该只接受要优化的参数作为输入，并返回相应的分数。然而，目标函数还需要接受你刚刚准备好的搜索所需的元素。自然地，你可以从函数内部引用它们。然而，将它们带入函数本身，在其内部内存空间中，是一种良好的实践。这有其优点；例如，你会使元素不可变，并且它们将与目标函数（通过序列化或如果你在多处理器级别上分配搜索任务）一起携带。你可以通过创建一个`make`函数来实现第二个结果，该函数接受元素，并通过`make`函数返回修改后的目标函数。通过这种简单的结构，你的目标函数将包含所有元素，如数据和模型，而你只需要传递要测试的参数。

让我们开始编写函数。我们将沿途讨论一些相关方面：

```py
# The objective function to be minimized
def make_objective(model, X, y, space, cv, scoring, validation=0.2):
    # This decorator converts your objective function 
    # with named arguments into one that accepts a list as argument,
    # while doing the conversion automatically.
    @use_named_args(space) 
    def objective(**params):
        model.set_params(**params)
        print("\nTesting: ", params)
        validation_scores = list()
        for k, (train_index, test_index) in enumerate(kf.split(X, y)):
            val_index = list()
            train_examples = int(train_examples * (1 - validation))
            train_index, val_index = (train_index[:train_examples], 
                                      train_index[train_examples:])

            start_time = time()
            model.fit(X.iloc[train_index,:], y[train_index],
                      early_stopping_rounds=50,
                      eval_set=[(X.iloc[val_index,:], y[val_index])], 
                      verbose=0
                    )
            end_time = time()

            rounds = model.best_iteration

            test_preds = model.predict(X.iloc[test_index,:])
            test_score = scoring(y[test_index], test_preds)
            print(f"CV Fold {k+1} rmse:{test_score:0.5f}-{rounds} 
                  rounds - it took {end_time-start_time:0.0f} secs")
            validation_scores.append(test_score) 
```

在函数的第一个部分，你只需创建一个目标函数，进行交叉验证并使用提前停止来拟合数据。我们使用了一种激进的提前停止策略以节省时间，但如果你认为它可能对你的问题更有效，你可以增加耐心轮数。请注意，验证示例是按顺序从训练折叠中的示例中取出的（参见代码中`train_index`和`val_index`的定义），而将折叠外的示例（由`kf`交叉验证分割得到的`test_index`）保留用于最终验证。如果你不希望对用于提前停止的数据产生自适应过拟合，这一点很重要。

在下一部分，在进入交叉验证循环并继续训练和测试剩余的交叉验证折之前，你分析出在出折集上获得的折的结果：

```py
 if len(history[k]) >= 10:
                threshold = np.percentile(history[k], q=25)
                if test_score > threshold:
                    print(f"Early stopping for under-performing fold: 
                          threshold is {threshold:0.5f}")
                    return np.mean(validation_scores)

            history[k].append(test_score)
        return np.mean(validation_scores)
    return objective 
```

注意，我们正在维护一个全局字典`history`，其中包含到目前为止从每个折中获得的成果。我们可以比较多个实验和交叉验证的结果；由于随机种子，交叉验证是可重复的，因此相同折的成果可以完美比较。如果当前折的结果与其他迭代中获得的先前折的结果（以底部四分位数作为参考）相比较差，那么想法是停止并返回迄今为止测试的折的平均值。这样做的原因是，如果一个折没有呈现可接受的结果，那么整个交叉验证可能也不会。因此，你可以直接退出并转向另一组更有希望的参数。这是一种交叉验证的早期停止，应该会加快你的搜索速度，并允许你在更短的时间内覆盖更多实验。

接下来，使用我们的`make_objective`函数，我们将所有元素（模型、数据、搜索空间、验证策略和评分函数）组合成一个单一函数，即目标函数。结果，我们现在有一个只接受要优化的参数并返回得分的函数，基于这个得分，优化的最小化引擎将决定下一个实验：

```py
objective = make_objective(model,
                           X_train, y_train,
                           space=space,
                           cv=kf,
                           scoring=scoring) 
```

由于我们想要控制优化的每一步并保存以供以后使用，我们还准备了一个回调函数，该函数将在最小化过程的每个迭代中保存执行实验及其结果列表。只需使用这两条信息，最小化引擎就可以在任何时候停止，然后可以从检查点恢复优化：

```py
def onstep(res):
    global counter
    x0 = res.x_iters   # List of input points
    y0 = res.func_vals # Evaluation of input points
    print('Last eval: ', x0[-1], 
          ' - Score ', y0[-1])
    print('Current iter: ', counter, 
          ' - Best Score ', res.fun, 
          ' - Best Args: ', res.x)
    # Saving a checkpoint to disk
    joblib.dump((x0, y0), 'checkpoint.pkl') 
    counter += 1 
```

到目前为止，我们已经准备好开始。贝叶斯优化需要一些起始点才能正常工作。我们通过随机搜索（使用`dummy_minimize`函数）创建了一系列实验，并保存了它们的结果：

```py
counter = 0
history = {i:list() for i in range(5)}
used_time = 0
gp_round = dummy_minimize(func=objective,
                          dimensions=space,
                          n_calls=30,
                          callback=[onstep],
                          random_state=0) 
```

然后，我们可以检索保存的实验并打印出贝叶斯优化测试的超参数集序列及其结果。实际上，我们可以在`x0`和`y0`列表中找到参数及其结果：

```py
x0, y0 = joblib.load('checkpoint.pkl')
print(len(x0)) 
```

到目前为止，我们甚至可以带一些对搜索空间、获取函数、调用次数或回调的更改来恢复贝叶斯优化：

```py
x0, y0 = joblib.load('checkpoint.pkl')
gp_round = gp_minimize(func=objective,
                       x0=x0,    # already examined values for x
                       y0=y0,    # observed values for x0
                       dimensions=space,
                       acq_func='gp_hedge',
                       n_calls=30,
                       n_initial_points=0,
                       callback=[onstep],
                       random_state=0) 
```

一旦我们满意地认为不需要继续调用优化函数，我们可以打印出最佳得分（基于我们的输入和验证方案）以及最佳超参数集：

```py
x0, y0 = joblib.load('checkpoint.pkl')
print(f"Best score: {gp_round.fun:0.5f}")
print("Best hyperparameters:")
for sp, x in zip(gp_round.space, gp_round.x):
    print(f"{sp.name:25} : {x}") 
```

基于最佳结果，我们可以重新训练我们的模型，以便在比赛中使用。

现在我们有了参数集及其结果（`x0`和`y0`列表），我们还可以探索不同的结果，并将那些输出相似但使用的参数集不同的结果聚类在一起。这将帮助我们训练一个具有相似性能但不同优化策略的更多样化的模型集。这是**混合**的理想情况，即通过平均多个模型来降低估计的方差，并获得更好的公共和私人排行榜分数。

参考第九章，关于**混合和堆叠解决方案的集成**的讨论。

## 将贝叶斯优化扩展到神经网络架构搜索

进入深度学习领域，神经网络似乎也有许多超参数需要调整：

+   批处理大小

+   学习率

+   优化器的类型及其内部参数

所有这些参数都会影响网络的学习方式，它们可以产生重大影响；仅仅批处理大小或学习率的一点点差异就可能导致网络能否将其错误降低到某个阈值以下。

话虽如此，这些学习参数并不是你在与**深度神经网络**（DNNs）一起工作时唯一可以优化的参数。网络在层中的组织方式和其架构的细节可以产生更大的影响。

事实上，从技术上来说，**架构**意味着深度神经网络的表示能力，这意味着，根据你使用的层，网络要么能够读取和处理数据中所有可用的信息，要么不能。与其他机器学习算法相比，虽然你的选择似乎无限，但唯一明显的限制是你处理神经网络部分以及将它们组合在一起的知识和经验。

优秀的深度学习实践者在组装高性能深度神经网络（DNNs）时常用的常见最佳实践主要依赖于：

+   依赖于预训练模型（因此你必须非常了解可用的解决方案，例如在 Hugging Face([`huggingface.co/models`](https://huggingface.co/models))或 GitHub 上找到的）

+   阅读前沿论文

+   复制同一比赛或之前的 Kaggle 笔记本

+   尝试和错误

+   灵感和运气

在 Geoffrey Hinton 教授的一次著名课程中，他提到，你可以使用自动化方法，如贝叶斯优化，来实现相似甚至更好的结果。贝叶斯优化还可以避免你陷入困境，因为你无法在众多可能的超参数组合中找到最佳组合。

关于 Geoffrey Hinton 教授课程的录音，请参阅[`www.youtube.com/watch?v=i0cKa0di_lo`](https://www.youtube.com/watch?v=i0cKa0di_lo)。

关于幻灯片，请参阅[`www.cs.toronto.edu/~hinton/coursera/lecture16/lec16.pdf`](https://www.cs.toronto.edu/~hinton/coursera/lecture16/lec16.pdf)。

正如我们之前提到的，即使在大多数复杂的 AutoML 系统中，当你有太多的超参数时，依赖于随机优化可能会产生更好的结果，或者与贝叶斯优化在相同的时间内产生相同的结果。此外，在这种情况下，你还得对抗一个具有尖锐转弯和表面的优化景观；在深度神经网络优化中，许多参数不会是连续的，而是布尔值，仅仅一个变化可能会意外地改善或恶化你网络的性能。

我们的经验告诉我们，随机优化可能不适合 Kaggle 比赛，因为：

+   你有限的时间和资源

+   你可以利用你之前的优化结果来找到更好的解决方案

在这种情况下，贝叶斯优化是理想的：你可以根据你拥有的时间和计算资源来设置它，分阶段进行，通过多次会话来细化你的设置。此外，你不太可能轻松地利用并行性来调整深度神经网络，因为它们使用 GPU，除非你手头上有多台非常强大的机器。通过顺序工作，贝叶斯优化只需要一台性能良好的机器来完成这项任务。最后，即使通过搜索难以找到最优架构，由于你利用了先前实验的信息，尤其是在开始时，可以完全避免那些不会工作的参数组合。而在随机优化中，除非你在过程中改变搜索空间，否则所有组合都始终有可能被测试。

然而，也存在一些缺点。贝叶斯优化使用从先前试验构建的代理函数来模拟超参数空间，这并不是一个没有错误的过程。过程最终只关注搜索空间的一部分而忽略其他部分（这些部分可能包含你正在寻找的最小值）的可能性并不是微乎其微的。解决这个问题的方法是运行大量实验以确保安全，或者交替进行随机搜索和贝叶斯优化，用随机试验挑战贝叶斯模型，迫使其以更优的方式重塑搜索模型。

在我们的例子中，我们再次使用 Kaggle 的“30 天机器学习”活动中的数据，这是一个回归任务。我们的例子基于 TensorFlow，但经过一些小的修改，它可以在其他深度学习框架上运行，例如 PyTorch 或 MXNet。

如前所述，你可以在 Kaggle 上找到这个例子：[`www.kaggle.com/lucamassaron/hacking-bayesian-optimization-for-dnns`](https://www.kaggle.com/lucamassaron/hacking-bayesian-optimization-for-dnns)。

让我们开始：

```py
import tensorflow as tf 
```

在导入 TensorFlow 包后，我们利用其`Dataset`函数创建一个可迭代的对象，能够为我们的神经网络提供数据批次：

```py
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe),   
                                             labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds
tf.keras.utils.get_custom_objects().update({'leaky-relu': tf.keras.layers.Activation(tf.keras.layers.LeakyReLU(alpha=0.2))}) 
```

我们还将漏激活 ReLU 作为我们模型的自定义对象；可以通过字符串调用它，无需直接使用该函数。

我们继续编写一个函数，该函数根据一组超参数创建我们的深度神经网络模型：

```py
def create_model(cat0_dim, cat1_dim, cat2_dim,
                 cat3_dim, cat4_dim, cat5_dim, 
                 cat6_dim, cat7_dim, cat8_dim, cat9_dim,
                 layers, layer_1, layer_2, layer_3, layer_4, layer_5, 
                 activation, dropout, batch_normalization, learning_rate, 
                 **others):

    dims = {'cat0': cat0_dim, 'cat1': cat1_dim, 'cat2': cat2_dim, 
            'cat3': cat3_dim, 'cat4': cat4_dim, 'cat5': cat5_dim,
            'cat6': cat6_dim, 'cat7': cat7_dim, 'cat8': cat8_dim, 
            'cat9': cat9_dim}

    vocab = {h:X_train['cat4'].unique().astype(int) 
             for h in ['cat0', 'cat1', 'cat2', 'cat3', 
                       'cat4', 'cat5', 'cat6', 'cat7', 
                       'cat8', 'cat9']}

    layers = [layer_1, layer_2, layer_3, layer_4, layer_5][:layers]

    feature_columns = list()
    for header in ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 
                   'cont6','cont7', 'cont8', 'cont9', 'cont10',
                   'cont11', 'cont12', 'cont13']:

        feature_columns.append(tf.feature_column.numeric_column(header))
    for header in ['cat0', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 
                   'cat6', 'cat7', 'cat8', 'cat9']:
        feature_columns.append(
            tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
            header, vocabulary_list=vocab[header]),  
            dimension=dims[header]))
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    network_struct = [feature_layer]
    for nodes in layers:
        network_struct.append(
                 tf.keras.layers.Dense(nodes, activation=activation))
        if batch_normalization is True:
                   network_struct.append(
                   tf.keras.layers.BatchNormalization())
        if dropout > 0:
            network_struct.append(tf.keras.layers.Dropout(dropout))
    model = tf.keras.Sequential(network_struct + 
                                [tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(
                          learning_rate=learning_rate),
                  loss= tf.keras.losses.MeanSquaredError(),
                  metrics=['mean_squared_error'])

    return model 
```

在内部，`create_model`函数中的代码根据提供的输入自定义神经网络架构。例如，作为函数的参数，您可以提供每个分类变量的嵌入维度，或者定义网络中存在的密集层的结构和数量。所有这些参数都与贝叶斯优化要探索的参数空间相关，因此创建模型的函数的每个输入参数都应该与搜索空间中定义的**采样函数**相关。您需要做的只是将采样函数放在一个列表中，按照`create_model`函数期望的顺序：

```py
# Setting the search space

space = [Integer(1, 2, name='cat0_dim'),
         Integer(1, 2, name='cat1_dim'),
         Integer(1, 2, name='cat2_dim'),
         Integer(1, 3, name='cat3_dim'),
         Integer(1, 3, name='cat4_dim'),
         Integer(1, 3, name='cat5_dim'),
         Integer(1, 4, name='cat6_dim'),
         Integer(1, 4, name='cat7_dim'),
         Integer(1, 6, name='cat8_dim'),
         Integer(1, 8, name='cat9_dim'),
         Integer(1, 5, name='layers'),
         Integer(2, 256, name='layer_1'),
         Integer(2, 256, name='layer_2'),
         Integer(2, 256, name='layer_3'),
         Integer(2, 256, name='layer_4'),
         Integer(2, 256, name='layer_5'),
         Categorical(['relu', 'leaky-relu'], name='activation'),
         Real(0.0, 0.5, 'uniform', name='dropout'),
         Categorical([True, False], name='batch_normalization'),
         Categorical([0.01, 0.005, 0.002, 0.001], name='learning_rate'),
         Integer(256, 1024, name='batch_size')
        ] 
```

如前所述，您现在将所有与搜索相关的元素组合成一个目标函数，由一个函数创建，该函数结合了您的基本搜索元素，如数据和交叉验证策略：

```py
def make_objective(model_fn, X, space, cv, scoring, validation=0.2):
    # This decorator converts your objective function with named arguments
    # into one that accepts a list as argument, while doing the conversion
    # automatically.
    @use_named_args(space) 
    def objective(**params):

        print("\nTesting: ", params)
        validation_scores = list()

        for k, (train_index, test_index) in enumerate(kf.split(X)):
            val_index = list()
            train_examples = len(train_index)
            train_examples = int(train_examples * (1 - validation))
            train_index, val_index = (train_index[:train_examples], 
                                      train_index[train_examples:])

            start_time = time()

            model = model_fn(**params)
            measure_to_monitor = 'val_mean_squared_error'
            modality='min'
            early_stopping = tf.keras.callbacks.EarlyStopping(
                                 monitor=measure_to_monitor,
                                 mode=modality,
                                 patience=5, 
                                 verbose=0)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                   'best.model',
                                   monitor=measure_to_monitor, 
                                   mode=modality, 
                                   save_best_only=True, 
                                   verbose=0)
            run = model.fit(df_to_dataset(
                                X_train.iloc[train_index, :], 
                                batch_size=params['batch_size']),
                            validation_data=df_to_dataset(
                                X_train.iloc[val_index, :], 
                                batch_size=1024),
                            epochs=1_000,
                            callbacks=[model_checkpoint, 
                                       early_stopping],
                            verbose=0)

            end_time = time()

            rounds = np.argmin(
                     run.history['val_mean_squared_error']) + 1

            model = tf.keras.models.load_model('best.model')
            shutil.rmtree('best.model')

            test_preds = model.predict(df_to_dataset(
                            X.iloc[test_index, :], shuffle=False, 
                            batch_size=1024)).flatten()
                            test_score = scoring(
                            X.iloc[test_index, :]['target'], 
                            test_preds)
            print(f"CV Fold {k+1} rmse:{test_score:0.5f} - {rounds} 
                  rounds - it took {end_time-start_time:0.0f} secs")
            validation_scores.append(test_score)

            if len(history[k]) >= 10:
                threshold = np.percentile(history[k], q=25)
                if test_score > threshold:
                    print(f"Early stopping for under-performing fold: 
                          threshold is {threshold:0.5f}")
                    return np.mean(validation_scores)

            history[k].append(test_score)
        return np.mean(validation_scores)
    return objective 
```

下一步是提供一个随机搜索的序列（作为从搜索空间中开始构建一些反馈的一种方式），并将结果作为起点收集。然后，我们可以将它们输入到贝叶斯优化中，并通过使用`forest_minimize`作为代理函数来继续：

```py
counter = 0
history = {i:list() for i in range(5)}
used_time = 0
gp_round = dummy_minimize(func=objective,
                          dimensions=space,
                          n_calls=10,
                          callback=[onstep],
                          random_state=0)
gc.collect()
x0, y0 = joblib.load('checkpoint.pkl')
gp_round = gp_minimize(func=objective,
                           x0=x0,  # already examined values for x
                           y0=y0,  # observed values for x0
                           dimensions=space,
                           n_calls=30,
                           n_initial_points=0,
                           callback=[onstep],
                           random_state=0)
gc.collect() 
```

注意，在随机搜索的前十轮之后，我们使用随机森林算法作为代理函数继续我们的搜索。这将确保比使用高斯过程获得更好的和更快的成果。

如前所述，在这个过程中，我们必须努力在现有时间和资源范围内使优化变得可行（例如，通过设置一个较低的`n_calls`数量）。因此，我们可以通过保存优化状态、检查获得的结果，并在之后决定继续或结束优化过程，不再投入更多时间和精力去寻找更好的解决方案，来分批进行搜索迭代。

## 使用 KerasTuner 创建更轻量化和更快的模型

如果前一节因为其复杂性而让您感到困惑，KerasTuner 可以为您提供一种快速设置优化的解决方案，无需太多麻烦。尽管它默认使用贝叶斯优化和高斯过程，但 KerasTuner 背后的新想法是**超带优化**。超带优化使用 bandit 方法来确定最佳参数（参见[`web.eecs.umich.edu/~mosharaf/Readings/HyperBand.pdf`](http://web.eecs.umich.edu/~mosharaf/Readings/HyperBand.pdf)）。这种方法与神经网络配合得相当好，因为神经网络的优化景观非常不规则和不连续，因此并不总是适合高斯过程。

请记住，你无法避免构建一个使用输入超参数构建自定义网络的函数；KerasTuner 只是使这个过程变得更加容易。

让我们从开始讲起。KerasTuner([`keras.io/keras_tuner/`](https://keras.io/keras_tuner/))被其创造者、Keras 的创始人*弗朗索瓦·肖莱特*宣布为“为 Keras 模型提供灵活且高效的超参数调整”。

肖莱特提出的运行 KerasTuner 的配方由简单的步骤组成，从你现有的 Keras 模型开始：

1.  将你的模型封装在一个函数中，其中`hp`作为第一个参数。

1.  在函数的开始处定义超参数。

1.  将 DNN 的静态值替换为超参数。

1.  编写代码，从给定的超参数构建一个复杂的神经网络模型。

1.  如果需要，在构建网络时动态定义超参数。

我们现在将通过使用一个示例来探索所有这些步骤如何在 Kaggle 比赛中为你工作。目前，KerasTuner 是任何 Kaggle 笔记本提供的堆栈的一部分，因此你不需要安装它。此外，TensorFlow 附加组件是笔记本预安装的包的一部分。

如果你没有使用 Kaggle 笔记本并且需要尝试 KerasTuner，你可以使用以下命令轻松安装：

```py
!pip install -U keras-tuner
!pip install -U tensorflow-addons 
```

你可以在 Kaggle 笔记本上找到这个示例已经设置好的链接：[`www.kaggle.com/lucamassaron/kerastuner-for-imdb/`](https://www.kaggle.com/lucamassaron/kerastuner-for-imdb/)。

我们的第一步是导入必要的包（为一些命令创建快捷方式，例如`pad_sequences`），并直接从 Keras 上传我们将使用的数据：

```py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
pad_sequences = keras.preprocessing.sequence.pad_sequences
imdb = keras.datasets.imdb(train_data, train_labels),
(test_data, test_labels) = imdb.load_data(num_words=10000)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.30,
                 shuffle=True, random_state=0) 
```

这次，我们使用的是 IMDb 数据集，该数据集包含在 Keras 包中([`keras.io/api/datasets/imdb/`](https://keras.io/api/datasets/imdb/))。该数据集具有一些有趣的特性：

+   这是一个包含 25,000 条 IMDb 电影评论的数据集。

+   评论通过情感（正面/负面）进行标记。

+   目标类别是平衡的（因此准确率作为评分指标）。

+   每条评论都被编码为一个单词索引列表（整数）。

+   为了方便，单词通过整体频率进行索引。

此外，它已经在 Kaggle 上的一项关于词嵌入的流行比赛中成功应用([`www.kaggle.com/c/word2vec-nlp-tutorial/overview`](https://www.kaggle.com/c/word2vec-nlp-tutorial/overview))。

这个例子涉及自然语言处理。这类问题通常是通过使用基于 LSTM 或 GRU 层的**循环神经网络**（**RNNs**）来解决的。BERT、RoBERTa 和其他基于转换器的模型通常能取得更好的结果——作为依赖大型语言语料库的预训练模型——但这并不一定在所有问题中都成立，RNNs 可以证明是一个强大的基线，可以击败或是一个神经模型集成的良好补充。在我们的例子中，所有单词都已经进行了数字索引。我们只是将表示填充的数字代码添加到现有的索引中（这样我们就可以轻松地将所有文本归一化到短语长度），句子的开始，一个未知单词和一个未使用的单词：

```py
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text]) 
```

下一步涉及创建一个用于**注意力**的自定义层。注意力是转换器模型的基础，也是近年来神经自然语言处理中最具创新性的想法之一。

关于这些类型层如何工作的所有细节，请参阅关于注意力的开创性论文：Vaswani, A. 等人。*注意力即一切*。神经信息处理系统进展。2017 ([`proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf`](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf))。

注意力的概念可以很容易地传达。LSTM 和 GRU 层输出处理过的序列，但并非这些输出序列中的所有元素对于你的预测都一定是重要的。你不必使用池化层在分层序列上平均所有输出序列，实际上你可以对它们进行**加权平均**（并且在训练阶段学习使用正确的权重）。这种加权过程（**注意力**）无疑会提高你将要传递的结果。当然，你可以通过使用多个注意力层（我们称之为**多头注意力**）使这种方法更加复杂，但在我们的例子中，一个单独的层就足够了，因为我们想证明在这个问题上使用注意力比简单地平均或简单地连接所有结果更有效：

```py
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, RepeatVector, dot, multiply, Permute, Lambda
K = keras.backend
def attention(layer):
    # --- Attention is all you need --- #
    _,_,units = layer.shape.as_list()
    attention = Dense(1, activation='tanh')(layer)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    attention = Permute([2, 1])(attention)
    representation = multiply([layer, attention])
    representation = Lambda(lambda x: K.sum(x, axis=-2), 
                            output_shape=(units,))(representation)
    # ---------------------------------- #
    return representation 
```

作为我们对这个问题的 DNN 架构进行实验的进一步变化，我们还想测试使用不同类型的优化器（如**修正的 Adam**（自适应学习率的 Adam 优化器；阅读这篇帖子了解更多信息：[`lessw.medium.com/new-state-of-the-art-ai-optimizer-rectified-adam-radam-5d854730807b`](https://lessw.medium.com/new-state-of-the-art-ai-optimizer-rectified-adam-radam-5d854730807b)）或**随机加权平均**（**SWA**）的有效性。SWA 是一种基于修改后的学习率计划来平均优化过程中遍历的权重的方法：如果你的模型倾向于过拟合或过度估计，SWA 有助于接近最优解，并且在 NLP 问题中已被证明是有效的。

```py
def get_optimizer(option=0, learning_rate=0.001):
    if option==0:
        return tf.keras.optimizers.Adam(learning_rate)
    elif option==1:
        return tf.keras.optimizers.SGD(learning_rate, 
                                       momentum=0.9, nesterov=True)
    elif option==2:
        return tfa.optimizers.RectifiedAdam(learning_rate)
    elif option==3:
        return tfa.optimizers.Lookahead(
                   tf.optimizers.Adam(learning_rate), sync_period=3)
    elif option==4:
        return tfa.optimizers.SWA(tf.optimizers.Adam(learning_rate))
    elif option==5:
        return tfa.optimizers.SWA(
                   tf.keras.optimizers.SGD(learning_rate, 
                                       momentum=0.9, nesterov=True))
    else:
        return tf.keras.optimizers.Adam(learning_rate) 
```

定义了两个关键函数后，我们现在面临最重要的编码函数：该函数将根据参数提供不同的神经网络架构。我们不会对所有我们想要连接到不同架构选择的参数进行编码；我们只提供 `hp` 参数，它应该包含我们想要使用的所有可能的参数，并且将由 KerasTuner 运行。除了函数输入中的 `hp` 之外，我们还固定了词汇表的大小和要填充的长度（如果实际长度较短，则添加虚拟值；如果长度较长，则截断短语）：

```py
layers = keras.layers
models = keras.models

def create_tunable_model(hp, vocab_size=10000, pad_length=256):
    # Instantiate model params
    embedding_size = hp.Int('embedding_size', min_value=8, 
                            max_value=512, step=8)
    spatial_dropout = hp.Float('spatial_dropout', min_value=0, 
                               max_value=0.5, step=0.05)
    conv_layers = hp.Int('conv_layers', min_value=1,
                         max_value=5, step=1)
    rnn_layers = hp.Int('rnn_layers', min_value=1,
                        max_value=5, step=1)
    dense_layers = hp.Int('dense_layers', min_value=1,
                          max_value=3, step=1)
    conv_filters = hp.Int('conv_filters', min_value=32, 
                          max_value=512, step=32)
    conv_kernel = hp.Int('conv_kernel', min_value=1,
                         max_value=8, step=1)
    concat_dropout = hp.Float('concat_dropout', min_value=0, 
                              max_value=0.5, step=0.05)
    dense_dropout = hp.Float('dense_dropout', min_value=0, 
                             max_value=0.5, step=0.05) 
```

函数的第一部分，我们简单地从 `hp` 参数中恢复所有设置。我们还明确指出了每个参数的搜索范围。与迄今为止我们看到的所有解决方案相反，这部分工作是在模型函数内部完成的，而不是外部。

函数继续通过使用从 `hp` 中提取的参数定义不同的层。在某些情况下，一个参数会开启或关闭网络的一部分，执行特定的数据处理。例如，在代码中我们插入了一个图的分支（`conv_filters` 和 `conv_kernel`），它使用卷积层处理单词序列，这些卷积层在它们的 1D 形式中，也可以对 NLP 问题很有用，因为它们可以捕捉到 LSTMs 可能更难把握的局部单词序列和意义。

现在我们可以定义实际的模型：

```py
 inputs = layers.Input(name='inputs',shape=[pad_length])
    layer  = layers.Embedding(vocab_size, embedding_size, 
                              input_length=pad_length)(inputs)
    layer  = layers.SpatialDropout1D(spatial_dropout)(layer)
    for l in range(conv_layers):
        if l==0:
            conv = layers.Conv1D(filters=conv_filters, 
                       kernel_size=conv_kernel, padding='valid',
                       kernel_initializer='he_uniform')(layer)
        else:
            conv = layers.Conv1D(filters=conv_filters,  
                       kernel_size=conv_kernel, padding='valid', 
                       kernel_initializer='he_uniform')(conv) 
    avg_pool_conv = layers.GlobalAveragePooling1D()(conv)
    max_pool_conv = layers.GlobalMaxPooling1D()(conv)
    representations = list()
    for l in range(rnn_layers):

        use_bidirectional = hp.Choice(f'use_bidirectional_{l}',
                                      values=[0, 1])
        use_lstm = hp.Choice(f'use_lstm_{l}', values=[0, 1])
        units = hp.Int(f'units_{l}', min_value=8, max_value=512, step=8)
        if use_lstm == 1:
            rnl = layers.LSTM
        else:
            rnl = layers.GRU
        if use_bidirectional==1:
            layer = layers.Bidirectional(rnl(units, 
                              return_sequences=True))(layer)
        else:
            layer = rnl(units, return_sequences=True)(layer)
        representations.append(attention(layer))
    layer = layers.concatenate(representations + [avg_pool_conv, 
                                                  max_pool_conv])
    layer = layers.Dropout(concat_dropout)(layer)
    for l in range(dense_layers):
        dense_units = hp.Int(f'dense_units_{l}', min_value=8, 
                             max_value=512, step=8)
        layer = layers.Dense(dense_units)(layer)
        layer = layers.LeakyReLU()(layer)
        layer = layers.Dropout(dense_dropout)(layer)
    layer = layers.Dense(1, name='out_layer')(layer)
    outputs = layers.Activation('sigmoid')(layer)
    model = models.Model(inputs=inputs, outputs=outputs) 
```

我们首先定义输入层，并使用随后的嵌入层对其进行转换，该嵌入层将序列值编码到密集层中。在处理过程中应用了一些 `SpatialDropout1D` 正则化，这是一个会随机丢弃输出矩阵的整个列的功能（标准 dropout 会在矩阵中随机丢弃单个元素）。在这些初始阶段之后，我们将网络分为基于卷积（`Conv1D`）的一个管道和基于循环层（GRU 或 LSTM）的另一个管道。在循环层之后，我们应用了注意力层。最后，这两个管道的输出被连接起来，经过几个更多的密集层后，到达最终的输出节点，一个 sigmoid，因为我们必须表示一个范围在 0 到 1 之间的概率。

在模型定义之后，我们设置学习参数并编译模型，然后返回它：

```py
 hp_learning_rate = hp.Choice('learning_rate', 
                                 values=[0.002, 0.001, 0.0005])
    optimizer_type = hp.Choice('optimizer', values=list(range(6)))
    optimizer = get_optimizer(option=optimizer_type,  
                              learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model 
```

注意，我们使用 Keras 的功能 API 构建了模型，而不是顺序 API。实际上，我们建议您避免使用顺序 API；它更容易设置，但严重限制了您的潜在架构。

到目前为止，大部分工作已经完成。作为一个建议，我们自己使用 KerasTuner 进行了许多优化后，我们更喜欢首先构建一个*非参数化*模型，使用我们想要测试的所有可能的架构特性，将网络中相互排斥的部分设置为最复杂的解决方案。在我们设置了生成函数并且我们的模型看起来运行正常之后，例如，我们可以表示其图，并成功地将一些示例作为测试拟合。之后，我们开始将参数化变量插入架构，并设置`hp`参数定义。

根据我们的经验，立即从参数化函数开始将需要更多的时间和调试努力。KerasTuner 背后的想法是让您将 DNN 视为一组模块化电路，并帮助您优化数据在其中的流动方式。

现在，我们导入 KerasTuner。首先，我们设置调优器本身，然后开始搜索：

```py
import keras_tuner as kt
tuner = kt.BayesianOptimization(hypermodel=create_tunable_model,
                                objective='val_acc',
                                max_trials=100,
                                num_initial_points=3,
                                directory='storage',
                                project_name='imdb',
                                seed=42)
tuner.search(train_data, train_labels, 
             epochs=30,
             batch_size=64, 
             validation_data=(val_data, val_labels),
             shuffle=True,
             verbose=2,
             callbacks = [EarlyStopping('val_acc',
                                        patience=3,
                                        restore_best_weights=True)]
             ) 
```

作为调优器，我们选择了贝叶斯优化，但您也可以尝试 Hyperband 调优器（[`keras.io/api/keras_tuner/tuners/hyperband/`](https://keras.io/api/keras_tuner/tuners/hyperband/)）并检查它是否更适合您的问题。我们将我们的模型函数提供给`hypermodel`参数。然后，我们使用字符串或函数设置目标，最大尝试次数（如果没有什么更多的事情要做，KerasTuner 将提前停止），以及初始随机尝试次数——越多越好——以告知贝叶斯过程。早期停止是建模 DNN 的标准且表现良好的实践，您绝对不能忽视。最后，但同样重要的是，我们设置了保存搜索结果的目录以及用于优化步骤可重复性的种子数字。

搜索阶段就像运行一个标准的 Keras 模型拟合一样，而且这一点非常重要——它接受回调。因此，您可以轻松地将早期停止添加到您的模型中。在这种情况下，给定的 epoch 数应被视为最大 epoch 数。您可能还希望优化批大小，我们在这个例子中没有这样做。这仍然需要一些额外的工作，但您可以通过阅读这个 GitHub 已关闭的问题来了解如何实现它：[`github.com/keras-team/keras-tuner/issues/122`](https://github.com/keras-team/keras-tuner/issues/122)。

优化完成后，您可以提取最佳参数并保存最佳模型，而无需重新训练：

```py
best_hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)
print(best_hps.values)
model.summary()
model.save("best_model.h5") 
```

在这个例子中，KerasTuner 找到了一个解决方案，它使用了：

+   一个更大的嵌入层

+   只简单的 GRU 和 LSTM 层（没有双向层）

+   多个一维卷积层（Conv1D）的堆叠

+   更多和更大的密集层

有趣的是，这个解决方案不仅更有效，而且比我们基于直觉和问题经验的前期尝试更轻、更快。

Chollet 本人建议使用 KerasTuner 不仅是为了让你的深度神经网络（DNNs）表现更好，而且是为了将它们缩小到更易于管理的尺寸，这在代码竞赛中可能起到决定性作用。这允许你在有限的推理时间内，结合更多协同工作的模型。

如果你想要查看更多使用 KerasTuner 的示例，François Chollet 还创建了一系列用于 Kaggle 竞赛的 Notebooks，以展示他的优化器的运作和功能：

+   [Keras-kerastuner 最佳实践](https://www.kaggle.com/fchollet/keras-kerastuner-best-practices) 用于 *数字识别* 数据集

+   [Keras-kerastuner 最佳实践](https://www.kaggle.com/fchollet/titanic-keras-kerastuner-best-practices) 用于 *泰坦尼克号* 数据集

+   [Keras-kerastuner 最佳实践](https://www.kaggle.com/fchollet/moa-keras-kerastuner-best-practices) 用于 *作用机制（MoA）预测* 竞赛

## Optuna 中的 TPE 方法

我们通过另一个有趣的工具和贝叶斯优化的方法来完成我们对贝叶斯优化的概述。正如我们讨论的，Scikit-optimize 使用高斯过程（以及树算法），并直接模拟代理函数和获取函数。

作为对这些主题的提醒，**代理函数**帮助优化过程模拟尝试一组超参数时的潜在性能结果。代理函数是使用之前的实验及其结果构建的；它只是一个应用于预测特定机器学习算法在特定问题上的行为的预测模型。对于提供给代理函数的每个参数输入，你都会得到一个预期的性能输出。这既直观又相当易于操作，正如我们所看到的。

**获取函数**则指出哪些超参数组合可以被测试，以改善代理函数预测机器学习算法性能的能力。它也有助于真正测试我们是否可以根据代理函数的预测达到顶级性能。这两个目标代表了贝叶斯优化过程中的*探索*部分（进行实验）和*利用*部分（测试性能）。

相反，基于**TPE**的优化器通过估计参数值的成功可能性来解决问题。换句话说，它们通过连续的细化来模拟参数自身的成功分布，为更成功的值组合分配更高的概率。

在这种方法中，通过这些分布将超参数集分为好和坏，这些分布充当贝叶斯优化中的代理和获取函数，因为分布告诉你在哪里采样以获得更好的性能或探索存在不确定性的地方。

要探索 TPE 的技术细节，我们建议阅读 Bergstra，J.等人撰写的*超参数优化算法*。神经网络信息处理系统 24 卷，2011 年（[`proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf`](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)）。

因此，TPE 可以通过从调整后的参数概率分布中进行采样，来模拟搜索空间并同时建议算法下一步可以尝试的内容。

很长一段时间里，**Hyperopt**是那些喜欢使用 TPE 而不是基于高斯过程的贝叶斯优化的用户的选项。然而，2018 年 10 月，Optuna 出现在开源领域，由于其多功能性（它也适用于神经网络甚至集成），速度和效率，以及与先前优化器相比找到更好解决方案的能力，它已成为 Kagglers 的首选选择。

在本节中，我们将展示设置搜索有多容易，在 Optuna 术语中，这被称为*研究*。你所需要做的就是编写一个目标函数，该函数接受 Optuna 要测试的参数作为输入，然后返回一个评估结果。验证和其他算法方面可以在目标函数内以直接的方式处理，也可以使用对函数本身外部的变量的引用（既可以是全局变量也可以是局部变量）。Optuna 还允许**剪枝**，即表示某个特定实验进展不佳，Optuna 可以停止并忘记它。Optuna 提供了一系列激活此回调的函数（见[`optuna.readthedocs.io/en/stable/reference/integration.html`](https://optuna.readthedocs.io/en/stable/reference/integration.html)）；在此之后，算法将为你高效地运行一切，这将显著减少优化所需的时间。

所有这些都在我们的下一个示例中。我们回到优化*30 Days of ML*竞赛。这次，我们试图找出哪些参数使 XGBoost 适用于这个竞赛。

你可以在[`www.kaggle.com/lucamassaron/optuna-bayesian-optimization`](https://www.kaggle.com/lucamassaron/optuna-bayesian-optimization)找到这个示例的 Notebook。

作为第一步，我们上传库和数据，就像之前一样：

```py
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import optuna
from optuna.integration import XGBoostPruningCallback
# Loading data 
X_train = pd.read_csv("../input/30-days-of-ml/train.csv").iloc[:100_000, :]
X_test = pd.read_csv("../input/30-days-of-ml/test.csv")
# Preparing data as a tabular matrix
y_train = X_train.target
X_train = X_train.set_index('id').drop('target', axis='columns')
X_test = X_test.set_index('id')
# Pointing out categorical features
categoricals = [item for item in X_train.columns if 'cat' in item]
# Dealing with categorical data using OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
X_train[categoricals] = ordinal_encoder.fit_transform(X_train[categoricals])
X_test[categoricals] = ordinal_encoder.transform(X_test[categoricals]) 
```

当使用 Optuna 时，你只需定义一个包含模型、交叉验证逻辑、评估指标和搜索空间的目標函数。

自然地，对于数据，你可以参考函数本身之外的对象，这使得函数的构建变得容易得多。例如，在 KerasTuner 中，你需要一个基于 Optuna 类的特殊输入参数：

```py
def objective(trial):

    params = {
            'learning_rate': trial.suggest_float("learning_rate", 
                                                 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_loguniform("reg_lambda", 
                                                   1e-9, 100.0),
            'reg_alpha': trial.suggest_loguniform("reg_alpha", 
                                                  1e-9, 100.0),
            'subsample': trial.suggest_float("subsample", 0.1, 1.0),
            'colsample_bytree': trial.suggest_float(
                                      "colsample_bytree", 0.1, 1.0),
            'max_depth': trial.suggest_int("max_depth", 1, 7),
            'min_child_weight': trial.suggest_int("min_child_weight", 
                                                  1, 7),
            'gamma': trial.suggest_float("gamma", 0.1, 1.0, step=0.1)
    }
    model = XGBRegressor(
        random_state=0,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        n_estimators=10_000,
        **params
    )

    model.fit(x, y, early_stopping_rounds=300, 
              eval_set=[(x_val, y_val)], verbose=1000,
              callbacks=[XGBoostPruningCallback(trial, 'validation_0-rmse')])
    preds = model.predict(x_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse 
```

在这个例子中，出于性能考虑，我们不会进行交叉验证，而是使用一个固定的数据集进行训练，一个用于验证（早期停止），一个用于测试目的。在这个例子中，我们使用 GPU，并且我们还对可用的数据进行子集化，以便将 60 次试验的执行时间缩短到合理的长度。如果你不想使用 GPU，只需从`XGBRegressor`实例化中移除`tree_method`和`predictor`参数。同时请注意，我们如何在`fit`方法中设置回调，以便提供关于模型性能的 Optuna 反馈，这样优化器就可以在实验表现不佳时提前停止，为其他尝试腾出空间。

```py
x, x_val, y, y_val = train_test_split(X_train, y_train, random_state=0,
                                      test_size=0.2)
x, x_test, y, y_test = train_test_split(x, y, random_state=0, test_size=0.25)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100) 
```

另一个值得注意的方面是，你可以根据你的问题选择优化最小化或最大化，因为 Scikit-optimize 只适用于最小化问题。

```py
print(study.best_value)
print(study.best_params) 
```

要完成运行，你只需打印或导出最佳测试性能和优化找到的最佳参数即可。

![Ruchi Bhatia](img/Ruchi_Bhatia.png)

Ruchi Bhatia

[`www.kaggle.com/ruchi798`](https://www.kaggle.com/ruchi798)

作为本章的总结，让我们来看最后一个访谈。这次，我们将与 Ruchi Bhatia 进行对话，她是数据集和笔记的大师级人物。Ruchi 目前是卡内基梅隆大学的硕士研究生，OpenMined 的数据科学家，以及 Z by HP 的数据科学全球大使。

你最喜欢的竞赛类型是什么？为什么？在技术和解决方法方面，你在 Kaggle 上的专长是什么？

*我最喜欢的竞赛类型是自然语言处理和数据分析竞赛。多语言能力在我的主要关注点和兴趣——自然语言处理中发挥了重要作用。*

*至于数据分析竞赛，我喜欢从复杂的数据中找出意义，并用数据支持我的答案！Kaggle 上的每一场竞赛都是新颖的，需要不同的技术。我主要遵循数据驱动的算法选择方法，没有固定的偏好。*

你是如何应对 Kaggle 竞赛的？这种方法与你在日常工作中所做的方法有何不同？

*当一项新的竞赛被宣布时，我的首要任务是深入理解问题陈述。有时问题陈述可能超出了我们的舒适区或领域，因此确保我们在进行探索性数据分析之前充分理解它们是至关重要的。在进行 EDA 的过程中，我的目标是理解数据分布，并专注于了解手头的数据。在这个过程中，我们可能会遇到模式，我们应该努力理解这些模式，并为异常值和特殊情况形成假设。*

*在此之后，我花时间理解竞争指标。创建一个无泄漏的交叉验证策略是我的下一步。之后，我选择一个基线模型并提交我的第一个版本。如果本地验证和竞赛排行榜之间的相关性不令人满意，我会根据需要迭代，以理解可能的差异并加以考虑。*

*然后我继续随着时间的推移改进我的建模方法。除此之外，调整参数和尝试新的实验有助于了解什么最适合手头的数据（确保在整个过程中防止过拟合）。最后，在竞赛的最后几周，我执行模型集成并检查我解决方案的鲁棒性。*

*至于我在 Kaggle 之外的项目，我大部分时间都花在数据收集、清理和从数据中获得相关价值上。*

Kaggle 是否帮助了你的职业生涯？如果是，是如何帮助的？

*Kaggle 极大地帮助我加速了我的职业生涯。它不仅帮助我发现我对数据科学的热情，还激励我有效地贡献并保持一致性。这是一个在指尖上有大量数据可以尝试动手实验并展示我们工作的全球平台。此外，我们的工作易于访问，因此我们可以触及更广泛的受众。*

*我已经将大部分 Kaggle 工作用于我的作品集，以表明我在迄今为止的旅程中完成的工作的多样性。Kaggle 竞赛旨在解决新颖和现实世界的问题，我认为雇主寻找我们解决这类问题的能力和天赋。我还整理了广泛的数据集，这有助于突出我在处理原始数据方面的敏锐度。这些项目帮助我获得了多个工作机会。*

在你的经验中，不经验丰富的 Kagglers 通常忽略了什么？你现在知道什么，而当你刚开始时希望知道的呢？

*在我的经验中，我注意到许多 Kagglers 在竞赛中的排名不符合他们的预期时，会感到沮丧。经过几周甚至几个月的辛勤工作，我明白他们为什么可能会早早放弃，但赢得 Kaggle 竞赛并非易事。有来自不同教育背景和工作经验的人参与竞争，有勇气尝试就足够了。我们应该专注于个人的成长，看看我们在旅程中走了多远。*

对于数据分析或机器学习，有没有任何特定的工具或库你推荐使用？

*综合性的探索性数据分析结合相关的可视化有助于我们发现数据趋势和背景，从而改进我们的方法。由于我相信可视化的重要性，我最喜欢的数据科学库将是 Seaborn 和 TensorBoard。Seaborn 用于 EDA，TensorBoard 用于机器学习工作流程中的可视化。我也偶尔使用 Tableau。*

当人们参加比赛时，他们应该记住或做最重要的事情是什么？

*当人们进入比赛时，我相信他们应该为深入分析问题陈述和研究做好准备。Kaggle 的比赛尤其具有挑战性，并且在许多情况下有助于解决现实生活中的问题。人们应该保持积极的心态，不要灰心丧气。Kaggle 的比赛提供了学习和成长的最佳机会！*

# 摘要

在本章中，我们详细讨论了超参数优化作为提高模型性能和在排行榜上获得更高分数的方法。我们首先解释了 Scikit-learn 的代码功能，例如网格搜索和随机搜索，以及较新的折半算法。

然后，我们进一步讨论了贝叶斯优化，并探讨了 Scikit-optimize、KerasTuner，最后是 Optuna。我们花了更多的时间讨论通过高斯过程直接建模代理函数以及如何对其进行黑客攻击，因为它可以让你有更强的直觉和更灵活的解决方案。我们认识到，目前 Optuna 已经成为 Kagglers 中的黄金标准，无论是表格竞赛还是深度神经网络竞赛，因为它在 Kaggle 笔记本允许的时间内更快地收敛到最优参数。

然而，如果你想从竞争中脱颖而出，你应该努力测试其他优化器的解决方案。

在下一章中，我们将继续讨论另一种提高你在 Kaggle 比赛中表现的方法：集成模型。通过了解平均、混合和堆叠的工作原理，我们将展示如何通过仅调整超参数之外的方式提升你的结果。

# 加入我们书的 Discord 空间

加入本书的 Discord 工作空间，参加每月的“问我任何问题”活动，与作者交流：

[`packt.link/KaggleDiscord`](https://packt.link/KaggleDiscord)

![二维码](img/QR_Code40480600921811704671.png)
