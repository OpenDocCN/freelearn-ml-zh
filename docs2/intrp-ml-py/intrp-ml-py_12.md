

# 单调约束和模型调优以提高解释性

大多数模型类都有超参数，可以通过调整来提高执行速度、增强预测性能和减少过拟合。减少过拟合的一种方法是在模型训练中引入正则化。在*第三章*，*解释性挑战*中，我们将正则化称为一种补救的解释性属性，它通过惩罚或限制来降低复杂性，迫使模型学习输入的更稀疏表示。正则化模型具有更好的泛化能力，这就是为什么强烈建议使用正则化调整模型以避免对训练数据的过拟合。作为副作用，正则化模型通常具有更少的特征和交互，这使得模型更容易解释——*更少的噪声意味着更清晰的信号*！

尽管有许多超参数，但我们只会关注那些通过控制过拟合来提高解释性的参数。在一定意义上，我们还将回顾通过前几章中探讨的类别不平衡相关超参数来减轻偏差。

*第二章*，*解释性的关键概念*，解释了三个影响解释性的模型属性：非线性、交互性和非单调性。如果模型自行其是，它可能会学习到一些虚假的、反直觉的非线性和交互性。正如在第十章，*为解释性进行特征选择和工程*中讨论的那样，可以通过仔细的特征工程来设置限制以防止这种情况。然而，我们如何为单调性设置限制呢？在本章中，我们将学习如何使用单调约束来实现这一点。同样，单调约束可以是模型与特征工程的对应物，而正则化可以是我们在第十章中涵盖的特征选择方法的模型对应物！

本章我们将涵盖的主要主题包括：

+   通过特征工程设置限制

+   调整模型以提高解释性

+   实现模型约束

# 技术要求

本章的示例使用了`mldatasets`、`pandas`、`numpy`、`sklearn`、`xgboost`、`lightgbm`、`catboost`、`tensorflow`、`bayes_opt`、`tensorflow_lattice`、`matplotlib`、`seaborn`、`scipy`、`xai`和`shap`库。如何安装这些库的说明在序言中。

本章的代码位于此处：[`packt.link/pKeAh`](https://packt.link/pKeAh)

# 任务

算法公平性问题具有巨大的社会影响，从福利资源的分配到救命手术的优先级，再到求职申请的筛选。这些机器学习算法可以决定一个人的生计或生命，而且往往是边缘化和最脆弱的群体从这些算法中受到最恶劣的对待，因为这些算法持续传播从数据中学到的系统性偏见。因此，贫困家庭可能被错误地归类为虐待儿童；种族少数群体在医疗治疗中可能被优先级过低；而女性可能被排除在高薪技术工作之外。即使在涉及不那么直接和个性化的风险的情况下，如在线搜索、Twitter/X 机器人账户和社交媒体档案，社会偏见如精英主义、种族主义、性别歧视和年龄歧视也会得到加强。

本章将继续延续第六章的主题，即*锚点和反事实解释*。如果您不熟悉这些技术，请回过头去阅读*第六章*，以获得对问题的深入了解。第六章中的再犯案例是算法偏差的一个例子。开发**COMPAS 算法**（其中**COMPAS**代表**矫正犯人管理配置文件替代制裁**）的公司的联合创始人承认，在没有与种族相关的问题的情况下很难给出分数。这种相关性是分数对非裔美国人产生偏见的主要原因之一。另一个原因是训练数据中黑人被告可能被过度代表。我们无法确定这一点，因为我们没有原始的训练数据，但我们知道非白人少数族裔在服刑人员群体中被过度代表。我们还知道，由于与轻微毒品相关罪行相关的编码歧视和黑人社区的过度执法，黑人通常在逮捕中被过度代表。

那么，我们该如何解决这个问题呢？

在*第六章*，*锚点和反事实解释*中，我们通过一个*代理模型*成功地证明了 COMPAS 算法存在偏见。对于本章，让我们假设记者发表了你的发现，一个算法正义倡导团体阅读了文章并联系了你。制作犯罪评估工具的公司没有对偏见承担责任，声称他们的工具只是反映了*现实*。该倡导团体雇佣你来证明机器学习模型可以被训练得对黑人被告的偏见显著减少，同时确保该模型仅反映经过验证的刑事司法*现实*。

这些被证实的现实包括随着年龄增长，再犯风险单调下降，以及与先前的强烈相关性，这种相关性随着年龄的增长而显著增强。学术文献支持的另一个事实是，女性在总体上显著不太可能再犯和犯罪。

在我们继续之前，我们必须认识到监督学习模型在从数据中捕获领域知识方面面临几个障碍。例如，考虑以下情况：

+   **样本、排除或偏见偏差**：如果您的数据并不能真正代表模型意图推广的环境，会怎样？如果是这样，领域知识将与您在数据中观察到的结果不一致。如果产生数据的那个环境具有固有的系统性或制度性偏见，那么数据将反映这些偏见。

+   **类别不平衡**：如第十一章“偏差缓解和因果推断方法”中所述，类别不平衡可能会使某些群体相对于其他群体更有利。在追求最高准确率的最有效途径中，模型将从这个不平衡中学习，这与领域知识相矛盾。

+   **非单调性**：特征直方图中的稀疏区域或高杠杆异常值可能导致模型在领域知识要求单调性时学习到非单调性，任何之前提到的问题都可能促成这一点。

+   **无影响力的特征**：一个未正则化的模型将默认尝试从所有特征中学习，只要它们携带一些信息，但这会阻碍从相关特征中学习或过度拟合训练数据中的噪声。一个更简约的模型更有可能支持由领域知识支持的特性。

+   **反直觉的交互作用**：如第十章“用于可解释性的特征选择和工程”中提到的，模型可能会偏好与领域知识支持的交互作用相反的反直觉交互作用。作为一种副作用，这些交互作用可能会使一些与它们相关的群体受益。在第六章“锚点和反事实解释”中，我们通过理解双重标准证明了这一点。

+   **例外情况**：我们的领域知识事实基于总体理解，但在寻找更细粒度的模式时，模型会发现例外，例如女性再犯风险高于男性的区域。已知现象可能不支持这些模型，但它们可能是有效的，因此我们必须小心不要在我们的调整努力中抹去它们。

该倡导组织已验证数据仅足以代表佛罗里达州的一个县，并且他们已经向您提供了一个平衡的数据集。第一个障碍很难确定和控制。第二个问题已经得到解决。现在，剩下的四个问题就交给你来处理了！

# 方法

您已经决定采取三步走的方法，如下所示：

+   **使用特征工程设置护栏**：借鉴第六章“锚点和反事实解释”中学习到的经验，以及我们已有的关于先验和年龄的领域知识，我们将设计一些特征。

+   **调整模型以提高可解释性**：一旦数据准备就绪，我们将使用不同的类别权重和过拟合预防技术调整许多模型。这些方法将确保模型不仅泛化能力更好，而且更容易解释。

+   **实施模型约束**：最后但同样重要的是，我们将对最佳模型实施单调性和交互约束，以确保它们不会偏离可信和公平的交互。

在最后两个部分中，我们将确保模型准确且公平地执行。我们还将比较数据和模型之间的再犯风险分布，以确保它们一致。

# 准备工作

你可以在这里找到这个示例的代码：[`github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/blob/main/12/Recidivism_part2.ipynb`](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/blob/main/12/Recidivism_part2.ipynb)

## 加载库

要运行此示例，您需要安装以下库：

+   `mldatasets` 用于加载数据集

+   `pandas` 和 `numpy` 用于操作

+   `sklearn`（scikit-learn）、`xgboost`、`lightgbm`、`catboost`、`tensorflow`、`bayes_opt` 和 `tensorflow_lattice` 用于分割数据和拟合模型

+   `matplotlib`、`seaborn`、`scipy`、`xai` 和 `shap` 以可视化解释

您应该首先加载所有这些库，如下所示：

```py
import math
import os
import copy
import mldatasets
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, metrics,\
    linear_model, svm, neural_network, ensemble
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from bayes_opt import BayesianOptimization
import tensorflow_lattice as tfl
from tensorflow.keras.wrappers.scikit_learn import\
                                                  KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import xai
import shap 
```

让我们检查 `tensorflow` 是否加载了正确的版本，使用 `print(tf.__version__)`。这应该是 2.8 版本及以上。

## 理解和准备数据

我们将数据以这种方式加载到我们称为 `recidivism_df` 的 DataFrame 中：

```py
recidivism_df = mldatasets.**load**("recidivism-risk-balanced") 
```

应该有超过 11,000 条记录和 11 个列。我们可以使用 `info()` 验证这一点，如下所示：

```py
recidivism_df.info() 
```

上一段代码输出了以下内容：

```py
RangeIndex: 11142 entries, 0 to 11141
Data columns (total 12 columns):
#   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
0   sex                      11142 non-null  object
1   age                      11142 non-null  int64
2   race                     11142 non-null  object
3   juv_fel_count            11142 non-null  int64
4   juv_misd_count           11142 non-null  int64
5   juv_other_count          11142 non-null  int64
6   priors_count             11142 non-null  int64
7   c_charge_degree          11142 non-null  object
8   days_b_screening_arrest  11142 non-null  float64
9   length_of_stay           11142 non-null  float64
10  compas_score             11142 non-null  int64
11  is_recid                 11142 non-null  int64
dtypes: float64(2), int64(7), object(3) 
```

输出检查无误。没有缺失值，除了三个特征（`sex`、`race` 和 `charge_degree`）外，所有特征都是数值型的。这是我们用于 *第六章*，*锚点和反事实解释* 的相同数据，因此数据字典完全相同。然而，数据集已经通过采样方法进行了平衡，这次它没有为我们准备，因此我们需要这样做，但在这样做之前，让我们了解平衡做了什么。

### 验证采样平衡

我们可以使用 XAI 的 `imbalance_plot` 检查 `race` 和 `is_recid` 的分布情况。换句话说，它将统计每个 `race`-`is_recid` 组合的记录数量。这个图将使我们能够观察每个 `race` 的被告中是否有再犯人数的不平衡。代码可以在以下片段中查看：

```py
categorical_cols_l = [
    'sex', 'race', 'c_charge_degree', 'is_recid', 'compas_score'
]
xai.**imbalance_plot**(
    recidivism_df,
    'race',
    'is_recid',
    categorical_cols=categorical_cols_l
) 
```

前面的代码输出了*图 12.1*，它描述了所有种族的`is_recid=0`和`is_recid=1`的数量相等。然而，**其他**种族在数量上与其他种族不相等。顺便提一下，这个数据集版本将所有其他种族都归入了**其他**类别，选择不`upsample` **其他**或`downsample`其他两个种族以实现总数相等，是因为它们在被告人口中代表性较低。这种平衡选择是在这种情况下可以做的许多选择之一。从人口统计学角度看，这完全取决于你的数据应该代表什么。被告？囚犯？普通民众中的平民？以及在哪一层面？县一级？州一级？国家一级？

输出结果如下：

![图片](img/B18406_12_01.png)

图 12.1：按种族分布的两年再犯率（is_recid）

接下来，让我们计算每个特征与目标变量单调相关性的程度。Spearman 等级相关系数在本章中将起到关键作用，因为它衡量了两个特征之间的单调性。毕竟，本章的一个技术主题是单调约束，主要任务是产生一个显著减少偏差的模型。

我们首先创建一个新的 DataFrame，其中不包含`compas_score`（`recidivism_corr_df`）。使用这个 DataFrame，我们输出一个带有`feature`列的彩色 DataFrame，其中包含前 10 个特征的名称，以及另一个带有所有 10 个特征与第 11 个特征（目标变量）的 Spearman 相关系数（`correlation_to_target`）。代码如下所示：

```py
recidivism_corr_df = recidivism_df.**drop**(
    ['compas_score'], axis=1
)
pd.**DataFrame**(
    {'feature': recidivism_corr_df.columns[:-1],
     'correlation_to_target':\
          scipy.stats.**spearmanr**(recidivism_corr_df).\
          correlation[10,:-1]
    }
).style.background_gradient(cmap='coolwarm') 
```

前面的代码输出了*图 12.2*所示的 DataFrame。最相关的特征是`priors_count`，其次是`age`、三个青少年计数和`sex`。`c_charge_degree`、`days_b_screening_arrest`、`length_of_stay`和`race`的系数可以忽略不计。

输出结果如下：

![表格描述自动生成](img/B18406_12_02.png)

图 12.2：在特征工程之前，所有特征对目标变量的 Spearman 系数

接下来，我们将学习如何使用特征工程将一些领域知识“嵌入”到特征中。

# 使用特征工程设置护栏

在*第六章*，*锚点和反事实解释*中，我们了解到除了`race`之外，在我们解释中最突出的特征是`age`、`priors_count`和`c_charge_degree`。幸运的是，数据现在已经平衡，因此这种不平衡导致的种族偏见现在已经消失。然而，通过锚点和反事实解释，我们发现了一些令人不安的不一致性。在`age`和`priors_count`的情况下，这些不一致性是由于这些特征的分布方式造成的。我们可以通过特征工程来纠正分布问题，从而确保模型不会从不均匀的分布中学习。在`c_charge_degree`的情况下，由于它是分类的，它缺乏可识别的顺序，这种缺乏顺序导致了不直观的解释。

在本节中，我们将研究**序列化**、**离散化**和**交互项**，这是通过特征工程设置护栏的三种方式。

## 序列化

```py
c_charge_degree category:
```

```py
recidivism_df.c_charge_degree.**value_counts()** 
```

前面的代码生成了以下输出：

```py
(F3)     6555
(M1)     2632
(F2)      857
(M2)      768
(F1)      131
(F7)      104
(MO3)      76
(F5)        7
(F6)        5
(NI0)       4
(CO3)       2
(TCX)       1 
```

每个电荷度数对应电荷的重力。这些重力有一个顺序，使用分类特征时会丢失。我们可以通过用相应的顺序替换每个类别来轻松解决这个问题。

我们可以对此顺序进行很多思考。例如，我们可以查看判决法或指南——对于不同的程度，实施了最低或最高的监禁年数。我们还可以查看这些人的平均暴力统计数据，并将这些信息分配给电荷度数。每个此类决策都存在潜在的偏见，如果没有充分的证据支持它，最好使用整数序列。所以，我们现在要做的就是创建一个字典（`charge_degree_code_rank`），将度数映射到从低到高对应的重力等级的数字。然后，我们可以使用`pandas`的`replace`函数使用这个字典来进行替换。以下代码片段中可以看到代码：

```py
charge_degree_code_rank = {
    '(F10)': 15, '(F9)':14, '(F8)':13,\
    '(F7)':12, '(TCX)':11, '(F6)':10, '(F5)':9,\
    '(F4)':8, '(F3)':7, '(F2)':6, '(F1)':5, '(M1)':4,\
    '(NI0)':4, '(M2)':3, '(CO3)':2, '(MO3)':1, '(X)':0
}
recidivism_df.c_charge_degree.**replace**(
    charge_degree_code_rank, inplace=True
) 
```

评估这种顺序如何对应再犯概率的一种方法是通过一条线图，显示随着电荷度数的增加，它如何变化。我们可以使用一个名为`plot_prob_progression`的函数来做这件事，它接受一个连续特征作为第一个参数（`c_charge_degree`），以衡量一个二元特征的概率（`is_recid`）。它可以按区间（`x_intervals`）分割连续特征，甚至可以使用分位数（`use_quantiles`）。最后，你可以定义轴标签和标题。以下代码片段中可以看到代码：

```py
mldatasets.**plot_prob_progression**(
    recidivism_df.**c_charge_degree**,
    recidivism_df.**is_recid**, x_intervals=12,
    use_quantiles=False,
    xlabel='Relative Charge Degree',
    title='Probability of Recidivism by Relative Charge Degree'
) 
```

前面的代码生成了图 12.3 中的图表。随着现在排名的电荷度数的增加，趋势是 2 年再犯的概率降低，除了排名 1。在概率下方有柱状图显示了每个排名的观测值的分布。由于分布非常不均匀，你应该谨慎对待这种趋势。你会注意到一些排名，如 0、8 和 13-15，没有在图表中，因为电荷度数的类别存在于刑事司法系统中，但在数据中不存在。

输出结果如下：

![图表，折线图  自动生成的描述](img/B18406_12_03.png)

图 12.3：按电荷度数的概率进展图

在特征工程方面，我们无法做更多的事情来改进 `c_charge_degree`，因为它已经代表了现在带有顺序的离散类别。除非我们有证据表明否则，任何进一步的转换都可能导致信息的大量丢失。另一方面，连续特征本质上具有顺序；然而，由于它们携带的精度水平，可能会出现问题。因为小的差异可能没有意义，但数据可能告诉模型否则。不均匀的分布和反直觉的交互只会加剧这个问题。

## 离散化

为了理解如何最佳地离散化我们的`年龄`连续特征，让我们尝试两种不同的方法。我们可以使用等宽离散化，也称为固定宽度箱或区间，这意味着箱的大小由 ![](img/B18406_12_001.png) 决定，其中 *N* 是箱的数量。另一种方法是使用等频离散化，也称为分位数，这确保每个箱大约有相同数量的观测值。尽管如此，有时由于直方图的偏斜性质，可能无法以 *N* 种方式分割它们，因此你可能最终得到 *N-1* 或 *N-2* 个分位数。

使用 `plot_prob_progression` 比较这两种方法很容易，但这次我们生成了两个图表，一个使用固定宽度箱（`use_quantiles=False`），另一个使用分位数（`use_quantiles=True`）。代码可以在下面的代码片段中看到：

```py
mldatasets.plot_**prob_progression**(
    recidivism_df.**age**,
    recidivism_df.**is_recid**,
    x_intervals=7,
    use_quantiles=False,
    title='Probability of Recidivism by Age Discretized in Fix-Width \
    Bins',
    xlabel='Age'
)
mldatasets.**plot_prob_progression**(
    recidivism_df.**age**,
    recidivism_df.**is_recid**,
    x_intervals=7, use_quantiles=True,
    title='Probability of Recidivism by Age Discretized \
    in Quantiles',
    xlabel='Age'
) 
Figure 12.4. By looking at the Observations portion of the fixed-width bin plot, you can tell that the histogram for the age feature is right-skewed, which causes the probability to shoot up for the last bin. The reason for this is that some outliers exist in this bin. On the other hand, the fixed-frequency (quantile) plot histogram is more even, and probability consistently decreases. In other words, it’s monotonic—as it should be, according to our domain knowledge on the subject.
```

输出结果如下：

![](img/B18406_12_04.png)

图 12.4：比较两种年龄离散化方法

很容易观察到为什么使用分位数对特征进行箱化是一个更好的方法。我们可以将 `age` 工程化为一个新的特征，称为 `age_group`。`pandas` 的 `qcut` 函数可以执行基于分位数的离散化。代码可以在下面的代码片段中看到：

```py
recidivism_df['age_group'] = pd.**qcut**(
    recidivism_df.**age**, 7, precision=0
).astype(str) 
```

因此，我们现在已经将`age`离散化为`age_group`。然而，必须注意的是，许多模型类会自动进行离散化，那么为什么还要这么做呢？因为这允许你控制其影响。否则，模型可能会选择不保证单调性的桶。例如，模型可能会在可能的情况下始终使用 10 个分位数。尽管如此，如果你尝试在`age`上使用这种粒度（`x_intervals=10`），你最终会在概率进展中遇到峰值。我们的目标是确保模型会学习到`age`和`is_recid`的发病率之间存在单调关系，如果我们允许模型选择可能或可能不达到相同目标的桶，我们就无法确定这一点。

我们将移除`age`，因为`age_group`包含了我们所需的所有信息。但是等等——你可能会问——移除这个变量会不会丢失一些重要信息？是的，但仅仅是因为它与`priors_count`的交互作用。所以，在我们丢弃任何特征之前，让我们检查这种关系，并意识到通过创建交互项，我们如何通过移除`age`来保留一些丢失的信息，同时保持交互。

## 交互项和非线性变换

我们从*第六章*，*锚点和反事实解释*中已经知道，`age`和`priors_count`是最重要的预测因子之一，我们可以观察到它们如何一起影响再犯的发病率（`is_recid`），使用`plot_prob_contour_map`。这个函数产生带有彩色等高线区域的等高线，表示不同的幅度。它们在地理学中很有用，可以显示海拔高度。在机器学习中，它们可以显示一个二维平面，表示特征与度量之间的交互。在这种情况下，维度是`age`和`priors_count`，度量是再犯的发病率。这个函数接收到的参数与`plot_prob_progression`相同，只是它接受对应于*x*轴和*y*轴的两个特征。代码可以在下面的代码片段中看到：

```py
mldatasets.plot_**prob_contour_map**(
    recidivism_df.**age**,
    recidivism_df.**priors_count**,
    recidivism_df.**is_recid**,
    use_quantiles=True,
    xlabel='Age',
    ylabel='Priors Count',
    title='Probability of Recidivism by Age/Priors Discretized in \
    Quantiles'
) 
Figure 12.5, which shows how, when discretized by quantiles, the probability of 2-year recidivism increases, the lower the age and the higher the priors_count. It also shows histograms for both features. priors_count is very right-skewed, so discretization is challenging, and the contour map does not offer a perfectly diagonal progression between the bottom right and top left. And if this plot looks familiar, it’s because it’s just like the partial dependence interaction plots we produced in *Chapter 4*, *Global Model-Agnostic Interpretation Methods*, except it’s not measured against the predictions of a model but the ground truth (is_recid). We must distinguish between what the data can tell us directly and what the model has learned from it.
```

输出结果如下：

![图片](img/B18406_12_05.png)

图 12.5：年龄和先前的计数再犯概率等高线图

我们现在可以构建一个包含两个特征的交互项。即使等高线图将特征离散化以观察更平滑的进展，我们也不需要将这种关系离散化。有意义的是将其作为每年`priors_count`的比率。但是从哪一年开始算起？当然是被告成年以来的年份。但是要获得这些年份，我们不能使用`age - 18`，因为这会导致除以零，所以我们将使用`17`代替。当然，有许多方法可以做到这一点。最好的方法是我们假设年龄有小数，通过减去 18，我们可以计算出非常精确的`priors_per_year`比率。然而，不幸的是，我们并没有这样的数据。你可以在下面的代码片段中看到代码：

```py
recidivism_df['priors_per_year'] =\
            recidivism_df['priors_count']/(recidivism_df['age'] - 17) 
```

黑盒模型通常会自动找到交互项。例如，神经网络中的隐藏层具有所有一阶交互项，但由于非线性激活，它并不仅限于线性组合。然而，“手动”定义交互项甚至非线性转换，一旦模型拟合完成，我们可以更好地解释这些交互项。此外，我们还可以对它们使用单调约束，这正是我们稍后将在`priors_per_year`上所做的。现在，让我们检查其单调性是否通过`plot_prob_progression`保持。查看以下代码片段：

```py
mldatasets.**plot_prob_progression**(
    recidivism_df.**priors_per_year**,
    recidivism_df.**is_recid**,
    x_intervals=8,
    xlabel='Priors Per Year',
    title='Probability of Recidivism by Priors per Year (\
    according to data)'
) 
```

前面的代码片段输出以下截图，显示了新特征的几乎单调进展：

![图表，折线图  自动生成的描述](img/B18406_12_06.png)

图 12.6：`priors_per_year`的先验概率进展

`priors_per_year`不是更单调的原因是 3.0 以上的`priors_per_year`区间非常稀疏。因此，对这些少数被告强制执行该特征的单调性将非常不公平，因为他们呈现了 75%的风险下降。解决这一问题的方法之一是将它们左移，将这些观察结果中的`priors_per_year`设置为`-1`，如下面的代码片段所示：

```py
recidivism_df.loc[recidivism_df.priors_per_year > 3,\
                  'priors_per_year'] = -1 
```

当然，这种移动会略微改变特征的解释，考虑到`-1`的少数值实际上意味着超过`3`。现在，让我们生成另一个等高线图，但这次是在`age_group`和`priors_per_year`之间。后者将按分位数（`y_intervals=6, use_quantiles=True`）进行离散化，以便更容易观察到再犯概率。以下代码片段显示了代码：

```py
mldatasets.**plot_prob_contour_map**(
    recidivism_df.**age_group**,
    recidivism_df.**priors_per_year**,
    recidivism_df.**is_recid**,
    y_intervals=6,
    use_quantiles=True,
    xlabel='Age Group',
    title='Probability of Recidivism by Age/Priors per Year \
    Discretized in Quantiles', ylabel='Priors Per Year'
) 
 generates the contours in *Figure 12.7*. It shows that, for the most part, the plot moves in one direction. We were hoping to achieve this outcome because it allows us, through one interaction feature, to control the monotonicity of what used to involve two features.
```

输出结果如下：

![图片，B18406_12_07.png]

图 12.7：`age_group`和`priors_per_year`的再犯概率等高线图

几乎一切准备就绪，但`age_group`仍然是分类的，所以我们必须将其编码成数值形式。

## 分类别编码

对于`age_group`的最佳分类编码方法是**序数编码**，也称为**标签编码**，因为它会保留其顺序。我们还应该对数据集中的其他两个分类特征进行编码，即`sex`和`race`。对于`sex`，序数编码将其转换为二进制形式——相当于**虚拟编码**。另一方面，`race`是一个更具挑战性的问题，因为它有三个类别，使用序数编码可能会导致偏差。然而，是否使用**独热编码**取决于你使用的模型类别。基于树的模型对序数特征没有偏差问题，但其他基于特征权重的模型，如神经网络和逻辑回归，可能会因为这种顺序而产生偏差。

考虑到数据集已经在`种族`上进行了平衡，因此这种情况发生的风险较低，我们稍后无论如何都会移除这个特征，所以我们将继续对其进行序数编码。

为了对三个特征进行序数编码，我们将使用 scikit-learn 的`OrdinalEncoder`。我们可以使用它的`fit_transform`函数一次性拟合和转换特征。然后，我们还可以趁机删除不必要的特征。请看下面的代码片段：

```py
cat_feat_l = ['sex', 'race', 'age_group']
ordenc = preprocessing.**OrdinalEncoder**(dtype=np.int8)
recidivism_df[cat_feat_l] =\
                  ordenc.**fit_transform**(recidivism_df[cat_feat_l])
recidivism_df.drop(['age', 'priors_count', 'compas_score'],\
                    axis=1, inplace=True) 
```

现在，我们还没有完全完成。我们仍然需要初始化我们的随机种子并划分我们的数据为训练集和测试集。

## 其他准备工作

下一步的准备工作很简单。为了确保可重复性，让我们在需要的地方设置随机种子，然后将我们的`y`设置为`is_recid`，将`X`设置为其他所有特征。我们对这两个进行`train_test_split`。最后，我们使用`X`后跟`y`重建`recidivism_df` DataFrame。这样做只有一个原因，那就是`is_recid`是最后一列，这将有助于下一步。代码可以在这里看到：

```py
rand = 9
os.environ['PYTHONHASHSEED'] = str(rand)
tf.random.set_seed(rand)
np.random.seed(rand)
y = recidivism_df['is_recid']
X = recidivism_df.drop(['is_recid'], axis=1).copy()
X_train, X_test, y_train, y_test = model_selection.**train_test_split**(
    X, y, test_size=0.2, random_state=rand
)
recidivism_df = X.join(y) 
```

现在，我们将验证 Spearman 的相关性是否在需要的地方有所提高，在其他地方保持不变。请看下面的代码片段：

```py
pd.DataFrame(
    {
        'feature': X.columns,
        'correlation_to_target':scipy.stats.**spearmanr**(recidivism_df).\
        correlation[10,:-1]
    }
).style.background_gradient(cmap='coolwarm') 
```

前面的代码输出了*图 12.8*中所示的 DataFrame。请将其与*图 12.2*进行比较。请注意，在分位数离散化后，`age`与目标变量的单调相关性略有降低。一旦进行序数编码，`c_charge_degree`的相关性也大大提高，而`priors_per_year`相对于`priors_count`也有所改善。其他特征不应受到影响，包括那些系数最低的特征。

输出如下：

![表格描述自动生成](img/B18406_12_08.png)图 12.8：所有特征与目标变量的 Spearman 相关系数（特征工程后）

系数最低的特征在模型中可能也是不必要的，但我们将让模型通过正则化来决定它们是否有用。这就是我们接下来要做的。

# 调整模型以提高可解释性

传统上，正则化是通过在系数或权重上施加惩罚项（如**L1**、**L2**或**弹性网络**）来实现的，这会减少最不相关特征的影响。如第十章“可解释性特征选择和工程”部分的*嵌入式方法*中所示，这种正则化形式在特征选择的同时也减少了过拟合。这使我们来到了正则化的另一个更广泛的概念，它不需要惩罚项。通常，这相当于施加限制或停止标准，迫使模型限制其复杂性。

除了正则化，无论是其狭义（基于惩罚）还是广义（过拟合方法），还有其他方法可以调整模型以提高可解释性——也就是说，通过调整训练过程来提高模型的公平性、责任性和透明度。例如，我们在第十章*特征选择和可解释性工程*中讨论的类别不平衡超参数，以及第十一章*偏差缓解和因果推断方法*中的对抗性偏差，都有助于提高公平性。此外，我们将在本章进一步研究的约束条件对公平性、责任性和透明度也有潜在的好处。

有许多不同的调整可能性和模型类别。如本章开头所述，我们将关注与可解释性相关的选项，但也将模型类别限制在流行的深度学习库（Keras）、一些流行的树集成（XGBoost、随机森林等）、**支持向量机**（**SVMs**）和逻辑回归。除了最后一个，这些都被认为是黑盒模型。

## 调整 Keras 神经网络

对于 Keras 模型，我们将通过超参数调整和**分层 K 折交叉验证**来选择最佳正则化参数。我们将按照以下步骤进行：

1.  首先，我们需要定义模型和要调整的参数。

1.  然后，我们进行调整。

1.  接下来，我们检查其结果。

1.  最后，我们提取最佳模型并评估其预测性能。

让我们详细看看这些步骤。

### 定义模型和要调整的参数

我们首先应该创建一个函数（`build_nn_mdl`）来构建和编译一个可正则化的 Keras 模型。该函数接受一些参数，以帮助调整模型。它接受一个包含隐藏层中神经元数量的元组（`hidden_layer_sizes`），以及应用于层核的 L1（`l1_reg`）和 L2（`l1_reg`）正则化值。最后，它还接受`dropout`参数，与 L1 和 L2 惩罚不同，它是一种**随机正则化方法**，因为它采用随机选择。请看以下代码片段：

```py
def **build_nn_mdl**(hidden_layer_sizes, l1_reg=0, l2_reg=0, dropout=0):
    nn_model = tf.keras.Sequential([
        tf.keras.Input(shape=[len(X_train.keys())]),\
        tf.keras.layers.experimental.preprocessing.**Normalization**()
    ])
    reg_args = {}
    if (l1_reg > 0) or (l2_reg > 0):
        reg_args = {'kernel_regularizer':\
                    tf.keras.regularizers.**l1_l2**(l1=l1_reg, l2=l2_reg)}
    for hidden_layer_size in hidden_layer_sizes:
        nn_model.add(tf.keras.layers.**Dense**(hidden_layer_size,\
                        activation='relu', ****reg_args**))
    if dropout > 0:
        nn_model.add(tf.keras.layers.**Dropout**(dropout))
    nn_model.add(tf.keras.layers.**Dense**(1, activation='sigmoid'))
    nn_model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.0004),
        metrics=['accuracy',tf.keras.metrics.AUC(name='auc')]
)
    return nn_model 
```

之前的功能将模型（`nn_model`）初始化为一个 `Sequential` 模型，其输入层与训练数据中的特征数量相对应，并添加一个 `Normalization()` 层来标准化输入。然后，如果任一惩罚项超过零，它将设置一个字典（`reg_args`），将 `kernel_regularizer` 分配给 `tf.keras.regularizers.l1_l2` 并用这些惩罚项初始化。一旦添加了相应的 `hidden_layer_size` 的隐藏（`Dense`）层，它将 `reg_args` 字典作为额外参数传递给每个层。在添加所有隐藏层之后，它可以选择添加 `Dropout` 层和具有 `sigmoid` 激活的最终 `Dense` 层。然后，模型使用 `binary_crossentropy` 和具有较慢学习率的 `Adam` 优化器编译，并设置为监控 `accuracy` 和 `auc` 指标。

### 运行超参数调整

现在我们已经定义了模型和要调整的参数，我们初始化了 `RepeatedStratifiedKFold` 交叉验证器，它将训练数据分成五份，总共重复三次（`n_repeats`），每次重复使用不同的随机化。然后我们为网格搜索超参数调整创建一个网格（`nn_grid`）。它只测试三个参数（`l1_reg`、`l2_reg` 和 `dropout`）的两个可能选项，这将产生 ![](img/B18406_12_002.png) 种组合。我们将使用 scikit-learn 包装器（`KerasClassifier`）来使我们的模型与 scikit-learn 网格搜索兼容。说到这一点，我们接下来初始化 `GridSearchCV`，它使用 Keras 模型（`estimator`）执行交叉验证网格搜索（`param_grid`）。我们希望它根据精度（`scoring`）选择最佳参数，并且在过程中不抛出错误（`error_score=0`）。最后，我们像使用任何 Keras 模型一样拟合 `GridSearchCV`，传递 `X_train`、`y_train`、`epochs` 和 `batch_size`。代码可以在以下代码片段中看到：

```py
cv = model_selection.**RepeatedStratifiedKFold**(
    n_splits=5,
    n_repeats=3,
    random_state=rand
)
nn_grid = {
    'hidden_layer_sizes':[(80,)],
    'l1_reg':[0,0.005],
    'l2_reg':[0,0.01],
    'dropout':[0,0.05]
}
nn_model = KerasClassifier(build_fn=build_nn_mdl)
nn_grid_search = model_selection.**GridSearchCV**(
    estimator=nn_model,
    cv=cv,
    n_jobs=-1,
    param_grid=nn_grid,
    scoring='precision',
    error_score=0
)
nn_grid_result = nn_grid_search.**fit**(
    X_train.astype(float),
    y_train.astype(float),
    epochs=400,batch_size=128
) 
```

接下来，我们可以检查网格搜索的结果。

### 检查结果

一旦完成网格搜索，你可以使用以下命令输出最佳参数：`print(nn_grid_result.best_params_)`。或者，你可以将所有结果放入一个 DataFrame 中，按最高精度（`sort_values`）排序，并按以下方式输出：

```py
pd.**DataFrame**(nn_grid_result.**cv_results**_)[
    [
        'param_hidden_layer_sizes',
        'param_l1_reg',
        'param_l2_reg',
        'param_dropout',
        'mean_test_score',
        'std_test_score',
        'rank_test_score'
    ]
].**sort_values**(by='rank_test_score') 
Figure 12.9. The unregularized model is dead last, showing that all regularized model combinations performed better. One thing to note is that given the 1.5–2% standard deviations (std_test_score) and that the top performer is only 2.2% from the lowest performer, in this case, the benefits are marginal from a precision standpoint, but you should use a regularized model nonetheless because of other benefits.
```

输出如下所示：

![表格描述自动生成](img/B18406_12_09.png)图 12.9：神经网络模型交叉验证网格搜索的结果

### 评估最佳模型

网格搜索产生的另一个重要元素是表现最佳模型（`nn_grid_result.best_estimator_`）。我们可以创建一个字典来存储我们将在本章中拟合的所有模型（`fitted_class_mdls`），然后使用 `evaluate_class_mdl` 评估这个正则化的 Keras 模型，并将评估结果同时保存在字典中。请查看以下代码片段：

```py
fitted_class_mdls = {}
fitted_class_mdls['keras_reg'] = mldatasets.**evaluate_class_mdl**(
    nn_grid_result.best_estimator_,
    X_train.astype(float),
    X_test.astype(float),
    y_train.astype(float),
    y_test.astype(float),
    plot_roc=False,
    plot_conf_matrix=True,
    **ret_eval_dict=****True**
) 
Figure 12.10. The accuracy is a little bit better than the original COMPAS model from *Chapter 6*, *Anchors and Counterfactual Explanations*, but the strategy to optimize for higher precision while regularizing yielded a model with nearly half as many false positives but 50% more false negatives.
```

输出如下所示：

![图表，树状图  自动生成的描述](img/B18406_12_10.png)图 12.10：正则化 Keras 模型的评估

通过使用自定义损失函数或类权重，可以进一步校准类平衡，正如我们稍后将要做的。接下来，我们将介绍如何调整其他模型类。

## 调整其他流行模型类

在本节中，我们将拟合许多不同的模型，包括未正则化和正则化的模型。为此，我们将从广泛的参数中选择，这些参数执行惩罚正则化，通过其他方式控制过拟合，并考虑类别不平衡。

### 相关模型参数的简要介绍

供您参考，有两个表格包含用于调整许多流行模型的参数。这些已经被分为两部分。Part A（图 12.11）包含五个具有惩罚正则化的 scikit-learn 模型。Part B（图 12.12）显示了所有树集成，包括 scikit-learn 的随机森林模型和来自最受欢迎的增强树库（XGBoost、LightGBM 和 CatBoost）的模型。

Part A 可以在这里查看：

![表格，日历  自动生成的描述](img/B18406_12_11.png)图 12.11：惩罚正则化 scikit-learn 模型的调整参数

在图 12.11 中，您可以在列中观察到模型，在行中观察到相应的参数名称及其默认值在右侧。在参数名称和默认值之间，有一个加号或减号，表示是否改变默认值的一个方向或另一个方向应该使模型更加保守。这些参数还按以下类别分组：

+   **算法**：一些训练算法不太容易过拟合，但这通常取决于数据。

+   **正则化**：仅在更严格的意义上。换句话说，控制基于惩罚的正则化的参数。

+   **迭代**：这控制执行多少个训练轮次、迭代或 epoch。调整这个方向或另一个方向可能会影响过拟合。在基于树的模型中，估计器或树的数量是类似的。

+   **学习率**：这控制学习发生的速度。它与迭代一起工作。学习率越低，需要的迭代次数越多以优化目标函数。

+   **提前停止**：这些参数控制何时停止训练。这允许您防止您的模型对训练数据过拟合。

+   **类别不平衡**：对于大多数模型，这在损失函数中惩罚了较小类别的误分类，对于基于树的模型，特别是这样，它被用来重新加权分割标准。无论如何，它只与分类器一起工作。

+   **样本权重**：我们在第十一章“偏差缓解和因果推断方法”中利用了这一点，根据样本分配权重以减轻偏差。

标题中既有分类模型也有回归模型，并且它们共享相同的参数。请注意，scikit-learn 的`LinearRegression`在`LogisticRegression`下没有特色，因为它没有内置的正则化。无论如何，我们将在本节中仅使用分类模型。

B 部分可以在这里看到：

![表格，日历  自动生成的描述](img/B18406_12_12.png)

![表格，日历  自动生成的描述](img/B18406_12_12.1.png)

图 12.12：树集成模型的调整参数

*图 12.12*与*图 12.11*非常相似，除了它有更多仅在树集成中可用的参数类别，如下所示：

+   **特征采样**：这种方法通过在节点分裂、节点或树训练中考虑较少的特征来实现。因为它随机选择特征，所以它是一种随机正则化方法。

+   **树的大小**：这通过最大深度、最大叶子数或其他限制其增长的参数来约束树，从而反过来抑制过拟合。

+   **分裂**：任何控制树中节点如何分裂的参数都可以间接影响过拟合。

+   **袋装**：也称为**自助聚合**，它首先通过自助采样开始，这涉及到从训练数据中随机抽取样本来拟合弱学习器。这种方法减少了方差，有助于减少过拟合，并且相应地，采样参数通常在超参数调整中很突出。

+   **约束**：我们将在下一节中进一步详细解释这些内容，但这是如何将特征约束以减少或增加对输出的影响。它可以在数据非常稀疏的领域减少过拟合。然而，减少过拟合通常不是主要目标，而交互约束可以限制哪些特征可以交互。

请注意，*图 12.12*中带有星号（`*`）的参数表示在`fit`函数中设置的，而不是用模型初始化的。此外，除了 scikit-learn 的`RandomForest`模型外，所有其他参数通常有许多别名。对于这些，我们使用 scikit-learn 的包装函数，但所有参数也存在于原生版本中。我们不可能在这里解释每个模型参数，但建议您直接查阅文档以深入了解每个参数的作用。本节的目的在于作为指南或参考。

接下来，我们将采取与我们对 Keras 模型所做类似的步骤，但一次针对许多不同的模型，最后我们将评估最适合公平性的最佳模型。

### 批量超参数调整模型

好的——既然我们已经快速了解了我们可以拉动的哪些杠杆来调整模型，那么让我们定义一个包含所有模型的字典，就像我们在其他章节中所做的那样。这次，我们包括了一个用于网格搜索的参数值的`grid`。看看下面的代码片段：

```py
class_mdls = {
    'logistic':{
        'model':linear_model.**LogisticRegression**(random_state=rand,\
                                                max_iter=1000),
        'grid':{
            'C':np.linspace(0.01, 0.49, 25),
            'class_weight':[{0:6,1:5}],
            'solver':['lbfgs', 'liblinear', 'newton-cg']
        }
     },
    'svc':{
        'model':svm.**SVC**(probability=True, random_state=rand),
        'grid':{'C':[15,25,40], 'class_weight':[{0:6,1:5}]}
    },
    'nu-svc':{
        'model':svm.**NuSVC**(
            probability=True,
            random_state=rand
        ),
        'grid':{
            'nu':[0.2,0.3], 'gamma':[0.6,0.7],\
            'class_weight':[{0:6,1:5}]}
        },
    'mlp':{
        'model':neural_network.**MLPClassifier**(
            random_state=rand,
            hidden_layer_sizes=(80,),
            early_stopping=True
        ),
        'grid':{
            'alpha':np.linspace(0.05, 0.15, 11),
            'activation':['relu','tanh','logistic']}
        },
        'rf':{
            'model':ensemble.**RandomForestClassifier**(
                random_state=rand, max_depth=7, oob_score=True, \
                bootstrap=True
             ),
            'grid':{
                'max_features':[6,7,8],
                'max_samples':[0.75,0.9,1],
                'class_weight':[{0:6,1:5}]}
            },
    'xgb-rf':{
        'model':xgb.**XGBRFClassifier**(
            seed=rand, eta=1, max_depth=7, n_estimators=200
        ),
        'grid':{
            'scale_pos_weight':[0.85],
            'reg_lambda':[1,1.5,2],
            'reg_alpha':[0,0.5,0.75,1]}
        },
    'xgb':{
        'model':xgb.**XGBClassifier**(
            seed=rand, eta=1, max_depth=7
        ),
        'grid':{
            'scale_pos_weight':[0.7],
            'reg_lambda':[1,1.5,2],
            'reg_alpha':[0.5,0.75,1]}
        },
    'lgbm':{
        'model':lgb.**LGBMClassifier**(
            random_seed=rand,
            learning_rate=0.7,
            max_depth=5
        ),
        'grid':{
            'lambda_l2':[0,0.5,1],
            'lambda_l1':[0,0.5,1],
            'scale_pos_weight':[0.8]}
        },
    'catboost':{
        'model':cb.**CatBoostClassifier**(
            random_seed=rand,
            depth=5,
            learning_rate=0.5,
            verbose=0
        ),
        'grid':{
            'l2_leaf_reg':[2,2.5,3],
            'scale_pos_weight':[0.65]}
        }
} 
```

下一步是为字典中的每个模型添加一个`for`循环，然后`deepcopy`它并使用`fit`来生成一个“基础”的非正则化模型。接下来，我们使用`evaluate_class_mdl`对其进行评估，并将其保存到我们之前为 Keras 模型创建的`fitted_class_mdls`字典中。现在，我们需要生成模型的正则化版本。因此，我们再次进行`deepcopy`，并遵循与 Keras 相同的步骤进行`RepeatedStratifiedKFold`交叉验证网格搜索，并且我们也以相同的方式进行评估，将结果保存到拟合模型字典中。代码如下所示：

```py
for mdl_name in class_mdls:
    base_mdl = copy.deepcopy(class_mdls[mdl_name]['model'])
    base_mdl = base_mdl.**fit**(X_train, y_train)
    fitted_class_mdls[mdl_name+'_base'] = \
        mldatasets.**evaluate_class_mdl**(
            base_mdl, X_train, X_test,y_train, y_test,
            plot_roc=False, plot_conf_matrix=False,
            show_summary=False, ret_eval_dict=True
    )
    reg_mdl = copy.deepcopy(class_mdls[mdl_name]['model'])
    grid = class_mdls[mdl_name]['grid']
    cv = model_selection.**RepeatedStratifiedKFold**(
        n_splits=5, n_repeats=3, random_state=rand
    )
    grid_search = model_selection.**GridSearchCV**(
    estimator=reg_mdl, cv=cv, param_grid=grid,
    scoring='precision', n_jobs=-1, error_score=0, verbose=0
    )
    grid_result = grid_search.**fit**(X_train, y_train)
    fitted_class_mdls[mdl_name+'_reg'] =\
        mldatasets.**evaluate_class_mdl**(
            grid_result.**best_estimator**_, X_train, X_test, y_train,
            y_test, plot_roc=False,
            plot_conf_matrix=False, show_summary=False,
            ret_eval_dict=True
    )
    fitted_class_mdls[mdl_name+'_reg']['cv_best_params'] =\
        grid_result.**best_params**_ 
```

一旦代码执行完毕，我们可以根据精确度对模型进行排名。

### 根据精确度评估模型

我们可以提取拟合模型字典的指标，并将它们放入一个 DataFrame 中，使用`from_dict`。然后我们可以根据最高的测试精确度对模型进行排序，并为最重要的两个列着色编码，这两个列是`precision_test`和`recall_test`。代码可以在下面的代码片段中看到：

```py
class_metrics = pd.DataFrame.from_dict(fitted_class_mdls, 'index')[
    [
        'accuracy_train',
        'accuracy_test',
        'precision_train',
        'precision_test',
        'recall_train',
        'recall_test',
        'roc-auc_test',
        'f1_test',
        'mcc_test'
    ]
]
with pd.option_context('display.precision', 3):
    html = class_metrics.sort_values(
        by='precision_test', ascending=False
    ).style.background_gradient(
        cmap='plasma',subset=['precision_test']
    ).background_gradient(
        cmap='viridis', subset=['recall_test'])
html 
```

前面的代码将输出*图 12.13*所示的 DataFrame。你可以看出，正则化树集成模型在排名中占据主导地位，其次是它们的非正则化版本。唯一的例外是正则化 Nu-SVC，它排名第一，而它的非正则化版本排名最后！

输出如下所示：

![表格描述自动生成](img/B18406_12_13.png)

图 12.13：根据交叉验证网格搜索的顶级模型

你会发现，Keras 正则化神经网络模型的精确度低于正则化逻辑回归，但召回率更高。确实，我们希望优化高精确度，因为它会影响假阳性，这是我们希望最小化的，但精确度可以达到 100%，而召回率可以是 0%，如果那样的话，你的模型就不好了。同时，还有公平性，这关乎于保持低假阳性率，并且在种族间均匀分布。因此，这是一个权衡的问题，追求一个指标并不能让我们达到目标。

### 评估最高性能模型的公平性

为了确定如何进行下一步，我们必须首先评估我们的最高性能模型在公平性方面的表现。我们可以使用`compare_confusion_matrices`来完成这项工作。正如你使用 scikit-learn 的`confusion_matrix`一样，第一个参数是真实值或目标值（通常称为`y_true`），第二个是模型的预测值（通常称为`y_pred`）。这里的区别是它需要两组`y_true`和`y_pred`，一组对应于观察的一个部分，另一组对应于另一个部分。在这四个参数之后，你给每个部分起一个名字，所以这就是以下两个参数告诉你的内容。最后，`compare_fpr=True`确保它将比较两个混淆矩阵之间的**假阳性率**（**FPR**）。看看下面的代码片段：

```py
y_test_pred = fitted_class_mdls['catboost_reg']['preds_test']
_ = mldatasets.**compare_confusion_matrices**(
    y_test[X_test.race==1],
    y_test_pred[X_test.race==1],
    y_test[X_test.race==0],
    y_test_pred[X_test.race==0],
    'Caucasian',
    'African-American',
    **compare_fpr=****True**
)
y_test_pred =  fitted_class_mdls['catboost_base']['preds_test']
_ = mldatasets.**compare_confusion_matrices**(
    y_test[X_test.race==1],
    y_test_pred[X_test.race==1],
    y_test[X_test.race==0],
    y_test_pred[X_test.race==0],
    'Caucasian',
    'African-American',
    **compare_fpr=****True**
) 
Figure 12.14 and *Figure 12.15*, corresponding to the regularized and base models, respectively. You can see *Figure 12.14* here:
```

![图表，树状图图表，描述自动生成](img/B18406_12_14.png)

图 12.14：正则化 CatBoost 模型之间的混淆矩阵

*图 12.15*告诉我们，正则化模型的 FPR 显著低于基础模型。您可以看到输出如下：

![图表，瀑布图，树状图图表，描述自动生成](img/B18406_12_15.png)

图 12.15：基础 CatBoost 模型之间的混淆矩阵

然而，如图 12.15 所示的基础模型与正则化模型的 FPR 比率为 1.11，而正则化模型的 FPR 比率为 1.47，尽管整体指标相似，但差异显著。但在尝试同时实现几个目标时，很难评估和比较模型，这就是我们将在下一节中要做的。

## 使用贝叶斯超参数调整和自定义指标优化公平性

我们的使命是生产一个具有高精确度和良好召回率，同时在不同种族间保持公平性的模型。因此，实现这一使命将需要设计一个自定义指标。

### 设计一个自定义指标

我们可以使用 F1 分数，但它对精确度和召回率的处理是平等的，因此我们不得不创建一个加权指标。我们还可以考虑每个种族的精确度和召回率的分布情况。实现这一目标的一种方法是通过使用标准差，它量化了这种分布的变化。为此，我们将用精确度的一半作为组间标准差来惩罚精确度，我们可以称之为惩罚后的精确度。公式如下：

![图片，B18406_12_003.png]

我们可以对召回率做同样的处理，如图所示：

![图片，B18406_12_004.png]

然后，我们为惩罚后的精确度和召回率做一个加权平均值，其中精确度是召回率的两倍，如图所示：

![图片，B18406_12_005.png]

为了计算这个新指标，我们需要创建一个可以调用`weighted_penalized_pr_average`的函数。它接受`y_true`和`y_pred`作为预测性能指标。然而，它还包括`X_group`，它是一个包含组值的`pandas`序列或数组，以及`group_vals`，它是一个列表，它将根据这些值对预测进行子集划分。在这种情况下，组是`race`，可以是 0 到 2 的值。该函数包括一个`for`循环，遍历这些可能的值，通过每个组对预测进行子集划分。这样，它可以计算每个组的精确度和召回率。之后，函数的其余部分只是简单地执行之前概述的三个数学运算。代码可以在以下片段中看到：

```py
def **weighted_penalized_pr_average**(y_true, y_pred, X_group,\
                    group_vals, penalty_mult=0.5,\
                    precision_mult=2,\
                    recall_mult=1):
    precision_all = metrics.**precision_score**(
        y_true, y_pred, zero_division=0
    )
    recall_all = metrics.**recall_score**(
        y_true, y_pred, zero_division=0
    )
    p_by_group = []
    r_by_group = []
    for group_val in group_vals:
        in_group = X_group==group_val
        p_by_group.append(metrics.**precision_score**(
            y_true[in_group], y_pred[in_group], zero_division=0
            )
        )
        r_by_group.append(metrics.**recall_score**(
            y_true[in_group], y_pred[in_group], zero_division=0
            )
        )
    precision_all = precision_all - \
                   (np.array(p_by_group).std()*penalty_mult)
    recall_all = recall_all -\
                (np.array(r_by_group).std()*penalty_mult)
    return ((precision_all*precision_mult)+
            (recall_all*recall_mult))/\
            (precision_mult+recall_mult) 
```

现在，为了使这个函数发挥作用，我们需要运行调整。

### 运行贝叶斯超参数调整

**贝叶斯优化**是一种 *全局优化方法*，它使用黑盒目标函数的后验分布及其连续参数。换句话说，它根据过去的结果顺序搜索下一个要测试的最佳参数。与网格搜索不同，它不会在网格上尝试固定参数组合，而是利用它已经知道的信息并探索未知领域。

`bayesian-optimization` 库是模型无关的。它所需的所有东西是一个函数以及它们的界限参数。它将在这些界限内探索这些参数的值。该函数接受这些参数并返回一个数字。这个数字，或目标，是贝叶斯优化算法将最大化的。

以下代码是用于 `objective` 函数的，它使用四个分割和三个重复初始化一个 `RepeatedStratifiedKFold` 交叉验证。然后，它遍历分割并使用它们拟合 `CatBoostClassifier`。最后，它计算每个模型训练的 `weighted_penalized_pr_average` 自定义指标并将其追加到一个列表中。最后，该函数返回所有 12 个训练样本的自定义指标的中位数。代码在以下片段中显示：

```py
def **hyp_catboost**(l2_leaf_reg, scale_pos_weight):
    cv = model_selection.**RepeatedStratifiedKFold**(
        n_splits=4,n_repeats=3, random_state=rand
    )
    metric_l = []
    for train_index, val_index in cv.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index],\
                               X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index],
                               y_train.iloc[val_index]
        mdl = cb.**CatBoostClassifier**(
            random_seed=rand, learning_rate=0.5, verbose=0, depth=5,\
            l2_leaf_reg=l2_leaf_reg, scale_pos_weight=scale_pos_weight
        )
        mdl = mdl.**fit**(X_train_cv, y_train_cv)
        y_val_pred = mdl.**predict**(X_val_cv)
        metric = **weighted_penalized_pr_average**(
            y_val_cv,y_val_pred, X_val_cv['race'], range(3)
        )
        metric_l.**append**(metric)
    return np.**median**(np.array(metric_l)) 
```

现在函数已经定义，运行贝叶斯优化过程很简单。首先，设置参数界限字典（`pbounds`），使用 `hyp_catboost` 函数初始化 `BayesianOptimization`，然后使用 `maximize` 运行它。`maximize` 函数接受 `init_points`，它设置初始使用随机探索运行的迭代次数。然后，`n_iter` 是它应该执行的优化迭代次数以找到最大值。我们将 `init_points` 和 `n_iter` 分别设置为 `3` 和 `7`，因为可能需要很长时间，但这些数字越大越好。代码可以在以下片段中看到：

```py
pbounds = {
    'l2_leaf_reg': (2,4),
    'scale_pos_weight': (0.55,0.85)
    }
optimizer = **BayesianOptimization**(
    **hyp_catboost**,
    pbounds, 
    random_state=rand
)
optimizer.maximize(init_points=3, n_iter=7) 
```

一旦完成，你可以访问最佳参数，如下所示：

```py
print(optimizer.max['params']) 
```

它将返回一个包含参数的字典，如下所示：

```py
{'l2_leaf_reg': 2.0207483077713997, 'scale_pos_weight': 0.7005623776446217} 
```

现在，让我们使用这些参数拟合一个模型并评估它。

### 使用最佳参数拟合和评估模型

使用这些参数初始化 `CatBoostClassifier` 与将 `best_params` 字典作为参数传递一样简单。然后，你所需要做的就是 `fit` 模型并评估它（`evaluate_class_mdl`）。代码在以下片段中显示：

```py
cb_opt = cb.**CatBoostClassifier**(
    random_seed=rand,
    depth=5,
    learning_rate=0.5,
    verbose=0,
    **optimizer.max['params']
)
cb_opt = cb_opt.**fit**(X_train, y_train)
fitted_class_mdls['catboost_opt'] = mldatasets.**evaluate_class_mdl**(
    cb_opt,
    X_train,
    X_test,
    y_train,
    y_test,
    plot_roc=False,
    plot_conf_matrix=True,
    **ret_eval_dict=****True**
) 
```

前面的代码片段输出了以下预测性能指标：

```py
Accuracy_train:  0.9652		Accuracy_test:   0.8192
Precision_test:  0.8330		Recall_test:     0.8058
ROC-AUC_test:    0.8791		F1_test:         0.8192 
```

这些是我们迄今为止达到的最高 `Accuracy_test`、`Precision_test` 和 `Recall_test` 指标。现在让我们看看模型使用 `compare_confusion_matrices` 进行公平性测试的表现。请看以下代码片段：

```py
y_test_pred = fitted_class_mdls['catboost_opt']['preds_test']
_ = mldatasets.**compare_confusion_matrices**(
    y_test[X_test.race==1],
    y_test_pred[X_test.race==1],
    y_test[X_test.race==0],
    y_test_pred[X_test.race==0],
    'Caucasian',
    'African-American',
    **compare_fpr=****True**
) 
```

前面的代码输出了 *图 12.16*，它显示了迄今为止我们获得的一些最佳公平性指标，如你所见：

![图表 描述自动生成](img/B18406_12_16.png)

图 12.16：优化后的 CatBoost 模型不同种族之间的混淆矩阵比较

这些结果很好，但我们不能完全确信模型没有种族偏见，因为特征仍然存在。衡量其影响的一种方法是通过特征重要性方法。

### 通过特征重要性来检查种族偏见

尽管 CatBoost 在大多数指标上，包括准确率、精确率和 F1 分数，都是我们表现最好的模型，但我们正在使用 XGBoost 前进，因为 CatBoost 不支持交互约束，我们将在下一节中实现。但首先，我们将比较它们在发现哪些特征重要方面的差异。此外，**SHapley Additive exPlanations**（**SHAP**）值提供了一种稳健的方法来衡量和可视化特征重要性，因此让我们为我们的优化 CatBoost 和正则化 XGBoost 模型计算它们。为此，我们需要用每个模型初始化`TreeExplainer`，然后使用`shap_values`为每个模型生成值，如下面的代码片段所示：

```py
fitted_cb_mdl = fitted_class_mdls['catboost_opt']['fitted']
shap_cb_explainer = shap.**TreeExplainer**(fitted_cb_mdl)
shap_cb_values = shap_cb_explainer.**shap_values**(X_test)
fitted_xgb_mdl = fitted_class_mdls['xgb_reg']['fitted']
shap_xgb_explainer = shap.**TreeExplainer**(fitted_xgb_mdl)
shap_xgb_values = shap_xgb_explainer.**shap_values**(X_test) 
```

接下来，我们可以使用 Matplotlib 的`subplot`功能并排生成两个`summary_plot`图，如下所示：

```py
ax0 = plt.subplot(1, 2, 1)
shap.**summary_plot**(
    **shap_xgb_values**,
    X_test,
    plot_type="dot",
    plot_size=None,
    show=False
)
ax0.set_title("XGBoost SHAP Summary")
ax1 = plt.subplot(1, 2, 2)
shap.**summary_plot**(
    **shap_cb_values**,
    X_test,
    plot_type="dot",
    plot_size=None,
    show=False
)
ax1.set_title("Catboost SHAP Summary") 
Figure 12.17, which shows how similar CatBoost and XGBoost are. This similarity shouldn’t be surprising because, after all, they are both gradient-boosted decision trees. The bad news is that race is in the top four for both. However, the prevalence of the shade that corresponds to lower feature values on the right suggests that African American (race=0) negatively correlates with recidivism.
```

输出结果如下：

![](img/B18406_12_17.png)

图 12.17：XGBoost 正则化和 CatBoost 优化模型的 SHAP 总结图

在任何情况下，从训练数据中移除`race`是有意义的，但我们必须首先确定模型为什么认为这是一个关键特征。请看以下代码片段：

```py
shap_xgb_interact_values =\
                shap_xgb_explainer.shap_interaction_values(X_test) 
```

在*第四章*，*全局模型无关解释方法*中，我们讨论了评估交互效应。现在是时候回顾这个话题了，但这次，我们将提取 SHAP 的交互值（`shap_interaction_values`）而不是使用 SHAP 的依赖图。我们可以很容易地使用`summary_plot`图对 SHAP 交互进行排序。SHAP 总结图非常有信息量，但它并不像交互热图那样直观。为了生成带有标签的热图，我们必须将`shap_xgb_interact_values`的总和放在 DataFrame 的第一个轴上，然后使用特征的名称命名列和行（`index`）。其余的只是使用 Seaborn 的`heatmap`函数将 DataFrame 绘制为热图。代码可以在下面的代码片段中看到：

```py
shap_xgb_interact_avgs = np.abs(
    **shap_xgb_interact_values**
).mean(0)
np.fill_diagonal(shap_xgb_interact_avgs, 0)
shap_xgb_interact_df = pd.**DataFrame**(shap_xgb_interact_avgs)
shap_xgb_interact_df.columns = X_test.columns
shap_xgb_interact_df.index = X_test.columns
sns.**heatmap**(shap_xgb_interact_df, cmap='Blues', annot=True,\
            annot_kws={'size':13}, fmt='.2f', linewidths=.5) 
```

上述代码生成了*图 12.18*所示的热图。它展示了`race`与`length_of_stay`、`age_group`和`priors per year`之间的相互作用最为强烈。当然，一旦我们移除`race`，这些相互作用就会消失。然而，鉴于这一发现，如果这些特征中内置了种族偏见，我们应该仔细考虑。研究支持了`age_group`和`priors_per_year`的必要性，这使`length_of_stay`成为审查的候选者。我们不会在本章中这样做，但这确实值得思考：

![图形用户界面，应用程序描述自动生成](img/B18406_12_18.png)

图 12.18：正则化 XGBoost 模型的 SHAP 交互值热图

从*图 12.18*中得到的另一个有趣的见解是特征如何被聚类。你可以在`c_charge_degree`和`priors_per_year`之间的右下象限画一个框，因为一旦我们移除`race`，大部分的交互都将位于这里。限制令人烦恼的交互有很多好处。例如，为什么所有青少年犯罪特征，如`juv_fel_count`，都应该与`age_group`交互？为什么`sex`应该与`length_of_stay`交互？接下来，我们将学习如何围绕右下象限设置一个围栏，通过**交互约束**限制这些特征之间的交互。我们还将确保`priors_per_year`的**单调约束**。

# 实现模型约束

我们将首先讨论如何使用 XGBoost 以及所有流行的树集成实现约束，因为它们的参数名称相同（见*图 12.12*）。然后，我们将使用 TensorFlow Lattice 进行操作。但在我们继续之前，让我们按照以下方式从数据中移除`race`：

```py
X_train_con = X_train.**drop**(['race'], axis=1).copy()
X_test_con = X_test.**drop**(['race'], axis=1).copy() 
```

现在，随着`race`的消失，模型可能仍然存在一些偏见。然而，我们进行的特征工程和将要施加的约束可以帮助模型与这些偏见对齐，考虑到我们在*第六章*中发现的**锚点和反事实解释**的双重标准。话虽如此，生成的模型可能在对测试数据的性能上会较差。这里有两大原因，如下所述：

+   **信息丢失**：种族，尤其是与其他特征的交互，影响了结果，因此不幸地携带了一些信息。

+   **现实与政策驱动理想的错位**：当实施这些约束的主要原因是确保模型不仅符合领域知识，而且符合理想，而这些理想可能不在数据中明显体现时，这种情况就会发生。我们必须记住，一整套制度化的种族主义可能已经玷污了真实情况。模型反映了数据，但数据反映了地面的现实，而现实本身是有偏见的。

考虑到这一点，让我们开始实施约束！

## XGBoost 的约束

在本节中，我们将采取三个简单的步骤。首先，我们将定义我们的训练参数，然后训练和评估一个约束模型，最后检查约束的效果。

### 设置正则化和约束参数

我们使用 `print(fitted_class_mdls['xgb_reg']['cv_best_params'])` 来获取我们正则化 XGBoost 模型的最佳参数。它们位于 `best_xgb_params` 字典中，包括 `eta` 和 `max_depth`。然后，为了对 `priors_per_year` 应用单调约束，我们首先需要知道其位置和单调相关性的方向。从 *图 12.8* 中，我们知道这两个问题的答案。它是最后一个特征，相关性是正的，所以 `mono_con` 元组应该有九个项目，最后一个是一个 `1`，其余的是 `0`s。至于交互约束，我们只允许最后五个特征相互交互，前四个也是如此。`interact_con` 元组是一个列表的列表，反映了这些约束。代码可以在下面的片段中看到：

```py
**best_xgb_params** = {'eta': 0.3, 'max_depth': 28,\
                   'reg_alpha': 0.2071, 'reg_lambda': 0.6534,\
                   'scale_pos_weight': 0.9114}
**mono_con** = (0,0,0,0,0,0,0,0,1)
**interact_con** = [[4, 5, 6, 7, 8],[0, 1, 2, 3]] 
```

接下来，我们将使用这些约束条件训练和评估 XGBoost 模型。

### 训练和评估约束模型

现在，我们将使用这些约束条件训练和评估我们的约束模型。首先，我们使用我们的约束和正则化参数初始化 `XGBClassifier` 模型，然后使用缺少 `race` 特征的训练数据 (`X_train_con`) 来拟合它。然后，我们使用 `evaluate_class_mdl` 评估预测性能，并与 `compare_confusion_matrices` 比较公平性，就像我们之前所做的那样。代码可以在下面的片段中看到：

```py
xgb_con = xgb.XGBClassifier(
    seed=rand,monotone_constraints=**mono_con**,\
    interaction_constraints=**interact_con**, ****best_xgb_params**
)
xgb_con = xgb_con.**fit**(X_train_con, y_train)
fitted_class_mdls['xgb_con'] = mldatasets.**evaluate_class_mdl**(
    xgb_con, X_train_con, X_test_con, y_train, y_test,\
    plot_roc=False, ret_eval_dict=True
)
y_test_pred = fitted_class_mdls['xgb_con']['preds_test']
_ = mldatasets.**compare_confusion_matrices**(
    y_test[X_test.race==1],
    y_test_pred[X_test.race==1],
    y_test[X_test.race==0],
    y_test_pred[X_test.race==0],
    'Caucasian',
    'African-American',
     **compare_fpr=****True**
) 
Figure 12.19 and some predictive performance metrics. If we compare the matrices to those in *Figure 12.16*, racial disparities, as measured by our FPR ratio, took a hit. Also, predictive performance is lower than the optimized CatBoost model across the board, by 2–4%. We could likely increase these metrics a bit by performing the same *Bayesian hyperparameter tuning* on this model.
```

可以在这里看到混淆矩阵的输出：

![图表描述自动生成](img/B18406_12_19.png)

图 12.19：约束 XGBoost 模型不同种族之间的混淆矩阵比较

有一个需要考虑的事情是，尽管种族不平等是本章的主要关注点，但我们还希望确保模型在其他方面也是最优的。正如之前所述，这是一个权衡。例如，被告的 `priors_per_year` 越多，风险越高，这是很自然的，我们通过单调约束确保了这一点。让我们验证这些结果！

### 检查约束

观察约束条件在作用中的简单方法是将 SHAP `summary_plot` 绘制出来，就像我们在 *图 12.17* 中所做的那样，但这次我们只绘制一个。请看下面的 ode 程序片段：

```py
fitted_xgb_con_mdl = fitted_class_mdls['xgb_con']['fitted']
shap_xgb_con_explainer = shap.**TreeExplainer**(fitted_xgb_con_mdl)
shap_xgb_con_values = shap_xgb_con_explainer.**shap_values**(
    X_test_con
)
shap.**summary_plot**(
    shap_xgb_con_values, X_test_con, plot_type="dot"
) 
```

上述代码生成了 *图 12.20*。这展示了从左到右的 `priors_per_year` 是一个更干净的梯度，这意味着较低的值持续产生负面影响，而较高的值产生正面影响——正如它们应该的那样！

你可以在这里看到输出：

![图表描述自动生成](img/B18406_12_20.png)

图 12.20：约束 XGBoost 模型的 SHAP 概述图

接下来，让我们通过 *图 12.7* 中的数据视角检查我们看到的 `age_group` 与 `priors_per_year` 的交互。我们也可以通过添加额外的参数来为模型使用 `plot_prob_contour_map`，如下所示：

+   拟合的模型 (`fitted_xgb_con_mdl`)

+   用于模型推理的 DataFrame (`X_test_con`)

+   在每个轴上比较的 DataFrame 中两列的名称（`x_col`和`y_col`）

结果是一个交互部分依赖图，类似于*第四章*中展示的，*全局模型无关解释方法*，只不过它使用数据集（`recidivism_df`）为每个轴创建直方图。我们现在将创建两个这样的图进行比较——一个用于正则化的 XGBoost 模型，另一个用于约束模型。此代码的示例如下：

```py
mldatasets.**plot_prob_contour_map**(
    recidivism_df.**age_group**, recidivism_df.**priors_per_year**,
    recidivism_df.**is_recid**, x_intervals=ordenc.categories_[2],
    y_intervals=6, use_quantiles=True, xlabel='Age Group',
    ylabel='Priors Per Year', model=**fitted_xgb_mdl**,
    X_df=**X_test**,x_col='age_group',y_col='priors_per_year',
    title='Probability of Recidivism by Age/Priors per Year \
          (according to XGBoost Regularized Model)'
)
mldatasets.**plot_prob_contour_map**(
    recidivism_df.**age_group**, recidivism_df.**priors_per_year**,
    recidivism_df.is_recid, x_intervals=ordenc.categories_[2],
    y_intervals=6, use_quantiles=True, xlabel='Age Group',
    ylabel='Priors Per Year', model=**fitted_xgb_con_mdl**,
    X_df=**X_test_con**,x_col='age_group',y_col='priors_per_year',
    title='(according to XGBoost Constrained Model)'
) 
```

上述代码生成了*图 12.21*中显示的图表。它表明正则化的 XGBoost 模型反映了数据（参见*图 12.7*）。另一方面，约束的 XGBoost 模型平滑并简化了等高线，如下所示：

![图表，自动生成描述](img/B18406_12_21.png)

图 12.21：根据 XGBoost 正则化和约束模型，针对 age_group 和 priors_per_year 的再犯概率等高线图

接下来，我们可以从*图 12.18*生成 SHAP 交互值热图，但针对的是约束模型。代码相同，但使用`shap_xgb_con_explainer` SHAP 解释器和`X_test_con`数据。代码的示例如下：

```py
shap_xgb_interact_values =\
        shap_xgb_con_explainer.**shap_interaction_values**(X_test_con)
shap_xgb_interact_df =\
        pd.**DataFrame**(np.sum(**shap_xgb_interact_values**, axis=0))
shap_xgb_interact_df.columns = X_test_con.columns
shap_xgb_interact_df.index = X_test_con.columns
sns.**heatmap**(
    shap_xgb_interact_df, cmap='RdBu', annot=True,
    annot_kws={'size':13}, fmt='.0f', linewidths=.5
) 
Figure 12.22. It shows how the interaction constraints were effective because of zeros in the lower-left and lower-right quadrants, which correspond to interactions between the two groups of features we separated. If we compare with *Figure 12.18*, we can also tell how the constraints shifted the most salient interactions, making age_group and length_of_stay by far the most important ones.
```

输出结果如下：

![包含应用的图片，自动生成描述](img/B18406_12_22.png)

图 12.22：约束 XGBoost 模型的 SHAP 交互值热图

现在，让我们看看 TensorFlow 是如何通过 TensorFlow Lattice 实现单调性和其他“形状约束”的。

## TensorFlow Lattice 的约束条件

神经网络在寻找`loss`函数的最优解方面可以非常高效。损失与我们要预测的后果相关联。在这种情况下，那将是 2 年的再犯率。在伦理学中，*功利主义*（或*后果主义*）的公平观只要模型的训练数据没有偏见，就没有问题。然而，*义务论*的观点是，伦理原则或政策驱动着伦理问题，并超越后果。受此启发，**TensorFlow Lattice**（**TFL**）可以在模型中将伦理原则体现为模型形状约束。

晶格是一种**插值查找表**，它通过插值近似输入到输出的网格。在高维空间中，这些网格成为超立方体。每个输入到输出的映射通过**校准层**进行约束，并且支持许多类型的约束——不仅仅是单调性。*图 12.23*展示了这一点：

![图表，自动生成描述](img/B18406_12_23.png)

图 12.23：TensorFlow Lattice 支持的约束条件

*图 12.23*展示了几个形状约束。前三个应用于单个特征（*x*），约束了![](img/B18406_12_006.png)线，代表输出。最后两个应用于一对特征（*x*[1]和*x*[2]），约束了彩色等高线图（![](img/B18406_12_007.png)）。以下是对每个约束的简要说明：

+   **单调性**：这使得函数（![](img/B18406_12_008.png)）相对于输入（*x*）总是增加（1）或减少（-1）。

+   **凸性**：这迫使函数（![](img/B18406_12_009.png)）相对于输入（*x*）是凸的（1）或凹的（-1）。凸性可以与单调性结合，产生*图 12.23*中的效果。

+   **单峰性**：这类似于单调性，不同之处在于它向两个方向延伸，允许函数（![](img/B18406_12_010.png)）有一个单一的谷底（1）或峰值（-1）。

+   **信任**：这迫使一个单调特征（*x*[1]）依赖于另一个特征（*x*[2]）。*图 12.23*中的例子是**爱德华兹信任**，但也有一个具有不同形状约束的**梯形信任**变体。

+   **支配性**：单调支配性约束一个单调特征（*x*[1]）定义斜率或效果的方向，当与另一个特征（*x*[2]）比较时。另一种选择，范围支配性，类似，但两个特征都是单调的。

神经网络特别容易过拟合，控制它的杠杆相对更难。例如，确切地说，隐藏节点、dropout、权重正则化和 epoch 的哪种组合会导致可接受的过拟合水平是难以确定的。另一方面，在基于树的模型中移动单个参数，即树深度，朝一个方向移动，可能会将过拟合降低到可接受的水平，尽管可能需要许多不同的参数才能使其达到最佳状态。

强制形状约束不仅增加了可解释性，还因为简化了函数而正则化了模型。TFL 还支持基于惩罚的正则化，针对每个特征或校准层的核，利用**拉普拉斯**、**海森**、**扭转**和**皱纹**正则化器通过 L1 和 L2 惩罚。这些正则化器的作用是使函数更加平坦、线性或平滑。我们不会详细解释，但可以说，存在正则化来覆盖许多用例。

实现框架的方法也有几种——太多，这里无法一一详述！然而，重要的是指出，这个例子只是实现它的几种方法之一。TFL 内置了**预定义的估计器**，它们抽象了一些配置。您还可以使用 TFL 层创建一个**自定义估计器**。对于 Keras，您可以使用**预制的模型**，或者使用 TensorFlow Lattice 层构建一个 Keras 模型。接下来，我们将进行最后一项操作！

### 初始化模型和 Lattice 输入

现在我们将创建一系列**输入层**，每个输入层包含一个特征。这些层连接到**校准层**，使每个输入适合符合个体约束和正则化的**分段线性**（**PWL**）函数，除了`sex`，它将使用分类校准。所有校准层都输入到一个多维**晶格层**，通过一个具有**sigmoid**激活的**密集层**产生输出。这个描述可能有点难以理解，所以您可以自由地跳到**图 12.24**以获得一些视觉辅助。

顺便说一下，有许多种类的层可供连接，以产生**深度晶格网络**（**DLN**），包括以下内容：

+   **线性**用于多个输入之间的线性函数，包括具有支配形状约束的函数。

+   **聚合**用于对多个输入执行聚合函数。

+   **并行组合**将多个校准层放置在单个函数中，使其与 Keras `Sequential`层兼容。

在这个例子中，我们不会使用这些层，但也许了解这些会激发您进一步探索 TensorFlow Lattice 库。无论如何，回到这个例子！

首先要定义的是`lattice_sizes`，它是一个元组，对应于每个维度的顶点数。在所选架构中，每个特征都有一个维度，因此我们需要选择九个大于或等于 2 的数字。对于分类特征的基数较小的特征或连续特征的拐点，需要较少的顶点。然而，我们也可能想通过故意选择更少的顶点来限制特征的表达能力。例如，`juv_fel_count`有 10 个唯一值，但我们将只给它分配两个顶点。`lattice_sizes`如下所示：

```py
lattice_sizes = [2, 2, 2, 2, 4, 5, 7, 7, 7] 
```

接下来，我们初始化两个列表，一个用于放置所有输入层（`model_inputs`）和另一个用于校准层（`lattice_inputs`）。然后，对于每个特征，我们逐一定义一个输入层使用`tf.keras.layers.Input`和一个校准层使用分类校准（`tfl.layers.CategoricalCalibration`）或 PWL 校准（`tfl.layers.PWLCalibration`）。每个特征的所有输入和校准层都将分别添加到各自的列表中。校准层内部发生的事情取决于特征。所有 PWL 校准都使用`input_keypoints`，它询问 PWL 函数应该在何处分段。有时，使用固定宽度（`np.linspace`）回答这个问题是最好的，而有时使用固定频率（`np.quantile`）。分类校准则使用桶（`num_buckets`），它对应于类别的数量。所有校准器都有以下参数：

+   `output_min`：校准器的最小输出

+   `output_max`：校准器的最大输出——始终必须与输出最小值 + 晶格大小 - 1 相匹配

+   `monotonicity`：是否应该单调约束 PWL 函数，如果是，如何约束

+   `kernel_regularizer`：如何正则化函数

除了这些参数之外，`convexity` 和 `is_cyclic`（对于单调单峰）可以修改约束形状。看看下面的代码片段：

```py
model_inputs = []
lattice_inputs = []
sex_input = **tf.keras.layers.Input**(shape=[1], name='sex')
lattice_inputs.append(**tfl.layers.CategoricalCalibration**(
    name='sex_calib',
    num_buckets=2,
    output_min=0.0,
    output_max=lattice_sizes[0] - 1.0,
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001),
    kernel_initializer='constant')(sex_input)
)
model_inputs.append(sex_input)
juvf_input = **tf.keras.layers.Input**(shape=[1],\
                                   name='juv_fel_count')
lattice_inputs.append(**tfl.layers.PWLCalibration**(
    name='juvf_calib',
    **monotonicity**='none',
    input_keypoints=np.linspace(0, 20, num=5, dtype=np.float32),
    output_min=0.0,
    output_max=lattice_sizes[1] - 1.0,\
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001),
    kernel_initializer='equal_slopes')(juvf_input)
)
model_inputs.append(juvf_input)
age_input = **tf.keras.layers.Input**(shape=[1], name='age_group')
lattice_inputs.append(**tfl.layers.PWLCalibration**(
    name='age_calib',
    **monotonicity**='none',
    input_keypoints=np.linspace(0, 6, num=7, dtype=np.float32),
    output_min=0.0,
    output_max=lattice_sizes[7] - 1.0,
    kernel_regularizer=('hessian', 0.0, 1e-4))(age_input)
)
model_inputs.append(age_input)
priors_input = **tf.keras.layers.Input**(shape=[1],\
                                     name='priors_per_year')
lattice_inputs.append(**tfl.layers.PWLCalibration**(
    name='priors_calib',
    **monotonicity**='increasing',
    input_keypoints=np.quantile(X_train_con['priors_per_year'],
                                np.linspace(0, 1, num=7)),
    output_min=0.0,
    output_max=lattice_sizes[8]-1.0)(priors_input))
model_inputs.append(priors_input) 
```

因此，我们现在有一个包含 `model_inputs` 的列表和另一个包含校准层的列表，这些校准层将成为 lattice 的输入（`lattice_inputs`）。我们现在需要做的就是将这些连接到一个 lattice 上。

### 使用 TensorFlow Lattice 层构建 Keras 模型

我们已经将这个模型的前两个构建块连接起来。现在，让我们创建最后两个构建块，从 lattice (`tfl.layers.Lattice`) 开始。作为参数，它接受 `lattice_sizes`、输出最小值和最大值以及它应该执行的 `monotonicities`。注意，最后一个参数 `priors_per_year` 的单调性设置为 `increasing`。然后，lattice 层将输入到最终的部件，即具有 `sigmoid` 激活的 `Dense` 层。代码如下所示：

```py
lattice = **tfl.layers.Lattice**(
    name='lattice',
    lattice_sizes=**lattice_sizes**,
    **monotonicities**=[
        'none', 'none', 'none', 'none', 'none',
        'none', 'none', 'none', **'increasing'**
    ],
    output_min=0.0, output_max=1.0)(**lattice_inputs**)
model_output = tf.keras.layers.**Dense**(1, name='output',
                                     activation='sigmoid')(lattice) 
```

前两个构建块作为 `inputs` 现在可以与最后两个作为 `outputs` 通过 `tf.keras.models.Model` 连接起来。哇！我们现在有一个完整的模型，代码如下所示：

```py
tfl_mdl = **tf.keras.models.Model**(inputs=model_inputs,
                                outputs=model_output) 
```

你总是可以运行 `tfl_mdl.summary()` 来了解所有层是如何连接的，但使用 `tf.keras.utils.plot_model` 更直观，如下面的代码片段所示：

```py
tf.keras.utils.plot_model(tfl_mdl, rankdir='LR') 
```

上述代码生成了 *图 12.24* 中显示的模型图：

![图  描述自动生成](img/B18406_12_24.png)

图 12.24：带有 TFL 层的 Keras 模型图

接下来，我们需要编译模型。我们将使用 `binary_crossentropy` 损失函数和 `Adam` 优化器，并使用准确率和 **曲线下面积**（**AUC**）作为指标，如下面的代码片段所示：

```py
tfl_mdl.**compile**(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy',tf.keras.metrics.AUC(name='auc')]
) 
```

我们现在几乎准备就绪了！接下来是最后一步。

### 训练和评估模型

如果你仔细观察 *图 12.24*，你会注意到模型没有一层输入，而是有九层，这意味着我们必须将我们的训练和测试数据分成九部分。我们可以使用 `np.split` 来做这件事，这将产生九个 NumPy 数组的列表。至于标签，TFL 不接受单维数组。使用 `expand_dims`，我们将它们的形状从 `(N,)` 转换为 `(N,1)`，如下面的代码片段所示：

```py
X_train_expand = np.**split**(
    X_train_con.values.astype(np.float32),
    indices_or_sections=9,
    axis=1
)
y_train_expand = np.**expand_dims**(
    y_train.values.astype(np.float32),
    axis=1
)
X_test_expand = np.**split**(
    X_test_con.values.astype(np.float32),
    indices_or_sections=9,
    axis=1)
y_test_expand = np.**expand_dims**(
    y_test.values.astype(np.float32),
    axis=1
) 
```

接下来是训练！为了防止过拟合，我们可以通过监控验证 AUC (`val_auc`) 来使用 `EarlyStopping`。为了解决类别不平衡问题，在 `fit` 函数中，我们使用 `class_weight`，如下面的代码片段所示：

```py
es = tf.keras.callbacks.**EarlyStopping**(
    monitor='**val_auc**',
    mode='max',
    patience=40,
    restore_best_weights=True
)
tfl_history = tfl_mdl.**fit**(
    X_train_expand,
    y_train_expand,
    **class_weight**={0:18, 1:16},
    batch_size=128,
    epochs=300,
    validation_split=0.2,
    shuffle=True,
    callbacks=[**es**]
) 
```

一旦模型训练完成，我们可以使用 `evaluate_class_mdl` 来输出预测性能的快速摘要，就像我们之前做的那样，然后使用 `compare_confusion_matrices` 来检查公平性，就像我们之前所做的那样。代码如下所示：

```py
fitted_class_mdls['tfl_con'] = mldatasets.**evaluate_class_mdl**(
    tfl_mdl,
    X_train_expand,
    X_test_expand,
    y_train.values.astype(np.float32),
    y_test.values.astype(np.float32),
    plot_roc=False,
    ret_eval_dict=True
)
y_test_pred = fitted_class_mdls['tfl_con']['preds_test']
_ = mldatasets.**compare_confusion_matrices**(
    y_test[X_test.race==1],
    y_test_pred[X_test.race==1],
    y_test[X_test.race==0],
    y_test_pred[X_test.race==0],
    'Caucasian',
    'African-American',
    compare_fpr=True
) 
Figure 12.25. The TensorFlow Lattice model performs much better than the regularized Keras model, yet the FPR ratio is better than the constrained XGBoost model. It must be noted that XGBoost’s parameters were previously tuned. With TensorFlow Lattice, a lot could be done to improve FPR, including using a custom loss function or better early-stopping metrics that somehow account for racial disparities.
```

输出如下所示：

![图表，树状图  描述自动生成](img/B18406_12_25.png)

图 12.25：约束 TensorFlow Lattice 模型在种族之间的混淆矩阵比较

接下来，我们将根据本章学到的内容得出一些结论，并确定我们是否完成了任务。

# 任务完成

通常，数据会因为表现不佳、不可解释或存在偏见而被责备，这可能是真的，但在准备和模型开发阶段可以采取许多不同的措施来改进它。为了提供一个类比，这就像烘焙蛋糕。你需要高质量的原料，是的。但似乎微小的原料准备和烘焙本身——如烘焙温度、使用的容器和时间——的差异可以产生巨大的影响。天哪！甚至是你无法控制的事情，如大气压力或湿度，也会影响烘焙！甚至在完成之后，你有多少种不同的方式可以评估蛋糕的质量？

本章讨论了这些许多细节，就像烘焙一样，它们既是**精确科学**的一部分，也是**艺术形式**的一部分。本章讨论的概念也具有深远的影响，特别是在如何优化没有单一目标且具有深远社会影响的问题方面。一种可能的方法是结合指标并考虑不平衡。为此，我们创建了一个指标：一个加权平均的精确率召回率，它惩罚种族不平等，并且我们可以为所有模型高效地计算它并将其放入模型字典（`fitted_class_mdls`）。然后，就像我们之前做的那样，我们将其放入 DataFrame 并输出，但这次是按照自定义指标（`wppra_test`）排序。代码可以在下面的代码片段中看到：

```py
for mdl_name in fitted_class_mdls:
    fitted_class_mdls[mdl_name]['wppra_test'] =\
    **weighted_penalized_pr_average**(
        y_test,
        fitted_class_mdls[mdl_name]['preds_test'],
        X_test['race'],
        range(3)
    )
class_metrics = pd.**DataFrame.from_dict**(fitted_class_mdls, 'index')[
    ['precision_test', 'recall_test', 'wppra_test']
]
with pd.option_context('display.precision', 3):
    html = class_metrics.**sort_values**(
        by='**wppra_test**',
        ascending=False
        ).style.background_gradient(
           cmap='plasma',subset=['precision_test']
        ).background_gradient(
           cmap='viridis', subset=['recall_test'])
html 
```

上一段代码生成了*图 12.26*中显示的 DataFrame：

![表格描述自动生成](img/B18406_12_26.png)

图 12.26：按加权惩罚精确率-召回率平均值自定义指标排序的本章顶级模型

在*图 12.26*中，很容易提出最上面的其中一个模型。然而，它们是用`race`作为特征进行训练的，并且没有考虑到证明的刑事司法*现实*。然而，性能最高的约束模型——XGBoost 模型（`xgb_con`）——没有使用`race`，确保了`priors_per_year`是单调的，并且不允许`age_group`与青少年犯罪特征相互作用，而且与原始模型相比，它在显著提高预测性能的同时做到了这一切。它也更公平，因为它将特权群体和弱势群体之间的 FPR 比率从 1.84x（*第六章*中的*图 6.2*）降低到 1.39x（*图 12.19*）。它并不完美，但这是一个巨大的改进！

任务是证明准确性和领域知识可以与公平性的进步共存，我们已经成功地完成了它。话虽如此，仍有改进的空间。因此，行动计划将不得不向您的客户展示受约束的 XGBoost 模型，并继续改进和构建更多受约束的模型。未受约束的模型应仅作为基准。

如果你将本章的方法与第十一章中学习的那些方法（*偏差缓解和因果推断方法*）相结合，你可以实现显著的公平性改进。我们没有将这些方法纳入本章，以专注于通常不被视为偏差缓解工具包一部分的模型（或内处理）方法，但它们在很大程度上可以协助达到这一目的，更不用说那些旨在使模型更可靠的模型调优方法了。

# 摘要

阅读本章后，你现在应该了解如何利用数据工程来增强可解释性，正则化来减少过拟合，以及约束来符合政策。主要的目标是设置护栏和遏制阻碍可解释性的复杂性。

在下一章中，我们将探讨通过对抗鲁棒性来增强模型可靠性的方法。

# 数据集来源

+   ProPublica 数据存储库 (2019). *COMPAS 再犯风险评分数据和分析*. 原始数据检索自 [`www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis`](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis)

# 进一步阅读

+   Hastie, T. J., Tibshirani, R. J. 和 Friedman, J. H. (2001). *统计学习的要素*. Springer-Verlag, 纽约，美国

+   Wang, S. & Gupta, M. (2020). *通过单调性形状约束的德性伦理*. AISTATS. [`arxiv.org/abs/2001.11990`](https://arxiv.org/abs/2001.11990)

+   Cotter, A., Gupta, M., Jiang, H., Ilan, E. L., Muller, J., Narayan, T., Wang, S. 和 Zhu, T. (2019). *集合函数的形状约束*. ICML. [`proceedings.mlr.press/v97/cotter19a.html`](http://proceedings.mlr.press/v97/cotter19a.html)

+   Gupta, M. R., Cotter A., Pfeifer, J., Voevodski, K., Canini, K., Mangylov, A., Moczydlowski, W. 和 van Esbroeck, A. (2016). *单调校准插值查找表. 机器学习研究杂志* 17(109):1−47\. [`arxiv.org/abs/1505.06378`](https://arxiv.org/abs/1505.06378)

+   Noble, S. (2018). *压迫算法：谷歌时代的数据歧视*. NYU Press

# 在 Discord 上了解更多

要加入本书的 Discord 社区——在那里您可以分享反馈，向作者提问，并了解新发布的内容——请扫描下面的二维码：

`packt.link/inml`

![](img/QR_Code107161072033138125.png)
