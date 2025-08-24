

# 第十章：可解释性特征选择与工程

在前三章中，我们讨论了复杂性如何阻碍**机器学习**（**ML**）的可解释性。这里有一个权衡，因为你可能需要一些复杂性来最大化预测性能，但又不能达到无法依赖模型来满足可解释性原则：公平性、责任和透明度的程度。本章是四个专注于如何调整以实现可解释性的章节中的第一个。提高可解释性最简单的方法之一是通过特征选择。它有许多好处，例如加快训练速度并使模型更容易解释。但如果这两个原因不能说服你，也许另一个原因会。

一个常见的误解是，复杂的模型可以自行选择特征并仍然表现良好，那么为什么还要费心选择特征呢？是的，许多模型类别都有机制可以处理无用的特征，但它们并不完美。而且，随着每个剩余的机制的加入，过拟合的可能性也会增加。过拟合的模型是不可靠的，即使它们更准确。因此，虽然仍然强烈建议使用模型机制，如正则化，以避免过拟合，但特征选择仍然是有用的。

在本章中，我们将理解无关特征如何对模型的输出产生不利影响，从而了解特征选择对模型可解释性的重要性。然后，我们将回顾基于过滤器的特征选择方法，如**斯皮尔曼相关系数**，并了解嵌入式方法，如**LASSO 和岭回归**。然后，我们将发现包装方法，如**顺序特征选择**，以及混合方法，如**递归特征消除**（**RFE**）。最后，尽管特征工程通常在选择之前进行，但在特征选择完成后，探索特征工程仍有其价值。

这些是我们将在本章中讨论的主要主题：

+   理解无关特征的影响

+   回顾基于过滤器的特征选择方法

+   探索嵌入式特征选择方法

+   发现包装、混合和高级特征选择方法

+   考虑特征工程

让我们开始吧！

# 技术要求

本章的示例使用了`mldatasets`、`pandas`、`numpy`、`scipy`、`mlxtend`、`sklearn-genetic-opt`、`xgboost`、`sklearn`、`matplotlib`和`seaborn`库。有关如何安装所有这些库的说明见**前言**。

本章的 GitHub 代码位于此处：[`packt.link/1qP4P`](https://packt.link/1qP4P)。

# 任务

据估计，全球有超过 1000 万个非营利组织，尽管其中很大一部分有公共资金，但大多数组织主要依赖私人捐赠者，包括企业和个人，以继续运营。因此，筹款是至关重要的任务，并且全年都在进行。

年复一年，捐款收入有所增长，但非营利组织面临一些问题：捐赠者的兴趣在变化，因此一年受欢迎的慈善机构可能在下一年被遗忘；非营利组织之间的竞争激烈，人口结构也在变化。在美国，平均捐赠者每年只捐赠两次慈善礼物，且年龄超过 64 岁。识别潜在捐赠者具有挑战性，而且吸引他们的活动可能成本高昂。

一个国家级退伍军人组织非营利分支拥有大约 190,000 名往届捐赠者的庞大邮件列表，并希望发送一份特别邮件请求捐款。然而，即使有特殊的批量折扣率，每地址的成本也高达 0.68 美元。这总计超过 130,000 美元。他们的市场预算只有 35,000 美元。鉴于他们已将此事列为高优先级，他们愿意扩展预算，但前提是**投资回报率**（**ROI**）足够高，以证明额外成本是合理的。

为了最大限度地减少使用他们有限的预算，他们希望尝试直接邮寄，目的是利用已知的信息来识别潜在捐赠者，例如过去的捐赠、地理位置和人口统计数据。他们将通过电子邮件联系其他捐赠者，这要便宜得多，整个列表的月成本不超过 1,000 美元。他们希望这种混合营销计划能产生更好的结果。他们还认识到，高价值捐赠者对个性化的纸质邮件响应更好，而较小的捐赠者无论如何对电子邮件的响应更好。

最多只有 6%的邮件列表捐赠者会对任何特定的活动进行捐赠。使用机器学习预测人类行为绝非易事，尤其是在数据类别不平衡的情况下。尽管如此，成功不是以最高的预测准确性来衡量的，而是以利润提升来衡量。换句话说，在测试数据集上评估的直接邮寄模型应该产生比如果他们向整个数据集进行群发邮件更多的利润。

他们寻求您的帮助，使用机器学习（ML）来生成一个模型，以识别最可能的捐赠者，但同时也保证一个高的 ROI。

您收到了非营利组织的数据集，该数据集大约平均分为训练数据和测试数据。如果您向测试数据集中的所有人发送邮件，您将获得 11,173 美元的利润，但如果您能够仅识别那些会捐赠的人，最大收益将达到 73,136 美元。您的目标是实现高利润提升和合理的 ROI。当活动进行时，它将识别整个邮件列表中最可能的捐赠者，非营利组织希望总支出不超过 35,000 美元。然而，数据集有 435 个列，一些简单的统计测试和建模练习表明，由于过度拟合，数据过于嘈杂，无法识别潜在捐赠者的可靠性。

# 方法

你决定首先使用所有特征拟合一个基础模型，并在不同的复杂度级别上评估它，以了解特征数量增加与预测模型过度拟合训练数据之间的关联。然后，你将采用一系列从简单的基于过滤的方法到最先进的方法的特征选择方法，以确定哪种方法实现了客户寻求的盈利性和可靠性目标。最后，一旦选定了最终特征列表，你就可以尝试特征工程。

由于问题的成本敏感性，阈值对于优化利润提升至关重要。我们将在稍后讨论阈值的作用，但一个显著的影响是，尽管这是一个分类问题，最好使用回归模型，然后使用预测来分类，这样只有一个阈值需要调整。也就是说，对于分类模型，你需要一个用于标签的阈值，比如那些捐赠超过 1 美元的，然后还需要另一个用于预测概率的阈值。另一方面，回归预测捐赠金额，阈值可以根据这个进行优化。

# 准备工作

此示例的代码可以在[`github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/blob/main/10/Mailer.ipynb`](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/blob/main/10/Mailer.ipynb)找到。

## 加载库

要运行此示例，我们需要安装以下库：

+   使用`mldatasets`加载数据集

+   使用`pandas`、`numpy`和`scipy`来操作它

+   使用`mlxtend`、`sklearn-genetic-opt`、`xgboost`和`sklearn`（scikit-learn）来拟合模型

+   使用`matplotlib`和`seaborn`创建和可视化解释

要加载库，请使用以下代码块：

```py
import math
import os
import mldatasets
import pandas as pd
import numpy as np
import timeit
from tqdm.notebook import tqdm
from sklearn.feature_selection import VarianceThreshold,\
                                    mutual_info_classif, SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression,\
                                    LassoCV, LassoLarsCV, LassoLarsIC
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA import shap
from sklearn-genetic-opt import GAFeatureSelectionCV
from scipy.stats import rankdata
from sklearn.discriminant_analysis import
LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns 
```

接下来，我们将加载并准备数据集。

## 理解和准备数据

我们将数据这样加载到两个 DataFrame（`X_train`和`X_test`）中，其中包含特征，以及两个相应的`numpy`数组标签（`y_train`和`y_test`）。请注意，这些 DataFrame 已经为我们预先准备，以删除稀疏或不必要的特征，处理缺失值，并对分类特征进行编码：

```py
X_train, X_test, y_train, y_test = mldatasets.load(
    "nonprofit-mailer",
    prepare=True
)
y_train = y_train.squeeze()
y_test = y_test.squeeze() 
```

所有特征都是数值型，没有缺失值，并且分类特征已经为我们进行了一热编码。在训练和测试邮件列表之间，应有超过 191,500 条记录和 435 个特征。你可以这样检查：

```py
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape) 
```

上一段代码应该输出以下内容：

```py
(95485, 435)
(95485,)
(96017, 435)
(96017,) 
```

接下来，我们可以使用变量成本 0.68（`var_cost`）验证测试标签是否有正确的捐赠者数量（`test_donors`）、捐赠金额（`test_donations`）和假设的利润范围（`test_min_profit`和`test_max_profit`）。我们可以打印这些信息，然后对训练数据集做同样的操作：

```py
var_cost = 0.68
y_test_donors = y_test[y_test > 0]
test_donors = len(y_test_donors)
test_donations = sum(y_test_donors)
test_min_profit = test_donations - (len(y_test)*var_cost)
test_max_profit = test_donations - (test_donors*var_cost)
print(
    '%s test donors totaling $%.0f (min profit: $%.0f,\
    max profit: $%.0f)'
    %(test_donors, test_donations, test_min_profit,\
       test_max_profit))
y_train_donors = y_train[y_train > 0]
train_donors = len(y_train_donors)
train_donations = sum(y_train_donors)
train_min_profit = train_donations – (len(y_train)*var_cost)
train_max_profit = train_donations – (train_donors*var_cost)
print(
    '%s train donors totaling $%.0f (min profit: $%.0f,\
    max profit: $%.0f)'
    %(train_donors, train_donations, train_min_profit,\
    train_max_profit)) 
```

上一段代码应该输出以下内容：

```py
4894 test donors totaling $76464 (min profit: $11173, max profit: $73136)
4812 train donors totaling $75113 (min profit: $10183, max profit: $71841) 
```

事实上，如果非营利组织向测试邮件列表上的每个人大量邮寄，他们可能会获得大约 11,000 美元的利润，但为了实现这一目标，他们必须严重超支。非营利组织认识到，通过仅识别和针对捐赠者来获得最大利润几乎是一项不可能完成的任务。因此，他们宁愿生产一个能够可靠地产生超过最低利润但成本更低的模型，最好是低于预算。

# 理解无关特征的影响

**特征选择**也称为**变量**或**属性选择**。这是你可以自动或手动选择一组对构建机器学习模型有用的特定特征的方法。

并非更多的特征就一定能导致更好的模型。无关特征可能会影响学习过程，导致过拟合。因此，我们需要一些策略来移除可能对学习产生不利影响的任何特征。选择较小特征子集的一些优点包括以下内容：

+   *理解简单的模型更容易*：例如，对于使用 15 个变量的模型，其特征重要性比使用 150 个变量的模型更容易理解。

+   *缩短训练时间*：减少变量的数量可以降低计算成本，加快模型训练速度，而且最值得注意的是，简单的模型具有更快的推理时间。

+   *通过减少过拟合来提高泛化能力*：有时，预测价值很小，许多变量只是噪声。然而，机器学习模型却会从这些噪声中学习，并在最小化泛化的同时触发对训练数据的过拟合。通过移除这些无关或噪声特征，我们可以显著提高机器学习模型的泛化能力。

+   *变量冗余*：数据集中存在共线性特征是很常见的，这可能意味着某些特征是冗余的。在这些情况下，只要没有丢失显著信息，我们只需保留一个相关的特征，删除其他特征即可。

现在，我们将拟合一些模型来展示过多特征的影响。

## 创建基础模型

让我们为我们的邮件列表数据集创建一个基础模型，看看这将如何展开。但首先，让我们设置随机种子以确保可重复性：

```py
rand = 9
os.environ['PYTHONHASHSEED']=str(rand)
np.random.seed(rand) 
```

在本章中，我们将使用 XGBoost 的**随机森林**（**RF**）回归器（`XGBRFRegressor`）。它就像 scikit-learn 一样，但更快，因为它使用了目标函数的二阶近似。它还有更多选项，例如设置学习率和单调约束，这些在*第十二章*，*单调约束和模型调优以提高可解释性*中进行了考察。我们以保守的初始`max_depth`值`4`初始化`XGBRFRegressor`，并始终使用`200`估计量以确保一致性。然后，我们使用我们的训练数据对其进行拟合。我们将使用`timeit`来测量它需要多长时间，并将其保存在变量（`baseline_time`）中供以后参考：

```py
stime = timeit.default_timer()
reg_mdl = xgb.XGBRFRegressor(max_depth=4, n_estimators=200, seed=rand)
fitted_mdl = reg_mdl.fit(X_train, y_train)
etime = timeit.default_timer()
baseline_time = etime-stime 
```

现在我们已经有一个基础模型了，让我们来评估它。

## 评估模型

接下来，让我们创建一个字典(`reg_mdls`)来存放我们将在本章中拟合的所有模型，以测试哪些特征子集会产生最好的模型。在这里，我们可以使用`evaluate_reg_mdl`来评估具有所有特征和`max_depth`值为`4`的随机森林模型(`rf_4_all`)。它将生成一个总结和一个带有回归线的散点图：

```py
reg_mdls = {}
reg_mdls['rf_4_all'] = mldatasets.evaluate_reg_mdl(
    fitted_mdl,
    X_train,
    X_test,
    y_train,
    y_test,
    plot_regplot=True,
    ret_eval_dict=True
) 
```

之前的代码生成了*图 10.1*中显示的指标和图表：

![图表描述自动生成](img/B18406_10_01.png)

图 10.1：基础模型的预测性能

对于像*图 10.1*这样的图表，通常期望看到一条对角线，所以一眼看去就能判断出这个模型不具备预测性。此外，均方根误差（RMSEs）可能看起来并不糟糕，但在这种不平衡的问题背景下，它们却是令人沮丧的。考虑一下这个情况：只有 5%的人捐赠，而其中只有 20%的人捐赠额超过 20 美元，所以平均误差 4.3 美元至 4.6 美元是巨大的。

那么，这个模型有没有用呢？答案在于我们用它来分类所使用的阈值。让我们首先定义一个从$0.40 到$25 的阈值数组(`threshs`)，我们首先以每 0.01 美元的间隔来设置这些阈值，直到达到$1，然后以每 0.1 美元的间隔设置，直到达到$3，之后以每 1 美元的间隔设置：

```py
threshs = np.hstack(
    [
      np.linspace(0.40,1,61),
      np.linspace(1.1,3,20),
      np.linspace(4,25,22)
    ]
) 
```

`mldatasets`中有一个函数可以计算每个阈值下的利润(`profits_by_thresh`)。它只需要实际的(`y_test`)和预测标签，然后是阈值(`threshs`)、可变成本(`var_costs`)和所需的`min_profit`。只要利润高于`min_profit`，它就会生成一个包含每个阈值的收入、成本、利润和投资回报率的`pandas` DataFrame。记住，我们在本章开始时将这个最低值设置为$11,173，因为针对低于这个金额的捐赠者是没有意义的。在为测试和训练数据集生成这些利润 DataFrame 之后，我们可以将这些最大和最小金额放入模型的字典中，以供以后使用。然后，我们使用`compare_df_plots`来绘制每个阈值的测试和训练的成本、利润和投资回报率比率，只要它超过了利润最低值：

```py
y_formatter = plt.FuncFormatter(
    lambda x, loc: "${:,}K".format(x/1000)
)
profits_test = mldatasets.profits_by_thresh(
    y_test,
    reg_mdls['rf_4_all']['preds_test'],
    threshs,
    var_costs=var_cost,
    min_profit=test_min_profit
)
profits_train = mldatasets.profits_by_thresh(
    y_train,
    reg_mdls['rf_4_all']['preds_train'],
    threshs,
    var_costs=var_cost,
    min_profit=train_min_profit
)
reg_mdls['rf_4_all']['max_profit_train'] =profits_train.profit.max()
reg_mdls['rf_4_all']['max_profit_test'] = profits_test.profit.max()
reg_mdls['rf_4_all']['max_roi'] = profits_test.roi.max()
reg_mdls['rf_4_all']['min_costs'] = profits_test.costs.min()
reg_mdls['rf_4_all']['profits_train'] = profits_train
reg_mdls['rf_4_all']['profits_test'] = profits_test
mldatasets.compare_df_plots(
    profits_test[['costs', 'profit', 'roi']],
    profits_train[['costs', 'profit', 'roi']],
    'Test',
    'Train',
    y_formatter=y_formatter,
    x_label='Threshold',\
    plot_args={'secondary_y':'roi'}
) 
Figure 10.2. You can tell that Test and Train are almost identical. Costs decrease steadily at a high rate and profit at a lower rate, while ROI increases steadily. However, some differences exist, such as ROI, which becomes a bit higher eventually, and although viable thresholds start at the same point, Train does end at a different threshold. It turns out the model can turn a profit, so despite the appearance of the plot in *Figure 10.1*, the model is far from useless:
```

![图表，折线图描述自动生成](img/B18406_10_02.png)

图 10.2：测试和训练数据集在阈值下基础模型的利润、成本和投资回报率比较

训练集和测试集的 RMSEs 差异是真实的。模型没有过拟合。主要原因是我们通过将`max_depth`值设置为`4`使用了相对较浅的树。我们可以通过计算有多少特征的`feature_importances_`值超过 0 来轻易地看到使用浅树的效果：

```py
reg_mdls['rf_4_all']['total_feat'] =\
    reg_mdls['rf_4_all']['fitted'].feature_importances_.shape[0] reg_mdls['rf_4_all']['num_feat'] = sum(
    reg_mdls['rf_4_all']['fitted'].feature_importances_ > 0
)
print(reg_mdls['rf_4_all']['num_feat']) 
```

之前的代码输出`160`。换句话说，只有 160 个在 435 个中使用了——在这样的浅树中只能容纳这么多的特征！自然地，这会导致降低过度拟合，但与此同时，在具有杂质度量的特征与随机选择特征之间的选择并不一定是最佳选择。

## 在不同的最大深度下训练基础模型

那么，如果我们使树更深会发生什么？让我们重复之前为浅层模型所做的所有步骤，但这次的最大深度在 5 到 12 之间：

```py
for depth in tqdm(range(5, 13)):
mdlname = 'rf_'+str(depth)+'_all'
stime = timeit.default_timer()
reg_mdl = xgb.XGBRFRegressor(
    max_depth=depth,
    n_estimators=200,
    seed=rand
)
fitted_mdl = reg_mdl.fit(X_train, y_train)
etime = timeit.default_timer()
reg_mdls[mdlname] = mldatasets.evaluate_reg_mdl(
    fitted_mdl,
    X_train,
    X_test,
    y_train,
    y_test,
    plot_regplot=False,
    show_summary=False,
    ret_eval_dict=True
)
reg_mdls[mdlname]['speed'] = (etime - stime)/baseline_time
reg_mdls[mdlname]['depth'] = depth
reg_mdls[mdlname]['fs'] = 'all'
profits_test = mldatasets.profits_by_thresh(
    y_test,
    reg_mdls[mdlname]['preds_test'],
    threshs,
    var_costs=var_cost,
    min_profit=test_min_profit
)
profits_train = mldatasets.profits_by_thresh(
    y_train,
    reg_mdls[mdlname]['preds_train'],
    threshs,
    var_costs=var_cost,
    min_profit=train_min_profit
)
reg_mdls[mdlname]['max_profit_train'] = profits_train.profit.max()
reg_mdls[mdlname]['max_profit_test'] = profits_test.profit.max()
reg_mdls[mdlname]['max_roi'] = profits_test.roi.max()
reg_mdls[mdlname]['min_costs'] = profits_test.costs.min()
reg_mdls[mdlname]['profits_train'] = profits_train
reg_mdls[mdlname]['profits_test'] = profits_test
reg_mdls[mdlname]['total_feat'] =\
reg_mdls[mdlname]['fitted'].feature_importances_.shape[0]
reg_mdls[mdlname]['num_feat'] = sum(
    reg_mdls[mdlname]['fitted'].feature_importances_ > 0) 
```

现在，让我们像之前使用 `compare_df_plots` 一样，绘制“最深”模型（最大深度为 12）的利润 DataFrame 的细节，生成*图 10.3*：

![图表，折线图描述自动生成](img/B18406_10_03.png)

*图 10.3*：比较测试和训练数据集对于“深”基础模型在阈值下的利润、成本和 ROI

看看这次*图 10.3*中不同的**测试**和**训练**。**测试**达到约 15,000 的最大值，而**训练**超过 20,000。**训练**的成本大幅下降，使得投资回报率比**测试**高几个数量级。此外，阈值范围也大不相同。你可能会问，这为什么会成为问题？如果我们必须猜测使用什么阈值来选择在下一封邮件中要针对的目标，**训练**的最佳阈值高于**测试**——这意味着使用过度拟合的模型，我们可能会错过目标，并在未见过的数据上表现不佳。

接下来，让我们将我们的模型字典（`reg_mdls`）转换为 DataFrame，并从中提取一些细节。然后，我们可以按深度排序它，格式化它，用颜色编码它，并输出它：

```py
def display_mdl_metrics(reg_mdls, sort_by='depth', max_depth=None):
    reg_metrics_df = pd.DataFrame.from_dict( reg_mdls, 'index')\
                        [['depth', 'fs', 'rmse_train', 'rmse_test',\
                          'max_profit_train',\
                          'max_profit_test', 'max_roi',\
                          'min_costs', 'speed', 'num_feat']]
    pd.set_option('precision', 2) 
    html = reg_metrics_df.sort_values(
        by=sort_by, ascending=False).style.\
        format({'max_profit_train':'${0:,.0f}',\
        'max_profit_test':'${0:,.0f}', 'min_costs':'${0:,.0f}'}).\
        background_gradient(cmap='plasma', low=0.3, high=1,
                            subset=['rmse_train', 'rmse_test']).\
        background_gradient(cmap='viridis', low=1, high=0.3,
                            subset=[
                                'max_profit_train', 'max_profit_test'
                                ]
                            )
    return html
display_mdl_metrics(reg_mdls) 
display_mdl_metrics function to output the DataFrame shown in *Figure 10.4*. Something that should be immediately visible is how RMSE train and RMSE test are inverses. One decreases dramatically, and another increases slightly as the depth increases. The same can be said for profit. ROI tends to increase with depth and training speed and the number of features used as well:
```

![表格描述自动生成](img/B18406_10_04.png)

*图 10.4*：比较所有基础 RF 模型在不同深度下的指标

我们可能会倾向于使用具有最高盈利能力的 `rf_11_all`，但使用它是有风险的！一个常见的误解是，黑盒模型可以有效地消除任何数量的无关特征。虽然它们通常能够找到有价值的东西并充分利用它，但过多的特征可能会因为过度拟合训练数据集中的噪声而降低它们的可靠性。幸运的是，存在一个甜蜜点，你可以以最小的过度拟合达到高盈利能力，但为了达到这一点，我们首先必须减少特征的数量！

# 检查基于过滤的特征选择方法

**基于过滤的方法**独立地从数据集中选择特征，而不使用任何机器学习。这些方法仅依赖于变量的特征，并且相对有效、计算成本低、执行速度快。因此，作为特征选择方法的低垂之果，它们通常是任何特征选择流程的第一步。

基于过滤的方法可以分为：

+   **单变量**：它们独立于特征空间，一次评估和评级一个特征。单变量方法可能存在的问题是，由于它们没有考虑特征之间的关系，可能会过滤掉太多信息。

+   **多元性**：这些方法考虑整个特征空间以及特征之间的相互作用。

总体而言，对于移除过时、冗余、常数、重复和不相关的特征，过滤方法非常有效。然而，由于它们没有考虑到只有机器学习模型才能发现的复杂、非线性、非单调的相关性和相互作用，当这些关系在数据中突出时，它们并不有效。

我们将回顾三种基于过滤的方法：

+   基础

+   相关性

+   排序

我们将在各自的章节中进一步解释它们。

## 基础过滤方法

在数据准备阶段，我们采用**基本过滤方法**，特别是在任何建模之前的数据清洗阶段。这样做的原因是，做出可能对模型产生不利影响的特征选择决策的风险很低。这些方法涉及常识性操作，例如移除不携带信息或重复信息的特征。

### 带有方差阈值的常数特征

**常数特征**在训练数据集中不发生变化，因此不携带任何信息，模型无法从中学习。我们可以使用一个名为`VarianceThreshold`的单变量方法，它移除低方差特征。我们将使用零作为阈值，因为我们只想过滤掉具有**零方差**的特征——换句话说，就是常数特征。它仅适用于数值特征，因此我们必须首先确定哪些特征是数值的，哪些是分类的。一旦我们将方法拟合到数值列上，`get_support()`返回的不是常数特征的列表，我们可以使用集合代数来返回仅包含常数特征的集合（`num_const_cols`）：

```py
num_cols_l = X_train.select_dtypes([np.number]).columns
cat_cols_l = X_train.select_dtypes([np.bool, np.object]).columns
num_const = VarianceThreshold(threshold=0)
num_const.fit(X_train[num_cols_l])
num_const_cols = list(
    set(X_train[num_cols_l].columns) -
    set(num_cols_l[num_const.get_support()])
) 
nunique() function on categorical features. It will return a pandas series, and then a lambda function can filter out only those with one unique value. Then, .index.tolist() returns the name of the features as a list. Now, we just join both lists of constant features, and voilà! We have all constants (all_const_cols). We can print them; there should be three:
```

```py
cat_const_cols = X_train[cat_cols_l].nunique()[lambda x:\
                                               x<2].index.tolist()
all_const_cols = num_const_cols + cat_const_cols
print(all_const_cols) 
```

在大多数情况下，仅移除常数特征是不够的。一个冗余特征可能几乎是常数或**准常数**。

### 带有 value_counts 的准常数特征

**准常数特征**几乎都是相同的值。与常数过滤不同，使用方差阈值不会起作用，因为高方差和准常数性不是互斥的。相反，我们将迭代所有特征并获取 `value_counts()`，它返回每个值的行数。然后，将这些计数除以总行数以获得百分比，并按最高百分比排序。如果最高值高于预先设定的阈值（`thresh`），则将其追加到准常数列列表（`quasi_const_cols`）中。请注意，选择此阈值必须非常谨慎，并且需要对问题有深入的理解。例如，在这种情况下，我们知道这是不平衡的，因为只有 5% 的人捐赠，其中大多数人捐赠的金额很低，所以即使是特征的一小部分也可能产生影响，这就是为什么我们的阈值如此之高，达到 99.9%：

```py
thresh = 0.999
quasi_const_cols = []
num_rows = X_train.shape[0]
for col in tqdm(X_train.columns):
    top_val = (
        X_train[col].value_counts() / num_rows
        ).sort_values(ascending=False).values[0]
    if top_val >= thresh:
        quasi_const_cols.append(col)
print(quasi_const_cols) 
```

前面的代码应该已经打印出五个特征，其中包括之前获得的三个。接下来，我们将处理另一种形式的不相关特征：重复项！

### 重复特征

通常，当我们讨论数据中的重复项时，我们首先想到的是重复的行，但**重复的列**也是问题所在。我们可以像查找重复行一样找到它们，使用 `pandas duplicated()` 函数，但首先需要将 DataFrame 转置，反转列和行：

```py
X_train_transposed = X_train.T
dup_cols = X_train_transposed[
    X_train_transposed.duplicated()].index.tolist()
print(dup_cols) 
```

前面的代码片段输出了一个包含两个重复列的列表。

### 移除不必要的特征

与其他特征选择方法不同，您应该用模型测试这些方法，您可以直接通过移除您认为无用的特征来应用基于基本过滤的特征选择方法。但以防万一，制作原始数据的副本是一个好习惯。请注意，我们不将常数列（`all_constant_cols`）包括在我们打算删除的列（`drop_cols`）中，因为准常数列已经包含它们：

```py
X_train_orig = X_train.copy()
X_test_orig = X_test.copy()
drop_cols = quasi_const_cols + dup_cols
X_train.drop(labels=drop_cols, axis=1, inplace=True)
X_test.drop(labels=drop_cols, axis=1, inplace=True) 
```

接下来，我们将探索剩余特征上的多变量过滤方法。

## 基于相关性的过滤方法

**基于相关性的过滤方法**量化两个特征之间关系的强度。这对于特征选择很有用，因为我们可能想要过滤掉高度相关的特征或那些与其他特征完全不相关的特征。无论如何，它是一种多变量特征选择方法——更确切地说，是双变量特征选择方法。

但首先，我们应该选择一个相关性方法：

+   **皮尔逊相关系数**：衡量两个特征之间的线性相关性。它输出一个介于 -1（负相关）和 1（正相关）之间的系数，0 表示没有线性相关性。与线性回归类似，它假设线性、正态性和同方差性——也就是说，线性回归线周围的误差项在所有值中大小相似。

+   **斯皮尔曼秩相关系数**：衡量两个特征单调性的强度，无论它们是否线性相关。单调性是指一个特征增加时，另一个特征持续增加或减少的程度。它在-1 和 1 之间衡量，0 表示没有单调相关性。它不做分布假设，可以与连续和离散特征一起使用。然而，它的弱点在于非单调关系。

+   **肯德尔 tau 相关系数**：衡量特征之间的序数关联——也就是说，它计算有序数字列表之间的相似性。它也介于-1 和 1 之间，但分别代表低和高。对于离散特征来说，它很有用。

数据集是连续和离散的混合，我们不能对其做出任何线性假设，因此`spearman`是正确的选择。尽管如此，所有三个都可以与`pandas`的`corr`函数一起使用：

```py
corrs = X_train.corr(method='spearman')
print(corrs.shape) 
```

前面的代码应该输出相关矩阵的形状，即`(428, 428)`。这个维度是有意义的，因为还剩下 428 个特征，每个特征都与 428 个特征有关，包括它自己。

我们现在可以在相关矩阵（`corrs`）中寻找要删除的特征。请注意，为了做到这一点，我们必须建立阈值。例如，我们可以说一个高度相关的特征具有超过 0.99 的绝对值系数，而对于一个不相关的特征则小于 0.15。有了这些阈值，我们可以找到只与一个特征相关并且与多个特征高度相关的特征。为什么是一个特征？因为在相关矩阵的对角线总是 1，因为一个特征总是与自己完美相关。以下代码中的`lambda`函数确保我们考虑到这一点：

```py
extcorr_cols = (abs(corrs) > 0.99).sum(axis=1)[lambda x: x>1]\
.index.tolist()
print(extcorr_cols)
uncorr_cols = (abs(corrs) > 0.15).sum(axis=1)[lambda x: x==1]\
.index.tolist()
print(uncorr_cols) 
```

前面的代码以如下方式输出两个列表：

```py
['MAJOR', 'HHAGE1', 'HHAGE3', 'HHN3', 'HHP1', 'HV1', 'HV2', 'MDMAUD_R', 'MDMAUD_F', 'MDMAUD_A']
['TCODE', 'MAILCODE', 'NOEXCH', 'CHILD03', 'CHILD07', 'CHILD12', 'CHILD18', 'HC15', 'MAXADATE'] 
```

第一个列表包含与除自身以外的其他特征高度相关的特征。虽然了解这一点很有用，但你不应在没有理解它们与哪些特征以及如何相关，以及与目标相关的情况下从该列表中删除特征。然后，只有在发现冗余的情况下，确保只删除其中一个。第二个列表包含与除自身以外的任何其他特征都不相关的特征，鉴于特征的数量众多，这在当前情况下是可疑的。话虽如此，我们也应该逐个检查它们，特别是要衡量它们与目标的相关性，看看它们是否冗余。然而，我们将冒险排除不相关的特征，创建一个特征子集（`corr_cols`）：

```py
corr_cols = X_train.columns[
    ~X_train.columns.isin(uncorr_cols)
].tolist()
print(len(corr_cols)) 
```

前面的代码应该输出`419`。现在让我们只使用这些特征来拟合 RF 模型。鉴于仍有超过 400 个特征，我们将使用`max_depth`值为`11`。除了这一点和一个不同的模型名称（`mdlname`）之外，代码与之前相同：

```py
mdlname = 'rf_11_f-corr'
stime = timeit.default_timer()
reg_mdl = xgb.XGBRFRegressor(
    max_depth=11,
    n_estimators=200,
    seed=rand
)
fitted_mdl = reg_mdl.fit(X_train[corr_cols], y_train)
reg_mdls[mdlname]['num_feat'] = sum(
    reg_mdls[mdlname]['fitted'].feature_importances_ > 0
) 
```

在比较前面模型的输出结果之前，让我们了解一下排名滤波方法。

## 基于排序过滤的方法

**基于排序过滤的方法**基于统计单变量排序测试，这些测试评估特征与目标之间的依赖强度。这些是一些最受欢迎的方法：

+   **方差分析 F 检验**（**ANOVA**）F 检验衡量特征与目标之间的线性依赖性。正如其名所示，它是通过分解方差来做到这一点的。它做出了与线性回归类似的假设，例如正态性、独立性和同方差性。在 scikit-learn 中，您可以使用 `f_regression` 和 `f_classification` 分别对回归和分类进行排序，以 F 检验产生的 F 分数来排序特征。

+   **卡方检验独立性**：这个测试衡量非负分类变量与二元目标之间的关联性，因此它只适用于分类问题。在 scikit-learn 中，您可以使用 `chi2`。

+   **互信息**（**MI**）：与前面两种方法不同，这种方法是从信息理论而不是经典统计假设检验中推导出来的。虽然名称不同，但这个概念我们在本书中已经讨论过，称为**库尔巴克-莱布勒**（**KL**）**散度**，因为它是对特征 *X* 和目标 *Y* 的 KL。scikit-learn 中的 Python 实现使用了一个数值稳定的对称 KL 衍生品，称为**Jensen-Shannon**（**JS**）散度，并利用 k-最近邻来计算距离。可以使用 `mutual_info_regression` 和 `mutual_info_classif` 分别对回归和分类进行特征排序。

在提到的三种选项中，最适合这个数据集的是 MI，因为我们不能假设特征之间存在线性关系，而且其中大部分也不是分类数据。我们可以尝试使用阈值为 $0.68 的分类，这至少可以覆盖发送邮件的成本。为此，我们必须首先使用该阈值创建一个二元分类目标（`y_train_class`）：

```py
y_train_class = np.where(y_train > 0.68, 1, 0) 
```

接下来，我们可以使用 `SelectKBest` 根据互信息分类（**MIC**）获取前 160 个特征。然后我们使用 `get_support()` 获取一个布尔向量（或掩码），它告诉我们哪些特征在前 160 个中，并使用这个掩码对特征列表进行子集化：

```py
mic_selection = SelectKBest(
    mutual_info_classif, k=160).fit(X_train, y_train_class)
mic_cols = X_train.columns[mic_selection.get_support()].tolist()
print(len(mic_cols)) 
```

前面的代码应该确认 `mic_cols` 列表中确实有 160 个特征。顺便说一下，这是一个任意数字。理想情况下，我们可以测试分类目标的不同阈值和 MI 的 *k* 值，寻找在最小过拟合的同时实现最高利润提升的模型。接下来，我们将使用与之前相同的 MIC 特征拟合 RF 模型。这次，我们将使用最大深度为 `5`，因为特征数量显著减少：

```py
mdlname = 'rf_5_f-mic'
stime = timeit.default_timer()
reg_mdl = xgb.XGBRFRegressor(max_depth=5, n_estimators=200, seed=rand)
fitted_mdl = reg_mdl.fit(X_train[mic_cols], y_train)
reg_mdls[mdlname]['num_feat'] = sum(
    reg_mdls[mdlname]['fitted'].feature_importances_ > 0
) 
```

现在，让我们像在 *图 10.3* 中所做的那样绘制 **测试** 和 **训练** 的利润，但这次是针对 MIC 模型。它将产生 *图 10.5* 中所示的内容：

![图表，折线图  自动生成的描述](img/B18406_10_05.png)

图 10.5：具有 MIC 特征的模型在阈值之间的利润、成本和 ROI 测试和训练数据集的比较

在*图 10.5*中，你可以看出**测试**和**训练**之间存在相当大的差异，但相似之处表明过拟合最小。例如，**训练**的最高盈利性可以在 0.66 和 0.75 之间找到，而**测试**主要在 0.66 和 0.7 之间，之后逐渐下降。

尽管我们已经视觉检查了 MIC 模型，但查看原始指标也是一种令人放心的方式。接下来，我们将使用一致的指标比较我们迄今为止训练的所有模型。

## 比较基于滤波的方法

我们已经将指标保存到一个字典（`reg_mdls`）中，我们很容易将其转换为 DataFrame，并像之前那样输出，但这次我们按`max_profit_test`排序：

```py
display_mdl_metrics(reg_mdls, 'max_profit_test') 
Figure 10.6. It is evident that the filter MIC model is the least overfitted of all. It ranked higher than more complex models with more features and took less time to train than any model. Its speed is an advantage for hyperparameter tuning. What if we wanted to find the best classification target thresholds or MIC *k*? We won’t do this now, but we would likely get a better model if we ran every combination, but it would take time to do and even more with more features:
```

![应用，表格，自动生成中等置信度的描述](img/B18406_10_06.png)

图 10.6：比较所有基础模型和基于滤波的特征选择模型的指标

在*图 10.6*中，我们可以看出，与具有更多特征和相同`max_depth`量的模型（`rf_11_all`）相比，相关滤波模型（`rf_11_f-corr`）的表现更差，这表明我们可能移除了一个重要的特征。正如该部分所警告的，盲目设置阈值并移除其上所有内容的问题在于你可能会无意中移除有用的东西。并非所有高度相关和无关的特征都是无用的，因此需要进一步检查。接下来，我们将探索一些嵌入方法，当与交叉验证结合使用时，需要更少的监督。

# 探索嵌入特征选择方法

**嵌入方法**存在于模型本身中，通过训练过程中自然选择特征。你可以利用具有这些特性的任何模型的内在属性来捕获所选特征：

+   **基于树的模型**：例如，我们多次使用以下代码来计算 RF 模型使用的特征数量，这是学习过程中自然发生特征选择的证据：

    ```py
    sum(reg_mdls[mdlname]['fitted'].feature_importances_ > 0) 
    ```

    XGBoost 的 RF 默认使用增益，这是在所有使用该特征进行特征重要性计算的分割中平均错误减少。我们可以将阈值提高到 0 以上，根据它们的相对贡献选择更少的特征。然而，通过限制树的深度，我们迫使模型选择更少的特征。

+   **具有系数的正则化模型**：我们将在*第十二章*，*单调约束和模型调优以提高可解释性*中进一步研究这个问题，但许多模型类可以采用基于惩罚的正则化，如 L1、L2 和弹性网络。然而，并非所有这些模型都具有可以提取以确定哪些特征被惩罚的内在参数，如系数。

本节将仅涵盖正则化模型，因为我们已经使用了一个基于树的模型。最好利用不同的模型类别来获得对哪些特征最重要的不同视角。

我们在*第三章*，*解释挑战*中介绍了一些这些模型，但这些都是一些结合基于惩罚的正则化和输出特征特定系数的模型类别：

+   **最小绝对收缩和选择算子**（**LASSO**）：因为它在损失函数中使用 L1 惩罚，所以 LASSO 可以将系数设置为 0。

+   **最小角度回归**（**LARS**）：类似于 LASSO，但基于向量，更适合高维数据。它也对等相关的特征更加公平。

+   **岭回归**：在损失函数中使用 L2 惩罚，因此只能将不相关的系数缩小到接近 0，但不能缩小到 0。

+   **弹性网络回归**：使用 L1 和 L2 范数的混合作为惩罚。

+   **逻辑回归**：根据求解器，它可以处理 L1、L2 或弹性网络惩罚。

之前提到的模型也有一些变体，例如**LASSO LARS**，它使用 LARS 算法进行 LASSO 拟合，或者甚至是**LASSO LARS IC**，它与前者相同，但在模型部分使用 AIC 或 BIC 准则：

+   **赤池信息准则**（**AIC**）：基于信息理论的一种相对拟合优度度量

+   **贝叶斯信息准则**（**BIC**）：与 AIC 具有相似的公式，但具有不同的惩罚项

好的，现在让我们使用`SelectFromModel`从 LASSO 模型中提取顶级特征。我们将使用`LassoCV`，因为它可以自动进行交叉验证以找到最优的惩罚强度。一旦拟合，我们就可以使用`get_support()`获取特征掩码。然后我们可以打印特征数量和特征列表：

```py
lasso_selection = SelectFromModel(
    LassoCV(n_jobs=-1, random_state=rand)
)
lasso_selection.fit(X_train, y_train)
lasso_cols = X_train.columns[lasso_selection.get_support()].tolist()
print(len(lasso_cols))
print(lasso_cols) 
```

上一段代码输出以下内容：

```py
7
['ODATEDW', 'TCODE', 'POP901', 'POP902', 'HV2', 'RAMNTALL', 'MAXRDATE'] 
```

现在，让我们尝试使用`LassoLarsCV`进行相同的操作：

```py
llars_selection = SelectFromModel(LassoLarsCV(n_jobs=-1))
llars_selection.fit(X_train, y_train)
llars_cols = X_train.columns[llars_selection.get_support()].tolist()
print(len(llars_cols))
print(llars_cols) 
```

上一段代码生成以下输出：

```py
8
['RECPGVG', 'MDMAUD', 'HVP3', 'RAMNTALL', 'LASTGIFT', 'AVGGIFT', 'MDMAUD_A', 'DOMAIN_SOCIALCLS'] 
```

LASSO 将除七个特征外的所有系数缩小到 0，而 LASSO LARS 也将八个系数缩小到 0。然而，请注意这两个列表之间没有重叠！好的，那么让我们尝试将 AIC 模型选择与 LASSO LARS 结合使用`LassoLarsIC`：

```py
llarsic_selection = SelectFromModel(LassoLarsIC(criterion='aic'))
llarsic_selection.fit(X_train, y_train)
llarsic_cols = X_train.columns[
    llarsic_selection.get_support()
].tolist()
print(len(llarsic_cols))
print(llarsic_cols) 
```

上一段代码生成以下输出：

```py
111
['TCODE', 'STATE', 'MAILCODE', 'RECINHSE', 'RECP3', 'RECPGVG', 'RECSWEEP',..., 'DOMAIN_URBANICITY', 'DOMAIN_SOCIALCLS', 'ZIP_LON'] 
```

这是一种相同的算法，但采用了不同的方法来选择正则化参数的值。注意这种不那么保守的方法将特征数量扩展到 111 个。到目前为止，我们使用的方法都具有 L1 范数。让我们尝试一个使用 L2 的——更具体地说，是 L2 惩罚逻辑回归。我们做的是之前所做的，但这次，我们使用二元分类目标（`y_train_class`）进行拟合：

```py
log_selection = SelectFromModel(
    LogisticRegression(
        C=0.0001,
        solver='sag',
        penalty='l2',
        n_jobs=-1,
        random_state=rand
    )
)
log_selection.fit(X_train, y_train_class)
log_cols = X_train.columns[log_selection.get_support()].tolist()
print(len(log_cols))
print(log_cols) 
```

上一段代码生成以下输出：

```py
87
['ODATEDW', 'TCODE', 'STATE', 'POP901', 'POP902', 'POP903', 'ETH1', 'ETH2', 'ETH5', 'CHIL1', 'HHN2',..., 'AMT_7', 'ZIP_LON'] 
```

现在我们有几个特征子集要测试，我们可以将它们的名称放入一个列表（`fsnames`）中，将特征子集列表放入另一个列表（`fscols`）中：

```py
fsnames = ['e-lasso', 'e-llars', 'e-llarsic', 'e-logl2']
fscols = [lasso_cols, llars_cols, llarsic_cols, log_cols] 
```

然后，我们可以遍历所有列表名称，并在每次迭代中增加`max_depth`，就像我们之前做的那样来拟合和评估我们的`XGBRFRegressor`模型：

```py
def train_mdls_with_fs(reg_mdls, fsnames, fscols, depths):
    for i, fsname in tqdm(enumerate(fsnames), total=len(fsnames)):
       depth = depths[i]
       cols = fscols[i]
       mdlname = 'rf_'+str(depth)+'_'+fsname
       stime = timeit.default_timer()
       reg_mdl = xgb.XGBRFRegressor(
           max_depth=depth, n_estimators=200, seed=rand
       )
       fitted_mdl = reg_mdl.fit(X_train[cols], y_train)
       reg_mdls[mdlname]['num_feat'] = sum(
           reg_mdls[mdlname]['fitted'].feature_importances_ > 0
       )
train_mdls_with_fs(reg_mdls, fsnames, fscols, [3, 4, 5, 6]) 
```

现在，让我们看看我们的嵌入式特征选择模型与过滤模型相比的表现如何。我们将重新运行之前运行的代码，输出*图 10.6*中显示的内容。这次，我们将得到*图 10.7*中显示的内容：

![表描述自动生成，置信度中等](img/B18406_10_07.png)

图 10.7：比较所有基础模型和基于过滤和嵌入式特征选择模型的指标

根据图 10.7，我们尝试的四种嵌入式方法中有三种产生了具有最低测试 RMSE（`rf_5_e-llarsic`、`rf_e-lasso`和`rf_4_e-llars`）的模型。它们也都比其他模型训练得快得多，并且比任何同等复杂性的模型都更有利可图。其中之一（`rf_5_e-llarsic`）甚至非常有利可图。与具有相似测试盈利能力的`rf_9_all`进行比较，看看性能如何从训练数据中偏离。

# 发现包装、混合和高级特征选择方法

到目前为止研究的特征选择方法在计算上成本较低，因为它们不需要模型拟合或拟合更简单的白盒模型。在本节中，我们将了解其他更全面的方法，这些方法具有许多可能的调整选项。这里包括的方法类别如下：

+   **包装**：通过使用测量指标改进的搜索策略来拟合机器学习模型，彻底搜索最佳特征子集。

+   **混合**：一种结合嵌入式和过滤方法以及包装方法的方法。

+   **高级**：一种不属于之前讨论的任何类别的的方法。例如包括降维、模型无关特征重要性和**遗传算法**（**GAs**）。

现在，让我们开始包装方法吧！

## 包装方法

包装方法背后的概念相当简单：评估特征的不同子集在机器学习模型上的表现，并选择在预定的目标函数上实现最佳得分的那个。这里变化的是搜索策略：

+   **顺序正向选择**（**SFS**）：这种方法开始时没有特征，然后每次添加一个。

+   **顺序正向浮点选择**（**SFFS**）：与之前相同，除了每次添加一个特征时，它可以移除一个特征，只要目标函数增加。

+   **顺序向后选择**（**SBS**）：这个过程从所有特征都存在开始，每次消除一个特征。

+   **顺序浮点向后选择**（**SFBS**）：与之前相同，除了每次移除一个特征时，它还可以添加一个特征，只要目标函数增加。

+   **穷举特征选择**（**EFS**）：这种方法寻求所有可能的特征组合。

+   **双向搜索**（**BDS**）：这个方法同时允许向前和向后进行函数选择，以获得一个独特的解决方案。

这些方法是贪婪算法，因为它们逐个解决问题，根据它们的即时利益选择部分。尽管它们可能达到全局最大值，但它们采取的方法更适合寻找局部最大值。根据特征的数量，它们可能过于计算密集，以至于不实用，特别是 EFS，它呈指数增长。另一个重要的区别是，向前方法随着特征的添加而提高准确性，而向后方法则随着特征的移除而监控准确性下降。为了缩短搜索时间，我们将做两件事：

1.  我们从其他方法共同选出的特征开始搜索，以拥有更小的特征空间进行选择。为此，我们将来自几种方法的特征列表合并成一个单一的`top_cols`列表：

    ```py
    top_cols = list(set(mic_cols).union(set(llarsic_cols)\
    ).union(set(log_cols)))
    len(top_cols) 
    ```

1.  样本我们的数据集，以便机器学习模型加速。我们可以使用`np.random.choice`进行随机选择行索引，而不进行替换：

    ```py
    sample_size = 0.1
    sample_train_idx = np.random.choice(
        X_train.shape[0],
        math.ceil(X_train.shape[0]*sample_size),
        replace=False
    )
    sample_test_idx = np.random.choice(
        X_test.shape[0],
        math.ceil(X_test.shape[0]*sample_size),
        replace=False
    ) 
    ```

在所提出的包装方法中，我们只执行 SFS，因为它们非常耗时。然而，对于更小的数据集，你可以尝试其他选项，这些选项`mlextend`库也支持。

### 顺序前向选择（SFS）

包装方法的第一参数是一个未拟合的估计器（一个模型）。在`SequentialFeatureSelector`中，我们放置了一个`LinearDiscriminantAnalysis`模型。其他参数包括方向（`forward=true`），是否浮动（`floating=False`），这意味着它可能会撤销之前对特征的排除或包含，我们希望选择的特征数量（`k_features=27`），交叉验证的数量（`cv=3`），以及要使用的损失函数（`scoring=f1`）。一些推荐的可选参数包括详细程度（`verbose=2`）和并行运行的工作数量（`n_jobs=-1`）。由于它可能需要一段时间，我们肯定希望它输出一些内容，并尽可能多地使用处理器：

```py
sfs_lda = SequentialFeatureSelector(
    LinearDiscriminantAnalysis(n_components=1),
    forward=True,
    floating=False,
    k_features=100,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1
)
sfs_lda = sfs_lda.fit(X_train.iloc[sample_train_idx][top_cols],\
                      y_train_class[sample_train_idx])
sfs_lda_cols = X_train.columns[list(sfs_lda.k_feature_idx_)].tolist() 
```

一旦我们拟合了 SFS，它将返回使用`k_feature_idx_`选定的特征的索引，我们可以使用这些索引来子集列并获取特征名称列表。

## 混合方法

从 435 个特征开始，仅 27 个特征子集的组合就有超过 10⁴²种！所以，你可以看到在如此大的特征空间中 EFS 是如何不切实际的。因此，除了在整个数据集上使用 EFS 之外，包装方法不可避免地会采取一些捷径来选择特征。无论你是向前、向后还是两者都进行，只要你不评估每个特征的组合，你就很容易错过最佳选择。

然而，我们可以利用包装方法的更严格、更全面的搜索方法，同时结合筛选和嵌入方法的效率。这种方法的结果是**混合方法**。例如，你可以使用筛选或嵌入方法仅提取前 10 个特征，并在这些特征上仅执行 EFS 或 SBS。

### 递归特征消除（RFE）

另一种更常见的方法是 SBS，但它不是仅基于改进一个指标来删除特征，而是使用模型的内在参数来对特征进行排序，并仅删除排名最低的特征。这种方法被称为**递归特征消除**（**RFE**），它是嵌入和包装方法之间的混合。我们只能使用具有`feature_importances_`或系数（`coef_`）的模型，因为这是该方法知道要删除哪些特征的方式。具有这些属性的 scikit-learn 模型类别被归类为`linear_model`、`tree`和`ensemble`。此外，XGBoost、LightGBM 和 CatBoost 的 scikit-learn 兼容版本也具有`feature_importances_`。

我们将使用交叉验证版本的递归特征消除（RFE），因为它更可靠。`RFECV`首先采用估计器（`LinearDiscriminantAnalysis`）。然后我们可以定义`step`，它设置每次迭代应删除多少特征，交叉验证的次数（`cv`），以及用于评估的指标（`scoring`）。最后，建议设置详细程度（`verbose=2`）并尽可能利用更多处理器（`n_jobs=-1`）。为了加快速度，我们将再次使用样本进行训练，并从`top_cols`的 267 开始：

```py
rfe_lda = RFECV(
    LinearDiscriminantAnalysis(n_components=1),
    step=2, cv=3, scoring='f1', verbose=2, n_jobs=-1
)
rfe_lda.fit(
    X_train.iloc[sample_train_idx][top_cols],
    y_train_class[sample_train_idx]
)
rfe_lda_cols = np.array(top_cols)[rfe_lda.support_].tolist() 
```

接下来，我们将尝试与主要三个特征选择类别（筛选、嵌入和包装）无关的不同方法。

## 高级方法

许多方法可以归类为高级特征选择方法，包括以下子类别：

+   **模型无关特征重要性**：任何在*第四章*、*全局模型无关解释方法*中提到的特征重要性方法都可以用于获取模型的特征选择中的顶级特征。

+   **遗传算法**：这是一种包装方法，因为它“包装”了一个评估多个特征子集预测性能的模型。然而，与我们所检查的包装方法不同，它并不总是做出最局部最优的选择。它更适合与大型特征空间一起工作。它被称为遗传算法，因为它受到了生物学的启发——自然选择，特别是。

+   **降维**：一些降维方法，如**主成分分析**（**PCA**），可以在特征基础上返回解释方差。对于其他方法，如因子分析，它可以从其他输出中推导出来。解释方差可以用于对特征进行排序。

+   **自动编码器**：我们不会深入探讨这一点，但深度学习可以利用自动编码器进行特征选择。这种方法在 Google Scholar 上有许多变体，但在工业界并不广泛采用。

在本节中，我们将简要介绍前两种方法，以便您了解它们如何实现。让我们直接进入正题！

### 模型无关特征重要性

在这本书的整个过程中，我们使用的一个流行的模型无关特征重要性方法是 SHAP，它有许多属性使其比其他方法更可靠。在下面的代码中，我们可以使用`TreeExplainer`提取我们最佳模型的`shap_values`：

```py
fitted_rf_mdl = reg_mdls['rf_11_all']['fitted']
shap_rf_explainer = shap.TreeExplainer(fitted_rf_mdl)
shap_rf_values = shap_rf_explainer.shap_values(
    X_test_orig.iloc[sample_test_idx]
)
shap_imps = pd.DataFrame(
    {'col':X_train_orig.columns, 'imp':np.abs(shap_rf_values).mean(0)}
).sort_values(by='imp',ascending=False)
shap_cols = shap_imps.head(120).col.tolist() 
```

然后，SHAP 值绝对值的平均值在第一维上为我们提供了每个特征的排名。我们将这个值放入一个 DataFrame 中，并按我们为 PCA 所做的方式对其进行排序。最后，也将前 120 个放入一个列表（`shap_cols`）。

### 遗传算法

算法遗传学（GAs）是一种受自然选择启发的随机全局优化技术，它像包装方法一样包装一个模型。然而，它们不是基于一步一步的序列。GAs 没有迭代，但有代，包括染色体的种群。每个染色体是特征空间的二进制表示，其中 1 表示选择一个特征，0 表示不选择。每一代都是通过以下操作产生的：

+   **选择**：就像自然选择一样，这部分是随机的（探索）和部分是基于已经有效的东西（利用）。有效的是其适应性。适应性是通过一个“scorer”来评估的，就像包装方法一样。适应性差的染色体被移除，而好的染色体则通过“交叉”繁殖。

+   **交叉**：随机地，一些好的位（或特征）从每个父代传递给子代。

+   **变异**：即使染色体已经证明有效，给定一个低的突变率，它偶尔也会突变或翻转其位之一，换句话说，特征。

我们将要使用的 Python 实现有许多选项。在这里我们不会解释所有这些选项，但如果您感兴趣，它们在代码中都有很好的文档说明。第一个属性是估计器。我们还可以定义交叉验证迭代次数（`cv=3`）和`scoring`来决定染色体是否适合。有一些重要的概率属性，例如突变位（`mutation_probability`）的概率和位交换（`crossover_probability`）的概率。在每一代中，`n_gen_no_change`提供了一种在代数没有改进时提前停止的手段，默认的`generations`是 40，但我们将使用 5。我们可以像任何模型一样拟合`GeneticSelectionCV`。这可能需要一些时间，因此最好定义详细程度并允许它使用所有处理能力。一旦完成，我们可以使用布尔掩码（`support_`）来子集特征：

```py
ga_rf = GAFeatureSelectionCV(
    RandomForestRegressor(random_state=rand, max_depth=3),
    cv=3,
    scoring='neg_root_mean_squared_error',
    crossover_probability=0.8,
    mutation_probability=0.1,
    generations=5, n_jobs=-1
)
ga_rf = ga_rf.fit(
    X_train.iloc[sample_train_idx][top_cols].values,
    y_train[sample_train_idx]
)
ga_rf_cols = np.array(top_cols)[ga_rf.best_features_].tolist() 
```

好的，现在我们已经在本节中介绍了各种包装、混合和高级特征选择方法，让我们一次性评估它们并比较结果。

## 评估所有特征选择模型

正如我们对待嵌入方法一样，我们可以将特征子集名称 (`fsnames`)、列表 (`fscols`) 和相应的 `depths` 放入列表中：

```py
fsnames = ['w-sfs-lda', 'h-rfe-lda', 'a-shap', 'a-ga-rf']
fscols = [sfs_lda_cols, rfe_lda_cols, shap_cols, ga_rf_cols]
depths = [5, 6, 5, 6] 
```

然后，我们可以使用我们创建的两个函数，首先遍历所有特征子集，用它们训练和评估一个模型。然后第二个函数输出评估结果，以 DataFrame 的形式包含先前训练的模型：

```py
train_mdls_with_fs(reg_mdls, fsnames, fscols, depths) 
display_mdl_metrics(reg_mdls, 'max_profit_test', max_depth=7) 
Figure 10.8:
```

![图形用户界面，应用程序描述自动生成](img/B18406_10_08.png)

图 10.8：比较所有特征选择模型的指标

*图 10.8* 展示了与包含所有特征相比，特征选择模型在相同深度下的盈利能力更强。此外，嵌入的 LASSO LARS 与 AIC (`e-llarsic`) 方法和 MIC (`f-mic`) 过滤方法在相同深度下优于所有包装、混合和高级方法。尽管如此，我们还是通过使用训练数据集的一个样本来阻碍了这些方法，这是加快过程所必需的。也许在其他情况下，它们会优于最顶尖的模型。然而，接下来的三种特征选择方法竞争力相当强：

+   基于 LDA 的 RFE：混合方法 (`h-rfe-lda`)

+   带有 L2 正则化的逻辑回归：嵌入方法 (`e-logl2`)

+   基于 RF 的 GAs：高级方法 (`a-ga-rf`)

在这本书中回顾的方法有很多变体，花费很多天去运行这些变体是有意义的。例如，也许 RFE 与 L1 正则化的逻辑回归或 GA 与支持向量机以及额外的突变会产生最佳模型。有如此多的不同可能性！然而，如果你被迫仅基于 *图 10.8* 中的利润来做出推荐，那么 111 特征的 `e-llarsic` 是最佳选择，但它也有比任何顶级模型更高的最低成本和更低的最高回报率。这是一个权衡。尽管它的测试 RMSE 值最高，但 160 特征的模型 (`f-mic`) 在最大利润训练和测试之间的差异相似，并且在最大回报率和最低成本方面超过了它。因此，这两个选项是合理的。但在做出最终决定之前，必须将不同阈值下的盈利能力进行比较，以评估每个模型在什么成本和回报率下可以做出最可靠的预测。

# 考虑特征工程

假设非营利组织选择了使用具有 LASSO LARS 与 AIC (`e-llarsic`) 选择特征的模型，但想评估你是否可以进一步改进它。现在你已经移除了可能只略微提高预测性能但主要增加噪声的 300 多个特征，你剩下的是更相关的特征。然而，你也知道，`e-llars` 选出的 8 个特征产生了与 111 个特征相同的 RMSE。这意味着虽然那些额外特征中有些东西可以提高盈利能力，但它并没有提高 RMSE。

从特征选择的角度来看，可以采取许多方法来解决这个问题。例如，检查`e-llarsic`和`e-llars`之间特征的交集和差异，并在这些特征上严格进行特征选择，以查看 RMSE 是否在任何组合中下降，同时保持或提高当前的盈利能力。然而，还有一种可能性，那就是特征工程。在这个阶段进行特征工程有几个重要的原因：

+   **使模型解释更容易理解**：例如，有时特征有一个不直观的尺度，或者尺度是直观的，但分布使得理解变得困难。只要对这些特征的转换不会降低模型性能，转换特征以更好地理解解释方法的输出是有价值的。随着你在更多工程化特征上训练模型，你会意识到什么有效以及为什么有效。这将帮助你理解模型，更重要的是，理解数据。

+   **对单个特征设置护栏**：有时，特征分布不均匀，模型倾向于在特征直方图的稀疏区域或存在重要异常值的地方过拟合。

+   **清理反直觉的交互**：一些模型发现的不合逻辑的交互，仅因为特征相关，但并非出于正确的原因而存在。它们可能是混淆变量，甚至可能是冗余的（例如我们在*第四章*，*全局模型无关解释方法*中找到的）。你可以决定设计一个交互特征或删除一个冗余的特征。

关于最后两个原因，我们将在*第十二章*，*单调约束和模型调优以实现可解释性*中更详细地研究特征工程策略。本节将重点介绍第一个原因，尤其是因为它是一个很好的起点，因为它将允许你更好地理解数据，直到你足够了解它，可以做出更转型的改变。

因此，我们剩下 111 个特征，但不知道它们如何与目标或彼此相关。我们首先应该做的是运行一个特征重要性方法。我们可以在`e-llarsic`模型上使用 SHAP 的`TreeExplainer`。`TreeExplainer`的一个优点是它可以计算 SHAP 交互值，`shap_interaction_values`。与`shap_values`输出一个`(N, 111)`维度的数组不同，其中*N*是观察数量，它将输出`(N, 111, 111)`。我们可以用它生成一个`summary_plot`图，该图对单个特征和交互进行排名。交互值唯一的区别是您使用`plot_type="compact_dot"`：

```py
winning_mdl = 'rf_5_e-llarsic'
fitted_rf_mdl = reg_mdls[winning_mdl]['fitted']
shap_rf_explainer = shap.TreeExplainer(fitted_rf_mdl)
shap_rf_interact_values = \
    shap_rf_explainer.shap_interaction_values(
        X_test.iloc[sample_test_idx][llarsic_cols]
    )
shap.summary_plot(
    shap_rf_interact_values,
    X_test.iloc[sample_test_idx][llarsic_cols],
    plot_type="compact_dot",
    sort=True
) 
Figure 10.9:
```

![图形用户界面，应用程序，表格  自动生成的描述](img/B18406_10_09.png)

图 10.9：SHAP 交互总结图

我们可以像阅读任何总结图一样阅读*图 10.9*，除了它包含了两次双变量交互——首先是一个特征，然后是另一个。例如，`MDMAUD_A* - CLUSTER`是从`MDMAUD_A`的角度来看该交互的交互 SHAP 值，因此特征值对应于该特征本身，但 SHAP 值是针对交互的。我们在这里可以达成一致的是，考虑到重要性值的规模和比较无序的双变量交互的复杂性，这个图很难阅读。我们将在稍后解决这个问题。

在这本书中，带有表格数据的章节通常以数据字典开始。这个例外是因为一开始有 435 个特征。现在，至少了解哪些是顶级特征是有意义的。完整的数据字典可以在[`kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/cup98dic.txt`](https://kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/cup98dic.txt)找到，但由于分类编码，一些特征已经发生了变化，因此我们将在这里更详细地解释它们：

+   `MAXRAMNT`: 连续型，迄今为止最大赠礼的美元金额

+   `HVP2`: 离散型，捐赠者社区中价值>= $150,000 的房屋比例（值在 0 到 100 之间）

+   `LASTGIFT`: 连续型，最近一次赠礼的美元金额

+   `RAMNTALL`: 连续型，迄今为止终身赠礼的美元金额

+   `AVGGIFT`: 连续型，迄今为止赠礼的平均美元金额

+   `MDMAUD_A`: 序数型，对于在其捐赠历史中任何时间点都捐赠了$100+赠礼的捐赠者的捐赠金额代码（值在 0 到 3 之间，对于从未超过$100 的捐赠者为-1）。金额代码是**RFA**（**最近/频率/金额**）主要客户矩阵代码的第三个字节，即捐赠的金额。类别如下：

0：少于$100（低金额）

1：$100 – 499（核心）

2: $500 – 999 (major)

3: $1,000 + (top)

+   `NGIFTALL`: 离散型，迄今为止终身赠礼的数量

+   `AMT_14`: 序数型，14 次之前推广的 RFA 捐赠金额代码，这对应于当时最后一次捐赠的金额：

0: $0.01 – 1.99

1: $2.00 – 2.99

2: $3.00 – 4.99

3: $5.00 – 9.99

4: $10.00 – 14.99

5: $15.00 – 24.99

6: $25.00 及以上

+   `DOMAIN_SOCIALCLS`: 名义型，**社会经济地位**（**SES**）的社区，它与`DOMAIN_URBANICITY`（0：城市，1：城市，2：郊区，3：镇，4：农村）结合，意味着以下：

1: 最高社会经济地位

2: 平均社会经济地位，但城市社区的平均水平以上

3: 最低社会经济地位，但城市社区的平均水平以下

4: 仅城市社区最低社会经济地位

+   `CLUSTER`: 名义型，表示捐赠者所属的集群组的代码

+   `MINRAMNT`: 连续型，迄今为止最小赠礼的美元金额

+   `LSC2`: 离散型，捐赠者社区中西班牙语家庭的比例（值在 0 到 100 之间）

+   `IC15`：离散值，捐赠者所在地区家庭收入低于$15,000 的家庭百分比（值在 0 到 100 之间）

可以从前面的字典和*图 10.9*中提炼出以下见解：

+   **赠款金额优先**：其中七个顶级功能与赠款金额相关，无论是总额、最小值、最大值、平均值还是最后值。如果你包括赠款总数（`NGIFTALL`），则有八个特征涉及捐赠历史，这完全合理。那么，这有什么相关性呢？因为这些特征很可能高度相关，理解它们可能是提高模型的关键。也许可以创建其他特征，更好地提炼这些关系。

+   **连续赠款金额特征的值较高具有高 SHAP 值**：像这样绘制任何这些特征的箱线图，例如`plt.boxplot(X_test.MAXRAMNT)`，你会看到这些特征是如何右偏斜的。也许通过将它们分成区间——称为“离散化”——或使用不同的尺度，如对数尺度（尝试`plt.boxplot(np.log(X_test.MAXRAMNT))`），可以帮助解释这些特征，同时也有助于找到捐赠可能性显著增加的区域。

+   **与第十四次促销的关系**：他们在两年前进行的促销与数据集标签中标记的促销之间发生了什么？促销材料是否相似？是否每两年发生一次季节性因素？也许你可以设计一个特征来更好地识别这种现象。

+   **分类不一致**：`DOMAIN_SOCIALCLS`根据`DOMAIN_URBANITY`值的不同而具有不同的类别。我们可以通过使用量表中的所有五个类别（最高、高于平均水平、平均水平、低于平均水平、最低）来使这一分类一致，即使这意味着非城市捐赠者只会使用三个类别。这样做的好处是更容易解释，而且不太可能对模型的性能产生不利影响。

SHAP 交互摘要图可以用来识别特征和交互排名以及它们之间的某些共同点，但在这种情况下（见*图 10.9*），阅读起来很困难。但要深入挖掘交互，你首先需要量化它们的影响。为此，让我们创建一个热图，只包含按其平均绝对 SHAP 值（`shap_rf_interact_avgs`）测量的顶级交互。然后，我们应该将所有对角线值设置为 0（`shap_rf_interact_avgs_nodiag`），因为这些不是交互，而是特征 SHAP 值，没有它们更容易观察交互。我们可以将这个矩阵放入 DataFrame 中，但它是一个有 111 列和 111 行的 DataFrame，所以为了过滤出具有最多交互的特征，我们使用`scipy`的`rankdata`对它们求和并排名。然后，我们使用排名来识别 12 个最具交互性的特征（`most_interact_cols`），并按这些特征子集 DataFrame。最后，我们将 DataFrame 绘制成热图：

```py
shap_rf_interact_avgs = np.abs(shap_rf_interact_values).mean(0)
shap_rf_interact_avgs_nodiag = shap_rf_interact_avgs.copy()
np.fill_diagonal(shap_rf_interact_avgs_nodiag, 0)
shap_rf_interact_df = pd.DataFrame(shap_rf_interact_avgs_nodiag)
shap_rf_interact_df.columns = X_test[llarsic_cols].columns
shap_rf_interact_df.index = X_test[llarsic_cols].columns
shap_rf_interact_ranks = 112 -rankdata(np.sum(
     shap_rf_interact_avgs_nodiag, axis=0)
)
most_interact_cols = shap_rf_interact_df.columns[
    shap_rf_interact_ranks < 13
]
shap_rf_interact_df = shap_rf_interact_df.loc[
most_interact_cols,most_interact_cols
]
sns.heatmap(
    shap_rf_interact_df,
    cmap='Blues',
    annot=True,
    annot_kws={'size':10},
    fmt='.3f',
    linewidths=.5
) 
Figure 10.10. It depicts the most salient feature interactions according to SHAP interaction absolute mean values. Note that these are averages, so given how right-skewed most of these features are, it is likely much higher for many observations. However, it’s still a good indication of relative impact:
```

![图表  描述自动生成](img/B18406_10_10.png)

图 10.10：SHAP 交互热图

我们可以通过 SHAP 的`dependence_plot`逐个理解特征交互。例如，我们可以选择我们的顶级特征`MAXRAMNT`，并将其与`RAMNTALL`、`LSC4`、`HVP2`和`AVGGIFT`等特征进行颜色编码的交互绘图。但首先，我们需要计算`shap_values`。然而，还有一些问题需要解决，我们之前已经提到了。这些问题与以下内容有关：

+   **异常值的普遍性**：我们可以通过使用特征和 SHAP 值的百分位数来限制*x*轴和*y*轴，分别用`plt.xlim`和`plt.ylim`来将这些异常值从图中剔除。这本质上是在 1st 和 99th 百分位数之间的案例上进行放大。

+   **金额特征的偏斜分布**：在涉及金钱的任何特征中，它通常是右偏斜的。有许多方法可以简化它，例如使用百分位数对特征进行分箱，但一个快速的方法是使用对数刻度。在`matplotlib`中，您可以通过`plt.xscale('log')`来实现这一点，而无需转换特征。

以下代码考虑了两个问题。您可以尝试取消注释`xlim`、`ylim`或`xscale`，以查看它们各自在理解`dependence_plot`时产生的巨大差异：

```py
shap_rf_values = shap_rf_explainer.shap_values(
    X_test.iloc[sample_test_idx] [llarsic_cols]
)
maxramt_shap = shap_rf_values[:,llarsic_cols.index("MAXRAMNT")]
shap.dependence_plot(
    "MAXRAMNT",
    shap_rf_values,
    X_test.iloc[sample_test_idx][llarsic_cols],
    interaction_index="AVGGIFT",
    show=False, alpha=0.1
)
plt.xlim(xmin=np.percentile(X_test.MAXRAMNT, 1),\
         xmax=np.percentile(X_test.MAXRAMNT, 99))
plt.ylim(ymin=np.percentile(maxramt_shap, 1),\
         ymax=np.percentile(maxramt_shap, 99))
plt.xscale('log') 
```

上一段代码生成了*图 10.11*中所示的内容。它显示了`MAXRAMNT`在 10 到 100 之间有一个转折点，模型输出的平均影响开始逐渐增加，这些与更高的`AVGGIFT`值相关：

![图表  描述自动生成，中等置信度](img/B18406_10_11.png)

图 10.11：MAXRAMNT 和 AVGGIFT 之间的 SHAP 交互图

从*图 10.11*中可以得到的教训是，这些特征的一定值以及可能的一些其他值可以增加捐赠的可能性，从而形成一个簇。从特征工程的角度来看，我们可以采用无监督方法，仅基于您已识别为相关的少数特征来创建特殊的簇特征。或者，我们可以采取更手动的方法，通过比较不同的图表来了解如何最好地识别簇。我们可以从这个过程中推导出二元特征，甚至可以推导出特征之间的比率，这些比率可以更清楚地描述交互或簇归属。

这里的想法不是试图重新发明轮子，去做模型已经做得很好的事情，而是首先追求一个更直观的模型解释。希望这甚至可以通过整理特征对预测性能产生积极影响，因为如果您更好地理解它们，也许模型也会！这就像平滑一个颗粒感强的图像；它可能会让您和模型都少一些困惑（有关更多信息，请参阅*第十三章*，*对抗鲁棒性*）！但通过模型更好地理解数据还有其他积极的影响。

事实上，课程不仅仅是关于特征工程或建模，还可以直接应用于促销活动。如果能够识别出转折点，能否用来鼓励捐款呢？或许如果你捐款超过*X*美元，就可以获得一个免费的杯子？或者设置一个每月捐款*X*美元的定期捐款，并成为“银牌”赞助者的专属名单之一？

我们将以这个好奇的笔记结束这个话题，但希望这能激发你去欣赏我们如何将模型解释的教训应用到特征选择、工程以及更多方面。

# 任务完成

为了完成这个任务，你主要使用特征选择工具集来减少过拟合。非营利组织对大约 30%的利润提升感到满意，总成本为 35,601 美元，比向测试数据集中的每个人发送邮件的成本低 30,000 美元。然而，他们仍然希望确保他们可以安全地使用这个模型，而不用担心会亏损。

在本章中，我们探讨了过拟合如何导致盈利曲线不一致。不一致性是关键的，因为它可能意味着基于训练数据选择的阈值在样本外数据上不可靠。因此，你使用`compare_df_plots`来比较测试集和训练集之间的盈利，就像你之前做的那样，但这次是为了选定的模型（`rf_5_e-llarsic`）：

```py
profits_test = reg_mdls['rf_5_e-llarsic']['profits_test']
profits_train = reg_mdls['rf_5_e-llarsic']['profits_train']
mldatasets.compare_df_plots(
    profits_test[['costs', 'profit', 'roi']],
    profits_train[['costs', 'profit', 'roi']],
    'Test',
    'Train',
    x_label='Threshold',
    y_formatter=y_formatter,
    plot_args={'secondary_y':'roi'}
) 
```

上述代码生成了*图 10.12*中所示的内容。你可以向非营利组织展示，以证明在**测试**中，0.68 美元是一个甜点，是可获得的第二高利润。它也在他们的预算范围内，实现了 41%的投资回报率。更重要的是，这些数字与**训练**数据非常接近。另一个令人高兴的是，**训练**和**测试**的利润曲线缓慢下降，而不是突然跌落悬崖。非营利组织可以确信，如果他们选择提高阈值，运营仍然会盈利。毕竟，他们希望针对整个邮件列表的捐赠者，为了使这从财务上可行，他们必须更加专属。比如说，他们在整个邮件列表上使用 0.77 美元的阈值，活动成本约为 46,000 美元，但利润超过 24,000 美元：

![图表，折线图  自动生成的描述](img/B18406_10_12.png)

图 10.12：通过 AIC 特征在不同阈值下，模型使用 LASSO LARS 的测试集和训练集的盈利、成本和投资回报率比较

恭喜！你已经完成了这个任务！

但有一个关键细节，如果我们不提出来，我们可能会疏忽。

尽管我们考虑到下一场活动来训练这个模型，但这个模型很可能会在未来直接营销活动中使用，而无需重新训练。这种模型的重用带来一个问题。有一个概念叫做**数据漂移**，也称为**特征漂移**，即随着时间的推移，模型关于目标变量特征的所学内容不再成立。另一个概念，**概念漂移**，是关于目标特征定义随时间变化的情况。例如，构成有利捐赠者的条件可能会改变。这两种漂移可能同时发生，并且涉及人类行为的问题，这是可以预料的。行为受到文化、习惯、态度、技术和时尚的影响，这些总是在不断发展。您可以警告非营利组织，您只能保证模型在下一场活动中是可靠的，但他们无法承担每次都雇佣您进行模型重新训练的费用！

您可以向客户提议创建一个脚本，直接监控他们的邮件列表数据库中的漂移情况。如果它检测到模型使用的特征有显著变化，它将向他们和您发出警报。在这种情况下，您可以触发模型的自动重新训练。然而，如果漂移是由于数据损坏造成的，您将没有机会解决这个问题。即使进行了自动重新训练，如果性能指标没有达到预定的标准，也无法部署。无论如何，您都应该密切关注预测性能，以确保可靠性。可靠性是模型可解释性的一个基本主题，因为它与问责制密切相关。本书不会涵盖漂移检测，但未来的章节将讨论数据增强（第十一章，*偏差缓解和因果推断方法*）和对抗鲁棒性（第十三章，*对抗鲁棒性*），这些都关乎可靠性。

# 摘要

在本章中，我们学习了无关特征如何影响模型结果，以及特征选择如何提供一套工具来解决此问题。然后，我们探讨了这套工具中的许多不同方法，从最基本过滤器方法到最先进的方法。最后，我们讨论了特征工程的可解释性问题。特征工程可以使模型更具可解释性，从而表现更好。我们将在第十二章，*单调约束和模型调优以实现可解释性*中更详细地介绍这个主题。

在下一章中，我们将讨论偏差缓解和因果推断的方法。

# 数据集来源

+   Ling, C. 和 Li, C.，1998 年，《直接营销的数据挖掘：问题和解决方案》。在第四届国际知识发现和数据挖掘会议（KDD’98）论文集中。AAAI 出版社，第 73-79 页：[`dl.acm.org/doi/10.5555/3000292.3000304`](https://dl.acm.org/doi/10.5555/3000292.3000304)

+   UCI 机器学习仓库，1998，KDD Cup 1998 数据集：[`archive.ics.uci.edu/ml/datasets/KDD+Cup+1998+Data`](https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1998+Data)

# 进一步阅读

+   Ross, B.C.，2014，*离散和连续数据集之间的互信息*。PLoS ONE，9：[`journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357`](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)

+   Geurts, P., Ernst, D., 和 Wehenkel, L.，2006，*极端随机树*。Machine Learning，63(1)，3-42：[`link.springer.com/article/10.1007/s10994-006-6226-1`](https://link.springer.com/article/10.1007/s10994-006-6226-1)

+   Abid, A.，Balin, M.F.，和 Zou, J.，2019，*用于可微分特征选择和重建的混凝土自编码器*。ICML：[`arxiv.org/abs/1901.09346`](https://arxiv.org/abs/1901.09346)

+   Tan, F., Fu, X., Zhang, Y., 和 Bourgeois, A.G.，2008，*基于遗传算法的特征子集选择方法*。Soft Computing，12，111-120：[`link.springer.com/article/10.1007/s00500-007-0193-8`](https://link.springer.com/article/10.1007/s00500-007-0193-8)

+   Calzolari, M.，2020，10 月 12 日，manuel-calzolari/sklearn-genetic：sklearn-genetic 0.3.0（版本 0.3.0）。Zenodo：[`doi.org/10.5281/zenodo.4081754`](http://doi.org/10.5281/zenodo.4081754)

# 在 Discord 上了解更多

要加入这本书的 Discord 社区——在那里您可以分享反馈、向作者提问，并了解新书发布——请扫描下面的二维码：

`packt.link/inml`

![二维码](img/QR_Code107161072033138125.png)
