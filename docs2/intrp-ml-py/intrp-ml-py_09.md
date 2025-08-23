# 9

# 多元预测和敏感性分析的解释方法

在整本书中，我们学习了我们可以用来解释监督学习模型的各种方法。它们在评估模型的同时，也能揭示其最有影响力的预测因子及其隐藏的相互作用。但是，正如“监督学习”这个术语所暗示的，这些方法只能利用已知的样本以及基于这些已知样本分布的排列。然而，当这些样本代表过去时，事情可能会变得复杂！正如诺贝尔物理学奖获得者尼尔斯·玻尔著名地打趣说，“预测是非常困难的，尤其是如果它关乎未来。”

事实上，当你看到时间序列中的数据点波动时，它们可能看起来像是在按照可预测的模式有节奏地跳舞——至少在最佳情况下是这样。就像一个舞者随着节奏移动，每一次重复的动作（或频率）都可以归因于季节性模式，而音量（或振幅）的逐渐变化则可以归因于同样可预测的趋势。这种舞蹈不可避免地具有误导性，因为总会有一些缺失的拼图碎片稍微改变数据点，比如供应商供应链中的延迟导致今天销售数据的意外下降。更糟糕的是，还有不可预见的、十年一遇、一代人一遇或甚至一次性的灾难性事件，这些事件可以彻底改变人们对时间序列运动的一些理解，就像一个舞厅舞者在抽搐。例如，在 2020 年，由于 COVID-19，无论好坏，各地的销售预测都变得毫无用处！

我们可以将这种情况称为极端异常事件，但我们必须认识到，模型并不是为了预测这些重大事件而构建的，因为它们几乎完全基于可能发生的情况进行训练。未能预测这些不太可能但后果严重的意外事件，这就是我们一开始就不应该过度依赖预测模型的原因，尤其是在没有讨论确定性或置信区间的情况下。

本章将探讨一个使用**长短期记忆**（**LSTM**）模型的多元预测问题。我们将首先使用传统的解释方法评估模型，然后使用我们在第七章“可视化卷积神经网络”中学习的**集成梯度**方法来生成我们模型的局部属性。

但更重要的是，我们将更好地理解 LSTM 的学习过程和局限性。然后，我们将采用预测近似方法以及 SHAP 的`KernelExplainer`进行全局和局部解释。最后，*预测和不确定性是内在相关的*，*敏感性分析*是一系列旨在衡量模型输出不确定性相对于其输入的方法，因此在预测场景中非常有用。我们还将研究两种这样的方法：**Morris**用于*因素优先级排序*和**Sobol**用于*因素固定*，这涉及到成本敏感性。

下面是我们将要讨论的主要主题：

+   使用传统解释方法评估时间序列模型

+   使用积分梯度生成 LSTM 属性

+   使用 SHAP 的`KernelExplainer`计算全局和局部属性

+   使用因素优先级识别有影响力的特征

+   使用因素固定量化不确定性和成本敏感性

让我们开始吧！

# 技术要求

本章的示例使用了`mldatasets`、`pandas`、`numpy`、`sklearn`、`tensorflow`、`matplotlib`、`seaborn`、`alibi`、`distython`、`shap`和`SALib`库。如何安装所有这些库的说明可以在本书的序言中找到。

本章的代码位于此处：[`packt.link/b6118`](https://packt.link/b6118)。

# 任务

高速公路交通拥堵是一个影响世界各地的城市的问题。随着发展中国家每千人车辆数量的稳步增加，而道路和停车基础设施不足以跟上这一增长，拥堵水平已经达到了令人担忧的程度。在美国，每千人车辆统计数字是世界上最高的之一（2019 年为每千人 838 辆）。因此，美国城市占全球 381 个城市中至少有 15%拥堵水平的 62 个城市。

明尼阿波利斯就是这样一座城市（参见*图 9.1*），那里的阈值最近已经超过并持续上升。为了将这个大都市地区置于适当的背景中，拥堵水平在 50%以上时极为严重，但中等程度的拥堵（15-25%）已经是未来可能出现严重拥堵的预警信号。一旦拥堵达到 25%，就很难逆转，因为任何基础设施的改善都将非常昂贵，而且还会进一步扰乱交通。最严重的拥堵点之一是在明尼阿波利斯和圣保罗这对双城之间的 94 号州际公路（I-94），当通勤者试图缩短旅行时间时，这会拥堵替代路线。了解这一点后，这两座城市的市长们已经获得了一些联邦资金来扩建这条公路：

![图形用户界面，应用程序描述自动生成](img/B18406_09_01.png)

图 9.1：TomTom 的 2019 年明尼阿波利斯交通指数

市长们希望能够吹嘘一个完成的扩建项目作为共同成就，以便在第二任期内再次当选。然而，他们很清楚，一个嘈杂、脏乱和阻碍交通的扩建项目可能会给通勤者带来很大的麻烦，因此，如果扩建项目不是几乎看不见，它可能会在政治上适得其反。因此，他们规定建筑公司尽可能在其他地方预制，并在低流量时段进行组装。这些时段的每小时交通量少于 1,500 辆车。他们一次只能在一个方向的高速公路上工作，并且在他们工作时只能阻挡不超过一半的车道。为了确保遵守这些规定，如果他们在任何交通量超过这个阈值时阻挡超过四分之三的高速公路，他们将对公司处以每辆车 15 美元的罚款。

除了这些，如果施工队伍在每小时交通量超过 1,500 辆车时在现场阻挡一半的高速公路，他们每天将花费 5,000 美元。为了更直观地了解这一点，在典型的交通高峰时段阻挡可能会使建筑公司每小时损失 67,000 美元，再加上每天 5,000 美元的费用！当地当局将使用沿路线的**自动交通记录器**（**ATR**）站点来监控交通流量，以及当地交通警察来记录施工时车道被阻挡的情况。

该项目已被规划为一个为期 2 年的建设项目；第一年将在 I-94 路线的西行车道进行扩建，而第二年将扩建东行车道。施工现场的建设仅从 5 月到 10 月进行，因为在这几个月里下雪不太可能延误施工。在整个余下的年份，他们将专注于预制。他们将尝试只在工作日工作，因为工人联盟为周末谈判了慷慨的加班费。因此，只有在有重大延误的情况下，周末才会进行施工。然而，工会同意在 5 月至 10 月期间以相同的费率在节假日工作。

建筑公司不想承担任何风险！因此，他们需要一个模型来预测 I-94 路线的交通流量，更重要的是，要了解哪些因素会创造不确定性并可能增加成本。他们已经聘请了一位机器学习专家来完成这项工作：就是你！

建筑公司提供的 ATR 数据包括截至 2018 年 9 月的每小时交通量，以及同一时间尺度的天气数据。它只包括西行车道，因为那部分扩建将首先进行。

# 方法

您已经使用近四年的数据（2012 年 10 月 – 2016 年 9 月）训练了一个有状态的**双向 LSTM**模型。您保留了最后一年用于测试（2017 年 9 月–2018 年）和之前一年用于验证（2016 年 9 月 –2017 年）。这样做是有道理的，因为测试和验证数据集与高速公路扩建项目预期的条件（3 月 – 11 月）相吻合。您曾考虑使用仅利用这些条件数据的其他分割方案，但您不想如此大幅度地减少训练数据，也许它们最终还是可能需要用于冬季预测。回望窗口定义了时间序列模型可以访问多少过去数据。您选择了 168 小时（1 周）作为回望窗口大小。鉴于模型的这种有状态性质，随着模型在训练数据中的前进，它可以学习每日和每周的季节性，以及一些只能在几周内观察到的趋势和模式。您还训练了另外两个模型。您概述了以下步骤以满足客户期望：

1.  使用 *RMSE*、*回归图*、*混淆矩阵* 等等，您将访问模型的预测性能，更重要的是，了解误差的分布情况。

1.  使用 *集成梯度*，您将了解是否采取了最佳建模策略，因为它可以帮助您可视化模型到达决策的每一条路径，并帮助您根据这一点选择模型。

1.  使用 *SHAP 的* `KernelExplainer` 和预测近似方法，您将推导出对所选模型有重要意义的特征的全局和局部理解。

1.  使用 *Morris 敏感性分析*，您将识别 *因子优先级*，它根据它们可以驱动输出变异性的程度对因素（换句话说，特征）进行排序。

1.  使用 *Sobol 敏感性分析*，您将计算 *因子固定*，这有助于确定哪些因素不具有影响力。它是通过量化输入因素对输出变异性的贡献和相互作用来做到这一点的。有了这个，您可以了解哪些因素可能对潜在的罚款和成本影响最大，从而产生基于变异性的成本敏感性分析。

# 准备工作

您可以在此处找到此示例的代码：[`github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/blob/main/09/Traffic_compact1.ipynb`](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python-2E/blob/main/09/Traffic_compact1.ipynb)。

## 加载库

要运行此示例，您需要安装以下库：

+   `mldatasets` 用于加载数据集

+   `pandas` 和 `numpy` 用于操作数据集

+   `tensorflow` 用于加载模型

+   `scikit-learn`、`matplotlib`、`seaborn`、`alibi`、`distython`、`shap` 和 `SALib` 用于创建和可视化解释

您应该首先加载所有这些内容：

```py
import math
import os
import mldatasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import \ TimeseriesGenerator
from keras.utils import get_file
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from alibi.explainers import IntegratedGradients
from distython import HEOM
import shap
from SALib.sample import morris as ms
from SALib.analyze import morris as ma
from SALib.plotting import morris as mp
from SALib.sample.saltelli import sample as ss
from SALib.analyze.sobol import analyze as sa
from SALib.plotting.bar import plot as barplot 
```

让我们通过使用`print(tf.__version__)`命令来检查 TensorFlow 是否加载了正确的版本。它应该是 2.0 或更高版本。

## 理解和准备数据

```py
are loading the data into a DataFrame called traffic_df. Please note that the prepare=True parameter is important because it performs necessary tasks such as subsetting the DataFrame to the required timeframe, since October 2015, some interpolation, correcting holidays, and performing one-hot encoding:
```

```py
traffic_df = mldatasets.load("traffic-volume-v2", prepare=True) 
```

应该有超过 52,000 条记录和 16 列。我们可以使用`traffic_df.info()`来验证这一点。输出应该符合预期。所有特征都是数值型的，没有缺失值，并且分类特征已经为我们进行了独热编码。

### 数据字典

由于分类编码，只有九个特征，但它们变成了 16 列：

+   `dow`: 序数型；以星期一开始的星期几（介于 0 到 6 之间）

+   `hr`: 序数型；一天中的小时（介于 0 到 23 之间）

+   `temp`: 连续型；摄氏度平均温度（介于-30 到 37 之间）

+   `rain_1h`: 连续型；该小时发生的降雨量（介于 0 到 21 毫米之间）

+   `snow_1h`: 连续型；该小时发生的雪量（当转换为液体形式时）（介于 0 到 2.5 厘米之间）

+   `cloud_coverage`: 连续型；云层覆盖率百分比（介于 0 到 100 之间）

+   `is_holiday`: 二元型；该天是星期一到星期五的国家或州假日吗？（1 表示是，0 表示否）？

+   `traffic_volume`: 连续型；捕获交通量的目标特征

+   `weather`: 分类；该小时天气的简短描述（晴朗 | 云层 | 雾 | 薄雾 | 雨 | 雪 | 未知 | 其他）

### 理解数据

理解时间序列问题的第一步是理解目标变量。这是因为它决定了你如何处理其他所有事情，从数据准备到建模。目标变量可能与时间有特殊的关系，例如季节性变动或趋势。

#### 理解周

首先，我们可以从每个季节中采样一个 168 小时的时间段，以更好地理解一周中每天之间的方差，然后了解它们如何在季节和假日之间变化：

```py
lb = 168
fig, (ax0,ax1,ax2,ax3) = plt.subplots(4,1, figsize=(15,8))
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4)
traffic_df[(lb*160):(lb*161)].traffic_volume.plot(ax=ax0)
traffic_df[(lb*173):(lb*174)].traffic_volume.plot(ax=ax1)
traffic_df[(lb*186):(lb*187)].traffic_volume.plot(ax=ax2)
traffic_df[(lb*199):(lb*200)].traffic_volume.plot(ax=ax3) 
```

前面的代码生成了*图 9.2*中显示的图表。如果你从左到右阅读它们，你会看到它们都是从星期三开始，以下一周的星期二结束。每周的每一天都是从低点开始和结束，中间有一个高点。工作日通常有两个高峰，对应早晨和下午高峰时段，而周末只有一个下午的峰值：

![](img/B18406_09_02.png)

图 9.2：代表每个季节的几个交通量样本周周期

存在一些主要异常值，例如 10 月 31 日星期六，这基本上是万圣节，并不是官方的节假日。还有 2 月 2 日（星期二）是严重的暴风雪的开始，而夏末的时期比其他样本周要混乱得多。结果发现，那一年州博览会发生了。像万圣节一样，它既不是联邦节日也不是地区节日，但重要的是要注意博览会场地位于明尼阿波利斯和圣保罗之间的一半。你还会注意到，在 7 月 29 日星期五午夜时，交通量有所上升，这可以归因于这是明尼阿波利斯音乐会的大日子。

在比较时间序列中的各个时期时，试图解释这些不一致性是一个很好的练习，因为它有助于你确定需要添加到模型中的变量，或者至少知道缺少了什么。在我们的案例中，我们知道我们的`is_holiday`变量不包括万圣节或整个州博览会周，也没有针对大型音乐或体育赛事的变量。为了构建一个更稳健的模型，寻找可靠的外部数据源并添加更多覆盖所有这些可能性的特征是明智的，更不用说验证现有变量了。目前，我们将使用我们拥有的数据。

#### 理解日子

对于高速公路扩建项目来说，了解平均工作日的交通状况至关重要。施工队伍只在工作日（周一至周五）工作，除非遇到延误，在这种情况下，他们也会在周末工作。我们还必须区分节假日和其他工作日，因为这些可能有所不同。

为了这个目的，我们将创建一个 DataFrame（`weekend_df`）并创建一个新列（`type_of_day`），将小时编码为“假日”、“工作日”或“周末”。然后，我们可以按此列和`hr`列进行分组，并使用`mean`和标准差（`std`）进行聚合。然后我们可以进行`pivot`，以便我们有一个列，其中包含每个`type_of_day`类别的平均交通量和标准差，其中行代表一天中的小时数（`hr`）。然后，我们可以绘制结果 DataFrame。我们可以创建包含标准差的区间：

```py
weekend_df = traffic_df[
    ['hr', 'dow', 'is_holiday', 'traffic_volume']].copy()
weekend_df['type_of_day'] = np.where(
    weekend_df.is_holiday == 1,
    'Holiday',
    np.where(weekend_df.dow >= 5, 'Weekend', 'Weekday')
)
weekend_df = weekend_df.groupby(
['type_of_day','hr']) ['traffic_volume']
    .agg(['mean','std'])
    .reset_index()
    .pivot(index='hr', columns='type_of_day', values=['mean', 'std']
)
weekend_df.columns = [
    ''.join(col).strip().replace('mean','')\
    for col in weekend_df.columns.values
]
fig, ax = plt.subplots(figsize=(15,8))
weekend_df[['Holiday','Weekday','Weekend']].plot(ax=ax)
plt.fill_between(
    weekend_df.index,
    np.maximum(weekend_df.Weekday - 2 * weekend_df.std_Weekday, 0),
    weekend_df.Weekday + 2 * weekend_df.std_Weekday,
    color='darkorange',
    alpha=0.2
)
plt.fill_between(
    weekend_df.index,\
    np.maximum(weekend_df.Weekend - 2 * weekend_df.std_Weekend, 0),
    weekend_df.Weekend + 2 * weekend_df.std_Weekend,
    color='green',
    alpha=0.1
)
plt.fill_between(
    weekend_df.index,\
    np.maximum(weekend_df.Holiday - 2 * weekend_df.std_Holiday, 0),
    weekend_df.Holiday + 2 * weekend_df.std_Holiday,
    color='cornflowerblue',
    alpha=0.1
) 
```

前面的代码片段产生了以下图表。它表示每小时平均交通量，但变化很大，这就是为什么建筑公司正在谨慎行事。图中绘制了代表每个阈值的水平线：

+   容量满载时为 5,300。

+   半容量时为 2,650，之后建筑公司将因每日指定金额被罚款。

+   无施工阈值是 1,500，之后建筑公司将因每小时指定金额被罚款。

他们只想在通常低于 1500 阈值的小时内工作，周一到周五。这五个小时将是晚上 11 点（前一天）到早上 5 点。如果他们必须周末工作，这个时间表通常会推迟到凌晨 1 点，并在早上 6 点结束。在工作日，变化相对较小，所以建筑公司坚持只在工作日工作是可以理解的。在这些小时里，节假日看起来与周末相似，但节假日的变化甚至比周末更大，这可能是更成问题的情况：

![图表描述自动生成](img/B18406_09_03.png)

图 9.3：节假日、工作日和周末的平均每小时交通量，以及间隔

通常，对于这样的项目，你会探索预测变量，就像我们对目标所做的那样。这本书是关于模型解释的，所以我们将通过解释模型来了解预测变量。但在我们到达模型之前，我们必须为它们准备数据。

### 数据准备

第一步数据准备是将数据分割成训练集、验证集和测试集。请注意，测试数据集包括最后 52 周（`2184`小时），而验证数据集包括之前的 52 周，因此它从`4368`小时开始，到 DataFrame 最后一行之前的`2184`小时结束：

```py
train = traffic_df[:-4368]
valid = traffic_df[-4368:-2184]
test = traffic_df[-2184:] 
```

现在 DataFrame 已经被分割，我们可以绘制它以确保其部分是按照预期分割的。我们可以使用以下代码来完成：

```py
plt.plot(train.index.values, train.traffic_volume.values,
          label='train')
plt.plot(valid.index.values, valid.traffic_volume.values,
           label='validation')
plt.plot(test.index.values, test.traffic_volume.values,
          label='test')
plt.ylabel('Traffic Volume')
plt.legend() 
```

上述代码生成了*图 9.4*。它显示，训练数据集分配了近 4 年的数据，而验证和测试各分配了一年。在这个练习中，我们将不再引用验证数据集，因为它只是在训练期间作为工具来评估模型在每个 epoch 后的预测性能。

![图表描述自动生成，置信度低](img/B18406_09_04.png)

图 9.4：时间序列分割为训练集、验证集和测试集

下一步是对数据进行 min-max 归一化。我们这样做是因为较大的值会导致所有神经网络的学习速度变慢，而 LSTM 非常容易发生**梯度爆炸和消失**。相对均匀且较小的数字可以帮助解决这些问题。我们将在本章后面讨论这个问题，但基本上，网络要么在数值上不稳定，要么在达到全局最小值方面无效。

我们可以使用`scikit`包中的`MinMaxScaler`进行 min-max 归一化。目前，我们只会对归一化器进行`fit`操作，以便我们可以在需要时使用它们。我们将为我们的目标（`traffic_volume`）创建一个名为`y_scaler`的归一化器，并为其余变量（`X_scaler`）创建另一个归一化器，使用整个数据集，以确保无论使用哪个部分（`train`、`valid`或`test`），转换都是一致的。所有的`fit`过程只是保存公式，使每个变量适合在零和一之间：

```py
y_scaler = MinMaxScaler()
y_scaler.fit(traffic_df[['traffic_volume']])
X_scaler = MinMaxScaler()
X_scaler.fit(traffic_df.drop(['traffic_volume'], axis=1)) 
```

现在，我们将使用我们的缩放器 `transform` 我们的训练和测试数据集，为每个创建 *y* 和 *X* 对：

```py
y_train = y_scaler.transform(train[['traffic_volume']])
X_train = X_scaler.transform(train.drop(['traffic_volume'], axis=1))
y_test = y_scaler.transform(test[['traffic_volume']])
X_test = X_scaler.transform(test.drop(['traffic_volume'], axis=1)) 
```

然而，对于时间序列模型，我们创建的 *y* 和 *X* 对并不有用，因为每个观测值都是一个时间步长。每个时间步长不仅仅是该时间步长发生的特征，而且在一定程度上是它之前发生的事情，称为滞后。例如，如果我们根据 168 个滞后观测值预测交通，对于每个标签，我们将需要每个特征的之前 168 小时的数据。因此，你必须为每个时间步长以及其滞后生成一个数组。幸运的是，`keras` 有一个名为 `TimeseriesGenerator` 的函数，它接受你的 *X* 和 *y* 并生成一个生成器，该生成器将数据馈送到你的模型。你必须指定一个特定的 `length`，这是滞后观测值的数量（也称为 **lookback window**）。默认的 `batch_size` 是一个，但我们使用 24，因为客户更喜欢一次获取 24 小时的预测，而且使用更大的批次大小进行训练和推理要快得多。

自然地，当你需要预测明天时，你需要明天的天气，但你可以用天气预报来补充时间步长：

```py
gen_train = TimeseriesGenerator(
    X_train,
    y_train,
    length=lb,
    batch_size=24
)
gen_test = TimeseriesGenerator(
    X_test,
    y_test,
    length=lb,
    batch_size=24
)
print(
    "gen_train:%s×%s→%s" % (len(gen_train),
    gen_train[0][0].shape, gen_train[0][1].shape)
)
print(
    "gen_test:%s×%s→%s" % (len(gen_test),
    gen_test[0][0].shape, gen_test[0][1].shape)
) 
gen_train) and the testing generator (gen_test), which use a length of 168 hours and a batch size of 24:
```

```py
gen_train:  1454 ×   (24, 168, 15)   →   (24, 1)
gen_test:   357  ×   (24, 168, 15)   →   (24, 1) 
```

任何使用 1 周滞后窗口和 24 小时批次大小训练的模型都需要这个生成器。每个生成器是与每个批次对应的元组的列表。这个元组的索引 0 是 *X* 特征数组，而索引 1 是 *y* 标签数组。因此，输出的第一个数字是列表的长度，即批次的数量。*X* 和 *y* 数组的维度随后。

例如，`gen_train` 有 1,454 个批次，每个批次有 24 个时间步长，长度为 168，有 15 个特征。从这些 24 个时间步长中预期的预测标签的形状是 `(24,1)`。

最后，在继续处理模型和随机解释方法之前，让我们尝试通过初始化我们的随机种子来使事情更具可重复性：

```py
rand = 9
os.environ['PYTHONHASHSEED']=str(rand)
tf.random.set_seed(rand)
np.random.seed(rand) 
```

### 加载 LSTM 模型

我们可以快速加载模型并像这样输出其摘要：

```py
model_name = 'LSTM_traffic_168_compact1.hdf5'
model_path = get_file(
    model_name,
    'https://github.com/PacktPublishing/Interpretable-\ 
    Machine-Learning-with-Python-2E/blob/main/models/{}?raw=true'
    .format(model_name)
)
lstm_traffic_mdl = keras.models.load_model(model_path)
lstm_traffic_mdl.summary() 
bidirectional LSTM layer with an output of (24, 168). 24 corresponds to the batch size, while 168 means that there’s not one but two 84-unit LSTMs going in opposite directions and meeting in the middle. It has a dropout of 10%, and then a dense layer with a single ReLu-activated unit. The ReLu ensures that all the predictions are over zero since negative traffic volume makes no sense:
```

```py
Model: "LSTM_traffic_168_compact1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Bidir_LSTM (Bidirectional)   (24, 168)                 67200     
_________________________________________________________________
Dropout (Dropout)            (24, 168)                 0         
_________________________________________________________________
Dense (Dense)                (24, 1)                   169       
=================================================================
Total params: 67,369
Trainable params: 67,369
Non-trainable params: 0
_________________________________________________________________ 
```

现在，让我们使用传统的解释方法评估 `LSTM_traffic_168_compact1` 模型。

# 使用传统的解释方法评估时间序列模型

时间序列回归模型可以像评估任何回归模型一样进行评估；也就是说，使用来自 **均方误差** 或 **R-squared** 分数的指标。当然，在某些情况下，你可能需要使用具有中位数、对数、偏差或绝对值的指标。这些模型不需要任何这些。

## 使用标准的回归指标

`evaluate_reg_mdl` 函数可以评估模型，输出一些标准的回归指标，并绘制它们。此模型的参数是拟合的模型 (`lstm_traffic_mdl`)，`X_train` (`gen_train`)，`X_test` (`gen_test`)，`y_train` 和 `y_test`。

可选地，我们可以指定一个`y_scaler`，以便模型使用标签的逆变换进行评估，这使得绘图和**均方根误差**（**RMSE**）更容易理解。在这种情况下，另一个非常必要的可选参数是`y_truncate=True`，因为我们的`y_train`和`y_test`的维度比预测标签大。这种差异发生是因为由于回望窗口，第一次预测发生在数据集的第一个时间步之后。因此，我们需要从`y_train`中减去这些时间步，以便与`gen_train`的长度匹配：

现在我们将使用以下代码评估这两个模型。为了观察预测的进度，我们将使用`predopts={"verbose":1}`。

```py
y_train_pred, y_test_pred, y_train, y_test =\
    mldatasets.evaluate_reg_mdl(lstm_traffic_mdl,
        gen_train,
        gen_test,
        y_train,
        y_test,
        scaler=y_scaler,
        y_truncate=True,
        predopts={"verbose":1}
) 
Figure 9.5. The *regression plot* is, essentially, a scatter plot of the observed versus predicted traffic volumes, fitted to a linear regression model to show how well they match. These plots show that the model tends to predict zero traffic when it’s substantially higher. Besides that, there are a number of extreme outliers, but it fits relatively well with a test RMSE of 430 and only a slightly better train RMSE:
```

![图片](img/B18406_09_05.png)

图 9.5：“LSTM_traffic_168_compact1”模型的预测性能评估

我们还可以通过比较观察到的与预测的交通来评估模型。按小时和类型分解错误可能会有所帮助。为此，我们可以创建包含这些值的 DataFrames - 每个模型一个。但首先，我们必须截断 DataFrame（`-y_test_pred.shape[0]`），以便它与预测数组的长度匹配，我们不需要所有列，所以我们只提供我们感兴趣的索引：`traffic_volume`是第 7 列，但我们还希望有`dow`（第 0 列）、`hr`（第 1 列）和`is_holiday`（第 6 列）。我们将`traffic_volume`重命名为`actual_traffic`，并创建一个名为`predicted_traffic`的新列，其中包含我们的预测。然后，我们将创建一个`type_of_day`列，就像我们之前做的那样，它告诉我们是否是节假日、工作日还是周末。最后，我们可以删除`dow`和`is_holiday`列，因为我们不再需要它们：

```py
evaluate_df = test.iloc[-y_test_pred.shape[0]:,[0,1,6,7]]
    .rename(columns={'traffic_volume':'actual_traffic'}
)
evaluate_df['predicted_traffic'] = y_test_pred
evaluate_df['type_of_day'] = np.where(
    evaluate_df.is_holiday == 1,
    'Holiday',
    np.where(evaluate_df.dow >= 5,
    'Weekend', 'Weekday')
)
evaluate_df.drop(['dow','is_holiday'], axis=1, inplace=True) 
```

您可以通过简单地运行一个带有`evaluate_df`的单元格来快速查看 DataFrames 的内容。它应该有 4 列。

### 预测误差聚合

可能是某些日期和时间更容易出现预测误差。为了更好地了解这些误差如何在时间上分布，我们可以按小时分段绘制`type_of_day`的 RMSE。为此，我们必须首先定义一个`rmse`函数，然后按`type_of_day`和`hr`对每个模型的评估 DataFrame 进行分组，并使用`apply`函数通过`rmse`函数进行聚合。然后我们可以通过转置来确保每个`type_of_day`都有一个按小时显示 RMSE 的列。然后我们可以平均这些列并将它们存储在一个系列中：

```py
def rmse(g):
    rmse = np.sqrt(
    metrics.mean_squared_error(g['actual_traffic'],
                               g['predicted_traffic'])
    )
    return pd.Series({'rmse': rmse})
evaluate_by_hr_df = evaluate_df.groupby(['type_of_day', 'hr'])
    .apply(rmse).reset_index()
    .pivot(index='hr', columns='type_of_day', values='rmse')
mean_by_daytype_s = evaluate_by_hr_df.mean(axis=0) 
```

现在我们有了包含节假日、工作日和周末每小时 RMSE 的 DataFrames，以及这些“类型”的日平均数，我们可以使用`evaluate_by_hr` DataFrame 来绘制它们。我们还将创建带有每个`type_of_day`平均值的虚线水平线，这些平均值来自`mean_by_daytype` `pandas`系列：

```py
evaluate_by_hr_df.plot()
ax = plt.gca()
ax.set_title('Hourly RMSE distribution', fontsize=16)
ax.set_ylim([0,2500])
ax.axhline(
    y=mean_by_daytype_s.Holiday,
    linewidth=2,
    color='cornflowerblue',
    dashes=(2,2)
)
ax.axhline(
    y=mean_by_daytype_s.Weekday,
    linewidth=2,
    color='darkorange',
    dashes=(2,2)
)
ax.axhline(
    y=mean_by_daytype_s.Weekend,
    linewidth=2,
    color='green',
    dashes=(2,2)
) 
```

上述代码生成了*图 9.6*中显示的图表。正如我们所见，该模型在假日有很高的 RMSE。然而，该模型可能高估了交通量，而在这种特定情况下，高估不如低估糟糕，因为低估可能导致交通延误和额外的罚款成本：

![图表，折线图，描述自动生成](img/B18406_09_06.png)图 9.6：“LSTM_traffic_168_compact1”模型按 type_of_day 类型划分的小时 RMSE

### 将模型评估视为分类问题

的确，就像分类问题可以有假阳性和假阴性，其中一个是比另一个更昂贵的，你可以用诸如低估和过度估计等概念来构建任何回归问题。这种构建特别有用，当其中一个比另一个更昂贵时。如果你有明确定义的阈值，就像我们在这个项目中做的那样，你可以像评估分类问题一样评估任何回归问题。我们将使用半容量和“无施工”阈值混淆矩阵来评估它。为了完成这项任务，我们可以使用`np.where`来获取实际值和预测值超过每个阈值的二进制数组。然后我们可以使用`compare_confusion_matrices`函数来比较模型的混淆矩阵：

```py
actual_over_half_cap = np.where(evaluate_df['actual_traffic'] >\
                                2650, 1, 0)
pred_over_half_cap = np.where(evaluate_df['predicted_traffic'] >\
                              2650, 1, 0)
actual_over_nc_thresh = np.where(evaluate_df['actual_traffic'] >\
                                 1500, 1, 0)
pred_over_nc_thresh = np.where(evaluate_df['predicted_traffic'] >\
                               1500, 1, 0)
mldatasets.compare_confusion_matrices(
    actual_over_half_cap,
    pred_over_half_cap,
    actual_over_nc_thresh,
    pred_over_nc_thresh,
    'Over Half-Capacity',
    'Over No-Construction Threshold'
) 
Figure 9.7.
```

![图表描述自动生成](img/B18406_09_07.png)

图 9.7：“LSTM_traffic_168_compact1”模型的超过半容量和“无施工”阈值的混淆矩阵

我们最感兴趣的是假阴性（左下象限）的百分比，因为当实际上交通量超过了阈值时，预测没有超过阈值将导致高额罚款。另一方面，假阳性的成本在于在交通量实际上没有超过阈值的情况下提前离开施工现场。尽管如此，安全总是比后悔好！如果你比较“无施工”阈值的假阴性（0.85%），它不到半容量阈值（3.08%）的三分之一。最终，最重要的是无施工阈值，因为目的是在接近半容量之前停止施工。

现在我们已经利用传统方法来理解模型的决策，让我们继续探讨一些更高级的模型无关方法。

# 使用集成梯度生成 LSTM 归因

我们第一次在*第七章*，*可视化卷积神经网络*中了解到**集成梯度**（**IG**）。与该章节中研究的其他基于梯度的归因方法不同，路径集成梯度不依赖于卷积层，也不限于分类问题。

事实上，因为它计算了输出相对于输入沿路径平均的梯度，所以输入和输出可以是任何东西！通常与**卷积神经网络**（**CNNs**）和**循环神经网络**（**RNNs**）一起使用整合梯度，就像我们在本章中解释的那样。坦白说，当你在网上看到 IG LSTM 的例子时，它有一个嵌入层，是一个 NLP 分类器，但 IG 对于甚至处理声音或遗传数据的 LSTMs 也非常有效！

整合梯度解释器和我们将继续使用的解释器可以访问交通数据集的任何部分。首先，让我们为所有这些创建一个生成器：

```py
y_all = y_scaler.transform(traffic_df[['traffic_volume']])
X_all = X_scaler.transform(
    traffic_df.drop(['traffic_volume'], axis=1)
)
gen_all = TimeseriesGenerator(
    X_all, y_all, length=lb, batch_size=24
) 
```

整合梯度是一种局部解释方法。所以，让我们获取一些我们可以解释的“感兴趣的样本实例”。我们知道假日可能需要专门的逻辑，所以让我们看看我们的模型是否注意到了`is_holiday`在某个例子（`holiday_afternoon_s`）中的重要性。早晨也是一个问题，尤其是由于天气条件，早晨的拥堵时间比平均水平更长，所以我们有一个例子（`peak_morning_s`）。最后，一个炎热的日子可能会有更多的交通，尤其是在周末（`hot_Saturday_s`）：

```py
X_df = traffic_df.drop(['traffic_volume'], axis=1 \
    ).reset_index(drop=True)
holiday_afternoon_s = X_df[
    (X_df.index >= 43800) & (X_df.dow==0) &\
    (X_df.hr==16) &(X_df.is_holiday==1)
].tail(1)
peak_morning_s = X_df[
    (X_df.index >= 43800) & (X_df.dow==2) &\
    (X_df.hr==8) & (X_df.weather_Clouds==1) & (X_df.temp<20)
].tail(1)
hot_Saturday_s = X_df[
    (X_df.index >= 43800) & (X_df.dow==5) &\
    (X_df.hr==12) & (X_df.temp>29) & (X_df.weather_Clear==1)
].tail(1) 
```

现在我们已经创建了一些实例，让我们实例化我们的解释器。来自`alibi`包的`IntegratedGradients`只需要一个深度学习模型，但建议为积分近似设置步骤数（`n_steps`）和内部批次大小。我们将为我们的模型实例化一个解释器：

```py
ig = IntegratedGradients(
    lstm_traffic_mdl, n_steps=25, internal_batch_size=24
) 
```

在我们对样本和解释器进行迭代之前，重要的是要意识到我们需要如何将样本输入到解释器中，因为它需要一个包含 24 个样本的批次。为此，一旦我们减去回望窗口（`nidx`），我们就必须获取样本的索引。然后，你可以从生成器（`gen_all`）中获取该样本的批次。每个批次包含 24 个时间步长，所以你需要将`nidx`向下取整到 24（`nidx//24`），以获取该样本的批次位置。一旦你获取了该样本的批次（`batch_X`）并打印了形状（`24, 168, 15`），第一个数字是 24 这一点不应该让你感到惊讶。当然，我们还需要获取批次内样本的索引（`nidx%24`），以获取该样本的数据：

```py
nidx = holiday_afternoon_s.index.tolist()[0] – lb
batch_X = gen_all[nidx//24][0]
print(batch_X.shape) 
```

`for`循环将使用之前解释的方法来定位样本的批次（`batch_X`）。这个`batch_X`被输入到`explain`函数中。这是因为这是一个回归问题，没有目标类别；也就是说，`target=None`。一旦产生了解释，`attributions`属性将包含整个批次的属性。我们只能获取样本的属性，并将其`transpose`以产生一个形状为`(15, lb)`的图像。`for`循环中的其余代码只是获取用于刻度的标签，然后绘制一个图像，该图像拉伸以适应我们的`figure`维度，以及其标签：

```py
samples = [holiday_afternoon_s, peak_morning_s, hot_saturday_s]
sample_names = ['Holiday Afternoon', 'Peak Morning' , 'Hot Saturday']
for s in range(len(samples)):
    nidx = samples[s].index.tolist()[0] - lb
    batch_X = gen_all[nidx//24][0]
    explanation = ig.explain(batch_X, target=None)
    attributions = explanation.attributions[0]
    attribution_img = np.transpose(attributions[nidx%24,:,:])
    end_date = traffic_df.iloc[samples[s].index
        ].index.to_pydatetime()[0]
    date_range = pd.date_range(
        end=end_date, periods=8, freq='1D').to_pydatetime().tolist()
    columns = samples[s].columns.tolist()  
    plt.title(
        'Integrated Gradient Attribution Map for "{}"'.\
        format(sample_names[s], lb), fontsize=16
    )
    divnorm = TwoSlopeNorm(
        vmin=attribution_img.min(),
        vcenter=0,
        vmax=attribution_img.max()
    )
    plt.imshow(
        attribution_img,
        interpolation='nearest' ,
        aspect='auto',
        cmap='coolwarm_r',
        norm=divnorm
)
    plt.xticks(np.linspace(0,lb,8).astype(int), labels=date_range)
    plt.yticks([*range(15)], labels=columns)
    plt.colorbar(pad=0.01,fraction=0.02,anchor=(1.0,0.0))
    plt.show() 
```

上述代码将生成*图 9.8*中显示的图表。在*y*轴上，您可以看到变量名称，而在*x*轴上，您可以看到对应于所讨论样本回望窗口的日期。*x*轴的最右侧是样本的日期，随着您向左移动，您会向时间后退。例如，假日下午的样本是 9 月 3 日下午 4 点，有一周的回望，所以每个向后的刻度代表该日期前一天。

![图形用户界面，应用程序，表格  自动生成的描述](img/B18406_09_08.png)

图 9.8：对于“LSTM_traffic_168_compact1”模型所有样本的注释集成梯度归因图

您可以通过查看*图 9.8*中的归因图强度来判断哪些小时/变量对预测很重要。每个归因图右侧的颜色条可以用作参考。红色中的负数表示负相关，而蓝色中的正数表示正相关。然而，一个相当明显的是，随着每个图向时间后退，强度往往会减弱。由于它是双向的，所以这种情况发生在两端。令人惊讶的是，这个过程发生得有多快。

让我们从底部开始。对于“热周六”，随着您接近预测时间（周六中午），星期几、小时、温度和晴朗的天气在这个预测中扮演的角色越来越重要。天气开始较凉爽，这解释了为什么在温度特征中红色区域出现在蓝色区域之前。

对于“高峰早晨”，归因是有意义的，因为它在之前有雨和多云之后变得晴朗，这导致高峰时段迅速达到顶峰而不是缓慢增加。在一定程度上，LSTM 已经学会了只有最近的天气才重要——不超过两三天。然而，集成梯度减弱的原因不仅仅是这一点。它们也因为**梯度消失问题**而减弱。这个问题发生在反向传播过程中，因为梯度值在每一步都要乘以权重矩阵，所以梯度可以指数级减少到零。

LSTM 被组织在一个非常长的序列中，这使得网络在长期捕捉依赖关系方面越来越无效。幸运的是，这些 LSTM 是**有状态的**，这意味着它们通过利用前一个批次的状态将批次按顺序连接起来。**状态性**确保了从长序列中学习，尽管存在梯度消失问题。这就是为什么当我们观察“假日下午”的归因图时，对于`is_holiday`有负归因，这是预料到没有高峰时段的合理原因。结果证明，9 月 3 日（劳动节）距离前一个假日（独立日）近两个月，而独立日是一个更盛大的节日。模型是否可能捕捉到这些模式呢？

我们可以尝试根据交通模式对假日进行子分类，看看这是否能帮助模型识别它们。我们还可以对以前的天气条件进行滚动汇总，以便模型更容易地捕捉到最近的天气模式。天气模式跨越数小时，因此汇总是直观的，而且更容易解释。解释方法可以为我们指明如何改进模型的方向，当然，改进的空间很大。

接下来，我们将尝试一种基于排列的方法！

# 使用 SHAP 的 KernelExplainer 计算全局和局部归因

排列方法通过对输入进行修改来评估它们将对模型输出产生多大的影响。我们首次在*第四章*，*全局模型无关解释方法*中讨论了这一点，但如果你还记得，有一个联盟框架可以执行这些排列，从而为不同特征的联盟产生每个特征的边际贡献的平均值。这个过程的结果是**Shapley** **值**，它具有如加法和对称性等基本数学性质。不幸的是，对于不是特别小的数据集，Shapley 值的计算成本很高，所以 SHAP 库有近似方法。其中一种方法就是`KernelExplainer`，我们在*第四章*中也解释了它，并在*第五章*，*局部模型无关解释方法*中使用它。它使用加权局部线性回归来近似 Shapley 值，就像 LIME 所做的那样。

## 为什么使用 KernelExplainer？

我们有一个深度学习模型，那么为什么我们不使用 SHAP 的`DeepExplainer`，就像我们在*第七章*，*可视化卷积神经网络*中使用的 CNN 一样呢？DeepExplainer 将 DeepLIFT 算法改编来近似 Shapley 值。它与任何用于表格数据、CNN 和具有嵌入层的 RNN（例如用于 NLP 分类器或用于检测基因组序列的 RNN）都配合得非常好。对于多元时间序列，它变得更加复杂，因为 DeepExplainer 不知道如何处理输入的三维数组。即使它知道，它还包括了之前时间步的数据，因此你无法在不考虑之前时间步的情况下对单个时间步进行排列。例如，如果排列规定温度降低五度，这不应该影响之前数小时内的所有温度吗？如果温度降低 20 度呢？这不意味着它可能处于不同的季节，并且天气完全不同——也许还有更多的云和雪？

SHAP 的`KernelExplainer`可以接收任何任意的黑盒`predict`函数。它还对输入维度做出了一些假设。幸运的是，我们可以在它排列之前更改输入数据，使其对`KernelExplainer`来说，就像它正在处理一个表格数据集一样。任意的`predict`函数不必简单地调用模型的`predict`函数——它可以在输入和输出过程中更改数据！

## 定义一个策略使其与多元时间序列模型一起工作

为了模仿基于排列输入数据的可能过去天气模式，我们可以创建一个生成模型或类似的东西。这种策略将帮助我们生成适合排列时间步的多种过去时间步，以及为特定类别生成图像。尽管这可能会导致更准确的预测，但我们不会使用这种策略，因为它非常耗时。

相反，我们将使用`gen_all`生成器中的现有示例来找到最适合排列输入的时间序列数据。我们可以使用距离度量来找到最接近排列输入的那个。然而，我们必须设置一些限制，因为如果排列是在周六早上 5 点，温度为 27 摄氏度，云量为 90%，那么最接近的观察可能是在周五早上 7 点，但无论天气交通如何，它都会完全不同。因此，我们可以实现一个过滤器函数，确保它只找到相同`dow`、`is_holiday`和`hr`的最近观察。过滤器函数还可以清理排列样本，删除或修改模型中任何无意义的部分，例如分类特征的连续值：

![图片](img/B18406_09_09.png)

图 9.9：排列近似策略

*图 9.9*展示了使用距离函数找到修改后的排列样本最近观察的过程。此函数返回最近的观察索引，但模型不能对单个观察（或时间步）进行预测，因此它需要其过去直到`lookback`窗口的小时历史。因此，它从生成器中检索正确的批次并对其进行预测，但预测将处于不同的尺度上，因此它们需要使用`y_scaler`进行逆变换。一旦`predict`函数迭代了所有样本并对它们进行了预测和缩放，它将它们发送回`KernelExplainer`，该工具输出它们的 SHAP 值。

## 为排列近似策略打下基础

您可以定义一个自定义的过滤器函数（`filt_fn`）。它接受一个包含整个数据集（`X_df`）的`pandas` DataFrame，您希望从中过滤，以及用于过滤的排列样本（`x`）和`lookback`窗口的长度。

该函数还可以修改排列后的样本。在这种情况下，我们必须这样做，因为模型中有许多特征是离散的，但排列过程使它们变得连续。正如我们之前提到的，所有过滤操作所做的只是通过限制选项来保护距离函数，防止它找到排列样本的非合理最近样本：

```py
def filt_fn(X_df, x, lookback):
    x_ = x.copy()
    x_[0] = round(x_[0]) #round dow
    x_[1] = round(x_[1]) #round hr
    x_[6] = round(x_[6]) #round is_holiday
    if x_[1] < 0:#if hr < 0
        x_[1] = 24 + x_[1]
        x_[0] = x_[0] – 1  #make it previous day
    if x_[0] < 0:#if dow < 0
        x_[0] = 7 + x_[0] #make it previous week
        X_filt_df = X_df[
            (X_df.index >= lookback) & (X_df.dow==x_[0]) &\
            (X_df.hr==x_[1]) & (X_df.is_holiday==x_[6]) &\
            (X_df.temp-5<=x_[2]) & (X_df.temp+5>=x_[2])
        ]
    return X_filt_df, x_ 
```

如果你参考 *图 9.9*，在过滤器函数之后，我们接下来应该定义的是距离函数。我们可以使用 `scipy.spatial.distance.cdist` 接受的任何标准距离函数，例如“欧几里得”、“余弦”或“汉明”。这些标准距离函数的问题在于，它们要么与连续变量很好地工作，要么与离散变量很好地工作，但不能两者都很好地工作。我们在这个数据集中两者都有！

幸运的是，存在一些可以处理这两种情况的替代方案，例如**异构欧几里得-重叠度量**（**HEOM**）和**异构值差异度量**（**HVDM**）。这两种方法根据变量的性质应用不同的距离度量。HEOM 使用归一化的欧几里得距离 ![](img/B18406_09_001.png) 对连续变量，对离散变量使用“重叠”距离；即如果相同则为零距离，否则为 1。

HVDM 更复杂，因为对于连续变量，它是两个值之间的绝对距离，除以所涉及特征的均方根的四倍 ![](img/B18406_09_002.png))，这是一个处理异常值的好距离度量。对于离散变量，它使用归一化的**值差异度量**，这是基于两个值的条件概率之间的差异。

尽管 HVDM 对于具有许多连续值的集合数据集比 HEOM 更好，但在这种情况下却是过度设计。一旦数据集通过星期几 (`dow`) 和小时 (`hr`) 过滤，剩余的离散特征都是二进制的，因此“重叠”距离是理想的，而对于剩下的三个连续特征（`temp`、`rain_1h`、`snow_1h` 和 `cloud_coverage`），欧几里得距离应该足够。`distython` 有一个 `HEOM` 距离方法，它只需要一个背景数据集 (`X_df.values`) 和分类特征的索引 (`cat_idxs`)。我们可以使用 `np.where` 命令编程识别这些特征。

如果你想验证这些是否是正确的，请在单元格中运行 `print(cat_idxs)`。只有索引 2、3、4 和 5 应该被省略：

```py
cat_idxs = np.where(traffic_df.drop(['traffic_volume'],\
                                    axis=1).dtypes != np.float64)[0]
heom_dist = HEOM(X_df.values, cat_idxs)
print(cat_idxs) 
```

现在，我们可以创建一个 `lambda` 函数，将 *图 9.9* 中描述的所有内容放在一起。它利用一个名为 `approx_predict_ts` 的函数来处理整个流程。它接受我们的过滤器函数 (`filt_fn`)、距离函数 (`heom_dist.heom`)、生成器 (`gen_all`) 和拟合的模型 (`lstm_traffic_mdl`)，并将它们链接在一起，如 *图 9.9* 所示。它还使用我们的缩放器 (`X_scaler` 和 `y_scaler`) 对数据进行缩放。距离是在转换后的特征上计算的，以提高准确性，并且预测在输出过程中进行反向转换：

```py
predict_fn = lambda X: mldatasets.approx_predict_ts(
    X, X_df,
    gen_all,
    lstm_traffic_mdl,
    dist_metric=heom_dist.heom,
    lookback=lookback,
    filt_fn=filt_fn,
    X_scaler=X_scaler,
    y_scaler=y_scaler
) 
```

我们现在可以使用`KernelExplainer`的预测函数，但应该在最能代表施工队预期工作条件的样本上进行；也就是说，他们计划在 3 月到 11 月工作，最好是工作日和交通量低的时间。为此，让我们创建一个只包括这些月份的 DataFrame（`working_season_df`），并使用`predict_fn`和 DataFrame 的 k-means 作为背景数据初始化一个`KernelExplainer`：

```py
working_season_df =\
    traffic_df[lookback:].drop(['traffic_volume'], axis=1).copy()
working_season_df =\
    working_season_df[(working_season_df.index.month >= 3) &\
                      (working_season_df.index.month <= 11)]
explainer = shap.KernelExplainer(
    predict_fn, shap.kmeans(working_season_df.values, 24)
) 
```

我们现在可以为`working_season_df` DataFrame 的随机观测值集生成 SHAP 值。

## 计算 SHAP 值

我们将从其中采样 48 个观测值。`KernelExplainer`相当慢，尤其是在使用我们的近似方法时。为了获得最佳的全球解释，最好使用大量的观测值，同时也要使用高`nsamples`，这是在解释每个预测时需要重新评估模型次数的数量。不幸的是，如果每种都有 50 个，那么解释器运行起来将需要数小时，这取决于你的可用计算资源，所以我们将会使用`nsamples=10`。你可以查看 SHAP 的进度条并相应地调整。一旦完成，它将生成包含 SHAP 值的特征重要性`summary_plot`：

```py
X_samp_df = working_season_df.sample(80, random_state=rand)
shap_values = explainer.shap_values(X_samp_df, nsamples=10)
shap.summary_plot(shap_values, X_samp_df) 
```

上述代码绘制了以下图形中显示的摘要。不出所料，`hr`和`dow`是最重要的特征，其次是某些天气特征。奇怪的是，温度和降雨似乎并没有在预测中起到作用，但晚春到秋季可能不是一个显著因素。或者，也许更多的观测值和更高的`nsample`将产生更好的全球解释：

![包含图形用户界面的图片描述自动生成](img/B18406_09_10.png)

图 9.10：基于 48 个采样观测值产生的 SHAP 摘要图

我们可以用上一节中选择的感兴趣实例进行相同的操作，以进行局部解释。让我们遍历所有这些数据点。然后，我们可以生成一个单一的`shap_values`，但这次使用`nsamples=80`，然后为每个生成一个`force_plot`：

```py
for s in range(len(samples)):
    print('Local Force Plot for "{}"'.format(sample_names[s]))
    shap_values_single = explainer.shap_values(
        datapoints[i], nsamples=80)
    shap.force_plot(
    explainer.expected_value,
    shap_values_single[0],
    samples[s],
    matplotlib=True
)
    plt.show() 
```

上述代码生成了*图 9.11*中显示的图形。“假日午后”的小时数(`hr=16`)推动预测值升高，而它是星期一(`dow=0`)和假日(`is_holiday=1`)的事实则推动预测值向相反方向移动。另一方面，“高峰早晨”主要由于小时数(`hr=8.0`)而处于高峰状态，但它有高`cloud_coverage`，肯定的`weather_Clouds`，而且没有降雨(`rain_1h=0.0`)。最后，“炎热的周六”由于星期数(`dow=5`)推动值降低，但异常高的值主要由于它是中午没有降雨和云层。奇怪的是，高于正常温度不是影响因素之一：

![时间线描述自动生成](img/B18406_09_11.png)

图 9.11：使用 SHAP 值和 nsamples=80 生成的力图，用于假日午后、高峰早晨和炎热周六

使用 SHAP 基于博弈论的方法，我们可以衡量现有观察值的排列如何使预测结果在许多可能特征联盟中边际变化。然而，这种方法可能非常有限，因为我们的背景数据中现有的方差塑造了我们对于结果方差的理解。

在现实世界中，*变异性通常由数据中未表示的内容决定——但可能性极小*。例如，在明尼阿波利斯夏季凌晨 5 点之前达到 25°C（77°F）并不常见，但随着全球变暖，它可能会变得频繁，因此我们想要模拟它如何影响交通模式。预测模型特别容易受到风险的影响，因此模拟是评估这种不确定性的关键解释组成部分。对不确定性的更好理解可以产生更稳健的模型，并直接指导决策。接下来，我们将讨论我们如何使用敏感性分析方法产生模拟。

# 使用因素优先级识别有影响力的特征

**莫里斯方法**是几种全局敏感性分析方法之一，范围从简单的**分数因子**到复杂的**蒙特卡洛过滤**。莫里斯位于这个光谱的某个位置，分为两个类别。它使用**一次一个采样**，这意味着在连续模拟之间只有一个值发生变化。它也是一个**基本效应**（**EE**）方法，这意味着它不量化模型中因素的确切效应，而是衡量其重要性和与其他因素的关系。顺便说一句，**因素**只是另一个在应用统计学中常用的特征或变量的名称。为了与相关理论保持一致，我们将在本节和下一节中使用这个词汇。

莫里斯的另一个特性是，它比我们接下来将要研究的基于方差的计算方法更节省计算资源。它可以提供比回归、导数或基于因子的简单且成本较低的方法更多的见解。它不能精确量化效应，但可以识别那些具有可忽略或交互效应的效应，这使得它成为在因素数量较少时筛选因素的理想方法。筛选也被称为**因素优先级**，因为它可以根据它们的分类来优先考虑因素。

## 计算莫里斯敏感性指数

莫里斯方法推导出与单个因素相关联的基本效应分布。每个基本效应分布都有一个平均值（*µ*）和标准差（*σ*）。这两个统计数据有助于将因素映射到不同的分类中。当模型非单调时，平均值可能是负数，因此莫里斯方法的一个变体通过绝对值（*µ*^*）进行调整，以便更容易解释。我们在这里将使用这种变体。

现在，让我们将此问题的范围限制在更易于管理的范围内。施工队将面临的道路交通不确定性将持续从 5 月到 10 月，周一至周五，晚上 11 点到凌晨 5 点。因此，我们可以从`working_season_df` DataFrame 中进一步提取子集，以生成一个工作小时 DataFrame（`working_hrs_df`），我们可以对其进行`describe`。我们将包括 1%、50%和 99%的百分位数，以了解中位数和异常值所在的位置：

```py
working_hrs_df = working_season_df[
    (working_season_df.dow < 5)
    & ((working_season_df.hr < 5) | (working_season_df.hr > 22))
]
working_hrs_df.describe(percentiles=[.01,.5,.99]).transpose() 
```

上述代码生成了*图 9.12*中的表格。我们可以使用这张表格来提取我们在模拟中使用的特征范围。通常，我们会使用超过现有最大值或最小值的合理值。对于大多数模型，任何特征值都可以在其已知限制之外增加或减少，并且由于模型学习到了单调关系，它可以推断出合理的结局。例如，它可能学习到超过某个点的降雨量将逐渐减少交通。那么，假设你想模拟每小时 30 毫米的严重洪水；它可以准确预测无交通：

![](img/B18406_09_12.png)

图 9.12：施工队计划工作期间的汇总统计

然而，因为我们使用的是从历史值中采样的预测近似方法，所以我们受到如何将边界推到已知范围之外的限制。因此，我们将使用 1%和 99%的百分位数值作为我们的限制。我们应该注意，这对于任何发现来说都是一个重要的注意事项，特别是对于可能超出这些限制的特征，例如`temp`、`rain_1h`和`snow_1h`。

从*图 9.12*的总结中，我们还需要注意的一点是，许多与天气相关的二元特征非常稀疏。你可以通过它们的极低平均值来判断。每个添加到敏感性分析模拟中的因素都会减慢其速度，因此我们只会选择前三个；即`weather_Clear`、`weather_Clouds`和`weather_Rain`。这些因素与其他六个因素一起在“问题”字典（`morris_problem`）中指定，其中包含它们的对应`names`、`bounds`和`groups`。现在，`bounds`是关键，因为它表示每个因素将模拟哪些值范围。我们将使用[0,4]（周一至周五）作为`dow`的值，以及[-1,4]（晚上 11 点到凌晨 4 点）作为`hr`的值。过滤器函数自动将负小时转换为前一天的小时，因此周二的一 1 相当于周一的 23。其余的界限是由百分位数确定的。请注意，`groups`中的所有因素都属于同一组，除了三个天气因素：

```py
morris_problem = {
    # There are nine variables
    'num_vars': 10,
    # These are their names
    'names': ['dow', 'hr', 'temp', 'rain_1h', 'snow_1h',\
              'cloud_coverage', 'is_holiday', 'weather_Clear',\
              'weather_Clouds', 'weather_Rain'],
    # Plausible ranges over which we'll move the variables
    'bounds': [
        [0, 4], # dow Monday - Firday
        [-1, 4], # hr
        [-12, 25.], # temp (C)
        [0., 3.1], # rain_1h
        [0., .3], # snow_1h
        [0., 100.], # cloud_coverage
        [0, 1], # is_holiday
        [0, 1], # weather_Clear
        [0, 1], # weather_Clouds
        [0, 1] # weather_Rain
    ],
    # Only weather is grouped together
    'groups': ['dow', 'hr', 'temp', 'rain_1h', 'snow_1h',\
                'cloud_coverage', 'is_holiday', 'weather', 'weather',\
                'weather']
} 
```

一旦定义了字典，我们就可以使用 `SALib` 的 `sample` 方法生成 Morris 方法样本。除了字典外，它还需要轨迹数量（`256`）和级别（`num_levels=4`）。该方法使用因素和级别的网格来构建输入随机逐个移动的轨迹（**OAT**）。这里需要注意的重要一点是，更多的级别会增加这个网格的分辨率，可能使分析更好。然而，这可能会非常耗时。最好从轨迹数量和级别之间的比例 25:1 或更高开始。

然后，你可以逐步降低这个比例。换句话说，如果你有足够的计算能力，你可以让 `num_levels` 与轨迹数量相匹配，但如果你有这么多可用的计算能力，你可以尝试 `optimal_trajectories=True`。然而，鉴于我们有组，`local_optimization` 必须设置为 `False`。`sample` 的输出是一个数组，每个因素一列，(*G* + 1) × *T* 行（其中 *G* 是组数，*T* 是轨迹数）。我们有八个组，256 个轨迹，所以 `print` 应该输出一个 2,304 行 10 列的形状：

```py
morris_sample = ms.sample(morris_problem, 256,\
                          num_levels=4, seed=rand)
print(morris_sample.shape) 
```

由于 `predict` 函数只与 15 个因素一起工作，我们应该修改样本，用零填充剩余的五个因素。我们使用零，因为这这些特征的中位数。中位数最不可能增加交通量，但你应该根据具体情况调整默认值。如果你还记得我们 **第二章** 的 **心血管疾病** （**CVD**）示例，*可解释性关键概念*，增加 CVD 风险的特性值有时是最小值或最大值。

`np.hstack` 函数可以将数组水平拼接，使得前八个因素之后跟着三个零因素。然后，有一个孤独的第九个样本因素对应于 `weather_Rain`，接着是两个零因素。结果数组应该与之前一样行数，但列数为 15：

```py
morris_sample_mod = np.hstack(
    (
        morris_sample[:,0:9],
        np.zeros((morris_sample.shape[0],3)),
        morris_sample[:,9:10],
        np.zeros((morris_sample.shape[0],2))
    )
)
print(morris_sample_mod.shape) 
```

被称为 `morris_sample_mod` 的 `numpy` 数组现在以我们的 `predict` 函数可以理解的形式包含了 Morris 样本。如果这是一个在表格数据集上训练过的模型，我们就可以直接利用模型的 `predict` 函数。然而，就像我们使用 SHAP 一样，我们必须使用近似方法。这次，我们不会使用 `predict_fn`，因为我们想在 `approx_predict_ts` 中设置一个额外的选项，`progress_bar=True`。其他一切都将保持不变。进度条将很有用，因为这可能需要一段时间。运行单元格，休息一下喝杯咖啡：

```py
morris_preds = mldatasets.approx_predict_ts(
    morris_sample_mod,
    X_df,
    gen_all,
    lstm_traffic_mdl,
    filt_fn=filt_fn,
    dist_metric=heom_dist.heom,
    lookback=lookback,
    X_scaler=X_scaler,
    y_scaler=y_scaler,
    progress_bar=True
) 
```

要使用`SALib`的`analyze`函数进行敏感性分析，你需要你的问题字典（`morris_problem`），原始的 Morris 样本（`morris_sample`），以及我们用这些样本生成的预测（`morris_preds`）。还有一个可选的置信区间水平参数（`conf_level`），但默认的 0.95 是好的。它使用重采样来计算这个置信水平，默认为 1,000。这个设置也可以通过可选的`num_resamples`参数来改变：

```py
morris_sensitivities = ma.analyze(
    morris_problem, morris_sample, morris_preds,\
    print_to_console=False
) 
```

## 分析基本影响

`analyze`将返回一个包含 Morris 敏感性指数的字典，包括平均值（*µ*）和标准差（*σ*）的基本影响，以及平均值（*µ*^*）的绝对值。在表格格式中更容易欣赏这些值，这样我们就可以将它们放入 DataFrame 中，并根据*µ*^*排序和着色，*µ*^*可以解释为因素的整体重要性。另一方面，*σ*表示因素与其它因素的交互程度：

```py
morris_df = pd.DataFrame(
    {
        'features':morris_sensitivities['names'],
        'μ':morris_sensitivities['mu'],
        'μ*':morris_sensitivities['mu_star'],
        'σ':morris_sensitivities['sigma']
    }
)
morris_df.sort_values('μ*', ascending=False).style\
    .background_gradient(cmap='plasma', subset=['μ*']) 
```

前面的代码输出了*图 9.13*中展示的 DataFrame。你可以看出`is_holiday`是其中最重要的因素之一，至少在问题定义中指定的范围（`morris_problem`）内是这样。还有一点需要注意，天气确实有绝对的基本影响，但交互效应并不确定。组别很难评估，尤其是当它们是稀疏的二进制因素时：

![时间线  自动生成的描述](img/B18406_09_13.png)

图 9.13：因素的基本影响分解

前面的图中的 DataFrame 不是可视化基本影响的最佳方式。当因素不多时，更容易绘制它们。`SALib`提供了两种绘图方法。水平条形图（`horizontal_bar_plot`）和协方差图（`covariance_plot`）可以并排放置。协方差图非常好，但它没有注释它所界定的区域。我们将在下一节中了解这些。因此，仅出于教学目的，我们将使用`text`来放置注释：

```py
fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12,8))
mp.horizontal_bar_plot(ax0, morris_sensitivities, {})
mp.covariance_plot(ax1, morris_sensitivities, {})
ax1.text(
    ax1.get_xlim()[1] * 0.45, ax1.get_ylim()[1] * 0.75,\
    'Non-linear and/or-monotonic', color='gray',\
    horizontalalignment='center'
)
ax1.text(ax1.get_xlim()[1] * 0.75, ax1.get_ylim()[1] * 0.5,\
    'Almost Monotonic', color='gray', horizontalalignment='center')
ax1.text(ax1.get_xlim()[1] * 0.83, ax1.get_ylim()[1] * 0.2,\
    'Monotonic', color='gray', horizontalalignment='center')
ax1.text(ax1.get_xlim()[1] * 0.9, ax1.get_ylim()[1] * 0.025,
    'Linear', color='gray', horizontalalignment='center') 
```

前面的代码生成了*图 9.14*中显示的图表。左边的条形图按*µ*^*对因素进行排序，而每根从条形中伸出的线表示相应的置信区间。右边的协方差图是一个散点图，*µ*^*位于*x*轴上，*σ*位于*y*轴上。因此，点越往右，它就越重要，而它在图中越往上，它与其它因素的交互就越多，就越不单调。自然地，这意味着那些交互不多且主要单调的因素符合线性回归的假设，如线性性和多重共线性。然而，线性与非线性或非单调之间的范围由*σ*和*µ*^*的比率对角确定：

![图表，散点图  自动生成的描述](img/B18406_09_14.png)

图 9.14：表示基本效应的条形图和协方差图

你可以通过前面的协方差图看出，所有因素都是非线性或非单调的。`hr`无疑是其中最重要的，其次是接下来的两个因素（`dow`和`temp`）相对靠近，然后是`weather`和`is_holiday`。`weather`组没有在图中显示，因为交互性结果不确定，但`cloud_coverage`、`rain_1h`和`snow_1h`的交互性比它们单独重要得多。

基本效应帮助我们理解如何根据它们对模型结果的影响来分类我们的因素。然而，这并不是一个稳健的方法来正确量化它们的影响或由因素相互作用产生的影响。为此，我们必须转向使用概率框架分解输出方差并将其追溯到输入的基于方差的全局方法。这些方法包括**傅里叶振幅敏感性测试**（**FAST**）和**Sobol**。我们将在下一节研究后一种方法。

# 使用因素固定量化不确定性和成本敏感性

使用 Morris 指数，很明显，所有因素都是非线性或非单调的。它们之间有很高的交互性——正如预期的那样！气候因素（`temp`、`rain_1h`、`snow_1h`和`cloud_coverage`）很可能与`hr`存在多重共线性。在`hr`、`is_holiday`、`dow`和目标之间也存在一些模式。许多这些因素肯定与目标没有单调关系。我们已经知道了这一点。例如，交通在一天中的小时数增加时并不总是增加。情况对一周中的某一天也是如此！

然而，我们不知道`is_holiday`和`temp`对模型的影响程度，尤其是在机组人员的工作时间内，这是一个重要的见解。话虽如此，使用 Morris 指数进行因素优先级排序通常被视为起点或“第一设置”，因为一旦确定存在交互效应，最好是解开它们。为此，有一个“第二设置”，称为**因素固定**。我们可以量化方差，通过这样做，量化所有因素带来的不确定性。

只有**基于方差的方法**才能以统计严谨的方式量化这些效应。**Sobol 敏感性分析**是这些方法之一，这意味着它将模型的输出方差分解成百分比，并将其归因于模型的输入和交互。像 Morris 一样，它有一个采样步骤，以及一个敏感性指数估计步骤。

与 Morris 不同，采样不遵循一系列级别，而是遵循输入数据的分布。它使用**准蒙特卡洛方法**，在超空间中采样点，这些点遵循输入的概率分布。**蒙特卡洛方法**是一系列执行随机采样的算法，通常用于优化或模拟。它们寻求在用蛮力或完全确定性的方法无法解决的问题上的捷径。蒙特卡洛方法在敏感性分析中很常见，正是出于这个原因。准蒙特卡洛方法有相同的目标。然而，它们收敛得更快，因为它们使用确定性低偏差序列而不是使用伪随机序列。Sobol 方法使用**Sobol 序列**，由同一位数学家设计。我们将使用另一种从 Sobol 派生出的采样方案，称为 Saltelli 的。

一旦生成样本，蒙特卡洛估计器就会计算基于方差的敏感性指数。这些指数能够量化非线性非加性效应和第二阶指数，这些指数与两个因素之间的相互作用相关。Morris 可以揭示模型中的交互性，但不能精确地说明它是如何表现的。Sobol 可以告诉你哪些因素在相互作用以及相互作用的程度。

## 生成和预测 Saltelli 样本

要使用`SALib`开始 Sobol 敏感性分析，我们必须首先定义一个问题。我们将与 Morris 做同样的事情。这次，我们将减少因素，因为我们意识到`weather`分组导致了不确定的结果。我们应该包括所有天气因素中最稀疏的；即`weather_Clear`。由于 Sobol 使用概率框架，将`temp`、`rain_1h`和`cloud_coverage`的范围扩展到它们的最大和最小值是没有害处的，如图*9.12*所示：

```py
sobol_problem = {
    'num_vars': 8,
    'names': ['dow', 'hr', 'temp', 'rain_1h', 'snow_1h',
              'cloud_coverage', 'is_holiday', 'weather_Clear'],
    'bounds': [
        [0, 4], # dow Monday through Friday
        [-1, 4], # hr
        [-3., 31.], # temp (C)
        [0., 21.], # rain_1h
        [0., 1.6], # snow_1h
        [0., 100.], # cloud_coverage
        [0, 1], # is_holiday
        [0, 1] # weather_Clear
      ],
    'groups': None
} 
```

生成样本看起来也应该很熟悉。Saltelli 的`sample`函数需要以下内容：

+   问题陈述（`sobol_problem`）

+   每个因素要生成的样本数量（`300`）

+   第二阶索引以进行计算（`calc_second_order=True`）

由于我们想要交互作用，`sample`的输出是一个数组，其中每一列代表一个因素，有![](img/B18406_09_003.png)行（其中*N*是样本数量，*F*是因素数量）。我们有八个因素，每个因素有 256 个样本，所以`print`应该输出 4,608 行和 8 列的形状。首先，我们将像之前一样使用`hstack`修改它，添加 7 个空因素以进行预测，从而得到 15 列：

```py
saltelli_sample = ss.sample(
    sobol_problem, 256, calc_second_order=True, seed=rand
)
saltelli_sample_mod = np.hstack(
    (saltelli_sample, np.zeros((saltelli_sample.shape[0],7)))
)
print(saltelli_sample_mod.shape) 
```

现在，让我们对这些样本进行预测。这可能需要一些时间，所以又是咖啡时间：

```py
saltelli_preds = mldatasets.pprox._predict_ts(
    saltelli_sample_mod,
    X_df,
    gen_all,
    lstm_traffic_mdl,
    filt_fn=filt_fn,
    dist_metric=heom_dist.heom,
    lookback=lookback,
    X_scaler=X_scaler,
    y_scaler=y_scaler,
    progress_bar=True
) 
```

## 执行 Sobol 敏感性分析

对于 Sobol 敏感性分析（`analyze`），你所需要的只是一个问题陈述（`sobol_problem`）和模型输出（`saltelli_preds`）。但是预测并不能讲述不确定性的故事。当然，预测的交通流量有方差，但只有当交通量超过 1,500 时，这个问题才会出现。不确定性是你想要与风险或回报、成本或收入、损失或利润相关联的东西——一些你可以与你问题相关联的实质性东西。

首先，我们必须评估是否存在任何风险。为了了解样本中的预测交通量是否在工作时间内超过了无建设阈值，我们可以使用`print(max(saltelli_preds[:,0]))`。最大交通水平应该在 1,800-1,900 左右，这意味着至少存在一些风险，即建筑公司将会支付罚款。我们不必使用预测（`saltelli_preds`）作为模型的输出，我们可以创建一个简单的二进制数组，当它超过 1,500 时为 1，否则为 0。我们将称之为`costs`，然后使用它运行`analyze`函数。注意，这里也设置了`calc_second_order=True`。如果`sample`和`analyze`没有一致的设置，它将抛出一个错误。与 Morris 一样，有一个可选的置信区间水平参数（`conf_level`），但默认的 0.95 是好的：

```py
costs = np.where(saltelli_preds > 1500, 1,0)[:,0]
factor_fixing_sa = sa.analyze(
    sobol_problem,
    costs,
    calc_second_order=True,
    print_to_console=False
) 
```

`analyze`将返回一个包含 Sobol 敏感性指数的字典，包括一阶（`S1`）、二阶（`S2`）和总阶（`ST`）指数，以及总置信区间（`ST_conf`）。这些指数对应于百分比，但除非模型是加性的，否则总数不一定相加。在表格格式中更容易欣赏这些值，这样我们可以将它们放入 DataFrame 中，并根据总数进行排序和着色，总数可以解释为因素的整体重要性。然而，我们将省略二阶指数，因为它们是二维的，类似于相关图：

```py
sobol_df = pd.DataFrame(
    {
        'features':sobol_problem['names'],
        '1st':factor_fixing_sa['S1'],
        'Total':factor_fixing_sa['ST'],
        'Total Conf':factor_fixing_sa['ST_conf'],
        'Mean of Input':saltelli_sample.mean(axis=0)[:8]
    }
)
sobol_df.sort_values('Total', ascending=False).style
    .background_gradient(cmap='plasma', subset=['Total']) 
```

上一段代码输出了*图 9.15*中展示的 DataFrame。你可以看出`temp`和`is_holiday`至少在问题定义中指定的边界内排在前面四位。另一个需要注意的事情是`weather_Clear`确实对其自身有更大的影响，但`rain_1h`和`cloud_coverage`似乎对潜在成本没有影响，因为它们的总一阶指数为零：

![时间线描述自动生成](img/B18406_09_15.png)

图 9.15：八个因素的 Sobol 全局敏感性指数

关于一阶值的一些有趣之处在于它们有多低，这表明交互作用占模型输出方差的大部分。我们可以很容易地使用二阶索引来证实这一点。这些索引和一阶索引的组合加起来就是总数：

```py
S2 = factor_fixing_sa['S2']
divnorm = TwoSlopeNorm(vmin=S2.min(), vcenter=0, vmax=S2.max())
sns.heatmap(S2, center=0.00, norm=divnorm, cmap='coolwarm_r',\
            annot=True, fmt ='.2f',\
            xticklabels=sobol_problem['names'],\
            yticklabels=sobol_problem['names']) 
```

上一段代码输出了*图 9.16*中的热图：

![图表，瀑布图 描述自动生成](img/B18406_09_16.png)

图 9.16：八个因素的 Sobol 二阶指数

在这里，您可以知道`is_holiday`和`weather_Clear`是两个对输出方差贡献最大的因素，其绝对值最高为 0.26。`dow`和`hr`与所有因素都有相当大的相互作用。

## 引入一个现实成本函数

现在，我们可以创建一个成本函数，它接受我们的输入（`saltelli_sample`）和输出（`saltelli_preds`），并计算双城将对建筑公司罚款多少，以及额外的交通可能产生的任何额外成本。

如果输入和输出都在同一个数组中，这样做会更好，因为我们需要从两者中获取详细信息来计算成本。我们可以使用`hstack`将样本及其对应的预测结果连接起来，生成一个包含八个列的数组（`saltelli_sample_preds`）。然后我们可以定义一个成本函数，它可以计算包含这些九个列的数组的成本（`cost_fn`）：

```py
#Join input and outputs into a sample+prediction array
saltelli_sample_preds = np.hstack((saltelli_sample, saltelli_preds)) 
```

我们知道，对于任何样本预测，半容量阈值都没有超过，所以我们甚至不需要在函数中包含每日罚款。除此之外，罚款是每辆超过每小时无施工阈值的车辆 15 美元。除了这些罚款之外，为了能够按时离开，建筑公司估计额外的成本：如果凌晨 4 点超过阈值，额外工资为 1,500 美元，周五额外 4,500 美元以加快设备移动速度，因为周末不能停在高速公路的路肩上。一旦我们有了成本函数，我们就可以遍历组合数组（`saltelli_sample_preds`），为每个样本计算成本。列表推导可以有效地完成这项工作：

```py
#Define cost function
def cost_fn(x):
    cost = 0
    if x[8] > 1500:
        cost = (x[8] - 1500) * 15
    if round(x[1]) == 4:
        cost = cost + 1500
        if round(x[0]) == 4:
            cost = cost + 4500
    return cost
#Use list comprehension to compute costs for sample+prediction array
costs2 = np.array([cost_fn(xi) for xi in saltelli_sample_preds])
#Print total fines for entire sample predictions
print('Total Fines: $%s' % '{:,.2f}'.format(sum(costs2))) 
```

`print`语句应该输出一个介于 17 万美元和 20 万美元之间的成本。但不必担心！建筑队每年只计划在现场工作大约 195 天，每天 5 小时，总共 975 小时。然而，有 4,608 个样本，这意味着由于交通过多，几乎有 5 年的预测成本。无论如何，计算这些成本的目的在于了解它们与模型输入的关系。更多的样本年意味着更紧密的置信区间：

```py
factor_fixing2_sa = sa.analyze(
    sobol_problem, costs2, calc_second_order=True,
    print_to_console=False
) 
```

现在，我们可以再次进行分析，但使用`costs2`，并将分析保存到`factor_fixing2_sa`字典中。最后，我们可以使用这个字典的值生成一个新的排序和彩色编码的 DataFrame，就像我们之前为*图 9.15*所做的那样，这将生成*图 9.17*中的输出。

如您从*图 9.17*中可以看出，一旦实际成本被考虑在内，`dow`、`hr`和`is_holiday`成为更具风险的因素，而与*图 9.15*相比，`snow_1h`和`temp`变得不那么相关：

![时间线 描述自动生成](img/B18406_09_17.png)

图 9.17：使用现实成本函数计算八个因素的 Sobol 全局敏感性指数

用表格难以欣赏的是敏感性指数的置信区间。为此，我们可以使用条形图，但首先，我们必须将整个字典转换成一个 DataFrame，以便`SALib`的绘图函数可以绘制它：

```py
factor_fixing2_df = factor_fixing2_sa.to_df()
fig, (ax) = plt.subplots(1,1, figsize=(15, 7))
sp.plot(factor_fixing2_df[0], ax=ax) 
```

前面的代码生成了*图 9.18*中的条形图。`dow`的 95%置信区间比其他重要因素大得多，考虑到一周中各天之间的差异很大，这并不令人惊讶。另一个有趣的见解是`weather_Clear`具有负一阶效应，因此正的总阶指数完全归因于二阶指数，这扩大了置信区间：

![图表，散点图  自动生成的描述](img/B18406_09_18.png)

图 9.18：使用现实成本函数绘制的条形图，包含 Sobol 敏感性总阶指数及其置信区间

要了解如何，让我们再次绘制*图 9.16*所示的散点图，但这次使用`factor_fixing2_sa`而不是`factor_fixing_sa`。*图 9.19*中的散点图应该描绘出模型中成本的现实反映：

![图表，瀑布图  自动生成的描述](img/B18406_09_19.png)

图 9.19：在考虑更现实的成本函数时，七个因素的 Sobol 二阶指数

前面的散点图显示了与*图 9.16*中相似的显著交互，但由于有更多的阴影，它们更加细腻。很明显，`weather_Clear`与`is_holiday`结合时具有放大作用，而对`dow`和`hr`则有调和作用。

# 任务完成

任务是训练一个交通预测模型，并了解哪些因素会创造不确定性，并可能增加建筑公司的成本。我们可以得出结论，潜在的 35,000 美元/年的罚款中有很大一部分可以归因于`is_holiday`因素。因此，建筑公司应该重新考虑工作假日。三月至十一月之间只有七个或八个假日，由于罚款，它们可能比在几个星期日工作成本更高。考虑到这个警告，任务已经成功，但仍有很多改进的空间。

当然，这些结论是针对`LSTM_traffic_168_compact1`模型——我们可以将其与其他模型进行比较。尝试将笔记本开头的`model_name`替换为`LSTM_traffic_168_compact2`，这是一个同样小巧但显著更稳健的模型，或者`LSTM_traffic_168_optimal`，这是一个更大但表现略好的模型，并重新运行笔记本。或者浏览名为`Traffic_compact2`和`Traffic_optimal`的笔记本，这些笔记本已经使用相应的模型重新运行。你会发现，可以训练和选择能够更好地管理不确定输入的模型。话虽如此，改进并不总是通过简单地选择更好的模型就能实现。

例如，可以进一步深入探讨的是`temp`、`rain_1h`和`snow_1h`的真正影响。我们的预测近似方法排除了 Sobol 测试极端天气事件的影响。如果我们修改模型以在单个时间步长上训练聚合的天气特征，并内置一些安全措施，我们就可以使用 Sobol 模拟天气极端情况。而且，敏感性分析的“第三设置”，即因素映射，可以帮助精确指出某些因素值如何影响预测结果，从而进行更稳健的成本效益分析，但这一点我们不会在本章中涉及。

在本书的第二部分，我们探讨了多种解释方法的生态系统：全局和局部；针对特定模型和非特定模型；基于排列和基于敏感度的。对于任何机器学习用例，可供选择的方法并不缺乏。然而，必须强调的是，**没有任何方法是完美的**。尽管如此，它们可以相互补充，以更接近地理解您的机器学习解决方案及其旨在解决的问题。

本章关注预测中的确定性，旨在揭示机器学习社区中的一个特定问题：过度自信。在“可解释性的商业案例”部分的*第一章*，《解释、可解释性、可解释性；以及这一切为什么都重要？》，描述了充斥在人类决策中的许多偏见。这些偏见通常是由对领域知识或我们模型令人印象深刻的成果的过度自信所驱动的。而这些令人印象深刻的成果使我们无法理解我们模型的局限性，随着公众对 AI 的不信任增加，这一点变得更加明显。

正如我们在*第一章*中讨论的，*解释、可解释性、可解释性；以及为什么这一切都很重要？*，机器学习仅用于解决*不完整问题*。否则，我们不如使用在闭环系统中发现的确定性程序编程。解决不完整问题的最佳方法是一个不完整的解决方案，它应该被优化以尽可能多地解决它。无论是通过梯度下降、最小二乘估计还是分割和修剪决策树，机器学习不会产生一个完美泛化的模型。机器学习中的这种不完整性正是我们需要解释方法的原因。简而言之：模型从我们的数据中学习，我们可以从我们的模型中学到很多，但只有当我们解释它们时才能做到！

然而，可解释性并不止于此。模型解释可以驱动决策并帮助我们理解模型的优势和劣势。然而，数据或模型本身的问题有时会使它们变得难以解释。在本书的*第三部分*中，我们将学习如何通过降低复杂性、减轻偏差、设置护栏和增强可靠性来调整模型和训练数据以提高可解释性。

统计学家 George E.P. Box 曾著名地开玩笑说，“*所有模型都是错误的，但有些是有用的*。”也许它们并不总是错误的，但机器学习从业者需要谦逊地接受，即使是高性能模型也应受到审查，并且我们对它们的假设也应受到质疑。机器学习模型的不确定性是可以预期的，不应该是羞耻或尴尬的来源。这使我们得出本章的另一个结论：不确定性伴随着后果，无论是成本还是利润提升，我们可以通过敏感性分析来衡量这些后果。

# 摘要

阅读本章后，你应该了解如何评估时间序列模型的预测性能，知道如何使用集成梯度对他们进行局部解释，以及如何使用 SHAP 产生局部和全局归因。你还应该知道如何利用敏感性分析因子优先级和因子固定来优化任何模型。

在下一章中，我们将学习如何通过特征选择和工程来降低模型的复杂性，使其更具可解释性。

# 数据集和图像来源

+   TomTom，2019 年，交通指数：[`nonews.co/wp-content/uploads/2020/02/TomTom2019.pdf`](https://nonews.co/wp-content/uploads/2020/02/TomTom2019.pdf)

+   UCI 机器学习仓库，2019 年，都市州际交通流量数据集：[`archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume`](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

# 进一步阅读

+   Wilson, D.R. 和 Martinez, T.，1997 年，*改进的异构距离函数*。J. Artif. Int. Res. 6-1\. 第 1-34 页：[`arxiv.org/abs/cs/9701101`](https://arxiv.org/abs/cs/9701101)

+   Morris, M., 1991, *《初步计算实验的因子抽样计划》*. Quality Engineering, 37, 307-310: [`doi.org/10.2307%2F1269043`](https://doi.org/10.2307%2F1269043)

+   Saltelli, A., Tarantola, S., Campolongo, F., and Ratto, M., 2007, *《实践中的敏感性分析：评估科学模型指南》*. Chichester: John Wiley & Sons.

+   Sobol, I.M., 2001, *《非线性数学模型的全球敏感性指数及其蒙特卡洛估计》*. MATH COMPUT SIMULAT, 55(1–3), 271-280: [`doi.org/10.1016/S0378-4754(00)00270-6`](https://doi.org/10.1016/S0378-4754(00)00270-6)

+   Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and S. Tarantola, 2010, *《模型输出的方差敏感性分析：总敏感性指数的设计和估计器》*. Computer Physics Communications, 181(2):259-270: [`doi.org/10.1016/j.cpc.2009.09.018`](https://doi.org/10.1016/j.cpc.2009.09.018)

# 在 Discord 上了解更多

要加入这本书的 Discord 社区——在那里您可以分享反馈、向作者提问，并了解新书发布——请扫描下面的二维码：

`packt.link/inml`

![](img/QR_Code107161072033138125.png)
