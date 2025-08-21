# 第七章：交叉验证和后模型工作流

在本章中，我们将涵盖以下内容：

+   使用交叉验证选择模型

+   K 折交叉验证

+   平衡交叉验证

+   使用 ShuffleSplit 的交叉验证

+   时间序列交叉验证

+   使用 scikit-learn 进行网格搜索

+   使用 scikit-learn 进行随机搜索

+   分类度量

+   回归度量

+   聚类度量

+   使用虚拟估计器比较结果

+   特征选择

+   L1 范数上的特征选择

+   使用 joblib 或 pickle 持久化模型

# 介绍

这也许是最重要的章节。本章所探讨的基本问题如下：

+   我们如何选择一个预测良好的模型？

这就是交叉验证的目的，不管模型是什么。这与传统统计略有不同，传统统计更关心我们如何更好地理解现象。（为什么要限制我对理解的追求？好吧，因为有越来越多的数据，我们不一定能够全部查看、反思并创建理论模型。）

机器学习关注预测以及机器学习算法如何处理新的未见数据并得出预测。即使它看起来不像传统统计，你可以使用解释和领域理解来创建新列（特征）并做出更好的预测。你可以使用传统统计来创建新列。

书的早期，我们从训练/测试拆分开始。交叉验证是许多关键训练和测试拆分的迭代，以最大化预测性能。

本章探讨以下内容：

+   交叉验证方案

+   网格搜索——在估计器中找到最佳参数是什么？

+   评估指标比较 `y_test` 与 `y_pred` ——真实目标集与预测目标集

下面一行包含交叉验证方案 `cv = 10`，用于 `neg_log_lost` 评分机制，该机制由 `log_loss` 指标构建：

```py
cross_val_score(SVC(), X_train, y_train, cv = 10, scoring='neg_log_loss')
```

Scikit-learn 的一部分力量在于在一行代码中包含如此多的信息。此外，我们还将看到一个虚拟估计器，查看特征选择，并保存训练好的模型。这些方法真正使得机器学习成为它所是的东西。

# 使用交叉验证选择模型

我们看到了自动交叉验证，在 第一章，*高性能机器学习 – NumPy* 中的 `cross_val_score` 函数。这将非常相似，除了我们将使用鸢尾花数据集的最后两列作为数据。本节的目的是选择我们可以选择的最佳模型。

在开始之前，我们将定义最佳模型为得分最高的模型。如果出现并列，我们将选择波动最小的模型。

# 准备就绪

在这个配方中，我们将执行以下操作：

+   加载鸢尾花数据集的最后两个特征（列）

+   将数据拆分为训练数据和测试数据

+   实例化两个**k 近邻**（**KNN**）算法，分别设置为三个和五个邻居。

+   对两个算法进行评分

+   选择得分最好的模型

从加载数据集开始：

```py
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,2:]
y = iris.target
```

将数据拆分为训练集和测试集。样本是分层抽样的，书中默认使用这种方法。分层抽样意味着目标变量在训练集和测试集中的比例相同（此外，`random_state`被设置为`7`）：

```py

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,random_state = 7)
```

# 如何操作……

1.  首先，实例化两个最近邻算法：

```py
from sklearn.neighbors import KNeighborsClassifier

kn_3 = KNeighborsClassifier(n_neighbors = 3)
kn_5 = KNeighborsClassifier(n_neighbors = 5)
```

1.  现在，使用`cross_val_score`对两个算法进行评分。查看`kn_3_scores`，这是得分的列表：

```py
from sklearn.model_selection import cross_val_score

kn_3_scores = cross_val_score(kn_3, X_train, y_train, cv=4)
kn_5_scores = cross_val_score(kn_5, X_train, y_train, cv=4)
kn_3_scores

array([ 0.9 , 0.92857143, 0.92592593, 1\. ])
```

1.  查看`kn_5_scores`，这是另一个得分列表：

```py
kn_5_scores

array([ 0.96666667, 0.96428571, 0.88888889, 1\. ])
```

1.  查看两个列表的基本统计信息。查看均值：

```py
print "Mean of kn_3: ",kn_3_scores.mean()
print "Mean of kn_5: ",kn_5_scores.mean()

Mean of kn_3: 0.938624338624
Mean of kn_5: 0.95496031746
```

1.  查看分布，查看标准差：

```py
print "Std of kn_3: ",kn_3_scores.std()
print "Std of kn_5: ",kn_5_scores.std()

Std of kn_3: 0.037152126551
Std of kn_5: 0.0406755710299
```

总体来说，当算法设置为五个邻居时，`kn_5`的表现比三个邻居稍好，但它的稳定性较差（它的得分有点分散）。

1.  现在我们进行最后一步：选择得分最高的模型。我们选择`kn_5`，因为它的得分最高。（该模型在交叉验证下得分最高。请注意，涉及的得分是最近邻的默认准确率得分：正确分类的比例除以所有尝试分类的数量。）

# 它是如何工作的……

这是一个 4 折交叉验证的示例，因为在`cross_val_score`函数中，`cv = 4`。我们将训练数据，或**交叉验证集**（`X_train`），拆分为四个部分，或称折叠。我们通过轮流将每个折叠作为测试集来迭代。首先，折叠 1 是测试集，而折叠 2、3 和 4 一起构成训练集。接下来，折叠 2 是测试集，而折叠 1、3 和 4 是训练集。我们还对折叠 3 和折叠 4 进行类似的操作：

![](img/1f4fc2e4-4e98-4165-9467-bafb11dff24c.png)

一旦我们将数据集拆分为折叠，我们就对算法进行四次评分：

1.  我们在折叠 2、3 和 4 上训练其中一个最近邻算法。

1.  然后我们对折叠 1 进行预测，即测试折。

1.  我们衡量分类准确性：将测试折与该折的预测结果进行比较。这是列表中第一个分类分数。

该过程执行了四次。最终输出是一个包含四个分数的列表。

总体来说，我们进行了整个过程两次，一次用于`kn_3`，一次用于`kn_5`，并生成了两个列表以选择最佳模型。我们从中导入的模块叫做`model_selection`，因为它帮助我们选择最佳模型。

# K 折交叉验证

在寻找最佳模型的过程中，你可以查看交叉验证折叠的索引，看看每个折叠中有哪些数据。

# 准备工作

创建一个非常小的玩具数据集：

```py
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2, 1, 2, 1, 2])
```

# 如何操作……

1.  导入`KFold`并选择拆分的数量：

```py
from sklearn.model_selection import KFold

kf= KFold(n_splits = 4)
```

1.  你可以遍历生成器并打印出索引：

```py
cc = 1
for train_index, test_index in kf.split(X):
 print "Round : ",cc,": ",
 print "Training indices :", train_index,
 print "Testing indices :", test_index
 cc += 1

Round 1 : Training indices : [2 3 4 5 6 7] Testing indices : [0 1]
Round 2 : Training indices : [0 1 4 5 6 7] Testing indices : [2 3]
Round 3 : Training indices : [0 1 2 3 6 7] Testing indices : [4 5]
Round 4 : Training indices : [0 1 2 3 4 5] Testing indices : [6 7]
```

你可以看到，例如，在第一轮中有两个测试索引，`0`和`1`。`[0 1]`构成了第一个折叠。`[2 3 4 5 6 7]`是折叠 2、3 和 4 的组合。

1.  你还可以查看拆分的次数：

```py
kf.get_n_splits()

4
```

分割数为 `4`，这是我们实例化 `KFold` 类时设置的。

# 还有更多...

如果愿意，可以查看折叠数据本身。将生成器存储为列表：

```py
indices_list = list(kf.split(X))
```

现在，`indices_list` 是一个元组的列表。查看第四个折叠的信息：

```py
indices_list[3] #the list is indexed from 0 to 3

(array([0, 1, 2, 3, 4, 5], dtype=int64), array([6, 7], dtype=int64))
```

此信息与前面的打印输出信息相匹配，但它以两个 NumPy 数组的元组形式给出。查看第四个折叠的实际数据。查看第四个折叠的训练数据：

```py
train_indices, test_indices = indices_list[3]

X[train_indices]

array([[1, 2],
 [3, 4],
 [5, 6],
 [7, 8],
 [1, 2],
 [3, 4]])

y[train_indices]

array([1, 2, 1, 2, 1, 2])
```

查看测试数据：

```py
X[test_indices]

array([[5, 6],
 [7, 8]])

y[test_indices]

array([1, 2])
```

# 平衡交叉验证

在将不同折叠中的不同数据集分割时，您可能会想知道：k 折交叉验证中每个折叠中的不同集合可能会非常不同吗？每个折叠中的分布可能会非常不同，这些差异可能导致得分的波动。

对此有一个解决方案，使用分层交叉验证。数据集的子集看起来像整个数据集的较小版本（至少在目标变量方面）。

# 准备工作

创建一个玩具数据集如下：

```py
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
```

# 如何做...

1.  如果我们在这个小型玩具数据集上执行 4 折交叉验证，每个测试折叠将只有一个目标值。可以使用 `StratifiedKFold` 来解决这个问题：

```py
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 4)
```

1.  打印出折叠的索引：

```py
cc = 1
for train_index, test_index in skf.split(X,y):
 print "Round",cc,":",
 print "Training indices :", train_index,
 print "Testing indices :", test_index
 cc += 1

Round 1 : Training indices : [1 2 3 5 6 7] Testing indices : [0 4]
Round 2 : Training indices : [0 2 3 4 6 7] Testing indices : [1 5]
Round 3 : Training indices : [0 1 3 4 5 7] Testing indices : [2 6]
Round 4 : Training indices : [0 1 2 4 5 6] Testing indices : [3 7]
```

观察 `skf` 类的 `split` 方法，即分层 k 折叠分割，具有两个参数 `X` 和 `y`。它试图在每个折叠集中以相同的分布分配目标 `y`。在这种情况下，每个子集都有 50% 的 `1` 和 50% 的 `2`，就像整个目标集 `y` 一样。

# 还有更多...

您可以使用 `StratifiedShuffleSplit` 重新洗牌分层折叠。请注意，这并不会尝试制作具有互斥测试集的四个折叠：

```py
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits = 5,test_size=0.25)

cc = 1
for train_index, test_index in sss.split(X,y):
 print "Round",cc,":",
 print "Training indices :", train_index,
 print "Testing indices :", test_index
 cc += 1

Round 1 : Training indices : [1 6 5 7 0 2] Testing indices : [4 3]
Round 2 : Training indices : [3 2 6 7 5 0] Testing indices : [1 4]
Round 3 : Training indices : [2 1 4 7 0 6] Testing indices : [3 5]
Round 4 : Training indices : [4 2 7 6 0 1] Testing indices : [5 3]
Round 5 : Training indices : [1 2 0 5 4 7] Testing indices : [6 3]
Round 6 : Training indices : [0 6 5 1 7 3] Testing indices : [2 4]
Round 7 : Training indices : [1 7 3 6 2 5] Testing indices : [0 4]
```

这些分割不是数据集的分割，而是随机过程的迭代，每个迭代的训练集大小为整个数据集的 75%，测试集大小为 25%。所有迭代都是分层的。

# 使用 ShuffleSplit 的交叉验证

ShuffleSplit 是最简单的交叉验证技术之一。使用这种交叉验证技术只需取数据的样本，指定的迭代次数。

# 准备工作

ShuffleSplit 是一种简单的验证技术。我们将指定数据集中的总元素数量，其余由它来处理。我们将通过估计单变量数据集的均值来示例化。这类似于重新采样，但它将说明为什么我们要在展示交叉验证时使用交叉验证。

# 如何做...

1.  首先，我们需要创建数据集。我们将使用 NumPy 创建一个数据集，其中我们知道底层均值。我们将对数据集的一半进行采样以估计均值，并查看它与底层均值的接近程度。生成一个均值为 1000，标准差为 10 的正态分布随机样本：

```py
%matplotlib inline

import numpy as np
true_mean = 1000
true_std = 10
N = 1000
dataset = np.random.normal(loc= true_mean, scale = true_std, size=N)
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(10, 7))
ax.hist(dataset, color='k', alpha=.65, histtype='stepfilled',bins=50)
ax.set_title("Histogram of dataset")
```

![](img/201288c4-5c4a-4e64-bb84-2d2395182bc7.png)

1.  估计数据集的一半的平均值：

```py
holdout_set = dataset[:500]
fitting_set = dataset[500:]
estimate = fitting_set[:N/2].mean()
estimate

999.69789261486721
```

1.  你也可以获取整个数据集的均值：

```py
data_mean = dataset.mean()
data_mean

999.55177343767843
```

1.  它不是 1,000，因为随机选择了点来创建数据集。为了观察 `ShuffleSplit` 的行为，写出以下代码并绘制图表：

```py
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(n_splits=100, test_size=.5, random_state=0)
mean_p = []
estimate_closeness = []
for train_index, not_used_index in shuffle_split.split(fitting_set):
 mean_p.append(fitting_set[train_index].mean())
 shuf_estimate = np.mean(mean_p)
 estimate_closeness.append(np.abs(shuf_estimate - dataset.mean()))

plt.figure(figsize=(10,5))
plt.plot(estimate_closeness)
```

![](img/7de5d845-f9e0-4854-ba09-c8bc4998efea.png)

估计的均值不断接近数据的均值 999.55177343767843，并在距离数据均值 0.1 时停滞。它比仅用一半数据估算出的均值更接近数据的均值。

# 时间序列交叉验证

scikit-learn 可以对时间序列数据（例如股市数据）进行交叉验证。我们将使用时间序列拆分，因为我们希望模型能够预测未来，而不是从未来泄漏信息。

# 准备工作

我们将为时间序列拆分创建索引。首先创建一个小的玩具数据集：

```py
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 1, 2, 3, 4])
```

# 如何做...

1.  现在创建一个时间序列拆分对象：

```py
tscv = TimeSeriesSplit(n_splits=7)
```

1.  遍历它：

```py
for train_index, test_index in tscv.split(X):

 X_train, X_test = X[train_index], X[test_index]
 y_train, y_test = y[train_index], y[test_index]

 print "Training indices:", train_index, "Testing indices:", test_index

Training indices: [0] Testing indices: [1]
Training indices: [0 1] Testing indices: [2]
Training indices: [0 1 2] Testing indices: [3]
Training indices: [0 1 2 3] Testing indices: [4]
Training indices: [0 1 2 3 4] Testing indices: [5]
Training indices: [0 1 2 3 4 5] Testing indices: [6]
Training indices: [0 1 2 3 4 5 6] Testing indices: [7]
```

1.  你也可以通过创建一个包含元组的列表来保存索引：

```py
tscv_list = list(tscv.split(X))
```

# 还有更多...

你也可以使用 NumPy 或 pandas 创建滚动窗口。时间序列交叉验证的主要要求是测试集必须出现在训练集之后；否则，你就会从未来预测过去。

时间序列交叉验证很有趣，因为根据数据集的不同，时间的影响是不同的。有时，你不需要将数据行按时间顺序排列，但你永远不能假设你知道过去的未来。

# 使用 scikit-learn 进行网格搜索

在模型选择和交叉验证章节的开头，我们尝试为鸢尾花数据集的最后两个特征选择最佳的最近邻模型。现在，我们将使用 `GridSearchCV` 在 scikit-learn 中重新聚焦这一点。

# 准备工作

首先，加载鸢尾花数据集的最后两个特征。将数据拆分为训练集和测试集：

```py
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,2:]
y = iris.target

from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,random_state = 7)
```

# 如何做...

1.  实例化一个最近邻分类器：

```py
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
```

1.  准备一个参数网格，这是网格搜索所必需的。参数网格是一个字典，包含你希望尝试的参数设置：

```py
param_grid = {'n_neighbors': list(range(3,9,1))}
```

1.  实例化一个网格搜索，传入以下参数：

    +   估计器

    +   参数网格

    +   一种交叉验证方法，`cv=10`

```py
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(knn_clf,param_grid,cv=10)
```

1.  拟合网格搜索估计器：

```py
gs.fit(X_train, y_train)
```

1.  查看结果：

```py
gs.best_params_

{'n_neighbors': 3}

gs.cv_results_['mean_test_score']

zip(gs.cv_results_['params'],gs.cv_results_['mean_test_score'])

[({'n_neighbors': 3}, 0.9553571428571429),
 ({'n_neighbors': 4}, 0.9375),
 ({'n_neighbors': 5}, 0.9553571428571429),
 ({'n_neighbors': 6}, 0.9553571428571429),
 ({'n_neighbors': 7}, 0.9553571428571429),
 ({'n_neighbors': 8}, 0.9553571428571429)]
```

# 它是如何工作的...

在第一章中，我们尝试了蛮力法，即使用 Python 扫描最佳得分：

```py
all_scores = []
for n_neighbors in range(3,9,1):
 knn_clf = KNeighborsClassifier(n_neighbors = n_neighbors)
 all_scores.append((n_neighbors, cross_val_score(knn_clf, X_train, y_train, cv=10).mean()))
 sorted(all_scores, key = lambda x:x[1], reverse = True)

[(3, 0.95666666666666667),
 (5, 0.95666666666666667),
 (6, 0.95666666666666667),
 (7, 0.95666666666666667),
 (8, 0.95666666666666667),
 (4, 0.94000000000000006)]
```

这种方法的问题是，它更加耗时且容易出错，尤其是当涉及更多参数或额外的转换（如使用管道时）时。

请注意，网格搜索和蛮力法方法都会扫描所有可能的参数值。

# 使用 scikit-learn 进行随机搜索

从实际角度来看，`RandomizedSearchCV` 比常规网格搜索更为重要。因为对于中等大小的数据，或者涉及少量参数的模型，进行完整网格搜索的所有参数组合计算开销太大。

计算资源最好用于非常好地分层采样，或者改进随机化过程。

# 准备就绪

如前所述，加载鸢尾花数据集的最后两个特征。将数据拆分为训练集和测试集：

```py
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:,2:]
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,random_state = 7)
```

# 如何操作...

1.  实例化一个最近邻分类器：

```py
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
```

1.  准备一个参数分布，这是进行随机网格搜索时必需的。参数分布是一个字典，包含你希望随机尝试的参数设置：

```py
param_dist = {'n_neighbors': list(range(3,9,1))}
```

1.  实例化一个随机网格搜索并传入以下参数：

    +   估算器

    +   参数分布

    +   一种交叉验证类型，`cv=10`

    +   运行此过程的次数，`n_iter`

```py
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(knn_clf,param_dist,cv=10,n_iter=6)
```

1.  拟合随机网格搜索估算器：

```py
rs.fit(X_train, y_train)
```

1.  查看结果：

```py
rs.best_params_

{'n_neighbors': 3}

zip(rs.cv_results_['params'],rs.cv_results_['mean_test_score'])

[({'n_neighbors': 3}, 0.9553571428571429),
 ({'n_neighbors': 4}, 0.9375),
 ({'n_neighbors': 5}, 0.9553571428571429),
 ({'n_neighbors': 6}, 0.9553571428571429),
 ({'n_neighbors': 7}, 0.9553571428571429),
 ({'n_neighbors': 8}, 0.9553571428571429)]
```

1.  在这种情况下，我们实际上对所有六个参数进行了网格搜索。你本可以扫描更大的参数空间：

```py
param_dist = {'n_neighbors': list(range(3,50,1))}
rs = RandomizedSearchCV(knn_clf,param_dist,cv=10,n_iter=15)
rs.fit(X_train,y_train)

rs.best_params_

{'n_neighbors': 16}
```

1.  尝试使用 IPython 计时此过程：

```py
%timeit rs.fit(X_train,y_train)

1 loop, best of 3: 1.06 s per loop
```

1.  计时网格搜索过程：

```py
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': list(range(3,50,1))}
gs = GridSearchCV(knn_clf,param_grid,cv=10)

%timeit gs.fit(X_train,y_train)

1 loop, best of 3: 3.24 s per loop
```

1.  查看网格搜索的最佳参数：

```py
gs.best_params_ 

{'n_neighbors': 3}
```

1.  结果表明，3-最近邻的得分与 16-最近邻相同：

```py
zip(gs.cv_results_['params'],gs.cv_results_['mean_test_score'])

[({'n_neighbors': 3}, 0.9553571428571429),
 ({'n_neighbors': 4}, 0.9375),
 ...
 ({'n_neighbors': 14}, 0.9553571428571429),
 ({'n_neighbors': 15}, 0.9553571428571429),
 ({'n_neighbors': 16}, 0.9553571428571429),
 ({'n_neighbors': 17}, 0.9464285714285714),
 ...
```

因此，我们在三分之一的时间内得到了相同的分数。

是否使用随机搜索，这是你需要根据具体情况做出的决定。你应该使用随机搜索来尝试了解某个算法的表现。可能无论参数如何，算法的表现都很差，这时你可以换一个算法。如果算法表现非常好，可以使用完整的网格搜索来寻找最佳参数。

此外，除了专注于穷举搜索，你还可以通过集成、堆叠或混合一组合理表现良好的算法来进行尝试。

# 分类指标

在本章早些时候，我们探讨了基于邻居数量 `n_neighbors` 参数选择几个最近邻实例的最佳方法。这是最近邻分类中的主要参数：基于 KNN 的标签对一个点进行分类。所以，对于 3-最近邻，根据三个最近点的标签对一个点进行分类。对这三个最近点进行多数投票。

该分类指标在此案例中是内部指标 `accuracy_score`，定义为正确分类的数量除以分类总数。还有其他指标，我们将在这里进行探讨。

# 准备就绪

1.  首先，从 UCI 数据库加载 Pima 糖尿病数据集：

```py
import pandas as pd

data_web_address = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

column_names = ['pregnancy_x',
'plasma_con',
'blood_pressure',
'skin_mm',
'insulin',
'bmi',
'pedigree_func',
'age',
'target']

feature_names = column_names[:-1]
all_data = pd.read_csv(data_web_address , names=column_names)
```

1.  将数据拆分为训练集和测试集：

```py
import numpy as np
import pandas as pd

X = all_data[feature_names]
y = all_data['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7,stratify=y)
```

1.  回顾上一部分，使用 KNN 算法运行随机搜索：

```py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

knn_clf = KNeighborsClassifier()

param_dist = {'n_neighbors': list(range(3,20,1))}

rs = RandomizedSearchCV(knn_clf,param_dist,cv=10,n_iter=17)
rs.fit(X_train, y_train)
```

1.  然后显示最佳准确率得分：

```py
rs.best_score_

0.75407166123778502
```

1.  此外，查看测试集上的混淆矩阵：

```py
y_pred = rs.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

array([[84, 16],
 [27, 27]])
```

混淆矩阵提供了更具体的关于模型表现的信息。有 27 次模型预测某人没有糖尿病，尽管他们实际上有。这比 16 个被认为有糖尿病的人实际上没有糖尿病的错误更为严重。

在这种情况下，我们希望最大化灵敏度或召回率。在检查线性模型时，我们查看了召回率或灵敏度的定义：

![](img/f1a676a1-db39-4460-b560-18ba206d5bd6.png)

因此，在这种情况下，灵敏度得分为 27/ (27 + 27) = 0.5。使用 scikit-learn，我们可以方便地按如下方式计算此值。

# 如何操作...

1.  从 metrics 模块导入 `recall_score`。使用 `y_test` 和 `y_pred` 测量集合的灵敏度：

```py
from sklearn.metrics import recall_score

recall_score(y_test, y_pred)

0.5
```

我们恢复了之前手动计算的召回得分。在随机搜索中，我们本可以使用 `recall_score` 来找到具有最高召回率的最近邻实例。

1.  导入 `make_scorer` 并使用带有两个参数的函数 `recall_score` 和 `greater_is_better`：

```py
from sklearn.metrics import make_scorer

recall_scorer = make_scorer(recall_score, greater_is_better=True)
```

1.  现在执行随机网格搜索：

```py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

knn_clf = KNeighborsClassifier()

param_dist = {'n_neighbors': list(range(3,20,1))}

rs = RandomizedSearchCV(knn_clf,param_dist,cv=10,n_iter=17,scoring=recall_scorer)

rs.fit(X_train, y_train)
```

1.  现在查看最高得分：

```py
rs.best_score_

0.5649632669176643
```

1.  查看召回得分：

```py
y_pred = rs.predict(X_test)
recall_score(y_test,y_pred)

0.5
```

1.  结果与之前相同。在随机搜索中，你本可以尝试 `roc_auc_score`，即曲线下面积（ROC AUC）：

```py
from sklearn.metrics import roc_auc_score

rs = RandomizedSearchCV(knn_clf,param_dist,cv=10,n_iter=17,scoring=make_scorer(roc_auc_score,greater_is_better=True))

rs.fit(X_train, y_train)

rs.best_score_

0.7100264217324479
```

# 还有更多...

你可以为分类设计自己的评分器。假设你是一个保险公司，并且你为混淆矩阵中的每个单元格分配了成本。相对成本如下：

![](img/b8334234-cca1-421f-8d45-e157d3765ba8.png)

我们正在查看的混淆矩阵的成本可以按如下方式计算：

```py
costs_array = confusion_matrix(y_test, y_pred) * np.array([[1,2],
 [100,20]])
costs_array

array([[  84,   32],
 [2700,  540]])
```

现在加总总成本：

```py
costs_array.sum()

3356
```

现在将其放入评分器中并运行网格搜索。评分器中的参数 `greater_is_better` 设置为 `False`，因为成本应尽可能低：

```py
def costs_total(y_test, y_pred):

 return (confusion_matrix(y_test, y_pred) * np.array([[1,2],
 [100,20]])).sum()

costs_scorer = make_scorer(costs_total, greater_is_better=False)

rs = RandomizedSearchCV(knn_clf,param_dist,cv=10,n_iter=17,scoring=costs_scorer)

rs.fit(X_train, y_train)

rs.best_score_

-1217.5879478827362
```

得分为负，因为当 `make_scorer` 函数中的 `greater_is_better` 参数为 false 时，得分会乘以 `-1`。网格搜索试图最大化该得分，从而最小化得分的绝对值。

测试集的成本如下：

```py
 costs_total(y_test,rs.predict(X_test)) 

3356
```

在查看这个数字时，别忘了查看测试集中涉及的个体数量，共有 154 人。每个人的平均成本约为 21.8 美元。

# 回归指标

使用回归指标的交叉验证在 scikit-learn 中非常简单。可以从 `sklearn.metrics` 导入评分函数并将其放入 `make_scorer` 函数中，或者你可以为特定的数据科学问题创建自定义评分器。

# 准备就绪

加载一个使用回归指标的数据集。我们将加载波士顿房价数据集并将其拆分为训练集和测试集：

```py
from sklearn.datasets import load_boston
boston = load_boston()

X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
```

我们对数据集了解不多。我们可以尝试使用高方差算法进行快速网格搜索：

```py
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV

knn_reg = KNeighborsRegressor()
param_dist = {'n_neighbors': list(range(3,20,1))}
rs = RandomizedSearchCV(knn_reg,param_dist,cv=10,n_iter=17)
rs.fit(X_train, y_train)
rs.best_score_

0.46455839325055914
```

尝试一个不同的模型，这次是一个线性模型：

```py
from sklearn.linear_model import Ridge
cross_val_score(Ridge(),X_train,y_train,cv=10).mean()

0.7439511908709866
```

默认情况下，两个回归器都衡量 `r2_score`，即 R 平方，因此线性模型更好。尝试一个不同的复杂模型，一个树的集成：

```py
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
cross_val_score(GradientBoostingRegressor(max_depth=7),X_train,y_train,cv=10).mean()

0.83082671732165492
```

集成模型的表现更好。你也可以尝试随机森林：

```py
cross_val_score(RandomForestRegressor(),X_train,y_train,cv=10).mean()

0.82474734196711685
```

现在我们可以通过最大化内部 R-squared 梯度提升评分器，专注于利用当前评分机制来改进梯度提升。尝试进行一两次随机搜索。这是第二次搜索：

```py
param_dist = {'n_estimators': [4000], 'learning_rate': [0.01], 'max_depth':[1,2,3,5,7]}
rs_inst_a = RandomizedSearchCV(GradientBoostingRegressor(), param_dist, n_iter = 5, n_jobs=-1)
rs_inst_a.fit(X_train, y_train)
```

为 R-squared 优化返回了以下结果：

```py
rs_inst_a.best_params_

{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 4000}

rs_inst_a.best_score_

0.88548410382780185
```

梯度提升中的树的深度为三。

# 如何做到...

现在我们将执行以下操作：

1.  创建一个评分函数。

1.  使用该函数创建一个 scikit-scorer。

1.  运行网格搜索以找到最佳的梯度提升参数，最小化误差函数。

让我们开始：

1.  使用 Numba **即时编译**（**JIT**）编译器创建平均百分比误差评分函数。原始的 NumPy 函数如下：

```py
def mape_score(y_test, y_pred):
 return (np.abs(y_test - y_pred)/y_test).mean()
```

1.  让我们使用 Numba JIT 编译器重写这个函数，稍微加速一些。你可以用类似 C 的代码，通过 Numba 按位置索引数组：

```py
from numba import autojit

@autojit
def mape_score(y_test, y_pred):
 sum_total = 0
 y_vec_length = len(y_test)
 for index in range(y_vec_length):
 sum_total += (1 - (y_pred[index]/y_test[index]))

 return sum_total/y_vec_length
```

1.  现在创建一个评分器。得分越低越好，不像 R-squared，那里的得分越高越好：

```py
from sklearn.metrics import make_scorer
mape_scorer = make_scorer(mape_score, greater_is_better=False)
```

1.  现在进行网格搜索：

```py
param_dist = {'n_estimators': [4000], 'learning_rate': [0.01], 'max_depth':[1,2,3,4,5]}
rs_inst_b = RandomizedSearchCV(GradientBoostingRegressor(), param_dist, n_iter = 3, n_jobs=-1,scoring = mape_scorer)
rs_inst_b.fit(X_train, y_train)
rs_inst_b.best_score_

0.021086502313661441

rs_inst_b.best_params_

{'learning_rate': 0.01, 'max_depth': 1, 'n_estimators': 4000}
```

使用此度量，最佳得分对应于深度为 1 的梯度提升树。

# 聚类度量

衡量聚类算法的性能比分类或回归要复杂一些，因为聚类是无监督机器学习。幸运的是，scikit-learn 已经非常直接地为我们提供了帮助。

# 准备开始

为了衡量聚类性能，首先加载鸢尾花数据集。我们将鸢尾花重新标记为两种类型：当目标是 0 时为类型 0，当目标是 1 或 2 时为类型 1：

```py
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data
y = np.where(iris.target == 0,0,1)
```

# 如何做到...

1.  实例化一个 k-means 算法并训练它。由于该算法是聚类算法，因此在训练时不要使用目标值：

```py
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2,random_state=0)
kmeans.fit(X) 
```

1.  现在导入所有必要的库，通过交叉验证对 k-means 进行评分。我们将使用 `adjusted_rand_score` 聚类性能指标：

```py
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

cross_val_score(kmeans,X,y,cv=10,scoring=make_scorer(adjusted_rand_score)).mean()

0.8733695652173914
```

评估聚类算法与评估分类算法非常相似。

# 使用虚拟估算器进行结果比较

本食谱是关于创建虚拟估算器的；这不是很华丽或令人兴奋的部分，但它为你最终构建的模型提供了一个参考点，值得一做。

# 准备开始

在这个食谱中，我们将执行以下任务：

1.  创建一些随机数据。

1.  拟合各种虚拟估算器。

我们将对回归数据和分类数据执行这两步操作。

# 如何做到...

1.  首先，我们将创建随机数据：

```py
from sklearn.datasets import make_regression, make_classification

X, y = make_regression()
from sklearn import dummy
dumdum = dummy.DummyRegressor()
dumdum.fit(X, y)

DummyRegressor(constant=None, quantile=None, strategy='mean')
```

1.  默认情况下，估算器将通过取值的均值并多次输出它来进行预测：

```py
dumdum.predict(X)[:5]

>array([-25.0450033, -25.0450033, -25.0450033, -25.0450033, -25.0450033])
```

还有两种其他策略可以尝试。我们可以预测一个提供的常量（参考此食谱中的第一条命令块中的 `constant=None`）。我们还可以预测中位数值。仅当策略为常量时才会考虑提供常量。

1.  我们来看看：

```py
predictors = [("mean", None),
("median", None),
("constant", 10)]
for strategy, constant in predictors:
 dumdum = dummy.DummyRegressor(strategy=strategy,
 constant=constant)
 dumdum.fit(X, y)
 print "strategy: {}".format(strategy), ",".join(map(str, dumdum.predict(X)[:5]))

strategy: mean -25.0450032962,-25.0450032962,-25.0450032962,-25.0450032962,-25.0450032962
strategy: median -37.734448002,-37.734448002,-37.734448002,-37.734448002,-37.734448002
strategy: constant 10.0,10.0,10.0,10.0,10.0
```

1.  我们实际上有四种分类器的选择。这些策略与连续情况类似，只不过更加倾向于分类问题：

```py
predictors = [("constant", 0),("stratified", None),("uniform", None),("most_frequent", None)]
#We'll also need to create some classification data:
X, y = make_classification()
for strategy, constant in predictors:
 dumdum = dummy.DummyClassifier(strategy=strategy,
 constant=constant)
 dumdum.fit(X, y)
 print "strategy: {}".format(strategy), ",".join(map(str,dumdum.predict(X)[:5]))

strategy: constant 0,0,0,0,0
strategy: stratified 1,0,1,1,1
strategy: uniform 1,0,1,0,1
strategy: most_frequent 0,0,0,0,0
```

# 它是如何工作的...

测试你的模型是否能够比最简单的模型表现更好总是个不错的做法，这正是虚拟估计器所能提供的。例如，假设你有一个欺诈检测模型。在这个模型中，数据集中只有 5%是欺诈行为。因此，我们很可能通过仅仅不猜测数据是欺诈的，就能拟合出一个相当不错的模型。

我们可以通过使用分层策略并执行以下命令来创建此模型。我们还可以得到一个很好的例子，说明类别不平衡是如何导致问题的：

```py
X, y = make_classification(20000, weights=[.95, .05])
dumdum = dummy.DummyClassifier(strategy='most_frequent')
dumdum.fit(X, y)

DummyClassifier(constant=None, random_state=None, strategy='most_frequent')

from sklearn.metrics import accuracy_score
print accuracy_score(y, dumdum.predict(X))

0.94615
```

我们实际上经常是正确的，但这并不是重点。重点是这是我们的基准。如果我们不能创建一个比这个更准确的欺诈检测模型，那么就不值得花费时间。

# 特征选择

本食谱以及接下来的两个将围绕自动特征选择展开。我喜欢将其视为参数调优的特征类比。就像我们通过交叉验证来寻找合适的参数一样，我们也可以找到一个合适的特征子集。这将涉及几种不同的方法。

最简单的想法是单变量选择。其他方法则涉及特征的组合使用。

特征选择的一个附加好处是，它可以减轻数据收集的负担。假设你已经基于一个非常小的数据子集建立了一个模型。如果一切顺利，你可能想扩大规模，在整个数据子集上预测模型。如果是这种情况，你可以在这个规模上减轻数据收集的工程负担。

# 准备工作

使用单变量特征选择时，评分函数将再次成为焦点。这一次，它们将定义我们可以用来消除特征的可比度量。

在本食谱中，我们将拟合一个包含大约 10,000 个特征的回归模型，但只有 1,000 个数据点。我们将逐步了解各种单变量特征选择方法：

```py
from sklearn import datasets
 X, y = datasets.make_regression(1000, 10000)
```

现在我们已经有了数据，我们将比较通过各种方法包含的特征。这实际上是你在处理文本分析或某些生物信息学领域时非常常见的情况。

# 如何操作...

1.  首先，我们需要导入`feature_selection`模块：

```py
from sklearn import feature_selection
f, p = feature_selection.f_regression(X, y)
```

1.  这里，`f`是与每个线性模型拟合相关的`f`分数，该模型仅使用一个特征。然后我们可以比较这些特征，并根据这种比较来剔除特征。`p`是与`f`值相关的`p`值。在统计学中，`p`值是指在给定的检验统计量值下，出现比当前值更极端的结果的概率。在这里，`f`值是检验统计量：

```py
f[:5]

array([ 1.23494617, 0.70831694, 0.09219176, 0.14583189, 0.78776466])

p[:5]

array([ 0.26671492, 0.40020473, 0.76147235, 0.7026321 , 0.37499074])
```

1.  正如我们所见，许多`p`值相当大。我们希望`p`值尽可能小。因此，我们可以从工具箱中取出 NumPy，选择所有小于`.05`的`p`值。这些将是我们用于分析的特征：

```py
import numpy as np
idx = np.arange(0, X.shape[1])
features_to_keep = idx[p < .05]
len(features_to_keep)

496
```

如你所见，我们实际上保留了相对较多的特征。根据模型的上下文，我们可以缩小这个`p`值。这将减少保留的特征数量。

另一个选择是使用`VarianceThreshold`对象。我们已经了解了一些它的内容，但需要明白的是，我们拟合模型的能力在很大程度上依赖于特征所产生的方差。如果没有方差，那么我们的特征就无法描述因变量的变化。根据文档的说法，它的一个优点是由于它不使用结果变量，因此可以用于无监督的情况。

1.  我们需要设定一个阈值，以决定去除哪些特征。为此，我们只需要取特征方差的中位数并提供它：

```py
var_threshold = feature_selection.VarianceThreshold(np.median(np.var(X, axis=1)))
var_threshold.fit_transform(X).shape

(1000L, 4888L)
```

如我们所见，我们已经去除了大约一半的特征，这也大致符合我们的预期。

# 它是如何工作的...

一般来说，这些方法都是通过拟合一个只有单一特征的基本模型来工作的。根据我们是分类问题还是回归问题，我们可以使用相应的评分函数。

让我们看看一个较小的问题，并可视化特征选择如何去除某些特征。我们将使用第一个示例中的相同评分函数，但只使用 20 个特征：

```py
X, y = datasets.make_regression(10000, 20)
f, p = feature_selection.f_regression(X, y)
```

现在让我们绘制特征的 p 值。我们可以看到哪些特征将被去除，哪些特征将被保留：

```py
%matplotlib inline
from matplotlib import pyplot as plt
f, ax = plt.subplots(figsize=(7, 5))
ax.bar(np.arange(20), p, color='k')
ax.set_title("Feature p values")
```

![](img/a8f4c0f9-e43a-4714-8c0b-05890091b2d1.png)

如我们所见，许多特征将不会被保留，但有一些特征会被保留。

# 基于 L1 范数的特征选择

我们将使用一些与 LASSO 回归配方中看到的类似的思想。在那个配方中，我们查看了具有零系数的特征数量。现在我们将更进一步，利用与 L1 范数相关的稀疏性来预处理特征。

# 准备工作

我们将使用糖尿病数据集来进行回归拟合。首先，我们将使用 ShuffleSplit 交叉验证拟合一个基本的线性回归模型。完成后，我们将使用 LASSO 回归来找到系数为零的特征，这些特征在使用 L1 惩罚时会被去除。这有助于我们避免过拟合（即模型过于专门化，无法适应它未训练过的数据）。换句话说，如果模型过拟合，它对外部数据的泛化能力较差。

我们将执行以下步骤：

1.  加载数据集。

1.  拟合一个基本的线性回归模型。

1.  使用特征选择去除不具信息量的特征。

1.  重新拟合线性回归模型，并检查它与完全特征模型相比的拟合效果。

# 如何操作...

1.  首先，让我们获取数据集：

```py
import sklearn.datasets as ds
diabetes = ds.load_diabetes()

X = diabetes.data
y = diabetes.target
```

1.  让我们创建`LinearRegression`对象：

```py
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
```

1.  让我们从 metrics 模块导入`mean_squared_error`函数和`make_scorer`包装器。从`model_selection`模块，导入`ShuffleSplit`交叉验证方案和`cross_val_score`交叉验证评分器。接下来，使用`mean_squared_error`度量来评分该函数：

```py
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score,ShuffleSplit

shuff = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
score_before = cross_val_score(lr,X,y,cv=shuff,scoring=make_scorer(mean_squared_error,greater_is_better=False)).mean()

score_before

-3053.393446308266
```

1.  现在我们已经有了常规拟合，让我们在去除系数为零的特征后检查一下。让我们拟合 LASSO 回归：

```py
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV()
lasso_cv.fit(X,y)
lasso_cv.coef_

array([ -0\. , -226.2375274 , 526.85738059, 314.44026013,
 -196.92164002, 1.48742026, -151.78054083, 106.52846989,
 530.58541123, 64.50588257])
```

1.  我们将删除第一个特征。我将使用 NumPy 数组来表示要包含在模型中的列：

```py
import numpy as np
columns = np.arange(X.shape[1])[lasso_cv.coef_ != 0]
columns
```

1.  好的，现在我们将使用特定的特征来拟合模型（请参见以下代码块中的列）：

```py
score_afterwards = cross_val_score(lr,X[:,columns],y,cv=shuff, scoring=make_scorer(mean_squared_error,greater_is_better=False)).mean()
score_afterwards

-3033.5012859289677
```

之后的得分并没有比之前好多少，尽管我们已经消除了一个无信息特征。在*还有更多内容...*部分，我们将看到一个额外的示例。

# 还有更多内容...

1.  首先，我们将创建一个具有许多无信息特征的回归数据集：

```py
X, y = ds.make_regression(noise=5)
```

1.  创建一个`ShuffleSplit`实例，进行 10 次迭代，`n_splits=10`。测量普通线性回归的交叉验证得分：

```py
shuff = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)

score_before = cross_val_score(lr,X,y,cv=shuff, scoring=make_scorer(mean_squared_error,greater_is_better=False)).mean()
```

1.  实例化`LassoCV`来消除无信息的列：

```py
lasso_cv = LassoCV()
lasso_cv.fit(X,y)
```

1.  消除无信息的列。查看最终得分：

```py
columns = np.arange(X.shape[1])[lasso_cv.coef_ != 0]
score_afterwards = cross_val_score(lr,X[:,columns],y,cv=shuff, scoring=make_scorer(mean_squared_error,greater_is_better=False)).mean()
print "Score before:",score_before
print "Score after: ",score_afterwards

Score before: -8891.35368845
Score after: -22.3488585347
```

在我们移除无信息特征后，最后的拟合效果要好得多。

# 使用 joblib 或 pickle 持久化模型

在这个教程中，我们将展示如何将模型保留以供以后使用。例如，你可能希望使用一个模型来预测结果并自动做出决策。

# 做好准备

创建数据集并训练分类器：

```py
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification()
dt = DecisionTreeClassifier()
dt.fit(X, y)
```

# 如何做...

1.  使用 joblib 保存分类器所做的训练工作：

```py
from sklearn.externals import joblib
joblib.dump(dt, "dtree.clf")

['dtree.clf']
```

# 打开已保存的模型

1.  使用 joblib 加载模型。使用一组输入进行预测：

```py
from sklearn.externals import joblib
pulled_model = joblib.load("dtree.clf")
y_pred = pulled_model.predict(X)
```

我们不需要重新训练模型，并且节省了大量训练时间。我们只是使用 joblib 重新加载了模型并进行了预测。

# 还有更多内容...

你也可以在 Python 2.x 中使用`cPickle`模块，或者在 Python 3.x 中使用`pickle`模块。就个人而言，我使用这个模块处理几种类型的 Python 类和对象：

1.  首先导入`pickle`：

```py
import cPickle as pickle #Python 2.x
# import pickle          # Python 3.x
```

1.  使用`dump()`模块方法。它有三个参数：要保存的数据、保存目标文件和 pickle 协议。以下代码将训练好的树保存到`dtree.save`文件：

```py
f = open("dtree.save", 'wb')
pickle.dump(dt,f, protocol = pickle.HIGHEST_PROTOCOL)
f.close()
```

1.  按如下方式打开`dtree.save`文件：

```py
f = open("dtree.save", 'rb')
return_tree = pickle.load(f)
f.close()
```

1.  查看树：

```py
return_tree

DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best'
```
