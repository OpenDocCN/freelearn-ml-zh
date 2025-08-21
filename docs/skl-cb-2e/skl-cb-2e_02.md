# 第二章：模型前工作流与预处理

在本章中，我们将看到以下内容：

+   为玩具分析创建样本数据

+   将数据缩放到标准正态分布

+   通过阈值创建二元特征

+   处理分类变量

+   通过各种策略填补缺失值

+   存在离群值时的线性模型

+   使用管道将所有内容整合起来

+   使用高斯过程进行回归

+   使用 SGD 进行回归

# 引言

什么是数据，我们对数据的处理目的是什么？

一个简单的答案是，我们试图将数据点放在纸上，将其绘制成图，思考并寻找能够很好地近似数据的简单解释。简单的几何线 *F=ma*（力与加速度成正比）解释了数百年的大量噪声数据。我有时倾向于将数据科学看作是数据压缩。

有时，当机器只接受输赢结果（例如玩跳棋的游戏结果）并进行训练时，我认为这就是人工智能。在这种情况下，它从未被明确教导如何玩游戏以获得胜利。

本章讨论了在 scikit-learn 中的数据预处理。你可以向数据集提问的问题如下：

+   数据集是否存在缺失值？

+   数据集中是否存在离群值（远离其他点的值）？

+   数据中的变量是什么类型的？它们是连续变量还是分类变量？

+   连续变量的分布是什么样的？数据集中的某些变量是否可以用正态分布（钟形曲线）来描述？

+   是否可以将任何连续变量转化为分类变量以简化处理？（如果数据的分布只有少数几个特定值，而不是类似连续范围的值，这种情况通常成立。）

+   涉及的变量单位是什么？你是否会在选择的机器学习算法中混合这些变量？

这些问题可能有简单或复杂的答案。幸运的是，你会多次提问，甚至在同一个数据集上，也会不断练习如何回答机器学习中的预处理问题。

此外，我们还将了解管道：一种很好的组织工具，确保我们在训练集和测试集上执行相同的操作，避免出错并且工作量相对较少。我们还将看到回归示例：**随机梯度下降**（**SGD**）和高斯过程。

# 为玩具分析创建样本数据

如果可能，使用你自己的一些数据来学习本书中的内容。如果你无法这样做，我们将学习如何使用 scikit-learn 创建玩具数据。scikit-learn 的伪造、理论构建的数据本身非常有趣。

# 准备工作

与获取内置数据集、获取新数据集和创建样本数据集类似，所使用的函数遵循 `make_*` 命名约定。为了明确，这些数据完全是人工合成的：

```py
from sklearn import datasets
datasets.make_*?

datasets.make_biclusters
datasets.make_blobs
datasets.make_checkerboard
datasets.make_circles
datasets.make_classification
...
```

为了省略输入，导入`datasets`模块为`d`，`numpy`为`np`：

```py
import sklearn.datasets as d
import numpy as np
```

# 如何实现...

本节将带你逐步创建几个数据集。除了示例数据集外，这些数据集将贯穿整本书，用于创建具有算法所需特征的数据。

# 创建回归数据集

1.  首先是可靠的——回归：

```py
reg_data = d.make_regression()
```

默认情况下，这将生成一个包含 100 x 100 矩阵的元组——100 个样本和 100 个特征。然而，默认情况下，只有 10 个特征负责目标数据的生成。元组的第二个成员是目标变量。实际上，也可以更深入地参与回归数据的生成。

1.  例如，为了生成一个 1000 x 10 的矩阵，其中五个特征负责目标的创建，偏置因子为 1.0，并且有两个目标，可以运行以下命令：

```py
complex_reg_data = d.make_regression(1000, 10, 5, 2, 1.0)
complex_reg_data[0].shape

(1000L, 10L)
```

# 创建一个不平衡的分类数据集

分类数据集也非常容易创建。创建一个基本的分类数据集很简单，但基本情况在实践中很少见——大多数用户不会转换，大多数交易不是欺诈性的，等等。

1.  因此，探索不平衡数据集上的分类是非常有用的：

```py
classification_set = d.make_classification(weights=[0.1])
np.bincount(classification_set[1])

array([10, 90], dtype=int64)
```

# 创建聚类数据集

聚类也会涉及到。实际上，有多个函数可以创建适用于不同聚类算法的数据集。

1.  例如，blobs 非常容易创建，并且可以通过 k-means 来建模：

```py
blobs_data, blobs_target = d.make_blobs()
```

1.  这将看起来像这样：

```py
import matplotlib.pyplot as plt
%matplotlib inline 
#Within an Ipython notebook
plt.scatter(blobs_data[:,0],blobs_data[:,1],c = blobs_target) 
```

![](img/0bf9b285-7654-42fc-a2cb-4f14e9d40e44.png)

# 它是如何工作的...

让我们通过查看源代码（做了一些修改以便清晰）来逐步了解 scikit-learn 如何生成回归数据集。任何未定义的变量假定其默认值为`make_regression`。

实际上，跟着做是非常简单的。首先，生成一个随机数组，大小由调用函数时指定：

```py
X = np.random.randn(n_samples, n_features)
```

给定基本数据集后，接着生成目标数据集：

```py
ground_truth = np.zeros((np_samples, n_target))
ground_truth[:n_informative, :] = 100*np.random.rand(n_informative, n_targets)
```

计算`X`和`ground_truth`的点积来得到最终的目标值。此时，如果有偏置，也会被加上：

```py
y = np.dot(X, ground_truth) + bias
```

点积其实就是矩阵乘法。因此，我们的最终数据集将包含`n_samples`，即数据集的行数，和`n_target`，即目标变量的数量。

由于 NumPy 的广播机制，偏置可以是一个标量值，并且这个值将添加到每个样本中。最后，只需简单地加入噪声并打乱数据集。瞧，我们得到了一个非常适合回归测试的数据集。

# 将数据缩放至标准正态分布

推荐的预处理步骤是将列缩放至标准正态分布。标准正态分布可能是统计学中最重要的分布。如果你曾接触过统计学，你几乎肯定见过 z 分数。事实上，这就是这个方法的核心——将特征从其原始分布转换为 z 分数。

# 准备开始

缩放数据的操作非常有用。许多机器学习算法在特征存在不同尺度时表现不同（甚至可能出错）。例如，如果数据没有进行缩放，支持向量机（SVM）的表现会很差，因为它们在优化中使用距离函数，而如果一个特征的范围是 0 到 10,000，另一个特征的范围是 0 到 1，距离函数会出现偏差。

`preprocessing`模块包含了几个有用的特征缩放函数：

```py
from sklearn import preprocessing
import numpy as np # we'll need it later
```

加载波士顿数据集：

```py
from sklearn.datasets import load_boston

boston = load_boston()
X,y = boston.data, boston.target
```

# 如何实现...

1.  继续使用波士顿数据集，运行以下命令：

```py
X[:, :3].mean(axis=0) #mean of the first 3 features

array([  3.59376071,  11.36363636,  11.13677866])

X[:, :3].std(axis=0)

array([  8.58828355,  23.29939569,   6.85357058])
```

1.  从一开始就可以学到很多东西。首先，第一个特征的均值最小，但变化范围比第三个特征更大。第二个特征的均值和标准差最大——它需要分布最广的数值：

```py
X_2 = preprocessing.scale(X[:, :3])
X_2.mean(axis=0)

array([  6.34099712e-17,  -6.34319123e-16,  -2.68291099e-15])

X_2.std(axis=0)

array([ 1., 1., 1.])
```

# 它是如何工作的...

中心化和缩放函数非常简单。它只是将均值相减并除以标准差。

通过图示和 pandas，第三个特征在变换之前如下所示：

```py
pd.Series(X[:,2]).hist(bins=50)
```

![](img/0d0ec1e7-2c6e-447a-a34c-c6a548aab0db.png)

变换后的样子如下：

```py
pd.Series(preprocessing.scale(X[:, 2])).hist(bins=50)
```

![](img/1a8a94aa-e223-4952-94f9-2ab4258b4705.png)

*x*轴标签已更改。

除了函数，还有一个易于调用的中心化和缩放类，特别适用于与管道一起使用，管道在后面会提到。这个类在跨个别缩放时也特别有用：

```py
my_scaler = preprocessing.StandardScaler()
my_scaler.fit(X[:, :3])
my_scaler.transform(X[:, :3]).mean(axis=0)

array([  6.34099712e-17,  -6.34319123e-16,  -2.68291099e-15])
```

将特征缩放到均值为零、标准差为一并不是唯一有用的缩放类型。

预处理还包含一个`MinMaxScaler`类，它可以将数据缩放到某个特定范围内：

```py
my_minmax_scaler = preprocessing.MinMaxScaler()
my_minmax_scaler.fit(X[:, :3])
my_minmax_scaler.transform(X[:, :3]).max(axis=0)

array([ 1., 1., 1.])

my_minmax_scaler.transform(X[:, :3]).min(axis=0)

array([ 0., 0., 0.])
```

很容易将`MinMaxScaler`类的最小值和最大值从默认的`0`和`1`更改为其他值：

```py

my_odd_scaler = preprocessing.MinMaxScaler(feature_range=(-3.14, 3.14))
```

此外，另一种选择是归一化。归一化会将每个样本缩放到长度为 1。这与之前进行的缩放不同，之前是缩放特征。归一化的操作可以通过以下命令实现：

```py
normalized_X = preprocessing.normalize(X[:, :3]) 
```

如果不清楚为什么这样做有用，可以考虑三组样本之间的欧几里得距离（相似度度量），其中一组样本的值为*(1, 1, 0)*，另一组的值为*(3, 3, 0)*，最后一组的值为*(1, -1, 0)*。

第一个和第三个向量之间的距离小于第一个和第二个向量之间的距离，尽管第一和第三是正交的，而第一和第二仅通过一个标量因子差异为三。由于距离通常用作相似度的度量，不对数据进行归一化可能会导致误导。

从另一个角度来看，尝试以下语法：

```py
(normalized_X * normalized_X).sum(axis = 1)

array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1\. 
 ...]
```

所有行都被归一化，并且由长度为 1 的向量组成。在三维空间中，所有归一化的向量都位于以原点为中心的球面上。剩下的信息是向量的方向，因为按定义，归一化是通过将向量除以其长度来完成的。然而，请始终记住，在执行此操作时，你已将原点设置为*(0, 0, 0)*，并且你已将数组中的任何数据行转换为相对于此原点的向量。

# 通过阈值化创建二元特征

在上一节中，我们讨论了如何将数据转换为标准正态分布。现在，我们将讨论另一种完全不同的转换方法。我们不再通过处理分布来标准化它，而是故意丢弃数据；如果有充分的理由，这可能是一个非常聪明的举动。通常，在看似连续的数据中，会存在一些可以通过二元特征确定的间断点。

此外，请注意，在上一章中，我们将分类问题转化为回归问题。通过阈值化，我们可以将回归问题转化为分类问题。在一些数据科学的场景中，这种情况是存在的。

# 准备工作

创建二元特征和结果是一种非常有用的方法，但应谨慎使用。让我们使用波士顿数据集来学习如何将值转换为二元结果。首先，加载波士顿数据集：

```py
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target.reshape(-1, 1)
```

# 如何实现...

与缩放类似，scikit-learn 中有两种方式可以将特征二值化：

+   `preprocessing.binarize`

+   `preprocessing.Binarizer`

波士顿数据集的`target`变量是以千为单位的房屋中位数值。这个数据集适合用来测试回归和其他连续预测模型，但可以考虑一种情况，我们只需要预测房屋的价值是否超过整体均值。

1.  为此，我们需要创建一个均值的阈值。如果值大于均值，返回`1`；如果小于均值，返回`0`：

```py
from sklearn import preprocessing
new_target = preprocessing.binarize(y,threshold=boston.target.mean())
new_target[:5]

 array([[ 1.],
 [ 0.],
 [ 1.],
 [ 1.],
 [ 1.]])
```

1.  这很简单，但让我们检查一下以确保它正常工作：

```py
(y[:5] > y.mean()).astype(int)

array([[1],
       [0],
       [1],
       [1],
       [1]])
```

1.  鉴于 NumPy 中操作的简单性，提出为什么要使用 scikit-learn 的内建功能是一个合理的问题。在*将一切通过管道组合起来*这一节中介绍的管道将有助于解释这一点；为了预见到这一点，让我们使用`Binarizer`类：

```py
binar = preprocessing.Binarizer(y.mean())
new_target = binar.fit_transform(y)
new_target[:5]

array([[ 1.],
       [ 0.],
       [ 1.],
       [ 1.],
       [ 1.]])
```

# 还有更多……

让我们也来了解稀疏矩阵和`fit`方法。

# 稀疏矩阵

稀疏矩阵的特殊之处在于零值并不被存储；这是为了节省内存空间。这样会为二值化器带来问题，因此，为了应对这一问题，针对稀疏矩阵，二值化器的特殊条件是阈值不能小于零：

```py
from scipy.sparse import coo
spar = coo.coo_matrix(np.random.binomial(1, .25, 100))
preprocessing.binarize(spar, threshold=-1)

ValueError: Cannot binarize a sparse matrix with threshold &lt; 0
```

# `fit`方法

`fit`方法是针对二值化转换存在的，但它不会对任何东西进行拟合；它只会返回该对象。该对象会存储阈值，并准备好进行`transform`方法。

# 处理分类变量

类别变量是一个问题。一方面，它们提供了有价值的信息；另一方面，它们很可能是文本——无论是实际的文本还是与文本对应的整数——例如查找表中的索引。

所以，显然我们需要将文本表示为整数，以便于模型处理，但不能仅仅使用 ID 字段或天真地表示它们。这是因为我们需要避免类似于*通过阈值创建二进制特征*食谱中出现的问题。如果我们处理的是连续数据，它必须被解释为连续数据。

# 准备工作

波士顿数据集对于本节不适用。虽然它对于特征二值化很有用，但不足以从类别变量中创建特征。为此，鸢尾花数据集就足够了。

为了使其生效，问题需要彻底转变。设想一个问题，目标是预测花萼宽度；在这种情况下，花卉的物种可能作为一个特征是有用的。

# 如何做……

1.  让我们先整理数据：

```py
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()

X = iris.data
y = iris.target
```

1.  将`X`和`y`，所有数值数据，放在一起。使用 scikit-learn 创建一个编码器来处理`y`列的类别：

```py
from sklearn import preprocessing
cat_encoder = preprocessing.OneHotEncoder()
cat_encoder.fit_transform(y.reshape(-1,1)).toarray()[:5]

array([[ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.]])
```

# 它是如何工作的……

编码器为每个类别变量创建附加特征，返回的值是一个稀疏矩阵。根据定义，结果是一个稀疏矩阵；新特征的每一行除了与特征类别关联的列外，其他位置都是`0`。因此，将这些数据存储为稀疏矩阵是合理的。现在，`cat_encoder`是一个标准的 scikit-learn 模型，这意味着它可以再次使用：

```py
cat_encoder.transform(np.ones((3, 1))).toarray() 

array([[ 0.,  1.,  0.],
 [ 0.,  1.,  0.],
 [ 0.,  1.,  0.]])
```

在上一章中，我们将一个分类问题转化为回归问题。在这里，有三列数据：

+   第一列是`1`，如果花是 Setosa，则为`1`，否则为`0`。

+   第二列是`1`，如果花是 Versicolor，则为`1`，否则为`0`。

+   第三列是`1`，如果花是 Virginica，则为`1`，否则为`0`。

因此，我们可以使用这三列中的任何一列来创建与上一章类似的回归；我们将执行回归以确定花卉的 Setosa 程度作为一个实数。如果我们对第一列进行二分类，这就是分类中的问题陈述，判断花卉是否是 Setosa。

scikit-learn 具有执行此类型的多输出回归的能力。与多类分类相比，让我们尝试一个简单的例子。

导入岭回归正则化线性模型。由于它是正则化的，通常表现得非常稳定。实例化一个岭回归器类：

```py
from sklearn.linear_model import Ridge
ridge_inst = Ridge()
```

现在导入一个多输出回归器，将岭回归器实例作为参数：

```py
from sklearn.multioutput import MultiOutputRegressor
multi_ridge = MultiOutputRegressor(ridge_inst, n_jobs=-1)
```

从本食谱前面的部分，将目标变量`y`转换为三部分目标变量`y_multi`，并使用`OneHotEncoder()`。如果`X`和`y`是管道的一部分，管道将分别转换训练集和测试集，这是更可取的做法：

```py
from sklearn import preprocessing
cat_encoder = preprocessing.OneHotEncoder()
y_multi = cat_encoder.fit_transform(y.reshape(-1,1)).toarray()
```

创建训练集和测试集：

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_multi, stratify=y, random_state= 7)
```

拟合多输出估计器：

```py
multi_ridge.fit(X_train, y_train)

MultiOutputRegressor(estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001),
           n_jobs=-1)
```

在测试集上预测多输出目标：

```py
y_multi_pre = multi_ridge.predict(X_test)
y_multi_pre[:5]

array([[ 0.81689644,  0.36563058, -0.18252702],
 [ 0.95554968,  0.17211249, -0.12766217],
 [-0.01674023,  0.36661987,  0.65012036],
 [ 0.17872673,  0.474319  ,  0.34695427],
 [ 0.8792691 ,  0.14446485, -0.02373395]])
```

使用前面配方中的 `binarize` 函数将每个实数转换为整数 `0` 或 `1`：

```py
from sklearn import preprocessing
y_multi_pred = preprocessing.binarize(y_multi_pre,threshold=0.5)
y_multi_pred[:5]

array([[ 1.,  0.,  0.],
 [ 1.,  0.,  0.],
 [ 0.,  0.,  1.],
 [ 0.,  0.,  0.],
 [ 1.,  0.,  0.]])
```

我们可以使用 `roc_auc_score` 来衡量整体的多输出性能：

```py
from sklearn.metrics import roc_auc_score

 roc_auc_score(y_test, y_multi_pre)

0.91987179487179482
```

或者，我们可以逐种花朵类型、逐列进行：

```py
from sklearn.metrics import accuracy_score

print ("Multi-Output Scores for the Iris Flowers: ")
for column_number in range(0,3):
 print ("Accuracy score of flower " + str(column_number),accuracy_score(y_test[:,column_number], y_multi_pred[:,column_number]))
 print ("AUC score of flower " + str(column_number),roc_auc_score(y_test[:,column_number], y_multi_pre[:,column_number]))
 print ("")

 Multi-Output Scores for the Iris Flowers:
 ('Accuracy score of flower 0', 1.0)
 ('AUC score of flower 0', 1.0)

 ('Accuracy score of flower 1', 0.73684210526315785)
 ('AUC score of flower 1', 0.76923076923076927)

 ('Accuracy score of flower 2', 0.97368421052631582)
 ('AUC score of flower 2', 0.99038461538461542)
```

# 还有更多……

在前面的多输出回归中，你可能会担心虚拟变量陷阱：输出之间的共线性。在不删除任何输出列的情况下，你假设存在第四种选择：即花朵可以不是三种类型中的任何一种。为了避免陷阱，删除最后一列，并假设花朵必须是三种类型之一，因为我们没有任何训练样本显示花朵不是三种类型中的一种。

在 scikit-learn 和 Python 中还有其他方法可以创建分类变量。如果你希望将项目的依赖项仅限于 scikit-learn，并且有一个相对简单的编码方案，`DictVectorizer` 类是一个不错的选择。然而，如果你需要更复杂的分类编码，patsy 是一个非常好的选择。

# DictVectorizer 类

另一个选择是使用 `DictVectorizer` 类。这可以直接将字符串转换为特征：

```py
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
my_dict = [{'species': iris.target_names[i]} for i in y]
dv.fit_transform(my_dict).toarray()[:5]

array([[ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 1.,  0.,  0.]])
```

# 通过各种策略填充缺失值

数据填充在实践中至关重要，幸运的是有很多方法可以处理它。在本配方中，我们将查看几种策略。然而，请注意，可能还有其他方法更适合你的情况。

这意味着 scikit-learn 具备执行常见填充操作的能力；它会简单地对现有数据应用一些变换并填充缺失值。然而，如果数据集缺失数据，并且我们知道这种缺失数据的原因——例如服务器响应时间在 100 毫秒后超时——那么通过其他包采用统计方法可能会更好，比如通过 PyMC 进行的贝叶斯处理，或通过 Lifelines 进行的危险模型，或是自定义的处理方法。

# 准备好

学习如何输入缺失值时，首先要做的就是创建缺失值。NumPy 的掩码处理将使这变得极其简单：

```py
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data
masking_array = np.random.binomial(1, .25,iris_X.shape).astype(bool)
iris_X[masking_array] = np.nan
```

为了稍微澄清一下，如果你对 NumPy 不太熟悉，在 NumPy 中可以使用其他数组来索引数组。所以，为了创建随机缺失数据，创建了一个与鸢尾花数据集形状相同的随机布尔数组。然后，可以通过掩码数组进行赋值。需要注意的是，由于使用了随机数组，因此你的 `masking_array` 可能与此处使用的不同。

为了确保这有效，请使用以下命令（由于我们使用了随机掩码，它可能不会直接匹配）：

```py
masking_array[:5]

array([[ True, False, False,  True],
       [False, False, False, False],
       [False, False, False, False],
       [ True, False, False, False],
       [False, False, False,  True]], dtype=bool)

iris_X [:5]

array([[ nan,  3.5,  1.4,  nan],
       [ 4.9,  3\. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ nan,  3.1,  1.5,  0.2],
       [ 5\. ,  3.6,  1.4,  nan]])
```

# 如何做……

1.  本书中的一个常见主题（由于 scikit-learn 中的主题）是可重用的类，这些类能够拟合和转换数据集，随后可以用来转换未见过的数据集。如下所示：

```py
from sklearn import preprocessing
impute = preprocessing.Imputer()
iris_X_prime = impute.fit_transform(iris_X)
iris_X_prime[:5]

array([[ 5.82616822,  3.5       ,  1.4       ,  1.22589286],
       [ 4.9       ,  3\.        ,  1.4       ,  0.2       ],
       [ 4.7       ,  3.2       ,  1.3       ,  0.2       ],
       [ 5.82616822,  3.1       ,  1.5       ,  0.2       ],
       [ 5\.        ,  3.6       ,  1.4       ,  1.22589286]])
```

1.  注意 `[0, 0]` 位置的差异：

```py
iris_X_prime[0, 0]

5.8261682242990664

iris_X[0, 0] 

nan
```

# 它是如何工作的...

填充操作通过采用不同的策略进行。默认值是均值，但总共有以下几种策略：

+   `mean`（默认值）

+   `median`（中位数）

+   `most_frequent`（众数）

scikit-learn 将使用所选策略计算数据集中每个非缺失值的值，然后简单地填充缺失值。例如，要使用中位数策略重新执行鸢尾花示例，只需用新策略重新初始化填充器：

```py
impute = preprocessing.Imputer(strategy='median')
iris_X_prime = impute.fit_transform(iris_X)
iris_X_prime[:5]

array([[ 5.8,  3.5,  1.4,  1.3],
       [ 4.9,  3\. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 5.8,  3.1,  1.5,  0.2],
       [ 5\. ,  3.6,  1.4,  1.3]])
```

如果数据中缺少值，那么其他地方可能也存在数据质量问题。例如，在上面提到的 *如何操作...* 部分中，`np.nan`（默认的缺失值）被用作缺失值，但缺失值可以用多种方式表示。考虑一种情况，缺失值是 `-1`。除了计算缺失值的策略外，还可以为填充器指定缺失值。默认值是 `nan`，它会处理 `np.nan` 值。

要查看此示例，请将 `iris_X` 修改为使用 `-1` 作为缺失值。这听起来很疯狂，但由于鸢尾花数据集包含的是永远可能测量的数据，许多人会用 `-1` 来填充缺失值，以表示这些数据缺失：

```py
iris_X[np.isnan(iris_X)] = -1
iris_X[:5]
```

填充这些缺失值的方法非常简单，如下所示：

```py
impute = preprocessing.Imputer(missing_values=-1)
iris_X_prime = impute.fit_transform(iris_X)
iris_X_prime[:5]

array([[ 5.1 , 3.5 , 1.4 , 0.2 ],
 [ 4.9 , 3\. , 1.4 , 0.2 ],
 [ 4.7 , 3.2 , 1.3 , 0.2 ],
 [ 5.87923077, 3.1 , 1.5 , 0.2 ],
 [ 5\. , 3.6 , 1.4 , 0.2 ]])
```

# 还有更多...

Pandas 也提供了一种填充缺失数据的功能。它可能更加灵活，但也较少可重用：

```py
import pandas as pd
iris_X_prime = np.where(pd.DataFrame(iris_X).isnull(),-1,iris_X)
iris_X_prime[:5]

array([[-1\. ,  3.5,  1.4, -1\. ],
       [ 4.9,  3\. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [-1\. ,  3.1,  1.5,  0.2],
       [ 5\. ,  3.6,  1.4, -1\. ]])
```

为了说明其灵活性，`fillna` 可以传入任何类型的统计量，也就是说，策略可以更加随意地定义：

```py
pd.DataFrame(iris_X).fillna(-1)[:5].values

array([[-1\. ,  3.5,  1.4, -1\. ],
       [ 4.9,  3\. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [-1\. ,  3.1,  1.5,  0.2],
       [ 5\. ,  3.6,  1.4, -1\. ]])
```

# 存在离群点的线性模型

在本示例中，我们将尝试使用 Theil-Sen 估计器来处理一些离群点，而不是传统的线性回归。

# 准备工作

首先，创建一条斜率为 `2` 的数据线：

```py
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

num_points = 100
x_vals = np.arange(num_points)
y_truth = 2 * x_vals
plt.plot(x_vals, y_truth)
```

![](img/d1ebad85-c139-418a-94ec-06c1d3ce5791.png)

给数据添加噪声，并将其标记为 `y_noisy`：

```py
y_noisy = y_truth.copy()
#Change y-values of some points in the line
y_noisy[20:40] = y_noisy[20:40] * (-4 * x_vals[20:40]) - 100

plt.title("Noise in y-direction")
plt.xlim([0,100])
plt.scatter(x_vals, y_noisy,marker='x')
```

![](img/f9790eae-a74e-4d0f-93b0-45057e7afff8.png)

# 如何操作...

1.  导入 `LinearRegression` 和 `TheilSenRegressor`。使用原始线作为测试集 `y_truth` 对估计器进行评分：

```py
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.metrics import r2_score, mean_absolute_error

named_estimators = [('OLS ', LinearRegression()), ('TSR ', TheilSenRegressor())]

for num_index, est in enumerate(named_estimators):
 y_pred = est[1].fit(x_vals.reshape(-1, 1),y_noisy).predict(x_vals.reshape(-1, 1))
 print (est[0], "R-squared: ", r2_score(y_truth, y_pred), "Mean Absolute Error", mean_absolute_error(y_truth, y_pred))
 plt.plot(x_vals, y_pred, label=est[0])

('OLS   ', 'R-squared: ', 0.17285546630270587, 'Mean Absolute Error', 44.099173357335729)
('TSR   ', 'R-squared: ', 0.99999999928066519, 'Mean Absolute Error', 0.0013976236426276058)
```

1.  绘制这些线条。请注意，**普通最小二乘法**（**OLS**）与真实线 `y_truth` 相差甚远，而 Theil-Sen 则与真实线重叠：

```py
plt.plot(x_vals, y_truth, label='True line')
plt.legend(loc='upper left')
```

![](img/87bf00b0-a0bf-4e0e-8e82-8fcdd3a54a7d.png)

1.  绘制数据集和估计的线条：

```py
for num_index, est in enumerate(named_estimators):
 y_pred = est[1].fit(x_vals.reshape(-1, 1),y_noisy).predict(x_vals.reshape(-1, 1))
 plt.plot(x_vals, y_pred, label=est[0])
plt.legend(loc='upper left')
plt.title("Noise in y-direction")
plt.xlim([0,100])
plt.scatter(x_vals, y_noisy,marker='x', color='red')
```

![](img/24418ea6-bb8c-4d1d-90bf-c357f9d6bd18.png)

# 它是如何工作的...

`TheilSenRegressor` 是一种鲁棒估计器，在存在离群点的情况下表现良好。它使用中位数的测量，更加稳健于离群点。在 OLS 回归中，误差会被平方，因此平方误差可能会导致好的结果变差。

你可以在 scikit-learn 版本 0.19.0 中尝试几种鲁棒估计器：

```py
from sklearn.linear_model import Ridge, LinearRegression, TheilSenRegressor, RANSACRegressor, ElasticNet, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error
named_estimators = [('OLS ', LinearRegression()),
('Ridge ', Ridge()),('TSR ', TheilSenRegressor()),('RANSAC', RANSACRegressor()),('ENet ',ElasticNet()),('Huber ',HuberRegressor())]

for num_index, est in enumerate(named_estimators):
 y_pred = est[1].fit(x_vals.reshape(-1, 1),y_noisy).predict(x_vals.reshape(-1, 1))
 print (est[0], "R-squared: ", r2_score(y_truth, y_pred), "Mean Absolute Error", mean_absolute_error(y_truth, y_pred))

('OLS   ', 'R-squared: ', 0.17285546630270587, 'Mean Absolute Error', 44.099173357335729)
('Ridge ', 'R-squared: ', 0.17287378039132695, 'Mean Absolute Error', 44.098937961740631)
('TSR   ', 'R-squared: ', 0.99999999928066519, 'Mean Absolute Error', 0.0013976236426276058)
('RANSAC', 'R-squared: ', 1.0, 'Mean Absolute Error', 1.0236256287043944e-14)
('ENet  ', 'R-squared: ', 0.17407294649885618, 'Mean Absolute Error', 44.083506446776603)
('Huber ', 'R-squared: ', 0.99999999999404421, 'Mean Absolute Error', 0.00011755074198335526)
```

如你所见，在存在异常值的情况下，稳健的线性估计器 Theil-Sen、**随机样本一致性**（**RANSAC**）和 Huber 回归器的表现优于其他线性回归器。

# 将一切整合到管道中

现在我们已经使用了管道和数据转换技术，我们将通过一个更复杂的例子，结合之前的几个实例，演示如何将它们组合成一个管道。

# 准备工作

在本节中，我们将展示管道的更多强大功能。当我们之前用它来填补缺失值时，只是简单体验了一下；这里，我们将把多个预处理步骤链起来，展示管道如何去除额外的工作。

让我们简单加载鸢尾花数据集，并给它添加一些缺失值：

```py
from sklearn.datasets import load_iris
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
iris_data = iris.data
mask = np.random.binomial(1, .25, iris_data.shape).astype(bool)
iris_data[mask] = np.nan
iris_data[:5]

array([[ nan,  3.5,  1.4,  0.2],
       [ 4.9,  3\. ,  1.4,  nan],
       [ nan,  3.2,  nan,  nan],
       [ nan,  nan,  1.5,  0.2],
       [ nan,  3.6,  1.4,  0.2]])
```

# 如何实现…

本章的目标是首先填补`iris_data`的缺失值，然后对修正后的数据集执行 PCA。你可以想象（我们稍后会做）这个工作流程可能需要在训练数据集和保留集之间拆分；管道将使这更容易，但首先我们需要迈出小小的一步。

1.  让我们加载所需的库：

```py
from sklearn import pipeline, preprocessing, decomposition
```

1.  接下来，创建`imputer`和`pca`类：

```py
pca = decomposition.PCA()
imputer = preprocessing.Imputer()
```

1.  现在我们已经有了需要的类，我们可以将它们加载到`Pipeline`中：

```py
pipe = pipeline.Pipeline([('imputer', imputer), ('pca', pca)])
iris_data_transformed = pipe.fit_transform(iris_data)
iris_data_transformed[:5]

array([[-2.35980262,  0.6490648 ,  0.54014471,  0.00958185],
       [-2.29755917, -0.00726168, -0.72879348, -0.16408532],
       [-0.00991161,  0.03354407,  0.01597068,  0.12242202],
       [-2.23626369,  0.50244737,  0.50725722, -0.38490096],
       [-2.36752684,  0.67520604,  0.55259083,  0.1049866 ]])
```

如果我们使用单独的步骤，这需要更多的管理。与每个步骤都需要进行拟合转换不同，这个步骤只需执行一次，更不用说我们只需要跟踪一个对象！

# 它是如何工作的…

希望大家已经明白，每个管道中的步骤都是通过元组列表传递给管道对象的，第一个元素是名称，第二个元素是实际的对象。在幕后，当调用像`fit_transform`这样的函数时，这些步骤会在管道对象上循环执行。

话虽如此，确实有一些快速且简便的方式来创建管道，就像我们之前有一种快速的方式来执行缩放操作一样，尽管我们可以使用`StandardScaler`来获得更强大的功能。`pipeline`函数将自动为管道对象创建名称：

```py
pipe2 = pipeline.make_pipeline(imputer, pca)
pipe2.steps

[('imputer',
 Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)),
 ('pca',
 PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
 svd_solver='auto', tol=0.0, whiten=False))]
```

这是在更详细的方法中创建的相同对象：

```py
iris_data_transformed2 = pipe2.fit_transform(iris_data)
iris_data_transformed2[:5]

array([[-2.35980262,  0.6490648 ,  0.54014471,  0.00958185],
       [-2.29755917, -0.00726168, -0.72879348, -0.16408532],
       [-0.00991161,  0.03354407,  0.01597068,  0.12242202],
       [-2.23626369,  0.50244737,  0.50725722, -0.38490096],
       [-2.36752684,  0.67520604,  0.55259083,  0.1049866 ]])
```

# 还有更多…

我们刚刚以很高的层次走过了管道，但不太可能希望直接应用基本的转换。因此，可以通过`set_params`方法访问管道中每个对象的属性，其中参数遵循`<step_name>__<step_parameter>`的约定。例如，假设我们想把`pca`对象改为使用两个主成分：

```py
pipe2.set_params(pca__n_components=2)

Pipeline(steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)), ('pca', PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False))])
```

请注意，前面的输出中有`n_components=2`。作为测试，我们可以输出之前已经做过两次的相同变换，输出将是一个 N x 2 的矩阵：

```py
iris_data_transformed3 = pipe2.fit_transform(iris_data)
iris_data_transformed3[:5]

array([[-2.35980262,  0.6490648 ],
 [-2.29755917, -0.00726168],
 [-0.00991161,  0.03354407],
 [-2.23626369,  0.50244737],
 [-2.36752684,  0.67520604]])
```

# 使用高斯过程进行回归

在这个例子中，我们将使用高斯过程进行回归。在线性模型部分，我们将看到如何通过贝叶斯岭回归表示系数的先验信息。

在高斯过程中，关注的是方差而非均值。然而，我们假设均值为 0，所以我们需要指定的是协方差函数。

基本设置类似于在典型回归问题中如何对系数设置先验。在高斯过程（Gaussian Process）中，可以对数据的函数形式设置先验，数据点之间的协方差用于建模数据，因此必须与数据相匹配。

高斯过程的一个大优点是它们可以进行概率预测：你可以获得预测的置信区间。此外，预测可以插值可用内核的观测值：回归的预测是平滑的，因此两个已知点之间的预测位于这两个点之间。

高斯过程的一个缺点是在高维空间中的效率较低。

# 准备就绪

那么，让我们使用一些回归数据，逐步了解高斯过程如何在 scikit-learn 中工作：

```py
from sklearn.datasets import load_boston
boston = load_boston()
boston_X = boston.data
boston_y = boston.target
train_set = np.random.choice([True, False], len(boston_y),p=[.75, .25])
```

# 如何做到……

1.  我们有数据，将创建一个 scikit-learn 的`GaussianProcessRegressor`对象。让我们看看`gpr`对象：

```py
sklearn.gaussian_process import GaussianProcessRegressor
gpr = GaussianProcessRegressor()
gpr

GaussianProcessRegressor(alpha=1e-10, copy_X_train=True, kernel=None,
             n_restarts_optimizer=0, normalize_y=False,
             optimizer='fmin_l_bfgs_b', random_state=None)
```

有几个重要的参数必须设置：

+   +   `alpha`：这是一个噪声参数。你可以为所有观测值指定一个噪声值，或者以 NumPy 数组的形式分配`n`个值，其中`n`是传递给`gpr`进行训练的训练集目标观测值的长度。

    +   `kernel`：这是一个逼近函数的内核。在 scikit-learn 的早期版本中，默认的内核是**径向基函数**（**RBF**），我们将通过常量内核和 RBF 内核构建一个灵活的内核。

    +   `normalize_y`：如果目标集的均值不为零，可以将其设置为 True。如果设置为 False，效果也相当不错。

    +   `n_restarts_optimizer`：设置为 10-20 以供实际使用。该值表示优化内核时的迭代次数。

1.  导入所需的内核函数并设置一个灵活的内核：

```py
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK

mixed_kernel = kernel = CK(1.0, (1e-4, 1e4)) * RBF(10, (1e-4, 1e4))
```

1.  最后，实例化并拟合算法。请注意，`alpha`对所有值都设置为`5`。我之所以选择这个数字，是因为它大约是目标值的四分之一：

```py
gpr = GaussianProcessRegressor(alpha=5,
 n_restarts_optimizer=20,
 kernel = mixed_kernel)

gpr.fit(boston_X[train_set],boston_y[train_set])
```

1.  将对未见数据的预测存储为`test_preds`：

```py
test_preds = gpr.predict(boston_X[~train_set])
```

1.  绘制结果：

```py
>from sklearn.model_selection import cross_val_predict

from matplotlib import pyplot as plt
%matplotlib inline

f, ax = plt.subplots(figsize=(10, 7), nrows=3)
f.tight_layout()

ax[0].plot(range(len(test_preds)), test_preds,label='Predicted Values');
ax[0].plot(range(len(test_preds)), boston_y[~train_set],label='Actual Values');
ax[0].set_title("Predicted vs Actuals")
ax[0].legend(loc='best')

ax[1].plot(range(len(test_preds)),test_preds - boston_y[~train_set]);
ax[1].set_title("Plotted Residuals")
ax[2].hist(test_preds - boston_y[~train_set]);
ax[2].set_title("Histogram of Residuals")
```

![](img/1a8840b6-2a13-4f42-9d39-ff0de8bb8cf4.png)

# 使用噪声参数进行交叉验证

你可能会想，这是否是最佳的噪声参数`alpha=5`？为了弄清楚这一点，尝试一些交叉验证。

1.  首先，使用`alpha=5`生成交叉验证得分。注意，`cross_val_score`对象中的评分器是`neg_mean_absolute_error`，因为该数据集的默认 R 方得分难以读取：

```py
from sklearn.model_selection import cross_val_score

gpr5 = GaussianProcessRegressor(alpha=5,
 n_restarts_optimizer=20,
 kernel = mixed_kernel)

scores_5 = (cross_val_score(gpr5,
 boston_X[train_set],
 boston_y[train_set],
 cv = 4,
 scoring = 'neg_mean_absolute_error'))
```

1.  查看`scores_5`中的得分：

```py
def score_mini_report(scores_list):
 print "List of scores: ", scores_list
 print "Mean of scores: ", scores_list.mean()
 print "Std of scores: ", scores_list.std()

 score_mini_report(scores_5)

List of scores:  [ -4.10973995  -4.93446898  -3.78162    -13.94513686]
Mean of scores:  -6.69274144767
Std of scores:  4.20818506589
```

注意，最后一次折叠中的得分与其他三次折叠不一样。

1.  现在使用`alpha=7`生成报告：

```py

gpr7 = GaussianProcessRegressor(alpha=7,
 n_restarts_optimizer=20,
 kernel = mixed_kernel)

scores_7 = (cross_val_score(gpr7,
 boston_X[train_set],
 boston_y[train_set],
 cv = 4,
 scoring = 'neg_mean_absolute_error'))

score_mini_report(scores_7)

List of scores:  [ -3.70606009  -4.92211642  -3.63887969 -14.20478333]
Mean of scores:  -6.61795988295
Std of scores:  4.40992783912
```

1.  这个得分看起来更好一些。现在，尝试将`alpha=7`，并将`normalize_y`设置为`True`：

```py
from sklearn.model_selection import cross_val_score

 gpr7n = GaussianProcessRegressor(alpha=7,
 n_restarts_optimizer=20,
 kernel = mixed_kernel,
 normalize_y=True)

 scores_7n = (cross_val_score(gpr7n,
 boston_X[train_set],
 boston_y[train_set],
 cv = 4,
 scoring = 'neg_mean_absolute_error'))
score_mini_report(scores_7n)

List of scores:  [-4.0547601  -4.91077385 -3.65226736 -9.05596047]
Mean of scores:  -5.41844044809
Std of scores:  2.1487361839
```

1.  这看起来更好，因为均值较高，标准差较低。让我们选择最后一个模型进行最终训练：

```py
gpr7n.fit(boston_X[train_set],boston_y[train_set])
```

1.  进行预测：

```py
test_preds = gpr7n.predict(boston_X[~train_set])
```

1.  可视化结果：

![](img/a5930ca5-55ac-412c-8b48-5282cc355af9.png)

1.  残差看起来更加集中。你也可以为 `alpha` 传递一个 NumPy 数组：

```py
gpr_new = GaussianProcessRegressor(alpha=boston_y[train_set]/4,
 n_restarts_optimizer=20,
 kernel = mixed_kernel)
```

1.  这将产生以下图表：

![](img/877acdd9-751d-40ab-a2db-8ab3246271bb.png)

数组 alphas 与 `cross_val_score` 不兼容，因此我无法通过查看最终图形并判断哪一个是最佳模型来选择该模型。所以，我们最终选择的模型是 `gpr7n`，并且设置了 `alpha=7` 和 `normalize_y=True`。

# 还有更多内容……

在这一切的背后，核函数计算了`X`中各点之间的协方差。它假设输入中相似的点应该导致相似的输出。高斯过程在置信度预测和光滑输出方面表现得非常好。（稍后我们将看到随机森林，尽管它们在预测方面非常准确，但并不会产生光滑的输出。）

我们可能需要了解我们估计值的不确定性。如果我们将 `eval_MSE` 参数设置为真，我们将得到 `MSE` 和预测值，从而可以进行预测。从机械学角度来看，返回的是预测值和 `MSE` 的元组：

```py
test_preds, MSE = gpr7n.predict(boston_X[~train_set], return_std=True)
MSE[:5]

array([ 1.20337425,  1.43876578,  1.19910262,  1.35212445,  1.32769539])
```

如下图所示，绘制所有带误差条的预测：

```py
f, ax = plt.subplots(figsize=(7, 5))
n = 133
rng = range(n)
ax.scatter(rng, test_preds[:n])
ax.errorbar(rng, test_preds[:n], yerr=1.96*MSE[:n])
ax.set_title("Predictions with Error Bars")
ax.set_xlim((-1, n));
```

![](img/0116515f-41b8-4bfd-9c78-911c85eac557.png)

在前面的代码中设置 `n=20` 以查看较少的点：

![](img/003714ee-b7a8-4bad-a8d8-81da8b3abe4e.png)

对于某些点，不确定性非常高。如你所见，许多给定点的估计值有很大的差异。然而，整体误差并不算太差。

# 使用 SGD 进行回归

在这个教程中，我们将首次体验随机梯度下降。我们将在回归中使用它。

# 准备就绪

SGD（随机梯度下降）通常是机器学习中的一个默默无闻的英雄。在许多算法背后，正是 SGD 在默默地执行工作。由于其简单性和速度，SGD 非常受欢迎——这两者在处理大量数据时是非常有用的。SGD 的另一个优点是，尽管它在许多机器学习算法的计算核心中扮演重要角色，但它之所以如此有效，是因为它能够简明地描述这一过程。归根结底，我们对数据应用一些变换，然后用损失函数将数据拟合到模型中。

# 如何实现……

1.  如果 SGD 在大数据集上表现良好，我们应该尝试在一个相对较大的数据集上测试它：

```py
from sklearn.datasets import make_regression
X, y = make_regression(int(1e6))  #1,000,000 rows
```

了解对象的组成和大小可能是值得的。幸运的是，我们处理的是 NumPy 数组，因此可以直接访问 `nbytes`。Python 内建的访问对象大小的方法对于 NumPy 数组不起作用。

1.  该输出可能与系统有关，因此你可能无法得到相同的结果：

```py
print "{:,}".format(X.nbytes)

800,000,000
```

1.  为了获得一些人类的视角，我们可以将 `nbytes` 转换为兆字节。大约每兆字节包含 100 万字节：

```py
X.nbytes / 1e6

800
```

1.  因此，每个数据点的字节数如下：

```py
X.nbytes / (X.shape[0]*X.shape[1])

8
```

好吧，对于我们想要实现的目标，这样不显得有点杂乱无章吗？然而，了解如何获取你正在处理的对象的大小是很重要的。

1.  所以，现在我们有了数据，我们可以简单地拟合一个`SGDRegressor`：

```py

from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()
train = np.random.choice([True, False], size=len(y), p=[.75, .25])
sgd.fit(X[train], y[train])

SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
       fit_intercept=True, l1_ratio=0.15, learning_rate='invscaling',
       loss='squared_loss', n_iter=5, penalty='l2', power_t=0.25,
       random_state=None, shuffle=True, verbose=0, warm_start=False)
```

所以，我们有了另一个庞大的对象。现在要了解的主要内容是我们的损失函数是`squared_loss`，这与线性回归中的损失函数相同。还需要注意的是，`shuffle`会对数据进行随机打乱。如果你想打破一个可能存在的虚假相关性，这个功能会非常有用。使用`X`时，scikit-learn 会自动包含一列 1。

1.  然后，我们可以像以前一样，使用 scikit-learn 一致的 API 进行预测。你可以看到我们实际上得到了一个非常好的拟合。几乎没有任何变化，且直方图看起来像是正态分布。

```py
y_pred = sgd.predict(X[~train])

%matplotlib inline
import pandas as pd

pd.Series(y[~train] - y_pred).hist(bins=50)
```

![](img/e0b40007-829c-455a-8bcc-bcb3866d8f80.png)

# 它是如何工作的…

很明显，我们使用的虚拟数据集还不错，但你可以想象一些具有更大规模的数据集。例如，如果你在华尔街工作，在任何一天，某个市场上的交易量可能达到 20 亿笔。现在，想象一下你有一周或一年的数据。处理大量数据时，内存中的算法并不适用。

之所以通常比较困难，是因为要执行 SGD，我们需要在每一步计算梯度。梯度有任何高等微积分课程中的标准定义。

算法的要点是，在每一步中，我们计算一组新的系数，并用学习率和目标函数的结果更新它。在伪代码中，这可能看起来像这样：

```py
while not converged:
 w = w – learning_rate*gradient(cost(w))
```

相关变量如下：

+   `w`：这是系数矩阵。

+   `learning_rate`：这表示每次迭代时步长的大小。如果收敛效果不好，可能需要调整这个值。

+   `gradient`：这是二阶导数的矩阵。

+   `cost`：这是回归的平方误差。稍后我们会看到，这个代价函数可以适应分类任务。这种灵活性是 SGD 如此有用的一个原因。

这不会太难，除了梯度函数很昂贵这一点。随着系数向量的增大，计算梯度变得非常昂贵。对于每一次更新步骤，我们需要为数据中的每一个点计算一个新的权重，然后更新。SGD 的工作方式略有不同；与批量梯度下降的定义不同，我们将每次用新的数据点来更新参数。这个数据点是随机选取的，因此得名随机梯度下降。

关于 SGD 的最后一点是，它是一种元启发式方法，赋予了多种机器学习算法强大的能力。值得查阅一些关于元启发式方法在各种机器学习算法中的应用的论文。前沿解决方案可能就隐藏在这些论文中。
