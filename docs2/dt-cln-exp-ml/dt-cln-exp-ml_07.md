# *第五章*: 特征选择

根据你开始数据分析工作和你的个人智力兴趣的不同，你可能会对**特征选择**这个话题有不同的看法。你可能认为，“嗯，嗯，这是一个重要的主题，但我真的想开始模型构建。”或者，在另一个极端，你可能会认为特征选择是模型构建的核心，并相信一旦你选择了特征，你就已经完成了模型构建的 90%。现在，让我们先达成共识，在我们进行任何严肃的模型指定之前，我们应该花一些时间来理解特征之间的关系——如果我们正在构建监督模型，那么它们与目标之间的关系。

以“少即是多”的态度来处理我们的特征选择工作是有帮助的。如果我们能用更少的特征达到几乎相同的准确度或解释更多的方差，我们应该选择更简单的模型。有时，我们实际上可以用更少的特征获得更好的准确度。这可能会很难理解，甚至对我们这些从构建讲述丰富和复杂故事模型的实践中成长起来的人来说有些令人失望。

但我们在拟合机器学习模型时，对参数估计的关注不如对预测准确性的关注。不必要的特征可能导致过拟合并消耗硬件资源。

有时，我们可能需要花费数月时间来指定模型的特征，即使数据中列的数量有限。例如，在*第二章*“检查特征与目标之间的双变量和多变量关系”中创建的双变量相关性，给我们一些预期的感觉，但一旦引入其他可能的解释特征，特征的重要性可能会显著变化。该特征可能不再显著，或者相反，只有在包含其他特征时才显著。两个特征可能高度相关，以至于包含两个特征与只包含一个特征相比，提供的额外信息非常有限。

本章将深入探讨适用于各种预测建模任务的特征选择技术。具体来说，我们将探讨以下主题：

+   为分类模型选择特征

+   为回归模型选择特征

+   使用正向和反向特征选择

+   使用穷举特征选择

+   在回归模型中递归消除特征

+   在分类模型中递归消除特征

+   使用 Boruta 进行特征选择

+   使用正则化和其他嵌入式方法

+   使用主成分分析

# 技术要求

本章中，我们将使用`feature_engine`、`mlxtend`和`boruta`包，以及`scikit-learn`库。您可以使用`pip`安装这些包。我选择了一个观测值数量较少的数据集用于本章的工作，因此代码即使在次优工作站上也能正常运行。

注意

在本章中，我们将专门使用美国劳工统计局进行的《青年纵向调查》数据。这项调查始于 1997 年，调查对象为 1980 年至 1985 年间出生的一代人，每年进行一次年度跟踪调查，直至 2017 年。我们将使用教育成就、家庭人口统计、工作周数和工资收入数据。工资收入列代表 2016 年赚取的工资。NLS 数据集可以下载供公众使用，网址为[`www.nlsinfo.org/investigator/pages/search`](https://www.nlsinfo.org/investigator/pages/search)。

# 为分类模型选择特征

最直接的特征选择方法是基于每个特征与目标变量的关系。接下来的两个部分将探讨基于特征与目标变量之间的线性或非线性关系来确定最佳*k*个特征的技术。这些被称为过滤方法。它们有时也被称为单变量方法，因为它们评估特征与目标变量之间的关系，而不考虑其他特征的影响。

当目标变量为分类变量时，我们使用的策略与目标变量为连续变量时有所不同。在本节中，我们将介绍前者，在下一节中介绍后者。

## 基于分类目标的互信息特征选择

当目标变量为分类变量时，我们可以使用**互信息**分类或**方差分析**（**ANOVA**）测试来选择特征。我们将首先尝试互信息分类，然后进行 ANOVA 比较。

互信息是衡量通过知道另一个变量的值可以获得多少关于变量的信息的度量。在极端情况下，当特征完全独立时，互信息分数为 0。

我们可以使用`scikit-learn`的`SelectKBest`类根据互信息分类或其他适当的度量选择具有最高预测强度的*k*个特征。我们可以使用超参数调整来选择*k*的值。我们还可以检查所有特征的分数，无论它们是否被识别为*k*个最佳特征之一，正如我们将在本节中看到的。

让我们先尝试互信息分类来识别与完成学士学位相关的特征。稍后，我们将将其与使用 ANOVA F 值作为选择依据进行比较：

1.  我们首先从`feature_engine`导入`OneHotEncoder`来编码一些数据，并从`scikit-learn`导入`train_test_split`来创建训练和测试数据。我们还需要`scikit-learn`的`SelectKBest`、`mutual_info_classif`和`f_classif`模块来进行特征选择：

    ```py
    import pandas as pd
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest,\
      mutual_info_classif, f_classif
    ```

1.  我们加载了具有完成学士学位的二进制变量和可能与学位获得相关的特征的数据集：`gender`特征，并对其他数据进行缩放：

    ```py
    nls97compba = pd.read_csv("data/nls97compba.csv")
    feature_cols = ['gender','satverbal','satmath',
      'gpascience', 'gpaenglish','gpamath','gpaoverall',
      'motherhighgrade','fatherhighgrade','parentincome']
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97compba[feature_cols],\
      nls97compba[['completedba']], test_size=0.3, random_state=0)
    ohe = OneHotEncoder(drop_last=True, variables=['gender'])
    X_train_enc = ohe.fit_transform(X_train)
    scaler = StandardScaler()
    standcols = X_train_enc.iloc[:,:-1].columns
    X_train_enc = \
      pd.DataFrame(scaler.fit_transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Female']])
    ```

    注意

    在本章中，我们将对 NLS 数据进行完整案例分析；也就是说，我们将删除任何特征缺失的观测值。这通常不是一个好的方法，尤其是在数据不是随机缺失或一个或多个特征有大量缺失值时尤其有问题。在这种情况下，最好使用我们在*第三章*中使用的某些方法，*识别和修复缺失值*。我们将在本章中进行完整案例分析，以使示例尽可能简单。

1.  现在，我们已经准备好为我们的学士学位完成模型选择特征。一种方法是用互信息分类。为此，我们将`SelectKBest`的`score_func`值设置为`mutual_info_classif`，并指出我们想要五个最佳特征。然后，我们调用`fit`并使用`get_support`方法来获取五个最佳特征：

    ```py
    ksel = SelectKBest(score_func=mutual_info_classif, k=5)
    ksel.fit(X_train_enc, y_train.values.ravel())
    selcols = X_train_enc.columns[ksel.get_support()]
    selcols
    Index(['satverbal', 'satmath', 'gpascience', 'gpaenglish', 'gpaoverall'], dtype='object')
    ```

1.  如果我们还想看到每个特征的得分，我们可以使用`scores_`属性，尽管我们需要做一些工作来将得分与特定的特征名称关联起来，并按降序排序：

    ```py
    pd.DataFrame({'score': ksel.scores_,
      'feature': X_train_enc.columns},
       columns=['feature','score']).\
       sort_values(['score'], ascending=False)
            feature              score
    5       gpaoverall           0.108
    1       satmath              0.074
    3       gpaenglish           0.072
    0       satverbal            0.069
    2       gpascience           0.047
    4       gpamath              0.038
    8       parentincome         0.024
    7       fatherhighgrade      0.022
    6       motherhighgrade      0.022
    9       gender_Female        0.015
    ```

    注意

    这是一个随机过程，所以每次运行它时我们都会得到不同的结果。

为了每次都能得到相同的结果，你可以将一个部分函数传递给`score_func`：

```py
from functools import partial
SelectKBest(score_func=partial(mutual_info_classif, 
                               random_state=0), k=5) 
```

1.  我们可以使用使用`get_support`创建的`selcols`数组来创建仅包含重要特征的 DataFrame。（我们也可以使用`SelectKBest`的`transform`方法。这将返回所选特征的值作为 NumPy 数组。）

    ```py
    X_train_analysis = X_train_enc[selcols] 
    X_train_analysis.dtypes
    satverbal       float64
    satmath         float64
    gpascience      float64
    gpaenglish      float64
    gpaoverall      float64
    dtype: object
    ```

这就是我们使用互信息来选择模型中*最佳 k 个特征*所需做的所有事情。

## 使用分类目标的特征选择的 ANOVA F 值

或者，我们可以使用方差分析（ANOVA）而不是互信息。方差分析评估每个目标类中特征的平均值差异。当我们假设特征和目标之间存在线性关系，并且我们的特征是正态分布时，这是一个很好的单变量特征选择指标。如果这些假设不成立，互信息分类是一个更好的选择。

让我们尝试使用 ANOVA 进行特征选择。我们可以将`SelectKBest`的`score_func`参数设置为`f_classif`，以便基于 ANOVA 进行选择：

```py
ksel = SelectKBest(score_func=f_classif, k=5)
```

```py
ksel.fit(X_train_enc, y_train.values.ravel())
```

```py
selcols = X_train_enc.columns[ksel.get_support()]
```

```py
selcols
```

```py
Index(['satverbal', 'satmath', 'gpascience', 'gpaenglish', 'gpaoverall'], dtype='object')
```

```py
pd.DataFrame({'score': ksel.scores_,
```

```py
  'feature': X_train_enc.columns},
```

```py
   columns=['feature','score']).\
```

```py
   sort_values(['score'], ascending=False)
```

```py
       feature                score
```

```py
5      gpaoverall           119.471
```

```py
3      gpaenglish           108.006
```

```py
2      gpascience            96.824
```

```py
1      satmath               84.901
```

```py
0      satverbal             77.363
```

```py
4      gpamath               60.930
```

```py
7      fatherhighgrade       37.481
```

```py
6      motherhighgrade       29.377
```

```py
8      parentincome          22.266
```

```py
9      gender_Female         15.098
```

这选择了与我们使用互信息时选择的相同特征。显示得分给我们一些关于所选的*k*值是否合理的指示。例如，第五到第六个最佳特征的得分下降（77-61）比第四到第五个（85-77）的下降更大。然而，从第六到第七个的下降更大（61-37），这表明我们至少应该考虑*k*的值为 6。

ANOVA 测试和之前我们做的互信息分类没有考虑在多元分析中仅重要的特征。例如，`fatherhighgrade`可能在具有相似 GPA 或 SAT 分数的个人中很重要。我们将在本章后面使用多元特征选择方法。在下一节中，我们将进行更多单变量特征选择，以探索适合连续目标的特征选择技术。

# 选择回归模型的特征

`scikit-learn`的选择模块在构建回归模型时提供了几个选择特征的选择。在这里，我不指线性回归模型。我只是在指*具有连续目标的模型*）。两个好的选择是基于 F 检验的选择和基于回归的互信息选择。让我们从 F 检验开始。

## 基于连续目标的特征选择的 F 检验

F 统计量是目标与单个回归器之间线性相关强度的度量。`Scikit-learn`有一个`f_regression`评分函数，它返回 F 统计量。我们可以使用它与`SelectKBest`一起选择基于该统计量的特征。

让我们使用 F 统计量来选择工资模型的特征。我们将在下一节中使用互信息来选择相同目标的特征：

1.  我们首先从`feature_engine`导入 one-hot 编码器，从`scikit-learn`导入`train_test_split`和`SelectKBest`。我们还导入`f_regression`以获取后续的 F 统计量：

    ```py
    import pandas as pd
    import numpy as np
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    ```

1.  接下来，我们加载 NLS 数据，包括教育成就、家庭收入和工资收入数据：

    ```py
    nls97wages = pd.read_csv("data/nls97wages.csv")
    feature_cols = ['satverbal','satmath','gpascience',
      'gpaenglish','gpamath','gpaoverall','gender',
      'motherhighgrade','fatherhighgrade','parentincome',
      'completedba']
    ```

1.  然后，我们创建训练和测试数据框，对`gender`特征进行编码，并对训练数据进行缩放。在这种情况下，我们需要对目标进行缩放，因为它是有连续性的：

    ```py
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97wages[feature_cols],\
      nls97wages[['wageincome']], test_size=0.3, random_state=0)
    ohe = OneHotEncoder(drop_last=True, variables=['gender'])
    X_train_enc = ohe.fit_transform(X_train)
    scaler = StandardScaler()
    standcols = X_train_enc.iloc[:,:-1].columns
    X_train_enc = \
      pd.DataFrame(scaler.fit_transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Male']])
    y_train = \
      pd.DataFrame(scaler.fit_transform(y_train),
      columns=['wageincome'], index=y_train.index)
    ```

    注意

    你可能已经注意到我们没有对测试数据进行编码或缩放。我们最终需要这样做以验证我们的模型。我们将在本章后面介绍验证，并在下一章中详细介绍。

1.  现在，我们已准备好选择特征。我们将`SelectKBest`的`score_func`设置为`f_regression`，并指出我们想要五个最佳特征。`SelectKBest`的`get_support`方法对每个被选中的特征返回`True`：

    ```py
    ksel = SelectKBest(score_func=f_regression, k=5)
    ksel.fit(X_train_enc, y_train.values.ravel())
    selcols = X_train_enc.columns[ksel.get_support()]
    selcols
    Index(['satmath', 'gpascience', 'parentincome',
     'completedba','gender_Male'],
          dtype='object')
    ```

1.  我们可以使用`scores_`属性来查看每个特征的得分：

    ```py
    pd.DataFrame({'score': ksel.scores_,
      'feature': X_train_enc.columns},
       columns=['feature','score']).\
       sort_values(['score'], ascending=False)

                  feature              score
    1             satmath              45
    9             completedba          38
    10            gender_Male          26
    8             parentincome         24
    2             gpascience           21
    0             satverbal            19
    5             gpaoverall           17
    4             gpamath              13
    3             gpaenglish           10
    6             motherhighgrade       9
    7             fatherhighgrade       8
    ```

F 统计量的缺点是它假设每个特征与目标之间存在线性关系。当这个假设不合理时，我们可以使用互信息进行回归。

## 对于具有连续目标的特征选择中的互信息

我们还可以使用`SelectKBest`通过回归中的互信息来选择特征：

1.  我们需要将`SelectKBest`的`score_func`参数设置为`mutual_info_regression`，但存在一个小问题。为了每次运行特征选择时都能得到相同的结果，我们需要设置一个`random_state`值。正如我们在前一小节中讨论的，我们可以使用一个部分函数来做到这一点。我们将`partial(mutual_info_regression, random_state=0)`传递给评分函数。

1.  我们可以运行`fit`方法，并使用`get_support`来获取选定的特征。我们可以使用`scores_`属性来为每个特征给出分数：

    ```py
    from functools import partial
    ksel = SelectKBest(score_func=\
      partial(mutual_info_regression, random_state=0),
      k=5)
    ksel.fit(X_train_enc, y_train.values.ravel())
    selcols = X_train_enc.columns[ksel.get_support()]
    selcols
    Index(['satmath', 'gpascience', 'fatherhighgrade', 'completedba','gender_Male'],dtype='object')
    pd.DataFrame({'score': ksel.scores_,
      'feature': X_train_enc.columns},
       columns=['feature','score']).\
       sort_values(['score'], ascending=False)
               feature               score
    1          satmath               0.101
    10         gender_Male           0.074
    7          fatherhighgrade       0.047
    2          gpascience            0.044
    9          completedba           0.044
    4          gpamath               0.016
    8          parentincome          0.015
    6          motherhighgrade       0.012
    0          satverbal             0.000
    3          gpaenglish            0.000
    5          gpaoverall            0.000
    ```

我们在回归中的互信息得到了与 F 检验相当相似的结果。`parentincome`通过 F 检验被选中，而`fatherhighgrade`通过互信息被选中。否则，选中的特征是相同的。

与 F 检验相比，互信息在回归中的关键优势是它不假设特征与目标之间存在线性关系。如果这个假设被证明是不合理的，互信息是一个更好的方法。（再次强调，评分过程中也存在一些随机性，每个特征的分数可能会在一定范围内波动。）

注意

我们选择`k=5`以获取五个最佳特征是非常随意的。我们可以通过一些超参数调整使其更加科学。我们将在下一章中介绍调整。

我们迄今为止使用的特征选择方法被称为*过滤器方法*。它们检查每个特征与目标之间的单变量关系。它们是一个好的起点。类似于我们在前几章中讨论的，在开始检查多元关系之前，拥有相关性的有用性，至少探索过滤器方法是有帮助的。然而，通常我们的模型拟合需要考虑当其他特征也被包含时，哪些特征是重要的，哪些不是。为了做到这一点，我们需要使用包装器或嵌入式方法进行特征选择。我们将在下一节中探讨包装器方法，从前向和后向特征选择开始。

# 使用前向和后向特征选择

前向和后向特征选择，正如其名称所暗示的，通过逐个添加（或对于后向选择，逐个减去）特征来选择特征，并在每次迭代后评估对模型性能的影响。由于这两种方法都是基于给定的算法来评估性能，因此它们被认为是**包装器**选择方法。

包装特征选择方法相对于我们之前探索的过滤方法有两个优点。首先，它们在包含其他特征时评估特征的重要性。其次，由于特征是根据其对特定算法性能的贡献来评估的，因此我们能够更好地了解哪些特征最终会起作用。例如，根据我们上一节的结果，`satmath`似乎是一个重要的特征。但有可能`satmath`只有在使用特定模型时才重要，比如线性回归，而不是决策树回归等其他模型。包装选择方法可以帮助我们发现这一点。

包装方法的缺点主要在于它们在每次迭代后都会重新训练模型，因此在计算上可能相当昂贵。在本节中，我们将探讨前向和后向特征选择。

## 使用前向特征选择

**前向特征选择**首先识别出与目标有显著关系的特征子集，这与过滤方法类似。但它随后评估所有可能的选择特征的组合，以确定与所选算法表现最佳的组合。

我们可以使用前向特征选择来开发一个完成学士学位的模型。由于包装方法要求我们选择一个算法，而这是一个二元目标，因此让我们使用`scikit-learn`的`mlxtend`模块中的`feature_selection`来进行选择特征的迭代：

1.  我们首先导入必要的库：

    ```py
    import pandas as pd
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.feature_selection import SequentialFeatureSelector
    ```

1.  然后，我们再次加载 NLS 数据。我们还创建了一个训练 DataFrame，对`gender`特征进行编码，并对剩余特征进行标准化：

    ```py
    nls97compba = pd.read_csv("data/nls97compba.csv")
    feature_cols = ['satverbal','satmath','gpascience',
      'gpaenglish','gpamath','gpaoverall','gender',
      'motherhighgrade','fatherhighgrade','parentincome']
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97compba[feature_cols],\
      nls97compba[['completedba']], test_size=0.3, random_state=0)
    ohe = OneHotEncoder(drop_last=True, variables=['gender'])
    X_train_enc = ohe.fit_transform(X_train)
    scaler = StandardScaler()
    standcols = X_train_enc.iloc[:,:-1].columns
    X_train_enc = \
      pd.DataFrame(scaler.fit_transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Female']])
    ```

1.  我们创建一个随机森林分类器对象，然后将该对象传递给`mlxtend`的特征选择器。我们指出我们想要选择五个特征，并且应该进行前向选择。（我们也可以使用顺序特征选择器进行后向选择。）运行`fit`后，我们可以使用`k_feature_idx_`属性来获取所选特征的列表：

    ```py
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    sfs = SequentialFeatureSelector(rfc, k_features=5,
      forward=True, floating=False, verbose=2,
      scoring='accuracy', cv=5)
    sfs.fit(X_train_enc, y_train.values.ravel())
    selcols = X_train_enc.columns[list(sfs.k_feature_idx_)]
    selcols
    Index(['satverbal', 'satmath', 'gpaoverall',
    'parentincome', 'gender_Female'], dtype='object')
    ```

你可能还记得本章的第一节，我们针对完成学士学位目标的多变量特征选择给出了不同的结果：

```py
Index(['satverbal', 'satmath', 'gpascience',
 'gpaenglish', 'gpaoverall'], dtype='object')
```

有三个特征——`satmath`、`satverbal`和`gpaoverall`——是相同的。但我们的前向特征选择已经将`parentincome`和`gender_Female`识别为比在单变量分析中选择的`gpascience`和`gpaenglish`更重要的特征。实际上，`gender_Female`在早期分析中的得分最低。这些差异可能反映了包装特征选择方法的优点。我们可以识别出除非包含其他特征，否则不重要的特征，并且我们正在评估对特定算法（在这种情况下是随机森林分类）性能的影响。

前向选择的缺点之一是，一旦选择了特征，它就不会被移除，即使随着更多特征的添加，它的重要性可能会下降。（回想一下，前向特征选择是基于该特征对模型的贡献迭代添加特征的。）

让我们看看我们的结果是否随着反向特征选择而变化。

## 使用反向特征选择

反向特征选择从所有特征开始，并消除最不重要的特征。然后，它使用剩余的特征重复此过程。我们可以使用 `mlxtend` 的 `SequentialFeatureSelector` 以与正向选择相同的方式用于反向选择。

我们从 `scikit-learn` 库实例化了一个 `RandomForestClassifier` 对象，然后将其传递给 `mlxtend` 的顺序特征选择器：

```py
rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
```

```py
sfs = SequentialFeatureSelector(rfc, k_features=5,
```

```py
  forward=False, floating=False, verbose=2,
```

```py
  scoring='accuracy', cv=5)
```

```py
sfs.fit(X_train_enc, y_train.values.ravel())
```

```py
selcols = X_train_enc.columns[list(sfs.k_feature_idx_)]
```

```py
selcols
```

```py
Index(['satverbal', 'gpascience', 'gpaenglish',
```

```py
 'gpaoverall', 'gender_Female'], dtype='object')
```

也许并不令人惊讶，我们在特征选择上得到了不同的结果。`satmath` 和 `parentincome` 不再被选中，而 `gpascience` 和 `gpaenglish` 被选中。

反向特征选择与前向特征选择的缺点相反。*一旦移除了特征，它就不会被重新评估，即使其重要性可能会随着不同的特征组合而改变*。让我们尝试使用穷举特征选择。

# 使用穷举特征选择

如果你的正向和反向选择的结果没有说服力，而且你不在意在喝咖啡或吃午餐的时候运行模型，你可以尝试穷举特征选择。**穷举特征选择**会在所有可能的特征组合上训练给定的模型，并选择最佳的特征子集。但这也需要付出代价。正如其名所示，这个过程可能会耗尽系统资源和你的耐心。

让我们为学士学位完成情况的模型使用穷举特征选择：

1.  我们首先加载所需的库，包括来自 `scikit-learn` 的 `RandomForestClassifier` 和 `LogisticRegression` 模块，以及来自 `mlxtend` 的 `ExhaustiveFeatureSelector`。我们还导入了 `accuracy_score` 模块，这样我们就可以使用选定的特征来评估模型：

    ```py
    import pandas as pd
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from mlxtend.feature_selection import ExhaustiveFeatureSelector
    from sklearn.metrics import accuracy_score
    ```

1.  接下来，我们加载 NLS 教育达成度数据，并创建训练和测试 DataFrame：

    ```py
    nls97compba = pd.read_csv("data/nls97compba.csv")
    feature_cols = ['satverbal','satmath','gpascience',
      'gpaenglish','gpamath','gpaoverall','gender',
      'motherhighgrade','fatherhighgrade','parentincome']
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97compba[feature_cols],\
      nls97compba[['completedba']], test_size=0.3, random_state=0)
    ```

1.  然后，我们对训练和测试数据进行编码和缩放：

    ```py
    ohe = OneHotEncoder(drop_last=True, variables=['gender'])
    ohe.fit(X_train)
    X_train_enc, X_test_enc = \
      ohe.transform(X_train), ohe.transform(X_test)
    scaler = StandardScaler()
    standcols = X_train_enc.iloc[:,:-1].columns
    scaler.fit(X_train_enc[standcols])
    X_train_enc = \
      pd.DataFrame(scaler.transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Female']])
    X_test_enc = \
      pd.DataFrame(scaler.transform(X_test_enc[standcols]),
      columns=standcols, index=X_test_enc.index).\
      join(X_test_enc[['gender_Female']])
    ```

1.  我们创建了一个随机森林分类器对象，并将其传递给 `mlxtend` 的 `ExhaustiveFeatureSelector`。我们告诉特征选择器评估所有一至五个特征的组合，并返回预测学位达成度最高的组合。运行 `fit` 后，我们可以使用 `best_feature_names_` 属性来获取选定的特征：

    ```py
    rfc = RandomForestClassifier(n_estimators=100, max_depth=2,n_jobs=-1, random_state=0)
    efs = ExhaustiveFeatureSelector(rfc, max_features=5,
      min_features=1, scoring='accuracy', 
      print_progress=True, cv=5)
    efs.fit(X_train_enc, y_train.values.ravel())
    efs.best_feature_names_
    ('satverbal', 'gpascience', 'gpamath', 'gender_Female')
    ```

1.  让我们评估这个模型的准确性。我们首先需要将训练和测试数据转换为只包含四个选定的特征。然后，我们可以仅使用这些特征再次拟合随机森林分类器，并生成学士学位完成情况的预测值。然后，我们可以计算我们正确预测目标的时间百分比，这是 67%：

    ```py
    X_train_efs = efs.transform(X_train)
    X_test_efs = efs.transform(X_test)
    rfc.fit(X_train_efs, y_train.values.ravel())
    y_pred = rfc.predict(X_test_efs)
    confusion = pd.DataFrame(y_pred, columns=['pred'],
      index=y_test.index).\
      join(y_test)
    confusion.loc[confusion.pred==confusion.completedba].shape[0]\
      /confusion.shape[0]
    0.6703296703296703
    ```

1.  如果我们只使用 scikit-learn 的`accuracy score`，我们也会得到相同的答案。（我们在上一步计算它，因为它相当直接，并且让我们更好地理解在这种情况下准确率的含义。）

    ```py
    accuracy_score(y_test, y_pred)
    0.6703296703296703
    ```

    注意

    准确率分数通常用于评估分类模型的性能。在本章中，我们将依赖它，但根据您模型的目的，其他指标可能同样重要或更重要。例如，我们有时更关心灵敏度，即我们的正确阳性预测与实际阳性数量的比率。我们在*第六章*中详细探讨了分类模型的评估，*准备模型评估*。

1.  现在我们尝试使用逻辑模型进行全面特征选择：

    ```py
    lr = LogisticRegression(solver='liblinear')
    efs = ExhaustiveFeatureSelector(lr, max_features=5,
      min_features=1, scoring='accuracy', 
      print_progress=True, cv=5)
    efs.fit(X_train_enc, y_train.values.ravel())
    efs.best_feature_names_
    ('satmath', 'gpascience', 'gpaenglish', 'motherhighgrade', 'gender_Female')
    ```

1.  让我们看看逻辑模型的准确率。我们得到了相当相似的准确率分数：

    ```py
    X_train_efs = efs.transform(X_train_enc)
    X_test_efs = efs.transform(X_test_enc)
    lr.fit(X_train_efs, y_train.values.ravel())
    y_pred = lr.predict(X_test_efs)
    accuracy_score(y_test, y_pred)
    0.6923076923076923
    ```

1.  逻辑模型的一个关键优势是它训练得更快，这对于全面特征选择来说确实有很大影响。如果我们为每个模型计时（除非你的电脑相当高端或者你不在乎离开电脑一会儿，否则这通常不是一个好主意），我们会看到平均训练时间有显著差异——从随机森林的惊人的 5 分钟到逻辑回归的 4 秒。（当然，这些绝对数字取决于机器。）

    ```py
    rfc = RandomForestClassifier(n_estimators=100, max_depth=2, 
      n_jobs=-1, random_state=0)
    efs = ExhaustiveFeatureSelector(rfc, max_features=5,
      min_features=1, scoring='accuracy', 
      print_progress=True, cv=5)
    %timeit efs.fit(X_train_enc, y_train.values.ravel())
    5min 8s ± 3 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    lr = LogisticRegression(solver='liblinear')
    efs = ExhaustiveFeatureSelector(lr, max_features=5,
      min_features=1, scoring='accuracy', 
      print_progress=True, cv=5)
    %timeit efs.fit(X_train_enc, y_train.values.ravel())
    4.29 s ± 45.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    ```

如我所述，全面特征选择可以提供关于要选择哪些特征的非常清晰的指导，但这可能对许多项目来说代价太高。实际上，它可能更适合于*诊断工作*而不是用于机器学习管道。如果一个线性模型是合适的，它可以显著降低计算成本。

前向、后向和全面特征选择等包装方法会消耗系统资源，因为它们需要每次迭代时都进行训练，而选择的算法越难实现，这个问题就越严重。**递归特征消除（RFE**）在过滤方法的简单性和包装方法提供的信息之间是一种折衷。它与后向特征选择类似，但它在每次迭代中通过基于模型的整体性能而不是重新评估每个特征来简化特征的移除。我们将在下一节中探讨递归特征选择。

# 在回归模型中递归消除特征

一个流行的包装方法是 RFE。这种方法从所有特征开始，移除权重最低的一个（基于系数或特征重要性度量），然后重复此过程，直到确定最佳拟合模型。当移除一个特征时，它会得到一个反映其移除点的排名。

RFE 可以用于回归模型和分类模型。我们将从在回归模型中使用它开始：

1.  我们导入必要的库，其中三个我们尚未使用：来自 `scikit-learn` 的 `RFE`、`RandomForestRegressor` 和 `LinearRegression` 模块：

    ```py
    import pandas as pd
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    ```

1.  接下来，我们加载工资的 NLS 数据并创建训练和测试 DataFrame：

    ```py
    nls97wages = pd.read_csv("data/nls97wages.csv")
    feature_cols = ['satverbal','satmath','gpascience',
      'gpaenglish','gpamath','gpaoverall','motherhighgrade',
      'fatherhighgrade','parentincome','gender','completedba']
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97wages[feature_cols],\
      nls97wages[['weeklywage']], test_size=0.3, random_state=0)
    ```

1.  我们需要编码 `gender` 特征并标准化其他特征以及目标（`wageincome`）。我们不编码或缩放二进制特征 `completedba`：

    ```py
    ohe = OneHotEncoder(drop_last=True, variables=['gender'])
    ohe.fit(X_train)
    X_train_enc, X_test_enc = \
      ohe.transform(X_train), ohe.transform(X_test)
    scaler = StandardScaler()
    standcols = feature_cols[:-2]
    scaler.fit(X_train_enc[standcols])
    X_train_enc = \
      pd.DataFrame(scaler.transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Male','completedba']])
    X_test_enc = \
      pd.DataFrame(scaler.transform(X_test_enc[standcols]),
      columns=standcols, index=X_test_enc.index).\
      join(X_test_enc[['gender_Male','completedba']])
    scaler.fit(y_train)
    y_train, y_test = \
      pd.DataFrame(scaler.transform(y_train),
      columns=['weeklywage'], index=y_train.index),\
      pd.DataFrame(scaler.transform(y_test),
      columns=['weeklywage'], index=y_test.index)
    ```

现在，我们准备进行一些递归特征选择。由于 RFE 是一种包装方法，我们需要选择一个算法，该算法将围绕选择进行包装。在这种情况下，回归的随机森林是有意义的。我们正在模拟一个连续的目标，并且不希望假设特征和目标之间存在线性关系。

1.  使用 `scikit-learn` 实现 RFE 比较简单。我们实例化一个 RFE 对象，在过程中指定我们想要的估计器。我们指示 `RandomForestRegressor`。然后我们拟合模型并使用 `get_support` 获取选定的特征。我们将 `max_depth` 限制为 `2` 以避免过拟合：

    ```py
    rfr = RandomForestRegressor(max_depth=2)
    treesel = RFE(estimator=rfr, n_features_to_select=5)
    treesel.fit(X_train_enc, y_train.values.ravel())
    selcols = X_train_enc.columns[treesel.get_support()]
    selcols
     Index(['satmath', 'gpaoverall', 'parentincome', 'gender_Male', 'completedba'], dtype='object')
    ```

注意，这与使用带有 F 检验的滤波方法（针对工资收入目标）得到的特征列表略有不同。在这里选择了 `gpaoverall` 和 `motherhighgrade`，而不是 `gender` 标志或 `gpascience`。

1.  我们可以使用 `ranking_` 属性来查看每个被消除的特征何时被移除：

    ```py
    pd.DataFrame({'ranking': treesel.ranking_,
      'feature': X_train_enc.columns},
       columns=['feature','ranking']).\
       sort_values(['ranking'], ascending=True)
               feature                ranking
    1          satmath                1
    5          gpaoverall             1
    8          parentincome           1
    9          gender_Male            1
    10         completedba            1
    6          motherhighgrade        2
    2          gpascience             3
    0          satverbal              4
    3          gpaenglish             5
    4          gpamath                6
    7          fatherhighgrade        7
    ```

在第一次交互后移除了 `fatherhighgrade`，在第二次交互后移除了 `gpamath`。

1.  让我们运行一些测试统计量。我们仅在随机森林回归器模型上拟合选定的特征。RFE 选择器的 `transform` 方法给我们的是 `treesel.transform(X_train_enc)` 中选定的特征。我们可以使用 `score` 方法来获取 r 平方值，也称为确定系数。r 平方是我们模型解释的总变异百分比的度量。我们得到了一个非常低的分数，表明我们的模型只解释了很少的变异。（请注意，这是一个随机过程，所以我们每次拟合模型时可能会得到不同的结果。）

    ```py
    rfr.fit(treesel.transform(X_train_enc), y_train.values.ravel())
    rfr.score(treesel.transform(X_test_enc), y_test)
    0.13612629794428466
    ```

1.  让我们看看使用带有线性回归模型的 RFE 是否能得到更好的结果。此模型返回与随机森林回归器相同的特征：

    ```py
    lr = LinearRegression()
    lrsel = RFE(estimator=lr, n_features_to_select=5)
    lrsel.fit(X_train_enc, y_train)
    selcols = X_train_enc.columns[lrsel.get_support()]
    selcols
    Index(['satmath', 'gpaoverall', 'parentincome', 'gender_Male', 'completedba'], dtype='object')
    ```

1.  让我们评估线性模型：

    ```py
    lr.fit(lrsel.transform(X_train_enc), y_train)
    lr.score(lrsel.transform(X_test_enc), y_test)
    0.17773742846314056
    ```

线性模型实际上并不比随机森林模型好多少。这可能是这样一个迹象，即我们可用的特征总体上只捕捉到每周工资变异的一小部分。这是一个重要的提醒，即我们可以识别出几个显著的特征，但仍然有一个解释力有限的模型。（也许这也是一个好消息，即我们的标准化测试分数，甚至我们的学位获得，虽然重要但不是多年后我们工资的决定性因素。）

让我们尝试使用分类模型进行 RFE。

# 在分类模型中递归消除特征

RFE 也可以是分类问题的一个很好的选择。我们可以使用 RFE 来选择完成学士学位模型的特征。你可能还记得，我们在本章前面使用穷举特征选择来选择该模型的特征。让我们看看使用 RFE 是否能获得更高的准确率或更容易训练的模型：

1.  我们导入本章迄今为止一直在使用的相同库：

    ```py
    import pandas as pd
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    from sklearn.metrics import accuracy_score
    ```

1.  接下来，我们从 NLS 教育成就数据中创建训练和测试数据：

    ```py
    nls97compba = pd.read_csv("data/nls97compba.csv")
    feature_cols = ['satverbal','satmath','gpascience',
      'gpaenglish','gpamath','gpaoverall','gender',
      'motherhighgrade','fatherhighgrade','parentincome']
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97compba[feature_cols],\
      nls97compba[['completedba']], test_size=0.3, 
      random_state=0)
    ```

1.  然后，我们编码和缩放训练和测试数据：

    ```py
    ohe = OneHotEncoder(drop_last=True, variables=['gender'])
    ohe.fit(X_train)
    X_train_enc, X_test_enc = \
      ohe.transform(X_train), ohe.transform(X_test)
    scaler = StandardScaler()
    standcols = X_train_enc.iloc[:,:-1].columns
    scaler.fit(X_train_enc[standcols])
    X_train_enc = \
      pd.DataFrame(scaler.transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Female']])
    X_test_enc = \
      pd.DataFrame(scaler.transform(X_test_enc[standcols]),
      columns=standcols, index=X_test_enc.index).\
      join(X_test_enc[['gender_Female']])
    ```

1.  我们实例化一个随机森林分类器并将其传递给 RFE 选择方法。然后我们可以拟合模型并获取所选特征。

    ```py
    rfc = RandomForestClassifier(n_estimators=100, max_depth=2, 
      n_jobs=-1, random_state=0)
    treesel = RFE(estimator=rfc, n_features_to_select=5)
    treesel.fit(X_train_enc, y_train.values.ravel())
    selcols = X_train_enc.columns[treesel.get_support()]
    selcols
    Index(['satverbal', 'satmath', 'gpascience', 'gpaenglish', 'gpaoverall'], dtype='object')
    ```

1.  我们还可以使用 RFE 的`ranking_`属性来展示特征的排名：

    ```py
    pd.DataFrame({'ranking': treesel.ranking_,
      'feature': X_train_enc.columns},
       columns=['feature','ranking']).\
       sort_values(['ranking'], ascending=True)

               feature                 ranking
    0          satverbal               1
    1          satmath                 1
    2          gpascience              1
    3          gpaenglish              1
    5          gpaoverall              1
    4          gpamath                 2
    8          parentincome            3
    7          fatherhighgrade         4
    6          motherhighgrade         5
    9          gender_Female           6
    ```

1.  让我们看看使用与我们的基线模型相同的随机森林分类器，使用所选特征的模型的准确率：

    ```py
    rfc.fit(treesel.transform(X_train_enc), y_train.values.ravel())
    y_pred = rfc.predict(treesel.transform(X_test_enc))
    accuracy_score(y_test, y_pred)
    0.684981684981685
    ```

回想一下，我们使用穷举特征选择获得了 67%的准确率。这里我们得到的准确率大致相同。然而，RFE 的好处是它比穷举特征选择更容易训练。

包装和类似包装特征选择方法中的另一种选择是`scikit-learn`集成方法。我们将在下一节中使用`scikit-learn`的随机森林分类器来使用它。

# 使用 Boruta 进行特征选择

Boruta 包在特征选择方面采用独特的方法，尽管它与包装方法有一些相似之处。对于每个特征，Boruta 创建一个影子特征，它与原始特征具有相同的值范围，但具有打乱后的值。然后它评估原始特征是否比影子特征提供更多信息，逐渐移除提供最少信息的特征。Boruta 在每个迭代中输出已确认、尝试和拒绝的特征。

让我们使用 Boruta 包来选择完成学士学位分类模型的特征（如果你还没有安装 Boruta 包，可以使用`pip`安装）：

1.  我们首先加载必要的库：

    ```py
    import pandas as pd
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    from sklearn.metrics import accuracy_score
    ```

1.  我们再次加载 NLS 教育成就数据并创建训练和测试 DataFrame：

    ```py
    nls97compba = pd.read_csv("data/nls97compba.csv")
    feature_cols = ['satverbal','satmath','gpascience',
      'gpaenglish','gpamath','gpaoverall','gender',
      'motherhighgrade','fatherhighgrade','parentincome']
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97compba[feature_cols],\
      nls97compba[['completedba']], test_size=0.3, random_state=0)
    ```

1.  接下来，我们对训练和测试数据进行编码和缩放：

    ```py
    ohe = OneHotEncoder(drop_last=True, variables=['gender'])
    ohe.fit(X_train)
    X_train_enc, X_test_enc = \
      ohe.transform(X_train), ohe.transform(X_test)
    scaler = StandardScaler()
    standcols = X_train_enc.iloc[:,:-1].columns
    scaler.fit(X_train_enc[standcols])
    X_train_enc = \
      pd.DataFrame(scaler.transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Female']])
    X_test_enc = \
      pd.DataFrame(scaler.transform(X_test_enc[standcols]),
      columns=standcols, index=X_test_enc.index).\
      join(X_test_enc[['gender_Female']])
    ```

1.  我们以与运行 RFE 特征选择相同的方式运行 Boruta 特征选择。我们再次使用随机森林作为基线方法。我们实例化一个随机森林分类器并将其传递给 Boruta 的特征选择器。然后我们拟合模型，该模型在`100`次迭代后停止，识别出提供信息的`9`个特征：

    ```py
    rfc = RandomForestClassifier(n_estimators=100, 
      max_depth=2, n_jobs=-1, random_state=0)
    borsel = BorutaPy(rfc, random_state=0, verbose=2)
    borsel.fit(X_train_enc.values, y_train.values.ravel())
    BorutaPy finished running.
    Iteration:            100 / 100
    Confirmed:            9
    Tentative:            1
    Rejected:             0
    selcols = X_train_enc.columns[borsel.support_]
    selcols
    Index(['satverbal', 'satmath', 'gpascience', 'gpaenglish', 'gpamath', 'gpaoverall', 'motherhighgrade', 'fatherhighgrade', 'parentincome', 'gender_Female'], dtype='object')
    ```

1.  我们可以使用`ranking_`属性来查看特征的排名：

    ```py
    pd.DataFrame({'ranking': borsel.ranking_,
      'feature': X_train_enc.columns},
       columns=['feature','ranking']).\
       sort_values(['ranking'], ascending=True)
               feature               ranking
    0          satverbal             1
    1          satmath               1
    2          gpascience            1
    3          gpaenglish            1
    4          gpamath               1
    5          gpaoverall            1
    6          motherhighgrade       1
    7          fatherhighgrade       1
    8          parentincome          1
    9          gender_Female         2
    ```

1.  为了评估模型的准确率，我们仅使用所选特征来拟合随机森林分类器模型。然后我们可以对测试数据进行预测并计算准确率：

    ```py
    rfc.fit(borsel.transform(X_train_enc.values), y_train.values.ravel())
    y_pred = rfc.predict(borsel.transform(X_test_enc.values))
    accuracy_score(y_test, y_pred)
    0.684981684981685
    ```

Boruta 的吸引力之一在于其对每个特征选择的说服力。如果一个特征被选中，那么它很可能提供了信息，这些信息不是通过排除它的特征组合所捕获的。然而，它在计算上相当昂贵，与穷举特征选择不相上下。它可以帮助我们区分哪些特征是重要的，但可能并不总是适合那些训练速度很重要的流水线。

最后几节展示了包装特征选择方法的某些优点和缺点。在下一节中，我们将探讨嵌入式选择方法。这些方法比过滤器方法提供更多信息，但又不具备包装方法的计算成本。它们通过将特征选择嵌入到训练过程中来实现这一点。我们将使用我们迄今为止所使用的数据来探讨嵌入式方法。

# 使用正则化和其他嵌入式方法

**正则化**方法是嵌入式方法。与包装方法一样，嵌入式方法根据给定的算法评估特征。但它们的计算成本并不高。这是因为特征选择已经嵌入到算法中，所以随着模型的训练而发生。

嵌入式模型使用以下过程：

1.  训练一个模型。

1.  估计每个特征对模型预测的重要性。

1.  移除重要性低的特征。

正则化通过向任何模型添加惩罚来约束参数来实现这一点。**L1 正则化**，也称为**lasso 正则化**，将回归模型中的某些系数缩小到 0，从而有效地消除了这些特征。

## 使用 L1 正则化

1.  我们将使用 L1 正则化和逻辑回归来选择学士学位达成模型的特征：我们需要首先导入所需的库，包括我们将首次使用的模块，`scikit-learn`中的`SelectFromModel`：

    ```py
    import pandas as pd
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import accuracy_score
    ```

1.  接下来，我们加载关于教育成就的 NLS 数据：

    ```py
    nls97compba = pd.read_csv("data/nls97compba.csv")
    feature_cols = ['satverbal','satmath','gpascience',
      'gpaenglish','gpamath','gpaoverall','gender',
      'motherhighgrade','fatherhighgrade','parentincome']
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97compba[feature_cols],\
      nls97compba[['completedba']], test_size=0.3, 
      random_state=0)
    ```

1.  然后，我们对训练数据和测试数据进行编码和缩放：

    ```py
    ohe = OneHotEncoder(drop_last=True, 
                        variables=['gender'])
    ohe.fit(X_train)
    X_train_enc, X_test_enc = \
      ohe.transform(X_train), ohe.transform(X_test)
    scaler = StandardScaler()
    standcols = X_train_enc.iloc[:,:-1].columns
    scaler.fit(X_train_enc[standcols])
    X_train_enc = \
      pd.DataFrame(scaler.transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Female']])
    X_test_enc = \
      pd.DataFrame(scaler.transform(X_test_enc[standcols]),
      columns=standcols, index=X_test_enc.index).\
      join(X_test_enc[['gender_Female']])
    ```

1.  现在，我们准备根据逻辑回归和 L1 惩罚进行特征选择：

    ```py
    lr = LogisticRegression(C=1, penalty="l1", 
                            solver='liblinear')
    regsel = SelectFromModel(lr, max_features=5)
    regsel.fit(X_train_enc, y_train.values.ravel())
    selcols = X_train_enc.columns[regsel.get_support()]
    selcols
    Index(['satmath', 'gpascience', 'gpaoverall', 
    'fatherhighgrade', 'gender_Female'], dtype='object')
    ```

1.  让我们来评估模型的准确性。我们得到了一个准确率分数为`0.68`：

    ```py
    lr.fit(regsel.transform(X_train_enc), 
           y_train.values.ravel())
    y_pred = lr.predict(regsel.transform(X_test_enc))
    accuracy_score(y_test, y_pred)
    0.684981684981685
    ```

这给我们带来了与学士学位完成的前向特征选择相当相似的结果。在那个例子中，我们使用随机森林分类器作为包装方法。

在这种情况下，Lasso 正则化是特征选择的一个好选择，尤其是当性能是一个关键关注点时。然而，它确实假设特征与目标之间存在线性关系，这可能并不合适。幸运的是，有一些嵌入式特征选择方法不做出这种假设。对于嵌入式模型来说，逻辑回归的一个好替代品是随机森林分类器。我们将使用相同的数据尝试这种方法。

## 使用随机森林分类器

在本节中，我们将使用随机森林分类器：

1.  我们可以使用`SelectFromModel`来使用随机森林分类器而不是逻辑回归：

    ```py
    rfc = RandomForestClassifier(n_estimators=100, 
      max_depth=2, n_jobs=-1, random_state=0)
    rfcsel = SelectFromModel(rfc, max_features=5)
    rfcsel.fit(X_train_enc, y_train.values.ravel())
    selcols = X_train_enc.columns[rfcsel.get_support()]
    selcols
    Index(['satverbal', 'gpascience', 'gpaenglish', 
      'gpaoverall'], dtype='object')
    ```

这实际上选择与 lasso 回归非常不同的特征。`satmath`、`fatherhighgrade`和`gender_Female`不再被选中，而`satverbal`和`gpaenglish`被选中。这很可能部分是由于线性假设的放宽。

1.  让我们评估随机森林分类器模型的准确性。我们得到了**0.67**的准确率。这几乎与我们在 lasso 回归中得到的分数相同：

    ```py
    rfc.fit(rfcsel.transform(X_train_enc), 
            y_train.values.ravel())
    y_pred = rfc.predict(rfcsel.transform(X_test_enc))
    accuracy_score(y_test, y_pred)
    0.673992673992674
    ```

嵌入式方法通常比包装方法 CPU-/GPU 密集度低，但仍然可以产生良好的结果。在本节的学士学位完成模型中，我们得到了与基于穷举特征选择模型相同的准确率。

我们之前讨论的每种方法都有重要的应用场景，正如我们所讨论的。然而，我们还没有真正讨论一个非常具有挑战性的特征选择问题。如果你简单地有太多的特征，其中许多特征在你的模型中独立地解释了某些重要内容，你会怎么做？在这里，“太多”意味着有如此多的特征，以至于模型无法高效地运行，无论是训练还是预测目标值。我们如何在不牺牲模型部分预测能力的情况下减少特征集？在这种情况下，**主成分分析（PCA**）可能是一个好的方法。我们将在下一节中讨论 PCA。

# 使用主成分分析

PCA 是一种与之前讨论的任何方法都截然不同的特征选择方法。PCA 允许我们用有限数量的组件替换现有的特征集，每个组件都解释了重要数量的方差。它是通过找到一个捕获最大方差量的组件，然后是一个捕获剩余最大方差量的第二个组件，然后是一个第三个组件，以此类推来做到这一点的。这种方法的一个关键优势是，这些被称为**主成分**的组件是不相关的。我们在*第十五章*，*主成分分析*中详细讨论 PCA。

虽然我在这里将主成分分析（PCA）视为一种特征选择方法，但可能更合适将其视为降维工具。当我们需要限制维度数量而又不希望牺牲太多解释力时，我们使用它来进行特征选择。

让我们再次使用 NLS 数据，并使用 PCA 为学士学位完成模型选择特征：

1.  我们首先加载必要的库。在本章中，我们还没有使用过的模块是`scikit-learn`的`PCA`：

    ```py
    import pandas as pd
    from feature_engine.encoding import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    ```

1.  接下来，我们再次创建训练和测试 DataFrame：

    ```py
    nls97compba = pd.read_csv("data/nls97compba.csv")
    feature_cols = ['satverbal','satmath','gpascience',
      'gpaenglish','gpamath','gpaoverall','gender',
      'motherhighgrade', 'fatherhighgrade','parentincome']
    X_train, X_test, y_train, y_test =  \
      train_test_split(nls97compba[feature_cols],\
      nls97compba[['completedba']], test_size=0.3,
      random_state=0)
    ```

1.  我们需要对数据进行缩放和编码。在 PCA 中，缩放尤其重要：

    ```py
    ohe = OneHotEncoder(drop_last=True, 
                        variables=['gender'])
    ohe.fit(X_train)
    X_train_enc, X_test_enc = \
      ohe.transform(X_train), ohe.transform(X_test)
    scaler = StandardScaler()
    standcols = X_train_enc.iloc[:,:-1].columns
    scaler.fit(X_train_enc[standcols])
    X_train_enc = \
      pd.DataFrame(scaler.transform(X_train_enc[standcols]),
      columns=standcols, index=X_train_enc.index).\
      join(X_train_enc[['gender_Female']])
    X_test_enc = \
      pd.DataFrame(scaler.transform(X_test_enc[standcols]),
      columns=standcols, index=X_test_enc.index).\
      join(X_test_enc[['gender_Female']])
    ```

1.  现在，我们实例化一个`PCA`对象并拟合模型：

    ```py
    pca = PCA(n_components=5)
    pca.fit(X_train_enc)
    ```

1.  `PCA`对象的`components_`属性返回了所有 10 个特征在每个 5 个成分上的得分。对第一个成分贡献最大的特征是得分绝对值最高的那些，在这种情况下，是`gpaoverall`、`gpaenglish`和`gpascience`。对于第二个成分，最重要的特征是`motherhighgrade`、`fatherhighgrade`和`parentincome`。`satverbal`和`satmath`驱动第三个成分。

在以下输出中，列**0**到**4**是五个主成分：

```py
pd.DataFrame(pca.components_,
  columns=X_train_enc.columns).T
                   0       1      2       3       4
satverbal         -0.34   -0.16  -0.61   -0.02   -0.19
satmath           -0.37   -0.13  -0.56    0.10    0.11
gpascience        -0.40    0.21   0.18    0.03    0.02
gpaenglish        -0.40    0.22   0.18    0.08   -0.19
gpamath           -0.38    0.24   0.12    0.08    0.23
gpaoverall        -0.43    0.25   0.23   -0.04   -0.03
motherhighgrade   -0.19   -0.51   0.24   -0.43   -0.59
fatherhighgrade   -0.20   -0.51   0.18   -0.35    0.70
parentincome      -0.16   -0.46   0.28    0.82   -0.08
gender_Female     -0.02    0.08   0.12   -0.04   -0.11
```

另一种理解这些得分的方式是，它们表明每个特征对成分的贡献程度。（实际上，如果对每个成分，你将 10 个得分平方然后求和，你会得到一个总和为 1。）

1.  让我们也检查每个成分解释了特征中多少方差。第一个成分单独解释了 46%的方差，第二个成分额外解释了 19%。我们可以使用 NumPy 的`cumsum`方法来查看五个成分累积解释了多少特征方差。我们可以用 5 个成分解释 10 个特征中的 87%的方差：

    ```py
    pca.explained_variance_ratio_
    array([0.46073387, 0.19036089, 0.09295703, 0.07163009, 0.05328056])
    np.cumsum(pca.explained_variance_ratio_)
    array([0.46073387, 0.65109476, 0.74405179, 0.81568188, 0.86896244])
    ```

1.  让我们根据这五个主成分来转换测试数据中的特征。这返回了一个只包含五个主成分的 NumPy 数组。我们查看前几行。我们还需要转换测试 DataFrame：

    ```py
    X_train_pca = pca.transform(X_train_enc)
    X_train_pca.shape
    (634, 5)
    np.round(X_train_pca[0:6],2)
    array([[ 2.79, -0.34,  0.41,  1.42, -0.11],
           [-1.29,  0.79,  1.79, -0.49, -0.01],
           [-1.04, -0.72, -0.62, -0.91,  0.27],
           [-0.22, -0.8 , -0.83, -0.75,  0.59],
           [ 0.11, -0.56,  1.4 ,  0.2 , -0.71],
           [ 0.93,  0.42, -0.68, -0.45, -0.89]])
    X_test_pca = pca.transform(X_test_enc)
    ```

现在，我们可以使用这些主成分来拟合一个关于学士学位完成情况的模型。让我们运行一个随机森林分类。

1.  我们首先创建一个随机森林分类器对象。然后，我们将带有主成分和目标值的训练数据传递给其`fit`方法。我们将带有成分的测试数据传递给分类器的`predict`方法，然后得到一个准确度分数：

    ```py
    rfc = RandomForestClassifier(n_estimators=100, 
      max_depth=2, n_jobs=-1, random_state=0)
    rfc.fit(X_train_pca, y_train.values.ravel())
    y_pred = rfc.predict(X_test_pca)
    accuracy_score(y_test, y_pred)
    0.7032967032967034
    ```

当特征选择挑战是我们有高度相关的特征，并且我们希望在不过度减少解释方差的情况下减少维度数量时，PCA 等降维技术可以是一个好的选择。在这个例子中，高中 GPA 特征一起移动，父母的教育水平和收入水平以及 SAT 特征也是如此。它们成为了我们前三个成分的关键特征。（可以认为我们的模型只需要那三个成分，因为它们共同解释了特征变异的 74%。）

根据你的数据和建模目标，PCA（主成分分析）有几种修改方式可能是有用的。这包括处理异常值和正则化的策略。通过使用核函数，PCA 还可以扩展到那些成分不能线性分离的情况。我们将在*第十五章**，*《主成分分析》*中详细讨论 PCA。

让我们总结一下本章所学的内容。

# 摘要

在本章中，我们讨论了从过滤方法到包装方法再到嵌入式方法的一系列特征选择方法。我们还看到了它们如何与分类和连续目标一起工作。对于包装和嵌入式方法，我们考虑了它们如何与不同的算法一起工作。

过滤方法运行和解释都非常简单，且对系统资源的影响较小。然而，它们在评估每个特征时并没有考虑其他特征。而且，它们也没有告诉我们这种评估可能会因所使用的算法而有所不同。包装方法没有这些限制，但计算成本较高。嵌入式方法通常是一个很好的折衷方案，它们根据多元关系和给定的算法选择特征，而不像包装方法那样对系统资源造成过多负担。我们还探讨了如何通过降维方法 PCA 来改进我们的特征选择。

你可能也注意到了，我在本章中稍微提到了一点模型验证。我们将在下一章更详细地介绍模型验证。
