# *第十六章*：K-Means 和 DBSCAN 聚类

数据聚类使我们能够将未标记的数据组织成具有更多共同点的观察组，这些共同点比组外的观察点更多。聚类有许多令人惊讶的应用，无论是作为机器学习管道的最终模型，还是作为其他模型的输入。这包括市场研究、图像处理和文档分类。我们有时也使用聚类来改进探索性数据分析或创建更有意义的可视化。

K-means 和**基于密度的应用噪声聚类**（**DBSCAN**），就像**主成分分析**（**PCA**）一样，是无监督学习算法。没有标签可以用作预测的基础。算法的目的是根据特征识别出相互关联的实例。彼此靠近且与其他实例距离较远的实例可以被认为是处于一个簇中。有几种方法可以衡量邻近程度。**基于划分的聚类**，如 k-means，和**基于密度的聚类**，如 DBSCAN，是两种更受欢迎的方法。我们将在本章中探讨这些方法。

具体来说，我们将讨论以下主题：

+   k-means 和 DBSCAN 聚类的关键概念

+   实现 k-means 聚类

+   实现 DBSCAN 聚类

# 技术要求

在本章中，我们将主要使用 pandas、NumPy 和 scikit-learn 库。

# k-means 和 DBSCAN 聚类的关键概念

在 k-means 聚类中，我们识别*k*个簇，每个簇都有一个中心，或**质心**。质心是使它与簇中其他数据点的总平方距离最小的点。

一个使用虚构数据的例子应该能有所帮助。*图 16.1*中的数据点似乎在三个簇中。（通常并不那么容易可视化簇的数量，*k*。）

![图 16.1 – 具有三个可识别簇的数据点](img/B17978_16_001.jpg)

图 16.1 – 具有三个可识别簇的数据点

我们执行以下步骤来构建簇：

1.  将一个随机点分配为每个簇的中心。

1.  计算每个点到每个簇中心的距离。

1.  根据数据点到中心点的邻近程度将数据点分配到簇中。这三个步骤在*图 16.2*中进行了说明。带有**X**的点是被随机选择的簇中心（将*k*设置为 3）。比其他簇中心点更接近簇中心点的数据点被分配到该簇。

![图 16.2 – 随机分配为簇中心的点](img/B17978_16_002.jpg)

图 16.2 – 随机分配为簇中心的点

1.  为新簇计算一个新的中心点。这如图 16.3 所示。

![图 16.3 – 新计算的簇中心](img/B17978_16_003.jpg)

图 16.3 – 新的聚类中心计算

1.  重复步骤 2 到 4，直到中心的变化不大。

K-means 聚类是一种非常流行的聚类算法，原因有几个。它相当直观，通常也相当快。然而，它确实有一些缺点。它将每个数据点都处理为聚类的一部分，因此聚类可能会被极端值拉扯。它还假设聚类将具有球形形状。

无监督模型的评估不如监督模型清晰，因为我们没有目标来比较我们的预测。聚类模型的一个相当常见的指标是**轮廓分数**。轮廓分数是所有实例的平均轮廓系数。**轮廓系数**如下：

![](img/B17978_16_0011.jpg)

这里，![](img/B17978_16_002.png)是第 i 个实例到下一个最近簇的所有实例的平均距离，而![](img/B17978_16_003.png)是到分配簇的实例的平均距离。这个系数的范围从-1 到 1，分数接近 1 意味着实例很好地位于分配的簇内。

评估我们的聚类的一个另一个指标是**惯性分数**。这是每个实例与其质心之间平方距离的总和。随着我们增加聚类数量，这个距离会减小，但最终增加聚类数量会带来边际收益的递减。通常使用**肘图**来可视化 k 值与惯性分数的变化。这个图被称为肘图，因为随着 k 的增加，斜率会接近 0，接近到它类似于一个肘部。这如图*图 16.4*所示。在这种情况下，我们会选择一个接近肘部的 k 值。

![图 16.4 – 惯性和 k 的肘图](img/B17978_16_004.jpg)

图 16.4 – 惯性和 k 的肘图

评估聚类模型时，经常使用的另一个指标是**Rand 指数**。Rand 指数告诉我们两个聚类如何频繁地将相同的簇分配给实例。Rand 指数的值将在 0 到 1 之间。我们通常使用调整后的 Rand 指数，它纠正了相似度计算中的偶然性。调整后的 Rand 指数的值有时可能是负数。

**DBSCAN**采用了一种不同的聚类方法。对于每个实例，它计算该实例指定距离内的实例数量。所有在ɛ距离内的实例都被认为是该实例的ɛ-邻域。当ɛ-邻域中的实例数量等于或超过我们指定的最小样本值时，该实例被认为是核心实例，ɛ-邻域被认为是聚类。任何与另一个实例距离超过ɛ的实例被认为是噪声。这如图*图 16.5*所示。

![图 16.5 – 最小样本数=五的 DBSCAN 聚类](img/B17978_16_005.jpg)

图 16.5 – 最小样本数=五的 DBSCAN 聚类

这种基于密度的方法有几个优点。簇不需要是球形的，它们可以采取任何形状。虽然我们不需要猜测簇的数量，但我们需要提供一个ɛ的值。异常值只是被解释为噪声，因此不会影响簇。（这一点暗示了 DBSCAN 的另一个有用应用：识别异常。）

我们将在本章后面使用 DBSCAN 进行聚类。首先，我们将检查如何使用 k-means 进行聚类，包括如何选择一个好的 k 值。

# 实现 k-means 聚类

我们可以使用与我们在前几章中开发的监督学习模型相同的某些数据来使用 k-means。区别在于我们不再有一个预测的目标。相反，我们感兴趣的是某些实例是如何聚集在一起的。想想典型的中学午餐休息时间人们如何分组，你就能得到一个大致的概念。

我们还需要做很多与监督学习模型相同的预处理工作。我们将在本节开始这部分。我们将处理关于女性和男性之间的收入差距、劳动力参与率、教育成就、青少年出生频率以及女性在最高级别参与政治的数据。

注意

收入差距数据集由联合国开发计划署在[`www.kaggle.com/datasets/undp/human-development`](https://www.kaggle.com/datasets/undp/human-development)上提供供公众使用。每个国家都有一个记录，包含 2015 年按性别汇总的就业、收入和教育数据。

让我们构建一个 k-means 聚类模型：

1.  我们加载了熟悉的库。我们还加载了`KMeans`和`silhouette_score`模块。回想一下，轮廓分数通常用于评估我们的模型在聚类方面做得有多好。我们还加载了`rand_score`，这将允许我们计算不同聚类之间的相似性指数：

    ```py
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.cluster import rand_score
    from sklearn.impute import KNNImputer
    import seaborn as sns
    import matplotlib.pyplot as plt
    ```

1.  接下来，我们加载收入差距数据：

    ```py
    un_income_gap = pd.read_csv("data/un_income_gap.csv")
    un_income_gap.set_index('country', inplace=True)
    un_income_gap['incomeratio'] = \
      un_income_gap.femaleincomepercapita / \
        un_income_gap.maleincomepercapita
    un_income_gap['educratio'] = \
      un_income_gap.femaleyearseducation / \
         un_income_gap.maleyearseducation
    un_income_gap['laborforcepartratio'] = \
      un_income_gap.femalelaborforceparticipation / \
         un_income_gap.malelaborforceparticipation
    un_income_gap['humandevratio'] = \
      un_income_gap.femalehumandevelopment / \
         un_income_gap.malehumandevelopment
    ```

1.  让我们看看一些描述性统计：

    ```py
    num_cols = ['educratio','laborforcepartratio','humandevratio',
      'genderinequality','maternalmortality','incomeratio',
      'adolescentbirthrate', 'femaleperparliament',
      'incomepercapita']
    gap = un_income_gap[num_cols]
    gap.agg(['count','min','median','max']).T
                         count   min    median  max
    educratio            170.00  0.24   0.93    1.35
    laborforcepartratio  177.00  0.19   0.75    1.04
    humandevratio        161.00  0.60   0.95    1.03
    genderinequality     155.00  0.02   0.39    0.74
    maternalmortality    178.00  1.00   64.00   1,100.00
    incomeratio          177.00  0.16   0.60    0.93
    adolescentbirthrate  183.00  0.60   40.90   204.80
    femaleperparliament  185.00  0.00   19.60   57.50
    incomepercapita      188.00  581.00 10,667.00  23,124.00
    ```

1.  我们还应该查看一些相关性。教育比率（女性教育水平与男性教育水平的比率）和人类发展比率高度相关，性别不平等和青少年出生率也是如此，以及收入比率和劳动力参与率：

    ```py
    corrmatrix = gap.corr(method="pearson")
    sns.heatmap(corrmatrix, 
      xticklabels=corrmatrix.columns,
      yticklabels=corrmatrix.columns, cmap="coolwarm")
    plt.title('Heat Map of Correlation Matrix')
    plt.tight_layout()
    plt.show()
    ```

这会产生以下图表：

![图 16.6 – 相关矩阵的热图](img/B17978_16_006.jpg)

图 16.6 – 相关矩阵的热图

1.  在运行我们的模型之前，我们需要对数据进行缩放。我们还使用**KNN 插补**来处理缺失值：

    ```py
    pipe1 = make_pipeline(MinMaxScaler(), KNNImputer(n_neighbors=5))
    gap_enc = pd.DataFrame(pipe1.fit_transform(gap),
      columns=num_cols, index=gap.index)
    ```

1.  现在，我们已经准备好运行 k-means 聚类。我们为簇的数量指定一个值。

在拟合模型后，我们可以生成一个影子分数。我们的影子分数并不高。这表明我们的聚类之间并没有很远。稍后，我们将看看是否可以通过更多或更少的聚类来获得更好的分数：

```py
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(gap_enc)
KMeans(n_clusters=3, random_state=0)
silhouette_score(gap_enc, kmeans.labels_)
0.3311928353317411
```

1.  让我们更仔细地观察聚类。我们可以使用`labels_`属性来获取聚类：

    ```py
    gap_enc['cluster'] = kmeans.labels_
    gap_enc.cluster.value_counts().sort_index()
    0     40
    1    100
    2     48
    Name: cluster, dtype: int64
    ```

1.  我们本可以使用`fit_predict`方法来获取聚类，如下所示：

    ```py
    pred = pd.Series(kmeans.fit_predict(gap_enc))
    pred.value_counts().sort_index()
    0     40
    1    100
    2     48
    dtype: int64
    ```

1.  有助于检查聚类在特征值方面的差异。聚类 0 的国家在孕产妇死亡率和青少年出生率方面比其他聚类的国家要高得多。聚类 1 的国家孕产妇死亡率非常低，人均收入很高。聚类 2 的国家劳动力参与率（女性劳动力参与率与男性劳动力参与率的比率）和收入比率非常低。回想一下，我们已经对数据进行归一化处理：

    ```py
    gap_cluster = gap_enc.join(cluster)
    gap_cluster[['cluster'] + num_cols].groupby(['cluster']).mean().T
    cluster              0            1           2
    educratio            0.36         0.66        0.54
    laborforcepartratio  0.80         0.67        0.32
    humandevratio        0.62         0.87        0.68
    genderinequality     0.79         0.32        0.62
    maternalmortality    0.44         0.04        0.11
    incomeratio          0.71         0.60        0.29
    adolescentbirthrate  0.51         0.15        0.20
    femaleperparliament  0.33         0.43        0.24
    incomepercapita      0.02         0.19        0.12
    ```

1.  我们可以使用`cluster_centers_`属性来获取每个聚类的中心。由于我们使用了九个特征进行聚类，因此有九个值代表三个聚类的中心：

    ```py
    centers = kmeans.cluster_centers_
    centers.shape
    (3, 9)
    np.set_printoptions(precision=2)
    centers
    array([[0.36, 0.8 , 0.62, 0.79, 0.44, 0.71, 0.51, 0.33, 0.02],
           [0.66, 0.67, 0.87, 0.32, 0.04, 0.6 , 0.15, 0.43, 0.19],
           [0.54, 0.32, 0.68, 0.62, 0.11, 0.29, 0.2 , 0.24, 0.12]])
    ```

1.  我们通过一些特征绘制聚类，以及中心。我们将该聚类的数字放置在该聚类的质心处：

    ```py
    fig = plt.figure()
    plt.suptitle("Cluster for each Country")
    ax = plt.axes(projection='3d')
    ax.set_xlabel("Maternal Mortality")
    ax.set_ylabel("Adolescent Birth Rate")
    ax.set_zlabel("Income Ratio")
    ax.scatter3D(gap_cluster.maternalmortality,
      gap_cluster.adolescentbirthrate,
      gap_cluster.incomeratio, c=gap_cluster.cluster, cmap="brg")
    for j in range(3):
      ax.text(centers2[j, num_cols.index('maternalmortality')],
      centers2[j, num_cols.index('adolescentbirthrate')],
      centers2[j, num_cols.index('incomeratio')],
      c='black', s=j, fontsize=20, fontweight=800)
    plt.tight_layout()
    plt.show()
    ```

这产生了以下图形：

![图 16.7 – 三个聚类的 3D 散点图](img/B17978_16_007.jpg)

图 16.7 – 三个聚类的 3D 散点图

我们可以看到，聚类 0 的国家孕产妇死亡率和青少年出生率较高。聚类 0 国家的收入比率较低。

1.  到目前为止，我们假设用于我们模型的最佳聚类数量是三个。让我们构建一个五聚类模型，看看那些结果如何。

影子分数从三聚类模型中下降。这可能表明至少有一些聚类非常接近：

```py
gap_enc = gap_enc[num_cols]
kmeans2 = KMeans(n_clusters=5, random_state=0)
kmeans2.fit(gap_enc)
silhouette_score(gap_enc, kmeans2.labels_)
0.2871811434351394
gap_enc['cluster2'] = kmeans2.labels_
gap_enc.cluster2.value_counts().sort_index()
0    21
1    40
2    48
3    16
4    63
Name: cluster2, dtype: int64
```

1.  让我们绘制新的聚类，以更好地了解它们的位置：

    ```py
    fig = plt.figure()
    plt.suptitle("Cluster for each Country")
    ax = plt.axes(projection='3d')
    ax.set_xlabel("Maternal Mortality")
    ax.set_ylabel("Adolescent Birth Rate")
    ax.set_zlabel("Income Ratio")
    ax.scatter3D(gap_cluster.maternalmortality,
      gap_cluster.adolescentbirthrate,
      gap_cluster.incomeratio, c=gap_cluster.cluster2, 
      cmap="brg")
    for j in range(5):
      ax.text(centers2[j, num_cols.index('maternalmortality')],
      centers2[j, num_cols.index('adolescentbirthrate')],
      centers2[j, num_cols.index('incomeratio')],
      c='black', s=j, fontsize=20, fontweight=800)
    plt.tight_layout()
    plt.show()
    ```

这产生了以下图形：

![图 16.8 – 五个聚类的 3D 散点图](img/B17978_16_008.jpg)

图 16.8 – 五个聚类的 3D 散点图

1.  我们可以使用一个称为 Rand 指数的统计量来衡量聚类之间的相似性：

    ```py
    rand_score(kmeans.labels_, kmeans2.labels_)
    0.7439412902491751
    ```

1.  我们尝试了三聚类和五聚类模型，但那些是否是好的选择？让我们查看一系列*k*值的分数：

    ```py
    gap_enc = gap_enc[num_cols]
    iner_scores = []
    sil_scores = []
    for j in range(2,20):
      kmeans=KMeans(n_clusters=j, random_state=0)
      kmeans.fit(gap_enc)
      iner_scores.append(kmeans.inertia_)
      sil_scores.append(silhouette_score(gap_enc,
        kmeans.labels_))
    ```

1.  让我们绘制惯性分数与肘图：

    ```py
    plt.title('Elbow Plot')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.plot(range(2,20),iner_scores)
    ```

这产生了以下图形：

![图 16.9 – 惯性分数的肘图](img/B17978_16_009.jpg)

图 16.9 – 惯性分数的肘图

1.  我们还创建了一个影子分数的图形：

    ```py
    plt.title('Silhouette Score')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.plot(range(2,20),sil_scores)
    ```

这产生了以下图形：

![图 16.10 – 影子分数图](img/B17978_16_010.jpg)

图 16.10 – 影子分数图

肘图表明，大约 6 或 7 的*k*值会是最优的。当*k*值超过这个值时，惯性开始减少。轮廓分数图表明*k*值更小，因为在那之后轮廓分数急剧下降。

K-means 聚类帮助我们理解了关于收入、教育和就业方面的男女差距数据。我们现在可以看到某些特征是如何相互关联的，这是之前简单的相关性分析所没有揭示的。然而，这很大程度上假设我们的聚类具有球形形状，我们不得不做一些工作来确认我们的*k*值是最好的。在 DBSCAN 聚类中，我们不会遇到任何相同的问题，所以我们将尝试在下一节中这样做。

# 实现 DBSCAN 聚类

DBSCAN 是一种非常灵活的聚类方法。我们只需要指定一个值用于ɛ，也称为**eps**。正如我们之前讨论的，ɛ值决定了实例周围的ɛ-邻域的大小。最小样本超参数表示围绕一个实例需要多少个实例才能将其视为核心实例。

注意

我们使用 DBSCAN 聚类我们在上一节中使用的相同收入差距数据。

让我们构建一个 DBSCAN 聚类模型：

1.  我们首先加载熟悉的库，以及`DBSCAN`模块：

    ```py
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.cluster import DBSCAN
    from sklearn.impute import KNNImputer
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import os
    import sys
    sys.path.append(os.getcwd() + "/helperfunctions")
    ```

1.  我们导入代码来加载和预处理我们在上一节中使用的工资收入数据。由于该代码没有变化，这里不需要重复：

    ```py
    import incomegap as ig
    gap = ig.gap
    num_cols = ig.num_cols
    ```

1.  我们现在准备好预处理数据并拟合一个 DBSCAN 模型。我们在这里选择ɛ值为 0.35 主要是通过试错。我们也可以遍历一系列ɛ值并比较轮廓分数：

    ```py
    pipe1 = make_pipeline(MinMaxScaler(),
      KNNImputer(n_neighbors=5))
    gap_enc = pd.DataFrame(pipe1.fit_transform(gap),
      columns=num_cols, index=gap.index)
    dbscan = DBSCAN(eps=0.35, min_samples=5)
    dbscan.fit(gap_enc)
    silhouette_score(gap_enc, dbscan.labels_)
    0.31106297603736455
    ```

1.  我们可以使用`labels_`属性来查看聚类。我们有 17 个噪声实例，那些聚类为-1 的实例。其余的观测值在一个或两个聚类中：

    ```py
    gap_enc['cluster'] = dbscan.labels_
    gap_enc.cluster.value_counts().sort_index()
    -1     17
     0    139
     1     32
    Name: cluster, dtype: int64
    gap_enc = \
     gap_enc.loc[gap_enc.cluster!=-1]
    ```

1.  让我们更仔细地看看哪些特征与每个聚类相关联。聚类 1 的国家在`maternalmortality`、`adolescentbirthrate`和`genderinequality`方面与聚类 0 的国家非常不同。这些特征在 k-means 聚类中也很重要，但 DBSCAN 中聚类数量更少，绝大多数实例都落入一个聚类中：

    ```py
    gap_enc[['cluster'] + num_cols].\
      groupby(['cluster']).mean().T
    cluster                     0            1
    educratio                   0.63         0.35
    laborforcepartratio         0.57         0.82
    humandevratio               0.82         0.62
    genderinequality            0.40         0.79
    maternalmortality           0.05         0.45
    incomeratio                 0.51         0.71
    adolescentbirthrate         0.16         0.50
    femaleperparliament         0.36         0.30
    incomepercapita             0.16         0.02
    ```

1.  让我们可视化聚类：

    ```py
    fig = plt.figure()
    plt.suptitle("Cluster for each Country")
    ax = plt.axes(projection='3d')
    ax.set_xlabel("Maternal Mortality")
    ax.set_ylabel("Adolescent Birth Rate")
    ax.set_zlabel("Gender Inequality")
    ax.scatter3D(gap_cluster.maternalmortality,
      gap_cluster.adolescentbirthrate,
      gap_cluster.genderinequality, c=gap_cluster.cluster, 
      cmap="brg")
    plt.tight_layout()
    plt.show()
    ```

这产生了以下图表：

![图 16.11 – 每个国家的聚类 3D 散点图](img/B17978_16_011.jpg)

图 16.11 – 每个国家的聚类 3D 散点图

DBSCAN 是聚类的一个优秀工具，尤其是当我们的数据特征意味着 k-means 聚类不是一个好选择时；例如，当聚类不是球形时。它还有的优点是不受异常值的影响。

# 摘要

我们有时需要将具有相似特征的实例组织成组。即使没有预测目标，这也有用。我们可以使用为可视化创建的聚类，就像我们在本章中所做的那样。由于聚类易于解释，我们可以利用它们来假设为什么某些特征会一起移动。我们还可以在后续分析中使用聚类结果。

本章探讨了两种流行的聚类技术：k-means 和 DBSCAN。这两种技术都直观、高效，并且能够可靠地处理聚类。
