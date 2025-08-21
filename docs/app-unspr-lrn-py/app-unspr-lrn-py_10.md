# 第十一章：*附录*

## 关于

本节内容旨在帮助学生完成书中列出的活动。它包括学生需要执行的详细步骤，以完成并实现书中的目标。

## 第一章：聚类介绍

### 活动 1：实现 k-means 聚类

解决方案：

1.  使用 pandas 加载 Iris 数据文件，pandas 是一个通过使用 DataFrame 使数据处理变得更加容易的包：

    ```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist
    iris = pd.read_csv('iris_data.csv', header=None)
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'species']
    ```

1.  将 `X` 特征和提供的 `y` 物种标签分开，因为我们希望将其视为无监督学习问题：

    ```py
    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = iris['species']
    ```

1.  了解我们的特征是什么样的：

    ```py
    X.head()
    ```

    输出结果如下：

    ![图 1.22：数据的前五行    ](img/C12626_01_22.jpg)

    ###### 图 1.22：数据的前五行

1.  将我们之前创建的 `k_means` 函数拿出来参考：

    ```py
    def k_means(X, K):
    #Keep track of history so you can see k-means in action
        centroids_history = []
        labels_history = []
        rand_index = np.random.choice(X.shape[0], K)  
        centroids = X[rand_index]
        centroids_history.append(centroids)
        while True:
    # Euclidean distances are calculated for each point relative to centroids, #and then np.argmin returns
    # the index location of the minimal distance - which cluster a point    is #assigned to
            labels = np.argmin(cdist(X, centroids), axis=1)
            labels_history.append(labels)
    #Take mean of points within clusters to find new centroids:
            new_centroids = np.array([X[labels == i].mean(axis=0)
                                    for i in range(K)])
            centroids_history.append(new_centroids)

            # If old centroids and new centroids no longer change, k-means is complete and end. Otherwise continue
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids

        return centroids, labels, centroids_history, labels_history
    ```

1.  将我们的 Iris `X` 特征 DataFrame 转换为 `NumPy` 矩阵：

    ```py
    X_mat = X.values
    ```

1.  在鸢尾矩阵上运行我们的 `k_means` 函数：

    ```py
    centroids, labels, centroids_history, labels_history = k_means(X_mat, 3)
    ```

1.  查看我们通过查看每个样本的预测物种列表得到的标签：

    ```py
    print(labels)
    ```

    输出结果如下：

    ![图 1.23：预测物种列表    ](img/C12626_01_23.jpg)

    ###### 图 1.23：预测物种列表

1.  可视化我们在数据集上实现的 k-means 方法：

    ```py
    plt.scatter(X['SepalLengthCm'], X['SepalWidthCm'])
    plt.title('Iris - Sepal Length vs Width')
    plt.show()
    ```

    输出结果如下：

    ![图 1.24：执行的 k-means 实现的图    ](img/C12626_01_24.jpg)

    ###### 图 1.24：执行的 k-means 实现的图

    如下所示可视化鸢尾物种的簇：

    ```py
    plt.scatter(X['SepalLengthCm'], X['SepalWidthCm'], c=labels, cmap='tab20b')
    plt.title('Iris - Sepal Length vs Width - Clustered')
    plt.show()
    ```

    输出结果如下：

    ![图 1.25：鸢尾物种的簇    ](img/C12626_01_25.jpg)

    ###### 图 1.25：鸢尾物种的簇

1.  使用 scikit-learn 实现计算轮廓系数（Silhouette Score）：

    ```py
    # Calculate Silhouette Score
    silhouette_score(X[['SepalLengthCm','SepalWidthCm']], labels)
    ```

    你将得到一个大约等于 0.369 的 SSI。由于我们只使用了两个特征，这是可以接受的，结合最终图中展示的聚类成员可视化。

## 第二章：层次聚类

活动 2：应用连接标准

解决方案：

1.  可视化我们在 *练习 7* 中创建的 `x` 数据集，*构建层次结构*：

    ```py
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    %matplotlib inline
    # Generate a random cluster dataset to experiment on. X = coordinate points, y = cluster labels (not needed)
    X, y = make_blobs(n_samples=1000, centers=8, n_features=2, random_state=800)
    # Visualize the data
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    ```

    输出结果如下：

    ![图 2.20：生成的簇数据集的散点图    ](img/C12626_02_20.jpg)

    ###### 图 2.20：生成的簇数据集的散点图

1.  创建一个包含所有可能连接方法超参数的列表：

    ```py
    methods = ['centroid', 'single', 'complete', 'average', 'weighted']
    ```

1.  遍历你刚才创建的列表中的每种方法，展示它们在同一数据集上的效果：

    ```py
    for method in methods:
        distances = linkage(X, method=method, metric="euclidean")
        clusters = fcluster(distances, 3, criterion="distance") 
        plt.title('linkage: ' + method)
        plt.scatter(X[:,0], X[:,1], c=clusters, cmap='tab20b')
        plt.show()
    ```

    输出结果如下：

![图 2.21：所有方法的散点图](img/C12626_02_21.jpg)

###### 图 2.21：所有方法的散点图

分析：

从前面的图中可以看出，通过简单地更改连接标准，可以显著改变聚类的效果。在这个数据集中，质心法和平均法最适合找到合理的离散簇。这一点从我们生成的八个簇的事实中可以看出，质心法和平均法是唯一显示出使用八种不同颜色表示的簇的算法。其他连接类型则表现不佳，尤其是单链接法。

### 活动 3：比较 k-means 与层次聚类

解决方案：

1.  从 scikit-learn 导入必要的包（`KMeans`、`AgglomerativeClustering` 和 `silhouette_score`），如下所示：

    ```py
    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import matplotlib.pyplot as plt
    ```

1.  将葡萄酒数据集读入 pandas DataFrame 并打印一个小样本：

    ```py
    wine_df = pd.read_csv("wine_data.csv")
    print(wine_df.head)
    ```

    输出如下：

    ![图 2.22：葡萄酒数据集的输出    ](img/C12626_02_22.jpg)

    ###### 图 2.22：葡萄酒数据集的输出

1.  可视化葡萄酒数据集以理解数据结构：

    ```py
    plt.scatter(wine_df.values[:,0], wine_df.values[:,1])
    plt.title("Wine Dataset")
    plt.xlabel("OD Reading")
    plt.ylabel("Proline")
    plt.show()
    ```

    输出如下：

    ![图 2.23：原始葡萄酒数据的聚类图    ](img/C12626_02_23.jpg)

    ###### 图 2.23：原始葡萄酒数据的聚类图

1.  在葡萄酒数据集上使用 sklearn 实现的 k-means，已知有三种葡萄酒类型：

    ```py
    km = KMeans(3)
    km_clusters = km.fit_predict(wine_df)
    ```

1.  使用 sklearn 实现的层次聚类对葡萄酒数据集进行处理：

    ```py
    ac = AgglomerativeClustering(3, linkage='average')
    ac_clusters = ac.fit_predict(wine_df)
    ```

1.  如下所示，绘制 k-means 预测的聚类：

    ```py
    plt.scatter(wine_df.values[:,0], wine_df.values[:,1], c=km_clusters)
    plt.title("Wine Clusters from Agglomerative Clustering")
    plt.xlabel("OD Reading")
    plt.ylabel("Proline")
    plt.show()
    ```

    输出如下：

    ![图 2.24：k-means 聚类的聚类图    ](img/C12626_02_24.jpg)

    ###### 图 2.24：k-means 聚类的聚类图

1.  如下所示，绘制层次聚类预测的聚类：

    ```py
    plt.scatter(wine_df.values[:,0], wine_df.values[:,1], c=ac_clusters)
    plt.title("Wine Clusters from Agglomerative Clustering")
    plt.xlabel("OD Reading")
    plt.ylabel("Proline")
    plt.show()
    ```

    输出如下：

    ![图 2.25：凝聚聚类的聚类图    ](img/C12626_02_25.jpg)

    ###### 图 2.25：凝聚聚类的聚类图

1.  比较每种聚类方法的轮廓得分：

    ```py
    print("Silhouette Scores for Wine Dataset:\n")
    print("k-means Clustering: ", silhouette_score(X[:,11:13], km_clusters))
    print("Agg Clustering: ", silhouette_score(X[:,11:13], ac_clusters))
    ```

    输出如下：

![图 2.26：葡萄酒数据集的轮廓得分](img/C12626_02_26.jpg)

###### 图 2.26：葡萄酒数据集的轮廓得分

从之前的轮廓得分可以看出，凝聚聚类在分离聚类时，平均簇内距离上略微优于 k-means 聚类。然而，并非所有版本的凝聚聚类都有这个效果。你可以尝试不同的连接类型，观察不同轮廓得分和聚类结果如何变化！

## 第三章：邻域方法与 DBSCAN

### 活动 4：从零实现 DBSCAN

解决方案：

1.  生成一个随机的聚类数据集，如下所示：

    ```py
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import numpy as np
    %matplotlib inline
    X_blob, y_blob = make_blobs(n_samples=500, centers=4, n_features=2, random_state=800)
    ```

1.  可视化生成的数据：

    ```py
    plt.scatter(X_blob[:,0], X_blob[:,1])
    plt.show()
    ```

    输出如下：

    ![图 3.14：生成数据的聚类图    ](img/C12626_03_14.jpg)

    ###### 图 3.14：生成数据的聚类图

1.  创建从零开始的函数，允许你在数据集上调用 DBSCAN：

    ```py
    def scratch_DBSCAN(x, eps, min_pts):
        """
        param x (list of vectors): your dataset to be clustered
        param eps (float): neigborhood radius threshold
        param min_pts (int): minimum number of points threshold for a nieghborhood to be a cluster
        """
         # Build a label holder that is comprised of all 0s
        labels = [0]* x.shape[0]
        # Arbitrary starting "current cluster" ID    
        C = 0

        # For each point p in x...
        # ('p' is the index of the datapoint, rather than the datapoint itself.)
        for p in range(0, x.shape[0]):

            # Only unvisited points can be evaluated as neighborhood centers
            if not (labels[p] == 0):
                continue

            # Find all of p's neighbors.
            neighbors = neighborhood_search(x, p, eps)

            # If there are not enough neighbor points, then it is classified as noise (-1).
            # Otherwise we can use this point as a neighborhood cluster
            if len(neighbors) < min_pts:
                labels[p] = -1    
            else: 
                C += 1
                neighbor_cluster(x, labels, p, neighbors, C, eps, min_pts)

        return labels
    def neighbor_cluster(x, labels, p, neighbors, C, eps, min_pts):
        # Assign the cluster label to original point
        labels[p] = C

        # Look at each neighbor of p (by index, not the points themselves) and evaluate
        i = 0
        while i < len(neighbors):    

            # Get the next point from the queue.        
            potential_neighbor_ix = neighbors[i]

            # If potential_neighbor_ix is noise from previous runs, we can assign it to current cluster
            if labels[potential_neighbor_ix] == -1:
                labels[potential_neighbor_ix] = C

            # Otherwise, if potential_neighbor_ix is unvisited, we can add it to current cluster
            elif labels[potential_neighbor_ix] == 0:
                labels[potential_neighbor_ix] = C

                # Further find neighbors of potential neighbor
                potential_neighbors_cluster = neighborhood_search(x, potential_neighbor_ix, eps)

                if len(potential_neighbors_cluster) >= min_pts:
                    neighbors = neighbors + potential_neighbors_cluster      

            # Evaluate next neighbor
            i += 1        
    def neighborhood_search(x, p, eps):
        neighbors = []

        # For each point in the dataset...
        for potential_neighbor in range(0, x.shape[0]):

            # If a nearby point falls below the neighborhood radius threshold, add to neighbors list
            if np.linalg.norm(x[p] - x[potential_neighbor]) < eps:
                neighbors.append(potential_neighbor)

        return neighbors
    ```

1.  使用你创建的 DBSCAN 实现，在生成的数据集中查找聚类。根据第五步中的性能，随意调整超参数：

    ```py
    labels = scratch_DBSCAN(X_blob, 0.6, 5)
    ```

1.  从零开始可视化你实现的 DBSCAN 聚类性能：

    ```py
    plt.scatter(X_blob[:,0], X_blob[:,1], c=labels)
    plt.title("DBSCAN from Scratch Performance")
    plt.show()
    ```

    输出如下：

![图 3.15：DBSCAN 实现的聚类图](img/C12626_03_15.jpg)

###### 图 3.15：DBSCAN 实现的聚类图

正如你可能注意到的，定制实现的运行时间相对较长。这是因为为了清晰起见，我们探索了该算法的非矢量化版本。接下来，你应该尽量使用 scikit-learn 提供的 DBSCAN 实现，因为它经过高度优化。

### 活动 5：比较 DBSCAN 与 k-means 和层次聚类

解决方案：

1.  导入必要的包：

    ```py
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import matplotlib.pyplot as plt
    %matplotlib inline
    ```

1.  从*第二章* *Hierarchical Clustering*加载葡萄酒数据集，并再次熟悉数据的外观：

    ```py
    # Load Wine data set
    wine_df = pd.read_csv("../CH2/wine_data.csv")
    # Show sample of data set
    print(wine_df.head())
    ```

    输出如下：

    ![图 3.16: 葡萄酒数据集的前五行    ](img/C12626_03_16.jpg)

    ###### 图 3.16: 葡萄酒数据集的前五行

1.  可视化数据：

    ```py
    plt.scatter(wine_df.values[:,0], wine_df.values[:,1])
    plt.title("Wine Dataset")
    plt.xlabel("OD Reading")
    plt.ylabel("Proline")
    plt.show()
    ```

    输出如下：

    ![图 3.17: 数据的图    ](img/C12626_03_17.jpg)

    ###### 图 3.17: 数据的图

1.  使用 k-means、凝聚聚类和 DBSCAN 生成聚类：

    ```py
    # Generate clusters from K-Means
    km = KMeans(3)
    km_clusters = km.fit_predict(wine_df)
    # Generate clusters using Agglomerative Hierarchical Clustering
    ac = AgglomerativeClustering(3, linkage='average')
    ac_clusters = ac.fit_predict(wine_df)
    ```

1.  评估 DSBSCAN 超参数的几个不同选项及其对轮廓分数的影响：

    ```py
    db_param_options = [[20,5],[25,5],[30,5],[25,7],[35,7],[35,3]]
    for ep,min_sample in db_param_options:
        # Generate clusters using DBSCAN
        db = DBSCAN(eps=ep, min_samples = min_sample)
        db_clusters = db.fit_predict(wine_df)
        print("Eps: ", ep, "Min Samples: ", min_sample)
        print("DBSCAN Clustering: ", silhouette_score(wine_df, db_clusters))
    ```

    输出如下：

    ![图 3.18: 打印聚类的轮廓分数    ](img/C12626_03_18.jpg)

    ###### 图 3.18: 打印聚类的轮廓分数

1.  根据最高轮廓分数生成最终聚类 (`eps`: 35, `min_samples`: 3)：

    ```py
    # Generate clusters using DBSCAN
    db = DBSCAN(eps=35, min_samples = 3)
    db_clusters = db.fit_predict(wine_df)
    ```

1.  可视化使用三种方法生成的聚类：

    ```py
    plt.title("Wine Clusters from K-Means")
    plt.scatter(wine_df['OD_read'], wine_df['Proline'], c=km_clusters,s=50, cmap='tab20b')
    plt.show()
    plt.title("Wine Clusters from Agglomerative Clustering")
    plt.scatter(wine_df['OD_read'], wine_df['Proline'], c=ac_clusters,s=50, cmap='tab20b')
    plt.show()
    plt.title("Wine Clusters from DBSCAN")
    plt.scatter(wine_df['OD_read'], wine_df['Proline'], c=db_clusters,s=50, cmap='tab20b')
    plt.show()
    ```

    输出如下：

    ![图 3.19: 使用不同算法绘制聚类图    ](img/C12626_03_19.jpg)

    ###### 图 3.19: 使用不同算法绘制聚类图

1.  评估每种方法的轮廓分数：

    ```py
    # Calculate Silhouette Scores
    print("Silhouette Scores for Wine Dataset:\n")
    print("K-Means Clustering: ", silhouette_score(wine_df, km_clusters))
    print("Agg Clustering: ", silhouette_score(wine_df, ac_clusters))
    print("DBSCAN Clustering: ", silhouette_score(wine_df, db_clusters))
    ```

    输出如下：

![图 3.20: Silhouette 分数](img/C12626_03_20.jpg)

###### 图 3.20: Silhouette 分数

如您所见，DBSCAN 并不总是自动适合您的聚类需求。使其与其他算法不同的一个关键特征是将噪声用作潜在聚类。在某些情况下，这很好，因为它可以去除离群值，但是，可能存在调整不足的情况，会将太多点分类为噪声。您能通过调整超参数来提高轮廓分数吗？

## 第四章: 维度缩减和 PCA

### 活动 6: 手动 PCA 与 scikit-learn 对比

解决方案

1.  导入 `pandas`、`numpy` 和 `matplotlib` 绘图库以及 scikit-learn 的 `PCA` 模型：

    ```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    ```

1.  加载数据集，并根据以前的练习选择仅选择萼片特征。显示数据的前五行：

    ```py
    df = pd.read_csv('iris-data.csv')
    df = df[['Sepal Length', 'Sepal Width']]
    df.head()
    ```

    输出如下：

    ![图 4.43: 数据的前五行    ](img/C12626_04_43.jpg)

    ###### 图 4.43: 数据的前五行

1.  计算数据的协方差矩阵：

    ```py
    cov = np.cov(df.values.T)
    cov
    ```

    输出如下：

    ![图 4.44: 数据的协方差矩阵    ](img/C12626_04_44.jpg)

    ###### 图 4.44: 数据的协方差矩阵

1.  使用 scikit-learn API 转换数据，仅使用第一个主成分。将转换后的数据存储在 `sklearn_pca` 变量中：

    ```py
    model = PCA(n_components=1)
    sklearn_pca = model.fit_transform(df.values)
    ```

1.  使用手动 PCA 仅使用第一个主成分来转换数据。将转换后的数据存储在 `manual_pca` 变量中。

    ```py
    eigenvectors, eigenvalues, _ = np.linalg.svd(cov, full_matrices=False)
    P = eigenvectors[0]
    manual_pca = P.dot(df.values.T)
    ```

1.  在同一图上绘制 `sklearn_pca` 和 `manual_pca` 的值以可视化差异：

    ```py
    plt.figure(figsize=(10, 7));
    plt.plot(sklearn_pca, label='Scikit-learn PCA');
    plt.plot(manual_pca, label='Manual PCA', linestyle='--');
    plt.xlabel('Sample');
    plt.ylabel('Transformed Value');
    plt.legend();
    ```

    输出如下：

    ![图 4.45: 数据的图    ](img/C12626_04_45.jpg)

    ###### 图 4.45: 数据的图

1.  请注意，两个图几乎完全相同，唯一的区别是一个是另一个的镜像，且两者之间存在偏移。显示`sklearn_pca`和`manual_pca`模型的主成分：

    ```py
    model.components_
    ```

    输出如下：

    ```py
    array([[ 0.99693955, -0.07817635]])
    ```

    现在打印`P`：

    ```py
    P
    ```

    输出如下：

    ```py
    array([-0.99693955,  0.07817635])
    ```

    注意符号的差异；值是相同的，但符号不同，产生了镜像的结果。这只是约定上的差异，并无实质意义。

1.  将`manual_pca`模型乘以`-1`并重新绘制：

    ```py
    manual_pca *= -1
    plt.figure(figsize=(10, 7));
    plt.plot(sklearn_pca, label='Scikit-learn PCA');
    plt.plot(manual_pca, label='Manual PCA', linestyle='--');
    plt.xlabel('Sample');
    plt.ylabel('Transformed Value');
    plt.legend();
    ```

    输出如下：

    ![图 4.46: 重新绘制的数据    ](img/C12626_04_46.jpg)

    ###### 图 4.46: 重新绘制的数据

1.  现在，我们需要做的就是处理两个之间的偏移。scikit-learn API 在变换之前会减去数据的均值。在使用手动 PCA 进行变换之前，从数据集中减去每列的均值：

    ```py
    mean_vals = np.mean(df.values, axis=0)
    offset_vals = df.values - mean_vals
    manual_pca = P.dot(offset_vals.T)
    ```

1.  将结果乘以`-1`：

    ```py
    manual_pca *= -1
    ```

1.  重新绘制单独的`sklearn_pca`和`manual_pca`值：

    ```py
    plt.figure(figsize=(10, 7));
    plt.plot(sklearn_pca, label='Scikit-learn PCA');
    plt.plot(manual_pca, label='Manual PCA', linestyle='--');
    plt.xlabel('Sample');
    plt.ylabel('Transformed Value');
    plt.legend();
    ```

    输出如下：

![图 4.47: 重新绘制的数据](img/C12626_04_47.jpg)

###### 图 4.47: 重新绘制数据

最终的图将展示通过这两种方法完成的降维实际上是相同的。不同之处在于`协方差`矩阵的符号差异，因为这两种方法只是使用了不同的特征作为比较的基准。最后，两数据集之间也存在偏移，这是由于在执行 scikit-learn PCA 变换之前已将均值样本减去。

### 活动 7: 使用扩展的鸢尾花数据集进行主成分分析（PCA）

解决方案：

1.  导入`pandas`和`matplotlib`。为了启用 3D 绘图，您还需要导入`Axes3D`：

    ```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
    ```

1.  读取数据集并选择`花萼长度`、`花萼宽度`和`花瓣宽度`列：

    ```py
    df = pd.read_csv('iris-data.csv')[['Sepal Length', 'Sepal Width', 'Petal Width']]
    df.head()
    ```

    输出如下：

    ![图 4.48: 花萼长度、花萼宽度和花瓣宽度    ](img/C12626_04_48.jpg)

    ###### 图 4.48: 花萼长度、花萼宽度和花瓣宽度

1.  绘制三维数据：

    ```py
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Sepal Length'], df['Sepal Width'], df['Petal Width']);
    ax.set_xlabel('Sepal Length (mm)');
    ax.set_ylabel('Sepal Width (mm)');
    ax.set_zlabel('Petal Width (mm)');
    ax.set_title('Expanded Iris Dataset');
    ```

    输出如下：

    ![图 4.49: 扩展的鸢尾花数据集图    ](img/C12626_04_49.jpg)

    ###### 图 4.49: 扩展的鸢尾花数据集图

1.  创建一个`PCA`模型，不指定主成分数：

    ```py
    model = PCA()
    ```

1.  将模型拟合到数据集：

    ```py
    model.fit(df.values)
    ```

    输出如下：

    ![图 4.50: 拟合到数据集的模型    ](img/C12626_04_50.jpg)

    ###### 图 4.50: 拟合到数据集的模型

1.  显示特征值或`explained_variance_ratio_`：

    ```py
    model.explained_variance_ratio_
    ```

    输出如下：

    ```py
    array([0.8004668 , 0.14652357, 0.05300962])
    ```

1.  我们希望减少数据集的维度，但仍保持至少 90% 的方差。为了保持 90% 的方差，所需的最小主成分数是多少？

    前两个主成分需要至少 90% 的方差。前两个主成分提供了数据集中 94.7% 的方差。

1.  创建一个新的`PCA`模型，这次指定所需的主成分数，以保持至少 90% 的方差：

    ```py
    model = PCA(n_components=2)
    ```

1.  使用新模型转换数据：

    ```py
    data_transformed = model.fit_transform(df.values)
    ```

1.  绘制转换后的数据：

    ```py
    plt.figure(figsize=(10, 7))
    plt.scatter(data_transformed[:,0], data_transformed[:,1]);
    ```

    输出如下：

    ![图 4.51: 转换数据的图    ](img/C12626_04_51.jpg)

    ###### 图 4.51：转换数据的图

1.  将转换后的数据恢复到原始数据空间：

    ```py
    data_restored = model.inverse_transform(data_transformed)
    ```

1.  在一个子图中绘制恢复后的三维数据，在第二个子图中绘制原始数据，以可视化去除部分方差的效果：

    ```py
    fig = plt.figure(figsize=(10, 14))
    # Original Data
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(df['Sepal Length'], df['Sepal Width'], df['Petal Width'], label='Original Data');
    ax.set_xlabel('Sepal Length (mm)');
    ax.set_ylabel('Sepal Width (mm)');
    ax.set_zlabel('Petal Width (mm)');
    ax.set_title('Expanded Iris Dataset');
    # Transformed Data
    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(data_restored[:,0], data_restored[:,1], data_restored[:,2], label='Restored Data');
    ax.set_xlabel('Sepal Length (mm)');
    ax.set_ylabel('Sepal Width (mm)');
    ax.set_zlabel('Petal Width (mm)');
    ax.set_title('Restored Iris Dataset');
    ```

    输出如下：

![图 4.52：扩展和恢复后的鸢尾花数据集图](img/C12626_04_52.jpg)

###### 图 4.52：扩展和恢复后的鸢尾花数据集图

看看*图 4.52*，我们可以看到，就像我们在二维图中做的那样，我们已经去除了数据中的大量噪声，但保留了关于数据趋势的最重要信息。可以看出，通常情况下，花萼长度与花瓣宽度成正比，并且在图中似乎有两个数据簇，一个在另一个上方。

#### 注意

在应用 PCA 时，重要的是要考虑所建模数据的大小以及可用的系统内存。奇异值分解过程涉及将数据分解为特征值和特征向量，且可能非常占用内存。如果数据集过大，你可能无法完成该过程，或者会遭遇显著的性能损失，甚至可能导致系统崩溃。

## 第五章：自编码器

### 活动 8：使用 ReLU 激活函数建模神经元

解决方案：

1.  导入`numpy`和 matplotlib：

    ```py
    import numpy as np
    import matplotlib.pyplot as plt
    ```

1.  允许在标签中使用 latex 符号：

    ```py
    plt.rc('text', usetex=True)
    ```

1.  将 ReLU 激活函数定义为 Python 函数：

    ```py
    def relu(x):
        return np.max((0, x))
    ```

1.  定义神经元的输入（`x`）和可调权重（`theta`）。在此示例中，输入（`x`）将是-5 到 5 之间线性间隔的 100 个数字。设置`theta = 1`：

    ```py
    theta = 1
    x = np.linspace(-5, 5, 100)
    x
    ```

    输出如下：

    ![图 5.35：打印输入    ](img/C12626_05_35.jpg)

    ###### 图 5.35：打印输入

1.  计算输出（`y`）：

    ```py
    y = [relu(_x * theta) for _x in x]
    ```

1.  绘制神经元的输出与输入的关系图：

    ```py
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_xlabel('$x$', fontsize=22);
    ax.set_ylabel('$h(x\Theta)$', fontsize=22);
    ax.spines['left'].set_position(('data', 0));
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    ax.tick_params(axis='both', which='major', labelsize=22)
    ```

    输出如下：

    ![图 5.36：神经元与输入的关系图    ](img/C12626_05_36.jpg)

    ###### 图 5.36：神经元与输入的关系图

1.  现在，将`theta = 5`并重新计算并存储神经元的输出：

    ```py
    theta = 5
    y_2 = [relu(_x * theta) for _x in x]
    ```

1.  现在，将`theta = 0.2`并重新计算并存储神经元的输出：

    ```py
    theta = 0.2
    y_3 = [relu(_x * theta) for _x in x]
    ```

1.  在一张图上绘制神经元的三条不同输出曲线（`theta = 1`，`theta = 5`，`theta = 0.2`）：

    ```py
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.plot(x, y, label='$\Theta=1$');
    ax.plot(x, y_2, label='$\Theta=5$', linestyle=':');
    ax.plot(x, y_3, label='$\Theta=0.2$', linestyle='--');
    ax.set_xlabel('$x\Theta$', fontsize=22);
    ax.set_ylabel('$h(x\Theta)$', fontsize=22);
    ax.spines['left'].set_position(('data', 0));
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    ax.tick_params(axis='both', which='major', labelsize=22);
    ax.legend(fontsize=22);
    ```

    输出如下：

![图 5.37：神经元的三条输出曲线](img/C12626_05_37.jpg)

###### 图 5.37：神经元的三条输出曲线

在这个活动中，我们创建了一个基于 ReLU 的人工神经网络神经元模型。我们可以看到，这个神经元的输出与 sigmoid 激活函数的输出非常不同。对于大于 0 的值，没有饱和区域，因为它仅返回函数的输入值。在负方向上，有一个饱和区域，如果输入小于 0，则返回 0。ReLU 函数是一种非常强大且常用的激活函数，在某些情况下，它比 sigmoid 函数更强大。ReLU 通常是一个很好的首选激活函数。

### 活动 9：MNIST 神经网络

解决方案：

在这个活动中，您将训练一个神经网络来识别 MNIST 数据集中的图像，并强化您的神经网络训练技能：

1.  导入`pickle`、`numpy`、`matplotlib`，以及从 Keras 中导入`Sequential`和`Dense`类：

    ```py
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Dense
    ```

1.  加载`mnist.pkl`文件，其中包含来自 MNIST 数据集的前 10,000 张图像及其对应的标签，这些数据可以在随附的源代码中找到。MNIST 数据集是一系列 28 x 28 灰度图像，表示手写数字 0 到 9。提取图像和标签：

    ```py
    with open('mnist.pkl', 'rb') as f:
        data = pickle.load(f)

    images = data['images']
    labels = data['labels']
    ```

1.  绘制前 10 个样本及其对应的标签：

    ```py
    plt.figure(figsize=(10, 7))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    ```

    输出如下：

    ![图 5.38：前 10 个样本    ](img/C12626_05_38.jpg)

    ###### 图 5.38：前 10 个样本

1.  使用独热编码对标签进行编码：

    ```py
    one_hot_labels = np.zeros((images.shape[0], 10))
    for idx, label in enumerate(labels):
        one_hot_labels[idx, label] = 1

    one_hot_labels
    ```

    输出如下：

    ![图 5.39：独热编码结果    ](img/C12626_05_39.jpg)

    ###### 图 5.39：独热编码结果

1.  为神经网络输入准备图像。提示：此过程包含两个独立的步骤：

    ```py
    images = images.reshape((-1, 28 ** 2))
    images = images / 255.
    ```

1.  在 Keras 中构建一个神经网络模型，该模型接受准备好的图像，包含一个具有 600 个单位的隐藏层，并使用 ReLU 激活函数，输出层的单元数与类别数相同。输出层使用`softmax`激活函数：

    ```py
    model = Sequential([
        Dense(600, input_shape=(784,), activation='relu'),
        Dense(10, activation='softmax'),
    ])
    ```

1.  使用多类交叉熵、随机梯度下降以及准确度作为性能指标来编译模型：

    ```py
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    ```

1.  训练模型。需要多少个 epoch 才能在训练数据上达到至少 95%的分类准确率？让我们来看看：

    ```py
    model.fit(images, one_hot_labels, epochs=20)
    ```

    输出如下：

![图 5.40：训练模型](img/C12626_05_40.jpg)

###### 图 5.40：训练模型

需要 15 个 epoch 才能在训练集上达到至少 95%的分类准确率。

在这个示例中，我们使用分类器训练时的数据来测量神经网络分类器的性能。一般来说，这种方法不应该使用，因为它通常报告比实际模型应该有的准确度更高。在监督学习问题中，有多种**交叉验证**技术应该被使用。由于这是一本关于无监督学习的书，交叉验证超出了本书的范围。

### 活动 10：简单的 MNIST 自编码器

解决方案：

1.  导入 `pickle`、`numpy` 和 `matplotlib`，以及从 Keras 导入 `Model`、`Input` 和 `Dense` 类：

    ```py
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Model
    from keras.layers import Input, Dense
    ```

1.  从提供的 MNIST 数据集样本（随附源代码 `mnist.pkl`）加载图像：

    ```py
    with open('mnist.pkl', 'rb') as f:
        images = pickle.load(f)['images']
    ```

1.  准备图像以输入神经网络。作为提示，此过程有 **两个** 独立步骤：

    ```py
    images = images.reshape((-1, 28 ** 2))
    images = images / 255.
    ```

1.  构建一个简单的自编码器网络，使图像在编码阶段后缩小到 10 x 10：

    ```py
    input_stage = Input(shape=(784,))
    encoding_stage = Dense(100, activation='relu')(input_stage)
    decoding_stage = Dense(784, activation='sigmoid')(encoding_stage)
    autoencoder = Model(input_stage, decoding_stage)
    ```

1.  使用二元交叉熵损失函数和 `adadelta` 梯度下降法编译自编码器：

    ```py
    autoencoder.compile(loss='binary_crossentropy',
                  optimizer='adadelta')
    ```

1.  拟合编码器模型：

    ```py
    autoencoder.fit(images, images, epochs=100)
    ```

    输出结果如下：

    ![图 5.41: 训练模型    ](img/C12626_05_41.jpg)

    ###### 图 5.41: 训练模型

1.  计算并存储编码阶段的前五个样本的输出：

    ```py
    encoder_output = Model(input_stage, encoding_stage).predict(images[:5])
    ```

1.  将编码器输出重塑为 10 x 10（10 x 10 = 100）像素并乘以 255：

    ```py
    encoder_output = encoder_output.reshape((-1, 10, 10)) * 255
    ```

1.  计算并存储解码阶段的前五个样本的输出：

    ```py
    decoder_output = autoencoder.predict(images[:5])
    ```

1.  将解码器的输出重塑为 28 x 28，并乘以 255：

    ```py
    decoder_output = decoder_output.reshape((-1, 28, 28)) * 255
    ```

1.  绘制原始图像、编码器输出和解码器：

    ```py
    images = images.reshape((-1, 28, 28))
    plt.figure(figsize=(10, 7))
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.subplot(3, 5, i + 6)
        plt.imshow(encoder_output[i], cmap='gray')
        plt.axis('off')   

        plt.subplot(3, 5, i + 11)
        plt.imshow(decoder_output[i], cmap='gray')
        plt.axis('off')    
    ```

    输出结果如下：

![图 5.42: 原始图像、编码器输出和解码器](img/C12626_05_42.jpg)

###### 图 5.42: 原始图像、编码器输出和解码器

到目前为止，我们已经展示了如何使用编码和解码阶段的简单单层隐藏层来将数据降到低维空间。我们还可以通过在编码和解码阶段添加额外的层来使这个模型更加复杂。

### 活动 11: MNIST 卷积自编码器

解决方案：

1.  导入 `pickle`、`numpy`、`matplotlib`，以及从 `keras.models` 导入 `Model` 类，导入从 `keras.layers` 的 `Input`、`Conv2D`、`MaxPooling2D` 和 `UpSampling2D`：

    ```py
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    ```

1.  加载数据：

    ```py
    with open('mnist.pkl', 'rb') as f:
        images = pickle.load(f)['images']
    ```

1.  将图像重新缩放为 0 和 1 之间的值：

    ```py
    images = images / 255.
    ```

1.  我们需要重塑图像，以便为卷积阶段添加一个单独的深度通道。将图像重塑为 28 x 28 x 1 的形状：

    ```py
    images = images.reshape((-1, 28, 28, 1))
    ```

1.  定义输入层。我们将使用与图像相同的输入形状：

    ```py
    input_layer = Input(shape=(28, 28, 1,))
    ```

1.  添加一个具有 16 层或滤波器的卷积阶段，使用 3 x 3 的权重矩阵、ReLU 激活函数，并使用相同的填充，这意味着输出的长度与输入图像相同：

    ```py
    hidden_encoding = Conv2D(
        16, # Number of layers or filters in the weight matrix
        (3, 3), # Shape of the weight matrix
        activation='relu',
        padding='same', # How to apply the weights to the images
    )(input_layer)
    ```

1.  向编码器添加一个最大池化层，使用 2 x 2 的内核：

    ```py
    encoded = MaxPooling2D((2, 2))(hidden_encoding)
    ```

1.  添加一个解码卷积层：

    ```py
    hidden_decoding = Conv2D(
        16, # Number of layers or filters in the weight matrix
        (3, 3), # Shape of the weight matrix
        activation='relu',
        padding='same', # How to apply the weights to the images
    )(encoded)
    ```

1.  添加上采样层：

    ```py
    upsample_decoding = UpSampling2D((2, 2))(hidden_decoding)
    ```

1.  添加最后的卷积阶段，使用一个层，按照初始图像深度进行：

    ```py
    decoded = Conv2D(
        1, # Number of layers or filters in the weight matrix
        (3, 3), # Shape of the weight matrix
        activation='sigmoid',
        padding='same', # How to apply the weights to the images
    )(upsample_decoding)
    ```

1.  通过将网络的第一层和最后一层传递给 `Model` 类来构建模型：

    ```py
    autoencoder = Model(input_layer, decoded)
    ```

1.  显示模型的结构：

    ```py
    autoencoder.summary()
    ```

    输出结果如下：

    ![图 5.43: 模型结构    ](img/C12626_05_43.jpg)

    ###### 图 5.43: 模型结构

1.  使用二元交叉熵损失函数和 `adadelta` 梯度下降法编译自编码器：

    ```py
    autoencoder.compile(loss='binary_crossentropy',
                  optimizer='adadelta')
    ```

1.  现在，我们来拟合模型；再次传递图像作为训练数据和期望输出。由于卷积网络计算时间较长，因此训练 20 个周期：

    ```py
    autoencoder.fit(images, images, epochs=20)
    ```

    输出结果如下：

    ![图 5.44: 训练模型    ](img/C12626_05_44.jpg)

    ###### 图 5.44：训练模型

1.  计算并存储前五个样本的编码阶段输出：

    ```py
    encoder_output = Model(input_layer, encoded).predict(images[:5])
    ```

1.  为可视化重新调整编码器输出的形状，每个图像的大小为 X*Y：

    ```py
    encoder_output = encoder_output.reshape((-1, 14 * 14, 16))
    ```

1.  获取前五张图像的解码器输出：

    ```py
    decoder_output = autoencoder.predict(images[:5])
    ```

1.  将解码器输出调整为 28 x 28 的大小：

    ```py
    decoder_output = decoder_output.reshape((-1, 28, 28))
    ```

1.  将原始图像调整回 28 x 28 的大小：

    ```py
    images = images.reshape((-1, 28, 28))
    ```

1.  绘制原始图像、平均编码器输出和解码器：

    ```py
    plt.figure(figsize=(10, 7))
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')

        plt.subplot(3, 5, i + 6)
        plt.imshow(encoder_output[i], cmap='gray')
        plt.axis('off')   

        plt.subplot(3, 5, i + 11)
        plt.imshow(decoder_output[i], cmap='gray')
        plt.axis('off')        
    ```

    输出如下：

![图 5.45：原始图像、编码器输出和解码器](img/C12626_05_45.jpg)

###### 图 5.45：原始图像、编码器输出和解码器

在此活动结束时，你将开发一个包含卷积层的自编码器神经网络。注意解码器表示中的改进。与全连接神经网络层相比，这种架构在性能上具有显著优势，非常适用于处理基于图像的数据集和生成人工数据样本。

## 第六章：t-分布随机邻域嵌入（t-SNE）

### 活动 12：葡萄酒 t-SNE

解决方案：

1.  导入 `pandas`、`numpy`、`matplotlib` 以及来自 scikit-learn 的 `t-SNE` 和 `PCA` 模型：

    ```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    ```

1.  使用随附源代码中的 `wine.data` 文件加载葡萄酒数据集，并显示数据集的前五行：

    ```py
    df = pd.read_csv('wine.data', header=None)
    df.head()
    ```

    输出如下：

    ![图 6.24：葡萄酒数据集的前五行。    ](img/C12626_06_24.jpg)

    ###### 图 6.24：葡萄酒数据集的前五行。

1.  第一列包含标签；提取该列并将其从数据集中移除：

    ```py
    labels = df[0]
    del df[0]
    ```

1.  执行 PCA 降维，将数据集缩减到前六个组件：

    ```py
    model_pca = PCA(n_components=6)
    wine_pca = model_pca.fit_transform(df)
    ```

1.  确定这六个组件描述的数据中的方差量：

    ```py
    np.sum(model_pca.explained_variance_ratio_)
    ```

    输出如下：

    ```py
    0.99999314824536
    ```

1.  创建一个使用指定随机状态和 `verbose` 值为 1 的 t-SNE 模型：

    ```py
    tsne_model = TSNE(random_state=0, verbose=1)
    tsne_model
    ```

    输出如下：

    ![图 6.25：创建 t-SNE 模型。    ](img/C12626_06_25.jpg)

    ###### 图 6.25：创建 t-SNE 模型。

1.  将 PCA 数据拟合到 t-SNE 模型：

    ```py
    wine_tsne = tsne_model.fit_transform(wine_pca.reshape((len(wine_pca), -1)))
    ```

    输出如下：

    ![图 6.26：拟合 PCA 数据 t-SNE 模型    ](img/C12626_06_26.jpg)

    ###### 图 6.26：拟合 PCA 数据 t-SNE 模型

1.  确认 t-SNE 拟合数据的形状是二维的：

    ```py
    wine_tsne.shape
    ```

    输出如下：

    ```py
    (172, 8)
    ```

1.  创建二维数据的散点图：

    ```py
    plt.figure(figsize=(10, 7))
    plt.scatter(wine_tsne[:,0], wine_tsne[:,1]);
    plt.title('Low Dimensional Representation of Wine');
    plt.show()
    ```

    输出如下：

    ![图 6.27：二维数据的散点图    ](img/C12626_06_27.jpg)

    ###### 图 6.27：二维数据的散点图

1.  创建一个带有类别标签的二维数据次级散点图，以可视化可能存在的任何聚类：

    ```py
    MARKER = ['o', 'v', '^',]
    plt.figure(figsize=(10, 7))
    plt.title('Low Dimensional Representation of Wine');
    for i in range(1, 4):
        selections = wine_tsne[labels == i]
        plt.scatter(selections[:,0], selections[:,1], marker=MARKER[i-1], label=f'Wine {i}', s=30);
        plt.legend();
    plt.show()
    ```

    输出如下：

![图 6.28：二维数据的次级图](img/C12626_06_28.jpg)

###### 图 6.28：二维数据的次级图

请注意，虽然酒类之间有重叠，但也可以看到数据中存在一些聚类。第一类酒主要位于图表的左上角，第二类酒位于右下角，第三类酒则位于前两者之间。这个表示方法当然不能用于高信心地对单个酒样进行分类，但它展示了一个总体趋势以及在之前无法看到的高维数据中包含的一系列聚类。

### 活动 13：t-SNE 酒类与困惑度

解决方案：

1.  导入 `pandas`、`numpy`、`matplotlib` 和 `t-SNE` 与 `PCA` 模型（来自 scikit-learn）：

    ```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    ```

1.  加载酒类数据集并检查前五行：

    ```py
    df = pd.read_csv('wine.data', header=None)
    df.head()
    ```

    输出结果如下：

    ![图 6.29：酒类数据的前五行    ](img/C12626_06_29.jpg)

    ###### 图 6.29：酒类数据的前五行

1.  第一列提供了标签；从数据框中提取它们并将其存储在一个单独的变量中。确保该列从数据框中移除：

    ```py
    labels = df[0]
    del df[0]
    ```

1.  对数据集执行 PCA 并提取前六个主成分：

    ```py
    model_pca = PCA(n_components=6)
    wine_pca = model_pca.fit_transform(df)
    wine_pca = wine_pca.reshape((len(wine_pca), -1))
    ```

1.  构建一个循环，遍历困惑度值（1、5、20、30、80、160、320）。对于每个循环，生成一个对应困惑度的 t-SNE 模型，并绘制标记酒类的散点图。注意不同困惑度值的效果：

    ```py
    MARKER = ['o', 'v', '^',]
    for perp in [1, 5, 20, 30, 80, 160, 320]:
        tsne_model = TSNE(random_state=0, verbose=1, perplexity=perp)
        wine_tsne = tsne_model.fit_transform(wine_pca)
        plt.figure(figsize=(10, 7))
        plt.title(f'Low Dimensional Representation of Wine. Perplexity {perp}');
        for i in range(1, 4):
            selections = wine_tsne[labels == i]
            plt.scatter(selections[:,0], selections[:,1], marker=MARKER[i-1], label=f'Wine {i}', s=30);
            plt.legend();
    ```

    `1`的困惑度值无法将数据分离成任何特定的结构：

![图 6.30：困惑度值为 1 的图表](img/C12626_06_30.jpg)

###### 图 6.30：困惑度值为 1 的图表

将困惑度增加到 5 会导致非常非线性的结构，难以分离，并且很难识别任何聚类或模式：

![图 6.31：困惑度为 5 的图表](img/C12626_06_31.jpg)

###### 图 6.31：困惑度为 5 的图表

困惑度为 20 最终开始显示某种马蹄形结构。虽然在视觉上明显，但实现起来仍然可能有些困难：

![图 6.32：困惑度为 20 的图表](img/C12626_06_32.jpg)

###### 图 6.32：困惑度为 20 的图表

困惑度为 30 显示出相当好的结果。投影结构之间存在一定的线性关系，并且酒类之间有一些分离：

![图 6.33：困惑度为 30 的图表](img/C12626_06_33.jpg)

###### 图 6.33：困惑度为 30 的图表

最后，活动中的最后两张图片展示了随着困惑度的增加，图表如何变得越来越复杂和非线性：

![图 6.34：困惑度为 80 的图表](img/C12626_06_34.jpg)

###### 图 6.34：困惑度为 80 的图表

这是困惑度为 160 的图表：

![图 6.35：困惑度为 160 的图表](img/C12626_06_35.jpg)

###### 图 6.35：困惑度为 160 的图表

查看每个困惑度值的单独图形，困惑度对数据可视化的影响立刻显现出来。非常小或非常大的困惑度值会产生一系列不寻常的形状，无法显示出任何持续的模式。最合理的值似乎是 30，它产生了我们在前一个活动中看到的最线性的图形。

在这次活动中，我们展示了在选择困惑度时需要小心，并且可能需要一些迭代来确定正确的值。

### 活动 14：t-SNE 葡萄酒与迭代

解决方案：

1.  导入`pandas`、`numpy`、`matplotlib`，以及从 scikit-learn 导入的`t-SNE`和`PCA`模型：

    ```py
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    ```

1.  加载葡萄酒数据集并检查前五行：

    ```py
    df = pd.read_csv('wine.data', header=None)
    df.head()
    ```

    输出结果如下：

    ![图 6.36：葡萄酒数据集的前五行](img/C12626_06_36.jpg)

    ](img/C12626_06_36.jpg)

    ###### 图 6.36：葡萄酒数据集的前五行

1.  第一列提供标签；从 DataFrame 中提取这些标签并存储到一个单独的变量中。确保将该列从 DataFrame 中移除：

    ```py
    labels = df[0]
    del df[0]
    ```

1.  在数据集上执行 PCA 并提取前六个主成分：

    ```py
    model_pca = PCA(n_components=6)
    wine_pca = model_pca.fit_transform(df)
    wine_pca = wine_pca.reshape((len(wine_pca), -1))
    ```

1.  构建一个循环，遍历迭代值（`250`、`500`、`1000`）。对于每次循环，生成一个具有相应迭代次数的 t-SNE 模型，并生成相同迭代次数且没有进度值的模型：

    ```py
    MARKER = ['o', 'v', '1', 'p' ,'*', '+', 'x', 'd', '4', '.']
    for iterations in [250, 500, 1000]:
        model_tsne = TSNE(random_state=0, verbose=1, n_iter=iterations, n_iter_without_progress=iterations)
        mnist_tsne = model_tsne.fit_transform(mnist_pca)
    ```

1.  构建一个葡萄酒类别的散点图。注意不同迭代值的效果：

    ```py
        plt.figure(figsize=(10, 7))
        plt.title(f'Low Dimensional Representation of MNIST (iterations = {iterations})');
        for i in range(10):
            selections = mnist_tsne[mnist['labels'] == i]
            plt.scatter(selections[:,0], selections[:,1], alpha=0.2, marker=MARKER[i], s=5);
            x, y = selections.mean(axis=0)
            plt.text(x, y, str(i), fontdict={'weight': 'bold', 'size': 30}) 
    ```

    输出结果如下：

![图 6.37：250 次迭代的葡萄酒类别散点图](img/C12626_06_37.jpg)

](img/C12626_06_37.jpg)

###### 图 6.37：250 次迭代的葡萄酒类别散点图

这是 500 次迭代的图：

![图 6.38：500 次迭代的葡萄酒类别散点图](img/C12626_06_38.jpg)

](img/C12626_06_38.jpg)

###### 图 6.38：500 次迭代的葡萄酒类别散点图

这是 1,000 次迭代的图：

![图 6.39：1,000 次迭代的葡萄酒类别散点图](img/C12626_06_39.jpg)

](img/C12626_06_39.jpg)

###### 图 6.39：1,000 次迭代的葡萄酒类别散点图

再次，我们可以看到随着迭代次数的增加，数据结构的改善。即使在像这样的相对简单的数据集中，250 次迭代也不足以将数据结构投影到低维空间中。

正如我们在相应活动中观察到的，设置迭代参数时需要找到一个平衡点。在这个例子中，250 次迭代是不足够的，至少需要 1,000 次迭代才能最终稳定数据。

## 第七章：主题建模

### 活动 15：加载和清理 Twitter 数据

解决方案：

1.  导入必要的库：

    ```py
    import langdetect
    import matplotlib.pyplot
    import nltk
    import numpy
    import pandas
    import pyLDAvis
    import pyLDAvis.sklearn
    import regex
    import sklearn
    ```

1.  从[`github.com/TrainingByPackt/Applied-Unsupervised-Learning-with-Python/tree/master/Lesson07/Activity15-Activity17`](https://github.com/TrainingByPackt/Applied-Unsupervised-Learning-with-Python/tree/master/Lesson07/Activity15-Activity17)加载 LA Times 健康 Twitter 数据(`latimeshealth.txt`)：

    #### 注意

    ```py
    path = '<Path>/latimeshealth.txt'
    df = pandas.read_csv(path, sep="|", header=None)
    df.columns = ["id", "datetime", "tweettext"]
    ```

1.  运行快速探索性分析以确定数据大小和结构：

    ```py
    def dataframe_quick_look(df, nrows):
    print("SHAPE:\n{shape}\n".format(shape=df.shape))
    print("COLUMN NAMES:\n{names}\n".format(names=df.columns))
    print("HEAD:\n{head}\n".format(head=df.head(nrows)))
    dataframe_quick_look(df, nrows=2)
    ```

    输出如下：

    ![图 7.54：数据的形状、列名和数据头    ](img/C12626_07_54.jpg)

    ###### 图 7.54：数据的形状、列名和数据头

1.  提取推文文本并转换为列表对象：

    ```py
    raw = df['tweettext'].tolist()
    print("HEADLINES:\n{lines}\n".format(lines=raw[:5]))
    print("LENGTH:\n{length}\n".format(length=len(raw)))
    ```

    输出如下：

    ![图 7.55：标题及其长度    ](img/C12626_07_55.jpg)

    ###### 图 7.55：标题及其长度

1.  编写函数，执行语言检测、基于空格的分词、替换屏幕名和 URL 为 `SCREENNAME` 和 `URL`。该函数还应删除标点、数字以及 `SCREENNAME` 和 `URL` 的替换。将所有内容转换为小写，除了 `SCREENNAME` 和 `URL`。它应删除所有停用词，执行词形还原，并保留五个或更多字母的单词：

    #### 注意

    ```py
    def do_language_identifying(txt):
        	try:
               the_language = langdetect.detect(txt)
        	except:
            	the_language = 'none'
        	return the_language
    def do_lemmatizing(wrd):
        	out = nltk.corpus.wordnet.morphy(wrd)
        	return (wrd if out is None else out)
    def do_tweet_cleaning(txt):
    # identify language of tweet
    # return null if language not english
        	lg = do_language_identifying(txt)
        	if lg != 'en':
            	return None
    # split the string on whitespace
        	out = txt.split(' ')
    # identify screen names
    # replace with SCREENNAME
        	out = ['SCREENNAME' if i.startswith('@') else i for i in out]
    # identify urls
    # replace with URL
        	out = ['URL' if bool(regex.search('http[s]?://', i)) else i for i in out]
          # remove all punctuation
        	out = [regex.sub('[^\\w\\s]|\n', '', i) for i in out]
          # make all non-keywords lowercase
        	keys = ['SCREENNAME', 'URL']
        	out = [i.lower() if i not in keys else i for i in out]
          # remove keywords
        	out = [i for i in out if i not in keys]
          # remove stopwords
        	list_stop_words = nltk.corpus.stopwords.words('english')
        	list_stop_words = [regex.sub('[^\\w\\s]', '', i) for i in list_stop_words]
        	out = [i for i in out if i not in list_stop_words]
          # lemmatizing
        	out = [do_lemmatizing(i) for i in out]
          # keep words 4 or more characters long
        	out = [i for i in out if len(i) >= 5]
        	return out
    ```

1.  对每条推文应用第 5 步中定义的函数：

    ```py
    clean = list(map(do_tweet_cleaning, raw))
    ```

1.  删除等于 `None` 的输出列表元素：

    ```py
    clean = list(filter(None.__ne__, clean))
    print("HEADLINES:\n{lines}\n".format(lines=clean[:5]))
    print("LENGTH:\n{length}\n".format(length=len(clean)))
    ```

    输出如下：

    ![图 7.56：删除 None 后的标题和长度    ](img/C12626_07_56.jpg)

    ###### 图 7.56：删除 None 后的标题和长度

1.  将每条推文的元素转换回字符串。使用空格连接：

    ```py
    clean_sentences = [" ".join(i) for i in clean]
    print(clean_sentences[0:10])
    ```

    输出列表的前 10 个元素应如下所示：

    ![图 7.57：用于建模的清洁推文    ](img/C12626_07_57.jpg)

    ###### 图 7.57：用于建模的清洁推文

1.  保持笔记本打开以供将来建模。

### 活动 16：潜在狄利克雷分配与健康推文

解决方案：

1.  指定 `number_words`、`number_docs` 和 `number_features` 变量：

    ```py
    number_words = 10
    number_docs = 10
    number_features = 1000
    ```

1.  创建词袋模型，并将特征名分配给另一个变量以供以后使用：

    ```py
    vectorizer1 = sklearn.feature_extraction.text.CountVectorizer(
        analyzer=»word»,
        max_df=0.95, 
        min_df=10, 
        max_features=number_features
    )
    clean_vec1 = vectorizer1.fit_transform(clean_sentences)
    print(clean_vec1[0])
    feature_names_vec1 = vectorizer1.get_feature_names()
    ```

    输出如下：

    ```py
    (0, 320)    1
    ```

1.  确定最佳主题数：

    ```py
    def perplexity_by_ntopic(data, ntopics):
        output_dict = {
            «Number Of Topics": [], 
            «Perplexity Score»: []
        }
        for t in ntopics:
            lda = sklearn.decomposition.LatentDirichletAllocation(
                n_components=t,
                learning_method="online",
                random_state=0
            )
            lda.fit(data)
            output_dict["Number Of Topics"].append(t)
            output_dict["Perplexity Score"].append(lda.perplexity(data))
        output_df = pandas.DataFrame(output_dict)
        index_min_perplexity = output_df["Perplexity Score"].idxmin()
        output_num_topics = output_df.loc[
            index_min_perplexity,  # index
            «Number Of Topics"  # column
        ]
        return (output_df, output_num_topics)
    df_perplexity, optimal_num_topics = perplexity_by_ntopic(
        clean_vec1, 
        ntopics=[i for i in range(1, 21) if i % 2 == 0]
    )
    print(df_perplexity)
    ```

    输出如下：

    ![图 7.58：主题数与困惑分数数据框架    ](img/C12626_07_58.jpg)

    ###### 图 7.58：主题数与困惑分数数据框架

1.  使用最佳主题数拟合 LDA 模型：

    ```py
    lda = sklearn.decomposition.LatentDirichletAllocation(
        n_components=optimal_num_topics,
        learning_method="online",
        random_state=0
    )
    lda.fit(clean_vec1)
    ```

    输出如下：

    ![图 7.59：LDA 模型    ](img/C12626_07_59.jpg)

    ###### 图 7.59：LDA 模型

1.  创建并打印词-主题表：

    ```py
    def get_topics(mod, vec, names, docs, ndocs, nwords):
        # word to topic matrix
        W = mod.components_
        W_norm = W / W.sum(axis=1)[:, numpy.newaxis]
        # topic to document matrix
        H = mod.transform(vec)
        W_dict = {}
        H_dict = {}
        for tpc_idx, tpc_val in enumerate(W_norm):
            topic = «Topic{}".format(tpc_idx)
            # formatting w
            W_indices = tpc_val.argsort()[::-1][:nwords]
            W_names_values = [
                (round(tpc_val[j], 4), names[j]) 
                for j in W_indices
            ]
            W_dict[topic] = W_names_values
            # formatting h
            H_indices = H[:, tpc_idx].argsort()[::-1][:ndocs]
            H_names_values = [
            (round(H[:, tpc_idx][j], 4), docs[j]) 
                for j in H_indices
            ]
            H_dict[topic] = H_names_values
        W_df = pandas.DataFrame(
            W_dict, 
            index=["Word" + str(i) for i in range(nwords)]
        )
        H_df = pandas.DataFrame(
            H_dict,
            index=["Doc" + str(i) for i in range(ndocs)]
        )
        return (W_df, H_df)
    W_df, H_df = get_topics(
        mod=lda,
        vec=clean_vec1,
        names=feature_names_vec1,
        docs=raw,
        ndocs=number_docs, 
        nwords=number_words
    )
    print(W_df)
    ```

    输出如下：

    ![图 7.60：健康推文数据的词-主题表    ](img/C12626_07_60.jpg)

    ###### 图 7.60：健康推文数据的词-主题表

1.  打印文档-主题表：

    ```py
    print(H_df)
    ```

    输出如下：

    ![图 7.61：文档主题表    ](img/C12626_07_61.jpg)

    ###### 图 7.61：文档主题表

1.  创建双图可视化：

    ```py
    lda_plot = pyLDAvis.sklearn.prepare(lda, clean_vec1, vectorizer1, R=10)
    pyLDAvis.display(lda_plot)
    ```

    ![图 7.62：在健康推文上训练的 LDA 模型的直方图和双图    ](img/C12626_07_62.jpg)

    ###### 图 7.62：在健康推文上训练的 LDA 模型的直方图和双图

1.  保持笔记本打开以供将来建模。

### 活动 17：非负矩阵分解

解决方案：

1.  创建适当的词袋模型，并将特征名作为另一个变量输出：

    ```py
    vectorizer2 = sklearn.feature_extraction.text.TfidfVectorizer(
        analyzer="word",
        max_df=0.5, 
        min_df=20, 
        max_features=number_features,
        smooth_idf=False
    )
    clean_vec2 = vectorizer2.fit_transform(clean_sentences)
    print(clean_vec2[0])
    feature_names_vec2 = vectorizer2.get_feature_names()
    ```

1.  定义并使用活动二中的主题数（`n_components`）值拟合 NMF 算法：

    ```py
    nmf = sklearn.decomposition.NMF(
        n_components=optimal_num_topics,
        init="nndsvda",
        solver="mu",
        beta_loss="frobenius",
        random_state=0, 
        alpha=0.1, 
        l1_ratio=0.5
    )
    nmf.fit(clean_vec2)
    ```

    输出如下：

    ![图 7.63：定义 NMF 模型    ](img/C12626_07_63.jpg)

    ###### 图 7.63：定义 NMF 模型

1.  获取主题-文档和单词-主题的结果表格。花几分钟时间探索单词分组，并尝试定义抽象的主题：

    ```py
    W_df, H_df = get_topics(
        mod=nmf,
        vec=clean_vec2,
        names=feature_names_vec2,
        docs=raw,
        ndocs=number_docs, 
        nwords=number_words
    )
    print(W_df)
    ```

    ![图 7.64：带有概率的单词-主题表    ](img/C12626_07_64.jpg)

    ###### 图 7.64：带有概率的单词-主题表

1.  调整模型参数并重新运行 *步骤 3* 和 *步骤 4*。

## 第八章：市场购物篮分析

### 活动 18：加载和准备完整在线零售数据

解决方案：

1.  加载在线零售数据集文件：

    ```py
    import matplotlib.pyplot as plt
    import mlxtend.frequent_patterns
    import mlxtend.preprocessing
    import numpy
    import pandas
    online = pandas.read_excel(
        io="Online Retail.xlsx", 
        sheet_name="Online Retail", 
        header=0
    )
    ```

1.  清洗并准备数据进行建模，包括将清洗后的数据转化为列表的列表：

    ```py
    online['IsCPresent'] = (
        online['InvoiceNo']
        .astype(str)
        .apply(lambda x: 1 if x.find('C') != -1 else 0)
    )
    online1 = (
        online
        .loc[online["Quantity"] > 0]
        .loc[online['IsCPresent'] != 1]
        .loc[:, ["InvoiceNo", "Description"]]
        .dropna()
    )
    invoice_item_list = []
    for num in list(set(online1.InvoiceNo.tolist())):
        tmp_df = online1.loc[online1['InvoiceNo'] == num]
        tmp_items = tmp_df.Description.tolist()
        invoice_item_list.append(tmp_items)
    ```

1.  对数据进行编码并将其重新构建为 DataFrame：

    ```py
    online_encoder = mlxtend.preprocessing.TransactionEncoder()
    online_encoder_array = online_encoder.fit_transform(invoice_item_list)
    online_encoder_df = pandas.DataFrame(
        online_encoder_array, 
        columns=online_encoder.columns_
    )
    online_encoder_df.loc[
        20125:20135, 
        online_encoder_df.columns.tolist()[100:110]
    ]
    ```

    输出结果如下：

![图 8.35：从完整在线零售数据集中构建的清洗、编码和重新构建的 DataFrame 子集](img/C12626_08_35.jpg)

###### 图 8.35：从完整在线零售数据集中构建的清洗、编码和重新构建的 DataFrame 子集

### 活动 19：在完整在线零售数据集上运行 Apriori 算法

解决方案：

1.  使用合理的参数设置在完整数据上运行 Apriori 算法：

    ```py
    mod_colnames_minsupport = mlxtend.frequent_patterns.apriori(
        online_encoder_df, 
        min_support=0.01,
        use_colnames=True
    )
    mod_colnames_minsupport.loc[0:6]
    ```

    输出结果如下：

    ![图 8.36：使用完整在线零售数据集的 Apriori 算法结果    ](img/C12626_08_36.jpg)

    ###### 图 8.36：使用完整在线零售数据集的 Apriori 算法结果

1.  将结果过滤到包含 `10 COLOUR SPACEBOY PEN` 的物品集。将其支持度值与 *练习 44* 中的支持度值进行比较，*执行 Apriori 算法*：

    ```py
    mod_colnames_minsupport[
        mod_colnames_minsupport['itemsets'] == frozenset(
            {'10 COLOUR SPACEBOY PEN'}
        )
    ]
    ```

    输出结果如下：

    ![图 8.37：包含 10 支 COLOUR SPACEBOY PEN 的物品集结果    ](img/C12626_08_37.jpg)

    ###### 图 8.37：包含 10 支 COLOUR SPACEBOY PEN 的物品集结果

    支持度值确实发生了变化。当数据集扩展到包含所有交易时，该物品集的支持度从 0.015 增加到 0.015793。也就是说，在用于练习的缩小数据集中，这个物品集出现在 1.5% 的交易中，而在完整数据集中，它出现在大约 1.6% 的交易中。

1.  添加一个新的列，包含物品集的长度。然后，过滤出长度为 2 且支持度在 [0.02, 0.021] 范围内的物品集。是否与 *练习 44* 中找到的物品集相同？*执行 Apriori 算法*，*步骤 6*？

    ```py
    mod_colnames_minsupport['length'] = (
        mod_colnames_minsupport['itemsets'].apply(lambda x: len(x))
    )
    mod_colnames_minsupport[
        (mod_colnames_minsupport['length'] == 2) & 
        (mod_colnames_minsupport['support'] >= 0.02) &
        (mod_colnames_minsupport['support'] < 0.021)
    ]
    ```

    ![图 8.38：基于长度和支持度过滤结果的部分    ](img/C12626_08_38.jpg)

    ###### 图 8.38：基于长度和支持度过滤结果的部分

    结果确实发生了变化。在查看具体的物品集及其支持度值之前，我们看到这个经过过滤的 DataFrame 比前一个练习中的 DataFrame 物品集要少。当我们使用完整数据集时，符合过滤标准的物品集较少；也就是说，只有 14 个物品集包含 2 个物品，并且支持度值大于或等于 0.02，且小于 0.021。在前一个练习中，有 17 个物品集符合这些标准。

1.  绘制 `support` 值：

    ```py
    mod_colnames_minsupport.hist("support", grid=False, bins=30)
    plt.title("Support")
    ```

![图 8.39：支持度值的分布](img/C12626_08_27.jpg)

###### 图 8.39：支持度值的分布

该图展示了完整交易数据集中的支持度值分布。正如你可能已经猜测的，分布是右偏的；也就是说，大多数项集的支持度较低，并且在高端范围内有一个长尾。考虑到存在大量独特的项集，单个项集在高比例交易中出现的情况并不令人惊讶。凭借这些信息，我们可以告诉管理层，即便是最突出的项集也仅在大约 10%的交易中出现，而绝大多数项集的出现频率不到 2%。这些结果可能不会支持改变商店布局，但很可能会对定价和折扣策略提供指导。通过公式化一些关联规则，我们可以获得更多有关如何构建这些策略的信息。

### 活动 20：在完整在线零售数据集上查找关联规则

解决方案：

1.  将关联规则模型拟合到完整数据集上。使用置信度指标，最小阈值为 0.6：

    ```py
    rules = mlxtend.frequent_patterns.association_rules(
        mod_colnames_minsupport, 
        metric="confidence",
        min_threshold=0.6, 
        support_only=False
    )
    rules.loc[0:6]
    ```

    输出如下：

    ![图 8.40：基于完整在线零售数据集的关联规则    ](img/C12626_08_40.jpg)

    ###### 图 8.40：基于完整在线零售数据集的关联规则

1.  计算关联规则的数量。这个数量是否与*练习 45*，*推导关联规则*，*步骤 1*中找到的数量不同？

    ```py
    print("Number of Associations: {}".format(rules.shape[0]))
    ```

    存在`498`条关联规则。

1.  绘制置信度与支持度的关系图：

    ```py
    rules.plot.scatter("support", "confidence", alpha=0.5, marker="*")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules")
    plt.show()
    ```

    输出如下：

    ![图 8.41：置信度与支持度的关系图    ](img/C12626_08_41.jpg)

    ###### 图 8.41：置信度与支持度的关系图

    该图揭示了该数据集中一些关联规则，它们具有相对较高的支持度和置信度值。

1.  查看提升值、杠杆值和信念值的分布：

    ```py
    rules.hist("lift", grid=False, bins=30)
    plt.title("Lift")
    ```

    输出如下：

![图 8.42：提升值的分布](img/C12626_08_42.jpg)

###### 图 8.42：提升值的分布

```py
rules.hist("leverage", grid=False, bins=30)
plt.title("Leverage")
```

输出如下：

![图 8.43：杠杆值的分布](img/C12626_08_43.jpg)

###### 图 8.43：杠杆值的分布

```py
plt.hist(
    rules[numpy.isfinite(rules['conviction'])].conviction.values, 
    bins = 30
)
plt.title("Conviction")
```

输出如下：

![图 8.44：信念值的分布](img/C12626_08_44.jpg)

###### 图 8.44：信念值的分布

得出关联规则后，我们可以向管理层提供更多信息，其中最重要的一点是，大约有七个项集在支持度和置信度上都有较高的值。查看支持度与置信度的散点图，看看哪些七个项集与其他项集有所区别。这七个项集的提升值也很高，从提升直方图中可以看出。看来我们已经识别出一些可以采取行动的关联规则，这些规则可以用来推动商业决策。

## 第九章：热点分析

### 活动 21：在一维空间中估算密度

解决方案：

1.  打开一个新笔记本并安装所有必要的库。

    ```py
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    import numpy
    import pandas
    import seaborn
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.neighbors
    seaborn.set()
    ```

1.  从标准正态分布中采样 1,000 个数据点。将 3.5 加到样本的最后 625 个值上（即，索引范围从 375 到 1,000）。为此，使用`numpy.random.RandomState`设置随机状态为 100，以保证采样值一致，然后使用`randn(1000)`调用随机生成数据点：

    ```py
    rand = numpy.random.RandomState(100)
    vals = rand.randn(1000)  # standard normal
    vals[375:] += 3.5
    ```

1.  将 1,000 个数据点的样本数据绘制为直方图，并在其下方添加散点图：

    ```py
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(vals, bins=50, density=True, label='Sampled Values')
    ax.plot(vals, -0.005 - 0.01 * numpy.random.random(len(vals)), '+k', label='Individual Points')
    ax.legend(loc='upper right')
    ```

    输出如下：

    ![图 9.29：随机样本的直方图，下方是散点图    ](img/C12626_09_29.jpg)

    ###### 图 9.29：随机样本的直方图，下方是散点图

1.  定义一组带宽值。然后，定义并拟合一个网格搜索交叉验证算法：

    ```py
    bandwidths = 10 ** numpy.linspace(-1, 1, 100)
    grid = sklearn.model_selection.GridSearchCV(
        estimator=sklearn.neighbors.KernelDensity(kernel="gaussian"),
        param_grid={"bandwidth": bandwidths},
        cv=10
    )
    grid.fit(vals[:, None])
    ```

1.  提取最佳带宽值：

    ```py
    best_bandwidth = grid.best_params_["bandwidth"]
    print(
        "Best Bandwidth Value: {}"
        .format(best_bandwidth)
    )
    ```

1.  重新绘制*步骤 3*中的直方图，并叠加估计的密度：

    ```py
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist(vals, bins=50, density=True, alpha=0.75, label='Sampled Values')
    x_vec = numpy.linspace(-4, 8, 10000)[:, numpy.newaxis]
    log_density = numpy.exp(grid.best_estimator_.score_samples(x_vec))
    ax.plot(
         x_vec[:, 0], log_density, 
         '-', linewidth=4, label='Kernel = Gaussian'
    )
    ax.legend(loc='upper right')
    ```

    输出如下：

![图 9.30：随机样本的直方图，叠加了最佳估计的密度](img/C12626_09_30.jpg)

###### 图 9.30：随机样本的直方图，叠加了最佳估计的密度

### 活动 22：分析伦敦的犯罪情况

解决方案：

1.  加载犯罪数据。使用保存下载目录的路径，创建年份-月份标签的列表，使用`read_csv`命令逐个加载单独的文件，然后将这些文件合并在一起：

    ```py
    base_path = (
        "~/Documents/packt/unsupervised-learning-python/"
        "lesson-9-hotspot-models/metro-jul18-dec18/"
        "{yr_mon}/{yr_mon}-metropolitan-street.csv"
    )
    print(base_path)
    yearmon_list = [
        "2018-0" + str(i) if i <= 9 else "2018-" + str(i) 
        for i in range(7, 13)
    ]
    print(yearmon_list)
    data_yearmon_list = []
    for idx, i in enumerate(yearmon_list):
        df = pandas.read_csv(
            base_path.format(yr_mon=i), 
            header=0
        )

        data_yearmon_list.append(df)

        if idx == 0:
            print("Month: {}".format(i))
            print("Dimensions: {}".format(df.shape))
            print("Head:\n{}\n".format(df.head(2)))
    london = pandas.concat(data_yearmon_list)
    ```

    输出如下：

    ![图 9.31：单个犯罪文件的示例    ](img/C12626_09_31.jpg)

    ###### 图 9.31：单个犯罪文件的示例

    该打印信息仅针对加载的第一个文件，即 2018 年 7 月伦敦大都会警察局的犯罪信息。该文件包含近 100,000 条记录。你会注意到数据集中包含大量有趣的信息，但我们将重点关注`Longitude`（经度）、`Latitude`（纬度）、`Month`（月份）和`Crime type`（犯罪类型）。

1.  打印完整（六个月）和合并数据集的诊断信息：

    ```py
    print(
        "Dimensions - Full Data:\n{}\n"
        .format(london.shape)
    )
    print(
        "Unique Months - Full Data:\n{}\n"
        .format(london["Month"].unique())
    )
    print(
        "Number of Unique Crime Types - Full Data:\n{}\n"
        .format(london["Crime type"].nunique())
    )
    print(
        "Unique Crime Types - Full Data:\n{}\n"
        .format(london["Crime type"].unique())
    )
    print(
        "Count Occurrences Of Each Unique Crime Type - Full Type:\n{}\n"
        .format(london["Crime type"].value_counts())
    )
    ```

    输出如下：

    ![图 9.32：完整犯罪数据集的描述信息    ](img/C12626_09_32.jpg)

    ###### 图 9.32：完整犯罪数据集的描述信息

1.  将数据框限制为四个变量（`Longitude`，`Latitude`，`Month`和`Crime type`）：

    ```py
    london_subset = london[["Month", "Longitude", "Latitude", "Crime type"]]
    london_subset.head(5)
    ```

    输出如下：

    ![图 9.33：数据框中犯罪数据的子集，仅保留经度、纬度、月份和犯罪类型列    ](img/C12626_09_33.jpg)

    ###### 图 9.33：数据框中犯罪数据的子集，仅保留经度、纬度、月份和犯罪类型列

1.  使用`seaborn`的`jointplot`函数，拟合并可视化 2018 年 7 月、9 月和 12 月自行车盗窃的三种核密度估计模型：

    ```py
    crime_bicycle_jul = london_subset[
        (london_subset["Crime type"] == "Bicycle theft") & 
        (london_subset["Month"] == "2018-07")
    ]
    seaborn.jointplot("Longitude", "Latitude", crime_bicycle_jul, kind="kde")
    ```

    输出如下：

    ![图 9.34：2018 年 7 月自行车盗窃的联合密度和边际密度估计    ](img/C12626_09_34.jpg)

    ###### 图 9.34：2018 年 7 月自行车盗窃的联合密度和边际密度估计

    ```py
    crime_bicycle_sept = london_subset[
        (london_subset["Crime type"] == "Bicycle theft") & 
        (london_subset["Month"] == "2018-09")
    ]
    seaborn.jointplot("Longitude", "Latitude", crime_bicycle_sept, kind="kde")
    ```

    输出如下：

    ![图 9.35：2018 年 9 月自行车盗窃的联合分布和边际分布估计]

    ](img/C12626_09_35.jpg)

    ###### 图 9.35：2018 年 9 月自行车盗窃的联合分布和边际分布估计

    ```py
    crime_bicycle_dec = london_subset[
        (london_subset["Crime type"] == "Bicycle theft") & 
        (london_subset["Month"] == "2018-12")
    ]
    seaborn.jointplot("Longitude", "Latitude", crime_bicycle_dec, kind="kde")
    ```

    输出结果如下：

    ![图 9.36：2018 年 12 月自行车盗窃的联合分布和边际分布估计]

    ](img/C12626_09_36.jpg)

    ###### 图 9.36：2018 年 12 月自行车盗窃的联合分布和边际分布估计

    从月份到月份，自行车盗窃的密度保持相当稳定。密度之间有细微的差异，这在所预期之内，因为这些估计的密度是基于三个月的样本数据。根据这些结果，警察或犯罪学家应该对预测未来最可能发生自行车盗窃的地点充满信心。

1.  重复*步骤 4*；这次，使用 2018 年 8 月、10 月和 11 月的商店盗窃数据：

    ```py
    crime_shoplift_aug = london_subset[
        (london_subset["Crime type"] == "Shoplifting") & 
        (london_subset["Month"] == "2018-08")
    ]
    seaborn.jointplot("Longitude", "Latitude", crime_shoplift_aug, kind="kde")
    ```

    输出结果如下：

    ![图 9.37：2018 年 8 月商店盗窃事件的联合分布和边际分布估计]

    ](img/C12626_09_37.jpg)

    ###### 图 9.37：2018 年 8 月商店盗窃事件的联合分布和边际分布估计

    ```py
    crime_shoplift_oct = london_subset[
        (london_subset["Crime type"] == "Shoplifting") & 
        (london_subset["Month"] == "2018-10")
    ]
    seaborn.jointplot("Longitude", "Latitude", crime_shoplift_oct, kind="kde")
    ```

    输出结果如下：

    ![图 9.38：2018 年 10 月商店盗窃事件的联合分布和边际分布估计]

    ](img/C12626_09_38.jpg)

    ###### 图 9.38：2018 年 10 月商店盗窃事件的联合分布和边际分布估计

    ```py
    crime_shoplift_nov = london_subset[
        (london_subset["Crime type"] == "Shoplifting") & 
        (london_subset["Month"] == "2018-11")
    ]
    seaborn.jointplot("Longitude", "Latitude", crime_shoplift_nov, kind="kde")
    ```

    输出结果如下：

    ![图 9.39：2018 年 11 月商店盗窃事件的联合分布和边际分布估计]

    ](img/C12626_09_39.jpg)

    ###### 图 9.39：2018 年 11 月商店盗窃事件的联合分布和边际分布估计

    与自行车盗窃的结果类似，商店盗窃的密度在各个月份之间保持相当稳定。2018 年 8 月的密度看起来与其他两个月有所不同；然而，如果你查看经纬度值，你会发现密度非常相似，只是发生了平移和缩放。这是因为可能存在一些离群点，迫使绘图区域变得更大。

1.  重复*步骤 5*；这次使用 2018 年 7 月、10 月和 12 月的入室盗窃数据：

    ```py
    crime_burglary_jul = london_subset[
        (london_subset["Crime type"] == "Burglary") & 
        (london_subset["Month"] == "2018-07")
    ]
    seaborn.jointplot("Longitude", "Latitude", crime_burglary_jul, kind="kde")
    ```

    输出结果如下：

![图 9.40：2018 年 7 月入室盗窃的联合分布和边际分布估计]

](img/C12626_09_40.jpg)

###### 图 9.40：2018 年 7 月入室盗窃的联合分布和边际分布估计

```py
crime_burglary_oct = london_subset[
    (london_subset["Crime type"] == "Burglary") & 
    (london_subset["Month"] == "2018-10")
]
seaborn.jointplot("Longitude", "Latitude", crime_burglary_oct, kind="kde")
```

输出结果如下：

![图 9.41：2018 年 10 月入室盗窃的联合分布和边际分布估计]

](img/C12626_09_41.jpg)

###### 图 9.41：2018 年 10 月入室盗窃的联合分布和边际分布估计

```py
crime_burglary_dec = london_subset[
    (london_subset["Crime type"] == "Burglary") & 
    (london_subset["Month"] == "2018-12")
]
seaborn.jointplot("Longitude", "Latitude", crime_burglary_dec, kind="kde")
```

输出结果如下：

![图 9.42：2018 年 12 月入室盗窃的联合分布和边际分布估计]

](img/C12626_09_42.jpg)

###### 图 9.42：2018 年 12 月入室盗窃的联合分布和边际分布估计

再次，我们可以看到各个月份的分布非常相似。唯一的区别是从七月到十二月，密度似乎变得更加宽广或分散。和往常一样，样本数据中固有的噪声和信息缺失导致了估计密度的微小变化。
