# 7

# 利用主动机器学习的工具和包

在本章中，我们将讨论一系列在主动机器学习中常用的 Python 库、框架和工具。这些资源对于实现各种主动机器学习技术至关重要。本章的内容旨在为不同水平的专家提供信息丰富且实用的内容，从初学者到经验丰富的程序员。目标是提供对我们将要涵盖的工具的坚实基础理解，以便有效地将主动机器学习技术融入您的项目中。

在本章中，重点将放在理解主动机器学习的 Python 包上。我们将使用流行的 Python 库`scikit-learn`和`modAL`。您将了解它们的特性以及它们如何应用于主动机器学习场景。我们还将探索一系列主动机器学习工具。除了本书前几节中涵盖的工具外，本章还将介绍一些额外的主动机器学习工具。每个工具都将提供其特性和潜在应用的概述，帮助您了解它们如何适应不同的主动机器学习环境。

在本章中，我们将讨论以下主题：

+   掌握用于增强主动机器学习的 Python 包

+   熟悉主动机器学习工具

# 技术要求

对于本章的练习，您需要安装以下包：

```py
pip install scikit-learn
pip install modAL-python
```

您将需要以下导入：

```py
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import numpy as np
import random
from modAL.models import ActiveLearner, Committee
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import uncertainty_sampling
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from modAL.disagreement import vote_entropy_sampling
```

# 掌握用于增强主动机器学习的 Python 包

本节提供了两个广受欢迎的 Python 包的全面概述，这些包以其在促进主动机器学习方面的能力而闻名：`scikit-learn`，一个功能丰富且用户友好的库，在机器学习社区中是基础性的，因为它拥有广泛的传统机器学习工具。另一方面，专门为主动机器学习设计的`modAL`，建立在`scikit-learn`的强大框架之上，引入了更动态、数据高效的机器学习技术。这两个包共同代表了一个强大的工具包，供那些希望利用主动机器学习方法优势的人使用。

## scikit-learn

虽然不是专门为主动机器学习设计的，但**scikit-learn**（[`scikit-learn.org/stable/index.html`](https://scikit-learn.org/stable/index.html)）是 Python 机器学习生态系统中的基础包。它提供了一系列算法和工具，这些工具通常与主动机器学习包一起使用——一个用于分类、回归、聚类和降维的算法集合。它还提供了模型评估和数据预处理的工具。

`scikit-learn`通常用作模型开发的基础，并且经常与主动机器学习包集成以进行模型训练和评估。

例如，`scikit-learn`可以通过根据购买行为、人口统计和参与度指标对客户进行聚类，在营销中执行客户细分。`scikit-learn`中流行的 K-means 聚类算法有助于识别用于针对性营销活动的不同客户群体。通过迭代改进聚类模型，可以结合活跃机器学习。例如，营销分析师可以对聚类算法不确定的模糊案例进行标记，随着时间的推移提高模型的准确性。

让我们用一个模拟示例来说明这一点。

首先，我们使用`KMeans`进行初始聚类。我们首先定义一些模拟客户数据（年龄，年收入）：

```py
X = np.array([[34, 20000], [42, 30000], [23, 25000], [32, 45000], 
    [38, 30000]])
```

然后我们使用`KMeans`进行聚类：

```py
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
```

然后，我们预测每个客户的聚类：

```py
clusters = kmeans.predict(X)
```

我们已经根据年龄和年收入将客户分为两个聚类。

接下来，我们设置活跃机器学习部分。假设我们有一个更大的、未标记的客户特征数据集`X_unlabeled`。在我们的客户细分场景中使用`KMeans`时，未标记数据将包括具有与我们用于聚类的相同特征（在我们的案例中是年龄和年收入）的客户记录，但没有分配的聚类标签。这些数据是我们将在活跃机器学习框架中应用和改进聚类和分类模型的数据：

```py
X_unlabeled = np.array([[28, 22000], [45, 55000], [37, 35000], 
    [50, 48000], [29, 27000], [41, 32000]])
```

我们需要一个模型（一个分类器）来对这个未标记的数据进行预测。为了说明，让我们使用一个简单的分类器`LogisticRegression`。我们初始化这个分类器，并使用聚类作为标签，在我们的初始数据集（`X`）上对其进行训练：

```py
classifier = LogisticRegression()
classifier.fit(X, clusters)
```

然后我们实现活跃机器学习循环。在每次迭代中，分类器对未标记数据进行标签预测。首先，我们需要创建一个`obtain_labels`占位符函数，用于获取所选数据点的真实标签。在实际场景中，此函数将涉及获取实际标签的过程，例如进行调查或专家分析。由于我们正在创建一个模拟示例，我们设计这个函数以基于某些假设逻辑随机分配标签：

```py
def obtain_labels(data):
    return np.random.choice([0, 1], size=len(data))
```

对于我们的活跃机器学习循环，我们需要选择我们想要进行的迭代次数以及每次迭代中要标记的样本数量：

```py
num_iterations = 10
num_to_label = 2
```

我们现在可以创建我们的活跃机器学习循环，它将执行以下操作：

1.  选择分类器最不自信的实例。

1.  获取这些实例的真实标签（在实际操作中，这可能涉及手动标记或额外数据收集）。

1.  使用这些新标签更新分类器。

1.  定期使用新标记的数据更新`KMeans`模型，以细化客户细分。

以下是一个代码片段，帮助我们实现这一点：

```py
for iteration in range(num_iterations):
    if len(X_unlabeled) == 0:
        break  # No more data to label
    # Predict on unlabeled data
    predictions = classifier.predict_proba(X_unlabeled)
    uncertainty = np.max(predictions, axis=1)
    # Select num_to_label instances with least confidence
    uncertain_indices = np.argsort(uncertainty)[:num_to_label]
    # Obtain labels for these instances
    new_labels = obtain_labels(X_unlabeled[uncertain_indices])
    # Update our dataset
    X = np.vstack([X, X_unlabeled[uncertain_indices]])
    clusters = np.hstack([clusters, new_labels])
    # Re-train classifier and KMeans
    classifier.fit(X, clusters)
    kmeans.fit(X)
    print(f"Iteration {iteration+1}, Labeled Data: {
        X_unlabeled[uncertain_indices]} with Labels: {new_labels}")
    # Remove labeled instances from unlabeled data
    X_unlabeled = np.delete(X_unlabeled, uncertain_indices, axis=0)
    # Shuffle unlabeled data to avoid any order bias
    X_unlabeled = shuffle(X_unlabeled)
```

上述代码返回以下结果：

```py
Iteration 1, Labeled Data: [[45 55000] [29 27000]] with Labels: [0 1]
Iteration 2, Labeled Data: [[37 35000] [28 22000]] with Labels: [1 1]
Iteration 3, Labeled Data: [[41 32000] [50 48000]] with Labels: [0 0]
```

我们的活动机器学习循环迭代指定次数，每次选择分类器做出的最不自信的预测，为这些实例获取标签，然后使用新数据更新分类器和`KMeans`模型。记住，`obtain_labels`函数是一种简化。在实际应用中，获取标签将涉及一个预言者手动标记样本，正如我们在*第三章*中描述的，*在循环中管理人类*。

## modAL

`scikit-learn`。它允许轻松地将活动学习策略集成到现有的机器学习工作流程中。它提供了各种活动机器学习策略，如不确定性采样和委员会查询。它还支持自定义查询策略，并易于与`scikit-learn`模型集成。

例如，它非常适合图像分类和回归等任务，在这些任务中，活动机器学习可以有效地选择用于注释的信息性样本。

让我们看看一个例子，其中我们使用流行的`torchvision`数据集对图像进行分类。鉴于图像的大量，活动机器学习可以帮助优先考虑哪些图像应该手动标记。我们将使用`modAL`框架的不确定性采样查询策略。它将能够识别最有信息量的图像（分类器最不确定的图像）并对其进行标记查询。

我们实现了一个`load_images`函数来从数据集目录中读取图像，然后我们将它们转换为灰度图并展平图像以进行训练。实际上，我们需要将图像数据转换为与`RandomForest`兼容的格式，因此每个图像（一个二维数组）被*展平*成一个一维数组。这意味着将图像转换成一个像素值的长时间向量。对于我们的 32x32 像素的灰度图像，展平形式将是一个包含 1,024 个元素的向量（32x32）：

```py
def load_data():
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])
        # Load the CIFAR10 dataset
    dataset = CIFAR10(root='data', train=True, download=True, 
        transform=transform)
    # Load all data into memory (for small datasets)
    dataloader = DataLoader(dataset, batch_size=len(dataset), 
        shuffle=False)
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    # Convert images and labels to numpy arrays
    X_all = images.numpy()
    y_all = np.array(labels)
    # Convert images from 3D to 1D (batch_size, 3, 32, 32) -> (batch_size, 3072) for RandomForest
    X_all = X_all.reshape(X_all.shape[0], -1)
    # Map numerical labels to string labels
    class_names = dataset.classes
    y_all = np.array([class_names[label] for label in y_all])
    return X_all, y_all
```

接下来，为了我们的示例，我们将数据集分为初始标记数据（图像存储在`X_initial`中，标签存储在`y_initial`中）和未标记数据（`X_unlabeled`）：

```py
X_initial, X_unlabeled, y_initial, _ = train_test_split(X_all, y_all, 
    test_size=0.75, random_state=42)
```

我们以 12,500 个标记图像和 37,500 个未标记图像开始我们的示例。

然后我们初始化`modAL`活动学习器：

```py
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=uncertainty_sampling,
    X_training=X_initial_flat, y_training=y_initial
)
ActiveLearner object is created. This learner uses RandomForestClassifier as its estimator. RandomForest is a popular ensemble learning method for classification, which operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees. The query strategy is set to X_initial_flat initial training data and y_training labels are provided to the learner.
```

最后，我们使用以下五次迭代循环模拟标签的查询：

```py
for i in range(5):
    query_idx, _ = learner.query(X_unlabeled)
    actual_label = y_all[query_idx[0]] 
    print(f"Selected unlabeled query is sample number {query_idx[0]}. Actual label: {actual_label}")
    learner.teach(X_unlabeled[query_idx].reshape(1, -1), actual_label.reshape(1,))
    X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
    y_all = np.delete(y_all, query_idx)
```

前面的循环返回以下内容：

```py
Selected unlabeled query is sample number 3100\. Actual label: cat
Selected unlabeled query is sample number 7393\. Actual label: deer
Selected unlabeled query is sample number 4728\. Actual label: horse
Selected unlabeled query is sample number 447\. Actual label: deer
Selected unlabeled query is sample number 17968\. Actual label: bird
```

循环代表五次活动机器学习迭代。在每次迭代中，模型查询数据集以标记新实例。学习者查询`X_unlabeled`训练数据，并返回它最不确定的样本的`query_idx`索引和`query_instance`实例。然后，使用查询的实例对学习者进行教学。在现实世界场景中，这一步将涉及从预言者（如人类注释员）那里获取查询实例的标签。然而，在这个模拟示例中，标签直接从`y_all`数据集中获取。

这个例子说明了使用 `modAL` 进行主动机器学习的过程，其中模型主动查询特定的实例进行学习，而不是从静态数据集中被动学习。

`modAL` 是一个优秀的 Python 包，它允许我们轻松实现复杂的主动机器学习方法。例如，让我们使用 `modAL` 包创建一个主动机器学习的用例，特别关注 **基于委员会** 的算法。在这个例子中，我们将使用一个分类器委员会从未标记数据集中查询最有信息量的样本。作为提醒，我们在 *第二章**，设计查询策略框架* 中定义了 *委员会查询方法*。

在这个例子中，让我们使用 Iris 数据集（*Fisher, R. A.. (1988). Iris. UCI Machine Learning Repository* [`doi.org/10.24432/C56C76`](https://doi.org/10.24432/C56C76)），这是分类任务中常用的选择。Iris 数据集是机器学习和统计学中的一个经典数据集，常用于演示分类算法。该数据集包含 150 个鸢尾花样本。每个样本有四个特征：花瓣长度、花瓣宽度、花萼长度和花萼宽度。这些特征是鸢尾花植物相应部分的厘米测量值。数据集中有三种（类别）鸢尾花植物：Iris setosa、Iris virginica 和 Iris versicolor。每个类别有 50 个样本，使得数据集在三个类别之间均衡。使用 Iris 数据集的典型任务是多类分类问题。目标是根据花萼和花瓣的测量值预测鸢尾花植物的物种。

我们将使用一个 K-Nearest Neighbors 分类器委员会。该委员会将使用 **委员会查询（QBC）策略** 来选择意见分歧最大的数据点。

我们首先加载 Iris 数据集（来自 `scikit-learn` 提供的数据集），并创建一个初始的小型标记数据集和一个较大的未标记数据集：

```py
X, y = load_iris(return_X_y=True)
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X, y, test_size=0.9, random_state=42)
```

我们初始化了二十个 `ActiveLearner` 实例，每个实例都使用一个 `RandomForestClassifier`，并将它们组合成一个 `Committee`：

```py
n_learners = 20
learners = [ActiveLearner(
        estimator=RandomForestClassifier(), X_training=X_labeled, \
        y_training=y_labeled
    ) for _ in range(n_learners)]
committee = Committee(learner_list=learners, 
    query_strategy=vote_entropy_sampling)
```

主动机器学习循环使用 `vote_entropy_sampling` 策略来选择委员会成员意见分歧最大的样本。

下面是我们的主动机器学习循环在五次迭代中的样子：

```py
n_queries = 5
for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_unlabeled)
    print(f"\nSelected unlabeled query is sample number {query_idx}. We simulate labeling this sample which is labeled as: {y_unlabeled[query_idx]}")
    committee.teach(X_unlabeled[query_idx], y_unlabeled[query_idx])
    # Remove the queried instance from the pool
    X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
    y_unlabeled = np.delete(y_unlabeled, query_idx)
    print(f"Number of unlabeled samples is {len(X_unlabeled)}")
    # Calculate and print committee score
    committee_score = committee.score(X, y)
    print(f"Iteration {idx+1}, Committee Score: {committee_score}")
```

`Committee` 对象的 `query` 方法用于从未标记的 `X_unlabeled` 数据集中选择最有信息量的样本。该委员会由多个学习者组成，使用其内部的查询策略 `vote_entropy_sampling` 来确定在 `X_unlabeled` 中它认为对学习最有价值的实例。

每次迭代中选择的样本用于教导（重新训练）所有委员会的学习者。每次查询后，评估委员会的性能：

```py
Selected unlabeled query is sample number [8]. We simulate labeling this sample which is labeled as: [0]
Number of unlabeled samples is 129
Iteration 1, Committee Score: 0.96
Selected unlabeled query is sample number [125]. We simulate labeling this sample which is labeled as: [2]
Number of unlabeled samples is 128
Iteration 2, Committee Score: 0.9466666666666667
Selected unlabeled query is sample number [42]. We simulate labeling this sample which is labeled as: [2]
Number of unlabeled samples is 127
Iteration 3, Committee Score: 0.9466666666666667
Selected unlabeled query is sample number [47]. We simulate labeling this sample which is labeled as: [1]
Number of unlabeled samples is 126
Iteration 4, Committee Score: 0.9733333333333334
Selected unlabeled query is sample number [95]. We simulate labeling this sample which is labeled as: [1]
Number of unlabeled samples is 125
Iteration 5, Committee Score: 0.9733333333333334
```

这个例子演示了如何使用`modAL`中的学习者委员会通过查询最有信息量的样本来主动提高模型性能。委员会的多样化意见有助于选择对学习更有信息量的样本，从而更有效地提高整体模型。例如，我们在输出中观察到，委员会的分数从 0.96 提高到 0.973。

在主动机器学习中，特别是在使用如前例所示的基于委员会的方法时，通常期望模型的性能（或在这种情况下，模型的委员会）会随着迭代而提高。这种改进是预期的，因为委员会正在被训练以越来越有信息量的样本，这些样本是根据委员会的不确定性或分歧选择的。

然而，有几个要点值得注意：

+   **增量改进**：性能的提高可能不是线性的或在整个迭代过程中都是一致的。在某些迭代中，模型可能显著改进，而在其他迭代中，改进可能很小甚至停滞。我们可以从我们的例子中看到这一点，其中委员会的分数从 0.96 降至 0.94，然后又回升到 0.973。

+   **取决于数据和模型**：改进的速度和一致性取决于数据的性质和学习算法的有效性。对于某些数据集或配置，改进可能快速且一致，而对于其他数据集，改进可能较慢或不可预测。

+   **递减回报**：随着最有信息量的样本被添加到训练集中，剩余的无标签样本可能变得不那么有信息量，导致在后续迭代中性能改进的递减回报。

+   `modAL`函数默认使用准确率（accuracy）作为衡量委员会综合预测与真实标签一致性的指标。随着委员会接触到更多数据的有代表性样本，其预测应该变得更加准确。

+   **评估方法**：评估委员会表现的方法也可能影响感知到的改进。如果评估是在静态测试集上进行的，改进可能更为明显。然而，如果评估是在训练集上进行的（包括新添加的样本），由于数据复杂性的增加或方差的变化，改进可能不那么明显。

总结来说，虽然在一个迭代周期内委员会性能分数的增加在主动机器学习中是一个常见的期望，但实际的改进模式可能因各种因素而异。定期监控和调整可能有必要，以确保主动机器学习过程产生预期的结果，正如我们在*第六章*，*评估和* *提高效率*中看到的。

选择合适的 Python 包进行主动机器学习取决于手头任务的特定要求，包括数据类型、所使用的机器学习模型和期望的主动学习策略。有效地整合这些包可以提高数据标注的效率、加快模型收敛速度，并整体提高机器学习模型的表现。

接下来，我们将探讨可以轻松用于对未标记数据进行主动机器学习的工具，如 Encord Active、Lightly、Cleanlab、Voxel51 和 UBIAI。

# 熟悉主动机器学习工具

在整本书中，我们介绍了讨论了几个关键的主动机器学习工具和标注平台，包括 Lightly、Encord、LabelBox、Snorkel AI、Prodigy、`modAL`和 Roboflow。为了进一步加深你的理解并帮助你选择最适合你特定项目需求的最合适的工具，让我们带着更深入的见解重新审视这些工具，并介绍一些额外的工具：

+   `scikit-learn`。它因其广泛的查询策略而脱颖而出，这些策略可以根据不同的主动机器学习场景进行定制。无论你是在处理分类、回归还是聚类任务，`modAL`都提供了一个强大且直观的界面，用于实现主动学习工作流程。

+   **Label Studio** ([`docs.humansignal.com/guide/active_learning.html?__hstc=90244869.a32555b92661e36e5f4b3b8a0f2cc99a.1706210819596.1706210819596.1706210819596.1&__hssc=90244869.2.1706210819596&__hsfp=3755259113&_gl=1*1i1r2ib*_ga*MTE1NzM0NDQ4Ny4xNzA2MjEwODE5*_ga_NQELN45JRH*MTcwNjIxMDgxOS4xLjEuMTcwNjIxMDgzNC4wLjAuMA`](https://docs.humansignal.com/guide/active_learning.html?__hstc=90244869.a32555b92661e36e5f4b3b8a0f2cc99a.1706210819596.1706210819596.1706210819596.1&__hssc=90244869.2.1706210819596&__hsfp=3755259113&_gl=1*1i1r2ib*_ga*MTE1NzM0NDQ4Ny4xNzA2MjEwODE5*_ga_NQELN45JRH*MTcwNjIxMDgxOS4xLjEuMTcwNjIxMDgzNC4wLjAuMA)): 一个开源的多类型数据标注工具，Label Studio 在适应不同形式的数据方面表现出色，包括文本、图像和音频。它允许将机器学习模型集成到标注过程中，从而通过主动机器学习提高标注效率。其灵活性扩展到可定制的标注界面，使其适用于数据标注的广泛应用。

+   **Prodigy** ([`prodi.gy/`](https://prodi.gy/))：Prodigy 提供了一种独特的主动机器学习和人机交互方法的结合。它是一个高效的标注工具，尤其适用于精炼 NLP 模型的训练数据。它的实时反馈循环允许快速迭代和模型改进，使其成为需要快速适应和精确数据标注的项目的一个理想选择。

+   **Lightly**([`www.lightly.ai/`](https://www.lightly.ai/)): 专注于图像数据集，Lightly 使用主动机器学习来识别训练中最具代表性和多样化的图像集。这确保了模型在平衡和多样化的数据集上训练，从而提高了泛化能力和性能。Lightly 对于数据丰富但标注资源有限的项目特别有用。

+   **Encord Active** ([`encord.com/active`](https://encord.com/active)): Encord Active 专注于图像和视频数据的主动机器学习，集成在一个全面的标注平台中。它通过识别和优先处理最有信息量的样本来简化标注流程，从而提高效率并减少人工标注的工作量。这个平台对于大规模计算机视觉项目特别有益。

+   **Cleanlab** ([`cleanlab.ai/`](https://cleanlab.ai/)): Cleanlab 以其在数据集中检测、量化并纠正标签错误的能力而脱颖而出。这种能力对于主动机器学习至关重要，因为标注数据的质量直接影响模型性能。它提供了一种确保数据完整性的系统方法，这对于训练稳健和可靠的模型至关重要。

+   **Voxel51** ([`voxel51.com/blog/supercharge-your-annotation-workflow-with-active-learning`](https://voxel51.com/blog/supercharge-your-annotation-workflow-with-active-learning)): 专注于视频和图像数据，Voxel51 提供了一个优先处理标注最有信息量数据的主动机器学习平台。这增强了标注工作流程，使其更加高效和有效。该平台特别擅长处理复杂的大型视频数据集，提供强大的视频分析和机器学习工具。

+   **UBIAI** ([`ubiai.tools/active-learning-2`](https://ubiai.tools/active-learning-2)): UBIAI 是一个专注于文本标注并支持主动机器学习的工具。它通过简化标注工作流程来简化训练和部署 NLP 模型的过程。其主动机器学习能力确保最有信息量的文本样本被优先标注，从而在更少的标注示例中提高模型准确性。

+   **Snorkel AI** ([`snorkel.ai`](https://snorkel.ai)): 以其创建、建模和管理训练数据的创新方法而闻名，Snorkel AI 使用一种称为弱监督的技术。这种方法结合了各种标注来源，以减少对大量标注数据集的依赖，补充主动机器学习策略以创建高效的训练数据管道。

+   **Deepchecks** ([`deepchecks.com/importance-of-active-learning-in-machine-learning`](https://deepchecks.com/importance-of-active-learning-in-machine-learning)): Deepchecks 提供了一套在主动机器学习环境中至关重要的验证检查。这些检查确保了数据集和模型的质量和多样性，从而促进了更准确、更稳健的机器学习系统的开发。它是维护数据完整性和模型可靠性在整个机器学习生命周期中的必备工具。

+   **LabelBox** ([`labelbox.com/guides/the-guide-to-getting-started-with-active-learning`](https://labelbox.com/guides/the-guide-to-getting-started-with-active-learning)): 作为一款全面的数据标注平台，LabelBox 在管理整个数据标注流程方面表现出色。它提供了一套创建、管理和迭代标注数据的工具，适用于各种数据类型，如图像、视频和文本。其对主动学习方法的支撑进一步提高了标注过程的效率，使其成为大规模机器学习项目的理想选择。

+   **Roboflow** ([`docs.roboflow.com/api-reference/active-learning`](https://docs.roboflow.com/api-reference/active-learning)): 专为计算机视觉项目设计，Roboflow 简化了图像数据准备的过程。它对于涉及图像识别和目标检测的任务尤其有价值。Roboflow 专注于简化图像数据准备、标注和管理，使其成为计算机视觉领域团队和个人不可或缺的资源。

此扩展列表中的每个工具都为机器学习项目带来了独特的功能，解决了特定挑战。从图像和视频标注到文本处理和数据完整性检查，这些工具提供了通过主动机器学习策略提高项目效率和效果所必需的功能。

# 摘要

本章全面探讨了各种对主动机器学习至关重要的 Python 库、框架和工具。通过深入了解流行的库如`scikit-learn`和`modAL`的复杂性，我们探讨了它们的性能以及如何在主动机器学习场景中有效地应用。此外，本章还通过介绍一系列其他主动机器学习工具，每个工具都有其独特的特性和潜在应用，扩展了您的工具箱。

不论你是刚开始在主动机器学习领域迈出第一步的新手，还是寻求提升技能的经验丰富的程序员，本章旨在为你提供主动机器学习工具和技术的坚实基础。在这里获得的知识不仅仅是理论性的；它是一份实用指南，帮助你掌握增强主动机器学习的 Python 包，并熟悉广泛的主动机器学习工具。这种理解将使你能够为特定的机器学习项目选择并应用最合适的工具，从而提高模型的高效性和有效性。

恭喜！你已经到达了本书的结尾。但请记住，你只是在主动机器学习的世界中开始了你的旅程。随着你在机器学习旅程中不断前进，请记住，主动机器学习的领域正在不断演变。了解新的发展、工具和技术将是保持你在工作中保持前沿方法的关键。本章涵盖的工具和概念为在令人兴奋且动态的主动机器学习领域中进一步探索和创新提供了坚实的基础。
