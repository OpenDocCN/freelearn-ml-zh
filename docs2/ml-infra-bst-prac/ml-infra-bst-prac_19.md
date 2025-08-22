

# 第十五章：机器学习系统的伦理

伦理涉及数据获取和管理，重点是收集数据，特别关注保护个人和组织免受可能对他们造成的任何伤害。然而，数据并不是机器学习（ML）系统中偏见的唯一来源。

算法和数据处理方式也容易引入数据偏见。尽管我们尽了最大努力，数据处理的一些步骤甚至可能强调偏见，使其超出算法范围，向基于机器学习的系统其他部分扩散，例如用户界面或决策组件。

因此，在本章中，我们将关注机器学习系统中的偏见。我们将首先探讨偏见的来源，并简要讨论这些来源。然后，我们将探讨发现偏见的方法、如何最小化偏见，以及最后如何向我们的系统用户传达潜在的偏见。

在本章中，我们将涵盖以下主要主题：

+   偏见与机器学习 – 是否可能拥有一个客观的人工智能？

+   测量和监控偏见

+   减少偏见

+   制定机制以防止机器学习偏见在整个系统中扩散

# 偏见与机器学习 – 是否可能拥有一个客观的人工智能？

在机器学习和软件工程的交织领域中，数据驱动决策和预测建模的吸引力无可否认。这些曾经主要在孤岛中运作的领域，现在在众多应用中汇聚，从软件开发工具到自动化测试框架。然而，随着我们越来越依赖数据和算法，一个紧迫的问题出现了：偏见问题。在这个背景下，偏见指的是在机器学习模型的决策和预测中表现出的系统性和不公平的差异，通常源于软件工程过程中的数据。

软件工程数据中偏见的来源是多方面的。它们可能源于历史项目数据、用户反馈循环，甚至软件本身的设计和目标。例如，如果一个软件工具主要使用特定人群的反馈进行测试和改进，它可能会无意中在那些群体之外的用户中表现不佳或行为不当。同样，如果训练数据来自缺乏团队构成或编码实践多样性的项目，缺陷预测模型可能会出现偏差。

这种偏见的后果不仅限于技术上的不准确。它们可能导致软件产品使某些用户群体感到疏远或处于不利地位，从而持续和放大现有的社会不平等。例如，一个开发环境可能对某一文化背景的提议比对另一文化背景的提议更响亮，或者一个软件推荐系统可能会偏向于知名开发者的应用程序，而忽视新来者。

通常，偏差被定义为对某个人或群体的倾向或偏见。在机器学习中，偏差是指模型系统地产生有偏见的结果。机器学习中存在几种类型的偏差：

+   **偏见偏差**：这是一种存在于经验世界中并进入机器学习模型和算法中的偏差——无论是故意还是无意。一个例子是种族偏见或性别偏见。

+   **测量偏差**：这是一种通过我们测量工具中的系统性错误引入的偏差。例如，我们通过计算 if/for 语句来衡量软件模块的 McCabe 复杂性，而排除 while 循环。

+   **采样偏差**：这是一种当我们的样本不能反映数据的真实分布时出现的偏差。可能的情况是我们从特定类别中采样过于频繁或过于稀少——这种偏差会影响推理。

+   **算法偏差**：这是一种在我们使用错误的算法来完成手头任务时出现的偏差。一个错误的算法可能无法很好地泛化，因此它可能会在推理中引入偏差。

+   **确认偏差**：这是一种在我们移除/选择与我们要捕捉的理论概念一致的数据点时引入的偏差。通过这样做，我们引入了证实我们理论的偏差，而不是反映经验世界。

这个列表绝不是排他的。偏差可以通过许多方式以多种方式引入，但始终是我们的责任去识别它、监控它并减少它。

幸运的是，有一些框架可以让我们识别偏差——公平机器学习、IBM AI 公平 360 和微软 Fairlearn，仅举几个例子。这些框架允许我们仔细审查我们的算法和数据集，以寻找最常见的偏差。

Donald 等人最近概述了减少软件工程中偏差的方法和工具，包括机器学习。那篇文章的重要部分是它侧重于用例，这对于理解偏差很重要；偏差不是普遍存在的，而是取决于数据集和该数据的使用案例。除了之前提出的偏差来源外，他们还认识到偏差是随着时间的推移而变化的，就像我们的社会变化和我们的数据变化一样。尽管 Donald 等人的工作具有普遍性，但它倾向于关注一种数据类型——自然语言——以及偏差可能存在的方式。他们概述了可以帮助识别诸如仇恨言论等现象的工具和技术。

在本章中，然而，我们将关注一个稍微更通用的框架，以说明如何一般性地处理偏差问题。

# 测量和监控偏差

让我们看看这些框架中的一个——IBM AI 公平性 360（[`github.com/Trusted-AI/AIF360`](https://github.com/Trusted-AI/AIF360)）。这个框架的基础是能够设置可以与偏见相关联的变量，然后计算其他变量之间的差异。所以，让我们深入一个如何计算数据集偏见的例子。由于偏见通常与性别或类似属性相关联，我们需要使用包含这种属性的数据集。到目前为止，在这本书中，我们还没有使用过包含这种属性的数据集，因此我们需要找到另一个。

让我们以泰坦尼克号生存数据集来检查男性和女性乘客在生存方面的偏见。首先，我们需要安装 IBM AI 公平性 360 框架：

```py
pip install aif360
```

然后，我们可以开始创建一个检查偏见的程序。我们需要导入适当的库并创建数据。在这个例子中，我们将创建薪资数据，该数据倾向于男性：

```py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
t i
data = {
    'Age': [25, 45, 35, 50, 23, 30, 40, 28, 38, 48, 27, 37, 47, 26, 36, 46],
    'Income': [50000, 100000, 75000, 120000, 45000, 55000, 95000, 65000, 85000, 110000, 48000, 58000, 98000, 68000, 88000, 105000],
    'Gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],  # 1: Male, 0: Female
    'Hired': [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]   # 1: Hired, 0: Not Hired
}
df = pd.DataFrame(data)
```

这份数据包含四个不同的属性——年龄、收入、性别以及是否建议雇佣这个人。很难发现性别之间是否存在偏见，但让我们应用 IBM 公平性算法来检查这一点：

```py
# Split data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
# Convert dataframes into BinaryLabelDataset format
train_bld = BinaryLabelDataset(df=train, label_names=['Hired'], protected_attribute_names=['Gender'])
test_bld = BinaryLabelDataset(df=test, label_names=['Hired'], protected_attribute_names=['Gender'])
# Compute fairness metric on original training dataset
metric_train_bld = BinaryLabelDatasetMetric(train_bld, unprivileged_groups=[{'Gender': 1}], privileged_groups=[{'Gender': 0}])
print(f'Original training dataset disparity: {metric_train_bld.mean_difference():.2f}')
# Mitigate bias by reweighing the dataset
RW = Reweighing(unprivileged_groups=[{'Gender': 1}], privileged_groups=[{'Gender': 0}])
train_bld_transformed = RW.fit_transform(train_bld)
# Compute fairness metric on transformed training dataset
metric_train_bld_transformed = BinaryLabelDatasetMetric(train_bld_transformed, unprivileged_groups=[{'Gender': 1}], privileged_groups=[{'Gender': 0}])
print(f'Transformed training dataset disparity: {metric_train_bld_transformed.mean_difference():.2f}')
```

上述代码创建了一个数据分割并计算了公平性指标——数据集差异。算法的重要部分在于我们设置了受保护属性——性别（`protected_attribute_names=['Gender']`）。我们手动设置了我们认为可能存在偏见的属性，这是一个重要的观察。公平性框架不会自动设置任何属性。然后，我们设置了该属性的哪些值表示特权组和非特权组——`unprivileged_groups=[{'Gender': 1}]`。一旦代码执行，我们就能了解数据集中是否存在偏见：

```py
Original training dataset disparity: 0.86
Transformed training dataset disparity: 0.50
```

这意味着算法可以减少差异，但并没有完全消除。差异值 0.86 表示对特权组（在这种情况下是男性）存在偏见。值 0.5 表示偏见已经减少，但仍然远未达到 0.0，这会表明没有偏见。偏见减少而没有被消除的事实可能表明数据量太少，无法完全减少偏见。

因此，让我们看看实际的包含偏见的真实数据集——泰坦尼克号数据集。该数据集包含受保护属性，如性别，并且它非常大，这样我们就有更好的机会进一步减少偏见：

```py
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
# Load Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)
```

现在我们已经准备好了数据集，我们可以编写脚本计算差异度量，该度量量化了基于控制变量的数据差异程度：

```py
# Preprocess the data
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})  # Convert 'Sex' to binary: 1 for male, 0 for female
df.drop(['Name'], axis=1, inplace=True)  # Drop the 'Name' column
# Split data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
# Convert dataframes into BinaryLabelDataset format
train_bld = BinaryLabelDataset(df=train, label_names=['Survived'], protected_attribute_names=['Sex'])
test_bld = BinaryLabelDataset(df=test, label_names=['Survived'], protected_attribute_names=['Sex'])
# Compute fairness metric on the original training dataset
metric_train_bld = BinaryLabelDatasetMetric(train_bld, unprivileged_groups=[{'Sex': 0}], privileged_groups=[{'Sex': 1}])
print(f'Original training dataset disparity: {metric_train_bld.mean_difference():.2f}')
# Mitigate bias by reweighing the dataset
RW = Reweighing(unprivileged_groups=[{'Sex': 0}], privileged_groups=[{'Sex': 1}])
train_bld_transformed = RW.fit_transform(train_bld)
# Compute fairness metric on the transformed training dataset
metric_train_bld_transformed = BinaryLabelDatasetMetric(train_bld_transformed, unprivileged_groups=[{'Sex': 0}], privileged_groups=[{'Sex': 1}])
print(f'Transformed training dataset disparity: {metric_train_bld_transformed.mean_difference():.2f}')
```

首先，我们需要将 DataFrame `df` 中的`'Sex'`列转换为二进制格式：男性为`1`，女性为`0`。然后，我们需要从 DataFrame 中删除`'Name'`列，因为它可能会与索引混淆。然后，使用`train_test_split`函数将数据分为训练集和测试集。20%的数据（`test_size=0.2`）保留用于测试，其余用于训练。`random_state=42`确保分割的可重复性。

接下来，我们将训练和测试的 DataFrame 转换为`BinaryLabelDataset`格式，这是公平框架使用的特定格式。目标变量（或标签）是`'Survived'`，受保护的属性（即我们在公平性方面关注的属性）是`'Sex'`。该框架将女性（`'Sex': 0`）视为无特权群体，将男性（`'Sex': 1`）视为特权群体。

`mean_difference`方法计算特权群体和无特权群体之间平均结果的差异。0 值表示完全公平，而非零值表示存在一些差异。然后，代码使用`Reweighing`方法来减轻训练数据集中的偏差。这种方法通过给数据集中的实例分配权重来确保公平性。转换后的数据集（`train_bld_transformed`）具有这些新的权重。然后，我们在转换后的数据集上计算相同的指标。这导致以下输出：

```py
Original training dataset disparity: 0.57
Transformed training dataset disparity: 0.00
```

这意味着算法已经平衡了数据集，使得男性和女性的生存率相同。现在我们可以使用这个数据集来训练一个模型：

```py
# Train a classifier (e.g., logistic regression) on the transformed dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(train_bld_transformed.features)
y_train = train_bld_transformed.labels.ravel()
clf = LogisticRegression().fit(X_train, y_train)
# Test the classifier
X_test = scaler.transform(test_bld.features)
y_test = test_bld.labels.ravel()
y_pred = clf.predict(X_test)
# Evaluate the classifier's performance
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
report = classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"])
print(report)
```

首先，我们初始化`StandardScaler`。这个缩放器通过去除均值并缩放到单位方差来标准化特征。然后，我们使用缩放器的`fit_transform`方法对训练数据集的特征（`train_bld_transformed.features`）进行转换和标准化。标准化的特征存储在`X_train`中。然后，我们使用`ravel()`方法从转换后的训练数据集中提取标签，得到`y_train`。之后，我们使用标准化的特征（`X_train`）和标签（`y_train`）来训练逻辑回归分类器（`clf`）。

然后，我们使用缩放器的转换方法对测试数据集的特征（`test_bld.features`）进行标准化，以获得`X_test`。我们对`y_test`数据也进行同样的操作。我们使用训练好的分类器（`clf`）对标准化的测试特征进行预测，并将结果存储在`y_pred`中。

最后，我们计算数据集的评估分数，并打印包含准确率、精确率和召回率的报告。

这样，我们就来到了关于偏差的最佳实践。

最佳实践 #73

如果数据集包含可能存在偏差的变量，请使用差异指标来快速了解数据。

虽然我们并不总是能够访问用于其计算的变量，例如性别或年龄，检查偏差是很重要的。如果我们没有，我们应该寻找可以与之相关的属性，并检查对这些属性的偏差。

## 其他偏差度量标准

我们迄今为止使用的数据集差异度量标准只是与偏差相关的一些度量标准。IBM AI Fairness 360 框架中可用的其他一些度量标准如下：

+   **真正率**：在受保护属性条件下的真正率的比率。这通常用于分类。

+   **假发现率**：在分类任务中，特权组和未特权组之间假发现率的差异。

+   **通用二元混淆矩阵**：在分类任务中对受保护属性进行混淆矩阵的条件。

+   特权实例与未特权实例之间的比率，可用于各种任务。

除了这些之外，还有一些度量标准，但我们在这里提到的这些度量标准说明了最重要的观点——或者两个观点。首先，我们可以看到需要有一个属性，称为受保护属性，这可以帮助我们理解偏差。没有这样的属性，框架无法进行任何计算，因此它无法为开发者提供任何有用的反馈。第二个观点是，这些度量标准是基于不同群体之间——特权组和未特权组——的不平衡，这是我们自行定义的。我们不能使用这个框架来发现隐藏的偏差。

隐藏的偏差是指没有直接由属性表示的偏差。例如，男性和女性在职业上有差异，因此职业可以是一个与性别相关但不等于性别的属性。这意味着我们不能将其视为受保护属性，但我们需要考虑它——基本上，没有纯粹男性或纯粹女性的职业，但不同的职业有不同的男性和女性的比例。

# 开发机制以防止机器学习偏差在整个系统中传播

不幸的是，通常无法完全从机器学习中去除偏差，因为我们往往无法访问减少偏差所需的属性。然而，我们可以减少偏差并降低偏差传播到整个系统的风险。

意识和教育是我们可以用来管理软件系统偏差的最重要措施之一。我们需要了解偏差的潜在来源及其影响。我们还需要识别与受保护属性（例如，性别）相关的偏差，并确定其他属性是否可以与之相关联（例如，职业和地址）。然后，我们需要教育我们的团队了解偏差模型伦理影响。

然后，我们需要多样化我们的数据收集。我们必须确保我们收集的数据能够代表我们要建模的群体。为了避免过度或不足代表某些群体，我们需要确保在应用之前对数据收集程序进行审查。我们还需要监控收集到的数据中的偏差并减少它们。例如，如果我们发现信用评分中存在偏差，我们可以引入数据，以防止我们的模型加强这种偏差。

在数据预处理期间，我们需要确保我们正确处理缺失数据。而不仅仅是删除数据点或用平均值填充它们，我们应该使用正确的填充方法，这种方法会考虑到特权和不特权群体之间的差异。

我们还需要积极工作于偏差检测。我们应该使用统计测试来检查数据分布是否偏向于某些群体，此时我们需要可视化分布并识别潜在的偏差。我们已经讨论了可视化技术；在这个阶段，我们可以补充说，我们需要为特权和不特权群体使用不同的符号，以便在同一个图表上可视化两个分布，例如。

除了与数据合作外，我们还需要在模型设计时考虑算法公平性。我们需要设置公平性约束，并引入可以帮助我们识别特权和不特权群体的属性。例如，如果我们知道不同的职业对性别存在一定的偏见，我们需要引入表面上的性别偏见属性，以帮助我们创建一个考虑到这一点并防止偏差传播到系统其他部分的模型。我们还可以在训练后对模型进行事后调整。例如，在预测薪水时，我们可以在预测后根据预定义的规则调整那个薪水。这有助于减少模型中固有的偏差。

我们还可以使用公平性增强干预措施，例如 IBM 的公平性工具和技术，包括去偏差、重新加权以及消除不同影响。这可以帮助我们实现可解释的模型，或者允许我们使用模型解释工具来理解决策是如何做出的。这有助于识别和纠正偏差。

最后，我们可以定期审计我们的模型以检查偏差和公平性。这包括自动检查和人工审查。这有助于我们了解是否存在无法自动捕捉的偏差，以及我们需要做出反应的偏差。

有了这些，我们来到了我的下一个最佳实践。

最佳实践 #74

通过定期的审计来补充自动化偏差管理。

我们需要接受数据中固有的偏差这一事实，因此我们需要相应地采取行动。而不是依赖算法来检测偏差，我们需要手动监控偏差并理解它。因此，我建议定期手动检查偏差。进行分类和预测，并通过将它们与无偏差的预期数据进行比较来检查它们是否增强了或减少了偏差。

# 摘要

作为软件工程师，我们的一项责任是确保我们开发的软件系统对社会的大局有益。我们热爱与技术开发打交道，但技术的发展需要负责任地进行。在本章中，我们探讨了机器学习中的偏差概念以及如何与之合作。我们研究了 IBM 公平性框架，该框架可以帮助我们识别偏差。我们还了解到，自动偏差检测过于有限，无法完全从数据中消除偏差。

有更多的框架可以探索，每天都有更多的研究和工具可用。这些框架更加具体，提供了一种捕捉更多特定领域偏差的方法——在医学和广告领域。因此，在本章的最后，我的建议是探索针对当前任务和领域的特定偏差框架。

# 参考文献

+   *Donald, A. 等，客户交互数据偏差检测：关于数据集、方法和工具的调查。IEEE* *Access，2023 年。*

+   *Bellamy, R.K. 等，AI 公平性 360：一个用于检测、理解和缓解不希望算法偏差的可扩展工具包。arXiv 预印本* *arXiv:1810.01943，2018 年。*

+   *Zhang, Y. 等。人工智能公平性简介。在 2020 年 CHI 计算机系统人类因素会议扩展摘要中。2020 年。*

+   *Alves, G. 等。减少机器学习模型在表格和文本数据上的无意偏差。在 2021 年 IEEE 第 8 届数据科学和高级分析会议（DSAA）中。2021 年。IEEE。*

+   *Raza, S.，D.J. Reji 和 C. Ding，Dbias：检测新闻文章中的偏差并确保公平性。国际数据科学和分析杂志，2022 年：* *第 1-21 页。*
