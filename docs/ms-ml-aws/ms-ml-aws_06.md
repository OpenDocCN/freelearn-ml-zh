# 第四章：使用基于树的算法预测用户行为

本章将介绍决策树、随机森林和梯度提升树。决策树方法是一种在数据科学中广泛使用的流行技术，它以可视化的方式展示了训练集中的信息如何表示为一个层次结构。根据观察结果遍历层次结构可以帮助你预测该事件发生的概率。我们将探讨如何使用这些算法来预测用户可能会点击在线广告的时间，基于现有的广告点击记录。此外，我们还将展示如何使用 AWS **弹性映射减少**（**EMR**）与 Apache Spark 以及 SageMaker XGBoost 服务在大数据环境中构建模型。

在本章中，我们将涵盖以下主题：

+   理解决策树

+   理解随机森林算法

+   理解梯度提升算法

+   预测日志流中的点击

# 理解决策树

决策树图形化地展示了将要做出的决策、可能发生的观察事件以及给定一组特定的可观察事件同时发生时的结果概率。决策树作为一种流行的机器学习算法，基于一组可观察事件的数据集和已知的输出结果，我们可以构建一个决策树来表示事件发生的概率。

下表展示了决策树如何生成的一个非常简单的例子：

| **汽车品牌** | **年份** | **价格** |
| --- | --- | --- |
| 宝马 | 2015 | >$40K |
| 宝马 | 2018 | >$40K |
| 本田  | 2015 | <$40K |
| 本田  | 2018 | >$40K |
| 尼桑 | 2015 | <$40K |
| 尼桑 | 2018 | >$40K |

这是一个非常简单的数据集，它由以下决策树表示：

![](img/d6743663-fd91-403f-9a57-775f6fe90608.png)

机器学习算法的目标是生成最能代表数据集中观测值的决策树。对于一个新观测值，如果我们遍历决策树，叶节点代表最有可能发生的类变量或事件。在先前的例子中，我们有一个包含关于二手车品牌和年份的信息的数据集。类变量（也称为**特征标签**）是汽车的价格。我们可以在数据集中观察到，无论年份变量的值如何，宝马汽车的价格都超过 40,000 美元。然而，如果汽车的品牌不是宝马，汽车的成本将由汽车生产的年份决定。这个例子基于非常少量的数据。然而，决策树代表了数据集中的信息，如果我们必须确定品牌为宝马且年份为 2015 的新车的成本，那么我们可以预测其成本将超过 40,000 美元。对于更复杂的决策树，叶节点还与一个概率相关联，该概率代表类值发生的概率。在本章中，我们将研究可以用来生成此类决策树的算法。

# 递归分割

决策树可以通过递归地将数据集划分为子集来构建。在每次分割过程中，我们根据所有输入属性评估分割，并使用成本函数来确定哪个分割的成本最低。成本函数通常评估将数据集分割成两个分支时的信息损失。这种将数据集分割成更小子集的过程也被称为递归分割。分割数据集的成本通常由每个数据集中具有相似类变量的记录如何分组在一起来决定。因此，最优的分割是在每个子集中的观测值将具有相同的类变量值时。

这种决策树的递归分割是一种自上而下的生成方法。这同样是一种贪婪算法，因为我们在每个点上做出了如何划分数据集的决定，而没有考虑它可能对后续分割的影响。

在先前的例子中，我们是根据汽车的品牌进行第一次分割的。这是因为我们的一个子集，其中品牌是宝马，有 100%的概率汽车的价格将超过 40,000 美元。同样，如果我们根据年份进行分割，我们也会得到一个年份等于 2018 的子集，该子集也有 100%的概率汽车的成本将超过 40,000 美元。因此，对于相同的数据集，我们可以生成多个代表数据集的决策树。我们将查看各种成本函数，它们基于相同的数据集生成不同的决策树。

# 决策树的类型

根据数据集中的类变量，大多数数据科学家必须处理以下两种主要的决策树类型：

+   **分类树**：分类树是用于预测离散值的决策树。这意味着用于生成分类树的数据库中的类别变量是离散值。本节开头关于汽车价格的例子就是一个分类树，因为它只有两个类别变量的值。

+   **回归树**：回归树是用于预测实数的决策树，例如在**第三章**中的例子，**使用回归算法预测房价**，我们预测的是房价。

术语**分类与回归树**（**CART**）用于描述生成决策树的算法。CART 是一种流行的决策树算法。其他流行的决策树算法包括 ID3 和 C4.5。这些算法在用于分割数据集的成本函数以及确定何时停止分割的准则方面各不相同。

# 成本函数

如在*递归分割*章节中所述，我们需要成本函数来确定在给定输入变量上分割是否比其他变量更好。这些成本函数的有效性对于构建决策树的质量至关重要。在本节中，我们将讨论生成决策树时两种流行的成本函数。

# Gini 不纯度

Gini 不纯度定义为在随机观测值被基于数据集中类别变量的分布进行分类的情况下，对随机观测值进行错误分类的可能性的度量。考虑一个包含  ![图片](img/1cb79df3-f4d0-4635-a7eb-a36ca0e0d4e7.png) 类别变量的数据集，其中  ![图片](img/2cfcf87b-3030-4cc6-ba1c-c60a62ddbc18.png) 是数据集中标记为  ![图片](img/f7eea272-393c-47d6-9b01-81233bc5a30d.png) 的观测值的比例。*Gini 不纯度*可以使用以下公式计算：

![图片](img/55adff7c-2b06-42b2-8fb0-ad42229aa761.png)  .. 4.1

Gini 不纯度告诉我们数据集中存在的噪声量，这是基于各种类别变量的分布情况。

例如，在*理解决策树*部分开头展示的汽车价格数据集中，我们有两个类别变量：大于 40,000 和小于 40,000。如果我们必须计算数据集的 Gini 不纯度，它可以按照以下公式计算：

![图片](img/31bfe354-0fab-4186-adb4-18ee95eaf77f.png)

因此，基础数据集中存在大量的噪声，因为每个类别变量都有 50% 的观测值。

然而，当我们创建一个以汽车品牌为分支时，该数据集子集的 Gini 不纯度计算如下：

![图片](img/a3bab794-0e25-46aa-a736-3190545ed1fa.png)

![图片](img/658abb2a-deee-4d28-ab16-80ec3f2aba2c.png)

![图片](img/0ce42a6c-fe7d-4249-9883-a8d70c2ec65f.png)

由于宝马分支只包含*>40K*的类值，因此该分支没有噪声，基尼不纯度的值为*0*。请注意，当数据子集只有一个类值时，基尼不纯度的值总是*0*。

基尼不纯度用于计算每个属性的基尼指数。基尼指数是我们创建分支的属性所有值的加权总和。对于具有![图片](img/f6e07e55-9112-4986-8e76-7b75afbfd099.png)个唯一值的属性，基尼增益是使用以下公式计算的。![图片](img/5e9ca5cf-4d11-4848-990b-c1b3ac2e3961.png)是数据集中属性值![图片](img/ea9db92c-195e-4477-be7f-68149a481142.png)为![图片](img/98d43c54-cf8a-464d-9376-503ccb926ee0.png)的观察值的比例：

![图片](img/43d86f3b-a08a-4d4e-b190-a83c7424690d.png)

因此，在我们前面的例子中，对于具有三个不同值的*Make*属性，基尼指数的计算如下：

![图片](img/54b85ded-6b89-4176-b5ee-b331e651c0e6.png)

同样，我们计算其他属性的基尼指数。在我们的例子中，*Year*属性的基尼指数为 0.4422。我们鼓励您自己计算这个值。我们的目标是选择产生最低基尼指数得分的属性。对于完美的分类，即每个分支中的所有类值都相同，基尼指数得分将为 0。

# 信息增益

信息增益基于熵的概念，这在物理学中常用来表示随机变量的不可预测性。例如，如果我们有一个无偏的硬币，硬币的熵表示为*1*，因为它具有最高的不可预测性。然而，如果一个硬币是偏的，并且有 100%的机会出现正面，那么硬币的熵为 0。

熵的概念也可以用来确定给定分支中类变量的不可预测性。分支的熵，用*H*表示，是根据以下公式计算的。![图片](img/faf67a04-8f90-4c08-b99d-a1ae95804c69.png)代表属性的熵。![图片](img/a03e73dd-2003-4c47-a55a-8b355a5ded39.png)是数据集中类变量的数量。![图片](img/5a31aed3-e450-45d4-9618-e8721da55b32.png)是数据集中属于该类的观察值的比例，![图片](img/fbd47654-3821-45aa-89dd-d73f69e93690.png)：

![图片](img/5d65e3d7-daba-4b80-b152-dade6bf59135.png)

**步骤 1**：在我们的例子中，对于整个数据集，我们可以按照以下方式计算数据集的熵。

![图片](img/01cf8bda-1f09-470f-aa69-9846c0257f32.png)

**步骤 2**：在我们的决策树中，我们根据汽车的型号来分割树。因此，我们也计算了树的每个分支的熵，如下所示：

![图片](img/db694e43-90fd-41a6-b5c3-1843aa950295.png)

![图片](img/dbc40d20-a5fb-47a0-bf2b-68bde5a06fbd.png)

![图片](img/b9e63287-3d3e-41bc-be71-2384ef33fdc2.png)

**步骤 3**：基于父节点和分支的熵，我们可以使用称为**信息增益**的度量来评估分支。对于父分支![图片](img/16b7bb2a-eac9-4d30-8422-29ecf3315a9c.png)和属性![图片](img/88e885c5-21d3-47e0-a455-15404f53baf9.png)，信息增益表示如下：

![图片](img/bdea57aa-19c4-4d55-b646-4844a3415ba5.png)

![图片](img/873963f4-165f-49bf-9406-1ab0817f9ce8.png)是子节点熵的加权总和。在我们的例子中，*Make*属性的![图片](img/b1162a12-fc43-4cda-ad99-31d329d8df34.png)计算如下：

![图片](img/ef8027cc-d825-411c-8cef-ccdf02baba9e.png)

因此，*Make*属性的信息增益计算如下：

![图片](img/85f98981-1681-43e7-b486-1da57abb3f24.png)

同样，我们可以计算其他属性的信息增益分数。应该使用信息增益最高的属性来分割数据集，以获得最高质量的决策树。信息增益在 ID3 和 C4.5 算法中使用。

# 停止分割树的准则

由于决策树生成算法是递归的，我们需要一个准则来指示何时停止分割树。我们可以设置各种准则来停止分割树。现在让我们看看常用准则的列表：

+   **节点中的观测数**：我们可以设置标准，如果分支中的观测数少于预先指定的数量，则停止分支的递归。一个很好的经验法则是当分支中有少于 5%的总训练数据时停止递归。如果我们过度分割数据，以至于每个节点只有一个数据点，这会导致决策树过度拟合训练数据。任何之前未见过的新观察结果将无法在这些树中得到准确分类。

+   **节点的纯度**：在*基尼不纯度*部分，我们学习了如何计算对随机观察进行分类的错误概率。我们也可以使用相同的方法来计算数据集的纯度。如果一个分支中的子集纯度大于预先指定的阈值，我们可以停止基于该分支的分割。

+   **树的深度**：我们也可以预先指定树的深度限制。如果任何分支的深度超过限制，我们可以停止进一步分割该分支。

+   **修剪树木**：另一种策略是让树木充分生长。这样做可以避免分支过早终止，而不考虑未来。然而，在完全构建了整棵树之后，树可能很大，某些分支可能存在过拟合。因此，应用修剪策略来评估树的每一分支；任何引入的杂质少于预先指定的父分支杂质量的分支将被消除。修剪决策树有多种技术。我们鼓励读者在实现决策树的库中进一步探索这个主题。

# 理解随机森林算法

使用决策树有两个主要缺点。首先，决策树使用基于成本函数选择在属性上分割的算法。决策树算法是一种贪婪算法，在做出关于将数据集分割成两个子集的每个决策时，都趋向于局部最优。然而，它并没有探索在属性上做出次优决策是否会导致未来有更优的决策树。因此，当我们运行此算法时，我们不会得到全局最优的树。其次，决策树倾向于过度拟合训练数据。例如，数据集中可用的少量观察结果可能导致一个分支，该分支提供了非常高的某个类事件发生的概率。这导致决策树在生成用于训练的数据集的正确预测方面表现得非常好。然而，对于他们以前从未见过的观察结果，由于过度拟合训练数据，决策树可能不准确。

为了解决这些问题，可以使用随机森林算法来提高现有决策树算法的准确性。在这种方法中，我们将训练数据划分为随机子集，并创建一个决策树集合，每个决策树基于一个子集。这解决了过拟合的问题，因为我们不再依赖于一棵树来做出对整个训练集过度拟合的决策。其次，这也帮助解决了仅基于成本函数在一个属性上分割的问题。随机森林中的不同决策树可能会基于它们训练的随机样本，基于不同的属性做出分割决策。

在预测阶段，随机森林算法从每个分支获取事件的概率，并使用投票方法生成预测。这有助于我们抑制可能过度拟合或做出次优决策的树的预测。将训练集划分为随机子集并训练多个机器学习模型的方法被称为**Bagging**。Bagging 方法也可以应用于其他机器学习算法。

# 理解梯度提升算法

梯度提升算法也被用来解决决策树算法的缺点。然而，与基于训练数据随机子集训练多个树的随机森林算法不同，梯度提升算法通过减少决策树中的错误来顺序地训练多个树。梯度提升决策树基于一种流行的机器学习技术，称为**自适应提升**，其中我们学习为什么机器学习模型会出错，然后训练一个新的机器学习模型来减少先前模型的错误。

梯度提升算法在数据中发现了难以在决策树中表示的规律，并给训练示例添加了更大的权重，这可能导致正确的预测。因此，与随机森林类似，我们从训练数据的一个子集中生成多个决策树。然而，在每一步中，训练数据的子集并不是随机选择的。相反，我们创建一个训练数据子集，其中优先考虑那些会导致决策树中错误更少的示例。当我们无法观察到可能导致更多优化的错误模式时，我们停止这个过程。

下一个部分提供了随机森林算法和梯度提升算法实现的示例。

# 预测日志流中的点击

在本节中，我们将向您展示如何使用基于树的方法来预测在给定一系列条件（如地区、广告展示的位置、一天中的时间、横幅的位置以及提供广告的应用程序）的情况下，谁会点击移动广告。

我们将在本章的其余部分使用的数据集来自 *Shioji, Enno, 2017, Adform 点击预测数据集*，[`doi.org/10.7910/DVN/TADBY7`](https://doi.org/10.7910/DVN/TADBY7)，哈佛数据集，V2。

主要任务是构建一个分类器，能够根据条件预测用户是否会点击广告。拥有这样的模型对于选择向用户展示哪些广告以及何时展示的广告技术平台非常有用。这些平台可以使用这些模型只为可能点击所提供广告的用户展示广告。

数据集足够大（5 GB），足以证明使用跨多台机器的技术来执行训练是合理的。我们首先将探讨如何使用 AWS EMR 结合 Apache Spark 来完成这项任务。我们还将展示如何使用 SageMaker 服务来完成这项任务。

# 弹性 MapReduce (EMR) 简介

EMR 是 AWS 服务，允许我们运行和扩展 Apache Spark、Hadoop、HBase、Presto、Hive 和其他大数据框架。我们将在第十五章中详细介绍 EMR 的更多细节，*调整集群以进行机器学习*。然而，现在让我们将 EMR 视为一种服务，允许我们启动几个运行软件（如 Apache Spark）的相互连接的机器，这些软件协调分布式处理。EMR 集群有一个主节点和几个从节点。主节点通常协调作业，而从节点处理和合并数据，为主节点提供结果。这个结果可以从一个简单的数字（例如，行数计数）到一个能够进行预测的机器学习模型不等。Apache Spark Driver 是协调完成操作所需作业的机器。驱动程序通常在主节点上运行，但它也可以配置在从节点上运行。Spark executors（Spark 用于处理数据的恶魔）通常在 EMR 从节点上运行。

EMR 还可以托管连接到集群的笔记本服务器。这样，我们可以运行笔记本段落，这将触发通过 Apache Spark 的任何分布式处理。在 Apache Spark 上托管笔记本有两种方式：EMR 笔记本和 JupyterHub EMR 应用程序。我们将在本章中使用第一种方法，并在第十五章“针对机器学习调整集群”中介绍 JupyterHub。

通过 EMR 笔记本，你可以通过控制台上的**EMR 笔记本**链接同时启动集群和笔记本（[`console.aws.amazon.com/elasticmapreduce/home`](https://console.aws.amazon.com/elasticmapreduce/home)）。

你可以通过点击“创建笔记本”按钮同时创建集群和笔记本，如下截图所示：

![截图](img/de2ef949-ffb7-4b85-b029-e104dab52790.png)

创建笔记本后，它将点击“打开”按钮，如下截图所示：

![截图](img/804659f6-2720-4e02-bb01-6f62a27b27e6.png)

点击“打开”按钮将打开笔记本，以便我们开始编码。正如以下截图所示，笔记本是一个标准的 Jupyter Notebook：

![截图](img/f9a106eb-25fa-4d0f-9af5-3bad1df69518.png)

或者，你可以单独创建集群，然后将笔记本附加到集群上。这样做的好处是你可以访问额外的先进选项。

我们建议至少使用 10 台机器（例如，10 个 m5.xlarge 节点）来及时运行本章中的代码。另外，如果你的作业完成时间超过一小时，我们建议你增加 Livy 会话超时时间。对于此类作业，笔记本可能会与集群断开连接。Livy 是负责笔记本与集群之间通信的软件。以下截图显示了创建集群选项，包括扩展 Livy 会话超时时间的方法：

![截图](img/92fd675d-eee4-4138-ae7d-4179f17419a4.png)

在第十五章“针对机器学习调整集群”中，我们将详细介绍集群配置的更多细节。

# 在 EMR 上使用 Apache Spark 进行训练

现在我们来探索在 EMR 上使用 Apache Spark 进行训练。

# 获取数据

第一步是将数据上传到 EMR。你可以直接从笔记本中这样做，或者将数据集本地下载后，使用 AWS 的命令行工具（awscli）将其上传到 S3。为了使用 AWS 的命令行工具，你需要在 IAM 控制台上创建 AWS 访问密钥。有关如何操作的详细信息，请参阅此处：[`docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html`](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)。

一旦你有了 AWS 访问密钥和秘密密钥，你可以在命令行上执行`aws configure`来配置它们。

首先，我们将通过以下`wget`命令获取数据集的一部分：

```py
wget -O /tmp/adform.click.2017.01.json.gz https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/TADBY7/JCI3VG
```

接下来，我们将解压并将 CSV 数据集上传到名为 `mastering-ml-aws` 的 `s3` 存储桶，如下命令所示：

```py
gunzip /tmp/adform.click.2017.01.json.gz

aws s3 cp /tmp/adform.click.2017.01.json s3://mastering-ml-aws/chapter4/training-data/adform.click.2017.01.json
```

一旦数据在 S3 中，我们就可以回到我们的笔记本并开始编写代码来训练分类器。

# 准备数据

与我们在前几章中在本地运行的示例相比（*第二章，*使用朴素贝叶斯分类 Twitter 推文和* 第三章，*使用回归算法预测房屋价值），EMR 笔记本具有隐式变量以访问 Spark 上下文。特别是，Spark 会话被命名为 `spark`。第一次运行的第一段将始终初始化上下文并触发 Spark 驱动器。

在下面的屏幕截图中，我们可以看到 Spark 应用程序启动以及 Spark UI 的链接：

![图片](img/02ff36ff-e8d8-4321-9ae9-36c3f28ffe4e.png)

下一步是加载我们的数据集并通过运行以下代码片段来探索前几行：

```py
ctr_df = spark.read.json(s3_train_path)
ctr_df.show(5)
```

上面的 `show` 命令的输出如下：

![图片](img/5e2044ea-ac04-4c4d-8a48-9b1fb485b0b0.png)

`spark.read.json` 方法，上一段代码块中的第一个命令，将 JSON 数据读取到一个数据集，类似于我们之前使用 `spark.read.csv` 处理 CSV 数据所做的那样。我们可以观察到我们的数据集有 10 个特征和一个 `l` 列，表示我们试图预测的标签，即用户是否点击了（1）或未点击（0）广告。你可能意识到一些特征是多值的（单元格中有多个值）并且一些是空的。为了简化本章的代码示例，我们将只选择前五个特征，通过构建一个新的数据集并将这些特征命名为 `f0` 到 `f4`，同时将空特征替换为值 `0`，并在多值特征的情况下只取第一个值：

```py
df = ctr_df.selectExpr("coalesce(c0[0],0) as f0",
                       "coalesce(c1[0],0) as f1",
                       "coalesce(c2[0],0) as f2",
                       "coalesce(c3[0],0) as f3",
                       "coalesce(c4[0],0) as f4",
                       "l as click")
```

上面的 `selectExpr` 命令允许我们使用类似 SQL 的操作。在这种情况下，我们将使用 coalesce 操作，该操作将任何空表达式转换为值 `0`。此外，请注意，我们总是只取多值特征的第一个值。

通常，丢弃特征是一个坏主意，因为它们可能携带重要的预测价值。同样，用固定值替换空值也可能不是最优的。我们应该考虑常见的缺失值填充技术，例如用点估计（中位数、众数和平均值是常用的）。或者，可以训练一个模型来从剩余特征中填充缺失值。为了保持本章专注于使用树，我们不会深入探讨缺失值的问题。

我们的 `df` 数据集现在看起来如下：

![图片](img/1172103f-359d-456d-84fb-10caa8390080.png)

现在我们做一件相当特定于 Spark 的事情，即将 CSV 的不同部分重新洗牌到不同的机器上并在内存中缓存它们。执行此操作的命令如下：

```py
df = df.repartition(100).cache()
```

由于我们将反复迭代处理相同的数据集，通过将其加载到内存中，这将显著加快对`df`进行的任何未来操作。重新分区有助于确保数据在整个集群中分布得更好，从而提高并行化程度。

`describe()` 方法构建了一个包含我们数据集中不同字段的基本统计信息（`min`，`max`，`mean`，`count`）的数据框，如下截图所示：

![图片](img/a789e41c-f793-4e40-b93e-490782298ade.png)

我们可以观察到，大多数特征的范围从较低的负值到非常大的整数，这表明这些是经过匿名处理的特征值，对它们应用了哈希函数。我们试图预测的字段是`click`，当用户点击广告时为`1`，未点击时为`0`。点击列的均值告诉我们存在一定程度的标签不平衡（大约 18%的实例是点击）。此外，`count`行告诉我们我们的数据集中总共有 1,200,000,000 行。

另一个有用的检查是了解分类值的基数。以下是我们笔记本中的截图，显示了每个特征得到的唯一值的数量：

![图片](img/5b320793-9f7f-49df-bf65-c31acbf00164.png)

如您所见，f4 特征是一个具有许多不同值的分类的例子。这类特征通常需要特别注意，正如我们将在本节后面看到的那样。

决策树和 Spark ML 库中的大多数库都需要我们的特征是数值型的。碰巧我们的特征已经是数值形式，但它们实际上代表的是被哈希成数字的分类。在第二章中，我们学习了为了训练分类器，我们需要提供一个数字向量。因此，我们需要将我们的分类转换成数字，以便将它们包含在我们的向量中。这种转换通常被称为**特征编码**。有两种流行的方法可以实现这一点：通过独热编码或分类编码（也称为**字符串索引**）。

在以下通用示例中，我们假设`site_id`特征只能取最多三个不同的值：`siteA`，`siteB`和`siteC`。这些示例还将说明我们如何将字符串特征编码成数字（不同于我们数据集中的整数哈希）的情况。

# 分类编码

分类编码（或字符串索引）是最简单的一种编码方式，其中我们为每个站点值分配一个数字。让我们看看以下表中的例子：

| `site_id` | `site_id_indexed` |
| --- | --- |
| `siteA` | `1` |
| `siteB` | `2` |
| `siteC` | `3` |

# 独热编码

在这种编码中，我们为每个可能的站点值创建新的二进制列，并在值存在时将其设置为`1`，如下表所示：

| `site_id` | `siteA` | `siteB` | `siteC` |
| --- | --- | --- | --- |
| `siteA` | `1` | `0` | `0` |
| `siteB` | `0` | `1` | `0` |
| `siteC` | `0` | `0` | `1` |

分类编码很简单；然而，它可能会创建特征的虚假排序，并且一些机器学习算法对这一点很敏感。独热编码有额外的优点，可以支持多值特征（例如，如果某行有两个站点，我们可以在两个列中都设置`1`）。然而，独热编码会增加我们的数据集特征数量，从而增加维度。向数据集中添加更多维度会使训练更加复杂，并可能降低其预测能力。这被称为**维度诅咒**。

让我们看看我们如何在数据集的一个样本上使用分类编码，将 C1 特征（一个分类特征）转换为数值：

```py
from pyspark.ml.feature import StringIndexer

string_indexer = StringIndexer(inputCol="f0", outputCol="f0_index")
string_indexer_model = string_indexer.fit(df)
ctr_df_indexed = string_indexer_model.transform(df).select('f0','f0_index')
ctr_df_indexed.show(5)
```

上述代码首先实例化一个`StringIndexer`，该索引器将在拟合时将列`f0`编码到新的列`f0_index`中，遍历数据集并找到不同的特征值，根据这些值的流行度分配索引。然后我们可以使用`transform()`方法为每个值获取索引。上述最后的`show()`命令的输出如下所示：

![](img/1d2c0b3a-c31c-4f11-bc7f-e7f75243b79e.png)

在上述截图中，我们可以看到每个原始（散列）分类值被分配的数值。

要对值执行独热编码，我们使用`OneHotEncoder`转换器：

```py
from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCol="f0_index", outputCol="f0_encoded")
encoder.transform(ctr_df_indexed).distinct().show(5)
```

上述命令生成了以下输出：

![](img/0303965c-ce6c-422a-bbbb-91c92eb20b7a.png)

注意不同的`f0`值是如何映射到相应的布尔向量的。我们只为一个特征进行了编码；然而，为了训练，我们需要对多个特征执行相同的过程。因此，我们构建了一个函数，该函数构建了我们管道所需的所有索引和编码阶段：

```py
def categorical_one_hot_encoding_stages(columns):
    indexers = [StringIndexer(inputCol=column, 
                              outputCol=column + "_index", 
                              handleInvalid='keep') 
                for column in columns]
    encoders = [OneHotEncoder(inputCol=column + "_index", 
                              outputCol=column + "_encoded") 
                for column in columns]
    return indexers + encoders
```

以下代码构建了一个包含`DecisionTree`估计器的训练管道：

```py
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

categorical_columns = ['f0','f1','f2','f3','f4']
encoded_columns = [column + '_encoded' for column in categorical_columns] 

categorical_stages = categorical_one_hot_encoding_stages(categorical_columns) vector_assembler = VectorAssembler(inputCols=encoded_columns,
                                   outputCol="features")
selector = ChiSqSelector(numTopFeatures=100, featuresCol="features",
                         outputCol="selected_features", labelCol="click")
decision_tree = DecisionTreeClassifier(labelCol="click",                                       
                                       featuresCol="selected_features")

pipeline = Pipeline(stages=categorical_stages + [vector_assembler, selector, 
                                                 decision_tree])
```

在前面的代码中，`VectorAssembler`构建了一个包含所有需要编码的特征以及数值特征的向量（`VectorAssembler`可以接受作为输入的列，这些列可以是向量或标量，因此如果您的数据集中存在数值特征，您可以直接使用它们）。由于独热编码值数量众多，特征向量可能非常大，这会使训练器非常慢或需要大量的内存。减轻这种情况的一种方法是用**卡方**特征选择器。在我们的管道中，我们选择了最佳的 100 个特征。这里的“最佳”是指具有更多预测能力的特征——注意卡方估计器如何同时考虑特征和标签来决定最佳特征。最后，我们包括决策引擎估计器阶段，这是实际创建分类器的阶段。

如果我们尝试将具有非常大的基数（cardinality）的特征索引串联起来，驱动程序将收集所有可能的值（为了保留一个值到索引的字典以进行转换）。在这种尝试中，驱动程序很可能会耗尽内存，因为我们正在查看数百万个不同的值来保留。对于这些情况，你需要其他策略，例如只保留最具预测能力的特征，或者只考虑最受欢迎的值。查看我们的文章，其中包含了解决这个问题的方案，请参阅[`medium.com/dataxutech/how-to-write-a-custom-spark-classifier-categorical-naive-bayes-60f27843b4ad`](https://medium.com/dataxutech/how-to-write-a-custom-spark-classifier-categorical-naive-bayes-60f27843b4ad)。

# 训练一个模型

我们的数据管道现在已构建完成，因此我们可以继续将数据集分割为测试和训练集，然后拟合模型：

```py
train_df, test_df = df.randomSplit([0.8, 0.2], seed=17)
pipeline_model = pipeline.fit(train_df)
```

一旦执行，Spark 驱动程序将确定在多台机器上分配训练模型所需处理的最佳方案。

通过跟随本节开头显示的 Spark UI 链接，我们可以查看在 EMR 上运行的不同作业的状态：

![图片](img/1ded0e39-d9eb-4dd5-9e98-a0a93532ff6a.png)

一旦模型训练完成，我们可以探索其背后的决策树。我们可以通过检查管道的最后一个阶段（即决策树模型）来实现这一点。

以下代码片段显示了以文本格式输出决策树的结果：

```py
print(pipeline_model.stages[-1].toDebugString)

DecisionTreeClassificationModel (uid=DecisionTreeClassifier_3cc3252e8007) of depth 5 with 11 nodes
  If (feature 3 in {1.0})
   Predict: 1.0
  Else (feature 3 not in {1.0})
   If (feature 21 in {1.0})
    Predict: 1.0
   Else (feature 21 not in {1.0})
    If (feature 91 in {1.0})
     Predict: 1.0
    Else (feature 91 not in {1.0})
     If (feature 27 in {1.0})
      Predict: 1.0
     Else (feature 27 not in {1.0})
      If (feature 29 in {1.0})
       Predict: 1.0
      Else (feature 29 not in {1.0})
       Predict: 0.0 
```

注意每个决策都是基于一个取值为`0`或`1`的特征。这是因为我们在管道中使用了 one-hot 编码。如果我们使用了分类编码（字符串索引），我们会看到涉及多个索引值的条件，例如以下示例：

```py
 If (feature 1 in {3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,17.0,27.0})
       Predict: 0.0
 Else (feature 1 not in {3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,17.0,27.0})
       Predict: 1.0
```

# 评估我们的模型

与我们在第二章中讨论的 Twitter 分类问题第二章，“使用朴素贝叶斯分类 Twitter 帖子”，这个数据集的标签非常倾斜。这是因为用户决定点击广告的情况很少。我们在第二章中使用的准确性度量第二章，“使用朴素贝叶斯分类 Twitter 帖子”，将不适用，因为一个从未预测点击的模型仍然会有非常高的准确性（所有非点击都会导致正确预测）。对于这种情况，两种可能的替代方案是使用 ROC 或**精确率-召回率曲线**衍生出的度量，这些将在下一节中展示。

# ROC 曲线下的面积

**接收者操作特征**（**ROC**）是真正例率和假正例率之间权衡的表示。真正例率描述了当实际类别为正时，模型预测正类别的良好程度。真正例率是模型预测的真正例与真正例和假负例之和的比率。假正例率描述了模型预测正类别的频率，当实际类别为负时。假正例率是假正例与假正例和真正例之和的比率。ROC 是一个图表，其中*x*轴表示范围在 0-1 之间的假正例率，而*y*轴表示真正例率。**曲线下面积**（**AUC**）是 ROC 曲线下面积的措施。AUC 是分类模型预测性的衡量指标。

下面的屏幕截图显示了三个接收者操作曲线的例子：

![图片](img/fc95ba2b-6336-4f9b-96e7-dec2cbbcb0d4.png)

在前面的图表中，虚线代表当`AUC`为`1`的情况。这种`AUC`出现在所有正结果都被正确分类时。实线代表`AUC`为`0.5`的情况。对于二元分类器来说，当机器学习模型的预测结果与随机生成结果相似时，`AUC`为`0.5`。这表明机器学习模型在预测结果方面并不比随机数生成器更好。虚线代表`AUC`为`0.66`的情况。这发生在机器学习模型正确预测了一些例子，但并非全部。然而，如果二元分类器的`AUC`高于`0.5`，则说明模型比随机猜测结果更好。但如果它低于`0.5`，这意味着机器学习模型比随机结果生成器更差。因此，AUC 是衡量机器学习模型和评估其有效性的良好指标。

# 精确率-召回率曲线下的面积

精确率-召回率曲线表示预测模型中精确率和召回率之间的权衡。**精确率**定义为模型做出的所有正预测中真正例的比例。**召回率**定义为正预测与实际正预测总数的比例。

注意，精确率-召回率曲线不模拟真正负值。这在数据集不平衡的情况下很有用。如果模型擅长分类真正负值并生成较少的假阳性，ROC 曲线可能会提供一个非常乐观的模型视图。

下面的图表展示了一个精确率-召回率曲线的例子：

![图片](img/d6012ad5-b4ee-4bdd-965c-db7d4adb5d20.png)

在前面的截图上，虚线表示当精确度-召回率曲线下的面积是 `0.5` 的时候。这表明精确度始终为 0.5，类似于随机数生成器。实线代表比随机更好的精确度-召回率曲线。精确度-召回率曲线也可以用来评估机器学习模型，类似于 ROC 面积。然而，当数据集不平衡时，应使用精确度-召回率曲线，而当数据集平衡时，应使用 ROC。

因此，回到我们的示例，我们可以使用 Spark 的`BinaryClassificationEvaluator`通过提供测试数据集上的实际和预测标签来计算分数。首先，我们将模型应用于我们的测试数据集以获取预测和分数：

```py
test_transformed = pipeline_model.transform(test_df)
```

通过应用前面的转换 `test_transformed`，将包含在 `test_df` 中的所有列以及一个额外的列 `rawPrediction`，该列将有一个用于评估的分数：

```py

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", 
                                          labelCol="click")
evaluator.evaluate(test_transformed, 
                   {evaluator.metricName: "areaUnderROC"})
```

前一个命令的输出是 0.43\. 我们得到一个低于 0.5 的 ROC 指标意味着我们的分类器甚至比随机分类器还要差，因此它不是一个好的点击预测模型！在下一节中，我们将展示如何使用集成模型来提高我们的预测能力。

# 在 EMR 上训练树集成

决策树可以用来理解我们的分类器所做的决策，尤其是在决策树较小且可读时。然而，决策树往往会对数据进行过拟合（通过学习训练数据集的细节而无法对新数据进行泛化）。因此，机器学习从业者倾向于使用树集成，如随机森林和梯度提升树，这些在本书的 *理解梯度提升算法* 和 *理解随机森林算法* 部分中已有解释。

在我们的代码示例中，要使用随机森林或梯度提升树，我们只需将我们的管道的最后一个阶段替换为相应的构造函数：

```py
from pyspark.ml.classification import RandomForestClassifier

random_forest = RandomForestClassifier(labelCol="click",                                                                                        
                                       featuresCol="features")

pipeline_rf = Pipeline(stages=categorical_stages + \
                              [vector_assembler, random_forest])
```

注意我们如何在我们的样本数据集上通过随机森林获得更好的 ROC 值：

```py
rf_pipeline_model = pipeline_rf.fit(train_df)

evaluator.evaluate(rf_pipeline_model.transform(test_df), 
                   {evaluator.metricName: "areaUnderROC"})

>> 0.62
```

我们可以看到现在我们得到了一个大于 0.5 的 ROC 值，这意味着我们的模型已经改进，现在比随机猜测要好。同样，你可以使用`pyspark.mllib.tree.GradientBoostedTrees`类训练梯度提升树。

# 使用 SageMaker 服务训练梯度提升树

在 *训练模型* 和 *评估我们的模型* 部分中，我们学习了如何在 EMR 上使用 Spark 构建和评估随机森林分类器。在本节中，我们将通过 SageMaker 笔记本了解如何使用 SageMaker 服务训练梯度提升树。XGBoost SageMaker 服务允许我们以分布式方式训练梯度提升树。鉴于我们的点击数据相对较大，使用此类服务将非常方便。

# 准备数据

为了使用 SageMaker 服务，我们需要将我们的训练和测试数据放在 S3 上。[`docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html`](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html) 中的文档要求我们将数据作为 CSV 文件降级，其中第一列表示训练标签（目标特征），其余列表示训练特征（支持其他格式，但我们将使用 CSV 作为示例）。为了以这种方式分割和准备数据，EMR 仍然是最佳选择，因为我们希望我们的数据准备也是分布式的。鉴于上一节“准备数据”中的测试和训练 Spark 数据集，我们可以应用管道模型，在这种情况下不是用于获取预测，而是用于获取每行的选定编码特征。

在以下片段中，我们对 `test_df` 和 `train_df` 都应用了模型转换：

```py
test_transformed = model.transform(test_df)
train_transformed = model.transform(train_df)
```

以下截图显示了 `test_transformed` 数据框的最后三列：

![图片](img/04932beb-ce2f-4b4b-af48-02cff4bab6f9.png)

转换后的数据集包括特征向量列（命名为 `selected_features`，大小为 100）。我们需要将这些两列转换成一个包含 101 列的 CSV 文件（`click` 和 `selected_features` 向量展开）。Spark 中的一个简单转换使我们能够做到这一点。我们定义了一个 `deconstruct_vector` 函数，我们将使用它来获取一个 Spark 数据框，其中标签和每个向量组件作为单独的列。然后我们将这些数据保存到 S3，用于训练和测试，作为没有标题的 CSV 文件，因为 SageMaker 需要。

在以下代码片段中，我们提供了 `deconstruct_vector` 函数以及保存数据框所需的系列转换：

```py
def deconstruct_vector(row):
    arr = row['selected_features'].toArray()
    return tuple([row['click']] + arr.tolist())

df_for_csv = train_transformed.select("click", "selected_features") \
                .rdd.map(deconstruct_vector).toDF() 

df_for_csv.write.csv('s3://mastering-ml-aws/chapter4/train-trans-vec-csv-1/', 
                     header=False)
```

以类似的方式，我们将在 `s3://mastering-ml-aws/chapter4/test-trans-vec-csv-no-label` 路径下保存一个额外的 CSV 文件，该文件不包括标签（只有特征）。我们将在下一节“使用 SageMaker XGBoost 训练”中，通过 SageMaker 批量转换作业使用这个数据集来评分测试数据集。

# 使用 SageMaker XGBoost 训练

现在我们训练和测试的数据集已经以正确的格式保存在 S3 上，我们可以启动我们的 SageMaker 笔记本实例并开始编写我们的训练器代码。让我们执行以下步骤：

1.  使用我们数据集的位置实例化 SageMaker 会话、容器和变量：

```py
import sagemaker
from sagemaker import get_execution_role
import boto3

sess = sagemaker.Session()
role = get_execution_role()
container = sagemaker.amazon.amazon_estimator.get_image_uri('us-east-1', 
                                                            'xgboost', 
                                                            'latest')

s3_validation_data = \
    's3://mastering-ml-aws/chapter4/test-trans-vec-csv-1/'
s3_train_data = \
    's3://mastering-ml-aws/chapter4/train-trans-vec-csv-1/'
s3_test_data = \
    's3://mastering-ml-aws/chapter4/test-trans-vec-csv-no-label/'
s3_output_location = \
    's3://mastering-ml-aws/chapter4/sagemaker/output/xgboost/'
```

1.  通过实例化 SageMaker 估算器并提供基本参数（例如要使用的机器数量和类型等详细信息可在 AWS 文档中找到，链接为 [`sagemaker.readthedocs.io/en/stable/estimators.html`](https://sagemaker.readthedocs.io/en/stable/estimators.html)）来创建一个分类器：

```py
sagemaker_model = sagemaker.estimator.Estimator(container,
    role,
    train_instance_count=1,
    train_instance_type='ml.c4.4xlarge',
    train_volume_size=30,
    train_max_run=360000,
    input_mode='File',
    output_path=s3_output_location,
    sagemaker_session=sess)
```

1.  设置训练器的超参数。详细信息可以在文档中找到（我们将在第十四章中更详细地介绍，*优化 SageMaker 和 Spark 机器学习模型*）。这里要查看的主要参数是目标，我们将其设置为二元分类（使用逻辑回归分数，这是 XGBoost 执行分类的标准方式）。XGBoost 也可以用于其他问题，如回归或多类分类：

```py
sagemaker_model.set_hyperparameters(objective='binary:logistic',
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    silent=0,
    num_round=50)
```

1.  在拟合模型之前，我们需要指定输入的位置和格式（接受几种格式；在我们的示例中我们选择了 CSV）：

```py
train_data = sagemaker.session.s3_input(s3_train_data, 
    distribution='FullyReplicated',
    content_type='text/csv', 
    s3_data_type='S3Prefix')

validation_data = sagemaker.session.s3_input(s3_validation_data, 
    distribution='FullyReplicated',
    content_type='text/csv', 
    s3_data_type='S3Prefix')

data_channels = {'train': train_data, 
                 'validation': validation_data}

sagemaker_model.fit(inputs=data_channels, 
                    logs=True)
```

1.  调用`fit`函数将使用提供的数据（即，我们通过 EMR/Spark 准备存储在 S3 中的数据）训练模型：

```py
INFO:sagemaker:Creating training-job with name: xgboost-2019-04-27-20-39-02-968
2019-04-27 20:39:03 Starting - Starting the training job...
2019-04-27 20:39:05 Starting - Launching requested ML instances......
...
train-error:0.169668#011validation-error:0.169047
2019-04-27 20:49:02 Uploading - Uploading generated training model
2019-04-27 20:49:02 Completed - Training job completed
Billable seconds: 480
```

日志将显示 XGBoost 正在优化的训练和验证错误的一些细节，以及作业状态和训练成本。

# 应用和评估模型

以下步骤将展示如何使用`sagemaker`创建批量预测，以便您可以评估模型。

为了获得预测，我们可以使用批量转换作业：

```py
transformer = sagemaker_model.transformer(instance_count=1, 
                                          instance_type='ml.m4.2xlarge',
                                          output_path=s3_output_location)
transformer.transform(s3_test_data, 
                      content_type='text/csv', 
                      split_type='Line')
transformer.wait()
```

对于输入`s3`目录中的每个文件，批量转换作业将生成一个包含分数的文件：

```py
aws s3 ls s3://mastering-ml-aws/chapter4/sagemaker/output/xgboost/ | head
```

前面命令生成了以下输出：

```py
2019-04-28 01:29:58 361031 part-00000-19e45462-84f7-46ac-87bf-d53059e0c60c-c000.csv.out
2019-04-28 01:29:58 361045 part-00001-19e45462-84f7-46ac-87bf-d53059e0c60c-c000.csv.out
```

然后，我们可以将这个单列 CSV 文件加载到`pandas`数据框中：

```py
import pandas as pd

scores_df = pd.read_csv(output_path + \
   'part-00000-19e45462-84f7-46ac-87bf-d53059e0c60c-c000.csv.out',
    header=None, 
    names=['score'])
```

这些分数代表概率（通过逻辑回归得出）。如果我们把目标设置为二元：hinge，我们将得到实际的预测。选择使用哪种类型取决于应用类型。在我们的案例中，收集概率似乎很有用，因为任何表明特定用户更有可能进行点击的迹象都将有助于提高营销定位。

SageMaker XGBoost 的一个优点是它提供了与 Python 标准序列化库（pickle）兼容的 XGBoost 模型的 S3 序列化。作为一个例子，我们将从 S3 中的测试数据的一部分运行模型以获取分数。通过这样做，我们可以通过以下步骤计算 ROC 曲线下的面积：

1.  在`s3`中定位模型 tar 包：

```py
aws s3 ls --recursive s3://mastering-ml-aws/chapter4/sagemaker/output/xgboost/ | grep model
```

输出如下所示：

```py
chapter4/sagemaker/output/xgboost/xgboost-2019-04-27-20-39-02-968/output/model.tar.gz
```

将模型从 S3 复制到我们的本地目录并解压 tar 包：

```py
aws s3 cp s3://mastering-ml-aws/chapter4/sagemaker/output/xgboost/xgboost-2019-04-27-20-39-02-968/output/model.tar.gz /tmp/model.tar.gz
tar xvf model.tar.gz
```

以下是前面命令的输出，显示了从 tar 包中解压的文件名：

```py
xgboost-model
```

1.  一旦模型被本地下载并解压，我们可以通过`pickle`序列化库将模型加载到内存中：

```py
import xgboost
import pickle as pkl

model_local = pkl.load(open('xgboost-model', 'rb'))
```

1.  定义我们列的名称（`f0`到`f99`为特征，`click`为标签）并从 S3 加载验证数据：

```py
column_names = ['click'] + ['f' + str(i) for i in range(0, 100)]
validation_df = pd.read_csv(s3_validation_data + \
                            'part-00000-25f35551-ffff-41d8-82a9-75f429553035-c000.csv',
                            header=None, 
                            names=column_names)
```

1.  要使用`xgboost`创建预测，我们需要从我们的`pandas`数据框组装一个矩阵。选择除第一个（标签）之外的所有列，然后构建一个 DMatrix。从`xgboost`模型调用 predict 方法以获取每行的分数：

```py
import xgboost
matrix = xgboost.DMatrix(validation_df[column_names[1:]])
validation_df['score'] = model_local.predict(matrix)
```

在下面的屏幕截图中，读者可以看到数据框的外观：

![图片](img/fd2a3999-60a2-431b-966a-ff6fe1cf1c58.png)

1.  给定`点击`列和`得分`列，我们可以构建 ROC AUC：

```py
from sklearn.metrics import roc_auc_score
roc_auc_score(validation_df['click'], validation_df['score'])
```

对于我们的样本，我们得到一个 AUC 值为 0.67，这与我们使用 Spark 的随机森林得到的结果相当。

在本章中，我们没有专注于为我们的数据集构建最优模型。相反，我们专注于提供简单且流行的转换和树模型，您可以使用它们来分类大量数据。

# 摘要

在本章中，我们介绍了理解树集成的基本理论概念，并展示了如何通过 Apache Spark 以及 SageMaker XGBoost 服务在 EMR 中训练和评估这些模型。决策树集成是最受欢迎的分类器之一，原因有很多：

+   它们能够在相对较短的训练时间和少量资源中找到复杂的模式。XGBoost 库被认为是 Kaggle 竞赛获胜者中最受欢迎的分类器（这些竞赛是为了寻找开放数据集的最佳模型而举办的）。

+   有可能理解为什么分类器正在预测一个特定的值。遵循决策树路径或只是查看特征重要性是快速理解树集成决策背后的理由的快捷方式。

+   通过 Apache Spark 和 XGBoost 可以提供分布式训练的实现。

在下一章中，我们将探讨如何使用机器学习根据客户的行为模式对客户进行聚类。

# 练习

1.  随机森林和梯度提升树之间主要区别是什么？

1.  解释为什么 Gini 不纯度可以解释为误分类率。

1.  解释为什么对分类特征进行特征编码是必要的。

1.  在本章中，我们提供了两种进行特征编码的方法。找出另一种对分类特征进行编码的方法。

1.  解释为什么我们在第二章中使用的准确度指标，“使用朴素贝叶斯分类 Twitter 帖子”不适合预测我们数据集中的点击。

1.  找出我们可以用于 XGBoost 算法的其他目标。你会在什么时候使用每个目标？
