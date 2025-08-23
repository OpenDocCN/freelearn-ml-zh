# *第10章*：使用DEAP和Microsoft NNI进行高级超参数调整

**DEAP**和**Microsoft NNI**是Python包，提供了其他包中未实现的多种超参数调整方法，这些包我们在第7-9章中讨论过。例如，遗传算法、粒子群优化、Metis、基于群体的训练以及更多。

在本章中，我们将学习如何使用DEAP和Microsoft NNI包进行超参数调整，从熟悉这些包以及我们需要注意的重要模块和参数开始。我们将学习如何利用DEAP和Microsoft NNI的默认配置进行超参数调整，并讨论其他可用的配置及其使用方法。此外，我们还将讨论超参数调整方法的实现如何与我们之前章节中学到的理论相关联，因为实现中可能会有一些细微的差异或调整。

在本章结束时，你将能够理解关于DEAP和Microsoft NNI你需要知道的所有重要事项，并能够实现这些包中可用的各种超参数调整方法。你还将能够理解每个类的重要参数以及它们与我们之前章节中学到的理论之间的关系。最后，凭借前几章的知识，你还将能够理解如果出现错误或意外结果时会发生什么，并了解如何设置方法配置以匹配你的特定问题。

本章将讨论以下主要主题：

+   介绍DEAP

+   实现遗传算法

+   实现粒子群优化

+   介绍Microsoft NNI

+   实现网格搜索

+   实现随机搜索

+   实现树结构Parzen估计器

+   实现序列模型算法配置

+   实现贝叶斯优化高斯过程

+   实现Metis

+   实现模拟退火

+   实现Hyper Band

+   实现贝叶斯优化Hyper Band

+   实现基于群体的训练

# 技术要求

我们将学习如何使用DEAP和Microsoft NNI实现各种超参数调整方法。为了确保你能够复制本章中的代码示例，你需要以下条件：

+   Python 3（版本3.7或以上）

+   已安装`pandas`包（版本1.3.4或以上）

+   已安装`NumPy`包（版本1.21.2或以上）

+   已安装`SciPy`包（版本1.7.3或以上）

+   已安装`Matplotlib`包（版本3.5.0或以上）

+   已安装`scikit-learn`包（版本1.0.1或以上）

+   已安装`DEAP`包（版本1.3）

+   已安装`Hyperopt`包（版本0.1.2）

+   已安装`NNI`包（版本2.7）

+   已安装`PyTorch`包（版本1.10.0）

本章的所有代码示例都可以在GitHub上找到：[https://github.com/PacktPublishing/Hyperparameter-Tuning-with-Python/blob/main/10_Advanced_Hyperparameter-Tuning-via-DEAP-and-NNI.ipynb](https://github.com/PacktPublishing/Hyperparameter-Tuning-with-Python/blob/main/10_Advanced_Hyperparameter-Tuning-via-DEAP-and-NNI.ipynb)。

# 介绍DEAP

执行`pip install deap`命令。

DEAP允许你以非常灵活的方式构建进化算法的优化步骤。以下步骤展示了如何利用DEAP执行任何超参数调整方法。更详细的步骤，包括代码实现，将在接下来的章节中通过各种示例给出：

1.  通过`creator.create()`模块定义*类型*类。这些类负责定义在优化步骤中将使用的对象类型。

1.  定义*初始化器*以及超参数空间，并在`base.Toolbox()`容器中注册它们。初始化器负责设置在优化步骤中将使用的对象的初始值。

1.  定义*算子*并将它们注册在`base.Toolbox()`容器中。算子指的是作为优化算法一部分需要定义的进化工具或**遗传算子**（见[*第5章*](B18753_05_ePub.xhtml#_idTextAnchor047)）。例如，遗传算法中的选择、交叉和变异算子。

1.  定义目标函数并将其注册在`base.Toolbox()`容器中。

1.  定义你自己的超参数调整算法函数。

1.  通过调用定义在*步骤5*中的函数来执行超参数调整。

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

1.  在测试数据上测试最终训练好的模型。

类型类指的是在优化步骤中使用的对象类型。这些类型类是从DEAP中实现的基础类继承而来的。例如，我们可以定义我们的适应度函数类型如下：

[PRE0]

[PRE1]

`base.Fitness`类是DEAP中实现的一个基础抽象类，可以用来定义我们自己的适应度函数类型。它期望一个`weights`参数来理解我们正在处理的优化问题的类型。如果是最大化问题，那么我们必须放置一个正权重，反之亦然，对于最小化问题。请注意，它期望一个元组数据结构而不是浮点数。这是因为DEAP还允许我们将`(1.0, -1.0)`作为`weights`参数，这意味着我们有两个目标函数，我们希望第一个最大化，第二个最小化，权重相等。

`creator.create()`函数负责基于基类创建一个新的类。在前面的代码中，我们使用名称“`FitnessMax`”创建了目标函数的类型类。此`creator.create()`函数至少需要两个参数：具体来说，是新创建的类的名称和基类本身。传递给此函数的其他参数将被视为新创建类的属性。除了定义目标函数的类型外，我们还可以定义将要执行的进化算法中个体的类型。以下代码展示了如何创建从Python内置的`list`数据结构继承的个体类型，该类型具有`fitness`属性：

[PRE2]

注意，`fitness`属性的类型为`creator.FitnessMax`，这是我们之前代码中刚刚创建的类型。

DEAP中的类型定义

在DEAP中有许多定义类型类的方法。虽然我们已经讨论了最直接且可以说是最常用的类型类，但你可能会遇到需要其他类型类定义的情况。有关如何在DEAP中定义其他类型的更多信息，请参阅官方文档（[https://deap.readthedocs.io/en/master/tutorials/basic/part1.html](https://deap.readthedocs.io/en/master/tutorials/basic/part1.html)）。

一旦我们完成了将在优化步骤中使用的对象类型的定义，我们现在需要使用初始化器初始化这些对象的值，并在`base.Toolbox()`容器中注册它们。你可以将此模块视为一个盒子或容器，其中包含初始化器和将在优化步骤中使用的其他工具。以下代码展示了我们如何为个体设置随机的初始值：

[PRE3]

[PRE4]

[PRE5]

[PRE6]

[PRE7]

前面的代码展示了如何在`base.Toolbox()`容器中注册`"individual"`对象，其中每个个体的尺寸为`10`。该个体是通过重复调用`random.random`方法10次生成的。请注意，在超参数调整设置中，每个个体的`10`尺寸实际上指的是我们在空间中拥有的超参数数量。以下展示了通过`toolbox.individual()`方法调用已注册个体的输出：

[PRE8]

如你所见，`toolbox.individual()`的输出只是一个包含10个随机值的列表，因为我们已经定义`creator.Individual`从Python内置的`list`数据结构继承。此外，我们在注册个体时也调用了`tools.initRepeat`，通过`random.random`方法重复10次。

你现在可能想知道，如何使用这个`toolbox.register()`方法定义实际的超参数空间？启动一串随机值显然没有意义。我们需要知道如何定义将为每个个体配备的超参数空间。为此，我们实际上可以利用DEAP提供的另一个工具，即`tools.InitCycle`。

其中`tools.initRepeat`将只调用提供的函数`n`次，在我们之前的例子中，提供的函数是`random.random`。在这里，`tools.InitCycle`期望一个函数列表，并将这些函数调用`n`次。以下代码展示了如何定义将为每个个体配备的超参数空间的一个示例：

1.  我们需要首先注册空间中我们拥有的每个超参数及其分布。请注意，我们也可以将所有必需的参数传递给采样分布函数的`toolbox.register()`。例如，在这里，我们传递了`truncnorm.rvs()`方法的`a=0,b=0.5,loc=0.005,scale=0.01`参数：

    [PRE9]

1.  一旦我们注册了所有现有的超参数，我们可以通过使用`tools.initCycle`并只进行一次重复循环来注册个体：

    [PRE10]

以下展示了通过`toolbox.individual()`方法调用已注册个体的输出：

[PRE11]

1.  一旦我们在工具箱中注册了个体，注册一个种群就非常简单。我们只需要利用`tools.initRepeat`模块并将定义的`toolbox.individual`作为参数传递。以下代码展示了如何一般性地注册一个种群。请注意，在这里，种群只是之前定义的五个个体的列表：

    [PRE12]

以下展示了调用`toolbox.population()`方法时的输出：

[PRE13]

如前所述，`base.Toolbox()`容器不仅负责存储初始化器，还负责存储在优化步骤中将使用的其他工具。进化算法（如GA）的另一个重要构建块是遗传算子。幸运的是，DEAP已经实现了我们可以通过`tools`模块利用的各种遗传算子。以下代码展示了如何为GA注册选择、交叉和变异算子的示例（参见[*第5章*](B18753_05_ePub.xhtml#_idTextAnchor047))：

[PRE14]

[PRE15]

[PRE16]

[PRE17]

[PRE18]

[PRE19]

`tools.selTournament`选择策略通过在随机选择的`tournsize`个个体中选出最佳个体，重复`NPOP`次来实现，其中`tournsize`是参加锦标赛的个体数量，而`NPOP`是种群中的个体数量。`tools.cxBlend`交叉策略通过执行两个连续个体基因的线性组合来实现，其中线性组合的权重由`alpha`超参数控制。`tools.mutPolynomialBounded`变异策略通过将连续个体基因传递给一个预定义的多项式映射来实现。

DEAP中的进化工具

DEAP中实现了各种内置的进化工具，我们可以根据自己的需求使用，包括初始化器、交叉、变异、选择和迁移工具。有关实现工具的更多信息，请参阅官方文档([https://deap.readthedocs.io/en/master/api/tools.html](https://deap.readthedocs.io/en/master/api/tools.html))。

要将预定义的目标函数注册到工具箱中，我们只需调用相同的`toolbox.register()`方法并传递目标函数，如下面的代码所示：

[PRE20]

在这里，`obj_func`是一个Python函数，它期望接收之前定义的`individual`对象。我们将在接下来的章节中看到如何创建这样的目标函数，以及如何定义我们自己的超参数调整算法函数，当我们讨论如何在DEAP中实现GA和PSO时。

DEAP还允许我们在调用目标函数时利用我们的并行计算资源。为此，我们只需将`multiprocessing`模块注册到工具箱中，如下所示：

[PRE21]

[PRE22]

[PRE23]

一旦我们注册了`multiprocessing`模块，我们就可以在调用目标函数时简单地应用它，如下面的代码所示：

[PRE24]

在本节中，我们讨论了DEAP包及其构建块。你可能想知道如何使用DEAP提供的所有构建块构建一个真实的超参数调整方法。不用担心；在接下来的两个章节中，我们将学习如何利用所有讨论的构建块使用GA和PSO方法进行超参数调整。

# 实现遗传算法

GA是启发式搜索超参数调整组（见[*第5章*](B18753_05_ePub.xhtml#_idTextAnchor047)）的变体之一，可以通过DEAP包实现。为了展示我们如何使用DEAP包实现GA，让我们使用随机森林分类器模型和与[*第7章*](B18753_07_ePub.xhtml#_idTextAnchor062)中示例相同的数据。本例中使用的数据库是Kaggle上提供的*Banking Dataset – Marketing Targets*数据库([https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets))。

目标变量由两个类别组成，`yes`或`no`，表示银行客户是否已订阅定期存款。因此，在这个数据集上训练机器学习模型的目的是确定客户是否可能想要订阅定期存款。在数据中提供的16个特征中，有7个数值特征和9个分类特征。至于目标类分布，训练和测试数据集中都有12%是`yes`，88%是`no`。有关数据的更详细信息，请参阅[*第7章*](B18753_07_ePub.xhtml#_idTextAnchor062)。

在执行 GA 之前，让我们看看具有默认超参数值的随机森林分类器是如何工作的。如 [*第 7 章*](B18753_07_ePub.xhtml#_idTextAnchor062) 所示，我们在测试集上评估具有默认超参数值的随机森林分类器时，F1 分数大约为 `0.436`。请注意，我们仍在使用如 [*第 7 章*](B18753_07_ePub.xhtml#_idTextAnchor062) 中解释的相同的 scikit-learn 管道定义来训练和评估随机森林分类器。

以下代码展示了如何使用 DEAP 包实现 GA。您可以在 *技术要求* 部分提到的 GitHub 仓库中找到更详细的代码：

1.  通过 `creator.create()` 模块定义 GA 参数和类型类：

    [PRE25]

设置随机种子以实现可重复性：

[PRE26]

定义我们的适应度函数类型。在这里，我们正在处理一个最大化问题和一个单一目标函数，因此我们设置 `weights=(1.0,)`：

[PRE27]

定义从 Python 内置 `list` 数据结构继承的个体类型，该类型具有 `fitness` 作为其属性：

[PRE28]

1.  定义初始化器以及超参数空间并将它们注册在 `base.Toolbox()` 容器中。

初始化工具箱：

[PRE29]

定义超参数的命名：

[PRE30]

注册空间中的每个超参数及其分布：

[PRE31]

通过使用 `tools.initCycle` 仅进行一次循环重复来注册个体：

[PRE32]

注册种群：

[PRE33]

1.  定义操作符并将它们注册在 `base.Toolbox()` 容器中。

注册选择策略：

[PRE34]

注册交叉策略：

[PRE35]

定义一个自定义变异策略。请注意，DEAP 中实现的全部变异策略实际上并不适合超参数调整目的，因为它们只能用于浮点或二进制值，而大多数情况下，我们的超参数空间将是一组真实和离散超参数的组合。以下函数展示了如何实现这样的自定义变异策略。您可以遵循相同的结构来满足您的需求：

[PRE36]

注册自定义变异策略：

[PRE37]

1.  定义目标函数并将其注册在 `base.Toolbox()` 容器中：

    [PRE38]

注册目标函数：

[PRE39]

1.  定义具有并行处理的遗传算法：

    [PRE40]

注册 `multiprocessing` 模块：

[PRE41]

定义空数组以存储每个试验中目标函数得分的最佳值和平均值：

[PRE42]

定义一个 `HallOfFame` 类，该类负责在种群中存储最新的最佳个体（超参数集）：

[PRE43]

定义初始种群：

[PRE44]

开始 GA 迭代：

[PRE45]

选择下一代个体/孩子/后代。

[PRE46]

复制选定的个体。

[PRE47]

在后代上应用交叉：

[PRE48]

在后代上应用变异。

[PRE49]

评估具有无效适应度的个体：

[PRE50]

种群完全由后代取代。

[PRE51]

1.  通过运行定义的算法在*步骤5*中执行超参数调整。在运行GA之后，我们可以根据以下代码获取最佳超参数集：

    [PRE52]

根据前面的代码，我们得到以下结果：

[PRE53]

我们也可以根据以下代码绘制试验历史或收敛图：

[PRE54]

根据前面的代码，以下图生成。如图所示，目标函数得分或适应度得分在整个试验次数中都在增加，因为种群被更新为改进的个体：

![图10.1 – 遗传算法收敛图]

![img/B18753_10_001.jpg]

图10.1 – 遗传算法收敛图

1.  使用找到的最佳超参数集在全部训练数据上训练模型：

    [PRE55]

1.  在测试数据上测试最终训练的模型：

    [PRE56]

根据前面的代码，当使用最佳超参数集在测试集上测试我们最终的训练随机森林模型时，F1分数大约为`0.608`。

在本节中，我们学习了如何使用DEAP包实现遗传算法（GA），从定义必要的对象开始，到使用并行处理和自定义变异策略定义GA过程，再到绘制试验历史和测试测试集中最佳超参数集。在下一节中，我们将学习如何使用DEAP包实现PSO超参数调整方法。

# 实现粒子群优化

PSO也是启发式搜索超参数调整组（见[*第5章*](B18753_05_ePub.xhtml#_idTextAnchor047)）的一种变体，可以使用DEAP包实现。我们仍将使用上一节中的相同示例来查看我们如何使用DEAP包实现PSO。

以下代码显示了如何使用DEAP包实现PSO。你可以在*技术要求*部分提到的GitHub仓库中找到更详细的代码：

1.  通过`creator.create()`模块定义PSO参数和类型类：

    [PRE57]

设置随机种子以实现可重复性：

[PRE58]

定义我们的适应度函数的类型。在这里，我们正在处理一个最大化问题和一个单一目标函数，这就是为什么我们设置`weights=(1.0,)`：

[PRE59]

定义从Python内置的`list`数据结构继承的粒子类型，该结构具有`fitness`、`speed`、`smin`、`smax`和`best`作为其属性。这些属性将在稍后更新每个粒子的位置时被利用（见[*第5章*](B18753_05_ePub.xhtml#_idTextAnchor047)）：

[PRE60]

1.  定义初始化器以及超参数空间，并在`base.Toolbox()`容器中注册它们。

初始化工具箱：

[PRE61]

定义超参数的命名：

[PRE62]

在空间中注册我们拥有的每个超参数及其分布。记住，PSO只与数值类型超参数一起工作。这就是为什么我们将`"model__criterion"`和`"model__class_weight"`超参数编码为整数：

[PRE63]

通过使用`tools.initCycle`仅进行一次重复循环来注册个体。注意，我们还需要将`speed`、`smin`和`smax`值分配给每个个体。为此，让我们定义一个名为`generate`的函数：

[PRE64]

通过使用`tools.initCycle`仅进行一次重复循环来注册个体：

[PRE65]

注册种群：

[PRE66]

1.  定义操作符并将它们注册到`base.Toolbox()`容器中。PSO中的主要操作符是粒子的位置更新操作符，该操作符在`updateParticle`函数中定义如下：

    [PRE67]

注册操作符。注意，`is_int`属性负责标记哪个超参数具有整数值类型：

[PRE68]

1.  定义目标函数并将其注册到`base.Toolbox()`容器中。注意，我们还在目标函数中解码了`"model__criterion"`和`"model__class_weight"`超参数：

    [PRE69]

注册目标函数：

[PRE70]

1.  定义具有并行处理的PSO：

    [PRE71]

注册`multiprocessing`模块：

[PRE72]

定义空数组以存储每个试验中目标函数分数的最佳和平均值：

[PRE73]

定义一个`HallOfFame`类，该类负责存储种群中的最新最佳个体（超参数集）：

[PRE74]

定义初始种群：

[PRE75]

开始PSO迭代：

[PRE76]

1.  通过运行第5步中定义的算法来执行超参数调整。在运行PSO之后，我们可以根据以下代码获取最佳超参数集。注意，在将它们传递给最终模型之前，我们需要解码`"model__criterion"`和`"model__class_weight"`超参数：

    [PRE77]

根据前面的代码，我们得到以下结果：

[PRE78]

1.  使用找到的最佳超参数集在全部训练数据上训练模型：

    [PRE79]

1.  在测试数据上测试最终训练好的模型：

    [PRE80]

根据前面的代码，我们在测试最终训练好的随机森林模型时，在测试集上获得了大约`0.569`的F1分数，该模型使用了最佳的超参数集。

在本节中，我们学习了如何使用DEAP包实现PSO，从定义必要的对象开始，将分类超参数编码为整数，并使用并行处理定义优化过程，直到在测试集上测试最佳超参数集。在下一节中，我们将开始学习另一个名为NNI的超参数调整包，该包由微软开发。

# 介绍微软NNI

`pip install nni`命令。

虽然NNI指的是*神经网络智能*，但它实际上支持包括但不限于scikit-learn、XGBoost、LightGBM、PyTorch、TensorFlow、Caffe2和MXNet在内的多个机器学习框架。

NNI实现了许多超参数调优方法；其中一些是内置的，而另一些则是从其他包如`Hyperopt`（见[*第8章*](B18753_08_ePub.xhtml#_idTextAnchor074)）和`SMAC3`中封装的。在这里，NNI中的超参数调优方法被称为**调优器**。由于调优器种类繁多，我们不会讨论NNI中实现的所有调优器。我们只会讨论在第3章至第6章中讨论过的调优器。除了调优器之外，一些超参数调优方法，如Hyper Band和BOHB，在NNI中被视为**顾问**。

NNI中的可用调优器

要查看NNI中所有可用调优器的详细信息，请参阅官方文档页面([https://nni.readthedocs.io/en/stable/hpo/tuners.html](https://nni.readthedocs.io/en/stable/hpo/tuners.html))。

与我们之前讨论的其他超参数调优包不同，在NNI中，我们必须准备一个包含模型定义的Python脚本，然后才能从笔记本中运行超参数调优过程。此外，NNI还允许我们从命令行工具中运行超参数调优实验，在那里我们需要定义几个其他附加文件来存储超参数空间信息和其他配置。

以下步骤展示了如何使用纯Python代码通过NNI执行任何超参数调优过程：

1.  在脚本中准备要调优的模型，例如，`model.py`。此脚本应包括模型架构定义、数据集加载函数、训练函数和测试函数。它还必须包括三个NNI API调用，如下所示：

    +   `nni.get_next_parameter()` 负责收集特定试验中要评估的超参数。

    +   `nni.report_intermediate_result()` 负责在每次训练迭代（epoch或步骤）中报告评估指标。请注意，此API调用不是强制的；如果您无法从您的机器学习框架中获取中间评估指标，则不需要此API调用。

    +   `nni.report_final_result()` 负责在训练过程完成后报告最终评估指标分数。

1.  定义超参数空间。NNI期望超参数空间以Python字典的形式存在，其中第一级键存储超参数的名称。第二级键存储采样分布的类型和超参数值范围。以下是如何以预期格式定义超参数空间的示例：

    [PRE81]

关于NNI的更多信息

关于NNI支持的采样分布的更多信息，请参阅官方文档([https://nni.readthedocs.io/en/latest/hpo/search_space.html](https://nni.readthedocs.io/en/latest/hpo/search_space.html))。

1.  接下来，我们需要通过`Experiment`类设置实验配置。以下展示了在我们可以运行超参数调整过程之前设置几个配置的步骤。

加载`Experiment`类。在这里，我们使用的是`'local'`实验模式，这意味着所有训练和超参数调整过程都将在我们的本地计算机上完成。NNI允许我们在各种平台上运行训练过程，包括但不限于**Azure Machine Learning**（**AML**）、Kubeflow和OpenAPI。更多信息，请参阅官方文档([https://nni.readthedocs.io/en/latest/reference/experiment_config.html](https://nni.readthedocs.io/en/latest/reference/experiment_config.html))：

[PRE82]

设置试验代码配置。在这里，我们需要指定运行在*步骤1*中定义的脚本的命令和脚本的相对路径。以下展示了如何设置试验代码配置的示例：

[PRE83]

设置超参数空间配置。要设置超参数空间配置，我们只需将定义的超参数空间传递到*步骤2*。以下代码展示了如何进行操作：

[PRE84]

设置要使用的超参数调整算法。以下展示了如何将TPE作为超参数调整算法应用于最大化问题的示例：

[PRE85]

设置试验次数和并发进程数。NNI允许我们设置在单次运行中同时评估多少个超参数集。以下代码展示了如何将试验次数设置为50，这意味着在特定时间将同时评估五个超参数集：

[PRE86]

值得注意的是，NNI还允许你根据时间长度而不是试验次数来定义停止标准。以下代码展示了你如何将实验时间限制为1小时：

[PRE87]

如果你没有提供`max_trial_number`和`max_experiment_duration`两个参数，那么实验将永远运行，直到你通过*Ctrl + C*命令强制停止它。

1.  运行超参数调整实验。要运行实验，我们可以在`Experiment`类上简单地调用`run`方法。在这里，我们还需要选择要使用的端口。我们可以通过启动的Web门户查看实验状态和各种有趣的统计数据。以下代码展示了如何在`local`模式下在端口`8080`上运行实验，这意味着你可以在`http://localhost:8080`上打开Web门户：

    [PRE88]

`run`方法有两个可用的布尔参数，即`wait_completion`和`debug`。当我们设置`wait_completion=True`时，我们无法在实验完成或发现错误之前运行笔记本中的其他单元格。`debug`参数使我们能够选择是否以调试模式启动实验。

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

1.  在测试数据上测试最终训练好的模型。

NNI Web Portal

关于Web门户中可用的更多功能，请参阅官方文档（[https://nni.readthedocs.io/en/stable/experiment/web_portal/web_portal.html](https://nni.readthedocs.io/en/stable/experiment/web_portal/web_portal.html)）。注意，我们将在[*第13章*](B18753_13_ePub.xhtml#_idTextAnchor125)中更详细地讨论Web门户，*跟踪超参数调整实验*。

如果你更喜欢使用命令行工具，以下步骤展示了如何使用命令行工具、JSON和YAML配置文件执行任何超参数调整流程：

1.  在脚本中准备要调整的模型。这一步骤与使用纯Python代码进行NNI超参数调整的前一个流程完全相同。

1.  定义超参数空间。超参数空间的预期格式与使用纯Python代码进行任何超参数调整流程的流程完全相同。然而，在这里，我们需要将Python字典存储在一个JSON文件中，例如，`hyperparameter_space.json`。

1.  通过`config.yaml`文件设置实验配置。需要设置的配置基本上与使用纯Python代码的NNI流程相同。然而，这里不是通过Python类来配置实验，而是将所有配置细节存储在一个单独的YAML文件中。以下是一个YAML文件示例：

    [PRE89]

1.  运行超参数调整实验。要运行实验，我们可以简单地调用`nnictl create`命令。以下代码展示了如何使用该命令在`local`的`8080`端口上运行实验：

    [PRE90]

实验完成后，你可以通过`nnictl stop`命令轻松停止进程。

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

1.  在测试数据上测试最终训练好的模型。

各种机器学习框架的示例

你可以在官方文档中找到使用你喜欢的机器学习框架通过NNI执行超参数调整的所有示例（[https://github.com/microsoft/nni/tree/master/examples/trials](https://github.com/microsoft/nni/tree/master/examples/trials)）。

scikit-nni

此外，还有一个名为`scikit-nni`的包，它将自动生成所需的`config.yml`和`search-space.json`，并根据你的自定义需求构建`scikit-learn`管道。有关此包的更多信息，请参阅官方仓库（[https://github.com/ksachdeva/scikit-nni](https://github.com/ksachdeva/scikit-nni)）。

除了调优器或超参数调优算法之外，NNI 还提供了 `nni.report_intermediate_result()` API 调用。NNI 中只有两个内置评估器：*中值停止* 和 *曲线拟合*。第一个评估器将在任何步骤中，只要某个超参数集的表现不如中值，就会停止实验。后者评估器将在学习曲线可能收敛到次优结果时停止实验。

在 NNI 中设置评估器非常简单。您只需在 `Experiment` 类或 `config.yaml` 文件中添加配置即可。以下代码展示了如何在 `Experiment` 类上配置中值停止评估器：

[PRE91]

NNI 中的自定义算法

NNI 还允许我们定义自己的自定义调优器和评估器。为此，您需要继承基类 `Tuner` 或 `Assessor`，编写几个必需的函数，并在 `Experiment` 类或 `config.yaml` 文件中添加更多详细信息。有关如何定义自己的自定义调优器和评估器的更多信息，请参阅官方文档（[https://nni.readthedocs.io/en/stable/hpo/custom_algorithm.html](https://nni.readthedocs.io/en/stable/hpo/custom_algorithm.html)）。

在本节中，我们讨论了 NNI 包及其如何进行一般性的超参数调优实验。在接下来的章节中，我们将学习如何使用 NNI 实现各种超参数调优算法。

# 实现网格搜索

网格搜索是 NNI 包可以实现的穷举搜索超参数调优组（参见 [*第 3 章*](B18753_03_ePub.xhtml#_idTextAnchor031)）的一种变体。为了向您展示我们如何使用 NNI 包实现网格搜索，我们将使用与上一节示例中相同的数据和管道。然而，在这里，我们将定义一个新的超参数空间，因为 NNI 只支持有限类型的采样分布。

以下代码展示了如何使用 NNI 包实现网格搜索。在这里，我们将使用 NNI 命令行工具（**nnictl**）而不是纯 Python 代码。更详细的代码可以在 *技术要求* 部分提到的 GitHub 仓库中找到：

1.  在脚本中准备要调优的模型。在这里，我们将脚本命名为 `model.py`。该脚本中定义了几个函数，包括 `load_data`、`get_default_parameters`、`get_model` 和 `run`。

`load_data` 函数加载原始数据并将其分为训练数据和测试数据。此外，它还负责返回数值和分类列名的列表：

[PRE92]

`get_default_parameters` 函数返回实验中使用的默认超参数值：

[PRE93]

`get_model` 函数定义了在此示例中使用的 `sklearn` 管道：

[PRE94]

为数值特征启动归一化预处理。

[PRE95]

为分类特征启动 One-Hot-Encoding 预处理。

[PRE96]

创建 ColumnTransformer 类以将每个预处理程序委托给相应的特征。

[PRE97]

创建预处理器和模型的 Pipeline。

[PRE98]

设置超参数值。

[PRE99]

`run` 函数负责训练模型并获取交叉验证分数：

[PRE100]

最后，我们可以在同一脚本中调用这些函数：

[PRE101]

1.  在名为 `hyperparameter_space.json` 的 JSON 文件中定义超参数空间：

    [PRE102]

1.  通过 `config.yaml` 文件设置实验配置：

    [PRE103]

1.  运行超参数调优实验。我们可以通过启动的网络门户查看实验状态和各种有趣的统计数据。以下代码展示了如何在 `local` 模式下通过端口 `8080` 运行实验，这意味着您可以在 `http://localhost:8080` 上打开网络门户：

    [PRE104]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。要获取最佳超参数集，您可以访问网络门户并在 *概览* 选项卡中查看。

根据在 *Top trials* 选项卡中显示的实验结果，以下是从实验中找到的最佳超参数值。注意，我们将在 [*第 13 章*](B18753_13_ePub.xhtml#_idTextAnchor125) *跟踪超参数调优实验* 中更详细地讨论网络门户：

[PRE105]

我们现在可以在全部训练数据上训练模型：

[PRE106]

1.  在测试数据上测试最终训练好的模型：

    [PRE107]

根据前面的代码，当我们在测试集上使用最佳超参数集测试我们最终的训练 Random Forest 模型时，F1 分数大约为 `0.517`。

在本节中，我们学习了如何通过 `nnictl` 使用 NNI 包实现网格搜索。在下一节中，我们将学习如何通过纯 Python 代码使用 NNI 实现 Random Search。

# 实现 Random Search

随机搜索是穷举搜索超参数调优组（见 [*第 3 章*](B18753_03_ePub.xhtml#_idTextAnchor031)）的一种变体，NNI 包可以实施。让我们使用与上一节示例中相同的数据、管道和超参数空间，向您展示如何使用纯 Python 代码通过 NNI 实现 Random Search。

以下代码展示了如何使用 NNI 包实现随机搜索。在这里，我们将使用纯 Python 代码而不是像上一节那样使用 `nnictl`。您可以在 *技术要求* 部分提到的 GitHub 仓库中找到更详细的代码：

1.  在脚本中准备要调优的模型。我们将使用与上一节相同的 `model.py` 脚本。

1.  以 Python 字典的形式定义超参数空间：

    [PRE108]

1.  通过 `Experiment` 类设置实验配置。注意，对于随机搜索调优器，只有一个参数，即随机的 `seed` 参数：

    [PRE109]

1.  运行超参数调优实验：

    [PRE110]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

[PRE111]

1.  根据前面的代码，我们得到了以下结果：

    [PRE112]

我们现在可以在全部训练数据上训练模型：

[PRE113]

1.  在测试数据上测试最终训练好的模型：

    [PRE114]

根据前面的代码，当使用最佳超参数集在测试集上测试我们最终训练的随机森林模型时，F1分数大约为`0.597`。

在本节中，我们学习了如何使用纯Python代码通过NNI实现随机搜索。在下一节中，我们将学习如何通过纯Python代码使用NNI实现树结构帕累托估计器。

# 实现树结构帕累托估计器

**树结构帕累托估计器**（**TPEs**）是贝叶斯优化超参数调整组（见[*第4章*](B18753_04_ePub.xhtml#_idTextAnchor036)）中NNI包可以实现的变体之一。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯Python代码实现TPE与NNI。

以下代码展示了如何使用纯Python代码通过NNI包实现TPE。你可以在*技术要求*节中提到的GitHub仓库中找到更详细的代码：

1.  在脚本中准备要调优的模型。我们将使用与上一节相同的`model.py`脚本。

1.  以Python字典的形式定义超参数空间。我们将使用与上一节相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，TPE调整器有三个参数：`optimize_mode`、`seed`和`tpe_args`。有关TPE调整器参数的更多信息，请参阅官方文档页面（[https://nni.readthedocs.io/en/stable/reference/hpo.html#tpe-tuner](https://nni.readthedocs.io/en/stable/reference/hpo.html#tpe-tuner)）：

    [PRE115]

1.  运行超参数调整实验：

    [PRE116]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

[PRE117]

根据前面的代码，我们得到以下结果：

[PRE118]

我们现在可以在全部训练数据上训练模型：

[PRE119]

在训练数据上拟合管道。

[PRE120]

1.  在测试数据上测试最终训练的模型：

    [PRE121]

根据前面的代码，当使用最佳超参数集在测试集上测试我们最终训练的随机森林模型时，F1分数大约为`0.618`。

在本节中，我们学习了如何使用纯Python代码通过NNI实现TPE。在下一节中，我们将学习如何通过纯Python代码使用NNI实现序列模型算法配置。

# 实现序列模型算法配置

`pip install "nni[SMAC]"`. 让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯Python代码实现SMAC与NNI。

以下代码展示了如何使用纯Python代码通过NNI包实现SMAC。你可以在*技术要求*节中提到的GitHub仓库中找到更详细的代码：

1.  在脚本中准备要调优的模型。我们将使用与上一节相同的`model.py`脚本。

1.  以Python字典的形式定义超参数空间。我们将使用与上一节相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，SMAC调优器有两个参数：`optimize_mode`和`config_dedup`。有关SMAC调优器参数的更多信息，请参阅官方文档页面([https://nni.readthedocs.io/en/stable/reference/hpo.html#smac-tuner](https://nni.readthedocs.io/en/stable/reference/hpo.html#smac-tuner))：

    [PRE122]

1.  运行超参数调优实验：

    [PRE123]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳的超参数组合：

[PRE124]

根据前面的代码，我们得到了以下结果：

[PRE125]

我们现在可以在全部训练数据上训练模型：

[PRE126]

1.  在测试数据上测试最终训练好的模型：

    [PRE127]

根据前面的代码，我们在测试集上使用最佳超参数组合测试最终训练好的随机森林模型时，F1分数大约为`0.619`。

在本节中，我们学习了如何使用纯Python代码通过NNI实现SMAC。在下一节中，我们将学习如何通过纯Python代码使用NNI实现贝叶斯优化高斯过程。

# 实现贝叶斯优化高斯过程

**贝叶斯优化高斯过程**（**BOGP**）是贝叶斯优化超参数调优组（见[*第4章*](B18753_04_ePub.xhtml#_idTextAnchor036)）的变体之一，NNI包可以实现。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯Python代码通过NNI实现BOGP。

以下代码展示了如何使用NNI包通过纯Python代码实现BOGP。更详细的代码可以在*技术要求*部分提到的GitHub仓库中找到：

1.  在脚本中准备要调优的模型。在这里，我们将使用一个新的脚本，名为`model_numeric.py`。在这个脚本中，我们为非数值超参数添加了一个映射，因为BOGP只能处理数值超参数：

    [PRE128]

1.  以Python字典的形式定义超参数空间。我们将使用与上一节类似的超参数空间，唯一的区别在于非数值超参数。在这里，所有非数值超参数都被编码为整数值类型：

    [PRE129]

1.  通过`Experiment`类设置实验配置。请注意，BOGP调优器有九个参数：`optimize_mode`、`utility`、`kappa`、`xi`、`nu`、`alpha`、`cold_start_num`、`selection_num_warm_up`和`selection_num_starting_points`。有关BOGP调优器参数的更多信息，请参阅官方文档页面([https://nni.readthedocs.io/en/stable/reference/hpo.html#gp-tuner](https://nni.readthedocs.io/en/stable/reference/hpo.html#gp-tuner))：

    [PRE130]

1.  运行超参数调优实验：

    [PRE131]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

[PRE132]

基于前面的代码，我们得到了以下结果：

[PRE133]

我们现在可以在全部训练数据上训练模型：

[PRE134]

在训练数据上拟合管道。

[PRE135]

1.  在测试数据上测试最终训练好的模型：

    [PRE136]

基于前面的代码，我们在测试集上使用最佳超参数集测试最终训练好的随机森林模型时，F1分数大约为`0.619`。

在本节中，我们学习了如何使用纯Python代码通过NNI实现BOGP。在下一节中，我们将学习如何通过纯Python代码使用NNI实现Metis。

# 实现Metis

**Metis**是贝叶斯优化超参数调整组（参见[*第4章*](B18753_04_ePub.xhtml#_idTextAnchor036)）的一个变体，NNI包可以实现。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯Python代码实现Metis。

以下代码展示了如何使用纯Python代码通过NNI包实现Metis。你可以在*技术要求*部分提到的GitHub仓库中找到更详细的代码：

1.  在脚本中准备要调整的模型。这里，我们将使用与上一节相同的脚本`model_numeric.py`，因为Metis只能与数值超参数一起工作。

1.  以Python字典的形式定义超参数空间。我们将使用与上一节相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，Metis调整器有六个参数：`optimize_mode`、`no_resampling`、`no_candidates`、`selection_num_starting_points`、`cold_start_num`和`exploration_probability`。有关Metis调整器参数的更多信息，请参阅官方文档页面([https://nni.readthedocs.io/en/stable/reference/hpo.html#metis-tuner](https://nni.readthedocs.io/en/stable/reference/hpo.html#metis-tuner))：

    [PRE137]

1.  运行超参数调整实验：

    [PRE138]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

[PRE139]

基于前面的代码，我们得到了以下结果：

[PRE140]

我们现在可以在全部训练数据上训练模型：

[PRE141]

1.  在测试数据上测试最终训练好的模型：

    [PRE142]

基于前面的代码，我们在测试集上使用最佳超参数集测试最终训练好的随机森林模型时，F1分数大约为`0.590`。

在本节中，我们学习了如何使用纯Python代码通过NNI实现Metis。在下一节中，我们将学习如何通过纯Python代码使用NNI实现模拟退火。

# 实现模拟退火

模拟退火是启发式搜索超参数调整组（参见[*第五章*](B18753_05_ePub.xhtml#_idTextAnchor047)）的一种变体，NNI包可以实现。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯Python代码实现模拟退火。

以下代码展示了如何使用纯Python代码通过NNI包实现模拟退火。你可以在*技术要求*部分提到的GitHub仓库中找到更详细的代码：

1.  在脚本中准备要调整的模型。我们将使用与*实现网格搜索*部分相同的`model.py`脚本。

1.  以Python字典的形式定义超参数空间。我们将使用与*实现网格搜索*部分相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，对于模拟退火调整器有一个参数，即`optimize_mode`：

    [PRE143]

1.  运行超参数调整实验：

    [PRE144]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳的超参数集：

[PRE145]

根据前面的代码，我们得到了以下结果：

[PRE146]

我们现在可以使用全部训练数据来训练模型：

[PRE147]

1.  在测试数据上测试最终训练好的模型：

    [PRE148]

根据前面的代码，当我们在测试集上使用最佳超参数集测试最终训练好的随机森林模型时，F1分数大约为`0.600`。

在本节中，我们学习了如何使用纯Python代码通过NNI实现模拟退火。在下一节中，我们将学习如何通过纯Python代码实现Hyper Band。

# 实现Hyper Band

Hyper Band是多保真优化超参数调整组（参见[*第六章*](B18753_06_ePub.xhtml#_idTextAnchor054)）的一种变体，NNI包可以实现。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯Python代码实现Hyper Band。

以下代码展示了如何使用纯Python代码通过NNI包实现Hyper Band。你可以在*技术要求*部分提到的GitHub仓库中找到更详细的代码：

1.  在脚本中准备要调整的模型。在这里，我们将使用一个名为`model_advisor.py`的新脚本。在这个脚本中，我们利用`nni.get_next_parameter()`输出的`TRIAL_BUDGET`值来更新`'model__n_estimators'`超参数。

1.  以Python字典的形式定义超参数空间。我们将使用与*实现网格搜索*部分类似的超参数空间，但我们将移除`'model__n_estimators'`超参数，因为它将成为Hyper Band的预算定义：

    [PRE149]

1.  通过`Experiment`类设置实验配置。请注意，Hyper Band顾问有四个参数：`optimize_mode`、`R`、`eta`和`exec_mode`。请参考官方文档页面以获取有关Hyper Band顾问参数的更多信息（[https://nni.readthedocs.io/en/latest/reference/hpo.html](https://nni.readthedocs.io/en/latest/reference/hpo.html#hyperband-tuner)）：

    [PRE150]

1.  运行超参数调优实验：

    [PRE151]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

[PRE152]

基于前面的代码，我们得到以下结果：

[PRE153]

我们现在可以在全部训练数据上训练模型：

[PRE154]

在训练数据上拟合管道。

[PRE155]

1.  在测试数据上测试最终训练的模型：

    [PRE156]

基于前面的代码，我们在使用最佳超参数集在测试集上测试最终训练的随机森林模型时，F1分数大约为`0.593`。

在本节中，我们学习了如何使用纯Python代码通过NNI实现Hyper Band。在下一节中，我们将学习如何通过纯Python代码使用NNI实现贝叶斯优化超参数搜索。

# 实现贝叶斯优化超参数搜索

**贝叶斯优化超参数搜索**（**BOHB**）是NNI包可以实现的Multi-Fidelity Optimization超参数调优组的一种变体（参见[*第6章*](B18753_06_ePub.xhtml#_idTextAnchor054)）。请注意，要在NNI中使用BOHB，我们需要使用以下命令安装额外的依赖项：

[PRE157]

让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯Python代码实现BOHB（贝叶斯优化超参数搜索）。

以下代码展示了如何使用纯Python代码通过NNI包实现Hyper Band。更详细的代码可以在*技术要求*部分提到的GitHub仓库中找到：

1.  在脚本中准备要调优的模型。我们将使用与上一节相同的`model_advisor.py`脚本。

1.  以Python字典的形式定义超参数空间。我们将使用与上一节相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，BOHB顾问有11个参数：`optimize_mode`、`min_budget`、`max_budget`、`eta`、`min_points_in_model`、`top_n_percent`、`num_samples`、`random_fraction`、`bandwidth_factor`、`min_bandwidth`和`config_space`。请参考官方文档页面以获取有关Hyper Band顾问参数的更多信息（[https://nni.readthedocs.io/en/latest/reference/hpo.html#bohb-tuner](https://nni.readthedocs.io/en/latest/reference/hpo.html#bohb-tuner)）：

    [PRE158]

1.  运行超参数调优实验：

    [PRE159]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

[PRE160]

基于前面的代码，我们得到以下结果：

[PRE161]

我们现在可以在全部训练数据上训练模型：

[PRE162]

1.  在测试数据上测试最终训练好的模型：

    [PRE163]

基于前面的代码，我们在测试集上使用最佳超参数集测试最终训练好的随机森林模型时，F1分数大约为`0.617`。

在本节中，我们学习了如何使用纯 Python 代码实现 NNI 的贝叶斯优化超参数搜索。在下一节中，我们将学习如何通过 `nnictl` 使用 NNI 实现 Population-Based Training。

# 实现基于群体的训练

**基于群体的训练**（**PBT**）是启发式搜索超参数调整组（参见 [*第 5 章*](B18753_05_ePub.xhtml#_idTextAnchor047)）的变体之一，NNI 包可以实现。为了向您展示如何使用纯 Python 代码通过 NNI 实现 PBT，我们将使用 NNI 包提供的相同示例。在这里，我们使用了 MNIST 数据集和卷积神经网络模型。我们将使用 PyTorch 来实现神经网络模型。有关 NNI 提供的代码示例的详细信息，请参阅 NNI GitHub 仓库（[https://github.com/microsoft/nni/tree/1546962f83397710fe095538d052dc74bd981707/examples/trials/mnist-pbt-tuner-pytorch](https://github.com/microsoft/nni/tree/1546962f83397710fe095538d052dc74bd981707/examples/trials/mnist-pbt-tuner-pytorch)）。

MNIST 数据集

MNIST 是一个手写数字数据集，这些数字已经被标准化并居中在一个固定大小的图像中。在这里，我们将使用 PyTorch 包直接提供的 MNIST 数据集（[https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST)）。

以下代码展示了如何使用 NNI 包实现 PBT。在这里，我们将使用 `nnictl` 而不是使用纯 Python 代码。更详细的代码可以在 *技术要求* 部分提到的 GitHub 仓库中找到：

1.  在脚本中准备要调整的模型。在这里，我们将使用来自 NNI GitHub 仓库的相同的 `mnist.py` 脚本。请注意，我们将脚本保存为新的名称：`model_pbt.py`。

1.  在名为 `hyperparameter_space_pbt.json` 的 JSON 文件中定义超参数空间。在这里，我们将使用来自 NNI GitHub 仓库的相同的 `search_space.json` 文件。

1.  通过 `config_pbt.yaml` 文件设置实验配置。请注意，PBT 调优器有六个参数：`optimize_mode`、`all_checkpoint_dir`、`population_size`、`factor`、`resample_probability` 和 `fraction`。有关 PBT 调优器参数的更多信息，请参阅官方文档页面（[https://nni.readthedocs.io/en/latest/reference/hpo.html#pbt-tuner](https://nni.readthedocs.io/en/latest/reference/hpo.html#pbt-tuner)）：

    [PRE164]

1.  运行超参数调优实验。我们可以通过启动的Web门户查看实验状态和各种有趣的统计数据。以下代码展示了如何在`local`模式下运行端口`8080`上的实验，这意味着你可以在`http://localhost:8080`上打开Web门户：

    [PRE165]

在本节中，我们学习了如何通过`nnictl`使用NNI官方文档中提供的相同示例来实现基于群体的训练。

# 摘要

在本章中，我们学习了关于DEAP和Microsoft NNI包的所有重要内容。我们还学习了如何借助这些包实现各种超参数调优方法，以及理解每个类的重要参数以及它们与我们之前章节中学到的理论之间的关系。从现在开始，你应该能够利用这些包来实现你选择的超参数调优方法，并最终提高你的机器学习模型性能。凭借第3章至第6章的知识，你还将能够调试代码，如果出现错误或意外结果，并且能够制定自己的实验配置以匹配你的特定问题。

在下一章中，我们将学习几种流行算法的超参数。每个算法都会有广泛的解释，包括但不限于每个超参数的定义、当每个超参数的值发生变化时将产生什么影响，以及基于影响的超参数优先级列表。
