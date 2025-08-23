# *第9章*: 通过Optuna进行超参数调优

`scikit-optimize`。

在本章中，您将了解`Optuna`包，从其众多功能开始，学习如何利用它进行超参数调优，以及您需要了解的关于`Optuna`的所有其他重要事项。我们不仅将学习如何利用`Optuna`及其默认配置进行超参数调优，还将讨论可用的配置及其用法。此外，我们还将讨论超参数调优方法的实现与我们在前几章中学到的理论之间的关系，因为实现中可能会有一些细微的差异或调整。

到本章结束时，您将能够了解关于`Optuna`的所有重要事项，并实现该包中提供的各种超参数调优方法。您还将能够理解每个类的重要参数以及它们与我们之前章节中学到的理论之间的关系。最后，凭借前几章的知识，您还将能够理解如果出现错误或意外结果时会发生什么，并了解如何设置方法配置以匹配您特定的难题。

本章将讨论以下主要主题：

+   介绍Optuna

+   实现TPE

+   实现随机搜索

+   实现网格搜索

+   实现模拟退火

+   实现逐次减半

+   实现Hyperband

# 技术要求

我们将学习如何使用`Optuna`实现各种超参数调优方法。为确保您能够重现本章中的代码示例，您需要以下条件：

+   Python 3（版本3.7或更高）

+   安装`pandas`包（版本1.3.4或更高）

+   安装`NumPy`包（版本1.21.2或更高）

+   安装`Matplotlib`包（版本3.5.0或更高）

+   安装`scikit-learn`包（版本1.0.1或更高）

+   安装`Tensorflow`包（版本2.4.1或更高）

+   安装`Optuna`包（版本2.10.0或更高）

本章的所有代码示例都可以在GitHub上找到：[https://github.com/PacktPublishing/Hyperparameter-Tuning-with-Python](https://github.com/PacktPublishing/Hyperparameter-Tuning-with-Python)。

# 介绍Optuna

`Optuna`是一个Python超参数调优包，提供了多种超参数调优方法的实现，例如网格搜索、随机搜索、树结构帕累托估计器（TPE）等。与假设我们始终在处理最小化问题（参见[*第8章*](B18753_08_ePub.xhtml#_idTextAnchor074)*，通过Hyperopt进行超参数调优*）的`Hyperopt`不同，我们可以告诉`Optuna`我们正在处理哪种优化问题：最小化或最大化。

`Optuna`有两个主要类，即**采样器**和**剪枝器**。采样器负责执行超参数调整优化，而剪枝器负责根据报告的值判断是否应该剪枝试验。换句话说，剪枝器就像*早期停止方法*，当我们认为继续过程没有额外好处时，我们将停止超参数调整迭代。

内置的采样器实现包括我们在第3章到第4章中学到的几种超参数调整方法，即网格搜索、随机搜索和TPE，以及本书范围之外的其他方法，例如CMA-ES、NSGA-II等。我们还可以定义自己的自定义采样器，例如模拟退火（SA），这将在下一节中讨论。此外，`Optuna`还允许我们集成来自另一个包的采样器，例如来自`scikit-optimize`（`skopt`）包，在那里我们可以利用许多基于贝叶斯优化的方法。

Optuna的集成

除了`skopt`之外，`Optuna`还提供了许多其他集成，包括但不限于`scikit-learn`、`Keras`、`PyTorch`、`XGBoost`、`LightGBM`、`FastAI`、`MLflow`等。有关可用集成的更多信息，请参阅官方文档([https://optuna.readthedocs.io/en/v2.10.0/reference/integration.html](https://optuna.readthedocs.io/en/v2.10.0/reference/integration.html))。

对于剪枝器，`Optuna`提供了基于统计和基于多保真优化（MFO）的方法。对于基于统计的组，有`MedianPruner`、`PercentilePruner`和`ThresholdPruner`。`MedianPruner`将在当前试验的最佳中间结果比前一个试验的结果中位数更差时剪枝。`PercentilePruner`将在当前最佳中间值是前一个试验的底部百分位数之一时进行剪枝。`ThresholdPruner`将简单地在任何预定义的阈值满足时进行剪枝。`Optuna`中实现的基于MFO的剪枝器是`SuccessiveHalvingPruner`和`HyperbandPruner`。两者都将资源定义为训练步骤或epoch的数量，而不是样本数量，如`scikit-learn`的实现。我们将在下一节中学习如何利用这些基于MFO的剪枝器。

要使用`Optuna`执行超参数调整，我们可以简单地执行以下简单步骤（更详细的步骤，包括代码实现，将在下一节中的各种示例中给出）：

1.  定义目标函数以及超参数空间。

1.  通过`create_study()`函数初始化`study`对象。

1.  通过在`study`对象上调用`optimize()`方法执行超参数调整。

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

1.  在测试数据上测试最终训练好的模型。

在`Optuna`中，我们可以在`objective`函数本身内直接定义超参数空间。无需定义另一个专门的独立对象来存储超参数空间。这意味着在`Optuna`中实现条件超参数变得非常容易，因为我们只需将它们放在`objective`函数中的相应`if-else`块内。`Optuna`还提供了非常实用的超参数采样分布方法，包括`suggest_categorical`、`suggest_discrete_uniform`、`suggest_int`和`suggest_float`。

`suggest_categorical`方法将建议从分类类型的超参数中获取值，这与`random.choice()`方法的工作方式类似。`suggest_discrete_uniform`可用于离散类型的超参数，其工作方式与Hyperopt中的`hp.quniform`非常相似（参见[*第8章*](B18753_08_ePub.xhtml#_idTextAnchor074)中通过Hyperopt进行超参数调整），通过从`[low, high]`范围内以`q`步长进行离散化均匀采样。`suggest_int`方法与`random.randint()`方法类似。最后是`suggest_float`方法。此方法适用于浮点类型的超参数，实际上是两个其他采样分布方法的包装，即`suggest_uniform`和`suggest_loguniform`。要使用`suggest_loguniform`，只需将`suggest_float`中的`log`参数设置为`True`。

为了更好地理解我们如何在`objective`函数内定义超参数空间，以下代码展示了如何使用`objective`函数定义一个`objective`函数的示例，以确保可读性并使我们能够以模块化方式编写代码。然而，您也可以直接将所有代码放在一个单独的`objective`函数中。本例中使用的数据和预处理步骤与[*第7章*](B18753_07_ePub.xhtml#_idTextAnchor062)中相同，即通过Scikit进行超参数调整。然而，在本例中，我们使用的是**神经网络**模型而不是随机森林，如下所示：

1.  创建一个函数来定义模型架构。在这里，我们创建了一个二元分类器模型，其中隐藏层的数量、单元数量、dropout率和每层的`activation`函数都是超参数空间的一部分，如下所示：

    [PRE0]

1.  创建一个函数来定义模型的优化器。请注意，我们在该函数中定义了条件超参数，其中针对不同选择的优化器有不同的超参数集，如下所示：

    [PRE1]

1.  创建`train`和`validation`函数。请注意，预处理代码在此处未显示，但您可以在*技术要求*部分提到的GitHub仓库中看到完整的代码。与[*第7章*](B18753_07_ePub.xhtml#_idTextAnchor062)中的示例一样，我们也将F1分数作为模型的评估指标，如下所示：

    [PRE2]

1.  创建`objective`函数。在这里，我们将原始训练数据分为用于超参数调整的训练数据`df_train_hp`和验证数据`df_val`。我们不会遵循k折交叉验证评估方法，因为这会在每个调整试验中花费太多时间让神经网络模型通过几个评估折（参见[*第1章*](B18753_01_ePub.xhtml#_idTextAnchor014)*，评估机器学习模型*）。

    [PRE3]

要在`Optuna`中执行超参数调整，我们需要通过`create_study()`函数初始化一个`study`对象。`study`对象提供了运行新的`Trial`对象和访问试验历史的接口。`Trial`对象简单地说是一个涉及评估`objective`函数过程的对象。此对象将被传递给`objective`函数，并负责管理试验的状态，在接收到参数建议时提供接口，就像我们在`objective`函数中之前看到的那样。以下代码展示了如何利用`create_study()`函数来初始化一个`study`对象：

[PRE4]

在`create_study()`函数中，有几个重要的参数。`direction`参数允许我们告诉`Optuna`我们正在处理哪种优化问题。此参数有两个有效值，即*‘maximize’*和*‘minimize’*。通过将`direction`参数设置为*‘maximize’*，这意味着我们告诉`Optuna`我们目前正在处理一个最大化问题。`Optuna`默认将此参数设置为*‘minimize’*。`sampler`参数指的是我们想要使用的超参数调整算法。默认情况下，`Optuna`将使用TPE作为采样器。`pruner`参数指的是我们想要使用的修剪算法，其中默认使用`MedianPruner()`。

Optuna中的修剪

虽然`MedianPruner()`默认被选中，但除非我们明确在`objective`函数中告诉`Optuna`这样做，否则修剪过程将不会执行。以下链接展示了如何使用`Optuna`的默认修剪器执行简单的修剪过程：[https://github.com/optuna/optuna-examples/blob/main/simple_pruning.py](https://github.com/optuna/optuna-examples/blob/main/simple_pruning.py)。

除了前面提到的三个参数之外，`create_study()`函数中还有其他参数，即`storage`、`study_name`和`load_if_exists`。`storage`参数期望一个数据库URL输入，它将由`Optuna`处理。如果我们没有传递数据库URL，`Optuna`将使用内存存储。`study_name`参数是我们想要赋予当前`study`对象的名称。如果我们没有传递名称，`Optuna`将自动为我们生成一个随机名称。最后但同样重要的是，`load_if_exists`参数是一个布尔参数，用于处理可能存在冲突的实验名称的情况。如果存储中已经生成了实验名称，并且我们将`load_if_exists`设置为`False`，那么`Optuna`将引发错误。另一方面，如果存储中已经生成了实验名称，但我们设置了`load_if_exists=True`，`Optuna`将只加载现有的`study`对象而不是创建一个新的对象。

一旦初始化了`study`对象并设置了适当的参数，我们就可以通过调用`optimize()`方法开始执行超参数调优。以下代码展示了如何进行操作：

[PRE5]

[PRE6]

在`optimize()`方法中存在几个重要的参数。第一个也是最重要的参数是`func`参数。这个参数期望一个可调用的对象，该对象实现了`objective`函数。在这里，我们并没有直接将`objective`函数传递给`func`参数，因为我们的`objective`函数需要两个输入，而默认情况下，`Optuna`只能处理一个输入的`objective`函数，即`Trial`对象本身。这就是为什么我们需要Python内置的`lambda`函数来将第二个输入传递给我们的`objective`函数。如果你的`objective`函数有超过两个输入，你也可以使用相同的`lambda`函数。

第二个最重要的参数是`n_trials`，它指的是超参数调优过程中的试验次数或迭代次数。另一个可以作为停止标准的实现参数是`timeout`参数。这个参数期望以秒为单位的停止标准。默认情况下，`Optuna`将`n_trials`和`timeout`参数设置为`None`。如果我们让它保持原样，那么`Optuna`将运行超参数调优过程，直到接收到终止信号，例如`Ctrl+C`或`SIGTERM`。

最后但同样重要的是，`Optuna`还允许我们通过一个名为`n_jobs`的参数来利用并行资源。默认情况下，`Optuna`将`n_jobs`设置为`1`，这意味着它将只利用一个工作。在这里，我们将`n_jobs`设置为`-1`，这意味着我们将使用计算机上的所有CPU核心来执行并行计算。

Optuna中超参数的重要性

`Optuna`提供了一个非常棒的模块来衡量搜索空间中每个超参数的重要性。根据2.10.0版本，实现了两种方法，即**fANOVA**和**Mean Decrease Impurity**方法。请参阅官方文档了解如何利用此模块以及实现方法的背后理论，文档链接如下：[https://optuna.readthedocs.io/en/v2.10.0/reference/importance.html](https://optuna.readthedocs.io/en/v2.10.0/reference/importance.html)。

在本节中，我们了解了`Optuna`的一般概念，我们可以利用的功能，以及如何使用此包进行超参数调整的一般步骤。`Optuna`还提供了各种可视化模块，可以帮助我们跟踪我们的超参数调整实验，这将在[*第13章*](B18753_13_ePub.xhtml#_idTextAnchor125)中讨论，*跟踪超参数调整实验*。在接下来的章节中，我们将通过示例学习如何使用`Optuna`执行各种超参数调整方法。

# 实现TPE

TPE是贝叶斯优化超参数调整组（见[*第4章*](B18753_04_ePub.xhtml#_idTextAnchor036)）的一种变体，是`Optuna`中的默认采样器。要在`Optuna`中使用TPE进行超参数调整，我们只需将`optuna.samplers.TPESampler()`类传递给`create_study()`函数的采样器参数。以下示例展示了如何在`Optuna`中实现TPE。我们将使用[*第7章*](B18753_07_ePub.xhtml#_idTextAnchor062)中示例中的相同数据，并按照前节中介绍的步骤进行如下操作：

1.  定义`objective`函数以及超参数空间。在这里，我们将使用与*介绍Optuna*部分中定义的相同的函数。请记住，我们在`objective`函数中使用的是训练-验证分割，而不是k折交叉验证方法。

1.  通过`create_study()`函数初始化`study`对象，如下所示：

    [PRE7]

1.  通过在`study`对象上调用`optimize()`方法来执行超参数调整，如下所示：

    [PRE8]

根据前面的代码，我们在验证数据上得到了大约`0.563`的F1分数。我们还得到了一个包含最佳超参数集的字典，如下所示：

[PRE9]

1.  使用找到的最佳超参数集在全部训练数据上训练模型。在这里，我们定义了一个名为`train_and_evaluate_final()`的另一个函数，其目的是基于前一步找到的最佳超参数集在全部训练数据上训练模型，并在测试数据上对其进行评估。您可以在*技术要求*部分提到的GitHub仓库中看到实现的函数。定义函数如下：

    [PRE10]

1.  在测试数据上测试最终训练好的模型。根据前一步的结果，当使用最佳超参数集在测试集上测试我们最终训练的神经网络模型时，F1 分数大约为 `0.604`。

`TPESampler`类有几个重要的参数。首先，是`gamma`参数，它指的是TPE中用于区分好样本和坏样本的阈值（参见[*第 4 章*](B18753_04_ePub.xhtml#_idTextAnchor036)）。`n_startup_trials`参数负责控制在进行TPE算法之前，将有多少次试验使用随机搜索。`n_ei_candidates`参数负责控制用于计算`预期改进获取函数`的候选样本数量。最后但同样重要的是，`seed`参数，它控制实验的随机种子。`TPESampler`类还有许多其他参数，请参阅以下链接的原版文档获取更多信息：[https://optuna.readthedocs.io/en/v2.10.0/reference/generated/optuna.samplers.TPESampler.html](https://optuna.readthedocs.io/en/v2.10.0/reference/generated/optuna.samplers.TPESampler.html)。

在本节中，我们学习了如何在`Optuna`中使用与[*第 7 章*](B18753_07_ePub.xhtml#_idTextAnchor062)示例中相同的数据执行超参数调优。如[*第 4 章*](B18753_04_ePub.xhtml#_idTextAnchor036)中所述，探索贝叶斯优化，`Optuna`也实现了多变量TPE，能够捕捉超参数之间的相互依赖关系。要启用多变量TPE，我们只需将`optuna.samplers.TPESampler()`中的`multivariate`参数设置为`True`。在下一节中，我们将学习如何使用`Optuna`进行随机搜索。

# 实现随机搜索

在`Optuna`中实现随机搜索与实现TPE（Tree-based Parzen Estimator）在`Optuna`中非常相似。我们只需遵循前一个章节的类似步骤，并在*步骤 2*中更改`optimize()`方法中的`sampler`参数。以下代码展示了如何进行操作：

[PRE11]

[PRE12]

使用完全相同的数据、预处理步骤、超参数空间和`objective`函数，我们在验证数据中评估的F1分数大约为 `0.548`。我们还得到了一个包含最佳超参数集的字典，如下所示：

[PRE13]

使用最佳超参数集在完整数据上训练模型后，我们在测试数据上训练的最终神经网络模型测试时，F1分数大约为`0.596`。请注意，尽管我们之前定义了许多超参数（参见前一小节中的`objective`函数），但在这里，我们并没有在结果中得到所有这些超参数。这是因为大多数超参数都是条件超参数。例如，由于为`*’num_layers’*`超参数选择的值是零，因此将不存在`*’n_units_layer_{layer_i}’*`、`*’dropout_rate_layer_{layer_i}’*`或`*‘actv_func _layer_{layer_i}’*`，因为这些超参数只有在`*’num_layers’*超参数大于零时才会存在。

在本节中，我们看到了如何使用`Optuna`的随机搜索方法进行超参数调整。在下一节中，我们将学习如何使用`Optuna`包实现网格搜索。

# 实现网格搜索

在`Optuna`中实现网格搜索与实现TPE和随机搜索略有不同。在这里，我们还需要定义搜索空间对象并将其传递给`optuna.samplers.GridSampler()`。搜索空间对象只是一个Python字典数据结构，其键是超参数的名称，而字典的值是对应超参数的可能值。如果搜索空间中的所有组合都已评估，即使传递给`optimize()`方法的`n_trials`数量尚未达到，`GridSampler`也会停止超参数调整过程。此外，无论我们传递给采样分布方法（如`suggest_categorical`、`suggest_discrete_uniform`、`suggest_int`和`suggest_float`）的范围如何，`GridSampler`都只会获取搜索空间中声明的值。

以下代码展示了如何在`Optuna`中执行网格搜索。在`Optuna`中实现网格搜索的总体步骤与*实现树结构帕累托估计器*一节中所述的步骤相似。唯一的区别是我们必须定义搜索空间对象，并在*步骤2*中的`optimize()`方法中将`sampler`参数更改为`optuna.samplers.GridSampler()`，如下所示：

[PRE14]

[PRE15]

[PRE16]

[PRE17]

[PRE18]

[PRE19]

[PRE20]

[PRE21]

[PRE22]

[PRE23]

[PRE24]

[PRE25]

根据前面的代码，我们在验证数据上评估的F1分数大约为`0.574`。我们还得到了一个包含最佳超参数集的字典，如下所示：

[PRE26]

使用最佳超参数集在完整数据上训练模型后，我们在测试数据上训练的最终神经网络模型测试时，F1分数大约为`0.610`。

值得注意的是，`GridSampler`将依赖于搜索空间来执行超参数采样。例如，在搜索空间中，我们只定义了`num_layers`的有效值为`[0,1]`。因此，尽管在`objective`函数中我们设置了`trial.suggest_int(`num_layers`,low=0,high=3)`（参见*介绍Optuna*部分），但在调整过程中只会测试`0`和`1`。记住，在`Optuna`中，我们可以通过`n_trials`或`timeout`参数指定停止标准。如果我们指定了这些标准之一，`GridSampler`将不会测试搜索空间中的所有可能组合；一旦满足停止标准，调整过程将停止。在这个例子中，我们设置了`n_trials=50`，就像前一个示例部分中那样。

在本节中，我们学习了如何使用`Optuna`的网格搜索方法进行超参数调整。在下一节中，我们将学习如何使用`Optuna`包实现模拟退火（SA）。

# 实现模拟退火

SA不是`Optuna`内置的超参数调整方法的一部分。然而，正如本章第一部分所述，我们可以在`Optuna`中定义自己的自定义采样器。在创建自定义采样器时，我们需要创建一个继承自`BaseSampler`类的类。在我们自定义类中需要定义的最重要方法是`sample_relative()`方法。此方法负责根据我们选择的超参数调整算法从搜索空间中采样相应的超参数。

完整的自定义`SimulatedAnnealingSampler()`类，包括几何退火调度计划（参见[*第5章*](B18753_05_ePub.xhtml#_idTextAnchor047)），已在*技术要求*部分中提到的GitHub仓库中定义，并可以查看。以下代码仅展示了类中`sample_relative()`方法的实现：

[PRE27]

[PRE28]

[PRE29]

[PRE30]

[PRE31]

[PRE32]

[PRE33]

[PRE34]

[PRE35]

[PRE36]

[PRE37]

[PRE38]

[PRE39]

[PRE40]

以下代码展示了如何在`Optuna`中使用SA进行超参数调整。在`Optuna`中实现SA的整体过程与*实现树结构帕累托估计器*部分中所述的过程类似。唯一的区别是我们必须在*步骤2*的`optimize()`方法中将`sampler`参数更改为`SimulatedAnnealingSampler()`，如下所示：

[PRE41]

[PRE42]

[PRE43]

使用完全相同的数据、预处理步骤、超参数空间和`objective`函数，我们在验证数据中得到的F1分数大约为`0.556`。我们还得到了一个包含最佳超参数集的字典，如下所示：

[PRE44]

在使用最佳超参数集在全部数据上训练模型后，当我们测试在测试数据上训练的最终神经网络模型时，F1分数大约为`0.559`。

在本节中，我们学习了如何使用`Optuna`的SA算法进行超参数调整。在下一节中，我们将学习如何在`Optuna`中利用逐次减半作为剪枝方法。

# 实现Successive Halving

`Optuna`意味着它负责在似乎没有继续进行过程的好处时停止超参数调整迭代。由于它被实现为剪枝器，`Optuna`中SH（Successive Halving）的资源定义（见[*第6章*](B18753_06_ePub.xhtml#_idTextAnchor054)）指的是模型的训练步数或epoch数，而不是样本数，正如`scikit-learn`实现中那样。

我们可以利用SH（Successive Halving）作为剪枝器，同时使用我们使用的任何采样器。本例展示了如何使用随机搜索算法作为采样器，SH作为剪枝器来执行超参数调整。整体流程与*实现TPE*部分中所述的流程类似。由于我们使用SH作为剪枝器，我们必须编辑我们的`objective`函数，以便在优化过程中使用剪枝器。在本例中，我们可以使用`Optuna`提供的`TFKeras`的回调集成，通过`optuna.integration.TFKerasPruningCallback`。我们只需在`train`函数中拟合模型时将此类传递给`callbacks`参数，如下面的代码所示：

[PRE45]

[PRE46]

[PRE47]

[PRE48]

[PRE49]

[PRE50]

[PRE51]

[PRE52]

[PRE53]

一旦我们告诉`Optuna`使用剪枝器，我们还需要在*实现树结构Parzen估计器*部分的*步骤2*中将`optimize()`方法中的`pruner`参数设置为`optuna.pruners.SuccessiveHalvingPruner()`，如下所示：

[PRE54]

[PRE55]

[PRE56]

[PRE57]

在这个例子中，我们也增加了试验次数从`50`到`100`，因为大多数试验无论如何都会被剪枝，如下所示：

[PRE58]

[PRE59]

[PRE60]

使用完全相同的数据、预处理步骤和超参数空间，我们在验证数据中得到的F1分数大约是`0.582`。在`100`次试验中，有`87`次试验被SH剪枝，这意味着只有`13`次试验完成。我们还得到了一个包含最佳超参数集的字典，如下所示：

[PRE61]

在使用最佳超参数集在全部数据上训练模型之后，我们在测试数据上训练的最终神经网络模型的F1分数大约是`0.597`。

值得注意的是，`SuccessiveHalvingPruner`有几个参数我们可以根据我们的需求进行自定义。`reduction_factor`参数指的是SH（Successive Halving）的乘数因子（见[*第6章*](B18753_06_ePub.xhtml#_idTextAnchor054)）。`min_resource`参数指的是第一次试验中要使用的最小资源数量。默认情况下，此参数设置为`‘auto’`，其中使用启发式算法根据第一次试验完成所需的步数来计算最合适的值。换句话说，`Optuna`只有在执行了`min_resource`训练步数或epoch数之后才能开始调整过程。

`Optuna`还提供了`min_early_stopping_rate`参数，其意义与我们定义在[*第6章*](B18753_06_ePub.xhtml#_idTextAnchor054)中的完全相同。最后但同样重要的是，`bootstrap_count`参数。此参数不是原始SH算法的一部分。此参数的目的是控制实际SH迭代开始之前需要完成的试验的最小数量。

你可能会想知道，关于控制最大资源和SH中候选人数的参数是什么？在这里，在`Optuna`中，最大资源的定义将根据定义的`objective`函数中的总训练步骤或epoch数自动推导。至于控制候选人数的参数，`Optuna`将此责任委托给`study.optimize()`方法中的`n_trials`参数。

在本节中，我们学习了如何在参数调整过程中利用SH作为剪枝器。在下一节中，我们将学习如何利用SH的扩展算法Hyperband作为`Optuna`中的剪枝方法。

# 实现Hyperband

实现`Optuna`与实现Successive Halving作为剪枝器非常相似。唯一的区别是我们必须在上一节中的*步骤2*中将`optimize()`方法中的`pruner`参数设置为`optuna.pruners.HyperbandPruner()`。以下代码展示了如何使用随机搜索算法作为采样器，HB作为剪枝器进行超参数调整：

[PRE62]

[PRE63]

[PRE64]

[PRE65]

`HyperbandPruner`的所有参数都与`SuccessiveHalvingPruner`相同，除了这里没有`min_early_stopping_rate`参数，而有一个`max_resource`参数。`min_early_stopping_rate`参数被移除，因为它根据每个括号的ID自动设置。`max_resource`参数负责设置分配给试验的最大资源。默认情况下，此参数设置为`‘auto’`，这意味着其值将设置为第一个完成的试验中的最大步长。

使用完全相同的数据、预处理步骤和超参数空间，我们在验证数据中得到的F1分数大约是`0.580`。在进行的`100`次试验中，有`79`次试验被SH剪枝，这意味着只有`21`次试验完成。我们还得到了一个包含最佳超参数集的字典，如下所示：

[PRE66]

在使用最佳超参数集在全部数据上训练模型后，当我们测试在测试数据上训练的最终神经网络模型时，F1分数大约是`0.609`。

在本节中，我们学习了如何在`Optuna`的参数调整过程中利用HB作为剪枝器。

# 摘要

在本章中，我们学习了`Optuna`包的所有重要方面。我们还学会了如何利用这个包实现各种超参数调优方法，并且理解了每个类的重要参数以及它们与我们之前章节中学到的理论之间的关系。从现在开始，你应该能够利用我们在上一章中讨论的包来实现你选择的超参数调优方法，并最终提升你的机器学习模型的性能。掌握了第3章至第6章的知识，你还将能够调试代码，如果出现错误或意外结果，你还将能够制定自己的实验配置以匹配你的特定问题。

在下一章中，我们将学习DEAP和Microsoft NNI包以及如何利用它们来执行各种超参数调优方法。下一章的目标与本章类似，即能够利用包进行超参数调优，并理解实现类中的每个参数。
