# 第八章：*第八章*：通过 Hyperopt 进行超参数调整

**Hyperopt** 是一个 Python 优化包，提供了多种超参数调整方法的实现，包括 **随机搜索**、**模拟退火**（**SA**）、**树结构帕累托估计器**（**TPE**）和 **自适应 TPE**（**ATPE**）。它还支持各种类型的超参数，以及不同类型的采样分布。

在本章中，我们将介绍 `Hyperopt` 包，从其功能和限制开始，学习如何利用它进行超参数调整，以及你需要了解的关于 `Hyperopt` 的所有其他重要事项。我们将学习如何利用 `Hyperopt` 的默认配置进行超参数调整，并讨论可用的配置及其用法。此外，我们还将讨论超参数调整方法的实现与我们在前几章中学到的理论之间的关系，因为实现中可能有一些细微的差异或调整。

到本章结束时，你将能够了解关于 `Hyperopt` 的所有重要事项，并能够实现该包中提供的各种超参数调整方法。你还将能够理解它们类的重要参数以及它们与我们之前章节中学到的理论之间的关系。最后，凭借前几章的知识，你将能够理解如果出现错误或意外结果时会发生什么，以及如何设置方法配置以匹配你的特定问题。

本章将涵盖以下主题：

+   介绍 Hyperopt

+   实现随机搜索

+   实现树结构帕累托估计器

+   实现自适应树结构帕累托估计器

+   实现模拟退火

# 技术要求

在本章中，我们将学习如何使用 Hyperopt 实现各种超参数调整方法。为了确保你能重现本章中的代码示例，你需要以下条件：

+   Python 3（版本 3.7 或更高版本）

+   `pandas` 包（版本 1.3.4 或更高版本）

+   `NumPy` 包（版本 1.21.2 或更高版本）

+   `Matplotlib` 包（版本 3.5.0 或更高版本）

+   `scikit-learn` 包（版本 1.0.1 或更高版本）

+   `Hyperopt` 包（版本 0.2.7 或更高版本）

+   `LightGBM` 包（版本 3.3.2 或更高版本）

本章的所有代码示例都可以在 GitHub 上找到，链接为 [`github.com/PacktPublishing/Hyperparameter-Tuning-with-Python`](https://github.com/PacktPublishing/Hyperparameter-Tuning-with-Python)。

# 介绍 Hyperopt

`Hyperopt`包中实现的全部优化方法都假设我们正在处理一个*最小化问题*。如果你的目标函数被分类为最大化问题，例如，当你使用准确率作为目标函数得分时，你必须*对你的目标函数添加一个负号*。

利用`Hyperopt`包进行超参数调整非常简单。以下步骤展示了如何执行`Hyperopt`包中提供的任何超参数调整方法。更详细的步骤，包括代码实现，将在接下来的章节中通过各种示例给出：

1.  定义要最小化的目标函数。

1.  定义超参数空间。

1.  (*可选*) 初始化`Trials()`对象并将其传递给`fmin()`函数。

1.  通过调用`fmin()`函数进行超参数调整。

1.  使用从`fmin()`函数输出中找到的最佳超参数集在全部训练数据上训练模型。

1.  在测试数据上测试最终训练好的模型。

目标函数的最简单情况是我们只返回目标函数得分的浮点类型。然而，我们也可以将其他附加信息添加到目标函数的输出中，例如评估时间或我们想要用于进一步分析的任何其他统计数据。当我们向目标函数得分的输出添加附加信息时，`Hyperopt`期望目标函数的输出形式为 Python 字典，该字典至少包含两个强制性的键值对——即`status`和`loss`。前者键存储运行的状态值，而后者键存储我们想要最小化的目标函数。

Hyperopt 中最简单的超参数空间形式是 Python 字典的形式，其中键指的是超参数的名称，值包含从其中采样的超参数分布。以下示例展示了我们如何在`Hyperopt`中定义一个非常简单的超参数空间：

```py
import numpy as np
```

```py
from hyperopt import hp
```

```py
hyperparameter_space = {
```

```py
“criterion”: hp.choice(“criterion”, [“gini”, “entropy”]),
```

```py
“n_estimators”: 5 + hp.randint(“n_estimators”, 195),
```

```py
“min_samples_split” : hp.loguniform(“min_samples_split”, np.log(0.0001), np.log(0.5))
```

```py
}
```

如您所见，`hyperparameter_space`字典的值是伴随空间中每个超参数的分布。`Hyperopt`提供了许多采样分布，我们可以利用，例如`hp.choice`、`hp.randint`、`hp.uniform`、`hp.loguniform`、`hp.normal`和`hp.lognormal`。`hp.choice`分布将随机从几个给定选项中选择一个。`hp.randint`分布将在`[0, high)`范围内随机选择一个整数，其中`high`是我们输入的值。在先前的示例中，我们传递了`195`作为`high`值并添加了`5`的值。这意味着`Hyperopt`将在`[5,200)`范围内随机选择一个整数。

剩余的分布都是针对实数/浮点超参数值的。请注意，Hyperopt 还提供了针对整数超参数值的分布，这些分布模仿了上述四个分布的分布情况——即`hp.quniform`、`hp.qloguniform`、`hp.qnormal`和`hp.qlognormal`。有关 Hyperopt 提供的采样分布的更多信息，请参阅其官方维基页面([`github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions`](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions))。

值得注意的是，Hyperopt 使我们能够定义一个**条件超参数空间**（参见*第四章**，贝叶斯优化*），以满足我们的需求。以下代码示例展示了我们如何定义这样的搜索空间：

```py
hyperparameter_space =
```

```py
hp.choice(“class_weight_type”, [
```

```py
{“class_weight”: None,
```

```py
“n_estimators”: 5 + hp.randint(“none_n_estimators”, 45),
```

```py
},
```

```py
{“class_weight”: “balanced”,
```

```py
“n_estimators”: 5 + hp.randint(“balanced_n_estimators”, 195),
```

```py
}
```

```py
])
```

如你所见，条件超参数空间和非条件超参数空间之间的唯一区别是在定义每个条件的超参数之前添加了`hp.choice`。在这个例子中，当`class_weight`为`None`时，我们只会在范围`[5,50)`内搜索最佳的`n_estimators`超参数。另一方面，当`class_weight`为`“balanced”`时，范围变为`[5,200)`。

一旦定义了超参数空间，我们就可以通过`fmin()`函数开始超参数调整过程。该函数的输出是从调整过程中找到的最佳超参数集。此函数中提供了几个重要的参数，你需要了解它们。`fn`参数指的是我们试图最小化的目标函数，`space`参数指的是我们实验中将要使用的超参数空间，`algo`参数指的是我们想要利用的超参数调整算法，`rstate`参数指的是调整过程的随机种子，`max_evals`参数指的是基于试验次数的调整过程停止标准，而`timeout`参数指的是基于秒数时间限制的停止标准。另一个重要的参数是`trials`参数，它期望接收`Hyperopt`的`Trials()`对象。

`Hyperopt`中的`Trials()`对象在调整过程中记录所有相关信息。此对象还负责存储我们放入目标函数字典输出中的所有附加信息。我们可以利用此对象进行调试或直接将其传递给`Hyperopt`内置的绘图模块。

`Hyperopt`包中实现了几个内置的绘图模块，例如`main_plot_history`、`main_plot_histogram`和`main_plot_vars`模块。第一个绘图模块可以帮助我们理解损失值与执行时间之间的关系。第二个绘图模块显示了所有试验中所有损失的直方图。第三个绘图模块对于理解每个超参数相对于损失值的热图非常有用。

最后但同样重要的是，值得注意的是，Hyperopt 还通过利用`Trials()`到`MongoTrials()`支持并行搜索过程。如果我们想使用 Spark 而不是 MongoDB，我们可以从`Trials()`切换到`SparkTrials()`。请参阅 Hyperopt 的官方文档以获取有关并行计算更多信息（[`github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB`](https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB) 和 [`hyperopt.github.io/hyperopt/scaleout/spark/`](http://hyperopt.github.io/hyperopt/scaleout/spark/)）。

在本节中，你已了解了`Hyperopt`包的整体功能，以及使用此包进行超参数调优的一般步骤。在接下来的几节中，我们将通过示例学习如何实现`Hyperopt`中可用的每种超参数调优方法。

# 实现随机搜索

要在 Hyperopt 中实现随机搜索（见*第三章*），我们可以简单地遵循上一节中解释的步骤，并将`rand.suggest`对象传递给`fmin()`函数中的`algo`参数。让我们学习如何利用`Hyperopt`包来执行随机搜索。我们将使用与*第七章**，通过 Scikit 进行超参数调优*相同的相同数据和`sklearn`管道定义，但使用稍有不同的超参数空间定义。让我们遵循上一节中介绍的步骤：

1.  定义要最小化的目标函数。在这里，我们利用定义的管道`pipe`，通过使用`sklearn`中的`cross_val_score`函数来计算*5 折交叉验证*分数。我们将使用*F1 分数*作为评估指标：

    ```py
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from hyperopt import STATUS_OK
    def objective(space):
        estimator_clone = clone(pipe).set_params(**space)
        return {‘loss’: -1 * np.mean(cross_val_score(estimator_clone, X_train_full, y_train, cv=5, scoring=’f1’, n_jobs=-1)), 
                ‘status’: STATUS_OK}
    ```

注意，定义的`objective`函数只接收一个输入，即预定义的超参数空间`space`，并输出一个包含两个强制性的键值对——即`status`和`loss`。还值得注意的是，我们之所以将平均交叉验证分数输出乘以`-1`，是因为`Hyperopt`始终假设我们正在处理一个最小化问题，而在这个例子中我们并非如此。

1.  定义超参数空间。由于我们使用`sklearn`管道作为我们的估计器，我们仍然需要遵循定义空间内超参数的命名约定（参见*第七章*）。请注意，命名约定只需应用于搜索空间字典键中的超参数名称，而不是采样分布对象内的名称：

    ```py
    from hyperopt import hp
    hyperparameter_space = { 
    “model__n_estimators”: 5 + hp.randint(“n_estimators”, 195), 
    “model__criterion”: hp.choice(“criterion”, [“gini”, “entropy”]),
    “model__class_weight”: hp.choice(“class_weight”, [“balanced”,”balanced_subsample”]),
    “model__min_samples_split”: hp.loguniform(“min_samples_split”, np.log(0.0001), np.log(0.5))
    }
    ```

1.  初始化`Trials()`对象。在这个例子中，我们将在调整过程完成后利用此对象进行绘图：

    ```py
    from hyperopt import Trials
    trials = Trials()
    ```

1.  通过调用`fmin()`函数进行超参数调整。在这里，我们通过传递定义的目标函数和超参数空间进行随机搜索。我们将`algo`参数设置为`rand.suggest`对象，并将试验次数设置为`100`作为停止标准。我们还设置了随机状态以确保可重复性。最后但同样重要的是，我们将定义的`Trials()`对象传递给`trials`参数：

    ```py
    from hyperopt import fmin, rand
    best = fmin(objective,
                space=hyperparameter_space,
                algo=rand.suggest,
                max_evals=100,
                rstate=np.random.default_rng(0),
                trials=trials
               )
    print(best)
    ```

根据前面的代码，我们得到目标函数分数大约为`-0.621`，这指的是平均 5 折交叉验证 F--分数的`0.621`。我们还得到一个包含最佳超参数集的字典，如下所示：

```py
{‘class_weight’: 0, ‘criterion’: 1, ‘min_samples_split’: 0.00047017001935242104, ‘n_estimators’: 186}
```

如所示，当我们使用`hp.choice`作为采样分布时，`Hyperopt`将仅返回超参数值的索引。在这里，通过参考预定义的超参数空间，`class_weight`的`0`表示*平衡*，而`criterion`的`1`表示*熵*。因此，最佳超参数集是`{‘model__class_weight’: ‘balanced’, ‘model__criterion’: ‘entropy’, ‘model__min_samples_split’: 0.0004701700193524210, ‘model__n_estimators’: 186}`。

1.  使用`fmin()`函数输出中找到的最佳超参数集在全部训练数据上训练模型：

    ```py
    pipe = pipe.set_params(**{‘model__class_weight’: “balanced”,
    ‘model__criterion’: “entropy”,
    ‘model__min_samples_split’: 0.00047017001935242104,
    ‘model__n_estimators’: 186})
    pipe.fit(X_train_full,y_train)
    ```

1.  在测试数据上测试最终训练的模型：

    ```py
    from sklearn.metrics import f1_score
    y_pred = pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

根据前面的代码，当我们在测试集上使用最佳超参数集测试最终训练的随机森林模型时，我们得到大约`0.624`的 F1 分数。

1.  最后但同样重要的是，我们还可以利用`Hyperopt`中实现的内置绘图模块。以下代码展示了如何进行这一操作。请注意，我们需要将调整过程中的`trials`对象传递给绘图模块，因为所有调整过程日志都存储在其中：

    ```py
    from hyperopt import plotting
    ```

现在，我们必须绘制损失值与执行时间的关系：

```py
plotting.main_plot_history(trials)
```

我们将得到以下输出：

![图 8.1 – 损失值与执行时间的关系![图 B18753_08_001.jpg](img/B18753_08_001.jpg)

图 8.1 – 损失值与执行时间的关系

现在，我们必须绘制所有试验的目标函数分数的直方图：

```py
plotting.main_plot_histogram(trials)
```

我们将得到以下输出。

![图 8.2 – 所有试验的目标函数分数直方图![图 B18753_08_002.jpg](img/B18753_08_002.jpg)

图 8.2 – 所有试验中目标函数分数的直方图

现在，我们必须绘制空间中每个超参数相对于损失值的热图：

```py
Plotting.main_plot_vars(trials)
```

我们将得到以下输出。

![图 8.3 – 空间中每个超参数相对于损失值的热图（越暗，越好）

![img/B18753_08_003.jpg]

图 8.3 – 空间中每个超参数相对于损失值的热图（越暗，越好）

在本节中，我们通过查看与*第七章*中展示的类似示例相同的示例，学习了如何在`Hyperopt`中执行随机搜索。我们还看到了通过利用 Hyperopt 内置的绘图模块，我们可以得到什么样的图形。

值得注意的是，我们不仅限于使用`sklearn`模型的实现来使用`Hyperopt`进行超参数调整。我们还可以使用来自其他包的实现，例如`PyTorch`、`Tensorflow`等。需要记住的一点是在进行交叉验证时要注意*数据泄露问题*（参见*第一章*，“评估机器学习模型”）。我们必须将所有数据预处理方法拟合到训练数据上，并将拟合的预处理程序应用于验证数据。

在下一节中，我们将学习如何利用`Hyperopt`通过可用的贝叶斯优化方法之一进行超参数调整。

# 实现树结构帕累托估计器

`Hyperopt`包。要使用此方法进行超参数调整，我们可以遵循与上一节类似的程序，只需将*步骤 4*中的`algo`参数更改为`tpe.suggest`。以下代码显示了如何在`Hyperopt`中使用 TPE 进行超参数调整：

```py
from hyperopt import fmin, tpe
```

```py
best = fmin(objective, 
```

```py
            space=hyperparameter_space, 
```

```py
            algo=tpe.suggest, 
```

```py
            max_evals=100, 
```

```py
            rstate=np.random.default_rng(0), 
```

```py
            trials=trials 
```

```py
           )
```

```py
print(best)
```

使用相同的数据、超参数空间和`fmin()`函数的参数，我们得到了大约`-0.620`的目标函数分数，这相当于平均 5 折交叉验证 F1 分数的`0.620`。我们还得到了一个包含最佳超参数集的字典，如下所示：

```py
{‘class_weight’: 1, ‘criterion’: 1, ‘min_samples_split’: 0.0005245304932726025, ‘n_estimators’: 138}
```

一旦使用最佳超参数集在全部数据上训练了模型，我们在测试数据上测试训练好的最终随机森林模型时，F1 分数大约为`0.621`。

在本节中，我们学习了如何使用`Hyperopt`中的 TPE 方法进行超参数调整。在下一节中，我们将学习如何使用`Hyperopt`包实现 TPE 的一个变体，称为自适应 TPE。

# 实现自适应 TPE

**自适应 TPE**（**ATPE**）是 TPE 超参数调优方法的变体，它基于与 TPE 相比的几个改进而开发，例如根据我们拥有的数据自动调整 TPE 方法的几个超参数。有关此方法的更多信息，请参阅原始白皮书。这些可以在作者的 GitHub 仓库中找到（[`github.com/electricbrainio/hypermax`](https://github.com/electricbrainio/hypermax)）。

虽然您可以直接使用 ATPE 的原始 GitHub 仓库来实验这种方法，但`Hyperopt`也已将其作为包的一部分包含在内。您只需遵循*实现随机搜索*部分中的类似程序，只需在*步骤 4*中将`algo`参数更改为`atpe.suggest`即可。以下代码展示了如何在`Hyperopt`中使用 ATPE 进行超参数调优。请注意，在`Hyperopt`中使用 ATPE 进行超参数调优之前，我们需要安装`LightGBM`包：

```py
from hyperopt import fmin, atpe
```

```py
best = fmin(objective, 
```

```py
            space=hyperparameter_space, 
```

```py
            algo=atpe.suggest, 
```

```py
            max_evals=100, 
```

```py
            rstate=np.random.default_rng(0), 
```

```py
            trials=trials 
```

```py
           )
```

```py
print(best)
```

使用相同的`fmin()`函数数据、超参数空间和参数，我们得到目标函数得分为约`-0.621`，这相当于平均 5 折交叉验证 F1 分数的`0.621`。我们还得到一个包含最佳超参数集的字典，如下所示：

```py
{‘class_weight’: 1, ‘criterion’: 1, ‘min_samples_split’: 0.0005096354197481012, ‘n_estimators’: 157}
```

一旦使用最佳超参数集在全部数据上训练了模型，我们在测试数据上测试最终训练的随机森林模型时，F1 分数大约为`0.622`。

在本节中，我们学习了如何使用`Hyperopt`中的 ATPE 方法进行超参数调优。在下一节中，我们将学习如何使用`Hyperopt`包实现属于启发式搜索组的超参数调优方法。

# 实现模拟退火

`Hyperopt`包。类似于 TPE 和 ATPE，要使用此方法进行超参数调优，我们只需遵循*实现随机搜索*部分中显示的程序；我们只需要在*步骤 4*中将`algo`参数更改为`anneal.suggest`。以下代码展示了如何在`Hyperopt`中使用 SA 进行超参数调优：

```py
from hyperopt import fmin, anneal
```

```py
best = fmin(objective, 
```

```py
            space=hyperparameter_space, 
```

```py
            algo=anneal.suggest, 
```

```py
            max_evals=100, 
```

```py
            rstate=np.random.default_rng(0), 
```

```py
            trials=trials 
```

```py
           )
```

```py
print(best)
```

使用相同的`fmin()`函数数据、超参数空间和参数，我们得到目标函数得分为约`-0.620`，这相当于平均 5 折交叉验证 F1 分数的`0.620`。我们还得到一个包含最佳超参数集的字典，如下所示：

```py
{‘class_weight’: 1, ‘criterion’: 1, ‘min_samples_split’: 0.00046660708302994583, ‘n_estimators’: 189}
```

一旦使用最佳超参数集在全部数据上训练了模型，我们在测试数据上测试最终训练的随机森林模型时，F1 分数大约为`0.625`。

虽然`Hyperopt`具有内置的绘图模块，但我们也可以通过利用`Trials()`对象来创建自定义的绘图函数。以下代码展示了如何可视化每个超参数在试验次数中的分布：

1.  获取每个试验中每个超参数的值：

    ```py
    plotting_data = np.array([[x[‘result’][‘loss’],
    x[‘misc’][‘vals’][‘class_weight’][0],
    x[‘misc’][‘vals’][‘criterion’][0],
    x[‘misc’][‘vals’][‘min_samples_split’][0],
    x[‘misc’][‘vals’][‘n_estimators’][0],
    ] for x in trials.trials])
    ```

1.  将值转换为 pandas DataFrame：

    ```py
    import pandas as pd
    plotting_data = pd.DataFrame(plotting_data,
    columns=[‘score’, ‘class_weight’, ‘criterion’, ‘min_samples_split’,’n_estimators’])
    ```

1.  绘制每个超参数分布与试验次数之间的关系图：

    ```py
    import matplotlib.pyplot as plt
    plotting_data.plot(subplots=True,figsize=(12, 12))
    plt.xlabel(“Iterations”)
    plt.show()
    ```

基于前面的代码，我们将得到以下输出：

![图 8.4 – 每个超参数分布与试验次数之间的关系![图片](img/B18753_08_004.jpg)

图 8.4 – 每个超参数分布与试验次数之间的关系

在本节中，我们学习了如何通过使用与“实现随机搜索”部分相同的示例来在`Hyperopt`中实现模拟退火（SA）。我们还学习了如何创建一个自定义绘图函数来可视化每个超参数分布与试验次数之间的关系。

# 摘要

在本章中，我们学习了关于`Hyperopt`包的所有重要内容，包括其功能和限制，以及如何利用它来进行超参数调整。我们了解到`Hyperopt`支持各种类型的采样分布方法，但只能与最小化问题一起工作。我们还学习了如何借助这个包实现各种超参数调整方法，这有助于我们理解每个类的重要参数以及它们与我们在前几章中学到的理论之间的关系。此时，你应该能够利用`Hyperopt`来实现你选择的超参数调整方法，并最终提高你的机器学习（ML）模型的性能。凭借从第三章到第六章的知识，你应该能够理解如果出现错误或意外结果时会发生什么，以及如何设置方法配置以匹配你的具体问题。

在下一章中，我们将学习关于`Optuna`包以及如何利用它来执行各种超参数调整方法。下一章的目标与本章节类似——即能够利用该包进行超参数调整，并理解实现类中的每个参数。
