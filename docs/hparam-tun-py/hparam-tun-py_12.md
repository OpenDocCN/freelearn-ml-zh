# 第十章：*第十章*：使用 DEAP 和 Microsoft NNI 进行高级超参数调整

**DEAP**和**Microsoft NNI**是 Python 包，提供了其他包中未实现的多种超参数调整方法，这些包我们在第 7-9 章中讨论过。例如，遗传算法、粒子群优化、Metis、基于群体的训练以及更多。

在本章中，我们将学习如何使用 DEAP 和 Microsoft NNI 包进行超参数调整，从熟悉这些包以及我们需要注意的重要模块和参数开始。我们将学习如何利用 DEAP 和 Microsoft NNI 的默认配置进行超参数调整，并讨论其他可用的配置及其使用方法。此外，我们还将讨论超参数调整方法的实现如何与我们之前章节中学到的理论相关联，因为实现中可能会有一些细微的差异或调整。

在本章结束时，你将能够理解关于 DEAP 和 Microsoft NNI 你需要知道的所有重要事项，并能够实现这些包中可用的各种超参数调整方法。你还将能够理解每个类的重要参数以及它们与我们之前章节中学到的理论之间的关系。最后，凭借前几章的知识，你还将能够理解如果出现错误或意外结果时会发生什么，并了解如何设置方法配置以匹配你的特定问题。

本章将讨论以下主要主题：

+   介绍 DEAP

+   实现遗传算法

+   实现粒子群优化

+   介绍 Microsoft NNI

+   实现网格搜索

+   实现随机搜索

+   实现树结构 Parzen 估计器

+   实现序列模型算法配置

+   实现贝叶斯优化高斯过程

+   实现 Metis

+   实现模拟退火

+   实现 Hyper Band

+   实现贝叶斯优化 Hyper Band

+   实现基于群体的训练

# 技术要求

我们将学习如何使用 DEAP 和 Microsoft NNI 实现各种超参数调整方法。为了确保你能够复制本章中的代码示例，你需要以下条件：

+   Python 3（版本 3.7 或以上）

+   已安装`pandas`包（版本 1.3.4 或以上）

+   已安装`NumPy`包（版本 1.21.2 或以上）

+   已安装`SciPy`包（版本 1.7.3 或以上）

+   已安装`Matplotlib`包（版本 3.5.0 或以上）

+   已安装`scikit-learn`包（版本 1.0.1 或以上）

+   已安装`DEAP`包（版本 1.3）

+   已安装`Hyperopt`包（版本 0.1.2）

+   已安装`NNI`包（版本 2.7）

+   已安装`PyTorch`包（版本 1.10.0）

本章的所有代码示例都可以在 GitHub 上找到：[`github.com/PacktPublishing/Hyperparameter-Tuning-with-Python/blob/main/10_Advanced_Hyperparameter-Tuning-via-DEAP-and-NNI.ipynb`](https://github.com/PacktPublishing/Hyperparameter-Tuning-with-Python/blob/main/10_Advanced_Hyperparameter-Tuning-via-DEAP-and-NNI.ipynb)。

# 介绍 DEAP

执行`pip install deap`命令。

DEAP 允许你以非常灵活的方式构建进化算法的优化步骤。以下步骤展示了如何利用 DEAP 执行任何超参数调整方法。更详细的步骤，包括代码实现，将在接下来的章节中通过各种示例给出：

1.  通过`creator.create()`模块定义*类型*类。这些类负责定义在优化步骤中将使用的对象类型。

1.  定义*初始化器*以及超参数空间，并在`base.Toolbox()`容器中注册它们。初始化器负责设置在优化步骤中将使用的对象的初始值。

1.  定义*算子*并将它们注册在`base.Toolbox()`容器中。算子指的是作为优化算法一部分需要定义的进化工具或**遗传算子**（见*第五章*）。例如，遗传算法中的选择、交叉和变异算子。

1.  定义目标函数并将其注册在`base.Toolbox()`容器中。

1.  定义你自己的超参数调整算法函数。

1.  通过调用定义在*步骤 5*中的函数来执行超参数调整。

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

1.  在测试数据上测试最终训练好的模型。

类型类指的是在优化步骤中使用的对象类型。这些类型类是从 DEAP 中实现的基础类继承而来的。例如，我们可以定义我们的适应度函数类型如下：

```py
from deap import base, creator
```

```py
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
```

`base.Fitness`类是 DEAP 中实现的一个基础抽象类，可以用来定义我们自己的适应度函数类型。它期望一个`weights`参数来理解我们正在处理的优化问题的类型。如果是最大化问题，那么我们必须放置一个正权重，反之亦然，对于最小化问题。请注意，它期望一个元组数据结构而不是浮点数。这是因为 DEAP 还允许我们将`(1.0, -1.0)`作为`weights`参数，这意味着我们有两个目标函数，我们希望第一个最大化，第二个最小化，权重相等。

`creator.create()`函数负责基于基类创建一个新的类。在前面的代码中，我们使用名称“`FitnessMax`”创建了目标函数的类型类。此`creator.create()`函数至少需要两个参数：具体来说，是新创建的类的名称和基类本身。传递给此函数的其他参数将被视为新创建类的属性。除了定义目标函数的类型外，我们还可以定义将要执行的进化算法中个体的类型。以下代码展示了如何创建从 Python 内置的`list`数据结构继承的个体类型，该类型具有`fitness`属性：

```py
creator.create("Individual", list, fitness=creator.FitnessMax)
```

注意，`fitness`属性的类型为`creator.FitnessMax`，这是我们之前代码中刚刚创建的类型。

DEAP 中的类型定义

在 DEAP 中有许多定义类型类的方法。虽然我们已经讨论了最直接且可以说是最常用的类型类，但你可能会遇到需要其他类型类定义的情况。有关如何在 DEAP 中定义其他类型的更多信息，请参阅官方文档（[`deap.readthedocs.io/en/master/tutorials/basic/part1.html`](https://deap.readthedocs.io/en/master/tutorials/basic/part1.html)）。

一旦我们完成了将在优化步骤中使用的对象类型的定义，我们现在需要使用初始化器初始化这些对象的值，并在`base.Toolbox()`容器中注册它们。你可以将此模块视为一个盒子或容器，其中包含初始化器和将在优化步骤中使用的其他工具。以下代码展示了我们如何为个体设置随机的初始值：

```py
import random
```

```py
from deap import tools
```

```py
toolbox = base.Toolbox()
```

```py
toolbox.register("individual",tools.initRepeat,creator.Individual,
```

```py
                 random.random, n=10)
```

前面的代码展示了如何在`base.Toolbox()`容器中注册`"individual"`对象，其中每个个体的尺寸为`10`。该个体是通过重复调用`random.random`方法 10 次生成的。请注意，在超参数调整设置中，每个个体的`10`尺寸实际上指的是我们在空间中拥有的超参数数量。以下展示了通过`toolbox.individual()`方法调用已注册个体的输出：

```py
[0.30752039354315985,0.2491982746819209,0.8423374678316783,0.3401579175109981,0.7699302429041264,0.046433183902334974,0.5287019598616896,0.28081693679292696,0.9562244184741888,0.0008450701833065954]
```

如你所见，`toolbox.individual()`的输出只是一个包含 10 个随机值的列表，因为我们已经定义`creator.Individual`从 Python 内置的`list`数据结构继承。此外，我们在注册个体时也调用了`tools.initRepeat`，通过`random.random`方法重复 10 次。

你现在可能想知道，如何使用这个`toolbox.register()`方法定义实际的超参数空间？启动一串随机值显然没有意义。我们需要知道如何定义将为每个个体配备的超参数空间。为此，我们实际上可以利用 DEAP 提供的另一个工具，即`tools.InitCycle`。

其中`tools.initRepeat`将只调用提供的函数`n`次，在我们之前的例子中，提供的函数是`random.random`。在这里，`tools.InitCycle`期望一个函数列表，并将这些函数调用`n`次。以下代码展示了如何定义将为每个个体配备的超参数空间的一个示例：

1.  我们需要首先注册空间中我们拥有的每个超参数及其分布。请注意，我们也可以将所有必需的参数传递给采样分布函数的`toolbox.register()`。例如，在这里，我们传递了`truncnorm.rvs()`方法的`a=0,b=0.5,loc=0.005,scale=0.01`参数：

    ```py
    from scipy.stats import randint,truncnorm,uniform
    toolbox.register(“param_1”, randint.rvs, 5, 200)
    toolbox.register(“param_2”, truncnorm.rvs, 0, 0.5, 0.005, 0.01)
    toolbox.register(“param_3”, uniform.rvs, 0, 1)
    ```

1.  一旦我们注册了所有现有的超参数，我们可以通过使用`tools.initCycle`并只进行一次重复循环来注册个体：

    ```py
    toolbox.register(“individual”,tools.initCycle,creator.Individual,
        (
            toolbox.param_1,
            toolbox.param_2,
            toolbox.param_3
        ),
        n=1,
    )
    ```

以下展示了通过`toolbox.individual()`方法调用已注册个体的输出：

```py
[172, 0.005840196235159121, 0.37250162585120816]
```

1.  一旦我们在工具箱中注册了个体，注册一个种群就非常简单。我们只需要利用`tools.initRepeat`模块并将定义的`toolbox.individual`作为参数传递。以下代码展示了如何一般性地注册一个种群。请注意，在这里，种群只是之前定义的五个个体的列表：

    ```py
    toolbox.register(“population”, tools.initRepeat, list, toolbox.individual, n=5)
    ```

以下展示了调用`toolbox.population()`方法时的输出：

```py
[[168, 0.009384417146554462, 0.4732188841620628],
[7, 0.009356636359759574, 0.6722125618177741],
[126, 0.00927973696427319, 0.7417964302134438],
[88, 0.008112369078803545, 0.4917555243983919],
[34, 0.008615337472475908, 0.9164442190622125]]
```

如前所述，`base.Toolbox()`容器不仅负责存储初始化器，还负责存储在优化步骤中将使用的其他工具。进化算法（如 GA）的另一个重要构建块是遗传算子。幸运的是，DEAP 已经实现了我们可以通过`tools`模块利用的各种遗传算子。以下代码展示了如何为 GA 注册选择、交叉和变异算子的示例（参见*第五章*)：

```py
# selection strategy
```

```py
toolbox.register("select", tools.selTournament, tournsize=3)
```

```py
# crossover strategy
```

```py
toolbox.register("mate", tools.cxBlend, alpha=0.5)
```

```py
# mutation strategy
```

```py
toolbox.register("mutate", tools.mutPolynomialBounded, eta = 0.1, low=-2, up=2, indpb=0.15)
```

`tools.selTournament`选择策略通过在随机选择的`tournsize`个个体中选出最佳个体，重复`NPOP`次来实现，其中`tournsize`是参加锦标赛的个体数量，而`NPOP`是种群中的个体数量。`tools.cxBlend`交叉策略通过执行两个连续个体基因的线性组合来实现，其中线性组合的权重由`alpha`超参数控制。`tools.mutPolynomialBounded`变异策略通过将连续个体基因传递给一个预定义的多项式映射来实现。

DEAP 中的进化工具

DEAP 中实现了各种内置的进化工具，我们可以根据自己的需求使用，包括初始化器、交叉、变异、选择和迁移工具。有关实现工具的更多信息，请参阅官方文档([`deap.readthedocs.io/en/master/api/tools.html`](https://deap.readthedocs.io/en/master/api/tools.html))。

要将预定义的目标函数注册到工具箱中，我们只需调用相同的`toolbox.register()`方法并传递目标函数，如下面的代码所示：

```py
toolbox.register("evaluate", obj_func)
```

在这里，`obj_func`是一个 Python 函数，它期望接收之前定义的`individual`对象。我们将在接下来的章节中看到如何创建这样的目标函数，以及如何定义我们自己的超参数调整算法函数，当我们讨论如何在 DEAP 中实现 GA 和 PSO 时。

DEAP 还允许我们在调用目标函数时利用我们的并行计算资源。为此，我们只需将`multiprocessing`模块注册到工具箱中，如下所示：

```py
import multiprocessing
```

```py
pool = multiprocessing.Pool()
```

```py
toolbox.register("map", pool.map)
```

一旦我们注册了`multiprocessing`模块，我们就可以在调用目标函数时简单地应用它，如下面的代码所示：

```py
fitnesses = toolbox.map(toolbox.evaluate, individual)
```

在本节中，我们讨论了 DEAP 包及其构建块。你可能想知道如何使用 DEAP 提供的所有构建块构建一个真实的超参数调整方法。不用担心；在接下来的两个章节中，我们将学习如何利用所有讨论的构建块使用 GA 和 PSO 方法进行超参数调整。

# 实现遗传算法

GA 是启发式搜索超参数调整组（见*第五章*）的变体之一，可以通过 DEAP 包实现。为了展示我们如何使用 DEAP 包实现 GA，让我们使用随机森林分类器模型和与*第七章*中示例相同的数据。本例中使用的数据库是 Kaggle 上提供的*Banking Dataset – Marketing Targets*数据库([`www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets`](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets))。

目标变量由两个类别组成，`yes`或`no`，表示银行客户是否已订阅定期存款。因此，在这个数据集上训练机器学习模型的目的是确定客户是否可能想要订阅定期存款。在数据中提供的 16 个特征中，有 7 个数值特征和 9 个分类特征。至于目标类分布，训练和测试数据集中都有 12%是`yes`，88%是`no`。有关数据的更详细信息，请参阅*第七章*。

在执行 GA 之前，让我们看看具有默认超参数值的随机森林分类器是如何工作的。如 *第七章* 所示，我们在测试集上评估具有默认超参数值的随机森林分类器时，F1 分数大约为 `0.436`。请注意，我们仍在使用如 *第七章* 中解释的相同的 scikit-learn 管道定义来训练和评估随机森林分类器。

以下代码展示了如何使用 DEAP 包实现 GA。您可以在 *技术要求* 部分提到的 GitHub 仓库中找到更详细的代码：

1.  通过 `creator.create()` 模块定义 GA 参数和类型类：

    ```py
    # GA Parameters
    NPOP = 50 #population size
    NGEN = 15 #number of trials
    CXPB = 0.5 #cross-over probability
    MUTPB = 0.2 #mutation probability
    ```

设置随机种子以实现可重复性：

```py
import random
random.seed(1)
```

定义我们的适应度函数类型。在这里，我们正在处理一个最大化问题和一个单一目标函数，因此我们设置 `weights=(1.0,)`：

```py
from deap import creator, base
creator.create(“FitnessMax”, base.Fitness, weights=(1.0,))
```

定义从 Python 内置 `list` 数据结构继承的个体类型，该类型具有 `fitness` 作为其属性：

```py
creator.create(“Individual”, list, fitness=creator.FitnessMax)
```

1.  定义初始化器以及超参数空间并将它们注册在 `base.Toolbox()` 容器中。

初始化工具箱：

```py
toolbox = base.Toolbox()
```

定义超参数的命名：

```py
PARAM_NAMES = [“model__n_estimators”,”model__criterion”,
             “model__class_weight”,”model__min_samples_split”
```

注册空间中的每个超参数及其分布：

```py
from scipy.stats import randint,truncnorm
toolbox.register(“model__n_estimators”, randint.rvs, 5, 200)
toolbox.register(“model__criterion”, random.choice, [“gini”, “entropy”])
toolbox.register(“model__class_weight”, random.choice, [“balanced”,”balanced_subsample”])
toolbox.register(“model__min_samples_split”, truncnorm.rvs, 0, 0.5, 0.005, 0.01)
```

通过使用 `tools.initCycle` 仅进行一次循环重复来注册个体：

```py
from deap import tools
toolbox.register(
    “individual”,
    tools.initCycle,
    creator.Individual,
    (
        toolbox.model__n_estimators,
        toolbox.model__criterion,
        toolbox.model__class_weight,
        toolbox.model__min_samples_split,
    ),
)
```

注册种群：

```py
toolbox.register(“population”, tools.initRepeat, list, toolbox.individual)
```

1.  定义操作符并将它们注册在 `base.Toolbox()` 容器中。

注册选择策略：

```py
toolbox.register(“select”, tools.selTournament, tournsize=3)
```

注册交叉策略：

```py
toolbox.register(“mate”, tools.cxUniform, indpb=CXPB)
```

定义一个自定义变异策略。请注意，DEAP 中实现的全部变异策略实际上并不适合超参数调整目的，因为它们只能用于浮点或二进制值，而大多数情况下，我们的超参数空间将是一组真实和离散超参数的组合。以下函数展示了如何实现这样的自定义变异策略。您可以遵循相同的结构来满足您的需求：

```py
def mutPolynomialBoundedMix(individual, eta, low, up, is_int, indpb, discrete_params):
    for i in range(len(individual)):
        if discrete_params[i]:
            if random.random() < indpb:
                individual[i] = random.choice(discrete_params[i])
        else:
            individual[i] = tools.mutPolynomialBounded([individual[i]], 
                                                          eta[i], low[i], up[i], indpb)[0][0]

        if is_int[i]:
            individual[i] = int(individual[i])

    return individual,
```

注册自定义变异策略：

```py
toolbox.register(“mutate”, mutPolynomialBoundedMix, 
                 eta = [0.1,None,None,0.1], 
                 low = [5,None,None,0], 
                 up = [200,None,None,1],
                 is_int = [True,False,False,False],
                 indpb=MUTPB,
                 discrete_params=[[],[“gini”, “entropy”],[“balanced”,”balanced_subsample”],[]]
                )
```

1.  定义目标函数并将其注册在 `base.Toolbox()` 容器中：

    ```py
    def evaluate(individual):
        # convert list of parameter values into dictionary of kwargs
        strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

        if strategy_params['model__min_samples_split'] > 1 or strategy_params['model__min_samples_split'] <= 0:
            return [-np.inf]

        tuned_pipe = clone(pipe).set_params(**strategy_params)
        return [np.mean(cross_val_score(tuned_pipe,X_train_full, y_train, cv=5, scoring='f1',))]
    ```

注册目标函数：

```py
toolbox.register(“evaluate”, evaluate)
```

1.  定义具有并行处理的遗传算法：

    ```py
    import multiprocessing
    import numpy as np
    ```

注册 `multiprocessing` 模块：

```py
pool = multiprocessing.Pool(16)
toolbox.register(“map”, pool.map)
```

定义空数组以存储每个试验中目标函数得分的最佳值和平均值：

```py
mean = np.ndarray(NGEN)
best = np.ndarray(NGEN)
```

定义一个 `HallOfFame` 类，该类负责在种群中存储最新的最佳个体（超参数集）：

```py
hall_of_fame = tools.HallOfFame(maxsize=3)
```

定义初始种群：

```py
pop = toolbox.population(n=NPOP)
```

开始 GA 迭代：

```py
for g in range(NGEN):
```

选择下一代个体/孩子/后代。

```py
    offspring = toolbox.select(pop, len(pop))
```

复制选定的个体。

```py
    offspring = list(map(toolbox.clone, offspring))
```

在后代上应用交叉：

```py
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
```

在后代上应用变异。

```py
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
```

评估具有无效适应度的个体：

```py
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
```

种群完全由后代取代。

```py
    pop[:] = offspring
    hall_of_fame.update(pop)
    fitnesses = [
        ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
    ]
    mean[g] = np.mean(fitnesses)
    best[g] = np.max(fitnesses)
```

1.  通过运行定义的算法在*步骤 5*中执行超参数调整。在运行 GA 之后，我们可以根据以下代码获取最佳超参数集：

    ```py
    params = {}
    for idx_hof, param_name in enumerate(PARAM_NAMES):
        params[param_name] = hall_of_fame[0][idx_hof]
    print(params)
    ```

根据前面的代码，我们得到以下结果：

```py
{'model__n_estimators': 101,
'model__criterion': 'entropy',
'model__class_weight': 'balanced',
'model__min_samples_split': 0.0007106340458649385}
```

我们也可以根据以下代码绘制试验历史或收敛图：

```py
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
fig, ax = plt.subplots(sharex=True, figsize=(8, 6))
sns.lineplot(x=range(NGEN), y=mean, ax=ax, label=”Average Fitness Score”)
sns.lineplot(x=range(NGEN), y=best, ax=ax, label=”Best Fitness Score”)
ax.set_title(“Fitness Score”,size=20)
ax.set_xticks(range(NGEN))
ax.set_xlabel(“Iteration”)
plt.tight_layout()
plt.show()
```

根据前面的代码，以下图生成。如图所示，目标函数得分或适应度得分在整个试验次数中都在增加，因为种群被更新为改进的个体：

![图 10.1 – 遗传算法收敛图]

![img/B18753_10_001.jpg]

图 10.1 – 遗传算法收敛图

1.  使用找到的最佳超参数集在全部训练数据上训练模型：

    ```py
    from sklearn.base import clone
    tuned_pipe = clone(pipe).set_params(**params)
    tuned_pipe.fit(X_train_full,y_train)
    ```

1.  在测试数据上测试最终训练的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

根据前面的代码，当使用最佳超参数集在测试集上测试我们最终的训练随机森林模型时，F1 分数大约为`0.608`。

在本节中，我们学习了如何使用 DEAP 包实现遗传算法（GA），从定义必要的对象开始，到使用并行处理和自定义变异策略定义 GA 过程，再到绘制试验历史和测试测试集中最佳超参数集。在下一节中，我们将学习如何使用 DEAP 包实现 PSO 超参数调整方法。

# 实现粒子群优化

PSO 也是启发式搜索超参数调整组（见*第五章*）的一种变体，可以使用 DEAP 包实现。我们仍将使用上一节中的相同示例来查看我们如何使用 DEAP 包实现 PSO。

以下代码显示了如何使用 DEAP 包实现 PSO。你可以在*技术要求*部分提到的 GitHub 仓库中找到更详细的代码：

1.  通过`creator.create()`模块定义 PSO 参数和类型类：

    ```py
    N = 50 #swarm size
    w = 0.5 #inertia weight coefficient
    c1 = 0.3 #cognitive coefficient
    c2 = 0.5 #social coefficient
    num_trials = 15 #number of trials
    ```

设置随机种子以实现可重复性：

```py
import random
random.seed(1)
```

定义我们的适应度函数的类型。在这里，我们正在处理一个最大化问题和一个单一目标函数，这就是为什么我们设置`weights=(1.0,)`：

```py
from deap import creator, base
creator.create(“FitnessMax”, base.Fitness, weights=(1.0,))
```

定义从 Python 内置的`list`数据结构继承的粒子类型，该结构具有`fitness`、`speed`、`smin`、`smax`和`best`作为其属性。这些属性将在稍后更新每个粒子的位置时被利用（见*第五章*）：

```py
creator.create(“Particle”, list, fitness=creator.FitnessMax,
               speed=list, smin=list, smax=list, best=None)
```

1.  定义初始化器以及超参数空间，并在`base.Toolbox()`容器中注册它们。

初始化工具箱：

```py
toolbox = base.Toolbox()
```

定义超参数的命名：

```py
PARAM_NAMES = [“model__n_estimators”,”model__criterion”,
             “model__class_weight”,”model__min_samples_split”
```

在空间中注册我们拥有的每个超参数及其分布。记住，PSO 只与数值类型超参数一起工作。这就是为什么我们将`"model__criterion"`和`"model__class_weight"`超参数编码为整数：

```py
from scipy.stats import randint,truncnorm
toolbox.register(“model__n_estimators”, randint.rvs, 5, 200)
toolbox.register(“model__criterion”, random.choice, [0,1])
toolbox.register(“model__class_weight”, random.choice, [0,1])
toolbox.register(“model__min_samples_split”, truncnorm.rvs, 0, 0.5, 0.005, 0.01)
```

通过使用`tools.initCycle`仅进行一次重复循环来注册个体。注意，我们还需要将`speed`、`smin`和`smax`值分配给每个个体。为此，让我们定义一个名为`generate`的函数：

```py
from deap import tools
def generate(speed_bound):
    part = tools.initCycle(creator.Particle,
                           [toolbox.model__n_estimators,
                            toolbox.model__criterion,
                            toolbox.model__class_weight,
                            toolbox.model__min_samples_split,
                           ]
                          )
    part.speed = [random.uniform(speed_bound[i]['smin'], speed_bound[i]['smax']) for i in range(len(part))]
    part.smin = [speed_bound[i]['smin'] for i in range(len(part))]
    part.smax = [speed_bound[i]['smax'] for i in range(len(part))]
    return part
```

通过使用`tools.initCycle`仅进行一次重复循环来注册个体：

```py
toolbox.register(“particle”, generate, 
                 speed_bound=[{'smin': -2.5,'smax': 2.5},
                              {'smin': -1,'smax': 1},
                              {'smin': -1,'smax': 1},
                              {'smin': -0.001,'smax': 0.001}])
```

注册种群：

```py
toolbox.register(“population”, tools.initRepeat, list, toolbox.particle)
```

1.  定义操作符并将它们注册到`base.Toolbox()`容器中。PSO 中的主要操作符是粒子的位置更新操作符，该操作符在`updateParticle`函数中定义如下：

    ```py
    import operator
    import math
    def updateParticle(part, best, c1, c2, w, is_int):
        w = [w for _ in range(len(part))]
        u1 = (random.uniform(0, 1)*c1 for _ in range(len(part)))
        u2 = (random.uniform(0, 1)*c2 for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, map(operator.mul, w, part.speed), map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin[i]:
                part.speed[i] = math.copysign(part.smin[i], speed)
            elif abs(speed) > part.smax[i]:
                part.speed[i] = math.copysign(part.smax[i], speed)
        part[:] = list(map(operator.add, part, part.speed))

        for i, pos in enumerate(part):
            if is_int[i]:
                part[i] = int(pos)
    ```

注册操作符。注意，`is_int`属性负责标记哪个超参数具有整数值类型：

```py
toolbox.register(“update”, updateParticle, c1=c1, c2=c2, w=w,
                is_int=[True,True,True,False]
                )
```

1.  定义目标函数并将其注册到`base.Toolbox()`容器中。注意，我们还在目标函数中解码了`"model__criterion"`和`"model__class_weight"`超参数：

    ```py
    def evaluate(particle):
        # convert list of parameter values into dictionary of kwargs
        strategy_params = {k: v for k, v in zip(PARAM_NAMES, particle)}
        strategy_params[“model__criterion”] = “gini” if strategy_params[“model__criterion”]==0 else “entropy”
        strategy_params[“model__class_weight”] = “balanced” if strategy_params[“model__class_weight”]==0 else “balanced_subsample”

        if strategy_params['model__min_samples_split'] > 1 or strategy_params['model__min_samples_split'] <= 0:
            return [-np.inf]

        tuned_pipe = clone(pipe).set_params(**strategy_params)

        return [np.mean(cross_val_score(tuned_pipe,X_train_full, y_train, cv=5, scoring='f1',))]
    ```

注册目标函数：

```py
toolbox.register(“evaluate”, evaluate)
```

1.  定义具有并行处理的 PSO：

    ```py
    import multiprocessing
    import numpy as np
    ```

注册`multiprocessing`模块：

```py
pool = multiprocessing.Pool(16)
toolbox.register(“map”, pool.map)
```

定义空数组以存储每个试验中目标函数分数的最佳和平均值：

```py
mean_arr = np.ndarray(num_trials)
best_arr = np.ndarray(num_trials)
```

定义一个`HallOfFame`类，该类负责存储种群中的最新最佳个体（超参数集）：

```py
hall_of_fame = tools.HallOfFame(maxsize=3)
```

定义初始种群：

```py
pop = toolbox.population(n=NPOP)
```

开始 PSO 迭代：

```py
best = None
for g in range(num_trials):
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for part, fit in zip(pop, fitnesses):
        part.fitness.values = fit

        if not part.best or part.fitness.values > part.best.fitness.values:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if not best or part.fitness.values > best.fitness.values:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values
    for part in pop:
        toolbox.update(part, best)

    hall_of_fame.update(pop)    
    fitnesses = [
        ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
    ]
    mean_arr[g] = np.mean(fitnesses)
    best_arr[g] = np.max(fitnesses)
```

1.  通过运行第 5 步中定义的算法来执行超参数调整。在运行 PSO 之后，我们可以根据以下代码获取最佳超参数集。注意，在将它们传递给最终模型之前，我们需要解码`"model__criterion"`和`"model__class_weight"`超参数：

    ```py
    params = {}
    for idx_hof, param_name in enumerate(PARAM_NAMES):
        if param_name == “model__criterion”:
            params[param_name] = “gini” if hall_of_fame[0][idx_hof]==0 else “entropy”
        elif param_name == “model__class_weight”:
            params[param_name] = “balanced” if hall_of_fame[0][idx_hof]==0 else “balanced_subsample”
        else:
            params[param_name] = hall_of_fame[0][idx_hof]   
    print(params)
    ```

根据前面的代码，我们得到以下结果：

```py
{'model__n_estimators': 75,
'model__criterion': 'entropy',
'model__class_weight': 'balanced',
'model__min_samples_split': 0.0037241038302412493}
```

1.  使用找到的最佳超参数集在全部训练数据上训练模型：

    ```py
    from sklearn.base import clone 
    tuned_pipe = clone(pipe).set_params(**params) 
    tuned_pipe.fit(X_train_full,y_train)
    ```

1.  在测试数据上测试最终训练好的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

根据前面的代码，我们在测试最终训练好的随机森林模型时，在测试集上获得了大约`0.569`的 F1 分数，该模型使用了最佳的超参数集。

在本节中，我们学习了如何使用 DEAP 包实现 PSO，从定义必要的对象开始，将分类超参数编码为整数，并使用并行处理定义优化过程，直到在测试集上测试最佳超参数集。在下一节中，我们将开始学习另一个名为 NNI 的超参数调整包，该包由微软开发。

# 介绍微软 NNI

`pip install nni`命令。

虽然 NNI 指的是*神经网络智能*，但它实际上支持包括但不限于 scikit-learn、XGBoost、LightGBM、PyTorch、TensorFlow、Caffe2 和 MXNet 在内的多个机器学习框架。

NNI 实现了许多超参数调优方法；其中一些是内置的，而另一些则是从其他包如`Hyperopt`（见*第八章*）和`SMAC3`中封装的。在这里，NNI 中的超参数调优方法被称为**调优器**。由于调优器种类繁多，我们不会讨论 NNI 中实现的所有调优器。我们只会讨论在第三章至第六章中讨论过的调优器。除了调优器之外，一些超参数调优方法，如 Hyper Band 和 BOHB，在 NNI 中被视为**顾问**。

NNI 中的可用调优器

要查看 NNI 中所有可用调优器的详细信息，请参阅官方文档页面([`nni.readthedocs.io/en/stable/hpo/tuners.html`](https://nni.readthedocs.io/en/stable/hpo/tuners.html))。

与我们之前讨论的其他超参数调优包不同，在 NNI 中，我们必须准备一个包含模型定义的 Python 脚本，然后才能从笔记本中运行超参数调优过程。此外，NNI 还允许我们从命令行工具中运行超参数调优实验，在那里我们需要定义几个其他附加文件来存储超参数空间信息和其他配置。

以下步骤展示了如何使用纯 Python 代码通过 NNI 执行任何超参数调优过程：

1.  在脚本中准备要调优的模型，例如，`model.py`。此脚本应包括模型架构定义、数据集加载函数、训练函数和测试函数。它还必须包括三个 NNI API 调用，如下所示：

    +   `nni.get_next_parameter()` 负责收集特定试验中要评估的超参数。

    +   `nni.report_intermediate_result()` 负责在每次训练迭代（epoch 或步骤）中报告评估指标。请注意，此 API 调用不是强制的；如果您无法从您的机器学习框架中获取中间评估指标，则不需要此 API 调用。

    +   `nni.report_final_result()` 负责在训练过程完成后报告最终评估指标分数。

1.  定义超参数空间。NNI 期望超参数空间以 Python 字典的形式存在，其中第一级键存储超参数的名称。第二级键存储采样分布的类型和超参数值范围。以下是如何以预期格式定义超参数空间的示例：

    ```py
    hyperparameter_space = {
        ' n_estimators ': {'_type': 'randint', '_value': [5, 200]},
        ' criterion ': {'_type': 'choice', '_value': ['gini', 'entropy']},
        ' min_samples_split ': {'_type': 'uniform', '_value': [0, 0.1]},
    } 
    ```

关于 NNI 的更多信息

关于 NNI 支持的采样分布的更多信息，请参阅官方文档([`nni.readthedocs.io/en/latest/hpo/search_space.html`](https://nni.readthedocs.io/en/latest/hpo/search_space.html))。

1.  接下来，我们需要通过`Experiment`类设置实验配置。以下展示了在我们可以运行超参数调整过程之前设置几个配置的步骤。

加载`Experiment`类。在这里，我们使用的是`'local'`实验模式，这意味着所有训练和超参数调整过程都将在我们的本地计算机上完成。NNI 允许我们在各种平台上运行训练过程，包括但不限于**Azure Machine Learning**（**AML**）、Kubeflow 和 OpenAPI。更多信息，请参阅官方文档([`nni.readthedocs.io/en/latest/reference/experiment_config.html`](https://nni.readthedocs.io/en/latest/reference/experiment_config.html))：

```py
from nni.experiment import Experiment
experiment = Experiment('local')
```

设置试验代码配置。在这里，我们需要指定运行在*步骤 1*中定义的脚本的命令和脚本的相对路径。以下展示了如何设置试验代码配置的示例：

```py
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'
```

设置超参数空间配置。要设置超参数空间配置，我们只需将定义的超参数空间传递到*步骤 2*。以下代码展示了如何进行操作：

```py
experiment.config.search_space = hyperparameter_space
```

设置要使用的超参数调整算法。以下展示了如何将 TPE 作为超参数调整算法应用于最大化问题的示例：

```py
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
```

设置试验次数和并发进程数。NNI 允许我们设置在单次运行中同时评估多少个超参数集。以下代码展示了如何将试验次数设置为 50，这意味着在特定时间将同时评估五个超参数集：

```py
experiment.config.max_trial_number = 50
experiment.config.trial_concurrency = 5
```

值得注意的是，NNI 还允许你根据时间长度而不是试验次数来定义停止标准。以下代码展示了你如何将实验时间限制为 1 小时：

```py
experiment.config.max_experiment_duration = '1h'
```

如果你没有提供`max_trial_number`和`max_experiment_duration`两个参数，那么实验将永远运行，直到你通过*Ctrl + C*命令强制停止它。

1.  运行超参数调整实验。要运行实验，我们可以在`Experiment`类上简单地调用`run`方法。在这里，我们还需要选择要使用的端口。我们可以通过启动的 Web 门户查看实验状态和各种有趣的统计数据。以下代码展示了如何在`local`模式下在端口`8080`上运行实验，这意味着你可以在`http://localhost:8080`上打开 Web 门户：

    ```py
    experiment.run(8080) 
    ```

`run`方法有两个可用的布尔参数，即`wait_completion`和`debug`。当我们设置`wait_completion=True`时，我们无法在实验完成或发现错误之前运行笔记本中的其他单元格。`debug`参数使我们能够选择是否以调试模式启动实验。

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

1.  在测试数据上测试最终训练好的模型。

NNI Web Portal

关于 Web 门户中可用的更多功能，请参阅官方文档（[`nni.readthedocs.io/en/stable/experiment/web_portal/web_portal.html`](https://nni.readthedocs.io/en/stable/experiment/web_portal/web_portal.html)）。注意，我们将在*第十三章*中更详细地讨论 Web 门户，*跟踪超参数调整实验*。

如果你更喜欢使用命令行工具，以下步骤展示了如何使用命令行工具、JSON 和 YAML 配置文件执行任何超参数调整流程：

1.  在脚本中准备要调整的模型。这一步骤与使用纯 Python 代码进行 NNI 超参数调整的前一个流程完全相同。

1.  定义超参数空间。超参数空间的预期格式与使用纯 Python 代码进行任何超参数调整流程的流程完全相同。然而，在这里，我们需要将 Python 字典存储在一个 JSON 文件中，例如，`hyperparameter_space.json`。

1.  通过`config.yaml`文件设置实验配置。需要设置的配置基本上与使用纯 Python 代码的 NNI 流程相同。然而，这里不是通过 Python 类来配置实验，而是将所有配置细节存储在一个单独的 YAML 文件中。以下是一个 YAML 文件示例：

    ```py
    searchSpaceFile: hyperparameter_space.json
    trial_command: python model.py
    trial_code_directory: .

    trial_concurrency: 5
    max_trial_number: 50

    tuner:
      name: TPE
      class_args:
        optimize_mode: maximize

    training_service:
      platform: local
    ```

1.  运行超参数调整实验。要运行实验，我们可以简单地调用`nnictl create`命令。以下代码展示了如何使用该命令在`local`的`8080`端口上运行实验：

    ```py
    nnictl create --config config.yaml --port 8080
    ```

实验完成后，你可以通过`nnictl stop`命令轻松停止进程。

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

1.  在测试数据上测试最终训练好的模型。

各种机器学习框架的示例

你可以在官方文档中找到使用你喜欢的机器学习框架通过 NNI 执行超参数调整的所有示例（[`github.com/microsoft/nni/tree/master/examples/trials`](https://github.com/microsoft/nni/tree/master/examples/trials)）。

scikit-nni

此外，还有一个名为`scikit-nni`的包，它将自动生成所需的`config.yml`和`search-space.json`，并根据你的自定义需求构建`scikit-learn`管道。有关此包的更多信息，请参阅官方仓库（[`github.com/ksachdeva/scikit-nni`](https://github.com/ksachdeva/scikit-nni)）。

除了调优器或超参数调优算法之外，NNI 还提供了 `nni.report_intermediate_result()` API 调用。NNI 中只有两个内置评估器：*中值停止* 和 *曲线拟合*。第一个评估器将在任何步骤中，只要某个超参数集的表现不如中值，就会停止实验。后者评估器将在学习曲线可能收敛到次优结果时停止实验。

在 NNI 中设置评估器非常简单。您只需在 `Experiment` 类或 `config.yaml` 文件中添加配置即可。以下代码展示了如何在 `Experiment` 类上配置中值停止评估器：

```py
experiment.config.assessor.name = 'Medianstop'
```

NNI 中的自定义算法

NNI 还允许我们定义自己的自定义调优器和评估器。为此，您需要继承基类 `Tuner` 或 `Assessor`，编写几个必需的函数，并在 `Experiment` 类或 `config.yaml` 文件中添加更多详细信息。有关如何定义自己的自定义调优器和评估器的更多信息，请参阅官方文档（[`nni.readthedocs.io/en/stable/hpo/custom_algorithm.html`](https://nni.readthedocs.io/en/stable/hpo/custom_algorithm.html)）。

在本节中，我们讨论了 NNI 包及其如何进行一般性的超参数调优实验。在接下来的章节中，我们将学习如何使用 NNI 实现各种超参数调优算法。

# 实现网格搜索

网格搜索是 NNI 包可以实现的穷举搜索超参数调优组（参见 *第三章*）的一种变体。为了向您展示我们如何使用 NNI 包实现网格搜索，我们将使用与上一节示例中相同的数据和管道。然而，在这里，我们将定义一个新的超参数空间，因为 NNI 只支持有限类型的采样分布。

以下代码展示了如何使用 NNI 包实现网格搜索。在这里，我们将使用 NNI 命令行工具（**nnictl**）而不是纯 Python 代码。更详细的代码可以在 *技术要求* 部分提到的 GitHub 仓库中找到：

1.  在脚本中准备要调优的模型。在这里，我们将脚本命名为 `model.py`。该脚本中定义了几个函数，包括 `load_data`、`get_default_parameters`、`get_model` 和 `run`。

`load_data` 函数加载原始数据并将其分为训练数据和测试数据。此外，它还负责返回数值和分类列名的列表：

```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
def load_data():
    df = pd.read_csv(f”{Path(__file__).parent.parent}/train.csv”,sep=”;”)

    #Convert the target variable to integer
    df['y'] = df['y'].map({'yes':1,'no':0})

    #Split full data into train and test data
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=0) 

    #Get list of categorical and numerical features
    numerical_feats = list(df_train.drop(columns='y').select_dtypes(include=np.number).columns)
    categorical_feats = list(df_train.drop(columns='y').select_dtypes(exclude=np.number).columns)

    X_train = df_train.drop(columns=['y'])
    y_train = df_train['y']
    X_test = df_test.drop(columns=['y'])
    y_test = df_test['y']

    return X_train, X_test, y_train, y_test, numerical_feats, categorical_feats
```

`get_default_parameters` 函数返回实验中使用的默认超参数值：

```py
def get_default_parameters():
    params = {
        'model__n_estimators': 5,
        'model__criterion': 'gini',
        'model__class_weight': 'balanced',
        'model__min_samples_split': 0.01,
    }

    return params
```

`get_model` 函数定义了在此示例中使用的 `sklearn` 管道：

```py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
def get_model(PARAMS, numerical_feats, categorical_feats): 
```

为数值特征启动归一化预处理。

```py
    numeric_preprocessor = StandardScaler()
```

为分类特征启动 One-Hot-Encoding 预处理。

```py
    categorical_preprocessor = OneHotEncoder(handle_unknown=”ignore”)
```

创建 ColumnTransformer 类以将每个预处理程序委托给相应的特征。

```py
    preprocessor = ColumnTransformer(
        transformers=[
            (“num”, numeric_preprocessor, numerical_feats),
            (“cat”, categorical_preprocessor, categorical_feats),
        ]
    )
```

创建预处理器和模型的 Pipeline。

```py
    pipe = Pipeline(
        steps=[(“preprocessor”, preprocessor), 
               (“model”, RandomForestClassifier(random_state=0))]
    )
```

设置超参数值。

```py
    pipe = pipe.set_params(**PARAMS)

    return pipe
```

`run` 函数负责训练模型并获取交叉验证分数：

```py
import nni
import logging
from sklearn.model_selection import cross_val_score
LOG = logging.getLogger('nni_sklearn')
def run(X_train, y_train, model):
    model.fit(X_train, y_train)
    score = np.mean(cross_val_score(model,X_train, y_train, 
                    cv=5, scoring='f1')
            )
    LOG.debug('score: %s', score)
    nni.report_final_result(score)
```

最后，我们可以在同一脚本中调用这些函数：

```py
if __name__ == '__main__':
    X_train, _, y_train, _, numerical_feats, categorical_feats = load_data()
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS, numerical_feats, categorical_feats)
        run(X_train, y_train, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
```

1.  在名为 `hyperparameter_space.json` 的 JSON 文件中定义超参数空间：

    ```py
    {“model__n_estimators”: {“_type”: “randint”, “_value”: [5, 200]}, “model__criterion”: {“_type”: “choice”, “_value”: [“gini”, “entropy”]}, “model__class_weight”: {“_type”: “choice”, “_value”: [“balanced”,”balanced_subsample”]}, “model__min_samples_split”: {“_type”: “uniform”, “_value”: [0, 0.1]}}
    ```

1.  通过 `config.yaml` 文件设置实验配置：

    ```py
    searchSpaceFile: hyperparameter_space.json
    experimentName: nni_sklearn
    trial_command: python '/mnt/c/Users/Louis\ Owen/Desktop/Packt/Hyperparameter-Tuning-with-Python/nni/model.py'
    trial_code_directory: .
    trial_concurrency: 10
    max_trial_number: 100 
    maxExperimentDuration: 1h
    tuner: 
      name: GridSearch
    training_service:
      platform: local
    ```

1.  运行超参数调优实验。我们可以通过启动的网络门户查看实验状态和各种有趣的统计数据。以下代码展示了如何在 `local` 模式下通过端口 `8080` 运行实验，这意味着您可以在 `http://localhost:8080` 上打开网络门户：

    ```py
    nnictl create --config config.yaml --port 8080
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。要获取最佳超参数集，您可以访问网络门户并在 *概览* 选项卡中查看。

根据在 *Top trials* 选项卡中显示的实验结果，以下是从实验中找到的最佳超参数值。注意，我们将在 *第十三章* *跟踪超参数调优实验* 中更详细地讨论网络门户：

```py
best_parameters = {
    “model__n_estimators”: 27,
    “model__criterion”: “entropy”,
    “model__class_weight”: “balanced_subsample”,
    “model__min_samples_split”: 0.05
}
```

我们现在可以在全部训练数据上训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_parameters)
# Fit the pipeline on train data 
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练好的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

根据前面的代码，当我们在测试集上使用最佳超参数集测试我们最终的训练 Random Forest 模型时，F1 分数大约为 `0.517`。

在本节中，我们学习了如何通过 `nnictl` 使用 NNI 包实现网格搜索。在下一节中，我们将学习如何通过纯 Python 代码使用 NNI 实现 Random Search。

# 实现 Random Search

随机搜索是穷举搜索超参数调优组（见 *第三章*）的一种变体，NNI 包可以实施。让我们使用与上一节示例中相同的数据、管道和超参数空间，向您展示如何使用纯 Python 代码通过 NNI 实现 Random Search。

以下代码展示了如何使用 NNI 包实现随机搜索。在这里，我们将使用纯 Python 代码而不是像上一节那样使用 `nnictl`。您可以在 *技术要求* 部分提到的 GitHub 仓库中找到更详细的代码：

1.  在脚本中准备要调优的模型。我们将使用与上一节相同的 `model.py` 脚本。

1.  以 Python 字典的形式定义超参数空间：

    ```py
    hyperparameter_space = { 
        'model__n_estimators': {'_type': 'randint', '_value': [5, 200]}, 
        'model__criterion': {'_type': 'choice', '_value': ['gini', 'entropy']}, 
        'model__class_weight': {'_type': 'choice', '_value': [“balanced”,”balanced_subsample”]}, 
        'model__min_samples_split': {'_type': 'uniform', '_value': [0, 0.1]}, 
    }  
    ```

1.  通过 `Experiment` 类设置实验配置。注意，对于随机搜索调优器，只有一个参数，即随机的 `seed` 参数：

    ```py
    experiment = Experiment('local')

    experiment.config.experiment_name = 'nni_sklearn_random_search'
    experiment.config.tuner.name = 'Random'
    experiment.config.tuner.class_args['seed'] = 0

    # Boilerplate code
    experiment.config.trial_command = “python '/mnt/c/Users/Louis\ Owen/Desktop/Packt/Hyperparameter-Tuning-with-Python/nni/model.py'”
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = hyperparameter_space
    experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 10
    experiment.config.max_experiment_duration = '1h'
    ```

1.  运行超参数调优实验：

    ```py
    experiment.run(8080, wait_completion = True, debug = False)
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

```py
best_trial = sorted(experiment.export_data(),key = lambda x: x.value, reverse = True)[0]
print(best_trial.parameter)
```

1.  根据前面的代码，我们得到了以下结果：

    ```py
    {'model__n_estimators': 194, 'model__criterion': 'entropy', 'model__class_weight': 'balanced_subsample', 'model__min_samples_split': 0.0014706304965369289}
    ```

我们现在可以在全部训练数据上训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_trial.parameter)
# Fit the pipeline on train data 
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练好的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

根据前面的代码，当使用最佳超参数集在测试集上测试我们最终训练的随机森林模型时，F1 分数大约为`0.597`。

在本节中，我们学习了如何使用纯 Python 代码通过 NNI 实现随机搜索。在下一节中，我们将学习如何通过纯 Python 代码使用 NNI 实现树结构帕累托估计器。

# 实现树结构帕累托估计器

**树结构帕累托估计器**（**TPEs**）是贝叶斯优化超参数调整组（见*第四章*）中 NNI 包可以实现的变体之一。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯 Python 代码实现 TPE 与 NNI。

以下代码展示了如何使用纯 Python 代码通过 NNI 包实现 TPE。你可以在*技术要求*节中提到的 GitHub 仓库中找到更详细的代码：

1.  在脚本中准备要调优的模型。我们将使用与上一节相同的`model.py`脚本。

1.  以 Python 字典的形式定义超参数空间。我们将使用与上一节相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，TPE 调整器有三个参数：`optimize_mode`、`seed`和`tpe_args`。有关 TPE 调整器参数的更多信息，请参阅官方文档页面（[`nni.readthedocs.io/en/stable/reference/hpo.html#tpe-tuner`](https://nni.readthedocs.io/en/stable/reference/hpo.html#tpe-tuner)）：

    ```py
    experiment = Experiment('local')
    experiment.config.experiment_name = 'nni_sklearn_tpe'
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args = {'optimize_mode': 'maximize', 'seed': 0}

    # Boilerplate code
    # same with previous section
    ```

1.  运行超参数调整实验：

    ```py
    experiment.run(8080, wait_completion = True, debug = False)
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

```py
best_trial = sorted(experiment.export_data(),key = lambda x: x.value, reverse = True)[0]
print(best_trial.parameter)
```

根据前面的代码，我们得到以下结果：

```py
{'model__n_estimators': 195, 'model__criterion': 'entropy', 'model__class_weight': 'balanced_subsample', 'model__min_samples_split': 0.0006636374717157983}
```

我们现在可以在全部训练数据上训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_trial.parameter)
```

在训练数据上拟合管道。

```py
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

根据前面的代码，当使用最佳超参数集在测试集上测试我们最终训练的随机森林模型时，F1 分数大约为`0.618`。

在本节中，我们学习了如何使用纯 Python 代码通过 NNI 实现 TPE。在下一节中，我们将学习如何通过纯 Python 代码使用 NNI 实现序列模型算法配置。

# 实现序列模型算法配置

`pip install "nni[SMAC]"`. 让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯 Python 代码实现 SMAC 与 NNI。

以下代码展示了如何使用纯 Python 代码通过 NNI 包实现 SMAC。你可以在*技术要求*节中提到的 GitHub 仓库中找到更详细的代码：

1.  在脚本中准备要调优的模型。我们将使用与上一节相同的`model.py`脚本。

1.  以 Python 字典的形式定义超参数空间。我们将使用与上一节相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，SMAC 调优器有两个参数：`optimize_mode`和`config_dedup`。有关 SMAC 调优器参数的更多信息，请参阅官方文档页面([`nni.readthedocs.io/en/stable/reference/hpo.html#smac-tuner`](https://nni.readthedocs.io/en/stable/reference/hpo.html#smac-tuner))：

    ```py
    experiment = Experiment('local')

    experiment.config.experiment_name = 'nni_sklearn_smac'
    experiment.config.tuner.name = 'SMAC'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    # Boilerplate code
    # same with previous section
    ```

1.  运行超参数调优实验：

    ```py
    experiment.run(8080, wait_completion = True, debug = False)
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳的超参数组合：

```py
best_trial = sorted(experiment.export_data(),key = lambda x: x.value, reverse = True)[0]
print(best_trial.parameter)
```

根据前面的代码，我们得到了以下结果：

```py
{'model__class_weight': 'balanced', 'model__criterion': 'entropy', 'model__min_samples_split': 0.0005502416428725066, 'model__n_estimators': 199}
```

我们现在可以在全部训练数据上训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_trial.parameter)
# Fit the pipeline on train data 
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练好的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

根据前面的代码，我们在测试集上使用最佳超参数组合测试最终训练好的随机森林模型时，F1 分数大约为`0.619`。

在本节中，我们学习了如何使用纯 Python 代码通过 NNI 实现 SMAC。在下一节中，我们将学习如何通过纯 Python 代码使用 NNI 实现贝叶斯优化高斯过程。

# 实现贝叶斯优化高斯过程

**贝叶斯优化高斯过程**（**BOGP**）是贝叶斯优化超参数调优组（见*第四章*）的变体之一，NNI 包可以实现。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯 Python 代码通过 NNI 实现 BOGP。

以下代码展示了如何使用 NNI 包通过纯 Python 代码实现 BOGP。更详细的代码可以在*技术要求*部分提到的 GitHub 仓库中找到：

1.  在脚本中准备要调优的模型。在这里，我们将使用一个新的脚本，名为`model_numeric.py`。在这个脚本中，我们为非数值超参数添加了一个映射，因为 BOGP 只能处理数值超参数：

    ```py
    non_numeric_mapping = params = {
       'model__criterion': ['gini','entropy'],
       'model__class_weight': ['balanced','balanced_subsample'],
        }
    ```

1.  以 Python 字典的形式定义超参数空间。我们将使用与上一节类似的超参数空间，唯一的区别在于非数值超参数。在这里，所有非数值超参数都被编码为整数值类型：

    ```py
    hyperparameter_space_numeric = { 
        'model__n_estimators': {'_type': 'randint', '_value': [5, 200]}, 
        'model__criterion': {'_type': 'choice', '_value': [0, 1]}, 
        'model__class_weight': {'_type': 'choice', '_value': [0, 1]}, 
        'model__min_samples_split': {'_type': 'uniform', '_value': [0, 0.1]}, 
    }  
    ```

1.  通过`Experiment`类设置实验配置。请注意，BOGP 调优器有九个参数：`optimize_mode`、`utility`、`kappa`、`xi`、`nu`、`alpha`、`cold_start_num`、`selection_num_warm_up`和`selection_num_starting_points`。有关 BOGP 调优器参数的更多信息，请参阅官方文档页面([`nni.readthedocs.io/en/stable/reference/hpo.html#gp-tuner`](https://nni.readthedocs.io/en/stable/reference/hpo.html#gp-tuner))：

    ```py
    experiment = Experiment('local')

    experiment.config.experiment_name = 'nni_sklearn_bogp'
    experiment.config.tuner.name = 'GPTuner'
    experiment.config.tuner.class_args = {
    'optimize_mode': 'maximize', 'utility': 'ei','xi': 0.01}
    # Boilerplate code
    experiment.config.trial_command = “python '/mnt/c/Users/Louis\ Owen/Desktop/Packt/Hyperparameter-Tuning-with-Python/nni/model_numeric.py'”
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = hyperparameter_space_numeric
    experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 10
    experiment.config.max_experiment_duration = '1h'
    ```

1.  运行超参数调优实验：

    ```py
    experiment.run(8080, wait_completion = True, debug = False)
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

```py
non_numeric_mapping = params = {
'model__criterion': ['gini','entropy'],
'model__class_weight': ['balanced','balanced_subsample'],
    }
best_trial = sorted(experiment.export_data(),key = lambda x: x.value, reverse = True)[0]
for key in non_numeric_mapping:
    best_trial.parameter[key] = non_numeric_mapping[key][best_trial.parameter[key]]
print(best_trial.parameter)
```

基于前面的代码，我们得到了以下结果：

```py
{'model__class_weight': 'balanced_subsample', 'model__criterion': 'entropy', 'model__min_samples_split': 0.00055461211818435, 'model__n_estimators': 159}
```

我们现在可以在全部训练数据上训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_trial.parameter)
```

在训练数据上拟合管道。

```py
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练好的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

基于前面的代码，我们在测试集上使用最佳超参数集测试最终训练好的随机森林模型时，F1 分数大约为`0.619`。

在本节中，我们学习了如何使用纯 Python 代码通过 NNI 实现 BOGP。在下一节中，我们将学习如何通过纯 Python 代码使用 NNI 实现 Metis。

# 实现 Metis

**Metis**是贝叶斯优化超参数调整组（参见*第四章*）的一个变体，NNI 包可以实现。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯 Python 代码实现 Metis。

以下代码展示了如何使用纯 Python 代码通过 NNI 包实现 Metis。你可以在*技术要求*部分提到的 GitHub 仓库中找到更详细的代码：

1.  在脚本中准备要调整的模型。这里，我们将使用与上一节相同的脚本`model_numeric.py`，因为 Metis 只能与数值超参数一起工作。

1.  以 Python 字典的形式定义超参数空间。我们将使用与上一节相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，Metis 调整器有六个参数：`optimize_mode`、`no_resampling`、`no_candidates`、`selection_num_starting_points`、`cold_start_num`和`exploration_probability`。有关 Metis 调整器参数的更多信息，请参阅官方文档页面([`nni.readthedocs.io/en/stable/reference/hpo.html#metis-tuner`](https://nni.readthedocs.io/en/stable/reference/hpo.html#metis-tuner))：

    ```py
    experiment = Experiment('local')

    experiment.config.experiment_name = 'nni_sklearn_metis'
    experiment.config.tuner.name = 'MetisTuner'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    # Boilerplate code 
    # same as previous section
    ```

1.  运行超参数调整实验：

    ```py
    experiment.run(8080, wait_completion = True, debug = False)
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

```py
non_numeric_mapping = params = {
'model__criterion': ['gini','entropy'],
'model__class_weight': ['balanced','balanced_subsample'],
    }
best_trial = sorted(experiment.export_data(),key = lambda x: x.value, reverse = True)[0]
for key in non_numeric_mapping:
    best_trial.parameter[key] = non_numeric_mapping[key][best_trial.parameter[key]]
print(best_trial.parameter)
```

基于前面的代码，我们得到了以下结果：

```py
{'model__n_estimators': 122, 'model__criterion': 'gini', 'model__class_weight': 'balanced', 'model__min_samples_split': 0.00173059072806428}
```

我们现在可以在全部训练数据上训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_trial.parameter)
# Fit the pipeline on train data 
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练好的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

基于前面的代码，我们在测试集上使用最佳超参数集测试最终训练好的随机森林模型时，F1 分数大约为`0.590`。

在本节中，我们学习了如何使用纯 Python 代码通过 NNI 实现 Metis。在下一节中，我们将学习如何通过纯 Python 代码使用 NNI 实现模拟退火。

# 实现模拟退火

模拟退火是启发式搜索超参数调整组（参见*第五章*）的一种变体，NNI 包可以实现。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯 Python 代码实现模拟退火。

以下代码展示了如何使用纯 Python 代码通过 NNI 包实现模拟退火。你可以在*技术要求*部分提到的 GitHub 仓库中找到更详细的代码：

1.  在脚本中准备要调整的模型。我们将使用与*实现网格搜索*部分相同的`model.py`脚本。

1.  以 Python 字典的形式定义超参数空间。我们将使用与*实现网格搜索*部分相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，对于模拟退火调整器有一个参数，即`optimize_mode`：

    ```py
    experiment = Experiment('local')

    experiment.config.experiment_name = 'nni_sklearn_anneal'
    experiment.config.tuner.name = 'Anneal'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    # Boilerplate code
    experiment.config.trial_command = “python '/mnt/c/Users/Louis\ Owen/Desktop/Packt/Hyperparameter-Tuning-with-Python/nni/model.py'”
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = hyperparameter_space
    experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 10
    experiment.config.max_experiment_duration = '1h'
    ```

1.  运行超参数调整实验：

    ```py
    experiment.run(8080, wait_completion = True, debug = False)
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳的超参数集：

```py
best_trial = sorted(experiment.export_data(),key = lambda x: x.value, reverse = True)[0]
print(best_trial.parameter)
```

根据前面的代码，我们得到了以下结果：

```py
{'model__n_estimators': 103, 'model__criterion': 'gini', 'model__class_weight': 'balanced_subsample', 'model__min_samples_split': 0.0010101249953063539}
```

我们现在可以使用全部训练数据来训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_trial.parameter)
# Fit the pipeline on train data 
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练好的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

根据前面的代码，当我们在测试集上使用最佳超参数集测试最终训练好的随机森林模型时，F1 分数大约为`0.600`。

在本节中，我们学习了如何使用纯 Python 代码通过 NNI 实现模拟退火。在下一节中，我们将学习如何通过纯 Python 代码实现 Hyper Band。

# 实现 Hyper Band

Hyper Band 是多保真优化超参数调整组（参见*第六章*）的一种变体，NNI 包可以实现。让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯 Python 代码实现 Hyper Band。

以下代码展示了如何使用纯 Python 代码通过 NNI 包实现 Hyper Band。你可以在*技术要求*部分提到的 GitHub 仓库中找到更详细的代码：

1.  在脚本中准备要调整的模型。在这里，我们将使用一个名为`model_advisor.py`的新脚本。在这个脚本中，我们利用`nni.get_next_parameter()`输出的`TRIAL_BUDGET`值来更新`'model__n_estimators'`超参数。

1.  以 Python 字典的形式定义超参数空间。我们将使用与*实现网格搜索*部分类似的超参数空间，但我们将移除`'model__n_estimators'`超参数，因为它将成为 Hyper Band 的预算定义：

    ```py
    hyperparameter_space_advisor = { 
        'model__criterion': {'_type': 'choice', '_value': ['gini', 'entropy']}, 
        'model__class_weight': {'_type': 'choice', '_value': [“balanced”,”balanced_subsample”]}, 
        'model__min_samples_split': {'_type': 'uniform', '_value': [0, 0.1]}, 
    }  
    ```

1.  通过`Experiment`类设置实验配置。请注意，Hyper Band 顾问有四个参数：`optimize_mode`、`R`、`eta`和`exec_mode`。请参考官方文档页面以获取有关 Hyper Band 顾问参数的更多信息（[`nni.readthedocs.io/en/latest/reference/hpo.html`](https://nni.readthedocs.io/en/latest/reference/hpo.html#hyperband-tuner)）：

    ```py
    experiment = Experiment('local')

    experiment.config.experiment_name = 'nni_sklearn_hyper_band'
    experiment.config.advisor.name = 'Hyperband'
    experiment.config.advisor.class_args['optimize_mode'] = 'maximize'
    experiment.config.advisor.class_args['R'] = 200
    experiment.config.advisor.class_args['eta'] = 3
    experiment.config.advisor.class_args['exec_mode'] = 'parallelism'

    # Boilerplate code
    experiment.config.trial_command = “python '/mnt/c/Users/Louis\ Owen/Desktop/Packt/Hyperparameter-Tuning-with-Python/nni/model_advisor.py'”
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = hyperparameter_space_advisor
    experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 10
    experiment.config.max_experiment_duration = '1h'
    ```

1.  运行超参数调优实验：

    ```py
    experiment.run(8080, wait_completion = True, debug = False)
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

```py
best_trial = sorted(experiment.export_data(),key = lambda x: x.value, reverse = True)[0]
best_trial.parameter['model__n_estimators'] = best_trial.parameter['TRIAL_BUDGET'] * 50
del best_trial.parameter['TRIAL_BUDGET']
print(best_trial.parameter)
```

基于前面的代码，我们得到以下结果：

```py
{'model__criterion': 'gini', 'model__class_weight': 'balanced_subsample', 'model__min_samples_split': 0.001676130360763284, 'model__n_estimators': 100}
```

我们现在可以在全部训练数据上训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_trial.parameter)
```

在训练数据上拟合管道。

```py
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

基于前面的代码，我们在使用最佳超参数集在测试集上测试最终训练的随机森林模型时，F1 分数大约为`0.593`。

在本节中，我们学习了如何使用纯 Python 代码通过 NNI 实现 Hyper Band。在下一节中，我们将学习如何通过纯 Python 代码使用 NNI 实现贝叶斯优化超参数搜索。

# 实现贝叶斯优化超参数搜索

**贝叶斯优化超参数搜索**（**BOHB**）是 NNI 包可以实现的 Multi-Fidelity Optimization 超参数调优组的一种变体（参见*第六章*）。请注意，要在 NNI 中使用 BOHB，我们需要使用以下命令安装额外的依赖项：

```py
pip install "nni[BOHB]"
```

让我们使用与上一节示例中相同的数据、管道和超参数空间，使用纯 Python 代码实现 BOHB（贝叶斯优化超参数搜索）。

以下代码展示了如何使用纯 Python 代码通过 NNI 包实现 Hyper Band。更详细的代码可以在*技术要求*部分提到的 GitHub 仓库中找到：

1.  在脚本中准备要调优的模型。我们将使用与上一节相同的`model_advisor.py`脚本。

1.  以 Python 字典的形式定义超参数空间。我们将使用与上一节相同的超参数空间。

1.  通过`Experiment`类设置实验配置。请注意，BOHB 顾问有 11 个参数：`optimize_mode`、`min_budget`、`max_budget`、`eta`、`min_points_in_model`、`top_n_percent`、`num_samples`、`random_fraction`、`bandwidth_factor`、`min_bandwidth`和`config_space`。请参考官方文档页面以获取有关 Hyper Band 顾问参数的更多信息（[`nni.readthedocs.io/en/latest/reference/hpo.html#bohb-tuner`](https://nni.readthedocs.io/en/latest/reference/hpo.html#bohb-tuner)）：

    ```py
    experiment = Experiment('local')

    experiment.config.experiment_name = 'nni_sklearn_bohb'
    experiment.config.advisor.name = 'BOHB'
    experiment.config.advisor.class_args['optimize_mode'] = 'maximize'
    experiment.config.advisor.class_args['max_budget'] = 200
    experiment.config.advisor.class_args['min_budget'] = 5
    experiment.config.advisor.class_args['eta'] = 3
    # Boilerplate code  
    # same as previous section
    ```

1.  运行超参数调优实验：

    ```py
    experiment.run(8080, wait_completion = True, debug = False)
    ```

1.  使用找到的最佳超参数集在全部训练数据上训练模型。

获取最佳超参数集：

```py
best_trial = sorted(experiment.export_data(),key = lambda x: x.value, reverse = True)[0]
best_trial.parameter['model__n_estimators'] = best_trial.parameter['TRIAL_BUDGET'] * 50
del best_trial.parameter['TRIAL_BUDGET']
print(best_trial.parameter)
```

基于前面的代码，我们得到以下结果：

```py
{'model__class_weight': 'balanced', 'model__criterion': 'gini', 'model__min_samples_split': 0.000396569883631686, 'model__n_estimators': 1100}
```

我们现在可以在全部训练数据上训练模型：

```py
from sklearn.base import clone
tuned_pipe = clone(pipe).set_params(**best_trial.parameter)
# Fit the pipeline on train data 
tuned_pipe.fit(X_train_full,y_train)
```

1.  在测试数据上测试最终训练好的模型：

    ```py
    y_pred = tuned_pipe.predict(X_test_full)
    print(f1_score(y_test, y_pred))
    ```

基于前面的代码，我们在测试集上使用最佳超参数集测试最终训练好的随机森林模型时，F1 分数大约为`0.617`。

在本节中，我们学习了如何使用纯 Python 代码实现 NNI 的贝叶斯优化超参数搜索。在下一节中，我们将学习如何通过 `nnictl` 使用 NNI 实现 Population-Based Training。

# 实现基于群体的训练

**基于群体的训练**（**PBT**）是启发式搜索超参数调整组（参见 *第五章*）的变体之一，NNI 包可以实现。为了向您展示如何使用纯 Python 代码通过 NNI 实现 PBT，我们将使用 NNI 包提供的相同示例。在这里，我们使用了 MNIST 数据集和卷积神经网络模型。我们将使用 PyTorch 来实现神经网络模型。有关 NNI 提供的代码示例的详细信息，请参阅 NNI GitHub 仓库（[`github.com/microsoft/nni/tree/1546962f83397710fe095538d052dc74bd981707/examples/trials/mnist-pbt-tuner-pytorch`](https://github.com/microsoft/nni/tree/1546962f83397710fe095538d052dc74bd981707/examples/trials/mnist-pbt-tuner-pytorch)）。

MNIST 数据集

MNIST 是一个手写数字数据集，这些数字已经被标准化并居中在一个固定大小的图像中。在这里，我们将使用 PyTorch 包直接提供的 MNIST 数据集（[`pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST`](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST)）。

以下代码展示了如何使用 NNI 包实现 PBT。在这里，我们将使用 `nnictl` 而不是使用纯 Python 代码。更详细的代码可以在 *技术要求* 部分提到的 GitHub 仓库中找到：

1.  在脚本中准备要调整的模型。在这里，我们将使用来自 NNI GitHub 仓库的相同的 `mnist.py` 脚本。请注意，我们将脚本保存为新的名称：`model_pbt.py`。

1.  在名为 `hyperparameter_space_pbt.json` 的 JSON 文件中定义超参数空间。在这里，我们将使用来自 NNI GitHub 仓库的相同的 `search_space.json` 文件。

1.  通过 `config_pbt.yaml` 文件设置实验配置。请注意，PBT 调优器有六个参数：`optimize_mode`、`all_checkpoint_dir`、`population_size`、`factor`、`resample_probability` 和 `fraction`。有关 PBT 调优器参数的更多信息，请参阅官方文档页面（[`nni.readthedocs.io/en/latest/reference/hpo.html#pbt-tuner`](https://nni.readthedocs.io/en/latest/reference/hpo.html#pbt-tuner)）：

    ```py
    searchSpaceFile: hyperparameter_space_pbt.json
    trialCommand: python '/mnt/c/Users/Louis\ Owen/Desktop/Packt/Hyperparameter-Tuning-with-Python/nni/model_pbt.py'
    trialGpuNumber: 1
    trialConcurrency: 10
    maxTrialNumber: 100
    maxExperimentDuration: 1h
    tuner:
      name: PBTTuner
      classArgs:
        optimize_mode: maximize
    trainingService:
      platform: local
      useActiveGpu: false
    ```

1.  运行超参数调优实验。我们可以通过启动的 Web 门户查看实验状态和各种有趣的统计数据。以下代码展示了如何在`local`模式下运行端口`8080`上的实验，这意味着你可以在`http://localhost:8080`上打开 Web 门户：

    ```py
    nnictl create --config config_pbt.yaml --port 8080
    ```

在本节中，我们学习了如何通过`nnictl`使用 NNI 官方文档中提供的相同示例来实现基于群体的训练。

# 摘要

在本章中，我们学习了关于 DEAP 和 Microsoft NNI 包的所有重要内容。我们还学习了如何借助这些包实现各种超参数调优方法，以及理解每个类的重要参数以及它们与我们之前章节中学到的理论之间的关系。从现在开始，你应该能够利用这些包来实现你选择的超参数调优方法，并最终提高你的机器学习模型性能。凭借第三章至第六章的知识，你还将能够调试代码，如果出现错误或意外结果，并且能够制定自己的实验配置以匹配你的特定问题。

在下一章中，我们将学习几种流行算法的超参数。每个算法都会有广泛的解释，包括但不限于每个超参数的定义、当每个超参数的值发生变化时将产生什么影响，以及基于影响的超参数优先级列表。
