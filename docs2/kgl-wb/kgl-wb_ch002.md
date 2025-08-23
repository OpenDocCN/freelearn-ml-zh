# 第二章：01 最著名的表格竞赛：Porto Seguro 的 Safe Driver Prediction

## 加入我们的 Discord 书社区

[`packt.link/EarlyAccessCommunity`](https://packt.link/EarlyAccessCommunity)

![图片](img/file0.png)

学习如何在任何 Kaggle 竞赛的排行榜上名列前茅需要耐心、勤奋和多次尝试，以便学会最佳的竞赛方式并取得优异成绩。因此，我们想到了一个工作簿，可以通过引导你尝试一些过去的 Kaggle 竞赛，并通过阅读讨论、重用笔记本、特征工程和训练各种模型来帮助你更快地掌握这些技能。

我们在书中从最著名的表格竞赛之一，Porto Seguro 的 Safe Driver Prediction 开始。在这个竞赛中，你需要解决保险中的一个常见问题，即确定下一年谁会提出汽车保险索赔。这样的信息对于提高那些更有可能提出索赔的司机的保险费，以及降低那些不太可能提出索赔的司机的保险费是有用的。

在阐述破解这个竞赛所需的关键洞察和技术细节时，我们将向您展示必要的代码，并要求您研究并回答在 Kaggle 书籍本身中可以找到的主题。因此，无需多言，让我们立即开始您的新学习之旅。

在本章中，你将学习：

+   如何调整和训练 LightGBM 模型。

+   如何构建去噪自编码器以及如何使用它来为神经网络提供数据。

+   如何有效地融合彼此差异很大的模型。

## 理解竞赛和数据

Porto Seguro 是巴西第三大保险公司（它在巴西和乌拉圭运营），提供汽车保险以及其他许多保险产品。在过去 20 年中，他们使用分析方法和机器学习来调整他们的价格，使汽车保险覆盖面更易于更多司机获得。为了探索实现他们任务的新方法，他们赞助了一个竞赛([`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction))，期望 Kagglers 能够提出解决他们核心分析问题的新方法。

该竞赛的目标是让 Kagglers 构建一个模型，预测司机在接下来一年内提出汽车保险索赔的概率，这是一个相当常见的任务（赞助商将其称为“保险的古典挑战”）。为此，赞助商提供了训练集和测试集，并且由于数据集不是很大，看起来准备得非常好，因此对任何人来说都是一个理想的竞赛。

如同在竞赛数据展示页面所述([`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data))：

> *“属于相似分组特征在特征名称中标记为如此（例如，ind，reg，car，calc）。此外，特征名称包括后缀 bin 来指示二元特征，cat 来指示分类特征。没有这些指定的特征要么是连续的，要么是序数的。-1 的值表示该特征在观测中缺失。目标列表示该保单持有人是否提交了索赔”*。

竞赛的数据准备非常仔细，以确保不泄露任何信息，尽管如此，关于特征的含义仍然保持保密，但很明显，可以将不同使用的标签指代到保险建模中常用的特定类型的特征：

+   ind 指代“个人特征”

+   “car”指的是“汽车特征”。

+   calc 指代“计算特征”

+   reg 指代“区域/地理特征”

至于个别特征，在比赛中已经有很多猜测。例如，参见[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/41489`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/41489)或[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/41488`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/41488)，这两篇文章都由 Raddar 撰写，或者再次尝试将特征归因于 Porto Seguro 的在线报价表[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/41057`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/41057)。尽管做出了所有这些努力，但直到现在，特征的含义仍然是个谜。

这个竞赛的有趣事实是：

1.  数据是真实的，尽管特征是匿名的

1.  数据准备得非常好，没有任何泄露（这里没有魔法特征）

1.  测试集不仅与训练测试集具有相同的分类级别，而且似乎来自相同的分布，尽管山本优雅认为使用 t-SNE 对数据进行预处理会导致对抗验证测试失败（[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44784`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44784)）。

> 作为第一次练习，参考 Kaggle Book 中关于对抗验证的内容和代码（从第 179 页开始），证明训练数据和测试数据很可能来自同一数据分布。

Tilii（蒙大拿州立大学的副教授 Mensur Dlakic）的一篇有趣的文章（[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/42197`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/42197)）展示了使用 tSNE 技术，指出“在保险参数方面，许多人非常相似，但其中一些人会提出索赔，而另一些人则不会”。Tilii 提到的情况在保险行业中很典型，即对于某些先验（保险参数）来说，发生某事的概率是相同的，但该事件是否发生则取决于我们观察情况的时间长短。

以保险行业中的物联网和遥测数据为例。分析驾驶员的行为以预测他们未来是否会提出索赔是很常见的。如果你的观察期太短（例如，像这次比赛一样，只有一年），那么即使是非常糟糕的驾驶员也可能不会提出索赔，因为这只是一个不太可能成为现实的低概率事件，而这一事件只有在一定时间后才会发生。类似的观点在 Andy Harless 的讨论中也有提及（[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/42735`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/42735)），他反而认为比赛的真正任务是猜测*“一个潜在连续变量的值，该变量决定了哪些驾驶员更有可能发生事故”*，因为实际上*“提出索赔并不是驾驶员的特征；它是偶然的结果”*。

## 理解评估指标

比赛中使用的指标是“标准化基尼系数”（以经济学中使用的类似基尼系数/指数命名），它之前在另一个比赛中也被使用过，即全美保险公司索赔预测挑战赛（[`www.kaggle.com/competitions/ClaimPredictionChallenge`](https://www.kaggle.com/competitions/ClaimPredictionChallenge)）。从那次比赛，我们可以清楚地了解这个指标的含义：

*当你提交一个条目时，观察结果是从“预测值最大”到“预测值最小”排序的。这是唯一一个你的预测发挥作用的地方，所以只有由你的预测决定的顺序才是重要的。将观察结果从左到右可视化，预测值最大的在左边。然后我们从左到右移动，问“在数据的最左侧 x%中，你积累了多少实际观察到的损失？”如果没有模型，你可以预期在 10%的预测中积累 10%的损失，所以没有任何模型（或“零”模型）会得到一条直线。我们将你的曲线和这条直线之间的区域称为基尼系数*。

*“一个‘完美’模型有一个可达到的最大面积。我们将通过将你的模型的基尼系数除以完美模型的基尼系数来使用归一化基尼系数。”*

竞赛中 Kilian Batzner 的笔记本提供了另一个很好的解释：[`www.kaggle.com/code/batzner/gini-coefficient-an-intuitive-explanation`](https://www.kaggle.com/code/batzner/gini-coefficient-an-intuitive-explanation)，尽管通过图表和玩具示例试图给一个不太常见的度量一个感觉，但在保险公司的精算部门。

> 在 Kaggle 书籍的第五章（第 95 页及以后），我们解释了如何处理比赛度量，特别是如果它们是新的且普遍未知的情况。作为一个练习，你能找出在 Kaggle 上有多少比赛使用了归一化基尼系数作为评估指标吗？

该度量可以通过 Mann–Whitney U 非参数统计检验和 ROC-AUC 分数来近似，因为它大约对应于 2 * ROC-AUC - 1。因此，最大化 ROC-AUC 等同于最大化归一化基尼系数（参见维基百科条目中的“与其他统计度量之间的关系”：[`en.wikipedia.org/wiki/Gini_coefficient`](https://en.wikipedia.org/wiki/Gini_coefficient)）。

该度量也可以近似表示为缩放预测排名与缩放目标值的协方差，从而得到一个更易于理解的排名关联度量（参见 Dmitriy Guller：[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/40576`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/40576)）

从目标函数的角度来看，你可以优化二进制对数损失（就像你在分类问题中做的那样）。ROC-AUC 和归一化基尼系数都不是可微的，它们只能用于验证集上的度量评估（例如，用于早期停止或降低神经网络中的学习率）。然而，优化对数损失并不总是能提高 ROC-AUC 和归一化基尼系数。实际上，存在一个可微的 ROC-AUC 近似（Calders, Toon, 和 Szymon Jaroszewicz. "Efficient AUC optimization for classification." European conference on principles of data mining and knowledge discovery. Springer, Berlin, Heidelberg, 2007 [`link.springer.com/content/pdf/10.1007/978-3-540-74976-9_8.pdf`](https://link.springer.com/content/pdf/10.1007/978-3-540-74976-9_8.pdf)）。然而，似乎在比赛中没有必要使用与对数损失不同的目标函数，以及将 ROC-AUC 或归一化基尼系数作为评估指标。

在这些笔记本中实际上有几个 Python 实现。我们在这里使用了 CPMP 的工作（[`www.kaggle.com/code/cpmpml/extremely-fast-gini-computation/notebook`](https://www.kaggle.com/code/cpmpml/extremely-fast-gini-computation/notebook)），该工作使用 Numba 来加速计算：它既精确又快速。

## 检查 Michael Jahrer 的顶级解决方案的想法

Michael Jahrer ([`www.kaggle.com/mjahrer`](https://www.kaggle.com/mjahrer)，竞赛大师级选手，同时也是“BellKor's Pragmatic Chaos”团队在 Netflix Prize 竞赛中的获奖者之一)，在竞赛期间长时间以较大优势领先公开排行榜，并在最终私人排行榜公布后，被宣布为获胜者。

稍后，在讨论论坛中，他发布了他解决方案的简要总结，由于他对去噪自编码器和神经网络的巧妙使用，这个总结已成为许多 Kagglers 的参考（[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629)）。尽管 Michael 没有附上关于他解决方案的任何 Python 代码（他将其称为“老式”和“低级”的，是直接用 C++/CUDA 编写的，没有使用 Python），但他的写作中充满了对所使用模型的引用，以及它们的超参数和架构。

首先，Michael 解释说，他的解决方案由六个模型（一个 LightGBM 模型和五个神经网络）的组合构成。此外，由于加权每个模型对组合的贡献（以及进行线性和非线性堆叠）可能无法获得优势，这可能是由于过拟合，他声称他转而使用了一个简单的模型组合（算术平均值），这些模型是从不同的种子构建的。

这样的洞察使我们的任务更容易复制他的方法，也因为他提到，仅仅将 LightGBM 的结果与他自己构建的神经网络的其中一个结果混合在一起，就足以保证在竞赛中获得第一名。这将限制我们的练习工作只关注两个优秀的单一模型，而不是一大堆模型。此外，他还提到，他做了很少的数据处理，但删除了一些列并对分类特征进行了独热编码。

## 构建 LightGBM 提交

我们的练习从基于 LightGBM 制定解决方案开始。您可以在以下地址找到已设置好的用于执行的代码：[`www.kaggle.com/code/lucamassaron/workbook-lgb`](https://www.kaggle.com/code/lucamassaron/workbook-lgb)。尽管我们已经提供了代码，但我们建议您直接从书中键入或复制代码并逐个执行代码块，理解每行代码的作用，并且您还可以进一步个性化解决方案，使其表现更佳。

我们首先导入关键包（Numpy、Pandas、Optuna 用于超参数优化、LightGBM 和一些实用函数）。我们还定义了一个配置类并实例化了它。随着我们对代码的探索，我们将讨论配置类中定义的参数。在此需要强调的是，通过使用包含所有参数的类，您将更容易在代码中一致地修改它们。在比赛的紧张时刻，很容易忘记更新在代码中多处引用的参数，并且当参数分散在单元格和函数中时，设置参数总是很困难。配置类可以节省您大量的精力并避免在过程中犯错误。

```py
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from path import Path
from sklearn.model_selection import StratifiedKFold
class Config:
    input_path = Path('../input/porto-seguro-safe-driver-prediction')
    optuna_lgb = False
    n_estimators = 1500
    early_stopping_round = 150
    cv_folds = 5
    random_state = 0
    params = {'objective': 'binary',
              'boosting_type': 'gbdt',
              'learning_rate': 0.01,
              'max_bin': 25,
              'num_leaves': 31,
              'min_child_samples': 1500,
              'colsample_bytree': 0.7,
              'subsample_freq': 1,
              'subsample': 0.7,
              'reg_alpha': 1.0,
              'reg_lambda': 1.0,
              'verbosity': 0,
              'random_state': 0}

config = Config()
```

下一步需要导入训练集、测试集和样本提交数据集。通过使用 pandas csv 读取函数，我们还设置了上传数据框的索引为每个数据示例的标识符（即‘id’列）。

由于属于相似分组特征被标记（使用 ind、reg、car、calc 标签在其标签中）以及二元和分类特征易于定位（它们分别使用 bin 和 cat 标签在其标签中），我们可以枚举它们并将它们记录到列表中。

```py
train = pd.read_csv(config.input_path / 'train.csv', index_col='id')
test = pd.read_csv(config.input_path / 'test.csv', index_col='id')
submission = pd.read_csv(config.input_path / 'sample_submission.csv', index_col='id')
calc_features = [feat for feat in train.columns if "_calc" in feat]
cat_features = [feat for feat in train.columns if "_cat" in feat]
```

然后，我们只提取目标（一个由 0 和 1 组成的二元目标）并将其从训练数据集中删除。

```py
target = train["target"]
train = train.drop("target", axis="columns")
```

到目前为止，正如 Michael Jahrer 所指出的，我们可以删除 calc 特征。这个想法在比赛中反复出现 ([`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/41970`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/41970))，尤其是在笔记本中，一方面是因为经验上可以验证删除它们可以提高公共排行榜的分数，另一方面是因为它们在梯度提升模型中的表现不佳（它们的重要性总是低于平均水平）。我们可以争论，由于它们是工程特征，它们在其原始特征方面不包含新信息，但它们只是向包含它们的任何训练模型添加噪声。

```py
train = train.drop(calc_features, axis="columns")
test = test.drop(calc_features, axis="columns")
```

> 在比赛中，Tilii 使用了 Boruta 进行特征消除测试 ([`github.com/scikit-learn-contrib/boruta_py`](https://github.com/scikit-learn-contrib/boruta_py))。您可以在以下链接找到他的内核：[`www.kaggle.com/code/tilii7/boruta-feature-elimination/notebook`](https://www.kaggle.com/code/tilii7/boruta-feature-elimination/notebook)。如您所查，Boruta 没有将 calc_feature 作为确认的特征。
> 
> > 练习：根据 Kaggle 书籍第 220 页提供的建议（“使用特征重要性评估你的工作”），作为一个练习，为这次比赛编写自己的特征选择笔记本，并检查应该保留哪些特征以及应该丢弃哪些特征。

分类特征则采用独热编码。由于我们想要重新训练它们的标签，并且由于相同的级别在训练和测试集中都存在（这是 Porto Seguro 团队在两个数据集之间进行仔细的训练/测试分割的结果），我们不是使用常规的 Scikit-Learn OneHotEncoder（[`scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)），而是使用 pandas 的 get_dummies 函数（[`pandas.pydata.org/docs/reference/api/pandas.get_dummies.html`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)）。由于 pandas 函数可能会根据特征及其级别在训练集和测试集中不同而产生不同的编码，我们确保独热编码的结果对于两者都是相同的。

```py
train = pd.get_dummies(train, columns=cat_features)
test = pd.get_dummies(test, columns=cat_features)
assert((train.columns==test.columns).all())
```

在对分类特征进行独热编码后，我们已经完成了所有数据处理。我们继续定义我们的评估指标，即正常化基尼系数，正如之前讨论的那样。由于我们打算使用 LightGBM 模型，我们必须添加一个合适的包装器（`gini_lgb`），以便将训练集和验证集的评估以 LightGBM 算法可以处理的形式返回（见：[`lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=higher_better#lightgbm.Booster.eval`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html?highlight=higher_better#lightgbm.Booster.eval) - “每个评估函数应接受两个参数：preds, eval_data，并返回`(eval_name`, `eval_result`, `is_higher_better`)或此类元组的列表”）。

```py
from numba import jit
@jit
def eval_gini(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_pred)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini
def gini_lgb(y_true, y_pred):
    eval_name = 'normalized_gini_coef'
    eval_result = eval_gini(y_true, y_pred)
    is_higher_better = True
    return eval_name, eval_result, is_higher_better
```

关于训练参数，我们发现迈克尔·雅赫尔在其帖子中建议的参数（[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629)）效果极佳。如果您在配置类中将`optuna_lgb`标志设置为 True，您也可以尝试通过 optuna（[`optuna.org/`](https://optuna.org/））进行搜索来找到相同的参数或类似性能的参数。在这里，优化尝试根据训练数据的 5 折交叉验证测试来找到关键参数（如学习率和正则化参数）的最佳值。为了加快速度，我们考虑了在验证过程中的早期停止（我们知道，这实际上可能有利于更好地拟合验证折的一些值 - 一个好的替代方案可能是移除早期停止回调并保持固定的训练轮数）。

```py
if config.optuna_lgb:

    def objective(trial):
        params = {
    'learning_rate': trial.suggest_float("learning_rate", 0.01, 1.0),
    'num_leaves': trial.suggest_int("num_leaves", 3, 255),
    'min_child_samples': trial.suggest_int("min_child_samples", 
                                           3, 3000),
    'colsample_bytree': trial.suggest_float("colsample_bytree", 
                                            0.1, 1.0),
    'subsample_freq': trial.suggest_int("subsample_freq", 0, 10),
    'subsample': trial.suggest_float("subsample", 0.1, 1.0),
    'reg_alpha': trial.suggest_loguniform("reg_alpha", 1e-9, 10.0),
    'reg_lambda': trial.suggest_loguniform("reg_lambda", 1e-9, 10.0),
        }

        score = list()
        skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, 
                              random_state=config.random_state)
        for train_idx, valid_idx in skf.split(train, target):
            X_train = train.iloc[train_idx]
            y_train = target.iloc[train_idx]
            X_valid = train.iloc[valid_idx] 
            y_valid = target.iloc[valid_idx]
            model = lgb.LGBMClassifier(**params,
                                    n_estimators=1500,
                                    early_stopping_round=150,
                                    force_row_wise=True)
            callbacks=[lgb.early_stopping(stopping_rounds=150, 
                                          verbose=False)]
            model.fit(X_train, y_train, 
                      eval_set=[(X_valid, y_valid)],  
                      eval_metric=gini_lgb, callbacks=callbacks)

            score.append(
                model.best_score_['valid_0']['normalized_gini_coef'])
        return np.mean(score)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300)
    print("Best Gini Normalized Score", study.best_value)
    print("Best parameters", study.best_params)

    params = {'objective': 'binary',
              'boosting_type': 'gbdt',
              'verbosity': 0,
              'random_state': 0}

    params.update(study.best_params)

else:
    params = config.params
```

在比赛中，Tilii 测试了使用 Boruta（[`github.com/scikit-learn-contrib/boruta_py`](https://github.com/scikit-learn-contrib/boruta_py)）进行特征消除。你可以在他的核函数中找到：[`www.kaggle.com/code/tilii7/boruta-feature-elimination/notebook`](https://www.kaggle.com/code/tilii7/boruta-feature-elimination/notebook)。正如你可以检查的那样，没有 calc_feature 被 Boruta 视为已确认的特征。

> 在 Kaggle 书中，我们解释了超参数优化（从第 241 页开始）并提供了 LightGBM 模型的一些关键超参数。作为一个练习，尝试通过减少或增加探索的参数以及尝试 Scikit-Learn 中的随机搜索或减半搜索等替代方法来改进超参数搜索（第 245-246 页）。

一旦我们有了最佳参数（或者我们简单地尝试 Jahrer 的参数），我们就可以开始训练和预测。根据最佳解决方案的建议，我们的策略是在每个交叉验证折上训练一个模型，并使用该折来对测试预测的平均值做出贡献。代码片段将生成测试预测和训练集上的出卷预测，这后来对于确定如何集成结果非常有用。

```py
preds = np.zeros(len(test))
oof = np.zeros(len(train))
metric_evaluations = list()
skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
for idx, (train_idx, valid_idx) in enumerate(skf.split(train, 
                                                       target)):
    print(f"CV fold {idx}")
    X_train, y_train = train.iloc[train_idx], target.iloc[train_idx]
    X_valid, y_valid = train.iloc[valid_idx], target.iloc[valid_idx]

    model = lgb.LGBMClassifier(**params,
                               n_estimators=config.n_estimators,
                    early_stopping_round=config.early_stopping_round,
                               force_row_wise=True)

    callbacks=[lgb.early_stopping(stopping_rounds=150), 
               lgb.log_evaluation(period=100, show_stdv=False)]

    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              eval_metric=gini_lgb, callbacks=callbacks)
    metric_evaluations.append(
                model.best_score_['valid_0']['normalized_gini_coef'])
    preds += (model.predict_proba(test,  
              num_iteration=model.best_iteration_)[:,1] 
              / skf.n_splits)
    oof[valid_idx] = model.predict_proba(X_valid,  
                    num_iteration=model.best_iteration_)[:,1]
```

模型训练不应该花费太多时间。最终，你可以在交叉验证过程中报告获得的归一化基尼系数。

```py
print(f"LightGBM CV normalized Gini coefficient:  
        {np.mean(metric_evaluations):0.3f}  
        ({np.std(metric_evaluations):0.3f})")
```

结果相当鼓舞人心，因为平均分数是 0.289，而值的方差相当小。

```py
LightGBM CV Gini Normalized Score: 0.289 (0.015)
```

剩下的工作是将出卷和测试预测保存为提交，并在公共和私人排行榜上验证结果。

```py
submission['target'] = preds
submission.to_csv('lgb_submission.csv')
oofs = pd.DataFrame({'id':train_index, 'target':oof})
oofs.to_csv('dnn_oof.csv', index=False)
```

获得的公共分数应该在 0.28442 左右。相关的私人分数约为 0.29121，将你在最终排行榜上的位置排在第 30 位。这是一个相当好的结果，但我们仍然需要将其与不同的模型，一个神经网络，进行混合。

尽管 Michael Jahrer 在他的帖子中提到，对训练集进行 Bagging（即对训练数据进行多次自助抽样并基于这些自助样本来训练多个模型）应该会增加性能，但增加的幅度并不大。

## 设置去噪自动编码器和深度神经网络

下一步不是设置一个去噪自动编码器（DAE）和一个可以从中学习和预测的神经网络。你可以在以下笔记本中找到运行代码：[`www.kaggle.com/code/lucamassaron/workbook-dae`](https://www.kaggle.com/code/lucamassaron/workbook-dae)。笔记本可以在 GPU 模式下运行（更快），但也可以通过一些轻微的修改在 CPU 上运行。

> 你可以在 Kaggle 书中了解更多关于去噪自动编码器在 Kaggle 比赛中应用的信息，在第 226 页及以后。

实际上，在比赛中使用 DAEs 重现 Michael Jahrer 方法的例子并不多，但我们从 OsciiArt 在另一个比赛中使用的工作 TensorFlow 实现中汲取了经验[`www.kaggle.com/code/osciiart/denoising-autoencoder`](https://www.kaggle.com/code/osciiart/denoising-autoencoder)。

在这里，我们首先导入所有必要的包，特别是 TensorFlow 和 Keras。由于我们将创建多个神经网络，我们指出 TensorFlow 不要使用所有可用的 GPU 内存，通过使用实验性的`set_memory_growth`命令。这将避免在过程中出现内存溢出问题。我们还记录了 Leaky Relu 激活作为自定义激活，这样我们就可以在 Keras 层中通过字符串来提及它。

```py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from path import Path
import gc
import optuna
from sklearn.model_selection import StratifiedKFold
from scipy.special import erfinv
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation, LeakyReLU
get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(alpha=0.2))})
```

与我们创建多个神经网络而不耗尽内存的意图相关，我们还定义了一个简单的函数来清理 GPU 中的内存并移除不再需要的模型。

```py
def gpu_cleanup(objects):
    if objects:
        del(objects)
    K.clear_session()
    gc.collect()
```

我们还重新配置了 Config 类，以便考虑与去噪自动编码器和神经网络相关的多个参数。正如之前关于 LightGBM 所述，将所有参数放在一个地方可以简化以一致方式修改它们。

```py
class Config:
    input_path = Path('../input/porto-seguro-safe-driver-prediction')
    dae_batch_size = 128
    dae_num_epoch = 50
    dae_architecture = [1500, 1500, 1500]
    reuse_autoencoder = False
    batch_size = 128
    num_epoch = 150
    units = [64, 32]
    input_dropout=0.06
    dropout=0.08
    regL2=0.09
    activation='selu'

    cv_folds = 5
    nas = False
    random_state = 0

config = Config()
```

如前所述，我们加载数据集并继续处理特征，通过移除计算特征和将分类特征进行 one-hot 编码。我们保留缺失的案例值为-1，正如 Michael Jahrer 在他的解决方案中指出的。

```py
train = pd.read_csv(config.input_path / 'train.csv', index_col='id')
test = pd.read_csv(config.input_path / 'test.csv', index_col='id')
submission = pd.read_csv(config.input_path / 'sample_submission.csv', index_col='id')
calc_features = [feat for feat in train.columns if "_calc" in feat]
cat_features = [feat for feat in train.columns if "_cat" in feat]
target = train["target"]
train = train.drop("target", axis="columns")
train = train.drop(calc_features, axis="columns")
test = test.drop(calc_features, axis="columns")
train = pd.get_dummies(train, columns=cat_features)
test = pd.get_dummies(test, columns=cat_features)
assert((train.columns==test.columns).all())
```

然而，与我们的先前方法不同，我们必须重新缩放所有非二进制或非 one-hot 编码的分类特征。重新缩放将允许自动编码器和神经网络优化算法更快地收敛到良好的解决方案，因为它将不得不在一个可比较和预定义的范围内处理值。与使用统计归一化不同，GaussRank 是一种允许将转换变量的分布修改为高斯分布的程序。

如某些论文所述，例如在批量归一化论文[`arxiv.org/pdf/1502.03167.pdf`](https://arxiv.org/pdf/1502.03167.pdf)中，如果提供高斯输入，神经网络的表现会更好。根据这篇 NVIDIA 博客文章[`developer.nvidia.com/blog/gauss-rank-transformation-is-100x-faster-with-rapids-and-cupy/`](https://developer.nvidia.com/blog/gauss-rank-transformation-is-100x-faster-with-rapids-and-cupy/)，GaussRank 在大多数情况下都有效，但当特征已经呈正态分布或极端不对称时（在这种情况下应用转换可能会降低性能）。

```py
print("Applying GaussRank to columns: ", end='')
to_normalize = list()
for k, col in enumerate(train.columns):
    if '_bin' not in col and '_cat' not in col and '_missing' not in col:
        to_normalize.append(col)
print(to_normalize)
def to_gauss(x): return np.sqrt(2) * erfinv(x) 
def normalize(data, norm_cols):
    n = data.shape[0]
    for col in norm_cols:
        sorted_idx = data[col].sort_values().index.tolist()
        uniform = np.linspace(start=-0.99, stop=0.99, num=n)
        normal = to_gauss(uniform)
        normalized_col = pd.Series(index=sorted_idx, data=normal)
        data[col] = normalized_col
    return data
train = normalize(train, to_normalize)
test = normalize(test, to_normalize)
```

我们可以在数据集的所有数值特征上分别应用 GaussRank 转换到训练和测试特征：

```py
Applying GaussRank to columns: ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']
```

在归一化特征时，我们只需将我们的数据转换为 float32 类型的 NumPy 数组，这是 GPU 的理想输入。

```py
features = train.columns
train_index = train.index
test_index = test.index
train = train.values.astype(np.float32)
test = test.values.astype(np.float32)
```

接下来，我们只准备一些有用的函数，如评估函数、归一化基尼系数和有助于表示 Keras 模型在训练集和验证集上拟合历史的绘图函数。

```py
def plot_keras_history(history, measures):
    rows = len(measures) // 2 + len(measures) % 2
    fig, panels = plt.subplots(rows, 2, figsize=(15, 5))
    plt.subplots_adjust(top = 0.99, bottom=0.01, 
                        hspace=0.4, wspace=0.2)
    try:
        panels = [item for sublist in panels for item in sublist]
    except:
        pass
    for k, measure in enumerate(measures):
        panel = panels[k]
        panel.set_title(measure + ' history')
        panel.plot(history.epoch, history.history[measure],  
                   label="Train "+measure)
        try:
            panel.plot(history.epoch,  
                       history.history["val_"+measure], 
                       label="Validation "+measure)
        except:
            pass
        panel.set(xlabel='epochs', ylabel=measure)
        panel.legend()

    plt.show(fig)
from numba import jit
@jit
def eval_gini(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_pred)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini
```

接下来的函数实际上要复杂一些，并且与去噪自动编码器和监督神经网络的功能更相关。`batch_generator`是一个函数，它将创建一个生成器，提供基于批次大小的数据块。它实际上不是一个独立的生成器，而是作为我们将要描述的更复杂的批次生成器的一部分，即`mixup_generator`。

```py
def batch_generator(x, batch_size, shuffle=True, random_state=None):
    batch_index = 0
    n = x.shape[0]
    while True:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                np.random.seed(seed=random_state)
                index_array = np.random.permutation(n)
        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0
        batch = x[index_array[current_index: current_index + current_batch_size]]
        yield batch
```

`mixup_generator`是另一个生成器，它返回部分交换值的批次数据，以创建一些噪声并增强数据，以便去噪自动编码器（DAE）不会过度拟合训练集。它基于一个交换率，固定为 Michael Jahrer 建议的 15%的特征。

该函数生成两组不同的数据批次，一组用于释放给模型，另一组则用作在释放批次中交换值的来源。基于随机选择，其基本概率为交换率，在每个批次中，决定交换两个批次之间的一定数量的特征。

这样可以确保去噪自动编码器不能总是依赖于相同的特征（因为它们可能随时被随机交换），但它必须关注所有特征（在某种程度上类似于 dropout）。以便在它们之间找到关系，并在过程结束时正确地重建数据。

```py
def mixup_generator(X, batch_size, swaprate=0.15, shuffle=True, random_state=None):
    if random_state is None:
        random_state = np.randint(0, 999)
    num_features = X.shape[1]
    num_swaps = int(num_features * swaprate)    
    generator_a = batch_generator(X, batch_size, shuffle, 
                                  random_state)
    generator_b = batch_generator(X, batch_size, shuffle, 
                                  random_state + 1)
    while True:
        batch = next(generator_a)
        mixed_batch = batch.copy()
        effective_batch_size = batch.shape[0]
        alternative_batch = next(generator_b)
        assert((batch != alternative_batch).any())
        for i in range(effective_batch_size):
            swap_idx = np.random.choice(num_features, num_swaps, 
                                        replace=False)
            mixed_batch[i, swap_idx] = alternative_batch[i, swap_idx]
        yield (mixed_batch, batch)
```

`get_DAE`函数用于构建去噪自动编码器。它接受一个参数来定义架构，在我们的情况下，已经设置为三个各含 1500 个节点的层（如迈克尔·耶雷尔的建议）。第一层应作为编码器，第二层是瓶颈层，理想情况下包含能够表达数据信息的潜在特征，最后一层是解码层，能够重建初始输入数据。这三层具有 relu 激活函数，没有偏差，每一层后面都跟着一个批量归一化层。最终输出与重建的输入数据具有线性激活。训练使用具有标准设置的 adam 优化器（优化的成本函数是均方误差 - mse）。

```py
def get_DAE(X, architecture=[1500, 1500, 1500]):
    features = X.shape[1]
    inputs = Input((features,))
    for i, nodes in enumerate(architecture):
        layer = Dense(nodes, activation='relu', 
                      use_bias=False, name=f"code_{i+1}")
        if i==0:
            x = layer(inputs)
        else:
            x = layer(x)
        x = BatchNormalization()(x)
    outputs = Dense(features, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', 
                  metrics=['mse', 'mae'])
    return model
```

这里仅报告`extract_dae_features`函数，仅用于教育目的。该函数有助于提取训练去噪自动编码器特定层的值。提取是通过构建一个新的模型来完成的，该模型结合了 DAE 输入层和所需的输出层。然后，简单的预测将提取我们需要的值（预测还允许固定首选的批次大小，以便满足任何内存需求）。

在竞赛的情况下，考虑到观察的数量和从自编码器中提取的特征数量，如果我们使用这个函数，得到的密集矩阵将太大，无法由 Kaggle 笔记本的内存处理。因此，我们的策略不会是将原始数据转换成瓶颈层的自编码器节点值，而是将自编码器与其冻结的层（直到瓶颈层）与监督神经网络融合，正如我们很快将要讨论的。

```py
def extract_dae_features(autoencoder, X, layers=[3]):
    data = []
    for layer in layers:
        if layer==0:
            data.append(X)
        else:
            get_layer_output = Model([autoencoder.layers[0].input], 
                                  [autoencoder.layers[layer].output])
            layer_output = get_layer_output.predict(X, 
                                                    batch_size=128)
            data.append(layer_output)
    data = np.hstack(data)
    return data
```

为了完成与 DAE 的工作，我们有一个最终函数，将所有之前的函数包装成一个无监督的训练过程（至少部分是无监督的，因为有一个早期停止监控器设置在验证集上）。该函数设置混合增强生成器，创建去噪自编码器架构，然后对其进行训练，监控其在验证集上的拟合度以实现早期停止，如果有过度拟合的迹象。最后，在返回训练好的 DAE 之前，它绘制了训练和验证拟合的图表，并将模型存储在磁盘上。

即使我们在该模型上尝试固定一个种子，与 LightGBM 模型相反，结果极其不稳定，它们可能会影响最终的集成结果。虽然结果可能是一个高分，但它可能会在私有排行榜上得分更高或更低，尽管在公共排行榜上获得的结果与公共排行榜非常相关，这将使你能够根据其公共结果始终选择最佳的最终提交。

```py
def autoencoder_fitting(X_train, X_valid, filename='dae',  
                        random_state=None, suppress_output=False):
    if suppress_output:
        verbose = 0
    else:
        verbose = 2
        print("Fitting a denoising autoencoder")
    tf.random.set_seed(seed=random_state)
    generator = mixup_generator(X_train, 
                                batch_size=config.dae_batch_size, 
                                swaprate=0.15, 
                                random_state=config.random_state)

    dae = get_DAE(X_train, architecture=config.dae_architecture)
    steps_per_epoch = np.ceil(X_train.shape[0] / 
                              config.dae_batch_size)
    early_stopping = EarlyStopping(monitor='val_mse', 
                                mode='min', 
                                patience=5, 
                                restore_best_weights=True,
                                verbose=0)
    history = dae.fit(generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=config.dae_num_epoch,
                    validation_data=(X_valid, X_valid),
                    callbacks=[early_stopping],
                    verbose=verbose)
    if not suppress_output: plot_keras_history(history, 
                                           measures=['mse', 'mae'])
    dae.save(filename)
    return dae
```

在处理了 DAE 之后，我们也有机会定义一个监督神经网络模型，该模型应该预测我们的索赔预期。作为第一步，我们定义了一个函数来定义工作中的一个单层：

+   随机正态初始化，因为经验上已经发现这种方法在这个问题中能收敛到更好的结果

+   一个具有 L2 正则化和可参数化激活函数的密集层

+   可排除和可调整的 dropout

下面是创建密集块的代码：

```py
def dense_blocks(x, units, activation, regL2, dropout):
    kernel_initializer = keras.initializers.RandomNormal(mean=0.0, 
                                stddev=0.1, seed=config.random_state)
    for k, layer_units in enumerate(units):
        if regL2 > 0:
            x = Dense(layer_units, activation=activation, 
                      kernel_initializer=kernel_initializer, 
                      kernel_regularizer=l2(regL2))(x)
        else:
            x = Dense(layer_units, 
                      kernel_initializer=kernel_initializer, 
                      activation=activation)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
    return x
```

如您可能已经注意到的，定义单层的函数相当可定制。同样，对于包装架构函数也是如此，它接受层的数量和其中的单元数、dropout 概率、正则化和激活类型作为输入。我们的想法是能够运行神经架构搜索（NAS），并找出在我们的问题中应该表现更好的配置。

关于该函数的最后一句话，在输入中需要提供训练好的 DAE，因为它的输入被用作神经网络模型的输入，而它的第一层与 DAE 的输出相连。这样，我们实际上是将两个模型合并为一个（DAE 的权重已经冻结，不可训练）。这个解决方案是为了避免需要转换所有训练数据，而只需要神经网络处理的单个批次，从而在系统中节省内存。

```py
def dnn_model(dae, units=[4500, 1000, 1000], 
            input_dropout=0.1, dropout=0.5,
            regL2=0.05,
            activation='relu'):

    inputs = dae.get_layer("code_2").output
    if input_dropout > 0:
        x = Dropout(input_dropout)(inputs)
    else:
        x = tf.keras.layers.Layer()(inputs)
    x = dense_blocks(x, units, activation, regL2, dropout)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=dae.input, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.binary_crossentropy,
                metrics=[AUC(name='auc')])
    return model
```

我们以一个训练过程的包装来总结，包括训练整个管道所需的步骤，以交叉验证折进行训练。

```py
def model_fitting(X_train, y_train, X_valid, y_valid, autoencoder, 
                 filename, random_state=None, suppress_output=False):
        if suppress_output:
            verbose = 0
        else:
            verbose = 2
            print("Fitting model")
        early_stopping = EarlyStopping(monitor='val_auc', 
                                    mode='max', 
                                    patience=10, 
                                    restore_best_weights=True,
                                    verbose=0)
        rlrop = ReduceLROnPlateau(monitor='val_auc', 
                                mode='max',
                                patience=2,
                                factor=0.75,
                                verbose=0)

        tf.random.set_seed(seed=random_state)
        model = dnn_model(autoencoder,
                    units=config.units,
                    input_dropout=config.input_dropout,
                    dropout=config.dropout,
                    regL2=config.regL2,
                    activation=config.activation)

        history = model.fit(X_train, y_train, 
                            epochs=config.num_epoch, 
                            batch_size=config.batch_size, 
                            validation_data=(X_valid, y_valid),
                            callbacks=[early_stopping, rlrop],
                            shuffle=True,
                            verbose=verbose)
        model.save(filename)

        if not suppress_output:  
            plot_keras_history(history, measures=['loss', 'auc'])
        return model, history
```

由于我们的 DAE 实现与 Jahrer 的不同，尽管背后的想法相同，我们不能完全依赖他对监督神经网络架构的指示，我们必须寻找理想的架构，就像我们在 LightGBM 模型中寻找最佳超参数一样。使用 Optuna 并利用我们为配置网络架构而设置的多个参数，我们可以运行这个代码片段几个小时，并了解什么可能工作得更好。

在我们的实验中，我们注意到：

+   我们应该使用一个具有少量节点的两层网络，分别是 64 和 32。

+   输入 dropout、层间 dropout 以及一些 L2 正则化确实有所帮助。

+   使用 SELU 激活函数会更好。

这里是运行整个优化实验的代码片段：

```py
if config.nas is True:
    def evaluate():
        metric_evaluations = list()
        skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        for k, (train_idx, valid_idx) in enumerate(skf.split(train, target)):

            X_train, y_train = train[train_idx, :], target[train_idx]
            X_valid, y_valid = train[valid_idx, :], target[valid_idx]
            if config.reuse_autoencoder:
                autoencoder = load_model(f"./dae_fold_{k}")
            else:
                autoencoder = autoencoder_fitting(X_train, X_valid,
                                                filename=f'./dae_fold_{k}', 
                                                random_state=config.random_state,
                                                suppress_output=True)

            model, _ = model_fitting(X_train, y_train, X_valid, y_valid,
                                        autoencoder=autoencoder,
                                        filename=f"dnn_model_fold_{k}", 
                                        random_state=config.random_state,
                                        suppress_output=True)

            val_preds = model.predict(X_valid, batch_size=128, verbose=0)
            best_score = eval_gini(y_true=y_valid, y_pred=np.ravel(val_preds))
            metric_evaluations.append(best_score)

            gpu_cleanup([autoencoder, model])

        return np.mean(metric_evaluations)
    def objective(trial):
        params = {
                'first_layer': trial.suggest_categorical("first_layer", [8, 16, 32, 64, 128, 256, 512]),
                'second_layer': trial.suggest_categorical("second_layer", [0, 8, 16, 32, 64, 128, 256]),
                'third_layer': trial.suggest_categorical("third_layer", [0, 8, 16, 32, 64, 128, 256]),
                'input_dropout': trial.suggest_float("input_dropout", 0.0, 0.5),
                'dropout': trial.suggest_float("dropout", 0.0, 0.5),
                'regL2': trial.suggest_uniform("regL2", 0.0, 0.1),
                'activation': trial.suggest_categorical("activation", ['relu', 'leaky-relu', 'selu'])
        }
        config.units = [nodes for nodes in [params['first_layer'], params['second_layer'], params['third_layer']] if nodes > 0]
        config.input_dropout = params['input_dropout']
        config.dropout = params['dropout']
        config.regL2 = params['regL2']
        config.activation = params['activation']

        return evaluate()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)
    print("Best Gini Normalized Score", study.best_value)
    print("Best parameters", study.best_params)
    config.units = [nodes for nodes in [study.best_params['first_layer'], study.best_params['second_layer'], study.best_params['third_layer']] if nodes > 0]
    config.input_dropout = study.best_params['input_dropout']
    config.dropout = study.best_params['dropout']
    config.regL2 = study.best_params['regL2']
    config.activation = study.best_params['activation']
```

> 如果你想要了解更多关于神经网络架构搜索（NAS）的信息，你可以查看 Kaggle 书，从第 276 页开始。在 DAE 和监督神经网络的情况下，寻找最佳架构至关重要，因为我们正在实施与 Michael Jahrer 解决方案肯定不同的东西。
> 
> > 作为一项练习，尝试通过使用 KerasTuner（可在 Kaggle 书中第 285 页及以后找到）来改进超参数搜索，这是一种优化神经网络的快速解决方案，它得到了 Keras 的创造者 François Chollet 的重要贡献。

在一切准备就绪之后，我们就可以开始训练了。大约一个小时后，在带有 GPU 的 Kaggle 笔记本上，你可以获得完整的测试和跨折预测。

```py
preds = np.zeros(len(test))
oof = np.zeros(len(train))
metric_evaluations = list()
skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
for k, (train_idx, valid_idx) in enumerate(skf.split(train, target)):
    print(f"CV fold {k}")

    X_train, y_train = train[train_idx, :], target[train_idx]
    X_valid, y_valid = train[valid_idx, :], target[valid_idx]
    if config.reuse_autoencoder:
        print("restoring previously trained dae")
        autoencoder = load_model(f"./dae_fold_{k}")
    else:
        autoencoder = autoencoder_fitting(X_train, X_valid,
                                        filename=f'./dae_fold_{k}', 
                                        random_state=config.random_state)

    model, history = model_fitting(X_train, y_train, X_valid, y_valid,
                                autoencoder=autoencoder,
                                filename=f"dnn_model_fold_{k}", 
                                random_state=config.random_state)

    val_preds = model.predict(X_valid, batch_size=128)
    best_score = eval_gini(y_true=y_valid, 
                           y_pred=np.ravel(val_preds))
    best_epoch = np.argmax(history.history['val_auc']) + 1
    print(f"[best epoch is {best_epoch}]\tvalidation_0-gini_dnn: {best_score:0.5f}\n")

    metric_evaluations.append(best_score)
    preds += (model.predict(test, batch_size=128).ravel() / 
              skf.n_splits)
    oof[valid_idx] = model.predict(X_valid, batch_size=128).ravel()
    gpu_cleanup([autoencoder, model])
```

就像我们对 LighGBM 模型所做的那样，我们可以通过查看平均折归一化基尼系数来了解结果。

```py
print(f"DNN CV normalized Gini coefficient: {np.mean(metric_evaluations):0.3f} ({np.std(metric_evaluations):0.3f})")
```

结果不会与之前使用 LightGBM 获得的结果完全一致。

```py
DNN CV Gini Normalized Score: 0.276 (0.015)
```

制作提交并提交将导致公开分数大约为 0.27737，私人分数大约为 0.28471（结果可能与我们之前提到的有很大差异），并不是一个很高的分数。

```py
submission['target'] = preds
submission.to_csv('dnn_submission.csv')
oofs = pd.DataFrame({'id':train_index, 'target':oof})
oofs.to_csv('dnn_oof.csv', index=False)
```

神经网络的少量结果似乎遵循了这样一个谚语：神经网络在表格问题上的表现不佳。无论如何，作为 Kagglers，我们知道所有模型都对在排行榜上取得成功有用，我们只需要找出如何最好地使用它们。当然，一个使用自动编码器的神经网络已经提出了一种受数据噪声影响较小、以不同方式阐述信息的解决方案，这比 GBM 要好。

## 结果集成

现在，拥有两个模型后，剩下的就是将它们混合在一起，看看我们是否能提高结果。正如 Jahrer 所建议的，我们直接尝试将它们混合，但我们并不局限于仅仅产生两个模型的平均值（因为我们的方法最终与 Jahrer 的方法略有不同），我们还将尝试为混合找到最佳权重。我们开始导入折叠外的预测，并准备好我们的评估函数。

```py
import pandas as pd
import numpy as np
from numba import jit
@jit
def eval_gini(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_pred)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini
lgb_oof = pd.read_csv("../input/workbook-lgb/lgb_oof.csv")
dnn_oof = pd.read_csv("../input/workbook-dae/dnn_oof.csv")
target = pd.read_csv("../input/porto-seguro-safe-driver-prediction/train.csv", usecols=['id','target']) 
```

一旦完成，我们将 LightGBM 和神经网络的折叠外预测转换为排名，因为归一化的基尼系数对排名敏感（就像 ROC-AUC 评估一样）。

```py
lgb_oof_ranks = (lgb_oof.target.rank() / len(lgb_oof))
dnn_oof_ranks = (dnn_oof.target.rank() / len(dnn_oof))
```

现在我们只是测试，通过使用不同的权重结合两个模型，我们是否能得到更好的折叠外数据评估。

```py
baseline = eval_gini(y_true=target.target, y_pred=lgb_oof_ranks)
print(f"starting from a oof lgb baseline {baseline:0.5f}\n")
best_alpha = 1.0
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    ensemble = alpha * lgb_oof_ranks + (1.0 - alpha) * dnn_oof_ranks
    score = eval_gini(y_true=target.target, y_pred=ensemble)
    print(f"lgd={alpha:0.1f} dnn={(1.0 - alpha):0.1f} -> {score:0.5f}")

    if score > baseline:
        baseline = score
        best_alpha = alpha

print(f"\nBest alpha is {best_alpha:0.1f}")
```

准备就绪后，通过运行代码片段，我们可以得到有趣的结果：

```py
starting from a oof lgb baseline 0.28850
lgd=0.1 dnn=0.9 -> 0.27352
lgd=0.2 dnn=0.8 -> 0.27744
lgd=0.3 dnn=0.7 -> 0.28084
lgd=0.4 dnn=0.6 -> 0.28368
lgd=0.5 dnn=0.5 -> 0.28595
lgd=0.6 dnn=0.4 -> 0.28763
lgd=0.7 dnn=0.3 -> 0.28873
lgd=0.8 dnn=0.2 -> 0.28923
lgd=0.9 dnn=0.1 -> 0.28916
Best alpha is 0.8
```

看起来，使用强权重（0.8）在 LightGBM 模型上和弱权重（0.2）在神经网络上混合可能会产生一个表现优异的模型。我们立即通过为模型设置相同的权重和我们已经找到的理想权重来测试这个假设。

```py
lgb_submission = pd.read_csv("../input/workbook-lgb/lgb_submission.csv")
dnn_submission = pd.read_csv("../input/workbook-dae/dnn_submission.csv")
submission = pd.read_csv(
"../input/porto-seguro-safe-driver-prediction/sample_submission.csv")
```

首先，我们尝试等权重解决方案：

```py
lgb_ranks = (lgb_submission.target.rank() / len(lgb_submission))
dnn_ranks = (dnn_submission.target.rank() / len(dnn_submission))
submission.target = lgb_ranks * 0.5 + dnn_ranks * 0.5
submission.to_csv("equal_blend_rank.csv", index=False)
```

这导致了公共得分为 0.28393，私人得分为 0.29093，大约位于最终排行榜的第 50 位，离我们的预期有点远。现在让我们尝试使用折叠外预测帮助我们找到的权重：

```py
lgb_ranks = (lgb_submission.target.rank() / len(lgb_submission))
dnn_ranks = (dnn_submission.target.rank() / len(dnn_submission))
submission.target = lgb_ranks * best_alpha +  dnn_ranks * (1.0 - best_alpha)
submission.to_csv("blend_rank.csv", index=False)
```

这里，结果导致公共得分为 0.28502，私人得分为 0.29192，最终在最终排行榜上大约位于第七位。确实是一个更好的结果，因为 LightGBM 是一个很好的模型，但它可能缺少一些数据中的细微差别，这些差别可以通过添加来自在去噪数据上训练的神经网络的某些信息来作为有利的纠正。

> 如 CPMP 在他的解决方案中指出的（[`www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44614`](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44614)），根据如何构建你的交叉验证，你可能会经历“折叠间基尼分数的巨大变化”。因此，CPMP 建议通过使用多个交叉验证的许多不同种子来减少估计的方差，并平均结果。
> 
> > 练习：作为一个练习，尝试修改我们使用的代码，以便创建更稳定的预测，特别是对于去噪自编码器。

## 摘要

在本章中，你已经处理了一个经典的表格竞赛。通过阅读竞赛的笔记本和讨论，我们提出了一种简单的解决方案，仅涉及两个易于混合的模型。特别是，我们提供了一个示例，说明如何使用去噪自编码器来生成一种特别适用于处理表格数据的替代数据处理方法。通过理解和复制过去竞赛中的解决方案，你可以在 Kaggle 竞赛中快速建立你的核心能力，并迅速在最近的竞赛和挑战中表现出色和稳定。

在下一章中，我们将探索 Kaggle 的另一个表格竞赛，这次是关于一个复杂的时间序列预测问题。
