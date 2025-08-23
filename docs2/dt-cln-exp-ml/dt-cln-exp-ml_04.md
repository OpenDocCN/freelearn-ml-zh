# 第三章：*第三章*：识别和修复缺失值

当我说，很少有看似微小且微不足道的事情像缺失值那样具有如此重大的影响时，我想我代表了许多数据科学家。我们花费大量时间担心缺失值，因为它们可能会对我们的分析产生戏剧性和出人意料的效应。这种情况最有可能发生在缺失值不是随机的情况下——也就是说，当它们与一个特征或目标相关时。例如，假设我们正在进行一项关于收入的纵向研究，但受教育程度较低的人更有可能在每年跳过收入问题。这有可能导致我们对教育参数估计的偏差。

当然，识别缺失值甚至不是战斗的一半。接下来，我们需要决定如何处理它们。我们是否要删除具有一个或多个特征缺失值的任何观测值？我们是否基于样本的统计量，如平均值，来插补一个值？或者我们是否基于更具体的统计量，如某个特定类别的平均值，来分配一个值？我们是否认为对于时间序列或纵向数据，最近的时序值可能最有意义？或者我们应该使用更复杂的多元技术来插补值，比如基于线性回归或**k 近邻**（**KNN**）？

所有这些问题的答案都是*是的*。在某个时候，我们将想要使用这些技术中的每一个。当我们做出关于缺失值插补的最终选择时，我们希望能够回答为什么或为什么不选择所有这些可能性。根据具体情况，每个答案都有意义。

在本章中，我们将探讨识别每个特征或目标缺失值的技巧，以及对于大量特征值缺失的观测值的策略。然后，我们将探讨插补值的策略，例如将值设置为整体平均值、给定类别的平均值或前向填充。我们还将检查用于插补缺失值的多元技术，并讨论它们何时适用。

特别地，在本章中，我们将涵盖以下主题：

+   识别缺失值

+   清理缺失值

+   使用回归进行值插补

+   使用 KNN 插补

+   使用随机森林进行插补

# 技术要求

本章将大量依赖 pandas 和 NumPy 库，但你不需要对这些库有任何先前的知识。如果你从科学发行版，如 Anaconda 或 WinPython 安装了 Python，这些库可能已经安装好了。我们还将使用`statsmodels`库进行线性回归，以及来自`sklearn`和`missingpy`的机器学习算法。如果你需要安装这些包中的任何一个，你可以通过在终端窗口或 Windows PowerShell 中运行`pip install [package name]`来安装。

# 识别缺失值

由于识别缺失值是分析师工作流程中如此重要的一个部分，我们使用的任何工具都需要使其能够轻松地定期检查这些值。幸运的是，pandas 使得识别缺失值变得相当简单。

我们将处理 `weeksworked16` 和 `weeksworked17`，分别代表 2016 年和 2017 年的工作周数。

注意

我们还将再次处理 COVID-19 数据。这个数据集为每个国家提供了一个观测值，指定了总 COVID-19 病例和死亡人数，以及每个国家的某些人口统计数据。

按照以下步骤来识别我们的缺失值：

1.  让我们从加载 NLS 和 COVID-19 数据开始：

    ```py
    import pandas as pd
    import numpy as np
    nls97 = pd.read_csv("data/nls97b.csv")
    nls97.set_index("personid", inplace=True)
    covidtotals = pd.read_csv("data/covidtotals.csv")
    covidtotals.set_index("iso_code", inplace=True)
    ```

1.  接下来，我们计算可能用作特征的列的缺失值数量。我们可以使用 `isnull` 方法来测试每个特征值是否缺失。如果值缺失，则返回 `True`，如果不缺失则返回 `False`。然后，我们可以使用 `sum` 来计算 `True` 值的数量，因为 `sum` 将每个 `True` 值视为 1，每个 `False` 值视为 0。我们使用 `axis=0` 来对每个列的行进行求和：

    ```py
    covidtotals.shape
    (221, 16)
    demovars = ['population_density','aged_65_older',
      'gdp_per_capita', 'life_expectancy', 
      'diabetes_prevalence']
    covidtotals[demovars].isnull().sum(axis=0)
    population_density        15
    aged_65_older             33
    gdp_per_capita            28
    life_expectancy            4
    diabetes_prevalence       21
    ```

如我们所见，221 个国家中有 33 个国家的 `aged_65_older` 有空值。我们几乎对所有国家的 `life_expectancy` 都有数据。

1.  如果我们想要每行的缺失值数量，我们可以在求和时指定 `axis=1`。以下代码创建了一个 Series，`demovarsmisscnt`，包含每个国家人口统计特征的缺失值数量。181 个国家所有特征都有值，11 个国家缺失五个特征中的四个，三个国家缺失所有特征：

    ```py
    demovarsmisscnt = covidtotals[demovars].isnull().sum(axis=1)
    demovarsmisscnt.value_counts().sort_index()
    0        181
    1        15
    2         6
    3         5
    4        11
    5         3
    dtype: int64
    ```

1.  让我们看看有四个或更多缺失值的几个国家。这些国家的人口统计数据非常少：

    ```py
    covidtotals.loc[demovarsmisscnt > = 4, ['location'] +
      demovars].sample(6, random_state=1).T
    iso_code                         FLK   NIU        MSR\
    location            Falkland Islands  Niue  Montserrat
    population_density               NaN   NaN         NaN
    aged_65_older                    NaN   NaN         NaN
    gdp_per_capita                   NaN   NaN         NaN
    life_expectancy                   81    74          74
    diabetes_prevalence              NaN   NaN         NaN
    iso_code                         COK    SYR        GGY
    location                Cook Islands  Syria   Guernsey
    population_density               NaN    NaN        NaN
    aged_65_older                    NaN    NaN        NaN
    gdp_per_capita                   NaN    NaN        NaN
    life_expectancy                   76      7        NaN
    diabetes_prevalence              NaN    NaN        NaN
    ```

1.  让我们也检查一下总病例和死亡病例的缺失值。29 个国家在每百万人口中的病例数有缺失值，36 个国家每百万死亡人数有缺失值：

    ```py
    totvars = 
      ['location','total_cases_mill','total_deaths_mill']
    covidtotals[totvars].isnull().sum(axis=0)
    location                0
    total_cases_mill       29
    total_deaths_mill      36
    dtype: int64
    ```

1.  我们还应该了解哪些国家同时缺失这两个数据。29 个国家同时缺失病例和死亡数据，而我们只有 185 个国家同时拥有这两个数据：

    ```py
    totvarsmisscnt = 
      covidtotals[totvars].isnull().sum(axis=1)
    totvarsmisscnt.value_counts().sort_index()
    0        185
    1        7
    2        29
    dtype: int64
    ```

有时，我们会有逻辑缺失值，需要将其转换为实际缺失值。这发生在数据集设计者使用有效值作为缺失值的代码时。这些值通常是 9、99 或 999 等数值，基于变量的允许数字位数。或者可能是一个更复杂的编码方案，其中存在不同原因导致的缺失值的代码。例如，在 NLS 数据集中，代码揭示了受访者为什么没有回答某个问题的原因：`-3` 是无效跳过，`-4` 是有效跳过，而 `-5` 是非访谈。

1.  NLS DataFrame 的最后四列包含受访者母亲和父亲完成的最高学历、家庭收入以及受访者出生时母亲的年龄的数据。让我们从受访者母亲完成的最高的学历开始，检查这些列的逻辑缺失值：

    ```py
    nlsparents = nls97.iloc[:,-4:]
    nlsparents.shape
    (8984, 4)
    nlsparents.loc[nlsparents.motherhighgrade.between(-5, 
      -1), 'motherhighgrade'].value_counts()
    -3        523
    -4        165
    Name: motherhighgrade, dtype: int64
    ```

1.  有 523 个无效跳过和 165 个有效跳过。让我们看看这四个特征中至少有一个非响应值的几个个体：

    ```py
    nlsparents.loc[nlsparents.apply(lambda x: x.between(
      -5,-1)).any(axis=1)]
            motherage  parentincome  fatherhighgrade  motherhighgrade
    personid  
    100284    22            50000            12         -3
    100931    23            60200            -3         13
    101122    25               -4            -3         -3
    101414    27            24656            10         -3
    101526    -3            79500            -4         -4
             ...              ...            ...       ...
    999087    -3           121000            -4         16
    999103    -3            73180            12         -4
    999406    19               -4            17         15
    999698    -3            13000            -4         -4
    999963    29               -4            12         13
    [3831 rows x 4 columns]
    ```

1.  对于我们的分析，非响应的原因并不重要。让我们只计算每个特征的非响应数量，无论非响应的原因是什么：

    ```py
    nlsparents.apply(lambda x: x.between(-5,-1).sum())
    motherage              608
    parentincome          2396
    fatherhighgrade       1856
    motherhighgrade        688
    dtype: int64
    ```

1.  在使用这些特征进行分析之前，我们应该将这些值设置为`missing`。我们可以使用`replace`将-5 到-1 之间的所有值设置为`missing`。当我们检查实际缺失值时，我们得到预期的计数：

    ```py
    nlsparents.replace(list(range(-5,0)), 
      np.nan, inplace=True)
    nlsparents.isnull().sum()
    motherage            608
    parentincome        2396
    fatherhighgrade     1856
    motherhighgrade      688
    dtype: int64
    ```

本节展示了识别每个特征的缺失值数量以及具有大量缺失值的观测值的一些非常有用的 pandas 技术。我们还学习了如何查找逻辑缺失值并将它们转换为实际缺失值。接下来，我们将首次探讨清理缺失值。

# 清理缺失值

在本节中，我们将介绍一些处理缺失值的最直接方法。这包括删除存在缺失值的观测值；将样本的汇总统计量，如平均值，分配给缺失值；以及根据数据适当子集的平均值分配值：

1.  让我们加载 NLS 数据并选择一些教育数据：

    ```py
    import pandas as pd
    nls97 = pd.read_csv("data/nls97b.csv")
    nls97.set_index("personid", inplace=True)
    schoolrecordlist = 
      ['satverbal','satmath','gpaoverall','gpaenglish',
      'gpamath','gpascience','highestdegree',
      'highestgradecompleted']
    schoolrecord = nls97[schoolrecordlist]
    schoolrecord.shape
    (8984, 8)
    ```

1.  我们可以使用在前一节中探讨的技术来识别缺失值。`schoolrecord.isnull().sum(axis=0)`为我们提供了每个特征的缺失值数量。绝大多数观测值在`satverbal`上存在缺失值，共有 7,578 个，占 8,984 个观测值中的大部分。只有 31 个观测值在`highestdegree`上存在缺失值：

    ```py
    schoolrecord.isnull().sum(axis=0)
    satverbal                        7578
    satmath                          7577
    gpaoverall                       2980
    gpaenglish                       3186
    gpamath                          3218
    gpascience                       3300
    highestdegree                      31
    highestgradecompleted            2321
    dtype: int64
    ```

1.  我们可以创建一个 Series，`misscnt`，它指定了每个观测值的缺失特征数量，`misscnt = schoolrecord.isnull().sum(axis=1)`。946 个观测值在教育数据上有七个缺失值，而 11 个观测值的所有八个特征都缺失：

    ```py
    misscnt = schoolrecord.isnull().sum(axis=1)
    misscnt.value_counts().sort_index()
    0         1087
    1          312
    2         3210
    3         1102
    4          176
    5          101
    6         2039
    7          946
    8           11
    dtype: int64
    ```

1.  让我们再看看一些具有七个或更多缺失值的观测值。看起来`highestdegree`通常是唯一存在的特征，这并不令人惊讶，因为我们已经发现`highestdegree`很少缺失：

    ```py
    schoolrecord.loc[misscnt>=7].head(4).T
    personid              101705  102061  102648  104627
    satverbal                NaN     NaN     NaN     NaN
    satmath                  NaN     NaN     NaN     NaN
    gpaoverall               NaN     NaN     NaN     NaN
    gpaenglish               NaN     NaN     NaN     NaN
    gpamath                  NaN     NaN     NaN     NaN
    gpascience               NaN     NaN     NaN     NaN
    highestdegree          1.GED  0.None   1.GED  0.None
    highestgradecompleted    NaN     NaN     NaN     NaN
    ```

1.  让我们删除在八个特征中至少有七个缺失值的观测值。我们可以通过将`dropna`的`thresh`参数设置为`2`来实现这一点。这将删除具有少于两个非缺失值的观测值；也就是说，0 个或 1 个非缺失值。使用`dropna`后，我们得到预期的观测值数量；即，8,984 - 946 - 11 = 8,027：

    ```py
    schoolrecord = schoolrecord.dropna(thresh=2)
    schoolrecord.shape
    (8027, 8)
    schoolrecord.isnull().sum(axis=1).value_counts().sort_index()
    0      1087
    1       312
    2      3210
    3      1102
    4       176
    5       101
    6      2039
    dtype: int64
    ```

`gpaoverall`有相当数量的缺失值——即 2,980 个——尽管我们有三分之二的观测值是有效的（(8,984 - 2,980)/8,984）。如果我们能很好地填充缺失值，我们可能能够将其作为特征挽救。这比仅仅删除这些观测值更可取。如果我们能避免，我们不想失去这些数据，尤其是如果缺失`gpaoverall`的个体在其他方面与我们预测相关的方面有所不同。

1.  最直接的方法是将`gpaoverall`的整体平均值分配给缺失值。以下代码使用 pandas Series 的`fillna`方法将`gpaoverall`的所有缺失值分配给 Series 的平均值。`fillna`的第一个参数是你想要为所有缺失值设置的值——在这种情况下，`schoolrecord.gpaoverall.mean()`。请注意，我们需要记住将`inplace`参数设置为`True`以覆盖现有值：

    ```py
    schoolrecord.gpaoverall.agg(['mean','std','count'])
    mean         281.84
    std           61.64
    count      6,004.00
    Name: gpaoverall, dtype: float64
    schoolrecord.gpaoverall.fillna(
      schoolrecord.gpaoverall.mean(), inplace=True)
    schoolrecord.gpaoverall.isnull().sum()
    0
    schoolrecord.gpaoverall.agg(['mean','std','count'])
    mean      281.84
    std        53.30
    count   8,027.00
    Name: gpaoverall, dtype: float64
    ```

平均值没有变化。然而，标准差有显著下降，从 61.6 下降到 53.3。这是使用数据集的平均值填充所有缺失值的一个缺点。

1.  NLS 数据中`wageincome`也有相当数量的缺失值。以下代码显示有 3,893 个观测值有缺失值：

    ```py
    wageincome = nls97.wageincome.copy(deep=True)
    wageincome.isnull().sum()
    copy method, setting deep to True. We wouldn't normally do this but, in this case, we don't want to change the values of wageincome in the underlying DataFrame. We have avoided this here because we will demonstrate a different method of imputing values in the next couple of code blocks.
    ```

1.  我们与其将`wageincome`的平均值分配给缺失值，不如使用另一种常见的值填充技术：我们可以将前一个观测值中的最近非缺失值分配给缺失值。`fillna`的`ffill`选项会为我们完成这项工作：

    ```py
    wageincome.fillna(method='ffill', inplace=True)
    wageincome.head().T
    personid
    100061       12,500
    100139      120,000
    100284       58,000
    100292       58,000
    100583       30,000
    Name: wageincome, dtype: float64
    wageincome.isnull().sum()
    0
    wageincome.agg(['mean','std','count'])
    mean      49,549.33
    std       40,014.34
    count      8,984.00
    Name: wageincome, dtype: float64
    ```

1.  我们可以通过将`fillna`的`method`参数设置为`bfill`来执行向后填充。这会将缺失值设置为最近的后续值。这会产生以下输出：

    ```py
    wageincome = nls97.wageincome.copy(deep=True)
    wageincome.std()
    40677.69679818673
    wageincome.fillna(method='bfill', inplace=True)
    wageincome.head().T
    personid
    100061       12,500
    100139      120,000
    100284       58,000
    100292       30,000
    100583       30,000
    Name: wageincome, dtype: float64
    wageincome.agg(['mean','std','count'])
    mean    49,419.05
    std     41,111.54
    count    8,984.00
    Name: wageincome, dtype: float64
    ```

如果缺失值是随机分布的，那么向前填充或向后填充与使用平均值相比有一个优点：它更有可能近似该特征的非缺失值的分布。请注意，标准差并没有大幅下降。

有时候，基于相似观测值的平均值或中位数来填充值是有意义的；比如说，那些对于相关特征具有相同值的观测值。如果我们正在为特征 X1 填充值，而 X1 与 X2 相关，我们可以利用 X1 和 X2 之间的关系来为 X1 填充一个可能比数据集的平均值更有意义的值。当 X2 是分类变量时，这通常很简单。在这种情况下，我们可以为 X2 的关联值填充 X1 的平均值。

1.  在 NLS DataFrame 中，2017 年的工作周数与获得的最高学位相关。以下代码显示了工作周数的平均值如何随着学位获得而变化。工作周数的平均值是 39，但没有学位的人（28.72）要低得多，而有专业学位的人（47.20）要高得多。在这种情况下，将 28.72 分配给未获得学位的个人缺失的工作周数，而不是 39，可能是一个更好的选择：

    ```py
    nls97.weeksworked17.mean()
    39.01664167916042
    nls97.groupby(['highestdegree'])['weeksworked17'
      ].mean()
    highestdegree
    0\. None                  28.72
    1\. GED                   34.59
    2\. High School           38.15
    3\. Associates            40.44
    4\. Bachelors             43.57
    5\. Masters               45.14
    6\. PhD                   44.31
    7\. Professional          47.20
    Name: weeksworked17, dtype: float64
    ```

1.  以下代码将缺失工作周数的观测值的平均工作周数分配给具有相同学位获得水平的观测值。我们通过使用`groupby`创建一个按`highestdegree`分组的 DataFrame，即`groupby(['highestdegree'])['weeksworked17']`来实现这一点。然后，我们在`apply`中使用`fillna`来填充这些缺失值，用最高学位组的平均值来填充。请注意，我们确保只对最高学位不缺失的观测值进行插补，`~nls97.highestdegree.isnull()`。对于既缺失最高学位又缺失工作周数的观测值，我们仍然会有缺失值：

    ```py
    nls97.loc[~nls97.highestdegree.isnull(),
      'weeksworked17imp'] = 
      nls97.loc[ ~nls97.highestdegree.isnull() ].
      groupby(['highestdegree'])['weeksworked17'].
      apply(lambda group: group.fillna(np.mean(group)))
    nls97[['weeksworked17imp','weeksworked17',
      'highestdegree']].head(10)
           weeksworked17imp  weeksworked17   highestdegree
    personid                                              
    100061            48.00         48.00   2\. High School
    100139            52.00         52.00   2\. High School
    100284             0.00          0.00          0\. None
    100292            43.57           NaN     4\. Bachelors
    100583            52.00         52.00   2\. High School
    100833            47.00         47.00   2\. High School
    100931            52.00         52.00    3\. Associates
    101089            52.00         52.00   2\. High School
    101122            38.15           NaN   2\. High School
    101132            44.00         44.00          0\. None
    nls97[['weeksworked17imp','weeksworked17']].\
      agg(['mean','count'])
           weeksworked17imp  weeksworked17
    mean          38.52         39.02
    count      8,953.00      6,670.00
    ```

这些插补策略——删除缺失值的观测值、分配数据集的平均值或中位数、使用前向或后向填充，或使用相关特征的组均值——对于许多预测分析项目来说都是可行的。当缺失值与目标不相关时，它们效果最好。当这一点成立时，插补值允许我们保留这些观测值的其他信息，而不会对我们的估计产生偏差。

然而，有时情况并非如此，需要更复杂的插补策略。接下来的几节将探讨清理缺失数据的多变量技术。

# 使用回归插补值

我们在上一个部分结束时，将组均值分配给缺失值，而不是整体样本均值。正如我们讨论的那样，当确定组的特征与具有缺失值的特征相关时，这很有用。使用回归来插补值在概念上与此相似，但我们通常在插补将基于两个或更多特征时使用它。

回归插补用相关特征的回归模型预测的值来替换一个特征的缺失值。这种特定类型的插补被称为确定性回归插补，因为插补值都位于回归线上，没有引入错误或随机性。

这种方法的潜在缺点是它可能会显著降低具有缺失值的特征的方差。我们可以使用随机回归插补来解决这个问题。在本节中，我们将探讨这两种方法。

NLS 数据集中的`wageincome`特征有几个缺失值。我们可以使用线性回归来插补值。工资收入值是 2016 年报告的 earnings：

1.  让我们从再次加载 NLS 数据开始，并检查 `wageincome` 以及可能与 `wageincome` 相关的特征是否存在缺失值。我们还会加载 `statsmodels` 库。

`info` 方法告诉我们，对于近 3,000 个观测值，我们缺少 `wageincome` 的值。其他特征的缺失值较少：

```py
import pandas as pd
import numpy as np
import statsmodels.api as sm
nls97 = pd.read_csv("data/nls97b.csv")
nls97.set_index("personid", inplace=True)
nls97[['wageincome','highestdegree','weeksworked16',
  'parentincome']].info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 8984 entries, 100061 to 999963
Data columns (total 4 columns):
 #  Column               Non-Null Count       Dtype
--  -------               --------------      -----  
 0wageincome            5091 non-null       float64
 1  highestdegree         8953 non-null       object 
 2  weeksworked16         7068 non-null       float64
 3  parentincome8984 non-null       int64
dtypes: float64(2), int64(1), object(1)
memory usage: 350.9+ KB
```

1.  让我们将 `highestdegree` 特征转换为数值。这将使我们在本节剩余部分进行的分析更容易：

    ```py
    nls97['hdegnum'] =
      nls97.highestdegree.str[0:1].astype('float')
    nls97.groupby(['highestdegree','hdegnum']).size() 
    highestdegree    hdegnum
    0\. None                0            953
    1\. GED                 1           1146
    2\. High School         2           3667
    3\. Associates          3            737
    4\. Bachelors           4           1673
    5\. Masters             5            603
    6\. PhD                 6             54
    7\. Professional        7            120
    ```

1.  正如我们已经发现的，我们需要将 `parentincome` 的逻辑缺失值替换为实际缺失值。之后，我们可以进行一些相关性分析。每个特征都与 `wageincome` 有关联，特别是 `hdegnum`：

    ```py
    nls97.parentincome.replace(list(range(-5,0)), np.nan,
      inplace=True)
    nls97[['wageincome','hdegnum','weeksworked16', 
      'parentincome']].corr()
                wageincome  hdegnum  weeksworked16  parentincome
    wageincome     1.00      0.40         0.18        0.27
    hdegnum        0.40      1.00         0.24        0.33
    weeksworked16  0.18      0.24         1.00        0.10
    parentincome   0.27      0.33         0.10        1.00
    ```

1.  我们应该检查对于工资收入存在缺失值的观测值是否在某些重要方面与那些没有缺失值的观测值不同。以下代码显示，这些观测值的学位获得水平、父母收入和工作周数显著较低。这是一个整体均值分配不是最佳选择的情况：

    ```py
    nls97['missingwageincome'] =
      np.where(nls97.wageincome.isnull(),1,0)
    nls97.groupby(['missingwageincome'])[['hdegnum', 
      'parentincome', 'weeksworked16']].agg(['mean', 
      'count'])
                     hdegnum    parentincome weeksworked16
                     mean count mean   count  mean   count
    missingwageincome                                    
    0                2.76 5072  48,409.13 3803 48.21  5052
    1                1.95 3881  43,565.87 2785 16.36  2016
    ```

1.  让我们尝试回归插补。首先，让我们进一步清理数据。我们可以用平均值替换缺失的 `weeksworked16` 和 `parentincome` 值。我们还应该将 `hdegnum` 合并到那些获得低于大学学位、拥有大学学位和拥有研究生学位的人群中。我们可以将它们设置为虚拟变量，当它们为 `False` 或 `True` 时分别具有 0 或 1 的值。这是在回归分析中处理分类数据的一个经过验证的方法，因为它允许我们根据组别估计不同的 *y* 截距：

    ```py
    nls97.weeksworked16.fillna(nls97.weeksworked16.mean(),
      inplace=True)
    nls97.parentincome.fillna(nls97.parentincome.mean(),
      inplace=True)
    nls97['degltcol'] = np.where(nls97.hdegnum<=2,1,0)
    nls97['degcol'] = np.where(nls97.hdegnum.between(3,4),
      1,0)
    nls97['degadv'] = np.where(nls97.hdegnum>4,1,0)
    ```

    注意

    scikit-learn 具有预处理功能，可以帮助我们完成这类任务。我们将在下一章中介绍其中的一些。

1.  接下来，我们定义一个函数 `getlm`，用于使用 `statsmodels` 模块运行线性模型。此函数具有目标或因变量名称的参数 `ycolname` 和特征或自变量名称的参数 `xcolnames`。大部分工作是由 `statsmodels` 的 `fit` 方法完成的；即 `OLS(y, X).fit()`：

    ```py
    def getlm(df, ycolname, xcolnames):
      df = df[[ycolname] + xcolnames].dropna()
      y = df[ycolname]
      X = df[xcolnames]
      X = sm.add_constant(X)
      lm = sm.OLS(y, X).fit()
      coefficients = pd.DataFrame(zip(['constant'] +
        xcolnames,lm.params, lm.pvalues), columns = [
        'features' , 'params','pvalues'])
      return coefficients, lm
    ```

1.  现在，我们可以使用 `getlm` 函数来获取参数估计和模型摘要。所有系数在 95% 的置信水平上都是正值且显著的，因为它们的 `pvalues` 小于 0.05。正如预期的那样，工资收入随着工作周数和父母收入的增加而增加。拥有大学学位与没有大学学位相比，可以增加近 16K 的收入。研究生学位甚至能将收入预测提升更多——比那些低于大学学位的人多近 37K：

    ```py
    xvars = ['weeksworked16', 'parentincome', 'degcol', 
      'degadv']
    coefficients, lm = getlm(nls97, 'wageincome', xvars)
    coefficients
         features           params           pvalues
    0    constant           7,389.37         0.00
    1    weeksworked16      494.07           0.00
    2    parentincome       0.18             0.00
    3    degcol             15,770.07        0.00
    4    degadv             36,737.84        0.00
    ```

1.  我们可以使用这个模型来插补工资收入缺失处的值。由于我们的模型包含一个常数，我们需要为预测添加一个常数。我们可以将预测转换为 DataFrame，然后将其与 NLS 数据的其余部分合并。然后，我们可以创建一个新的工资收入特征，`wageincomeimp`，当工资收入缺失时获取预测值，否则获取原始工资收入值。让我们也看看一些预测，看看它们是否有意义：

    ```py
    pred = lm.predict(sm.add_constant(nls97[xvars])).
      to_frame().rename(columns= {0: 'pred'})
    nls97 = nls97.join(pred)
    nls97['wageincomeimp'] = 
      np.where(nls97.wageincome.isnull(),
      nls97.pred, nls97.wageincome)
    pd.options.display.float_format = '{:,.0f}'.format
    nls97[['wageincomeimp','wageincome'] + xvars].head(10)
    wageincomeimp  wageincome  weeksworked16  parentincome  degcol  degadv
    personid                                          
    100061     12,500     12,500    48     7,400    0    0
    100139    120,000    120,000    53    57,000    0    0
    100284     58,000     58,000    47    50,000    0    0
    100292     36,547        NaN     4    62,760    1    0
    100583     30,000     30,000    53    18,500    0    0
    100833     39,000     39,000    45    37,000    0    0
    100931     56,000     56,000    53    60,200    1    0
    101089     36,000     36,000    53    32,307    0    0
    101122     35,151        NaN    39    46,362    0    0
    101132          0          0    22     2,470    0    0
    ```

1.  我们应该查看我们预测的一些汇总统计信息，并将其与实际工资收入值进行比较。插补的工资收入特征的均值低于原始工资收入均值。这并不奇怪，因为我们已经看到，工资收入缺失的个体在正相关特征上的值较低。令人惊讶的是标准差的急剧下降。这是确定性回归插补的一个缺点：

    ```py
    nls97[['wageincomeimp','wageincome']].
      agg(['count','mean','std'])
           wageincomeimp  wageincome
    count        8,984        5,091
    mean        42,559       49,477
    std         33,406       40,678
    ```

1.  随机回归插补基于我们模型的残差向预测中添加一个正态分布的错误。我们希望这个错误具有 0 的均值和与残差相同的方差。我们可以使用 NumPy 的正常函数来实现这一点，`np.random.normal(0, lm.resid.std(), nls97.shape[0])`。`lm.resid.std()`参数获取我们模型残差的方差。最后一个参数值，`nls97.shape[0]`，表示要创建多少个值；在这种情况下，我们希望为数据中的每一行创建一个值。

我们可以将这些值与我们的数据合并，然后向我们的预测中添加错误，`randomadd`：

```py
randomadd = np.random.normal(0, lm.resid.std(),
  nls97.shape[0])
randomadddf = pd.DataFrame(randomadd, 
  columns=['randomadd'], index=nls97.index)
nls97 = nls97.join(randomadddf)
nls97['stochasticpred'] = nls97.pred + nls97.randomadd
nls97['wageincomeimpstoc'] =
  np.where(nls97.wageincome.isnull(),
  nls97.stochasticpred, nls97.wageincome)
```

1.  这应该会增加方差，但不会对均值产生太大影响。让我们来确认这一点：

    ```py
    nls97[['wageincomeimpstoc','wageincome']].agg([
      'count','mean','std'])

           wageincomeimpstoc  wageincome
    count        8,984           5,091
    mean        42,517          49,477
    std         41,381          40,678
    ```

这似乎已经起作用了。我们的随机预测与原始工资收入特征的方差几乎相同。

回归插补是利用我们拥有的所有数据为特征插补值的好方法。它通常优于我们在上一节中检查的插补方法，尤其是在缺失值不是随机的情况下。如果我们使用随机回归插补，我们不会人为地降低我们的方差。

在我们开始使用机器学习进行这项工作之前，这是我们用于插补的多变量方法的首选。现在我们有选择使用 KNN 等算法进行这项任务，在某些情况下，这种方法比回归插补具有优势。与回归插补不同，KNN 插补不假设特征之间存在线性关系，或者那些特征是正态分布的。我们将在下一节中探讨 KNN 插补。

# 使用 KNN 插补

KNN 是一种流行的机器学习技术，因为它直观、易于运行，并且在特征和观测数不是很多时能产生良好的结果。出于同样的原因，它通常用于插补缺失值。正如其名称所暗示的，KNN 识别出与每个观测值特征最相似的 k 个观测值。当它用于插补缺失值时，KNN 使用最近邻来确定要使用的填充值。

我们可以使用 KNN 插补来完成与上一节回归插补相同的插补：

1.  让我们从导入 scikit-learn 的`KNNImputer`并再次加载 NLS 数据开始：

    ```py
    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer
    nls97 = pd.read_csv("data/nls97b.csv")
    nls97.set_index("personid", inplace=True)
    ```

1.  接下来，我们必须准备特征。我们将学位获得合并为三个类别 – 大专以下、大学和大学后学位 – 每个类别由不同的虚拟变量表示。我们还必须将父母收入的逻辑缺失值转换为实际缺失值：

    ```py
    nls97['hdegnum'] =
      nls97.highestdegree.str[0:1].astype('float')
    nls97['degltcol'] = np.where(nls97.hdegnum<=2,1,0)
    nls97['degcol'] = 
      np.where(nls97.hdegnum.between(3,4),1,0)
    nls97['degadv'] = np.where(nls97.hdegnum>4,1,0)
    nls97.parentincome.replace(list(range(-5,0)), np.nan, 
      inplace=True)
    ```

1.  让我们创建一个只包含工资收入和一些相关特征的 DataFrame：

    ```py
    wagedatalist = ['wageincome','weeksworked16', 
      'parentincome','degltcol','degcol','degadv']
    wagedata = nls97[wagedatalist]
    ```

1.  现在，我们已准备好使用 KNN 插补器的`fit_transform`方法来获取传递的 DataFrame `wagedata`中所有缺失值的值。`fit_transform`返回一个包含`wagedata`中所有非缺失值以及插补值的 NumPy 数组。我们可以使用与`wagedata`相同的索引将此数组转换为 DataFrame。这将使在下一步中合并数据变得容易。

    注意

    我们将在整本书中使用这种技术，当我们使用 scikit-learn 的`transform`和`fit_transform`方法处理 NumPy 数组时。

我们需要指定用于最近邻数量的值，即 k。我们使用一个经验法则来确定 k – 将观测数的平方根除以 2 (*sqrt(N)/2*)。在这种情况下，k 为 47：

```py
impKNN = KNNImputer(n_neighbors=47)
newvalues = impKNN.fit_transform(wagedata)
wagedatalistimp = ['wageincomeimp','weeksworked16imp',
  'parentincomeimp','degltcol','degcol','degadv']
wagedataimp = pd.DataFrame(newvalues,
  columns=wagedatalistimp, index=wagedata.index)
```

1.  现在，我们必须将插补的工资收入和工作周数列与原始 NLS 工资数据合并，并做出一些观察。请注意，使用 KNN 插补时，我们不需要对相关特征的缺失值进行任何预插补（使用回归插补时，我们将工作周数和父母收入设置为数据集的平均值）。但这确实意味着，即使没有很多信息，KNN 插补也会返回一个插补值，例如以下代码块中的`personid`的`101122`：

    ```py
    wagedata = wagedata.join(wagedataimp[['wageincomeimp', 
      'weeksworked16imp']])
    wagedata[['wageincome','weeksworked16','parentincome',
      'degcol','degadv','wageincomeimp']].head(10)
    wageincome  weeksworked16  parentincome degcol  degadv wageincomeimp
    personid         
    100061     12,500    48     7,400    0    0     12,500
    100139    120,000    53    57,000    0    0    120,000
    100284     58,000    47    50,000    0    0     58,000
    100292        NaN     4    62,760    1    0     28,029
    100583     30,000    53    18,500    0    0     30,000
    100833     39,000    45    37,000    0    0     39,000
    100931     56,000    53    60,200    1    0     56,000
    101089     36,000    53    32,307    0    0     36,000
    101122        NaN   NaN       NaN    0    0     33,977
    101132          0     22    2,470    0    0          0
    ```

1.  让我们来看看原始特征和插补特征的汇总统计。不出所料，插补工资收入的平均值低于原始平均值。正如我们在上一节中发现的，缺失工资收入的观测值在学位获得、工作周数和父母收入方面较低。我们还在工资收入中失去了一些方差：

    ```py
    wagedata[['wageincome','wageincomeimp']].agg(['count',
      'mean','std'])
    wageincome  wageincomeimp 
    count          5,091        8,984
    mean          49,477        44,781
    std           40,678        32,034
    ```

KNN 填充时不假设底层数据的分布。在回归填充中，线性回归的标准假设适用——也就是说，特征之间存在线性关系，并且它们是正态分布的。如果情况不是这样，KNN 可能是更好的填充方法。

尽管有这些优点，KNN 填充确实存在局限性。首先，我们必须根据对 k 的一个良好初始假设来调整模型，有时这个假设仅基于我们对数据集大小的了解。KNN 计算成本高，可能不适合非常大的数据集。最后，当要填充的特征与预测特征之间的相关性较弱时，KNN 填充可能表现不佳。作为 KNN 填充的替代方案，随机森林填充可以帮助我们避免 KNN 和回归填充的缺点。我们将在下一节中探讨随机森林填充。

# 使用随机森林进行填充

随机森林是一种集成学习方法。它使用自助聚合，也称为 bagging，来提高模型精度。它通过重复多次取多棵树的平均值来进行预测，从而得到越来越好的估计。在本节中，我们将使用 `MissForest` 算法，这是随机森林算法的一个应用，用于寻找缺失值填充。

`MissForest` 首先填充缺失值的均值或众数（对于连续或分类特征分别适用），然后使用随机森林来预测值。使用这个转换后的数据集，将缺失值替换为初始预测，`MissForest` 生成新的预测，可能用更好的预测来替换初始预测。`MissForest` 通常会经历至少四次这样的迭代过程。

运行 `MissForest` 甚至比使用我们在上一节中使用的 KNN 填充器还要简单。我们将为之前处理过的相同工资收入数据填充值：

1.  让我们先导入 `MissForest` 模块并加载 NLS 数据：

    ```py
    import pandas as pd
    import numpy as np
    import sys
    import sklearn.neighbors._base
    sys.modules['sklearn.neighbors.base'] =
      sklearn.neighbors._base
    from missingpy import MissForest
    nls97 = pd.read_csv("data/nls97b.csv")
    nls97.set_index("personid", inplace=True)
    ```

    注意

    我们需要解决 `sklearn.neighbors._base` 名称冲突的问题，它可以是 `sklearn.neighbors._base` 或 `sklearn.neighbors.base`，具体取决于您使用的 scikit-learn 版本。在撰写本文时，`MissForest` 使用的是旧名称。

1.  让我们进行与上一节相同的数据清理：

    ```py
    nls97['hdegnum'] = 
      nls97.highestdegree.str[0:1].astype('float')
    nls97.parentincome.replace(list(range(-5,0)), np.nan,
      inplace=True)
    nls97['degltcol'] = np.where(nls97.hdegnum<=2,1,0)
    nls97['degcol'] = np.where(nls97.hdegnum.between(3,4), 
      1,0)
    nls97['degadv'] = np.where(nls97.hdegnum>4,1,0)
    wagedatalist = ['wageincome','weeksworked16',
      'parentincome','degltcol','degcol','degadv']
    wagedata = nls97[wagedatalist]
    ```

1.  现在，我们已经准备好运行 `MissForest`。请注意，这个过程与我们使用 KNN 填充器的过程非常相似：

    ```py
    imputer = MissForest()
    newvalues = imputer.fit_transform(wagedata)
    wagedatalistimp = ['wageincomeimp','weeksworked16imp', 
      'parentincomeimp','degltcol','degcol','degadv']
    wagedataimp = pd.DataFrame(newvalues, 
      columns=wagedatalistimp , index=wagedata.index)
    Iteration: 0
    Iteration: 1
    Iteration: 2
    Iteration: 3
    ```

1.  让我们查看一些填充值和一些汇总统计信息。填充值的均值较低，这是预料之中的，因为我们已经了解到缺失值不是随机分布的，拥有较低学历和较少工作周数的人更有可能缺少工资收入的数据：

    ```py
    wagedata = wagedata.join(wagedataimp[['wageincomeimp', 
      'weeksworked16imp']])
    wagedata[['wageincome','weeksworked16','parentincome',
      'degcol','degadv','wageincomeimp']].head(10)
         wageincome  weeksworked16  parentincome  degcol  degadv  wageincomeimp
    personid                                         
    100061     12,500    48     7,400    0    0     12,500
    100139    120,000    53    57,000    0    0    120,000
    100284     58,000    47    50,000    0    0     58,000
    100292        NaN     4    62,760    1    0     42,065
    100583     30,000    53    18,500    0    0     30,000
    100833     39,000    45    37,000    0    0     39,000
    100931     56,000    53    60,200    1    0     56,000
    101089     36,000     5    32,307    0    0     36,000
    101122        NaN   NaN       NaN    0    0     32,384
    101132          0    22     2,470    0    0          0
    wagedata[['wageincome','wageincomeimp', 
      'weeksworked16','weeksworked16imp']].agg(['count', 
      'mean','std'])
        wageincome  wageincomeimp  weeksworked16  weeksworked16imp
    count    5,091        8,984        7,068         8,984
    mean    49,477       43,140           39            37
    std     40,678       34,725           21            21
    ```

`MissForest`使用随机森林算法生成高度准确的预测。与 KNN 不同，它不需要用 k 的初始值进行调优。它也比 KNN 计算成本更低。也许最重要的是，随机森林插补对特征之间低或非常高的相关性不太敏感，尽管在本例中这不是一个问题。

# 摘要

在本章中，我们探讨了缺失值插补最流行的方法，并讨论了每种方法的优缺点。分配一个总体样本均值通常不是一个好的方法，尤其是在缺失值的观测与其他观测在重要方面不同时。我们还可以显著减少我们的方差。前向填充或后向填充允许我们保持数据中的方差，但它在观测的邻近性有意义时效果最好，例如时间序列或纵向数据。在大多数非平凡情况下，我们将希望使用多元技术，例如回归、KNN 或随机森林插补。

到目前为止，我们还没有涉及到数据泄露的重要问题以及如何创建独立的训练和测试数据集。为了避免数据泄露，我们需要在开始特征工程时独立于测试数据工作训练数据。我们将在下一章更详细地研究特征工程。在那里，我们将编码、转换和缩放特征，同时也要小心地将训练数据和测试数据分开。
