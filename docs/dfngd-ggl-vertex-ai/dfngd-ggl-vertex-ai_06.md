

# 第六章：构建 ML 模型的低代码选项

**BigQuery 机器学习**，通常简称为 **BQML**，是 Google Cloud 提供的工具之一，它无缝地将数据仓库和 ML 世界融合在一起。设计用于弥合数据分析师和 ML 模型之间的差距，BQML 使个人能够在 BigQuery 的范围内直接构建、评估和预测 ML 模型，无需移动数据或掌握新的工具集。

这种集成不仅简化了模型创建的过程，还为熟悉 SQL 的人提供了一个直观的过渡。只需几个语句，您就可以从数据分析过渡到预测洞察。

在本章中，我们将讨论以下主题：

+   什么是 BQML？

+   使用 BQML 进行特征转换

+   使用 BQML 构建 ML 模型

+   使用 BQML 进行推理

# 什么是 BQML？

BQML 是 Google Cloud 提供的强大内置 ML 服务，允许用户使用熟悉的 SQL 查询创建、训练和部署 ML 模型。BQML 设计用于简化那些可能没有强大数据科学或编程背景的人构建和部署 ML 模型的过程。在本章中，我们将探讨 BQML 的关键特性和功能，以及您如何使用它来利用 Google Cloud AI 的力量为您的项目服务。

BQML 提供了一种无缝的方式将机器学习（ML）集成到您的数据分析工作流程中，无需深入了解 ML 概念或编程语言。使用 BQML，您可以做到以下几点：

+   使用 SQL 查询创建和训练 ML 模型

+   使用训练模型进行预测

+   评估您模型的性能

+   执行特征转换和超参数调整

+   理解模型解释和权重

+   导出和导入模型

利用 BQML 提供了许多优势：

+   BQML 消除了将数据加载到本地内存中的需求，从而解决了大型数据集带来的约束

+   BQML 通过处理标准任务，如将数据分为训练集和测试集、选择和调整学习率以及选择优化方法来简化 ML 流程

+   BQML 模型的自动版本控制使得跟踪变更并在需要时回滚到早期版本变得轻而易举

+   在提供预测时，BQML 模型可以无缝集成到 Vertex AI

使用 BQML 也有一些限制：

+   **模型类型有限**：BQML 支持一组受限的 ML 模型，如线性回归、逻辑回归、k-means 聚类、矩阵分解等。它可能无法满足需要高级或专用模型的项目需求。

+   **可定制性**：BQML 对 ML 的自动化方法意味着可定制的范围有限。用户可能无法像在其他 ML 框架中那样微调模型或尝试不同的模型架构。

+   **可扩展性**：尽管 BQML 设计用于处理大型数据集，但在处理极大数据集或复杂模型时，它可能不如其他分布式机器学习框架那样有效地扩展。

+   `ML.PREDICT` 函数根据图像进行预测。BQML 现在也支持将 **远程模型** 作为 API 端点添加，这为添加托管在 Vertex AI 端点上的任何模型或添加其他基于云的机器学习服务（如 Vision API）以支持更多用例打开了可能性。

+   **特征工程**：BQML 可能不是进行广泛特征工程的最佳选择，因为它更侧重于简化机器学习过程。用户可能需要在 BQML 之外执行特征工程以进行高级特征工程任务。我们将在本章的 *特征工程* 部分更详细地讨论这些限制。

+   **外部数据源**：BQML 主要与 Google BigQuery 数据一起工作，这限制了其在数据源方面的灵活性。如果您想使用来自不同来源或格式的数据，您可能需要首先将其导入到 BigQuery 中。

+   **模型可移植性**：BQML 模型与 Google Cloud 紧密集成。将模型导出以在 Google 生态系统之外使用可能具有挑战性，可能需要额外的工作。

现在，让我们看看您如何开始使用 BigQuery 解决机器学习问题。

# 开始使用 BigQuery

Google BigQuery 是一个无服务器、完全管理的数据仓库，它利用 Google 基础设施的强大处理能力进行超快的 SQL 查询。由于 BigQuery 不是 Vertex AI 的一部分，我们不会在本书中深入探讨该工具的功能，但这里有一个快速指南，介绍如何开始使用 BigQuery：这将提供足够的信息，帮助您跟随本章后面的练习。 

1.  **设置 Google Cloud 项目**：在您可以使用 BigQuery 之前，您需要设置一个 **Google Cloud Platform**（**GCP**）项目。前往 Google Cloud 控制台并创建一个新的项目。如果您之前从未使用过 GCP，您可能需要创建一个账户并设置账单信息。

1.  **启用 BigQuery API**：在您的 GCP 项目中，导航到 **API & Services** 部分，并启用 BigQuery API。

1.  **访问 BigQuery 控制台**：一旦启用 API，您可以通过 GCP 仪表板或直接通过 BigQuery 控制台链接（[`console.cloud.google.com/bigquery`](https://console.cloud.google.com/bigquery)）访问 BigQuery 控制台。

1.  **创建数据集**：数据集是 BigQuery 中表、视图和其他数据对象的容器。要创建一个数据集，请点击 BigQuery 控制台中您 GCP 项目名称旁边的垂直省略号，选择 **创建数据集**，并填写数据集的名称。然后，点击 **创建数据集**。

1.  **加载数据**：BigQuery 支持多种数据格式，包括 CSV、JSON 等。你可以从 Google Cloud Storage 将数据加载到 BigQuery 中，通过 API 请求直接发送数据，或手动上传文件。要加载数据，请导航到左侧的 BigQuery 控制台中的数据集，点击 **创建表**，然后按照提示操作。

1.  **运行 SQL 查询**：将数据加载到 BigQuery 后，你可以运行 SQL 查询。使用 BigQuery 控制台中的查询编辑器开始使用 SQL 分析你的数据。

现在，让我们看看如何使用 BigQuery 的原生函数进行大规模特征/数据转换，为机器学习模型准备训练数据。

# 使用 BQML 进行特征转换

BQML 支持两种类型的特征预处理：

+   **自动预处理**：在训练过程中，BQML 会执行自动预处理。有关更详细的信息，请执行自动预处理，如缺失数据插补、独热编码和时间戳转换与编码。

+   BQML 提供的 `TRANSFORM` 子句用于定义使用手动预处理函数的定制预处理。这些函数也可以在 `TRANSFORM` 子句之外使用。

虽然 BQML 支持一些特征工程任务，但与更灵活且功能丰富的机器学习框架相比，它存在某些限制：

+   **有限的预处理函数**：BQML 提供了一组基本的 SQL 函数用于数据预处理，例如缩放和编码。然而，它可能缺乏其他机器学习库（如 **scikit-learn** 或 **TensorFlow**）中可用的某些高级预处理技术或专用函数。

+   **无自动特征选择**：BQML 不提供自动特征选择方法来识别数据集中最重要的变量。你必须根据你的领域知识和直觉手动选择和构建特征，或者使用外部工具进行特征选择。

+   **复杂特征转换**：基于 SQL 的 BQML 方法可能不适合某些涉及非线性组合、滚动窗口或数据中的序列模式等复杂特征转换。在这种情况下，你可能需要在使用 BQML 之前使用其他工具或编程语言对数据进行预处理。

+   **自定义特征生成**：BQML 缺乏创建自定义特征的灵活性，例如像更通用的机器学习库那样轻松创建特定领域的函数或转换。你可能需要在外部实现这些自定义特征，这可能会很繁琐且效率较低。

+   **特征工程管道**：BQML 不提供内置机制来创建和管理可重用的特征工程管道。相比之下，其他机器学习框架提供了构建模块化和可维护管道的功能，简化了将相同的转换应用于训练和验证数据集或在模型部署期间的过程。

虽然 BQML 简化了机器学习过程，但它可能不是需要大量或高级特征工程的项目最佳选择。在这种情况下，您可能需要使用外部工具或库预处理您的数据，然后将转换后的数据导入 BigQuery，以便使用 BQML 进行进一步分析。

## 手动预处理

BQML 提供了一系列手动预处理函数，可以使用 `CREATE MODEL` 语法在训练前预处理您的数据。这些函数也可以在 `TRANSFORM` 子句之外使用。这些预处理函数可以是标量，对单行进行操作，或者分析型，对所有行进行操作，并基于所有行收集的统计信息输出结果。当在训练期间 `TRANSFORM` 子句中使用 ML 分析函数时，相同的统计信息会自动应用于预测输入。

以下表格列出了 BigQuery 中所有支持的数据预处理函数：

| **函数名称** | **描述** |
| --- | --- |
| `ML.BUCKETIZE` | 根据提供的分割点将数值表达式分桶到用户定义的类别。 |
| `ML.POLYNOMIAL_EXPAND` | 生成给定数值特征的指定度数的多项式组合。 |
| `ML.FEATURE_CROSS` | 生成指定度数的分类特征的交叉特征。 |
| `ML.NGRAMS` | 根据给定的 *n* 值范围从标记数组中提取 n-gram。 |
| `ML.QUANTILE_BUCKETIZE` | 根据几个桶将数值表达式分桶到基于分位数的类别。 |
| `ML.HASH_BUCKETIZE` | 将字符串表达式分桶到基于哈希的固定数量桶。 |
| `ML.MIN_MAX_SCALER` | 将数值表达式缩放到范围 [0, 1]，并在所有行中使用 `MIN` 和 `MAX` 进行限制。 |
| `ML.STANDARD_SCALER` | 标准化数值表达式。 |
| `ML.MAX_ABS_SCALER` | 通过除以最大绝对值将数值表达式缩放到范围 [-1, 1]。 |
| `ML.ROBUST_SCALER` | 使用对异常值具有鲁棒性的统计方法缩放数值表达式。 |
| `ML.NORMALIZER` | 使用给定的 p-范数将数组表达式归一化到单位范数。 |
| `ML.IMPUTER` | 使用指定的值（例如，平均值、中位数或最频繁值）替换表达式中的 `NULL`。 |
| `ML.ONE_HOT_ENCODER` | 使用单热编码方案编码字符串表达式。 |
| `ML.LABEL_ENCODER` | 将字符串表达式编码为 `[0, n_categories]` 范围内的 `INT64`。 |

表 6.1 – 数据转换函数

这里列出了所有函数及其输入和输出，以及每个函数的示例：

+   `ML.BUCKETIZE`

    根据提供的分割点将数值表达式分桶到用户定义的类别。

    输入：

    +   `numerical_expression`: 用于分桶的数值表达式。

    +   `array_split_points`: 包含分割点的排序数值数组。

    +   `exclude_boundaries`（可选）：如果为 `TRUE`，则从 `array_split_points` 中移除两个边界。默认值为 `FALSE`。

    输出：

    `STRING` 是 `numerical_expression` 字段分割成桶的名称。

    这里是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT 21 AS age UNION ALL
    SELECT 33 AS age UNION ALL
    SELECT 45 AS age UNION ALL SELECT 59 AS age UNION ALL
    SELECT 66 AS age )
    SELECT age, ML.BUCKETIZE(age, [18, 35, 50, 65]) AS age_bucket
    FROM dataset;
    ```

    在 BigQuery 中提交此查询应生成以下输出：

    ```py
    age  age_bucket
    21  bin_2
    33  bin_2
    45  bin_3
    59  bin_4
    bin_5
    ```

+   `ML.FEATURE_CROSS`

    这可以生成指定度数的分类特征的交叉特征：

    ```py
    Input: array_expression, degree
    Output: ARRAY<STRUCT<name STRING, value FLOAT64>>
    ```

    这里是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT "dog" AS animal, "brown" AS color UNION ALL
    SELECT "cat" AS animal, "black" AS color UNION ALL
    SELECT "bird" AS animal, "yellow" AS color UNION ALL
    SELECT "fish" AS animal, "orange" AS color)
    SELECT animal, color, ML.FEATURE_CROSS(STRUCT(animal, color)) AS animal_color
    FROM dataset;
    ```

    在 BigQuery 中提交此查询应生成以下输出：

| **动物** | **颜色** | **动物 _ 颜色** |
| --- | --- | --- |
| 狗 | 棕色 | 狗 _ 棕色 |
| 猫 | 黑色 | 猫 _ 黑色 |
| 鸟 | 黄色 | 鸟 _ 黄色 |
| 鱼 | 橙色 | 鱼 _ 橙色 |

表 6.2 – BigQuery 查询输出

注意，如果 `ARRAY` 参数中包含更多列，则可以使用 `ML.FEATURE_CROSS` 函数创建多个列的交叉。

+   `ML.NGRAMS`

    这根据给定的 *n* 值范围从标记数组中提取 n-gram。

    输入：

    +   `array_input`: `ARRAY` 的 `STRING`。这些字符串是要合并的标记。

    +   `range`: `ARRAY` 的两个 `INT64` 元素或单个 `INT64`。`ARRAY` 输入中的这两个排序的 `INT64` 元素是要返回的 n-gram 大小范围。单个 `INT64` 等同于 [x, x] 范围。

    +   `separator`: 可选的 `STRING`。`separator` 连接输出中的相邻标记。默认值是空格。

    输出：`ARRAY` 的 `STRING`。

    这里是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT "apple" AS fruit, "cherry" AS fruit2,"pear" AS fruit3 UNION ALL
    SELECT "banana" AS fruit, "banana" AS fruit2,"melon" AS fruit3 UNION ALL
    SELECT "cherry" AS fruit, "cherry" AS fruit2, "pineapple" AS fruit3)
    SELECT fruit,fruit2,fruit3, ML.NGRAMS([fruit,fruit2,fruit3], [2]) AS fruit_ngrams
    FROM dataset;
    ```

    在 BigQuery 中提交此查询应生成以下输出：

| **水果** | **水果 2** | **水果 3** | **水果 _ngrams** |
| --- | --- | --- | --- |
| 苹果 | 樱桃 | 梨 | [苹果 樱桃, 樱桃 梨] |
| 香蕉 | 香蕉 | 西瓜 | [香蕉 香蕉, 香蕉 西瓜] |
| 樱桃 | 樱桃 | 菠萝 | [樱桃 樱桃, 樱桃 菠萝] |

表 6.3 – BigQuery 输出

+   `ML.QUANTILE_BUCKETIZE`

    这个功能将数值表达式根据几个桶划分为基于分位数的类别。

    输入：

    +   `numerical_expression`: 要划分的数值表达式

    +   `num_buckets`: `INT64`。将 `numerical_expression` 分割成桶的数量

    输出：`STRING`。

    这里是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT 21 AS age UNION ALL
    SELECT 33 AS age UNION ALL
    SELECT 45 AS age UNION ALL
    SELECT 59 AS age UNION ALL
    SELECT 66 AS age)
    SELECT age, ML.QUANTILE_BUCKETIZE(age, 4) OVER() AS age_bucket
    FROM dataset
    ORDER BY age;
    ```

    在此示例中，我们创建了一个包含五行年龄数据的虚拟表数据集。然后，我们使用 `ML.QUANTILE_BUCKETIZE` 函数将 `age` 列划分为四个分位数桶。结果 `age_bucket` 列显示了数据集的每一行属于哪个分位数桶。

    这里是输出示例：

    ```py
    age  age_bucket
    21  bin_1
    33  bin_2
    45  bin_3
    59  bin_4
    66  bin_4
    ```

+   `ML.HASH_BUCKETIZE`

    这将字符串表达式划分为基于哈希的固定数量的桶。

    输入：

    +   `string_expression`: `STRING`。要划分的字符串表达式。

    +   `hash_bucket_size`: `INT64`。桶的数量。预期 `hash_bucket_size >= 0`。如果 `hash_bucket_size = 0`，则函数仅对字符串进行哈希处理，而不对哈希值进行桶划分。

    输出：`INT64`。

    这里是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT "horse" AS animal UNION ALL
    SELECT "cat" AS animal UNION ALL
    SELECT "dog" AS animal UNION ALL
    SELECT "fish" AS animal)
    SELECT animal, ML.HASH_BUCKETIZE(animal, 2) AS animal_bucket FROM dataset;
    ```

    在本例中，我们创建了一个包含四行数据的虚拟表数据集。然后，我们使用`ML.HASH_BUCKETIZE`函数将`animal`列哈希到两个桶中。结果`animal_bucket`列显示了数据集的每一行属于哪个哈希桶。

    注意，可以通过指定第二个参数的不同值来使用`ML.HASH_BUCKETIZE`函数将列的值哈希到不同数量的桶中。

    下面是输出：

    ```py
    animal  animal_bucket
    horse    1
    cat    0
    dog    0
    fish  0
    ```

    在本例中，我们创建了一个包含四行数据的虚拟表数据集。然后，我们使用`ML.HASH_BUCKETIZE`函数将`animal`列哈希到两个桶中。结果`animal_bucket`列显示了数据集的每一行属于哪个哈希桶。

+   `ML.MIN_MAX_SCALER`

    将数值表达式缩放到范围[0, 1]，范围由所有行的`MIN`和`MAX`值限制。

    输入：`numerical_expression`。要缩放的数值表达式。

    输出：`DOUBLE`。

    下面是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT 10 AS age UNION ALL
    SELECT 20 AS age UNION ALL
    SELECT 30 AS age UNION ALL
    SELECT 40 AS age UNION ALL
    SELECT 50 AS age)
    SELECT age, ML.MIN_MAX_SCALER(age) Over() AS scaled_age
    FROM dataset;
    ```

    在本例中，我们创建了一个包含五行年龄数据的虚拟表数据集。然后，我们使用`ML.MIN_MAX_SCALER`函数将`age`列缩放到 0 到 1 的范围。请注意，可以通过指定不同的`MIN`和`MAX`参数值来使用`ML.MIN_MAX_SCALER`函数将列的值缩放到不同的范围。

    下面是输出：

    ```py
    age  scaled_age
    50  1.0
    20  0.25
    40  0.75
    10  0.0
    30  0.5
    ```

+   `ML.STANDARD_SCALER`

    此函数对数值表达式进行标准化。

    输入：`numerical_expression`。要缩放的数值表达式。

    输出：`DOUBLE`。

    下面是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT 10 AS age UNION ALL
    SELECT 20 AS age UNION ALL
    SELECT 30 AS age UNION ALL
    SELECT 40 AS age UNION ALL
    SELECT 50 AS age)
    SELECT age, ML.STANDARD_SCALER(age) OVER() AS scaled_age
    FROM dataset;
    ```

    在本例中，我们创建了一个包含五行年龄数据的虚拟表数据集。然后，我们使用`ML.STANDARD_SCALER`函数将`age`列标准化，使其均值为 0，标准差为 1。结果`scaled_age`列显示了`age`列的标准化值。

    下面是输出：

    ```py
    age  scaled_age
    40  0.63245553203367588
    10  -1.2649110640673518
    50  1.2649110640673518
    20  -0.63245553203367588
    30     0.0
    ```

+   `ML.MAX_ABS_SCALER`

    此函数通过除以最大绝对值来将数值表达式缩放到范围[-1, 1]。

    输入：`numerical_expression`。

    输出：`DOUBLE`。

    下面是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT -10 AS age UNION ALL
    SELECT 20 AS age UNION ALL
    SELECT -30 AS age UNION ALL
    SELECT 40 AS age UNION ALL
    SELECT -50 AS age)
    SELECT age, ML.MAX_ABS_SCALER(age)
    OVER() AS scaled_age
    FROM dataset;
    ```

    在本例中，我们创建了一个包含五行年龄数据的虚拟表数据集。然后，我们使用`ML.MAX_ABS_SCALER`函数缩放`age`列，使得该列中最大绝对值元素的绝对值缩放到 1。结果`scaled_age`列显示了`age`列的缩放值。

    下面是输出：

    ```py
    age    scaled_age
    -10    -0.2
    -50    -1.0
    20    0.4
    40    0.8
    -30    -0.6
    ```

+   `ML.NORMALIZER`

    此函数使用给定的 p-norm 将`array_expression`归一化，使其具有单位范数。

    输入：`array_expression, p`。

    输出：`ARRAY<DOUBLE>`。

    下面是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT 1 AS x, 2 AS y UNION ALL
    SELECT 3 AS x, 4 AS y UNION ALL
    SELECT 5 AS x, 6 AS y UNION ALL
    SELECT 7 AS x, 8 AS y UNION ALL
    SELECT 9 AS x, 10 AS y)
    SELECT x, y, ML.NORMALIZER([x, y], 1) AS norm_xy
    FROM dataset;
    ```

    下面是输出：

    ```py
    x  y  norm_xy
    1  2  "[0.4472135954999579,0.8944271909999158]"
    3  4  "[0.6,0.8]"
    5  6  "[0.6401843996644799,0.7682212795973758]"
    7  8  "[0.658504607868518,0.75257669470687782]"
    9  10  "[0.6689647316224497,0.7432941462471663]"
    ```

+   `ML.IMPUTER`

    此函数使用指定的值（例如，均值、中位数或最频繁值）替换表达式中的`NULL`。

    输入：`expression, strategy`。

    输出：数值表达式的`DOUBLE`。`STRING`表达式的`STRING`。

    下面是一个示例 SQL 语句：

    ```py
    WITH dataset AS (
    SELECT 10 AS age, 20 AS height UNION ALL
    SELECT 20 AS age, 30 AS height UNION ALL
    SELECT 30 AS age, 40 AS height UNION ALL
    SELECT 40 AS age, 50 AS height UNION ALL
    SELECT 50 AS age, NULL AS height)
    SELECT age, height, ML.IMPUTER(height,"median") OVER() AS imputed_height FROM dataset;
    ```

    下面是输出：

    ```py
    age  height  imputed_height
    20    30    30.0
    10    20    20.0
    40    50    50.0
    50    null    30.0
    30    40    40.0
    ```

+   `ML.ONE_HOT_ENCODER`

    此函数使用单热编码方案对`string_expression`进行编码。

    输入：

    +   `string_expression`：要编码的`STRING`表达式。

    +   `drop`（可选）：这决定了在编码过程中要丢弃哪个类别。默认值是`none`，意味着保留所有类别。

    +   `top_k`（可选）：`INT64`。这限制了编码词汇表只包含`top_k`个最频繁的类别。默认值是 32,000，最大支持值是 1 百万，以避免因高维性而受到影响。

    +   `frequency_threshold`（可选）：`INT64`。它限制了编码词汇表只包含频率`>= frequency_threshold`的类别。默认值是`5`。

    输出：它是一个包含编码值的`STRUCT`数组，其中`index`是编码值的索引，`value`是编码值的值。

    这里是一个示例 SQL 语句：

    ```py
    WITH
      input_data AS (
        SELECT 'red' AS color UNION ALL
        SELECT 'blue' AS color UNION ALL
        SELECT 'green' AS color UNION ALL
        SELECT 'green' AS color UNION ALL
        SELECT 'purple' AS color),
      vocab AS (
        SELECT color, COUNT(*) AS frequency
        FROM input_data
        GROUP BY color)
    SELECT color,
      ML.ONE_HOT_ENCODER(color) OVER()  AS encoding
    FROM input_data
    ```

    上述查询的结果如下所示：

| **颜色** | **encoding.index** | **encoding.value** |
| --- | --- | --- |
| 绿色 | 0 | 1.0 |
| 红色 | 0 | 1.0 |
| 紫色 | 0 | 1.0 |
| 蓝色 | 0 | 1.0 |
| 绿色 | 0 | 1.0 |

表 6.4：前一个查询的输出

+   `ML.LABEL_ENCODER`

    此函数将字符串值转换为指定范围内的`INT64`数字。该函数按字母顺序组织编码术语，且在此词汇表中未找到的任何类别将表示为`0`。当在`TRANSFORM`子句中使用时，训练过程中省略的词汇表和类别将无缝应用于预测。

    输入：

    +   `string_expression`：要编码的`STRING`表达式。

    +   `top_k`：可选的`INT64`。这限制了编码词汇表只包含`top_k`个最频繁的类别。默认值是 32,000，最大支持值是 1 百万，以避免因高维性而受到影响。

    +   `frequency_threshold`：可选的`INT64`。这限制了编码词汇表只包含频率`>= frequency_threshold`的类别。默认值是`5`。

    输出：`INT64`。这是指定范围内字符串表达式的编码值。

    这里是一个示例 SQL 语句：

    ```py
    WITH data AS (
      SELECT 'apple' AS fruit UNION ALL
      SELECT 'banana' UNION ALL
      SELECT 'orange' UNION ALL
      SELECT 'apple' UNION ALL
      SELECT 'pear' UNION ALL
      SELECT 'kiwi' UNION ALL
      SELECT 'banana')
    SELECT fruit, ML.LABEL_ENCODER(fruit, 2,2) OVER() AS encoded_fruit FROM data
    ```

    上述查询的结果如下所示：

| **水果** | **encoded_fruit** |
| --- | --- |
| 橙子 | 0 |
| 梨 | 0 |
| 香蕉 | 2 |
| 苹果 | 1 |
| 草莓 | 0 |
| 苹果 | 1 |
| 香蕉 | 2 |

表 6.5：前一个查询的输出

现在我们来看看你可以使用 BQML 构建的不同类型的机器学习模型。

# 使用 BQML 构建机器学习模型

BQML 支持多种不同用例的模型训练。目前支持的关键模型类别包括监督学习模型、无监督学习模型、时间序列模型、导入的模型和远程模型。

下表展示了 BigQuery 支持的一些关键机器学习模型类型：

| **模型类型** | **模型类型** | **手动定义** **特征预处理** | **在 BQML 中的超参数调整** |
| --- | --- | --- | --- |
| 监督学习 | 线性和逻辑回归 | 支持 | 支持 |
| 监督学习 | 深度神经网络 | 支持 | 支持 |
| 监督学习 | 广度与深度 | 支持 | 支持 |
| 监督 | 提升树 | 支持 | 支持 |
| 监督 | 随机森林 | 支持 | 支持 |
| 监督 | AutoML 表格 | 不支持 | 自动 |
| 无监督 | k-means | 支持 | 支持 |
| 无监督 | 矩阵分解 | 不支持 | 支持 |
| 无监督 | PCA | 支持 | 不支持 |
| 无监督 | 自动编码器 | 支持 | 不支持 |
| 时间序列 | `ARIMA_PLUS` | 仅自动预处理 | 支持`(``auto.ARIMA4)*` |
| 时间序列 | `ARIMA_PLUS_XREG` | 仅自动预处理 | 支持`(``auto.ARIMA4)*` |

表 6.6 – 支持的关键机器学习模型的功能

在 BigQuery 中，有两个其他重要的模型创建选项可供您使用，以帮助您利用 BigQuery 外部构建的机器学习模型 – 导入模型和远程模型。

BQML 允许您导入在 BigQuery 外部训练的模型，以便可以在 BigQuery 内部进行推理。

支持以下模型框架用于导入：

+   TensorFlow

+   TensorFlow Lite

+   ONNX

+   XGBoost

BQML 允许您将现有的 Vertex AI 端点注册为远程模型。一旦在 BigQuery 中注册，您就可以从 BigQuery 内部向 Vertex AI 端点发送预测请求以进行推理。

# 创建 BQML 模型

用于启动模型创建的 BigQuery 函数被称为`CREATE`。在本节中，我们将探讨当用户使用`CREATE`函数创建不同类型的 BQML 模型时可供使用的选项。您目前不一定需要阅读每个模型的详细信息。这应该更多地用作参考，按需使用。

### 线性或逻辑回归模型

下面的语法用于创建回归模型，以及您需要在查询中提供的一些必需和可选参数：

```py
{CREATE OR REPLACE MODEL} model_name
[OPTIONS(MODEL_TYPE = { 'LINEAR_REG' | 'LOGISTIC_REG' },
    INPUT_LABEL_COLS = string_array,
    OPTIMIZE_STRATEGY = { 'AUTO_STRATEGY'  },
    L1_REG = float64_value,
    L2_REG = float64_value,
    MAX_ITERATIONS = int64_value,
    LEARN_RATE_STRATEGY = { 'LINE_SEARCH' | 'CONSTANT' },
    LEARN_RATE = float64_value,
    EARLY_STOP = { TRUE },
    MIN_REL_PROGRESS = float64_value,
    DATA_SPLIT_METHOD = { 'AUTO_SPLIT'},
    DATA_SPLIT_EVAL_FRACTION = float64_value,
    DATA_SPLIT_COL = string_value,
    LS_INIT_LEARN_RATE = float64_value,
    WARM_START = { FALSE },
    AUTO_CLASS_WEIGHTS = { TRUE  },
    CLASS_WEIGHTS = struct_array,
    ENABLE_GLOBAL_EXPLAIN = { FALSE },
    CALCULATE_P_VALUES = { FALSE },
    FIT_INTERCEPT = { FALSE },
    CATEGORY_ENCODING_METHOD = { 'ONE_HOT_ENCODING`, 'DUMMY_ENCODING' })];
```

可以在`CREATE MODEL`语句中指定的关键选项如下：

+   `MODEL_TYPE`：指定所需的模型类型（例如，线性或逻辑回归）。

+   `INPUT_LABEL_COLS`：定义训练数据中的标签列名。

+   `OPTIMIZE_STRATEGY`：选择训练线性回归模型的方法：

    +   `AUTO_STRATEGY`：根据几个条件选择训练方法：

        +   如果指定了`l1_reg`或`warm_start`，则采用`batch_gradient_descent`策略

        +   `batch_gradient_descent` 如果训练特征的总体基数超过 10,000 也会被使用

        +   当可能存在过拟合问题时，特别是当训练样本数小于总基数 10 倍时，选择`batch_gradient_descent`

    +   对于所有其他场景，实现`NORMAL_EQUATION`策略

    +   `BATCH_GRADIENT_DESCENT`：在模型训练中启用批量梯度下降方法，通过使用梯度函数优化损失函数。

    +   `NORMAL_EQUATION`: 使用解析公式推导线性回归问题的最小二乘解。在以下情况下不允许使用`NORMAL_EQUATION`策略：

        +   `l1_reg`被定义

        +   `warm_start`被定义

        +   训练特征的总基数超过 10,000

+   `L1_REG`: 设置应用的 L1 正则化的量。

+   `L2_REG`: 设置应用的 L2 正则化的量。

+   `MAX_ITERATIONS`: 确定训练迭代的最大次数或步骤。

+   `LEARN_RATE_STRATEGY`: 选择在训练期间指定学习率的策略。

+   `LEARN_RATE`: 定义梯度下降的学习率。

+   `EARLY_STOP`: 指示是否在第一次迭代后，如果相对损失改进最小，则停止训练。

+   `MIN_REL_PROGRESS`: 设置继续训练的最小相对损失改进。

+   `DATA_SPLIT_METHOD`: 选择将输入数据分割为训练集和评估集的方法。这里的选项有`'AUTO_SPLIT'`、`'RANDOM'`、`'CUSTOM'`、`'SEQ'`和`'NO_SPLIT'`。

+   `DATA_SPLIT_EVAL_FRACTION`: 指定在`'RANDOM'`和`'SEQ'`分割中用于评估的数据的分数。

+   `DATA_SPLIT_COL`: 识别用于分割数据的列。

+   `LS_INIT_LEARN_RATE`: 为`'LINE_SEARCH'`策略设置初始学习率。

+   `WARM_START`: 使用新的训练数据、新的模型选项或两者重新训练模型。

+   `AUTO_CLASS_WEIGHTS`: 使用每个类别的权重来平衡类别标签，这些权重与该类别的频率成反比。

+   `CLASS_WEIGHTS`: 定义每个类别标签使用的权重。

+   `ENABLE_GLOBAL_EXPLAIN`: 使用可解释 AI 计算全局特征重要性评估的全局解释。

+   `CALCULATE_P_VALUES`: 在训练期间计算 p 值和标准误差。

+   `FIT_INTERCEPT`: 在训练期间将截距拟合到模型中。

+   `CATEGORY_ENCODING_METHOD`: 指定对非数值特征使用的编码方法。

### 创建深度神经网络模型和宽深度模型

这是创建深度学习模型的语法，以及作为查询一部分需要提供的不同必需和可选参数：

```py
 {CREATE OR REPLACE MODEL} model_name
[OPTIONS(MODEL_TYPE= {'DNN_CLASSIFIER'},
         ACTIVATION_FN = { 'RELU' },
         AUTO_CLASS_WEIGHTS = { TRUE | FALSE },
         BATCH_SIZE = int64_value,
         CLASS_WEIGHTS = struct_array,
         DROPOUT = float64_value,
         EARLY_STOP = { TRUE | FALSE },
         HIDDEN_UNITS = int_array,
         L1_REG = float64_value,
         L2_REG = float64_value,
         LEARN_RATE = float64_value,
         INPUT_LABEL_COLS = string_array,
         MAX_ITERATIONS = int64_value,
         MIN_REL_PROGRESS = float64_value,
         OPTIMIZER={'ADAGRAD'},
         WARM_START = { FALSE },
         DATA_SPLIT_METHOD={'AUTO_SPLIT'},
         DATA_SPLIT_EVAL_FRACTION = float64_value,
         DATA_SPLIT_COL = string_value,
         ENABLE_GLOBAL_EXPLAIN = { FALSE },
         INTEGRATED_GRADIENTS_NUM_STEPS = int64_value,
         TF_VERSION = { '2.8.0' })];
```

以下选项可以作为模型创建请求的一部分指定：

+   `model_name`: 你正在创建或替换的 BQML 模型名称。

+   `model_type`: 指定模型的类型，可以是`'DNN_CLASSIFIER'`或`'DNN_REGRESSOR'`。

+   `activation_fn`: 对于 DNN 模型类型，这指定了神经网络的激活函数。选项有`'RELU'`、`'RELU6'`、`'CRELU'`、`'ELU'`、`'SELU'`、`'SIGMOID'`和`'TANH'`。

+   `auto_class_weights`: 指定是否使用每个类别的权重来平衡类别标签，这些权重与该类别的频率成反比。仅与`DNN_CLASSIFIER`模型一起使用。

+   `batch_size`: 对于 DNN 模型类型，这指定了输入到神经网络的样本的迷你批大小。

+   `class_weights`: 用于每个类别标签的权重。如果`AUTO_CLASS_WEIGHTS`为`TRUE`，则不能指定此选项。

+   `data_split_method`: 将输入数据分割为训练集和评估集的方法。选项有`'AUTO_SPLIT'`、`'RANDOM'`、`'CUSTOM'`、`'SEQ'`和`'NO_SPLIT'`。

+   `data_split_eval_fraction`: 与`'RANDOM'`和`'SEQ'`分割一起使用。它指定用于评估的数据比例。

+   `data_split_col`: 识别用于分割数据的列。

+   `dropout`: 对于 DNN 模型类型，这指定了神经网络中单元的丢弃率。

+   `early_stop`: 是否在相对损失改进小于为`MIN_REL_PROGRESS`指定的值的第一次迭代后停止训练。

+   `enable_global_explain`: 指定是否使用可解释人工智能来计算全局解释，以评估全局特征对模型的重要性。

+   `hidden_units`: 对于 DNN 模型类型，这指定了神经网络的隐藏层。

+   `input_label_cols`: 训练数据中的标签列名称。

+   `integrated_gradients_num_steps`: 指定在解释示例及其基线之间采样步数的数量，以近似积分梯度属性方法中的积分。

+   `l1_reg`: 优化器的 L1 正则化强度。

+   `l2_reg`: 优化器的 L2 正则化强度。

+   `learn_rate`: 训练的初始学习率。

+   `max_iterations`: 训练迭代的最大次数。

+   `optimizer`: 对于 DNN 模型类型，这指定了训练模型的优化器。选项有`'ADAGRAD'`、`'ADAM'`、`'FTRL'`、`'RMSPROP'`和`'SGD'`。

+   `warm_start`: 是否使用新的训练数据、新的模型选项或两者重新训练模型。

+   `tf_version`: 指定模型训练的 TensorFlow 版本。

### 创建提升树和随机森林模型

创建提升树和随机森林模型的语法，以及作为查询一部分需要提供的不同必需和可选参数：

```py
{CREATE OR REPLACE MODEL} model_name
[OPTIONS(MODEL_TYPE = { 'BOOSTED_TREE_CLASSIFIER' },
         BOOSTER_TYPE = {'GBTREE' },
         NUM_PARALLEL_TREE = int64_value,
         DART_NORMALIZE_TYPE = {'TREE' },
         TREE_METHOD={'AUTO' },
         MIN_TREE_CHILD_WEIGHT = int64_value,
         COLSAMPLE_BYTREE = float64_value,
         COLSAMPLE_BYLEVEL = float64_value,
         COLSAMPLE_BYNODE = float64_value,
         MIN_SPLIT_LOSS = float64_value,
         MAX_TREE_DEPTH = int64_value,
         SUBSAMPLE = float64_value,
         AUTO_CLASS_WEIGHTS = { TRUE },
         CLASS_WEIGHTS = struct_array,
         INSTANCE_WEIGHT_COL = string_value,
         L1_REG = float64_value,
         L2_REG = float64_value,
         EARLY_STOP = { TRUE },
         LEARN_RATE = float64_value,
         INPUT_LABEL_COLS = string_array,
         MAX_ITERATIONS = int64_value,
         MIN_REL_PROGRESS = float64_value,
         DATA_SPLIT_METHOD = {'AUTO_SPLIT'},
         DATA_SPLIT_EVAL_FRACTION = float64_value,
         DATA_SPLIT_COL = string_value,
         ENABLE_GLOBAL_EXPLAIN = { TRUE},
         XGBOOST_VERSION = {'1.1'})];
```

在创建模型请求中可以指定的选项包括：

+   `MODEL_TYPE`: 指定模型是提升树分类器、提升树回归器、随机森林分类器还是随机森林回归器。选项有`'BOOSTED_TREE_CLASSIFIER'`、`'BOOSTED_TREE_REGRESSOR'`、`'RANDOM_FOREST_CLASSIFIER'`和`'RANDOM_FOREST_REGRESSOR'`。

+   `BOOSTER_TYPE`（仅适用于`Boosted_Tree_Models`）：指定用于提升树模型的提升器类型。**GBTREE**代表**梯度提升树**，**DART**代表**Dropouts meet Multiple Additive Regression Trees**。

+   `NUM_PARALLEL_TREE`: 指定要生长的并行树的数量。较大的数字可以提高性能，但也会增加训练时间和内存使用。

+   `DART_NORMALIZE_TYPE`（仅适用于`Boosted_Tree_Models`）：指定用于`'TREE'`的归一化方法，表示在提升过程中根据丢弃的树的数量进行归一化，而`'FOREST'`表示根据森林中树的总数进行归一化。

+   `TREE_METHOD`: 指定用于构建集成中每个决策树的方法。`'AUTO'` 表示算法将根据数据选择最佳方法，`'EXACT'` 表示精确贪婪算法，`'APPROX'` 表示近似贪婪算法，而`'HIST'` 表示基于直方图的算法。

+   `MIN_TREE_CHILD_WEIGHT`: 指定树的孩子节点中所需的实例权重最小总和。如果总和低于此值，则节点将不会分割。

+   `COLSAMPLE_BYTREE`: 指定为每棵树随机采样的列的比例。

+   `COLSAMPLE_BYLEVEL`: 指定为树的每个级别随机采样的列的比例。

+   `COLSAMPLE_BYNODE`: 指定为树的每个分割节点随机采样的列的比例。

+   `MIN_SPLIT_LOSS`: 指定分割节点所需的最低损失减少量。

+   `MAX_TREE_DEPTH`: 指定每棵树的最大深度。

+   `SUBSAMPLE`: 指定为每棵树随机采样的训练实例的比例。

+   `AUTO_CLASS_WEIGHTS`: 如果设置为`TRUE`，算法将根据数据自动确定分配给每个类的权重。

+   `CLASS_WEIGHTS`: 指定分配给每个类的权重。这可以用于平衡数据，如果类别不平衡的话。

+   `INSTANCE_WEIGHT_COL`: 指定包含实例权重的列的名称。

+   `L1_REG`: 指定 L1 正则化参数。

+   `L2_REG`: 指定 L2 正则化参数。

+   `EARLY_STOP`: 如果设置为`TRUE`，当性能提升低于某个阈值时，训练过程将提前停止。选项是`TRUE`和`FALSE`。

+   `LEARN_RATE`（仅适用于`Boosted_Tree_Models`）：指定学习率，它控制提升过程中每次迭代的步长大小。

+   `INPUT_LABEL_COLS`: 指定包含输入特征和标签的列的名称。

+   `MAX_ITERATIONS`（仅适用于`Boosted_Tree_Models`）：指定要执行的提升迭代次数的最大值。

+   `MIN_REL_PROGRESS`: 指定继续训练过程所需的最低相对进度。

+   `DATA_SPLIT_METHOD`: 指定用于将数据分割为训练集和验证集的方法。`'AUTO_SPLIT'` 表示算法将自动分割数据，`'RANDOM'` 表示随机分割，`'CUSTOM'` 表示用户自定义分割，`'SEQ'` 表示顺序分割，而`'NO_SPLIT'` 表示不分割（使用所有数据进行训练）。

+   `DATA_SPLIT_EVAL_FRACTION`: 指定分割数据时用于验证的数据比例。

+   `DATA_SPLIT_COL`: 指定用于分割数据的列的名称。

+   `ENABLE_GLOBAL_EXPLAIN`: 如果设置为 `TRUE`，则算法将计算全局特征重要性分数。选项有 `TRUE` 和 `FALSE`。

+   `XGBOOST_VERSION`: 指定要使用的 XGBoost 版本。

### 导入模型

BQML 还允许您导入在 BigQuery 外部训练的深度学习模型。这是一个非常有用的功能，因为它为您提供了使用 BigQuery 外部更定制设置训练模型的灵活性，同时还能使用 BigQuery 的计算基础设施进行推理。

这是您可以使用导入功能的方式：

```py
{CREATE OR REPLACE MODEL} model_name
[OPTIONS(MODEL_TYPE = {'TENSORFLOW'} ,
MODEL_PATH = string_value)]
```

这里是作为导入功能一部分的可用选项：

+   `MODEL_TYPE`: 指定模型是 TensorFlow、TensorFlow Lite 还是 ONNX 模型。选项有 `'TENSORFLOW'`、`'ONNX'` 和 `'TENSORFLOW_LITE'`。

+   `MODEL_PATH`: 提供要导入到 BQML 的模型的云存储 URI。

### k-means 模型

这里是创建 k-means 模型的语法，以及您需要作为查询一部分提供的不同必需和可选参数：

```py
{CREATE OR REPLACE MODEL} model_name
[OPTIONS(MODEL_TYPE = { 'KMEANS' },
    NUM_CLUSTERS = int64_value,
    KMEANS_INIT_METHOD = { 'RANDOM' },
    KMEANS_INIT_COL = string_value,
    DISTANCE_TYPE = { 'EUCLIDEAN' | 'COSINE' },
    STANDARDIZE_FEATURES = { TRUE  },
    MAX_ITERATIONS = int64_value,
    EARLY_STOP = { TRUE  },
    MIN_REL_PROGRESS = float64_value,
    WARM_START = {  FALSE })];
```

让我们看看可以作为模型创建查询一部分指定的选项：

+   `MODEL_TYPE`: 指定模型类型。此选项是必需的。

+   `NUM_CLUSTERS`（可选）：对于 k-means 模型，这指定了在输入数据中要识别的簇的数量。默认值为 `log10(n)`，其中 `n` 是训练示例的数量。

+   `KMEANS_INIT_METHOD`（可选）：对于 k-means 模型，这指定了初始化簇的方法。默认值为 `'RANDOM'`。选项有 `'RANDOM'`、`'KMEANS++'` 和 `'CUSTOM'`。

+   `KMEANS_INIT_COL`（可选）：对于 k-means 模型，这标识了将用于初始化质心的列。此选项只能在 `KMEANS_INIT_METHOD` 的值为 `CUSTOM` 时指定。相应的列必须是 `BOOL` 类型，并且 `NUM_CLUSTERS` 模型选项必须在查询中存在，其值必须等于此列中 `TRUE` 行的总数。BQML 不能使用此列作为特征，并自动将其排除在特征之外。

+   `DISTANCE_TYPE`（可选）：对于 k-means 模型，这指定了计算两点之间距离的度量类型。默认值为 `'EUCLIDEAN'`。

+   `STANDARDIZE_FEATURES`（可选）：对于 k-means 模型，这指定了是否标准化数值特征。默认值为 `TRUE`。

+   `MAX_ITERATIONS`（可选）：最大训练迭代次数或步骤。默认值为 `20`。

+   `EARLY_STOP`（可选）：是否在相对损失改进小于为 `MIN_REL_PROGRESS` 指定值的第一次迭代后停止训练。默认值为 `TRUE`。

+   `MIN_REL_PROGRESS`（可选）：当 `EARLY_STOP` 设置为 `TRUE` 时，继续训练所需的最低相对损失改进。例如，0.01 的值指定每次迭代必须将损失降低 1%，以便继续训练。默认值为 `0.01`。

+   `WARM_START` (可选): 是否使用新的训练数据、新的模型选项或两者重新训练模型。除非明确覆盖，否则用于训练模型的初始选项将用于热启动运行。`MODEL_TYPE` 的值和训练数据模式必须在热启动模型的重新训练中保持不变。默认值是 `FALSE`。

现在让我们看看 BQML 对超参数调优的支持。

# BQML 的超参数调优

BQML 允许您在构建机器学习模型时通过使用 `CREATE MODEL` 语句来微调超参数。这个过程称为超参数调优，是一种常用的方法，通过找到最佳的超参数集来提高模型精度。

这里是一个 BigQuery SQL 语句的示例：

```py
{CREATE OR REPLACE MODEL} model_name
OPTIONS(Existing Training Options,
   NUM_TRIALS = int64_value, [, MAX_PARALLEL_TRIALS = int64_value ]
   [, HPARAM_TUNING_ALGORITHM = { 'VIZIER_DEFAULT' | 'RANDOM_SEARCH' | 'GRID_SEARCH' } ]
   [, hyperparameter={HPARAM_RANGE(min, max) | HPARAM_CANDIDATES([candidates]) }... ]
   [, HPARAM_TUNING_OBJECTIVES = { 'R2_SCORE' | 'ROC_AUC' | ... } ]
   [, DATA_SPLIT_METHOD = { 'AUTO_SPLIT' | 'RANDOM' | 'CUSTOM' | 'SEQ' | 'NO_SPLIT' } ]
   [, DATA_SPLIT_COL = string_value ]
   [, DATA_SPLIT_EVAL_FRACTION = float64_value ]
   [, DATA_SPLIT_TEST_FRACTION = float64_value ]
) AS query_statement
```

让我们看看可以作为模型创建查询一部分指定的选项：

+   `NUM_TRIALS`

    +   描述：这确定了要训练的最大子模型数量。在训练 `num_trials` 个子模型或搜索空间耗尽后，将停止调优。最大值是 100。

    +   参数：`int64_value` 必须是一个范围从 1 到 100 的 `INT64` 值。

注意

建议至少使用 (`num_hyperparameters` * 10) 次试验来进行模型调优。

+   `MAX_PARALLEL_TRIALS`

    +   描述：这表示可以同时运行的试验的最大数量。默认值是 1，最大值是 5。

    +   参数：`int64_value` 必须是一个范围从 1 到 5 的 `INT64` 值。

注意

较大的 `max_parallel_trials` 值可以加快超参数调优的速度，但对于 `VIZIER_DEFAULT` 调优算法，它可能会降低最终模型的质量，因为并行试验无法从并发训练结果中受益。

+   `HPARAM_TUNING_ALGORITHM`

    +   描述：这确定了超参数调优的算法，并支持以下值：

        +   `VIZIER_DEFAULT` (默认并推荐): 使用默认的 Vertex AI Vizier 算法，该算法结合了贝叶斯优化和高斯过程等高级搜索算法，并采用迁移学习来利用先前调优的模型。

        +   `RANDOM_SEARCH`: 采用随机搜索来探索搜索空间。

        +   `GRID_SEARCH`: 使用网格搜索来探索搜索空间。这仅在每个超参数的搜索空间都是离散的情况下才可用。

+   `HYPERPARAMETER`

    语法：`hyperparameter={HPARAM_RANGE(min, max) | HPARAM_CANDIDATES([candidates]) }...`

    此参数配置超参数的搜索空间。请参考每种模型类型的超参数和目标，以了解哪些可调超参数受支持：

    +   `HPARAM_RANGE(min, max)`: 指定超参数的连续搜索空间 – 例如，`learn_rate` = `HPARAM_RANGE(0.0001, 1.0)`

    +   `HPARAM_CANDIDATES([candidates])`: 指定具有离散值的超参数 – 例如，`OPTIMIZER=HPARAM_CANDIDATES([`adagrad`, `sgd`, `ftrl`]`)`

+   `HPARAM_TUNING_OBJECTIVES`

    此参数指定模型的客观指标。候选指标是模型评估指标的一个子集。目前仅支持一个客观指标。请参阅*表 6.7*，以查看支持的模型类型、超参数和调整目标。

| **模型类型** | **超参数目标** | **超参数** |
| --- | --- | --- |
| `LINEAR_REG` |

+   `mean_absolute_error`

+   `mean_squared_error`

+   `mean_squared_log_error`

+   `median_absolute_error`

+   `r2_score (默认)`

+   `explained_variance`

|

+   `l1_reg`

+   `l2_reg`

|

| `LOGISTIC_REG` |
| --- |

+   `precision`

+   `recall`

+   `accuracy`

+   `f1_score`

+   `log_loss`

+   `roc_auc (默认)`

|

+   `l1_reg`

+   `l2_reg`

|

| `KMEANS` |
| --- |

+   `davies_bouldin_index`

| `num_clusters` |
| --- |
| `MATRIX_``FACTORIZATION (``隐式/显式``) |

+   `mean_average_precision (``显式模型``)`

+   `mean_squared_error (隐式/显式)`

+   `normalized_discounted_cumulative_gain (``显式模型``)`

+   `average_rank (``显式模型``)`

|

+   `num_factors`

+   `l2_reg`

+   `wals_alpha(隐式模型仅)`

|

| `DNN_CLASSIFIER` |
| --- |

+   `precision`

+   `recall`

+   `accuracy`

+   `f1_score`

+   `log_loss`

+   `roc_auc (默认)`

|

+   `batch_size`

+   `dropout`

+   `hidden_units`

+   `learn_rate`

+   `optimizer`

+   `l1_reg`

+   `l2_reg`

+   `activation_fn`

|

| `DNN_REGRESSOR` |
| --- |

+   `mean_absolute_error`

+   `mean_squared_error`

+   `mean_squared_log_error`

+   `median_absolute_error`

+   `r2_score (默认)`

+   `explained_variance`

|

| `BOOSTED_TREE_``CLASSIFIER` |
| --- |

+   `precision`

+   `recall`

+   `accuracy`

+   `f1_score`

+   `log_loss`

+   `roc_auc (默认)`

|

+   `learn_rate`

+   `l1_reg`

+   `l2_reg`

+   `dropout`

+   `max_tree_depth`

+   `subsample`

+   `min_split_loss`

+   `num_parallel_tree`

+   `min_tree_child_weight`

+   `colsample_bytree`

+   `colsample_bylevel`

+   `colsample_bynode`

+   `booster_type`

+   `dart_normalize_type`

+   `tree_method`

|

| `BOOSTED_TREE_``REGRESSOR` |
| --- |

+   `mean_absolute_error`

+   `mean_squared_error`

+   `mean_squared_log_error`

+   `median_absolute_error`

+   `r2_score (默认)`

+   `explained_variance`

|

| `RANDOM_FOREST_``CLASSIFIER` |
| --- |

+   `precision`

+   `recall`

+   `accuracy`

+   `f1_score`

+   `log_loss`

+   `roc_auc (默认)`

|

+   `l1_reg`

+   `l2_reg`

+   `max_tree_depth`

+   `subsample`

+   `min_split_loss`

+   `num_parallel_tree`

+   `min_tree_child_weight`

+   `colsample_bytree`

+   `colsample_bylevel`

+   `colsample_bynode`

+   `tree_method`

|

| `RANDOM_FOREST_``REGRESSOR` |
| --- |

+   `mean_absolute_error`

+   `mean_squared_error`

+   `mean_squared_log_error`

+   `median_absolute_error`

+   `r2_score (默认)`

+   `explained_variance`

|

表 6.7 – 模型类型支持的超参数目标

[`cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-hyperparameter-tuning`](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-hyperparameter-tuning)

现在我们来看看在尝试评估机器学习模型时可以使用的 BQML 功能。

# 评估训练模型

一旦训练了 BQML 模型，您将想要评估关键性能统计指标，具体取决于模型类型。您可以通过使用`ML.EVALUATE`函数来实现，如下所示：

```py
ML.EVALUATE(MODEL model_name
           [, {TABLE table_name | (query_statement)}]
           [, STRUCT<threshold FLOAT64,
                     perform_aggregation BOOL,
                     horizon INT64,
                     confidence_level FLOAT64> settings]))])
```

让我们看看您可以在评估查询中指定的选项：

+   `model_name`：正在评估的模型名称

+   `table_name`（可选）：包含评估数据的表的名称

+   `query_statement`（可选）：用于生成评估数据的查询

+   `threshold`（可选）：在评估期间用于二分类模型的自定义阈值值

+   `perform_aggregation`（可选）：标识预测准确度评估级别的布尔值

+   `horizon`（可选）：用于计算评估指标的预测时间点的数量

+   `confidence_level`（可选）：预测区间内未来值的百分比

`ML.Evaluate`函数的输出取决于正在评估的模型类型：

| **模型类型** | **返回字段** |
| --- | --- |
| 回归模型 |

+   `mean_absolute_error`

+   `mean_squared_error`

+   `mean_squared_log_error`

+   `median_absolute_error`

+   `r2_score`

+   `explained_variance`

|

| 分类模型 |
| --- |

+   `precision`

+   `recall`

+   `accuracy`

+   `f1_score`

+   `log_loss`

+   `roc_auc`

|

| k-means 模型 |
| --- |

+   `Davies-Bouldin 指数`

+   `均方距离`

|

| 带有隐式反馈的矩阵分解模型 |
| --- |

+   `mean_average_precision`

+   `mean_squared_error`

+   `normalized_discounted_cumulative_gain`

+   `average_rank`

|

| 带有显式反馈的矩阵分解模型 |
| --- |

+   `mean_absolute_error`

+   `mean_squared_error`

+   `mean_squared_log_error`

+   `median_absolute_error`

+   `r2_score`

+   `explained_variance`

|

| PCA 模型 |
| --- |

+   `total_explained_variance_ratio`

|

| 带有输入数据和 perform_aggregation = false 的时序 ARIMA_PLUS 或 ARIMA_PLUS_XREG 模型 |
| --- |

+   `time_series_id_col` 或 `time_series_id_cols`

+   `time_series_timestamp_col`

+   `time_series_data_col`

+   `forecasted_time_series_data_col`

+   `lower_bound`

+   `upper_bound`

+   `absolute_error`

+   `absolute_percentage_error`

|

| 带有输入数据和 perform_aggregation = true 的时序 ARIMA_PLUS 或 ARIMA_PLUS_XREG 模型 |
| --- |

+   `time_series_id_col` 或 `time_series_id_cols`

+   `mean_absolute_error`

+   `mean_squared_error`

+   `root_mean_squared_error`

+   `mean_absolute_percentage_error`

+   `symmetric_mean_absolute_percentage_error`

|

| 没有输入数据的时序 ARIMA_PLUS 模型 |
| --- |

+   `time_series_id_col` 或 `time_series_id_cols`

+   `non_seasonal_p`

+   `non_seasonal_d`

+   `non_seasonal_q`

+   `has_drift`

+   `log_likelihood`

+   `AIC`

+   `variance`

+   `seasonal_periods`

+   `has_holiday_effect`

+   `has_spikes_and_dips`

+   `has_step_change`

|

| 自动编码器模型 |
| --- |

+   `mean_absolute_error`

+   `mean_squared_error`

+   `mean_squared_log_error`

|

| 远程模型 |
| --- |

+   `remote_eval_metrics`

|

表 6.8 – ML.Evaluate 输出

在下一节中，我们将探讨如何使用您的 BQML 模型进行推理。

# 使用 BQML 进行推理

在监督式机器学习中，最终目标是使用训练好的模型对新数据进行预测。BQML 提供了 `ML.PREDICT` 函数来实现这一目的。使用此函数，您可以通过向训练好的模型提供新数据来轻松预测结果。只要至少完成了一次迭代，该函数就可以在模型创建期间、创建后或失败后使用。该函数返回一个与输入表行数相同的表，其中包含输入表的所有列以及模型的所有输出列，输出列名称以 `predicted_` 为前缀。

```py
ML.PREDICT(MODEL model_name,
          {TABLE table_name | (query_statement)}
          [, STRUCT<threshold FLOAT64,
          keep_original_columns BOOL> settings)])
```

`ML.PREDICT` 函数的输出字段取决于所使用的模型类型：

| **模型类型** | **输出列** |
| --- | --- |
| 线性回归 Boosted 树回归器随机森林回归器 DNN 回归器 | `predicted_<label_column_name>` |
| 二元逻辑回归 Boosted 树分类器随机森林分类器 DNN 分类器多类逻辑回归 | `predicted_<label_column_name>,` `predicted_<label_column_name>_probs` |
| k-means | `centroid_id, nearest_centroids_distance` |
| PCA | `principal_component_<index>,` 输入列（如果 keep_original_columns 设置为 true） |
| 自编码器 | `latent_col_<index>,` 输入列 |
| TensorFlow Lite | TensorFlow Lite 模型预测方法的输出 |
| 远程模型 | 包含所有 Vertex AI 端点输出字段的输出列，以及包含 Vertex AI 端点状态消息的 remote_model_status 字段 |
| ONNX 模型 | ONNX 模型预测方法的输出 |
| XGBoost 模型 | XGBoost 模型预测方法的输出 |

表 6.9 – ML.Predict 输出

现在，让我们通过一个实际操作练习来使用 BQML 训练一个机器学习模型，并使用它进行预测。

# 用户练习

请参阅本书 GitHub 仓库中 *第六章* 的笔记本，*构建 ML 模型的低代码选项*，以进行围绕训练 BQML 模型的实际操作练习。在这个练习中，您将使用 BigQuery 中可用的公共数据集之一来训练一个模型，预测客户下个月违约的可能性。

# 摘要

BQML 是一种强大的工具，适用于希望轻松训练机器学习模型且使用低代码选项在 GCP 中构建和部署模型的数据科学家和分析师。使用 BQML，用户可以利用 BigQuery 的强大功能，快速轻松地创建模型，而无需编写复杂的代码。

在本章中，我们探讨了 BQML 的功能和优势。我们看到了它如何通过 SQL 查询提供简单直观的界面来训练模型。我们还探讨了 BQML 的关键特性，包括在 BigQuery 中直接执行数据预处理和特征工程的能力，以及通过原生评估函数评估模型性能的能力。

BQML 的一个关键优点是它与 BigQuery 的集成，这使得它易于扩展和管理大数据集。这使得它成为处理大量数据并需要快速构建和部署模型的公司和组织的一个很好的选择。

BQML 的另一个优点是它支持广泛的 ML 模型，包括线性回归、逻辑回归、k-means 聚类等。这使得它成为一个多功能的工具，可以用于各种用例，从预测客户流失到聚类数据进行分析。

我们还讨论了 BQML 的一些局限性。例如，虽然它提供了低代码选项来构建和部署模型，但它可能不适合需要自定义模型或大量特征工程更复杂的使用场景。此外，虽然 BQML 提供了一系列用于评估模型性能的指标，但用户可能需要进行额外的分析才能完全理解他们模型的有效性。

尽管存在这些局限性，BQML 仍然是数据科学家和分析人员快速轻松构建和部署 ML 模型的有力工具。它与 BigQuery 和其他 GCP 服务的集成使其成为需要处理大量数据的公司和组织的一个很好的选择，而其对广泛模型和指标的支持使其成为各种用例的多功能工具。

总体而言，BQML 是 GCP 中可用的 ML 工具套件中的一个宝贵补充。它的低代码界面、与 BigQuery 的集成以及对广泛模型的支撑，使其成为数据科学家和分析人员的一个很好的选择，他们希望专注于数据和洞察，而不是复杂的代码和基础设施。使用 BQML，用户可以快速轻松地构建和部署模型，从而从他们的数据中提取有价值的见解并做出数据驱动的决策。

在下一章中，我们将探讨如何在 Vertex AI 上使用其无服务器训练功能来训练完全自定义的 TensorFlow 深度学习模型。这一章还将深入探讨使用 TensorFlow 构建模型、打包以提交给 Vertex AI、监控训练进度以及评估训练后的模型。
