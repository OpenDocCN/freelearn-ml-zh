# *第十章*：使用 SageMaker Model Monitor 监控生产中的机器学习模型

将模型投入生产进行推理并不是机器学习生命周期的结束。这只是重要话题的开始：我们如何确保模型按照设计以及在实际生活中按预期运行？使用 SageMaker Studio 监控模型在生产中的表现，尤其是在模型从未见过的数据上，变得容易起来。您将学习如何为在 SageMaker 中部署的模型设置模型监控，检测数据漂移和性能漂移，并在 SageMaker Studio 中可视化结果，以便您可以自动让系统检测您的机器学习模型的退化。

在本章中，我们将学习以下内容：

+   理解机器学习中的漂移

+   在 SageMaker Studio 中监控数据和模型性能的漂移

+   在 SageMaker Studio 中审查模型监控结果

# 技术要求

对于本章，您需要访问[`github.com/PacktPublishing/Getting-Started-with-Amazon-SageMaker-Studio/tree/main/chapter10`](https://github.com/PacktPublishing/Getting-Started-with-Amazon-SageMaker-Studio/tree/main/chapter10)中的代码。

# 理解机器学习中的漂移

在生产中，一个机器学习模型需要对其性能进行仔细和持续的监控。一旦模型经过训练和评估，并不能保证它在生产环境中的表现与测试环境中的表现相同。与软件应用不同，软件应用可以通过单元测试来测试所有可能的边缘情况，而监控和检测机器学习模型的问题则相对困难。这是因为机器学习模型使用概率、统计和模糊逻辑来推断每个输入数据点的结果，而测试，即模型评估，通常是在没有真正了解生产数据的情况下进行的。数据科学家在生产之前能做的最好的事情就是从与真实世界数据非常接近的样本中创建训练数据，并使用样本外策略评估模型，以便得到一个无偏的关于模型在未见数据上表现的想法。在生产中，模型对进入的数据是完全未知的；如何评估实时模型性能，以及如何对评估采取行动，是机器学习模型生产化的一个关键话题。

模型性能可以通过两种方法进行监控。一种更直接的方法是捕获未见数据的真实情况，并将预测与真实情况进行比较。第二种方法是，将推理数据的统计分布和特征与训练数据进行比较，作为判断模型是否按预期行为的一个代理。

第一种方法需要在预测事件发生后确定真实结果（基线），这样我们就可以直接计算数据科学家在模型评估期间使用的相同性能指标。然而，在某些用例中，真实结果（基线）可能落后于事件很长时间，甚至可能根本不可用。

第二种方法基于这样一个前提，即机器学习模型从训练数据中统计和概率性地学习，当提供来自不同统计分布的新数据集时，其行为会有所不同。当数据不是来自相同的统计分布时，模型会返回无意义的输出。这被称为**协变量漂移**。因此，检测数据中的协变量漂移可以更实时地估计模型的表现。

**Amazon SageMaker 模型监控器**是 SageMaker 中的一个功能，它通过设置数据捕获、计算基线统计信息和按计划监控流量到您的 SageMaker 端点的漂移，持续监控托管在 SageMaker 上的模型质量。SageMaker 模型监控器有四种类型的监控器：

+   **模型质量监控器**：通过计算预测的准确性和实际基线标签来监控模型的性能

+   **数据质量监控器**：通过将特征与基线训练数据的特征进行比较，监控推理数据的统计特征

+   **模型可解释性监控器**：与 SageMaker Clarify 集成，使用 Shapley 值在时间上计算特征归因

+   **模型偏差监控器**：与 SageMaker Clarify 集成，监控数据和模型预测的偏差

一旦为端点设置了模型监控，您就可以在 SageMaker Studio 中可视化随时间推移的漂移和任何数据问题。让我们按照本章中的 ML 用例学习如何在 SageMaker Studio 中设置 SageMaker 模型监控器。我们将重点关注模型质量和数据质量监控。

# 在 SageMaker Studio 中监控数据和性能漂移

在本章中，让我们考虑一个 ML 场景：我们训练一个 ML 模型并将其托管在端点上。我们还向端点创建了人工推理流量，每个数据点都注入了随机扰动。这是为了向数据引入噪声、缺失和漂移。然后我们继续使用 SageMaker 模型监控器创建数据质量监控器和模型质量监控器。我们使用一个简单的 ML 数据集，即 UCI 的鲍鱼数据集([`archive.ics.uci.edu/ml/datasets/abalone`](https://archive.ics.uci.edu/ml/datasets/abalone))，进行此演示。使用此数据集，我们训练一个回归模型来预测环数，这与鲍鱼的年龄成比例。

## 训练和托管模型

我们将遵循以下步骤来设置模型监控之前所需的内容——获取数据、训练模型、托管它并创建流量：

1.  使用**Python 3 (Data Science)**内核和**ml.t3.median**实例打开`Getting-Started-with-Amazon-SageMaker-Studio/chapter10/01-train_host_predict.ipynb`笔记本。

1.  运行前三个单元格以设置库和 SageMaker 会话。

1.  从源读取数据并进行最小处理，即把分类变量`Sex`编码为整数，以便我们稍后可以使用`XGBoost`算法进行训练。此外，我们将目标列`Rings`的类型改为浮点数，以确保真实值和模型预测（回归）的值在模型监控中保持一致。

1.  将数据随机分为训练集（80%）、验证集（10%）和测试集（10%）。然后，将数据保存到本地驱动器以进行模型推理，并将其上传到 S3 进行模型训练。

1.  对于模型训练，我们使用 SageMaker 内置的`XGBoost`算法，对于回归问题使用`reg:squarederror`目标函数：

    ```py
    image = image_uris.retrieve(region=region, 
                      framework='xgboost', version='1.3-1')
    xgb = sagemaker.estimator.Estimator(...)
    xgb.set_hyperparameters(objective='reg:squarederror', num_round=20)
    data_channels={'train': train_input, 'validation': val_input}
    xgb.fit(inputs=data_channels, ...)
    ```

训练大约需要 5 分钟。

1.  模型训练完成后，我们可以使用 SageMaker 端点通过`xgb.deploy()`托管模型，就像我们在*第七章*，“在云中托管 ML 模型：最佳实践”中学到的那样。然而，默认情况下，SageMaker 端点不会保存传入的推理数据副本。为了监控模型性能和数据漂移，我们需要指导端点持久化传入的推理数据。我们使用`sagemaker.model_monitor.DataCaptureConfig`设置端点后的数据捕获，用于监控目的：

    ```py
    data_capture_config = DataCaptureConfig(enable_capture=True, 
               sampling_percentage=100,                                         
               destination_s3_uri=s3_capture_upload_path)
    ```

在`destination_s3_uri`中指定 S3 存储桶位置。`sampling_percentage`可以是`100`（%）或更低，具体取决于你预期的实际流量量。我们需要确保我们捕获足够大的样本量，以便稍后进行任何统计比较。如果模型推理流量稀疏，例如每小时 100 次推理，你可能希望使用 100%的样本进行模型监控。如果你有高频率的模型推理用例，你可能能够使用更小的百分比。

1.  我们可以使用`data_capture_config`将模型部署到端点：

    ```py
    predictor = xgb.deploy(...,
                   data_capture_config=data_capture_config)
    ```

1.  一旦端点准备就绪，让我们在验证数据集上应用回归模型，以创建用于模型质量监控的基线数据集。基线数据集应包含 CSV 文件中的真实值和模型预测的两列。然后，我们将 CSV 上传到 S3 存储桶位置：

    ```py
    pred=predictor.predict(df_val[columns_no_target].values)
    pred_f = [float(i) for i in pred[0]]
    df_val['Prediction']=pred_f
    model_quality_baseline_suffix = 'abalone/abalone_val_model_quality_baseline.csv'
    df_val[['Rings', 'Prediction']].to_csv(model_quality_baseline_suffix, index=False)
    model_quality_baseline_s3 = sagemaker.s3.S3Uploader.upload(
            local_path=model_quality_baseline_suffix,
            desired_s3_uri=desired_s3_uri,
            sagemaker_session=sess)
    ```

接下来，我们可以使用测试数据集在端点上做一些预测。

## 创建推理流量和真实值

为了模拟现实生活中的推理流量，我们从测试数据集中取样本并添加随机扰动，例如随机缩放和删除特征。我们可以预期这会模拟数据漂移并扭曲模型性能。然后，我们将扰动数据发送到端点进行预测，并将真实值保存到 S3 存储桶位置。请按照以下步骤在同一笔记本中操作：

1.  这里，我们有两个添加随机扰动的函数：`add_randomness()`和`drop_randomly()`。前者函数随机乘以每个特征值（除了`Sex`函数），并随机分配一个二进制值给`Sex`。后者函数随机删除一个特征，并用`NaN`（不是一个数字）填充它。

1.  我们还有一个`generate_load_and_ground_truth()`函数，用于从测试数据的每一行读取数据，应用扰动，调用端点进行预测，在字典`gt_data`中构建真实值，并将其作为 JSON 文件上传到 S3 存储桶。值得注意的是，为了确保我们建立推理数据和真实值之间的对应关系，我们将每一对关联到`inference_id`。这种关联将允许模型监控器合并推理和真实值以进行分析：

    ```py
    def generate_load_and_ground_truth():
        gt_records=[]
        for i, row in df_test.iterrows():
            suffix = uuid.uuid1().hex
            inference_id = f'{i}-{suffix}'

            gt = row['Rings']
            data = row[columns_no_target].values
            new_data = drop_random(add_randomness(data))
            new_data = convert_nparray_to_string(new_data)
            out = predictor.predict(data = new_data, 
                           inference_id = inference_id)
            gt_data = {'groundTruthData': {
                                'data': str(gt), 
                                'encoding': 'CSV',
                            },
                       'eventMetadata': {
                                'eventId': inference_id,
                            },
                       'eventVersion': '0',
                       }
            gt_records.append(gt_data)
        upload_ground_truth(gt_records, ground_truth_upload_path, datetime.utcnow()) 
    ```

我们在`generate_load_and_ground_truth_forever()`函数中用`while`循环包装这个函数，这样我们就可以使用线程进程生成持久流量，直到笔记本关闭：

```py
def generate_load_and_ground_truth_forever():
    while True:
        generate_load_and_ground_truth()
from threading import Thread
thread = Thread(target=generate_load_and_ground_truth_forever)
thread.start()
```

1.  最后，在我们设置第一个模型监控器之前，让我们看看如何捕获推理流量：

    ```py
    capture_file = get_obj_body(capture_files[-1])
    print(json.dumps(json.loads(capture_file.split('\n')[-2]), indent=2))
    {
      "captureData": {
        "endpointInput": {
          "observedContentType": "text/csv",
          "mode": "INPUT",
          "data": "1.0,0.54,0.42,0.14,0.805,0.369,0.1725,0.21",
          "encoding": "CSV"
        },
        "endpointOutput": {
          "observedContentType": "text/csv; charset=utf-8",
          "mode": "OUTPUT",
          "data": "9.223058700561523",
          "encoding": "CSV"
        }
      },
      "eventMetadata": {
        "eventId": "a9d22bac-094a-4610-8dde-689c6aa8189b",
        "inferenceId": "846-01234f26730011ecbb8b139195a02686",
        "inferenceTime": "2022-01-11T17:00:39Z"
      },
      "eventVersion": "0"
    }
    ```

注意`captureData.endpointInput.data`函数通过`predictor.predict()`使用`eventMetadata.` `inferenceId`中的唯一推理 ID 来获取推理数据。模型端点的输出在`captureData.endpointOutput.data`中。

我们已经完成了所有准备工作。现在我们可以继续在 SageMaker Studio 中创建模型监控器。

## 创建数据质量监控器

数据质量监控器会将传入的推理数据的统计信息与基线数据集的统计信息进行比较。您可以通过 SageMaker Studio UI 或 SageMaker Python SDK 设置数据质量监控器。我将通过 Studio UI 演示简单的设置过程：

1.  在左侧侧边栏的**端点**注册表中找到您新托管的端点，如图*图 10.1*所示。双击条目以在主工作区域中打开它：

![图 10.1 – 打开端点详情页

![img/B17447_10_001.jpg]

图 10.1 – 打开端点详情页

1.  点击**数据质量**选项卡，然后**创建监控计划**，如图*图 10.2*所示：

![图 10.2 – 在端点详情页创建数据质量监控计划

![img/B17447_10_002.jpg]

图 10.2 – 在端点详情页创建数据质量监控计划

1.  在设置的第一个步骤中，如图*图 10.3*所示，我们选择一个 IAM 角色，该角色具有访问权限，可以读取和写入我们在以下页面指定的存储桶位置的结果。让我们选择`3600`秒（1 小时），这样监控作业就不会流入下一个小时。在页面底部，我们保持**启用指标**开启，这样模型监控器计算的指标也会发送到 Amazon CloudWatch。这允许我们在 CloudWatch 中可视化和分析指标。点击**继续**：

![图 10.3 – 数据质量监控器设置步骤 1

![img/B17447_10_003.jpg]

图 10.3 – 数据质量监控设置步骤 1

1.  在第二步中，如图*图 10.4*所示，我们配置了每小时监控作业的基础设施和输出位置。需要配置的基础设施是每小时将要创建的 SageMaker Processing 作业。我们将计算实例（实例类型、数量和磁盘卷大小）保留为默认设置。然后，我们提供一个监控结果的输出存储桶位置以及加密和网络（VPC）选项：

![图 10.4 – 数据质量监控设置步骤 2

![img/B17447_10_004.jpg]

图 10.4 – 数据质量监控设置步骤 2

1.  在第三步中，如图*图 10.5*所示，我们配置了基线计算。监控设置完成后，将启动一个一次性 SageMaker Processing 作业来计算基线统计数据。未来的周期性监控作业将使用基线统计数据来判断是否发生了漂移。我们将 CSV 文件位置提供到基线数据集的 S3 位置。我们将训练数据上传到 S3 存储桶，完整路径在`train_data_s3`变量中。我们提供一个 S3 输出位置到基线 S3 输出位置。因为我们的训练数据 CSV 文件的第一行包含一个特征名称，所以我们选择具有 1 GB 基线卷的`ml.m5.xlarge`实例就足够了。点击**继续**：

![图 10.5 – 数据质量监控设置步骤 3

![img/B17447_10_005.jpg]

图 10.5 – 数据质量监控设置步骤 3

1.  在最后的**附加配置**页面，您可以选择为周期性监控作业提供预处理和后处理脚本。您可以使用自己的脚本自定义特征和模型输出。当您使用自定义容器进行模型监控时，此扩展不受支持。在我们的案例中，我们使用 SageMaker 的内置容器。有关预处理和后处理脚本的更多信息，请访问[`docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-pre-and-post-processing.html`](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-pre-and-post-processing.html)。

如果您回到**端点详情**页面，在**数据质量**选项卡下，如图*图 10.2*所示，您现在可以看到一个新的具有**已安排**状态的监控计划。现在正在启动一个基线作业来从基线训练数据集中计算各种统计数据。基线作业完成后，第一个每小时监控作业将在小时顶部的 20 分钟内作为一个 SageMaker Processing 作业启动。监控作业计算在小时内收集的推理数据的统计数据，并将其与基线进行比较。我们将在*在 SageMaker Studio 中查看模型监控结果*部分中回顾监控结果。

现在，让我们继续创建模型质量监控器以监控模型性能。

## 创建模型质量监控器

创建模型质量监控器遵循与创建数据质量监控器类似的过程，但更注重处理 S3 中的模型预测和真实标签。让我们按照以下步骤设置一个模型质量监控器，以监控模型性能随时间的变化：

1.  在同一端点的**端点详情**页上，转到**模型质量**选项卡，并点击**创建监控计划**，如图*图 10.6*所示：

![图 10.6 – 在端点详情页创建模型质量监控计划

![img/B17447_10_006.jpg]

图 10.6 – 在端点详情页创建模型质量监控计划

1.  在第一页，**计划**，我们选择监控作业的 IAM 角色、调度频率等，类似于之前*创建数据质量监控器部分*中的*步骤 3*。

1.  在第二页，**监控作业配置**，如图*图 10.7*所示，我们配置监控作业的实例和输入/输出：

![图 10.7 – 设置模型质量监控器的输入和输出

![img/B17447_10_007.jpg]

图 10.7 – 设置模型质量监控器的输入和输出

输入既指模型从端点预测的结果，也指我们在笔记本中上传的真实标签文件。在`24`小时内。对于**推理属性**的`0`，指定第一个值是模型输出，并保留**概率**为空。

注意

如果您的模型内容类型是 JSON/JSON Lines，您会在`{prediction: {"predicted_label":1, "probability":0.68}}`中指定 JSON 路径，您会在`"prediction.probability"`中指定`"prediction.predicted_label"`，在**概率**中。

对于笔记本中的`ground_truth_upload_path`变量。对于**S3 输出位置**，我们指定一个 S3 存储桶位置，以便模型监控器保存输出。最后，您可以可选地配置监控作业的加密和 VPC。点击**继续**以进行下一步。

1.  在第三页，将笔记本中的`model_quality_baseline_s3`变量拖放到**基线数据集 S3 位置**字段。对于**基线 S3 输出位置**，我们提供一个 S3 位置以保存基线结果。在**基线数据集格式**中选择**带标题的 CSV**。将实例类型和配置保留为默认值。

这是为了配置 SageMaker Processing 作业进行一次性基线计算。在最后三个字段中，我们放入相应的 CSV 标题名称——`Rings`用于`Prediction`的**基线推理属性**——并将字段保留为空，因为我们的模型不产生概率。点击**继续**：

![图 10.8 – 配置模型质量监控器的基线计算

![img/B17447_10_008.jpg]

图 10.8 – 配置模型质量监控器的基线计算

1.  在**附加配置**中，我们可以向监控器提供预处理和后处理脚本，就像数据质量监控器的情况一样。让我们跳过这一部分，通过点击**启用模型监控**来完成设置。

现在，我们已经创建了模型质量监控器。您可以在**ENDPOINT 详情**页面的**模型质量**选项卡下看到监控计划处于**已安排**状态。类似于数据质量监控器，将启动一个基线处理作业来使用基线数据集计算基线模型性能。每小时监控作业也将作为一个 SageMaker Processing 作业在每小时顶部 20 分钟内启动，以便从每小时收集的推理数据中计算模型性能指标，并与基线进行比较。我们将在下一节中回顾监控结果，*在 SageMaker Studio 中查看模型监控结果*。

# 在 SageMaker Studio 中查看模型监控结果

SageMaker Model Monitor 对传入的推理数据进行各种统计计算，将它们与预先计算的基线统计进行比较，并将结果报告回指定的 S3 存储桶，您可以在 SageMaker Studio 中可视化这些结果。

对于数据质量监控器，SageMaker Model Monitor 预构建的默认容器，即我们所使用的，对基线数据集和推理数据进行每特征统计。这些统计包括平均值、总和、标准差、最小值和最大值。数据质量监控器还会检查数据缺失情况，并检查传入推理数据的数据类型。您可以在[`docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-interpreting-statistics.html`](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-interpreting-statistics.html)找到完整的列表。

对于模型质量监控器，SageMaker 根据配置的 ML 问题类型计算模型性能指标。在本章的回归示例中，SageMaker 的模型质量监控器正在计算**平均绝对误差**（**MAE**）、**平均平方误差**（**MSE**）、**均方根误差**（**RMSE**）和**R 平方**（**r2**）值。您可以在[`docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html`](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)找到回归、二分类和多分类问题计算的完整指标列表。

您可以在**ENDPOINT 详情**页面的**监控作业历史记录**选项卡中查看随时间启动的监控作业列表，如图 10.9 所示：

![图 10.9 – 查看监控作业列表。双击行项目可进入特定作业的详细信息页]

![img/B17447_10_009.jpg]

图 10.9 – 查看监控作业列表。双击行项目可进入特定作业的详细信息页

当您双击行项目时，您将被带到特定监控作业的详细信息页面，如图*图 10.10*所示。由于我们在将数据发送到端点之前对其进行了扰动，因此数据包含不规则性，例如缺失。这被数据质量监控器捕获：

![图 10.10 – 数据质量监控作业的详细信息及违规情况

![img/B17447_10_010.jpg]

图 10.10 – 数据质量监控作业的详细信息及违规情况

我们还可以打开模型质量监控作业，以了解模型是否按预期运行。如图*图 10.11*所示，我们可以看到所有计算出的指标都提出了违规。我们知道这将会发生，因为这很大程度上是由于我们对数据引入的扰动。SageMaker 模型监控器能够检测到这样的问题：

![图 10.11 – 模型监控作业的详细信息及违规情况

![img/B17447_10_011.jpg]

图 10.11 – 模型监控作业的详细信息及违规情况

我们还可以从监控作业中创建可视化。让我们按照以下步骤创建数据质量监控器的图表：

1.  在**端点详细信息**页面，转到**数据质量**选项卡，然后点击如图*图 10.12*所示的**添加图表**按钮：

![图 10.12 – 为数据质量监控器添加可视化

![img/B17447_10_012.jpg]

图 10.12 – 为数据质量监控器添加可视化

1.  右侧将出现一个图表属性配置侧边栏，如图*图 10.13*所示。我们可以通过指定时间线、统计信息和我们要绘制的特征来创建图表。根据您启用监控器的时间长度，您可以选择一个时间范围进行可视化。例如，我选择了**1 天**的时间范围、**平均**统计信息和**feature_baseline_drift_Length**来查看过去一天中**Length**特征的平均基线漂移度量：

![图 10.13 – 在 SageMaker Studio 中可视化特征漂移

![img/B17447_10_013.jpg]

图 10.13 – 在 SageMaker Studio 中可视化特征漂移

1.  您可以通过点击**添加图表**按钮来可选地添加更多图表。

1.  同样，我们可以使用过去 24 小时的**mse**指标来可视化模型性能，如图*图 10.14*所示：

![图 10.14 – 在 SageMaker Studio 中可视化 mse 回归指标

![img/B17447_10_014.jpg]

图 10.14 – 在 SageMaker Studio 中可视化 mse 回归指标

注意

为了节省成本，当您完成示例后，请确保取消注释并运行`01-train_host_predict.ipynb`中的最后几个单元格，以删除监控计划和端点，从而停止向您的 AWS 账户收费。

# 摘要

在本章中，我们专注于机器学习中的数据漂移和模型漂移，以及如何使用 SageMaker 模型监控器和 SageMaker Studio 来监控它们。我们演示了如何在 SageMaker Studio 中设置数据质量监控器和模型质量监控器，以持续监控模型的行为和输入数据的特征，在一个回归模型部署在 SageMaker 端点并且持续推理流量击中端点的情况下。我们引入了一些随机扰动到推理流量中，并使用 SageMaker 模型监控器来检测模型和数据的不当行为。通过这个例子，你还可以将 SageMaker 模型监控器部署到你的用例中，为你的生产模型提供可见性和保障。

在下一章中，我们将学习如何使用 SageMaker Projects、Pipelines 和模型注册表来操作化一个机器学习项目。我们将讨论当前机器学习中的一个重要趋势，即**持续集成/持续交付**（**CI/CD**）和**机器学习运营**（**MLOps**）。我们将演示如何使用 SageMaker 的功能，如 Projects、Pipelines 和模型注册表，使你的机器学习项目可重复、可靠和可重用，并具有强大的治理能力。
