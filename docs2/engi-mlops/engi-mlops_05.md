# 第四章：机器学习管道

在本章中，我们将通过使用 MLOps 方法通过实际示例探索和实现 **机器学习**（**ML**）管道。我们将通过解决我们在 *第三章* 中一直在工作的业务问题，即 *代码遇见数据*，来学习更多。这种理论与实践相结合的学习方法将确保您将全面了解为您的或您公司的问题构建和实现 ML 管道的知识。ML 管道具有模块化脚本或代码，执行所有传统的 ML 步骤，例如在训练或重新训练任何模型之前进行数据预处理、特征工程和特征缩放。

我们从本章开始，通过执行特征工程和缩放，将我们在上一章中处理过的预处理数据摄取进来，以便为 ML 训练做好准备。我们将发现 ML 管道的原理，并在业务问题中实现它们。接下来，我们将探讨 ML 模型训练、超参数调整和训练模型的测试。最后，我们将学习如何打包模型及其所需工件。我们将注册模型以进行进一步评估，并将部署 ML 模型。在本章中，我们将涵盖以下主要主题：

+   掌握 ML 管道的基础知识

+   数据摄取和特征工程

+   ML 训练和超参数优化

+   模型测试和定义指标

+   模型打包

+   注册模型和生产工件

# 掌握 ML 管道的基础知识

在我们开始实现 ML 管道之前，让我们先掌握基础知识。我们将反思 ML 管道，并为 ML 管道实现设置所需资源，然后我们将开始数据摄取。让我们通过反思 *第一章* 中 *MLOps 工作流程基础* 的 *图 14* 中讨论的 ML 管道来揭开 ML 管道的神秘面纱。

![图 4.1 – 机器学习管道](img/B16572_04_001.jpg)

图 4.1 – 机器学习管道

如 *图 4.1* 所示，一个全面的 ML 管道包括以下步骤：

1.  数据摄取

1.  模型训练

1.  模型测试

1.  模型打包

1.  模型注册

为了获得不同的视角，我们将同时使用基于云的 Azure ML 服务和开源的 MLflow 来实现管道的所有这些步骤。Azure ML 和 MLflow 是 MLOps 的强大组合：它们展示了 *表 4.1* 中显示的功能。从以下表格中我们可以看到，它们在功能上也是独特的。

![表 4.2 – MLflow 与 Azure ML 服务比较](img/01.jpg)![表 4.2 – MLflow 与 Azure ML 服务比较](img/02.jpg)

表 4.2 – MLflow 与 Azure ML 服务比较

为了实现机器学习管道，我们需要为我们的数据集提供存储资源，并为我们的机器学习模型提供计算资源。如前所述，在*第二章*“描述您的机器学习问题”中，我们将执行实现机器学习管道和业务问题的计算，如图*图 4.2*所示。

我们在本地计算机或 PC 上处理数据以开始，并预处理数据以进行机器学习训练。对于机器学习训练和管道实现，我们使用云上配置的计算资源（Microsoft Azure）。尽管管道的机器学习训练可以在您的本地计算机上进行，但我们将使用云上的计算资源来学习如何配置和使用所需的计算资源以实现机器学习管道。

![图 4.3 – 数据和机器学习任务的计算位置

![img/B16572_04_002.jpg]

图 4.3 – 数据和机器学习任务的计算位置

不再拖延，让我们按照以下步骤配置所需的机器学习管道计算资源：

1.  前往您的机器学习工作空间。![图 4.4 – Azure 机器学习工作空间

    ]

    图 4.4 – Azure 机器学习工作空间

1.  前往**计算**选项，点击**创建**按钮以探索云上可用的计算选项。

1.  选择适合机器学习模型训练的最佳和高效的计算选项。

    根据您的训练需求和成本限制选择合适的计算选项，并为其命名。例如，在*图 4.4*中，为实验`unique_code-ml-compute1`选择了计算或虚拟机。*图 4.4*中选择的计算选项是成本最低的计算选项之一，这足以实现业务问题的机器学习管道。为了更快地实现和训练机器学习模型，建议使用`STANDARD_DS11_V2`（2 核，14 GB RAM）虚拟机大小。使用此选项，训练一个模型将大约需要 12 分钟。

1.  配置之前创建的计算资源。在命名并创建所需的计算资源后，您的计算资源已配置、就绪并运行在云上，如图*图 4.5*所示。

![图 4.6 – AzureML 工作空间中的配置计算

![img/B16572_04_005.jpg]

图 4.6 – AzureML 工作空间中的配置计算

在配置完成后，选择**JupyterLab**选项。JupyterLab 是一个开源的基于 Web 的用户界面。它集成了文本编辑器、代码编辑器、终端和可扩展方式集成的自定义组件等功能。我们将使用它作为连接到配置的计算资源以训练机器学习模型的编程接口。

现在，我们将开始实际实现机器学习管道。按照以下步骤实现机器学习管道：

1.  要开始实施，克隆您已导入到 Azure DevOps 项目的仓库。要克隆仓库，请从 **仓库** 菜单的右上角点击 **克隆** 按钮，然后点击 **生成 Git 凭据** 按钮。将创建一个哈希密码。![图 4.7 – 克隆 Azure DevOps Git 仓库（生成 Git 凭据）    ](img/B16572_04_006.jpg)

    图 4.7 – 克隆 Azure DevOps Git 仓库（生成 Git 凭据）

1.  从 **命令行** 部分复制 HTTPS 链接以获取 Azure DevOps 仓库链接，如下所示：

    ```py
    https://xxxxxxxxx@dev.azure.com/xxxxx/Learn_MLOps/_git/Learn_MLOps
    ```

1.  复制从 *步骤 1* 生成的密码，并将其添加到 *步骤 2* 中的链接，方法是在 `@` 字符之前用 `:` 分隔的第一个用户名之后添加密码。然后就可以使用以下 `git clone` 命令而不会出现权限错误：

    ```py
    git clone https://user:password_hash@dev.azure.com/user/repo_created
    ```

1.  一旦启动了 JupyterLab，我们将访问终端以将仓库克隆到 Azure 计算中。要访问终端，您必须从 **启动器** 选项卡中选择 **终端** 选项。另一种直接访问终端的方法是使用 Azure ML 工作区中计算实例列表中的应用程序 URI 列表中的 **终端** 链接。转到 JupyterLab 的 **终端** 选项并执行以下操作（如图 4.7 所示）：

    ```py
    git clone https://xxxxxxxxx@dev.azure.com/xxxxx/Learn_MLOps/_git/Learn_MLOps
    ```

    这是输出：

    ![图 4.8 – 在 Azure 计算上克隆 Azure DevOps Git 仓库    ](img/B16572_04_007.jpg)

    图 4.8 – 在 Azure 计算上克隆 Azure DevOps Git 仓库

1.  进入 `04_MLpipelines` 文件夹，并按照克隆仓库中的 `ML-pipeline.ipynb` 文件中的实施步骤进行操作。所有以下步骤都在 `ML-pipeline.ipynb` 中实现。建议遵循文件说明以更好地理解实施，并按照您的设置在新的文件中自行执行代码。

    到目前为止，我们已经配置了计算资源并在计算中克隆了 GitHub 仓库。

1.  接下来，我们通过导入所需的库开始实施 `ML-pipeline.ipynb` 文件，例如 `pandas`、`numpy`、`azureml`、`pickle`、`mlflow` 等，如下面的代码块所示：

    ```py
    import pandas as pd
    import numpy as np
    import warnings
    from math import sqrt
    warnings.filterwarnings('ignore')
    from azureml.core.run import Run
    from azureml.core.experiment import Experiment
    from azureml.core.workspace import Workspace
    from azureml.core.model import Model
    from azureml.core.authentication import ServicePrincipalAuthentication
    from azureml.train.automl import AutoMLConfig
    import pickle
    from matplotlib import pyplot as plt
    from matplotlib.pyplot import figure
    import mlflow
    ```

1.  接下来，我们使用 setup MLflow（用于跟踪实验）。使用 `get_mlflow_tracking_url()` 函数获取 MLflow 实验和工件应记录的位置的跟踪 ID（在这种情况下，我们获取已配置的训练计算的跟踪 ID）。然后使用 `set_tracking_uri()` 函数连接到已配置训练计算的跟踪 URI（特定资源的统一资源标识符）。跟踪 URI 可以是远程服务器、数据库连接字符串或本地目录中记录数据的本地路径。在我们的情况下，我们默认将跟踪 URI 指向本地路径（在已配置的训练计算上）：

    ```py
    uri = workspace.mlruns folder where MLflow artifacts and logs will be saved for experiments. 
    ```

通过设置 MLflow 实验的跟踪 URI，您已为 MLflow 设置保存其工件和日志的`mlruns`文件夹（在您配置的计算上）的位置。在执行这些命令后，检查当前路径。您将找到`mlruns`文件夹。

# 数据摄入和特征工程

数据对于训练机器学习模型至关重要；没有数据，就没有机器学习。数据摄入是机器学习管道的触发步骤。它通过从各种数据源提取数据并摄入模型训练所需的数据来处理数据的量、速度、真实性和多样性。

机器学习管道通过摄入用于训练机器学习模型的正确数据来启动。我们将从上一章中注册的预处理数据开始。按照以下步骤访问和导入预处理数据，使其为机器学习训练做好准备：

1.  使用 Azure ML SDK 中的`Workspace()`函数，如下访问机器学习工作区中的数据存储：

    ```py
    from azureml.core import Workspace, Dataset
    subscription_id = 'xxxxxx-xxxxxx-xxxxxxx-xxxxxxx'
    resource_group = 'Learn_MLOps'
    workspace_name = 'MLOps_WS'
    workspace = subscription_id, resource_group, and workspace_name and initiate a workspace object using these credentials. When these instructions are successfully executed in the JupyterLab, you can run the remaining blocks of code in the next cells.
    ```

1.  导入上一章中准备好的预处理数据集。使用 Azureml SDK 中的`Dataset`函数的`.get_by_name()`函数导入预处理数据集，并使用该函数检索所需的数据集：

    ```py
    # Importing pre-processed dataset
    dataset = Dataset.get_by_name (workspace, name='processed_weather_data_portofTurku')
    print(dataset.name, dataset.version)
    ```

1.  在成功检索或挂载数据集后，可以通过打印`dataset.name`和`dataset.version`来确认，应打印`processed_weather_data_portofTurku 1`或您之前给数据集命名的名称。

1.  在检索到预处理数据后，将其分为训练集和验证集至关重要，以便在训练阶段和后续阶段训练机器学习模型并进行测试或评估。因此，我们将其分为 80%（训练集）和 20%（测试集）的分割比，如下所示：

    ```py
    df_training = df.iloc[:77160]
    df_test = df.drop(df_training.index)
    df_training.to_csv('Data/training_data.csv',index=False)
    df_test.to_csv('Data/test_data.csv',index=False)
    ```

1.  在成功分割数据后，这两个数据集被存储并注册到数据存储（连接到 Azure ML 工作区）中，如下所示：

    ```py
    datastore = workspace.get_default_datastore()
    datastore.upload(src_dir='Data', target_path='data')
    training_dataset = /
    Dataset.Tabular.from_delimited_files(datastore.path('data/training_data.csv'))
    validation_dataset = /
    Dataset.Tabular.from_delimited_files(datastore.path('data/validation_data.csv'))
    training_ds = training_dataset.register(workspace=workspace, name='training_dataset',
    description='Dataset to use for ML training')
    test_ds = validation_dataset.register(workspace=workspace,
                                     name='test_dataset',
    description='Dataset for validation ML models')
    ```

通过使用`register()`函数，我们能够注册训练集和测试集，这些数据集可以在以后从数据存储中导入。

接下来，我们将导入训练数据并将其摄入到机器学习管道中，并使用测试数据集来测试模型在生产中对未见数据的性能或用于模型分析。

## 数据摄入（训练数据集）

要将训练数据摄入到机器学习管道中，我们首先使用`get_by_name()`函数导入它，然后使用`to_pandas_dataframe()`函数将其转换为 pandas 数据框：

```py
dataset = Dataset.get_by_name (workspace, name='training_dataset')
print(dataset.name, dataset.version)
df = dataset.to_pandas_dataframe ( )
```

现在已检索到训练数据集，并将用于进一步训练机器学习模型。目标是训练分类模型以预测是否会下雨。因此，选择`Temperature`（温度）、`Humidity`（湿度）、`Wind_speed`（风速）、`Wind_bearing`（风向）、`Visibility`（能见度）、`Pressure`（气压）和`Current_weather_conditions`（当前天气状况）特征来训练二元分类模型，以预测未来的天气状况（4 小时后）。

按照以下步骤选择特征并进行缩放：

1.  在训练机器学习模型之前，选择正确的特征和缩放数据是至关重要的。因此，我们按照以下方式选择特征。变量`X`中的值代表自变量，而变量`Y`是因变量（预测的天气）：

    ```py
    X = df[['Temperature_C', 'Humidity', 'Wind_speed_kmph', 'Wind_bearing_degrees', 'Visibility_km', 'Pressure_millibars', 'Current_weather_condition']].values
    y = df['Future_weather_condition'].values
    ```

1.  使用`sklearn`中的`train_test_split()`函数将训练数据分割成训练集和测试集（用于训练后的验证）。为了通过保持与先前实验相同配置的样本来重现训练会话，需要固定随机种子（`random_state`）。因此，我们将使用`random_state=1`：

    ```py
    # Splitting the Training dataset into Train and Test set for ML training
    from sklearn.model_selection import train_test_split
    X_train, X_val,  y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    ```

    在 80%（训练数据）和 20%（测试数据）的分割下，训练集和测试集现在已准备好进行特征缩放和机器学习模型训练。

1.  为了使机器学习模型训练达到最佳和高效，数据需要在同一尺度上。因此，我们使用`sklearn`中的`StandardScaler()`对数据进行缩放，以校准数据中的所有数值到同一尺度：

    ```py
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    ```

    在这一步中，使用`StandardScaler`对训练数据的数值进行缩放，并将所有值转换到基于`X_train`值的`-1`到`1`的范围内。现在我们已准备好训练机器学习模型（有趣的部分）！

# 机器学习训练和超参数优化

我们已经准备好进行有趣的部分，即训练机器学习模型！这一步使模型训练成为可能；它包含模块化脚本或代码，执行所有传统的机器学习训练步骤，例如拟合和转换数据以训练模型，以及超参数调整以收敛到最佳模型。这一步骤的输出是一个训练好的机器学习模型。

为了解决业务问题，我们将使用**支持向量机**分类器和**随机森林**分类器训练两个著名的模型。这些模型的选择基于它们的流行度和结果的一致性；你可以自由选择你喜欢的模型——在这个步骤中没有限制。首先，我们将训练支持向量机分类器，然后是随机森林分类器。

## 支持向量机

**支持向量机（SVM**）是一种流行的监督学习算法（用于分类和回归）。数据点在 N 维空间中使用超平面进行分类。它以较少的计算能力产生显著的准确性而闻名。建议在理论上了解 SVM，以便更好地理解实践中的模型训练。了解更多关于 SVM 的信息，请访问：[`www.kdnuggets.com/2017/02/yhat-support-vector-machine.html`](https://www.kdnuggets.com/2017/02/yhat-support-vector-machine.html)。

让我们开始训练 SVM 分类器：

1.  我们首先使用 Azure SDK 中的`Experiment()`函数启动训练或实验。此函数的目的是在 Azure ML 工作区中启动一个训练运行或实验，以便监控和记录模型训练性能：

    ```py
    myexperiment = Experiment(workspace, "support-vector-machine")
    ```

1.  同样，MLflow 实验也被启动，以观察不同的视角：

    ```py
    mlflow.set_experiment("mlflow-support-vector-machine")
    ```

    现在，我们在 Azure ML 工作区和 MLflow 中启动了实验。以下训练步骤将被监控和记录。

1.  接下来，我们进行超参数调整，以找到收敛最佳模型的最佳参数。这可以手动完成，但更高效和自动的解决方案，如 Grid Search 或 Random Search 存在。对于训练，SVM 分类器使用以下 Grid Search。我们通过使用`SVC()`和`GridSearchCV()`函数从`sklearn`中，并在 Azure ML 和 MLflow 上记录运行来继续操作：

    ```py
    from sklearn.svm import SVC
    from sklearn import svm 
    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.STANDARD_DS11_V2 (2 cores, 14 GB RAM) compute machine). The result or the output of the Grid Search suggests the best performing parameters to be C=1 and the kernel as rbf. Using run.log(), we have logged the dataset used to train the model (the training set) and keep track of the experiment. This data is logged to the Azure ML workspace and the MLflow experiments. 
    ```

1.  最后，使用最佳参数，以下是用`C=1`和`kernel='rbf '`训练新模型的步骤：

    ```py
    svc = SVC(C=svc_grid.get_params(deep=True)['estimator__C'], kernel=svc_grid.get_params(deep=True)['estimator__kernel'])
    svc.fit(X_train, y_train)
    # Logging training parameters to AzureML and MLFlow experiments
    run.log("C", svc_grid.get_params(deep=True)['estimator__C'])
    run.log("Kernel", svc_grid.get_params(deep=True)['estimator__kernel'])
    After training the SVC classifier, the following output is shown:
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
    ```

    通过这种方式，我们已经训练了 SVM 模型！现在我们将训练随机森林分类器模型。

## 随机森林分类器

随机森林是另一种流行的监督学习模型（用于分类和回归）。随机森林是一种集成学习方法，它使用多个决策树。在执行模型训练之前，建议了解随机森林模型的理论工作原理。要了解更多关于随机森林模型的信息，请访问[`www.kdnuggets.com/2020/01/random-forest-powerful-ensemble-learning-algorithm.html`](https://www.kdnuggets.com/2020/01/random-forest-powerful-ensemble-learning-algorithm.html)。

1.  要开始训练随机森林分类器，按照以下方式在 Azure ML 工作区和 MLflow 实验中初始化实验：

    ```py
    myexperiment = Experiment(workspace, "support-vector-machine")
    mlflow.set_experiment("mlflow-support-vector-machine")
    ```

1.  实验成功启动后，可以通过从`sklearn.ensemble`导入`RandomForestClassifier()`函数并使用所需参数来启动训练，如下所示。这些参数是随机选择的（没有进行`Grid Search`）。可以使用`Grid Search`或`RandomizedSearch`来确定最佳参数并优化算法：

    ```py
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier (max_depth=10, random_state=0, n_estimators=100)
    ```

1.  模型训练是通过传递训练数据到`fit(X_train, y_train)`函数来完成的。训练数据集和参数如下记录到 Azure ML 和 MLflow 实验中：

    ```py
    # initialize runs in Azureml and mlflow
    run = myexperiment.start_logging()
    mlflow.start_run()
    # Log dataset used 
    run.log("dataset name", dataset.name)
    run.log("dataset Version", dataset.version)
    rf.fit(X_train, y_train)
    # Logging training parameters to AzureML and MLFlow experiments
    run.log("max_depth", 10)
    run.log("random_state", 0)
    run.log("n_estimators", 100)
    ```

1.  训练后，输出如下所示：

    ```py
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    max_depth=10, max_features='auto', max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
    oob_score=False, random_state=0, verbose=0, warm_start=False)
    ```

    这是完成随机森林模型训练后的预期结果。通过这种方式，你已经成功完成了随机森林模型的训练，总共训练了两个机器学习模型：SVM 分类器和随机森林分类器。

    训练后，从准确度和其他指标方面测试模型性能至关重要，以了解模型是否足够适合生产或测试环境。

接下来，我们将测试在训练模型之前分割的测试数据的训练模型的性能。

# 模型测试和定义指标

在这一步，我们将评估训练好的模型在单独的数据点集上的性能，这些数据点被称为测试数据（在数据摄入步骤中提前分割和版本化）。根据用例选择的指标评估训练模型的推断。这一步的输出是关于训练模型性能的报告。

为了全面分析模型性能，我们将测量准确率、精确率、召回率和 F 分数。这在业务问题背景下的实际含义如下：

+   **准确率**：正确预测的数据测试样本总数与预测总数之比。

+   **精确率**：精确率衡量的是正确预测为正例的正例的比例。*精确率 = 真正例 / (真正例 + 假正例)*

+   **召回率**：召回率衡量的是正确识别的实际正例的比例。*召回率 = 真正例 / (真正例 + 假负例)*

+   **F 分数**：在计算 F 分数时，既考虑了精确率也考虑了召回率。它是精确率和召回率的调和平均（平均值）。*F1 Score = 2*(Recall * Precision) / (Recall + Precision)*。

我们将在验证数据集上测量这些指标。让我们看看 SVM 分类器和随机森林分类器的结果。

## 测试 SVM 分类器

使用 `sklearn.metrics`，我们计算模型在测试数据样本上的性能指标 `accuracy`、`f1_score`、`precision` 和 `recall`，并使用 `run.log()` 函数将它们记录到 Azure ML 工作区和 MLflow 实验，如下所示。

从 `sklearn.metrics` 中导入 `accuracy_score`、`f1_score`、`precision_score` 和 `recall_score`：

```py
predicted_svc = svc.predict(X_test)
acc = accuracy_score(y_test, predicted_svc)
fscore = f1_score(y_test, predicted_svc, average="macro")
precision = precision_score(y_test, predicted_svc, average="macro")
recall = recall_score(y_test, predicted_svc, average="macro")
run.log("Test_accuracy", acc)
run.log("Precision", precision)
run.log("Recall", recall)
run.log("F-Score", fscore)
run.log("Git-sha", sha)
```

根据实验，测试数据指标的输出已记录在 Azure ML 工作区。您可以在注册模型（我们将在 *注册模型和生产工件* 中注册模型）后稍后读取这些日志。

## 测试随机森林分类器

与我们为 SVM 分类器模型所做的工作类似，使用 `sklearn.metrics` 我们计算 `accuracy`、`f1_score`、`precision` 和 `recall`：

```py
acc = accuracy_score(y_test, predicted_rf)
fscore = f1_score(y_test, predicted_rf, average="macro")
precision = precision_score(y_test, predicted_rf, average="macro")
recall = recall_score(y_test, predicted_rf, average="macro")
run.log("Test_accuracy", acc)
run.log("Precision", precision)
run.log("Recall", recall)
run.log("F-Score", fscore)
run.log("Git-sha", sha)
```

使用 `run.log()` 函数，将模型在测试数据样本上的性能指标输出记录到 Azure ML 工作区和 MLflow 实验。

# 模型打包

在上一步测试了训练好的模型之后，可以将模型序列化为文件以导出到测试或生产环境。序列化文件可能会遇到兼容性挑战，例如模型互操作性，如果操作不当。模型互操作性是一个挑战，尤其是在使用不同框架训练模型时。例如，如果模型 1 使用 `sklearn` 训练，而模型 2 使用 TensorFlow 训练，那么模型 1 就不能使用 TensorFlow 导入或导出以进行进一步的模型微调或模型推理。

为了避免这个问题，ONNX 提供了一个模型互操作性的开放标准。ONNX 代表 Open Neural Network Exchange。它为导入和导出模型提供了一个序列化标准。我们将使用 ONNX 格式序列化模型，以避免兼容性和互操作性问题的出现。

使用 ONNX，通过 `skl2onnx` 库将训练好的模型序列化。模型以 `svc.onnx` 文件的形式序列化，以便进一步将模型导出和导入到测试和生产环境：

```py
# Convert into SVC model into ONNX format file
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([None, 6]))]
onx = convert_sklearn(svc, initial_types=initial_type)
with open("outputs/svc.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

此代码的输出是一个序列化的 `svc.onnx` 文件。同样，使用 ONNX，我们将随机森林模型转换为名为 `rf.onnx` 的序列化文件，以便进一步将模型导出和导入到测试和生产环境：

```py
# Convert into RF model into ONNX format file
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [('float_input', FloatTensorType([None, 6]))]
onx = convert_sklearn(rf, initial_types=initial_type)
with open("outputs/rf.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

此代码的输出是一个序列化的 `rf.onnx` 文件。接下来，我们将这些序列化模型注册到模型注册表中。

# 注册模型和生产工件

在这一步，之前步骤中已序列化或容器化的模型被注册并存储在模型注册表中。一个注册的模型被编译为一个逻辑容器，用于一个或多个作为模型功能使用的文件。例如，由多个文件组成的模型可以注册为模型注册表中的单个模型。通过下载注册的模型，可以接收所有文件。注册的模型可以按需部署和使用进行推理。

让我们使用 Azure ML SDK 中的 `model.register()` 函数，将上一节中序列化的模型注册。通过使用此函数，序列化的 ONNX 文件被注册到工作区，以便进一步使用和部署到测试和生产环境。现在，让我们注册序列化的 SVM 分类器模型（`svc.onnx`）：

```py
# Register Model on AzureML WS
model = Model.register (model_path = './outputs/svc.onnx', # this points to a local file 
                       model_name = "support-vector-classifier", 
                       tags = {'dataset': dataset.name, 'version': dataset.version, 'hyparameter-C': '1', 'testdata-accuracy': '0.9519'}, 
                       model_framework='pandas==0.23.4',
                       description = "Support vector classifier to predict weather at port of Turku",
                       workspace = workspace)
print('Name:', model.name)
print('Version:', model.version)
```

模型通过命名和标记模型以符合需求进行注册。我们可以通过检查注册的模型名称和版本来确认模型注册成功。输出将反映您在注册时使用的模型名称（例如，`support-vector-classifier`）并显示模型版本为 `1`。同样，让我们注册序列化的随机森林分类器模型（`rf.onnx`）：

```py
# Register Model on AzureML WS
model = Model.register (model_path = './outputs/rf.onnx', # this points to a local file 
                       model_name = "random-forest-classifier", 
                       tags = {'dataset': dataset.name, 'version': dataset.version, 'hyparameter-C': '1', 'testdata-accuracy': '0.9548'}, 
                       model_framework='pandas==0.23.4',
                       description = "Random forest classifier to predict weather at port of Turku",
                       workspace = workspace)
print('Name:', model.name)
print('Version:', model.version)
```

模型注册成功后，`print` 函数的输出将反映您在注册时使用的模型名称（例如，`random-forest-classifier`）并显示模型版本为 `1`。最后，我们将注册用于推理的生产工件。现在，您可以在 Azure ML 工作区的 **模型** 部分中看到这两个模型，如图 4.8 所示：

![图 4.9 – 已注册的 SVM 模型（含测试指标）](img/B16572_04_008.jpg)

图 4.9 – 已注册的 SVM 模型（含测试指标）

这样，您可以可视化并分析在 Azure ML 工作区中训练和测试的每个模型的训练和测试日志。它提供了对模型训练和测试的鸟瞰图，同时为注册的模型提供了可追溯性。

## 注册生产工件

为了在实时模型推理中缩放输入数据，需要一个标量来将数据缩放到与 ML 训练时数据缩放相同的尺度。我们将使用用于`scaling X_train`的相同缩放函数，并将这个变量序列化到一个`pickle`文件中。最后，我们将这个`pickle`文件注册到工作区，以便后续检索和使用（特别是对于测试和生产环境中的模型推理）。使用`pickle`，可以通过以下方式使用`pickle.dump()`函数将缩放变量`sc`写入一个`pickle`文件。

使用`import pickle as open('./outputs/scaler.pkl', 'wb') as scaler_pkl`导入`pickle`：

```py
    pickle.dump(sc, scaler_pkl)
```

代码的输出将保存一个名为`scaler.pkl`的序列化`pickle`文件，用于缩放器。接下来，我们将此文件注册到模型注册表中，以便以后与我们的模型一起下载和部署以进行推理。缩放器是通过以下`model.register()`函数注册的：

```py
# Register Model on AzureML WS
scaler = Model.register(model_path = './outputs/scaler.pkl', # this points to a local file 
                       model_name = "scaler", # this is the name the model is registered as
                       tags = {'dataset': dataset.name, 'version': dataset.version}, 
                       model_framework='pandas==0.23.4',
                       description = "Scaler used for scaling incoming inference data",
                       workspace = workspace)
print('Name:', scaler.name)
print('Version:', scaler.version)
```

在保存和注册缩放对象后，可以在 Azure ML 工作区中找到已注册的对象。同样，已注册的模型也可以跟踪，如图*图 4.8*所示：

![图 4.10 – 已注册的模型![img/B16572_04_009.jpg](img/B16572_04_009.jpg)

图 4.10 – 已注册的模型

恭喜！SVM 分类器和随机森林分类器，以及序列化的缩放器，都已注册到模型注册表中。这些模型可以在以后下载和部署。这标志着 ML 管道的成功实施！

# 摘要

在本章中，我们介绍了 ML 管道的理论，并通过构建用于商业问题的 ML 管道来实践。我们为训练这些 ML 模型设置了工具、资源和开发环境。我们从数据摄取步骤开始，接着是模型训练步骤、测试步骤、打包步骤，最后完成了注册步骤。恭喜！到目前为止，你已经实现了 MLOps 工作流程的关键构建块。

在下一章中，我们将探讨评估和打包生产模型。
