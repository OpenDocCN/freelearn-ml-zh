# 第八章：使用你的算法和代码

在上一章中，你学习了如何使用内置框架（如 **scikit-learn** 和 **TensorFlow**）进行模型的训练和部署。通过 **脚本模式**，这些框架使你能够轻松使用自己的代码，而无需管理任何训练或推理容器。

在某些情况下，你的业务或技术环境可能使得使用这些容器变得困难，甚至不可能使用。也许你需要完全控制容器的构建方式，也许你希望实现自己的预测逻辑，或者你正在使用 SageMaker 本身不原生支持的框架或语言。

在本章中，你将学习如何根据自己的需求定制训练和推理容器。你还将学习如何使用 SageMaker SDK 或命令行开源工具来训练和部署你自己的自定义代码。

本章将涵盖以下主题：

+   理解 SageMaker 如何调用你的代码

+   定制内置框架容器

+   使用 SageMaker 训练工具包构建自定义训练容器

+   使用 Python 和 R 构建完全自定义的训练和推理容器

+   使用自定义 Python 代码在 MLflow 上进行训练和部署

+   为 SageMaker 处理构建完全自定义容器

# 技术要求

你将需要一个 AWS 账户才能运行本章中的示例。如果你还没有 AWS 账户，请访问[`aws.amazon.com/getting-started/`](https://aws.amazon.com/getting-started/)创建一个。你还应该熟悉 AWS 免费套餐（[`aws.amazon.com/free/`](https://aws.amazon.com/free/)），它允许你在一定的使用限制内免费使用许多 AWS 服务。

你需要为你的账户安装并配置 AWS **命令行界面**（**CLI**）（[`aws.amazon.com/cli/`](https://aws.amazon.com/cli/)）。

你需要一个正常工作的 Python 3.x 环境。安装 Anaconda 发行版（[`www.anaconda.com/`](https://www.anaconda.com/)）不是强制性的，但强烈推荐，因为它包含了我们需要的许多项目（Jupyter、`pandas`、`numpy` 等）。

你将需要一个正常工作的 Docker 安装环境。你可以在[`docs.docker.com`](https://docs.docker.com)找到安装说明和文档。

本书中包含的代码示例可以在 GitHub 上找到：[`github.com/PacktPublishing/Learn-Amazon-SageMaker-second-edition`](https://github.com/PacktPublishing/Learn-Amazon-SageMaker-second-edition)。你需要安装 Git 客户端来访问这些示例（[`git-scm.com/`](https://git-scm.com/)）。

# 理解 SageMaker 如何调用你的代码

当我们使用内置算法和框架时，并没有太多关注 SageMaker 实际上是如何调用训练和部署代码的。毕竟，“内置”意味着：直接拿来需要的工具并开始工作。

当然，如果我们想使用自定义代码和容器，情况就不一样了。我们需要了解它们如何与 SageMaker 接口，以便我们能够准确地实现它们。

在本节中，我们将详细讨论这一接口。让我们从文件布局开始。

### 理解 SageMaker 容器内的文件布局

为了简化我们的工作，SageMaker 估算器会自动将超参数和输入数据复制到训练容器中。同样，它们会自动将训练好的模型（以及任何检查点）从容器复制到 S3。在部署时，它们会执行反向操作，将模型从 S3 复制到容器中。

如你所想，这要求遵循文件布局约定：

+   超参数以 JSON 字典形式存储在`/opt/ml/input/config/hyperparameters.json`中。

+   输入通道存储在`/opt/ml/input/data/CHANNEL_NAME`中。我们在前一章中看到，通道名称与传递给`fit()` API 的名称匹配。

+   模型应保存在`/opt/ml/model`中，并从该位置加载。

因此，我们需要在自定义代码中使用这些路径。现在，让我们看看如何调用训练和部署代码。

### 理解自定义训练选项

在*第七章*，*使用内置框架扩展机器学习服务*中，我们研究了脚本模式以及 SageMaker 如何使用它来调用我们的训练脚本。此功能由框架容器中额外的 Python 代码启用，即 SageMaker 训练工具包（[`github.com/aws/sagemaker-training-toolkit`](https://github.com/aws/sagemaker-training-toolkit)）。

简而言之，训练工具包将入口脚本、其超参数和依赖项复制到容器内。它还将从输入通道中复制数据到容器中。然后，它会调用入口脚本。有好奇心的朋友可以阅读`src/sagemaker_training/entry_point.py`中的代码。

在自定义训练代码时，你有以下选项：

+   自定义现有的框架容器，只添加你额外的依赖和代码。脚本模式和框架估算器将可用。

+   基于 SageMaker 训练工具包构建自定义容器。脚本模式和通用的`Estimator`模块将可用，但你需要安装其他所有依赖。

+   构建一个完全自定义的容器。如果你想从头开始，或者不想在容器中添加任何额外的代码，这是最合适的选择。你将使用通用的`Estimator`模块进行训练，并且脚本模式将不可用。你的训练代码将直接调用（稍后会详细说明）。

### 理解自定义部署选项

框架容器包括用于部署的额外 Python 代码。以下是最流行框架的仓库：

+   **TensorFlow**: [`github.com/aws/sagemaker-tensorflow-serving-container`](https://github.com/aws/sagemaker-tensorflow-serving-container)。模型通过**TensorFlow Serving**提供服务 ([`www.tensorflow.org/tfx/guide/serving`](https://www.tensorflow.org/tfx/guide/serving))。

+   **PyTorch**: [`github.com/aws/sagemaker-pytorch-inference-toolkit`](https://github.com/aws/sagemaker-pytorch-inference-toolkit)。模型通过**TorchServe**提供服务 ([`pytorch.org/serve`](https://pytorch.org/serve))。

+   **Apache MXNet**: [`github.com/aws/sagemaker-mxnet-inference-toolkit`](https://github.com/aws/sagemaker-mxnet-inference-toolkit)。模型通过**多模型服务器**提供服务 ([`github.com/awslabs/multi-model-server`](https://github.com/awslabs/multi-model-server))，并集成到**SageMaker 推理工具包**中 ([`github.com/aws/sagemaker-inference-toolkit`](https://github.com/aws/sagemaker-inference-toolkit))。

+   **Scikit-learn**: [`github.com/aws/sagemaker-scikit-learn-container`](https://github.com/aws/sagemaker-scikit-learn-container)。模型通过多模型服务器提供服务。

+   **XGBoost**: [`github.com/aws/sagemaker-xgboost-container`](https://github.com/aws/sagemaker-xgboost-container)。模型通过多模型服务器提供服务。

就像训练一样，你有三个选项：

+   自定义现有的框架容器。模型将使用现有的推理逻辑提供服务。

+   基于 SageMaker 推理工具包构建自定义容器。模型将由多模型服务器提供服务。

+   构建一个完全自定义的容器，去掉任何推理逻辑，改为实现自己的推理逻辑。

无论是使用单一容器进行训练和部署，还是使用两个不同的容器，都取决于你。许多不同的因素会影响决策：谁构建容器、谁运行容器等等。只有你能决定对你的特定设置来说，哪个选项是最佳的。

现在，让我们运行一些示例吧！

# 自定义现有框架容器

当然，我们也可以简单地写一个 Dockerfile，引用其中一个深度学习容器镜像 ([`github.com/aws/deep-learning-containers/blob/master/available_images.md`](https://github.com/aws/deep-learning-containers/blob/master/available_images.md))，并添加我们自己的命令。请参见以下示例：

```py
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-cpu-py37-ubuntu18.04
. . .
```

相反，让我们在本地机器上自定义并重新构建**PyTorch**训练和推理容器。这个过程与其他框架类似。

构建环境

需要安装并运行 Docker。为了避免在拉取基础镜像时受到限制，建议你创建一个`docker login`或**Docker Desktop**。

为了避免奇怪的依赖问题（我在看你，macOS），我还建议你在`m5.large`实例上构建镜像（应该足够），但请确保分配的存储空间超过默认的 8GB。我推荐 64GB。你还需要确保该 EC2 实例的**IAM**角色允许你推送和拉取 EC2 镜像。如果你不确定如何创建并连接到 EC2 实例，可以参考这个教程：[`docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html`](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html)。

## 在 EC2 上设置构建环境

我们将通过以下步骤开始：

1.  一旦你的 EC2 实例启动，我们通过`ssh`连接到它。首先，我们安装 Docker，并将`ec2-user`添加到`docker`组。这将允许我们以非 root 用户身份运行 Docker 命令：

    ```py
    $ sudo yum -y install docker
    $ sudo usermod -a -G docker ec2-user
    ```

1.  为了应用此权限更改，我们登出并重新登录。

1.  我们确保`docker`正在运行，并登录到 Docker Hub：

    ```py
    $ service docker start
    $ docker login
    ```

1.  我们安装`git`、Python 3 和`pip`：

    ```py
    $ sudo yum -y install git python3-devel python3-pip
    ```

我们的 EC2 实例现在已经准备好，我们可以继续构建容器。

## 构建训练和推理容器

这可以通过以下步骤完成：

1.  我们克隆`deep-learning-containers`仓库，该仓库集中管理 TensorFlow、PyTorch、Apache MXNet 和 Hugging Face 的所有训练和推理代码，并添加了便捷的脚本来构建这些容器：

    ```py
    $ git clone https://github.com/aws/deep-learning-containers.git
    $ cd deep-learning-containers
    ```

1.  我们为我们的账户 ID、运行所在的区域以及我们将在 Amazon ECR 中创建的新仓库的名称设置环境变量：

    ```py
    $ export ACCOUNT_ID=123456789012
    $ export REGION=eu-west-1
    $ export REPOSITORY_NAME=my-pt-dlc
    ```

1.  我们在 Amazon ECR 中创建仓库并登录。有关详细信息，请参阅文档（[`docs.aws.amazon.com/ecr/index.html`](https://docs.aws.amazon.com/ecr/index.html)）：

    ```py
    $ aws ecr create-repository 
    --repository-name $REPOSITORY_NAME --region $REGION
    $ aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
    ```

1.  我们创建一个虚拟环境，并安装 Python 依赖：

    ```py
    $ python3 -m venv dlc
    $ source dlc/bin/activate
    $ pip install -r src/requirements.txt
    ```

1.  在这里，我们想要为 PyTorch 1.8 构建训练和推理容器，支持 CPU 和 GPU。我们可以在`pytorch/training/docker/1.8/py3/`找到相应的 Docker 文件，并根据需要进行定制。例如，我们可以将 Deep Graph Library 固定为版本 0.6.1：

    ```py
    && conda install -c dglteam -y dgl==0.6.1 \
    ```

1.  编辑完 Docker 文件后，我们查看最新 PyTorch 版本的构建配置文件（`pytorch/buildspec.yml`）。我们决定自定义镜像标签，以确保每个镜像都能清楚地识别：

    ```py
    BuildCPUPTTrainPy3DockerImage:
        tag: !join [ *VERSION, "-", *DEVICE_TYPE, "-", *TAG_PYTHON_VERSION, "-", *OS_VERSION, "-training" ]
    BuildGPUPTTrainPy3DockerImage:
        tag: !join [ *VERSION, "-", *DEVICE_TYPE, "-", *TAG_PYTHON_VERSION, "-", *CUDA_VERSION, "-", *OS_VERSION, "-training" ]
    BuildCPUPTInferencePy3DockerImage:
        tag: !join [ *VERSION, "-", *DEVICE_TYPE, "-", *TAG_PYTHON_VERSION, "-", *OS_VERSION, "-inference" ]
    BuildGPUPTInferencePy3DockerImage:
        tag: !join [ *VERSION, "-", *DEVICE_TYPE, "-", *TAG_PYTHON_VERSION, "-", *CUDA_VERSION, "-", *OS_VERSION, "-inference"]
    ```

1.  最后，我们运行设置脚本并启动构建过程：

    ```py
    $ bash src/setup.sh pytorch
    $ python src/main.py --buildspec pytorch/buildspec.yml --framework pytorch --device_types cpu,gpu --image_types training,inference
    ```

1.  稍等片刻，所有四个镜像（加上一个示例镜像）都已构建完成，我们可以在本地 Docker 中看到它们：

    ```py
    $ docker images
    123456789012.dkr.ecr.eu-west-1.amazonaws.com/my-pt-dlc   1.8.1-gpu-py36-cu111-ubuntu18.04-example-2021-05-28-10-14-15     
    123456789012.dkr.ecr.eu-west-1.amazonaws.com/my-pt-dlc   1.8.1-gpu-py36-cu111-ubuntu18.04-training-2021-05-28-10-14-15    
    123456789012.dkr.ecr.eu-west-1.amazonaws.com/my-pt-dlc   1.8.1-gpu-py36-cu111-ubuntu18.04-inference-2021-05-28-10-14-15
    123456789012.dkr.ecr.eu-west-1.amazonaws.com/my-pt-dlc   1.8.1-cpu-py36-ubuntu18.04-inference-2021-05-28-10-14-15         
    123456789012.dkr.ecr.eu-west-1.amazonaws.com/my-pt-dlc   1.8.1-cpu-py36-ubuntu18.04-training-2021-05-28-10-14-15          
    ```

1.  我们也可以在 ECR 仓库中看到它们，如下图所示：![图 8.1 – 在 ECR 中查看镜像    ](img/B17705_08_1.jpg)

    图 8.1 – 在 ECR 中查看镜像

1.  现在这些镜像可以通过 SageMaker SDK 使用。让我们用新的 CPU 镜像进行训练。我们只需要将其名称传递给`PyTorch`估算器的`image_uri`参数即可。请注意，我们可以去掉`py_version`和`framework_version`：

    ```py
    Estimator = PyTorch(
        image_uri='123456789012.dkr.ecr.eu-west-1.amazonaws.com/my-pt-dlc:1.8.1-cpu-py36-ubuntu18.04-training-2021-05-28-10-14-15',
        role=sagemaker.get_execution_role(),
        entry_point='karate_club_sagemaker.py',
        hyperparameters={'node_count': 34, 'epochs': 30},
        instance_count=1,
        instance_type='ml.m5.large')
    ```

如你所见，定制深度学习容器非常简单。现在，让我们深入一步，仅使用训练工具包进行操作。

# 使用带有 scikit-learn 的 SageMaker 训练工具包

在这个示例中，我们将使用 SageMaker 训练工具包构建一个自定义 Python 容器。我们将使用它在波士顿房价数据集上训练一个 scikit-learn 模型，使用脚本模式和 `SKLearn` 估算器。

我们需要三个构建模块：

+   训练脚本。由于脚本模式将可用，我们可以使用与 *第七章* 中的 scikit-learn 示例完全相同的代码，*使用内置框架扩展机器学习服务*。

+   我们需要一个 Dockerfile 和 Docker 命令来构建自定义容器。

+   我们还需要一个配置为使用我们自定义容器的 `SKLearn` 估算器。

让我们来处理容器：

1.  一个 Dockerfile 可能会变得相当复杂，但这里我们不需要这么做！我们从 Docker Hub 上提供的官方 Python 3.7 镜像开始 ([`hub.docker.com/_/python`](https://hub.docker.com/_/python))。我们安装 scikit-learn、`numpy`、`pandas`、`joblib` 和 SageMaker 训练工具包：

    ```py
    FROM python:3.7
    RUN pip3 install --no-cache scikit-learn numpy pandas joblib sagemaker-training
    ```

1.  我们使用 `docker build` 命令构建镜像，并将其标记为 `sklearn-customer:sklearn`：

    ```py
    $ docker build -t sklearn-custom:sklearn -f Dockerfile .
    ```

    镜像构建完成后，我们可以找到其标识符：

    ```py
    $ docker images
    REPOSITORY          TAG         IMAGE ID   
    sklearn-custom      sklearn     bf412a511471         
    ```

1.  使用 AWS CLI，我们在 Amazon ECR 中创建一个仓库来托管这个镜像，并登录到该仓库：

    ```py
    $ aws ecr create-repository --repository-name sklearn-custom --region eu-west-1
    $ aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.eu-west-1.amazonaws.com/sklearn-custom:latest
    ```

1.  使用镜像标识符，我们用仓库标识符标记镜像：

    ```py
    $ docker tag bf412a511471 123456789012.dkr.ecr.eu-west-1.amazonaws.com/sklearn-custom:sklearn
    ```

1.  我们将镜像推送到仓库：

    ```py
    $ docker push 123456789012.dkr.ecr.eu-west-1.amazonaws.com/sklearn-custom:sklearn
    ```

    现在，镜像已经准备好用于使用 SageMaker 估算器进行训练。

1.  我们定义一个 `SKLearn` 估算器，将 `image_uri` 参数设置为我们刚创建的容器的名称：

    ```py
    sk = SKLearn(
        role=sagemaker.get_execution_role(),
        entry_point='sklearn-boston-housing.py',
        image_name='123456789012.dkr.ecr.eu-west-1.amazonaws.com/sklearn-custom:sklearn',
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=output,
        hyperparameters={
             'normalize': True,
             'test-size': 0.1
        }
    )
    ```

1.  我们设置训练通道的位置，并像往常一样启动训练。在训练日志中，我们看到我们的代码确实以脚本模式被调用：

    ```py
    /usr/local/bin/python -m sklearn-boston-housing 
    --normalize True --test-size 0.1
    ```

如你所见，定制训练容器非常简单。得益于 SageMaker 训练工具包，你可以像使用内置框架容器一样工作。我们这里使用了 scikit-learn，你也可以对其他所有框架做同样的操作。

然而，我们不能将这个容器用于部署，因为它不包含任何模型服务代码。我们应该添加定制代码来启动一个 Web 应用程序，这正是我们在下一个示例中要做的。

# 为 scikit-learn 构建完全自定义的容器

在这个示例中，我们将构建一个完全自定义的容器，里面没有任何 AWS 代码。我们将使用它在波士顿房价数据集上训练一个 scikit-learn 模型，使用通用的 `Estimator` 模块。使用相同的容器，我们将通过 Flask Web 应用程序部署该模型。

我们将按逻辑步骤进行操作，首先处理训练，然后更新代码以处理部署。

## 使用完全自定义容器进行训练

由于我们不能再依赖脚本模式，因此需要修改训练代码。这就是修改后的代码，你很容易就能理解它是怎么回事：

```py
#!/usr/bin/env python
import pandas as pd
import joblib, os, json
if __name__ == '__main__':
    config_dir = '/opt/ml/input/config'
    training_dir = '/opt/ml/input/data/training'
    model_dir = '/opt/ml/model'
    with open(os.path.join(config_dir, 
    'hyperparameters.json')) as f:
        hp = json.load(f)
        normalize = hp['normalize']
        test_size = float(hp['test-size'])
        random_state = int(hp['random-state'])
    filename = os.path.join(training_dir, 'housing.csv')
    data = pd.read_csv(filename)
    # Train model
    . . . 
    joblib.dump(regr, 
                os.path.join(model_dir, 'model.joblib'))
```

使用 SageMaker 容器的标准文件布局，我们从 JSON 文件中读取超参数。然后，我们加载数据集，训练模型，并将其保存在正确的位置。

还有一个非常重要的区别，我们需要深入了解 Docker 来解释它。SageMaker 将以 `docker run <IMAGE_ID> train` 运行训练容器，并将 `train` 参数传递给容器的入口点。

如果你的容器有预定义的入口点，`train` 参数将会传递给它，比如 `/usr/bin/python train`。如果容器没有预定义的入口点，`train` 就是将要执行的实际命令。

为了避免烦人的问题，我建议你的训练代码满足以下要求：

+   命名为 `train`——没有扩展名，只是 `train`。

+   使其可执行。

+   确保它在 `PATH` 值中。

+   脚本的第一行应该定义解释器的路径，例如 `#!/usr/bin/env python`。

这应该能保证无论你的容器是否有预定义的入口点，都能正确调用你的训练代码。

我们将在 Dockerfile 中处理这个问题，从官方的 Python 镜像开始。注意，我们不再安装 SageMaker 训练工具包：

```py
FROM python:3.7
RUN pip3 install --no-cache scikit-learn numpy pandas joblib
COPY sklearn-boston-housing-generic.py /usr/bin/train
RUN chmod 755 /usr/bin/train
```

脚本名称正确。它是可执行的，且 `/usr/bin` 在 `PATH` 中。

我们应该准备好了——让我们创建自定义容器并用它启动训练任务：

1.  我们构建并推送镜像，使用不同的标签：

    ```py
    $ docker build -t sklearn-custom:estimator -f Dockerfile-generic .
    $ docker tag <IMAGE_ID> 123456789012.dkr.ecr.eu-west-1.amazonaws.com/sklearn-custom:estimator
    $ docker push 123456789012.dkr.ecr.eu-west-1.amazonaws.com/sklearn-custom:estimator
    ```

1.  我们更新笔记本代码，使用通用的 `Estimator` 模块：

    ```py
    from sagemaker.estimator import Estimator
    sk = Estimator(
        role=sagemaker.get_execution_role(),
        image_name='123456789012.dkr.ecr.eu-west-1.amazonaws.com/sklearn-custom:estimator',
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=output,
        hyperparameters={
             'normalize': True,
             'test-size': 0.1,
             'random-state': 123
        }
    )
    ```

1.  我们照常进行训练。

现在让我们添加代码来部署这个模型。

## 部署完全自定义容器

Flask 是一个非常流行的 Python Web 框架（[`palletsprojects.com/p/flask`](https://palletsprojects.com/p/flask)）。它简单且文档齐全。我们将用它来构建一个托管在容器中的简单预测 API。

就像我们的训练代码一样，SageMaker 要求将部署脚本复制到容器内。镜像将以 `docker run <IMAGE_ID> serve` 运行。

HTTP 请求将发送到端口 `8080`。容器必须提供 `/ping` URL 进行健康检查，并提供 `/invocations` URL 处理预测请求。我们将使用 CSV 格式作为输入。

因此，你的部署代码需要满足以下要求：

+   命名为 `serve`——没有扩展名，只是 `serve`。

+   使其可执行。

+   确保它在 `PATH` 中。

+   确保容器暴露了端口`8080`。

+   提供代码处理 `/ping` 和 `/invocations` URL。

这是更新后的 Dockerfile。我们安装 Flask，复制部署代码，并开放端口 `8080`：

```py
FROM python:3.7
RUN pip3 install --no-cache scikit-learn numpy pandas joblib
RUN pip3 install --no-cache flask
COPY sklearn-boston-housing-generic.py /usr/bin/train
COPY sklearn-boston-housing-serve.py /usr/bin/serve
RUN chmod 755 /usr/bin/train /usr/bin/serve
EXPOSE 8080
```

这是我们如何用 Flask 实现一个简单的预测服务：

1.  我们导入所需的模块。我们从 `/opt/ml/model` 加载模型并初始化 Flask 应用程序：

    ```py
    #!/usr/bin/env python
    import joblib, os
    import pandas as pd
    from io import StringIO
    import flask
    from flask import Flask, Response
    model_dir = '/opt/ml/model'
    model = joblib.load(os.path.join(model_dir, 
                        'model.joblib'))
    app = Flask(__name__)
    ```

1.  我们实现 `/ping` URL 来进行健康检查，方法是简单地返回 HTTP 代码 200（OK）：

    ```py
    @app.route("/ping", methods=["GET"])
    def ping():
        return Response(response="\n", status=200)
    ```

1.  我们实现了`/invocations` URL。如果内容类型不是`text/csv`，我们返回 HTTP 代码 415（不支持的媒体类型）。如果是，我们解码请求体并将其存储在文件样式的内存缓冲区中。然后，我们读取 CSV 样本，进行预测，并发送结果：

    ```py
    @app.route("/invocations", methods=["POST"])
    def predict():
        if flask.request.content_type == 'text/csv':
            data = flask.request.data.decode('utf-8')
            s = StringIO(data)
            data = pd.read_csv(s, header=None)
            response = model.predict(data)
            response = str(response)
        else:
            return flask.Response(
                response='CSV data only', 
                status=415, mimetype='text/plain')
        return Response(response=response, status=200)
    ```

1.  在启动时，脚本在 8080 端口上启动 Flask 应用程序：

    ```py
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=8080)
    ```

    即使您还不熟悉 Flask，这也不算太难。

1.  我们重新构建并推送镜像，然后使用相同的评估器再次进行训练。这里不需要进行任何更改。

1.  我们部署模型：

    ```py
    sk_predictor = sk.deploy(instance_type='ml.t2.medium',
                             initial_instance_count=1)
    ```

    提醒

    如果您在这里看到一些奇怪的行为（端点未部署、密秘的错误消息等），Docker 可能出了问题。`sudo service docker restart`应该能解决大多数问题。在`/tmp`中清理`tmp*`可能也会有所帮助。

1.  我们准备了一些测试样本，将内容类型设置为`text/csv`，并调用预测 API：

    ```py
    test_samples = ['0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1,296.0, 15.30, 396.90, 4.98',             
    '0.02731, 0.00, 7.070, 0, 0.4690, 6.4210, 78.90, 4.9671, 2,242.0, 17.80, 396.90, 9.14']
    sk_predictor.serializer =
        sagemaker.serializers.CSVSerializer()
    response = sk_predictor.predict(test_samples)
    print(response)
    ```

    您应该会看到类似于此的内容。API 已成功调用：

    ```py
    b'[[29.801388899699845], [24.990809475886078]]'
    ```

1.  完成后，我们删除端点：

    ```py
    sk_predictor.delete_endpoint()
    ```

在下一个例子中，我们将使用 R 环境来训练和部署模型。这将让我们有机会暂时离开 Python 世界。正如您将看到的那样，事情并没有真正不同。

# 构建一个完全自定义的 R 容器

R 是一种用于数据探索和分析的流行语言。在本例中，我们将构建一个自定义容器，以在波士顿房屋数据集上训练和部署线性回归模型。

整个过程与为 Python 构建自定义容器类似。我们将使用`plumber`而不是 Flask 来构建我们的预测 API：

## 使用 R 和 plumber 进行编码

如果你对 R 不太熟悉，不要担心。这是一个非常简单的例子，我相信你能跟上：

1.  我们编写一个函数来训练我们的模型。它从常规路径加载超参数和数据集。如果我们请求的话，它会对数据集进行标准化：

    ```py
    # train_function.R
    library("rjson")
    train <- function() {
        hp <- fromJSON(file = 
              '/opt/ml/input/config/hyperparameters.json')
        normalize <- hp$normalize
        data <- read.csv(file = 
                '/opt/ml/input/data/training/housing.csv', 
                header=T)
        if (normalize) {
            data <- as.data.frame(scale(data))
        }
    ```

    它训练一个线性回归模型，考虑所有特征来预测房屋的中位数价格（`medv`列）。最后，它将模型保存在正确的位置：

    ```py
        model = lm(medv~., data)
        saveRDS(model, '/opt/ml/model/model.rds')
    }
    ```

1.  我们编写一个函数来提供预测服务。使用`plumber`注解，我们为健康检查定义了`/ping` URL，为预测定义了`/invocations` URL：

    ```py
    # serve_function.R
    #' @get /ping
    function() {
      return('')
    }
    #' @post /invocations
    function(req) {
        model <- readRDS('/opt/ml/model/model.rds')
        conn <- textConnection(gsub('\\\\n', '\n', 
                               req$postBody))
        data <- read.csv(conn)
        close(conn)
        medv <- predict(model, data)
        return(medv)
    }
    ```

1.  将这两个部分结合在一起，我们编写一个主函数，它将作为我们脚本的入口点。SageMaker 将传递`train`或`serve`命令行参数，并在我们的代码中调用相应的函数：

    ```py
    library('plumber')
    source('train_function.R')
    serve <- function() {
        app <- plumb('serve_function.R')
        app$run(host='0.0.0.0', port=8080)}
    args <- commandArgs()
    if (any(grepl('train', args))) {
        train()
    }
    if (any(grepl('serve', args))) {
        serve()
    }
    ```

这就是我们需要的所有 R 代码。现在，让我们来处理容器。

## 构建自定义容器

我们需要构建一个自定义容器，存储 R 运行时和我们的脚本。Dockerfile 如下所示：

1.  我们从**Docker Hub**的官方 R 镜像开始，并添加我们需要的依赖项（这些是我在我的机器上需要的；您的情况可能有所不同）：

    ```py
    FROM r-base:latest
    WORKDIR /opt/ml/
    RUN apt-get update
    RUN apt-get install -y libcurl4-openssl-dev libsodium-dev
    RUN R -e "install.packages(c('rjson', 'plumber')) "
    ```

1.  然后，我们将我们的代码复制到容器中，并将主函数定义为其显式入口点：

    ```py
    COPY main.R train_function.R serve_function.R /opt/ml/
    ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/main.R", "--no-save"]
    ```

1.  我们在 ECR 中创建一个新仓库。然后，我们构建镜像（这可能需要一段时间，并涉及编译步骤），并推送它：

    ```py
    $ aws ecr create-repository --repository-name r-custom --region eu-west-1
    $ aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.eu-west-1.amazonaws.com/r-custom:latest
    $ docker build -t r-custom:latest -f Dockerfile .
    $ docker tag <IMAGE_ID> 123456789012.dkr.ecr.eu-west-1.amazonaws.com/r-custom:latest
    $ docker push 123456789012.dkr.ecr.eu-west-1.amazonaws.com/r-custom:latest
    ```

一切准备就绪，让我们开始训练并部署。

## 在 SageMaker 上训练和部署自定义容器

跳转到 Jupyter 笔记本，我们使用 SageMaker SDK 训练并部署我们的容器：

1.  我们配置一个带有自定义容器的`Estimator`模块：

    ```py
    r_estimator = Estimator(
        role = sagemaker.get_execution_role(),
        image_uri='123456789012.dkr.ecr.eu-west-1.amazonaws.com/r-custom:latest',
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=output,
        hyperparameters={'normalize': False}
    )
    r_estimator.fit({'training':training})
    ```

1.  一旦训练任务完成，我们像往常一样部署模型：

    ```py
    r_predictor = r_estimator.deploy(
        initial_instance_count=1, 
        instance_type='ml.t2.medium')
    ```

1.  最后，我们读取完整的数据集（为什么不呢？）并将其发送到端点：

    ```py
    import pandas as pd
    data = pd.read_csv('housing.csv')
    data.drop(['medv'], axis=1, inplace=True)
    data = data.to_csv(index=False)
    r_predictor.serializer = 
        sagemaker.serializers.CSVSerializer()
    response = r_predictor.predict(data)
    print(response)
    ```

    输出应该像这样：

    ```py
    b'[30.0337,25.0568,30.6082,28.6772,27.9288\. . .
    ```

1.  完成后，我们删除端点：

    ```py
    r_predictor.delete_endpoint()
    ```

无论你使用 Python、R，还是其他语言，构建和部署你自己的自定义容器都相对容易。然而，你仍然需要构建自己的网站应用程序，这可能是你既不知道如何做，也不喜欢做的事。如果我们有一个工具来处理所有这些麻烦的容器和网站问题，那该多好？

事实上，确实有一个平台：**MLflow**。

# 使用你自己的代码在 MLflow 上进行训练和部署

MLflow 是一个开源机器学习平台（[`mlflow.org`](https://mlflow.org)）。它由 Databricks（[`databricks.com`](https://databricks.com)）发起，Databricks 还为我们带来了**Spark**。MLflow 有许多功能，包括能够将 Python 训练的模型部署到 SageMaker。

本节并非旨在作为 MLflow 教程。你可以在[`www.mlflow.org/docs/latest/index.html`](https://www.mlflow.org/docs/latest/index.html)找到文档和示例。

## 安装 MLflow

在本地机器上，让我们为 MLflow 设置一个虚拟环境并安装所需的库。以下示例是在 MLflow 1.17 上测试的：

1.  我们首先初始化一个名为`mlflow-example`的新虚拟环境。然后，我们激活它：

    ```py
    $ virtualenv mlflow-example
    $ source mlflow-example/bin/activate
    ```

1.  我们安装了 MLflow 和训练脚本所需的库：

    ```py
    $ pip install mlflow gunicorn pandas sklearn xgboost boto3
    ```

1.  最后，我们下载已经在*第七章*中使用过的直接营销数据集，*使用内置框架扩展机器学习服务*：

    ```py
    $ wget -N https://sagemaker-sample-data-us-west-2.s3-us-west-2.amazonaws.com/autopilot/direct_marketing/bank-additional.zip
    $ unzip -o bank-additional.zip
    ```

设置完成。让我们开始训练模型。

## 使用 MLflow 训练模型

训练脚本为此次运行设置了 MLflow 实验，以便我们可以记录元数据（超参数、指标等）。然后，它加载数据集，训练一个 XGBoost 分类器，并记录模型：

```py
# train-xgboost.py
import mlflow.xgboost
import xgboost as xgb
from load_dataset import load_dataset
if __name__ == '__main__':
    mlflow.set_experiment('dm-xgboost')
    with mlflow.start_run(run_name='dm-xgboost-basic') 
    as run:
        x_train, x_test, y_train, y_test = load_dataset(
            'bank-additional/bank-additional-full.csv')
        cls = xgb.XGBClassifier(
                  objective='binary:logistic', 
                  eval_metric='auc')
        cls.fit(x_train, y_train)
        auc = cls.score(x_test, y_test)
        mlflow.log_metric('auc', auc)
        mlflow.xgboost.log_model(cls, 'dm-xgboost-model')
        mlflow.end_run()
```

`load_dataset()`函数按其名称所示执行，并记录多个参数：

```py
# load_dataset.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
def load_dataset(path, test_size=0.2, random_state=123):
    data = pd.read_csv(path)
    data = pd.get_dummies(data)
    data = data.drop(['y_no'], axis=1)
    x = data.drop(['y_yes'], axis=1)
    y = data['y_yes']
    mlflow.log_param("dataset_path", path)
    mlflow.log_param("dataset_shape", data.shape)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("one_hot_encoding", True)
    return train_test_split(x, y, test_size=test_size, 
                            random_state=random_state)
```

让我们训练模型并在 MLflow Web 应用程序中可视化其结果：

1.  在我们刚刚在本地机器上创建的虚拟环境中，我们像运行任何 Python 程序一样运行训练脚本：

    ```py
    $ python train-xgboost.py
    INFO: 'dm-xgboost' does not exist. Creating a new experiment
    AUC  0.91442097596504
    ```

1.  我们启动 MLflow Web 应用程序：

    ```py
    $ mlflow ui &
    ```

1.  当我们将浏览器指向[`localhost:5000`](http://localhost:5000)时，我们可以看到运行的信息，如下图所示：

![图 8.2 – 在 MLflow 中查看我们的任务](img/B17705_08_2.jpg)

图 8.2 – 在 MLflow 中查看我们的任务

训练成功了。在我们将模型部署到 SageMaker 之前，我们必须构建一个 SageMaker 容器。事实证明，这是一件非常简单的事。

## 使用 MLflow 构建 SageMaker 容器

只需在本地机器上执行一个命令：

```py
$ mlflow sagemaker build-and-push-container
```

MLflow 会自动构建一个与 SageMaker 兼容的 Docker 容器，并包含所有所需的依赖项。然后，它会在 Amazon ECR 中创建一个名为 `mlflow-pyfunc` 的仓库，并将镜像推送到该仓库。显然，这需要你正确设置 AWS 凭证。MLflow 将使用 AWS CLI 配置的默认区域。

一旦这个命令完成，你应该会在 ECR 中看到镜像，如下图所示：

![图 8.3 – 查看我们的容器在 ECR 中](img/B17705_08_3.jpg)

](img/B17705_08_3.jpg)

图 8.3 – 查看我们的容器在 ECR 中

我们的容器现在已经准备好部署了。

### 在本地部署模型使用 MLflow

我们将使用以下步骤部署我们的模型：

1.  我们可以通过一个命令在本地部署我们的模型，传递其运行标识符（可以在 MLflow 运行的 URL 中看到）和要使用的 HTTP 端口。这将启动一个基于 `gunicorn` 的本地 Web 应用程序：

    ```py
    $ mlflow sagemaker run-local -p 8888 -m runs:/d08ab8383ee84f72a92164d3ca548693/dm-xgboost-model
    ```

    你应该看到类似这样的内容：

    ```py
    [2021-05-26 20:21:23 +0000] [370] [INFO] Starting gunicorn 20.1.0
    [2021-05-26 20:21:23 +0000] [370] [INFO] Listening at: http://127.0.0.1:8000 (370)
    [2021-05-26 20:21:23 +0000] [370] [INFO] Using worker: gevent
    [2021-05-26 20:21:23 +0000] [381] [INFO] Booting worker with pid: 381 
    ```

1.  我们的预测代码非常简单。我们从数据集加载 CSV 样本，将它们转换为 JSON 格式，并使用 `requests` 库发送到端点。`requests` 是一个流行的 Python HTTP 库（[`requests.readthedocs.io`](https://requests.readthedocs.io)）：

    ```py
    # predict-xgboost-local.py 
    import json
    import requests
    from load_dataset import load_dataset
    port = 8888
    if __name__ == '__main__':
        x_train, x_test, y_train, y_test = load_dataset(
            'bank-additional/bank-additional-full.csv')
        input_data = x_test[:10].to_json(orient='split')
        endpoint = 'http://localhost:{}/invocations'
                   .format(port)
        headers = {'Content-type': 'application/json; 
                    format=pandas-split'}
        prediction = requests.post(
            endpoint, 
            json=json.loads(input_data),
            headers=headers)
        print(prediction.text)
    ```

1.  在另一个 shell 中运行此代码，调用本地模型并输出预测结果：

    ```py
    $ source mlflow-example/bin/activate
    $ python predict-xgboost-local.py
    [0.00046298891538754106, 0.10499032586812973, . . . 
    ```

1.  完成后，我们使用 *Ctrl* + *C* 终止本地服务器。

现在我们确信我们的模型在本地可以正常工作，我们可以将它部署到 SageMaker。

### 使用 MLflow 在 SageMaker 上部署模型

这又是一个一行命令：

1.  我们需要传递一个应用程序名称、模型路径和 SageMaker 角色的名称。你可以使用你在前几章中使用过的相同角色：

    ```py
    $ mlflow sagemaker deploy \
    --region-name eu-west-1 \
    -t ml.t2.medium \
    -a mlflow-xgb-demo \
    -m runs:/d08ab8383ee84f72a92164d3ca548693/dm-xgboost-model \
    -e arn:aws:iam::123456789012:role/Sagemaker-fullaccess
    ```

1.  几分钟后，端点进入服务状态。我们使用以下代码调用它。它加载测试数据集，并将前 10 个样本以 JSON 格式发送到以我们的应用程序命名的端点：

    ```py
    # predict-xgboost.py 
    import boto3
    from load_dataset import load_dataset
    app_name = 'mlflow-xgb-demo'
    region = 'eu-west-1'
    if __name__ == '__main__':
        sm = boto3.client('sagemaker', region_name=region)
        smrt = boto3.client('runtime.sagemaker', 
                            region_name=region)
        endpoint = sm.describe_endpoint(
                  EndpointName=app_name)
        print("Status: ", endpoint['EndpointStatus'])
        x_train, x_test, y_train, y_test = load_dataset(
            'bank-additional/bank-additional-full.csv')
        input_data = x_test[:10].to_json(orient="split")
        prediction = smrt.invoke_endpoint(
            EndpointName=app_name,
            Body=input_data,
            ContentType='application/json;
                         format=pandas-split')
        prediction = prediction['Body']
                     .read().decode("ascii")
        print(prediction)
    ```

    等一下！我们没有使用 SageMaker SDK。这是怎么回事？

    在这个例子中，我们处理的是一个现有的端点，而不是通过拟合估算器并部署预测器创建的端点。

    我们仍然可以使用 SageMaker SDK 重新构建一个预测器，正如我们将在 *第十一章* 中看到的那样，*部署机器学习模型*。不过，我们使用了我们亲爱的老朋友 `boto3`，AWS 的 Python SDK。我们首先调用 `describe_endpoint()` API 来检查端点是否在服务中。然后，我们使用 `invoke_endpoint()` API 来……调用端点！现在，我们暂时不需要了解更多。

    我们在本地机器上运行预测代码，输出结果如下：

    ```py
    $ python3 predict-xgboost.py
    Status:  InService
    0.00046298891538754106, 0.10499032586812973, 0.016391035169363022, . . .
    ```

1.  完成后，我们使用 MLflow CLI 删除端点。这会清理为部署创建的所有资源：

    ```py
    $ mlflow sagemaker delete -a mlflow-xgb-demo –region-name eu-west-1
    ```

使用 MLflow 的开发经验非常简单。它还有许多其他功能，您可能想要探索。

到目前为止，我们已经运行了训练和预测的示例。SageMaker 还有一个领域可以让我们使用自定义容器，**SageMaker Processing**，我们在[*第二章*中研究了它，*处理数据准备技术*。为了结束这一章，让我们为 SageMaker Processing 构建一个自定义的 Python 容器。

# 为 SageMaker 处理构建一个完全自定义的容器

我们将重用来自*第六章*的新闻头条示例，*训练自然语言处理模型*：

1.  我们从一个基于最小 Python 镜像的 Dockerfile 开始。我们安装依赖项，添加处理脚本，并将其定义为我们的入口点：

    ```py
    FROM python:3.7-slim
    RUN pip3 install --no-cache gensim nltk sagemaker
    RUN python3 -m nltk.downloader stopwords wordnet
    ADD preprocessing-lda-ntm.py /
    ENTRYPOINT ["python3", "/preprocessing-lda-ntm.py"]
    ```

1.  我们构建镜像并将其标记为`sm-processing-custom:latest`：

    ```py
    python:3.7 instead of python:3.7-slim. This makes it faster to push and download.
    ```

1.  使用 AWS CLI，我们在 Amazon ECR 中创建一个存储库来托管这个镜像，并登录到该存储库：

    ```py
    $ aws ecr create-repository --repository-name sm-processing-custom --region eu-west-1
    $ aws ecr get-login-password | docker login --username AWS --password-stdin 123456789012.dkr.ecr.eu-west-1.amazonaws.com/sm-processing-custom:latest
    ```

1.  使用镜像标识符，我们用存储库标识符为镜像打标签：

    ```py
    $ docker tag <IMAGE_ID> 123456789012.dkr.ecr.eu-west-1.amazonaws.com/sm-processing-custom:latest
    ```

1.  我们将镜像推送到存储库：

    ```py
    $ docker push 123456789012.dkr.ecr.eu-west-1.amazonaws.com/sm-processing-custom:latest
    ```

1.  切换到 Jupyter 笔记本，我们使用新的容器配置一个通用的`Processor`对象，它相当于我们用于训练的通用`Estimator`模块。因此，不需要`framework_version`参数：

    ```py
    from sagemaker.processing import Processor
    sklearn_processor = Processor( 
        image_uri='123456789012.dkr.ecr.eu-west-1.amazonaws.com/sm-processing-custom:latest',
        role=sagemaker.get_execution_role(),
        instance_type='ml.c5.2xlarge',
        instance_count=1)
    ```

1.  使用相同的`ProcessingInput`和`ProcessingOutput`对象，我们运行处理作业。由于我们的处理代码现在存储在容器内，我们不需要像使用`SKLearnProcessor`时那样传递`code`参数：

    ```py
    from sagemaker.processing import ProcessingInput, ProcessingOutput
    sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source=input_data,
                destination='/opt/ml/processing/input')
        ],
        outputs=[
            ProcessingOutput(
                output_name='train_data',
                source='/opt/ml/processing/train/')
        ],
        arguments=[
            '--filename', 'abcnews-date-text.csv.gz'
        ]
    )
    ```

1.  一旦训练作业完成，我们可以在 S3 中获取其输出。

这就结束了我们对 SageMaker 中自定义容器的探索。如您所见，只要它适合 Docker 容器，几乎可以运行任何东西。

# 概述

内置框架非常有用，但有时您需要一些稍微——或完全——不同的东西。无论是从内置容器开始还是从头开始，SageMaker 让您可以完全按照自己的需求构建训练和部署容器。自由为所有人！

在本章中，您学习了如何为数据处理、训练和部署定制 Python 和 R 容器。您还看到了如何使用 SageMaker SDK 及其常规工作流程来使用这些容器。您还了解了 MLflow，这是一个非常好的开源工具，允许您使用 CLI 训练和部署模型。

这就结束了我们对 SageMaker 建模选项的广泛介绍：内置算法、内置框架和自定义代码。在下一章中，您将学习 SageMaker 的功能，帮助您扩展训练作业。
