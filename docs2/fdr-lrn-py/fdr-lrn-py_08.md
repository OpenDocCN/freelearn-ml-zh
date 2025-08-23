

# 第八章：介绍现有的联邦学习框架

本章的目标是介绍现有的**联邦学习**（**FL**）框架和平台，将每个平台应用于涉及玩具**机器学习**（**ML**）问题的联邦学习场景。本章关注的平台是 Flower、TensorFlow Federated、OpenFL、IBM FL 和 STADLE——选择这些平台背后的想法是通过涵盖现有的 FL 平台范围来帮助你。

到本章结束时，你应该对如何使用每个平台进行联邦学习有一个基本的了解，并且你应该能够根据其相关的优势和劣势选择一个平台用于联邦学习应用。

在本章中，我们将涵盖以下主题：

+   现有 FL 框架的介绍

+   使用现有框架在电影评论数据集上实现示例 NLP FL 任务

+   使用现有框架实现示例计算机视觉 FL 任务，使用非-IID 数据集

# 技术要求

你可以在本书的 GitHub 仓库中找到本章的补充代码文件：

https://github.com/PacktPublishing/Federated-Learning-with-Python

重要提示

你可以使用代码文件进行个人或教育目的。请注意，我们不会支持商业部署，并且不会对使用代码造成的任何错误、问题或损害负责。

本章中的每个实现示例都是在运行 Ubuntu 20.04 的 x64 机器上运行的。

NLP 示例的训练代码实现需要以下库来运行：

+   Python 3 (版本 ≥ 3.8)

+   NumPy

+   TensorFlow（版本 ≥ 2.9.1）

+   TensorFlow Hub (`pip` `install tensorflow-hub`)

+   TensorFlow Datasets (`pip` `install tensorflow-datasets`)

+   TensorFlow Text (`pip` `install tensorflow-text`)

由于模型的大小，建议使用带有适当 TensorFlow 安装的 GPU 来节省 NLP 示例的训练时间。

训练非-IID（非独立同分布）计算机视觉示例的代码实现需要以下库来运行：

+   Python 3（版本 ≥ 3.8）

+   NumPy

+   PyTorch（版本 ≥ 1.9）

+   Torchvision（版本 ≥ 0.10.0，与 PyTorch 版本相关联）

每个 FL 框架的安装说明列在以下子节中。

## TensorFlow Federated

你可以安装以下库来使用 TFF：

+   `tensorflow_federated`（使用`pip install tensorflow_federated`命令）

+   `nest_asyncio`（使用`pip install nest_asyncio`命令）

## OpenFL

你可以使用`pip install openfl`命令安装 OpenFL。

或者，你可以使用以下命令从源代码构建：

```py
git clone https://github.com/intel/openfl.git 
cd openfl
pip install .
```

## IBM FL

安装 IBM FL 的本地版本需要位于代码仓库中的 wheel 安装文件。要执行此安装，请运行以下命令：

```py
git clone https://github.com/IBM/federated-learning-lib.git
cd federated-learning-lib
pip install federated_learning_lib-*-py3-none-any.whl
```

## Flower

你可以使用`pip install flwr`命令安装 Flower。

## STADLE

您可以使用 `pip install stadle-client` 命令安装 STADLE 客户端库。

# 联邦学习框架简介

首先，我们介绍后续实现重点章节中将要使用的联邦学习框架和平台。

## Flower

Flower ([`flower.dev/`](https://flower.dev/)) 是一个开源且与机器学习框架无关的联邦学习框架，旨在让用户易于使用。Flower 采用标准的客户端-服务器架构，其中客户端被设置为从服务器接收模型参数，在本地数据上训练，并将新的本地模型参数发送回服务器。

联邦学习过程的高级编排由 Flower 所称的策略决定，服务器使用这些策略来处理客户端选择和参数聚合等方面。

Flower 使用 **远程过程调用** (**RPCs**) 来通过客户端执行从服务器发送的消息以执行所述编排。框架的可扩展性允许研究人员尝试新的方法，例如新的聚合算法和通信方法（如模型压缩）。

## TensorFlow Federated (TFF)

TFF ([`www.tensorflow.org/federated`](https://www.tensorflow.org/federated)) 是一个基于 TensorFlow 的开源 FL/计算框架，旨在允许研究人员轻松地使用现有的 TensorFlow/Keras 模型和训练管道模拟联邦学习。它包括联邦核心层，允许实现通用联邦计算，以及联邦学习层，它建立在核心之上，并为 FL 特定过程提供接口。

TFF 专注于 FL 的单机本地模拟，使用包装器从标准的 TensorFlow 等价物创建 TFF 特定的数据集、模型和联邦计算（FL 过程中执行的核心客户端和服务器计算）。从通用联邦计算构建一切的关注使研究人员能够按需实现每个步骤，从而支持实验。

## OpenFL

OpenFL ([`github.com/intel/openfl`](https://github.com/intel/openfl)) 是英特尔开发的开源联邦学习框架，专注于允许跨隔离区隐私保护机器学习。OpenFL 允许根据联盟（指整个 FL 系统）的预期生命周期选择两种不同的工作流程。

在基于聚合器的流程中，单个实验及其相关的联邦学习计划从聚合器发送到参与 *协作者*（代理）以作为 FL 流程的本地训练步骤运行——实验完成后，联盟停止。在基于导演的流程中，使用持久组件而不是短生命周期的组件，以便按需运行实验。以下图表描述了基于导演的流程的架构和用户：

![图 8.1 – 基于总监的工作流程架构（改编自 https://openfl.readthedocs.io/en/latest/source/openfl/components.html）![图片](img/B18369_08_01.jpg)

图 8.1 – 基于总监的工作流程架构（改编自 https://openfl.readthedocs.io/en/latest/source/openfl/components.html）

**总监经理**负责实验的运行，与位于协作节点上的长期**信使**组件合作，管理每个实验的短期组件（协作者+聚合者）。在针对跨数据孤岛的场景时，OpenFL 对管理数据分片给予了独特的关注，包括数据表示在不同孤岛中不同的情况。

## IBM FL

IBM FL 是一个也专注于企业联邦学习的框架。它遵循简单的聚合者-参与者设计，其中一些拥有本地数据的参与者通过向聚合者发送增量模型训练结果并与生成的聚合模型（遵循标准的客户端-服务器联邦学习架构）合作，与其他参与者协作。IBM FL 对多种融合（聚合）算法和旨在对抗偏见的某些公平技术提供官方支持——这些算法的详细信息可以在位于 https://github.com/IBM/federated-learning-lib 的存储库中找到。IBM FL 的一个具体目标是高度可扩展，使用户能够轻松地进行必要的修改，以满足特定的功能需求。它还支持基于 Jupyter-Notebook 的仪表板，以帮助协调联邦学习实验。

## STADLE

与之前的框架不同，STADLE ([`stadle.ai/`](https://stadle.ai/)) 是一个与机器学习框架无关的联邦学习和分布式学习 SaaS 平台，旨在允许无缝地将联邦学习集成到生产就绪的应用程序和机器学习管道中。STADLE 的目标是最大限度地减少集成所需的特定于联邦学习的代码量，使联邦学习对新手来说易于访问，同时仍然为那些想要进行实验的人提供灵活性。

使用 STADLE SaaS 平台，不同技术能力的用户可以在所有规模上协作进行联邦学习项目。性能跟踪和模型管理功能使用户能够生成具有强大性能的验证联邦模型，而直观的配置面板允许对联邦学习过程进行详细控制。STADLE 使用两级组件层次结构，允许多个聚合器并行操作，以匹配需求。以下图展示了高级架构：

![图 8.2 – STADLE 多聚合器架构![图片](img/B18369_08_02.jpg)

图 8.2 – STADLE 多聚合器架构

STADLE 客户端的开发通过`pip`安装和易于理解的配置文件简化，同时公开提供了一些示例，供用户参考 STADLE 如何集成到现有的机器学习代码中的不同方式。

## PySyft

尽管由于代码库的持续变化，PySyft ([`github.com/OpenMined/PySyft`](https://github.com/OpenMined/PySyft)) 的实现不包括在本章中，但它仍然是隐私保护深度学习空间中的主要参与者。PySyft 背后的核心原则是允许在不对数据进行直接访问的情况下对存储在机器上的数据进行计算。这是通过在用户和数据位置之间添加一个中间层来实现的，该层向参与工作的机器发送计算请求，将计算结果返回给用户，同时保持每个工人存储和使用的用于执行计算的数据的隐私。

这种通用能力直接扩展到 FL，重新设计正常深度学习训练流程的每一步，使其成为对每个参与 FL 的工人（代理）存储的模型参数和数据的计算。为了实现这一点，PySyft 利用钩子封装标准的 PyTorch/TensorFlow 库，修改必要的内部函数，以便支持模型训练和测试作为 PySyft 隐私保护计算。

现在已经解释了 FL 框架背后的高级思想，我们将转向其实际应用中的实现级细节，以两个示例场景为例。首先，我们来看如何修改现有的用于 NLP 模型的集中式训练代码，使其能够使用 FL。

# 示例 - NLP 模型的联邦训练

通过上述每个 FL 框架将第一个 ML 问题转换为 FL 场景的将是 NLP 领域的分类问题。从高层次来看，NLP 是指计算语言学和 ML 的交集，其总体目标是使计算机能够从人类语言中达到某种程度的*理解* - 这种理解的细节根据要解决的具体问题而大相径庭。

在这个例子中，我们将对电影评论进行情感分析，将它们分类为正面或负面。我们将使用的数据集是 SST-2 数据集 (https://nlp.stanford.edu/sentiment/)，包含以字符串格式表示的电影评论和相关的二进制标签 0/1，分别代表负面和正面情感。

我们将使用进行二元分类的模型是一个带有自定义分类头的预训练 BERT 模型。BERT 模型允许我们将句子编码成一个高维数值向量，然后将其传递到分类头以输出二元标签预测；有关 BERT 模型的更多信息，请参阅 https://huggingface.co/blog/bert-101。我们选择使用一个预训练模型，该模型在大量训练后已经学会了如何生成句子的通用编码，而不是从头开始进行训练。这允许我们将训练集中在分类头上，以微调模型在 SST-2 数据集上的性能，从而节省时间并保持性能。

现在，我们将通过本地（集中式）训练代码，该代码将作为展示如何使用每个 FL 框架的基础，从 Keras 模型定义和数据集加载器开始。

## 定义情感分析模型

在`sst_model.py`文件中定义的`SSTModel`对象是我们将在这个示例中使用的 Keras 模型。

首先，我们导入必要的库：

```py
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_text
import tensorflow_hub as hub
import tensorflow_datasets as tfds
```

TensorFlow Hub 用于轻松下载预训练的 BERT 权重到 Keras 层。当从 TensorFlow Hub 加载 BERT 权重时使用 TensorFlow Text。TensorFlow Datasets 将允许我们下载和缓存 SST-2 数据集。

接下来，我们定义模型并初始化模型层对象：

```py
class SSTModel(keras.Model):
    def __init__(self):
        super(SSTModel, self).__init__()
        self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.small_bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
        self.small_bert.trainable = False
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1, activation='sigmoid')
```

`preprocessor`对象将原始句子输入批次转换为 BERT 模型使用的格式。我们从 TensorFlow Hub 加载预处理器和 BERT 层，然后初始化构成分类头的密集层。我们使用 sigmoid 激活函数在最后将输出压缩到区间（0,1），以便与真实标签进行比较。

然后，我们可以定义模型的正向传递：

```py
    def call(self, inputs):
        input_dict = self.preprocessor(inputs)
        bert_output = self.small_bert(input_dict)['pooled_output']
        output = self.fc1(keras.activations.relu(bert_output, alpha=0.2))
        scores = self.fc3(self.fc2(output))

        return scores
```

我们将 leaky ReLU 应用于 BERT 输出，在传递到分类头层之前添加非线性。

## 创建数据加载器

我们还实现了一个函数，使用 TensorFlow Datasets 库加载 SST-2 数据集。首先，加载训练数据并将其转换为 NumPy 数组，以便在训练期间使用：

```py
def load_sst_data(client_idx=None, num_clients=1):
    x_train = []
    y_train = []
    for d in tfds.load(name="glue/sst2", split="train"):
        x_train.append(d['sentence'].numpy())
        y_train.append(d['label'].numpy())
    x_train = np.array(x_train)
    y_train = np.array(y_train)
```

我们以类似的方式加载测试数据：

```py
    x_test = []
    y_test = []
    for d in tfds.load(name="glue/sst2", split="validation"):
        x_test.append(d['sentence'].numpy())
        y_test.append(d['label'].numpy())
    x_test = np.array(x_test)
    y_test = np.array(y_test)
```

如果指定了`client_idx`和`num_clients`，我们返回训练数据集的相应分区——这将用于执行联邦学习：

```py
    if (client_idx is not None):
        shard_size = int(x_train.size / num_clients)
        x_train = x_train[client_idx*shard_size:(client_idx+1)*shard_size]
        y_train = x_train[client_idx*shard_size:(client_idx+1)*shard_size]
    return (x_train, y_train), (x_test, y_test)
```

接下来，我们检查位于`local_training.py`中的执行本地训练的代码。

## 训练模型

我们首先导入必要的库：

```py
import tensorflow as tf
from tensorflow import keras
from sst_model import SSTModel, load_sst_data
```

然后，我们可以使用之前定义的数据集加载器（不进行拆分）来加载训练和测试分割：

```py
(x_train,y_train), (x_test,y_test) = load_sst_data()
```

现在，我们可以编译模型并开始训练：

```py
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.0005, amsgrad=False),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.BinaryAccuracy()]
)
model.fit(x_train, y_train, batch_size=64, epochs=3)
```

最后，我们在测试分割上评估模型：

```py
_, acc = model.evaluate(x_test, y_test, batch_size=64)
print(f"Accuracy of model on test set: {(100*acc):.2f}%")
```

经过三个训练周期后，模型应该达到大约 82%的测试准确率。

现在我们已经通过了本地训练代码，我们可以检查如何修改代码以使用上述每个 FL 框架进行联邦学习。

## 采用 FL 训练方法

为了展示如何将 FL 应用于 SST 模型训练场景，我们首先需要将原始 SST-2 数据集拆分成不相交的子集，这些子集代表 FL 应用中的本地数据集。为了简化问题，我们将研究三个代理各自在数据集的不同三分之一上训练的情况。

目前，这些子集是从数据集中随机采样而不重复的 – 在下一节“在非-IID 数据上对图像分类模型进行联邦训练”中，我们将研究本地数据集是从原始数据集的有偏采样中创建的情况。我们不会在本地训练三个 epoch，而是将进行三轮 FL，每轮本地训练阶段在本地数据上训练一个 epoch。FedAvg 将在每一轮结束时用于聚合本地训练的模型。在这三轮之后，将使用最终的聚合模型计算上述验证指标，从而允许比较本地训练案例和 FL 案例。

## 集成 TensorFlow Federated 用于 SST-2

如前所述，**TensorFlow Federated**（**TFF**）框架是在 TensorFlow 和 Keras 深度学习库之上构建的。模型实现是使用 Keras 完成的；因此，将 TFF 集成到本地训练代码中相对简单。

第一步是在加载数据集之前添加 TFF 特定的导入和 FL 特定的参数：

```py
import nest_asyncio
nest_asyncio.apply()
import tensorflow_federated as tff
NUM_CLIENTS = 3
NUM_ROUNDS = 3
```

TFF 允许我们通过向 FL 过程传递适当数量的数据集（本地数据集）来模拟一定数量的代理。为了在预处理后将 SST-2 数据集分成三份，我们可以使用以下代码：

```py
client_datasets = [load_sst_data(idx, NUM_CLIENTS)[0] for idx in range(NUM_CLIENTS)]
```

接下来，我们必须使用 TFF API 函数包装 Keras 模型，以便轻松创建相应的`tff.learning.Model`对象。我们创建一个函数，初始化 SST 模型，并将其与输入规范（关于每个数据元素大小的信息）一起传递给这个 API 函数，返回结果 – TFF 将在 FL 过程中内部使用此函数来创建模型：

```py
def sst_model_fn():
    sst_model = SSTModel()
    sst_model.build(input_shape=(None,64))
    return tff.learning.from_keras_model(
        sst_model,
        input_spec=tf.TensorSpec(shape=(None), dtype=tf.string),
        loss=keras.metrics.BinaryCrossentropy()
    )
```

使用`sst_model_fn`函数以及用于更新本地模型和聚合模型的优化器，可以创建 TFF FedAvg 过程。对于服务器优化器函数使用 1.0 的学习率，允许在每一轮结束时用新的聚合模型替换旧的模型（而不是计算旧模型和新模型的加权平均值）：

```py
fed_avg_process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn = sst_model_fn,
    client_optimizer_fn = lambda: keras.optimizers.Adam(learning_rate=0.001),
    server_optimizer_fn = lambda: keras.optimizers.SGD(learning_rate=1.0)
)
```

最后，我们初始化并运行联邦学习过程 10 轮。每次`fed_avg_process.next()`调用通过在客户端数据集上使用三个模型进行本地训练，然后使用 FedAvg 进行聚合来模拟一轮。第一轮后的状态被传递到下一次调用，作为该轮的起始 FL 状态：

```py
state = fed_avg_process.initialize()
for round in range(NUM_ROUNDS):
    state = fed_avg_process.next(state, client_datasets).state
```

FL 过程完成后，我们将最终的聚合 `tff.learning.Model` 对象转换回原始的 Keras 模型格式，以便计算验证指标：

```py
fed_weights = fed_avg_process.get_model_weights(state)
fed_sst_model = SSTModel()
fed_sst_model.build(input_shape=(None, 64))
fed_sst_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.005, amsgrad=False),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.BinaryAccuracy()]
)
fed_weights.assign_weights_to(fed_sst_model)
_, (x_test, y_test) = load_sst_data()
_, acc = fed_sst_model.evaluate(x_test, y_test, batch_size=64)
print(f"Accuracy of federated model on test set: {(100*acc):.2f}%")
```

聚合模型的最终准确率应约为 82%。

从这一点来看，应该很清楚 TFF FedAvg 的结果几乎与本地训练场景的结果相同。

## 集成 OpenFL 用于 SST-2

请记住，OpenFL 支持两种不同的工作流程：基于聚合器的工作流程和基于导演的工作流程。本例将使用基于导演的工作流程，涉及长期存在的组件，可以处理传入的 FL 任务请求。这选择是因为希望有一个持久的 FL 设置来部署多个项目；然而，两种工作流程都执行相同的核心 FL 过程，因此表现出类似的表现。

为了帮助在此情况下进行模型序列化，我们只聚合分类头权重，在训练和验证时运行时重建完整模型（TensorFlow Hub 缓存下载的层，因此下载过程只发生一次）。我们在 `sst_model.py` 中包含以下函数以帮助进行此修改：

```py
def get_sst_full(preprocessor, bert, classification_head):
    sst_input = keras.Input(shape=(), batch_size=64, dtype=tf.string)
    scores = classification_head(bert(preprocessor(sst_input))['pooled_output'])
    return keras.Model(inputs=sst_input, outputs=scores, name='sst_model')
def get_classification_head():
    classification_head = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(768,)),
        layers.Dense(64, activation='relu', input_shape=(512,)),
        layers.Dense(1, activation='sigmoid', input_shape=(64,))
    ])
    return classification_head
```

由于 OpenFL 专注于解决数据孤岛问题，从 SST-2 数据创建本地数据集比 TFF 情况稍微复杂一些。创建数据集所需的对象将在名为 `sst_fl_dataset.py` 的单独文件中实现。

首先，我们包括必要的导入。我们导入的两个 OpenFL 特定对象是处理数据集加载和分片的 `ShardDescriptor` 对象，以及处理数据集访问的 `DataInterface` 对象：

```py
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.interface.interactive_api.experiment import DataInterface
import tensorflow as tf
from sst_model import load_sst_data
```

### 实现 ShardDescriptor

我们首先实现了 `SSTShardDescriptor` 类。当创建此分片描述符时，我们保存 `rank`（客户端编号）和 `worldsize`（客户端总数）值，然后加载训练和验证数据集：

```py
class SSTShardDescriptor(ShardDescriptor):
    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        (x_train,y_train), (x_test,y_test) = load_sst_data(self.rank-1, self.worldsize)
        self.data_by_type = {
            'train': tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64),
            'val': tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
        }
```

我们实现了 `ShardDescriptor` 类函数以获取可用的数据集类型（在这种情况下为训练和验证）以及基于客户端排名的相应数据集/分片：

```py
    def get_shard_dataset_types(self):
        return list(self.data_by_type)
    def get_dataset(self, dataset_type='train'):
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return self.data_by_type[dataset_type]
```

我们还指定了正在使用的数据集的具体属性。请注意，样本形状设置为 `1`。`SSTModel` 的预处理层允许我们传入字符串作为输入，这些字符串被视为类型为 `tf.string` 且长度为 `1` 的输入向量：

```py
    @property
    def sample_shape(self):
        return ["1"]
    @property
    def target_shape(self):
        return ["1"]
    @property
    def dataset_description(self) -> str:
        return (f'SST dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
```

这样，`SSTShardDescriptor` 的实现就完成了。

### 实现数据接口

接下来，我们将 `SSTFedDataset` 类实现为 `DataInterface` 的子类。这是通过实现分片描述符获取器和设置器方法来完成的，设置器方法准备要提供给训练/验证 FL 任务的数据：

```py
class SSTFedDataset(DataInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    @property
    def shard_descriptor(self):
        return self._shard_descriptor
    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        self._shard_descriptor = shard_descriptor

        self.train_set = shard_descriptor.get_dataset('train')
        self.valid_set = shard_descriptor.get_dataset('val')
```

我们还实现了 API 函数以授予数据集访问和数据集大小信息（用于聚合）：

```py
    def get_train_loader(self):
        return self.train_set
    def get_valid_loader(self):
        return self.valid_set
    def get_train_data_size(self):
        return len(self.train_set) * 64
    def get_valid_data_size(self):
        return len(self.valid_set) * 64
```

这样，就可以构建并使用本地 SST-2 数据集了。

### 创建 FLExperiment

现在，我们专注于在新的文件`fl_sim.py`中实现 FL 过程的实际实现。首先，我们导入必要的库——从 OpenFL 中，我们导入以下内容：

+   `TaskInterface`：允许我们为模型定义 FL 训练和验证任务；注册的任务是 director 指示每个 envoy 执行的任务

+   `ModelInterface`：允许我们将我们的 Keras 模型转换为 OpenFL 在注册任务中使用的格式

+   `Federation`：管理与 director 连接相关的信息

+   `FLExperiment`：使用`TaskInterface`、`ModelInterface`和`Federation`对象来执行 FL 过程

必要的导入如下所示：

```py
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from openfl.interface.interactive_api.experiment import TaskInterface
from openfl.interface.interactive_api.experiment import ModelInterface
from openfl.interface.interactive_api.experiment import FLExperiment
from openfl.interface.interactive_api.federation import Federation
from sst_model import get_classification_head, get_sst_full
from sst_fl_dataset import SSTFedDataset
```

接下来，我们使用默认的`director`连接信息创建`Federation`对象：

```py
client_id = 'api'
director_node_fqdn = 'localhost'
director_port = 50051
federation = Federation(
    client_id=client_id,
    director_node_fqdn=director_node_fqdn,
    director_port=director_port, 
    tls=False
)
```

然后，我们使用相关的优化器和损失函数初始化模型——这些对象被 OpenFL 的`KerasAdapter`用于创建`ModelInterface`对象。我们在一个虚拟的 Keras 输入上调用模型，以便在将模型传递给`ModelInterface`之前初始化所有权重：

```py
classification_head = get_classification_head()
optimizer = keras.optimizers.Adam(learning_rate=0.005, amsgrad=False)
loss = keras.losses.BinaryCrossentropy()
framework_adapter = 'openfl.plugins.frameworks_adapters.keras_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=classification_head, optimizer=optimizer, framework_plugin=framework_adapter)
```

接下来，我们创建一个`TaskInterface`对象，并使用它来注册训练任务。请注意，将优化器包含在任务的装饰器函数中会导致训练数据集被传递给任务；否则，验证数据集将被传递给任务：

```py
TI = TaskInterface()
@TI.register_fl_task(model='model', data_loader='train_data', device='device', optimizer='optimizer')
def train(model, train_data, optimizer, device):
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    small_bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    small_bert.trainable = False
    full_model = get_sst_full(preprocessor, small_bert, model)
    full_model.compile(loss=loss, optimizer=optimizer)
    history = full_model.fit(train_data, epochs=1)
    return {'train_loss':history.history['loss'][0]}
```

类似地，我们使用`TaskInterface`对象注册验证任务。请注意，我们可以收集由`evaluate`函数生成的指标，并将值作为跟踪性能的手段：

```py
@TI.register_fl_task(model='model', data_loader='val_data', device='device')
def validate(model, val_data, device):
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    small_bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    small_bert.trainable = False
    full_model = get_sst_full(preprocessor, small_bert, model)
    full_model.compile(loss=loss, optimizer=optimizer)
    loss, acc = full_model.evaluate(val_data, batch_size=64)
    return {'val_acc':acc, 'val_loss':loss,}
```

现在，我们可以使用之前实现的`SSTFedDataset`类加载数据集，并使用创建的`ModelInterface`、`TaskInterface`和`SSTFedDatasets`对象创建并启动一个新的`FLExperiment`：

```py
fed_dataset = SSTFedDataset()
fl_experiment = FLExperiment(federation=federation, experiment_name='sst_experiment')
fl_experiment.start(
    model_provider=MI,
    task_keeper=TI,
    data_loader=fed_dataset,
    rounds_to_train=3,
    opt_treatment='CONTINUE_LOCAL'
)
```

### 定义配置文件

最后一步是创建由`director`和`envoys`使用的配置文件，以便实际加载数据并启动 FL 过程。首先，我们创建包含以下信息的`director_config`：

```py
settings:
  listen_host: localhost
  listen_port: 50051
  sample_shape: ["1"]
  target_shape: ["1"]
```

这被保存在`director/director_config.yaml`中。

我们随后创建了三个`envoy`配置文件。第一个文件（`envoy_config_1.yaml`）包含以下内容：

```py
params:
  cuda_devices: []
optional_plugin_components: {}
shard_descriptor:
  template: sst_fl_dataset.SSTShardDescriptor
  params:
    rank_worldsize: 1, 3
```

第二个和第三个`envoy`配置文件与第一个相同，只是`rank_worldsize`的值分别为`2, 3`和`3, 3`。这些配置文件以及所有代码文件都存储在实验目录中。目录结构应如下所示：

+   `director`

    +   `director_config.yaml`

+   `experiment`

    +   `envoy_config_1.yaml`

    +   `envoy_config_2.yaml`

    +   `envoy_config_3.yaml`

    +   `sst_fl_dataset.py`

    +   `sst_model.py`

    +   `fl_sim.py（包含`FLExperiment`创建的文件）`

一切准备就绪后，我们现在可以使用 OpenFL 执行 FL。

### 运行 OpenFL 示例

首先，从`director`文件夹中运行以下命令以启动 director（确保 OpenFL 已安装在工作环境中）：

```py
fx director start --disable-tls -c director_config.yaml
```

接下来，在实验目录中分别在不同的终端运行以下命令：

```py
fx envoy start -n envoy_1 -–disable-tls --envoy-config-path envoy_config_1.yaml -dh localhost -dp 50051
fx envoy start -n envoy_2 -–disable-tls --envoy-config-path envoy_config_2.yaml -dh localhost -dp 50051
fx envoy start -n envoy_3 -–disable-tls --envoy-config-path envoy_config_3.yaml -dh localhost -dp 50051
```

最后，通过运行 `fl_sim.py` 脚本来启动 `FLExperiment`。完成三轮后，聚合模型应该达到大约 82% 的验证准确率。再次强调，性能几乎与本地训练场景相同。

## 集成 IBM FL 用于 SST-2

IBM FL 在执行联邦学习时使用保存的模型版本。以下代码（`create_saved_model.py`）初始化一个模型（在虚拟输入上调用模型以初始化参数）然后以 Keras `SavedModel` 格式保存模型供 IBM FL 使用：

```py
import tensorflow as tf
from tensorflow import keras
from sst_model import SSTModel
sst_model = SSTModel()
optimizer = keras.optimizers.Adam(learning_rate=0.005, amsgrad=False)
loss = keras.losses.BinaryCrossentropy(),
sst_model.compile(loss=loss, optimizer=optimizer)
sst_input = keras.Input(shape=(), dtype=tf.string)
sst_model(sst_input)
sst_model.save('sst_model_save_dir')
```

运行此命令一次以将模型保存到名为 `sst_model_save_dir` 的文件夹中 – 我们将指示 IBM FL 从此目录加载保存的模型。

### 创建 DataHandler

接下来，我们创建一个 IBM FL `DataHandler` 类的子类，该类负责向模型提供训练和验证数据 – 这个子类将加载、预处理并存储 SST 数据集作为类属性。我们首先导入必要的库：

```py
from ibmfl.data.data_handler import DataHandler
import tensorflow as tf
from sst_model import load_sst_data
```

这个类的 `init` 函数加载数据信息参数，然后使用这些参数来加载正确的数据集分片 SST-2：

```py
class SSTDataHandler(DataHandler):
    def __init__(self, data_config=None):
        super().__init__()
        if (data_config is not None):
            if ('client_id' in data_config):
                self.client_id = int(data_config['client_id'])
            if ('num_clients' in data_config):
                self.num_clients = int(data_config['num_clients'])
        train_data, val_data = load_sst_data(self.client_id-1, self.num_clients)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(64)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(64)
```

我们还实现了返回用于训练/验证期间使用的加载数据集的 API 函数：

```py
    def get_data(self):
        return self.train_dataset, self.val_dataset
```

### 定义配置文件

下一步是创建在启动聚合器和初始化聚会时使用的配置 JSON 文件。聚合配置首先指定它将用于与聚会通信的连接信息：

```py
{
    "connection": {
        "info": {
            "ip": "127.0.0.1",
            "port": 5000,
            "tls_config": {
                "enable": "false"
            }
        },
        "name": "FlaskConnection",
        "path": "ibmfl.connection.flask_connection",
        "sync": "False"
    },
```

接下来，我们指定用于聚合的融合处理器：

```py
    "fusion": {
        "name": "IterAvgFusionHandler",
        "path": "ibmfl.aggregator.fusion.iter_avg_fusion_handler"
    },
```

我们还指定了与本地训练和聚合相关的超参数。`perc_quorum` 指的是在聚合开始之前必须参与聚会的比例：

```py
    "hyperparams": {
        "global": {
            "max_timeout": 10800,
            "num_parties": 1,
            "perc_quorum": 1,
            "rounds": 3
        },
        "local": {
            "optimizer": {
                "lr": 0.0005
            },
            "training": {
                "epochs": 1
            }
        }
    },
```

最后，我们指定要使用的 IBM FL 协议处理器：

```py
    "protocol_handler": {
        "name": "ProtoHandler",
        "path": "ibmfl.aggregator.protohandler.proto_handler"
    }
}
```

此配置保存在 `agg_config.json` 文件中。

我们还创建了用于使用本地数据进行联邦学习的基聚会配置文件。我们首先指定聚合器和聚会的连接信息：

```py
{
    "aggregator":
        {
            "ip": "127.0.0.1",
            "port": 5000
        },
    "connection": {
        "info": {
            "ip": "127.0.0.1",
            "port": 8085,
            "id": "party",
            "tls_config": {
                "enable": "false"
            }
        },
        "name": "FlaskConnection",
        "path": "ibmfl.connection.flask_connection",
        "sync": "false"
    },
```

然后，我们指定要使用的数据处理器和本地训练处理器 – 此组件使用模型信息和本地数据训练 SST 模型：

```py
    "data": {
        "info": {
            "client_id": 0,
            "num_clients": 3
        },
        "name": "SSTDataHandler",
        "path": "sst_data_handler"
    },
    "local_training": {
        "name": "LocalTrainingHandler",
        "path": "ibmfl.party.training.local_training_handler"
    },
```

指定模型格式和信息 – 这是我们指向之前创建的保存模型的地方：

```py
    "model": {
        "name": "TensorFlowFLModel",
        "path": "ibmfl.model.tensorflow_fl_model",
        "spec": {
            "model-name": "sst_model",
            "model_definition": "sst_model_save_dir"
        }
    },
```

最后，我们指定协议处理器：

```py
    "protocol_handler": {
        "name": "PartyProtocolHandler",
        "path": "ibmfl.party.party_protocol_handler"
    }
}
```

### 创建 IBM FL 聚会

使用这种方式，剩下的只是启动每个聚会的代码，保存在 `fl_sim.py` 文件中。我们首先导入必要的库：

```py
import argparse
import json
from ibmfl.party.party import Party
```

我们包含一个 `argparse` 参数，允许指定聚会编号 – 这用于修改基本聚会配置文件，以便从同一文件启动不同的聚会：

```py
parser = argparse.ArgumentParser()
parser.add_argument("party_id", type=int)
args = parser.parse_args()
party_id = args.party_id
with open('party_config.json') as cfg_file:
    party_config = json.load(cfg_file)
party_config['connection']['info']['port'] += party_id
party_config['connection']['info']['id'] += f'_{party_id}'
party_config['data']['info']['client_id'] = party_id
```

最后，我们使用修改后的配置信息创建并启动一个新的 `Party` 对象：

```py
party = Party(config_dict=party_config)
party.start()
party.register_party()
```

使用这种方式，我们现在可以开始使用 IBM FL 进行联邦学习。

### 运行 IBM FL 示例

首先，通过运行以下命令来启动 `aggregator`：

```py
python -m ibmfl.aggregator.aggregator agg_config.json
```

在聚合器完成设置后，输入 `START` 并按 *Enter* 键以打开聚合器以接收传入的连接。然后，你可以在单独的终端中使用以下命令启动三个参与者：

```py
python fl_sim.py 1
python fl_sim.py 2
python fl_sim.py 3
```

最后，在聚合器窗口中输入 `TRAIN` 并按 *Enter* 键开始 FL 流程。当完成三轮后，你可以在同一窗口中输入 `SAVE` 以保存最新的聚合模型。

## 将 Flower 集成到 SST-2 中

必须在现有本地训练代码之上集成的两个主要 Flower 组件是客户端和策略子类实现。客户端子类实现允许我们与 Flower 接口，API 函数允许在客户端和服务器之间传递模型参数。策略子类实现允许我们指定服务器执行的聚合方法的细节。

我们首先编写代码来实现并启动客户端（存储在 `fl_sim.py` 中）。首先，导入必要的库：

```py
import argparse
import tensorflow as tf
from tensorflow import keras
from sst_model import SSTModel, load_sst_data
import flwr as fl
```

我们添加一个命令行参数来指定客户端 ID，以便允许相同的客户端脚本被所有三个代理重用：

```py
parser = argparse.ArgumentParser()
parser.add_argument("client_id", type=int)
args = parser.parse_args()
client_id = args.client_id
NUM_CLIENTS = 3
```

然后我们加载 SST-2 数据集：

```py
(x_train,y_train), (x_test,y_test) = load_sst_data(client_id-1, NUM_CLIENTS)
```

注意，我们使用客户端 ID 从训练数据集中获取相应的分片。

接下来，我们创建模型和相关优化器以及损失对象，确保在哑输入上调用模型以初始化权重：

```py
sst_model = SSTModel()
sst_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.005, amsgrad=False),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.BinaryAccuracy()]
)
sst_input = keras.Input(shape=(), dtype=tf.string)
sst_model(sst_input)
```

### 实现 Flower 客户端

现在我们可以实现 Flower 客户端对象，该对象将在服务器之间传递模型参数。要实现客户端子类，我们必须定义三个函数：

+   `get_parameters(self, config)`: 返回模型参数值

+   `fit(self, parameters, config)`: 将本地模型的权重设置为接收到的参数，执行本地训练，并返回新的模型参数以及数据集大小和训练指标

+   `evaluate(self, parameters, config)`: 将本地模型的权重设置为接收到的参数，然后在验证/测试数据上评估模型，并返回性能指标

使用 `fl.client.NumPyClient` 作为超类允许我们利用 Keras 模型的 `get_weights` 和 `set_weights` 函数，这些函数将模型参数转换为 NumPy 数组的列表：

```py
class SSTClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return sst_model.get_weights()
    def fit(self, parameters, config):
        sst_model.set_weights(parameters)
        history = sst_model.fit(x_train, y_train, epochs=1)
        return sst_model.get_weights(), len(x_train), {'train_loss':history.history['loss'][0]}
```

`evaluate` 函数也被定义：

```py
    def evaluate(self, parameters, config):
        sst_model.set_weights(parameters)
        loss, acc = sst_model.evaluate(x_test, y_test, batch_size=64)
        return loss, len(x_train), {'val_acc':acc, 'val_loss':loss}
```

使用此客户端实现，我们最终可以使用以下行使用默认连接信息启动客户端：

```py
fl.client.start_numpy_client(server_address="[::]:8080", client=SSTClient())
```

### 创建 Flower 服务器

在运行 Flower 之前，我们需要创建一个脚本（`server.py`），该脚本将启动 Flower 服务器。我们开始导入必要的库和 `MAX_ROUNDS` 参数：

```py
import flwr as fl
import tensorflow as tf
from tensorflow import keras
from sst_model import SSTModel
MAX_ROUNDS = 3
```

因为我们希望在执行联邦学习后保存模型，所以我们创建了一个 flower FedAvg 策略的子类，并在聚合阶段的最后一步添加了一个保存模型的步骤：

```py
class SaveKerasModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        agg_weights = super().aggregate_fit(server_round, results, failures)
        if (server_round == MAX_ROUNDS):
            sst_model = SSTModel()
            sst_input = keras.Input(shape=(), dtype=tf.string)
            sst_model(sst_input)

            sst_model.set_weights(fl.common.parameters_to_ndarrays(agg_weights[0]))
            sst_model.save('final_agg_sst_model')
        return agg_weights
```

使用这种策略，我们可以运行以下行来启动服务器（通过 `config` 参数传递 `MAX_ROUNDS` 参数）：

```py
fl.server.start_server(strategy=SaveKerasModelStrategy(), config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS))
```

现在我们可以启动服务器和客户端，允许使用 Flower 进行 FL。

### 运行 Flower 示例

要启动服务器，首先运行 `server.py` 脚本。

每个客户端都可以通过在单独的终端窗口中运行以下命令来启动：

```py
python fl_sim.py 1
python fl_sim.py 2
python fl_sim.py 3
```

FL 最终的聚合模型将被保存在 `final_agg_sst_model` 目录中，作为一个 `SavedModel` 对象。

## 集成 STADLE 用于 SST-2

STADLE 与之前考察的 FL 框架不同，它提供了一个基于云的平台（STADLE Ops），用于处理聚合器的部署和 FL 流程的管理。因为服务器端的部署可以通过该平台完成，所以使用 STADLE 进行 FL 所需实现的只是客户端的实现。这种集成是通过创建一个客户端对象来完成的，该对象偶尔发送本地模型，并从上一轮返回聚合模型。为此，我们需要创建代理配置文件，并修改本地训练代码以与 STADLE 接口。

首先，我们创建代理的配置文件，如下所示：

```py
{
    "model_path": "./data/agent",
    "aggr_ip": "localhost",
    "reg_port": "8765",
    "token": "stadle12345",
    "base_model": {
        "model_fn": "SSTModel",
        "model_fn_src": "sst_model",
        "model_format": "Keras",
        "model_name": "Keras-SST-Model"
    }
}
```

这些参数的详细信息可以在 https://stadle-documentation.readthedocs.io/en/latest/documentation.html#configuration-of-agent 找到。请注意，这里列出的聚合器 IP 和注册端口号是占位符，在连接到 STADLE Ops 平台时将被修改。

接下来，我们修改本地训练代码以与 STADLE 一起工作。我们首先导入所需的库：

```py
import argparse
import tensorflow as tf
from tensorflow import keras
from sst_model import SSTModel, load_sst_data
from stadle import BasicClient
```

再次，我们添加一个命令行参数来指定代理应接收的训练数据分区：

```py
parser = argparse.ArgumentParser()
parser.add_argument("client_id", type=int)
args = parser.parse_args()
client_id = args.client_id
NUM_CLIENTS = 3
(x_train,y_train), (x_test,y_test) = load_sst_data(client_id-1, NUM_CLIENTS)
```

接下来，我们实例化一个 `BasicClient` 对象——这是 STADLE 客户端组件，用于处理本地训练过程与服务器端聚合器之间的通信。我们使用之前定义的配置文件来创建此客户端：

```py
stadle_client = BasicClient(config_file="config_agent.json", agent_name=f"sst_agent_{client_id}")
```

最后，我们实现 FL 训练循环。在每一轮中，客户端从上一轮（从基础模型开始）获取聚合模型，并在本地数据上进一步训练，然后再通过客户端将其发送回聚合器：

```py
for round in range(3):
    sst_model = stadle_client.wait_for_sg_model()
    history = sst_model.fit(x_train, y_train, epochs=1)
    loss = history.history['loss'][0]
    stadle_client.send_trained_model(sst_model, {'loss_training': loss})
stadle_client.disconnect()
```

`wait_for_sg_model` 函数从服务器返回最新的聚合模型，而 `send_trained_model` 函数将具有所需性能指标的本地训练模型发送到服务器。有关这些集成步骤的更多信息，请参阅 https://stadle-documentation.readthedocs.io/en/latest/usage.html#client-side-stadle-integration。

现在客户端实现完成后，我们可以使用 STADLE Ops 平台启动一个聚合器并启动一个 FL 流程。

### 创建 STADLE Ops 项目

首先，访问 stadle.ai 并创建一个新账户。一旦登录，你应该会被引导到 STADLE Ops 的项目信息页面：

![图 8.3 – STADLE Ops 中的项目信息页面

![图片 B18369_08_03.jpg]

图 8.3 – STADLE Ops 中的项目信息页面

点击**创建新项目**，然后填写项目信息并点击**创建项目**。项目信息页面应已更改以显示以下内容：

![图 8.4 – 新项目添加到项目信息页面

![图片 B18369_08_04.jpg]

图 8.4 – 新项目添加到项目信息页面

点击**启动聚合器**下方的加号图标以启动项目的新聚合器，然后在确认提示中点击**确定**。现在您可以导航到左侧的**仪表板**页面，页面看起来如下所示：

![图 8.5 – STADLE Ops 仪表板页面

![图片 B18369_08_05.jpg]

图 8.5 – STADLE Ops 仪表板页面

将`config_agent.json`文件中的`aggr_ip`和`reg_port`占位符参数值分别替换为**连接 IP 地址**和**连接端口**下的值。

这样，我们现在就可以开始 FL 训练过程了。

### 运行 STADLE 示例

第一步是将基础模型对象发送到服务器，使其能够反过来将模型分发给训练代理。这可以通过以下命令完成：

```py
stadle upload_model --config_path config_agent.json
```

一旦命令成功运行，STADLE Ops 仪表板上的**基础模型信息**部分应更新以显示模型信息。现在我们可以通过运行以下命令来启动三个代理：

```py
python fl_sim.py 1
python fl_sim.py 2
python fl_sim.py 3
```

经过三轮后，代理将终止，最终的聚合模型将在项目仪表板上显示，并以 Keras SavedModel 格式可供下载。建议查阅位于[`stadle.ai/user_guide/guide`](https://stadle.ai/user_guide/guide)的用户指南，以获取有关 STADLE Ops 平台各种功能的更多信息。

评估每个联邦学习框架产生的结果聚合模型，得出的结论相同——聚合模型的性能基本上与集中式训练模型的性能相匹配。正如在*第七章*的“数据集分布”部分所解释的，*模型聚合*，这通常是预期的结果。自然要问的是，当本地数据集不是独立同分布（IID）时，性能会受到怎样的影响——这是下一节的重点。

# 示例 – 在非 IID 数据上对图像分类模型进行联邦训练

在前面的例子中，我们考察了如何通过在联邦学习过程中在原始训练数据集（本地数据集）的不相交子集上训练多个客户端来将集中式深度学习问题转换为联邦学习的类似问题。这个本地数据集创建的一个关键点是，子集是通过随机采样创建的，导致所有本地数据集在原始数据集相同的分布下都是独立同分布的。因此，FedAvg 与本地训练场景相似的性能是可以预期的——每个客户端的模型在训练过程中本质上都有相同的局部最小值集合要移动，这使得所有本地训练都对全局目标有益。

回想一下，在*第七章*“模型聚合”中，我们探讨了 FedAvg 如何容易受到严重非独立同分布的本地数据集引起的训练目标发散的影响。为了探索 FedAvg 在变化非独立同分布严重程度上的性能，本例在从 CIFAR-10 数据集（位于[`www.cs.toronto.edu/~kriz/cifar.html`](https://www.cs.toronto.edu/~kriz/cifar.html)）中采样的构建的非独立同分布的本地数据集上训练了 VGG-16 模型（一个基于简单深度学习的图像分类模型）。CIFAR-10 是一个著名的简单图像分类数据集，包含 60,000 张图像，分为 10 个不同的类别；在 CIFAR-10 上训练的模型的目标是正确预测与输入图像相关联的类别。相对较低复杂性和作为基准数据集的普遍性使 CIFAR-10 成为探索 FedAvg 对非独立同分布数据的响应的理想选择。

重要提示

为了避免包含冗余的代码示例，本节重点介绍允许在 PyTorch 模型上使用非独立同分布的本地数据集执行联邦学习的关键代码行。建议在阅读本节之前，先阅读本章中“示例 – NLP 模型的联邦训练”部分中的示例，以便了解每个联邦学习框架所需的核心组件。本例的实现可以在本书的 GitHub 仓库中找到，完整内容位于[`github.com/PacktPublishing/Federated-Learning-with-Python`](https://github.com/PacktPublishing/Federated-Learning-with-Python)树/main/ch8/cv_code)，供参考使用。

本例的关键点是确定如何构建非独立同分布（non-IID）数据集。我们将通过改变训练数据集中每个类别的图像数量来改变每个本地数据集的类别标签分布。例如，一个偏向于汽车和鸟类的数据集可能包含 5,000 张汽车的图像，5,000 张鸟类的图像，以及每个其他类别 500 张图像。通过创建 10 个类别的三个不相交子集，并构建偏向这些类别的本地数据集，我们产生了三个本地数据集，其非独立同分布的严重程度与从未选择的类别中包含的图像数量成比例。

## 偏斜 CIFAR-10 数据集

我们首先将三个类别子集映射到客户端 ID，并设置从原始数据集中选取的类别（`sel_count`）和其他类别（`del_count`）的图像比例：

```py
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
class_id_map = {
    1: classes[:3],
    2: classes[3:6],
    3: classes[6:]
}
sel_count = 1.0, def_count = 0.2
```

然后我们从原始数据集中采样适当数量的图像，使用数据集中图像的索引来构建有偏的 CIFAR-10 子集：

```py
class_counts = int(def_count * 5000) * np.ones(len(classes))
for c in classes:
    if c in class_rank_map[self.rank]:
        class_counts[trainset.class_to_idx[c]] = int(sel_count * 5000)
class_counts_ref = np.copy(class_counts)
imbalanced_idx = []
for i,img in enumerate(trainset):
    c = img[1]
    if (class_counts[c] > 0):
        imbalanced_idx.append(i)
        class_counts[c] -= 1
trainset = torch.utils.data.Subset(trainset, imbalanced_idx)
```

然后使用有偏的训练集创建用于本地训练的有偏 `trainloader`。当我们提到对未来的训练数据进行偏差时，这就是运行的代码。

我们现在将演示如何使用不同的 FL 框架来运行这个非-IID FL 流程。请参阅上一节 *示例 - NLP 模型的联邦训练* 中的安装说明和框架特定实现，以了解本节中省略的基本概念。

## 集成 OpenFL 用于 CIFAR-10

与 Keras NLP 示例类似，我们首先在 `cifar_fl_dataset.py` 中为非-IID 的 CIFAR-10 数据集创建 `ShardDescriptor` 和 `DataInterface` 子类。为了适应新的数据集，只需要进行少数几个更改。

首先，我们修改 `self.data_by_type` 字典，以便存储修改后的 CIFAR 数据集：

```py
        train_dataset, val_dataset = self.load_cifar_data()
        self.data_by_type = {
            'train': train_dataset,
            'val': val_dataset
        }
```

`load_cifar_data` 函数使用 `torchvision` 加载训练和测试数据，然后根据传递给对象的排名对训练数据进行偏差。

由于数据元素的维度现在已知（CIFAR-10 图像的大小），我们还使用固定值修改了形状属性：

```py
    @property
    def sample_shape(self):
        return ["32", "32"]
    @property
    def target_shape(self):
        return ["10"] 
```

然后我们实现 `CifarFedDataset` 类，它是 `DataInterface` 类的子类。对于这个实现不需要进行重大修改；因此，我们现在可以使用带有 OpenFL 的有偏 CIFAR-10 数据集。

现在我们转向实际的 FL 流程实现 (`fl_sim.py`)。一个关键的区别是必须使用框架适配器来从 PyTorch 模型创建 `ModelInterface` 对象：

```py
model = vgg16()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=model, optimizer=optimizer, framework_plugin=framework_adapter)
```

唯一的另一个主要更改是修改传递给 `TaskInterface` 对象的培训和验证函数，以反映本地训练代码中这些函数的 PyTorch 实现。

最后一步是创建导演和使节使用的配置文件。导演配置中唯一必要的更改是更新 CIFAR-10 数据的 `sample_shape` 和 `target_shape`：

```py
settings:
  listen_host: localhost
  listen_port: 50051
  sample_shape: ["32","32"]
  target_shape: ["10"]
```

这个文件保存在 `director/director_config.yaml` 中。

使节配置文件除了更新对象和文件名之外不需要任何更改——目录结构应该如下所示：

+   `director`

    +   `director_config.yaml`

+   `experiment`

    +   `envoy_config_1.yaml`

    +   `envoy_config_2.yaml`

    +   `envoy_config_3.yaml`

    +   `cifar_fl_dataset.py`

    +   `fl_sim.py`

您可以参考 *在 *集成 OpenFL 用于 SST-2* 部分的 *运行 OpenFL 示例* 来运行此示例。

## 集成 IBM FL 用于 CIFAR-10

记住，IBM FL 需要保存训练过程中使用的模型版本。我们首先在 `create_saved_model.py` 中运行以下代码以创建保存的 VGG-16 PyTorch 模型：

```py
import torch
from torchvision.models import vgg16
model = vgg16()
torch.save(model, 'saved_vgg_model.pt')
```

接下来，我们为倾斜的 CIFAR-10 数据集创建`DataHandler`子类。唯一的核心更改是修改`load_and_preprocess_data`函数，以加载 CIFAR-10 数据并对训练集进行偏差。

下一步是创建启动聚合器和初始化各方时使用的配置 JSON 文件。聚合器配置（`agg_config.json`）无需进行重大更改，而各方配置的核心更改仅是修改模型信息以与 PyTorch 兼容：

```py
    "model": {
        "name": "PytorchFLModel",
        "path": "ibmfl.model.pytorch_fl_model",
        "spec": {
            "model-name": "vgg_model",
            "model_definition": "saved_vgg_model.pt",
            "optimizer": "optim.SGD",
            "criterion": "nn.CrossEntropyLoss"
        }
    },
```

由于广泛使用配置文件，`fl_sim.py`中负责启动各方代码基本上无需修改。

您可以参考*在 SST-2 中集成 IBM FL*部分的*运行 IBM FL 示例*来运行此示例。

## 集成 Flower 用于 CIFAR-10

在加载 CIFAR-10 数据并对训练数据进行偏差后，Flower 实现所需的核心更改是`NumPyClient`子类。与 Keras 示例不同，`get_parameters`和`set_parameters`方法依赖于 PyTorch 模型状态字典，并且更为复杂：

```py
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.numpy() for _, val in model.state_dict().items()]
    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict)
```

我们修改`fit`函数，使其与本地训练示例中的训练代码相匹配，并修改`evaluate`函数，使其与本地训练评估代码相类似。请注意，我们调用`self.set_parameters(parameters)`来更新本地模型实例的最新权重。

我们还在启动 Flower 客户端和服务器时将`grpc_max_message_length`参数设置为 1 GB，以适应更大的 VGG16 模型大小。客户端初始化函数现在是以下内容：

```py
fl.client.start_numpy_client(
    server_address="[::]:8080",
    client=CifarClient(),
    grpc_max_message_length=1024**3
)
```

最后，我们修改了`server.py`中的聚合器代码——我们之前用于在最后一轮结束时保存聚合模型的自定义策略需要修改以与 PyTorch 模型兼容：

```py
        if (server_round == MAX_ROUNDS):
            vgg_model = vgg16()

            np_weights = fl.common.parameters_to_ndarrays(agg_weights[0])
            params_dict = zip(vgg_model.state_dict().keys(), np_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

            torch.save(state_dict, "final_agg_vgg_model.pt")
```

使用此策略，我们可以运行以下行来启动服务器（在此处也添加了`grpc_max_message_length`参数）：

```py
fl.server.start_server(
    strategy=SavePyTorchModelStrategy(),
    config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS),
    grpc_max_message_length=1024**3
)
```

请参考*在 SST-2 中集成 Flower*部分的*运行 Flower 示例*来运行此示例。

## 集成 STADLE 用于 CIFAR-10

我们首先修改`config_agent.json`配置文件，以使用`torchvision`库中的 VGG16 模型：

```py
{
    "model_path": "./data/agent",
    "aggr_ip": "localhost",
    "reg_port": "8765",
    "token": "stadle12345",
    "base_model": {
        "model_fn": "vgg16",
        "model_fn_src": "torchvision.models",
        "model_format": "PyTorch",
        "model_name": "PyTorch-VGG-Model"
    }
}
```

要将 STADLE 集成到本地训练代码中，我们初始化`BasicClient`对象，并修改训练循环，每两个本地训练轮次发送一次本地模型，并等待新的聚合模型：

```py
    stadle_client = BasicClient(config_file="config_agent.json")
    for epoch in range(num_epochs):
        state_dict = stadle_client.wait_for_sg_model().state_dict()
        model.load_state_dict(state_dict)
        # Normal training code...
        if (epoch % 2 == 0):
            stadle_client.send_trained_model(model)
```

注意

位于[`github.com/PacktPublishing/Federated-Learning-with-Python`](https://github.com/PacktPublishing/Federated-Learning-with-Python)的代码包含此集成示例的完整实现，供参考。要启动聚合器并使用 CIFAR-10 STADLE 示例进行联邦学习，请参考*创建 STADLE Ops 项目*和*在 SST-2 中运行 STADLE 示例*部分。

测试构建的局部数据集中不同水平的偏差，应得出与*第七章*中“数据集分布”部分所陈述的相同结论——*模型聚合*对于非独立同分布情况——随着非独立同分布严重程度的增加，收敛速度和模型性能降低。本节的目标是在 SST-2 示例中理解每个联邦学习框架的基础上，突出与修改后的数据集上使用 PyTorch 模型所需的关键变化。结合本节和[`github.com/PacktPublishing/Federated-Learning-with-Python`](https://github.com/PacktPublishing/Federated-Learning-with-Python)中的代码示例，应有助于理解此示例集成。

# 摘要

在本章中，我们通过两个不同示例的背景，介绍了几个联邦学习（FL）框架。从第一个示例中，你学习了如何通过将数据分割成互不重叠的子集，将传统的集中式机器学习（ML）问题转化为类似的联邦学习场景。现在很清楚，随机采样会导致局部数据集是独立同分布（IID），这使得 FedAvg 能够达到与集中式等效的任何联邦学习框架相同的性能水平。

在第二个示例中，你了解到了一组数据集可以是非独立同分布（不同类别标签分布）的许多方法之一，并观察到了不同严重程度的非独立同分布数据集如何影响 FedAvg 的性能。我们鼓励你探索如何通过替代聚合方法在这些情况下改进 FedAvg。

这两个示例也应该让你对不同联邦学习框架工作时的一般趋势有了坚实的理解；虽然具体的实现级细节可能会改变（由于该领域的快速变化），但核心概念和实现细节将仍然是基础。

在下一章中，我们将继续转向联邦学习的商业应用方面，通过研究涉及联邦学习在特定领域应用的几个案例研究。
