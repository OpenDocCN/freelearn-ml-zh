

# 第五章：利用主动学习处理大数据

在本章中，我们将探讨如何使用**机器学习**（**ML**）来处理大数据，例如视频。开发视频分析的 ML 模型带来了一系列独特的挑战。视频本身很大，在高效处理方面提出了重大障碍。使用 ML 进行视频分析已成为许多行业和应用的越来越重要的技术。从依赖计算机视觉模型分析实时视频流中的道路条件的自动驾驶汽车，到能够自动检测可疑活动的安全系统，ML 正在改变视频数据所能实现的可能性。这些模型可以自动化耗时的人工分析并提供可扩展的视频理解。实现高效和可扩展的视频分析管道需要克服关键障碍，如大量数据标注。

我们将引导您了解一种前沿的 ML 方法，这将帮助您选择最具信息量的帧进行标注，从而提高分析的整体准确性和效率。

在本章中，我们将涵盖以下主题：

+   实现视频分析的 ML 模型

+   使用 Lightly 选择最具信息量的帧

# 技术要求

在本章中，您需要安装以下软件包：

```py
pip install ultralytics lightly docker encord
```

您还需要以下导入：

```py
import os
from IPython.display import display, Markdown
from ultralytics import YOLO
from pathlib import Path
import json
import contextlib
from typing import Iterator
import docker
from docker.models.containers import Container
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasetType
from lightly.openapi_generated.swagger_client import DatasourcePurpose
from encord.orm.cloud_integration import CloudIntegration
from encord.orm.dataset import AddPrivateDataResponse
from encord.user_client import EncordUserClient
from encord.orm.dataset import CreateDatasetResponse, StorageLocation
```

接下来，您需要创建一个 Lightly 账户并设置您的 API 令牌，如下所示：

```py
lightly_token = "your_lightly_token"
```

然后，您必须设置 Lightly 客户端以连接到 API：

```py
client = ApiWorkflowClient(token=lightly_token)
```

这个演示是在 AWS SageMaker 笔记本（`ml.g4dn.2xlarge 实例`）上运行的。这个实例有一个 GPU 和足够的内存来运行这个演示，因为我们需要访问 Docker，这在 Google Colab 中是不可能的。

如果您想将选定的帧发送到标注平台，还需要一个 Encord 账户（[`app.encord.com/`](https://app.encord.com/））。

# 实现视频分析的 ML 模型

活跃式机器学习（Active ML）在管理大数据项目方面发挥着变革性的作用，通过战略性地优化数据标注过程，从而以更少的手动努力提高模型性能。例如，在大型图像识别任务中，如从数百万社交媒体照片中识别特定对象，活跃学习可以通过定位最有可能提升模型能力的图像，显著减少工作量。同样，在**自然语言处理**（NLP）应用中，处理来自新闻文章、论坛和客户反馈等来源的大量文本数据，活跃式机器学习有助于选择性地标注对理解复杂语言细微差别或情感最有价值的文档。这种方法不仅简化了标注大规模数据集所需的工作量，而且确保了基于此类数据训练的模型更加准确、高效，并且能够处理大数据源中固有的现实世界变化。将这种方法扩展到视频分析中，由于视频内容中增加了复杂性和数据量，活跃式机器学习变得更加关键。活跃式机器学习可以用来识别对模型学习有显著贡献的关键帧或片段，在确保全面理解和分析视频数据的同时，大幅减少标注负担。这种在视频项目中采用的有针对性的方法不仅节省了资源，而且加速了复杂视频分析模型的发展，这些模型能够以更高的精度和效率执行活动识别、事件检测和情感分析等任务。

在本节中，我们将探讨如何利用活跃式机器学习开发用于视频的机器学习模型。机器学习视频分析系统需要实施策略来高效地整理视频帧。视频通常是大型文件，标注视频中的所有帧是不可能的。此外，根据**每秒帧数**（FPS）的速率，视频往往包含大量重复数据，如果对这样的数据进行标注，将浪费时间和金钱。常见的做法是以 1 FPS 的速率进行标注，而不是像 30 FPS 那样，以减少需要标注的帧数。然而，这并不是一个最佳解决方案，因为它会导致相似帧被标注，代表类别的平衡性失衡，以及所选数据缺乏多样性。此外，如果实施这样的流程，将被标注的许多帧可能一开始就无需标注，因为模型可能已经在某些帧上表现得很出色。因此，标注模型有信心且正确的帧是浪费时间和金钱。

换句话说，手动标注视频数据集中的所有帧对于机器学习来说是不切实际的，由于以下因素，活跃学习变得至关重要：

+   **数据量**：视频数据由大量帧组成，这使得全面的手动标注变得极其耗时且昂贵。例如，对 30 FPS 视频仅 10 分钟的每一帧进行标注就需要标注 18,000 张图片。

+   **冗余**：连续的视频帧高度冗余，因为它们包含几乎相同的内容。手动标注这种重复数据是不高效的。

+   **成本**：雇佣人工标注员仔细标注视频帧的费用会非常高昂，使得大多数项目在经济上不可行。仅 10 小时的视频标注费用就可能高达数千美元。

这就是主动机器学习非常有价值的地方。它通过识别最有可能提高模型性能的最具信息量的帧来优化标注工作，正如我们在前面的章节中看到的。人工标注员可以专注于这些高价值帧。通过将标注过程引导至最大化性能提升，需要手动标注的帧数大大减少，使得视频机器学习变得可行且经济高效。

总结来说，全面手动视频数据标注既不实际，在经济上也不可行。主动学习为标注提供了关键优化，使得模型能够以可行、经济和适应性强的方式分析视频。

现在，让我们通过一个使用常见主动机器学习工具**Lightly**（[https://www.lightly.ai/](https://www.lightly.ai/)）的实例来探讨现实世界的情况。

# 使用 Lightly 选择最具信息量的帧

在本节中，我们将使用一个名为**Lightly**的主动机器学习工具。Lightly 是一个数据整理工具，它配备了一个网络平台，使用户能够选择最优的样本子集以最大化模型精度。Lightly 的算法可以在不到 24 小时内处理大量数据，例如 1000 万张图片或 1 万段视频。

网络应用程序允许用户使用诸如锐度、亮度、对比度、文件大小等过滤器来探索他们的数据集。然后，他们可以使用这些过滤器来探索这些特征之间的相关性。

用户还可以在应用内搜索相似图像或对象，并查看嵌入（**主成分分析**（**PCA**），**T 分布随机近邻嵌入**（**TSNE**），和**均匀流形近似与投影**（**UMAP**））。嵌入是指由深度神经网络学习到的图像的向量表示。它们以允许分析图像之间相似性和关系的方式捕捉图像的视觉特征和语义。当图像通过卷积神经网络时，分类前的最后一层通常是图像特征的密集表示。这个密集层为每个图像输出一个具有数百或数千维度的向量。这个向量被称为嵌入。在嵌入空间中映射时，具有相似特征的图像的嵌入将更接近或相邻。具有非常不同特征的图像在嵌入空间中将更远。

有几种技术可以将这些高维嵌入可视化到二维或三维空间，以便于人类分析：

+   **PCA**：PCA 将嵌入的维度降低到二维或三维，以便于绘制。它将它们投影到捕捉最大变差的维度上。在 PCA 投影后，具有相似显著视觉特征的图像将更靠近。

+   **TSNE**：TSNE 是一种技术，在保持相似图像靠近和不相似图像分开的同时，将高维嵌入表示在较低维度。二维或三维映射试图模拟更高维空间中的局部结构。

+   **UMAP**：UMAP 是一种更近期的技术，在很多情况下比 TSNE 更好地保留投影中的全局数据结构。它将具有相似嵌入的图像映射到投影附近，将不相似的图像映射得更远。

嵌入以向量的形式捕捉图像特征和语义。PCA、TSNE 和 UMAP 等技术然后将这些高维向量投影到二维或三维，以便于可视化和分析图像之间的相似性关系。Lightly 应用利用这些投影来实现图像相似性搜索。

## 使用 Lightly 选择用于对象检测的最佳帧进行标记

为了选择最佳帧，我们需要进行一系列步骤。

### 数据集和预训练模型

在这个例子中，我们将使用一只狗追球的视频。出于演示目的，我们只会使用一个视频，如*图 5.1*所示。这个视频可以在本书的 GitHub 仓库中找到（[`github.com/PacktPublishing/Active-Machine-Learning-with-Python/blob/main/Chapter_5/videos/project_demo/dog_running_ball.mp4`](https://github.com/PacktPublishing/Active-Machine-Learning-with-Python/blob/main/Chapter_5/videos/project_demo/dog_running_ball.mp4)）：

![图 5.1 – 用于测试我们的 Lightly 演示的视频](img/B21789_05_1.jpg)

图 5.1 – 用于测试我们的 Lightly 演示的视频

在本节中，我们的主要目标是使用 Lightly 提供的不同采样策略选择这个视频中最具信息量的帧。

我们的视频 `dog_running_ball.mp4` 存储在名为 `videos` 的文件夹下的一个名为 `project_demo` 的子文件夹中。

一旦我们在 `videos/project_demo` 文件夹中有视频，下一步就是加载一个预训练的对象检测模型：

```py
Model = YOLO("yolov8x.pt")
```

这个来自 `ultralytics` 的预训练模型支持 80 个类别，我们可以使用以下命令进行可视化：

```py
model.namesg
```

这将返回以下输出：

```py
{0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 ...
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
...
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}
```

在我们这个例子中，我们知道我们正在处理一个狗追球的视频，因此我们将关注 `狗` 和 `运动` `球` 类别。

然后，我们必须为 Lightly 运行准备一些变量，例如预测输出文件夹的路径、任务的名称以及我们在这轮运行中想要关注的、以改进模型的重要类别（`狗` 和 `运动球`）：

```py
important_classes = {"dog": 16, " sports ball": 32}
classes = list(important_classes.values())
```

我们这里的类别是 `16` 和 `32`。

### 创建所需的 Lightly 文件

我们将把我们的推理预测保存到一个名为 `predictions` 的子文件夹中；我们的任务名称是 `yolov8_demo_dog_detection`：

```py
predictions_rooth_path = Path("predictions")
task_name = "yolov8_demo_dog_detection"
predictions_path = Path(predictions_rooth_path / task_name)
```

然后，我们需要创建 Lightly 将使用的所有配置文件：

+   `tasks.json` 文件，它指定了我们当前任务的名称。任务名称是预测文件夹中相应子文件夹的名称。

+   `schema.json` 文件，它允许 Lightly 了解预测的格式。

+   包含我们 `videos` 文件夹中视频名称的元数据 `schema.json` 文件。

创建这些配置文件的代码可以在本书 GitHub 仓库的*第五章*部分找到([`github.com/PacktPublishing/Active-Machine-Learning-with-Python/tree/main/Chapter_5`](https://github.com/PacktPublishing/Active-Machine-Learning-with-Python/tree/main/Chapter_5))。

我们现在可以使用我们之前加载的预训练模型运行对象检测推理。

### 推理

我们将以低置信度阈值 0.3 在我们的测试视频上运行对象检测推理，因为我们想要较低的置信度分数。代码已经设置好以处理子文件夹中的多个视频。然而，出于测试目的，我们只有一个视频。我们将跳过不属于重要类别的预测：

```py
videos = Path("videos/project_demo/").glob("*.mp4")
for video in videos:
    print(video)
    results = model.predict(video, conf=0.3)
    predictions = [result.boxes.data for result in results]
    number_of_frames = len(predictions)
    padding = len(str(number_of_frames))
    fname = video
    for idx, prediction in enumerate(predictions):
        populate_predictions_json_files(prediction, fname, padding)
```

`populate_predictions_json_files` 函数定义在这本书的 GitHub 仓库中的代码里([`github.com/PacktPublishing/Active-Machine-Learning-with-Python/tree/main/Chapter_5`](https://github.com/PacktPublishing/Active-Machine-Learning-with-Python/tree/main/Chapter_5))。

一旦我们运行了这段代码，我们将在 `predictions` 子文件夹中以 Lightly 支持的格式收到推理运行的输出，如图 *图 5**.2* 所示：

![图 5.2 – Lightly JSON 预测（快照）](img/B21789_05_2.jpg)

图 5.2 – Lightly JSON 预测（快照）

如果我们查看这些文件之一，JSON 文件包含的信息是推理的结果。这包括边界框的坐标，以及相应的类别 ID 和置信度分数，如图 *图 5.3* 所示：

![图 5.3 – Lightly 预测 JSON 文件格式，用于第一帧](img/B21789_05_3.jpg)

图 5.3 – Lightly 预测 JSON 文件格式，用于第一帧

现在，我们已经准备好安排活动 Lightly 机器学习运行。

### 安排活动机器学习运行

我们可以使用以下代码注册 Lightly 工作进程，该代码返回我们的 Lightly 工作进程的 ID：

注意

如果已存在具有此名称的工作进程，则将返回现有工作进程的 ID。

```py
worker_id = client.register_compute_worker(name="Demo")
```

Lightly 使用 Docker 通过 `lightly/worker:latest` 镜像运行活动机器学习。可以使用命令行拉取：

```py
docker pull lightly/worker:latest
```

Docker 是一个开源平台，用于在软件容器内开发、运输和运行应用程序。它允许将代码及其所有依赖项和库打包成一个标准化的单元，用于软件开发。使用 Docker 容器消除了由环境差异引起的不兼容性问题。简而言之，它使得在运行脚本时易于复制，因为 Docker 镜像中的环境已经配置了正确的已安装包。

接下来，我们需要安排我们的活动机器学习运行。这个过程涉及几个步骤：

1.  创建一个名为 `demo_dataset` 的 Lightly 数据集：

    ```py
    client.create_dataset(dataset_name="demo_dataset",
        dataset_type=DatasetType.VIDEOS)
    dataset_id = client.dataset_id
    ```

1.  使用 `project_demo` 项目作为数据输入或输出位置来设置数据源。在我们的案例中，我们使用本地存储选项 ([`docs.lightly.ai/docs/local-storage`](https://docs.lightly.ai/docs/local-storage)) 进行演示，但理想情况下，您应该使用云服务选项 ([`docs.lightly.ai/docs/cloud-storage`](https://docs.lightly.ai/docs/cloud-storage))，该选项使用 AWS、Azure 或 GCP：

    ```py
    client.set_local_config(
        relative_path="project_demo",
        purpose=DatasourcePurpose.INPUT
    )
    client.set_local_config(
        relative_path="project_demo",
        purpose=DatasourcePurpose.LIGHTLY
    )
    ```

1.  安排我们想要使用的活动机器学习运行计划：一个用于寻找不同对象的策略，一个用于平衡类别比例的策略，以及一个用于使用预测分数来表示对象的频率和最不自信的结果的策略。在这个例子中，我们采样了五个样本，并试图在`狗`和`运动``球`类别之间达到 50/50 的平衡：

    ```py
    scheduled_run_id = client.schedule_compute_worker_run(
        worker_config={},
        selection_config={
            "n_samples": 5,
            "strategies": [
                {
                    # strategy to find diverse objects
                    "input": {
                        "type": "EMBEDDINGS",
                        "task": task_name,
                    },
                    "strategy": {
                        "type": "DIVERSITY",
                    },
                },
                {
                    # strategy to balance the class ratios
                    "input": {
                        "type": "PREDICTIONS",
                        "name": "CLASS_DISTRIBUTION",
                        "task": task_name,
                    },
                    "strategy": {
                        "type": "BALANCE",
                        "target": {
                            dog: 0.50,
                            'sports ball': 0.50,
                        }
                    },
                },
                {
                    # strategy to use prediction score (Active Learning)
                    "input": {
                        "type": "SCORES",
                        "task": task_name,
                        "score": "object_frequency"
                    },
                    "strategy": {
                        "type": "WEIGHTS"
                    },
                },
                {
                    # strategy to use prediction score (Active Learning)
                    "input": {
                        "type": "SCORES",
                        "task": task_name,
                        "score": "objectness_least_confidence"
                    },
                    "strategy": {
                        "type": "WEIGHTS"
                    },
                },
            ],
        },
        lightly_config={},
        runs_on=['Demo'],
    )
    ```

1.  现在，组织本地文件，使其与 Lightly 期望的格式相匹配：

    ```py
    !mkdir lightly && mkdir lightly/project_demo && mkdir lightly/project_demo/.lightly
    !mv metadata lightly/project_demo/.lightly && mv predictions lightly/project_demo/.lightly
    ```

1.  当我们查看文件夹时，我们会看到以下结构：

![图 5.4 – 为 Lightly 本地存储组织的文件夹](img/B21789_05_4.jpg)

图 5.4 – 为 Lightly 本地存储组织的文件夹

现在，我们已经准备好开始运行。

### 启动工作进程和活动机器学习运行

我们需要创建一个函数，该函数将用于在 Docker 容器中挂载的已组织文件夹中启动工作进程：

```py
@contextlib.contextmanager
def start_worker(lightly_token, lightly_worker_id, image_name="lightly/worker:latest", WORKER_LABEL="Demo") -> Iterator[Container]:
    docker_client = docker.from_env()
    volumes = ["/home/user/videos:/input_mount",
              "/home/user/lightly:/lightly_mount"]
    container = docker_client.containers.run(
        image_name,
        f"token={lightly_token} worker.worker_id={lightly_worker_id}",
        detach=True,
        labels={"lightly_worker_label": WORKER_LABEL},
        volumes=volumes,
    )
    try:
        yield container
    finally:
        try:
            container.kill()
        except docker.errors.APIError:
            # if a container was killed from outside, we don't care
            pass
```

让我们开始运行：

```py
with start_worker(lightly_token, lightly_worker_id=worker_id):
    print('Worker running ...')
    last_run_info = None
    no_update_count = 0
    while True:
        run_info = client.get_compute_worker_run_info(
            scheduled_run_id=scheduled_run_id
        )
        print(run_info)
        if run_info.in_end_state():
            assert run_info.ended_successfully(), 
                "Run did not end successfully"
            break
        if run_info != last_run_info:
            no_update_count = 0
        else:
            no_update_count += 1
            if no_update_count >= 10000:
                raise RuntimeError(
                    f"Test timout: no run_info update\n"
                    f"last_run_info: {str(last_run_info)}, 
                        run_info: {str(run_info)}"
                )
        last_run_info = run_info
```

我们可以在 Lightly 平台上检查我们运行的进度，如图 *图 5.5* 所示，以及通过查看前一段代码的输出：

![图 5.5 – Lightly 活动机器学习运行的视图](img/B21789_05_5.jpg)

图 5.5 – Lightly 活动机器学习运行的视图

完成后，我们可以访问运行详细的报告和所有必要的日志，例如内存日志和默认日志。我们还可以在 Lightly 网络应用程序中查看我们的数据集（[`app.lightly.ai/dataset`](https://app.lightly.ai/dataset)）。

让我们探索我们的结果。在 *图 5*.6 中，我们可以看到我们有几个新的子文件夹，包括 `frames` 和 `crops`。这些文件夹包含所选的帧和裁剪（裁剪边界框）：

![图 5.6 – Lightly 活动机器学习运行后的结果文件夹](img/B21789_05_6.jpg)

图 5.6 – Lightly 活动机器学习运行后的结果文件夹

让我们可视化所选的帧：

```py
most_informative_frames = glob('lightly/project_demo/.lightly/frames/*.png')
for img_path in most_informative_frames:
    plt.imshow(Image.open(img_path))
    plt.show()
```

这返回了 *图 5*.7 中显示的图像：

![图 5.7 – Lightly 活动机器学习运行选择的五个最有信息量的帧](img/B21789_05_7.jpg)

图 5.7 – Lightly 活动机器学习运行选择的五个最有信息量的帧

我们还可以探索报告，它为我们提供了关于我们新样本子集的大量信息。

报告保存在 `lightly/project_demo/.lightly/runs/run_id` 下。我们可以查看运行 ID，然后像 *图 5*.8 中所示的那样在本地复制 `report.pdf` 文件：

![图 5.8 – 复制报告文档](img/B21789_05_8.jpg)

图 5.8 – 复制报告文档

报告中信息量很大，所以我们只关注某些部分。

首先，让我们快速看一下我们开始时的内容和最终的结果。如图 *图 5*.9 所示，我们最初有 290 帧和一个视频，只创建了一个包含五个帧的子集，这与我们请求的五个样本相对应。请注意，可以通过 `n_samples` 选择样本比例，也可以通过 `proportionSamples` 选择数据百分比。因此，要选择 30% 的数据，我们可以设置以下内容：

```py
"proportionSamples": 0.30
```

我们可以这样做，而不是运行以下操作：

```py
"n_samples": 5
```

这是输出：

![图 5.9 – Lightly 报告 – 数据集信息](img/B21789_05_9.jpg)

图 5.9 – Lightly 报告 – 数据集信息

现在，让我们检查嵌入图，如图 *图 5*.10 所示。在检查 UMAP 和 PCA 嵌入图后，我们会看到没有明显的聚类，这表明帧之间缺乏一致性。这种不一致性可以归因于视频的动态拍摄条件，包括不同的角度、与狗的距离以及由于在移动和跟随狗的过程中遇到有阴影和无阴影区域而造成的照明变化。这些因素导致在帧中捕获了多样化的视觉输入：

![图 5.10 – Lightly 报告 – 嵌入图](img/B21789_05_10.jpg)

图 5.10 – 轻量级报告 – 嵌入图

通过 *图 5*.11，我们更好地理解了活动机器学习算法选择的帧。我们可以看到它很好地从嵌入空间的不同位置选择了帧：

![图 5.11 – 轻量级报告 – 嵌入图选定帧](img/B21789_05_11.jpg)

图 5.11 – 轻量级报告 – 嵌入图选定帧

在 *图 5.12* 中，我们可以观察到类别分布已经得到了有效平衡：`狗` 类占 54.5%，而 `运动球` 类占 45.5% 的选择，与我们的 50/50 类别平衡比率非常接近。这种平衡得益于我们配置的类别平衡比率。尽管如此，达到这种平衡往往具有挑战性，尤其是在数据集中一个类别显著多于其他类别时：

![图 5.12 – 轻量级报告 – 类别分布](img/B21789_05_12.jpg)

图 5.12 – 轻量级报告 – 类别分布

现在，我们可以探索一些模型预测结果，如图 *图 5.13* 所示。在这里，我们有不同检测数量的预测示例。总体而言，模型表现出强大的性能，始终在帧内识别出狗。然而，它似乎在检测球方面遇到了更多困难。值得注意的是，球在其中一个样本帧中被识别出来，这是一个令人鼓舞的迹象。这种检测准确性的差异很可能是由于测试中使用的球的本性；它偏离了原始模型训练时使用的传统运动球，如网球或足球。这个上下文有助于解释观察到的模型性能变化，可以通过标记球并重新训练模型来修复：

![图 5.13 – 轻量级报告 – 模型预测示例](img/B21789_05_13.jpg)

图 5.13 – 轻量级报告 – 模型预测示例

最后，*图 5.14* 展示了视频中的选定帧。每个选定的帧对应一条垂直线，我们可以确认有五条线对应我们选定的五个帧：

![图 5.14 – 轻量级报告 – 视频采样密度](img/B21789_05_14.jpg)

图 5.14 – 轻量级报告 – 视频采样密度

现在我们有了希望标记的最具信息量的帧，我们可以将它们发送到项目中使用的注释平台，例如 Encord Annotate ([`docs.encord.com/docs/annotate-overview`](https://docs.encord.com/docs/annotate-overview)) 或 Roboflow ([`roboflow.com/`](https://roboflow.com/))，如 *第三章* 中所述，*管理闭环中的人类*。在这个例子中，我们将使用 Encord Annotate，因为它提供了一个将选定帧可视化为视频的功能。

在运行此代码之前，您需要根据他们提供的文档创建 Encord SSH 私钥 ([`docs.encord.com/reference/authentication-sdk`](https://docs.encord.com/reference/authentication-sdk))。

然后，您可以使用以下代码进行验证：

```py
encord_private_key = "-----BEGIN OPENSSH PRIVATE KEY-----{your_key}-----END OPENSSH PRIVATE KEY-----"
user_client = EncordUserClient.create_with_ssh_private_key(
    encord_private_key)
```

下一步是创建一个名为我们项目名称的 Encord 数据集（[`docs.encord.com/docs/annotate-datasets`](https://docs.encord.com/docs/annotate-datasets)） – 即 `project_demo`。这个 Encord 数据集是我们想要标注的数据：

```py
print('\nCreating Encord Dataset...')
encord_dataset_created = user_client.create_dataset(
    'project_demo', StorageLocation.CORD_STORAGE
)
```

这将返回一个字典，包含创建的数据集的 `title`、`type`、`dataset_hash`、`user_hash` 和 `backing_folder_uuid` 值。在这里，我们使用 Encord 的云存储，但你也可以使用自定义云存储。例如，如果你使用 AWS S3，那么你可以使用 `StorageLocation.AWS` 代替。

现在，我们可以查询数据集哈希，因为在上传图像时将需要它：

```py
dataset_hash = encord_dataset_created.dataset_hash
```

接下来，我们可以用我们选择的帧填充数据集：

```py
dataset = user_client.get_dataset(dataset_hash)
image_files = sorted(
    [
        p.as_posix()
        for p in Path("lightly/project_demo/.lightly/frames").iterdir()
        if p.suffix in {".jpg", ".png"}
    ]
)
dataset.create_image_group(image_files, create_video=True)
```

这将返回一个包含上传数据的 `data_hash`、`backing_item_uuid` 和 `title` 值的字典。

注意，我们使用了 `create_video=True`，这样我们就可以从图像组创建一个压缩视频；这些被称为图像序列（[`docs.encord.com/docs/annotate-supported-data#image-sequences`](https://docs.encord.com/docs/annotate-supported-data#image-sequences)）。当可视化帧时，这有助于保持视频的时间上下文，并且通常对标注者非常有帮助。它还允许标注者使用诸如自动标注（[`docs.encord.com/docs/automated-labeling`](https://docs.encord.com/docs/automated-labeling)）等特性，包括插值。这通过自动估计两个手动标注帧之间的标签位置，大大加快了标注过程。

在这一点上，我们可以在 Encord 网络应用程序的 `Index/Datasets` 部分查看我们的数据集（[`app.encord.com/datasets`](https://app.encord.com/datasets)），如图 *图 5*.15 所示。我们可以观察到图像被保存为 `img_sequence`，这意味着它们将以视频的形式显示：

![图 5.15 – 使用我们选择的五个帧保存为图像序列的 Encord 数据集](img/B21789_05_15.jpg)

图 5.15 – 使用我们选择的五个帧保存为图像序列的 Encord 数据集

在 Encord 中，我们定义了用于此标注项目的本体（[`docs.encord.com/reference/ontologies-sdk`](https://docs.encord.com/reference/ontologies-sdk)），如图 *图 5*.16 所示。我们在 *第三章* 的 *设计交互式学习系统和工作流程* 部分介绍了本体的概念：

![图 5.16 – 在 Encord 中创建的包含我们的两个类别，狗和球的本体](img/B21789_05_16.jpg)

图 5.16 – 在 Encord 中创建的包含我们的两个类别，狗和球的本体

从 *图 5*.16 中可视化的页面，我们可以复制本体 ID 并使用它来创建 Encord Annotate 项目（[`docs.encord.com/docs/annotate-overview`](https://docs.encord.com/docs/annotate-overview)）：

```py
project_hash = user_client.create_project(
        project_title='project_demo',
        dataset_hashes=[dataset_hash],
        ontology_hash='a0e16402-a5b4-417e-a4b1-7871ed386362')
```

我们将看到以下内容：

![图 5.17 – Encord Annotate 项目中的 project_demo 样本](img/B21789_05_17.jpg)

图 5.17 – Encord Annotate 项目中的 project_demo 样本

我们的数据现在已准备好进行标注。我们可以在 Web 应用的 `Annote/标注项目` 部分查看我们的 Encord Annotate 项目（[`app.encord.com/projects`](https://app.encord.com/projects)）。然后，在 `project_demo` 标注项目中，我们可以将所选帧作为视频查看，如图 *5.18* 所示：

![图 5.18 – Encord 中所选帧图像序列的标注视图](img/B21789_05_18.jpg)

图 5.18 – Encord 中所选帧图像序列的标注视图

在 *图 5.18* 所示的视图中，我们可以看到所有帧都作为视频呈现，用户可以使用滑块来浏览帧。还有我们在本体论中的 **类别** 部分定义的两个类别（狗和球），标注者可以选择来标注帧，如图 *5.19* 所示：

![图 5.19 – 在所选帧上标注的示例](img/B21789_05_19.jpg)

图 5.19 – 在所选帧上标注的示例

从这个页面，标注者可以使用之前提到的自动标注功能轻松标注对象；完成标注后，他们可以提交结果以供审查。

您也可以使用以下代码访问所选帧：

```py
client_lightly_dataset = ApiWorkflowClient(
    token=lightly_token, dataset_id=dataset_id)
filenames_and_read_urls \ 
    client_lightly_dataset.export_filenames_and_read_urls_by_tag_name(
        tag_name="initial-tag"  # name of the tag in the dataset
)
print(f'There are {len(filenames_and_read_urls)} frames')
```

这将返回以下输出：

```py
There are 5 frames
```

这是正确的！现在，让我们打印结果，以便我们更好地理解：

```py
print(filenames_and_read_urls)
```

这将返回以下输出：

```py
[{'fileName': 'dog_running_ball-024-mp4.png', 'readUrl': 'https://api.lightly.ai/v1/datasets/...', 'datasourceUrl': 'project_demo/.lightly/frames/dog_running_ball-024-mp4.png'}, .... {'fileName': 'dog_running_ball-180-mp4.png', 'readUrl': 'https://api.lightly.ai/v1/datasets/...', 'datasourceUrl': 'project_demo/.lightly/frames/dog_running_ball-180-mp4.png'}]
```

这些 URL 可以用来将所选帧发送到用于项目的标注平台。请注意，由于我们在演示中使用本地存储，数据不容易被标注平台和云服务访问，因此应使用云服务。本地数据也可以通过在 Lightly 中提供 `localhost` 来可视化（[`docs.lightly.ai/docs/local-storage#optional-view-local-data-in-lightly-platform`](https://docs.lightly.ai/docs/local-storage#optional-view-local-data-in-lightly-platform)）。

在本节中，我们使用 Lightly 通过几种策略选择了一只狗追球的测试视频中最具信息量的帧。这些策略包括寻找多样化的对象、平衡类别比率、使用预测分数来计算对象频率，以及考虑最不自信的结果。Lightly 还有很多其他功能来改进这些结果，并且有很好的组织文档（[`docs.lightly.ai/`](https://docs.lightly.ai/)）。

接下来，我们将讨论如何使用 Lightly 进行**自监督学习**（**SSL**）。

## 带有主动机器学习的 SSL

Lightly 提供了其他有用的功能，例如**SSL**，它允许用户在嵌入图像之前，在他们的数据上微调 SSL 模型。SSL 算法利用未标记图像或视频中的结构和上下文来生成替代监督信号，使模型能够自行发现强大的视觉表示。例如，模型可能被训练在微调下游任务之前，以识别空间模式、着色、旋转或时间顺序作为先验目标。本质上，SSL 允许模型利用大量的未标记视频和图像来发现数据本身中的有用特征和模式，避免对人工标记的依赖，这在规模上可能是不切实际的。模型通过精心设计的先验任务自动监督其特征学习，同时利用视频中的时间连续性等特性。因此，当开发特定领域的模型时，例如医学视频，这个 Lightly 功能可以非常有益。额外的训练步骤提高了嵌入的质量，因为模型可以在不要求更多标记的情况下适应特定领域。

启用 SSL 很简单，只需在我们的`worker_config`和`lightly_config`子字典中添加几行代码，这两个子字典都是`scheduled_run_id`字典的一部分：

```py
scheduled_run_id = client.schedule_compute_worker_run(
    worker_config={
        "enable_training": True
    },
    selection_config={
        "n_samples": 5,
        "strategies": [....], # same as before
    },
    lightly_config={
        'loader': {
            'num_workers': -1,
        },
        'trainer': {
            'max_epochs': 10,
            "gpus": 0, # cpu
        },
    },
    runs_on=['Demo'],
)
```

在这里，我们配置了主动的机器学习运行，在生成嵌入之前，在 CPU 上执行 10 个时期的 SSL 训练。

现在，让我们看看我们的输出。被选中的帧大多与之前我们选择的帧不同——即`dog_running_ball-024-mp4.png`、`dog_running_ball-101-mp4.png`、`dog_running_ball-033-mp4.png`、`dog_running_ball-224-mp4.png`和`dog_running_ball-049-mp4.png`——与`dog_running_ball-024-mp4.png`、`dog_running_ball-183-mp4.png`、`dog_running_ball-151-mp4.png`、`dog_running_ball-194-mp4.png`和`dog_running_ball-180-mp4.png`相比。

因此，只有帧`024`再次被选中。*图 5.20*显示了五个最有信息量的帧：

![图 5.20 – 通过 Lightly 主动机器学习 SSL 运行选择五个最有信息量的帧](img/B21789_05_20.jpg)

图 5.20 – 通过 Lightly 主动机器学习 SSL 运行选择五个最有信息量的帧

在 SSL 和非 SSL 运行中都被选中的帧在边缘被突出显示。

*图 5.20*显示，添加 SSL 步骤明显改变了帧的选择标准。主要的是，SSL 之后选择的帧往往显示狗在更远的距离，与未使用 SSL 选择的帧形成鲜明对比，后者主要是特写镜头，展示了狗拿着球。这种转变突出了 SSL 对模型关注点和帧选择偏好的影响：

![图 5.21 – Lightly 报告 – 比较非 SSL 运行（顶部图像）和 SSL 运行（底部图像）之间的视频采样密度](img/B21789_05_21.jpg)

图 5.21 – 简要报告 – 比较非 SSL 运行（顶部图像）和 SSL 运行（底部图像）之间的视频采样密度

在检查*图 5.21**.21*中显示的新嵌入图后，很明显，嵌入模型在聚类帧方面表现更好。尽管有这种改进，但聚类仍然没有明确界定，这表明在 SSL 训练中增加 epoch 的数量可以进一步细化这一方面：

![图 5.22 – SSL 活跃机器学习运行的新嵌入图](img/B21789_05_22.jpg)

图 5.22 – SSL 活跃机器学习运行的新嵌入图

将 Lightly SSL 功能纳入机器学习管道是一个简单添加，可以为特定领域的数据提供显著的好处。通过利用这项高级技术，我们观察到在测试中经过 SSL 处理的模型生成的嵌入得到了显著改善。这种增强不仅提高了整体性能，还确保了管道针对正在处理的数据的独特特征进行了优化。

# 摘要

在本章中，我们学习了如何使用 Lightly 通过不同的采样策略高效地选择视频中最具信息量的帧，以改进目标检测模型。我们还看到了如何将这些选定的帧发送到标记平台 Encord，从而完成一个端到端的使用案例。最后，我们探讨了如何通过将 SSL 步骤纳入活跃的机器学习管道来进一步优化采样。

在接下来的工作中，我们的重点将转向探索如何有效地评估、监控和测试活跃的机器学习管道。这一步骤对于确保管道在整个部署过程中保持稳健和可靠至关重要。通过实施全面的评估策略，我们可以评估管道相对于预定义的指标和基准的性能。此外，持续的监控将使我们能够识别任何潜在的问题或与预期行为不符的情况，从而使我们能够采取主动措施以保持最佳性能。

此外，对活跃的机器学习管道进行严格的测试对于验证其功能并验证其准确性至关重要。通过系统化的测试程序，我们可以确保管道在各种场景和输入条件下表现一致。这将涉及设计并执行各种测试用例，涵盖广泛潜在的使用案例和边缘场景。

通过彻底评估、监控和测试活跃的机器学习管道，我们可以对其可靠性和性能建立信心。这个稳健的框架将使我们能够做出明智的决定，并从处理的数据中获得有价值的见解，最终导致特定领域领域的成果改进和决策能力增强。
