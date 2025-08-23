# 4

# 使用 Python 实现联邦学习服务器

联邦学习（**federated learning**，**FL**）系统的服务器端实现对于实现真正的 FL 启用应用至关重要。我们在前一章中讨论了基本系统架构和流程。在本章中，我们将讨论更多实际实现，以便您可以创建一个简单的 FL 系统服务器和聚合器，各种**机器学习**（**ML**）应用可以连接并在此测试。

本章描述了在*第三章*“联邦学习系统的工作原理”中讨论的 FL 服务器端组件的实际实现方面。基于对 FL 系统整个工作流程的理解，您将能够进一步使用这里和 GitHub 上提供的示例代码来实现。一旦您理解了使用示例代码的基本实现原则，根据您自己的设计增强 FL 服务器功能将是一个有趣的部分。

在本章中，我们将涵盖以下主题：

+   聚合器的主要软件组件

+   实现 FL 服务器端功能

+   使用状态管理维护聚合模型

+   聚合本地模型

+   运行 FL 服务器

+   实现和运行数据库服务器

+   FL 服务器的潜在增强

# 技术要求

本章中介绍的所有代码文件都可以在 GitHub 上找到：[`github.com/tie-set/simple-fl`](https://github.com/tie-set/simple-fl)。

重要提示

您可以使用代码文件用于个人或教育目的。然而，请注意，我们不会支持商业部署，并且不会对使用代码造成的任何错误、问题或损害负责。

# 聚合器和数据库的主要软件组件

在前一章中介绍了具有 FL 服务器的聚合器架构。在这里，我们将介绍实现 FL 系统基本功能的代码。聚合器和数据库端的 Python 软件组件列在`fl_main`的`aggregator`目录中，以及`lib/util`和`pseudodb`文件夹中，如*图 4.1*所示：

![图 4.1 – 聚合器的 Python 软件组件以及内部库和伪数据库]

](img/B18369_04_01.jpg)

图 4.1 – 聚合器的 Python 软件组件以及内部库和伪数据库

以下是对聚合器中 Python 代码文件的简要描述。

## 聚合器端代码

在本节中，我们将涉及与 FL 服务器线程、FL 状态管理以及模型聚合本身相关的聚合器端的主要 Python 文件。这些聚合器端的代码文件位于`aggregator`文件夹中。仓库中的代码仅捕获模型聚合的视角，而不是创建全面 FL 平台的整个工程方面。

### FL 服务器代码（server_th.py）

这是实现联邦学习过程整个基本流程的主要代码，包括聚合器本身、代理和数据库之间的通信过程，以及协调代理参与和 ML 模型的聚合。它还初始化从第一个连接的代理发送的全球集群模型。它管理接收本地模型和集群模型合成例程，在收集足够的本地模型后形成集群全局模型。

### 联邦学习状态管理器（state_manager.py）

状态管理器缓冲本地模型和集群模型数据，这些数据对于聚合过程是必需的。当聚合器从代理接收本地模型时，缓冲区将被填充，在进入联邦学习过程的下一轮时将被清除。聚合标准检查函数也定义在此文件中。

### 聚合代码（aggregation.py）

聚合 Python 代码将列出聚合模型的基本算法。在本章中使用的代码示例中，我们只介绍称为**联邦平均**（**FedAvg**）的平均方法，它考虑了本地数据集的大小来平均收集的本地模型的权重，从而生成集群全局模型。

## lib/util 代码

内部库的 Python 文件（`communication_handler.py`、`data_struc.py`、`helpers.py`、`messengers.py`和`states.py`）将在*附录*、*探索内部库*中进行解释。

## 数据库端代码

数据库端代码包括伪数据库和位于`pseudodb`文件夹中的 SQLite 数据库 Python 代码文件。伪数据库代码运行一个服务器以接收来自聚合器的消息，并将它们处理为可用于联邦学习过程的 ML 模型数据。

### 伪数据库代码（pseudo_db.py）

伪数据库 Python 代码的功能是接受来自聚合器与本地和全球集群模型相关的消息，并将信息推送到数据库。它还将在本地文件系统中保存 ML 模型二进制文件。

### SQLite 数据库代码（sqlite_db.py）

SQLite 数据库 Python 代码在指定的路径创建实际的 SQLite 数据库。它还具有将有关本地和全球集群模型的数据条目插入数据库的功能。

现在聚合器和数据库端软件组件已经定义好了，让我们继续进行聚合器的配置。

## 面向聚合器的配置

以下代码是聚合器端配置参数的示例，这些参数定义在`config_aggregator.json`文件中，该文件位于`setups`文件夹中：

```py
{
    "aggr_ip": "localhost",
    "db_ip": "localhost",
    "reg_socket": "8765",
    "exch_socket": "7890",
    "recv_socket": "4321",
    "db_socket": "9017",
    "round_interval": 5,
    "aggregation_threshold": 1.0,
    "polling": 1
}
```

参数包括聚合器的 IP（FL 服务器的 IP）、数据库服务器的 IP 以及数据库和代理的各种端口号。轮询间隔是检查聚合标准的时间间隔，聚合阈值定义了开始聚合过程所需的收集的本地 ML 模型的百分比。轮询标志与是否利用`polling`方法进行聚合器和代理之间的通信有关。

现在我们已经介绍了聚合器侧的配置文件的概念，让我们继续了解代码的设计和实现。

# 实现 FL 服务器端功能

在本节中，我们将通过实际的代码示例来解释如何使用 FL 服务器系统实现聚合器的第一个版本，这些示例代码位于`aggregator`目录下的`server_th.py`文件中。通过这种方式，您将了解 FL 服务器系统的核心功能以及它们的实现方式，以便您能够进一步扩展更多功能。因此，我们只涵盖进行简单 FL 过程所必需的重要和核心功能。潜在的提升将在本章后面的部分列出，即*FL 服务器的潜在提升*。

`server_th.py`处理与 FL 服务器端相关的所有基本功能方面，所以让我们在下一节中看看。

## 导入 FL 服务器的库

FL 服务器端的代码从导入必要的库开始。特别是，`lib.util`处理使 FL 实现变得容易的基本支持功能。代码的详细信息可以在 GitHub 仓库中找到。

服务器代码导入`StateManager`和`Aggregator`用于 FL 过程。关于状态管理器和聚合的代码将在本章后面的部分讨论，即*使用状态管理器维护聚合模型*和*聚合本地模型*。

下面是导入必要库的代码：

```py
import asyncio, logging, time, numpy as np
from typing import List, Dict, Any
from fl_main.lib.util.communication_handler import init_fl_server, send, send_websocket, receive 
from fl_main.lib.util.data_struc import convert_LDict_to_Dict
from fl_main.lib.util.helpers import read_config, set_config_file
from fl_main.lib.util.messengers import generate_db_push_message, generate_ack_message, generate_cluster_model_dist_message, generate_agent_participation_confirmation_message
from fl_main.lib.util.states import ParticipateMSGLocation, ModelUpMSGLocation, PollingMSGLocation, ModelType, AgentMsgType
from .state_manager import StateManager
from .aggregation import Aggregator
```

在我们导入必要的库之后，让我们继续设计一个 FL `Server`类。

## 定义 FL 服务器类

在实践中，定义`Server`类是明智的，使用它可以创建一个具有在*第三章*，*联邦学习系统的工作原理*中讨论的功能的 FL 服务器实例，如下所示：

```py
class Server:
    """
    FL Server class defining the functionalities of 
    agent registration, global model synthesis, and
    handling mechanisms of messages by agents. 
    """
```

再次强调，`server`类主要提供代理注册和全局模型合成的功能，并处理上传的本地模型和来自代理的轮询消息的机制。它还充当聚合器和数据库以及聚合器和代理之间的接口。

FL 服务器类的功能现在很清晰——接下来是初始化和配置服务器。

## 初始化 FL 服务器

在 `__init__` 构造函数内部的以下代码是 `Server` 实例初始化过程的示例。

```py
def __init__(self):
    config_file = set_config_file("aggregator")
    self.config = read_config(config_file)
    self.sm = StateManager()
    self.agg = Aggregator(self.sm)
    self.aggr_ip = self.config['aggr_ip']
    self.reg_socket = self.config['reg_socket']
    self.recv_socket = self.config['recv_socket']
    self.exch_socket = self.config['exch_socket']
    self.db_ip = self.config['db_ip']
    self.db_socket = self.config['db_socket']
    self.round_interval = self.config['round_interval']
    self.is_polling = bool(self.config['polling'])
    self.sm.agg_threshold = 
                     self.config['aggregation_threshold']
```

然后，`self.config` 存储了前面代码块中讨论的 `config_aggregator.json` 文件中的信息。

`self.sm` 和 `self.agg` 分别是下面讨论的状态管理类和聚合类实例。

`self.aggr_ip` 从聚合器的配置文件中读取一个 IP 地址。

然后，将设置 `reg_socket` 和 `recv_socket`，其中 `reg_socket` 用于代理注册自身以及存储为 `self.aggr_ip` 的聚合器 IP 地址，而 `recv_socket` 用于从代理接收本地模型，以及存储为 `self.aggr_ip` 的聚合器 IP 地址。在这个示例代码中，`reg_socket` 和 `recv_socket` 都可以读取自聚合器的配置文件。

`exch_socket` 是用于将全局模型连同代理的 IP 地址一起发送回代理的端口号，该端口号在初始化过程中通过配置参数进行初始化。

随后，将配置连接到数据库服务器的信息，其中 `dp_ip` 和 `db_socket` 分别是数据库服务器的 IP 地址和端口号，所有这些信息都是从 `config_aggregator.json` 文件中读取的。

`round_interval` 是一个检查是否满足启动模型聚合过程聚合标准的时间间隔。

`is_polling` 标志与是否使用代理端的 `polling` 方法有关。轮询标志必须与代理端配置文件中使用的标志相同。

`agg_threshold` 也是在 `ready_for_local_aggregation` 函数中使用的超过收集到的本地模型数量的百分比，如果收集到的模型百分比等于或超过 `agg_threshold`，则 FL 服务器开始本地模型的聚合过程。

在这个示例代码中，`self.round_interval` 和 `self.agg_threshold` 都是从配置文件中读取的。

现在配置已经设置好了，我们将讨论如何注册尝试参与联邦学习（FL）过程的代理。

## 代理注册函数

在本节中，描述了简化和异步的 `register` 函数，用于接收指定模型结构的参与消息，并返回用于未来模型交换的套接字信息。它还向代理发送欢迎消息作为响应。

代理的注册过程在下面的示例代码中进行了描述：

```py
async def register(self, websocket: str, path):        
    msg = await receive(websocket)
    es = self._get_exch_socket(msg)
    agent_nm = msg[int(ParticipateMSGLocation.agent_name)]
    agent_id = msg[int(ParticipateMSGLocation.agent_id)]
    ip = msg[int(ParticipateMSGLocation.agent_ip)]
    id, es = self.sm.add_agent(agent_nm, agent_id, ip, es)
    if self.sm.round == 0:
        await self._initialize_fl(msg)
    await self._send_updated_global_model( \
        websocket, id, es)
```

在这个示例代码中，从代理接收到的消息，在此定义为 `msg`，是通过从 `communication_handler` 代码导入的 `receive` 函数进行解码的。

特别是，`self.sm.add_agent(agent_name, agent_id, addr, es)` 函数接收代理名称、代理 ID、代理 IP 地址以及包含在 `msg` 消息中的 `exch_socket` 号码，以便接受来自该代理的消息，即使代理暂时断开连接然后再次连接。

之后，注册函数检查是否应该根据 `self.sm.round` 追踪的 FL 轮次继续到初始化模型的过程。如果 FL 进程尚未开始，即 `self.sm.round` 为 `0`，它将调用 `_initialize_fl(msg)` 函数以初始化 FL 进程。

然后，FL 服务器通过调用 `_send_updated_global_model(websocket, id, es)` 函数将更新的全局模型发送回代理。该函数接受 WebSocket、代理 ID 和 `exch_socket` 作为参数，并向代理创建一个回复消息以通知其参与消息是否已被接受。

在此示例代码中，简化了代理与 FL 服务器的注册过程。在生产环境中，所有来自代理的系统信息都将推送到数据库，以便任何失去与 FL 服务器连接的代理都可以通过重新连接到 FL 服务器来随时恢复。

通常情况下，如果 FL 服务器安装在云端，并且代理从其本地环境连接到 FL 服务器，由于安全设置（如防火墙）等原因，聚合器到代理的这种回推机制将不会工作。本书中我们不详细讨论安全问题，因此鼓励您使用 `simple-fl` 代码中实现的 `polling` 方法在基于云的聚合器和本地代理之间进行通信。

### 获取套接字信息以将全局模型推回代理

下面的 `_get_exch_socket` 函数从代理接收参与消息，并根据消息中的模拟标志决定使用哪个端口来联系代理：

```py
def _get_exch_socket(self, msg):
    if msg[int(ParticipateMSGLocation.sim_flag)]:
        es = msg[int(ParticipateMSGLocation.exch_socket)]
    else:
        es = self.exch_socket
    return es
```

在这个实现练习中，我们支持进行模拟运行，通过这种方式，你可以在一台机器上运行数据库、聚合器和多个代理的所有 FL 系统组件。

### 如有必要，初始化 FL 进程

异步的 `_initialize_fl` 函数用于初始化一个 FL 进程，该进程仅在 FL 轮次为 `0` 时被调用。以下是其代码实现：

```py
async def _initialize_fl(self, msg):
    agent_id = msg[int(ParticipateMSGLocation.agent_id)]
    model_id = msg[int(ParticipateMSGLocation.model_id)]
    gene_time = msg[int(ParticipateMSGLocation.gene_time)]
    lmodels = msg[int(ParticipateMSGLocation.lmodels)] 
    perf_val = msg[int(ParticipateMSGLocation.meta_data)]
    init_flag = \
        bool(msg[int(ParticipateMSGLocation.init_flag)])
    self.sm.initialize_model_info(lmodels, init_flag)
    await self._push_local_models( \
        agent_id, model_id, lmodels, gene_time, perf_val)
    self.sm.increment_round()
```

从接收到的消息中提取代理 ID (`agent_id`)、模型 ID (`model_id`)、来自代理的本地模型 (`lmodels`)、模型的生成时间 (`gene_time`)、性能数据 (`perf_val`) 和 `init_flag` 的值后，调用状态管理器代码中的 `initialize_model_info` 函数，该函数将在本章后面的部分进行解释。

此函数随后通过调用本节中描述的`_push_local_models`函数将本地模型推送到数据库。您可以参考*将本地和全局模型推送到数据库的函数*部分。

然后，轮次增加以进入 FL 的第一轮。

### 使用更新的全局模型确认代理参与

在初始化（集群）全局模型后，需要通过此注册过程将全局模型发送到连接到聚合器的代理。以下异步的`_send_updated_global_model`函数处理将全局模型发送到代理的过程，它以 WebSocket 信息、代理 ID 和用于联系代理的端口号作为参数。以下代码块描述了该过程：

```py
async def _send_updated_global_model( \
                   self, websocket, agent_id, exch_socket):
    model_id = self.sm.cluster_model_ids[-1]
    cluster_models = \
       convert_LDict_to_Dict(self.sm.cluster_models)
    reply = generate_agent_participation_confirm_message(
       self.sm.id, model_id, cluster_models, self.sm.round,
       agent_id, exch_socket, self.recv_socket)
    await send_websocket(reply, websocket)
```

如果 FL 过程已经启动，即`self.sm.round`已经大于 0，我们将从它们的缓冲区获取集群模型，并使用`convert_LDict_to_Dict`库函数将它们转换为字典格式。

然后，使用`generate_` `agent_participation_confirm_message`函数包装回复消息，并通过调用`send_websocket(reply, websocket)`函数将其发送给刚刚连接或重新连接到聚合器的代理。请参阅*将全局模型发送到代理的函数*部分。

既然我们已经了解了代理的注册过程，让我们继续到处理本地机器学习模型和轮询消息的实现。

## 处理来自本地代理的消息的服务器

FL 服务器上的异步`receive_msg_from_agent`过程持续运行，以接收本地模型更新并将它们推送到数据库和内存缓冲区，临时保存本地模型。它还响应来自本地代理的轮询消息。以下代码解释了这一功能：

```py
async def receive_msg_from_agent(self, websocket, path):
    msg = await receive(websocket)
    if msg[int(ModelUpMSGLocation.msg_type)] == \
                                       AgentMsgType.update:
        await self._process_lmodel_upload(msg)
    elif msg[int(PollingMSGLocation.msg_type)] == \
                                      AgentMsgType.polling:
        await self._process_polling(msg, websocket)  
```

接下来，我们将查看由`receive_msg_from_agent`函数调用的两个函数，如前述代码块所示，它们是`_process_lmodel_upload`和`_process_polling`函数。

### 处理本地代理的模型上传

异步的`_process_lmodel_upload`函数处理`AgentMsgType.update`消息。以下代码块是关于接收本地机器学习模型并将它们放入状态管理器缓冲区的函数：

```py
async def _process_lmodel_upload(self, msg):
    lmodels = msg[int(ModelUpMSGLocation.lmodels)]
    agent_id = msg[int(ModelUpMSGLocation.agent_id)]
    model_id = msg[int(ModelUpMSGLocation.model_id)]
    gene_time = msg[int(ModelUpMSGLocation.gene_time)]
    perf_val = msg[int(ModelUpMSGLocation.meta_data)]
    await self._push_local_models( \ 
        agent_id, model_id, lmodels, gene_time, perf_val)
    self.sm.buffer_local_models( \ 
        lmodels, participate=False, meta_data=perf_val)
```

首先，它从接收到的消息中提取代理 ID（`agent_id`）、模型 ID（`model_id`）、来自代理的本地模型（`lmodels`）、模型的生成时间（`gene_time`）和性能数据（`perf_val`），然后调用`_push_local_models`函数将本地模型推送到数据库。

然后调用`buffer_local_models`函数以将本地模型（`lmodels`）保存在内存缓冲区中。`buffer_local_models`函数在*使用状态管理器维护聚合模型*部分中描述。

### 处理代理的轮询

以下异步的`_process_polling`函数处理`AgentMsgType.polling`消息：

```py
async def _process_polling(self, msg, websocket):
    if self.sm.round > \
                   int(msg[int(PollingMSGLocation.round)]):
        model_id = self.sm.cluster_model_ids[-1]
        cluster_models = \
            convert_LDict_to_Dict(self.sm.cluster_models)
        msg = generate_cluster_model_dist_message( \
            self.sm.id, model_id, self.sm.round, \
            cluster_models)
        await send_websocket(msg, websocket)
    else:
        msg = generate_ack_message()
        await send_websocket(msg, websocket)  
```

如果 FL 轮次（`self.sm.round`）大于本地 FL 轮次，该轮次包含在本地代理自身维护的接收消息中，这意味着模型聚合是在代理上次向聚合器轮询的时间和现在之间的期间完成的。

在此情况下，通过`generate_cluster_model_dist_message`函数将`cluster_models`转换为字典格式后，通过`send_websocket`函数打包成响应消息并回传给代理。

否则，聚合器将仅通过`generate_ack_message`函数生成的*ACK*消息返回给代理。

现在我们已经准备好聚合从代理接收到的本地模型，让我们来看看模型聚合例程。

## 全局模型合成例程

在 FL 服务器中设计的`async def model_synthesis_routine(self)`过程通过定期检查存储的模型数量，并在收集到足够多的本地模型以满足聚合阈值时执行全局模型合成。

以下代码描述了模型合成例程过程，该过程定期检查聚合标准并执行模型合成：

```py
async def model_synthesis_routine(self):
    while True:
        await asyncio.sleep(self.round_interval)
        if self.sm.ready_for_local_aggregation():  
            self.agg.aggregate_local_models()
            await self._push_cluster_models()
            if self.is_polling == False:
                await self._send_cluster_models_to_all()
            self.sm.increment_round()
```

此过程是异步的，使用`while`循环运行。

特别是，一旦满足`ready_for_local_aggregation`（在*使用状态管理器维护聚合模型*部分中解释）设定的条件，就会调用从`aggregator.py`文件导入的`aggregate_local_models`函数，该函数基于`FedAvg`对收集到的本地模型权重进行平均。关于`aggregate_local_models`函数的进一步解释可以在*聚合本地模型*部分找到。

然后，调用`await self._push_cluster_models()`以将聚合的集群全局模型推送到数据库。

`await self._send_cluster_models_to_all()`用于在未使用`polling`方法的情况下，将更新的全局模型发送给连接到聚合器的所有代理。

最后但同样重要的是，FL 轮次通过`self.sm.increment_round()`递增。

一旦生成集群全局模型，就需要使用以下章节中描述的函数将模型发送到连接的代理。

## 用于将全局模型发送到代理的函数

将全局模型发送到连接的代理的功能由`_send_cluster_models_to_all`函数处理。这是一个异步函数，用于将集群全局模型发送到本聚合器下的所有代理，如下所示：

```py
async def _send_cluster_models_to_all(self):
    model_id = self.sm.cluster_model_ids[-1]
    cluster_models = \
        convert_LDict_to_Dict(self.sm.cluster_models)
    msg = generate_cluster_model_dist_message( \
        self.sm.id, model_id, self.sm.round, \
        cluster_models)
    for agent in self.sm.agent_set:
        await send(msg, agent['agent_ip'], agent['socket'])
```

在获取集群模型信息后，它使用`generate_cluster_model_dist_message`函数创建包含集群模型、轮次、模型 ID 和聚合器 ID 信息的消息，并调用`communication_handler`库中的`send`函数，将全局模型发送到通过代理参与过程注册的`agent_set`中的所有代理。

已经解释了将集群全局模型发送到连接的代理。接下来，我们将解释如何将本地和集群模型推送到数据库。

## 将本地和全局模型推送到数据库的函数

`_push_local_models`和`_push_cluster_models`函数都内部调用以将本地模型和集群全局模型推送到数据库。

### 将本地模型推送到数据库

以下是将一组本地模型推送到数据库的`_push_local_models`函数：

```py
async def _push_local_models(self, agent_id: str, \
        model_id: str, local_models: Dict[str, np.array], \
        gene_time: float, performance: Dict[str, float]) \
        -> List[Any]:
    return await self._push_models(
        agent_id, ModelType.local, local_models, \
        model_id, gene_time, performance)
```

`_push_local_models`函数接受诸如代理 ID、本地模型、模型 ID、模型的生成时间以及性能数据等参数，如果有响应消息则返回。

### 将集群模型推送到数据库

以下是将集群全局模型推送到数据库的`_push_cluster_models`函数：

```py
async def _push_cluster_models(self) -> List[Any]:
    model_id = self.sm.cluster_model_ids[-1] 
    models = convert_LDict_to_Dict(self.sm.cluster_models)
    meta_dict = dict({ \
        "num_samples" : self.sm.own_cluster_num_samples})
    return await self._push_models( \
        self.sm.id, ModelType.cluster, models, model_id, \
        time.time(), meta_dict)
```

在此代码中，`_push_cluster_models`函数不接收任何参数，因为这些参数可以从状态管理器的实例信息和缓冲内存数据中获取。例如，`self.sm.cluster_model_ids[-1]`获取最新集群模型的 ID，而`self.sm.cluster_models`存储最新的集群模型本身，并将其转换为字典格式的`models`以发送到数据库。它还创建`mata_dict`来存储样本数量。

### 将机器学习模型推送到数据库

上述两个函数都按照如下方式调用`_push_models`函数：

```py
async def _push_models(
    self, component_id: str, model_type: ModelType,
    models: Dict[str, np.array], model_id: str,
    gene_time: float, performance_dict: Dict[str, float])
    -> List[Any]:
    msg = generate_db_push_message(component_id, \
        self.sm.round, model_type, models, model_id, \
        gene_time, performance_dict)
    resp = await send(msg, self.db_ip, self.db_socket)
    return resp
```

在此代码示例中，`_push_models`函数接受诸如`component_id`（聚合器或代理的 ID）、`model_type`（如本地或集群模型）、`models`本身、`model_id`、`gene_time`（模型创建的时间）以及`performance_dict`（作为模型的性能指标）等参数。然后，通过`generate_db_push_message`函数创建要发送到数据库的消息（使用`send`函数），这些参数包括 FL 轮次信息。它从数据库返回响应消息。

现在我们已经解释了与 FL 服务器相关的所有核心功能，让我们来看看状态管理器的角色，它维护聚合过程所需的全部模型。

# 使用状态管理器维护聚合所需的模型

在本节中，我们将解释`state_manager.py`，该文件处理维护模型以及与本地模型聚合相关的必要易失性信息。

## 导入状态管理器的库

此代码导入了以下内容。`data_struc`、`helpers` 和 `states` 的内部库在 *附录*、*探索内部库* 中介绍：

```py
import numpy as np
import logging
import time
from typing import Dict, Any
from fl_main.lib.util.data_struc import LimitedDict
from fl_main.lib.util.helpers import generate_id, generate_model_id
from fl_main.lib.util.states import IDPrefix
```

在导入必要的库之后，让我们定义状态管理器类。

## 定义状态管理器类

状态管理器类（`Class StateManager`），如 `state_manager.py` 中所见，在以下代码中定义：

```py
class StateManager:
    """
    StateManager instance keeps the state of an aggregator.
    Functions are listed with this indentation.
    """
```

这跟踪聚合器的状态信息。聚合器和代理的易变状态也应存储，例如本地模型、连接到聚合器的代理信息、聚合过程生成的聚类模型以及当前轮次编号。

在定义了状态管理器之后，让我们继续初始化状态管理器。

## 初始化状态管理器

在 `__init__` 构造函数中，配置了与联邦学习过程相关的信息。以下代码是构建状态管理器的一个示例：

```py
def __init__(self):
    self.id = generate_id()
    self.agent_set = list()
    self.mnames = list()
    self.round = 0
    self.local_model_buffers = LimitedDict(self.mnames)
    self.local_model_num_samples = list()
    self.cluster_models = LimitedDict(self.mnames)
    self.cluster_model_ids = list()
    self.initialized = False
    self.agg_threshold = 1.0
```

`self.id` 聚合器的 ID 可以使用来自 `util.helpers` 库的 `generate_id()` 函数随机生成。

`self.agent_set` 是连接到聚合器的代理集合，其中集合的格式是字典信息的集合，在这种情况下与代理相关。

`self.mnames` 以列表格式存储要聚合的 ML 模型中每一层的名称。

`self.round` 被初始化为 `0` 以初始化联邦学习的轮次。

`local_model_buffers` 是由代理收集并存储在内存空间中的本地模型列表。`local_model_buffers` 接受来自代理的每个联邦学习轮次的本地模型，一旦聚合过程完成该轮次，此缓冲区将被清除并开始接受下一轮的本地模型。

`self.local_model_num_samples` 是一个列表，用于存储收集在缓冲区中的模型的样本数量。

`self.cluster_models` 是以 `LimitedDict` 格式存储的全局聚类模型集合，而 `self.cluster_model_ids` 是聚类模型 ID 的列表。

一旦设置了初始全局模型，`self.initialized` 变为 `True`，否则为 `False`。

`self.agg_threshold` 被初始化为 `1.0`，该值会被 `config_aggregator.json` 文件中指定的值覆盖。

在初始化状态管理器之后，让我们接下来调查初始化全局模型。

## 初始化全局模型

以下 `initialize_model_info` 函数设置了其他代理将使用的初始全局模型：

```py
def initialize_model_info(self, lmodels, \
                          init_weights_flag):
    for key in lmodels.keys():
        self.mnames.append(key)
    self.local_model_buffers = LimitedDict(self.mnames)
    self.cluster_models = LimitedDict(self.mnames)
    self.clear_lmodel_buffers()
    if init_weights_flag:
        self.initialize_models(lmodels, \
                            weight_keep=init_weights_flag)
    else:
        self.initialize_models(lmodels, weight_keep=False)
```

它填充了从初始代理发送的本地模型（`lmodels`）中提取的模型名称（`self.mnames`）。与模型名称一起，`local_model_buffers` 和 `cluster_models` 也被重新初始化。在清除本地模型缓冲区后，它调用 `initialize_models` 函数。

以下 `initialize_models` 函数根据作为模型参数接收的初始基础模型（以字典格式 `str` 或 `np.array`）初始化神经网络的结构（`numpy.array`）：

```py
def initialize_models(self, models: Dict[str, np.array], \
                                weight_keep: bool = False):
    self.clear_saved_models()
    for mname in self.mnames:
        if weight_keep:
            m = models[mname]
        else:
            m = np.zeros_like(models[mname])
        self.cluster_models[mname].append(m)
        id = generate_model_id(IDPrefix.aggregator, \
                 self.id, time.time())
        self.cluster_model_ids.append(id)
        self.initialized = True
```

对于模型的每一层，这里定义为模型名称，此函数填写模型参数。根据`weight_keep`标志，模型以零或接收到的参数初始化。这样，初始的集群全局模型与随机模型 ID 一起构建。如果代理发送的 ML 模型与这里定义的模型架构不同，聚合器将拒绝接受该模型或向代理发送错误信息。不返回任何内容。

因此，我们已经涵盖了全局模型的初始化。在下一节中，我们将解释 FL 过程的主体部分，即检查聚合标准。

## 检查聚合标准

以下名为`ready_for_local_aggregation`的代码用于检查聚合标准：

```py
def ready_for_local_aggregation(self) -> bool:
    if len(self.mnames) == 0:
            return False
    num_agents = int(self.agg_threshold * \
                                       len(self.agent_set))
    if num_agents == 0: num_agents = 1
    num_collected_lmodels = \
        len(self.local_model_buffers[self.mnames[0]])
    if num_collected_lmodels >= num_agents:
        return True
    else:
        return False            
```

此`ready_for_local_aggregation`函数返回一个`bool`值，以标识聚合器是否可以开始聚合过程。如果满足聚合标准（例如收集足够的本地模型以进行聚合），则返回`True`，否则返回`False`。聚合阈值`agg_threshold`在`config_aggregator.json`文件中配置。

下一节是关于缓存用于聚合过程的本地模型。

## 缓存本地模型

以下在`buffer_local_models`上的代码将代理的本地模型存储在本地模型缓冲区中：

```py
def buffer_local_models(self, models: Dict[str, np.array], 
        participate=False, meta_data: Dict[Any, Any] = {}):
    if not participate:  
        for key, model in models.items():
            self.local_model_buffers[key].append(model)
        try:
            num_samples = meta_data["num_samples"]
        except:
            num_samples = 1
        self.local_model_num_samples.append( \
                int(num_samples))
    else:  
        pass
    if not self.initialized:
        self.initialize_models(models)
```

参数包括以字典格式表示的本地`models`以及如样本数量等元信息。

首先，此函数通过检查参与标志来确定从代理发送的本地模型是初始模型还是不是。如果是初始模型，它将调用`initialize_model`函数，如前述代码块所示。

否则，对于用模型名称定义的模型的每一层，它将`numpy`数组存储在`self.local_model_buffers`中。`key`是模型名称，前述代码中提到的`model`是模型的实际参数。可选地，它可以接受代理用于重新训练过程的样本数量或数据源，并将其推送到`self.` `local_model_num_samples`缓冲区。

当 FL 服务器在`receive_msg_from_agent`过程中从代理接收本地模型时，会调用此函数。

这样，本地模型缓冲区已经解释完毕。接下来，我们将解释如何清除已保存的模型，以便聚合可以继续进行，而无需在缓冲区中存储不必要的模型。

## 清除已保存的模型

以下`clear_saved_models`函数清除本轮存储的所有集群模型：

```py
def clear_saved_models(self):
    for mname in self.mnames:
        self.cluster_models[mname].clear()
```

此函数在初始化 FL 过程之初被调用，集群全局模型被清空，以便再次开始新一轮的 FL。

以下函数，`clear_lmodel_buffers`函数，清除所有缓存的本地模型，为下一轮 FL 做准备：

```py
def clear_lmodel_buffers(self):
    for mname in self.mnames:
        self.local_model_buffers[mname].clear()
    self.local_model_num_samples = list()
```

在进行下一轮 FL 之前清除`local_model_buffers`中的本地模型是至关重要的。如果没有这个过程，要聚合的模型将与来自其他轮次的非相关模型混合，最终 FL 的性能有时会下降。

接下来，我们将解释在 FL 过程中添加代理的基本框架。

## 添加代理

这个`add_agent`函数处理使用系统内存进行简短的代理注册：

```py
def add_agent(self, agent_name: str, agent_id: str, \
                               agent_ip: str, socket: str):
    for agent in self.agent_set:
        if agent_name == agent['agent_name']:
            return agent['agent_id'], agent['socket']
    agent = {
        'agent_name': agent_name,
        'agent_id': agent_id,
        'agent_ip': agent_ip,
        'socket': socket
    }
    self.agent_set.append(agent)
    return agent_id, socket
```

此函数仅向`self.agent_set`列表添加与代理相关的信息。代理信息包括代理名称、代理 ID、代理 IP 地址以及用于联系代理的`socket`编号。`socket`编号可以在将集群全局模型发送到连接到聚合器的代理时使用，以及在聚合器和代理之间使用`push`方法进行通信时使用。此函数仅在代理注册过程中调用，并返回代理 ID 和`socket`编号。

如果代理已经注册，这意味着`agent_set`中已经存在具有相同名称的代理，它将返回现有代理的代理 ID 和`socket`编号。

再次强调，从聚合器到代理的此`push`通信方法在特定安全情况下不起作用。建议使用代理使用的`polling`方法，以不断检查聚合器是否有更新的全局模型。

可以使用数据库扩展代理注册机制，这将为您提供更好的分布式系统管理。

接下来，我们将涉及 FL 轮次的增加。

## 增加 FL 轮次

`increment_round`函数仅精确增加由状态管理器管理的轮次编号：

```py
def increment_round(self):
    self.round += 1
```

增加轮次是 FL 过程中支持连续学习操作的关键部分。此函数仅在注册初始全局模型或每次模型聚合过程之后调用。

现在我们已经了解了 FL 如何与状态管理器协同工作，在接下来的部分，我们将讨论模型聚合框架。

# 聚合本地模型

`aggregation.py`代码处理使用一系列聚合算法对本地模型进行聚合。在代码示例中，我们只支持在以下章节中讨论的**FedAvg**。

## 导入聚合器的库

`aggregation.py`代码导入以下内容：

```py
import logging
import time
import numpy as np
from typing import List
from .state_manager import StateManager
from fl_main.lib.util.helpers import generate_model_id
from fl_main.lib.util.states import IDPrefix
```

在*使用状态管理器维护聚合模型*部分中讨论了导入的状态管理器的角色和功能，并在*附录*、*探索内部库*中介绍了`helpers`和`states`库。

在导入必要的库之后，让我们定义聚合器类。

## 定义和初始化聚合器类

以下`class Aggregator`的代码定义了聚合器的核心过程，它提供了一套数学函数用于计算聚合模型：

```py
class Aggregator:
    """
    Aggregator class instance provides a set of 
    mathematical functions to compute aggregated models.
    """
```

以下 `__init__` 函数只是设置聚合器的状态管理器以访问模型缓冲区：

```py
def __init__(self, sm: StateManager):
    self.sm = sm
```

一旦聚合器类被定义和初始化，让我们看看实际的 FedAvg 算法实现。

## 定义 aggregate_local_models 函数

以下 `aggregate_local_models` 函数是聚合本地模型的代码：

```py
def aggregate_local_models(self):
    for mname in self.sm.mnames:
        self.sm.cluster_models[mname][0] \
            = self._average_aggregate( \
                self.sm.local_model_buffers[mname], \
                self.sm.local_model_num_samples)
    self.sm.own_cluster_num_samples = \
        sum(self.sm.local_model_num_samples)
    id = generate_model_id( \
        IDPrefix.aggregator, self.sm.id, time.time())
    self.sm.cluster_model_ids.append(id)
    self.sm.clear_lmodel_buffers()
```

此函数可以在聚合标准满足后调用，例如在 `config_aggregator.json` 文件中定义的聚合阈值。聚合过程使用状态管理器内存中缓存的本地 ML 模型。这些本地 ML 模型来自注册的代理。对于由 `mname` 定义的模型的每一层，模型权重由 `_average_aggregate` 函数如下平均，以实现 FedAvg。在平均所有层的模型参数后，`cluster_models` 被更新，并发送给所有代理。

然后，清除本地模型缓冲区，为下一轮 FL 流程做好准备。

## FedAvg 函数

以下函数 `_average_aggregate`，由前面的 `aggregate_local_models` 函数调用，是实现 `FedAvg` 聚合方法的代码：

```py
def _average_aggregate(self, buffer: List[np.array], 
                       num_samples: List[int]) -> np.array:
    denominator = sum(num_samples)
    model = float(num_samples[0])/denominator * buffer[0]
    for i in range(1, len(buffer)):
        model += float(num_samples[i]) / 
                                    denominator * buffer[i]
    return model
```

在 `_average_aggregate` 函数中，计算足够简单，对于给定的 ML 模型列表的每个缓冲区，它为模型取平均参数。模型聚合的基本原理在 *第三章*，*联邦学习系统的工作原理* 中讨论。它使用 `np.array` 返回加权聚合模型。

现在我们已经涵盖了 FL 服务器和聚合器的所有基本功能，接下来，我们将讨论如何运行 FL 服务器本身。

# 运行 FL 服务器

这里是一个运行 FL 服务器的示例。为了运行 FL 服务器，你只需执行以下代码：

```py
if __name__ == "__main__":
    s = Server()
    init_fl_server(s.register, 
                   s.receive_msg_from_agent, 
                   s.model_synthesis_routine(), 
                   s.aggr_ip, s.reg_socket, s.recv_socket)
```

FL 服务器实例的 `register`、`receive_msg_from_agent` 和 `model_synthesis_routine` 函数用于启动代理的注册过程、接收代理的消息以及启动模型合成过程以创建全局模型，所有这些都是使用 `communication_handler` 库中的 `init_fl_server` 函数启动的。

我们已经使用 FL 服务器涵盖了聚合器的所有核心模块。它们可以与数据库服务器一起工作，这将在下一节中讨论。

# 实现和运行数据库服务器

数据库服务器可以托管在聚合器服务器所在的同一台机器上，也可以与聚合器服务器分开。无论数据库服务器是否托管在同一台机器上，这里引入的代码都适用于这两种情况。数据库相关的代码可以在本书提供的 GitHub 仓库的 `fl_main/pseudodb` 文件夹中找到。

## 面向数据库的配置

以下代码是作为 `config_db.json` 保存的数据库端配置参数的示例：

```py
{
    "db_ip": "localhost",
    "db_socket": "9017",
    "db_name": "sample_data",
    "db_data_path": "./db",
    "db_model_path": "./db/models"
}
```

特别是，`db_data_path`是 SQLite 数据库的位置，`db_model_path`是 ML 模型二进制文件的位置。`config_db.json`文件可以在`setup`文件夹中找到。

接下来，让我们定义数据库服务器并导入必要的库。

## 定义数据库服务器

`pseudo_db.py`代码的主要功能是接收包含本地和集群全局模型的包含消息。

### 导入伪数据库所需的库

首先，`pseudo_db.py`代码导入了以下内容：

```py
import pickle, logging, time, os
from typing import Any, List
from .sqlite_db import SQLiteDBHandler
from fl_main.lib.util.helpers import generate_id, read_config, set_config_file
from fl_main.lib.util.states import DBMsgType, DBPushMsgLocation, ModelType
from fl_main.lib.util.communication_handler import init_db_server, send_websocket, receive
```

它导入了基本通用库以及`SQLiteDBHandler`（在*使用 SQLite 定义数据库*部分中讨论）和来自`lib/util`库的函数，这些函数在*附录*、*探索内部库*中讨论。

### 定义 PseudoDB 类

然后定义了`PseudoDB`类以创建一个实例，该实例从聚合器接收模型及其数据并将其推送到实际的数据库（在这个例子中是 SQLite）：

```py
class PseudoDB:
    """
    PseudoDB class instance receives models and their data
    from an aggregator, and pushes them to database
    """
```

现在，让我们继续初始化`PseudoDB`的实例。

### 初始化 PseudoDB

然后，初始化过程`__init__`被定义为以下内容：

```py
def __init__(self):
    self.id = generate_id()
    self.config = read_config(set_config_file("db"))
    self.db_ip = self.config['db_ip']
    self.db_socket = self.config['db_socket']
    self.data_path = self.config['db_data_path']
    if not os.path.exists(self.data_path):
        os.makedirs(self.data_path)
    self.db_file = \
        f'{self.data_path}/model_data{time.time()}.db'
    self.dbhandler = SQLiteDBHandler(self.db_file)
    self.dbhandler.initialize_DB()
    self.db_model_path = self.config['db_model_path']
    if not os.path.exists(self.db_model_path):
        os.makedirs(self.db_model_path)
```

初始化过程生成实例 ID 并设置各种参数，如数据库套接字（`db_socket`）、数据库 IP 地址（`db_ip`）、数据库路径（`data_path`）和数据库文件（`db_file`），所有这些均从`config_db.json`配置。

`dbhandler`存储`SQLiteDBHandler`的实例并调用`initialize_DB`函数来创建 SQLite 数据库。

如果不存在，则创建`data_path`和`db_model_path`的文件夹。

在`PseudoDB`的初始化过程之后，我们需要设计通信模块以接收来自聚合器的消息。我们再次使用 WebSocket 与聚合器进行通信，并将此模块作为服务器启动以接收和响应来自聚合器的消息。在这个设计中，我们不将来自数据库服务器的消息推送到聚合器或代理，以简化 FL 机制。

### 处理来自聚合器的消息

以下是对`async def handler`函数的代码，该函数以`websocket`作为参数，接收来自聚合器的消息并返回所需的信息：

```py
async def handler(self, websocket, path):
    msg = await receive(websocket)
    msg_type = msg[DBPushMsgLocation.msg_type]
    reply = list()
    if msg_type == DBMsgType.push:
        self._push_all_data_to_db(msg)
        reply.append('confirmation')
    else:
        raise TypeError(f'Undefined DB Message Type: \
                                              {msg_type}.')
    await send_websocket(reply, websocket)
```

在`handler`函数中，一旦它解码了从聚合器接收到的消息，`handler`函数会检查消息类型是否为`push`。如果是，它将尝试通过调用`_push_all_data_to_db`函数将本地或集群模型推送到数据库。否则，它将显示错误消息。然后可以将确认消息发送回聚合器。

在这里，我们只定义了`push`消息的类型，但你可以定义尽可能多的类型，同时增强数据库模式设计。

### 将所有数据推送到数据库

以下`_push_all_data_to_db`代码将模型信息推送到数据库：

```py
def _push_all_data_to_db(self, msg: List[Any]):
    pm = self._parse_message(msg)
    self.dbhandler.insert_an_entry(*pm)
    model_id = msg[int(DBPushMsgLocation.model_id)]
    models = msg[int(DBPushMsgLocation.models)]
    fname = f'{self.db_model_path}/{model_id}.binaryfile'
    with open(fname, 'wb') as f:
        pickle.dump(models, f)
```

模型的信息通过`_parse_message`函数提取，并传递给`_insert_an_entry`函数。然后，实际的模型保存在本地服务器文件系统中，其中模型的文件名和路径由这里的`db_model_path`和`fname`定义。

### 解析消息

`_parse_message`函数仅从接收到的消息中提取参数：

```py
def _parse_message(self, msg: List[Any]):
    component_id = msg[int(DBPushMsgLocation.component_id)]
    r = msg[int(DBPushMsgLocation.round)]
    mt = msg[int(DBPushMsgLocation.model_type)]
    model_id = msg[int(DBPushMsgLocation.model_id)]
    gene_time = msg[int(DBPushMsgLocation.gene_time)]
    meta_data = msg[int(DBPushMsgLocation.meta_data)]
    local_prfmc = 0.0
    if mt == ModelType.local:
        try: local_prfmc = meta_data["accuracy"]
        except: pass
    num_samples = 0
    try: num_samples = meta_data["num_samples"]
    except: pass
    return component_id, r, mt, model_id, gene_time, \
                                   local_prfmc, num_samples
```

此函数将接收到的消息解析为与代理 ID 或聚合器 ID（`component_id`）、轮数（`r`）、消息类型（`mt`）、`model_id`、模型生成时间（`gene_time`）以及以字典格式（`meta_data`）的性能数据相关的参数。当模型类型为本地时，提取本地性能数据`local_prfmc`。从`meta_dect`中提取在本地设备上使用的样本数据量。所有这些提取的参数在最后返回。

在以下部分，我们将解释使用 SQLite 框架实现的数据库。

## 使用 SQLite 定义数据库

`sqlite_db.py`代码创建 SQLite 数据库并处理从数据库中存储和检索数据。

### 导入 SQLite 数据库的库

`sqlite_db.py`按照如下方式导入基本通用库和`ModelType`：

```py
import sqlite3
import datetime
import logging
from fl_main.lib.util.states import ModelType
```

`lib/util`中的`ModelType`定义了模型类型：本地模型和（全局）集群模型。

### 定义和初始化 SQLiteDBHandler 类

然后，以下与`SQLiteDBHandler`类相关的代码创建并初始化 SQLite 数据库，并将模型插入 SQLite 数据库：

```py
class SQLiteDBHandler:
    """
    SQLiteDB Handler class that creates and initialize
    SQLite DB, and inserts models to the SQLiteDB
    """
```

初始化非常简单——只需将`PseudoDB`实例传递的`db_file`参数设置为`self.db_file`：

```py
def __init__(self, db_file):
    self.db_file = db_file
```

### 初始化数据库

在下面的`initialize_DB`函数中，使用 SQLite（`sqlite3`）定义了本地和集群模型的数据库表：

```py
def initialize_DB(self):
    conn = sqlite3.connect(f'{self.db_file}')
    c = conn.cursor()
    c.execute('''CREATE TABLE local_models(model_id, \
        generation_time, agent_id, round, performance, \
        num_samples)''')
    c.execute('''CREATE TABLE cluster_models(model_id, \
        generation_time, aggregator_id, round, \
        num_samples)''')
    conn.commit()
    conn.close()
```

在本例中，表被简化了，这样你可以轻松地跟踪上传的本地模型及其性能，以及由聚合器创建的全局模型。

`local_models`表具有模型 ID（`model_id`）、模型生成时间（`generation_time`）、上传的本地模型代理 ID（`agent_id`）、轮次信息（`round`）、本地模型的性能数据（`performance`）以及用于 FedAvg 聚合使用的样本数量（`num_samples`）。

`cluster_models`具有模型 ID（`model_id`）、模型生成时间（`generation_time`）、聚合器 ID（`aggregator_id`）、轮次信息（`round`）和样本数量（`num_samples`）。

### 将条目插入数据库

以下代码用于`insert_an_entry`，使用`sqlite3`库将接收到的参数数据插入：

```py
def insert_an_entry(self, component_id: str, r: int, mt: \
    ModelType, model_id: str, gtime: float, local_prfmc: \
    float, num_samples: int):
    conn = sqlite3.connect(self.db_file)
    c = conn.cursor()
    t = datetime.datetime.fromtimestamp(gtime)
    gene_time = t.strftime('%m/%d/%Y %H:%M:%S')
    if mt == ModelType.local:
        c.execute('''INSERT INTO local_models VALUES \
        (?, ?, ?, ?, ?, ?);''', (model_id, gene_time, \
        component_id, r, local_prfmc, num_samples))
    elif mt == ModelType.cluster:
        c.execute('''INSERT INTO cluster_models VALUES \
        (?, ?, ?, ?, ?);''', (model_id, gene_time, \
        component_id, r, num_samples))
    conn.commit()
    conn.close()
```

此函数接受`component_id`（代理 ID 或聚合器 ID）、轮数（`r`）、消息类型（`mt`）、模型 ID（`model_id`）、模型生成时间（`gtime`）、本地模型性能数据（`local_prfmc`）和要插入的样本数量（`num_samples`）作为参数，使用 SQLite 库的`execute`函数插入条目。

如果模型类型是*本地*，则将模型信息插入到`local_models`表中。如果模型类型是*集群*，则将模型信息插入到`cluster_models`表中。

其他功能，例如更新和删除数据库中的数据，在此示例代码中未实现，您需要自行编写这些附加功能。

在下一节中，我们将解释如何运行数据库服务器。

## 运行数据库服务器

下面是使用 SQLite 数据库运行数据库服务器的代码：

```py
if __name__ == "__main__":
    pdb = PseudoDB()
    init_db_server(pdb.handler, pdb.db_ip, pdb.db_socket)
```

`PseudoDB`类的实例被创建为`pdb`。`pdb.handler`、数据库的 IP 地址（`pdb.db_ip`）和数据库套接字（`pdb.db_socket`）用于启动从`init_db_server`函数中启用并由`communication_handler`库在`util/lib`文件夹中提供的聚合器接收本地和集群模型的过程。

现在，我们了解了如何实现和运行数据库服务器。本章中讨论的数据库表和模式设计得尽可能简单，以便我们理解 FL 服务器流程的基本原理。在下一节中，我们将讨论 FL 服务器的潜在增强功能。

# FL 服务器的潜在增强功能

在本章中讨论了 FL 服务器的一些关键潜在增强功能。

## 重新设计数据库

在本书中，数据库有意设计为包含最少的表信息，需要扩展，例如通过在数据库中添加聚合器本身、代理、初始基础模型和项目信息等表。例如，本章中描述的 FL 系统不支持服务器和代理进程的终止和重启。因此，FL 服务器的实现并不完整，因为它在系统停止或失败时丢失了大部分信息。

## 自动注册初始模型

为了简化注册初始模型的过程的解释，我们使用模型名称定义了 ML 模型的层。在系统中注册此模型可以自动化，这样只需加载具有`.pt/.pth`和`.h5`等文件扩展名的特定 ML 模型（如 PyTorch 或 Keras 模型），FL 系统的用户就可以开始这个过程。

## 本地模型和全局模型性能指标

再次，为了简化对 FL 服务器和数据库端功能的解释，准确度值仅被用作模型性能标准之一。通常，机器学习应用有更多指标需要跟踪作为性能数据，并且它们需要与数据库和通信协议设计一起增强。

## 微调聚合

为了简化聚合本地模型的过程，我们仅使用了 FedAvg，这是一种加权平均方法。样本数量可以根据本地环境动态变化，这一方面由你增强。还有各种模型聚合方法，这些方法将在本书的*第七章*“模型聚合”中解释，以便你可以根据要创建和集成到 FL 系统中的 ML 应用选择最佳的聚合方法。

# 摘要

在本章中，通过实际的代码示例解释了 FL 服务器端实现的基本原理和原则。跟随本章内容后，你现在应该能够使用模型聚合机制构建 FL 服务器端功能。

这里介绍的服务器端组件包括基本的通信、代理和初始模型的注册、用于聚合的状态信息管理，以及创建全局集群模型的聚合机制。此外，我们还讨论了仅存储 ML 模型信息的数据库实现。代码被简化，以便你能够理解服务器端功能的原则。构建更可持续、弹性、可扩展的 FL 系统的许多其他方面的进一步改进取决于你。

在下一章中，我们将讨论实现 FL 客户端和代理功能的原则。客户端需要为机器学习应用提供一些精心设计的 API 以供插件使用。因此，本章将讨论 FL 客户端的核心功能库以及库集成到非常简单的 ML 应用中，以实现整个 FL 过程。
