# *第二章*：边缘工作负载的基础

本章将探讨关于**边缘工作负载**的更深入细节以及您的第一次动手实践。您将了解AWS IoT Greengrass如何满足设计和交付现代边缘机器学习解决方案的需求。您将学习如何通过部署一个检查设备兼容性要求的工具来准备您的边缘设备与AWS协同工作。此外，您还将学习如何安装IoT Greengrass核心软件并部署您的第一个IoT Greengrass核心设备。您将了解组件的结构，检查IoT Greengrass中的软件的基本单元，并编写您的第一个边缘工作负载组件。

到本章结束时，您应该开始熟悉IoT Greengrass及其本地开发生命周期的基础知识。

在本章中，我们将涵盖以下主要内容：

+   边缘机器学习解决方案的解剖结构

+   IoT Greengrass大获全胜

+   检查与IoT设备测试器的兼容性

+   安装IoT Greengrass

+   您的第一个边缘组件

# 技术要求

本章的技术要求与[*第一章*](B17595_01_Final_SS_ePub.xhtml#_idTextAnchor013)“使用机器学习的数据驱动边缘简介”中描述的手动先决条件部分相同。请参阅该章节中提到的完整要求。提醒一下，您将需要以下内容：

+   基于Linux的系统用于部署IoT Greengrass软件。建议使用Raspberry Pi 3B或更高版本。安装说明与其他基于Linux的系统类似。当其他系统（如Raspberry Pi）的手动步骤不同时，请参考以下GitHub仓库以获取进一步指导。

+   安装和使用AWS **命令行界面**（**CLI**）的系统，以便访问AWS管理控制台网站（通常是您的PC/笔记本电脑）。

您可以从GitHub仓库的`chapter2`文件夹中访问本章的技术资源，网址为[https://github.com/PacktPublishing/Intelligent-Workloads-at-the-Edge/tree/main/chapter2](https://github.com/PacktPublishing/Intelligent-Workloads-at-the-Edge/tree/main/chapter2)。

# 边缘机器学习解决方案的解剖结构

上一章介绍了边缘解决方案的概念以及定义具有机器学习应用的边缘解决方案的三个关键工具。本章提供了关于边缘解决方案各层的更多细节。本节中讨论的三个层如下：

+   **业务逻辑层**包括决定解决方案行为的定制代码。

+   **物理接口层**通过传感器和执行器将您的解决方案连接到模拟世界。

+   **网络接口层**将您的解决方案连接到更广泛网络中的其他数字实体。

了解这些层对于重要，因为它们将告知您作为物联网架构师在设计边缘机器学习解决方案时如何进行权衡。首先，我们将从定义业务逻辑层开始。

## 为业务逻辑设计代码

业务逻辑层是您的边缘解决方案所有代码的存放地。这些代码可以有多种形式，例如*预编译的二进制文件*（如C程序）、*shell脚本*、*由运行时评估的代码*（如Java或Python程序）和*机器学习模型*。此外，代码可以以几种不同的方式组织，例如将所有内容打包到一个单体应用程序中、将代码拆分为服务或库，或将代码捆绑到容器中运行。所有这些选项都对架构和发送边缘机器学习解决方案有影响，例如安全性、成本、一致性、生产力和耐用性。业务逻辑层交付代码的一些挑战如下：

+   编写和测试将在您的边缘硬件平台上运行的代码。例如，编写将在硬件平台的新版本推出时工作的代码，您将希望最小化维护满足所有硬件平台需求的代码分支数量。

+   设计一个包含许多功能的强大边缘解决方案。例如，将功能捆绑在一起以处理新的传感器数据、分析数据和与不与常见依赖项或本地资源冲突的Web服务进行通信。

+   与一个团队一起编写代码，该团队正在共同开发一个边缘解决方案。例如，一个由许多贡献者组成的单体应用程序可能需要每个作者完全了解解决方案才能进行增量更改。

为了解决编写业务层逻辑的挑战，将代码发送到边缘的最佳实践是在实际可行的情况下使用**独立服务**。

### 独立服务

在您的*Home Base Solutions hub设备*（我们从[*第1章*](B17595_01_Final_SS_ePub.xhtml#_idTextAnchor013)，《利用机器学习的数据驱动边缘介绍》）中，代码将以独立服务的形式部署和运行。在这个背景下，一个**服务**是一个包含业务逻辑的自包含单元，它可以被另一个实体调用以执行任务，或者自行执行任务。**隔离**意味着服务将捆绑其操作所需的代码、资源和依赖项。例如，您将在[*第7章*](B17595_07_Final_SS_ePub.xhtml#_idTextAnchor138)，《边缘机器学习工作负载》中创建的服务将运行代码以从数据源或图像集合中读取，定期使用捆绑的机器学习模型进行推理，然后将任何推理结果发布到本地流或云端。选择这种独立服务模式有两个原因：

+   第一个原因是**服务导向架构**使架构师能够设计相互解耦的能力。**解耦**意味着我们使用数据结构，如缓冲区、队列和流，在服务之间添加一层抽象，减少依赖性，使服务能够独立运行。

    您可以在不触及其他正在运行的服务的情况下部署单个服务的更新，因此可以降低对它们的影响风险。解耦的服务导向架构是设计良好架构的云解决方案的最佳实践，同时也非常适合边缘机器学习解决方案，在这些解决方案中，多个服务同时运行，并强调对可靠性的需求。例如，一个与传感器接口的服务将新的测量值写入数据结构，仅此而已；它只有一个任务，不需要了解数据是如何被后续功能消费的。

+   第二个原因是**代码隔离**使开发者能够专注于代码的功能，而不是代码的去向或依赖项在目标位置的管理方式。通过使用隔离原则将运行时依赖项和资源与代码捆绑在一起，我们获得了更强的确定性，即代码将在部署的任何地方都能确定性地工作。开发者可以释放出用于依赖项管理的努力，并且更有信心代码将在边缘平台上以相同的方式工作，这可能与他们的开发环境不同。这并不是说边缘解决方案的开发者不需要测试他们的代码对物理接口（如传感器和执行器）的行为。然而，这意味着开发团队能够交付自包含的服务，这些服务可以独立工作，而不管在聚合的边缘解决方案中部署的其他服务如何。

    隔离的例子包括**Python**虚拟环境，它明确指定了Python运行时版本和包，以及**Docker Engine**，它使用容器来打包依赖项、资源和在主机上实现进程隔离。以下图表展示了使用隔离服务实现的关注点分离：

![图 2.1 – 使用解耦、隔离服务的边缘解决方案

![图 B17595_02_001.jpg](img/B17595_02_001.jpg)

图 2.1 – 使用解耦、隔离服务的边缘解决方案

同时，隔离和服务模式为边缘机器学习解决方案提供了引人注目的好处。当然，开发中的每一个决策都伴随着权衡。如果作为单一代码单体部署，解决方案会更简单，并且更快地推出最小可行产品。我们选择更复杂的架构，因为这会导致更好的弹性和随时间扩展的解决方案。我们依靠强大的模式和良好的工具来平衡这种复杂性。

物联网 Greengrass 是按照这种模式设计的。在本章及整本书中，你将学习如何使用这种模式与物联网 Greengrass 结合，开发结构良好的边缘机器学习解决方案。

## 物理接口

网络物理解决方案是通过使用物理接口与模拟世界进行交互来定义的。这些接口分为两类：*用于从模拟世界获取测量的传感器*和*用于对它施加变化的执行器*。一些机器两者都做，例如冰箱可以感知内部温度并激活其压缩机循环制冷剂。在这些情况下，传感和执行器的聚合是逻辑的，这意味着传感器和执行器之间存在关系，但在功能上是独立的，并且通过开关、电路或微控制器等机制进行协调。

执行模拟到数字转换的传感器通过采样电信号中的电压并将其转换为数字值。这些数字值通过代码解释以推导出数据，如温度、光和压力。执行器将数字信号转换为模拟动作，通常通过操纵通往开关或电路的电压来实现。深入探讨物理接口的电气工程超出了本书的范围。请参阅 *参考文献* 部分以获取关于该主题深入研究的建议。以下图表展示了一个简单的模拟示例，包括冰箱以及恒温器（传感器）、开关（控制器）和压缩机（执行器）之间的关系：

![图 2.2 – 带有传感器和执行器的模拟控制器

](img/B17595_02_002.jpg)

图 2.2 – 带有传感器和执行器的模拟控制器

理解由网络物理解决方案提供的输入和输出模式以及与通过边缘机器学习解决方案提供的更高层次结果之间的关系是很重要的。在本书中交付的项目中，你将获得实际应用这些模式的经验。Home Base Solutions 中心设备的一些服务将作为物理层的接口，提供来自传感器的新的测量值，并将命令转换为改变本地设备状态的命令。如果你正在使用物理边缘设备，例如 Raspberry Pi，你将获得一些使用代码与该设备的物理接口交互的经验。

## 网络接口

在介绍我们的边缘解决方案结构时，需要引入的第三层是*网络接口*。我们定义的物理-网络解决方案与边缘解决方案之间的区别在于，边缘解决方案在某个时刻将通过网络与另一个实体进行交互。例如，我们为Home Base Solutions设计的新设备监控套件使用监控套件与中心设备之间的无线通信。为了将监控器的传感器信号从模拟转换为数字，这两个设备之间没有物理连接。

同样，中心设备也会与云服务交换消息，以存储用于训练机器学习模型的数据、向设备部署新资源以及向客户通知已识别的事件。以下图示说明了消息流以及**传感器**、**执行器**、中心设备（**网关**）和云服务之间的关系：

![图2.3 – 一个边缘设备与本地传感器、执行器和云交换消息

![img/B17595_02_003.jpg]

图2.3 – 一个边缘设备与本地传感器、执行器和云交换消息

在物联网解决方案中，无线通信很常见，并且特定的实现可以在广泛的距离范围内实现连接。每个规范和实现都会在范围、数据传输速率、硬件成本和能耗之间做出权衡。短距离无线电规范，如**Zigbee**（IEEE 802.15.4）、**蓝牙**（IEEE 802.15.1）和**WiFi**（IEEE 802.11），适用于连接个人和局域网内的设备。长距离无线电规范，如传统的蜂窝网络（例如，**GSM**、**CDMA**和**LTE**）以及**低功耗广域网络**（**LPWANs**）如**LoRaWAN**和**NB-IoT**，为部署（无论是静态还是漫游）在特定校园、城市或地区的设备提供了连接选项。

有线通信仍然用于连接边缘设备，如电视、游戏机和PC，通过**以太网**连接到家庭网络解决方案，如交换机和路由器。由于家庭网络路由器上以太网端口的数量有限（通常只有1-8个端口），设备放置的限制以及在家中的布线负担，有线连接在智能家居产品中不太常见。

例如，家庭基站解决方案的设备监控套件可能会使用Zigbee或等效实现，以及电池来平衡能量消耗与预期的数据速率。如果套件需要从附近的插座获取电源，Wi-Fi就成为一个更可行的选择；然而，它将限制整体产品的实用性，因为要监控的设备类型放置位置并不总是有额外的插座。此外，直接使用以太网将套件连接到集线器也没有意义，因为客户可能不会觉得家里到处都是额外的电线很吸引人。与套件通信的集线器设备可以使用以太网或Wi-Fi连接到客户的本地网络，从而访问公共互联网。

现在你已经更好地理解了边缘解决方案的三个层次，让我们来评估选定的边缘运行时解决方案以及它是如何实现每一层的。

# 物联网Greengrass大获全胜

在一本关于使用物联网Greengrass提供边缘机器学习解决方案的书中，最重要的一个问题是要回答*为什么是物联网Greengrass？*在评估边缘机器学习解决方案的独特挑战和实现它们所需的关键工具时，你希望选择尽可能解决你问题的工具，同时在提高生产效率方面不给你带来麻烦。物联网Greengrass是一个专门构建的工具，其价值主张将物联网和机器学习解决方案置于前沿。

物联网Greengrass在解决常见需求的*无差别的繁重工作*问题时具有指导性，但在实现你的业务逻辑时则不具有指导性。这意味着开箱即用的体验提供了许多快速迭代的能力，同时不会阻碍你如何使用它们来实现最终目标。以下是一些物联网Greengrass提供的能力列表：

+   **边缘安全**：物联网Greengrass以root权限安装，并使用操作系统用户权限来保护在边缘部署的代码和资源，防止篡改。

+   **云安全**：物联网Greengrass使用与公共密钥基础设施互操作的**传输层安全**（**TLS**）来在边缘和云之间交换消息。在部署期间使用HTTPS和AWS签名版本4来验证请求者的身份并保护传输中的数据。

+   **运行时编排**：开发者可以按自己的喜好设计应用程序（使用单体、服务或容器），并轻松地将它们部署到边缘。物联网Greengrass提供了智能集成组件生命周期事件的钩子，或者开发者可以忽略它们，只需一个命令就可以引导应用程序。可以添加或更新单个组件，而不会中断其他正在运行的服务。依赖关系树允许开发者将库的安装和配置活动抽象出来，从而与代码工件解耦。

+   **日志和监控**：默认情况下，IoT Greengrass 为每个组件创建日志，并允许开发者指定哪些日志文件应同步到云端以供操作使用。此外，云服务会自动跟踪设备健康状况，使团队成员更容易识别和响应不健康的设备。

+   **扩展车队规模**：向单个设备部署更新与向设备车队部署更新并没有太大区别。定义组、将类似设备分类在一起，然后使用托管部署服务向设备组推送更新很容易。

+   **原生集成**：AWS 为部署到 IoT Greengrass 解决方案提供了许多组件，这些组件增强了基线功能，并可用于与其他 AWS 服务集成。一个流管理组件使您能够在边缘定义、写入和消费流。一个 Docker 应用程序管理器允许您从公共存储库或**Amazon Elastic Container Registry**中的私有存储库下载 Docker 镜像。预训练和优化的 ML 模型可用于对象检测和图像分类等任务，这些任务由**Deep Learning Runtime**和**TensorFlow Lite**提供支持。

在扮演作为构建解决方案的 Home Base 解决方案架构师的角色时，您可能会建议工程团队投入时间和资源来构建所有这些功能，并测试其是否已准备好投入生产。然而，IoT Greengrass 基线服务和可选附加组件已准备好加速开发周期，并由 AWS 审核通过，其中**安全性是首要任务**。

IoT Greengrass 并不会为您做所有事情。实际上，IoT Greengrass 的全新安装不会做任何事情，只是等待进一步的指令，即部署。把它想象成一张空白画布、颜料和画笔。它提供了您开始所需的一切，但您必须开发它所运行的解决方案。让我们回顾 IoT Greengrass 的运营模式，包括边缘和云端的运营模式。

## 检查 IoT Greengrass 架构

IoT Greengrass 既是运行在 AWS 上的托管服务，也是边缘运行时工具。托管服务是您定义设备的地方，包括单个设备和分组。当您想要将新的部署推送到边缘时，实际上是在托管服务中调用一个 API，然后该 API 负责与边缘运行时通信，以协调该部署的交付。以下是一个序列图，展示了您作为开发者配置组件并请求运行 IoT Greengrass 核心软件的设备接收并运行该组件的过程：

![图 2.4 – 通过 IoT Greengrass 推送部署

![图片 B17595_02_004.jpg](img/B17595_02_004.jpg)

图 2.4 – 通过 IoT Greengrass 推送部署

**组件**是部署到运行IoT Greengrass的设备上的功能的基本单元。组件由一个名为**配方**的清单文件定义，它告诉IoT Greengrass该组件的名称、版本、依赖项和指令。除此之外，组件还可以定义零个或多个在部署期间获取的**工件**。这些工件可以是二进制文件、源代码、编译代码、存档、图像或数据文件；实际上，任何存储在磁盘上的文件或资源。组件配方可以定义对其他组件的依赖关系，这些依赖关系将通过IoT Greengrass软件通过图来解析。

在部署活动期间，在该部署中添加或更新一个或多个组件。组件的工件从云中下载到本地设备；然后，通过评估生命周期指令来启动组件。生命周期指令可能是启动期间发生的事情；它可能是要运行的主要命令，例如启动Java应用程序或组件运行结束后要做的事情。组件可能会无限期地保持运行状态，或者执行任务后退出。以下图表提供了一个组件图的示例：

![Figure 2.5 – 组件的生命周期和依赖关系的示例图]

![img/B17595_02_005.jpg]

Figure 2.5 – 组件的生命周期和依赖关系的示例图

这是我们准备开始将边缘设备准备好运行带有IoT Greengrass的解决方案之前需要涵盖的所有内容！

在以下章节中，您将验证您的边缘设备是否已准备好运行IoT Greengrass软件，安装软件，然后编写您的第一个组件。此外，您将通过即将到来的动手活动更深入地了解组件和部署。

# 检查与IoT设备测试器的兼容性

**IoT设备测试器**（**IDT**）是AWS提供的一个软件客户端，用于评估设备在AWS IoT解决方案中的使用准备情况。它通过运行一系列资格测试来帮助开发者验证目标系统是否已准备好运行IoT Greengrass核心软件。此外，它还运行一系列测试来证明边缘能力已经存在，例如建立与AWS的MQTT连接或在本地运行机器学习模型。IDT适用于您正在本地测试的一个设备，或者可以扩展以运行针对任何数量的设备组的定制测试套件，只要它们可以通过网络访问。

在您作为Home Base Solutions的物联网架构师的角色下，您应使用IDT来证明您的目标边缘设备平台（在这种情况下，平台指的是硬件和软件）能够运行所选的运行时编排工具，即物联网Greengrass。使用工具来证明兼容性的这种模式是手动评估目标平台和/或假设满足列出的某些要求组合的最佳实践。例如，一个潜在的设备平台可能会宣传其硬件要求和操作系统符合您的需求，但它可能缺少一个在开发生命周期后期才显现的关键库依赖项。最好尽早证明您需要的所有东西都已存在并得到考虑。

注意

IDT不仅能够使硬件具备运行物联网Greengrass核心软件的资格。该工具还可以使运行FreeRTOS的硬件具备资格，以验证设备能够与AWS IoT Core进行互操作。开发者可以编写自己的自定义测试，并将它们捆绑成套件，以纳入您的**软件开发生命周期**（**SDLC**）。

以下步骤将帮助您准备您的Raspberry Pi设备作为边缘系统（即，在我们虚构项目中的Home Base Solutions中心设备）使用，并在最终运行IDT软件之前配置您的AWS账户。如果您的设备已经配置了AWS账户并准备使用，您可以选择跳过*启动Raspberry Pi*和*配置AWS账户和权限*部分。如果您正在使用不同的平台作为边缘设备，您只需确保您可以通过SSH从您的指挥控制设备访问设备，并且有一个具有root权限的系统用户。

## 启动Raspberry Pi

以下步骤是在带有Raspberry Pi OS 2021年5月版干净安装的Raspberry Pi 3B上运行的。请参阅[https://www.raspberrypi.org/software/](https://www.raspberrypi.org/software/)以获取**Raspberry Pi Imager**工具。使用您的指挥控制系统运行映像工具，用Raspberry Pi OS的新映像闪存Micro SD卡。对于本书的项目，我们建议您使用一张空白磁盘，以避免任何预存软件和配置更改的意外后果。以下是Raspberry Pi Imager工具和要选择的映像的截图：

![图2.6 – Raspberry Pi Imager工具和要选择的映像

](img/B17595_02_006.jpg)

图2.6 – Raspberry Pi Imager工具和要选择的映像

以下是在使用Imager工具对Micro SD卡进行闪存操作后需要执行的步骤列表：

1.  将Micro SD卡插入Raspberry Pi。

1.  通过插入电源插头启动Raspberry Pi。

1.  完成首次启动向导。更新默认密码，设置您的区域首选项，并连接到Wi-Fi（如果您使用的是以太网，则此步骤为可选）。

1.  打开终端应用程序并运行 `sudo apt-get update` 和 `sudo apt-get upgrade`。

1.  重新启动 Pi。

1.  打开终端应用程序并运行 `hostname` 命令。复制该值并做好笔记；例如，在您的命令和控制系统的便签文件中写下它。在 Raspberry Pi 设备上，默认值为 `raspberrypi`。

1.  打开 Raspberry Pi 预设应用程序并启用 SSH 接口。IDT 访问设备时必须启用此接口。打开 **预设**，选择 **Raspberry Pi 配置**，选择 **接口**，并启用 SSH。

在这个里程碑时刻，您的 Raspberry Pi 设备已配置为加入与您的命令和控制系统相同的本地网络，并且可以通过远程 shell 会话访问。如果您使用不同的设备或虚拟机作为动手实践的边缘设备，您应该能够通过 SSH 访问该设备。一个检查此操作是否正常工作的好方法是尝试使用终端应用程序（或 `ssh pi@raspberrypi`）从您的命令和控制系统连接到您的边缘设备（如果您的主机名与 *步骤 6* 中的不同，请替换 `raspberrypi`）。接下来，您将配置您的 AWS 账户，以便在边缘设备上运行 IDT。

## 配置 AWS 账户和权限

在您的命令和控制系统上完成本节中的所有步骤。对于尚未拥有 AWS 账户的读者（如果您已经有账户访问权限，请跳到 *步骤 5*），请执行以下操作：

1.  创建您的 AWS 账户。在您的网络浏览器中导航到 [https://portal.aws.amazon.com/billing/signup](https://portal.aws.amazon.com/billing/signup) 并完成提示。您需要一个电子邮件地址、电话号码和信用卡。

1.  使用 root 登录 AWS 管理控制台，并导航到 **身份与访问管理**（**IAM**）服务。您可以在 [https://console.aws.amazon.com/iam/](https://console.aws.amazon.com/iam/) 找到它。

1.  使用 IAM 服务控制台设置您的管理组和用户账户。最佳实践是为自己创建一个新用户，而不是继续使用 root 用户登录。您将使用这个新用户通过 AWS 管理控制台或 AWS CLI 完成任何后续的 AWS 步骤：

    1.  创建一个名为 `AdministratorAccess` 的新用户（使用过滤器字段并输入它会更方便）。此策略由 AWS 管理，以授予用户管理员级别的权限。最佳实践是将权限与组相关联，然后将用户分配到组中以继承权限。这使得审计权限和理解用户从命名良好的组中拥有的访问权限变得更容易。

1.  从 AWS 管理控制台中注销。

    注意

    到目前为止，您应该可以访问一个具有管理员用户的 AWS 账户。完成以下步骤以设置 AWS CLI（如果您已经使用管理员用户配置了 AWS CLI，请跳到 *步骤 7*）。

1.  安装 AWS CLI。特定平台的说明可以在 [https://aws.amazon.com/cli/](https://aws.amazon.com/cli/) 找到。在这本书中，AWS CLI 步骤将使用 AWS CLI v2。

1.  安装完成后，配置 AWS CLI 并使用您为 `aws configure` 下载的凭据。

1.  当提示输入 `json`、`yaml`、`text`、`table` 时。作者的偏好是 `json`，并将反映在书中出现的任何 AWS CLI 输出示例中。

接下来，您将使用您的 `Admin` 用户创建一些更多资源，为以下部分使用 IDT 和安装 IoT Greengrass 核心软件做准备。这些是权限资源，类似于您的 `Admin` 用户，将被 IDT 和 IoT Greengrass 软件用于代表您与 AWS 交互。

+   使用 *步骤 3A* 中的自定义登录链接登录 AWS 管理控制台。使用 CSV 文件中提供的 `Admin` 用户名和密码。*   返回 IAM 服务控制台，可以在 [https://console.aws.amazon.com/iam/](https://console.aws.amazon.com/iam/) 找到。*   创建一个名为 `idtgg` 的新用户（简称 *IDT 和 Greengrass*）并选择 **程序访问** 类型。此用户不需要密码即可访问管理控制台。跳过权限和标签部分。确保您还下载了包含此用户凭据的 CSV 文件。*   创建一个名为 `idt-gg-permissions` 的新策略。第一步是定义策略的权限。选择 `chapter2/policies/idtgg-policy.json`。跳过标签部分。在审查部分，输入 `idt-gg-permissions` 的名称，输入 **IoT 设备测试器和 IoT Greengrass 权限** 的描述，并选择 **创建策略**。*   创建一个名为 `idt-gg-permissions` 策略的新用户组，并选择 `idtgg` 用户。选择 **创建组**。您现在已设置了一个新的组，附加了权限，并分配了作为 IDT 客户端和 IoT Greengrass 配置工具的身份验证和授权的程序访问用户。*   在您的终端或 PowerShell 应用程序中，为此新用户配置一个新的 AWS CLI 配置文件：

    1.  运行 `aws configure --profile idtgg`。

    1.  当提示输入访问密钥和秘密密钥时，请使用在 *步骤 9* 中下载的凭据 CSV 文件中的新值。

    1.  当提示默认区域时，请使用本书的默认值 **us-west-2** 或您在本书中所有项目中使用的 AWS 区域。

这就完成了配置您的 AWS 账户、权限和 CLI 的所有准备工作。下一个里程碑是安装 IDT 客户端并准备测试 Home Base Solutions 原型中心设备。

## 配置 IDT

您将从您的命令和控制系统中运行 IDT 软件，IDT 将通过 SSH 远程访问边缘设备系统以运行测试。

注意

以下步骤反映了编写时的IDT配置和使用情况。如果您遇到困难，可能是因为最新版本与编写时我们使用的版本不同。您可以通过AWS文档中的IDT获取有关安装、配置和使用的最新指南。请参阅[https://docs.aws.amazon.com/greengrass/v2/developerguide/device-tester-for-greengrass-ug.html](https://docs.aws.amazon.com/greengrass/v2/developerguide/device-tester-for-greengrass-ug.html)。

按照以下步骤使用IDT验证边缘设备系统是否已准备好运行物联网Greengrass。以下所有步骤均使用macOS上的物联网Greengrass核心软件v2.4.0和IDT v4.2.0以及测试套件GGV2Q_2.0.1完成。对于Windows、Linux或更晚的AWS软件版本，请根据需要更改命令和目录：

1.  在您的命令和控制系统中，打开网页浏览器并导航到[https://docs.aws.amazon.com/greengrass/v2/developerguide/dev-test-versions.html](https://docs.aws.amazon.com/greengrass/v2/developerguide/dev-test-versions.html)。

1.  在Windows的`C:\projects\idt`下或macOS和Linux的`~/projects/idt`下：![图2.7 – 下载IDT的AWS文档网站；确切文本和版本可能不同

    ![img/B17595_02_007.jpg]

    图2.7 – 下载IDT的AWS文档网站；确切文本和版本可能不同

1.  在目录中解压存档内容。在文件资源管理器中，双击存档以提取它们。如果使用终端，请使用类似`unzip devicetester_greengrass_v2_4.0.2_testsuite_1.1.0_mac.zip`的命令。这是macOS上的目录外观：![图2.8 – macOS Finder显示解压IDT存档后的目录内容

    ![img/B17595_02_008.jpg]

    图2.8 – macOS Finder显示解压IDT存档后的目录内容

1.  在您的浏览器中打开一个新标签页，并将以下链接粘贴以提示下载最新的物联网Greengrass核心软件：[https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-2.4.0.zip](https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-2.4.0.zip)（如果此链接无效，您可以在[https://docs.aws.amazon.com/greengrass/v2/developerguide/quick-installation.html#download-greengrass-core-v2](https://docs.aws.amazon.com/greengrass/v2/developerguide/quick-installation.html#download-greengrass-core-v2)找到最新指南）。

1.  将下载的文件重命名为`aws.greengrass.nucleus.zip`，并将其移动到IDT目录，例如`~/projects/idt/devicetester_greengrass_v2_mac/products/aws.greengrass.nucleus.zip`：![图2.9 – IDT将使用的物联网Greengrass软件

    ![img/B17595_02_009.jpg]

    图2.9 – IDT将使用的物联网Greengrass软件

1.  打开一个文本文件，例如`~/projects/idt/devicetester_greengrass_v2_mac/configs/config.json`，并更新以下值：

    1.  （可选）如果您不是使用书籍默认的`us-west-2`，请更新`awsRegion`。

    1.  要使用您之前配置的 `idtgg` 配置文件，按照以下方式设置 `auth` 的值：

    [PRE0]

1.  打开一个文本文件，例如 `~/projects/idt/devicetester_greengrass_v2_mac/configs/device.json`，并更新以下值：

    1.  `"id": "pool1"`。

    1.  `"sku": "hbshub"`（`hbshub` 代表 Home Base Solutions hub）。

    1.  在 `"features"` 下方，对于名为 `"arch"` 的名称-值对，设置 `"value": "armv7l"`（这是针对树莓派设备的；或者，您也可以选择适合您设备的适当架构）。

    1.  在 `"features"` 下方，对于 `"ml"`、`"docker"` 和 `"streamManagement"` 等剩余的名称-值对，设置 `"value": "no"`。目前，我们将禁用这些测试，因为我们没有立即使用测试功能的计划。如果您想评估设备的兼容性，请随意启用它们，尽管在全新镜像的树莓派上测试可能会失败。

    1.  在 `"devices"` 下方，设置 `"id": "raspberrypi"`（或您喜欢的任何设备 ID）。

    1.  在 `"connectivity"` 下方，将 `"ip"` 的值设置为您的边缘设备的 IP 地址（对于树莓派用户，该值是 *Booting the Raspberry Pi* 部分的 *步骤 6* 的输出）。

    1.  在 `"auth"` 下方，设置 `"method": "password"`。

    1.  在 `"credentials"` 下方，将 `"user"` 的值设置为用于 SSH 到边缘设备的用户名（通常，对于树莓派用户，这将是指 `"pi"`）。

    1.  在 `"credentials"` 下方，将 `"password"` 的值设置为用于 SSH 到边缘设备的密码。

    1.  在 `"credentials"` 下方，删除 `"privKeyPath"` 的行。

    1.  保存此文件的更改。您可以在本书的 GitHub 仓库中查看此文件的示例版本，位于 `chapter2/policies/idt-device-sample.json`。

1.  打开一个文本文件，例如 `~/projects/idt/devicetester_greengrass_v2_mac/configs/userdata.json`，并更新以下值。请确保指定绝对路径而不是相对路径：

    1.  `"TempResourcesDirOnDevice": "/tmp/idt"`。

    1.  `"InstallationDirRootOnDevice": "/greengrass"`。

    1.  `"GreengrassNucleusZip": "Users/ryan/projects/idt/devicetester_greengrass_v2_mac/products/aws.greengrass.nucleus.zip"`（根据本节 *步骤 5* 中您存储 `aws.greengrass.nucleus.zip` 文件的位置进行更新）。

    1.  保存此文件的更改。您可以在本书的 GitHub 仓库中查看此文件的示例版本，位于 `chapter2/policies/idt-userdata-sample.json`。

1.  打开一个应用程序，例如 macOS/Linux 上的 Terminal 或 Windows 上的 PowerShell。

1.  将您当前的工作目录更改为 IDT 启动器所在的位置：

    1.  在 macOS 上 `~/projects/idt/devicetester_greengrass_v2_mac/bin`

    1.  在 Linux 上 `~/projects/idt/devicetester_greengrass_v2_linux/bin`

    1.  在 Windows 上 `C:\projects\idt\devicetester_greengrass_v2_win\bin`

1.  运行命令以启动 IDT：

    1.  在 macOS 上执行 `./devicetester_mac_x86-64 run-suite --userdata userdata.json`

    1.  在 Linux 上 `./devicetester_linux_x86-64 run-suite --userdata userdata.json`

    1.  在 Windows 上 `devicetester_win_x86-64.exe run-suite --userdata userdata.json`

    运行IDT将启动一个本地应用程序，通过SSH连接到您的边缘设备并完成一系列测试。它将在遇到第一个失败的测试案例时停止，或者一直运行直到所有测试案例都通过。如果您正在按照前面的步骤运行IDT针对新的Raspberry Pi安装，您应该观察到以下类似的输出：

    [PRE1]

    故意省略了在Raspberry Pi上安装Java的步骤，以演示IDT如何识别缺失的依赖项；对于这种欺骗表示歉意！如果您运行了IDT测试套件并且通过了所有测试案例，那么您已经提前完成了计划，可以跳转到*安装IoT Greengrass*部分。

1.  为了修复这个缺失的依赖项，请返回您的Raspberry Pi界面并打开终端应用程序。

1.  使用`sudo apt-get install default-jdk`在Pi上安装Java。

1.  返回您的命令和控制系统，并再次运行IDT（重复*步骤11*中的命令）。

您的测试套件现在应该通过Java要求测试。如果您遇到其他失败，您将需要使用`idt/devicetester_greengrass_v2_mac/results`文件夹中的测试报告和日志来分类和修复它们。一些常见的错误包括缺少AWS凭证、权限不足的AWS凭证以及指向`userdata.json`中定义的资源的不正确路径。一个完全通过的测试案例集看起来像这样：

[PRE2]

这就完成了使用IDT分析并协助准备设备使用IoT Greengrass的入门介绍。在这里，最佳实践是使用软件测试，不仅是为了您自己的代码，还要评估边缘设备本身是否准备好与您的解决方案一起工作。依靠像IDT这样的工具，这些工具承担着证明设备准备好使用的繁重工作，并为每种新注册的设备或主要解决方案版本发布进行验证。您应该能够为您的下一个项目配置IDT，并使新的设备或设备组能够运行IoT Greengrass。在下一节中，您将学习如何在您的设备上安装IoT Greengrass，以便配置您的第一个边缘组件。

# 安装IoT Greengrass

现在您已经使用IDT验证了您的边缘设备与IoT Greengrass兼容，本章的下一个里程碑是安装IoT Greengrass。

从您的边缘设备（即原型Home Base Solutions中心），打开终端应用程序，或者使用您的命令和控制设备通过SSH远程访问它：

1.  切换到您的用户主目录：`cd ~/`。

1.  下载IoT Greengrass核心软件：`curl -s` [https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip](https://d2s8p88vqu9w66.cloudfront.net/releases/greengrass-nucleus-latest.zip) > greengrass-nucleus-latest.zip。

1.  解压存档：`unzip greengrass-nucleus-latest.zip -d greengrass && rm greengrass-nucleus-latest.zip`。

1.  您的边缘设备需要AWS凭证以便代表您配置云资源。您可以使用在上一节*配置AWS账户和权限*中为`idtgg`用户创建的相同凭证：

    1.  `export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE`

    1.  `export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`

1.  使用以下命令安装物联网Greengrass核心软件。如果您使用的是除`us-west-2`以外的AWS区域，请更新`--aws-region`参数的值。您可以从`chapter2/commands/provision-greengrass.sh`复制并粘贴此命令：

    [PRE3]

1.  就这样！此配置命令的最后几行输出应该看起来像这样：

    [PRE4]

在与IDT套件验证兼容性后，物联网Greengrass核心软件的安装和初始资源的配置过程会更加顺畅。现在您的边缘设备已经安装了第一个基本工具：运行时编排器。让我们回顾一下在此配置步骤中在边缘和AWS上创建的资源。

## 查看到目前为止已创建的内容

在您的边缘设备上，物联网Greengrass软件安装在了`/greengrass/v2`文件路径下。在该目录中，生成了用于连接AWS的公钥和私钥对、服务日志、用于存储食谱和工件本地包的本地存储库，以及用于推送到此设备的过去和现在部署的目录。您可以自由探索`/greengrass/v2`目录以熟悉设备上存储的内容；尽管如此，您需要使用`sudo`提升权限才能浏览所有内容。

安装添加了第一个`aws.greengrass.Nucleus`。核组件是物联网Greengrass的基础；它是唯一必需的组件，它促进了所有其他组件的关键功能，如部署、编排和生命周期管理。没有核组件，就没有物联网Greengrass。

此外，安装还创建了第一个`--deploy-dev-tools true`参数。该部署安装了一个名为`aws.greengrass.Cli`的组件。第二个组件包括一个名为`greengrass-cli`的脚本，用于本地开发任务，例如审查部署、组件和日志。它还可以用于创建新组件和部署。记住，使用物联网Greengrass，您可以在设备本地工作，或者通过AWS将其远程部署到设备。远程部署在[*第4章*](B17595_04_Final_SS_ePub.xhtml#_idTextAnchor073)中介绍，*将云扩展到边缘*。

在AWS中，创建了一些不同的资源。首先，从物联网Greengrass配置参数`--thing-name`中创建了一个新的`hbshub001`。同样，从`--thing-group-name`配置参数中创建了一个新的`hbshubprototypes`。事物组包含零个或多个事物和事物组。物联网Greengrass的设计使用事物组来识别应该运行相同部署的边缘设备集合。例如，如果您配置了另一个中心原型设备，您会将其添加到同一个`hbshubprototypes`事物组中，这样新的原型部署就会传播到您的所有原型设备上。

此外，您的`hbshub001`事物附加了一个名为`/greengrass/v2`的目录实体，并用于建立与AWS的相互认证连接。证书是AWS在设备使用其私钥（证书附加到`hbshub001`事物记录）连接时识别设备的方式，并且知道如何查找设备的权限。这些权限定义在另一个称为**物联网策略**的资源中。

物联网策略类似于AWS IAM策略，因为它定义了当参与者与AWS交互时允许执行的操作的明确权限。在物联网策略的情况下，参与者是设备，权限包括打开连接、发布和接收消息以及访问部署中定义的静态资源。设备通过其证书获得权限，这意味着事物附加到证书，而证书附加到一个或多个策略。以下是这些基本资源在边缘和云中如何相互关联的草图：

![Figure 2.10 – Illustrating the relationships between the IoT Core thing registry and edge resources]

![img/B17595_02_010.jpg]

图2.10 – 展示物联网核心事物注册表与边缘资源之间的关系

在物联网Greengrass的云服务中，为您的设备及其首次部署定义了一些额外的资源。物联网Greengrass **核心** 是一个设备（也称为事物）的映射，包括在设备上运行的组件和部署，以及设备所属的相关事物组。此外，核心还存储元数据，例如安装的物联网Greengrass核心软件的版本和最后一次已知健康检查的状态。以下是包含物联网Greengrass资源的关联关系图的另一种视图：

![Figure 2.11 – Illustrating the relationships between IoT Core, IoT Greengrass, and the edge device]

![img/B17595_02_011.jpg]

图2.11 – 展示物联网核心、物联网Greengrass和边缘设备之间的关系

现在您已经安装了物联网Greengrass，并对配置过程中创建的资源有了了解，让我们回顾一下组件部署后的样子，以便您实施*Hello, world*组件。

# 创建您的第一个边缘组件

任何开发者教育的最基本里程碑是*Hello, world*示例。对于您首次部署到IoT Greengrass的边缘组件，您将创建一个简单的*Hello, world*应用程序，以加强组件定义、依赖图以及如何创建新部署的概念。

## 检查现有组件

在开始编写新组件之前，花点时间熟悉已经通过IoT Greengrass CLI部署的现有组件。此CLI是在安装过程中通过`--deploy-dev-tools true`参数安装的。此工具旨在帮助您进行本地开发循环；然而，作为最佳实践，它不会在生产解决方案中安装。它安装在`/greengrass/v2/bin/greengrass-cli`。以下步骤演示了如何使用此工具：

1.  尝试调用`help`命令。在您的边缘设备的终端应用程序中运行`/greengrass/v2/bin/greengrass-cli help`。

1.  您应该查看`help`命令的输出，包括对`component`、`deployment`和`logs`命令的引用。尝试在`component`命令上调用`help`命令：`/greengrass/v2/bin/greengrass-cli help component`。

1.  您应该查看有关如何使用`component`命令的说明。接下来，尝试调用`component list`命令以显示所有本地安装的组件，`/greengrass/v2/bin/greengrass-cli component list`：

    [PRE5]

1.  `sudo /greengrass/v2/bin/greengrass-cli component list`

    [PRE6]

1.  您不需要运行以下*A*和*B*命令。它们被包含在这里是为了向您展示如何稍后找到文件内容：

    1.  要找到配方文件，请使用`sudo ls /greengrass/v2/packages/recipes/`。

    1.  要检查文件，请使用`sudo` `less /greengrass/v2/recipes/rQVjcR-rX_XGFHg0WYKAnptIez3HKwtctL_2BKKZegM@2.4.0.recipe.yaml`（请注意，您的文件名将不同）：

    [PRE7]

在此文件中，有一些重要的观察结果需要审查：

+   组件名称使用与`namespacing` Java包类似的反向域名方案。本书项目中的自定义组件将以`com.hbs.hub`开头，表示为Home Base Solutions hub产品编写的组件。

+   此组件与特定版本的IoT Greengrass核心绑定，这就是为什么版本是2.4.0。您的组件可以在此处指定任何版本，最佳实践是遵循语义版本规范。

+   `ComponentType`属性仅由AWS插件（如此CLI）使用。您的自定义组件将不会定义此属性。

+   此组件仅与特定版本的核心一起工作，因此它定义了对`aws.greengrass.nucleus`组件的软依赖。您的自定义组件默认情况下不需要指定核心依赖。这是您将定义对其他组件的依赖的地方，例如，确保在加载具有Python应用程序的组件之前已安装Python3的组件。

+   这个组件在全局级别或针对清单的`linux`平台版本没有定义特定的生命周期活动。

+   定义的艺术品是为特定的物联网Greengrass服务文件。你可以在`/greengrass/v2/packages/artifacts`目录中查看这些文件。当从云中部署时，你的工件URI将使用`s3://path/to/my/file`模式。在本地开发期间，你的清单不需要定义工件，因为它们预期已经存在于磁盘上。

+   注意两个工件上的权限。ZIP文件可以被任何系统用户读取。相比之下，JAR文件只能被`OWNER`读取，在这个场景中，意味着在安装时定义的默认系统用户，例如，`ggc_user`用户。

通过对组件结构的审查，现在是时候编写自己的组件了，包括工件和配方。

## 编写你的第一个组件

如前所述，我们希望创建的第一个组件是一个简单的`Hello, world`应用程序。在这个组件中，你将创建一个shell脚本，使用`echo`命令打印`Hello, world`。这个shell脚本是你的组件的工件。此外，你将编写一个配方文件，告诉物联网Greengrass如何将这个shell脚本用作组件。最后，你将使用本地物联网Greengrass CLI部署这个组件并检查它是否工作。

本地组件开发使用本地磁盘上可用的工件和配方文件，因此你需要为你的工作文件创建一些文件夹。在`/greengrass/v2`中没有专门用于存储工作文件的文件夹。因此，你需要创建一个简单的文件夹树并将组件文件放在那里：

1.  从你的边缘设备的终端应用程序中，切换到你的用户主目录：`cd ~/`。

1.  创建一个新的文件夹来存放你的本地组件资源：`mkdir -p hbshub/{artifacts,recipes}`。

1.  接下来，为一个新的工件创建路径并在其文件夹中添加一个shell脚本。让我们选择组件名为`com.hbs.hub.HelloWorld`，并从1.0.0版本开始。切换到工件文件夹：`cd hbshub/artifacts`。

1.  为你的组件工件创建一个新的目录：`mkdir -p com.hbs.hub.HelloWorld/1.0.0`。

1.  为shell脚本创建一个新文件：`touch com.hbs.hub.HelloWorld/1.0.0/hello.sh`。

1.  给这个文件设置写权限：`chmod +x com.hbs.hub.HelloWorld/1.0.0/hello.sh`。

1.  在编辑器中打开文件：`nano com.hbs.hub.HelloWorld/1.0.0/hello.sh`。

1.  在这个编辑器内部，添加以下内容（这些内容也可以在这个章节的GitHub仓库中找到）：

    [PRE8]

1.  测试你的脚本是否带参数或不带参数。如果没有提供参数替换`world`，脚本将打印`Hello, world`：

    1.  `./com.hbs.hub.HelloWorld/1.0.0/hello.sh`

    1.  `./com.hbs.hub.HelloWorld/1.0.0/hello.sh friend`

1.  这就是你的组件工件所需的所有内容。接下来，你将学习如何利用从菜谱文件内部传递参数的优势。切换到菜谱目录：`cd ~/hbshub/recipes`。

1.  打开编辑器创建菜谱文件：`nano com.hbs.hub.HelloWorld-1.0.0.json`。

1.  将以下内容添加到文件中。你也可以从本书的GitHub仓库复制此文件：

    [PRE9]

    这个菜谱很简单：它定义了一个生命周期步骤来运行我们位于已部署工件路径中的`hello.sh`脚本。尚未介绍的一个新功能是组件配置。`ComponentConfiguration`对象允许开发者定义任意键值对，这些键值对可以在整个菜谱文件中引用。在这种情况下，我们定义一个默认值作为脚本的参数传递。在部署组件时，这个值可以被覆盖，以自定义每个边缘设备如何使用已部署的组件。

    那么，在你编写了菜谱并提供了工件之后，如何测试一个组件呢？下一步是创建一个新的部署，告诉本地IoT Greengrass环境加载你的新组件并开始评估其生命周期事件。这正是IoT Greengrass CLI可以提供帮助的地方。

1.  使用以下命令创建一个包含你的新组件的新部署：

    [PRE10]

1.  你应该会看到一个类似以下内容的响应：

    [PRE11]

1.  你可以使用以下命令验证组件是否已成功部署（并且已经完成运行）：`sudo /greengrass/v2/bin/greengrass-cli component list`：

    [PRE12]

1.  你可以在组件的日志文件中查看此组件的输出：`sudo less /greengrass/v2/logs/com.hbs.hub.HelloWorld.log`（记住，`/greengrass/v2目录`属于root用户，因此必须使用`sudo`来访问日志文件）：

    [PRE13]

恭喜！你已经使用IoT Greengrass将你的第一个组件部署到了Home Base Solutions原型中心。在日志输出中，你可以观察到两个值得注意的观察结果。首先，你可以在向IoT Greengrass报告成功退出代码之前，查看组件生命周期状态从`STARTING`到`RUNNING`的时序。组件在那个点结束，所以我们不会在日志中看到一个显示它移动到`FINISHED`状态的条目，尽管这在`greengrass.log`文件中是可见的。

其次，你可以查看写入`STDOUT`的消息，其中包含感叹号（`world!`）。这意味着脚本收到了你的组件默认配置，而不是回退到`hello.sh`中内置的默认配置（`world`）。你还可以在菜谱文件中使用部署命令中包含的自定义值来覆盖默认配置值`world!`。你将在[*第4章*](B17595_04_Final_SS_ePub.xhtml#_idTextAnchor073)“扩展云到边缘”中学习如何使用该技术配置车队。

# 摘要

在本章中，你学习了我们将在这本书中使用的特定工具的基础知识，该工具满足任何边缘机器学习解决方案的关键需求之一，即运行时编排器。IoT Greengrass提供了开箱即用的功能，使开发者能够专注于他们的业务解决方案，而不是构建一个灵活、健壮的边缘运行时和部署机制的工作。你学习了在IoT Greengrass中，软件的基本单元是组件，它由一个配方和一组工件组成，组件通过部署进入解决方案。你学习了如何使用IDT验证设备是否准备好与IoT Greengrass一起工作，如何安装IoT Greengrass，开发你的第一个组件，并在本地环境中运行它。

在下一章中，我们将通过探索它如何启用网关功能、边缘常用的协议、安全最佳实践以及构建用于在网络物理解决方案中感知和执行的新组件，来更深入地了解IoT Greengrass的工作原理。

# 知识检查

在进入下一章之前，通过回答以下问题来测试你的知识。答案可以在书的末尾找到：

1.  以下哪一个是边缘机器学习解决方案中组织代码的最佳实践？是单体应用还是隔离服务？

1.  在你的边缘架构中将服务解耦有什么好处？

1.  将你的代码和依赖项从其他服务中隔离有什么好处？

1.  在选择物联网解决方案中的有线和无线网络实现时，需要考虑哪一种权衡？

1.  一个同时使用传感器和执行器的智能家居设备的例子是什么？

1.  定义物联网Greengrass组件的资源有两种类型是什么？

1.  对或错：组件必须在它的配方中定义至少一个工件。

1.  为什么默认情况下，只有root系统用户可以与物联网Greengrass目录中的文件交互是一个好的设计原则？

1.  对或错：组件可以部署到IoT Greengrass设备，无论是本地还是远程。

1.  你能想到三种不同的方法来更新你的`Hello, world`组件的行为，使其打印`Hello, Home Base Solutions customer!`吗？

# 参考文献

请参考以下资源，以获取本章讨论的概念的更多信息：

+   [https://semver.org](https://semver.org)上的语义版本规范。

+   《面向服务的架构：服务和微服务分析与设计》由Erl Thomas著，Pearson，2016年。

+   《模拟与数字电路基础》由Anant Agarwal、Jeffrey H. Lang和Morgan Kaufmann著，2005年。
