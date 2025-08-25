# 第十五章：*附录*：启动 H2O 集群的替代方法

本*附录*将向您展示如何在您的本地机器上启动 H2O-3 和 Sparkling Water 集群，以便您可以在本书中运行代码示例。我们还将向您展示如何在 H2O AI Cloud 的 90 天免费试用环境中启动 H2O 集群。此试用环境包括 Enterprise Steam，用于在 Kubernetes 基础设施上启动和管理 H2O 集群。

环境注意事项

**架构**：如*第二章*中介绍，*平台组件和关键概念*，您将使用客户端环境（实现了 H2O-3 或 Sparkling Water 库）来运行针对远程 H2O-3 或 Sparkling Water 架构的命令，该架构分布在 Kubernetes 或 Hadoop 集群的多台服务器节点上。然而，对于小型数据集，架构可以在与客户端相同的机器上作为单个进程本地启动。

**版本**：本书中的功能和使用代码示例使用以下版本：H2O-3 版本 3.34.0.7，以及 Sparkling Water 版本 3.34.0.7-1-3.2 以在 Spark 3.2 上运行。您将使用最新的（最新）稳定版本来设置您的环境，这将允许您运行本书中的相同代码示例，但也将包括在本书编写后添加到 H2O-3 和 Sparkling Water 中的功能。

**语言**：您可以在 Python、R 或 Java/Scala 中设置客户端环境。本书我们将使用 Python。您的 Python 客户端可以是 Jupyter notebook、PyCharm 或其他。

让我们学习如何在本地环境中完全运行 H2O-3。

# 本地 H2O-3 集群

这是运行 H2O-3 的一种最简单的方法，适用于本书代码示例中使用的较小数据集。它将在您的本地机器上启动 H2O-3（与企业集群环境相对），并且不涉及 H2O Enterprise Steam。

首先，我们将一次性设置我们的 H2O-3 Python 环境。

## 第 1 步 – 在 Python 中安装 H2O-3

要设置您的 H2O-3 Python 客户端，只需在您的 Python 环境中安装三个模块依赖项，然后安装 `h2o-3` Python 模块。您必须使用 Python 2.7.x、3.5.x、3.6.x 或 3.7.x。

更具体地说，请执行以下操作：

1.  在您的 Python 环境中安装依赖项：

    ```py
    pip install requests
    pip install tabulate
    pip install future 
    ```

1.  在您的 Python 环境中安装 H2O-3 库：

    ```py
    pip install h2o
    ```

请参阅[`h2o-release.s3.amazonaws.com/h2o/rel-zumbo/1/index.html`](http://h2o-release.s3.amazonaws.com/h2o/rel-zumbo/1/index.html)（**在 Python 中安装**选项卡）以在 Conda 中安装 H2O-3。

您现在可以本地运行 H2O-3 了。让我们看看如何做到这一点。

## 第 2 步 – 启动您的 H2O-3 集群并编写代码

要启动本地单节点 H2O-3 集群，只需在您的 Python IDE 中运行以下命令：

```py
import h2o
```

```py
h2o.init()
```

```py
# write h2o-3 code, including code samples in this book
```

您现在可以编写您的 H2O-3 代码，包括本书中的所有示例。请参阅*第二章*，*平台组件和关键概念*，以获取`Hello World`代码示例及其底层解释。

Java 依赖项 - 仅当本地运行时

H2O-3 集群（不是 Python 客户端）在 Java 上运行。因为您在这里在本地机器上运行集群（代表单节点集群），您必须安装 Java。当您使用 Python 客户端连接到您的企业 Kubernetes 或 Hadoop 环境中的远程 H2O 集群时，不需要 Java。

现在，让我们看看如何设置我们的环境，以便在我们的本地机器上编写 Sparkling Water 代码。

# 本地 Sparkling Water 集群

在本地运行 Sparkling Water 与在本地运行 H2O-3 类似，但需要 Spark 依赖项。有关 Spark、Python 和 H2O 组件的完整解释，请参阅此链接：[`docs.h2o.ai/sparkling-water/3.2/latest-stable/doc/pysparkling.html`](https://docs.h2o.ai/sparkling-water/3.2/latest-stable/doc/pysparkling.html)。

我们在这里将使用 Spark 3.2。要使用不同版本的 Spark，请访问以下链接的 H2O 下载页面中的**Sparkling Water**部分：[`h2o.ai/resources/download/`](https://h2o.ai/resources/download/)。

对于您的 Sparkling Water Python 客户端，您必须使用 Python 2.7.x、3.5.x、3.6.x 或 3.7.x。我们在这里将从 Jupyter 笔记本中运行 Sparkling Water。

## 第 1 步 - 在本地安装 Spark

按照以下步骤在本地安装 Spark：

1.  前往[`spark.apache.org/downloads.html`](https://spark.apache.org/downloads.html)下载 Spark。进行以下选择，然后下载：

    +   Spark 版本：3.2.x

    +   包类型：为 Hadoop 3.3 及更高版本预构建

1.  解压下载的文件。

1.  设置以下环境变量（以下为 macOS 示例）：

    ```py
    export SPARK_HOME="/path/to/spark/folder"
    export MASTER="local[*]"
    ```

现在，让我们在我们的 Python 环境中安装 Sparkling Water 库。

## 第 2 步 - 在 Python 中安装 Sparkling Water

安装以下模块：

1.  在您的 Python 环境中安装依赖项：

    ```py
    pip install requests
    pip install tabulate
    pip install future 
    ```

1.  安装 Sparkling Water Python 模块（称为`PySparkling`）。请注意，此处模块引用 Spark 3.2：

    ```py
    pip install h2o_pysparkling_3.2
    ```

接下来，让我们安装一个交互式 Shell。

## 第 3 步 - 安装 Sparkling Water Python 交互式 Shell

要在本地运行 Sparkling Water，我们需要安装一个交互式 Shell 来在 Spark 上启动 Sparkling Water 集群。（这仅在本地运行 Sparkling Water 时需要；Enterprise Steam 在您的企业集群上运行时负责此操作。）为此，请执行以下步骤：

1.  通过导航到[`h2o.ai/resources/download/`](https://h2o.ai/resources/download/)的**Sparkling Water**部分，点击**Sparkling Water For Spark 3.2**，然后最终点击**下载 Sparkling Water**按钮来下载交互式 Shell。

1.  解压下载的文件。

现在，让我们启动一个 Sparkling Water 集群，并从 Jupyter 笔记本中访问它。

## 第 4 步 – 在 Sparkling Water 外壳上启动 Jupyter 笔记本

我们假设您已经在与步骤*2*中安装的相同 Python 环境中安装了 Jupyter Notebook。执行以下步骤以启动一个 Jupyter 笔记本：

1.  在命令行中，导航到您在第*3*步中解压缩下载的目录。

1.  启动 Sparkling Water 交互式外壳和其中的 Jupyter 笔记本：

    +   对于 macOS，请使用以下命令：

        ```py
        PYSPARK_DRIVER_PYTHON="ipython" \
        PYSPARK_DRIVER_PYTHON_OPTS="notebook" \
        bin/pysparkling
        ```

    +   对于 Windows，请使用以下命令：

        ```py
        SET PYSPARK_DRIVER_PYTHON=ipython
        SET PYSPARK_DRIVER_PYTHON_OPTS=notebook
        bin/pysparkling
        ```

您的 Jupyter 笔记本应该在浏览器中启动。

现在，让我们编写 Sparkling Water 代码。

## 第 5 步 – 启动您的 Sparkling Water 集群并编写代码

在您的 Jupyter 笔记本中，输入以下代码以开始：

1.  启动您的 Sparkling Water 集群：

    ```py
    from pysparkling import *
    import h2o
    hc = H2OContext.getOrCreate()
    hc
    ```

1.  测试安装：

    ```py
    localdata = "/path/to/my/csv"
    mysparkdata = spark.read.load(localdata, format="csv")
    myH2Odata = hc.asH2OFrame(mysparkdata)
    ```

您现在可以使用 H2O 和 Spark 代码构建模型，准备就绪。

# H2O-3 集群在 H2O AI Cloud 的 90 天免费试用环境中

在这里，您必须与企业 Steam 交互以运行 H2O-3。在这种情况下，您需要在 Python 客户端环境中安装`h2osteam`模块，就像我们在本地运行 H2O-3 时做的那样，除了安装`h2o`模块。

## 第 1 步 – 获取 H2O AI Cloud 的 90 天试用期

在这里获取 H2O AI Cloud 的试用访问权限：[`h2o.ai/freetrial`](https://h2o.ai/freetrial)。

当您完成所有步骤并可以登录到 H2O AI Cloud 时，我们就可以开始运行作为 H2O AI Cloud 平台一部分的 H2O-3 集群。以下是下一步。

## 第 2 步 – 设置您的 Python 环境

要设置您的 Python 客户端环境，请执行以下步骤：

1.  登录到 H2O AI Cloud，然后从侧边栏点击**Python 客户端**选项，点击`h2osteam`库：

![图 15.1 – 企业 Steam]

](img/B16721_15_001.jpg)

![图 15.1 – 企业 Steam]

1.  通过运行以下命令在您的 Python 环境中安装`h2osteam`库：

    ```py
    pip install /path/to/download.whl
    ```

这里，`/path/to/download.whl`被替换为您实际的路径。

1.  您还需要安装`h2o`库。为此，执行以下操作：

    ```py
    pip install requests
    pip install tabulate
    pip install future 
    pip install h2o
    ```

现在，让我们使用 Steam 启动一个 H2O 集群，然后在 Python 中编写 H2O 代码。

## 第 3 步 – 启动您的集群

按照以下步骤启动您的 H2O 集群，该集群在 Kubernetes 服务器集群上完成：

1.  在企业 Steam 中，点击侧边栏上的**H2O**，然后点击**启动新集群**按钮。

1.  您现在可以配置您的 H2O 集群并给它命名。请确保从下拉菜单中选择最新的 H2O 版本，它应该与您在上一步骤中安装的库相匹配。

1.  配置完成后，点击**启动集群**按钮，等待集群启动完成。

1.  您需要企业 Steam 的 URL 来从 Jupyter 笔记本或其他 Python 客户端连接到它。在 Steam 中，将 URL 从`https`复制到`h2o.ai`，包括在内。

## 第 4 步 – 编写 H2O-3 代码

我们现在可以开始编写代码（例如在 Jupyter 中）来构建我们刚刚启动的 H2O-3 集群上的模型。在打开 Python 客户端后执行以下步骤：

1.  导入您的库并连接到 Enterprise Steam：

    ```py
    import h2o
    import h2osteam
    from h2osteam.clients import H2oKubernetesClient
    conn = h2osteam.login(
        url="https://SteamURL, 
    verify_ssl=False,
        username="yourH2OAICloudUserName", 
        password=" yourH2OAICloudPassword")
    ```

    重要提示

    在撰写本文时，90 天 H2O AI Cloud 试用版的 URL 为[`steam.cloud.h2o.ai`](https://steam.cloud.h2o.ai)。

    对于密码，您可以使用您登录 H2O AI Cloud 试用环境的登录密码，或者您可以使用从 Enterprise Steam 配置页面生成的临时个人访问令牌。

1.  连接到您在 Enterprise Steam 中启动的 H2O 集群：

    ```py
    cluster = H2oKubernetesClient().get_cluster(
        name="yourClusterName", 
        created_by="yourH2OAICloudUserName")
    cluster.connect()
    # you are now ready to write code to run on this H2O cluster
    ```

您现在可以编写您的 H2O-3 代码，包括本书中的所有示例。
