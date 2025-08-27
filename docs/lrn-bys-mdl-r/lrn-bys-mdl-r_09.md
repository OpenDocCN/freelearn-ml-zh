# 第九章。大数据规模下的贝叶斯建模

当我们在第三章中学习了贝叶斯推理原理，即《介绍贝叶斯推理》时，我们发现随着训练数据量的增加，数据对参数估计的贡献超过了先验分布的贡献。此外，参数估计的不确定性也会降低。因此，你可能想知道为什么在大规模数据分析中需要贝叶斯建模。为了回答这个问题，让我们看看这样一个问题，即为电子商务产品构建推荐系统。

在典型的电子商务商店中，可能会有数百万用户和数万种产品。然而，每个用户在其一生中可能只购买过商店中所有产品的一小部分（不到 10%）。让我们假设电子商务商店正在收集每个销售产品的用户反馈，以 1 到 5 分的评分尺度进行。然后，商店可以创建一个用户-产品评分矩阵来捕捉所有用户的评分。在这个矩阵中，行对应于用户，列对应于产品。每个单元格的值将是用户（对应于行）对产品（对应于列）给出的评分。现在，很容易看出，尽管这个矩阵的整体大小很大，但只有不到 10%的条目会有值，因为每个用户只从商店购买了不到 10%的产品。所以，这是一个高度稀疏的数据集。每当存在一个机器学习任务，尽管整体数据量很大，但数据高度稀疏时，可能会发生过拟合，并且应该依赖于贝叶斯方法（参考本章“参考文献”部分的第 1 条）。此外，许多模型，如贝叶斯网络、潜在狄利克雷分配和深度信念网络，都是建立在贝叶斯推理范式之上的。

当这些模型在大数据集上训练，例如来自路透社的文本语料库时，那么潜在的问题是大规模贝叶斯建模。实际上，贝叶斯建模是计算密集型的，因为我们必须估计参数的整个后验分布，并且还要对预测进行模型平均。大数据集的存在将使情况变得更糟。那么，我们能够使用哪些计算框架在 R 中进行大规模贝叶斯学习呢？在接下来的两个部分中，我们将讨论这个领域的一些最新发展。

# 使用 Hadoop 进行分布式计算

在过去十年中，当两位来自 Google 的研究工程师开发了一种名为 **MapReduce** 框架的计算范式以及一个相关的分布式文件系统 Google 文件系统（本章 *参考文献* 部分的参考 2）时，分布式计算取得了巨大的进步。后来，Yahoo 开发了一个名为 **Hadoop** 的开源分布式文件系统版本，这成为了大数据计算的一个标志。Hadoop 通过将数据分布到多台计算机并在每个节点上从磁盘本地执行计算，非常适合处理无法适应单个大型计算机内存的大量数据。一个例子是从日志文件中提取相关信息，通常一个月的数据量在千兆字节级别。

要使用 Hadoop，必须使用 MapReduce 框架编写程序以并行化计算。Map 操作将数据分割成多个键值对，并发送到不同的节点。在每个节点上，对每个键值对进行计算。然后，有一个洗牌操作，将具有相同键值的所有键值对聚集在一起。之后，Reduce 操作将前一步计算中对应相同键的所有结果求和。通常，这些 MapReduce 操作可以使用称为 **Pig** 的高级语言编写。也可以使用 **RHadoop** 包在 R 中编写 MapReduce 程序，我们将在下一节中描述。

# 从 R 使用 RHadoop 操作 Hadoop

RHadoop 是一系列开源包的集合，R 用户可以使用这些包管理和分析存储在 **Hadoop 分布式文件系统**（**HDFS**）中的数据。在后台，RHadoop 将这些操作转换为 Java 中的 MapReduce 操作并在 HDFS 上运行。

RHadoop 中的各种包及其用途如下：

+   **rhdfs**：使用此包，用户可以从 R 连接到 HDFS 并执行基本操作，如读取、写入和修改文件。

+   **rhbase**：这是连接到 HBASE 数据库并读取、写入和修改表的包。

+   **plyrmr**：使用此包，R 用户可以执行常见的数据操作任务，如数据集的切片和切块。这与 **plyr** 或 **reshape2** 等包的功能类似。

+   **rmr2**：使用此包，用户可以在 R 中编写 MapReduce 函数并在 HDFS 中执行它们。

与本书中讨论的其他包不同，与 RHadoop 相关的包不可从 CRAN 获取。它们可以从 GitHub 仓库 [`github.com/RevolutionAnalytics`](https://github.com/RevolutionAnalytics) 下载，并从本地驱动器安装。

下面是一个使用 rmr2 包编写的 MapReduce 代码示例，用于统计语料库中的单词数量（本章 *参考文献* 部分的参考 3）：

1.  第一步是加载 `rmr2` 库：

    ```py
    >library(rmr2)
    >LOCAL <- T #to execute rmr2 locally
    ```

1.  第二步涉及编写 Map 函数。这个函数将文本文档中的每一行分割成单词。每个单词被视为一个标记。该函数输出键值对，其中每个不同的单词是一个 *键*，*值 = 1*：

    ```py
    >#map function
    >map.wc <- function(k,lines){ 
           words.list <- strsplit(lines,'\\s+^' )
            words <- unlist(words.list)         
            return(keyval(words,1))
        }
    ```

1.  第三步涉及编写 Reduce 函数。这个函数将来自不同 Mapper 的相同 *键* 进行分组并求和它们的 *值*。由于在这种情况下，每个单词都是一个 *键*，且 *值 = 1*，Reduce 的输出将是单词的计数：

    ```py
    >#reduce function
    >reduce.wc<-function(word,counts){
              return(keyval(word,sum(counts) ))
    }
    ```

1.  第四步涉及编写一个结合 Map 和 Reduce 函数的词频统计函数，并在名为 `hdfs.data` 的文件上执行此函数，该文件存储在包含输入文本的 HDFS 中：

    ```py
    >#word count function
    >wordcount<-function(input,output=NULL){
                mapreduce(input = input,output = output,input.format = "text",map = map.wc,reduce = reduce.wc,combine = T)
    }
    >out<-wordcount(hdfs.data,hdfs.out)
    ```

1.  第五步涉及从 HDFS 获取输出文件并打印前五行：

    ```py
    >results<-from.dfs(out)
    >results.df<-as.data.frame(results,stringAsFactors=F)
    >colnames(results.df)<-c('word^' ,^' count^')
    >head(results.df)
    ```

# Spark – 内存分布式计算

Hadoop 的问题之一是在 MapReduce 操作之后，结果文件被写入硬盘。因此，当进行大量数据处理操作时，硬盘上会有许多读写操作，这使得 Hadoop 的处理速度非常慢。此外，网络延迟，即在不同节点之间洗牌数据所需的时间，也加剧了这个问题。另一个缺点是，无法从存储在 HDFS 中的文件中进行实时查询。对于机器学习问题，在训练阶段，MapReduce 不会在迭代中持久化。所有这些都使得 Hadoop 不是一个理想的机器学习平台。

2009 年，加州大学伯克利分校的 AMP 实验室发明了这种解决方案。这是罗马尼亚出生的计算机科学家 Matei Zaharia 博士的博士论文 *Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing*（本章“参考文献”部分的第 4 条参考文献）的成果，该论文催生了 Spark 项目，最终成为 Apache 下的一个完全开源项目。Spark 是一个内存分布式计算框架，解决了之前提到的许多 Hadoop 问题。此外，它支持比 MapReduce 更多的操作类型。Spark 可以用于处理迭代算法、交互式数据挖掘和流式应用。它基于一个称为 **Resilient Distributed Datasets**（**RDD**）的抽象。类似于 HDFS，它也是容错的。

Spark 是用一种名为 Scala 的语言编写的。它具有从 Java 和 Python 使用的接口，从最近的 1.4.0 版本开始；它还支持 R。这被称为 SparkR，我们将在下一节中描述。Spark 中可用的四个库类别是 SQL 和 DataFrames、Spark Streaming、MLib（机器学习）和 GraphX（图算法）。目前，SparkR 仅支持 SQL 和 DataFrames；其他肯定在路线图中。Spark 可以从 Apache 项目页面[`spark.apache.org/downloads.html`](http://spark.apache.org/downloads.html)下载。从 1.4.0 版本开始，SparkR 包含在 Spark 中，无需单独下载。

# SparkR

与 RHadoop 类似，SparkR 是一个 R 包，允许 R 用户通过`RDD`类使用 Spark API。例如，使用 SparkR，用户可以从 RStudio 上运行 Spark 作业。SparkR 可以从 RStudio 中调用。为了启用此功能，请在 R 启动时初始化环境的`.Rprofile`文件中包含以下行：

```py
Sys.setenv(SPARK_HOME/.../spark-1.5.0-bin-hadoop2.6")
#provide the correct path where spark downloaded folder is kept for SPARK_HOME 
.libPaths(c(file.path(Sys.getenv("SPARK_HOME"),""R",""lib"),".libPaths()))
```

完成这些后，启动 RStudio 并输入以下命令以开始使用 SparkR：

```py
>library(SparkR)
>sc <- sparkR.init(master="local")
```

如前所述，当本章撰写时，SparkR 支持 R 的有限功能。这主要包括数据切片和切块以及汇总统计函数。当前版本不支持使用贡献的 R 包；然而，计划在未来的版本中实现。在机器学习方面，当前 SparkR 支持`glm()`函数。我们将在下一节中做一个示例。

# 使用 SparkR 进行线性回归

在以下示例中，我们将说明如何使用 SparkR 进行机器学习。为此，我们将使用与第五章中线性回归相同的能源效率测量数据集，*贝叶斯回归模型*：

```py
>library(SparkR)
>sc <- sparkR.init(master="local")
>sqlContext <- sparkRSQL.init(sc)

#Importing data
>df <- read.csv("/Users/harikoduvely/Projects/Book/Data/ENB2012_data.csv",header = T)
>#Excluding variable Y2,X6,X8 and removing records from 768 containing mainly null values
>df <- df[1:768,c(1,2,3,4,5,7,9)]
>#Converting to a Spark R Dataframe
>dfsr <- createDataFrame(sqlContext,df) 
>model <- glm(Y1 ~ X1 + X2 + X3 + X4 + X5 + X7,data = dfsr,family = "gaussian")
 > summary(model)
```

# 云计算集群

为了使用 Hadoop 和相关 R 包处理大型数据集，需要一个计算机集群。在当今世界，使用亚马逊、微软和其他人提供的云计算服务很容易。只需支付使用的 CPU 和存储量。无需在基础设施上进行前期投资。前四大云计算服务是亚马逊的 AWS、微软的 Azure、谷歌的 Compute Cloud 和 IBM 的 Bluemix。在本节中，我们将讨论在 AWS 上运行 R 程序。特别是，你将学习如何创建 AWS 实例；在该实例中安装 R、RStudio 和其他包；开发和运行机器学习模型。

## 亚马逊网络服务

广为人知的 AWS，即亚马逊网络服务，始于 2002 年亚马逊的一个内部项目，旨在满足支持其电子商务业务的动态计算需求。这作为一个**基础设施即服务**项目发展起来，2006 年亚马逊向世界推出了两项服务，**简单存储服务**（**S3**）和**弹性计算云**（**EC2**）。从那时起，AWS 以惊人的速度增长。如今，他们拥有超过 40 种不同类型的服务，使用数百万台服务器。

## 在 AWS 上创建和运行计算实例

学习如何设置 AWS 账户并开始使用 EC2 的最佳地方是亚马逊 Kindle 商店中免费提供的电子书，名为 *Amazon Elastic Compute Cloud (EC2) 用户指南*（参考本章节 *参考文献* 部分的第 6 条）。

这里，我们仅总结涉及此过程的基本步骤：

1.  创建 AWS 账户。

1.  登录 AWS 管理控制台 ([`aws.amazon.com/console/`](https://aws.amazon.com/console/))。

1.  点击 EC2 服务。

1.  选择 **Amazon Machine Instance (AMI**)。

1.  选择实例类型。

1.  创建公私钥对。

1.  配置实例。

1.  添加存储。

1.  标记实例。

1.  配置安全组（指定谁可以访问实例的策略）。

1.  审查并启动实例。

使用 SSH（从 Linux/Ubuntu）、Putty（从 Windows）或使用配置安全时提供的私钥和启动时给出的 IP 地址通过浏览器登录到您的实例。这里，我们假设您启动的实例是一个 Linux 实例。

## 安装 R 和 RStudio

要安装 R 和 RStudio，您需要成为认证用户。因此，创建一个新用户并授予用户管理员权限（sudo）。之后，从 Ubuntu shell 执行以下步骤：

1.  编辑 `/etc/apt/sources.list` 文件。

1.  在末尾添加以下行：

    ```py
    deb http://cran.rstudio.com/bin/linux/ubuntu trusty .
    ```

1.  获取运行存储库的密钥：

    ```py
    sudo apt-key adv  --keyserver keyserver.ubuntu.com –recv-keys 51716619E084DAB9

    ```

1.  更新包列表：

    ```py
    sudo apt-get update

    ```

1.  安装 R 的最新版本：

    ```py
    sudo apt-get install r-base-core

    ```

1.  安装 gdebi 以从本地磁盘安装 Debian 软件包：

    ```py
    sudo apt-get install gdebi-core

    ```

1.  下载 RStudio 软件包：

    ```py
    wget http://download2.rstudio.org/r-studio-server-0.99.446-amd64.deb

    ```

1.  安装 RStudio：

    ```py
    sudo gdebi r-studio-server-0.99.446-amd64.deb

    ```

安装成功完成后，运行在您的 AWS 实例上的 RStudio 可以通过浏览器访问。为此，请打开浏览器并输入 URL `<your.aws.ip.no>:8787`。

如果你能够使用运行在 AWS 实例上的 RStudio，那么你可以从 RStudio 安装其他包，例如 rhdfs、rmr2 等，并在 R 中构建任何机器学习模型，然后在 AWS 云上运行它们。

除了 R 和 RStudio，AWS 还支持 Spark（因此也支持 SparkR）。在下一节中，您将学习如何在 EC2 集群上运行 Spark。

## 在 EC2 上运行 Spark

您可以使用位于您本地机器 Spark 的 `ec2` 目录中的 `spark-ec2` 脚本来在 Amazon EC2 上启动和管理 Spark 集群。要在 EC2 上启动 Spark 集群，请按照以下步骤操作：

1.  在您的本地机器 Spark 文件夹中的 `ec2` 目录下。

1.  运行以下命令：

    ```py
    ./spark-ec2 -k <keypair> -i <key-file> -s <num-slaves> launch <cluster-name>

    ```

    在这里，`<keypair>` 是您用于启动本章“在 AWS 上创建和运行计算实例”部分中提到的 EC2 服务的密钥对名称。`<key-file>` 是您本地机器上私钥已下载并保存的路径。工作节点数由 `<num-slaves>` 指定。

1.  要在集群上运行您的程序，请首先使用以下命令通过 SSH 连接到集群：

    ```py
    ./spark-ec2 -k <keypair> -i <key-file> login <cluster-name>

    ```

    登录到集群后，您可以使用与在本地机器上使用相同的方式使用 Spark。

在 Spark 文档和 AWS 文档中可以找到有关如何在 EC2 上使用 Spark 的更多详细信息（章节“参考文献”部分的第 5、6 和 7 条）。

## Microsoft Azure

Microsoft Azure 对 R 和 Spark 提供了全面支持。Microsoft 收购了 Revolution Analytics 公司，该公司开始构建并支持 R 的企业版。除此之外，Azure 还有一个机器学习服务，其中包含一些贝叶斯机器学习模型的 API。有关如何在 Azure 上启动实例以及如何使用其机器学习服务的视频教程可以在 Microsoft Virtual Academy 网站上找到（章节“参考文献”部分的第 8 条）。

## IBM Bluemix

Bluemix 通过其实例上可用的完整 R 库集对 R 完全支持。IBM 在其路线图计划中也包含了将 Spark 集成到其云服务中。更多详细信息可以在他们的文档页面上找到（章节“参考文献”部分的第 9 条）。

# 其他用于大规模机器学习的 R 包

除了 RHadoop 和 SparkR 之外，还有几个专门为大规模机器学习构建的本地 R 包。在这里，我们简要概述它们。感兴趣的读者应参考 *CRAN 任务视图：使用 R 进行高性能和并行计算*（章节“参考文献”部分的第 10 条）。

虽然 R 是单线程的，但存在几个用于 R 的并行计算包。其中一些知名的包是 **Rmpi**（流行的消息传递接口的 R 版本）、**multicore**、**snow**（用于构建 R 集群）和 **foreach**。从 R 2.14.0 版本开始，一个新的名为 **parallel** 的包开始随基础 R 一起发货。我们将在下面讨论其一些功能。

## 并行 R 包

**并行**包建立在多核和 snow 包之上。它适用于在多个数据集上运行单个程序，例如 K 折交叉验证。它可以用于在单个机器上的多个 CPU/核心或跨多台机器进行并行化。对于跨机器集群的并行化，它通过 Rmpi 包调用 MPI（消息传递接口）。

我们将通过计算列表 1:100000 的数字平方的简单示例来说明并行包的使用。此示例在 Windows 上无法工作，因为相应的 R 不支持多核包。它可以在任何 Linux 或 OS X 平台上进行测试。

执行此操作的顺序方法是使用`lapply`函数，如下所示：

```py
>nsquare <- function(n){return(n*n)}
>range <- c(1:100000)
>system.time(lapply(range,nsquare))
```

使用并行包中的`mclapply`函数，可以在更短的时间内完成这个计算：

```py
>library(parallel) #included in R core packages, no separate installation required
>numCores<-detectCores( )  #to find the number of cores in the machine
>system.time(mclapply(range,nsquare,mc.cores=numCores))
```

如果数据集非常大，需要使用计算机集群，我们可以使用`parLapply`函数在集群上运行程序。这需要 Rmpi 包：

```py
>install.packages(Rmpi)#one time
>library(Rmpi)
>numNodes<-4 #number of workers nodes
>cl<-makeCluster(numNodes,type="MPI")
>system.time(parLapply(cl,range,nsquare))
>stopCluster(cl)
>mpi.exit( )
```

## The foreach R package

这是在 R 中的一种新的循环结构，可以在多核或集群上并行执行。它有两个重要的运算符：`%do%`用于重复执行任务，`%dopar%`用于并行执行任务。

例如，我们可以在前一个章节中讨论的平方函数中使用 foreach 包的单行命令来实现：

```py
>install.packages(foreach)#one time
>install.packages(doParallel)#one time
>library(foreach)
>library(doParallel)
>system.time(foreach(i=1:100000)   %do%  i²) #for executing sequentially
>system.time(foreach(i=1:100000)   %dopar%  i²) #for executing in parallel
```

我们还将使用`foreach`函数的例子来演示快速排序：

```py
>qsort<- function(x) {
  n <- length(x)
  if (n == 0) {
    x
  } else {
    p <- sample(n,1)
    smaller <- foreach(y=x[-p],.combine=c) %:% when(y <= x[p]) %do% y
    larger  <- foreach(y=x[-p],.combine=c) %:% when(y >  x[p]) %do% y
    c(qsort(smaller),x[p],qsort(larger))
  }
}
qsort(runif(12))
```

这些包仍在进行大量开发。它们还没有被广泛用于贝叶斯建模。它们很容易用于贝叶斯推断应用，如蒙特卡洛模拟。

# 练习

1.  回顾第六章中的分类问题，*贝叶斯分类模型*。使用 SparkR 的`glm()`函数重复相同的问题。

1.  使用 SparkR 回顾本章中我们做的线性回归问题。在创建 AWS 实例后，使用 AWS 上的 RStudio 服务器重复这个问题。

# 参考文献

1.  "变分贝叶斯概率矩阵分解算法的 MapReduce 实现"。在：IEEE 大数据会议。第 145-152 页。2013 年

1.  Dean J. and Ghemawat S. "MapReduce: Simplified Data Processing on Large Clusters". Communications of the ACM 51 (1). 107-113

1.  [`github.com/jeffreybreen/tutorial-rmr2-airline/blob/master/R/1-wordcount.R`](https://github.com/jeffreybreen/tutorial-rmr2-airline/blob/master/R/1-wordcount.R)

1.  Chowdhury M., Das T., Dave A., Franklin M.J., Ma J., McCauley M., Shenker S., Stoica I.和 Zaharia M. "弹性分布式数据集：内存集群计算的容错抽象"。NSDI 2012。2012 年

1.  *亚马逊弹性计算云（EC2）用户指南*，亚马逊网络服务 Kindle 电子书，更新于 2014 年 4 月 9 日

1.  Spark 文档在 AWS 上的说明，请参阅[`spark.apache.org/docs/latest/ec2-scripts.html`](http://spark.apache.org/docs/latest/ec2-scripts.html)

1.  AWS 上的 Spark 文档，请参阅[`aws.amazon.com/elasticmapreduce/details/spark/`](http://aws.amazon.com/elasticmapreduce/details/spark/)

1.  微软虚拟学院网站，请参阅[`www.microsoftvirtualacademy.com/training-courses/getting-started-with-microsoft-azure-machine-learning`](http://www.microsoftvirtualacademy.com/training-courses/getting-started-with-microsoft-azure-machine-learning)

1.  IBM Bluemix 教程，请参阅[`www.ibm.com/developerworks/cloud/bluemix/quick-start-bluemix.html`](http://www.ibm.com/developerworks/cloud/bluemix/quick-start-bluemix.html)

1.  CRAN Task View for contributed packages in R at [`cran.r-project.org/web/views/HighPerformanceComputing.html`](https://cran.r-project.org/web/views/HighPerformanceComputing.html)

# 摘要

在本书的最后一章中，我们介绍了各种实现大规模机器学习的框架。这些框架对贝叶斯学习也非常有用。例如，为了从后验分布中进行模拟，可以在机器集群上运行 Gibbs 抽样。我们学习了如何使用 RHadoop 包从 R 连接到 Hadoop，以及如何使用 SparkR 与 Spark 一起使用 R。我们还讨论了如何在 AWS 等云服务中设置集群，以及如何在它们上运行 Spark。还涵盖了某些本地并行化框架，如 parallel 和 foreach 函数。

本书的主要目的是向读者介绍使用 R 进行贝叶斯建模的领域。读者应该已经对贝叶斯机器学习模型背后的理论和概念有了很好的理解。由于示例主要是为了说明目的，我敦促读者将这些技术应用于实际问题，以更深入地理解贝叶斯推断的主题。
