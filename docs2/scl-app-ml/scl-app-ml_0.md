# 前言

许多人认为 Scala 是大数据领域的 Java 的继任者。它特别擅长在不显著影响性能的情况下分析大量数据，因此 Scala 被许多开发者和数据科学家采用。

本学习路径旨在将使用 Scala 的整个机器学习世界展现在您面前。我们将从向您介绍 Scala 中用于摄取、存储、操作、处理和可视化数据的库开始。接着，我们将介绍 Scala 中的机器学习，并深入探讨如何利用 Scala 构建和研习可以从数据中学习系统的技巧。最后，我们将全面掌握 Scala 机器学习，并传授您构建复杂机器学习项目的专业知识。

# 本学习路径涵盖的内容

模块 1，*Scala 数据科学入门*，为您提供了 Raspberry Pi 的介绍。它帮助您使用 PyGame 构建游戏，并使用 Raspberry Pi 创建实际应用。它进一步展示了 OpenCV 的高级概念中的 GPIO 和摄像头。本模块还深入探讨了设置 Web 服务器和创建网络工具。

模块 2，*Scala 机器学习入门*，通过图表、正式数学符号、源代码片段和实用技巧引导您构建 AI 应用。对 Akka 框架和 Apache Spark 集群的回顾结束了教程。

模块 3，*Scala 机器学习精通*，是本课程的最后一步。它将把您的知识提升到新的水平，并帮助您利用这些知识构建高级应用，如社交媒体挖掘、智能新闻门户等。在用 REPL 快速复习函数式编程概念后，您将看到一些设置开发环境和处理数据的实际示例。然后，我们将探讨使用 k-means 和决策树与 Spark 和 MLlib 一起工作。

# 您需要为本学习路径准备的内容

您需要以下设置来完成所有三个模块：

## 模块 1

本课程提供的示例要求您拥有一个可工作的 Scala 安装和 SBT，即*简单构建工具*，这是一个用于编译和运行 Scala 代码的命令行实用程序。我们将在下一节中向您介绍如何安装这些工具。

我们不要求使用特定的 IDE。代码示例可以编写在您喜欢的文本编辑器或 IDE 中。

### 安装 JDK

Scala 代码编译成 Java 字节码。要运行字节码，您必须安装 Java 虚拟机（JVM），它包含在 Java 开发工具包（JDK）中。有几种 JDK 实现，在本课程中，您选择哪一个并不重要。您可能已经在计算机上安装了 JDK。要检查这一点，请在终端中输入以下内容：

```py
$ java -version
java version "1.8.0_66"
Java(TM) SE Runtime Environment (build 1.8.0_66-b17)
Java HotSpot(TM) 64-Bit Server VM (build 25.66-b17, mixed mode)

```

如果您没有安装 JDK，您将收到一个错误，表明 `java` 命令不存在。

如果您已经安装了 JDK，您仍然应该验证您正在运行一个足够新的版本。重要的是次要版本号：`1.8.0_66` 中的 `8`。Java 的 `1.8.xx` 版本通常被称为 Java 8。对于本课程的前十二章，Java 7 就足够了（您的版本号应该是 `1.7.xx` 或更新的版本）。然而，您将需要 Java 8 来完成最后两章，因为 Play 框架需要它。因此，我们建议您安装 Java 8。

在 Mac 上，安装 JDK 最简单的方法是使用 Homebrew：

```py
$ brew install java

```

这将安装来自 Oracle 的 Java 8，特别是 Java 标准版开发工具包。

Homebrew 是 Mac OS X 的包管理器。如果您不熟悉 Homebrew，我强烈建议您使用它来安装开发工具。您可以在 [`brew.sh`](http://brew.sh) 上找到 Homebrew 的安装说明。

要在 Windows 上安装 JDK，请访问 [`www.oracle.com/technetwork/java/javase/downloads/index.html`](http://www.oracle.com/technetwork/java/javase/downloads/index.html)（或者，如果此 URL 不存在，请访问 Oracle 网站，然后点击“下载”并下载**Java 平台，标准版**）。选择 Windows x86 用于 32 位 Windows，或 Windows x64 用于 64 位。这将下载一个安装程序，您可以通过运行它来安装 JDK。

要在 Ubuntu 上安装 JDK，使用您发行版的包管理器安装 OpenJDK：

```py
$ sudo apt-get install openjdk-8-jdk

```

如果您正在运行一个足够旧的 Ubuntu 版本（14.04 或更早），此包将不可用。在这种情况下，您可以选择回退到 `openjdk-7-jdk`，这将允许您运行前十二章的示例，或者通过 PPA（非标准包存档）安装来自 Oracle 的 Java 标准版开发工具包：

```py
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer

```

您需要告诉 Ubuntu 优先使用 Java 8，方法如下：

```py
$ sudo update-java-alternatives -s java-8-oracle

```

### 安装和使用 SBT

简单构建工具（SBT）是一个用于管理依赖项、构建和运行 Scala 代码的命令行工具。它是 Scala 的默认构建工具。要安装 SBT，请遵循 SBT 网站的说明 ([`www.scala-sbt.org/0.13/tutorial/Setup.html`](http://www.scala-sbt.org/0.13/tutorial/Setup.html))。

当您启动一个新的 SBT 项目时，SBT 会为您下载一个特定的 Scala 版本。因此，您不需要直接在您的计算机上安装 Scala。从 SBT 管理整个依赖套件，包括 Scala 本身，是非常强大的：您不必担心在同一个项目上工作的开发者使用不同版本的 Scala 或库。

由于我们将在本课程中广泛使用 SBT，让我们创建一个简单的测试项目。如果您之前已经使用过 SBT，请跳过此部分。

创建一个名为`sbt-example`的新目录并导航到它。在这个目录内，创建一个名为`build.sbt`的文件。该文件编码了项目的所有依赖项。在`build.sbt`中写入以下内容：

```py
// build.sbt

scalaVersion := "2.11.7"
```

这指定了我们想要为项目使用的 Scala 版本。在`sbt-example`目录中打开一个终端并输入：

```py
$ sbt

```

这将启动一个交互式 shell。让我们打开一个 Scala 控制台：

```py
> console

```

这将使你能够访问项目上下文中的 Scala 控制台：

```py
scala> println("Scala is running!")
Scala is running!

```

除了在控制台运行代码，我们还将编写 Scala 程序。在`sbt-example`目录中打开一个编辑器，并输入一个基本的“hello, world”程序。将文件命名为`HelloWorld.scala`：

```py
// HelloWorld.scala

object HelloWorld extends App {
  println("Hello, world!")
}
```

返回 SBT 并输入：

```py
> run

```

这将编译源文件并运行可执行文件，打印出`"Hello, world!"`。

除了编译和运行你的 Scala 代码，SBT 还管理 Scala 依赖项。让我们指定对 Breeze 库的依赖，这是一个用于数值算法的库。按照以下方式修改`build.sbt`文件：

```py
// build.sbt

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2"
)
```

SBT 要求语句之间用空行分隔，所以请确保在`scalaVersion`和`libraryDependencies`之间留一个空行。在这个例子中，我们指定了对 Breeze 版本`"0.11.2"`的依赖。我们是如何知道使用这些坐标来指定 Breeze 的？大多数 Scala 包在其文档中都会引用确切的 SBT 字符串以获取最新版本。

如果不是这种情况，或者你正在指定对 Java 库的依赖，请访问 Maven Central 网站([`mvnrepository.com`](http://mvnrepository.com))并搜索感兴趣的包，例如“Breeze”。该网站提供了一系列包，包括几个名为`breeze_2.xx`的包。下划线后面的数字表示该包为哪个 Scala 版本编译。点击`"breeze_2.11"`以获取不同 Breeze 版本的列表。选择`"0.11.2"`。你将看到一个包含包管理器的列表以供选择（Maven、Ivy、Leiningen 等）。选择 SBT。这将打印出类似以下的一行：

```py
libraryDependencies += "org.scalanlp" % "breeze_2.11" % "0.11.2"
```

这些是你想要复制到`build.sbt`文件中的坐标。请注意，我们只是指定了`"breeze"`，而不是`"breeze_2.11"`。通过在包名前加上两个百分号`%%`，SBT 会自动解析到正确的 Scala 版本。因此，指定`%% "breeze"`与`% "breeze_2.11"`相同。

现在返回你的 SBT 控制台并运行：

```py
> reload

```

这将从 Maven Central 获取 Breeze JAR 文件。你现在可以在控制台或脚本中（在 Scala 项目的上下文中）导入 Breeze。让我们在控制台中测试一下：

```py
> console
scala> import breeze.linalg._
import breeze.linalg._

scala> import breeze.numerics._
import breeze.numerics._

scala> val vec = linspace(-2.0, 2.0, 100)
vec: breeze.linalg.DenseVector[Double] = DenseVector(-2.0, -1.9595959595959596, ...

scala> sigmoid(vec)
breeze.linalg.DenseVector[Double] = DenseVector(0.11920292202211755, 0.12351078065 ...

```

现在，你应该能够编译、运行并指定 Scala 脚本依赖项。

## 模块 2

掌握 Scala 编程语言是先决条件。阅读数学公式，方便地定义在信息框中，是可选的。然而，对数学和统计学的一些基本知识可能有助于理解某些算法的内部工作原理。

本课程使用以下库：

+   Scala 2.10.3 或更高版本

+   Java JDK 1.7.0_45 或 1.8.0_25

+   SBT 0.13 或更高版本

+   JFreeChart 1.0.1

+   Apache Commons Math 库 3.5 (第三章，*数据预处理*，第四章，*无监督学习*，和第六章，*回归和正则化*)

+   印度理工学院孟买 CRF 0.2 (第七章，*顺序数据模型*)

+   LIBSVM 0.1.6 (第八章，*核模型和支持向量机*)

+   Akka 2.2.4 或更高版本（或 Typesafe activator 1.2.10 或更高版本）(第十二章，*可扩展框架*)

+   Apache Spark 1.3.0 或更高版本(第十二章，*可扩展框架*)

## 第 3 模块

本课程基于开源软件。首先，是 Java。您可以从 Oracle 的 Java 下载页面下载 Java。您必须接受许可协议并选择适合您平台的适当镜像。不要使用 OpenJDK——它与 Hadoop/Spark 存在一些问题。

其次，Scala。如果您使用 Mac，我建议安装 Homebrew：

```py
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

```

您还将获得多个开源软件包。要安装 Scala，请运行`brew install scala`。在 Linux 平台上安装需要从[`www.scala-lang.org/download/`](http://www.scala-lang.org/download/)网站下载适当的 Debian 或 RPM 软件包。我们将使用当时最新的版本，即 2.11.7。

Spark 发行版可以从[`spark.apache.org/downloads.html`](http://spark.apache.org/downloads.html)下载。我们使用为 Hadoop 2.6 及更高版本预先构建的镜像。由于它是 Java，您只需解压软件包并从`bin`子目录中的脚本开始使用即可。

R 和 Python 软件包分别可在[`cran.r-project.org/bin`](http://cran.r-project.org/bin)和[`python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz`](http://python.org/ftp/python/%24PYTHON_VERSION/Python-%24PYTHON_VERSION.tar.xz)网站上找到。文本中具体说明了如何配置它们。尽管我们使用的软件包应该是版本无关的，但我在这本书中使用了 R 版本 3.2.3 和 Python 版本 2.7.11。

# 本学习路径适合谁

本学习路径是为熟悉 Scala 并希望学习如何创建、验证和应用机器学习算法的工程师和科学家设计的。它也将对有 Scala 编程背景的软件开发人员有益，他们希望应用机器学习。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们您对这个课程的想法——您喜欢什么或不喜欢什么。读者反馈对我们很重要，因为它帮助我们开发出您真正能从中受益的课程。

要发送给我们一般反馈，请简单地发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及课程标题。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为课程做出贡献，请参阅我们的作者指南，网址为[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您是课程书的自豪拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从您的账户中下载此课程的示例代码文件，网址为[`www.packtpub.com`](http://www.packtpub.com)。如果您在其他地方购买了此课程，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的**支持**标签上。

1.  点击**代码下载与勘误**。

1.  在**搜索**框中输入课程的名称。

1.  选择您想要下载代码文件的课程。

1.  从下拉菜单中选择您购买此课程的来源。

1.  点击**代码下载**。

您还可以通过点击 Packt Publishing 网站上课程网页上的**代码文件**按钮来下载代码文件。您可以通过在**搜索**框中输入课程名称来访问此页面。请注意，您需要登录到您的 Packt 账户。

文件下载完成后，请确保您使用最新版本解压缩或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

该课程的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Scala-Applied-Machine-Learning-Code`](https://github.com/PacktPublishing/Scala-Applied-Machine-Learning-Code)。

## 勘误

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的课程中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进后续版本的课程。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的课程，点击**勘误提交表单**链接，并输入您的勘误详情来报告。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问 [`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入课程名称。所需信息将在**勘误**部分显示。

## 盗版

互联网上对版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<copyright@packtpub.com>` 联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面的帮助。

## 询问

如果您在这门课程的任何方面遇到问题，您可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决问题。
