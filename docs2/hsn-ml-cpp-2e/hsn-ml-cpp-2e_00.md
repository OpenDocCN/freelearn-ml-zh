# 前言

C++ 可以让你的机器学习（ML）模型运行得更快、更高效。本书教你机器学习的基础知识，并展示如何使用 C++ 库。它解释了如何创建监督学习和无监督学习模型。

你将亲自动手调整和优化模型以适应不同的使用场景，本书将帮助你进行模型选择和性能测量。本书涵盖了产品推荐、集成学习、异常检测、情感分析和使用现代 C++ 库进行目标识别等技术。此外，你还将学习如何处理移动平台上的生产部署挑战，以及 ONNX 模型格式如何帮助你完成这些任务。

本版更新了关键主题，如使用迁移学习和基于 Transformer 的模型实现情感分析，以及使用 MLflow 跟踪和可视化机器学习实验。此外，还增加了一个关于使用 Optuna 进行超参数选择的新章节。关于将模型部署到移动平台的部分得到了扩展，增加了使用 C++ 在 Android 上进行实时目标检测的详细解释。

在阅读完这本 C++ 书籍之后，你将拥有实际的机器学习和 C++ 知识，以及使用 C++ 构建强大机器学习系统的技能。

# 本书面向的对象

如果你想要使用流行的 C++ 语言开始学习机器学习算法和技术，那么这本书就是为你准备的。除了作为 C++ 机器学习入门的有用课程外，这本书还会吸引那些希望使用 C++ 在生产环境中实现不同机器学习模型的数据分析师、数据科学家和机器学习开发者，这对于某些特定平台，例如嵌入式设备，可能很有用。要开始阅读这本书，需要具备 C++ 编程语言、线性代数和基本微积分知识。

# 本书涵盖的内容

*第一章*，*使用 C++ 的机器学习简介*，将引导你了解机器学习的基本知识，包括线性代数概念、机器学习算法类型及其构建模块。

*第二章*，*数据处理*，展示了如何从不同的文件格式加载数据用于机器学习模型训练，以及如何在各种 C++ 库中初始化数据集对象。

*第三章*，*衡量性能和选择模型*，展示了如何衡量各种类型机器学习模型的性能，如何选择最佳的超参数集以实现更好的模型性能，以及如何在各种 C++ 和外部库中使用网格搜索方法进行模型选择。

*第四章*, *聚类*，讨论了根据对象的本质特征进行分组算法，解释了我们通常为什么使用无监督算法来解决这类任务，最后概述了各种聚类算法及其在不同 C++库中的实现和使用。

*第五章*, *异常检测*，讨论了异常和新颖性检测任务的基础知识，并引导你了解不同类型的异常检测算法、它们的实现以及在各种 C++库中的使用。

*第六章*, *降维*，讨论了各种降维算法，这些算法保留了数据的本质特征，以及它们在不同 C++库中的实现和使用。

*第七章*, *分类*，展示了分类任务是什么以及它与聚类任务的区别。你将了解各种分类算法、它们的实现以及在各种 C++库中的使用。

*第八章*, *推荐系统*，使你对推荐系统概念熟悉。你将了解处理推荐任务的不同方法，并看到如何使用 C++语言解决这类任务。

*第九章*, *集成学习*，讨论了将多个机器学习模型结合以获得更高准确性和处理学习问题的各种方法。你将遇到使用不同 C++库的集成实现。

*第十章*, *用于图像分类的神经网络*，使你对人工神经网络的 fundamentals 熟悉。你将遇到基本构建块、所需的数学概念和学习算法。你将了解提供神经网络实现功能的不同 C++库。此外，本章还将展示使用 PyTorch 库实现图像分类的深度卷积网络的实现。

*第十一章*, *使用 BERT 和迁移学习进行情感分析*，介绍了**大型语言模型**（**LLMs**），并简要描述了它们的工作原理。它还将展示如何使用迁移学习技术，利用预训练的 LLMs，通过 PyTorch 库实现情感分析。

*第十二章*, *导出和导入模型*，展示了如何使用各种 C++库保存和加载模型参数和架构。此外，你将看到如何使用 ONNX 格式，通过 Caffe2 库的 C++ API 加载和使用预训练模型。

*第十三章*，*跟踪和可视化机器学习实验*，展示了如何使用 MLflow 工具包来跟踪和可视化您的机器学习实验。可视化对于理解实验中的模式、关系和趋势至关重要。实验跟踪允许您比较结果、识别最佳实践并避免重复错误。

*第十四章*，*在移动平台上部署模型*，指导您使用 Android 平台上的神经网络开发用于设备相机图像目标检测的应用程序。

# 为了充分利用本书

为了能够编译和运行本书中包含的示例，您需要配置特定的开发环境。所有代码示例都已使用 Ubuntu Linux 22.04 版本的发行版进行测试。以下列表概述了您需要在 Ubuntu 平台上安装的包：

+   `unzip`

+   `build-essential`

+   `gdb`

+   `git`

+   `libfmt-dev`

+   `wget`

+   `cmake`

+   `python3`

+   `python3-pip`

+   `python-is-python3`

+   `libblas-dev`

+   `libopenblas-dev`

+   `libfftw3-dev`

+   `libatlas-base-dev`

+   `liblapacke-dev`

+   `liblapack-dev`

+   `libboost-all-dev`

+   `libopencv-core4.5d`

+   `libopencv-imgproc4.5d`

+   `libopencv-dev`

+   `libopencv-highgui4.5d`

+   `libopencv-highgui-dev`

+   `libhdf5-dev`

+   `libjson-c-dev`

+   `libx11-dev`

+   `openjdk-8-jdk`

+   `openjdk-17-jdk`

+   `ninja-build`

+   `gnuplot`

+   `vim`

+   `python3-venv`

+   `libcpuinfo-dev`

+   `libspdlog-dev`

您需要一个版本不低于 2.27 的 `cmake` 包。在 Ubuntu 22.04 上，您需要手动下载并安装它。例如，可以按照以下步骤操作：

```py
wget https://github.com/Kitware/CMake/releases/download/v3.27.5/cmake-3.27.5-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh
export PATH="/usr/bin/cmake/bin:${PATH}"
```

此外，您还需要为 Python 安装额外的包，这可以通过以下命令完成：

```py
pip install pyyaml
pip install typing
pip install typing_extensions
pip install optuna
pip install torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install mlflow==2.15.0
```

除了开发环境之外，您还需要检查第三方库的源代码示例并构建它们。大多数这些库都在积极开发中，因此您需要提供特定的版本（Git 标签），以便我们确保代码示例的兼容性。以下表格显示了您需要检查的库、它们的仓库 URL 以及要检查的提交的标签或哈希号：

| **库仓库** | **分支/标签** | **提交** |
| --- | --- | --- |
| [`bitbucket.org/blaze-lib/blaze.git`](https://bitbucket.org/blaze-lib/blaze.git) | v3.8.2 |  |
| [`github.com/arrayfire/arrayfire`](https://github.com/arrayfire/arrayfire) | v3.8.3 |  |
| [`github.com/flashlight/flashlight.git`](https://github.com/flashlight/flashlight.git) | v0.4.0 |  |
| [`github.com/davisking/dlib`](https://github.com/davisking/dlib) | v19.24.6 |  |
| [`gitlab.com/conradsnicta/armadillo-code`](https://gitlab.com/conradsnicta/armadillo-code) | 14.0.x |  |
| [`github.com/xtensor-stack/xtl`](https://github.com/xtensor-stack/xtl) | 0.7.7 |  |
| [`github.com/xtensor-stack/xtensor`](https://github.com/xtensor-stack/xtensor) | 0.25.0 |  |
| [`github.com/xtensor-stack/xtensor-blas`](https://github.com/xtensor-stack/xtensor-blas) | 0.21.0 |  |
| [`github.com/nlohmann/json.git`](https://github.com/nlohmann/json.git) | v3.11.3 |  |
| [`github.com/mlpack/mlpack`](https://github.com/mlpack/mlpack) | 4.5.0 |  |
| [`gitlab.com/libeigen/eigen.git`](https://gitlab.com/libeigen/eigen.git) | 3.4.0 |  |
| [`github.com/BlueBrain/HighFive`](https://github.com/BlueBrain/HighFive) | v2.10.0 |  |
| [`github.com/yhirose/cpp-httplib`](https://github.com/yhirose/cpp-httplib) | v0.18.1 |  |
| [`github.com/Kolkir/plotcpp`](https://github.com/Kolkir/plotcpp) |  | c86bd4f5d9029986f0d5f368450d79f0dd32c7e4 |
| [`github.com/ben-strasser/fast-cpp-csv-parser`](https://github.com/ben-strasser/fast-cpp-csv-parser) |  | 4ade42d5f8c454c6c57b3dce9c51c6dd02182a66 |
| [`github.com/lisitsyn/tapkee`](https://github.com/lisitsyn/tapkee) |  | Ba5f052d2548ec03dcc6a4ac0ed8deeb79f1d43a |
| [`github.com/Microsoft/onnxruntime.git`](https://github.com/Microsoft/onnxruntime.git) | v1.19.2 |  |
| [`github.com/pytorch/pytorch`](https://github.com/pytorch/pytorch) | v2.3.1 |  |

注意，由于可能与 `onnxruntime` 使用的 `protobuf` 库版本冲突，最好最后编译和安装 PyTorch。

此外，对于最后一章，你可能需要安装 Android Studio IDE。你可以从官方网站下载它，网址为 [`developer.android.com/studio`](https://developer.android.com/studio)。除了 IDE，你还需要安装和配置 Android SDK、NDK 以及基于 Android 的 OpenCV 库。以下工具的版本是必需的：

| **名称** | **版本** |
| --- | --- |
| OpenCV | 4.10.0 |
| Linux 的 Android 命令行工具 | 9477386 |
| Android NDK | 26.1.10909125 |
| Android 平台 | 35 |

你可以使用 Android IDE 或命令行工具配置这些工具，如下所示：

```py
wget https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-android-sdk.zip
unzip opencv-4.10.0-android-sdk.zip
wget https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip
unzip commandlinetools-linux-9477386_latest.zip
./cmdline-tools/bin/sdkmanager --sdk_root=$ANDROID_SDK_ROOT "cmdline-tools;latest"
./cmdline-tools/latest/bin/sdkmanager --licenses
./cmdline-tools/latest/bin/sdkmanager "platform-tools" "tools"
./cmdline-tools/latest/bin/sdkmanager "platforms;android-35"
./cmdline-tools/latest/bin/sdkmanager "build-tools;35.0.0"
./cmdline-tools/latest/bin/sdkmanager "system-images;android-35;google_apis;arm64-v8a"
./cmdline-tools/latest/bin/sdkmanager --install "ndk;26.1.10909125"
```

另一种配置开发环境的方法是通过使用 Docker。Docker 允许你配置一个具有特定组件的轻量级虚拟机。你可以从官方 Ubuntu 软件包仓库安装 Docker。然后，使用本书提供的脚本自动配置环境。你将在 `examples` 仓库中找到 `build-env` 文件夹。以下步骤展示了如何使用 Docker 配置脚本：

1.  首先配置您的 GitHub 账户。然后，您将能够配置使用 SSH 进行 GitHub 认证，如文章《使用 SSH 连接 GitHub》中所述（[`docs.github.com/en/authentication/connecting-to-github-with-ssh`](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)）；这是首选方式。或者，您可以使用 HTTPS，并在克隆新仓库时每次都提供您的用户名和密码。如果您使用双重认证（2FA）来保护您的 GitHub 账户，那么您需要使用个人访问令牌而不是密码，如《创建个人访问令牌》文章中所述（[`docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token`](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)）。

1.  运行以下命令以创建镜像、运行它并配置环境：

    ```py
    cd docker
    docker build -t buildenv:1.0 .
    ```

1.  使用以下命令启动一个新的 Docker 容器，并将书籍示例源代码与之共享：

    ```py
    docker run -it -v [host_examples_path]:[container_examples_path] [tag name] bash
    ```

在这里，`host_examples`是从[`github.com/PacktPublishing/Hands-on-Machine-learning-with-C-Second-Edition.git`](https://github.com/PacktPublishing/Hands-on-Machine-learning-with-C-Second-Edition.git)检查出的示例源代码的路径，而`container_examples_path`是容器中的目标挂载路径，例如，`/samples`。

运行前面的命令后，您将处于具有必要配置的软件包、编译的第三方库和编程示例包的命令行环境。您可以使用此环境编译和运行本书中的代码示例。每个编程示例都配置为使用 CMake 构建系统，因此您将以相同的方式构建它们。以下脚本展示了一个构建代码示例的可能场景：

```py
cd Chapter01
mkdir build
cd build
cmake ..
cmake --build . --target all
```

这是手动方法。我们还提供了可用于构建每个示例的现成脚本。这些脚本位于存储库的`build_scripts`文件夹中。例如，第一章的构建脚本为`build_ch1.sh`，可以直接从这个文件夹中运行。

如果您打算手动配置构建环境，请注意`LIBS_DIR`变量，它应该指向所有第三方库安装的文件夹；使用提供的 Docker 环境构建脚本，它将指向`$HOME/development/libs`。

此外，您还可以配置您的本地机器环境，以便与 Docker 容器共享 X 服务器，从而能够从这个容器中运行图形用户界面应用程序。这将允许您从 Docker 容器中使用，例如，Android Studio IDE 或 C++ IDE（如 Qt Creator），而无需本地安装。以下脚本展示了如何进行此操作：

```py
xhost +local:root
docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it -v [host_examples_path]:[container_examples_path] [tag name] bash
```

为了更舒适地理解和构建代码示例，我们建议你仔细阅读每个第三方库的文档，并花一些时间学习 Docker 系统的基本知识和 Android 平台开发。此外，我们假设你具备足够的 C++语言和编译器的实际知识，并且熟悉 CMake 构建系统。

如果你正在使用这本书的数字版，我们建议你亲自输入代码或通过 GitHub 仓库（以下章节中有链接）访问代码。这样做将帮助你避免与代码复制粘贴相关的任何潜在错误。

# 下载示例代码文件

你可以从 GitHub 下载这本书的示例代码文件，链接为[`github.com/PacktPublishing/Hands-on-Machine-learning-with-C-Second-Edition`](https://github.com/PacktPublishing/Hands-on-Machine-learning-with-C-Second-Edition)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter/X 用户名。以下是一个示例：“`Dlib`库没有很多分类算法。”

代码块设置如下：

```py
std::vector<fl::Tensor> fields{train_x, train_y};
auto dataset = std::make_shared<fl::TensorDataset>(fields);
int batch_size = 8;
auto batch_dataset = std::make_shared<fl::BatchDataset>(dataset,
                                                        batch_size);
```

当我们希望引起你对代码块中特定部分的注意时，相关的行或项目将以粗体显示：

```py
[default]
exten => s,1,Dial(Zap/1|30)
exten => s,2,Voicemail(u100)
exten => s,102,Voicemail(b100)
exten => i,1,Voicemail(s0)
```

任何命令行输入或输出都按照以下方式编写：

```py
$ mkdir css
$ cd css
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词以**粗体**显示。以下是一个示例：“在 N 类的一对一策略中，训练了 N 个分类器，每个分类器将其类别与其他所有类别分开。”

小贴士或重要提示

看起来是这样的。

# 联系我们

欢迎读者反馈。

**一般反馈**：如果你对本书的任何方面有疑问，请通过电子邮件发送给我们 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们将非常感激如果你能向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**：如果你在互联网上以任何形式发现我们作品的非法副本，我们将非常感激如果你能提供位置地址或网站名称。请通过电子邮件 copyright@packtpub.com 与我们联系，并提供材料的链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且你感兴趣的是撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享你的想法

一旦你阅读了《Hands-On Machine Learning with C++》，我们很乐意听到你的想法！请[点击此处直接进入本书的亚马逊评论页面](https://packt.link/r/1-805-12057-3)并分享你的反馈。

你的评论对我们和科技社区都很重要，并将帮助我们确保我们提供高质量的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

你喜欢在路上阅读，但无法携带你的印刷书籍到处走吗？

你的电子书购买是否与你的选择设备不兼容？

别担心，现在每购买一本 Packt 图书，你都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何设备上阅读。直接从你最喜欢的技术书籍中搜索、复制和粘贴代码到你的应用程序中。

优惠不仅限于此，你还可以获得独家折扣、时事通讯和每日免费内容的每日电子邮件。

按照以下简单步骤获取福利：

1.  扫描下面的二维码或访问以下链接

![](img/B19849_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781805120575`](https://packt.link/free-ebook/9781805120575)

1.  提交你的购买证明

1.  就这些！我们将直接将你的免费 PDF 和其他福利发送到你的电子邮件。

# 第一部分：机器学习概述

在这部分，我们将借助 C++和各种机器学习框架的示例，深入探讨机器学习的基础知识。我们将展示如何从各种文件格式加载数据，并描述模型性能测量技术和最佳模型选择方法。

本部分包括以下章节：

+   *第一章*，*使用 C++的机器学习简介*

+   *第二章*，*数据处理*

+   *第三章*，*性能测量和模型选择*
