# 前言

# 关于本书

你是否觉得很难理解像 WhatsApp 和 Amazon 这样的知名公司如何从大量杂乱无章的数据中提取有价值的洞察？*无监督学习工作坊*将让你自信地应对混乱且没有标签的数据集，以轻松互动的方式使用无监督算法。

本书从介绍最流行的无监督学习聚类算法开始。你将了解层次聚类与 k-means 的不同，并理解如何将 DBSCAN 应用于复杂且噪声较多的数据。接下来，你将使用自编码器进行高效的数据编码。

随着学习的深入，你将使用 t-SNE 模型将高维信息转换为低维，以便更好地可视化，同时还会使用主题建模来实现自然语言处理。在后续章节中，你将使用市场篮分析来发现顾客和商家之间的关键关系，然后使用热点分析来估算某一地区的人口密度。

到本书结束时，你将掌握在混乱的数据集中应用无监督算法来发现有用模式和洞察的技能。

## 读者群体

如果你是刚入门的数据科学家，想要学习如何实现机器学习算法以构建预测模型，那么本书适合你。为了加快学习进程，建议你具备扎实的 Python 编程语言基础，因为你将编辑类和函数，而不是从零开始创建它们。

## 关于各章节

*第一章*，*聚类简介*，介绍了聚类（无监督学习中最知名的算法家族），然后深入讲解最简单、最流行的聚类算法——k-means。

*第二章*，*层次聚类*，讲解了另一种聚类技术——层次聚类，并说明它与 k-means 的不同。本章将教你两种主要的聚类方法：凝聚型和分裂型。

*第三章*，*邻域方法和 DBSCAN*，探讨了涉及邻居的聚类方法。与另外两种聚类方法不同，邻域方法允许存在未被分配到任何特定聚类的异常点。

*第四章*，*降维与 PCA*，教你如何通过主成分分析来减少特征数量，同时保持整个特征空间的解释能力，从而在大型特征空间中导航。

*第五章*，*自编码器*，向你展示如何利用神经网络找到数据编码。数据编码就像是特征的组合，能够降低特征空间的维度。自编码器还会解码数据并将其恢复到原始形式。

*第六章*，*t-分布随机邻居嵌入*，讨论了将高维数据集降维到二维或三维进行可视化的过程。与 PCA 不同，t-SNE 是一种非线性概率模型。

*第七章*，*主题建模*，探讨了自然语言处理的基本方法论。你将学习如何处理文本数据，并将潜在的狄利克雷分配和非负矩阵分解模型应用于标记与文本相关的主题。

*第八章*，*市场篮子分析*，探讨了零售业务中使用的经典分析技术。你将以可扩展的方式，构建解释项目组之间关系的关联规则。

*第九章*，*热点分析*，教你如何使用样本数据估算某些随机变量的真实人口密度。该技术适用于许多领域，包括流行病学、天气、犯罪和人口学。

## 约定

文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名显示如下：

"使用我们从`matplotlib.pyplot`导入的散点图功能绘制坐标点。"

屏幕上看到的词（例如，在菜单或对话框中）以相同的格式显示。

一段代码如下所示：

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
seeds = pd.read_csv('Seed_Data.csv')
```

新术语和重要词汇显示如下：

"**无监督学习**是一个实践领域，旨在帮助在杂乱的数据中找到模式，是当前机器学习中最令人兴奋的发展领域之一。"

长的代码片段会被截断，并在截断代码的顶部放置对应的 GitHub 代码文件名称。整个代码的永久链接将放在代码片段下方。应该如下所示：

```py
Exercise1.04-Exercise1.05.ipynb
def k_means(X, K):
    # Keep track of history so you can see K-Means in action
    centroids_history = []
    labels_history = []
    rand_index = np.random.choice(X.shape[0], K)
    centroids = X[rand_index]
    centroids_history.append(centroids)
The complete code for this step can be found at https://packt.live/2JM8Q1S.
```

## 代码呈现

跨越多行的代码使用反斜杠（`\`）分隔。当代码执行时，Python 将忽略反斜杠，并将下一行代码视为当前行的直接延续。

例如：

```py
history = model.fit(X, y, epochs=100, batch_size=5, verbose=1, \
                    validation_split=0.2, shuffle=False)
```

注释被添加到代码中，以帮助解释特定的逻辑。单行注释使用`#`符号表示，如下所示：

```py
# Print the sizes of the dataset
print("Number of Examples in the Dataset = ", X.shape[0])
print("Number of Features for each example = ", X.shape[1])
```

多行注释被三引号包围，如下所示：

```py
"""
Define a seed for the random number generator to ensure the 
result will be reproducible
"""
seed = 1
np.random.seed(seed)
random.set_seed(seed)
```

## 设置你的环境

在我们详细探讨本书之前，我们需要设置特定的软件和工具。在接下来的部分中，我们将看到如何进行这些操作。

## 硬件要求

为了获得最佳用户体验，我们推荐 8GB 内存。

## 安装 Python

接下来的部分将帮助你在 Windows、macOS 和 Linux 系统中安装 Python。

### 在 Windows 上安装 Python

1.  在官方安装页面[`www.python.org/downloads/windows/`](https://www.python.org/downloads/windows/)上找到你所需的 Python 版本。

1.  确保根据你的计算机系统安装正确的“-bit”版本，可以是 32-bit 或 64-bit。你可以在操作系统的 **系统属性** 窗口中查看此信息。

1.  下载安装程序后，简单地双击文件并按照屏幕上的用户友好提示进行操作。

### 在 Linux 上安装 Python

1.  打开终端并通过运行 `python3 --version` 验证 Python 3 是否已安装。

1.  要安装 Python 3，运行以下命令：

    ```py
    sudo apt-get update
    sudo apt-get install python3.7
    ```

1.  如果遇到问题，有很多在线资源可以帮助你排查问题。

### 在 macOS 上安装 Python

以下是在 macOS 上安装 Python 的步骤：

1.  通过按住 *Cmd* + *Space*，在打开的搜索框中输入 `terminal`，然后按 *Enter* 打开终端。

1.  通过运行 `xcode-select --install` 命令在命令行中安装 Xcode。

1.  安装 Python 3 最简单的方法是使用 Homebrew，可以通过运行 `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"` 在命令行中安装 Homebrew。

1.  将 Homebrew 添加到你的 `PATH` 环境变量中。在命令行中运行 `sudo nano ~/.profile` 打开你的配置文件，并在文件底部插入 `export PATH="/usr/local/opt/python/libexec/bin:$PATH"`。

1.  最后一步是安装 Python。在命令行中运行 `brew install python`。

1.  请注意，如果安装 Anaconda，最新版本的 Python 将会自动安装。

## 安装 pip

Python 默认不包含 `pip`（Python 的包管理器），因此我们需要手动安装它。一旦安装了 `pip`，就可以按照 *安装库* 部分提到的方法安装其余的库。安装 `pip` 的步骤如下：

1.  转到 [`bootstrap.pypa.io/get-pip.py`](https://bootstrap.pypa.io/get-pip.py) 并将文件保存为 `get-pip.py`。

1.  转到保存`get-pip.py`的文件夹。在该文件夹中打开命令行（Linux 用户使用 Bash，Mac 用户使用 Terminal）。

1.  在命令行中执行以下命令：

    ```py
    python get-pip.py
    ```

    请注意，你应该先安装 Python 后再执行此命令。

1.  一旦安装了 `pip`，你就可以安装所需的库。要安装 pandas，只需执行 `pip install pandas`。要安装某个特定版本的库，例如 `pandas` 的版本 0.24.2，可以执行 `pip install pandas=0.24.2`。

## 安装 Anaconda

Anaconda 是一个 Python 包管理器，能够轻松地安装并使用本课程所需的库。

## 在 Windows 上安装 Anaconda

1.  Windows 上的 Anaconda 安装非常用户友好。请访问下载页面，在 [`www.anaconda.com/distribution/#download-section`](https://www.anaconda.com/distribution/#download-section) 获取安装可执行文件。

1.  双击计算机上的安装程序。

1.  按照屏幕上的提示完成 Anaconda 的安装。

1.  安装完成后，你可以访问 Anaconda Navigator，它将像其他应用程序一样正常显示。

## 在 Linux 上安装 Anaconda

1.  访问 Anaconda 下载页面，获取安装 shell 脚本，网址为[`www.anaconda.com/distribution/#download-section`](https://www.anaconda.com/distribution/#download-section)。

1.  要直接将 shell 脚本下载到你的 Linux 实例中，你可以使用`curl`或`wget`下载库。下面的示例展示了如何使用`curl`从 Anaconda 下载页面找到的 URL 下载文件：

    ```py
    curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    ```

1.  下载完 shell 脚本后，你可以使用以下命令运行它：

    ```py
    bash Anaconda3-2019.03-Linux-x86_64.sh
    ```

    运行上述命令后，你将进入一个非常用户友好的安装过程。系统会提示你选择安装 Anaconda 的位置以及如何配置 Anaconda。在这种情况下，你只需保持所有默认设置。

## 在 macOS X 上安装 Anaconda

1.  在 macOS 上安装 Anaconda 非常用户友好。访问下载页面获取安装可执行文件，网址为[`www.anaconda.com/distribution/#download-section`](https://www.anaconda.com/distribution/#download-section)。

1.  确保选择了 macOS，并双击`Download`按钮以下载 Anaconda 安装程序。

1.  按照屏幕上的提示完成 Anaconda 安装。

1.  安装完成后，你可以访问 Anaconda Navigator，它将像其他应用程序一样正常显示。

## 设置虚拟环境

1.  安装完 Anaconda 后，你必须创建环境来安装你希望使用的包。Anaconda 环境的一个优点是，你可以为你正在进行的具体项目构建独立的环境。要创建新的环境，使用以下命令：

    ```py
    conda create --name my_packt_env python=3.7
    ```

    在这里，我们将环境命名为`my_packt_env`并指定 Python 版本为 3.7。这样，你可以在环境中安装多个版本的 Python，这些版本将是虚拟隔离的。

1.  创建环境后，你可以使用名称恰当的`activate`命令激活它：

    ```py
    conda activate my_packt_env
    ```

    就这样。你现在已经进入了自己的定制化环境，可以根据项目需要安装包。要退出环境，你只需使用`conda deactivate`命令。

## 安装库

`pip` 已预先安装在 Anaconda 中。一旦 Anaconda 安装到你的计算机上，你可以使用`pip`安装所有所需的库，例如，`pip install numpy`。或者，你可以使用`pip install –r requirements.txt`安装所有必需的库。你可以在[`packt.live/2CnpCEp`](https://packt.live/2CnpCEp)找到`requirements.txt`文件。

练习和活动将在 Jupyter Notebooks 中执行。Jupyter 是一个 Python 库，可以像其他 Python 库一样通过 `pip install jupyter` 安装，但幸运的是，它已经随 Anaconda 一起预安装了。要打开一个笔记本，只需在终端或命令提示符中运行命令 `jupyter notebook`。

在*第九章*，*热点分析*中，使用了 `mpl_toolkits` 中的 `basemap` 模块来生成地图。这个库可能会很难安装。最简单的方法是安装 Anaconda，它包含了 `mpl_toolkits`。安装 Anaconda 后，可以通过 `conda install basemap` 来安装 `basemap`。如果你希望避免重复安装库，而是一次性安装所有库，你可以按照下一节的说明进行操作。

## 设置机器

如果你是按章节逐步安装依赖项，可能会遇到库的版本不同的情况。为了同步系统，我们提供了一个包含所用库版本的 `requirements.txt` 文件。使用此文件安装库后，你就不需要在整本书中再次安装任何其他库。假设你现在已经安装了 Anaconda，可以按照以下步骤进行操作：

1.  从 GitHub 下载 `requirements.txt` 文件。

1.  转到 `requirements.txt` 文件所在的文件夹，并打开命令提示符（Linux 为 Bash，Mac 为 Terminal）。

1.  在其上执行以下命令：

    ```py
    conda install --yes --file requirements.txt --channel conda-forge
    ```

    它应该会安装本书中所有编程活动所需的包。

## 访问代码文件

你可以在 [`packt.live/34kXeMw`](https://packt.live/34kXeMw) 找到本书的完整代码文件。你也可以通过使用 [`packt.live/2ZMUWW0`](https://packt.live/2ZMUWW0) 提供的交互式实验环境，直接在你的网页浏览器中运行许多活动和练习。

我们已尽力支持所有活动和练习的交互式版本，但我们仍然推荐进行本地安装，以便在无法获得此支持的情况下使用。

如果你在安装过程中遇到任何问题或有任何疑问，请通过电子邮件联系我们：`workshops@packt.com`。
