# 前言

# 关于本书

机器学习算法几乎是所有现代应用程序的核心部分。为了加快学习过程并提高准确性，你需要一个足够灵活和强大的工具，帮助你快速、轻松地构建机器学习算法。通过*《机器学习工作坊（第二版）》*，你将掌握 scikit-learn 库，并成为开发巧妙机器学习算法的高手。

*《机器学习工作坊（第二版）》*首先通过分析批发客户的真实数据集，展示了无监督和监督学习算法的工作原理。在掌握基础知识后，你将使用 scikit-learn 开发一个人工神经网络，并通过调整超参数来提高其性能。在工作坊的后期，你将研究一家银行的营销活动数据集，并构建可以列出可能订阅定期存款的客户的机器学习模型。你还将学习如何比较这些模型并选择最优模型。

到了*《机器学习工作坊（第二版）》*的结尾，你不仅会学到监督学习和无监督学习模型之间的区别及其在现实世界中的应用，还将掌握开始编程自己机器学习算法所需的技能。

## 读者群体

*《机器学习工作坊（第二版）》*是机器学习初学者的理想之选。你需要有 Python 编程经验，但无需具备 scikit-learn 和机器学习的先前知识。

## 关于章节

*第一章*，*Scikit-Learn 简介*，介绍了本书的两个主要主题：机器学习和 scikit-learn。它解释了输入数据预处理的关键步骤，如何分离特征与目标，如何处理杂乱数据，以及如何重新调整数据值的尺度。

*第二章*，*无监督学习——真实生活应用*，通过讲解三种最常见的聚类算法，解释了机器学习中聚类的概念。

*第三章*，*监督学习——关键步骤*，描述了通过监督学习算法可以解决的不同任务：分类和回归。

*第四章*，*监督学习算法：预测年收入*，教授解决监督学习数据问题的不同概念和步骤。

*第五章*，*人工神经网络：预测年收入*，展示了如何使用神经网络解决监督学习分类问题，并通过执行误差分析来分析结果。

*第六章*，*构建你自己的程序*，解释了开发全面机器学习解决方案所需的所有步骤。

## 约定

文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名如下所示：

使用`seaborn`库加载`titanic`数据集。

您在屏幕上看到的单词（例如，在菜单或对话框中）以相同的格式显示。

代码块设置如下：

```py
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head(10)
```

新术语和重要单词以这种方式显示：

“缺失信息或包含异常值或噪声的数据被认为是**杂乱数据**。”

## 代码展示

跨多行的代码使用反斜杠（`\`）分割。当代码执行时，Python 会忽略反斜杠，并将下一行代码视为当前行的直接延续。

例如：

```py
history = model.fit(X, y, epochs=100, batch_size=5, verbose=1, \
                    validation_split=0.2, shuffle=False)
```

注释被添加到代码中，用于帮助解释特定的逻辑部分。单行注释使用`#`符号表示，如下所示：

```py
# Print the sizes of the dataset
print("Number of Examples in the Dataset = ", X.shape[0])
print("Number of Features for each example = ", X.shape[1])
```

多行注释用三引号括起来，如下所示：

```py
"""
Define a seed for the random number generator to ensure the 
result will be reproducible
"""
seed = 1
np.random.seed(seed)
random.set_seed(seed)
```

## 设置您的开发环境

在我们详细探讨这本书之前，我们需要安装一些特定的软件和工具。在接下来的章节中，我们将看到如何操作。

## 在 Windows 和 MacOS 上安装 Python

按照以下步骤安装 Python 3.7 在 Windows 和 macOS 上：

1.  访问[`www.python.org/downloads/release/python-376/`](https://www.python.org/downloads/release/python-376/)下载 Python 3.7。

1.  在页面底部，找到标题为`Files`的表格：

    对于 Windows，点击`Windows x86-64 executable installer`以安装 64 位版本，或点击`Windows x86 executable installer`以安装 32 位版本。

    对于 macOS，点击`macOS 64-bit/32-bit installer`以安装适用于 macOS 10.6 及更高版本的版本，或点击`macOS 64-bit installer`以安装适用于 macOS 10.9 及更高版本的版本。

1.  运行您已下载的安装程序。

## 在 Linux 上安装 Python

1.  打开终端并输入以下命令：

    ```py
    sudo apt-get install python3.7
    ```

## 安装 pip

`pip`默认会随 Python 3.7 的安装一同安装。然而，也有可能它没有被安装。要检查是否已安装，请在终端或命令提示符中执行以下命令：

```py
pip --version
```

由于您计算机上以前版本的`pip`已经使用了`pip`命令，您可能需要使用`pip3`命令。

如果`pip`命令（或`pip3`）未被您的机器识别，请按照以下步骤安装它：

1.  要安装`pip`，访问[`pip.pypa.io/en/stable/installing/`](https://pip.pypa.io/en/stable/installing/)并下载`get-pip.py`文件。

1.  然后，在终端或命令提示符中，使用以下命令安装它：

    ```py
    python get-pip.py
    ```

由于您机器上已安装的旧版本 Python 可能仍在使用`python`命令，您可能需要使用`python3 get-pip.py`命令。

## 安装库

`pip`随 Anaconda 预安装。一旦在机器上安装了 Anaconda，所有需要的库可以通过`pip`安装，例如，`pip install numpy`。另外，您也可以使用`pip install –r requirements.txt`安装所有需要的库。您可以在[`packt.live/2Ar1i3v`](https://packt.live/2Ar1i3v)找到`requirements.txt`文件。

练习和活动将在 Jupyter Notebook 中执行。Jupyter 是一个 Python 库，可以像其他 Python 库一样安装——即使用 `pip install jupyter`，但幸运的是，它已经随 Anaconda 预装。要打开一个 notebook，只需在终端或命令提示符中运行命令 `jupyter notebook`。

## 打开 Jupyter Notebook

1.  打开终端/命令提示符。

1.  在终端/命令提示符中，进入你克隆的书籍仓库所在的目录。

1.  通过输入以下命令来打开 Jupyter Notebook：

    ```py
    jupyter notebook
    ```

1.  执行前面的命令后，你将能够通过机器的默认浏览器使用 Jupyter Notebook。

## 访问代码文件

你可以在[`packt.live/2wkiC8d`](https://packt.live/2wkiC8d)找到本书的完整代码文件。你还可以通过使用[`packt.live/3cYbopv`](https://packt.live/3cYbopv)上的交互式实验环境直接在网页浏览器中运行许多活动和练习。

我们已尽力支持所有活动和练习的交互式版本，但我们也建议进行本地安装，以应对在没有此支持的情况下的使用需求。

本书中使用的高质量彩色图片可以在[`packt.live/3exaFfJ`](https://packt.live/3exaFfJ)找到。

如果你在安装过程中遇到任何问题或有任何疑问，请通过电子邮件联系我们：workshops@packt.com。
