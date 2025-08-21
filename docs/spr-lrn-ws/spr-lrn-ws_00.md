前言

#### 第一章：本书概览

想了解机器学习技术和数据分析如何在全球范围内引领企业发展吗？从生物信息学分析到气候变化预测，机器学习在我们的社会中扮演着越来越重要的角色。

尽管现实世界中的应用可能看起来复杂，但本书通过逐步互动的方式简化了初学者的监督学习。通过使用实时数据集，您将学习如何使用 Python 进行监督学习，以构建高效的预测模型。

从监督学习的基础知识开始，您将很快理解如何自动化手动任务，并通过 Jupyter 和像 pandas 这样的 Python 库评估数据。接下来，您将使用数据探索和可视化技术开发强大的监督学习模型，然后了解如何区分变量，并使用散点图、热图和箱线图表示它们之间的关系。在使用回归和分类模型处理实时数据集以预测未来结果后，您将掌握高级集成技术，如提升方法和随机森林。最后，您将了解监督学习中模型评估的重要性，并学习评估回归和分类任务的度量标准。

在本书结束时，您将掌握自己进行实际监督学习 Python 项目所需的技能。

#### 读者对象

如果您是初学者或刚刚入门的数据科学家，正在学习如何实现机器学习算法来构建预测模型，那么本书适合您。为了加速学习过程，建议具备扎实的 Python 编程基础，因为您将编辑类或函数，而不是从零开始创建。

#### 章节概览

第一章，基础知识，将介绍监督学习、Jupyter 笔记本以及一些最常见的 pandas 数据方法。

第二章，探索性数据分析与可视化，教授您如何对新数据集进行探索和分析。

第三章，线性回归，教授您如何解决回归问题和进行分析，介绍线性回归、多个线性回归和梯度下降法。

第四章，自回归，教授您如何实现自回归方法来预测依赖于过去值的未来值。

第五章，分类技术，介绍分类问题，包括线性回归和逻辑回归、k 近邻算法和决策树的分类方法。

第六章，集成建模，教授您如何检视不同的集成建模方法，包括它们的优点和局限性。

第七章，模型评估，展示了如何通过使用超参数和模型评估指标来提高模型的性能。

#### 约定

文本中的代码字、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账户名以如下方式显示：“使用 pandas 的 read_csv 函数加载包含 synth_temp.csv 数据集的 CSV 文件，然后显示前五行数据。”

屏幕上显示的单词，例如在菜单或对话框中，也会以这种形式出现在文本中：“通过点击 Jupyter notebook 首页的 titanic.csv 文件来打开它。”

一段代码的设置如下：

print(data[pd.isnull(data.damage_millions_dollars)].shape[0])

print(data[pd.isnull(data.damage_millions_dollars) &

(data.damage_description != 'NA')].shape[0])

新术语和重要单词以如下方式显示：“监督式学习意味着数据的标签在训练过程中已经提供，从而让模型能够基于这些标签进行学习。”

#### 代码展示

跨越多行的代码使用反斜杠（ \ ）进行拆分。当代码执行时，Python 会忽略反斜杠，将下一行的代码视为当前行的延续。

例如：

history = model.fit(X, y, epochs=100, batch_size=5, verbose=1, \

validation_split=0.2, shuffle=False)

注释被添加到代码中，以帮助解释特定的逻辑。单行注释使用#符号，如下所示：

# 打印数据集的大小

print("数据集中的示例数量 = ", X.shape[0])

print("每个示例的特征数 = ", X.shape[1])

多行注释用三个引号括起来，如下所示：

"""

为随机数生成器定义一个种子，以确保结果可复现

结果将是可重复的

"""

seed = 1

np.random.seed(seed)

random.set_seed(seed)

#### 设置你的环境

在详细探讨本书内容之前，我们需要设置一些特定的软件和工具。在接下来的部分，我们将看到如何操作。

#### 安装与设置

本书中的所有代码都是在 Jupyter Notebooks 和 Python 3.7 上执行的。安装 Anaconda 后，Jupyter Notebooks 和 Python 3.7 可供使用。以下部分列出了在 Windows、macOS 和 Linux 系统上安装 Anaconda 的说明。

#### 在 Windows 上安装 Anaconda

以下是完成安装所需遵循的步骤：

访问 https://www.anaconda.com/products/individual 并点击下载按钮。

在 Anaconda 安装程序/Windows 部分，选择 Python 3.7 版本的安装程序。

确保安装与你的计算机架构（32 位或 64 位）相符的版本。你可以在操作系统的“系统属性”窗口中找到此信息。

下载完成后，双击文件，按照屏幕上的指示完成安装。

这些安装将在你系统的 'C' 盘执行。不过，你可以选择更改安装目标。

#### 在 macOS 上安装 Anaconda

访问 [`www.anaconda.com/products/individual`](https://www.anaconda.com/products/individual) 并点击下载按钮。

在 Anaconda 安装程序/MacOS 部分，选择 (Python 3.7) 64 位图形安装程序。

下载完安装程序后，双击文件，并按照屏幕上的指示完成安装。

#### 在 Linux 上安装 Anaconda

访问 [`www.anaconda.com/products/individual`](https://www.anaconda.com/products/individual) 并点击下载按钮。

在 Anaconda 安装程序/Linux 部分，选择 (Python 3.7) 64 位 (x86) 安装程序。

下载完安装程序后，在终端运行以下命令：bash ~/Downloads/Anaconda-2020.02-Linux-x86_64.sh

按照终端中出现的指示完成安装。

你可以通过访问此网站了解有关各种系统安装的更多详情：[`docs.anaconda.com/anaconda/install/`](https://docs.anaconda.com/anaconda/install/)。

#### 安装库

pip 在 Anaconda 中预装。安装完 Anaconda 后，可以使用 pip 安装所有必需的库，例如：pip install numpy。或者，你也可以使用 pip install –r requirements.txt 来安装所有必需的库。你可以在 [`packt.live/3hSJgYy`](https://packt.live/3hSJgYy) 找到 requirements.txt 文件。

练习和活动将在 Jupyter Notebooks 中执行。Jupyter 是一个 Python 库，可以像其他 Python 库一样通过 pip install jupyter 安装，但幸运的是，它在 Anaconda 中已预装。要打开笔记本，只需在终端或命令提示符中运行命令 jupyter notebook。

#### 访问代码文件

你可以在 [`packt.live/2TlcKDf`](https://packt.live/2TlcKDf) 找到本书的完整代码文件。你也可以通过使用互动实验环境 [`packt.live/37QVpsD`](https://packt.live/37QVpsD) 直接在浏览器中运行许多活动和练习。

我们已经尽力支持所有活动和练习的互动版本，但我们仍然推荐进行本地安装，以防这些互动支持不可用。

如果你在安装过程中遇到任何问题或有任何疑问，请通过邮件联系我们：workshops@packt.com。
