# 结论

恭喜！您已经迈出了成为机器学习实践者的重要一步。您不仅熟悉各种基础机器学习算法，而且知道如何将它们应用于监督学习和无监督学习问题。此外，您还接触到了一个新颖且令人兴奋的主题——OpenVINO 工具包。在前一章中，我们学习了如何安装 OpenVINO 并运行交互式的人脸检测和图像分类演示等。我相信您已经享受了学习这些主题的过程。

在我们分别之前，我想给您一些建议，指引您一些额外的资源，并给出一些如何进一步提升您的机器学习和数据科学技能的建议。在本章中，我们将学习如何处理机器学习问题并构建自己的估计器。我们将学习如何使用 C++编写基于 OpenCV 的分类器，以及使用 Python 编写基于 scikit-learn 的分类器。

在本章中，我们将涵盖以下主题：

+   处理机器学习问题

+   在 C++中编写自己的 OpenCV 分类器

+   在 Python 中编写自己的 scikit-learn 分类器

+   接下来该做什么

# 技术要求

您可以从以下链接中获取本章的代码：[`github.com/PacktPublishing/Machine-Learning-for-OpenCV-Second-Edition/tree/master/Chapter13`](https://github.com/PacktPublishing/Machine-Learning-for-OpenCV-Second-Edition/tree/master/Chapter13)。

以下是软件和硬件要求的总结：

+   您需要 OpenCV 版本 4.1.x（4.1.0 或 4.1.1 都完全可以）。

+   您需要 Python 3.6 版本（任何 3.x 版本的 Python 都适用）。

+   您需要 Anaconda Python 3 来安装 Python 和所需的模块。

+   您可以使用任何操作系统——macOS、Windows 以及基于 Linux 的操作系统，配合这本书使用。我们建议您的系统至少拥有 4 GB 的 RAM。

+   您不需要 GPU 来运行本书附带提供的代码。

# 处理机器学习问题

当您在野外遇到一个新的机器学习问题时，您可能会迫不及待地跳进去，用您最喜欢的算法来解决问题——可能是您最理解或最享受实现的一个。但事先知道哪种算法会在您特定的问题上表现最佳通常是不可能的。

而不是，你需要退一步，从更大的角度去看待问题。在你深入之前，你将想要定义你试图解决的真正问题。例如，你已经有了一个具体的目标，还是只是想要进行一些探索性分析，在数据中找到一些有趣的东西？通常，你可能会从一个一般的目标开始，比如检测垃圾邮件、制作电影推荐，或者自动在社交媒体平台上上传的图片中标记你的朋友。然而，正如我们在整本书中看到的，解决一个问题的方法往往有很多种。例如，我们使用逻辑回归、k-means 聚类和深度学习来识别手写数字。定义问题将帮助你提出正确的问题，并在过程中做出正确的选择。

作为一种经验法则，你可以使用以下五个步骤来处理现实世界中的机器学习问题：

1.  **对问题进行分类**：这是一个两步的过程：

    +   **按输入分类**：简单来说，如果你有标记过的数据，那么这是一个监督学习问题。如果你有未标记的数据，并希望找到结构，那么这是一个无监督学习问题。如果你想要通过与环境的交互来优化目标函数，那么这是一个强化学习问题。

    +   **按输出分类**：如果你的模型输出是一个数字，那么这是一个回归问题。如果你的模型输出是一个类别（或分类），那么这是一个分类问题。如果你的模型输出是一组输入组，那么这是一个聚类问题。

1.  **找到可用的算法**：现在你已经对问题进行了分类，你可以识别出可以使用我们拥有的工具实现的适用且实用的算法。微软创建了一个方便的算法速查表，显示了哪些算法可以用于哪些类别的问题。尽管速查表是为**微软 Azure**量身定制的，但你可能会发现它通常很有帮助。

机器学习算法速查表 PDF（由微软 Azure 提供）可以从[`aka.ms/MLCheatSheet`](http://aka.ms/MLCheatSheet)下载。

1.  **实现所有适用的算法**（**原型设计**）：对于任何给定的问题，通常有一小批候选算法可以完成这项工作。那么，你如何知道选择哪一个呢？通常，这个问题的答案并不直接，所以你必须求助于试错。原型设计最好分两步进行：

    1.  你应该追求快速且粗略地实现几个算法，并尽量减少特征工程。在这个阶段，你应该主要关注看到哪个算法在粗略尺度上表现更好。这一步有点像招聘：你正在寻找任何理由来缩短候选算法的列表。一旦你将列表缩减到几个候选算法，真正的原型设计就开始了。

    1.  理想情况下，你想要设置一个机器学习管道，使用一组精心选择的评估标准来比较每个算法在数据集上的性能（参见第十一章，*使用超参数调整选择正确的模型*）。在这个阶段，你应该只处理少数几个算法，因此你可以将注意力转向真正的魔法所在：特征工程。

1.  **特征工程**：选择正确的算法可能比选择正确的特征来表示数据更为重要。你可以在第四章，*表示数据和特征工程*中了解有关特征工程的所有内容。

1.  **优化超参数**：最后，你还需要优化算法的超参数。例如，可能包括 PCA 的主成分数量、k 最近邻算法中的参数`k`，或者神经网络中的层数和学习率。你可以参考第十一章，*使用超参数调整选择正确的模型*，以获取灵感。

# 构建自己的估计器

在这本书中，我们探讨了 OpenCV 提供的各种机器学习工具和算法。如果出于某种原因，OpenCV 没有提供我们想要的，我们总是可以退回到 scikit-learn。

然而，当处理更高级的问题时，你可能会发现自己想要执行一些非常具体的数据处理，而这些处理既不是 OpenCV 也不是 scikit-learn 提供的，或者你可能想要对现有算法进行轻微的调整。在这种情况下，你可能想要创建自己的估计器。

# 使用 C++编写自己的 OpenCV 分类器

由于 OpenCV 是那些底层不包含任何 Python 代码的 Python 库之一（我在开玩笑，但确实如此），你将不得不在 C++中实现自定义估计器。这可以通过以下四个步骤完成：

1.  实现一个包含主要源代码的 C++源文件。你需要包含两个头文件，一个包含所有 OpenCV 的核心功能(`opencv.hpp`)，另一个包含机器学习模块(`ml.hpp`)：

```py
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
```

然后，可以通过从`StatModel`类继承来创建一个估计器类：

```py
class MyClass : public cv::ml::StatModel
{
    public:
```

接下来，你定义类的`构造函数`和`析构函数`：

```py
MyClass()
{
    print("MyClass constructor\n");
}
~MyClass() {}
```

然后，你还必须定义一些方法。这些是你需要填写以使分类器真正做一些工作的内容：

```py
int getVarCount() const
{
    // returns the number of variables in training samples
    return 0;
}

bool empty() const
{
    return true;
}

bool isTrained() const
{
    // returns true if the model is trained
    return false;
}

bool isClassifier() const
{
    // returns true if the model is a classifier
    return true;
}
```

主要工作是在`train`方法中完成的，该方法有两种形式（接受`cv::ml::TrainData`或`cv::InputArray`作为输入）：

```py
bool train(const cv::Ptr<cv::ml::TrainData>& trainData,
          int flags=0) const
{
    // trains the model
    return false;
}

bool train(cv::InputArray samples, int layout, 
          cv::InputArray responses)
{
    // trains the model
    return false;
}
```

你还需要提供一个`predict`方法和一个`scoring`函数：

```py
        float predict(cv::InputArray samples,
                     cv::OutputArray results=cv::noArray(),
                     int flags=0) const
        {
            // predicts responses for the provided samples
            return 0.0f;
        }

        float calcError(const cv::Ptr<cv::ml::TrainData>& data,
                       bool test, cv::OutputArray resp)
        {
            // calculates the error on the training or test dataset
            return 0.0f;
        }
   };
```

最后一件要做的事情是包含一个`main`函数，该函数实例化类：

```py
   int main()
   {
       MyClass myclass;
       return 0;
   }
```

1.  编写一个名为`CMakeLists.txt`的 CMake 文件：

```py
cmake_minimum_required(VERSION 2.8)
project(MyClass)
find_package(OpenCV REQUIRED)
add_executable(MyClass MyClass.cpp)
target_link_libraries(MyClass ${OpenCV_LIBS})
```

1.  通过在命令行中输入以下命令来编译文件：

```py
$ cmake
$ make
```

1.  运行由最后一个命令生成的可执行文件`MyClass`方法，它应该导致以下输出：

```py
$ ./MyClass
MyClass constructor
```

# 在 Python 中编写基于 scikit-learn 的分类器

或者，你可以使用 scikit-learn 库编写自己的分类器。

你可以通过导入`BaseEstimator`和`ClassifierMixin`来实现这一点。后者将提供一个相应的`score`方法，它适用于所有分类器：

1.  可选地，首先，你可以重写`score`方法以提供你自己的`score`指标方法：

```py
In [1]: import numpy as np...     from sklearn.base import BaseEstimator, ClassifierMixin
```

1.  然后，你可以定义一个继承自`BaseEstimator`和`ClassifierMixin`的类：

```py
In [2]: class MyClassifier(BaseEstimator, ClassifierMixin):...         """An example classifier"""
```

1.  你需要提供一个构造函数、`fit`和`predict`方法。构造函数定义了所有参数...

# 接下来该怎么做

本书的目标是让你了解机器学习的世界，并为你成为机器学习从业者做好准备。现在你已经了解了所有基本算法，你可能想要深入研究一些主题。

虽然理解我们在本书中实现的所有算法的所有细节不是必要的，但了解它们背后的某些理论可能会让你成为一个更好的数据科学家。

如果你正在寻找更高级的材料，那么你可能想要考虑以下一些经典作品：

+   Stephen Marsland，*机器学习：算法视角*，*第二版*，Chapman and Hall/Crc，ISBN 978-146658328-3，2014

+   Christopher M. Bishop，*模式识别与机器学习*。Springer，ISBN 978-038731073-2，2007

+   Trevor Hastie，Robert Tibshirani，和 Jerome Friedman，*统计学习元素：数据挖掘、推理和预测*。*第二版*，Springer，ISBN 978-038784857-0，2016

当谈到软件库时，我们已经学习了两个基本库——OpenCV 和 scikit-learn。通常，使用 Python 非常适合尝试和评估模型，但更大的网络服务和应用程序更常见地使用 Java 或 C++编写。

例如，C++包是**Vowpal Wabbit**（VW），它自带命令行界面。对于在集群上运行机器学习算法，人们通常使用基于**Spark**的**Scala**库`mllib`。如果你不坚持使用 Python，你也可以考虑使用 R，这是数据科学家常用的另一种语言。R 是一种专门为统计分析设计的语言，以其可视化能力和许多（通常是高度专业化的）统计建模包而闻名。

无论你接下来选择哪种软件，我认为最重要的建议是持续练习你的技能。但这一点你已经知道了。有许多优秀的数据集正等着你去分析：

+   在整本书中，我们充分利用了 scikit-learn 内置的示例数据集。此外，scikit-learn 提供了一种从外部服务加载数据集的方法，例如[mldata.org](http://mldata.org/)。有关更多信息，请参阅[`scikit-learn.org/stable/datasets/index.html`](http://scikit-learn.org/stable/datasets/index.html)。

+   Kaggle 是一家在其网站上托管各种数据集和竞赛的公司，[`www.kaggle.com`](http://www.kaggle.com)。竞赛通常由各种公司、非营利组织和大学举办，获胜者可以赢得一些相当可观的现金奖励。竞赛的缺点是它们已经提供了一种特定的指标来优化，并且通常是一个固定、预处理的数据集。

+   OpenML 平台([`www.openml.org`](http://www.mldata.org))托管了超过 20,000 个数据集，与超过 50,000 个相关的机器学习任务相关联。

+   另一个流行的选择是 UC Irvine 机器学习仓库([`archive.ics.uci.edu/ml/index.php`](http://archive.ics.uci.edu/ml/index.php))，通过可搜索的界面托管了 370 多个流行且维护良好的数据集。

最后，如果你在寻找更多的 Python 示例代码，现在有许多优秀的书籍都附带了自己的 GitHub 仓库：

+   杰克·范德普拉斯，*Python 数据科学手册：与数据工作的必备工具*。O'Reilly，ISBN 978-149191205-8，2016，[`github.com/jakevdp/PythonDataScienceHandbook`](https://github.com/jakevdp/PythonDataScienceHandbook)

+   安德烈亚斯·穆勒和莎拉·吉多，*使用 Python 进行机器学习入门：数据科学家指南*。O'Reilly，ISBN 978-144936941-5，2016，[`github.com/amueller/introduction_to_ml_with_python`](https://github.com/amueller/introduction_to_ml_with_python)

+   塞巴斯蒂安·拉斯奇卡，*Python 机器学习*。Packt，ISBN 978-178355513-0，2015，[`github.com/rasbt/python-machine-learning-book`](https://github.com/rasbt/python-machine-learning-book)

# 摘要

在这一章中，我们学习了如何处理机器学习问题，并构建了自己的估计器。我们学习了如何用 C++编写基于 OpenCV 的分类器，以及用 Python 编写基于 scikit-learn 的分类器。

在这本书中，我们涵盖了大量的理论和实践。我们讨论了各种基本的机器学习算法，无论是监督学习还是无监督学习，还介绍了最佳实践以及避免常见陷阱的方法，并且触及了数据分析、机器学习和可视化的各种命令和包。

如果你已经走到这一步，你已经朝着机器学习精通迈出了重要的一步。从现在开始，我坚信你将能够独立完成得很好。

剩下的只有告别了！...
