# 第二章

量子计算中的工具

*提供工具，我们就能完成任务。*

— 温斯顿·丘吉尔

我们都非常期待在我们的笔记本电脑中拥有一个“Q1 Pro”量子芯片，但——遗憾的是——这项技术尚未成熟。尽管如此，我们确实有一些实际的量子计算机，尽管它们有局限性，但能够执行量子算法。此外，我们那些老式的经典计算机实际上在模拟理想量子计算机方面做得相当不错，至少对于少量量子位来说是这样。

在本章中，我们将探讨允许我们使用量子电路模型实现量子算法并在模拟器或真实量子硬件上运行的工具。

我们将首先介绍一些最广泛使用的量子软件框架和平台。然后，我们将了解如何使用本书中将更广泛使用的两个软件框架：Qiskit 和 PennyLane。

本章我们将涵盖以下主题：

+   量子计算工具：非详尽概述

+   使用 Qiskit

+   使用 PennyLane

阅读本章后，您将对量子计算可用的软件工具和平台有一个广泛的了解。此外，您将知道如何使用 Qiskit 和 PennyLane 在模拟器和真实量子硬件上实现和执行量子算法。

# 2.1 量子计算工具：非详尽概述

在这本书中，我们将主要使用两个量子框架：**Qiskit**和**PennyLane**。这些框架功能强大，非常广泛使用，并且拥有强大的用户社区支持，但它们绝不是唯一有趣的选择。目前有大量的量子计算软件框架，以至于有时会让人感到不知所措！

## 2.1.1 框架和平台的非详尽调查

在本节中，我们将简要介绍一些最受欢迎的框架。这些框架大多数都是免费的，无论是作为*免费啤酒*还是作为*自由* *言论*。

+   **Quirk**：我们可以从一个简单但强大的量子电路模拟器开始：Quirk ([`algassert.com/quirk`](https://algassert.com/quirk))。与其他所有我们将讨论的框架不同，这个框架不使用代码，而是使用一个作为网络应用程序运行的图形用户界面。这使得它非常适合运行算法演示或快速原型设计。

    file:///C:/Users/ketank/AppData/Local/Temp/sigil-PiDscY/file290.png![PIC](img/file285.png)

    使用 Quirk，您可以通过**拖放界面**构建量子电路。它包括最常见的量子门，以及一些用于算法演示的定制门。您可以通过查看*图* *2.1.1* 来了解 Quirk 的实际应用。

+   **Q# 和微软的 QDK**：大多数量子软件框架都依赖于一个 **宿主** 经典编程语言（通常是 **Python**）。**Q#**（读作 “*Q* *sharp*”）是这一规则的例外：它是微软开发的专门用于量子计算的语言。这种语言用于微软的 **量子开发工具包**（**QDK**）([`azure.microsoft.com/en-us/resources/development-kit/quantum-computing/`](https://azure.microsoft.com/en-us/resources/development-kit/quantum-computing/))，其中包含几个用于运行量子算法和评估其性能的模拟器。使用 QDK，你还可以使用 **Azure Quantum** 将你的量子算法发送到真实的量子计算机。

    此外，微软还提供了使 Q# 能够交互式运行并与其他编程语言（如 Python）互操作性的解决方案。此外，允许你在真实硬件上运行量子算法的 Azure Quantum 服务也与其他量子软件框架兼容。

+   **QuEST**。**量子精确模拟工具包**（**QuEST**）([`quest.qtechtheory.org/`](https://quest.qtechtheory.org/)) 是一个用 **C++** 编写的模拟框架，注重性能。使用它，你可以编写一个量子算法的单个实现，该实现可以编译成二进制代码，生成一个模拟算法的程序。使用这个框架，你可以更接近底层硬件，并确保所有可用的硬件资源都得到最佳利用。这使得 QuEST 成为大量量子比特硬件密集型模拟的一个非常有吸引力的选择。换句话说，如果你曾经想测试你的计算机能够处理多少量子比特，QuEST 可能是可行的选择（不过，你可能需要准备一个灭火器，以防万一事情变得很热！）。

    QuEST 的源代码可在网上免费获取：([`github.com/quest-kit/QuEST`](https://github.com/quest-kit/QuEST))。它主要是 C 和 C++ 代码，可以在任何设备上使用。

+   **Cirq** ([`quantumai.google/cirq`](https://quantumai.google/cirq)) 是由 **Google** 开发的量子软件框架，它使用 Python 作为宿主语言。它可以用来设计量子算法，并在经典设备上模拟或在真实量子硬件上发送它们。此外，Cirq 集成在 [TensorFlow Quantum](https://www.tensorflow.org/quantum) ([`www.tensorflow.org/quantum`](https://www.tensorflow.org/quantum)) 中，这是 Google 的量子机器学习框架。

+   **Qiskit** ([`qiskit.org/`](https://qiskit.org/)) 是 **IBM** 的量子框架。它依赖于 Python 作为宿主语言，并提供了一系列模拟器，同时允许将算法提交到 IBM 的真实量子硬件。除此之外，Qiskit 还提供了一整套量子电路和算法库，其中许多我们将在本书中使用！

    尤其是在机器学习方面，Qiskit 包括一些有趣的模型，以及训练和执行它们的工具；它还提供了用于训练量子机器学习模型的**PyTorch**接口（我们将在下一节详细探讨 Qiskit 及其所有秘密）。

+   **PennyLane** ([`pennylane.ai/`](https://pennylane.ai/)) 是一个专门为量子机器学习构建的量子框架，但它也可以完美地用作通用量子计算框架。它是量子编程场景中的新来者，由**Xanadu**开发。像 Qiskit 一样，它使用 Python 作为宿主语言。在 PennyLane 中编写的任何量子算法都可以发送到真实的量子计算机上并在广泛的模拟器中执行。

    PennyLane 是现有框架中最好的框架之一，当涉及到互操作性时。多亏了一大批**插件**，你可以将 PennyLane 电路导出到其他框架并在那里执行它们——利用这些其他框架可能具有的一些功能。

    在机器学习方面，PennyLane 提供了一些内置工具，但它也与经典机器学习框架如**scikit-learn**、**Keras**、**TensorFlow**和**PyTorch**高度互操作。我们将在本章后面详细讨论这个框架。

+   **Ocean** ([`docs.ocean.dwavesys.com/en/stable/`](https://docs.ocean.dwavesys.com/en/stable/)) 是加拿大公司**D-Wave**开发的 Python 库。与其他我们之前提到的软件包不同，Ocean 的目标不是量子电路的实现和执行。相反，这个库允许你定义不同类型的组合优化问题的实例，并用经典算法以及 D-Wave 的**量子退火器**（专门用于解决优化问题的特殊量子计算机）来解决这些问题。

    从**第三章***3*，**“QUBO：二次无约束二进制优化”**开始，我们将介绍理解如何使用 Ocean 定义问题的概念。在第四章***4*，**“量子绝热计算和量子退火”**中，我们将学习如何全面使用 Ocean，无论是用经典算法解决这些优化问题，还是在实际的量子计算机上用量子算法！**

**了解更多...

这些框架的源代码（以及如果有，二进制文件）可以从它们的官方网站下载。有关 Qiskit、PennyLane 和 Ocean 的详细安装说明，请参阅**附录*****D*，**“安装工具”**。

**Amazon Braket**：亚马逊网络服务提供 Amazon Braket ([`aws.amazon.com/braket/`](https://aws.amazon.com/braket/))，这是一种付费云服务，使得使用各种真实量子计算机的实现成为可能。为了在这些计算机上执行代码，他们提供了自己的*设备无关的 SDK*，但他们也完全支持 PennyLane 和 Qiskit，甚至还有插件可以与 Ocean ([`amazon-braket-ocean-plugin-python.readthedocs.io/`](https://amazon-braket-ocean-plugin-python.readthedocs.io/))）一起工作。

## 2.1.2 Qiskit、PennyLane 和 Ocean

正如我们刚才看到的，在量子计算软件框架方面有丰富的选择，这可能会让你想知道为什么我们要坚持使用 Qiskit、PennyLane 和 Ocean。当然，这并不是一个随机的选择；我们有充分的理由！

关于 Qiskit，它非常庞大：它包含的内置算法和功能水平是无可匹敌的。不仅如此，它还拥有一个非常强大的用户社区，并且得到了大多数量子硬件提供商的支持。换句话说，Qiskit 可以很容易地被认为是量子计算的*通用语言*。

相反，PennyLane 不像 Qiskit 那样被广泛使用（至少目前是这样），但我们相信它是量子计算领域最有前途的新来者之一。

尤其是在量子机器学习方面，很难说有什么比 PennyLane 更好的东西。一方面，PennyLane 运行非常顺畅，并且有精美的文档，另一方面，它与其它量子以及**机器学习**（**ML**）框架的互操作性无人能敌。

正因如此，我们相信 Qiskit 和 PennyLane 比例如**Q#**或**Cirq**（当然，它们本身也是优秀的框架）是更好的选择。至于 QuEST，确实，Qiskit 和 PennyLane 提供的模拟器的性能可能不如 QuEST 提供的性能好。但我们也应该考虑到，QuEST 的用户友好性远不如 Qiskit 或 PennyLane，并且缺少它们许多功能；例如，QuEST 没有内置的工具或接口用于训练量子机器学习模型。无论如何，我们应该指出，虽然在使用 Qiskit 和 PennyLane 捆绑的模拟器上运行电路可能不如在 QuEST 上运行效率高，但对我们来说，我们可以从中获得的性能已经足够好了！尽管如此，如果你仍然渴望获得 QuEST 可以提供的性能提升，你应该知道有一个社区插件允许 PennyLane 与 QuEST 模拟器一起工作。

最后，我们选择了 Ocean，因为它在这一点上完全独特，可能是唯一一个允许你与**量子****退火器**一起工作的软件包，既可以定义问题，也可以在实际量子硬件上运行它们。它也非常容易学习……至少在你理解了如何在**伊辛**和 QUBO 模型中定义组合优化问题之后。但别担心；我们将在*第**3*章*QUBO:* *二次无约束二进制优化*中广泛研究这些框架，到*第**4*章*量子* *绝热计算和量子退火*时，我们将准备好编写我们第一个使用 Ocean 的程序。

**在此阶段，我们对量子计算当前工具的格局有了很好的全局理解。在接下来的章节中，我们将迈出使用它们的第一步，并且我们将从 Qiskit 开始。**

# 2.2 使用 Qiskit

在本节中，我们将学习如何使用 Qiskit 框架。我们首先将讨论 Qiskit 的一般结构，然后我们将研究如何在 Qiskit 中使用量子门和测量来实现量子电路。接着，我们将探讨如何使用 Qiskit 提供的模拟器和 IBM 提供的免费真实量子计算机来运行这些电路。本节至关重要，因为在这本书中我们将广泛使用 Qiskit。

重要提示

量子计算是一个快速发展的领域……以及它的软件框架！我们将使用 Qiskit 的**版本 0.39.2**。请记住，如果你使用的是不同版本，事情可能会有所变化。如果有疑问，你应该始终参考文档([`qiskit.org/documentation/`](https://qiskit.org/documentation/))。

## 2.2.1 Qiskit 框架概述

Qiskit 框架[102]由*图**2.1*中展示的组件组成。Qiskit 的基石是**Qiskit Terra**。这个包负责处理量子电路并提供构建它们所需的工具。它还包括一个基于 Python 的基本模拟器（**BasicAer**），并且它可以与**IBM 量子提供商**一起工作，在 IBM 的量子硬件上执行电路。

![图 2.1：Qiskit 框架的组件](img/file286.jpg)

**图 2.1**：Qiskit 框架的组件

Qiskit Aer 建立在 Qiskit Terra 之上，并提供了一套用**C++**编写的性能高效的量子模拟器，旨在更有效地使用硬件资源。

我们可以将 Qiskit Aer 和 Terra 视为 Qiskit 框架的核心；它们在安装 Qiskit 时被包含在内。然而，除了这些组件之外，还有一些其他组件：

+   **Qiskit Machine Learning**实现了一些适合**NISQ**设备的知名量子机器学习算法。我们将在本书的*第**III*，*天堂之配：量子机器学习*[*III*]中广泛使用这个包。此包还提供了一个可选的与**PyTorch**的接口，可用于量子机器学习模型的训练。

**Qiskit Optimization**实现了一些量子优化算法。我们将在本书的*第**II*，*时间即金钱：量子优化工具*[*II*]中使用它们。

    **本书重点介绍量子机器学习和量子优化，但量子计算在其他特定领域有更多令人兴奋的应用。两个很好的例子是自然科学，特别是量子物理和化学，以及金融。你可以在**Qiskit Nature**和**Qiskit Finance**包中找到一些与这些领域问题相关的算法。有趣的是，你应该知道，为量子力学系统进行更有效的计算的可能性是探索量子计算想法的最初动机之一。我们将在*第**7*，*变分量子本征值求解器*[*7*]中简要探讨一些这些应用。

    **Qiskit Experiments**提供了一系列用于与有噪声的量子计算机工作的工具，即当前受不同类型错误和外部噪声影响的量子设备，用于表征、基准测试和校准它们。

    +   最后，**Qiskit Metal**和**Qiskit Dynamics**是 Qiskit 最近添加的两个新功能。Qiskit Metal 可用于设计真实的量子设备，而 Qiskit Dynamics 提供了用于与量子系统模型一起工作的工具。***

**练习 2.1

按照附录**D*，*安装工具*中的说明，安装 Qiskit 包的**版本 0.39.2**。

*一旦安装了 Qiskit，你可以在 Python 运行中通过`import` `qiskit`来加载它。如果你想检查你正在运行的 Qiskit 版本，你可能想通过`qiskit``.``__version__`来查找它，但那样会给你 Qiskit Terra 包的版本，而不是 Qiskit 本身的版本！如果你想找到 Qiskit 框架的版本，你将不得不访问`qiskit``.``__qiskit_version__`。这将给你一个包含 Qiskit 所有组件版本（包括 Qiskit 本身）的字典。因此，Qiskit 的版本将是`qiskit``.``__qiskit_version__``[``’``qiskit``’``]`。

要了解更多信息…

Qiskit 更新得很频繁。为了跟上新功能，我们建议你访问[`qiskit.org/documentation/release_notes.html`](https://qiskit.org/documentation/release_notes.html)。

现在我们已经准备好了，是时候用我们的脚踩在 Terra 上构建一些电路了！

## 2.2.2 使用 Qiskit Terra 构建量子电路

为了开始，让我们首先按照以下方式导入 Qiskit：

```py
from qiskit import *

```

注意在前一个子节中，我们是如何使用`import qiskit`导入 Qiskit 来检查其版本号的。在本章的剩余部分，我们将假设 Qiskit 已经被导入为`from qiskit import *`。

我们现在将探索如何使用 Qiskit 实现量子算法（以量子电路的形式）。

### 初始化电路

在 Qiskit 中，电路被表示为`QuantumCircuit`类的对象。当我们初始化这样的对象时，我们可以根据我们希望电路有多少个量子位和比特来提供一些可选参数。例如，如果我们希望我们的电路有`n`个量子位，我们可以调用`QuantumCircuit(n)`。如果我们还希望它有`m`个经典比特来存储测量量子位的结果，我们可以运行`QuantumCircuit(n, m)`。一旦我们有一个量子电路对象，我们可以通过调用`draw`方法在终端中获取它的 ASCII 表示。例如，如果我们执行了`QuantumCircuit(2,2).draw()`，我们会得到以下结果：

```py

q_0: 

q_1: 

c_0: 

c_1:

```

当然，在这个表示中，我们只能看到我们创建的量子位和比特的名称，因为到目前为止，我们电路中只有这些。

这些 ASCII 表示是好的，但我们都可以同意，它们并不一定非常时尚。如果你想得到更花哨的东西，你可以将可选参数`’mpl’`（代表**matplotlib**）传递给`draw`。记住，如果你在终端上使用 Python 而不是在**Jupyter 笔记本**（老式笔记本获胜！）中，你可能还必须使用`draw(`mpl`, interactive=True)`。然而，如果您的环境不支持图形用户界面，这可能不起作用。

在 Qiskit 中，量子位和经典比特被分组在量子寄存器和经典寄存器中。默认情况下，当你创建一个电路`QuantumCircuit(n, m)`时，Qiskit 会将你的量子位分组在量子寄存器`q`中，你的比特分组在经典寄存器`c`中。然而，你可能想要不同的寄存器排列，或者你可能想要给它们不同的名称。为了做到这一点，你可以创建自己的寄存器，这些寄存器将是`QuantumRegister`和`ClassicalRegister`类的对象。在初始化这些寄存器时，你可以自由指定一些`size`和`name`参数。一旦你创建了一些量子寄存器和经典寄存器`reg_1`，…，`reg_n`，你可以在电路中通过调用`QuantumCircuit(reg_1, ..., reg_n)`将它们堆叠起来。这样，我们可以执行以下代码：

```py

qreg1 = QuantumRegister(size = 2, name = "qrg1") 

qreg2 = QuantumRegister(1, "qrg2") 

creg = ClassicalRegister(1, "oldschool") 

qc = QuantumCircuit(qreg1, creg, qreg2)

```

如果我们运行`qc.draw()`，我们会得到以下结果：

```py

   qrg1_0: 

   qrg1_1: 

     qrg2: 

oldschool:

```

### 量子门

现在我们有一个包含许多量子比特的电路——这是一个不错的入门方式！默认情况下，所有这些量子比特都将初始化为状态![\left| 0 \right\rangle](img/rangle")，但当然，如果我们想进行一些计算，我们最好能够将一些量子门放到桌面上。

你可以通过执行电路`qc`的方法来向电路`qc`添加量子门。例如，如果你想对电路`qc`的第一个量子比特应用![X](img/file9.png "X")门，你可以直接运行`qc``.``x``(0)`。

重要提示

正如 Python 中经常发生的那样，电路中的量子比特是 0 索引的！这意味着第一个量子比特将被标记为 0，第二个为 1，依此类推。

当我们在电路中具有不同的量子寄存器时，我们仍然可以通过它们的索引来引用量子比特。我们添加的第一个寄存器的量子比特将具有第一个索引，后续的索引将对应于第二个寄存器的量子比特，依此类推。在这一点上，经典寄存器和量子寄存器是完全独立的。

然而，如果我们有很多寄存器，这可能会有些不方便，但不用担心，Qiskit 已经为你准备好了。虽然通过索引引用门可能很方便，但我们也可以直接引用它们！假设我们有一个像之前那样的设置，电路`qc`带有量子寄存器`qreg1`和`qreg2`。运行`qc``.``x``(2)`会产生与执行`qc``.``x``(``qreg2``[0])`相同的效果。此外，如果我们调用`qc``.``x``(``qreg1``)`，那将等同于依次应用`qc``.``x``(0)`和`qc``.``x``(1)`。

以下是一些 Qiskit 方法，用于在量子比特`q0`上应用一些最常见的单量子比特门（我们在*章节* **1.3.3* 和 **1.3.4* 中研究过的那些）：

为了应用 Pauli 门之一，![X](img/file9.png "X")，![Y](img/file11.png "Y")，或![Z](img/file8.png "Z")，我们可以分别调用`x``(``q0``)`，`y``(``q0``)`，或`z``(``q0` `)`。

+   方法`h``(``q0` `)`可以用来应用 Hadamard 门。

+   我们可以使用`theta`参数化的旋转门![R_{X}](img/file118.png "R_{X}")，![R_{Y}](img/file119.png "R_{Y}")，或![R_{Z}](img/file120.png "R_{Z}")，分别通过`rx``(``theta``,` `q0``)`，`ry``(``theta``,` `q0``)`，或`rz``(``theta``,` `q0` `)`方法应用。

+   我们可以将由`theta`，`phi`和`lambd`参数化的通用单量子比特门![U(\theta,\varphi,\lambda)](img/lambda)")作为`u``(``theta``,` `phi``,` `lambd``,` `q0` `)`应用。

当然，我们也有多量子比特门的方法。最值得注意的是，一个受控![X](img/file9.png "X")、![Y](img/file11.png "Y")、![Z](img/file8.png "Z")或![H](img/file10.png "H")门，控制量子比特`q0`在目标`qt`上，可以通过`cx``(``q0``,` `qt``)`、`cy``(``q0``,` `qt``)`、`cz``(``q0``,` `qt``)`和`ch``(``q0``,` `qt``)`方法分别应用。在完全类似的情况下，一个由值`theta`参数化的受控旋转门![R_{X}](img/file118.png "R_{X}")、![R_{Y}](img/file119.png "R_{Y}")或![R_{Z}](img/file120.png "R_{Z}")可以通过`crx``(``theta``,` `q0``,` `qt``)`、`cry``(``theta``,` `q0``,` `qt``)`和`crz``(``theta``,` `q0``,` `qt``)`方法添加，其中，正如之前一样，`q0`代表控制量子比特，`qt`代表目标量子比特。

重要提示

记住，受控![X](img/file9.png "X")门是著名的**CNOT**。我们喜欢纠缠量子比特，所以我们肯定会大量使用那个`cx`方法！

![a](img/file288.jpg)

**(a)**

![b](img/file289.jpg)

**(b**)

**图 2.2**：我们可以在 Qiskit 中构建的示例量子电路。

现在可能是退后一步，看看我们所做的一切都变得生动起来的好时机。例如，让我们尝试构建图*2.2a*（图 2.2a）中所示的电路。利用我们所学的所有知识，我们可以在 Qiskit 中如下实现这个电路：

```py

import numpy as np 

qc = QuantumCircuit(2) # Initialise the circuit. 

# We can now apply the gates sequentially. 

qc.x(0) 

qc.rx(np.pi/4, 1) 

qc.cx(0, 1) 

qc.u(np.pi/3, 0, np.pi, 0)

```

现在如果我们运行`qc``.``draw``(``"``mpl``"``)`来验证我们的实现是否正确，我们将得到图*2.3*（图 2.3）中所示的输出。

![图 2.3：图 2.2a 中电路的 Qiskit 输出。](img/file290.png)

**图 2.3**：图*2.2a*（图 2.2a）中电路的 Qiskit 输出。

练习 2.2

按照图*2.2b*（图 2.2b）构建电路。绘制结果并使用输出验证你的电路实现是否正确。

### 测量

我们现在知道如何向电路添加量子门，所以我们还缺少一个成分：测量算子。实际上，这非常简单。如果你想在电路的任何位置进行测量（在计算基中），你可以通过调用`measure``(``qbits``,` `bits``)`方法来完成，其中`qbits`应该是一个包含你想要测量的所有量子比特的列表，而`bits`应该是一个包含你想要存储测量结果的经典比特的列表。当然，列表必须具有相同的长度。

如果你只想测量所有量子比特，而不想麻烦地创建适当大小的经典寄存器，你只需调用`measure_all`方法。这将根据量子比特的数量添加相应数量的比特到你的电路中，并对每个量子比特进行测量，并将结果发送到这些比特。如果你已经添加了经典比特来存储测量结果，你仍然可以使用它们与`measure_all`方法：你只需要将`add_bits`参数设置为`False`。

练习 2.3

实现你自己的 `measure_all` 方法。你可能需要使用 `QuantumCircuit` 类的 `add_register` 方法，该方法接受一些寄存器对象作为参数并将它们附加到电路中。

因此，现在我们可以构建自己的量子电路，但我们仍然需要找到一种方法来使用 Qiskit 运行它们。我们将在下一小节中这样做。让我们飞向 Aer！

## 2.2.3 使用 Qiskit Aer 模拟量子电路

正如我们在介绍 `Qiskit` 框架时提到的，`Terra` 包包含一个基于 Python 的模拟器，**BasicAer**。虽然这个模拟器对于大多数基本任务来说足够好，但它很大程度上被 `Aer` 包中包含的模拟器所超越，所以我们在这里只讨论这些。

如果我们想使用 Aer 模拟器，仅导入 Qiskit 是不够的。这次，我们还需要运行以下代码：

```py

from qiskit.providers.aer import AerSimulator

```

一旦我们完成了必要的导入，我们可以根据是否已配置我们的系统使用 GPU 来创建一个 Aer 模拟器对象，以下是一些方法：

```py

sim = AerSimulator() 

sim_GPU = AerSimulator(device = ’GPU’)

```

如果你有一个 GPU 并且正确配置了你的 Qiskit 安装（有关说明，请参阅 *附录* **D*，*安装工具*），使用 GPU 驱动的模拟器将产生对需求较高的模拟任务更好的结果。然而，你应该记住，对于资源消耗较少的模拟，使用 GPU 实际上可能会因为通信开销而导致性能更差。在本节的剩余部分，我们将使用不带 GPU 的模拟器。如果我们使用 `sim_GPU`，一切都将完全类似。

*要了解更多信息...

要使用 GPU 模拟电路，你需要 `qiskit-aer-gpu` 包。这个包是用 CUDA 编写的。因此，它只支持 NVIDIA GPU。

如我们所知，当我们测量量子态时，结果是概率性的。因此，我们通常会运行给定电路的多次执行或 **射击**，然后对结果进行一些统计分析。如果我们想模拟电路 `qc` 的 `nshots` 次射击的执行，我们必须运行 `job = execute(qc, sim, shots=nshots)`，并且我们可以通过调用 `result = job.result()` 来检索一个 *结果* 对象；顺便说一下，`shots` 的默认值是 `1024`。有了这个结果对象，我们可以使用 `result.get_counts()` 获取模拟的频率计数，这将给我们一个包含每个结果的绝对频率的字典。

让我们通过一个例子来尝试使这一点更清晰。我们将考虑一个非常简单的双量子比特电路，其中顶部的量子比特有一个哈达德门。然后我们将测量电路中的两个量子比特，并模拟 ![1024](img/file291.png "1024") 次射击：

```py

qc = QuantumCircuit(2, 2) 

qc.h(0) 

qc.measure(range(2), range(2)) 

job = execute(qc, sim, shots = 1024) 

result = job.result() 

counts = result.get_counts() 

print(counts)

```

要了解更多信息...

如果你正在运行一个名为 `job` 的对象，并且想要检查其状态，你可以从 `qiskit.providers.ibmq.job` 导入 `job_monitor` 并运行 `job_monitor(job)`。

重要提示

当获取 Qiskit 中测量的结果时，你需要记住最上面的量子比特成为最低有效位，依此类推。也就是说，如果你有两个量子比特，并且在测量时，上面的一个（![0](img/file12.png "0")量子比特）的值为![0](img/file12.png "0")，下面的一个（![1](img/file13.png "1")量子比特）的值为![1](img/file13.png "1")，结果将被解释为`10`，而不是`01`。

这与我们迄今为止所做的是相反的——也是世界上大多数人所认同的相反。因此，我们称之为状态![10](img/file161.png "10")，Qiskit 会称之为`01`。

对于大多数实际用途，我们可以简单地忽略这个问题，并假设当我们需要访问含有![n](img/file244.png "n")个量子比特的电路中的![q](img/file292.png "q")量子比特时，我们需要使用索引![n - q - 1](img/file293.png "n - q - 1")（记住我们从![0](img/file12.png "0")开始计数量子比特）。当我们实际使用 Qiskit 时，我们会隐式地这样做。

从理论上讲，我们知道测量之前的状态是 ![√(1/2)(|00> + |10>)](img/2)(|00> + |10>))"，因此我们预期(Qiskit)的结果 ![01](img/file159.png "01") 和 ![00](img/file157.png "00") 的频率分布应该是均匀的，我们不应该看到其他结果的出现。实际上，当我们运行代码时，我们得到了以下结果：

```py

{’01’: 519, ’00’: 505}

```

不言而喻，你不会得到相同的结果！但你肯定会得到一些具有相同风味的东西。

注意，在前面的测量中，我们将第一个量子比特（![0](img/file12.png "0")）中值为![1](img/file13.png "1")的状态标记为![10](img/file161.png "10")（这与我们一直使用的符号一致）。尽管如此，Qiskit 与其自身的符号保持一致，将其对应的结果标记为![01](img/file159.png "01")。

要了解更多...

如果你想在 Qiskit 中执行电路时获得可重复的结果，你需要使用`execute`函数的两个参数：`seed_transpiler`和`seed_simulator`。它们用于设置在电路转换过程中使用的伪随机数生成器的初始值——我们将在本节后面讨论这一点——以及从测量结果中进行采样。如果你使用一些固定的种子，你将始终得到相同的结果。这可以很有用，例如，用于调试目的。

所有这些数字都很好，但我们都知道，一张图片胜过千言万语。幸运的是，IBM 的同事们同意这一点，并且他们足够周到地将一些花哨的可视化工具直接捆绑到 Qiskit 中。例如，我们可以运行以下指令：

```py

from qiskit.visualization import * 

plot_histogram(counts)

```

我们会得到*图* *2.4* 中所示的图表。此函数接受可选参数 `filename`，如果提供，则将图保存为给定的字符串作为文件名。

![图 2.4：由 Qiskit 生成的直方图](img/file294.png)

**图 2.4**：由 Qiskit 生成的直方图

作为一个有趣的事实，你应该知道 Aer 模拟器可以使用不同的方法来模拟电路的执行。然而，除非我们要求其他方式，否则我们的电路将始终使用**状态向量**方法进行模拟，正如其名称所暗示的那样，它通过电路计算系统的精确量子状态（或状态向量）以生成结果。

现在，如果模拟器确实计算了系统的量子状态，为什么我们只满足于一些模拟样本，而不去获取实际的状态呢？当然，当我们与真实的量子计算机一起工作时，电路的状态是我们无法访问的（我们只能通过执行测量来获得结果），但是，嘿，模拟电路也应该有其自身的优势！

如果我们想在量子电路 `qc` 的任何一点访问状态向量，我们只需调用 `qc.save_statevector()`，就像我们添加另一个门一样。然后，一旦电路被模拟并且我们从执行作业中获得了结果，我们可以使用 `get_statevector` 方法来获取状态向量，就像我们使用 `get_counts` 一样。实际上，如果我们的电路也有测量，我们可以在同一时间做这两件事。例如，我们可以考虑这个例子：

```py

qc = QuantumCircuit(2, 2) 

qc.h(0) 

qc.save_statevector() 

qc.measure(0,0) 

qc.measure(1,1) 

result = execute(qc, sim, shots = 1024).result() 

sv = result.get_statevector() 

print(sv) 

counts = result.get_counts() 

print(counts)

```

注意，在这段代码中，我们使用两个单独的指令来测量电路中的两个量子比特，而不是仅仅调用 `qc.measure(range(2), range(2))`。当我们运行它时，我们得到以下输出：

```py

Statevector([0.70710678+0.j, 0.70710678+0.j, 0\.        +0.j, 
             0\.        +0.j], 
            dims=(2, 2)) 
{’00’: 486, ’01’: 538}

```

这正是我们预期的。我们只需要记住![\( \left. \frac{1}{\sqrt{2}} \approx 0.7071\ldots \right. \)](img/)")！在 Qiskit 给出的输出中，状态向量数组的第一个元素是基态![\( \left| {00} \right\rangle \)](img/)")的振幅，第二个元素是![\( \left| {10} \right\rangle \)](img/)")的振幅（记住 Qiskit 命名基态的约定，所以对于 Qiskit，这个状态的标签将是`01`），接下来的一个是![\( \left| {01} \right\rangle \)](img/)")和![\( \left| {11} \right\rangle \)](img/)")的振幅，依次类推。

要了解更多…

有可能保存多个状态向量以便稍后检索。为此，需要在 `save_statevector` 中传递可选参数 `label`，指定一个唯一标识电路中状态向量的标签。然后，可以从结果对象 `result` 中使用 `result.data()` 提取状态向量作为字典。

Aer 模拟器为我们提供的另一个可能性是计算表示电路已执行的所有变换的幺正矩阵。为了获取这个矩阵，我们可以使用 `save_unitary` 和 `get_unitary` 方法，这些方法将与 `save_statevector` 和 `get_statevector` 完全类似。尽管这听起来可能很神奇，但有一个小问题需要注意，那就是这些矩阵不能使用状态向量方法计算；相反，需要使用 **幺正** 方法，这种方法不支持测量，也不允许访问电路的状态向量。无论如何，这并不是什么大问题，因为只要模拟电路相应调整，就可以结合不同的模拟方法。

为了看到这个例子在实际中的应用，让我们运行以下示例：

```py

sim_u = AerSimulator(method = ’unitary’) 

qc = QuantumCircuit(1) 

qc.h(0) 

qc.save_unitary() 

result = execute(qc, sim_u).result() 

U = result.get_unitary(decimals = 4) 

print(U)

```

当我们执行此代码时，我们得到以下输出：

```py

Operator([[ 0.7071+0.j,  0.7071-0.j], 

          [ 0.7071+0.j, -0.7071+0.j]], 

         input_dims=(2,), output_dims=(2,))

```

正如它应该的那样，是哈达玛门的矩阵。

顺便说一下，注意我们是如何使用可选参数 `decimals` 来限制输出精度的。这也可以在 `get_statevector` 方法中使用。

我们现在能够使用 Qiskit 构建和模拟电路，但我们还缺少一些东西：如何在真实的量子硬件上实际运行它们。这正是下一个小节要讨论的内容。

## 2.2.4 让我们面对现实：使用 IBM Quantum

现在我们知道了如何使用 Qiskit Aer 提供的工具来执行量子电路的理想模拟，但我们知道真实的量子计算机，即使有局限性，确实存在并且可以访问，那么为什么不尝试一下呢（当然，这里是一个双关语）？

IBM 免费提供对其部分真实量子计算机的访问。为了获得访问权限，你只需注册一个免费的 IBM ID 账户。有了它，你可以登录到 IBM Quantum 网站 ([`quantum-computing.ibm.com/`](https://quantum-computing.ibm.com/)) 并获取你的 API 令牌（有关更多详细信息，请参阅 *附录* **D*，*安装工具*）。

*一旦你有了你的令牌，接下来你应该做的就是前往你的本地环境并执行指令 `IBMQ.save_account("TOKEN")`，其中当然应该用你的实际令牌替换 `TOKEN`。完成这些后，我们可以运行以下代码片段：*

```py

provider = IBMQ.load_account() 

print(provider.backends(simulator = False))

```

这将允许我们加载我们的账户详情并获取所有可用真实量子设备的列表。如果你有一个普通的免费账户，你可能会得到以下类似的输出：

```py

[<IBMQBackend(’ibmq_lima’) from IBMQ(hub=’ibm-q’, 
    group=’open’, project=’main’)>, 
<IBMQBackend(’ibmq_belem’) from IBMQ(hub=’ibm-q’, 
    group=’open’, project=’main’)>, 
<IBMQBackend(’ibmq_quito’) from IBMQ(hub=’ibm-q’, 
    group=’open’, project=’main’)>, 
<IBMQBackend(’ibmq_manila’) from IBMQ(hub=’ibm-q’, 
    group=’open’, project=’main’)>, 
<IBMQBackend(’ibm_nairobi’) from IBMQ(hub=’ibm-q’, 
    group=’open’, project=’main’)>, 
<IBMQBackend(’ibm_oslo’) from IBMQ(hub=’ibm-q’, 
    group=’open’, project=’main’)>]

```

如果我们使用参数 `simulator` `=` `True`，我们会得到所有可用的云模拟器的列表。它们的主要优势是其中一些能够运行比普通计算机能处理的更多量子比特的电路。

选择这些提供者的一种天真方式就是从列表中选择一个元素，例如，取 `dev` `=` `provider``.``backends``(``simulator` `=` `False``)[0]`。或者，如果你知道你想要使用的设备名称（例如 `ibmq_lima`），你可以简单地运行 `dev` `=` `provider``.``get_backend``(``’``ibmq_lima``’``)。一旦你选择了一个设备，即一个后端对象，你可以通过调用 `configuration` 方法（不带参数）来获取一些其配置细节。这将返回一个包含设备信息的对象。例如，为了知道提供者 `dev` 有多少个量子比特，我们只需访问 `dev``.``configuration``().``n_qubits`。

然而，与其随机选择一个设备或根据一个花哨的位置名称选择，我们首先可以尝试进行一些筛选。当调用 `get_backend` 时，我们可以传递一个可选的 `filters` 参数。这应该是一个单参数函数，它只为我们要选择的设备返回 `True`。例如，如果我们想要获取所有至少有 ![5](img/file296.png "5") 个量子比特的真实设备列表，我们可以使用以下代码：

```py

dev_list = provider.backends( 

    filters = lambda x: x.configuration().n_qubits >= 5, 

    simulator = False)

```

在所有这些设备中，可能明智的做法是只使用最不忙碌的那个。为此，我们可以简单地执行以下操作：

```py

from qiskit.providers.ibmq import * 

dev = least_busy(dev_list)

```

要了解更多...

`least_busy` 接受一个名为 `reservation_lookahead` 的可选参数。这是设备需要空闲无预约的时间（分钟数），才能被认为是忙碌程度最低的候选者。该参数的默认值是 ![60](img/file297.png "60")。所以如果 `least_busy` 没有返回一个合适的设备，你可以设置 `reservation_lookahead` `=` `None` 来考虑那些处于预约状态下的计算机。

现在，在选定的设备上运行一个电路，该设备具有一定数量的射击次数，将完全类似于在模拟器上运行它。实际上，我们可以在两者上运行并比较结果！

```py

from qiskit.providers.ibmq.job import job_monitor 

# Let us set up a simple circuit. 

qc = QuantumCircuit(2) 

qc.h(0) 

qc.cx(0,1) 

qc.measure_all() 

# First, we run the circuit using the statevector simulator. 

sim = AerSimulator() 

result = execute(qc, sim, shots = 1024).result() 

counts_sim = result.get_counts() 

# Now we run it on the real device that we selected before. 

job = execute(qc, dev, shots = 1024) 

job_monitor(job) 

result = job.result() 

counts_dev = result.get_counts()

```

获取结果可能需要一段时间（你将通过 `job_monitor` (`job`) 指令获得状态更新）。实际上，有时你可能需要经历相当长的等待时间，因为许多用户同时提交作业。但只要有耐心，结果最终会到来！一旦执行完成，我们可以打印结果，可能会得到类似以下的内容：

```py

print(counts_sim) 

print(counts_dev)

```

```py

{’11’: 506, ’00’: 518} 
{’00’: 431, ’01’: 48, ’10’: 26, ’11’: 519}

```

这已经很接近了，但还远非理想！我们可以看到在真实硬件上的执行中，我们得到了一些输出——即 `10` 和 `01`——这些输出在最初甚至都不应该被允许。这是真实量子计算机的**噪声**效应，这使得它们偏离了完美的数学模拟。

要了解更多...

在这里，我们只进行了理想模拟。你还可以在 Qiskit 中进行有噪声的模拟，这可以更真实地模拟我们今天可用的量子计算机的行为。此外，你可以配置这些模拟以使用与 IBM 拥有的真实量子设备中测量的相同噪声参数。我们将在*第* *7* *章* *VQE：变分量子本征值求解器* *中学习如何做到这一点*。

*重要提示

当在真实的量子硬件上执行量子电路时，你必须意识到现实中的量子系统只实现某些门，因此组成电路的一些门可能需要使用可用的门进行分解。例如，将多量子比特门分解为仅作用于一个或两个量子比特的门是典型的做法，或者通过首先交换量子比特，然后应用实际存在的 CNOT 门，最后再交换回量子比特，来模拟量子计算机中未直接连接的量子比特之间的 CNOT 门。

这个过程被称为**编译器转换**，并且，使用我们考虑的代码，我们已经让 Qiskit 自动处理所有细节。然而，你可以深入了解这一点，并且尽可能多地对其进行修改！例如，你可以使用`transpile`方法手动定义编译器转换，指定计算机中存在的门或实际连接的量子比特，以及其他事项。有关更多详细信息，请参阅[`qiskit.org/documentation/stubs/qiskit.compiler.transpile.html`](https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html)。

在本节中，我们对 Qiskit 框架的一般结构有了很好的理解，我们学习了如何在其中实现电路，以及如何通过 IBM Quantum 对其进行模拟和运行。在下一节中，我们将对另一个非常有趣的框架：PennyLane，做同样的处理。让我们开始吧！

# 2.3 使用 PennyLane

PennyLane 的结构[103]比 Qiskit 简单。PennyLane 主要包含一个核心软件包，它包含了你期望的所有功能：它允许你实现量子电路，它附带一些出色的内置模拟器，并且它还允许你使用原生工具和**TensorFlow**接口训练量子机器学习模型。

除了这个核心包之外，PennyLane 可以通过一系列插件进行扩展，这些插件提供了与其他量子计算框架和平台的接口。在撰写本文时，这些包括**Qiskit**、**Amazon Braket**、**Microsoft QDK**和**Cirq**等许多我们未在介绍中提到的其他框架。此外，还有一个社区插件**PyQuest**，它使得 PennyLane 能够与 QuEST 模拟器([`github.com/johannesjmeyer/pennylane-pyquest`](https://github.com/johannesjmeyer/pennylane-pyquest))兼容。

简而言之，使用 PennyLane，你不仅仅是获得了两个世界的最佳之处。你真的可以获取任何世界的最佳之处！

重要提示

我们将使用**版本 0.26**的 PennyLane。如果你使用的是不同版本，某些事情可能会有所不同。如果有疑问，你应该始终检查文档([`pennylane.readthedocs.io/en/stable/`](https://pennylane.readthedocs.io/en/stable/))。

练习 2.4

请按照*附录* **D*中*安装工具*的说明，安装 PennyLane 及其 Qiskit 插件的**版本 0.26**。

*一旦你安装了 PennyLane，你就可以导入它。按照 PennyLane 文档中规定的约定，我们将如下操作：

```py

import pennylane as qml

```

执行此指令后，你可以通过打印字符串`qml``.``__version__`来检查你正在运行的 PennyLane 版本。

现在我们已经设置好了，让我们构建我们的第一个电路，好吗？

## 2.3.1 电路工程 101

在 PennyLane 中构建量子电路的方式与在 Qiskit 中构建的方式根本不同。

在 Qiskit 中，如果我们想实现一个量子电路，我们会初始化一个`QuantumCircuit`对象，并使用一些方法来操作它；其中一些方法用于向电路添加门，一些用于执行测量，还有一些用于指定我们想要提取有关电路状态信息的位置。

在 PennyLane 中，另一方面，如果你想运行一个电路，你需要两个元素：一个`Device`对象和一个指定电路的函数。

用简单的话来说，`Device`对象是 PennyLane 对量子设备的虚拟模拟。它是一个具有方法的对象，允许它运行任何给定的电路（通过模拟器、通过与其他平台的接口，或任何其他方式！）例如，如果我们有一个电路，并且我们想在`default``.``qubit`模拟器上使用两个量子比特运行它（关于这一点将在本节后面详细说明），我们需要使用这个设备：

```py

dev = qml.device(’default.qubit’, wires = 2)

```

顺便说一下，注意一下可用的量子比特数量是设备对象本身的属性。

现在我们有了设备，我们需要定义电路的规格。正如我们之前提到的，这就像定义一个函数一样简单。在这个函数中，我们将执行与我们要使用的量子门动作相对应的指令。最后，函数的输出将是我们要从电路中获取的任何信息——无论是电路的状态、一些测量样本，还是其他任何信息。当然，我们可以获取的输出将取决于我们使用的设备。

让我们用一个例子来说明这一点：

```py

def qc(): 

    qml.PauliX(wires = 0) 

    qml.Hadamard(wires = 0) 

    return qml.state()

```

在这里，我们有一个非常基本的电路规格。在这个电路中，我们首先在第一个量子比特上应用一个![X](img/file9.png "X")门，然后在第一个量子比特上再应用一个![H](img/file10.png "H")门，之后我们得到状态向量（使用`qml.state()`）。我们通过依次调用`qml.PauliX`和`qml.Hadamard`，指定我们想要门作用在其上的线来实现这一点。在大多数非参数化门中，`wires`是第一个位置参数，它没有默认值，因此你需要提供一个。在单量子比特门的情况下，这个值必须是一个整数，表示门要作用的量子比特。类似地，对于多量子比特门，`wires`必须是一个整数列表。

你可能已经注意到，PennyLane 中门类的命名约定与 Qiskit 中门方法的命名约定不同。![X](img/file9.png "X")、![Y](img/file11.png "Y")和![Z](img/file8.png "Z")泡利门的函数分别是`qml.PauliX`、`qml.PauliY`和`qml.PauliZ`。同样，正如我们刚刚看到的，哈达玛门的函数是`qml.Hadamard`。

关于旋转门，我们可以在线`w`上应用由`theta`参数化的![R_{X}](img/file118.png "R_{X}")、![R_{Y}](img/file119.png "R_{Y}")和![R_{Z}](img/file120.png "R_{Z}")，分别使用指令`qml.RX(phi=theta, wires=w)`、`qml.RY(phi=theta, wires=w)`和`qml.RZ(phi=theta, wires=w)`。此外，通用的单量子比特门![U(\theta,\varphi,\lambda)](img/lambda)")可以通过调用`qml.U3(theta, phi, lambd, w)`在线`w`上应用。

最后，受控泡利门可以通过指令`qml.CNOT(w)`、`qml.CY(w)`和`qml.CZ(w)`应用于一对量子比特`w` `=` `[w0, w1]`。第一条线`w0`意味着要作为控制量子比特，而第二条线必须是目标量子比特。通过指令`qml.CRX(theta, w)`、`qml.CRY(theta, w)`和`qml.CRZ(theta, w)`可以分别添加由角度`theta`参数化的受控![X](img/file9.png "X")、![Y](img/file11.png "Y")和![Z](img/file8.png "Z")旋转。

在任何情况下，我们现在有一个双量子比特设备`dev`和一个电路函数`qc`。我们如何将这两个组合在一起并运行电路？很简单，我们只需要执行以下操作：

```py

qcirc = qml.QNode(qc, dev) # Assemble the circuit & the device. 

qcirc() # Run it!

```

如果我们运行这个，我们将得到以下结果，

```py

tensor([ 0.70710678+0.j,  0\.        +0.j, -0.70710678+0.j, 
         0\.        +0.j], requires_grad=True)

```

这完全合理，因为我们知道

| ![此处应有图片](img/otimes I)\left&#124; {00} \right\rangle = (H \otimes I)\left&#124; {10} \right\rangle = \frac{1}{\sqrt{2}}\left( {\left&#124; {00} \right\rangle - \left&#124; {10} \right\rangle} \right) \approx (0.7071\ldots)\left( {\left&#124; {00} \right\rangle - \left&#124; {10} \right\rangle} \right).") |
| --- |

作为有趣的事实，在 PennyLane 术语中，将电路函数和设备组合的结果称为**量子节点**（或简称**QNode**）。

重要提示

与 Qiskit 不同，PennyLane 像大多数人一样标记状态：将最重要的比特分配给第一个量子比特。因此，PennyLane 的输出`10`对应于状态![\left| {10} \right\rangle](img/rangle")。

注意，与 PennyLane 对状态标记的约定一致，状态向量以列表形式返回，其中包含计算基中各状态的概率幅。第一个元素对应于状态![\left| {0\cdots 0} \right\rangle](img/rangle")的概率幅，第二个对应于![\left| {0\cdots 01} \right\rangle](img/rangle")的概率幅，依此类推。

在前面的示例中，我们应该指出，在函数`qc`的定义中，我们没有指定电路的量子比特数量——我们将其留给了设备。当我们创建一个 QNode 时，PennyLane 假设设备有足够的量子比特来执行电路规范。如果情况不是这样，当执行相应的 QNode 时，我们将遇到`WireError`异常。

如果你——就像我们中的大多数人一样——很懒，那么定义一个函数并将其与设备组装起来的整个过程可能看起来非常累人。幸运的是，PennyLane 的团队非常友好，提供了一条捷径。如果你有一个设备`dev`并且想要为其定义一个电路，你可以简单地做以下操作：

```py

@qml.qnode(dev) # We add this decorator to use the device dev. 

def qcirc(): 

    qml.PauliX(wires = 0) 

    qml.Hadamard(wires = 0) 

    return qml.state() 

# Now qcirc is already a QNode. We can just run it! 

qcirc()

```

现在看起来要酷多了！通过在电路函数定义之前放置`@qml``.``qnode``(``dev``)`装饰器，它自动变成了一个 QNode，我们无需做任何其他事情。

我们已经看到 PennyLane 中的电路是如何实现为简单的函数的，这引发了一个问题：我们是否可以在这些函数中使用参数？答案是响亮的肯定。让我们假设我们想要构建一个由某个 `theta` 参数化的单量子比特电路，该电路通过这个参数执行 ![X](img/file9.png "X")-旋转。这样做就像这样：

```py

dev = qml.device(’default.qubit’, wires = 1) 

@qml.qnode(dev) 

def qcirc(theta): 

    qml.RX(theta, wires = 0) 

    return qml.state()

```

并且，有了这个，对于我们所选择的任何 `theta` 值，我们都可以运行 `qcirc(theta)` 并得到我们的结果。这种方式处理参数非常方便和方便。当然，你可以在电路定义中使用循环和依赖于电路参数的条件。可能性是无限的！

如果在任何时候你需要绘制 PennyLane 中的电路，那不是问题：这相当直接。一旦你有一个量子节点 `qcirc`，你可以将这个节点传递给 `qml.draw` 函数。这将返回一个函数，`qml.draw(qcirc)`，它将接受与 `qcirc` 相同的参数，并将给出一个字符串，为这些参数的每个选择绘制电路。我们可以用一个例子更清楚地看到这一点。让我们执行以下代码来绘制我们刚刚考虑的 `qcirc` 电路，其中 ![theta = 2](img/theta = 2")：

```py

print(qml.draw(qcirc)(theta = 2))

```

运行后，我们得到以下电路表示：

```py

0: --RX(2.00)--|  State

```

到目前为止，我们只执行了返回电路执行结束时状态向量的模拟，但，自然地，这只是 PennyLane 提供的许多选项之一。这些是一些，但不是所有的返回值，我们可以在电路函数中拥有：

+   如果我们想在电路执行结束时获取其状态，我们可以像之前看到的那样，返回 `qml.state()`。

+   如果我们希望得到一个列表，其中包含列表 `w` 中每个状态在计算基的概率，我们可以返回 `qml.probs(wires = w)`。

+   我们可以通过返回 `qml.sample(wires = w)` 来获取一些线 `w` 在计算基中的测量样本；`wires` 参数是可选的（如果没有提供值，则测量所有量子比特）。当我们得到一个样本时，我们必须通过在调用设备时设置 `shots` 参数或在调用 QNode 时设置它来指定其大小。

我们将在 *第* **10** 章 *量子神经网络* 中探索一些额外的返回值可能性。我们已经知道如何获取电路的状态。为了说明我们可能使用的其他返回值，让我们执行以下代码：*

*```py

dev = qml.device(’default.qubit’, wires = 3) 

# Get probabilities 

@qml.qnode(dev) 

def qcirc(): 

    qml.Hadamard(wires = 1) 

    return qml.probs(wires = [1, 2]) # Only the last 2 wires. 

prob = qcirc() 

print("Probs. wires [1, 2] with H in wire 1:", prob) 

# Get a sample, not having specified shots in the device. 

@qml.qnode(dev) 

def qcirc(): 

    qml.Hadamard(wires = 0) 

    return qml.sample(wires = 0) # Only the first wire. 

s1 = qcirc(shots = 4) # We specify the shots here. 

print("Sample 1 after H:", s1) 

# Get a sample with shots in the device. 

dev = qml.device(’default.qubit’, wires = 2, shots = 4) 

@qml.qnode(dev) 

def qcirc(): 

    qml.Hadamard(wires=0) 

    return qml.sample() # Will sample all wires. 

s2 = qcirc() 

print("Sample 2 after H x I:", s2)

```

这次执行得到的输出如下（你返回的样本可能会不同）：

```py

Probs. wires [1, 2] with H in wire 1: [0.5 0\.  0.5 0\. ] 
Sample 1 after H: [0 1 0 0] 
Sample 2 after H x I: [[1 0], [0 0], [0 0], [1 0]]

```

这里可能有一些内容需要解释。首先，我们得到一个概率列表；根据 PennyLane 的约定，这些概率是得到![00](img/file157.png "00")、![01](img/file159.png "01")、![10](img/file161.png "10")和![11](img/file163.png "11")的概率。在这些可能的结果中，第一个（最左边的）位表示第一个测量的量子比特的结果：在我们的情况下，因为我们测量的是线 `[1,` `2]`，即电路的第二根线，线![1](img/file13.png "1")。第二个（最右边的）位表示第二个测量的量子比特的结果：在我们的情况下，电路的第三根线。例如，概率列表中的第一个数字表示得到![00](img/file157.png "00")的概率（即两根线上都是![0](img/file12.png "0")）。列表中的第二个数字表示得到![01](img/file159.png "01")的概率（即线![1](img/file13.png "1")上是![0](img/file12.png "0")，线![2](img/file302.png "2")上是![1](img/file13.png "1")）。以此类推。

最后，在接下来的两个例子中，我们得到了一些测量样本。在第一种情况下，我们指定只测量第一个量子比特（线![0](img/file12.png "0")），当我们调用 QNode 时，我们要求![4](img/file143.png "4")次射击；因为我们定义设备时没有指定默认的射击次数，所以我们需要在执行时指定。这样，我们就得到了第一个量子比特的样本。在我们的情况下，结果先是 0，然后是 1，然后是两个更多的 0。

在最后一个例子中，我们定义了一个双量子比特电路，并测量了所有线。我们在定义设备时已经指定了默认的射击次数 (![4](img/file143.png "4"))，所以在调用 QNode 时不需要做。执行后，我们得到了测量样本。列表中的每个项目都对应一个样本。在每个样本中，第一个元素给出了测量电路第一个量子比特的结果，第二个元素给出了测量第二个量子比特的结果，依此类推。例如，在我们的情况下，我们看到在第一次测量中，第一个量子比特得到了![1](img/file13.png "1")，第二个量子比特得到了![0](img/file12.png "0")。

练习 2.5

在*图* *2.2* 中实现电路，并验证你得到的状态向量与我们使用 Qiskit Aer 模拟得到的状态向量相同。

请记住，正如我们之前提到的，Qiskit 和 PennyLane 在命名基态时使用不同的约定。请注意这一点！

要了解更多……

如果你想要使用 PennyLane 的模拟器得到可重复的结果，你可以在导入`numpy`包作为`np`之后，使用指令`np.random.seed(s)`设置一个种子`s`。

到目前为止，我们一直在使用基于`default.qubit`模拟器的设备，这是一个基于 Python 的模拟器，具有一些基本功能。当我们深入量子机器学习的世界时，我们将介绍更多的模拟器。然而，现在，你至少应该了解`lightning.qubit`模拟器的存在，它依赖于 C++后端，并在性能上提供了显著提升，尤其是在具有大量量子比特的电路中。它的使用方式与`default.qubit`模拟器类似。此外，还有一个`lightning.gpu`模拟器，可以使 Lightning 模拟器依赖于你的 GPU。它可以作为插件安装。正如 Qiskit 的情况一样，在撰写本书时，它只支持 NVIDIA GPU（并且主要是相当现代的！）。

## 2.3.2 PennyLane 的互操作性

我们已经多次提到 PennyLane 的一个优点是它能够与其他量子框架进行通信。现在，我们将通过 PennyLane 的 Qiskit 接口来尝试展示这一点。你会说 Qiskit 吗？

当你安装 PennyLane 的 Qiskit 插件时，你将获得一组新的设备：最值得注意的是，一个`qiskit.aer`设备，它允许你直接从 PennyLane 使用 Aer 模拟器，以及一个`qiskit.ibmq`设备，它使你能够在 IBM Quantum 提供的真实量子计算机上运行电路。

### 爱在 Aer 中

如果我们想在 PennyLane 中使用 Aer 模拟器模拟电路，我们只需要使用一个带有`qiskit.aer`模拟器的设备——当然，前提是你已经安装了适当的插件（参考*附录* **D*，*安装工具*）。这将使我们能够获取测量样本以及测量概率（分别通过`qml.sample`和`qml.probs`）。实际上，这些 Aer 设备返回的测量概率是精确概率的近似：它们是通过采样并返回经验概率获得的。默认情况下，在 Aer 设备中，射击次数固定为![1024](img/file291.png "1024")，遵循 Qiskit 的约定。当然，射击次数可以像任何其他 PennyLane 设备一样进行调整。

*我们可以通过以下代码示例看到`qiskit.aer`设备在行动中：

```py

dev = qml.device(’qiskit.aer’, wires = 2) 

@qml.qnode(dev) 

def qcirc(): 

    qml.Hadamard(wires = 0) 

    return qml.probs(wires = 0) 

s = qcirc() 

print("The probabilities are", s)

```

当我们运行这个程序时，我们可以得到以下类似输出：

```py

The probabilities are [0.48535156 0.51464844]

```

这表明，确实，结果不是解析的，而是经验性的，并从样本中提取出来的。如果你想获得状态向量，你需要在创建设备时使用以下指令：

```py

dev = qml.device(’qiskit.aer’, wires = 2, 

    backend=’aer_simulator_statevector’, shots = None)

```

这将允许你使用`qml.state()`来检索状态振幅，就像我们使用 PennyLane 设备时做的那样。此外，如果你尝试使用`qml.probs`与这个设备对象获取概率，你现在将得到解析结果。例如，如果你在这个设备上运行前面的示例，你将始终获得`[0.5, 0.5]`。

### 连接到 IBMQ

能够（部分）使用 Aer 模拟器可能是 PennyLane Qiskit 接口最吸引人的特性。然而，能够连接到 IBM 的量子计算机是一个更令人兴奋的可能性。

为了连接到 IBM 量子设备，我们首先加载 Qiskit 并获取最不繁忙的硬件后端名称，就像我们在上一节中所做的那样：

```py

from qiskit import * 

from qiskit.providers.ibmq import * 

# Save our token if we haven’t already. 

IBMQ.save_account(’TOKEN’) 

# Load the account and get the name of the least busy backend. 

prov = IBMQ.load_account() 

bck = least_busy(prov.backends(simulator = False)).name() 

# Invoke the PennyLane IBMQ device. 

dev = qml.device(’qiskit.ibmq’, wires = 1, 

    backend = bck, provider = prov) 

# Send a circuit and get some results! 

@qml.qnode(dev) 

def qcirc(): 

    qml.Hadamard(wires = 0) 

    return qml.probs(wires = 0) 

print(qcirc())

```

执行前面的代码后，我们得到了我们期望的结果：

```py

[0.51660156 0.48339844]

```

这就是如何使用 PennyLane 将作业发送到 IBM 量子计算机的方法！

要了解更多...

当然，你可以使用任何可以通过你的 IBM 账户访问的量子设备，而不仅仅是使用最不繁忙的那个。你只需要将之前代码中后端的定义替换为我们在上一节中直接指定的特定计算机即可。

通过对 Qiskit 和 PennyLane 的工作原理的介绍，以及我们在*第* **1* *章* *[*1*]*中学习的所有数学概念，我们现在准备好开始使用实际的量子算法解决问题。这正是我们将在下一章中做的事情。量子游戏开始——愿你的量子游戏总是顺利！*

*# 摘要

在本章中，我们探索了一些可以让我们实现、模拟和运行量子算法的框架和平台。我们还学习了如何使用这些框架中的两个：Qiskit 和 PennyLane，它们被广泛使用。除此之外，我们还学习了如何使用 IBM 量子平台在真实硬件上执行量子电路，无论是从 Qiskit 还是 PennyLane 发送。

在本章中，你获得了技能，现在你可以实施并执行你自己的电路。此外，你已经为阅读本书的其余部分做好了充分的准备，因为我们将会大量使用 Qiskit 和 PennyLane。

在下一章中，我们将迈出第一步，将所有这些知识付诸实践。我们将深入量子优化的世界！*****************
