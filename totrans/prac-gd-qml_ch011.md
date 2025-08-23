# 第3章

与二次无约束二进制优化问题一起工作

*宇宙的语言在我们学会这种语言并* *熟悉其字符之前是无法被阅读的*。

——伽利略·伽利莱

从本章开始，我们将研究被提出用于解决量子计算机优化问题的不同算法。我们将与**量子退火器**以及实现**量子电路模型**的计算机一起工作。我们将使用**量子近似优化算法**（**QAOA**）、**Grover的** **自适应搜索**（**GAS**）和**变分量子本征求解器**（**VQE**）等方法。我们还将学习如何将这些算法适应不同类型的问题，以及如何在模拟器和实际的量子计算机上运行它们。

但在我们能够做所有这些之前，我们需要一种语言，我们可以用它以使量子计算机能够解决问题的方式陈述问题。在这方面，使用**二次无约束二进制** **优化**（**QUBO**）框架，我们可以以直接映射到量子设置的方式公式的许多不同的优化问题，使我们能够使用大量的量子算法来尝试找到最优或至少接近最优的解决方案。

本章将介绍我们用于处理QUBO公式的所有工具。我们将从研究图中的**最大割**（或**Max-Cut**）问题开始，这可能是QUBO框架中可以公式的最简单问题，然后我们将逐步深入。

本章我们将涵盖以下主题：

+   最大割问题与伊辛模型

+   进入量子领域：以量子方式公式的优化问题

+   从伊辛到QUBO以及返回

+   基于QUBO模型的组合优化问题

阅读本章后，你将准备好以适合使用量子计算机解决优化问题的格式编写自己的优化问题。

# 3.1 最大割问题与伊辛模型

为了让我们了解如何使用量子计算机解决优化问题，我们需要习惯一些在本章中我们将发展的抽象和技巧。为了开始，我们将考虑在称为**图**的数学结构中找到我们所说的**最大割**的问题。这可能是在我们将在以下章节中使用的形式主义中可以写出的最简单问题。这将帮助我们获得直觉，并为以后公式的更复杂问题提供一个坚实的基础。

## 3.1.1 图和割集

当你得到一个图时，你实际上得到了一些*元素*，我们将它们称为**顶点**，以及这些顶点对之间的某些*连接*，我们将它们称为**边**。参见*图*[*3.1*](#Figure3.1)以了解一个具有五个顶点和六条边的图的示例。

*![图3.1：图的示例](img/file303.jpg)

**图3.1**：图的示例

给定一个图，**最大割问题**在于找到它的**最大割**。也就是说，我们想要将图的顶点分成两个集合——这就是我们所说的将图**割**成两部分——使得不同集合的割边数量达到最大。我们称这样的边数为**割的大小**，并称这些边为**割边**。你可以想象，例如，顶点代表公司的员工，边被添加到那些相处不太融洽的人之间，你需要组成两个团队，通过将潜在的敌人分到不同的团队中来尽量减少冲突。

*图*[*3.2*](#Figure3.2)展示了*图*[*3.1*](#Figure3.1)的两种不同割，使用不同颜色表示属于不同集合的顶点，使用虚线表示割的不同部分的边。如图所示，*图*[*3.2a*](#Figure3.2a)的割大小为![5](img/file296.png "5")，而*图*[*3.2b*](#Figure3.2b)的割大小为![4](img/file143.png "4")。实际上，很容易检查出这个图没有任何割的尺寸能超过![5](img/file296.png "5")，因为顶点![0,1](img/file304.png "0,1")和![2](img/file302.png "2")不能全部属于不同的集合，因此至少有一条边![（0,1）](img/file305.png "(0,1)"), ![（0,2）](img/file306.png "(0,2)"), 或![（1,2）](img/file307.png "(1,2)")不会被割。因此，*图*[*3.2a*](#Figure3.2a)是一个**最大**或**最优**割。

**![（a）](img/file308.jpg)**

**（a）**

![（b）](img/file309.jpg)

**（b）**

**图3.2**：同一图的两种不同割

练习3.1

图中的最大割不一定是唯一的。在*图*[*3.1*](#Figure3.1)中找到最大割，其中顶点![0](img/file12.png "0")和![1](img/file13.png "1")属于同一集合。

*因此，我们现在已经知道了最大割问题是什么。但是，我们如何将其数学化呢？我们将在下一小节中详细了解这一点。*

## 3.1.2 问题表述

令人惊讶的是，我们可以将最大割问题表述为一个与图、边或顶点无关的组合优化问题。为了做到这一点，我们为图中的每个顶点![i = 0,\ldots,n - 1](img/file311.png "i = 0,\ldots,n - 1")关联一个变量![z_{i}](img/file310.png "z_{i}")。变量![z_{i}](img/file310.png "z_{i}")将取值![1](img/file13.png "1")或![- 1](img/file312.png "- 1")。变量的每个值赋都确定了一个割：取值为![1](img/file13.png "1")的变量对应的顶点将属于一个集合，而取值为![- 1](img/file312.png "- 1")的变量对应的顶点将属于另一个集合。例如，对于*图* * [*3.2a*](#Figure3.2a)的割，我们可能有![z_{0} = z_{2} = z_{3} = 1](img/file313.png "z_{0} = z_{2} = z_{3} = 1")和![z_{1} = z_{4} = - 1](img/file314.png "z_{1} = z_{4} = - 1")。请注意，为了我们的目的，我们也可以用![z_{0} = z_{2} = z_{3} = - 1](img/file315.png "z_{0} = z_{2} = z_{3} = - 1")和![z_{1} = z_{4} = 1](img/file316.png "z_{1} = z_{4} = 1")的赋值来表示那个割。*

*将最大割问题表述为组合优化问题的关键观察是注意到，如果两个顶点![j](img/file258.png "j")和![k](img/file317.png "k")之间存在一条边，那么这条边被割断当且仅当![z_{j}z_{k} = - 1](img/file318.png "z_{j}z_{k} = - 1")。这是因为如果这两个顶点属于同一个集合，那么要么![z_{j} = z_{k} = 1](img/file319.png "z_{j} = z_{k} = 1")，要么![z_{j} = z_{k} = - 1](img/file320.png "z_{j} = z_{k} = - 1")，从而![z_{j}z_{k} = 1](img/file321.png "z_{j}z_{k} = 1")。然而，如果它们属于不同的集合，那么要么![z_{j} = 1](img/file322.png "z_{j} = 1")且![z_{k} = - 1](img/file323.png "z_{k} = - 1")，要么![z_{j} = - 1](img/file324.png "z_{j} = - 1")且![z_{k} = 1](img/file325.png "z_{k} = 1")，从而![z_{j}z_{k} = - 1](img/file318.png "z_{j}z_{k} = - 1")。*

因此，我们的问题可以写成

![\begin{array}{rlrl} {\text{Minimize~}\quad} & {\sum\limits_{(j,k) \in E}z_{j}z_{k}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,\ldots,n - 1\qquad} & & \qquad \\ \end{array}](img/file326.png "\begin{array}{rlrl} {\text{Minimize~}\quad} & {\sum\limits_{(j,k) \in E}z_{j}z_{k}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,\ldots,n - 1\qquad} & & \qquad \\ \end{array}")

其中![E](img/file327.png "E")是图中边的集合，顶点是![\{ 0,\ldots,n - 1\}](img/file328.png "\{ 0,\ldots,n - 1\}")。例如，对于*图* * [*3.1*](#Figure3.1)中的图，我们会有以下表述：*

*![\begin{array}{rlrl} {\text{Minimize~}\quad} & {z_{0}z_{1} + z_{0}z_{2} + z_{1}z_{2} + z_{1}z_{3} + z_{2}z_{4} + z_{3}z_{4}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,\ldots,4.\qquad} & & \qquad \\ \end{array}](img/file329.png "\begin{array}{rlrl} {\text{Minimize~}\quad} & {z_{0}z_{1} + z_{0}z_{2} + z_{1}z_{2} + z_{1}z_{3} + z_{2}z_{4} + z_{3}z_{4}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,\ldots,4.\qquad} & & \qquad \\ \end{array}")

注意，切割 ![z_{0} = z_{2} = z_{3} = 1](img/file313.png "z_{0} = z_{2} = z_{3} = 1"), ![z_{1} = z_{4} = - 1](img/file314.png "z_{1} = z_{4} = - 1")（如图 *图* * [*3.2a*](#Figure3.2a) 所示），在要最小化的函数中达到的值为 ![- 4](img/file330.png "- 4")，这是此特定情况的最小可能值——但请注意，它并不与切割的边数相吻合！另一方面，切割 ![z_{0} = z_{3} = - 1](img/file331.png "z_{0} = z_{3} = - 1"), ![z_{1} = z_{2} = z_{4} = 1](img/file332.png "z_{1} = z_{2} = z_{4} = 1")，达到的值为 ![- 2](img/file333.png "- 2")，再次表明 *图* * [*3.2b*](#Figure3.2b) 中的切割不是最优的。

**练习 3.2

将 *图* *[*3.3*](#Figure3.3) 中的最大切割问题写成优化问题。当 ![z_{0} = z_{1} = z_{2} = 1](img/file334.png "z_{0} = z_{1} = z_{2} = 1") 和 ![z_{3} = z_{4} = z_{5} = - 1](img/file335.png "z_{3} = z_{4} = z_{5} = - 1") 时，要最小化的函数的值是多少？这是一个最优切割吗？

*![图 3.3: 另一个图的示例](img/file336.jpg)

**图 3.3**：另一个图的示例

初看起来，解决最大切割问题似乎足够简单。然而，它是一个 **NP-hard** 问题（有关这类问题的更多详细信息，请参阅 *附录* *[*C*](ch026.xhtml#x1-233000C), *计算复杂性*）。这意味着如果我们能够用经典算法有效地解决它，我们就会得到 ![P = NP](img/file337.png "P = NP")，这是科学界普遍认为不真实的事情。即使我们能够找到一个经典算法，在因子 ![\left. 16\slash 17 \right.](img/file338.png "\left. 16\slash 17 \right.") 内近似最优切割，这种情况也会发生，正如 Håstad 在 2001 年发表的一篇论文中证明的那样 [[50](ch030.xhtml#Xhastad01optimal)]。因此，即使我们求助于寻找足够精确的近似值，这个问题确实很困难！

*要了解更多…

如果你想了解更多关于 ![P](img/file1.png "P"), ![NP](img/file2.png "NP"), 和 ![NP](img/file2.png "NP")-hard 问题，请查看 *附录* * [*C*](ch026.xhtml#x1-233000C), *计算复杂性*。我们将在 *第* * [*5*](ch013.xhtml#x1-940005), *QAOA: 量子* *近似优化算法* * 中讨论量子算法对于最大切割问题所能达到的近似比率。

**我们现在能够将Max-Cut表述为一个变量取值为![1](img/file13.png "1")和![ - 1](img/file312.png "- 1")的最小化问题。这仅仅是巧合，还是有更多问题可以用类似的方式表述？继续阅读，你将在下一小节中找到答案。

## 3.1.3 伊辛模型

如前几页所阐述的Max-Cut问题，可以看作是统计物理中一个看似无关问题的特例：寻找伊辛模型实例的最小**能量**状态。对于物理学爱好者来说，这是一个描述具有**自旋**粒子的铁磁相互作用的数学模型，通常排列在晶格中（参见*图* *[*3.4*](#Figure3.4) 并参考Gallavotti的书籍[[42](ch030.xhtml#Xgallavotti1999statistical)]以获取更多详细信息）。粒子自旋由变量![z_{j}](img/file339.png "z_{j}")表示，可以取值![1](img/file13.png "1")（自旋向上）或![ - 1](img/file312.png "- 1")（自旋向下）——听起来熟悉，不是吗？

*![图3.4：伊辛模型的示例](img/file340.jpg)

**图3.4**：伊辛模型的示例

系统的总能量由称为**哈密顿**函数的量给出（关于这一点将在本章后面详细介绍）定义为

![ - \sum\limits_{j,k}J_{jk}z_{j}z_{k} - \sum\limits_{j}h_{j}z_{j}](img/file341.png "- \sum\limits_{j,k}J_{jk}z_{j}z_{k} - \sum\limits_{j}h_{j}z_{j}")

其中，系数![J_{jk}](img/file342.png "J_{jk}")代表粒子![j](img/file258.png "j")和![k](img/file317.png "k")之间的相互作用（通常，只有相邻粒子的系数不为零）和系数![h_{j}](img/file343.png "h_{j}")代表外部磁场对粒子![j](img/file258.png "j")的影响。

寻找系统的最小能量状态，在于获得一个使得哈密顿函数达到最小值的自旋配置。正如你可以轻松检查的那样，当所有![J_{jk}](img/file342.png "J_{jk}")系数为![ - 1](img/file312.png "- 1")，所有![h_{j}](img/file343.png "h_{j}")系数为![0](img/file12.png "0")时，问题与在图中获得最大割集的问题完全相同——尽管在完全不同的背景下！当然，这使得寻找给定伊辛模型的最小能量状态成为一个![NP](img/file2.png "NP")-难问题。

要了解更多…

我们将在*第4章* *[*4*](ch012.xhtml#x1-750004)，*量子绝热计算与量子退火*中使用的量子退火器，是专门用于从可以用伊辛模型描述的系统低能量状态中进行采样的量子计算机。我们将利用这一特性来尝试近似Max-Cut和其他许多相关问题的解。

*让我们以寻找伊辛模型最小能量状态的问题为例。想象一下，我们有一些粒子按照 *图* *[*3.4*](#Figure3.4) 中的方式排列，其中边上的数字代表系数 ![J_{jk}](img/file342.png "J_{jk}")，我们假设外部磁场是均匀的，并且所有系数 ![h_{j}](img/file343.png "h_{j}") 都等于 ![1](img/file13.png "1")。然后，问题可以表述如下：*

*![\begin{array}{rlrl} {\text{Minimize~}\quad} & {z_{0}z_{1} - 2z_{1}z_{2} + z_{2}z_{3} - 3z_{0}z_{4} + z_{1}z_{5} + z_{2}z_{6} - 3z_{3}z_{7}\qquad} & & \qquad \\ & {+ z_{4}z_{5} - 2z_{5}z_{6} + z_{6}z_{7} - 3z_{4}z_{8} + z_{5}z_{9} + z_{6}z_{10} - 3z_{7}z_{11}\qquad} & & \qquad \\ & {+ z_{8}z_{9} - 2z_{9}z_{10} + z_{10}z_{11} - z_{0} - z_{1} - z_{2} - z_{3} - z_{4} - z_{5}\qquad} & & \qquad \\ & {- z_{6} - z_{7} - z_{8} - z_{9} - z_{10} - z_{11}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,\ldots,11.\qquad} & & \qquad \\ \end{array}](img/file344.png "\begin{array}{rlrl} {\text{Minimize~}\quad} & {z_{0}z_{1} - 2z_{1}z_{2} + z_{2}z_{3} - 3z_{0}z_{4} + z_{1}z_{5} + z_{2}z_{6} - 3z_{3}z_{7}\qquad} & & \qquad \\  & {+ z_{4}z_{5} - 2z_{5}z_{6} + z_{6}z_{7} - 3z_{4}z_{8} + z_{5}z_{9} + z_{6}z_{10} - 3z_{7}z_{11}\qquad} & & \qquad \\  & {+ z_{8}z_{9} - 2z_{9}z_{10} + z_{10}z_{11} - z_{0} - z_{1} - z_{2} - z_{3} - z_{4} - z_{5}\qquad} & & \qquad \\  & {- z_{6} - z_{7} - z_{8} - z_{9} - z_{10} - z_{11}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,\ldots,11.\qquad} & & \qquad \\ \end{array}")

这似乎比我们迄今为止所看到的 Max-Cut 问题的公式更复杂一些，但它显然遵循相同的模式。然而，你可能想知道所有这些与量子计算有什么关系，因为所有这些公式只涉及经典变量。这是一个很好的观点！现在是时候利用我们对量子比特和量子门的知识，尝试以不同的、量子化的视角来看待所有这些问题了。

# 3.2 进入量子：以量子方式制定优化问题

在本节中，我们将揭示我们迄今为止在本章中所做的一切工作是如何遵循一个秘密计划的！选择 ![z](img/file81.png "z") 作为我们问题中变量的名称是完全随意的吗？当然不是！如果你想起了我们在 *第* *[*1*](ch008.xhtml#x1-180001) *章* *[*1*] *《量子计算基础》* 中引入的那些可爱的 ![Z](img/file8.png "Z") 量子门和矩阵，你就走在了正确的道路上。它将是引入 *量子因子* 到我们问题的关键，正如我们将在下一小节中开始看到的。

*## 3.2.1 从经典变量到量子比特

到目前为止，我们考虑的最大切割问题和伊辛模型都是纯粹的经典公式。它们没有提到量子元素，如量子比特、量子门或测量。但事实上，我们比你想的更接近于为这些问题提供一个量子公式。我们将从一个非常简单的最大切割问题实例开始，并展示我们如何轻松地将它转化为*量子形式*。考虑图 *[*3.5*](#Figure3.5)。我们已经知道相应的最大切割问题可以写成以下形式：*

*![\begin{array}{rlrl} {\text{最小化~}\quad} & {z_{0}z_{1} + z_{0}z_{2}\qquad} & & \qquad \\ {\text{约束~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}](img/file345.png "\begin{array}{rlrl} {\text{最小化~}\quad} & {z_{0}z_{1} + z_{0}z_{2}\qquad} & & \qquad \\ {\text{约束~}\quad} & {z_{j} \in \{ - 1,1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}")

![图 3.5：一个非常简单的最大切割问题](img/file346.jpg)

**图 3.5**：一个非常简单的最大切割问题

为了将这个公式转化为量子公式，我们需要做出一个关键观察：我们心爱的 ![Z](img/file8.png "Z") 矩阵可以用来评估我们需要最小化的函数中的不同项。具体来说，很容易验证

![\left\langle 0 \right|Z\left| 0 \right\rangle = \begin{pmatrix} 1 & 0 \\ \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & {- 1} \\ \end{pmatrix}\begin{pmatrix} 1 \\ 0 \\ \end{pmatrix} = 1,\qquad\left\langle 1 \right|Z\left| 1 \right\rangle = \begin{pmatrix} 0 & 1 \\ \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & {- 1} \\ \end{pmatrix}\begin{pmatrix} 0 \\ 1 \\ \end{pmatrix} = - 1.](img/file347.png "\left\langle 0 \right|Z\left| 0 \right\rangle = \begin{pmatrix} 1 & 0 \\ \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & {- 1} \\ \end{pmatrix}\begin{pmatrix} 1 \\ 0 \\ \end{pmatrix} = 1,\qquad\left\langle 1 \right|Z\left| 1 \right\rangle = \begin{pmatrix} 0 & 1 \\ \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & {- 1} \\ \end{pmatrix}\begin{pmatrix} 0 \\ 1 \\ \end{pmatrix} = - 1.")

现在，考虑张量积 ![Z \otimes Z \otimes I](img/file348.png "Z \otimes Z \otimes I") 和基态 ![\left| {010} \right\rangle](img/file251.png "\left| {010} \right\rangle")。我们知道从 *第 1.5.1 节* 中，*[*1.5.1*](ch008.xhtml#x1-360001.5.1)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*，*[*3.5*](#Figure3.5)*

*![\begin{array}{rlrl} {\left\langle {010} \right|Z \otimes Z \otimes I\left| {010} \right\rangle} & {= \left\langle {010} \right|\left( {Z\left| 0 \right\rangle \otimes Z\left| 1 \right\rangle \otimes I\left| 0 \right\rangle} \right)\qquad} & & \qquad \\ & {= \left\langle 0 \right|Z\left| 0 \right\rangle\left\langle 1 \right|Z\left| 1 \right\rangle\left\langle 0 \right|I\left| 0 \right\rangle = 1 \cdot ( - 1) \cdot 1 = - 1.\qquad} & & \qquad \\ \end{array}](img/file349.png "\begin{array}{rlrl} {\left\langle {010} \right|Z \otimes Z \otimes I\left| {010} \right\rangle} & {= \left\langle {010} \right|\left( {Z\left| 0 \right\rangle \otimes Z\left| 1 \right\rangle \otimes I\left| 0 \right\rangle} \right)\qquad} & & \qquad \\  & {= \left\langle 0 \right|Z\left| 0 \right\rangle\left\langle 1 \right|Z\left| 1 \right\rangle\left\langle 0 \right|I\left| 0 \right\rangle = 1 \cdot ( - 1) \cdot 1 = - 1.\qquad} & & \qquad \\ \end{array}")*

我们将![\left| {010} \right\rangle](img/file251.png "\left| {010} \right\rangle")解释为表示一个切割，其中顶点![0](img/file12.png "0")和![2](img/file302.png "2")被分配到一个集合中（因为![\left| {010} \right\rangle](img/file251.png "\left| {010} \right\rangle")中![0](img/file12.png "0")和![2](img/file302.png "2")的量子比特值为![0](img/file12.png "0"))，而顶点![1](img/file13.png "1")被分配到另一个集合（因为![1](img/file13.png "1")量子比特在![\left| {010} \right\rangle](img/file251.png "\left| {010} \right\rangle")中的值为![1](img/file13.png "1")）。然后，事实是乘积![\left\langle {010} \right|Z \otimes Z \otimes I\left| {010} \right\rangle](img/file350.png "\left\langle {010} \right|Z \otimes Z \otimes I\left| {010} \right\rangle")评估为![- 1](img/file312.png "- 1")意味着切割的不同集合中存在极端；这是因为我们使用了![Z \otimes Z \otimes I](img/file348.png "Z \otimes Z \otimes I")，其中![Z](img/file8.png "Z")算子作用于量子比特![0](img/file12.png "0")和![1](img/file13.png "1")。这种行为类似于我们在最小化问题的经典公式函数中的![z_{0}z_{1}](img/file351.png "z_{0}z_{1}")项所具有的行为。

实际上，![Z \otimes Z \otimes I](img/file348.png "Z \otimes Z \otimes I")通常简写为![Z_{0}Z_{1}](img/file352.png "Z_{0}Z_{1}")（下标表示每个![Z](img/file8.png "Z")门的位置；其他位置假设为恒等变换），按照这个惯例，例如，我们会有：

![\\left\\langle {010} \\right\\vert Z_{0}Z_{2}\\left\\vert {010} \\right\\rangle = \\left\\langle 0 \\right\\vert Z\\left\\vert 0 \\right\\rangle\\left\\langle 1 \\right\\vert I\\left\\vert 1 \\right\\rangle\\left\\langle 0 \\right\\vert Z\\left\\vert 0 \\right\\rangle = 1 \\cdot 1 \\cdot 1 = 1](img/file353.png "\\left\\langle {010} \\right\\vert Z_{0}Z_{2}\\left\\vert {010} \\right\\rangle = \\left\\langle 0 \\right\\vert Z\\left\\vert 0 \\right\\rangle\\left\\langle 1 \\right\\vert I\\left\\vert 1 \\right\\rangle\\left\\langle 0 \\right\\vert Z\\left\\vert 0 \\right\\rangle = 1 \\cdot 1 \\cdot 1 = 1")

因为在这个特定的分配下，边 ![\\left( {0,2} \\right)](img/file306.png "\\left( {0,2} \\right)") 没有被切断。

当然，对于任何基态 ![\\left| x \\right\\rangle](img/file267.png "\\left| x \\right\\rangle")，其中 ![x \\in \\{ 000,001,\\ldots,111\\}](img/file354.png "x \\in \\{ 000,001,\\ldots,111\\}")，这也成立，所以如果边 ![\\left( {j,k} \\right)](img/file356.png "\\left( {j,k} \\right)") 在分配 ![x](img/file269.png "x") 下被切断，那么 ![\\left\\langle x \\right\\vert Z_{j}Z_{k}\\left\\vert x \\right\\rangle](img/file355.png "\\left\\langle x \\right\\vert Z_{j}Z_{k}\\left\\vert x \\right\\rangle") 将是 ![- 1](img/file312.png "- 1")，否则将是 ![1](img/file13.png "1")。我们只需要注意，如果 ![j](img/file258.png "j") 和 ![k](img/file317.png "k") 在切断的不同部分，那么它们的量子比特将具有不同的值，乘积将是 ![- 1](img/file312.png "- 1")。

此外，根据线性性质，以下等式成立。

![\\left\\langle x \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert x \\right\\rangle = \\left\\langle x \\right\\vert Z_{0}Z_{1}\\left\\vert x \\right\\rangle + \\left\\langle x \\right\\vert Z_{0}Z_{2}\\left\\vert x \\right\\rangle.](img/file357.png "\\left\\langle x \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert x \\right\\rangle = \\left\\langle x \\right\\vert Z_{0}Z_{1}\\left\\vert x \\right\\rangle + \\left\\langle x \\right\\vert Z_{0}Z_{2}\\left\\vert x \\right\\rangle.")

因此，我们可以将问题重新表述为寻找一个基态 ![\\left| x \\right\\rangle](img/file267.png "\\left| x \\right\\rangle")，使得 ![\\left\\langle x \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert x \\right\\rangle](img/file358.png "\\left\\langle x \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert x \\right\\rangle") 达到最小值。

练习 3.3

计算以下表达式 ![\\left\\langle {010} \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert {010} \\right\\rangle](img/file359.png "\\left\\langle {010} \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert {010} \\right\\rangle") 和 ![\\left\\langle {100} \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert {100} \\right\\rangle](img/file360.png "\\left\\langle {100} \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert {100} \\right\\rangle")。这些状态中是否有任何一个能最小化 ![\\left\\langle x \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert x \\right\\rangle](img/file358.png "\\left\\langle x \\right\\vert \\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \\right)\\left\\vert x \\right\\rangle")？

但这并不是故事的结束。对于任何基态 ![\left| x \right\rangle](img/file267.png "\left| x \right\rangle")，它要么满足 ![Z_{j}Z_{k}\left| x \right\rangle = \left| x \right\rangle](img/file361.png "Z_{j}Z_{k}\left| x \right\rangle = \left| x \right\rangle")，要么满足 ![Z_{j}Z_{k}\left| x \right\rangle = - \left| x \right\rangle](img/file362.png "Z_{j}Z_{k}\left| x \right\rangle = - \left| x \right\rangle")，这很容易验证。请注意，这证明了每个 ![\left| x \right\rangle](img/file267.png "\left| x \right\rangle") 都是 ![Z_{j}Z_{k}](img/file363.png "Z_{j}Z_{k}") 的一个 **特征向量**，其 **特征值** 要么是 ![1](img/file13.png "1")，要么是 ![- 1](img/file312.png "- 1")（有关特征向量和特征值的更多信息，请参阅 *附录* * [*B*](ch025.xhtml#x1-226000B)，*基础线性代数*）。因此，对于 ![x \neq y](img/file364.png "x \neq y")，我们将有*

*![\left\langle y \right|Z_{j}Z_{k}\left| x \right\rangle = \pm \left\langle y \middle| x \right\rangle = 0,](img/file365.png "\left\langle y \right|Z_{j}Z_{k}\left| x \right\rangle = \pm \left\langle y \middle| x \right\rangle = 0,")

因为 ![\left\langle y \middle| x \right\rangle = 0](img/file366.png "\left\langle y \middle| x \right\rangle = 0") 当 ![x \neq y](img/file364.png "x \neq y") 时，正如我们在 *第* * [*1.5.1*](ch008.xhtml#x1-360001.5.1) * *节* * 中所证明的。

*因此，由于一个一般状态 ![\left| \psi \right\rangle](img/file43.png "\left| \psi \right\rangle") 总可以写成 ![\left| \psi \right\rangle = {\sum}_{x}a_{x}\left| x \right\rangle](img/file367.png "\left| \psi \right\rangle = {\sum}_{x}a_{x}\left| x \right\rangle")，根据线性原理，可以得出以下结论

![\begin{array}{rlrl} {\left\langle \psi \right|Z_{j}Z_{k}\left| \psi \right\rangle} & {= \left( {\sum\limits_{y}a_{y}^{\ast}\left\langle y \right|} \right)Z_{j}Z_{k}\left( {\sum\limits_{x}a_{x}\left| x \right\rangle} \right) = \sum\limits_{y}\sum\limits_{x}a_{y}^{\ast}a_{x}\left\langle y \right|Z_{j}Z_{k}\left| x \right\rangle\qquad} & & \qquad \\ & {= \sum\limits_{x}\left| a_{x} \right|^{2}\left\langle x \right|Z_{j}Z_{k}\left| x \right\rangle,\qquad} & & \qquad \\ \end{array}](img/file368.png "\begin{array}{rlrl} {\left\langle \psi \right|Z_{j}Z_{k}\left| \psi \right\rangle} & {= \left( {\sum\limits_{y}a_{y}^{\ast}\left\langle y \right|} \right)Z_{j}Z_{k}\left( {\sum\limits_{x}a_{x}\left| x \right\rangle} \right) = \sum\limits_{y}\sum\limits_{x}a_{y}^{\ast}a_{x}\left\langle y \right|Z_{j}Z_{k}\left| x \right\rangle\qquad} & & \qquad \\  & {= \sum\limits_{x}\left| a_{x} \right|^{2}\left\langle x \right|Z_{j}Z_{k}\left| x \right\rangle,\qquad} & & \qquad \\ \end{array}")

在这里，我们使用了 ![a_{x}^{\ast}a_{x} = \left| a_{x} \right|^{2}](img/file369.png "a_{x}^{\ast}a_{x} = \left| a_{x} \right|^{2}").

因此，再次根据线性原理，以下结论成立

![公式](img/file370.png "公式")

我们知道 \([\sum]_{x}\left| a_{x} \right|^{2} = 1\)(![公式](img/file371.png "公式"))，并且每个 \(\left| a_{x} \right|^{2}\)(![公式](img/file372.png "公式")) 都是非负的，因此成立。

![公式](img/file373.png "公式")

其中 ![\left| x_{\min} \right\rangle](img/file374.png "\left| x_{\min} \right\rangle") 是一个基态（可能不止一个），对于这个基态，![\left\langle x \right|\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \right)\left| x \right\rangle](img/file358.png "\left\langle x \right|\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \right)\left| x \right\rangle") 取得最小值，因此，![x_{\min}](img/file375.png "x_{\min}") 代表了一个最大割。

这可能看起来有点抽象。但我们证明的是，所有可能量子状态的最小值总是在基态之一上达到——这些基态是我们唯一可以直接解释为表示割的态。然后，我们可以将寻找图 *图* *[*3.5*](#Figure3.5) 的最大割的问题重新表述如下：

*![\begin{array}{rlrl} & {\text{最小化~}\quad\left\langle \psi \right|\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \right)\left| \psi \right\rangle = \left\langle \psi \right|Z_{0}Z_{1}\left| \psi \right\rangle + \left\langle \psi \right|Z_{0}Z_{2}\left| \psi \right\rangle,\qquad} & & \qquad \\ & {{\text{其中~}\left| \psi \right\rangle\text{~取自3个量子比特上的量子态集合。}}\qquad} & & \qquad \\ \end{array}](img/file376.png "\begin{array}{rlrl}  & {\text{最小化~}\quad\left\langle \psi \right|\left( {Z_{0}Z_{1} + Z_{0}Z_{2}} \right)\left| \psi \right\rangle = \left\langle \psi \right|Z_{0}Z_{1}\left| \psi \right\rangle + \left\langle \psi \right|Z_{0}Z_{2}\left| \psi \right\rangle,\qquad} & & \qquad \\  & {{\text{其中~}\left| \psi \right\rangle\text{~取自3个量子比特上的量子态集合。}}\qquad} & & \qquad \\ \end{array}")

注意我们引入的变化。在我们之前的公式中，我们只是在基态上最小化，但现在我们知道所有可能状态的最小值是在基态上达到的，所以我们现在是在所有可能量子状态上最小化。这将在未来章节中介绍量子算法来解决这类问题时使我们的生活变得更简单，因为我们有理由使用任何量子状态，而不仅仅局限于那些来自基态的状态。

重要提示

虽然最小能量总是在一个基态上实现，但它也可能在非基态上实现。事实上，如果两个不同的基态![\left| x \right\rangle](img/file267.png "\left| x \right\rangle")和![\left| y \right\rangle](img/file268.png "\left| y \right\rangle")实现了最小能量，那么任何叠加![a\left| x \right\rangle + b\left| y \right\rangle](img/file377.png "a\left| x \right\rangle + b\left| y \right\rangle")也是最小能量的。例如，对于![Z_{0}Z_{1}](img/file352.png "Z_{0}Z_{1}")，其两个基态![\left| {01} \right\rangle](img/file199.png "\left| {01} \right\rangle")和![\left| {10} \right\rangle](img/file200.png "\left| {10} \right\rangle")的能量都是![- 1](img/file312.png "- 1")。那么，任何叠加![a\left| {01} \right\rangle + b\left| {10} \right\rangle](img/file378.png "a\left| {01} \right\rangle + b\left| {10} \right\rangle")也实现了能量![- 1](img/file312.png "- 1")，这是![Z_{0}Z_{1}](img/file352.png "Z_{0}Z_{1}")可能的最小能量。

可以很容易地验证，我们前面的论点适用于任何数量的量子比特和任何张量积之和![Z_{j}Z_{k}](img/file363.png "Z_{j}Z_{k}"), 因此，如果我们有一个顶点集合![V](img/file379.png "V")的图，大小为![n](img/file244.png "n")，边集合为![E](img/file327.png "E")，我们可以将图的Max-Cut问题重写如下：

![最小化~\quad\sum\limits_{(j,k) \in E}\left\langle \psi \right|Z_{j}Z_{k}\left| \psi \right\rangle,\qquad](img/file380.png "\begin{array}{rlrl}  & {\text{最小化~}\quad\sum\limits_{(j,k) \in E}\left\langle \psi \right|Z_{j}Z_{k}\left| \psi \right\rangle,\qquad} & & \qquad \\  & {{\text{其中~}\left| \psi \right\rangle\text{~取自~n~个量子态的集合。}}\qquad} & & \qquad \\ \end{array}")

重要提示

让我们退一步，看看我们已经证明了什么。首先，注意像

![\sum\limits_{(j,k) \in E}Z_{j}Z_{k}](img/file381.png "\sum\limits_{(j,k) \in E}Z_{j}Z_{k}")

这些是**厄米**或**自伴**的。这意味着它们等于它们的共轭转置，这很容易验证，并且它们具有特定的性质，例如具有实特征值并且能够与它们的特征向量形成正交归一基（更多详情请参阅*附录* *[*B*](ch025.xhtml#x1-226000B)，*基础线性代数*）。在我们的情况下，我们已经证明了计算基*是*这样的正交归一基的特征向量。此外，数量

*![\left\langle \psi \right|\left( {\sum\limits_{(j,k) \in E}Z_{j}Z_{k}} \right)\left| \psi \right\rangle = \sum\limits_{(j,k) \in E}\left\langle \psi \right|Z_{j}Z_{k}\left| \psi \right\rangle,](img/file382.png "\left\langle \psi \right|\left( {\sum\limits_{(j,k) \in E}Z_{j}Z_{k}} \right)\left| \psi \right\rangle = \sum\limits_{(j,k) \in E}\left\langle \psi \right|Z_{j}Z_{k}\left| \psi \right\rangle,")

这通常被称为![{\sum}_{(j,k) \in E}Z_{j}Z_{k}](img/file383.png "{\sum}_{(j,k) \in E}Z_{j}Z_{k}")的**期望值**，在那些称为**基态**的特征向量之一上达到其最小值。

这个结果被称为**变分原理**，我们将在*第7章*[*7*](ch015.xhtml#x1-1190007)，*变分量子本征值求解器*中更一般的形式中重新讨论。**对于伊辛模型，情况完全相同。我们可以进行类似的推理，这次还涉及到形式为![Z_{j}](img/file384.png "Z_{j}")的项。每个![Z_{j}](img/file384.png "Z_{j}")都是一个张量积，除了![j](img/file258.png "j")-th位置上的因子等于单位矩阵外，其他所有因子都等于单位矩阵，该位置上的因子是![Z](img/file8.png "Z")。然后，找到具有![n](img/file244.png "n")个粒子和系数![J_{jk}](img/file342.png "J_{jk}")和![h_{j}](img/file343.png "h_{j}")的伊辛模型的最小能量状态，相当于以下问题：

![\begin{array}{rlrl} & {\text{Minimize~}\qquad - \sum\limits_{(j,k) \in E}J_{jk}\left\langle \psi \right|Z_{j}Z_{k}\left| \psi \right\rangle - \sum\limits_{j}h_{j}\left\langle \psi \right|Z_{j}\left| \psi \right\rangle,\qquad} & & \qquad \\ & {{\text{where~}\left| \psi \right\rangle\text{~is~taken~from~the~set~of~quantum~states~on~}n\text{~qubits.}}\qquad} & & \qquad \\ \end{array}](img/file385.png "\begin{array}{rlrl}  & {\text{Minimize~}\qquad - \sum\limits_{(j,k) \in E}J_{jk}\left\langle \psi \right|Z_{j}Z_{k}\left| \psi \right\rangle - \sum\limits_{j}h_{j}\left\langle \psi \right|Z_{j}\left| \psi \right\rangle,\qquad} & & \qquad \\  & {{\text{where~}\left| \psi \right\rangle\text{~is~taken~from~the~set~of~quantum~states~on~}n\text{~qubits.}}\qquad} & & \qquad \\ \end{array}")

因此，我们已经能够将几个组合优化问题转化为*量子形式*。更具体地说，我们将我们的问题重写为寻找一个称为系统**哈密顿量**的自伴矩阵的基态的实例。然而，请注意，我们实际上并不需要获得确切的基态。如果我们能够准备一个状态![\left| \psi \right\rangle](img/file43.png "\left| \psi \right\rangle")，使得振幅![a_{x_{\min}} = \left\langle x_{\min} \middle| \psi \right\rangle](img/file386.png "a_{x_{\min}} = \left\langle x_{\min} \middle| \psi \right\rangle")的绝对值很大，那么当我们测量![\left| \psi \right\rangle](img/file43.png "\left| \psi \right\rangle")时，找到![x_{\min}](img/file375.png "x_{\min}")的概率就会很高。这种方法将是我们将在*第4章*到*第7章*中介绍的计算方法背后的原理。

在以下几节中，我们将看到将组合优化问题重写为基态问题实例的可能性不仅是一个愉快的巧合，而是一种常态，我们将展示如何以这种形式写出许多其他重要的问题。但在我们转向这一点之前，让我们编写一些代码来处理那些![Z](img/file8.png "Z")矩阵的张量积，并计算它们的期望值。

## 3.2.2 使用Qiskit计算期望值

在*第2章*[*2*](ch009.xhtml#x1-400002)，*量子计算中的工具*中，我们介绍了Qiskit可以用来处理量子电路并在模拟器和真实量子计算机上执行它们的主要方式。但Qiskit还允许我们处理量子状态和哈密顿量，将它们与张量积结合并计算它们的期望值，这在处理优化问题时可能很有用，正如我们刚才看到的。学习如何执行这些计算将有助于使我们所介绍的概念更加具体。此外，在*第5章*[*5*](ch013.xhtml#x1-940005)，*QAOA：量子近似优化算法*中，我们将大量使用Qiskit中的哈密顿量，因此我们需要了解如何初始化和操作它们。**

**让我们先通过一个例子来展示如何在Qiskit中定义一个三比特基态，例如![\left| {100} \right\rangle](img/file387.png "\left| {100} \right\rangle")。我们可以用几种不同的方法来完成这个任务。例如，我们首先定义一个单比特态![\left| 0 \right\rangle](img/file6.png "\left| 0 \right\rangle")和![\left| 1 \right\rangle](img/file14.png "\left| 1 \right\rangle")，然后计算它们的张量积。实现这一目标有几种可能的方法。第一种方法是直接使用振幅来初始化一个`Statevector`对象。为此，我们需要导入该类，然后使用输入`[1,0]`（![\left| 0 \right\rangle](img/file6.png "\left| 0 \right\rangle")的振幅）调用其构造函数，如下面的代码片段所示：

[PRE0]

如果您运行此代码，您将得到以下输出：

[PRE1]

这表明，我们确实创建了一个量子态并将其设置为![\left| 0 \right\rangle](img/file6.png "\left| 0 \right\rangle")。当然，要将量子态初始化为![\left| 1 \right\rangle](img/file14.png "\left| 1 \right\rangle")，我们可以执行以下操作：

[PRE2]

获得以下输出：

[PRE3]

实现相同结果的另一种可能更方便的方法是从整数（如`0`或`1`）初始化`Statevector`对象。我们将使用`from_int`方法，并且也很重要地使用`dims`参数来指示状态向量的大小。否则，`0`可能被解释为![\left| 0 \right\rangle](img/file6.png "\left| 0 \right\rangle")、![\left| {00} \right\rangle](img/file198.png "\left| {00} \right\rangle")、![\left| {000} \right\rangle](img/file388.png "\left| {000} \right\rangle")或...（如我们在*第* *[*1.4.1*](ch008.xhtml#x1-280001.4.1)中提到的）。在我们的例子中，我们将`dims`设置为`2`，但通常，我们必须将`dims`设置为![2^{n}](img/file256.png "2^{n}")，其中![n](img/file244.png "n")是比特数，因为这是一个![n](img/file244.png "n")比特系统的振幅数量。然后，我们可以运行*

*[PRE4]

这将产生以下预期的输出：

[PRE5]

在任何情况下，我们现在都可以通过使用`tensor`方法计算张量积来构建更高比特数的态，如下面的几行所示：

[PRE6]

运行它们后，我们将得到以下输出：

[PRE7]

注意，值为![1](img/file13.png "1")的振幅位于第五位。它对应于![\left| {100} \right\rangle](img/file387.png "\left| {100} \right\rangle")，因为在二进制中![100](img/file389.png "100")等于![4](img/file143.png "4")，我们是从![0](img/file12.png "0")开始计数的。

如你所想，当我们处理许多量子比特时，我们计算张量积的方式以及作为振幅向量的表示都可能变得难以解析。以下行展示了使用张量积的更简洁方式以及展示状态的一个更美观的方式，但它们与之前显示的代码达到完全相同的结果：

[PRE8]

在这种情况下，输出将仅仅是 ![\left| 100\rangle \right.](img/file390.png "\left| 100\rangle \right.")。更易于阅读，对吧？

构建状态 ![\left| 100\rangle \right.](img/file390.png "\left| 100\rangle \right.") 的一个更快的方法是再次使用 `from_int` 方法，如下所示

[PRE9]

其中我们指定我们正在使用三个量子比特，通过设置 `dims` `=` `8`（因为我们需要8个振幅来定义一个三量子比特状态）。

因此，我们现在知道了创建基态的多种方法。那么，关于处于叠加态的状态呢？嗯，这很简单，因为在 Qiskit 中，你可以简单地通过振幅乘以基态并将它们相加来创建叠加态。例如，以下指令

[PRE10]

创建状态 ![\left. 1\slash\sqrt{2}\left| {000} \right\rangle + 1\slash\sqrt{2}\left| {111} \right\rangle \right.](img/file391.png "\left. 1\slash\sqrt{2}\left| {000} \right\rangle + 1\slash\sqrt{2}\left| {111} \right\rangle \right.").

重要提示

可能看起来我们在之前的代码中包含了一些不必要的括号。然而，如果你去掉它们，你将不会得到预期的结果。Qiskit 将 `^` 运算符重载为张量积运算。但在 Python 中，`^` 的优先级低于 `+`，因此我们需要括号来确保按照期望的顺序执行操作。

要了解更多…

设置量子状态值的另一种间接方法是创建一个准备该状态的量子电路，并运行它以获得状态向量，就像我们在 *第* *[*2*](ch009.xhtml#x1-400002)，*量子计算的工具* *第* *[*2*](ch009.xhtml#x1-400002) *章* 中学习的那样；或者你也可以直接将量子电路传递给 `Statevector` 构造函数。例如，要创建基态，你只需要一个在需要设置为 ![1](img/file13.png "1") 的量子比特上具有 ![X](img/file9.png "X") 门电路的电路。如果你使用这种方法，然而，你需要小心记住在 Qiskit 电路中，量子比特 ![0](img/file12.png "0") 被表示为矢量中的最右边。因此，如果你有一个名为 `qc` 的三量子比特 `QuantumCircuit`，并且你使用 `qc``.``x` `(0)`，你将获得 ![\left| {001} \right\rangle](img/file392.png "\left| {001} \right\rangle")!*

*为了计算期望值，量子态是不够的。我们还需要创建哈密顿量。现在，我们将学习如何处理![Z](img/file8.png "Z")门张量积，就像我们在上一节中使用的那样，从可以存储在Qiskit `Pauli`对象中的简单门开始。Qiskit提供了几种初始化它们的方法，就像`Statevector`对象的情况一样。第一种方法是使用字符串来指定乘积中对![Z](img/file8.png "Z")和![I](img/file53.png "I")矩阵的位置。例如，如果我们正在处理三个量子比特，并且我们想要创建![Z_{0}Z_{1}](img/file352.png "Z_{0}Z_{1}")（您可能还记得，这是张量积![Z \otimes Z \otimes I](img/file348.png "Z \otimes Z \otimes I")），我们可以使用以下指令：

[PRE11]

它们给出了以下输出：

[PRE12]

表示![Z_{0}Z_{1}](img/file352.png "Z_{0}Z_{1}")的矩阵大小为![8 \times 8](img/file393.png "8 \times 8")，如您所见，它可能难以阅读。幸运的是，我们可以利用对角矩阵的张量积总是对角的事实，并使用以下指令仅打印非零系数：

[PRE13]

它们将给出：

[PRE14]

要了解更多...

当构建`Pauli`对象时，我们还可以指定张量积中哪些位置是![Z](img/file8.png "Z")矩阵，通过传递一个包含![Z](img/file8.png "Z")（表示存在）和零（表示不存在![Z](img/file8.png "Z")或等价地，存在![I](img/file53.png "I")）的向量。由于构造方法更通用，它可以用来创建其他张量积，因此我们需要指定另一个包含![X](img/file9.png "X")矩阵位置的向量，我们暂时将其设置为全零。

例如，你可以运行类似`Z0Z1` `=` `Pauli``(([0,1,1],[0,0,0]))`的命令以获得![Z \otimes Z \otimes I](img/file348.png "Z \otimes Z \otimes I")。请注意，由于Qiskit中量子比特编号的约定，我们需要使用`[0,1,1]`作为![Z](img/file8.png "Z")位置向量的值，而不是`[1,1,0]`。

使用`Pauli`对象工作的主要缺点是，你不能将它们相加或乘以标量。为了得到类似![Z_{0}Z_{1} + Z_{1}Z_{2}](img/file394.png "Z_{0}Z_{1} + Z_{1}Z_{2}")的东西，我们首先需要将`Pauli`对象转换为`PauliOp`，然后我们可以像以下代码所示那样将它们相加：

[PRE15]

在这种情况下，输出如下：

[PRE16]

由于对角矩阵之和是对角矩阵，我们已使用稀疏表示法来更紧凑地显示`H_cut`的非零项。请注意，即使某些对角项为零，因为![Z_{0}Z_{1}](img/file352.png "Z_{0}Z_{1}")的一些元素与![Z_{0}Z_{2}](img/file395.png "Z_{0}Z_{2}")的一些元素相抵消。

获取相同哈密顿量的更紧凑方式是：

[PRE17]

这将评估为：

[PRE18]

注意，我们使用了`^`来计算张量积，并使用括号来正确设置操作优先级。

当然，可以构建更复杂的哈密顿量，甚至包括系数。

例如，

[PRE19]

定义哈密顿量![\left. - 1\slash 2Z_{0}Z_{1} + 2Z_{0}Z_{2} - Z_{1}Z_{2} + Z_{1} - 5Z_{2} \right.](img/file396.png "\left. - 1\slash 2Z_{0}Z_{1} + 2Z_{0}Z_{2} - Z_{1}Z_{2} + Z_{1} - 5Z_{2} \right.").

现在，我们已经准备好计算期望值。多亏了我们迄今为止编写的和执行的代码，`psi`存储![\left| {100} \right\rangle](img/file387.png "\left| {100} \right\rangle")，而`H_cut`存储![Z_{0}Z_{1} + Z_{1}Z_{2}](img/file394.png "Z_{0}Z_{1} + Z_{1}Z_{2}")。然后，计算![\left\langle {100} \right|\left( {Z_{0}Z_{1} + Z_{1}Z_{2}} \right)\left| {100} \right\rangle](img/file397.png "\left\langle {100} \right|\left( {Z_{0}Z_{1} + Z_{1}Z_{2}} \right)\left| {100} \right\rangle")就像运行以下指令一样简单：

[PRE20]

这将给出以下输出：

[PRE21]

由于![Z_{0}Z_{1} + Z_{0}Z_{2}](img/file398.png "Z_{0}Z_{1} + Z_{0}Z_{2}")是图*图*[*3.5*](#Figure3.5)的最大切割问题的哈密顿量，这表明由![\left| {100} \right\rangle](img/file387.png "\left| {100} \right\rangle")（一个集合中的顶点![0](img/file12.png "0")和另一个集合中的![1](img/file13.png "1")和![2](img/file302.png "2")）表示的分配切断了图的两个边，因此是一个最优解。注意输出是如何表示为复数的，因为内积通常可以有虚部。然而，这些期望值始终是实数，与虚数单位（在Python中表示为`j`）相关的系数将只是![0](img/file12.png "0")。

*练习3.4

编写代码以计算图*图*[*3.5*](#Figure3.5)中所有可能切割的期望值。有多少个最优解？

*如果您想逐步评估如![\left\langle \psi \right|H_{\text{cut}}\left| \psi \right\rangle](img/file399.png "\left\langle \psi \right|H_{\text{cut}}\left| \psi \right\rangle")之类的表达式，您也可以使用Qiskit首先计算![H_{\text{cut}}\left| \psi \right\rangle](img/file400.png "H_{\text{cut}}\left| \psi \right\rangle")，然后计算![\left| \psi \right\rangle](img/file43.png "\left| \psi \right\rangle")与该向量的内积。这可以通过以下指令实现：

[PRE22]

在这里，使用`evolve`方法来计算矩阵-向量乘法，而`inner`显然用于内积。

重要提示

我们必须强调，所有这些操作都是数值操作，而不是我们可以在实际量子计算机上运行的操作。事实上，正如您已经知道的，在真实设备上，我们无法访问完整的态矢量：这是我们只能在模拟器上运行电路时才能做到的事情。无论如何，我们知道态矢量的大小会随着量子比特数量的指数增长，因此在许多情况下，模拟可能变得不可行。但别担心。在第 *5* 章 *QAOA：量子近似优化算法* 中，我们将学习如何使用量子计算机来估计![Z](img/file8.png "Z")矩阵张量的期望值。在第 *7* 章 *VQE：变分量子本征值求解器* 中，我们将对更一般的张量产品做同样的处理。实际上，我们将使用的程序将阐明为什么我们称这些量为 *期望值*！**

**但关于张量积和期望值就先到这里。相反，在下一节中，我们将介绍一种新的形式，这将使我们能够比使用伊辛模型更自然地表述一些优化问题。**

# 3.3 从伊辛模型到QUBO模型及其反向转换

考虑以下问题。假设你被给出一组整数![S](img/file73.png "S")和一个目标整数值![T](img/file74.png "T")，并要求你判断是否存在![S](img/file73.png "S")的任何子集，其和为![T](img/file74.png "T")。例如，如果![S = \{ 1,3,4,7, - 4\}](img/file401.png "S = \{ 1,3,4,7, - 4\}")且![T = 6](img/file402.png "T = 6")，那么答案是肯定的，因为![3 + 7 - 4 = 6](img/file403.png "3 + 7 - 4 = 6")。然而，如果![S = \{ 2, - 2,4,8, - 12\}](img/file404.png "S = \{ 2, - 2,4,8, - 12\}")且![T = 1](img/file405.png "T = 1")，答案是否定的，因为集合中的所有数字都是偶数，它们无法相加得到一个奇数。

这个被称为 **子集和** 的问题已知是 ![NP](img/file2.png "NP")**-完全** 的（例如，参见 Sipser 的书中的 *第7.5节* [[90](ch030.xhtml#Xsipser2012introduction)] 以获取证明）。结果证明，我们可以 **将** 子集和问题简化为寻找伊辛模型的最小能量自旋配置（这是一个 ![NP](img/file2.png "NP")-难问题 –参见 *第3.1.3节*），这意味着我们可以将任何子集和实例重写为伊辛基态问题（检查 *附录* * [*C*](ch026.xhtml#x1-233000C)，*计算复杂性*，以复习 **简化**）。**

**然而，如何做到这一点可能并不直接明显。

实际上，通过使用取 ![1](img/file13.png "1") 或 ![- 1](img/file312.png "- 1") 值的变量而不是二进制变量来提出子集和问题作为最小化问题要简单得多。确实，如果我们给定 ![S = \{ a_{0},\ldots,a_{m}\}](img/file406.png "S = \{ a_{0},\ldots,a_{m}\}") 和一个整数 ![T](img/file74.png "T")，我们可以定义二进制变量 ![x_{j}](img/file407.png "x_{j}"), ![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m")，并考虑

![c(x_{0},x_{1},\ldots,x_{m}) = \left( {a_{0}x_{0} + a_{1}x_{1} + \ldots + a_{m}x_{m} - T} \right)^{2}.](img/file409.png "c(x_{0},x_{1},\ldots,x_{m}) = \left( {a_{0}x_{0} + a_{1}x_{1} + \ldots + a_{m}x_{m} - T} \right)^{2}.")

显然，如果我们可以找到满足 ![c(x_{0},x_{1},\ldots,x_{m}) = 0](img/file410.png "c(x_{0},x_{1},\ldots,x_{m}) = 0") 的二进制值 ![x_{j}](img/file407.png "x_{j}"), ![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m")，则子集和问题有正解。在这种情况下，等于 ![1](img/file13.png "1") 的变量 ![x_{j}](img/file407.png "x_{j}") 将指示哪些数字被选中进行求和。但是 ![c(x_{0},x_{1},\ldots,x_{m})](img/file411.png "c(x_{0},x_{1},\ldots,x_{m})") 总是非负的，因此我们将子集和问题简化为寻找 ![c(x_{0},x_{1},\ldots,x_{m})](img/file411.png "c(x_{0},x_{1},\ldots,x_{m})") 的最小值：如果最小值为 ![0](img/file12.png "0")，则子集和有正解；否则，没有。

例如，对于之前考虑的 ![S = \{ 1,4, - 2\}](img/file412.png "S = \{ 1,4, - 2\}") 和 ![T = 2](img/file413.png "T = 2") 的情况，问题将是

![\begin{array}{rlrl} {\text{最小化~}\quad} & {x_{0}^{2} + 8x_{0}x_{1} - 4x_{0}x_{2} - 4x_{0} + 16x_{1}^{2} - 16x_{1}x_{2} - 16x_{1} + 4x_{2}^{2} + 8x_{2} + 4\qquad} & & \qquad \\ {\text{约束~}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,\ldots,m\qquad} & & \qquad \\ \end{array}](img/file414.png "\begin{array}{rlrl} {\text{最小化~}\quad} & {x_{0}^{2} + 8x_{0}x_{1} - 4x_{0}x_{2} - 4x_{0} + 16x_{1}^{2} - 16x_{1}x_{2} - 16x_{1} + 4x_{2}^{2} + 8x_{2} + 4\qquad} & & \qquad \\ {\text{约束~}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,\ldots,m\qquad} & & \qquad \\ \end{array}")

在这里，我们将 ![{(x_{0} + 4x_{1} - 2x_{2} - 2)}^{2}](img/file415.png "{(x_{0} + 4x_{1} - 2x_{2} - 2)}^{2}") 展开以获得要优化的表达式。如果您愿意，可以通过考虑二进制变量总是满足 ![x_{j}^{2} = x_{j}](img/file416.png "x_{j}^{2} = x_{j}") 来稍微简化它。无论如何，![x_{0} = 0,x_{1} = x_{2} = 1](img/file417.png "x_{0} = 0,x_{1} = x_{2} = 1") 将是此问题的最优解。

注意，在这些所有情况下，我们需要最小化的函数 ![c(x_{0},x_{1},\ldots,x_{m})](img/file411.png "c(x_{0},x_{1},\ldots,x_{m})") 是一个关于二元变量 ![x_{j}](img/file407.png "x_{j}") 的 ![2](img/file302.png "2") 次多项式。因此，我们可以推广这种设置，并定义 **二次无约束** **二元优化** （**QUBO**）问题，其形式如下

![\begin{array}{rlrl} {\text{Minimize~}\quad} & {q(x_{0},\ldots,x_{m})\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,\ldots,m\qquad} & & \qquad \\ \end{array}](img/file418.png "\begin{array}{rlrl} {\text{Minimize~}\quad} & {q(x_{0},\ldots,x_{m})\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,\ldots,m\qquad} & & \qquad \\ \end{array}")

其中 ![q(x_{0},\ldots,x_{m})](img/file419.png "q(x_{0},\ldots,x_{m})") 是关于 ![x_{j}](img/file407.png "x_{j}") 变量的二次多项式。为什么这些问题被称为 QUBO 应该现在很清楚了：我们在没有限制的情况下（因为零和一的任何组合都是可接受的）最小化二次表达式。

从先前对子集和问题的简化中可以得出，QUBO 问题属于 ![NP](img/file2.png "NP")-hard。事实上，QUBO 模型非常灵活，它使我们能够以自然的方式表述许多优化问题。例如，将任何伊辛最小化问题重新表述为 QUBO 实例相当容易。如果你需要最小化

![- \sum\limits_{j,k}J_{jk}z_{j}z_{k} - \sum\limits_{j}h_{j}z_{j}](img/file341.png "- \sum\limits_{j,k}J_{jk}z_{j}z_{k} - \sum\limits_{j}h_{j}z_{j}")

某些变量 ![z_{j}](img/file339.png "z_{j}"), ![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m")，取值 ![1](img/file13.png "1") 或 ![- 1](img/file312.png "- 1")，你可以定义新的变量 ![\left. x_{j} = (1 - z_{j})\slash 2 \right.](img/file420.png "\left. x_{j} = (1 - z_{j})\slash 2 \right.")。显然，当 ![z_{j}](img/file339.png "z_{j}") 为 ![1](img/file13.png "1") 时，![x_{j}](img/file407.png "x_{j}") 将为 ![0](img/file12.png "0")，而当 ![z_{j}](img/file339.png "z_{j}") 为 ![- 1](img/file312.png "- 1") 时，![x_{j}](img/file407.png "x_{j}") 将为 ![1](img/file13.png "1")。此外，如果你进行替换 ![z_{j} = 1 - 2x_{j}](img/file421.png "z_{j} = 1 - 2x_{j}"), 你将得到一个关于二元变量 ![x_{j}](img/file407.png "x_{j}") 的二次多项式，其值正好与原始伊辛模型的能量函数相同。如果你对变量 ![x_{j}](img/file407.png "x_{j}") 的多项式进行最小化，你就可以恢复出 ![z_{j}](img/file339.png "z_{j}") 的自旋值，这些值实现了最小能量。

如果您有所疑问，是的，您也可以使用替换公式 ![z_{j} = 2x_{j} - 1](img/file422.png "z_{j} = 2x_{j} - 1") 将伊辛问题转换为QUBO形式。在这种情况下，![z_{j}](img/file339.png "z_{j}") 的值为 ![- 1](img/file312.png "- 1") 将转换为 ![x_{j}](img/file407.png "x_{j}") 的值为 ![0](img/file12.png "0")，而 ![z_{j}](img/file339.png "z_{j}") 的值为 ![1](img/file13.png "1") 将转换为 ![x_{j}](img/file407.png "x_{j}") 的值为 ![1](img/file13.png "1")。然而，我们将坚持使用转换公式 ![z_{j} = 1 - 2x_{j}](img/file421.png "z_{j} = 1 - 2x_{j}") 在本书的其余部分。

例如，如果伊辛能量由 ![\left. ( - 1\slash 2)z_{0}z_{1} + z_{2} \right.](img/file423.png "\left. ( - 1\slash 2)z_{0}z_{1} + z_{2} \right.") 给出，那么，在转换公式 ![z_{j} = 1 - 2x_{j}](img/file421.png "z_{j} = 1 - 2x_{j}") 下，相应的QUBO问题将是以下：

![\begin{array}{rlrl} {\text{Minimize~}\quad} & {- 2x_{0}x_{1} + x_{0} + x_{1} - 2x_{2} + \frac{1}{2}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}](img/file424.png "\begin{array}{rlrl} {\text{Minimize~}\quad} & {- 2x_{0}x_{1} + x_{0} + x_{1} - 2x_{2} + \frac{1}{2}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}")

您也可以通过使用替换公式 ![\left. x_{j} = (1 - z_{j})\slash 2 \right.](img/file420.png "\left. x_{j} = (1 - z_{j})\slash 2 \right.") 从QUBO问题转换到伊辛模型实例。然而，您需要注意一些细节。让我们用一个例子来说明。假设您的QUBO问题是要求最小化 ![x_{0}^{2} + 2x_{0}x_{1} - 3](img/file425.png "x_{0}^{2} + 2x_{0}x_{1} - 3")。那么，当您替换 ![x_{j}](img/file407.png "x_{j}") 变量时，您将得到

![\frac{z_{0}^{2}}{4} + \frac{z_{0}z_{1}}{2} - z_{0} - \frac{z_{1}}{2} - \frac{9}{4}.](img/file426.png "\frac{z_{0}^{2}}{4} + \frac{z_{0}z_{1}}{2} - z_{0} - \frac{z_{1}}{2} - \frac{9}{4}.")

但是，伊辛模型不允许平方变量或独立项！虽然解决这些问题并不困难。关于平方变量，我们可以简单地注意到，总是有 ![z_{j}^{2} = 1](img/file427.png "z_{j}^{2} = 1")，因为 ![z_{j}](img/file339.png "z_{j}") 要么是 ![1](img/file13.png "1")，要么是 ![- 1](img/file312.png "- 1")。因此，我们将每个平方变量替换为常数 ![1](img/file13.png "1")。在我们的情况下，我们会得到

![\frac{1}{4} + \frac{z_{0}z_{1}}{2} - z_{0} - \frac{z_{1}}{2} - \frac{9}{4} = \frac{z_{0}z_{1}}{2} - z_{0} - \frac{z_{1}}{2} - 2.](img/file428.png "\frac{1}{4} + \frac{z_{0}z_{1}}{2} - z_{0} - \frac{z_{1}}{2} - \frac{9}{4} = \frac{z_{0}z_{1}}{2} - z_{0} - \frac{z_{1}}{2} - 2.")

然后，我们可以简单地去掉独立项，因为我们处理的是一个最小化问题，它不会影响最优变量的选择（然而，当你想要恢复最小化函数的原始值时，你应该把它加回来）。在前面的例子中，等价的伊辛最小化问题将是以下内容：

![\begin{array}{rlrl} {\text{Minimize~}\quad} & {\frac{z_{0}z_{1}}{2} - z_{0} - \frac{z_{1}}{2}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ 1, - 1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}](img/file429.png "\begin{array}{rlrl} {\text{Minimize~}\quad} & {\frac{z_{0}z_{1}}{2} - z_{0} - \frac{z_{1}}{2}\qquad} & & \qquad \\ {\text{subject~to~}\quad} & {z_{j} \in \{ 1, - 1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}")

容易验证这个问题有两个最优解：![z_{0} = z_{1} = 1](img/file430.png "z_{0} = z_{1} = 1") 和 ![z_{0} = 1,z_{1} = - 1](img/file431.png "z_{0} = 1,z_{1} = - 1")，两者都达到了 ![- 1](img/file312.png "- 1") 的值。如果我们把之前丢弃的价值独立项 ![- 2](img/file333.png "- 2") 加回来，我们就能在 QUBO 问题中获得最优成本 ![- 3](img/file432.png "- 3")。这些解分别对应于 ![x_{0} = x_{1} = 0](img/file433.png "x_{0} = x_{1} = 0") 和 ![x_{0} = 0,x_{1} = 1](img/file434.png "x_{0} = 0,x_{1} = 1")，确实评估为 ![- 3](img/file432.png "- 3")，并且对于原始问题是最佳的。

练习 3.5

将子集和问题 ![S = \{ 1, - 2,3, - 4\}](img/file435.png "S = \{ 1, - 2,3, - 4\}") 和 ![T = 0](img/file436.png "T = 0") 写成 QUBO 问题，并将其转换为伊辛模型的实例。

因此，我们现在知道了如何从 QUBO 问题到伊辛能量最小化问题，然后再返回，我们可以使用任一形式——在任何给定时刻，哪个更方便就用哪个。实际上，正如我们将在 **第 4 章**[*4*](ch012.xhtml#x1-750004) 和 **第 5 章**[*5*](ch013.xhtml#x1-940005) 中学到的那样，伊辛模型是解决量子计算机组合优化问题的首选公式。此外，我们将使用的软件工具（Qiskit 和 D-Wave 的 Ocean）将帮助我们通过使用本节中描述的变换，将我们的 QUBO 问题重写为伊辛形式。

**我们现在拥有了所有需要的数学工具，如果我们想用量子计算机解决组合优化问题，我们可以玩转我们的新玩具，并用它们来编写一些重要的 QUBO 形式的问题。**

# 3.4 使用 QUBO 模型的组合优化问题

在本章的最后一节，我们将介绍一些技术，这些技术将使我们能够将许多重要的优化问题写成QUBO和Ising实例，这样我们就可以稍后用不同的量子算法来解决它们。这些例子还将帮助您了解如何在这些模型下制定自己的优化问题，这是使用量子计算机解决它们的第一步。

## 3.4.1 二进制线性规划

**二进制线性规划**问题涉及在满足线性约束的二元变量上优化线性函数。因此，其一般形式为

![\begin{array}{rlrl} {\text{Minimize}\quad} & {c_{0}x_{0} + c_{1}x_{1} + \ldots + c_{m}x_{m}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {Ax \leq b,\qquad} & & \qquad \\ & {x_{j} \in \{ 0,1\},\qquad j = 0,\ldots,m,\qquad} & & \qquad \\ \end{array}](img/file437.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {c_{0}x_{0} + c_{1}x_{1} + \ldots + c_{m}x_{m}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {Ax \leq b,\qquad} & & \qquad \\  & {x_{j} \in \{ 0,1\},\qquad j = 0,\ldots,m,\qquad} & & \qquad \\ \end{array}")

其中![c_{j}](img/file438.png "c_{j}")是整数系数，![A](img/file183.png "A")是整数矩阵，![x](img/file269.png "x")是![ (x_{0},\ldots,x_{m})](img/file439.png "(x_{0},\ldots,x_{m})")的转置，而![b](img/file17.png "b")是整数列向量。

这种类型问题的例子可以是以下内容：

![\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} + 3x_{1} - 2x_{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{0} + x_{2} \leq 1,\qquad} & & \qquad \\ & {3x_{0} - x_{1} + 3x_{2} \leq 4\qquad} & & \qquad \\ & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}](img/file440.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} + 3x_{1} - 2x_{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{0} + x_{2} \leq 1,\qquad} & & \qquad \\  & {3x_{0} - x_{1} + 3x_{2} \leq 4\qquad} & & \qquad \\  & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}")

二进制线性规划（也称为**零一线性规划**）是![NP](img/file2.png "NP")-难问题。事实上，决策版本的目标是确定是否存在任何零和一的分配可以满足线性约束（不进行实际优化）是理查德·M·卡普最初发表的21个![NP](img/file2.png "NP")-完全问题之一，发表在他的著名论文《可约性》[[56](ch030.xhtml#Xkarp1972reducibility)]中。满足约束的分配被称为**可行**。

要将二进制线性规划写成QUBO形式，我们需要进行一些转换。第一个转换是将不等式约束转换为等式约束，通过添加**松弛变量**来实现。这可以通过一个例子来更好地理解。在先前的问题中，我们有两个约束：![x_{0} + x_{2} \leq 1](img/file441.png "x_{0} + x_{2} \leq 1") 和 ![3x_{0} - x_{1} + 3x_{2} \leq 4](img/file442.png "3x_{0} - x_{1} + 3x_{2} \leq 4")。在第一个约束中，左边表达式的最小值是 ![0](img/file12.png "0")，当 ![x_{0}](img/file443.png "x_{0}") 和 ![x_{2}](img/file444.png "x_{2}") 都为0时达到。因此，如果我们向那个左边表达式添加一个新的二进制松弛变量 ![y_{0}](img/file445.png "y_{0}")，并用 ![\leq](img/file446.png "\leq") 替换为 ![=](img/file447.png "=")，我们就有

![x_{0} + x_{2} + y_{0} = 1,](img/file448.png "x_{0} + x_{2} + y_{0} = 1,")

这只有在 ![x_{0} + x_{2} \leq 1](img/file441.png "x_{0} + x_{2} \leq 1") 可以满足的情况下才能成立。实际上，如果 ![x_{0} = x_{2} = 0](img/file449.png "x_{0} = x_{2} = 0")，那么我们可以取 ![y_{0} = 1](img/file450.png "y_{0} = 1")；如果 ![x_{0} = 0](img/file451.png "x_{0} = 0") 和 ![x_{2} = 1](img/file452.png "x_{2} = 1")，或者 ![x_{0} = 1](img/file453.png "x_{0} = 1") 和 ![x_{2} = 0](img/file454.png "x_{2} = 0")，我们可以取 ![y_{0} = 0](img/file455.png "y_{0} = 0")。如果 ![x_{0} = x_{2} = 1](img/file456.png "x_{0} = x_{2} = 1")，则无法满足约束。这就是为什么我们可以用 ![x_{0} + x_{2} + y_{0} = 1](img/file457.png "x_{0} + x_{2} + y_{0} = 1") 替换 ![x_{0} + x_{2} \leq 1](img/file441.png "x_{0} + x_{2} \leq 1")，而不改变可行解的集合。

同样地，![3x_{0} - x_{1} + 3x_{2}](img/file458.png "3x_{0} - x_{1} + 3x_{2}") 的最小值是 ![- 1](img/file312.png "- 1")，并且当 ![x_{0} = 0](img/file451.png "x_{0} = 0")，![x_{1} = 1](img/file459.png "x_{1} = 1")，和 ![x_{2} = 0](img/file454.png "x_{2} = 0") 时达到。

重要提示

注意，在某个约束条件下最小化二进制变量上的这些线性表达式的通用规则是将具有正系数的变量设置为 ![0](img/file12.png "0")，将具有负系数的变量设置为 ![1](img/file13.png "1")。

然后，为了使 ![3x_{0} - x_{1} + 3x_{2}](img/file458.png "3x_{0} - x_{1} + 3x_{2}") 达到 ![4](img/file143.png "4")，这是约束的右边，我们可能需要添加一个高达 ![5](img/file296.png "5") 的数。但是，为了表示高达 ![5](img/file296.png "5") 的非负数，我们只需要三个比特位，因此我们可以添加三个新的二进制变量，![y_{1}](img/file460.png "y_{1}"), ![y_{2}](img/file461.png "y_{2}"), 和 ![y_{3}](img/file462.png "y_{3}")，并考虑 ![3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 4y_{3} = 4,](img/file463.png "3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 4y_{3} = 4,")

这可以通过满足![3x_{0} - x_{1} + 3x_{2} \leq 4](img/file442.png "3x_{0} - x_{1} + 3x_{2} \leq 4")来实现。

要了解更多信息…

实际上，请注意![y_{1} + 2y_{2} + 4y_{3}](img/file464.png "y_{1} + 2y_{2} + 4y_{3}")可能增加到![7](img/file465.png "7")，但我们只需要增加到![5](img/file296.png "5")。因此，我们也可以使用

![3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 2y_{3} = 4](img/file466.png "3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 2y_{3} = 4")

作为![3x_{0} - x_{1} + 3x_{2} \leq 4](img/file442.png "3x_{0} - x_{1} + 3x_{2} \leq 4")的替代。

将所有这些放在一起，我们的原始问题等价于以下问题：

![\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} + 3x_{1} - 2x_{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{0} + x_{2} + y_{0} = 1,\qquad} & & \qquad \\ & {3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 2y_{3} = 4\qquad} & & \qquad \\ & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2,\qquad} & & \qquad \\ & {y_{j} \in \{ 0,1\},\qquad j = 0,1,2,3.\qquad} & & \qquad \\ \end{array}](img/file467.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} + 3x_{1} - 2x_{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{0} + x_{2} + y_{0} = 1,\qquad} & & \qquad \\  & {3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 2y_{3} = 4\qquad} & & \qquad \\  & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2,\qquad} & & \qquad \\  & {y_{j} \in \{ 0,1\},\qquad j = 0,1,2,3.\qquad} & & \qquad \\ \end{array}")

现在，我们准备将问题写成QUBO实例。我们唯一需要做的是将约束作为**惩罚项**纳入我们试图最小化的表达式中。为此，我们使用一个整数![B](img/file184.png "B")（稍后我们将选择一个具体的值）并考虑以下问题

![\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} + 3x_{1} - 2x_{2} + B{(x_{0} + x_{2} + y_{0} - 1)}^{2}\qquad} & & \qquad \\ & {+ B{(3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 2y_{3} - 4)}^{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2,\qquad} & & \qquad \\ & {y_{j}, \in \{ 0,1\},\qquad j = 0,1,2,3,\qquad} & & \qquad \\ \end{array}](img/file468.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} + 3x_{1} - 2x_{2} + B{(x_{0} + x_{2} + y_{0} - 1)}^{2}\qquad} & & \qquad \\  & {+ B{(3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 2y_{3} - 4)}^{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2,\qquad} & & \qquad \\  & {y_{j}, \in \{ 0,1\},\qquad j = 0,1,2,3,\qquad} & & \qquad \\ \end{array}")

这已经处于QUBO形式。

由于新问题是无约束的，我们需要设置 ![B](img/file184.png "B") 足够大，以至于违反约束不会*带来收益*。如果其中一个原始约束被违反，乘以 ![B](img/file184.png "B") 的项将大于 ![0](img/file12.png "0")。此外，我们在问题的原始公式中想要最小化的表达式是 ![- 5x_{0} + 3x_{1} - 2x_{2}](img/file469.png "- 5x_{0} + 3x_{1} - 2x_{2}"), 它可以达到最小值 ![- 7](img/file470.png "- 7")（当 ![x_{0} = x_{2} = 1](img/file456.png "x_{0} = x_{2} = 1") 和 ![x_{1} = 0](img/file471.png "x_{1} = 0"))，以及最大值 ![3](img/file472.png "3")（当 ![x_{0} = x_{2} = 0](img/file449.png "x_{0} = x_{2} = 0") 和 ![x_{1} = 1](img/file459.png "x_{1} = 1"))）。因此，如果我们选择，例如，![B = 11](img/file473.png "B = 11")，任何违反约束的分配都将得到至少大于 ![4](img/file143.png "4") 的值，并且如果至少有一个可行解（对于这个特定问题来说就是这种情况），它永远不会被选为QUBO问题的最优解。

以这种方式，一个最优解与原始问题最优解相同的QUBO问题是以下这个：

![\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} + 3x_{1} - 2x_{2} + 11{(x_{0} + x_{2} + y_{0} - 1)}^{2}\qquad} & & \qquad \\ & {+ 11{(3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 2y_{3} - 4)}^{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2,\qquad} & & \qquad \\ & {y_{j}, \in \{ 0,1\},\qquad j = 0,1,2,3.\qquad} & & \qquad \\ \end{array}](img/file474.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} + 3x_{1} - 2x_{2} + 11{(x_{0} + x_{2} + y_{0} - 1)}^{2}\qquad} & & \qquad \\  & {+ 11{(3x_{0} - x_{1} + 3x_{2} + y_{1} + 2y_{2} + 2y_{3} - 4)}^{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2,\qquad} & & \qquad \\  & {y_{j}, \in \{ 0,1\},\qquad j = 0,1,2,3.\qquad} & & \qquad \\ \end{array}")

如果你将最小化表达式展开，你将得到一个关于 ![x_{j}](img/file407.png "x_{j}") 变量的二次多项式，这正是我们在QUBO公式中需要的。

要了解更多信息...

**整数线性规划**是二进制线性规划的一种推广，其中使用非负整数变量而不是二进制变量。在某些这类问题的实例中，约束允许我们推断出整数变量是有界的。例如，如果你有约束

![2a_{0} + 3a_{1} \leq 10](img/file475.png "2a_{0} + 3a_{1} \leq 10")

然后，你可以推导出![a_{0} \leq 5](img/file476.png "a_{0} \leq 5")和![a_{1} \leq 3](img/file477.png "a_{1} \leq 3")。由于![a_{0}](img/file478.png "a_{0}")和![a_{1}](img/file479.png "a_{1}")都是非负的，我们可以用与为二进制整数规划引入松弛变量相同的方式，将它们替换为二进制变量的表达式。例如，我们可以将![a_{0}](img/file478.png "a_{0}")替换为![x_{0} + 2x_{1} + 4x_{2}](img/file480.png "x_{0} + 2x_{1} + 4x_{2}")，将![a_{1}](img/file479.png "a_{1}")替换为![x_{3} + 2x_{4}](img/file481.png "x_{3} + 2x_{4}")。这样，整数线性规划问题就转化为一个等价的二进制线性规划问题，进而可以写成QUBO问题。

在本节中我们研究的过程可以应用于将任何二进制线性规划问题转化为QUBO问题。你只需要首先引入松弛变量，然后添加惩罚项来替代原始约束。这非常有用，因为许多重要问题可以直接写成二进制线性规划形式。在下一个小节中，我们将给出一个突出的例子。

## 3.4.2 背包问题

在著名的**背包问题**中，你被给出一个对象列表![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m")，每个对象都有一个重量![w_{j}](img/file482.png "w_{j}")和一个价值![c_{j}](img/file438.png "c_{j}")。你还被给出一个最大重量![W](img/file483.png "W")，目标是找到一组对象，使得总价值最大化，同时不超过允许的最大重量。想象一下，如果你正在去旅行，你想要尽可能多地携带有价值的东西，但又不想背一个太重的背包。

例如，你可以有价值为![c_{0} = 5](img/file484.png "c_{0} = 5")、![c_{1} = 3](img/file485.png "c_{1} = 3")和![c_{2} = 4](img/file486.png "c_{2} = 4")的对象，重量分别为![w_{0} = 3](img/file487.png "w_{0} = 3")、![w_{1} = 1](img/file488.png "w_{1} = 1")和![w_{2} = 1](img/file489.png "w_{2} = 1")。如果最大重量是![4](img/file143.png "4")，那么最优解将是选择对象![0](img/file12.png "0")和![2](img/file302.png "2")，总价值为![9](img/file490.png "9")。然而，如果最大重量是![3](img/file472.png "3")，那么这个解是不可行的。在这种情况下，我们应该选择对象![1](img/file13.png "1")和![2](img/file302.png "2")，以获得总价值为![7](img/file465.png "7")。

虽然乍一看这个问题可能看起来容易解决，但事实是（惊讶，惊讶！）它是![NP](img/file2.png "NP")-难问题。实际上，如果我们考虑一个决策版本的问题，其中我们还给出了一个价值![V](img/file379.png "V")，并询问是否存在一组对象，其价值至少为![V](img/file379.png "V")，同时满足重量约束，那么这个问题是![NP](img/file2.png "NP")-完全的。

要了解更多…

证明背包问题的决策版本是 ![NP](img/file2.png "NP")-完全的是容易的，因为我们已经知道子集和问题（Subset Sum problem）是 ![NP](img/file2.png "NP")-完全的。假设，然后，你被给了一个子集和问题的实例，其集合为 ![S = \{ a_{0},\ldots,a_{m}\}](img/file406.png "S = \{ a_{0},\ldots,a_{m}\}")，目标总和为 ![T](img/file74.png "T")。然后，你可以通过考虑具有值 ![c_{j} = a_{j}](img/file491.png "c_{j} = a_{j}") 和重量 ![w_{j} = a_{j}](img/file492.png "w_{j} = a_{j}") 的对象 ![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m")，最大重量 ![W = T](img/file493.png "W = T") 和最小总价值 ![V = T](img/file494.png "V = T")，将这个问题重新表述为一个背包问题的实例。然后，背包决策问题的解将给出对象的选择 ![j_{0},\ldots,j_{k}](img/file495.png "j_{0},\ldots,j_{k}")，使得由于重量约束 ![a_{j_{0}} + \ldots + a_{j_{k}} \leq W = T](img/file496.png "a_{j_{0}} + \ldots + a_{j_{k}} \leq W = T")，并且由于最小价值条件 ![a_{j_{0}} + \ldots + a_{j_{k}} \geq V = T](img/file497.png "a_{j_{0}} + \ldots + a_{j_{k}} \geq V = T")。显然，这种对象的选择也将是子集和问题的解。

将背包问题写成二进制线性规划是直接的。我们只需要定义二进制变量 ![x_{j}](img/file407.png "x_{j}"), ![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m"), 它们表示我们是否选择对象 ![j](img/file258.png "j")（如果 ![x_{j} = 1](img/file498.png "x_{j} = 1")），或者不选择（如果 ![x_{j} = 0](img/file499.png "x_{j} = 0")）并考虑

![\begin{array}{rlrl} {\text{Minimize}\quad} & {- c_{0}x_{0} - c_{1}x_{1} - \ldots - c_{m}x_{m}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {w_{0}x_{0} + w_{1}x_{1} + \ldots + w_{m}x_{m} \leq W,\qquad} & & \qquad \\ & {x_{j} \in \{ 0,1\},\qquad j = 0,\ldots,m,\qquad} & & \qquad \\ \end{array}](img/file500.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {- c_{0}x_{0} - c_{1}x_{1} - \ldots - c_{m}x_{m}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {w_{0}x_{0} + w_{1}x_{1} + \ldots + w_{m}x_{m} \leq W,\qquad} & & \qquad \\  & {x_{j} \in \{ 0,1\},\qquad j = 0,\ldots,m,\qquad} & & \qquad \\ \end{array}")

其中 ![c_{j}](img/file438.png "c_{j}") 是对象值，![w_{j}](img/file482.png "w_{j}") 是它们的重量，而 ![W](img/file483.png "W") 是背包的最大重量。请注意，由于原始问题要求最大化价值，我们现在最小化负价值，这是完全等价的。

例如，在我们之前考虑的例子中，对象值为 ![5,3](img/file501.png "5,3") 和 ![4](img/file143.png "4")，重量为 ![3,1](img/file502.png "3,1") 和 ![1](img/file13.png "1")，最大重量为 ![3](img/file472.png "3")，问题将是以下：

![\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} - 3x_{1} - 4x_{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {3x_{0} + x_{1} + x_{2} \leq 3,\qquad} & & \qquad \\ & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}](img/file503.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {- 5x_{0} - 3x_{1} - 4x_{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {3x_{0} + x_{1} + x_{2} \leq 3,\qquad} & & \qquad \\  & {x_{j} \in \{ 0,1\},\qquad j = 0,1,2.\qquad} & & \qquad \\ \end{array}")

当然，然后我们可以添加松弛变量并引入惩罚项，就像我们在上一个子节中所做的那样，将程序重新编写为一个QUBO问题。这正是我们需要用我们的量子算法来解决这些问题的！

练习 3.6

考虑具有值 ![3,1,7,7](img/file504.png "3,1,7,7") 和重量 ![2,1,5,4](img/file505.png "2,1,5,4") 的对象。将最大重量为 ![8](img/file506.png "8") 的背包问题作为二进制线性程序编写。

了解更多...

背包问题的变体允许我们选择多个相同的对象放入背包中。在这种情况下，我们应该使用整数变量而不是二进制变量来表示每个对象被选择的次数。然而，请注意，一旦我们知道允许的最大重量，每个整数变量都是有限的。因此，我们可以使用我们在上一个子节末尾解释的技术来用二进制变量替换整数变量。然后，当然，我们可以使用QUBO形式主义重新编写问题。

在我们接下来的优化问题示例中，我们将回到使用图的工作。实际上，在下一个小节中，我们将处理一个非常多彩的问题！

## 3.4.3 图着色

在本节和下一节中，我们将研究一些与图相关但在不同领域有许多应用的问题。第一个是图着色，其中我们被给出一幅图，并被要求以这种方式为每个顶点分配一种颜色，即通过边的顶点（也称为**相邻**顶点）接收不同的颜色。通常，我们被要求使用尽可能少的颜色或使用不超过给定数量的不同颜色来完成这项任务。如果我们可以用 ![k](img/file317.png "k") 种颜色着色一个图，我们说它是![k](img/file317.png "k")**-可着色**的。着色图所需的颜色最小数量称为其**色数**。

在 *图* *[*3.6*](#Figure3.6) 中，我们展示了同一图的三个着色方案。*图* *[*3.6a*](#Figure3.6a) 不是一个有效的着色，因为存在相邻顶点共享相同颜色的情况。*图* *[*3.6b*](#Figure3.6b) 是有效的，但不是最优的，因为我们不需要超过三种颜色来着色这个图，正如 *图* *[*3.6c*](#Figure3.6c) 所证明的那样。

**![(a) 无效着色](img/file507.jpg)

**(a)** 无效着色

![(b) 非最优着色](img/file508.jpg)

**(b)** 非最优着色

![(c) 最优着色](img/file509.jpg)

**(c)** 最优着色

**图3.6**：图的不同的着色

图着色问题可能看起来像是一个儿童游戏。然而，许多非常相关的实际问题都可以写成图着色的实例。例如，想象一下，如果你的公司有几个项目，你需要为每个项目分配监督者，但由于时间重叠或其他限制，有些项目是不兼容的。你可以创建一个图，其中项目是顶点，如果两个项目不兼容，则它们通过边连接。然后，找到图的色数就等同于找到你需要分配的最少项目领导人数。此外，找到一种着色方法将为你提供一种在满足约束条件的情况下分配监督者的方式。

要了解更多信息…

图着色的历史可以追溯到19世纪中叶，它充满了令人惊讶的情节转折。它起源于一个看似简单的问题：找到最少需要多少种颜色才能以这种方式着色地理地图，即任何相邻的两个国家都使用不同的颜色。尽管它的起源**谦逊**，但它甚至演变成了一场关于计算机辅助数学证明有效性的哲学辩论！

在Robin Wilson的《四色足够了》[[98](ch030.xhtml#Xwilson2021four)]中，可以找到对这个漫长而曲折过程的非常有趣的通俗叙述。

判断一个图是否是![2](img/file302.png "2")-可着色的相对简单。确实，请注意，![2](img/file302.png "2")-可着色图的顶点可以根据它们接收到的颜色分配到两个不相交的集合中，并且这些集合中的顶点之间没有边——这就是为什么这些图被称为**二分图**。但这是一个众所周知的事实（最初由König在1936年证明），一个图是二分图当且仅当它没有奇长**环**（参见Diestel的书中第1.6节[[31](ch030.xhtml#Xdiestel2017graph)]）。我们可以通过计算图的邻接矩阵的幂来检查是否存在环（参见Rosen关于离散数学的书中第10.4.7节[[81](ch030.xhtml#Xrosen2019discrete)]）。然而，对于任何![k \geq 3](img/file510.png "k \geq 3")，检查一个图是否是![k](img/file317.png "k")-可着色的都是![NP](img/file2.png "NP")-完全的（参见Garey、Johnson和Stockmeyer的论文[[43](ch030.xhtml#Xgarey1976simplified)]），因此，计算图的色数是![NP](img/file2.png "NP")-困难的。

假设我们有一个顶点为 ![0,\ldots,m](img/file511.png "0,\ldots,m") 的图。为了使用 QUBO 框架确定图是否是 ![k](img/file317.png "k")-可着色的，我们将定义一些二进制变量 ![x_{jl}](img/file512.png "x_{jl}")，其中 ![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m") 和 ![l = 0,\ldots,k - 1](img/file513.png "l = 0,\ldots,k - 1")。如果顶点 ![j](img/file258.png "j") 接收到 ![l](img/file514.png "l")-色（为了简单起见，颜色通常与数字相关联），则变量 ![x_{jl}](img/file512.png "x_{jl}") 将获得值 ![1](img/file13.png "1")，否则为 ![0](img/file12.png "0")。然后，顶点 ![j](img/file258.png "j") 接收到恰好一种颜色的条件可以代数地写成

![\sum\limits_{l = 0}^{k - 1}x_{jl} = 1.](img/file515.png "\sum\limits_{l = 0}^{k - 1}x_{jl} = 1.")

为了使这个条件成立，必须存在 ![l](img/file514.png "l") 使得 ![x_{jl} = 1](img/file516.png "x_{jl} = 1") 并且对于任何 ![h \neq l](img/file518.png "h \neq l")，![x_{jh} = 0](img/file517.png "x_{jh} = 0")，正好符合我们的需求。

另一方面，我们需要施加约束，使得相邻顶点不会被分配相同的颜色。注意，在两个顶点 ![j](img/file258.png "j") 和 ![h](img/file519.png "h") 接收到相同的颜色 ![l](img/file514.png "l") 的情况下，我们会有 ![x_{jl}x_{hl} = 1](img/file520.png "x_{jl}x_{hl} = 1")。因此，对于相邻顶点 ![j](img/file258.png "j") 和 ![h](img/file519.png "h")，我们需要施加

![\sum\limits_{l = 0}^{k - 1}x_{jl}x_{hl} = 0.](img/file521.png "\sum\limits_{l = 0}^{k - 1}x_{jl}x_{hl} = 0.")

我们可以将这些约束作为我们在 QUBO 问题中要最小化的表达式的惩罚项来写

![\begin{array}{rlrl} {\text{Minimize}\quad} & {\sum\limits_{j = 0}^{m}\left( {\sum\limits_{l = 0}^{k - 1}x_{jl} - 1} \right)^{2} + \sum\limits_{(j,h) \in E}\sum\limits_{l = 0}^{k - 1}x_{jl}x_{hl}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{jl} \in \{ 0,1\},\qquad j = 0,\ldots,m,l = 0,\ldots,k - 1,\qquad} & & \qquad \\ \end{array}](img/file522.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {\sum\limits_{j = 0}^{m}\left( {\sum\limits_{l = 0}^{k - 1}x_{jl} - 1} \right)^{2} + \sum\limits_{(j,h) \in E}\sum\limits_{l = 0}^{k - 1}x_{jl}x_{hl}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{jl} \in \{ 0,1\},\qquad j = 0,\ldots,m,l = 0,\ldots,k - 1,\qquad} & & \qquad \\ \end{array}")

其中 ![E](img/file327.png "E") 是图的边集。注意，我们不需要对项 ![{\sum}_{l = 0}^{k - 1}x_{jl}x_{hl}](img/file523.png "{\sum}_{l = 0}^{k - 1}x_{jl}x_{hl}") 进行平方，因为它们总是非负的。如果我们发现问题的最优解是 ![0](img/file12.png "0")，那么图是 ![k](img/file317.png "k")-可着色的。否则，它不是。

练习 3.7

考虑一个有顶点![0,1,2](img/file524.png "0,1,2")和![3](img/file472.png "3")，以及边 ![(0,1)](img/file305.png "(0,1)"), ![(0,2)](img/file306.png "(0,2)"), ![(1,3)](img/file525.png "(1,3)"), 和 ![(2,3)](img/file526.png "(2,3)") 的图。写出检查图是否![2](img/file302.png "2")-可着色的QUBO版本的问题。

在下一个子节中，我们将研究图上的另一个优化问题。你喜欢旅行吗？那么，准备好利用QUBO形式化方法来优化你的旅行计划。

## 3.4.4 旅行商问题

**旅行商问题**（或简称**TSP**）是组合优化中最著名的问题之一。问题的目标非常简单：你需要找到一个路线，通过给定集合中的每个城市一次且仅一次，同时最小化某个全局量（行驶距离、花费时间、总成本等）。

我们可以使用图来数学地表述这个问题。在这个表述中，我们将被给出一个由![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m")表示的城市集合的顶点集，并且对于每对顶点![j](img/file258.png "j")和![l](img/file514.png "l")，我们也会被给出从![j](img/file258.png "j")到![l](img/file514.png "l")的旅行成本![w_{jl}](img/file527.png "w_{jl}")（这个成本在一般情况下不需要与![w_{lj}](img/file528.png "w_{lj}")相同）。然后我们需要在图中找到一个**路径**（即一组边，其中一条边的终点是下一条边的起点）访问每个顶点一次，并且只访问一次，并且使所有边的成本总和最小化。

要了解更多…

如你所猜，TSP是![NP](img/file2.png "NP")-难（有关更多详细信息，请参阅Korte和Vygen的关于组合优化的书籍的第15章 [[61](ch030.xhtml#Xkorte2012combinatorial)]）。事实上，给定一个图、边的成本集合和一个值![C](img/file234.png "C")，判断是否存在一条访问所有城市且成本小于或等于![C](img/file234.png "C")的路径是![NP](img/file2.png "NP")-完全的。

例如，在*图* *[*3.7*](#Figure3.7)中，我们可以看到一个有四个城市的TSP实例。出现在标签边的数字是它们的成本。为了简化，我们假设对于每对顶点，在这个例子中，旅行成本在两个方向上都是相同的。

*![图3.7：旅行商问题的示例](img/file529.jpg)*

**图3.7**：旅行商问题的示例

为了在QUBO框架中表述TSP问题，我们将定义二进制变量![x_{jl}](img/file512.png "x_{jl}")，以指示访问不同顶点的顺序。更具体地说，如果顶点![j](img/file258.png "j")是旅行中的![l](img/file514.png "l")-th个，则![x_{jl}](img/file512.png "x_{jl}")将为![1](img/file13.png "1")，而![x_{jh}](img/file530.png "x_{jh}")将为![0](img/file12.png "0")，对于![h \neq l](img/file518.png "h \neq l")。因此，对于每个顶点![j](img/file258.png "j")，我们需要施加以下约束

![\sum\limits_{l = 0}^{m}x_{jl} = 1,](img/file531.png "\sum\limits_{l = 0}^{m}x_{jl} = 1,")

因为每个顶点都需要恰好访问一次。但我们还需要施加

![\sum\limits_{j = 0}^{m}x_{jl} = 1](img/file532.png "\sum\limits_{j = 0}^{m}x_{jl} = 1")

对于每个位置![l](img/file514.png "l")，因为我们一次只能访问一个城市。

如果满足这两个约束条件，我们将得到一条访问每个顶点一次且仅一次的路径。然而，这还不够。我们还想最小化路径的总成本，因此我们需要一个表达式，用![x_{jl}](img/file512.png "x_{jl}")变量来表示这个成本。请注意，如果顶点![j](img/file258.png "j")和![k](img/file317.png "k")在路径中是连续的，则使用边![j](img/file258.png "j")和![k](img/file317.png "k")。也就是说，如果且仅存在一个![l](img/file514.png "l")，使得![j](img/file258.png "j")在位置![l](img/file514.png "l")被访问，而![k](img/file317.png "k")在位置![l + 1](img/file533.png "l + 1")被访问。在这种情况下，使用该边的成本将由![w_{jk}x_{jl}x_{kl + 1}](img/file534.png "w_{jk}x_{jl}x_{kl + 1}")给出，因为![x_{jl}x_{kl + 1} = 1](img/file535.png "x_{jl}x_{kl + 1} = 1")。但如果![j](img/file258.png "j")和![k](img/file317.png "k")在路径中不是连续的，那么对于每个![l](img/file514.png "l")，![x_{jl}x_{kl + 1} = 0](img/file536.png "x_{jl}x_{kl + 1} = 0")，这也是我们路线中该路径的成本——我们没有使用它，所以不需要为此付费！

因此，旅行的总成本由以下给出

![\sum\limits_{l = 0}^{m - 1}\sum\limits_{j = 0}^{m}\sum\limits_{k = 0}^{m}w_{jk}x_{jl}x_{kl + 1},](img/file537.png "\sum\limits_{l = 0}^{m - 1}\sum\limits_{j = 0}^{m}\sum\limits_{k = 0}^{m}w_{jk}x_{jl}x_{kl + 1},")

其中我们假设对于![j = 0,\ldots,m](img/file408.png "j = 0,\ldots,m")，![w_{jj} = 0](img/file538.png "w_{jj} = 0")——停留在同一个地方不需要付费！

然后，我们可以将约束作为惩罚项纳入最小化函数，并将TSP问题表述为

![\begin{array}{rlrl} {\text{Minimize}\quad} & {\sum\limits_{l = 0}^{m - 1}\sum\limits_{j = 0}^{m}\sum\limits_{k = 0}^{m}w_{jk}x_{jl}x_{kl + 1} + B\left( {\sum\limits_{l = 0}^{m}x_{jl} - 1} \right)^{2} + B\left( {\sum\limits_{j = 0}^{m}x_{jl} - 1} \right)^{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{jl} \in \{ 0,1\},\qquad j,l = 0,\ldots,m,\qquad} & & \qquad \\ \end{array}](img/file539.png "\begin{array}{rlrl} {\text{Minimize}\quad} & {\sum\limits_{l = 0}^{m - 1}\sum\limits_{j = 0}^{m}\sum\limits_{k = 0}^{m}w_{jk}x_{jl}x_{kl + 1} + B\left( {\sum\limits_{l = 0}^{m}x_{jl} - 1} \right)^{2} + B\left( {\sum\limits_{j = 0}^{m}x_{jl} - 1} \right)^{2}\qquad} & & \qquad \\ {\text{subject~to}\quad} & {x_{jl} \in \{ 0,1\},\qquad j,l = 0,\ldots,m,\qquad} & & \qquad \\ \end{array}")

其中 ![B](img/file184.png "B") 被选择，以确保不可行解永远不会达到最优值。例如，如果我们选择

![B = 1 + \sum\limits_{j,k = 0}^{m}w_{jk},](img/file540.png "B = 1 + \sum\limits_{j,k = 0}^{m}w_{jk},")

那些违反约束条件的解决方案将获得一个比任何有效旅行路线成本更大的惩罚，并且不会被选为最优解。

练习 3.8

在 *图* *[*3.7*](#Figure3.7) 中，获取 TSP 问题中路线成本的公式。

*我们已经展示了如何使用 QUBO 公式来构建几个重要的问题。但我们所关注的这些问题绝不是这些技术可以解决的唯一问题。在下一小节中，我们将给出一些寻找更多问题的提示。

## 3.4.5 其他问题和其他公式

在本章中，我们介绍了伊辛模型和 QUBO 模型，并展示了如何使用它们来构建组合优化问题。实际上，在本章的最后部分，我们研究了几个著名的问题，包括二进制线性规划和旅行商问题，并给出了它们的 QUBO 公式。

使用这些框架来构建优化问题的可能性并不局限于我们所工作的例子。其他可以轻松写成 QUBO 和伊辛实例的重要问题包括在图中寻找团、确定逻辑公式是否可满足以及在约束条件下安排工作。使用本章中描述的技术，你现在可以为自己的这些问题和其他问题编写自己的公式。

然而，有一些已经作为 QUBO 实例构建的问题的参考文献是有用的，这些可以即插即用或作为你问题不符合其中任何一个时的灵感来源。这类公式的良好综述是 Lucas 编制的 [[65](ch030.xhtml#Xlucas2014ising)]，它包括了 Karp 的所有 21 个 ![NP](img/file2.png "NP")-完全问题以及更多。

在使用QUBO框架时，你应该始终牢记的一个重要事情是，通常，解决问题的方式不止一种。例如，有时直接将问题表述为二元线性规划，然后使用我们研究过的转换方法来获得QUBO，最终得到问题的伊辛版本，这是很直接的。然而，也可能存在不同的方法，你可能会发现一个更紧凑的表述，例如减少变量的数量或最小化表达式的长度。

近年来，对重要的组合优化问题的替代QUBO表述的比较已成为一个非常活跃的研究领域。保持开放的心态（并关注科学文献）是很好的建议，因为在许多情况下，选择正确的表述可能是获得更好结果的关键因素，尤其是在使用量子计算机解决优化问题时。

要了解更多…

Salehi、Glos和Miszczak最近的一篇论文[[83](ch030.xhtml#Xsalehi2022unconstrained)]探讨了使用QUBO形式化表示TSP及其一些变体，并研究了不同的表述如何影响量子优化算法的性能。

我们接下来的目标将是使用实际的量子设备来解决本章所关注的类型的问题。准备好学习如何使用量子退火器！

# 摘要

本章致力于介绍两种不同的数学框架，即伊辛模型和QUBO形式化，这使得我们能够以我们稍后能够使用量子计算机帮助找到近似解的方式编写组合优化问题。我们从一些简单的例子开始，逐步过渡到一些著名的问题，如图着色和旅行商问题。

为了实现这一点，我们研究了在不同过程中编写量子计算机优化问题的不同技术。例如，我们看到了如何使用松弛变量，以及如何用惩罚项替换约束。我们还学习了如何将整数变量转换为一系列二进制变量。

在本章所涵盖的所有内容之后，你现在已经准备好用可以在量子计算机上运行的优化算法所要求的语言编写自己的问题。本书本部分的其余章节将致力于学习如何实现和运行这些量子优化算法。实际上，在下一章中，我们将解释如何使用一种称为**量子退火器**的量子计算机类型来解决QUBO和伊辛问题。*****************************************
