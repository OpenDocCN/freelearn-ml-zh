# 附录 B

基础线性代数

*代数是慷慨的。她经常给你比她要求的还多。*

— 让-勒罗恩·达朗贝尔

在本章中，我们将提供一个非常广泛的线性代数概述。最重要的是，这是为了复习。如果你想从基础知识学习线性代数，我们建议阅读 Sheldon Axler 的精彩著作[116]。如果你对抽象代数全情投入，我们还可以推荐 Dummit 和 Foote 的杰出著作[115]。把这些事情处理好之后，让我们来做一些代数吧！

当大多数人想到向量时，他们会想到指向某个方向的复杂箭头。但是，当其他人看到箭头时，我们数学家——在我们不懈追求抽象的过程中——看到的是向量空间中的元素。那么什么是向量空间呢？简单！

# B.1 向量空间

设 ![F](img/file1320.png "F") 为实数或复数。一个 ![F](img/file1320.png "F")-向量空间是一个集合 ![V](img/file379.png "V")，以及一个“加法”函数（通常用 ![+](img/file1509.png "+") 表示，原因很明显）和一个“标量乘法”函数（像通常的乘法一样表示）。加法需要取任意两个向量并返回另一个向量，即 ![+](img/file1509.png "+") 需要是一个函数 ![\left. V \times V\rightarrow V \right.](img/right.")。标量乘法，正如其名所示，必须取一个标量（![F](img/file1320.png "F") 的一个元素）和一个向量，并返回一个向量，即它需要是一个函数 ![\left. F \times V\rightarrow V \right.](img/right.")。此外，向量空间必须满足，对于任意 ![α_{1},α_{2} \in F](img/in F") 和 ![v_{1},v_{2},v_{3} \in V](img/in V")，以下性质：

+   加法的结合律：![（v_{1} + v_{2}）+ v_{3} = v_{1} + （v_{2} + v_{3}）](img/file1514.png "(v_{1} + v_{2}) + v_{3} = v_{1} + (v_{2} + v_{3})")

+   加法的交换律：![v_{1} + v_{2} = v_{2} + v_{1}](img/file1515.png "v_{1} + v_{2} = v_{2} + v_{1}")

+   加法的单位元：必须存在一个 ![0 \in V](img/in V")，使得对于每个向量 ![v \in V](img/in V")，有 ![v + 0 = v](img/file1518.png "v + 0 = v")

+   加法的相反数：必须存在一个 ![−v_{1} \in V](img/in V")，使得 ![v_{1} + （−v_{1}）= 0](img/file1520.png "v_{1} + ( - v_{1}) = 0")

+   标量乘法与 ![F](img/file1320.png "F") 中的乘法的兼容性：![（α_{1} · α_{2}）· v_{1} = α_{1} · （α_{2} · v_{1}）](img/alpha_{2}) \cdot v_{1} = \alpha_{1} \cdot (\alpha_{2} \cdot v_{1})")

+   关于向量加法的分配律：![\alpha_{1}(v_{1} + v_{2}) = \alpha_{1}v_{1} + \alpha_{1}v_{2}](img/alpha_{1}(v_{1} + v_{2}) = \alpha_{1}v_{1} + \alpha_{1}v_{2}")

+   关于标量加法的分配律：![(\alpha_{1} + \alpha_{2})v_{1} = \alpha_{1}v_{1} + \alpha_{2}v_{1}](img/alpha_{2})v_{1} = \alpha_{1}v_{1} + \alpha_{2}v_{1}")

+   标量乘法的恒等式：![1 \cdot v_{1} = v_{1}](img/cdot v_{1} = v_{1}")

要了解更多…

如果你，就像我们一样，喜欢抽象，你应该知道向量空间通常是在一个任意的**域**上定义的——而不仅仅是实数或复数！如果你想了解更多，我们建议阅读 Dummit 和 Foote 合著的书籍[115]。

这些是一些向量空间的例子：

+   具有通常加法和乘法的实数集是一个实向量空间。

+   具有复数加法和乘法的复数集是一个复向量空间。此外，可以通过将标量乘法限制为复数与实数的乘法来简单地将其转换为实向量空间。

+   具有通常分量加法和标量乘法（实数）的集合![R^{n}](img/file1212.png "R^{n}")是一个向量空间。如果我们固定![n = 2,3](img/file1525.png "n = 2,3")，那么我们就能找到大家都在谈论的那些花哨的箭头！

+   对我们来说最重要的是，具有分量加法和复数标量乘法的集合![C^{n}](img/file1526.png "C^{n}")是一个向量空间。

+   只举一个可爱的例子，所有在实数闭有限区间上的光滑函数的集合是一个向量空间。你可以尝试自己定义函数的加法和标量乘法。

当我们提到一个集合![V](img/file379.png "V")上的向量空间，其具有![+](img/file1509.png "+")加法和标量乘法![\cdot](img/cdot")时，我们应该将其表示为![(V, + , \cdot )](img/cdot )")，以表明我们正在考虑哪个函数作为加法函数，以及我们正在考虑哪个函数作为标量乘法。然而，说实话，![(V, + , \cdot )](img/cdot )")的写法很麻烦，我们数学家——就像所有人类一样——都有一种自然的懒惰倾向。所以，我们通常只写![V](img/file379.png "V")，并让![+](img/file1509.png "+")和![\cdot](img/cdot")在合理的情况下从上下文中推断出来。

# B.2 基础和坐标

一些 ![F](img/file1320.png "F")-向量空间 ![V](img/file379.png "V") 是**有限维的**：这意味着存在一个有限向量族 ![\{ v_{1},…,v_{n}\} \subseteq V](img/subseteq V")，对于 ![V](img/file379.png "V") 中的任何向量 ![v \in V](img/in V")，存在一些唯一的标量 ![α_{1},…,α_{n} \in F](img/in F")，使得

| ![v = α_{1}v_{1} + ⋯ + α_{n}v_{n}.](img/file1530.png "v = α_{1}v_{1} + ⋯ + α_{n}v_{n}.") |
| --- |

数量 ![α_{1},…,α_{n}](img/file1531.png "α_{1},…,α_{n}") 被称为 ![v](img/file1532.png "v") 关于基 ![\{ v_{1},…,v_{n}\}](img/}") 的**坐标**。自然数 ![n](img/file244.png "n") 被称为向量空间的维度，这是一个生活事实，即向量空间的任何两个基都需要有相同数量的元素，因此维度是良好定义的。如果你想证明（你应该想！），检查你最喜欢的线性代数教科书；我们建议的两个中的任何一个都应该能完成任务。

有限维向量空间的两个例子是 ![R^{n}](img/file1212.png "R^{n}") 和 ![C^{n}](img/file1526.png "C^{n}")（具有自然加法和乘法运算）。例如，![C^{3}](img/file1533.png "C^{3}") 或 ![R^{3}](img/file1214.png "R^{3}") 的一个基将是

| ![\{(1,0,0),(0,1,0),(0,0,1)\}.](img/{(1,0,0),(0,1,0),(0,0,1)\}.") |
| --- |

为了进一步说明这一点，如果我们考虑 ![C^{3}](img/file1533.png "C^{3}") 中的向量 ![（i，3 + 2i，- 2）](img/file1535.png "（i，3 + 2i，- 2）"), 我们将得到

| ![（i，3 + 2i，- 2）= i · （1，0，0）+ （3 + 2i）· （0，1，0）+ （- 2）· （0，0，1），](img/file1536.png "（i，3 + 2i，- 2）= i · （1，0，0）+ （3 + 2i）· （0，1，0）+ （- 2）· （0，0，1），") |
| --- |

并且这种用这些基向量表示的形式显然是唯一的。更重要的是，这个基如此自然和常见，以至于它有一个名字，即**标准基**，其向量通常表示为 ![\{ e_{1},e_{2},e_{3}\}](img/}"). 对于任何 ![n](img/file244.png "n")，可以在 ![R^{n}](img/file1212.png "R^{n}") 和 ![C^{n}](img/file1526.png "C^{n}") 上定义一个类似的基础。

要了解更多...

我们在这本书中广泛使用标准基，但使用不同的符号。我们将其称为**计算基**。

当你在有限维向量空间中有一个向量时，有时使用你选择的某个基的坐标来工作比使用其“原始”表达式更方便。为了做到这一点，我们有时用一个坐标![\alpha_{1},\ldots,\alpha_{n}](img/alpha_{n}")的列矩阵来表示一个向量![v](img/file1532.png "v")。例如，在上一个例子中，向量![(1,3 + 2i, - 2)")])将被表示为坐标列矩阵

| ![\begin{pmatrix} 1 \\ {3 + 2i} \\ {- 2} \\ \end{pmatrix}](img/end{pmatrix}") |
| --- |

相对于规范基![\{ e_{1},e_{2},e_{3}\}](img/}").

重要提示

非常重要的是要记住，向量的坐标列矩阵总是相对于某个基来定义的。

例如，如果我们考虑基![\{ e_{1},e_{3},e_{2}\}](img/}"), 则上述向量的坐标将是

| ![\begin{pmatrix} 1 \\ {- 2} \\ {3 + 2i} \\ \end{pmatrix}.](img/end{pmatrix}.") |
| --- |

并且，是的，顺序很重要。

# B.3 线性映射和特征值

现在我们已经知道了什么是向量空间，很自然地会想知道我们如何定义某些![F](img/file1320.png "F")-向量空间![V](img/file379.png "V")和![W](img/file483.png "W")之间的变换![\left. L:V\rightarrow W \right.](img/right.")。公平地说，你可以随意定义任何这样的变换![L](img/file1012.png "L")——我们不是在这里限制你的数学自由。但是，如果你想![L](img/file1012.png "L")与![V](img/file379.png "V")和![W](img/file483.png "W")的向量空间结构良好地互动，你将希望它是线性的。也就是说，你将希望对于任何向量![v_{1},v_{2} \in V](img/in V")和任何标量![\alpha \in F]，

| ![L(v_{1} + v_{2}) = L(v_{1}) + L(v_{2}),\qquad L(\alpha \cdot v_{1}) = \alpha L(v_{1}).](img/file1545.png "L(v_{1} + v_{2}) = L(v_{1}) + L(v_{2}),\qquad L(\alpha \cdot v_{1}) = \alpha L(v_{1}).") |
| --- |

请记住，这些表达式左侧的加法和标量乘法是![V](img/file379.png "V")的，而右侧的操作是![W](img/file483.png "W")的。

线性映射非常奇妙。它们不仅具有非常美好的性质，而且定义起来也非常简单。如果 ![v_{1},\ldots,v_{n}](img/ldots,v_{n}") 是 ![V](img/file379.png "V") 的一个基，并且你想定义一个线性映射 ![L:V\rightarrow W](img/right.")，你所需要做的就是为每一个 ![L(v_{k})](img/file1547.png "L(v_{k})") 给出一个值——任何值——对于每一个 ![k = 1,\ldots,n](img/ldots,n")。然后，通过线性性质，这个函数可以扩展到 ![V](img/file379.png "V") 的所有元素，如下所示

| ![L(\alpha_{1}v_{1} + \cdots + \alpha_{n}v_{n}) = \alpha_{1}L(v_{1}) + \cdots + \alpha_{n}L(v_{n})](img/alpha_{n}v_{n}) = \alpha_{1}L(v_{1}) + \cdots + \alpha_{n}L(v_{n})") |
| --- |

对于任何标量 ![\alpha_{1},\ldots,\alpha_{n} \in F](img/in F")，都成立。此外，如果我们让 ![\{ w_{1},\ldots,w_{m}\}](img/}") 成为 ![W](img/file483.png "W") 的一个基，并且让 ![a_{k,l} \in F](img/in F") 成为满足条件的唯一标量，即

| ![L(v_{k}) = a_{1k}w_{1} + \cdots + a_{nk}w_{n},](img/file1552.png "L(v_{k}) = a_{1k}w_{1} + \cdots + a_{nk}w_{n},") |
| --- |

那么，对于任何 ![v = \alpha_{1}v_{1} + \cdots + \alpha_{n}v_{n} \in V](img/in V")，相对于 ![\{ w_{1},\ldots,w_{m}\}](img/}") 的 ![L(v)](img/file1553.png "L(v)") 的坐标将是

| ![\begin{pmatrix} a_{11} & \cdots & a_{1n} \\ {\vdots} & \ddots & {\vdots} \\ a_{n1} & \cdots & a_{nn} \\ \end{pmatrix}\begin{pmatrix} \alpha_{1} \\ {\vdots} \\ \alpha_{n} \\ \end{pmatrix}.](img/end{pmatrix}.") |
| --- |

用更简化的术语来说，

| ![\begin{pmatrix} &#124; \\ {L(v)} \\ &#124; \\ \end{pmatrix} = \begin{pmatrix} a_{11} & \cdots & a_{1n} \\ {\vdots} & \ddots & {\vdots} \\ a_{n1} & \cdots & a_{nn} \\ \end{pmatrix}\begin{pmatrix} &#124; \\ v \\ &#124; \\ \end{pmatrix},](img/ {L(v)} \\ &#124; \\ \end{pmatrix} = \begin{pmatrix} a_{11} & \cdots & a_{1n} \\ {\vdots} & \ddots & {\vdots} \\ a_{n1} & \cdots & a_{nn} \\ \end{pmatrix}\begin{pmatrix} &#124; \\ v \\ &#124; \\ \end{pmatrix},") |
| --- |

其中列矩阵表示向量相对于基 ![\{ v_{1},\ldots,v_{n}\}](img/}") 和 ![\{ w_{1},\ldots,w_{m}\}](img/}") 的坐标。我们说矩阵 ![{(a_{kl})}_{kl}](img/file1557.png "{(a_{kl})}_{kl}") 是 ![L](img/file1012.png "L") 相对于这些基的**坐标矩阵**。如果 ![V = W](img/file1558.png "V = W") 并且我们有一个映射 ![\left. L:V\rightarrow V \right.](img/right.")，我们说 ![L](img/file1012.png "L") 是一个**内射自同构**，并且通常我们考虑每个地方都使用相同的基。

在任何向量空间上都可以定义一种非常特殊的内射自同构：**恒等算子**。这只是一个函数 ![\text{id}](img/text{id}")，它将任何向量 ![v](img/file1532.png "v") 映射到 ![\text{id}(v) = v](img/text{id}(v) = v")。如果 ![\left. L:V\rightarrow V \right.](img/right.") 是一个内射自同构，我们说一个函数 ![L^{- 1}](img/file1562.png "L^{- 1}") 是 ![L](img/file1012.png "L") 的**逆**，如果 ![L \circ L^{- 1}](img/circ L^{- 1}") 和 ![L^{- 1} \circ L](img/circ L") 都等于恒等算子——实际上，在有限维向量空间上的内射自同构中，检查这两个条件中的任何一个就已经足够了。具有坐标矩阵 ![A](img/file183.png "A") 的映射的逆的坐标矩阵仅仅是通常的逆矩阵 ![A^{- 1}](img/file1565.png "A^{- 1}")。更重要的是，一个线性映射是可逆的，当且仅当其坐标矩阵也是可逆的。

当你有一个内射自同构 ![\left. L:V\rightarrow V \right.](img/right.") 时，可能存在一些向量 ![0 \neq v \in V](img/in V")，对于这些向量存在一个标量 ![\lambda](img/lambda")，使得 ![L(v) = \lambda v](img/file1568.png "L(v) = \lambda v")。这些向量被称为**特征向量**，相应的值 ![\lambda](img/lambda") 被称为它们的**特征值**。在某些情况下，你将能够找到一个特征向量基 ![v_{1},\ldots,v_{n}](img/ldots,v_{n}")，其中有一些相关的特征向量 ![\lambda_{1},\ldots,\lambda_{n}](img/lambda_{n}")。相对于这个基，![L](img/file1012.png "L") 的坐标矩阵将是一个对角矩阵

| ![\begin{pmatrix} \lambda_{1} & & \\ & \ddots & \\ & & \lambda_{n} \\ \end{pmatrix}.](img/end{pmatrix}.") |
| --- |

# B.4 内积和伴随算子

在一个 ![F](img/file1320.png "F")-向量空间 ![V](img/file379.png "V") 上，我们可能希望定义一个**内积** ![\left\langle - \middle| - \right\rangle](img/rangle"). 这将是一个操作，它接受任何一对向量并返回一个标量，即一个函数 ![\left. V \times V\rightarrow F \right.](img/right."), 对于任何 ![u,v_{1},v_{2} \in V](img/in V"), 和 ![\alpha_{1},\alpha_{2} \in F](img/in F") 满足以下性质：

+   **共轭对称性**: ![\left\langle v_{1} \middle| v_{2} \right\rangle = \left\langle v_{2} \middle| v_{1} \right\rangle^{\ast}](img/ast}"). 当然，如果向量空间定义在 ![R](img/file1575.png "R") 上，那么 ![\left\langle v_{2} \middle| v_{1} \right\rangle^{\ast} = \left\langle v_{2} \middle| v_{1} \right\rangle](img/rangle"), 所以 ![\left\langle v_{1} \middle| v_{2} \right\rangle = \left\langle v_{2} \middle| v_{1} \right\rangle](img/rangle").

+   **线性性**: ![\left\langle u \middle| \alpha_{1}v_{1} + \alpha_{2}v_{2} \right\rangle = \alpha_{1}\left\langle u \middle| v_{1} \right\rangle + \alpha_{2}\left\langle u \middle| v_{2} \right\rangle](img/rangle").

+   **正定性**: 如果 ![u \neq 0](img/neq 0"), ![\left\langle u \middle| u \right\rangle](img/rangle") 是实数且大于 ![0](img/file12.png "0").

很容易验证以下是在 ![C^{n}](img/file1526.png "C^{n}") 上的一个内积：

| ![\left\langle (\alpha_{1},\ldots,\alpha_{n}) \middle&#124; (\beta_{1},\ldots,\beta_{n}) \right\rangle = \alpha_{1}^{\ast}\beta_{1} + \cdots + \alpha_{n}^{\ast}\beta_{n}.](img/alpha_{n}) \middle&#124; (\beta_{1},\ldots,\beta_{n}) \right\rangle = \alpha_{1}^{\ast}\beta_{1} + \cdots + \alpha_{n}^{\ast}\beta_{n}.") |
| --- |

当我们有一个具有内积的向量空间——通常被称为**内积空间**——两个向量 ![v](img/file1532.png "v") 和 ![w](img/file1267.png "w") 被称为正交的，如果 ![\left\langle v \middle| w \right\rangle = 0](img/rangle = 0"). 此外，如果一个基的所有向量都是成对正交的，那么这个基被称为正交基。

在内积的帮助下，我们可以在向量空间上定义一个**范数**。我们不会深入探讨范数的细节，但非常粗略地，我们可以将它们视为测量向量长度的方法（请不要考虑箭头，请不要考虑箭头……）。由标量积![\left\langle \cdot \middle| \cdot \right\rangle](img/rangle")诱导的范数是

| ![\left\| v \right\| = \sqrt{\left\langle v \middle | v \right\rangle}.](img/rangle}.") |
| --- | --- | --- |

我们说一个基是**正交归一**的，如果除了正交之外，它所有向量的范数都等于![1](img/file13.png "1")。

当我们给定一个矩阵![A = (a_{kl})](img/file1585.png "A = (a_{kl})")时，我们定义它的**共轭转置**为![A^{\dagger} = (a_{kl}^{\ast})](img/ast})")，即

| ![\begin{pmatrix} a_{11} & \cdots & a_{1n} \\ {\vdots} & \ddots & {\vdots} \\ a_{n1} & \cdots & a_{nn} \\ \end{pmatrix}^{\dagger} = \begin{pmatrix} a_{11}^{\ast} & \cdots & a_{n1}^{\ast} \\ {\vdots} & \ddots & {\vdots} \\ a_{1n}^{\ast} & \cdots & a_{nn}^{\ast} \\ \end{pmatrix}.](img/end{pmatrix}.") |
| --- |

以下恒等式可以很容易地在方阵和线性映射中验证：

| ![{(A + B)}^{\dagger} = A^{\dagger} + B^{\dagger},\qquad{(AB)}^{\dagger} = B^{\dagger}A^{\dagger}.](img/file1588.png "{(A + B)}^{\dagger} = A^{\dagger} + B^{\dagger},\qquad{(AB)}^{\dagger} = B^{\dagger}A^{\dagger}.") |
| --- |

在这里，![AB](img/file1589.png "AB")表示通常的矩阵乘法。

如果![\left. L:V\rightarrow V \right.](img/right.")是一个有限维向量空间![V](img/file379.png "V")上的自同构，我们可以定义它的**厄米共轭**为唯一一个线性映射![\left. L^{\dagger}:V\rightarrow V \right.](img/right.")，该映射相对于某个基的坐标基是![L](img/file1012.png "L")相对于该基的坐标矩阵的共轭转置。可以证明这个概念是良好定义的，也就是说，无论你选择什么基，你总是得到相同的线性映射。

要了解更多...

我们给出的定义，嗯，并不是最严谨的。通常，当你有一对内积空间![V](img/file379.png "V")和![W](img/file483.png "W")以及内积![\left\langle \cdot \middle| \cdot \right\rangle_{V}](img/rangle_{V}")和![\left\langle \cdot \middle| \cdot \right\rangle_{W}](img/rangle_{W}")时，线性映射![\left. L:V\rightarrow W \right.](img/right.")的伴随定义为唯一的线性映射![\left. L^{\dagger}:W\rightarrow V \right.](img/right.")，使得对于每个![v \in V](img/in V")和![w \in W](img/in W")，

| ![\left\langle w \middle&#124; L(v) \right\rangle_{W} = \left\langle L^{\dagger}(w) \middle&#124; v \right\rangle_{V}.](img/middle&#124; L(v) \right\rangle_{W} = \left\langle L^{\dagger}(w) \middle&#124; v \right\rangle_{V}.") |
| --- |

我们邀请您验证，对于我们所考虑的特殊情况（![V = W](img/file1558.png "V = W")有限维），两种定义是一致的。

我们说一个自同构![L](img/file1012.png "L")是**自伴**或**厄米**的，如果![L = L^{\dagger}](img/dagger}")。这是一个生活的事实（再次，我们鼓励您检查您最喜欢的线性代数教科书），每个厄米算子都有一个实特征值的正交归一基。

此外，我们还称一个自同构![U](img/file51.png "U")是**酉**的，如果![U^{\dagger}U = UU^{\dagger} = I](img/dagger} = I")，其中![I](img/file53.png "I")表示单位矩阵。

# B.5 矩阵指数

每个微积分学生都熟悉指数函数，它被定义为![{\exp}(x) = e^{x}](img/exp}(x) = e^{x}")。如果你深入到数学分析的奇妙之处，你会了解到指数函数实际上被定义为级数的和，即

![{\exp}(x) = \sum\limits_{k = 1}^{\infty}\frac{x^{k}}{k!}.](img/exp}(x) = \sum\limits_{k = 1}^{\infty}\frac{x^{k}}{k!}.")

结果表明，这个定义可以远远超出实数的范围。例如，欧拉公式——我们在*附录**A*中介绍过的，复*数*——是将指数函数的定义扩展到每个![x \in C](img/in C")的结果。

*对我们来说最重要的是，指数函数可以扩展到…矩阵！这样，一个方阵的指数就被定义为，不出所料，

![{\exp}(A) = \sum\limits_{k = 1}^{\infty}\frac{A^{k}}{k!}.](img/exp}(A) = \sum\limits_{k = 1}^{\infty}\frac{A^{k}}{k!}.")

更重要的是，这个定义也适用于自同构。如果一个自同构![L](img/file1012.png "L")的坐标矩阵![A](img/file183.png "A")（相对于特定的基），我们可以定义![L](img/file1012.png "L")的指数为相对于考虑的基具有坐标矩阵![{\exp}(A)](img/exp}(A)")的自同构。可以验证这个概念是良好定义的：无论我们考虑哪个基，我们总是得到相同自同构。

当然，仅仅通过求和无穷级数来计算矩阵的指数可能不是最好的主意。幸运的是，有一个更简单的方法。如果一个矩阵是对角的，可以证明

| ![exp\begin{pmatrix} \lambda_{1} & & \\ & \ddots & \\ & & \lambda_{n} \\ \end{pmatrix} = \begin{pmatrix} e^{\lambda_{1}} & & \\ & \ddots & \\ & & e^{\lambda_{n}} \\ \end{pmatrix}.](img/end{pmatrix}.") |
| --- |

正如我们在上一节中提到的，当一个自同构是厄米算子时，我们总能找到一个基，使得自同构的坐标矩阵是对角线（特征向量基），这使得我们能够计算厄米算子的指数。一般来说，总是可以计算矩阵的指数[115, 第 12.3 节]，但在这里我们不会讨论如何做。

为了结束这个附录，我们将简要地触及一个相对无关的话题，尽管如此，我们仍会在本书的一些部分使用它：模运算。

# B.6 模运算速成课程

如果你的手表显示是 15:00，而我们问你时间，你会说它是 03:00。但你是在撒谎，不是吗？你的手表显示是 15:00，但你刚刚说它是 3:00。你有什么问题？嗯，可能没什么问题。当你告诉我们时间时，你潜意识中是在进行模![12](img/file601.png "12")的算术。

大概来说，当你用![n](img/file244.png "n")进行模数运算时，你所做的一切就是假设![n](img/file244.png "n")和![0](img/file12.png "0")代表同一个数。例如，当你用模![4](img/file143.png "4")进行算术运算时，

| ![0 \equiv 4 \equiv 8 \equiv 12 \equiv 16\;({mod}\; 4),](img/; 4),") |
| --- |
| ![1 \equiv 5 \equiv 9 \equiv 13 \equiv 17\;({mod}\; 4),](img/; 4),") |
| ![2 \equiv 6 \equiv 10 \equiv 14 \equiv 18\;({mod}\; 4),](img/; 4),") |

等等，诸如此类。注意我们是如何写成 ![\equiv](img/equiv") 而不是 ![=](img/file447.png "=") 来表示那些数字本身并不相等，只是它们模 ![4](img/file143.png "4") 相等——这也是为什么我们在右边有那个可爱的 ![(\text{mod~}4)](img/text{mod~}4)") 的原因。

在这个模运算环境中，你可以像平常一样进行加法和乘法运算。例如，当在模 ![4](img/file143.png "4") 下工作时，

| ![2 \times 3 = 6 \equiv 2\;({mod}\; 4).](img/; 4).") |
| --- |

哈哈！看看我们做了什么！现在你可以告诉你的所有朋友，![2](img/file302.png "2") 乘以 ![3](img/file472.png "3") 等于 ![2](img/file302.png "2")（然后你可以悄悄地说“模 ![4](img/file143.png "4")”，仍然技术上正确）。但是，等等，这里是我们最喜欢的：

| ![1 + 1 \equiv 0\;({mod}\; 2).](img/; 2).") |
| --- |

最后，所有那些声称“一加一不一定等于二”的人都有道理，对吧？他们肯定是在谈论模运算。我们毫无疑问。

要了解更多…

你对模运算还不够满意吗？Dummit 和 Foote 有你覆盖！祝你好玩。 [115]*
