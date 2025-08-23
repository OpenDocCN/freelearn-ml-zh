# 附录A

复数

![e^(iπ) + 1 = 0](img/file77.png "e^(iπ) + 1 = 0")

— 莱昂哈德·欧拉

复数集是所有形式为![a + bi](img/file1493.png "a + bi")的数的集合，其中![a](img/file16.png "a")和![b](img/file17.png "b")是实数，且![i² = - 1](img/file543.png "i² = - 1")。这可能不是最正式的介绍方式，但对我们来说足够了！

复数的操作方式相当直接。设![a](img/file16.png "a")、![b](img/file17.png "b")、![x](img/file269.png "x")和![y](img/file270.png "y")是一些实数。我们按如下方式加复数

| ![（a + bi）+（x + yi）=（a + b）+（x + y）i.](img/file1494.png "（a + bi）+（x + yi）=（a + b）+（x + y）i.") |
| --- |

关于乘法，我们有

| ![（a + bi）·（x + yi）= ax + ayi + bix + byi² = (ax - by) + (ay + bx)i.](img/file1495.png "（a + bi）·（x + yi）= ax + ayi + bix + byi² = (ax - by) + (ay + bx)i.") |
| --- |

特别地，当![b = 0](img/file1496.png "b = 0")时，我们可以推导出，

| ![a(x + yi) = ax + (ay)i.](img/file1497.png "a(x + yi) = ax + (ay)i.") |
| --- |

给定任何复数![z = a + bi](img/file1498.png "z = a + bi")，它的**实部**，我们记为![Rez](img/file1499.png "Rez")，是![a](img/file16.png "a")，它的**虚部**，我们记为![Imz](img/file1500.png "Imz")，是![b](img/file17.png "b")。此外，任何这样的数![z](img/file81.png "z")都可以在二维平面上表示为一个向量![({Re}z,{Im}z) = (a,b)](img/file1501.png "({Re}z,{Im}z) = (a,b)")。这个向量的长度被称为![z](img/file81.png "z")的**模**，它被计算为

| ![ | z | = √(a² + b²).](img/file1502.png " | z | = √(a² + b²).") |
| --- | --- | --- | --- | --- |

如果![z = a + bi](img/file1498.png "z = a + bi")是一个复数，它的**共轭**是![z* = a - bi](img/file1503.png "z* = a - bi")。用通俗的话说，如果你想得到任何复数的共轭，你只需要改变它的虚部的符号。很容易验证，对于任何复数![z](img/file81.png "z")，

| ![ | z | ² = zz*,](img/file1504.png " | z | ² = zz*,"") |
| --- | --- | --- | --- | --- |

这无意中表明，![zz*](img/file1505.png "zz*")始终是一个非负实数。

涉及复数使用最著名的公式之一是欧拉公式，它表明，对于任何实数![θ](img/file89.png "θ")，

| ![e^(iθ) = cosθ + i sinθ.](img/file1506.png "e^(iθ) = cosθ + i sinθ.") |
| --- |

这个公式可以通过扩展定义它的常规级数中的指数函数来轻松推导。特别是，根据欧拉公式和指数运算的常规性质，对于任何实数![a](img/file16.png "a")和![b](img/file17.png "b")，我们必须有，

| ![e^{(a + ib)} = e^{a}e^{ib} = e^{a}(\cos\theta + i\sin\theta).](img/file1507.png "e^{(a + ib)} = e^{a}e^{ib} = e^{a}(\cos\theta + i\sin\theta).") |
| --- |

只为了结束这个附录，让我们与你分享一些关于我们心爱的复数的有趣趣事：

+   每个具有复系数的次数为 ![n](img/file244.png "n") 的多项式恰好有 ![n](img/file244.png "n") 个根，如果我们考虑重数

+   任何复可微函数 ![\left. C\rightarrow C \right.](img/file1508.png "\left. C\rightarrow C \right.") 都是光滑且解析的

要了解更多…

如果你想了解更多关于复数的信息，我们邀请你阅读我们俩——中间隔了几年的时间——在大学复变函数课程中使用的同一本书：Bak 和 Newman 的 *复变函数学* [[117](ch030.xhtml#Xcomplex-newman)]。
