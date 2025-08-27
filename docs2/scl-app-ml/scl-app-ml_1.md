# 第一部分。模块 1

> **Scala 数据科学**
> 
> *利用 Scala 的不同工具构建可扩展、健壮的数据科学应用程序*

# 第一章。Scala 和数据科学

20 世纪后半叶是硅的时代。在 50 年里，计算能力从极其稀缺到完全平凡。21 世纪前半叶是互联网的时代。在过去的 20 年里，谷歌、推特和 Facebook 等巨头崛起——这些巨头永远改变了我们看待知识的方式。

互联网是一个庞大的信息枢纽。人类产生的 90%的数据在过去 18 个月内已经生成。能够利用这些数据洪流以获得真正理解的程序员、统计学家和科学家将对商业、政府和慈善机构如何做出决策产生越来越大的影响。

本书旨在介绍一些您将需要的工具，以从数据洪流中综合提炼出真正的洞察力。

# 数据科学

数据科学是从数据中提取有用信息的过程。作为一个学科，它仍然有些模糊不清，定义的数量几乎与专家的数量一样多。而不是再添加另一个定义，我将遵循**德鲁·康威**的描述（[`drewconway.com/zia/2013/3/26/the-data-science-venn-diagram`](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram)）。他描述数据科学为三个正交技能集的融合：

+   数据科学家必须具备**黑客技能**。数据存储和传输通过计算机进行。计算机、编程语言和库是数据科学家的锤子和凿子；他们必须自信且准确地使用它们来塑造数据。这正是 Scala 发挥作用的地方：它是编程工具箱中一个强大的工具。

+   数据科学家必须对**统计学和数值算法**有扎实的理解。优秀的数据科学家将理解机器学习算法的工作原理以及如何解读结果。他们不会被误导的指标、欺骗性的统计数据或误解的因果关系所迷惑。

+   一个优秀的数据科学家必须对**问题领域**有扎实的理解。数据科学过程涉及以科学严谨的方式构建和发现关于问题领域知识。因此，数据科学家必须提出正确的问题，了解以往的结果，并理解数据科学努力如何融入更广泛的商业或研究背景。

德鲁·康威用一张维恩图优雅地总结了这一点，展示了数据科学位于黑客技能、数学和统计学知识以及实质性专业知识交汇处：

![数据科学](img/image01158.jpeg)

当然，人们精通这些领域中的多个领域是罕见的。数据科学家通常在跨职能团队中工作，不同成员为不同领域提供专业知识。然而，为了有效运作，团队中的每个成员都必须对所有三个领域有一个一般的工作知识。

为了更具体地概述数据科学项目中的工作流程，让我们设想我们正在尝试编写一个分析公众对政治运动看法的应用程序。数据科学管道可能看起来是这样的：

+   **获取数据**：这可能涉及从文本文件中提取信息、轮询传感器网络或查询 Web API。例如，我们可以查询 Twitter API 以获取包含相关标签的推文列表。

+   **数据摄取**：数据通常来自许多不同的来源，可能是非结构化或半结构化的。数据摄取涉及将数据从数据源移动到处理它以提取结构化信息，并将这些信息存储在数据库中。例如，对于推文，我们可能会提取用户名、推文中提到的其他用户的名称、标签、推文文本以及推文是否包含某些关键词。

+   **探索数据**：我们通常对想要从数据中提取的信息有一个明确的想法，但对如何做到这一点却知之甚少。例如，让我们设想我们已经摄取了包含与我们政治运动相关的标签的数千条推文。从我们的推文数据库到最终目标——了解公众对我们运动的总体看法——没有明确的路径。数据探索涉及规划我们如何到达那里。这一步骤通常会揭示新的问题或数据来源，这需要回到管道的第一步。例如，对于我们的推文数据库，我们可能会决定需要有人手动标记一千条或更多的推文，以表达对政治运动的“积极”或“消极”情绪。然后我们可以使用这些推文作为训练集来构建模型。

+   **构建特征**：机器学习算法的好坏取决于进入它的特征。数据科学家的大部分时间都花在转换和组合现有特征以创建与我们要解决的问题更紧密相关的新特征上。例如，我们可能会构建一个新特征，对应于推文中“积极”语气单词或单词对的数量。

+   **模型构建和训练**：在构建了进入模型的特征之后，数据科学家现在可以在他们的数据集上训练机器学习算法。这通常涉及尝试不同的算法和优化模型的**超参数**。例如，我们可能会决定使用随机森林算法来决定一条推文是对活动“正面”还是“负面”的看法。构建模型涉及选择合适的树的数量以及如何计算不纯度度量。对统计学和问题领域的良好理解将有助于这些决策。

+   **模型外推和预测**：数据科学家现在可以使用他们新的模型来尝试推断关于先前未见数据点的信息。他们可能会将一条新的推文通过他们的模型来确认它是否对政治活动持正面或负面的看法。

+   **从模型中提取智能和洞察力**：数据科学家将数据分析过程的结果与业务领域的知识相结合，以指导业务决策。他们可能会发现某些信息与目标受众或目标受众的特定部分产生更好的共鸣，从而实现更精确的目标。向利益相关者提供信息的关键部分涉及数据可视化和展示：数据科学家创建图表、可视化和报告，以帮助使得出的见解清晰且引人入胜。

这远非一个线性的管道。通常，在一个阶段获得的见解将要求数据科学家回溯到管道的先前阶段。确实，从原始数据生成业务见解通常是一个迭代的过程：数据科学家可能会进行快速的第一遍扫描以验证问题的前提，然后通过添加新的数据源或新特征或尝试新的机器学习算法来逐步完善方法。

在这本书中，你将学习如何在 Scala 中处理管道的每个步骤，利用现有库构建健壮的应用程序。

# 数据科学中的编程

这本书不是一本关于数据科学的书。它是一本关于如何使用 Scala 编程语言进行数据科学的书。那么，在处理数据时编程的作用在哪里呢？

计算机在数据科学管道的每个步骤中都发挥作用，但并不一定是同样的方式。如果我们只是编写临时脚本以探索数据或试图构建一个可扩展的应用程序，该应用程序通过一个被充分理解的管道推送数据以持续提供业务智能，那么我们构建的程序的风格将会有很大的不同。

让我们假设我们为一家制作手机游戏的公司工作，在这些游戏中你可以购买游戏内福利。大多数用户从不购买任何东西，但一小部分人可能会花很多钱。我们想要构建一个模型，根据他们的游戏模式识别大额消费者。

第一步是探索数据，找到正确的特征，并在数据的一个子集上构建模型。在这个探索阶段，我们有一个明确的目标，但几乎没有关于如何实现目标的想法。我们希望有一个轻量级、灵活的语言，以及强大的库，以便尽快得到一个工作模型。

一旦我们有了工作模型，我们需要将其部署到我们的游戏平台上，以分析所有当前用户的用法模式。这是一个非常不同的问题：我们对程序的目标和如何实现目标有相对清晰的理解。挑战在于设计能够扩展以处理所有用户并适应未来用法模式变化的软件。

实际上，我们编写的软件类型通常位于从单个一次性脚本到必须能够抵御未来扩展和负载增加的生产级代码的连续谱上。在编写任何代码之前，数据科学家必须了解他们的软件在这个谱上的位置。让我们称这个为**持久性**谱。

# 为什么选择 Scala？

你想编写一个处理数据的程序。你应该选择哪种语言？

有几种不同的选择。你可能会选择一种动态语言，如 Python 或 R，或者一种更传统的面向对象语言，如 Java。在本节中，我们将探讨 Scala 与这些语言的差异以及何时使用它可能是有意义的。

在选择语言时，架构师需要在可证明的正确性和开发速度之间进行权衡。你需要强调哪些方面将取决于应用程序的要求以及你的程序在持久性谱上的位置。这是一个短脚本，将被少数人使用，他们可以轻松修复任何出现的问题？如果是这样，你可能在很少使用的代码路径中允许一定数量的错误：当开发者遇到问题时，他们可以立即修复问题。相比之下，如果你正在开发一个计划发布给更广泛世界的数据库引擎，你很可能会更重视正确性而不是快速开发。例如，SQLite 数据库引擎以其广泛的测试套件而闻名，测试代码量是应用代码的 800 倍（[`www.sqlite.org/testing.html`](https://www.sqlite.org/testing.html)）。

在估计程序**正确性**时，重要的是不是感知到没有错误，而是你能够证明某些错误确实不存在的程度。

在代码运行之前，有几种方法可以证明不存在错误：

+   在静态类型语言中，静态类型检查发生在编译时，但这也可以用于支持类型注解或类型提示的强类型动态语言。类型检查有助于验证我们是否按预期使用函数和类。

+   静态分析器和代码检查器可以检查未定义的变量或可疑的行为（例如，代码中永远无法到达的部分）。

+   在编译语言中声明某些属性为不可变或常量。

+   单元测试以证明特定代码路径上没有 bug。

有几种方法可以检查运行时某些错误的缺失：

+   在静态类型和动态语言中均支持动态类型检查

+   断言验证假设的程序不变性或预期契约

在接下来的章节中，我们将探讨 Scala 在数据科学领域与其他语言的比较。

## 静态类型和类型推断

Scala 的静态类型系统非常灵活。程序行为的大量信息可以编码在类型中，使得编译器能够保证一定程度的正确性。这对于很少使用的代码路径特别有用。动态语言无法在特定执行分支运行之前捕获错误，因此错误可能长时间存在，直到程序遇到它。在静态类型语言中，任何编译器可以捕获的 bug 都会在程序开始运行之前在编译时被捕获。

静态类型面向对象语言常因冗余而被批评。以 Java 中`Example`类实例的初始化为例：

```py
Example myInstance = new Example() ;
```

我们必须重复两次类名——一次是为了定义`myInstance`变量的编译时类型，另一次是为了构造实例本身。这感觉像是多余的工作：编译器知道`myInstance`的类型是`Example`（或`Example`的父类），因为我们绑定了一个`Example`类型的值。

Scala，像大多数函数式语言一样，使用类型推断来允许编译器从绑定到它们的实例中推断变量的类型。我们可以在 Scala 中这样写等效的行：

```py
val myInstance = new Example()
```

Scala 编译器在编译时推断`myInstance`具有`Example`类型。很多时候，指定函数的参数和返回值的类型就足够了。然后编译器可以推断函数体中定义的所有变量的类型。Scala 代码通常比等效的 Java 代码更简洁、更易读，而不牺牲任何类型安全性。

## Scala 鼓励不可变性

Scala 鼓励使用不可变对象。在 Scala 中，定义一个属性为不可变非常容易：

```py
val amountSpent = 200
```

默认集合是不可变的：

```py
val clientIds = List("123", "456") // List is immutable
clientIds(1) = "589" // Compile-time error
```

具有不可变对象消除了常见的错误来源。知道某些对象一旦实例化后就不能改变，可以减少错误可能潜入的地方。而不是考虑对象的生命周期，我们可以专注于构造函数。

## Scala 和函数式程序

Scala 鼓励使用函数式代码。大量的 Scala 代码由使用高阶函数来转换集合组成。作为程序员，你不需要处理遍历集合的细节。让我们编写一个`occurrencesOf`函数，它返回元素在列表中出现的索引：

```py
def occurrencesOfA:List[Int] = {
  for { 
    (currentElem, index) <- collection.zipWithIndex
    if (currentElem == elem)
  } yield index
}
```

这是如何工作的？我们首先声明一个新的列表，`collection.zipWithIndex`，其元素是`(collection(0), 0)`、`(collection(1), 1)`等等：集合的元素及其索引的配对。

然后，我们告诉 Scala 我们想要遍历这个集合，将`currentElem`变量绑定到当前元素，将`index`绑定到索引。我们对迭代应用一个过滤器，只选择那些`currentElem == elem`的元素。然后我们告诉 Scala 只返回`index`变量。

在 Scala 中，我们不需要处理迭代过程的细节。语法非常声明式：我们告诉编译器我们想要集合中每个等于`elem`的元素的索引，然后让编译器去担心如何遍历集合。

考虑 Java 中的等效代码：

```py
static <T> List<Integer> occurrencesOf(T elem, List<T> collection) {
  List<Integer> occurrences = new ArrayList<Integer>() ;
  for (int i=0; i<collection.size(); i++) {
    if (collection.get(i).equals(elem)) {
      occurrences.add(i) ;
    }
  }
  return occurrences ;
}
```

在 Java 中，你首先定义一个（可变的）列表，用于存放你找到的实例。然后通过定义一个计数器来遍历这个集合，依次考虑每个元素，并在需要时将其索引添加到实例列表中。为了使这个方法正常工作，我们需要正确处理许多其他部分。这些部分的存在是因为我们必须告诉 Java 如何遍历集合，并且它们是 bug 的常见来源。

此外，由于大量的代码被迭代机制占用，定义函数逻辑的行更难找到：

```py
static <T> List<Integer> occurrencesOf(T elem, List<T> collection) {
  List<Integer> occurences = new ArrayList<Integer>() ;
  for (int i=0; i<collection.size(); i++) {
    if (collection.get(i).equals(elem)) { 
      occurrences.add(i) ;
    }
  }
  return occurrences ;
}
```

注意，这并不是对 Java 的攻击。事实上，Java 8 增加了一系列函数式构造，如 lambda 表达式、与 Scala 的`Option`相对应的`Optional`类型或流处理。相反，这是为了展示函数式方法在最小化错误潜力和最大化清晰度方面的好处。

## 空指针不确定性

我们经常需要表示值的可能不存在。例如，假设我们正在从 CSV 文件中读取用户名列表。CSV 文件包含姓名和电子邮件信息。然而，一些用户选择不将他们的电子邮件输入到系统中，因此这些信息不存在。在 Java 中，人们通常会使用字符串或`Email`类来表示电子邮件，并通过将那个引用设置为`null`来表示特定用户的电子邮件信息不存在。同样，在 Python 中，我们可能会使用`None`来表示值的缺失。

这种方法很危险，因为我们没有编码电子邮件信息的可能缺失。在任何非平凡程序中，决定实例属性是否可以为`null`需要考虑这个实例定义的每一个场合。这很快就会变得不切实际，因此程序员要么假设变量不是`null`，要么编写过于防御性的代码。

Scala（跟随其他函数式语言的趋势）引入了`Option[T]`类型来表示可能缺失的属性。然后我们可能会写出以下内容：

```py
class User {
  ...
  val email:Option[Email]
  ...
}
```

我们现在已经在类型信息中编码了电子邮件可能不存在的情况。对于使用`User`类的任何程序员来说，电子邮件信息可能不存在是显而易见的。更好的是，编译器知道`email`字段可能不存在，这迫使我们处理这个问题，而不是鲁莽地忽略它，让应用程序在运行时因为空指针异常而崩溃。

所有这些都回到了实现一定程度的可证明正确性的目标。从不使用`null`，我们知道我们永远不会遇到空指针异常。在没有`Option[T]`的语言中实现相同级别的正确性需要编写单元测试来验证当电子邮件属性为`null`时客户端代码的行为是否正确。

注意，在 Java 中，可以使用例如 Google 的 Guava 库([`code.google.com/p/guava-libraries/wiki/UsingAndAvoidingNullExplained`](https://code.google.com/p/guava-libraries/wiki/UsingAndAvoidingNullExplained))或 Java 8 中的`Optional`类来实现这一点。这更多是一个约定：在 Java 中使用`null`来表示值的缺失已经很长时间是规范了。

## 更容易的并行性

编写利用并行架构的程序具有挑战性。尽管如此，解决除了最简单的数据科学问题之外的所有问题仍然是必要的。

并行编程困难，因为我们作为程序员，往往倾向于按顺序思考。在并发程序中推理不同事件可能发生的顺序是非常具有挑战性的。

Scala 提供了几个抽象，这些抽象极大地简化了并行代码的编写。这些抽象通过限制实现并行性的方式来工作。例如，并行集合强制用户将计算表述为对集合的操作序列（如**map**、**reduce**和**filter**）。演员系统要求开发者从封装应用程序状态并通过传递消息进行通信的演员的角度来思考。

限制程序员自由编写他们想要的并行代码似乎有些矛盾，但这可以避免与并发相关的大多数问题。然而，限制程序行为的方式有助于思考其行为。例如，如果一个演员表现不佳，我们知道问题要么在于这个演员的代码，要么在于演员收到的某个消息。

作为具有一致、限制性抽象的强大功能的例子，让我们使用并行集合来解决一个简单的概率问题。我们将计算在 100 次抛硬币中至少得到 60 次正面的概率。我们可以使用蒙特卡洛方法来估计这一点：通过抽取 100 个随机的布尔值来模拟 100 次抛硬币，并检查真值的数量是否至少为 60。我们重复这个过程，直到结果收敛到所需的精度，或者我们等得不耐烦了。

让我们在 Scala 控制台中演示这个过程：

```py
scala> val nTosses = 100
nTosses: Int = 100

scala> def trial = (0 until nTosses).count { i =>
 util.Random.nextBoolean() // count the number of heads
}
trial: Int

```

`trial`函数运行一组 100 次投掷，返回正面朝上的次数：

```py
scala> trial
Int = 51

```

为了得到我们的答案，我们只需要尽可能多地重复`trial`，并汇总结果。重复相同的操作集非常适合并行集合：

```py
scala> val nTrials = 100000
nTrials: Int = 100000

scala> (0 until nTrials).par.count { i => trial >= 60 }
Int = 2745

```

因此，概率大约是 2.5%到 3%。我们只需使用`par`方法并行化范围`(0 until nTrials)`，就可以将计算分布到我们计算机的每个 CPU 上。这证明了具有一致抽象的好处：并行集合使我们能够轻易地将任何可以用集合上的高阶函数表述的计算并行化。

显然，并非每个问题都像简单的蒙特卡洛问题那样容易并行化。然而，通过提供丰富的直观抽象，Scala 使得编写并行应用程序变得可行。

## 与 Java 的互操作性

Scala 运行在 Java 虚拟机上。Scala 编译器将程序编译成 Java 字节码。因此，Scala 开发者可以原生地访问 Java 库。鉴于用 Java 编写的应用程序数量庞大，无论是开源的还是作为组织中的遗留代码的一部分，Scala 和 Java 的互操作性有助于解释 Scala 的快速采用。

互操作性不仅仅是单向的：一些 Scala 库，如 Play 框架，在 Java 开发者中越来越受欢迎。

# 何时不使用 Scala

在前面的章节中，我们描述了 Scala 的强类型系统、对不可变性的偏好、函数能力和并行抽象如何使得编写可靠的程序变得容易，并最小化意外行为的风险。

你可能有哪些理由在下一个项目中避免使用 Scala？一个重要的原因是熟悉度。Scala 引入了许多概念，例如隐式参数、类型类和通过特质使用组合，这些可能对来自面向对象世界的程序员来说并不熟悉。Scala 的类型系统非常强大，但要充分了解它以发挥其全部功能需要时间，并需要适应新的编程范式。最后，对于来自 Java 或 Python 的程序员来说，处理不可变数据结构可能会感到不适应。

然而，这些都是可以通过时间克服的缺点。Scala 在库可用性方面确实不如其他数据科学语言。IPython Notebook 与 matplotlib 结合使用，是数据探索的无与伦比的资源。有持续的努力在 Scala 中提供类似的功能（例如 Spark Notebooks 或 Apache Zeppelin），但没有项目达到相同的成熟度。当探索数据或尝试不同的模型时，类型系统也可能成为轻微的障碍。

因此，在这个作者有偏见的观点中，Scala 在编写更“永久”的程序方面表现出色。如果你正在编写一个一次性脚本或探索数据，你可能会发现 Python 更适合。如果你正在编写需要重用并需要一定程度的可证明正确性的东西，你会发现 Scala 非常强大。

# 摘要

现在必要的介绍已经结束，是时候编写一些 Scala 代码了。在下一章中，你将学习如何利用 Breeze 在 Scala 中进行数值计算。在我们的第一次数据科学探索中，我们将使用逻辑回归来预测给定一个人的身高和体重来预测其性别。

# 参考文献

到目前为止，关于 Scala 的最佳书籍是马丁·奥德斯基（Martin Odersky）、莱克斯·斯波恩（Lex Spoon）和比尔·文纳（Bill Venners）合著的《Programming in Scala》。这本书不仅权威（马丁·奥德斯基是 Scala 的推动力），而且易于接近和阅读。

《Scala Puzzlers》由安德鲁·菲利普斯（Andrew Phillips）和内尔明·谢里福维奇（Nermin Šerifović）所著，提供了一种有趣的方式来学习更高级的 Scala。

《Scala for Machine Learning》由帕特里克·R·尼古拉斯（Patrick R. Nicholas）所著，提供了如何使用 Scala 编写机器学习算法的示例。

# 第二章：使用 Breeze 操作数据

数据科学在很大程度上是关于结构化数据的操作。大量结构化数据集可以被视为表格数据：每一行代表一个特定的实例，而列代表该实例的不同属性。表格表示的普遍性解释了像 Microsoft Excel 这样的电子表格程序或像 SQL 数据库这样的工具的成功。

要对数据科学家有用，一种语言必须支持数据列或表格的操作。例如，Python 通过 NumPy 和 pandas 来实现这一点。不幸的是，在 Scala 中，没有一个单一、连贯的数值计算生态系统可以与 Python 中的 SciPy 生态系统相媲美。

在本章中，我们将介绍 Breeze，这是一个用于快速线性代数和数据数组操作以及许多其他科学计算和数据科学所需特性的库。

# 代码示例

访问本书中的代码示例最简单的方法是克隆 GitHub 仓库：

```py
$ git clone 'https://github.com/pbugnion/s4ds'

```

每章的代码示例都在一个单独的独立文件夹中。你还可以在 GitHub 上在线浏览代码。

# 安装 Breeze

如果你已经下载了本书的代码示例，使用 Breeze 最简单的方法是进入`chap02`目录，并在命令行中输入`sbt console`。这将打开一个 Scala 控制台，你可以在其中导入 Breeze。

如果你想要构建一个独立的项目，安装 Breeze（以及任何 Scala 模块）最常见的方式是通过 SBT。为了获取本章所需的依赖项，将以下行复制到名为`build.sbt`的文件中，注意在`scalaVersion`之后留一个空行：

```py
scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2"
)
```

在终端中输入`sbt console`，在`build.sbt`文件相同的目录中打开 Scala 控制台。您可以通过从 Scala 提示符导入 Breeze 来检查 Breeze 是否正常工作：

```py
scala> import breeze.linalg._
import breeze.linalg._

```

# 获取 Breeze 的帮助

本章对 Breeze 进行了相当详细的介绍，但并不旨在提供完整的 API 参考。

要获取 Breeze 功能的完整列表，请查阅 GitHub 上的 Breeze Wiki 页面[`github.com/scalanlp/breeze/wiki`](https://github.com/scalanlp/breeze/wiki)。对于某些模块来说，这是非常完整的，而对于其他模块来说则不那么完整。源代码（[`github.com/scalanlp/breeze/`](https://github.com/scalanlp/breeze/））详细且提供了大量信息。要了解特定函数的预期用法，请查看该函数的单元测试。

# 基本的 Breeze 数据类型

Breeze 是一个功能丰富的库，提供了对数据数组进行快速和简单操作的功能，包括优化、插值、线性代数、信号处理和数值积分的例程。

Breeze 背后的基本线性代数操作依赖于`netlib-java`库，该库可以使用系统优化的**BLAS**和**LAPACK**库（如果存在）。因此，Breeze 中的线性代数操作通常非常快。Breeze 仍在快速发展中，因此可能有些不稳定。

## 向量

Breeze 使得操作一维和二维数据结构变得简单。首先，通过 SBT 打开 Scala 控制台并导入 Breeze：

```py
$ sbt console
scala> import breeze.linalg._
import breeze.linalg._

```

让我们直接定义一个向量：

```py
scala> val v = DenseVector(1.0, 2.0, 3.0)
breeze.linalg.DenseVector[Double] = DenseVector(1.0, 2.0, 3.0)

```

我们刚刚定义了一个包含三个元素的向量，`v`。向量只是一维数据数组，提供了针对数值使用的定制方法。它们可以像其他 Scala 集合一样进行索引：

```py
scala> v(1)
Double = 2.0

```

它们还支持与标量的逐元素操作：

```py
scala> v :* 2.0 // :* is 'element-wise multiplication'
breeze.linalg.DenseVector[Double] = DenseVector(2.0, 4.0, 6.0)

```

它们还支持与另一个向量的逐元素操作：

```py
scala> v :+ DenseVector(4.0, 5.0, 6.0) // :+ is 'element-wise addition'
breeze.linalg.DenseVector[Double] = DenseVector(5.0, 7.0, 9.0)

```

Breeze 使得编写向量操作直观，并且比 Scala 原生等价物更易于阅读。

注意，Breeze 将在编译时拒绝将操作数强制转换为正确的类型：

```py
scala> v :* 2 // element-wise multiplication by integer
<console>:15: error: could not find implicit value for parameter op:
...

```

它还会在运行时拒绝将不同长度的向量相加：

```py
scala> v :+ DenseVector(8.0, 9.0)
java.lang.IllegalArgumentException: requirement failed: Vectors must have same length: 3 != 2
...

```

在 Breeze 中对向量的基本操作对习惯于使用 NumPy、MATLAB 或 R 的人来说将感觉自然。

到目前为止，我们只看了*逐元素*运算符。这些运算符都是以冒号开头的。所有常见的运算符都存在：`:+`、`:*`、`:-`、`:/`、`:%`（余数）和`:^`（幂）以及布尔运算符。要查看运算符的完整列表，请查看`DenseVector`或`DenseMatrix`的 API 文档（[`github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet`](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet)）。

除了逐元素操作外，Breeze 向量还支持您可能期望的数学向量的操作，例如点积：

```py
scala> val v2 = DenseVector(4.0, 5.0, 6.0)
breeze.linalg.DenseVector[Double] = DenseVector(4.0, 5.0, 6.0)

scala> v dot v2
Double = 32.0

```

### 小贴士

**逐元素运算符的陷阱**

除了我们迄今为止看到的用于逐元素加法和减法的 `:+` 和 `:-` 运算符之外，我们还可以使用更传统的 `+` 和 `-` 运算符：

```py
scala> v + v2
breeze.linalg.DenseVector[Double] = DenseVector(5.0, 7.0, 9.0)

```

然而，在将 `:+` 或 `:*` 与 `:+` 运算符混合使用时，必须非常小心运算符优先级规则。`:+` 和 `:*` 运算符的运算优先级非常低，因此它们将被最后评估。这可能会导致一些不符合直觉的行为：

```py
scala> 2.0 :* v + v2 // !! equivalent to 2.0 :* (v + v2)
breeze.linalg.DenseVector[Double] = DenseVector(10.0, 14.0, 18.0)

```

相比之下，如果我们使用 `:+` 而不是 `+`，则运算符的数学优先级将被尊重：

```py
scala> 2.0 :* v :+ v2 // equivalent to (2.0 :* v) :+ v2
breeze.linalg.DenseVector[Double] = DenseVector(6.0, 9.0, 12.0)

```

总结来说，应尽可能避免将 `:+` 风格的运算符与 `+` 风格的运算符混合使用。

## 稠密和稀疏矢量和矢量特性

我们迄今为止查看的所有矢量都是稠密矢量。Breeze 也支持稀疏矢量。当处理主要由零组成的数字数组时，使用稀疏矢量可能更有效。矢量何时有足够的零以证明切换到稀疏表示取决于操作类型，因此你应该运行自己的基准测试以确定使用哪种类型。尽管如此，一个好的启发式方法是，如果你的矢量大约有 90% 是零，那么使用稀疏表示可能会有所裨益。

稀疏矢量在 Breeze 中以 `SparseVector` 和 `HashVector` 类的形式提供。这两种类型都支持与 `DenseVector` 相同的许多操作，但使用不同的内部实现。`SparseVector` 实例非常节省内存，但添加非零元素的速度较慢。`HashVector` 更灵活，但代价是内存占用和遍历非零元素的计算时间增加。除非你需要从应用程序中挤出最后一点内存，否则我建议使用 `HashVector`。本书中不会进一步讨论这些内容，但如果需要，读者应该会发现它们的使用非常直观。`DenseVector`、`SparseVector` 和 `HashVector` 都实现了 `Vector` 特性，从而提供了统一的接口。

### 小贴士

Breeze 仍然处于实验阶段，并且截至本文撰写时，有些不稳定。我发现处理 `Vector` 特性的特定实现（如 `DenseVector` 或 `SparseVector`）比直接处理 `Vector` 特性更可靠。在本章中，我们将明确地将每个矢量类型指定为 `DenseVector`。

## 矩阵

Breeze 允许以类似的方式构建和操作二维数组：

```py
scala> val m = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
breeze.linalg.DenseMatrix[Double] =
1.0  2.0  3.0
4.0  5.0  6.0

scala> 2.0 :* m
breeze.linalg.DenseMatrix[Double] =
2.0  4.0   6.0
8.0  10.0  12.0

```

## 构建矢量和矩阵

我们已经看到了如何通过将它们的值传递给构造函数（或者更确切地说，传递给伴随对象的 `apply` 方法）来显式构建矢量和矩阵：`DenseVector(1.0, 2.0, 3.0)`。Breeze 提供了构建矢量和矩阵的几种其他强大方法：

```py
scala> DenseVector.onesDouble
breeze.linalg.DenseVector[Double] = DenseVector(1.0, 1.0, 1.0, 1.0, 1.0)

scala> DenseVector.zerosInt
breeze.linalg.DenseVector[Int] = DenseVector(0, 0, 0)

```

`linspace` 方法（在 `breeze.linalg` 包对象中可用）创建一个等间隔值的 `Double` 矢量。例如，要创建一个在 `0` 和 `1` 之间均匀分布的 10 个值的矢量，请执行以下操作：

```py
scala> linspace(0.0, 1.0, 10)
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 0.1111111111111111, ..., 1.0)

```

`tabulate` 方法允许我们通过函数构造向量和矩阵：

```py
scala> DenseVector.tabulate(4) { i => 5.0 * i }
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 5.0, 10.0, 15.0)

scala> DenseMatrix.tabulateInt { 
 (irow, icol) => irow*2 + icol 
}
breeze.linalg.DenseMatrix[Int] =
0  1  2
2  3  4

```

`DenseVector.tabulate` 的第一个参数是向量的长度，第二个参数是一个函数，它返回向量在特定位置的值。这有助于创建数据范围，以及其他用途。

`rand` 函数允许我们创建随机向量和矩阵：

```py
scala> DenseVector.rand(2)
breeze.linalg.DenseVector[Double] = DenseVector(0.8072865137359484, 0.5566507203838562)

scala> DenseMatrix.rand(2, 3)
breeze.linalg.DenseMatrix[Double] =
0.5755491874682879   0.8142161471517582  0.9043780212739738
0.31530195124023974  0.2095094278911871  0.22069103504148346

```

最后，我们可以从 Scala 数组构造向量：

```py
scala> DenseVector(Array(2, 3, 4))
breeze.linalg.DenseVector[Int] = DenseVector(2, 3, 4)

```

要从其他 Scala 集合构造向量，必须使用 *splat* 操作符，`:_ *`：

```py
scala> val l = Seq(2, 3, 4)
l: Seq[Int] = List(2, 3, 4)

scala> DenseVector(l :_ *)
breeze.linalg.DenseVector[Int] = DenseVector(2, 3, 4)

```

## 高级索引和切片

我们已经看到如何通过索引选择向量 `v` 中的特定元素，例如 `v(2)`。Breeze 还提供了几个强大的方法来选择向量的部分。

让我们先创建一个向量来玩耍：

```py
scala> val v = DenseVector.tabulate(5) { _.toDouble }
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 1.0, 2.0, 3.0, 4.0)

```

与原生 Scala 集合不同，Breeze 向量支持负索引：

```py
scala> v(-1) // last element
Double = 4.0

```

Breeze 允许我们使用范围来切片向量：

```py
scala> v(1 to 3)
breeze.linalg.DenseVector[Double] = DenseVector(1.0, 2.0, 3.0)

scala v(1 until 3) // equivalent to Python v[1:3]
breeze.linalg.DenseVector[Double] = DenseVector(1.0, 2.0)

scala> v(v.length-1 to 0 by -1) // reverse view of v
breeze.linalg.DenseVector[Double] = DenseVector(4.0, 3.0, 2.0, 1.0, 0.0)

```

### 小贴士

通过范围进行索引返回原始向量的一个 *视图*：当运行 `val v2 = v(1 to 3)` 时，不会复制任何数据。这意味着切片非常高效。从大向量中取切片不会增加内存占用。这也意味着在更新切片时应该小心，因为它也会更新原始向量。我们将在本章后续部分讨论向量和矩阵的修改。

Breeze 还允许我们从向量中选择任意一组元素：

```py
scala> val vSlice = v(2, 4) // Select elements at index 2 and 4
breeze.linalg.SliceVector[Int,Double] = breeze.linalg.SliceVector@9c04d22

```

这创建了一个 `SliceVector`，它类似于 `DenseVector`（两者都实现了 `Vector` 接口），但实际上并没有为值分配内存：它只是知道如何将其索引映射到父向量的值。可以将 `vSlice` 视为 `v` 的一个特定视图。我们可以通过将其转换为 `DenseVector` 来具体化视图（给它自己的数据，而不是作为 `v` 的观察透镜）：

```py
scala> vSlice.toDenseVector
breeze.linalg.DenseVector[Double] = DenseVector(2.0, 4.0)

```

注意，如果切片中的某个元素超出了范围，只有在访问该元素时才会抛出异常：

```py
scala> val vSlice = v(2, 7) // there is no v(7)
breeze.linalg.SliceVector[Int,Double] = breeze.linalg.SliceVector@2a83f9d1

scala> vSlice(0) // valid since v(2) is still valid
Double = 2.0

scala> vSlice(1) // invalid since v(7) is out of bounds
java.lang.IndexOutOfBoundsException: 7 not in [-5,5)
 ...

```

最后，可以使用布尔数组来索引向量。让我们先定义一个数组：

```py
scala> val mask = DenseVector(true, false, false, true, true)
breeze.linalg.DenseVector[Boolean] = DenseVector(true, false, false, true, true)

```

然后，`v(mask)` 结果是一个包含 `v` 中 `mask` 为 `true` 的元素的视图：

```py
scala> v(mask).toDenseVector
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 3.0, 4.0)

```

这可以用作过滤向量中某些元素的一种方式。例如，要选择小于 `3.0` 的 `v` 的元素：

```py
scala> val filtered = v(v :< 3.0) // :< is element-wise "less than"
breeze.linalg.SliceVector[Int,Double] = breeze.linalg.SliceVector@2b1edef3

scala> filtered.toDenseVector
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 1.0, 2.0)

```

矩阵的索引方式与向量非常相似。矩阵索引函数接受两个参数——第一个参数选择行（s），第二个参数切片列（s）：

```py
scala> val m = DenseMatrix((1.0, 2.0, 3.0), (5.0, 6.0, 7.0))
m: breeze.linalg.DenseMatrix[Double] =
1.0  2.0  3.0
5.0  6.0  7.0

scala> m(1, 2)
Double = 7.0

scala> m(1, -1)
Double = 7.0

scala> m(0 until 2, 0 until 2)
breeze.linalg.DenseMatrix[Double] =
1.0  2.0
5.0  6.0

```

您还可以混合不同类型的行和列切片：

```py
scala> m(0 until 2, 0)
breeze.linalg.DenseVector[Double] = DenseVector(1.0, 5.0)

```

注意，在这种情况下，Breeze 返回一个向量。一般来说，切片操作返回以下对象：

+   当传递单个索引作为行和列参数时，返回一个标量

+   当行参数是一个范围，列参数是一个单个索引时，返回一个向量

+   当列参数是一个范围，行参数是一个单个索引时，返回一个向量的转置

+   否则返回一个矩阵

符号`::`可以用来表示*沿特定方向的每个元素*。例如，我们可以选择`m`的第二列：

```py
scala> m(::, 1)
breeze.linalg.DenseVector[Double] = DenseVector(2.0, 6.0)

```

## 修改向量和矩阵

Breeze 中的向量和矩阵是可变的。上述描述的大多数切片操作也可以用来设置向量和矩阵的元素：

```py
scala> val v = DenseVector(1.0, 2.0, 3.0)
v: breeze.linalg.DenseVector[Double] = DenseVector(1.0, 2.0, 3.0)

scala> v(1) = 22.0 // v is now DenseVector(1.0, 22.0, 3.0)

```

我们不仅限于修改单个元素。实际上，上述概述的所有索引操作都可以用来设置向量和矩阵的元素。在修改向量和矩阵的切片时，使用逐元素赋值运算符`:=`：

```py
scala> v(0 until 2) := DenseVector(50.0, 51.0) // set elements at position 0 and 1
breeze.linalg.DenseVector[Double] = DenseVector(50.0, 51.0)

scala> v
breeze.linalg.DenseVector[Double] = DenseVector(50.0, 51.0, 3.0)

```

赋值运算符`:=`在 Breeze 中像其他逐元素运算符一样工作。如果右侧是一个标量，它将自动广播到给定形状的向量：

```py
scala> v(0 until 2) := 0.0 // equivalent to v(0 until 2) := DenseVector(0.0, 0.0)
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 0.0)

scala> v
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 0.0, 3.0)

```

所有逐元素运算符都有一个更新对应的运算符。例如，`:+=`运算符类似于逐元素加法运算符`:+`，但也会更新其左侧的操作数：

```py
scala> val v = DenseVector(1.0, 2.0, 3.0)
v: breeze.linalg.DenseVector[Double] = DenseVector(1.0, 2.0, 3.0)

scala> v :+= 4.0
breeze.linalg.DenseVector[Double] = DenseVector(5.0, 6.0, 7.0)

scala> v
breeze.linalg.DenseVector[Double] = DenseVector(5.0, 6.0, 7.0)

```

注意更新操作是如何就地更新向量并返回它的。

我们已经学会了如何在 Breeze 中切片向量和矩阵来创建原始数据的新的视图。这些视图并不独立于它们创建的向量——更新视图将更新底层向量，反之亦然。这最好用一个例子来说明：

```py
scala> val v = DenseVector.tabulate(6) { _.toDouble }
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

scala> val viewEvens = v(0 until v.length by 2)
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 2.0, 4.0)

scala> viewEvens := 10.0 // mutate viewEvens
breeze.linalg.DenseVector[Double] = DenseVector(10.0, 10.0, 10.0)

scala> viewEvens
breeze.linalg.DenseVector[Double] = DenseVector(10.0, 10.0, 10.0)

scala> v  // v has also been mutated!
breeze.linalg.DenseVector[Double] = DenseVector(10.0, 1.0, 10.0, 3.0, 10.0, 5.0)

```

如果我们记住，当我们创建一个向量和矩阵时，我们实际上是在创建一个底层数据数组的视图，而不是创建数据本身，这会很快变得直观：

![修改向量和矩阵](img/image01159.jpeg)

`v`向量的切片`v(0 to 6 by 2)`只是`v`底层数组的另一种视图。这个视图本身不包含数据。它只包含指向原始数组中数据的指针。内部，视图只是存储为指向底层数据的指针和一个遍历该数据的配方：在这个切片的情况下，配方就是“从底层数据的第一元素开始，以两步的间隔到达底层数据的第七元素”。

当我们想要创建数据的独立副本时，Breeze 提供了一个`copy`函数。在之前的例子中，我们可以构建`viewEvens`的副本如下：

```py
scala> val copyEvens = v(0 until v.length by 2).copy
breeze.linalg.DenseVector[Double] = DenseVector(10.0, 10.0, 10.0)

```

现在，我们可以独立于`v`更新`copyEvens`。

## 矩阵乘法、转置和向量的方向

到目前为止，我们主要关注了向量和矩阵的逐元素操作。现在让我们看看矩阵乘法和相关操作。

矩阵乘法运算符是`*`：

```py
scala> val m1 = DenseMatrix((2.0, 3.0), (5.0, 6.0), (8.0, 9.0))
breeze.linalg.DenseMatrix[Double] =
2.0  3.0
5.0  6.0
8.0  9.0

scala> val m2 = DenseMatrix((10.0, 11.0), (12.0, 13.0))
breeze.linalg.DenseMatrix[Double] 
10.0  11.0
12.0  13.0

scala> m1 * m2
56.0   61.0
122.0  133.0
188.0  205.0

```

除了矩阵-矩阵乘法之外，我们还可以在矩阵和向量之间使用矩阵乘法运算符。Breeze 中的所有向量都是列向量。这意味着，当矩阵和向量相乘时，向量应被视为一个(*n * 1*)矩阵。让我们通过一个矩阵-向量乘法的例子来了解一下。我们想要以下操作：

![矩阵乘法、转置和向量的方向](img/image01160.jpeg)

```py
scala> val v = DenseVector(1.0, 2.0)
breeze.linalg.DenseVector[Double] = DenseVector(1.0, 2.0)

scala> m1 * v 
breeze.linalg.DenseVector[Double] = DenseVector(8.0, 17.0, 26.0)

```

相比之下，如果我们想要：

![矩阵乘法、转置和向量的方向](img/image01161.jpeg)

我们必须将 `v` 转换为一个行向量。我们可以使用转置操作来完成此操作：

```py
scala> val vt = v.t
breeze.linalg.Transpose[breeze.linalg.DenseVector[Double]] = Transpose(DenseVector(1.0, 2.0))

scala> vt * m2
breeze.linalg.Transpose[breeze.linalg.DenseVector[Double]] = Transpose(DenseVector(34.0, 37.0))

```

注意，`v.t` 的类型是 `Transpose[DenseVector[_]]`。在逐元素操作方面，`Transpose[DenseVector[_]]` 与 `DenseVector` 几乎以相同的方式表现，但它不支持修改或切片。

## 数据预处理和特征工程

我们现在已经发现了 Breeze 的基本组件。在接下来的几节中，我们将将它们应用于实际例子，以了解它们如何组合在一起形成数据科学的一个强大基础。

数据科学的一个重要部分涉及预处理数据集以构建有用的特征。让我们通过一个例子来了解这个过程。为了跟随这个例子并访问数据，您需要下载本书的代码示例（[www.github.com/pbugnion/s4ds](http://www.github.com/pbugnion/s4ds)）。

您将在本书附带的代码的 `chap02/data/` 目录中找到一个 CSV 文件，其中包含 181 名男性和女性的真实身高和体重，以及自我报告的身高和体重。原始数据集是作为一项关于身体形象的研究的一部分收集的。有关更多信息，请参阅以下链接：[`vincentarelbundock.github.io/Rdatasets/doc/car/Davis.html`](http://vincentarelbundock.github.io/Rdatasets/doc/car/Davis.html)。

本书提供的包中有一个辅助函数，用于将数据加载到 Breeze 数组中：

```py
scala> val data = HWData.load
HWData [ 181 rows ]

scala> data.genders
breeze.linalg.Vector[Char] = DenseVector(M, F, F, M, ... )

```

`data` 对象包含五个向量，每个向量长度为 181：

+   `data.genders`：一个描述参与者性别的 `Char` 向量

+   `data.heights`：一个包含参与者真实身高的 `Double` 向量

+   `data.weights`：一个包含参与者真实体重的 `Double` 向量

+   `data.reportedHeights`：一个包含参与者自我报告身高的 `Double` 向量

+   `data.reportedWeights`：一个包含参与者自我报告体重的 `Double` 向量

让我们先计算研究中男性和女性的数量。我们将定义一个只包含 `'M'` 的数组，并与 `data.genders` 进行逐元素比较：

```py
scala> val maleVector = DenseVector.fill(data.genders.length)('M')
breeze.linalg.DenseVector[Char] = DenseVector(M, M, M, M, M, M,... )

scala> val isMale = (data.genders :== maleVector)
breeze.linalg.DenseVector[Boolean] = DenseVector(true, false, false, true ...)

```

`isMale` 向量与 `data.genders` 的长度相同。当参与者为男性时，它为 `true`，否则为 `false`。我们可以使用这个布尔数组作为数据集中其他数组的掩码（记住 `vector(mask)` 选择 `vector` 中掩码为 `true` 的元素）。让我们获取我们数据集中男性的身高：

```py
scala> val maleHeights = data.heights(isMale)
breeze.linalg.SliceVector[Int,Double] = breeze.linalg.SliceVector@61717d42

scala> maleHeights.toDenseVector
breeze.linalg.DenseVector[Double] = DenseVector(182.0, 177.0, 170.0, ...

```

要计算我们数据集中男性的数量，我们可以使用指示函数。这个函数将布尔数组转换为一个双精度浮点数数组，将 `false` 映射到 `0.0`，将 `true` 映射到 `1.0`：

```py
scala> import breeze.numerics._
import breeze.numerics._

scala> sum(I(isMale))
Double: 82.0

```

让我们计算实验中男性和女性的平均身高。我们可以使用 `mean(v)` 计算向量的平均值，通过导入 `breeze.stats._` 来访问它：

```py
scala> import breeze.stats._
import breeze.stats._

scala> mean(data.heights)
Double = 170.75690607734808

```

要计算男性的平均身高，我们可以使用我们的`isMale`数组来切片`data.heights`；`data.heights(isMale)`是`data.heights`数组的一个视图，其中包含所有男性的身高值：

```py
scala> mean(data.heights(isMale)) // mean male height
Double = 178.0121951219512

scala> mean(data.heights(!isMale)) // mean female height
Double = 164.74747474747474

```

作为稍微复杂一点的例子，让我们看看在这个实验中男性和女性实际体重与报告体重之间的差异。我们可以得到一个报告体重与真实体重之间百分比差异的数组：

```py
scala> val discrepancy = (data.weights - data.reportedWeights) / data.weights
breeze.linalg.Vector[Double] = DenseVector(0.0, 0.1206896551724138, -0.018867924528301886, -0.029411764705882353, ... )

```

注意 Breeze 对数学运算符的重载如何使我们能够轻松优雅地操作数据数组。

我们现在可以计算这个数组中男性的平均值和标准差：

```py
scala> mean(discrepancy(isMale))
res6: Double = -0.008451852933123775

scala> stddev(discrepancy(isMale))
res8: Double = 0.031901519634244195

```

我们还可以计算高估自己身高的男性的比例：

```py
scala> val overReportMask = (data.reportedHeights :> data.heights).toDenseVector
breeze.linalg.DenseVector[Boolean] = DenseVector(false, false, false, false...

scala> sum(I(overReportMask :& isMale))
Double: 10.0

```

因此，有十名男性认为自己比实际更高。逐元素 AND 运算符`:&`返回一个向量，对于其两个参数都为真的所有索引，该向量是真实的。因此，向量`overReportMask :& isMale`对于所有报告身高超过实际身高的男性参与者都是真实的。

## Breeze – 函数优化

在研究了特征工程之后，我们现在来看看数据科学管道的另一端。通常，机器学习算法定义了一个损失函数，该函数是一组参数的函数。损失函数的值表示模型拟合数据的程度。然后，参数被优化以最小化（或最大化）损失函数。

在第十二章《使用 MLlib 的分布式机器学习》中，我们将探讨**MLlib**，这是一个包含许多知名算法的机器学习库。通常，我们不需要担心直接优化损失函数，因为我们可以依赖 MLlib 提供的机器学习算法。然而，了解优化基础知识仍然很有用。

Breeze 有一个`optimize`模块，其中包含用于寻找局部最小值的函数：

```py
scala> import breeze.optimize._
import breeze.optimize._

```

让我们创建一个我们想要优化的玩具函数：

![Breeze – 函数优化](img/image01162.jpeg)

我们可以用以下方式在 Scala 中表示这个函数：

```py
scala> def f(xs:DenseVector[Double]) = sum(xs :^ 2.0)
f: (xs: breeze.linalg.DenseVector[Double])Double

```

大多数局部优化器也要求提供正在优化的函数的梯度。梯度是与函数的参数相同维度的向量。在我们的例子中，梯度是：

![Breeze – 函数优化](img/image01163.jpeg)

我们可以用一个接受向量参数并返回相同长度向量的函数来表示梯度：

```py
scala> def gradf(xs:DenseVector[Double]) = 2.0 :* xs
gradf: (xs:breeze.linalg.DenseVector[Double])breeze.linalg.DenseVector[Double]

```

例如，在点`(1, 1, 1)`处，我们有：

```py
scala> val xs = DenseVector.onesDouble
breeze.linalg.DenseVector[Double] = DenseVector(1.0, 1.0, 1.0)

scala> f(xs)
Double = 3.0

scala> gradf(xs)
breeze.linalg.DenseVector[Double] = DenseVector(2.0, 2.0, 2.0)

```

让我们设置优化问题。Breeze 的优化方法要求我们传递一个实现了`DiffFunction`特质的实现，该特质只有一个方法，即`calculate`。此方法必须返回一个包含函数及其梯度的元组：

```py
scala> val optTrait = new DiffFunction[DenseVector[Double]] {
 def calculate(xs:DenseVector[Double]) = (f(xs), gradf(xs))
}
breeze.optimize.DiffFunction[breeze.linalg.DenseVector[Double]] = <function1>

```

我们现在可以运行优化了。优化模块提供了一个`minimize`函数，它正好符合我们的需求。我们传递给它`optTrait`和优化的起始点：

```py
scala> val minimum = minimize(optTrait, DenseVector(1.0, 1.0, 1.0))
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 0.0, 0.0)

```

真实最小值在`(0.0, 0.0, 0.0)`。因此，优化器正确地找到了最小值。

`minimize`函数默认使用**L-BFGS**方法运行优化。它接受几个额外的参数来控制优化。我们将在下一节中探讨这些参数。

## 数值微分

在前面的例子中，我们明确指定了`f`的梯度。虽然这通常是好的做法，但计算函数的梯度往往很繁琐。Breeze 提供了一个使用有限差分的梯度近似函数。重用与上一节相同的目标准函数`def f(xs:DenseVector[Double]) = sum(xs :^ 2.0)`：

```py
scala> val approxOptTrait = new ApproximateGradientFunction(f)
breeze.optimize.ApproximateGradientFunction[Int,breeze.linalg.DenseVector[Double]] = <function1>

```

特性`approxOptTrait`有一个`gradientAt`方法，它返回在一点处的梯度近似值：

```py
scala> approxOptTrait.gradientAt(DenseVector.ones(3))
breeze.linalg.DenseVector[Double] = DenseVector(2.00001000001393, 2.00001000001393, 2.00001000001393)

```

注意，这可能会相当不准确。`ApproximateGradientFunction`构造函数接受一个可选的`epsilon`参数，该参数控制计算有限差分时步长的大小。改变`epsilon`的值可以提高有限差分算法的准确性。

`ApproximateGradientFunction`实例实现了`DiffFunction`特性。因此，它可以直接传递给`minimize`。

```py
scala> minimize(approxOptTrait, DenseVector.onesDouble)
breeze.linalg.DenseVector[Double] = DenseVector(-5.000001063126813E-6, -5.000001063126813E-6, -5.000001063126813E-6)

```

这再次给出了接近零的结果，但比我们明确指定梯度时稍远一些。一般来说，通过解析计算函数的梯度将比依赖 Breeze 的数值梯度更有效、更准确。可能最好只在数据探索期间或检查解析梯度时使用数值梯度。

## 正则化

`minimize`函数接受许多与机器学习算法相关的可选参数。特别是，我们可以指示优化器在执行优化时使用正则化参数。正则化在损失函数中引入惩罚，以防止参数任意增长。这有助于避免过拟合。我们将在第十二章中更详细地讨论正则化，*使用 MLlib 的分布式机器学习*。

例如，要使用具有超参数`0.5`的`L2Regularization`：

```py
scala> minimize(optTrait, DenseVector(1.0, 1.0, 1.0), L2Regularization(0.5))
breeze.linalg.DenseVector[Double] = DenseVector(0.0, 0.0, 0.0)

```

在这个例子中，正则化没有区别，因为参数在最小值处为零。

要查看可以传递给`minimize`的可选参数列表，请查阅在线的 Breeze 文档。

# 一个例子——逻辑回归

现在让我们想象我们想要构建一个分类器，它接受一个人的**身高**和**体重**，并为他们被分配为**男性**或**女性**的概率。我们将重用本章前面引入的身高和体重数据。让我们先绘制数据集：

![一个例子——逻辑回归](img/image01164.jpeg)

181 名男性和女性的身高与体重数据

分类算法有很多种。初步观察数据表明，我们可以通过在图上画一条直线来大致区分男性和女性。因此，线性方法是对分类的合理初步尝试。在本节中，我们将使用逻辑回归来构建分类器。

逻辑回归的详细解释超出了本书的范围。对逻辑回归不熟悉的读者可参考 *Hastie*、*Tibshirani* 和 *Friedman* 所著的 *《统计学习的要素》*。我们在这里只做简要概述。

逻辑回归使用以下 sigmoid 函数估计给定身高和体重属于男性的概率：

![一个示例 – 逻辑回归](img/image01165.jpeg)

在这里，*f* 是一个线性函数：

![一个示例 – 逻辑回归](img/image01166.jpeg)

在这里，![一个示例 – 逻辑回归](img/image01167.jpeg) 是我们需要使用训练集确定的参数数组。如果我们将身高和体重视为 *features = (height, weight)* 矩阵，我们可以将 sigmoid 核 *f* 重新写为 *features* 矩阵与 *params* 向量的矩阵乘法：

![一个示例 – 逻辑回归](img/image01168.jpeg)

为了进一步简化这个表达式，通常会在 *features* 矩阵中添加一个值始终为 *1* 的虚拟特征。然后我们可以将 *params(0)* 乘以这个特征，这样我们就可以将整个 sigmoid 核 *f* 写作一个单一的矩阵-向量乘法：

![一个示例 – 逻辑回归](img/image01169.jpeg)

特征矩阵，*features*，现在是一个 (*181 * 3*) 矩阵，其中每一行代表一个特定参与者的 *(1, height, weight)*。

为了找到参数的最优值，我们可以最大化似然函数，*L(params|features)*。似然函数以一组给定的参数值作为输入，并返回这些特定参数导致训练集的概率。对于一组参数和相关的概率函数 *P(male|features[i])*，似然函数为：

![一个示例 – 逻辑回归](img/image01170.jpeg)

如果我们神奇地提前知道人口中每个人的性别，我们可以将男性的概率分配为 *P(male)=1*，女性的概率分配为 *P(male)=0*。此时，似然函数将是 **1**。相反，任何不确定性都会导致似然函数的降低。如果我们选择一组参数，这些参数始终导致分类错误（男性低 *P(male)* 或女性高 *P(male)*），则似然函数将降至 *0*。

最大似然对应于最有可能描述观察数据的参数值。因此，为了找到最能描述我们的训练集的参数，我们只需要找到最大化 *L(params|features)* 的参数。然而，最大化似然函数本身很少被做，因为它涉及到将许多小的值相乘，这很快就会导致浮点下溢。最好最大化似然的对数，它与似然有相同的最大值。最后，由于大多数优化算法都是针对最小化函数而不是最大化函数而设计的，因此我们将最小化 ![一个示例 – 逻辑回归](img/image01171.jpeg)。

对于逻辑回归，这相当于最小化：

![一个示例 – 逻辑回归](img/image01172.jpeg)

在这里，求和遍历训练数据中的所有参与者，![一个示例 – 逻辑回归](img/image01173.jpeg) 是训练集中第 *i*- 次观察的向量 ![一个示例 – 逻辑回归](img/image01174.jpeg)，而 ![一个示例 – 逻辑回归](img/image01175.jpeg) 如果该人是男性则为 *1*，如果是女性则为 *0*。

为了最小化 *成本* 函数，我们还必须知道其相对于参数的梯度。这是：

![一个示例 – 逻辑回归](img/image01176.jpeg)

我们将首先通过它们的平均值和标准差来缩放身高和体重。虽然这对逻辑回归来说不是严格必要的，但通常是一个好的实践。它有助于优化，如果我们想使用正则化方法或构建超线性特征（允许将男性和女性分开的边界是曲线而不是直线），则成为必要。

对于这个示例，我们将离开 Scala shell 并编写一个独立的 Scala 脚本。以下是完整的代码列表。不要担心这看起来令人畏惧。我们将在下一分钟将其分解成可管理的块：

```py
import breeze.linalg._
import breeze.numerics._
import breeze.optimize._
import breeze.stats._

object LogisticRegressionHWData extends App {

  val data = HWData.load

  // Rescale the features to have mean of 0.0 and s.d. of 1.0
  def rescaled(v:DenseVector[Double]) =
    (v - mean(v)) / stddev(v)

  val rescaledHeights = rescaled(data.heights)
  val rescaledWeights = rescaled(data.weights)

  // Build the feature matrix as a matrix with 
  //181 rows and 3 columns.
  val rescaledHeightsAsMatrix = rescaledHeights.toDenseMatrix.t
  val rescaledWeightsAsMatrix = rescaledWeights.toDenseMatrix.t

  val featureMatrix = DenseMatrix.horzcat(
    DenseMatrix.onesDouble,
    rescaledHeightsAsMatrix,
    rescaledWeightsAsMatrix
  )

  println(s"Feature matrix size: ${featureMatrix.rows} x " +s"${featureMatrix.cols}")

  // Build the target variable to be 1.0 where a participant
  // is male, and 0.0 where the participant is female.
  val target = data.genders.values.map {
    gender => if(gender == 'M') 1.0 else 0.0
  }

  // Build the loss function ready for optimization.
  // We will worry about refactoring this to be more 
  // efficient later.
  def costFunction(parameters:DenseVector[Double]):Double = {
    val xBeta = featureMatrix * parameters
    val expXBeta = exp(xBeta)
    - sum((target :* xBeta) - log1p(expXBeta))
  }

  def costFunctionGradient(parameters:DenseVector[Double])
  :DenseVector[Double] = {
    val xBeta = featureMatrix * parameters
    val probs = sigmoid(xBeta)
    featureMatrix.t * (probs - target)
  }

  val f = new DiffFunction[DenseVector[Double]] {
    def calculate(parameters:DenseVector[Double]) =
      (costFunction(parameters), costFunctionGradient(parameters))
  }

  val optimalParameters = minimize(f, DenseVector(0.0, 0.0, 0.0))

  println(optimalParameters)
  // => DenseVector(-0.0751454743, 2.476293647, 2.23054540)
}
```

这听起来很复杂！让我们一步一步来。在明显的导入之后，我们开始：

```py
object LogisticRegressionHWData extends App {
```

通过扩展内置的 `App` 特质，我们告诉 Scala 将整个对象视为一个 `main` 函数。这仅仅消除了 `def main(args:Array[String])` 的样板代码。然后我们加载数据并将身高和体重缩放到具有 `mean` 为零和标准差为之一的值：

```py
def rescaled(v:DenseVector[Double]) =
  (v - mean(v)) / stddev(v)

val rescaledHeights = rescaled(data.heights)
val rescaledWeights = rescaled(data.weights)
```

`rescaledHeights` 和 `rescaledWeights` 向量将是我们的模型特征。现在我们可以为这个模型构建训练集矩阵。这是一个 (*181 * 3*) 矩阵，其中第 *i*- 行是 `(1, height(i), weight(i))`，对应于第 *i* 个参与者的身高和体重值。我们首先将 `rescaledHeights` 和 `rescaledWeights` 从向量转换为 (*181 * 1*) 矩阵

```py
val rescaledHeightsAsMatrix = rescaledHeights.toDenseMatrix.t
val rescaledWeightsAsMatrix = rescaledWeights.toDenseMatrix.t
```

我们还必须创建一个只包含 *1* 的 (*181 * 1*) 矩阵作为虚拟特征。我们可以使用以下方法来完成：

```py
DenseMatrix.onesDouble
```

现在，我们需要将我们的三个 (*181 * 1*) 矩阵组合成一个形状为 (*181 * 3*) 的单个特征矩阵。我们可以使用 `horzcat` 方法将三个矩阵连接起来：

```py
val featureMatrix = DenseMatrix.horzcat(
  DenseMatrix.onesDouble,
  rescaledHeightsAsMatrix,
  rescaledWeightsAsMatrix
)
```

数据预处理阶段的最后一步是创建目标变量。我们需要将 `data.genders` 向量转换为包含一和零的向量。我们将男性赋值为 1，女性赋值为 0。因此，我们的分类器将预测任何给定的人是男性的概率。我们将使用 `.values.map` 方法，这是 Scala 集合上的 `.map` 方法的等价方法：

```py
val target = data.genders.values.map {
  gender => if(gender == 'M') 1.0 else 0.0
}
```

注意，我们也可以使用我们之前发现的指示函数：

```py
val maleVector = DenseVector.fill(data.genders.size)('M')
val target = I(data.genders :== maleVector)
```

这会导致分配一个临时数组，`maleVector`，因此如果实验中有许多参与者，这可能会增加程序的内存占用。

现在我们有一个表示训练集的矩阵和一个表示目标变量的向量。我们可以写出我们想要最小化的损失函数。如前所述，我们将最小化 ![一个示例 - 逻辑回归](img/image01177.jpeg)。损失函数接受一组线性系数的值作为输入，并返回一个数字，表示这些线性系数的值如何拟合训练数据：

```py
def costFunction(parameters:DenseVector[Double]):Double = {
  val xBeta = featureMatrix * parameters 
  val expXBeta = exp(xBeta)
  - sum((target :* xBeta) - log1p(expXBeta))
}
```

注意，我们使用 `log1p(x)` 来计算 *log(1+x)*。这对于 `x` 的较小值是稳健的，不会下溢。

让我们探索成本函数：

```py
costFunction(DenseVector(0.0, 0.0, 0.0)) // 125.45963968135031
costFunction(DenseVector(0.0, 0.1, 0.1)) // 113.33336518036882
costFunction(DenseVector(0.0, -0.1, -0.1)) // 139.17134594294433
```

我们可以看到，对于稍微正的身高和体重参数值，成本函数略低。这表明对于稍微正的身高和体重值，似然函数更大。这反过来又意味着（正如我们根据图所期望的），身高和体重高于平均值的人更有可能是男性。

我们还需要一个函数来计算损失函数的梯度，因为这有助于优化：

```py
def costFunctionGradient(parameters:DenseVector[Double])
:DenseVector[Double] = {
  val xBeta = featureMatrix * parameters 
  val probs = sigmoid(xBeta)
  featureMatrix.t * (probs - target)
}
```

在定义了损失函数和梯度之后，我们现在可以设置优化：

```py
 val f = new DiffFunction[DenseVector[Double]] {
   def calculate(parameters:DenseVector[Double]) = 
     (costFunction(parameters), costFunctionGradient(parameters))
 }
```

现在剩下的就是运行优化。逻辑回归的成本函数是凸函数（它有一个唯一的极小值），所以在原则上优化起点是不相关的。在实践中，通常从一个系数向量开始，该向量在所有地方都是零（相当于将每个参与者的男性概率赋值为 0.5）：

```py
  val optimalParameters = minimize(f, DenseVector(0.0, 0.0, 0.0))
```

这返回最优参数的向量：

```py
DenseVector(-0.0751454743, 2.476293647, 2.23054540)
```

我们如何解释最优参数的值？身高和体重的系数都是正的，这表明身高和体重较高的人更有可能是男性。

我们还可以直接从系数中得到决策边界（分隔更可能属于女性的（身高，体重）对和更可能属于男性的（身高，体重）对的线）。决策边界是：

![一个示例 - 逻辑回归](img/image01178.jpeg)![一个示例 - 逻辑回归](img/image01179.jpeg)

身高和体重数据（通过均值偏移并按标准差缩放）。橙色线是逻辑回归决策边界。逻辑回归预测边界上方的个体是男性。

# 向可重用代码迈进

在上一节中，我们在单个脚本中执行了所有计算。虽然这对于数据探索来说是不错的，但这意味着我们无法重用我们已经构建的逻辑回归代码。在本节中，我们将开始构建一个机器学习库，你可以在不同的项目中重用它。

我们将逻辑回归算法提取到它自己的类中。我们构建一个`LogisticRegression`类：

```py
import breeze.linalg._
import breeze.numerics._
import breeze.optimize._

class LogisticRegression(
    val training:DenseMatrix[Double], 
    val target:DenseVector[Double])
{
```

该类接受一个表示训练集的矩阵和一个表示目标变量的向量作为输入。注意我们如何将它们分配给`vals`，这意味着它们在类创建时设置，并且将在类销毁之前保持不变。当然，`DenseMatrix`和`DenseVector`对象是可变的，所以`training`和`target`指向的值可能会改变。由于编程最佳实践规定可变状态会使程序行为难以推理，我们将避免利用这种可变性。

让我们添加一个计算成本函数及其梯度的方法：

```py
  def costFunctionAndGradient(coefficients:DenseVector[Double])
  :(Double, DenseVector[Double]) = {
    val xBeta = training * coefficients
    val expXBeta = exp(xBeta)
    val cost = - sum((target :* xBeta) - log1p(expXBeta))
    val probs = sigmoid(xBeta)
    val grad = training.t * (probs - target)
    (cost, grad)
  }
```

我们现在已经准备好运行优化过程，以计算最佳系数，这些系数能够最好地重现训练集。在传统的面向对象语言中，我们可能会定义一个`getOptimalCoefficients`方法，该方法返回一个包含系数的`DenseVector`。然而，Scala 却更加优雅。由于我们已经将`training`和`target`属性定义为`vals`，因此最优系数的可能值集只有一个。因此，我们可以定义一个`val optimalCoefficients = ???`类属性来保存最优系数。问题是这会强制所有计算都在实例构造时发生。这对用户来说可能是意外的，也可能造成浪费：如果用户只对访问成本函数感兴趣，例如，用于最小化的时间将会被浪费。解决方案是使用`lazy val`。这个值只有在客户端代码请求时才会被评估：

```py
lazy val optimalCoefficients = ???
```

为了帮助计算系数，我们将定义一个私有辅助方法：

```py
private def calculateOptimalCoefficients
:DenseVector[Double] = {
  val f = new DiffFunction[DenseVector[Double]] {
    def calculate(parameters:DenseVector[Double]) = 
      costFunctionAndGradient(parameters)
  }

  minimize(f, DenseVector.zerosDouble)
}

lazy val optimalCoefficients = calculateOptimalCoefficients
```

我们已经将逻辑回归重构为它自己的类，我们可以在不同的项目中重用它。

如果我们打算重用身高体重数据，我们可以将其重构为一个自己的类，该类便于数据加载、特征缩放以及任何我们经常重用的其他功能。

# Breeze 的替代方案

Breeze 是线性代数和数值计算方面功能最丰富且易于使用的 Scala 框架。然而，不要仅凭我的话就下结论：尝试其他表格数据库。特别是，我推荐尝试*Saddle*，它提供了一个类似于 pandas 或 R 中的数据框的`Frame`对象。在 Java 领域，*Apache Commons Maths 库*提供了一个非常丰富的数值计算工具包。在第十章（part0097.xhtml#aid-2SG6I1 "第十章. 使用 Spark 进行分布式批量处理"）、第十一章（part0106.xhtml#aid-352RK2 "第十一章. Spark SQL 和 DataFrames"）和第十二章（part0117.xhtml#aid-3FIHQ2 "第十二章. 使用 MLlib 进行分布式机器学习"）中，我们将探讨*Spark*和*MLlib*，它们允许用户运行分布式机器学习算法。

# 摘要

这就结束了我们对 Breeze 的简要概述。我们学习了如何操作基本的 Breeze 数据类型，如何使用它们进行线性代数，以及如何执行凸优化。然后我们运用我们的知识清理了一个真实的数据集，并在其上执行了逻辑回归。

在下一章中，我们将讨论 breeze-viz，这是一个用于 Scala 的绘图库。

# 参考文献

《*统计学习的要素*》，由*Hastie*、*Tibshirani*和*Friedman*著，对机器学习的数学基础进行了清晰、实用的描述。任何希望不仅仅盲目地将机器学习算法作为黑盒应用的人，都应该拥有一本翻阅得破旧的这本书。

《*Scala for Machine Learning*》，由*Patrick R. Nicholas*著，描述了许多有用的机器学习算法在 Scala 中的实际应用。

Breeze 的文档（[`github.com/scalanlp/breeze/wiki/Quickstart`](https://github.com/scalanlp/breeze/wiki/Quickstart)）、API 文档（[`www.scalanlp.org/api/breeze/#package`](http://www.scalanlp.org/api/breeze/#package)）和源代码（[`github.com/scalanlp/breeze`](https://github.com/scalanlp/breeze)）提供了关于 Breeze 的最新文档资源。

# 第三章. 使用 breeze-viz 进行绘图

数据可视化是数据科学的一个基本组成部分。可视化需求可以分为两大类：在开发和新模型验证期间，以及在管道末尾，从数据和模型中提取意义，为外部利益相关者提供洞察。

这两种可视化类型相当不同。在数据探索和模型开发阶段，可视化库最重要的特性是它的易用性。它应该尽可能少地执行步骤，从拥有数字数组（或 CSV 文件或在数据库中）的数据到在屏幕上显示数据。图表的寿命也相当短：一旦数据科学家从图表或可视化中学到了所有他能学到的知识，它通常就会被丢弃。相比之下，当为外部利益相关者开发可视化小部件时，人们愿意为了更大的灵活性而容忍增加的开发时间。可视化可以具有相当长的寿命，特别是如果底层数据随时间变化的话。

在 Scala 中，用于第一种可视化的首选工具是 breeze-viz。当为外部利益相关者开发可视化时，基于 Web 的可视化（如 D3）和 Tableau 往往更受欢迎。

在本章中，我们将探索 breeze-viz。在第十四章，*使用 D3 和 Play Framework 进行可视化*中，我们将学习如何为 JavaScript 可视化构建 Scala 后端。

Breeze-viz 是（无需猜测）Breeze 的可视化库。它封装了**JFreeChart**，一个非常流行的 Java 图表库。Breeze-viz 仍然处于实验阶段。特别是，它比 Python 中的 matplotlib、R 或 MATLAB 的功能要少得多。尽管如此，breeze-viz 允许访问底层的 JFreeChart 对象，因此用户可以始终回退到直接编辑这些对象。breeze-viz 的语法受到了 MATLAB 和 matplotlib 的启发。

# 深入了解 Breeze

让我们开始吧。我们将在 Scala 控制台中工作，但与这个例子类似的程序可以在本章对应的示例中的`BreezeDemo.scala`文件中找到。创建一个包含以下行的`build.sbt`文件：

```py
scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2"
)
```

启动`sbt`控制台：

```py
$ sbt console

scala> import breeze.linalg._
import breeze.linalg._

scala> import breeze.plot._
import breeze.plot._

scala> import breeze.numerics._
import breeze.numerics._

```

让我们先绘制一个 sigmoid 曲线，![深入 Breeze](img/image01180.jpeg)。我们将首先使用 Breeze 生成数据。回想一下，`linspace`方法创建了一个在两个值之间均匀分布的双精度浮点数向量：

```py
scala> val x = linspace(-4.0, 4.0, 200)
x: DenseVector[Double] = DenseVector(-4.0, -3.959798...

scala> val fx = sigmoid(x)
fx: DenseVector[Double] = DenseVector(0.0179862099620915,...

```

现在我们已经准备好了用于绘制的数据。第一步是创建一个图形：

```py
scala> val fig = Figure()
fig: breeze.plot.Figure = breeze.plot.Figure@37e36de9

```

这创建了一个空的 Java Swing 窗口（它可能出现在你的任务栏或等效位置）。一个图形可以包含一个或多个绘图。让我们向我们的图形添加一个绘图：

```py
scala> val plt = fig.subplot(0)
plt: breeze.plot.Plot = breeze.plot.Plot@171c2840

```

现在，让我们忽略传递给`.subplot`的`0`作为参数。我们可以向我们的`plot`添加数据点：

```py
scala> plt += plot(x, fx)
breeze.plot.Plot = breeze.plot.Plot@63d6a0f8

```

`plot`函数接受两个参数，对应于要绘制的数据系列的*x*和*y*值。要查看更改，您需要刷新图形：

```py
scala> fig.refresh()

```

现在看看 Swing 窗口。你应该看到一个漂亮的 sigmoid 曲线，类似于下面的一个。在窗口上右键单击可以让你与绘图交互并将图像保存为 PNG：

![深入 Breeze](img/image01181.jpeg)

您还可以按以下方式程序化地保存图像：

```py
scala> fig.saveas("sigmoid.png")

```

Breeze-viz 目前仅支持导出为 PNG 格式。

# 自定义绘图

现在我们图表上有一条曲线。让我们再添加几条：

```py
scala> val f2x = sigmoid(2.0*x)
f2x: breeze.linalg.DenseVector[Double] = DenseVector(3.353501304664E-4...

scala> val f10x = sigmoid(10.0*x)
f10x: breeze.linalg.DenseVector[Double] = DenseVector(4.24835425529E-18...

scala> plt += plot(x, f2x, name="S(2x)")
breeze.plot.Plot = breeze.plot.Plot@63d6a0f8

scala> plt += plot(x, f10x, name="S(10x)")
breeze.plot.Plot = breeze.plot.Plot@63d6a0f8

scala> fig.refresh()

```

现在查看这个图，你应该看到三种不同颜色的所有三条曲线。注意，我们在添加到绘图时命名了数据系列，使用 `name=""` 关键字参数。要查看名称，我们必须设置 `legend` 属性：

```py
scala> plt.legend = true

```

![自定义绘图](img/image01182.jpeg)

我们的绘图还有很多可以改进的地方。让我们首先将 *x* 轴的范围限制，以移除绘图两侧的空白带：

```py
scala> plt.xlim = (-4.0, 4.0)
plt.xlim: (Double, Double) = (-4.0,4.0)

```

现在，请注意，虽然 *x* 刻度合理地分布，但只有两个 *y* 刻度：在 *0* 和 *1* 处。每增加 *0.1* 就有一个刻度将很有用。Breeze 没有提供直接设置此功能的方法。相反，它暴露了当前绘图所属的底层 JFreeChart 轴对象：

```py
scala> plt.yaxis
org.jfree.chart.axis.NumberAxis = org.jfree.chart.axis.NumberAxis@0

```

`Axis` 对象支持一个 `.setTickUnit` 方法，允许我们设置刻度间距：

```py
scala> import org.jfree.chart.axis.NumberTickUnit
import org.jfree.chart.axis.NumberTickUnit

scala> plt.yaxis.setTickUnit(new NumberTickUnit(0.1))

```

JFreeChart 允许对 `Axis` 对象进行广泛的定制。有关可用方法的完整列表，请参阅 JFreeChart 文档 ([`www.jfree.org/jfreechart/api/javadoc/org/jfree/chart/axis/Axis.html`](http://www.jfree.org/jfreechart/api/javadoc/org/jfree/chart/axis/Axis.html))。

让我们在 *x=0* 处添加一条垂直线，在 *f(x)=1* 处添加一条水平线。我们需要访问底层 JFreeChart 绘图来添加这些线条。这在我们的 Breeze `Plot` 对象中作为 `.plot` 属性（有些令人困惑）可用：

```py
scala> plt.plot
org.jfree.chart.plot.XYPlot = org.jfree.chart.plot.XYPlot@17e4db6c

```

我们可以使用 `.addDomainMarker` 和 `.addRangeMarker` 方法向 JFreeChart `XYPlot` 对象添加垂直和水平线条：

```py
scala> import org.jfree.chart.plot.ValueMarker
import org.jfree.chart.plot.ValueMarker

scala> plt.plot.addDomainMarker(new ValueMarker(0.0))

scala> plt.plot.addRangeMarker(new ValueMarker(1.0))

```

让我们也给坐标轴添加标签：

```py
scala> plt.xlabel = "x"
plt.xlabel: String = x

scala> plt.ylabel = "f(x)"
plt.ylabel: String = f(x)

```

如果你已经运行了所有这些命令，你应该有一个看起来像这样的图形：

![自定义绘图](img/image01183.jpeg)

我们现在知道如何自定义图形的基本构建块。下一步是学习如何更改曲线的绘制方式。

# 自定义线条类型

到目前为止，我们只是使用默认设置绘制了线条。Breeze 允许我们自定义线条的绘制方式，至少在一定程度上。

对于这个例子，我们将使用在 第二章 中讨论的高度-体重数据，*使用 Breeze 操作数据*。我们将在这里使用 Scala shell 进行演示，但你将在 `BreezeDemo.scala` 程序中找到一个遵循示例 shell 会话的程序。

本章的代码示例附带一个用于加载数据的模块，`HWData.scala`，它从 CSV 文件中加载数据：

```py
scala> val data = HWData.load
data: HWData = HWData [ 181 rows ]

scala> data.heights
breeze.linalg.DenseVector[Double] = DenseVector(182.0, ...

scala> data.weights
breeze.linalg.DenseVector[Double] = DenseVector(77.0, 58.0...

```

让我们创建一个身高与体重的散点图：

```py
scala> val fig = Figure("height vs. weight")
fig: breeze.plot.Figure = breeze.plot.Figure@743f2558

scala> val plt = fig.subplot(0)
plt: breeze.plot.Plot = breeze.plot.Plot@501ea274

scala> plt += plot(data.heights, data.weights, '+',         colorcode="black")
breeze.plot.Plot = breeze.plot.Plot@501ea274

```

这产生了身高-体重数据的散点图：

![自定义线条类型](img/image01184.jpeg)

注意，我们传递了第三个参数给`plot`方法，`'+'`。这控制了绘图样式。截至本文写作时，有三种可用的样式：`'-'`（默认），`'+'`和`'.'`。尝试这些样式以查看它们的作用。最后，我们传递一个`colorcode="black"`参数来控制线的颜色。这可以是颜色名称或 RGB 三元组，以字符串形式编写。因此，要绘制红色点，我们可以传递`colorcode="[255,0,0]"`。

观察身高体重图，身高和体重之间显然存在趋势。让我们尝试通过数据点拟合一条直线。我们将拟合以下函数：

![自定义线型](img/image01185.jpeg)

### 注意

科学文献表明，拟合类似![自定义线型](img/image01186.jpeg)的东西会更好。如果你愿意，应该可以轻松地将二次线拟合到数据上。

我们将使用 Breeze 的最小二乘函数来找到`a`和`b`的值。`leastSquares`方法期望一个特征矩阵的输入和一个目标向量，就像我们在上一章中定义的`LogisticRegression`类一样。回想一下，在第二章中，*使用 Breeze 操作数据*，当我们为逻辑回归分类准备训练集时，我们引入了一个虚拟特征，每个参与者都是 1，以提供*y*截距的自由度。我们在这里将使用相同的方法。因此，我们的特征矩阵包含两列——一列在所有地方都是`1`，另一列是身高：

```py
scala> val features = DenseMatrix.horzcat(
 DenseMatrix.onesDouble,
 data.heights.toDenseMatrix.t
)
features: breeze.linalg.DenseMatrix[Double] =
1.0  182.0
1.0  161.0
1.0  161.0
1.0  177.0
1.0  157.0
...

scala> import breeze.stats.regression._
import breeze.stats.regression._

scala> val leastSquaresResult = leastSquares(features, data.weights)
leastSquaresResult: breeze.stats.regression.LeastSquaresRegressionResult = <function1>

```

`leastSquares`方法返回一个`LeastSquareRegressionResult`实例，它包含一个`coefficients`属性，包含最佳拟合数据的系数：

```py
scala> leastSquaresResult.coefficients
breeze.linalg.DenseVector[Double] = DenseVector(-131.042322, 1.1521875)

```

因此，最佳拟合线是：

![自定义线型](img/image01187.jpeg)

让我们提取系数。一种优雅的方法是使用 Scala 的模式匹配功能：

```py
scala> val Array(a, b) = leastSquaresResult.coefficients.toArray
a: Double = -131.04232269750622
b: Double = 1.1521875435418725

```

通过编写`val Array(a, b) = ...`，我们告诉 Scala 表达式右侧是一个包含两个元素的数组，并将该数组的第一个元素绑定到值`a`，第二个元素绑定到`b`。参见附录，*模式匹配和提取器*，以了解模式匹配的讨论。

我们现在可以将最佳拟合线添加到我们的图中。我们首先生成均匀间隔的虚拟身高值：

```py
scala> val dummyHeights = linspace(min(data.heights), max(data.heights), 200)
dummyHeights: breeze.linalg.DenseVector[Double] = DenseVector(148.0, ...

scala> val fittedWeights = a :+ (b :* dummyHeights)
fittedWeights: breeze.linalg.DenseVector[Double] = DenseVector(39.4814...

scala> plt += plot(dummyHeights, fittedWeights, colorcode="red")
breeze.plot.Plot = breeze.plot.Plot@501ea274

```

让我们也将最佳拟合线的方程添加到图中作为注释。我们首先生成标签：

```py
scala> val label = f"weight = $a%.4f + $b%.4f * height"
label: String = weight = -131.0423 + 1.1522 * height

```

要添加注释，我们必须访问底层的 JFreeChart 绘图：

```py
scala> import org.jfree.chart.annotations.XYTextAnnotation
import org.jfree.chart.annotations.XYTextAnnotation

scala> plt.plot.addAnnotation(new XYTextAnnotation(label, 175.0, 105.0))

```

`XYTextAnnotation` 构造函数接受三个参数：注释字符串和定义注释在图上中心的 (*x*, *y*) 坐标对。注释的坐标以数据的坐标系表示。因此，调用 `new XYTextAnnotation(label, 175.0, 105.0)` 将生成一个中心在对应 175 厘米高度和 105 公斤重量的点的注释：

![自定义线型](img/image01188.jpeg)

# 更高级的散点图

Breeze-viz 提供了一个 `scatter` 函数，为散点图添加了显著的定制程度。特别是，我们可以使用标记点的大小和颜色向图中添加额外的信息维度。

`scatter` 函数接受其前两个参数为 *x* 和 *y* 点的集合。第三个参数是一个函数，它将整数 `i` 映射到表示 *第 i 个* 点大小的 `Double` 值。点的大小以 *x* 轴的单位来衡量。如果你有 Scala 集合或 Breeze 向量的尺寸，你可以使用该集合的 `apply` 方法作为该函数。让我们看看这在实践中是如何工作的。

与之前的示例一样，我们将使用 REPL，但你可以在 `BreezeDemo.scala` 中找到一个示例程序：

```py
scala> val fig = new Figure("Advanced scatter example")
fig: breeze.plot.Figure = breeze.plot.Figure@220821bc

scala> val plt = fig.subplot(0)
plt: breeze.plot.Plot = breeze.plot.Plot@668f8ae0

scala> val xs = linspace(0.0, 1.0, 100)
xs: breeze.linalg.DenseVector[Double] = DenseVector(0.0, 0.010101010101010102, 0.0202 ...

scala> val sizes = 0.025 * DenseVector.rand(100) // random sizes
sizes: breeze.linalg.DenseVector[Double] = DenseVector(0.014879265631723166, 0.00219551...

scala> plt += scatter(xs, xs :^ 2.0, sizes.apply)
breeze.plot.Plot = breeze.plot.Plot@668f8ae0

```

选择自定义颜色的工作方式类似：我们传递一个 `colors` 参数，它将整数索引映射到 `java.awt.Paint` 对象。直接使用这些可能比较麻烦，所以 Breeze 提供了一些默认调色板。例如，`GradientPaintScale` 将给定域内的双精度值映射到均匀的颜色渐变。让我们将范围 `0.0` 到 `1.0` 的双精度值映射到红色和绿色之间的颜色：

```py
scala> val palette = new GradientPaintScale(
 0.0, 1.0, PaintScale.RedToGreen)
palette: breeze.plot.GradientPaintScale[Double] = <function1>

scala> palette(0.5) // half-way between red and green
java.awt.Paint = java.awt.Color[r=127,g=127,b=0]

scala> palette(1.0) // green
java.awt.Paint = java.awt.Color[r=0,g=254,b=0]

```

除了 `GradientPaintScale`，breeze-viz 还提供了一个 `CategoricalPaintScale` 类用于分类调色板。有关不同调色板的概述，请参阅 `PaintScale.scala` 源文件，位于 `scala`：[`github.com/scalanlp/breeze/blob/master/viz/src/main/scala/breeze/plot/PaintScale.scala`](https://github.com/scalanlp/breeze/blob/master/viz/src/main/scala/breeze/plot/PaintScale.scala)。

让我们利用我们新获得的知识来绘制一个多彩散点图。我们将假设与上一个示例相同的初始化。我们将为每个点分配一个随机颜色：

```py
scala> val palette = new GradientPaintScale(0.0, 1.0, PaintScale.MaroonToGold)
palette: breeze.plot.GradientPaintScale[Double] = <function1>

scala> val colors = DenseVector.rand(100).mapValues(palette)
colors: breeze.linalg.DenseVector[java.awt.Paint] = DenseVector(java.awt.Color[r=162,g=5,b=0], ...

scala> plt += scatter(xs, xs :^ 2.0, sizes.apply, colors.apply)
breeze.plot.Plot = breeze.plot.Plot@8ff7e27

```

![更高级的散点图](img/image01189.jpeg)

# 多图示例 – 散点矩阵图

在本节中，我们将学习如何在同一图中拥有多个图。

允许在同一图中绘制多个图的关键新方法是 `fig.subplot(nrows, ncols, plotIndex)`。这个方法是我们迄今为止一直在使用的 `fig.subplot` 方法的重载版本，它既设置了图中的行数和列数，又返回一个特定的子图。它接受三个参数：

+   `nrows`：图中子图行数

+   `ncols`：图中子图列数

+   `plotIndex`：要返回的图的索引

熟悉 MATLAB 或 matplotlib 的用户会注意到`.subplot`方法与这些框架中的同名方法相同。这可能会显得有点复杂，所以让我们看一个例子（你可以在`BreezeDemo.scala`中找到这个代码）：

```py
import breeze.plot._

def subplotExample {
  val data = HWData.load
  val fig = new Figure("Subplot example")

  // upper subplot: plot index '0' refers to the first plot
  var plt = fig.subplot(2, 1, 0)
  plt += plot(data.heights, data.weights, '.')

  // lower subplot: plot index '1' refers to the second plot
  plt = fig.subplot(2, 1, 1)
  plt += plot(data.heights, data.reportedHeights, '.', colorcode="black")

  fig.refresh
}
```

运行此示例将生成以下图表：

![多图示例 – 散点图矩阵图表](img/image01190.jpeg)

现在我们已经基本掌握了如何在同一张图上添加多个子图的方法，让我们做一些更有趣的事情。我们将编写一个类来绘制散点图矩阵。这些对于探索不同特征之间的相关性非常有用。

如果你不太熟悉散点图矩阵，请查看本节末尾的图表，以了解我们正在构建的内容。想法是为每对特征构建一个散点图的正方形矩阵。矩阵中的元素（*i*，*j*）是特征*i*与特征*j*的散点图。由于一个变量与其自身的散点图用途有限，通常会在对角线上绘制每个特征的直方图。最后，由于特征*i*与特征*j*的散点图包含与特征*j*与特征*i*的散点图相同的信息，通常只绘制矩阵的上三角或下三角。

让我们先编写用于单个图表的函数。这些函数将接受一个`Plot`对象，它引用正确的子图，以及要绘制的数据的向量：

```py
import breeze.plot._
import breeze.linalg._

class ScatterplotMatrix(val fig:Figure) {

  /** Draw the histograms on the diagonal */
  private def plotHistogram(plt:Plot)(
  data:DenseVector[Double], label:String) {
     plt += hist(data)
     plt.xlabel = label
  }

  /** Draw the off-diagonal scatter plots */
  private def plotScatter(plt:Plot)(
    xdata:DenseVector[Double],
    ydata:DenseVector[Double],
    xlabel:String,
    ylabel:String) {
      plt += plot(xdata, ydata, '.')
      plt.xlabel = xlabel
      plt.ylabel = ylabel
  }

...
```

注意到使用了`hist(data)`来绘制直方图。`hist`函数的参数必须是一个数据点的向量。`hist`方法会将这些数据点分箱，并以直方图的形式表示它们。

现在我们已经有了绘制单个图表的机制，我们只需要将所有东西连接起来。难点是知道如何根据矩阵中的给定行和列位置选择正确的子图。我们可以通过调用`fig.subplot(nrows, ncolumns, plotIndex)`来选择单个图表，但将（*行*，*列*）索引对转换为单个`plotIndex`并不明显。图表按顺序编号，首先是左到右，然后是上到下：

```py
0 1 2 3
4 5 6 7
...
```

让我们编写一个简短的功能来选择一个在（*行*，*列*）索引对上的图表：

```py
  private def selectPlot(ncols:Int)(irow:Int, icol:Int):Plot = {
    fig.subplot(ncols, ncols, (irow)*ncols + icol)
  }
```

现在我们已经可以绘制矩阵图本身了：

```py
  /** Draw a scatterplot matrix.
    *
    * This function draws a scatterplot matrix of the correlation
    * between each pair of columns in `featureMatrix`.
    *
    * @param featureMatrix A matrix of features, with each column
    *   representing a feature.
    * @param labels Names of the features.
    */
  def plotFeatures(featureMatrix:DenseMatrix[Double], labels:List[String]) {
    val ncols = featureMatrix.cols
    require(ncols == labels.size,
      "Number of columns in feature matrix "+ "must match length of labels"
    )
    fig.clear
    fig.subplot(ncols, ncols, 0)

    (0 until ncols) foreach { irow =>
      val p = selectPlot(ncols)(irow, irow)
      plotHistogram(p)(featureMatrix(::, irow), labels(irow))

      (0 until irow) foreach { icol =>
        val p = selectPlot(ncols)(irow, icol)
        plotScatter(p)(
          featureMatrix(::, irow),
          featureMatrix(::, icol),
          labels(irow),
          labels(icol)
        )
      }
    }
  }
}
```

让我们为我们的类编写一个示例。我们将再次使用身高体重数据：

```py
import breeze.linalg._
import breeze.numerics._
import breeze.plot._

object ScatterplotMatrixDemo extends App {

  val data = HWData.load
  val m = new ScatterplotMatrix(Figure("Scatterplot matrix demo"))

  // Make a matrix with three columns: the height, weight and
  // reported weight data.
  val featureMatrix = DenseMatrix.horzcat(
    data.heights.toDenseMatrix.t,
    data.weights.toDenseMatrix.t,
    data.reportedWeights.toDenseMatrix.t
  )
  m.plotFeatures(featureMatrix,List("height", "weight", "reportedWeights"))

}
```

在 SBT 中运行此代码将生成以下图表：

![多图示例 – 散点图矩阵图表](img/image01191.jpeg)

# 无文档管理

Breeze-viz 的文档不幸地相当不完善。这可能会使学习曲线变得有些陡峭。幸运的是，它仍然是一个非常小的项目：在撰写本文时，只有十个源文件（[`github.com/scalanlp/breeze/tree/master/viz/src/main/scala/breeze/plot`](https://github.com/scalanlp/breeze/tree/master/viz/src/main/scala/breeze/plot)）。了解 breeze-viz 确切做什么的一个好方法是阅读源代码。例如，要查看 `Plot` 对象上可用的方法，请阅读源文件 `Plot.scala`。如果您需要 Breeze 提供的功能之外的功能，请查阅 JFreeChart 的文档，以了解您是否可以通过访问底层的 JFreeChart 对象来实现所需的功能。

# Breeze-viz 参考文档

在编程书中编写参考是一项危险的活动：您很快就会过时。尽管如此，鉴于 breeze-viz 的文档很少，这一部分变得更加相关——与不存在的东西竞争更容易。请带着一点盐来接受这一部分，如果本节中的某个命令不起作用，请查看源代码：

| 命令 | 描述 |
| --- | --- |
| `plt += plot(xs, ys)` | 这将一系列 (`xs`, `ys`) 值绘制出来。`xs` 和 `ys` 值必须是类似集合的对象（例如 Breeze 向量、Scala 数组或列表）。 |
| `plt += scatter(xs, ys, size)` | `plt += scatter(xs, ys, size, color)` | 这将一系列 (`xs`, `ys`) 值作为散点图绘制。`size` 参数是一个 `(Int) => Double` 函数，它将点的索引映射到其大小（与 *x* 轴相同的单位）。`color` 参数是一个 `(Int) => java.awt.Paint` 函数，它将整数映射到颜色。阅读 *更高级的散点图* 部分以获取更多详细信息。 |
| `plt += hist(xs)` | `plt += hist(xs, bins=10)` | 这将 `xs` 分组并绘制直方图。`bins` 参数控制分组的数量。 |
| `plt += image(mat)` | 这将图像或矩阵绘制出来。`mat` 参数应该是 `Matrix[Double]`。有关详细信息，请阅读 `breeze.plot` 中的 `package.scala` 源文件（[`github.com/scalanlp/breeze/blob/master/viz/src/main/scala/breeze/plot/package.scala`](https://github.com/scalanlp/breeze/blob/master/viz/src/main/scala/breeze/plot/package.scala)）。 |

总结 `plot` 对象上可用的选项也是有用的：

| 属性 | 描述 |
| --- | --- |
| `plt.xlabel = "x-label"` | `plt.ylabel = "y-label"` | 这设置轴标签 |
| `plt.xlim = (0.0, 1.0)` | `plt.ylim = (0.0, 1.0)` | 这设置轴的最大和最小值 |
| `plt.logScaleX = true` | `plt.logScaleY = true` | 这将轴切换到对数刻度 |
| `plt.title = "title"` | 这设置绘图标题 |

# Breeze-viz 之外的数据可视化

Scala 中用于数据可视化的其他工具正在出现：基于 IPython 笔记本的 Spark 笔记本([`github.com/andypetrella/spark-notebook#description`](https://github.com/andypetrella/spark-notebook#description))和 Apache Zeppelin([`zeppelin.incubator.apache.org`](https://zeppelin.incubator.apache.org))。这两个都依赖于 Apache Spark，我们将在本书的后面部分探讨。

# 摘要

在本章中，我们学习了如何使用 breeze-viz 绘制简单的图表。在本书的最后一章中，我们将学习如何使用 JavaScript 库构建交互式可视化。

接下来，我们将学习关于基本 Scala 并发构造的知识——特别是并行集合。

# 第四章。并行集合和 Futures

数据科学通常涉及处理中等或大量数据。由于之前单个 CPU 速度的指数增长已经放缓，而数据量仍在增加，因此有效地利用计算机必须涉及并行计算。

在本章中，我们将探讨在单台计算机上并行化计算和数据处理的方法。几乎所有的计算机都有多个处理单元，将这些核心的计算分布在这些核心上可以是一种加快中等规模计算的有效方式。

在单个芯片上并行化计算适合涉及千兆字节或几太字节数据的计算。对于更大的数据流，我们必须求助于将计算并行分布在多台计算机上。我们将在第十章，*使用 Spark 的分布式批量处理*中讨论 Apache Spark，这是一个并行数据处理框架。

在这本书中，我们将探讨在单台机器上利用并行架构的三个常见方法：并行集合、futures 和 actors。我们将在本章中考虑前两种，并将 actors 的研究留给第九章，*使用 Akka 的并发*。

# 并行集合

并行集合提供了一种极其简单的方式来并行化独立任务。由于读者熟悉 Scala，他们将知道许多任务可以表述为集合上的操作，如*map*、*reduce*、*filter*或*groupBy*。并行集合是 Scala 集合的一种实现，它将这些操作并行化以在多个线程上运行。

让我们从例子开始。我们想要计算句子中每个字母出现的频率：

```py
scala> val sentence = "The quick brown fox jumped over the lazy dog"
sentence: String = The quick brown fox jumped ...

```

首先，让我们将我们的句子从字符串转换为字符向量：

```py
scala> val characters = sentence.toVector
Vector[Char] = Vector(T, h, e,  , q, u, i, c, k, ...)

```

现在，我们可以将`characters`转换为*并行*向量，即`ParVector`。为此，我们使用`par`方法：

```py
scala> val charactersPar = characters.par
ParVector[Char] = ParVector(T, h, e,  , q, u, i, c, k,  , ...)

```

`ParVector`集合支持与常规向量相同的操作，但它们的方法是在多个线程上并行执行的。

首先，让我们从`charactersPar`中过滤掉空格：

```py
scala> val lettersPar = charactersPar.filter { _ != ' ' }
ParVector[Char] = ParVector(T, h, e, q, u, i, c, k, ...)

```

注意 Scala 如何隐藏执行细节。`filter`操作使用了多个线程，而你几乎都没有注意到！并行向量的接口和行为与其串行对应物相同，除了我们将在下一节中探讨的一些细节。

现在我们使用`toLower`函数将字母转换为小写：

```py
scala> val lowerLettersPar = lettersPar.map { _.toLower }
ParVector[Char] = ParVector(t, h, e, q, u, i, c, k, ...)

```

与之前一样，`map`方法是在并行中应用的。为了找到每个字母的出现频率，我们使用`groupBy`方法将字符分组到包含该字符所有出现的向量中：

```py
scala> val intermediateMap = lowerLettersPar.groupBy(identity)
ParMap[Char,ParVector[Char]] = ParMap(e -> ParVector(e, e, e, e), ...)

```

注意`groupBy`方法创建了一个`ParMap`实例，它是不可变映射的并行等价物。为了得到每个字母出现的次数，我们在`intermediateMap`上执行`mapValues`调用，将每个向量替换为其长度：

```py
scala> val occurenceNumber = intermediateMap.mapValues { _.length }
ParMap[Char,Int] = ParMap(e -> 4, x -> 1, n -> 1, j -> 1, ...)

```

恭喜！我们用几行代码编写了一个多线程算法，用于查找每个字母在文本中出现的频率。你应该会发现将其修改为查找文档中每个单词的出现频率非常简单，这是分析文本数据时常见的预处理问题。

并行集合使得并行化某些操作流程变得非常容易：我们只需在`characters`向量上调用`.par`。所有后续操作都被并行化。这使得从串行实现切换到并行实现变得非常容易。

## 并行集合的限制

并行集合的部分力量和吸引力在于它们提供了与它们的串行对应物相同的接口：它们有`map`方法、`foreach`方法、`filter`方法等等。总的来说，这些方法在并行集合上的工作方式与在串行集合上相同。然而，也有一些值得注意的注意事项。其中最重要的一点与副作用有关。如果在并行集合上的操作有副作用，这可能会导致竞争条件：一种最终结果取决于线程执行操作顺序的情况。

集合中的副作用最常见的情况是我们更新集合外部定义的变量。为了给出一个意外的行为的不平凡例子，让我们定义一个`count`变量，并使用并行范围将其增加一千次：

```py
scala> var count = 0
count: Int = 0

scala> (0 until 1000).par.foreach { i => count += 1 }

scala> count
count: Int = 874 // not 1000!

```

这里发生了什么？传递给`foreach`的函数有一个副作用：它增加`count`，这是一个在函数作用域之外的变量。这是一个问题，因为`+=`运算符是一系列两个操作：

+   获取`count`的值并将其加一

+   将结果赋值回`count`

要理解这为什么会引起意外的行为，让我们想象`foreach`循环已经被并行化到两个线程上。**线程 A**可能在变量为**832**时读取**count**变量，并将其加一得到**833**。在它有时间将**833**重新赋值给**count**之前，**线程 B**读取**count**，仍然为**832**，然后加一得到**833**。然后**线程 A**将**833**赋值给**count**。接着**线程 B**也将**833**赋值给**count**。我们进行了两次更新，但只增加了计数一次。问题出现是因为`+=`可以被分解成两个指令：它不是**原子性**的。这为线程提供了交错操作的空间：

![并行集合的限制](img/image01192.jpeg)

竞态条件的解剖：线程 A 和线程 B 都在尝试并发更新`count`，导致其中一个更新被覆盖。`count`的最终值是 833 而不是 834。

为了给出一个稍微更现实的例子，说明非原子性引起的问题，让我们看看一种不同的方法来计算句子中每个字母出现的频率。我们在循环外部定义一个可变的`Char -> Int`哈希表。每次我们遇到一个字母时，我们就在映射中增加相应的整数：

```py
scala> import scala.collection.mutable
import scala.collection.mutable

scala> val occurenceNumber = mutable.Map.empty[Char, Int]
occurenceNumber: mutable.Map[Char,Int] = Map()

scala> lowerLettersPar.foreach { c => 
 occurenceNumber(c) = occurenceNumber.getOrElse(c, 0) + 1
}

scala> occurenceNumber('e') // Should be 4
Int = 2

```

这种差异是由于`foreach`循环中操作的非原子性造成的。

通常，避免在集合的高阶函数中引入副作用是一个好习惯。它们使得代码更难以理解，并阻止从串行集合切换到并行集合。避免暴露可变状态也是一个好习惯：不可变对象可以在线程之间自由共享，并且不会受到副作用的影响。

并行集合的另一个限制出现在归约（或折叠）操作中。用于组合项的功能必须是**结合律**的。例如：

```py
scala> (0 until 1000).par.reduce {_ - _ } // should be -499500
Int = 63620

```

**减号**运算符，`–`，不是结合律的。连续操作应用的顺序很重要：`(a – b) – c`不等于`a – (b – c)`。用于归约并行集合的功能必须是结合律的，因为归约发生的顺序与集合的顺序无关。

## 错误处理

在单线程程序中，异常处理相对简单：如果发生异常，函数可以处理它或将其升级。当引入并行性时，这并不那么明显：一个线程可能会失败，但其他线程可能仍然成功返回。

如果并行集合方法在任何一个元素上失败，它们将抛出异常，就像它们的串行对应物一样：

```py
scala> Vector(2, 0, 5).par.map { 10 / _ }
java.lang.ArithmeticException: / by zero
...

```

有时候我们并不希望这种行为。例如，我们可能正在使用并行集合并行检索大量网页。我们可能不介意如果有一些页面无法获取。

Scala 的`Try`类型是为了沙盒化可能会抛出异常的代码而设计的。它与`Option`类似，因为它是一个单元素容器：

```py
scala> import scala.util._
import scala.util._

scala> Try { 2 + 2 }
Try[Int] = Success(4)

```

与表示表达式是否有有用值的 `Option` 类型不同，`Try` 类型表示表达式是否可以在不抛出异常的情况下执行。它具有以下两个值：

+   `尝试 { 2 + 2 } == Success(4)` 如果 `Try` 语句中的表达式成功评估

+   `尝试 { 2 / 0 } == Failure(java.lang.ArithmeticException: / by zero)` 如果 `Try` 块中的表达式导致异常

通过一个例子来解释会更清晰。为了看到 `Try` 类型的实际应用，我们将尝试以容错的方式获取网页。我们将使用内置的 `Source.fromURL` 方法来获取网页并打开页面内容的迭代器。如果它无法获取网页，它将抛出一个错误：

```py
scala> import scala.io.Source
import scala.io.Source

scala> val html = Source.fromURL("http://www.google.com")
scala.io.BufferedSource = non-empty iterator

scala> val html = Source.fromURL("garbage")
java.net.MalformedURLException: no protocol: garbage
...

```

而不是让表达式传播出去并使我们的代码崩溃，我们可以将 `Source.fromURL` 的调用包裹在 `Try` 中：

```py
scala> Try { Source.fromURL("http://www.google.com") }
Try[BufferedSource] = Success(non-empty iterator)

scala> Try { Source.fromURL("garbage") }
Try[BufferedSource] = Failure(java.net.MalformedURLException: no protocol: garbage)

```

为了看到我们的 `Try` 语句的力量，现在让我们以容错的方式并行检索 URL 列表：

```py
scala> val URLs = Vector("http://www.google.com", 
 "http://www.bbc.co.uk",
 "not-a-url"
)
URLs: Vector[String] = Vector(http://www.google.com, http://www.bbc.co.uk, not-a-url)

scala> val pages = URLs.par.map { url =>
 url -> Try { Source.fromURL(url) } 
}
pages: ParVector[(String, Try[BufferedSource])] = ParVector((http://www.google.com,Success(non-empty iterator)), (http://www.bbc.co.uk,Success(non-empty iterator)), (not-a-url,Failure(java.net.MalformedURLException: no protocol: not-a-url)))

```

然后，我们可以使用 `collect` 语句对成功获取的页面进行操作。例如，获取每个页面的字符数：

```py
scala> pages.collect { case(url, Success(it)) => url -> it.size }
ParVector[(String, Int)] = ParVector((http://www.google.com,18976), (http://www.bbc.co.uk,132893))

```

通过充分利用 Scala 的内置 `Try` 类和并行集合，我们只用几行代码就构建了一个容错、多线程的 URL 检索器。（与那些在代码示例前加上 '为了清晰起见省略错误处理' 的 Java/C++ 书籍相比。）

### 小贴士

**`Try` 类型与 try/catch 语句的比较**

具有命令式或面向对象背景的程序员会更熟悉用于处理异常的 try/catch 块。我们可以通过将获取 URL 的代码包裹在 try 块中，在调用引发异常时返回 null 来实现类似的功能。

然而，除了更冗长之外，返回 null 的效果也不太令人满意：我们失去了有关异常的所有信息，而 null 的表达性不如 `Failure(exception)`。此外，返回 `Try[T]` 类型迫使调用者考虑函数可能失败的可能性，通过将这种可能性编码在返回值的类型中。相比之下，只返回 `T` 并用 null 值编码失败，允许调用者忽略失败，从而增加了在程序中完全不同的点上抛出令人困惑的 `NullPointerException` 的可能性。

简而言之，`Try[T]` 只是一种更高阶的类型，就像 `Option[T]` 或 `List[T]` 一样。将失败的可能性以与代码中其他部分相同的方式处理，增加了程序的一致性，并鼓励程序员明确处理异常的可能性。

## 设置并行级别

到目前为止，我们将并行集合视为黑盒：将 `par` 添加到普通集合中，所有操作都会并行执行。通常，我们可能希望对任务执行的细节有更多的控制。

内部，并行集合通过在多个线程上分配操作来工作。由于线程共享内存，并行集合不需要复制任何数据。更改并行集合可用的线程数将更改用于执行任务的 CPU 数量。

并行集合有一个`tasksupport`属性，用于控制任务执行：

```py
scala> val parRange = (0 to 100).par
parRange: ParRange = ParRange(0, 1, 2, 3, 4, 5,...

scala> parRange.tasksupport
TaskSupport = scala.collection.parallel.ExecutionContextTaskSupport@311a0b3e

scala> parRange.tasksupport.parallelismLevel
Int = 8 // Number of threads to be used

```

集合的任务支持对象是一个*执行上下文*，这是一个能够在一个单独的线程中执行 Scala 表达式的抽象。默认情况下，Scala 2.11 中的执行上下文是一个*工作窃取线程池*。当一个并行集合提交任务时，上下文将这些任务分配给其线程。如果一个线程发现它已经完成了其队列中的任务，它将尝试从其他线程中窃取未完成的任务。默认执行上下文维护一个线程池，线程数等于 CPU 的数量。

并行集合分配工作的线程数可以通过更改任务支持来改变。例如，为了通过四个线程并行化由范围执行的操作：

```py
scala> import scala.collection.parallel._
import scala.collection.parallel._

scala> parRange.tasksupport = new ForkJoinTaskSupport(
 new scala.concurrent.forkjoin.ForkJoinPool(4)
)
parRange.tasksupport: scala.collection.parallel.TaskSupport = scala.collection.parallel.ForkJoinTaskSupport@6e1134e1

scala> parRange.tasksupport.parallelismLevel
Int: 4

```

## 例子 - 使用并行集合的交叉验证

让我们将你到目前为止学到的知识应用到解决数据科学问题中。机器学习管道的许多部分可以轻易并行化。其中之一就是交叉验证。

我们在这里将简要介绍交叉验证，但你可以参考*《统计学习的要素》*，由*哈斯蒂*，*蒂布斯哈尼*和*弗里德曼*撰写，以获得更深入的讨论。

通常，一个监督式机器学习问题涉及在一个训练集上训练一个算法。例如，当我们构建一个基于身高和体重计算一个人性别概率的模型时，训练集是每个参与者的(身高，体重)数据，以及每行的男/女标签。一旦算法在训练集上训练完成，我们就可以用它来分类新的数据。这个过程只有在训练集能够代表我们可能遇到的新数据时才有意义。

训练集有有限数量的条目。因此，不可避免地，它将具有一些不是代表整个人群的独特性，仅仅是因为其有限性。这些独特性将在预测新人是男性还是女性时导致预测错误，这超出了算法在训练集本身上的预测错误。交叉验证是估计由不反映整个人群的训练集独特性引起的错误的工具。

交叉验证通过将训练集分成两部分：一个较小的、新的训练集和一个交叉验证集来工作。算法在减少的训练集上训练。然后我们看看算法对交叉验证集的建模效果如何。由于我们知道交叉验证集的正确答案，我们可以衡量当展示新信息时我们的算法表现得多好。我们重复这个程序多次，使用不同的交叉验证集。

有几种不同的交叉验证类型，它们在如何选择交叉验证集方面有所不同。在本章中，我们将探讨重复随机子采样：我们从训练数据中随机选择*k*行来形成交叉验证集。我们这样做很多次，计算每个子样本的交叉验证误差。由于每个迭代都是相互独立的，我们可以简单地并行化这个过程。因此，它是一个并行集合的良好候选者。我们将在第十二章中查看交叉验证的另一种形式，即*k*折交叉验证*，*使用 MLlib 进行分布式机器学习*。

我们将构建一个执行并行交叉验证的类。我鼓励你在编写代码的同时进行，但你将在 GitHub 上找到与这些示例对应的源代码（[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)）。我们将使用并行集合来处理并行性，并在内部循环中使用 Breeze 数据类型。`build.sbt`文件与我们在第二章中使用的相同，即*使用 Breeze 操作数据*：

```py
scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
 "org.scalanlp" %% "breeze" % "0.11.2",
 "org.scalanlp" %% "breeze-natives" % "0.11.2"
)

```

我们将构建一个`RandomSubsample`类。该类公开了一个类型别名`CVFunction`，用于一个接受两个索引列表的函数——第一个对应于减少的训练集，第二个对应于验证集——并返回一个与交叉验证误差相对应的`Double`值：

```py
type CVFunction = (Seq[Int], Seq[Int]) => Double
```

`RandomSubsample`类将公开一个名为`mapSamples`的单个方法，它接受一个`CVFunction`，重复传递不同的索引分区，并返回一个错误向量。这个类看起来是这样的：

```py
// RandomSubsample.scala

import breeze.linalg._
import breeze.numerics._

/** Random subsample cross-validation
  * 
  * @param nElems Total number of elements in the training set.
  * @param nCrossValidation Number of elements to leave out of training set.
*/
class RandomSubsample(val nElems:Int, val nCrossValidation:Int) {

  type CVFunction = (Seq[Int], Seq[Int]) => Double

  require(nElems > nCrossValidation,
    "nCrossValidation, the number of elements " +
    "withheld, must be < nElems")

  private val indexList = DenseVector.range(0, nElems)

  /** Perform multiple random sub-sample CV runs on f
    *
    * @param nShuffles Number of random sub-sample runs.
    * @param f user-defined function mapping from a list of
    *   indices in the training set and a list of indices in the
    *   test-set to a double indicating the out-of sample score
    *   for this split.
    * @returns DenseVector of the CV error for each random split.
    */
  def mapSamples(nShuffles:Int)(f:CVFunction)
  :DenseVector[Double] = {
    val cvResults = (0 to nShuffles).par.map { i =>

      // Randomly split indices between test and training
      val shuffledIndices = breeze.linalg.shuffle(indexList)
      val Seq(testIndices, trainingIndices) =
        split(shuffledIndices, Seq(nCrossValidation))

 // Apply f for this split
      f(trainingIndices.toScalaVector, 
        testIndices.toScalaVector)
    }
    DenseVector(cvResults.toArray)
  }
}
```

让我们更详细地看看发生了什么，从传递给构造函数的参数开始：

```py
class RandomSubsample(val nElems:Int, val nCrossValidation:Int)
```

我们在类构造函数中传递训练集中元素的总数和为交叉验证留出的元素数。因此，将 100 传递给`nElems`，将 20 传递给`nCrossValidation`意味着我们的训练集将包含 80 个随机元素的总数据，而测试集将包含 20 个元素。

然后我们构建一个介于`0`和`nElems`之间的所有整数的列表：

```py
private val indexList = DenseVector.range(0, nElems)
```

对于交叉验证的每一次迭代，我们将对这个列表进行洗牌，并取前`nCrossValidation`个元素作为测试集行索引，剩余的作为训练集行索引。

我们公开了一个单一的方法`mapSamples`，它接受两个柯里化参数：`nShuffles`，表示执行随机子采样的次数，以及`f`，一个`CVFunction`：

```py
  def mapSamples(nShuffles:Int)(f:CVFunction):DenseVector[Double] 
```

在设置好所有这些之后，进行交叉验证的代码看起来很简单。我们生成一个从`0`到`nShuffles`的并行范围，并对范围中的每个项目，生成一个新的训练-测试分割并计算交叉验证错误：

```py
    val cvResults = (0 to nShuffles).par.map { i =>
      val shuffledIndices = breeze.linalg.shuffle(indexList)
      val Seq(testIndices, trainingIndices) = 
        split(shuffledIndices, Seq(nCrossValidation))
      f(trainingIndices.toScalaVector, testIndices.toScalaVector)
    }
```

这个函数的唯一难点是将打乱后的索引列表分割成训练集索引列表和测试集索引列表。我们使用 Breeze 的`split`方法。这个方法接受一个向量作为其第一个参数，一个分割点列表作为其第二个参数，并返回原始向量的片段列表。然后我们使用模式匹配来提取各个部分。

最后，`mapSamples`将`cvResults`转换为 Breeze 向量：

```py
DenseVector(cvResults.toArray) 
```

让我们看看它是如何工作的。我们可以通过在第二章中开发的逻辑回归示例上运行交叉验证来测试我们的类，*使用 Breeze 操作数据*。在该章中，我们开发了一个`LogisticRegression`类，它在构造时接受一个训练集（以`DenseMatrix`的形式）和一个目标（以`DenseVector`的形式）。然后该类计算最能代表训练集的参数。我们首先将向`LogisticRegression`类添加两个方法，以使用训练好的模型对先前未见过的示例进行分类：

+   `predictProbabilitiesMany`方法使用训练好的模型来计算目标变量被设置为 1 的概率。在我们的示例中，这是给定身高和体重为男性的概率。

+   `classifyMany`方法将分类标签（一个或零）分配给测试集的成员。如果`predictProbabilitiesMany`返回的值大于`0.5`，我们将分配一个一。

使用这两个函数，我们的`LogisticRegression`类变为：

```py
// Logistic Regression.scala

class LogisticRegression(
  val training:DenseMatrix[Double],
  val target:DenseVector[Double]
) {
  ...
  /** Probability of classification for each row
    * in test set.
    */
  def predictProbabilitiesMany(test:DenseMatrix[Double])
  :DenseVector[Double] = {
    val xBeta = test * optimalCoefficients
    sigmoid(xBeta)
  }

  /** Predict the value of the target variable 
    * for each row in test set.
    */
  def classifyMany(test:DenseMatrix[Double])
  :DenseVector[Double] = {
    val probabilities = predictProbabilitiesMany(test)
    I((probabilities :> 0.5).toDenseVector)
  }
  ...
}
```

现在，我们可以为我们的`RandomSubsample`类编写一个示例程序。我们将使用与第二章中相同的身高-体重数据，*使用 Breeze 操作数据*。数据预处理将类似。本章的代码示例提供了一个辅助模块`HWData`，用于将身高-体重数据加载到 Breeze 向量中。数据本身位于本章代码示例的`data/`目录中（可在 GitHub 上找到[`github.com/pbugnion/s4ds/tree/master/chap04`](https://github.com/pbugnion/s4ds/tree/master/chap04))。

对于每个新的子样本，我们创建一个新的`LogisticRegression`实例，在训练集的子集上对其进行训练，以获得此训练-测试分割的最佳系数，并使用`classifyMany`在此分割的交叉验证集上生成预测。然后我们计算分类错误并报告每个训练-测试分割的平均分类错误：

```py
// RandomSubsampleDemo.scala

import breeze.linalg._
import breeze.linalg.functions.manhattanDistance
import breeze.numerics._
import breeze.stats._

object RandomSubsampleDemo extends App {

  /* Load and pre-process data */
  val data = HWData.load

  val rescaledHeights:DenseVector[Double] =
    (data.heights - mean(data.heights)) / stddev(data.heights)

  val rescaledWeights:DenseVector[Double] =
    (data.weights - mean(data.weights)) / stddev(data.weights)

  val featureMatrix:DenseMatrix[Double] =
    DenseMatrix.horzcat(
      DenseMatrix.onesDouble,
      rescaledHeights.toDenseMatrix.t,
      rescaledWeights.toDenseMatrix.t
    )

  val target:DenseVector[Double] = data.genders.values.map { 
    gender => if(gender == 'M') 1.0 else 0.0 
  }

  /* Cross-validation */
  val testSize = 20
  val cvCalculator = new RandomSubsample(data.npoints, testSize)

  // Start parallel CV loop
  val cvErrors = cvCalculator.mapSamples(1000) { 
    (trainingIndices, testIndices) =>

    val regressor = new LogisticRegression(
      data.featureMatrix(trainingIndices, ::).toDenseMatrix,
      data.target(trainingIndices).toDenseVector
    )
    // Predictions on test-set
    val genderPredictions = regressor.classifyMany(
      data.featureMatrix(testIndices, ::).toDenseMatrix
    )
    // Calculate number of mis-classified examples
    val dist = manhattanDistance(
      genderPredictions, data.target(testIndices)
    )
    // Calculate mis-classification rate
    dist / testSize.toDouble
  }

  println(s"Mean classification error: ${mean(cvErrors)}")
}
```

在高度-体重数据上运行此程序给出 10%的分类错误。

现在我们有一个完全工作、并行化的交叉验证类。Scala 的并行范围使得在不同的线程中重复计算相同的函数变得简单。

# 未来

并行集合提供了一个简单而强大的并行操作框架。然而，在一方面它们是有限的：必须预先知道总工作量，并且每个线程必须执行相同的函数（可能在不同的输入上）。

假设我们想要编写一个程序，每隔几秒从网页（或查询一个 Web API）中获取数据，并从该网页中提取数据以进行进一步处理。一个典型的例子可能涉及查询 Web API 以维护特定股票价格的最新值。从外部网页获取数据通常需要几百毫秒。如果我们在这个主线程上执行此操作，它将无谓地浪费 CPU 周期等待 Web 服务器回复。

解决方案是将获取网页的代码包装在一个*未来*中。未来是一个包含计算未来结果的单一元素容器。当你创建一个未来时，其中的计算会被卸载到不同的线程，以避免阻塞主线程。当计算完成时，结果会被写入未来，从而使其对主线程可访问。

作为例子，我们将编写一个程序，查询“Markit on demand”API 以获取给定股票的价格。例如，谷歌当前股价的 URL 是[`dev.markitondemand.com/MODApis/Api/v2/Quote?symbol=GOOG`](http://dev.markitondemand.com/MODApis/Api/v2/Quote?symbol=GOOG)。请将此粘贴到您网络浏览器的地址框中。您将看到一个 XML 字符串出现，其中包含当前股价等信息。让我们先不使用未来来以编程方式获取这个信息：

```py
scala> import scala.io._
import scala.io_

scala> val url = "http://dev.markitondemand.com/MODApis/Api/v2/Quote?symbol=GOOG"
url: String = http://dev.markitondemand.com/MODApis/Api/v2/Quote?symbol=GOOG

scala> val response = Source.fromURL(url).mkString
response: String = <StockQuote><Status>SUCCESS</Status>
...

```

注意查询 API 需要一点时间。现在让我们做同样的事情，但使用一个未来（现在不用担心导入，我们将在后面详细讨论它们的含义）：

```py
scala> import scala.concurrent._
import scala.concurrent._

scala> import scala.util._
import scala.util._

scala> import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.ExecutionContext.Implicits.global

scala> val response = Future { Source.fromURL(url).mkString }
response: Future[String] = Promise$DefaultPromise@3301801b

```

如果你运行这个程序，你会在 API 有机会响应之前立即返回控制台。为了使这一点更加明显，让我们通过添加`Thread.sleep`调用来模拟一个慢速连接：

```py
scala> val response = Future { 
 Thread.sleep(10000) // sleep for 10s
 Source.fromURL(url).mkString
}
response: Future[String] = Promise$DefaultPromise@231f98ef

```

当你运行这个程序时，你不需要等待十秒钟才能出现下一个提示符：你立即重新获得控制台的控制权。未来中的代码是异步执行的：它的执行与主程序流程独立。

我们如何检索计算的成果？我们注意到`response`的类型是`Future[String]`。我们可以通过查询未来的`isCompleted`属性来检查未来中包装的计算是否完成：

```py
scala> response.isCompleted
Boolean = true

```

未来暴露了一个包含计算结果的`value`属性：

```py
scala> response.value
Option[Try[String]] = Some(Success(<StockQuote><Status>SUCCESS</Status>
...

```

Future 的 `value` 属性类型为 `Option[Try[T]]`。我们已经在并行集合的上下文中看到了如何使用 `Try` 类型优雅地处理异常。在这里也是同样的用法。一个 future 的 `value` 属性在 future 完成之前是 `None`，然后它被设置为 `Some(Success(value))` 如果 future 成功运行，或者设置为 `Some(Failure(error))` 如果抛出了异常。

在 shell 中重复调用 `f.value` 直到 future 完成，效果很好，但这并不能推广到更复杂的程序中。相反，我们希望告诉计算机一旦 future 完成，就执行某些操作：我们希望将 *回调* 函数绑定到 future。我们可以通过设置 future 的 `onComplete` 属性来实现这一点。让我们告诉 future 在完成时打印 API 响应：

```py
scala> response.onComplete {
 case Success(s) => println(s)
 case Failure(e) => println(s"Error fetching page: $e")
}

scala> 
// Wait for response to complete, then prints:
<StockQuote><Status>SUCCESS</Status><Name>Alphabet Inc</Name><Symbol>GOOGL</Symbol><LastPrice>695.22</LastPrice><Chan...

```

传递给 `onComplete` 的函数在 future 完成时运行。它接受一个类型为 `Try[T]` 的单个参数，包含 future 的结果。

### 小贴士

**失败是正常的：如何构建弹性应用程序**

通过将运行代码的输出包装在 `Try` 类型中，futures 强制客户端代码考虑代码可能失败的可能性。客户端可以隔离失败的影响，避免整个应用程序崩溃。例如，他们可能会记录异常。在 Web API 查询的情况下，他们可能会将受影响的 URL 添加到稍后再次查询的列表中。在数据库失败的情况下，他们可能会回滚事务。

通过将失败视为一等公民，而不是通过在末尾附加异常控制流来处理，我们可以构建出更加弹性的应用程序。

## Future 组合 – 使用 future 的结果

在上一节中，你学习了关于 `onComplete` 方法将回调绑定到 future 的内容。这在 future 完成时引发副作用非常有用。然而，它并不能让我们轻松地转换 future 的返回值。

为了继续我们的股票示例，让我们想象我们想要将查询响应从字符串转换为 XML 对象。让我们首先在 `build.sbt` 中将 `scala-xml` 库作为依赖项包含进来：

```py
libraryDependencies += "org.scala-lang" % "scala-xml" % "2.11.0-M4"
```

让我们重新启动控制台并重新导入 `scala.concurrent._`、`scala.concurrent.ExecutionContext.Implicits.global` 和 `scala.io._` 这三个依赖。我们还想导入 `XML` 库：

```py
scala> import scala.xml.XML
import scala.xml.XML

```

我们将使用与上一节相同的 URL：

[`dev.markitondemand.com/MODApis/Api/v2/Quote?symbol=GOOG`](http://dev.markitondemand.com/MODApis/Api/v2/Quote?symbol=GOOG)

有时将 `future` 视为一个集合是有用的，如果计算成功，则包含一个元素；如果失败，则包含零个元素。例如，如果 Web API 已成功查询，我们的 `future` 包含响应的字符串表示。像 Scala 中的其他容器类型一样，`futures` 支持一个 `map` 方法，该方法将函数应用于 `future` 中包含的元素，返回一个新的 `future`，如果 `future` 中的计算失败则不执行任何操作。但在计算可能尚未完成的情况下，这意味着什么呢？`map` 方法在 `future` 完成时立即应用，就像 `onComplete` 方法一样。

我们可以使用 `future` 的 `map` 方法异步地对 `future` 的结果应用转换。让我们再次轮询 "Markit on demand" API。这次，我们不会打印结果，而是将其解析为 XML。

```py
scala> val strResponse = Future { 
 Thread.sleep(20000) // Sleep for 20s
 val res = Source.fromURL(url).mkString
 println("finished fetching url")
 res
}
strResponse: Future[String] = Promise$DefaultPromise@1dda9bc8

scala> val xmlResponse = strResponse.map { s =>
 println("applying string to xml transformation")
 XML.loadString(s) 
}
xmlResponse: Future[xml.Elem] = Promise$DefaultPromise@25d1262a

// wait while the remainder of the 20s elapses
finished fetching url
applying string to xml transformation

scala> xmlResponse.value
Option[Try[xml.Elem]] = Some(Success(<StockQuote><Status>SUCCESS</Status>...

```

通过在 `future` 上注册后续的映射，我们为运行 `future` 的执行器提供了一个路线图，说明要执行的操作。

如果任何步骤失败，包含异常的失败 `Try` 实例将被传播：

```py
scala> val strResponse = Future { 
 Source.fromURL("empty").mkString 
}

scala> val xmlResponse = strResponse.map { 
 s => XML.loadString(s) 
}

scala> xmlResponse.value 
Option[Try[xml.Elem]] = Some(Failure(MalformedURLException: no protocol: empty))

```

如果你将失败的 `future` 视为一个空容器，这种行为是有意义的。当将映射应用于空列表时，它返回相同的空列表。同样，当将映射应用于空（失败）的 `future` 时，返回空 `future`。

## 阻塞直到完成

用于获取股票价格的代码在 shell 中运行良好。然而，如果你将其粘贴到独立程序中，你会注意到没有任何内容被打印出来，程序立即结束。让我们看看这个简单示例：

```py
// BlockDemo.scala
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

object BlockDemo extends App {
  val f = Future { Thread.sleep(10000) }
  f.onComplete { _ => println("future completed") }
  // "future completed" is not printed
}
```

程序在主线程完成其任务后立即停止运行，在这个例子中，这仅仅涉及创建 `future`。特别是，"future completed" 这一行永远不会打印出来。如果我们想让主线程等待 `future` 执行，我们必须明确告诉它阻塞执行，直到 `future` 完成。这是通过使用 `Await.ready` 或 `Await.result` 方法来完成的。这两种方法都会阻塞主线程的执行，直到 `future` 完成。我们可以通过添加以下行来使上述程序按预期工作：

```py
Await.ready(f, 1 minute)
```

`Await` 方法将 `future` 作为第一个参数，将 `Duration` 对象作为第二个参数。如果 `future` 完成所需时间超过指定的持续时间，则抛出 `TimeoutException`。传递 `Duration.Inf` 来设置无限超时。

`Await.ready` 和 `Await.result` 之间的区别在于后者返回 `future` 内部的值。特别是，如果 `future` 导致异常，则该异常将被抛出。相比之下，`Await.ready` 返回 `future` 本身。

通常，应尽可能避免阻塞：`future`的整个目的就是在后台线程中运行代码，以保持执行主线程的响应性。然而，在程序结束时进行阻塞是一个常见且合理的用例。如果我们正在运行一个大规模的集成过程，我们可能会调度几个`future`来查询 Web API、从文本文件中读取或向数据库中插入数据。将代码嵌入到`future`中比按顺序执行这些操作更具有可扩展性。然而，由于大部分密集型工作都在后台线程中运行，当主线程完成时，我们会留下许多未完成的`future`。在这个阶段，阻塞直到所有`future`完成是有意义的。

## 使用执行上下文控制并行执行

现在我们已经知道了如何定义`future`，让我们看看如何控制它们的运行。特别是，你可能想要控制运行大量`future`时使用的线程数量。

当定义一个`future`时，它会传递一个`execution context`，无论是直接还是隐式。执行上下文是一个对象，它公开了一个`execute`方法，该方法接受一段代码并运行它，可能是异步的。通过更改执行上下文，我们可以更改运行`future`的“后端”。我们已经看到了如何使用执行上下文来控制并行集合的执行。

到目前为止，我们只是通过导入`scala.concurrent.ExecutionContext.Implicits.global`来使用默认的执行上下文。这是一个具有与底层 CPU 数量相同的线程的 fork/join 线程池。

现在让我们定义一个新的使用十六个线程的执行上下文：

```py
scala> import java.util.concurrent.Executors
import java.util.concurrent.Executors

scala> val ec = ExecutionContext.fromExecutorService(
 Executors.newFixedThreadPool(16)
)
ec: ExecutionContextExecutorService = ExecutionContextImpl$$anon$1@1351ce60

```

定义了执行上下文后，我们可以显式地将它传递给正在定义的`future`：

```py
scala> val f = Future { Thread.sleep(1000) } (ec)
f: Future[Unit] = Promise$DefaultPromise@458b456

```

或者，我们可以隐式地定义执行上下文：

```py
scala> implicit val context = ec
context: ExecutionContextExecutorService = ExecutionContextImpl$$anon$1@1351ce60

```

然后将它作为隐式参数传递给所有新构建的`future`：

```py
scala> val f = Future { Thread.sleep(1000) }
f: Future[Unit] = Promise$DefaultPromise@3c4b7755

```

你可以关闭执行上下文来销毁线程池：

```py
scala> ec.shutdown()

```

当执行上下文收到关闭命令时，它将完成当前任务的执行，但会拒绝任何新的任务。

## 期货示例 - 股票价格获取器

让我们将本节中介绍的一些概念结合起来，构建一个命令行应用程序，该程序提示用户输入股票名称并获取该股票的价值。难点在于，为了保持用户界面响应，我们将使用`future`来获取股票信息：

```py
// StockPriceDemo.scala

import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.io._
import scala.xml.XML
import scala.util._

object StockPriceDemo extends App {

 /* Construct URL for a stock symbol */
  def urlFor(stockSymbol:String) =
    ("http://dev.markitondemand.com/MODApis/Api/v2/Quote?" + 
     s"symbol=${stockSymbol}")

  /* Build a future that fetches the stock price */
  def fetchStockPrice(stockSymbol:String):Future[BigDecimal] = {
    val url = urlFor(stockSymbol)
    val strResponse = Future { Source.fromURL(url).mkString }
    val xmlResponse = strResponse.map { s => XML.loadString(s) }
    val price = xmlResponse.map { 
      r => BigDecimal((r \ "LastPrice").text) 
    }
    price
  }

  /* Command line interface */
  println("Enter symbol at prompt.")
  while (true) {
    val symbol = readLine("> ") // Wait for user input
    // When user puts in symbol, fetch data in background
    // thread and print to screen when complete
    fetchStockPrice(symbol).onComplete { res =>
      println()
      res match {
        case Success(price) => println(s"$symbol: USD $price")
        case Failure(e) => println(s"Error fetching  $symbol: $e")
      }
      print("> ") // Simulate the appearance of a new prompt
    }
  }

}
```

尝试运行程序并输入一些股票的代码：

```py
[info] Running StockPriceDemo
Enter symbol at prompt:
> GOOG
> MSFT
>
GOOG: USD 695.22
>
MSFT: USD 47.48
> AAPL
> 
AAPL: USD 111.01

```

让我们总结一下代码的工作原理。当你输入一个股票时，主线程构建一个期货，从 API 获取股票信息，将其转换为 XML，并提取价格。我们使用`(r \ "LastPrice").text`从 XML 节点`r`中提取`LastPrice`标签内的文本。然后我们将值转换为大数据。当转换完成后，结果通过`onComplete`绑定回调打印到屏幕。异常处理通过我们使用`.map`方法处理转换来自然处理。

通过将获取股票价格的代码包装在期货中，我们释放了主线程，使其仅用于响应用户。这意味着如果我们的互联网连接很慢，用户界面不会阻塞。

这个例子有些人为，但你很容易将其包装得更加复杂：股票价格可以写入数据库，我们还可以添加额外的命令来绘制股票价格随时间的变化，例如。

在本节中，我们只是刚刚触及了期货所能提供的表面内容。当我们查看第七章（part0059.xhtml#aid-1O8H61 "第七章。Web API"）、*Web API*和第九章（part0077.xhtml#aid-29DRA1 "第九章。使用 Akka 的并发"）、*使用 Akka 的并发*时，我们将更详细地回顾期货。

期货是数据科学家构建可扩展系统工具箱的关键部分。将昂贵的计算（无论是 CPU 时间还是墙时间）移动到后台线程可以极大地提高可扩展性。因此，期货是许多 Scala 库（如**Akka**和**Play**框架）的重要组成部分。

# 摘要

通过提供高级并发抽象，Scala 使得编写并行代码直观且简单。并行集合和期货是数据科学家工具箱中不可或缺的部分，它们允许他们以最小的努力并行化代码。然而，尽管这些高级抽象消除了直接处理线程的需求，但理解 Scala 并发模型内部机制是必要的，以避免竞争条件。

在下一章中，我们将暂停并发的研究，学习如何与 SQL 数据库交互。然而，这只是一个临时的举措：期货将在本书剩余的许多章节中扮演重要的角色。

# 参考文献

*Aleksandar Prokopec*，*在 Scala 中学习并发编程*。这是对 Scala 并发编程基础的一个详细介绍。特别是，它比本章更详细地探讨了并行集合和期货。

Daniel Westheide 的博客为许多 Scala 概念提供了出色的介绍，特别是：

+   **未来**: [`danielwestheide.com/blog/2013/01/09/the-neophytes-guide-to-scala-part-8-welcome-to-the-future.html`](http://danielwestheide.com/blog/2013/01/09/the-neophytes-guide-to-scala-part-8-welcome-to-the-future.html)

+   **尝试** **类型**: [`danielwestheide.com/blog/2012/12/26/the-neophytes-guide-to-scala-part-6-error-handling-with-try.html`](http://danielwestheide.com/blog/2012/12/26/the-neophytes-guide-to-scala-part-6-error-handling-with-try.html)

关于交叉验证的讨论，请参阅 Hastie、Tibshirani 和 Friedman 合著的《统计学习基础》。

# 第五章：Scala 和 SQL 通过 JDBC

数据科学的一个基本目标是处理大型数据集的困难。许多公司或研究组感兴趣的数据都无法方便地存储在单个计算机的 RAM 中。因此，以易于查询的方式存储数据是一个复杂的问题。

关系型数据库在解决数据存储问题上取得了成功。最初于 1970 年提出([`www.seas.upenn.edu/~zives/03f/cis550/codd.pdf`](http://www.seas.upenn.edu/~zives/03f/cis550/codd.pdf))，今天仍在使用的绝大多数数据库仍然是关系型。在这段时间里，每兆字节 RAM 的价格下降了十亿倍。同样，硬盘容量也从几十或几百兆字节增加到太字节。值得注意的是，尽管数据存储容量呈指数增长，但关系型模型仍然占据主导地位。

几乎所有关系型数据库都使用 SQL（**结构化查询语言**）的变体进行描述和查询。随着分布式计算的出现，SQL 数据库作为事实上的数据存储标准的地位正受到其他类型数据库的挑战，这些数据库通常被统称为 NoSQL。许多 NoSQL 数据库比 SQL 数据库更具有分区容错性：它们可以被分割成几个部分，分别存储在不同的计算机上。尽管作者预计 NoSQL 数据库将越来越受欢迎，但 SQL 数据库作为数据持久化机制可能仍然普遍存在；因此，本书的很大一部分内容将致力于从 Scala 与 SQL 交互。

虽然 SQL 是标准化的，但大多数实现并不完全遵循该标准。此外，大多数实现都提供了对标准的扩展。这意味着，尽管本书中的许多概念将适用于所有 SQL 后端，但确切的语法可能需要调整。在这里，我们将仅考虑 MySQL 实现。

在本章中，您将学习如何使用 JDBC（一个基本的 Java API）从 Scala 与 SQL 数据库交互。在下一章中，我们将考虑 Slick，这是一个**对象关系映射器**（ORM），它为与 SQL 的交互提供了更符合 Scala 风格的感觉。

本章大致分为两部分：我们首先将讨论连接和与 SQL 数据库交互的基本功能，然后讨论可以用来创建优雅、松散耦合和一致的数据访问层的有用功能模式。

本章假设你具备基本的 SQL 工作知识。如果你不具备，你最好先阅读本章末尾提到的参考书籍之一。

# 与 JDBC 交互

JDBC 是 Java 连接 SQL 数据库的 API。它仍然是 Scala 连接 SQL 数据库的最简单方式。此外，大多数用于与数据库交互的高级抽象仍然使用 JDBC 作为后端。

JDBC 本身不是一个库。相反，它暴露了一组用于与数据库交互的接口。关系数据库供应商随后提供这些接口的特定实现。

让我们先创建一个`build.sbt`文件。我们将声明对 MySQL JDBC 连接器的依赖：

```py
scalaVersion := "2.11.7"

libraryDependencies += "mysql" % "mysql-connector-java" % "5.1.36"
```

# JDBC 的第一步

让我们从命令行连接到 JDBC 开始。为了跟随示例，你需要访问一个正在运行的 MySQL 服务器。如果你已经将 MySQL 连接器添加到依赖列表中，可以通过输入以下命令打开 Scala 控制台：

```py
$ sbt console

```

让我们导入 JDBC：

```py
scala> import java.sql._
import java.sql._

```

我们接下来需要告诉 JDBC 使用特定的连接器。这通常是通过反射完成的，在运行时加载驱动：

```py
scala> Class.forName("com.mysql.jdbc.Driver")
Class[_] = class com.mysql.jdbc.Driver

```

这个操作将在运行时将适当的驱动加载到命名空间中。如果你觉得这有点神奇，可能不值得担心它具体是如何工作的。这是本书中我们将考虑的唯一反射示例，而且它并不是特别符合 Scala 的惯例。

## 连接到数据库服务器

已经指定了 SQL 连接器，我们现在可以连接到数据库。假设我们有一个名为`test`的数据库，它在`127.0.0.1`主机上监听端口`3306`。我们创建连接如下：

```py
scala> val connection = DriverManager.getConnection(
 "jdbc:mysql://127.0.0.1:3306/test",
 "root", // username when connecting
 "" // password
)
java.sql.Connection = com.mysql.jdbc.JDBC4Connection@12e78a69

```

`getConnection`的第一个参数是一个类似 URL 的字符串，格式为`jdbc:mysql://host[:port]/database`。第二个和第三个参数是用户名和密码。如果你无需密码即可连接，请传入空字符串。

## 创建表

现在我们已经建立了数据库连接，让我们与服务器交互。对于这些示例，你将发现打开一个 MySQL 壳（或 MySQL GUI，如**MySQLWorkbench**）以及 Scala 控制台很有用。你可以在终端中输入以下命令打开 MySQL 壳：

```py
$ mysql

```

例如，我们将创建一个小表来跟踪著名物理学家。在`mysql`壳中，我们会运行以下命令：

```py
mysql> USE test;
mysql> CREATE TABLE physicists (
 id INT(11) AUTO_INCREMENT PRIMARY KEY,
 name VARCHAR(32) NOT NULL
);

```

要在 Scala 中实现相同的功能，我们需要向连接发送一个 JDBC 语句：

```py
scala> val statementString = """
CREATE TABLE physicists (
 id INT(11) AUTO_INCREMENT PRIMARY KEY,
 name VARCHAR(32) NOT NULL
)
"""

scala> val statement = connection.prepareStatement(statementString)
PreparedStatement = JDBC4PreparedStatement@c983201: CREATE TABLE ...

scala> statement.executeUpdate()
results: Int = 0

```

现在，我们先忽略`executeUpdate`方法的返回值。

## 插入数据

现在我们已经创建了一个表，让我们向其中插入一些数据。我们可以使用 SQL `INSERT`语句来完成这个操作：

```py
scala> val statement = connection.prepareStatement("""
 INSERT INTO physicists (name) VALUES ('Isaac Newton')
""")

scala> statement.executeUpdate()
Int = 1

```

在这种情况下，`executeUpdate`返回`1`。当插入行时，它返回插入的行数。同样，如果我们使用了一个`SQL UPDATE`语句，这将返回更新的行数。对于不直接操作行的语句（如上一节中的`CREATE TABLE`语句），`executeUpdate`仅返回`0`。

让我们直接跳入`mysql` shell 来验证插入是否正确执行：

```py
mysql> select * from physicists ;
+----+--------------+
| id | name         |
+----+--------------+
|  1 | Isaac Newton |
+----+--------------+
1 row in set (0.00 sec)

```

让我们快速总结一下到目前为止我们所看到的：要执行不返回结果的 SQL 语句，请使用以下方法：

```py
val statement = connection.prepareStatement("SQL statement string")
statement.executeUpdate()

```

在数据科学的背景下，我们经常需要一次插入或更新多行。例如，我们可能有一个物理学家列表：

```py
scala> val physicistNames = List("Marie Curie", "Albert Einstein", "Paul Dirac")

```

我们希望将这些全部插入到数据库中。虽然我们可以为每个物理学家创建一个语句并将其发送到数据库，但这非常低效。更好的解决方案是创建一个*批量*语句并将它们一起发送到数据库。我们首先创建一个语句模板：

```py
scala> val statement = connection.prepareStatement("""
 INSERT INTO physicists (name) VALUES (?)
""")
PreparedStatement = JDBC4PreparedStatement@621a8225: INSERT INTO physicists (name) VALUES (** NOT SPECIFIED **)

```

这与之前的`prepareStatement`调用相同，只是我们将物理学家名字替换为`?`占位符。我们可以使用`statement.setString`方法设置占位符的值：

```py
scala> statement.setString(1, "Richard Feynman")

```

这将用字符串`Richard Feynman`替换语句中的第一个占位符：

```py
scala> statement
com.mysql.jdbc.JDBC4PreparedStatement@5fdd16c3:
INSERT INTO physicists (name) VALUES ('Richard Feynman')

```

注意，JDBC 有些反直觉地，从 1 而不是 0 开始计算占位符位置。

我们现在已创建了更新批量的第一个语句。运行以下命令：

```py
scala> statement.addBatch()

```

通过运行前面的命令，我们启动了一个批量插入：该语句被添加到一个临时缓冲区中，当我们运行`executeBatch`方法时将执行。让我们将列表中的所有物理学家添加进去：

```py
scala> physicistNames.foreach { name => 
 statement.setString(1, name)
 statement.addBatch()
}

```

我们现在可以执行批量中的所有语句：

```py
scala> statement.executeBatch
Array[Int] = Array(1, 1, 1, 1)

```

`executeBatch`的返回值是一个数组，表示批量中每个项目更改或插入的行数。

注意，我们使用了`statement.setString`来用特定的名字填充模板。`PreparedStatement`对象为所有基本类型都有`setXXX`方法。要获取完整的列表，请阅读`PreparedStatement` API 文档([`docs.oracle.com/javase/7/docs/api/java/sql/PreparedStatement.html`](http://docs.oracle.com/javase/7/docs/api/java/sql/PreparedStatement.html))。

## 读取数据

现在我们知道了如何将数据插入数据库，让我们看看相反的情况：读取数据。我们使用 SQL `SELECT`语句来查询数据库。让我们首先在 MySQL shell 中这样做：

```py
mysql> SELECT * FROM physicists;
+----+-----------------+
| id | name            |
+----+-----------------+
|  1 | Isaac Newton    |
|  2 | Richard Feynman |
|  3 | Marie Curie     |
|  4 | Albert Einstein |
|  5 | Paul Dirac      |
+----+-----------------+
5 rows in set (0.01 sec)

```

要在 Scala 中提取此信息，我们定义一个`PreparedStatement`：

```py
scala> val statement = connection.prepareStatement("""
 SELECT name FROM physicists
""")
PreparedStatement = JDBC4PreparedStatement@3c577c9d:
SELECT name FROM physicists

```

我们通过运行以下命令来执行此语句：

```py
scala> val results = statement.executeQuery()
results: java.sql.ResultSet = com.mysql.jdbc.JDBC4ResultSet@74a2e158

```

这返回一个 JDBC `ResultSet`实例。`ResultSet`是一个表示数据库中一组行的抽象。请注意，我们使用了`statement.executeQuery`而不是`statement.executeUpdate`。一般来说，应该使用`executeQuery`来执行返回数据（以`ResultSet`形式）的语句。修改数据库而不返回数据（插入、创建、更改或更新语句等）的语句使用`executeUpdate`执行。

`ResultSet`对象的行为有点像迭代器。它暴露了一个`next`方法，该方法将其自身推进到下一个记录，如果`ResultSet`中还有记录，则返回`true`：

```py
scala> results.next // Advance to the first record
Boolean = true

```

当`ResultSet`实例指向一个记录时，我们可以通过传递字段名来提取该记录中的字段：

```py
scala> results.getString("name")
String = Isaac Newton

```

我们还可以使用位置参数提取字段。字段从一开始索引：

```py
scala> results.getString(1) // first positional argument
String = Isaac Newton

```

当我们完成特定记录的处理时，我们调用 `next` 方法将 `ResultSet` 移动到下一个记录：

```py
scala> results.next // advances the ResultSet by one record
Boolean = true

scala> results.getString("name")
String = Richard Feynman

```

![读取数据](img/image01193.jpeg)

`ResultSet` 对象支持 `getXXX(fieldName)` 方法来访问记录的字段，以及一个 `next` 方法来移动到结果集中的下一个记录。

可以使用 `while` 循环遍历结果集：

```py
scala> while(results.next) { println(results.getString("name")) }
Marie Curie
Albert Einstein
Paul Dirac

```

### 小贴士

关于读取可空字段有一个警告。虽然当面对 null SQL 字段时，人们可能期望 JDBC 返回 null，但返回类型取决于使用的 `getXXX` 命令。例如，`getInt` 和 `getLong` 将对任何 null 字段返回 `0`。同样，`getDouble` 和 `getFloat` 返回 `0.0`。这可能导致代码中的一些微妙错误。一般来说，应该小心使用返回 Java 值类型（`int`、`long`）而不是对象的 getter。要确定数据库中的值是否为 `null`，首先使用 `getInt`（或 `getLong` 或 `getDouble`，视情况而定）查询它，然后使用返回布尔值的 `wasNull` 方法：

```py
scala> rs.getInt("field")
0
scala> rs.wasNull // was the last item read null?
true

```

这种（令人惊讶的）行为使得从 `ResultSet` 实例中读取变得容易出错。本章第二部分的一个目标是为您提供构建在 `ResultSet` 接口之上的抽象层的工具，以避免直接调用如 `getInt` 这样的方法。

在 Scala 中直接从 `ResultSet` 对象读取值感觉很不自然。在本章的后续部分，我们将探讨通过构建一个层来访问结果集，您可以通过类型类来访问这个层。

现在我们已经知道如何读取和写入数据库。现在我们已经完成了对数据库的操作，我们将关闭结果集、预处理语句和连接：

```py
scala> results.close

scala> statement.close

scala> connection.close

```

虽然在 Scala shell 中关闭语句和连接并不重要（它们将在你退出时关闭），但在运行程序时很重要；否则，对象将持续存在，导致“内存不足异常”。在下一节中，我们将探讨使用 **贷款模式** 建立连接和语句，这是一种设计模式，在完成使用资源后自动关闭它。

# JDBC 概述

现在我们对 JDBC 有了一个概述。本章的其余部分将专注于编写位于 JDBC 之上的抽象，使数据库访问感觉更自然。在我们这样做之前，让我们总结一下到目前为止我们所看到的。

我们已经使用了三个 JDBC 类：

+   `Connection` 类表示与特定 SQL 数据库的连接。以下是如何实例化连接的示例：

    ```py
    import java.sql._
    Class.forName("com.mysql.jdbc.Driver")val connection = DriverManager.getConnection(
      "jdbc:mysql://127.0.0.1:3306/test",
      "root", // username when connecting
      "" // password
    )
    ```

    我们主要使用 `Connection` 实例来生成 `PreparedStatement` 对象：

    ```py
    connection.prepareStatement("SELECT * FROM physicists")
    ```

+   `PreparedStatement` 实例代表即将发送到数据库的 SQL 语句。它还代表一个 SQL 语句的模板，其中包含尚未填充的值占位符。该类公开以下方法：

    | `statement.executeUpdate` | 这将语句发送到数据库。用于修改数据库且不返回任何数据的 SQL 语句，例如 `INSERT`、`UPDATE`、`DELETE` 和 `CREATE` 语句。 |
    | --- | --- |
    | `val results = statement.executeQuery` | 这将语句发送到数据库。用于返回数据的 SQL 语句（主要是 `SELECT` 语句）。这返回一个 `ResultSet` 实例。 |
    | `statement.addBatch` `statement.executeBatch` | `addBatch` 方法将当前语句添加到语句批处理中，而 `executeBatch` 将语句批处理发送到数据库。 |
    | `statement.setString(1, "Scala")` `statement.setInt(1, 42)` `statement.setBoolean(1, true)` | 在 `PreparedStatement` 中填写占位符值。第一个参数是语句中的位置（从 1 开始计数）。第二个参数是值。这些方法的常见用例包括批量更新或插入：我们可能有一个想要插入数据库的 Scala 对象列表。我们使用 `.setXXX` 方法为列表中的每个对象填写占位符，然后使用 `.addBatch` 将此语句添加到批处理中。然后我们可以使用 `.executeBatch` 将整个批处理发送到数据库。 |
    | `statement.setNull(1, java.sql.Types.BOOLEAN)` | 这将语句中的特定项设置为 `NULL`。第二个参数指定 `NULL` 类型。如果我们正在设置布尔列中的单元格，例如，这应该是 `Types.BOOLEAN`。`java.sql.Types` 包的 API 文档中提供了类型列表（[`docs.oracle.com/javase/7/docs/api/java/sql/Types.html`](http://docs.oracle.com/javase/7/docs/api/java/sql/Types.html)）。 |

+   `ResultSet` 实例表示由 `SELECT` 或 `SHOW` 语句返回的一组行。`ResultSet` 提供了访问当前行字段的方法：

    | `rs.getString(i)` `rs.getInt(i)` | 这些方法获取当前行中第 `i` 个字段的值；`i` 从 1 开始计算。 |
    | --- | --- |
    | `rs.getString("name")` `rs.getInt("age")` | 这些方法获取特定字段的值，字段通过列名索引。 |
    | `rs.wasNull` | 这返回最后读取的列是否为 `NULL`。当读取 Java 值类型（如 `getInt`、`getBoolean` 或 `getDouble`）时，这尤其重要，因为这些在读取 `NULL` 值时返回默认值。 |

`ResultSet` 实例公开了 `.next` 方法以移动到下一行；`.next` 返回 `true`，直到 `ResultSet` 前进到最后一行之后。

# JDBC 的函数式包装

我们现在对 JDBC 提供的工具有一个基本的概述。到目前为止，我们与之交互的所有对象在 Scala 中都显得有些笨拙和不合适。它们并不鼓励函数式编程风格。

当然，优雅本身可能不是目标（或者至少，你可能很难说服你的 CEO 因为代码缺乏优雅而推迟产品的发布）。然而，它通常是一个症状：要么代码不可扩展或耦合度过高，要么容易引入错误。对于 JDBC 来说，后者尤其如此。忘记检查`wasNull`？这会反过来咬你。忘记关闭你的连接？你会得到一个“内存不足异常”（希望不是在生产环境中）。

在接下来的几节中，我们将探讨我们可以用来包装 JDBC 类型以减轻许多这些风险的模式。我们在这里引入的模式在 Scala 库和应用程序中非常常见。因此，除了编写健壮的类与 JDBC 交互之外，了解这些模式，我希望，将使你对 Scala 编程有更深入的理解。

# 使用借款模式实现更安全的 JDBC 连接

我们已经看到了如何连接到 JDBC 数据库并向数据库发送执行语句。然而，这种技术有些容易出错：你必须记得关闭语句；否则，你会很快耗尽内存。在更传统的命令式风格中，我们在每个连接周围编写以下 try-finally 块：

```py
// WARNING: poor Scala code
val connection = DriverManager.getConnection(url, user, password)
try {
  // do something with connection
}
finally {
  connection.close()
}
```

Scala，凭借一等函数，为我们提供了一个替代方案：*借款模式*。我们编写一个负责打开连接、将连接借给客户端代码以执行一些有趣的操作，并在客户端代码完成后关闭连接的函数。因此，客户端代码不再负责关闭连接。

让我们创建一个新的`SqlUtils`对象，并使用`usingConnection`方法利用借款模式：

```py
// SqlUtils.scala

import java.sql._

object SqlUtils {

  /** Create an auto-closing connection using 
    * the loan pattern */
  def usingConnectionT(f:Connection => T):T = {

    // Create the connection
    val Url = s"jdbc:mysql://$host:$port/$db"
    Class.forName("com.mysql.jdbc.Driver")
    val connection = DriverManager.getConnection(
      Url, user, password)

    // give the connection to the client, through the callable 
    // `f` passed in as argument
    try {
      f(connection)
    }
    finally {
      // When client is done, close the connection
      connection.close()
    }
  }
}
```

让我们看看这个函数的实际应用：

```py
scala> SqlUtils.usingConnection("test") {
 connection => println(connection)
}
com.mysql.jdbc.JDBC4Connection@46fd3d66

```

因此，客户端不需要记住关闭连接，对于客户端来说，代码感觉更像是 Scala。

我们的`usingConnection`函数是如何工作的？函数定义是`def usingConnection( ... )(f : Connection => T ):T`。它接受第二组参数，即作用于`Connection`对象的功能。`usingConnection`的主体创建连接，然后将其传递给`f`，最后关闭连接。这种语法与 Ruby 中的代码块或 Python 中的`with`语句有些相似。

### 小贴士

当混合借款模式与延迟操作时要小心。这尤其适用于从`f`返回迭代器、流和未来。一旦执行线程离开`f`，连接就会被关闭。在此点之前未实例化的任何数据结构将无法继续访问连接。

借款模式当然不仅仅局限于数据库连接。当你遇到以下模式时，它非常有用，以下为伪代码：

```py
open resource (eg. database connection, file ...)
use resource somehow // loan resource to client for this part.
close resource
```

# 使用“pimp my library”模式丰富 JDBC 语句

在上一节中，我们看到了如何使用贷款模式创建自关闭的连接。这允许我们打开数据库连接，而无需记住关闭它们。然而，我们仍然需要记住关闭我们打开的任何`ResultSet`和`PreparedStatement`：

```py
// WARNING: Poor Scala code
SqlUtils.usingConnection("test") { connection =>
  val statement = connection.prepareStatement(
    "SELECT * FROM physicists")
  val results = statement.executeQuery
  // do something useful with the results
  results.close
  statement.close
}
```

需要打开和关闭语句有些丑陋且容易出错。这也是贷款模式的另一个自然用例。理想情况下，我们希望编写以下内容：

```py
usingConnection("test") { connection =>
  connection.withQuery("SELECT * FROM physicists") {
    resultSet => // process results
  }
}
```

我们如何能在`Connection`类上定义一个`.withQuery`方法？我们并不控制`Connection`类的定义，因为它属于 JDBC API 的一部分。我们希望能够以某种方式重新打开`Connection`类的定义来添加`withQuery`方法。

Scala 不允许我们重新打开类来添加新方法（这种做法被称为猴子补丁）。然而，我们仍然可以使用**pimp my library**模式通过隐式转换来增强现有库（[`www.artima.com/weblogs/viewpost.jsp?thread=179766`](http://www.artima.com/weblogs/viewpost.jsp?thread=179766)）。我们首先定义一个包含`withQuery`方法的`RichConnection`类。这个`RichConnection`类是由现有的`Connection`实例创建的。

```py
// RichConnection.scala

import java.sql.{Connection, ResultSet}

class RichConnection(val underlying:Connection) {

  /** Execute a SQL query and process the ResultSet */
  def withQueryT(f:ResultSet => T):T = {
    val statement = underlying.prepareStatement(query)
    val results = statement.executeQuery
    try {
      f(results) // loan the ResultSet to the client
    }
    finally {
      // Ensure all the resources get freed.
      results.close
      statement.close
    }
  }
}
```

我们可以通过将每个`Connection`实例包装在一个`RichConnection`实例中来使用这个类：

```py
// Warning: poor Scala code
SqlUtils.usingConnection("test") { connection =>
  val richConnection = new RichConnection(connection)
  richConnection.withQuery("SELECT * FROM physicists") {
    resultSet => // process resultSet
  }
}
```

这增加了不必要的样板代码：我们必须记住将每个连接实例转换为`RichConnection`才能使用`withQuery`。幸运的是，Scala 提供了一个更简单的方法，即隐式转换：我们告诉 Scala 如何从`Connection`转换为`RichConnection`，反之亦然，并告诉它如果需要则自动（隐式）执行此转换：

```py
// Implicits.scala
import java.sql.Connection

// Implicit conversion methods are often put in 
// an object called Implicits.
object Implicits {
  implicit def pimpConnection(conn:Connection) = 
    new RichConnection(conn)
  implicit def depimpConnection(conn:RichConnection) =  
    conn.underlying
}
```

现在，每当`pimpConnection`和`depimpConnection`在当前作用域内时，Scala 将自动使用它们将`Connection`实例转换为`RichConnection`，并在需要时将其转换回`Connection`。

现在，我们可以编写以下内容（我添加了类型信息以强调）：

```py
// Bring the conversion functions into the current scope
import Implicits._ 

SqlUtils.usingConnection("test") { (connection:Connection) =>
  connection.withQuery("SELECT * FROM physicists") {
    // Wow! It's like we have just added 
    // .withQuery to the JDBC Connection class!
    resultSet => // process results
  }
}
```

这可能看起来像魔法，所以让我们退后一步，看看当我们对一个`Connection`实例调用`withQuery`时会发生什么。Scala 编译器首先会查看`Connection`类的定义是否包含`withQuery`方法。当它发现没有时，它会寻找将`Connection`实例转换为定义`withQuery`的类的隐式方法。它会发现`pimpConnection`方法允许从`Connection`转换为定义`withQuery`的`RichConnection`。Scala 编译器会自动使用`pimpConnection`将`Connection`实例转换为`RichConnection`。

注意，我们使用了`pimpConnection`和`depimpConnection`这样的名称作为转换函数，但它们可以是任何名称。我们从未明确调用这些方法。

让我们总结一下如何使用*pimp my library*模式向现有类添加方法：

1.  编写一个包装你想要增强的类的类：`class RichConnection(val underlying:Connection)`。添加你希望原始类拥有的所有方法。

1.  编写一个方法将你的原始类转换为你的增强类，作为名为（传统上）`Implicits` 的对象的一部分。确保你告诉 Scala 使用 `implicit` 关键字自动使用这个转换：`implicit def pimpConnection(conn:Connection):RichConnection`。你还可以告诉 Scala 通过添加反向转换方法自动将增强类转换回原始类。

1.  通过导入隐式转换方法允许隐式转换：`import Implicits._`。

# 在流中封装结果集

JDBC `ResultSet` 对象与 Scala 集合配合得非常糟糕。真正能够用它做些有用的事情的唯一方法就是直接使用 `while` 循环遍历它。例如，为了获取我们数据库中物理学家的名字列表，我们可以编写以下代码：

```py
// WARNING: poor Scala code
import Implicits._ // import implicit conversions

SqlUtils.usingConnection("test") { connection =>
  connection.withQuery("SELECT * FROM physicists") { resultSet =>
    var names = List.empty[String]
    while(resultSet.next) {
      val name = resultSet.getString("name")
      names = name :: names
    }
 names
  }
}
//=> List[String] = List(Paul Dirac, Albert Einstein, Marie Curie, Richard Feynman, Isaac Newton)
```

`ResultSet` 接口感觉很不自然，因为它与 Scala 集合的行为非常不同。特别是，它不支持我们在 Scala 中视为理所当然的高阶函数：没有 `map`、`filter`、`fold` 或 `for` 语句。幸运的是，编写一个封装 `ResultSet` 的 *stream* 非常简单。Scala 流是一个延迟评估的列表：它在需要时评估集合中的下一个元素，并在不再使用时忘记之前的元素。

我们可以定义一个 `stream` 方法，如下封装 `ResultSet`：

```py
// SqlUtils.scala
object SqlUtils {   
  ... 
  def stream(results:ResultSet):Stream[ResultSet] = 
    if (results.next) { results #:: stream(results) }
    else { Stream.empty[ResultSet] }
}
```

这可能看起来相当令人困惑，所以让我们慢慢来。我们定义一个 `stream` 方法来封装 `ResultSet`，返回一个 `Stream[ResultSet]`。当客户端在空结果集上调用 `stream` 时，这只会返回一个空流。当客户端在非空 `ResultSet` 上调用 `stream` 时，`ResultSet` 实例会向前推进一行，客户端会得到 `results #:: stream(results)`。流上的 `#::` 操作符类似于列表上的 `::` 操作符：它将 `results` 预先添加到现有的 `Stream` 中。关键的区别是，与列表不同，`stream(results)` 不会在必要时进行评估。因此，这避免了在内存中重复整个 `ResultSet`。

让我们使用我们全新的 `stream` 函数来获取我们数据库中所有物理学家的名字：

```py
import Implicits._

SqlUtils.usingConnection("test") { connection =>
  connection.withQuery("SELECT * FROM physicists") { results =>
    val resultsStream = SqlUtils.stream(results)
    resultsStream.map { _.getString("name") }.toVector
  }
}
//=> Vector(Richard Feynman, Albert Einstein, Marie Curie, Paul Dirac)
```

相比直接使用结果集，流式处理结果允许我们以更自然的方式与数据交互，因为我们现在处理的是 Scala 集合。

当你在 `withQuery` 块（或者更一般地，在自动关闭结果集的块）中使用 `stream` 时，你必须在函数内部始终将流具体化，因此调用了 `toVector`。否则，流将等待其元素被需要时才具体化它们，而那时，`ResultSet` 实例将被关闭。

# 通过类型类实现更松散的耦合

到目前为止，我们一直在数据库中读取和写入简单类型。让我们假设我们想要向我们的数据库添加一个 `gender` 列。我们将把性别作为枚举存储在我们的物理学家数据库中。我们的表现在如下所示：

```py
mysql> CREATE TABLE physicists (
 id INT(11) AUTO_INCREMENT PRIMARY KEY,
 name VARCHAR(32) NOT NULL,
 gender ENUM("Female", "Male") NOT NULL
);

```

我们如何在 Scala 中表示性别？一种好的方法是使用枚举：

```py
// Gender.scala

object Gender extends Enumeration {
  val Male = Value
  val Female = Value
}
```

然而，我们现在在从数据库反序列化对象时遇到了问题：JDBC 没有内置机制将 SQL `ENUM`类型转换为 Scala `Gender`类型。我们可以通过每次需要读取性别信息时手动转换来实现这一点：

```py
resultsStream.map { 
  rs => Gender.withName(rs.getString("gender")) 
}.toVector
```

然而，我们需要在所有想要读取`gender`字段的地方都写下这些代码。这违反了 DRY（不要重复自己）原则，导致代码难以维护。如果我们决定更改数据库中存储性别的方式，我们就需要找到代码中所有读取`gender`字段的地方并对其进行更改。

一个稍微好一些的解决方案是在`ResultSet`类中添加一个`getGender`方法，使用我们在本章中广泛使用的 pimp my library 习语。这个解决方案仍然不是最优的。我们正在向`ResultSet`添加不必要的特定性：它现在与我们的数据库结构耦合。

我们可以通过继承`ResultSet`来创建一个子类，例如`PhysicistResultSet`，这样就可以读取特定表中的字段。然而，这种方法是不可组合的：如果我们还有另一个表，它跟踪宠物，包括名称、种类和性别字段，我们就必须要么在新的`PetResultSet`中重新实现读取性别的代码，要么提取一个`GenderedResultSet`超类。随着表数量的增加，继承层次结构将变得难以管理。更好的方法可以让我们组合所需的功能。特别是，我们希望将从一个结果集中提取 Scala 对象的过程与遍历结果集的代码解耦。

## 类型类

Scala 提供了一个优雅的解决方案，使用*类型类*。类型类是 Scala 架构师箭袋中的一个非常强大的箭头。然而，它们可能有一定的学习曲线，尤其是在面向对象编程中没有直接等效物。

我不会提供抽象的解释，而是直接进入一个例子：我将描述如何利用类型类将`ResultSet`中的字段转换为 Scala 类型。目标是定义一个`readT`方法在`ResultSet`上，该方法知道如何精确地将对象反序列化为类型`T`。此方法将替换并扩展`ResultSet`中的`getXXX`方法：

```py
// results is a ResultSet instance
val name = results.readString
val gender = results.readGender.Value
```

我们首先定义一个抽象的`SqlReader[T]`特质，它公开一个`read`方法，用于从`ResultSet`中读取特定字段并返回类型为`T`的实例：

```py
// SqlReader.scala

import java.sql._

trait SqlReader[T] {
  def read(results:ResultSet, field:String):T
}
```

现在，我们需要为每个我们想要读取的`T`类型提供一个`SqlReader[T]`的具体实现。让我们为`Gender`和`String`字段提供具体实现。我们将实现放在`SqlReader`伴生对象中：

```py
// SqlReader.scala

object SqlReader {
  implicit object StringReader extends SqlReader[String] {
    def read(results:ResultSet, field:String):String =
      results.getString(field)
  }

  implicit object GenderReader extends SqlReader[Gender.Value] {
    def read(results:ResultSet, field:String):Gender.Value =
      Gender.withName(StringReader.read(results, field))
  }
}
```

我们现在可以使用我们的`ReadableXXX`对象从结果集中读取：

```py
import SqlReader._
val name = StringReader.read(results, "name")
val gender = GenderReader.read(results, "gender")
```

这已经比使用以下方法好一些：

```py
Gender.withName(results.getString("gender"))
```

这是因为将 `ResultSet` 字段映射到 `Gender.Value` 的代码集中在一个单独的位置：`ReadableGender`。然而，如果我们能告诉 Scala 在需要读取 `Gender.Value` 时使用 `ReadableGender`，在需要读取字符串值时使用 `ReadableString`，那就太好了。这正是类型类的作用。

## 面向类型类的编码

我们定义了一个 `Readable[T]` 接口，它抽象化了如何从 `ResultSet` 字段中读取类型为 `T` 的对象。我们如何告诉 Scala 需要使用这个 `Readable` 对象将 `ResultSet` 字段转换为适当的 Scala 类型？

关键是我们在 `GenderReader` 和 `StringReader` 对象定义前使用的 `implicit` 关键字。它允许我们编写：

```py
implicitly[SqlReader[Gender.Value]].read(results, "gender")
implicitly[SqlReader[String]].read(results, "name")
```

通过编写 `implicitly[SqlReader[T]]`，我们是在告诉 Scala 编译器找到一个扩展 `SqlReader[T]` 并标记为隐式使用的类（或对象）。你可以通过在命令行粘贴以下内容来尝试，例如：

```py
scala> :paste

import Implicits._ // Connection to RichConnection conversion
SqlUtils.usingConnection("test") {
 _.withQuery("select * from physicists") {
 rs => {
 rs.next() // advance to first record
 implicitly[SqlReader[Gender.Value]].read(rs, "gender")
 }
 }
}

```

当然，在所有地方使用 `implicitly[SqlReader[T]]` 并不是特别优雅。让我们使用“pimp my library”惯用法向 `ResultSet` 添加一个 `read[T]` 方法。我们首先定义一个 `RichResultSet` 类，我们可以用它来“pimp” `ResultSet` 类：

```py
// RichResultSet.scala

import java.sql.ResultSet

class RichResultSet(val underlying:ResultSet) {
  def readT : SqlReader:T = {
    implicitly[SqlReader[T]].read(underlying, field)
  }
}
```

这里的唯一不熟悉的部分应该是 `read[T : SqlReader]` 泛型定义。我们在这里声明，如果存在 `SqlReader[T]` 的实例，`read` 将接受任何 `T` 类型。这被称为*上下文限制*。

我们还必须在 `Implicits` 对象中添加隐式方法，以将 `ResultSet` 转换为 `RichResultSet`。你现在应该熟悉这个了，所以我就不会详细说明了。你现在可以为任何具有 `SqlReader[T]` 隐式对象的 `T` 调用 `results.readT`：

```py
import Implicits._

SqlUtils.usingConnection("test") { connection =>
  connection.withQuery("SELECT * FROM physicists") {
    results =>
      val resultStream = SqlUtils.stream(results)
      resultStream.map { row => 
 val name = row.readString
 val gender = row.readGender.Value
        (name, gender)
      }.toVector
  }
}
//=> Vector[(String, Gender.Value)] = Vector((Albert Einstein,Male), (Marie Curie,Female))
```

让我们总结一下使类型类正常工作所需的步骤。我们将在从 SQL 反序列化的上下文中进行此操作，但你将能够将这些步骤适应以解决其他问题：

+   定义一个抽象的泛型特质，为类型类提供接口，例如，`SqlReader[T]`。任何与 `T` 无关的功能都可以添加到这个基本特质中。

+   为基本特质创建伴随对象，并为每个 `T` 添加扩展特质的隐式对象，例如，

    ```py
    implicit object StringReader extends SqlReader[T].
    ```

+   类型类始终用于泛型方法。一个依赖于类型类存在的参数的方法必须在泛型定义中包含上下文限制，例如，`def readT : SqlReader:T`。要访问此方法中的类型类，使用 `implicitly` 关键字：`implicitly[SqlReader[T]]`。

## 何时使用类型类

当你需要为许多不同类型实现特定行为，但此行为的具体实现在这类之间有所不同时，类型类很有用。例如，我们需要能够从 `ResultSet` 中读取几种不同的类型，但每种类型的读取方式各不相同：对于字符串，我们必须使用 `getString` 从 `ResultSet` 中读取，而对于整数，我们必须使用 `getInt` 后跟 `wasNull`。

一个很好的经验法则是当你开始想“哦，我完全可以写一个泛型方法来做这件事。啊，但是等等，我必须将 `Int` 实现作为一个特定的边缘情况来写，因为它的行为不同。哦，还有 `Gender` 实现。我想知道是否有更好的方法？”时，类型类可能就很有用了。

## 类型类的优势

数据科学家经常必须处理新的输入流、变化的需求和新数据类型。因此，拥有一个易于扩展或更改的对象关系映射层对于有效地应对变化至关重要。最小化代码实体之间的耦合和关注点的分离是确保代码能够根据新数据更改的唯一方法。

使用类型类，我们保持了访问数据库中的记录（通过 `ResultSet` 类）和将单个字段转换为 Scala 对象的方式之间的正交性：这两者可以独立变化。这两个关注点之间的唯一耦合是通过 `SqlReader[T]` 接口。

这意味着这两个关注点可以独立进化：要读取新的数据类型，我们只需实现一个 `SqlReader[T]` 对象。相反，我们可以在不重新实现字段转换方式的情况下向 `ResultSet` 添加功能。例如，我们可以添加一个 `getColumn` 方法，它返回 `ResultSet` 实例中一个字段的 `Vector[T]` 所有值：

```py
def getColumnT : SqlReader:Vector[T] = {
  val resultStream = SqlUtils.stream(results)
  resultStream.map { _.readT }.toVector
}
```

注意我们如何在不增加对单个字段读取方式耦合的情况下完成这件事。

# 创建数据访问层

让我们汇总我们所看到的一切，并为从数据库中检索 `Physicist` 对象构建一个 *数据映射器* 类。这些类（也称为 *数据访问对象*）有助于将对象的内部表示与其在数据库中的表示解耦。

我们首先定义了 `Physicist` 类：

```py
// Physicist.scala
case class Physicist(
  val name:String,
  val gender:Gender.Value
)
```

数据访问对象将公开一个单一的方法，`readAll`，它返回包含我们数据库中所有物理学家的 `Vector[Physicist]`：

```py
// PhysicistDao.scala

import java.sql.{ ResultSet, Connection }
import Implicits._ // implicit conversions

object PhysicistDao {

  /* Helper method for reading a single row */
  private def readFromResultSet(results:ResultSet):Physicist = {
    Physicist(
      results.readString,
      results.readGender.Value
    )
  }

  /* Read the entire 'physicists' table. */
  def readAll(connection:Connection):Vector[Physicist] = {
    connection.withQuery("SELECT * FROM physicists") {
      results =>
        val resultStream = SqlUtils.stream(results)
        resultStream.map(readFromResultSet).toVector
    }
  }
}
```

客户端代码可以使用数据访问层，如下例所示：

```py
object PhysicistDaoDemo extends App {

  val physicists = SqlUtils.usingConnection("test") {
    connection => PhysicistDao.readAll(connection)
  }

  // physicists is a Vector[Physicist] instance.
  physicists.foreach { println }
  //=> Physicist(Albert Einstein,Male)
  //=> Physicist(Marie Curie,Female)
}
```

# 摘要

在本章中，我们学习了如何使用 JDBC 与 SQL 数据库进行交互。我们编写了一个库来封装原生 JDBC 对象，目的是为它们提供一个更功能化的接口。

在下一章中，你将了解 Slick，这是一个 Scala 库，它提供了与关系数据库交互的功能包装器。

# 参考资料

JDBC 的 API 文档非常完整：[`docs.oracle.com/javase/7/docs/api/java/sql/package-summary.html`](http://docs.oracle.com/javase/7/docs/api/java/sql/package-summary.html)

`ResultSet` 接口的 API 文档（[`docs.oracle.com/javase/7/docs/api/java/sql/ResultSet.html`](http://docs.oracle.com/javase/7/docs/api/java/sql/ResultSet.html)）、`PreparedStatement` 类（[`docs.oracle.com/javase/7/docs/api/java/sql/PreparedStatement.html`](http://docs.oracle.com/javase/7/docs/api/java/sql/PreparedStatement.html)）和 `Connection` 类（[`docs.oracle.com/javase/7/docs/api/java/sql/Connection.html`](http://docs.oracle.com/javase/7/docs/api/java/sql/Connection.html)）的文档尤其相关。

数据映射模式在 Martin Fowler 的《企业应用架构模式》中描述得非常详细。在他的网站上也有简要的描述（[`martinfowler.com/eaaCatalog/dataMapper.html`](http://martinfowler.com/eaaCatalog/dataMapper.html)）。

对于 SQL 的入门，我建议阅读 *Alan Beaulieu* 的《Learning SQL》(*O'Reilly*)。

对于类型类的另一篇讨论，请阅读 [`danielwestheide.com/blog/2013/02/06/the-neophytes-guide-to-scala-part-12-type-classes.html`](http://danielwestheide.com/blog/2013/02/06/the-neophytes-guide-to-scala-part-12-type-classes.html)。

本文描述了如何使用类型类在 Scala 中更优雅地重新实现一些常见的面向对象设计模式：

[`staticallytyped.wordpress.com/2013/03/24/gang-of-four-patterns-with-type-classes-and-implicits-in-scala-part-2/`](https://staticallytyped.wordpress.com/2013/03/24/gang-of-four-patterns-with-type-classes-and-implicits-in-scala-part-2/)

这篇由 *Martin Odersky* 撰写的文章详细介绍了 *Pimp my Library* 模式：

[`www.artima.com/weblogs/viewpost.jsp?thread=179766`](http://www.artima.com/weblogs/viewpost.jsp?thread=179766)

# 第六章。Slick – SQL 的函数式接口

在 第五章 中，我们探讨了如何使用 JDBC 访问 SQL 数据库。由于与 JDBC 交互感觉有些不自然，我们通过自定义封装扩展了 JDBC。这些封装是为了提供一个函数式接口，以隐藏 JDBC 的命令式本质。

由于直接从 Scala 与 JDBC 交互的难度以及 SQL 数据库的普遍性，你可能会期待存在现有的 Scala 库来封装 JDBC。*Slick* 就是这样一个库。

Slick 自称为 *函数式-关系映射* 库，这是对更传统的 *对象-关系映射* 名称的一种戏谑，后者用来表示从关系数据库构建对象的库。它提供了一个函数式接口来访问 SQL 数据库，允许客户端以类似于原生 Scala 集合的方式与之交互。

# FEC 数据

在本章中，我们将使用一个稍微复杂一些的示例数据集。**美国联邦选举委员会（Federal Electoral Commission of the United States，简称 FEC**）记录了所有超过 200 美元的总统候选人捐款。这些记录是公开可用的。我们将查看导致巴拉克·奥巴马连任的 2012 年大选前的捐款情况。数据包括对两位总统候选人奥巴马和罗姆尼的捐款，以及共和党初选中其他竞争者的捐款（没有民主党初选）。

在本章中，我们将使用 FEC 提供的交易数据，将其存储在表中，并学习如何查询和分析它。

第一步是获取数据。如果你已经从 Packt 网站下载了代码示例，你应该已经在本章代码示例的 `data` 目录中有两个 CSV 文件。如果没有，你可以使用以下链接下载文件：

+   `data.scala4datascience.com/fec/ohio.csv.gz`（或 `ohio.csv.zip`）

+   `data.scala4datascience.com/fec/us.csv.gz`（或 `us.csv.zip`）

解压缩这两个文件，并将它们放置在本章源代码示例相同的 `data/` 目录中。数据文件对应以下内容：

+   `ohio.csv` 文件是俄亥俄州所有捐赠者捐赠的 CSV 文件。

+   `us.csv` 文件是全国范围内所有捐赠者捐赠的 CSV 文件。这是一个相当大的文件，有六百万行。

这两个 CSV 文件包含相同的列。使用俄亥俄州数据集可以获得更快的响应，或者如果你想处理更大的数据集，可以使用全国数据文件。数据集是从 [`www.fec.gov/disclosurep/PDownload.do`](http://www.fec.gov/disclosurep/PDownload.do) 下载的贡献列表中改编的。

让我们先创建一个 Scala case class 来表示一笔交易。在本章的上下文中，一笔交易是一个个人向候选人捐赠的单笔捐款：

```py
// Transaction.scala
import java.sql.Date

case class Transaction(
  id:Option[Int], // unique identifier
  candidate:String, // candidate receiving the donation
  contributor:String, // name of the contributor
  contributorState:String, // contributor state
  contributorOccupation:Option[String], // contributor job
  amount:Long, // amount in cents
  date:Date // date of the donation
)
```

本章的代码仓库包含一个 `FECData` 单例对象中的辅助函数，用于从 CSV 文件中加载数据：

```py
scala> val ohioData = FECData.loadOhio
s4ds.FECData = s4ds.FECData@718454de

```

调用 `FECData.loadOhio` 或 `FECData.loadAll` 将创建一个具有单个属性 `transactions` 的 `FECData` 对象，该属性是一个遍历来自俄亥俄州或整个美国的所有捐赠的迭代器：

```py
scala> val ohioTransactions = ohioData.transactions
Iterator[Transaction] = non-empty iterator

scala> ohioTransactions.take(5).foreach(println)
Transaction(None,Paul, Ron,BROWN, TODD W MR.,OH,Some(ENGINEER),5000,2011-01-03)
Transaction(None,Paul, Ron,DIEHL, MARGO SONJA,OH,Some(RETIRED),2500,2011-01-03)
Transaction(None,Paul, Ron,KIRCHMEYER, BENJAMIN,OH,Some(COMPUTER PROGRAMMER),20120,2011-01-03)
Transaction(None,Obama, Barack,KEYES, STEPHEN,OH,Some(HR EXECUTIVE / ATTORNEY),10000,2011-01-03)
Transaction(None,Obama, Barack,MURPHY, MIKE W,OH,Some(MANAGER),5000,2011-01-03)

```

现在我们有一些数据可以操作，让我们尝试将其放入数据库中，以便我们可以运行一些有用的查询。

## 导入 Slick

要将 Slick 添加到依赖项列表中，你需要在 `build.sbt` 文件中的依赖项列表中添加 `"com.typesafe.slick" %% "slick" % "2.1.0"`。你还需要确保 Slick 有权访问 JDBC 驱动程序。在本章中，我们将连接到 MySQL 数据库，因此必须将 MySQL 连接器 `"mysql" % "mysql-connector-java" % "5.1.37"` 添加到依赖项列表中。

通过导入特定的数据库驱动程序来导入 Slick。由于我们使用 MySQL，我们必须导入以下内容：

```py
scala> import slick.driver.MySQLDriver.simple._
import slick.driver.MySQLDriver.simple._

```

要连接到不同类型的 SQL 数据库，导入相关的驱动程序。查看可用的驱动程序的最简单方法是查阅`slick.driver`包的 API 文档，该文档可在[`slick.typesafe.com/doc/2.1.0/api/#scala.slick.driver.package`](http://slick.typesafe.com/doc/2.1.0/api/#scala.slick.driver.package)找到。所有常见的 SQL 类型都受支持（包括**H2**、**PostgreSQL**、**MS SQL Server**和**SQLite**）。

## 定义模式

让我们创建一个表来表示我们的交易。我们将使用以下模式：

```py
CREATE TABLE transactions(
    id INT(11) AUTO_INCREMENT PRIMARY KEY,
    candidate VARCHAR(254) NOT NULL,
    contributor VARCHAR(254) NOT NULL,
    contributor_state VARCHAR(2) NOT NULL,
    contributor_occupation VARCHAR(254),
    amount BIGINT(20) NOT NULL,
    date DATE 
);
```

注意，捐赠金额是以*分*表示的。这允许我们使用整数字段（而不是定点小数，或者更糟糕的是浮点数）。

### 注意

你永远不应该使用浮点格式来表示金钱，实际上，任何离散量，因为浮点数无法精确表示大多数分数：

```py
scala> 0.1 + 0.2
Double = 0.30000000000000004

```

这种看似荒谬的结果发生是因为在双精度浮点数中无法精确存储 0.3。

这篇文章广泛讨论了浮点格式限制：

[`docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html`](http://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)

要在数据库中的表上使用 Slick，我们首先需要告诉 Slick 有关数据库模式。我们通过创建一个扩展`Table`抽象类的类来实现这一点。定义模式的方式相当直接，所以让我们直接进入代码。我们将我们的模式存储在`Tables`单例中。我们定义一个`Transactions`类，它提供了从`Transaction`实例的集合到类似`transactions`表的 SQL 表的映射：

```py
// Tables.scala

import java.sql.Date
import slick.driver.MySQLDriver.simple._

/** Singleton object for table definitions */
object Tables {

  // Transactions table definition
  class Transactions(tag:Tag)
  extends TableTransaction {
    def id = columnInt
    def candidate = columnString
    def contributor = columnString
    def contributorState = columnString"))
    def contributorOccupation = column[Option[String]](
      "contributor_occupation")
    def amount = columnLong
    def date = columnDate

    def * = (id.?, candidate, contributor, 
      contributorState, contributorOccupation, amount, date) <> (
      Transaction.tupled, Transaction.unapply)
  }

  val transactions = TableQuery[Transactions]

}
```

让我们逐行分析。我们首先定义一个`Transactions`类，它必须将其第一个参数作为 Slick `Tag`对象。`Tag`对象由 Slick 内部用于构建 SQL 语句。`Transactions`类扩展了一个`Table`对象，传递给它标签和数据库中表的名称。我们可以选择性地通过扩展`TableTransaction, "transactions")`而不是仅仅`TableTransaction`来添加数据库名称。`Table`类型由`Transaction`参数化。这意味着在数据库上运行`SELECT`语句返回`Transaction`对象。同样，我们将通过传递一个事务或事务列表到相关的 Slick 方法来在数据库中插入数据。

让我们更详细地看看`Transactions`类的定义。类的主体首先列出数据库列。例如，`id`列定义如下：

```py
def id = columnInt
```

我们告诉 Slick 它应该读取名为`id`的列，并将其转换为 Scala 整数。此外，我们还告诉 Slick 该列是主键，并且它是自动增长的。Slick 文档中包含`column`可用选项的列表。

`candidate` 和 `contributor` 列很简单：我们告诉 Slick 从数据库中读取这些作为 `String`。`contributor_state` 列有点更有趣。除了指定它应该作为 `String` 从数据库中读取外，我们还告诉 Slick 它应该以 `VARCHAR(2)` 类型存储在数据库中。

我们表中的 `contributor_occupation` 列可以包含 `NULL` 值。在定义模式时，我们将 `Option[String]` 类型传递给列方法：

```py
def contributorOccupation = column[Option[String]]("contributor_occupation")
```

当从数据库读取时，对于指定为 `Option[T]` 的列，`NULL` 字段将被转换为 `None`。相反，如果字段有值，它将以 `Some(value)` 返回。

类体的最后一行是最有趣的部分：它指定了如何将读取的原始数据转换为 `Transaction` 对象以及如何将 `Transaction` 对象转换为准备插入的原始字段：

```py
def * = (id.?, candidate, contributor, 
contributorState, contributorOccupation, amount, date) <> (
Transaction.tupled, Transaction.unapply)
```

第一部分只是从数据库中读取的字段元组：`(id.?, candidate, contributor, contributorState, contributorOccupation, amount, date)`，附带少量元数据。第二部分是一对函数，描述了如何将此元组转换为 `Transaction` 对象以及如何反向转换。在这种情况下，由于 `Transaction` 是一个案例类，我们可以利用为案例类自动提供的 `Transaction.tupled` 和 `Transaction.unapply` 方法。

注意我们是如何在 `id` 条目后面跟着 `.?` 的。在我们的 `Transaction` 类中，捐赠 `id` 具有类型 `Option[Int]`，但数据库中的列具有类型 `INT`，并附加了 `O.AutoInc` 选项。`.?` 后缀告诉 Slick 如果 `id` 是 `None`，则使用数据库提供的默认值（在这种情况下，数据库的自动递增）。

最后，我们定义了值：

```py
val transactions = TableQuery[Transactions]
```

这是实际与数据库交互的句柄。例如，正如我们稍后将会看到的，要获取巴拉克·奥巴马的捐赠列表，我们将运行以下查询（现在不用担心查询的细节）：

```py
Tables.transactions.filter {_.candidate === "Obama, Barack"}.list
```

让我们总结一下我们的 `Transactions` 映射器类的部分：

+   `Transactions` 类必须扩展由我们想要返回的类型参数化的 `Table` 抽象类：`Table[Transaction]`。

+   我们使用 `column` 显式定义从数据库中读取的列，例如，`def contributorState = columnString"))`。`[String]` 类型参数定义了此列读取为的 Scala 类型。第一个参数是 SQL 列名。请参阅 Slick 文档以获取附加参数的完整列表（[`slick.typesafe.com/doc/2.1.0/schemas.html`](http://slick.typesafe.com/doc/2.1.0/schemas.html)）。

+   我们描述了如何使用 `def * = (id.?, candidate, ...) <> (Transaction.tupled, Transaction.unapply)` 将列值的元组转换为 Scala 对象，反之亦然。

## 连接到数据库

到目前为止，你已经学习了如何定义`Table`类，这些类将 SQL 表中的行转换为 Scala 案例类。为了超越表定义并开始与数据库服务器交互，我们必须连接到数据库。与上一章类似，我们将假设有一个 MySQL 服务器在本地主机的`3306`端口上运行。

我们将使用控制台来演示本章的功能，但你可以在`SlickDemo.scala`中找到一个等效的示例程序。让我们打开一个 Scala 控制台并连接到在`3306`端口上运行的数据库：

```py
scala> import slick.driver.MySQLDriver.simple._
import slick.driver.MySQLDriver.simple._

scala> val db = Database.forURL(
 "jdbc:mysql://127.0.0.1:3306/test",
 driver="com.mysql.jdbc.Driver"
)
db: slick.driver.MySQLDriver.backend.DatabaseDef = slick.jdbc.JdbcBackend$DatabaseDef@3632d1dd

```

如果你已经阅读了上一章，你会认出第一个参数是一个 JDBC 风格的 URL。URL 首先定义了一个协议，在本例中是`jdbc:mysql`，然后是数据库服务器的 IP 地址和端口，最后是数据库名称（这里为`test`）。

`forURL`的第二个参数是 JDBC 驱动程序的类名。此驱动程序在运行时使用反射导入。请注意，此处指定的驱动程序必须与静态导入的 Slick 驱动程序匹配。

定义了数据库后，我们现在可以使用它来创建一个连接：

```py
scala> db.withSession { implicit session =>
 // do something useful with the database
 println(session)
}
scala.slick.jdbc.JdbcBackend$BaseSession@af5a276

```

需要访问数据库的流畅函数隐式地接受一个`Session`参数：如果作用域内可用一个标记为隐式的`Session`实例，它们将使用它。因此，在`session`前加上`implicit`关键字可以节省我们每次在数据库上运行操作时显式传递`session`。

如果你已经阅读了上一章，你会认出 Slick 处理关闭连接的需求使用的是*借款模式*：数据库连接以`session`对象的形式创建，并临时传递给客户端。当客户端代码返回时，会话被关闭，确保所有打开的连接都被关闭。因此，客户端代码免除了关闭连接的责任。

借款模式在生产代码中非常有用，但在 shell 中可能会有些繁琐。Slick 允许我们显式地创建一个会话，如下所示：

```py
scala> implicit val session = db.createSession
session: slick.driver.MySQLDriver.backend.Session = scala.slick.jdbc.JdbcBackend$BaseSession@2b775b49

scala> session.close

```

## 创建表格

让我们使用我们新的连接在数据库中创建事务表。我们可以通过`TableQuery[Transactions]`实例上的`ddl`属性访问创建和删除表的方法：

```py
scala> db.withSession { implicit session =>
 Tables.transactions.ddl.create
}

```

如果你跳入`mysql` shell，你会看到已经创建了一个`transactions`表：

```py
mysql> describe transactions ;
+------------------------+--------------+------+-----+
| Field                  | Type         | Null | Key |
+------------------------+--------------+------+-----+
| id                     | int(11)      | NO   | PRI |
| candidate              | varchar(254) | NO   |     |
| contributor            | varchar(254) | NO   |     |
| contributor_state      | varchar(2)   | NO   |     |
| contributor_occupation | varchar(254) | YES  |     |
| amount                 | bigint(20)   | NO   |     |
| date                   | date         | NO   |     |
+------------------------+--------------+------+-----+
7 rows in set (0.01 sec)

```

`ddl`属性还包括一个`drop`方法来删除表。顺便提一下，`ddl`代表“数据定义语言”，通常用来指代与模式定义和约束定义相关的 SQL 部分。

## 插入数据

Slick 的`TableQuery`实例让我们可以通过与 Scala 集合类似的接口与 SQL 表交互。

让我们首先创建一个交易。我们将假装在 2010 年 6 月 22 日发生了一次捐赠。不幸的是，在 Scala 中创建日期并将其传递给 JDBC 的代码特别繁琐。我们首先创建一个`java.util.Date`实例，然后我们必须将其转换为`java.sql.Date`以用于我们新创建的交易：

```py
scala> import java.text.SimpleDateFormat
import java.text.SimpleDateFormat

scala> val date = new SimpleDateFormat("dd-MM-yyyy").parse("22-06-2010")
date: java.util.Date = Tue Jun 22 00:00:00 BST 2010

scala> val sqlDate = new java.sql.Date(date.getTime())
sqlDate: java.sql.Date = 2010-06-22

scala> val transaction = Transaction(
 None, "Obama, Barack", "Doe, John", "TX", None, 200, sqlDate
)
transaction: Transaction = Transaction(None,Obama, Barack,Doe, John,TX,None,200,2010-06-22)

```

`TableQuery`实例提供的界面大部分与可变列表相似。为了在事务表中插入单行，我们可以使用`+=`运算符：

```py
scala> db.withSession {
 implicit session => Tables.transactions += transaction
}
Int = 1

```

在底层，这将创建一个 JDBC 预处理语句并运行该语句的`executeUpdate`方法。

如果你一次提交多行，你应该使用 Slick 的批量插入运算符：`++=`。它接受一个`List[Transaction]`作为输入，并利用 JDBC 的`addBatch`和`executeBatch`功能将所有交易批量插入：

让我们插入所有 FEC 交易，这样我们就有一些数据在下一节运行查询时可以操作。我们可以通过调用以下代码来加载俄亥俄州的交易迭代器：

```py
scala> val transactions = FECData.loadOhio.transactions
transactions: Iterator[Transaction] = non-empty iterator

```

我们也可以加载整个美国的交易：

```py
scala> val transactions = FECData.loadAll.transactions
transactions: Iterator[Transaction] = non-empty iterator

```

为了避免一次性将所有交易实体化——从而可能超出我们计算机的可用内存——我们将从迭代器中取出交易批次并插入它们：

```py
scala> val batchSize = 100000
batchSize: Int = 100000

scala> val transactionBatches = transactions.grouped(batchSize)
transactionBatches: transactions.GroupedIterator[Transaction] = non-empty iterator

```

迭代器的`grouped`方法将迭代器分割成批次。将长集合或迭代器分割成可管理的批次，以便可以逐个处理，这在集成或处理大型数据集时非常重要。

我们现在要做的就是遍历我们的批次，在插入的同时将它们存入数据库：

```py
scala> db.withSession { implicit session =>
 transactionBatches.foreach { 
 batch => Tables.transactions ++= batch.toList
 }
}

```

虽然这样可行，但在进行长时间运行的集成过程时，有时查看进度报告是有用的。由于我们将集成分成了批次，我们知道（到最近的批次为止）我们集成到了哪里。让我们在每一个批次的开始处打印进度信息：

```py
scala> db.withSession { implicit session =>
 transactionBatches.zipWithIndex.foreach { 
 case (batch, batchNumber) =>
 println(s"Processing row ${batchNumber*batchSize}")
 Tables.transactions ++= batch.toList
 }
}
Processing row 0
Processing row 100000
...

```

我们使用`.zipWithIndex`方法将我们的批次迭代器转换为(*批次*, *当前索引*)对的迭代器。在一个完整规模的应用中，进度信息可能会被写入日志文件而不是屏幕。

Slick 精心设计的界面使得插入数据非常直观，与原生 Scala 类型很好地集成。

## 查询数据

在上一节中，我们使用 Slick 将捐赠数据插入到我们的数据库中。现在让我们探索这些数据。

在定义`Transactions`类时，我们定义了一个`TableQuery`对象`transactions`，它作为访问交易表的句柄。它提供了一个类似于 Scala 迭代器的接口。例如，要查看数据库中的前五个元素，我们可以调用`take(5)`：

```py
scala> db.withSession { implicit session =>
 Tables.transactions.take(5).list
}
List[Tables.Transactions#TableElementType] = List(Transaction(Some(1),Obama, Barack,Doe, ...

```

内部，Slick 使用 SQL `LIMIT`来实现`.take`方法。实际上，我们可以通过查询上的`.selectStatement`方法来获取 SQL 语句：

```py
scala> db.withSession { implicit session =>
 println(Tables.transactions.take(5).selectStatement)
}
select x2.`id`, x2.`candidate`, x2.`contributor`, x2.`contributor_state`, x2.`contributor_occupation`, x2.`amount`, x2.`date` from (select x3.`date` as `date`, x3.`contributor` as `contributor`, x3.`amount` as `amount`, x3.`id` as `id`, x3.`candidate` as `candidate`, x3.`contributor_state` as `contributor_state`, x3.`contributor_occupation` as `contributor_occupation` from `transactions` x3 limit 5) x2

```

我们的流畅查询由以下两个部分组成：

+   `.take(n)`: 这一部分称为 *调用器*。调用器构建 SQL 语句，但不会实际将其发送到数据库。你可以将多个调用器链接在一起来构建复杂的 SQL 语句。

+   `.list`: 这一部分将调用器准备好的语句发送到数据库，并将结果转换为 Scala 对象。这需要一个 `session` 参数，可能是隐式的。

# 调用器

**调用器**是 Slick 查询的组成部分，用于构建 SQL 选择语句。Slick 提供了各种调用器，允许构建复杂的查询。让我们看看其中的一些调用器：

+   `map` 调用器用于选择单个列或对列应用操作：

    ```py
    scala> db.withSession { implicit session =>
     Tables.transactions.map {
     _.candidate 
     }.take(5).list 
    }
    List[String] = List(Obama, Barack, Paul, Ron, Paul, Ron, Paul, Ron, Obama, Barack)

    ```

+   `filter` 调用器等同于 SQL 中的 `WHERE` 语句。请注意，Slick 字段必须使用 `===` 进行比较：

    ```py
    scala> db.withSession { implicit session => 
     Tables.transactions.filter {
     _.candidate === "Obama, Barack"
     }.take(5).list
    }
    List[Tables.Transactions#TableElementType] = List(Transaction(Some(1),Obama, Barack,Doe, John,TX,None,200,2010-06-22), ...

    ```

    同样，要过滤掉对巴拉克·奥巴马的捐款，使用 `=!=` 运算符：

    ```py
    scala> db.withSession { implicit session => 
     Tables.transactions.filter { 
     _.candidate =!= "Obama, Barack"
     }.take(5).list
    }
    List[Tables.Transactions#TableElementType] = List(Transaction(Some(2),Paul, Ron,BROWN, TODD W MR.,OH,...

    ```

+   `sortBy` 调用器等同于 SQL 中的 `ORDER BY` 语句：

    ```py
    scala> db.withSession { implicit session => 
     Tables.transactions.sortBy { 
     _.date.desc 
     }.take(5).list
    }
    List[Tables.Transactions#TableElementType] = List(Transaction(Some(65536),Obama, Barack,COPELAND, THOMAS,OH,Some(COLLEGE TEACHING),10000,2012-01-02)

    ```

+   `leftJoin`、`rightJoin`、`innerJoin` 和 `outerJoin` 调用器用于连接表。由于本教程不涉及多表之间的交互，我们无法演示连接操作。请参阅 Slick 文档（[`slick.typesafe.com/doc/2.1.0/queries.html#joining-and-zipping`](http://slick.typesafe.com/doc/2.1.0/queries.html#joining-and-zipping)）以了解这些操作的示例。

+   `length`、`min`、`max`、`sum` 和 `avg` 等聚合调用器可用于计算汇总统计信息。这些操作必须使用 `.run` 而不是 `.list` 执行，因为它们返回单个数字。例如，要获取巴拉克·奥巴马的总捐款：

    ```py
    scala> db.withSession { implicit session => 
     Tables.transactions.filter {
     _.candidate === "Obama, Barack"
     }.map { _.amount  }.sum.run
    }
    Option[Int] = Some(849636799) // (in cents)

    ```

# 列上的操作

在上一节中，你学习了不同的调用器以及它们如何映射到 SQL 语句。然而，我们只是简要地提到了列本身支持的方法：我们可以使用 `===` 进行相等比较，但 Slick 列还支持哪些其他操作？

大多数 SQL 函数都得到了支持。例如，要获取以 `"O"` 开头的候选人的总捐款，我们可以运行以下命令：

```py
scala> db.withSession { implicit session =>
 Tables.transactions.filter { 
 _.candidate.startsWith("O") 
 }.take(5).list 
}
List[Tables.Transactions#TableElementType] = List(Transaction(Some(1594098)...

```

同样，要计算在 2011 年 1 月 1 日至 2011 年 2 月 1 日之间发生的捐款，我们可以在 `date` 列上使用 `.between` 方法：

```py
scala> val dateParser = new SimpleDateFormat("dd-MM-yyyy")
dateParser: java.text.SimpleDateFormat = SimpleDateFormat

scala> val startDate = new java.sql.Date(dateParser.parse("01-01-2011").getTime())
startDate: java.sql.Date = 2011-01-01

scala> val endDate = new java.sql.Date(dateParser.parse("01-02-2011").getTime())
endDate: java.sql.Date = 2011-02-01

scala> db.withSession { implicit session =>
 Tables.transactions.filter { 
 _.date.between(startDate, endDate)
 }.length.run 
}
Int = 9772

```

等同于 SQL 中的 `IN (...)` 操作符，用于选择特定集合中的值的是 `inSet`。例如，要选择所有对巴拉克·奥巴马和米特·罗姆尼的交易，我们可以使用以下命令：

```py
scala> val candidateList = List("Obama, Barack", "Romney, Mitt")
candidateList: List[String] = List(Obama, Barack, Romney, Mitt)

scala> val donationCents = db.withSession { implicit session =>
 Tables.transactions.filter {
 _.candidate.inSet(candidateList)
 }.map { _.amount }.sum.run
}
donationCents: Option[Long] = Some(2874484657)

scala> val donationDollars = donationCents.map { _ / 100 }
donationDollars: Option[Long] = Some(28744846)

```

因此， Mitt Romney 和 Barack Obama 之间，他们共收到了超过 2800 万美元的注册捐款。

我们还可以使用 `!` 运算符否定布尔列。例如，要计算除巴拉克·奥巴马和米特·罗姆尼之外所有候选人的总捐款金额：

```py
scala> db.withSession { implicit session =>
 Tables.transactions.filter { 
 ! _.candidate.inSet(candidateList) 
 }.map { _.amount }.sum.run
}.map { _ / 100 }
Option[Long] = Some(1930747)

```

列操作是通过在基`Column`实例上隐式转换来添加的。有关字符串列上可用的所有方法的完整列表，请参阅`StringColumnExtensionMethods`类的 API 文档（[`slick.typesafe.com/doc/2.1.0/api/#scala.slick.lifted.StringColumnExtensionMethods`](http://slick.typesafe.com/doc/2.1.0/api/#scala.slick.lifted.StringColumnExtensionMethods)）。对于布尔列上可用的方法，请参阅`BooleanColumnExtensionMethods`类的 API 文档（[`slick.typesafe.com/doc/2.1.0/api/#scala.slick.lifted.BooleanColumnExtensionMethods`](http://slick.typesafe.com/doc/2.1.0/api/#scala.slick.lifted.BooleanColumnExtensionMethods)）。对于数值列上可用的方法，请参阅`NumericColumnExtensionMethods`的 API 文档（[`slick.typesafe.com/doc/2.1.0/api/#scala.slick.lifted.NumericColumnExtensionMethods`](http://slick.typesafe.com/doc/2.1.0/api/#scala.slick.lifted.NumericColumnExtensionMethods)）。

# 使用“按组分组”进行聚合

Slick 还提供了一个`groupBy`方法，其行为类似于原生 Scala 集合的`groupBy`方法。让我们获取每个候选人的所有捐款的候选人列表：

```py
scala> val grouped = Tables.transactions.groupBy { _.candidate }
grouped: scala.slick.lifted.Query[(scala.slick.lifted.Column[...

scala> val aggregated = grouped.map {
 case (candidate, group) =>
 (candidate -> group.map { _.amount }.sum)
}
aggregated: scala.slick.lifted.Query[(scala.slick.lifted.Column[...

scala> val groupedDonations = db.withSession { 
 implicit session => aggregated.list 
}
groupedDonations: List[(String, Option[Long])] = List((Bachmann, Michele,Some(7439272)),...

```

让我们分解一下。第一条语句`transactions.groupBy { _.candidate }`指定了分组的关键字。你可以将其视为构建一个中间列表，其中包含`(String, List[Transaction])`元组，将组关键字映射到满足此关键字的表的所有行列表。这种行为与在 Scala 集合上调用`groupBy`相同。

`groupBy`调用必须后跟一个`map`来聚合组。传递给`map`的函数必须将`groupBy`调用创建的`(String, List[Transaction])`元组对作为其唯一参数。`map`调用负责聚合`List[Transaction]`对象。我们选择首先提取每个交易的`amount`字段，然后对这些字段进行求和。最后，我们在整个管道上调用`.list`来实际运行查询。这仅仅返回一个 Scala 列表。让我们将总捐款从分转换为美元：

```py
scala> val groupedDonationDollars = groupedDonations.map {
 case (candidate, donationCentsOption) =>
 candidate -> (donationCentsOption.getOrElse(0L) / 100)
}
groupedDonationDollars: List[(String, Long)] = List((Bachmann, Michele,74392),...

scala> groupedDonationDollars.sortBy { 
 _._2 
}.reverse.foreach { println }
(Romney, Mitt,20248496)
(Obama, Barack,8496347)
(Paul, Ron,565060)
(Santorum, Rick,334926)
(Perry, Rick,301780)
(Gingrich, Newt,277079)
(Cain, Herman,210768)
(Johnson, Gary Earl,83610)
(Bachmann, Michele,74392)
(Pawlenty, Timothy,42500)
(Huntsman, Jon,23571)
(Roemer, Charles E. 'Buddy' III,8579)
(Stein, Jill,5270)
(McCotter, Thaddeus G,3210)

```

# 访问数据库元数据

通常，尤其是在开发期间，你可能从删除（如果存在）表并重新创建它开始脚本。我们可以通过通过`MTable`对象访问数据库元数据来检查表是否已定义。要获取与特定模式匹配的表列表，我们可以运行`MTable.getTables(pattern)`：

```py
scala> import slick.jdbc.meta.MTable
import slick.jdbc.meta.MTable

scala> db.withSession { implicit session =>
 MTable.getTables("transactions").list
}
List[scala.slick.jdbc.meta.MTable] = List(MTable(MQName(fec.transactions),TABLE,,None,None,None) ...)

```

因此，为了删除（如果存在）交易表，我们可以运行以下操作：

```py
scala> db.withSession { implicit session =>
 if(MTable.getTables("transactions").list.nonEmpty) {
 Tables.transactions.ddl.drop
 }
}

```

`MTable`实例包含关于表的大量元数据。如果你在前一个示例中删除了它，现在就重新创建`transactions`表。然后，为了找到关于表的主键的信息：

```py
scala> db.withSession { implicit session =>
 val tableMeta = MTable.getTables("transactions").first
 tableMeta.getPrimaryKeys.list
}
List[MPrimaryKey] = List(MPrimaryKey(MQName(test.transactions),id,1,Some(PRIMARY)))

```

要获取`MTable`实例上可用的方法完整列表，请参阅 Slick 文档([`slick.typesafe.com/doc/2.1.0/api/index.html#scala.slick.jdbc.meta.MTable`](http://slick.typesafe.com/doc/2.1.0/api/index.html#scala.slick.jdbc.meta.MTable))。

# Slick 与 JDBC 的比较

本章和上一章介绍了两种不同的与 SQL 交互的方式。在上一章中，我们描述了如何使用 JDBC 并在其之上构建扩展以使其更易于使用。在本章中，我们介绍了 Slick，这是一个在 JDBC 之上提供函数式接口的库。

你应该选择哪种方法？如果你正在启动一个新项目，你应该考虑使用 Slick。即使你花费相当多的时间编写位于 JDBC 之上的包装器，你也不太可能达到 Slick 提供的流畅性。

如果你正在对一个大量使用 JDBC 的现有项目进行工作，我希望上一章的示例表明，只需一点时间和努力，你就可以编写 JDBC 包装器，以减少 JDBC 的命令式风格和 Scala 的函数式方法之间的阻抗。

# 摘要

在前两章中，我们广泛地探讨了如何从 Scala 查询关系型数据库。在本章中，你学习了如何使用 Slick，这是一个“函数式关系型”映射器，允许像与 Scala 集合交互一样与 SQL 数据库交互。

在下一章中，你将学习如何通过查询 Web API 来摄取数据。

# 参考文献

要了解更多关于 Slick 的信息，你可以参考 Slick 文档([`slick.typesafe.com/doc/2.1.0/`](http://slick.typesafe.com/doc/2.1.0/))及其 API 文档([`slick.typesafe.com/doc/2.1.0/api/#package`](http://slick.typesafe.com/doc/2.1.0/api/#package))。

# 第七章。Web API

数据科学家和数据工程师从各种不同的来源获取数据。通常，数据可能以 CSV 文件或数据库转储的形式出现。有时，我们必须通过 Web API 获取数据。

个人或组织建立 Web API 以通过互联网（或内部网络）向程序分发数据。与数据旨在由 Web 浏览器消费并显示给用户的网站不同，Web API 提供的数据对查询它的程序类型是中立的。服务于 HTML 的 Web 服务器和支撑 API 的 Web 服务器基本上以相同的方式进行查询：通过 HTTP 请求。

我们已经在第四章中看到了一个 Web API 的例子，*并行集合和未来*，其中我们查询了“Markit on demand”API 以获取当前的股票价格。在本章中，我们将更详细地探讨如何与 Web API 交互；具体来说，如何将 API 返回的数据转换为 Scala 对象，以及如何通过 HTTP 头（例如，用于身份验证）向请求添加额外信息。

“按需 Markit”API 返回的数据格式化为 XML 对象，但越来越多的新 Web API 返回的数据格式化为 JSON。因此，本章我们将重点关注 JSON，但这些概念很容易应用到 XML 上。

JSON 是一种用于格式化结构化数据的语言。许多读者在过去可能已经遇到过 JSON，但如果没有，本章后面将简要介绍其语法和概念。你会发现它非常直观。

在本章中，我们将轮询 GitHub API。在过去的几年里，GitHub 已经成为开源软件协作的事实上工具。它提供了一个功能强大、特性丰富的 API，可以以编程方式访问网站上的几乎所有数据。

让我们来尝尝我们能做什么。在您的网络浏览器地址栏中输入`api.github.com/users/odersky`。这将返回 API 提供的特定用户（在这种情况下是 Martin Odersky）的数据：

```py
{
  "login": "odersky",
  "id": 795990,
  ...
  "public_repos": 8,
  "public_gists": 3,
  "followers": 707,
  "following": 0,
  "created_at": "2011-05-18T14:51:21Z",
  "updated_at": "2015-09-15T15:14:33Z"
}
```

数据以 JSON 对象的形式返回。本章致力于学习如何以编程方式访问和解析这些数据。在第十三章《使用 Play 的 Web API》中，你将学习如何构建你自己的 Web API。

### 小贴士

GitHub API 非常广泛且文档齐全。在本章中，我们将探索 API 的一些功能。要查看 API 的完整范围，请访问文档([`developer.github.com/v3/`](https://developer.github.com/v3/))。

# JSON 快速浏览

JSON 是一种用于传输结构化数据的格式。它灵活，易于计算机生成和解析，对于人类来说相对易于阅读。它已成为持久化程序数据结构和在程序之间传输数据的一种非常常见的方式。

JSON 有四种基本类型：**数字**、**字符串**、**布尔值**和**null**，以及两种复合类型：**数组**和**对象**。对象是无序的键值对集合，其中键始终是字符串，值可以是任何简单或复合类型。我们已经看到了一个 JSON 对象：API 调用`api.github.com/users/odersky`返回的数据。

数组是有序的简单或复合类型列表。例如，在您的浏览器中输入[api.github.com/users/odersky/repos](http://api.github.com/users/odersky/repos)，以获取一个对象数组，每个对象代表一个 GitHub 仓库：

```py
[
  {
    "id": 17335228,
    "name": "dotty",
    "full_name": "odersky/dotty",
    ...
  },
  {
    "id": 15053153,
    "name": "frontend",
    "full_name": "odersky/frontend",
    ...
  },
  ...
]
```

我们可以通过在对象或数组内部嵌套其他对象来构建复杂结构。然而，大多数 Web API 返回的 JSON 结构最多只有一两个嵌套层级。如果你不熟悉 JSON，我鼓励你通过你的网络浏览器探索 GitHub API。

# 查询 Web API

从 Scala 中查询 Web API 最简单的方法是使用 `Source.fromURL`。我们已经在 第四章，*并行集合和未来* 中使用过它，当时我们查询了 "Markit on demand" API。`Source.fromURL` 提供了一个类似于 `Source.fromFile` 的接口：

```py
scala> import scala.io._
import scala.io._

scala> val response = Source.fromURL(
 "https://api.github.com/users/odersky"
).mkString
response: String = {"login":"odersky","id":795990, ...

```

`Source.fromURL` 返回响应字符的迭代器。我们使用其 `.mkString` 方法将迭代器实体化为一个字符串。现在我们有了作为 Scala 字符串的响应。下一步是使用 JSON 解析器解析字符串。

# Scala 中的 JSON – 一个模式匹配练习

Scala 中有几个用于操作 JSON 的库。我们更喜欢 json4s，但如果你是另一个 JSON 库的死忠粉丝，你应该能够轻松地适应本章中的示例。让我们创建一个包含对 `json4s` 依赖的 `build.sbt` 文件：

```py
// build.sbt
scalaVersion := "2.11.7"

libraryDependencies += "org.json4s" %% "json4s-native" % "3.2.11"
```

我们可以导入 `json4s` 到 SBT 控制台会话中：

```py
scala> import org.json4s._
import org.json4s._

scala> import org.json4s.native.JsonMethods._
import org.json4s.native.JsonMethods._

```

让我们使用 `json4s` 解析 GitHub API 查询的响应：

```py
scala> val jsonResponse = parse(response)
jsonResponse: org.json4s.JValue = JObject(List((login,JString(odersky)),(id,JInt(795990)),...

```

`parse` 方法接受一个字符串（包含格式良好的 JSON），并将其转换为 `JValue`，这是所有 `json4s` 对象的超类型。此特定查询的响应运行时类型为 `JObject`，它是表示 JSON 对象的 `json4s` 类型。

`JObject` 是一个围绕 `List[JField]` 的包装器，`JField` 表示对象中的单个键值对。我们可以使用 *提取器* 来访问这个列表：

```py
scala> val JObject(fields) = jsonResponse
fields: List[JField] = List((login,Jstring(odersky)),...

```

这里发生了什么？通过编写 `val JObject(fields) = ...`，我们告诉 Scala：

+   右侧的运行时类型为 `JObject`

+   进入 `JObject` 实例并将字段列表绑定到常量 `fields`

熟悉 Python 的读者可能会注意到与元组解包的相似之处，尽管 Scala 提取器要强大得多，也更加灵活。提取器被广泛用于从 `json4s` 类型中提取 Scala 类型。

### 小贴士

**使用案例类进行模式匹配**

Scala 编译器是如何知道如何处理像这样的提取器的：

```py
val JObject(fields) = ...
```

`JObject` 是一个具有以下构造函数的案例类：

```py
case class JObject(obj:List[JField])
```

所有案例类都附带一个提取器，它可以精确地反转构造函数。因此，编写 `val JObject(fields)` 将将 `fields` 绑定到 `JObject` 的 `obj` 属性。有关提取器如何工作的更多详细信息，请参阅 附录，*模式匹配和提取器*。

现在我们已经从 `JObject` 中提取了 `fields`，这是一个（普通的旧 Scala）字段列表。`JField` 是一个键值对，键是一个字符串，值是 `JValue` 的子类型。同样，我们可以使用提取器来提取字段中的值：

```py
scala> val firstField = fields.head
firstField: JField = (login,JString(odersky))

scala> val JField(key, JString(value)) = firstField
key: String = login
value: String = odersky

```

我们将右侧与模式 `JField(_, JString(_))` 匹配，将第一个元素绑定到 `key`，第二个绑定到 `value`。如果右侧不匹配模式会发生什么？

```py
scala> val JField(key, JInt(value)) = firstField
scala.MatchError: (login,JString(odersky)) (of class scala.Tuple2)
...

```

代码在运行时抛出`MatchError`。以下示例展示了嵌套模式匹配的强大功能：在一行代码中，我们成功验证了`firstField`的类型，确认其值为`JString`类型，并将键和值分别绑定到`key`和`value`变量。作为另一个例子，如果我们*知道*第一个字段是登录字段，我们既可以验证这一点，也可以提取其值：

```py
scala> val JField("login", JString(loginName)) = firstField
loginName: String = odersky

```

注意这种编程风格是*声明式*而不是*命令式*：我们在右侧声明我们想要一个`JField("login", JString(_))`变量。然后让语言找出如何检查变量类型。模式匹配是函数式语言中的一个常见主题。

我们还可以在遍历字段时使用模式匹配。当在 for 循环中使用时，模式匹配定义了一个*部分函数*：只有与模式匹配的元素才会通过循环。这让我们能够过滤出匹配模式的元素集合，并对这些元素应用转换。例如，我们可以从我们的`fields`列表中提取每个字符串字段：

```py
scala> for {
 JField(key, JString(value)) <- fields
} yield (key -> value)
List[(String, String)] = List((login,odersky), (avatar_url,https://avatars.githubusercontent.com/...

```

我们可以使用它来搜索特定字段。例如，提取`"followers"`字段：

```py
scala> val followersList = for {
 JField("followers", JInt(followers)) <- fields
} yield followers
followersList: List[Int] = List(707)

scala> val followers = followersList.headOption
blogURL: Option[Int] = Some(707)

```

我们首先提取所有匹配模式`JField("follower", JInt(_))`的字段，返回`JInt`内部的整数。由于源集合`fields`是一个列表，这返回一个整数列表。然后我们使用`headOption`从该列表中提取第一个值，它如果列表至少有一个元素，则返回列表的头部，如果列表为空，则返回`None`。

我们不仅限于一次提取一个字段。例如，要一起提取`"id"`和`"login"`字段：

```py
scala> {
 for {
 JField("login", JString(loginName)) <- fields
 JField("id", JInt(id)) <- fields
 } yield (id -> loginName)
}.headOption 
Option[(BigInt, String)] = Some((795990,odersky))

```

Scala 的模式匹配和提取器为你提供了一种极其强大的方法来遍历`json4s`树，提取我们需要的字段。

## JSON4S 类型

我们已经发现了`json4s`类型层次结构的一部分：字符串被包裹在`JString`对象中，整数（或大整数）被包裹在`JInt`中，依此类推。在本节中，我们将退后一步，正式化类型结构和它们提取到的 Scala 类型。这些都是`json4s`的运行时类型：

+   `val JString(s) // => 提取为 String`

+   `val JDouble(d) // => 提取为 Double`

+   `val JDecimal(d) // => 提取为 BigDecimal`

+   `val JInt(i) // => 提取为 BigInt`

+   `val JBool(b) // => 提取为布尔值`

+   `val JObject(l) // => 提取为[JField]列表`

+   `val JArray(l) // => 提取为[JValue]列表`

+   `JNull // => 表示 JSON null`

所有这些类型都是`JValue`的子类。`parse`的编译时结果是`JValue`，你通常需要使用提取器将其转换为具体类型。

层次结构中的最后一个类型是`JField`，它表示键值对。`JField`只是`(String, JValue)`元组的类型别名。因此，它不是`JValue`的子类型。我们可以使用以下提取器提取键和值：

```py
val JField(key, JInt(value)) = ...
```

## 使用 XPath 提取字段

在前面的章节中，你学习了如何使用提取器遍历 JSON 对象。在本节中，我们将探讨另一种遍历 JSON 对象和提取特定字段的方法：*XPath DSL*（领域特定语言）。XPath 是一种用于遍历树状结构的查询语言。它最初是为在 XML 文档中定位特定节点而设计的，但它同样适用于 JSON。当我们从“Markit on demand”API 返回的 XML 文档中提取股票价格时，我们已经看到了 XPath 语法的示例，这在第四章中，*并行集合和未来*。我们使用 `r \ "LastPrice"` 提取了标签为 `"LastPrice"` 的节点。`\` 操作符是由 `scala.xml` 包定义的。

`json4s` 包提供了一个类似的 DSL 来从 `JObject` 实例中提取字段。例如，我们可以从 JSON 对象 `jsonResponse` 中提取 `"login"` 字段：

```py
scala> jsonResponse \ "login"
org.json4s.JValue = JString(odersky)

```

这返回了一个 `JValue`，我们可以使用提取器将其转换为 Scala 字符串：

```py
scala> val JString(loginName) = jsonResponse \ "login"
loginName: String = odersky

```

注意 XPath DSL 和遍历文件系统的相似性：我们可以将 `JObject` 实例视为目录。字段名对应文件名，字段值对应文件内容。这在嵌套结构中更为明显。GitHub API 的 `users` 端点没有嵌套文档，所以让我们尝试另一个端点。我们将查询与这本书对应的仓库的 API："[`api.github.com/repos/pbugnion/s4ds`](https://api.github.com/repos/pbugnion/s4ds)"。响应具有以下结构：

```py
{
  "id": 42269470,
  "name": "s4ds",
  ...
  "owner": { "login": "pbugnion", "id": 1392879 ... }
  ...
}
```

让我们获取这个文档，并使用 XPath 语法提取仓库所有者的登录名：

```py
scala> val jsonResponse = parse(Source.fromURL(
 "https://api.github.com/repos/pbugnion/s4ds"
).mkString)
jsonResponse: JValue = JObject(List((id,JInt(42269470)), (name,JString(s4ds))...

scala> val JString(ownerLogin) = jsonResponse \ "owner" \ "login"
ownerLogin: String = pbugnion

```

再次，这类似于遍历文件系统：`jsonResponse \ "owner"` 返回与 `"owner"` 对象相对应的 `JObject`。这个 `JObject` 可以进一步查询 `"login"` 字段，返回与该键关联的值 `JString(pbugnion)`。

如果 API 响应是一个数组呢？文件系统类比就有些不适用了。让我们查询列出马丁·奥德斯基仓库的 API 端点：[`api.github.com/users/odersky/repos`](https://api.github.com/users/odersky/repos)。响应是一个包含 JSON 对象的数组，每个对象代表一个仓库：

```py
[
  {
    "id": 17335228,
    "name": "dotty",
    "size": 14699,
    ...
  },
  {
    "id": 15053153,
    "name": "frontend",
    "size": 392
    ...
  },
  {
    "id": 2890092,
    "name": "scala",
    "size": 76133,
    ...
  },
  ...
]
```

让我们获取这个文档并将其解析为 JSON：

```py
scala> val jsonResponse = parse(Source.fromURL(
 "https://api.github.com/users/odersky/repos"
).mkString)
jsonResponse: JValue = JArray(List(JObject(List((id,JInt(17335228)), (name,Jstring(dotty)), ...

```

这返回了一个 `JArray`。XPath DSL 在 `JArray` 上的工作方式与在 `JObject` 上相同，但现在，它返回的是一个与数组中每个对象的路径匹配的字段数组。让我们获取所有马丁·奥德斯基的仓库的大小：

```py
scala> jsonResponse \ "size"
JValue = JArray(List(JInt(14699), JInt(392), ...

```

现在，我们有一个包含每个仓库中 `"size"` 字段值的 `JArray`。我们可以使用 `for` 理解遍历这个数组，并使用提取器将元素转换为 Scala 对象：

```py
scala> for {
 JInt(size) <- (jsonResponse \ "size")
} yield size
List[BigInt] = List(14699, 392, 76133, 32010, 98166, 1358, 144, 273)

```

因此，结合提取器和 XPath DSL，我们得到了从 JSON 对象中提取信息的有力、互补的工具。

XPath 语法远比这里能涵盖的要多，包括从当前根的任何深度提取嵌套字段的能力，或者匹配谓词或特定类型的字段。我们发现，设计良好的 API 可以消除许多这些更强大功能的需求，但请查阅文档（`json4s.org`）以了解您可以做什么的概述。

在下一节中，我们将探讨如何直接将 JSON 提取到案例类中。

# 使用案例类进行提取

在前面的章节中，我们使用 Scala 提取器从 JSON 响应中提取了特定字段。我们可以做得更好，提取完整的案例类。

当我们超出 REPL 时，编程最佳实践规定我们应该尽快从 `json4s` 类型移动到 Scala 对象，而不是在程序中传递 `json4s` 类型。从 `json4s` 类型转换为 Scala 类型（或表示域对象的案例类）是良好的实践，因为：

+   它将程序与从 API 收到的数据结构解耦，我们对这些结构几乎没有控制权。

+   它提高了类型安全性：从编译器的角度来看，`JObject` 总是 `JObject`，无论它包含哪些字段。相比之下，编译器永远不会将 `User` 错误地认为是 `Repository`。

`Json4s` 允许我们直接从 `JObject` 实例中提取案例类，这使得将 `JObject` 实例转换为自定义类型层的编写变得简单。

让我们定义一个表示 GitHub 用户的案例类：

```py
scala> case class User(id:Long, login:String)
defined class User

```

要从 `JObject` 中提取案例类，我们首先必须定义一个隐式 `Formats` 值，该值定义了简单类型应该如何序列化和反序列化。我们将使用 `json4s` 提供的默认 `DefaultFormats`：

```py
scala> implicit val formats = DefaultFormats
formats: DefaultFormats.type = DefaultFormats$@750e685a

```

我们现在可以提取 `User` 的实例。让我们为马丁·奥德斯基（Martin Odersky）做这个操作：

```py
scala> val url = "https://api.github.com/users/odersky"
url: String = https://api.github.com/users/odersky

scala> val jsonResponse = parse(Source.fromURL(url).mkString)
jsonResponse: JValue = JObject(List((login,JString(odersky)), ...

scala> jsonResponse.extract[User]
User = User(795990,odersky)

```

只要对象格式良好，这种方法就有效。`extract` 方法在 `JObject` 中寻找与 `User` 属性匹配的字段。在这种情况下，`extract` 会注意到 `JObject` 包含 `"login": "odersky"` 字段，并且 `JString("odersky")` 可以转换为 Scala 字符串，因此它将 `"odersky"` 绑定到 `User` 中的 `login` 属性。

如果属性名称与 JSON 对象中的字段名称不同怎么办？我们必须首先将对象转换为具有正确字段的形式。例如，让我们将 `User` 类中的 `login` 属性重命名为 `userName`：

```py
scala> case class User(id:Long, userName:String)
defined class User

```

如果我们尝试在 `jsonResponse` 上使用 `extract[User]`，我们将得到一个映射错误，因为反序列化器在响应中缺少 `login` 字段。我们可以通过在 `jsonResponse` 上使用 `transformField` 方法来重命名 `login` 字段来修复这个问题：

```py
scala> jsonResponse.transformField { 
 case("login", n) => "userName" -> n 
}.extract[User]
User = User(795990,odersky)

```

关于可选字段怎么办？假设 GitHub API 返回的 JSON 对象并不总是包含 `login` 字段。我们可以在对象模型中通过将 `login` 参数的类型指定为 `Option[String]` 而不是 `String` 来表示这一点：

```py
scala> case class User(id:Long, login:Option[String])
defined class User

```

这正如你所期望的那样工作。当响应包含非空的`login`字段时，调用`extract[User]`会将其反序列化为`Some(value)`，如果它缺失或为`JNull`，则会产生`None`：

```py
scala> jsonResponse.extract[User]
User = User(795990,Some(odersky))

scala> jsonResponse.removeField { 
 case(k, _) => k == "login" // remove the "login" field
}.extract[User]
User = User(795990,None)

```

让我们将这个功能封装在一个小的程序中。该程序将接受一个命令行参数，即用户的登录名，提取一个`User`实例，并将其打印到屏幕上：

```py
// GitHubUser.scala

import scala.io._
import org.json4s._
import org.json4s.native.JsonMethods._

object GitHubUser {

  implicit val formats = DefaultFormats

  case class User(id:Long, userName:String)

  /** Query the GitHub API corresponding to `url` 
    * and convert the response to a User.
    */
  def fetchUserFromUrl(url:String):User = {
    val response = Source.fromURL(url).mkString
    val jsonResponse = parse(response)
    extractUser(jsonResponse)
  }

  /** Helper method for transforming the response to a User */
  def extractUser(obj:JValue):User = {
    val transformedObject = obj.transformField {
      case ("login", name) => ("userName", name)
    }
    transformedObject.extract[User]
  }

  def main(args:Array[String]) {
    // Extract username from argument list
    val name = args.headOption.getOrElse { 
      throw new IllegalArgumentException(
        "Missing command line argument for user.")
    }

    val user = fetchUserFromUrl(
      s"https://api.github.com/users/$name")

    println(s"** Extracted for $name:")
    println()
    println(user)

  }

}
```

我们可以从 SBT 控制台按照以下方式运行此程序：

```py
$ sbt
> runMain GitHubUser pbugnion
** Extracted for pbugnion:
User(1392879,pbugnion)

```

# 使用未来（futures）进行并发和异常处理

尽管我们在上一节中编写的程序可以工作，但它非常脆弱。如果我们输入一个不存在的用户名，或者 GitHub API 发生变化或返回格式错误的响应，它将会崩溃。我们需要使其具有容错性。

如果我们还想获取多个用户呢？按照目前的程序编写方式，它是完全单线程的。`fetchUserFromUrl`方法会向 API 发起调用并阻塞，直到 API 返回数据。一个更好的解决方案是并行地获取多个用户。

正如你在第四章，*并行集合和未来*中学到的，实现容错性和并行执行有两种简单的方法：我们可以将所有用户名放入并行集合中，并将获取和提取用户的代码封装在`Try`块中，或者我们可以将每个查询封装在未来的框架中。

当查询网络 API 时，有时请求可能会异常地长时间。为了避免这阻碍其他线程，最好依赖于未来（futures）而不是并行集合（parallel collections）来实现并发，正如我们在第四章末尾的*并行集合或未来？*部分所看到的，*并行集合和未来*。

让我们重写上一节中的代码，以并行且容错地获取多个用户。我们将把`fetchUserFromUrl`方法改为异步查询 API。这与第四章中的内容没有太大区别，我们在其中查询了"Markit on demand" API：

```py
// GitHubUserConcurrent.scala

import scala.io._
import scala.concurrent._
import scala.concurrent.duration._
import ExecutionContext.Implicits.global
import scala.util._

import org.json4s._
import org.json4s.native.JsonMethods._

object GitHubUserConcurrent {

  implicit val formats = DefaultFormats

  case class User(id:Long, userName:String)

  // Fetch and extract the `User` corresponding to `url`
  def fetchUserFromUrl(url:String):Future[User] = {
    val response = Future { Source.fromURL(url).mkString }
    val parsedResponse = response.map { r => parse(r) }
    parsedResponse.map { extractUser }
  }

  // Helper method for extracting a user from a JObject
  def extractUser(jsonResponse:JValue):User = {
    val o = jsonResponse.transformField {
      case ("login", name) => ("userName", name)
    }
    o.extract[User]
  }

  def main(args:Array[String]) {
    val names = args.toList

    // Loop over each username and send a request to the API 
    // for that user 
    val name2User = for {
      name <- names
      url = s"https://api.github.com/users/$name"
      user = fetchUserFromUrl(url)
    } yield name -> user

    // callback function
    name2User.foreach { case(name, user) =>
      user.onComplete {
        case Success(u) => println(s" ** Extracted for $name: $u")
        case Failure(e) => println(s" ** Error fetching $name:$e")
      }
    }

    // Block until all the calls have finished.
    Await.ready(Future.sequence(name2User.map { _._2 }), 1 minute)
  }
}
```

让我们通过`sbt`运行这段代码：

```py
$ sbt
> runMain GitHubUserConcurrent odersky derekwyatt not-a-user-675
 ** Error fetching user not-a-user-675: java.io.FileNotFoundException: https://api.github.com/users/not-a-user-675
 ** Extracted for odersky: User(795990,odersky)
 ** Extracted for derekwyatt: User(62324,derekwyatt)

```

代码本身应该是直截了当的。这里使用的所有概念都已在本章或第四章中探讨过，除了最后一行：

```py
Await.ready(Future.sequence(name2User.map { _._2 }), 1 minute)
```

这条语句指示程序等待我们列表中的所有未来都已完成。`Await.ready(..., 1 minute)`将一个未来作为其第一个参数，并在该未来返回之前阻塞执行。第二个参数是对这个未来的超时。唯一的缺点是我们需要将单个未来传递给`Await`，而不是未来列表。我们可以使用`Future.sequence`将一组未来合并为一个单一的未来。这个未来将在序列中的所有未来都完成后完成。

# 身份验证 – 添加 HTTP 头部

到目前为止，我们一直在使用未经身份验证的 GitHub API。这限制了我们每小时只能进行六十次请求。现在我们可以在并行查询 API 的情况下，几秒钟内就能超过这个限制。

幸运的是，如果您在查询 API 时进行身份验证，GitHub 会慷慨得多。限制增加到每小时 5,000 次请求。您必须有一个 GitHub 用户账户才能进行身份验证，所以如果您需要，请现在就创建一个账户。创建账户后，导航到[`github.com/settings/tokens`](https://github.com/settings/tokens)并点击**生成新令牌**按钮。接受默认设置，并在屏幕上输入令牌描述，应该会出现一个长的十六进制数字。现在先复制令牌。

## HTTP – 快速概述

在使用我们新生成的令牌之前，让我们花几分钟时间回顾一下 HTTP 是如何工作的。

HTTP 是在不同计算机之间传输信息的协议。这是我们本章一直在使用的协议，尽管 Scala 在`Source.fromURL`调用中隐藏了这些细节。它也是您在将网络浏览器指向网站时使用的协议，例如。

在 HTTP 中，一台计算机通常会向远程服务器发送一个*请求*，服务器会返回一个*响应*。请求包含一个*动词*，它定义了请求的类型，以及一个标识*资源*的 URL。例如，当我们在我们浏览器中键入[api.github.com/users/pbugnion](http://api.github.com/users/pbugnion)时，这被转换为一个针对`users/pbugnion`资源的 GET（动词）请求。我们迄今为止所做的一切调用都是 GET 请求。您可能会使用不同类型的请求，例如 POST 请求，来修改（而不仅仅是查看）GitHub 上的某些内容。

除了动词和资源之外，HTTP 请求还有两个其他部分：

+   *头部*包含了关于请求的元数据，例如响应的预期格式和字符集或身份验证凭据。头部只是一个键值对的列表。我们将使用`Authorization`头部将我们刚刚生成的 OAuth 令牌传递给 API。这篇维基百科文章列出了常用的头部字段：[en.wikipedia.org/wiki/List_of_HTTP_header_fields](http://en.wikipedia.org/wiki/List_of_HTTP_header_fields)。

+   请求体在 GET 请求中不被使用，但在修改它们查询的资源时变得重要。例如，如果我想通过编程方式在 GitHub 上创建一个新的仓库，我会向`/pbugnion/repos`发送 POST 请求。POST 体将是一个描述新仓库的 JSON 对象。在本章中，我们不会使用请求体。

## 在 Scala 中向 HTTP 请求添加头信息

我们将在 HTTP 请求中传递 OAuth 令牌作为头信息。不幸的是，`Source.fromURL`方法在创建 GET 请求时添加头信息并不特别适合。我们将改用库，`scalaj-http`。

让我们在`build.sbt`的依赖项中添加`scalaj-http`：

```py
libraryDependencies += "org.scalaj" %% "scalaj-http" % "1.1.6"
```

我们现在可以导入`scalaj-http`：

```py
scala> import scalaj.http._
import scalaj.http._

```

我们首先创建一个`HttpRequest`对象：

```py
scala> val request = Http("https://api.github.com/users/pbugnion")
request:scalaj.http.HttpRequest = HttpRequest(api.github.com/users/pbugnion,GET,...

```

我们现在可以向请求添加授权头（在此处添加你自己的令牌字符串）：

```py
scala> val authorizedRequest = request.header("Authorization", "token e836389ce ...")
authorizedRequest:scalaj.http.HttpRequest = HttpRequest(api.github.com/users/pbugnion,GET,...

```

### 提示

`.header`方法返回一个新的`HttpRequest`实例。它不会就地修改请求。因此，仅仅调用`request.header(...)`实际上并没有将头信息添加到请求本身，这可能会引起混淆。

让我们发起请求。我们通过请求的`asString`方法来完成，该方法查询 API，获取响应，并将其解析为 Scala `String`：

```py
scala> val response = authorizedRequest.asString
response:scalaj.http.HttpResponse[String] = HttpResponse({"login":"pbugnion",...

```

响应由三个部分组成：

+   状态码，对于成功的请求应该是`200`：

    ```py
    scala> response.code 
    Int = 200

    ```

+   响应体，这是我们感兴趣的部分：

    ```py
    scala> response.body 
    String = {"login":"pbugnion","id":1392879,...

    ```

+   响应头（关于响应的元数据）：

    ```py
    scala> response.headers 
    Map[String,String] = Map(Access-Control-Allow-Credentials -> true, ...

    ```

要验证授权是否成功，查询`X-RateLimit-Limit`头：

```py
scala> response.headers("X-RateLimit-Limit")
String = 5000

```

这个值是你可以从单个 IP 地址向 GitHub API 发起的最大每小时请求数量。

现在我们已经对如何向 GET 请求添加认证有了些了解，让我们修改我们的用户获取脚本，使用 OAuth 令牌进行认证。我们首先需要导入`scalaj-http`：

```py
import scalaj.http._
```

将令牌的值注入到代码中可能有些棘手。你可能想将其硬编码，但这会阻止你共享代码。更好的解决方案是使用*环境变量*。环境变量是在你的终端会话中存在的一组变量，该会话中的所有进程都可以访问这些变量。要获取当前环境变量的列表，请在 Linux 或 Mac OS 上输入以下内容：

```py
$ env
HOME=/Users/pascal
SHELL=/bin/zsh
...

```

在 Windows 上，等效的命令是`SET`。让我们将 GitHub 令牌添加到环境变量中。在 Mac OS 或 Linux 上使用以下命令：

```py
$ export GHTOKEN="e83638..." # enter your token here

```

在 Windows 上，使用以下命令：

```py
$ SET GHTOKEN="e83638..."

```

如果你打算在许多项目中重用这个环境变量，那么在每次会话中输入`export GHTOKEN=...`会很快变得令人厌烦。一个更持久的解决方案是将`export GHTOKEN="e83638…"`添加到你的 shell 配置文件中（如果你使用 Bash，则是你的`.bashrc`文件）。只要你的`.bashrc`只能被用户读取，这将是安全的。任何新的 shell 会话都将能够访问`GHTOKEN`环境变量。

我们可以使用`sys.env`从 Scala 程序中访问环境变量，它返回一个包含变量的`Map[String, String]`。让我们在我们的类中添加一个`lazy val token`，包含`token`值：

```py
lazy val token:Option[String] = sys.env.get("GHTOKEN") orElse {
  println("No token found: continuing without authentication")
  None
}
```

现在我们有了令牌，唯一需要更改代码的部分，以添加身份验证，就是`fetchUserFromUrl`方法：

```py
def fetchUserFromUrl(url:String):Future[User] = {
  val baseRequest = Http(url)
  val request = token match {
    case Some(t) => baseRequest.header(
      "Authorization", s"token $t")
    case None => baseRequest
  }
  val response = Future { 
    request.asString.body 
  }
  val parsedResponse = response.map { r => parse(r) }
  parsedResponse.map(extractUser)
}
```

此外，为了获得更清晰的错误信息，我们可以检查响应的状态码是否为 200。由于这很简单，所以留作练习。

# 概述

在本章中，你学习了如何查询 GitHub API，将响应转换为 Scala 对象。当然，仅仅将结果打印到屏幕上并不十分有趣。在下一章中，我们将探讨数据摄取过程的下一步：将数据存储在数据库中。我们将查询 GitHub API 并将结果存储在 MongoDB 数据库中。

在第十三章，*使用 Play 构建 Web API*中，我们将探讨构建我们自己的简单 Web API。

# 参考文献

GitHub API，凭借其详尽的文档，是探索如何构建丰富 API 的好地方。它有一个**入门**部分值得一读：

[`developer.github.com/guides/getting-started/`](https://developer.github.com/guides/getting-started/)

当然，这不仅仅针对 Scala：它使用 cURL 查询 API。

请阅读`json4s`的文档([`json4s.org`](http://json4s.org))和源代码([`github.com/json4s/json4s`](https://github.com/json4s/json4s))以获取完整的参考。这个包的许多部分我们还没有探索，特别是如何从 Scala 构建 JSON。

# 第八章. Scala 和 MongoDB

在第五章，*通过 JDBC 的 Scala 和 SQL*和第六章，*Slick – SQL 的功能接口*中，你学习了如何在 SQL 数据库中插入、转换和读取数据。这些数据库在数据科学中仍然（并且可能仍然）非常受欢迎，但 NoSQL 数据库正在成为强劲的竞争者。

数据存储的需求正在迅速增长。公司正在生产和存储更多的数据点，希望获得更好的商业智能。他们也在组建越来越大的数据科学家团队，所有这些人都需要访问数据存储。随着数据负载的增加，保持恒定的访问时间需要利用并行架构：我们需要将数据库分布在几台计算机上，这样当服务器负载增加时，我们只需添加更多机器来提高吞吐量。

在 MySQL 数据库中，数据自然地分布在不同的表中。复杂的查询需要跨多个表进行连接。这使得在多台计算机上分区数据库变得困难。NoSQL 数据库的出现填补了这一空白。

在本章中，你将学习如何与 MongoDB 交互，这是一个提供高性能且易于分布的开源数据库。MongoDB 是更受欢迎的 NoSQL 数据库之一，拥有强大的社区。它提供了速度和灵活性的合理平衡，使其成为存储具有不确定查询要求的大型数据集（如数据科学中可能发生的情况）的 SQL 的自然替代品。本章中的许多概念和食谱也适用于其他 NoSQL 数据库。

# MongoDB

MongoDB 是一个*面向文档*的数据库。它包含文档集合。每个文档都是一个类似 JSON 的对象：

```py
{
    _id: ObjectId("558e846730044ede70743be9"),
    name: "Gandalf",
    age: 2000,
    pseudonyms: [ "Mithrandir", "Olorin", "Greyhame" ],
    possessions: [ 
        { name: "Glamdring", type: "sword" }, 
        { name: "Narya", type: "ring" }
    ]
}
```

正如 JSON 一样，文档是一组键值对，其中值可以是字符串、数字、布尔值、日期、数组或子文档。文档被分组在集合中，集合被分组在数据库中。

你可能会想，这与 SQL 并没有太大的不同：一个文档类似于一行，一个集合对应一个表。但存在两个重要的区别：

+   文档中的值可以是简单值、数组、子文档或子文档数组。这使得我们可以在单个集合中编码一对一和多对多关系。例如，考虑巫师集合。在 SQL 中，如果我们想为每个巫师存储化名，我们必须使用一个单独的`wizard2pseudonym`表，并为每个巫师-化名对创建一行。在 MongoDB 中，我们只需使用一个数组。在实践中，这意味着我们可以通常用一个文档来表示一个实体（例如客户、交易或巫师）。在 SQL 中，我们通常需要跨多个表进行连接，以检索特定实体的所有信息。

+   MongoDB 是*无模式的*。集合中的文档可以具有不同的字段集，不同文档中同一字段的类型也可以不同。在实践中，MongoDB 集合有一个松散的模式，由客户端或约定强制执行：大多数文档将具有相同字段的子集，字段通常将包含相同的数据类型。具有灵活的模式使得调整数据结构变得容易，因为不需要耗时的`ALTER TABLE`语句。缺点是，在数据库端没有简单的方法来强制执行我们的灵活模式。

注意到`_id`字段：这是一个唯一键。如果我们插入一个没有`_id`字段的文档，MongoDB 将自动生成一个。

本章提供了从 Scala 与 MongoDB 数据库交互的配方，包括维护类型安全和最佳实践。我们不会涵盖高级 MongoDB 功能（如聚合或数据库的分布式）。我们假设您已在您的计算机上安装了 MongoDB（[`docs.mongodb.org/manual/installation/`](http://docs.mongodb.org/manual/installation/)）。对 MongoDB 有非常基本的了解也会有所帮助（我们将在本章末尾讨论一些参考资料，但任何在线可用的基本教程都足以满足本章的需求）。

# 使用 Casbah 连接到 MongoDB

Scala 的官方 MongoDB 驱动程序称为**Casbah**。Casbah 不是完整的驱动程序，而是包装了 Java Mongo 驱动程序，提供了一个更函数式的接口。还有其他 Scala 的 MongoDB 驱动程序，我们将在本章末尾简要讨论。现在，我们将坚持使用 Casbah。

让我们从向我们的`build.sbt`文件添加 Casbah 开始：

```py
scalaVersion := "2.11.7"

libraryDependencies += "org.mongodb" %% "casbah" % "3.0.0"
```

Casbah 还期望`slf4j`绑定（一个 Scala 日志框架）可用，因此让我们也添加`slf4j-nop`：

```py
libraryDependencies += "org.slf4j" % "slf4j-nop" % "1.7.12"
```

我们现在可以启动 SBT 控制台并在 Scala shell 中导入 Casbah：

```py
$ sbt console
scala> import com.mongodb.casbah.Imports._
import com.mongodb.casbah.Imports._

scala> val client = MongoClient()
client: com.mongodb.casbah.MongoClient = com.mongodb.casbah.MongoClient@4ac17318

```

这将连接到默认主机（`localhost`）和默认端口（`27017`）上的 MongoDB 服务器。要连接到不同的服务器，将主机和端口作为参数传递给`MongoClient`：

```py
scala> val client = MongoClient("192.168.1.1", 27017)
client: com.mongodb.casbah.MongoClient = com.mongodb.casbah.MongoClient@584c6b02

```

注意，创建客户端是一个延迟操作：它不会在需要之前尝试连接到服务器。这意味着如果您输入了错误的 URL 或密码，您直到尝试访问服务器上的文档时才会知道。

一旦我们与服务器建立了连接，访问数据库就像使用客户端的`apply`方法一样简单。例如，要访问`github`数据库：

```py
scala> val db = client("github")
db: com.mongodb.casbah.MongoDB = DB{name='github'}

```

然后，我们可以访问`"users"`集合：

```py
scala> val coll = db("users")
coll: com.mongodb.casbah.MongoCollection = users

```

## 使用身份验证连接

MongoDB 支持多种不同的身份验证机制。在本节中，我们假设您的服务器正在使用**SCRAM-SHA-1**机制，但您应该会发现将代码适应不同类型的身份验证很简单。

最简单的身份验证方式是在连接时通过 URI 传递`username`和`password`：

```py
scala> val username = "USER"
username: String = USER

scala> val password = "PASSWORD"
password: String = PASSWORD

scala> val uri = MongoClientURI(
 s"mongodb://$username:$password@localhost/?authMechanism=SCRAM-SHA-1"
)
uri: MongoClientURI = mongodb://USER:PASSWORD@localhost/?authMechanism=SCRAM-SHA-1

scala> val mongoClient = MongoClient(uri)
client: com.mongodb.casbah.MongoClient = com.mongodb.casbah.MongoClient@4ac17318

```

通常，您不希望在代码中以明文形式放置密码。您可以在命令行中提示输入密码或通过环境变量传递，就像我们在第七章中处理 GitHub OAuth 令牌那样。以下代码片段演示了如何通过环境传递凭据：

```py
// Credentials.scala

import com.mongodb.casbah.Imports._

object Credentials extends App {

  val username = sys.env.getOrElse("MONGOUSER",
    throw new IllegalStateException(
      "Need a MONGOUSER variable in the environment")
  )
  val password = sys.env.getOrElse("MONGOPASSWORD",
    throw new IllegalStateException(
      "Need a MONGOPASSWORD variable in the environment")
  )

  val host = "127.0.0.1"
  val port = 27017

  val uri = s"mongodb://$username:$password@$host:$port/?authMechanism=SCRAM-SHA-1"

  val client = MongoClient(MongoClientURI(uri))
}
```

您可以通过以下方式在 SBT 中运行它：

```py
$ MONGOUSER="pascal" MONGOPASSWORD="scalarulez" sbt
> runMain Credentials

```

# 插入文档

让我们在新创建的数据库中插入一些文档。我们希望存储有关 GitHub 用户的信息，使用以下文档结构：

```py
{
    id: <mongodb object id>,
    login: "pbugnion",
    github_id: 1392879,
    repos: [ 
        {
            name: "scikit-monaco",
            id: 14821551,
            language: "Python"
        },
        {
            name: "contactpp",
            id: 20448325,
            language: "Python"
        }
    ]
}
```

Casbah 提供了一个`DBObject`类来表示 Scala 中的 MongoDB 文档（和子文档）。让我们首先为每个存储库子文档创建一个`DBObject`实例：

```py
scala> val repo1 = DBObject("name" -> "scikit-monaco", "id" -> 14821551, "language" -> "Python")
repo1: DBObject = { "name" : "scikit-monaco" , "id" : 14821551, "language" : "Python"}

```

如您所见，`DBObject` 只是一个键值对列表，其中键是字符串。值具有编译时类型 `AnyRef`，但如果您尝试添加无法序列化的值，Casbah 将在运行时失败。

我们还可以直接从键值对列表创建 `DBObject` 实例。这在将 Scala 映射转换为 `DBObject` 时特别有用：

```py
scala> val fields:Map[String, Any] = Map(
 "name" -> "contactpp",
 "id" -> 20448325,
 "language" -> "Python"
)
Map[String, Any] = Map(name -> contactpp, id -> 20448325, language -> Python)

scala> val repo2 = DBObject(fields.toList)
repo2: dDBObject = { "name" : "contactpp" , "id" : 20448325, "language" : "Python"}

```

`DBObject` 类提供了与映射相同的大多数方法。例如，我们可以访问单个字段：

```py
scala> repo1("name")
AnyRef = scikit-monaco

```

我们可以通过向现有对象添加字段来构造一个新对象：

```py
scala> repo1 + ("fork" -> true)
mutable.Map[String,Any] = { "name" : "scikit-monaco" , "id" : 14821551, "language" : "python", "fork" : true}

```

注意返回类型：`mutable.Map[String,Any]`。Casbah 不是直接实现如 `+` 之类的方法，而是通过提供到和从 `mutable.Map` 的隐式转换将它们添加到 `DBObject` 中。

新的 `DBObject` 实例也可以通过连接两个现有实例来创建：

```py
scala> repo1 ++ DBObject(
 "locs" -> 6342, 
 "description" -> "Python library for Monte Carlo integration"
)
DBObject = { "name" : "scikit-monaco" , "id" : 14821551, "language" : "Python", "locs" : 6342 , "description" : "Python library for Monte Carlo integration"}

```

`DBObject` 实例可以使用 `+=` 操作符插入到集合中。让我们将我们的第一个文档插入到 `user` 集合中：

```py
scala> val userDocument = DBObject(
 "login" -> "pbugnion", 
 "github_id" -> 1392879, 
 "repos" -> List(repo1, repo2)
)
userDocument: DBObject = { "login" : "pbugnion" , ... }

scala> val coll = MongoClient()("github")("users")
coll: com.mongodb.casbah.MongoCollection = users

scala> coll += userDocument
com.mongodb.casbah.TypeImports.WriteResult = WriteResult{, n=0, updateOfExisting=false, upsertedId=null}

```

包含单个文档的数据库有点无聊，所以让我们添加一些直接从 GitHub API 查询的更多文档。您在上一章中学习了如何查询 GitHub API，所以我们不会在这里详细说明如何进行此操作。

在本章的代码示例中，我们提供了一个名为 `GitHubUserIterator` 的类，该类查询 GitHub API（特别是 `/users` 端点）以获取用户文档，将它们转换为案例类，并将它们作为迭代器提供。您可以在本章的代码示例（可在 GitHub 上找到 [`github.com/pbugnion/s4ds/tree/master/chap08`](https://github.com/pbugnion/s4ds/tree/master/chap08)）中的 `GitHubUserIterator.scala` 文件中找到该类。访问该类最简单的方法是在本章代码示例的目录中打开一个 SBT 控制台。API 随后按登录 ID 递增的顺序获取用户：

```py
scala> val it = new GitHubUserIterator
it: GitHubUserIterator = non-empty iterator

scala> it.next // Fetch the first user
User = User(mojombo,1,List(Repo(...

```

`GitHubUserIterator` 返回 `User` 案例类的实例，该类定义如下：

```py
// User.scala
case class User(login:String, id:Long, repos:List[Repo])

// Repo.scala
case class Repo(name:String, id:Long, language:String)
```

让我们编写一个简短的程序来获取 500 个用户并将它们插入到 MongoDB 数据库中。我们需要通过 GitHub API 进行身份验证来检索这些用户。`GitHubUserIterator` 构造函数接受 GitHub OAuth 令牌作为可选参数。我们将通过环境注入令牌，就像我们在上一章中所做的那样。

在分解代码之前，我们首先给出整个代码列表——如果您正在手动输入，您需要将 `GitHubUserIterator.scala` 从本章的代码示例复制到您运行此操作的目录中，以便访问 `GitHubUserIterator` 类。该类依赖于 `scalaj-http` 和 `json4s`，因此您可以选择复制代码示例中的 `build.sbt` 文件，或者在您的 `build.sbt` 文件中指定这些包作为依赖项。

```py
// InsertUsers.scala

import com.mongodb.casbah.Imports._

object InsertUsers {

  /** Function for reading GitHub token from environment. */
  lazy val token:Option[String] = sys.env.get("GHTOKEN") orElse {
    println("No token found: continuing without authentication")
    None
  }

  /** Transform a Repo instance to a DBObject */
  def repoToDBObject(repo:Repo):DBObject = DBObject(
    "github_id" -> repo.id,
    "name" -> repo.name,
    "language" -> repo.language
  )

  /** Transform a User instance to a DBObject */
  def userToDBObject(user:User):DBObject = DBObject(
    "github_id" -> user.id,
    "login" -> user.login,
    "repos" -> user.repos.map(repoToDBObject)
  )

  /** Insert a list of users into a collection. */
  def insertUsers(coll:MongoCollection)(users:Iterable[User]) {
    users.foreach { user => coll += userToDBObject(user) }
  }

  /**  Fetch users from GitHub and passes them to `inserter` */
  def ingestUsers(nusers:Int)(inserter:Iterable[User] => Unit) {
    val it = new GitHubUserIterator(token)
    val users = it.take(nusers).toList
    inserter(users)
  }

  def main(args:Array[String]) {
    val coll = MongoClient()("github")("users")
    val nusers = 500
    coll.dropCollection()
    val inserter = insertUsers(coll)_
    ingestUsers(inserter)(nusers)
  }

}
```

在深入了解程序的工作原理之前，让我们通过 SBT 运行它。您将想要使用身份验证查询 API 以避免达到速率限制。回想一下，我们需要设置 `GHTOKEN` 环境变量：

```py
$ GHTOKEN="e83638..." sbt
$ runMain InsertUsers

```

程序运行大约需要五分钟（取决于您的互联网连接）。为了验证程序是否工作，我们可以查询 `github` 数据库中 `users` 集合中的文档数量：

```py
$ mongo github --quiet --eval "db.users.count()"
500

```

让我们分解一下代码。我们首先加载 OAuth 令牌以验证 GitHub API。令牌存储为环境变量 `GHTOKEN`。`token` 变量是一个 `lazy val`，因此令牌只在形成对 API 的第一个请求时加载。我们已经在 第七章，*Web APIs* 中使用了这种模式。

然后我们定义两个方法，将领域模型中的类转换成 `DBObject` 实例：

```py
def repoToDBObject(repo:Repo):DBObject = ...
def userToDBObject(user:User):DBObject = ...
```

带着这两个方法，我们可以轻松地将用户添加到我们的 MongoDB 集合中：

```py
def insertUsers(coll:MongoCollection)(users:Iterable[User]) {
  users.foreach { user => coll += userToDBObject(user) }
}
```

我们使用柯里化来拆分 `insertUsers` 的参数。这使得我们可以将 `insertUsers` 作为函数工厂使用：

```py
val inserter = insertUsers(coll)_
```

这创建了一个新的方法 `inserter`，其签名是 `Iterable[User] => Unit`，用于将用户插入到 `coll` 中。为了了解这如何有用，让我们编写一个函数来包装整个数据导入过程。这是这个函数的第一个尝试可能看起来像这样：

```py
def ingestUsers(nusers:Int)(inserter:Iterable[User] => Unit) {
  val it = new GitHubUserIterator(token)
  val users = it.take(nusers).toList
  inserter(users)
}
```

注意 `ingestUsers` 方法如何将其第二个参数作为一个指定用户列表如何插入到数据库中的方法。这个函数封装了插入到 MongoDB 集合的整个特定代码。如果我们决定，在未来的某个时间点，我们讨厌 MongoDB 并且必须将文档插入到 SQL 数据库或写入平面文件，我们只需要将不同的 `inserter` 函数传递给 `ingestUsers`。其余的代码保持不变。这展示了使用高阶函数带来的更高灵活性：我们可以轻松构建一个框架，并让客户端代码插入它需要的组件。

如前所述定义的 `ingestUsers` 方法有一个问题：如果 `nusers` 值很大，它将在构建整个用户列表时消耗大量内存。更好的解决方案是将它分解成批次：我们从 API 获取一批用户，将它们插入到数据库中，然后继续处理下一批。这样我们可以通过改变批次大小来控制内存使用。它也更加容错：如果程序崩溃，我们只需从最后一个成功插入的批次重新启动。

`.grouped` 方法，适用于所有可迭代对象，用于批量处理。它返回一个遍历原始可迭代对象片段的迭代器：

```py
scala> val it = (0 to 10)
it: Range.Inclusive = Range(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

scala> it.grouped(3).foreach { println } // In batches of 3
Vector(0, 1, 2)
Vector(3, 4, 5)
Vector(6, 7, 8)
Vector(9, 10)

```

让我们重写我们的 `ingestUsers` 方法以使用批次。我们还会在每个批次后添加一个进度报告，以便给用户一些反馈：

```py
/**  Fetch users from GitHub and pass them to `inserter` */
def ingestUsers(nusers:Int)(inserter:Iterable[User] => Unit) {
  val batchSize = 100
  val it = new GitHubUserIterator(token)
  print("Inserted #users: ")
  it.take(nusers).grouped(batchSize).zipWithIndex.foreach {
    case (users, batchNumber) =>
      print(s"${batchNumber*batchSize} ")
      inserter(users)
  }
  println()
}
```

让我们更仔细地看看高亮行。我们从用户迭代器 `it` 开始。然后我们取前 `nusers` 个用户。这返回一个 `Iterator[User]`，它不会在 GitHub 数据库中的每个用户上愉快地运行，而是在 `nusers` 后终止。然后我们将这个迭代器分组为 100 个用户的批次。`.grouped` 方法返回 `Iterator[Iterator[User]]`。然后我们将每个批次与其索引进行连接，这样我们就可以知道我们目前正在处理哪个批次（我们在 `print` 语句中使用这个）。`.zipWithIndex` 方法返回 `Iterator[(Iterator[User], Int)]`。我们在循环中使用一个 case 语句来解包这个元组，将 `users` 绑定到 `Iterator[User]`，将 `batchNumber` 绑定到索引。让我们通过 SBT 运行这个例子：

```py
$ GHTOKEN="2502761..." sbt 
> runMain InsertUsers
[info] Running InsertUsers
Inserted #users: 0 100 200 300 400
[success] Total time: 215 s, completed 01-Nov-2015 18:44:30

```

# 从数据库中提取对象

现在我们有一个包含一些用户的数据库。让我们从 REPL 中查询这个数据库：

```py
scala> import com.mongodb.casbah.Imports._
import com.mongodb.casbah.Imports._

scala> val collection = MongoClient()("github")("users")
MongoCollection = users

scala> val maybeUser = collection.findOne
Option[collection.T] = Some({ "_id" : { "$oid" : "562e922546f953739c43df02"} , "github_id" : 1 , "login" : "mojombo" , "repos" : ...

```

`findOne` 方法返回一个包含在选项中的单个 `DBObject` 对象，除非集合为空，在这种情况下它返回 `None`。因此，我们必须使用 `get` 方法来提取对象：

```py
scala> val user = maybeUser.get
collection.T = { "_id" : { "$oid" : "562e922546f953739c43df02"} , "github_id" : 1 , "login" : "mojombo" , "repos" : ...

```

如您在本章早期所学的，`DBObject` 是一个类似于映射的对象，其键的类型为 `String`，值类型为 `AnyRef`：

```py
scala> user("login")
AnyRef = mojombo

```

通常，我们在从数据库导入对象时，希望尽可能早地恢复编译时类型信息：我们不希望在可以更具体的情况下传递 `AnyRef`。我们可以使用 `getAs` 方法提取字段并将其转换为特定类型：

```py
scala> user.getAsString
Option[String] = Some(mojombo)

```

如果文档中缺少字段或值无法转换，`getAs` 将返回 `None`：

```py
scala> user.getAsInt
Option[Int] = None

```

聪明的读者可能会注意到，`getAs[T]` 提供的接口与我们定义在 第五章 中的 JDBC 结果集上的 `read[T]` 方法类似，*通过 JDBC 的 Scala 和 SQL*。

如果 `getAs` 失败（例如，因为字段缺失），我们可以使用 `orElse` 部分函数来恢复：

```py
scala> val loginName = user.getAsString orElse { 
 println("No login field found. Falling back to 'name'")
 user.getAsString
}
loginName: Option[String] = Some(mojombo)

```

`getAsOrElse` 方法允许我们在转换失败时替换默认值：

```py
scala> user.getAsOrElseInt
Int = 1392879

```

注意，我们还可以使用 `getAsOrElse` 抛出异常：

```py
scala> user.getAsOrElseString
)
java.lang.IllegalArgumentException: Missing value for name
...

```

文档中嵌入的数组可以转换为 `List[T]` 对象，其中 `T` 是数组中元素的类型：

```py
scala> user.getAsOrElse[List[DBObject]]("repos",
 List.empty[DBObject])
List[DBObject] = List({ "github_id" : 26899533 , "name" : "30daysoflaptops.github.io" ...

```

一次检索一个文档并不很有用。要检索集合中的所有文档，请使用 `.find` 方法：

```py
scala> val userIterator = collection.find()
userIterator: collection.CursorType = non-empty iterator

```

这将返回一个 `DBObject` 迭代器。要实际从数据库中获取文档，您需要通过将其转换为集合来具体化迭代器，例如使用 `.toList`：

```py
scala> val userList = userIterator.toList
List[DBObject] = List({ "_id" : { "$oid": ...

```

让我们把所有这些放在一起。我们将编写一个玩具程序，该程序打印我们集合中每个用户的平均存储库数量。代码通过获取集合中的每个文档，从每个文档中提取存储库数量，然后对这些数量进行平均来实现：

```py
// RepoNumber.scala

import com.mongodb.casbah.Imports._

object RepoNumber {

  /** Extract the number of repos from a DBObject
    * representing a user.
    */   
  def extractNumber(obj:DBObject):Option[Int] = {
    val repos = obj.getAs[List[DBObject]]("repos") orElse {
      println("Could not find or parse 'repos' field")
      None
    }
    repos.map { _.size }
  }

  val collection = MongoClient()("github")("users")

  def main(args:Array[String]) {    
    val userIterator = collection.find()

    // Convert from documents to Option[Int]
    val repoNumbers = userIterator.map { extractNumber }

    // Convert from Option[Int] to Int
    val wellFormattedNumbers = repoNumbers.collect { 
      case Some(v) => v 
    }.toList

    // Calculate summary statistics
    val sum = wellFormattedNumbers.reduce { _ + _ }
    val count = wellFormattedNumbers.size

    if (count == 0) {
      println("No repos found")
    }
    else {
      val mean = sum.toDouble / count.toDouble
      println(s"Total number of users with repos: $count")
      println(s"Total number of repos: $sum")
      println(s"Mean number of repos: $mean")
    }
  }
}
```

让我们通过 SBT 运行这个例子：

```py
> runMain RepoNumber
Total number of users with repos: 500
Total number of repos: 9649
Mean number of repos: 19.298

```

代码从 `extractNumber` 函数开始，该函数从每个 `DBObject` 中提取存储库数量。如果文档不包含 `repos` 字段，则返回值是 `None`。

代码的主体部分首先创建一个遍历集合中 `DBObject` 的迭代器。然后，这个迭代器通过 `extractNumber` 函数进行映射，将其转换为 `Option[Int]` 的迭代器。然后我们对这个迭代器运行 `.collect`，收集所有不是 `None` 的值，在这个过程中将 `Option[Int]` 转换为 `Int`。然后我们才使用 `.toList` 将迭代器实体化为列表。得到的列表 `wellFormattedNumbers` 具有类型 `List[Int]`。然后我们只取这个列表的平均值并将其打印到屏幕上。

注意，除了 `extractNumber` 函数外，这个程序没有处理任何与 Casbah 特定的类型相关的事务：`.find()` 返回的迭代器只是一个 Scala 迭代器。这使得 Casbah 的使用变得简单：你需要熟悉的唯一数据类型是 `DBObject`（与 JDBC 的 `ResultSet` 进行比较，我们必须显式地将其包装在流中，例如）。

# 复杂查询

我们现在知道如何将 `DBObject` 实例转换为自定义 Scala 类。在本节中，你将学习如何构建只返回集合中部分文档的查询。

在上一节中，你学习了如何如下检索集合中的所有文档：

```py
scala> val objs = collection.find().toList
List[DBObject] = List({ "_id" : { "$oid" : "56365cec46f9534fae8ffd7f"} ,...

```

`collection.find()` 方法返回一个遍历集合中所有文档的迭代器。通过在这个迭代器上调用 `.toList`，我们将其实体化为列表。

我们可以通过传递一个查询文档到 `.find` 方法来自定义返回哪些文档。例如，我们可以检索特定登录名的文档：

```py
scala> val query = DBObject("login" -> "mojombo")
query: DBObject = { "login" : "mojombo"}

scala> val objs = collection.find(query).toList
List[DBObject] = List({ "_id" : { "$oid" : "562e922546f953739c43df02"} , "login" : "mojombo",...

```

MongoDB 查询以 `DBObject` 实例的形式表达。`DBObject` 中的键对应于集合文档中的字段，而值是控制该字段允许值的表达式。因此，`DBObject("login" -> "mojombo")` 将选择所有 `login` 字段为 `mojombo` 的文档。使用 `DBObject` 实例表示查询可能看起来有些晦涩，但如果你阅读 MongoDB 文档([`docs.mongodb.org/manual/core/crud-introduction/`](https://docs.mongodb.org/manual/core/crud-introduction/))，你会很快明白：查询在 MongoDB 中本身就是 JSON 对象。因此，Casbah 中的查询表示为 `DBObject` 与其他 MongoDB 客户端实现保持一致。它还允许熟悉 MongoDB 的人迅速开始编写 Casbah 查询。

MongoDB 支持更复杂的查询。例如，要查询 `"github_id"` 在 `20` 到 `30` 之间的所有人，我们可以编写以下查询：

```py
scala> val query = DBObject("github_id" -> 
 DBObject("$gte" -> 20, "$lt" -> 30))
query: DBObject = { "github_id" : { "$gte" : 20 , "$lt" : 30}}

scala> collection.find(query).toList
List[com.mongodb.casbah.Imports.DBObject] = List({ "_id" : { "$oid" : "562e922546f953739c43df0f"} , "github_id" : 23 , "login" : "takeo" , ...

```

我们使用`DBObject("$gte" -> 20, "$lt" -> 30)`限制了`github_id`可以取的值的范围。`"$gte"`字符串表示`github_id`必须大于或等于`20`。同样，`"$lt"`表示小于操作符。要获取查询时可以使用的所有操作符的完整列表，请查阅 MongoDB 参考文档([`docs.mongodb.org/manual/reference/operator/query/`](http://docs.mongodb.org/manual/reference/operator/query/))。

到目前为止，我们只看了顶级字段的查询。Casbah 还允许我们使用点符号查询子文档和数组中的字段。在数组值的上下文中，这将返回数组中至少有一个值与查询匹配的所有文档。例如，要检索所有在 Scala 中拥有主要语言为 Scala 的仓库的用户：

```py
scala> val query = DBObject("repos.language" -> "Scala")
query: DBObject = { "repos.language" : "Scala"}

scala> collection.find(query).toList
List[DBObject] = List({ "_id" : { "$oid" : "5635da4446f953234ca634df"}, "login" : "kevinclark"...

```

# Casbah 查询 DSL

使用`DBObject`实例来表示查询可能非常冗长且难以阅读。Casbah 提供了一个 DSL 来更简洁地表示查询。例如，要获取所有`github_id`字段在`20`和`30`之间的文档，我们会编写以下代码：

```py
scala> collection.find("github_id" $gte 20 $lt 30).toList
List[com.mongodb.casbah.Imports.DBObject] = List({ "_id" : { "$oid" : "562e922546f953739c43df0f"} , "github_id" : 23 , "login" : "takeo" , "repos" : ...

```

DSL 提供的运算符将自动构造`DBObject`实例。尽可能多地使用 DSL 运算符通常会导致代码更易读、更易于维护。

进入查询 DSL 的详细内容超出了本章的范围。您会发现使用它相当简单。要获取 DSL 支持的运算符的完整列表，请参阅 Casbah 文档[`mongodb.github.io/casbah/3.0/reference/query_dsl/`](http://mongodb.github.io/casbah/3.0/reference/query_dsl/)。我们在此总结了最重要的运算符：

| 运算符 | 描述 |
| --- | --- |
| `"login" $eq "mojombo"` | 这将选择`login`字段正好是`mojombo`的文档 |
| `"login" $ne "mojombo"` | 这将选择`login`字段不是`mojombo`的文档 |
| `"github_id" $gt 1 $lt 20` | 这将选择`github_id`大于`1`且小于`20`的文档 |
| `"github_id" $gte 1 $lte 20` | 这将选择`github_id`大于或等于`1`且小于或等于`20`的文档 |
| `"login" $in ("mojombo", "defunkt")` | `login`字段是`mojombo`或`defunkt` |
| `"login" $nin ("mojombo", "defunkt")` | `login`字段不是`mojombo`或`defunkt` |
| `"login" $regex "^moj.*"` | `login`字段匹配特定的正则表达式 |
| `"login" $exists true` | `login`字段存在 |
| `$or("login" $eq "mojombo", "github_id" $gte 22)` | `login`字段是`mojombo`或`github_id`字段大于或等于`22` |
| `$and("login" $eq "mojombo", "github_id" $gte 22)` | `login`字段是`mojombo`且`github_id`字段大于或等于`22` |

我们还可以使用点符号来查询数组和子文档。例如，以下查询将计算所有在 Scala 中拥有仓库的用户：

```py
scala> collection.find("repos.language" $eq "Scala").size
Int = 30

```

# 自定义类型序列化

到目前为止，我们只尝试了序列化和反序列化简单类型。如果我们想将存储在仓库数组中的语言字段解码为枚举而不是字符串呢？例如，我们可以定义以下枚举：

```py
scala> object Language extends Enumeration {
 val Scala, Java, JavaScript = Value
}
defined object Language

```

Casbah 允许我们定义与特定 Scala 类型相关的自定义序列化器：我们可以通知 Casbah，每当它在 `DBObject` 中遇到 `Language.Value` 类型的实例时，该实例应通过一个自定义转换器进行转换，例如转换为字符串，然后再将其写入数据库。

要定义一个自定义序列化器，我们需要定义一个扩展 `Transformer` 特质的类。这个特质暴露了一个方法，`transform(o:AnyRef):AnyRef`。让我们定义一个 `LanguageTransformer` 特质，它将 `Language.Value` 转换为 `String`：

```py
scala> import org.bson.{BSON, Transformer}
import org.bson.{BSON, Transformer}

scala> trait LanguageTransformer extends Transformer {
 def transform(o:AnyRef):AnyRef = o match {
 case l:Language.Value => l.toString
 case _ => o
 }
}
defined trait LanguageTransformer

```

我们现在需要注册特质，以便在需要解码类型 `Language.Value` 的实例时使用。我们可以使用 `addEncodingHook` 方法来完成此操作：

```py
scala> BSON.addEncodingHook(
 classOf[Language.Value], new LanguageTransformer {})

```

我们现在可以构建包含 `Language` 枚举值的 `DBObject` 实例：

```py
scala> val repoObj = DBObject(
 "github_id" -> 1234L,
 "language" -> Language.Scala
)
repoObj: DBObject = { "github_id" : 1234 , "language" : "Scala"}

```

反过来呢？我们如何告诉 Casbah 将 `"language"` 字段读取为 `Language.Value`？这不可能通过自定义反序列化器实现：`"Scala"` 现在作为字符串存储在数据库中。因此，在反序列化时，`"Scala"` 与 `"mojombo"` 没有区别。因此，当 `"Scala"` 被序列化时，我们失去了类型信息。

因此，虽然自定义编码钩子在序列化时很有用，但在反序列化时则不那么有用。一种更干净、更一致的替代方案是使用 *类型类* 来自定义序列化和反序列化。我们已经在 第五章，*通过 JDBC 的 Scala 和 SQL* 的上下文中广泛介绍了如何使用这些类型类，用于将数据序列化和反序列化到 SQL。这里的程序将非常相似：

1.  定义一个具有 `read(v:Any)`:`T` 方法的 `MongoReader[T]` 类型类。

1.  在 `MongoReader` 伴生对象中为所有感兴趣的类型定义具体的 `MongoReader` 实现，例如 `String`、`Language.Value`。

1.  使用 *pimp my library* 模式，为 `DBObject` 增强一个 `read[T:MongoReader]` 方法。

例如，`Language.Value` 的 `MongoReader` 实现如下：

```py
implicit object LanguageReader extends MongoReader[Language.Value] {
  def read(v:Any):Language.Value = v match {
    case s:String => Language.withName(s)
  }
}
```

我们可以用同样的方式使用 `MongoWriter` 类型类。使用类型类是自定义序列化和反序列化的惯用和可扩展的方法。

我们在本章相关的代码示例（在 `typeclass` 目录中）提供了一个类型类的完整示例。

# 除此之外

本章我们只考虑了 Casbah。然而，MongoDB 还有其他驱动程序。

*ReactiveMongo* 是一个专注于数据库异步读写操作的驱动程序。所有查询都返回一个未来对象，强制执行异步行为。这非常适合数据流或 Web 应用程序。

*Salat*位于 Casbah 之上，旨在提供对 case 类进行简单序列化和反序列化的功能。

完整的驱动程序列表可在[`docs.mongodb.org/ecosystem/drivers/scala/`](https://docs.mongodb.org/ecosystem/drivers/scala/)找到。

# 摘要

在本章中，你学习了如何与 MongoDB 数据库交互。通过将上一章学到的结构（从 Web API 中提取信息）与本章学到的结构结合起来，我们现在可以构建一个用于数据摄取的并发、响应式程序。

在下一章中，你将学习如何使用 Akka 演员构建更灵活的分布式、并发结构。

# 参考文献

《*MongoDB: The Definitive Guide*》，由*Kristina Chodorow*所著，是 MongoDB 的良好入门指南。它完全不涉及使用 Scala 与 MongoDB 交互，但对于熟悉 MongoDB 的人来说，Casbah 足够直观。

类似地，MongoDB 文档([`docs.mongodb.org/manual/`](https://docs.mongodb.org/manual/))提供了对 MongoDB 的深入讨论。

Casbah 本身有很好的文档([`mongodb.github.io/casbah/3.0/`](http://mongodb.github.io/casbah/3.0/))。有一个*入门指南*，与本章有些类似，还有一个完整的参考指南，将填补本章留下的空白。

这个片段，[`gist.github.com/switzer/4218526`](https://gist.github.com/switzer/4218526)，实现了类型类，用于将领域模型中的对象序列化和反序列化为`DBObject`。前提与本章中建议的类型类用法略有不同：我们将 Scala 类型转换为`AnyRef`，以便在`DBObject`中使用作为值。然而，这两种方法互为补充：可以想象有一组类型类将`User`或`Repo`转换为`DBObject`，另一组将`Language.Value`转换为`AnyRef`。

# 第九章。使用 Akka 进行并发

本书的大部分内容都专注于利用多核和分布式架构。在第四章，*并行集合和未来*中，你学习了如何使用并行集合将批处理问题分布到多个线程上，以及如何使用未来执行异步计算。在第七章，*Web API*中，我们将这些知识应用于使用多个并发线程查询 GitHub API。

并发抽象，如未来和并行集合，通过限制你可以做的事情来简化并发编程的巨大复杂性。例如，并行集合强制你将并行化问题表述为集合上的纯函数序列。

演员提供了一种不同的并发思考方式。演员在封装 *状态* 方面非常出色。管理不同执行线程之间共享的状态可能是开发并发应用程序最具挑战性的部分，正如我们将在本章中发现的那样，演员使其变得可管理。

# GitHub 关注者图

在前两章中，我们探讨了 GitHub API，学习了如何使用 *json-4s* 查询 API 并解析结果。

让我们想象一下，我们想要提取 GitHub 关注者图：我们想要一个程序，它将从特定用户开始，提取该用户关注者，然后提取他们的关注者，直到我们告诉它停止。问题是，我们事先不知道需要获取哪些 URL：当我们下载特定用户关注者的登录名时，我们需要验证我们是否已经获取了这些用户。如果没有，我们将它们添加到需要获取其关注者的用户队列中。算法爱好者可能会认出这作为 *广度优先搜索*。

让我们概述一下我们如何以单线程方式编写它。核心组件是一组已访问用户和未来用户队列：

```py
val seedUser = "odersky" // the origin of the network

// Users whose URLs need to be fetched 
val queue = mutable.Queue(seedUser) 

// set of users that we have already fetched 
// (to avoid re-fetching them)
val fetchedUsers = mutable.Set.empty[String] 

while (queue.nonEmpty) {
  val user = queue.dequeue
  if (!fetchedUsers(user)) {
    val followers = fetchFollowersForUser(user)
    followers foreach { follower =>           
      // add the follower to queue of people whose 
      // followers we want to find.
      queue += follower
    }
    fetchedUsers += user
  }
}
```

在这里，`fetchFollowersForUser` 方法的签名是 `String => Iterable[String]`，它负责接受一个登录名，将其转换为 GitHub API 中的 URL，查询 API，并从响应中提取关注者列表。我们在这里不会实现它，但您可以在本书代码示例的 `chap09/single_threaded` 目录中找到一个完整的示例（[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)）。如果您已经阅读了 第七章

假设 Chris 接到了一个订单。他会查看订单，决定是否可以自己处理，如果不能，他会将订单转发给 Mark 或 Sally。让我们假设订单要求一个小程序，所以 Bob 将订单转发给了 Sally。Sally 非常忙，正在处理一批积压的订单，因此她不能立即处理订单信息，它将在她的邮箱中短暂停留。当她最终开始处理订单时，她可能会决定将订单分成几个部分，其中一些部分她会分配给 Kevin 和 Bob。

当 Bob 和 Kevin 完成任务时，他们会向 Sally 发送消息以通知她。当订单的每个部分都得到满足时，Sally 会将这些部分汇总起来，直接向客户或向 Chris 发送结果的消息。

跟踪哪些工作必须完成以完成订单的任务落在 Sally 身上。当她收到 Bob 和 Kevin 的消息时，她必须更新她正在进行的任务列表，并检查与这个订单相关的每个任务是否完成。这种协调在传统的*synchronize*块中会更具有挑战性：对正在进行的任务列表和已完成任务列表的每次访问都需要同步。通过将这种逻辑嵌入只能一次处理一个消息的 Sally 中，我们可以确保不会出现竞态条件。

我们的初创公司运作良好，因为每个人只负责做一件事：Chris 要么委托给 Mark 或 Sally，Sally 将订单拆分成几个部分并分配给 Bob 和 Kevin，而 Bob 和 Kevin 完成每个部分。你可能会想，“等等，所有的逻辑都嵌入在 Bob 和 Kevin 中，他们是底层的员工，做所有实际的工作”。与员工不同，演员成本低，所以如果嵌入在演员中的逻辑变得过于复杂，很容易引入额外的委托层，直到任务足够简单。

我们初创公司的员工拒绝进行多任务处理。当他们得到一份工作时，他们会完全处理它，然后转到下一个任务。这意味着他们不会因为多任务处理的复杂性而变得混乱。通过一次处理一个消息，演员大大减少了引入并发错误（如竞态条件）的范围。

更重要的是，通过提供一个程序员可以直观理解的抽象——即人类工作者——Akka 使得关于并发的推理变得更加容易。

# 使用 Akka 的 Hello World

让我们安装 Akka。我们将它添加到我们的 `build.sbt` 文件中：

```py
scalaVersion := "2.11.7"

libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.4.0"
```

我们现在可以按照以下方式导入 Akka：

```py
import akka.actor._
```

对于我们第一次进入演员的世界，我们将构建一个接收并回显每个接收到的消息的演员。本节中的代码示例位于本书提供的示例代码目录 `chap09/hello_akka` 中（[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)）：

```py
// EchoActor.scala
import akka.actor._

class EchoActor extends Actor with ActorLogging {
  def receive = {
    case msg:String => 
      Thread.sleep(500)
      log.info(s"Received '$msg'") 
  }
}
```

让我们分析这个例子，从构造函数开始。我们的演员类必须扩展 `Actor`。我们还添加了 `ActorLogging`，这是一个实用特性，它添加了 `log` 属性。

`Echo` 演员公开一个单一的方法，`receive`。这是演员与外部世界通信的唯一方式。为了有用，所有演员都必须公开一个 `receive` 方法。`receive` 方法是一个部分函数，通常使用多个 `case` 语句实现。当演员开始处理消息时，它将匹配每个 `case` 语句，直到找到匹配的一个。然后执行相应的代码块。

我们的 echo 演员接受一种类型的消息，一个普通的字符串。当这个消息被处理时，演员会等待半秒钟，然后将消息回显到日志文件中。

让我们实例化几个 Echo 演员，并发送它们消息：

```py
// HelloAkka.scala

import akka.actor._

object HelloAkka extends App {

  // We need an actor system before we can 
  // instantiate actors
  val system = ActorSystem("HelloActors")

  // instantiate our two actors
  val echo1 = system.actorOf(Props[EchoActor], name="echo1")
  val echo2 = system.actorOf(Props[EchoActor], name="echo2")

  // Send them messages. We do this using the "!" operator
  echo1 ! "hello echo1"
  echo2 ! "hello echo2"
  echo1 ! "bye bye"

  // Give the actors time to process their messages, 
  // then shut the system down to terminate the program
  Thread.sleep(500)
  system.shutdown
}
```

运行此命令会得到以下输出：

```py
[INFO] [07/19/2015 17:15:23.954] [HelloActor-akka.actor.default-dispatcher-2] [akka://HelloActor/user/echo1] Received 'hello echo1'
[INFO] [07/19/2015 17:15:23.954] [HelloActor-akka.actor.default-dispatcher-3] [akka://HelloActor/user/echo2] Received 'hello echo2'
[INFO] [07/19/2015 17:15:24.955] [HelloActor-akka.actor.default-dispatcher-2] [akka://HelloActor/user/echo1] Received 'bye bye'

```

注意，`echo1` 和 `echo2` 演员显然是并发执行的：`hello echo1` 和 `hello echo2` 同时被记录。传递给 `echo1` 的第二个消息在演员完成处理 `hello echo1` 后才被处理。

有几点需要注意：

+   要开始实例化演员，我们首先必须创建一个演员系统。通常每个应用程序只有一个演员系统。

+   我们实例化演员的方式看起来有点奇怪。我们不是调用构造函数，而是创建一个演员属性对象，`Props[T]`。然后我们要求演员系统使用这些属性创建一个演员。实际上，我们从不使用 `new` 实例化演员：它们要么是通过调用演员系统中的 `actorOf` 方法或另一个演员内的类似方法（稍后详细介绍）创建的。

我们从不从外部调用演员的方法。与演员交互的唯一方式是向其发送消息。我们使用 *tell* 操作符，`!` 来这样做。因此，从外部无法干扰演员的内部结构（或者至少，Akka 使得干扰演员的内部结构变得困难）。

# 作为消息的案例类

在我们的 "hello world" 示例中，我们构建了一个预期接收字符串消息的演员。任何不可变的对象都可以作为消息传递。使用案例类来表示消息非常常见。这比使用字符串更好，因为增加了额外的类型安全性：编译器会在案例类中捕获错误，而不会在字符串中。

让我们重写我们的 `EchoActor` 以接受案例类的实例作为消息。我们将使其接受两种不同的消息：`EchoMessage(message)` 和 `EchoHello`，后者只是回显默认消息。本节和下一节的示例位于本书提供的示例代码中的 `chap09/hello_akka_case_classes` 目录（[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)）。

定义演员可以接收的消息是 Akka 的一个常见模式：

```py
// EchoActor.scala

object EchoActor { 
  case object EchoHello
  case class EchoMessage(msg:String)
}
```

让我们更改演员定义以接受这些消息：

```py
class EchoActor extends Actor with ActorLogging {
  import EchoActor._ // import the message definitions
  def receive = {
    case EchoHello => log.info("hello")
    case EchoMessage(s) => log.info(s)  
  }
}
```

我们现在可以向我们的演员发送 `EchoHello` 和 `EchoMessage`：

```py
echo1 ! EchoActor.EchoHello
echo2 ! EchoActor.EchoMessage("We're learning Akka.")
```

# 演员构建

演员构建是 Akka 新手常见的难题来源。与（大多数）普通对象不同，你永远不会显式实例化演员。例如，你永远不会写 `val echo = new EchoActor`。实际上，如果你这样做，Akka 会抛出异常。

在 Akka 中创建演员是一个两步过程：首先创建一个 `Props` 对象，它封装了构建演员所需的属性。构建 `Props` 对象的方式取决于演员是否接受构造函数参数。如果构造函数不接受参数，我们只需将演员类作为类型参数传递给 `Props`：

```py
val echoProps = Props[EchoActor]
```

如果演员的构造函数接受参数，我们必须在定义 `Props` 对象时将这些参数作为额外的参数传递。让我们考虑以下演员，例如：

```py
class TestActor(a:String, b:Int) extends Actor { ... }
```

我们如下将构造函数参数传递给 `Props` 对象：

```py
val testProps = Props(classOf[TestActor], "hello", 2)
```

`Props` 实例只是封装了创建演员的配置。它实际上并没有创建任何东西。要创建演员，我们将 `Props` 实例传递给定义在 `ActorSystem` 实例上的 `system.actorOf` 方法：

```py
val system = ActorSystem("HelloActors")
val echo1 = system.actorOf(echoProps, name="hello-1")
```

`name` 参数是可选的，但用于日志和错误消息很有用。`.actorOf` 返回的值不是演员本身：它是对演员的 *引用*（可以将其视为演员居住的地址），具有 `ActorRef` 类型。`ActorRef` 是不可变的，但它可以被序列化和复制，而不会影响底层演员。

除了在演员系统中调用 `actorOf` 之外，还有另一种创建演员的方法：每个演员都公开了一个 `context.actorOf` 方法，该方法接受一个 `Props` 实例作为参数。上下文仅可以从演员内部访问：

```py
class TestParentActor extends Actor {
  val echoChild = context.actorOf(echoProps, name="hello-child")
  ...
}
```

从演员系统创建的演员与从另一个演员的上下文创建的演员之间的区别在于演员层次结构：每个演员都有一个父演员。在另一个演员的上下文中创建的任何演员都将具有该演员作为其父演员。由演员系统创建的演员有一个预定义的演员，称为 *用户守护者*，作为其父演员。当我们在本章末尾研究演员生命周期时，我们将了解演员层次结构的重要性。

一个非常常见的习惯用法是在演员的伴生对象中定义一个 `props` 方法，它作为该演员 `Props` 实例的工厂方法。让我们修改 `EchoActor` 伴生对象：

```py
object EchoActor {
  def props:Props = Props[EchoActor]

  // message case class definitions here
}
```

然后，我们可以按照以下方式实例化演员：

```py
val echoActor = system.actorOf(EchoActor.props)
```

# 演员的解剖结构

在深入一个完整的应用程序之前，让我们看看演员框架的不同组件以及它们是如何协同工作的：

+   **邮箱**: 邮箱基本上是一个队列。每个演员都有自己的邮箱。当你向一个演员发送消息时，消息会落在它的邮箱中，直到演员从队列中取出并通过其 `receive` 方法处理它。

+   **消息**: 消息使得演员之间的同步成为可能。消息可以具有任何类型，唯一的要求是它应该是不可变的。通常，最好使用案例类或案例对象来获得编译器的帮助，以检查消息类型。

+   **演员引用**: 当我们使用 `val echo1 = system.actorOf(Props[EchoActor])` 创建一个演员时，`echo1` 具有类型 `ActorRef`。`ActorRef` 是一个代理，用于表示演员，并且是其他世界与之交互的方式：当你发送一个消息时，你是将它发送到 `ActorRef`，而不是直接发送给演员。实际上，在 Akka 中，你永远无法直接获取到演员的句柄。演员可以使用 `.self` 方法为自己获取一个 `ActorRef`。

+   **演员上下文**: 每个演员都有一个 `context` 属性，通过它可以访问创建或访问其他演员的方法，以及获取有关外部世界的信息。我们已经看到了如何使用 `context.actorOf(props)` 创建新的演员。我们还可以通过 `context.parent` 获取演员的父引用。演员还可以使用 `context.stop(actorRef)` 停止另一个演员，其中 `actorRef` 是我们想要停止的演员的引用。

+   **调度器**: 调度器是实际执行演员中代码的机器。默认调度器使用 fork/join 线程池。Akka 允许我们为不同的演员使用不同的调度器。调整调度器可能有助于优化性能并给某些演员赋予优先权。演员运行的调度器可以通过 `context.dispatcher` 访问。调度器实现了 `ExecutionContext` 接口，因此它们可以用来运行未来。

# 追随者网络爬虫

本章的最终目标是构建一个爬虫来探索 GitHub 的粉丝图谱。我们已经在本章前面概述了如何以单线程方式完成这项工作。现在让我们设计一个演员系统来并发地完成这项任务。

代码中的动态部分是管理哪些用户已被获取或正在被获取的数据结构。这些需要封装在一个演员中，以避免多个演员尝试同时更改它们时产生的竞争条件。因此，我们将创建一个*fetcher 管理器*演员，其任务是跟踪哪些用户已被获取，以及我们接下来要获取哪些用户。

代码中可能成为瓶颈的部分是查询 GitHub API。因此，我们希望能够扩展同时执行此操作的工作者数量。我们将创建一个*fetcher*池，这些演员负责查询特定用户的 API 以获取粉丝。最后，我们将创建一个演员，其责任是解释 API 的响应。这个演员将把其对响应的解释转发给另一个演员，该演员将提取粉丝并将他们交给 fetcher 管理器。

这就是程序架构的样貌：

![粉丝网络爬虫](img/image01195.jpeg)

GitHub API 爬虫的演员系统

我们程序中的每个演员都执行单一的任务：fetcher 只查询 GitHub API，而队列管理器只将工作分配给 fetcher。Akka 的最佳实践是尽可能给演员分配狭窄的责任范围。这有助于在扩展时获得更好的粒度（例如，通过添加更多的 fetcher 演员，我们只是并行化瓶颈）和更好的弹性：如果一个演员失败，它只会影响其责任范围。我们将在本章后面探讨演员的失败。

我们将分几个步骤构建应用程序，在编写程序的同时探索 Akka 工具包。让我们从`build.sbt`文件开始。除了 Akka，我们还将`scalaj-http`和`json4s`标记为依赖项：

```py
// build.sbt
scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
  "org.json4s" %% "json4s-native" % "3.2.10",
  "org.scalaj" %% "scalaj-http" % "1.1.4",
  "com.typesafe.akka" %% "akka-actor" % "2.3.12"
)
```

# Fetcher 演员

我们应用程序的核心是 fetcher 演员，它负责从 GitHub 获取粉丝详情。最初，我们的演员将接受一个单一的消息，`Fetch(user)`。它将获取与`user`对应的粉丝，并将响应记录到屏幕上。我们将使用在第七章中开发的配方，即*Web APIs*，使用 OAuth 令牌查询 GitHub API。我们将通过演员构造函数注入令牌。

让我们从伴随对象开始。这将包含`Fetch(user)`消息的定义和两个工厂方法来创建`Props`实例。您可以在本书提供的示例代码中的`chap09/fetchers_alone`目录中找到本节的代码示例（[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)）：

```py
// Fetcher.scala
import akka.actor._
import scalaj.http._
import scala.concurrent.Future

object Fetcher {
  // message definitions
  case class Fetch(login:String)

  // Props factory definitions
  def props(token:Option[String]):Props = 
    Props(classOf[Fetcher], token)
  def props():Props = Props(classOf[Fetcher], None)
}
```

现在我们来定义 fetcher 本身。我们将把对 GitHub API 的调用封装在一个 future 中。这避免了单个缓慢的请求阻塞演员。当我们的演员收到一个 `Fetch` 请求时，它会将这个请求封装在一个 future 中，发送出去，然后可以处理下一个消息。让我们继续实现我们的演员：

```py
// Fetcher.scala
class Fetcher(val token:Option[String])
extends Actor with ActorLogging {
  import Fetcher._ // import message definition

  // We will need an execution context for the future.
  // Recall that the dispatcher doubles up as execution
  // context.
  import context.dispatcher

  def receive = {
    case Fetch(login) => fetchUrl(login)
  }

  private def fetchUrl(login:String) {
    val unauthorizedRequest = Http(
      s"https://api.github.com/users/$login/followers")
    val authorizedRequest = token.map { t =>
      unauthorizedRequest.header("Authorization", s"token $t")
    }

    // Prepare the request: try to use the authorized request
    // if a token was given, and fall back on an unauthorized 
    // request
    val request = authorizedRequest.getOrElse(unauthorizedRequest)

    // Fetch from github
    val response = Future { request.asString }
    response.onComplete { r =>
      log.info(s"Response from $login: $r")
    }
  }

}
```

让我们实例化一个演员系统并创建四个 fetcher 来检查我们的演员是否按预期工作。我们将从环境变量中读取 GitHub 令牌，如第七章 Web APIs 中所述，然后创建四个演员并要求每个演员获取特定 GitHub 用户的关注者。我们等待五秒钟以完成请求，然后关闭系统：

```py
// FetcherDemo.scala
import akka.actor._

object FetcherDemo extends App {
  import Fetcher._ // Import the messages

  val system = ActorSystem("fetchers")

  // Read the github token if present.
  val token = sys.env.get("GHTOKEN")

  val fetchers = (0 until 4).map { i =>
    system.actorOf(Fetcher.props(token))
  }

  fetchers(0) ! Fetch("odersky")
  fetchers(1) ! Fetch("derekwyatt")
  fetchers(2) ! Fetch("rkuhn")
  fetchers(3) ! Fetch("tototoshi")

  Thread.sleep(5000) // Wait for API calls to finish
  system.shutdown // Shut system down

}
```

让我们通过 SBT 运行代码：

```py
$ GHTOKEN="2502761..." sbt run
[INFO] [11/08/2015 16:28:06.500] [fetchers-akka.actor.default-dispatcher-2] [akka://fetchers/user/$d] Response from tototoshi: Success(HttpResponse([{"login":"akr4","id":10892,"avatar_url":"https://avatars.githubusercontent.com/u/10892?v=3","gravatar_id":""...

```

注意我们如何明确地需要使用 `system.shutdown` 来关闭演员系统。程序会挂起，直到系统关闭。然而，关闭系统将停止所有演员，因此我们需要确保他们已经完成工作。我们通过插入对 `Thread.sleep` 的调用来实现这一点。

使用 `Thread.sleep` 等待直到 API 调用完成以关闭演员系统是一种比较粗糙的方法。更好的方法可能是让演员向系统发出信号，表明他们已经完成了任务。当我们实现 *fetcher manager* 演员时，我们将看到这种模式的示例。

Akka 包含一个功能丰富的 *scheduler* 来安排事件。我们可以使用调度器来替换对 `Thread.sleep` 的调用，并安排在五秒后关闭系统。这比 `Thread.sleep` 更好，因为调度器不会阻塞调用线程。要使用调度器，我们需要导入全局执行上下文和持续时间模块：

```py
// FetcherDemoWithScheduler.scala

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
```

然后，我们可以通过用以下代码替换对 `Thread.sleep` 的调用来安排系统关闭：

```py
system.scheduler.scheduleOnce(5.seconds) { system.shutdown }
```

除了 `scheduleOnce`，调度器还公开了一个 `schedule` 方法，允许您定期安排事件的发生（例如，每两秒一次）。这对于心跳检查或监控系统非常有用。有关更多信息，请参阅位于 [`doc.akka.io/docs/akka/snapshot/scala/scheduler.html`](http://doc.akka.io/docs/akka/snapshot/scala/scheduler.html) 的调度器 API 文档。

注意，我们在这里实际上有点作弊，因为我们没有获取每个关注者的信息。关注者查询的响应实际上是分页的，因此我们需要获取几页才能获取所有关注者。在演员中添加逻辑来完成这项工作并不复杂。我们目前将忽略这一点，并假设每个用户的关注者数量上限为 100。

# 路由

在上一个示例中，我们创建了四个 fetchers 并将消息依次派发给了它们。我们在其中分配任务的一组相同 actor 中有一个池。手动将消息路由到正确的 actor 以最大化我们池的利用率是痛苦且容易出错的。幸运的是，Akka 为我们提供了几个路由策略，我们可以使用这些策略在我们的 actor 池中分配工作。让我们用自动路由重写之前的示例。您可以在本书提供的示例代码中的`chap09/fetchers_routing`目录下找到本节的代码示例（[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)）。我们将重用与之前章节相同的`Fetchers`及其伴随对象的定义。

让我们先导入路由包：

```py
// FetcherDemo.scala
import akka.routing._
```

一个*router*是一个 actor，它将其接收到的消息转发给其子 actor。定义 actor 池的最简单方法就是告诉 Akka 创建一个 router 并传递一个为其子 actor 的`Props`对象。然后，router 将直接管理工作者的创建。在我们的示例中（我们将在文本中仅注释与上一个示例不同的部分，但您可以在本章的示例代码的`fetchers_routing`目录中找到完整的代码），我们用以下代码替换了自定义的`Fetcher`创建代码：

```py
// FetcherDemo.scala

// Create a router with 4 workers of props Fetcher.props()
val router = system.actorOf(
  RoundRobinPool(4).props(Fetcher.props(token))
)
```

然后，我们可以直接将 fetch 消息发送到路由器。路由器将以轮询的方式将消息路由到其子 actor：

```py
List("odersky", "derekwyatt", "rkuhn", "tototoshi").foreach { 
  login => router ! Fetch(login)
}
```

在这个示例中，我们使用了轮询路由器。Akka 提供了许多不同类型的路由器，包括具有动态池大小的路由器，以满足不同类型的负载均衡。前往 Akka 文档查看所有可用的路由器列表，请参阅[`doc.akka.io/docs/akka/snapshot/scala/routing.html`](http://doc.akka.io/docs/akka/snapshot/scala/routing.html)。

# Actor 之间的消息传递

仅记录 API 响应并不很有用。为了遍历跟随者图，我们必须执行以下操作：

+   检查响应的返回码以确保 GitHub API 对我们的请求满意

+   将响应解析为 JSON

+   提取跟随者的登录名，如果我们还没有获取它们，将它们推入队列

您在第七章中学习了如何做所有这些事情，*Web APIs*，但不是在 actor 的上下文中。

我们可以将额外的处理步骤添加到我们的`Fetcher` actor 的`receive`方法中：我们可以通过 future composition 添加进一步的转换到 API 响应。然而，让 actor 执行多个不同的操作，并且可能以多种方式失败，是一种反模式：当我们学习管理 actor 生命周期时，我们会看到如果 actor 包含多个逻辑片段，我们的 actor 系统就变得难以推理。

因此，我们将使用三个不同 actor 的管道：

+   我们已经遇到的 fetchers 仅负责从 GitHub 获取 URL。如果 URL 格式不正确或无法访问 GitHub API，它们将失败。

+   响应解释器负责从 GitHub API 获取响应并将其解析为 JSON。如果在任何步骤中失败，它将仅记录错误（在实际应用中，我们可能会根据失败类型采取不同的纠正措施）。如果它成功提取 JSON，它将把 JSON 数组传递给跟随者提取器。

+   跟随者提取器将从 JSON 数组中提取跟随者，并将它们传递给需要获取其跟随者的用户队列。

我们已经构建了 fetchers，尽管我们需要修改它们，以便将 API 响应转发给响应解释器，而不仅仅是记录日志。

你可以在本书提供的示例代码中的`chap09/all_workers`目录中找到本节的代码示例（[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)）。第一步是修改 fetchers，使其在记录响应而不是转发响应。为了能够将响应转发给响应解释器，fetchers 将需要一个对这个演员的引用。我们将通过 fetcher 构造函数传递这个引用，现在的构造函数是：

```py
// Fetcher.scala
class Fetcher(
  val token:Option[String], 
  val responseInterpreter:ActorRef) 
extends Actor with ActorLogging {
  ...
}
```

我们还必须修改伴随对象中的`Props`工厂方法：

```py
// Fetcher.scala
def props(
  token:Option[String], responseInterpreter:ActorRef
):Props = Props(classOf[Fetcher], token, responseInterpreter)
```

我们还必须修改`receive`方法，将 HTTP 响应转发给解释器，而不仅仅是记录日志：

```py
// Fetcher.scala
class Fetcher(...) extends Actor with ActorLogging {
  ...
  def receive = {
    case Fetch(login) => fetchFollowers(login)
  }

  private def fetchFollowers(login:String) {
    val unauthorizedRequest = Http(
      s"https://api.github.com/users/$login/followers")
    val authorizedRequest = token.map { t =>
      unauthorizedRequest.header("Authorization", s"token $t")
    }

    val request = authorizedRequest.getOrElse(unauthorizedRequest)
    val response = Future { request.asString }

    // Wrap the response in an InterpretResponse message and
    // forward it to the interpreter.
    response.onComplete { r =>
      responseInterpreter !
        ResponseInterpreter.InterpretResponse(login, r)
    }
  }
}
```

*响应解释器*获取响应，判断其是否有效，将其解析为 JSON，并将其转发给跟随者提取器。响应解释器将需要一个指向跟随者提取器的引用，我们将通过构造函数传递这个引用。

让我们从定义`ResponseInterpreter`伴随对象开始。它将只包含响应解释器可以接收的消息定义以及一个用于创建`Props`对象的工厂，以帮助进行实例化：

```py
// ResponseInterpreter.scala
import akka.actor._
import scala.util._

import scalaj.http._
import org.json4s._
import org.json4s.native.JsonMethods._

object ResponseInterpreter {

  // Messages
  case class InterpretResponse(
    login:String, response:Try[HttpResponse[String]]
  )

  // Props factory
  def props(followerExtractor:ActorRef) = 
    Props(classOf[ResponseInterpreter], followerExtractor)
}
```

`ResponseInterpreter`的主体应该感觉熟悉：当演员收到一个提供要解释的响应的消息时，它使用你在第七章，“Web APIs”中学到的技术将其解析为 JSON。如果我们成功解析响应，我们将把解析后的 JSON 转发给跟随者提取器。如果我们无法解析响应（可能是由于格式不正确），我们只需记录错误。我们可以以其他方式从中恢复，例如，通过将此登录重新添加到队列管理器中以便再次获取：

```py
// ResponseInterpreter.scala
class ResponseInterpreter(followerExtractor:ActorRef) 
extends Actor with ActorLogging {
  // Import the message definitions
  import ResponseInterpreter._

  def receive = {
    case InterpretResponse(login, r) => interpret(login, r)
  }

  // If the query was successful, extract the JSON response
  // and pass it onto the follower extractor.
  // If the query failed, or is badly formatted, throw an error
  // We should also be checking error codes here.
  private def interpret(
    login:String, response:Try[HttpResponse[String]]
  ) = response match {
    case Success(r) => responseToJson(r.body) match {
      case Success(jsonResponse) => 
        followerExtractor ! FollowerExtractor.Extract(
          login, jsonResponse)
      case Failure(e) => 
        log.error(
          s"Error parsing response to JSON for $login: $e")
    }
    case Failure(e) => log.error(
      s"Error fetching URL for $login: $e")
  }

  // Try and parse the response body as JSON. 
  // If successful, coerce the `JValue` to a `JArray`.
  private def responseToJson(responseBody:String):Try[JArray] = {
    val jvalue = Try { parse(responseBody) }
    jvalue.flatMap {
      case a:JArray => Success(a)
      case _ => Failure(new IllegalStateException(
        "Incorrectly formatted JSON: not an array"))
    }
  }
}
```

现在我们已经有了三分之二的工人演员。最后一个链接是跟随者提取器。这个演员的工作很简单：它接受响应解释器传递给它的`JArray`，并将其转换为跟随者列表。目前，我们只是记录这个列表，但当我们构建获取器管理者时，跟随者提取器将发送消息要求管理者将其添加到要获取的登录队列中。

如前所述，伴随者仅定义了此演员可以接收的消息以及一个 Props 工厂方法：

```py
// FollowerExtractor.scala
import akka.actor._

import org.json4s._
import org.json4s.native.JsonMethods._

object FollowerExtractor {

  // Messages
  case class Extract(login:String, jsonResponse:JArray)

  // Props factory method
  def props = Props[FollowerExtractor]
}
```

`FollowerExtractor`类接收包含表示跟随者的`JArray`信息的`Extract`消息。它提取`login`字段并记录它：

```py
class FollowerExtractor extends Actor with ActorLogging {
  import FollowerExtractor._
  def receive = {
    case Extract(login, followerArray) => {
      val followers = extractFollowers(followerArray)
      log.info(s"$login -> ${followers.mkString(", ")}")
    }
  }

  def extractFollowers(followerArray:JArray) = for {
    JObject(follower) <- followerArray
    JField("login", JString(login)) <- follower
  } yield login
}
```

让我们编写一个新的`main`方法来测试所有我们的演员：

```py
// FetchNetwork.scala

import akka.actor._
import akka.routing._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

object FetchNetwork extends App {

  import Fetcher._ // Import messages and factory method

  // Get token if exists
  val token = sys.env.get("GHTOKEN")

  val system = ActorSystem("fetchers")

  // Instantiate actors
  val followerExtractor = system.actorOf(FollowerExtractor.props)
  val responseInterpreter =   
    system.actorOf(ResponseInterpreter.props(followerExtractor))

  val router = system.actorOf(RoundRobinPool(4).props(
    Fetcher.props(token, responseInterpreter))
  )

  List("odersky", "derekwyatt", "rkuhn", "tototoshi") foreach {
    login => router ! Fetch(login)
  }

  // schedule a shutdown
  system.scheduler.scheduleOnce(5.seconds) { system.shutdown }

}
```

让我们通过 SBT 运行这个例子：

```py
$ GHTOKEN="2502761d..." sbt run
[INFO] [11/05/2015 20:09:37.048] [fetchers-akka.actor.default-dispatcher-3] [akka://fetchers/user/$a] derekwyatt -> adulteratedjedi, joonas, Psycojoker, trapd00r, tyru, ...
[INFO] [11/05/2015 20:09:37.050] [fetchers-akka.actor.default-dispatcher-3] [akka://fetchers/user/$a] tototoshi -> akr4, yuroyoro, seratch, yyuu, ...
[INFO] [11/05/2015 20:09:37.051] [fetchers-akka.actor.default-dispatcher-3] [akka://fetchers/user/$a] odersky -> misto, gkossakowski, mushtaq, ...
[INFO] [11/05/2015 20:09:37.052] [fetchers-akka.actor.default-dispatcher-3] [akka://fetchers/user/$a] rkuhn -> arnbak, uzoice, jond3k, TimothyKlim, relrod, ...

```

# 队列控制和拉取模式

我们现在已经在我们的爬虫应用程序中定义了三个工作演员。下一步是定义管理者。*获取器管理者*负责维护一个要获取的登录队列以及一组我们已经看到的登录名称，以避免多次获取相同的登录。

第一个尝试可能涉及构建一个演员，它维护一组我们已经看到的用户，并在给定一个新用户获取时将其调度到轮询路由器。这种方法的问题在于获取器邮箱中的消息数量会迅速积累：对于每个 API 查询，我们可能会得到数十个跟随者，每个跟随者都可能回到获取器的收件箱。这使我们很难控制堆积的工作量。

这个问题可能导致的第一个问题是 GitHub API 速率限制：即使有认证，我们每小时也限制在 5,000 个请求。当我们达到这个阈值时，停止查询会有用。如果每个获取器都有数百个需要获取的用户积压，我们就无法做出响应。

一个更好的替代方案是使用*拉取*系统：当获取器发现自己空闲时，它们会从中央队列请求工作。在 Akka 中，当我们有一个生产者比消费者处理得更快时，拉取系统很常见（参考[`www.michaelpollmeier.com/akka-work-pulling-pattern/`](http://www.michaelpollmeier.com/akka-work-pulling-pattern/)）。

管理者和获取器之间的对话将如下进行：

+   如果管理者从没有工作状态变为有工作状态，它会向所有获取器发送`WorkAvailable`消息。

+   当获取器收到`WorkAvailable`消息或完成一项工作时，它会向队列管理者发送`GiveMeWork`消息。

+   当队列管理者收到`GiveMeWork`消息时，如果没有工作可用或它被限制，它会忽略请求。如果有工作，它向演员发送`Fetch(user)`消息。

让我们从修改我们的 fetcher 开始。您可以在本书提供的示例代码中的`chap09/ghub_crawler`目录中找到本节的代码示例([`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds))。我们将通过构造函数传递 fetcher 管理器的引用。我们需要更改伴随对象以添加`WorkAvailable`消息，并将`props`工厂包括到管理器的引用中：

```py
// Fecther.scala
object Fetcher {
  case class Fetch(url:String)
  case object WorkAvailable

  def props(
    token:Option[String], 
    fetcherManager:ActorRef, 
    responseInterpreter:ActorRef):Props =
      Props(classOf[Fetcher], 
        token, fetcherManager, responseInterpreter)
}
```

我们还需要更改`receive`方法，以便在处理完一个请求或接收到`WorkAvailable`消息后，它能够查询`FetcherManager`以获取更多工作。

这是 fetchers 的最终版本：

```py
class Fetcher(
  val token:Option[String], 
  val fetcherManager:ActorRef,
  val responseInterpreter:ActorRef) 
extends Actor with ActorLogging {
  import Fetcher._
  import context.dispatcher

  def receive = {
    case Fetch(login) => fetchFollowers(login)
    case WorkAvailable => 
      fetcherManager ! FetcherManager.GiveMeWork
  }

  private def fetchFollowers(login:String) {
    val unauthorizedRequest = Http(
      s"https://api.github.com/users/$login/followers")
    val authorizedRequest = token.map { t =>
      unauthorizedRequest.header("Authorization", s"token $t")
    }
    val request = authorizedRequest.getOrElse(unauthorizedRequest)
    val response = Future { request.asString }

    response.onComplete { r =>
      responseInterpreter ! 
        ResponseInterpreter.InterpretResponse(login, r)
      fetcherManager ! FetcherManager.GiveMeWork
    }
  }

}
```

现在我们已经有一个工作定义的 fetchers，让我们构建`FetcherManager`。这是我们迄今为止构建的最复杂的 actor，在我们深入构建它之前，我们需要更多地了解 Akka 工具包的组件。

# 访问消息的发送者

当我们的 fetcher 管理器收到`GiveMeWork`请求时，我们需要将工作发送回正确的 fetcher。我们可以使用`sender`方法访问发送消息的 actor，这是`Actor`的一个方法，它返回正在处理的消息对应的`ActorRef`。因此，fetcher 管理器中对应于`GiveMeWork`的`case`语句是：

```py
def receive = {
  case GiveMeWork =>
    login = // get next login to fetch
    sender ! Fetcher.Fetch(login)
  ...
}
```

由于`sender`是一个*方法*，它的返回值会随着每个新传入的消息而改变。因此，它应该仅与`receive`方法同步使用。特别是，在 future 中使用它是危险的：

```py
def receive = {
  case DoSomeWork =>
    val work = Future { Thread.sleep(20000) ; 5 }
    work.onComplete { result => 
      sender ! Complete(result) // NO!
    }
}
```

问题在于，当 future 在消息处理后的 20 秒完成时，actor 很可能会处理不同的消息，因此`sender`的返回值将改变。因此，我们将`Complete`消息发送给一个完全不同的 actor。

如果您需要在`receive`方法之外回复消息，例如当 future 完成时，您应该将当前发送者的值绑定到一个变量上：

```py
def receive = {
  case DoSomeWork =>
    // bind the current value of sender to a val
    val requestor = sender
    val work = Future { Thread.sleep(20000) ; 5 }
    work.onComplete { result => requestor ! Complete(result) }
}
```

# 有状态演员

fetcher 管理器的行为取决于它是否有工作要分配给 fetchers：

+   如果它有工作要提供，它需要用`Fetcher.Fetch`消息响应`GiveMeWork`消息

+   如果没有工作，它必须忽略`GiveMeWork`消息，并且如果添加了工作，它必须向 fetchers 发送一个`WorkAvailable`消息。

在 Akka 中编码状态的概念很简单。我们指定不同的`receive`方法，并根据状态从一个切换到另一个。我们将为我们的 fetcher 管理器定义以下`receive`方法，对应于每个状态：

```py
// receive method when the queue is empty
def receiveWhileEmpty: Receive = { 
    ... 
}

// receive method when the queue is not empty
def receiveWhileNotEmpty: Receive = {
    ...
}
```

注意，我们必须将接收方法的返回类型定义为`Receive`。为了将演员从一个方法切换到另一个方法，我们可以使用`context.become(methodName)`。例如，当最后一个登录名从队列中弹出时，我们可以通过`context.become(receiveWhileEmpty)`过渡到使用`receiveWhileEmpty`方法。我们通过将`receiveWhileEmpty`分配给`receive`方法来设置初始状态：

```py
def receive = receiveWhileEmpty
```

# 跟随者网络爬虫

我们现在准备编写网络爬虫剩余部分的代码。最大的缺失部分是获取器管理器。让我们从伴随对象开始。与工作演员类似，这仅包含演员可以接收的消息定义以及创建`Props`实例的工厂：

```py
// FetcherManager.scala
import scala.collection.mutable
import akka.actor._

object FetcherManager {
  case class AddToQueue(login:String)
  case object GiveMeWork

  def props(token:Option[String], nFetchers:Int) = 
    Props(classOf[FetcherManager], token, nFetchers)
}
```

管理器可以接收两种消息：`AddToQueue`，它告诉管理器将用户名添加到需要获取跟随者的用户队列中，以及由获取器在失业时发出的`GiveMeWork`。

管理器将负责启动获取器、响应解释器和跟随者提取器，以及维护一个用户名内部队列和一组我们已看到的用户名：

```py
// FetcherManager.scala

class FetcherManager(val token:Option[String], val nFetchers:Int) 
extends Actor with ActorLogging {

  import FetcherManager._

  // queue of usernames whose followers we need to fetch
  val fetchQueue = mutable.Queue.empty[String]

  // set of users we have already fetched. 
  val fetchedUsers = mutable.Set.empty[String]

  // Instantiate worker actors
  val followerExtractor = context.actorOf(
    FollowerExtractor.props(self))
  val responseInterpreter = context.actorOf(
    ResponseInterpreter.props(followerExtractor))
  val fetchers = (0 until nFetchers).map { i =>
    context.actorOf(
      Fetcher.props(token, self, responseInterpreter))
  }

  // receive method when the actor has work:
  // If we receive additional work, we just push it onto the
  // queue.
  // If we receive a request for work from a Fetcher,
  // we pop an item off the queue. If that leaves the 
  // queue empty, we transition to the 'receiveWhileEmpty'
  // method.
  def receiveWhileNotEmpty:Receive = {
    case AddToQueue(login) => queueIfNotFetched(login)
    case GiveMeWork =>
      val login = fetchQueue.dequeue
      // send a Fetch message back to the sender.
      // we can use the `sender` method to reply to a message
      sender ! Fetcher.Fetch(login)
      if (fetchQueue.isEmpty) { 
        context.become(receiveWhileEmpty) 
      }
  }

  // receive method when the actor has no work:
  // if we receive work, we add it onto the queue, transition
  // to a state where we have work, and notify the fetchers
  // that work is available.
  def receiveWhileEmpty:Receive = {
    case AddToQueue(login) =>
      queueIfNotFetched(login)
      context.become(receiveWhileNotEmpty)
      fetchers.foreach { _ ! Fetcher.WorkAvailable }
    case GiveMeWork => // do nothing
  }

  // Start with an empty queue.
  def receive = receiveWhileEmpty

  def queueIfNotFetched(login:String) {
    if (! fetchedUsers(login)) {
      log.info(s"Pushing $login onto queue") 
      // or do something useful...
      fetchQueue += login
      fetchedUsers += login
    }
  }
}
```

我们现在有了获取器管理器。除了跟随者提取器之外，其余的代码可以保持不变。而不是记录跟随者名称，它必须向管理器发送`AddToQueue`消息。我们将在构造时传递管理器的引用：

```py
// FollowerExtractor.scala
import akka.actor._
import org.json4s._
import org.json4s.native.JsonMethods._

object FollowerExtractor {

  // messages
  case class Extract(login:String, jsonResponse:JArray)

  // props factory method
  def props(manager:ActorRef) = 
    Props(classOf[FollowerExtractor], manager)
}

class FollowerExtractor(manager:ActorRef)
extends Actor with ActorLogging {
  import FollowerExtractor._

  def receive = {
    case Extract(login, followerArray) =>
      val followers = extractFollowers(followerArray)
      followers foreach { f => 
        manager ! FetcherManager.AddToQueue(f) 
      }
  }

  def extractFollowers(followerArray:JArray) = for {
    JObject(follower) <- followerArray
    JField("login", JString(login)) <- follower
  } yield login

}
```

运行所有这些的`main`方法非常简单，因为所有实例化演员的代码都已移动到`FetcherManager`。我们只需要实例化管理器，并给它网络中的第一个节点，然后它会完成其余的工作：

```py
// FetchNetwork.scala
import akka.actor._

object FetchNetwork extends App {

  // Get token if exists
  val token = sys.env.get("GHTOKEN")

  val system = ActorSystem("GithubFetcher")
  val manager = system.actorOf(FetcherManager.props(token, 2))
  manager ! FetcherManager.AddToQueue("odersky")

}
```

注意我们不再尝试关闭演员系统了。我们将让它运行，爬取网络，直到我们停止它或达到认证限制。让我们通过 SBT 运行这个程序：

```py
$ GHTOKEN="2502761d..." sbt "runMain FetchNetwork"
[INFO] [11/06/2015 06:31:04.614] [GithubFetcher-akka.actor.default-dispatcher-2] [akka://GithubFetcher/user/$a] Pushing odersky onto queue
[INFO] [11/06/2015 06:31:05.563] [GithubFetcher-akka.actor.default-dispatcher-4] [akka://GithubFetcher/user/$a] Pushing misto onto queueINFO] [11/06/2015 06:31:05.563] [GithubFetcher-akka.actor.default-dispatcher-4] [akka://GithubFetcher/user/$a] Pushing gkossakowski onto queue
^C

```

我们的程序实际上并没有对检索到的跟随者做任何有用的操作，除了记录它们。我们可以将`log.info`调用替换为，例如，将节点存储在数据库中或在屏幕上绘制图形。

# 容错性

真实程序会失败，并且以不可预测的方式失败。Akka 以及整个 Scala 社区都倾向于明确规划失败，而不是试图编写不可失败的应用程序。一个*容错*系统是指当其一个或多个组件失败时仍能继续运行的系统。单个子系统的失败并不一定意味着应用程序的失败。这如何应用于 Akka？

演员模型提供了一个自然的单元来封装失败：演员。当一个演员在处理消息时抛出异常，默认行为是演员重启，但异常不会泄露并影响系统的其余部分。例如，让我们在响应解释器中引入一个任意的失败。我们将修改`receive`方法，当它被要求解释`misto`（Martin Odersky 的一个关注者）的响应时抛出异常：

```py
// ResponseInterpreter.scala
def receive = {
  case InterpretResponse("misto", r) => 
    throw new IllegalStateException("custom error")
  case InterpretResponse(login, r) => interpret(login, r)
}
```

如果你通过 SBT 重新运行代码，你会注意到记录了一个错误。然而，程序并没有崩溃。它只是继续正常运行：

```py
[ERROR] [11/07/2015 12:05:58.938] [GithubFetcher-akka.actor.default-dispatcher-2] [akka://GithubFetcher/user/$a/$b] custom error
java.lang.IllegalStateException: custom error
 at ResponseInterpreter$
 ...
[INFO] [11/07/2015 12:05:59.117] [GithubFetcher-akka.actor.default-dispatcher-2] [akka://GithubFetcher/user/$a] Pushing samfoo onto queue

```

`misto`的任何关注者都不会被添加到队列中：他从未通过`ResponseInterpreter`阶段。让我们逐步了解当异常被抛出时会发生什么：

+   解释器接收到`InterpretResponse("misto", ...)`消息。这导致它抛出异常并死亡。其他演员不受异常的影响。

+   使用与最近去世的演员相同的 Props 实例创建响应解释器的新实例。

+   当响应解释器完成初始化后，它被绑定到与去世演员相同的`ActorRef`。这意味着，对于系统的其余部分来说，没有任何变化。

+   邮箱绑定到`ActorRef`而不是演员，因此新的响应解释器将与其前任具有相同的邮箱，而不包括有问题的消息。

因此，无论出于什么原因，我们的爬虫在抓取或解析用户响应时崩溃，应用程序的影响将最小——我们只是不会抓取此用户的关注者。

当演员重启时，它携带的任何内部状态都会丢失。因此，例如，如果 fetcher 管理器死亡，我们将丢失队列的当前值和已访问用户。可以通过以下方式减轻丢失内部状态的风险：

+   采用不同的失败策略：例如，在失败的情况下，我们可以继续处理消息而不重启演员。当然，如果演员死亡是因为其内部状态不一致，这几乎没有什么用处。在下一节中，我们将讨论如何更改失败恢复策略。

+   通过定期将其写入磁盘并从备份中重新启动时加载来备份内部状态。

+   通过确保所有“风险”操作都委派给其他演员来保护携带关键状态的演员。在我们的爬虫示例中，所有与外部服务的交互，例如查询 GitHub API 和解析响应，都是通过不带内部状态的演员进行的。正如我们在前面的示例中看到的，如果这些演员中的任何一个死亡，应用程序的影响将最小。相比之下，宝贵的 fetcher 管理器只允许与经过清理的输入进行交互。这被称为*错误内核*模式：可能引起错误的代码被委派给自杀式演员。

# 自定义监督策略

在失败时重新启动演员的默认策略并不总是我们想要的。特别是对于携带大量数据的演员，我们可能希望在异常后恢复处理而不是重新启动演员。Akka 允许我们通过在演员的监督者中设置*监督策略*来自定义此行为。

请记住，所有演员都有父母，包括顶级演员，它们是称为*用户守护者*的特殊演员的孩子。默认情况下，演员的监督者是它的父母，监督者决定在失败时对演员做什么。

因此，要改变演员对失败的反应方式，你必须设置其父母的监督策略。你通过设置`supervisorStrategy`属性来完成此操作。默认策略等同于以下内容：

```py
val supervisorStrategy = OneForOneStrategy() {
  case _:ActorInitializationException => Stop
  case _:ActorKilledException => Stop
  case _:DeathPactException => Stop
  case _:Exception => Restart
}
```

一个监督策略有两个组成部分：

+   `OneForOneStrategy`确定策略仅适用于失败的演员。相比之下，我们可以使用`AllForOneStrategy`，它将相同的策略应用于所有被监督者。如果一个子进程失败，所有子进程都将被重新启动（或停止或恢复）。

+   一个部分函数映射`Throwables`到`Directive`，这是一个关于在失败时如何操作的指令。例如，默认策略将`ActorInitializationException`（如果构造函数失败会发生）映射到`Stop`指令，以及（几乎所有）其他异常映射到`Restart`。

有四个指令：

+   `Restart`：这会销毁有缺陷的演员并重新启动它，将新生成的演员绑定到旧的`ActorRef`。这将清除演员的内部状态，这可能是一件好事（演员可能因为某些内部不一致性而失败）。

+   `Resume`：演员只需继续处理其收件箱中的下一个消息。

+   `Stop`：演员停止并不会被重新启动。这在用于完成单个操作的丢弃演员中很有用：如果这个操作失败，演员就不再需要了。

+   `Escalate`：监督者本身重新抛出异常，希望它的监督者知道如何处理它。

监督者无法访问其子进程失败的情况。因此，如果一个演员的子进程可能需要不同的恢复策略，最好创建一组中间监督演员来监督不同的子进程组。

作为设置监督策略的示例，让我们调整`FetcherManager`的监督策略以采用全为一策略，并在其中一个子进程失败时停止其子进程。我们首先进行相关导入：

```py
import akka.actor.SupervisorStrategy._
```

然后，我们只需在`FetcherManager`定义中设置`supervisorStrategy`属性：

```py
class FetcherManager(...) extends Actor with ActorLogging {

  ...

  override val supervisorStrategy = AllForOneStrategy() {
    case _:ActorInitializationException => Stop
    case _:ActorKilledException => Stop
    case _:Exception => Stop
  }

  ...
}
```

如果你通过 SBT 运行它，你会注意到当代码遇到响应解释器抛出的自定义异常时，系统会停止。这是因为除了检索管理器之外的所有演员现在都已失效。

# 生命周期钩子

Akka 允许我们通过*生命周期钩子*指定在演员的生命周期中响应特定事件的代码。Akka 定义了以下钩子：

+   `preStart()`：在演员构造函数完成后但开始处理消息之前运行。这对于运行依赖于演员完全构建的初始化代码很有用。

+   `postStop()`：当演员停止处理消息后死亡时运行。在终止演员之前运行清理代码很有用。

+   `preRestart(reason: Throwable, message: Option[Any])`：在演员收到重启命令后立即调用。`preRestart`方法可以访问抛出的异常和有问题的消息，允许进行纠正操作。`preRestart`的默认行为是停止每个子演员，然后调用`postStop`。

+   `postRestart(reason:Throwable)`：在演员重启后调用。默认行为是调用`preStart()`。

让我们使用系统钩子来在程序运行之间持久化`FetcherManager`的状态。您可以在本书提供的示例代码中的`chap09/ghub_crawler_fault_tolerant`目录中找到本节的代码示例（[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)）。这将使 fetcher 管理器具有容错性。我们将使用`postStop`将当前队列和已访问用户集合写入文本文件，并使用`preStart`从磁盘读取这些文本文件。让我们首先导入读取和写入文件所需的库：

```py
// FetcherManager.scala

import scala.io.Source 
import scala.util._
import java.io._
```

我们将在`FetcherManager`伴随对象（更好的方法是将它们存储在配置文件中）中存储两个文本文件的名称：

```py
// FetcherManager.scala
object FetcherManager {
  ...
  val fetchedUsersFileName = "fetched-users.txt"
  val fetchQueueFileName = "fetch-queue.txt"
}
```

在`preStart`方法中，我们从文本文件中加载已获取用户集合和要获取用户的后备队列，并在`postStop`方法中，我们用这些数据结构的新值覆盖这些文件：

```py
class FetcherManager(
  val token:Option[String], val nFetchers:Int
) extends Actor with ActorLogging {

  ...

  /** pre-start method: load saved state from text files */
  override def preStart {
    log.info("Running pre-start on fetcher manager")

    loadFetchedUsers
    log.info(
      s"Read ${fetchedUsers.size} visited users from source"
    )

    loadFetchQueue
    log.info(
      s"Read ${fetchQueue.size} users in queue from source"
    )

    // If the saved state contains a non-empty queue, 
    // alert the fetchers so they can start working.
    if (fetchQueue.nonEmpty) {
      context.become(receiveWhileNotEmpty)
      fetchers.foreach { _ ! Fetcher.WorkAvailable }
    }

  }

  /** Dump the current state of the manager */
  override def postStop {
    log.info("Running post-stop on fetcher manager")
    saveFetchedUsers
    saveFetchQueue
  }

     /* Helper methods to load from and write to files */
  def loadFetchedUsers {
    val fetchedUsersSource = Try { 
      Source.fromFile(fetchedUsersFileName) 
    }
    fetchedUsersSource.foreach { s =>
      try s.getLines.foreach { l => fetchedUsers += l }
      finally s.close
    }
  }

  def loadFetchQueue {
    val fetchQueueSource = Try { 
      Source.fromFile(fetchQueueFileName) 
    }
    fetchQueueSource.foreach { s =>
      try s.getLines.foreach { l => fetchQueue += l }
      finally s.close
    }
  }

  def saveFetchedUsers {
    val fetchedUsersFile = new File(fetchedUsersFileName)
    val writer = new BufferedWriter(
      new FileWriter(fetchedUsersFile))
    fetchedUsers.foreach { user => writer.write(user + "\n") }
    writer.close()
  }

  def saveFetchQueue {
    val queueUsersFile = new File(fetchQueueFileName)
    val writer = new BufferedWriter(
      new FileWriter(queueUsersFile))
    fetchQueue.foreach { user => writer.write(user + "\n") }
    writer.close()
  }

...
}
```

现在我们保存了爬虫在关闭时的状态，我们可以为程序设置比仅仅在无聊时中断程序更好的终止条件。在生产中，例如，当我们在数据库中有足够的名字时，我们可能会停止爬虫。在这个例子中，我们将简单地让爬虫运行 30 秒然后关闭它。

让我们修改`main`方法：

```py
// FetchNetwork.scala
import akka.actor._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

object FetchNetwork extends App {

  // Get token if exists
  val token = sys.env.get("GHTOKEN")

  val system = ActorSystem("GithubFetcher")
  val manager = system.actorOf(FetcherManager.props(token, 2))

  manager ! FetcherManager.AddToQueue("odersky")

  system.scheduler.scheduleOnce(30.seconds) { system.shutdown }

}
```

30 秒后，我们只需调用`system.shutdown`，这将递归地停止所有演员。这将停止 fetcher 管理器，并调用`postStop`生命周期钩子。程序运行一次后，我在`fetched-users.txt`文件中有 2,164 个名字。再次运行它将用户数量增加到 3,728。

我们可以通过在代码运行时让 fetcher manager 定期转储数据结构来进一步提高容错性。由于写入磁盘（或数据库）存在一定的风险（如果数据库服务器宕机或磁盘空间不足怎么办？），将数据结构的写入委托给一个自定义演员而不是危及管理者会更好。

我们的爬虫有一个小问题：当 fetcher manager 停止时，它会停止 fetcher actors、响应解释器和 follower extractor。然而，目前通过这些演员的用户信息都没有被存储。这也导致代码末尾有少量未投递的消息：如果响应解释器在 fetcher 之前停止，fetcher 将尝试向一个不存在的演员投递。这仅涉及少数用户。为了恢复这些登录名，我们可以创建一个收割者 actor，其任务是协调正确顺序地杀死所有工作 actor 并收集它们的内部状态。这种模式已在 Derek Wyatt 的博客文章中记录（[`letitcrash.com/post/30165507578/shutdown-patterns-in-akka-2`](http://letitcrash.com/post/30165507578/shutdown-patterns-in-akka-2)）。

# 我们还没有讨论的内容

Akka 是一个非常丰富的生态系统，远远超出了单章所能公正地描述的范围。有一些重要的工具包部分是你需要的，但我们在这里没有涵盖。我们将给出简要描述，但你可以在 Akka 文档中查找更多详细信息：

+   询问操作符 `?` 提供了 `!` 操作符的替代方案，我们曾用它向演员发送消息。与“tell”不同，后者只是向演员发送消息，询问操作符期望得到响应。当我们需要向演员提问而不是仅仅告诉他们做什么时，这很有用。询问模式已在[`doc.akka.io/docs/akka/snapshot/scala/actors.html#Ask__Send-And-Receive-Future`](http://doc.akka.io/docs/akka/snapshot/scala/actors.html#Ask__Send-And-Receive-Future)中记录。

+   Deathwatch 允许演员观察另一个演员，并在它死亡时接收消息。这对于可能依赖于另一个演员但不是其直接管理者的演员很有用。这已在[`doc.akka.io/docs/akka/snapshot/scala/actors.html#Lifecycle_Monitoring_aka_DeathWatch`](http://doc.akka.io/docs/akka/snapshot/scala/actors.html#Lifecycle_Monitoring_aka_DeathWatch)中记录。

+   在我们的爬虫中，我们通过构造函数显式传递了演员的引用。我们也可以使用演员层次结构通过类似文件系统中的文件语法查找演员。[`doc.akka.io/docs/akka/snapshot/scala/actors.html#Identifying_Actors_via_Actor_Selection`](http://doc.akka.io/docs/akka/snapshot/scala/actors.html#Identifying_Actors_via_Actor_Selection)。

+   我们简要探讨了如何使用不同的接收方法实现有状态的 actor，并使用`context.become`在它们之间切换。Akka 提供了一个更强大的替代方案，基于有限状态机，以编码更复杂的状态集和转换：[`doc.akka.io/docs/akka/snapshot/scala/fsm.html`](http://doc.akka.io/docs/akka/snapshot/scala/fsm.html)。

+   在本章中，我们没有讨论如何在多个节点上分发 actor 系统。消息传递架构与分布式设置配合良好：[`doc.akka.io/docs/akka/2.4.0/common/cluster.html`](http://doc.akka.io/docs/akka/2.4.0/common/cluster.html)。

# 摘要

在本章中，你学习了如何将 actor 编织在一起以解决一个困难的并发问题。更重要的是，我们看到了 Akka 的 actor 框架如何鼓励我们以许多独立的封装可变数据块的形式来思考并发问题，这些数据块通过消息传递进行同步。Akka 使得并发编程更容易推理，并且更有趣。

# 参考文献

*德里克·惠特尼*的书籍《Akka 并发》是 Akka 的绝佳入门指南。对于想要进行严肃 Akka 编程的人来说，这绝对是一个必看的起点。

**“让它崩溃”**博客([`letitcrash.com`](http://letitcrash.com))是官方 Akka 博客，其中包含了许多用于解决常见问题的惯用语句和模式的示例。

# 第十章. 使用 Spark 进行分布式批处理

在第四章《并行集合与未来》中，我们发现了如何使用并行集合来解决“令人尴尬”的并行问题：这些问题可以被分解成一系列不需要（或非常少）任务间通信的任务。

Apache Spark 提供了类似于 Scala 并行集合（以及更多）的行为，但它不是在相同计算机的不同 CPU 上分发任务，而是允许任务在计算机集群中分发。这提供了任意水平的横向可伸缩性，因为我们只需简单地添加更多计算机到集群中。

在本章中，我们将学习 Apache Spark 的基础知识，并使用它来探索一组电子邮件，提取特征，以构建垃圾邮件过滤器为目标。我们将在第十二章《使用 MLlib 进行分布式机器学习》中探索实际构建垃圾邮件过滤器的几种方法。

# 安装 Spark

在前面的章节中，我们通过在`build.sbt`文件中指定依赖项，并依赖 SBT 从 Maven Central 仓库中获取它们来包含依赖项。对于 Apache Spark，显式下载源代码或预构建的二进制文件更为常见，因为 Spark 附带了许多命令行脚本，这些脚本极大地简化了作业的启动和与集群的交互。

访问[`spark.apache.org/downloads.html`](http://spark.apache.org/downloads.html)下载 Spark 1.5.2 版本，选择“为 Hadoop 2.6 或更高版本预构建”的包。如果您需要定制，也可以从源代码构建 Spark，但我们将坚持使用预构建版本，因为它不需要配置。

点击**下载**将下载一个 tar 包，您可以使用以下命令解包：

```py
$ tar xzf spark-1.5.2-bin-hadoop2.6.tgz

```

这将创建一个`spark-1.5.2-bin-hadoop2.6`目录。要验证 Spark 是否正确工作，请导航到`spark-1.5.2-bin-hadoop2.6/bin`并使用`./spark-shell`启动 Spark shell。这只是一个加载了 Spark 库的 Scala shell。

您可能希望将`bin/`目录添加到系统路径中。这样，您就可以从系统中的任何位置调用该目录中的脚本，而无需引用完整路径。在 Linux 或 Mac OS 上，您可以通过在 shell 配置文件（Mac OS 上的`.bash_profile`，Linux 上的`.bashrc`或`.bash_profile`）中输入以下行来将变量添加到系统路径：

```py
export PATH=/path/to/spark/bin:$PATH
```

这些更改将在新的 shell 会话中生效。在 Windows（如果您使用 PowerShell）上，您需要在`Documents`文件夹中`WindowsPowerShell`目录下的`profile.ps1`文件中输入此行：

```py
$env:Path += ";C:\Program Files\GnuWin32\bin"
```

如果操作正确，您应该能够在系统中的任何目录下通过在终端中输入`spark-shell`来打开 Spark shell。

# 获取示例数据

在本章中，我们将探索 Ling-Spam 电子邮件数据集（原始数据集的描述见[`csmining.org/index.php/ling-spam-datasets.html`](http://csmining.org/index.php/ling-spam-datasets.html)）。从[`data.scala4datascience.com/ling-spam.tar.gz`](http://data.scala4datascience.com/ling-spam.tar.gz)（或[ling-spam.zip](http://ling-spam.zip)，取决于您首选的压缩方式）下载数据集，并将内容解压缩到包含本章代码示例的目录中。该存档包含两个目录，`spam/`和`ham/`，分别包含垃圾邮件和合法邮件。

# 弹性分布式数据集

Spark 将所有计算表达为对分布式集合的转换和操作的序列，称为**弹性分布式数据集**（**RDD**）。让我们通过 Spark shell 来探索 RDD 是如何工作的。导航到示例目录，并按照以下方式打开 Spark shell：

```py
$ spark-shell
scala> 

```

让我们从加载一个 RDD 中的电子邮件开始：

```py
scala> val email = sc.textFile("ham/9-463msg1.txt")
email: rdd.RDD[String] = MapPartitionsRDD[1] at textFile

```

`email`是一个 RDD，每个元素对应输入文件中的一行。注意我们是如何通过在名为`sc`的对象上调用`textFile`方法来创建 RDD 的：

```py
scala> sc
spark.SparkContext = org.apache.spark.SparkContext@459bf87c

```

`sc`是一个`SparkContext`实例，代表 Spark 集群（目前是本地机器）的入口点（现在只需我们的本地机器）。当我们启动 Spark shell 时，会创建一个上下文并将其自动绑定到变量`sc`。

让我们使用`flatMap`将电子邮件拆分为单词：

```py
scala> val words = email.flatMap { line => line.split("\\s") }
words: rdd.RDD[String] = MapPartitionsRDD[2] at flatMap

```

如果您熟悉 Scala 中的集合，这会感觉很自然：`email` RDD 的行为就像一个字符串列表。在这里，我们使用表示空白字符的正则表达式 `\s` 进行拆分。我们不仅可以使用 `flatMap` 显式地操作 RDD，还可以使用 Scala 的语法糖来操作 RDD：

```py
scala> val words = for { 
 line <- email
 word <- line.split("\\s") 
} yield word
words: rdd.RDD[String] = MapPartitionsRDD[3] at flatMap

```

让我们检查结果。我们可以使用 `.take(n)` 来提取 RDD 的前 *n* 个元素：

```py
scala> words.take(5)
Array[String] = Array(Subject:, tsd98, workshop, -, -)

```

我们也可以使用 `.count` 来获取 RDD 中的元素数量：

```py
scala> words.count
Long = 939

```

RDD 支持集合支持的大多数操作。让我们使用 `filter` 从我们的电子邮件中删除标点符号。我们将删除包含任何非字母数字字符的所有单词。我们可以通过过滤掉匹配此 *正则表达式* 的元素来实现：`[^a-zA-Z0-9]`。

```py
scala> val nonAlphaNumericPattern = "[^a-zA-Z0-9]".r
nonAlphaNumericPattern: Regex = [^a-zA-Z0-9]

scala> val filteredWords = words.filter { 
 word => nonAlphaNumericPattern.findFirstIn(word) == None 
}
filteredWords: rdd.RDD[String] = MapPartitionsRDD[4] at filter

scala> filteredWords.take(5)
Array[String] = Array(tsd98, workshop, 2nd, call, paper)

scala> filteredWords.count
Long = 627

```

在这个例子中，我们从一个文本文件创建了一个 RDD。我们还可以使用 Spark 上下文上可用的 `sc.parallelize` 方法从 Scala 可迭代对象创建 RDD：

```py
scala> val words = "the quick brown fox jumped over the dog".split(" ") 
words: Array[String] = Array(the, quick, brown, fox, ...)

scala> val wordsRDD = sc.parallelize(words)
wordsRDD: RDD[String] = ParallelCollectionRDD[1] at parallelize at <console>:23

```

这对于调试和在 shell 中试验行为很有用。与并行化相对应的是 `.collect` 方法，它将 RDD 转换为 Scala 数组：

```py
scala> val wordLengths = wordsRDD.map { _.length }
wordLengths: RDD[Int] = MapPartitionsRDD[2] at map at <console>:25

scala> wordLengths.collect
Array[Int] = Array(3, 5, 5, 3, 6, 4, 3, 3)

```

`.collect` 方法需要整个 RDD 都能在主节点上适应内存。因此，它要么用于调试较小的数据集，要么用于管道的末尾，以缩减数据集。

如您所见，RDD 提供了一个类似于 Scala 可迭代对象的 API。关键的区别是 RDD 是 *分布式* 和 *容错的*。让我们探讨这在实践中意味着什么。

## RDD 是不可变的

一旦创建 RDD，就不能更改它。对 RDD 的所有操作要么创建新的 RDD，要么创建其他 Scala 对象。

## RDD 是惰性的

当你在交互式 shell 中对 Scala 集合执行 map 和 filter 等操作时，REPL 会将新集合的值打印到屏幕上。这并不适用于 Spark RDD。这是因为 RDD 上的操作是惰性的：只有在需要时才会进行评估。

因此，当我们写：

```py
val email = sc.textFile(...)
val words = email.flatMap { line => line.split("\\s") }
```

我们正在创建一个 RDD，`words`，它知道如何从其父 RDD，`email`，构建自己，而 `email` 又知道它需要读取一个文本文件并将其拆分为行。然而，直到我们通过调用一个返回 Scala 对象的 *动作* 来强制评估 RDD，实际上没有任何命令发生。如果我们尝试从一个不存在的文本文件中读取，这一点尤为明显：

```py
scala> val inp = sc.textFile("nonexistent")
inp: rdd.RDD[String] = MapPartitionsRDD[5] at textFile

```

我们可以无障碍地创建 RDD。我们甚至可以在 RDD 上定义更多的转换。程序只有在这些转换最终评估时才会崩溃：

```py
scala> inp.count // number of lines
org.apache.hadoop.mapred.InvalidInputException: Input path does not exist: file:/Users/pascal/...

```

`.count` 动作预期返回我们 RDD 中元素的数量作为一个整数。Spark 除了评估 `inp` 没有其他选择，这会导致异常。

因此，可能更合适的是将 RDD 视为一个操作管道，而不是一个更传统的集合。

## RDD 知道它们的血缘关系

RDD 只能从稳定存储（例如，通过从 Spark 集群中每个节点都存在的文件加载数据）或通过基于其他 RDD 的转换集构建。由于 RDD 是惰性的，它们需要在需要时知道如何构建自身。它们通过知道自己的父 RDD 是谁以及需要应用于父 RDD 的操作来实现这一点。由于父 RDD 是不可变的，这是一个定义良好的过程。

`toDebugString` 方法提供了一个 RDD 构建过程的图示：

```py
scala> filteredWords.toDebugString
(2) MapPartitionsRDD[6] at filter at <console>:27 []
 |  MapPartitionsRDD[3] at flatMap at <console>:23 []
 |  MapPartitionsRDD[1] at textFile at <console>:21 []
 |  ham/9-463msg1.txt HadoopRDD[0] at textFile at <console>:21 []

```

## RDD 具有弹性

如果你在一个单机上运行应用程序，你通常不需要担心应用程序中的硬件故障：如果计算机失败，你的应用程序无论如何都是注定要失败的。

相比之下，分布式架构应该是容错的：单个机器的故障不应该导致整个应用程序崩溃。Spark RDD 是以容错性为设计理念的。让我们假设其中一个工作节点失败，导致与 RDD 相关的一些数据被破坏。由于 Spark RDD 知道如何从其父 RDD 构建自身，因此不会永久丢失数据：丢失的元素可以在需要时在另一台计算机上重新计算。

## RDD 是分布式的

当你从一个文本文件等构建 RDD 时，Spark 会将 RDD 拆分为多个分区。每个分区将完全本地化在单个机器上（尽管通常每台机器有多个分区）。

RDD 上的许多转换可以在每个分区独立执行。例如，当执行`.map`操作时，输出 RDD 中的给定元素依赖于父 RDD 中的单个元素：数据不需要在分区之间移动。`.flatMap`和`.filter`操作也是如此。这意味着由这些操作之一产生的 RDD 中的分区依赖于父 RDD 中的单个分区。

另一方面，`.distinct`转换，它从 RDD 中删除所有重复元素，需要将给定分区中的数据与每个其他分区中的数据进行比较。这需要在节点之间进行*洗牌*。洗牌，特别是对于大型数据集，是一个昂贵的操作，如果可能的话应该避免。

## RDD 上的转换和操作

RDD 支持的操作集可以分为两类：

+   **转换**从当前 RDD 创建一个新的 RDD。转换是惰性的：它们不会立即被评估。

+   **操作**强制评估 RDD，通常返回一个 Scala 对象，而不是 RDD，或者有一些形式的副作用。操作会立即评估，触发构成此 RDD 的所有转换的执行。

在下面的表格中，我们给出了一些有用的转换和操作的示例。对于完整和最新的列表，请参阅 Spark 文档([`spark.apache.org/docs/latest/programming-guide.html#rdd-operations`](http://spark.apache.org/docs/latest/programming-guide.html#rdd-operations))。

对于这些表格中的示例，我们假设你已经创建了一个包含以下内容的 RDD：

```py
scala> val rdd = sc.parallelize(List("quick", "brown", "quick", "dog"))

```

以下表格列出了 RDD 上的常见转换。请记住，转换始终生成一个新的 RDD，并且它们是惰性操作：

| 转换 | 注意事项 | **示例（假设** `rdd` **是** `{ "quick", "brown", "quick", "dog" }`） |
| --- | --- | --- |
| `rdd.map(func)` |   | `rdd.map { _.size } // => { 5, 5, 5, 3 }` |
| `rdd.filter(pred)` |   | `rdd.filter { _.length < 4 } // => { "dog" }` |
| `rdd.flatMap(func)` |   | `rdd.flatMap { _.toCharArray } // => { 'q', 'u', 'i', 'c', 'k', 'b', 'r', 'o' … }` |
| `rdd.distinct()` | 从 RDD 中移除重复元素。 | `rdd.distinct // => { "dog", "brown", "quick" }` |
| `rdd.pipe(command, [envVars])` | 通过外部程序进行管道传输。RDD 元素逐行写入进程的`stdin`。从`stdout`读取输出。 | `rdd.pipe("tr a-z A-Z") // => { "QUICK", "BROWN", "QUICK", "DOG" }` |

以下表格描述了 RDD 上的常见动作。请记住，动作始终生成 Scala 类型或引起副作用，而不是创建一个新的 RDD。动作强制评估 RDD，触发 RDD 下支撑的转换的执行。

| 动作 | 节点 | **示例（假设** `rdd` **是** `{ "quick", "brown", "quick", "dog" }`） |
| --- | --- | --- |
| `rdd.first` | RDD 中的第一个元素。 | `rdd.first // => quick` |
| `rdd.collect` | 将 RDD 转换为数组（数组必须在主节点上能够适应内存）。 | `rdd.collect // => ArrayString` |
| `rdd.count` | RDD 中的元素数量。 | `rdd.count // => 4` |
| `rdd.countByValue` | 元素到该元素出现次数的映射。映射必须在主节点上适应。 | `rdd.countByValue // => Map(quick -> 2, brown -> 1, dog -> 1)` |
| `rdd.take(n)` | 返回 RDD 中前*n*个元素的数组。 | `rdd.take(2) // => Array(quick, brown)` |
| `rdd.takeOrdered(n:Int)(implicit ordering: Ordering[T])` | 根据元素的默认排序或作为第二个参数传递的排序，按顺序获取 RDD 中的前*n*个元素。有关如何定义自定义比较函数的说明，请参阅 Scala 文档中的`Ordering`（[`www.scala-lang.org/api/current/index.html#scala.math.Ordering`](http://www.scala-lang.org/api/current/index.html#scala.math.Ordering)）。 | `rdd.takeOrdered(2) // => Array(brown, dog)``rdd.takeOrdered(2) (Ordering.by { _.size }) // => Array[String] = Array(dog, quick)` |
| `rdd.reduce(func)` | 根据指定的函数减少 RDD。使用 RDD 中的第一个元素作为基数。`func`应该是交换律和结合律的。 | `rdd.map { _.size }.reduce { _ + _ } // => 18` |
| `rdd.aggregate(zeroValue)(seqOp, combOp)` | 用于返回类型与 RDD 类型不同的值的减少情况。在这种情况下，我们需要提供一个用于单个分区内减少的函数（`seqOp`）和一个用于合并两个分区值的函数（`combOp`）。 | `rdd.aggregate(0) ( _ + _.size, _ + _ ) // => 18` |

## 持久化 RDDs

我们了解到 RDDs 只保留构建元素所需的操作序列，而不是元素本身的值。这当然大大减少了内存使用，因为我们不需要在内存中保留 RDDs 的中间版本。例如，假设我们想要遍历事务日志以识别特定账户上发生的所有交易：

```py
val allTransactions = sc.textFile("transaction.log")
val interestingTransactions = allTransactions.filter { 
  _.contains("Account: 123456")
}
```

所有交易的集合将很大，而感兴趣账户上的交易集合将小得多。Spark 记住如何构建数据集，而不是数据集本身的政策意味着我们任何时候都不需要在内存中保留所有输入文件的行。

有两种情况我们可能希望避免每次使用 RDD 时重新计算其元素：

+   对于交互式使用：我们可能已经检测到账户“123456”上的欺诈行为，并希望调查这种情况是如何发生的。我们可能需要在这个 RDD 上执行许多不同的探索性计算，而不必每次都重新读取整个日志文件。因此，持久化`interestingTransactions`是有意义的。

+   当算法重新使用中间结果或数据集时。一个典型的例子是逻辑回归。在逻辑回归中，我们通常使用迭代算法来找到最小化损失函数的“最优”系数。在我们迭代算法的每一步中，我们必须从训练集中计算损失函数及其梯度。如果可能的话，我们应该避免重新计算训练集（或从输入文件中重新加载它）。

Spark 在 RDD 上提供了一个`.persist`方法来实现这一点。通过在 RDD 上调用`.persist`，我们告诉 Spark 在下次计算时将数据集保留在内存中。

```py
scala> words.persist
rdd.RDD[String] = MapPartitionsRDD[3] at filter

```

Spark 支持不同的持久化级别，您可以通过传递参数给`.persist`来调整：

```py
scala> import org.apache.spark.storage.StorageLevel
import org.apache.spark.storage.StorageLevel

scala> interestingTransactions.persist(
 StorageLevel.MEMORY_AND_DISK)
rdd.RDD[String] = MapPartitionsRDD[3] at filter

```

Spark 提供了几个持久化级别，包括：

+   `MEMORY_ONLY`: 默认存储级别。RDD 存储在 RAM 中。如果 RDD 太大而无法适应内存，则其部分将不会持久化，并且需要即时重新计算。

+   `MEMORY_AND_DISK`: 尽可能多地存储 RDD 在内存中。如果 RDD 太大，它将溢出到磁盘。这只有在 RDD 计算成本很高的情况下才有意义。否则，重新计算它可能比从磁盘读取更快。

如果你持续持久化多个 RDD 并且内存不足，Spark 将会清除最近最少使用的 RDD（根据选择的持久化级别，可能是丢弃它们或将它们保存到磁盘）。RDD 还提供了一个`unpersist`方法，可以显式地告诉 Spark 一个 RDD 不再需要。

持久化 RDD 可能会对性能产生重大影响。因此，在调整 Spark 应用程序时，什么和如何持久化变得非常重要。找到最佳持久化级别通常需要一些调整、基准测试和实验。Spark 文档提供了何时使用哪种持久化级别的指南([`spark.apache.org/docs/latest/programming-guide.html#rdd-persistence`](http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence))，以及调整内存使用的通用技巧([`spark.apache.org/docs/latest/tuning.html`](http://spark.apache.org/docs/latest/tuning.html))。

重要的是，`persist`方法不会强制 RDD 进行评估。它只是通知 Spark 引擎，下次计算此 RDD 中的值时，应该保存而不是丢弃。

## 键值 RDD

到目前为止，我们只考虑了 Scala 值类型的 RDD。支持更复杂数据类型的 RDD 支持额外的操作。Spark 为*键值 RDD*添加了许多操作：类型参数为元组`(K, V)`的 RDD，对于任何类型`K`和`V`。

让我们回到我们的示例电子邮件：

```py
scala> val email = sc.textFile("ham/9-463msg1.txt")
email: rdd.RDD[String] = MapPartitionsRDD[1] at textFile

scala> val words = email.flatMap { line => line.split("\\s") }
words: rdd.RDD[String] = MapPartitionsRDD[2] at flatMap

```

让我们将`words` RDD 持久化到内存中，以避免反复从磁盘重新读取`email`文件：

```py
scala> words.persist

```

要访问键值操作，我们只需要对我们的 RDD 应用一个转换，创建键值对。现在，我们将使用单词作为键。对于每个值，我们将使用 1：

```py
scala> val wordsKeyValue = words.map { _ -> 1 }
wordsKeyValue: rdd.RDD[(String, Int)] = MapPartitionsRDD[32] at map 

scala> wordsKeyValue.first
(String, Int) = (Subject:,1)

```

键值 RDD 除了核心 RDD 操作外，还支持几个操作。这些操作通过隐式转换添加，使用我们在第五章中探讨的“pimp my library”模式，即*通过 JDBC 使用 Scala 和 SQL*。这些额外的转换分为两大类：*按键*转换和 RDD 之间的*连接*。

按键转换是聚合相同键对应值的操作。例如，我们可以使用`reduceByKey`来计算每个单词在电子邮件中出现的次数。此方法接受属于同一键的所有值，并使用用户提供的函数将它们组合起来：

```py
scala> val wordCounts = wordsKeyValue.reduceByKey { _ + _ }
wordCounts: rdd.RDD[(String, Int)] = ShuffledRDD[35] at reduceByKey

scala> wordCounts.take(5).foreach { println }
(university,6)
(under,1)
(call,3)
(paper,2)
(chasm,2)

```

注意，`reduceByKey`通常需要将 RDD 进行洗牌，因为给定键的每个出现可能不在同一个分区：

```py
scala> wordCounts.toDebugString
(2) ShuffledRDD[36] at reduceByKey at <console>:30 []
 +-(2) MapPartitionsRDD[32] at map at <console>:28 []
 |  MapPartitionsRDD[7] at flatMap at <console>:23 []
 |      CachedPartitions: 2; MemorySize: 50.3 KB; ExternalBlockStoreSize: 0.0 B; DiskSize: 0.0 B
 |  MapPartitionsRDD[3] at textFile at <console>:21 []
 |      CachedPartitions: 2; MemorySize: 5.1 KB; ExternalBlockStoreSize: 0.0 B; DiskSize: 0.0 B
 |  ham/9-463msg1.txt HadoopRDD[2] at textFile at <console>:21 []

```

注意，键值 RDD 与 Scala Maps 不同：相同的键可以出现多次，并且它们不支持*O(1)*查找。可以使用`.collectAsMap`操作将键值 RDD 转换成 Scala Map：

```py
scala> wordCounts.collectAsMap
scala.collection.Map[String,Int] = Map(follow -> 2, famous -> 1...

```

这需要将整个 RDD 拉到主 Spark 节点上。因此，您需要在主节点上拥有足够的内存来存放映射。这通常是管道中的最后一个阶段，用于过滤大型 RDD，只保留我们所需的信息。

下面表格中描述了许多按键操作。对于表格中的示例，我们假设 `rdd` 是以下方式创建的：

```py
scala> val words = sc.parallelize(List("quick", "brown","quick", "dog"))
words: RDD[String] = ParallelCollectionRDD[25] at parallelize at <console>:21

scala> val rdd = words.map { word => (word -> word.size) }
rdd: RDD[(String, Int)] = MapPartitionsRDD[26] at map at <console>:23

scala> rdd.collect
Array[(String, Int)] = Array((quick,5), (brown,5), (quick,5), (dog,3))

```

| 转换 | 备注 | **示例（假设** `rdd` **是** `{ quick -> 5, brown -> 5, quick -> 5, dog -> 3 }` **）** |
| --- | --- | --- |
| `rdd.mapValues` | 对值应用一个操作。 | `rdd.mapValues { _ * 2 } // => { quick -> 10, brown -> 10, quick -> 10, dog ->6 }` |
| `rdd.groupByKey` | 返回一个键值 RDD，其中对应相同键的值被分组到可迭代对象中。 | `rdd.groupByKey // => { quick -> Iterable(5, 5), brown -> Iterable(5), dog -> Iterable(3) }` |
| `rdd.reduceByKey(func)` | 返回一个键值 RDD，其中对应相同键的值使用用户提供的函数进行组合。 | `rdd.reduceByKey { _ + _ } // => { quick -> 10, brown -> 5, dog -> 3 }` |
| `rdd.keys` | 返回一个键的 RDD。 | `rdd.keys // => { quick, brown, quick, dog }` |
| `rdd.values` | 返回一个值的 RDD。 | `rdd.values // => { 5, 5, 5, 3 }` |

键值 RDD 上的第二种操作类型涉及通过键将不同的 RDD 连接在一起。这与 SQL 连接有些相似，其中键是要连接的列。让我们加载一封垃圾邮件，并应用我们对我们的正常邮件应用相同的转换：

```py
scala> val spamEmail = sc.textFile("spam/spmsgb17.txt")
spamEmail: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[52] at textFile at <console>:24

scala> val spamWords = spamEmail.flatMap { _.split("\\s") }
spamWords: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[53] at flatMap at <console>:26

scala> val spamWordCounts = spamWords.map { _ -> 1 }.reduceByKey { _ + _ }
spamWordsCount: org.apache.spark.rdd.RDD[(String, Int)] = ShuffledRDD[55] at reduceByKey at <console>:30

scala> spamWordCounts.take(5).foreach { println }
(banner,3)
(package,14)
(call,1)
(country,2)
(offer,1)

```

`spamWordCounts` 和 `wordCounts` 都是键值 RDD，其中键对应于消息中的唯一单词，值是该单词出现的次数。由于电子邮件将共享许多相同的单词，因此 `spamWordCounts` 和 `wordCounts` 之间的键将存在一些重叠。让我们在这两个 RDD 之间进行一个 *内部连接*，以获取两个电子邮件中都出现的单词：

```py
scala> val commonWordCounts = wordCounts.join(spamWordCounts)
res93: rdd.RDD[(String, (Int, Int))] = MapPartitionsRDD[58] at join at <console>:41

scala> commonWordCounts.take(5).foreach { println }
(call,(3,1))
(include,(6,2))
(minute,(2,1))
(form,(1,7))
((,(36,5))

```

内部连接产生的 RDD 中的值将是成对出现的。成对中的第一个元素是第一个 RDD 中该键的值，第二个元素是第二个 RDD 中该键的值。因此，单词 *call* 在合法电子邮件中出现了三次，在垃圾邮件中出现了两次。

Spark 支持所有四种连接类型。例如，让我们执行一个左连接：

```py
scala> val leftWordCounts = wordCounts.leftOuterJoin(spamWordCounts)
leftWordCounts: rdd.RDD[(String, (Int, Option[Int]))] = MapPartitionsRDD[64] at leftOuterJoin at <console>:40

scala> leftWordCounts.take(5).foreach { println }
(call,(3,Some(1)))
(paper,(2,None))
(chasm,(2,None))
(antonio,(1,None))
(event,(3,None))

```

注意，我们成对中的第二个元素具有 `Option[Int]` 类型，以适应 `spamWordCounts` 中缺失的键。例如，单词 *paper* 在合法电子邮件中出现了两次，在垃圾邮件中从未出现。在这种情况下，用零表示缺失比用 `None` 更有用。使用 `getOrElse` 替换 `None` 为默认值很简单：

```py
scala> val defaultWordCounts = leftWordCounts.mapValues { 
 case(leftValue, rightValue) => (leftValue, rightValue.getOrElse(0))
}
org.apache.spark.rdd.RDD[(String, (Int, Option[Int]))] = MapPartitionsRDD[64] at leftOuterJoin at <console>:40

scala> defaultwordCounts.take(5).foreach { println }
(call,(3,1))
(paper,(2,0))
(chasm,(2,0))
(antonio,(1,0))
(event,(3,0))

```

下表列出了键值 RDD 上最常见的连接：

| **转换** | **结果（假设** `rdd1` 是 `{ quick -> 1, brown -> 2, quick -> 3, dog -> 4 }` **和** `rdd2` **是** `{ quick -> 78, brown -> 79, fox -> 80 }` **）** |
| --- | --- |
| `rdd1.join(rdd2)` | `{ quick -> (1, 78), quick -> (3, 78), brown -> (2, 79) }` |
| `rdd1.leftOuterJoin(rdd2)` | `{ dog -> (4, None), quick -> (1, Some(78)), quick -> (3, Some(78)), brown -> (2, Some(79)) }` |
| `rdd1.rightOuterJoin(rdd2)` | `{ quick -> (Some(1), 78), quick -> (Some(3), 78), brown -> (Some(2), 79), fox -> (None, 80) }` |
| `rdd1.fullOuterJoin(rdd2)` | `{ dog -> (Some(4), None), quick -> (Some(1), Some(78)), quick -> (Some(3), Some(78)), brown -> (Some(2), Some(79)), fox -> (None, Some(80)) }` |

要获取完整的转换列表，请查阅 `PairRDDFunctions` 的 API 文档，[`spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.rdd.PairRDDFunctions`](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.rdd.PairRDDFunctions)。

## 双精度浮点 RDD

在前面的部分中，我们看到了 Spark 通过隐式转换向键值 RDD 添加了功能。同样，Spark 向 doubles 的 RDD 添加了统计功能。让我们提取火腿消息的单词频率，并将值从整数转换为双精度浮点数：

```py
scala> val counts = wordCounts.values.map { _.toDouble }
counts: rdd.RDD[Double] = MapPartitionsRDD[9] at map

```

然后，我们可以使用 `.stats` 动作来获取摘要统计信息：

```py
scala> counts.stats
org.apache.spark.util.StatCounter = (count: 397, mean: 2.365239, stdev: 5.740843, max: 72.000000, min: 1.000000)

```

因此，最常见的单词出现了 72 次。我们还可以使用 `.histogram` 动作来了解值的分布：

```py
scala> counts.histogram(5)
(Array(1.0, 15.2, 29.4, 43.6, 57.8, 72.0),Array(391, 1, 3, 1, 1))

```

`.histogram` 方法返回一个数组的对。第一个数组表示直方图桶的界限，第二个数组表示该桶中元素的数量。因此，有 `391` 个单词出现次数少于 `15.2` 次。单词的分布非常偏斜，以至于使用常规大小的桶并不合适。我们可以通过传递一个桶边界的数组到 `histogram` 方法来传递自定义的桶。例如，我们可能以对数方式分布桶：

```py
scala> counts.histogram(Array(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0))
res13: Array[Long] = Array(264, 94, 22, 11, 1, 4, 1)

```

# 构建和运行独立程序

到目前为止，我们主要通过 Spark shell 与 Spark 进行交互。在接下来的部分中，我们将构建一个独立的应用程序，并在本地或 EC2 集群上启动 Spark 程序。

## 本地运行 Spark 应用程序

第一步是编写 `build.sbt` 文件，就像运行标准的 Scala 脚本一样。我们下载的 Spark 二进制文件需要针对 Scala 2.10 运行（你需要从源代码编译 Spark 以运行针对 Scala 2.11 的版本。这并不困难，只需遵循 [`spark.apache.org/docs/latest/building-spark.html#building-for-scala-211`](http://spark.apache.org/docs/latest/building-spark.html#building-for-scala-211) 上的说明即可）。

```py
// build.sbt file

name := "spam_mi"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.4.1"
)
```

然后，我们运行 `sbt package` 来编译和构建程序的 jar 包。jar 包将在 `target/scala-2.10/` 目录下构建，并命名为 `spam_mi_2.10-0.1-SNAPSHOT.jar`。你可以使用本章提供的示例代码尝试这个操作。

然后，我们可以使用位于 Spark 安装目录 `bin/` 文件夹中的 `spark-submit` 脚本在本地上运行 jar 包：

```py
$ spark-submit target/scala-2.10/spam_mi_2.10-0.1-SNAPSHOT.jar
... runs the program
```

可以通过传递参数到 `spark-submit` 来控制分配给 Spark 的资源。使用 `spark-submit --help` 来查看完整的参数列表。

如果 Spark 程序有依赖项（例如，对其他 Maven 包的依赖），最简单的方法是使用 *SBT* *打包* 插件将它们捆绑到应用程序 jar 中。让我们假设我们的应用程序依赖于 breeze-viz。现在的 `build.sbt` 文件如下所示：

```py
// build.sbt

name := "spam_mi"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.2" % "provided",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2"
)
```

SBT 打包是一个 SBT 插件，它构建 *胖* jar：包含程序本身以及程序的所有依赖项的 jar。

注意，我们在依赖列表中将 Spark 标记为“提供”，这意味着 Spark 本身将不会包含在 jar 文件中（无论如何它都是由 Spark 环境提供的）。要包含 SBT 打包插件，请在 `project/` 目录下创建一个名为 `assembly.sbt` 的文件，并包含以下行：

```py
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.0")
```

您需要重新启动 SBT 以使更改生效。然后您可以使用 SBT 中的 `assembly` 命令创建打包 jar。这将在 `target/scala-2.10` 目录中创建一个名为 `spam_mi-assembly-0.1-SNAPSHOT.jar` 的 jar。您可以使用 `spark-submit` 运行此 jar。

## 减少日志输出和 Spark 配置

Spark 默认情况下非常详细。默认日志级别设置为 `INFO`。为了避免错过重要信息，将日志设置更改为 `WARN` 是有用的。要全局更改默认日志级别，请进入您安装 Spark 的目录中的 `conf` 目录。您应该会找到一个名为 `log4j.properties.template` 的文件。将此文件重命名为 `log4j.properties` 并查找以下行：

```py
log4j.rootCategory=INFO, console
```

将此行更改为：

```py
log4j.rootCategory=WARN, console
```

在该目录中还有其他几个配置文件，您可以使用它们来更改 Spark 的默认行为。有关配置选项的完整列表，请访问 [`spark.apache.org/docs/latest/configuration.html`](http://spark.apache.org/docs/latest/configuration.html)。

## 在 EC2 上运行 Spark 应用程序

在本地运行 Spark 对于测试很有用，但使用分布式框架的全部意义在于运行能够利用多台不同计算机能力的程序。我们可以在任何能够通过 HTTP 互相通信的计算机集上设置 Spark。通常，我们还需要设置一个分布式文件系统，如 HDFS，这样我们就可以在集群间共享输入文件。为了本例的目的，我们将在一个 Amazon EC2 集群上设置 Spark。

Spark 附带一个名为 `ec2/spark-ec2` 的 shell 脚本，用于设置 EC2 集群并安装 Spark。它还会安装 HDFS。您需要 Amazon Web Services (AWS) 的账户才能遵循这些示例 ([`aws.amazon.com`](https://aws.amazon.com))。您需要 AWS 访问密钥和秘密密钥，您可以通过 AWS 网络控制台的 **账户** / **安全凭证** / **访问凭证** 菜单访问它们。您需要通过环境变量将这些密钥提供给 `spark-ec2` 脚本。如下注入到您的当前会话中：

```py
$ export AWS_ACCESS_KEY_ID=ABCDEF...
$ export AWS_SECRET_ACCESS_KEY=2dEf...
```

您也可以将这些行写入您的 shell 配置脚本（例如`.bashrc`文件或等效文件），以避免每次运行`setup-ec2`脚本时都需要重新输入。我们已在第六章中讨论了环境变量，*Slick – SQL 的函数式接口*。

您还需要通过在 EC2 网页控制台中点击**密钥对**来创建一个密钥对，创建一个新的密钥对并下载证书文件。我将假设您将密钥对命名为`test_ec2`，证书文件为`test_ec2.pem`。请确保密钥对是在*N*. Virginia 区域创建的（通过在 EC2 管理控制台右上角选择正确的区域），以避免在本章的其余部分中显式指定区域。您需要将证书文件的访问权限设置为仅用户可读：

```py
$ chmod 400 test_ec2.pem
```

我们现在可以启动集群了。导航到`ec2`目录并运行：

```py
$ ./spark-ec2 -k test_ec2 -i ~/path/to/certificate/test_ec2.pem -s 2 launch test_cluster
```

这将创建一个名为`test_cluster`的集群，包含一个主节点和两个从节点。从节点的数量通过`-s`命令行参数设置。集群启动需要一段时间，但您可以通过查看 EC2 管理控制台中的**实例**窗口来验证实例是否正在正确启动。

设置脚本支持许多选项，用于自定义实例类型、硬盘数量等。您可以通过将`--help`命令行选项传递给`spark-ec2`来探索这些选项。

通过向`spark-ec2`脚本传递不同的命令，可以控制集群的生命周期，例如：

```py
# shut down 'test_cluster'
$ ./spark-ec2 stop test_cluster

# start 'test_cluster'
$ ./spark-ec2 -i test_ec2.pem start test_cluster

# destroy 'test_cluster'
$ ./spark-ec2 destroy test_cluster
```

关于在 EC2 上使用 Spark 的更多详细信息，请参阅官方文档：[`spark.apache.org/docs/latest/ec2-scripts.html#running-applications`](http://spark.apache.org/docs/latest/ec2-scripts.html#running-applications)。

# 垃圾邮件过滤

让我们将所学的一切用于实际，并为我们的垃圾邮件过滤器进行一些数据探索。我们将使用 Ling-Spam 电子邮件数据集：[`csmining.org/index.php/ling-spam-datasets.html`](http://csmining.org/index.php/ling-spam-datasets.html)。该数据集包含 2412 封正常邮件和 481 封垃圾邮件，所有邮件都是通过语言学邮件列表收到的。我们将提取最能说明邮件是垃圾邮件还是正常邮件的单词。

任何自然语言处理工作流程的第一步是去除停用词和词形还原。去除停用词包括过滤掉非常常见的词，如*the*、*this*等等。词形还原涉及将同一词的不同形式替换为规范形式：*colors*和*color*都会映射到*color*，而*organize*、*organizing*和*organizes*都会映射到*organize*。去除停用词和词形还原非常具有挑战性，超出了本书的范围（如果你需要去除停用词并对数据集进行词形还原，你应该使用斯坦福 NLP 工具包：[`nlp.stanford.edu/software/corenlp.shtml`](http://nlp.stanford.edu/software/corenlp.shtml)）。幸运的是，Ling-Spam 电子邮件数据集已经被清理和词形还原了（这就是为什么电子邮件中的文本看起来很奇怪）。

当我们构建垃圾邮件过滤器时，我们将使用电子邮件中特定词的存在作为我们模型的特征。我们将使用*词袋模型*方法：我们考虑电子邮件中哪些词出现，但不考虑词序。

直观上，在决定一封电子邮件是否为垃圾邮件时，一些词会比其他词更重要。例如，包含*language*的电子邮件很可能是 ham，因为邮件列表是用于语言学讨论的，而*language*是一个不太可能被垃圾邮件发送者使用的词。相反，那些两种消息类型都常见的词，例如*hello*，不太可能有很大的作用。

量化一个词在确定一条消息是否为垃圾邮件中的重要性的一种方法是通过**互信息**（**MI**）。互信息是在我们知道一条消息包含特定词的情况下，关于该消息是 ham 还是 spam 的信息增益。例如，特定电子邮件中存在*language*这一事实对于判断该电子邮件是垃圾邮件还是 ham 非常有信息量。同样，*dollar*这个词的存在也是有信息量的，因为它经常出现在垃圾邮件中，而很少出现在 ham 邮件中。相比之下，*morning*这个词的存在是无信息量的，因为它在垃圾邮件和 ham 邮件中出现的频率大致相同。电子邮件中特定词的存在与该电子邮件是垃圾邮件还是 ham 之间的互信息公式是：

![垃圾邮件过滤](img/image01196.jpeg)

其中![垃圾邮件过滤](img/image01197.jpeg)是电子邮件包含特定词和属于该类别（ham 或 spam）的联合概率，![垃圾邮件过滤](img/image01198.jpeg)是特定词出现在电子邮件中的概率，![垃圾邮件过滤](img/image01199.jpeg)是任何电子邮件属于该类别的概率。互信息通常用于决策树。

### 注意

互信息表达式的推导超出了本书的范围。感兴趣的读者可以参考*David MacKay*的杰出著作*信息论、推理和学习算法*，特别是*依赖随机变量*这一章。

我们计算互信息的关键组成部分是评估一个单词出现在垃圾邮件或垃圾邮件中的概率。给定我们的数据集，这个概率的最佳近似值是该单词出现的消息比例。因此，例如，如果 *language* 出现在 40% 的消息中，我们将假设该语言出现在任何消息中的概率 ![垃圾邮件过滤](img/image01200.jpeg) 为 0.4。同样，如果 40% 的消息是垃圾邮件，而 *language* 出现在这些垃圾邮件的 50% 中，我们将假设该语言出现在电子邮件中的概率，以及该电子邮件是垃圾邮件的概率 ![垃圾邮件过滤](img/image01201.jpeg)。

让我们编写一个 `wordFractionInFiles` 函数来计算给定语料库中每个单词出现的消息比例。我们的函数将接受一个参数，即一个路径，该路径使用 shell 通配符标识一组文件，例如 `ham/*`，并且它将返回一个键值 RDD，其中键是单词，值是该单词出现在这些文件中的概率。我们将该函数放入一个名为 `MutualInformation` 的对象中。

我们首先给出该函数的整个代码列表。如果一开始看不懂没关系：我们将在代码之后详细解释这些难点。你可能发现将这些命令在 shell 中输入是有用的，例如将 `fileGlob` 替换为 `"ham/*"`：

```py
// MutualInformation.scala
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

object MutualInformation extends App {

  def wordFractionInFiles(sc:SparkContext)(fileGlob:String)
  :(RDD[(String, Double)], Long) = {

    // A set of punctuation words that need to be filtered out.
    val wordsToOmit = SetString", 
      "(", "@", "/", "Subject:"
    )

    val messages = sc.wholeTextFiles(fileGlob)
    // wholeTextFiles generates a key-value RDD of 
    // file name -> file content

    val nMessages = messages.count()

    // Split the content of each message into a Set of unique
    // words in that message, and generate a new RDD mapping:
    // message -> word
    val message2Word = messages.flatMapValues {
      mailBody => mailBody.split("\\s").toSet
    }

    val message2FilteredWords = message2Word.filter { 
      case(email, word) => ! wordsToOmit(word) 
    }

    val word2Message = message2FilteredWords.map { _.swap }

    // word -> number of messages it appears in.
    val word2NumberMessages = word2Message.mapValues { 
      _ => 1 
    }.reduceByKey { _ + _ }

    // word -> fraction of messages it appears in
    val pPresent = word2NumberMessages.mapValues { 
      _ / nMessages.toDouble 
    }

    (pPresent, nMessages)
  }
}
```

让我们在 Spark shell 中玩这个函数。为了能够从 shell 中访问此函数，我们需要创建一个包含 `MutualInformation` 对象的 jar。编写一个类似于上一节中展示的 `build.sbt` 文件，并使用 `sbt package` 将代码打包成 jar。然后，使用以下命令打开 Spark shell：

```py
$ spark-shell --jars=target/scala-2.10/spam_mi_2.10-0.1-SNAPSHOT.jar

```

这将在类路径上打开一个带有我们新创建的 jar 的 Spark shell。让我们在 `ham` 邮件上运行我们的 `wordFractionInFiles` 方法：

```py
scala> import MutualInformation._
import MutualInformation._

scala> val (fractions, nMessages) = wordFractionInFiles(sc)("ham/*")
fractions: org.apache.spark.rdd.RDD[(String, Double)] = MapPartitionsRDD[13] at mapValues
nMessages: Long = 2412

```

让我们获取 `fractions` RDD 的快照：

```py
scala> fractions.take(5)
Array[(String, Double)] = Array((rule-base,0.002902155887230514), (reunion,4.1459369817578774E-4), (embarrasingly,4.1459369817578774E-4), (mller,8.291873963515755E-4), (sapore,4.1459369817578774E-4))

```

很想看到在垃圾邮件中出现频率最高的单词。我们可以使用 `.takeOrdered` 动作来获取 RDD 的顶部值，并使用自定义排序。`.takeOrdered` 的第二个参数期望是一个类型类 `Ordering[T]` 的实例，其中 `T` 是我们 RDD 的类型参数：在这种情况下是 `(String, Double)`。`Ordering[T]` 是一个具有单个 `compare(a:T, b:T)` 方法的特质，它描述了如何比较 `a` 和 `b`。创建 `Ordering[T]` 的最简单方法是通过伴随对象的 `by` 方法，该方法定义了一个用于比较 RDD 元素的关键字。

我们希望按值对我们的键值 RDD 进行排序，并且由于我们想要最常见的单词，而不是最不常见的，我们需要反转这种排序：

```py
scala> fractions.takeOrdered(5)(Ordering.by { - _._2 })
res0: Array[(String, Double)] = Array((language,0.6737147595356551), (university,0.6048922056384743), (linguistic,0.5149253731343284), (information,0.45480928689883915), ('s,0.4369817578772803))

```

毫不奇怪，`language` 出现在 67% 的垃圾邮件中，`university` 出现在 60% 的垃圾邮件中，等等。对垃圾邮件的类似调查显示感叹号字符 *!* 出现在 83% 的垃圾邮件中，*our* 出现在 61%，*free* 出现在 57%。

我们现在可以开始编写应用程序的主体，以计算每个单词与消息是否为垃圾邮件或正常邮件之间的互信息。我们将代码的主体放入`MutualInformation`对象中，该对象已经包含了`wordFractionInFiles`方法。

第一步是创建一个 Spark 上下文：

```py
// MutualInformation.scala
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

object MutualInformation extends App {

  def wordFractionInFiles(sc:SparkContext)(fileGlob:String)
  :(RDD[(String, Double)], Long) = {
    ...
  }

  val conf = new SparkConf().setAppName("lingSpam")
  val sc = new SparkContext(conf)

```

注意，当我们使用 Spark shell 时，我们不需要这样做，因为 shell 自带一个预构建的上下文绑定到变量`sc`。

我们现在可以计算在给定消息是*垃圾邮件*的情况下，包含特定单词的条件概率![垃圾邮件过滤](img/image01202.jpeg)。这仅仅是包含该单词的*垃圾邮件*语料库中的消息比例。这反过来又让我们可以推断包含特定单词且为*垃圾邮件*的联合概率![垃圾邮件过滤](img/image01203.jpeg)。我们将对所有四个类别的组合进行此操作：任何给定单词是否存在于消息中，以及该消息是否为垃圾邮件或正常邮件：

```py
    /* Conditional probabilities RDD:
       word -> P(present | spam) 
    */
    val (pPresentGivenSpam, nSpam) = wordFractionInFiles(sc)("spam/*")
    val pAbsentGivenSpam = pPresentGivenSpam.mapValues { 1.0 - _ }
    val (pPresentGivenHam, nHam) = wordFractionInFiles(sc)("ham/*")
    val pAbsentGivenHam = pPresentGivenHam.mapValues { 1.0 - _ }

    // pSpam is the fraction of spam messages
    val nMessages = nSpam + nHam
    val pSpam = nSpam / nMessages.toDouble

    // pHam is the fraction of ham messages
    val pHam = 1.0 - pSpam 

    /* pPresentAndSpam is a key-value RDD of joint probabilities
       word -> P(word present, spam) 
    */
    val pPresentAndSpam = pPresentGivenSpam.mapValues { 
      _ * pSpam 
    }
    val pPresentAndHam = pPresentGivenHam.mapValues { _ * pHam }
    val pAbsentAndSpam = pAbsentGivenSpam.mapValues { _ * pSpam }
    val pAbsentAndHam = pAbsentGivenHam.mapValues { _ * pHam }
```

我们将在计算中的几个地方重用这些 RDD，所以让我们告诉 Spark 将它们保存在内存中，以避免需要重新计算：

```py
    pPresentAndSpam.persist
    pPresentAndHam.persist
    pAbsentAndSpam.persist
    pAbsentAndHam.persist
```

我们现在需要计算单词存在的概率![垃圾邮件过滤](img/image01198.jpeg)。这仅仅是`pPresentAndSpam`和`pPresentAndHam`的和，对于每个单词。棘手的部分是并非所有单词都存在于正常邮件和垃圾邮件中。因此，我们必须对这些 RDD 执行全外连接。这将给出一个 RDD，将每个单词映射到一个`Option[Double]`值的对。对于在正常邮件或垃圾邮件中不存在的单词，我们必须使用默认值。一个合理的默认值是![垃圾邮件过滤](img/image01204.jpeg)对于垃圾邮件（更严格的方法是使用*加性平滑*）。这意味着如果语料库是两倍大，该单词将出现一次。

```py
    val pJoined = pPresentAndSpam.fullOuterJoin(pPresentAndHam)
    val pJoinedDefault = pJoined.mapValues {
      case (presentAndSpam, presentAndHam) => 
        (presentAndSpam.getOrElse(0.5/nSpam * pSpam), 
        presentAndHam.getOrElse(0.5/nHam * pHam))
    }
```

注意，我们也可以选择 0 作为默认值。这会使信息增益的计算变得有些复杂，因为我们不能对零值取对数，并且似乎不太可能一个特定的单词在电子邮件中出现的概率恰好为零。

我们现在可以构建一个 RDD，将单词映射到![垃圾邮件过滤](img/image01198.jpeg)，即单词存在于垃圾邮件或正常邮件中的概率：

```py
    val pPresent = pJoinedDefault.mapValues { 
      case(presentAndHam, presentAndSpam) => 
        presentAndHam + presentAndSpam 
    }
    pPresent.persist

    val pAbsent = pPresent.mapValues { 1.0 - _ }
    pAbsent.persist
```

我们现在拥有所有需要的 RDD 来计算单词在消息中存在与否与消息是否为正常邮件或垃圾邮件之间的互信息。我们需要使用前面概述的互信息方程将它们全部结合起来。

我们将首先定义一个辅助方法，该方法给定联合概率 RDD *P(X, Y)* 和边缘概率 *P(X)* 和 *P(Y)*，计算![垃圾邮件过滤](img/image01205.jpeg)。在这里，*P(X)* 例如可以是单词存在于消息中的概率![垃圾邮件过滤](img/image01198.jpeg)，而 *P(Y)* 将会是该消息是*垃圾邮件*的概率![垃圾邮件过滤](img/image01206.jpeg)：

```py
    def miTerm(
      pXYs:RDD[(String, Double)], 
      pXs:RDD[(String, Double)], 
      pY: Double,
      default: Double // for words absent in PXY
    ):RDD[(String, Double)] = 
      pXs.leftOuterJoin(pXYs).mapValues { 
        case (pX, Some(pXY)) => pXY * math.log(pXY/(pX*pY)) 
        case (pX, None) => default * math.log(default/(pX*pY))
    }
```

我们可以使用我们的函数来计算互信息总和中的四个项：

```py
    val miTerms = List(
      miTerm(pPresentAndSpam, pPresent, pSpam, 0.5/nSpam * pSpam),
      miTerm(pPresentAndHam, pPresent, pHam, 0.5/nHam * pHam),
      miTerm(pAbsentAndSpam, pAbsent, pSpam, 0.5/nSpam * pSpam),
      miTerm(pAbsentAndHam, pAbsent, pHam, 0.5/nHam * pHam)
    )
```

最后，我们只需要将这些四个项相加：

```py
    val mutualInformation = miTerms.reduce { 
      (term1, term2) => term1.join(term2).mapValues { 
         case (l, r) => l + r 
      } 
    }
```

RDD `mutualInformation` 是一个键值 RDD，将每个词映射到衡量该词在区分邮件是否为垃圾邮件或正常邮件时的信息量的度量。让我们打印出最能够表明邮件是否为正常邮件或垃圾邮件的二十个词：

```py
    mutualInformation.takeOrdered(20)(Ordering.by { - _._2 })
      .foreach { println }
```

让我们使用 `spark-submit` 来运行这个示例：

```py
$ sbt package
$ spark-submit target/scala-2.10/spam_mi_2.10-0.1-SNAPSHOT.jar
(!,0.1479941771292119)
(language,0.14574624861510874)
(remove,0.11380645864246142)
(free,0.1073496947123657)
(university,0.10695975885487692)
(money,0.07531772498093084)
(click,0.06887598051593441)
(our,0.058950906866052394)
(today,0.05485248095680509)
(sell,0.05385519653184113)
(english,0.053509319455430575)
(business,0.05299311289740539)
(market,0.05248394151802276)
(product,0.05096229706182162)
(million,0.050233193237964546)
(linguistics,0.04990172586630499)
(internet,0.04974101556655623)
(company,0.04941817269989519)
(%,0.04890193809823071)
(save,0.04861393414892205)
```

因此，我们发现像 `language` 或 `free` 或 `!` 这样的词包含最多的信息，因为它们几乎只存在于垃圾邮件或正常邮件中。一个非常简单的分类算法只需取前 10 个（按互信息排序）垃圾邮件词和前 10 个正常邮件词，然后查看一条消息是否包含更多垃圾邮件词或正常邮件词。我们将在第十二章分布式机器学习与 MLlib 中更深入地探讨用于分类的机器学习算法。

# 揭开盖子

在本章的最后部分，我们将非常简要地讨论 Spark 的内部工作原理。对于更详细的讨论，请参阅本章末尾的 *参考文献* 部分。

当你打开一个 Spark 上下文，无论是显式地还是通过启动 Spark shell，Spark 会启动一个包含当前任务和过去任务执行详情的 Web UI。让我们看看我们在上一节中编写的示例互信息程序的实际操作。为了防止程序完成后上下文关闭，你可以在 `main` 方法的最后（在调用 `takeOrdered` 之后）插入一个对 `readLine` 的调用。这会期望用户输入，因此程序执行将暂停，直到你按下 *enter* 键。

要访问 UI，将你的浏览器指向 `127.0.0.1:4040`。如果你有其他正在运行的 Spark shell 实例，端口可能是 `4041`、`4042` 等等。

![揭开盖子](img/image01207.jpeg)

UI 的第一页告诉我们，我们的应用程序包含三个 *作业*。作业是动作的结果。实际上，我们的应用程序中确实有三个动作：前两个是在 `wordFractionInFiles` 函数中调用的：

```py
val nMessages = messages.count()
```

最后一个作业是由对 `takeOrdered` 的调用产生的，它强制执行计算互信息的整个 RDD 转换管道。

Web UI 允许我们深入了解每个作业。点击作业表中的 `takeOrdered` 作业。你将被带到一页，其中更详细地描述了该作业：

![揭开盖子](img/image01208.jpeg)

特别值得注意的是 **DAG 可视化** 项。这是一个执行计划的图，用于满足动作，并提供了对 Spark 内部工作原理的洞察。

当你通过在 RDD 上调用一个动作来定义一个作业时，Spark 会查看 RDD 的 lineage 并构建一个映射依赖关系的图： lineage 中的每个 RDD 都由一个节点表示，从该 RDD 的父节点到自身的有向边。这种图称为**有向无环图**（DAG），是一种用于依赖关系解析的有用数据结构。让我们使用 web UI 来探索我们程序中`takeOrdered`作业的 DAG。这个图相当复杂，因此很容易迷路，所以这里有一个简化的复制品，它只列出了程序中绑定到变量名的 RDD。

![揭开盖子](img/image01209.jpeg)

如您所见，在图的底部，我们有`mutualInformation` RDD。这是我们为我们的动作需要构建的 RDD。这个 RDD 依赖于求和中的一些中间元素，例如`igFragment1`、`igFragment2`等。我们可以通过依赖关系列表回溯，直到达到图的另一端：不依赖于其他 RDD，只依赖于外部源的 RDD。

一旦构建了图，Spark 引擎就会制定一个执行作业的计划。计划从只有外部依赖（例如从磁盘加载文件或从数据库中检索而构建的 RDD）或已经缓存了数据的 RDD 开始。图上的每个箭头都被转换为一组*任务*，每个任务将一个转换应用于数据的一个分区。

任务被分组到*阶段*中。一个阶段由一组可以在不需要中间洗牌的情况下执行的任务组成。

# 数据洗牌和分区

要理解 Spark 中的数据洗牌，我们首先需要了解 RDD 中数据是如何分区的。当我们通过例如从 HDFS 加载文件或读取本地存储中的文件来创建一个 RDD 时，Spark 无法控制哪些数据位被分布在哪些分区中。这对于键值 RDD 来说是一个问题：这些 RDD 通常需要知道特定键的出现位置，例如执行连接操作。如果键可以在 RDD 的任何位置出现，我们必须查看每个分区以找到该键。

为了防止这种情况，Spark 允许在键值 RDD 上定义一个*分区器*。分区器是 RDD 的一个属性，它决定了特定键落在哪个分区。当一个 RDD 设置了分区器时，键的位置完全由分区器决定，而不是由 RDD 的历史或键的数量决定。具有相同分区器的两个不同的 RDD 将把相同的键映射到相同的分区。

分区通过它们对转换的影响来影响性能。在键值 RDD 上有两种类型的转换：

+   窄变换，例如`mapValues`。在窄变换中，用于计算子 RDD 中分区的数据位于父分区的一个分区。因此，窄变换的数据处理可以完全本地执行，无需在节点之间通信数据。

+   广泛变换，例如`reduceByKey`。在广泛变换中，用于计算任何单个分区的数据可以位于父分区中的所有分区。一般来说，由广泛变换产生的 RDD 将设置一个分区器。例如，`reduceByKey`变换的输出默认是哈希分区：特定键最终所在的分区由`hash(key) % numPartitions`确定。

因此，在我们的互信息示例中，`pPresentAndSpam`和`pPresentAndHam`将具有相同的分区结构，因为它们都有默认的哈希分区器。所有子 RDD 都保留相同的键，一直到`mutualInformation`。例如，单词`language`将在每个 RDD 中位于相同的分区。

为什么这些都很重要？如果一个 RDD 设置了分区器，那么这个分区器将保留在所有后续的窄变换中，这些变换源自该 RDD。让我们回到我们的互信息示例。`pPresentGivenHam`和`pPresentGivenSpam`这两个 RDD 都源自`reduceByKey`操作，并且它们都有字符串键。因此，它们都将有相同的哈希分区器（除非我们明确设置不同的分区器）。当我们构建`pPresentAndSpam`和`pPresentAndHam`时，这个分区器将被保留。当我们构建`pPresent`时，我们执行`pPresentAndSpam`和`pPresentAndHam`的完全外连接。由于这两个 RDD 有相同的分区器，子 RDD `pPresent`有窄依赖：我们只需将`pPresentAndSpam`的第一个分区与`pPresentAndHam`的第一个分区连接起来，将`pPresentAndSpam`的第二个分区与`pPresentAndHam`的第二个分区连接起来，依此类推，因为任何字符串键都会在两个 RDD 中被哈希到相同的分区。相比之下，如果没有分区器，我们就必须将`pPresentAndSpam`的每个分区的数据与`pPresentAndSpam`的每个分区连接起来。这将需要将数据发送到所有持有`pPresentAndSpam`的节点，这是一个耗时的操作。

由于广泛依赖关系，需要将数据发送到网络中以构建子 RDD 的过程称为*洗牌*。优化 Spark 程序的大部分艺术在于减少洗牌，并在必要时减少洗牌量。

# 摘要

在本章中，我们探讨了 Spark 的基础知识，并学习了如何构建和操作 RDD。在下一章中，我们将学习关于 Spark SQL 和 DataFrame 的知识，这是一组隐式转换，允许我们以类似于 pandas DataFrame 的方式操作 RDD，以及如何使用 Spark 与不同的数据源进行交互。

# 参考资料

+   *《学习 Spark》*，由 *Holden Karau*，*Andy Konwinski*，*Patrick Wendell* 和 *Matei Zaharia* 撰写，*O'Reilly* 出版，提供了比本章更全面的 Spark 介绍。我强烈推荐它。

+   如果你对学习信息理论感兴趣，我推荐 *David MacKay* 的书 *《信息理论、推理和学习算法》*。

+   *《信息检索》*，由 *Manning*，*Raghavan* 和 *Schütze* 撰写，描述了如何分析文本数据（包括词形还原和词干提取）。在线

+   在 Ling-Spam 数据集上，以及如何分析它：[`www.aueb.gr/users/ion/docs/ir_memory_based_antispam_filtering.pdf`](http://www.aueb.gr/users/ion/docs/ir_memory_based_antispam_filtering.pdf).

+   这篇博客文章更详细地探讨了 Spark Web UI。[`databricks.com/blog/2015/06/22/understanding-your-spark-application-through-visualization.html`](https://databricks.com/blog/2015/06/22/understanding-your-spark-application-through-visualization.html).

+   这篇由 *Sandy Ryza* 撰写的博客文章是关于 Spark 内部机制的两部分系列文章的第一部分，讨论了如何利用它们来提高性能：[`blog.cloudera.com/blog/2015/03/how-to-tune-your-apache-spark-jobs-part-1/`](http://blog.cloudera.com/blog/2015/03/how-to-tune-your-apache-spark-jobs-part-1/).

# 第十一章。Spark SQL 和 DataFrame

在上一章中，我们学习了如何使用 Spark 构建一个简单的分布式应用程序。我们使用的数据是以文本文件形式存储的一组电子邮件。

我们了解到 Spark 是围绕 **弹性分布式数据集**（**RDDs**）的概念构建的。我们探索了几种类型的 RDD：简单的字符串 RDD、键值 RDD 和双精度浮点数 RDD。在键值 RDD 和双精度浮点数 RDD 的情况下，Spark 通过隐式转换增加了比简单 RDD 更多的功能。还有一种重要的 RDD 类型我们尚未探索：**DataFrame**（之前称为 **SchemaRDD**）。DataFrame 允许操作比我们迄今为止探索的更复杂的对象。

DataFrame 是一种分布式表格数据结构，因此非常适合表示和操作结构化数据。在本章中，我们将首先通过 Spark shell 研究 DataFrame，然后使用上一章中介绍的 Ling-spam 电子邮件数据集，看看 DataFrame 如何集成到机器学习管道中。

# DataFrames – 快速入门

让我们从打开 Spark shell 开始：

```py
$ spark-shell

```

让我们假设我们对在患者群体上运行分析以估计他们的整体健康状况感兴趣。我们已经为每位患者测量了他们的身高、体重、年龄以及他们是否吸烟。

我们可能将每位患者的读数表示为一个案例类（你可能希望将其中一些内容写入文本编辑器，然后使用 `:paste` 命令将其粘贴到 Scala shell 中）：

```py
scala> case class PatientReadings(
 val patientId: Int,
 val heightCm: Int,
 val weightKg: Int,
 val age:Int, 
 val isSmoker:Boolean 
)
defined class PatientReadings

```

通常，我们会有成千上万的病人，可能存储在数据库或 CSV 文件中。我们将在本章后面讨论如何与外部源交互。现在，让我们直接在 shell 中硬编码一些读取值：

```py
scala> val readings = List(
 PatientReadings(1, 175, 72, 43, false),
 PatientReadings(2, 182, 78, 28, true),
 PatientReadings(3, 164, 61, 41, false),
 PatientReadings(4, 161, 62, 43, true)
)
List[PatientReadings] = List(...

```

我们可以通过使用 `sc.parallelize` 将 `readings` 转换为 RDD：

```py
scala> val readingsRDD = sc.parallelize(readings)
readingsRDD: RDD[PatientReadings] = ParallelCollectionRDD[0] at parallelize at <console>:25

```

注意，我们 RDD 的类型参数是 `PatientReadings`。让我们使用 `.toDF` 方法将 RDD 转换为 DataFrame：

```py
scala> val readingsDF = readingsRDD.toDF
readingsDF: sql.DataFrame = [patientId: int, heightCm: int, weightKg: int, age: int, isSmoker: boolean]

```

我们已经创建了一个 DataFrame，其中每一行对应于特定病人的读取值，列对应于不同的特征：

```py
scala> readingsDF.show
+---------+--------+--------+---+--------+
|patientId|heightCm|weightKg|age|isSmoker|
+---------+--------+--------+---+--------+
|        1|     175|      72| 43|   false|
|        2|     182|      78| 28|    true|
|        3|     164|      61| 41|   false|
|        4|     161|      62| 43|    true|
+---------+--------+--------+---+--------+

```

创建 DataFrame 最简单的方法是使用 RDD 上的 `toDF` 方法。我们可以将任何 `RDD[T]`（其中 `T` 是一个 case class 或一个元组）转换为 DataFrame。Spark 将将 case class 的每个属性映射到 DataFrame 中适当类型的列。它使用反射来发现属性的名字和类型。还有几种其他方法可以构建 DataFrame，无论是从 RDD 还是外部来源，我们将在本章后面探讨。

DataFrames 支持许多操作来操作行和列。例如，让我们添加一个用于 **体质指数**（**BMI**）的列。BMI 是一种常见的将 *身高* 和 *体重* 聚合起来以判断某人是否超重或体重不足的方法。BMI 的公式是：

![DataFrames – 快速入门](img/image01210.jpeg)

让我们先创建一个以米为单位的身高列：

```py
scala> val heightM = readingsDF("heightCm") / 100.0 
heightM: sql.Column = (heightCm / 100.0)

```

`heightM` 具有数据类型 `Column`，表示 DataFrame 中的数据列。列支持许多算术和比较运算符，这些运算符按元素方式应用于列（类似于在 第二章中遇到的 Breeze 向量，*使用 Breeze 操作数据*)。列上的操作是惰性的：当定义时，`heightM` 列实际上并没有计算。现在让我们定义一个 BMI 列：

```py
scala> val bmi = readingsDF("weightKg") / (heightM*heightM)
bmi: sql.Column = (weightKg / ((heightCm / 100.0) * (heightCm / 100.0)))

```

在我们的读取 DataFrame 中添加 `bmi` 列将很有用。由于 DataFrames，就像 RDDs 一样，是不可变的，我们必须定义一个新的 DataFrame，它与 `readingsDF` 完全相同，但增加了一个用于 BMI 的列。我们可以使用 `withColumn` 方法来实现，该方法接受新列的名称和一个 `Column` 实例作为参数：

```py
scala> val readingsWithBmiDF = readingsDF.withColumn("BMI", bmi)
readingsWithBmiDF: sql.DataFrame = [heightCm: int, weightKg: int, age: int, isSmoker: boolean, BMI: double]

```

我们迄今为止看到的所有操作都是 *转换*：它们定义了一个操作管道，创建新的 DataFrames。这些转换在我们调用 **动作**（如 `show`）时执行：

```py
scala> readingsWithBmiDF.show
+---------+--------+--------+---+--------+------------------+
|patientId|heightCm|weightKg|age|isSmoker|               BMI|
+---------+--------+--------+---+--------+------------------+
|        1|     175|      72| 43|   false|23.510204081632654|
|        2|     182|      78| 28|    true| 23.54788069073783|
|        3|     164|      61| 41|   false|22.679952409280194|
|        4|     161|      62| 43|    true|  23.9188302920412|
+---------+--------+--------+---+--------+------------------+

```

除了创建额外的列，DataFrames 还支持过滤满足特定谓词的行。例如，我们可以选择所有吸烟者：

```py
scala> readingsWithBmiDF.filter {
 readingsWithBmiDF("isSmoker") 
}.show
+---------+--------+--------+---+--------+-----------------+
|patientId|heightCm|weightKg|age|isSmoker|              BMI|
+---------+--------+--------+---+--------+-----------------+
|        2|     182|      78| 28|    true|23.54788069073783|
|        4|     161|      62| 43|    true| 23.9188302920412|
+---------+--------+--------+---+--------+-----------------+

```

或者，为了选择体重超过 70 公斤的人：

```py
scala> readingsWithBmiDF.filter { 
 readingsWithBmiDF("weightKg") > 70 
}.show
+---------+--------+--------+---+--------+------------------+
|patientId|heightCm|weightKg|age|isSmoker|               BMI|
+---------+--------+--------+---+--------+------------------+
|        1|     175|      72| 43|   false|23.510204081632654|
|        2|     182|      78| 28|    true| 23.54788069073783|
+---------+--------+--------+---+--------+------------------+

```

在表达式中重复 DataFrame 名称可能会变得繁琐。Spark 定义了操作符 `$` 来引用当前 DataFrame 中的列。因此，上面的 `filter` 表达式可以更简洁地写成：

```py
scala> readingsWithBmiDF.filter { $"weightKg" > 70 }.show
+---------+--------+--------+---+--------+------------------+
|patientId|heightCm|weightKg|age|isSmoker|               BMI|
+---------+--------+--------+---+--------+------------------+
|        1|     175|      72| 43|   false|23.510204081632654|
|        2|     182|      78| 28|    true| 23.54788069073783|
+---------+--------+--------+---+--------+------------------+

```

`.filter` 方法是重载的。它接受一个布尔值列，如上所述，或者一个标识当前 DataFrame 中布尔列的字符串。因此，为了过滤我们的 `readingsWithBmiDF` DataFrame 以子选择吸烟者，我们也可以使用以下方法：

```py
scala> readingsWithBmiDF.filter("isSmoker").show
+---------+--------+--------+---+--------+-----------------+
|patientId|heightCm|weightKg|age|isSmoker|              BMI|
+---------+--------+--------+---+--------+-----------------+
|        2|     182|      78| 28|    true|23.54788069073783|
|        4|     161|      62| 43|    true| 23.9188302920412|
+---------+--------+--------+---+--------+-----------------+

```

当比较相等时，你必须使用特殊的 *三重等于* 操作符来比较列：

```py
scala> readingsWithBmiDF.filter { $"age" === 28 }.show
+---------+--------+--------+---+--------+-----------------+
|patientId|heightCm|weightKg|age|isSmoker|              BMI|
+---------+--------+--------+---+--------+-----------------+
|        2|     182|      78| 28|    true|23.54788069073783|
+---------+--------+--------+---+--------+-----------------+

```

类似地，你必须使用 `!==` 来选择不等于某个值的行：

```py
scala> readingsWithBmiDF.filter { $"age" !== 28 }.show
+---------+--------+--------+---+--------+------------------+
|patientId|heightCm|weightKg|age|isSmoker|               BMI|
+---------+--------+--------+---+--------+------------------+
|        1|     175|      72| 43|   false|23.510204081632654|
|        3|     164|      61| 41|   false|22.679952409280194|
|        4|     161|      62| 43|    true|  23.9188302920412|
+---------+--------+--------+---+--------+------------------+

```

# 聚合操作

我们已经看到了如何将操作应用于 DataFrame 中的每一行以创建新列，以及如何使用过滤器从原始 DataFrame 中选择子集行来构建新的 DataFrame。DataFrame 的最后一系列操作是分组操作，相当于 SQL 中的 `GROUP BY` 语句。让我们计算吸烟者和非吸烟者的平均 BMI。我们必须首先告诉 Spark 按列（在这种情况下是 `isSmoker` 列）对 DataFrame 进行分组，然后应用聚合操作（在这种情况下是平均）以减少每个组：

```py
scala> val smokingDF = readingsWithBmiDF.groupBy(
 "isSmoker").agg(avg("BMI"))
smokingDF: org.apache.spark.sql.DataFrame = [isSmoker: boolean, AVG(BMI): double]

```

这已经创建了一个包含两列的新 DataFrame：分组列和我们要对其聚合的列。让我们展示这个 DataFrame：

```py
scala> smokingDF.show
+--------+------------------+
|isSmoker|          AVG(BMI)|
+--------+------------------+
|    true|23.733355491389517|
|   false|23.095078245456424|
+--------+------------------+

```

除了平均之外，还有几个操作符可以用于对每个组进行聚合。以下表格中概述了一些更重要的一些，但要获取完整列表，请参阅[`spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.functions$`](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.functions%24)的 *聚合函数* 部分：

| 操作符 | 备注 |
| --- | --- |
| `avg(column)` | 指定列的组平均值。 |
| `count(column)` | 在指定列中每个组中的元素数量。 |
| `countDistinct(column, ... )` | 每个组中不同元素的数量。这也可以接受多个列以返回跨多个列的唯一元素计数。 |
| `first(column), last(column)` | 每个组中的第一个/最后一个元素 |
| `max(column), min(column)` | 每个组中的最大/最小元素 |
| `sum(column)` | 每个组中值的总和 |

每个聚合操作符都接受列名，作为字符串，或者类型为 `Column` 的表达式。后者允许对复合表达式进行聚合。如果我们想得到样本中吸烟者和非吸烟者的平均身高（以米为单位），我们可以使用：

```py
scala> readingsDF.groupBy("isSmoker").agg { 
 avg($"heightCm"/100.0) 
}.show
+--------+-----------------------+
|isSmoker|AVG((heightCm / 100.0))|
+--------+-----------------------+
|    true|                  1.715|
|   false|     1.6949999999999998|
+--------+-----------------------+

```

我们还可以使用复合表达式来定义要分组的列。例如，为了计算每个 `age` 组中患者的数量，按十年递增，我们可以使用：

```py
scala> readingsDF.groupBy(floor($"age"/10)).agg(count("*")).show
+-----------------+--------+
|FLOOR((age / 10))|count(1)|
+-----------------+--------+
|              4.0|       3|
|              2.0|       1|
+-----------------+--------+

```

我们使用短横线 `"*"` 来表示对每一列的计数。

# 将 DataFrames 合并

到目前为止，我们只考虑了单个 DataFrame 上的操作。Spark 还提供了类似 SQL 的连接来组合 DataFrame。假设我们还有一个将患者 ID 映射到（收缩压）血压测量的 DataFrame。我们将假设我们有一个将患者 ID 映射到血压的列表对：

```py
scala> val bloodPressures = List((1 -> 110), (3 -> 100), (4 -> 125))
bloodPressures: List[(Int, Int)] = List((1,110), (3,100), (4,125))

scala> val bloodPressureRDD = sc.parallelize(bloodPressures)
res16: rdd.RDD[(Int, Int)] = ParallelCollectionRDD[74] at parallelize at <console>:24

```

我们可以从这个元组 RDD 构建一个 DataFrame。然而，与从案例类 RDD 构建 DataFrame 不同，Spark 无法推断列名。因此，我们必须将这些列名显式传递给 `.toDF`：

```py
scala> val bloodPressureDF = bloodPressureRDD.toDF(
 "patientId", "bloodPressure")
bloodPressureDF: DataFrame = [patientId: int, bloodPressure: int]

scala> bloodPressureDF.show
+---------+-------------+
|patientId|bloodPressure|
+---------+-------------+
|        1|          110|
|        3|          100|
|        4|          125|
+---------+-------------+

```

让我们将 `bloodPressureDF` 与 `readingsDF` 通过患者 ID 作为连接键进行连接：

```py
scala> readingsDF.join(bloodPressureDF, 
 readingsDF("patientId") === bloodPressureDF("patientId")
).show
+---------+--------+--------+---+--------+---------+-------------+
|patientId|heightCm|weightKg|age|isSmoker|patientId|bloodPressure|
+---------+--------+--------+---+--------+---------+-------------+
|        1|     175|      72| 43|   false|        1|          110|
|        3|     164|      61| 41|   false|        3|          100|
|        4|     161|      62| 43|    true|        4|          125|
+---------+--------+--------+---+--------+---------+-------------+

```

这执行了一个*内连接*：只有同时存在于两个 DataFrame 中的患者 ID 被包含在结果中。连接类型可以作为额外的参数传递给 `join`。例如，我们可以执行一个*左连接*：

```py
scala> readingsDF.join(bloodPressureDF,
 readingsDF("patientId") === bloodPressureDF("patientId"),
 "leftouter"
).show
+---------+--------+--------+---+--------+---------+-------------+
|patientId|heightCm|weightKg|age|isSmoker|patientId|bloodPressure|
+---------+--------+--------+---+--------+---------+-------------+
|        1|     175|      72| 43|   false|        1|          110|
|        2|     182|      78| 28|    true|     null|         null|
|        3|     164|      61| 41|   false|        3|          100|
|        4|     161|      62| 43|    true|        4|          125|
+---------+--------+--------+---+--------+---------+-------------+

```

可能的连接类型有 `inner`、`outer`、`leftouter`、`rightouter` 或 `leftsemi`。这些都应该很熟悉，除了 `leftsemi`，它对应于*左半连接*。这与内连接相同，但在连接后只保留左侧的列。因此，这是一种过滤 DataFrame 以找到存在于另一个 DataFrame 中的行的方法。

# 自定义 DataFrame 上的函数

到目前为止，我们只使用内置函数来操作 DataFrame 的列。虽然这些通常足够用，但我们有时需要更大的灵活性。Spark 允许我们通过**用户定义函数**（**UDFs**）将自定义转换应用于每一行。假设我们想使用我们在第二章中推导出的方程，即*使用 Breeze 操作数据*，来计算给定身高和体重的男性概率。我们计算出决策边界如下：

![自定义 DataFrame 上的函数](img/image01211.jpeg)

任何具有 *f > 0* 的人，在给定他们的身高、体重以及用于第二章，*使用 Breeze 操作数据*（该数据基于学生，因此不太可能代表整个人群）的训练集的情况下，比女性更有可能是男性。要将厘米单位的身高转换为归一化身高 *rescaledHeight*，我们可以使用以下公式：

![自定义 DataFrame 上的函数](img/image01212.jpeg)

同样，要将体重（以千克为单位）转换为归一化体重 *rescaledWeight*，我们可以使用以下公式：

![自定义 DataFrame 上的函数](img/image01213.jpeg)

从训练集中计算了 *height* 和 *weight* 的平均值和标准差。让我们编写一个 Scala 函数，该函数返回给定身高和体重的人更有可能是男性：

```py
scala> def likelyMale(height:Int, weight:Int):Boolean = {
 val rescaledHeight = (height - 171.0)/8.95
 val rescaledWeight = (weight - 65.7)/13.4
 -0.75 + 2.48*rescaledHeight + 2.23*rescaledWeight > 0
}

```

要在 Spark DataFrame 上使用此函数，我们需要将其注册为**用户定义函数**（**UDF**）。这将我们的函数，该函数接受整数参数，转换为接受列参数的函数：

```py
scala> val likelyMaleUdf = sqlContext.udf.register(
 "likelyMaleUdf", likelyMale _)
likelyMaleUdf: org.apache.spark.sql.UserDefinedFunction = UserDefinedFunction(<function2>,BooleanType,List())

```

要注册一个 UDF，我们必须能够访问一个 `sqlContext` 实例。SQL 上下文提供了 DataFrame 操作的入口点。Spark shell 在启动时创建一个 SQL 上下文，绑定到变量 `sqlContext`，并在 shell 会话关闭时销毁它。

传递给 `register` 函数的第一个参数是 UDF 的名称（我们将在以后编写 DataFrame 上的 SQL 语句时使用 UDF 名称，但现在你可以忽略它）。然后我们可以像使用 Spark 中包含的内置转换一样使用 UDF：

```py
scala> val likelyMaleColumn = likelyMaleUdf(
 readingsDF("heightCm"), readingsDF("weightKg"))
likelyMaleColumn: org.apache.spark.sql.Column = UDF(heightCm,weightKg)

scala> readingsDF.withColumn("likelyMale", likelyMaleColumn).show
+---------+--------+--------+---+--------+----------+
|patientId|heightCm|weightKg|age|isSmoker|likelyMale|
+---------+--------+--------+---+--------+----------+
|        1|     175|      72| 43|   false|      true|
|        2|     182|      78| 28|    true|      true|
|        3|     164|      61| 41|   false|     false|
|        4|     161|      62| 43|    true|     false|
+---------+--------+--------+---+--------+----------+

```

正如你所见，Spark 将 UDF 的底层函数应用于 DataFrame 中的每一行。我们不仅限于使用 UDF 创建新列。我们还可以在 `filter` 表达式中使用它们。例如，为了选择可能对应女性的行：

```py
scala> readingsDF.filter(
 ! likelyMaleUdf($"heightCm", $"weightKg")
).show
+---------+--------+--------+---+--------+
|patientId|heightCm|weightKg|age|isSmoker|
+---------+--------+--------+---+--------+
|        3|     164|      61| 41|   false|
|        4|     161|      62| 43|    true|
+---------+--------+--------+---+--------+

```

使用 UDF 允许我们定义任意的 Scala 函数来转换行，为数据处理提供了巨大的额外功能。

# DataFrame 的不可变性和持久性

与 RDD 一样，DataFrame 是不可变的。当你对一个 DataFrame 定义一个转换时，这总是创建一个新的 DataFrame。原始 DataFrame 不能就地修改（这与 pandas DataFrame 明显不同，例如）。

DataFrame 上的操作可以分为两类：*转换*，它导致创建一个新的 DataFrame，和*动作*，它通常返回一个 Scala 类型或有一个副作用。例如 `filter` 或 `withColumn` 是转换，而 `show` 或 `head` 是动作。

转换是惰性的，就像 RDD 上的转换一样。当你通过转换现有的 DataFrame 生成一个新的 DataFrame 时，这会导致创建新 DataFrame 的执行计划的详细阐述，但数据本身并不会立即被转换。你可以使用 `queryExecution` 方法访问执行计划。

当你在 DataFrame 上调用一个动作时，Spark 会像处理一个常规 RDD 一样处理该动作：它隐式地构建一个无环图来解析依赖关系，处理构建被调用动作的 DataFrame 所需要的转换。

与 RDD 类似，我们可以在内存或磁盘上持久化 DataFrame：

```py
scala> readingsDF.persist
readingsDF.type = [patientId: int, heightCm: int,...]

```

这与持久化 RDD 的方式相同：下次计算 RDD 时，它将被保留在内存中（前提是有足够的空间），而不是被丢弃。持久化级别也可以设置：

```py
scala> import org.apache.spark.storage.StorageLevel
import org.apache.spark.storage.StorageLevel

scala> readingsDF.persist(StorageLevel.MEMORY_AND_DISK)
readingsDF.type = [patientId: int, heightCm: int, ...]

```

# DataFrame 上的 SQL 语句

到现在为止，你可能已经注意到 DataFrame 上的许多操作都是受 SQL 操作启发的。此外，Spark 允许我们将 DataFrame 注册为表，并直接使用 SQL 语句查询它们。因此，我们可以将临时数据库作为程序流程的一部分构建。

让我们将 `readingsDF` 注册为临时表：

```py
scala> readingsDF.registerTempTable("readings")

```

这注册了一个临时表，该表可以在 SQL 查询中使用。注册临时表依赖于 SQL 上下文的存在。当 SQL 上下文被销毁时（例如，当我们关闭 shell 时），临时表将被销毁。

让我们探索我们可以使用我们的临时表和 SQL 上下文做什么。我们首先可以获取上下文中当前注册的所有表的列表：

```py
scala> sqlContext.tables
DataFrame = [tableName: string, isTemporary: boolean]

```

这将返回一个 DataFrame。一般来说，所有在 SQL 上下文中返回数据的操作都会返回 DataFrames：

```py
scala> sqlContext.tables.show
+---------+-----------+
|tableName|isTemporary|
+---------+-----------+
| readings|       true|
+---------+-----------+

```

我们可以通过向 SQL 上下文中传递 SQL 语句来查询这个表：

```py
scala> sqlContext.sql("SELECT * FROM readings").show
+---------+--------+--------+---+--------+
|patientId|heightCm|weightKg|age|isSmoker|
+---------+--------+--------+---+--------+
|        1|     175|      72| 43|   false|
|        2|     182|      78| 28|    true|
|        3|     164|      61| 41|   false|
|        4|     161|      62| 43|    true|
+---------+--------+--------+---+--------+

```

在 `sqlContext` 中注册的任何 UDF 都可以通过它们注册时给出的名称访问。因此，我们可以在 SQL 查询中使用它们：

```py
scala> sqlContext.sql("""
SELECT 
 patientId, 
 likelyMaleUdf(heightCm, weightKg) AS likelyMale
FROM readings
""").show
+---------+----------+
|patientId|likelyMale|
+---------+----------+
|        1|      true|
|        2|      true|
|        3|     false|
|        4|     false|
+---------+----------+

```

你可能会想知道为什么有人想要将 DataFrames 注册为临时表并在这些表上运行 SQL 查询，当同样的功能可以直接在 DataFrames 上使用时。主要原因是为了与外部工具交互。Spark 可以运行一个 SQL 引擎，该引擎公开 JDBC 接口，这意味着知道如何与 SQL 数据库交互的程序将能够使用临时表。

我们没有足够的空间在这本书中介绍如何设置分布式 SQL 引擎，但您可以在 Spark 文档中找到详细信息（[`spark.apache.org/docs/latest/sql-programming-guide.html#distributed-sql-engine`](http://spark.apache.org/docs/latest/sql-programming-guide.html#distributed-sql-engine)）。

# 复杂数据类型 - 数组、映射和 struct

到目前为止，我们 DataFrame 中的所有元素都是简单类型。DataFrames 支持三种额外的集合类型：数组、映射和 struct。

## Structs

我们将要查看的第一个复合类型是 **struct**。一个 struct 类似于 case class：它存储一组键值对，具有一组固定的键。如果我们将包含嵌套 case class 的 case class RDD 转换为 DataFrame，Spark 将将嵌套对象转换为 struct。

让我们想象一下，我们想要序列化《指环王》中的角色。我们可能会使用以下对象模型：

```py
case class Weapon(name:String, weaponType:String)
case class LotrCharacter(name:String, val weapon:Weapon)
```

我们想要创建一个 `LotrCharacter` 实例的 DataFrame。让我们创建一些虚拟数据：

```py
scala> val characters = List(
 LotrCharacter("Gandalf", Weapon("Glamdring", "sword")),
 LotrCharacter("Frodo", Weapon("Sting", "dagger")),
 LotrCharacter("Aragorn", Weapon("Anduril", "sword"))
)
characters: List[LotrCharacter] = List(LotrCharacter...

scala> val charactersDF = sc.parallelize(characters).toDF
charactersDF: DataFrame = [name: string, weapon: struct<name:string,weaponType:string>]

scala> charactersDF.printSchema
root
 |-- name: string (nullable = true)
 |-- weapon: struct (nullable = true)
 |    |-- name: string (nullable = true)
 |    |-- weaponType: string (nullable = true)

scala> charactersDF.show
+-------+-----------------+
|   name|           weapon|
+-------+-----------------+
|Gandalf|[Glamdring,sword]|
|  Frodo|   [Sting,dagger]|
|Aragorn|  [Anduril,sword]|
+-------+-----------------+

```

在 case class 中的 `weapon` 属性在 DataFrame 中被转换为 struct 列。要从 struct 中提取子字段，我们可以将字段名传递给列的 `.apply` 方法：

```py
scala> val weaponTypeColumn = charactersDF("weapon")("weaponType")
weaponTypeColumn: org.apache.spark.sql.Column = weapon[weaponType]

```

我们可以使用这个派生列就像我们使用任何其他列一样。例如，让我们过滤我们的 DataFrame，只包含挥舞着剑的角色：

```py
scala> charactersDF.filter { weaponTypeColumn === "sword" }.show
+-------+-----------------+
|   name|           weapon|
+-------+-----------------+
|Gandalf|[Glamdring,sword]|
|Aragorn|  [Anduril,sword]|
+-------+-----------------+

```

## Arrays

让我们回到之前的例子，并假设除了身高、体重和年龄测量值之外，我们还有我们患者的电话号码。每个患者可能有零个、一个或多个电话号码。我们将定义一个新的 case class 和新的虚拟数据：

```py
scala> case class PatientNumbers(
 patientId:Int, phoneNumbers:List[String])
defined class PatientNumbers

scala> val numbers = List(
 PatientNumbers(1, List("07929123456")),
 PatientNumbers(2, List("07929432167", "07929234578")),
 PatientNumbers(3, List.empty),
 PatientNumbers(4, List("07927357862"))
)

scala> val numbersDF = sc.parallelize(numbers).toDF
numbersDF: org.apache.spark.sql.DataFrame = [patientId: int, phoneNumbers: array<string>]

```

在我们的 case class 中，`List[String]` 数组被转换为 `array<string>` 数据类型：

```py
scala> numbersDF.printSchema
root
 |-- patientId: integer (nullable = false)
 |-- phoneNumbers: array (nullable = true)
 |    |-- element: string (containsNull = true)

```

与 structs 类似，我们可以为数组中的特定索引构造一个列。例如，我们可以选择每个数组中的第一个元素：

```py
scala> val bestNumberColumn = numbersDF("phoneNumbers")(0)
bestNumberColumn: org.apache.spark.sql.Column = phoneNumbers[0]

scala> numbersDF.withColumn("bestNumber", bestNumberColumn).show
+---------+--------------------+-----------+
|patientId|        phoneNumbers| bestNumber|
+---------+--------------------+-----------+
|        1|   List(07929123456)|07929123456|
|        2|List(07929432167,...|07929432167|
|        3|              List()|       null|
|        4|   List(07927357862)|07927357862|
+---------+--------------------+-----------+

```

## Maps

最后的复合数据类型是映射。映射在存储键值对方面与 structs 类似，但 DataFrame 创建时键的集合不是固定的。因此，它们可以存储任意键值对。

当构建 DataFrame 时，Scala 映射将被转换为 DataFrame 映射。然后可以以类似结构体的方式查询它们。

# 与数据源交互

在数据科学或工程中，一个主要挑战是处理用于持久化数据的丰富输入和输出格式。我们可能以 CSV 文件、JSON 文件或通过 SQL 数据库的形式接收或发送数据，仅举几例。

Spark 提供了一个统一的 API，用于将 DataFrame 序列化和反序列化到不同的数据源。

## JSON 文件

Spark 支持从 JSON 文件加载数据，前提是 JSON 文件中的每一行都对应一个 JSON 对象。每个对象将被映射到 DataFrame 行。JSON 数组被映射到数组，嵌套对象被映射到结构体。

如果没有一些数据，本节可能会显得有些枯燥，所以让我们从 GitHub API 生成一些数据。不幸的是，GitHub API 并不返回每行一个 JSON 格式的对象。本章的代码库包含一个名为 `FetchData.scala` 的脚本，该脚本将下载并格式化 Martin Odersky 的存储库的 JSON 条目，并将对象保存到名为 `odersky_repos.json` 的文件中（如果你想的话，请更改 `FetchData.scala` 中的 GitHub 用户）。你也可以从 [data.scala4datascience.com/odersky_repos.json](http://data.scala4datascience.com/odersky_repos.json) 下载预先构建的数据文件。

让我们进入 Spark shell 并将此数据加载到 DataFrame 中。从 JSON 文件读取就像将文件名传递给 `sqlContext.read.json` 方法一样简单：

```py
scala> val df = sqlContext.read.json("odersky_repos.json")
df: DataFrame = [archive_url: string, assignees_url: ...]

```

从 JSON 文件读取数据时，数据被加载为 DataFrame。Spark 会自动从 JSON 文档中推断模式。我们的 DataFrame 中有许多列。让我们子选择一些列以获得更易于管理的 DataFrame：

```py
scala> val reposDF = df.select("name", "language", "fork", "owner")
reposDF: DataFrame = [name: string, language: string, ...] 

scala> reposDF.show
+----------------+----------+-----+--------------------+
|            name|  language| fork|               owner|
+----------------+----------+-----+--------------------+
|           dotty|     Scala| true|[https://avatars....|
|        frontend|JavaScript| true|[https://avatars....|
|           scala|     Scala| true|[https://avatars....|
|      scala-dist|     Scala| true|[https://avatars....|
|scala.github.com|JavaScript| true|[https://avatars....|
|          scalax|     Scala|false|[https://avatars....|
|            sips|       CSS|false|[https://avatars....|
+----------------+----------+-----+--------------------+

```

让我们将 DataFrame 保存回 JSON：

```py
scala> reposDF.write.json("repos_short.json")

```

如果你查看运行 Spark shell 的目录中的文件，你会注意到一个 `repos_short.json` 目录。在里面，你会看到名为 `part-000000`、`part-000001` 等的文件。当序列化 JSON 时，DataFrame 的每个分区都是独立序列化的。如果你在多台机器上运行此操作，你将在每台计算机上找到序列化输出的部分。

你可以选择性地传递一个 `mode` 参数来控制 Spark 如何处理现有的 `repos_short.json` 文件：

```py
scala> import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SaveMode

scala> reposDF.write.mode(
 SaveMode.Overwrite).json("repos_short.json")

```

可用的保存模式有 `ErrorIfExists`、`Append`（仅适用于 Parquet 文件）、`Overwrite` 和 `Ignore`（如果文件已存在则不保存）。

## Parquet 文件

Apache Parquet 是一种流行的文件格式，非常适合存储表格数据。它常用于 Hadoop 生态系统中的序列化，因为它允许在不读取整个文件的情况下高效地提取特定列和行。

Parquet 文件的序列化和反序列化与 JSON 相同，只需将 `json` 替换为 `parquet`：

```py
scala> reposDF.write.parquet("repos_short.parquet")

scala> val newDF = sqlContext.read.parquet("repos_short.parquet")
newDF: DataFrame = [name: string, language: string, fo...]

scala> newDF.show
+----------------+----------+-----+--------------------+
|            name|  language| fork|               owner|
+----------------+----------+-----+--------------------+
|           dotty|     Scala| true|[`avatars....|
|        frontend|JavaScript| true|[https://avatars....|
|           scala|     Scala| true|[https://avatars....|
|      scala-dist|     Scala| true|[https://avatars....|
|scala.github.com|JavaScript| true|[https://avatars....|
|          scalax|     Scala|false|[https://avatars....|
|            sips|       CSS|false|[https://avatars....|
+----------------+----------+-----+--------------------+

```

通常，Parquet 在存储大量对象集合时比 JSON 更节省空间。如果可以从行中推断出分区，Parquet 在检索特定列或行时也更为高效。因此，除非您需要输出可由外部程序读取或反序列化，否则 Parquet 相对于 JSON 具有优势。

# 独立程序

到目前为止，我们一直通过 Spark shell 使用 Spark SQL 和 DataFrame。要在独立程序中使用它，您需要从 Spark 上下文中显式创建它：

```py
val conf = new SparkConf().setAppName("applicationName")
val sc = new SparkContext(conf)
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
```

此外，导入嵌套在`sqlContext`中的`implicits`对象允许将 RDD 转换为 DataFrame：

```py
import sqlContext.implicits._

```

在下一章中，我们将广泛使用 DataFrame 来操纵数据，使其准备好与 MLlib 一起使用。

# 摘要

在本章中，我们探讨了 Spark SQL 和 DataFrame。DataFrame 在 Spark 的核心引擎之上增加了一层丰富的抽象层，极大地简化了表格数据的操作。此外，源 API 允许从丰富的数据文件中序列化和反序列化 DataFrame。

在下一章中，我们将基于我们对 Spark 和 DataFrame 的知识来构建一个使用 MLlib 的垃圾邮件过滤器。

# 参考文献

DataFrame 是 Spark 相对较新的功能。因此，相关的文献和文档仍然很少。首先应该查阅 Scala 文档，可在以下网址找到：[http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.DataFrame`](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.DataFrame)。

DataFrame `Column`类型上可用的操作 Scaladocs 可在以下网址找到：[`spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Column`](http://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Column)。

关于 Parquet 文件格式的详细文档也可在以下网址找到：[`parquet.apache.org`](https://parquet.apache.org)。

# 第十二章。使用 MLlib 进行分布式机器学习

机器学习描述了构建从数据中进行预测的算法。它是大多数数据科学流程的核心组件，通常被认为是增加最大价值的组件：机器学习算法的准确性决定了数据科学项目的成功。它还可能是数据科学流程中需要从软件工程以外的领域获取最多知识的部分：机器学习专家不仅熟悉算法，还熟悉统计学和业务领域。

选择和调整机器学习算法来解决特定问题涉及大量的探索性分析，以尝试确定哪些特征是相关的，特征之间的相关性如何，数据集中是否存在异常值等等。设计合适的机器学习管道是困难的。再加上数据集的大小和可扩展性需求带来的额外复杂性，您就面临了一个真正的挑战。

**MLlib** 有助于缓解这种困难。MLlib 是 Spark 的一个组件，它提供了核心 Spark 库之上的机器学习算法。它提供了一套学习算法，这些算法在分布式数据集上并行化效果很好。

MLlib 已经发展成两个独立的层。MLlib 本身包含核心算法，而 **ml**，也称为 *pipeline API*，定义了一个用于将算法粘合在一起的 API，并提供了一个更高层次的抽象。这两个库在它们操作的数据类型上有所不同：原始的 MLlib 在 DataFrame 引入之前就已经存在，主要作用于特征向量的 RDD。pipeline API 在 DataFrame 上操作。

在本章中，我们将研究较新的 pipeline API，只有在管道 API 缺少功能时才会深入研究 MLlib。

本章并不试图教授我们所展示算法背后的机器学习基础。我们假设读者对机器学习工具和技术有足够的了解，至少能够表面上理解这里展示的算法做什么，我们将深入解释统计学习机制的部分工作留给更好的作者（我们在本章末尾提供了一些参考文献）。

MLlib 是一个快速发展的丰富库。本章的目标不是提供一个完整的库概述。我们将通过构建一个用于训练垃圾邮件过滤器的机器学习管道来工作，在这个过程中了解我们需要的 MLlib 的各个部分。阅读完本章后，您将了解库的不同部分是如何结合在一起的，并且可以使用在线文档或更专业的书籍（请参阅本章末尾的参考文献）来了解这里未涵盖的 MLlib 部分。

# 介绍 MLlib – 垃圾邮件分类

让我们用一个具体的例子来介绍 MLlib。我们将查看使用我们在第十章分布式批量处理与 Spark 中使用的 Ling-Spam 数据集进行的垃圾邮件分类，我们将创建一个使用逻辑回归来估计给定消息是否为垃圾邮件的概率的垃圾邮件过滤器。

我们将通过 Spark shell 运行示例，但你将在本章的示例中找到类似程序，在`LogisticRegressionDemo.scala`中。如果你还没有安装 Spark，请参考第十章分布式批量处理 Spark，获取安装说明。

让我们从加载 Ling-Spam 数据集中的电子邮件开始。如果你在第十章分布式批量处理 Spark 中没有这样做，请从[data.scala4datascience.com/ling-spam.tar.gz](http://data.scala4datascience.com/ling-spam.tar.gz)或[data.scala4datascience.com/ling-spam.zip](http://data.scala4datascience.com/ling-spam.zip)下载数据，根据你想要`tar.gz`文件还是`zip`文件来选择，然后解压存档。这将创建一个`spam`目录和一个`ham`目录，分别包含垃圾邮件和正常邮件。

让我们使用`wholeTextFiles`方法来加载垃圾邮件和正常邮件：

```py
scala> val spamText = sc.wholeTextFiles("spam/*")
spamText: RDD[(String, String)] = spam/...

scala> val hamText = sc.wholeTextFiles("ham/*")
hamText: RDD[(String, String)] = ham/...

```

`wholeTextFiles`方法创建一个键值 RDD，其中键是文件名，值是文件内容：

```py
scala> spamText.first
(String, String) =
(file:spam/spmsga1.txt,"Subject: great part-time summer job! ...")

scala> spamText.count
Long = 481

```

管道 API 中的算法在 DataFrame 上工作。因此，我们必须将我们的键值 RDD 转换为 DataFrame。我们定义一个新的 case class，`LabelledDocument`，它包含一个消息文本和一个类别标签，用于标识消息是`spam`还是`ham`：

```py
scala> case class LabelledDocument(
 fileName:String, 
 text:String, 
 category:String
)
defined class LabelledDocument

scala> val spamDocuments = spamText.map {
 case (fileName, text) => 
 LabelledDocument(fileName, text, "spam")
}
spamDocuments: RDD[LabelledDocument] = MapPartitionsRDD[2] at map

scala> val hamDocuments = hamText.map {
 case (fileName, text) => 
 LabelledDocument(fileName, text, "ham")
}
hamDocuments: RDD[LabelledDocument] = MapPartitionsRDD[3] at map

```

要创建模型，我们需要将所有文档放入一个 DataFrame 中。因此，我们将两个`LabelledDocument` RDD 合并，并将其转换为 DataFrame。`union`方法将 RDD 连接起来：

```py
scala> val allDocuments = spamDocuments.union(hamDocuments)
allDocuments: RDD[LabelledDocument] = UnionRDD[4] at union

scala> val documentsDF = allDocuments.toDF
documentsDF: DataFrame = [fileName: string, text: string, category: string]

```

让我们做一些基本的检查来验证我们已经加载了所有文档。我们首先将 DataFrame 保存在内存中，以避免需要从原始文本文件中重新创建它。

```py
scala> documentsDF.persist
documentsDF.type = [fileName: string, text: string, category: string]

scala> documentsDF.show
+--------------------+--------------------+--------+
|            fileName|                text|category|
+--------------------+--------------------+--------+
|file:/Users/pasca...|Subject: great pa...|    spam|
|file:/Users/pasca...|Subject: auto ins...|    spam|
|file:/Users/pasca...|Subject: want bes...|    spam|
|file:/Users/pasca...|Subject: email 57...|    spam|
|file:/Users/pasca...|Subject: n't miss...|    spam|
|file:/Users/pasca...|Subject: amaze wo...|    spam|
|file:/Users/pasca...|Subject: help loa...|    spam|
|file:/Users/pasca...|Subject: beat irs...|    spam|
|file:/Users/pasca...|Subject: email 57...|    spam|
|file:/Users/pasca...|Subject: best , b...|    spam|
|...                                               |
+--------------------+--------------------+--------+

scala> documentsDF.groupBy("category").agg(count("*")).show
+--------+--------+
|category|COUNT(1)|
+--------+--------+
|    spam|     481|
|     ham|    2412|
+--------+--------+

```

现在让我们将 DataFrame 分割成训练集和测试集。我们将使用测试集来验证我们构建的模型。现在，我们将只使用一个分割，用 70%的数据训练模型，用剩余的 30%进行测试。在下一节中，我们将探讨交叉验证，它提供了一种更严格的方式来检查我们模型的准确性。

我们可以使用 DataFrame 的`.randomSplit`方法实现 70-30 的分割：

```py
scala> val Array(trainDF, testDF) = documentsDF.randomSplit(
 Array(0.7, 0.3))
trainDF: DataFrame = [fileName: string, text: string, category: string]
testDF: DataFrame = [fileName: string, text: string, category: string]

```

`.randomSplit`方法接受一个权重数组，并返回一个 DataFrame 数组，其大小大约由权重指定。例如，我们传递了权重`0.7`和`0.3`，表示任何给定行有 70%的机会最终进入`trainDF`，有 30%的机会进入`testDF`。请注意，这意味着分割的 DataFrame 大小不是固定的：`trainDF`大约是`documentsDF`的 70%，但不是正好 70%：

```py
scala> trainDF.count / documentsDF.count.toDouble
Double = 0.7013480815762184

```

如果你需要一个固定大小的样本，请使用 DataFrame 的`.sample`方法来获取`trainDF`，并过滤`documentDF`以排除`trainDF`中的行。

我们现在可以开始使用 MLlib 了。我们的分类尝试将涉及在*词频向量*上执行逻辑回归：我们将计算每个单词在每个消息中出现的频率，并使用发生频率作为特征。在深入代码之前，让我们退一步来讨论机器学习管道的结构。

# 管道组件

管道由一系列组件组成，这些组件连接在一起，使得一个组件生成的 DataFrame 被用作下一个组件的输入。可用的组件分为两类：*转换器*和*估计器*。

## 转换器

**转换器**将一个 DataFrame 转换成另一个，通常是通过添加一个或多个列。

我们垃圾邮件分类算法的第一步是将每个消息分割成一个单词数组。这被称为**分词**。我们可以使用 MLlib 提供的`Tokenizer`转换器：

```py
scala> import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature._

scala> val tokenizer = new Tokenizer()
tokenizer: org.apache.spark.ml.feature.Tokenizer = tok_75559f60e8cf 

```

可以通过获取器和设置器来定制转换器的行为。获取可用参数列表的最简单方法是通过调用`.explainParams`方法：

```py
scala> println(tokenizer.explainParams)
inputCol: input column name (undefined)
outputCol: output column name (default: tok_75559f60e8cf__output)

```

我们看到，可以使用两个参数来定制`Tokenizer`实例的行为：`inputCol`和`outputCol`，分别描述包含输入（要分词的字符串）和输出（单词数组）的列的标题。我们可以使用`setInputCol`和`setOutputCol`方法设置这些参数。

我们将`inputCol`设置为`"text"`，因为在我们的训练和测试 DataFrame 中，该列被命名为`text`。我们将`outputCol`设置为`"words"`：

```py
scala> tokenizer.setInputCol("text").setOutputCol("words")
org.apache.spark.ml.feature.Tokenizer = tok_75559f60e8cf

```

在适当的时候，我们将`tokenizer`集成到管道中，但现在，让我们只使用它来转换训练 DataFrame，以验证其是否正确工作。

```py
scala> val tokenizedDF = tokenizer.transform(trainDF)
tokenizedDF: DataFrame = [fileName: string, text: string, category: string, words: array<string>]

scala> tokenizedDF.show
+--------------+----------------+--------+--------------------+
|      fileName|            text|category|               words|
+--------------+----------------+--------+--------------------+
|file:/Users...|Subject: auto...|    spam|[subject:, auto, ...|
|file:/Users...|Subject: want...|    spam|[subject:, want, ...|
|file:/Users...|Subject: n't ...|    spam|[subject:, n't, m...|
|file:/Users...|Subject: amaz...|    spam|[subject:, amaze,...|
|file:/Users...|Subject: help...|    spam|[subject:, help, ...|
|file:/Users...|Subject: beat...|    spam|[subject:, beat, ...|
|...                                                          |
+--------------+----------------+--------+--------------------+

```

`tokenizer`转换器生成一个新的 DataFrame，其中包含一个额外的列`words`，包含`text`列中的单词数组。

显然，我们可以使用我们的`tokenizer`来转换具有正确模式的任何 DataFrame。例如，我们可以将其用于测试集。机器学习的许多方面都涉及到在不同的数据集上调用相同的（或非常相似的）管道。通过提供管道抽象，MLlib 简化了由许多清理、转换和建模组件组成的复杂机器学习算法的推理。

我们管道中的下一步是计算每个消息中每个单词的出现频率。我们最终将使用这些频率作为算法中的特征。我们将使用`HashingTF`转换器将单词数组转换为每个消息的词频向量。

`HashingTF`转换器从输入可迭代对象构建一个词频的稀疏向量。单词数组中的每个元素都被转换为一个哈希码。这个哈希码被截断为一个介于*0*和输出向量中元素总数*大数 n*之间的值。词频向量仅仅是截断哈希的出现次数。

让我们手动运行一个示例来理解它是如何工作的。我们将计算`Array("the", "dog", "jumped", "over", "the")`的词频向量。在这个例子中，我们将稀疏输出向量中的元素数量*n*设为 16。第一步是计算数组中每个元素的哈希码。我们可以使用内置的`##`方法，该方法为任何对象计算哈希码：

```py
scala> val words = Array("the", "dog", "jumped", "over", "the")
words: Array[String] = Array(the, dog, jumped, over, the)

scala> val hashCodes = words.map { _.## }
hashCodes: Array[Int] = Array(114801, 99644, -1148867251, 3423444, 114801)

```

为了将哈希码转换为有效的向量索引，我们将每个哈希码对向量的大小（在这种情况下为`16`）取模：

```py
scala> val indices = hashCodes.map { code => Math.abs(code % 16) }
indices: Array[Int] = Array(1, 12, 3, 4, 1)

```

然后，我们可以创建一个从索引到该索引出现次数的映射：

```py
scala> val indexFrequency = indices.groupBy(identity).mapValues {
 _.size.toDouble
}
indexFrequency: Map[Int,Double] = Map(4 -> 1.0, 1 -> 2.0, 3 -> 1.0, 12 -> 1.0)

```

最后，我们可以将此映射转换为稀疏向量，其中向量的每个元素的值是此特定索引出现的频率：

```py
scala> import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg._

scala> val termFrequencies = Vectors.sparse(16, indexFrequency.toSeq)
termFrequencies: linalg.Vector = (16,[1,3,4,12],[2.0,1.0,1.0,1.0])

```

注意，稀疏向量的`.toString`输出由三个元素组成：向量的总大小，后跟两个列表：第一个是索引系列，第二个是这些索引处的值系列。

使用稀疏向量提供了一种紧凑且高效的方式来表示消息中单词出现的频率，这正是`HashingTF`在底层的工作方式。缺点是，从单词到索引的映射不一定唯一：通过向量的长度截断哈希码将不同的字符串映射到相同的索引。这被称为*碰撞*。解决方案是使*n*足够大，以最小化碰撞的频率。

### 小贴士

`HashingTF`类似于构建一个哈希表（例如，Scala 映射），其键是单词，值是单词在消息中出现的次数，但有一个重要区别：它不试图处理哈希冲突。因此，如果两个单词映射到相同的哈希，它们将具有错误的频率。使用此算法而不是仅构建哈希表有两个优点：

+   我们不需要在内存中维护一个不同单词的列表。

+   每封电子邮件都可以独立于其他所有电子邮件转换为向量：我们不需要在不同的分区上执行降维操作来获取映射中的键集。这极大地简化了将此算法应用于分布式环境中的每封电子邮件，因为我们可以在每个分区上独立应用`HashingTF`转换。

主要缺点是我们必须使用能够高效利用稀疏表示的机器学习算法。这种情况适用于逻辑回归，我们将在下面使用。

如您所预期的那样，`HashingTF`转换器接受输入和输出列作为参数。它还接受一个参数，定义向量中不同的哈希桶的数量。增加桶的数量会减少冲突的数量。在实践中，建议的值在![Transformers](img/image01214.jpeg)和![Transformers](img/image01215.jpeg)之间。

```py
scala> val hashingTF = (new HashingTF()
 .setInputCol("words")
 .setOutputCol("features")
 .setNumFeatures(1048576))
hashingTF: org.apache.spark.ml.feature.HashingTF = hashingTF_3b78eca9595c

scala> val hashedDF = hashingTF.transform(tokenizedDF)
hashedDF: DataFrame = [fileName: string, text: string, category: string, words: array<string>, features: vector]

scala> hashedDF.select("features").show
+--------------------+
|            features|
+--------------------+
|(1048576,[0,33,36...|
|(1048576,[0,36,40...|
|(1048576,[0,33,34...|
|(1048576,[0,33,36...|
|(1048576,[0,33,34...|
|(1048576,[0,33,34...|
+--------------------+

```

`features`列中的每个元素都是一个稀疏向量：

```py
scala> import org.apache.spark.sql.Row
import org.apache.spark.sql.Row

scala> val firstRow = hashedDF.select("features").first
firstRow: org.apache.spark.sql.Row = ...

scala> val Row(v:Vector) = firstRow
v: Vector = (1048576,[0,33,36,37,...],[1.0,3.0,4.0,1.0,...])

```

因此，我们可以将我们的向量解释为：哈希到元素`33`的单词出现三次，哈希到元素`36`的单词出现四次等等。

## 估计器

现在，我们已经为逻辑回归准备好了特征。在运行逻辑回归之前，最后一步是创建目标变量。我们将 DataFrame 中的`category`列转换为二进制 0/1 目标列。Spark 提供了一个`StringIndexer`类，该类将列中的字符串集替换为双精度浮点数。`StringIndexer`不是一个转换器：它必须首先与一组类别拟合以计算从字符串到数值值的映射。这引入了管道 API 中的第二类组件：*估计器*。

与“开箱即用”的转换器不同，估计器必须与 DataFrame 拟合。对于我们的字符串索引器，拟合过程包括获取唯一字符串列表（`"spam"`和`"ham"`）并将每个这些映射到双精度浮点数。拟合过程输出一个转换器，该转换器可以用于后续的 DataFrames。

```py
scala> val indexer = (new StringIndexer()
 .setInputCol("category")
 .setOutputCol("label"))
indexer: org.apache.spark.ml.feature.StringIndexer = strIdx_16db03fd0546

scala> val indexTransform = indexer.fit(trainDF)
indexTransform: StringIndexerModel = strIdx_16db03fd0546

```

拟合过程产生的转换器有一个`labels`属性，描述了它应用的映射：

```py
scala> indexTransform.labels
Array[String] = Array(ham, spam)

```

每个标签都将映射到数组中的索引：因此，我们的转换器将`ham`映射到`0`，将`spam`映射到`1`：

```py
scala> val labelledDF = indexTransform.transform(hashedDF)
labelledDF: org.apache.spark.sql.DataFrame = [fileName: string, text: string, category: string, words: array<string>, features: vector, label: double]

scala> labelledDF.select("category", "label").distinct.show
+--------+-----+
|category|label|
+--------+-----+
|     ham|  0.0|
|    spam|  1.0|
+--------+-----+

```

现在我们有了适合逻辑回归的正确格式的特征向量和分类标签。执行逻辑回归的组件是一个估计器：它被拟合到一个训练 DataFrame 中，以创建一个训练好的模型。然后，可以使用该模型来转换测试 DataFrame。

```py
scala> import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegression

scala> val classifier = new LogisticRegression().setMaxIter(50)
classifier: LogisticRegression = logreg_a5e921e7c1a1 

```

`LogisticRegression`估计器默认期望特征列命名为`"features"`，标签列（目标）命名为`"label"`。没有必要明确设置这些，因为它们与`hashingTF`和`indexer`设置的列名匹配。有几个参数可以设置以控制逻辑回归的工作方式：

```py
scala> println(classifier.explainParams)
elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. (default: 0.0)
fitIntercept: whether to fit an intercept term (default: true)
labelCol: label column name (default: label)
maxIter: maximum number of iterations (>= 0) (default: 100, current: 50)
regParam: regularization parameter (>= 0) (default: 0.0)
threshold: threshold in binary classification prediction, in range [0, 1] (default: 0.5)
tol: the convergence tolerance for iterative algorithms (default: 1.0E-6)
...

```

现在，我们只设置`maxIter`参数。稍后我们将研究其他参数的影响，例如正则化。现在，让我们将分类器拟合到`labelledDF`：

```py
scala> val trainedClassifier = classifier.fit(labelledDF)
trainedClassifier: LogisticRegressionModel = logreg_353d18f6a5f0

```

这产生了一个转换器，我们可以将其应用于具有`features`列的 DataFrame。转换器附加了一个`prediction`列和一个`probability`列。例如，我们可以使用`trainedClassifier`将`labelledDF`（训练集本身）转换：

```py
scala> val labelledDFWithPredictions = trainedClassifier.transform(
 labelledDF)
labelledDFWithPredictions: DataFrame = fileName: string, ...

scala> labelledDFWithPredictions.select($"label", $"prediction").show
+-----+----------+
|label|prediction|
+-----+----------+
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
|  1.0|       1.0|
+-----+----------+

```

检查我们模型性能的一个快速方法是仅计算误分类消息的数量：

```py
scala> labelledDFWithPredictions.filter { 
 $"label" !== $"prediction" 
}.count
Long = 1

```

在这种情况下，逻辑回归成功地将训练集中除了一条消息外的所有消息正确分类。考虑到特征数量众多，以及垃圾邮件和合法电子邮件中使用的单词之间的相对清晰界限，这也许并不令人惊讶。

当然，对模型的真实测试不是它在训练集上的表现，而是在测试集上的表现。为了测试这一点，我们可以将测试 DataFrame 通过与我们用于训练模型的相同阶段，用它们产生的拟合转换器替换估计器。MLlib 提供了*管道*抽象来简化这个过程：我们将有序列表的转换器和估计器包装在管道中。然后，这个管道被拟合到一个对应于训练集的 DataFrame 中。拟合产生一个`PipelineModel`实例，相当于管道，但估计器被转换器替换，如图所示：

![估计器

让我们构建我们的逻辑回归垃圾邮件过滤器的管道：

```py
scala> import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.Pipeline

scala> val pipeline = new Pipeline().setStages(
 Array(indexer, tokenizer, hashingTF, classifier)
)
pipeline: Pipeline = pipeline_7488113e284d

```

一旦定义了管道，我们就将其拟合到包含训练集的 DataFrame 中：

```py
scala> val fittedPipeline = pipeline.fit(trainDF)
fittedPipeline: org.apache.spark.ml.PipelineModel = pipeline_089525c6f100

```

当将管道拟合到 DataFrame 时，估计器和转换器被处理得不同：

+   转换器被应用到 DataFrame 中，并直接复制到管道模型中。

+   估计器被拟合到 DataFrame 中，生成一个转换器。然后，转换器被应用到 DataFrame 上，并附加到管道模型中。

我们现在可以将管道模型应用到测试集上：

```py
scala> val testDFWithPredictions = fittedPipeline.transform(testDF)
testDFWithPredictions: DataFrame = fileName: string, ...

```

这在 DataFrame 中添加了一个`prediction`列，其中包含我们的逻辑回归模型的预测结果。为了衡量我们算法的性能，我们在测试集上计算分类错误：

```py
scala> testDFWithPredictions.filter { 
 $"label" !== $"prediction" 
}.count
Long = 20

```

因此，我们的朴素逻辑回归算法，没有模型选择或正则化，将 2.3%的电子邮件误分类。当然，由于训练集和测试集的划分是随机的，你可能会得到略微不同的结果。

让我们将包含预测的培训和测试 DataFrame 保存为`parquet`文件：

```py
scala> import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SaveMode

scala> (labelledDFWithPredictions
 .select("fileName", "label", "prediction", "probability")
 .write.mode(SaveMode.Overwrite)
 .parquet("transformedTrain.parquet"))

scala> (testDFWithPredictions
 .select("fileName", "label", "prediction", "probability")
 .write.mode(SaveMode.Overwrite)
 .parquet("transformedTest.parquet"))

```

### 小贴士

在垃圾邮件分类中，误报比漏报要严重得多：将合法邮件误判为垃圾邮件，比让垃圾邮件通过要糟糕得多。为了解决这个问题，我们可以提高分类的阈值：只有得分达到 0.7 或以上的邮件才会被分类为垃圾邮件。这引发了选择正确阈值的问题。一种方法是对不同阈值在测试集上产生的误报率进行调查，并选择最低的阈值以获得可接受的误报率。可视化这一点的良好方法是使用 ROC 曲线，我们将在下一节中探讨。

# 评估

不幸的是，截至版本 1.5.2，管道 API 中评估模型质量的功能仍然有限。逻辑回归确实输出一个包含多个评估指标（通过训练模型的`summary`属性可用）的摘要，但这些是在训练集上计算的。通常，我们希望在训练集和单独的测试集上评估模型性能。因此，我们将深入到底层的 MLlib 层以访问评估指标。

MLlib 提供了一个模块，`org.apache.spark.mllib.evaluation`，其中包含一系列用于评估模型质量的类。在这里，我们将使用`BinaryClassificationMetrics`类，因为垃圾邮件分类是一个二分类问题。其他评估类为多分类模型、回归模型和排序模型提供指标。

如前所述，我们将在 shell 中阐述这些概念，但您将在本章代码示例中的`ROC.scala`脚本中找到类似代码。我们将使用*breeze-viz*来绘制曲线，因此，在启动 shell 时，我们必须确保相关的库在类路径上。我们将使用 SBT assembly，如第十章[分布式批处理与 Spark 中所述，*分布式批处理与 Spark*（特别是*构建和运行独立程序*部分），来创建一个包含所需依赖项的 JAR 文件。然后我们将这个 JAR 文件传递给 Spark shell，这样我们就可以导入 breeze-viz。让我们编写一个`build.sbt`文件，声明对 breeze-viz 的依赖：

```py
// build.sbt
name := "spam_filter"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.2" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.5.2" % "provided",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2"
)
```

使用以下命令将依赖项打包到 jar 中：

```py
$ sbt assembly

```

这将在`target/scala-2.10`目录下创建一个名为`spam_filter-assembly-0.1-SNAPSHOT.jar`的 JAR 文件。要将此 JAR 文件包含在 Spark shell 中，请使用`--jars`命令行参数重新启动 shell：

```py
$ spark-shell --jars=target/scala-2.10/spam_filter-assembly-0.1-SNAPSHOT.jar

```

为了验证打包是否成功，尝试导入`breeze.plot`：

```py
scala> import breeze.plot._
import breeze.plot._

```

让我们加载测试集，包括预测，我们在上一节中创建并保存为`parquet`文件：

```py
scala> val testDFWithPredictions = sqlContext.read.parquet(
 "transformedTest.parquet")
testDFWithPredictions: org.apache.spark.sql.DataFrame = [fileName: string, label: double, prediction: double, probability: vector]

```

`BinaryClassificationMetrics`对象期望一个`RDD[(Double, Double)]`对象，其中包含一对分数（分类器分配给特定电子邮件是垃圾邮件的概率）和标签（电子邮件是否实际上是垃圾邮件）。我们可以从我们的 DataFrame 中提取这个 RDD：

```py
scala> import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vector

scala> import org.apache.spark.sql.Row
import org.apache.spark.sql.Row

scala> val scoresLabels = testDFWithPredictions.select(
 "probability", "label").map {
 case Row(probability:Vector, label:Double) => 
 (probability(1), label)
}
org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[3] at map at <console>:23

scala> scoresLabels.take(5).foreach(println)
(0.9999999967713409,1.0)
(0.9999983827108793,1.0)
(0.9982059900606365,1.0)
(0.9999790713978142,1.0)
(0.9999999999999272,1.0)

```

我们现在可以构建`BinaryClassificationMetrics`实例：

```py
scala> import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import mllib.evaluation.BinaryClassificationMetrics

scala> val bm = new BinaryClassificationMetrics(scoresLabels)
bm: BinaryClassificationMetrics = mllib.evaluation.BinaryClassificationMetrics@254ed9ba

```

`BinaryClassificationMetrics` 对象包含许多用于评估分类模型性能的有用指标。我们将探讨**接收者操作****特征**（**ROC**）曲线。

### 提示

**ROC 曲线**

想象逐渐降低，从 1.0 开始，我们假设特定电子邮件是垃圾邮件的概率阈值。显然，当阈值设置为 1.0 时，没有电子邮件会被分类为垃圾邮件。这意味着不会有**假阳性**（我们错误地将正常邮件分类为垃圾邮件），但也意味着不会有**真阳性**（我们正确地将垃圾邮件识别为垃圾邮件）：所有垃圾邮件都会被错误地识别为正常邮件。

随着我们逐渐降低我们假设特定电子邮件是垃圾邮件的概率阈值，我们的垃圾邮件过滤器，希望如此，将开始识别大量电子邮件为垃圾邮件。其中绝大多数，如果我们的算法设计得很好，将是真正的垃圾邮件。因此，我们的真正例率增加。随着我们逐渐降低阈值，我们开始将我们不太确定的邮件分类为垃圾邮件。这将增加正确识别为垃圾邮件的邮件数量，但也会增加误报的数量。

ROC 曲线图对于每个阈值值，绘制真正例率与假正例率的比率。在最佳情况下，曲线始终为 1：这发生在所有垃圾邮件消息都被赋予 1.0 分，而所有正常邮件都被赋予 0.0 分时。相比之下，最坏的情况发生在曲线为对角线 *P(真正例) = P(假正例)* 时，这发生在我们的算法不如随机时。通常，ROC 曲线位于两者之间，形成一个位于对角线之上的凸壳。这个壳越深，我们的算法就越好。

![评估](img/image01217.jpeg)

（左）对于一个明显优于随机性能的模型的 ROC 曲线：曲线在低误报率下达到非常高的真正例率。

（中间）对于一个显著优于随机性能的模型的 ROC 曲线。

（右）对于一个仅略优于随机性能的模型：对于任何给定的阈值，真正例率仅略高于假正例率，这意味着近一半的示例被错误分类。

我们可以使用 `BinaryClassificationMetrics` 实例上的 `.roc` 方法计算 ROC 曲线上的点数组。这返回一个 `RDD[(Double, Double)]`，包含每个阈值值的 (*假正例*，*真正例*) 比率。我们可以将其收集为数组：

```py
scala> val rocArray = bm.roc.collect
rocArray: Array[(Double, Double)] = Array((0.0,0.0), (0.0,0.16793893129770993), ...

```

当然，一个数字数组并不很有启发性，所以让我们用 breeze-viz 绘制 ROC 曲线。我们首先将我们的配对数组转换为两个数组，一个为假正例，一个为真正例：

```py
scala> val falsePositives = rocArray.map { _._1 }
falsePositives: Array[Double] = Array(0.0, 0.0, 0.0, 0.0, 0.0, ...

scala> val truePositives = rocArray.map { _._2 }
truePositives: Array[Double] = Array(0.0, 0.16793893129770993, 0.19083969465...

```

让我们绘制这两个数组：

```py
scala> import breeze.plot._
import breeze.plot.

scala> val f = Figure()
f: breeze.plot.Figure = breeze.plot.Figure@3aa746cd

scala> val p = f.subplot(0)
p: breeze.plot.Plot = breeze.plot.Plot@5ed1438a

scala> p += plot(falsePositives, truePositives)
p += plot(falsePositives, truePositives)

scala> p.xlabel = "false positives"
p.xlabel: String = false positives

scala> p.ylabel = "true positives"
p.ylabel: String = true positives

scala> p.title = "ROC"
p.title: String = ROC

scala> f.refresh

```

ROC 曲线在 x 的一个较小值时达到 *1.0*：也就是说，我们以相对较少的误报为代价检索到所有真正例。为了更准确地可视化曲线，限制 *x* 轴的范围从 *0* 到 *0.1* 是有益的。

```py
scala> p.xlim = (0.0, 0.1)
p.xlim: (Double, Double) = (0.0,0.1)

```

我们还需要告诉 breeze-viz 使用适当的刻度间隔，这需要深入到 breeze-viz 之下的 JFreeChart 层：

```py
scala> import org.jfree.chart.axis.NumberTickUnit
import org.jfree.chart.axis.NumberTickUnit

scala> p.xaxis.setTickUnit(new NumberTickUnit(0.01))

scala> p.yaxis.setTickUnit(new NumberTickUnit(0.1))

```

我们现在可以保存这个图表：

```py
scala> f.saveas("roc.png")

```

这将生成以下图表，存储在 `roc.png` 中：

![评估](img/image01218.jpeg)

使用逻辑回归进行垃圾邮件分类的 ROC 曲线。注意，我们已经将假正例轴限制在 0.1

通过观察图表，我们看到我们可以过滤掉 85% 的垃圾邮件而没有单个 **误报**。当然，我们需要一个更大的测试集来真正验证这个假设。

图形有助于真正理解模型的行为。有时，我们只是想有一个衡量模型质量的单一指标。ROC 曲线下的面积可以是一个很好的这样的指标：

```py
scala> bm.areaUnderROC
res21: Double = 0.9983061235861147

```

这可以解释如下：给定从测试集中随机抽取的两个消息，其中一个为垃圾邮件，另一个为正常邮件，模型将垃圾邮件分配给垃圾邮件消息的似然性大于正常邮件消息的似然性的概率为 99.8%。

其他衡量模型质量的指标包括特定阈值下的精确度和召回率，或者 F1 分数。所有这些都可以通过`BinaryClassificationMetrics`实例提供。API 文档列出了可用的方法：[`spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.evaluation.BinaryClassificationMetrics`](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.evaluation.BinaryClassificationMetrics)。

# 逻辑回归中的正则化

机器学习的一个危险是过拟合：算法不仅捕获了训练集中的信号，而且还捕获了由训练集有限大小产生的统计噪声。

在逻辑回归中减轻过拟合的一种方法是使用正则化：我们在优化时对参数的大值施加惩罚。我们可以通过向代价函数添加与参数幅度成比例的惩罚来实现这一点。形式上，我们将逻辑回归代价函数（在第二章，*使用 Breeze 操作数据*中描述）重新写为：

![逻辑回归中的正则化](img/image01219.jpeg)

其中 ![逻辑回归中的正则化](img/image01220.jpeg) 是标准的逻辑回归代价函数：

![逻辑回归中的正则化](img/image01221.jpeg)

在这里，*params* 是参数向量，![逻辑回归中的正则化](img/image01222.jpeg) 是第 *i* 个训练示例的特征向量，而 ![逻辑回归中的正则化](img/image01223.jpeg) 是当第 *i* 个训练示例是垃圾邮件时为 *1*，否则为 *0*。这与第二章，*使用 Breeze 操作数据*中引入的逻辑回归代价函数相同，除了添加了正则化项 ![逻辑回归中的正则化](img/image01224.jpeg) 和参数向量的 ![逻辑回归中的正则化](img/image01225.jpeg) 范数。*n* 的最常见值是 2，在这种情况下 ![逻辑回归中的正则化](img/image01226.jpeg) 只是参数向量的幅度：

![逻辑回归中的正则化](img/image01227.jpeg)

额外的正则化项驱动算法减小参数向量的幅度。在使用正则化时，特征必须具有可比的幅度。这通常通过归一化特征来实现。MLlib 提供的逻辑回归估计器默认情况下归一化所有特征。这可以通过`setStandardization`参数关闭。

Spark 有两个可以调整的超参数来控制正则化：

+   正则化的类型，通过`elasticNetParam`参数设置。0 值表示![逻辑回归中的正则化](img/image01228.jpeg)正则化。

+   正则化的程度（成本函数中的![逻辑回归中的正则化](img/image01229.jpeg)），通过`regParam`参数设置。正则化参数的高值表示强烈的正则化。一般来说，过拟合的危险越大，正则化参数应该越大。

让我们创建一个新的逻辑回归实例，该实例使用正则化：

```py
scala> val lrWithRegularization = (new LogisticRegression()
 .setMaxIter(50))
lrWithRegularization: LogisticRegression = logreg_16b65b325526

scala> lrWithRegularization.setElasticNetParam(0) lrWithRegularization.type = logreg_1e3584a59b3a

```

为了选择![逻辑回归中的正则化](img/image01229.jpeg)的适当值，我们将管道拟合到训练集，并计算测试集上![逻辑回归中的正则化](img/image01229.jpeg)的几个值的分类误差。在章节的后面，我们将学习 MLlib 中的交叉验证，它提供了一种更严格的方法来选择超参数。

```py
scala> val lambdas = Array(0.0, 1.0E-12, 1.0E-10, 1.0E-8)
lambdas: Array[Double] = Array(0.0, 1.0E-12, 1.0E-10, 1.0E-8)

scala> lambdas foreach { lambda =>
 lrWithRegularization.setRegParam(lambda)
 val pipeline = new Pipeline().setStages(
 Array(indexer, tokenizer, hashingTF, lrWithRegularization))
 val model = pipeline.fit(trainDF)
 val transformedTest = model.transform(testDF)
 val classificationError = transformedTest.filter { 
 $"prediction" !== $"label"
 }.count
 println(s"$lambda => $classificationError")
}
0 => 20
1.0E-12 => 20
1.0E-10 => 20
1.0E-8 => 23

```

对于我们的例子，我们看到任何尝试添加 L[2]正则化的尝试都会导致分类精度的下降。

# 交叉验证和模型选择

在上一个例子中，我们通过在训练时保留 30%的数据，并在该子集上进行测试来验证我们的方法。这种方法并不特别严格：确切的结果取决于随机的训练-测试分割。此外，如果我们想测试几个不同的超参数（或不同的模型）以选择最佳模型，我们可能会无意中选择的模型最能反映测试集中特定行，而不是整体人群。

这可以通过*交叉验证*来克服。我们已经在第四章中遇到了交叉验证，*并行集合和未来*。在那个章节中，我们使用了随机子样本交叉验证，其中我们随机创建训练-测试分割。

在本章中，我们将使用**k 折交叉验证**：我们将训练集分成*k*部分（其中，通常*k*是*10*或*3*），使用*k-1*部分作为训练集，最后的部分作为测试集。重复*k*次训练/测试周期，每次保持不同的部分作为测试集。

交叉验证通常用于选择模型的最佳超参数集。为了说明选择合适的超参数，我们将回到我们的正则化逻辑回归示例。我们不会自己直觉超参数，而是选择给我们最佳交叉验证分数的超参数。

我们将探讨设置正则化类型（通过`elasticNetParam`）和正则化程度（通过`regParam`）。找到一个好的参数值的一个粗略但有效的方法是执行网格搜索：我们计算正则化参数感兴趣值对的交叉验证分数。

我们可以使用 MLlib 的`ParamGridBuilder`构建参数网格。

```py
scala> import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

scala> val paramGridBuilder = new ParamGridBuilder()
paramGridBuilder: ParamGridBuilder = ParamGridBuilder@1dd694d0

```

要将优化超参数添加到网格中，我们使用`addGrid`方法：

```py
scala> val lambdas = Array(0.0, 1.0E-12, 1.0E-10, 1.0E-8)
Array[Double] = Array(0.0, 1.0E-12, 1.0E-10, 1.0E-8)

scala> val elasticNetParams = Array(0.0, 1.0)
elasticNetParams: Array[Double] = Array(0.0, 1.0)

scala> paramGridBuilder.addGrid(
 lrWithRegularization.regParam, lambdas).addGrid(
 lrWithRegularization.elasticNetParam, elasticNetParams)
paramGridBuilder.type = ParamGridBuilder@1dd694d0

```

一旦添加了所有维度，我们只需在构建器上调用`build`方法来构建网格：

```py
scala> val paramGrid = paramGridBuilder.build
paramGrid: Array[org.apache.spark.ml.param.ParamMap] =
Array({
 logreg_f7dfb27bed7d-elasticNetParam: 0.0,
 logreg_f7dfb27bed7d-regParam: 0.0
}, {
 logreg_f7dfb27bed7d-elasticNetParam: 1.0,
 logreg_f7dfb27bed7d-regParam: 0.0
} ...)

scala> paramGrid.length
Int = 8

```

如我们所见，网格只是一个参数集的一维数组，在拟合逻辑回归模型之前传递给模型。

设置交叉验证管道的下一步是定义一个用于比较模型性能的指标。在本章的早期，我们看到了如何使用`BinaryClassificationMetrics`来估计模型的质量。不幸的是，`BinaryClassificationMetrics`类是核心 MLLib API 的一部分，而不是新的管道 API，因此它（不容易）兼容。管道 API 提供了一个`BinaryClassificationEvaluator`类。这个类直接在 DataFrame 上工作，因此非常适合管道 API 流程：

```py
scala> import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

scala> val evaluator = new BinaryClassificationEvaluator()
evaluator: BinaryClassificationEvaluator = binEval_64b08538f1a2

scala> println(evaluator.explainParams)
labelCol: label column name (default: label)
metricName: metric name in evaluation (areaUnderROC|areaUnderPR) (default: areaUnderROC)
rawPredictionCol: raw prediction (a.k.a. confidence) column name (default: rawPrediction)

```

从参数列表中，我们看到`BinaryClassificationEvaluator`类支持两个指标：ROC 曲线下的面积和精确率-召回率曲线下的面积。它期望输入一个包含`label`列（模型真实值）和`rawPrediction`列（包含电子邮件是垃圾邮件或正常邮件的概率的列）的 DataFrame。

我们现在拥有了运行交叉验证所需的所有参数。我们首先构建管道，然后将管道、评估器和要运行交叉验证的参数数组传递给`CrossValidator`的一个实例：

```py
scala> val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, hashingTF, lrWithRegularization))
pipeline: Pipeline = pipeline_3ed29f72a4cc

scala> val crossval = (new CrossValidator()
 .setEstimator(pipeline)
 .setEvaluator(evaluator)
 .setEstimatorParamMaps(paramGrid)
 .setNumFolds(3))
crossval: CrossValidator = cv_5ebfa1143a9d 

```

我们现在将`crossval`拟合到`trainDF`：

```py
scala> val cvModel = crossval.fit(trainDF)
cvModel: CrossValidatorModel = cv_5ebfa1143a9d

```

这一步可能需要相当长的时间（在单台机器上可能超过一小时）。这创建了一个对应于具有最佳参数表示`trainDF`的逻辑回归对象的 transformer，`cvModel`。我们可以用它来预测测试 DataFrame 上的分类错误：

```py
scala> cvModel.transform(testDF).filter { 
 $"prediction" !== $"label" 
}.count
Long = 20

```

因此，交叉验证产生了一个与原始、无超参数的朴素逻辑回归模型表现相同的模型。`cvModel`还包含参数网格中每组的评估分数列表：

```py
scala> cvModel.avgMetrics
Array[Double] = Array(0.996427805316161, ...)

```

将其与超参数相关联的最简单方法是将它与`cvModel.getEstimatorParamMaps`一起压缩。这给我们一个（超参数值，交叉验证分数）对的列表：

```py
scala> val params2score = cvModel.getEstimatorParamMaps.zip(
 cvModel.avgMetrics)
Array[(ml.param.ParamMap,Double)] = Array(({
 logreg_8f107aabb304-elasticNetParam: 0.0,
 logreg_8f107aabb304-regParam: 0.0
},0.996427805316161),...

scala> params2score.foreach {
 case (params, score) => 
 val lambda = params(lrWithRegularization.regParam)
 val elasticNetParam = params(
 lrWithRegularization.elasticNetParam)
 val l2Orl1 = if(elasticNetParam == 0.0) "L2" else "L1"
 println(s"$l2Orl1, $lambda => $score")
}
L2, 0.0 => 0.996427805316161
L1, 0.0 => 0.996427805316161
L2, 1.0E-12 => 0.9964278053175655
L1, 1.0E-12 => 0.9961429402772803
L2, 1.0E-10 => 0.9964382546369551
L1, 1.0E-10 => 0.9962223090037103
L2, 1.0E-8 => 0.9964159754613495
L1, 1.0E-8 => 0.9891008277659763

```

最佳的超参数集对应于 L[2]正则化，正则化参数为`1E-10`，尽管这仅对应于 AUC 的微小提升。

这完成了我们的垃圾邮件过滤器示例。我们已经成功地为这个特定的 Ling-Spam 数据集训练了一个垃圾邮件过滤器。为了获得更好的结果，可以尝试更好的特征提取：我们可以移除停用词或使用 TF-IDF 向量，而不是仅使用词频向量作为特征，我们还可以添加额外的特征，如消息长度，甚至*n-grams*。我们还可以尝试非线性算法，如随机森林。所有这些步骤都很容易添加到管道中。

# 逻辑回归之外

我们在本章中专注于逻辑回归，但 MLlib 提供了许多其他算法，这些算法可以更有效地捕捉数据中的非线性。管道 API 的一致性使得尝试不同的算法并查看它们的性能变得容易。管道 API 提供了用于分类的决策树、随机森林和梯度提升树，以及一个简单的前馈神经网络，这仍然是实验性的。它还提供了 Lasso 和岭回归以及用于回归的决策树，以及用于降维的 PCA。

更低级别的 MLlib API 还提供了降维的主成分分析，包括*k*-means 和潜在狄利克雷分配在内的几种聚类方法，以及使用交替最小二乘法的推荐系统。

# 摘要

MLlib 直面设计可扩展机器学习算法的挑战。在本章中，我们用它来训练一个简单的可扩展垃圾邮件过滤器。MLlib 是一个庞大且快速发展的库。了解它能提供什么最好的方式是尝试将你使用其他库（如 scikit-learn）编写的代码移植过来。

在下一章中，我们将探讨如何构建 Web API 和交互式可视化，以便与世界分享我们的结果。

# 参考文献

最佳的参考资料是在线文档，包括：

+   管道 API: [`spark.apache.org/docs/latest/ml-features.html`](http://spark.apache.org/docs/latest/ml-features.html)

+   完整的转换器列表: [`spark.apache.org/docs/latest/mllib-guide.html#sparkml-high-level-apis-for-ml-pipelines`](http://spark.apache.org/docs/latest/mllib-guide.html#sparkml-high-level-apis-for-ml-pipelines)

*《Spark 高级分析》*，由*Sandy Ryza*、*Uri Laserson*、*Sean Owen*和*Josh Wills*所著，提供了 Spark 机器学习的详细和最新介绍。

有几本书比我们在这里介绍的更详细地介绍了机器学习。我们在这本书中多次提到了*《统计学习的要素》*，由*Friedman*、*Tibshirani*和*Hastie*所著。这是目前可用的最完整的机器学习数学基础介绍之一。

安德鲁·纳格（Andrew Ng）的机器学习课程在[`www.coursera.org/`](https://www.coursera.org/)提供了机器学习的良好介绍。它使用 Octave/MATLAB 作为编程语言，但应该可以轻松地适应 Breeze 和 Scala。

# 第十三章。使用 Play 的网络 API

在本书的前 12 章中，我们介绍了任何想要构建数据科学应用程序的人的基本工具和库：我们学习了如何与 SQL 和 MongoDB 数据库交互，如何使用 Spark 构建快速批处理应用程序，如何使用 MLlib 应用最先进的机器学习算法，以及如何在 Akka 中构建模块化并发应用程序。

在本书的最后几章中，我们将扩展讨论，探讨一个网络框架：*Play*。你可能会 wonder 为什么网络框架会出现在一本数据科学书中；当然，这样的主题最好留给软件工程师或网络开发者。然而，数据科学家很少存在于真空之中。他们经常需要将结果或见解传达给利益相关者。对于一个精通统计学的某人来说，ROC 曲线可能很有说服力，但对于技术不那么精通的人来说，它可能没有那么大的分量。事实上，当见解伴随着引人入胜的视觉呈现时，销售见解可能会更容易。

许多现代交互式数据可视化应用程序都是运行在网页浏览器中的网络应用程序。通常，这些应用程序涉及**D3.js**，这是一个用于构建数据驱动网页的 JavaScript 库。在本章和下一章中，我们将探讨如何将 D3 与 Scala 集成。

编写一个网络应用程序是一项复杂的任务。我们将把这个任务分成本章和下一章来讨论。在本章中，我们将学习如何编写一个 REST API，我们可以将其用作应用程序的后端，或者直接查询。在下一章中，我们将探讨如何将前端代码与 Play 集成，以查询后端暴露的 API 并使用 D3 进行展示。在本章中，我们假设你对 HTTP 有至少基本的了解：你应该至少阅读过第七章，*网络 API*。

许多数据科学家或希望成为数据科学家的人可能不太熟悉网络技术的内部工作原理。学习如何构建复杂的网站或网络 API 可能会令人望而却步。因此，本章从对动态网站和网络应用程序架构的一般讨论开始。如果你已经熟悉服务器端编程和网络框架，你可以轻松地跳过前几节。

# 客户端-服务器应用程序

网站通过两台计算机之间的交互来工作：客户端和服务器。如果你在网页浏览器中输入[www.github.com/pbugnion/s4ds/graphs](http://www.github.com/pbugnion/s4ds/graphs)，你的浏览器会查询 GitHub 服务器之一。服务器会在其数据库中查找有关你感兴趣的仓库的信息。它将以 HTML、CSS 和 JavaScript 的形式将此信息提供给你的电脑。然后，你的浏览器负责以正确的方式解释这个响应。

如果你查看相关的 URL，你会注意到该页面上有几个图表。即使断开互联网连接，你仍然可以与这些图表进行交互。所有与图表交互所需的信息，在加载该网页时，都已经以 JavaScript 的形式传输。当你与图表互动时，使这些变化发生的 CPU 周期是在你的电脑上消耗的，而不是 GitHub 服务器。代码是在客户端执行的。相反，当你请求有关新仓库的信息时，该请求由 GitHub 服务器处理。这被称为服务器端处理。

可以在服务器上使用像 Play 这样的网络框架。对于客户端代码，我们只能使用客户端浏览器能理解的编程语言：HTML 用于布局，CSS 用于样式，JavaScript 用于逻辑，或者可以编译成 JavaScript 的语言。

# 网络框架简介

本节简要介绍了现代网络应用的设计方式。如果你已经熟悉编写后端代码，可以跳过这一部分。

大体上，网络框架是一套用于构建网络应用的工具和代码库。为了理解网络框架提供的内容，让我们退一步思考，如果你没有网络框架，你需要做什么。

你想要编写一个程序，监听 80 端口，并将 HTML（或 JSON 或 XML）发送回请求它的客户端。如果你要向每个客户端发送相同的文件，这很简单：只需在启动服务器时从文件中加载 HTML，并将其发送给请求它的客户端。

到目前为止，一切顺利。但如果你现在想根据客户端请求自定义 HTML，你会怎么做？你可能选择根据客户端在其浏览器中输入的 URL 的一部分，或者根据 HTTP 请求中的特定元素来做出不同的响应。例如，[amazon.com](http://amazon.com)上的产品页面与支付页面不同。你需要编写代码来解析 URL 和请求，然后将请求路由到相应的处理器。

现在，您可能希望根据请求的具体元素动态定制返回的 HTML。每个产品在[amazon.com](http://amazon.com)的页面都有相同的轮廓，但具体元素是不同的。为每个产品存储整个 HTML 内容将是浪费的。更好的方法是存储每个产品的详细信息到数据库中，并在客户端请求该产品的信息时将其注入到 HTML 模板中。您可以使用*模板处理器*来完成这项工作。当然，编写一个好的模板处理器是困难的。

您可能部署了您的 Web 框架，并意识到它无法处理指向它的流量。您决定响应客户端请求的处理程序应该异步运行。现在您必须处理并发。

Web 框架本质上提供了将一切连接在一起的“电线”。除了捆绑 HTTP 服务器外，大多数框架还会有一个路由器，它会根据 URL 自动将请求路由到正确的处理器。在大多数情况下，处理器将异步运行，这为您提供了更好的可扩展性。许多框架都有一个模板处理器，允许您直观地编写 HTML（有时是 JSON 或 XML）模板。一些 Web 框架还提供访问数据库、解析 JSON 或 XML、制定 HTTP 请求以及本地化和国际化的功能。

# 模型-视图-控制器架构

许多 Web 框架强加了程序架构：在不做出一些关于这些组件是什么的假设的情况下，很难提供将不同组件连接在一起的“电线”。**模型-视图-控制器**（**MVC**）架构在 Web 上特别受欢迎，并且是 Play 框架所假设的架构。让我们依次看看每个组件：

+   模型是应用程序背后的数据。例如，我期望 GitHub 背后的应用程序有用户、存储库、组织、拉取请求等模型。在 Play 框架中，模型通常是案例类的实例。模型的核心责任是记住应用程序的当前状态。

+   视图是模型或一组模型在屏幕上的表示。

+   控制器处理客户端交互，可能会更改模型。例如，如果您在 GitHub 上为项目添加星标，控制器将更新相关的模型。控制器通常携带很少的应用程序状态：记住事情是模型的工作。![模型-视图-控制器架构](img/image01230.jpeg)

    MVC 架构：应用程序的状态由模型提供。视图向用户提供模型的视觉表示，控制器处理逻辑：当用户按下按钮或提交表单时应该做什么。

MVC 框架工作得很好，因为它将用户界面与底层数据和结构分离，并结构化了动作的流程：控制器可以更新模型状态或视图，模型可以向视图发送信号，告诉它更新，而视图只是显示这些信息。模型不携带与用户界面相关的任何信息。这种关注点的分离导致了对信息流的更容易的心理模型、更好的封装和更高的可测试性。

# 单页应用程序

客户端-服务器二元性给优雅的 MVC 架构增加了一层复杂性。模型应该放在哪里？控制器呢？传统上，模型和控制器几乎完全运行在服务器上，服务器只是将相关的 HTML 视图推送到客户端。

客户端 JavaScript 框架的增长，如 AngularJS，导致了将更多代码放入客户端的逐渐转变。控制器和模型的临时版本通常都在客户端运行。服务器仅作为 Web API 运行：例如，如果用户更新了模型，控制器将向服务器发送一个 HTTP 请求，通知它变化。

然后，将运行在服务器端和客户端的程序视为两个独立的应用程序是有意义的：服务器在数据库中持久化数据，例如，并提供一个程序接口来访问这些数据，通常是通过返回 JSON 或 XML 数据的 Web 服务。客户端程序维护自己的模型和控制器，并在需要新的模型或需要通知服务器模型持久视图应该更改时轮询服务器。

极端情况下，这导致了**单页应用程序**。在单页应用程序中，客户端第一次从服务器请求页面时，他会收到构建整个应用程序框架所需的 HTML 和 JavaScript。如果客户端需要从服务器获取更多数据，他将轮询服务器的 API。这些数据以 JSON 或 XML 的形式返回。

这在抽象上可能看起来有些复杂，所以让我们思考一下亚马逊网站可能作为单页应用程序的结构。我们在这里只关注产品页面，因为那已经足够复杂了。让我们想象一下，你正在主页上，点击了一个特定产品的链接。运行在你电脑上的应用程序知道如何显示产品，例如通过 HTML 模板。JavaScript 也有一个模型的原型，例如：

```py
{
    product_id: undefined,
    product_name: undefined,
    product_price: undefined,
    ...
}
```

目前缺少的是关于你刚刚选择的产品应该在这些字段中放入哪些数据的知识：当网站加载时，信息不可能发送到你的电脑，因为没有方法知道你可能点击的产品（发送关于每个产品的信息将成本过高）。因此，Amazon 客户端向服务器发送关于该产品的信息请求。Amazon 服务器以 JSON 对象（或可能是 XML）的形式回复。然后客户端使用该信息更新其模型。当更新完成后，将触发一个事件来更新视图：

![单页应用程序](img/image01231.jpeg)

单页应用程序中的客户端-服务器通信：当客户端首次访问网站时，它会接收到包含应用程序全部逻辑的 HTML、CSS 和 JavaScript 文件。从那时起，客户端只有在请求额外数据时才将服务器用作 API。运行在用户浏览器中的应用程序和运行在服务器上的应用程序几乎是独立的。唯一的耦合是通过服务器暴露的 API 结构。

# 构建应用程序

在本章和下一章中，我们将构建一个依赖于用 Play 编写的 API 的单页应用程序。我们将构建一个看起来像这样的网页：

![构建应用程序](img/image01232.jpeg)

用户输入 GitHub 上某人的名字，可以查看他们的仓库列表以及一个总结他们使用语言的图表。您可以在 `app.scala4datascience.com` 上找到已部署的应用程序。不妨试一试。

要一窥其内部结构，请输入 `app.scala4datascience.com/api/repos/odersky`。这将返回一个类似以下的 JSON 对象：

```py
[{"name":"dotty","language":"Scala","is_fork":true,"size":14653},
{"name":"frontend","language":"JavaScript","is_fork":true,"size":392},
{"name":"legacy-svn-scala","language":"Scala","is_fork":true,"size":296706},
...
```

我们将在本章构建 API，并在下一章编写前端代码。

# Play 框架

Play 框架是在 Akka 之上构建的 Web 框架。它在业界有着可靠的记录，因此是构建可扩展 Web 应用程序的一个可靠选择。

Play 是一个有**明确立场**的 Web 框架：它期望你遵循 MVC 架构，并且对应该使用哪些工具有着强烈的看法。它自带 JSON 和 XML 解析器，自带访问外部 API 的工具，以及关于如何访问数据库的建议。

与我们在本书中开发的命令行脚本相比，Web 应用程序要复杂得多，因为它们包含更多的组件：后端代码、路由信息、HTML 模板、JavaScript 文件、图片等等。Play 框架对你的项目目录结构有很强的假设。从头开始构建这个结构既无聊又容易出错。幸运的是，我们可以使用**Typesafe activators**来启动项目（你也可以从[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)的 Git 仓库下载代码，但我鼓励你从基本的 activator 结构开始项目，并使用完成的版本作为示例，边学边做）。

Typesafe activator 是 SBT 的一个定制版本，它包含模板，可以帮助 Scala 程序员快速启动。要安装 activator，你可以从[`www.typesafe.com/activator/download`](https://www.typesafe.com/activator/download)下载一个 JAR 文件，或者在 Mac OS 上通过 homebrew：

```py
$ brew install typesafe-activator

```

你可以从终端启动 activator 控制台。如果你下载了 activator：

```py
$ ./path/to/activator/activator new

```

或者，如果你是通过 Homebrew 安装的：

```py
$ activator new

```

这将在当前目录中启动一个新的项目。它首先会询问你想要从哪个模板开始。选择 `play-scala`。然后它会询问你的应用程序的名称。我选择了 `ghub-display`，但你可以发挥创意！

让我们探索新创建的项目结构（我只保留了最重要的文件）：

```py
├── app
│   ├── controllers
│   │   └── Application.scala
│   └── views
│       ├── main.scala.html
│       └── index.scala.html
├── build.sbt
├── conf
│   ├── application.conf
│   └── routes
├── project
│   ├── build.properties
│   └── plugins.sbt
├── public
│   ├── images
│   │   └── favicon.png
│   ├── javascripts
│   │   └── hello.js
│   └── stylesheets
│       └── main.css
└── test
 ├── ApplicationSpec.scala
 └── IntegrationSpec.scala

```

让我们运行应用程序：

```py
$ ./activator
[ghub-display] $ run

```

打开浏览器并导航到 URL `127.0.0.1:9000/`。页面可能需要几秒钟才能加载。一旦加载完成，你应该看到一个默认页面，上面写着**您的应用程序已准备就绪**。

在我们修改任何内容之前，让我们了解一下这个过程。当你要求浏览器带你到 `127.0.0.1:9000/` 时，你的浏览器会向监听该地址的服务器发送一个 HTTP 请求（在这个例子中，是 Play 框架捆绑的 Netty 服务器）。请求是一个针对路由 `/` 的 GET 请求。Play 框架会在 `conf/routes` 中查找是否有满足 `/` 的路由：

```py
$ cat conf/routes
# Home page
GET     /                           controllers.Application.index
...
```

我们可以看到，`conf/routes` 文件确实包含了针对 GET 请求的路由 `/`。该行的第二部分，`controllers.Application.index`，是处理该路由的 Scala 函数的名称（稍后我们会详细讨论）。让我们进行实验。将路由端点更改为 `/hello`。刷新浏览器而不更改 URL。这将触发应用程序的重新编译。你现在应该看到一个错误页面：

![Play 框架](img/image01233.jpeg)

错误页面告诉你，应用程序不再有针对路由 `/` 的操作。如果你导航到 `127.0.0.1:9000/hello`，你应该再次看到着陆页面。

除了学习一点路由的工作原理外，我们还了解了关于开发 Play 应用程序的两大要点：

+   在开发模式下，当你刷新浏览器并且有代码变更时，代码会被重新编译

+   编译时和运行时错误会传播到网页

让我们将路由改回 `/`。关于路由还有很多要说的，但我们可以等到我们开始构建应用程序时再说。

`conf/routes` 文件告诉 Play 框架使用 `controllers.Application.index` 方法来处理对 `/` 的请求。让我们看看 `app/controllers` 中的 `Application.scala` 文件，其中定义了 `index` 方法：

```py
// app/controllers/Application.scala
package controllers

import play.api._
import play.api.mvc._

class Application extends Controller {

  def index = Action {
    Ok(views.html.index("Your new application is ready."))
  }

}
```

我们可以看到 `controllers.Application.index` 指的是 `Application` 类中的 `index` 方法。这个方法返回类型为 `Action`。`Action` 只是一个将 HTTP 请求映射到响应的函数。在详细解释之前，让我们将操作更改为：

```py
def index = Action {
  Ok("hello, world")
}
```

刷新浏览器，你应该会看到登录页面被替换为 `"hello world"`。通过让我们的操作返回 `Ok("hello, world")`，我们是在请求 Play 返回一个状态码为 200 的 HTTP 响应（表示请求成功）和正文 `"hello world"`。

让我们回到 `index` 的原始内容：

```py
Action {
  Ok(views.html.index("Your new application is ready.")) 
}
```

我们可以看到这调用了 `views.html.index` 方法。这看起来可能有些奇怪，因为任何地方都没有 `views` 包。但是，如果你查看 `app/views` 目录，你会注意到两个文件：`index.scala.html` 和 `main.scala.html`。这些是模板，在编译时，它们被转换成 Scala 函数。让我们看看 `main.scala.html`：

```py
// app/views/main.scala.html
@(title: String)(content: Html)

<!DOCTYPE html>

<html lang="en">
    <head>
        <title>@title</title>
        <!-- not so important stuff -->
    </head>
    <body>
        @content
    </body>
</html>
```

在编译时，这个模板被编译为 `views.html` 包中的 `main(title: String)(content: Html)` 函数。请注意，函数的包和名称来自模板文件名，而函数参数来自模板的第一行。模板包含嵌入的 `@title` 和 `@content` 值，这些值由传递给函数的参数填充。让我们在 Scala 控制台中实验一下：

```py
$ activator console
scala> import views.html._
import views.html._

scala> val title = "hello"
title: String = hello

scala> val content = new play.twirl.api.Html("<b>World</b>")
content: play.twirl.api.Html = <b>World</b>

scala> main(title)(content)
res8: play.twirl.api.HtmlFormat.Appendable =
<!DOCTYPE html>

<html lang="en">
 <head>
 <title>hello</title>
 <!-- not so important stuff -->
 </head>
 <body>
 <b>World</b>
 </body>
</html>

```

我们可以调用 `views.html.main`，就像我们调用一个普通的 Scala 函数一样。我们传递的参数被嵌入到由 `views/main.scala.html` 中的模板定义的正确位置。

这就结束了我们对 Play 的入门之旅。让我们简要回顾一下我们学到了什么：当一个请求到达 Play 服务器时，服务器读取 URL 和 HTTP 动词，并检查这些是否存在于其 `conf/routes` 文件中。然后，它将请求传递给为该路由定义的控制器中的 `Action`。这个 `Action` 返回一个 HTTP 响应，该响应被反馈到浏览器。在构建响应时，`Action` 可能会使用模板，对于它来说，模板只是一个 `(arguments list) => String` 或 `(arguments list) => HTML` 的函数。

# 动态路由

如我们所见，路由是将 HTTP 请求映射到 Scala 处理器。路由存储在 `conf/routes` 中。一个路由由一个 HTTP 动词、端点和一个 Scala 函数定义：

```py
// verb   // end-point              // Scala handler
GET       /                         controllers.Application.index
```

我们学会了通过向`routes`文件中添加行来添加新路由。然而，我们并不局限于静态路由。Play 框架允许我们在路由中包含通配符。通配符的值可以作为参数传递给控制器。为了了解这是如何工作的，让我们创建一个以人的名字作为参数的控制器。在`app.controllers`中的`Application`对象中添加：

```py
// app/controllers/Application.scala

class Application extends Controller {

  ...

  def hello(name:String) = Action {
    Ok(s"hello, $name")
  }
}
```

我们现在可以定义由该控制器处理的路由：

```py
// conf/routes
GET  /hello/:name             controllers.Application.hello(name)
```

如果你现在将浏览器指向`127.0.0.1:9000/hello/Jim`，你将在屏幕上看到**hello, Jim**。

任何在`:`和随后的`/`之间的字符串都被视为通配符：它将匹配任何字符组合。通配符的值可以传递给控制器。请注意，通配符可以出现在 URL 的任何位置，并且可以有多个通配符。以下都是有效的路由定义，例如：

```py
GET /hello/person-:name        controllers.Application.hello(name)
// ... matches /hello/person-Jim

GET /hello/:name/picture  controllers.Application.pictureFor(name)
// ... matches /hello/Jim/picture

GET /hello/:first/:last controllers.Application.hello(first, last)
// ... matches /hello/john/doe
```

选择路由并将参数传递给控制器有许多其他选项。请参阅 Play 框架的文档，以全面讨论路由的可能性：[`www.playframework.com/documentation/2.4.x/ScalaRouting`](https://www.playframework.com/documentation/2.4.x/ScalaRouting)。

### 提示

**URL 设计**

通常认为，将 URL 尽可能简化是最佳实践。URL 应反映网站信息的层次结构，而不是底层实现。GitHub 就是很好的例子：它的 URL 直观易懂。例如，这本书的仓库 URL 是：

[`github.com/pbugnion/s4ds`](https://github.com/pbugnion/s4ds)

要访问该仓库的问题页面，请在路由中添加`/issues`。要访问第一个问题，请在该路由中添加`/1`。这些被称为**语义 URL**([`en.wikipedia.org/wiki/Semantic_URL`](https://en.wikipedia.org/wiki/Semantic_URL))。

# 动作

我们已经讨论了路由以及如何向控制器传递参数。现在让我们谈谈我们可以用控制器做什么。

路由中定义的方法必须返回一个`play.api.mvc.Action`实例。`Action`类型是`Request[A] => Result`类型的薄包装，其中`Request[A]`标识一个 HTTP 请求，`Result`是 HTTP 响应。

## 组合响应

如我们在第七章中看到的，HTTP 响应由以下组成：

+   状态码（例如，成功响应的 200 或缺失页面的 404）

+   响应头，一个表示与响应相关的元数据的键值列表

+   响应正文。这可以是网页的 HTML，或 JSON、XML 或纯文本（或许多其他格式）。这通常是真正感兴趣的部分。

Play 框架定义了一个`play.api.mvc.Result`对象，它表示一个响应。该对象包含一个`header`属性，包含状态码和头信息，以及一个包含正文的`body`属性。

生成 `Result` 的最简单方法就是使用 `play.api.mvc.Results` 中的工厂方法之一。我们已经看到了 `Ok` 方法，它生成状态码为 200 的响应：

```py
def hello(name:String) = Action {
  Ok("hello, $name")
}
```

让我们退一步，打开一个 Scala 控制台，以便我们理解它是如何工作的：

```py
$ activator console
scala> import play.api.mvc._
import play.api.mvc._

scala> val res = Results.Ok("hello, world")
res: play.api.mvc.Result = Result(200, Map(Content-Type -> text/plain; charset=utf-8))

scala> res.header.status
Int = 200

scala> res.header.headers
Map[String,String] = Map(Content-Type -> text/plain; charset=utf-8)

scala> res.body
play.api.libs.iteratee.Enumerator[Array[Byte]] = play.api.libs.iteratee.Enumerator$$anon$18@5fb83873

```

我们可以看到 `Results.Ok(...)` 是如何创建一个状态为 `200` 的 `Result` 对象，在这个例子中，它包含一个表示内容类型的单个头信息。体部分稍微复杂一些：它是一个枚举器，当需要时可以推送到输出流。枚举器包含传递给 `Ok` 的参数：在这个例子中是 `"hello, world"`。

`Results` 中有许多用于返回不同状态码的工厂方法。其中一些更相关的如下：

+   `Action { Results.NotFound }`

+   `Action { Results.BadRequest("bad request") }`

+   `Action { Results.InternalServerError("error") }`

+   `Action { Results.Forbidden }`

+   `Action { Results.Redirect("/home") }`

要获取 `Result` 工厂方法的完整列表，请参阅 Results 的 API 文档 ([`www.playframework.com/documentation/2.4.x/api/scala/index.html#play.api.mvc.Results`](https://www.playframework.com/documentation/2.4.x/api/scala/index.html#play.api.mvc.Results))。

到目前为止，我们一直限制自己只将字符串作为 `Ok` 结果的内容传递：`Ok("hello, world")`。然而，我们并不局限于传递字符串。我们可以传递一个 JSON 对象：

```py
scala> import play.api.libs.json._
import play.api.libs.json._

scala> val jsonObj = Json.obj("hello" -> "world")
jsonObj: play.api.libs.json.JsObject = {"hello":"world"}

scala> Results.Ok(jsonObj)
play.api.mvc.Result = Result(200, Map(Content-Type -> application/json; charset=utf-8))

```

当我们开始构建 API 时，我们将更详细地介绍与 JSON 的交互。我们也可以传递 HTML 作为内容。这通常是在返回视图时的情况。

```py
scala> val htmlObj = views.html.index("hello")
htmlObj: play.twirl.api.HtmlFormat.Appendable =

<!DOCTYPE html>

<html lang="en">
 <head>
...

scala> Results.Ok(htmlObj)
play.api.mvc.Result = Result(200, Map(Content-Type -> text/html; charset=utf-8))

```

注意 `Content-Type` 头是如何根据传递给 `Ok` 的内容类型设置的。`Ok` 工厂使用 `Writeable` 类型类将它的参数转换为响应体。因此，对于任何存在 `Writeable` 类型类的类型，都可以用作 `Ok` 的参数。如果你对类型类不熟悉，你可能想阅读第五章 *使用类型类进行松耦合* 的部分，*Scala 和 SQL 通过 JDBC*。

## 理解和解析请求

我们现在知道了如何制定（基本的）响应。等式的另一半是 HTTP 请求。回想一下，`Action` 只是一个将 `Request => Result` 映射的函数。我们可以使用以下方式访问请求：

```py
def hello(name:String) = Action { request => 
  ...
}
```

需要引用请求的一个原因是访问查询字符串中的参数。让我们修改我们之前写的 `Hello, <name>` 示例，使其可选地包含查询字符串中的标题。因此，一个 URL 可以格式化为 `/hello/Jim?title=Dr`。`request` 实例公开了 `getQueryString` 方法，用于访问查询字符串中的特定键。如果键存在于查询中，该方法返回 `Some[String]`，否则返回 `None`。我们可以将我们的 `hello` 控制器重写为：

```py
def hello(name:String) = Action { request =>
  val title = request.getQueryString("title")
  val titleString = title.map { _ + " " }.getOrElse("")
  Ok(s"Hello, $titleString$name")
}
```

通过在浏览器中访问 URL `127.0.0.1:9000/hello/Odersky?title=Dr` 来尝试这个示例。浏览器应该显示 `Hello, Dr Odersky`。

到目前为止，我们一直专注于 GET 请求。这些请求没有正文。其他类型的 HTTP 请求，最常见的是 POST 请求，确实包含正文。Play 允许用户在定义操作时传递 *正文解析器*。请求正文将通过正文解析器传递，它将将其从字节流转换为 Scala 类型。作为一个非常简单的例子，让我们定义一个新的路由，它接受 POST 请求：

```py
POST      /hello            controllers.Application.helloPost
```

我们将预定义的 `parse.text` 正文解析器应用于传入的请求正文。这会将请求正文转换为字符串。`helloPost` 控制器看起来像：

```py
def helloPost = Action(parse.text) { request =>
  Ok("Hello. You told me: " + request.body)
}
```

### 小贴士

在浏览器中轻松测试 POST 请求是不可能的。您可以使用 cURL。cURL 是一个用于发送 HTTP 请求的命令行实用程序。它在 Mac OS 上默认安装，并且应该可以通过 Linux 发行版的包管理器获得。以下示例将发送一个正文为 `"I think that Scala is great"` 的 POST 请求：

```py
$ curl --data "I think that Scala is great" --header "Content-type:text/plain"  127.0.0.1:9000/hello
```

这将在终端打印以下行：

`Hello. You told me: I think that Scala is great`

有几种内置的正文解析器：

+   `parse.file(new File("filename.txt"))` 将正文保存到文件中。

+   `parse.json` 将正文解析为 JSON（我们将在下一节中了解更多关于与 JSON 交互的内容）。

+   `parse.xml` 将正文解析为 XML。

+   `parse.urlFormEncoded` 将解析由提交 HTML 表单返回的正文。`request.body` 属性是一个从 `String` 到 `Seq[String]` 的 Scala 映射，将每个表单元素映射到其值（们）。

要获取正文解析器的完整列表，最佳来源是 `play.api.mvc.BodyParsers.parse` 的 Scala API 文档，可在以下网址找到：[`www.playframework.com/documentation/2.5.x/api/scala/index.html#play.api.mvc.BodyParsers$parse$`](https://www.playframework.com/documentation/2.5.x/api/scala/index.html#play.api.mvc.BodyParsers%24parse%24)。

# 与 JSON 交互

如我们在前面的章节中所发现的，JSON 正在成为通过 HTTP 通信结构化数据的默认语言。如果您开发一个 Web 应用程序或 Web API，您可能需要消费或发射 JSON，或者两者都要。

在 第七章，*Web APIs* 中，我们学习了如何通过 `json4s` 解析 JSON。Play 框架包括它自己的 JSON 解析器和发射器。幸运的是，它的行为与 `json4s` 非常相似。

让我们假设我们正在构建一个总结 GitHub 仓库信息的 API。当查询特定用户时，我们的 API 将输出一个 JSON 数组，列出该用户的仓库（类似于 GitHub API，但只包含部分字段）。

让我们先定义一个仓库的模型。在 Play 应用程序中，模型通常存储在 `app/models` 文件夹中的 `models` 包下：

```py
// app/models/Repo.scala

package models

case class Repo (
  val name:String,
  val language:String,
  val isFork: Boolean,
  val size: Long
)
```

让我们在应用程序中添加一个路由，为特定用户提供仓库数组。在 `conf/routes` 中添加以下行：

```py
// conf/routes
GET   /api/repos/:username       controllers.Api.repos(username)
```

现在我们来实现控制器的框架。我们将为我们的 API 创建一个新的控制器，暂时命名为 `Api`。目前，我们只是让控制器返回一些示例数据。代码如下（我们将在稍后解释细节）：

```py
// app/controllers/Api.scala
package controllers
import play.api._
import play.api.mvc._
import play.api.libs.json._

import models.Repo

class Api extends Controller {

  // Some dummy data.
  val data = ListRepo,
    Repo("frontend", "JavaScript", true, 392)
  )

  // Typeclass for converting Repo -> JSON
  implicit val writesRepos = new Writes[Repo] {
    def writes(repo:Repo) = Json.obj(
      "name" -> repo.name,
      "language" -> repo.language,
      "is_fork" -> repo.isFork,
      "size" -> repo.size
    )
  }

  // The controller
  def repos(username:String) = Action {

    val repoArray = Json.toJson(data) 
    // toJson(data) relies on existence of 
    // `Writes[List[Repo]]` type class in scope

    Ok(repoArray)
  }
}
```

如果你将你的网络浏览器指向 `127.0.0.1:9000/api/repos/odersky`，你现在应该看到以下 JSON 对象：

```py
[{"name":"dotty","language":"Scala","is_fork":true,"size":14315},{"name":"frontend","language":"JavaScript","is_fork":true,"size":392}]
```

这段代码中唯一棘手的部分是将 `Repo` 转换为 JSON。我们在 `data` 上调用 `Json.toJson`，它是 `List[Repo]` 类型的实例。`toJson` 方法依赖于传递给它的类型 `T` 存在的类型类 `Writes[T]`。

Play 框架广泛使用类型类来定义如何将模型转换为特定格式。回想一下，我们学习了如何在 SQL 和 MongoDB 的上下文中编写类型类。Play 框架的期望非常相似：为了使 `Json.toJson` 方法能够在 `Repo` 类型的实例上工作，必须有 `Writes[Repo]` 的实现可用，该实现指定了如何将 `Repo` 对象转换为 JSON。

在 Play 框架中，`Writes[T]` 类型类定义了一个单一的方法：

```py
trait Writes[T] {
  def writes(obj:T):Json
}
```

`Writes` 方法为内置简单类型和集合已经内置到 Play 框架中，因此我们不需要担心定义 `Writes[Boolean]` 等。

`Writes[Repo]` 实例通常直接在控制器中定义，如果它仅用于该控制器，或者在 `Repo` 伴生对象中定义，这样它就可以在多个控制器中使用。为了简单起见，我们只是将其嵌入到控制器中。

注意类型类如何实现关注点的分离。模型仅定义了 `Repo` 类型，而没有附加任何行为。`Writes[Repo]` 类型类只知道如何将 `Repo` 实例转换为 JSON，但不知道它在什么上下文中被使用。最后，控制器只知道如何创建 JSON HTTP 响应。

恭喜你，你刚刚定义了一个返回 JSON 的 Web API！在下一节中，我们将学习如何从 GitHub Web API 获取数据，以避免不断返回相同的数组。

# 查询外部 API 和消费 JSON

到目前为止，我们已经学会了如何响应 `/api/repos/:username` 的请求，向用户提供一个示例 JSON 数组形式的仓库。在本节中，我们将用从 GitHub 下载的实际仓库数据替换示例数据。

在 第七章，*Web APIs* 中，我们学习了如何使用 Scala 的 `Source.fromURL` 方法以及 `scalaj-http` 查询 GitHub API。Play 框架实现自己的库以与外部 Web 服务交互应该不会让人感到惊讶。

让我们编辑 `Api` 控制器以从 GitHub 获取有关用户仓库的信息，而不是使用示例数据。当以用户名作为参数调用时，控制器将：

1.  向 GitHub API 发送 GET 请求以获取该用户的仓库信息。

1.  解释响应，将体从 JSON 对象转换为 `List[Repo]`。

1.  将 `List[Repo]` 转换为 JSON 数组，形成响应。

我们首先给出完整的代码列表，然后再详细解释更复杂的部分：

```py
// app/controllers/Api.scala

package controllers

import play.api._
import play.api.mvc._
import play.api.libs.ws.WS // query external APIs
import play.api.Play.current
import play.api.libs.json._ // parsing JSON
import play.api.libs.functional.syntax._
import play.api.libs.concurrent.Execution.Implicits.defaultContext

import models.Repo

class Api extends Controller {

  // type class for Repo -> Json conversion
  implicit val writesRepo = new Writes[Repo] {
    def writes(repo:Repo) = Json.obj(
      "name" -> repo.name,
      "language" -> repo.language,
      "is_fork" -> repo.isFork,
      "size" -> repo.size
    )
  }

  // type class for Github Json -> Repo conversion
  implicit val readsRepoFromGithub:Reads[Repo] = (
    (JsPath \ "name").read[String] and
    (JsPath \ "language").read[String] and
    (JsPath \ "fork").read[Boolean] and
    (JsPath \ "size").read[Long]
  )(Repo.apply _)

  // controller
  def repos(username:String) = Action.async {

    // GitHub URL
    val url = s"https://api.github.com/users/$username/repos"
    val response = WS.url(url).get() // compose get request

    // "response" is a Future
    response.map { r =>
      // executed when the request completes
      if (r.status == 200) {

        // extract a list of repos from the response body
        val reposOpt = Json.parse(r.body).validate[List[Repo]]
        reposOpt match {
          // if the extraction was successful:
          case JsSuccess(repos, _) => Ok(Json.toJson(repos))

          // If there was an error during the extraction
          case _ => InternalServerError
        }
      }
      else {
        // GitHub returned something other than 200
        NotFound
      }

    }
  }

}
```

如果您已经编写了所有这些，请将浏览器指向例如 `127.0.0.1:9000/api/repos/odersky` 来查看 Martin Odersky 拥有的仓库列表：

```py
[{"name":"dotty","language":"Scala","is_fork":true,"size":14653},{"name":"frontend","language":"JavaScript","is_fork":true,"size":392},...
```

这个代码示例内容很多，所以让我们将其分解。

## 调用外部网络服务

查询外部 API 的第一步是导入 `WS` 对象，它定义了创建 HTTP 请求的工厂方法。这些工厂方法依赖于命名空间中隐含的 Play 应用程序的引用。确保这种情况的最简单方法是导入 `play.api.Play.current`，这是对当前应用程序的引用。

现在让我们忽略 `readsRepoFromGithub` 类型类，直接跳到控制器主体。我们想要通过 GET 请求访问的 URL 是 `"https://api.github.com/users/$username/repos"`，其中 `$username` 是适当的值。我们使用 `WS.url(url).get()` 创建一个 GET 请求。我们还可以向现有请求添加头信息。例如，为了指定内容类型，我们可以这样写：

```py
WS.url(url).withHeaders("Content-Type" -> "application/json").get()
```

我们可以使用头信息通过以下方式传递 GitHub OAuth 令牌：

```py
val token = "2502761d..."
WS.url(url).withHeaders("Authorization" -> s"token $token").get()
```

要形成 POST 请求而不是 GET 请求，将最后的 `.get()` 替换为 `.post(data)`。在这里，`data` 可以是 JSON、XML 或字符串。

添加 `.get` 或 `.post` 触发请求，返回一个 `Future[WSResponse]`。到现在为止，您应该熟悉 futures。通过编写 `response.map { r => ... }`，我们指定在 future 结果返回时要执行的可转换操作，该操作验证响应的状态，如果响应的状态码不是 200，则返回 `NotFound`。

## 解析 JSON

如果状态码是 200，回调将解析响应体为 JSON，并将解析后的 JSON 转换为 `List[Repo]` 实例。我们已经知道如何使用 `Writes[Repo]` 类型类将 `Repo` 对象转换为 JSON。反过来，从 JSON 到 `Repo` 对象的转换要复杂一些，因为我们必须考虑格式不正确的 JSON。为此，Play 框架在 JSON 对象上提供了 `.validate[T]` 方法。此方法尝试将 JSON 转换为类型 `T` 的实例，如果 JSON 格式良好，则返回 `JsSuccess`，否则返回 `JsError`（类似于 Scala 的 `Try` 对象）。`.validate` 方法依赖于类型类 `Reads[Repo]` 的存在。让我们在 Scala 控制台中实验一下：

```py
$ activator console

scala> import play.api.libs.json._
import play.api.libs.json._

scala> val s = """ 
 { "name": "dotty", "size": 150, "language": "Scala", "fork": true }
"""
s: String = "
 { "name": "dotty", "size": 150, "language": "Scala", "fork": true }
"

scala> val parsedJson = Json.parse(s)
parsedJson: play.api.libs.json.JsValue = {"name":"dotty","size":150,"language":"Scala","fork":true}

```

使用 `Json.parse` 将字符串转换为 `JsValue` 实例，它是 JSON 实例的超类型。我们可以使用类似 XPath 的语法访问 `parsedJson` 中的特定字段（如果您不熟悉类似 XPath 的语法，您可能想阅读第六章，*Slick – 一个 SQL 的函数式接口*）：

```py
scala> parsedJson \ "name"
play.api.libs.json.JsLookupResult = JsDefined("dotty")

```

XPath 类似的查找返回一个类型为 `JsLookupResult` 的实例。它包含两个值：如果路径有效，则为 `JsDefined`，如果无效，则为 `JsUndefined`：

```py
scala> parsedJson \ "age"
play.api.libs.json.JsLookupResult = JsUndefined('age' is undefined on object: {"name":"dotty","size":150,"language":"Scala","fork":true})

```

要以类型安全的方式将 `JsLookupResult` 实例转换为 String，我们可以使用 `.validate[String]` 方法：

```py
scala> (parsedJson \ "name").validate[String]
play.api.libs.json.JsResult[String] = JsSuccess(dotty,) 

```

`.validate[T]` 方法返回 `JsSuccess`，如果 `JsDefined` 实例可以被成功转换为 `T`，否则返回 `JsError`。为了说明后者，让我们尝试将其验证为 `Int`：

```py
scala> (parsedJson \ "name").validate[Int]
dplay.api.libs.json.JsResult[Int] = JsError(List((,List(ValidationError(List(error.expected.jsnumber),WrappedArray())))))

```

在 `JsUndefined` 实例上调用 `.validate` 也会返回 `JsError`：

```py
scala> (parsedJson \ "age").validate[Int]
play.api.libs.json.JsResult[Int] = JsError(List((,List(ValidationError(List('age' is undefined on object: {"name":"dotty","size":150,"language":"Scala","fork":true}),WrappedArray())))))

```

要将 `JsResult[T]` 实例转换为类型 `T` 的实例，我们可以使用模式匹配：

```py
scala> val name = (parsedJson \ "name").validate[String] match {
 case JsSuccess(n, _) => n
 case JsError(e) => throw new IllegalStateException(
 s"Error extracting name: $e")
}
name: String = dotty

```

我们现在可以使用 `.validate` 以类型安全的方式将 JSON 转换为简单类型。但在代码示例中，我们使用了 `.validate[Repo]`。这只有在命名空间中隐式可用 `Reads[Repo]` 类型类时才有效。

定义 `Reads[T]` 类型类最常见的方式是通过在 `import play.api.libs.functional.syntax._` 中提供的 DSL（领域特定语言）。该 DSL 通过链式操作返回 `JsSuccess` 或 `JsError` 来工作。具体讨论这个 DSL 的工作原理超出了本章的范围（例如，可以参考 Play 框架关于 JSON 组合器的文档页面：[`www.playframework.com/documentation/2.4.x/ScalaJsonCombinators`](https://www.playframework.com/documentation/2.4.x/ScalaJsonCombinators)）。我们将坚持讨论语法。

```py
scala> import play.api.libs.functional.syntax._
import play.api.libs.functional.syntax._

scala> import models.Repo
import models.Repo

scala> implicit val readsRepoFromGithub:Reads[Repo] = (
 (JsPath \ "name").read[String] and
 (JsPath \ "language").read[String] and
 (JsPath \ "fork").read[Boolean] and
 (JsPath \ "size").read[Long]
)(Repo.apply _)
readsRepoFromGithub: play.api.libs.json.Reads[models.Repo] = play.api.libs.json.Reads$$anon$8@a198ddb

```

`Reads` 类型类分为两个阶段定义。第一阶段通过 `read[T]` 方法与 `and` 链接起来，结合成功和错误。第二阶段使用案例类（或 `Tuple` 实例）的伴生对象的 `apply` 方法来构建对象，前提是第一阶段成功完成。现在我们已经定义了类型类，我们可以在 `JsValue` 对象上调用 `validate[Repo]`：

```py
scala> val repoOpt = parsedJson.validate[Repo]
play.api.libs.json.JsResult[models.Repo] = JsSuccess(Repo(dotty,Scala,true,150),)

```

然后，我们可以使用模式匹配从 `JsSuccess` 实例中提取 `Repo` 对象：

```py
scala> val JsSuccess(repo, _) = repoOpt
repo: models.Repo = Repo(dotty,Scala,true,150)

```

到目前为止，我们只讨论了验证单个仓库。Play 框架为集合类型定义了类型类，因此，如果 `Reads[Repo]` 被定义，`Reads[List[Repo]]` 也会被定义。

现在我们已经了解了如何从 JSON 中提取 Scala 对象，让我们回到代码。如果我们能够成功将仓库转换为 `List[Repo]`，我们再次将其作为 JSON 发射。当然，将 GitHub 的仓库 JSON 表示转换为 Scala 对象，然后直接转换为我们的对象 JSON 表示可能看起来很复杂。然而，如果这是一个真实的应用程序，我们会有额外的逻辑。例如，我们可以将仓库存储在缓存中，并尝试从缓存中获取而不是查询 GitHub API。尽早将 JSON 转换为 Scala 对象可以解耦我们编写的代码与 GitHub 返回仓库的方式。

## 异步操作

代码示例中新增的最后部分是调用 `Action.async`，而不是仅仅 `Action`。回想一下，`Action` 实例是 `Request => Result` 方法的薄包装。然而，我们的代码返回一个 `Future[Result]`，而不是 `Result`。在这种情况下，使用 `Action.async` 来构建动作，而不是直接使用 `Action`。使用 `Action.async` 告诉 Play 框架，创建 `Action` 的代码是异步的。

# 使用 Play 创建 API：总结

在最后一节中，我们部署了一个响应 GET 请求的 API。由于这需要很多理解，让我们总结一下如何进行 API 创建：

1.  在 `/conf/routes` 中定义适当的路由，根据需要使用 URL 中的通配符。

1.  在 `/app/models` 中创建 Scala 的案例类来表示 API 使用的模型。

1.  创建 `Write[T]` 方法，以便将模型写入 JSON 或 XML，这样它们就可以通过 API 返回。

1.  将路由绑定到控制器。如果控制器需要做更多的工作，将工作包装在 future 中以避免阻塞服务器。

Play 框架中还有许多你可能需要的更有用的组件，例如，例如如何使用 Slick 访问 SQL 数据库。不幸的是，我们没有时间在这篇介绍中涵盖这些内容。Play 框架有详尽、写得很好的文档，将填补这篇教程中的空白。

# Rest API：最佳实践

随着互联网的成熟，REST（表示状态转移）API 正在成为网络 API 最可靠的设计模式。如果一个 API 遵循以下指导原则，则称为 *RESTful*：

+   API 被设计为一组资源。例如，GitHub API 提供了关于用户、仓库、关注者等信息。每个用户或仓库都是一个特定的资源。每个资源都可以通过不同的 HTTP 端点进行访问。

+   网址应当简洁明了，并能清楚地标识资源。例如，`api.github.com/users/odersky` 就很简单，清楚地告诉我们应该期待关于用户马丁·奥德斯基的信息。

+   没有一个包含系统所有信息的 *全局资源*。相反，顶级资源包含指向更专业资源的链接。例如，GitHub API 中的用户资源包含指向该用户仓库和该用户关注者的链接，而不是直接在用户资源中嵌入所有这些信息。

+   API 应该是可发现的。对特定资源的请求的响应应包含相关资源的 URL。当你查询 GitHub 上的用户资源时，响应包含访问该用户关注者、仓库等的 URL。客户端应使用 API 提供的 URL，而不是尝试在客户端构建它们。这使客户端对 API 的变化更加稳健。

+   应尽可能在服务器上保持最少的状态。例如，在查询 GitHub API 时，我们必须在每次请求中传递认证令牌，而不是期望我们的认证状态在服务器上被*记住*。使每次交互独立于历史记录提供了更好的可扩展性：如果任何交互可以由任何服务器处理，负载均衡就更容易实现。

# 摘要

在本章中，我们介绍了 Play 框架作为构建 Web API 的工具。我们构建了一个返回用户 GitHub 仓库 JSON 数组的 API。在下一章中，我们将在此基础上构建 API，并构建一个单页应用程序来图形化表示这些数据。

# 参考资料

+   以下维基百科页面提供了关于语义 URL 的信息：[`zh.wikipedia.org/wiki/%E8%AF%AD%E4%B9%89_URL`](https://zh.wikipedia.org/wiki/%E8%AF%AD%E4%B9%89_URL) 和 [`apiux.com/2013/04/03/url-design-restful-web-services/`](http://apiux.com/2013/04/03/url-design-restful-web-services/)。

+   对于 Play 框架的更深入讨论，我建议阅读*Julien Richard-Foy*的《Play 框架基础》。

+   《实践中的 REST：超媒体和系统架构》，作者*Jim Webber*、*Savas Parastatidis*和*Ian Robinson*，描述了如何设计 REST API。

# 第十四章. 使用 D3 和 Play 框架进行可视化

在上一章中，我们学习了 Play 框架，一个 Scala 的 Web 框架。我们构建了一个返回描述用户 GitHub 仓库的 JSON 数组的 API。

在本章中，我们将构建一个完整的 Web 应用程序，显示一个表格和一个图表，描述用户的仓库。我们将学习如何将**D3.js**，一个用于构建数据驱动 Web 页面的 JavaScript 库，与 Play 框架集成。这将使你走上构建引人注目的交互式可视化之路，展示使用机器学习获得的结果。

本章假设你已经熟悉 HTML、CSS 和 JavaScript。我们将在本章末尾提供参考资料。你还应该阅读上一章。

# GitHub 用户数据

我们将构建一个单页应用程序，其后端使用上一章开发的 API。该应用程序包含一个表单，用户可以输入 GitHub 账户的登录名。应用程序查询 API 以获取该用户的仓库列表，并在屏幕上以表格和饼图的形式显示，总结该用户使用的编程语言：

![GitHub 用户数据](img/image01234.jpeg)

要查看应用程序的实时版本，请访问 [`app.scala4datascience.com`](http://app.scala4datascience.com)。

# 我需要后端吗？

在上一章中，我们学习了支撑互联网工作方式的客户端-服务器模型：当你在你浏览器中输入一个网站 URL 时，服务器会向你的浏览器提供 HTML、CSS 和 JavaScript，然后浏览器以适当的方式渲染它们。

这对你意味着什么？当构建 Web 应用时，你应考虑的第二个问题可能是你是否需要进行任何服务器端处理（在“这真的值得付出努力吗？”之后）。你能否仅仅创建一个带有一些 JavaScript 的 HTML 网页？ 

如果构建整个应用所需的数据足够小，你就可以不使用后端：通常只有几兆字节。如果你的应用更大，你需要一个后端来传输客户端当前需要的仅有的数据。令人惊讶的是，你通常可以在没有后端的情况下构建可视化：虽然数据科学通常习惯于处理 TB 级别的数据，但数据科学流程的目标通常是将这些庞大的数据集压缩成几个有意义的数字。

拥有后端还可以让你包含对客户端不可见的逻辑。如果你需要验证密码，显然你不能将执行该操作的代码发送到客户端计算机：它需要在看不见的地方，在服务器上完成。

如果你的应用足够小，且不需要进行任何服务器端处理，就停止阅读这一章，如果你需要的话，复习一下 JavaScript，现在先别考虑 Scala。不必担心构建后端会使你的生活更轻松。

然而，显然我们并没有为我们要构建的应用拥有这样的自由度：用户可以输入 GitHub 上任何人的名字。查找关于该用户的信息需要后端访问巨大的存储和查询能力（我们通过仅将请求转发到 GitHub API 并重新解释响应来模拟）。

# 通过 web-jars 使用 JavaScript 依赖项

开发 Web 应用的一个挑战是，我们需要编写两个近乎独立的程序：服务器端程序和客户端程序。这些程序通常需要不同的技术。特别是，对于任何非最简单应用，我们必须跟踪 JavaScript 库，并在构建过程中集成处理 JavaScript 代码（例如，进行压缩）。

Play 框架通过*web-jars*管理 JavaScript 依赖项。这些只是打包成 jar 文件的 JavaScript 库。它们部署在 Maven Central 上，这意味着我们只需将它们添加到我们的`build.sbt`文件中作为依赖项。对于这个应用，我们需要以下 JavaScript 库：

+   Require.js，一个用于编写模块化 JavaScript 的库

+   JQuery

+   Bootstrap

+   Underscore.js，一个添加了许多功能构造和客户端模板的库。

+   D3，一个图形绘图库

+   NVD3，一个基于 D3 构建的图形库

如果您计划编写本章提供的示例，最简单的方法是从上一章的代码开始（您可以从 GitHub 下载第十三章的代码，*使用 Play 的 Web API*：[`github.com/pbugnion/s4ds/tree/master/chap13`](https://github.com/pbugnion/s4ds/tree/master/chap13)）。从现在起，我们将假设这是一个起点。

让我们在`build.sbt`文件中包含对 web-jars 的依赖项：

```py
libraryDependencies ++= Seq(
  "org.webjars" % "requirejs" % "2.1.22",
  "org.webjars" % "jquery" % "2.1.4",
  "org.webjars" % "underscorejs" % "1.8.3",
  "org.webjars" % "nvd3" % "1.8.1",
  "org.webjars" % "d3js" % "3.5.6",
  "org.webjars" % "bootstrap" % "3.3.6"
)
```

通过运行`activator update`来获取模块。一旦完成此操作，您将注意到`target/web/public/main/lib`中的 JavaScript 库。

# 迈向网络应用：HTML 模板

在上一章中，我们简要介绍了如何在 HTML 文件中交错 Scala 片段来构建 HTML 模板的方法。我们了解到模板被编译为 Scala 函数，并学习了如何从控制器中调用这些函数。

在单页应用中，控制浏览器实际显示的大多数逻辑位于客户端 JavaScript 中，而不是服务器端。服务器提供的页面包含基本的 HTML 框架。

让我们为我们的应用程序创建 HTML 布局。我们将将其保存在`views/index.scala.html`中。该模板将仅包含应用程序的布局，但不会包含任何关于任何用户仓库的信息。要获取这些信息，应用程序必须查询上一章开发的 API。该模板不接收任何参数，因为所有动态 HTML 生成都将发生在客户端。

我们使用 Bootstrap 网格布局来控制 HTML 布局。如果您不熟悉 Bootstrap 布局，请查阅[`getbootstrap.com/css/#grid-example-basic`](http://getbootstrap.com/css/#grid-example-basic)的文档。

```py
// app/views/index.scala.html
<!DOCTYPE html>

<html lang="en">
  <head>
    <title>Github User display</title>
    <link rel="stylesheet" media="screen" 
      href="@routes.Assets.versioned("stylesheets/main.css")">
    <link rel="shortcut icon" type="image/png"
      href="@routes.Assets.versioned("images/favicon.png")">
    <link rel="stylesheet" media="screen" 
      href=@routes.Assets.versioned("lib/nvd3/nv.d3.css") >
    <link rel="stylesheet" media="screen"
      href=@routes.Assets.versioned(
      "lib/bootstrap/css/bootstrap.css")>
  </head>

  <body>
    <div class="container">

      <!-- Title row -->
      <div class="row">
        <h1>Github user search</h1>
      </div>

      <!-- User search row -->
      <div class="row">
        <label>Github user: </label>
        <input type="text" id="user-selection">
        <span id="searching-span"></span> <hr />
      </div>

      <!-- Results row -->
      <div id="response" class="row"></div>
    </div>
  </body>
</html>
```

在 HTML 头部，我们链接应用程序所需的 CSS 样式表。我们不是明确指定路径，而是使用`@routes.Assets.versioned(...)`函数。这解析为编译后资产存储位置的 URI。传递给函数的参数应该是从`target/web/public/main`到所需资产的路径。

当用户访问我们的服务器上的路由`/`时，我们希望提供此视图的编译版本。因此，我们需要将此路由添加到`conf/routes`中：

```py
# conf/routes
GET   /      controllers.Application.index
```

路由由`Application`控制器中的`index`函数提供。此控制器需要做的只是提供`index`视图：

```py
// app/controllers/Application.scala
package controllers

import play.api._
import play.api.mvc._

class Application extends Controller {

  def index = Action {
    Ok(views.html.index())
  }
}
```

通过在应用程序根目录中运行`activator run`来启动 Play 框架，并将您的网络浏览器指向`127.0.0.1:9000/`。您应该看到我们的网络应用程序框架。当然，应用程序目前还没有做任何事情，因为我们还没有编写任何 JavaScript 逻辑。

![迈向网络应用：HTML 模板](img/image01235.jpeg)

# 通过 RequireJS 模块化 JavaScript

将 JavaScript 库注入命名空间的最简单方法是将它们添加到 HTML 框架中，通过 HTML 头部的`<script>...</script>`标签。例如，要添加 JQuery，我们会在文档的头部添加以下行：

```py
<script src=@routes.Assets.versioned("lib/jquery/jquery.js") type="text/javascript"></script>
```

虽然这可行，但它不适合大型应用程序的扩展，因为每个库都被导入到全局命名空间中。现代客户端 JavaScript 框架，如 AngularJS，提供了一种定义和加载模块的替代方法，这有助于保持封装性。

我们将使用 RequireJS。简而言之，RequireJS 让我们可以通过函数封装 JavaScript 模块。例如，如果我们想编写一个包含用于隐藏`div`的函数的模块`example`，我们会按照以下方式定义该模块：

```py
// example.js
define(["jquery", "underscore"], function($, _) {

  // hide a div
  function hide(div_name) {
    $(div_name).hide() ;
  }

  // what the module exports.
  return { "hide": hide }

}) ;
```

我们将我们的模块封装在一个名为`define`的函数的回调中。`define`函数接受两个参数：一个依赖项列表和一个函数定义。`define`函数将依赖项绑定到回调的参数列表：在这种情况下，JQuery 中的函数将被绑定到`$`，而 Underscore 中的函数将被绑定到`_`。这创建了一个模块，它暴露了回调函数返回的任何内容。在这种情况下，我们导出`hide`函数，并将其绑定到名称`"hide"`。因此，我们的示例模块暴露了`hide`函数。

要加载此模块，我们将它作为依赖项传递给我们要在其中使用它的模块：

```py
define(["example"], function(example) {

  function hide_all() {
 example.hide("#top") ;
 example.hide("#bottom") ;
  }

  return { "hide_all": hide_all } ;
});
```

注意`example`中的函数是如何被封装的，而不是存在于全局命名空间中。我们通过`example.<function-name>`来调用它们。此外，在`example`模块内部定义的任何函数或变量都保持私有。

有时，我们希望 JavaScript 代码存在于模块之外。对于启动应用程序的脚本来说，这通常是情况。对于这些脚本，将`define`替换为`require`：

```py
require(["jquery", "example"], function($, example) {
  $(document).ready(function() {
    example.hide("#header") ;
  });
}) ;
```

现在我们已经对 RequireJS 有了概述，那么我们如何在 Play 框架中使用它？第一步是添加对 RequireJS web jar 的依赖，我们已经这样做了。Play 框架还添加了一个 RequireJS SBT 插件([`github.com/sbt/sbt-rjs`](https://github.com/sbt/sbt-rjs))，如果您使用了`play-scala`激活器，则默认应该已安装。如果缺少此插件，可以在`plugins.sbt`文件中添加以下行：

```py
// project/plugins.sbt

addSbtPlugin("com.typesafe.sbt" % "sbt-rjs" % "1.0.7")
```

我们还需要将插件添加到阶段列表中。这允许插件在打包应用程序为 jar 时操作 JavaScript 资源。将以下行添加到`build.sbt`文件中：

```py
pipelineStages := Seq(rjs)
```

您需要重新启动激活器以使更改生效。

我们现在已准备好在我们的应用程序中使用 RequireJS。我们可以在视图的头部部分添加以下行来使用它：

```py
// index.scala.html

<html>
  <head>
...

    <script
      type="text/javascript"
      src=@routes.Assets.versioned("lib/requirejs/require.js").url
      data-main=@routes.Assets.versioned("javascripts/main.js").url>
    </script>

  </head>
...
</html>
```

当视图被编译时，这会被解析为类似以下标签：

```py
<script type="text/javascript" 
  data-main="/assets/javascripts/main.js" 
  src="img/require.min.js">
</script>
```

传递给 `data-main` 的参数是应用程序的入口点。当 RequireJS 加载时，它将执行 `main.js`。因此，该脚本必须引导我们的应用程序。特别是，它应该包含一个用于 RequireJS 的配置对象，使其知道所有库的位置。

# 引导应用程序

当我们将 `require.js` 链接到我们的应用程序时，我们告诉它使用 `main.js` 作为我们的入口点。为了测试这是否工作，让我们首先输入一个虚拟的 `main.js`。Play 应用程序中的 JavaScript 文件位于 `/public/javascripts`：

```py
// public/javascripts/main.js

require([], function() {
  console.log("hello, JavaScript"); 
});
```

为了验证这已经工作，转到 `127.0.0.1:9000` 并打开浏览器控制台。你应该在控制台中看到 `"hello, JavaScript"`。

现在我们来编写一个更有用的 `main.js`。我们首先配置 RequireJS，给它提供我们将在应用程序中使用的模块的位置。不幸的是，我们使用的图形库 NVD3 与 RequireJS 不是很兼容，所以我们不得不使用一个丑陋的技巧来使其工作。这使我们的 `main.js` 文件变得有些复杂：

```py
// public/javascripts/main.js

(function (requirejs) {
  'use strict';

  // -- RequireJS config --
  requirejs.config({
    // path to the web jars. These definitions allow us 
    // to use "jquery", rather than "../lib/jquery/jquery",
    // when defining module dependencies.
    paths: {
      "jquery": "../lib/jquery/jquery",
      "underscore": "../lib/underscorejs/underscore",
      "d3": "../lib/d3js/d3",
      "nvd3": "../lib/nvd3/nv.d3",
      "bootstrap": "../lib/bootstrap/js/bootstrap"
    },

    shim: {
      // hack to get nvd3 to work with requirejs.
      // see this so question:
      // http://stackoverflow.com/questions/13157704/how-to-integrate-d3-with-require-js#comment32647365_13171592        
      nvd3: {
        deps: ["d3.global"],
        exports: "nv"
      },
      bootstrap : { deps :['jquery'] }
    }

  }) ;
})(requirejs) ;

// hack to get nvd3 to work with requirejs.
// see this so question on Stack Overflow:
// http://stackoverflow.com/questions/13157704/how-to-integrate-d3-with-require-js#comment32647365_13171592
define("d3.global", ["d3"], function(d3global) {
  d3 = d3global;
});

require([], function() {
  // Our application
  console.log("hello, JavaScript");
}) ;
```

现在我们已经设置了配置，我们可以深入到应用程序的 JavaScript 部分。

# 客户端程序架构

基本思想很简单：用户在输入框中搜索 GitHub 上某人的名字。当他输入名字时，我们向本章 earlier 设计的 API 发送请求。当 API 的响应返回时，程序将响应绑定到模型，并发出一个事件通知模型已更改。视图监听此事件并从模型中刷新。

## 设计模型

让我们先定义客户端模型。模型保存有关当前显示的用户仓库的信息。它会在第一次搜索后填充。

```py
// public/javascripts/model.js

define([], function(){
   return {
    ghubUser: "", // last name that was searched for
    exists: true, // does that person exist on github?
    repos: [] // list of repos
  } ;
});
```

要查看模型的填充值，请转到 `app.scala4datascience.com` 上的完整应用程序示例，在浏览器中打开一个 JavaScript 控制台，搜索应用程序中的用户（例如，`odersky`），然后在控制台中输入以下内容：

```py
> require(["model"], function(model) { console.log(model) ; }) 
{ghubUser: "odersky", exists: true, repos: Array}

> require(["model"], function(model) { 
 console.log(model.repos[0]); 
})
{name: "dotty", language: "Scala", is_fork: true, size: 14653}

```

这些导入 `"model"` 模块，将其绑定到变量 `model`，然后打印信息到控制台。

## 事件总线

我们需要一个机制来通知视图当模型更新时，因为视图需要从新的模型中刷新。这通常通过 Web 应用程序中的 *事件* 来处理。jQuery 允许我们将回调绑定到特定事件。当该事件发生时，回调将被执行。

例如，要将回调绑定到事件 `"custom-event"`，在 JavaScript 控制台中输入以下内容：

```py
> $(window).on("custom-event", function() { 
 console.log("custom event received") ; 
});

```

我们可以使用以下方式触发事件：

```py
> $(window).trigger("custom-event"); 
custom event received

```

在 jQuery 中，事件需要注册在 *事件总线* 上的 DOM 元素。在这个例子中，我们使用了 `window` DOM 元素作为我们的事件总线，但任何 jQuery 元素都可以。将事件定义集中到单个模块中是有帮助的。因此，我们将创建一个包含两个函数的 `events` 模块：`trigger`，用于触发一个事件（由一个字符串指定），和 `on`，用于将回调绑定到特定事件：

```py
// public/javascripts/events.js

define(["jquery"], function($) {

  var bus = $(window) ; // widget to use as an event bus

  function trigger(eventType) {
    $(bus).trigger(eventType) ;
  }

  function on(eventType, f) {
    $(bus).on(eventType, f) ;
  }

  return {
    "trigger": trigger,
    "on": on
  } ;
});
```

我们现在可以使用 `events` 模块发出和接收事件。你可以在应用程序的实时版本（在 `app.scala4datascience.com`）的 JavaScript 控制台中测试这一点。让我们首先注册一个监听器：

```py
> require(["events"], function(events) {
  // register event listener
  events.on("hello_event", function() {
    console.log("Received event") ;
  }) ;
}); 
```

如果我们现在触发事件 `"hello_event"`，监听器将打印 `"Received event"`：

```py
> require(["events"], function(events) {
  // trigger the event
  events.trigger("hello_event") ;
}) ;
```

使用事件使我们能够将控制器与视图解耦。控制器不需要了解任何关于视图的信息，反之亦然。控制器只需要在模型更新时发出一个 `"model_updated"` 事件，而视图在接收到该事件时需要从模型刷新。

## 通过 jQuery 进行 AJAX 调用

现在我们可以编写我们应用程序的控制器。当用户在文本输入框中输入名称时，我们查询 API，更新模型并触发一个 `model_updated` 事件。

我们使用 jQuery 的 `$.getJSON` 函数来查询我们的 API。这个函数将其第一个参数作为 URL，第二个参数作为回调函数。API 调用是异步的：`$.getJSON` 在执行后立即返回。因此，所有请求处理都必须在回调中进行。如果请求成功，将调用回调，但我们可以定义始终调用或失败时调用的额外处理程序。让我们在浏览器控制台（无论是你自己的，如果你正在运行上一章开发的 API，还是 `app.scala4datascience.com` 上的）中尝试一下。回想一下，API 正在监听 `/api/repos/:user` 的端点：

```py
> $.getJSON("/api/repos/odersky", function(data) { 
 console.log("API response:");
 console.log(data);
 console.log(data[0]); 
}) ;
{readyState: 1, getResponseHeader: function, ...}

API response:
[Object, Object, Object, Object, Object, ...]
{name: "dotty", language: "Scala", is_fork: true, size: 14653}

```

`getJSON` 立即返回。几秒钟后，API 响应，此时响应将通过回调传递。

回调仅在成功时执行。它接受 API 返回的 JSON 对象作为其参数。要绑定在 API 请求失败时执行的回调，请调用 `getJSON` 返回值的 `.fail` 方法：

```py
> $.getJSON("/api/repos/junk123456", function(data) { 
 console.log("called on success"); 
}).fail(function() { 
 console.log("called on failure") ; 
}) ;
{readyState: 1, getResponseHeader: function, ...}

called on failure

```

我们还可以使用 `getJSON` 返回值的 `.always` 方法来指定无论 API 查询是否成功都执行的回调。

现在我们知道了如何使用 `$.getJSON` 查询我们的 API，我们可以编写控制器。控制器监听 `#user-selection` 输入字段的更改。当发生更改时，它向 API 发送 AJAX 请求以获取该用户的信息。它绑定一个回调，当 API 回复包含存储库列表时更新模型。我们将定义一个 `controller` 模块，该模块导出一个名为 `initialize` 的单个函数，该函数创建事件监听器：

```py
// public/javascripts/controller.js
define(["jquery", "events", "model"], function($, events, model) {

  function initialize() {
    $("#user-selection").change(function() {

      var user = $("#user-selection").val() ;
      console.log("Fetching information for " + user) ;

      // Change cursor to a 'wait' symbol 
      // while we wait for the API to respond
      $("*").css({"cursor": "wait"}) ; 

      $.getJSON("/api/repos/" + user, function(data) {
        // Executed on success
        model.exists = true ;
        model.repos = data ;
      }).fail(function() {
        // Executed on failure
        model.exists = false ;
        model.repos = [] ;
      }).always(function() {
        // Always executed
        model.ghubUser = user ;

        // Restore cursor
        $("*").css({"cursor": "initial"}) ;

        // Tell the rest of the application 
        // that the model has been updated.
        events.trigger("model_updated") ;
      });
    }) ;
  } ;

  return { "initialize": initialize };

});
```

我们的控制模块仅公开 `initialize` 方法。一旦初始化完成，控制器将通过事件监听器与应用程序的其余部分交互。我们将在 `main.js` 中调用控制器的 `initialize` 方法。目前，该文件的最后一行只是一个空的 `require` 块。让我们导入我们的控制器并初始化它：

```py
// public/javascripts/main.js

require(["controller"], function(controller) {
  controller.initialize();
});
```

为了测试这一点，我们可以将一个虚拟监听器绑定到 `"model_updated"` 事件。例如，我们可以使用以下片段将当前模型记录到浏览器 JavaScript 控制台中（您可以直接在 JavaScript 控制台中编写此片段）：

```py
> require(["events", "model"], 
function(events, model) {
  events.on("model_updated", function () { 
    console.log("model_updated event received"); 
    console.log(model); 
  });
}); 
```

如果你搜索一个用户，模型将被打印到控制台。我们现在有了控制器。最后一步是编写视图。

## 响应视图

如果请求失败，我们只需在响应 div 中显示 **未找到**。这部分代码编写起来最简单，所以让我们先做这个。我们定义一个 `initialize` 方法来生成视图。然后视图监听 `"model_updated"` 事件，该事件在控制器更新模型后触发。一旦初始化完成，与响应视图交互的唯一方式是通过 `"model_updated"` 事件：

```py
// public/javascripts/responseView.js

define(["jquery", "model", "events"],
function($, model, events) {

  var failedResponseHtml = 
    "<div class='col-md-12'>Not found</div>" ;

  function initialize() {
    events.on("model_updated", function() {
      if (model.exists) {
        // success – we will fill this in later.
        console.log("model exists")
      }
      else {
        // failure – the user entered
        // is not a valid GitHub login 
        $("#response").html(failedResponseHtml) ;
      }
    }) ;
  }

  return { "initialize": initialize } ;

});
```

为了启动视图，我们必须从 `main.js` 中调用 `initialize` 函数。只需在 `require` 块中添加对 `responseView` 的依赖，并调用 `responseView.initialize()`。经过这些修改，`main.js` 中的最终 `require` 块如下：

```py
// public/javascripts/main.js

require(["controller", "responseView"],
function(controller, responseView) {
  controller.initialize();
  responseView.initialize() ;
}) ;
```

你可以通过在用户输入中输入垃圾信息来故意导致 API 请求失败，以检查这一切是否正常工作。

当用户输入有效的 GitHub 登录名并且 API 返回一个仓库列表时，我们必须在屏幕上显示这些信息。我们显示一个表格和一个饼图，该饼图按语言聚合仓库大小。我们将定义饼图和表格在两个单独的模块中，分别称为 `repoGraph.js` 和 `repoTable.js`。现在让我们假设这些模块存在，并且它们公开了一个接受 `model` 和要在其中出现的 `div` 名称的 `build` 方法。

让我们更新 `responseView` 的代码，以适应用户输入有效的 GitHub 用户名：

```py
// public/javascripts/responseView.js

define(["jquery", "model", "events", "repoTable", "repoGraph"],
function($, model, events, repoTable, repoGraph) {

  // HTHML to inject when the model represents a valid user 
 var successfulResponseHtml = 
 "<div class='col-md-6' id='response-table'></div>" +
 "<div class='col-md-6' id='response-graph'></div>" ;

  // HTML to inject when the model is for a non-existent user
  var failedResponseHtml = 
    "<div class='col-md-12'>Not found</div>" ;

  function initialize() {
    events.on("model_updated", function() {
      if (model.exists) {
 $("#response").html(successfulResponseHtml) ;
 repoTable.build(model, "#response-table") ;
 repoGraph.build(model, "#response-graph") ;
      }
      else {
        $("#response").html(failedResponseHtml) ;
      }
    }) ;
  }

  return { "initialize": initialize } ;

});
```

让我们回顾一下在 API 调用成功时会发生什么。我们在 `#response` div 中注入以下 HTML 片段：

```py
var successfulResponseHtml = 
  "<div class='col-md-6' id='response-table'></div>" +
  "<div class='col-md-6' id='response-graph'></div>" ;
```

这添加了两个 HTML div，一个用于仓库表，另一个用于图表。我们使用 Bootstrap 类来垂直分割响应 div。

现在我们将注意力转向表格视图，它需要暴露一个单独的 `build` 方法，正如前一小节所描述的。我们只需在 HTML 表格中显示仓库。我们将使用 *Underscore 模板* 来动态构建表格。Underscore 模板的工作方式与 Scala 中的字符串插值类似：我们定义一个带有占位符的模板。让我们在浏览器控制台中试一试：

```py
> require(["underscore"], function(_) {
  var myTemplate = _.template(
    "Hello, <%= title %> <%= name %>!"
  ) ;
});
```

这创建了一个 `myTemplate` 函数，该函数接受具有 `title` 和 `name` 属性的对象：

```py
> require(["underscore"], function(_) {
  var myTemplate = _.template( ... ); 
  var person = { title: "Dr.", name: "Odersky" } ;
  console.log(myTemplate(person)) ;
});
```

Underscore 模板因此提供了一种方便的机制，可以将对象格式化为字符串。我们将为表格中的每一行创建一个模板，并将每个仓库的模型传递给模板：

```py
// public/javascripts/repoTable.js

define(["underscore", "jquery"], function(_, $) {

  // Underscore template for each row
  var rowTemplate = _.template("<tr>" +
    "<td><%= name %></td>" +
    "<td><%= language %></td>" +
    "<td><%= size %></td>" +
    "</tr>") ;

  // template for the table
  var repoTable = _.template(
    "<table id='repo-table' class='table'>" +
      "<thead>" +
        "<tr>" +
          "<th>Name</th><th>Language</th><th>Size</th>" +
        "</tr>" +
      "</thead>" +
      "<tbody>" +
        "<%= tbody %>" +
      "</tbody>" +
    "</table>") ;

  // Builds a table for a model
  function build(model, divName) {
    var tbody = "" ;
    _.each(model.repos, function(repo) {
      tbody += rowTemplate(repo) ;
    }) ;
    var table = repoTable({tbody: tbody}) ;
    $(divName).html(table) ;
  }

  return { "build": build } ;
}) ;
```

# 使用 NVD3 绘制图表

D3 是一个提供用于在 JavaScript 中构建交互式可视化的低级组件的库。通过提供低级组件，它为开发者提供了极大的灵活性。然而，学习曲线可能相当陡峭。在这个例子中，我们将使用 NVD3，这是一个为 D3 提供预制图表的库。这可以大大加快初始开发速度。我们将代码放置在文件 `repoGraph.js` 中，并公开一个名为 `build` 的单一方法，该方法接受一个模型和一个 div 作为参数，并在该 div 中绘制饼图。饼图将汇总所有用户仓库中的语言使用情况。

生成饼图的代码几乎与 NVD3 文档中给出的示例相同，该文档可在 [`nvd3.org/examples/pie.html`](http://nvd3.org/examples/pie.html) 找到。传递给图表的数据必须作为一个对象数组可用。每个对象必须包含一个 `label` 字段和一个 `size` 字段。`label` 字段标识语言，而 `size` 字段是该用户用该语言编写的所有仓库的总大小。以下是一个有效的数据数组：

```py
[ 
  { label: "Scala", size: 1234 },
  { label: "Python", size: 4567 }
]
```

要以这种格式获取数据，我们必须在我们的模型中聚合特定语言编写的仓库的大小。我们编写了 `generateDataFromModel` 函数来将模型中的 `repos` 数组转换为适合 NVD3 的数组。聚合的核心操作是通过调用 Underscore 的 `groupBy` 方法来按语言分组仓库。此方法与 Scala 的 `groupBy` 方法完全相同。考虑到这一点，`generateDataFromModel` 函数如下：

```py
// public/javascripts/repoGraph.js

define(["underscore", "d3", "nvd3"], 
function(_, d3, nv) {

  // Aggregate the repo size by language.
  // Returns an array of objects like:
  // [ { label: "Scala", size: 1245}, 
  //   { label: "Python", size: 432 } ]
  function generateDataFromModel(model) {

    // Build an initial object mapping each
    // language to the repositories written in it
    var language2Repos = _.groupBy(model.repos, 
      function(repo) { return repo.language ; }) ;

    // Map each { "language":  [ list of repos ], ...} 
    // pairs to a single document { "language": totalSize }
    // where totalSize is the sum of the individual repos.
    var plotObjects = _.map(language2Repos, 
      function(repos, language) {
        var sizes = _.map(repos, function(repo) { 
          return repo.size; 
        });
        // Sum over the sizes using 'reduce'
        var totalSize = _.reduce(sizes, 
          function(memo, size) { return memo + size; },
        0) ;
        return { label: language, size: totalSize } ;
      }) ;

     return plotObjects;
  }
```

我们现在可以使用 NVD3 的 `addGraph` 方法来构建饼图：

```py
  // Build the chart.
  function build(model, divName) {
    var transformedModel = generateDataFromModel(model) ;
    nv.addGraph(function() {

      var height = 350;
      var width = 350; 

      var chart = nv.models.pieChart()
        .x(function (d) { return d.label ; })
        .y(function (d) { return d.size ;})
        .width(width)
        .height(height) ;

      d3.select(divName).append("svg")
        .datum(transformedModel)
        .transition()
        .duration(350)
        .attr('width', width)
        .attr('height', height)
        .call(chart) ;

      return chart ;
    });
  }

  return { "build" : build } ;

});
```

这是我们应用程序的最后一个组件。将您的浏览器指向 `127.0.0.1:9000`，您应该会看到应用程序正在运行。

恭喜！我们已经构建了一个功能齐全的单页 Web 应用程序。

# 摘要

在本章中，我们学习了如何使用 Play 框架编写一个功能齐全的 Web 应用程序。恭喜你走到了这一步。构建 Web 应用程序可能会让许多数据科学家超出他们的舒适区，但了解足够的 Web 知识来构建基本的应用程序将允许你以引人入胜、吸引人的方式分享你的结果，同时也有助于与软件工程师和 Web 开发者进行沟通。

这就结束了我们对 Scala 库的快速浏览。在这本书的过程中，我们学习了如何使用 Breeze 高效地解决线性代数和优化问题，如何以函数式的方式在 SQL 数据库中插入和查询数据，以及如何与 Web API 交互以及如何创建它们。我们回顾了一些数据科学家用于编写并发或并行应用程序的工具，从并行集合和未来到 Spark 通过 Akka。我们看到了这些结构在 Scala 库中的普遍性，从 Play 框架中的未来到 Spark 的骨干 Akka。如果你已经读到这儿，给自己点个赞。

这本书对其涵盖的库进行了最简短的介绍，希望这足以让你尝到每个工具的用途，你可以用它完成什么，以及它在更广泛的 Scala 生态系统中的位置。如果你决定在你的数据科学流程中使用这些工具，你需要更详细地阅读文档，或者一本更完整的参考书。每章末尾列出的参考应该是一个良好的起点。

Scala 和数据科学都在快速发展。不要对某个特定的工具包或概念过于执着。保持对当前发展的关注，最重要的是，保持务实：找到适合工作的正确工具。Scala 和这里讨论的库通常会是那个工具，但并不总是：有时，一个 shell 命令或简短的 Python 脚本会更有效。记住，编程技能只是数据科学家知识体系的一个方面。即使你想在数据科学的工程方面专长，也要了解问题领域和机器学习数学基础。

最重要的是，如果你花时间阅读这本书，你很可能认为编程和数据科学不仅仅是工作。在 Scala 中编码可以令人满意和有成就感，所以享受乐趣，做得很棒！

# 参考文献

网上到处都是 HTML 和 CSS 教程。简单的 Google 搜索就能给你一个比我提供的任何参考列表都要好的资源了解。

Mike Bostock 的网站上有丰富的美丽 D3 可视化：[`bost.ocks.org/mike/`](http://bost.ocks.org/mike/)。为了更好地了解 D3，我推荐*Scott Murray 的《Web 交互数据可视化》*。

你也可以参考前一章中给出的关于 Play 框架和设计 REST API 的参考书籍。

# 附录 A. 模式匹配和提取器

模式匹配是 Scala 中控制流的一个强大工具。对于从命令式语言转向 Scala 的人来说，它往往被低估和未充分利用。

在深入理论之前，我们先来看几个模式匹配的例子。我们首先定义一个元组：

```py
scala> val names = ("Pascal", "Bugnion")
names: (String, String) = (Pascal,Bugnion)

```

我们可以使用模式匹配来提取这个元组的元素并将它们绑定到变量上：

```py
scala> val (firstName, lastName) = names
firstName: String = Pascal
lastName: String = Bugnion

```

我们刚刚提取了`names`元组的两个元素，并将它们绑定到变量`firstName`和`lastName`。注意左侧定义了一个模式，右侧必须匹配：我们声明变量`names`必须是一个包含两个元素的元组。为了使模式更具体，我们还可以指定元组中元素的预期类型：

```py
scala> val (firstName:String, lastName:String) = names
firstName: String = Pascal
lastName: String = Bugnion

```

如果左侧的模式与右侧不匹配会发生什么？

```py
scala> val (firstName, middleName, lastName) = names
<console>:13: error: constructor cannot be instantiated to expected type;
found   : (T1, T2, T3)
required: (String, String)
 val (firstName, middleName, lastName) = names

```

这会导致编译错误。其他类型的模式匹配失败会导致运行时错误。

模式匹配非常具有表现力。要实现没有模式匹配的相同行为，你必须明确地做以下操作：

+   验证变量`names`是一个包含两个元素的元组

+   提取第一个元素并将其绑定到`firstName`

+   提取第二个元素并将其绑定到`lastName`

如果我们期望元组中的某些元素具有特定的值，我们可以在模式匹配过程中验证这一点。例如，我们可以验证`names`元组的第一个元素匹配`"Pascal"`：

```py
scala> val ("Pascal", lastName) = names
lastName: String = Bugnion

```

除了元组之外，我们还可以在 Scala 集合上进行匹配：

```py
scala> val point = Array(1, 2, 3)
point: Array[Int] = Array(1, 2, 3)

scala> val Array(x, y, z) = point
x: Int = 1
y: Int = 2
z: Int = 3

```

注意这种模式匹配与数组构造的相似性：

```py
scala> val point = Array(x, y, z)
point: Array[Int] = Array(1, 2, 3)

```

语法上，Scala 将模式匹配表达为实例构造过程的逆过程。我们可以将模式匹配视为对象的解构，将对象的组成部分绑定到变量。

当匹配集合时，有时人们只对匹配第一个元素或前几个元素感兴趣，并丢弃集合的其余部分，无论其长度如何。运算符`_*`将匹配任意数量的元素：

```py
scala> val Array(x, _*) = point
x: Int = 1

```

默认情况下，由`_*`运算符匹配的模式部分不会被绑定到变量。我们可以如下捕获它：

```py
scala> val Array(x, xs @ _*) = point
x: Int = 1
xs: Seq[Int] = Vector(2, 3)

```

除了元组和集合之外，我们还可以匹配 case 类。让我们首先定义一个表示名称的 case：

```py
scala> case class Name(first: String, last: String)
defined class Name

scala> val name = Name("Martin", "Odersky")
name: Name = Name(Martin,Odersky)

```

我们可以以与匹配元组相同的方式匹配`Name`的实例：

```py
scala> val Name(firstName, lastName) = name
firstName: String = Martin
lastName: String = Odersky

```

所有这些模式也可以用在`match`语句中：

```py
scala> def greet(name:Name) = name match {
 case Name("Martin", "Odersky") => "An honor to meet you"
 case Name(first, "Bugnion") => "Wow! A family member!"
 case Name(first, last) => s"Hello, $first"
}
greet: (name: Name)String

```

# 在 for 推导式中的模式匹配

模式匹配在*for*推导式中非常有用，可以提取与特定模式匹配的集合中的项目。让我们构建一个`Name`实例的集合：

```py
scala> val names = List(Name("Martin", "Odersky"), 
 Name("Derek", "Wyatt"))
names: List[Name] = List(Name(Martin,Odersky), Name(Derek,Wyatt))

```

我们可以在 for 推导式中使用模式匹配来提取类的内部结构：

```py
scala> for { Name(first, last) <- names } yield first
List[String] = List(Martin, Derek)

```

到目前为止，没有什么特别突破性的。但如果我们想提取所有名字为`"Martin"`的人的姓氏呢？

```py
scala> for { Name("Martin", last) <- names } yield last
List[String] = List(Odersky)

```

写作`Name("Martin", last) <- names`提取与模式匹配的`names`元素。你可能认为这是一个人为的例子，确实如此，但第七章，*Web APIs*中的例子展示了这种语言模式的有用性和多功能性，例如，用于从 JSON 对象中提取特定字段。

# 模式匹配内部机制

如果你定义了一个案例类，就像我们用 `Name` 看到的那样，你将免费获得对构造函数的模式匹配。你应该尽可能使用案例类来表示你的数据，从而减少实现自己的模式匹配的需求。尽管如此，了解模式匹配的工作原理仍然是有用的。

当你创建一个案例类时，Scala 会自动构建一个伴随对象：

```py
scala> case class Name(first: String, last: String)
defined class Name

scala> Name.<tab>
apply   asInstanceOf   curried   isInstanceOf   toString   tupled   unapply

```

用于（内部）模式匹配的方法是 `unapply`。此方法接受一个对象作为参数，并返回 `Option[T]，其中 T 是案例类的值元组的元组。

```py
scala> val name = Name("Martin", "Odersky")
name: Name = Name(Martin,Odersky)

scala> Name.unapply(name)
Option[(String, String)] = Some((Martin,Odersky))

```

`unapply` 方法是一个 *提取器*。它扮演着与构造函数相反的角色：它接受一个对象，并提取构建该对象所需的参数列表。当你写 `val Name(firstName, lastName)`，或者当你使用 `Name` 作为匹配语句中的案例时，Scala 会调用 `Name.unapply` 来匹配你正在匹配的对象。`Some[(String, String)]` 的值表示模式匹配，而 `None` 的值表示模式失败。

要编写自定义提取器，你只需要一个具有 `unapply` 方法的对象。虽然 `unapply` 通常位于你正在解构的类的伴随对象中，但这并不一定是这种情况。实际上，它根本不需要对应于现有的类。例如，让我们定义一个匹配任何非零双精度浮点数的 `NonZeroDouble` 提取器：

```py
scala> object NonZeroDouble { 
 def unapply(d:Double):Option[Double] = {
 if (d == 0.0) { None } else { Some(d) } 
 }
}
defined object NonZeroDouble

scala> val NonZeroDouble(denominator) = 5.5
denominator: Double = 5.5

scala> val NonZeroDouble(denominator) = 0.0
scala.MatchError: 0.0 (of class java.lang.Double)
 ... 43 elided

```

尽管没有相应的 `NonZeroDouble` 类，我们仍然定义了一个 `NonZeroDouble` 提取器。

这个 `NonZeroDouble` 提取器在匹配对象中很有用。例如，让我们定义一个 `safeDivision` 函数，当分母为零时返回默认值：

```py
scala> def safeDivision(numerator:Double, 
 denominator:Double, fallBack:Double) =
 denominator match {
 case NonZeroDouble(d) => numerator / d
 case _ => fallBack
 }
safeDivision: (numerator: Double, denominator: Double, fallBack: Double)Double

scala> safeDivision(5.0, 2.0, 100.0)
Double = 2.5

scala> safeDivision(5.0, 0.0, 100.0)
Double = 100.0

```

这是一个简单的例子，因为 `NonZeroDouble.unapply` 方法非常简单，但如果你定义一个更复杂的测试，你可能会看到它的有用性和表达性。定义自定义提取器让你能够定义强大的控制流结构来利用 `match` 语句。更重要的是，它们使使用提取器的客户端能够以声明性的方式考虑控制流：客户端可以声明他们需要一个 `NonZeroDouble`，而不是指示编译器检查值是否为零。

# 提取序列

前一节解释了从案例类中提取数据以及如何编写自定义提取器，但没有解释如何在序列上提取：

```py
scala> val Array(a, b) = Array(1, 2)
a: Int = 1
b: Int = 2

```

与依赖 `unapply` 方法不同，序列依赖于在伴随对象中定义的 `unapplySeq` 方法。这应该返回 `Option[Seq[A]]`：

```py
scala> Array.unapplySeq(Array(1, 2))
Option[IndexedSeq[Int]] = Some(Vector(1, 2))

```

让我们写一个例子。我们将编写一个用于 Breeze 向量的提取器（目前 Breeze 向量不支持模式匹配）。为了避免与`DenseVector`伴生对象冲突，我们将我们的`unapplySeq`写在单独的对象中，称为`DV`。我们的`unapplySeq`方法需要做的只是将其参数转换为 Scala `Vector`实例。为了避免泛型混淆概念，我们将仅为此`[Double]`向量编写此实现：

```py
scala> import breeze.linalg._
import breeze.linalg._

scala> object DV {
 // Just need to convert to a Scala vector.
 def unapplySeq(v:DenseVector[Double]) = Some(v.toScalaVector)
}
defined object DV

```

让我们尝试我们的新提取器实现：

```py
scala> val vec = DenseVector(1.0, 2.0, 3.0)
vec: breeze.linalg.DenseVector[Double] = DenseVector(1.0, 2.0, 3.0)

scala> val DV(x, y, z) = vec
x: Double = 1.0
y: Double = 2.0
z: Double = 3.0

```

# 摘要

模式匹配是控制流的一个强大工具。它鼓励程序员以声明式的方式思考：声明你期望一个变量匹配某个模式，而不是明确告诉计算机如何检查它是否匹配这个模式。这可以节省许多代码行并提高清晰度。

# 参考

对于 Scala 中模式匹配的概述，没有比*Programming in Scala*更好的参考书籍了，作者是*Martin Odersky*、*Bill Venners*和*Lex Spoon*。第一版的在线版本可在以下网址找到：[`www.artima.com/pins1ed/case-classes-and-pattern-matching.html`](https://www.artima.com/pins1ed/case-classes-and-pattern-matching.html)。

*Daniel Westheide*的博客涵盖了稍微高级一点的 Scala 结构，并且是非常有用的阅读材料：[`danielwestheide.com/blog/2012/11/21/the-neophytes-guide-to-scala-part-1-extractors.html`](http://danielwestheide.com/blog/2012/11/21/the-neophytes-guide-to-scala-part-1-extractors.html)。
