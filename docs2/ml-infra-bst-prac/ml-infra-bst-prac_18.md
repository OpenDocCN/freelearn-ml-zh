# 14

# 数据获取和管理中的伦理

**机器学习**（**ML**）需要大量数据，这些数据可能来自各种来源，但并非所有来源都同样易于使用。在软件工程中，我们可以设计和开发使用来自其他系统数据的系统。我们还可以使用并非真正来自人类的数据；例如，我们可以使用关于系统缺陷或复杂性的数据。然而，为了给社会提供更多价值，我们需要使用包含有关人或其财产信息的数据；例如，当我们训练机器识别人脸或车牌时。然而，无论我们的用例如何，我们都需要遵循伦理准则，最重要的是，我们的软件不应造成任何伤害。

我们本章从探讨几个不道德的系统示例开始，这些系统显示出偏见；例如，惩罚某些少数群体的信用评级系统。我还会解释使用开源数据和揭露受试者身份的问题。然而，本章的核心是对数据管理和软件系统伦理框架的解释和讨论，包括**电气和电子工程师协会**（**IEEE**）和**计算机协会**（**ACM**）的行为准则。

在本章中，我们将涵盖以下主要内容：

+   计算机科学和软件工程中的伦理

+   数据无处不在，但我们真的能利用它吗？

+   来自开源系统数据的伦理

+   从人类收集的数据背后的伦理

+   合同和法律义务

# 计算机科学和软件工程中的伦理

现代伦理观源于第二次世界大战后制定的《纽伦堡法典》。该法典基于几个原则，但最重要的是，每个涉及人类受试者的研究都需要获得许可。这是至关重要的，因为它防止了在实验中对人类的滥用。研究中的每个参与者也应能够在任何时候撤回他们的许可。让我们看看所有 10 个原则：

1.  人类受试者的自愿同意绝对必要。

1.  实验应当产生有益于社会的丰富结果，这些结果无法通过其他方法或研究手段获得，并且其本质不是随机的和不必要的。

1.  实验应当设计得基于动物实验的结果和对研究疾病或其他问题的自然历史的了解，以便预期的结果将证明进行实验的合理性。

1.  实验应当以避免所有不必要的身体和精神上的痛苦和伤害的方式进行。

1.  不应进行任何实验，除非有先验理由相信将发生死亡或致残伤害，除非在这些实验中，实验医生也作为受试者。

1.  需要承担的风险程度不应超过实验解决所涉问题的 humanitarian 重要性所确定的程度。

1.  应做好适当的准备，并提供充足的设施以保护实验对象免受伤害、残疾或死亡等远程可能性的影响。

1.  实验应由具有科学资格的人员进行。在整个实验过程中，进行或参与实验的人员应要求具备最高程度的技能和谨慎。

1.  在实验过程中，如果实验对象达到他们认为继续实验似乎不可能的身体或心理状态，他们应有权自由结束实验。

1.  在实验过程中，负责的科学家必须准备好在任何阶段终止实验，如果他们有合理的理由相信，在行使他们所需的诚信、高超技能和谨慎判断时，实验的继续可能导致实验对象受伤、残疾或死亡。

《纽伦堡法典》为现代人体实验和研究中的伦理标准奠定了基础。它在后续伦理指南和法规的发展中产生了重大影响，例如《赫尔辛基宣言》（**DoH**）以及各种国家和国际关于人体研究伦理的法规。这些原则强调了尊重研究参与者的权利和福祉的重要性，并确保研究以道德和负责任的方式进行。

上述原则旨在指导实验为社会创造价值，同时尊重实验参与者。第一条原则是关于同意的，这很重要，因为我们希望防止使用那些对实验不知情的人。在机器学习的背景下，这意味着我们在收集包含人类数据时必须非常小心；例如，当我们收集图像数据以训练物体识别时，或者当我们从包含个人信息的开源存储库中收集数据时。

虽然我们在软件工程中确实进行实验，但这些原则比我们想象的更为普遍。在本章中，我们探讨这些原则如何影响人工智能系统工程中的数据伦理。

除了这些原则之外，我们还关注数据偏差的来源以及如何避免它。

# 数据无处不在，但我们真的能利用它吗？

我们保护受试者和数据的一种方式是使用适当的数据使用许可。许可在某种程度上是一种合同，许可方授予许可方以特定方式使用数据的权限。许可既用于软件产品（算法、组件）也用于数据。以下是一些在当代软件中最常用的许可模型：

+   **专有许可证**：这是一种许可人拥有数据并授予使用数据用于某些目的（通常是盈利目的）的许可模式。在这种合同中，各方通常规定数据可以如何使用、使用多长时间，以及双方的责任。

+   **许可开放许可证**：这些许可证为许可人提供了几乎无限制的数据访问权限，同时限制了许可人的责任。通常，许可人不需要向许可人提供其产品或衍生作品。

+   **非许可开放许可证**：这些许可证提供了几乎无限制的访问权限，同时要求某种形式的互惠。通常，这种互惠是以要求许可人提供产品或衍生作品访问权限的形式出现的。

自然，这三类许可证也有变体。因此，让我们看看一种流行的开源许可证——来自 Hugging Face 的 Unsplash 许可证：

```py
Unsplash
All unsplash.com images have the Unsplash license copied below:
https://unsplash.com/license
License
Unsplash photos are made to be used freely. Our license reflects that.
All photos can be downloaded and used for free
Commercial and non-commercial purposes
No permission needed (though attribution is appreciated!)
What is not permitted
Photos cannot be sold without significant modification.
Compiling photos from Unsplash to replicate a similar or competing service.
Tip: How to give attribution
Even though attribution isn't required, Unsplash photographers appreciate it as it provides exposure to their work and encourages them to continue sharing.
Photo by <person name> on Unsplash
Longform
Unsplash grants you an irrevocable, nonexclusive, worldwide copyright license to download, copy, modify, distribute, perform, and use photos from Unsplash for free, including for commercial purposes, without permission from or attributing the photographer or Unsplash. This license does not include the right to compile photos from Unsplash to replicate a similar or competing service.
Other Images
All other images were either taken by the authors, or created by friends of the authors and all permissions to modify, distribute, copy, perform and use are given to the authors.
```

许可证来源于此数据集：[`huggingface.co/datasets/google/dreambooth/blob/main/dataset/references_and_licenses.txt`](https://huggingface.co/datasets/google/dreambooth/blob/main/dataset/references_and_licenses.txt)。让我们更详细地了解一下许可证的含义。首先，这是 Unsplash 许可证允许的内容：

+   **免费使用**：我们可以免费下载和使用 Unsplash 上的图像。

+   **商业和非商业用途**：我们可以将图像用于与商业相关的目的（如广告、网站和产品包装）以及个人项目或非营利活动。

+   **无需许可**：我们不需要从摄影师或 Unsplash 那里获得同意或许可来使用图像。这使得整个过程变得无烦恼且方便。

+   **修改和重新分发**：我们可以根据您的需求以任何方式修改原始图像并分发它。然而，如果您想出售该图像，它应该与原始图像有显著的不同。

+   **不可撤销、非独占、全球范围的许可证**：一旦下载了照片，您就有权无限期地使用它（不可撤销）。非独占意味着其他人也可以使用相同的图像，而全球意味着在使用上没有地理限制。

同时，许可证禁止某些活动：

+   **销售未修改的照片**：我们不能以原始形式或未经对它们进行重大修改的情况下出售照片。

+   **复制 Unsplash 的服务**：我们不能使用这些图像创建直接竞争或类似 Unsplash 的服务。换句话说，下载大量 Unsplash 图像然后使用这些图像开始自己的股票照片服务将违反许可证。

许可证还规定了致谢的形式——虽然在使用图片时不必提及摄影师或 Unsplash，但这是鼓励的。致谢是认可摄影师努力的一种方式。提供的示例，“*照片由<人名>在 Unsplash 上拍摄*”是建议的致谢格式。

它还规定了如果数据集中有其他图片（例如，由第三方添加）会发生什么：对于不是来自 Unsplash 的图片，作者要么自己拍了这些照片，要么让熟人拍摄。他们有完全的权限无限制地使用、分发、修改和表演这些图片。

我们还可以查看**创意共享**（**CC**）的一个许可示例；例如，**创意共享，署名，版本 4.0**（**CC-BY-4.0**）许可（[`creativecommons.org/licenses/by/4.0/legalcode`](https://creativecommons.org/licenses/by/4.0/legalcode)）。简而言之，此许可证允许在以下条件下共享和重新分配数据：

+   **署名**，这意味着当我们使用数据时，我们必须给予数据作者适当的信用，并必须提供参考链接并指出我们对数据所做的更改。

+   **无额外限制**，这意味着我们不能对以这种方式许可的数据的使用施加任何额外的限制。如果我们使用的数据在我们的产品中使用，我们不得使用任何手段来限制他人使用这些数据。

许可证的全文有点长，不适合全部引用，但让我们分析其中的一部分，以便我们可以看到它与非常宽松的 Unsplash 许可之间的差异。首先，许可证提供了一系列定义，这些定义有助于法律纠纷。例如，*第 1.i 节*定义了共享的含义：

```py
Share means to provide material to the public by any means or process that requires permission under the Licensed Rights, such as reproduction, public display, public performance, distribution, dissemination, communication, or importation, and to make material available to the public including in ways that members of the public may access the material from a place and at a time individually chosen by them.
```

前面的法律文本可能听起来很复杂，但当我们阅读它时，文本具体说明了数据共享的含义。例如，它说“*进口*”是数据共享的一种方式。这意味着将数据作为 Python 包、C#库或 GitHub 仓库的一部分提供，是数据共享的一种方式。

我们还可以查看许可证的*第二部分*，其中使用了*第一部分*中的定义。以下是一个*第 2.a.1 节*的例子：

```py
Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to:
* reproduce and Share the Licensed Material, in whole or in part; and
* produce, reproduce, and Share Adapted Material.
```

在第一部分，我们了解到许可证允许我们免费使用材料；也就是说，我们不必为此付费。它还指定了使用地点（全球范围内）并说明我们不是唯一使用这些数据的人（非独占）。然后，许可证还说明我们可以复制材料，无论是全部还是部分，或者制作改编材料。

然而，某些权利并未转让给我们这些许可方，我们可以在*第 2.b.1 节*中了解到：

```py
Moral rights, such as the right of integrity, are not licensed under this Public License, nor are publicity, privacy, and/or other similar personality rights; however, to the extent possible, the Licensor waives and/or agrees not to assert any such rights held by the Licensor to the limited extent necessary to allow You to exercise the Licensed Rights, but not otherwise.
```

本节解释了数据上的道德权利不会转让。特别是，如果我们遇到私人数据或可以与个人联系的数据，我们不会获得对其的权利。以下三个部分很重要：

+   **道德权利**：道德权利是版权的一个子集，与作品的个人和声誉方面有关，而不是纯粹的经济方面。这些权利可能因司法管辖区而异，但通常包括以下内容：

    +   **完整性权**：这是作者对任何可能对其荣誉或声誉造成损害的扭曲、毁损或其他修改作品提出异议的权利。

    +   **署名权**：这是作者对其作品作为创作者获得认可的权利。

    CC 许可证规定这些道德权利不受许可。这意味着尽管某人可能能够根据 CC 许可证定义的方式使用作品，但他们没有无限制地修改作品的权利，这些修改可能会损害原始创作者的声誉或荣誉。

+   **公开、隐私和其他类似的人格权利**：这些权利涉及个人的个人数据、肖像、姓名或声音。它们是保护免受不希望曝光或剥削的权利。CC 许可证也不授予用户侵犯这些权利的权利。

    例如，如果一个人的照片在 CC 许可证下，尽管我们可能能够根据该许可证允许的方式使用照片本身，但这并不意味着我们可以在未经其同意的情况下以商业或促销方式使用该人的肖像。

+   **放弃或不行使权利**：然而，许可人正在放弃或同意不执行这些道德或个人权利，以必要的程度来行使许可权利。这意味着许可人不会通过行使他们的道德或个人权利来干涉我们根据 CC 许可证允许的使用作品，但仅限于一定程度。他们并没有完全放弃这些权利；他们只是在与 CC 许可证相关的环境中限制其执行。

我强烈建议读者阅读许可证并反思其构建方式，包括规范许可人责任的部分。

然而，我们继续讨论不如 CC-BY-4.0 许可证那么宽容的许可证。例如，让我们看看所谓的 copyleft 许可证之一：**署名-非商业-禁止演绎 4.0 国际**（**CC BY-NC-ND 4.0**）。此许可证允许我们在以下条件下复制和重新分发数据（以下内容引用并改编自以下 CC 网页：[`creativecommons.org/licenses/by-nc-nd/4.0/）：`](https://creativecommons.org/licenses/by-nc-nd/4.0/):)

+   **署名**：我们必须给予适当的信用，提供链接到许可证，并指出是否进行了更改。我们可以在任何合理的方式下这样做，但不能以任何方式暗示许可人支持我们或我们的使用。

+   **NonCommercial**: 我们不得将材料用于商业目的。

+   **NoDerivatives**: 如果我们对材料进行混搭、转换或在此基础上构建，我们可能不得分发修改后的材料。

+   **No additional restrictions**: 我们不得应用法律条款或技术措施，这些条款或措施在法律上限制了他人执行许可证允许的任何行为。

许可证文本的主体结构与之前引用的 CC-BY-4.0 许可类似。让我们看看许可证不同的部分——*第 2.a.1 节*：

```py
Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to:
* reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and
* produce and reproduce, but not Share, Adapted Material for NonCommercial purposes only.
```

第一项意味着我们只能为了非商业目的分享许可材料。第二项将此扩展到改编材料。

那么，一个可以直接提出的问题是：*如何选择我们创建的数据的许可证？* 这就是我的下一个最佳实践发挥作用的地方。

最佳实践 #70

如果您为了非商业目的创建自己的数据，请使用限制您责任的许可衍生许可证之一。

在设计商业许可证时，请务必咨询您的法律顾问，以确保您为您的数据和产品选择最佳的许可证模式。然而，如果您在开源数据上工作，请尝试使用一个规范两个方面的许可证——是否可以使用您的数据进行商业目的，以及您的责任是什么。

在我的工作中，我尽量保持开放，这仅仅是因为我坚信开放科学，但这并不一定普遍适用。因此，我经常保护自己的数据不被商业使用——这就是非商业目的的原因。

第二个方面是责任。当我们收集数据时，我们无法考虑数据所有可能的用途，这意味着数据被误用或在使用我们的数据时犯错误的风险总是存在的。我们不希望因为我们没有做的事情而被起诉，所以在这种许可证中限制我们的责任总是个好主意。我们可以通过几种方式来限制它；其中之一是声明许可方有责任确保数据以道德的方式使用或遵守适用于适当地区的所有规则和法规。

这引出了下一个要点——开源系统数据的伦理使用。

# 开源系统数据的伦理背景

专有系统通常有许可证来规范数据的所有权及其用途。例如，来自公司的代码审查数据通常属于公司。员工为公司工作通常意味着他们放弃了对为公司生成数据的权利。在法律意义上这是必要的，因为员工为此得到了补偿——通常是以工资的形式。

然而，员工没有转移给公司的权利是自由使用他们的个人数据。这意味着当我们与源系统，如 Gerrit 审查系统，一起工作时，我们不应在没有涉及人员的许可下提取个人信息。如果我们执行无法对数据进行屏蔽的查询，我们必须确保个人信息（尽可能快地）被匿名化，并且不会泄露给分析。我们必须确保此类个人信息不会被公开提供。

我们可以在以下领域找到指导，那就是挖掘软件仓库的领域；例如，在 Gold 和 Krinke 最近的一篇文章中。尽管这项研究样本量较小，但作者们触及了与 GitHub 或 Gerrit 等软件仓库数据伦理使用相关的重要问题。他们提出了几个数据来源，其中最受欢迎的是：

+   **版本控制数据，如 CVS、Subversion、Git 或 Mercurial**：存在与许可和个人信息在仓库中的存在相关的挑战。

+   **问题跟踪器数据，如 Bugzilla**：挑战大多与这些仓库中个人数据的存在或数据与个人隐式关联的能力有关。

+   **邮件存档**：邮件列表具有灵活性，可用于不同的目的；例如，问题跟踪、代码审查、问答论坛等。然而，它们可能包含敏感的个人信息或可能导致影响个人的结论。

+   **构建日志**：**版本控制系统**（**VCS**）通常使用某种**持续集成**（**CI**）系统来自动化软件构建。如果构建结果被存档，它们可以为测试和构建实践的研究提供数据。

+   **Stack Overflow**：Stack Overflow 提供了他们的官方数据存档，并且（数据存档的）子集已被直接用作挑战，或包含历史信息的数据集的一部分。尽管有要求用户允许使用其数据进行分析的许可法规，但并非所有行为都是道德的。

+   **IDE 事件**：主要挑战与每个 IDE 都是为个人设置且可以访问非常个人化的数据和用户行为有关。

考虑到 GitHub 等仓库中在线可用的源代码量，分析它们以各种目的具有诱惑力。我们可以挖掘仓库中的源代码，以了解软件是如何构建、设计和测试的。这种源代码的使用受到每个仓库提供的许可证的限制。

同时，当挖掘有关源代码的数据时，我们可以挖掘有关贡献者、他们的拉取请求评论以及他们工作的组织的数据。这种对代码库的使用由许可证和个人数据使用的伦理准则所规范。如本章开头所述，使用个人数据需要从个人那里获得同意，这意味着我们应该请求我们分析数据的人的许可。然而，在实践中，这是不可能的，有两个原因——一个是联系所有贡献者的实际能力，另一个是代码库的使用条款。大多数代码库禁止联系贡献者进行研究。

因此，我们需要应用与隐私和利益平衡相关的原则。在大多数情况下，个人的隐私比我们进行的研究的价值更重要，因此我的下一个最佳实践是。

最佳实践 #71

限制自己只研究源代码和其他工件，并且只有在主体同意的情况下才使用个人数据。

有许多伦理准则，但它们大多数得出相同的结论——在使用个人数据时需要同意。因此，我们应该谨慎使用开源数据，仅分析我们研究必需的数据。在分析来自人们的数据时，始终要获得知情同意。正如文章所建议的，我们可以遵循以下指导原则来研究代码库。

研究开源代码库有一些指导原则：

+   当研究源代码时，我们主要需要关注许可证，并确保我们不会将源代码与人们的个人数据混合。

+   当研究提交时，我们需要尊重做出贡献的人的身份，因为他们可能没有意识到自己正在被研究。我们必须确保我们的研究不会对这些个人造成伤害或损害。如果我们需要研究个人数据，那么我们需要向机构伦理委员会申请批准并获得明确同意。在某些许可证的情况下，许可证扩展到提交中的消息——例如，Apache License 2.0：“*任何形式的电子、口头或书面通信发送给许可方或其代表，包括但不限于在电子邮件列表、源代码控制系统和问题* *跟踪系统*中的通信*”。

+   当挖掘 IDE 事件时，我们应该获得同意，因为个人可能没有意识到自己正在被研究。与从公开仓库中收集数据的研究相比，IDE 事件可能包含更多个人数据，因为这些工具与用户的软件生态系统集成。

+   **在挖掘构建日志时**：这里适用的原则与提交时相同，因为这些可以很容易地联系起来。

+   **在挖掘 Stack Overflow 时**：尽管 Stack Overflow 许可允许我们进行某些研究，因为用户需要允许这样做（根据使用条款），但我们需要确保研究的好处与分析数据相关的风险之间有一个平衡，这些数据通常是自由文本和一些源代码。

+   **在挖掘问题跟踪器时**：一些问题跟踪器，如 Debian，提供了分析数据的可能性，但需要与伦理委员会同步。

+   在挖掘邮件列表时，我们需要尝试从伦理委员会获得许可，因为这些列表通常包含个人信息，在使用之前应进行审查。

2012 年由肯纳利和迪特里奇撰写的《Menlo 报告》为信息技术和通信技术（包括软件工程）的伦理研究提供了一系列指导方针。他们定义了四个原则（以下内容引用自报告）：

+   **尊重个人**：作为研究对象的参与是自愿的，并基于知情同意；将个人视为自主代理并尊重他们决定自身最佳利益的权力；尊重那些尚未成为研究目标但受到影响的人；自主能力减弱、无法自行做出决定的人有权得到保护。

+   **仁慈**：不造成伤害；最大化可能的利益并最小化可能的伤害；系统地评估伤害风险和利益。

+   **正义**：每个人在如何被对待的问题上都应得到平等的考虑，研究的好处应根据个人的需求、努力、社会贡献和功绩公平分配；受影响者的选择应公平，负担应在受影响者之间公平分配。

+   **尊重法律和公共利益**：进行法律尽职调查；在方法和结果上保持透明；对行动负责。

报告鼓励研究人员和工程师进行利益相关者识别以及他们的观点和考虑因素的识别。其中一项考虑因素是存在恶意行为者，他们可能会滥用我们工作的成果并/或伤害我们分析的个人数据。我们必须始终保护我们分析数据的人，并确保我们不会给他们造成任何伤害。同样适用于我们与之合作的组织。

# 从人类收集的数据背后的伦理

在欧洲，规范我们如何使用数据的最重要的法律框架之一是**通用数据保护条例**（**GDPR**）([`eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679`](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679))。它规范了处理个人数据的范围，并要求组织获得收集、处理和使用个人数据的许可，同时要求组织为个人提供撤销许可的能力。该条例是旨在保护个人（我们）免受那些有能力收集和处理我们数据的公司滥用的最严格国际条例。

尽管我们使用了来自 GitHub 和类似存储库的大量数据，但我们也在存储数据的存储库中。其中之一是 Zenodo，它越来越多地被用来存储数据集。它的使用条款要求我们获得适当的权限。以下是它的使用条款（[`about.zenodo.org/terms/）：`](https://about.zenodo.org/terms/):）

```py
1\. Zenodo is an open dissemination research data repository for the preservation and making available of research, educational and informational content. Access to Zenodo's content is open to all, for non-military purposes only.
2\. Content may be uploaded free of charge by those without ready access to an organised data centre.
3\. The uploader is exclusively responsible for the content that they upload to Zenodo and shall indemnify and hold CERN free and harmless in connection with their use of the service. The uploader shall ensure that their content is suitable for open dissemination, and that it complies with these terms and applicable laws, including, but not limited to, privacy, data protection and intellectual property rights *. In addition, where data that was originally sensitive personal data is being uploaded for open dissemination through Zenodo, the uploader shall ensure that such data is either anonymised to an appropriate degree or fully consent cleared **.
4\. Access to Zenodo, and all content, is provided on an "as-is" basis. Users of content ("Users") shall respect applicable license conditions. Download and use of content from Zenodo does not transfer any intellectual property rights in the content to the User.
5\. Users are exclusively responsible for their use of content, and shall indemnify and hold CERN free and harmless in connection with their download and/or use. Hosting and making content available through Zenodo does not represent any approval or endorsement of such content by CERN.
6\. CERN reserves the right, without notice, at its sole discretion and without liability, (i) to alter, delete or block access to content that it deems to be inappropriate or insufficiently protected, and (ii) to restrict or remove User access where it considers that use of Zenodo interferes with its operations or violates these Terms of Use or applicable laws.
7\. Unless specified otherwise, Zenodo metadata may be freely reused under the CC0 waiver.
8\. These Terms of Use are subject to change by CERN at any time and without notice, other than through posting the updated Terms of Use on the Zenodo website.
* Uploaders considering Zenodo for the storage of unanonymised or encrypted/unencrypted sensitive personal data are advised to use bespoke platforms rather than open dissemination services like Zenodo for sharing their data
** See further the user pages regarding uploading for information on anonymisation of datasets that contain sensitive personal information.
```

重要的是关于内容责任的部分：

+   如果你上传内容，你对其负有完全责任。

+   我们必须确保我们的内容适合公众查看，并遵守所有相关法律和这些条款。这包括与隐私、数据保护和**知识产权**（**IP**）相关的法律和条款。

+   如果我们正在上传敏感的个人数据，我们必须确保它要么被适当匿名化，要么我们有权分享它。

我无法强调这一点——我们在 Zenodo 提供的信息是我们的责任，因此我们应该确保我们不会通过向每个人开放信息而造成任何伤害。如果有未匿名化的数据，我们应该考虑其他类型的存储；例如，需要身份验证或访问控制的存储，因此我的下一个最佳实践。

最佳实践 #72

任何个人数据都应该存储在身份验证和访问控制之后，以防止恶意行为者访问它。

尽管我们可能有权使用个人、非匿名化数据，但我们应将此类数据存储在身份验证之后。我们需要保护这些数据背后的个人。我们还需要使用访问控制和监控，以防我们需要回溯谁访问了数据；例如，当出现错误时。

# 合同和法律义务

为了完成这一章，我想讨论最后一个话题。尽管有大量数据可用，但我们必须确保我们做了尽职调查，并找出哪些合同和义务适用于我们。

许可证是一种合同类型，但并非唯一。几乎所有大学都会对研究人员施加合同和义务。这可能包括需要从伦理审查委员会请求许可或需要使数据可供其他研究人员审查。

专业行为准则是一种义务的另一种类型；例如，来自 ACM 的准则（[`www.acm.org/code-of-ethics`](https://www.acm.org/code-of-ethics)）。这些行为准则通常源自于《纽伦堡法典》，并要求我们确保我们的工作是为了社会的利益。

最后，当与商业组织合作时，我们可能需要签署所谓的**保密协议**（**NDA**）。此类协议通常是为了确保我们未经事先许可不泄露信息。它们常常被误解为需要隐藏信息，但在大多数情况下，这意味着我们需要确保我们报告的专有信息不会损害组织。在大多数情况下，我们可能需要确保我们的报告是关于一般实践，而不是特定公司。如果我们发现与我们的工业合作伙伴存在缺陷，我们需要与他们讨论，并帮助他们改进——因为我们需要为社会的最佳利益而工作。

因此，我强烈建议您了解哪些行为准则、义务和合同适用于您。

# 参考文献

+   *Code, N., 《纽伦堡法典》。在控制委员会法律下的纽伦堡军事法庭对战争罪犯的审判，1949 年。第 10 卷（1949 年）：第 181-2 页。*

+   *Wohlin, C. 等人，《软件工程中的实验》。2012 年：Springer Science & Business Media。*

+   *Gold, N.E. 和 J. Krinke，软件仓库挖掘中的伦理。实证软件工程，2022 年。第 27 卷第 1 期：第 17 页。*

+   *Kenneally, E. 和 D. Dittrich，《门洛报告》：指导信息和通信技术研究的伦理原则。可在 SSRN 2445102 找到，2012 年。*
