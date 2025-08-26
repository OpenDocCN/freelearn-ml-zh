# 探索 JavaScript 的潜力

本章我们将涵盖以下主题：

+   为什么选择 JavaScript？

+   为什么选择机器学习，为什么现在？

+   JavaScript 的优缺点

+   CommonJS 创新计划

+   Node.js

+   TypeScript 语言

+   ES6 的改进

+   准备开发环境

# 为什么选择 JavaScript？

我从 2010 年开始用 JavaScript 写有关 **机器学习**（**ML**）的文章。当时，Node.js 还很新，JavaScript 正开始作为一种语言崭露头角。在互联网的大部分历史中，JavaScript 被视为一种玩具语言，用于在网页上创建简单的动态交互。

随着 2005 年 **Prototype JavaScript 框架** 的发布，人们对 JavaScript 的看法开始改变，该框架旨在简化 AJAX 请求并帮助开发者处理跨浏览器的 `XMLHttpRequest`。Prototype 框架还引入了熟悉的美元函数作为 `document.getElementById` 的别名，例如 `$(“myId”)`。

一年后，John Resig 发布了广受欢迎的 jQuery 库。在撰写本文时，[w3techs.com](https://w3techs.com/) 报告称，jQuery 被用于 96% 的已知 JavaScript 库的网站（这占所有网站的 73%）。jQuery 致力于使常见的 JavaScript 操作跨浏览器兼容且易于实现，为全球的网页开发者带来了重要的工具，如 AJAX 请求、**文档对象模型**（**DOM**）遍历和操作，以及动画。

然后，在 2008 年，Chrome 浏览器和 Chrome V8 JavaScript 引擎被发布。Chrome 和 V8 引入了相对于旧浏览器的显著性能提升：JavaScript 现在变得更快，这主要归功于 V8 引擎的创新即时编译器，它可以直接从 JavaScript 构建机器代码。

随着 jQuery 和 Chrome 浏览器的兴起，JavaScript 的受欢迎程度逐渐增加。开发者们历史上从未真正喜欢 JavaScript 这种编程语言，但有了 jQuery 的加入，在快速且现代的浏览器上运行，很明显 JavaScript 是一个未被充分利用的工具，并且能够完成比之前更多的事情。

2009 年，JavaScript 开发者社区决定将 JavaScript 从浏览器环境解放出来。CommonJS 创新计划在当年早期启动，几个月后 Node.js 随之诞生。CommonJS 模块的目标是开发一个标准库，并改善 JavaScript 的生态系统，使其能够在浏览器环境之外使用。作为这项努力的一部分，CommonJS 标准化了模块加载接口，允许开发者构建可以与他人共享的库。

2009 年中旬 Node.js 的发布，通过为 JavaScript 开发者提供了一个新的思考范式——将 JavaScript 作为服务器端语言，震撼了 JavaScript 世界。将 Chrome V8 引擎打包在内，使得 Node.js 出奇地快，尽管 V8 引擎并不应该独占软件性能的功劳。Node.js 实例使用事件循环来处理请求，因此尽管它是单线程的，但它可以处理大量的并发连接。

JavaScript 在服务器端的创新之处，其令人惊讶的性能，以及 npm 注册表的早期引入，让开发者能够发布和发现模块，这些都吸引了成千上万的开发者。与 Node.js 一起发布的标准库主要是低级 I/O API，开发者们竞相发布第一个优秀的 HTTP 请求包装器，第一个易于使用的 HTTP 服务器，第一个高级图像处理库，等等。JavaScript 生态系统的快速早期增长，让那些不愿采用新技术的开发者们产生了信心。JavaScript 第一次被视为一种真正的编程语言，而不仅仅是由于网络浏览器而容忍的东西。

当 JavaScript 作为编程平台逐渐成熟时，Python 社区正忙于研究机器学习，这在一定程度上受到了谷歌在市场上的成功启发。基础且非常流行的数值处理库 NumPy 于 2006 年发布，尽管它以某种形式存在了十年。一个名为**scikit-learn**的机器学习库于 2010 年发布，那是我决定开始向 JavaScript 开发者教授机器学习的时刻。

Python 中机器学习的流行以及使用工具（如 scikit-learn）构建和训练模型的便捷性，让我和许多人感到惊讶。在我看来，这种流行度的激增引发了一个机器学习泡沫；因为模型构建和运行变得如此容易，我发现许多开发者实际上并不了解他们所使用的算法和技术的工作原理。许多开发者哀叹他们的模型表现不佳，却不知道他们自己才是链条中的薄弱环节。

在当时，机器学习被视为神秘、神奇、学术性的，只有少数天才才能接触，而且只有 Python 开发者才能接触。我的看法不同。机器学习只是没有魔法涉及的一类算法。大多数算法实际上很容易理解和推理！

我不想向开发者展示如何在 Python 中导入贝叶斯，而是想展示如何从头开始构建算法，这是建立直觉的重要一步。我还想让我学生很大程度上忽略当时流行的 Python 库，因为我想要强化这样一个观念：机器学习算法可以用任何语言编写，Python 不是必需的。

我选择了 JavaScript 作为我的教学平台。坦白说，我选择 JavaScript 部分原因是因为当时很多人认为它是一种*糟糕*的语言。我的信息是*机器学习很简单，你甚至可以用 JavaScript 来做！* 幸运的是，对于我来说，Node.js 和 JavaScript 都变得极其流行，我的早期关于 JavaScript 中机器学习的文章在接下来的几年里被超过一百万名好奇的开发者阅读。

我还选择 JavaScript 部分原因是因为我不想让机器学习被视为只有学者、计算机科学家或甚至大学毕业生才能使用的工具。我相信，并且仍然相信，只要足够练习和重复，任何有能力的开发者都可以彻底理解这些算法。我选择 JavaScript 是因为它让我能够接触到新的前端和全栈 Web 开发者群体，其中许多人自学成才或从未正式学习过计算机科学。如果目标是使机器学习领域去神秘化和民主化，我觉得接触 Web 开发者社区比接触当时整体更熟悉机器学习的后端 Python 程序员社区要好得多。

Python 一直是，并且仍然是机器学习的首选语言，部分原因是语言的成熟度，部分原因是生态系统的成熟度，部分原因是 Python 早期机器学习努力的积极反馈循环。然而，JavaScript 世界的最新发展使得 JavaScript 对机器学习项目更具吸引力。我认为在几年内，我们将看到 JavaScript 在机器学习领域迎来一场重大的复兴，特别是在笔记本电脑和移动设备变得越来越强大，JavaScript 本身也日益流行的情况下。

# 为什么是机器学习，为什么是现在？

一些机器学习技术早在计算机本身出现之前就已经存在，但许多我们现在使用的现代机器学习算法都是在 20 世纪 70 年代和 80 年代发现的。当时它们很有趣但不实用，主要局限于学术界。

什么变化使得机器学习在流行度上有了巨大的提升？首先，计算机终于足够快，可以运行非平凡的神经网络和大型机器学习模型。然后发生了两件事：谷歌和**亚马逊网络服务**（**AWS**）。谷歌以一种非常明显的方式证明了机器学习对市场的价值，然后 AWS 使可扩展的计算和存储资源变得容易获得（AWS 使其民主化并创造了新的竞争）。

谷歌的 PageRank 算法，这个为谷歌搜索提供动力的机器学习算法，让我们了解了机器学习的商业应用。谷歌的创始人谢尔盖和拉里向世界宣布，他们搜索引擎和随之而来的广告业务的巨大成功归功于 PageRank 算法：一个相对简单的线性代数方程，包含一个巨大的矩阵。

注意，神经网络也是相对简单的线性代数方程，包含一个巨大的矩阵。

那就是所有荣耀中的机器学习（ML）；大数据带来了深刻的洞察力，这转化为巨大的市场成功。这使得全世界对机器学习产生了经济上的兴趣。

AWS 通过推出 EC2 和按小时计费，民主化了计算资源。研究人员和早期阶段的初创公司现在可以快速启动大型计算集群，训练他们的模型，并将集群规模缩小，避免了对强大服务器的巨额资本支出。这创造了新的竞争，并产生了一代专注于机器学习的初创公司、产品和倡议。

近期，机器学习在开发者和商业社区中又掀起了一股热潮。第一代专注于机器学习的初创公司和产品现在已经成熟，并在市场上证明了机器学习的价值，在许多情况下，这些公司正在接近或超越其竞争对手。公司保持市场竞争力的愿望推动了机器学习解决方案的需求。

2015 年末，谷歌推出了神经网络的库**TensorFlow**，通过民主化神经网络的方式激发了开发者们的热情，这与 EC2 民主化计算能力的方式非常相似。此外，那些专注于开发者的第一代初创公司也已经成熟，现在我们可以通过简单的 API 请求 AWS 或**Google Cloud Platform**（**GCP**），在图像上运行整个预训练的**卷积神经网络**（**CNN**），并告诉我我是否在看着一只猫、一个女人、一个手提包、一辆车，或者同时看着这四者。

随着机器学习的民主化，它将逐渐失去其竞争优势，也就是说，公司将不再能够使用机器学习来超越竞争，因为他们的竞争对手也将使用机器学习。现在，该领域的每个人都使用相同的算法，竞争变成了数据战。如果我们想在技术上保持竞争，如果我们想找到下一个 10 倍改进，那么我们可能需要等待，或者最好是促成下一个重大的技术突破。

如果机器学习在市场上的成功不是如此之大，那么这个故事就结束了。所有重要的算法都将为所有人所知，战斗将转移到谁能够收集到最好的数据，在自己的园地里筑起围墙，或者最好地利用自己的生态系统。

但是，将 TensorFlow 这样的工具引入市场改变了这一切。现在，神经网络已经实现了民主化。构建模型、在 GPU 上训练和运行它以及生成真实结果出奇地简单。围绕神经网络的学术迷雾已经消散，现在成千上万的开发者正在尝试各种技术、进行实验和改进。这将引发机器学习（ML）的第二次重大浪潮，尤其是专注于神经网络。新一代以机器学习和神经网络为重点的初创公司和产品正在诞生，几年后当它们成熟时，我们应该会看到许多重大突破，以及一些突破性的公司。

我们看到的每一个新的市场成功都将创造对机器学习（ML）开发者的需求。人才库的增加和技术的民主化导致技术突破。每一次新的技术突破进入市场都会创造新的市场成功，并且随着该领域的加速发展，这个循环将持续下去。我认为，纯粹从经济角度来看，我们真的正走向一个**人工智能**（**AI**）的繁荣。

# JavaScript 的优势和挑战

尽管我对 JavaScript 在机器学习（ML）未来的乐观态度，但今天的大多数开发者仍然会选择 Python 来开发他们的新项目，几乎所有的大型生产系统都是用 Python 或其他更典型的机器学习语言开发的。

JavaScript，就像任何其他工具一样，有其优点和缺点。历史上对 JavaScript 的许多批评都集中在几个常见的主题上：类型强制转换中的奇怪行为、原型面向对象模型、组织大型代码库的困难，以及使用许多开发者称之为*回调地狱*的深度嵌套异步函数调用。幸运的是，大多数这些历史上的抱怨都通过引入**ES6**（即**ECMAScript 2015**），这个 JavaScript 语法的最新更新而得到了解决。

尽管最近语言有所改进，但大多数开发者仍然会建议不要使用 JavaScript 进行机器学习，原因之一是生态系统。Python 的机器学习生态系统如此成熟和丰富，以至于很难为选择其他生态系统找到理由。但这种逻辑是自我实现的也是自我挫败的；如果我们想让 JavaScript 的生态系统成熟，我们需要勇敢的人去跨越障碍，解决真实的机器学习问题。幸运的是，JavaScript 已经连续几年成为 GitHub 上最受欢迎的编程语言，并且几乎在所有指标上都在增长。

使用 JavaScript 进行机器学习有一些优势。其普及度是一个；虽然目前 JavaScript 中的机器学习并不非常流行，但 JavaScript 语言本身是流行的。随着机器学习应用需求的增加，以及硬件变得更快更便宜，机器学习在 JavaScript 世界中的普及是自然而然的事情。学习 JavaScript 的通用资源很多，维护 Node.js 服务器和部署 JavaScript 应用也是如此。**Node 包管理器（npm**）生态系统也很大，仍在增长，尽管成熟的机器学习包并不多，但有许多构建良好、有用的工具即将成熟。

使用 JavaScript 的另一个优势是语言的通用性。现代网络浏览器本质上是一个可携带的应用程序平台，它允许你在几乎任何设备上运行你的代码，基本上无需修改。像**electron**（虽然许多人认为它很臃肿）这样的工具允许开发者快速开发并部署可下载的桌面应用程序到任何操作系统。Node.js 让你可以在服务器环境中运行你的代码。React Native 将你的 JavaScript 代码带到原生移动应用程序环境中，并可能最终允许你开发桌面应用程序。JavaScript 不再局限于动态网络交互，现在它是一种通用、跨平台的编程语言。

最后，使用 JavaScript 使得机器学习（ML）对网页和前端开发者变得可访问，这个群体在历史上一直被排除在机器学习讨论之外。由于服务器是计算能力所在的地方，因此服务器端应用通常是机器学习工具的首选。这一事实在历史上使得网页开发者难以进入机器学习领域，但随着硬件的改进，即使是复杂的机器学习模型也可以在客户端运行，无论是桌面还是移动浏览器。

如果网页开发者、前端开发者和 JavaScript 开发者今天开始学习机器学习，那么这个社区将能够改善我们所有人明天可用的机器学习工具。如果我们采用这些技术并使其民主化，让尽可能多的人接触到机器学习背后的概念，我们最终将提升社区并培养下一代机器学习研究人员。

# CommonJS 倡议

2009 年，一位名叫 Kevin Dangoor 的 Mozilla 工程师意识到，服务器端 JavaScript 需要大量的帮助才能变得有用。服务器端 JavaScript 的概念已经存在，但由于许多限制，尤其是 JavaScript 生态系统方面的限制，它并不受欢迎。

在 2009 年 1 月的一篇博客文章中，Dangoor 列举了一些 JavaScript 需要帮助的例子。他写道，JavaScript 生态系统需要一个标准库和标准接口，用于文件和数据库访问等。此外，JavaScript 环境需要一个方法来打包、发布和安装库和依赖项，以便其他人可以使用，还需要一个包仓库来托管所有上述内容。

所有这些最终导致了**CommonJS**倡议的诞生，它对 JavaScript 生态系统最显著的贡献是 CommonJS 模块格式。如果你有任何 Node.js 的工作经验，你可能已经熟悉 CommonJS：你的`package.json`文件是用 CommonJS 模块包规范格式编写的，而在一个文件中编写`var app = require(‘./app.js’)`并在`app.js`中写入`module.exports = App`，就是在使用 CommonJS 模块规范。

模块和包的标准化为 JavaScript 的普及率显著提升铺平了道路。开发者现在可以使用模块来编写跨越多个文件的复杂应用程序，而不会污染全局命名空间。包和库的开发者能够构建和发布比 JavaScript 标准库更高层次的抽象库。Node.js 和 npm 很快就会抓住这些概念，围绕包共享构建一个主要生态系统。

# Node.js

2009 年 Node.js 的发布可能是 JavaScript 历史上最重要的时刻之一，尽管没有前一年 Chrome 浏览器和 Chrome 的 V8 JavaScript 引擎的发布，这一时刻是不可能实现的。

那些还记得 Chrome 浏览器发布的人也会认识到为什么 Chrome 能在浏览器大战中占据主导地位：Chrome 速度快，设计简约，风格现代，易于开发，而且 JavaScript 在 Chrome 上的运行速度比在其他浏览器上要快得多。

Chrome 背后是开源的 Chromium 项目，该项目反过来又开发了**V8** JavaScript 引擎。V8 为 JavaScript 世界带来的创新是其新的执行模型：V8 包含一个即时编译器，它将 JavaScript 直接转换为原生机器代码，而不是实时解释 JavaScript。这一策略取得了成功，其卓越的性能和开源状态使得其他人也开始将其用于自己的目的。

Node.js 采用了 V8 JavaScript 引擎，在其周围添加了一个事件驱动架构，并添加了用于磁盘和文件访问的低级 I/O API。事件驱动架构最终证明是一个关键决策。其他服务器端语言和技术，如 PHP，通常使用线程池来管理并发请求，每个线程在处理请求时本身会阻塞。Node.js 是一个单线程进程，但使用事件循环避免了阻塞操作，并更倾向于异步、回调驱动的逻辑。尽管许多人认为 Node.js 的单线程特性是一个缺点，但 Node.js 仍然能够以良好的性能处理许多并发请求，这对吸引开发者到这个平台来说已经足够了。

几个月后，npm 项目发布了。在 CommonJS 所取得的基石工作上，npm 允许包开发者将他们的模块发布到一个集中的注册表（称为 **npm 注册表**），并允许包消费者使用 npm 命令行工具安装和维护依赖项。

如果没有 npm，Node.js 很可能无法进入主流。Node.js 服务器本身提供了 JavaScript 引擎、事件循环和一些低级 API，但随着开发者处理更大的项目，他们往往希望有更高层次的抽象。在发起 HTTP 请求或从磁盘读取文件时，开发者并不总是需要担心二进制数据、编写头信息和其他低级问题。npm 和 npm 注册表让开发者社区能够以模块的形式编写和分享他们自己的高级抽象，其他开发者可以简单地安装并 `require()` 这些模块。

与其他通常内置高级抽象的编程语言不同，Node.js 允许专注于提供低级构建块，而社区则负责其他部分。社区通过构建出色的抽象，如 `Express.js` 网络应用程序框架、`Sequelize ORM` 以及数以万计的其他库，这些库只需简单的 `npm install` 命令即可使用。

随着 Node.js 的出现，那些没有先前服务器端语言知识的 JavaScript 开发者现在能够构建完整的全栈应用程序。前端代码和后端代码现在可以由相同的开发者使用同一种语言编写。

有雄心的开发者现在用 JavaScript 构建整个应用程序，尽管他们在路上遇到了一些问题和解决方案。完全用 JavaScript 编写的单页应用程序变得流行，但也变得难以模板化和组织。社区通过构建框架来回应，例如 **Backbone.js**（Angular 和 React 等框架的精神前辈）、**RequireJS**（CommonJS 和 AMD 模块加载器）以及模板语言如 **Mustache**（JSX 的精神前辈）。

当开发者遇到单页应用程序的 SEO 问题，他们发明了**同构应用程序**的概念，或者能够在服务器端（以便网络爬虫可以索引内容）和客户端（以保持应用程序快速和 JavaScript 驱动）渲染的代码。这导致了更多 JavaScript 框架如**MeteorJS**的发明。

最终，构建单页应用的 JavaScript 开发者意识到，通常他们的服务器端和数据库需求很轻量，只需要认证、数据存储和检索。这导致了无服务器技术或**数据库即服务**（DBaaS）平台如**Firebase**的发展，这反过来又为移动 JavaScript 应用程序的普及铺平了道路。Cordova/PhoneGap 项目大约在同一时间出现，允许开发者将他们的 JavaScript 代码包裹在原生的 iOS 或 Android WebView 组件中，并将他们的 JavaScript 应用程序部署到移动应用商店。

在本书的整个过程中，我们将非常依赖 Node.js 和 npm。本书中的大多数示例将使用 npm 上可用的 ML 包。

# TypeScript 语言

在 npm 上开发和共享新包并不是 JavaScript 流行带来的唯一结果。JavaScript 作为主要编程语言的日益普及导致许多开发者哀叹缺乏 IDE 和语言工具支持。历史上，IDE 在 C 和 Java 等编译和静态类型语言的开发者中更受欢迎，因为这些类型的语言更容易解析和静态分析。直到最近，才出现了针对 JavaScript 和 PHP 等语言的优秀 IDE，而 Java 已经有多年针对它的 IDE。

微软希望为他们的大规模 JavaScript 项目提供更好的工具和支持，但 JavaScript 语言本身存在一些问题，阻碍了这一进程。特别是，JavaScript 的动态类型（例如，`var number` 可能一开始是整数 **5**，但后来被分配给一个对象）排除了使用静态分析工具来确保类型安全，并且也使得 IDE 难以找到正确的变量或对象来自动完成。此外，微软希望有一个基于类和接口的面向对象范式，但 JavaScript 的面向对象编程范式是基于**原型**的，而不是类。

因此，微软发明了 TypeScript 语言，以支持大规模的 JavaScript 开发工作。TypeScript 将类、接口和静态类型引入了语言。与 Google 的 Dart 不同，微软确保 TypeScript 总是 JavaScript 的严格超集，这意味着所有有效的 JavaScript 也是有效的 TypeScript。TypeScript 编译器在编译时进行静态类型检查，帮助开发者尽早捕获错误。对静态类型的支持还有助于 IDE 更准确地解释代码，从而为开发者提供更好的体验。

TypeScript 对 JavaScript 语言的早期改进中，有一些已经被 ECMAScript 2015（或我们称之为 ES6）所取代。例如，TypeScript 的模块加载器、类语法和箭头函数语法已被 ES6 所吸收，现在 TypeScript 只使用这些结构的 ES6 版本；然而，TypeScript 仍然为 JavaScript 带来了静态类型，这是 ES6 无法实现的。

我在这里提到 TypeScript，因为虽然我们不会在本书的示例中使用 TypeScript，但我们考察的一些机器学习库的示例是用 TypeScript 编写的。

例如，在 `deeplearn.js` 教程页面上的一个示例显示了如下代码：

```py
const graph = new Graph();
 // Make a new input in the graph, called 'x', with shape [] (a Scalar).
 const x: Tensor = graph.placeholder('x', []);
 // Make new variables in the graph, 'a', 'b', 'c' with shape [] and   
    random
 // initial values.
 const a: Tensor = graph.variable('a', Scalar.new(Math.random()));
 const b: Tensor = graph.variable('b', Scalar.new(Math.random()));
 const c: Tensor = graph.variable('c', Scalar.new(Math.random()));
```

语法看起来像 ES6 JavaScript，除了在 `const x: Tensor = …:` 中看到的新的冒号表示法，这段代码是在告诉 TypeScript 编译器 `const x` 必须是 `Tensor` 类的实例。当 TypeScript 编译此代码时，它首先检查 `x` 在所有使用的地方是否期望是 `Tensor`（如果不是，将抛出错误），然后简单地丢弃编译到 JavaScript 时的类型信息。将前面的 TypeScript 代码转换为 JavaScript 只需从变量定义中移除冒号和 `Tensor` 关键字即可。

您可以在跟随本书的过程中在自己的示例中使用 TypeScript，但是您必须更新我们稍后设置的构建过程以支持 TypeScript。

# ES6 的改进

定义 JavaScript 语言本身的规范的 ECMAScript 委员会在 2015 年 6 月发布了一个新的规范，称为 ECMAScript 6/ECMAScript 2015。这个新标准简称为 **ES6**，是对 JavaScript 编程语言的重大修订，并增加了一些旨在使 JavaScript 程序开发更容易的新范式。

虽然 ECMAScript 定义了 JavaScript 语言的规范，但语言的实际实现依赖于浏览器供应商和各种 JavaScript 引擎的维护者。ES6 本身只是一个指南，由于浏览器供应商各自有自己的时间表来实现新的语言特性，JavaScript 语言及其实现略有分歧。ES6 定义的特性，如类，在主要浏览器中不可用，但开发者仍然想使用它们。

来到 **Babel**，JavaScript 转译器。Babel 可以读取和解析不同的 JavaScript 版本（如 ES6、ES7、ES8 和 React JSX），并将其转换为或编译为浏览器标准的 ES5。即使今天，浏览器厂商还没有完全实现 ES6，所以 Babel 对于希望编写 ES6 代码的开发者来说仍然是一个必不可少的工具。

本书中的示例将使用 ES6。如果你还不熟悉新的语法，以下是本书中将使用的一些主要特性。

# Let 和 const

在 ES5 JavaScript 中，我们使用 `var` 关键字来定义变量。在大多数情况下，`var` 可以简单地替换为 `let`，这两个构造之间的主要区别是变量相对于代码块的可见性。以下来自 **MDN 网络文档**（或之前称为 **Mozilla 开发者网络**）的例子（[`developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/let`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/let)）展示了这两个之间的微妙差异：

```py
function varTest() {
  var x = 1;
  if (true) {
    var x = 2;  // same variable!
    console.log(x);  // 2
  }
  console.log(x);  // 2
 }

 function letTest() {
  let x = 1;
  if (true) {
    let x = 2;  // different variable
    console.log(x);  // 2
  }
  console.log(x);  // 1
 }
```

因此，虽然你必须在像前面那样的情况下更加小心，但在大多数情况下，你只需将 `var` 替换为 `let`。

与 `let` 不同，`const` 关键字定义了一个常量变量；也就是说，你无法在以后重新分配用 `const` 初始化的变量。例如，以下代码会导致一个类似于 `invalid assignment to const a` 的错误信息：

```py
const a = 1;
a = 2;
```

另一方面，使用 `var` 或 `let` 来定义 `a` 的相同代码将成功运行。

注意，如果 `a` 是一个对象，你可以修改 `a` 的对象属性。

以下代码将成功运行：

```py
const obj = {};
obj.name = ‘My Object’;
```

然而，尝试重新定义对象，如 `obj = {name: “other object”}`，会导致错误。

我发现，在大多数编程环境中，`const` 通常比 `let` 更合适，因为大多数你使用的变量永远不会需要重新定义。我的建议是尽可能多地使用 `const`，只有在有理由在以后重新定义变量时才使用 `let`。

# 类

在 ES6 中，一个非常受欢迎的变化是类和类的继承的添加。之前，JavaScript 中的面向对象编程需要原型继承，这让许多开发者觉得不直观，就像以下 ES5 的例子：

```py
var Automobile = function(weight, speed) {
   this.weight = weight;
   this.speed = speed;
}
Automobile.prototype.accelerate = function(extraSpeed) {
   this.speed += extraSpeed;
}
var RaceCar = function (weight, speed, boost) {
   Automobile.call(this, weight, speed);
   this.boost = boost;
}
RaceCar.prototype = Object.create(Automobile.prototype);
RaceCar.prototype.constructor = RaceCar;
RaceCar.prototype.accelerate = function(extraSpeed) {
  this.speed += extraSpeed + this.boost;
}
```

在前面的代码中，扩展一个对象需要在子类的 `constructor` 函数中调用父类，创建父类原型对象的克隆，并用子类的原型构造函数覆盖父类的原型构造函数。这些步骤被大多数开发者视为不直观且繁重。

然而，使用 ES6 类，代码将看起来像这样：

```py
class Automobile {
 constructor(weight, speed) {
   this.weight = weight;
   this.speeed = speed;
 }
 accelerate(extraSpeed) {
   this.speed += extraSpeed;
 }
}
class RaceCar extends Automobile {
 constructor(weight, speed, boost) {
   super(weight, speed);
   this.boost = boost;
 }
 accelerate(extraSpeed) {
   this.speed += extraSpeed + this.boost;
 }
}
```

前面的语法更符合我们对面向对象编程的预期，并且使继承变得更加简单。

需要注意的是，在底层，ES6 类仍然使用 JavaScript 的原型继承范式。类只是现有系统之上的语法糖，因此这两种方法之间除了代码整洁性外，没有显著的区别。

# 模块导入

ES6 还定义了一个模块导入和导出接口。使用较旧的 CommonJS 方法，模块通过 `module.exports` 构造导出，模块通过 `require(filename)` 函数导入。ES6 方法看起来略有不同。在一个文件中，定义并导出一个类，如下面的代码所示：

```py
Class Automobile {
…
}
export default Automobile
```

在另一个文件中，导入类，如下面的代码所示：

```py
import Automobile from ‘./classes/automobile.js’;
const myCar = new Automobile();
```

目前，Babel 将 ES6 模块编译成与 CommonJS 模块相同的格式，所以如果你使用 Babel，你可以使用 ES6 模块语法或 CommonJS 模块语法。

# 箭头函数

ES5 JavaScript 中的一个奇特、有用但有些令人烦恼的方面是其对异步回调的广泛使用。你可能非常熟悉类似以下这样的 jQuery 代码：

```py
$(“#link”).click(function() {
  var $self = $(this);
  doSomethingAsync(1000, function(resp) {
    $self.addClass(“wasFaded”);
    var processedItems = resp.map(function(item) {
      return processItem(item);
    });
    return shipItems(processedItems);
  });
});
```

我们被迫创建一个名为 `$self` 的变量，因为原始的 `this` 上下文在我们的内部匿名函数中丢失了。我们还因为需要创建三个单独的匿名函数而有大量的样板代码和难以阅读的代码。

箭头函数语法既是帮助我们用更短的语法编写匿名函数的语法糖，也是对函数式编程的更新，它保留了箭头函数内部 `this` 的上下文。

例如，上述代码可以用 ES6 写成如下所示：

```py
$(“#link”).click(function() {
  dozsSomethingAsync(1000, resp => {
    $(this).addClass(“wasFaded”);
    const processedItems = resp.map(item => processItem(Item));
    return shipItems(processedItems);
  });
});
```

你可以在上述代码中看到，我们不再需要 `$self` 变量来保留 `this`，并且我们的 `.map` 调用要简单得多，不再需要 `function` 关键字、括号、大括号或 `return` 语句。

现在让我们看看一些等效函数。让我们看看以下代码：

```py
const double = function(number) {
  return number * 2;
}
```

上述代码类似于：

```py
const double = number => number * 2;
// Is equal to:
const double = (number) => { return number * 2; }
```

在上述示例中，我们可以省略 `number` 参数周围的括号，因为该函数只需要一个参数。如果函数需要两个参数，我们就会像下一个示例中那样需要添加括号。此外，如果我们的函数体只需要一行，我们可以省略函数体的大括号和 `return` 语句。

让我们看看另一个等效示例，具有多个参数，如下面的代码所示：

```py
const sorted = names.sort(function (a, b) {
  return a.localeCompare(b);
});
```

上述代码类似于：

```py
const sorted = names.sort((a, b) => a.localeCompare(b));
```

我发现箭头函数在像上述这样的情况下最有用，当你正在做数据转换，尤其是在使用 `Array.map`、`Array.filter`、`Array.reduce` 和 `Array.sort` 调用具有简单函数体时。由于 jQuery 倾向于使用 `this` 上下文提供数据，而匿名箭头函数不会提供 `this`，因此箭头函数在 jQuery 中不太有用。

# 对象字面量

ES6 对对象字面量进行了一些改进。有几个改进，但你最常看到的是对象属性的隐式命名。在 ES5 中，它将是这样的：

```py
var name = ‘Burak’;
var title = ‘Author’;
var object = {name: name, title: title};
```

在 ES6 中，如果属性名和变量名与前面相同，你可以简化为以下形式：

```py
const name = ‘Burak’;
const title = ‘Author’;
const object = {name, title};
```

此外，ES6 引入了对象扩展运算符，它简化了浅层对象合并。例如，看看以下 ES5 中的代码：

```py
function combinePreferences(userPreferences) {
 var defaultPreferences = {size: ‘large’, mode: ‘view’};
 return Object.assign({}, defaultPreferences, userPreferences);
}
```

上述代码将从`defaultPreferences`创建一个新的对象，并合并`userPreferences`中的属性。将空对象传递给`Object.assign`实例的第一个参数确保我们创建一个新的对象，而不是覆盖`defaultPreferences`（在前面示例中这不是问题，但在实际使用场景中是问题）。

现在，让我们看看 ES6 中的相同代码：

```py
function combinePreferences(userPreferences) {
 var defaultPreferences = {size: ‘large’, mode: ‘view’};
 return {...defaultPreferences, ...userPreferences};
}
```

这种方法与 ES5 示例做的是同样的事情，但在我看来，它比`Object.assign`方法更快、更容易阅读。例如，熟悉 React 和 Redux 的开发者经常在管理 reducer 状态操作时使用对象扩展运算符。

# for...of 函数

在 ES5 中，通过数组中的`for`循环通常使用`for (index in array)`语法，它看起来像这样：

```py
var items = [1, 2, 3 ];
for (var index in items) {
var item = items[index];
…
 }
```

此外，ES6 添加了`for...of`语法，这可以节省你一步，正如你从下面的代码中可以看到的那样：

```py
const items = [1, 2, 3 ];
for (const item of items) {
 …
 }
```

# 承诺

以一种形式或另一种形式，承诺在 JavaScript 中已经存在了一段时间。所有 jQuery 用户都熟悉这个概念。**承诺**是对一个异步生成并在未来可能可用的变量的引用。

如果你之前没有使用某种第三方承诺库或 jQuery 的 deferred，那么在 ES5 中处理事情的方式是接受一个异步方法的回调函数，并在成功完成后运行该回调，如下面的代码所示：

```py
function updateUser(user, settings, onComplete, onError) {
  makeAsyncApiRequest(user, settings, function(response) {
    if (response.isValid()) {
      onComplete(response.getBody());
    } else {
      onError(response.getError())
    }
  });
}
updateUser(user, settings, function(body) { ... }, function(error) { ... });
```

在 ES6 中，你可以返回一个封装异步请求的`Promise`，它要么被解决，要么被拒绝，如下面的代码所示：

```py
function updateUser(user, settings) {
  return new Promise((resolve, reject) => {
    makeAsyncApiRequest(user, settings, function(response) {
      if (response.isValid()) {
        resolve(response.getBody());
      } else {
        reject(response.getError())
      }
    });
  });
}
updateUser(user, settings)
  .then(
    body => { ... },
    error => { ... }
  );
```

承诺的真正力量在于它们可以被当作对象传递，并且承诺处理器可以被链式调用。

# async/await 函数

`async`和`await`关键字不是 ES6 特性，而是 ES8 特性。虽然承诺极大地改进了我们处理异步调用的方式，但承诺也容易受到大量方法链的影响，在某些情况下，迫使我们使用异步范式，而实际上我们只想编写一个异步但看起来像同步函数的函数。

现在让我们看看 MDN 异步函数参考页面上的以下示例（[`developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function)）：

```py
function resolveAfter2Seconds() {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve('resolved');
    }, 2000);
  });
}
async function asyncCall() {
  console.log('calling');
  var result = await resolveAfter2Seconds();
  console.log(result);
  // expected output: "resolved"
}
asyncCall();
```

`resolveAfter2Seconds` 函数是一个普通的 JavaScript 函数，它返回一个 ES6 promise。魔法在于 `asyncCall` 函数，它被 `async` 关键字标记。在 `asyncCall` 内部，我们使用 `await` 关键字调用 `resolveAfter2Seconds`，而不是使用在 ES6 中更熟悉的 promise `.then(result => console.log(result))` 构造。`await` 关键字使我们的 `async` 函数在继续之前等待 promise 解析，并直接返回 `Promise` 的结果。以这种方式，`async`/`await` 可以将使用 promises 的异步函数转换为类似同步函数，这应该有助于保持深层嵌套的 promise 调用和异步函数调用整洁且易于阅读。

`async` 和 `await` 功能是 ES8 的部分，而不是 ES6，所以当我们几分钟内设置 Babel 时，我们需要确保在我们的配置中包含所有新的 ECMAScript 版本，而不仅仅是 ES6。

# 准备开发环境

本书中的示例将使用网络浏览器环境和 Node.js 环境。虽然 Node.js 版本 8 和更高版本支持 ES6+，但并非所有浏览器供应商都完全支持 ES6+ 功能，因此我们将使用 Babel 将所有代码进行转译。

本书将尽可能为所有示例使用相同的工程项目结构，无论它们是在 Node.js 命令行中执行还是在浏览器中运行。因为我们正在尝试标准化这个项目结构，所以并非每个项目都会使用我们在本节中设置的所有功能。

您将需要的工具是：

+   您喜欢的代码编辑器，例如 Vim、Emacs、Sublime Text 或 WebStorm

+   一个最新的网络浏览器，如 Chrome 或 Firefox

+   Node.js 版本 8 LTS 或更高；本书将使用 9.4.0 版本进行所有示例

+   Yarn 软件包管理器（可选；您也可以使用 npm）

+   各种构建工具，如 Babel 和 Browserify

# 安装 Node.js

如果您是 macOS 用户，通过软件包管理器如 **Homebrew** 或 **MacPorts** 安装 Node.js 是最简单的方法。为了与本书中的示例获得最佳兼容性，请安装 9.4.0 或更高版本的 Node.js。

Windows 用户也可以使用 **Chocolatey** 软件包管理器来安装 Node.js，否则您可以遵循 Node.js 当前下载页面上的说明：[`nodejs.org/en/`](https://nodejs.org/en/).

Linux 用户如果通过其发行版的软件包管理器安装 Node.js，应小心谨慎，因为提供的 Node.js 版本可能非常旧。如果您的软件包管理器使用低于 V8 的版本，您可以选择为软件包管理器添加仓库、从源代码构建或根据您的系统安装二进制文件。

安装 Node.js 后，通过在命令行中运行 `node --version` 确保它运行并且是正确的版本。输出将如下所示：

```py
$ node --version
 V9.4.0
```

这也是测试 `npm` 是否正常工作的好时机：

```py
$ npm --version
 5.6.0
```

# 可选安装 Yarn

Yarn 是一个类似于 npm 且与 npm 兼容的包管理工具，尽管我发现它运行更快，更容易使用。如果您在 macOS 上使用 Homebrew，您可以使用`brew install yarn`简单地安装它；否则，请按照 Yarn 安装指南页面上的说明操作（[`yarnpkg.com/en/docs/install#windows-stable`](https://yarnpkg.com/en/docs/install#windows-stable)）。

如果您想使用 npm 而不是 Yarn，您也可以；它们都尊重相同的`package.json`格式，尽管它们在`add`、`require`和`install`等命令的语法上略有不同。如果您使用 npm 而不是 Yarn，只需将命令替换为正确的函数；使用的包名都将相同。

# 创建和初始化示例项目

使用命令行、您喜欢的 IDE 或文件浏览器，在您的机器上创建一个名为`MLinJSBook`的目录，并创建一个名为`Ch1-Ex1`的子目录。

将命令行导航到`Ch1-Ex1`文件夹，并运行命令`yarn init`，它类似于`npm init`，将创建一个`package.json`文件，并提示您输入基本信息。根据提示进行回答，答案并不重要，但是当被提示输入应用程序的入口点时，请输入`dist/index.js`。

接下来，我们需要安装一些我们将用于大多数示例项目的构建工具：

+   `babel-core`：Babel 转译器核心

+   `babel-preset-env`：解析 ES6、ES7 和 ES8 代码的 Babel 解析器预设

+   `browserify`：一个可以将多个文件编译成一个文件的 JavaScript 打包器

+   `babelify`：Browserify 的 Babel 插件

通过以下命令安装这些作为开发环境需求：

```py
yarn add -D babel-cli browserify babelify babel-preset-env
```

# 创建一个 Hello World 项目

为了测试一切是否正在构建和运行，我们将创建一个非常简单的包含两个文件的 Hello World 项目，并添加我们的构建脚本。

首先，在您的`Ch1-Ex1`文件夹下创建两个子文件夹：`src`和`dist`。我们将为所有项目使用此约定：`src`将包含 JavaScript 源代码，`dist`将包含构建源代码以及项目所需的任何附加资源（图像、CSS、HTML 文件等）。

在`src`文件夹中，创建一个名为`greeting.js`的文件，并包含以下代码：

```py
const greeting = name => 'Hello, ' + name + '!';
export default greeting;
```

然后创建另一个名为`index.js`的文件，并包含以下内容：

```py
import greeting from './greeting';
console.log(greeting(process.argv[2] || 'world'));
```

这个小型应用程序测试我们是否可以使用基本的 ES6 语法和模块加载，以及访问传递给 Node.js 的命令行参数。

接下来，打开`Ch1-Ex1`中的`package.json`文件，并将以下部分添加到文件中：

```py
"scripts": {
 "build-web": "browserify src/index.js -o dist/index.js -t [ babelify -  
  -presets [ env ] ]",
 "build-cli": "browserify src/index.js --node -o dist/index.js -t [  
  babelify --presets [ env ] ]",
 "start": "yarn build-cli && node dist/index.js"
},
```

这定义了三个简单的命令行脚本：

+   `Build-web`使用 Browserify 和 Babel 将`src/index.js`接触到的所有内容编译成一个名为`dist/index.js`的单个文件

+   `Build-cli`与`build-web`类似，但它还使用了 Browserify 的 node 选项标志；如果没有这个选项，我们就无法访问传递给 Node.js 的命令行参数

+   `Start`仅适用于 CLI/Node.js 示例，并且构建和运行源代码

你的`package.json`文件现在应该看起来像以下这样：

```py
{
"name": "Ch1-Ex1",
"version": "0.0.1",
"description": "Chapter one example",
"main": "src/index.js",
"author": "Burak Kanber",
"license": "MIT",
"scripts": {
  "build-web": "browserify src/index.js -o dist/index.js -t [ babelify --presets [ env ] ]",
  "build-cli": "browserify src/index.js --node -o dist/index.js -t [ babelify --presets [ env ] ]",
  "start": "yarn build-cli && node dist/index.js"
},
"dependencies": {
  "babel-core": "⁶.26.0",
  "babel-preset-env": "¹.6.1",
  "babelify": "⁸.0.0",
  "browserify": "¹⁵.1.0"
}}
```

让我们对这个简单应用进行一些测试。首先，确保`yarn build-cli`命令可以正常工作。你应该会看到以下类似的内容：

```py
$ yarn build-cli
yarn run v1.3.2
$ browserify src/index.js --node -o dist/index.js -t [ babelify --presets [ env ] ]
Done in 0.59s.
```

在这一点上，确认`dist/index.js`文件已经被构建，并尝试直接运行它，使用以下代码：

```py
$ node dist/index.js
Hello, world!
```

也尝试将你的名字作为参数传递给命令，使用以下代码：

```py
$ node dist/index.js Burak
Hello, Burak!
```

现在，让我们尝试`build-web`命令，如下所示代码。因为这个命令省略了`node`选项，我们预计我们的参数将不会起作用：

```py
$ yarn build-web
yarn run v1.3.2
$ browserify src/index.js -o dist/index.js -t [ babelify --presets [ env ] ]
Done in 0.61s.
$ node dist/index.js Burak
Hello, world!
```

没有使用`node`选项，我们的参数不会被传递到脚本中，并且默认显示`Hello, world!`，这是预期的结果。

最后，让我们使用以下代码测试我们的`yarn start`命令，以确保它构建了应用程序的 CLI 版本，并且也传递了我们的命令行参数，使用以下代码：

```py
$ yarn start "good readers"
yarn run v1.3.2
$ yarn build-cli && node dist/index.js 'good readers'
$ browserify src/index.js --node -o dist/index.js -t [ babelify --presets [ env ] ]
Hello, good readers!
Done in 1.05s.
```

`yarn start`命令成功构建了应用程序的 CLI 版本，并将我们的命令行参数传递给了程序。

我们将尽力为本书中的每个示例使用相同的结构，然而，请注意每个章节的开头，因为每个示例可能需要一些额外的设置工作。

# 摘要

在本章中，我们讨论了 JavaScript 在机器学习中的应用中的重要时刻，从 Google 的推出([`www.google.com/`](https://www.google.com/))开始，到 2017 年底 Google 的`deeplearn.js`库发布结束。

我们讨论了使用 JavaScript 进行机器学习的优势，以及我们面临的挑战，特别是在机器学习生态系统方面。

然后，我们游览了 JavaScript 语言最近最重要的进展，并对最新的 JavaScript 语言规范 ES6 进行了简要介绍。

最后，我们使用 Node.js、Yarn 包管理器、Babel 和 Browserify——这些工具将在本书的其余部分示例中使用——设置了一个示例开发环境。

在下一章中，我们将开始探索和处理数据本身。
