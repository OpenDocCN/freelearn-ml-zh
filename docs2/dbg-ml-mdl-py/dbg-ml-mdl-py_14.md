# 14

# 机器学习最新进展简介

监督学习一直是机器学习在不同行业和应用领域成功应用的重点，直到 2020 年。然而，其他技术，如生成模型，后来引起了机器学习开发者和用户的关注。因此，了解这些技术将有助于你拓宽对机器学习能力的理解，超越监督学习。

本章将涵盖以下主题：

+   生成模型

+   强化学习

+   自监督学习

到本章结束时，你将了解生成模型、**强化学习**（**RL**）和**自监督学习**（**SSL**）的含义、广泛使用的技术和好处。你还将使用 Python 和 PyTorch 练习其中的一些技术。

# 技术要求

以下要求适用于本章，因为它们将帮助你更好地理解概念，能够在项目中使用它们，并使用提供的代码进行实践：

+   Python 库要求：

    +   `torch` >= 2.0.0

    +   `torchvision` >= 0.15.1

    +   `matplotlib` >= 3.7.1

你可以在 GitHub 上找到本章的代码文件，地址为[`github.com/PacktPublishing/Debugging-Machine-Learning-Models-with-Python/tree/main/Chapter14`](https://github.com/PacktPublishing/Debugging-Machine-Learning-Models-with-Python/tree/main/Chapter14)。

# 生成模型

生成模型，或更普遍的生成 AI，为你提供了生成接近预期或参考数据点集或分布的数据的机会，通常称为真实数据。生成模型最成功的应用之一是语言建模。**生成预训练 Transformer**（**GPT**）-4 和 ChatGPT（[`openai.com/blog/ChatGPT`](https://openai.com/blog/ChatGPT)），这是一个建立在 GPT-4 和 GPT-3.5 之上的聊天机器人，以及类似的工具如 Perplexity（[`www.perplexity.ai/`](https://www.perplexity.ai/)），引起了工程师、科学家、金融和医疗保健等不同行业的人士以及许多其他生成模型相关岗位的人士的兴趣。当使用 Chat-GPT 或 GPT-4 时，你可以提出一个问题或提供询问的描述，称为提示，然后这些工具会生成一系列陈述或数据来为你提供你请求的答案、信息或文本。

除了在文本生成中成功应用生成模型之外，许多其他生成模型的应用可以帮助你在工作或学习中。例如，GPT-4 及其之前的版本或其他类似模型，如 LLaMA（Touvron 等，2023 年），可用于代码生成和补全（[`github.com/features/copilot/`](https://github.com/features/copilot/) 和 [`github.com/sahil280114/codealpaca`](https://github.com/sahil280114/codealpaca)）。你可以编写你感兴趣生成的代码，它会为你生成相应的代码。尽管生成的代码可能并不总是按预期工作，但通常在经过几次尝试后，它至少接近预期。

生成模型还有许多其他成功的应用，例如在图像生成（[`openai.com/product/dall-e-2`](https://openai.com/product/dall-e-2)）、药物发现（Cheng 等，2021 年）、时尚设计（Davis 等，2023 年）、制造业（Zhao 等，2023 年）等领域。

从 2023 年开始，许多传统商业工具和服务开始整合生成 AI 功能。例如，你现在可以使用生成 AI 在 Adobe Photoshop 中编辑照片，只需用简单的英语说明你需要什么（[`www.adobe.com/ca/products/photoshop/generative-fill.html`](https://www.adobe.com/ca/products/photoshop/generative-fill.html)）。WolframAlpha 也将它的符号计算能力与生成 AI 结合，你可以用简单的英语请求特定的符号过程（[`www.wolframalpha.com/input?i=Generative+Adversarial+Networks`](https://www.wolframalpha.com/input?i=Generative+Adversarial+Networks)）。可汗学院（[`www.khanacademy.org/`](https://www.khanacademy.org/））制定了一种策略，帮助教师和学生从生成 AI 中受益，特别是 ChatGPT，而不是对学生的教育造成伤害。

这些成功故事是通过依赖为生成模型设计的不同深度学习技术实现的，我们将在下面简要回顾。

## 生成深度学习技术

在 PyTorch 或其他深度学习框架（如 TensorFlow）中，有多种生成模型方法可供使用。在这里，我们将回顾其中的一些，以帮助你开始了解它们是如何工作的，以及你如何在 Python 中使用它们。

### 基于 Transformer 的文本生成

你已经了解到，2017 年引入的转换器（Vaswani et al., 2017）被用于生成最近最成功的语言模型，这在*第十三章*“高级深度学习技术”中有详细描述。然而，这些模型并不仅限于像翻译这样的传统自然语言处理任务，它们还可以用于生成建模，帮助我们生成有意义的文本，例如，回答我们提出的问题。这正是 GPT 模型、Chat-GPT 以及许多其他生成语言模型背后的方法。提供简短文本作为提问或问题的过程也被称为提示（prompting），在这个过程中，我们需要提供一个好的提示以获得好的答案。我们将在“基于文本的生成模型的提示工程”部分讨论最优提示。

### 变分自编码器（VAEs）

自编码器是一种技术，你可以将特征数量减少到一个信息丰富的嵌入集，你可以将其视为**主成分分析**（PCA）的更复杂版本，以更好地理解它。它是通过首先尝试将原始空间编码到新的嵌入（称为编码），然后解码嵌入，并为每个数据点（称为解码）重新生成原始特征来实现的。在 VAE（Kingma and Welling, 2013）中，它不是生成一组特征（嵌入），而是为每个新特征生成一个分布。例如，不是将原始的 1,000 个特征减少到 100 个特征，每个特征有一个浮点值，而是得到 100 个新的变量，每个变量都是一个正态分布（或高斯分布）。这个过程的美妙之处在于，然后你可以从这些分布中选择不同的值来为每个变量生成一组新的 100 个嵌入。在解码它们的过程中，这些嵌入被解码，并生成一组具有原始大小（1,000）的新特征。这个过程可以用于不同类型的数据，如图像（Vahdat et al., 2020）和图（Simonovsky et al., 2018; Wengong et al., 2018）。你可以在[`github.com/AntixK/PyTorch-VAE`](https://github.com/AntixK/PyTorch-VAE)找到实现 PyTorch 的 VAE 的集合。

### 生成对抗网络（GANs）

在 2014 年引入的这项技术（Goodfellow et al., 2020）中，一个类似于监督分类模型的判别器和生成器协同工作。生成器，可能是一个用于生成所需数据类型（如图像）的神经网络架构，旨在生成图像以欺骗判别器，使其将生成的数据识别为真实数据。判别器学习如何保持区分生成数据和真实数据的能力。在某些情况下，生成的数据被称为假数据，例如在深度伪造（[`www.businessinsider.com/guides/tech/what-is-deepfake`](https://www.businessinsider.com/guides/tech/what-is-deepfake)）等技术模型中。然而，生成的数据可以作为新数据点在不同应用中使用的机会，例如药物发现（Prykhodko et al., 2019）。你可以使用`torchgan`来实现 GANs（[`torchgan.readthedocs.io/en/latest/`](https://torchgan.readthedocs.io/en/latest/)）。

由于基于生成模型之上涌现出一批基于提示的技术，我们将提供如何最优设计提示的更好理解。

## 基于文本生成模型的提示工程

提示工程不仅是在机器学习中的一个新兴话题，而且已经成为一个高薪的职位名称。在提示工程中，我们的目标是提供最优的提示以生成最佳可能的结果（例如，文本、代码和图像），并将生成模型的问题识别为改进它们的机会。对大型语言和生成模型的基本理解、你的语言熟练度和特定领域的数据生成领域的专业知识可以帮助你更好地进行提示。有一些免费资源可以帮助你学习提示工程，例如 Andrew Ng 和 OpenAI 提供的一门课程（[`www.deeplearning.ai/short-courses/ChatGPT-prompt-engineering-for-developers/`](https://www.deeplearning.ai/short-courses/ChatGPT-prompt-engineering-for-developers/)）以及微软发布的一些关于提示工程的入门内容（[`learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/prompt-engineering`](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/prompt-engineering)）。然而，我们不会让你自己从头开始学习这个话题。在这里，我们将提供一些关于最优提示的指导，这将帮助你提高你的提示技能。

### 目标提示

在我们的日常对话中，无论是在工作、大学还是家里，我们都有方法确保对方更好地理解我们的意思，从而得到更好的回应。例如，如果你对朋友说“给我那个”，而不是“给我桌子上那瓶水”，你的朋友可能不会给你那瓶水，或者对你具体指的是什么感到困惑。在提示中，如果你清楚地解释了针对一个非常具体的任务你想要什么，你可以得到更好的回应和生成数据，比如图像。以下是一些用于更好提示的技术：

+   **明确要求**：你可以提供具体信息，例如你希望生成的数据的格式，如项目符号或代码，以及你所指的任务，如撰写电子邮件与编写商业计划。

+   **指定数据生成对象**：你甚至可以指定为谁生成数据的专业技能或职位，例如为机器学习工程师、业务经理或软件开发人员生成一段文本。

+   **指定时间**：你可以指定你想要的信息，比如技术发布日期、某事首次宣布的时间、事件的年代顺序，以及像埃隆·马斯克这样的名人随时间变化的净资产变化等等。

+   **简化概念**：你可以提供一个简化的版本，确保模型不会被你提示的复杂性所困惑。

虽然这些技巧可以帮助你更好地进行提示，但如果要求文本响应或生成无关数据，仍然有可能得到高度自信的虚假答案。这通常被称为幻觉。减少无关或不正确响应或数据生成机会的一种方法是为模型提供测试。当我们用 Python 编写函数和类时，我们可以设计单元测试来确保它们的输出符合预期，正如在第*8 章*中讨论的，*使用测试驱动开发控制风险*。

## 使用 PyTorch 进行生成建模

你可以使用本章前面讨论的不同技术，基于 PyTorch 开发生成模型。我们在这里想练习使用 VAEs。VAE 的目标是为数据的低维表示找到一个概率分布。例如，模型学习关于输入参数表示的均值和方差（或对数方差），假设潜在空间（即潜在变量或表示的空间）为正态或高斯分布。

我们首先导入所需的库和模块，并从 PyTorch 加载`Flowers102`数据集：

```py
transform = transforms.Compose([    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
train_dataset = datasets.Flowers102(root='./data',
    download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32,
    shuffle=True)
```

然后，我们定义了一个用于 VAE 的类，如下所示，其中定义了两个线性层来编码图像的输入像素。然后，通过两个线性层定义了潜在空间概率分布的均值和方差，以便将潜在变量解码回原始输入数量以生成与输入数据相似的图像。在潜在空间中学习的分布的均值和方差将被用来生成新的潜在变量，并可能生成新的数据：

```py
class VAE(nn.Module):    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )self.fc_mean = nn.Linear(128, 32)
         self.fc_var = nn.Linear(128, 32)
         self.decoder = nn.Sequential(
             nn.Linear(32, 128),
             nn.ReLU(),
             nn.Linear(128, 512),
             nn.ReLU(),
             nn.Linear(512, 32 * 32 * 3),
             nn.Sigmoid(),
         )
    def forward(self, x):
        h = self.encoder(x.view(-1, 32 * 32 * 3))
        mean, logvar = self.fc_mean(h), self.fc_var(h)
        std = torch.exp(0.5*logvar)
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return self.decoder(z), mean, logvar
```

现在，我们初始化定义的`VAE`类，并将`Adam`优化器作为优化算法，学习率为`0.002`：

```py
model = VAE()optimizer = optim.Adam(model.parameters(), lr=2e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

然后，我们定义了一个使用`binary_cross_entropy`的损失函数，如下所示，以比较重新生成的像素与输入像素：

```py
def loss_function(recon_x, x, mu, logvar):    BCE = nn.functional.binary_cross_entropy(recon_x,
        x.view(-1, 32 * 32 * 3), reduction='sum')
    KLD = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

现在，我们准备使用之前加载的`Flowers102`数据集来训练模型：

```py
n_epoch = 400for epoch in range(n_epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mean, logvar = model(data)
        loss = loss_function(recon_batch, data, mean,
            logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch: {epoch} Average loss: {
        train_loss / len(train_loader.dataset):.4f}')
```

然后，我们可以使用这个训练好的模型来生成几乎像花朵一样的图像（见*图 14**.1*）。通过超参数优化，例如改变模型的架构，你可以获得更好的结果。你可以在*第十二章*中回顾深度学习中的超参数优化，*超越机器学习调试的深度学习*。

![图 14.1 – 我们之前开发的简单 VAE 生成的示例图像](img/B16369_14_01.jpg)

图 14.1 – 我们之前开发的简单 VAE 生成的示例图像

这只是一个使用 PyTorch 进行生成建模的简单示例。尽管生成建模取得了成功，但最近使用生成模型开发的工具（如 Chat-GPT）的部分成功归功于强化学习的智能使用，我们将在下一节中讨论。

# 强化学习

**强化学习**（**RL**）不是一个新想法或技术。其最初的想法可以追溯到 20 世纪 50 年代，当时由理查德·贝尔曼提出，并引入了贝尔曼方程（Sutton 和 Barto，2018）。然而，它最近与人类反馈的结合，我们将在下一节中解释，为它在开发机器学习技术中的效用提供了新的机会。强化学习的一般思想是通过经验学习，或与指定环境的交互，而不是像监督学习那样使用收集到的数据点集进行训练。在强化学习中，考虑了一个代理，它学习如何改进动作以获得更大的奖励（Kaelbling 等人，1996）。代理在接收到前一步采取的动作的奖励后，会迭代地改进其采取行动的方法，或更技术性地说是策略。

在 RL 的历史上，两个重要的发展和用途导致了其流行度的增加，包括 Q-learning（Watkins，1989）的发展以及使用 Q-learning 将 RL 和深度学习（Mnih 等人，2013）相结合。尽管 RL 背后的成功故事和它模仿人类经验学习的直觉，但已经证明深度强化学习不是数据高效的，需要大量的数据或迭代经验，这使得它与人类学习在本质上不同（Botvinick 等人，2019）。

最近，**带有人类反馈的强化学习**（**RLHF**）被用作强化学习成功应用于改进生成模型结果的应用，我们将在下文中讨论。

## 带有人类反馈的强化学习（RLHF）

使用带有人类反馈的强化学习，奖励是根据人类反馈计算的，无论是专家还是非专家，这取决于问题。然而，奖励并不是一个预定义的数学公式，考虑到问题的复杂性，如语言模型。人类提供的反馈会导致模型逐步改进。例如，RLHF 语言模型的训练过程可以总结如下 ([`huggingface.co/blog/rlhf`](https://huggingface.co/blog/rlhf))：

1.  训练语言模型，这被称为预训练。

1.  数据收集和训练奖励模型。

1.  使用奖励模型通过强化学习微调语言模型。

然而，学习如何使用 PyTorch 设计基于 RLHF 的模型可能有助于更好地理解这一概念。

### 基于 PyTorch 的 RLHF

从 RLHF 中受益的一个主要挑战是设计用于人类反馈收集和整理的基础设施，然后提供它们来计算奖励，然后改进主要预训练模型。在这里，我们不想深入探讨 RLHF 的这一方面，而是通过一个简单的代码示例来了解如何将此类反馈纳入机器学习模型。有一些很好的资源，如 [`github.com/lucidrains/PaLM-rlhf-pytorch`](https://github.com/lucidrains/PaLM-rlhf-pytorch)，可以帮助你更好地理解 RLHF 以及如何使用 Python 和 PyTorch 来实现它。

在这里，我们将使用 GPT-2 ([`huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_gpt2.html`](https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_gpt2.html)) 作为预训练模型。首先，我们导入必要的库和模块，并初始化模型、分词器和优化器，这里选择的是 `Adam`：

```py
import torchfrom transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import optim
from torch.utils.data import DataLoader
# Pretrain a GPT-2 language model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

现在，假设我们已经收集了人类反馈并正确格式化，我们可以使用它来创建一个来自 PyTorch 的 DataLoader：

```py
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
```

下一步是设计一个奖励模型，我们使用一个两层全连接神经网络：

```py
class Reward_Model(torch.nn.Module):    def __init__(self, input_size, hidden_size, output_size):
        super(RewardModel, self).__init__()
        self.fc_layer1 = torch.nn.Linear(input_size,
            hidden_size)
        self.fc_layer2 = torch.nn.Linear(hidden_size,
            output_size)
    def forward(self, x):
        x = torch.relu(self.fc_layer1(x))
        x = self.fc_layer2(x)
        return x
```

然后，我们使用先前定义的类初始化奖励模型：

```py
reward_model = Reward_Model(input_size, hidden_size, output_size)
```

我们现在可以使用收集到的人类反馈和奖励模型来改进我们的预训练模型。如果你注意以下代码，与没有奖励模型的神经网络相比，这个简单的循环遍历 epochs 和 batches 进行模型训练的主要区别在于奖励计算，然后将其用于损失计算：

```py
for epoch in range(n_epochs):    for batch in dataloader:
        input_ids = tokenizer.encode(batch['input'],
            return_tensors='pt')
        output_ids = tokenizer.encode(batch['output'],
            return_tensors='pt')
        reward = reward_model(batch['input'])
        loss = model(input_ids, labels=output_ids).loss * reward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

这是一个设计基于 RLHF 的模型改进的非常简单的例子，用于帮助你更好地理解这个概念。例如，[`github.com/lucidrains/PaLM-rlhf-pytorch`](https://github.com/lucidrains/PaLM-rlhf-pytorch)这样的资源将帮助你实现更复杂的方法，将此类人类反馈纳入模型改进。

接下来，让我们探讨机器学习中的另一个有趣话题，称为自监督学习。

# 自监督学习（SSL）

**自监督学习**（**SSL**）不是一个新概念。它与强化学习类似，但由于其在学习数据表示方面的有效性，在深度学习结合之后引起了人们的关注。此类模型的例子包括用于语言建模的 Word2vec（Mikolov 等人，2013 年）和 Meta 的 RoBERTa 模型，这些模型使用 SSL 训练，在多个语言建模任务上取得了最先进的性能。SSL 的想法是为机器学习模型定义一个目标，该目标不依赖于预先标记或数据点的量化——例如，使用前一时间步预测视频中的对象或人的位置，遮盖图像或序列数据的一部分，并试图填充这些被遮盖的部分。此类模型的一个广泛应用的例子是在强化学习中学习图像和文本的表示，然后在其他上下文中使用这些表示，例如，在带有数据标签的小数据集上进行监督建模（Kolesnikov 等人，2019 年，王等人，2020 年）。

SSL 的范畴下有多个技术，以下列举三个：

+   **对比学习**：对比学习的想法是学习表示，使得相似的数据点比不相似的数据点更接近（Jaiswal 等人，2020 年）。

+   **自回归模型**：在自回归建模中，模型旨在根据之前的数据点预测下一个数据点，无论是基于时间还是特定的序列顺序。这在语言建模中是一个非常流行的技术，例如 GPT 模型预测句子中的下一个单词（Radford 等人，2019 年）。

+   **通过修复缺失部分进行自监督**：在这种方法中，我们遮盖数据的一部分，并训练模型来填补缺失的部分。例如，图像的一部分可能被遮盖，模型被训练来预测被遮盖的部分。遮盖自动编码器是这种技术的一个例子，其中自动编码器的解码过程中填充了图像被遮盖的部分（张等人，2022 年）。

接下来，我们将通过一个简单的 Python 和 PyTorch 自监督建模示例进行练习。

## 使用 PyTorch 的自监督学习

从编程的角度来看，与监督学习相比，SSL 深度学习的主要区别在于定义训练和测试的目标和数据。在这里，我们想使用我们用来练习 RLHF 的`Flowers102`数据集进行练习。

我们首先使用两个编码和解码`torch.nn.Conv2d()`层定义神经网络类，如下所示：

```py
class Conv_AE(nn.Module):    def __init__(self):
        super(Conv_AE, self).__init__()
        # Encoding data
        self.encoding_conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.encoding_conv2 = nn.Conv2d(8, 32, 3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Decoding data
        self.decoding_conv1 = nn.ConvTranspose2d(32, 8, 2,
            stride=2)
        self.decoding_conv2 = nn.ConvTranspose2d(8, 3, 2,
            stride=2)
    def forward(self, x):
        # Encoding data
        x = torch.relu(self.encoding_conv1(x))
        x = self.pool(x)
        x = torch.relu(self.encoding_conv2(x))
        x = self.pool(x)
        # Decoding data
        x = torch.relu(self.decoding_conv1(x))
        x = self.decoding_conv2(x)
        x = torch.sigmoid(x)
        return x
```

然后，我们初始化模型，指定`torch.nn.MSELoss()`作为预测图像和真实图像比较的标准，以及`torch.optim.Adam()`作为优化器，学习率为`0.001`：

```py
model = Conv_AE().to(device)criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

以下函数帮助我们实现对每个图像随机 8x8 部分的掩码，然后自动编码器学习填充这些部分：

```py
def create_mask(size=(32, 32), mask_size=8):    mask = np.ones((3, size[0], size[1]), dtype=np.float32)
    height, width = size
    m_height, m_width = mask_size, mask_size
    top = np.random.randint(0, height - m_height)
    left = np.random.randint(0, width - m_width)
    mask[:, top:top+m_height, left:left+m_width] = 0
    return torch.from_numpy(mask)
```

然后，我们按照以下方式训练模型 200 个 epoch。正如你在*图 14.2*中可以看到的，图像首先被掩码，然后在解码步骤中，自动编码器试图重建完整的图像，包括掩码部分：

```py
n_epoch = 200for epoch in range(n_epoch):
    for data in train_loader:
        img, _ = data
        # Creating mask for small part in training images
        mask = create_mask().to(device)
        img_masked = img * mask
        img = img.to(device)
        img_masked = img_masked.to(device)
        optimizer.zero_grad()
        outputs = model(img_masked)
        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()
```

正如你在*图 14.2*中看到的填充图像的示例中所示，模型能够正确地找到模式。然而，通过适当的超参数优化和设计具有更好神经网络架构的模型，你可以实现更高的性能和更好的模型。

![图 14.2 – 使用卷积自动编码器模型的示例图像（第一行）、其掩码版本（第二行）和再生版本（第三行）](img/B16369_14_02.jpg)

图 14.2 – 使用卷积自动编码器模型的示例图像（第一行）、其掩码版本（第二行）和再生版本（第三行）

你可以通过提供的资源和参考资料了解更多关于 SSL 和本章中提供的技术，以更好地理解这些概念。

# 摘要

在本章中，你获得了对机器学习建模中监督学习之外的最新进展的高级理解，包括生成建模、强化学习和自监督学习。你还了解了最佳提示和提示工程，以便从基于生成模型并接受用户文本提示的工具和应用中受益。你提供了相关的代码仓库和 Python 和 PyTorch 中可用的功能，这将帮助你开始学习这些高级技术。这些知识不仅帮助你更好地理解它们是如何工作的，如果你遇到它们，而且开始使用这些高级技术构建自己的模型。

在下一章中，你将了解在机器学习建模和实践中识别因果关系的好处，以及使用 Python 库实现因果建模的示例。

# 问题

1.  生成深度学习技术的例子有哪些？

1.  使用 transformers 的生成文本模型的例子有哪些？

1.  GAN 中的生成器和判别器是什么？

1.  你可以使用哪些技术来提高提示的效果？

1.  你能解释一下强化学习如何有助于导入生成模型的结果吗？

1.  简要解释对比学习。

# 参考文献

+   Cheng, Yu, 等人. “*药物发现中的分子设计：深度生成模型的全面综述*。” *生物信息学简报* 22.6 (2021): bbab344.

+   Davis, Richard Lee, 等人. “*塑造未来：解锁深度生成模型在设计空间探索中的创造潜能*。” *2023 年 CHI 会议关于人机交互系统人类因素扩展摘要* (2023).

+   Zhao, Yaoyao Fiona, 等人，编。 “*高级制造设计*。” *机械设计杂志* 145.1 (2023): 010301.

+   Touvron, Hugo, 等人. “*Llama：开放且高效的基金会语言模型*。” arXiv 预印本 arXiv:2302.13971 (2023).

+   Vaswani, Ashish, 等人. “*注意力即所需*。” *神经信息处理系统进展* 30 (2017).

+   Kingma, Diederik P., 和 Max Welling. “*自动编码变分贝叶斯*。” arXiv 预印本 arXiv:1312.6114 (2013).

+   Vahdat, Arash, 和 Jan Kautz. “*NVAE：一种深度分层变分自动编码器*。” *神经信息处理系统进展* 33 (2020): 19667-19679.

+   Simonovsky, Martin, 和 Nikos Komodakis. “*Graphvae：使用变分自动编码器生成小图*。” *人工神经网络与机器学习–ICANN 2018：第 27 届国际人工神经网络会议，希腊罗得岛，2018 年 10 月 4-7 日，会议论文集，第一部分* 27\. Springer 国际出版社 (2018).

+   Jin, Wengong, Regina Barzilay, 和 Tommi Jaakkola. “*用于分子图生成的连接树变分自动编码器*。” *机器学习国际会议* PMLR (2018).

+   Goodfellow, Ian, 等人. “*生成对抗网络*。” *ACM 通讯* 63.11 (2020): 139-144.

+   Karras, Tero, Samuli Laine, 和 Timo Aila. “*基于风格的生成对抗网络生成器架构*。” *IEEE/CVF 计算机视觉与模式识别会议论文集* (2019).

+   Prykhodko, Oleksii, 等人. “*基于潜在向量生成对抗网络的从头分子生成方法*。” *化学信息学杂志* 11.1 (2019): 1-13.

+   Sutton, Richard S., 和 Andrew G. Barto. *强化学习：入门*。 MIT 出版社 (2018).

+   Kaelbling, Leslie Pack, Michael L. Littman, 和 Andrew W. Moore. “*强化学习：综述*。” *人工智能研究杂志* 4 (1996): 237-285.

+   Watkins, Christopher John Cornish Hellaby. *从延迟奖励中学习*。 (1989).

+   Mnih, Volodymyr, 等人. “*使用深度强化学习玩 Atari*。” arXiv 预印本 arXiv:1312.5602 (2013).

+   Botvinick, Matthew, 等人. “*强化学习，快与慢*。” *认知科学趋势* 23.5 (2019): 408-422.

+   Kolesnikov, Alexander, Xiaohua Zhai 和 Lucas Beyer。“*重新审视自监督视觉表示学习*。” *IEEE/CVF 计算机视觉与模式识别会议论文集* (2019).

+   Wang, Jiangliu, Jianbo Jiao 和 Yun-Hui Liu。“*通过速度预测进行自监督视频表示学习*。” *计算机视觉–ECCV 2020: 第 16 届欧洲计算机视觉会议*，英国格拉斯哥，2020 年 8 月 23 日至 28 日，第 17 卷 16。Springer 国际出版社 (2020)。

+   Jaiswal, Ashish, 等人。“*关于对比自监督学习的综述*。” *Technologies* 9.1 (2020): 2.

+   Radford, Alec, 等人。“*语言模型是无监督的多任务学习者*。” OpenAI 博客 1.8 (2019): 9.

+   Zhang, Chaoning, 等人。“*关于视觉和更多领域的掩码自动编码器在自监督学习中的应用综述*。” arXiv 预印本 arXiv:2208.00173 (2022).

# 第五部分：模型调试的高级主题

在本书的结论部分，我们将探讨机器学习中最关键的一些主题。我们将首先解释相关性和因果性的区别，阐明它们在模型开发中的不同影响。过渡到安全和隐私的主题，我们将讨论确保我们的模型既强大又尊重用户数据的紧迫问题、挑战和技术。我们将以对人类在循环机器学习的解释来结束本书，强调人类专业知识与自动化系统之间的协同作用，以及这种合作如何为更有效的解决方案铺平道路。

本部分包含以下章节：

+   *第十五章*, *相关性 versus 因果性*

+   *第十六章*，*机器学习中的安全和隐私*

+   *第十七章*，*人类在循环机器学习*
