

# 第七章：归属作者和规避方法

互联网通过为个人提供一个公共平台来表达他们的观点、思想、发现和担忧，从而推动了基本权利——言论自由。任何人都可以通过文章、博客文章或视频来表达自己的观点，并在某些情况下免费发布（例如在 Blogspot、Facebook 或 YouTube 上）。然而，这也导致恶意行为者能够自由地生成虚假信息、诽谤、诬告和滥用内容。作者归属是一项基于内容识别文本作者的任务。归属作者可以帮助执法机构追踪仇恨言论和威胁的肇事者，或者帮助社交媒体公司检测协调攻击和 Sybil 账户。

另一方面，个人可能希望作为作者保持匿名。他们可能想要保护自己的身份，以避免审查或公众关注。这就是作者混淆发挥作用的地方。作者混淆是修改文本的任务，使得作者不能通过归属技术被识别。

在本章中，我们将涵盖以下主要主题：

+   作者归属和混淆

+   作者归属技术

+   作者混淆技术

到本章结束时，你将了解作者归属、其背后的社会技术方面以及规避它的方法。

# 技术要求

你可以在 GitHub 上找到本章的代码文件，网址为[`github.com/PacktPublishing/10-Machine-Learning-Blueprints-You-Should-Know-for-Cybersecurity/tree/main/Chapter%207`](https://github.com/PacktPublishing/10-Machine-Learning-Blueprints-You-Should-Know-for-Cybersecurity/tree/main/Chapter%207)。

# 作者归属和混淆

在本节中，我们将讨论作者归属的确切含义以及设计归属系统的动机。虽然有一些很好的理由去做这件事，但也有一些邪恶的理由；因此，我们还将讨论混淆的重要性，以保护免受邪恶攻击者的攻击。

## 什么是作者归属？

**作者归属**是识别给定文本作者的任务。归属背后的基本思想是不同的作者有不同的写作风格，这将反映在词汇、语法、结构和整体组织上。归属可以基于启发式方法（如相似度、共同词分析或人工专家分析）。近年来，**机器学习**（**ML**）的进步也使得构建能够学习检测给定文本作者的分类器成为可能。

作者归属不是一个新问题——这一领域的研究可以追溯到 1964 年。一系列被称为《联邦党人文集》的论文已经发表，其中包含 140 多篇政治论文。虽然这项工作是由 3 个人共同撰写的，但其中 12 篇论文被 2 位作者声称。Mosteller 和 Wallace 的研究涉及贝叶斯模型和统计分析，使用*n*-gram 产生了作者之间的统计显著差异，这是已知的第一项作者归属的实际工作。

作者归属之所以重要，原因如下：

+   **历史意义**：科学家和研究人员依赖历史文件和文本作为某些事件的证据。有时，这些文件可能具有巨大的政治和文化意义，了解作者将有助于将它们置于适当的背景中并确定其可信度。例如，如果发现一个描述某些历史时期并正面描绘独裁者或已知恶意行为者的账户，确定作者身份就很重要，因为这可能会改变文本的可信度。作者归属有助于确定文本是否可以被视为权威来源。

+   **知识产权**：与《联邦党人文集》一样，关于某些创意或学术作品的归属常常存在争议。这发生在多人声称对同一本书、文章或研究论文拥有所有权时。在其他时候，某个人可能被指控剽窃他人的作品。在这种情况下，追踪特定文本的作者至关重要。作者归属可以帮助识别作者，匹配风格和语气上的相似性，并解决知识产权争议。

+   **犯罪调查**：罪犯经常使用文本作为与受害者或执法部门沟通的手段。这可能表现为勒索信或威胁。如果文本量很大，可能反映出作者的一些风格习惯。执法官员使用作者归属方法来确定收到的信息是否符合任何已知罪犯的风格。

+   **滥用检测**：在互联网和社交媒体上，Sybil 账户正成为一个日益严峻的挑战。这些账户是由同一实体控制，但伪装成不同的人。Sybil 账户有恶意目的，例如多个 Facebook 账户生成虚假互动，或多个 Amazon 账户撰写虚假产品评论。由于它们由同一实体控制，产生的内容（帖子、推文、评论）通常相似。作者归属可以用来识别由同一作者撰写的帖子内容的账户组。

随着互联网和社交媒体平台的普及，网络犯罪呈上升趋势，恶意行为者正在捕食不知情的受害者。因此，作者归属分析也是一个网络安全问题。下一节将描述作者身份混淆，这是一个与作者归属分析相反的任务。

## 什么是作者身份混淆？

在上一节中，我们讨论了作者归属分析，即识别给定文本作者的任务。作者身份混淆是一个与作者归属分析完全相反的任务。

给定一段文本，作者身份混淆的目标是通过操纵和修改文本，使其最终结果如下：

+   文本中的意义和要点保持不变

+   风格、结构和词汇被适当地修改，使得文本无法归因于原始作者（即，避免作者归属分析技术）

个人可能会使用混淆技术来隐藏他们的身份。考虑以下句子*“我们在这个国家的最高政府层级观察到了巨大的腐败。”*如果这句话被改写为*“分析显示，这个国家政府最高层发生了巨大的腐败事件，”*意义保持不变。然而，风格明显不同，与原始作者的风格相差甚远。这是一种有效的混淆。分析文本的分析师不太可能轻易地将它映射到同一作者。

注意，混淆的两个目标（即，保留原始意义和去除风格标记）同样重要，它们之间存在权衡。我们可以通过对文本进行极端修改来获得高程度的混淆，但到那时，文本可能已经失去了其原始意义和意图。另一方面，我们可以通过极其微小的调整来保留意义——但这可能不会导致有效的混淆。

作者身份混淆既有积极的应用，也有消极的应用。恶意行为者可以使用混淆技术来对抗之前讨论的归属目的，并避免被检测。例如，一个想要保持不被发现，同时发送勒索笔记和电子邮件的罪犯，可以通过选择不同的词汇、语法结构和组织来混淆他们的文本。然而，正如以下详细说明的那样，混淆在公民权和人权方面有几个重要的应用案例：

+   **压迫性政府**：如前所述，互联网极大地促进了人类自由表达的权利。然而，一些政府可能会试图通过针对那些对他们提出批评的个体来限制这些权利。例如，一个专制政府可能希望禁止报道反对其议程的内容或揭露腐败和恶意的计划。在这样的时刻，记者和个人可能希望保持匿名——他们的身份被识别可能会导致他们被捕。混淆技术将改变他们所写的文本，以便他们想要传达的内容得以保留，但写作风格将与其通常的风格有显著不同。

+   **敏感问题**：即使政府本质上不是压迫性的，某些问题也可能敏感且具有争议。这类问题的例子包括宗教、种族歧视、性暴力报告、同性恋和生殖健康保健。撰写关于这类问题的个人可能会冒犯公众或某些其他群体或教派。作者归属混淆允许这样的个人发布此类内容，同时保持匿名（或者至少使辨别文本作者的难度增加）。

+   **隐私和匿名性**：许多人认为隐私是一项基本的人权。因此，即使问题不敏感或政府不腐败，用户也有权在想要的时候保护他们的身份。每个人都应该自由地发布他们想要的内容并隐藏他们的身份。作者归属混淆允许用户在表达自己的同时保持隐私。

现在你已经很好地理解了作者归属和混淆及其实际需求，让我们用 Python 来实现它。

# 作者归属技术

上一节描述了作者归属和混淆的重要性。本节将重点关注归属方面——我们如何设计和构建模型来确定给定文本的作者。

## 数据集

在作者归属和混淆领域已有先前的研究。在此任务上进行基准测试的标准数据集是*Brennan-Greenstadt 语料库*。这个数据集是通过在美国一所大学进行的调查收集的。招募了 12 位作者，每位作者都需要提交一篇至少包含 5,000 个单词的预先撰写的文本。

同一作者后来发布了一个修改和改进的版本的数据——称为**扩展 Brennan-Greenstadt 语料库**。为了生成这个数据集，作者通过从 Amazon Mechanical Turk（**MTurk**）招募参与者进行了一次大规模调查。MTurk 是一个允许研究人员和科学家进行人类受试者研究的平台。用户注册 MTurk 并填写详细的问卷，这使得研究人员更容易调查他们想要的细分市场或人口统计特征（按性别、年龄、国籍）。参与者每完成一个**人类交互任务**（**HIT**）就能获得报酬。

为了创建扩展语料库，使用了 MTurk，以确保提交的内容多样化且不限于大学生。每篇写作都是科学或学术性的（例如论文、研究报告或观点文章）。提交的内容仅包含文本，不包含其他信息（如参考文献、引用、URL、图片、脚注、尾注和章节分隔）。引用应尽量减少，因为大部分文本应该是作者原创的。每个样本至少有 500 个单词。

**Brennan-Greenstadt 语料库**和**扩展 Brennan-Greenstadt 语料库**均可免费向公众在线提供。为了简化，我们将使用**Brennan-Greenstadt 语料库**（其中包含大学生的写作样本）进行实验。然而，鼓励读者在扩展语料库上重现结果，并根据需要调整模型。过程和代码将保持不变——你只需更改底层数据集。

为了方便，我们提供了我们使用的数据集（[`github.com/PacktPublishing/10-Machine-Learning-Blueprints-You-Should-Know-for-Cybersecurity/blob/main/Chapter%207/Chapter_7.ipynb`](https://github.com/PacktPublishing/10-Machine-Learning-Blueprints-You-Should-Know-for-Cybersecurity/blob/main/Chapter%207/Chapter_7.ipynb)）。该数据集包含一个根文件夹，每个作者都有一个子文件夹。每个子文件夹包含作者的写作样本。您需要解压数据并将其放置在您想要的文件夹中（并相应地更改以下代码中的`data_root_dir`）。

记住，对于我们的实验，我们需要读取数据集，使得输入（特征）在一个数组中，标签在另一个数组中。以下代码片段解析文件夹结构，并生成这种格式的数据：

```py
def read_dataset(num_authors = 99):
  X = []
  y = []
  data_root_dir = "../data/corpora/amt/"
  authors_to_ignore = []
  authorCount = 0
  for author_name in os.listdir(data_root_dir):
      # Check if the maximum number of authors has been parsed
      if authorCount > self.numAuthors:
         break
      if author_name not in authors_to_ignore:
         label = author_name
         documents_path = data_root_dir + author_name + "/"
         authorCount += 1
         for doc in os.listdir(documents_path):
            if validate_file(doc):
              text = open(docPath + doc, errors = "ignore").read()
              X.append(text)
              y.append(label)
  return X, y
```

数据集还包含一些维护文件以及一些指示训练、测试和验证数据的文件。我们需要一个函数来过滤掉这些文件，以便这些信息不会被读取到数据中。以下是我们将使用的方法：

```py
def validate_file(file_name):
    filterWords = ["imitation", "demographics", "obfuscation", "verification"]
    for fw in filterWords:
        if fw in file_name:
            return False
    return True
```

我们已经读取了数据集，现在可以从中提取特征。对于作者归属，大多数特征是风格测量和手工制作的。在下文中，我们将探讨一些在先前工作中取得成功的特征。

## 特征提取

现在，我们将实现一系列函数，每个函数都从我们的数据中提取特定的特征。每个函数都将输入文本作为参数，处理它，并将特征作为输出返回。

我们像往常一样开始，导入所需的库：

```py
import os
import nltk
import re
import spacy
from sortedcontainers import SortedDict
from keras.preprocessing import text
import numpy as np
```

作为第一个特征，我们将使用输入中的字符数：

```py
def CountChars(input):
    num_chars = len(input)
    return num_chars
```

接下来，我们将设计一个特征，该特征衡量平均单词长度（每个单词的字符数）。为此，我们首先将文本分割成单词数组，并通过删除任何特殊字符（如大括号、符号和标点符号）来清理它。然后，我们分别计算字符数和单词数。它们的比率是我们希望的特征：

```py
def averageCharacterPerWord(input):
    text_array = text.text_to_word_sequence(input,
                                            filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"',
                                            lower=False, split=" ")
    num_words = len(text_array)
    text_without_spaces = input.replace(" ", "")
    num_chars = len(text_without_spaces)
    avgCharPerWord = 1.0 * num_chars / num_words
    return avgCharPerWord
```

现在，我们计算字母的频率。我们首先创建一个 26 个元素的数组，其中每个元素计算该字母在文本中出现的次数。第一个元素对应于 A，下一个对应于 B，依此类推。请注意，由于我们在计算字母，我们需要将文本转换为小写。然而，如果这是我们的特征，它将严重依赖于文本的长度。因此，我们通过总字符数进行归一化。因此，数组的每个元素都描述了该特定字母在文本中的百分比：

```py
def frequencyOfLetters(input):
    input = input.lower()  # because its case sensitive
    input = input.lower().replace(" ", "")
    num_chars = len(input)
    characters = "abcdefghijklmnopqrstuvwxyz".split()
    frequencies = []
    for each_char in characters:
      char_count = input.count(each_char)
      if char_count < 0:
        frequencies.append(0)
      else:
        frequencies.append(char_count/num_chars)
    return frequencies
```

接下来，我们将计算常见二元组的频率。语言学和语音学的先前研究已经表明哪些二元组在英语写作中是常见的。我们将首先编制这样一个二元组的列表。然后，我们将遍历列表并计算每个二元组的频率，并计算一个向量。最后，我们归一化这个向量，结果代表我们的特征：

```py
def CommonLetterBigramFrequency(input):
    common_bigrams = ['th','he','in','er','an','re','nd',
                      'at','on','nt','ha','es','st','en',
                      'ed','to','it','ou','ea','hi','is',
                      'or','ti','as','te','et','ng','of',
                      'al','de','se','le','sa','si','ar',
                      've','ra','ld','ur']
    bigramCounter = []
    input = input.lower().replace(" ", "")
    for bigram in common_bigrams:
      bigram_count = input.count(bigram)
      if bigram_count == -1:
        bigramCounter.append(0)
      else:
        bigramCounter.append(bigram_count)
    total_bigram_count = np.sum(bigramCounter)
    bigramCounterNormalized = []
    for bigram_count in bigramCounter:
      bigramCounterNormalized.append(bigram_count / total_bigram_count)
    return bigramCounterNormalized
```

就像常见的二元组一样，我们也计算常见三元组的频率（由三个字母组成的序列）。最终的特征表示一个归一化向量，类似于我们之前对二元组所做的那样：

```py
def CommonLetterTrigramFrequency(input):
    common_trigrams = ["the", "and", "ing", "her", "hat",
                       "his", "tha", "ere", "for", "ent",
                       "ion", "ter", "was", "you", "ith",
                       "ver", "all", "wit", "thi", "tio"]
    trigramCounter = []
    input = input.lower().replace(" ", "")
    for trigram in common_trigrams:
      trigram_count = input.count(trigram)
      if trigram_count == -1:
        trigramCounter.append(0)
      else:
        trigramCounter.append(trigram_count)
    total_trigram_count = np.sum(trigramCounter)
    trigramCounterNormalized = []
    for trigram_count in trigramCounter:
      trigramCounterNormalized.append(trigram_count / total_trigram_count)
    return trigramCounterNormalized
```

下一个特征是字符中数字的比例。首先，我们计算文本中的总字符数。然后，我们逐个字符解析文本并检查每个字符是否为数字。我们计算所有此类出现的次数，并将其除以之前计算的总数——这给我们提供了我们的特征：

```py
def digitsPercentage(input):
    num_chars = len(input)
    num_digits = 0
    for each_char in input:
      if each_char.isnumeric():
        num_digits = num_digits + 1
    digit_percent = num_digits / num_chars
    return digit_percent
```

同样，下一个特征是字符中字母的比例。我们首先需要将文本转换为小写。就像先前的特征一样，我们逐个字符解析，现在检查我们遇到的每个字符是否在 `[a-z]` 范围内：

```py
def charactersPercentage(input):
    input = input.lower().replace(" ", "")
    characters = "abcdefghijklmnopqrstuvwxyz"
    total_chars = len(input)
    char_count = 0
    for each_char in input:
      if each_char in characters:
        char_count = char_count + 1
    char_percent = char_count / total_chars
    return char_percent
```

之前，我们计算了字母的频率。按照类似的思路，我们计算从 `0` 到 `9` 的每个数字的频率并将其归一化。归一化向量被用作我们的特征：

```py
def frequencyOfDigits(input):
    input = input.lower().replace(" ", "")
    num_chars = len(input)
    digits = "0123456789".split()
    frequencies = []
    for each_digit in digits:
      digit_count = input.count(each_digit)
      if digit_count < 0:
        frequencies.append(0)
      else:
        frequencies.append(digit_count/num_chars)
    return frequencies
```

现在，我们将计算字符中是大写字母的比例。我们遵循与计数字符相同的程序，但现在我们计数的是大写字母。结果是归一化的，归一化的值形成我们的特征：

```py
def upperCaseCharactersPercentage(input):
    input = input.replace(" ", "")
    upper_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num_chars = len(input)
    upper_count = 0
    for each_char in upper_characters:
      char_count = input.count(each_char)
      if char_count > 0:
        upper_count = upper_count + char_count
    upper_percent = upper_count / num_chars
    return upper_percent
```

现在，我们将计算文本中特殊字符的频率。我们首先在文件中编译一个感兴趣的特殊字符列表。我们解析文件并计算每个字符的频率并形成一个向量。最后，我们通过总字符数来归一化这个向量。请注意，以下函数使用一个静态文件，其中存储了字符列表——你需要更改这一行代码以反映你的系统上文件存储的路径：

```py
def frequencyOfSpecialCharacters(input):
    SPECIAL_CHARS_FILE = "static_files/writeprints_special_chars.txt"
    num_chars = len(input)
    special_counts = []
    special_characters = open(SPECIAL_CHARS_FILE , "r").readlines()
    for each_char in special_characters:
      special = each_char.strip().rstrip()
      special_count = input.count(special)
      if special_count < 0:
        special_counts.append(0)
      else:
        special_counts.append(special_count / num_chars)
    return special_counts
```

接下来，我们将计算文本中短单词的数量。我们将短单词定义为少于或最多三个字符的单词。这是一个相当启发式定义；没有全球公认的短单词标准。你可以尝试不同的值来查看它是否会影响结果：

```py
def CountShortWords(input):
    words = text.text_to_word_sequence(input, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    short_word_count = 0
    for word in words:
        if len(word) <= 3:
            short_word_count = short_word_count + 1
    return short_word_count
```

作为一个非常简单的特征，我们计算输入中的总单词数。这涉及到将文本分割成一个单词数组（清理特殊字符）并计算数组的长度：

```py
def CountWords(input):
    words = text.text_to_word_sequence(input, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    return len(words)
```

现在，我们计算平均单词长度。我们简单地计算文本中每个单词的长度，并使用所有这些长度值的平均值作为特征：

```py
def averageWordLength(input):
    words = text.text_to_word_sequence(inputText, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
    lengths = []
    for word in words:
        lengths.append(len(word))
    return np.mean(lengths)
```

现在，我们已经有了所有计算特征的函数。每个函数都将文本作为参数并对其进行处理，以产生我们设计的特征。现在，我们将编写一个包装函数来将这些功能组合在一起。这个函数在被传递文本时，将运行所有我们的特征提取函数并计算每个特征。每个特征都将附加到一个向量中。这形成了我们的最终特征向量：

```py
def calculate_features(input):
  features = []
  features.extend([CountWords(input)])
  features.extend([averageWordLength(input)])
  features.extend([CountShortWords(input)])
  features.extend([CountChars(input)])
  features.extend([averageCharacterPerWord(input)])
  features.extend([frequencyOfLetters(input)])
  features.extend([CommonLetterBigramFrequency(input)])
  features.extend([CommonLetterTrigramFrequency(input)])
  features.extend([digitsPercentage(input)])
  features.extend([charactersPercentage(input)])
  features.extend([frequencyOfDigits(input)])
  features.extend([upperCaseCharactersPercentage(input)])
  features.extend([frequencyOfSpecialCharacters(input)])
  features.extend([frequencyOfPunctuationCharacters(input)])
  features.extend([posTagFrequency(input)])
```

现在，所有要做的事情就是将这个函数应用到我们的数据集上：

```py
X_original, Y = read_dataset(num_authors = 6)
X_Features = []
for x in X_original:
  x_features = calculate_features(x)
  X.append(x_features)
```

执行此操作后，`X`将包含我们设计的特征数组，而`Y`将包含相应的标签。困难的部分已经完成！接下来，我们将转向建模阶段。

## 训练属性器

在上一节中，我们处理了我们的数据集，手工制作了几个特征，现在每个文本都有一个特征向量以及相应的真实标签。在这个时候，这本质上是一个**监督学习**（**SL**）问题；我们有特征和标签，并希望学习它们之间的关联。我们将像处理迄今为止看到的所有其他监督问题一样来处理这个问题。

为了回顾，以下是我们将采取的步骤：

1.  将数据分割成训练集和测试集。

1.  在训练集上训练一个监督分类器。

1.  评估训练模型在测试集上的性能。

首先，我们按照以下方式分割数据。请注意，我们有多种作者，因此我们有多个标签。我们必须确保训练集和测试集中标签的分布大致相似；否则，我们的模型将偏向于特定作者。如果某个作者没有出现在训练集中，模型将完全无法检测到他们。

然后，我们在训练集上训练我们的分类模型（逻辑回归、决策树、随机森林、**深度神经网络**（**DNN**））。我们使用这个模型对测试集中的数据进行预测，并将预测与真实值进行比较。由于这个程序在前面的章节中已经介绍过，我们这里不再进行详细解释。

下一个示例代码片段展示了如何使用随机森林执行前面的步骤。读者应该用其他模型重复它：

```py
# Import Packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
# Training and Test Datasets
X_train, X_test, Y_train, Y_test = train_test_split(X_Features, Y)
# Train the model
model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train, Y_train)
# Plot the confusion matrix
Y_predicted = model.predict(X_test)
confusion = confusion_matrix(Y_test, Y_predicted)
plt.figure(figsize = (10,8))
sns.heatmap(confusion, annot = True,
            fmt = 'd', cmap="YlGnBu")
```

当你运行这个程序时，你会注意到混淆矩阵现在看起来不同了。之前我们有一个 2x2 的矩阵，现在我们得到一个 6x6 的矩阵。这是因为我们的数据集现在包含六个不同的标签（每个作者一个）。因此，对于给定类别的每个数据点，有六个可能的类别可以预测。

计算准确度仍然是相同的；我们需要找到预测正确的示例的比例。以下是一个执行此操作的函数：

```py
def calculate_accuracy(actual, predicted):
  total_examples = len(actual)
  correct_examples = 0
  for idx in range(total_examples):
    if actual[i] == predicted[i]:
      correct_examples = correct_examples + 1
  accuracy = correct_examples / total_examples
  return accuracy
```

在多类问题中，精确度和召回率的定义不再像计算假阳性和假阴性那样简单。相反，这些指标是按类别计算的。例如，如果有六个标签（1-6），那么对于类别 2，我们可以说以下内容：

+   真正的阳性是指实际和预测的类别都是 2 的情况

+   假阳性是指预测的类别是 2，但实际类别不是 2 的情况

+   真负例是指实际和预测的类别都不是 2 的情况

+   假阴性是指预测的类别不是 2，但实际类别是 2 的情况

使用这些定义和计算指标的标准表达式，我们可以计算每类的指标。每类的精确度和召回率可以平均计算整体精确度、召回率和 F1 分数。

幸运的是，我们不需要手动实现这个按类别的指标计算。`scikit-learn`有一个内置的分类报告，可以为你计算并生成这些指标。这可以按照以下方式使用：

```py
from sklearn.metrics import classification_report
classification_report(Y_test, Y_predicted)
```

这完成了我们关于作者归属的实现和分析。接下来，我们将建议一些读者可以追求的实验，以进一步探索这个主题。

## 改进作者归属

我们已经介绍了用于作者归属的 vanilla 模型和技术。然而，这里有很大的改进空间。作为数据科学家，我们必须愿意探索新的想法和技术，并持续改进我们的模型。以下是一些建议，读者可以尝试看看是否能获得更好的性能。

### 额外特征

我们使用了被称为 Writeprints 特征集的特征集。这在先前的研究中已经显示出成功。然而，这并不是特征的全列表。读者可以探索更多手工制作和自动化的特征来评估性能是否有所提高。以下是一些特征的示例：

+   文本情感

+   文本极性

+   函数词的数量和分数

+   **词频-逆文档频率**（**TF-IDF**）特征

+   由 Word2vec 生成的词嵌入

+   由**双向编码器表示从** **Transformers** （**BERT**）生成的上下文词嵌入

### 数据配置

我们进行的实验是在数据集的六个作者子集上进行的。在现实世界中，这个问题要开放得多，可能有更多的作者。值得探索随着作者数量的变化，模型性能如何变化。特别是，读者应该探索以下内容：

+   如果我们只选择 3 位作者，性能指标会是什么？如果我们选择 12 位呢？

+   如果我们将这个问题建模为二元分类，性能会如何变化？我们不是预测作者，而是预测特定文本是否由特定作者撰写。这将涉及为每位作者训练一个单独的分类器。这比多类方法显示出更好的预测能力和实际应用吗？

### 模型改进

为了简洁和避免重复，我们只展示了随机森林的例子。然而，读者应该尝试更多模型，包括但不限于以下模型：

+   **支持向量** **机**（**SVMs**）

+   朴素贝叶斯分类器

+   逻辑回归

+   决策树

+   深度神经网络（DNN）

当特征数量增加时，**神经网络**（**NN**）算法将特别有用。当嵌入和 TF-IDF 分数被添加时，特征将不再容易解释——神经网络擅长在它们能够发现高维特征的情况下。

这完成了我们对作者身份归因的讨论。在下一节中，我们将讨论一个与归因任务相反的问题。

# 作者身份混淆技术

到目前为止，我们已经看到了如何将作者身份归因于作者，以及如何构建检测作者的模型。在本节中，我们将转向作者身份混淆问题。正如本章初始部分所讨论的，作者身份混淆是一种故意操纵文本的艺术，以去除可能泄露作者身份的任何风格特征。

代码灵感来源于一个免费在线可用的实现（[`github.com/asad1996172/Obfuscation-Systems`](https://github.com/asad1996172/Obfuscation-Systems)），并进行了一些小的调整。

首先，我们将导入所需的库。在这里最重要的库是斯坦福大学开发的**自然语言工具包**（**NLTK**）库（[`www.nltk.org/`](https://www.nltk.org/））。这个库包含了几种标准现成的**自然语言处理**（**NLP**）任务，如分词、**词性标注**（**POS**）、**命名实体识别**（**NER**）等。它提供了一套强大的功能，极大地简化了文本数据中的特征提取。我们鼓励您详细探索这个库。**词义消歧**（**WSD**）的实现（[https://github.com/asad1996172/Obfuscation-Systems/blob/master/Document%20Simplification%20PAN17/WSD_with_UKB.py](https://github.com/asad1996172/Obfuscation-Systems/blob/master/Document%20Simplification%20PAN17/WSD_with_UKB.py)）可以在网上找到，并应下载到本地。

导入库的代码如下所示：

```py
import nltk
import re
import random
import pickle
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import WSD_with_UKB as wsd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
```

首先，我们将实现一个用于扩展和收缩替换的函数。我们首先从`pickle`文件中读取提取收缩列表（您必须相应地更改路径）。结果是，键是缩写，与之关联的值是相应的扩展。我们通过句子进行解析，并计算出现的扩展和收缩的数量。如果主要是收缩，我们将它们替换为扩展，如果主要是扩展，我们将它们替换为收缩。如果两者相同，我们则什么都不做：

```py
def contraction_replacement(sentence):
    # Read Contractions
    CONTRACTION_FILE = 'contraction_extraction.pickle'
    with open(CONTRACTION_FILE, 'rb') as contraction_file:
        contractions = pickle.load(contraction_file)
    # Calculate contraction counts
    all_contractions = contractions.keys()
    contractions_count = 0
    for contraction in all_contractions:
        if contraction.lower() in sentence.lower():
            contractions_count += 1
    # Calculate expansion counts
    all_expansions = contractions.values()
    expansions_count = 0
    for expansion in all_expansions:
        if expansion.lower() in sentence.lower():
            expansions_count += 1
    if contractions_count > expansions_count:
        # There are more contractions than expansions
        # So we should replace all contractions with their expansions
        temp_contractions = dict((k.lower(), v) for k, v in contractions.items())
        for contraction in all_contractions:
            if contraction.lower() in sentence.lower():
                case_insensitive = re.compile(re.escape(contraction.lower()), re.IGNORECASE)
                sentence = case_insensitive.sub(temp_contractions[contraction.lower()], sentence)
        contractions_applied = True
    elif expansions_count > contractions_count:
        # There are more expansions than contractions
        # So we should replace expansions by contractions
        inv_map = {v: k for k, v in contractions.items()}
        temp_contractions = dict((k.lower(), v) for k, v in inv_map.items())
        for expansion in all_expansions:
            if expansion.lower() in sentence.lower():
                case_insensitive = re.compile(re.escape(expansion.lower()), re.IGNORECASE)
                sentence = case_insensitive.sub(temp_contractions[expansion.lower()], sentence)
        contractions_applied = True
    else:
        # Both expansions and contractions are equal
        # So do nothing
        contractions_applied = False
    return sentence, contractions_applied
```

接下来，我们将从文本中移除任何括号。这意味着我们必须搜索与括号相关的字符——`(, ), [, ], {, }`——并将它们从文本中移除：

```py
def remove_parenthesis(sentence):
    parantheses = ['(', ')', '{', '}', '[', ']']
    for paranthesis in parantheses:
      sentence = sentence.replace(paranthesis, "")
    return sentence
```

我们将实现一个函数来从文本中清除语篇标记。我们首先读取一个语篇标记列表（您需要根据您在本地保存的方式更改文件名和路径）。然后，我们遍历列表，如果找到，则从文本中移除每个项目：

```py
def remove_discourse_markers(sentence):
    # Read Discourse Markers
    DISCOURSE_FILE = 'discourse_markers.pkl'
    with open(DISCOURSE_FILE , 'rb') as discourse_file:
        discourse_markers = pickle.load(discourse_file)
    sent_tokens = sentence.lower().split()
    for marker in discourse_markers:
        if marker.lower() in sent_tokens:
            case_insensitive = re.compile(re.escape(marker.lower()), re.IGNORECASE)
            sentence = case_insensitive.sub('', sentence)
    return sentence
```

接下来，我们将实现一个函数来从文本中移除同位语。我们将使用**正则表达式**（**regex**）匹配来完成这项工作：

```py
def remove_appositions(sentence):
    sentence = re.sub(r" ?\,[^)]+\,", "", sentence)
    return sentence
```

现在，我们将实现一个函数来更改所有格的表达式。我们首先使用正则表达式匹配来找到形式为“X of Y”的表达式。然后，我们将这个表达式替换为“Y’s X”。例如，“Jacob 的书”将变为“Jacob’s book”。请注意，我们不是确定性地进行这种替换。我们将随机选择是否替换（替换的概率为 2/3）：

```py
def apply_possessive_transformation(text):
    if re.match(r"(\w+) of (\w+)", text):
        rnd = random.choice([False, True, False])
        if rnd:
            return re.sub(r"(\w+) of (\w+)" , r"\2's \1", text)
    return text
```

接下来，我们将应用方程转换，其中我们将用它们的文本表示替换数学表达式。我们将定义一个字典，其中定义了常见的符号及其文本表示（例如，“+”翻译为“加”，"*"翻译为“乘以”）。然后，我们将找到文本中每个符号的出现，并进行必要的替换：

```py
def apply_equation_transformation(text):
    words = RegexpTokenizer(r'\w+').tokenize(text)
    symbol_to_text =   {
                '+': ' plus ',
                '-': ' minus ',
                '*': ' multiplied by ',
                '/': ' divided by ',
                '=': ' equals ',
                '>': ' greater than ',
                '<': ' less than ',
                '<=': ' less than or equal to ',
                '>=': ' greater than or equal to ',
            }
    for n,w in enumerate(words):
        for symbol in symbol_to_text:
            if symbol in w:
                words[n] = words[n].replace(symbol, symbol_to_text[sym])
    sentence = ''
    for word in words:
      sentence = sentence + word + " "
    return sentence
```

下一步是同义词替换。然而，作为一个同义词替换的辅助函数，我们需要一个*去分词*函数。这是与分词相反的操作，可以使用以下代码实现：

```py
def untokenize(words):
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()
```

现在，我们将实现实际的同义词替换：

```py
def synonym_substitution(sentence, all_words):
    new_tokens = []
    output = wsd.process_text(sentence)
    for token, synset in output:
        if synset != None:
            try:
                # Get the synset name
                synset = synset.split('-')
                offset = int(synset[0])
                pos = synset[1]
                synset_name = wn.synset_from_pos_and_offset(pos, offset)
                # List of Synonyms
                synonyms = synset_name.lemma_names()
                for synonym in synonyms:
                    if synonym.lower() not in all_words:
                        token = synonym
                        break
            except Exception as e:
                # Some error in the synset naming....
                continue
        new_tokens.append(token)
    final = untokenize(new_tokens)
    final = final.capitalize()
    return final
```

最后，我们将所有这些整合到一个包装函数中。这个函数将接受文本，并将我们所有的转换（缩写替换、括号删除、话语和同位语删除、同义词替换、方程式转换和所有格转换）应用于文本的每一句话，然后将句子重新组合成混淆文本：

```py
def obfuscate_text(input_text):
    obfuscated_text = []
    sentences = sent_tokenize(input_text)
    tokens = set(nltk.word_tokenize(input_text.lower()))
    for sentence in sentences:
        # 1\. Apply Contractions
        sentence, contractions_applied = contraction_replacement(sentence, contractions)
        # 2\. Remove Parantheses
        sentence = remove_parenthesis(sentence)
        # 3\. Remove Discourse Markers
        sentence = remove_discourse_markers(sentence, discourse_markers)
        # 4\. Remove Appositions
        sentence = remove_appositions(sentence)
        # 5\. Synonym Substitution
        sentence = synonym_substitution(sentence, tokens)
        # 6\. Apply possessive transformation
        sentence = apply_possessive_transformation(sentence)
        # 7\. Apply equation transformation
        sentence = apply_equation_transformation(sentence)
        obfuscated_text.append(sentence)
    obfuscated_text = " ".join(obfuscated_text)
    return obfuscated_text
```

我们现在将测试这种混淆的有效性。我们将训练一个基础模型，然后将其应用于混淆后的数据。这正好反映了现实世界中的威胁模型；在训练时，我们无法访问混淆后的数据。以下是我们将遵循的评估模型的过程：

1.  将数据分为训练集和测试集。

1.  从训练数据中提取特征。

1.  基于这些特征训练一个作者归属的机器学习模型。

1.  在测试数据上应用混淆器，将原始文本转换为混淆文本。

1.  从混淆文本中提取特征，并使用它们在先前训练的模型上进行推理。

我们加载数据，并像以前一样进行分割：

```py
from sklearn.model_selection import train_test_split
# Read Data
X, Y = read_dataset(num_authors = 6)
# Split it into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
```

然后，我们提取特征并训练一个模型。请注意，我们只从训练数据中提取特征，而不是测试数据（我们需要对其进行混淆）：

```py
# Extract features from training data
X_train_features = []
for x in X_train:
  x_features = calculate_features(x)
  X_train_features.append(x_features)
# Train the model
model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train_features, Y_train)
```

现在，我们将使用我们之前定义的函数对测试数据进行混淆，然后从数据的混淆版本中提取特征：

```py
X_test_obfuscated = []
for x in X_test:
  # Obfuscate
  x_obfuscated = obfuscate_text(x)
  # Extract features
  x_obfuscated_features = calculate_features(x_obfuscated)
  X_test_obfuscated.append(x_obfuscated_features)
```

最后，我们可以使用新生成的（混淆的）数据在训练的模型上进行推理：

```py
# Calculate accuracy on original
Y_pred_original = model.predict(X_test)
accuracy_orig = calculate_accuracy(Y_test, Y_pred_original)
# Calculate accuracy on obfuscated
Y_pred_obfuscated = model.predict(X_test_obfuscated)
accuracy_obf = calculate_accuracy(Y_test, Y_pred_obfuscated)
```

比较两个准确率的值应该能给出由于混淆而引起的性能下降。第一个计算值代表原始数据的准确率，第二个值代表应用我们的混淆策略后模型的准确率。当第二个值低于第一个值时，我们的混淆是成功的。

接下来，我们将概述一些提高我们混淆器性能的策略。

## 提高混淆技术

在这里，我们描述了一些可能的变化和改进，可以帮助我们提高混淆器的性能。强烈鼓励读者尝试这些方法，以检验哪些方法能展现出最佳性能。

### 高级操作

在我们的示例混淆器中，我们实现了基本的混淆策略，如同义词替换、改变缩写、删除括号等。这里有大量的特征可以被操作。以下是一些可能性：

+   **反义词替换**：用反义词的否定来替换单词。例如，*good*被替换为*not bad*。

+   **功能词操作**：在句首添加额外的辅助词，或者移除没有价值的现有词。例如，“因此，我们已经证明这个计划是可行的”变为“我们已经证明，这个计划是可行的。”

+   **标点符号操作**：添加标点符号（两个问号、两个感叹号、句尾的点号）或移除现有的标点符号。这可能会影响句子的语法和结构，这取决于你的使用情况，可能或可能不被接受。

### 语言模型

最近的发展，如变压器和注意力机制，导致了几个改进的语言模型的发展，这些模型具有出色的文本生成能力。这些模型可以用来生成混淆文本。一个例子是使用基于变压器的文档摘要器作为混淆器。摘要器的目的是以简短和简洁的方式重现原始文档中的文本。希望这样做能够去除文本的风格特征。鼓励读者尝试各种摘要模型，并比较混淆前后的准确性。请注意，检查文本与原文在意义上的相似性也很重要。

这完成了我们对作者归属混淆模型的讨论！

# 摘要

本章重点讨论了安全和隐私中的两个重要问题。我们首先讨论了作者归属，这是一个识别谁写了特定文本的任务。我们设计了一系列语言和基于文本的特征，并训练了机器学习模型进行作者归属。然后，我们转向作者归属混淆，这是一个通过改变文本以去除作者识别特征和风格标记来规避归属模型的任务。我们研究了这一系列混淆方法。对于这两个任务，我们研究了可以提高性能的改进。

作者归属和混淆在网络安全中都有重要的应用。归属可以用来检测 Sybil 账户、追踪网络犯罪分子、保护知识产权。同样，混淆可以帮助保护个人的匿名性并提供隐私保证。本章使网络安全和隐私领域的机器学习从业者能够有效地处理这两个任务。

在下一章中，我们将稍微改变方向，探讨如何使用图机器学习检测假新闻。
