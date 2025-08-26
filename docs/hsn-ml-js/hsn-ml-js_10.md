# 第十章：自然语言处理实践

自然语言处理是解析、分析和重建自然语言（如书面或口语英语、法语或德语）的科学（和艺术）。这不是一项容易的任务；**自然语言处理**（**NLP**）是一个完整的研究领域，拥有充满活力的学术研究社区和来自主要科技公司的重大资金支持。每当谷歌、苹果、亚马逊和微软投资其谷歌助手、Siri、Alexa 和 Cortana 产品时，NLP 领域就会获得更多资金。简而言之，NLP 是您能够与手机交谈，手机也能对您说话的原因。

Siri 不仅仅是 NLP。作为消费者，我们喜欢批评我们的**人工智能**（**AI**）助手当它们犯下可笑的错误。但它们确实是工程奇迹，它们能够做到任何正确的事情都是一个奇迹！

如果我看向我的手机并说，“Ok Google，给我去 7-Eleven 的路线”，我的手机将自动唤醒并对我回应，“好的，去 Main Ave 的 7-Eleven，下一个右转”。让我们思考一下要完成这个任务需要什么：

+   我的睡眠中的手机正在监控我预先训练的“OK Google”短语。

+   音频缓冲区在训练的 OK Google 声音波上得到音频哈希匹配，并唤醒手机。

+   手机开始捕捉音频，这只是一个表示声音波强度的数字时间序列向量。

+   语音音频被解码为音素，或语音声音的文本表示。为每个话语生成几个候选者。

+   将候选音素组合在一起，试图形成单词。算法使用最大似然或其他估计器来确定哪种组合最有可能是在当前上下文中实际使用的句子。

+   结果句子必须解析其意义，因此执行了许多类型的预处理，并且每个单词都被标记为其可能的**词性**（**POS**）。

+   一个学习系统（通常是人工神经网络）将尝试根据短语的主题、宾语和动词确定意图。

+   实际意图必须由子例程执行。

+   必须制定对用户的响应。在响应无法脚本化的情况下，它必须通过算法生成。

+   文本到语音算法将响应解码为音素，然后必须合成听起来自然的语音，该语音随后通过手机的扬声器播放。

恭喜你，你正在走向获得你的 Slurpee！您的体验由多个人工神经网络、各种 NLP 工具的多种用途、庞大的数据集以及数百万工程师小时的努力来构建和维护。这种体验还解释了 NLP 和 ML 之间的密切关系——它们不是同一件事，但它们在技术前沿并肩作战。

显然，NLP 的内容远不止 25 页所能涵盖的主题。本章的目标不是全面介绍；它的目标是使你熟悉在解决涉及自然语言的 ML 问题时最常用的策略。我们将快速浏览七个与 NLP 相关的概念：

+   测量字符串距离

+   TF-IDF 度量

+   文本分词

+   词干提取

+   语音学

+   词性标注

+   使用 Word2vec 进行词嵌入

如果这些主题看起来令人畏惧，请不要担心。我们将逐一介绍每个主题，并展示许多示例。在 NLP 中涉及许多术语，以及许多边缘情况，所以乍一看这个主题似乎难以接近。但毕竟，这个主题是**自然语言**：我们每天都在说它！一旦我们学会了术语，这个主题就变得相当直观，因为我们大家对语言都有非常强烈的直观理解。

我们将从一个简单的问题开始我们的讨论：你如何测量*quit*和*quote*之间的距离？我们已经知道我们可以测量空间中两点之间的距离，那么现在让我们来看看如何测量两个单词之间的距离。

# 字符串距离

总是能够测量两点之间某种形式的距离是非常方便的。在之前的章节中，我们使用了点之间的距离来辅助聚类和分类。我们也可以在 NLP 中对单词和段落做同样的事情。当然，问题是单词由字母组成，而距离由数字组成——那么我们如何从两个单词中得出一个数字呢？

输入 Levenshtein 距离*—*这是一个简单的度量，它衡量将一个字符串转换为另一个字符串所需的单字符编辑次数。Levenshtein 距离允许插入、删除和替换。Levenshtein 距离的一种修改版本，称为**Damerau-Levenshtein 距离**，也允许交换两个相邻字母。

为了用示例说明这个概念，让我们尝试将单词**crate**转换为单词**plate**：

+   将**r**替换为**l**以得到**clate**

+   将**c**替换为**p**以得到**plate**

crate 和 plate 之间的 Levenshtein 距离因此是 2。

**板**和**激光器**之间的距离是 3：

+   删除**p**以得到**late**

+   插入一个**r**以得到**later**

+   将**t**替换为**s**以得到**laser**

让我们在代码中确认这些示例。创建一个名为`Ch10-NLP`的新目录，并添加以下`package.json`文件：

```py
{
  "name": "Ch10-NLP",
  "version": "1.0.0",
  "description": "ML in JS Example for Chapter 10 - NLP",
  "main": "src/index.js",
  "author": "Burak Kanber",
  "license": "MIT",
  "scripts": {
    "start": "node src/index.js"
  },
  "dependencies": {
    "compromise": "¹¹.7.0",
    "natural": "⁰.5.6",
    "wordnet-db": "³.1.6"
  }
}
```

然后从命令行运行`yarn install`来安装依赖项。这个`package.json`文件与之前章节中的文件略有不同，因为`wordnet-db`依赖项与 Browserify 打包器不兼容。因此，我们将不得不在本章中省略一些高级 JavaScript 功能。

创建一个名为`src`的目录，并向其中添加一个`index.js`文件，你将在其中添加以下内容：

```py
const compromise = require('compromise');
const natural = require('natural');
```

你将在本章的其余部分使用这些导入，所以请将它们保存在`index.js`文件中。然而，本章中我们使用的其余代码将是可互换的；如果你愿意，在处理本章中的示例时可以删除旧的不相关代码。

让我们使用`natural.js`库来看看 Levenshtein 距离：

```py
[
    ['plate', 'laser'],
    ['parachute', 'parasail'],
    ['parachute', 'panoply']
]
    .forEach(function(pair) {
        console.log("Levenshtein distance between '"+pair[0]+"' and '"+pair[1]+"': "
            + natural.LevenshteinDistance.apply(null, pair)
        );
    });
```

在命令行中运行`yarn start`，你会看到以下输出：

```py
Levenshtein distance between 'plate' and 'laser': 3
Levenshtein distance between 'parachute' and 'parasail': 5
Levenshtein distance between 'parachute' and 'panoply': 7
```

尝试对几对单词进行实验，看看你是否能在大脑中计算出距离，以获得对它的直观感受。

Levenshtein 距离有许多用途，因为它是一个度量标准，而不是任何特定的工具。其他系统，如拼写检查器、建议器和模糊匹配器，在自己的算法中使用 Levenshtein 或编辑距离度量。

让我们看看一个更高级的度量标准：TF-IDF 分数，它表示一个特定单词在文档集中有多有趣或重要。

# 词频-逆文档频率

在搜索相关性、文本挖掘和信息检索中最受欢迎的度量标准之一是**词频-逆文档频率**（**TF-IDF**）分数。本质上，TF-IDF 衡量一个词对特定文档的重要性。因此，TF-IDF 度量标准因此只在单词属于更大文档集的文档的上下文中才有意义。

想象一下，你有一批文档，比如不同主题的博客文章，你希望使其可搜索。你的应用程序的最终用户运行了一个搜索查询，搜索的是*fashion style*。那么，你如何找到匹配的文档并根据相关性对它们进行排序？

TF-IDF 分数由两个单独但相关的组成部分组成。第一个是*词频*，即在给定文档中一个特定词的相对频率。如果一个 100 字的博客文章中包含单词*fashion*四次，那么该文档中单词*fashion*的词频是 4%。

注意，词频只需要一个词和一个文档作为参数；TF-IDF 的词频组件不需要整个文档集。

单独的词频不足以确定相关性。像*this*和*the*这样的词在大多数文本中都非常常见，并且会有很高的词频，但这些词通常与任何搜索都不相关。

因此，我们在计算中引入了第二个度量标准：逆文档频率。这个度量标准本质上是一个给定单词出现在文档中的百分比的倒数。如果你有 1,000 篇博客文章，而单词*fashion*出现在其中的 50 篇，那么该单词的非逆文档频率是 5%。逆文档频率是这个概念的扩展，通过取逆文档频率的对数给出。

如果 n[fashion]是包含单词*fashion*的文档数量，而*N*是文档总数，那么逆文档频率由*log(N / n[fashion])*给出。在我们的例子中，单词*fashion*的逆文档频率大约是 1.3。

如果我们现在考虑单词*the*，它可能出现在 90%的文档中，我们发现*the*的逆文档频率是 0.0451，远小于我们为*fashion*得到的 1.3。因此，逆文档频率衡量的是给定单词在文档集中的稀有程度或独特性；值越高，意味着单词越稀有。计算逆文档频率所需的参数是术语本身和文档语料库（与仅需要一个文档的词频不同）。

TF-IDF 分数是通过将词频和逆文档频率相乘来计算的。结果是单个指标，它封装了单个术语对特定文档的重要性或兴趣，考虑了您所看到的所有文档。像*the*和*that*这样的词可能在任何单个文档中具有高词频，但由于它们在所有文档中都普遍存在，它们的总体 TF-IDF 分数将非常低。像*fashion*这样的词，只存在于文档的子集中，将具有更高的 TF-IDF 分数。当比较两个都包含单词*fashion*的单独文档时，使用它更频繁的文档将具有更高的 TF-IDF 分数，因为两个文档的逆文档频率部分将是相同的。

在对搜索结果进行相关性评分时，最常见的方法是计算搜索查询中每个术语以及语料库中每个文档的 TF-IDF 分数。每个查询术语的个别 TF-IDF 分数可以相加，得到的总和可以称为该特定文档的**相关性分数**。一旦所有匹配的文档都以这种方式评分，就可以按相关性排序并按此顺序显示它们。大多数全文搜索引擎，如 Lucene 和 Elasticsearch，都使用这种相关性评分方法。

让我们通过使用`natural.js` TF-IDF 工具来实际看看。将以下内容添加到`index.js`中：

```py
const fulltextSearch = (query, documents) => {
    const db = new natural.TfIdf();
    documents.forEach(document => db.addDocument(document));
    db.tfidfs(query, (docId, score) => {
        console.log("DocID " + docId + " has score: " + score);
    });
};

fulltextSearch("fashion style", [
    "i love cooking, it really relaxes me and makes me feel at home",
    "food and restaurants are basically my favorite things",
    "i'm not really a fashionable person",
    "that new fashion blogger has a really great style",
    "i don't love the cinematic style of that movie"
]);
```

此代码定义了一个`fulltextSearch`函数，该函数接受一个搜索查询和要搜索的文档数组。每个文档都添加到 TF-IDF 数据库对象中，其中它被`natural.js`自动分词。使用`yarn start`运行程序，您将看到以下输出：

```py
DocID 0 has score: 0
DocID 1 has score: 0
DocID 2 has score: 0
DocID 3 has score: 3.4271163556401456
DocID 4 has score: 1.5108256237659907
```

前两个文档与时尚或风格无关，返回的分数为零。这些文档中*时尚*和*风格*的词频组件为零，因此整体分数变为零。第三个文档的分数也是零。然而，该文档确实提到了时尚，但是分词器无法将单词*时尚的*与*时尚*相匹配，因为没有进行词干提取。我们将在本章后面的部分深入讨论分词和词干提取，但就目前而言，了解*词干提取*是一种将单词还原为其词根形式的操作就足够了。

第三个和第四个文档的分数不为零。第三个文档的分数更高，因为它包含了*时尚*和*风格*这两个词，而第四个文档只包含了*风格*这个词。这个简单的指标在捕捉相关性方面做得出奇的好，这也是为什么它被广泛使用的原因。

让我们更新我们的代码以添加一个词干提取操作。在应用词干提取到文本之后，我们预计第二个文档也将有一个非零的相关性分数，因为*时尚的*应该被词干提取器转换为*时尚*。将以下代码添加到`index.js`中：

```py
const stemmedFulltextSearch = (query, documents) => {
    const db = new natural.TfIdf();
    const tokenizer = new natural.WordTokenizer();
    const stemmer = natural.PorterStemmer.stem;
    const stemAndTokenize = text => tokenizer.tokenize(text).map(token => stemmer(token));

    documents.forEach(document => db.addDocument(stemAndTokenize(document)));
    db.tfidfs(stemAndTokenize(query), (docId, score) => {
        console.log("DocID " + docId + " has score: " + score);
    });
};

stemmedFulltextSearch("fashion style", [
    "i love cooking, it really relaxes me and makes me feel at home",
    "food and restaurants are basically my favorite things",
    "i'm not really a fashionable person",
    "that new fashion blogger has a really great style",
    "i don't love the cinematic style of that movie"
]);
```

我们已经添加了一个`stemAndTokenize`辅助方法，并将其应用于添加到数据库中的文档以及搜索查询。使用`yarn start`运行代码，你会看到更新的输出：

```py
DocID 0 has score: 0
DocID 1 has score: 0
DocID 2 has score: 1.5108256237659907
DocID 3 has score: 3.0216512475319814
DocID 4 has score: 1.5108256237659907
```

如预期的那样，第二个文档现在有一个非零分数，因为词干提取器能够将单词*时尚的*转换为*时尚*。第二个和第四个文档的分数相同，但这仅仅是因为这是一个非常简单的例子；在一个更大的语料库中，我们不会期望*时尚*和*风格*这两个词的逆文档频率是相等的。

TF-IDF 不仅用于搜索相关性和排名。这个指标在许多用例和问题领域中得到了广泛的应用。TF-IDF 的一个有趣用途是文章摘要。在文章摘要中，目标是减少一段文字，只保留几个能够有效总结该段落的句子。

解决文章摘要问题的方法之一是将文章中的每个句子或段落视为一个单独的文档。在为 TF-IDF 索引每个句子之后，然后评估每个单词的 TF-IDF 分数，并使用这些分数对整个句子进行评分。选择前三或五个句子，并按原始顺序显示它们，你将得到一个不错的摘要。

让我们看看这个实际应用，使用`natural.js`和`compromise.js`。将以下代码添加到`index.js`中：

```py
const summarize = (article, maxSentences = 3) => {
    const sentences = compromise(article).sentences().out('array');
    const db = new natural.TfIdf();
    const tokenizer = new natural.WordTokenizer();
    const stemmer = natural.PorterStemmer.stem;
    const stemAndTokenize = text => tokenizer.tokenize(text).map(token => stemmer(token));
    const scoresMap = {};

    // Add each sentence to the document
    sentences.forEach(sentence => db.addDocument(stemAndTokenize(sentence)));

    // Loop over all words in the document and add that word's score to an overall score for each sentence
    stemAndTokenize(article).forEach(token => {
        db.tfidfs(token, (sentenceId, score) => {
            if (!scoresMap[sentenceId]) scoresMap[sentenceId] = 0;
            scoresMap[sentenceId] += score;
        });
    });

    // Convert our scoresMap into an array so that we can easily sort it
    let scoresArray = Object.entries(scoresMap).map(item => ({score: item[1], sentenceId: item[0]}));
    // Sort the array by descending score
    scoresArray.sort((a, b) => a.score < b.score ? 1 : -1);
    // Pick the top maxSentences sentences
    scoresArray = scoresArray.slice(0, maxSentences);
    // Re-sort by ascending sentenceId
    scoresArray.sort((a, b) => parseInt(a.sentenceId) < parseInt(b.sentenceId) ? -1 : 1);
    // Return sentences
    return scoresArray
        .map(item => sentences[item.sentenceId])
        .join('. ');

};
```

之前的`summarize`方法实现了以下步骤：

+   使用`compromise.js`从文章中提取句子

+   将每个单独的句子添加到 TF-IDF 数据库中

+   对于文章中的每个单词，计算其在每个句子中的 TF-IDF 分数

+   将每个单词的 TF-IDF 分数添加到每个句子的总分数列表（`scoresMap`对象）中

+   将`scoresMap`转换为数组，以便排序更简单

+   按降序相关性分数对`scoresArray`进行排序

+   删除除了得分最高的句子之外的所有句子

+   按句子的时间顺序重新排序`scoresArray`

+   通过连接得分最高的句子来构建摘要

让我们在代码中添加一个简单的文章，并尝试使用三句和五句的摘要。在这个例子中，我会使用本节的前几段，但你可以用任何你喜欢的内容替换文本。将以下内容添加到`index.js`中：

```py
const summarizableArticle = "One of the most popular metrics used in search relevance, text mining, and information retrieval is the term frequency - inverse document frequency score, or tf-idf for short. In essence, tf-idf measures how significant a word is to a particular document. The tf-idf metric therefore only makes sense in the context of a word in a document that's part of a larger corpus of documents. Imagine you have a corpus of documents, like blog posts on varying topics, that you want to make searchable. The end user of your application runs a search query for fashion style. How do you then find matching documents and rank them by relevance? The tf-idf score is made of two separate but related components. The first is term frequency, or the relative frequency of a specific term in a given document. If a 100-word blog post contains the word fashion four times, then the term frequency of the word fashion is 4% for that one document. Note that term frequency only requires a single term and a single document as parameters; the full corpus of documents is not required for the term frequency component of tf-idf. Term frequency by itself is not sufficient to determine relevance, however. Words like this and the appear very frequently in most text and will have high term frequencies, but those words are not typically relevant to any search.";

console.log("3-sentence summary:");
console.log(summarize(summarizableArticle, 3));
console.log("5-sentence summary:");
console.log(summarize(summarizableArticle, 5));
```

当你使用`yarn start`运行代码时，你会看到以下输出：

```py
3-sentence summary:
 the tf idf metric therefore only makes sense in the context of a word in a document that's part of a larger corpus of documents. if a 100-word blog post contains the word fashion four times then the term frequency of the word fashion is 4% for that one document. note that term frequency only requires a single term and a single document as parameters the full corpus of documents is not required for the term frequency component of tf idf

 5-sentence summary:
 one of the most popular metrics used in search relevance text mining and information retrieval is the term frequency inverse document frequency score or tf idf for short. the tf idf metric therefore only makes sense in the context of a word in a document that's part of a larger corpus of documents. the first is term frequency or the relative frequency of a specific term in a given document. if a 100-word blog post contains the word fashion four times then the term frequency of the word fashion is 4% for that one document. note that term frequency only requires a single term and a single document as parameters the full corpus of documents is not required for the term frequency component of tf idf
```

这些摘要的质量展示了`tf-idf 度量`的强大功能和灵活性，同时也突出了这样一个事实：你并不总是需要高级的 ML 或 AI 算法来完成有趣的任务。TF-IDF 有许多其他用途，所以你应该考虑在需要将单词或术语与语料库中的文档的相关性相关联时使用此度量。

在本节中，我们使用了分词器和词干提取器，但没有正式介绍它们。这些是 NLP 中的核心概念，所以现在让我们正式介绍它们。

# 分词

分词是将输入字符串（如句子、段落，甚至是一个对象，如电子邮件）转换为单个*tokens*的行为。一个非常简单的分词器可能会将句子或段落按空格分割，从而生成单个单词的 tokens。然而，tokens 不一定是单词，输入字符串中的每个单词也不一定需要被分词器返回，分词器生成的每个 tokens 也不一定需要在原始文本中存在，而且一个 tokens 也不一定只代表一个单词。因此，我们使用*token*这个词而不是*word*来描述分词器的输出，因为 tokens 并不总是单词。

在使用机器学习算法处理文本之前进行分词的方式对算法的性能有重大影响。许多 NLP 和 ML 应用使用*词袋模型*方法，其中只关注单词或 tokens，而不关注它们的顺序，就像我们在第五章中探讨的朴素贝叶斯分类器一样，*分类算法*。然而，生成*二元组*（即相邻单词的成对）的分词器实际上在用于词袋模型算法时，会保留原始文本的一些位置和语义意义。

文本标记化有许多方法。如前所述，最简单的方法是将句子通过空格拆分以生成一个*标记流*，其中包含单个单词。然而，简单方法存在许多问题。首先，算法将大写单词视为与其小写版本不同；Buffalo 和 buffalo 被视为两个不同的单词或标记。有时这是可取的，有时则不然。过于简化的标记化还将像*won't*这样的缩写视为独立且与单词*will not*不同，后者将被拆分为两个单独的标记，*will*和*not*。

在大多数情况下，即在 80%的应用中，一个人应该考虑的最简单的标记化是，将所有文本转换为小写，删除标点符号和新行，删除格式化和标记，如 HTML，甚至删除*停用词*或常见单词，如*this*或*the*。在其他情况下，需要更高级的标记化，在某些情况下，需要更简单的标记化。

在本节中，我一直在描述标记化行为作为一个复合过程，包括大小写转换、删除非字母数字字符和停用词过滤。然而，标记化库将各自有自己的观点，关于标记化器的角色和责任。您可能需要将库的标记化工具与其他工具结合使用，以实现所需的效果。

首先，让我们构建自己的简单标记化器。这个标记化器将字符串转换为小写，删除非字母数字字符，并删除长度少于三个字符的单词。将以下内容添加到您的`index.js`文件中，要么替换 Levenshtein 距离代码，要么添加到其下方：

```py
const tokenizablePhrase = "I've not yet seen 'THOR: RAGNAROK'; I've heard it's a great movie though. What'd you think of it?";

const simpleTokenizer = (text) =>
    text.toLowerCase()
        .replace(/(\w)'(\w)/g, '$1$2')
        .replace(/\W/g, ' ')
        .split(' ')
        .filter(token => token.length > 2);

console.log(simpleTokenizer(tokenizablePhrase));
```

这个`simpleTokenizer`会将字符串转换为小写，删除单词中间的撇号（因此*won't*变为*wont*），并通过将所有其他非单词字符替换为空格来过滤掉所有其他非单词字符。然后，它通过空格字符拆分字符串，返回一个数组，并最终删除任何少于三个字符的项目。

运行`yarn start`，您将看到以下内容：

```py
[ 'ive', 'not', 'yet', 'seen', 'thor',
 'ragnarok', 'ive', 'heard', 'its',
 'great', 'movie', 'though',
 'whatd', 'you', 'think' ]
```

这个标记流可以被提供给一个算法，无论是按顺序还是无序。例如，朴素贝叶斯分类器将忽略顺序，并将每个单词视为独立进行分析。

让我们比较我们的简单标记化器与`natural.js`和`compromise.js`提供的两个标记化器。将以下内容添加到您的`index.js`文件中：

```py
console.log("Natural.js Word Tokenizer:");
console.log((new natural.WordTokenizer()).tokenize(tokenizablePhrase));
```

使用`yarn start`运行代码将产生以下输出：

```py
Natural.js Word Tokenizer:
 [ 'I', 've', 'not', 'yet', 'seen',
 'THOR', 'RAGNAROK', 'I', 've',
 'heard', 'it', 's', 'a', 'great', 'movie',
 'though', 'What', 'd', 'you', 'think',
 'of', 'it' ]
```

如您所见，短单词已被保留，并且像*I've*这样的缩写已被拆分为单独的标记。此外，大小写也被保留。

让我们尝试另一个`natural.js`标记化器：

```py
console.log("Natural.js WordPunct Tokenizer:");
console.log((new natural.WordPunctTokenizer()).tokenize(tokenizablePhrase));
```

这将产生：

```py
Natural.js WordPunct Tokenizer:
 [ 'I', '\'', 've', 'not', 'yet', 'seen',
 '\'', 'THOR', ': ', 'RAGNAROK', '\'', '; ',
 'I', '\'', 've', 'heard', 'it', '\'', 's',
 'a', 'great', 'movie', 'though', '.', 'What',
 '\'', 'd', 'you', 'think', 'of',
 'it', '?' ]
```

然而，这个标记化器继续在标点符号上拆分，但标点符号本身被保留。在标点符号重要的应用中，这可能是有需求的。

其他分词库，例如`compromise.js`中的分词库，采取了一种更智能的方法，甚至在分词的同时进行词性标注，以便在分词过程中解析和理解句子。让我们尝试几种`compromise.js`的分词技术：

```py
console.log("Compromise.js Words:");
console.log(compromise(tokenizablePhrase).words().out('array'));
console.log("Compromise.js Adjectives:");
console.log(compromise(tokenizablePhrase).adjectives().out('array'));
console.log("Compromise.js Nouns:");
console.log(compromise(tokenizablePhrase).nouns().out('array'));
console.log("Compromise.js Questions:");
console.log(compromise(tokenizablePhrase).questions().out('array'));
console.log("Compromise.js Contractions:");
console.log(compromise(tokenizablePhrase).contractions().out('array'));
console.log("Compromise.js Contractions, Expanded:");
console.log(compromise(tokenizablePhrase).contractions().expand().out('array'));
```

使用`yarn start`运行新代码，您将看到以下内容：

```py
Compromise.js Words:
 [ 'i\'ve', '', 'not', 'yet', 'seen',
 'thor', 'ragnarok', 'i\'ve', '', 'heard',
 'it\'s', '', 'a', 'great', 'movie', 'though',
 'what\'d', '', 'you', 'think', 'of', 'it' ]
 Compromise.js Adjectives:
 [ 'great' ]
 Compromise.js Nouns:
 [ 'thor', 'ragnarok', 'movie' ]
 Compromise.js Questions:
 [ 'what\'d you think of it' ]
 Compromise.js Contractions:
 [ 'i\'ve', 'i\'ve', 'it\'s', 'what\'d' ]
 Compromise.js Contractions, Expanded:
 [ 'i have', 'i have', 'it is', 'what did' ]
```

`words()`分词器不会像`natural.js`分词器那样将缩写词分开。此外，`compromise.js`还为您提供从文本中提取特定实体类型的能力。我们可以分别提取形容词、名词、动词、疑问词、缩写词（甚至具有扩展缩写词的能力）；我们还可以使用`compromise.js`提取日期、标签、列表、从句和数值。

您的标记不必直接映射到输入文本中的单词和短语。例如，当为电子邮件系统开发垃圾邮件过滤器时，您可能会发现将一些来自电子邮件头部的数据包含在标记流中可以大幅提高准确性。电子邮件是否通过 SPF 和 DKIM 检查可能对您的垃圾邮件过滤器来说是一个非常强烈的信号。您还可能发现区分正文文本和主题行也是有益的；可能的情况是，作为超链接出现的单词比纯文本中的单词是更强的信号。

通常，对这种半结构化数据进行分词的最简单方法是在标记前加上一个或一组通常不允许分词器使用的字符。例如，电子邮件主题行中的标记可能以`_SUBJ:`为前缀，而出现在超链接中的标记可能以`_LINK:`为前缀。为了说明这一点，这里是一个电子邮件标记流的示例：

```py
['_SPF:PASS',
 '_DKIM:FAIL',
 '_SUBJ:buy',
 '_SUBJ:pharmaceuticals',
 '_SUBJ:online',
 '_LINK:pay',
 '_LINK:bitcoin',
 'are',
 'you',
 'interested',
 'buying',
 'medicine',
 'online']
```

即使朴素贝叶斯分类器以前从未见过关于药品的引用，它也可能发现大多数垃圾邮件邮件都未能通过 DKIM 检查，但仍将此消息标记为垃圾邮件。或者，也许您与会计部门紧密合作，他们经常收到有关付款的电子邮件，但几乎从未收到包含指向外部网站的超链接中的单词`pay`的合法电子邮件；在纯文本中出现的`*pay*`标记与在超链接中出现的`_LINK:pay`标记之间的区分可能对电子邮件是否被分类为垃圾邮件有决定性的影响。

实际上，最早期的垃圾邮件过滤突破之一，由 Y Combinator 的保罗·格雷厄姆开发，就是使用这种带有注释的电子邮件标记的方法，显著提高了早期垃圾邮件过滤器的准确性。

另一种分词方法是*n-gram*分词，它将输入字符串分割成 N 个相邻标记的 N 大小组。实际上，所有分词都是 n-gram 分词，然而，在前面的例子中，N 被设置为 1。更典型的是，n-gram 分词通常指的是 N > 1 的方案。最常见的是*二元组*和*三元组*分词。

二元和三元标记化的目的是保留围绕单个单词的一些上下文。与情感分析相关的一个例子是易于可视化。短语*I did not love the movie*将被标记化（使用单语标记化器，或 n-gram 标记化器，其中 N = 1）为*I*，*did*，*not*，*love*，*the*，*movie*。当使用如朴素贝叶斯这样的词袋算法时，算法将看到单词*love*并猜测句子具有积极情感，因为词袋算法不考虑单词之间的关系。

另一方面，二元标记化器可以欺骗一个简单的算法去考虑单词之间的关系，因为每一对单词都变成了一个标记。使用二元标记化器处理的前一个短语将变成*I did*，*did not*，*not love*，*love the*，*the movie*。尽管每个标记由两个单独的单词组成，但算法是在标记上操作的，因此会将*not love*与*I love*区别对待。因此，情感分析器将围绕每个单词有更多的上下文，并能区分否定（*not love*）和积极短语。

让我们在先前的示例句子上尝试`natural.js`二元标记化器。将以下代码添加到`index.js`中：

```py
console.log("Natural.js bigrams:");
console.log(natural.NGrams.bigrams(tokenizablePhrase));
```

使用`yarn start`运行代码将产生：

```py
Natural.js bigrams:
 [ [ 'I', 've' ],
 [ 've', 'not' ],
 [ 'not', 'yet' ],
 [ 'yet', 'seen' ],
 [ 'seen', 'THOR' ],
 [ 'THOR', 'RAGNAROK' ],
 [ 'RAGNAROK', 'I' ],
 [ 'I', 've' ],
 [ 've', 'heard' ],
 [ 'heard', 'it' ],
 [ 'it', 's' ],
 [ 's', 'a' ],
 [ 'a', 'great' ],
 [ 'great', 'movie' ],
 [ 'movie', 'though' ],
 [ 'though', 'What' ],
 [ 'What', 'd' ],
 [ 'd', 'you' ],
 [ 'you', 'think' ],
 [ 'think', 'of' ],
 [ 'of', 'it' ] ]
```

n-gram 标记化最大的问题是它会显著增加数据域的熵。当在 n-gram 上训练算法时，你不仅要确保算法学习到所有重要的单词，还要学习到所有重要的**单词对**。单词对的数量比唯一的单词数量要多得多，因此 n-gram 标记化只有在你有一个非常庞大且全面的训练集时才能工作。

一种巧妙地绕过 n-gram 熵问题的方法，尤其是在处理情感分析中的否定时，是将否定词后面的标记以与处理电子邮件标题和主题行相同的方式进行转换。例如，短语*not love*可以被标记为*not*, *_NOT:love*，或者*not*, *!love*，甚至只是*!love*（将*not*作为一个单独的标记丢弃）。

在这个方案下，短语*I did not love the movie*将被标记化为*I*，*did*，*not*，*_NOT:love*，*the*，*movie*。这种方法的优势在于上下文否定仍然得到了保留，但总的来说，我们仍然使用低熵的单语标记，这些标记可以用较小的数据集进行训练。

标记文本有许多方法，每种方法都有其优缺点。正如往常一样，你选择的方法将取决于手头的任务、可用的训练数据以及问题域本身。

在接下来的几节中，请始终牢记分词的主题，因为这些主题也可以应用于分词过程。例如，您可以在分词后对单词进行词干提取以进一步减少熵，或者您可以根据它们的 TF-IDF 分数过滤您的标记，因此只使用文档中最有趣的单词。

为了继续我们关于熵的讨论，让我们花一点时间来讨论*词干提取*。

# 词干提取

词干提取是一种可以应用于单个单词的转换类型，尽管通常词干操作发生在分词之后。在分词后进行词干提取非常常见，以至于`natural.js`提供了一个`tokenizeAndStem`便利方法，可以附加到`String`类原型上。

具体来说，词干提取将单词还原为其词根形式，例如将*running*转换为*run*。在分词后对文本进行词干提取可以显著减少数据集的熵，因为它本质上去除了具有相似意义但时态或词形不同的单词。您的算法不需要分别学习单词*run*、*runs*、*running*和*runnings*，因为它们都将被转换为*run*。

最受欢迎的词干提取算法，即*Porter*词干提取器，是一种定义了多个阶段规则的启发式算法。但本质上，它归结为从单词末尾切掉标准的动词和名词词形变化，并处理出现的特定边缘情况和常见不规则形式。

从某种意义上说，词干提取是一种压缩算法，它丢弃了关于词形变化和特定单词形式的信息，但保留了由词根留下的概念信息。因此，在词形变化或语言形式本身很重要的场合不应使用词干提取。

由于同样的原因，词干提取在概念信息比形式更重要的情况下表现优异。主题提取就是一个很好的例子：无论是某人写关于自己作为跑者的经历还是观看田径比赛的经历，他们都是在写关于跑步。

由于词干提取减少了数据熵，因此在数据集较小或适度大小时非常有效地使用。然而，词干提取不能随意使用。如果在不必要的情况下使用词干提取，非常大的数据集可能会因准确性降低而受到惩罚。您在提取文本时会破坏信息，具有非常大的训练集的模型可能已经能够使用这些额外信息来生成更好的预测。

在实践中，您永远不需要猜测您的模型是否在带词干或不带词干的情况下表现更好：您应该尝试两种方法，看看哪种表现更好。我无法告诉您何时使用词干提取，我只能告诉您为什么它有效，以及为什么有时它不起作用。

让我们尝试一下`natural.js`的 Porter 词干提取器，并将其与之前的分词结合起来。将以下内容添加到`index.js`中：

```py
console.log("Tokenized and stemmed:");
console.log(
    (new natural.WordTokenizer())
        .tokenize(
            "Writing and write, lucky and luckies, part parts and parted"
        )
        .map(natural.PorterStemmer.stem)
```

使用`yarn start`运行代码，你会看到以下内容：

```py
Tokenized and stemmed:
 [ 'write', 'and', 'write',
 'lucki', 'and', 'lucki',
 'part', 'part', 'and', 'part' ]
```

这个简单的例子说明了不同形式的单词是如何被简化为其概念意义的。它还说明了，并不能保证词干提取器会创建出**真实**的单词（你不会在词典中找到`lucki`），而只是它会为一系列结构相似的单词减少熵。

有其他词干提取算法试图从更语言学角度来解决这个问题。这种类型的词干提取被称为**词元化**，而词元的对应物称为**词元**，或单词的词典形式。本质上，词元化器是一个词干提取器，它首先确定单词的词性（通常需要一个词典，如*WordNet*），然后应用针对该特定词性的深入规则，可能涉及更多的查找表。例如，单词*better*在词干提取中保持不变，但通过词元化它被转换成单词*good*。在大多数日常任务中，词元化并不是必要的，但在你的问题需要更精确的语言学规则或显著减少熵时可能是有用的。

我们在讨论自然语言处理或语言学时，不能不讨论最常见的交流方式：语音。语音转文字或文字转语音系统实际上是如何知道如何说出英语中定义的数十万个单词，以及任意数量的名字的呢？答案是**声音学**。

# 声音学

语音检测，如语音转文字系统中使用的，是一个出人意料困难的问题。说话的风格、发音、方言和口音，以及节奏、音调、速度和发音的变化如此之多，再加上音频是一个简单的一维时间域信号的事实，因此，即使是当今最先进的智能手机技术也只是**良好，而非卓越**。

虽然现代语音转文字技术比我要展示的深入得多，但我希望向你展示**声音学算法**的概念。这些算法将一个单词转换成类似声音散列的东西，使得识别听起来相似的字词变得容易。

**元音算法**就是这样一种声音学算法。它的目的是将一个单词简化为一个简化的声音形式，最终目标是能够索引相似的发音。元音算法使用 16 个字符的字母表：0BFHJKLMNPRSTWXY。0 字符代表**th**音，*X*代表**sh**或**ch**音，其他字母按常规发音。几乎所有的元音信息都在转换中丢失，尽管如果它们是一个单词的第一个声音，一些元音会被保留。

一个简单的例子说明了音位算法可能在哪里有用。想象一下，你负责一个搜索引擎，人们不断搜索“知识就是力量，法国是培根”。你熟悉艺术史，会明白实际上是弗朗西斯·培根说过“知识就是力量”，而你的用户只是听错了引言。你希望在你的搜索结果中添加一个“你是指：**弗朗西斯·培根**”的链接，但你不知道如何解决这个问题。

让我们看看 Metaphone 算法如何将`France is Bacon`和`Francis Bacon`这两个术语音位化。在`index.js`中添加以下内容：

```py
console.log(
    (new natural.WordTokenizer())
        .tokenize("Francis Bacon and France is Bacon")
        .map(t => natural.Metaphone.process(t))
);
```

当你使用`yarn start`运行代码时，你会看到以下内容：

```py
[ 'FRNSS', 'BKN', 'ANT', 'FRNS', 'IS', 'BKN' ]
```

弗朗西斯已经变成了`FRNSS`，法国变成了`FRNS`，而培根变成了`BKN`。直观上，这些字符串代表了用来发音单词的最易区分的音素。

在音位化之后，我们可以使用 Levenshtein 距离来衡量两个单词之间的相似度。如果你忽略空格，*FRNSS BKN*和*FRNS IS BKN*之间只有一个 Levenshtein 距离（添加了*I*）；因此这两个短语听起来非常相似。你可以使用这些信息，结合搜索词的其余部分和反向查找，来确定`France is Bacon`是`Francis Bacon`可能的误读，并且`Francis Bacon`实际上是你在搜索结果中应该展示的正确主题。像`France is Bacon`这样的音位拼写错误和误解非常普遍，以至于我们在一些拼写检查工具中也使用它们。

在语音到文本系统中，使用了一种类似的方法。录音系统尽力捕捉你发出的特定元音和辅音音素，并使用音位索引（音位映射到各种词典单词的反向查找）来提出一组候选单词。通常，一个神经网络将确定哪种单词组合最有可能，考虑到音位形式的置信度和结果语句的语义意义或无意义。最有意义的单词集就是展示给你的。

`natural.js`库还提供了一个方便的方法来比较两个单词，如果它们听起来相似则返回`true`。尝试以下代码：

```py
console.log(natural.Metaphone.compare("praise", "preys"));
console.log(natural.Metaphone.compare("praise", "frays"));
```

运行时，这将返回`true`然后`false`。

当你的问题涉及发音或处理类似发音的单词和短语时，你应该考虑使用音位算法。这通常限于更专业的领域，但语音到文本和文本到语音系统变得越来越受欢迎，你可能会发现自己需要更新你的搜索算法以适应语音相似音素，如果用户未来通过语音与你服务互动的话。

说到语音系统，现在让我们看看 POS 标注以及它是如何用于从短语中提取语义信息的——例如，您可能对智能手机助手下达的命令。

# 词性标注

**词性**（**POS**）标注器分析一段文本，如一个句子，并确定句子中每个单词的词性。唯一实现这一点的方法是字典查找，因此它不是一个仅从第一原理开发的算法。

POS 标注的一个很好的用例是从命令中提取意图。例如，当你对 Siri 说“请从约翰的比萨店为我订一份披萨”时，人工智能系统将使用词性对命令进行标注，以便从命令中提取主语、谓语、宾语以及任何其他相关细节。

此外，POS 标注通常用作其他 NLP 操作的辅助工具。例如，主题提取就大量使用了 POS 标注，以便将人、地点和主题从动词和形容词中分离出来。

请记住，由于英语语言的歧义性，POS 标注永远不会完美。许多词既可以作名词也可以作动词，因此许多 POS 标注器将为给定单词返回一系列候选词性。执行 POS 标注的库具有广泛的复杂性，从简单的启发式方法到字典查找，再到基于上下文尝试确定词性的高级模型。

`compromise.js` 库具有灵活的 POS 标注器和匹配/提取系统。`compromise.js` 库的独特之处在于它旨在“足够好”但不是全面的；它仅训练了英语中最常见的单词，这对于大多数情况来说足够提供 80-90% 的准确性，同时仍然是一个快速且小巧的库。

让我们看看 `compromise.js` 的 POS 标注和匹配的实际效果。将以下代码添加到 `index.js` 中：

```py
const siriCommand = "Hey Siri, order me a pizza from John's pizzeria";
const siriCommandObject = compromise(siriCommand);

console.log(siriCommandObject.verbs().out('array'));
console.log(siriCommandObject.nouns().out('array'));
```

使用 `compromise.js` 允许我们从命令中提取仅动词，或仅名词（以及其他词性）。使用 `yarn start` 运行代码将产生：

```py
[ 'order' ]
[ 'siri', 'pizza', 'john\'s pizzeria' ]
```

POS 标记器已将 `order` 识别为句子中的唯一动词；然后可以使用此信息来加载 Siri 人工智能系统中内置的用于下订单的正确子程序。然后可以将提取出的名词发送到子程序，以确定要下何种类型的订单以及从哪里下。

令人印象深刻的是，POS 标注器还将 `John's pizzeria` 识别为一个单独的名词，而不是将 `John's` 和 `pizzeria` 视为单独的名词。标注器已经理解 `John's` 是一个所有格，因此适用于其后的单词。

我们还可以使用 `compromise.js` 编写用于常见命令的解析和提取规则。让我们试一个例子：

```py
console.log(
    compromise("Hey Siri, order me a pizza from John's pizzeria")
        .match("#Noun [#Verb me a #Noun+ *+ #Noun+]").out('text')
);

console.log(
    compromise("OK Google, write me a letter to the congressman")
        .match("#Noun [#Verb me a #Noun+ *+ #Noun+]").out('text')
);
```

使用 `yarn start` 运行代码将产生：

```py
order me a pizza from John's
write me a letter to the congressman
```

相同的匹配选择器能够捕捉这两个命令，通过匹配组（用`[]`表示）忽略命令的接收者（Siri 或 Google）。因为这两个命令都遵循动词-名词-名词的模式，所以两者都会匹配选择器。

当然，仅凭这个选择器本身是不够构建一个完整的 AI 系统，如 Siri 或 Google Assistant 的。这个工具将在 AI 系统过程的早期使用，以便根据预定义但灵活的命令格式确定用户的整体意图。你可以编程一个系统来响应诸如“打开我的#名词”这样的短语，其中名词可以是“日历”、“电子邮件”或`Spotify`，或者“给#名词写一封电子邮件”，等等。这个工具可以用作构建自己的语音或自然语言命令系统的第一步，以及用于各种主题提取应用。

在本章中，我们讨论了 NLP 中使用的基石工具。许多高级 NLP 任务将 ANN 作为学习过程的一部分，但对于许多新手实践者来说，如何将单词和自然语言发送到 ANN 的输入层并不明确。在下一节中，我们将讨论“词嵌入”，特别是 Word2vec 算法，它可以用来将单词输入到 ANN 和其他系统中。

# 词嵌入和神经网络

在本章中，我们讨论了各种 NLP 技术，特别是关于文本预处理。在许多用例中，我们需要与 ANN 交互以执行最终分析。分析的类型与这一节无关，但想象你正在开发一个情感分析 ANN。你适当地标记和词干化你的训练文本，然后，当你尝试在预处理后的文本上训练你的 ANN 时，你意识到你不知道如何将单词输入到神经网络中。

最简单的方法是将网络中的每个输入神经元映射到一个独特的单词。在处理文档时，你可以将输入神经元的值设置为该单词在文档中的词频（或绝对计数）。你将拥有一个网络，其中一个输入神经元对单词“时尚”做出反应，另一个神经元对“技术”做出反应，另一个神经元对“食物”做出反应，等等。

这种方法可以工作，但它有几个缺点。ANN 的拓扑结构必须预先定义，因此在开始训练网络之前，你必须知道你的训练集中有多少独特的单词；这将成为输入层的大小。这也意味着一旦网络被训练，它就无法学习新单词。要向网络添加新单词，你实际上必须从头开始构建和训练一个新的网络。

此外，在整个文档语料库中，你可能会遇到成千上万的独特单词。这会对 ANN 的效率产生巨大的负面影响，因为你将需要一个有 10,000 个神经元的输入层。这将大大增加网络所需的训练时间，以及系统的内存和处理需求。

每个神经元对应一个单词的方法在直观上感觉效率不高。虽然你的语料库包含 10,000 个独特的单词，但其中大多数将是罕见的，并且只出现在少数文档中。对于大多数文档，只有几百个输入神经元会被激活，其他则设置为零。这相当于所谓的**稀疏矩阵**或**稀疏向量**，或者是一个大部分值都是零的向量。

因此，当自然语言与人工神经网络（ANNs）交互时，需要一种更高级的方法。一种被称为**词嵌入**的技术族可以分析文本语料库，并将每个单词转换为一个固定长度的数值向量。这个向量与哈希（如 md5 或 sha1）作为任意数据的固定长度表示方式类似，也是单词的固定长度表示。

词嵌入提供了几个优势，尤其是在与人工神经网络结合使用时。由于单词向量长度固定，网络的拓扑结构可以在事先决定，并且也可以处理初始训练后新词的出现。

单词向量也是**密集向量**，这意味着你不需要在你的网络中有 10,000 个输入神经元。单词向量（以及输入层）的大小一个好的值是在 100-300 项之间。这个因素本身就可以显著降低你的 ANN 的维度，并允许模型训练和收敛更快。

有许多词嵌入算法可供选择，但当前最先进的选项是谷歌开发的 Word2vec 算法。这个特定的算法还有一个令人向往的特性：在向量表示方面，相似的单词会聚集在一起。

在本章的早期，我们看到了我们可以使用字符串距离来衡量两个单词之间的印刷距离。我们还可以使用两个单词的音位表示之间的字符串距离来衡量它们听起来有多相似。当使用 Word2vec 时，你可以测量两个单词向量之间的距离，以获取两个单词之间的**概念**距离。

Word2vec 算法本身是一个浅层神经网络，它在你文本语料库上自我训练。该算法使用 n-gram 来发展单词之间的上下文感觉。如果你的语料库中“时尚”和“博主”经常一起出现，Word2vec 将为这些单词分配相似的向量。如果“时尚”和“数学”很少一起出现，它们的结果向量将被一定距离分开。因此，两个词向量之间的距离代表了它们的概念和上下文距离，或者两个单词在语义内容和上下文方面有多相似。

Word2vec 算法的这一特性也赋予了最终处理数据的 ANN 自己的效率和准确性优势，因为词向量将为相似单词激活相似的输入神经元。Word2vec 算法不仅降低了问题的维度，还为词嵌入添加了上下文信息。这种额外的上下文信息正是 ANN 非常擅长捕捉的信号类型。

以下是一个涉及自然语言和人工神经网络的常见工作流程示例：

+   对所有文本进行分词和词干提取

+   从文本中移除停用词

+   确定适当的 ANN 输入层大小；使用此值既用于输入层也用于 Word2vec 的维度

+   使用 Word2vec 为你的文本生成词嵌入

+   使用词嵌入来训练 ANN 以完成你的任务

+   在评估新文档时，在将其传递给 ANN 之前对文档进行分词、词干提取和向量化

使用 Word2vec 等词嵌入算法不仅可以提高你模型的速度和内存性能，而且由于 Word2vec 算法保留的上下文信息，它可能还会提高你模型的准确性。还应注意的是，Word2vec 就像 n-gram 分词一样，是欺骗朴素词袋算法考虑词上下文的一种可能方式，因为 Word2vec 算法本身使用 n-gram 来开发嵌入。

虽然词嵌入主要在自然语言处理中使用，但同样的方法也可以用于其他领域，例如遗传学和生物化学。在这些领域中，有时能够将蛋白质或氨基酸序列向量化是有利的，这样相似的结构的向量嵌入也将相似。

# 摘要

自然语言处理是一个研究领域，拥有许多高级技术，并在机器学习、计算语言学和人工智能中有广泛的应用。然而，在本章中，我们专注于在日常工作任务中最普遍使用的特定工具和策略。

本章中介绍的技术是构建模块，可以混合搭配以实现许多不同的结果。仅使用本章中的信息，你可以构建一个简单的全文搜索引擎，一个用于语音或书面命令的意图提取器，一个文章摘要器，以及许多其他令人印象深刻的工具。然而，当这些技术与高级学习模型（如 ANNs 和 RNNs）结合时，NLP 的最令人印象深刻的应用才真正出现。

尤其是您学习了关于单词度量，如字符串距离和 TF-IDF 相关性评分；预处理和降维技术，如分词和词干提取；语音算法，如 Metaphone 算法；词性提取和短语解析；以及使用词嵌入算法将单词转换为向量。

您还通过众多示例介绍了两个优秀的 JavaScript 库，`natural.js`和`compromise.js`，这些库可以轻松完成与机器学习相关的多数 NLP 任务。您甚至能用 20 行代码编写一个文章摘要器！

在下一章中，我们将讨论如何将您迄今为止所学的一切整合到一个实时、面向用户的 JavaScript 应用程序中。
