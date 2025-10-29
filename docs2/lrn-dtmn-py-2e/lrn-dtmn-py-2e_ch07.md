# 第七章：使用图挖掘进行关注推荐

图可以用来表示广泛的现象。这尤其适用于在线社交网络和物联网（IoT）。图挖掘是商业的大事，例如 Facebook 这样的网站就是基于在图上进行的数据分析实验。

社交媒体网站建立在用户参与度之上。没有活跃新闻源或有趣的朋友关注，用户不会参与网站。相反，拥有更多有趣朋友和关注者的用户参与度更高，看到更多广告。这导致网站收入流增加。

在本章中，我们将探讨如何在图上定义相似性，以及如何在数据挖掘环境中使用它们。这同样基于现象模型。我们研究了一些基本的图概念，如子图和连通分量。这导致了对聚类分析的研究，我们将在第十章[lrn-dtmn-py-2e_ch10.html]《聚类新闻文章》中更深入地探讨。

本章涵盖的主题包括：

+   聚类数据以发现模式

+   从之前的实验中加载数据集

+   从 Twitter 获取关注者信息

+   创建图和网络

+   寻找用于聚类分析的子图

# 加载数据集

在本章中，我们的任务是推荐在线社交网络中的用户，基于共享连接。我们的逻辑是，如果两个用户有相同的关注者，他们非常相似，值得互相推荐。我们希望我们的推荐具有较高的价值。我们只能推荐这么多人，否则会变得乏味，因此我们需要找到能够吸引用户的推荐。

为了做到这一点，我们使用上一章的消歧模型来找到只谈论*Python 作为编程语言*的用户。在本章中，我们将一个数据挖掘实验的结果作为另一个数据挖掘实验的输入。一旦我们选定了 Python 程序员，我们就使用他们的友谊来找到高度相似的用户群。两个用户之间的相似度将由他们有多少共同朋友来定义。我们的直觉是，两个人共同的朋友越多，他们成为朋友的可能性就越大（因此应该在我们的社交媒体平台上）。

我们将使用上一章介绍过的 API 从 Twitter 创建一个小型社交图谱。我们寻找的数据是感兴趣于类似话题（再次强调，是 Python 编程语言）的用户子集以及他们所有朋友的列表（他们关注的人）。有了这些数据，我们将检查两个用户之间的相似度，基于他们有多少共同朋友。

除了 Twitter 之外，还有许多其他的在线社交网络。我们选择 Twitter 进行这个实验的原因是他们的 API 使得获取这类信息变得相当容易。信息也来自其他网站，如 Facebook、LinkedIn 和 Instagram 等。然而，获取这些信息要困难得多。

要开始收集数据，设置一个新的 Jupyter Notebook 和一个`twitter`连接实例，就像我们在上一章中所做的那样。你可以重用上一章中的应用信息或创建一个新的：

```py
import twitter
consumer_key = "<Your Consumer Key Here>"
consumer_secret = "<Your Consumer Secret Here>"
access_token = "<Your Access Token Here>"
access_token_secret = "<Your Access Token Secret Here>"
authorization = twitter.OAuth(access_token, 
access_token_secret, consumer_key, consumer_secret)
t = twitter.Twitter(auth=authorization, retry=True)

```

此外，设置文件名。你将想要为这个实验使用一个与你在第六章，“使用朴素贝叶斯进行社交媒体洞察”中使用的不同的文件夹，确保你不会覆盖你之前的数据集！

```py
import os 
data_folder = os.path.join(os.path.expanduser("~"), "Data", "twitter")
output_filename = os.path.join(data_folder, "python_tweets.json")

```

接下来，我们需要一个用户列表。我们将像上一章中所做的那样进行推文搜索，寻找提到单词`python`的推文。首先，创建两个列表来存储推文文本和相应的用户。我们稍后需要用户 ID，所以现在创建一个映射字典。代码如下：

```py
original_users = [] 
tweets = []
user_ids = {}

```

我们现在将执行对单词 python 的搜索，就像我们在上一章中所做的那样，遍历搜索结果，并只保存文本（按照上一章的要求）的推文：

```py
search_results = t.search.tweets(q="python", count=100)['statuses']
for tweet in search_results:
    if 'text' in tweet:
        original_users.append(tweet['user']['screen_name']) 
        user_ids[tweet['user']['screen_name']] = tweet['user']['id']
        tweets.append(tweet['text'])

```

运行此代码将获取大约 100 条推文，在某些情况下可能稍微少一些。尽管如此，并非所有这些推文都与编程语言相关。我们将通过使用上一章中训练的模型来解决这个问题。

# 使用现有模型进行分类

正如我们在上一章中学到的，提到单词 python 的推文并不一定都与编程语言相关。为了做到这一点，我们将使用上一章中使用的分类器来获取基于编程语言的推文。我们的分类器并不完美，但它将比仅仅进行搜索的结果有更好的专业化。

在这个例子中，我们只对那些在推文中提到 Python 编程语言的用户感兴趣。我们将使用上一章中的分类器来确定哪些推文与编程语言相关。从那里，我们只选择那些提到编程语言的用户。

为了进行我们更广泛的实验的这一部分，我们首先需要保存上一章中的模型。打开我们在上一章中制作的 Jupyter Notebook，即我们构建和训练分类器的那个。

如果你已经关闭了它，那么 Jupyter Notebook 将不会记住你所做的一切，你需要再次运行这些单元格。要做到这一点，点击笔记本上的单元格菜单并选择运行所有。

在所有单元格都计算完毕后，选择最终的空白单元格。如果你的笔记本在末尾没有空白单元格，请选择最后一个单元格，选择插入菜单，然后选择插入单元格下方选项。

我们将使用`joblib`库来保存我们的模型并加载它。

`joblib`是`scikit-learn`包内置的外部包之一。无需额外安装步骤！这个库提供了保存和加载模型以及简单并行处理（这在`scikit-learn`中相当常用）的工具。

首先，导入库并为我们的模型创建一个输出文件名（确保目录存在，否则它们不会被创建）。我已经将这个模型存储在我的`Models`目录中，但您可以选择将它们存储在其他位置。代码如下：

```py
from sklearn.externals import joblib
output_filename = os.path.join(os.path.expanduser("~"), "Models", "twitter", "python_context.pkl")

```

接下来，我们使用`joblib`库中的`dump`函数，它的工作方式与`json`库中同名版本类似。我们传递模型本身和输出文件名：

```py
joblib.dump(model, output_filename)

```

运行此代码将把我们的模型保存到指定的文件名。接下来，回到您在上一个子节中创建的新 Jupyter 笔记本，并加载此模型。

在这个笔记本中，您需要再次设置模型的文件名，方法是将以下代码复制过来：

```py
model_filename = os.path.join(os.path.expanduser("~"), "Models", "twitter", "python_context.pkl")

```

确保文件名是您刚才保存模型时使用的那个。接下来，我们需要重新创建我们的 BagOfWords 类，因为它是一个自定义构建的类，不能直接由 joblib 加载。只需从上一章的代码中复制整个 BagOfWords 类及其依赖项：

```py
import spacy
from sklearn.base import TransformerMixin

# Create a spaCy parser
nlp = spacy.load('en')

class BagOfWords(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        results = []
        for document in X:
            row = {}
            for word in list(nlp(document, tag=False, parse=False, entity=False)):
                if len(word.text.strip()): # Ignore words that are just whitespace
                    row[word.text] = True
                    results.append(row)
        return results

```

在生产环境中，您需要单独开发自己的自定义转换器，并将它们导入到笔记本中。这个小技巧简化了工作流程，但您可以自由地通过创建一个通用功能库来集中化重要代码进行实验。

现在加载模型只需要调用`joblib`的`load`函数：

```py
from sklearn.externals import joblib
context_classifier = joblib.load(model_filename)

```

我们的 context_classifier 与我们在第六章中看到的笔记本中的模型对象工作方式完全相同，即《使用朴素贝叶斯进行社交媒体洞察》。它是一个 Pipeline 的实例，与之前的三个步骤相同（`BagOfWords`、`DictVectorizer`和`BernoulliNB`分类器）。在这个模型上调用 predict 函数会给出我们的推文是否与编程语言相关的预测。代码如下：

```py
y_pred = context_classifier.predict(tweets)

```

`y_pred`中的第*i*个项如果是（预测为）与编程语言相关的推文，则为 1，否则为 0。从这里，我们可以获取相关推文及其相关用户：

```py
relevant_tweets = [tweets[i] for i in range(len(tweets)) if y_pred[i] == 1]
relevant_users = [original_users[i] for i in range(len(tweets)) if y_pred[i] == 1]

```

使用我的数据，这对应着 46 个相关用户。比之前的 100 条推文/用户略低，但现在我们有了一个构建社交网络的基础。我们总是可以添加更多数据以获取更多用户，但 40+个用户足以作为第一次遍历通过这一章节。我建议回来，添加更多数据，并再次运行代码，看看您获得什么结果。

# 从 Twitter 获取关注者信息

现在我们需要获取我们初始用户集的每个用户的联系人。联系人是指用户正在关注的人。这个 API 叫做 friends/ids，它既有优点也有缺点。好消息是它可以在单个 API 调用中返回多达 5,000 个朋友 ID。坏消息是每 15 分钟你只能调用 15 次，这意味着你至少需要花 1 分钟每个用户来获取所有后继者——如果他们有超过 5,000 个朋友（这比你想的更常见）。

代码与之前 API 使用（获取推文）的代码类似。我们将将其打包成一个函数，因为我们将在下一两个部分中使用这段代码。我们的函数接受一个 Twitter 用户的 ID 值，并返回他们的联系人。虽然对一些人来说可能很惊讶，但许多 Twitter 用户有超过 5,000 个朋友。因此，我们需要使用 Twitter 的分页功能，这允许 Twitter 通过单独的 API 调用返回多页数据。当你向 Twitter 请求信息时，它会给你你的信息以及一个光标，这是一个 Twitter 用来跟踪你的请求的整数。如果没有更多信息，这个光标是 0；否则，你可以使用提供的光标来获取下一页的结果。传递这个光标让 Twitter 继续你的查询，返回下一组数据给你。

在函数中，我们保持循环直到这个光标不等于 0（因为，当它等于时，就没有更多数据可以收集了）。然后我们向用户的后继者发起请求并将他们添加到我们的列表中。我们这样做是在一个 try 块中，因为可能发生一些我们可以处理的错误。后继者的 ID 存储在结果字典的 ids 键中。在获取到这些信息后，我们更新光标。它将在循环的下一迭代中使用。最后，我们检查我们是否有超过 10,000 个朋友。如果是这样，我们就跳出循环。代码如下：

```py
import time

def get_friends(t, user_id):
    friends = []
    cursor = -1
    while cursor != 0: 
        try:
            results = t.friends.ids(user_id= user_id, cursor=cursor, count=5000)
            friends.extend([friend for friend in results['ids']])
            cursor = results['next_cursor'] 
            if len(friends) >= 10000:
                break
        except TypeError as e:
            if results is None:
                print("You probably reached your API limit, waiting for 5 minutes")
                sys.stdout.flush() 
                time.sleep(5*60) # 5 minute wait 
            else: 
                # Some other error happened, so raise the error as normal
                raise e
        except twitter.TwitterHTTPError as e:
            print(e)
            break
        finally:
            # Break regardless -- this stops us going over our API limit
            time.sleep(60)

```

在这里插入一个警告是值得的。我们正在处理来自互联网的数据，这意味着奇怪的事情会经常发生。我在开发这段代码时遇到的一个问题是，一些用户有很多很多很多的朋友。为了解决这个问题，我们在这里设置一个安全退出机制，当我们达到超过 10,000 个用户时退出函数。如果你想收集完整的数据集，你可以删除这些行，但请注意，它可能会在某个特定的用户上长时间卡住。

上述函数的大部分是错误处理，因为在处理外部 API 时可能会出现很多问题！

最可能发生的错误是我们意外达到 API 限制（虽然我们有一个睡眠来停止它，但在你停止并运行代码之前，它可能发生）。在这种情况下，结果为`None`，我们的代码将因`TypeError`而失败。在这种情况下，我们等待 5 分钟并再次尝试，希望我们已经到达下一个 15 分钟窗口。这时可能还会发生另一个`TypeError`。如果其中一个发生了，我们将抛出它，并需要单独处理。

第二种可能发生的错误是在 Twitter 端发生的，比如请求一个不存在的用户或其他基于数据的错误，导致`TwitterHTTPError`（这与 HTTP 404 错误的概念类似）。在这种情况下，不要再尝试这个用户，只需返回我们确实获取到的任何关注者（在这种情况下，可能为 0）。

最后，Twitter 只允许我们每 15 分钟请求 15 次关注者信息，因此我们将在继续之前等待 1 分钟。我们这样做是为了在发生错误的情况下也能执行。

# 构建网络

现在我们将构建我们的用户网络，其中如果两个用户相互关注，则用户之间有联系。构建这个网络的目标是给我们一个可以用来将用户列表分割成组的数据结构。从这些组中，我们然后可以向同一组的人推荐其他人。从我们的原始用户开始，我们将获取每个用户的联系人并存储在字典中。使用这个概念，我们可以从一组初始用户向外扩展图。

从我们的原始用户开始，我们将获取每个用户的联系人并将它们存储在字典中（在从我们的`*user_id*`字典中获取用户的 ID 之后）：

```py
friends = {} 
for screen_name in relevant_users:
    user_id = user_ids[screen_name]
    friends[user_id] = get_friends(t, user_id)

```

接下来，我们将删除任何没有联系人的用户。对于这些用户，我们实际上无法以这种方式做出推荐。相反，我们可能需要查看他们的内容或关注他们的人。不过，我们将把这一点排除在本章的范围之外，所以我们只需删除这些用户。代码如下：

```py
friends = {user_id:friends[user_id] 
           for user_id in friends
           if len(friends[user_id]) > 0}

```

现在我们有 30 到 50 个用户，具体取决于你的初始搜索结果。我们现在将这个数量增加到 150。以下代码将需要相当长的时间来运行——考虑到 API 的限制，我们每分钟只能获取一个用户的联系人。简单的数学告诉我们，150 个用户将需要 150 分钟，这至少是 2 小时 30 分钟。考虑到我们将花费在获取这些数据上的时间，确保我们只获取好的用户是值得的。

那么，什么样的用户才算好呢？鉴于我们将基于共享联系来做出推荐，我们将根据共享联系来搜索用户。我们将获取现有用户的联系人，从那些与现有用户联系更紧密的用户开始。为此，我们维护一个计数，记录用户出现在我们朋友的列表中的所有次数。在考虑你的采样策略时，考虑应用程序的目标是值得考虑的。为此目的，获取大量类似用户可以使推荐更加适用。

要做到这一点，我们只需遍历我们所有的朋友列表，然后计算每个朋友出现的次数。

```py
from collections import defaultdict
def count_friends(friends): 
    friend_count = defaultdict(int)
    for friend_list in friends.values(): 
        for friend in friend_list:
            friend_count[friend] += 1 
    return friend_count

```

计算我们当前的朋友数量，然后我们可以从我们的样本中获取最连接的人（即现有列表中最多的朋友）。代码如下：

```py
friend_count = count_friends(friends)
from operator import itemgetter
best_friends = sorted(friend_count, key=friend_count.get, reverse=True)

```

从这里，我们设置一个循环，直到我们有了 150 个用户的联系人。然后我们遍历我们最好的朋友（这是按照拥有他们作为朋友的人数顺序发生的）直到我们找到一个我们尚未检查的用户。然后我们获取该用户的联系人并更新`friends`计数。最后，我们找出我们列表中尚未出现的最连接的用户：

```py
while len(friends) < 150:
    for user_id, count in best_friends:
        if user_id in friends:
            # Already have this user, move to next one
            continue
        friends[user_id] = get_friends(t, user_id) 
        for friend in friends[user_id]: 
            friend_count[friend] += 1
        best_friends = sorted(friend_count.items(), key=itemgetter(1), reverse=True)
        break

```

然后，代码将循环并继续，直到我们达到 150 个用户。

你可能想要将这些值设置得更低，比如 40 或 50 个用户（或者甚至暂时跳过这段代码）。然后，完成本章的代码，感受一下结果是如何工作的。之后，将此循环中的用户数量重置为 150，让代码运行几小时，然后回来重新运行后面的代码。

由于收集这些数据可能花费了近 3 个小时，因此将其保存下来是个好主意，以防我们不得不关闭电脑。使用`json`库，我们可以轻松地将我们的朋友字典保存到文件中：

```py
import json
friends_filename = os.path.join(data_folder, "python_friends.json")
with open(friends_filename, 'w') as outf: 
    json.dump(friends, outf)

```

如果需要加载文件，请使用`json.load`函数：

```py
with open(friends_filename) as inf:
    friends = json.load(inf)

```

# 创建一个图

在我们的实验的这个阶段，我们有一个用户及其联系人的列表。这给我们一个图，其中一些用户是其他用户的联系人（尽管不一定反过来）。

**图**是一组节点和边。节点通常是感兴趣的物体——在这种情况下，它们是我们的用户。这个初始图中的边表示用户 A 是用户 B 的朋友。我们称之为**有向图**，因为节点的顺序很重要。仅仅因为用户 A 是用户 B 的朋友，并不意味着用户 B 也是用户 A 的朋友。下面的示例网络展示了这一点，以及一个与用户 B 是朋友并且反过来也被用户 B 添加为朋友的用户 C：

![图片](img/B06162_07_01.png)

在 Python 中，用于处理图（包括创建、可视化和计算）的最佳库之一是**NetworkX**。

再次强调，你可以使用 Anaconda 安装 NetworkX：`conda install networkx`

首先，我们使用 NetworkX 创建一个有向图。按照惯例，在导入 NetworkX 时，我们使用缩写 nx（尽管这并不是必需的）。代码如下：

```py
import networkx as nx 
G = nx.DiGraph()

```

我们将只可视化我们的关键用户，而不是所有朋友（因为有很多这样的朋友，而且很难可视化）。我们获取主要用户，然后将它们作为节点添加到我们的图中：

```py
main_users = friends.keys() 
G.add_nodes_from(main_users)

```

接下来我们设置边。如果第二个用户是第一个用户的朋友，我们就从用户到另一个用户创建一条边。为此，我们遍历给定用户的全部朋友。我们确保这个朋友是我们主要用户之一（因为我们目前不感兴趣可视化其他用户），如果他们是，就添加这条边。

```py
for user_id in friends:
    for friend in friends[user_id]:
        if str(friend) in main_users: 
            G.add_edge(user_id, friend) 

```

现在我们可以使用 NetworkX 的 draw 函数来可视化网络，该函数使用 matplotlib。为了在我们的笔记本中获得图像，我们使用 matplotlib 的 inline 函数，然后调用 draw 函数。代码如下：

```py
 %matplotlib inline 
 nx.draw(G)

```

结果有点难以理解；它们只显示了节点环，很难从数据集中得出具体的东西。根本不是一张好图：

![](img/B06162_07_02.png)

我们可以使用 pyplot 来处理图形的创建，这是 NetworkX 进行图形绘制所使用的。导入`pyplot`，创建一个更大的图形，然后调用 NetworkX 的`draw`函数来增加图像的大小：

```py
from matplotlib import pyplot as plt
plt.figure(3,figsize=(20,20))
nx.draw(G, alpha=0.1, edge_color='b')

```

通过放大图并增加透明度，现在可以清楚地看到图的轮廓：

![](img/B06162_07_03.png)

在我的图中，有一个主要用户群，他们之间高度连接，而大多数其他用户几乎没有连接。正如你所见，它们在中心非常连接！

这实际上是我们选择新用户的方法的一个特性——我们选择那些在我们图中已经很好地连接的用户，因此他们很可能只是使这个群体更大。对于社交网络，一般来说，用户拥有的连接数遵循幂律。一小部分用户有很多连接，而其他人只有几个。图的形状通常描述为具有*长尾*。

通过放大图的某些部分，你可以开始看到结构。可视化和分析这样的图很困难 - 我们将在下一节中看到一些使这个过程更容易的工具。

# 创建相似性图

实验的最终步骤是根据用户分享的朋友数量来推荐用户。如前所述，我们的逻辑是，如果两个用户有相同的朋友，他们非常相似。基于这一点，我们可以向一个用户推荐另一个用户。

因此，我们将使用现有的图（其中包含与友谊相关的边）并从其信息中创建一个新的图。节点仍然是用户，但边将是**加权边**。加权边只是一个具有权重属性的边。逻辑是，较高的权重表示两个节点之间的相似性高于较低的权重。这取决于上下文。如果权重代表距离，则较低的权重表示更多的相似性。

对于我们的应用，权重将是连接该边的两个用户的相似度（基于他们共享的朋友数量）。这个图也具有不定向的性质。这归因于我们的相似度计算，其中用户 A 对用户 B 的相似度与用户 B 对用户 A 的相似度相同。

其他相似度度量是定向的。一个例子是相似用户的比率，即共同朋友数量除以用户的总朋友数量。在这种情况下，你需要一个有向图。

计算两个类似列表之间的相似性有很多种方法。例如，我们可以计算两人共同拥有的朋友数量。然而，这个度量对于拥有更多朋友的人来说总是会更高。相反，我们可以通过除以两人拥有的不同朋友的总量来归一化它。这被称为**Jaccard 相似度**。

Jaccard 相似度，总是在 0 到 1 之间，表示两个之间的重叠百分比。正如我们在第二章中看到的，*使用 scikit-learn 估计器进行分类*，归一化是数据挖掘练习的重要部分，通常是一个好习惯。有些边缘情况你可能不会对数据进行归一化，但默认情况下首先进行归一化。

要计算 Jaccard 相似度，我们将两个粉丝集合的交集除以它们的并集。这些是集合操作，而我们手头有列表，因此我们需要先将朋友列表转换为集合。代码如下：

```py
friends = {user: set(friends[user]) for user in friends}

```

然后我们创建一个函数来计算两个朋友列表集合的相似度。代码如下：

```py
def compute_similarity(friends1, friends2):
    return len(friends1 & friends2) / (len(friends1 | friends2)  + 1e-6)

```

我们在相似度上添加 1e-6（或 0.000001）以确保我们永远不会遇到除以零的错误，在两个用户都没有朋友的情况下。它足够小，不会真正影响我们的结果，但足够大，以超过零。

从这里，我们可以创建用户之间的加权相似度图。在接下来的章节中，我们将大量使用这个图，因此我们将创建一个函数来执行此操作。让我们看看阈值参数：

```py
def create_graph(followers, threshold=0): 
    G = nx.Graph()
    for user1 in friends.keys(): 
        for user2 in friends.keys(): 
            if user1 == user2:
                continue
            weight = compute_similarity(friends[user1], friends[user2])
            if weight >= threshold:
                G.add_node(user1) 
                G.add_node(user2)
                G.add_edge(user1, user2, weight=weight)
    return G

```

我们现在可以通过调用此函数来创建一个图。我们从一个没有阈值的开始，这意味着所有链接都被创建。代码如下：

```py
G = create_graph(friends)

```

结果是一个非常紧密连接的图——所有节点都有边，尽管其中许多边的权重为 0。我们将通过绘制具有与边权重成比例的线宽的图来看到边的权重——较粗的线表示较高的权重。

由于节点数量较多，将图放大以获得更清晰的连接感是有意义的：

```py
plt.figure(figsize=(10,10))

```

我们将带有权重的边绘制出来，因此需要先绘制节点。NetworkX 使用布局来确定节点和边的位置，基于某些标准。可视化网络是一个非常困难的问题，尤其是随着节点数量的增加。存在各种用于可视化网络的技巧，但它们的工作程度在很大程度上取决于你的数据集、个人偏好和可视化的目标。我发现`spring_layout`工作得相当好，但其他选项，如`circular_layout`（如果没有其他选项可用，这是一个很好的默认选项）、`random_layout`、`shell_layout`和`spectral_layout`也存在，并且在这些其他选项失败的地方有它们的应用。

访问[`networkx.lanl.gov/reference/drawing.html`](http://networkx.lanl.gov/reference/drawing.html)获取 NetworkX 中布局的更多详细信息。尽管它增加了一些复杂性，但`draw_graphviz`选项工作得相当好，值得调查以获得更好的可视化效果。在现实世界的应用中，这非常值得考虑。

让我们使用`spring_layout`进行可视化：

```py
pos = nx.spring_layout(G)

```

使用我们的`pos`布局，我们可以然后定位节点：

```py
nx.draw_networkx_nodes(G, pos)

```

接下来，我们绘制边。为了获取权重，我们遍历图中的边（按特定顺序）并收集权重：

```py
edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]

```

我们接下来绘制边：

```py
nx.draw_networkx_edges(G, pos, width=edgewidth)

```

结果将取决于你的数据，但通常将显示一个具有大量节点且连接相当紧密的图，以及一些与其他网络连接较差的节点。

![图片](img/B06162_07_04.png)

与前一个图相比，这个图的不同之处在于，边是根据我们的相似性度量来决定节点之间的相似性，而不是根据一个人是否是另一个人的朋友（尽管两者之间有相似之处！）。现在我们可以从这个图中提取信息，以便做出我们的推荐。

# 寻找子图

从我们的相似性函数中，我们可以简单地为每个用户对结果进行排名，返回最相似的用户作为推荐——就像我们处理产品推荐那样。这行得通，确实是执行此类分析的一种方式。

相反，我们可能希望找到所有用户都彼此相似的用户群。我们可以建议这些用户开始组建一个群组，为这个细分市场创建广告，或者甚至只是使用这些集群本身来进行推荐。找到这些相似用户群的任务被称为**聚类分析**。

聚类分析是一项困难的任务，具有分类任务通常不具备的复杂性。例如，评估分类结果相对容易——我们将我们的结果与真实情况（来自我们的训练集）进行比较，并查看我们正确了多少百分比。然而，在聚类分析中，通常没有真实情况。评估通常归结为根据我们对聚类应该看起来怎样的先入为主的观念，来判断聚类是否合理。

聚类分析另一个复杂之处在于，模型不能针对预期结果进行训练以学习——它必须基于聚类数学模型进行一些近似，而不是用户希望通过分析实现的目标。

由于这些问题，聚类分析更倾向于一种探索性工具，而不是预测工具。一些研究和应用使用聚类进行数据分析，但其作为预测模型的有用性取决于分析师选择参数并找到看起来“正确”的图，而不是特定的评估指标。

# 连接分量

聚类最简单的方法之一是在图中找到**连接分量**。连接分量是图中通过边连接的节点集合。并非所有节点都需要相互连接才能成为连接分量。然而，为了使两个节点处于同一个连接分量中，必须存在一种方法，可以通过沿着边从节点移动到另一个节点。

连接分量在计算时不会考虑边权重；它们只检查边的存在。因此，接下来的代码将移除任何权重低的边。

NetworkX 有一个用于计算连接分量的函数，我们可以调用我们的图。首先，我们使用我们的`create_graph`函数创建一个新的图，但这次我们传递一个阈值为 0.1，以仅获取权重至少为 0.1 的边，这表明两个节点用户之间有 10%的共同关注者：

```py
G = create_graph(friends, 0.1)

```

然后，我们使用 NetworkX 在图中找到连接分量：

```py
sub_graphs = nx.connected_component_subgraphs(G)

```

为了了解图的大小，我们可以遍历组并打印一些基本信息：

```py
for i, sub_graph in enumerate(sub_graphs):
    n_nodes = len(sub_graph.nodes()) 
    print("Subgraph {0} has {1} nodes".format(i, n_nodes))

```

结果将告诉你每个连接分量有多大。我的结果有一个包含 62 个用户的大子图和许多小子图，每个子图有十几个或更少的用户。

我们可以改变**阈值**来改变连接分量。这是因为更高的阈值连接节点的边更少，因此将具有更小的连接分量和更多的连接分量。我们可以通过运行前面的代码并使用更高的阈值来看到这一点：

```py
G = create_graph(friends, 0.25) 
sub_graphs = nx.connected_component_subgraphs(G) 
for i, sub_graph in enumerate(sub_graphs): 
    n_nodes = len(sub_graph.nodes()) 
    print("Subgraph {0} has {1} nodes".format(i, n_nodes))

```

上述代码给出了更小的子图和更多的子图。我最大的簇被分割成至少三个部分，而且没有任何一个簇的用户数量超过 10 个。以下图显示了簇的一个示例，以及该簇内的连接。请注意，由于这是一个连通组件，该组件中的节点到图中其他节点的边不存在（至少，当阈值设置为 0.25 时）。

我们可以绘制整个图，用不同的颜色显示每个连通组件。由于这些连通组件彼此不相连，将它们绘制在单个图上实际上没有太多意义。这是因为节点和组件的位置是任意的，可能会使可视化变得混乱。相反，我们可以将每个组件分别绘制在单独的子图中。

在一个新的单元中，获取连通组件以及连通组件的数量：

```py
sub_graphs = nx.connected_component_subgraphs(G) 
n_subgraphs = nx.number_connected_components(G)

```

`sub_graphs`是一个生成器，而不是连通组件的列表。因此，使用`nx.number_connected_components`来找出有多少连通组件；不要使用`len`，因为它由于 NetworkX 存储信息的方式而不起作用。这就是为什么我们需要在这里重新计算连通组件。

创建一个新的 pyplot 图，并留出足够的空间来显示所有的连通组件。因此，我们允许图的大小随着连通组件数量的增加而增加。

接下来，遍历每个连通组件，并为每个组件添加一个子图。`add_subplot`的参数是子图行的数量、列的数量以及我们感兴趣的子图的索引。我的可视化使用三列，但你可以尝试其他值而不是三（只是记得要更改两个值）：

```py
fig = plt.figure(figsize=(20, (n_subgraphs * 3)))
for i, sub_graph in enumerate(sub_graphs): 
    ax = fig.add_subplot(int(n_subgraphs / 3) + 1, 3, i + 1)
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False)
    pos = nx.spring_layout(G) 
    nx.draw_networkx_nodes(G, pos, sub_graph.nodes(), ax=ax, node_size=500) 
    nx.draw_networkx_edges(G, pos, sub_graph.edges(), ax=ax)

```

结果可视化每个连通组件，让我们对每个组件中的节点数量以及它们的连接程度有一个概念。

![图片](img/B06162_07_05.png)

如果你没有在你的图上看到任何东西，请尝试重新运行以下行：

`sub_graphs = nx.connected_component_subgraphs(G)`

`sub_graphs`对象是一个生成器，并且在被使用后会被“消耗”。

# 优化标准

我们找到这些连通组件的算法依赖于**阈值**参数，该参数决定了是否将边添加到图中。反过来，这直接决定了我们发现的连通组件的数量和大小。从这里，我们可能想要确定一个关于哪个是*最佳*阈值的概念。这是一个非常主观的问题，没有明确的答案。这是任何聚类分析任务的一个主要问题。

然而，我们可以确定我们认为一个好的解决方案应该是什么样子，并基于这个想法定义一个度量标准。作为一个一般规则，我们通常希望解决方案是这样的：

+   同一簇（连通组件）中的样本彼此之间高度*相似*。

+   不同簇中的样本彼此之间高度*不同*。

**轮廓系数**是一个量化这些点的度量标准。给定一个单个样本，我们定义轮廓系数如下：

![](img/B06162_07_06.png)

其中 *a* 是 **簇内距离** 或样本簇中其他样本的平均距离，而 <q>b</q> 是 **簇间距离** 或到下一个最近邻簇中其他样本的平均距离。

为了计算整体轮廓系数，我们取每个样本的轮廓系数的平均值。提供接近最大值 1 的轮廓系数的聚类具有所有样本都相似的簇，并且这些簇分布得很广。接近 0 的值表明簇全部重叠，簇之间几乎没有区别。接近最小值-1 的值表明样本可能位于错误的簇中，也就是说，它们在其他簇中会更好。

使用这个度量标准，我们希望找到一个解决方案（即阈值的一个值），通过改变阈值参数来最大化轮廓系数。为此，我们创建了一个函数，该函数将阈值作为参数并计算轮廓系数。

然后我们将这个函数传递给 SciPy 的**优化**模块，其中包含用于通过改变参数来找到函数最小值的`minimize`函数。虽然我们感兴趣的是最大化轮廓系数，但 SciPy 没有最大化函数。相反，我们最小化轮廓系数的倒数（这基本上是同一件事）。

scikit-learn 库有一个用于计算轮廓系数的函数，`sklearn.metrics.silhouette_score`；然而，它没有修复 SciPy 最小化函数所需的函数格式。最小化函数需要将变量参数放在第一位（在我们的情况下，是阈值值），并将任何参数放在其后。在我们的情况下，我们需要将好友字典作为参数传递，以便计算图。

轮廓系数在至少有两个节点（为了计算距离）的情况下才定义。在这种情况下，我们将问题范围定义为无效。有几种处理方法，但最简单的是返回一个非常差的分数。在我们的情况下，轮廓系数可以取的最小值是-1，我们将返回-99 来表示无效问题。任何有效解决方案的得分都将高于这个值。

下面的函数结合了所有这些问题，它提供了一个接受阈值值和好友列表作为参数的函数，并计算轮廓系数。它是通过使用 NetworkX 的`to_scipy_sparse_matrix`函数从图中构建矩阵来实现的。

```py
import numpy as np
from sklearn.metrics import silhouette_score

def compute_silhouette(threshold, friends):
    G = create_graph(friends, threshold=threshold) 
    if len(G.nodes()) < 2:
        return -99
    sub_graphs = nx.connected_component_subgraphs(G)

    if not (2 <= nx.number_connected_components() < len(G.nodes()) - 1): 
        return -99

    label_dict = {}
    for i, sub_graph in enumerate(sub_graphs): 
        for node in sub_graph.nodes(): 
            label_dict[node] = i

    labels = np.array([label_dict[node] for node in G.nodes()])
    X = nx.to_scipy_sparse_matrix(G).todense()
    X = 1 - X
    return silhouette_score(X, labels, metric='precomputed')

```

对于评估稀疏数据集，我建议您查看 V-Measure 或调整互信息。这两个都在 scikit-learn 中实现，但它们在执行评估时具有非常不同的参数。

在编写本文时，scikit-learn 中的轮廓系数实现不支持稀疏矩阵。因此，我们需要调用`todense`函数。通常，这不是一个好主意——稀疏矩阵通常用于数据通常不应该以密集格式存在的情况。在这种情况下，这将是可行的，因为我们的数据集相对较小；然而，不要尝试在更大的数据集上这样做。

这里发生了两种形式的逆运算。第一种是取相似度的逆来计算距离函数；这是必需的，因为轮廓系数只接受距离。第二种是将轮廓系数分数取逆，这样我们就可以使用 SciPy 的优化模块进行最小化。

最后，我们创建我们将要最小化的函数。这个函数是`compute_silhouette`函数的逆，因为我们希望较低的分数更好。我们可以在我们的`compute_silhouette`函数中这样做——我已经在这里将它们分开，以阐明涉及的不同步骤。

```py
def inverted_silhouette(threshold, friends):
    return -compute_silhouette(threshold, friends)

```

这个函数从一个原始函数创建一个新的函数。当调用新函数时，所有相同的参数和关键字都会传递给原始函数，并返回返回值，只是在返回之前取了相反数。

现在我们可以进行实际的优化了。我们调用我们定义的逆`compute_silhouette`函数上的最小化函数：

```py
from scipy.optimize import minimize
result = minimize(inverted_silhouette, 0.1, args=(friends,))

```

这个函数运行起来会花费相当长的时间。我们的图形创建函数并不快，计算轮廓系数的函数也不快。减小`maxiter`参数的值会导致迭代次数减少，但我们面临找到次优解的风险。

运行这个函数，我得到了一个阈值为 0.135，返回了 10 个组件。最小化函数返回的分数是-0.192。然而，我们必须记住我们取了这个值的相反数。这意味着我们的分数实际上是 0.192。这个值是正的，这表明簇倾向于比不分离得更好（这是好事）。我们可以运行其他模型并检查它是否会产生更好的分数，这意味着簇被更好地分离了。

我们可以使用这个结果来推荐用户——如果一个用户在特定的连通组件中，那么我们可以推荐该组件中的其他用户。这种推荐遵循我们使用 Jaccard 相似度来找到用户之间良好连接的做法，我们使用连通组件将它们分成簇，以及我们使用优化技术来找到在这种设置下的最佳模型。

然而，大量的用户可能根本不连通，因此我们将使用不同的算法为他们找到簇。我们将在第十章聚类新闻文章*中看到其他聚类分析方法。

# 摘要

在本章中，我们探讨了社交网络中的图以及如何对它们进行聚类分析。我们还探讨了如何使用我们在第六章中创建的分类模型通过 scikit-learn 来保存和加载模型，*《使用朴素贝叶斯进行社交媒体洞察》*。

我们创建了一个来自社交网络 Twitter 的朋友关系图。然后，我们根据他们的朋友来考察两个用户之间的相似性。拥有更多共同朋友的用户被认为更相似，尽管我们通过考虑他们拥有的总朋友数来对此进行归一化。这是一种常用的方法，可以根据相似用户推断知识（如年龄或一般讨论主题）。我们可以使用这种逻辑来向其他人推荐用户——如果他们关注用户 X，而用户 Y 与用户 X 相似，他们可能会喜欢用户 Y。这在许多方面与之前章节中提到的基于交易的相似性相似。

本分析的目标是推荐用户，而我们使用聚类分析使我们能够找到相似用户的聚类。为此，我们在基于此相似性度量创建的加权图中找到了连通组件。我们使用 NetworkX 包来创建图、使用我们的图以及找到这些连通组件。

然后，我们使用了轮廓系数，这是一个评估聚类解决方案好坏的指标。根据簇内和簇间距离的概念，更高的分数表示更好的聚类。SciPy 的优化模块被用来找到最大化这个值的解。

在本章中，我们看到了一些对立的概念在行动中的表现。相似性是两个对象之间的度量，其中更高的值表示这些对象之间有更多的相似性。相反，距离是一个度量，其中较低的值表示更多的相似性。我们看到的另一个对比是损失函数，其中较低的分数被认为是更好的（也就是说，我们损失更少）。它的对立面是得分函数，其中较高的分数被认为是更好的。

为了扩展本章的工作，检查 scikit-learn 中的 V-measure 和调整互信息分数。这些取代了本章中使用的轮廓系数。最大化这些指标得到的聚类是否比轮廓系数的聚类更好？进一步地，如何判断？通常，聚类分析的问题在于你无法客观地判断，可能需要人工干预来选择最佳选项。

在下一章中，我们将看到如何从另一种新型数据——图像中提取特征。我们将讨论如何使用神经网络来识别图像中的数字，并开发一个程序来自动击败验证码图像。
