# 第八章：使用卷积神经网络进行图像分类

在这一章中，我们将探索计算机视觉的广阔而精彩的世界。

如果你曾经想过使用图像数据构建一个预测性的机器学习模型，本章将作为一个易于消化且实用的资源。我们将一步一步地构建一个图像分类模型，对其进行交叉验证，然后以更好的方式构建它。在本章的结尾，我们将有一个*相当不错*的模型，并讨论一些未来增强的路径。

当然，了解一些预测建模的基础将有助于让这一过程顺利进行。正如你很快就会看到的那样，将图像转换为可用于我们模型的特征的过程可能会让人觉得是新鲜的，但一旦提取了特征，模型构建和交叉验证的过程就完全一样。

在本章中，我们将构建一个卷积神经网络，用于对 Zalando Research 数据集中的服装图像进行分类——该数据集包含 70,000 张图像，每张图像展示了 10 种可能的服装类别之一，例如 T 恤/上衣、裤子、毛衣、连衣裙、外套、凉鞋、衬衫、运动鞋、包或短靴。但首先，我们将一起探索一些基础知识，从图像特征提取开始，并逐步了解卷积神经网络的工作原理。

那么，让我们开始吧。真的！

本章将涵盖以下内容：

+   图像特征提取

+   卷积神经网络：

    +   网络拓扑

    +   卷积层与滤波器

    +   最大池化层

    +   展平

    +   全连接层与输出

+   使用 Keras 构建卷积神经网络来对 Zalando Research 数据集中的图像进行分类

# 图像特征提取

在处理非结构化数据时，无论是文本还是图像，我们必须首先将数据转换为机器学习模型可以使用的数字表示。将非数字数据转换为数字表示的过程称为**特征提取**。对于图像数据来说，我们的特征就是图像的像素值。

首先，假设我们有一张 1,150 x 1,150 像素的灰度图像。这样的一张图像将返回一个 1,150 x 1,150 的像素强度矩阵。对于灰度图像，像素值的范围是从 0 到 255，其中 0 代表完全黑色的像素，255 代表完全白色的像素，而 0 到 255 之间的值则代表不同的灰色阴影。

为了展示这在代码中的表现，我们来提取我们灰度猫卷饼图像的特征。该图像可以在 GitHub 上找到，链接是 [`github.com/PacktPublishing/Python-Machine-Learning-Blueprints-Second-Edition/tree/master/Chapter08`](https://github.com/PacktPublishing/Python-Machine-Learning-Blueprints-Second-Edition/tree/master/Chapter08)，文件名是`grayscale_cat_burrito.jpg`。

我已经将本章中使用的图像资源提供给你，链接为[`github.com/mroman09/packt-image-assets`](https://github.com/mroman09/packt-image-assets)。你可以在那里找到我们的猫肉卷！

现在，我们来看一下以下代码中的一个示例：

```py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

cat_burrito = mpimg.imread('images/grayscale_cat_burrito.jpg')
cat_burrito
```

如果你无法通过运行前面的代码读取`.jpg`文件，只需运行`pip install pillow`安装`PIL`。

在前面的代码中，我们导入了`pandas`和两个子模块：`image`和`pyplot`，来自`matplotlib`。我们使用了`matplotlib.image`中的`imread`方法来读取图像。

运行前面的代码会得到以下输出：

![](img/cd46ced0-2f96-410d-9801-0f7aa0f1fb90.png)

输出是一个二维的`numpy` ndarray，包含了我们模型的特征。像大多数应用机器学习的场景一样，您可能需要对这些提取的特征执行若干预处理步骤，其中一些我们将在本章稍后与 Zalando 时尚数据集一起探讨，但这些就是图像的原始提取特征！

提取的灰度图像特征的形状为`image_height`行 × `image_width`列。我们可以通过运行以下代码轻松检查图像的形状：

```py
cat_burrito.shape
```

前面的代码返回了以下输出：

![](img/69f79aee-22b1-481c-b230-2035e84f9d23.png)

我们也可以轻松检查`ndarray`中的最大和最小像素值：

```py
print(cat_burrito.max())
print(cat_burrito.min())
```

这将返回以下结果：

![](img/0eb77b3e-f4b9-483c-9426-4fa4498031ae.png)

最后，我们可以通过运行以下代码从`ndarray`中显示灰度图像：

```py
plt.axis('off')
plt.imshow(cat_burrito, cmap='gray');
```

前面的代码返回了我们的图像，该图像可在[`github.com/PacktPublishing/Python-Machine-Learning-Blueprints-Second-Edition/tree/master/Chapter08`](https://github.com/PacktPublishing/Python-Machine-Learning-Blueprints-Second-Edition/tree/master/Chapter08)上找到，文件名为`output_grayscale_cat_burrito.png`。

彩色图像的特征提取过程是相同的；不过，对于彩色图像，我们的`ndarray`输出的形状将是三维的——一个**张量**——表示图像的**红、绿、蓝**（**RGB**）像素值。在这里，我们将执行与之前相同的过程，这次是在猫肉卷的彩色版本上进行。该图像可在 GitHub 上通过[`github.com/PacktPublishing/Python-Machine-Learning-Blueprints-Second-Edition/tree/master/Chapter08`](https://github.com/PacktPublishing/Python-Machine-Learning-Blueprints-Second-Edition/tree/master/Chapter08) 访问，文件名为`color_cat_burrito.jpg`。

我们通过以下代码提取猫肉卷的彩色版本特征：

```py
color_cat_burrito = mpimg.imread('images/color_cat_burrito.jpg')
color_cat_burrito.shape
```

运行此代码将返回以下输出：

![](img/642c0bf0-c57c-491e-92e5-b0923f98f55f.png)

同样，在这里我们看到该图像包含三个通道。我们的`color_cat_burrito`变量是一个张量，包含三个矩阵，告诉我们图像中每个像素的 RGB 值。

我们可以通过运行以下代码来显示`ndarray`中的彩色图像：

```py
plt.axis('off')
plt.imshow(color_cat_burrito);
```

这返回了我们的彩色图像。图像可以在 GitHub 上找到，链接为[`github.com/PacktPublishing/Python-Machine-Learning-Blueprints-Second-Edition/tree/master/Chapter08`](https://github.com/PacktPublishing/Python-Machine-Learning-Blueprints-Second-Edition/tree/master/Chapter08)，文件名为`output_color_cat_burrito.png`。

这是我们图像特征提取的第一步。我们一次处理一张图像，并通过几行代码将这些图像转换为数值。通过这一过程，我们看到，从灰度图像中提取特征会产生一个二维的 ndarray，而从彩色图像中提取特征会产生一个像素强度值的张量。

然而，这里有一个小问题。记住，这只是一张单独的图像，一条单独的训练样本，一行*数据*。以我们的灰度图像为例，如果我们将这个矩阵展平为一行，我们将拥有`image_height` x `image_width`列，或者在我们的例子中，是 1,322,500 列。我们可以通过运行以下代码片段来确认这一点：

```py
# flattening our grayscale cat_burrito and checking the length
len(cat_burrito.flatten())
```

这是一个问题！与其他机器学习建模任务一样，高维度会导致模型性能问题。在如此高维度的情况下，我们构建的任何模型都可能会出现过拟合，且训练时间会很慢。

这个维度问题是这种计算机视觉任务的普遍问题。即便是一个分辨率较低的数据集，比如 400 x 400 像素的灰度猫卷饼图像，也会导致每张图像有 160,000 个特征。

然而，解决这个问题的已知方法是：卷积神经网络。在接下来的部分，我们将继续使用卷积神经网络进行特征提取，以构建这些原始图像像素的低维表示。我们将讨论它们的工作原理，并继续了解它们在图像分类任务中为何如此高效。

# 卷积神经网络

卷积神经网络是一类解决我们在上一部分提到的高维度问题的神经网络，因此在图像分类任务中表现出色。事实证明，给定图像区域中的图像像素是高度相关的——它们为我们提供了关于该特定图像区域的相似信息。因此，使用卷积神经网络，我们可以扫描图像的区域，并在较低维度的空间中总结该区域。正如我们将看到的，这些低维表示，称为**特征图**，告诉我们关于各种形状存在的许多有趣的事情——从最简单的线条、阴影、环路和漩涡，到非常抽象、复杂的形式，特定于我们的数据，在我们的例子中是猫耳、猫脸或玉米饼——并且在比原始图像更少的维度中完成这一切。

在使用卷积神经网络从图像中提取这些低维特征之后，我们将把卷积神经网络的输出传入一个适合进行分类或回归任务的网络。在我们的例子中，当建模 Zalando 研究数据集时，卷积神经网络的输出将传入一个全连接神经网络，用于多分类。

那么这是怎么运作的呢？我们将讨论几个关键组件，这些组件对于卷积神经网络在灰度图像上的应用非常重要，它们对我们构建理解至关重要。

# 网络拓扑

你可能见过类似于上述的图示，它将卷积神经网络与前馈神经网络架构进行了对比。我们很快也会构建一个类似的东西！但这里描绘的是什么呢？看看这个：

![](img/82007989-4f27-47f2-bd42-69f1e2c7a49c.png)

在前面的图示中，最左边是我们的输入。这些是我们图像的提取特征，是一个值范围从 0 到 255 的矩阵（就像灰度猫卷饼的情况一样），描述了图像中像素的强度。

接下来，我们将数据通过交替的卷积层和最大池化层。这些层定义了所描绘架构中的卷积神经网络组件。在接下来的两部分中，我们将描述这些层类型的作用。

之后，我们将数据传递给一个全连接层，然后到达输出层。这两层描述了一个全连接神经网络。你可以在这里使用任何你喜欢的多分类算法，而不是全连接神经网络——例如**逻辑回归**或**随机森林分类器**——但对于我们的数据集，我们将使用全连接神经网络。

所描绘的输出层与其他任何多分类分类器相同。以我们的猫卷饼示例为例，假设我们正在构建一个模型，用来预测图像属于五种不同类别中的哪一种：鸡肉猫卷饼、牛排猫卷饼、牧羊猫卷饼、素食猫卷饼，或者鱼类猫卷饼（我让你发挥想象，想象我们的训练数据可能是什么样子）。输出层将是图像属于五个类别之一的预测概率，`max(probability)` 表示我们模型认为最有可能的类别。

从高层次上看，我们已经介绍了前面网络的架构或**拓扑**。我们讨论了输入与卷积神经网络组件以及前面拓扑中的全连接神经网络组件之间的关系。现在让我们更深入一点，加入一些概念，帮助我们更详细地描述拓扑：

+   网络有多少个卷积层？两个。

+   那么在每个卷积层中，有多少个特征图？卷积层 1 有 7 个，卷积层 2 有 12 个。

+   网络有多少个池化层？两个。

+   完全连接层有多少层？一层。

+   完全连接层中有多少个**神经元**？10 个。

+   输出是多少？五。

模型者决定使用两个卷积层而不是其他数量的层，或使用单个完全连接层而不是其他层数，应该被视为模型的**超参数**。也就是说，这些是我们作为模型开发者应该进行实验和交叉验证的内容，而不是模型显式学习和优化的参数。

仅通过查看网络的拓扑结构，你可以推断出一些关于你正在解决的问题的有用信息。正如我们讨论的那样，我们的网络输出层包含五个节点，这表明该神经网络是为了解决一个有五个类别的多类别分类任务。如果它是回归问题或二元分类问题，我们的网络架构通常（在大多数情况下）会只有一个输出节点。我们还知道，模型者在第一个卷积层使用了七个滤波器，在第二个卷积层使用了 12 个内核，因为每一层产生的特征图数量（我们将在下一节详细讨论这些内核是什么）。

太好了！我们学到了一些有用的术语，它们将帮助我们描述我们的网络，并构建对网络如何工作的概念理解。现在让我们探索一下我们架构中的卷积层。

# 卷积层和滤波器

卷积层和滤波器是卷积神经网络的核心。在这些层中，我们将一个滤波器（在本文中也称为**窗口**或**内核**）滑动过我们的 ndarray 特征，并在每一步进行内积运算。以这种方式对 ndarray 和内核进行卷积，最终得到一个低维的图像表示。让我们看看在这张灰度图像上是如何工作的（可在 image-assets 库中找到）：

![](img/cfd2a688-480d-4f87-82e0-cd686e4bdf3c.png)

上图是一个 5 x 5 像素的灰度图像，显示了一个黑色的对角线，背景是白色的。

从以下图示中提取特征后，我们得到如下的像素强度矩阵：

![](img/f0fe030e-ccaa-4d16-9edb-32c7012b9689.png)

接下来，假设我们（或 Keras）实例化了以下内核：

![](img/d0f28169-8372-4341-954f-3867f5b576af.png)

现在我们将可视化卷积过程。窗口的移动从图像矩阵的左上角开始。我们将窗口向右滑动一个预定的步长。在这种情况下，步长为 1，但通常步长大小应该视为模型的另一个超参数。一旦窗口到达图像的最右边缘，我们将窗口向下滑动 1（即步长大小），然后将窗口移回到图像的最左边，重新开始内积计算的过程。

现在让我们一步一步来做：

1.  将内核滑过矩阵的左上部分并计算内积：

![](img/16616ed8-ac7b-4479-84a4-013bb46bb2c5.png)

我将显式地展示第一步的内积计算，以便你能轻松跟上：

`(0x0)+(255x0)+(255x0)+(255x0)+(0x1)+(255x0)+(255x0)+(255x0)+(0x0) = 0`

我们将结果写入特征图并继续！

1.  计算内积并将结果写入我们的特征图：

![](img/6a00c530-8e36-4547-9038-4cfadbd29e61.png)

1.  第 3 步：

![](img/65e5b537-dd45-4afd-a1e5-6f1679b47ddb.png)

1.  我们已经到达图像的最右边缘。将窗口向下滑动 1 个单位，即我们的步长大小，然后从图像的最左边开始重新开始这个过程：

![](img/a19790b1-455f-4a34-96f0-aeb86d966c6a.png)

1.  第 5 步：

![](img/1d16c2c3-909d-4a32-b221-8111f72cb5c9.png)

1.  第 6 步：

![](img/bb8ce393-a48b-40c5-86cf-e7d5cac709a1.png)

1.  第 7 步：

![](img/6c75d988-2688-4c58-a16e-ba6a3b23da98.png)

1.  第 8 步：

![](img/b403905a-5501-4067-92ab-e41a221b564d.png)

1.  第 9 步：

![](img/71dd0a5d-47a9-4e7d-804e-f1e833273591.png)

看！我们已经将原始的 5 x 5 图像表示为 3 x 3 的矩阵（我们的特征图）。在这个简单的示例中，我们已经将维度从 25 个特征减少到只有 9 个特征。让我们看看这个操作后的结果图像：

![](img/9dac7706-ca00-4455-8cb1-6fa21d61e9f2.png)

如果你觉得这看起来和我们原始的黑色对角线图像一样，只是变小了，你是对的。内核的取值决定了识别到的内容，在这个具体示例中，我们使用了所谓的**单位矩阵内核**。如果使用其他值的内核，它将返回图像的其他特征——例如检测线条、边缘、轮廓、高对比度区域等。

我们将在每个卷积层同时应用多个内核对图像进行处理。使用的内核数量由模型设计者决定——这是另一个超参数。理想情况下，你希望在实现可接受的交叉验证结果的同时，尽可能使用最少的内核。越简单越好！然而，根据任务的复杂性，使用更多内核可能会带来性能提升。相同的思路也适用于调节模型的其他超参数，比如网络中的层数或每层的神经元数。我们在追求简洁与复杂性之间做出权衡，同时在通用性、速度、细节和精度之间进行选择。

核心数量是我们的选择，而每个核心的取值是我们模型的一个参数，这个参数是通过训练数据学习得到的，并在训练过程中通过优化减少成本函数来调整。

我们已经看到如何一步步将过滤器与图像特征进行卷积，以创建单一的特征图。那么，当我们同时应用多个核时会发生什么呢？这些特征图如何通过网络的每一层传递？让我们看看以下截图：

![](img/5550bd45-05ec-4fa9-9611-e5e383a9564c.png)

图片来源：Lee 等人，《卷积深度信念网络用于可扩展的无监督学习层次表示》，来自 Stack Exchange。源文本请参见：https://ai.stanford.edu/~ang/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf

上面的截图展示了一个训练过面孔图像的网络在每个卷积层生成的特征图。在网络的早期层（最底部），我们检测到简单的视觉结构——简单的线条和边缘。我们是通过使用我们的身份核来做到这一点的！这一层的输出会传递到下一层（中间一行），该层将这些简单的形状组合成抽象的形式。我们在这里看到，边缘的组合构建了面部的组成部分——眼睛、鼻子、耳朵、嘴巴和眉毛。中间层的输出又会传递到最终层，该层将边缘的组合合成完整的物体——在这种情况下，是不同人的面孔。

这个整个过程的一个特别强大的特性是，所有这些特征和表示都是从数据中学习出来的。在任何时候，我们都不会明确地告诉我们的模型：*模型，对于这个任务，我想在第一个卷积层使用一个身份核和一个底部 Sobel 核，因为我认为这两个核将提取出最丰富的特征图*。一旦我们设置了要使用的核数量的超参数，模型通过优化学习到哪些线条、边缘、阴影及其复杂组合最适合判断什么是面孔，什么不是。模型进行这种优化时，并没有使用任何关于面孔、猫卷饼或衣服的领域特定的硬编码规则。

卷积神经网络还有许多其他迷人的特性，这些我们在本章中不再讨论。然而，我们确实探讨了其基础知识，并且希望你能感受到使用卷积神经网络来提取高表达性、信号丰富、低维度特征的重要性。

接下来，我们将讨论*最大池化层*。

# 最大池化层

我们已经讨论了减少维度空间的重要性，以及如何使用卷积层来实现这一点。我们使用最大池化层有同样的原因——进一步减少维度。很直观地说，正如名字所示，最大池化是我们将一个窗口滑动到特征图上，并取该窗口的最大值。让我们回到我们对角线示例中的特征图来说明这一点，如下所示：

![](img/8c08fee6-0ca0-4ec3-85ef-5a8894ca291d.png)

让我们看看当我们使用 2 x 2 窗口进行最大池化时，前面的特征图会发生什么。再说一遍，我们这里只是返回`max(窗口中的值)`：

1.  返回`max(0,255,255,0)`，结果是 255：

![](img/6c75d908-819d-450b-ad34-eb16754aca77.png)

1.  第二步：

![](img/cfa2a066-bab6-4b25-bc96-c98f85bda1e5.png)

1.  第三步：

![](img/38d64d16-026a-4c48-9afd-413a09c42d45.png)

1.  第四步：

![](img/32f83e69-0ce7-41c5-971b-06e015cd1e3a.png)

通过使用 2 x 2 窗口进行最大池化，我们去掉了一列和一行，将表示从 3 x 3 变成了 2 x 2——不错吧！

还有其他形式的池化，比如平均池化和最小池化；然而，你会发现最大池化是最常用的。

接下来，我们将讨论展平，这是一个步骤，我们将执行此操作，将我们的最大池化特征图转换为适合建模的形状。

# 展平

到目前为止，我们专注于尽可能构建一个紧凑且富有表现力的特征表示，并通过卷积神经网络和最大池化层来实现这一目标。我们转换的最后一步是将我们的卷积和最大池化后的 ndarray（在我们的示例中是一个 2 x 2 的矩阵）展平为一行训练数据。

我们的最大池化对角线黑线示例在代码中看起来像下面这样：

```py
import numpy as np
max_pooled = np.array([[255,255],[255,255]])
max_pooled
```

运行这段代码将返回以下输出：

![](img/43d89914-dd37-48cd-af87-1bc40ec8cce9.png)

我们可以通过运行以下代码来检查形状：

```py
max_pooled.shape
```

这会返回以下输出：

![](img/9325c607-8ce4-4e88-bdcf-e6355d65483f.png)

为了将这个矩阵转换为单个训练样本，我们只需要运行`flatten()`。让我们来做这个，并查看我们展平后的矩阵的形状：

```py
flattened = max_pooled.flatten()
flattened.shape
```

这会生成以下输出：

![](img/e63af8b4-430b-47df-a61b-c6fc09bce587.png)

最初是一个 5 x 5 的像素强度矩阵，现在变成了一个包含四个特征的单行数据。我们现在可以将其传入一个全连接的神经网络。

# 全连接层和输出

全连接层是我们将输入——通过卷积、最大池化和展平操作得到的行——映射到目标类别或类别的地方。在这里，每个输入都与下一层中的每个**神经元**或**节点**相连接。这些连接的强度，或称为**权重**，以及每个节点中存在的**偏置**项，是模型的参数，这些参数在整个训练过程中不断优化，以最小化目标函数。

我们模型的最后一层将是输出层，它给出我们的模型预测结果。输出层中神经元的数量以及我们应用的**激活函数**由我们要解决的问题类型决定：回归、二分类或多分类。当我们在下一节开始使用 Zalando Research 的时尚数据集时，我们将看到如何为多分类任务设置全连接层和输出层。

全连接层和输出层——即我们架构中的前馈神经网络组件——属于一种与我们在本节中讨论的卷积神经网络不同的神经网络类型。我们在本节中简要描述了前馈网络的工作原理，目的是为了帮助理解我们架构中的分类器组件如何工作。你可以随时将这一部分架构替换为你更熟悉的分类器，例如**logit**！

有了这些基础知识，你现在可以开始构建你的网络了！

# 使用 Keras 构建卷积神经网络，分类 Zalando Research 数据集中的图像

在本节中，我们将构建卷积神经网络来分类 Zalando Research 的服装图片，使用该公司的时尚数据集。该数据集的仓库可以在[`github.com/zalandoresearch/fashion-mnist`](https://github.com/zalandoresearch/fashion-mnist)找到。

该数据集包含 70,000 张灰度图像——每张图像展示了一种服装——这些服装来自 10 种可能的服装类型。具体而言，目标类别如下：T 恤/上衣、裤子、毛衣、连衣裙、外套、凉鞋、衬衫、运动鞋、包和踝靴。

Zalando 是一家总部位于德国的电子商务公司，发布了这个数据集，以为研究人员提供经典手写数字 MNIST 数据集的替代方案。此外，这个数据集，他们称之为**Fashion MNIST**，在预测准确性上稍有挑战——MNIST 手写数字数据集可以在没有大量预处理或特别深度神经网络的情况下以 99.7%的准确率进行预测。

好的，让我们开始吧！请按照以下步骤操作：

1.  将仓库克隆到我们的桌面。在终端中运行以下命令：

```py
cd ~/Desktop/
git clone git@github.com:zalandoresearch/fashion-mnist.git
```

如果你还没有安装 Keras，请通过命令行运行`pip install keras`进行安装。我们还需要安装 TensorFlow。为此，请在命令行中运行`pip install tensorflow`。

1.  导入我们将要使用的库：

```py
import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.utils import np_utils, plot_model
from PIL import Image
import matplotlib.pyplot as plt
```

这些库中的许多应该已经很熟悉了。然而，对于你们中的一些人来说，这可能是第一次使用 Keras。Keras 是一个流行的 Python 深度学习库。它是一个可以运行在 TensorFlow、CNTK 或 Theano 等机器学习框架之上的封装库。

对于我们的项目，Keras 将在后台运行 TensorFlow。直接使用 TensorFlow 可以让我们更明确地控制网络的行为；然而，由于 TensorFlow 使用数据流图来表示其操作，因此这可能需要一些时间来适应。幸运的是，Keras 抽象了很多内容，它的 API 对于熟悉`sklearn`的人来说非常容易学习。

另一个可能对你们中的一些人来说是新的库是**Python Imaging Library**（**PIL**）。PIL 提供了一些图像处理功能。我们将使用它来可视化我们 Keras 网络的拓扑结构。

1.  加载数据。Zalando 为我们提供了一个辅助脚本，帮助我们进行数据加载。我们只需要确保`fashion-mnist/utils/`在我们的路径中：

```py
sys.path.append('/Users/Mike/Desktop/fashion-mnist/utils/')
import mnist_reader
```

1.  使用辅助脚本加载数据：

```py
X_train, y_train = mnist_reader.load_mnist('/Users/Mike/Desktop/fashion-mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('/Users/Mike/Desktop/fashion-mnist/data/fashion', kind='t10k')
```

1.  看一下`X_train`、`X_test`、`y_train`和`y_test`的形状：

```py
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

运行这段代码会给出以下输出：

![](img/a8500b89-ac19-4d4c-b8b6-8506b8303741.png)

在这里，我们可以看到我们的训练集包含 60,000 张图像，测试集包含 10,000 张图像。每张图像当前是一个长度为 784 的值的向量。现在，让我们检查一下数据类型：

```py
print(type(X_train))
print(type(y_train))
print(type(X_test))
print(type(y_test))
```

这将返回以下内容：

![](img/ec3a5302-7c97-4748-8bae-72a155ad21db.png)

接下来，让我们看看数据的样子。记住，在当前形式下，每张图像是一个值的向量。我们知道这些图像是灰度图，所以为了可视化每张图像，我们必须将这些向量重新构造成一个 28 x 28 的矩阵。我们来做一下这个，并瞥一眼第一张图像：

```py
image_1 = X_train[0].reshape(28,28)
plt.axis('off')
plt.imshow(image_1, cmap='gray');
```

这将生成以下输出：

![](img/02c86512-feb3-4b76-bf86-1b412698621e.png)

太棒了！我们可以通过运行以下代码来查看这张图像所属的类别：

```py
y_train[0]
```

这将生成以下输出：

![](img/03c7a2b5-8520-4a03-9418-10238801bc15.png)

类别被编码为 0-9。在 README 文件中，Zalando 提供了我们需要的映射：

![](img/e7cf2657-275b-4cfa-9f61-c0dc75214a1b.png)

鉴于此，我们现在知道我们的第一张图像是一个踝靴。太棒了！让我们创建一个明确的映射，将这些编码值与它们的类别名称对应起来。稍后这会很有用：

```py
mapping = {0: "T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 
 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle Boot"}
```

很好。我们已经看到了单张图片，但我们仍然需要了解数据中的内容。图片长什么样子？理解这一点可以告诉我们一些信息。举个例子，我很想看看这些类别在视觉上有多么明显区分。看起来与其他类别相似的类别，比起那些更独特的类别，会更难以被分类器区分。

在这里，我们定义一个辅助函数，帮助我们完成可视化过程：

```py
def show_fashion_mnist(plot_rows, plot_columns, feature_array, target_array, cmap='gray', random_seed=None):
 '''Generates a plot_rows * plot_columns grid of randomly selected images from a feature         array. Sets the title of each subplot equal to the associated index in the target array and     unencodes (i.e. title is in plain English, not numeric). Takes as optional args a color map     and a random seed. Meant for EDA.'''
 # Grabs plot_rows*plot_columns indices at random from X_train. 
 if random_seed is not None:
 np.random.seed(random_seed)

 feature_array_indices = np.random.randint(0,feature_array.shape[0], size = plot_rows*plot_columns)

 # Creates our plots
 fig, ax = plt.subplots(plot_rows, plot_columns, figsize=(18,18))

 reshaped_images_list = []

 for feature_array_index in feature_array_indices:
 # Reshapes our images, appends tuple with reshaped image and class to a reshaped_images_list.
 reshaped_image = feature_array[feature_array_index].reshape((28,28))
 image_class = mapping[target_array[feature_array_index]]
 reshaped_images_list.append((reshaped_image, image_class))

 # Plots each image in reshaped_images_list to its own subplot
 counter = 0
 for row in range(plot_rows):
 for col in range(plot_columns):
 ax[row,col].axis('off')
 ax[row, col].imshow(reshaped_images_list[counter][0], 
                                cmap=cmap)
 ax[row, col].set_title(reshaped_images_list[counter][1])
 counter +=1
```

这个函数做什么？它从数据中随机选择一组图像，创建一个图像网格，这样我们就能同时查看多张图像。

它的参数包括所需的图像行数（`plot_rows`）、图像列数（`plot_columns`）、我们的`X_train`（`feature_array`）和`y_train`（`target_array`），并生成一个`plot_rows` x `plot_columns`大小的图像矩阵。作为可选参数，您可以指定一个`cmap`，即色图（默认值为`‘gray'`，因为这些是灰度图像），以及一个`random_seed`，如果复制可视化很重要的话。

让我们看看如何运行，如下所示：

```py
show_fashion_mnist(4,4, X_train, y_train, random_seed=72)
```

这将返回以下结果：

![](img/21924c3b-e0af-4a11-a139-27f612ad96c5.png)

可视化输出

移除`random_seed`参数，并多次重新运行这个函数。具体来说，运行以下代码：

```py
show_fashion_mnist(4,4, X_train, y_train)
```

你可能已经注意到，在这个分辨率下，一些类看起来非常相似，而其他一些类则非常不同。例如，t-shirt/top 目标类的样本可能看起来与 shirt 和 coat 目标类的样本非常相似，而 sandal 目标类似乎与其他类明显不同。在考虑模型可能的弱点与强项时，这是值得思考的内容。

现在让我们来看看数据集中目标类的分布情况。我们需要做上采样或下采样吗？让我们检查一下：

```py
y = pd.Series(np.concatenate((y_train, y_test)))
plt.figure(figsize=(10,6))
plt.bar(x=[mapping[x] for x in y.value_counts().index], height = y.value_counts());
plt.xlabel("Class")
plt.ylabel("Number of Images per Class")
plt.title("Distribution of Target Classes");
```

运行上述代码会生成以下图表：

![](img/6bc6b269-edaa-4f0c-a68a-980ae4c01d42.png)

太棒了！这里不需要做类平衡调整。

接下来，让我们开始预处理数据，为建模做好准备。

正如我们在*图像特征提取*部分讨论的，这些灰度图像包含从 0 到 255 的像素值。我们通过运行以下代码来确认这一点：

```py
print(X_train.max())
print(X_train.min())
print(X_test.max())
print(X_test.min())
```

这将返回以下值：

![](img/0782a868-cbe5-4b34-a5ed-33ae7d0d6511.png)

为了建模的目的，我们需要将这些值归一化到 0-1 的范围内。这是准备图像数据进行建模时的常见预处理步骤。保持这些值在这个范围内可以让我们的神经网络更快收敛。我们可以通过运行以下代码来归一化数据：

```py
# First we cast as float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Then normalize
X_train /= 255
X_test /= 255
```

我们的数据现在已从 0.0 缩放到 1.0。我们可以通过运行以下代码来确认这一点：

```py
print(X_train.max())
print(X_train.min())
print(X_test.max())
print(X_test.min())
```

这将返回以下输出：

![](img/5fca7448-9f2a-4680-be20-ec3b55f749de.png)

在运行我们的第一个 Keras 网络之前，我们需要执行的下一个预处理步骤是调整数据的形状。记住，`X_train`和`X_test`当前的形状分别是(60,000, 784)和(10,000, 784)。我们的图像仍然是向量。为了能够将这些美丽的卷积核应用到整个图像上，我们需要将它们调整为 28 x 28 的矩阵形式。此外，Keras 要求我们明确声明数据的通道数。因此，当我们将这些灰度图像调整为建模格式时，我们将声明`1`：

```py
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
```

最后，我们将对`y`向量进行独热编码，以符合 Keras 的目标形状要求：

```py
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
```

我们现在准备开始建模。我们的第一个网络将有八个隐藏层。前六个隐藏层将由交替的卷积层和最大池化层组成。然后我们将把这个网络的输出展平，并将其输入到一个两层的前馈神经网络中，最后生成预测。代码如下所示：

```py
model = Sequential()
model.add(Conv2D(filters = 35, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 35, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 45, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

让我们深入描述每一行的内容：

+   **第 1 行**：这里，我们只是实例化了我们的模型对象。接下来，我们将通过一系列的`.add()`方法调用依次定义架构——即层的数量。这就是 Keras API 的魅力所在。

+   **第 2 行**：这里，我们添加了第一个卷积层。我们指定了`35`个卷积核，每个大小为 3 x 3。之后，我们指定了图像输入的形状，28 x 28 x 1。我们只需要在网络的第一次`.add()`调用中指定输入形状。最后，我们将激活函数指定为`relu`。激活函数在将输出传递到下一层之前，对输出进行变换。我们将在`Conv2D`和`Dense`层上应用激活函数。这些变换具有许多重要的性质。在这里使用`relu`可以加速网络的收敛，[`www.cs.toronto.edu/~fritz/absps/imagenet.pdf`](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)，并且与其他激活函数相比，`relu`计算起来并不昂贵——我们只是将负值变为 0，其他正值保持不变。从数学上讲，`relu`函数为`max(0, value)`。在本章中，我们将为每一层使用`relu`激活函数，除了输出层。

+   **第 3 行**：这里，我们添加了第一个最大池化层。我们指定该层的窗口大小为 2 x 2。

+   **第 4 行**：这是我们的第二个卷积层。我们设置它的方式与第一个卷积层完全相同。

+   **第 5 行**：这是第二个最大池化层。我们设置该层的方式与第一个最大池化层完全相同。

+   **第 6 行**：这是我们第三个也是最后一个卷积层。这次，我们增加了额外的过滤器（`45`个，而之前的层是`35`个）。这只是一个超参数，我鼓励你尝试不同的变体。

+   **第 7 行**：这是第三个也是最后一个最大池化层。它的配置与之前的所有最大池化层完全相同。

+   **第 8 行**：这是我们展平卷积神经网络输出的地方。

+   **第 9 行**：这是我们全连接网络的第一层。在这一层，我们指定了`64`个神经元，并使用`relu`激活函数。

+   **第 10 行**：这是我们全连接网络的第二层。在这一层，我们指定了`32`个神经元，并使用`relu`激活函数。

+   **第 11 行**：这是我们的输出层。我们指定了`10`个神经元，等于我们数据中目标类别的数量。由于这是一个多分类问题，我们指定了`softmax`激活函数。输出将表示图像属于类别 0 到 9 的预测概率。这些概率的和为`1`。这 10 个概率中，最高的一个将代表我们的模型认为最有可能的类别。

+   **第 12 行**：这是我们编译 Keras 模型的地方。在编译步骤中，我们指定了优化器`Adam`，这是一种**梯度下降**算法，能够自动调整学习率。我们还指定了**损失函数**——在这种情况下，使用`categorical cross entropy`，因为我们正在执行多分类问题。最后，在 metrics 参数中，我们指定了`accuracy`。通过指定这一点，Keras 将在每个 epoch 结束时告诉我们训练和验证准确率。

我们可以通过运行以下命令来获取模型的总结：

```py
model.summary()
```

这将输出如下内容：

![](img/485b7f6f-0585-468a-b359-a5b0cd770863.png)

请注意，当数据通过模型时，输出形状如何变化。特别是，观察扁平化操作后的输出形状——只有 45 个特征。`X_train`和`X_test`中的原始数据每行有 784 个特征，所以这非常棒！

你需要安装`pydot`来渲染可视化。要安装它，请在终端运行`pip install pydot`。你可能需要重新启动内核以使安装生效。

使用 Keras 中的`plot_model`函数，我们可以以不同的方式可视化网络的拓扑结构。要做到这一点，请运行以下代码：

```py
plot_model(model, to_file='Conv_model1.png', show_shapes=True)
Image.open('Conv_model1.png')
```

运行前面的代码将保存拓扑到`Conv_model1.png`并生成如下内容：

![](img/b88953fb-ec6c-4b7a-bf4a-62caa3e1e736.png)

这个模型需要几分钟才能拟合。如果你担心系统的硬件规格，可以通过将训练周期数减少到`10`来轻松缩短训练时间。

运行以下代码块将拟合模型：

```py
my_fit_model = model.fit(X_train, y_train, epochs=25, validation_data=
                        (X_test, y_test))
```

在拟合步骤中，我们指定了`X_train`和`y_train`。然后我们指定了希望训练模型的周期数。接着，我们输入验证数据——`X_test`和`y_test`——以观察模型在外样本上的表现。我喜欢将`model.fit`步骤保存为变量`my_fit_model`，这样我们就可以在后面轻松地可视化每个 epoch 的训练和验证损失。

随着代码运行，你将看到每个 epoch 后模型的训练损失、验证损失和准确率。我们可以使用以下代码绘制模型的训练损失和验证损失：

```py
plt.plot(my_fit_model.history['val_loss'], label="Validation")
plt.plot(my_fit_model.history['loss'], label = "Train")
plt.xlabel("Epoch", size=15)
plt.ylabel("Cat. Crossentropy Loss", size=15)
plt.title("Conv Net Train and Validation loss over epochs", size=18)
plt.legend();
```

运行前面的代码将生成如下图。你的图可能不会完全相同——这里有几个随机过程在进行——但它应该大致相同：

![](img/d0a689ed-6b00-4d32-baf9-733f49b1804e.png)

一瞥这个图表，我们可以看出我们的模型出现了过拟合。我们看到每个训练周期的训练损失持续下降，但验证损失没有同步变化。让我们看看准确率，以了解这个模型在分类任务中的表现如何。我们可以通过运行以下代码来查看：

```py
plt.plot(my_fit_model.history['val_acc'], label="Validation")
plt.plot(my_fit_model.history['acc'], label = "Train")
plt.xlabel("Epoch", size=15)
plt.ylabel("Accuracy", size=15)
plt.title("Conv Net Train and Validation accuracy over epochs", 
           size=18)
plt.legend();
```

这将生成以下结果：

![](img/cfd56e4a-0ee6-4060-ad8b-04cc3521cc0b.png)

这个图表也告诉我们我们已经过拟合。但看起来我们的验证准确率已接近 80% 高位，真不错！为了获得模型达成的最大准确率以及出现该准确率的训练周期，我们可以运行以下代码：

```py
print(max(my_fit_model.history['val_acc']))
print(my_fit_model.history['val_acc'].index(max(my_fit_model.history['v
      al_acc'])))
```

你的结果可能会与我的不同，但这是我的输出：

![](img/7e6eb9cd-38e2-4a6b-a1f6-fd4b19c2adac.png)

使用我们的卷积神经网络，我们在第 21 个周期达到了最高的分类准确率 89.48%。这真是太棒了！但我们仍然需要解决过拟合问题。接下来，我们将通过使用**dropout 正则化**来重建模型。

Dropout 正则化是我们可以应用于神经网络全连接层的一种正则化方法。使用 dropout 正则化时，我们在训练过程中随机丢弃神经元及其连接。通过这样做，网络不会过于依赖于与特定节点相关的权重或偏置，从而能更好地进行样本外泛化。

这里，我们添加了 dropout 正则化，指定在每个 `Dense` 层丢弃 `35%` 的神经元：

```py
model = Sequential()
model.add(Conv2D(filters = 35, kernel_size=(3,3), input_shape=
         (28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 35, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 45, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

运行前面的代码将编译我们的新模型。让我们通过重新运行以下代码来再次查看总结：

```py
model.summary()
```

运行前面的代码会返回以下输出：

![](img/819a4e32-36b5-4fe9-91b8-2ceb69e1f8aa.png)

让我们通过重新运行以下代码来重新拟合我们的模型：

```py
my_fit_model = model.fit(X_train, y_train, epochs=25, validation_data=
                        (X_test, y_test))
```

一旦模型重新拟合完成，重新运行绘图代码以可视化损失。这是我的结果：

![](img/ad4333d4-48bf-43f0-962c-725832e0a4d3.png)

看起来更好了！我们训练和验证损失之间的差距缩小了，这是我们预期的效果，尽管仍然有一些改进的空间。

接下来，重新绘制准确率曲线。这是我这次训练的结果：

![](img/97fd4ea0-c7dc-412b-bd43-8da26dc2cf95.png)

从过拟合的角度看，这也显得更好。太棒了！在应用正则化后，我们达成的最佳分类准确率是多少？让我们运行以下代码：

```py
print(max(my_fit_model.history['val_acc']))
print(my_fit_model.history['val_acc'].index(max(my_fit_model.history['v
      al_acc'])))
```

我这次运行模型的输出如下：

![](img/f8b9853f-45f0-440c-89fd-fa95f6b5b572.png)

有意思！我们达到的最佳验证准确率比未正则化模型的还低，但差距不大。而且它仍然相当不错！我们的模型告诉我们，我们预测正确衣物类别的概率为 88.85%。

评估我们在这里取得的成果的一种方式是将我们的模型准确率与数据集的**基准准确率**进行比较。基准准确率就是通过天真地选择数据集中最常出现的类别所得到的分数。对于这个特定的数据集，因为类别是完美平衡的且有 10 个类别，所以基准准确率是 10%。我们的模型轻松超越了这一基准准确率，显然它已经从数据中学到了一些东西！

从这里开始，你可以去探索很多不同的方向！尝试构建更深的模型，或者对我们在模型中使用的众多超参数进行网格搜索。像评估任何其他模型一样评估你的分类器性能——试着构建一个混淆矩阵，了解我们预测准确的类别以及哪些类别的表现较弱！

# 总结

我们在这里确实覆盖了很多内容！我们讨论了如何从图像中提取特征，卷积神经网络是如何工作的，接着我们又将卷积神经网络构建成一个全连接网络架构。在这个过程中，我们还学习了许多新的术语和概念！

希望在阅读本章后，你会觉得这些图像分类技术——你曾经可能认为只有巫师才能掌握的知识——实际上只是一系列为了直观原因进行的数学优化！希望这些内容能够帮助你在处理感兴趣的图像处理项目时取得进展！
