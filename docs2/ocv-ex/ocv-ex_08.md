# 第八章：视频监控、背景建模和形态学操作

在本章中，我们将学习如何检测从静态摄像机拍摄的视频中的移动对象。这在视频监控系统中被广泛使用。我们将讨论可以用来构建此系统的不同特征。我们将了解背景建模，并看看我们如何可以使用它来构建实时视频中的背景模型。一旦我们这样做，我们将结合所有模块来检测视频中的感兴趣对象。

到本章结束时，你应该能够回答以下问题：

+   什么是天真背景减法？

+   什么是帧差分？

+   如何构建背景模型？

+   如何在静态视频中识别新对象？

+   形态学图像处理是什么？它与背景建模有何关系？

+   如何使用形态学算子实现不同的效果？

# 理解背景减法

背景减法在视频监控中非常有用。基本上，背景减法技术在需要检测静态场景中移动对象的情况下表现非常好。现在，这对视频监控有什么用呢？视频监控的过程涉及处理恒定的数据流。数据流始终在流入，我们需要分析它以识别任何可疑活动。让我们考虑酒店大堂的例子。所有墙壁和家具都有固定的位置。现在，如果我们构建一个背景模型，我们可以用它来识别大堂中的可疑活动。我们可以利用背景场景保持静态的事实（在这个例子中恰好是真实的）。这有助于我们避免任何不必要的计算开销。

正如其名所示，这个算法通过检测背景并将图像中的每个像素分配到两个类别：背景（假设它是静态和稳定的）或前景。然后它从当前帧中减去背景以获得前景。根据静态假设，前景对象将自然对应于在背景前移动的对象或人。

为了检测移动对象，我们首先需要构建背景模型。这不同于直接帧差分，因为我们实际上是在建模背景并使用此模型来检测移动对象。当我们说我们在“建模背景”时，我们基本上是在构建一个可以用来表示背景的数学公式。因此，这比简单的帧差分技术表现得更好。这种技术试图检测场景中的静态部分，然后更新背景模型。然后，这个背景模型用于检测背景像素。因此，它是一种自适应技术，可以根据场景进行调整。

# 天真背景减法

让我们从背景减法的讨论开始。背景减法过程是什么样的？考虑以下这张图片：

![天真背景减法](img/B04283_08_01.jpg)

上一张图片表示的是背景场景。现在，让我们向这个场景中引入一个新的物体：

![天真背景减法](img/B04283_08_02.jpg)

如前一张图片所示，场景中有一个新的物体。因此，如果我们计算这个图像和我们的背景模型之间的差异，你应该能够识别电视遥控器的位置：

![天真背景减法](img/B04283_08_03.jpg)

整个过程看起来是这样的：

![天真背景减法](img/B04283_08_04.jpg)

## 它是否工作得很好？

我们之所以称之为**天真**的方法，是有原因的。它在理想条件下是有效的，而我们都知道，在现实世界中没有任何事情是理想的。它能够相当好地计算给定物体的形状，但这是在一定的约束条件下完成的。这种方法的一个主要要求是物体的颜色和强度应该与背景有足够的差异。影响这类算法的一些因素包括图像噪声、光照条件、相机的自动对焦等等。

一旦一个新的物体进入我们的场景并停留下来，就很难检测到它前面的新物体。这是因为我们没有更新我们的背景模型，而新的物体现在成为了我们背景的一部分。考虑以下这张图片：

![是否工作得很好？](img/B04283_08_05.jpg)

现在，假设一个新的物体进入我们的场景：

![是否工作得很好？](img/B04283_08_06.jpg)

我们将其识别为一个新的物体，这是可以的。假设另一个物体进入场景：

![是否工作得很好？](img/B04283_08_07.jpg)

由于这两个不同物体的位置重叠，将很难识别它们的位置。这是在减去背景并应用阈值后的结果：

![是否工作得很好？](img/B04283_08_08.jpg)

在这种方法中，我们假设背景是静态的。如果背景的某些部分开始移动，那么这些部分将开始被检测为新的物体。因此，即使移动很小，比如挥动的旗帜，也会导致我们的检测算法出现问题。这种方法对光照变化也很敏感，并且无法处理任何相机移动。不用说，这是一个非常敏感的方法！我们需要能够在现实世界中处理所有这些事情的东西。

# 帧差分

我们知道我们无法保持一个静态的背景图像来检测对象。因此，解决这个问题的一种方法就是使用帧差分。这是我们能够使用的最简单技术之一，可以用来查看视频的哪些部分在移动。当我们考虑实时视频流时，连续帧之间的差异提供了大量信息。这个概念相当直接。我们只需计算连续帧之间的差异并显示差异。

如果我快速移动我的笔记本电脑，我们可以看到类似这样的情况：

![帧差分](img/B04283_08_09.jpg)

我们不再使用笔记本电脑，而是移动物体并看看会发生什么。如果我快速摇头，它看起来会是这样：

![帧差分](img/B04283_08_10.jpg)

如前图所示，只有视频的移动部分被突出显示。这为我们提供了一个很好的起点，可以看到视频中哪些区域在移动。让我们看看计算帧差异的函数：

```py
Mat frameDiff(Mat prevFrame, Mat curFrame, Mat nextFrame)
{
    Mat diffFrames1, diffFrames2, output;

    // Compute absolute difference between current frame and the next frame
    absdiff(nextFrame, curFrame, diffFrames1);

    // Compute absolute difference between current frame and the previous frame
    absdiff(curFrame, prevFrame, diffFrames2);

    // Bitwise "AND" operation between the above two diff images
    bitwise_and(diffFrames1, diffFrames2, output);

    return output;
}
```

帧差分相当直接。你计算当前帧与前一帧以及当前帧与下一帧之间的绝对差异。然后我们应用位与操作符来处理这些帧差异。这将突出显示图像中的移动部分。如果你只是计算当前帧与前一帧之间的差异，它往往会很嘈杂。因此，我们需要在连续帧差异之间使用位与操作符，以便在观察移动对象时获得一些稳定性。

让我们看看可以提取并返回摄像头帧的函数：

```py
Mat getFrame(VideoCapture cap, float scalingFactor)
{
    //float scalingFactor = 0.5;
    Mat frame, output;

    // Capture the current frame
    cap >> frame;

    // Resize the frame
    resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

    // Convert to grayscale
    cvtColor(frame, output, CV_BGR2GRAY);

    return output;
}
```

如我们所见，这相当直接。我们只需要调整帧的大小并将其转换为灰度图。现在我们已经准备好了辅助函数，让我们看看`main`函数，看看它是如何整合在一起的：

```py
int main(int argc, char* argv[])
{
    Mat frame, prevFrame, curFrame, nextFrame;
    char ch;

    // Create the capture object
    // 0 -> input arg that specifies it should take the input from the webcam
    VideoCapture cap(0);

    // If you cannot open the webcam, stop the execution!
    if( !cap.isOpened() )
        return -1;

    //create GUI windows
    namedWindow("Frame");

    // Scaling factor to resize the input frames from the webcam
    float scalingFactor = 0.75;

    prevFrame = getFrame(cap, scalingFactor);
    curFrame = getFrame(cap, scalingFactor);
    nextFrame = getFrame(cap, scalingFactor);

    // Iterate until the user presses the Esc key
    while(true)
    {
        // Show the object movement
        imshow("Object Movement", frameDiff(prevFrame, curFrame, nextFrame));

        // Update the variables and grab the next frame
        prevFrame = curFrame;
        curFrame = nextFrame;
        nextFrame = getFrame(cap, scalingFactor);

        // Get the keyboard input and check if it's 'Esc'
        // 27 -> ASCII value of 'Esc' key
        ch = waitKey( 30 );
        if (ch == 27) {
            break;
        }
    }

    // Release the video capture object
    cap.release();

    // Close all windows
    destroyAllWindows();

    return 1;
}
```

## 效果如何？

如我们所见，帧差分解决了我们之前遇到的一些重要问题。它可以快速适应光照变化或相机移动。如果一个物体进入画面并停留在那里，它将不会被检测到未来的帧中。这种方法的主要担忧之一是检测颜色均匀的对象。它只能检测颜色均匀对象的边缘。这是因为该对象的大部分区域将导致非常低的像素差异，如下面的图像所示：

![效果如何？](img/B04283_08_11.jpg)

假设这个物体稍微移动了一下。如果我们将其与前一帧进行比较，它看起来会是这样：

![效果如何？](img/B04283_08_12.jpg)

因此，我们在这个对象上有很少的像素被标记。另一个担忧是，很难检测一个物体是朝向相机移动还是远离相机。

# 高斯混合方法

在我们讨论**高斯混合**（**MOG**）之前，让我们看看什么是*混合模型*。混合模型只是一个可以用来表示我们数据中存在子群体的统计模型。我们并不真正关心每个数据点属于哪个类别。我们只需要确定数据内部是否有多个组。现在，如果我们用高斯函数来表示每个子群体，那么它就被称为高斯混合。让我们考虑以下图像：

![高斯混合方法](img/B04283_08_13.jpg)

现在，随着我们在这个场景中收集更多的帧，图像的每一部分都将逐渐成为背景模型的一部分。这也是我们之前讨论过的。如果一个场景是静态的，模型会自动调整以确保背景模型得到更新。前景掩码，本应表示前景对象，此时看起来像一张黑图，因为每个像素都是背景模型的一部分。

### 注意

OpenCV 实现了多个用于高斯混合方法的算法。其中之一被称为**MOG**，另一个被称为**MOG2**。要获取详细说明，您可以参考[`docs.opencv.org/master/db/d5c/tutorial_py_bg_subtraction.html#gsc.tab=0`](http://docs.opencv.org/master/db/d5c/tutorial_py_bg_subtraction.html#gsc.tab=0)。您还可以查看用于实现这些算法的原始研究论文。

让我们在场景中引入一个新的对象，并看看使用 MOG 方法的前景掩码是什么样的：

![高斯混合方法](img/B04283_08_14.jpg)

让我们等待一段时间，并向场景中引入一个新的对象。让我们看看使用 MOG2 方法的新前景掩码是什么样的：

![高斯混合方法](img/B04283_08_15.jpg)

如您在前面的图像中看到的，新对象被正确地识别了。让我们看看代码的有趣部分（完整的代码可以在`.cpp`文件中找到）：

```py
int main(int argc, char* argv[])
{
    // Variable declarations and initializations

    // Iterate until the user presses the Esc key
    while(true)
    {
        // Capture the current frame
        cap >> frame;

        // Resize the frame
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);

        // Update the MOG background model based on the current frame
        pMOG->operator()(frame, fgMaskMOG);

        // Update the MOG2 background model based on the current frame
        pMOG2->operator()(frame, fgMaskMOG2);

        // Show the current frame
        //imshow("Frame", frame);

        // Show the MOG foreground mask
        imshow("FG Mask MOG", fgMaskMOG);

        // Show the MOG2 foreground mask
        imshow("FG Mask MOG 2", fgMaskMOG2);

        // Get the keyboard input and check if it's 'Esc'
        // 27 -> ASCII value of 'Esc' key
        ch = waitKey( 30 );
        if (ch == 27) {
            break;
        }
    }

    // Release the video capture object
    cap.release();

    // Close all windows
    destroyAllWindows();

    return 1;
}
```

## 代码中发生了什么？

让我们快速浏览一下代码，看看那里发生了什么。我们使用高斯混合模型来创建一个背景减除对象。这个对象代表了一个模型，它将在我们遇到来自摄像头的新的帧时进行更新。正如我们在代码中所看到的，我们初始化了两个背景减除模型：`BackgroundSubtractorMOG` 和 `BackgroundSubtractorMOG2`。它们代表了用于背景减除的两个不同算法。第一个指的是由 *P. KadewTraKuPong* 和 *R. Bowden* 撰写的论文，标题为 *An improved adaptive background mixture model for real-time tracking with shadow detection*。您可以在 [`personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf`](http://personal.ee.surrey.ac.uk/Personal/R.Bowden/publications/avbs01/avbs01.pdf) 上查看。第二个指的是由 *Z.Zivkovic* 撰写的论文，标题为 *Improved adaptive Gausian Mixture Model for background subtraction*。您可以在 [`www.zoranz.net/Publications/zivkovic2004ICPR.pdf`](http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf) 上查看。我们启动一个无限循环 `while`，并持续从摄像头读取输入帧。对于每一帧，我们更新背景模型，如下面的代码所示：

```py
pMOG->operator()(frame, fgMaskMOG);
pMOG2->operator()(frame, fgMaskMOG2);
```

背景模型在这些步骤中更新。现在，如果一个新物体进入场景并停留，它将成为背景模型的一部分。这有助于我们克服朴素背景减除模型的最大缺点之一。

# 形态学图像处理

如前所述，背景减除方法受许多因素影响。它们的准确性取决于我们如何捕获数据以及如何处理数据。影响这些算法的最大因素之一是噪声水平。当我们说 *噪声* 时，我们指的是图像中的颗粒感、孤立的黑/白像素等问题。这些问题往往会影响我们算法的质量。这就是形态学图像处理发挥作用的地方。形态学图像处理在许多实时系统中被广泛使用，以确保输出质量。

形态学图像处理是指处理图像中特征形状的过程。例如，你可以使形状变厚或变薄。形态学算子依赖于图像中像素的顺序，但不是它们的值。这就是为什么它们非常适合在二值图像中操作形状。形态学图像处理也可以应用于灰度图像，但像素值不会很重要。

## 基本原理是什么？

形态学算子使用结构元素来修改图像。什么是结构元素？结构元素基本上是一个可以用来检查图像中小区域的形状。它被放置在图像的所有像素位置，以便它可以检查该邻域。我们基本上取一个小窗口并将其叠加在像素上。根据响应，我们在该像素位置采取适当的行动。

让我们考虑以下输入图像：

![基本原理是什么？](img/B04283_08_16.jpg)

我们将对这张图像应用一系列形态学操作，以查看形状如何变化。

# 瘦身形状

我们可以使用一个称为**腐蚀**的操作来实现这种效果。这是一个通过剥离图像中所有形状的边界层来使形状变薄的操作：

![瘦身形状](img/B04283_08_17.jpg)

让我们看看执行形态学腐蚀的函数：

```py
Mat performErosion(Mat inputImage, int erosionElement, int erosionSize)
{
    Mat outputImage;
    int erosionType;

    if(erosionElement == 0)
        erosionType = MORPH_RECT;

    else if(erosionElement == 1)
        erosionType = MORPH_CROSS;

    else if(erosionElement == 2)
        erosionType = MORPH_ELLIPSE;

    // Create the structuring element for erosion
    Mat element = getStructuringElement(erosionType, Size(2*erosionSize + 1, 2*erosionSize + 1), Point(erosionSize, erosionSize));

    // Erode the image using the structuring element
    erode(inputImage, outputImage, element);

    // Return the output image
    return outputImage;
}
```

你可以在`.cpp`文件中查看完整的代码，以了解如何使用此函数。基本上，我们使用内置的 OpenCV 函数构建一个结构元素。此对象用作探针，根据某些条件修改每个像素。这些**条件**指的是图像中特定像素周围发生的情况。例如，它是被白色像素包围的吗？或者它是被黑色像素包围的吗？一旦我们得到答案，我们就可以在该像素位置采取适当的行动。

# 加厚形状

我们使用一个称为**膨胀**的操作来实现加厚。这是一个通过向图像中所有形状添加边界层来使形状变厚的操作：

![加厚形状](img/B04283_08_18.jpg)

这里是执行此操作的代码：

```py
Mat performDilation(Mat inputImage, int dilationElement, int dilationSize)
{
    Mat outputImage;
    int dilationType;

    if(dilationElement == 0)
        dilationType = MORPH_RECT;

    else if(dilationElement == 1)
        dilationType = MORPH_CROSS;

    else if(dilationElement == 2)
        dilationType = MORPH_ELLIPSE;

    // Create the structuring element for dilation
    Mat element = getStructuringElement(dilationType, Size(2*dilationSize + 1, 2*dilationSize + 1), Point(dilationSize, dilationSize));

    // Dilate the image using the structuring element
    dilate(inputImage, outputImage, element);

    // Return the output image
    return outputImage;
}
```

# 其他形态学算子

这里有一些其他有趣的形态学算子。让我们首先看看输出图像。我们可以在本节末尾查看代码。

## 形态学开运算

这是一个**打开**形状的操作。这个算子常用于图像中的噪声去除。我们可以通过在图像上应用腐蚀后跟膨胀来实现形态学开运算。形态学开运算过程基本上通过将小对象放置在背景中来从图像的前景中移除小对象：

![形态学开运算](img/B04283_08_19.jpg)

这里是执行形态学开运算的函数：

```py
Mat performOpening(Mat inputImage, int morphologyElement, int morphologySize)
{
    Mat outputImage, tempImage;
    int morphologyType;

    if(morphologyElement == 0)
        morphologyType = MORPH_RECT;

    else if(morphologyElement == 1)
        morphologyType = MORPH_CROSS;

    else if(morphologyElement == 2)
        morphologyType = MORPH_ELLIPSE;

    // Create the structuring element for erosion
    Mat element = getStructuringElement(morphologyType, Size(2*morphologySize + 1, 2*morphologySize + 1), Point(morphologySize, morphologySize));

    // Apply morphological opening to the image using the structuring element
    erode(inputImage, tempImage, element);
    dilate(tempImage, outputImage, element);

    // Return the output image
    return outputImage;
}
```

如我们所见，我们对图像应用腐蚀和膨胀来执行形态学开运算。

## 形态学闭合

这是一个**闭合**形状的操作，通过填充间隙来实现。这个操作也用于噪声去除。我们通过在图像上应用膨胀后跟腐蚀来实现形态学闭合。这个操作通过将背景中的小对象变成前景来移除前景中的小孔。

![形态学闭合](img/B04283_08_20.jpg)

让我们快速看一下执行形态学闭合的函数：

```py
Mat performClosing(Mat inputImage, int morphologyElement, int morphologySize)
{
    Mat outputImage, tempImage;
    int morphologyType;

    if(morphologyElement == 0)
        morphologyType = MORPH_RECT;

    else if(morphologyElement == 1)
        morphologyType = MORPH_CROSS;

    else if(morphologyElement == 2)
        morphologyType = MORPH_ELLIPSE;

    // Create the structuring element for erosion
    Mat element = getStructuringElement(morphologyType, Size(2*morphologySize + 1, 2*morphologySize + 1), Point(morphologySize, morphologySize));

    // Apply morphological opening to the image using the structuring element
    dilate(inputImage, tempImage, element);
    erode(tempImage, outputImage, element);

    // Return the output image
    return outputImage;
}
```

## 绘制边界

我们通过形态学梯度来实现这一点。这是一种通过取图像膨胀和腐蚀的差来绘制形状边界的操作：

![绘制边界](img/B04283_08_21.jpg)

让我们来看看执行形态学梯度的函数：

```py
Mat performMorphologicalGradient(Mat inputImage, int morphologyElement, int morphologySize)
{
    Mat outputImage, tempImage1, tempImage2;
    int morphologyType;

    if(morphologyElement == 0)
        morphologyType = MORPH_RECT;

    else if(morphologyElement == 1)
        morphologyType = MORPH_CROSS;

    else if(morphologyElement == 2)
        morphologyType = MORPH_ELLIPSE;

    // Create the structuring element for erosion
    Mat element = getStructuringElement(morphologyType, Size(2*morphologySize + 1, 2*morphologySize + 1), Point(morphologySize, morphologySize));

    // Apply morphological gradient to the image using the structuring element
    dilate(inputImage, tempImage1, element);
    erode(inputImage, tempImage2, element);

    // Return the output image
    return tempImage1 - tempImage2;
}
```

## 白顶帽变换

当顶帽变换，也简称为顶帽变换，从图像中提取更细的细节时。我们可以通过计算输入图像与其形态学开运算的差来应用白顶帽变换。这使我们能够识别出图像中比结构元素小且比周围区域亮的物体。因此，根据结构元素的大小，我们可以从给定图像中提取各种物体：

![白顶帽变换](img/B04283_08_22.jpg)

如果你仔细观察输出图像，你可以看到那些黑色矩形。这意味着结构元素能够适应那里，因此这些区域被涂成了黑色。这里是执行此操作的函数：

```py
Mat performTopHat(Mat inputImage, int morphologyElement, int morphologySize)
{
    Mat outputImage;
    int morphologyType;

    if(morphologyElement == 0)
        morphologyType = MORPH_RECT;

    else if(morphologyElement == 1)
        morphologyType = MORPH_CROSS;

    else if(morphologyElement == 2)
        morphologyType = MORPH_ELLIPSE;

    // Create the structuring element for erosion
    Mat element = getStructuringElement(morphologyType, Size(2*morphologySize + 1, 2*morphologySize + 1), Point(morphologySize, morphologySize));

    // Apply top hat operation to the image using the structuring element
    outputImage = inputImage - performOpening(inputImage, morphologyElement, morphologySize);

    // Return the output image
    return outputImage;
}
```

## 黑顶帽变换

黑顶帽变换，也简称为黑帽变换，同样可以从图像中提取更细的细节。我们可以通过计算图像的形态学闭运算与图像本身的差来应用黑顶帽变换。这使我们能够识别出图像中比结构元素小且比周围区域暗的物体。

![黑顶帽变换](img/B04283_08_23.jpg)

让我们来看看执行黑帽变换的函数：

```py
Mat performBlackHat(Mat inputImage, int morphologyElement, int morphologySize)
{
    Mat outputImage;
    int morphologyType;

    if(morphologyElement == 0)
        morphologyType = MORPH_RECT;

    else if(morphologyElement == 1)
        morphologyType = MORPH_CROSS;

    else if(morphologyElement == 2)
        morphologyType = MORPH_ELLIPSE;

    // Create the structuring element for erosion
    Mat element = getStructuringElement(morphologyType, Size(2*morphologySize + 1, 2*morphologySize + 1), Point(morphologySize, morphologySize));

    // Apply black hat operation to the image using the structuring element
    outputImage = performClosing(inputImage, morphologyElement, morphologySize) - inputImage;

    // Return the output image
    return outputImage;
}
```

# 概述

在本章中，我们学习了用于背景建模和形态学图像处理的算法。我们讨论了简单的背景减法及其局限性。我们学习了如何使用帧差分来获取运动信息，以及它在追踪不同类型物体时可能对我们的约束。我们还讨论了高斯混合模型，包括其公式和实现细节。然后我们讨论了形态学图像处理。我们学习了它可用于各种目的，并通过不同的操作演示了其用例。

在下一章中，我们将讨论如何追踪物体以及可以用来实现这一目标的多种技术。
