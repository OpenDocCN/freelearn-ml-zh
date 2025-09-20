# 检测面部部位并叠加口罩

在第六章“学习对象分类”中，我们学习了对象分类以及如何使用机器学习来实现它。在本章中，我们将学习如何检测和跟踪不同的面部部位。我们将从理解面部检测流程及其构建方式开始讨论。然后，我们将使用这个框架来检测面部部位，如眼睛、耳朵、嘴巴和鼻子。最后，我们将学习如何在实时视频中将这些面部部位叠加有趣的口罩。

到本章结束时，我们应该熟悉以下主题：

+   理解 Haar 级联

+   整数图像及其必要性

+   构建通用的面部检测流程

+   在实时视频流中检测和跟踪面部、眼睛、耳朵、鼻子和嘴巴

+   在视频中自动叠加人脸面具、太阳镜和有趣的鼻子

# 技术要求

本章需要您对 C++ 编程语言有基本的熟悉度。本章中使用的所有代码都可以从以下 GitHub 链接下载：[`github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition/tree/master/Chapter_07`](https://github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition/tree/master/Chapter_07)。代码可以在任何操作系统上执行，尽管它仅在 Ubuntu 上进行了测试。

查看以下视频以查看代码的实际应用：

[`bit.ly/2SlpTK6`](http://bit.ly/2SlpTK6)

# 理解 Haar 级联

Haar 级联是基于 Haar 特征的级联分类器。什么是级联分类器？它简单地说是一系列弱分类器的串联，这些弱分类器可以用来创建一个强分类器。我们所说的**弱**和**强**分类器是什么意思？弱分类器是性能有限的分类器。它们没有正确分类所有事物的能力。如果你把问题简化到极致，它们可能达到可接受的水平。另一方面，强分类器在正确分类我们的数据方面非常出色。我们将在接下来的几段中看到这一切是如何结合在一起的。Haar 级联的另一个重要部分是**Haar 特征**。这些特征是矩形和这些区域之间差异的简单求和。让我们考虑以下图表：

![](img/50bdb239-0bd8-49d2-af12-631b7735f346.png)

如果我们想要计算区域 ABCD 的 Haar 特征，我们只需要计算该区域中白色像素和蓝色像素之间的差异。正如我们从四个图中可以看到的，我们使用不同的模式来构建 Haar 特征。还有许多其他模式也被用作此目的。我们在多个尺度上这样做，以使系统具有尺度不变性。当我们说多个尺度时，我们只是将图像缩小以再次计算相同的特征。这样，我们可以使系统对给定对象的尺寸变化具有鲁棒性。

结果表明，这个拼接系统是检测图像中对象的一个非常好的方法。在 2001 年，保罗·维奥拉和迈克尔·琼斯发表了一篇开创性的论文，其中他们描述了一种快速有效的对象检测方法。如果你对了解更多信息感兴趣，你可以查看他们的论文，链接为[`www.cs.ubc.ca/~lowe/425/slides/13-ViolaJones.pdf`](http://www.cs.ubc.ca/~lowe/425/slides/13-ViolaJones.pdf)。

让我们深入探讨，了解他们实际上做了什么。他们基本上描述了一个使用简单分类器级联提升的算法。这个系统被用来构建一个能够真正表现良好的强大分类器。他们为什么使用这些简单的分类器而不是更复杂的分类器，后者可能更准确呢？嗯，使用这种技术，他们能够避免构建一个需要具有高精度性能的单个分类器的问题。这些单步分类器往往很复杂且计算密集。他们的技术之所以效果如此之好，是因为简单的分类器可以是弱学习器，这意味着它们不需要很复杂。考虑构建一个表格检测器的问题。我们希望构建一个能够自动学习表格外观的系统。基于这个知识，它应该能够识别任何给定图像中是否存在表格。为了构建这个系统，第一步是收集可以用来训练我们系统的图像。在机器学习领域有许多技术可以用来训练这样的系统。记住，如果我们想让我们的系统表现良好，我们需要收集大量的表格和非表格图像。在机器学习的术语中，表格图像被称为**正样本**，而非表格图像被称为**负样本**。我们的系统将摄取这些数据，并学会区分这两类。为了构建一个实时系统，我们需要保持我们的分类器既简单又好。唯一的问题是简单分类器不太准确。如果我们试图使它们更准确，那么这个过程最终会变得计算密集，从而变慢。在机器学习中，准确性和速度之间的这种权衡非常常见。因此，我们通过串联多个弱分类器来创建一个强大且统一的分类器来克服这个问题。我们不需要弱分类器非常准确。为了确保整体分类器的质量，Viola 和 Jones 在级联步骤中描述了一种巧妙的技术。你可以阅读论文来了解整个系统。

现在我们已经了解了整个流程，让我们看看如何构建一个能够在实时视频中检测人脸的系统。第一步是从所有图像中提取特征。在这种情况下，算法需要这些特征来学习和理解人脸的外观。他们在论文中使用了 Haar 特征来构建特征向量。一旦我们提取了这些特征，我们就将它们通过一个分类器的级联。我们只是检查所有不同的矩形子区域，并丢弃其中没有人脸的子区域。这样，我们就能快速得出结论，看一个给定的矩形是否包含人脸。

# 什么是积分图像？

为了提取这些 Haar 特征，我们必须计算图像中许多矩形区域中像素值的总和。为了使其尺度不变，我们需要在多个尺度（各种矩形大小）上计算这些面积。如果天真地实现，这将是一个非常计算密集的过程；我们不得不迭代每个矩形的所有像素，包括如果它们包含在不同的重叠矩形中，则多次读取相同的像素。如果你想要构建一个可以实时运行的系统，你不能在计算上花费这么多时间。我们需要找到一种方法来避免在面积计算中的这种巨大冗余，因为我们多次迭代相同的像素。为了避免它，我们可以使用一种称为积分图像的东西。这些图像可以在线性时间内初始化（通过仅迭代图像两次）并且然后通过读取仅四个值来提供任何大小矩形的像素总和。为了更好地理解它，让我们看一下以下图示：

![](img/17189123-7d03-4a17-a954-222e9ca17c79.png)

如果我们想计算图中任何矩形的面积，我们不必迭代该区域的所有像素。让我们考虑由图像中的左上角点和任何点 P（作为对角点）形成的矩形。让 A[P]表示这个矩形的面积。例如，在上一个图像中，A[B]表示由左上角点和**B**作为对角点形成的 5 x 2 矩形的面积。为了清晰起见，让我们看一下以下图示：

![](img/fb75616a-63ba-49d5-8c05-d2cdfae54485.png)

让我们考虑上一张图片中的左上角方块。蓝色像素表示从左上角像素到点**A**之间的区域。这表示为 A[A]。其余的图分别用它们各自的名字表示：A[B]、A[C]和 A[D]。现在，如果我们想计算如图所示的 ABCD 矩形的面积，我们会使用以下公式：

**矩形面积**：*ABCD* = *A[C]* - (*A[B]* + *A[D]* - *A[A]*)

这个特定公式有什么特别之处呢？正如我们所知，从图像中提取 Haar 特征包括计算这些求和，并且我们不得不在图像的多个尺度上进行多次计算。许多这些计算是重复的，因为我们会在相同的像素上反复迭代。这非常慢，以至于构建实时系统是不切实际的。因此，我们需要这个公式。正如你所见，我们不必多次迭代相同的像素。如果我们想计算任何矩形的面积，前面方程右侧的所有值都在我们的积分图像中 readily available。我们只需挑选正确的值，将它们代入前面的方程，并提取特征。

# 在实时视频中叠加人脸面具

OpenCV 提供了一个优秀的面部检测框架。我们只需要加载级联文件，并使用它来检测图像中的面部。当我们从摄像头捕获视频流时，我们可以在我们的脸上叠加有趣的口罩。看起来可能像这样：

![](img/4b3881f9-9bf6-4642-864b-b47979fcd9dc.png)

让我们看看代码的主要部分，看看如何将这个面具叠加到输入视频流中的面部上。完整的代码可以在本书提供的可下载代码包中找到：

```py
#include "opencv2/core/utility.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

...

int main(int argc, char* argv[]) 
{ 
    string faceCascadeName = argv[1]; 

    // Variable declaration and initialization 
    ...
    // Iterate until the user presses the Esc key 
    while(true) 
    { 
        // Capture the current frame 
        cap >> frame; 

        // Resize the frame 
        resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA); 

        // Convert to grayscale 
        cvtColor(frame, frameGray, COLOR_BGR2GRAY); 

        // Equalize the histogram 
        equalizeHist(frameGray, frameGray); 

        // Detect faces 
        faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0|HAAR_SCALE_IMAGE, Size(30, 30) ); 
```

让我们快速停下来看看这里发生了什么。我们从摄像头读取输入帧并将其调整到我们选择的大小。捕获的帧是一个彩色图像，面部检测是在灰度图像上进行的。因此，我们将其转换为灰度并均衡直方图。为什么我们需要均衡直方图？我们需要这样做来补偿任何问题，例如光照或饱和度。如果图像太亮或太暗，检测效果会较差。因此，我们需要均衡直方图以确保我们的图像具有健康的像素值范围：

```py
        // Draw green rectangle around the face 
        for(auto& face:faces) 
        { 
            Rect faceRect(face.x, face.y, face.width, face.height); 

            // Custom parameters to make the mask fit your face. You may have to play around with them to make sure it works. 
            int x = face.x - int(0.1*face.width); 
            int y = face.y - int(0.0*face.height); 
            int w = int(1.1 * face.width); 
            int h = int(1.3 * face.height); 

            // Extract region of interest (ROI) covering your face 
            frameROI = frame(Rect(x,y,w,h));
```

在这个阶段，我们已经知道脸的位置。因此，我们提取感兴趣的区域，以便在正确的位置叠加面具：

```py
            // Resize the face mask image based on the dimensions of the above ROI 
            resize(faceMask, faceMaskSmall, Size(w,h)); 

            // Convert the previous image to grayscale 
            cvtColor(faceMaskSmall, grayMaskSmall, COLOR_BGR2GRAY); 

            // Threshold the previous image to isolate the pixels associated only with the face mask 
            threshold(grayMaskSmall, grayMaskSmallThresh, 230, 255, THRESH_BINARY_INV); 
```

我们隔离与面部面具相关的像素。我们希望以这种方式叠加面具，使其看起来不像一个矩形。我们希望叠加对象的精确边界，使其看起来自然。现在让我们叠加面具：

```py
            // Create mask by inverting the previous image (because we don't want the background to affect the overlay) 
            bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv); 

            // Use bitwise "AND" operator to extract precise boundary of face mask 
            bitwise_and(faceMaskSmall, faceMaskSmall, maskedFace, grayMaskSmallThresh); 

            // Use bitwise "AND" operator to overlay face mask 
            bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv); 

            // Add the previously masked images and place it in the original frame ROI to create the final image 
            add(maskedFace, maskedFrame, frame(Rect(x,y,w,h))); 
        } 

    // code dealing with memory release and GUI 

    return 1; 
} 
```

# 代码中发生了什么？

首先要注意的是，这段代码需要两个输入参数——**人脸级联 XML**文件和**掩码图像**。你可以使用在`resources`文件夹下提供的`haarcascade_frontalface_alt.xml`和`facemask.jpg`文件。我们需要一个分类器模型，它可以用来检测图像中的面部，OpenCV 提供了一个预构建的 XML 文件，可以用于此目的。我们使用`faceCascade.load()`函数来加载 XML 文件，并检查文件是否正确加载。我们初始化视频捕获对象以从摄像头捕获输入帧。然后将其转换为灰度以运行检测器。`detectMultiScale`函数用于提取输入图像中所有面的边界。我们可能需要根据需要调整图像的大小，因此该函数的第二个参数负责这一点。这个缩放因子是我们每次缩放时跳过的距离；由于我们需要在多个尺度上查找面部，下一个大小将是当前大小的 1.1 倍。最后一个参数是一个阈值，它指定了需要保留当前矩形的相邻矩形数量。它可以用来增加面部检测器的鲁棒性。我们启动`while`循环，并在用户按下*Esc*键之前，在每一帧中持续检测面部。一旦检测到面部，我们就需要在其上叠加一个面具。我们可能需要稍微调整尺寸以确保面具贴合得很好。这种定制略为主观，并且取决于所使用的面具。现在我们已经提取了感兴趣区域，我们需要在这个区域上方放置我们的面具。如果我们用其白色背景叠加面具，看起来会很奇怪。我们必须提取面具的确切曲线边界，然后进行叠加。我们希望颅骨面具的像素是可见的，而剩余区域应该是透明的。

如我们所见，输入面具有一个白色背景。因此，我们通过对掩码图像应用阈值来创建一个面具。通过试错，我们可以看到`240`的阈值效果很好。在图像中，所有强度值大于`240`的像素将变为`0`，而其他所有像素将变为`255`。至于感兴趣区域，我们必须在这个区域中熄灭所有像素。为此，我们只需使用刚刚创建的掩码的逆即可。在最后一步，我们只需将带掩码的版本相加，以产生最终的输出图像。

# 戴上你的太阳镜

现在我们已经了解了如何检测人脸，我们可以将这个概念推广到检测人脸的不同部分。我们将使用眼睛检测器在实时视频中叠加太阳镜。重要的是要理解 Viola-Jones 框架可以应用于任何对象。准确性和鲁棒性将取决于对象的独特性。例如，人脸具有非常独特的特征，因此很容易训练我们的系统变得鲁棒。另一方面，像毛巾这样的对象太通用，它没有这样的区分特征，因此构建鲁棒的毛巾检测器更困难。一旦你构建了眼睛检测器和叠加眼镜，它看起来可能就像这样：

![图片](img/27045d2d-9d41-4a3f-8b1a-1906be8ea46a.png)

让我们看看代码的主要部分：

```py
...
int main(int argc, char* argv[]) 
{ 
    string faceCascadeName = argv[1]; 
    string eyeCascadeName = argv[2]; 

    // Variable declaration and initialization
    ....
    // Face detection code 
    ....
    vector<Point> centers; 
    ....     
    // Draw green circles around the eyes 
    for( auto& face:faces ) 
    { 
        Mat faceROI = frameGray(face[i]); 
        vector<Rect> eyes; 

        // In each face, detect eyes eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30)); 
```

如我们所见，我们只在人脸区域运行眼睛检测器。我们不需要在整个图像中搜索眼睛，因为我们知道眼睛总是在人脸上的：

```py
            // For each eye detected, compute the center 
            for(auto& eyes:eyes) 
            { 
                Point center( face.x + eye.x + int(eye.width*0.5), face.y + eye.y + int(eye.height*0.5) ); 
                centers.push_back(center); 
            } 
        } 

        // Overlay sunglasses only if both eyes are detected 
        if(centers.size() == 2) 
        { 
            Point leftPoint, rightPoint; 

            // Identify the left and right eyes 
            if(centers[0].x < centers[1].x) 
            { 
                leftPoint = centers[0]; 
                rightPoint = centers[1]; 
            } 
            else 
            { 
                leftPoint = centers[1]; 
                rightPoint = centers[0]; 
            } 
```

我们只在找到两只眼睛时检测眼睛并将它们存储起来。然后我们使用它们的坐标来确定哪一个是左眼，哪一个是右眼：

```py
            // Custom parameters to make the sunglasses fit your face. You may have to play around with them to make sure it works. 
            int w = 2.3 * (rightPoint.x - leftPoint.x); 
            int h = int(0.4 * w); 
            int x = leftPoint.x - 0.25*w; 
            int y = leftPoint.y - 0.5*h; 

            // Extract region of interest (ROI) covering both the eyes 
            frameROI = frame(Rect(x,y,w,h)); 

            // Resize the sunglasses image based on the dimensions of the above ROI 
            resize(eyeMask, eyeMaskSmall, Size(w,h)); 
```

在前面的代码中，我们调整了太阳镜的大小，以适应我们在网络摄像头中的人脸比例。让我们检查剩余的代码：

```py
            // Convert the previous image to grayscale 
            cvtColor(eyeMaskSmall, grayMaskSmall, COLOR_BGR2GRAY); 

            // Threshold the previous image to isolate the foreground object 
            threshold(grayMaskSmall, grayMaskSmallThresh, 245, 255, THRESH_BINARY_INV); 

            // Create mask by inverting the previous image (because we don't want the background to affect the overlay) 
            bitwise_not(grayMaskSmallThresh, grayMaskSmallThreshInv); 

            // Use bitwise "AND" operator to extract precise boundary of sunglasses 
            bitwise_and(eyeMaskSmall, eyeMaskSmall, maskedEye, grayMaskSmallThresh); 

            // Use bitwise "AND" operator to overlay sunglasses 
            bitwise_and(frameROI, frameROI, maskedFrame, grayMaskSmallThreshInv); 

            // Add the previously masked images and place it in the original frame ROI to create the final image 
            add(maskedEye, maskedFrame, frame(Rect(x,y,w,h))); 
        } 

        // code for memory release and GUI 

    return 1; 
} 
```

# 查看代码内部

你可能已经注意到代码的流程看起来与我们讨论的“在实时视频中叠加人脸遮罩”部分中的人脸检测代码相似。我们加载了一个人脸检测级联分类器以及眼睛检测级联分类器。那么，为什么在检测眼睛时我们需要加载人脸级联分类器呢？好吧，我们实际上并不需要使用人脸检测器，但它有助于我们限制眼睛位置的搜索。我们知道眼睛总是位于某人的脸上，因此我们可以将眼睛检测限制在人脸区域。第一步是检测人脸，然后在该区域运行我们的眼睛检测代码。由于我们将在一个更小的区域上操作，这将更快，效率更高。

对于每一帧，我们首先检测人脸。然后我们继续在这个区域检测眼睛的位置。完成这一步后，我们需要叠加太阳镜。为此，我们需要调整太阳镜图像的大小，确保它适合我们的脸。为了得到正确的比例，我们可以考虑被检测到的两只眼睛之间的距离。只有当我们检测到两只眼睛时，我们才叠加太阳镜。这就是为什么我们首先运行眼睛检测器，收集所有中心点，然后叠加太阳镜。一旦我们有了这个，我们只需要叠加太阳镜遮罩。用于遮罩的原理与我们用于叠加人脸遮罩的原理非常相似。你可能需要根据你的需求自定义太阳镜的大小和位置。你可以尝试不同的太阳镜类型，看看它们看起来如何。

# 跟踪鼻子、嘴巴和耳朵

现在你已经知道了如何使用该框架跟踪不同的事物，你也可以尝试跟踪你的鼻子、嘴巴和耳朵。让我们使用一个鼻子检测器来叠加一个有趣的鼻子：

![图片](img/b51f1ac4-21ab-427a-bed0-ad67d269f491.png)

你可以参考代码文件以获取此检测器的完整实现。以下级联文件`haarcascade_mcs_nose.xml`、`haarcascade_mcs_mouth.xml`、`haarcascade_mcs_leftear.xml`和`haarcascade_mcs_rightear.xml`可以用来跟踪不同的面部部位。尝试使用它们，并尝试给自己叠加一个胡须或德古拉耳朵。

# 摘要

在本章中，我们讨论了 Haar 级联和积分图像。我们了解了人脸检测流程是如何构建的。我们学习了如何在实时视频流中检测和跟踪人脸。我们讨论了如何使用人脸检测框架来检测各种面部部位，如眼睛、耳朵、鼻子和嘴巴。最后，我们学习了如何使用人脸部位检测的结果在输入图像上叠加面具。

在下一章中，我们将学习关于视频监控、背景去除和形态学图像处理的内容。
