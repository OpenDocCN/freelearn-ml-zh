# 第九章：描述和匹配兴趣点

在本章中，我们将介绍以下内容：

+   匹配局部模板

+   描述和匹配局部强度模式

+   使用二进制描述符匹配关键点

# 简介

在上一章中，我们学习了如何通过检测图像中的特殊点来进行局部图像分析。这些关键点是选择得足够独特，以至于如果一个关键点在某个对象的图像上被检测到，那么在描述同一对象的其它图像中，我们期望同样点也会被检测到。我们还描述了一些更复杂的兴趣点检测器，这些检测器可以为关键点分配一个代表性的尺度因子和/或一个方向。正如我们将在本章中看到的，这些附加信息可以用来根据视点变化对场景表示进行归一化。

为了基于兴趣点进行图像分析，我们现在需要构建丰富的表示，以独特地描述每个关键点。本章将探讨已经提出的从兴趣点中提取描述符的不同方法。这些描述符通常是描述关键点和其邻域的 1D 或 2D 的二进制、整数或浮点数向量。一个好的描述符应该足够独特，能够唯一地表示图像中的每个关键点；它应该足够鲁棒，即使在可能的光照变化或视点变化的情况下，也能以相似的方式表示相同的点。理想情况下，它还应该紧凑，以减少内存负载并提高计算效率。

使用关键点完成的最常见操作之一是图像匹配。这项任务可以用来关联同一场景的两个图像，或者检测图像中目标对象的出现。在这里，我们将研究一些基本的匹配策略，这些策略将在下一章中进一步讨论。

# 匹配局部模板

特征点匹配是通过将一个图像中的点与另一个图像（或图像集中的点）进行对应的过程。当图像点对应于现实世界中相同场景元素时，图像点应该匹配。

单个像素当然不足以对两个关键点的相似性做出判断。这就是为什么在匹配过程中必须考虑每个关键点周围的图像块。如果两个块对应于同一场景元素，那么可以预期它们的像素将具有相似值。本食谱中提出的解决方案是直接逐像素比较像素块。这可能是特征点匹配中最简单的方法，但正如我们将看到的，它并不是最可靠的。尽管如此，在几种情况下，它仍然可以给出良好的结果。

## 如何做到这一点...

通常，补丁被定义为以关键点位置为中心的奇数大小的正方形。然后可以通过比较补丁内相应的像素强度值来测量两个正方形补丁之间的相似性。简单的**平方差之和**（**SSD**）是一个流行的解决方案。特征匹配策略的工作方式如下。首先，在每个图像中检测关键点。在这里，我们使用 FAST 检测器：

```py
    // Define feature detector 
    cv::Ptr<cv::FeatureDetector> ptrDetector;   // generic detector 
    ptrDetector= // we select the FAST detector 
                cv::FastFeatureDetector::create(80);    

    // Keypoint detection 
    ptrDetector->detect(image1,keypoints1); 
    ptrDetector->detect(image2,keypoints2); 

```

注意我们使用了通用的 `cv::Ptr<cv::FeatureDetector>` 指针类型，它可以指向任何特征检测器。然后，只需通过更改调用 `detect` 函数时使用的检测器，就可以在不同的兴趣点检测器上测试此代码。

第二步是定义一个矩形，例如大小为 `11x11` 的矩形，该矩形将用于定义每个关键点周围的区域：

```py
    // Define a square neighborhood 
    const int nsize(11);                       // size of the neighborhood 
    cv::Rect neighborhood(0, 0, nsize, nsize); // 11x11 
    cv::Mat patch1; 
    cv::Mat patch2; 

```

一个图像中的关键点与另一个图像中的所有关键点进行比较。对于第一图像中的每个关键点，确定第二图像中最相似的补丁。这个过程通过使用两个嵌套循环来实现，如下面的代码所示：

```py
    // For all keypoints in first image 
    // find best match in second image 
    cv::Mat result; 
    std::vector<cv::DMatch> matches; 

    // for all keypoints in image 1 
    for (int i=0; i<keypoints1.size(); i++) { 

      // define image patch 
      neighborhood.x = keypoints1[i].pt.x-nsize/2; 
      neighborhood.y = keypoints1[i].pt.y-nsize/2; 

      // if neighborhood of points outside image, 
      // then continue with next point 
      if (neighborhood.x<0 || neighborhood.y<0 ||   
          neighborhood.x+nsize >= image1.cols || 
          neighborhood.y+nsize >= image1.rows) 
      continue; 

      // patch in image 1 
      patch1 = image1(neighborhood); 

      // to contain best correlation value; 
      cv::DMatch bestMatch; 

      // for all keypoints in image 2 
      for (int j=0; j<keypoints2.size(); j++) { 

        // define image patch 
        neighborhood.x = keypoints2[j].pt.x-nsize/2; 
        neighborhood.y = keypoints2[j].pt.y-nsize/2; 

        // if neighborhood of points outside image, 
        // then continue with next point 
        if (neighborhood.x<0 || neighborhood.y<0 ||   
            neighborhood.x + nsize >= image2.cols ||   
            neighborhood.y + nsize >= image2.rows) 
        continue; 

       // patch in image 2 
       patch2 = image2(neighborhood); 

       // match the two patches 
       cv::matchTemplate(patch1,patch2,result, cv::TM_SQDIFF); 

       // check if it is a best match 
       if (result.at<float>(0,0) < bestMatch.distance) { 

         bestMatch.distance= result.at<float>(0,0); 
         bestMatch.queryIdx= i; 
         bestMatch.trainIdx= j; 
       } 
     } 

     // add the best match 
     matches.push_back(bestMatch); 
   } 

```

注意 `cv::matchTemplate` 函数的使用，我们将在下一节中描述它，该函数计算补丁相似度得分。当识别出潜在的匹配时，此匹配通过使用 `cv::DMatch` 对象来表示。这个实用类存储了两个匹配 `关键点` 的索引以及它们的相似度得分。

两个图像补丁越相似，这些补丁对应于同一场景点的概率就越高。这就是为什么按相似度得分对结果匹配点进行排序是一个好主意：

```py
    // extract the 25 best matches 
    std::nth_element(matches.begin(),   
                     matches.begin() + 25,matches.end()); 
    matches.erase(matches.begin() + 25,matches.end()); 

```

然后，您可以简单地保留通过给定相似度阈值的匹配。在这里，我们选择只保留 `N` 个最佳匹配点（我们使用 `N=25` 以便于可视化匹配结果）。

有趣的是，OpenCV 中有一个函数可以通过连接两个图像并使用线条连接每个对应点来显示匹配结果。该函数的使用方法如下：

```py
    //Draw the matching results 
    cv::Mat matchImage; 
    cv::drawMatches(image1,keypoints1,         // first image 
                    image2,keypoints2,         // second image 
                    matches,                   // vector of matches 
                    cv::Scalar(255,255,255),   // color of lines 
                    cv::Scalar(255,255,255));  // color of points 

```

这里是匹配结果：

![如何操作...](img/image_09_001-1.jpg)

## 它是如何工作的...

获得的结果当然不是完美的，但通过视觉检查匹配的图像点显示有许多成功的匹配。还可以观察到教堂两座塔的对称性造成了一些混淆。此外，由于我们试图将左图像中的所有点与右图像中的点匹配，我们得到了右图像中的一个点与左图像中的多个点匹配的情况。这是一种不对称的匹配情况，可以通过例如，只保留右图像中每个点的最佳得分匹配来纠正。

为了比较每张图像的图像块，我们在这里使用了一个简单的标准，即使用 `cv::TM_SQDIFF` 标志指定的像素级平方差之和。如果我们比较图像 `I1` 中的点 `(x,y)` 与图像 `I2` 中的潜在匹配点 `(x',y')`，那么相似度度量如下：

![如何工作...](img/image_09_002-1.jpg)

在这里，`(i,j)` 点的和提供了覆盖以每个点为中心的正方形模板的偏移量。由于相似块中相邻像素之间的差异应该很小，因此最佳匹配的块应该是具有最小和的块。这正是匹配函数主循环中所做的；也就是说，对于一张图像中的每个关键点，我们识别另一张图像中给出平方差异和最低的关键点。我们还可以拒绝那些和超过某个阈值值的匹配。在我们的情况下，我们只是按相似度从高到低排序它们。

在我们的例子中，匹配是通过大小为 `11x11` 的正方形块完成的。更大的邻域会产生更独特的块，但它也使它们对局部场景变化更敏感。

只要两张图像从相似的角度和相似的光照显示场景，比较两个图像窗口的简单平方差之和就会相对有效。确实，简单的光照变化会增加或减少一个块的所有像素强度，从而导致大的平方差。为了使匹配对光照变化更不变，可以使用其他公式来测量两个图像窗口之间的相似度。OpenCV 提供了这些公式中的许多。一个非常有用的公式是归一化平方差之和（`cv::TM_SQDIFF_NORMED` 标志）：

![如何工作...](img/image_09_003-1.jpg)

其他相似度度量基于相关性的概念，这在信号处理理论中定义为以下（使用 `cv::TM_CCORR` 标志）：

![如何工作...](img/image_09_004-1.jpg)

当两个块相似时，此值将达到最大。

识别的匹配存储在一个 `cv::DMatch` 实例的向量中。本质上，`cv::DMatch` 数据结构包含一个指向第一个关键点向量中元素的第一个索引，以及一个指向第二个关键点向量中匹配特征的第二个索引。它还包含一个代表两个匹配描述符之间距离的实数值。这个距离值用于比较两个 `cv::DMatch` 实例时 `operator<` 的定义。

在上一节中绘制匹配时，我们希望限制线条数量以使结果更易于阅读。因此，我们只显示了具有最低距离的 `25` 个匹配。为此，我们使用了 `std::nth_element` 函数，该函数将第 N 个元素放置在第 N 个位置，所有较小的元素都放置在此元素之前。一旦完成，向量就简单地清除了其剩余的元素。

## 更多内容...

`cv::matchTemplate`函数是我们特征匹配方法的核心。我们在这里以非常具体的方式使用了它，即比较两个图像块。然而，这个函数被设计成可以以更通用的方式使用。

### 模板匹配

图像分析中的一个常见任务是检测图像中特定模式或对象的发生。这可以通过定义一个对象的小图像、模板，并在给定的图像中搜索相似的发生来实现。通常，搜索被限制在感兴趣的区域，我们认为对象可能位于该区域。然后，模板在这个区域上滑动，并在每个像素位置计算相似度度量。这是`cv::matchTemplate`函数执行的操作。输入是一个小尺寸的模板图像和搜索图像。

结果是一个对应于每个像素位置相似度得分的浮点值`cv::Mat`函数。如果模板的大小为`MxN`，图像的大小为`WxH`，则结果矩阵的大小将为`(W-M+1)x(H-N+1)`。通常，你将关注最高相似度的位置；因此，典型的模板匹配代码将如下所示（假设目标变量是我们的模板）：

```py
    // define search region 
    cv::Mat roi(image2, // here top half of the image 
    cv::Rect(0,0,image2.cols,image2.rows/2)); 

    // perform template matching 
    cv::matchTemplate(roi,            // search region 
                      target,         // template 
                      result,         // result 
                      cv::TM_SQDIFF); // similarity measure 

    // find most similar location 
    double minVal, maxVal; 
    cv::Point minPt, maxPt; 
    cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt); 

    // draw rectangle at most similar location 
    // at minPt in this case 
    cv::rectangle(roi, cv::Rect(minPt.x, minPt.y,  
                                target.cols, target.rows), 255); 

```

记住这是一个代价高昂的操作，所以你应该限制搜索区域，并使用只有几个像素大小的模板。

## 参见

+   下一个方法，*描述和匹配局部强度模式*，描述了`cv::BFMatcher`类，该类实现了在本方法中使用的匹配策略。

# 描述和匹配局部强度模式

在第八章 "检测兴趣点"中讨论的 SURF 和 SIFT 关键点检测算法，为每个检测到的特征定义了一个位置、一个方向和一个尺度。尺度因子信息对于定义每个特征点周围分析窗口的大小是有用的。因此，定义的邻域将包含无论对象所属特征的缩放级别如何的相同视觉信息。这个方法将向你展示如何使用特征描述符来描述兴趣点的邻域。在图像分析中，这个邻域包含的视觉信息可以用来表征每个特征点，以便使每个点与其他点区分开来。特征描述符通常是描述特征点的 N 维向量，以对光照变化和小型透视变形不变的方式描述。通常，描述符可以使用简单的距离度量进行比较，例如欧几里得距离。因此，它们构成了一个强大的工具，可以用于对象匹配应用。

## 如何做...

`cv::Feature2D`抽象类定义了用于计算一系列关键点描述符的成员函数。由于大多数基于特征的方法都包括检测器和描述符组件，相关的类包括一个`detect`函数（用于检测兴趣点）和一个`compute`函数（用于计算它们的描述符）。这是`cv::SURF`和`cv::SIFT`类的情况。例如，以下是如何使用一个`cv::SURF`实例在两张图像中检测和描述特征点的方法：

```py
    // Define keypoints vector 
    std::vector<cv::KeyPoint> keypoints1; 
    std::vector<cv::KeyPoint> keypoints2; 

    // Define feature detector 
    cv::Ptr<cv::Feature2D> ptrFeature2D =     
                          cv::xfeatures2d::SURF::create(2000.0); 

    // Keypoint detection 
    ptrFeature2D->detect(image1,keypoints1); 
    ptrFeature2D->detect(image2,keypoints2); 

    // Extract the descriptor 
    cv::Mat descriptors1; 
    cv::Mat descriptors2; 
    ptrFeature2D->compute(image1,keypoints1,descriptors1); 
    ptrFeature2D->compute(image2,keypoints2,descriptors2); 

```

对于 SIFT，你只需调用`cv::SIFT::create`函数。计算兴趣点描述符的结果是一个矩阵（即一个`cv::Mat`实例），它将包含与关键点向量中元素数量一样多的行。这些行中的每一行都是一个 N 维描述符向量。在 SURF 描述符的情况下，它有一个默认大小为`64`，而对于 SIFT，默认维度是`128`。这个向量表征了特征点周围的强度模式。两个特征点越相似，它们的描述符向量应该越接近。请注意，你不必一定使用 SURF（SIFT）描述符与 SURF（SIFT）点一起使用；检测器和描述符可以以任何组合使用。

这些描述符现在将被用来匹配我们的关键点。正如我们在之前的食谱中所做的那样，第一张图像中的每个特征描述符向量都会与第二张图像中的所有特征描述符进行比较。获得最佳分数的配对（即两个描述符向量之间距离最低的配对）将被保留为该特征的最佳匹配。这个过程会重复应用于第一张图像中的所有特征。非常方便的是，这个过程在 OpenCV 的`cv::BFMatcher`类中得到了实现，因此我们不需要重新实现之前构建的双循环。这个类可以这样使用：

```py
    // Construction of the matcher  
    cv::BFMatcher matcher(cv::NORM_L2); 
    // Match the two image descriptors 
    std::vector<cv::DMatch> matches; 
    matcher.match(descriptors1,descriptors2, matches); 

```

这个类是`cv::DescriptorMatcher`类的子类，它定义了不同匹配策略的通用接口。结果是`cv::DMatch`实例的向量。

使用当前 SURF 的 Hessian 阈值，我们为第一张图像获得了`74`个关键点，为第二张图像获得了`71`个。 brute-force 方法将产生`74`个匹配。使用与之前食谱中的`cv::drawMatches`类产生以下图像：

![如何做...](img/B05388_09_02.jpg)

如所示，这些匹配中的几个正确地将左侧的一个点与其右侧的对应点联系起来。你可能会注意到一些错误；其中一些是由于观察到的建筑具有对称的立面，这使得一些局部匹配变得模糊。对于 SIFT，在相同数量的关键点下，我们得到了以下匹配结果：

![如何做...](img/B05388_09_03.jpg)

## 它是如何工作的...

良好的特征描述符必须对光照和视点的微小变化以及图像噪声的存在保持不变性。因此，它们通常基于局部强度差异。这是 SURF 描述符的情况，它们在关键点周围局部应用以下简单的核：

![如何工作...](img/image_09_007.jpg)

第一个核简单地测量水平方向上的局部强度差异（指定为`dx`），第二个核测量垂直方向上的这种差异（指定为`dy`）。用于提取描述符向量的邻域大小通常定义为特征缩放因子的`20`倍（即`20σ`）。然后，这个正方形区域被分割成`4x4`个更小的正方形子区域。对于每个子区域，在`5x5`均匀分布的位置（核大小为`2σ`）计算核响应（`dx`和`dy`）。所有这些响应如下求和，以从每个子区域提取四个描述符值：

![如何工作...](img/image_09_008-1.jpg)

由于有`4x4=16`个子区域，我们总共有`64`个描述符值。注意，为了给邻近像素更多的权重，即更接近关键点的值，核响应通过一个以关键点为中心的高斯加权（`σ=3.3`）。

`dx`和`dy`响应也用于估计特征的方向。这些值在以`σ`为间隔均匀分布的半径为`6σ`的圆形邻域内（以`4σ`的核大小）计算。对于给定的方向，在一定的角度间隔（`π/3`）内的响应被求和，给出最长向量的方向被定义为主导方向。

SIFT 是一种更丰富的描述符，它使用图像梯度而不是简单的强度差异。它还将每个关键点周围的正方形邻域分割成`4x4`子区域（也可以使用`8x8`或`2x2`子区域）。在这些区域内部，构建梯度方向的直方图。方向被离散化为`8`个桶，每个梯度方向条目通过一个与梯度幅度成比例的值增加。

这由以下图示说明，其中每个星形箭头集代表梯度方向的局部直方图：

![如何工作...](img/B05388_09_05.jpg)

将每个子区域的`16`个`8`桶直方图连接起来，然后产生一个`128`维度的描述符。注意，对于 SURF 来说，梯度值通过一个以关键点为中心的高斯滤波器加权，以使描述符对定义的邻域边缘梯度方向突然变化的敏感性降低。然后，将最终的描述符归一化，以使距离测量更加一致。

使用 SURF 和 SIFT 特征和描述符，可以实现尺度不变匹配。以下是一个示例，展示了两个不同尺度图像的 SURF 匹配结果（这里，显示了`50`个最佳匹配）：

![它是如何工作的...](img/B05388_09_06.jpg)

注意到`cv::Feature2D`类包括一个方便的成员函数，它可以同时检测兴趣点并计算它们的描述符，例如：

```py
    ptrFeature2D->detectAndCompute(image, cv::noArray(),  
                                   keypoints, descriptors); 

```

## 还有更多...

任何匹配算法产生的匹配结果总是包含大量不正确的匹配。为了提高匹配集的质量，存在许多策略。这里讨论了其中三种。

### 交叉检查匹配

验证获得的匹配的一个简单方法是在第二次重复相同的程序，但这次，第二幅图像的每个关键点都与第一幅图像的所有关键点进行比较。只有当我们从两个方向获得相同的配对关键点时，才认为匹配是有效的（也就是说，每个关键点是另一个的最佳匹配）。`cv::BFMatcher`函数提供了使用此策略的选项。这确实是一个标志；当设置为 true 时，它强制函数执行互匹配交叉检查：

```py
    cv::BFMatcher matcher2(cv::NORM_L2,    // distance measure 
                           true);          // cross-check flag 

```

改进的匹配结果如下所示（以 SURF 为例）：

![交叉检查匹配](img/B05388_09_07.jpg)

### 比率测试

我们已经指出，场景对象中的重复元素由于匹配视觉上相似结构的歧义性，会导致不可靠的结果。在这种情况下发生的情况是，一个关键点会与多个其他关键点很好地匹配。由于选择错误对应关系的概率很高，在这些模糊的情况下拒绝匹配可能更可取。

要使用此策略，我们接下来需要找到每个关键点的最佳两个匹配点。这可以通过使用`cv::DescriptorMatcher`类的`knnMatch`方法来完成。由于我们只想找到两个最佳匹配，我们指定`k=2`：

```py
    // find the best two matches of each keypoint 
    std::vector<std::vector<cv::DMatch>> matches; 
    matcher.knnMatch(descriptors1,descriptors2,  
                     matches, 2);  // find the k best matches 

```

下一步是拒绝所有与第二最佳匹配距离相似的最好匹配。由于`knnMatch`产生一个`std::vector`类的`std::vector`（这个第二个向量的大小为`k`），我们通过遍历每个关键点匹配并执行比率测试来完成此操作，即计算第二最佳距离与最佳距离的比率（如果两个最佳距离相等，则此比率将为 1）。所有具有高比率的匹配都被判断为模糊，因此被拒绝。以下是我们可以这样做的方法：

```py
    //perform ratio test 
    double ratio= 0.85; 
    std::vector<std::vector<cv::DMatch>>::iterator it; 
    for (it= matches.begin(); it!= matches.end(); ++it) { 

      // first best match/second best match 
      if ((*it)[0].distance/(*it)[1].distance < ratio) { 
        // it is an acceptable match 
        newMatches.push_back((*it)[0]); 
      } 
    } 
    // newMatches is the updated match set 

```

初始由`74`对组成的匹配集现在已减少到`23`对：

![比率测试](img/B05388_09_08.jpg)

### 距离阈值法

一种更简单的策略是拒绝那些其描述符之间的距离过高的匹配。这是通过使用`cv::DescriptorMatcher`类的`radiusMatch`方法来完成的：

```py
    // radius match 
    float maxDist= 0.4; 
    std::vector<std::vector<cv::DMatch>> matches2; 
    matcher.radiusMatch(descriptors1, descriptors2, matches2, maxDist); 
                       // maximum acceptable distance 
                       // between the 2 descriptors 

```

结果再次是一个`std::vector`实例的`std::vector`，因为该方法将保留所有距离小于指定阈值的匹配。这意味着一个给定的关键点可能在另一张图像中有多于一个的匹配点。相反，其他关键点将没有任何与之关联的匹配（相应的内部`std::vector`类的大小将为`0`）。对于我们的示例，结果是`50`对匹配集：

![距离阈值](img/B05388_09_09.jpg)

显然，你可以将这些策略结合起来以提高你的匹配结果。

## 参见

+   在第八章检测兴趣点的*检测尺度不变特征*配方中，介绍了相关的 SURF 和 SIFT 特征检测器，并提供了更多关于该主题的参考资料

+   在第十章估计图像中的投影关系的*使用随机样本一致性匹配图像*配方中，解释了如何使用图像和场景几何来获得质量更好的匹配集

+   在第十四章从示例中学习的*使用支持向量机和方向梯度直方图检测对象和人物*配方中，描述了 HOG，这是与 SIFT 类似的另一个描述符

+   由 E. Vincent 和 R. Laganière 在 2001 年发表的《Machine, Graphics and Vision》杂志上的文章*立体对中特征点的匹配：一些匹配策略的比较研究*，描述了其他可以用来提高匹配集质量的简单匹配策略

# 使用二进制描述符匹配关键点

在前面的配方中，我们学习了如何使用从图像强度梯度中提取的丰富描述符来描述关键点。这些描述符是具有`64`、`128`或有时甚至更长的维度的浮点向量，这使得它们在操作上成本较高。为了减少与这些描述符相关的内存和计算负载，引入了使用由简单的位序列（0 和 1）组成的描述符的想法。这里的挑战是使它们易于计算，同时保持它们对场景和视点的鲁棒性。本配方描述了一些这些二进制描述符。特别是，我们将查看 ORB 和 BRISK 描述符，我们在第八章检测兴趣点中介绍了它们相关的特征点检测器。

## 如何做到这一点...

由于 OpenCV 检测器和描述符具有通用接口，使用二进制描述符如 ORB 与使用 SURF 和 SIFT 等描述符并无区别。完整的基于特征的图像匹配序列如下：

```py
    // Define keypoint vectors and descriptors 
    std::vector<cv::KeyPoint> keypoints1; 
    std::vector<cv::KeyPoint> keypoints2; 
    cv::Mat descriptors1; 
    cv::Mat descriptors2; 

    // Define feature detector/descriptor 
    // Construct the ORB feature object 
    cv::Ptr<cv::Feature2D> feature = cv::ORB::create(60); 
                           // approx. 60 feature points 

    // Keypoint detection and description 
    // Detect the ORB features 
    feature->detectAndCompute(image1, cv::noArray(),  
                              keypoints1, descriptors1); 
    feature->detectAndCompute(image2, cv::noArray(),  
                              keypoints2, descriptors2); 

    // Construction of the matcher  
    cv::BFMatcher matcher(cv::NORM_HAMMING); // always use hamming norm 
    // for binary descriptors 
    // Match the two image descriptors 
    std::vector<cv::DMatch> matches; 
    matcher.match(descriptors1, descriptors2, matches); 

```

唯一的区别在于使用了汉明范数（`cv::NORM_HAMMING` 标志），它通过计算两个二进制描述符中不同位的数量来衡量它们之间的距离。在许多处理器上，此操作通过使用异或操作后跟简单的位计数来高效实现。匹配结果如下：

![如何做...](img/B05388_09_10.jpg)

使用另一个流行的二进制特征检测器/描述符：BRISK，也会得到相似的结果。在这种情况下，`cv::Feature2D` 实例是通过 `cv::BRISK::create` 调用创建的。正如我们在上一章所学，它的第一个参数是一个阈值，用于控制检测到的点的数量。

## 它是如何工作的...

ORB 算法在多个尺度上检测方向特征点。基于此结果，ORB 描述符通过使用简单的强度比较来提取每个关键点的表示。实际上，ORB 是基于之前提出的描述符 BRIEF 构建的。随后，它通过简单地选择关键点周围定义的邻域内的随机点对来创建二进制描述符。然后比较两个像素点的强度值，如果第一个点具有更高的强度，则将值 `1` 分配给相应的描述符位值。否则，分配值 `0`。在多个随机点对上重复此测试会生成一个由多个位组成的描述符；通常，使用 `128` 到 `512` 位（成对测试）。

这就是 ORB 所使用的方案。然后，需要做出的决定是使用哪组点对来构建描述符。实际上，即使点对是随机选择的，一旦它们被选中，就必须执行相同的二进制测试集来构建所有关键点的描述符，以确保结果的一致性。为了使描述符更具独特性，直觉告诉我们，某些选择可能比其他选择更好。此外，由于每个关键点的方向是已知的，当将强度模式分布与此方向归一化时（即，当点坐标相对于此关键点方向给出时），这引入了一些偏差。从这些考虑和实验验证中，ORB 已经确定了一组具有高方差和最小成对相关性的 `256` 个点对。换句话说，所选的二进制测试是那些在各种关键点上具有相等机会为 `0` 或 `1` 的测试，并且尽可能相互独立。

BRISK 的描述符非常相似。它也是基于成对强度比较，但有两大不同之处。首先，不是从邻域的`31x31`点中随机选择点，而是从由同心圆（由`60`个点组成）的采样模式中选择点，这些同心圆的点是等间距的。其次，这些样本点的强度是一个高斯平滑值，其`σ`值与中心关键点的距离成比例。从这些点中，BRISK 选择`512`个点对。

## 还有更多...

存在着几个其他的二进制描述符，感兴趣的读者应该查阅科学文献以了解更多关于这个主题的信息。由于它也包含在 OpenCV contrib 模块中，我们在这里将描述一个额外的描述符。

### FREAK

**FREAK**代表**快速视网膜关键点**。这同样是一种二进制描述符，但它没有关联的检测器。它可以应用于任何一组检测到的关键点，例如 SIFT、SURF 或 ORB。

与 BRISK 一样，FREAK 描述符也是基于同心圆上的采样模式。然而，为了设计他们的描述符，作者使用了人类眼睛的类比。他们观察到在视网膜上，随着距离黄斑的距离增加，神经节细胞的密度会降低。因此，他们构建了一个由`43`个点组成的采样模式，其中点的密度在中心点附近要大得多。为了获得其强度，每个点都通过一个大小也随中心距离增加的高斯核进行过滤。

为了确定应该执行成对比较，已经通过遵循与 ORB 使用的类似策略进行了实验验证。通过分析数千个关键点，保留了具有最高方差和最低相关性的二进制测试，从而得到`512`对。

FREAK 还引入了在级联中进行描述符比较的想法。也就是说，首先执行代表较粗信息（对应于在较大高斯核外围进行的测试）的前`128`位。只有当比较的描述符通过这个初始步骤后，才会执行剩余的测试。

使用 ORB 检测到的关键点，我们通过简单地创建`cv::DescriptorExtractor`实例来提取 FREAK 描述符，如下所示：

```py
    // to describe with FREAK  
    feature = cv::xfeatures2d::FREAK::create(); 

```

匹配结果如下：

![FREAK](img/B05388_09_11.jpg)

下图说明了本配方中展示的三个描述符所使用的采样模式：

![FREAK](img/B05388_09_12.jpg)

第一个正方形是 ORB/BRIEF 描述符，其中在正方形网格上随机选择点对。通过线条连接的点对代表比较两个像素强度的可能测试。在这里，我们只展示了八个这样的点对；默认的 ORB 使用 `256` 对。中间的正方形对应于 BRISK 采样模式。点在显示的圆上均匀采样（为了清晰起见，我们在这里只标识第一个圆上的点）。最后，第三个正方形显示了 FREAK 的对数极坐标采样网格。虽然 BRISK 具有均匀的点分布，但 FREAK 在中心附近的点密度更高。例如，在 BRISK 中，你会在外圆上找到 `20` 个点，而在 FREAK 的情况下，其外圆只包括六个点。

## 参见

+   在 第八章 的 *Detecting FAST features at multiple scales* 菜谱中，介绍了相关的 BRISK 和 ORB 特征检测器，并提供了更多关于该主题的参考资料

+   由 *E. M. Calonder*、*V. Lepetit*、*M. Ozuysal*、*T. Trzcinski*、*C. Strecha* 和 *P. Fua* 撰写的 *BRIEF: Computing a Local Binary Descriptor Very Fast* 文章，发表在 *IEEE Transactions on Pattern Analysis and Machine Intelligence*，2012 年，描述了启发所提出二值描述符的 BRIEF 特征描述符

+   由 *A. Alahi*、*R. Ortiz* 和 *P. Vandergheynst* 撰写的 *FREAK: Fast Retina Keypoint* 文章，发表在 *IEEE Conference on Computer Vision and Pattern Recognition*，2012 年，描述了 FREAK 特征描述符
