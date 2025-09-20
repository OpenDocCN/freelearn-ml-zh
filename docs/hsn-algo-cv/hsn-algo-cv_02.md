# 第二章：开始使用 OpenCV

在上一章中，我们介绍了计算机视觉的基础知识，并展示了某些行业如何广泛使用它来改善他们的服务和产品。然后，我们学习了该领域最基本的概念，例如图像和像素。我们了解了色彩空间，并在本章结束时简要讨论了计算机视觉库和框架。我们将从上次结束的地方继续，即向您介绍最强大和最广泛使用的计算机视觉库之一，称为**OpenCV**。

OpenCV 是一个庞大的类、函数、模块和其他相关资源的集合，用于构建跨平台的计算机视觉应用程序。在本章中，我们将了解 OpenCV 的结构，它包含的模块及其用途，以及它支持的编程语言。我们将学习如何获取 OpenCV，并简要介绍您可以使用它来构建应用程序的可能工具。然后，我们将学习如何使用 CMake 的强大功能轻松创建使用 OpenCV 的项目。尽管这意味着我们的主要焦点将是 C++类和函数，但 wherever it makes sense，我们也会涵盖它们的 Python 等价物，以便熟悉这两种语言的开发者可以跟随本章中介绍的主题。

在了解了使用 OpenCV 库的初始阶段之后，我们将继续学习`Mat`类。我们将看到上一章中关于图像的所有概念是如何嵌入到 OpenCV 中`Mat`类的结构中的。我们还将讨论与`Mat`类兼容（或与之密切相关）的其他各种类。OpenCV 在函数中处理输入和输出参数的方法是本章后面部分将要讨论的一个重要主题。最后，我们将学习如何使用 OpenCV 在计算机视觉应用程序中应用输入、处理和输出的三个步骤。这需要学习使用 OpenCV 访问（并将数据写入）图像和视频文件。

本章作为对上一章入门章节的直接衔接，将阐述使用动手和实践示例学习计算机视觉算法的基础。

在本章中，我们将探讨以下内容：

+   OpenCV 是什么，在哪里可以获取它，以及如何使用它？

+   如何使用 CMake 创建 OpenCV 项目？

+   理解`Mat`类及其如何用于访问像素

+   如何使用`Mat_`、`Matx`和`UMat`类？

+   如何使用`imread`和`imwrite`函数读取和写入图像？

+   如何使用`VideoCapture`和`VideoWriter`类读取和写入视频？

+   如何通过网络（使用**实时流协议**（**RTSP**））访问摄像头和视频流？

# 技术要求

+   Microsoft Visual Studio、Xcode 或任何可以用于开发 C++程序的 IDE

+   Visual Studio Code 或任何其他可以用来编辑 CMake 文件、Python 源文件等的代码编辑器

+   Python 3.X

+   CMake 3.X

+   OpenCV 3.X

尝试使用您正在尝试学习的最新版本的技术和软件总是最好的。本书涵盖的主题，以及计算机视觉总体来说，也不例外，所以请确保下载并安装所提及软件的最新版本。

在必要时，提供了一组简短的安装和配置说明。您可以使用以下 URL 下载本章的源代码和示例：

[`github.com/PacktPublishing/Hands-On-Algorithms-for-Computer-Vision/tree/master/Chapter02`](https://github.com/PacktPublishing/Hands-On-Algorithms-for-Computer-Vision/tree/master/Chapter02)

# OpenCV 简介

**OpenCV**，或**开源计算机视觉**，是一组包含构建计算机视觉应用程序所需的类和函数的库、工具和模块。OpenCV 库被全球的计算机视觉开发者下载了数百万次，它速度快，经过优化，适用于现实生活中的项目（包括商业项目）。截至本书编写时，OpenCV 的最新版本是 3.4.1，这也是本书所有示例中使用的版本。OpenCV 支持 C/C++、Python 和 Java 语言，并且可以用于构建适用于桌面和移动操作系统的计算机视觉应用程序，包括 Windows、Linux、macOS、Android 和 iOS。

需要注意的是，OpenCV 库和 OpenCV 框架都用来指代 OpenCV，在计算机视觉社区中，这两个术语大多数时候是互换使用的。出于同样的原因，我们在这本书中也将互换使用这些术语。然而，严格来说，框架通常是指用于实现共同目标的关联库和工具集的术语，例如 OpenCV。

OpenCV 由以下两种类型的模块组成：

+   **主要模块**：这些模块默认包含在 OpenCV 发布版本中，它们包含了所有核心 OpenCV 功能以及用于图像处理任务、过滤、转换等更多功能的模块，我们将在本节中讨论这些功能。

+   **额外模块**：这些模块包括所有默认不包含在 OpenCV 库中的 OpenCV 功能，它们主要包含额外的计算机视觉相关功能。例如，额外模块包括用于文本识别和非自由特征检测器的库。请注意，我们的重点是主要模块，并涵盖其中的功能，但 wherever 可能有所帮助，我们也会尝试提及额外模块中的可能选项，供您自行研究。

# OpenCV 中的主要模块

如前所述，OpenCV 包含多个主模块，其中包含其所有核心和默认功能。以下是这些模块的列表：

+   `core`: 此模块包含所有核心 OpenCV 功能。例如，所有基本结构，包括`Mat`类（我们将在后面详细学习）和矩阵运算，都是嵌入到这个模块中的功能之一。

+   `imgproc`: 此模块包含所有图像处理功能，如滤波、变换和直方图。

+   `imgcodecs`: 此模块包括用于读取和写入图像的函数。

+   `videoio`: 此模块与`imgcodecs`模块类似，但根据其名称，它用于处理视频。

+   `highgui`: 本书将广泛使用此模块，它包含用于显示结果和创建 GUI 的所有功能。请注意，尽管`highgui`模块对于本书的目的以及在学习计算机视觉算法的同时可视化结果来说已经足够，但它并不适用于全面的应用。请参阅本章末尾的*进一步阅读*部分，以获取更多关于用于全面计算机视觉应用的正确 GUI 创建工具的参考资料。

+   `video`: 包含 OpenCV 的视频分析功能，如运动检测和跟踪、卡尔曼滤波以及臭名昭著的 CAM Shift 算法（用于对象跟踪）。

+   `calib3d`: 此模块包括校准和 3D 重建功能。此模块能力的知名示例是两个图像之间变换的估计。

+   `features2d`: 此模块包含支持的关键点检测和描述符提取算法。正如我们将在即将到来的章节中学习的那样，此模块包含一些最广泛使用的对象检测和分类算法。

+   `objdetect`: 如其名所示，此模块用于使用 OpenCV 进行对象检测。我们将在本书的最后一章学习这个模块包含的功能。

+   `dnn`: 与`objdetect`模块类似，此模块也用于对象检测和分类等目的。`dnn`模块在 OpenCV 主模块列表中相对较新，它包含了与深度学习相关的所有功能。

+   `ml`: 此机器学习模块包含用于处理分类和回归的类和函数。简单来说，所有严格相关的机器学习功能都包含在这个模块中。

+   `flann`: 这是 OpenCV 对**快速近似最近邻库**（**FLANN**）的接口。FLANN 包含一套广泛的优化算法，用于处理大型数据集中高维特征的最近邻搜索。这里提到的算法大多与其他模块中的算法结合使用，例如`features2d`。

+   `photo`：这是一个有趣的模块，用于处理与摄影相关的计算机视觉任务，它包含用于处理去噪、HDR 成像以及使用其邻域恢复照片区域的类和函数。

+   `stitching`：此模块包含用于图像拼接的类和函数。请注意，拼接本身是一个非常复杂的任务，它需要旋转估计和图像扭曲等功能，所有这些也都是这个非常有趣的 OpenCV 模块的一部分。

+   `shape`：此模块用于处理形状变换、匹配和距离相关主题。

+   `superres`：属于分辨率增强类别的算法包含在超分辨率模块中。

+   `videostab`：此模块包含用于视频稳定的算法。

+   `viz`：也称为 3D 可视化模块，它包含用于在 3D 可视化窗口上显示小部件的类和函数。此模块不会成为本书讨论的主题之一，但我们只是提一下。

除了我们刚刚提到的模块之外，OpenCV 还包含一些基于 CUDA（由 Nvidia 创建的 API）的主模块。这些模块很容易通过其名称区分，名称以单词 `cuda` 开头。由于这些模块的可用性完全取决于特定类型的硬件，并且几乎所有这些模块中的功能都由其他模块以某种方式覆盖，我们现在将跳过它们。但值得注意的是，如果您需要的算法已经实现在这些模块中，并且您的硬件满足它们的最小要求，那么使用 OpenCV 的 `cuda` 模块可以显著提高您应用程序的性能。

# 下载和构建/安装 OpenCV

OpenCV 大部分没有预构建和可直接使用的版本（本节中我们将讨论一些例外），类似于大多数开源库，它需要从源代码进行配置和构建。在本节中，我们将简要描述如何在计算机上构建（和安装）OpenCV。但首先，您需要将 OpenCV 源代码获取到您的计算机上。您可以使用以下链接进行此操作：

[`opencv.org/releases.html`](https://opencv.org/releases.html)

在本页面上，您可以找到 OpenCV 的发布版本。截至本书编写时，最新版本是 3.4.1，因此您应该下载它，或者如果有更高版本，则直接使用那个版本。

如以下截图所示，对于 OpenCV 的每个发布版本，都有各种可下载条目，例如 Win、iOS 和 Android 套件，但您应该下载源代码并根据自己的平台自行构建 OpenCV：

![](img/00013.jpeg)

OpenCV 3.4.1 默认提供 Android、iOS 和 64 位 MSVC14 和 MSVC15（与 Microsoft Visual C++ 2015 和 Microsoft Visual C++ 2017 相同）库的预构建版本。因此，如果您想为这些平台中的任何一个构建应用程序，您可以下载相关的包并完全跳过 OpenCV 的构建过程。

要从源代码构建 OpenCV，您需要在您的计算机上安装以下工具：

+   **支持 C++11 的 C/C++ 编译器**：在 Windows 上，这意味着任何较新的 Microsoft Visual C++ 编译器，例如 MSVC15（2017）或 MSVC14（2015）。在 Linux 操作系统上，您可以使用最新的 GCC，而在 macOS 上，您可以使用包含所有必需工具的 Xcode 命令行工具。

+   **CMake**：确保您使用最新版本的 CMake，例如 3.10，以确保与较新版本的 OpenCV 安全兼容，尽管您可以使用 CMake 3.1 及以后的版本。

+   **Python**：如果您打算使用 Python 编程语言，这一点尤为重要。

OpenCV 包含大量工具和库，您可以通过多种方式自定义您的构建。例如，您可以使用 Qt 框架、Intel **线程构建块**（**TBB**）、**Intel 集成性能原语**（**IPP**）和其他第三方库来进一步增强和自定义您的 OpenCV 构建，但由于我们将使用默认设置和工具集使用 OpenCV，所以我们忽略了上述第三方工具的要求列表。

在获取我们刚才提到的所有先决条件后，您可以通过使用 CMake 和相应的编译器配置和构建 OpenCV，具体取决于您的操作系统和所需的平台。

以下截图显示了具有默认配置集的 CMake 工具。通常，除非您想对 OpenCV 的构建应用自己的自定义设置，否则您不需要对配置进行任何更改：

![](img/00014.jpeg)

注意，当首次打开 CMake 时，您需要设置源代码文件夹和构建文件夹，分别如前述截图所示，即“源代码在哪里：”和“在哪里构建二进制文件：”。点击“配置”按钮后，您需要设置一个生成器并应用设置，然后按“生成”。

生成后，您只需使用终端或命令提示符实例切换到 CMake 输出文件夹，并执行以下命令：

```py
    make
    make install
```

请注意，运行这些命令中的每一个可能需要一些时间，具体取决于您的计算机速度和配置。另外，请注意，`make` 命令可能因您打算使用的工具集而异。例如，如果您使用 Microsoft Visual Studio，那么您需要将 `make` 替换为 `nmake`，或者如果您使用 MinGW，那么您必须将 `make` 替换为 `mingw32-make`。

在构建过程完成后，你可以开始使用 OpenCV。你需要注意的是配置你的 C++ 项目，以便它们可以使用你的 OpenCV 库和安装。

在 Windows 操作系统上，你需要确保你正在构建的应用程序可以访问 OpenCV DLL 文件。这可以通过将所有必需的 DLL 文件复制到与你的应用程序构建相同的文件夹中，或者简单地通过将 OpenCV DLL 文件的路径添加到 PATH 环境变量中来实现。在继续之前，请务必注意这一点，否则即使你的应用程序构建成功且在编译时没有报告任何问题，它们在执行时也可能会崩溃。

如果你打算使用 Python 来构建计算机视觉应用程序，那么事情对你来说将会非常简单，因为你可以使用 `pip`（包管理器）来安装 Python 的 OpenCV，使用以下命令：

```py
    pip install opencv-python
```

这将自动获取最新的 OpenCV 版本及其所有依赖项（如 `numpy`），或者如果你已经安装了 OpenCV，你可以使用以下命令来确保它升级到最新版本：

```py
    pip install --upgrade opencv-python
```

不言而喻，你需要一个正常工作的互联网连接才能使这些命令生效。

# 使用 OpenCV 与 C++ 或 Python

在本节中，我们将通过一个非常简单的示例来展示你如何在 C++ 或 Python 项目中使用 OpenCV，我们将这个示例称为 `HelloOpenCV`。你可能已经知道，这样的项目的目的是以下之一：

+   要开始使用一个全新的库，例如你之前从未使用过的 OpenCV

+   为了确保你的 OpenCV 安装是功能性的并且运行良好

所以，即使你不是 OpenCV 的初学者，仍然值得阅读以下说明并运行本节中的简单示例来测试你的 OpenCV 编译或安装。

我们将开始使用 OpenCV 的必要步骤，在 C++ 项目中：

1.  创建一个名为 `HelloOpenCV` 的新文件夹

1.  在这个文件夹内创建两个新的文本文件，并将它们命名为 `CMakeLists.txt` 和 `main.cpp`

1.  确保文件 `CMakeLists.txt` 包含以下内容：

```py
cmake_minimum_required(VERSION 3.1) 

project(HelloOpenCV) 

set(OpenCV_DIR "path_to_opencv") 
find_package(OpenCV REQUIRED) 
include_directories(${OpenCV_INCLUDE_DIRS}) 

add_executable(${PROJECT_NAME} "main.cpp") 

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}) 
```

在前面的代码中，你需要将 `"path_to_opencv"` 替换为包含 `OpenCVConfig.cmake` 和 `OpenCVConfig-version.cmake` 文件的文件夹路径，这个文件夹是你安装 OpenCV 库的地方。如果你使用的是 Linux 操作系统并且使用了预构建的 OpenCV 库，你可能不需要 `OpenCV` 文件夹的精确路径。

1.  至于 `main.cpp` 文件，请确保它包含以下内容，这是我们将会运行的实际的 C++ 代码：

```py
#include <iostream> 
#include <opencv2/opencv.hpp> 

using namespace std; 
using namespace cv; 

int main() 
{ 
    Mat image = imread("MyImage.png"); 
    if(!image.empty()) 
    { 
        imshow("image", image); 
        waitKey(); 
    } 
    else 
    { 
        cout << "Empty image!" << endl; 
    } 

    return 0; 
} 
```

我们将在本节和即将到来的章节中逐一介绍前面代码中使用的函数，然而，目前值得注意的是，这个程序正在尝试打开并显示存储在磁盘上的图像。如果成功，图像将一直显示，直到按下任意键，否则将显示`Empty image!`消息。请注意，在正常情况下，这个程序不应该崩溃，并且应该能够成功构建。所以，如果你遇到相反的情况，那么你需要回顾本章前面讨论的主题。

1.  我们的项目已经准备好了。现在，我们可以使用 CMake 来生成 Visual Studio 或其他任何我们想要的类型的项目（取决于我们将要使用的平台、编译器和 IDE），然后构建并运行它。请注意，CMake 只是用来确保创建了一个跨平台且与 IDE 无关的 C++项目。

通过运行此示例项目，你的输入图像（在本例中为`MyImage.png`）将被读取并显示，直到按下键盘上的任意键。如果在读取图像的过程中出现任何问题，则将显示`Empty image!`消息。

我们可以通过以下代码在 Python 中创建和运行相同的项目：

```py
import cv2 

image = cv2.imread("MyImage.png") 
if image is not None : 
    cv2.imshow("image", image) 
    cv2.waitKey() 

else: 
    print("Empty image!") 
```

在这里，相似之处非常明显。与 Python 版本中的相同代码完全相同的`imshow`和`waitKey`函数也被使用。正如之前提到的，现在不要担心任何函数的确切使用方式，只需确保你能够运行这些程序，无论是 C++还是 Python，或者两者都可以，并且能够看到显示的图像。

如果你能够成功运行本节中的`HelloOpenCV`示例项目，那么你就可以毫无问题地继续学习本章的下一节以及本书的下一章。如果你在讨论的主题上仍然遇到问题，或者你觉得你需要对这些主题有更深入的理解，你可以从章节的开始重新学习它们，或者更好的是，你可以参考本章末尾的“进一步阅读”部分中提到的附加书籍。

# 理解`Mat`类

请参考上一章中提供的计算机视觉中图像的描述，以及任何图像实际上都是一个具有给定宽度、高度、通道数和深度的矩阵。带着这个描述，我们可以说 OpenCV 的`Mat`类可以用来处理图像数据，并且它支持图像所需的所有属性，如宽度和高度。实际上，`Mat`类是一个 n 维数组，可以用来存储具有任何给定数据类型的单通道或多通道数据，并且它包含许多成员和方法，可以以多种方式创建、修改或操作它。

在本节中，我们将通过示例用例和代码示例来学习`Mat`类的一些最重要的成员和方法。

OpenCV C++ `Mat` 类在 Python 中的等效物最初不是 OpenCV 类，它由 `numpy.ndarray` 类型表示。NumPy 是一个 Python 库，包含广泛的数值算法和数学运算，并支持处理大型多维数组和矩阵。Python 中的 `numpy.ndarray` 类型被用作 `Mat` 的原因是因为它提供了最佳（如果不是相同的）成员和方法集，这些成员和方法是 OpenCV `Mat` 类在 C++ 中所需的。有关 `numpy.ndarray` 支持的成员和方法的完整列表，请参阅 NumPy 文档。

# 构建 Mat 对象

`Mat` 包含大约 20 个不同的构造函数，可以根据所需的初始化方式创建其实例。让我们看看一些最常用的构造函数和一些示例。

创建一个宽度为 `1920` 和高度为 `1080`，包含三个通道且包含 32 位浮点值的 `Mat` 对象（或类实例）如下所示：

```py
Mat image(1080, 1920, CV_32FC3); 
```

注意，`Mat` 构造函数中的 `type` 参数接受一种特殊类型的参数，即包含深度、类型和通道数的常量值。模式如下所示：

```py
CV_<depth><type>C<channels> 
```

`<depth>` 可以替换为 8、16、32 或 64，这代表用于存储每个像素中每个元素的位数。每个像素实际需要的位数可以通过将此数字乘以通道数来计算，或者换句话说，`<channels>`。最后，`<type>` 需要替换为 `U`、`S` 或 `F`，分别代表无符号整数、有符号整数和浮点值。例如，你可以使用以下方式创建宽度为 `800` 和高度为 `600` 像素的标准灰度和彩色图像。请注意，只有通道数不同，深度和类型参数代表 8 位无符号整数：

```py
Mat grayscaleImage(600, 800, CV_8UC1); 
Mat colorImage(600, 800, CV_8UC3); 
```

你可以使用以下构造函数创建一个宽度为 `W`、高度为 `H`、包含 8 位无符号整数元素的三个通道 RGB 图像，并初始化所有元素为 `R`、`G` 和 `B` 颜色值：

```py
int W = 800, H = 600, R = 50, G = 150, B = 200; 
Mat image(H, W, CV_8UC3, Scalar(R, G, B));
```

重要的是要注意，OpenCV 中颜色的默认顺序是 BGR（而不是 RGB），这意味着交换了 `B` 和 `R` 值。如果我们希望在应用程序运行时某点显示处理后的图像，这一点尤为重要。

因此，前述代码中标量初始化器的正确方式如下：

```py
Scalar(B, G, R) 
```

如果我们需要更高维度的 `Mat` 对象，可以使用以下方法。注意，在以下示例中，创建了一个七维度的 `Mat` 对象。每个维度的尺寸由 `sizes` 数组提供，并且高维 `Mat` 中的每个元素，称为 `hdm`，包含两个 32 位浮点值通道：

```py
const int dimensions = 7; 
const int sizes[dimensions] = {800, 600, 3, 2, 1, 1, 1}; 
Mat hdm(7, sizes, CV_32FC2); 
```

另一种实现相同功能的方法是使用 C++ 向量，如下所示：

```py
vector<int> sizes = {800, 600, 3, 2, 1, 1, 1}; 
Mat hdm(sizes, CV_32FC2); 
```

同样，你可以提供一个额外的`Scalar`参数来初始化`Mat`中的所有值。请注意，`Scalar`中的值的数量必须与通道数相匹配。例如，为了初始化前面提到的七维`Mat`中的所有元素，我们可以使用以下构造函数：

```py
Mat hdm(sizes, CV_32FC2, Scalar(1.25, 3.5)); 
```

`Mat`类允许我们使用已存储的图像数据来初始化它。使用此构造函数，你可以使你的`Mat`类包含`data`指针指向的数据。请注意，此构造函数不会创建原始数据的完整副本，它只使新创建的`Mat`对象指向它。这允许非常高效地初始化和构建`Mat`类，但显然的缺点是在不需要时没有处理内存清理，因此在使用此构造函数时需要格外小心：

```py
Mat image(1080, 1920, CV_8UC3, data);
```

注意，与之前的构造函数及其初始化器不同，这里的`data`不是一个`Scalar`，而是一个指向包含`1920` x `1080`像素，三通道图像数据的内存块的指针。使用指向内存空间的指针初始化`Mat`对象的方法也可以用于`Mat`类的更高维度。

最后一种构造函数类型，也是`Mat`类最重要的构造函数之一，是**感兴趣区域**（**ROI**）构造函数。此构造函数用于使用另一个`Mat`对象内的区域初始化`Mat`对象。让我们用一个例子来分解这一点。想象你有一张图片，你想要对该图片中的特定区域进行一些修改，换句话说，就是对 ROI 进行修改。你可以使用以下构造函数创建一个可以访问 ROI 的`Mat`类，并且对其所做的任何更改都将影响原始图像的相同区域。以下是你可以这样做的步骤：

```py
Mat roi(image, Rect(240, 140, 300, 300)); 
```

如果在`image`（它本身是一个`Mat`对象）包含以下图像左侧的图片时使用前面的构造函数，那么`roi`将能够访问该图像中突出显示的区域，并且它将包含右侧看到的图像：

![](img/00015.jpeg)

OpenCV 中的`Rect`类用于表示具有左上角点、宽度和高度的矩形。例如，前面代码示例中使用的`Rect`类具有左上角点为`240`和`140`，宽度为`300`像素，高度为`300`像素，如下所示：

```py
Rect(240, 140, 300, 300)
```

如前所述，以任何方式修改 ROI 将导致原始图像被修改。例如，我们可以将以下类似图像处理算法应用于`roi`（现在不必担心以下算法的性质，因为我们将在接下来的章节中了解更多关于它的内容，只需关注 ROI 的概念）：

```py
dilate(roi, roi, Mat(), Point(-1,-1), 5); 
```

如果我们尝试显示图像，结果将与以下类似。注意，在先前的图像中突出显示的区域在以下图像中已修改（膨胀），尽管我们是在`roi`上应用了更改，而不是图像本身：

![图片](img/00016.jpeg)

与使用对应于图像矩形区域的`Rect`类构造 ROI `Mat`对象类似，你也可以创建一个对应于原始`Mat`对象中列和行的 ROI。例如，先前的例子中的相同区域也可以使用以下构造函数中看到的范围进行访问：

```py
Mat roi(image, Range(140, 440), Range(240, 540)); 
```

OpenCV 中的`Range`类表示一个具有`start`和`end`的区间。根据`start`和`end`的值，可以检查`Range`类是否为空。在先前的构造函数中，第一个`Range`类对应于原始图像的行，从第`140`行到第`440`行。第二个`Range`对应于原始图像的列，从第`240`列到第`540`列。提供的两个区间的交集被认为是最终的 ROI。

# 删除 Mat 对象

可以通过使用其`release`函数来清理`Mat`对象，然而，由于`release`函数在`Mat`类的析构函数中被调用，通常没有必要调用此函数。需要注意的是，`Mat`类在多个指向它的对象之间共享相同的数据。这具有减少数据复制和内存使用的优势，并且由于所有引用计数都是自动完成的，你通常不需要关心任何事情。

在需要特别注意如何以及何时清理你的对象和数据的情况下，你需要格外小心，这种情况发生在你使用数据指针构造`Mat`对象时，如前所述。在这种情况下，调用`Mat`类的`release`函数或其析构函数将与用于构造它的外部数据无关，清理将完全由你负责。

# 访问像素

除了使用 ROI 访问图像矩形区域的像素，如前几节所述，还有一些其他方法可以实现相同的目标，甚至可以访问图像的个别像素。要能够访问图像中的任何单个像素（换句话说，一个`Mat`对象），你可以使用`at`函数，如下例所示：

```py
image.at<TYPE>(R, C) 
```

在前面的示例中，使用 `at` 函数时，`TYPE` 必须替换为一个有效的类型名，该类型名必须符合图像的通道数和深度。`R` 必须替换为像素的行号，而 `C` 替换为像素的列号，我们想要访问的像素。请注意，这与许多库中常用的像素访问方法略有不同，其中第一个参数是 *X*（或左），第二个参数是 *Y*（或顶）。所以基本上，参数在这里是颠倒的。以下是一些访问不同类型 `Mat` 对象中单个像素的示例。

按如下方式访问具有 8 位整数元素的单一通道 `Mat` 对象中的像素（灰度图像）：

```py
image.at<uchar>(R, C)
```

按如下方式访问具有浮点元素的单一通道 `Mat` 对象中的像素：

```py
image.at<float>(R, C) 
```

按如下方式访问具有 8 位整数元素的三个通道 `Mat` 对象中的像素：

```py
image.at<Vec3b>(R, C) 
```

在前面的代码中，使用了 `Vec3b`（3 字节向量）类型。OpenCV 定义了各种类似的向量类型以方便使用。以下是您可以使用 `at` 函数或用于其他目的的 OpenCV `Vec` 类型模式：

```py
Vec<N><Type> 
```

`<N>` 可以替换为 2、3、4、6 或 8（在 1 的情况下可以省略）并且它对应于 `Mat` 对象中的通道数。另一方面，`<Type>` 可以是以下之一，它们代表了每个像素每个通道中存储的数据类型：

+   `b` 代表 `uchar`（无符号字符）

+   `s` 代表 `short`（有符号字）

+   `w` 代表 `ushort`（无符号字）

+   `i` 代表 `int`

+   `f` 代表 `float`

+   `d` 代表 `double`

例如，`Vec4b` 可以用来访问具有 `uchar` 元素的四个通道 `Mat` 对象的像素，而 `Vec6f` 可以用来访问具有 `float` 元素的六个通道 `Mat` 对象的像素。重要的是要注意，`Vec` 类型可以像数组一样处理以访问单个通道。以下是一个如何使用 `uchar` 元素访问三个通道 `Mat` 对象的第二个通道的示例：

```py
image.at<Vec3b>(R, C)[1]
```

重要的是要注意，我们所说的“访问”既包括读取和写入像素及其各个通道。例如，以下示例是应用棕褐色滤镜到图像的一种方法：

```py
for(int i=0; i<image.rows; i++) 
{ 
    for(int j=0; j<image.cols; j++) 
    { 
        int inputBlue = image.at<Vec3b>(i,j)[0]; 
        int inputGreen = image.at<Vec3b>(i,j)[1]; 
        int inputRed = image.at<Vec3b>(i,j)[2]; 

        int red = 
                inputRed * 0.393 + 
                inputGreen * 0.769 + 
                inputBlue * 0.189; 

        if(red > 255 ) red = 255; 

        int green = 
                inputRed * 0.349 + 
                inputGreen * 0.686 + 
                inputBlue * 0.168; 

        if(green > 255) green = 255; 

        int blue = 
                inputRed * 0.272 + 
                inputGreen * 0.534 + 
                inputBlue * 0.131; 

        if(blue > 255) blue = 255; 

        image.at<Vec3b>(i,j)[0] = blue; 
        image.at<Vec3b>(i,j)[1] = green; 
        image.at<Vec3b>(i,j)[2] = red; 
    } 
} 
```

首先，这里需要注意的几点是图像的 `rows` 和 `cols` 成员，它们基本上代表了图像中的行数（或高度）和列数（或宽度）。同时注意 `at` 函数是如何被用来提取通道值和将更新后的值写入其中的。关于示例中用于乘法以获得正确棕褐色调的值，不必担心，因为它们是针对色调本身特定的，并且基本上任何类型的操作都可以应用于单个像素以改变它们。

以下图像展示了将前面的示例代码应用于三通道彩色图像的结果（左——原始图像，右——过滤后的图像）：

![](img/00017.jpeg)

访问图像中的像素的另一种方法是使用 `Mat` 类的 `forEach` 函数。`forEach` 可以用来并行地对所有像素应用操作，而不是逐个遍历它们。以下是一个简单的示例，展示了如何使用 `forEach` 将所有像素的值除以 `5`，如果它在灰度图像上执行，这将导致图像变暗：

```py
image.forEach<uchar>([](uchar &p, const int *) 
{ 
    p /= 5; 
}); 
```

在前面的代码中，第二个参数，或位置参数（这里不需要，因此省略）是像素位置的指针。

使用之前的 `for` 循环，我们需要编写以下代码：

```py
for(int i=0; i<image.rows; i++) 
    for(int j=0; j<image.cols; j++) 
        image.at<uchar>(i,j) /= 5; 
```

OpenCV 还允许使用类似于 STL 的迭代器来访问或修改图像中的单个像素。以下是一个使用类似于 STL 的迭代器编写的相同示例：

```py
MatIterator_<uchar> it_begin = image.begin<uchar>(); 
MatIterator_<uchar> it_end = image.end<uchar>(); 
for( ; it_begin != it_end; it_begin++) 
{ 
    *it_begin /= 5; 
}
```

值得注意的是，在前三个示例中的相同操作也可以通过以下简单的语句来完成：

```py
image /= 5; 
```

这是因为 OpenCV 中的 `Mat` 对象将此语句视为逐元素除法操作，我们将在接下来的章节中了解更多关于它的内容。以下图像展示了将前面的示例应用于灰度图像的结果（左——原始图像，右——修改后的图像）：

![](img/00018.gif)

显然，`forEach`、C++ 的 `for` 循环和类似于 STL 的迭代器都可以用来访问和修改 `Mat` 对象内的像素。我们将满足本节讨论的 `Mat` 类的函数和成员，但请确保探索它提供的用于以高效方式处理图像及其底层属性的庞大功能集。

# 读取和写入图像

OpenCV 允许使用 `imread` 函数从磁盘读取图像到 `Mat` 对象，我们在本章的前一个示例中简要使用过该函数。`imread` 函数接受一个输入图像文件名和一个 `flag` 参数，并返回一个填充有输入图像的 `Mat` 对象。输入图像文件必须具有 OpenCV 支持的图像格式之一。以下是一些最受欢迎的支持格式：

+   **Windows 位图**: `*.bmp`, `*.dib`

+   **JPEG 文件**: `*.jpeg`, `*.jpg`, `*.jpe`

+   **便携式网络图形**: `*.png`

+   **便携式图像格式**: `*.pbm`, `*.pgm`, `*.ppm`, `*.pxm`, `*.pnm`

+   **TIFF 文件**: `*.tiff`, `*.tif`

请确保始终检查 OpenCV 文档以获取完整和更新的列表，特别是对于可能适用于某些操作系统上某些格式的异常情况和注意事项。至于

`flag` 参数，它可以是 `ImreadModes` 枚举中的一个值或其组合，该枚举在 OpenCV 中定义。以下是一些最广泛使用且易于理解的条目：

+   `IMREAD_UNCHANGED`

+   `IMREAD_GRAYSCALE`

+   `IMREAD_COLOR`

+   `IMREAD_IGNORE_ORIENTATION`

例如，以下代码可以用来从磁盘读取图像，而不读取图像 EXIF 数据中存储的朝向值，并将其转换为灰度：

```py
Mat image = imread("MyImage.png", 
    IMREAD_GRAYSCALE | IMREAD_IGNORE_ORIENTATION);
```

**可交换图像文件格式** (**EXIF**) 是一种标准，用于为数字相机拍摄的照片添加标签和附加数据（或元数据）。这些标签可能包括制造商和相机型号以及拍照时相机的方向。OpenCV 能够读取某些标签（如方向）并对其进行解释，或者在前面示例代码的情况下，忽略它们。

在读取图像后，你可以调用 `empty` 来查看它是否成功读取。你也可以使用 `channels` 来获取通道数，`depth` 来获取深度，`type` 来获取图像类型，等等。或者，你可以调用 `imshow` 函数来显示它，就像我们在本章前面看到的那样。

类似地，`imreadmulti` 函数可以用来将多页图像读取到 `Mat` 对象的向量中。这里明显的区别是 `imreadmulti` 返回一个 `bool` 值，可以用来检查页面的成功读取，并通过引用填充传递给它的 `vector<Mat>` 对象。

要将图像写入磁盘上的文件，你可以使用 `imwrite` 函数。`imwrite` 函数接受要写入的文件名、一个 `Mat` 对象以及包含写入参数的 `int` 值 `vector`，在默认参数的情况下可以忽略。请参阅 OpenCV 中的以下枚举，以获取使用 `imwrite` 函数可以使用的完整参数列表，以改变写入过程的行为：

+   `ImwriteFlags`

+   `ImwriteEXRTypeFlags`

+   `ImwritePNGFlags`

+   `ImwritePAMFlags`

以下是一个示例代码，展示了如何使用 `imwrite` 函数将 `Mat` 对象写入磁盘上的图像文件。请注意，图像的格式是从提供的扩展名派生出来的，在这种情况下是 `png`：

```py
bool success = imwrite("c:/my_images/image1.png", image); 
cout << (success ? 
            "Image was saved successfully!" 
          : 
            "Image could not be saved!") 
     << endl; 
```

除了 `imread` 和 `imwrite` 函数，这些函数用于从磁盘上的图像文件中读取和写入图像外，你还可以使用 `imdecode` 和 `imencode` 函数来读取存储在内存缓冲区中的图像或向其写入。我们将这两个函数留给你去发现，并继续下一个主题，即使用 OpenCV 访问视频。

# 读取和写入视频

OpenCV，使用其 `videoio` 模块或更确切地说，使用 `VideoCapture` 和 `VideoWriter` 类，允许我们读取和写入视频文件。在视频的情况下，明显的区别是它们包含一系列连续的图像（或者更好的说法，帧），而不是单个图像。因此，它们通常在一个循环中读取和处理或写入，该循环覆盖视频中的全部或任何所需数量的帧。让我们从以下示例代码开始，展示如何使用 OpenCV 的 `VideoCapture` 类读取和播放视频：

```py
VideoCapture vid("MyVideo.mov"); 
// check if video file was opened correctly 
if(!vid.isOpened()) 
{ 
    cout << "Can't read the video file"; 
    return -1; 
} 
// get frame rate per second of the video file 
double fps = vid.get(CAP_PROP_FPS); 
if(fps == 0) 
{ 
    cout << "Can't get video FPS"; 
    return -1; 
} 
// required delay between frames in milliseconds 
int delay_ms = 1000.0 / fps; 
// infinite loop 
while(true) 
{ 
    Mat frame; 
    vid >> frame; 
    if(frame.empty()) 
        break; 
    // process the frame if necessary ... 
    // display the frame 
    imshow("Video", frame); 
    // stop playing if space is pressed 
    if(waitKey(delay_ms) == ' ') 
        break; 
} 
// release the video file 
vid.release(); 
```

如前述代码所示，视频文件名在构造 `VideoCapture` 类时传递。如果文件存在且您的计算机（和 OpenCV）支持该格式，则会自动打开视频文件。因此，您可以使用 `isOpened` 函数检查视频文件是否成功打开。之后，使用 `VideoCapture` 类的 `get` 函数检索已打开的视频文件的 **每秒帧率**（**FPS**）。`get` 是 `VideoCapture` 的一个极其重要的函数，它允许我们检索打开的视频文件的广泛属性。以下是 `get` 函数可以提供的示例参数，以获取所需的结果：

+   `CAP_PROP_POS_FRAMES`: 要解码或捕获的下一个帧的基于 0 的索引

+   `CAP_PROP_FRAME_WIDTH`: 视频流中帧的宽度

+   `CAP_PROP_FRAME_HEIGHT`: 视频流中帧的高度

+   `CAP_PROP_FPS`: 视频的帧率

+   `CAP_PROP_FRAME_COUNT`: 视频文件中的帧数

要获取完整的列表，您可以参考 OpenCV 中 `VideoCaptureProperties` 枚举文档。回到前面的示例代码，在通过 `get` 函数检索到帧率之后，它被用来计算两个帧之间所需的延迟，以便在播放时不会太快或太慢。然后，在一个无限循环中，使用 `>>` 操作符读取帧并显示。请注意，这个操作符实际上是使用 `VideoCapture` 函数（如 `read`、`grab` 和 `retrieve`）的简化和便捷方式。我们已经熟悉了 `imshow` 函数及其用法。另一方面，`waitKey` 函数的使用方式与之前略有不同，它可以用来插入延迟并等待按键。在这种情况下，之前计算出的所需延迟（以毫秒为单位）被插入到显示的帧之间，如果按下空格键，循环将中断。最后的 `release` 函数基本上是自我解释的。

除了我们使用 `VideoCapture` 类及其方法的方式之外，我们还可以调用其 `open` 函数来打开视频文件，如果我们不想将文件名传递给构造函数，或者如果视频文件在 `VideoCapture` 构造时不存在。`VideoCapture` 的另一个重要功能是 `set` 函数。将 `set` 视为 `get` 函数的完全相反，因为它允许设置 `VideoCapture` 和打开的视频文件的参数。尝试使用之前在 `VideoCaptureProperties` 枚举中提到的不同参数进行实验。

要能够写入视频文件，您可以使用与 `VideoCapture` 类非常类似的方式使用 `VideoWriter` 类。以下是一个示例，展示了如何创建 `VideoWriter` 对象：

```py
VideoWriter wrt("C:/output.avi", 
                VideoWriter::fourcc('M','J','P','G'), 
                30, Size(1920, 1080)); 
```

这将在`"C:/output.avi"`创建一个视频文件，分辨率为`1920` x `1080`像素，每秒`30`帧，准备好填充帧。但什么是`fourcc`？**四字符代码**（**FourCC**）简单地说是一个四字节的代码，表示（或更准确地说，是编解码器）将要用于记录视频文件的格式。在这个例子中，我们使用了一个最常见的 FourCC 值，但你可以在网上查找更全面的 FourCC 值及其规格列表。

在创建`VideoWriter`对象之后，你可以使用`<<`运算符或`write`函数将图像（与视频大小完全相同）写入视频文件：

```py
wrt << image; 
```

或者你也可以使用以下代码：

```py
vid.write(frame); 
```

最后，你可以调用`release`函数以确保视频文件被释放，并且所有更改都写入其中。

除了上述使用`VideoCapture`和`VideoWriter`类的方法之外，你还可以设置它们所使用的首选后端。有关更多信息，请参阅 OpenCV 文档中的`VideoCaptureAPIs`枚举。如果省略，如我们示例中所示，则使用计算机支持的默认后端。

# 访问摄像头

OpenCV 支持通过使用与访问视频文件相同的`VideoCapture`类来访问系统上可用的摄像头。唯一的区别是，你不需要将文件名传递给`VideoCapture`类的构造函数或其 open 函数，你必须提供一个对应于每个可用摄像头的基于 0 的索引号。例如，计算机上的默认摄像头可以通过以下示例代码访问和显示：

```py
VideoCapture cam(0); 
// check if camera was opened correctly 
if(!cam.isOpened()) 
    return -1; 

// infinite loop 
while(true) 
{ 
    Mat frame; 
    cam >> frame; 
    if(frame.empty()) 
        break; 

    // process the frame if necessary ... 

    // display the frame 
    imshow("Camera", frame); 

    // stop camera if space is pressed 
    if(waitKey(10) == ' ') 
        break; 
} 

cam.release(); 
```

如你所见，唯一的区别在于构造函数。这个`VideoCapture`类的实现允许用户以相同的方式处理任何类型的视频源，因此处理摄像头而不是视频文件时几乎可以编写相同的代码。这与下一节中描述的网络流的情况相同。

# 访问 RTSP 和网络流

OpenCV 允许用户从网络流中读取视频帧，或者更确切地说，从位于网络上的 RTSP 流中读取，例如本地网络或甚至互联网。要能够做到这一点，你需要将 RTSP 流的 URL 传递给`VideoCapture`构造函数或其 open 函数，就像它是一个本地硬盘上的文件一样。以下是最常见的模式和可以使用的示例 URL：

```py
rtsp://user:password@website.com/somevideo 
```

在这个 URL 中，`user`被替换为实际的用户名，`password`为该用户的密码，依此类推。如果网络流不需要用户名和密码，它们可以被省略。

# 类似于 Mat 的类

除了`Mat`类之外，OpenCV 还提供了一些其他与`Mat`非常相似的类，但它们的使用方式和时机不同。以下是可以替代或与`Mat`类一起使用的最重要的`Mat`类似类：

+   `Mat_`：这是`Mat`类的一个子类，但它提供了比`at`函数更好的访问方法，即使用`().Mat_`是一个模板类，显然需要在编译时提供元素的类型，这是`Mat`类本身可以避免的。

+   `Matx`：这最适合用于尺寸较小的矩阵，或者更准确地说，在编译时已知且尺寸较小的矩阵。

+   `UMat`：这是`Mat`类的一个较新实现，它允许我们使用 OpenCL 进行更快的矩阵操作。

使用`UMat`可以显著提高您计算机视觉应用程序的性能，但由于它与`Mat`类使用方式完全相同，因此在本书的章节中我们将忽略它；然而，在实际应用中，尤其是在实时计算机视觉应用中，您必须始终确保使用更优化、性能更好的类和函数，例如`UMat`。

# 摘要

我们已经涵盖了所有关键主题，这些主题使我们能够通过实际示例和真实场景轻松地掌握计算机视觉算法。我们本章从学习 OpenCV 及其整体结构开始，包括其模块和主要构建块。这帮助我们获得了我们将要工作的计算机视觉库的视角，但更重要的是，这让我们对处理计算机视觉算法时可能发生的情况有了概述。然后，我们学习了在哪里以及如何获取 OpenCV，以及如何在我们的系统上安装或构建它。我们还学习了如何创建、构建和运行使用 OpenCV 库的 C++和 Python 项目。然后，通过学习所有关于`Mat`类以及处理图像中的像素，我们学习了如何修改和显示图像。本章的最后部分包括了我们需要了解的所有关于从磁盘上的文件读取和写入图像的知识，无论是单页（或多页）图像还是视频文件，以及摄像头和网络流。我们通过学习 OpenCV Mat 家族中的一些其他类型来结束本章，这些类型可以帮助提高我们的应用程序。

现在我们已经了解了图像的真实本质（即它本质上是一个矩阵），我们可以从矩阵及其类似实体的可能操作开始。在下一章中，我们将学习所有属于计算机视觉领域的矩阵和数组操作。到下一章结束时，我们将能够使用 OpenCV 执行大量的像素级和图像级操作和转换。

# 问题

1.  列出三个额外的 OpenCV 模块及其用途。

1.  启用`BUILD_opencv_world`标志构建 OpenCV 3 会有什么影响？

1.  使用本章中描述的 ROI 像素访问方法，我们如何构建一个`Mat`类，使其能够访问中间像素，以及另一个图像中所有相邻的像素（即中间的九个像素）？

1.  除了本章中提到的之外，请说出 `Mat` 类的另一种像素访问方法。

1.  仅使用 `at` 方法和 `for` 循环编写一个程序，该程序创建三个单独的颜色图像，每个图像只包含从磁盘读取的 RGB 图像的一个通道。

1.  使用类似 STL 的迭代器，计算灰度图像的平均像素值。

1.  编写一个使用 `VideoCapture`、`waitKey` 和 `imwrite` 的程序，当按下 *S* 键时显示您的网络摄像头并保存可见图像。如果按下空格键，此程序将停止网络摄像头并退出。

# 进一步阅读

+   *使用 OpenCV 3 和 Qt5 进行计算机视觉*：[`www.packtpub.com/application-development/computer-vision-opencv-3-and-qt5`](https://www.packtpub.com/application-development/computer-vision-opencv-3-and-qt5)

+   *学习 Qt5*：[`www.packtpub.com/web-development/learn-qt-5`](https://www.packtpub.com/web-development/learn-qt-5)

+   *Qt5 项目*：[`www.packtpub.com/application-development/qt-5-projects`](https://www.packtpub.com/application-development/qt-5-projects)
