# 第二章：处理文件、摄像头和 GUI

安装 OpenCV 并运行示例很有趣，但在这个阶段，我们想亲自尝试。本章介绍了 OpenCV 的 I/O 功能。我们还讨论了项目概念以及这个项目的面向对象设计的开始，我们将在随后的章节中详细阐述。

通过从查看 I/O 能力和设计模式开始，我们将以制作三明治的方式构建我们的项目：从外到内。面包切片和涂抹，或者端点和胶水，在填充或算法之前。我们选择这种方法是因为计算机视觉主要是外向的——它考虑的是我们计算机之外的现实世界——我们希望通过一个公共接口将我们后续的所有算法工作应用到现实世界中。

# 基本 I/O 脚本

大多数 CV 应用程序需要获取图像作为输入。大多数也会生成图像作为输出。一个交互式 CV 应用程序可能需要一个摄像头作为输入源和一个窗口作为输出目标。然而，其他可能的源和目标包括图像文件、视频文件和原始字节。例如，原始字节可能通过网络连接传输，或者如果我们将过程图形纳入我们的应用程序，它们可能由算法生成。让我们看看这些可能性中的每一个。

## 读取/写入图像文件

OpenCV 提供了`imread()`和`imwrite()`函数，支持各种静态图像的文件格式。支持的格式因系统而异，但应始终包括 BMP 格式。通常，PNG、JPEG 和 TIFF 也应包括在支持的格式中。

让我们探索在 Python 和 NumPy 中图像表示的结构。

不论格式如何，每个像素都有一个值，但区别在于像素的表示方式。例如，我们可以通过简单地创建一个 2D NumPy 数组从头开始创建一个黑色方形图像：

```py
img = numpy.zeros((3,3), dtype=numpy.uint8)
```

如果我们将此图像打印到控制台，我们将获得以下结果：

```py
array([[0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]], dtype=uint8)
```

每个像素由一个单一的 8 位整数表示，这意味着每个像素的值在 0-255 范围内。

现在让我们使用`cv2.cvtColor`将此图像转换为**蓝绿红**（**BGR**）：

```py
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
```

让我们观察图像是如何变化的：

```py
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], dtype=uint8)
```

如您所见，每个像素现在由一个包含三个元素的数组表示，其中每个整数分别代表 B、G 和 R 通道。其他颜色空间，如 HSV，将以相同的方式表示，尽管值范围不同（例如，HSV 颜色空间的色调值范围为 0-180）以及通道数量不同。

您可以通过检查`shape`属性来检查图像的结构，该属性返回行、列和通道数（如果有多个通道）。

考虑以下示例：

```py
>>> img = numpy.zeros((3,3), dtype=numpy.uint8)
>>> img.shape
```

上述代码将打印`(3,3)`。如果您然后将图像转换为 BGR，形状将是`(3,3,3)`，这表明每个像素有三个通道。

图像可以从一种文件格式加载并保存到另一种格式。例如，让我们将图像从 PNG 转换为 JPEG：

```py
import cv2

image = cv2.imread('MyPic.png')
cv2.imwrite('MyPic.jpg', image)
```

### 注意

我们使用的 OpenCV 功能大多位于 `cv2` 模块中。你可能会遇到其他依赖 `cv` 或 `cv2.cv` 模块的 OpenCV 指南，这些是旧版本。Python 模块被称为 `cv2` 并不是因为它是 OpenCV 2.x.x 的 Python 绑定模块，而是因为它引入了一个更好的 API，它利用面向对象编程，而不是之前的 `cv` 模块，后者遵循更过程化的编程风格。

默认情况下，即使文件使用灰度格式，`imread()` 也返回 BGR 颜色格式的图像。BGR 代表与 **红-绿-蓝**（**RGB**）相同的颜色空间，但字节顺序相反。

可以选择指定 `imread()` 的模式为以下枚举之一：

+   `IMREAD_ANYCOLOR = 4`

+   `IMREAD_ANYDEPTH = 2`

+   `IMREAD_COLOR = 1`

+   `IMREAD_GRAYSCALE = 0`

+   `IMREAD_LOAD_GDAL = 8`

+   `IMREAD_UNCHANGED = -1`

例如，让我们将 PNG 文件作为灰度图像加载（在此过程中丢失任何颜色信息），然后将其保存为灰度 PNG 图像：

```py
import cv2

grayImage = cv2.imread('MyPic.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('MyPicGray.png', grayImage)
```

为了避免不必要的麻烦，在使用 OpenCV 的 API 时，至少使用图像的绝对路径（例如，Windows 上的 `C:\Users\Joe\Pictures\MyPic.png` 或 Unix 上的 `/home/joe/pictures/MyPic.png`），路径必须是相对的，除非它是绝对路径。图像的路径，除非是绝对路径，否则相对于包含 Python 脚本的文件夹，所以在前面的例子中，`MyPic.png` 必须与你的 Python 脚本在同一文件夹中，否则找不到图像。

无论模式如何，`imread()` 都会丢弃任何 alpha 通道（透明度）。`imwrite()` 函数要求图像以 BGR 或灰度格式存在，并且每个通道需要支持一定数量的位，该位数为输出格式所能支持。例如，`bmp` 需要每个通道 8 位，而 PNG 允许每个通道 8 或 16 位。

## 在图像和原始字节之间进行转换

从概念上讲，一个字节是一个介于 0 到 255 之间的整数。在所有今天的实时图形应用中，一个像素通常由每个通道一个字节表示，尽管其他表示也是可能的。

OpenCV 图像是一个 `.array` 类型的 2D 或 3D 数组。一个 8 位灰度图像是一个包含字节的 2D 数组。一个 24 位 BGR 图像是一个 3D 数组，它也包含字节值。我们可以通过使用表达式来访问这些值，例如 `image[0, 0]` 或 `image[0, 0, 0]`。第一个索引是像素的 *y* 坐标或行，`0` 表示顶部。第二个索引是像素的 *x* 坐标或列，`0` 表示最左边。如果适用，第三个索引代表一个颜色通道。

例如，在一个 8 位灰度图像中，如果左上角有一个白色像素，`image[0, 0]` 是 `255`。对于一个 24 位 BGR 图像，如果左上角有一个蓝色像素，`image[0, 0]` 是 `[255, 0, 0]`。

### 注意

作为使用表达式（如 `image[0, 0]` 或 `image[0, 0] = 128`）的替代，我们可以使用表达式，如 `image.item((0, 0))` 或 `image.setitem((0, 0), 128)`。后者的表达式对于单像素操作更有效率。然而，正如我们将在后续章节中看到的，我们通常希望对图像的大块区域进行操作，而不是单个像素。

假设图像每个通道有 8 位，我们可以将其转换为标准的 Python `bytearray`，它是一维的：

```py
byteArray = bytearray(image)
```

相反，如果 `bytearray` 中的字节顺序适当，我们可以将其转换并重塑为 `numpy.array` 类型，得到一个图像：

```py
grayImage = numpy.array(grayByteArray).reshape(height, width)
bgrImage = numpy.array(bgrByteArray).reshape(height, width, 3)
```

作为更完整的示例，让我们将包含随机字节的 `bytearray` 转换为灰度图像和 BGR 图像：

```py
import cv2
import numpy
import os

# Make an array of 120,000 random bytes.
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)

# Convert the array to make a 400x300 grayscale image.
grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite('RandomGray.png', grayImage)

# Convert the array to make a 400x100 color image.
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', bgrImage)
```

运行此脚本后，我们应该在脚本目录中有一对随机生成的图像，`RandomGray.png` 和 `RandomColor.png`。

### 注意

在这里，我们使用 Python 的标准 `os.urandom()` 函数生成随机原始字节，然后将其转换为 NumPy 数组。请注意，也可以直接（并且更高效地）使用语句生成随机 NumPy 数组，例如 `numpy.random.randint(0, 256, 120000).reshape(300, 400)`。我们使用 `os.urandom()` 的唯一原因是为了帮助演示从原始字节到转换的过程。

## 使用 numpy.array 访问图像数据

现在你已经更好地理解了图像的形成方式，我们可以开始对其进行基本操作。我们知道，在 OpenCV 中加载图像最简单（也是最常见）的方法是使用 `imread` 函数。我们也知道这将返回一个图像，实际上是一个数组（二维或三维，取决于你传递给 `imread()` 的参数）。

`y.array` 结构针对数组操作进行了很好的优化，并允许进行某些在普通 Python 列表中不可用的批量操作。这类 `.array` 类型特定的操作在 OpenCV 中的图像处理中非常有用。让我们从最基本的例子开始，逐步探索图像处理：假设你想操作 BGR 图像中坐标为 (0, 0) 的像素，并将其转换为白色像素。

```py
import cv

import numpy as np
img = cv.imread('MyPic.png')
img[0,0] = [255, 255, 255]
```

如果你使用标准的 `imshow()` 调用显示图像，你将在图像的左上角看到一个白色点。当然，这并不很有用，但它展示了可以完成的事情。现在让我们利用 `numpy.array` 的能力，以比普通 Python 数组快得多的速度对数组进行转换操作。

假设你想要改变特定像素的蓝色值，例如，坐标为（150，120）的像素。`numpy.array`类型提供了一个非常方便的方法，`item()`，它接受三个参数：x（或左）位置、y（或顶）以及数组中（x，y）位置的索引（记住，在 BGR 图像中，某个位置的数是一个包含 B、G 和 R 值的三个元素的数组，顺序如下）并返回索引位置的值。另一个`itemset()`方法将特定像素的特定通道的值设置为指定的值（`itemset()`接受两个参数：一个包含三个元素（x、y 和索引）的元组以及新值）。

在这个例子中，我们将（150，120）处的蓝色值从其当前值（127）更改为任意值 255：

```py
import cv
import numpy as  np
img = cv.imread('MyPic.png')
print img.item(150, 120, 0)  // prints the current value of B for that pixel
img.itemset( (150, 120, 0), 255)
print img.item(150, 120, 0)  // prints 255
```

记住，我们使用`numpy.array`做这件事有两个原因：`numpy.array`是一个针对这类操作进行了高度优化的库，而且我们通过 NumPy 优雅的方法而不是第一个示例中的原始索引访问获得了更易读的代码。

这段特定的代码本身并没有做什么，但它开启了一个可能性的世界。然而，建议你使用内置的过滤器和方法来操作整个图像；上述方法仅适用于小区域。

现在，让我们看看一个非常常见的操作，即操作通道。有时，你可能想要将特定通道（B、G 或 R）的所有值置零。

### 小贴士

使用循环来操作 Python 数组在运行时间上非常昂贵，应该尽量避免。使用数组索引允许高效地操作像素。这是一个昂贵且缓慢的操作，特别是如果你在操作视频时，你会发现输出会有抖动。然后，一个名为索引的功能就派上用场了。将图像中所有 G（绿色）值设置为`0`就像使用以下代码一样简单：

```py
import cv
import  as  np
img = cv.imread('MyPic.png')
img[:, :, 1] = 0
```

这是一段相当令人印象深刻且易于理解的代码。相关行是最后一行，它基本上指示程序从所有行和列中获取所有像素，并将结果值的三元素数组的索引一设置为`0`。如果你显示这张图片，你会注意到绿色完全消失。

通过使用 NumPy 的数组索引访问原始像素，我们可以做许多有趣的事情；其中之一是定义**感兴趣区域**（**ROI**）。一旦定义了区域，我们可以执行一系列操作，例如，将此区域绑定到一个变量上，然后甚至定义第二个区域并将第一个区域的值赋给它（在图像中将一部分图像复制到另一个位置）：

```py
import cv
import numpy as  np
img = cv.imread('MyPic.png')
my_roi = img[0:100, 0:100]
img[300:400, 300:400] = my_roi
```

确保两个区域在大小上是一致的非常重要。如果不是，NumPy 会（正确地）抱怨两个形状不匹配。

最后，我们可以从`numpy.array`中获得一些有趣的细节，例如使用此代码获取图像属性：

```py
import cv
import numpy  as  np
img = cv.imread('MyPic.png')
print img.shape
print img.size
print img.dtype
```

这三个属性按此顺序排列：

+   **形状**：NumPy 返回一个包含宽度、高度以及如果图像是彩色的则包含通道数的元组。这对于调试图像类型很有用；如果图像是单色或灰度，则不会包含通道值。

+   **大小**：此属性指图像的像素大小。

+   **数据类型**：此属性指用于图像的数据类型（通常是未签名的整数类型及其支持的字节数，即`uint8`）。

总而言之，强烈建议您在处理 OpenCV 时熟悉 NumPy，特别是`numpy.array`，因为它是使用 Python 进行图像处理的基础。

## 读取/写入视频文件

OpenCV 提供了`VideoCapture`和`VideoWriter`类，支持各种视频文件格式。支持的格式因系统而异，但应始终包括 AVI。通过其`read()`方法，`VideoCapture`类可以轮询新帧，直到达到视频文件的末尾。每个帧都是以 BGR 格式的图像。

相反，可以将图像传递给`VideoWriter`类的`write()`方法，该方法将图像追加到`VideoWriter`中的文件。让我们看看一个例子，从 AVI 文件中读取帧并使用 YUV 编码写入另一个文件：

```py
import cv2

videoCapture = cv2.VideoCapture('MyInputVid.avi')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
    'MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

success, frame = videoCapture.read()
while success: # Loop until there are no more frames.
    videoWriter.write(frame)
    success, frame = videoCapture.read()
```

`VideoWriter`类构造函数的参数值得特别注意。必须指定视频的文件名。任何具有此名称的现有文件都将被覆盖。还必须指定视频编解码器。可用的编解码器可能因系统而异。以下是一些选项：

+   `cv2.VideoWriter_fourcc('I','4','2','0')`：此选项是不压缩的 YUV 编码，4:2:0 色度子采样。这种编码广泛兼容，但生成的文件很大。文件扩展名应为`.avi`。

+   `cv2.VideoWriter_fourcc('P','I','M','1')`：此选项是 MPEG-1。文件扩展名应为`.avi`。

+   `cv2.VideoWriter_fourcc('X','V','I','D')`：此选项是 MPEG-4，如果您希望生成的视频大小为平均大小，这是一个首选选项。文件扩展名应为`.avi`。

+   `cv2.VideoWriter_fourcc('T','H','E','O')`：此选项是 Ogg Vorbis。文件扩展名应为`.ogv`。

+   `cv2.VideoWriter_fourcc('F','L','V','1')`：此选项是 Flash 视频。文件扩展名应为`.flv`。

必须指定帧率和帧大小。由于我们是从另一个视频复制视频帧，这些属性可以从`VideoCapture`类的`get()`方法中读取。

## 捕获相机帧

相机帧流也由`VideoCapture`类表示。然而，对于相机，我们通过传递相机的设备索引而不是视频的文件名来构建一个`VideoCapture`类。让我们考虑一个例子，从相机捕获 10 秒的视频并将其写入 AVI 文件：

```py
import cv2

cameraCapture = cv2.VideoCapture(0)
fps = 30 # an assumption
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
    'MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1
while success and numFramesRemaining > 0:
    videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
cameraCapture.release()
```

很不幸，`VideoCapture` 类的 `get()` 方法并不能返回相机帧率的准确值；它总是返回 `0`。在官方文档[`docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html`](http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html)中写道：

> *"当查询 `VideoCapture` 类后端不支持的一个属性时，返回值 `0`。"*

这通常发生在只支持基本功能的驱动程序的系统上。

为了为相机创建一个合适的 `VideoWriter` 类，我们不得不要么对帧率做出假设（就像我们在之前的代码中所做的那样），要么使用计时器来测量它。后者方法更好，我们将在本章后面讨论。

相机的数量及其顺序当然是系统相关的。不幸的是，OpenCV 并没有提供查询相机数量或其属性的方法。如果使用无效的索引来构造 `VideoCapture` 类，该类将不会输出任何帧；其 `read()` 方法将返回 `(false, None)`。为了避免尝试从未正确打开的 `VideoCapture` 中检索帧，一个很好的方法是使用 `VideoCapture.isOpened` 方法，它返回一个布尔值。

当我们需要同步一组相机或多头相机（如立体相机或 Kinect）时，`read()` 方法是不合适的。这时，我们使用 `grab()` 和 `retrieve()` 方法代替。对于一组相机，我们使用以下代码：

```py
success0 = cameraCapture0.grab()
success1 = cameraCapture1.grab()
if success0 and success1:
    frame0 = cameraCapture0.retrieve()
    frame1 = cameraCapture1.retrieve()
```

## 在窗口中显示图像

OpenCV 中最基本的一个操作就是显示图像。这可以通过 `imshow()` 函数实现。如果你来自任何其他 GUI 框架的背景，你可能会认为调用 `imshow()` 来显示图像就足够了。这仅部分正确：图像会被显示，然后立即消失。这是设计上的考虑，以便在处理视频时能够不断刷新窗口框架。以下是一个显示图像的非常简单的示例代码：

```py
import cv2
import numpy as np

img = cv2.imread('my-image.png')
cv2.imshow('my image', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

`imshow()` 函数接受两个参数：我们想要在其中显示图像的窗口名称，以及图像本身。当我们探讨在窗口中显示帧时，我们将更详细地讨论 `waitKey()`。

命名为 `destroyAllWindows()` 的函数会销毁 OpenCV 创建的所有窗口。

## 在窗口中显示相机帧

OpenCV 允许使用 `namedWindow()`、`imshow()` 和 `destroyWindow()` 函数创建、重绘和销毁命名窗口。此外，任何窗口都可以通过 `waitKey()` 函数捕获键盘输入，通过 `setMouseCallback()` 函数捕获鼠标输入。让我们看看一个示例，其中我们展示了实时相机输入的帧：

```py
import cv2

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)

print 'Showing camera feed. Click window or press any key to stop.'
success, frame = cameraCapture.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()

cv2.destroyWindow('MyWindow')
cameraCapture.release()
```

`waitKey()` 的参数是等待键盘输入的毫秒数。返回值是 `-1`（表示没有按键被按下）或一个 ASCII 键码，例如 `27` 对应于 *Esc*。有关 ASCII 键码的列表，请参阅 [`www.asciitable.com/`](http://www.asciitable.com/)。此外，请注意，Python 提供了一个标准函数 `ord()`，可以将字符转换为它的 ASCII 键码。例如，`ord('a')` 返回 `97`。

### 小贴士

在某些系统上，`waitKey()` 可能返回一个编码了不仅仅是 ASCII 键码的值。（已知当 OpenCV 使用 GTK 作为其后端 GUI 库时，Linux 上会发生一个错误。）在所有系统上，我们可以确保通过从返回值中读取最后一个字节来仅提取 ASCII 键码，如下所示：

```py
keycode = cv2.waitKey(1)
if keycode != -1:
    keycode &= 0xFF
```

OpenCV 的窗口函数和 `waitKey()` 是相互依赖的。只有当调用 `waitKey()` 时，OpenCV 窗口才会更新，并且只有当 OpenCV 窗口获得焦点时，`waitKey()` 才会捕获输入。

传递给 `setMouseCallback()` 的鼠标回调应接受五个参数，如我们的代码示例所示。回调的 `param` 参数被设置为 `setMouseCallback()` 的可选第三个参数。默认情况下，它是 `0`。回调的事件参数是以下动作之一：

+   `cv2.EVENT_MOUSEMOVE`: 此事件表示鼠标移动

+   `cv2.EVENT_LBUTTONDOWN`: 此事件表示左键按下

+   `cv2.EVENT_RBUTTONDOWN`: 这表示右键按下

+   `cv2.EVENT_MBUTTONDOWN`: 这表示中间按钮按下

+   `cv2.EVENT_LBUTTONUP`: 这表示左键释放

+   `cv2.EVENT_RBUTTONUP`: 此事件表示右键释放

+   `cv2.EVENT_MBUTTONUP`: 此事件表示中间按钮释放

+   `cv2.EVENT_LBUTTONDBLCLK`: 此事件表示左键被双击

+   `cv2.EVENT_RBUTTONDBLCLK`: 这表示右键被双击

+   `cv2.EVENT_MBUTTONDBLCLK`: 这表示中间按钮被双击

鼠标回调的标志参数可能是以下事件的位运算组合：

+   `cv2.EVENT_FLAG_LBUTTON`: 此事件表示左键被按下

+   `cv2.EVENT_FLAG_RBUTTON`: 此事件表示右键被按下

+   `cv2.EVENT_FLAG_MBUTTON`: 此事件表示中间按钮被按下

+   `cv2.EVENT_FLAG_CTRLKEY`: 此事件表示按下 *Ctrl* 键

+   `cv2.EVENT_FLAG_SHIFTKEY`: 此事件表示按下 *Shift* 键

+   `cv2.EVENT_FLAG_ALTKEY`: 此事件表示按下 *Alt* 键

不幸的是，OpenCV 不提供处理窗口事件的方法。例如，当窗口的关闭按钮被点击时，我们无法停止我们的应用程序。由于 OpenCV 有限的的事件处理和 GUI 功能，许多开发者更喜欢将其与其他应用程序框架集成。在本章的后面部分，我们将设计一个抽象层，以帮助将 OpenCV 集成到任何应用程序框架中。

# Project Cameo（人脸追踪和图像处理）

OpenCV 通常通过一种类似于食谱的方法来研究，它涵盖了大量的算法，但没有关于高级应用程序开发的内容。在一定程度上，这种方法是可以理解的，因为 OpenCV 的潜在应用非常多样。OpenCV 被用于广泛的领域：照片/视频编辑器、动作控制游戏、机器人的 AI，或者记录参与者眼动行为的心理学实验。在如此不同的用例中，我们真的能够研究出一套有用的抽象吗？

我相信我们可以，而且越早开始创建抽象，越好。我们将围绕单个应用程序来结构化我们对 OpenCV 的研究，但在每个步骤中，我们将设计这个应用程序的一个组件，使其可扩展和可重用。

我们将开发一个交互式应用程序，该应用程序在实时摄像头输入上执行面部跟踪和图像操作。这类应用程序涵盖了 OpenCV 的广泛功能，并挑战我们创建一个高效、有效的实现。

具体来说，我们的应用程序将执行实时面部融合。给定两个摄像头输入流（或者，可选地，预先录制的视频输入），应用程序将把一个流中的面部叠加到另一个流中的面部上。将应用滤镜和扭曲，使这个混合场景看起来和感觉上统一。用户应该体验到参与现场表演的感觉，进入另一个环境和角色。这种用户体验在像迪士尼乐园这样的游乐园中很受欢迎。

在这样的应用程序中，用户会立即注意到缺陷，例如帧率低或跟踪不准确。为了获得最佳结果，我们将尝试使用传统成像和深度成像的几种方法。

我们将把我们的应用程序命名为 Cameo。在珠宝中，Cameo 是指一个人的小肖像，或者在电影中是指名人扮演的非常短暂的角色。

# Cameo – 面向对象设计

Python 应用程序可以编写为纯过程式风格。这通常用于小型应用程序，例如我们之前讨论的基本 I/O 脚本。然而，从现在开始，我们将使用面向对象风格，因为它促进了模块化和可扩展性。

从我们对 OpenCV I/O 功能的概述中，我们知道所有图像都是相似的，无论它们的来源或目的地。无论我们如何获取图像流或将其发送到何处作为输出，我们都可以将相同的应用特定逻辑应用于这个流中的每一帧。在像 Cameo 这样的应用程序中，分离 I/O 代码和应用代码变得特别方便，因为它使用多个 I/O 流。

我们将创建名为`CaptureManager`和`WindowManager`的类，作为 I/O 流的高级接口。我们的应用程序代码可以使用`CaptureManager`读取新帧，并且可选地将每个帧派发到一个或多个输出，包括静态图像文件、视频文件和窗口（通过`WindowManager`类）。`WindowManager`类允许我们的应用程序代码以面向对象的方式处理窗口和事件。

`CaptureManager`和`WindowManager`都是可扩展的。我们可以实现不依赖于 OpenCV 进行 I/O 的版本。实际上，*附录 A*，*与 Pygame 集成*，*使用 Python 的 OpenCV 计算机视觉*，使用了一个`WindowManager`子类。

## 使用 CaptureManager 从管理器中抽象视频流。

正如我们所见，OpenCV 可以从视频文件或摄像头捕获、显示和记录一系列图像，但在每种情况下都有一些特殊考虑。我们的`CaptureManager`类抽象了一些差异，并提供了一个更高级别的接口，将捕获流中的图像派发到一个或多个输出——静态图像文件、视频文件或窗口。

`CaptureManager`类使用`VideoCapture`类初始化，并具有`enterFrame()`和`exitFrame()`方法，这些方法通常在应用程序主循环的每次迭代中调用。在调用`enterFrame()`和`exitFrame()`之间，应用程序可以（任意次数）设置`channel`属性并获取`frame`属性。`channel`属性最初为`0`，只有多头摄像头使用其他值。`frame`属性是当调用`enterFrame()`时对应当前通道状态的图像。

`CaptureManager`类还具有`writeImage()`、`startWritingVideo()`和`stopWritingVideo()`方法，这些方法可以在任何时候调用。实际的文件写入将推迟到`exitFrame()`。此外，在`exitFrame()`方法中，`frame`属性可能会在窗口中显示，具体取决于应用程序代码是否提供了一个`WindowManager`类，无论是作为`CaptureManager`构造函数的参数，还是通过设置`previewWindowManager`属性。

如果应用程序代码操作`frame`，则这些操作将反映在记录的文件和窗口中。`CaptureManager`类有一个名为`shouldMirrorPreview`的构造函数参数和属性，如果我们要在窗口中镜像（水平翻转）`frame`但不在记录的文件中，则该参数应为`True`。通常，当面对摄像头时，用户更喜欢镜像的实时摄像头流。

回想一下，`VideoWriter`类需要一个帧率，但 OpenCV 并没有提供任何方法来获取摄像头的准确帧率。`CaptureManager`类通过使用帧计数器和 Python 的标准`time.time()`函数来估计帧率来绕过这个限制。这种方法并不是万无一失的。根据帧率波动和系统依赖的`time.time()`实现，在某些情况下，估计的准确性可能仍然很差。然而，如果我们部署到未知的硬件上，这比仅仅假设用户的摄像头具有特定的帧率要好。

让我们创建一个名为`managers.py`的文件，该文件将包含我们的`CaptureManager`实现。这个实现相当长。因此，我们将分几个部分来看它。首先，让我们添加导入、构造函数和属性，如下所示：

```py
import cv2
import numpy
import time

class CaptureManager(object):

    def __init__(self, capture, previewWindowManager = None,
                 shouldMirrorPreview = False):

        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._framesElapsed = long(0)
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage (self):

        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None
```

注意，大多数的`member`变量都是非公开的，这可以通过变量名前的下划线前缀来表示，例如`self._enteredFrame`。这些非公开变量与当前帧的状态以及任何文件写入操作相关。正如之前讨论的，应用程序代码只需要配置一些事情，这些事情作为构造函数参数和可设置的公共属性来实现：摄像头通道、窗口管理器以及是否镜像摄像头预览的选项。

本书假设读者对 Python 有一定程度的熟悉；然而，如果你对那些`@`注解（例如，`@property`）感到困惑，请参考 Python 文档中关于`decorators`的部分，这是语言的一个内置特性，允许一个函数被另一个函数包装，通常用于在应用程序的多个地方应用用户定义的行为（参考[`docs.python.org/2/reference/compound_stmts.html#grammar-token-decorator`](https://docs.python.org/2/reference/compound_stmts.html#grammar-token-decorator))）。

### 注意

Python 没有私有成员变量的概念，单下划线前缀（`_`）只是一个约定。

根据这个约定，在 Python 中，以单个下划线为前缀的变量应被视为受保护的（只能在类及其子类中访问），而以双下划线为前缀的变量应被视为私有的（只能在类内部访问）。

继续我们的实现，让我们将`enterFrame()`和`exitFrame()`方法添加到`managers.py`中：

```py
    def enterFrame(self):
        """Capture the next frame, if any."""

        # But first, check that any previous frame was exited.
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame (self):
        """Draw to the window. Write to files. Release the frame."""

        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return

        # Update the FPS estimate and related variables.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate =  self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # Draw to the window, if any.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # Write to the image file, if any.
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # Write to the video file, if any.
        self._writeVideoFrame()

        # Release the frame.
        self._frame = None
        self._enteredFrame = False
```

注意，`enterFrame()`的实现只是获取（同步）一个帧，而实际从通道检索则推迟到后续读取`frame`变量。`exitFrame()`的实现从当前通道获取图像，估算帧率，通过窗口管理器（如果有）显示图像，并满足任何待处理的将图像写入文件的请求。

其他几个方法也与文件写入有关。为了完成我们的类实现，让我们将剩余的文件写入方法添加到`managers.py`中：

```py
    def writeImage(self, filename):
        """Write the next exited frame to an image file."""
        self._imageFilename = filename

    def startWritingVideo(
            self, filename,
            encoding = cv2.VideoWriter_fourcc('I','4','2','0')):
        """Start writing exited frames to a video file."""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo (self):
        """Stop writing exited frames to a video file."""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

def _writeVideoFrame(self):

        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(
                        cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(
                        cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename, self._videoEncoding,
                fps, size)

        self._videoWriter.write(self._frame)
```

`writeImage()`、`startWritingVideo()`和`stopWritingVideo()`公共方法只是记录文件写入操作的参数，而实际的写入操作则推迟到`exitFrame()`的下一次调用。非公共方法`_writeVideoFrame()`以我们早期脚本中熟悉的方式创建或追加视频文件。（见*读取/写入视频文件*部分。）然而，在帧率未知的情况下，我们在捕获会话的开始处跳过一些帧，以便我们有时间建立对帧率的估计。

尽管我们当前的`CaptureManager`实现依赖于`VideoCapture`，但我们可以实现不使用 OpenCV 作为输入的其他实现。例如，我们可以创建一个子类，它通过套接字连接实例化，其字节流可以解析为图像流。我们还可以创建一个使用第三方相机库的子类，该库具有与 OpenCV 提供的不同硬件支持。然而，对于 Cameo，我们的当前实现是足够的。

## 使用`managers.WindowManager`管理窗口和键盘

正如我们所见，OpenCV 提供了创建窗口、销毁窗口、显示图像和处理事件的函数。这些函数不是窗口类的成员方法，而是需要将窗口的名称作为参数传递。由于这个接口不是面向对象的，它不符合 OpenCV 的一般风格。此外，它可能与我们可能最终想要使用的其他窗口或事件处理接口不兼容。

为了面向对象和适应性，我们将此功能抽象成一个具有`createWindow()`、`destroyWindow()`、`show()`和`processEvents()`方法的`WindowManager`类。作为一个属性，`WindowManager`类有一个名为`keypressCallback`的函数对象，该对象（如果非`None`）在`processEvents()`中响应任何按键时被调用。`keypressCallback`对象必须接受一个单一参数，例如 ASCII 键码。

让我们在`managers.py`中添加以下`WindowManager`的实现：

```py
class WindowManager(object):

    def __init__(self, windowName, keypressCallback = None):
        self.keypressCallback = keypressCallback

        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow (self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow (self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents (self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # Discard any non-ASCII info encoded by GTK.
            keycode &= 0xFF
            self.keypressCallback(keycode)
```

我们当前的实施方案仅支持键盘事件，这对 Cameo 来说将足够。然而，我们可以修改`WindowManager`以支持鼠标事件。例如，类的接口可以扩展以包括一个`mouseCallback`属性（以及可选的构造函数参数），但其他方面可以保持不变。通过添加回调属性，我们可以使用除 OpenCV 之外的事件框架以相同的方式支持其他事件类型。

*附录 A*，*与 Pygame 集成*，*使用 Python 的 OpenCV 计算机视觉*展示了使用 Pygame 的窗口处理和事件框架实现的`WindowManager`子类，而不是使用 OpenCV 的。这个实现通过正确处理退出事件（例如，当用户点击窗口的关闭按钮时）改进了基本的`WindowManager`类。潜在地，许多其他事件类型也可以通过 Pygame 来处理。

## 应用所有内容到 cameo.Cameo

我们的应用程序由一个`Cameo`类表示，包含两个方法：`run()`和`onKeypress()`。在初始化时，`Cameo`类创建一个带有`onKeypress()`作为回调的`WindowManager`类，以及使用摄像头和`WindowManager`类的`CaptureManager`类。当调用`run()`时，应用程序执行一个主循环，在该循环中处理帧和事件。由于事件处理的结果，可能会调用`onKeypress()`。空格键会触发截图，*Tab*键会导致屏幕录制（视频录制）开始/停止，而*Esc*键会导致应用程序退出。

在`managers.py`相同的目录下，让我们创建一个名为`cameo.py`的文件，包含以下`Cameo`的实现：

```py
import cv2
from managers import WindowManager, CaptureManager

class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            # TODO: Filter the frame (Chapter 3).

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress (self, keycode):
        """Handle a keypress.

        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        escape -> Quit.

        """
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(
                    'screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

if __name__=="__main__":
    Cameo().run()
```

当运行应用程序时，请注意，实时摄像头视频流是镜像的，而截图和屏幕录制则不是。这是预期的行为，因为我们初始化`CaptureManager`类时传递了`True`给`shouldMirrorPreview`。

到目前为止，我们除了镜像预览之外，没有以任何方式操作帧。我们将在第三章，*过滤图像*中开始添加更多有趣的效果。

# 摘要

到目前为止，我们应该有一个显示摄像头视频流、监听键盘输入，并且（在命令下）记录截图或屏幕录制的应用程序。我们现在准备通过在每一帧的开始和结束之间插入一些图像过滤代码（第三章，*过滤图像*）来扩展应用程序。可选地，我们也准备集成其他摄像头驱动程序或应用程序框架（*附录 A*，*与 Pygame 集成*，*使用 Python 的 OpenCV 计算机视觉*），除了 OpenCV 支持的那些。

我们现在也拥有了处理图像和理解通过 NumPy 数组进行图像操作原理的知识。这为理解下一个主题，过滤图像，奠定了完美的基础。
