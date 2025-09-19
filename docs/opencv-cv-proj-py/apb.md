# 附录 B：为自定义目标生成 Haar 级联

该附录显示了如何生成 Haar 级联 XML 文件，例如第 4 章“使用 Haar 级联跟踪人脸”时所使用的 XML 文件。 通过生成自己的级联文件，我们可以潜在地跟踪任何模式或对象，而不仅仅是面部。 但是，好的结果可能不会很快出现。 我们必须仔细收集图像，配置脚本参数，执行实际测试并进行迭代。 可能涉及大量的人工时间和处理时间。

# 收集正面和负面的训练图像

你知道抽认卡的教学法吗？ 这是一种向幼儿教授单词和识别技巧的方法。 老师给全班同学展示了一系列图片，并说了以下内容：

> “这是牛。Mo！这是马。邻居！”

级联文件的生成方式类似于抽认卡教学法。 要了解如何识别母牛，计算机需要预先识别为母牛的**正面训练图像**和预先识别为非牛的**负面训练图像**。 作为训练器，我们的第一步是收集这两套图像。

在确定要使用多少个正面训练图像时，我们需要考虑用户查看目标的各种方式。 理想，最简单的情况是目标是始终在平坦表面上的 2D 图案。 在这种情况下，一个正面的训练图像可能就足够了。 但是，在其他情况下，可能需要数百甚至数千张训练图像。 假设目标是您所在国家的国旗。 当在文档上打印时，标志的外观可能可预测，但是当在顺风飘扬的织物上打印时，标志的外观变化很大。 诸如人脸之类的自然 3D 目标的外观范围可能更大。 理想情况下，我们的一组正面训练图像应代表我们的相机可能捕获的许多变化。 可选地，我们的任何正面训练图像都可以包含目标的多个实例。

对于我们的负面训练集，我们希望大量图像不包含目标的任何实例，但确实包含相机可能捕获的其他内容。 例如，如果一面旗帜是我们的目标，那么我们的负面训练集可能包括各种天气情况下的天空照片。 （天空不是旗帜，但经常在旗帜后面看到。）不过不要假设太多。 如果相机的环境无法预测，并且在许多设置中都出现了目标，请使用各种各样的负面训练图像。 考虑构建一套通用的环境图像，您可以在多个训练方案中重复使用这些图像。

# 查找用于训练的可执行文件

为了使级联训练尽可能自动化，OpenCV 提供了两个可执行文件。 它们的名称和位置取决于操作系统和 OpenCV 的特定设置，如以下两节所述。

## 在 Windows 上

Windows 上的两个可执行文件称为`ONopencv_createsamples.exe`和`ONopencv_traincascade.exe`。 它们不是预建的。 而是，仅当您从源代码编译 OpenCV 时，它们才存在。 根据您在第 1 章，“设置 OpenCV”中选择的编译方法，它们的父文件夹是以下文件夹之一：

*   MinGW：`<unzip_destination>\bin`
*   Visual Studio 或 Visual C++ Express：`<unzip_destination>\bin\Release`

如果要将可执行文件的文件夹添加到系统的`Path`变量中，请参考第 1 章“设置 OpenCV”的“在 Windows XP，Windows Vista，Windows 7 和 Windows 8 上选择信息框”部分中的说明。 否则，请注意可执行文件的完整路径，因为我们将需要在运行它们时使用它。

## 在 Mac，Ubuntu 和其他类似 Unix 的系统上

Mac 上的两个可执行文件 Ubuntu 和其他类似 Unix 的系统称为`opencv_createsamples`和`opencv_traincascade`。 它们的父文件夹是以下文件夹之一，具体取决于您的系统和在第 1 章，“设置 OpenCV”中选择的方法：

*   带有 MacPorts 的 Mac：`/opt/local/bin`
*   带有 Homebrew 的 Mac：`/opt/local/bin`或`/opt/local/sbin`
*   具有 Apt 的 Ubuntu：`/usr/bin`
*   使用我的自定义安装脚本的 Ubuntu：`/usr/local/bin`
*   其他类 Unix 系统：`/usr/bin`和`/usr/local/bin`

除 Mac 带有 Homebrew 的情况外，默认情况下，可执行文件的文件夹应位于`PATH`中。 对于 Homebrew，如果要将相关文件夹添加到`PATH`，请参阅第 1 章，“设置 OpenCV”的“将 Homebrew 与现成的包配合使用（不支持深度相机）”部分的第二步中的说明。 否则，请注意可执行文件的完整路径，因为我们需要在运行它们时使用它。

# 创建训练集和级联

此后，我们将两个可执行文件称为`<opencv_createsamples>`和`<opencv_traincascade>`。 切记替换适合您的系统和设置的路径和文件名。

这些可执行文件具有某些数据文件作为输入和输出。 以下是生成这些数据文件的典型方法：

1.  手动创建一个描述负面训练图像集的文本文件。 我们将此文件称为`<negative_description>`。
2.  手动创建一个描述正面训练图像集的文本文件。 我们将此文件称为`<positive_description>`。
3.  以`<negative_description>`和`<positive_description>`作为参数运行`<opencv_createsamples>`。 该可执行文件将创建一个描述训练数据的二进制文件。 我们将后一个文件称为`<binary_description>`。
4.  以`<binary_description>`作为参数运行`<opencv_traincascade>`。 该可执行文件创建二进制级联文件，我们将其称为`<cascade>`。

我们可以选择`<negative_description>`，`<positive_description>`，`<binary_description>`和`<cascade>`的实际名称和路径。

现在，让我们详细了解三个步骤。

## 创建`<negative_description>`

`<negative_description>`是一个文本文件，列出了所有负面训练图像的相对路径。 路径应以换行符分隔。 例如，假设我们具有以下目录结构，其中`<negative_description>`是`negative/desc.txt`：

```py
negative
    desc.txt
    images
        negative 0.png
        negative 1.png
```

然后，`negative/desc.txt`的内容可以如下：

```py
"img/negative 0.png"
"img/negative 1.png"
```

对于少量图像，我们可以手动编写这样的文件。 对于大量图像，我们应该改用命令行来查找与特定模式匹配的相对路径，并将这些匹配项输出到文件中。 继续我们的示例，我们可以通过在 Windows 的“命令提示符”中运行以下命令来生成`negative/desc.txt`：

```py
> cd negative
> forfiles /m images\*.png /c "cmd /c echo @relpath" > desc.txt

```

请注意，在这种情况下，相对路径的格式为`.\images\negative 0.png`，这是可以接受的。

另外，在类似 Unix 的外壳中，例如 Mac 或 Ubuntu 上的 Terminal，我们可以运行以下命令：

```py
$ cd negative
$ find img/*.png | sed -e "s/^/\"/g;s/$/\"/g" > desc.txt

```

## 创建`<positive_description>`

如果我们有多个正面训练图像，则需要`<positive_description>`。 否则，请继续下一节。 `<positive_description>`是一个文本文件，列出了所有积极训练图像的相对路径。 在每个路径之后，`<positive_description>`还包含一系列数字，这些数字指示在图像中找到了多少个目标实例，以及哪些子矩形包含了这些目标实例。 对于每个子矩形，数字按以下顺序排列：x，y，宽度和高度。 考虑以下示例：

```py
"img/positive 0.png"  1  120 160 40 40
"img/positive 1.png"  2  200 120 40 60  80 60 20 20
```

在此，`img/positive 0.png`在子矩形中包含目标的一个实例，该子矩形的左上角为`(120, 160)`，右下角为`(160, 200)`。 同时，`img/positive 1.png`包含目标的两个实例。 一个实例位于子矩形中，该子矩形的左上角为`(200, 120)`，而其右下角为`(240, 180)`。 另一个实例位于子矩形中，该子矩形的左上角为`(80, 60)`，右下角为`(100, 80)`。

要创建这样的文件，我们可以以与`<negative_description>`相同的方式开始生成图像路径列表。 然后，我们必须基于对图像的专家（人类）分析，手动添加有关目标实例的数据。

## 通过运行`<opencv_createsamples>`创建`<binary_description>`

假设我们有多个正训练图像，因此，我们创建了`<positive_description>`，我们现在可以通过运行以下命令来生成`<binary_description>`：

```py
$ <opencv_createsamples> -vec <binary_description> -info <positive_description> -bg <negative_description>

```

另外，如果我们有一个正面的训练图像（我们将其称为`<positive_image>`），则应改为运行以下命令：

```py
$ <opencv_createsamples> -vec <binary_description> -image <positive_image> -bg <negative_description>

```

有关`<opencv_createsamples>`的其他（可选）标志的信息，请参见[这个页面](http://docs.opencv.org/doc/user_guide/ug_traincascade.html)上的官方文档。

## 通过运行`<opencv_traincascade>`创建`<cascade>`

最后，我们可以通过运行以下命令生成`<cascade>`：

```py
$ <opencv_traincascade> -data <cascade> -vec <binary_description> -bg <negative_description>

```

有关`<opencv_traincascade>`的其他（可选）标志的信息，请参见[这个页面](http://docs.opencv.org/doc/user_guide/ug_traincascade.html)上的官方文档。

### 提示

**发声**

为求好运，请在运行`<opencv_traincascade>`时发出模仿的声音。 例如，说“Moo！” 如果正面训练图像是母牛。

# 测试和改进`<cascade>`

`<cascade>`是与 OpenCV 的`CascadeClassifier`类的构造器兼容的 XML 文件。 对于如何使用`CascadeClassifier`的示例，请参考第 4 章“用 Haar 级联跟踪人脸”的`FaceTracker`实现。通过复制和修改`FaceTracker`和`Cameo`，您应该能够创建一个简单的测试应用，该应用在自定义目标的跟踪实例周围绘制矩形。

也许在您第一次尝试级联训练时，您将不会获得可靠的跟踪结果。 要提高训练效果，请执行以下操作：

*   考虑使分类问题更具体。 例如，`bald, shaven, male face without glasses`级联可能比普通的`face`级联更容易训练。 稍后，随着结果的改善，您可以尝试再次扩大问题范围。
*   收集更多的训练图像，更多！
*   确保`<negative_description>`包含*所有*负面训练图像，*仅包含*负面训练图像。
*   确保`<positive_description>`包含*所有*正面训练图像，*仅包含*正面训练图像。
*   确保`<positive_description>`中指定的子矩形正确。
*   查看并尝试使用`<opencv_createsamples>`和`<opencv_traincascade>`的可选标志。 这些标志在[这个页面](http://docs.opencv.org/doc/user_guide/ug_traincascade.html)的官方文档中进行了描述。

祝你好运，寻找图像！

# 总结

我们已经讨论了用于生成与 OpenCV 的`CascadeClassifier`兼容的级联文件的数据和可执行文件。 现在，您可以开始收集您喜欢的事物的图像并为其训练分类器！