# 第一章。设置 OpenCV

你选择这本书，可能已经对 OpenCV 有了一定的了解。也许，你听说过一些科幻般的功能，比如人脸检测，并对此产生了兴趣。如果是这样，你做出了完美的选择。**OpenCV** 代表 **开源计算机视觉**。它是一个免费的计算机视觉库，允许你操纵图像和视频，以完成各种任务，从显示网络摄像头的视频流到可能教会机器人识别现实生活中的物体。

在这本书中，你将学习如何利用 Python 编程语言充分发挥 OpenCV 的巨大潜力。Python 是一种优雅的语言，学习曲线相对平缓，功能非常强大。本章是快速设置 Python 2.7、OpenCV 以及其他相关库的指南。设置完成后，我们还将查看 OpenCV 的 Python 示例脚本和文档。

### 注意

如果你希望跳过安装过程，直接进入操作，你可以下载我在 [`techfort.github.io/pycv/`](http://techfort.github.io/pycv/) 提供的 **虚拟机**（**VM**）。

此文件与 VirtualBox 兼容，这是一个免费使用的虚拟化应用程序，允许你构建和运行虚拟机。我构建的虚拟机基于 Ubuntu Linux 14.04，并安装了所有必要的软件，以便你可以立即开始编码。

这个虚拟机需要至少 2 GB 的 RAM 才能平稳运行，所以请确保为虚拟机分配至少 2 GB（但理想情况下，超过 4 GB）的 RAM，这意味着你的主机机器至少需要 6 GB 的 RAM 才能维持其运行。

本章涵盖了以下相关库：

+   **NumPy**：这个库是 OpenCV Python 绑定的依赖项。它提供了包括高效数组在内的数值计算功能。

+   **SciPy**：这个库是一个与 NumPy 密切相关的科学计算库。它不是 OpenCV 所必需的，但它在操纵 OpenCV 图像中的数据时很有用。

+   **OpenNI**：这个库是 OpenCV 的可选依赖项。它增加了对某些深度相机（如华硕 XtionPRO）的支持。

+   **SensorKinect**：这个库是一个 OpenNI 插件，也是 OpenCV 的可选依赖项。它增加了对微软 Kinect 深度相机的支持。

对于本书的目的，OpenNI 和 SensorKinect 可以被认为是可选的。它们在 第四章 中使用，*深度估计和分割*，但在其他章节或附录中并未使用。

### 注意

本书专注于 OpenCV 3，这是 OpenCV 库的新版主要发布。有关 OpenCV 的所有附加信息可在 [`opencv.org`](http://opencv.org) 获取，其文档可在 [`docs.opencv.org/master`](http://docs.opencv.org/master) 获取。

# 选择和使用正确的设置工具

我们可以根据我们的操作系统和想要进行多少配置来选择各种设置工具。让我们概述一下 Windows、Mac、Ubuntu 和其他类 Unix 系统的工具。

## Windows 上的安装

Windows 没有预装 Python。然而，有预编译的 Python、NumPy、SciPy 和 OpenCV 的安装向导可用。或者，我们可以从源代码构建。OpenCV 的构建系统使用 CMake 进行配置，并使用 Visual Studio 或 MinGW 进行编译。

如果我们想要支持深度相机，包括 Kinect，我们首先应该安装 OpenNI 和 SensorKinect，它们作为预编译的二进制文件和安装向导提供。然后，我们必须从源代码构建 OpenCV。

### 注意

预编译版的 OpenCV 不支持深度相机。

在 Windows 上，OpenCV 2 对 32 位 Python 的支持优于 64 位 Python；然而，由于今天大多数销售的计算机都是 64 位系统，我们的说明将参考 64 位。所有安装程序都有 32 位版本，可以从与 64 位相同的网站下载。

以下步骤中的一些涉及编辑系统的`PATH`变量。这项任务可以在**控制面板**的**环境变量**窗口中完成。

1.  在 Windows Vista / Windows 7 / Windows 8 上，点击**开始**菜单并启动**控制面板**。现在，导航到**系统**和**安全** | **系统** | **高级系统设置**。点击**环境变量…**按钮。

1.  在 Windows XP 上，点击**开始**菜单并导航到**控制面板** | **系统**。选择**高级**选项卡。点击**环境变量…**按钮。

1.  现在，在**系统变量**下，选择**Path**并点击**编辑…**按钮。

1.  按指示进行更改。

1.  要应用更改，点击所有**确定**按钮（直到我们回到**控制面板**的主窗口）。

1.  然后，注销并重新登录（或者重新启动）。

### 使用二进制安装程序（不支持深度相机）

如果您愿意，可以选择单独安装 Python 及其相关库；然而，有一些 Python 发行版包含安装程序，可以设置整个 SciPy 堆栈（包括 Python 和 NumPy），这使得设置开发环境变得非常简单。

其中一个发行版是 Anaconda Python（可在[`09c8d0b2229f813c1b93­c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0­Windows-x86_64.exe`](http://09c8d0b2229f813c1b93%C2%ADc95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0%C2%ADWindows-x86_64.exe)下载）。一旦下载了安装程序，运行它并记得按照前面的步骤将 Anaconda 安装路径添加到您的`PATH`变量中。

这里是设置 Python7、NumPy、SciPy 和 OpenCV 的步骤：

1.  从[`www.python.org/ftp/python/2.7.9/python-2.7.9.amd64.msi`](https://www.python.org/ftp/python/2.7.9/python-2.7.9.amd64.msi)下载并安装 32 位 Python 2.7.9。

1.  从 [`www.lfd.uci.edu/~gohlke/pythonlibs/#numpyhttp://sourceforge.net/projects/numpy/files/NumPy/1.6.2/numpy-1.6.2-win32-superpack-python2.7.exe/download`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpyhttp://sourceforge.net/projects/numpy/files/NumPy/1.6.2/numpy-1.6.2-win32-superpack-python2.7.exe/download) 下载并安装 NumPy 1.6.2（注意，由于 Windows 上缺少 NumPy 所依赖的 64 位 Fortran 编译器，在 Windows 64 位上安装 NumPy 有点棘手。前一个链接中的二进制文件是非官方的）。

1.  从 [`www.lfd.uci.edu/~gohlke/pythonlibs/#scipyhttp://sourceforge.net/projects/scipy/files/scipy/0.11.0/scipy-0.11.0­win32-superpack-python2.7.exe/download`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipyhttp://sourceforge.net/projects/scipy/files/scipy/0.11.0/scipy-0.11.0%C2%ADwin32-superpack-python2.7.exe/download) 下载并安装 SciPy 11.0（这与 NumPy 相同，这些都是社区安装程序）。

1.  从 [`github.com/Itseez/opencv`](https://github.com/Itseez/opencv) 下载 OpenCV 3.0.0 的自解压 ZIP 文件。运行此 ZIP 文件，并在提示时输入目标文件夹，我们将称之为 `<unzip_destination>`。将创建一个子文件夹 `<unzip_destination>\opencv`。

1.  将 `<unzip_destination>\opencv\build\python\2.7\cv2.pyd` 复制到 `C:\Python2.7\Lib\site-packages`（假设我们将 Python 2.7 安装在默认位置）。如果您使用 Anaconda 安装了 Python 2.7，请使用 Anaconda 安装文件夹而不是默认的 Python 安装。现在，新的 Python 安装可以找到 OpenCV。

1.  如果我们想要默认使用新的 Python 安装运行 Python 脚本，则需要执行一个最终步骤。编辑系统的 `PATH` 变量，并追加 `;C:\Python2.7`（假设我们将 Python 2.7 安装在默认位置）或您的 Anaconda 安装文件夹。删除任何以前的 Python 路径，例如 `;C:\Python2.6`。注销并重新登录（或者重新启动）。

### 使用 CMake 和编译器

Windows 不自带任何编译器或 CMake。我们需要安装它们。如果我们想要支持包括 Kinect 在内的深度相机，我们还需要安装 OpenNI 和 SensorKinect。

假设我们已经通过二进制文件（如前所述）或源代码安装了 32 位 Python 2.7、NumPy 和 SciPy。现在，我们可以继续安装编译器和 CMake，可选地安装 OpenNI 和 SensorKinect，然后从源代码构建 OpenCV：

1.  从 [`www.cmake.org/files/v3.1/cmake-3.1.2-win32-x86.exe`](http://www.cmake.org/files/v3.1/cmake-3.1.2-win32-x86.exe) 下载并安装 CMake 3.1.2。在运行安装程序时，选择“**将 CMake 添加到系统 PATH 以供所有用户使用**”或“**将 CMake 添加到当前用户的系统 PATH**”。不用担心没有 64 位版本的 CMake，因为 CMake 只是一个配置工具，它本身不执行任何编译。相反，在 Windows 上，它创建可以与 Visual Studio 打开的工程文件。

1.  从 [`www.visualstudio.com/products/free-developer-offers-vs.aspx?slcid=0x409&type=web or MinGW`](https://www.visualstudio.com/products/free-developer-offers-vs.aspx?slcid=0x409&type=web%20or%20MinGW) 下载并安装 Microsoft Visual Studio 2013（如果您在 Windows 7 上工作，请选择桌面版）。

    注意，您需要使用您的 Microsoft 账户进行登录，如果您没有，您可以在现场创建一个。安装软件，安装完成后重新启动。

    对于 MinGW，从 [`sourceforge.net/projects/mingw/files/Installer/mingw-get-setup.exe/download`](http://sourceforge.net/projects/mingw/files/Installer/mingw-get-setup.exe/download) 和 [`sourceforge.net/projects/mingw/files/OldFiles/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe/download`](http://sourceforge.net/projects/mingw/files/OldFiles/mingw-get-inst/mingw-get-inst-20120426/mingw-get-inst-20120426.exe/download) 获取安装程序。在运行安装程序时，确保目标路径不包含空格，并且包含可选的 C++ 编译器。编辑系统的 `PATH` 变量，并追加 `;C:\MinGW\bin`（假设 MinGW 安装在默认位置）。重新启动系统。

1.  可选，从 OpenNI 在 GitHub 主页提供的链接中下载并安装 OpenNI 1.5.4.0。[`github.com/OpenNI/OpenNI`](https://github.com/OpenNI/OpenNI)。

1.  您可以从 [`github.com/avin2/SensorKinect/blob/unstable/Bin/SensorKinect093-Bin-Win32-v5.1.2.1.msi?raw=true`](https://github.com/avin2/SensorKinect/blob/unstable/Bin/SensorKinect093-Bin-Win32-v5.1.2.1.msi?raw=true)（32 位）下载并安装 SensorKinect 0.93。对于 64 位 Python，从 [`github.com/avin2/SensorKinect/blob/unstable/Bin/SensorKinect093-Bin-Win64-v5.1.2.1.msi?raw=true`](https://github.com/avin2/SensorKinect/blob/unstable/Bin/SensorKinect093-Bin-Win64-v5.1.2.1.msi?raw=true)（64 位）下载设置文件。请注意，这个存储库已经停用超过三年了。

1.  从 [`github.com/Itseez/opencv`](https://github.com/Itseez/opencv) 下载 OpenCV 3.0.0 的自解压 ZIP 文件。运行自解压 ZIP 文件，当提示时，输入任何目标文件夹，我们将称之为 `<unzip_destination>`。然后创建一个子文件夹，`<unzip_destination>\opencv`。

1.  打开命令提示符，使用以下命令创建一个新文件夹，我们的构建将放在那里：

    ```py
    > mkdir<build_folder>

    ```

    更改`build`文件夹的目录：

    ```py
    > cd <build_folder>

    ```

1.  现在，我们已经准备好配置我们的构建。为了理解所有选项，我们可以阅读`<unzip_destination>\opencv\CMakeLists.txt`中的代码。然而，出于本书的目的，我们只需要使用那些将为我们提供带有 Python 绑定的发布构建的选项，以及可选的通过 OpenNI 和 SensorKinect 的深度相机支持。

1.  打开 CMake（`cmake-gui`）并指定 OpenCV 源代码的位置以及您希望构建库的文件夹。点击**配置**。选择要生成的项目。在这种情况下，选择 Visual Studio 12（对应于 Visual Studio 2013）。CMake 完成项目配置后，将输出一个构建选项列表。如果您看到红色背景，这意味着可能需要重新配置项目：CMake 可能会报告它未能找到某些依赖项。OpenCV 的许多依赖项是可选的，所以现在不必过于担心。

    ### 注意

    如果构建未完成或您遇到问题，请尝试安装缺失的依赖项（通常作为预构建的二进制文件提供），然后从这一步重新构建 OpenCV。

    您可以选择/取消选择构建选项（根据您在机器上安装的库），然后再次点击**配置**，直到获得清晰的背景（白色）。

1.  在此过程结束时，您可以点击**生成**，这将创建一个`OpenCV.sln`文件在您选择的构建文件夹中。然后，您可以导航到`<build_folder>/OpenCV.sln`并使用 Visual Studio 2013 打开该文件，然后继续构建项目，`ALL_BUILD`。您需要构建 OpenCV 的**调试**和**发布**版本，因此请先以**调试**模式构建库，然后选择**发布**并重新构建它（*F7*是启动构建的键）。

1.  在此阶段，您将在 OpenCV 构建目录中有一个`bin`文件夹，其中将包含所有生成的`.dll`文件，这将允许您将 OpenCV 包含到您的项目中。

    或者，对于 MinGW，运行以下命令：

    ```py
    > cmake -D:CMAKE_BUILD_TYPE=RELEASE -D:WITH_OPENNI=ON -G "MinGWMakefiles" <unzip_destination>\opencv

    ```

    如果未安装 OpenNI，则省略`-D:WITH_OPENNI=ON`。（在这种情况下，将不支持深度相机。）如果 OpenNI 和 SensorKinect 安装到非默认位置，请修改命令以包含`-D:OPENNI_LIB_DIR=<openni_install_destination>\Lib -D:OPENNI_INCLUDE_DIR=<openni_install_destination>\Include -D:OPENNI_PRIME_SENSOR_MODULE_BIN_DIR=<sensorkinect_install_destination>\Sensor\Bin`。

    或者，对于 MinGW，运行以下命令：

    ```py
    > mingw32-make

    ```

1.  将`<build_folder>\lib\Release\cv2.pyd`（来自 Visual Studio 构建）或`<build_folder>\lib\cv2.pyd`（来自 MinGW 构建）复制到`<python_installation_folder>\site-packages`。

1.  最后，编辑系统的`PATH`变量，并追加`;<build_folder>/bin/Release`（对于 Visual Studio 构建）或`;<build_folder>/bin`（对于 MinGW 构建）。重新启动您的系统。

## 在 OS X 上安装

以前的一些 Mac 版本预装了由 Apple 定制的 Python 2.7 版本，用于满足系统的内部需求。然而，这种情况已经改变，标准的 OS X 版本现在都带有标准的 Python 安装。在 [python.org](http://python.org) 上，您还可以找到与新的 Intel 系统和旧 PowerPC 兼容的通用二进制文件。

### 注意

您可以从 [`www.python.org/downloads/release/python-279/`](https://www.python.org/downloads/release/python-279/) 获取此安装程序（参考 Mac OS X 32 位 PPC 或 Mac OS X 64 位 Intel 链接）。从下载的 `.dmg` 文件安装 Python 将简单地覆盖您当前系统上的 Python 安装。

对于 Mac，获取标准 Python 2.7、NumPy、SciPy 和 OpenCV 有几种可能的方法。所有方法最终都需要使用 Xcode 开发者工具从源代码编译 OpenCV。然而，根据方法的不同，这项任务可以通过第三方工具以各种方式自动化。我们将通过使用 MacPorts 或 Homebrew 来查看这些方法。这些工具可以执行 CMake 可以执行的所有操作，并且帮助我们解决依赖关系，并将我们的开发库与系统库分开。

### 提示

我推荐使用 MacPorts，尤其是如果您想通过 OpenNI 和 SensorKinect 编译具有深度相机支持的 OpenCV。相关的补丁和构建脚本，包括我维护的一些，已经为 MacPorts 准备好。相比之下，Homebrew 目前还没有提供编译具有深度相机支持的 OpenCV 的现成解决方案。

在继续之前，让我们确保 Xcode 开发者工具已正确设置：

1.  从 Mac App Store 或 [`developer.apple.com/xcode/downloads/`](https://developer.apple.com/xcode/downloads/) 下载并安装 Xcode。在安装过程中，如果有安装 **命令行工具** 的选项，请选择它。

1.  打开 Xcode 并接受许可协议。

1.  如果安装程序没有提供安装 **命令行工具** 的选项，则需要执行最后一步。导航到 **Xcode** | **首选项** | **下载**，然后点击 **命令行工具** 旁边的 **安装** 按钮。等待安装完成并退出 Xcode。

或者，您可以通过运行以下命令（在终端中）来安装 Xcode 命令行工具：

```py
$ xcode-select –install

```

现在，我们有了任何方法所需的编译器。

### 使用带有现成软件包的 MacPorts

我们可以使用 MacPorts 软件包管理器帮助我们设置 Python 2.7、NumPy 和 OpenCV。MacPorts 提供了终端命令，可以自动化下载、编译和安装各种开源软件（**OSS**）。MacPorts 还会根据需要安装依赖项。对于每件软件，依赖项和构建配方都定义在一个名为 Portfile 的配置文件中。MacPorts 存储库是 **Portfiles** 的集合。

从已经设置好 Xcode 及其命令行工具的系统开始，以下步骤将使用 MacPorts 为我们提供 OpenCV 安装：

1.  从 [`www.macports.org/install.php`](http://www.macports.org/install.php) 下载并安装 MacPorts。

1.  如果您想支持 Kinect 深度相机，您需要告诉 MacPorts 下载我编写的自定义 Portfiles 的位置。为此，编辑 `/opt/local/etc/macports/sources.conf`（假设 MacPorts 安装在默认位置）。在以下行 `rsync://rsync.macports.org/release/ports/ [default]` 之上添加以下行：

    ```py
    http://nummist.com/opencv/ports.tar.gz
    ```

    保存文件。现在，MacPorts 知道它必须首先在我的在线仓库中搜索 Portfiles，然后是默认在线仓库。

1.  打开终端并运行以下命令来更新 MacPorts：

    ```py
    $ sudo port selfupdate

    ```

    当提示时，输入您的密码。

1.  现在（如果我们使用我的仓库），运行以下命令来安装具有 Python 2.7 绑定和深度相机支持的 OpenCV，包括 Kinect：

    ```py
    $ sudo port install opencv +python27 +openni_sensorkinect

    ```

    或者（无论是否使用我的仓库），运行以下命令来安装具有 Python 2.7 绑定和深度相机支持的 OpenCV，不包括 Kinect：

    ```py
    $ sudo port install opencv +python27 +openni

    ```

    ### 注意

    依赖项，包括 Python 2.7、NumPy、OpenNI 和（在第一个示例中）SensorKinect，也将自动安装。

    通过在命令中添加 `+python27`，我们指定我们想要具有 Python 2.7 绑定的 `opencv` 变体（构建配置）。同样，`+openni_sensorkinect` 指定具有通过 OpenNI 和 SensorKinect 提供的最广泛支持的深度相机变体。如果您不打算使用深度相机，可以省略 `+openni_sensorkinect`，或者如果您打算使用与 OpenNI 兼容的深度相机但不是 Kinect，可以将其替换为 `+openni`。在安装之前，我们可以输入以下命令来查看所有可用变体的完整列表：

    ```py
    $ port variants opencv

    ```

    根据我们的定制需求，我们可以在 `install` 命令中添加其他变体。为了获得更大的灵活性，我们可以编写自己的变体（如下一节所述）。

1.  此外，运行以下命令来安装 SciPy：

    ```py
    $ sudo port install py27-scipy

    ```

1.  Python 安装的执行文件名为 `python2.7`。如果我们想将默认的 `python` 可执行文件链接到 `python2.7`，请运行此命令：

    ```py
    $ sudo port install python_select
    $ sudo port select python python27

    ```

### 使用 MacPorts 和您自己的自定义软件包

通过几个额外的步骤，我们可以更改 MacPorts 编译 OpenCV 或其他软件的方式。如前所述，MacPorts 的构建配方定义在名为 Portfiles 的配置文件中。通过创建或编辑 Portfiles，我们可以访问高度可配置的构建工具，如 CMake，同时还能享受 MacPorts 的功能，如依赖关系解析。

假设我们已安装 MacPorts。现在，我们可以配置 MacPorts 以使用我们编写的自定义 Portfiles：

1.  在某个位置创建一个文件夹来存放我们的自定义 Portfiles。我们将把这个文件夹称为 `<local_repository>`。

1.  编辑`/opt/local/etc/macports/sources.conf`文件（假设 MacPorts 安装到默认位置）。在`rsync://rsync.macports.org/release/ports/ [default]`行上方，添加此行：

    ```py
    file://<local_repository>
    ```

    例如，如果`<local_repository>`是`/Users/Joe/Portfiles`，请添加以下行：

    ```py
    file:///Users/Joe/Portfiles
    ```

    注意三重斜杠并保存文件。现在，MacPorts 知道它必须首先在`<local_repository>`中搜索 Portfiles，然后是其默认在线仓库。

1.  打开终端并更新 MacPorts 以确保我们拥有默认仓库的最新 Portfile：

    ```py
    $ sudo port selfupdate

    ```

1.  以默认仓库的`opencv` Portfile 为例，让我们也复制目录结构，这决定了包在 MacPorts 中的分类方式：

    ```py
    $ mkdir <local_repository>/graphics/
    $ cp /opt/local/var/macports/sources/rsync.macports.org/release/ports/graphics/opencv <local_repository>/graphics

    ```

    或者，对于包含 Kinect 支持的示例，我们可以从[`nummist.com/opencv/ports.tar.gz`](http://nummist.com/opencv/ports.tar.gz)下载我的在线仓库，解压它，并将整个`graphics`文件夹复制到`<local_repository>`：

    ```py
    $ cp <unzip_destination>/graphics <local_repository>

    ```

1.  编辑`<local_repository>/graphics/opencv/Portfile`。注意，此文件指定了 CMake 配置标志、依赖项和变体。有关 Portfile 编辑的详细信息，请参阅[`guide.macports.org/#development`](http://guide.macports.org/#development)。

    要查看与 OpenCV 相关的 CMake 配置标志，我们需要查看其源代码。从[`github.com/Itseez/opencv/archive/3.0.0.zip`](https://github.com/Itseez/opencv/archive/3.0.0.zip)下载源代码存档，将其解压到任何位置，并阅读`<unzip_destination>/OpenCV-3.0.0/CMakeLists.txt`。

    在对 Portfile 进行任何编辑后，请保存它。

1.  现在，我们需要在本地仓库中生成一个索引文件，以便 MacPorts 可以找到新的 Portfile：

    ```py
    $ cd <local_repository>
    $ portindex

    ```

1.  从现在起，我们可以将我们的自定义`opencv`文件视为任何其他 MacPorts 包。例如，我们可以按照以下方式安装它：

    ```py
    $ sudo port install opencv +python27 +openni_sensorkinect

    ```

    注意，由于它们在`/opt/local/etc/macports/sources.conf`中的列表顺序，我们的本地仓库的 Portfile 优先于默认仓库的 Portfile。

### 使用带有预装包的 Homebrew（不支持深度相机）

Homebrew 是另一个可以帮助我们的包管理器。通常，MacPorts 和 Homebrew 不应安装在同一台机器上。

从已经设置好 Xcode 及其命令行工具的系统开始，以下步骤将通过 Homebrew 为我们提供 OpenCV 安装：

1.  打开终端并运行以下命令以安装 Homebrew：

    ```py
    $ ruby -e "$(curl -fsSkLraw.github.com/mxcl/homebrew/go)"

    ```

1.  与 MacPorts 不同，Homebrew 不会自动将其可执行文件放入`PATH`。要这样做，创建或编辑`~/.profile`文件，并在代码顶部添加此行：

    ```py
    export PATH=/usr/local/bin:/usr/local/sbin:$PATH

    ```

    保存文件并运行以下命令以刷新`PATH`：

    ```py
    $ source ~/.profile

    ```

    注意，现在由 Homebrew 安装的可执行文件优先于由系统安装的可执行文件。

1.  要运行 Homebrew 的自我诊断报告，请运行以下命令：

    ```py
    $ brew doctor

    ```

    遵循它提供的任何故障排除建议。

1.  现在，更新 Homebrew：

    ```py
    $ brew update

    ```

1.  运行以下命令安装 Python 2.7：

    ```py
    $ brew install python

    ```

1.  现在，我们可以安装 NumPy。Homebrew 对 Python 库包的选择有限，所以我们使用一个名为`pip`的单独的包管理工具，它包含在 Homebrew 的 Python 中：

    ```py
    $ pip install numpy

    ```

1.  SciPy 包含一些 Fortran 代码，因此我们需要一个合适的编译器。我们可以使用 Homebrew 来安装`gfortran`编译器：

    ```py
    $ brew install gfortran

    ```

    现在，我们可以安装 SciPy：

    ```py
    $ pip install scipy

    ```

1.  要在 64 位系统上安装 OpenCV（自 2006 年底以来所有新的 Mac 硬件），请运行以下命令：

    ```py
    $ brew install opencv

    ```

### 小贴士

**下载示例代码**

您可以从您在[`www.packtpub.com`](http://www.packtpub.com)的账户中下载您购买的所有 Packt Publishing 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

### 使用 Homebrew 和自定义包

Homebrew 使得编辑现有的包定义变得容易：

```py
$ brew edit opencv

```

包定义实际上是 Ruby 编程语言中的脚本。有关编辑它们的提示，可以在 Homebrew Wiki 页面[`github.com/mxcl/homebrew/wiki/Formula-Cookbook`](https://github.com/mxcl/homebrew/wiki/Formula-Cookbook)上找到。脚本可以指定 Make 或 CMake 配置标志，以及其他内容。

要查看哪些 CMake 配置标志与 OpenCV 相关，我们需要查看其源代码。从[`github.com/Itseez/opencv/archive/3.0.0.zip`](https://github.com/Itseez/opencv/archive/3.0.0.zip)下载源代码存档，将其解压缩到任何位置，并阅读`<unzip_destination>/OpenCV-2.4.3/CMakeLists.txt`。

在对 Ruby 脚本进行编辑后，保存它。

定制的包可以像普通包一样处理。例如，它可以按照以下方式安装：

```py
$ brew install opencv

```

## 在 Ubuntu 及其衍生版上安装

首先最重要的是，这里有一个关于 Ubuntu 操作系统版本的快速说明：Ubuntu 有一个 6 个月的发布周期，其中每个发布都是主要版本（撰写本文时为 14）的.04 或.10 小版本。然而，每两年，Ubuntu 会发布一个被归类为**长期支持**（**LTS**）的版本，这将通过 Canonical（Ubuntu 背后的公司）为您提供五年的支持。如果您在企业环境中工作，安装 LTS 版本无疑是明智的。目前可用的最新版本是 14.04。

Ubuntu 预装了 Python 2.7。标准的 Ubuntu 仓库包含没有深度相机支持的 OpenCV 2.4.9 包。在撰写本文时，OpenCV 3 尚未通过 Ubuntu 仓库提供，因此我们必须从源代码构建它。幸运的是，大多数 Unix-like 和 Linux 系统已经预装了所有必要的软件，可以从头开始构建项目。从源代码构建时，OpenCV 可以通过 OpenNI 和 SensorKinect 支持深度相机，它们作为预编译的二进制文件和安装脚本提供。

### 使用 Ubuntu 仓库（不支持深度相机）

我们可以使用`apt`包管理器通过运行以下命令安装 Python 及其所有必要的依赖项：

```py
> sudo apt-get install build-essential
> sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec­dev libavformat-dev libswscale-dev 
> sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

```

同样，我们也可以使用 Ubuntu 软件中心，它是`apt`包管理器的图形前端。

### 从源代码构建 OpenCV

现在我们已经安装了整个 Python 栈和`cmake`，我们可以构建 OpenCV。首先，我们需要从[`github.com/Itseez/opencv/archive/3.0.0-beta.zip`](https://github.com/Itseez/opencv/archive/3.0.0-beta.zip)下载源代码。

在终端中解压归档并将其移动到解压文件夹中。

然后，运行以下命令：

```py
> mkdir build
> cd build
> cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
> make
> make install

```

安装完成后，您可能想查看 OpenCV 的 Python 示例，位于`<opencv_folder>/opencv/samples/python`和`<script_folder>/opencv/samples/python2`。

## 在其他类 Unix 系统上的安装

对于 Ubuntu（如前所述）的方法可能适用于任何从 Ubuntu 14.04 LTS 或 Ubuntu 14.10 衍生出来的 Linux 发行版，如下所示：

+   Kubuntu 14.04 LTS 或 Kubuntu 14.10

+   Xubuntu 14.04 LTS 或 Xubuntu 14.10

+   Linux Mint 17

在 Debian Linux 及其衍生版本中，`apt`包管理器的工作方式与 Ubuntu 相同，尽管可用的包可能不同。

在 Gentoo Linux 及其衍生版本中，Portage 包管理器与 MacPorts（如前所述）类似，尽管可用的包可能不同。

在 FreeBSD 衍生版本上，安装过程再次类似于 MacPorts；实际上，MacPorts 源自 FreeBSD 采用的`ports`安装系统。请参考出色的 FreeBSD 手册[`www.freebsd.org/doc/handbook/`](https://www.freebsd.org/doc/handbook/)以了解软件安装过程的概述。

在其他类 Unix 系统中，包管理器和可用的包可能不同。请查阅您的包管理器文档，并搜索名称中包含 `opencv` 的包。请记住，OpenCV 及其 Python 绑定可能被拆分为多个包。

此外，寻找系统提供商、仓库维护者或社区发布的任何安装说明。由于 OpenCV 使用相机驱动程序和媒体编解码器，在多媒体支持较差的系统上，使所有功能正常工作可能很棘手。在某些情况下，可能需要重新配置或重新安装系统包以实现兼容性。

如果有 OpenCV 的包可用，请检查它们的版本号。本书建议使用 OpenCV 3 或更高版本。此外，请检查这些包是否提供 Python 绑定和通过 OpenNI 和 SensorKinect 的深度相机支持。最后，检查开发者社区中是否有人报告了使用这些包的成功或失败情况。

如果我们想从源代码自定义构建 OpenCV，可能有助于参考之前讨论的 Ubuntu 安装脚本，并将其适应到另一个系统上的包管理器和包。

# 安装 Contrib 模块

与 OpenCV 2.4 不同，一些模块包含在名为`opencv_contrib`的仓库中，该仓库可在[`github.com/Itseez/opencv_contrib`](https://github.com/Itseez/opencv_contrib)找到。我强烈建议安装这些模块，因为它们包含 OpenCV 中没有的额外功能，例如人脸识别模块。

下载完成后（无论是通过`zip`还是`git`，我推荐使用`git`，这样您可以通过简单的`git pull`命令保持更新），您可以重新运行`cmake`命令，包括构建带有`opencv_contrib`模块的 OpenCV，如下所示：

```py
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules <opencv_source_directory>

```

因此，如果您已遵循标准程序并在 OpenCV 下载文件夹中创建了一个构建目录，您应该运行以下命令：

```py
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules  -D CMAKE_INSTALL_PREFIX=/usr/local ..
make

```

# 运行示例

运行几个示例脚本是测试 OpenCV 是否正确设置的好方法。这些示例包含在 OpenCV 的源代码存档中。

在 Windows 上，我们应该已经下载并解压了 OpenCV 的自解压 ZIP 文件。在`<unzip_destination>/opencv/samples`中找到示例。

在 Unix-like 系统上，包括 Mac，从[`github.com/Itseez/opencv/archive/3.0.0.zip`](https://github.com/Itseez/opencv/archive/3.0.0.zip)下载源代码存档，并将其解压到任何位置（如果我们还没有这样做）。在`<unzip_destination>/OpenCV-3.0.0/samples`中找到示例。

一些示例脚本需要命令行参数。然而，以下脚本（以及其他一些脚本）可以在没有任何参数的情况下运行：

+   `python/camera.py`：此脚本显示一个网络摄像头流（假设已经插入了网络摄像头）。

+   `python/drawing.py`：此脚本绘制一系列形状，例如屏幕保护程序。

+   `python2/hist.py`：此脚本显示一张照片。按*A*、*B*、*C*、*D*或*E*查看照片的变体以及相应的颜色或灰度值直方图。

+   `python2/opt_flow.py`（Ubuntu 包中缺失）：此脚本显示带有叠加光流可视化（例如运动方向）的网络摄像头流。例如，慢慢在摄像头前挥手以查看效果。按*1*或*2*进行不同的可视化。

要退出脚本，请按*Esc*（不是窗口的关闭按钮）。

如果我们遇到`ImportError: No module named 'cv2.cv'`的消息，那么这意味着我们正在从不知道 OpenCV 的 Python 安装中运行脚本。这种情况有两个可能的解释：

+   OpenCV 安装过程中可能有一些步骤失败或被遗漏。返回并检查这些步骤。

+   如果机器上有多个 Python 安装，我们可能使用了错误的 Python 版本来启动脚本。例如，在 Mac 上，可能的情况是 OpenCV 是为 MacPorts Python 安装的，但我们使用的是系统的 Python 来运行脚本。返回并回顾有关编辑系统路径的安装步骤。此外，尝试使用如下命令手动从命令行启动脚本：

    ```py
    $ python python/camera.py

    ```

    您还可以使用以下命令：

    ```py
    $ python2.7 python/camera.py

    ```

    作为选择不同 Python 安装的另一种可能方法，尝试编辑示例脚本以删除`#!`行。这些行可能明确地将脚本与错误的 Python 安装关联起来（针对我们的特定设置）。

# 查找文档、帮助和更新

OpenCV 的文档可以在网上找到，网址为[`docs.opencv.org/`](http://docs.opencv.org/)。该文档包括 OpenCV 新 C++ API、新 Python API（基于 C++ API）、旧 C API 及其旧 Python API（基于 C API）的综合 API 参考。在查找类或函数时，请务必阅读有关新 Python API（`cv2`模块）的部分，而不是旧 Python API（`cv`模块）的部分。

该文档还可用作几个可下载的 PDF 文件：

+   **API 参考**：此文档可在[`docs.opencv.org/modules/refman.html`](http://docs.opencv.org/modules/refman.html)找到

+   **教程**：这些文档可在[`docs.opencv.org/doc/tutorials/tutorials.html`](http://docs.opencv.org/doc/tutorials/tutorials.html)找到（这些教程使用 C++代码；教程代码的 Python 版本可在阿比德·拉赫曼·K.的仓库[`goo.gl/EPsD1`](http://goo.gl/EPsD1)找到）

如果您在飞机或其他没有互联网接入的地方编写代码，您肯定希望保留文档的离线副本。

如果文档似乎没有回答您的问题，请尝试与 OpenCV 社区交流。以下是一些您可以找到有帮助人士的网站：

+   **OpenCV 论坛**：[`www.answers.opencv.org/questions/`](http://www.answers.opencv.org/questions/)

+   **大卫·米兰·埃斯克里瓦的博客**（本书的审稿人之一）：[`blog.damiles.com/`](http://blog.damiles.com/)

+   **阿比德·拉赫曼·K.的博客**（本书的审稿人之一）：[`www.opencvpython.blogspot.com/`](http://www.opencvpython.blogspot.com/)

+   **阿德里安·罗斯布鲁克的网站**（本书的审稿人之一）：[`www.pyimagesearch.com/`](http://www.pyimagesearch.com/)

+   **乔·米尼奇诺为此书的网站**（本书的作者）：[`techfort.github.io/pycv/`](http://techfort.github.io/pycv/)

+   **乔·豪斯为此书的网站**（本书第一版的作者）：[`nummist.com/opencv/`](http://nummist.com/opencv/)

最后，如果你是一位希望尝试最新（不稳定）OpenCV 源代码中的新功能、错误修复和示例脚本的进阶用户，请查看项目的仓库：[`github.com/Itseez/opencv/`](https://github.com/Itseez/opencv/)。

# 摘要

到目前为止，我们应该已经安装了一个可以完成本书中描述的项目所需所有功能的 OpenCV。根据我们采取的方法，我们可能还拥有一套可用的工具和脚本，可用于重新配置和重建 OpenCV 以满足我们未来的需求。

我们知道在哪里可以找到 OpenCV 的 Python 示例。这些示例涵盖了本书范围之外的不同功能范围，但它们作为额外的学习辅助工具是有用的。

在下一章中，我们将熟悉 OpenCV API 的最基本功能，即显示图像、视频，通过摄像头捕获视频，以及处理基本的键盘和鼠标输入。
