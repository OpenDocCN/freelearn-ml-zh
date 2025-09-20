# 第十二章：设置 Docker 容器

Docker 是一个方便的平台，可以将应用程序及其依赖项打包在一个可复制的虚拟环境中，该环境可以在不同的操作系统上运行。特别是，它与任何**Linux 系统**很好地集成。

可复制的虚拟环境在一个**Dockerfile**中进行了描述，该文件包含了一系列指令，这些指令应该被执行以实现所需的虚拟环境。这些指令主要包括安装过程，这与使用 Linux shell 的安装过程非常相似。一旦环境创建完成，您可以确信您的应用程序将在任何其他机器上具有相同的行为。

在 Docker 术语中，生成的虚拟环境被称为**Docker 镜像**。您可以创建虚拟环境的实例，这被称为**Docker 容器**。容器创建后，您可以在容器内执行您的代码。

请按照官方网站上的安装说明操作，以便在您选择的操作系统上安装并运行 Docker：[`docs.docker.com/install/`](https://docs.docker.com/install/)

为了您的方便，我们包括了 Dockerfile，这将使复制我们在本书中运行代码所使用的环境变得非常容易，无论您的计算机上安装了什么操作系统。首先，我们描述了一个仅使用 CPU 而不使用 GPU 加速的 Dockerfile。

# 定义 Dockerfile

Dockerfile 中的说明从基镜像开始，然后在那个镜像之上执行所需的安装和修改。

在撰写本文时，TensorFlow 不支持**Python 3.8**。如果您计划运行第七章，*学习识别交通标志*，或第九章，*学习分类和定位对象*，其中使用了 TensorFlow，您可以从**Python 3.7**开始，然后使用`pip`安装 TensorFlow，或者您可以选择`tensorflow/tensorflow:latest-py3`作为基镜像。

让我们回顾一下创建我们环境的步骤：

1.  我们从一个基镜像开始，这是一个基于**Debian**的基本 Python 镜像：

```py
FROM python:3.8
```

1.  我们安装了一些有用的包，这些包将在 OpenCV 和其他依赖项的安装过程中特别使用：

```py
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libgtk2.0-dev \
        libtbb2 libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libv4l-dev \
        libdc1394-22-dev \
        qt4-default \
        libatk-adaptor \
        libcanberra-gtk-module \
        x11-apps \
        libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*
```

1.  我们一起下载**OpenCV 4.2**以及贡献者包，这些包对于非免费算法，如**尺度不变特征变换**（**SIFT**）和**加速鲁棒特征**（**SURF**）是必需的：

```py
WORKDIR /
RUN wget --output-document cv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && unzip cv.zip \
    && wget --output-document contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip contrib.zip \
    && mkdir /opencv-${OPENCV_VERSION}/cmake_binary
```

1.  我们安装了一个与**OpenCV 4.2**兼容的 NumPy 版本：

```py
RUN pip install --upgrade pip && pip install --no-cache-dir numpy==1.18.1
```

1.  我们使用适当的标志编译 OpenCV：

```py
RUN cd /opencv-${OPENCV_VERSION}/cmake_binary \
    && cmake -DBUILD_TIFF=ON \
        -DBUILD_opencv_java=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_OPENGL=ON \
        -DWITH_OPENCL=ON \
        -DWITH_IPP=ON \
        -DWITH_TBB=ON \
        -DWITH_EIGEN=ON \
        -DWITH_V4L=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-${OPENCV_VERSION}/modules \
        -D OPENCV_ENABLE_NONFREE=ON \
        -DCMAKE_INSTALL_PREFIX=$(python3.8 -c "import sys; print(sys.prefix)") \
        -DPYTHON_EXECUTABLE=$(which python3.8) \
        -DPYTHON_INCLUDE_DIR=$(python3.8 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_PACKAGES_PATH=$(python3.8 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        .. \
    && make install \
    && rm /cv.zip /contrib.zip \
    && rm -r /opencv-${OPENCV_VERSION} /opencv_contrib-${OPENCV_VERSION}
```

1.  我们将 OpenCV Python 二进制文件链接到适当的位置，以便解释器可以找到它：

```py
RUN ln -s \
  /usr/local/python/cv2/python-3.8/cv2.cpython-38m-x86_64-linux-gnu.so \
  /usr/local/lib/python3.8/site-packages/cv2.so
```

如果您使用的基镜像与`python:3.8`不同，这种链接可能重复或导致错误。

1.  我们安装了本书中使用的其他 Python 包：

```py
RUN pip install --upgrade pip && pip install --no-cache-dir pathlib2 wxPython==4.0.5

RUN pip install --upgrade pip && pip install --no-cache-dir scipy==1.4.1 matplotlib==3.1.2 requests==2.22.0 ipython numba==0.48.0 jupyterlab==1.2.6 rawpy==0.14.0
```

因此，现在我们已经组合了 Dockerfile，我们可以按照以下方式构建相应的 Docker 镜像：

```py
$ docker build -f dockerfiles/Dockerfile  -t cv  dockerfiles
```

我们将镜像命名为 `cv`，并将位于 `dockerfiles/Dockerfile` 的 Dockerfile 传递给构建镜像。当然，您可以将您的 Dockerfile 放在任何其他位置。Docker 的最后一个参数是必需的，它指定了一个可能被使用的上下文；例如，如果 Dockerfile 包含从相对路径复制文件的指令。在我们的情况下，我们没有这样的指令，并且它可以是任何有效的路径。

一旦构建了镜像，我们就可以按照以下方式启动 `docker` 容器：

```py
$ docker run --device /dev/video0 --env DISPLAY=$DISPLAY  -v="/tmp/.X11-unix:/tmp/.X11-unix:rw"  -v `pwd`:/book -it book
```

在这里，我们传递了 `DISPLAY` 环境变量，挂载了 `/tmp/.X11-unix`，并指定了 `/dev/video0` 设备，以便容器可以使用桌面环境并连接到相机，其中容器在本书的大部分章节中使用。

如果 Docker 容器无法连接到您系统的 *X* 服务器，您可能需要在您的系统上运行 **`$ xhost +local:docker`** 以允许连接。

因此，现在我们已经启动并运行了组合的 Docker 镜像，让我们来探讨如何使用 Docker 支持 GPU 加速。

# 使用 GPU

我们使用 Docker 创建的环境对机器的设备访问有限。特别是，您已经看到，在运行 Docker 容器时，我们已经指定了相机设备，并且挂载了 `/tmp/.X11-unix` 以允许 Docker 容器连接到正在运行的桌面环境。

当我们拥有自定义设备，如 GPU 时，集成过程变得更加复杂，因为 Docker 容器需要适当的方式与设备通信。幸运的是，对于 **NVIDIA GPU**，这个问题通过 **NVIDIA Container Toolkit**（[`github.com/NVIDIA/nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)）得到了解决。

在安装工具包之后，您可以构建和运行带有 GPU 加速的 Docker 容器。Nvidia 提供了一个基础镜像，这样您就可以在其上构建您的镜像，而无需担心对 GPU 的适当访问。要求是在您的系统上安装了适当的 Nvidia 驱动程序，并且有一个 Nvidia GPU。

在我们的情况下，我们主要使用 GPU 来加速 TensorFlow。TensorFlow 本身提供了一个可以用于带有 GPU 加速运行 TensorFlow 的镜像。因此，为了有一个带有 GPU 加速的容器，我们可以简单地选择 TensorFlow 的 Docker 镜像，并在其上安装所有其他软件，如下所示：

```py
FROM tensorflow/tensorflow:2.1.0-gpu-py3
```

这个声明将选择带有 GPU 加速和 Python 3 支持的 TensorFlow 版本 `2.1.0`。请注意，这个版本的 TensorFlow 镜像使用 **Python 3.6**。尽管如此，您可以使用 Dockerfile 的剩余部分来构建 附录 A 中描述的 CPU，*分析和加速您的应用程序*，您将能够运行本书中的代码。

一旦创建完图像，在启动容器时，你唯一需要做的修改就是传递一个额外的参数：`--runtime=nvidia`。
