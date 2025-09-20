# 评估

# 第一章

1.  提高性能的三种选项如下：

    +   拥有更快的时钟速度

    +   单个处理器每个时钟周期完成更多的工作

    +   许多可以并行工作的小型处理器。这个选项被 GPU 用来提高性能。

1.  正确

1.  CPU 的设计是为了提高延迟，而 GPU 的设计是为了提高吞吐量。

1.  汽车需要 4 小时才能到达目的地，但它只能容纳 5 人，而可以容纳 40 人的公交车需要 6 小时才能到达目的地。公交车每小时可以运输 6.66 人，而汽车每小时可以运输 1.2 人。因此，汽车具有更好的延迟，而公交车具有更好的吞吐量。

1.  图像不过是一个二维数组。大多数计算机视觉应用都涉及这些二维数组的处理。这涉及到对大量数据进行类似操作，这些操作可以通过 GPU 高效地执行。因此，GPU 和 CUDA 在计算机视觉应用中非常有用。

1.  错误

1.  `printf`语句在主机上执行

# 第二章

1.  通过传递参数作为值来减去两个数字的 CUDA 程序如下：

```py
include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void gpuSub(int d_a, int d_b, int *d_c) 
{
 *d_c = d_a - d_b;
}
int main(void) 
{
  int h_c;
  int *d_c;
  cudaMalloc((void**)&d_c, sizeof(int));
 gpuSub << <1, 1 >> > (4, 1, d_c);
 cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
 printf("4-1 = %d\n", h_c);
 cudaFree(d_c);
 return 0;
}

```

1.  通过传递参数作为引用来乘以两个数字的 CUDA 程序如下：

```py
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
 __global__ void gpuMul(int *d_a, int *d_b, int *d_c) 
{
 *d_c = *d_a * *d_b;
}
int main(void) 
{
 int h_a,h_b, h_c;
 int *d_a,*d_b,*d_c;
 h_a = 1;
 h_b = 4;
 cudaMalloc((void**)&d_a, sizeof(int));
 cudaMalloc((void**)&d_b, sizeof(int));
 cudaMalloc((void**)&d_c, sizeof(int));
 cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
 cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);
 gpuMul << <1, 1 >> > (d_a, d_b, d_c);
 cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
 printf("Passing Parameter by Reference Output: %d + %d = %d\n", h_a, h_b, h_c);
 cudaFree(d_a);
 cudaFree(d_b);
 cudaFree(d_c);
 return 0;
 }
```

1.  为`gpuMul`内核启动 5000 个线程的三种方法如下：

```py
1\. gpuMul << <25, 200 >> > (d_a, d_b, d_c);
2\. gpuMul << <50, 100 >> > (d_a, d_b, d_c);
3\. gpuMul << <10, 500 >> > (d_a, d_b, d_c);
```

1.  错误

1.  查找具有 5.0 或更高版本的 GPU 设备的程序如下

```py
int main(void) 
{ 
  int device; 
  cudaDeviceProp device_property; 
  cudaGetDevice(&device); 
  printf("ID of device: %d\n", device); 
  memset(&device_property, 0, sizeof(cudaDeviceProp)); 
  device_property.major = 5; 
  device_property.minor = 0; 
  cudaChooseDevice(&device, &device_property); 
  printf("ID of device which supports double precision is: %d\n", device);                                                                         
  cudaSetDevice(device); 
} 
```

1.  查找数字立方的 CUDA 程序如下：

```py
#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#define N 50
__global__ void gpuCube(float *d_in, float *d_out) 
{
     //Getting thread index for current kernel
     int tid = threadIdx.x; // handle the data at this index
     float temp = d_in[tid];
     d_out[tid] = temp*temp*temp;
 }
int main(void) 
{
     float h_in[N], h_out[N];
     float *d_in, *d_out;
     cudaMalloc((void**)&d_in, N * sizeof(float));
     cudaMalloc((void**)&d_out, N * sizeof(float));
      for (int i = 0; i < N; i++) 
    {
         h_in[i] = i;
     }
   cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
   gpuSquare << <1, N >> >(d_in, d_out);
  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Cube of Number on GPU \n");
     for (int i = 0; i < N; i++) 
     {
         printf("The cube of %f is %f\n", h_in[i], h_out[i]);
     }
     cudaFree(d_in);
     cudaFree(d_out);
     return 0;
 }
```

1.  特定应用的通信模式如下所示：

    1.  图像处理 - 模板

    1.  移动平均 - 聚合

    1.  按升序排序数组 - 散射

    1.  在数组中查找数字的立方 - 映射

# 第三章

1.  选择线程数和块数的最佳方法如下：

```py
gpuAdd << <512, 512 >> >(d_a, d_b, d_c);
```

每个块可以启动的线程数量有限，对于最新的处理器来说，这个数量是 512 或 1024。同样，每个网格的块数量也有限制。因此，如果有大量线程，那么最好通过少量块和线程来启动内核，如上所述。

1.  以下是为查找 50000 个数字的立方而编写的 CUDA 程序：

```py
#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#define N 50000
__global__ void gpuCube(float *d_in, float *d_out) 
{
      int tid = threadIdx.x + blockIdx.x * blockDim.x; 
while (tid < N)
{
    float temp = d_in[tid];
    d_out[tid] = temp*temp*temp;
    tid += blockDim.x * gridDim.x;
 }
}
int main(void) 
{
     float h_in[N], h_out[N];
     float *d_in, *d_out;
     cudaMalloc((void**)&d_in, N * sizeof(float));
     cudaMalloc((void**)&d_out, N * sizeof(float));
      for (int i = 0; i < N; i++) 
    {
         h_in[i] = i;
     }
   cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
   gpuSquare << <512, 512 >> >(d_in, d_out);
  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Cube of Number on GPU \n");
     for (int i = 0; i < N; i++) 
     {
         printf("The cube of %f is %f\n", h_in[i], h_out[i]);
     }
     cudaFree(d_in);
     cudaFree(d_out);
     return 0;
 }
```

1.  正确，因为它只需要访问局部内存，这是一种更快的内存。

1.  当内核的变量不适合寄存器文件时，它们使用局部内存。这被称为寄存器溢出。因为一些数据不在寄存器中，它将需要更多时间从内存中检索它。这将花费更多时间，因此程序的性能将受到影响。

1.  不，因为所有线程都在并行运行。所以数据可能在写入之前就被读取，因此可能不会给出期望的输出。

1.  正确。在原子操作中，当一个线程正在访问特定的内存位置时，其他所有线程都必须等待。当许多线程访问相同的内存位置时，这将产生时间开销。因此，原子操作会增加 CUDA 程序的执行时间。

1.  Stencil 通信模式非常适合纹理内存。

1.  当在 `if` 语句中使用 `__syncthreads` 指令时，对于具有此条件的线程，`false` 永远不会到达这个点，`__syncthreads` 将持续等待所有线程到达这个点。因此，程序将永远不会终止。

# 第四章

1.  CPU 计时器将包括操作系统中的线程延迟和调度的时间开销，以及其他许多因素。使用 CPU 测量的时间也将取决于高精度 CPU 计时器的可用性。主机在 GPU 内核运行时经常执行异步计算，因此 CPU 计时器可能无法给出内核执行的准确时间。

1.  从 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp` 打开 Nvidia Visual profiler。然后，转到 -> 新会话并选择矩阵乘法示例的 `.exe` 文件。您可以可视化您代码的性能。

1.  除以零、变量类型或大小不正确、不存在变量、下标超出范围等是语义错误的例子。

1.  可以给出线程发散的例子如下：

```py
__global__ void gpuCube(float *d_in, float *d_out) 
{
     int tid = threadIdx.x; 
if(tid%2 == 0)
{
     float temp = d_in[tid];
     d_out[tid] = temp*temp*temp;
 }
else
{
    float temp = d_in[tid];
    d_out[tid] = temp*temp*temp;
}
}
```

在代码中，奇数和偶数线程执行不同的操作，因此它们完成所需的时间不同。在 `if` 语句之后，这些线程将再次合并。这将产生时间开销，因为快速线程必须等待慢速线程。这将降低代码的性能。

1.  `cudaHostAlloc` 函数应谨慎使用，因为这种内存不会被交换到磁盘上；您的系统可能会耗尽内存。这可能会影响系统上运行的其他应用程序的性能。

1.  在 CUDA 流操作中，操作顺序很重要，因为我们希望重叠内存复制操作与内核执行操作。因此，操作队列应设置为这些操作可以相互重叠，否则使用 CUDA 流不会提高程序的性能。

1.  对于 1024 x 1024 的图像，线程数应为 32x32（如果您的系统支持每个块 1024 个线程），块数也应为 32 x 32，这可以通过将图像大小除以每个块线程数来确定。

# 第五章

1.  图像处理和计算机视觉领域之间存在差异。图像处理关注通过修改像素值来提高图像的视觉质量，而计算机视觉关注从图像中提取重要信息。因此，在图像处理中，输入和输出都是图像，而在计算机视觉中，输入是图像，但输出是从该图像中提取的信息。

1.  OpenCV 库在 C、C++、Java 和 Python 语言中都有接口，并且可以在 Windows、Linux、Mac 和 Android 等所有操作系统上使用，而无需修改单行代码。这个库还可以利用多核处理。它可以利用 OpenGL 和 CUDA 进行并行处理。由于 OpenCV 轻量级，它也可以在树莓派等嵌入式平台上使用。这使得它在实际场景中部署计算机视觉应用成为理想选择。

1.  初始化图像为红色的命令如下：

```py
 Mat img3(1960,1960, CV_64FC3, Scalar(0,0,255) )
```

1.  从网络摄像头捕获视频并将其存储在磁盘上的程序如下：

```py
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
   VideoCapture cap(0); 
   if (cap.isOpened() == false) 
   {
     cout << "Cannot open Webcam" << endl;
     return -1;
 }
  Size frame_size(640, 640);
  int frames_per_second = 30;

  VideoWriter v_writer("images/video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), frames_per_second, frame_size, true); 
  cout<<"Press Q to Quit" <<endl;
  String win_name = "Webcam Video";
  namedWindow(win_name); //create a window
   while (true)
   {
     Mat frame;
     bool flag = cap.read(frame); // read a new frame from video 
     imshow(win_name, frame);
     v_writer.write(frame);
  if (waitKey(1) == 'q')
  {
     v_writer.release(); 
     break;
  }
 }
return 0;
}
```

1.  OpenCV 使用 BGR 颜色格式来读取和显示图像。

1.  从网络摄像头捕获视频并将其转换为灰度的程序如下：

```py
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
   VideoCapture cap(0); 
 if (cap.isOpened() == false) 
 {
    cout << "Cannot open Webcam" << endl;
    return -1;
 }
 cout<<"Press Q to Quit" <<endl;
 String win_name = "Webcam Video";
 namedWindow(win_name); //create a window
 while (true)
 {
    Mat frame;
    bool flag = cap.read(frame); // read a new frame from video 
    cvtColor(frame, frame,cv::COLOR_BGR2GRAY);
    imshow(win_name, frame);
  if (waitKey(1) == 'q')
  {
      break;
  }
 }
return 0;
}
```

1.  测量加法和减法操作性能的 OpenCV 程序如下：

```py
#include <iostream>
#include "opencv2/opencv.hpp"

int main (int argc, char* argv[])
{
    //Read Two Images 
    cv::Mat h_img1 = cv::imread("images/cameraman.tif");
    cv::Mat h_img2 = cv::imread("images/circles.png");
    //Create Memory for storing Images on device
    cv::cuda::GpuMat d_result1,d_result2,d_img1, d_img2;
    cv::Mat h_result1,h_result2;
int64 work_begin = getTickCount(); 
    //Upload Images to device     
    d_img1.upload(h_img1);
    d_img2.upload(h_img2);

    cv::cuda::add(d_img1,d_img2, d_result1);
    cv::cuda::subtract(d_img1, d_img2,d_result2);
    //Download Result back to host
    d_result1.download(h_result1);
     d_result2.download(h_result2);
    int64 delta = getTickCount() - work_begin;
//Frequency of timer
    double freq = getTickFrequency();
    double work_fps = freq / delta;
    std::cout<<"Performance of Thresholding on CPU: " <<std::endl;
    std::cout <<"Time: " << (1/work_fps) <<std::endl;   
    cv::waitKey();
    return 0;
}
```

1.  OpenCV 程序用于执行位运算 AND 和 OR 操作如下：

```py
include <iostream>
#include "opencv2/opencv.hpp"

int main (int argc, char* argv[])
{
    cv::Mat h_img1 = cv::imread("images/cameraman.tif");
    cv::Mat h_img2 = cv::imread("images/circles.png");
    cv::cuda::GpuMat d_result1,d_result2,d_img1, d_img2;
    cv::Mat h_result1,h_result2;
    d_img1.upload(h_img1);
    d_img2.upload(h_img2);

    cv::cuda::bitwise_and(d_img1,d_img2, d_result1);
    cv::cuda::biwise_or(d_img1, d_img2,d_result2);

    d_result1.download(h_result1);
     d_result2.download(h_result2);
cv::imshow("Image1 ", h_img1);
    cv::imshow("Image2 ", h_img2);
    cv::imshow("Result AND operation ", h_result1);
cv::imshow("Result OR operation ", h_result2);
    cv::waitKey();
    return 0;
}
```

# 第六章

1.  打印任何颜色图像在`(200,200)`位置像素强度的 OpenCV 函数如下：

```py
cv::Mat h_img2 = cv::imread("images/autumn.tif",1);
cv::Vec3b intensity1 = h_img1.at<cv::Vec3b>(cv::Point(200, 200));
std::cout<<"Pixel Intensity of color Image at (200,200) is:" << intensity1 << std::endl;
```

1.  使用双线性插值方法将图像调整大小到`(300,200)`像素的 OpenCV 函数如下：

```py
cv::cuda::resize(d_img1,d_result1,cv::Size(300, 200), cv::INTER_LINEAR);
```

1.  使用`AREA`插值将图像上采样 2 倍的 OpenCV 函数如下：

```py
int width= d_img1.cols;
int height = d_img1.size().height;
cv::cuda::resize(d_img1,d_result2,cv::Size(2*width, 2*height), cv::INTER_AREA); 
```

1.  错误。随着滤波器大小的增加，模糊程度也会增加。

1.  错误。中值滤波器不能去除高斯噪声。它可以去除椒盐噪声。

1.  在应用拉普拉斯算子以去除噪声敏感性之前，必须使用平均或高斯滤波器对图像进行模糊处理。

1.  实现顶帽和黑帽形态学操作的 OpenCV 函数如下：

```py
cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5)); 
  d_img1.upload(h_img1);
  cv::Ptr<cv::cuda::Filter> filtert,filterb;
  filtert = cv::cuda::createMorphologyFilter(cv::MORPH_TOPHAT,CV_8UC1,element);
  filtert->apply(d_img1, d_resulte);
  filterb = cv::cuda::createMorphologyFilter(cv::MORPH_BLACKHAT,CV_8UC1,element);
  filterb->apply(d_img1, d_resultd);
```

# 第七章

1.  从视频中检测黄色物体的 OpenCV 代码如下：请注意，这里没有重复样板代码。

```py
cuda::cvtColor(d_frame, d_frame_hsv, COLOR_BGR2HSV);

//Split HSV 3 channels
cuda::split(d_frame_hsv, d_frame_shsv);

//Threshold HSV channels for Yellow color
cuda::threshold(d_frame_shsv[0], d_thresc[0], 20, 30, THRESH_BINARY);
cuda::threshold(d_frame_shsv[1], d_thresc[1], 100, 255, THRESH_BINARY);
cuda::threshold(d_frame_shsv[2], d_thresc[2], 100, 255, THRESH_BINARY);

//Bitwise AND the channels
cv::cuda::bitwise_and(d_thresc[0], d_thresc[1],d_intermediate);
cv::cuda::bitwise_and(d_intermediate, d_thresc[2], d_result);
d_result.download(h_result);
imshow("Thresholded Image", h_result); 
imshow("Original", frame);
```

1.  当物体的颜色与背景颜色相同时，基于颜色的目标检测将失败。即使有光照变化，也可能失败。

1.  Canny 边缘检测算法的第一步是高斯模糊，这可以去除图像中存在的噪声。之后，计算梯度。因此，检测到的边缘将比之前看到的其他边缘检测算法受噪声影响更小。

1.  当图像受到高斯或椒盐噪声的影响时，霍夫变换的结果非常差。为了改善结果，必须在预处理步骤中通过高斯和中值滤波器对图像进行滤波。

1.  当计算 FAST 关键点的强度阈值较低时，则更多的关键点将通过段测试并被分类为关键点。随着这个阈值的增加，检测到的关键点数量将逐渐减少。

1.  在 SURF 中，Hessian 阈值的较大值将导致更少但更显著的特征点，而较小值将导致更多但不太显著的特征点。

1.  当 Haar 级联的缩放因子从 1.01 增加到 1.05 时，图像大小在每一尺度上都会以更大的因子减小。因此，每帧需要处理的图像更少，这减少了计算时间；然而，这可能导致无法检测到某些对象。

1.  `MoG` 相比于 `GMG` 算法在背景减法方面更快且噪声更少。可以将开闭等形态学操作应用于 GMG 的输出，以减少存在的噪声。

# 第八章

1.  Jetson TX1 在每秒 Tera 级浮点运算性能方面优于 Raspberry Pi。因此，Jetson TX1 可以用于计算密集型应用，如计算机视觉和深度学习，以实现实时部署。

1.  Jetson TX1 开发板支持多达六个 2 通道或三个 4 通道相机。它附带一个 500 万像素的相机。

1.  必须使用 USB 集线器来连接 Jetson TX1 与超过两个 USB 外设。

1.  True

1.  False. Jetson TX1 包含一个 1.73 GHz 运行的 ARM Cortex A57 四核 CPU。

1.  尽管 Jetson TX1 预装了 Ubuntu，但它不包含计算机视觉应用所需的任何软件包。Jetpack 包含 Tegra (L4T) 板支持包的 Linux，TensorRT，用于计算机视觉应用中的深度学习推理，最新的 CUDA 工具包，cuDNN，这是 CUDA 深度神经网络库，Visionworks，也用于计算机视觉和深度学习应用，以及 OpenCV。因此，通过安装 Jetpack，我们可以快速安装构建计算机视觉应用所需的全部软件包。

# 第九章

1.  Jetson TX1 上的 GPU 设备全局内存大约为 4 GB，GPU 时钟速度约为 1 GHz。这个时钟速度比本书之前使用的 GeForce 940 GPU 慢。内存时钟速度仅为 13 MHz，而 GeForce 940 为 2.505 GHz，这使得 Jetson TX1 更慢。L2 缓存为 256 KB，而 GeForce 940 为 1 MB。大多数其他属性与 GeForce 940 相似。

1.  True

1.  在最新的 Jetpack 中，OpenCV 没有编译 CUDA 支持，也没有 GStreamer 支持，这是从代码中访问相机所需的。因此，移除 Jetpack 中包含的 OpenCV 安装，并使用 CUGA 和 GStreamer 支持编译新的 OpenCV 版本是个好主意。

1.  False. OpenCV 可以从连接到 Jetson TX1 板的 USB 和 CSI 相机捕获视频。

1.  True. CSI 相机更接近硬件，因此读取帧的速度比 USB 相机快，因此在计算密集型应用中最好使用 CSI 相机。

1.  Python OpenCV 绑定不支持 CUDA 加速，因此对于计算密集型任务，最好使用 C++ OpenCV 绑定。

1.  No. Jetson TX1 预装了 python2 和 python3 解释器，同时 OpenCV 也为 Jetson TX1 编译了；它还安装了 python 二进制文件，因此无需单独安装 python OpenCV 绑定。

# 第十章

1.  Python 是开源的，拥有庞大的用户社区，他们通过模块为语言做出贡献。这些模块可以轻松地用少量代码在短时间内开发应用程序。Python 语言的语法易于阅读和解释，这使得它对新程序员来说更容易学习。它是一种允许逐行执行代码的解释型语言。这些都是 Python 相对于 C/C++ 的几个优点。

1.  在编译型语言中，整个代码被检查并转换为机器代码，而在解释型语言中，每次只翻译一条语句。解释型语言分析源代码所需的时间较少，但与编译型语言相比，整体执行时间较慢。解释型语言不会像编译型语言那样生成中间代码。

1.  错误。Python 是一种解释型语言，这使得它比 C/C++ 慢。

1.  PyOpenCL 可以利用任何图形处理单元，而 PyCUDA 需要 Nvidia GPU 和 CUDA 工具包。

1.  正确。Python 允许在 Python 脚本中包含 C/C++ 代码，因此计算密集型任务可以写成 C/C++ 代码以实现更快的处理，并为它创建 Python 包装器。PyCUDA 可以利用这一功能来处理内核代码。

# 第十一章

1.  C/C++ 编程语言用于在 `SourceModule` 类中编写内核函数，并且这个内核函数由 `nvcc`（Nvidia C）编译器编译。

1.  内核调用函数如下：

```py
myfirst_kernel(block=(512,512,1),grid=(1024,1014,1))
```

1.  错误。在 PyCUDA 程序中，块执行的顺序是随机的，PyCUDA 程序员无法确定。

1.  驱动类中的指令消除了为数组单独分配内存、将其上传到设备以及将结果下载回主机的要求。所有操作都在内核调用期间同时执行。这使得代码更简单，更容易阅读。

1.  在数组中每个元素加二的 PyCUDA 代码如下所示：

```py
import pycuda.gpuarray as gpuarray
import numpy
import pycuda.driver as drv

start = drv.Event()
end=drv.Event()
start.record()
start.synchronize()
n=10
h_b = numpy.random.randint(1,5,(1,n))
d_b = gpuarray.to_gpu(h_b.astype(numpy.float32))
h_result = (d_b + 2).get()
end.record()
end.synchronize()

print("original array:")
print(h_b)
print("doubled with gpuarray:")
print(h_result)
secs = start.time_till(end)*1e-3
print("Time of adding 2 on GPU with gpuarray")
print("%fs" % (secs))
```

1.  使用 Python 时间测量选项来测量 PyCUDA 程序的性能不会给出准确的结果。它将包括许多其他因素中的线程延迟在操作系统中的时间开销和调度。使用时间类测量的时间也将取决于高精度 CPU 定时器的可用性。很多时候，主机在进行异步计算的同时 GPU 内核正在运行，因此 Python 的 CPU 计时器可能无法给出内核执行的正确时间。我们可以通过使用 CUDA 事件来克服这些缺点。

1.  正确

# 第十二章

1.  错误。这一行代表一个读取-修改-写入操作，当多个线程试图增加相同的内存位置时，如直方图计算的情况，可能会产生错误的结果。

1.  在使用共享内存的情况下，较少的线程试图访问共享内存中的 256 个内存元素，而不是没有共享内存时所有线程的情况。这将有助于减少原子操作中的时间开销。

1.  在使用共享内存的情况下，内核调用函数如下：

```py
atomic_hist(
        drv.Out(h_result), drv.In(h_a), numpy.uint32(SIZE),
        block=(n_threads,1,1), grid=(NUM_BIN,1),shared= 256*4)
```

在调用内核时，应该定义共享内存的大小。这可以通过在内核调用函数中使用共享参数来指定。

1.  直方图是一种统计特征，它提供了关于图像对比度和亮度的关键信息。如果它具有均匀分布，那么图像将具有良好的对比度。直方图还传达了关于图像亮度的信息。如果直方图集中在图表的左侧，那么图像将太暗，如果集中在右侧，那么图像将太亮。

1.  真的。因为 RGB 和 BGR 颜色格式相同，只是通道的顺序不同。转换的方程式仍然保持不变。

1.  与多维线程和块相比，处理单维线程和块更简单。它简化了内核函数内部的索引机制，因此在每个章节中出现的示例中都进行了这种简化。如果我们正在处理多维线程和块，则这不是强制性的。

1.  `imshow`函数，用于在屏幕上显示图像，需要一个无符号整数的图像。因此，在屏幕上显示之前，所有由内核函数计算的结果都转换为`numpy`库的`uint8`数据类型。
