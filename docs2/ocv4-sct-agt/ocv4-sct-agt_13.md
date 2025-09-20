# 第十章：学习更多关于 OpenCV 中特征检测的知识

在 第四章*，使用优雅的手势控制手机应用* 中，我们使用了 Good Features to Track 算法来检测图像中的可追踪特征。OpenCV 提供了几个其他特征检测算法的实现。其中另外两种算法，称为最小特征值角和 Harris 角，是 Good Features to Track 的前身，它们在原有基础上进行了改进。一个官方教程在代码示例中展示了如何使用特征值角和 Harris 角，请参阅 [`docs.opencv.org/master/d9/dbc/tutorial_generic_corner_detector.h`](https://docs.opencv.org/master/d9/dbc/tutorial_generic_corner_detector.html)[tml](https://docs.opencv.org/master/d9/dbc/tutorial_generic_corner_detector.html)。

OpenCV 中一些更高级的特征检测算法被称为 FAST、ORB、SIFT、SURF 和 FREAK。与 Good Features to Track 相比，这些更高级的替代方案评估了更大范围的可能特征，计算成本也更高。对于我们这样的基本光流任务来说，这有点过度。一旦我们检测到人脸，我们在这个区域不需要很多特征来区分垂直运动（点头）和水平运动（摇晃）。对于我们的手势识别任务，以快速帧率运行远比运行大量特征更重要。另一方面，一些计算机视觉任务需要大量特征。图像识别就是一个很好的例子。如果我们在一幅《蒙娜丽莎》的海报上涂上红色口红，得到的图像就不是《蒙娜丽莎》（至少不是达芬奇的版本）。图像的细节可能被认为是其身份的基本要素。然而，光线或视角的变化并不会改变图像的身份，因此特征检测和匹配系统仍然需要对这些变化具有一定的鲁棒性。

对于涵盖图像识别和跟踪的项目，请参阅 Packt Publishing 出版的《Android Application Programming with OpenCV 3》的第四章、第五章和第六章。

对于 OpenCV 中几个特征检测器和匹配器的基准测试，请参阅 Ievgen Khvedchenia 博客上的系列文章，包括 [`computer-vision-talks.com/2011-07-13-comparison-of-the-opencv-feature-detection-algorithms/`](http://computer-vision-talks.com/2011-07-13-comparison-of-the-opencv-feature-detection-algorithms/)。你还可以在 Roy Shilkrot 和 David Millán Escrivá（Packt Publishing，2018 年）所著的《Mastering OpenCV 4》的第九章“**为任务找到最佳的 OpenCV 算法*”中的“*算法示例比较性能测试*”部分找到更多最新的基准测试。

关于多个算法及其 OpenCV 实现的教程，请参阅官方 OpenCV-Python 教程中的*特征检测与描述*部分，链接为[`docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html`](http://docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html).
