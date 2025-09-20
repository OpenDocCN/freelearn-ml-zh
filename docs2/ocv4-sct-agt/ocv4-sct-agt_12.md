# 使 WxUtils.py 与 Raspberry Pi 兼容

在 第二章*全球寻找豪华住宿* 中，我们编写了一个文件，`WxUtils.py`，其中包含一个实用函数 `wxBitmapFromCvImage`，用于将 OpenCV 图像转换为 wxPython 位图。我们在本书的 Python 项目中使用了这个实用函数。

我们对 `wxBitmapFromCvImage` 的实现部分依赖于 wxPython 的 `wx.BitmapFromBuffer` 函数。在某些版本的 Raspberry Pi 和 Raspbian 上，`wx.BitmapFromBuffer` 由于一个特定于平台的错误而失败。作为解决方案，我们可以使用 `wx.ImageFromBuffer` 和 `wx.BitmapFromImage` 函数进行一个效率较低的两步转换。以下是一些代码，用于检查我们是否在运行早期的 Raspberry Pi 模型（基于 CPU 型号），并相应地实现我们的 `wxBitmapFromCVImage` 函数：

```py
import numpy # Hint to PyInstaller
import cv2
import wx

WX_MAJOR_VERSION = int(wx.__version__.split('.')[0])

# Try to determine whether we are on Raspberry Pi.
IS_RASPBERRY_PI = False
try:
    with open('/proc/cpuinfo') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Hardware') and \
                    line.endswith('BCM2708'):
                IS_RASPBERRY_PI = True
                break
except:
    pass

if IS_RASPBERRY_PI:
    def wxBitmapFromCvImage(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        wxImage = wx.ImageFromBuffer(w, h, image)
        if WX_MAJOR_VERSION < 4:
            bitmap = wx.BitmapFromImage(wxImage)
        else:
            bitmap = wx.Bitmap(wxImage)
        return bitmap
else:
    def wxBitmapFromCvImage(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        # The following conversion fails on Raspberry Pi.
        if WX_MAJOR_VERSION < 4:
            bitmap = wx.BitmapFromBuffer(w, h, image)
        else:
            bitmap = wx.Bitmap.FromBuffer(w, h, image)
        return bitmap

```

如果你用前面显示的代码替换 `WxUtils.py` 的内容，我们的 `wxBitmapFromCvImage` 工具函数将在 Raspberry Pi 以及其他系统上工作。
