# 前言

计算机视觉系统被部署在北冰洋上，以在夜间发现冰山。它们被飞越亚马逊雨林，以创建火灾、灾害和非法伐木的空中地图。它们被设置在全世界各地的港口和机场，以扫描嫌疑人和违禁品。它们被送到马里亚纳海沟的深处，以引导自主潜艇。它们被用于手术室，帮助外科医生可视化手术计划以及患者的当前状况。它们作为热寻的防空火箭的制导系统从战场发射。

我们可能很少——或者永远——访问这些地方。然而，故事往往鼓励我们想象极端的环境以及一个人在这些严酷条件下的工具依赖。也许很合适，当代小说中最受欢迎的角色之一是一个几乎普通的男人（英俊，但不过分英俊；聪明，但不过分聪明），他穿着西装，为英国政府工作，总是选择同样的饮料，同样的女人，同样的语调来开玩笑，并且带着一套奇特的工具被派去执行危险的任务。

邦德，詹姆斯·邦德。

本书严肃地讨论了有用的技术和技巧，并从间谍小说中汲取了大量的灵感。邦德系列电影在侦探、伪装、智能设备、图像捕捉以及有时甚至计算机视觉方面都充满了创意。有了想象力，加上对新技能的执着学习，我们可以成为与邦德的工程师 Q 相媲美的下一代设备制造商！

# 本书面向的对象

本书是为那些想要将计算机视觉变成他们生活方式中实用和有趣一部分的发明家（和间谍）而写的。你应该已经熟悉二维图形概念、面向对象的语言、GUI、网络和命令行。本书不假设你对任何特定库或平台有经验。详细的说明涵盖了从设置开发环境到部署完成的应用程序的所有内容。

想要学习多种技术和技巧，并将它们整合在一起，是非常有益的！本书将帮助你拓展视野，了解计算机视觉相关的多种系统和应用领域，并将帮助你应用多种方法来检测、识别、跟踪和增强人脸、物体和动作。

# 本书涵盖的内容

第一章，《准备任务》，帮助我们安装 OpenCV、Python 开发环境和 Android 开发环境在 Windows、macOS 或 Linux 系统上。在这一章中，我们还安装了 Windows 或 macOS 上的 Unity 开发环境。

第二章，*在全球范围内寻找豪华住宿*，帮助我们根据色彩方案对房地产图像进行分类。我们是在豪华住宅外面还是在斯大林式公寓里面？在这一章中，我们使用搜索引擎中的分类器来标记其图像结果。

第三章，*训练一个智能闹钟来识别恶棍和他的猫*，帮助我们检测和识别人脸和猫脸，作为控制闹钟的手段。恩斯特·斯塔罗·布洛菲尔德带着他的蓝眼睛安哥拉猫回来了吗？

第四章，*用优雅的手势控制手机应用*，帮助我们检测动作并识别手势，作为在智能手机上控制猜谜游戏的手段。即使其他人不知道，手机也知道邦德是在点头。

第五章，*给你的汽车配备后视摄像头和危险检测功能*，帮助我们检测汽车车灯，分类它们的颜色，估计与它们的距离，并向驾驶员提供反馈。那辆车是在跟踪我们吗？

第六章，*基于笔和纸草图创建物理模拟*，帮助我们把迷宫中的球谜画在纸上，并看到它作为智能手机上的物理模拟栩栩如生。物理和时机是关键！

第七章，*用运动放大摄像头看到心跳*，帮助我们实时放大实时视频中的运动，使人的心跳和呼吸变得清晰可见。看到激情！

第八章，*停止时间，像蜜蜂一样看世界*，通过采用高速、红外或紫外成像的专业相机来改进上一章的项目。超越人类视觉的极限！

附录 A，*使 WxUtils.py 与 Raspberry Pi 兼容*，帮助我们解决影响某些 Raspberry Pi 环境中 wxPython GUI 库的兼容性问题。

附录 B，*在 OpenCV 中学习更多关于特征检测的知识*，帮助我们发现 OpenCV 更多特征检测的能力，而不仅仅是我们在本书的项目中使用的那些。

附录 C，*与蛇一起奔跑（或，Python 的第一步）*，帮助我们学习在 Python 环境中运行 Python 代码并测试 OpenCV 安装。

# 为了充分利用这本书

本书支持多种操作系统作为开发环境，包括 Windows 7 SP 1 或更高版本、macOS X 10.7（Lion）或更高版本、Debian Jessie、Raspbian、Ubuntu 14.04 或更高版本、Linux Mint 17 或更高版本、Fedora 28 或更高版本、**红帽企业 Linux**（**RHEL**）8 或更高版本、CentOS 8 或更高版本、openSUSE Leap 42.3、openSUSE Leap 15.0 或更高版本，以及 openSUSE Tumbleweed。

本书包含六个项目，以下为项目要求：

+   其中四个项目在 Windows、macOS 或 Linux 上运行，并需要一个摄像头。这些项目可以选择使用 Raspberry Pi 或其他运行 Linux 的单板计算机。

+   一个项目运行在 Android 5.0（Lollipop）或更高版本，并需要一个前置摄像头（大多数 Android 设备都有）。

+   另一个项目运行在 Android 4.1（Jelly Bean）或更高版本，需要一个后置摄像头和重力传感器（大多数 Android 设备都有）。对于开发，它需要一个 Windows 或 macOS 机器和大约价值 95 美元的游戏开发软件。

书中涵盖了所有必需库和工具的设置说明。还包括了 Raspberry Pi 的可选设置说明。

# 下载示例代码文件

您可以从[www.packtpub.com](http://www.packtpub.com)上的账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)上登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

下载文件后，请确保您使用最新版本的以下软件解压缩或提取文件夹：

+   Windows 的 WinRAR/7-Zip

+   Mac 的 Zipeg/iZip/UnRarX

+   Linux 的 7-Zip/PeaZip

本书还托管在 GitHub 上，地址为[`github.com/PacktPublishing/OpenCV-4-for-Secret-Agents-Second-Edition`](https://github.com/PacktPublishing/OpenCV-4-for-Secret-Agents-Second-Edition)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)上找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789345360_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789345360_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“您可以通过编辑 `/etc/modules` 来检查 `bcm2835-v4l2` 是否已经列在那里。”

代码块设置如下：

```py
set PYINSTALLER=pyinstaller

REM Remove any previous build of the app.
rmdir build /s /q
rmdir dist /s /q

REM Train the classifier.
python HistogramClassifier.py
```

当我们希望将您的注意力引到代码块的一个特定部分时，相关的行或项目将以粗体显示：

```py
        <activity
            android:name=".CameraActivity"
            android:screenOrientation="landscape"
            android:theme="@android:style/Theme.NoTitleBar.Fullscreen">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />

                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
```

任何命令行输入或输出都应如下编写：

```py
$ echo "bcm2835-v4l2" | sudo tee -a /etc/modules 
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下所示。以下是一个示例：“点击 Android 平台，然后点击 Switch Platform 按钮。”

警告或重要注意事项如下所示。

小技巧和窍门看起来是这样的。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：请发送电子邮件至 `feedback@packtpub.com` 并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送电子邮件给我们。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一错误。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**联系作者**：您可以直接通过 josephhowse@nummist.com 发送电子邮件给 Joseph Howse。他维护本书的 GitHub 仓库 [`github.com/PacktPublishing/OpenCV-4-for-Secret-Agents-Second-Edition`](https://github.com/PacktPublishing/OpenCV-4-for-Secret-Agents-Second-Edition)，以及他自己的书籍支持网页 [`nummist.com/opencv`](http://nummist.com/opencv)，因此您可能希望在这些网站上查找他的更新。

**盗版**：如果您在互联网上以任何形式发现我们作品的非法副本，我们将不胜感激，如果您能提供位置地址或网站名称。请通过 `copyright@packtpub.com` 联系我们，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用过这本书，为什么不在这本书购买的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
