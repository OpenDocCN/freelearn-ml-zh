# 第十一章：使用 Python 蛇（或，Python 的第一步）

本附录假设你已经根据 第一章，“准备任务”中的说明，设置了 Python 环境和 OpenCV 的 Python 绑定。现在，如果你是 Python 新手，你可能想知道如何测试这个环境和运行 Python 代码。

Python 提供了一个交互式解释器，因此你可以测试代码而无需将源代码保存到文件中。打开操作系统的终端或命令提示符，并输入以下命令：

```py
$ python
```

Python 将打印其版本信息，然后显示其交互式解释器的提示符 `>>>`。你可以在该提示符下输入代码，Python 将打印代码的返回值（如果有）。例如，如果我们输入 `1+1`，我们应该看到以下文本：

```py
>>> 1+1
2
```

现在，让我们尝试导入 OpenCV Python 模块，该模块称为 `cv2`：

```py
>>> import cv2
```

如果你的 OpenCV 安装状态良好，这一行代码应该会静默运行。另一方面，如果你看到错误，你应该返回并检查设置步骤。如果错误提示为 `ImportError: No module named 'cv2'`，这表明 Python 没能在 Python 的 `site-packages` 文件夹中找到 `cv2.pyd` 文件（OpenCV Python 模块）。在 Windows 上，如果错误提示为 `ImportError: DLL load failed`，这表明 Python 成功找到了 `cv2.pyd` 文件，但未能找到模块的 DLL 依赖之一，例如 OpenCV 的 DLL 或（在自定义构建中使用 TBB）的 TBB DLL；可能包含 DLL 的文件夹未包含在系统的 `Path` 中。

假设我们成功导入了 `cv2`，现在我们可以获取其版本号，如下面的代码片段所示：

```py
>>> cv2.__version__
'4.0.1'
```

确保输出与您认为已安装的 OpenCV 版本匹配。如果不匹配，请返回并检查设置步骤。

当你准备好退出 Python 交互式解释器时，输入以下命令：

```py
>>> quit()
```

在本书的项目中，当你遇到一个具有代码中 `__main__` 部分的 Python 脚本（`.py` 文件）时，如果你将其作为参数传递给 Python 解释器，Python 可以执行此脚本。例如，假设我们想从 第二章，“全球寻找豪华住宿”中运行 `Luxocator.py` 脚本。在操作系统的终端或命令提示符中，我们会运行以下命令：

```py
$ python Luxocator.py
```

然后，Python 将执行 `Luxocator.py` 的 `__main__` 部分。该脚本的这一部分将依次调用其他部分。

你不需要任何特殊工具来创建和编辑 `.py` 文件。一个文本编辑器就足够了，极简主义者可能会更喜欢它。或者，各种专门的 Python 编辑器和 IDE 提供了自动完成等特性。我有时使用文本编辑器，有时使用名为 **PyCharm** ([`www.jetbrains.com/pycharm/`](https://www.jetbrains.com/pycharm/)) 的 IDE，它有一个免费的社区版。

在这里，我们仅介绍了最基本的内容，以便您能够运行和编辑 Python 代码。本书本身并不包含 Python 语言的指南，尽管，当然，本书的项目可以通过实例帮助您学习 Python（以及其他语言）。如果您想通过更专注于语言的学习资源来补充本书的阅读，您可以在官方的*Python For Beginners*指南中找到许多选项，该指南位于[`www.python.org/about/gettingstarted/`](https://www.python.org/about/gettingstarted/)，以及 Packt Publishing 的 Python 技术页面[`www.packtpub.com/tech/Python`](https://www.packtpub.com/tech/Python)。
