# 前言

## 关于

本节简要介绍了作者、本书涵盖的内容、开始学习所需的技能，以及完成所有包含的活动和练习所需的硬件和软件要求。

## 关于本书

在本书中，你将了解 AWS 上可用的各种人工智能和机器学习服务。通过实践动手练习，你将学习如何使用这些服务生成令人印象深刻的结果。到本书结束时，你将基本了解如何在自己的项目中使用广泛的 AWS 服务。

### 关于作者

Jeffrey Jackovich 是本书的作者，他是一位好奇的数据科学家，拥有健康技术和并购（M&A）背景。他拥有丰富的以业务为导向的医疗保健知识，但喜欢使用 R 和 Python 分析所有类型的数据。他热爱数据科学过程中的挑战，并在摩洛哥担任和平队志愿者时磨练了他的独创性格。他正在波士顿大学完成计算机信息系统硕士学位，主修数据分析。

Ruze Richards 是本书的作者，同时也是一位数据科学家和云架构师，他大部分职业生涯都在为企业和小型初创公司构建高性能分析系统。他对人工智能和机器学习特别热情，最初作为一名物理学家，对神经网络产生了兴趣，随后在 AT&T 贝尔实验室工作以进一步追求这一兴趣领域。随着新的一波热情以及云计算上实际可用的计算能力，他非常高兴能够传播知识并帮助人们实现目标。

### 目标

+   在 AWS 平台上开始使用机器学习

+   使用人工智能和亚马逊 Comprehend 分析非结构化文本

+   创建一个聊天机器人，并使用语音和文本输入与之交互

+   通过你的聊天机器人检索外部数据

+   开发自然语言界面

+   使用亚马逊 Rekognition 将人工智能应用于图像和视频

### 读者对象

这本书非常适合想要了解亚马逊云服务（Amazon Web Services）的人工智能和机器学习能力的数据科学家、程序员和机器学习爱好者。

### 方法

本书以实践为导向，教你使用 AWS 进行机器学习。它包含多个活动，使用真实业务场景让你练习并应用你的新技能，在一个高度相关的环境中。

### 最小硬件要求

为了获得最佳的学生体验，我们推荐以下硬件配置：

+   处理器：英特尔酷睿 i5 或同等性能

+   内存：4GB RAM

+   存储：35GB 可用空间

### 软件要求

你还需要提前安装以下软件：

1.  操作系统：Windows 7 SP1 64 位、Windows 8.1 64 位或 Windows 10 64 位

1.  浏览器：最新版本的谷歌 Chrome

1.  AWS 免费层账户

**约定**

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称如下所示：“命令形式为，"`s3://myBucketName/myKey.`"”

代码块设置如下：

```py
aws comprehend detect-dominant-language ^
--region us-east-1 ^
--text "Machine Learning is fascinating."
```

新术语和重要词汇以粗体显示。屏幕上显示的单词，例如在菜单或对话框中，在文本中显示如下：“存储在 S3 中的数据使用**应用程序编程**接口（**API**）管理，该接口可通过互联网（**HTTPS**）访问。”

### 安装和设置

在开始本书之前，您需要一个 AWS 账户。您还需要设置 AWS 命令行界面（AWXSCLI），具体步骤如下。您在整个书中还需要 Python 3.6、pip 和 AWS Rekognition 账户。

**AWS 账户**

对于 AWS 免费层账户，您需要一个个人电子邮件地址、信用卡或借记卡，以及可以接收短信的手机，以便您验证账户。要创建新账户，请点击此链接 [`aws.amazon.com/free/`](https://aws.amazon.com/free/)。

**AWSCLI 设置**

从以下链接安装 AWS CLI 设置 [`s3.amazonaws.com/aws-cli/AWSCLISetup.exe`](https://github.com/TrainingByPackt/Applied-Data-Science-with-Python-and-Jupyter)。下载 AWS CLI 设置文件（包括 32 位和 64 位 MSI 安装程序，并将自动安装正确的版本）。要验证安装是否成功，请打开命令提示符并输入`aws --version`。

**安装 Python**

按照以下链接中的说明安装 Python 3.6：[`realpython.com/installing-python/`](https://realpython.com/installing-python/)。

**安装 pip**

1.  要安装 pip，转到命令提示符并输入`pip install awscli` `--upgrade --user`。使用命令"`aws --version`"验证安装是否成功。

1.  安装`pip`后，将 AWS 可执行文件添加到您的操作系统 PATH 环境变量中。使用 MSI 安装时，这应该会自动发生，但如果`"aws --version"`命令不起作用，您可能需要手动设置它。

1.  要修改你的`PATH`变量（Windows），输入`环境变量`，然后选择`编辑您的账户的系统环境变量`，选择路径，并将路径添加到变量值字段中，用分号分隔。

**安装虚拟环境**

从以下链接安装适合您操作系统的 Anaconda 版本 [`www.anaconda.com/download/`](https://www.anaconda.com/download/)。Anaconda 可以帮助您安装所需软件，而不会与冲突的包发生冲突。

1.  要检查 Anaconda 发行版是否更新，输入`conda update conda`。

1.  要创建虚拟环境，输入`conda create -n yourenvname python=3.6 anaconda`并按`y`继续，这将安装 Python 版本以及所有相关的 Anaconda 打包库到`path_to_you_anaconda_location/anaconda/envs/yourenvname`。

1.  要在 macOS 和 Linux 上激活账户，请输入`source activate yourenvname`，在 Windows 上请输入`activate yourenvname`。

1.  要将额外的 Python 包安装到虚拟环境中，请输入`conda install –n yourenvname [package]`。

1.  要取消激活虚拟环境，请输入`deactivate`。

**配置和凭证文件**

要定位配置文件，请查看以下特定于操作系统的命令。更多信息请参阅：[`docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html`](https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html)。

**Amazon Rekognition 账户**

您需要创建一个新的 Amazon Rekognition 免费层账户，客户在前 12 个月内每月可免费分析高达 5,000 张图片。要创建免费账户，请点击[`aws.amazon.com/rekognition/`](https://aws.amazon.com/rekognition/)链接。

**安装代码包**

**其他资源**

本书代码包也托管在 GitHub 上：[`github.com/TrainingByPackt/Machine-Learning-with-AWS`](https://github.com/TrainingByPackt/Machine-Learning-with-AWS)。

我们还有其他代码包，这些代码包来自我们丰富的书籍和视频目录，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！
