# 前言

本书的功能目标是展示如何使用 Python 包进行数据分析；如何从**电子健康记录**（**EHR**）调查中导入、收集、清理和优化数据；以及如何利用这些数据制作预测模型，并辅以实际示例。

# 适合读者

*简明医疗分析*适合你，如果你是一个掌握 Python 或相关编程语言的开发者，即使你对医疗或医疗数据的预测建模不熟悉，仍然可以受益。对分析和医疗计算感兴趣的临床医生也会从本书中获益。本书还可以作为医疗机器学习入门课程的教科书。

# 本书内容概述

第一章，*医疗分析入门*，提供医疗分析的定义，列出一些基础话题，提供该主题的历史，给出医疗分析实际应用的例子，并包括本书中软件的下载、安装和基本使用说明。

第二章，*医疗基础*，包括美国医疗体系的结构和服务概述，提供与医疗分析相关的立法背景，描述临床患者数据和临床编码系统，并提供医疗分析的细分。

第三章，*机器学习基础*，描述了用于医学决策的部分模型框架，并描述了机器学习流程，从数据导入到模型评估。

第四章，*计算基础 – 数据库*，提供 SQL 语言的介绍，并通过医疗预测分析示例展示 SQL 在医疗中的应用。

第五章，*计算基础 – Python 入门*，提供 Python 及其在分析中重要库的基本概述。我们讨论 Python 中的变量类型、数据结构、函数和模块。还介绍了`pandas`和`scikit-learn`库。

第六章，*衡量医疗质量*，描述了医疗绩效评估中使用的指标，概述了美国的基于价值的计划，并展示如何在 Python 中下载和分析基于提供者的数据。

第七章，*医疗中的预测模型制作*，描述了公开可用的临床数据集所包含的信息，包括下载说明。然后，我们展示如何使用 Python、`pandas`和 scikit-learn 来制作预测模型。

第八章，*医疗保健预测模型——回顾*，回顾了通过比较机器学习结果和传统方法所得结果，当前在选择性疾病和应用领域中，医疗保健预测分析的进展。

第九章，*未来—医疗保健与新兴技术*，讨论了通过使用互联网推动医疗保健分析的一些进展，向读者介绍了医疗保健中的深度学习技术，并陈述了医疗保健分析面临的一些挑战和局限性。

# 充分利用本书的内容

需要了解的一些有用信息包括：

+   高中数学，如基本的概率论、统计学和代数

+   对编程语言和/或基本编程概念的基本了解

+   对医疗保健的基本了解以及一些临床术语的工作知识

请按照第一章的*医疗保健分析简介*中的说明设置 Anaconda 和 SQLite。

# 下载示例代码文件

您可以从您的账户中下载本书的示例代码文件，网址：[www.packtpub.com](http://www.packtpub.com)。如果您从其他地方购买了本书，您可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，文件将直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  登录或注册：[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

下载文件后，请确保使用最新版本的工具解压或提取文件夹：

+   Windows 版的 WinRAR/7-Zip

+   Mac 版的 Zipeg/iZip/UnRarX

+   Linux 版的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，地址是[`github.com/PacktPublishing/Healthcare-Analytics-Made-Simple`](https://github.com/PacktPublishing/Healthcare-Analytics-Made-Simple)。如果代码有更新，更新将发布在现有的 GitHub 仓库中。

我们还有来自我们丰富书籍和视频目录的其他代码包，您可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**查看。不要错过！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以在此下载：[`www.packtpub.com/sites/default/files/downloads/HealthcareAnalyticsMadeSimple_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/HealthcareAnalyticsMadeSimple_ColorImages.pdf)。

# 使用的约定

本书中使用了若干文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入以及 Twitter 账号。例如：“将下载的 `WebStorm-10*.dmg` 磁盘映像文件挂载为系统中的另一个磁盘。”

代码块如下所示：

```py
string_1 = '1'
string_2 = '2'
string_sum = string_1 + string_2
print(string_sum)
```

当我们希望特别提醒你注意代码块中的某一部分时，相关行或项目会以粗体显示：

```py
test_split_string = 'Jones,Bill,49,Atlanta,GA,12345'
output = test_split_string.split(',')
print(output)
```

**粗体**：表示新术语、重要词汇或你在屏幕上看到的词汇。例如，菜单或对话框中的词汇会以这种方式出现在文本中。这里有一个例子：“从管理面板中选择系统信息。”

警告或重要提示会以这种形式出现。

提示和技巧会以这种形式出现。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中注明书名。如果你对本书的任何内容有疑问，请通过 `questions@packtpub.com` 与我们联系。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误还是会发生。如果你在本书中发现错误，请报告给我们。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择你的书籍，点击勘误提交表单链接，并填写相关信息。

**盗版**：如果你在互联网上发现我们作品的任何非法复制形式，我们将感激你提供该位置地址或网站名称。请通过 `copyright@packtpub.com` 与我们联系，并提供相关资料链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且有意写书或为书籍贡献内容，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦你阅读并使用了本书，为什么不在你购买书籍的网站上留下评论呢？潜在的读者可以通过看到并参考你的公正意见来做出购买决策，我们在 Packt 也能了解你对我们产品的看法，而我们的作者也可以看到你对他们书籍的反馈。谢谢！

如果你想了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
