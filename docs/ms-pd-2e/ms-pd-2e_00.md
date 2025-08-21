# 前言

pandas 是一个流行的 Python 库，全球的数据科学家和分析师都在使用它来处理和分析数据。本书介绍了 pandas 中一些实用的数据操作技术，用于在不同领域进行复杂的数据分析。它提供的功能和特性使得数据分析变得比许多其他流行语言（如 Java、C、C++和 Ruby）更加容易和快速。

# 本书适用对象

本书适合那些希望使用 pandas 探索高级数据分析和科学计算技术的数据科学家、分析师和 Python 开发人员。你只需要对 Python 编程有一些基本了解，并且熟悉基本的数据分析概念，就能开始阅读本书。

# 本书内容

第一章，*pandas 和数据分析简介*，将介绍 pandas，并解释它在数据分析流程中的作用。我们还将探讨 pandas 的一些流行应用，以及 Python 和 pandas 如何用于数据分析。

第二章，*pandas 及其支持软件的安装*，将介绍如何安装 Python（如有必要）、pandas 库以及所有必要的依赖项，适用于 Windows、macOS X 和 Linux 平台。我们还将探讨 pandas 的命令行技巧、选项和设置。

第三章，*使用 NumPy 和 pandas 的数据结构*，将快速介绍 NumPy 的强大功能，并展示它如何在使用 pandas 时让工作变得更加轻松。我们还将使用 NumPy 实现一个神经网络，并探索多维数组的一些实际应用。

第四章，*使用 pandas 处理不同数据格式的 I/O*，将教你如何读取和写入常见格式，如**逗号分隔值**（**CSV**），以及所有选项，还会介绍一些更为特殊的文件格式，如 URL、JSON 和 XML。我们还将从数据对象创建这些格式的文件，并在 pandas 中创建一些特殊的图表。

第五章，*在 pandas 中索引和选择数据*，将向你展示如何从 pandas 数据结构中访问和选择数据。我们将详细介绍基础索引、标签索引、整数索引、混合索引以及索引的操作。

第六章，*在 pandas 中分组、合并和重塑数据*，将考察能够重新排列数据的各种功能，并通过实际数据集来使用这些功能。我们还将学习数据分组、合并和重塑的技巧。

第七章，*pandas 中的特殊数据操作*，将讨论并详细介绍 pandas 中一些特殊数据操作的方法、语法和用法。

第八章，*使用 Matplotlib 处理时间序列和绘图*，将介绍如何处理时间序列和日期。我们还将探讨一些必要的主题，这些是您在使用 pandas 时需要了解的内容。

第九章，*在 Jupyter 中使用 pandas 制作强大的报告*，将探讨一系列样式和 pandas 格式化选项的应用。我们还将学习如何在 Jupyter Notebook 中创建仪表盘和报告。

第十章，*使用 pandas 和 NumPy 进行统计学入门*，将深入探讨如何利用 pandas 执行统计计算，涉及包和计算方法。

第十一章，*贝叶斯统计与最大似然估计简要介绍*，将探讨一种替代的统计方法——贝叶斯方法。我们还将研究关键的统计分布，并展示如何使用各种统计包在 `matplotlib` 中生成和绘制分布图。

第十二章，*使用 pandas 进行数据案例研究*，将讨论如何使用 pandas 解决实际的数据案例研究。我们还将学习如何使用 Python 进行网页抓取以及数据验证。

第十三章，*pandas 库架构*，将讨论 pandas 库的架构和代码结构。本章还将简要演示如何使用 Python 扩展提高性能。

第十四章，*pandas 与其他工具的比较*，将重点比较 pandas 与 R 以及其他工具，如 SQL 和 SAS。我们还将探讨切片和选择的相关内容。

第十五章，*机器学习简要介绍*，将通过简要介绍 `scikit-learn` 库进行机器学习，并展示 pandas 如何融入该框架中，从而结束本书内容。

# 为了从本书中获得最大的收益

执行代码时将使用以下软件：

+   Windows/macOS/Linux

+   Python 3.6

+   pandas

+   IPython

+   R

+   scikit-learn

对于硬件，没有特别的要求。Python 和 pandas 可以在 Mac、Linux 或 Windows 机器上运行。

# 下载示例代码文件

您可以从您的账户在 [www.packt.com](http://www.packt.com) 下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，文件将直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  登录或注册并访问 [www.packt.com](http://www.packt.com)。

1.  选择 **SUPPORT** 选项卡。

1.  点击 **Code Downloads & Errata**。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

下载文件后，请确保使用以下最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Mastering-Pandas-Second-Edition`](https://github.com/PacktPublishing/Mastering-Pandas-Second-Edition)。如果代码有更新，将会在现有的 GitHub 仓库中进行更新。

我们还从我们丰富的书籍和视频目录中提供其他代码包，您可以在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 查看它们！快去看看吧！

# 下载彩色图像

我们还提供了包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以在此下载： [`static.packt-cdn.com/downloads/9781789343236_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781789343236_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。示例：“Python 具有内置的`array`模块用于创建数组。”

一段代码设置如下：

```py
source_python("titanic.py")
titanic_in_r <- get_data_head("titanic.csv")
```

所有命令行输入或输出均按如下方式编写：

```py
 python --version
```

**粗体**：表示新术语、重要单词或屏幕上显示的单词。例如，菜单或对话框中的单词在文本中会以这种方式出现。示例：“其他目录中的任何笔记本可以通过‘上传’选项传输到 Jupyter Notebook 的当前工作目录中。”

警告或重要备注以此方式显示。

小贴士和技巧以此方式显示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提到书名，并通过电子邮件联系我们，地址为 `customercare@packtpub.com`。

**勘误**：虽然我们已尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，我们将不胜感激，如果您能将错误报告给我们。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并填写相关详情。

**盗版**：如果您在互联网上发现我们作品的任何非法复制形式，我们将不胜感激，如果您能提供其位置地址或网站名称。请通过电子邮件联系我们，地址为 `copyright@packt.com`，并附上链接。

**如果您有兴趣成为作者**：如果您在某个领域有专长，并且有意撰写或贡献书籍，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了本书，为什么不在您购买书籍的网站上留下评论呢？潜在读者可以看到并利用您的公正意见来做出购买决策，我们在 Packt 也能了解您对我们产品的看法，而我们的作者也能看到您对他们书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
