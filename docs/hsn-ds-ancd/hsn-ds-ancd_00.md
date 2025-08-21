# 前言

Anaconda 是一个开源数据科学平台，汇聚了最优秀的数据科学工具。它是一个数据科学技术栈，包含了基于 Python、Scala 和 R 的 100 多个流行包。在其包管理器`conda`的帮助下，用户可以使用多种语言的数百个包，轻松地进行数据预处理、建模、聚类、分类和验证。

本书将帮助你入门 Anaconda，并学习如何在实际应用中利用它执行数据科学操作。你将从设置 Anaconda 平台、Jupyter 环境以及安装相关包开始。然后，你将学习数据科学和线性代数的基础，以便执行数据科学任务。准备好后，你将开始进行数据科学操作，如数据清洗、排序和数据分类。接下来，你将学习如何执行聚类、回归、预测、构建机器学习模型并进行优化等任务。你还将学习如何进行数据可视化，并分享项目。

在本课程中，你将学习如何使用不同的包，利用 Anaconda 来获得最佳结果。你将学习如何高效使用`conda`——Anaconda 的包、依赖和环境管理器。你还将接触到 Anaconda 的几个强大功能，例如附加项目、项目插件、共享项目驱动以及付费版中可用的强大计算节点，帮助你完成高级数据处理流程。你将学习如何构建可扩展且功能高效的包，并学习如何进行异构数据探索、分布式计算等任务。你将学会如何发现并分享包、笔记本和环境，从而提高生产力。你还将了解 Anaconda Accelerate，这一功能能够帮助你轻松实现服务水平协议（SLA）并优化计算能力。

本书将介绍四种编程语言：R、Python、Octave 和 Julia。选择这四种语言有几个原因。首先，这四种语言都是开源的，这是未来发展的趋势之一。其次，使用 Anaconda 平台的一个显著优势是，它允许我们在同一个平台上实现多种不同语言编写的程序。然而，对于许多新读者来说，同时学习四种语言可能会是一个相当具有挑战性的任务。最好的策略是先集中学习 R 和 Python，过一段时间，或者完成整本书后，再学习 Octave 或 Julia。

+   **R**：这是一款免费的统计计算和图形绘制软件环境。它可以在多种 UNIX 平台上编译和运行，例如 Windows 和 macOS。我们认为 R 是许多优秀编程语言中最容易上手的，特别是那些提供免费软件的语言。作者已出版一本名为 *Financial Modeling using R* 的书；您可以通过 [`canisius.edu/~yany/webs/amazon2018R.shtml`](http://canisius.edu/~yany/webs/amazon2018R.shtml) 查看其亚马逊链接。

+   **Python**：这是一种解释型的高级编程语言，适用于通用编程。在商业分析/数据科学领域，Python 可能是最受欢迎的选择之一。在 2017 年，作者出版了一本名为 *Python for Finance*（第二版）的书；您可以通过 [`canisius.edu/~yany/webs/amazonP4F2.shtml`](http://canisius.edu/~yany/webs/amazonP4F2.shtml) 查看其亚马逊链接。

+   **Octave**：这是一款具有高级编程语言特性的软件下载工具，主要用于数值计算。Octave 帮助解决线性和非线性问题的数值解法，并执行其他数值实验。Octave 也是免费的。它的语法与在华尔街及其他行业中非常流行的 MATLAB 兼容。

+   **Julia**：这是一种用于数值计算的高级高效动态编程语言。它提供了复杂的编译器、分布式并行执行、数值精度和丰富的数学函数库。Julia 的基础库大部分是用 Julia 自身编写的，还集成了成熟的、行业领先的开源 C 和 Fortran 库，用于线性代数、随机数生成、信号处理和字符串处理。

祝您阅读愉快！

# 本书适合谁

*Hands-On Data Science with Anaconda* 适合那些寻找市场上最佳工具来执行数据科学操作的开发人员。如果您是数据分析师或数据科学专家，想要通过使用多个语言中最优秀的库来提高数据科学应用程序的效率，这本书也非常适合您。要求具有基本的 R 或 Python 编程知识以及基本的线性代数知识。

# 如何最大化利用本书

本书的章节需要配备至少 8GB 或 16GB 内存的 PC 或 Mac（内存越大，效果越好）。您的机器至少需要配备 2.2 GHz 的 Core i3/i5 处理器或相当的 AMD 处理器。

# 下载示例代码文件

您可以从您的 [www.packtpub.com](http://www.packtpub.com) 账户中下载本书的示例代码文件。如果您从其他地方购买了本书，您可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便将文件直接发送到您的电子邮件。

您可以通过以下步骤下载代码文件：

1.  登录或注册 [www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”标签。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

下载文件后，请确保使用最新版的工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Data-Science-with-Anaconda`](https://github.com/PacktPublishing/Hands-On-Data-Science-with-Anaconda)。如果代码有任何更新，GitHub 上的现有代码库会进行更新。

我们还提供了其他书籍和视频的代码包，可以在我们的丰富目录中找到，网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。赶紧去看看吧！

# 下载彩色图像

我们还提供了一份 PDF 文件，包含本书中使用的截图/图表的彩色图像。你可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnDataSciencewithAnaconda_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/HandsOnDataSciencewithAnaconda_ColorImages.pdf)。

# 使用的约定

本书中，你会看到多种文本样式，用以区分不同类型的信息。以下是这些样式的一些示例及其含义的解释。

文本中的代码字、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名都以如下方式显示：“最广泛使用的 Python 图形和图像包是`matplotlib`。”

一段代码块的显示方式如下：

```py
import matplotlib.pyplot as plt  
plt.plot([2,3,8,12]) 
plt.show() 
```

当我们希望引起你对代码块中特定部分的注意时，相关的行或项会以粗体显示：

```py
import matplotlib.pyplot as plt  
plt.plot([2,3,8,12]) 
plt.show() 
```

任何命令行输入或输出都以如下方式显示：

```py
install.packages("rattle") 
```

**粗体**：表示新术语、重要词汇或屏幕上出现的词汇。例如，菜单或对话框中的词汇会以这种方式显示在文本中。举个例子：“对于数据源，我们从七种潜在格式中选择，如 File、ARFF、ODBC、R 数据集、RData 文件，然后我们可以从那里加载数据。”

警告或重要提示以这种方式显示。

小贴士和技巧以这种方式展示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：发送邮件至`feedback@packtpub.com`，并在邮件主题中提到书名。如果你有关于本书的任何问题，请通过`questions@packtpub.com`与我们联系。

**勘误**：尽管我们已尽力确保内容的准确性，但错误仍然可能发生。如果你在本书中发现错误，我们将非常感谢你向我们报告。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择你的书籍，点击“勘误提交表单”链接，并填写相关信息。

**盗版**：如果您在互联网上发现任何非法复制的我们的作品，我们将非常感激您提供相关地址或网站名称。请通过 `copyright@packtpub.com` 联系我们，并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域拥有专业知识，并且有兴趣撰写或为一本书做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，为什么不在您购买的站点上留下评论呢？潜在读者可以参考您的公正意见来做出购买决策，Packt 能了解您对我们产品的看法，我们的作者也能看到您对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问 [packtpub.com](https://www.packtpub.com/)。
