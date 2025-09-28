# 前言

欢迎来到 *使用 Python 和 R 扩展 Excel 的世界*！在这本书中，我们深入探讨了 Excel 与 Python 和 R 强大功能的结合，提供了利用这些语言进行数据操作、分析、可视化和更多方面的全面指南。

加入我们的旅程，探索 Excel、R 和 Python 的交汇点，让你在当今数据驱动型环境中脱颖而出。

# 本书面向读者

本书是为有一定数据分析经验且熟悉 Excel 基础的 R 和/或 Python 中级或高级用户设计的。

# 本书涵盖内容

*第一章*，*读取 Excel 工作表*，深入探讨了将数据从 Excel 导入 R/Python 的过程。你将从将第一个 Excel 表格导入 R 开始，了解 Excel 文件的高级细节，并以 Python 的对应部分结束。

*第二章*，*编写 Excel 工作表*，解释了在 R/Python 中分析数据后，如何有效地与 Excel 用户沟通发现。本章提供了从 R/Python 创建 Excel 表格以及导出分析结果的见解。

*第三章*，*从 R 和 Python 执行 VBA 代码*，探讨了除了将结果输出到 Excel 之外，你可能还想向结果 Excel 表格中添加 VBA 宏和函数，以进一步增强分析结果最终用户的功能。我们可以在本章中做到这一点。

*第四章*，*进一步自动化 - 任务调度和电子邮件*，涵盖了如何使用 R 的 RDCOMClient 包（与 Outlook 一起工作）和 Blastula 包（在 R 中帮助自动化分析和发送报告）。在 Python 中，`smtplib` 包具有相同的功能。

*第五章*，*格式化您的 Excel 工作表*，讨论了如何使用包在 Excel 中创建带有格式化数据的表格和表格，以及如何使用它们创建精美的 Excel 报告。

*第六章*，*插入 ggplot2/matplotlib 图表*，展示了如何从 `ggplot2` 和 `matplotlib` 创建图形。用户还可以使用 `ggplot2` 主题以及其他主题在 R/Python 中创建美观的图形，并将它们放置在 Excel 中。

*第七章*，*数据透视表和汇总表*，探讨了使用 R 和 Python 在 Excel 中的数据透视表世界。学习如何直接从 R/Python 创建和操作数据透视表，以便与 Excel 无缝交互。

*第八章*，*使用 R 和 Python 进行数据探索分析*，解释了如何从 Excel 中提取数据并执行 R 的 `{skimr}` 和 Python 的 `pandas` 及 `ppscore`。

*第九章*, *统计分析：线性与逻辑回归*，教您如何在 Excel 数据上使用 R 和 Python 进行简单的统计分析。

*第十章*, *时间序列分析：统计、绘图和预测*，解释了如何使用 R 中的`forecast`包和 Python 中的`kats`以及**长短期记忆**（**LSTM**）进行简单的时间序列分析。

*第十一章*, *从 Excel 直接或通过 API 调用 R/Python*，从 Excel 本地和通过 API 调用 R 和 Python。本章还涵盖了使用 BERT 和`xlwings`从 Excel 调用本地 R/Python 安装的开源工具，以及开源和商业 API 解决方案。

*第十二章*, *使用 R 和 Python 在 Excel 中进行数据分析与可视化——案例研究*，展示了通过调用 R 或 Python 在 Excel 中执行数据可视化和机器学习的案例研究。

# 要充分利用本书

在深入本书之前，对 R 或 Python（或两者）有一个中级理解是有帮助的，包括使用 pandas、NumPy 和 tidyverse 等库进行数据操作和分析的中级熟练度。熟悉 Excel 基础知识，如导航电子表格和执行简单的数据操作，也是假设的。此外，对统计概念和数据可视化技术的初步了解将有助于跟随本书中提供的示例和练习。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| R | Windows（用于 VBA 部分），macOS 或 Linux（用于除 VBA 之外的所有内容） |
| Python 3.11 |  |
| Excel（包括 VBA） |  |

每章将提供相关包和工具的安装指南。

**如果您正在使用本书的数字版，我们建议您自己输入代码或从本书的 GitHub 仓库（下一节中有一个链接）获取代码。这样做将帮助您避免与代码的复制和粘贴相关的任何潜在错误** **。**

免责声明

作者承认使用了尖端的人工智能，如 ChatGPT，其唯一目的是增强本书的语言和清晰度，从而确保读者有一个顺畅的阅读体验。重要的是要注意，内容本身是由作者创作的，并由专业出版团队编辑的。

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件：[`github.com/PacktPublishing/Extending-Excel-with-Python-and-R`](https://github.com/PacktPublishing/Extending-Excel-with-Python-and-R)。如果代码有更新，它将在 GitHub 仓库中更新。

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“`styledtables` 包只能通过 `devtools` 包从 GitHub 安装。”

代码块设置如下：

```py
install.packages("devtools")
# Install development version from GitHub
devtools::install_github(
'R-package/styledTables',
build_vignettes = TRUE
)
```

任何命令行输入或输出都按以下方式编写：

```py
python –m pip install pywin32==306
```

`iris_data.xlsm` 文件中包含宏，可以通过转到 **开发者** | **宏（或 Visual Basic**） 来查看宏是否存在。”

小贴士或重要注意事项

看起来像这样。

# 联系我们

我们读者的反馈始终受到欢迎。

**总体反馈**：如果您对本书的任何方面有任何疑问，请通过电子邮件发送至 customercare@packtpub.com，并在邮件主题中提及书名。

**勘误**：尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，我们将不胜感激，如果您能向我们报告，请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) 并填写表格。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过电子邮件发送至 copyright@packt.com 并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

读完 *使用 Python 和 R 扩展 Excel* 后，我们很乐意听到您的想法！请 [点击此处直接转到该书的 Amazon 评论页面](https://packt.link/r/1804610690) 并分享您的反馈。

您的评论对我们和科技社区都很重要，并将帮助我们确保我们提供高质量的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

您喜欢在路上阅读，但又无法携带您的印刷书籍到处走？

您的电子书购买是否与您选择的设备不兼容？

别担心，现在每购买一本 Packt 书籍，您都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何地点、任何设备上阅读。直接从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

优惠不会就此结束，您还可以获得独家折扣、时事通讯和每日免费内容的每日电子邮件。

按照以下简单步骤获取好处：

1.  扫描二维码或访问以下链接

![下载本书的免费 PDF 副本

二维码

](https://packt.link/free-ebook/9781804610695)

[`packt.link/free-ebook/9781804610695`](https://packt.link/free-ebook/9781804610695)

1.  提交您的购买证明

1.  就这样！我们将直接将您的免费 PDF 和其他福利发送到您的邮箱

# 第一部分：基础知识 - 从 R 和 Python 读取和写入 Excel 文件

这一部分为在 R 和 Python 中处理 Excel 文件奠定了基础。章节涵盖了诸如使用 R 和 Python 等流行库读取和写入 Excel 电子表格等基本任务，使您能够使用`RDCOMClient`、`blastula`、`schedule`和`smtplib`等工具自动化任务，并进一步通过这些工具增强您的 Excel 工作流程，例如安排运行和发送`emails.readxl`、`openxlsx`、`xlsx`、`pandas`和`openpyxl`。此外，您还将学习如何执行 VBA 代码。

本部分包含以下章节：

+   *第一章*，*读取 Excel 电子表格*

+   *第二章*，*编写 Excel 电子表格*

+   *第三章*，*从 R 和 Python 执行 VBA 代码*

+   *第四章*，*进一步自动化 - 任务调度和电子邮件*
