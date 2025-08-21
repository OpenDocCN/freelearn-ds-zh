# 前言

Python 是一种通用编程语言，越来越多地被用于数据分析和可视化。Matplotlib 是一个流行的 Python 数据可视化包，用于设计有效的图表和图形。本书是一本实用的资源，帮助你使用 Matplotlib 库在 Python 中进行数据可视化。

本书教你如何使用 Matplotlib 创建吸引人的图表、图形和绘图。你还将快速了解第三方包 Seaborn、pandas、Basemap 和 Geopandas，并学习如何将它们与 Matplotlib 配合使用。之后，你将把图表嵌入并定制到 GTK+、Qt 5 和 WXWIDGETS 等第三方工具中。

通过本书提供的实用示例，你还将能够调整可视化的外观和风格。接下来，你将通过基于云平台的第三方软件包（如 Flask 和 Django）在网上探索 Matplotlib 2.1.x。最后，你将通过实际的世界级示例，将交互式、实时的可视化技术整合到当前的工作流程中。

本书结束时，你将完全掌握流行的 Python 数据可视化库 Matplotlib 2.1.x，并能利用其强大功能构建吸引人、有洞察力且强大的可视化图表。

# 适用对象

本书适用于任何希望使用 Matplotlib 库创建直观数据可视化的人。如果你是数据科学家或分析师，且希望使用 Python 创建吸引人的可视化图表，你会发现本书非常有用。你只需具备一些 Python 编程基础，就能开始阅读本书。

# 本书涵盖的内容

第一章，*Matplotlib 简介*，让你熟悉 Matplotlib 的功能和特性。

第二章，*Matplotlib 入门*，带你掌握使用 Matplotlib 语法进行基本绘图的技巧。

第三章，*使用绘图样式和类型装饰图表*，展示了如何美化你的图表，并选择能有效传达数据的合适图表类型。

第四章，*高级 Matplotlib*，教你如何使用非线性刻度、轴刻度、绘图图像和一些流行的第三方软件包将多个相关的图形组合到一个图形中的子图。

第五章，*将 Matplotlib 嵌入 GTK+3 中*，展示了如何在使用 GTK+3 的应用程序中嵌入 Matplotlib 的示例。

第六章，*将 Matplotlib 嵌入 Qt 5 中*，解释了如何将图形嵌入 QWidget，使用布局管理器将图形打包到 QWidget 中，创建计时器，响应事件，并相应地更新 Matplotlib 图表。我们使用 QT Designer 绘制了一个简单的 GUI 来展示 Matplotlib 嵌入。

第七章，*在 wxWidgets 中嵌入 Matplotlib，使用 wxPython*，展示了如何在 wxWidgets 框架中使用 Matplotlib，特别是使用 wxPython 绑定。

第八章，*将 Matplotlib 与 Web 应用程序集成*，教你如何开发一个简单的网站，显示比特币的价格。

第九章，*Matplotlib 在实际应用中的使用*，通过实际案例开始探索更高级的 Matplotlib 用法。

第十章，*将数据可视化集成到工作流程中*，涵盖了一个结合数据分析技巧与可视化技术的迷你项目。

# 为了充分利用本书

需要安装 Python 3.4 或更高版本。可以从[`www.python.org/download/`](https://www.python.org/download/)获取默认的 Python 发行版。软件包的安装在各章节中都有介绍，但你也可以参考官方文档页面获取更多细节。建议使用 Windows 7+、macOS 10.10+或 Linux 系统，且电脑内存至少为 4GB。

# 下载示例代码文件

你可以从你的账户在[www.packtpub.com](http://www.packtpub.com)下载本书的示例代码文件。如果你是从其他地方购买的本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，文件将直接发送到你的邮箱。

你可以通过以下步骤下载代码文件：

1.  登录或注册账户，访问[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择 SUPPORT 标签。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

文件下载后，请确保使用最新版的解压或提取工具解压文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Zipeg/iZip/UnRarX 适用于 Mac

+   7-Zip/PeaZip 适用于 Linux

本书的代码包也托管在 GitHub 上，访问地址为[`github.com/PacktPublishing/Matplotlib-for-Python-Developers-Second-Edition/`](https://github.com/PacktPublishing/Matplotlib-for-Python-Developers-Second-Edition/)。如果代码有更新，它将被更新到现有的 GitHub 仓库。

我们还提供了其他代码包，来自我们丰富的书籍和视频目录，访问地址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快来看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书使用的截图/图表的彩色图片。你可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/MatplotlibforPythonDevelopersSecondEdition_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/MatplotlibforPythonDevelopersSecondEdition_ColorImages.pdf)。

# 使用的约定

本书中使用了一些文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。以下是一个例子：“另一个调节参数是`dash_capstyle`。”

一段代码块的格式如下：

```py
import matplotlib.pyplot as plt
plt.figure(figsize=(4,4))
x = [0.1,0.3]
plt.pie(x)
plt.show()
```

当我们希望您特别注意某个代码块的部分时，相关的行或项目会以粗体显示：

```py
        self.SetSize((500, 550))
        self.button_1 = wx.Button(self, wx.ID_ANY, "button_1")
##Code being added***
        self.Bind(wx.EVT_BUTTON, self.__updat_fun, self.button_1)
        #Setting up the figure, canvas and axes
```

任何命令行输入或输出都以以下方式书写：

```py
python3 first_gtk_example.py
```

**粗体**：表示新术语、重要词汇或屏幕上出现的文字。例如，菜单或对话框中的词汇在文本中呈现如下。这里有一个例子：“在文件和类中选择 Qt，在中间面板选择 Qt Designer 表单。”

警告或重要说明以这种方式呈现。

提示和技巧呈现如下。

# 获取联系

我们始终欢迎读者的反馈。

**一般反馈**：通过电子邮件联系`feedback@packtpub.com`并在邮件主题中提及书名。如果您对本书的任何部分有疑问，请通过`questions@packtpub.com`与我们联系。

**勘误**：尽管我们已经尽力确保内容的准确性，但错误总会发生。如果您发现本书中的错误，我们将非常感激您向我们报告。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接并输入详细信息。

**盗版**：如果您在互联网上发现我们的作品的任何非法复制品，无论形式如何，我们将非常感激您提供位置地址或网站名称。请通过`copyright@packtpub.com`与我们联系，并附上该材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专长并且有意写书或为书籍贡献内容，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，为什么不在购买该书的站点上留下评论呢？潜在读者可以看到并利用您的公正意见做出购买决策，我们 Packt 也能了解您对我们产品的看法，我们的作者能看到您对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。

# Matplotlib 简介

*“一幅画胜过千言万语。” - 弗雷德·R·巴纳德*

欢迎加入创造优秀数据可视化的旅程。在这个大数据爆炸的时代，我们可能都清楚数据分析的重要性。开发者们渴望参与数据挖掘的游戏，并构建工具来收集和建模各种数据。即使是非数据分析师，像性能测试结果和用户反馈等信息，也往往在改善正在开发的软件中至关重要。虽然强大的统计技能无疑为成功的软件开发和数据分析奠定了基础，但即使是最好的数据处理结果，好的故事叙述也是至关重要的。图形数据表现的质量往往决定了你能否在探索性数据分析过程中提取出有用的信息，并在演示中传达出核心信息。

Matplotlib 是一个多功能且强大的 Python 绘图库；它提供了简洁易用的方式来生成各种高质量的数据图形，并且在定制化方面提供了巨大的灵活性。

在本章中，我们将介绍 Matplotlib，内容包括：它的功能、为什么你要使用它以及如何开始使用。我们将覆盖以下主题：

+   什么是 Matplotlib？

+   Matplotlib 的优点

+   Matplotlib 有哪些新特性？

+   Matplotlib 网站和在线文档。

+   输出格式和后端。

+   配置 Matplotlib。
