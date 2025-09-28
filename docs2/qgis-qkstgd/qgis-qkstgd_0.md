# 前言

QGIS 是一个用户友好的开源 GIS 软件。最近，开源 GIS 和 QGIS 正变得越来越受欢迎。

# 本书面向对象

本书是 QGIS 3.4 的快速入门指南。本书适合任何想学习如何使用 QGIS 分析地理空间数据和创建丰富地图应用的人。本书假设没有先前的知识或经验。

# 本书涵盖内容

第一章，*开始使用 QGIS 3*，涵盖了 QGIS 3 的介绍和安装，包括不同版本的简要讨论。然后我们将查看打开 QGIS、软件布局、菜单和工具栏。

第二章，*加载数据*，我们将下载并打开各种类型的 GIS 数据。我们将讨论不同类型的数据，我们将主要使用 GeoPackage 格式。我们还将提及 GeoTIFF（栅格）和 Shapefile（矢量）格式。我们将使用现有数据通过缩放/平移/选择与画布进行交互。最后，我们将讨论保存项目并查看投影。

第三章，*创建数据*，将指导您创建一个 GeoPackage 并围绕它构建一个简单的项目。我们将创建矢量数据。我们将查看编辑工具，以及捕捉和纠正错误。我们将查看属性表以及如何填充它。最后，我们将加载栅格数据并讨论如何创建栅格数据。

第四章，*数据样式化*，我们将对我们的 GIS 数据进行样式化。我们将查看样式选项（有很多）对于矢量和栅格数据。我们将查看 QGIS 中的图层样式面板。

第五章，*创建地图*，将涉及使用前几章中的数据来创建地图。我们将简要查看标签和更详细地探讨如何创建更好、更美观的地图。最后，我们将查看图集功能。

第六章，*空间处理*，将指导您使用 GeoPackage 中的数据来分析数据。本章将介绍处理工具箱。我们将查看单个工具，并查看对数据的空间查询。

第七章，*扩展 QGIS 3*，将探讨扩展 QGIS。这将侧重于插件、模型构建器和少量命令行工作。

# 为了最大限度地利用本书

尽可能按照章节顺序阅读；这将帮助您从本书中获得最大收益。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册以将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac OS

+   7-Zip/PeaZip for Linux

本书代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/QGIS-Quick-Start-Guide`](https://github.com/PacktPublishing/QGIS-Quick-Start-Guide)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。查看它们吧！

# 下载彩色图像

我们还提供包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789341157_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789341157_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“双击`.exe`文件，然后逐步通过以下安装屏幕。”

任何命令行输入或输出都如下所示：

```py
Airport_layer = iface.addVectorLayer('D:/QGIS_quickstart/qgis_sample_data/shapefiles/airports.shp','airports','ogr')
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下所示。以下是一个示例：“点击“完成”以退出。”

警告或重要提示如下所示。

小贴士和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`发送给我们。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，如果您能向我们报告，我们将不胜感激。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/).

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，而我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/).
