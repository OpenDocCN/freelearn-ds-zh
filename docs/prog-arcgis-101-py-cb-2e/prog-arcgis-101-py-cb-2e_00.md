# 前言

ArcGIS 是 ESRI 提供的行业标准地理信息系统。

本书将向您展示如何使用 Python 编程语言为 ArcGIS Desktop 环境创建地理处理脚本、工具和快捷方式。

通过向您展示如何使用 Python 编程语言与 ArcGIS Desktop 自动化地理处理任务、管理地图文档和图层、查找和修复损坏的数据链接、编辑要素类和表中的数据等，本书将使您成为一个更有效率和高效的 GIS 专业人士。

*使用 Python 编程 ArcGIS 10.1 烹饪书* 从在 ArcGIS Desktop 环境中介绍基本的 Python 编程概念开始。使用如何操作的指导风格，然后你将学习如何使用 Python 自动化常见的 ArcGIS 地理处理任务。

在这本书中，你还将涵盖特定的 ArcGIS 脚本主题，这些主题将帮助你在使用 ArcGIS 时节省时间和精力。主题包括管理地图文档文件、自动化地图生产和打印、查找和修复损坏的数据源、创建自定义地理处理工具，以及处理要素类和表格等。

在 *使用 Python 编程 ArcGIS 10.1 烹饪书* 中，你将学习如何使用一种实用方法编写地理处理脚本，该方法围绕在烹饪书风格格式中完成特定任务而设计。

# 本书涵盖的内容

第一章，*ArcGIS 中 Python 语言的 fundamentals*，将涵盖 Python 中许多基本语言结构。最初，你将学习如何创建新的 Python 脚本或编辑现有脚本。从那里，你将深入了解语言特性，例如在代码中添加注释、变量以及使 Python 编程变得简单紧凑的内置类型系统。此外，我们还将探讨 Python 提供的各种内置数据类型，例如字符串、数字、列表和字典。除此之外，我们还将涵盖语句，包括决策支持和循环结构，用于在代码中做出决策和/或多次遍历代码块。

第二章，*使用 ArcPy 编写基本地理处理脚本*，将教授 ArcPy Python 站点包的基本概念，包括基本模块、函数和类的概述。读者将能够使用 ArcPy 和 Python 编写地理处理脚本。

第三章，*管理地图文档和图层*，将使用 Arcpy Mapping 模块来管理地图文档和图层文件。你将学习如何从地图文档文件中添加和删除地理图层，将图层插入到数据框架中，以及在地图文档中移动图层。读者还将学习如何更新图层属性和符号。

第四章, *查找和修复损坏的数据链接*，将教授如何在地图文档文件中生成损坏数据源列表，并应用各种 Arcpy Mapping 函数来修复这些数据源。读者将学习如何自动化修复多个地图文档中的数据源的过程。

第五章, *自动化地图制作和打印*，将教授如何自动化创建生产质量地图的过程。这些地图可以打印出来，导出为图像文件格式，或导出为 PDF 文件，以便包含在地图集中。

第六章, *从脚本中执行地理处理工具*，将教授如何编写脚本以访问和运行 ArcGIS 提供的地理处理工具。

第七章, *创建自定义地理处理工具*，将教授如何创建可以添加到 ArcGIS 并与其他用户共享的自定义地理处理工具。自定义地理处理工具通过某种方式处理或分析地理数据的 Python 脚本附加。

第八章, *查询和选择数据*，将教授如何从脚本中执行**按属性选择**和**按位置选择**地理处理工具以选择要素和记录。读者将学习如何构建为**按属性选择**工具提供可选的 WHERE 子句的查询。还将介绍使用要素层和表视图作为临时数据集的使用。

第九章, *使用 ArcPy 数据访问模块选择、插入和更新地理数据和表*，将教授如何创建地理处理脚本，用于从地理数据层和表中选择、插入或更新数据。利用新的 ArcGIS 10.1 数据访问模块，地理处理脚本可以从要素类和表中创建内存中的数据表，称为游标。读者将学习如何创建各种类型的游标，包括搜索、插入和更新。

第十章, *列出和描述 GIS 数据*，将教授如何通过使用 Arcpy Describe 函数来获取关于地理数据集的描述性信息。作为多步骤过程中的第一步，地理处理脚本通常需要生成地理数据列表，然后对这些数据集执行各种地理处理操作。

第十一章, *使用插件自定义 ArcGIS 界面*，将教授如何通过创建 Python 插件来自定义 ArcGIS 界面。插件提供了一种通过模块化代码库添加用户界面元素到 ArcGIS Desktop 的方法，该代码库旨在执行特定操作。界面组件可以包括按钮、工具、工具栏、菜单、组合框、工具调色板和应用扩展。插件使用 Python 脚本和一个 XML 文件创建，该文件定义了用户界面应该如何显示。

第十二章, *错误处理和故障排除*，将教授如何在地理处理脚本执行过程中优雅地处理发生的错误和异常。可以使用 Python 的`try/except`结构捕获 Arcpy 和 Python 错误，并相应地处理。

附录 A, *自动化 Python 脚本*，将教授如何安排地理处理脚本在指定时间运行。许多地理处理脚本需要很长时间才能完全执行，需要定期在工作时间之外安排运行。读者将学习如何创建包含地理处理脚本的批处理文件，并在指定时间执行这些文件。

附录 B, *每个 GIS 程序员都应该知道如何用 Python 做的五件事*，将教授如何编写执行各种通用任务的脚本，这些任务包括读取和写入定界文本文件、发送电子邮件、与 FTP 服务器交互、创建 ZIP 文件以及读取和写入 JSON 和 XML 文件。每个 GIS 程序员都应该知道如何编写包含此功能的 Python 脚本。

# 阅读这本书你需要什么

要完成本书中的练习，你需要安装 ArcGIS Desktop 10.1，无论是基本、标准还是高级许可级别。安装 ArcGIS Desktop 10.1 还将安装 Python 2.7 以及 IDLE Python 代码编辑器。

# 这本书面向谁

*使用 Python 编程 ArcGIS 10.1 食谱*是为希望用 Python 革命性地改变其 ArcGIS 工作流程的 GIS 专业人士编写的。无论你是 ArcGIS 的新手还是经验丰富的专业人士，你几乎每天都会花时间执行各种地理处理任务。这本书将教你如何使用 Python 编程语言来自动化这些地理处理任务，并使你成为一个更高效、更有效的 GIS 专业人士。

# 习惯用法

在这本书中，你会发现许多不同风格的文本，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词如下所示："我们使用 IDLE 加载了`ListFeatureClasses.py`脚本。"

代码块设置如下：

```py
import arcpy
fc = "c:/ArcpyBook/data/TravisCounty/TravisCounty.shp"
# Fetch each feature from the cursor and examine the extent properties and spatial reference
for row in arcpy.da.SearchCursor(fc, ["SHAPE@"]):
  # get the extent of the county boundary
  ext = row[0].extent
  # print out the bounding coordinates and spatial reference
  print "XMin: " + ext.XMin
  print "XMax: " + ext.XMax
  print "YMin: " + ext.YMin
  print "YMax: " + ext.YMax
  print "Spatial Reference: " + ext.spatialReference.name
```

当我们希望将您的注意力引到代码块的一个特定部分时，相关的行或项目将以粗体显示：

```py
import arcpy

fc = "c:/data/city.gdb/streets"

# For each row print the Object ID field, and use the SHAPE@AREA
# token to access geometry properties

with arcpy.da.SearchCursor(fc, ("OID@", "SHAPE@AREA")) as cursor:
  for row in cursor:
    print("Feature {0} has an area of {1}".format(row[0], row[1]))
```

任何命令行输入或输出都按如下方式编写：

```py
[<map layer u'City of Austin Bldg Permits'>, <map layer u'Hospitals'>, <map layer u'Schools'>, <map layer u'Streams'>, <map layer u'Streets'>, <map layer u'Streams_Buff'>, <map layer u'Floodplains'>, <map layer u'2000 Census Tracts'>, <map layer u'City Limits'>, <map layer u'Travis County'>]

```

**新术语**和**重要词汇**以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中显示如下：“转到**开始** | **程序** | **ArcGIS** | **Python 2.7** | **IDLE**”。

### 注意

警告或重要提示以如下方式显示在框中。

### 提示

技巧和窍门显示如下。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们您对这本书的看法——您喜欢什么或可能不喜欢什么。读者反馈对我们开发您真正从中获得最大收益的标题非常重要。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及书名。

如果您有一本书需要我们出版，并希望看到我们出版，请通过[www.packtpub.com](http://www.packtpub.com)上的**建议书名**表单或发送电子邮件至`<suggest@packtpub.com>`给我们留言。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们关于[www.packtpub.com/authors](http://www.packtpub.com/authors)的作者指南。

# 客户支持

现在您是 Packt 书籍的骄傲拥有者，我们有一些事情可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从[`www.PacktPub.com`](http://www.PacktPub.com)上下载您购买的所有 Packt 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问[`www.PacktPub.com/support`](http://www.PacktPub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 勘误

尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/support`](http://www.packtpub.com/support)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站，或添加到该标题的勘误部分下的现有勘误列表中。您可以通过从[`www.packtpub.com/support`](http://www.packtpub.com/support)选择您的标题来查看任何现有勘误。

## 侵权

在互联网上，版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，无论形式如何，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<copyright@packtpub.com>` 联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面提供的帮助。

## 问题

如果您在本书的任何方面遇到问题，可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决。
