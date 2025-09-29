# 前言

ArcGIS 是 Esri 提供的行业标准地理信息系统。

本书将向你展示如何使用 Python 编程语言为 ArcGIS for Desktop 环境创建地理处理脚本、工具和快捷方式。

通过展示如何使用 Python 编程语言与 ArcGIS for Desktop 一起自动化地理处理任务、管理地图文档和图层、查找和修复损坏的数据链接、编辑要素类和表格中的数据等，本书将使你成为一个更有效率和高效的 GIS 专业人士。

《使用 Python 编程 ArcGIS 第二版 Cookbook*》，首先在 ArcGIS for Desktop 环境下介绍基本的 Python 编程概念。采用如何操作的指导风格，接下来你将学习如何使用 Python 来自动化常见的 ArcGIS 地理处理任务。

在这本书中，你还将涵盖一些特定的 ArcGIS 脚本主题，这些主题将帮助你在使用 ArcGIS 时节省时间和精力。这些主题包括管理地图文档文件、自动化地图制作和打印、查找和修复损坏的数据源、创建自定义地理处理工具，以及处理要素类和表格等。

在《使用 Python 编程 ArcGIS 第二版 Cookbook*》中，你将学习如何使用一种以实现特定任务为中心的实用方法，以烹饪书风格的格式编写地理处理脚本。

# 本书涵盖的内容

第一章, *ArcGIS 的 Python 语言基础*，将涵盖 Python 中许多基本语言结构。最初，你将学习如何创建新的 Python 脚本或编辑现有脚本。从那里，你将了解语言特性，如给你的代码添加注释、变量和内置的打字系统，这些使得使用 Python 编程变得简单和紧凑。此外，我们将探讨 Python 提供的各种内置数据类型，如字符串、数字、列表和字典。除此之外，我们还将涵盖语句，包括决策支持和循环结构，用于在代码中做出决策，以及/或多次遍历代码块。

第二章, *管理地图文档和图层*，将使用 ArcPy 映射模块来管理地图文档和图层文件。你将学习如何从地图文档文件中添加和删除地理图层，将图层插入到数据框架中，以及在地图文档中移动图层。你还将学习如何更新图层属性和符号。

第三章, *查找和修复损坏的数据链接*，将教你如何在地图文档文件中生成损坏数据源列表，并应用各种 ArcPy 映射函数来修复这些数据源。你将学习如何自动化修复多个地图文档中数据源的过程。

第四章, *自动化地图制作和打印*，将教会你如何自动化创建生产质量地图的过程。这些地图可以打印出来，导出为图像文件格式，或导出为 PDF 文件，以便包含在地图集中。

第五章, *从脚本中执行地理处理工具*，将教会你如何编写脚本，访问并运行 ArcGIS 提供的地理处理工具。

第六章, *创建自定义地理处理工具*，将教会你如何创建可以添加到 ArcGIS 并与其他用户共享的自定义地理处理工具。自定义地理处理工具附加到一个 Python 脚本上，该脚本以某种方式处理或分析地理数据。

第七章, *查询和选择数据*，将教会你如何从脚本中执行“按属性选择”和“按位置选择”地理处理工具以选择要素和记录。你将学习如何构建为“按属性选择”工具提供可选的 WHERE 子句的查询。还将介绍使用要素层和表格视图作为临时数据集。

第八章, *使用 ArcPy 数据访问模块与要素类和表格*，将教会你如何创建地理处理脚本，从地理数据层和表格中选择、插入或更新数据。使用新的 ArcGIS 10.1 数据访问模块，地理处理脚本可以从要素类和表格创建内存中的数据表，称为游标。你将学习如何创建各种类型的游标，包括搜索、插入和更新。

第九章, *列出和描述 GIS 数据*，将教会你如何通过使用 ArcPy Describe 函数来获取关于地理数据集的描述性信息。作为多步骤过程的第一步，地理处理脚本通常需要生成地理数据列表，然后对这些数据集执行各种地理处理操作。

第十章, *使用插件自定义 ArcGIS 界面*，将教会你如何通过创建 Python 插件来自定义 ArcGIS 界面。插件提供了一种通过模块化代码库执行特定操作的方式，将用户界面元素添加到 ArcGIS 桌面中。界面组件可以包括按钮、工具、工具栏、菜单、组合框、工具面板和应用扩展。插件使用 Python 脚本和一个 XML 文件创建，该文件定义了用户界面应该如何显示。

第十一章，*错误处理和故障排除*，将向您介绍如何在地理处理脚本执行过程中优雅地处理发生的错误和异常。ArcPy 和 Python 错误可以使用 Python 的 try/except 结构捕获并相应处理。

第十二章，*使用 Python 进行高级 ArcGIS*，涵盖了使用 Python 的 ArcGIS REST API 访问由 ArcGIS Server 和 ArcGIS Online 公开的服务。您将学习如何进行 HTTP 请求并解析响应、导出地图、查询地图服务、执行地理编码等。本章还涉及一些与 ArcPy FieldMap 和 FieldMappings 相关以及与 ValueTables 一起工作的杂项主题。

第十三章，*在 ArcGIS Pro 中使用 Python*，涵盖了新 ArcGIS Pro 环境与 ArcGIS for Desktop 在 Python 方面的某些区别，特别是用于编写和执行代码的 Python 窗口。

附录 A，*自动化 Python 脚本*，将教会您如何安排地理处理脚本在指定时间运行。许多地理处理脚本需要很长时间才能完全执行，需要定期在工作时间之外安排运行。您将学习如何创建包含地理处理脚本的批处理文件，并在指定时间执行这些文件。

附录 B，*每个 GIS 程序员都应该知道的五个 Python 食谱*，将教会您如何编写使用 Python 执行各种通用任务的脚本。这些任务，如读取和写入定界文本文件、发送电子邮件、与 FTP 服务器交互、创建 ZIP 文件以及读取和写入 JSON 和 XML 文件，都是常见的。每个 GIS 程序员都应该知道如何编写包含这些功能的 Python 脚本。

# 您需要为本书准备的材料

要完成本书中的练习，您需要安装 ArcGIS for Desktop 10.3，许可证级别为基本、标准或高级。安装 ArcGIS for Desktop 10.3 还将安装 Python 2.7 以及 IDLE Python 代码编辑器。文本也适用于使用 ArcGIS for Desktop 10.2 或 10.1 的用户。第十三章，*在 ArcGIS Pro 中使用 Python*，需要 ArcGIS Pro 版本 1.0。

# 本书面向的对象

《使用 Python 编程 ArcGIS 烹饪书，第二版》是为希望用 Python 革新其 ArcGIS 工作流程的 GIS 专业人员编写的。无论您是 ArcGIS 的新手还是经验丰富的专业人士，您几乎每天都会花时间执行各种地理处理任务。本书将教会您如何使用 Python 编程语言来自动化这些地理处理任务，并使您成为一个更高效、更有效的 GIS 专业人员。

# 部分

在本书中，您将找到几个频繁出现的标题（准备工作、如何操作、工作原理、更多内容、相关内容）。

为了清楚地说明如何完成食谱，我们使用以下这些部分：

## 准备工作

本节告诉您在食谱中可以期待什么，并描述了如何设置任何软件或任何为食谱所需的初步设置。

## 如何操作…

本节包含遵循食谱所需的步骤。

## 工作原理…

本节通常包含对前一个章节发生情况的详细解释。

## 更多内容…

本节包含有关食谱的附加信息，以便使读者对食谱有更深入的了解。

## 相关内容

本节提供了对其他有用信息的链接，以帮助读者了解食谱。

# 惯例

在本书中，您将找到多种文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词汇如下显示：“我们已使用 IDLE 加载了 `ListFeatureClasses.py` 脚本。”

代码块设置如下：

```py
   import arcpy
   fc = "c:/ArcpyBook/data/TravisCounty/TravisCounty.shp"
   # Fetch each feature from the cursor and examine the extent # properties and spatial reference
   for row in arcpy.da.SearchCursor(fc, ["SHAPE@"]):
     # get the extent of the county boundary
     ext = row[0].extent
     # print out the bounding coordinates and spatial reference
     print("XMin: " + ext.XMin)
     print("XMax: " + ext.XMax)
     print("YMin: " + ext.YMin)
     print("YMax: " + ext.YMax)
     print("Spatial Reference: " + ext.spatialReference.name)
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
   import arcpy
   fc = "c:/data/city.gdb/streets"
   # For each row print the Object ID field, and use the SHAPE@AREA
   # token to access geometry properties
   with arcpy.da.SearchCursor(fc, ("OID@", "SHAPE@AREA")) as cursor:
     for row in cursor:
       print("Feature {0} has an area of {1}".format(row[0], row[1]))
```

任何命令行输入或输出都应如下编写：

```py
[<map layer u'City of Austin Bldg Permits'>, <map layer u'Hospitals'>, <map layer u'Schools'>, <map layer u'Streams'>, <map layer u'Streets'>, <map layer u'Streams_Buff'>, <map layer u'Floodplains'>, <map layeru'2000 Census Tracts'>, <map layer u'City Limits'>, <map layer u'Travis County'>]

```

**新术语**和**重要词汇**以粗体显示。屏幕上显示的词汇，例如在菜单或对话框中，在文本中如下显示：“转到**开始** | **所有程序** | **ArcGIS** | **Python 2.7** | **IDLE**”。

### 注意

警告或重要注意事项以如下框的形式出现。

### 小贴士

小贴士和技巧看起来像这样。

# 读者反馈

我们读者的反馈总是受欢迎的。请告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们很重要，因为它帮助我们开发出您真正能从中获得最大收益的标题。

要向我们发送一般反馈，只需发送电子邮件至 `<feedback@packtpub.com>`，并在邮件主题中提及本书的标题。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南，网址为 [www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经成为 Packt 书籍的骄傲拥有者，我们有一些事情可以帮助您从购买中获得最大收益。

## 下载示例代码

您可以从您在[`www.packtpub.com`](http://www.packtpub.com)的账户下载示例代码文件，适用于您购买的所有 Packt Publishing 书籍。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 勘误

尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**勘误**部分。

## 盗版

互联网上版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过发送链接到疑似盗版材料至 `<copyright@packtpub.com>` 来与我们联系。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面的帮助。

## 问题

如果您对本书的任何方面有问题，您可以通过发送邮件至 `<questions@packtpub.com>` 来联系我们，我们将尽力解决问题。
