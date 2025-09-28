# 前言

我孩子的学校附近最近的医院有多远？过去三个月里我城市里的财产犯罪发生在哪里？从我家到办公室的最短路线是什么？我应该为公司的送货卡车指定什么路线以最大化设备利用率并最小化燃料消耗？应该在何处建造下一个消防站以最小化响应时间？

人们每天都在这个星球上提出这些问题，以及其他类似的问题。回答这些问题需要一个能够在两个或更多维度中思考的机制。从历史上看，桌面 GIS 应用是唯一能够回答这些问题的应用。这种方法——尽管完全功能——但对于普通人来说并不可行；大多数人不需要这些应用所能提供的一切功能，或者他们不知道如何使用它们。此外，越来越多的基于位置的服务提供了人们使用的特定功能，并且即使是从他们的智能手机上也可以访问。显然，这些服务的普及化需要强大的后端平台来处理大量的地理操作。

由于需要或希望具有可扩展性、对大数据集的支持以及直接输入机制，大多数开发人员已经选择采用空间数据库作为他们的支持平台。目前有几种空间数据库软件可供选择，有些是专有的，有些是开源的。PostGIS 是一种开源空间数据库软件，可能是所有空间数据库软件中最易于访问的。

PostGIS 作为扩展运行，为 PostgreSQL 数据库提供空间功能。在这个能力下，PostGIS 允许将空间数据与数据库中通常发现的数据一起包含。通过将所有数据放在一起，可以提出像“在考虑每个响应时间的距离后，所有警察局的排名是什么？”这样的问题。通过构建在 PostGIS 提供的核心功能和 PostgreSQL 固有的可扩展性之上，可以实现新的或增强的功能。此外，这本书还邀请在新的 GIS 应用和基于位置的服务中包含位置隐私保护机制，以便用户感到受到尊重，而不是在分享他们的信息时必然处于风险之中，尤其是像他们的行踪这样敏感的信息。

*《PostGIS 食谱，第二版》*采用问题解决方法，帮助你获得对 PostGIS 的扎实理解。希望这本书能回答一些常见空间问题，并给你提供灵感和信心，在寻找解决具有挑战性的空间问题的解决方案时使用和增强 PostGIS。

# 本书面向对象

这本书是为那些寻找使用 PostGIS 解决空间问题最佳方法的人编写的。这些问题可能像找到特定位置最近的餐厅那样简单，也可能像找到从点 A 到点 B 的最短和/或最有效路线那样复杂。

对于刚开始使用 PostGIS 或甚至空间数据集的读者，本书的结构旨在帮助他们熟悉并精通在数据库中运行空间操作。对于经验丰富的用户，本书提供了深入了解高级主题的机会，如点云、栅格地图代数和 PostGIS 编程。

# 本书涵盖的内容

第一章，*在 PostGIS 中移动数据进和出*，涵盖了可用于将空间和非空间数据导入和导出到 PostGIS 的过程。这些过程包括使用 PostGIS 和第三方提供的实用程序，如 GDAL/OGR。

第二章，*有效结构*，讨论了如何使用通过 PostgreSQL 提供的机制来组织 PostGIS 数据。这些机制用于规范化可能不干净和无结构的导入数据。

第三章，*处理矢量数据的基础*，介绍了在 PostGIS 中对矢量数据（在 PostGIS 中称为几何和地理）通常执行的操作。包括处理无效几何、确定几何之间的关系以及简化复杂几何的操作。

第四章，*处理矢量数据的高级食谱*，深入分析几何的高级主题。您将学习如何利用 KNN 过滤器提高邻近查询的性能、从 LiDAR 数据创建多边形以及计算可用于邻域分析的多边形。

第五章，*处理栅格数据*，展示了在 PostGIS 中操作栅格的实用工作流程。您将学习如何导入栅格、修改栅格、对栅格进行分析以及以标准栅格格式导出栅格。

第六章，*使用 pgRouting*，介绍了 pgRouting 扩展，它将图遍历和分析功能引入 PostGIS。本章中的食谱回答了从点 A 条件导航到点 B 以及准确模拟复杂路线（如水道）等现实世界问题。

第七章，*进入第 N 维*，专注于在 PostGIS 中处理和分析多维空间数据（包括来自 LiDAR 的点云）所使用的工具和技术。包括将点云加载到 PostGIS 中、从点云创建 2.5D 和 3D 几何以及应用几个摄影测量原理。

第八章, 《PostGIS 编程》展示了如何使用 Python 语言编写操作和与 PostGIS 交互的应用程序。这些应用程序包括将外部数据集读写到 PostGIS 的方法，以及使用 OpenStreetMap 数据集的基本地理编码引擎。

第九章，《PostGIS 与 Web》介绍了使用 OGC 和 REST 网络服务将 PostGIS 数据和功能提供给 Web 的方法。本章讨论了使用 MapServer 和 GeoServer 提供 OGC、WFS 和 WMS 服务，以及从 OpenLayers 和 Leaflet 等客户端消费这些服务。然后展示了如何使用 GeoDjango 构建 Web 应用程序，以及如何将你的 PostGIS 数据包含在 Mapbox 应用程序中。

第十章, 《维护、优化和性能调整》从 PostGIS 退后一步，专注于 PostgreSQL 数据库服务器的功能。通过利用 PostgreSQL 提供的工具，你可以确保你的空间和非空间数据长期有效，并最大化各种 PostGIS 操作的性能。此外，它还探讨了新的特性，如 PostgreSQL 中的地理空间分片和并行性。

第十一章, 《使用桌面客户端》介绍了如何使用各种开源桌面 GIS 应用程序来消费和操作 PostGIS 中的空间数据。讨论了几个应用程序，以突出不同的空间数据交互方法，并帮助你找到适合任务的正确工具。

第十二章, 《位置隐私保护机制简介》对位置隐私的概念进行了初步介绍，并展示了两种不同的位置隐私保护机制的实现，这些机制可以包含在商业应用程序中，为用户的位置数据提供基本保护。

# 为了充分利用本书

在进一步学习本书之前，你需要安装最新版本的 PostgreSQL 和 PostGIS（分别为 9.6 或 10.3 和 2.3 或 2.41）。如果你更喜欢图形 SQL 工具，也可以安装 pgAdmin（1.18）。对于大多数计算环境（Windows、Linux、macOS X），安装程序和软件包包括 PostGIS 的所有必需依赖项。PostGIS 的最小必需依赖项包括 PROJ.4、GEOS、libjson 和 GDAL。

为了理解并适应本书中的代码，需要具备基本的 SQL 语言知识。

# 下载示例代码文件

你可以从[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果你在其他地方购买了这本书，你可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给你。

您可以通过以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 登录或注册。

1.  选择“支持”标签页。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书代码包也托管在 GitHub 上，地址为 [`github.com/PacktPublishing/PostGIS-Cookbook-Second-Edition`](https://github.com/PacktPublishing/PostGIS-Cookbook-Second-Edition)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富的书籍和视频目录的代码包可供选择，这些书籍和视频可在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/PostGISCookbookSecondEdition_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/PostGISCookbookSecondEdition_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。以下是一个示例：“我们将导入存储从各种 RSS 源收集的一系列网络新闻的 `firenews.csv` 文件。”

代码块如下设置：

```py
SELECT ROUND(SUM(chp02.proportional_sum(ST_Transform(a.geom,3734), b.geom, b.pop))) AS population 
  FROM nc_walkzone AS a, census_viewpolygon as b
  WHERE ST_Intersects(ST_Transform(a.geom, 3734), b.geom)
  GROUP BY a.id;
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
SELECT ROUND(SUM(chp02.proportional_sum(ST_Transform(a.geom,3734), b.geom, b.pop))) AS population 
  FROM nc_walkzone AS a, census_viewpolygon as b
  WHERE ST_Intersects(ST_Transform(a.geom, 3734), b.geom)
  GROUP BY a.id;
```

任何命令行输入或输出都按以下方式编写：

```py
> raster2pgsql -s 4322 -t 100x100 -F -I -C -Y C:\postgis_cookbook\data\chap5\PRISM\us_tmin_2012.*.asc chap5.prism | psql -d postgis_cookbook
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“点击“下一步”按钮将您带到下一个屏幕。”

警告或重要注意事项如下所示。

小贴士和技巧如下所示。

# 部分

在本书中，您将找到一些频繁出现的标题（*准备工作*、*如何操作…*、*如何工作…*、*更多内容…* 和 *相关内容*）。

为了清楚地说明如何完成食谱，请按照以下方式使用这些部分：

# 准备工作

本节告诉您在食谱中可以期待什么，并描述了为食谱设置任何软件或任何必需的初步设置的方法。

# 如何操作…

本节包含遵循食谱所需的步骤。

# 如何工作…

本节通常包含对上一节发生情况的详细解释。

# 更多内容…

本节包含有关食谱的附加信息，以便您对食谱有更深入的了解。

# 相关内容

本节提供了对食谱其他有用信息的链接。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**: 请通过`feedback@packtpub.com`发送电子邮件，并在邮件主题中提及书籍标题。如果你对这本书的任何方面有疑问，请通过`questions@packtpub.com`给我们发送电子邮件。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们将不胜感激，如果你能向我们报告这一点。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择你的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`与我们联系，并附上材料的链接。

**如果你有兴趣成为作者**: 如果你有一个你擅长的主题，并且你对撰写或为书籍做出贡献感兴趣，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦你阅读并使用了这本书，为什么不在你购买它的网站上留下评论呢？潜在的读者可以看到并使用你的无偏见意见来做出购买决定，我们 Packt 可以了解你对我们的产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问[packtpub.com](https://www.packtpub.com/)。
