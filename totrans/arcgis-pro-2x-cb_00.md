# 前言

ArcGIS Pro 是 Esri 最新的桌面 GIS 应用程序，具有强大的可视化、维护和分析数据工具。ArcGIS Pro 利用现代功能区界面和 64 位处理来提高 GIS 的速度和效率。它允许用户快速轻松地创建令人惊叹的 2D 和 3D 地图。

如果你想要全面了解如何使用 ArcGIS Pro 进行各种类型的地理空间分析，如何处理各种数据格式，以及如何通过 ArcGIS Online 分享你的 ArcGIS Pro 结果，那么这本书就是为你准备的。

从对 ArcGIS Pro 和如何处理项目进行复习开始，这本书将迅速带你了解应用内支持的各种数据格式的使用方法。你将学习每种格式的限制，例如 Shapefiles、Geodatabase 和 CAD 文件，并了解如何将数据转换为最适合你需求的格式。接下来，你将学习如何将来自外部源的数据表链接到现有的 GIS 数据中，以扩展在 ArcGIS Pro 中可用的数据量。从那里，你将深入研究使用 ArcGIS Pro 编辑 2D 和 3D 数据的方法，并了解拓扑如何用于确保数据完整性。我们将探索不同的分析工具，这些工具允许我们执行 2D 和 3D 分析。最后，本书将展示如何通过 ArcGIS Online 分享数据和地图，以及如何与网络和移动应用程序一起使用。本书还将介绍 ArcGIS Arcade，这是 Esri 的新表达式语言，它支持整个 ArcGIS 平台。

# 本书面向的对象

如果你有限的经验使用 ArcGIS，并想了解更多关于 ArcGIS Pro 的工作原理以及它包含的强大数据维护、分析和共享工具，那么这本书就是为你准备的。它也是那些从 ArcGIS Desktop（ArcMap 和 ArcCatalog）迁移到 ArcGIS Pro 的用户的一个极好的资源。

# 本书涵盖的内容

第一章，*ArcGIS Pro 功能和术语*，回顾了 ArcGIS Pro 的基本功能和术语。

第二章，*创建和存储数据*，检查了 ArcGIS Pro 使用不同存储数据格式的能力。

第三章，*链接数据在一起*，解释了如何将外部数据链接到你的 GIS 中，以便用于分析和显示。

第四章，*编辑空间和表格数据*，探讨了在 GIS 数据库中创建和编辑新特征的多种工具。

第五章，*使用拓扑验证和编辑数据*，展示了如何使用拓扑来提高数据的准确性并增加编辑效率。

第六章，*投影和坐标系统基础*，解释了坐标系统在 GIS 中的重要性以及如何将数据从一个系统移动到另一个系统。

第七章，*转换数据*，指导您使用各种方法将 GIS 数据从一种存储格式转换为另一种格式。

第八章，*邻近度分析*，探讨了确定地图上特征之间距离远近的不同工具。

第九章，*空间统计和热点分析*，展示了您如何定位簇、发现模式并确定一组特征的时空中心。

第十章，*3D 地图和 3D 分析师*，展示了您如何使用 ArcGIS Pro 和 3D 分析师扩展执行 3D 分析，例如计算视线和体积。

第十一章，*介绍 Arcade*，展示了您如何使用新的 Arcade 表达式语言创建标签和符号表达式。

第十二章，*介绍 ArcGIS Online*，指导您连接到您的 ArcGIS Online 账户以及如何访问他人发布的用于创建网络地图的内容。

第十三章，*将您的内容发布到 ArcGIS Online*，指导您将您的内容发布到 ArcGIS Online，以便您的组织中的其他人可以访问。

第十四章，*使用 ArcGIS Online 创建 Web 应用*，展示了您如何创建自己的 Web GIS 应用，而无需成为程序员。

# 为了充分利用本书

1.  本书假设读者至少对 ArcGIS Pro 有一些了解。建议您阅读并完成 Packt Publishing 出版的《学习 ArcGIS Pro》一书中的练习，或者有使用 ArcGIS Pro 或 ArcMap 的先前实际经验。

1.  为了完成本书中的所有食谱，您需要安装 ArcGIS Pro 2.1 或更高版本，并具有标准或更高版本的许可证，以及 ArcGIS Pro 的 3D 分析师扩展许可证。如果您仅限于基本许可证或没有 3D 分析师许可证，您仍然可以完成大多数食谱，但不是全部。

1.  您需要一个 ArcGIS Online 的用户名和登录密码，并且至少具有发布者级别的权限。

1.  您需要按照*下载示例代码文件*部分中的说明下载和安装样本数据文件。

1.  如果您没有 ArcGIS Pro、3D Analyst 扩展或 ArcGIS Online 的许可证，您可以从 Esri 请求试用许可证，请访问[`www.esri.com/arcgis/trial`](http://www.esri.com/arcgis/trial)。

# 下载示例代码文件

您可以从 [www.packtpub.com](http://www.packtpub.com) 的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  请登录或注册至 [www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误表”。

1.  在搜索框中输入书籍名称，并遵循屏幕上的说明。

一旦文件下载完成，请确保您使用最新版本的软件解压缩或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/ArcGIS-Pro-2.x-Cookbook`](https://github.com/PacktPublishing/ArcGIS-Pro-2.x-Cookbook)。我们还有其他来自我们丰富图书和视频目录的代码包，可在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/ArcGISPro2.xCookbook_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/ArcGISPro2.xCookbook_ColorImages.pdf)。

# 使用的约定

本书使用了许多文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“通过在左侧区域单击 `C:\` 来导航到 `C:\Student\ArcGISProCookbook\Chapter2\RasterVector`。”

代码块设置如下：

```py
if (cond=="Good")
  {
  return "<CLR green='255'>"+name+"</CLR>"
  }
if (cond=="Fair")
  {
  return name
  }
else
  {
  return "<BOL><CLR red='255'>"+name+"</CLR></BOL>"
  }
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下所示。以下是一个示例：“在功能区中选择“地图”选项卡，然后单击位于“书签”下的小箭头。”

警告或重要注意事项如下所示。

小技巧和技巧如下所示。

# 章节

在本书中，您会发现一些频繁出现的标题（如 *准备就绪*、*如何操作...*、*工作原理...*、*还有更多...* 和 *另请参阅*）。

为了清楚地说明如何完成食谱，请按照以下方式使用这些章节：

# 准备就绪

本节告诉您在食谱中可以期待什么，并描述如何设置任何软件或任何为食谱所需的初步设置。

# 如何操作…

本节包含遵循食谱所需的步骤。

# 工作原理…

本节通常包含对前节发生事件的详细解释。

# 还有更多…

本节包含有关食谱的附加信息，以便您对食谱有更深入的了解。

# 另请参阅

本节提供了对食谱其他有用信息的链接。

# 联系我们

我们始终欢迎读者的反馈。

**总体反馈**：请发送邮件至 `feedback@packtpub.com` 并在邮件主题中提及书籍标题。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 发送邮件给我们。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一错误。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，我们将不胜感激，如果您能提供位置地址或网站名称。请通过 `copyright@packtpub.com` 发送邮件给我们，并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且对撰写或参与一本书籍感兴趣，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买书籍的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
