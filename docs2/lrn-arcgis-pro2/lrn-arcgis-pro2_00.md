前言

*Learning ArcGIS Pro* 解释了如何成功使用这款新的强大桌面 **地理信息系统**（**GIS**）应用程序来创建地图、执行空间分析和维护数据。本书包含基于真实案例的实践练习，将向你展示如何使用 ArcGIS Pro 可视化、分析和维护 GIS 数据。

配备了强大的工具，ArcGIS Pro 2 是 Esri 最新的桌面 GIS 应用程序，它使用现代功能区界面和 64 位处理器，使使用 GIS 更加快速和高效。本版 *Learning ArcGIS Pro* 将向你展示如何使用这款强大的桌面 GIS 应用程序来创建地图、执行空间分析和维护数据。

本书首先向您展示如何安装 ArcGIS，并列出软件和硬件先决条件。然后，您将了解命名用户许可的概念，并学习如何导航新的功能区界面以利用 ArcGIS Pro 的功能来管理地理空间数据。一旦您熟悉了新界面，您将构建您的第一个 GIS 项目，并学习如何使用不同的项目资源。

本书展示了如何通过添加图层、设置和管理符号和标签来创建 2D 和 3D 地图。你还将发现如何使用分析工具来可视化地理空间数据。在后面的章节中，你将介绍 Arcade，ArcGIS 的新轻量级表达式语言，然后进一步学习使用 Arcade 表达式创建复杂标签。

你将学习如何导航用户界面来创建地图、执行分析和管理数据。你将能够根据离散属性值或值范围显示数据，并使用 Arcade 在 GIS 地图上根据一个或多个属性标记要素。

你还将学习如何使用地图系列功能创建地图集，并能够与其他 GIS 社区成员共享 ArcGIS Pro 地图、项目和数据。本书还探讨了最广泛使用的地理处理工具，用于执行空间分析，并解释了如何根据常见工作流程创建任务以标准化流程。你还将学习如何使用 ModelBuilder 和 Python 脚本自动化流程。

在本 ArcGIS Pro 书籍结束时，你将掌握使用 ArcGIS Pro 2.x 的核心技能。

# 第一章：本书面向对象

如果你想要学习如何使用 ArcGIS Pro 创建地图、编辑和分析地理空间数据，这本书就是为你准备的。不需要 GIS 基础知识或任何 GIS 工具或 ArcGIS 软件套件的实践经验。只需要基本的 Windows 技能，如导航和文件管理即可。

# 本书涵盖内容

第一章，*介绍 ArcGIS Pro*，介绍了 ArcGIS Pro 并解释了它与其他 ArcGIS 产品的一些功能。它还提供了其功能的一般概述，并讨论了安装和许可要求。

第二章，*导航功能区界面*，介绍了 ArcGIS Pro 基于功能区界面和常用界面面板或窗口。它解释了如何使用它来访问 ArcGIS Pro 项目中的数据、地图和工具。

第三章，*创建二维地图*，展示了如何在 ArcGIS Pro 项目框架内创建二维地图。您将学习如何添加和管理图层，控制符号，标注要素，并配置其他属性。

第四章，*创建三维场景*，展示了用户如何在他们的项目中创建三维地图。您将学习如何添加图层，拉伸图层以显示高度，并应用三维符号。

第五章，*创建和使用项目*，介绍了使用项目来管理 GIS 内容的概念。您将学习如何创建和组织项目。您还将学习如何创建模板项目。

第六章，*创建布局*，展示了如何使用 ArcGIS Pro 创建有效的布局。

第七章，*使用地图系列创建地图集*，解释了启用和配置地图系列功能所需的过程，以便您可以生成自己的地图集。大地图难以操作，并且变得难以使用。小地图又无法在野外展示所需级别的细节。常见的做法是为该区域创建地图集或系列。ArcGIS Pro 内置了创建这些地图集的功能，本章将向您展示如何操作。

第八章，*学习编辑空间数据*，为您提供了 ArcGIS Pro 中编辑工作流程的基本理解，并解释了如何使用许多最常用的工具来维护和更新您的 GIS 数据。

第九章，*学习编辑表格数据*，解释了如何编辑和维护您 GIS 中要素的属性数据。

第十章，*使用地理处理工具进行分析*，介绍了许多最常用的工具，解释了它们可以在哪里访问，并涵盖了将决定您在 ArcGIS Pro 中可用的工具的因素。

第十一章，*创建和使用任务*，展示了如何创建任务以改进您办公室内常见工作流程的效率和标准化。

第十二章，*使用 ModelBuilder 和 Python 自动化流程*，介绍了创建简单模型和 Python 脚本所需的基本概念和技能，用于 ArcGIS Pro。

第十三章，*与他人共享您的作品*，展示了在 ArcGIS Pro 中与他人共享地图、数据和流程的不同方法。

第十四章，*使用 Arcade 表达式进行标签和符号化*，介绍了 Arcade 的基本用法和语法。Arcade 是一种新的轻量级表达式语言，用于 ArcGIS。它允许您创建可以生成文本标签或控制符号的表达式。

GIS 术语表，提供与重要 GIS 术语相关的定义和示例描述。

# 为了充分利用本书

您需要以下内容来使用本书：

+   **ArcGIS Pro 2.6 或更高版本**—基本或更高许可级别

+   互联网连接

+   来自 GitHub 的练习数据

以下表格解释了**操作系统**（**OS**）的要求：

| **本书涵盖的软件** | **操作系统要求** |
| --- | --- |
| **ArcGIS Pro 2.6** | **Windows 8.1** 或 **Windows 10** (64 位) |
| **ArcGIS Online** | 不适用 |

本书将向您介绍 ArcGIS Pro 的安装过程，以及如何确定您的计算机能否运行该应用程序。

## 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packtpub.com/support](https://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载”。

1.  在搜索框中输入本书的名称，并遵循屏幕上的说明。

文件下载后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Learning-ArcGIS-Pro-2-Second-Edition`](https://github.com/PacktPublishing/Learning-ArcGIS-Pro-2-Second-Edition)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。查看它们！

## 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`static.packt-cdn.com/downloads/9781839210228_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781839210228_ColorImages.pdf)。

## 使用的约定

本书使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“在注释描述之后的下一行是一个`import`命令，该命令加载`arcpy`模型，以便脚本可以访问 ArcGIS 功能。”

代码块设置如下：

```py
#Specifies the input variables for the script tools
#If the data is moved or in a different database then these paths will need to be updated
Parcels = "C:\\Student\\IntroArcPro\\Databases\\Trippville_GIS.gdb\\Base\\Parcels"
Parcels_Web = "C:\\Student\\IntroArcPro\\Chapter11\\Ex11.gdb\\Parcels_Web"
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中如下所示。以下是一个示例：“在 ModelBuilder 选项卡上的工具按钮组中点击工具按钮。”

警告或重要注意事项如下所示。

小贴士和技巧如下所示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并给我们发送电子邮件至`customercare@packtpub.com`。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您能向我们报告。请访问[www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法副本，我们将非常感激您能提供位置地址或网站名称。请通过链接材料与我们联系至`copyright@packtpub.com`。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

## 评论

请留下评论。一旦您阅读并使用了这本书，为何不在您购买书籍的网站上留下评论？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/)。
