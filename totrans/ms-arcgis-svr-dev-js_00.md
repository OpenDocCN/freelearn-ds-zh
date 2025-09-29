# 前言

在这个数字时代，我们期望地图无论在哪里都能可用。我们在手机上搜索驾驶路线。我们使用应用程序查找商业地点和附近的餐厅。维护人员使用数字地图定位地下几英尺的资产。政府官员和高级管理人员需要最新的位置数据来做出影响许多人生的重要决策。

ESRI 在数字制图领域一直是 30 年的行业领导者。通过 ArcGIS Server 等产品，他们使公司、政府机构和数字制图师能够将地图和地理数据发布到互联网上。使用 ArcGIS API for JavaScript 编写的网站和应用使这些地图在桌面和移动浏览器上均可访问。该 API 还提供了创建用于数据展示、收集和分析的强大应用程序所需的构建块。

虽然有很多示例、博客文章和书籍可以帮助你开始使用 ArcGIS API for JavaScript 开发应用程序，但它们通常不会深入探讨。它们没有讨论使用这些工具开发应用程序的陷阱、限制和最佳实践。这本书试图实现的就是这一步。

# 本书涵盖内容

第一章, *你的第一个地图应用程序*，介绍了 ArcGIS Server 和 ArcGIS API for JavaScript。在本章中，你将学习创建地图和添加图层的基础知识。

第二章, *深入 API*，概述了 ArcGIS API for JavaScript 中可用的许多 API 组件、小部件和功能。本章提供了如何使用这些组件的解释。

第三章, *Dojo 小部件系统*，探讨了 ArcGIS API for JavaScript 构建在其中的综合 JavaScript 框架——Dojo 框架。本章探讨了异步模块设计（AMD）的内部工作原理以及如何为我们的地图应用程序构建自定义小部件。

第四章, *在 REST 中寻找平静*，探讨了 ArcGIS REST API，该 API 定义了 ArcGIS Server 如何与浏览器中的应用程序通信。

第五章, *编辑地图数据*，介绍了如何在 ArcGIS Server 中编辑存储的地理数据。本章探讨了 ArcGIS JavaScript API 提供的模块、小部件和用户控件。

第六章, *绘制你的进度图*，探讨了如何通过图表和图形传达地图要素信息。本章不仅讨论了使用 Dojo 框架制作的图表和图形，还探讨了如何集成其他图表库，如 D3.js 和 HighCharts.js。

第七章, *与其他人合作*，讨论了将其他流行的库集成到使用 ArcGIS API for JavaScript 编写的应用程序中的方法。本章还探讨了将框架与 jQuery、Backbone.js、Angular.js 和 Knockout.js 结合使用。

第八章, *美化您的地图*，涵盖了在映射应用程序中使用 CSS 样式。本章检查了 Dojo 框架如何布局元素，并探讨了添加 Bootstrap 框架来美化映射应用程序。

第九章, *移动开发*, 探讨了在移动设备上开发的需求，并讨论了尝试使您的应用程序“移动友好”的陷阱。本章实现了由 Dojo 框架提供的 dojox/mobile 框架，作为为移动使用样式元素的方法。

第十章, *测试*, 讨论了测试驱动开发和行为驱动开发的需求。本章讨论了如何使用 Intern 和 Jasmine 测试框架测试应用程序。

第十一章, *ArcGIS 开发的未来*，探讨了新的映射和应用服务，如 ArcGIS Online 和 Web AppBuilder。您将学习如何围绕 ArcGIS Online 网上地图开发应用程序。

# 您需要这本书什么

对于本书的所有章节，您需要一个现代浏览器和一个文本编辑器来创建文件。您可以选择任何文本编辑器，但具有语法高亮和某种 JavaScript 代码高亮的编辑器会很有帮助。对于第五章, *编辑地图数据* 和 第九章, *移动开发* 中的练习，您还需要访问一个网络托管服务器来测试用 Java、PHP 或 .NET 编写的文件。最后，对于第十章, *测试* 中的练习，您需要在您的机器上安装最新版本的 Node.js。

# 这本书适合谁

本书适合有一定 ArcGIS Server 经验的 Web 开发人员，或者有一定 HTML、JavaScript 和 CSS 编写经验的地理空间专业人士。本书假设您已经查看了一些由 ESRI 提供的 ArcGIS API for JavaScript 示例。更重要的是，本书适合希望更深入地使用 ArcGIS Server 进行应用程序开发的读者。Packt Publishing 提供的其他书籍，如 Eric Pimpler 的 *Building Web and Mobile ArcGIS Server Applications with JavaScript* 或 Hussein Nasser 的 *Building Web Applications with ArcGIS*，可能更适合作为 ArcGIS Server 和 ArcGIS API for JavaScript 的入门。

# 惯例

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称将如下所示：“我们正在处理人口普查数据，让我们称它为`census.html`。”

代码块设置如下：

```py
<!DOCTYPE html>
<html>
<head></head>
<body></body>
</html>
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
<!DOCTYPE html>
<html>
<head>
 <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>
 <meta http-equiv="X-UA-Compatible" content="IE=Edge" />
 <meta name="viewport" content="initial-scale=1, maximum-scale=1,user-scalable=no"/>
 <title>Census Map</title>
 <link rel="stylesheet" href="http://js.arcgis.com/3.13/esri/css/esri.css" />
 <style>
 html, body {
 border: 0;
 margin: 0;
 padding: 0;
 height: 100%;
 width: 100%;
 }
 </style>
 <script type="text/javascript">
 dojoConfig = {parseOnLoad: true, debug: true};
 </script>
 <script type="text/javascript" src="img/" ></script>
</head>
```

**新术语**和**重要词汇**将以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中如下所示：“我们将其精确地定位在右上角，并为**人口普查**按钮留出一点垂直居中的空间。”

### 注意

警告或重要提示将以这样的框显示。

### 小贴士

小技巧和技巧如下所示。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们您对这本书的看法——您喜欢什么或不喜欢什么。读者反馈对我们来说非常重要，因为它帮助我们开发出您真正能从中获得最大收益的书籍。

要向我们发送一般反馈，只需发送电子邮件至`<feedback@packtpub.com>`，并在邮件主题中提及书籍的标题。

如果您在某个主题上具有专业知识，并且您对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在您已经是 Packt 图书的骄傲拥有者了，我们有一些东西可以帮助您从您的购买中获得最大收益。

## 下载示例代码

您可以从您的账户中下载示例代码文件，这些文件适用于您购买的所有 Packt Publishing 书籍。[`www.packtpub.com`](http://www.packtpub.com)。如果您在其他地方购买了这本书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

## 下载本书的彩色图像

我们还为您提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。彩色图像将帮助您更好地理解输出的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/6459OT_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/6459OT_ColorImages.pdf)下载此文件。

## 错误清单

尽管我们已经尽一切努力确保我们内容的准确性，但错误仍然会发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。这样做可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问 [`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将在**勘误**部分显示。

## 侵权

侵权是互联网上所有媒体持续存在的问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<copyright@packtpub.com>` 联系我们，并提供涉嫌侵权材料的链接。

我们感谢您在保护我们作者和我们提供有价值内容的能力方面的帮助。

## 询问

如果您在这本书的任何方面遇到问题，您可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决问题。
