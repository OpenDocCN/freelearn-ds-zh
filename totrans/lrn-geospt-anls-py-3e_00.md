# 前言

本书首先为您介绍地理空间分析背景，然后介绍所使用的技术和科技流程，并将该领域划分为其组成部分的专业领域，例如 **地理信息系统** (**GIS**)、遥感、高程数据、高级建模和实时数据。本书的重点是利用强大的 Python 语言和框架有效地进行地理空间分析，我们将专注于使用纯 Python 以及某些 Python 工具和 API，并使用通用算法。读者将能够分析各种形式的地理空间数据，了解实时数据跟踪，并了解如何将所学知识应用于有趣的场景。

尽管在示例中使用了多个第三方地理空间库，但我们将尽最大努力在可能的情况下使用纯 Python，不依赖任何库。这种专注于纯 Python 3 示例的做法将使本书区别于该领域的几乎所有其他资源。我们还将介绍一些在本书前版中未提及的流行库。

# 本书面向对象

本书面向任何希望了解数字制图和分析，并使用 Python 或任何其他脚本语言进行数据自动化或手动处理的人。本书主要针对希望使用 Python 进行地理空间建模和 GIS 分析的 Python 开发者、研究人员和分析人员。

# 要充分利用本书

本书假设您具备 Python 编程语言的基本知识。您将需要 Python（3.7 或更高版本），最低硬件要求为 300-MHz 处理器，128 MB 的 RAM，1.5 GB 的可用硬盘空间，以及 Windows、Linux 或 macOS X 操作系统。

# 下载示例代码文件

您可以从 [www.packt.com](http://www.packt.com) 账户下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问 [www.packtpub.com/support](https://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择“支持”标签。

1.  点击“代码下载”。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

文件下载完成后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   Windows 版本的 WinRAR/7-Zip

+   Mac 版本的 Zipeg/iZip/UnRarX

+   Linux 版本的 7-Zip/PeaZip

本书代码包也托管在 GitHub 上，地址为 [https://github.com/PacktPublishing/Learning-Geospatial-Analysis-with-Python-Third-Edition](https://github.com/PacktPublishing/Learning-Geospatial-Analysis-with-Python-Third-Edition)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还提供其他代码包，这些代码包来自我们丰富的书籍和视频目录，可在 **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)** 上找到。查看它们吧！

# 下载彩色图像

我们还提供包含本书中使用的截图/图表的彩色图像的PDF文件。您可以从这里下载：[https://static.packt-cdn.com/downloads/9781789959277_ColorImages.pdf](https://static.packt-cdn.com/downloads/9781789959277_ColorImages.pdf)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟URL、用户输入和Twitter昵称。以下是一个示例：“为了演示这一点，以下示例访问了我们刚刚看到的相同文件，但使用`urllib`而不是`ftplib`。”

代码块以如下方式设置：

[PRE0]

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

[PRE1]

任何命令行输入或输出都应如下编写：

[PRE2]

**粗体**：表示新术语、重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“从管理面板中选择系统信息。”

警告或重要提示显示如下。

小贴士和技巧显示如下。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并 通过 `customercare@packtpub.com` 发送邮件给我们。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告这一点。请访问 [www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式发现我们作品的非法副本，我们将不胜感激，如果您能提供位置地址或网站名称，请通过 `copyright@packt.com` 联系我们，并提供材料链接。

**如果您想成为一名作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦您阅读并使用过这本书，为何不在您购买它的网站上留下评论？潜在读者可以查看并使用您的客观意见来做出购买决定，我们Packt可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

如需了解Packt的更多信息，请访问 [packt.com](http://www.packt.com/)。
