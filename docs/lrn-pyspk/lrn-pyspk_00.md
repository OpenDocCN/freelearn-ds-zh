# 前言

据估计，到 2013 年，全世界产生了大约 4.4 泽字节的数据；也就是说，4.4 *十亿* 太字节！到 2020 年，我们（人类）预计将产生十倍于此的数据。随着数据以每秒 literally 的速度增长，以及人们对从中获取意义的日益增长的需求，2004 年，谷歌员工杰弗里·迪恩和桑杰·格马瓦特发表了开创性的论文 *MapReduce：在大型集群上简化数据处理*。从那时起，利用这一概念的技术开始迅速增长，Apache Hadoop 最初是最受欢迎的。它最终创建了一个包括 Pig、Hive 和 Mahout 等抽象层的 Hadoop 生态系统——所有这些都利用了简单的 map 和 reduce 概念。

然而，尽管 MapReduce 每天能够处理 PB 级的数据，但它仍然是一个相当受限的编程框架。此外，大多数任务都需要读写磁盘。看到这些缺点，2009 年，Matei Zaharia 开始在他的博士期间研究 Spark。Spark 最初于 2012 年发布。尽管 Spark 基于相同的 MapReduce 概念，但它处理数据和组织任务的高级方式使其比 Hadoop（对于内存计算）快 100 倍。

在这本书中，我们将使用 Python 引导您了解 Apache Spark 的最新版本。我们将向您展示如何读取结构化和非结构化数据，如何使用 PySpark 中的一些基本数据类型，构建机器学习模型，操作图，读取流数据，并在云中部署您的模型。每一章都将解决不同的问题，到本书结束时，我们希望您能够足够了解以解决我们没有空间在此处涵盖的其他问题。

# 本书涵盖的内容

第一章，*理解 Spark*，介绍了 Spark 世界，概述了技术和作业组织概念。

第二章，*弹性分布式数据集*，涵盖了 RDD，这是 PySpark 中可用的基本、无模式的数据库结构。

第三章，*DataFrame*，提供了关于一种数据结构的详细概述，这种数据结构在效率方面连接了 Scala 和 Python 之间的差距。

第四章，*为建模准备数据*，指导读者在 Spark 环境中清理和转换数据的过程。

第五章，*介绍 MLlib*，介绍了在 RDD 上工作的机器学习库，并回顾了最有用的机器学习模型。

第六章，*介绍 ML 包*，涵盖了当前主流的机器学习库，并概述了目前所有可用的模型。

第七章, *GraphFrames*，将引导您了解一种新的结构，使使用图解决问题变得简单。

第八章, *TensorFrames*，介绍了 Spark 与 TensorFlow 深度学习世界之间的桥梁。

第九章, *使用 Blaze 的多语言持久性*，描述了 Blaze 如何与 Spark 配合使用，以便更容易地从各种来源抽象数据。

第十章, *结构化流*，提供了 PySpark 中可用的流工具概述。

第十一章, *打包 Spark 应用程序*，将引导您了解将代码模块化并通过命令行界面提交到 Spark 以执行步骤。

更多信息，我们提供了以下两个附加章节：

*安装 Spark*: [`www.packtpub.com/sites/default/files/downloads/InstallingSpark.pdf`](https://www.packtpub.com/sites/default/files/downloads/InstallingSpark.pdf)

*免费 Spark 云服务*: [`www.packtpub.com/sites/default/files/downloads/FreeSparkCloudOffering.pdf`](https://www.packtpub.com/sites/default/files/downloads/FreeSparkCloudOffering.pdf)

# 您需要为本书准备什么

为了阅读本书，您需要一个个人电脑（可以是 Windows 机器、Mac 或 Linux）。要运行 Apache Spark，您将需要 Java 7+以及安装并配置好的 Python 2.6+或 3.4+环境；我们使用的是 Python 3.5 版本的 Anaconda 发行版，可以从[`www.continuum.io/downloads`](https://www.continuum.io/downloads)下载。

我们在本书中随机使用的 Python 模块都是 Anaconda 预安装的。我们还使用了 GraphFrames 和 TensorFrames，这些模块可以在启动 Spark 实例时动态加载：要加载这些模块，您只需要一个互联网连接。如果这些模块中的一些目前没有安装到您的机器上，也没有关系——我们将引导您完成安装过程。

# 本书面向对象

本书面向所有希望学习大数据中增长最快的技术的读者：Apache Spark。我们希望即使是数据科学领域的资深从业者也能发现一些示例令人耳目一新，一些高级主题引人入胜。

# 规范

在本书中，您将找到多种文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名显示如下：

代码块设置如下：

```py
data = sc.parallelize(
    [('Amber', 22), ('Alfred', 23), ('Skye',4), ('Albert', 12), 
     ('Amber', 9)])
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
rdd1 = sc.parallelize([('a', 1), ('b', 4), ('c',10)])
rdd2 = sc.parallelize([('a', 4), ('a', 1), ('b', '6'), ('d', 15)])
rdd3 = rdd1.leftOuterJoin(rdd2)
```

任何命令行输入或输出都按以下方式编写：

```py
java -version

```

**新术语**和**重要词汇**将以粗体显示。你会在屏幕上看到这些词汇，例如在菜单或对话框中，文本将显示为：“点击**下一步**按钮将你带到下一屏幕。”

### 注意

警告或重要注意事项将以如下框显示。

### 小贴士

小贴士和技巧看起来像这样。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们你对这本书的看法——你喜欢什么或可能不喜欢什么。读者反馈对我们开发你真正能从中获得最大收益的标题非常重要。

要发送一般反馈，请简单地将电子邮件发送到`<feedback@packtpub.com>`，并在邮件主题中提及书籍标题。

如果你在某个领域有专业知识，并且对撰写或参与书籍感兴趣，请参阅我们的作者指南：[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在你已经是 Packt 书籍的骄傲拥有者，我们有一些东西可以帮助你从购买中获得最大收益。

## 下载示例代码

你可以从[`www.packtpub.com`](http://www.packtpub.com)的账户下载你购买的所有 Packt 书籍的示例代码文件。如果你在其他地方购买了这本书，你可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给你。

所有代码也都在 GitHub 上提供：[`github.com/drabastomek/learningPySpark`](https://github.com/drabastomek/learningPySpark)。

你可以通过以下步骤下载代码文件：

1.  登录或使用电子邮件地址和密码注册我们的网站。

1.  将鼠标指针悬停在顶部的**支持**标签上。

1.  点击**代码下载与勘误**。

1.  在**搜索**框中输入书籍名称。

1.  选择你想要下载代码文件的书籍。

1.  从下拉菜单中选择你购买此书籍的地方。

1.  点击**代码下载**。

文件下载完成后，请确保使用最新版本解压缩或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

本书代码包也托管在 GitHub 上：[`github.com/PacktPublishing/Learning-PySpark`](https://github.com/PacktPublishing/Learning-PySpark)。我们还有其他来自我们丰富图书和视频目录的代码包可供选择，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

## 下载本书的彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。彩色图像将帮助你更好地理解输出的变化。你可以从[`www.packtpub.com/sites/default/files/downloads/LearningPySpark_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/LearningPySpark_ColorImages.pdf)下载此文件。

## 勘误

尽管我们已经尽最大努力确保内容的准确性，错误仍然可能发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者感到沮丧，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站，或添加到该标题的勘误部分下的现有勘误列表中。您可以通过从[`www.packtpub.com/support`](http://www.packtpub.com/support)选择您的标题来查看任何现有的勘误。

## 侵权

互联网上版权材料的侵权是一个跨所有媒体持续存在的问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现我们作品的任何非法副本，无论形式如何，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

如果您在本书的任何方面遇到问题，请通过 `<copyright@packtpub.com>` 联系我们，并提供疑似侵权材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面提供的帮助。

## 询问

如果您在本书的任何方面遇到问题，请通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决。
