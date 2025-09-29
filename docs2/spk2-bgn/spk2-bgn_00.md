# 前言

命名为 Spark 的数据处理框架最初是为了证明，通过在多次迭代中重复使用数据集，它可以在 Hadoop MapReduce 作业表现不佳的地方提供价值。研究论文《Mesos：数据中心细粒度资源共享平台》讨论了 Spark 设计背后的哲学。加州大学伯克利分校的研究人员为了测试 Mesos 而构建的一个非常简单的参考实现，后来发展成为一个完整的数据处理框架，后来成为 Apache 项目中最活跃的项目之一。它从一开始就被设计用来在 Hadoop、Mesos 等集群上以及独立模式下进行分布式数据处理。Spark 是一个基于 JVM 的数据处理框架，因此它可以在支持 JVM 应用程序的大多数操作系统上运行。Spark 在 UNIX 和 Mac OS X 平台上广泛安装，Windows 的采用率正在增加。

Spark 提供了使用 Scala、Java、Python 和 R 编程语言统一的编程模型。换句话说，无论使用哪种语言来编写 Spark 应用程序，API 在所有语言中几乎都是相同的。这样，组织可以采用 Spark 并在他们选择的编程语言中开发应用程序。这也使得如果需要的话，Spark 应用程序可以从一种语言快速迁移到另一种语言而无需太多努力。Spark 的大部分开发都是使用 Scala 进行的，因此 Spark 编程模型本质上支持函数式编程原则。最基础的 Spark 数据抽象是弹性分布式数据集（RDD），所有其他库都是基于它构建的。基于 RDD 的 Spark 编程模型是开发者可以构建数据处理应用程序的最低级别。

Spark 发展迅速，以满足更多数据处理用例的需求。当在产品路线图上采取这样的前瞻性步骤时，出现了使编程对商业用户更高级别的需求。建立在 Spark Core 之上的 Spark SQL 库，通过其 DataFrame 抽象，是为了满足大量非常熟悉无处不在的 SQL 的开发者的需求。

数据科学家使用 R 来满足他们的计算需求。R 的最大局限性是所有需要处理的数据都应该*适合*在运行 R 程序的计算机的主内存中。Spark 的 R API 将数据科学家引入了他们熟悉的数据框抽象的分布式数据处理世界。换句话说，使用 Spark 的 R API，数据处理可以在 Hadoop 或 Mesos 上并行进行，远远超出宿主计算机的本地内存限制。

在当前大规模应用程序收集数据的时代，摄入数据的速度非常高。许多应用用例要求对流数据进行实时处理。建立在 Spark Core 之上的 Spark Streaming 库正是如此。

静态数据或流数据被输入到机器学习算法中，以训练数据模型并使用它们来回答业务问题。在 Spark 之前创建的所有机器学习框架在处理计算机的内存、无法进行并行处理、重复读写周期等方面都有许多限制。Spark 没有这些限制，因此建立在 Spark Core 和 Spark DataFrames 之上的 Spark MLlib 机器学习库最终成为了一个最佳的机器学习库，它将数据处理管道和机器学习活动粘合在一起。

图是一种非常有用的数据结构，在许多特殊用例中被广泛使用。在图数据结构中处理数据的算法计算量很大。在 Spark 之前，出现了许多图处理框架，其中一些在处理方面非常快，但预处理数据以生成图数据结构在大多数这些图处理应用中变成了一个很大的瓶颈。建立在 Spark 之上的 Spark GraphX 库填补了这一空白，使得数据处理和图处理成为连锁活动。

在过去，存在许多数据处理框架，其中许多是专有的，迫使组织陷入供应商锁定陷阱。Spark 为各种数据处理需求提供了一个非常有用的替代方案，无需任何许可费用；同时，它得到了许多领先公司的支持，提供了专业的生产支持。

# 本书涵盖内容

第一章, *Spark 基础* 讨论了 Spark 作为框架的基本原理，包括其 API 和附带库，以及 Spark 交互的整个数据处理生态系统。

第二章, *Spark 编程模型* 讨论了 Spark 中使用的基于函数式编程方法论原则的统一编程模型，并涵盖了弹性分布式数据集（RDD）、Spark 转换和 Spark 动作的基本原理。

第三章, *Spark SQL* 讨论了 Spark SQL，这是 Spark 中最强大的库之一，用于使用无处不在的 SQL 构造与 Spark DataFrame API 结合来操作数据，以及它是如何与 Spark 程序一起工作的。本章还讨论了 Spark SQL 如何用于从各种数据源访问数据，从而实现数据源的数据处理统一。

第四章, 《Spark Programming with R》讨论了 SparkR 或 R on Spark，这是 Spark 的 R API；这使用户能够利用他们熟悉的数据框抽象来使用 Spark 的数据处理能力。它为 R 用户熟悉 Spark 数据处理生态系统提供了一个非常好的基础。

第五章，《Spark Data Analysis with Python》讨论了使用 Spark 进行数据处理和使用 Python 进行数据分析，利用 Python 可用的各种图表和绘图库。本章讨论了将这两个相关活动结合在一起作为一个 Spark 应用程序，其中 Python 是首选的编程语言。

第六章，《Spark Stream Processing》讨论了 Spark Streaming，这是捕获和处理作为流摄取的数据的最强大的 Spark 库之一。还讨论了作为分布式消息代理的 Kafka 和一个作为消息消费者的 Spark Streaming 应用程序。

第七章，《Spark Machine Learning》讨论了 Spark MLlib，这是用于开发入门级机器学习应用程序的最强大的 Spark 库之一。

第八章，《Spark Graph Processing》讨论了 Spark GraphX，这是处理图数据结构最强大的 Spark 库之一，并附带大量用于在图中处理数据的算法。本章涵盖了 GraphX 的基础和一些使用 GraphX 提供的算法实现的用例。

第九章，《Designing Spark Applications》讨论了 Spark 数据处理应用程序的设计和开发，涵盖了本书前几章中提到的 Spark 的各种功能。

# 你需要这本书什么

要运行代码示例并进行进一步的活动以了解更多关于该主题的信息，至少需要在独立机器上安装 Spark 2.0.0 或更高版本。对于第六章，即 Spark Stream Processing，需要安装和配置 Kafka 作为消息代理，其命令行生产者产生消息，而使用 Spark 开发的应用程序作为这些消息的消费者。

# 这本书面向谁

如果你是一名应用程序开发人员、数据科学家或对将 Spark 的数据处理能力与 R 相结合、将数据处理、流处理、机器学习和图处理整合到一个统一且高度互操作的框架中感兴趣，并使用 Scala 或 Python 通过统一的 API 进行数据处理的解决方案架构师，这本书适合你。

# 习惯用法

在本书中，您将找到多种文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称如下所示：“将此属性`spark.driver.memory`自定义为一个更高的值是一个好主意。”

代码块设置如下：

```py
Python 3.5.0 (v3.5.0:374f501f4567, Sep 12 2015, 11:00:19)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
```

任何命令行输入或输出都如下所示：

```py
$ python 
Python 3.5.0 (v3.5.0:374f501f4567, Sep 12 2015, 11:00:19)  
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin 
Type "help", "copyright", "credits" or "license" for more information. 
>>> 

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中如下所示：“本书中的快捷键基于`Mac OS X 10.5+`方案。”

### 注意

警告或重要注意事项如下所示。

### 小贴士

技巧和窍门如下所示。

# 读者反馈

我们的读者反馈总是受欢迎的。告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们很重要，因为它帮助我们开发出您真正能从中获得最大价值的标题。要发送一般反馈，请简单地发送电子邮件至 feedback@packtpub.com，并在邮件主题中提及书的标题。如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已成为 Packt 图书的骄傲拥有者，我们有一些东西可以帮助您充分利用您的购买。

## 下载示例代码

您可以从您的账户下载本书的示例代码文件，账户地址为[`www.packtpub.com`](http://www.packtpub.com)。如果您在其他地方购买了此书，您可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的**支持**选项卡上。

1.  点击**代码下载与勘误**。

1.  在**搜索**框中输入书的名称。

1.  选择您想要下载代码文件的书籍。

1.  从下拉菜单中选择您购买此书的来源。

1.  点击**代码下载**。

文件下载完成后，请确保您使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

本书代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Apache-Spark-2-for-Beginners`](https://github.com/PacktPublishing/Apache-Spark-2-for-Beginners)。我们还有其他来自我们丰富图书和视频目录的代码包可供在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们！

## 下载本书的颜色图像

我们还为您提供了一个包含本书中使用的截图/图表彩色图像的 PDF 文件。彩色图像将帮助您更好地理解输出中的变化。您可以从 [`www.packtpub.com/sites/default/files/downloads/ApacheSpark2forBeginners_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/Bookname_ColorImages.pdf) 下载此文件。

## 错误清单

尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以避免其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何错误清单，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**错误提交表单**链接，并输入您的错误详细信息来报告它们。一旦您的错误得到验证，您的提交将被接受，错误将被上传到我们的网站或添加到该标题的错误清单部分。

查看之前提交的错误清单，请访问 [`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**错误清单**部分。

## 盗版

在互联网上盗版受版权保护的材料是所有媒体中持续存在的问题。在 Packt，我们非常重视保护我们的版权和许可证。如果您在互联网上发现任何形式的非法复制我们的作品，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 copyright@packtpub.com 联系我们，并提供指向疑似盗版材料的链接。

我们感谢您的帮助，以保护我们的作者和我们为您提供有价值内容的能力。

## 咨询

如果您对本书的任何方面有问题，您可以通过 questions@packtpub.com 联系我们，我们将尽力解决问题。
