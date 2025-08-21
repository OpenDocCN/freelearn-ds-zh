# 前言

Apache Spark 是一个基于内存的集群数据处理系统，提供广泛的功能，如大数据处理、分析、机器学习等。通过这个学习路径，您可以将 Apache Spark 的知识提升到一个新的水平，学习如何扩展 Spark 的功能，并在此平台上构建自己的数据流和机器学习程序。您将使用 Apache Spark 的不同模块，如使用 Spark SQL 进行交互式查询、使用 DataFrames 和数据集、使用 Spark Streaming 实现流分析，以及使用 MLlib 和各种外部工具在 Spark 上应用机器学习和深度学习技术。通过这个精心设计的学习...

# 本书面向的读者

如果您是一名中级 Spark 开发者，希望掌握 Apache Spark 2.x 的高级功能和用例，这个学习路径非常适合您。希望学习如何集成和使用 Apache Spark 功能并构建强大大数据管道的大数据专业人士也会发现这个学习路径很有用。要理解本学习路径中解释的概念，您必须了解 Apache Spark 和 Scala 的基础知识。

# 本书内容

*第一章*，*Apache Spark V2 初体验及新特性*，概述了 Apache Spark，介绍了其模块内的功能，以及如何进行扩展。它涵盖了 Apache Spark 标准模块之外的生态系统中可用的处理和存储工具。还提供了性能调优的技巧。

*第二章*，*Apache Spark 流处理*，讲述了使用 Apache Spark Streaming 的连续应用程序。您将学习如何增量处理数据并创建可行的见解。

*第三章*，*结构化流处理*，讲述了使用 DataFrame 和 Dataset API 定义连续应用程序的新方式——结构化流处理。

*第四章*，*Apache Spark MLlib*，介绍了...

# 充分利用本书

**操作系统：** 首选 Linux 发行版（包括 Debian、Ubuntu、Fedora、RHEL 和 CentOS），具体来说，推荐使用完整的 Ubuntu 14.04（LTS）64 位（或更高版本）安装，VMware player 12 或 VirtualBox。您也可以在 Windows（XP/7/8/10）或 Mac OS X（10.4.7+）上运行 Spark 作业。

**硬件配置：** 处理器建议使用 Core i3、Core i5（推荐）或 Core i7（以获得最佳效果）。然而，多核处理将提供更快的数据处理和可扩展性。对于独立模式，您至少需要 8-16 GB RAM（推荐），对于单个虚拟机至少需要 32 GB RAM——集群模式则需要更多。您还需要足够的存储空间来运行繁重的作业（取决于您将处理的数据集大小），并且最好至少有 50 GB 的可用磁盘存储空间（对于独立模式和 SQL 仓库）。

此外，您还需要以下内容：

+   VirtualBox 5.1.22 或更高版本

+   Hortonworks HDP Sandbox V2.6 或更高版本

+   Eclipse Neon 或更高版本

+   Eclipse Scala 插件

+   Eclipse Git 插件

+   Spark 2.0.0（或更高版本）

+   Hadoop 2.7（或更高版本）

+   Java（JDK 和 JRE）1.7+/1.8+

+   Scala 2.11.x（或更高版本）

+   Python 2.7+/3.4+

+   R 3.1+ 和 RStudio 1.0.143（或更高版本）

+   Maven Eclipse 插件（2.9 或更高版本）

+   Maven 编译器插件 for Eclipse（2.3.2 或更高版本）

+   Maven 装配插件 for Eclipse（2.4.1 或更高版本）

+   Oracle JDK SE 1.8.x

+   JetBrain IntelliJ 社区版 2016.2.X 或更高版本

+   IntelliJ 的 Scala 插件 2016.2.x

+   Jfreechart 1.0.19

+   breeze-core 0.12

+   Cloud9 1.5.0 JAR

+   Bliki-core 3.0.19

+   hadoop-streaming 2.2.0

+   Jcommon 1.0.23

+   Lucene-analyzers-common 6.0.0

+   Lucene-core-6.0.0

+   Spark-streaming-flume-assembly 2.0.0

+   Spark-streaming-kafka-assembly 2.0.0

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册于[www.packt.com](http://www.packt.com)。

1.  选择支持选项卡。

1.  点击代码下载与勘误。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

下载文件后，请确保使用最新版本的以下软件解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

[本书代码包](https://github.com/PacktPublishing/Apache-Spark-2-Data-Processing-and-Real-Time-Analytics)也托管在 GitHub 上。

# 使用的约定

本书中，您会发现多种文本样式用于区分不同类型的信息。以下是这些样式的示例及其含义的解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“接下来的代码行读取链接并将其分配给`BeautifulSoup`函数。”

代码块设置如下：

```scala
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
```

任何命令行输入或输出如下所示：

```scala
$./bin/spark-submit --class com.chapter11.RandomForestDemo \
--master spark://ip-172-31-21-153.us-west-2.compute:7077 \
--executor-memory 2G \
--total-executor-cores 2 \
file:///home/KMeans-0.0.1-SNAPSHOT.jar \
file:///home/mnist.bz2
```

**粗体**：新术语和重要词汇以粗体显示。屏幕上看到的词汇，例如在菜单或对话框中，在文本中这样显示：“配置全局库。选择 Scala SDK 作为您的全局库。”

警告或重要提示以这种方式出现。

提示和技巧以这种方式出现。
