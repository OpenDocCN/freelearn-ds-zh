# 前言

数据持续增长，加上对这些数据进行越来越复杂的决策的需求，正在创造巨大的障碍，阻止组织利用传统的分析方法及时获取洞察力。大数据领域与这些框架密切相关，其范围由这些框架能处理的内容来定义。无论您是在审查数百万访问者的点击流以优化在线广告位置，还是在筛选数十亿交易以识别欺诈迹象，对于从海量数据中自动获取洞察力的高级分析（如机器学习和图处理）的需求比以往任何时候都更加明显。

Apache Spark，作为大数据处理、分析和数据科学在所有学术界和行业中的事实标准，提供了机器学习和图处理库，使公司能够轻松应对复杂问题，利用高度可扩展和集群化的计算机的强大能力。Spark 的承诺是进一步推动使用 Scala 编写分布式程序感觉像为 Spark 编写常规程序。Spark 将在提高 ETL 管道性能和减轻一些痛苦方面做得很好，这些痛苦来自 MapReduce 程序员每天对 Hadoop 神明的绝望呼唤。

在本书中，我们使用 Spark 和 Scala 进行努力，将最先进的高级数据分析与机器学习、图处理、流处理和 SQL 引入 Spark，并将它们贡献给 MLlib、ML、SQL、GraphX 和其他库。

我们从 Scala 开始，然后转向 Spark 部分，最后，涵盖了一些关于使用 Spark 和 Scala 进行大数据分析的高级主题。在附录中，我们将看到如何扩展您的 Scala 知识，以用于 SparkR、PySpark、Apache Zeppelin 和内存中的 Alluxio。本书不是要从头到尾阅读的。跳到一个看起来像您要完成的任务或简单激起您兴趣的章节。

祝阅读愉快！

# 本书内容

第一章，*Scala 简介*，将教授使用基于 Scala 的 Spark API 进行大数据分析。Spark 本身是用 Scala 编写的，因此作为起点，我们将讨论 Scala 的简要介绍，例如其历史、目的以及如何在 Windows、Linux 和 Mac OS 上安装 Scala。之后，将简要讨论 Scala web 框架。然后，我们将对 Java 和 Scala 进行比较分析。最后，我们将深入 Scala 编程，开始使用 Scala。

第二章，*面向对象的 Scala*，说道面向对象编程（OOP）范式提供了全新的抽象层。简而言之，本章讨论了面向对象编程语言的一些最大优势：可发现性、模块化和可扩展性。特别是，我们将看到如何处理 Scala 中的变量；Scala 中的方法、类和对象；包和包对象；特征和特征线性化；以及 Java 互操作性。

第三章，*函数式编程概念*，展示了 Scala 中的函数式编程概念。更具体地，我们将学习几个主题，比如为什么 Scala 是数据科学家的武器库，为什么学习 Spark 范式很重要，纯函数和高阶函数（HOFs）。还将展示使用 HOFs 的实际用例。然后，我们将看到如何在 Scala 的标准库中处理高阶函数在集合之外的异常。最后，我们将看看函数式 Scala 如何影响对象的可变性。

第四章《集合 API》介绍了吸引大多数 Scala 用户的功能之一——集合 API。它非常强大和灵活，并且具有许多相关操作。我们还将演示 Scala 集合 API 的功能以及如何使用它来适应不同类型的数据并解决各种不同的问题。在本章中，我们将涵盖 Scala 集合 API、类型和层次结构、一些性能特征、Java 互操作性以及 Scala 隐式。

第五章《应对大数据 - Spark 加入派对》概述了数据分析和大数据；我们看到大数据带来的挑战，以及它们是如何通过分布式计算来处理的，以及函数式编程提出的方法。我们介绍了谷歌的 MapReduce、Apache Hadoop，最后是 Apache Spark，并看到它们是如何采纳这种方法和这些技术的。我们将探讨 Apache Spark 的演变：为什么首先创建了 Apache Spark 以及它如何为大数据分析和处理的挑战带来价值。

第六章《开始使用 Spark - REPL 和 RDDs》涵盖了 Spark 的工作原理；然后，我们介绍了 RDDs，这是 Apache Spark 背后的基本抽象，看到它们只是暴露类似 Scala 的 API 的分布式集合。我们将研究 Apache Spark 的部署选项，并在本地运行它作为 Spark shell。我们将学习 Apache Spark 的内部工作原理，RDD 是什么，RDD 的 DAG 和谱系，转换和操作。

第七章《特殊 RDD 操作》着重介绍了如何定制 RDD 以满足不同的需求，以及这些 RDD 提供了新的功能（和危险！）此外，我们还研究了 Spark 提供的其他有用对象，如广播变量和累加器。我们将学习聚合技术、洗牌。

第八章《引入一点结构 - SparkSQL》教您如何使用 Spark 分析结构化数据，作为 RDD 的高级抽象，以及 Spark SQL 的 API 如何使查询结构化数据变得简单而健壮。此外，我们介绍数据集，并查看数据集、数据框架和 RDD 之间的区别。我们还将学习使用数据框架 API 进行复杂数据分析的连接操作和窗口函数。

第九章《带我上流 - Spark Streaming》带您了解 Spark Streaming 以及我们如何利用它来使用 Spark API 处理数据流。此外，在本章中，读者将学习使用实际示例处理实时数据流的各种方法，以消费和处理来自 Twitter 的推文。我们将研究与 Apache Kafka 的集成以进行实时处理。我们还将研究结构化流，它可以为您的应用程序提供实时查询。

第十章《一切都相连 - GraphX》中，我们将学习许多现实世界的问题可以使用图来建模（和解决）。我们将以 Facebook 为例看图论，Apache Spark 的图处理库 GraphX，VertexRDD 和 EdgeRDDs，图操作符，aggregateMessages，TriangleCounting，Pregel API 以及 PageRank 算法等用例。

第十一章，“学习机器学习-Spark MLlib 和 ML”，本章的目的是提供统计机器学习的概念介绍。我们将重点介绍 Spark 的机器学习 API，称为 Spark MLlib 和 ML。然后我们将讨论如何使用决策树和随机森林算法解决分类任务，以及使用线性回归算法解决回归问题。我们还将展示在训练分类模型之前如何从使用独热编码和降维算法在特征提取中受益。在后面的部分，我们将逐步展示开发基于协同过滤的电影推荐系统的示例。

第十二章，“高级机器学习最佳实践”，提供了一些关于使用 Spark 进行机器学习的高级主题的理论和实践方面。我们将看到如何使用网格搜索、交叉验证和超参数调整来调整机器学习模型以获得最佳性能。在后面的部分，我们将介绍如何使用 ALS 开发可扩展的推荐系统，这是一个基于模型的推荐算法的示例。最后，将演示主题建模应用作为文本聚类技术。

第十三章，“我的名字是贝叶斯，朴素贝叶斯”，指出大数据中的机器学习是一个革命性的组合，对学术界和工业界的研究领域产生了巨大影响。大数据对机器学习、数据分析工具和算法提出了巨大挑战，以找到真正的价值。然而，基于这些庞大数据集进行未来预测从未容易。考虑到这一挑战，在本章中，我们将深入探讨机器学习，了解如何使用简单而强大的方法构建可扩展的分类模型，以及多项式分类、贝叶斯推断、朴素贝叶斯、决策树和朴素贝叶斯与决策树的比较分析等概念。

第十四章，“整理数据的时候到了-Spark MLlib 对数据进行聚类”，让您了解 Spark 在集群模式下的工作原理及其基础架构。在之前的章节中，我们看到了如何使用不同的 Spark API 开发实际应用程序。最后，我们将看到如何在集群上部署完整的 Spark 应用程序，无论是使用现有的 Hadoop 安装还是不使用。

第十五章，“使用 Spark ML 进行文本分析”，概述了使用 Spark ML 进行文本分析的广泛领域。文本分析是机器学习中的一个广泛领域，在许多用例中非常有用，例如情感分析、聊天机器人、电子邮件垃圾邮件检测、自然语言处理等。我们将学习如何使用 Spark 进行文本分析，重点关注使用包含 1 万个样本的 Twitter 数据集进行文本分类的用例。我们还将研究 LDA，这是一种从文档中生成主题的流行技术，而不需要了解实际文本内容，并将在 Twitter 数据上实现文本分类，以了解所有内容是如何结合在一起的。

第十六章，“Spark 调优”，深入挖掘 Apache Spark 内部，并表示虽然 Spark 在让我们感觉好像只是使用另一个 Scala 集合方面做得很好，但我们不应忘记 Spark 实际上是在分布式系统中运行。因此，在本章中，我们将介绍如何监视 Spark 作业、Spark 配置、Spark 应用程序开发中的常见错误以及一些优化技术。

第十七章，*去集群之旅-在集群上部署 Spark*，探讨了 Spark 在集群模式下的工作方式及其基础架构。我们将看到集群中的 Spark 架构，Spark 生态系统和集群管理，以及如何在独立、Mesos、Yarn 和 AWS 集群上部署 Spark。我们还将看到如何在基于云的 AWS 集群上部署您的应用程序。

第十八章，*测试和调试 Spark*，解释了在分布式环境中测试应用程序有多么困难；然后，我们将看到一些解决方法。我们将介绍如何在分布式环境中进行测试，以及测试和调试 Spark 应用程序。

第十九章，*PySpark 和 SparkR*，涵盖了使用 R 和 Python 编写 Spark 代码的另外两种流行 API，即 PySpark 和 SparkR。特别是，我们将介绍如何开始使用 PySpark 并与 PySpark 交互 DataFrame API 和 UDF，然后我们将使用 PySpark 进行一些数据分析。本章的第二部分涵盖了如何开始使用 SparkR。我们还将看到如何进行数据处理和操作，以及如何使用 SparkR 处理 RDD 和 DataFrames，最后，使用 SparkR 进行一些数据可视化。

附录 A，*使用 Alluxio 加速 Spark*，展示了如何使用 Alluxio 与 Spark 来提高处理速度。Alluxio 是一个开源的分布式内存存储系统，可用于提高跨平台的许多应用程序的速度，包括 Apache Spark。我们将探讨使用 Alluxio 的可能性以及 Alluxio 集成如何在运行 Spark 作业时提供更高的性能而无需每次都将数据缓存到内存中。

附录 B，*使用 Apache Zeppelin 进行交互式数据分析*，从数据科学的角度来看，交互式可视化数据分析也很重要。Apache Zeppelin 是一个基于 Web 的笔记本，用于具有多个后端和解释器的交互式和大规模数据分析。在本章中，我们将讨论如何使用 Apache Zeppelin 进行大规模数据分析，使用 Spark 作为后端的解释器。

# 本书所需的内容

所有示例都是在 Ubuntu Linux 64 位上使用 Python 版本 2.7 和 3.5 实现的，包括 TensorFlow 库版本 1.0.1。然而，在本书中，我们只展示了与 Python 2.7 兼容的源代码。与 Python 3.5+兼容的源代码可以从 Packt 存储库下载。您还需要以下 Python 模块（最好是最新版本）：

+   Spark 2.0.0（或更高）

+   Hadoop 2.7（或更高）

+   Java（JDK 和 JRE）1.7+/1.8+

+   Scala 2.11.x（或更高）

+   Python 2.7+/3.4+

+   R 3.1+和 RStudio 1.0.143（或更高）

+   Eclipse Mars，Oxygen 或 Luna（最新）

+   Maven Eclipse 插件（2.9 或更高）

+   Eclipse 的 Maven 编译器插件（2.3.2 或更高）

+   Eclipse 的 Maven 汇编插件（2.4.1 或更高）

**操作系统：**首选 Linux 发行版（包括 Debian，Ubuntu，Fedora，RHEL 和 CentOS），更具体地说，对于 Ubuntu，建议安装完整的 14.04（LTS）64 位（或更高版本），VMWare player 12 或 Virtual box。您可以在 Windows（XP/7/8/10）或 Mac OS X（10.4.7+）上运行 Spark 作业。

**硬件配置：**处理器 Core i3，Core i5（推荐）或 Core i7（以获得最佳结果）。然而，多核处理将提供更快的数据处理和可伸缩性。您至少需要 8-16 GB RAM（推荐）以独立模式运行，至少需要 32 GB RAM 以单个 VM 运行-并且对于集群来说需要更高。您还需要足够的存储空间来运行繁重的作业（取决于您处理的数据集大小），最好至少有 50 GB 的免费磁盘存储空间（用于独立的单词丢失和 SQL 仓库）。

# 这本书适合谁

任何希望通过利用 Spark 的力量来学习数据分析的人都会发现这本书非常有用。我们不假设您具有 Spark 或 Scala 的知识，尽管先前的编程经验（特别是使用其他 JVM 语言）将有助于更快地掌握这些概念。在过去几年中，Scala 的采用率一直在稳步上升，特别是在数据科学和分析领域。与 Scala 齐头并进的是 Apache Spark，它是用 Scala 编程的，并且在分析领域被广泛使用。本书将帮助您利用这两种工具的力量来理解大数据。

# 约定

在本书中，您将找到一些区分不同信息类型的文本样式。以下是一些这些样式的示例及其含义的解释。文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“下一行代码读取链接并将其分配给`BeautifulSoup`函数。”

代码块设置如下：

```scala
package com.chapter11.SparkMachineLearning
import org.apache.spark.mllib.feature.StandardScalerModel
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.sql.{ DataFrame }
import org.apache.spark.sql.SparkSession

```

当我们希望引起您对代码块的特定部分的注意时，相关的行或项目将以粗体显示：

```scala
val spark = SparkSession
                 .builder
                 .master("local[*]")
                 .config("spark.sql.warehouse.dir", "E:/Exp/")
                 .config("spark.kryoserializer.buffer.max", "1024m")
                 .appName("OneVsRestExample")        
           .getOrCreate()

```

任何命令行输入或输出都以以下方式编写：

```scala
$./bin/spark-submit --class com.chapter11.RandomForestDemo \
--master spark://ip-172-31-21-153.us-west-2.compute:7077 \
--executor-memory 2G \
--total-executor-cores 2 \
file:///home/KMeans-0.0.1-SNAPSHOT.jar \
file:///home/mnist.bz2

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如菜单或对话框中的单词，会以这种方式出现在文本中：“单击“下一步”按钮将您移至下一个屏幕。”

警告或重要说明会以这种方式出现。

提示和技巧会以这种方式出现。
