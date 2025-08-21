# 前言

已经写了一本关于 Hadoop 生态系统的介绍性书籍，我很高兴 Packt 邀请我写一本关于 Apache Spark 的书。作为一个有支持和维护背景的实用主义者，我对系统构建和集成很感兴趣。因此，我总是问自己“系统如何被使用？”，“它们如何相互配合？”，“它们与什么集成？”在本书中，我将描述 Spark 的每个模块，并通过实际例子解释它们如何被使用。我还将展示如何通过额外的库（如来自[`h2o.ai/`](http://h2o.ai/)的 H2O）扩展 Spark 的功能。

我将展示 Apache Spark 的图处理模块如何与 Aurelius（现在是 DataStax）的 Titan 图数据库一起使用。这将通过将 Spark GraphX 和 Titan 组合在一起，提供基于图的处理和存储的耦合。流处理章节将展示如何使用 Apache Flume 和 Kafka 等工具将数据传递给 Spark 流。

考虑到过去几年已经有大规模迁移到基于云的服务，我将检查[`databricks.com/`](https://databricks.com/)提供的 Spark 云服务。我将从实际的角度来做，本书不试图回答“服务器还是云”的问题，因为我认为这是另一本书的主题；它只是检查了可用的服务。

# 本书涵盖的内容

第一章 *Apache Spark*，将全面介绍 Spark，其模块的功能以及用于处理和存储的工具。本章将简要介绍 SQL、流处理、GraphX、MLlib、Databricks 和 Hive on Spark 的细节。

第二章 *Apache Spark MLlib*，涵盖了 MLlib 模块，其中 MLlib 代表机器学习库。它描述了本书中将使用的 Apache Hadoop 和 Spark 集群，以及涉及的操作系统——CentOS。它还描述了正在使用的开发环境：Scala 和 SBT。它提供了安装和构建 Apache Spark 的示例。解释了使用朴素贝叶斯算法进行分类的示例，以及使用 KMeans 进行聚类的示例。最后，使用 Bert Greevenbosch（[www.bertgreevenbosch.nl](http://www.bertgreevenbosch.nl)）的工作扩展 Spark 以包括一些人工神经网络（ANN）工作的示例。我一直对神经网络很感兴趣，能够在本章中使用 Bert 的工作（在得到他的许可后）是一件令人愉快的事情。因此，本章的最后一个主题是使用简单的 ANN 对一些小图像进行分类，包括扭曲的图像。结果和得分都相当不错！

第三章 *Apache Spark Streaming*，涵盖了 Apache Spark 与 Storm 的比较，特别是 Spark Streaming，但我认为 Spark 提供了更多的功能。例如，一个 Spark 模块中使用的数据可以传递到另一个模块中并被使用。此外，正如本章所示，Spark 流处理可以轻松集成大数据移动技术，如 Flume 和 Kafka。

因此，流处理章节首先概述了检查点，并解释了何时可能需要使用它。它给出了 Scala 代码示例，说明了如何使用它，并展示了数据如何存储在 HDFS 上。然后，它继续给出了 Scala 的实际示例，以及 TCP、文件、Flume 和 Kafka 流处理的执行示例。最后两个选项通过处理 RSS 数据流并最终将其存储在 HDFS 上来展示。

第四章 *Apache Spark SQL*，用 Scala 代码术语解释了 Spark SQL 上下文。它解释了文本、Parquet 和 JSON 格式的文件 I/O。使用 Apache Spark 1.3，它通过示例解释了数据框架的使用，并展示了它们提供的数据分析方法。它还通过基于 Scala 的示例介绍了 Spark SQL，展示了如何创建临时表，以及如何对其进行 SQL 操作。

接下来，介绍了 Hive 上下文。首先创建了一个本地上下文，然后执行了 Hive QL 操作。然后，介绍了一种方法，将现有的分布式 CDH 5.3 Hive 安装集成到 Spark Hive 上下文中。然后展示了针对此上下文的操作，以更新集群上的 Hive 数据库。通过这种方式，可以创建和调度 Spark 应用程序，以便 Hive 操作由实时 Spark 引擎驱动。

最后，介绍了创建用户定义函数（UDFs），然后使用创建的 UDFs 对临时表进行 SQL 调用。

第五章 *Apache Spark GraphX*，介绍了 Apache Spark GraphX 模块和图形处理模块。它通过一系列基于示例的图形函数工作，从基于计数到三角形处理。然后介绍了 Kenny Bastani 的 Mazerunner 工作，该工作将 Neo4j NoSQL 数据库与 Apache Spark 集成。这项工作已经得到 Kenny 的许可；请访问[www.kennybastani.com](http://www.kennybastani.com)。

本章通过 Docker 的介绍，然后是 Neo4j，然后介绍了 Neo4j 接口。最后，通过提供的 REST 接口介绍了一些 Mazerunner 提供的功能。

第六章 *基于图形的存储*，检查了基于图形的存储，因为本书介绍了 Apache Spark 图形处理。我寻找一个能够与 Hadoop 集成、开源、能够高度扩展，并且能够与 Apache Spark 集成的产品。

尽管在社区支持和开发方面仍然相对年轻，但我认为 Aurelius（现在是 DataStax）的 Titan 符合要求。截至我写作时，可用的 0.9.x 版本使用 Apache TinkerPop 进行图形处理。

本章提供了使用 Gremlin shell 和 Titan 创建和存储图形的示例。它展示了如何将 HBase 和 Cassandra 用于后端 Titan 存储。

第七章 *使用 H2O 扩展 Spark*，讨论了在[`h2o.ai/`](http://h2o.ai/)开发的 H2O 库集，这是一个可以用来扩展 Apache Spark 功能的机器学习库系统。在本章中，我研究了 H2O 的获取和安装，以及用于数据分析的 Flow 接口。还研究了 Sparkling Water 的架构、数据质量和性能调优。

最后，创建并执行了一个深度学习的示例。第二章 *Spark MLlib*，使用简单的人工神经网络进行神经分类。本章使用了一个高度可配置和可调整的 H2O 深度学习神经网络进行分类。结果是一个快速而准确的训练好的神经模型，你会看到的。

第八章 *Spark Databricks*，介绍了[`databricks.com/`](https://databricks.com/) AWS 基于云的 Apache Spark 集群系统。它提供了逐步设置 AWS 账户和 Databricks 账户的过程。然后，它逐步介绍了[`databricks.com/`](https://databricks.com/)账户功能，包括笔记本、文件夹、作业、库、开发环境等。

它检查了 Databricks 中基于表的存储和处理，并介绍了 Databricks 实用程序功能的 DBUtils 包。这一切都是通过示例完成的，以便让您对这个基于云的系统的使用有一个很好的理解。

第九章，*Databricks 可视化*，通过专注于数据可视化和仪表板来扩展 Databricks 的覆盖范围。然后，它检查了 Databricks 的 REST 接口，展示了如何使用各种示例 REST API 调用远程管理集群。最后，它从表的文件夹和库的角度看数据移动。

本章的集群管理部分显示，可以使用 Spark 发布的脚本在 AWS EC2 上启动 Apache Spark。[`databricks.com/`](https://databricks.com/)服务通过提供一种轻松创建和调整多个基于 EC2 的 Spark 集群的方法，进一步提供了这种功能。它为集群管理和使用提供了额外的功能，以及用户访问和安全性，正如这两章所示。考虑到为我们带来 Apache Spark 的人们创建了这项服务，它一定值得考虑和审查。

# 本书所需内容

本书中的实际示例使用 Scala 和 SBT 进行基于 Apache Spark 的代码开发和编译。还使用了基于 CentOS 6.5 Linux 服务器的 Cloudera CDH 5.3 Hadoop 集群。Linux Bash shell 和 Perl 脚本都用于帮助 Spark 应用程序并提供数据源。在 Spark 应用程序测试期间，使用 Hadoop 管理命令来移动和检查数据。

考虑到之前的技能概述，读者对 Linux、Apache Hadoop 和 Spark 有基本的了解会很有帮助。话虽如此，鉴于今天互联网上有大量信息可供查阅，我不想阻止一个勇敢的读者去尝试。我相信从错误中学到的东西可能比成功更有价值。

# 这本书是为谁准备的

这本书适用于任何对 Apache Hadoop 和 Spark 感兴趣的人，他们想了解更多关于 Spark 的知识。它适用于希望了解如何使用 Spark 扩展 H2O 等系统的用户。对于对图处理感兴趣但想了解更多关于图存储的用户。如果读者想了解云中的 Apache Spark，那么他/她可以了解由为他们带来 Spark 的人开发的[`databricks.com/`](https://databricks.com/)。如果您是具有一定 Spark 经验的开发人员，并希望加强对 Spark 世界的了解，那么这本书非常适合您。要理解本书，需要具备 Linux、Hadoop 和 Spark 的基本知识；同时也需要合理的 Scala 知识。

# 约定

在本书中，您会发现一些文本样式，用于区分不同类型的信息。以下是这些样式的一些示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“第一步是确保`/etc/yum.repos.d`目录下存在 Cloudera 存储库文件，在服务器 hc2nn 和所有其他 Hadoop 集群服务器上。”

代码块设置如下：

```scala
export AWS_ACCESS_KEY_ID="QQpl8Exxx"
export AWS_SECRET_ACCESS_KEY="0HFzqt4xxx"

./spark-ec2  \
    --key-pair=pairname \
    --identity-file=awskey.pem \
    --region=us-west-1 \
    --zone=us-west-1a  \
    launch cluster1
```

任何命令行输入或输出都是这样写的：

```scala
[hadoop@hc2nn ec2]$ pwd

/usr/local/spark/ec2

[hadoop@hc2nn ec2]$ ls
deploy.generic  README  spark-ec2  spark_ec2.py

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如菜单或对话框中的单词，会以这样的方式出现在文本中：“选择**用户操作**选项，然后选择**管理访问密钥**。”

### 注意

警告或重要说明会以这样的方式出现在框中。

### 提示

提示和技巧会以这样的方式出现。
