# 前言

随着大型计算系统的出现，不同领域的组织以实时方式生成大量数据。作为大数据处理的最新参与者，Apache Flink 旨在以极快的速度处理连续的数据流。

这本书将成为您使用 Apache Flink 进行批处理和流数据处理的权威指南。该书首先介绍了 Apache Flink 生态系统，设置它并使用 DataSet 和 DataStream API 处理批处理和流式数据集。随后，本书将探讨如何将 SQL 的强大功能引入 Flink，并探索用于查询和操作数据的 Table API。在书的后半部分，读者将学习 Apache Flink 的其余生态系统，以实现事件处理、机器学习和图处理等复杂任务。该书的最后部分将包括诸如扩展 Flink 解决方案、性能优化和将 Flink 与 Hadoop、ElasticSearch、Cassandra 和 Kafka 等其他工具集成的主题。

无论您是想深入了解 Apache Flink，还是想探索如何更好地利用这一强大技术，您都会在本书中找到一切。本书涵盖了许多真实世界的用例，这将帮助您串联起各个方面。

# 本书涵盖的内容

第一章，“介绍 Apache Flink”，向您介绍了 Apache Flink 的历史、架构、特性和在单节点和多节点集群上的安装。

第二章，“使用 DataStream API 进行数据处理”，为您提供了有关 Flink 流优先概念的详细信息。您将了解有关 DataStream API 提供的数据源、转换和数据接收器的详细信息。

第三章，“使用批处理 API 进行数据处理”，为您介绍了批处理 API，即 DataSet API。您将了解有关数据源、转换和接收器的信息。您还将了解 API 提供的连接器。

第四章，“使用 Table API 进行数据处理”，帮助您了解如何将 SQL 概念与 Flink 数据处理框架相结合。您还将学习如何将这些概念应用于实际用例。

第五章，“复杂事件处理”，为您提供了如何使用 Flink CEP 库解决复杂事件处理问题的见解。您将了解有关模式定义、检测和警报生成的详细信息。

第六章，“使用 FlinkML 进行机器学习”，详细介绍了机器学习概念以及如何将各种算法应用于实际用例。

第七章，“Flink 图形 API - Gelly”，向您介绍了图形概念以及 Flink Gelly 为我们解决实际用例提供的功能。它向您介绍了 Flink 提供的迭代图处理能力。

第八章，“使用 Flink 和 Hadoop 进行分布式数据处理”，详细介绍了如何使用现有的 Hadoop-YARN 集群提交 Flink 作业。它详细介绍了 Flink 在 YARN 上的工作原理。

第九章，“在云上部署 Flink”，提供了有关如何在云上部署 Flink 的详细信息。它详细介绍了如何在 Google Cloud 和 AWS 上使用 Flink。

第十章，“最佳实践”，涵盖了开发人员应遵循的各种最佳实践，以便以高效的方式使用 Flink。它还讨论了日志记录、监控最佳实践以控制 Flink 环境。

# 您需要为本书准备什么

您需要一台带有 Windows、Mac 或 UNIX 等任何操作系统的笔记本电脑或台式电脑。最好有一个诸如 Eclipse 或 IntelliJ 的 IDE，当然，您需要很多热情。

# 这本书是为谁准备的

这本书适用于希望在分布式系统上处理批处理和实时数据的大数据开发人员，以及寻求工业化分析解决方案的数据科学家。

# 惯例

在本书中，您会发现一些区分不同信息类型的文本样式。以下是一些这些样式的示例及其含义的解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下: "这将在`/flinkuser/.ssh`文件夹中生成公钥和私钥。"

代码块设置如下:

```java
CassandraSink.addSink(input)
  .setQuery("INSERT INTO cep.events (id, message) values (?, ?);")
  .setClusterBuilder(new ClusterBuilder() {
    @Override
    public Cluster buildCluster(Cluster.Builder builder) {
      return builder.addContactPoint("127.0.0.1").build();
    }
  })
  .build();
```

任何命令行输入或输出都以以下方式编写:

```java
$sudo tar -xzf flink-1.1.4-bin-hadoop27-scala_2.11.tgz 
$cd flink-1.1.4 
$bin/start-local.sh

```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如菜单或对话框中的单词，会在文本中显示为: "一旦我们的所有工作都完成了，关闭集群就变得很重要。为此，我们需要再次转到 AWS 控制台，然后单击**终止**按钮"。

### 注意

警告或重要说明会出现在这样的框中。

### 提示

提示和技巧会显示如此。
