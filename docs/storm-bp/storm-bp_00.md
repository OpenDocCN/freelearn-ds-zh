# 前言

对及时可行的信息的需求正在推动软件系统在更短的时间内处理越来越多的数据。此外，随着连接设备数量的增加，以及这些设备应用于越来越广泛的行业，这种需求变得越来越普遍。传统的企业运营系统被迫处理最初只与互联网规模公司相关的数据规模。这一巨大的转变迫使更传统的架构和方法崩溃，这些架构和方法曾将在线交易系统和离线分析分开。相反，人们正在重新想象从数据中提取信息的含义。框架和基础设施也在发展以适应这一新愿景。

具体来说，数据生成现在被视为一系列离散事件。这些事件流与数据流相关，一些是操作性的，一些是分析性的，但由一个共同的框架和基础设施处理。

风暴是实时流处理最流行的框架。它提供了在高容量、关键任务应用中所需的基本原语和保证。它既是集成技术，也是数据流和控制机制。许多大公司都将风暴作为其大数据平台的支柱。

使用本书的设计模式，您将学会开发、部署和操作能够处理数十亿次交易的数据处理流。

《风暴蓝图：分布式实时计算模式》涵盖了广泛的分布式计算主题，不仅包括设计和集成模式，还包括技术立即有用和常用的领域和应用。本书通过真实世界的例子向读者介绍了风暴，从简单的风暴拓扑开始。示例逐渐复杂，引入了高级风暴概念以及更复杂的部署和运营问题。

# 本书涵盖的内容

第一章，“分布式词频统计”，介绍了使用风暴进行分布式流处理的核心概念。分布式词频统计示例演示了更复杂计算所需的许多结构、技术和模式。在本章中，我们将对风暴计算结构有基本的了解。我们将建立开发环境，并了解用于调试和开发风暴应用的技术。

第二章，“配置风暴集群”，深入探讨了风暴技术栈以及设置和部署到风暴集群的过程。在本章中，我们将使用 Puppet provisioning 工具自动化安装和配置多节点集群。

第三章，“Trident 拓扑和传感器数据”，涵盖了 Trident 拓扑。Trident 在风暴之上提供了更高级的抽象，抽象了事务处理和状态管理的细节。在本章中，我们将应用 Trident 框架来处理、聚合和过滤传感器数据以检测疾病爆发。

第四章，“实时趋势分析”，介绍了使用风暴和 Trident 的趋势分析技术。实时趋势分析涉及识别数据流中的模式。在本章中，您将与 Apache Kafka 集成，并实现滑动窗口来计算移动平均值。

第五章，“实时图分析”，涵盖了使用 Storm 进行图分析，将数据持久化到图数据库并查询数据以发现关系。图数据库是将数据存储为图结构的数据库，具有顶点、边和属性，并主要关注实体之间的关系。在本章中，您将使用 Twitter 作为数据源，将 Storm 与流行的图数据库 Titan 整合。

第六章，“人工智能”，将 Storm 应用于通常使用递归实现的人工智能算法。我们揭示了 Storm 的一些局限性，并研究了适应这些局限性的模式。在本章中，使用**分布式远程过程调用**（**DRPC**），您将实现一个 Storm 拓扑，能够为同步查询提供服务，以确定井字游戏中的下一步最佳移动。

第七章，“集成 Druid 进行金融分析”，演示了将 Storm 与非事务系统集成的复杂性。为了支持这样的集成，本章介绍了一种利用 ZooKeeper 管理分布式状态的模式。在本章中，您将把 Storm 与 Druid 整合，Druid 是一个用于探索性分析的开源基础设施，用于提供可配置的实时分析金融事件的系统。

第八章，“自然语言处理”，介绍了 Lambda 架构的概念，将实时和批处理配对，创建一个用于分析的弹性系统。在第七章，“集成 Druid 进行金融分析”的基础上，您将整合 Hadoop 基础设施，并研究 MapReduce 作业，以在主机故障时在 Druid 中回填分析。

第九章，“在 Hadoop 上部署 Storm 进行广告分析”，演示了将现有的在 Hadoop 上运行的 Pig 脚本批处理过程转换为实时 Storm 拓扑的过程。为此，您将利用 Storm-YARN，它允许用户利用 YARN 来部署和运行 Storm 集群。在 Hadoop 上运行 Storm 允许企业 consoliolidate operations and utilize the same infrastructure for both real time and batch processing.

第十章，“云中的 Storm”，涵盖了在云服务提供商托管环境中运行和部署 Storm 的最佳实践。具体来说，您将利用 Apache Whirr，一组用于云服务的库，来部署和配置 Storm 及其支持技术，以在通过**亚马逊网络服务**（**AWS**）**弹性计算云**（**EC2**）提供的基础设施上进行部署。此外，您将利用 Vagrant 创建用于开发和测试的集群环境。

# 您需要本书的什么

以下是本书使用的软件列表：

| 章节编号 | 需要的软件 |
| --- | --- |
| 1 | Storm（0.9.1） |
| 2 | Zookeeper（3.3.5）Java（1.7）Puppet（3.4.3）Hiera（1.3.1） |
| 3 | 三叉戟（通过 Storm 0.9.1） |
| 4 | Kafka（0.7.2）OpenFire（3.9.1） |
| 5 | Twitter4J（3.0.3）Titan（0.3.2）Cassandra（1.2.9） |
| 6 | 没有新软件 |
| 7 | MySQL（5.6.15）Druid（0.5.58） |
| 8 | Hadoop（0.20.2） |
| 9 | Storm-YARN（1.0-alpha）Hadoop（2.1.0-beta） |
| 10 | Whirr（0.8.2）Vagrant（1.4.3） |

# 这本书是为谁准备的

*Storm Blueprints: Patterns for Distributed Real-time Computation*通过描述基于真实示例应用的广泛适用的分布式计算模式，使初学者和高级用户都受益。本书介绍了 Storm 和 Trident 中的核心原语以及成功部署和操作所需的关键技术。

尽管该书主要关注使用 Storm 进行 Java 开发，但这些模式适用于其他语言，书中描述的技巧、技术和方法适用于架构师、开发人员、系统和业务运营。 

对于 Hadoop 爱好者来说，这本书也是对 Storm 的很好介绍。该书演示了这两个系统如何相互补充，并提供了从批处理到实时分析世界的潜在迁移路径。

该书提供了将 Storm 应用于各种问题和行业的示例，这应该可以转化为其他面临处理大型数据集的问题的领域。因此，解决方案架构师和业务分析师将受益于这些章节介绍的高级系统架构和技术。

# 约定

在本书中，您会发现一些文本样式，用于区分不同类型的信息。以下是一些这些样式的示例，以及它们的含义解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“所有 Hadoop 配置文件都位于`$HADOOP_CONF_DIR`中。例如，此示例的三个关键配置文件是：`core-site.xml`、`yarn-site.xml`和`hdfs-site.xml`。”

一块代码设置如下：

```scala
<configuration>
    <property>
        <name>fs.default.name</name>
        <value>hdfs://master:8020</value>
    </property>
</configuration>
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```scala
13/10/09 21:40:10 INFO yarn.StormAMRMClient: Use NMClient to launch supervisors in container.  
13/10/09 21:40:10 INFO impl.ContainerManagementProtocolProxy: Opening proxy : slave05:35847 
13/10/09 21:40:12 INFO yarn.StormAMRMClient: Supervisor log: http://slave05:8042/node/containerlogs/container_1381197763696_0004_01_000002/boneill/supervisor.log 
13/10/09 21:40:14 INFO yarn.MasterServer: HB: Received allocated containers (1) 13/10/09 21:40:14 INFO yarn.MasterServer: HB: Supervisors are to run, so queueing (1) containers... 
13/10/09 21:40:14 INFO yarn.MasterServer: LAUNCHER: Taking container with id (container_1381197763696_0004_01_000004) from the queue. 
13/10/09 21:40:14 INFO yarn.MasterServer: LAUNCHER: Supervisors are to run, so launching container id (container_1381197763696_0004_01_000004) 
13/10/09 21:40:16 INFO yarn.StormAMRMClient: Use NMClient to launch supervisors in container.  13/10/09 21:40:16 INFO impl.ContainerManagementProtocolProxy: Opening proxy : dlwolfpack02.hmsonline.com:35125 
13/10/09 21:40:16 INFO yarn.StormAMRMClient: Supervisor log: http://slave02:8042/node/containerlogs/container_1381197763696_0004_01_000004/boneill/supervisor.log

```

任何命令行输入或输出都是这样写的：

```scala
hadoop fs -mkdir /user/bone/lib/
hadoop fs -copyFromLocal ./lib/storm-0.9.0-wip21.zip /user/bone/lib/

```

**新术语**和**重要单词**以粗体显示。例如，屏幕上看到的单词，菜单或对话框中的单词等，会在文本中以这种方式出现：“在页面顶部的**筛选器**下拉菜单中选择**公共图像**。”

### 注意

警告或重要说明会出现在这样的框中。

### 提示

技巧和窍门看起来像这样。
