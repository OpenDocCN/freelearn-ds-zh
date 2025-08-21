# 前言

实时数据处理不再是少数大公司的奢侈品，而已经成为希望竞争的企业的必需品，而 Apache Storm 是开发实时处理管道的事实标准之一。Storm 的关键特性是它具有水平可扩展性，容错性，并提供了保证的消息处理。Storm 可以解决各种类型的分析问题：机器学习、日志处理、图分析等。

精通 Storm 将作为一本*入门指南*，面向经验不足的开发人员，也是有经验的开发人员实施高级用例的参考。在前两章中，您将学习 Storm 拓扑的基础知识和 Storm 集群的各种组件。在后面的章节中，您将学习如何构建一个可以与各种其他大数据技术进行交互的 Storm 应用程序，以及如何创建事务性拓扑。最后，最后两章涵盖了日志处理和机器学习的案例研究。我们还将介绍如何使用 Storm 调度程序将精细的工作分配给精细的机器。

# 本书涵盖内容

第一章，*实时处理和 Storm 介绍*，介绍了 Storm 及其组件。

第二章，*Storm 部署、拓扑开发和拓扑选项*，涵盖了将 Storm 部署到集群中，在 Storm 集群上部署示例拓扑，以及如何使用 Storm UI 监视 storm 管道以及如何动态更改日志级别设置。

第三章，*Storm 并行性和数据分区*，涵盖了拓扑的并行性，如何在代码级别配置并行性，保证消息处理以及 Storm 内部生成的元组。

第四章，*Trident 介绍*，介绍了 Trident 的概述，对 Trident 数据模型的理解，以及如何编写 Trident 过滤器和函数。本章还涵盖了 Trident 元组上的重新分区和聚合操作。

第五章，*Trident 拓扑和用途*，介绍了 Trident 元组分组、非事务性拓扑和一个示例 Trident 拓扑。该章还介绍了 Trident 状态和分布式 RPC。

第六章，*Storm 调度程序*，介绍了 Storm 中可用的不同类型的调度程序：默认调度程序、隔离调度程序、资源感知调度程序和自定义调度程序。

第七章，*Storm 集群的监控*，涵盖了通过编写使用 Nimbus 发布的统计信息的自定义监控 UI 来监控 Storm。我们解释了如何使用 JMXTrans 将 Ganglia 与 Storm 集成。本章还介绍了如何配置 Storm 以发布 JMX 指标。

第八章，*Storm 和 Kafka 的集成*，展示了 Storm 与 Kafka 的集成。本章从 Kafka 的介绍开始，涵盖了 Storm 的安装，并以 Storm 与 Kafka 的集成来解决任何实际问题。

第九章，*Storm 和 Hadoop 集成*，概述了 Hadoop，编写 Storm 拓扑以将数据发布到 HDFS，Storm-YARN 的概述，以及在 YARN 上部署 Storm 拓扑。

第十章，*Storm 与 Redis、Elasticsearch 和 HBase 集成*，教您如何将 Storm 与各种其他大数据技术集成。

第十一章，*使用 Storm 进行 Apache 日志处理*，介绍了一个示例日志处理应用程序，其中我们解析 Apache Web 服务器日志并从日志文件中生成一些业务信息。

第十二章，*Twitter 推文收集和机器学习*，将带您完成一个案例研究，实现了 Storm 中的机器学习拓扑。

# 您需要为这本书做好准备

本书中的所有代码都在 CentOS 6.5 上进行了测试。它也可以在其他 Linux 和 Windows 变体上运行，只需在命令中进行适当的更改。

我们已经尝试使各章节都是独立的，并且每章中都包括了该章节中使用的所有软件的设置和安装。这些是本书中使用的软件包：

+   CentOS 6.5

+   Oracle JDK 8

+   Apache ZooKeeper 3.4.6

+   Apache Storm 1.0.2

+   Eclipse 或 Spring Tool Suite

+   Elasticsearch 2.4.4

+   Hadoop 2.2.2

+   Logstash 5.4.1

+   Kafka 0.9.0.1

+   Esper 5.3.0

# 这本书是为谁写的

如果您是一名 Java 开发人员，并且渴望进入使用 Apache Storm 进行实时流处理应用的世界，那么这本书适合您。本书从基础知识开始，不需要之前在 Storm 方面的经验。完成本书后，您将能够开发不太复杂的 Storm 应用程序。

# 约定

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下：“在 Nimbus 机器的 `storm.yaml` 文件中添加以下行以在 Nimbus 节点上启用 JMX。”

代码块设置如下：

```scala
<dependency>
  <groupId>org.apache.storm</groupId>
  <artifactId>storm-core</artifactId>
  <version>1.0.2</version>
  <scope>provided<scope>
</dependency>
```

任何命令行输入或输出都将按如下方式编写：

```scala
cd $ZK_HOME/conf touch zoo.cfg
```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如菜单或对话框中的单词，会在文本中显示为：“现在，单击“连接”按钮以查看监督节点的指标。”

警告或重要说明看起来像这样。

技巧和窍门看起来像这样。
