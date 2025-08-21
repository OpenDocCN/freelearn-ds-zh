# 前言

风暴最初是 Twitter 公司的一个项目，已经毕业并加入 Apache 联盟，因此从 Twitter 风暴改名。这是 Nathan Marz 的心血结晶，现在被 Cloudera 的包括 Apache Hadoop（CDH）和 Hortonworks 数据平台（HDP）等联盟所采用。

Apache Storm 是一个高度可扩展、分布式、快速、可靠的实时计算系统，旨在处理非常高速的数据。Cassandra 通过提供闪电般快速的读写能力来补充计算能力，这是目前与风暴一起提供的最佳数据存储组合。

风暴计算和卡桑德拉存储的结合正在帮助技术传道者解决涉及复杂和大数据量情况的各种业务问题，例如实时客户服务、仪表板、安全性、传感器数据分析、数据货币化等等。

本书将使用户能够利用风暴的处理能力与 Cassandra 的速度和可靠性相结合，开发实时用例的生产级企业解决方案。

# 本书内容

第一章，“让我们了解风暴”，让您熟悉需要分布式计算解决方案的问题。它将带您了解风暴及其出现的过程。

第二章，“开始您的第一个拓扑”，教您如何设置开发者环境——沙盒，并执行一些代码示例。

第三章，“通过示例了解风暴内部”，教您如何准备风暴的喷嘴和自定义喷嘴。您将了解风暴提供的各种分组类型及其在实际问题中的应用。

第四章，“集群模式下的风暴”，教您如何设置多节点风暴集群，使用户熟悉分布式风暴设置及其组件。本章还将让您熟悉风暴 UI 和各种监控工具。

第五章，“风暴高可用性和故障转移”，将风暴拓扑与 RabbitMQ 代理服务相结合，并通过各种实际示例探讨风暴的高可用性和故障转移场景。

第六章，“向风暴添加 NoSQL 持久性”，向您介绍 Cassandra，并探讨可用于与 Cassandra 一起工作的各种包装 API。我们将使用 Hector API 连接风暴和 Cassandra。

第七章，“Cassandra 分区”、“高可用性和一致性”，带您了解 Cassandra 的内部。您将了解并应用高可用性、暗示的转交和最终一致性的概念，以及它们在 Cassandra 中的上下文中的应用。

第八章，“Cassandra 管理和维护”，让您熟悉 Cassandra 的管理方面，如扩展集群、节点替换等，从而为您提供处理 Cassandra 实际情况所需的全部经验。

第九章，“风暴管理和维护”，让您熟悉风暴的管理方面，如扩展集群、设置并行性和故障排除风暴。

第十章，*Storm 中的高级概念*，让您了解 Trident API。您将使用一些示例和说明来构建 Trident API。

第十一章，*使用 Storm 进行分布式缓存和 CEP*，让您了解分布式缓存，其需求以及在 Storm 中解决实际用例的适用性。它还将教育您关于 Esper 作为 CEP 与 Storm 结合使用。

附录，*测验答案*，包含对真假陈述和填空部分问题的所有答案。

*奖励章节*，*使用 Storm 和 Cassandra 解决实际用例*，解释了一些实际用例和使用诸如 Storm 和 Cassandra 等技术解决这些用例的蓝图。这一章节可以在[`www.packtpub.com/sites/default/files/downloads/Bonus_Chapter.pdf`](https://www.packtpub.com/sites/default/files/downloads/Bonus_Chapter.pdf)上找到。

# 您需要本书的什么

对于本书，您将需要 Linux/Ubuntu 操作系统、Eclipse 和 8GB 的 RAM。有关设置其他组件（如 Storm、RabbitMQ、Cassandra、内存缓存、Esper 等）的步骤在相应主题的章节中有所涵盖。

# 这本书适合谁

本书适用于希望使用 Storm 开始进行近实时分析的 Java 开发人员。这将作为开发高可用性和可扩展解决方案以解决复杂实时问题的专家指南。除了开发，本书还涵盖了 Storm 和 Cassandra 的管理和维护方面，这是任何解决方案投入生产的强制要求。

# 约定

在本书中，您会发现一些文本样式，用于区分不同类型的信息。以下是一些这些样式的示例，以及它们的含义解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下："在 Storm 中定义的`NumWorker`配置或`TOPOLOGY_WORKERS`配置"。

代码块设置如下：

```scala
// instantiates the new builder object
TopologyBuilder builder = new TopologyBuilder();
// Adds a new spout of type "RandomSentenceSpout" with a  parallelism hint of 5
builder.setSpout("spout", new RandomSentenceSpout(), 5);
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目会被突出显示：

```scala
  public void execute(Tuple tuple) {
      String sentence = tuple.getString(0);
      for(String word: sentence.split(" ")) {
          _collector.emit(tuple, new Values(word)); //1
      }
      _collector.ack(tuple); //2
  }
  public void declareOutputFields(OutputFieldsDeclarer  declarer) {
      declarer.declare(new Fields("word")); //3
  }
}
```

任何命令行输入或输出都以以下方式编写：

```scala
sudo apt-get -qy install rabbitmq-server

```

**新术语**和**重要单词**以粗体显示。例如，屏幕上看到的菜单或对话框中的单词会以这种方式出现在文本中："转到**管理**选项卡，选择**策略**，然后单击**添加策略**"。

### 注意

警告或重要说明会以这样的方式出现在一个框中。

### 提示

提示和技巧是这样显示的。
