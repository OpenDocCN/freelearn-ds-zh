# 前言

本书旨在帮助您熟悉 Apache Kafka，并解决与发布者-订阅者架构中数百万条消息消耗相关的挑战。它旨在让您开始使用 Kafka 进行编程，以便您将有一个坚实的基础，深入研究不同类型的 Kafka 生产者和消费者的实现和集成。

除了解释 Apache Kafka 之外，我们还花了一章的时间探索 Kafka 与其他技术（如 Apache Hadoop 和 Apache Storm）的集成。我们的目标不仅是让您了解 Apache Kafka 是什么，还要让您了解如何将其作为更广泛技术基础设施的一部分来使用。最后，我们将带您了解 Kafka 的操作，我们还将谈论管理问题。

# 本书涵盖的内容

第一章，“介绍 Kafka”，讨论了组织如何意识到数据的真正价值，并正在改进收集和处理数据的机制。它还描述了如何使用不同版本的 Scala 安装和构建 Kafka 0.8.x。

第二章，“设置 Kafka 集群”，描述了设置单个或多个经纪人 Kafka 集群所需的步骤，并分享了 Kafka 经纪人属性列表。

第三章，“Kafka 设计”，讨论了用于构建 Kafka 坚实基础的设计概念。它还详细讨论了 Kafka 如何处理消息压缩和复制。

第四章，“编写生产者”，提供了有关如何编写基本生产者和使用消息分区的一些高级 Java 生产者的详细信息。

第五章，“编写消费者”，提供了有关如何编写基本消费者和使用消息分区的一些高级 Java 消费者的详细信息。

第六章，“Kafka 集成”，简要介绍了 Storm 和 Hadoop，并讨论了 Kafka 如何与 Storm 和 Hadoop 集成，以满足实时和批处理需求。

第七章，“操作 Kafka”，描述了集群管理和集群镜像所需的 Kafka 工具的信息，并分享了如何将 Kafka 与 Camus、Apache Camel、Amazon Cloud 等集成的信息。

# 本书所需内容

在最简单的情况下，一个安装了 JDK 1.6 的基于 Linux（CentOS 6.x）的单台机器将为您提供一个平台，以探索本书中几乎所有练习。我们假设您熟悉命令行 Linux，因此任何现代发行版都足够。

一些示例需要多台机器才能看到工作情况，因此您需要至少访问三台这样的主机；虚拟机适用于学习和探索。

由于我们还讨论了大数据技术，如 Hadoop 和 Storm，您通常需要一个地方来运行您的 Hadoop 和 Storm 集群。

# 这本书适合谁

这本书是为那些想要了解 Apache Kafka 的人准备的；主要受众是具有软件开发经验但没有接触过 Apache Kafka 或类似技术的人。

这本书也适合企业应用程序开发人员和大数据爱好者，他们曾经使用其他基于发布者-订阅者系统，并且现在想要探索作为未来可扩展解决方案的 Apache Kafka。

# 约定

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例，以及它们的含义解释。

文本中的代码词显示如下：“从 Oracle 的网站下载`jdk-7u67-linux-x64.rpm`版本。”

代码块设置如下：

```java
String messageStr = new String("Hello from Java Producer");
KeyedMessage<Integer, String> data = new KeyedMessage<Integer, String>(topic, messageStr);
producer.send(data);
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```java
Properties props = new Properties();
props.put("metadata.broker.list","localhost:9092");
props.put("serializer.class","kafka.serializer.StringEncoder");
props.put("request.required.acks", "1");
ProducerConfig config = new ProducerConfig(props); 
Producer<Integer, String> producer = new Producer<Integer, 
    String>(config);
```

任何命令行输入或输出都以以下形式书写：

```java
[root@localhost kafka-0.8]# java SimpleProducer kafkatopic Hello_There

```

**新术语**和**重要单词**以粗体显示。

### 注意

警告或重要说明会以这样的方式出现在方框中。

### 提示

提示和技巧会以这种形式出现。
