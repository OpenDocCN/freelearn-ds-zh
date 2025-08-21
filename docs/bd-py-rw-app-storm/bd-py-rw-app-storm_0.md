# 前言

Apache Storm 是一个强大的框架，用于创建摄取和处理大量数据的复杂工作流。借助其 spouts 和 bolts 的通用概念，以及简单的部署和监控工具，它允许开发人员专注于其工作流的具体内容，而无需重新发明轮子。

然而，Storm 是用 Java 编写的。虽然它支持除 Java 以外的其他编程语言，但工具不完整，文档和示例很少。

本书的作者之一创建了 Petrel，这是第一个支持使用 100% Python 创建 Storm 拓扑的框架。他亲身经历了在 Java 工具集上构建 Python Storm 拓扑的困难。本书填补了这一空白，为所有经验水平的 Python 开发人员提供了一个资源，帮助他们构建自己的应用程序使用 Storm。

# 本书涵盖的内容

第一章，*熟悉 Storm*，提供了有关 Storm 用例、不同的安装模式和 Storm 配置的详细信息。

第二章，*Storm 解剖*，告诉您有关 Storm 特定术语、流程、Storm 中的容错性、调整 Storm 中的并行性和保证元组处理的详细解释。

第三章，*介绍 Petrel*，介绍了一个名为 Petrel 的框架，用于在 Python 中构建 Storm 拓扑。本章介绍了 Petrel 的安装，并包括一个简单的示例。

第四章，*示例拓扑-推特*，提供了一个关于实时计算推特数据统计的拓扑的深入示例。该示例介绍了 tick tuples 的使用，这对于需要按计划计算统计信息或执行其他操作的拓扑非常有用。在本章中，您还将看到拓扑如何访问配置数据。

第五章，*使用 Redis 和 MongoDB 进行持久化*，更新了示例推特拓扑，用于使用 Redis，一种流行的键值存储。它向您展示如何使用内置的 Redis 操作简化复杂的 Python 计算逻辑。本章还介绍了将推特数据存储在 MongoDB 中的示例，MongoDB 是一种流行的 NoSQL 数据库，并使用其聚合功能生成报告。

第六章，*实践中的 Petrel*，教授实际技能，将使开发人员在使用 Storm 时更加高效。您将学习如何使用 Petrel 为您的 spout 和 bolt 组件创建在 Storm 之外运行的自动化测试。您还将看到如何使用图形调试器来调试在 Storm 内运行的拓扑结构。

【附录】，*使用 Supervisord 管理 Storm*，是使用监督者在集群上监控和控制 Storm 的实际演示。

# 本书所需内容

您需要一台安装有 Python 2.7、Java 7 JDK 和 Apache Storm 0.9.3 的计算机。推荐使用 Ubuntu，但不是必需的。

# 本书适合对象

本书适用于初学者和高级 Python 开发人员，他们希望使用 Storm 实时处理大数据。虽然熟悉 Java 运行时环境有助于安装和配置 Storm，但本书中的所有代码示例都是用 Python 编写的。

# 约定

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例，以及它们的含义解释。

文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄显示如下："可以使用`storm.yaml`在`conf`文件夹中进行 Storm 配置"。

代码块设置如下：

```scala
import nltk.corpus

from petrel import storm
from petrel.emitter import BasicBolt

class SplitSentenceBolt(BasicBolt):
    def __init__(self):
        super(SplitSentenceBolt, self).__init__(script=__file__)
        self.stop = set(nltk.corpus.stopwords.words('english'))
        self.stop.update(['http', 'https', 'rt'])
```

当我们希望引起您对代码块的特定部分的注意时，相关行或项目将以粗体显示：

```scala
import logging
from collections import defaultdict

from petrel import storm
from petrel.emitter import BasicBolt
```

任何命令行输入或输出都以以下方式编写：

```scala
tail -f petrel24748_totalrankings.log
```

**新术语**和**重要单词**以粗体显示。例如，屏幕上看到的单词，如菜单或对话框中的单词，会以这样的方式出现在文本中："最后，点击**创建您的 Twitter 应用程序**"。

### 注意

警告或重要说明会以这样的方式出现在框中。

### 提示

提示和技巧会以这样的方式出现。
