# 第九章：设计 Spark 应用程序

从功能性角度思考。设想一种应用功能，它被设计成一个管道，每个部分通过管道连接，共同完成手头工作的某一部分。这一切都关乎数据处理，而 Spark 正是以高度灵活的方式进行数据处理的。数据处理始于进入处理管道的种子数据。种子数据可以是系统摄取的新数据片段，也可以是企业数据存储中的某种主数据集，需要对其进行切片和切块以生成不同的视图，以满足各种目的和业务需求。在设计和开发数据处理应用程序时，这种切片和切块将成为常态。

任何应用程序开发实践都始于对领域的研究、业务需求的`分析`以及技术工具的选择。在这里也不例外。尽管本章将探讨 Spark 应用程序的设计和开发，但最初的焦点将放在数据处理应用程序的整体架构、用例、数据以及将数据从一种状态转换为另一种状态的应用程序上。Spark 只是一个驱动程序，它利用其强大的基础设施将数据处理逻辑和数据组合在一起，以产生期望的结果。

本章我们将涵盖以下主题：

+   Lambda 架构

+   使用 Spark 进行微博

+   数据字典

+   编码风格

+   数据摄取

# Lambda 架构

应用程序架构对于任何类型的软件开发都至关重要。它是决定软件如何构建的蓝图，具有一定程度的通用性，并在需要时具备定制能力。对于常见的应用需求，已有一些流行的架构可供选择，无需从头开始构建架构。这些公共架构框架由一些顶尖人才设计，旨在惠及大众。这些流行的架构非常有用，因为它们没有准入门槛，并被许多人使用。对于 Web 应用程序开发、数据处理等，都有流行的架构可供选择。

Lambda 架构是一种新兴且流行的架构，非常适合开发数据处理应用程序。市场上有许多工具和技术可用于开发数据处理应用程序。但无论采用何种技术，数据处理应用程序组件的分层和组合方式都由架构框架驱动。这就是为什么 Lambda 架构是一种与技术无关的架构框架，根据需要，可以选择适当的技术来开发各个组件。*图 1*捕捉了 Lambda 架构的精髓：

![Lambda 架构](img/image_09_001.jpg)

图 1

Lambda 架构由三个层次组成：

+   批处理层是主要的数据存储。任何类型的处理都在此数据集上进行。这是黄金数据集。

+   服务层处理主数据集并准备特定目的的视图，在此称为有目的的视图。此中间处理步骤对于服务查询或为特定需求生成输出是必要的。查询和特定数据集准备不直接访问主数据集。

+   速度层关注数据流处理。数据流以实时方式处理，如果业务需要，会准备易变的实时视图。查询或生成输出的特定过程可能同时消耗有目的的数据视图和实时视图中的数据。

利用 Lambda 架构原理来构建大数据处理系统，此处将使用 Spark 作为数据处理工具。Spark 完美适配所有三层不同数据处理需求。

本章将讨论一个微博应用的若干精选数据处理用例。应用功能、部署基础设施及可扩展性因素不在本工作讨论范围之内。在典型的批处理层中，主数据集可以是简单的可分割序列化格式或 NoSQL 数据存储，具体取决于数据访问方法。如果应用用例均为批处理操作，则标准序列化格式已足够。但如果用例要求随机访问，NoSQL 数据存储将是理想选择。为简化起见，此处所有数据文件均本地存储为纯文本文件。

典型的应用开发以完全功能性应用告终。但在此，用例通过 Spark 数据处理应用实现。数据处理始终作为主应用功能的一部分，并按计划以批处理模式运行或作为监听器等待数据并进行处理。因此，针对每个用例，开发了独立的 Spark 应用，并根据情况安排其运行或处于监听模式。

# 基于 Lambda 架构的微博

博客作为一种出版媒介已有数十年历史，形式多样。在博客初期，只有专业或有抱负的作家通过博客发表文章。这传播了一个错误观念，即只有严肃内容才通过博客发布。近年来，微博客的概念将公众纳入了博客文化。微博客是人们思维过程的突然爆发，以几句话、照片、视频或链接的形式呈现。Twitter 和 Tumblr 等网站以最大规模推广了这种文化，拥有数亿活跃用户。

## SfbMicroBlog 概览

**SfbMicroBlog**是一个拥有数百万用户发布短消息的微博应用。新用户若要使用此应用，需先注册用户名和密码。要发布消息，用户必须先登录。用户在不登录的情况下唯一能做的就是阅读其他用户发布的公开消息。用户可以关注其他用户。关注是一种单向关系。如果用户 A 关注用户 B，用户 A 可以看到用户 B 发布的所有消息；同时，用户 B 看不到用户 A 发布的消息，因为用户 B 没有关注用户 A。默认情况下，所有用户发布的所有消息都是公开的，任何人都可以看到。但用户可以设置，使消息仅对其关注者可见。成为关注者后，也可以取消关注。

用户名必须在所有用户中唯一。登录需要用户名和密码。每个用户必须有一个主要电子邮件地址，否则注册过程将无法完成。为了额外的安全性和密码恢复，可以在个人资料中保存备用电子邮件地址或手机号码。

消息不得超过 140 个字符。消息可以包含以#符号为前缀的单词，以便将它们归类到各种话题下。消息可以包含以@符号为前缀的用户名，以便通过发布的消息直接向用户发送消息。换句话说，用户可以在他们的消息中提及任何其他用户，而无需成为其关注者。

一旦发布，消息无法更改。一旦发布，消息无法删除。

## 熟悉数据

所有进入主数据集的数据都通过一个数据流。数据流经过处理，对每条消息的适当头部进行检查，并采取正确的行动将其存储在数据存储中。以下列表包含了通过同一数据流进入存储的重要数据项：

+   **用户**：此数据集包含用户登录时或用户数据发生变更时的用户详细信息。

+   **关注者**：此数据集包含当用户选择关注另一用户时捕获的关系数据。

+   **消息**：此数据集包含注册用户发布的消息。

这些数据集构成了黄金数据集。基于此主数据集，创建了各种视图，以满足应用中关键业务功能的需求。以下列表包含了主数据集的重要视图：

+   **用户发布的消息**：此视图包含系统中每个用户发布的消息。当特定用户想要查看自己发布的消息时，会使用此视图生成的数据。这也被该用户的关注者使用。这是一种特定目的使用主数据集的情况。消息数据集为此视图提供了所有必需的数据。

+   **向用户发送消息**：在消息中，可以通过在@符号后加上收件人的用户名来指定特定用户。此数据视图包含被@符号标记的用户及其对应的消息。在实现中有一个限制：一条消息只能有一个收件人。

+   **标签消息**：在消息中，以#符号开头的单词成为可搜索的消息。例如，消息中的#spark 一词表示该消息可通过#spark 进行搜索。对于给定的标签，用户可以在一个列表中查看所有公开消息以及他/她所关注用户的消息。此视图包含标签与相应消息的配对。在实现中有一个限制：一条消息只能有一个标签。

+   **关注者用户**：此视图包含关注特定用户的用户列表。在*图 2*中，用户**U1**和**U3**位于关注**U4**的用户列表中。

+   **被关注用户**：此视图包含被特定用户关注的用户列表。在*图 2*中，用户**U2**和**U4**位于被用户**U1**关注的用户列表中：

![熟悉数据](img/image_09_002.jpg)

图 2

简而言之，*图 3*给出了解决方案的 Lambda 架构视图，并详细说明了数据集及其对应的视图：

![熟悉数据](img/image_09_003-1.jpg)

图 3

## 设置数据字典

数据字典描述了数据、其含义以及与其他数据项的关系。对于 SfbMicroBlog 应用程序，数据字典将是一个非常简约的实现所选用例的工具。以此为基础，读者可以扩展并实现自己的数据项，并包含数据处理用例。数据字典为所有主数据集以及数据视图提供。

下表展示了用户数据集的数据项：

| **用户数据** | **类型** | **用途** |
| --- | --- | --- |
| 用户 ID | 长整型 | 用于唯一标识用户，同时也是用户关系图中的顶点标识 |
| 用户名 | 字符串 | 用于系统中用户的唯一标识 |
| 名字 | 字符串 | 用于记录用户的名 |
| 姓氏 | 字符串 | 用于记录用户的姓 |
| 邮箱 | 字符串 | 用于与用户沟通 |
| 备用邮箱 | 字符串 | 用于密码找回 |
| 主电话 | 字符串 | 用于密码找回 |

下表捕捉了关注者数据集的数据项：

| **关注者数据** | **类型** | **用途** |
| --- | --- | --- |
| 关注者用户名 | 字符串 | 用于识别关注者身份 |
| 被关注用户名 | 字符串 | 用于识别被关注者 |

下表捕捉了消息数据集的数据项：

| **消息数据** | **类型** | **用途** |
| --- | --- | --- |
| 用户名 | 字符串 | 用于记录发布消息的用户 |
| 消息 ID | 长整型 | 用于唯一标识一条消息 |
| 消息 | 字符串 | 用于记录正在发布的消息 |
| 时间戳 | 长整型 | 用于记录消息发布的时间 |

下表记录了用户查看消息的数据项：

| **消息至用户数据** | **类型** | **目的** |
| --- | --- | --- |
| 来自用户名 | 字符串 | 用于记录发布消息的用户 |
| 目标用户名 | 字符串 | 用于记录消息的接收者；它是前缀带有@符号的用户名 |
| 消息 ID | 长整型 | 用于唯一标识一条消息 |
| 消息 | 字符串 | 用于记录正在发布的消息 |
| 时间戳 | 长整型 | 用于记录消息发布的时间 |

下表记录了标记消息视图的数据项：

| **标记消息数据** | **类型** | **目的** |
| --- | --- | --- |
| 标签 | 字符串 | 前缀带有#符号的单词 |
| 用户名 | 字符串 | 用于记录发布消息的用户 |
| 消息 ID | 长整型 | 用于唯一标识一条消息 |
| 消息 | 字符串 | 用于记录正在发布的消息 |
| 时间戳 | 长整型 | 用于记录消息发布的时间 |

用户的关注关系相当直接，由存储在数据存储中的一对用户标识号组成。

# 实现 Lambda 架构

本章开头介绍了 Lambda 架构的概念。由于它是一种与技术无关的架构框架，因此在设计应用程序时，必须记录特定实现中使用的技术选择。以下各节正是这样做的。

## 批处理层

批处理层的核心是一个数据存储。对于大数据应用，数据存储有很多选择。通常，**Hadoop 分布式文件系统**（**HDFS**）与 Hadoop YARN 结合使用是目前公认的平台，主要是因为它能够在 Hadoop 集群中划分和分布数据。

任何持久存储支持的两种数据访问类型：

+   批量写入/读取

+   随机写入/读取

这两种类型都需要单独的数据存储解决方案。对于批量数据操作，通常使用可分割的序列化格式，如 Avro 和 Parquet。对于随机数据操作，通常使用 NoSQL 数据存储。其中一些 NoSQL 解决方案位于 HDFS 之上，而有些则不是。无论它们是否位于 HDFS 之上，它们都提供数据的划分和分布。因此，根据用例和使用的分布式平台，可以选择适当的解决方案。

当涉及到 HDFS 中的数据存储时，常用的格式如 XML 和 JSON 会失败，因为 HDFS 会对文件进行分区并分布。当这种情况发生时，这些格式具有开始标签和结束标签，文件中随机位置的分割会使数据变得脏乱。因此，可分割的文件格式如 Avro 或 Parquet 在 HDFS 中存储效率更高。

在 NoSQL 数据存储解决方案方面，市场上有许多选择，特别是在开源世界中。其中一些 NoSQL 数据存储，如 Hbase，位于 HDFS 之上。其他一些 NoSQL 数据存储，如 Cassandra 和 Riak，不需要 HDFS，可以在常规操作系统上部署，并且可以以无主模式部署，从而在集群中没有单点故障。NoSQL 存储的选择再次取决于组织内特定技术的使用、现有的生产支持合同以及其他许多参数。

### 提示

本书并不推荐特定的数据存储技术与 Spark 结合使用，因为 Spark 驱动程序对于大多数流行的序列化格式和 NoSQL 数据存储都十分丰富。换句话说，大多数数据存储供应商已经开始大力支持 Spark。另一个有趣的趋势是，许多主流的 ETL 工具已经开始支持 Spark，因此使用这些 ETL 工具的用户可能会在其 ETL 处理管道中使用 Spark 应用程序。

在本应用程序中，为了保持简单并避免运行应用程序所需的复杂基础设施设置，既没有使用基于 HDFS 的数据存储，也没有使用任何基于 NoSQL 的数据存储。整个过程中，数据以文本文件格式存储在本地系统上。对在 HDFS 或其他 NoSQL 数据存储上尝试示例感兴趣的读者可以继续尝试，只需对应用程序的数据写入/读取部分进行一些更改。

## 服务层

服务层可以通过 Spark 使用多种方法实现。如果数据是非结构化的且纯粹基于对象，则适合使用低级别的 RDD 方法。如果数据是结构化的，DataFrame 是理想选择。这里讨论的使用案例涉及结构化数据，因此只要有可能，就会使用 Spark SQL 库。从数据存储中读取数据并创建 RDD。将 RDD 转换为 DataFrames，并使用 Spark SQL 完成所有服务需求。这样，代码将简洁且易于理解。

## 速度层

速度层将作为 Spark Streaming 应用程序实现，使用 Kafka 作为代理，其自己的生产者生成消息。Spark Streaming 应用程序将作为 Kafka 主题的消费者，接收正在生产的数据。正如在涵盖 Spark Streaming 的章节中所讨论的，生产者可以是 Kafka 控制台生产者或 Kafka 支持的任何其他生产者。但这里的 Spark Streaming 应用程序作为消费者，不会实现将处理过的消息持久化到文本文件的逻辑，因为这在现实世界的用例中并不常见。以这个应用程序为基础，读者可以实现自己的持久化机制。

### 查询

所有查询都来自速度层和服务层。由于数据以 DataFrames 的形式提供，如前所述，该用例的所有查询都是使用 Spark SQL 实现的。显而易见的原因是 Spark SQL 作为一种整合技术，统一了数据源和目的地。当读者使用本书中的示例，并准备将其应用于现实世界的用例时，整体方法可以保持不变，但数据源和目的地可能会有所不同。以下是服务层可以生成的一些查询。读者可以根据需要修改数据字典，并能够编写这些视图或查询，这取决于他们的想象力：

+   查找按给定标签分组的消息

+   查找发送给指定用户的消息

+   查找指定用户的关注者

+   查找指定用户关注的用户

# 使用 Spark 应用程序

这个应用程序的工作主力是数据处理引擎，由多个 Spark 应用程序组成。一般来说，它们可以分为以下类型：

+   摄取数据的 Spark Streaming 应用程序：这是主要的监听应用程序，接收作为流过来的数据，并将其存储在适当的主数据集中。

+   创建目的视图和查询的 Spark 应用程序：这是用于从主数据集中创建各种目的视图的应用程序。除此之外，查询也包含在这个应用程序中。

+   进行自定义数据处理的 Spark GraphX 应用程序：这是用于处理用户-关注者关系的应用程序。

所有这些应用程序都是独立开发的，并且独立提交，但流处理应用程序将始终作为侦听应用程序运行，以处理传入的消息。除了主要的数据流应用程序外，所有其他应用程序都像常规作业一样进行调度，例如 UNIX 系统中的 cron 作业。在此应用程序中，所有这些应用程序都在生成各种目的的视图。调度取决于应用程序的类型以及主数据集和视图之间可以承受多少延迟。这完全取决于业务功能。因此，本章将重点放在 Spark 应用程序开发上，而不是调度上，以保持对前面章节中学到的内容的专注。

### 提示

在实现现实世界的用例时，将速度层的数据持久化到文本文件中并不是理想的选择。为了简化，所有数据都存储在文本文件中，以便为所有级别的读者提供最简单的设置。使用 Spark Streaming 实现的速度层是一个没有持久化逻辑的骨架实现。读者可以对其进行增强，以将持久化引入到他们所需的数据存储中。

# 编码风格

编码风格已在前面章节中讨论过，并且已经完成了大量 Spark 应用程序编程。到目前为止，本书已经证明 Spark 应用程序开发可以使用 Scala、Python 和 R 进行。在大多数前面章节中，首选语言是 Scala 和 Python。在本章中，这一趋势将继续。仅对于 Spark GraphX 应用程序，由于没有 Python 支持，应用程序将仅使用 Scala 开发。

编码风格将简单明了。为了专注于 Spark 特性，故意避免了应用程序开发中的错误处理和其他最佳实践。在本章中，只要有可能，代码都是从相应语言的 Spark REPL 运行的。由于完整应用程序的结构以及编译、构建和运行它们作为应用程序的脚本已经在讨论 Spark Streaming 的章节中涵盖，源代码下载将提供完整的即用型应用程序。此外，涵盖 Spark Streaming 的章节讨论了完整 Spark 应用程序的结构，包括构建和运行 Spark 应用程序的脚本。本章中将要开发的应用程序也将采用相同的方法。当运行本书初始章节中讨论的此类独立 Spark 应用程序时，读者可以启用 Spark 监控，并查看应用程序的行为。为了简洁起见，这些讨论将不再在此处重复。

# 设置源代码

*图 4*展示了本章使用的源代码和数据目录的结构。由于读者应已熟悉这些内容，且已在第六章，*Spark 流处理*中涵盖，此处不再赘述。运行使用 Kafka 的程序需要外部库文件依赖。为此，下载 JAR 文件的说明位于`lib`文件夹中的`TODO.txt`文件。`submitPy.sh`和`submit.sh`文件也使用了 Kafka 安装中的一些`Kafka`库。所有这些外部 JAR 文件依赖已在第六章，*Spark 流处理*中介绍。

![设置源代码](img/image_09_004.jpg)

图 4

# 理解数据摄取

Spark Streaming 应用作为监听应用，接收来自其生产者的数据。由于 Kafka 将用作消息代理，Spark Streaming 应用将成为其消费者应用，监听主题以接收生产者发送的消息。由于批处理层的母数据集包含以下数据集，因此为每个主题及其数据集分别设置 Kafka 主题是理想的。

+   用户数据集：用户

+   关注者数据集：关注者

+   消息数据集：消息

*图 5*展示了基于 Kafka 的 Spark Streaming 应用的整体结构：

![理解数据摄取](img/image_09_005.jpg)

图 5

由于已在第六章，*Spark 流处理*中介绍了 Kafka 设置，此处仅涵盖应用代码。

以下脚本从终端窗口运行。确保`$KAFKA_HOME`环境变量指向 Kafka 安装目录。同时，在单独的终端窗口中启动 Zookeeper、Kafka 服务器、Kafka 生产者以及 Spark Streaming 日志事件数据处理应用非常重要。一旦按照脚本创建了必要的 Kafka 主题，相应的生产者就必须开始发送消息。在进一步操作之前，请参考已在第六章，*Spark 流处理*中介绍的 Kafka 设置细节。

尝试在终端窗口提示符下执行以下命令：

```scala
 $ # Start the Zookeeper 
$ cd $KAFKA_HOME
$ $KAFKA_HOME/bin/zookeeper-server-start.sh
 $KAFKA_HOME/config/zookeeper.properties
      [2016-07-30 12:50:15,896] INFO binding to port 0.0.0.0/0.0.0.0:2181
	  (org.apache.zookeeper.server.NIOServerCnxnFactory)

	$ # Start the Kafka broker in a separate terminal window
	$ $KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties
      [2016-07-30 12:51:39,206] INFO [Kafka Server 0], started 
	  (kafka.server.KafkaServer)

	$ # Create the necessary Kafka topics. This is to be done in a separate terminal window
	$ $KAFKA_HOME/bin/kafka-topics.sh --create --zookeeper localhost:2181
	--replication-factor 1 --partitions 1 --topic user
      Created topic "user".
    $ $KAFKA_HOME/bin/kafka-topics.sh --create --zookeeper localhost:2181
	--replication-factor 1 --partitions 1 --topic follower
      Created topic "follower".

	$ $KAFKA_HOME/bin/kafka-topics.sh --create --zookeeper localhost:2181
	--replication-factor 1 --partitions 1 --topic message
      Created topic "message".

	$ # Start producing messages and publish to the topic "message"
	$ $KAFKA_HOME/bin/kafka-console-producer.sh --broker-list localhost:9092 
	--topic message

```

本节提供了 Scala 代码的详细信息，该代码用于 Kafka 主题消费者应用程序，该应用程序处理 Kafka 生产者产生的消息。在运行以下代码片段之前，假设 Kafka 已启动并运行，所需的生产者正在产生消息，然后，如果运行该应用程序，它将开始消费这些消息。Scala 数据摄取程序通过将其提交到 Spark 集群来运行。从 Scala 目录开始，如*图 4*所示，首先编译程序，然后运行它。应查阅`README.txt`文件以获取额外指令。执行以下两条命令以编译和运行程序：

```scala
 $ ./compile.sh
	$ ./submit.sh com.packtpub.sfb.DataIngestionApp 1

```

以下代码是将要使用前面命令编译和运行的程序清单：

```scala
 /**
	The following program can be compiled and run using SBT
	Wrapper scripts have been provided with thisThe following script can be run to compile the code
	./compile.sh
	The following script can be used to run this application in Spark.
	The second command line argument of value 1 is very important.
	This is to flag the shipping of the kafka jar files to the Spark cluster
	./submit.sh com.packtpub.sfb.DataIngestionApp 1
	**/
	package com.packtpub.sfb
	import java.util.HashMap
	import org.apache.spark.streaming._
	import org.apache.spark.sql.{Row, SparkSession}
	import org.apache.spark.streaming.kafka._
	import org.apache.kafka.clients.producer.{ProducerConfig, KafkaProducer, ProducerRecord}
	import org.apache.spark.storage.StorageLevel
	import org.apache.log4j.{Level, Logger}
	object DataIngestionApp {
	def main(args: Array[String]) {
	// Log level settings
	LogSettings.setLogLevels()
	//Check point directory for the recovery
	val checkPointDir = "/tmp"
    /**
    * The following function has to be used to have checkpointing and driver recovery
    * The way it should be used is to use the StreamingContext.getOrCreate with this function and do a start of that
	* This function example has been discussed but not used in the chapter covering Spark Streaming. But here it is being used    */
    def sscCreateFn(): StreamingContext = {
	// Variables used for creating the Kafka stream
	// Zookeeper host
	val zooKeeperQuorum = "localhost"
	// Kaka message group
	val messageGroup = "sfb-consumer-group"
	// Kafka topic where the programming is listening for the data
	// Reader TODO: Here only one topic is included, it can take a comma separated string containing the list of topics. 
	// Reader TODO: When using multiple topics, use your own logic to extract the right message and persist to its data store
	val topics = "message"
	val numThreads = 1     
	// Create the Spark Session, the spark context and the streaming context      
	val spark = SparkSession
	.builder
	.appName(getClass.getSimpleName)
	.getOrCreate()
	val sc = spark.sparkContext
	val ssc = new StreamingContext(sc, Seconds(10))
	val topicMap = topics.split(",").map((_, numThreads.toInt)).toMap
	val messageLines = KafkaUtils.createStream(ssc, zooKeeperQuorum, messageGroup, topicMap).map(_._2)
	// This is where the messages are printed to the console. 
	// TODO - As an exercise to the reader, instead of printing messages to the console, implement your own persistence logic
	messageLines.print()
	//Do checkpointing for the recovery
	ssc.checkpoint(checkPointDir)
	// return the Spark Streaming Context
	ssc
    }
	// Note the function that is defined above for creating the Spark streaming context is being used here to create the Spark streaming context. 
	val ssc = StreamingContext.getOrCreate(checkPointDir, sscCreateFn)
	// Start the streaming
    ssc.start()
	// Wait till the application is terminated               
    ssc.awaitTermination() 
	}
	}
	object LogSettings {
	/** 
	Necessary log4j logging level settings are done 
	*/
	def setLogLevels() {
    val log4jInitialized = Logger.getRootLogger.getAllAppenders.hasMoreElements
    if (!log4jInitialized) {
	// This is to make sure that the console is clean from other INFO messages printed by Spark
	Logger.getRootLogger.setLevel(Level.INFO)
    }
	}
	}

```

Python 数据摄取程序通过将其提交到 Spark 集群来运行。从 Python 目录开始，如*图 4*所示，运行程序。应查阅`README.txt`文件以获取额外指令。运行此 Python 程序时，所有 Kafka 安装要求仍然有效。以下命令用于运行程序。由于 Python 是解释型语言，此处无需编译：

```scala
 $ ./submitPy.sh DataIngestionApp.py 1

```

以下代码片段是同一应用程序的 Python 实现：

```scala
 # The following script can be used to run this application in Spark
# ./submitPy.sh DataIngestionApp.py 1
  from __future__ import print_function
  import sys
  from pyspark import SparkContext
  from pyspark.streaming import StreamingContext
  from pyspark.streaming.kafka import KafkaUtils
  if __name__ == "__main__":
# Create the Spark context
  sc = SparkContext(appName="DataIngestionApp")
  log4j = sc._jvm.org.apache.log4j
  log4j.LogManager.getRootLogger().setLevel(log4j.Level.WARN)
# Create the Spark Streaming Context with 10 seconds batch interval
  ssc = StreamingContext(sc, 10)
# Check point directory setting
  ssc.checkpoint("\tmp")
# Zookeeper host
  zooKeeperQuorum="localhost"
# Kaka message group
  messageGroup="sfb-consumer-group"
# Kafka topic where the programming is listening for the data
# Reader TODO: Here only one topic is included, it can take a comma separated  string containing the list of topics. 
# Reader TODO: When using multiple topics, use your own logic to extract the right message and persist to its data store
topics = "message"
numThreads = 1    
# Create a Kafka DStream
kafkaStream = KafkaUtils.createStream(ssc, zooKeeperQuorum, messageGroup, {topics: numThreads})
messageLines = kafkaStream.map(lambda x: x[1])
# This is where the messages are printed to the console. Instead of this, implement your own persistence logic
messageLines.pprint()
# Start the streaming
ssc.start()
# Wait till the application is terminated   
ssc.awaitTermination() 

```

# 生成目标视图和查询

以下 Scala 和 Python 的实现是本章前面部分讨论的创建目标视图和查询的应用程序。在 Scala REPL 提示符下，尝试以下语句：

```scala
 //TODO: Change the following directory to point to your data directory
scala> val dataDir = "/Users/RajT/Documents/Writing/SparkForBeginners/To-PACKTPUB/Contents/B05289-09-DesigningSparkApplications/Code/Data/"
      dataDir: String = /Users/RajT/Documents/Writing/SparkForBeginners/To-PACKTPUB/Contents/B05289-09-DesigningSparkApplications/Code/Data/
    scala> //Define the case classes in Scala for the entities
	scala> case class User(Id: Long, UserName: String, FirstName: String, LastName: String, EMail: String, AlternateEmail: String, Phone: String)
      defined class User
    scala> case class Follow(Follower: String, Followed: String)
      defined class Follow
    scala> case class Message(UserName: String, MessageId: Long, ShortMessage: String, Timestamp: Long)
      defined class Message
    scala> case class MessageToUsers(FromUserName: String, ToUserName: String, MessageId: Long, ShortMessage: String, Timestamp: Long)
      defined class MessageToUsers
    scala> case class TaggedMessage(HashTag: String, UserName: String, MessageId: Long, ShortMessage: String, Timestamp: Long)
      defined class TaggedMessage
    scala> //Define the utility functions that are to be passed in the applications
	scala> def toUser =  (line: Seq[String]) => User(line(0).toLong, line(1), line(2),line(3), line(4), line(5), line(6))
      toUser: Seq[String] => User
    scala> def toFollow =  (line: Seq[String]) => Follow(line(0), line(1))
      toFollow: Seq[String] => Follow
    scala> def toMessage =  (line: Seq[String]) => Message(line(0), line(1).toLong, line(2), line(3).toLong)
      toMessage: Seq[String] => Message
    scala> //Load the user data into a Dataset
	scala> val userDataDS = sc.textFile(dataDir + "user.txt").map(_.split("\\|")).map(toUser(_)).toDS()
      userDataDS: org.apache.spark.sql.Dataset[User] = [Id: bigint, UserName: string ... 5 more fields]
    scala> //Convert the Dataset into data frame
	scala> val userDataDF = userDataDS.toDF()
      userDataDF: org.apache.spark.sql.DataFrame = [Id: bigint, UserName: string ... 5 more fields]
    scala> userDataDF.createOrReplaceTempView("user")
	scala> userDataDF.show()
      +---+--------+---------+--------+--------------------+----------------+--------------+

      | Id|UserName|FirstName|LastName|               EMail|  AlternateEmail|         Phone|

      +---+--------+---------+--------+--------------------+----------------+--------------+

      |  1| mthomas|     Mark|  Thomas| mthomas@example.com|mt12@example.com|+4411860297701|

      |  2|mithomas|  Michael|  Thomas|mithomas@example.com| mit@example.com|+4411860297702|

      |  3|  mtwain|     Mark|   Twain|  mtwain@example.com| mtw@example.com|+4411860297703|

      |  4|  thardy|   Thomas|   Hardy|  thardy@example.com|  th@example.com|+4411860297704|

      |  5| wbryson|  William|  Bryson| wbryson@example.com|  bb@example.com|+4411860297705|

      |  6|   wbrad|  William|Bradford|   wbrad@example.com|  wb@example.com|+4411860297706|

      |  7| eharris|       Ed|  Harris| eharris@example.com|  eh@example.com|+4411860297707|

      |  8|   tcook|   Thomas|    Cook|   tcook@example.com|  tk@example.com|+4411860297708|

      |  9| arobert|     Adam|  Robert| arobert@example.com|  ar@example.com|+4411860297709|

      | 10|  jjames|    Jacob|   James|  jjames@example.com|  jj@example.com|+4411860297710|

      +---+--------+---------+--------+--------------------+----------------+--------------+
    scala> //Load the follower data into an Dataset
	scala> val followerDataDS = sc.textFile(dataDir + "follower.txt").map(_.split("\\|")).map(toFollow(_)).toDS()
      followerDataDS: org.apache.spark.sql.Dataset[Follow] = [Follower: string, Followed: string]
    scala> //Convert the Dataset into data frame
	scala> val followerDataDF = followerDataDS.toDF()
      followerDataDF: org.apache.spark.sql.DataFrame = [Follower: string, Followed: string]
    scala> followerDataDF.createOrReplaceTempView("follow")
	scala> followerDataDF.show()
      +--------+--------+

      |Follower|Followed|

      +--------+--------+

      | mthomas|mithomas|

      | mthomas|  mtwain|

      |  thardy| wbryson|

      |   wbrad| wbryson|

      | eharris| mthomas|

      | eharris|   tcook|

      | arobert|  jjames|

      +--------+--------+
    scala> //Load the message data into an Dataset
	scala> val messageDataDS = sc.textFile(dataDir + "message.txt").map(_.split("\\|")).map(toMessage(_)).toDS()
      messageDataDS: org.apache.spark.sql.Dataset[Message] = [UserName: string, MessageId: bigint ... 2 more fields]
    scala> //Convert the Dataset into data frame
	scala> val messageDataDF = messageDataDS.toDF()
      messageDataDF: org.apache.spark.sql.DataFrame = [UserName: string, MessageId: bigint ... 2 more fields]
    scala> messageDataDF.createOrReplaceTempView("message")
	scala> messageDataDF.show()
      +--------+---------+--------------------+----------+

      |UserName|MessageId|        ShortMessage| Timestamp|

      +--------+---------+--------------------+----------+

      | mthomas|        1|@mithomas Your po...|1459009608|

      | mthomas|        2|Feeling awesome t...|1459010608|

      |  mtwain|        3|My namesake in th...|1459010776|

      |  mtwain|        4|Started the day w...|1459011016|

      |  thardy|        5|It is just spring...|1459011199|

      | wbryson|        6|Some days are rea...|1459011256|

      |   wbrad|        7|@wbryson Stuff ha...|1459011333|

      | eharris|        8|Anybody knows goo...|1459011426|

      |   tcook|        9|Stock market is p...|1459011483|

      |   tcook|       10|Dont do day tradi...|1459011539|

      |   tcook|       11|I have never hear...|1459011622|

      |   wbrad|       12|#Barcelona has pl...|1459157132|

      |  mtwain|       13|@wbryson It is go...|1459164906|

      +--------+---------+--------------------+----------+ 

```

这些步骤完成了将所有必需数据从持久存储加载到 DataFrames 的过程。这里，数据来自文本文件。在实际应用中，数据可能来自流行的 NoSQL 数据存储、传统 RDBMS 表，或是从 HDFS 加载的 Avro 或 Parquet 序列化数据存储。

以下部分使用这些 DataFrames 创建了各种目标视图和查询：

```scala
 scala> //Create the purposed view of the message to users
	scala> val messagetoUsersDS = messageDataDS.filter(_.ShortMessage.contains("@")).map(message => (message.ShortMessage.split(" ").filter(_.contains("@")).mkString(" ").substring(1), message)).map(msgTuple => MessageToUsers(msgTuple._2.UserName, msgTuple._1, msgTuple._2.MessageId, msgTuple._2.ShortMessage, msgTuple._2.Timestamp))
      messagetoUsersDS: org.apache.spark.sql.Dataset[MessageToUsers] = [FromUserName: string, ToUserName: string ... 3 more fields]

	scala> //Convert the Dataset into data frame
	scala> val messagetoUsersDF = messagetoUsersDS.toDF()
      messagetoUsersDF: org.apache.spark.sql.DataFrame = [FromUserName: string, ToUserName: string ... 3 more fields]

	scala> messagetoUsersDF.createOrReplaceTempView("messageToUsers")
	scala> messagetoUsersDF.show()
      +------------+----------+---------+--------------------+----------+

      |FromUserName|ToUserName|MessageId|        ShortMessage| Timestamp|

      +------------+----------+---------+--------------------+----------+

      |     mthomas|  mithomas|        1|@mithomas Your po...|1459009608|

      |       wbrad|   wbryson|        7|@wbryson Stuff ha...|1459011333|

      |      mtwain|   wbryson|       13|@wbryson It is go...|1459164906|

      +------------+----------+---------+--------------------+----------+
    scala> //Create the purposed view of tagged messages 
	scala> val taggedMessageDS = messageDataDS.filter(_.ShortMessage.contains("#")).map(message => (message.ShortMessage.split(" ").filter(_.contains("#")).mkString(" "), message)).map(msgTuple => TaggedMessage(msgTuple._1, msgTuple._2.UserName, msgTuple._2.MessageId, msgTuple._2.ShortMessage, msgTuple._2.Timestamp))
      taggedMessageDS: org.apache.spark.sql.Dataset[TaggedMessage] = [HashTag: string, UserName: string ... 3 more fields]

	scala> //Convert the Dataset into data frame
	scala> val taggedMessageDF = taggedMessageDS.toDF()
      taggedMessageDF: org.apache.spark.sql.DataFrame = [HashTag: string, UserName: string ... 3 more fields]

	scala> taggedMessageDF.createOrReplaceTempView("taggedMessages")
	scala> taggedMessageDF.show()
      +----------+--------+---------+--------------------+----------+

      |   HashTag|UserName|MessageId|        ShortMessage| Timestamp|

      +----------+--------+---------+--------------------+----------+

      |#Barcelona| eharris|        8|Anybody knows goo...|1459011426|

      |#Barcelona|   wbrad|       12|#Barcelona has pl...|1459157132|

      +----------+--------+---------+--------------------+----------+

	scala> //The following are the queries given in the use cases
	scala> //Find the messages that are grouped by a given hash tag
	scala> val byHashTag = spark.sql("SELECT a.UserName, b.FirstName, b.LastName, a.MessageId, a.ShortMessage, a.Timestamp FROM taggedMessages a, user b WHERE a.UserName = b.UserName AND HashTag = '#Barcelona' ORDER BY a.Timestamp")
      byHashTag: org.apache.spark.sql.DataFrame = [UserName: string, FirstName: string ... 4 more fields]

	scala> byHashTag.show()
      +--------+---------+--------+---------+--------------------+----------+

      |UserName|FirstName|LastName|MessageId|        ShortMessage| Timestamp|

      +--------+---------+--------+---------+--------------------+----------+

      | eharris|       Ed|  Harris|        8|Anybody knows goo...|1459011426|

      |   wbrad|  William|Bradford|       12|#Barcelona has pl...|1459157132|

      +--------+---------+--------+---------+--------------------+----------+

	scala> //Find the messages that are addressed to a given user
	scala> val byToUser = spark.sql("SELECT FromUserName, ToUserName, MessageId, ShortMessage, Timestamp FROM messageToUsers WHERE ToUserName = 'wbryson' ORDER BY Timestamp")
      byToUser: org.apache.spark.sql.DataFrame = [FromUserName: string, ToUserName: string ... 3 more fields]

	scala> byToUser.show()
      +------------+----------+---------+--------------------+----------+

      |FromUserName|ToUserName|MessageId|        ShortMessage| Timestamp|

      +------------+----------+---------+--------------------+----------+

      |       wbrad|   wbryson|        7|@wbryson Stuff ha...|1459011333|

      |      mtwain|   wbryson|       13|@wbryson It is go...|1459164906|

      +------------+----------+---------+--------------------+----------+
    scala> //Find the followers of a given user
	scala> val followers = spark.sql("SELECT b.FirstName as FollowerFirstName, b.LastName as FollowerLastName, a.Followed FROM follow a, user b WHERE a.Follower = b.UserName AND a.Followed = 'wbryson'")
      followers: org.apache.spark.sql.DataFrame = [FollowerFirstName: string, FollowerLastName: string ... 1 more field]
    scala> followers.show()
      +-----------------+----------------+--------+

      |FollowerFirstName|FollowerLastName|Followed|

      +-----------------+----------------+--------+

      |          William|        Bradford| wbryson|

      |           Thomas|           Hardy| wbryson|

      +-----------------+----------------+--------+

	scala> //Find the followedUsers of a given user
	scala> val followedUsers = spark.sql("SELECT b.FirstName as FollowedFirstName, b.LastName as FollowedLastName, a.Follower FROM follow a, user b WHERE a.Followed = b.UserName AND a.Follower = 'eharris'")
      followedUsers: org.apache.spark.sql.DataFrame = [FollowedFirstName: string, FollowedLastName: string ... 1 more field]
    scala> followedUsers.show()
      +-----------------+----------------+--------+

      |FollowedFirstName|FollowedLastName|Follower|

      +-----------------+----------------+--------+

      |           Thomas|            Cook| eharris|

      |             Mark|          Thomas| eharris|

      +-----------------+----------------+--------+ 

```

在前述的 Scala 代码片段中，由于所选编程语言为 Scala，因此使用了基于数据集和 DataFrame 的编程模型。现在，由于 Python 不是强类型语言，Python 不支持 Dataset API，因此下面的 Python 代码结合使用了 Spark 的传统 RDD 基础编程模型和基于 DataFrame 的编程模型。在 Python REPL 提示符下，尝试以下语句：

```scala
 >>> from pyspark.sql import Row
	>>> #TODO: Change the following directory to point to your data directory
	>>> dataDir = "/Users/RajT/Documents/Writing/SparkForBeginners/To-PACKTPUB/Contents/B05289-09-DesigningSparkApplications/Code/Data/"
	>>> #Load the user data into an RDD
	>>> userDataRDD = sc.textFile(dataDir + "user.txt").map(lambda line: line.split("|")).map(lambda p: Row(Id=int(p[0]), UserName=p[1], FirstName=p[2], LastName=p[3], EMail=p[4], AlternateEmail=p[5], Phone=p[6]))
	>>> #Convert the RDD into data frame
	>>> userDataDF = userDataRDD.toDF()
	>>> userDataDF.createOrReplaceTempView("user")
	>>> userDataDF.show()
      +----------------+--------------------+---------+---+--------+--------------+--------+

      |  AlternateEmail|               EMail|FirstName| Id|LastName|         Phone|UserName|

      +----------------+--------------------+---------+---+--------+--------------+--------+

      |mt12@example.com| mthomas@example.com|     Mark|  1|  Thomas|+4411860297701| mthomas|

      | mit@example.com|mithomas@example.com|  Michael|  2|  Thomas|+4411860297702|mithomas|

      | mtw@example.com|  mtwain@example.com|     Mark|  3|   Twain|+4411860297703|  mtwain|

      |  th@example.com|  thardy@example.com|   Thomas|  4|   Hardy|+4411860297704|  thardy|

      |  bb@example.com| wbryson@example.com|  William|  5|  Bryson|+4411860297705| wbryson|

      |  wb@example.com|   wbrad@example.com|  William|  6|Bradford|+4411860297706|   wbrad|

      |  eh@example.com| eharris@example.com|       Ed|  7|  Harris|+4411860297707| eharris|

      |  tk@example.com|   tcook@example.com|   Thomas|  8|    Cook|+4411860297708|   tcook|

      |  ar@example.com| arobert@example.com|     Adam|  9|  Robert|+4411860297709| arobert|

      |  jj@example.com|  jjames@example.com|    Jacob| 10|   James|+4411860297710|  jjames|

      +----------------+--------------------+---------+---+--------+--------------+--------+

	>>> #Load the follower data into an RDD
	>>> followerDataRDD = sc.textFile(dataDir + "follower.txt").map(lambda line: line.split("|")).map(lambda p: Row(Follower=p[0], Followed=p[1]))
	>>> #Convert the RDD into data frame
	>>> followerDataDF = followerDataRDD.toDF()
	>>> followerDataDF.createOrReplaceTempView("follow")
	>>> followerDataDF.show()
      +--------+--------+

      |Followed|Follower|

      +--------+--------+

      |mithomas| mthomas|

      |  mtwain| mthomas|

      | wbryson|  thardy|

      | wbryson|   wbrad|

      | mthomas| eharris|

      |   tcook| eharris|

      |  jjames| arobert|

      +--------+--------+

	>>> #Load the message data into an RDD
	>>> messageDataRDD = sc.textFile(dataDir + "message.txt").map(lambda line: line.split("|")).map(lambda p: Row(UserName=p[0], MessageId=int(p[1]), ShortMessage=p[2], Timestamp=int(p[3])))
	>>> #Convert the RDD into data frame
	>>> messageDataDF = messageDataRDD.toDF()
	>>> messageDataDF.createOrReplaceTempView("message")
	>>> messageDataDF.show()
      +---------+--------------------+----------+--------+

      |MessageId|        ShortMessage| Timestamp|UserName|

      +---------+--------------------+----------+--------+

      |        1|@mithomas Your po...|1459009608| mthomas|

      |        2|Feeling awesome t...|1459010608| mthomas|

      |        3|My namesake in th...|1459010776|  mtwain|

      |        4|Started the day w...|1459011016|  mtwain|

      |        5|It is just spring...|1459011199|  thardy|

      |        6|Some days are rea...|1459011256| wbryson|

      |        7|@wbryson Stuff ha...|1459011333|   wbrad|

      |        8|Anybody knows goo...|1459011426| eharris|

      |        9|Stock market is p...|1459011483|   tcook|

      |       10|Dont do day tradi...|1459011539|   tcook|

      |       11|I have never hear...|1459011622|   tcook|

      |       12|#Barcelona has pl...|1459157132|   wbrad|

      |       13|@wbryson It is go...|1459164906|  mtwain|

      +---------+--------------------+----------+--------+ 

```

这些步骤完成了将所有必需数据从持久存储加载到 DataFrames 的过程。这里，数据来自文本文件。在实际应用中，数据可能来自流行的 NoSQL 数据存储、传统 RDBMS 表，或是从 HDFS 加载的 Avro 或 Parquet 序列化数据存储。以下部分使用这些 DataFrames 创建了各种目标视图和查询：

```scala
 >>> #Create the purposed view of the message to users
	>>> messagetoUsersRDD = messageDataRDD.filter(lambda message: "@" in message.ShortMessage).map(lambda message : (message, " ".join(filter(lambda s: s[0] == '@', message.ShortMessage.split(" "))))).map(lambda msgTuple: Row(FromUserName=msgTuple[0].UserName, ToUserName=msgTuple[1][1:], MessageId=msgTuple[0].MessageId, ShortMessage=msgTuple[0].ShortMessage, Timestamp=msgTuple[0].Timestamp))
	>>> #Convert the RDD into data frame
	>>> messagetoUsersDF = messagetoUsersRDD.toDF()
	>>> messagetoUsersDF.createOrReplaceTempView("messageToUsers")
	>>> messagetoUsersDF.show()
      +------------+---------+--------------------+----------+----------+

      |FromUserName|MessageId|        ShortMessage| Timestamp|ToUserName|

      +------------+---------+--------------------+----------+----------+

      |     mthomas|        1|@mithomas Your po...|1459009608|  mithomas|

      |       wbrad|        7|@wbryson Stuff ha...|1459011333|   wbryson|

      |      mtwain|       13|@wbryson It is go...|1459164906|   wbryson|

      +------------+---------+--------------------+----------+----------+

	>>> #Create the purposed view of tagged messages 
	>>> taggedMessageRDD = messageDataRDD.filter(lambda message: "#" in message.ShortMessage).map(lambda message : (message, " ".join(filter(lambda s: s[0] == '#', message.ShortMessage.split(" "))))).map(lambda msgTuple: Row(HashTag=msgTuple[1], UserName=msgTuple[0].UserName, MessageId=msgTuple[0].MessageId, ShortMessage=msgTuple[0].ShortMessage, Timestamp=msgTuple[0].Timestamp))
	>>> #Convert the RDD into data frame
	>>> taggedMessageDF = taggedMessageRDD.toDF()
	>>> taggedMessageDF.createOrReplaceTempView("taggedMessages")
	>>> taggedMessageDF.show()
      +----------+---------+--------------------+----------+--------+

      |   HashTag|MessageId|        ShortMessage| Timestamp|UserName|

      +----------+---------+--------------------+----------+--------+

      |#Barcelona|        8|Anybody knows goo...|1459011426| eharris|

      |#Barcelona|       12|#Barcelona has pl...|1459157132|   wbrad|

      +----------+---------+--------------------+----------+--------+

	>>> #The following are the queries given in the use cases
	>>> #Find the messages that are grouped by a given hash tag
	>>> byHashTag = spark.sql("SELECT a.UserName, b.FirstName, b.LastName, a.MessageId, a.ShortMessage, a.Timestamp FROM taggedMessages a, user b WHERE a.UserName = b.UserName AND HashTag = '#Barcelona' ORDER BY a.Timestamp")
	>>> byHashTag.show()
      +--------+---------+--------+---------+--------------------+----------+

      |UserName|FirstName|LastName|MessageId|        ShortMessage| Timestamp|

      +--------+---------+--------+---------+--------------------+----------+

      | eharris|       Ed|  Harris|        8|Anybody knows goo...|1459011426|

      |   wbrad|  William|Bradford|       12|#Barcelona has pl...|1459157132|

      +--------+---------+--------+---------+--------------------+----------+

	>>> #Find the messages that are addressed to a given user
	>>> byToUser = spark.sql("SELECT FromUserName, ToUserName, MessageId, ShortMessage, Timestamp FROM messageToUsers WHERE ToUserName = 'wbryson' ORDER BY Timestamp")
	>>> byToUser.show()
      +------------+----------+---------+--------------------+----------+

      |FromUserName|ToUserName|MessageId|        ShortMessage| Timestamp|

      +------------+----------+---------+--------------------+----------+

      |       wbrad|   wbryson|        7|@wbryson Stuff ha...|1459011333|

      |      mtwain|   wbryson|       13|@wbryson It is go...|1459164906|

      +------------+----------+---------+--------------------+----------+

	>>> #Find the followers of a given user
	>>> followers = spark.sql("SELECT b.FirstName as FollowerFirstName, b.LastName as FollowerLastName, a.Followed FROM follow a, user b WHERE a.Follower = b.UserName AND a.Followed = 'wbryson'")>>> followers.show()
      +-----------------+----------------+--------+

      |FollowerFirstName|FollowerLastName|Followed|

      +-----------------+----------------+--------+

      |          William|        Bradford| wbryson|

      |           Thomas|           Hardy| wbryson|

      +-----------------+----------------+--------+

	>>> #Find the followed users of a given user
	>>> followedUsers = spark.sql("SELECT b.FirstName as FollowedFirstName, b.LastName as FollowedLastName, a.Follower FROM follow a, user b WHERE a.Followed = b.UserName AND a.Follower = 'eharris'")
	>>> followedUsers.show()
      +-----------------+----------------+--------+

      |FollowedFirstName|FollowedLastName|Follower|

      +-----------------+----------------+--------+

      |           Thomas|            Cook| eharris|

      |             Mark|          Thomas| eharris| 
 +-----------------+----------------+--------+ 

```

为了实现用例，提议的视图和查询被开发为一个单一的应用程序。但实际上，将所有视图和查询集中在一个应用程序中并不是一个好的设计实践。通过持久化视图并在定期间隔内刷新它们来分离它们是更好的做法。如果仅使用一个应用程序，可以通过缓存和使用自定义制作的环境对象来访问视图，这些对象会被广播到 Spark 集群。

# 理解自定义数据处理流程

此处创建的视图旨在服务于各种查询并产生期望的输出。还有其他一些数据处理应用程序类别，通常是为了实现现实世界的用例而开发的。从 Lambda 架构的角度来看，这也属于服务层。这些自定义数据处理流程之所以属于服务层，主要是因为它们大多使用或处理来自主数据集的数据，并创建视图或输出。自定义处理的数据也很可能保持为视图，下面的用例就是其中之一。

在 SfbMicroBlog 微博应用程序中，一个非常常见的需求是查看给定用户 A 是否以某种方式与用户 B 直接或间接相连。此用例可以通过使用图数据结构来实现，以查看这两个用户是否在同一连接组件中，是否以传递方式相连，或者是否根本不相连。为此，使用 Spark GraphX 库构建了一个图，其中所有用户作为顶点，关注关系作为边。在 Scala REPL 提示符下，尝试以下语句：

```scala
 scala> import org.apache.spark.rdd.RDD
    import org.apache.spark.rdd.RDD    
	scala> import org.apache.spark.graphx._
    import org.apache.spark.graphx._    
	scala> //TODO: Change the following directory to point to your data directory
	scala> val dataDir = "/Users/RajT/Documents/Writing/SparkForBeginners/To-PACKTPUB/Contents/B05289-09-DesigningSparkApplications/Code/Data/"
dataDir: String = /Users/RajT/Documents/Writing/SparkForBeginners/To-PACKTPUB/Contents/B05289-09-DesigningSparkApplications/Code/Data/

	scala> //Define the case classes in Scala for the entities
	scala> case class User(Id: Long, UserName: String, FirstName: String, LastName: String, EMail: String, AlternateEmail: String, Phone: String)
      defined class User

	scala> case class Follow(Follower: String, Followed: String)
      defined class Follow

	scala> case class ConnectedUser(CCId: Long, UserName: String)
      defined class ConnectedUser

	scala> //Define the utility functions that are to be passed in the applications
	scala> def toUser =  (line: Seq[String]) => User(line(0).toLong, line(1), line(2),line(3), line(4), line(5), line(6))
      toUser: Seq[String] => User

	scala> def toFollow =  (line: Seq[String]) => Follow(line(0), line(1))
      toFollow: Seq[String] => Follow

	scala> //Load the user data into an RDD
	scala> val userDataRDD = sc.textFile(dataDir + "user.txt").map(_.split("\\|")).map(toUser(_))
userDataRDD: org.apache.spark.rdd.RDD[User] = MapPartitionsRDD[160] at map at <console>:34

	scala> //Convert the RDD into data frame
	scala> val userDataDF = userDataRDD.toDF()
userDataDF: org.apache.spark.sql.DataFrame = [Id: bigint, UserName: string ... 5 more fields]

	scala> userDataDF.createOrReplaceTempView("user")
	scala> userDataDF.show()
      +---+--------+---------+--------+-----------+----------------+--------------+

Id|UserName|FirstName|LastName| EMail|  AlternateEmail|   Phone|

      +---+--------+---------+--------+----------+-------------+--------------+

|  1| mthomas|     Mark|  Thomas| mthomas@example.com|mt12@example.com|
+4411860297701|

|  2|mithomas|  Michael|  Thomas|mithomas@example.com| mit@example.com|
+4411860297702|

|  3|  mtwain|     Mark|   Twain|  mtwain@example.com| mtw@example.com|
+4411860297703|

|  4|  thardy|   Thomas|   Hardy|  thardy@example.com|  th@example.com|
+4411860297704|

|  5| wbryson|  William|  Bryson| wbryson@example.com|  bb@example.com|
+4411860297705|

|  6|   wbrad|  William|Bradford|   wbrad@example.com|  wb@example.com|
+4411860297706|

|  7| eharris|       Ed|  Harris| eharris@example.com|  eh@example.com|
+4411860297707|

|  8|   tcook|   Thomas|    Cook|   tcook@example.com|  tk@example.com|
+4411860297708|

|  9| arobert|     Adam|  Robert| arobert@example.com|  ar@example.com|
+4411860297709|

| 10|  jjames|    Jacob|   James|  jjames@example.com|  jj@example.com|
+4411860297710|    
      +---+--------+---------+--------+-------------+--------------+--------------+

	scala> //Load the follower data into an RDD
	scala> val followerDataRDD = sc.textFile(dataDir + "follower.txt").map(_.split("\\|")).map(toFollow(_))
followerDataRDD: org.apache.spark.rdd.RDD[Follow] = MapPartitionsRDD[168] at map at <console>:34

	scala> //Convert the RDD into data frame
	scala> val followerDataDF = followerDataRDD.toDF()
followerDataDF: org.apache.spark.sql.DataFrame = [Follower: string, Followed: string]

	scala> followerDataDF.createOrReplaceTempView("follow")
	scala> followerDataDF.show()
      +--------+--------+

      |Follower|Followed|

      +--------+--------+

      | mthomas|mithomas|

      | mthomas|  mtwain|

      |  thardy| wbryson|

      |   wbrad| wbryson|

      | eharris| mthomas|

      | eharris|   tcook|

      | arobert|  jjames|

      +--------+--------+

	scala> //By joining with the follower and followee users with the master user data frame for extracting the unique ids
	scala> val fullFollowerDetails = spark.sql("SELECT b.Id as FollowerId, c.Id as FollowedId, a.Follower, a.Followed FROM follow a, user b, user c WHERE a.Follower = b.UserName AND a.Followed = c.UserName")
fullFollowerDetails: org.apache.spark.sql.DataFrame = [FollowerId: bigint, FollowedId: bigint ... 2 more fields]

	scala> fullFollowerDetails.show()
      +----------+----------+--------+--------+

      |FollowerId|FollowedId|Follower|Followed|

      +----------+----------+--------+--------+

      |         9|        10| arobert|  jjames|

      |         1|         2| mthomas|mithomas|

      |         7|         8| eharris|   tcook|

      |         7|         1| eharris| mthomas|

      |         1|         3| mthomas|  mtwain|

      |         6|         5|   wbrad| wbryson|

      |         4|         5|  thardy| wbryson|

      +----------+----------+--------+--------+

	scala> //Create the vertices of the connections graph
	scala> val userVertices: RDD[(Long, String)] = userDataRDD.map(user => (user.Id, user.UserName))
userVertices: org.apache.spark.rdd.RDD[(Long, String)] = MapPartitionsRDD[194] at map at <console>:36

	scala> userVertices.foreach(println)
      (6,wbrad)

      (7,eharris)

      (8,tcook)

      (9,arobert)

      (10,jjames)

      (1,mthomas)

      (2,mithomas)

      (3,mtwain)

      (4,thardy)

      (5,wbryson)

	scala> //Create the edges of the connections graph 
	scala> val connections: RDD[Edge[String]] = fullFollowerDetails.rdd.map(conn => Edge(conn.getAsLong, conn.getAsLong, "Follows"))
      connections: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[String]] = MapPartitionsRDD[217] at map at <console>:29

	scala> connections.foreach(println)
	Edge(9,10,Follows)
	Edge(7,8,Follows)
	Edge(1,2,Follows)
	Edge(7,1,Follows)
	Edge(1,3,Follows)
	Edge(6,5,Follows)
	Edge(4,5,Follows)
	scala> //Create the graph using the vertices and the edges
	scala> val connectionGraph = Graph(userVertices, connections)
      connectionGraph: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@3c207acd 

```

用户图已经完成，其中用户位于顶点，连接关系形成边。在此图数据结构上，运行图处理算法，即连接组件算法。以下代码片段实现了这一点：

```scala
 scala> //Calculate the connected users
	scala> val cc = connectionGraph.connectedComponents()
      cc: org.apache.spark.graphx.Graph[org.apache.spark.graphx.VertexId,String] = org.apache.spark.graphx.impl.GraphImpl@73f0bd11

	scala> // Extract the triplets of the connected users
	scala> val ccTriplets = cc.triplets
      ccTriplets: org.apache.spark.rdd.RDD[org.apache.spark.graphx.EdgeTriplet[org.apache.spark.graphx.VertexId,String]] = MapPartitionsRDD[285] at mapPartitions at GraphImpl.scala:48

	scala> // Print the structure of the triplets
	scala> ccTriplets.foreach(println)
      ((9,9),(10,9),Follows)

      ((1,1),(2,1),Follows)

      ((7,1),(8,1),Follows)

      ((7,1),(1,1),Follows)

      ((1,1),(3,1),Follows)

      ((4,4),(5,4),Follows) 
 ((6,4),(5,4),Follows) 

```

创建了连接组件图`cc`及其三元组`ccTriplets`，现在可以使用它们来运行各种查询。由于图是基于 RDD 的数据结构，如果需要进行查询，将图 RDD 转换为 DataFrames 是一种常见做法。以下代码演示了这一点：

```scala
 scala> //Print the vertex numbers and the corresponding connected component id. The connected component id is generated by the system and it is to be taken only as a unique identifier for the connected component
   scala> val ccProperties = ccTriplets.map(triplet => "Vertex " + triplet.srcId + " and " + triplet.dstId + " are part of the CC with id " + triplet.srcAttr)
      ccProperties: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[288] at map at <console>:48

	scala> ccProperties.foreach(println)
      Vertex 9 and 10 are part of the CC with id 9

      Vertex 1 and 2 are part of the CC with id 1

      Vertex 7 and 8 are part of the CC with id 1

      Vertex 7 and 1 are part of the CC with id 1

      Vertex 1 and 3 are part of the CC with id 1

      Vertex 4 and 5 are part of the CC with id 4

      Vertex 6 and 5 are part of the CC with id 4

	scala> //Find the users in the source vertex with their CC id
	scala> val srcUsersAndTheirCC = ccTriplets.map(triplet => (triplet.srcId, triplet.srcAttr))
      srcUsersAndTheirCC: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, org.apache.spark.graphx.VertexId)] = MapPartitionsRDD[289] at map at <console>:48

	scala> //Find the users in the destination vertex with their CC id
	scala> val dstUsersAndTheirCC = ccTriplets.map(triplet => (triplet.dstId, triplet.dstAttr))
      dstUsersAndTheirCC: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, org.apache.spark.graphx.VertexId)] = MapPartitionsRDD[290] at map at <console>:48

	scala> //Find the union
	scala> val usersAndTheirCC = srcUsersAndTheirCC.union(dstUsersAndTheirCC)
      usersAndTheirCC: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, org.apache.spark.graphx.VertexId)] = UnionRDD[291] at union at <console>:52

	scala> //Join with the name of the users
	scala> //Convert the RDD to DataFrame
	scala> val usersAndTheirCCWithName = usersAndTheirCC.join(userVertices).map{case (userId,(ccId,userName)) => (ccId, userName)}.distinct.sortByKey().map{case (ccId,userName) => ConnectedUser(ccId, userName)}.toDF()
      usersAndTheirCCWithName: org.apache.spark.sql.DataFrame = [CCId: bigint, UserName: string]

	scala> usersAndTheirCCWithName.createOrReplaceTempView("connecteduser")
	scala> val usersAndTheirCCWithDetails = spark.sql("SELECT a.CCId, a.UserName, b.FirstName, b.LastName FROM connecteduser a, user b WHERE a.UserName = b.UserName ORDER BY CCId")
      usersAndTheirCCWithDetails: org.apache.spark.sql.DataFrame = [CCId: bigint, UserName: string ... 2 more fields]

	scala> //Print the usernames with their CC component id. If two users share the same CC id, then they are connected
	scala> usersAndTheirCCWithDetails.show()
      +----+--------+---------+--------+

      |CCId|UserName|FirstName|LastName|

      +----+--------+---------+--------+

      |   1|mithomas|  Michael|  Thomas|

      |   1|  mtwain|     Mark|   Twain|

      |   1|   tcook|   Thomas|    Cook|

      |   1| eharris|       Ed|  Harris|

      |   1| mthomas|     Mark|  Thomas|

      |   4|   wbrad|  William|Bradford|

      |   4| wbryson|  William|  Bryson|

      |   4|  thardy|   Thomas|   Hardy|

      |   9|  jjames|    Jacob|   James|

      |   9| arobert|     Adam|  Robert| 
 +----+--------+---------+--------+ 

```

使用上述有目的的视图实现来获取用户列表及其连接组件标识号，如果需要查明两个用户是否相连，只需读取这两个用户的记录，并查看它们是否具有相同的连接组件标识号。

# 参考资料

如需更多信息，请访问以下链接：

+   [Lambda 架构网](http://lambda-architecture.net/)

+   [Context-Object-Pattern.pdf](https://www.dre.vanderbilt.edu/~schmidt/PDF/Context-Object-Pattern.pdf)

# 摘要

本章以单一应用程序的使用案例作为全书的结尾，这些案例是利用本书前面章节学到的 Spark 概念实现的。从数据处理应用架构的角度来看，本章介绍了 Lambda 架构作为一种与技术无关的数据处理应用架构框架，在大数据应用开发领域具有巨大的适用性。

从数据处理应用开发的角度来看，涵盖了基于 RDD 的 Spark 编程、基于 Dataset 的 Spark 编程、基于 Spark SQL 的 DataFrames 处理结构化数据、基于 Spark Streaming 的监听程序持续监听传入消息并处理它们，以及基于 Spark GraphX 的应用程序处理关注者关系。到目前为止所涵盖的使用案例为读者提供了广阔的空间，以添加自己的功能并增强本章讨论的应用程序用例。
