# 第十一章：构建数据科学应用

数据科学应用引起了很多兴奋，主要是因为它们在利用数据和提取可消费的结果方面的承诺。已经有几个成功的数据产品对我们的日常生活产生了变革性影响。无处不在的推荐系统、电子邮件垃圾邮件过滤器、定向广告和新闻内容已经成为生活的一部分。音乐和电影已经成为来自 iTunes 和 Netflix 等提供商的数据产品。企业，特别是在零售等领域，正在积极寻求通过使用数据驱动的方法来研究市场和客户行为以获得竞争优势的方式。

到目前为止，在前几章中，我们已经讨论了数据分析工作流程直到模型构建阶段。但是模型真正的价值在于它实际部署到生产系统中。数据科学工作流的最终产品是一个操作化的数据产品。在本章中，我们将讨论数据分析工作流的这个最终阶段。我们不会涉及实际的代码片段，而是退一步，获取完整的画面，包括非技术方面。

完整的画面不仅仅局限于开发过程本身。它包括用户应用程序、Spark 本身的发展，以及大数据领域正在发生的快速变化。我们将首先从用户应用程序的开发过程开始，讨论每个阶段的各种选项。然后我们将深入探讨最新的 Spark 2.0 版本和未来计划中的功能和增强功能。最后，我们将尝试对大数据趋势，特别是 Hadoop 生态系统，进行广泛的概述。参考资料和有用的链接将包括在各个部分中，以及章节末尾，以获取有关特定上下文的更多信息。

# 开发范围

数据分析工作流程大致可以分为两个阶段，即构建阶段和操作化阶段。第一阶段通常是一次性的练习，需要大量人为干预。一旦我们获得了合理的最终结果，我们就准备好将产品操作化。第二阶段从第一阶段生成的模型开始，并将其作为某个生产工作流程的一部分提供。在本节中，我们将讨论以下内容：

+   期望

+   演示选项

+   开发和测试

+   数据质量管理

## 期望

数据科学应用的主要目标是构建“可操作”的洞察力，可操作是关键词。许多用例，如欺诈检测，需要以可消费的方式生成并在几乎实时中提供洞察力，如果您期望有任何可操作性。数据产品的最终用户因用例而异。他们可能是电子商务网站的客户，也可能是大型企业的决策者。最终用户不一定总是人类。它可能是金融机构中的风险评估软件工具。一刀切的方法并不适用于许多软件产品，数据产品也不例外。然而，对于数据产品，有一些共同的期望，如下所列：

+   首要和最重要的期望是基于现实世界数据的洞察力生成时间应该在“可操作”的时间范围内。实际时间范围根据用例而异。

+   数据产品应该整合到一些（通常已经存在的）生产工作流程中。

+   洞察力应该被转化为人们可以使用的东西，而不是晦涩的数字或难以解释的图表。演示应该是不显眼的。

+   数据产品应该能够根据输入的数据自我调整（自适应）。

+   理想情况下，应该有一种方式来接收人类反馈，这可以作为自我调整的来源。

+   应该有一个定期和自动地定量评估其有效性的机制。

## 演示选项

数据产品的多样性要求多样的展示方式。有时，数据分析练习的最终结果是发表研究论文。有时它可能是仪表板的一部分，其中它成为单个网页上发布结果的几个来源之一。它们可能是公开的，针对人类消费，也可能是隐蔽的，供其他软件应用程序使用。您可以使用通用引擎，如 Spark 来构建解决方案，但展示必须与目标用户群高度对齐。

有时，您只需要写一封电子邮件，附上您的发现，或者只是导出一个洞察力的 CSV 文件。或者您可能需要围绕您的数据产品开发一个专用的 Web 应用程序。这里讨论了一些其他常见的选项，您必须选择适合手头问题的正确选项。

### 交互式笔记本

交互式笔记本是允许您创建和共享包含代码块、结果、方程、图像、视频和解释文本的文档的 Web 应用程序。它们可以被视为可执行文档或带有可视化和方程支持的 REPL shell。这些文档可以导出为 PDF、Markdown 或 HTML。笔记本包含多个“内核”或“计算引擎”来执行代码块。

如果您的数据分析工作流的最终目标是生成书面报告，交互式笔记本是最合适的选择。有几种笔记本，其中许多都支持 Spark。这些笔记本在探索阶段也是有用的工具。我们在之前的章节中已经介绍了 IPython 和 Zeppelin 笔记本。

#### 参考资料

+   IPython Notebook：数据科学的综合工具：[`conferences.oreilly.com/strata/strata2013/public/schedule/detail/27233`](http://conferences.oreilly.com/strata/strata2013/public/schedule/detail/27233)

+   Sparkly Notebook：使用 Spark 进行交互式分析和可视化：[`www.slideshare.net/felixcss/sparkly-notebook-interactive-analysis-and-visualization-with-spark`](http://www.slideshare.net/felixcss/sparkly-notebook-interactive-analysis-and-visualization-with-spark)

### Web API

**应用程序编程接口**（**API**）是软件到软件的接口；描述可用功能、如何使用以及输入和输出的规范。软件（服务）提供者将其部分功能公开为 API。开发人员可以开发一个消耗此 API 的软件组件。例如，Twitter 提供 API 以在 Twitter 上获取或发布数据，或以编程方式查询数据。Spark 爱好者可以编写一个软件组件，自动收集所有关于#Spark 的推文，根据他们的要求进行分类，并在其个人网站上发布这些数据。Web API 是 API 的一种类型，其中接口被定义为一组**超文本传输协议**（**HTTP**）请求消息以及响应消息结构的定义。如今，REST-ful（表述性状态转移）已成为事实上的标准。

您可以将数据产品实现为 API，也许这是最强大的选择。然后可以将其插入一个或多个应用程序，比如管理仪表板以及营销分析工作流。您可以开发一个特定领域的“见解即服务”作为公共 Web API，并采用订阅模式。Web API 的简单性和普遍性使其成为构建数据产品的最具吸引力的选择。

#### 参考资料

+   应用程序编程接口：[`en.wikipedia.org/wiki/Application_programming_interface`](https://en.wikipedia.org/wiki/Application_programming_interface)

+   准备好使用 API 了吗？解锁数据经济最有前途的渠道的三个步骤：[`www.forbes.com/sites/mckinsey/2014/01/07/ready-for-apis-three-steps-to-unlock-the-data-economys-most-promising-channel/#61e7103b89e5`](http://www.forbes.com/sites/mckinsey/2014/01/07/ready-for-apis-three-steps-to-unlock-the-data-economys-most-promising-channel/#61e7103b89e5)

+   基于大数据增长的洞察作为服务：[`www.kdnuggets.com/2015/12/insights-as-a-service-big-data.html`](http://www.kdnuggets.com/2015/12/insights-as-a-service-big-data.html)

### PMML 和 PFA

有时，您可能需要以其他数据挖掘工具能够理解的方式公开您的模型。模型和完整的预处理和后处理步骤应转换为标准格式。PMML 和 PFA 是数据挖掘领域的两种标准格式。

**预测模型标记语言**（**PMML**）是基于 XML 的预测模型交换格式，Apache Spark API 可以将模型转换为 PMML。 PMML 消息可能包含多种数据转换以及一个或多个预测模型。不同的数据挖掘工具可以导出或导入 PMML 消息，无需自定义代码。

**Analytics 的便携格式**（**PFA**）是下一代预测模型交换格式。它交换 JSON 文档，并立即继承 JSON 文档相对于 XML 文档的所有优势。此外，PFA 比 PMML 更灵活。

#### 参考资料

+   PMML FAQ：预测模型标记语言：[`www.kdnuggets.com/2013/01/pmml-faq-predictive-model-markup-language.html`](http://www.kdnuggets.com/2013/01/pmml-faq-predictive-model-markup-language.html)

+   Analytics 的便携格式：将模型移至生产：[`www.kdnuggets.com/2016/01/portable-format-analytics-models-production.html`](http://www.kdnuggets.com/2016/01/portable-format-analytics-models-production.html)

+   PFA 的用途是什么？：[`dmg.org/pfa/docs/motivation/`](http://dmg.org/pfa/docs/motivation/)

## 开发和测试

Apache Spark 是一种通用的集群计算系统，可以独立运行，也可以在几个现有的集群管理器上运行，如 Apache Mesos、Hadoop、Yarn 和 Amazon EC2。此外，一些大数据和企业软件公司已经将 Spark 集成到其产品中：Microsoft Azure HDInsight、Cloudera、IBM Analytics for Apache Spark、SAP HANA 等等。由 Apache Spark 的创始人创立的 Databricks 公司拥有自己的产品，用于数据科学工作流程，从数据摄取到生产。您的责任是了解组织的要求和现有的人才储备，并决定哪个选项对您最好。

无论选择哪个选项，都要遵循任何软件开发生命周期中的通常最佳实践，例如版本控制和同行审查。尽量在适用的地方使用高级 API。生产中使用的数据转换流水线应与构建模型时使用的流水线相同。记录在数据分析工作流程中出现的任何问题。通常这些问题可能导致业务流程改进。

一如既往，测试对于产品的成功非常重要。您必须维护一组自动化脚本，以提供易于理解的结果。测试用例应至少涵盖以下内容：

+   遵守时间框架和资源消耗要求

+   对不良数据的弹性（例如，数据类型违规）

+   在模型构建阶段未遇到的分类特征中的新值

+   在目标生产系统中预期的非常少的数据或过重的数据

监视日志、资源利用率等，以发现任何性能瓶颈。Spark UI 提供了大量信息来监视 Spark 应用程序。以下是一些常见的提示，将帮助您提高性能：

+   缓存可能多次使用的任何输入或中间数据。

+   查看 Spark UI 并识别导致大量洗牌的作业。检查代码，看看是否可以减少洗牌。

+   操作可能会将数据从工作节点传输到驱动程序。请注意，您不要传输任何绝对不必要的数据。

+   Stragglers；比其他任务运行速度慢；可能会增加整体作业完成时间。出现任务运行缓慢可能有几个原因。如果作业因为一个慢节点而运行缓慢，您可以将`spark.speculation`设置为`true`。然后 Spark 会自动在不同节点上重新启动这样的任务。否则，您可能需要重新审视逻辑，看看是否可以改进。

### 参考

+   调查 Spark 的性能：[`radar.oreilly.com/2015/04/investigating-sparks-performance.html`](http://radar.oreilly.com/2015/04/investigating-sparks-performance.html)

+   Patrick Wendell 的《Apache Spark 的调整和调试》：[`sparkhub.databricks.com/video/tuning-and-debugging-apache-spark/`](https://sparkhub.databricks.com/video/tuning-and-debugging-apache-spark/)

+   如何调整 Apache Spark 作业：http://blog.cloudera.com/blog/2015/03/how-to-tune-your-apache-spark-jobs-part-1/和 part 2

## 数据质量管理

首先，让我们不要忘记，我们正在尝试从不可靠、通常是非结构化和不受控的数据源构建容错的软件数据产品。因此，在数据科学工作流程中，数据质量管理变得更加重要。有时数据可能仅来自受控数据源，例如组织中的自动化内部流程工作流。但在所有其他情况下，您需要仔细制定数据清洗流程，以保护后续处理。

元数据包括数据的结构和含义，显然是要处理的最关键的存储库。它是关于单个数据源结构和该结构中每个组件含义的信息。您可能并不总是能够编写一些脚本并提取这些数据。单个数据源可能包含具有不同结构的数据，或者单个组件（列）在不同时间可能意味着不同的事情。例如，标签如所有者或高在不同的数据源中可能意味着不同的事情。收集和理解所有这样的细微差别并记录是一项繁琐的迭代任务。元数据的标准化是数据转换开发的先决条件。

适用于大多数用例的一些广泛指导原则列在这里：

+   所有数据源必须进行版本控制和时间戳标记

+   数据质量管理过程通常需要最高管理机构的参与

+   屏蔽或匿名化敏感数据

+   经常被忽视的一个重要步骤是保持可追溯性；每个数据元素（比如一行）与其原始来源之间的链接

# Scala 的优势

Apache Spark 允许您使用 Python、R、Java 或 Scala 编写应用程序。这种灵活性带来了选择适合您需求的正确语言的责任。但无论您通常选择的语言是什么，您可能会考虑为您的 Spark 应用程序选择 Scala。在本节中，我们将解释为什么。

首先，让我们离题一下，以便对命令式和函数式编程范式有一个高层次的理解。像 C、Python 和 Java 这样的语言属于命令式编程范式。在命令式编程范式中，程序是一系列指令，它有一个程序状态。程序状态通常表示为一组变量及其在任何给定时间点的值。赋值和重新赋值是相当常见的。变量的值预计会在执行期间由一个或多个函数改变。函数中的变量值修改不仅限于局部变量。全局变量和公共类变量是这些变量的一些例子。

相比之下，使用函数式编程语言编写的程序可以被视为无状态的表达式求值器。数据是不可变的。如果一个函数使用相同的输入参数集合调用，那么预期会产生相同的结果（即引用透明）。这是由于全局变量等变量上下文的干扰的缺失。这意味着函数求值的顺序并不重要。函数可以作为参数传递给其他函数。递归调用取代了循环。无状态使得并行编程更容易，因为它消除了锁定和可能的死锁的需要。当执行顺序不那么重要时，协调变得更加简单。这些因素使得函数式编程范式非常适合并行编程。

纯函数式编程语言很难使用，因为大多数程序需要状态改变。大多数函数式编程语言，包括老式的 Lisp，都允许在变量中存储数据（副作用）。一些语言，如 Scala，汲取了多种编程范式。

回到 Scala，它是一种基于 JVM 的、静态类型的多范式编程语言。它的内置类型推断机制允许程序员省略一些冗余的类型信息。这给人一种动态语言所提供的灵活性的感觉，同时保留了更好的编译时检查和快速运行时的健壮性。Scala 是一种面向对象的语言，因为每个值都是一个对象，包括数值值。函数是一级对象，可以用作任何数据类型，并且可以作为参数传递给其他函数。Scala 与 Java 及其工具很好地互操作，因为 Scala 在 JVM 上运行。Java 和 Scala 类可以自由混合。这意味着 Scala 可以轻松地与 Hadoop 生态系统进行交互。

选择适合你的应用程序的编程语言时，所有这些因素都应该被考虑进去。

# Spark 的开发状态

Apache Spark 已成为截至 2015 年底 Hadoop 生态系统中活跃度最高的项目，以贡献者数量计算。Spark 始于 2009 年的加州大学伯克利分校 AMPLAB 的一个研究项目，与 Apache Hadoop 等项目相比，Spark 仍然相对年轻，并且仍在积极开发中。2015 年有三个版本发布，从 1.3 到 1.5，分别包含了 DataFrames API、SparkR 和 Project Tungsten 等功能。1.6 版本于 2016 年初发布，包括新的 Dataset API 和数据科学功能的扩展。Spark 2.0 于 2016 年 7 月发布，作为一个重大版本，具有许多新功能和增强功能，值得单独介绍。

## Spark 2.0 的功能和增强

Apache Spark 2.0 包括三个主要的新功能和其他性能改进和底层更改。本节试图给出一个高层次的概述，同时在必要时深入细节，以便理解概念。

### 统一 Datasets 和 DataFrames

DataFrames 是支持数据抽象概念上等同于关系数据库中的表或 R 和 Python 中的 DataFrame（pandas 库）的高级 API。Datasets 是 DataFrame API 的扩展，提供了一个类型安全的、面向对象的编程接口。Datasets 为 DataFrames 添加了静态类型。在 DataFrames 之上定义结构提供了信息给核心，从而实现了优化。它还有助于在分布式作业开始之前及早捕捉分析错误。

RDD、Dataset 和 DataFrame 是可互换的。RDD 仍然是低级 API。DataFrame、Dataset 和 SQL 共享相同的优化和执行管道。机器学习库可以使用 DataFrame 或 Dataset。DataFrame 和 Dataset 都在钨上运行，这是一个改进运行时性能的倡议。它们利用钨的快速内存编码，负责在 JVM 对象和 Spark 内部表示之间进行转换。相同的 API 也适用于流，引入了连续 DataFrame 的概念。

### 结构化流

结构化流 API 是构建在 Spark SQL 引擎上并扩展了 DataFrame 和 Dataset 的高级 API。结构化流统一了流式、交互式和批处理查询。在大多数情况下，流数据需要与批处理和交互式查询相结合，形成连续的应用程序。这些 API 旨在满足这一要求。Spark 负责在流数据上增量和连续地运行查询。

结构化流的第一个版本将专注于 ETL 工作负载。用户将能够指定输入、查询、触发器和输出类型。输入流在逻辑上等同于只追加的表。用户定义查询的方式与传统 SQL 表相同。触发器是一个时间段，比如一秒。提供的输出模式包括完整输出、增量或原地更新（例如，DB 表）。

举个例子：您可以在流中聚合数据，使用 Spark SQL JDBC 服务器提供数据，并将其传递给 MySQL 等数据库进行下游应用。或者您可以运行针对最新数据的临时 SQL 查询。您还可以构建和应用机器学习模型。

### 项目钨 2 期

项目钨的核心思想是通过本机内存管理和运行时代码生成将 Spark 的性能更接近于裸金属。它首次包含在 Spark 1.4 中，并在 1.5 和 1.6 中添加了增强功能。它主要通过以下方式显着提高 Spark 应用程序的内存和 CPU 效率：

+   显式管理内存并消除 JVM 对象模型和垃圾收集的开销。例如，一个四字节的字符串在 JVM 对象模型中占用大约 48 字节。由于 Spark 不是通用应用程序，并且对内存块的生命周期比垃圾收集器更了解，因此它可以比 JVM 更有效地管理内存。

+   设计友好缓存的算法和数据结构。

+   Spark 执行代码生成以将查询的部分编译为 Java 字节码。这将被扩展以覆盖大多数内置表达式。

Spark 2.0 推出了第 2 阶段，速度提高了一个数量级，并包括：

+   通过消除昂贵的迭代器调用和在多个运算符之间融合来进行整体代码生成，从而使生成的代码看起来像手动优化的代码

+   优化的输入和输出

## 未来会有什么？

预计 Apache Spark 2.1 将具有以下功能：

+   **连续 SQL**（**CSQL**）

+   BI 应用程序集成

+   更多流式数据源和接收器的支持

+   包括用于结构化流的额外运算符和库

+   机器学习包的增强功能

+   钨中的列式内存支持

# 大数据趋势

大数据处理一直是 IT 行业的一个重要组成部分，尤其是在过去的十年中。Apache Hadoop 和其他类似的努力致力于构建存储和处理海量数据的基础设施。在经过 10 多年的发展后，Hadoop 平台被认为是成熟的，几乎可以与大数据处理等同起来。Apache Spark 是一个通用计算引擎，与 Hadoop 生态系统兼容，并且在 2015 年取得了相当大的成功。

构建数据科学应用程序需要了解大数据领域和可用软件产品。我们需要仔细地映射符合我们要求的正确模块。有几个具有重叠功能的选项，选择合适的工具说起来容易做起来难。应用程序的成功很大程度上取决于组装正确的技术和流程。好消息是，有几个开源选项可以降低大数据分析的成本；同时，你也可以选择由 Databricks 等公司支持的企业级端到端平台。除了手头的用例，跟踪行业趋势同样重要。

最近 NOSQL 数据存储的激增及其自己的接口正在添加基于 SQL 的接口，尽管它们不是关系数据存储，也可能不遵守 ACID 属性。这是一个受欢迎的趋势，因为在关系和非关系数据存储之间趋同于一个古老的接口可以提高程序员的生产力。

在过去的几十年里，操作（OLTP）和分析（OLAP）系统一直被维护为独立的系统，但这是又一个融合正在发生的地方。这种融合使我们接近实时的用例，比如欺诈预防。Apache Kylin 是 Hadoop 生态系统中的一个开源分布式分析引擎，提供了一个极快的大规模 OLAP 引擎。

物联网的出现正在加速实时和流式分析，带来了许多新的用例。云释放了组织的运营和 IT 管理开销，使他们可以集中精力在他们的核心竞争力上，特别是在大数据处理方面。基于云的分析引擎、自助数据准备工具、自助 BI、及时数据仓库、高级分析、丰富的媒体分析和敏捷分析是一些常用的词汇。大数据这个词本身正在慢慢消失或变得隐含。

在大数据领域有许多具有重叠功能的软件产品和库，如此信息图所示（http://mattturck.com/wp-content/uploads/2016/02/matt_turck_big_data_landscape_v11.png）。选择适合你的应用程序的正确模块是一项艰巨但非常重要的任务。以下是一些让你开始的项目的简短列表。该列表不包括像 Cassandra 这样的知名名称，而是试图包括具有互补功能的模块，大多来自 Apache 软件基金会：

+   **Apache Arrow**（[`arrow.apache.org/`](https://arrow.apache.org/)）是用于加速分析处理和交换的内存中的列式层。这是一个高性能、跨系统和内存数据表示，预计将带来 100 倍的性能改进。

+   **Apache Parquet**（[`parquet.apache.org/`](https://parquet.apache.org/)）是一种列式存储格式。Spark SQL 提供对读写 parquet 文件的支持，同时自动捕获数据的结构。

+   **Apache Kafka**（[`kafka.apache.org/`](http://kafka.apache.org/)）是一种流行的、高吞吐量的分布式消息系统。Spark streaming 具有直接的 API，支持从 Kafka 进行流式数据摄入。

+   **Alluxio**（[`alluxio.org/`](http://alluxio.org/)），以前称为 Tachyon，是一个以内存为中心的虚拟分布式存储系统，可以在内存速度下跨集群共享数据。它旨在成为大数据的事实存储统一层。Alluxio 位于计算框架（如 Spark）和存储系统（如 Amazon S3、HDFS 等）之间。

+   **GraphFrames**（https://databricks.com/blog/2016/03/03/introducing-graphframes.html）是建立在 DataFrames API 之上的 Apache Spark 的图处理库。

+   Apache Kylin（[`kylin.apache.org/`](http://kylin.apache.org/)）是一个分布式分析引擎，旨在提供 SQL 接口和 Hadoop 上的多维分析（OLAP）支持，支持极大的数据集。

+   Apache Sentry（[`sentry.apache.org/`](http://sentry.apache.org/)）是一个用于强制执行基于角色的细粒度授权的系统，用于存储在 Hadoop 集群上的数据和元数据。在撰写本书时，它处于孵化阶段。

+   Apache Solr（[`lucene.apache.org/solr/`](http://lucene.apache.org/solr/)）是一个极快的搜索平台。查看这个[演示](https://spark-summit.org/2015/events/integrating-spark-and-solr/)，了解如何集成 Solr 和 Spark。

+   TensorFlow（[`www.tensorflow.org/`](https://www.tensorflow.org/)）是一个具有广泛内置深度学习支持的机器学习库。查看这篇[博客](https://databricks.com/blog/2016/01/25/deep-learning-with-spark-and-tensorflow.html)了解它如何与 Spark 一起使用。

+   Zeppelin（[`zeppelin.incubator.apache.org/`](http://zeppelin.incubator.apache.org/)）是一个基于 Web 的笔记本，可以进行交互式数据分析。它在数据可视化章节中有所涉及。

# 摘要

在本章中，我们讨论了如何使用 Spark 构建真实世界的应用程序。我们讨论了由技术和非技术方面组成的数据分析工作流的大局。

# 参考资料

+   Spark Summit 网站上有大量关于 Apache Spark 和相关项目的信息，来自已完成的活动

+   KDnuggets 对 Matei Zaharia 的采访

+   2015 年，为什么 Spark 达到了临界点，来自 KDnuggets 的 Matthew Mayo

+   上线：准备您的第一个 Spark 生产部署是一个非常好的起点

+   来自 Scala 官网的“什么是 Scala？”

+   Scala 的创始人 Martin Odersky 解释了为什么 Scala 将命令式和函数式编程融合在一起的原因
