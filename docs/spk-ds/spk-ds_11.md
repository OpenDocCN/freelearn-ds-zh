# 第十一章：构建数据科学应用

数据科学应用正在引起广泛关注，主要因为它们在利用数据并提取可消费的结果方面的巨大潜力。已经有几个成功的数据产品对我们的日常生活产生了深远影响。无处不在的推荐系统、电子邮件垃圾邮件过滤器、定向广告和新闻内容已经成为生活的一部分。音乐和电影也已成为数据产品，从 iTunes 和 Netflix 等平台流媒体播放。企业，尤其是在零售等领域，正在积极寻求通过数据驱动的方法研究市场和客户行为，从而获得竞争优势。

在前面的章节中，我们已经讨论了数据分析工作流的模型构建阶段。但模型的真正价值体现在它实际部署到生产系统中的时候。最终产品，即数据科学工作流的成果，是一个已操作化的数据产品。在本章中，我们将讨论数据分析工作流的这一关键阶段。我们不会涉及具体的代码片段，而是退一步，全面了解整个过程，包括非技术性方面。

完整的图景不仅限于开发过程。它还包括用户应用程序、Spark 本身的开发，以及大数据领域中快速变化的情况。我们将首先从用户应用程序的开发过程开始，并讨论每个阶段的各种选项。接着，我们将深入了解最新 Spark 2.0 版本中的特性和改进以及未来计划。最后，我们将尝试全面概述大数据趋势，特别是 Hadoop 生态系统。此外，每个部分的末尾会提供相关参考资料和有用链接，供读者进一步了解特定的背景信息。

# 开发范围

数据分析工作流大致可分为两个阶段：构建阶段和操作化阶段。第一阶段通常是一次性的工作，且需要大量人工干预。一旦我们获得了合理的最终结果，就可以准备将产品操作化。第二阶段从第一阶段生成的模型开始，并将其作为生产工作流的一部分进行部署。在本节中，我们将讨论以下内容：

+   期望

+   演示选项

+   开发与测试

+   数据质量管理

## 期望

数据科学应用的主要目标是构建“可操作”的洞察，"可操作"是关键字。许多使用案例，如欺诈检测，要求洞察必须生成并以可消费的方式接近实时地提供，才能期待有任何行动的可能。数据产品的最终用户根据使用案例而不同。它们可能是电子商务网站的客户，或者是某大型企业的决策者。最终用户不一定总是人类，可能是金融机构中的风险评估软件工具。单一的通用方法并不适用于许多软件产品，数据产品也不例外。然而，数据产品有一些共同的期望，如下所列：

+   首要的期望是，基于真实世界数据的洞察生成时间框架应处于“可操作”时间范围内。实际的时间框架会根据使用案例而有所不同。

+   数据产品应能够融入某些（通常是已经存在的）生产工作流程中。

+   洞察结果应被转化为人们可以使用的东西，而不是晦涩难懂的数字或难以解释的图表。展示方式应该是简洁的。

+   数据产品应该具备根据输入的数据自我调整（自适应）的能力。

+   理想情况下，必须有某种方式接收人工反馈，并将其用作自我调节的来源。

+   应该有一个机制，定期且自动地定量评估其有效性。

## 演示选项

数据产品的多样性要求不同的展示方式。有时候，数据分析的最终结果是发布研究论文。有时候，它可能是仪表板的一部分，成为多个来源在同一网页上发布结果的其中之一。它们可能是显式的，目标是供人类使用，或者是隐式的，供其他软件应用使用。你可能会使用像 Spark 这样的通用引擎来构建你的解决方案，但展示方式必须高度对准目标用户群体。

有时候，你所需要做的只是写一封电子邮件，分享你的发现，或者仅仅导出一个 CSV 文件的洞察结果。或者，你可能需要围绕数据产品开发一个专门的 Web 应用程序。这里讨论了一些常见的选项，你必须选择适合当前问题的那一个。

### 互动笔记本

互动笔记本是网络应用程序，允许你创建和分享包含代码块、结果、方程式、图像、视频和解释文本的文档。它们可以作为可执行文档或具有可视化和方程式支持的 REPL Shell 进行查看。这些文档可以导出为 PDF、Markdown 或 HTML 格式。笔记本包含多个“内核”或“计算引擎”，用于执行代码块。

互动式笔记本是如果你的数据分析工作流的最终目标是生成书面报告时最合适的选择。市面上有几种笔记本，并且其中很多都支持 Spark。这些笔记本在探索阶段也非常有用。我们在前几章已经介绍过 IPython 和 Zeppelin 笔记本。

#### 参考文献

+   IPython Notebook：数据科学的综合工具：[`conferences.oreilly.com/strata/strata2013/public/schedule/detail/27233`](http://conferences.oreilly.com/strata/strata2013/public/schedule/detail/27233)

+   Sparkly Notebook：与 Spark 进行交互式分析与可视化：[`www.slideshare.net/felixcss/sparkly-notebook-interactive-analysis-and-visualization-with-spark`](http://www.slideshare.net/felixcss/sparkly-notebook-interactive-analysis-and-visualization-with-spark)

### Web API

**应用程序编程接口**（**API**）是软件与软件之间的接口；它是一个描述可用功能、如何使用这些功能以及输入输出是什么的规范。软件（服务）提供方将其某些功能暴露为 API。开发者可以开发一个软件组件来消费这个 API。例如，Twitter 提供 API 来获取或发布数据到 Twitter，或者通过编程方式查询数据。一位 Spark 爱好者可以编写一个软件组件，自动收集所有关于 #Spark 的推文，按需求进行分类，并将这些数据发布到他们的个人网站。Web API 是一种接口，其中接口被定义为一组**超文本传输协议**（**HTTP**）请求消息，并附带响应消息结构的定义。如今，RESTful（表现层状态转移）已成为事实上的标准。

你可以将你的数据产品实现为一个 API，也许这是最强大的选择。它可以插入到一个或多个应用中，比如管理仪表板以及市场营销分析工作流。你可能会开发一个特定领域的“洞察即服务”，作为一个带订阅模式的公共 Web API。Web API 的简洁性和普及性使其成为构建数据产品时最具吸引力的选择。

#### 参考文献

+   应用程序编程接口：[`en.wikipedia.org/wiki/Application_programming_interface`](https://en.wikipedia.org/wiki/Application_programming_interface)

+   准备好使用 API 了吗？三步解锁数据经济最有前景的渠道：[`www.forbes.com/sites/mckinsey/2014/01/07/ready-for-apis-three-steps-to-unlock-the-data-economys-most-promising-channel/#61e7103b89e5`](http://www.forbes.com/sites/mckinsey/2014/01/07/ready-for-apis-three-steps-to-unlock-the-data-economys-most-promising-channel/#61e7103b89e5)

+   如何基于大数据发展洞察即服务：[`www.kdnuggets.com/2015/12/insights-as-a-service-big-data.html`](http://www.kdnuggets.com/2015/12/insights-as-a-service-big-data.html)

### PMML 和 PFA

有时你可能需要以其他数据挖掘工具能理解的方式暴露你的模型。模型以及所有的预处理和后处理步骤应该转换为标准格式。PMML 和 PFA 就是数据挖掘领域的两种标准格式。

**预测模型标记语言**（**PMML**）是一种基于 XML 的预测模型交换格式，Apache Spark API 可以直接将模型转换为 PMML。一个 PMML 消息可以包含大量的数据转换，以及一个或多个预测模型。不同的数据挖掘工具可以在无需定制代码的情况下导入或导出 PMML 消息。

**分析的可移植格式**（**PFA**）是下一代预测模型交换格式。它交换 JSON 文档，并直接继承了 JSON 文档相比 XML 文档的所有优点。此外，PFA 比 PMML 更具灵活性。

#### 参考资料

+   PMML 常见问题解答：预测模型标记语言：[`www.kdnuggets.com/2013/01/pmml-faq-predictive-model-markup-language.html`](http://www.kdnuggets.com/2013/01/pmml-faq-predictive-model-markup-language.html)

+   分析的可移植格式：将模型移至生产环境：[`www.kdnuggets.com/2016/01/portable-format-analytics-models-production.html`](http://www.kdnuggets.com/2016/01/portable-format-analytics-models-production.html)

+   PFA 是做什么的？：[`dmg.org/pfa/docs/motivation/`](http://dmg.org/pfa/docs/motivation/)

## 开发与测试

Apache Spark 是一个通用的集群计算系统，可以独立运行，也可以在多个现有集群管理器上运行，如 Apache Mesos、Hadoop、Yarn 和 Amazon EC2。此外，许多大数据和企业软件公司已经将 Spark 集成到他们的产品中：Microsoft Azure HDInsight、Cloudera、IBM Analytics for Apache Spark、SAP HANA，等等。Databricks 是由 Apache Spark 创始人创办的公司，提供自己的数据科学工作流产品，涵盖从数据获取到生产的全过程。你的责任是了解组织的需求和现有的人才储备，并决定哪个选项最适合你。

无论选择哪种选项，都应遵循软件开发生命周期中的常规最佳实践，如版本控制和同行评审。在适用的情况下尽量使用高级 API。生产环境中使用的数据转换管道应该与构建模型时使用的相同。记录在数据分析工作流中出现的任何问题，这些问题往往可以促使业务流程的改进。

一如既往，测试对产品的成功至关重要。你必须维护一套自动化脚本，提供易于理解的测试结果。最少的测试用例应该覆盖以下内容：

+   遵守时间框架和资源消耗要求

+   对不良数据（例如数据类型违规）的弹性

+   New value in a categorical feature that was not encountered during the model building phase

+   Very little data or too heavy data that is expected in the target production system

Monitor logs, resource utilization, and so on to uncover any performance bottlenecks. The Spark UI provides a wealth of information to monitor Spark applications. The following are some common tips that will help you improve performance:

+   Cache any input or intermediate data that might be used multiple times.

+   Look at the Spark UI and identify jobs that are causing a lot of shuffle. Check the code and see whether you can reduce the shuffles.

+   Actions may transfer the data from workers to the driver. See that you are not transferring any data that is not absolutely necessary.

+   Stragglers; that run slower than others; may increase the overall job completion time. There may be several reasons for a straggler. If a job is running slow due to a slow node, you may set `spark.speculation` to `true`. Then Spark automatically relaunches such a task on a different node. Otherwise, you may have to revisit the logic and see whether it can be improved.

### References

+   Investigating Spark's performance: [`radar.oreilly.com/2015/04/investigating-sparks-performance.html`](http://radar.oreilly.com/2015/04/investigating-sparks-performance.html)

+   Tuning and Debugging in Apache Spark by Patrick Wendell: [`sparkhub.databricks.com/video/tuning-and-debugging-apache-spark/`](https://sparkhub.databricks.com/video/tuning-and-debugging-apache-spark/)

+   How to tune your Apache Spark jobs: [`blog.cloudera.com/blog/2015/03/how-to-tune-your-apache-spark-jobs-part-1/`](http://blog.cloudera.com/blog/2015/03/how-to-tune-your-apache-spark-jobs-part-1/) 和 part 2

## Data quality management

At the outset, let's not forget that we are trying to build fault-tolerant software data products from unreliable, often unstructured, and uncontrolled data sources. So data quality management gains even more importance in a data science workflow. Sometimes the data may solely come from controlled data sources, such as automated internal process workflows in an organization. But in all other cases, you need to carefully craft your data cleansing processes to protect the subsequent processing.

Metadata consists of the structure and meaning of data, and obviously the most critical repository to work with. It is the information about the structure of individual data sources and what each component in that structure means. You may not always be able to write some script and extract this data. A single data source may contain data with different structures or an individual component (column) may mean different things during different times. A label such as owner or high may mean different things in different data sources. Collecting and understanding all such nuances and documenting is a tedious, iterative task. Standardization of metadata is a prerequisite to data transformation development.

Some broad guidelines that are applicable to most use cases are listed here:

+   所有数据源必须进行版本控制并加上时间戳

+   数据质量管理过程通常需要最高层次的主管部门参与

+   屏蔽或匿名化敏感数据

+   一个常常被忽视的重要步骤是保持可追溯性；即每个数据元素（比如一行）与其原始来源之间的链接

# Scala 的优势

Apache Spark 允许你用 Python、R、Java 或 Scala 编写应用程序。随着这种灵活性的出现，你也需要承担选择适合自己需求的编程语言的责任。不过，无论你通常选择哪种语言，你可能都希望考虑在 Spark 驱动的应用程序中使用 Scala。在本节中，我们将解释为什么这么做。

让我们稍微跑题，首先高层次地了解一下命令式和函数式编程范式。像 C、Python 和 Java 这样的语言属于命令式编程范式。在命令式编程范式中，程序是一系列指令，并且它有一个程序状态。程序状态通常表现为在任何给定时刻变量及其值的集合。赋值和重新赋值是比较常见的。变量值在执行过程中会随着一个或多个函数的执行而变化。函数中的变量值修改不仅限于局部变量。全局变量和公共类变量就是此类变量的例子。

相比之下，用函数式编程语言如 Erlang 编写的程序可以看作是无状态的表达式求值器。数据是不可变的。如果函数以相同的输入参数被调用，那么它应该产生相同的结果（即参照透明性）。这是由于没有受到全局变量等变量上下文的干扰。这意味着函数评估的顺序不重要。函数可以作为参数传递给其他函数。递归调用取代了循环。无状态性使得并行编程变得更加容易，因为它消除了锁和潜在死锁的需求。当执行顺序不重要时，协调变得更为简化。这些因素使得函数式编程范式与并行编程非常契合。

纯函数式编程语言难以使用，因为大多数程序都需要状态的改变。包括老牌 Lisp 在内的大多数函数式编程语言都允许将数据存储在变量中（副作用）。一些语言，比如 Scala，融合了多种编程范式。

回到 Scala，它是一种基于 JVM 的静态类型多范式编程语言。其内建的类型推断机制允许程序员省略一些冗余的类型信息。这使得 Scala 在保持良好编译时检查和快速运行时的同时，具备了动态语言的灵活性。Scala 是面向对象的语言，意味着每个值都是一个对象，包括数值。函数是第一类对象，可以作为任何数据类型使用，并且可以作为参数传递给其他函数。由于 Scala 运行在 JVM 上，它与 Java 及其工具有良好的互操作性，Java 和 Scala 类可以自由混合使用。这意味着 Scala 可以轻松地与 Hadoop 生态系统进行交互。

在选择适合您应用的编程语言时，应该考虑所有这些因素。

# Spark 开发状态

到 2015 年底，Apache Spark 已成为 Hadoop 生态系统中最活跃的项目之一，按贡献者数量来看。Spark 最初是 2009 年在 UC Berkeley AMPLAB 作为研究项目启动的，与 Apache Hadoop 等项目相比仍然相对年轻，且仍在积极开发中。2015 年有三次发布，从 1.3 到 1.5，包含了如 DataFrames API、SparkR 和 Project Tungsten 等特性。1.6 版本于 2016 年初发布，包含了新的数据集 API 和数据科学功能的扩展。Spark 2.0 于 2016 年 7 月发布，作为一个重要版本，包含了许多新特性和增强功能，值得单独拿出一节来介绍。

## Spark 2.0 的特性和增强功能

Apache Spark 2.0 包含了三个主要的新特性以及其他一些性能改进和内部更改。本节尝试提供一个高层次的概述，并在需要时深入细节，帮助理解其概念。

### 统一数据集和数据框架

数据框架（DataFrames）是支持数据抽象的高级 API，其概念上等同于关系型数据库中的表格或 R 和 Python 中的 DataFrame（如 pandas 库）。数据集（Datasets）是数据框架 API 的扩展，提供类型安全的面向对象编程接口。数据集为数据框架增加了静态类型。在数据框架上定义结构为核心提供了优化信息，也有助于在分布式作业开始之前就能提前发现分析错误。

RDD、数据集（Datasets）和数据框（DataFrames）是可以互换的。RDD 仍然是低级 API。数据框、数据集和 SQL 共享相同的优化和执行管道。机器学习库使用的是数据框或数据集。数据框和数据集都在 Tungsten 上运行，Tungsten 是一个旨在提升运行时性能的计划。它们利用了 Tungsten 的快速内存编码技术，负责在 JVM 对象和 Spark 内部表示之间进行转换。相同的 API 也适用于流数据，引入了连续数据框的概念。

### 结构化流式计算

结构化流式 API 是基于 Spark SQL 引擎构建的高级 API，扩展了数据框和数据集。结构化流式计算统一了流处理、交互式查询和批处理查询。在大多数使用场景中，流数据需要与批处理和交互式查询结合，形成持续的应用程序。这些 API 旨在满足这一需求。Spark 负责增量和持续地执行流数据上的查询。

结构化流式计算的首次发布将专注于 ETL 工作负载。用户将能够指定输入、查询、触发器和输出类型。输入流在逻辑上等同于一个仅追加的表。用户可以像在传统 SQL 表上那样定义查询。触发器是一个时间框架，例如一秒。提供的输出模式包括完整输出、增量输出或就地更新（例如，数据库表）。

以这个例子为例：你可以对流数据进行聚合，通过 Spark SQL JDBC 服务器提供服务，并将其传递给数据库（例如 MySQL）用于下游应用。或者，你可以运行临时 SQL 查询，操作最新的数据。你还可以构建并应用机器学习模型。

### 项目 Tungsten 第二阶段

项目 Tungsten 的核心思想是通过本地内存管理和运行时代码生成，将 Spark 的性能推向接近硬件的极限。它首次包含在 Spark 1.4 中，并在 1.5 和 1.6 中进行了增强。其重点是通过以下几种方式显著提升 Spark 应用程序的内存和 CPU 效率：

+   明确管理内存并消除 JVM 对象模型和垃圾回收的开销。例如，一个四字节的字符串在 JVM 对象模型中大约占用 48 字节。由于 Spark 不是一个通用应用程序，并且比垃圾回收器更了解内存块的生命周期，它能够比 JVM 更高效地管理内存。

+   设计适合缓存的数据结构和算法。

+   Spark 执行代码生成，将查询的部分编译为 Java 字节码。这一过程已扩展到覆盖大多数内置表达式。

Spark 2.0 推出了第二阶段，它的速度提升了一个数量级，并且包括：

+   通过消除高开销的迭代器调用和跨多个操作符的融合，实现了整体阶段的代码生成，使生成的代码看起来像手工优化的代码

+   优化的输入和输出

## 接下来有什么？

预计 Apache Spark 2.1 将具备以下特性：

+   **持续 SQL** (**CSQL**)

+   BI 应用程序集成

+   支持更多的流式数据源和汇聚点

+   包括用于结构化流式处理的额外运算符和库

+   机器学习包的增强

+   Tungsten 中的列存储内存支持

# 大数据趋势

大数据处理在过去的十年中成为 IT 行业的一个重要组成部分。Apache Hadoop 和其他类似的努力致力于构建存储和处理海量数据的基础设施。Hadoop 平台已经运行超过 10 年，被认为成熟，几乎可以与大数据处理划上等号。Apache Spark 是一个通用的计算引擎，与 Hadoop 生态系统兼容，并且在 2015 年非常成功。

构建数据科学应用程序需要了解大数据领域和可用软件产品。我们需要仔细地映射适合我们需求的正确组件。有几个功能重叠的选择，挑选合适的工具比说起来容易得多。应用程序的成功在很大程度上取决于组合适当的技术和流程。好消息是，有几个开源选项可以降低大数据分析的成本；与此同时，你还可以通过像 Databricks 这样的公司支持的企业级端到端平台。除了手头的用例外，追踪行业趋势也同样重要。

最近 NOSQL 数据存储的激增，带来了它们自己的接口，即使它们不是关系型数据存储，也可能不遵循 ACID 属性。这是一个受欢迎的趋势，因为在关系型和非关系型数据存储之间收敛到一个单一的古老接口，提高了程序员的生产力。

在过去几十年里，运营（OLTP）和分析（OLAP）系统一直被维护为独立的系统，但这正是收敛正在发生的地方之一。这种收敛将我们带到几乎实时用例，如欺诈预防。Apache Kylin 是 Hadoop 生态系统中的一个开源分布式分析引擎，提供了一个极其快速的 OLAP 引擎。

物联网的出现加速了实时和流式分析，引入了大量新的用例。云计算解放了组织的运营和 IT 管理开销，使它们可以集中精力于其核心竞争力，特别是在大数据处理方面。基于云的分析引擎，自助数据准备工具，自助 BI，及时数据仓库，高级分析，丰富媒体分析和敏捷分析是一些常用的流行词。大数据这个术语本身正在慢慢消失或变得隐含。

在大数据领域，有大量功能重叠的软件产品和库，如下图所示（http://mattturck.com/wp-content/uploads/2016/02/matt_turck_big_data_landscape_v11.png）。为你的应用选择合适的模块是一个艰巨但非常重要的任务。以下是一个简短的项目列表，帮助你入门。该列表排除了像 Cassandra 这样的流行名字，尽量包含具有互补功能的模块，并且大多数来自 Apache 软件基金会：

+   **Apache Arrow** ([`arrow.apache.org/`](https://arrow.apache.org/)) 是一个内存中的列式存储层，用于加速分析处理和数据交换。它是一个高性能、跨系统的内存数据表示，预计能带来 100 倍的性能提升。

+   **Apache Parquet** ([`parquet.apache.org/`](https://parquet.apache.org/)) 是一种列式存储格式。Spark SQL 提供对读取和写入 parquet 文件的支持，同时自动捕获数据的结构。

+   **Apache Kafka** ([`kafka.apache.org/`](http://kafka.apache.org/)) 是一个流行的高吞吐量分布式消息系统。Spark Streaming 提供直接的 API 来支持从 Kafka 进行流数据摄取。

+   **Alluxio** ([`alluxio.org/`](http://alluxio.org/))，前身为 Tachyon，是一个以内存为中心的虚拟分布式存储系统，能够在集群之间以内存速度共享数据。它旨在成为大数据的事实上的存储统一层。Alluxio 位于计算框架（如 Spark）和存储系统（如 Amazon S3、HDFS 等）之间。

+   **GraphFrames** ([`databricks.com/blog/2016/03/03/introducing-graphframes.html`](https://databricks.com/blog/2016/03/03/introducing-graphframes.html)) 是一个基于 Apache Spark 的图处理库，建立在 DataFrames API 之上。

+   **Apache Kylin** ([`kylin.apache.org/`](http://kylin.apache.org/)) 是一个分布式分析引擎，旨在提供 SQL 接口和多维分析（OLAP），支持 Hadoop 上的超大规模数据集。

+   **Apache Sentry** ([`sentry.apache.org/`](http://sentry.apache.org/)) 是一个系统，用于对存储在 Hadoop 集群中的数据和元数据执行细粒度的基于角色的授权。它在撰写本书时处于孵化阶段。

+   **Apache Solr** ([`lucene.apache.org/solr/`](http://lucene.apache.org/solr/)) 是一个非常快速的搜索平台。查看这个 [演示](https://spark-summit.org/2015/events/integrating-spark-and-solr/) 了解如何将 Solr 与 Spark 集成。

+   **TensorFlow** ([`www.tensorflow.org/`](https://www.tensorflow.org/)) 是一个机器学习库，广泛支持深度学习。查看这个 [博客](https://databricks.com/blog/2016/01/25/deep-learning-with-spark-and-tensorflow.html)，了解如何与 Spark 一起使用。

+   **Zeppelin** ([`zeppelin.incubator.apache.org/`](http://zeppelin.incubator.apache.org/)) 是一个基于 Web 的笔记本，支持交互式数据分析。它在数据可视化章节中有介绍。

# 总结

在本章的最后，我们讨论了如何使用 Spark 构建现实世界的应用程序。我们讨论了包含技术性和非技术性方面的数据分析工作流的宏观视角。

# 参考文献

+   Spark Summit 网站包含了关于 Apache Spark 和相关项目的大量信息，来自已完成的活动。

+   与 *Matei Zaharia* 的访谈，由 KDnuggets 撰写。

+   来自 KDnuggets 的 *为什么 Spark 在 2015 年达到了临界点*，作者是 **Matthew Mayo**。

+   上线：准备你的第一个 Spark 生产部署是一个非常好的起点。

+   *什么是 Scala？* 来自 Scala 官网。

+   **马丁·奥德斯基**（*Martin Odersky*），Scala 的创始人，解释了为什么 Scala 将命令式编程和函数式编程融合在一起。
