# 第 11 章. 打包 Spark 应用程序

到目前为止，我们一直在使用一种非常方便的方式来在 Spark 中开发代码 - Jupyter 笔记本。当您想开发一个概念验证并记录您所做的工作时，这种方法非常出色。

然而，如果您需要安排作业，Jupyter 笔记本将无法工作，因此它每小时运行一次。此外，打包您的应用程序相当困难，因为很难将脚本分割成具有良好定义的 API 的逻辑块 - 所有的内容都位于单个笔记本中。

在本章中，我们将学习如何以模块化的形式编写您的脚本，并编程提交作业到 Spark。

在开始之前，您可能想查看 *Bonus Chapter 2, Free Spark Cloud Offering*，其中我们提供了如何订阅和使用 Databricks 的社区版或 Microsoft 的 HDInsight Spark 提供的说明；如何做到这一点的说明可以在此处找到：[https://www.packtpub.com/sites/default/files/downloads/FreeSparkCloudOffering.pdf](https://www.packtpub.com/sites/default/files/downloads/FreeSparkCloudOffering.pdf)。

在本章中，您将学习：

+   `spark-submit` 命令是什么

+   如何以编程方式打包和部署您的应用程序

+   如何模块化您的 Python 代码并将其与 PySpark 脚本一起提交

# `spark-submit` 命令

提交作业到 Spark（无论是本地还是集群）的入口点是 `spark-submit` 脚本。然而，该脚本不仅允许您提交作业（尽管这是其主要目的），还可以终止作业或检查其状态。

### 注意

在内部，`spark-submit` 命令将调用 `spark-class` 脚本，该脚本反过来启动一个启动器 Java 应用程序。对于感兴趣的人来说，可以查看 Spark 的 GitHub 仓库：[https://github.com/apache/spark/blob/master/bin/sparksubmit](https://github.com/apache/spark/blob/master/bin/spark-submit)。

`spark-submit` 命令为在多种 Spark 支持的集群管理器（如 Mesos 或 Yarn）上部署应用程序提供了一个统一的 API，从而让您无需分别对每个应用程序进行配置。

在一般层面上，语法如下所示：

[PRE0]

我们很快就会查看所有选项的列表。`app arguments` 是您想要传递给应用程序的参数。

### 注意

您可以使用 `sys.argv`（在 `import sys` 之后）自行解析命令行参数，或者可以使用 Python 的 `argparse` 模块。

## 命令行参数

当使用 `spark-submit` 时，您可以传递大量针对 Spark 引擎的参数。

### 注意

在以下内容中，我们将仅介绍针对 Python 的特定参数（因为 `spark-submit` 也可以用于提交用 Scala 或 Java 编写的应用程序，并打包为 `.jar` 文件）。

我们将逐一介绍参数，以便您对从命令行可以执行的操作有一个良好的概述：

+   `--master`：用于设置主（头）节点 URL 的参数。允许的语法是：

    +   `local`: 用于在你的本地机器上执行你的代码。如果你传递`local`，Spark将随后在单个线程中运行（不利用任何并行性）。在多核机器上，你可以指定Spark要使用的确切核心数，通过指定`local[n]`，其中`n`是要使用的核心数，或者使用`local[*]`运行Spark，使其以机器上的核心数创建尽可能多的线程。

    +   `spark://host:port`: 这是一个Spark独立集群的URL和端口号（不运行任何作业调度器，如Mesos或Yarn）。

    +   `mesos://host:port`: 这是一个部署在Mesos上的Spark集群的URL和端口号。

    +   `yarn`: 用于从运行Yarn作为工作负载均衡器的头节点提交作业。

+   `--deploy-mode`: 参数允许你决定是否在本地（使用`client`）或集群中的某个工作机器上（使用`cluster`选项）启动Spark驱动程序进程。此参数的默认值为`client`。以下是Spark文档的摘录，它更具体地解释了差异（来源：[http://bit.ly/2hTtDVE](http://bit.ly/2hTtDVE)）：

    > 一种常见的部署策略是从与你的工作机器物理上位于同一位置的门控机器上的[屏幕会话](https://example.org)提交你的应用程序（例如，独立EC2集群中的主节点）。在这种配置中，客户端模式是合适的。在客户端模式下，驱动程序直接在spark-submit过程中启动，该过程作为集群的客户端。应用程序的输入和输出连接到控制台。因此，这种模式特别适合涉及REPL（例如Spark shell）的应用程序。
    > 
    > 或者，如果你的应用程序是从远离工作机器的机器（例如，在你的笔记本电脑上本地）提交的，那么通常使用集群模式以最小化驱动程序和执行器之间的网络延迟。目前，独立模式不支持Python应用程序的集群模式。

+   `--name`: 你的应用程序的名称。请注意，如果你在创建`SparkSession`时以编程方式指定了应用程序的名称（我们将在下一节中介绍），则命令行参数将覆盖该参数。我们将在讨论`--conf`参数时简要解释参数的优先级。

+   `--py-files`: 要包含的`.py`、`.egg`或`.zip`文件的逗号分隔列表，用于Python应用程序。这些文件将被发送到每个执行器以供使用。在本章的后面部分，我们将向你展示如何将你的代码打包成模块。

+   `--files`: 该命令给出一个以逗号分隔的文件列表，这些文件也将被发送到每个执行器以供使用。

+   `--conf`: 参数允许你从命令行动态更改应用程序的配置。语法是`<Spark属性>=<属性值>`。例如，你可以传递`--conf spark.local.dir=/home/SparkTemp/`或`--conf spark.app.name=learningPySpark`；后者相当于之前解释的提交`--name`属性。

    ### 注意

    Spark从三个地方使用配置参数：在创建`SparkContext`时，你在应用程序中指定的`SparkConf`参数具有最高优先级，然后是任何从命令行传递给`spark-submit`脚本的参数，最后是`conf/spark-defaults.conf`文件中指定的任何参数。

+   `--properties-file`：包含配置的文件。它应该具有与`conf/spark-defaults.conf`文件相同的属性集，因为它将被读取而不是它。

+   `--driver-memory`：指定为驱动程序分配多少内存的应用程序参数。允许的值具有类似于1,000M、2G的语法。默认值为1,024M。

+   `--executor-memory`：指定为每个执行器分配多少内存的应用程序参数。默认值为1G。

+   `--help`：显示帮助信息并退出。

+   `--verbose`：在运行应用程序时打印额外的调试信息。

+   `--version`：打印Spark的版本。

仅在Spark独立和`cluster`部署模式中，或在Yarn上部署的集群中，你可以使用`--driver-cores`来指定驱动程序的核心数（默认为1）。在Spark独立或Mesos的`cluster`部署模式中，你还有机会使用以下任何一个：

+   `--supervise`：如果指定，当驱动程序丢失或失败时，将重新启动驱动程序。这也可以通过将`--deploy-mode`设置为`cluster`在Yarn中设置。

+   `--kill`：将根据其`submission_id`结束进程

+   `--status`：如果指定此命令，它将请求指定应用程序的状态。

在Spark独立和Mesos（仅使用`client`部署模式）中，你也可以指定`--total-executor-cores`，这是一个将请求所有执行器（而不是每个执行器）指定的核心数的参数。另一方面，在Spark独立和YARN中，只有`--executor-cores`参数指定了每个执行器的核心数（在YARN模式下默认为1，或在独立模式下为工作节点上的所有可用核心）。

此外，当提交到YARN集群时，你可以指定：

+   `--queue`：此参数指定一个队列，将作业提交到YARN（默认为`default`）。

+   `--num-executors`：指定为作业请求多少个执行器机器的参数。如果启用了动态分配，初始执行器数量至少为指定的数量。

现在我们已经讨论了所有参数，是时候将其付诸实践了。

# 以编程方式部署应用程序

与Jupyter笔记本不同，当你使用`spark-submit`命令时，你需要自己准备`SparkSession`并配置它，以确保应用程序正常运行。

在本节中，我们将学习如何创建和配置`SparkSession`，以及如何使用Spark外部模块。

### 注意

如果你还没有在Databricks或Microsoft（或任何Spark的提供者）上创建你的免费账户，不要担心——我们仍然会使用你的本地机器，因为这更容易让我们开始。然而，如果你决定将你的应用程序迁移到云端，实际上只需要在提交作业时更改`--master`参数。

## 配置你的SparkSession

使用Jupyter和通过编程方式提交作业之间的主要区别在于，你必须创建你的Spark上下文（如果你计划使用HiveQL，还包括Hive），而当你使用Jupyter运行Spark时，上下文会自动为你启动。

在本节中，我们将开发一个简单的应用程序，该应用程序将使用Uber的公共数据，这些数据是在2016年6月的纽约地区完成的行程；我们从[https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-06.csv](https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-06.csv)（注意，它是一个几乎3GB的文件）下载的数据集。原始数据集包含1100万次行程，但为了我们的示例，我们只检索了330万次，并且只选择了所有可用列的一个子集。

### 注意

转换后的数据集可以从[http://www.tomdrabas.com/data/LearningPySpark/uber_data_nyc_2016-06_3m_partitioned.csv.zip](http://www.tomdrabas.com/data/LearningPySpark/uber_data_nyc_2016-06_3m_partitioned.csv.zip)下载。下载文件并将其解压到GitHub的`Chapter13`文件夹中。文件可能看起来很奇怪，因为它实际上是一个包含四个文件的目录，当Spark读取时，将形成一个数据集。

那么，让我们开始吧！

## 创建SparkSession

与之前的版本相比，Spark 2.0在创建`SparkContext`方面变得稍微简单一些。实际上，Spark目前使用`SparkSession`来暴露高级功能，而不是显式创建`SparkContext`。以下是这样做的方法：

[PRE1]

前面的代码就是你需要的一切！

### 小贴士

如果你仍然想使用RDD API，你仍然可以。然而，你不再需要创建一个`SparkContext`，因为`SparkSession`在底层会自动启动一个。为了获取访问权限，你可以简单地调用（借鉴前面的示例）：`sc = spark.SparkContext`。

在这个示例中，我们首先创建`SparkSession`对象并调用其`.builder`内部类。`.appName(...)`方法允许我们给我们的应用程序一个名字，而`.getOrCreate()`方法要么创建一个，要么检索一个已经创建的`SparkSession`。给应用程序一个有意义的名字是一个好习惯，因为它有助于（1）在集群上找到你的应用程序，并且（2）减少每个人的困惑。

### 注意

在底层，Spark会话创建一个`SparkContext`对象。当你对`SparkSession`调用`.stop()`时，它实际上会终止内部的`SparkContext`。

## 代码模块化

以这种方式构建您的代码以便以后可以重用始终是一件好事。Spark 也可以这样做 - 您可以将方法模块化，然后在以后某个时间点重用它们。这也有助于提高代码的可读性和可维护性。

在这个例子中，我们将构建一个模块，它将对我们的数据集进行一些计算：它将计算从接货点到卸货点的“直线距离”（以英里为单位）（使用 Haversine 公式），并将计算出的距离从英里转换为公里。

### 注意

关于 Haversine 公式的更多信息可以在这里找到：[http://www.movable-type.co.uk/scripts/latlong.html](http://www.movable-type.co.uk/scripts/latlong.html)。

因此，首先，我们将构建一个模块。

### 模块结构

我们将我们额外方法的代码放在了 `additionalCode` 文件夹中。

### 提示

如果您还没有这样做，请查看此书的 GitHub 仓库 [https://github.com/drabastomek/learningPySpark/tree/master/Chapter11](https://github.com/drabastomek/learningPySpark/tree/master/Chapter11)。

文件夹的树状结构如下：

![模块结构](img/B05793_11_01.jpg)

如您所见，它具有某种正常 Python 包的结构：在最上面我们有 `setup.py` 文件，这样我们就可以打包我们的模块，然后内部包含我们的代码。

在我们的情况下，`setup.py` 文件如下所示：

[PRE2]

我们在这里不会深入探讨结构（它本身相当直观）：您可以在以下链接中了解更多关于如何为其他项目定义 `setup.py` 文件的信息 [https://pythonhosted.org/an_example_pypi_project/setuptools.html](https://pythonhosted.org/an_example_pypi_project/setuptools.html)。

工具文件夹中的 `__init__.py` 文件包含以下代码：

[PRE3]

它有效地暴露了 `geoCalc.py` 和 `converters`（稍后将有更多介绍）。

### 计算两点之间的距离

我们提到的第一个方法使用 Haversine 公式来计算地图上任意两点之间的直接距离（笛卡尔坐标）。执行此操作的代码位于模块的 `geoCalc.py` 文件中。

`calculateDistance(...)` 是 `geoCalc` 类的一个静态方法。它接受两个地理点，这些点以元组或包含两个元素（按顺序为纬度和经度）的列表的形式表示，并使用 Haversine 公式来计算距离。计算距离所需的地球半径以英里表示，因此计算出的距离也将以英里为单位。

### 转换距离单位

我们构建了工具包，使其更加通用。作为包的一部分，我们公开了用于在各个测量单位之间进行转换的方法。

### 注意

目前我们只限制距离，但功能可以进一步扩展到其他领域，如面积、体积或温度。

为了便于使用，任何实现为`converter`的类都应该公开相同的接口。这就是为什么建议这样的类从我们的`BaseConverter`类派生（参见`base.py`）：

[PRE4]

这是一个纯抽象类，不能被实例化：它的唯一目的是强制派生类实现`convert(...)`方法。有关实现细节，请参阅`distance.py`文件。对于熟悉Python的人来说，代码应该是自解释的，所以我们不会一步一步地解释它。

### 构建一个蛋

现在我们已经将所有代码放在一起，我们可以打包它。PySpark的文档指出，你可以使用`--py-files`开关将`.py`文件传递给`spark-submit`脚本，并用逗号分隔。然而，将我们的模块打包成`.zip`或`.egg`会更方便。这时`setup.py`文件就派上用场了——你只需要在`additionalCode`文件夹中调用它：

[PRE5]

如果一切顺利，你应该看到三个额外的文件夹：`PySparkUtilities.egg-info`、`build`和`dist`——我们感兴趣的是位于`dist`文件夹中的文件：`PySparkUtilities-0.1.dev0-py3.5.egg`。

### 提示

在运行前面的命令后，你可能发现你的`.egg`文件名略有不同，因为你可能有不同的Python版本。你仍然可以在Spark作业中使用它，但你需要调整`spark-submit`命令以反映你的`.egg`文件名。

### Spark中的用户定义函数

在PySpark中对`DataFrame`进行操作时，你有两种选择：使用内置函数来处理数据（大多数情况下这足以实现所需的功能，并且推荐这样做，因为代码性能更好）或创建自己的用户定义函数。

要定义一个用户定义函数（UDF），你必须将Python函数包装在`.udf(...)`方法中，并定义其返回值类型。这就是我们在脚本中这样做的方式（检查`calculatingGeoDistance.py`文件）：

[PRE6]

我们可以使用这样的函数来计算距离并将其转换为英里：

[PRE7]

使用`.withColumn(...)`方法，我们创建额外的列，包含我们感兴趣的价值。

### 注意

这里需要提醒一点。如果你使用PySpark内置函数，即使你调用它们为Python对象，底层调用会被转换并执行为Scala代码。然而，如果你在Python中编写自己的方法，它不会被转换为Scala，因此必须在驱动程序上执行。这会导致性能显著下降。查看Stack Overflow上的这个答案以获取更多详细信息：[http://stackoverflow.com/questions/32464122/spark-performance-for-scala-vs-python](http://stackoverflow.com/questions/32464122/spark-performance-for-scala-vs-python)。

现在我们把所有的拼图放在一起，最终提交我们的工作。

## 提交一个工作

在你的CLI中输入以下内容（我们假设你保持文件夹结构与GitHub上的结构不变）：

[PRE8]

我们需要对`launch_spark_submit.sh`shell脚本进行一些解释。在Bonus [第1章](ch01.html "第1章。理解Spark")，*安装Spark*中，我们配置了Spark实例以运行Jupyter（通过设置`PYSPARK_DRIVER_PYTHON`系统变量为`jupyter`）。如果你在这样配置的机器上简单地使用`spark-submit`，你很可能会遇到以下错误的一些变体：

[PRE9]

因此，在运行`spark-submit`命令之前，我们首先必须取消设置该变量，然后运行代码。这会迅速变得极其繁琐，所以我们通过`launch_spark_submit.sh`脚本自动化了它：

[PRE10]

如你所见，这不过是`spark-submit`命令的一个包装器。

如果一切顺利，你将在CLI中看到以下*意识流*：

![提交作业](img/B05793_11_02.jpg)

从输出中你可以获得许多有用的信息：

+   当前Spark版本：2.1.0

+   Spark UI（用于跟踪作业进度的工具）已成功在`http://localhost:4040`启动

+   我们的成功添加了`.egg`文件到执行

+   `uber_data_nyc_2016-06_3m_partitioned.csv`已成功读取

+   每个作业和任务的启动和停止都被列出

作业完成后，你将看到以下类似的内容：

![提交作业](img/B05793_11_03.jpg)

从前面的截图，我们可以看到距离被正确报告。你还可以看到Spark UI进程现在已经停止，并且所有清理工作都已执行。

## 监控执行

当你使用`spark-submit`命令时，Spark会启动一个本地服务器，允许你跟踪作业的执行情况。以下是窗口的外观：

![监控执行](img/B05793_11_04.jpg)

在顶部，你可以切换到**作业**或**阶段**视图；**作业**视图允许你跟踪执行整个脚本的独立作业，而**阶段**视图允许你跟踪所有执行的阶段。

你还可以通过点击阶段的链接来查看每个阶段的执行配置文件，并跟踪每个任务的执行。在以下截图中，你可以看到Stage 3的执行配置文件，其中运行了四个任务：

![监控执行](img/B05793_11_05.jpg)

### 小贴士

在集群设置中，你将看到**driver/localhost**而不是**驱动器/本地主机**，而是驱动器编号和主机的IP地址。

在一个作业或阶段内部，你可以点击DAG可视化来查看你的作业或阶段是如何执行的（左边的以下图表显示了**作业**视图，而右边的显示了**阶段**视图）：

![监控执行](img/B05793_11_06.jpg)

# Databricks作业

如果你使用Databricks产品，从Databricks笔记本的开发到生产的一个简单方法就是使用Databricks作业功能。它将允许你：

+   安排Databricks笔记本在现有或新集群上运行

+   按您希望的频率（从分钟到月份）安排

+   为您的作业安排超时和重试

+   当作业开始、完成或出错时收到警报

+   查看历史作业运行以及审查单个笔记本作业运行的记录

这种功能极大地简化了您作业提交的调度和生产工作流程。请注意，您需要将您的 Databricks 订阅（从社区版）升级才能使用此功能。

要使用此功能，请转到 Databricks **作业**菜单并点击**创建作业**。从这里，填写作业名称，然后选择您想要转换为作业的笔记本，如图所示：

![Databricks 作业](img/B05793_11_07.jpg)

一旦您选择了笔记本，您还可以选择是否使用正在运行的现有集群，或者让作业调度器为该作业启动一个**新集群**，如图所示：

![Databricks 作业](img/B05793_11_08.jpg)

一旦您选择了笔记本和集群；您可以设置计划、警报、超时和重试。一旦您完成设置作业，它应该看起来类似于以下截图中的**人口与价格线性回归作业**：

![Databricks 作业](img/B05793_11_09.jpg)

您可以通过点击**活动运行**下方的**立即运行**链接来测试作业。

如**Meetup Streaming RSVPs 作业**中所述，您可以查看您已完成运行的记录；如图所示，对于这个笔记本，有**50**个完成的作业运行：

![Databricks 作业](img/B05793_11_10.jpg)

通过点击作业运行（在这种情况下，**运行 50**），您可以查看该作业运行的结果。您不仅可以查看开始时间、持续时间和服务状态，还可以查看该特定作业的结果：

![Databricks 作业](img/B05793_11_11.jpg)

### 备注

**REST 作业服务器**

运行作业的一种流行方式是使用 REST API。如果您使用 Databricks，您可以使用 Databricks REST API 运行作业。如果您更喜欢管理自己的作业服务器，一个流行的开源 REST 作业服务器是 `spark-jobserver` - 一个用于提交和管理 Apache Spark 作业、jar 和作业上下文的 RESTful 接口。该项目最近（在撰写本文时）进行了更新，以便它可以处理 PySpark 作业。

更多信息，请参阅 [https://github.com/spark-jobserver/spark-jobserver](https://github.com/spark-jobserver/spark-jobserver)。

# 摘要

在本章中，我们向您介绍了如何从命令行将用 Python 编写的应用程序提交到 Spark 的步骤。我们讨论了 `spark-submit` 参数的选择。我们还向您展示了如何打包您的 Python 代码，并将其与 PySpark 脚本一起提交。此外，我们还向您展示了如何跟踪作业的执行。

此外，我们还提供了一个关于如何使用Databricks Jobs功能运行Databricks笔记本的快速概述。此功能简化了从开发到生产的过渡，允许您将笔记本作为一个端到端工作流程执行。

这本书的内容到此结束。我们希望您享受了这次旅程，并且书中包含的材料能帮助您开始使用Python与Spark进行工作。祝您好运！
