# 第七章：Apache Spark GraphX

在本章中，我们希望探讨 Apache Spark GraphX 模块和图处理的一般性。因此，本章将涵盖在 GraphX 之上实现图分析工作流程的主题。*GraphX 编码*部分，用 Scala 编写，将提供一系列图编码示例。在用 Scala 编写代码以使用 Spark GraphX 模块之前，我们认为提供关于图处理中图实际是什么的概述将是有用的。以下部分使用几个简单图作为示例，提供了一个简短的介绍。

在本章中，我们将涵盖：

+   从原始数据创建图

+   计数

+   过滤

+   PageRank

+   三角形计数

+   连通组件

# 概述

图可以被视为一种数据结构，由一组顶点和连接它们的边组成。图中的顶点或节点可以是任何对象（例如人），而边则是它们之间的关系。边可以是无向的或有向的，意味着关系从一个节点操作到另一个节点。例如，节点**A**是节点**B**的父母。

在下面的图中，圆圈代表顶点或节点（**A**至**D**），而粗线代表它们之间的边或关系（**E1**至**E6**）。每个节点或边可能具有属性，这些值由相关的灰色方块表示（**P1**至**P7**）：

因此，如果一个图代表了一个物理...

# 使用 GraphX 进行图分析/处理

本节将探讨使用上一节中展示的家庭关系图数据样本，在 Scala 中进行 Apache Spark GraphX 编程。此数据将被访问为一组顶点和边。尽管此数据集较小，但通过这种方式构建的图可能非常庞大。例如，我们仅使用四个 Apache Spark 工作者就能够分析一家大型银行的 30 TB 金融交易数据。

# 原始数据

我们正在处理两个数据文件。它们包含将用于本节的顶点和边数据，这些数据构成了一个图：

```scala
graph1_edges.csvgraph1_vertex.csv
```

`顶点`文件仅包含六行，代表上一节中使用的图。每个`顶点`代表一个人，并具有顶点 ID 号、姓名和年龄值：

```scala
1,Mike,482,Sarah,453,John,254,Jim,535,Kate,226,Flo,52
```

`边`文件包含一组有向`边`值，形式为源顶点 ID、目标顶点 ID 和关系。因此，记录 1 在`Flo`和`Mike`之间形成了一个`姐妹`关系：

```scala
6,1,Sister1,2,Husband2,1,Wife5,1,Daughter5,2,Daughter3,1,Son3,2,Son4,1,Friend1,5,Father1,3,Father2,5,Mother2,3,Mother
```

让我们，检查一些...

# 创建图

本节将解释通用 Scala 代码，直到从数据创建 GraphX 图。这将节省时间，因为相同的代码在每个示例中都被重复使用。一旦解释完毕，我们将专注于每个代码示例中的实际基于图的操作。

1.  通用代码首先导入 Spark 上下文、GraphX 和 RDD 功能，以便在 Scala 代码中使用：

```scala
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
```

1.  然后定义一个应用程序，它`扩展`了`App`类。应用程序名称从`graph1`到`graph5`每个示例都会更改。运行应用程序时将使用此应用程序名称`spark-submit`：

```scala
object graph1 extends App {
```

1.  如前所述，有两个数据文件包含`顶点`和`边`信息：

```scala
val vertexFile = "graph1_vertex.csv"
val edgeFile   = "graph1_edges.csv"
```

1.  **Spark 主 URL**定义为应用程序名称，该名称将在应用程序运行时出现在 Spark 用户界面中。创建一个新的 Spark 配置对象，并将 URL 和名称分配给它：

```scala
val sparkMaster = "spark://localhost:7077"
val appName = "Graph 1"
val conf = new SparkConf()
conf.setMaster(sparkMaster)
conf.setAppName(appName)
```

1.  使用刚刚定义的配置创建一个新的 Spark 上下文：

```scala
val sparkCxt = new SparkContext(conf)
```

1.  然后，使用`sparkCxt.textFile`方法将文件中的`顶点`信息加载到称为顶点的 RDD 基础结构中。数据存储为长`VertexId`和字符串，以表示人的姓名和年龄。数据行按逗号分割，因为这是基于 CSV 的数据：

```scala
val vertices: RDD[(VertexId, (String, String))] =
     sparkCxt.textFile(vertexFile).map { line =>
       val fields = line.split(",")
       ( fields(0).toLong, ( fields(1), fields(2) ) )
}
```

1.  同样，`边`数据加载到称为边的 RDD 基础数据结构中。基于 CSV 的数据再次按逗号值分割。前两个数据值转换为长值，因为它们表示源和目标顶点 ID。最后代表边关系的值保持为`字符串`。请注意，RDD 结构边中的每个记录实际上现在是一个`Edge`记录：

```scala
val edges: RDD[Edge[String]] =
     sparkCxt.textFile(edgeFile).map { line =>
       val fields = line.split(",")
       Edge(fields(0).toLong, fields(1).toLong, fields(2))
}
```

1.  如果缺少连接或`顶点`，则定义默认值；然后从基于 RDD 的结构顶点和边以及`默认`记录构建图：

```scala
val default = ("Unknown", "Missing")
val graph = Graph(vertices, edges, default)
```

1.  这创建了一个基于 GraphX 的结构，称为`图`，现在可以用于每个示例。请记住，尽管这些数据样本可能很小，但您可以使用这种方法创建非常大的图。

这些算法中的许多都是迭代应用，例如 PageRank 和三角计数。因此，程序将生成许多迭代的 Spark 作业。

# 示例 1 – 计数

图已加载，我们知道数据文件中的数据量。但在实际图中，顶点和边的数据内容是什么？使用以下所示的顶点和边`计数`函数提取此信息非常简单：

```scala
println( "vertices : " + graph.vertices.count )println( "edges   : " + graph.edges.count )
```

运行`graph1`示例，使用先前创建的`.jar`文件和示例名称，将提供`计数`信息。主 URL 用于连接到 Spark 集群，并为执行器内存和总执行器核心提供一些默认参数：

```scala
spark-submit \--class graph1 \--master spark://localhost:7077 \--executor-memory 700M \--total-executor-cores ...
```

# 示例 2 – 过滤

如果我们需要从主图中创建一个子图，并根据人物年龄或关系进行过滤，会发生什么？第二个示例 Scala 文件`graph2`中的示例代码展示了如何实现这一点：

```scala
val c1 = graph.vertices.filter { case (id, (name, age)) => age.toLong > 40 }.count
val c2 = graph.edges.filter { case Edge(from, to, property)
   => property == "Father" | property == "Mother" }.count
println( "Vertices count : " + c1 )
println( "Edges   count : " + c2 )
```

已经从主图创建了两个示例计数：第一个仅根据年龄过滤基于人的顶点，选取那些年龄大于四十岁的人。请注意，存储为字符串的`年龄`值已转换为长整型以进行比较。

第二个示例根据`Mother`或`Father`的关系属性过滤边。创建并打印了两个计数值`c1`和`c2`，作为 Spark 运行输出，如下所示：

```scala
Vertices count : 4
Edges   count : 4
```

# 示例 3 – PageRank

PageRank 算法为图中的每个顶点提供一个排名值。它假设连接到最多边的顶点是最重要的。

搜索引擎使用 PageRank 为网页搜索期间的页面显示提供排序，如下面的代码所示：

```scala
val tolerance = 0.0001val ranking = graph.pageRank(tolerance).verticesval rankByPerson = vertices.join(ranking).map {   case (id, ( (person,age) , rank )) => (rank, id, person)}
```

示例代码创建了一个容差值，并使用它调用图的`pageRank`方法。然后，顶点被排名到一个新的值排名中。为了使排名更有意义，排名值与原始值进行了连接...

# 示例 4 – 三角形计数

三角形计数算法提供了一个基于顶点的与该顶点相关的三角形数量的计数。例如，顶点`Mike` (1) 连接到`Kate` (5)，`Kate` 连接到`Sarah` (2)，`Sarah` 连接到`Mike` (1)，从而形成一个三角形。这在需要生成无三角形的最小生成树图进行路线规划时可能很有用。

执行三角形计数并打印它的代码很简单，如下所示。对图的顶点执行`triangleCount`方法。结果保存在值`tCount`中并打印出来：

```scala
val tCount = graph.triangleCount().vertices
println( tCount.collect().mkString("\n") )
```

应用程序作业的结果显示，顶点`Flo` (4) 和`Jim` (6) 没有三角形，而`Mike` (1) 和`Sarah` (2) 如预期那样拥有最多，因为他们有最多的关系：

```scala
(4,0)
(6,0)
(2,4)
(1,4)
(3,2)
(5,2)
```

# 示例 5 – 连通组件

当从数据中创建一个大图时，它可能包含不相连的子图或彼此隔离的子图，并且可能不包含它们之间的桥接或连接边。这些算法提供了一种连接性的度量。根据你的处理需求，了解所有顶点是否连接可能很重要。

此示例的 Scala 代码调用了两个图方法，`connectedComponents`和`stronglyConnectedComponents`。`strong`方法需要一个最大迭代计数，已设置为`1000`。这些计数作用于图的顶点：

```scala
val iterations = 1000val connected = graph.connectedComponents().verticesval connectedS = graph.stronglyConnectedComponents(iterations).vertices ...
```

# 总结

本章通过示例展示了如何使用基于 Scala 的代码调用 Apache Spark 中的 GraphX 算法。使用 Scala 是因为它比 Java 需要更少的代码来开发示例，从而节省时间。请注意，GraphX 不适用于 Python 或 R。可以使用基于 Scala 的 shell，并且代码可以编译成 Spark 应用程序。

已经介绍了最常见的图算法，你现在应该知道如何使用 GraphX 解决任何图问题。特别是，既然你已经理解了 GraphX 中的图仍然由 RDD 表示和支持，那么你已经熟悉使用它们了。本章的配置和代码示例也将随书提供下载。
