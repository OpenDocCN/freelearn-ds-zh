# 第八章：Spark 图处理

图是数学概念，也是计算机科学中的数据结构。它在许多现实世界的应用场景中有着广泛的应用。图用于建模实体之间的成对关系。这里的实体称为顶点，两个顶点通过一条边相连。图由一组顶点和连接它们的边组成。

从概念上讲，这是一种看似简单的抽象，但当涉及到处理大量顶点和边时，它计算密集，消耗大量处理时间和计算资源。以下是一个具有四个顶点和三条边的图的表示：

![Spark 图处理](img/B05289_08_01_new.jpg)

图 1

本章我们将涵盖以下主题：

+   图及其用途

+   图计算库 GraphX

+   网页排名算法 PageRank

+   连通组件算法

+   图框架 GraphFrames

+   图查询

# 理解图及其用途

有许多应用程序结构可以被建模为图。在社交网络应用中，用户之间的关系可以被建模为一个图，其中用户构成图的顶点，用户之间的关系构成图的边。在多阶段作业调度应用中，各个任务构成图的顶点，任务的排序构成图的边。在道路交通建模系统中，城镇构成图的顶点，连接城镇的道路构成图的边。

给定图的边有一个非常重要的属性，即*连接的方向*。在许多应用场景中，连接的方向并不重要。城市间道路连接的情况就是这样一个例子。但如果应用场景是在城市内提供驾驶方向，那么交通路口之间的连接就有方向。任意两个交通路口之间都有道路连接，但也可能是一条单行道。因此，这都取决于交通流向的方向。如果道路允许从交通路口 J1 到 J2 的交通，但不允许从 J2 到 J1，那么驾驶方向的图将显示从 J1 到 J2 的连接，而不是从 J2 到 J1。在这种情况下，连接 J1 和 J2 的边有方向。如果 J2 和 J3 之间的道路在两个方向都开放，那么连接 J2 和 J3 的边没有方向。所有边都有方向的图称为**有向图**。

### 提示

在图形表示中，对于有向图，必须给出边的方向。如果不是有向图，则可以不带任何方向地表示边，或者向两个方向表示边，这取决于个人选择。*图 1*不是有向图，但表示时向连接的两个顶点都给出了方向。

*图 2*中，社交网络应用用例中两个用户之间的关系被表示为一个图。用户构成顶点，用户之间的关系构成边。用户 A 关注用户 B。同时，用户 A 是用户 B 的儿子。在这个图中，有两条平行边共享相同的源和目标顶点。包含平行边的图称为多图。*图 2*所示的图也是一个有向图。这是一个**有向多图**的好例子。

![理解图及其用途](img/image_08_002.jpg)

图 2

在现实世界的用例中，图的顶点和边代表了现实世界的实体。这些实体具有属性。例如，在社交网络应用的用户社交连接图中，用户构成顶点，并拥有诸如姓名、电子邮件、电话号码等属性。同样，用户之间的关系构成图的边，连接用户顶点的边可以具有如关系等属性。任何图处理应用库都应足够灵活，以便为图的顶点和边附加任何类型的属性。

# 火花图 X 库

在开源世界中，有许多用于图处理的库，如 Giraph、Pregel、GraphLab 和 Spark GraphX 等。Spark GraphX 是近期进入这一领域的新成员。

Spark GraphX 有何特别之处？Spark GraphX 是一个建立在 Spark 数据处理框架之上的图处理库。与其他图处理库相比，Spark GraphX 具有真正的优势。它可以利用 Spark 的所有数据处理能力。然而，在现实中，图处理算法的性能并不是唯一需要考虑的方面。

在许多应用中，需要建模为图的数据并不自然地以那种形式存在。在很多情况下，为了使图处理算法能够应用，需要花费大量的处理器时间和计算资源来将数据转换为正确的格式。这正是 Spark 数据处理框架与 Spark GraphX 库结合发挥价值的地方。使用 Spark 工具包中众多的工具，可以轻松完成使数据准备好供 Spark GraphX 消费的数据处理任务。总之，作为 Spark 家族一部分的 Spark GraphX 库，结合了 Spark 核心数据处理能力的强大功能和一个非常易于使用的图处理库。

再次回顾*图 3*所示的更大画面，以设定背景并了解正在讨论的内容，然后再深入到用例中。与其他章节不同，本章中的代码示例将仅使用 Scala，因为 Spark GraphX 库目前仅提供 Scala API。

![Spark GraphX 库](img/image_08_003.jpg)

*图 3*

## GraphX 概览

在任何现实世界的用例中，理解由顶点和边组成的图的概念很容易。但当涉及到实现时，即使是优秀的设计师和程序员也不太了解这种数据结构。原因很简单：与其他无处不在的数据结构（如列表、集合、映射、队列等）不同，图在大多数应用程序中并不常用。考虑到这一点，概念被逐步引入，一步一个脚印，通过简单和微不足道的例子，然后才涉及一些现实世界的用例。

Spark GraphX 库最重要的方面是一种数据类型，Graph，它扩展了 Spark **弹性分布式数据集**（**RDD**）并引入了一种新的图抽象。Spark GraphX 中的图抽象是有向多图，其所有顶点和边都附有属性。这些顶点和边的每个属性可以是 Scala 类型系统支持的用户定义类型。这些类型在 Graph 类型中参数化。给定的图可能需要为顶点或边使用不同的数据类型。这是通过使用继承层次结构相关的类型系统实现的。除了所有这些基本规则外，该库还包括一组图构建器和算法。

图中的一个顶点由一个唯一的 64 位长标识符 `org.apache.spark.graphx.VertexId` 标识。除了 VertexId 类型，简单的 Scala 类型 Long 也可以使用。此外，顶点可以采用任何类型作为属性。图中的边应具有源顶点标识符、目标顶点标识符和任何类型的属性。

*图 4* 展示了一个图，其顶点属性为字符串类型，边属性也为字符串类型。除了属性外，每个顶点都有一个唯一标识符，每条边都有源顶点编号和目标顶点编号。

![GraphX 概览](img/image_08_004.jpg)

*图 4*

在处理图时，有方法获取顶点和边。但这些孤立的图对象在处理时可能不足以满足需求。

如前所述，一个顶点具有其唯一的标识符和属性。一条边由其源顶点和目标顶点唯一标识。为了便于在图处理应用中处理每条边，Spark GraphX 库的三元组抽象提供了一种简便的方法，通过单个对象访问源顶点、目标顶点和边的属性。

以下 Scala 代码片段用于使用 Spark GraphX 库创建*图 4*中所示的图。创建图后，会调用图上的许多方法，这些方法展示了图的各种属性。在 Scala REPL 提示符下，尝试以下语句：

```scala
scala> import org.apache.spark._
  import org.apache.spark._
    scala> import org.apache.spark.graphx._

	import org.apache.spark.graphx._
	scala> import org.apache.spark.rdd.RDD
	import org.apache.spark.rdd.RDD
  scala> //Create an RDD of users containing tuple values with a mandatory
  Long and another String type as the property of the vertex
  scala> val users: RDD[(Long, String)] = sc.parallelize(Array((1L,
  "Thomas"), (2L, "Krish"),(3L, "Mathew")))
  users: org.apache.spark.rdd.RDD[(Long, String)] = ParallelCollectionRDD[0]
  at parallelize at <console>:31
  scala> //Created an RDD of Edge type with String type as the property of the edge
  scala> val userRelationships: RDD[Edge[String]] = sc.parallelize(Array(Edge(1L, 2L, "Follows"),    Edge(1L, 2L, "Son"),Edge(2L, 3L, "Follows")))
userRelationships: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[String]] = ParallelCollectionRDD[1] at parallelize at <console>:31
    scala> //Create a graph containing the vertex and edge RDDs as created beforescala> val userGraph = Graph(users, userRelationships)
	userGraph: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@ed5cf29

	scala> //Number of edges in the graph
	scala> userGraph.numEdges
      res3: Long = 3
    scala> //Number of vertices in the graph
	scala> userGraph.numVertices
      res4: Long = 3
	  scala> //Number of edges coming to each of the vertex. 
	  scala> userGraph.inDegrees
res7: org.apache.spark.graphx.VertexRDD[Int] = VertexRDDImpl[19] at RDD at
 VertexRDD.scala:57
scala> //The first element in the tuple is the vertex id and the second
 element in the tuple is the number of edges coming to that vertex
 scala> userGraph.inDegrees.foreach(println)
      (3,1)

      (2,2)
    scala> //Number of edges going out of each of the vertex. scala> userGraph.outDegrees
	res9: org.apache.spark.graphx.VertexRDD[Int] = VertexRDDImpl[23] at RDD at VertexRDD.scala:57
    scala> //The first element in the tuple is the vertex id and the second
	element in the tuple is the number of edges going out of that vertex
	scala> userGraph.outDegrees.foreach(println)
      (1,2)

      (2,1)
    scala> //Total number of edges coming in and going out of each vertex. 
	scala> userGraph.degrees
res12: org.apache.spark.graphx.VertexRDD[Int] = VertexRDDImpl[27] at RDD at
 VertexRDD.scala:57
    scala> //The first element in the tuple is the vertex id and the second 
	element in the tuple is the total number of edges coming in and going out of that vertex.
	scala> userGraph.degrees.foreach(println)
      (1,2)

      (2,3)

      (3,1)
    scala> //Get the vertices of the graph
	scala> userGraph.vertices
res11: org.apache.spark.graphx.VertexRDD[String] = VertexRDDImpl[11] at RDD at VertexRDD.scala:57
    scala> //Get all the vertices with the vertex number and the property as a tuplescala> userGraph.vertices.foreach(println)
      (1,Thomas)

      (3,Mathew)

      (2,Krish)
    scala> //Get the edges of the graph
	scala> userGraph.edges
res15: org.apache.spark.graphx.EdgeRDD[String] = EdgeRDDImpl[13] at RDD at
 EdgeRDD.scala:41
    scala> //Get all the edges properties with source and destination vertex numbers
	scala> userGraph.edges.foreach(println)
      Edge(1,2,Follows)

      Edge(1,2,Son)

      Edge(2,3,Follows)
    scala> //Get the triplets of the graph
	scala> userGraph.triplets
res18: org.apache.spark.rdd.RDD[org.apache.spark.graphx.EdgeTriplet[String,String]]
 = MapPartitionsRDD[32] at mapPartitions at GraphImpl.scala:48
    scala> userGraph.triplets.foreach(println)
	((1,Thomas),(2,Krish),Follows)
	((1,Thomas),(2,Krish),Son)
	((2,Krish),(3,Mathew),Follows)

```

读者将熟悉使用 RDD 进行 Spark 编程。上述代码片段阐明了使用 RDD 构建图的顶点和边的过程。RDD 可以使用各种数据存储中持久化的数据构建。在现实世界的用例中，大多数情况下数据将来自外部源，如 NoSQL 数据存储，并且有方法使用此类数据构建 RDD。一旦构建了 RDD，就可以使用它们来构建图。

上述代码片段还解释了图提供的各种方法，以获取给定图的所有必要详细信息。这里涉及的示例用例是一个规模非常小的图。在现实世界的用例中，图的顶点和边的数量可能达到数百万。由于所有这些抽象都作为 RDD 实现，因此固有的不可变性、分区、分布和并行处理的开箱即用特性使得图处理高度可扩展。最后，以下表格展示了顶点和边的表示方式：

**顶点表**：

| **顶点 ID** | **顶点属性** |
| --- | --- |
| 1 | Thomas |
| 2 | Krish |
| 3 | Mathew |

**边表**：

| **源顶点 ID** | **目标顶点 ID** | **边属性** |
| --- | --- | --- |
| 1 | 2 | Follows |
| 1 | 2 | Son |
| 2 | 3 | Follows |

**三元组表**：

| **源顶点 ID** | **目标顶点 ID** | **源顶点属性** | **边属性** | **目标顶点属性** |
| --- | --- | --- | --- | --- |
| 1 | 2 | Thomas | Follows | Krish |
| 1 | 2 | Thomas | Son | Krish |
| 2 | 3 | Krish | Follows | Mathew |

### 注意

需要注意的是，这些表格仅用于解释目的。实际的内部表示遵循 RDD 表示的规则和规定。

如果任何内容表示为 RDD，它必然会被分区并分布。但如果分区分布不受图的控制，那么在图处理性能方面将是次优的。因此，Spark GraphX 库的创建者提前充分考虑了这个问题，并实施了图分区策略，以便以 RDD 形式获得优化的图表示。

## 图分区

了解图 RDD 如何分区并在各个分区之间分布是很重要的。这对于确定图的各个组成部分 RDD 的分区和分布的高级优化非常有用。

通常，给定图有三个 RDD。除了顶点 RDD 和边 RDD 之外，还有一个内部使用的路由 RDD。为了获得最佳性能，构成给定边所需的所有顶点都保持在存储该边的同一分区中。如果某个顶点参与了多个边，并且这些边位于不同的分区中，那么该特定顶点可以存储在多个分区中。

为了跟踪给定顶点冗余存储的分区，还维护了一个路由 RDD，其中包含顶点详细信息以及每个顶点可用的分区。

*图 5*对此进行了解释：

![图分区](img/image_08_005.jpg)

图 5

*图 5*中，假设边被划分为分区 1 和 2。同样假设顶点被划分为分区 1 和 2。

在分区 1 中，所有边所需的顶点都可在本地获取。但在分区 2 中，只有一个边的顶点可在本地获取。因此，缺失的顶点也存储在分区 2 中，以便所有所需的顶点都可在本地获取。

为了跟踪复制情况，顶点路由 RDD 维护了给定顶点可用的分区编号。在*图 5*中，在顶点路由 RDD 中，使用标注符号来显示这些顶点被复制的分区。这样，在处理边或三元组时，所有与组成顶点相关的信息都可在本地获取，性能将高度优化。由于 RDD 是不可变的，即使它们存储在多个分区中，与信息更改相关的问题也被消除。

## 图处理

向用户展示的图的组成元素是顶点 RDD 和边 RDD。就像任何其他数据结构一样，由于底层数据的变化，图也会经历许多变化。为了使所需的图操作支持各种用例，有许多算法可用，使用这些算法可以处理图数据结构中隐藏的数据，以产生所需的业务成果。在深入了解处理图的算法之前，了解一些使用航空旅行用例的图处理基础知识是很有帮助的。

假设有人试图寻找从曼彻斯特到班加罗尔的廉价返程机票。在旅行偏好中，此人提到他/她不在乎中转次数，但价格应为最低。假设机票预订系统为往返旅程选择了相同的中转站，并生成了以下具有最低价格的路线或航段：

曼彻斯特 → 伦敦 → 科伦坡 → 班加罗尔

班加罗尔 → 科伦坡 → 伦敦 → 曼彻斯特

该路线规划是一个图的完美示例。如果将前行旅程视为一个图，将返程视为另一个图，那么返程图可以通过反转前行旅程图来生成。在 Scala REPL 提示符下，尝试以下语句：

```scala
scala> import org.apache.spark._
import org.apache.spark._
scala> import org.apache.spark.graphx._
import org.apache.spark.graphx._
scala> import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD
scala> //Create the vertices with the stops
scala> val stops: RDD[(Long, String)] = sc.parallelize(Array((1L, "Manchester"), (2L, "London"),(3L, "Colombo"), (4L, "Bangalore")))
stops: org.apache.spark.rdd.RDD[(Long, String)] = ParallelCollectionRDD[33] at parallelize at <console>:38
scala> //Create the edges with travel legs
scala> val legs: RDD[Edge[String]] = sc.parallelize(Array(Edge(1L, 2L, "air"),    Edge(2L, 3L, "air"),Edge(3L, 4L, "air"))) 
legs: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[String]] = ParallelCollectionRDD[34] at parallelize at <console>:38 
scala> //Create the onward journey graph
scala> val onwardJourney = Graph(stops, legs)onwardJourney: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@190ec769scala> onwardJourney.triplets.map(triplet => (triplet.srcId, (triplet.srcAttr, triplet.dstAttr))).sortByKey().collect().foreach(println)
(1,(Manchester,London))
(2,(London,Colombo))
(3,(Colombo,Bangalore))
scala> val returnJourney = onwardJourney.reversereturnJourney: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@60035f1e
scala> returnJourney.triplets.map(triplet => (triplet.srcId, (triplet.srcAttr,triplet.dstAttr))).sortByKey(ascending=false).collect().foreach(println)
(4,(Bangalore,Colombo))
(3,(Colombo,London))
(2,(London,Manchester))

```

前行旅程航段的起点和终点在返程航段中被反转。当图被反转时，只有边的起点和终点顶点被反转，顶点的身份保持不变。

换言之，每个顶点的标识符保持不变。在处理图时，了解三元组属性的名称很重要。它们对于编写程序和处理图很有用。在同一个 Scala REPL 会话中，尝试以下语句：

```scala
scala> returnJourney.triplets.map(triplet => (triplet.srcId,triplet.dstId,triplet.attr,triplet.srcAttr,triplet.dstAttr)).foreach(println) 
(2,1,air,London,Manchester) 
(3,2,air,Colombo,London) 
(4,3,air,Bangalore,Colombo) 

```

下表列出了可用于处理图并从图中提取所需数据的三元组属性。前面的代码片段和下表可以交叉验证，以便完全理解：

| **三元组属性** | **描述** |
| --- | --- |
| `srcId` | 源顶点标识符 |
| `dstId` | 目标顶点标识符 |
| `attr` | 边属性 |
| `srcAttr` | 源顶点属性 |
| `dstAttr` | 目标顶点属性 |

在图中，顶点是 RDD，边是 RDD，仅凭这一点，就可以进行转换。

现在，为了演示图转换，我们使用相同的用例，但稍作改动。假设一个旅行社从航空公司获得了某些路线的特别折扣价格。旅行社决定保留折扣，并向客户提供市场价格，为此，他们将航空公司给出的价格提高了 10%。这个旅行社注意到机场名称显示不一致，并希望确保在整个网站上显示时有一致的表示，因此决定将所有停靠点名称改为大写。在同一个 Scala REPL 会话中，尝试以下语句：

```scala
 scala> // Create the vertices 
scala> val stops: RDD[(Long, String)] = sc.parallelize(Array((1L,
 "Manchester"), (2L, "London"),(3L, "Colombo"), (4L, "Bangalore"))) 
stops: org.apache.spark.rdd.RDD[(Long, String)] = ParallelCollectionRDD[66] at parallelize at <console>:38 
scala> //Create the edges 
scala> val legs: RDD[Edge[Long]] = sc.parallelize(Array(Edge(1L, 2L, 50L),    Edge(2L, 3L, 100L),Edge(3L, 4L, 80L))) 
legs: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[Long]] = ParallelCollectionRDD[67] at parallelize at <console>:38 
scala> //Create the graph using the vertices and edges 
scala> val journey = Graph(stops, legs) 
journey: org.apache.spark.graphx.Graph[String,Long] = org.apache.spark.graphx.impl.GraphImpl@8746ad5 
scala> //Convert the stop names to upper case 
scala> val newStops = journey.vertices.map {case (id, name) => (id, name.toUpperCase)} 
newStops: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, String)] = MapPartitionsRDD[80] at map at <console>:44 
scala> //Get the edges from the selected journey and add 10% price to the original price 
scala> val newLegs = journey.edges.map { case Edge(src, dst, prop) => Edge(src, dst, (prop + (0.1*prop))) } 
newLegs: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[Double]] = MapPartitionsRDD[81] at map at <console>:44 
scala> //Create a new graph with the original vertices and the new edges 
scala> val newJourney = Graph(newStops, newLegs) 
newJourney: org.apache.spark.graphx.Graph[String,Double]
 = org.apache.spark.graphx.impl.GraphImpl@3c929623 
scala> //Print the contents of the original graph 
scala> journey.triplets.foreach(println) 
((1,Manchester),(2,London),50) 
((3,Colombo),(4,Bangalore),80) 
((2,London),(3,Colombo),100) 
scala> //Print the contents of the transformed graph 
scala>  newJourney.triplets.foreach(println) 
((2,LONDON),(3,COLOMBO),110.0) 
((3,COLOMBO),(4,BANGALORE),88.0) 
((1,MANCHESTER),(2,LONDON),55.0) 

```

实质上，这些转换确实是 RDD 转换。如果有关于这些不同的 RDD 如何组合在一起形成图的概念理解，任何具有 RDD 编程熟练度的程序员都能很好地进行图处理。这是 Spark 统一编程模型的另一个证明。

前面的用例对顶点和边 RDD 进行了映射转换。类似地，过滤转换是另一种常用的有用类型。除了这些，所有转换和操作都可以用于处理顶点和边 RDD。

## 图结构处理

在前一节中，通过单独处理所需的顶点或边完成了一种图处理。这种方法的一个缺点是处理过程分为三个不同的阶段，如下：

+   从图中提取顶点或边

+   处理顶点或边

+   使用处理过的顶点和边重新创建一个新图

这种方法繁琐且容易出错。为了解决这个问题，Spark GraphX 库提供了一些结构化操作符，允许用户将图作为一个单独的单元进行处理，从而生成一个新的图。

前一节已经讨论了一个重要的结构化操作，即图的反转，它生成一个所有边方向反转的新图。另一个常用的结构化操作是从给定图中提取子图。所得子图可以是整个父图，也可以是父图的子集，具体取决于对父图执行的操作。

当从外部数据源创建图时，边可能包含无效顶点。如果顶点和边来自两个不同的数据源或应用程序，这种情况非常可能发生。使用这些顶点和边创建的图，其中一些边将包含无效顶点，处理结果将出现意外。以下是一个用例，其中一些包含无效顶点的边通过结构化操作进行修剪以消除这种情况。在 Scala REPL 提示符下，尝试以下语句：

```scala
scala> import org.apache.spark._
  import org.apache.spark._    scala> import org.apache.spark.graphx._
  import org.apache.spark.graphx._    scala> import org.apache.spark.rdd.RDD
  import org.apache.spark.rdd.RDD    scala> //Create an RDD of users containing tuple values with a mandatory
  Long and another String type as the property of the vertex
  scala> val users: RDD[(Long, String)] = sc.parallelize(Array((1L,
  "Thomas"), (2L, "Krish"),(3L, "Mathew")))
users: org.apache.spark.rdd.RDD[(Long, String)] = ParallelCollectionRDD[104]
 at parallelize at <console>:45
    scala> //Created an RDD of Edge type with String type as the property of
	the edge
	scala> val userRelationships: RDD[Edge[String]] =
	sc.parallelize(Array(Edge(1L, 2L, "Follows"), Edge(1L, 2L,
	"Son"),Edge(2L, 3L, "Follows"), Edge(1L, 4L, "Follows"), Edge(3L, 4L, "Follows")))
	userRelationships:
	org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[String]] =
	ParallelCollectionRDD[105] at parallelize at <console>:45
    scala> //Create a vertex property object to fill in if an invalid vertex id is given in the edge
	scala> val missingUser = "Missing"
missingUser: String = Missing
    scala> //Create a graph containing the vertex and edge RDDs as created
	before
	scala> val userGraph = Graph(users, userRelationships, missingUser)
userGraph: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@43baf0b9
    scala> //List the graph triplets and find some of the invalid vertex ids given and for them the missing vertex property is assigned with the value "Missing"scala> userGraph.triplets.foreach(println)
      ((3,Mathew),(4,Missing),Follows)  
      ((1,Thomas),(2,Krish),Son)    
      ((2,Krish),(3,Mathew),Follows)    
      ((1,Thomas),(2,Krish),Follows)    
      ((1,Thomas),(4,Missing),Follows)
    scala> //Since the edges with the invalid vertices are invalid too, filter out
	those vertices and create a valid graph. The vertex predicate here can be any valid filter condition of a vertex. Similar to vertex predicate, if the filtering is to be done on the edges, instead of the vpred, use epred as the edge predicate.
	scala> val fixedUserGraph = userGraph.subgraph(vpred = (vertexId, attribute) => attribute != "Missing")
fixedUserGraph: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@233b5c71 
 scala> fixedUserGraph.triplets.foreach(println)
  ((2,Krish),(3,Mathew),Follows)
  ((1,Thomas),(2,Krish),Follows)
  ((1,Thomas),(2,Krish),Son)

```

在大型图中，根据具体用例，有时可能存在大量平行边。在某些用例中，可以将平行边的数据合并并仅保留一条边，而不是维护许多平行边。在前述用例中，最终没有无效边的图，存在平行边，一条具有属性`Follows`，另一条具有`Son`，它们具有相同的源和目标顶点。

将这些平行边合并为一条具有从平行边串联属性的单一边是可行的，这将减少边的数量而不丢失信息。这是通过图的 groupEdges 结构化操作实现的。在同一 Scala REPL 会话中，尝试以下语句：

```scala
scala> // Import the partition strategy classes 
scala> import org.apache.spark.graphx.PartitionStrategy._ 
import org.apache.spark.graphx.PartitionStrategy._ 
scala> // Partition the user graph. This is required to group the edges 
scala> val partitionedUserGraph = fixedUserGraph.partitionBy(CanonicalRandomVertexCut) 
partitionedUserGraph: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@5749147e 
scala> // Generate the graph without parallel edges and combine the properties of duplicate edges 
scala> val graphWithoutParallelEdges = partitionedUserGraph.groupEdges((e1, e2) => e1 + " and " + e2) 
graphWithoutParallelEdges: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@16a4961f 
scala> // Print the details 
scala> graphWithoutParallelEdges.triplets.foreach(println) 
((1,Thomas),(2,Krish),Follows and Son) 
((2,Krish),(3,Mathew),Follows) 

```

之前的图结构变化通过聚合边减少了边的数量。当边属性为数值型且通过聚合进行合并有意义时，也可以通过移除平行边来减少边的数量，这能显著减少图处理时间。

### 注意

本代码片段中一个重要点是，在边上执行 group-by 操作之前，图已经进行了分区。

默认情况下，给定图的边及其组成顶点无需位于同一分区。为了使 group-by 操作生效，所有平行边必须位于同一分区。CanonicalRandomVertexCut 分区策略确保两个顶点之间的所有边，无论方向如何，都能实现共置。

在 Spark GraphX 库中还有更多结构化操作符可供使用，查阅 Spark 文档可以深入了解这些操作符，它们可根据具体用例进行应用。

# 网球锦标赛分析

既然基本的图处理基础已经就位，现在是时候采用一个使用图的现实世界用例了。这里，我们使用图来模拟一场网球锦标赛的结果。使用图来模拟 2015 年巴克莱 ATP 世界巡回赛单打比赛的结果。顶点包含球员详情，边包含个人比赛记录。边的形成方式是，源顶点是赢得比赛的球员，目标顶点是输掉比赛的球员。边属性包含比赛类型、赢家在比赛中获得的分数以及球员之间的交锋次数。这里使用的积分系统是虚构的，仅仅是赢家在那场比赛中获得的权重。小组赛初赛权重最低，半决赛权重更高，决赛权重最高。通过这种方式模拟结果，处理图表以找出以下详细信息：

+   列出所有比赛详情。

+   列出所有比赛，包括球员姓名、比赛类型和结果。

+   列出所有小组 1 的获胜者及其比赛中的积分。

+   列出所有小组 2 的获胜者及其比赛中的积分。

+   列出所有半决赛获胜者及其比赛中的积分。

+   列出决赛获胜者及其比赛中的积分。

+   列出球员在整个锦标赛中获得的总积分。

+   通过找出得分最高的球员来列出比赛获胜者。

+   在小组赛阶段，由于循环赛制，同一组球员可能会多次相遇。查找是否有任何球员在这场锦标赛中相互比赛超过一次。

+   列出至少赢得一场比赛的球员。

+   列出至少输掉一场比赛的球员。

+   列出至少赢得一场比赛且至少输掉一场比赛的球员。

+   列出完全没有获胜的球员。

+   列出完全没有输掉比赛的球员。

对于不熟悉网球比赛的人来说，无需担心，因为这里不讨论比赛规则，也不需要理解这个用例。实际上，我们只将其视为两人之间的比赛，其中一人获胜，另一人输掉。在 Scala REPL 提示符下，尝试以下语句：

```scala
scala> import org.apache.spark._
  import org.apache.spark._    
  scala> import org.apache.spark.graphx._
  import org.apache.spark.graphx._    
  scala> import org.apache.spark.rdd.RDD
  import org.apache.spark.rdd.RDD
    scala> //Define a property class that is going to hold all the properties of the vertex which is nothing but player information
	scala> case class Player(name: String, country: String)
      defined class Player
    scala> // Create the player vertices
	scala> val players: RDD[(Long, Player)] = sc.parallelize(Array((1L, Player("Novak Djokovic", "SRB")), (3L, Player("Roger Federer", "SUI")),(5L, Player("Tomas Berdych", "CZE")), (7L, Player("Kei Nishikori", "JPN")), (11L, Player("Andy Murray", "GBR")),(15L, Player("Stan Wawrinka", "SUI")),(17L, Player("Rafael Nadal", "ESP")),(19L, Player("David Ferrer", "ESP"))))
players: org.apache.spark.rdd.RDD[(Long, Player)] = ParallelCollectionRDD[145] at parallelize at <console>:57
    scala> //Define a property class that is going to hold all the properties of the edge which is nothing but match informationscala> case class Match(matchType: String, points: Int, head2HeadCount: Int)
      defined class Match
    scala> // Create the match edgesscala> val matches: RDD[Edge[Match]] = sc.parallelize(Array(Edge(1L, 5L, Match("G1", 1,1)), Edge(1L, 7L, Match("G1", 1,1)), Edge(3L, 1L, Match("G1", 1,1)), Edge(3L, 5L, Match("G1", 1,1)), Edge(3L, 7L, Match("G1", 1,1)), Edge(7L, 5L, Match("G1", 1,1)), Edge(11L, 19L, Match("G2", 1,1)), Edge(15L, 11L, Match("G2", 1, 1)), Edge(15L, 19L, Match("G2", 1, 1)), Edge(17L, 11L, Match("G2", 1, 1)), Edge(17L, 15L, Match("G2", 1, 1)), Edge(17L, 19L, Match("G2", 1, 1)), Edge(3L, 15L, Match("S", 5, 1)), Edge(1L, 17L, Match("S", 5, 1)), Edge(1L, 3L, Match("F", 11, 1))))
matches: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[Match]] = ParallelCollectionRDD[146] at parallelize at <console>:57
    scala> //Create a graph with the vertices and edges
	scala> val playGraph = Graph(players, matches)
playGraph: org.apache.spark.graphx.Graph[Player,Match] = org.apache.spark.graphx.impl.GraphImpl@30d4d6fb 

```

包含网球锦标赛的图已经创建，从现在开始，所有要做的是处理这个基础图并从中提取信息以满足用例需求：

```scala
scala> //Print the match details
	scala> playGraph.triplets.foreach(println)
((15,Player(Stan Wawrinka,SUI)),(11,Player(Andy Murray,GBR)),Match(G2,1,1))    
((15,Player(Stan Wawrinka,SUI)),(19,Player(David Ferrer,ESP)),Match(G2,1,1))    
((7,Player(Kei Nishikori,JPN)),(5,Player(Tomas Berdych,CZE)),Match(G1,1,1))    
((1,Player(Novak Djokovic,SRB)),(7,Player(Kei Nishikori,JPN)),Match(G1,1,1))    
((3,Player(Roger Federer,SUI)),(1,Player(Novak Djokovic,SRB)),Match(G1,1,1))    
((1,Player(Novak Djokovic,SRB)),(3,Player(Roger Federer,SUI)),Match(F,11,1))    
((1,Player(Novak Djokovic,SRB)),(17,Player(Rafael Nadal,ESP)),Match(S,5,1))    
((3,Player(Roger Federer,SUI)),(5,Player(Tomas Berdych,CZE)),Match(G1,1,1))    
((17,Player(Rafael Nadal,ESP)),(11,Player(Andy Murray,GBR)),Match(G2,1,1))    
((3,Player(Roger Federer,SUI)),(7,Player(Kei Nishikori,JPN)),Match(G1,1,1))    
((1,Player(Novak Djokovic,SRB)),(5,Player(Tomas Berdych,CZE)),Match(G1,1,1))    
((17,Player(Rafael Nadal,ESP)),(15,Player(Stan Wawrinka,SUI)),Match(G2,1,1))    
((11,Player(Andy Murray,GBR)),(19,Player(David Ferrer,ESP)),Match(G2,1,1))    
((3,Player(Roger Federer,SUI)),(15,Player(Stan Wawrinka,SUI)),Match(S,5,1))    
((17,Player(Rafael Nadal,ESP)),(19,Player(David Ferrer,ESP)),Match(G2,1,1))
    scala> //Print matches with player names and the match type and the resultscala> playGraph.triplets.map(triplet => triplet.srcAttr.name + " won over " + triplet.dstAttr.name + " in  " + triplet.attr.matchType + " match").foreach(println)
      Roger Federer won over Tomas Berdych in  G1 match    
      Roger Federer won over Kei Nishikori in  G1 match    
      Novak Djokovic won over Roger Federer in  F match    
      Novak Djokovic won over Rafael Nadal in  S match    
      Roger Federer won over Stan Wawrinka in  S match    
      Rafael Nadal won over David Ferrer in  G2 match    
      Kei Nishikori won over Tomas Berdych in  G1 match    
      Andy Murray won over David Ferrer in  G2 match    
      Stan Wawrinka won over Andy Murray in  G2 match    
      Stan Wawrinka won over David Ferrer in  G2 match    
      Novak Djokovic won over Kei Nishikori in  G1 match    
      Roger Federer won over Novak Djokovic in  G1 match    
      Rafael Nadal won over Andy Murray in  G2 match    
      Rafael Nadal won over Stan Wawrinka in  G2 match    
      Novak Djokovic won over Tomas Berdych in  G1 match 

```

值得注意的是，在图形中使用三元组对于提取给定网球比赛的所有必需数据元素非常方便，包括谁在比赛、谁获胜以及比赛类型，这些都可以从一个对象中获取。以下分析用例的实现涉及筛选锦标赛的网球比赛记录。这里仅使用了简单的筛选逻辑，但在实际用例中，任何复杂的逻辑都可以在函数中实现，并作为参数传递给筛选转换：

```scala
scala> //Group 1 winners with their group total points
scala> playGraph.triplets.filter(triplet => triplet.attr.matchType == "G1").map(triplet => (triplet.srcAttr.name, triplet.attr.points)).foreach(println)
      (Kei Nishikori,1)    
      (Roger Federer,1)    
      (Roger Federer,1)    
      (Novak Djokovic,1)    
      (Novak Djokovic,1)    
      (Roger Federer,1)
    scala> //Find the group total of the players
	scala> playGraph.triplets.filter(triplet => triplet.attr.matchType == "G1").map(triplet => (triplet.srcAttr.name, triplet.attr.points)).reduceByKey(_+_).foreach(println)
      (Roger Federer,3)    
      (Novak Djokovic,2)    
      (Kei Nishikori,1)
    scala> //Group 2 winners with their group total points
	scala> playGraph.triplets.filter(triplet => triplet.attr.matchType == "G2").map(triplet => (triplet.srcAttr.name, triplet.attr.points)).foreach(println)
      (Rafael Nadal,1)    
      (Rafael Nadal,1)    
      (Andy Murray,1)    
      (Stan Wawrinka,1)    
      (Stan Wawrinka,1)    
      (Rafael Nadal,1) 

```

以下分析用例的实现涉及按键分组并进行汇总计算。它不仅限于查找网球比赛记录点的总和，如以下用例实现所示；实际上，可以使用用户定义的函数进行计算：

```scala
scala> //Find the group total of the players
	scala> playGraph.triplets.filter(triplet => triplet.attr.matchType == "G2").map(triplet => (triplet.srcAttr.name, triplet.attr.points)).reduceByKey(_+_).foreach(println)
      (Stan Wawrinka,2)    
      (Andy Murray,1)    
      (Rafael Nadal,3)
    scala> //Semi final winners with their group total points
	scala> playGraph.triplets.filter(triplet => triplet.attr.matchType == "S").map(triplet => (triplet.srcAttr.name, triplet.attr.points)).foreach(println)
      (Novak Djokovic,5)    
      (Roger Federer,5)
    scala> //Find the group total of the players
	scala> playGraph.triplets.filter(triplet => triplet.attr.matchType == "S").map(triplet => (triplet.srcAttr.name, triplet.attr.points)).reduceByKey(_+_).foreach(println)
      (Novak Djokovic,5)    
      (Roger Federer,5)
    scala> //Final winner with the group total points
	scala> playGraph.triplets.filter(triplet => triplet.attr.matchType == "F").map(triplet => (triplet.srcAttr.name, triplet.attr.points)).foreach(println)
      (Novak Djokovic,11)
    scala> //Tournament total point standing
	scala> playGraph.triplets.map(triplet => (triplet.srcAttr.name, triplet.attr.points)).reduceByKey(_+_).foreach(println)
      (Stan Wawrinka,2)

      (Rafael Nadal,3)    
      (Kei Nishikori,1)    
      (Andy Murray,1)    
      (Roger Federer,8)    
      (Novak Djokovic,18)
    scala> //Find the winner of the tournament by finding the top scorer of the tournament
	scala> playGraph.triplets.map(triplet => (triplet.srcAttr.name, triplet.attr.points)).reduceByKey(_+_).map{ case (k,v) => (v,k)}.sortByKey(ascending=false).take(1).map{ case (k,v) => (v,k)}.foreach(println)
      (Novak Djokovic,18)
    scala> //Find how many head to head matches held for a given set of players in the descending order of head2head count
	scala> playGraph.triplets.map(triplet => (Set(triplet.srcAttr.name , triplet.dstAttr.name) , triplet.attr.head2HeadCount)).reduceByKey(_+_).map{case (k,v) => (k.mkString(" and "), v)}.map{ case (k,v) => (v,k)}.sortByKey().map{ case (k,v) => v + " played " + k + " time(s)"}.foreach(println)
      Roger Federer and Novak Djokovic played 2 time(s)    
      Roger Federer and Tomas Berdych played 1 time(s)    
      Kei Nishikori and Tomas Berdych played 1 time(s)    
      Novak Djokovic and Tomas Berdych played 1 time(s)    
      Rafael Nadal and Andy Murray played 1 time(s)    
      Rafael Nadal and Stan Wawrinka played 1 time(s)    
      Andy Murray and David Ferrer played 1 time(s)    
      Rafael Nadal and David Ferrer played 1 time(s)    
      Stan Wawrinka and David Ferrer played 1 time(s)    
      Stan Wawrinka and Andy Murray played 1 time(s)    
      Roger Federer and Stan Wawrinka played 1 time(s)    
      Roger Federer and Kei Nishikori played 1 time(s)    
      Novak Djokovic and Kei Nishikori played 1 time(s)    
      Novak Djokovic and Rafael Nadal played 1 time(s) 

```

以下分析用例的实现涉及从查询中查找唯一记录。Spark 的 distinct 转换可以实现这一点：

```scala
 scala> //List of players who have won at least one match
	scala> val winners = playGraph.triplets.map(triplet => triplet.srcAttr.name).distinct
winners: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[201] at distinct at <console>:65
    scala> winners.foreach(println)
      Kei Nishikori    
      Stan Wawrinka    
      Andy Murray    
      Roger Federer    
      Rafael Nadal    
      Novak Djokovic
    scala> //List of players who have lost at least one match
	scala> val loosers = playGraph.triplets.map(triplet => triplet.dstAttr.name).distinct
loosers: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[205] at distinct at <console>:65
    scala> loosers.foreach(println)
      Novak Djokovic    
      Kei Nishikori    
      David Ferrer    
      Stan Wawrinka    
      Andy Murray    
      Roger Federer    
      Rafael Nadal    
      Tomas Berdych
    scala> //List of players who have won at least one match and lost at least one match
	scala> val wonAndLost = winners.intersection(loosers)
wonAndLost: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[211] at intersection at <console>:69
    scala> wonAndLost.foreach(println)
      Novak Djokovic    
      Rafael Nadal    
      Andy Murray    
      Roger Federer    
      Kei Nishikori    
      Stan Wawrinka 
    scala> //List of players who have no wins at all
	scala> val lostAndNoWins = loosers.collect().toSet -- wonAndLost.collect().toSet
lostAndNoWins: 
scala.collection.immutable.Set[String] = Set(David Ferrer, Tomas Berdych)
    scala> lostAndNoWins.foreach(println)
      David Ferrer    
      Tomas Berdych
    scala> //List of players who have no loss at all
	scala> val wonAndNoLosses = winners.collect().toSet -- loosers.collect().toSet
 wonAndNoLosses: 
	  scala.collection.immutable.Set[String] = Set() 
scala> //The val wonAndNoLosses returned an empty set which means that there is no single player in this tournament who have only wins
scala> wonAndNoLosses.foreach(println)

```

在这个用例中，并没有花费太多精力来美化结果，因为它们被简化为简单的 RDD 结构，可以使用本书前几章已经介绍的 RDD 编程技术根据需要进行操作。

Spark 的高度简洁和统一的编程模型，结合 Spark GraphX 库，帮助开发者用很少的代码构建实际用例。这也表明，一旦使用相关数据构建了正确的图形结构，并使用支持的图形操作，就可以揭示隐藏在底层数据中的许多真相。

# 应用 PageRank 算法

由 Sergey Brin 和 Lawrence Page 撰写的研究论文，题为*The Anatomy of a Large-Scale Hypertextual Web Search Engine*，彻底改变了网络搜索，Google 基于这一 PageRank 概念构建了其搜索引擎，并主导了其他网络搜索引擎。

使用 Google 搜索网页时，其算法排名高的页面会被显示。在图形的上下文中，如果基于相同的算法对顶点进行排名，可以得出许多新的推断。从表面上看，这个 PageRank 算法似乎只对网络搜索有用。但它具有巨大的潜力，可以应用于许多其他领域。

在图形术语中，如果存在一条边 E，连接两个顶点，从 V1 到 V2，根据 PageRank 算法，V2 比 V1 更重要。在一个包含大量顶点和边的巨大图形中，可以计算出每个顶点的 PageRank。

上一节中提到的网球锦标赛分析用例，PageRank 算法可以很好地应用于此。在此采用的图表示中，每场比赛都表示为一个边。源顶点包含获胜者的详细信息，而目标顶点包含失败者的详细信息。在网球比赛中，如果可以将这称为某种虚构的重要性排名，那么在一场比赛中，获胜者的重要性排名高于失败者。

如果在前述用例中采用的图来演示 PageRank 算法，那么该图必须反转，使得每场比赛的获胜者成为每个边的目标顶点。在 Scala REPL 提示符下，尝试以下语句：

```scala
scala> import org.apache.spark._
  import org.apache.spark._ 
  scala> import org.apache.spark.graphx._
  import org.apache.spark.graphx._    
  scala> import org.apache.spark.rdd.RDD
  import org.apache.spark.rdd.RDD
    scala> //Define a property class that is going to hold all the properties of the vertex which is nothing but player informationscala> case class Player(name: String, country: String)
      defined class Player
    scala> // Create the player verticesscala> val players: RDD[(Long, Player)] = sc.parallelize(Array((1L, Player("Novak Djokovic", "SRB")), (3L, Player("Roger Federer", "SUI")),(5L, Player("Tomas Berdych", "CZE")), (7L, Player("Kei Nishikori", "JPN")), (11L, Player("Andy Murray", "GBR")),(15L, Player("Stan Wawrinka", "SUI")),(17L, Player("Rafael Nadal", "ESP")),(19L, Player("David Ferrer", "ESP"))))
players: org.apache.spark.rdd.RDD[(Long, Player)] = ParallelCollectionRDD[212] at parallelize at <console>:64
    scala> //Define a property class that is going to hold all the properties of the edge which is nothing but match informationscala> case class Match(matchType: String, points: Int, head2HeadCount: Int)
      defined class Match
    scala> // Create the match edgesscala> val matches: RDD[Edge[Match]] = sc.parallelize(Array(Edge(1L, 5L, Match("G1", 1,1)), Edge(1L, 7L, Match("G1", 1,1)), Edge(3L, 1L, Match("G1", 1,1)), Edge(3L, 5L, Match("G1", 1,1)), Edge(3L, 7L, Match("G1", 1,1)), Edge(7L, 5L, Match("G1", 1,1)), Edge(11L, 19L, Match("G2", 1,1)), Edge(15L, 11L, Match("G2", 1, 1)), Edge(15L, 19L, Match("G2", 1, 1)), Edge(17L, 11L, Match("G2", 1, 1)), Edge(17L, 15L, Match("G2", 1, 1)), Edge(17L, 19L, Match("G2", 1, 1)), Edge(3L, 15L, Match("S", 5, 1)), Edge(1L, 17L, Match("S", 5, 1)), Edge(1L, 3L, Match("F", 11, 1))))
matches: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[Match]] = ParallelCollectionRDD[213] at parallelize at <console>:64
    scala> //Create a graph with the vertices and edgesscala> val playGraph = Graph(players, matches)
playGraph: org.apache.spark.graphx.Graph[Player,Match] = org.apache.spark.graphx.impl.GraphImpl@263cd0e2
    scala> //Reverse this graph to have the winning player coming in the destination vertex
	scala> val rankGraph = playGraph.reverse
rankGraph: org.apache.spark.graphx.Graph[Player,Match] = org.apache.spark.graphx.impl.GraphImpl@7bb131fb
    scala> //Run the PageRank algorithm to calculate the rank of each vertex
	scala> val rankedVertices = rankGraph.pageRank(0.0001).vertices
rankedVertices: org.apache.spark.graphx.VertexRDD[Double] = VertexRDDImpl[1184] at RDD at VertexRDD.scala:57
    scala> //Extract the vertices sorted by the rank
	scala> val rankedPlayers = rankedVertices.join(players).map{case 
	(id,(importanceRank,Player(name,country))) => (importanceRank,
	name)}.sortByKey(ascending=false)

	rankedPlayers: org.apache.spark.rdd.RDD[(Double, String)] = ShuffledRDD[1193] at sortByKey at <console>:76

	scala> rankedPlayers.collect().foreach(println)
      (3.382662570589846,Novak Djokovic)    
      (3.266079758089846,Roger Federer)    
      (0.3908953124999999,Rafael Nadal)    
      (0.27431249999999996,Stan Wawrinka)    
      (0.1925,Andy Murray)    
      (0.1925,Kei Nishikori)    
      (0.15,David Ferrer)    
      (0.15,Tomas Berdych) 

```

如果仔细审查上述代码，可以看出排名最高的玩家赢得了最多的比赛。

# 连通分量算法

在图中，寻找由相连顶点组成的子图是一个非常常见的需求，具有广泛的应用。在任何图中，两个通过一条或多条边组成的路径相连的顶点，并且不与同一图中的任何其他顶点相连，被称为连通分量。例如，在图 G 中，顶点 V1 通过一条边与 V2 相连，V2 通过另一条边与 V3 相连。在同一图 G 中，顶点 V4 通过另一条边与 V5 相连。在这种情况下，V1 和 V3 相连，V4 和 V5 相连，而 V1 和 V5 不相连。在图 G 中，有两个连通分量。Spark GraphX 库实现了连通分量算法。

在社交网络应用中，如果用户之间的连接被建模为图，那么检查给定用户是否与另一用户相连，可以通过检查这两个顶点是否存在连通分量来实现。在计算机游戏中，从点 A 到点 B 的迷宫穿越可以通过将迷宫交汇点建模为顶点，将连接交汇点的路径建模为图中的边，并使用连通分量算法来实现。

在计算机网络中，检查数据包是否可以从一个 IP 地址发送到另一个 IP 地址，是通过使用连通分量算法实现的。在物流应用中，例如快递服务，检查包裹是否可以从点 A 发送到点 B，也是通过使用连通分量算法实现的。*图 6*展示了一个具有三个连通分量的图：

![连通分量算法](img/image_08_006.jpg)

图 6

*图 6*是图的图形表示。其中，有三个*簇*的顶点通过边相连。换句话说，该图中有三个连通分量。

这里再次以社交网络应用中用户相互关注的用例为例，以阐明其原理。通过提取图的连通分量，可以查看任意两个用户是否相连。*图 7* 展示了用户图：

![连通分量算法](img/image_08_007.jpg)

图 7

在 *图 7* 所示的图中，很明显可以看出存在两个连通分量。可以轻松判断 Thomas 和 Mathew 相连，而 Thomas 和 Martin 不相连。如果提取连通分量图，可以看到 Thomas 和 Martin 将具有相同的连通分量标识符，同时 Thomas 和 Martin 将具有不同的连通分量标识符。在 Scala REPL 提示符下，尝试以下语句：

```scala
	 scala> import org.apache.spark._

  import org.apache.spark._    
  scala> import org.apache.spark.graphx._

  import org.apache.spark.graphx._    
  scala> import org.apache.spark.rdd.RDD

  import org.apache.spark.rdd.RDD    

  scala> // Create the RDD with users as the vertices
  scala> val users: RDD[(Long, String)] = sc.parallelize(Array((1L, "Thomas"), (2L, "Krish"),(3L, "Mathew"), (4L, "Martin"), (5L, "George"), (6L, "James")))

users: org.apache.spark.rdd.RDD[(Long, String)] = ParallelCollectionRDD[1194] at parallelize at <console>:69

	scala> // Create the edges connecting the users
	scala> val userRelationships: RDD[Edge[String]] = sc.parallelize(Array(Edge(1L, 2L, "Follows"),Edge(2L, 3L, "Follows"), Edge(4L, 5L, "Follows"), Edge(5L, 6L, "Follows")))

userRelationships: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[String]] = ParallelCollectionRDD[1195] at parallelize at <console>:69

	scala> // Create a graph
	scala> val userGraph = Graph(users, userRelationships)

userGraph: org.apache.spark.graphx.Graph[String,String] = org.apache.spark.graphx.impl.GraphImpl@805e363

	scala> // Find the connected components of the graph
	scala> val cc = userGraph.connectedComponents()

cc: org.apache.spark.graphx.Graph[org.apache.spark.graphx.VertexId,String] = org.apache.spark.graphx.impl.GraphImpl@13f4a9a9

	scala> // Extract the triplets of the connected components
	scala> val ccTriplets = cc.triplets

ccTriplets: org.apache.spark.rdd.RDD[org.apache.spark.graphx.EdgeTriplet[org.apache.spark.graphx.VertexId,String]] = MapPartitionsRDD[1263] at mapPartitions at GraphImpl.scala:48

	scala> // Print the structure of the tripletsscala> ccTriplets.foreach(println)
      ((1,1),(2,1),Follows)    

      ((4,4),(5,4),Follows)    

      ((5,4),(6,4),Follows)    

      ((2,1),(3,1),Follows)

	scala> //Print the vertex numbers and the corresponding connected component id. The connected component id is generated by the system and it is to be taken only as a unique identifier for the connected component
	scala> val ccProperties = ccTriplets.map(triplet => "Vertex " + triplet.srcId + " and " + triplet.dstId + " are part of the CC with id " + triplet.srcAttr)

ccProperties: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[1264] at map at <console>:79

	scala> ccProperties.foreach(println)

      Vertex 1 and 2 are part of the CC with id 1    

      Vertex 5 and 6 are part of the CC with id 4    

      Vertex 2 and 3 are part of the CC with id 1    

      Vertex 4 and 5 are part of the CC with id 4

	scala> //Find the users in the source vertex with their CC id
	scala> val srcUsersAndTheirCC = ccTriplets.map(triplet => (triplet.srcId, triplet.srcAttr))

srcUsersAndTheirCC: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, org.apache.spark.graphx.VertexId)] = MapPartitionsRDD[1265] at map at <console>:79

	scala> //Find the users in the destination vertex with their CC id
	scala> val dstUsersAndTheirCC = ccTriplets.map(triplet => (triplet.dstId, triplet.dstAttr))

dstUsersAndTheirCC: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, org.apache.spark.graphx.VertexId)] = MapPartitionsRDD[1266] at map at <console>:79

	scala> //Find the union
	scala> val usersAndTheirCC = srcUsersAndTheirCC.union(dstUsersAndTheirCC)

usersAndTheirCC: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, org.apache.spark.graphx.VertexId)] = UnionRDD[1267] at union at <console>:83

	scala> //Join with the name of the users
	scala> val usersAndTheirCCWithName = usersAndTheirCC.join(users).map{case (userId,(ccId,userName)) => (ccId, userName)}.distinct.sortByKey()

usersAndTheirCCWithName: org.apache.spark.rdd.RDD[(org.apache.spark.graphx.VertexId, String)] = ShuffledRDD[1277] at sortByKey at <console>:85

	scala> //Print the user names with their CC component id. If two users share the same CC id, then they are connected
	scala> usersAndTheirCCWithName.collect().foreach(println)

      (1,Thomas)    

      (1,Mathew)    

      (1,Krish)    

      (4,Martin)    

      (4,James)    

      (4,George) 

```

Spark GraphX 库中还有一些其他的图处理算法，对完整算法集的详细讨论足以写成一本书。关键在于，Spark GraphX 库提供了非常易于使用的图算法，这些算法很好地融入了 Spark 的统一编程模型。

# 理解 GraphFrames

Spark GraphX 库是支持编程语言最少的图处理库。Scala 是 Spark GraphX 库唯一支持的编程语言。GraphFrames 是由 Databricks、加州大学伯克利分校和麻省理工学院开发的新图处理库，作为外部 Spark 包提供，建立在 Spark DataFrames 之上。由于它是基于 DataFrames 构建的，因此所有可以在 DataFrames 上执行的操作都可能适用于 GraphFrames，支持 Scala、Java、Python 和 R 等编程语言，并具有统一的 API。由于 GraphFrames 基于 DataFrames，因此数据的持久性、对多种数据源的支持以及在 Spark SQL 中强大的图查询功能是用户免费获得的额外好处。

与 Spark GraphX 库类似，在 GraphFrames 中，数据存储在顶点和边中。顶点和边使用 DataFrames 作为数据结构。本章开头介绍的第一个用例再次用于阐明基于 GraphFrames 的图处理。

### 注意

**注意**：GraphFrames 是外部 Spark 包。它与 Spark 2.0 存在一些不兼容。因此，以下代码片段不适用于 Spark 2.0。它们适用于 Spark 1.6。请访问他们的网站以检查 Spark 2.0 支持情况。

在 Spark 1.6 的 Scala REPL 提示符下，尝试以下语句。由于 GraphFrames 是外部 Spark 包，在启动相应的 REPL 时，需要导入库，并在终端提示符下使用以下命令启动 REPL，确保库加载无误：

```scala
	 $ cd $SPARK_1.6__HOME 
	$ ./bin/spark-shell --packages graphframes:graphframes:0.1.0-spark1.6 
	Ivy Default Cache set to: /Users/RajT/.ivy2/cache 
	The jars for the packages stored in: /Users/RajT/.ivy2/jars 
	:: loading settings :: url = jar:file:/Users/RajT/source-code/spark-source/spark-1.6.1
	/assembly/target/scala-2.10/spark-assembly-1.6.2-SNAPSHOT-hadoop2.2.0.jar!
	/org/apache/ivy/core/settings/ivysettings.xml 
	graphframes#graphframes added as a dependency 
	:: resolving dependencies :: org.apache.spark#spark-submit-parent;1.0 
	confs: [default] 
	found graphframes#graphframes;0.1.0-spark1.6 in list 
	:: resolution report :: resolve 153ms :: artifacts dl 2ms 
	:: modules in use: 
	graphframes#graphframes;0.1.0-spark1.6 from list in [default] 
   --------------------------------------------------------------------- 
   |                  |            modules            ||   artifacts   | 
   |       conf       | number| search|dwnlded|evicted|| number|dwnlded| 
   --------------------------------------------------------------------- 
   |      default     |   1   |   0   |   0   |   0   ||   1   |   0   | 
   --------------------------------------------------------------------- 
   :: retrieving :: org.apache.spark#spark-submit-parent 
   confs: [default] 
   0 artifacts copied, 1 already retrieved (0kB/5ms) 
   16/07/31 09:22:11 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable 
   Welcome to 
      ____              __ 
     / __/__  ___ _____/ /__ 
    _\ \/ _ \/ _ `/ __/  '_/ 
   /___/ .__/\_,_/_/ /_/\_\   version 1.6.1 
       /_/ 

	  Using Scala version 2.10.5 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_66) 
	  Type in expressions to have them evaluated. 
	  Type :help for more information. 
	  Spark context available as sc. 
	  SQL context available as sqlContext. 
	  scala> import org.graphframes._ 
	  import org.graphframes._ 
	  scala> import org.apache.spark.rdd.RDD 
	  import org.apache.spark.rdd.RDD 
	  scala> import org.apache.spark.sql.Row 
	  import org.apache.spark.sql.Row 
	  scala> import org.apache.spark.graphx._ 
	  import org.apache.spark.graphx._ 
	  scala> //Create a DataFrame of users containing tuple values with a mandatory Long and another String type as the property of the vertex 
	  scala> val users = sqlContext.createDataFrame(List((1L, "Thomas"),(2L, "Krish"),(3L, "Mathew"))).toDF("id", "name") 
	  users: org.apache.spark.sql.DataFrame = [id: bigint, name: string] 
	  scala> //Created a DataFrame for Edge with String type as the property of the edge 
	  scala> val userRelationships = sqlContext.createDataFrame(List((1L, 2L, "Follows"),(1L, 2L, "Son"),(2L, 3L, "Follows"))).toDF("src", "dst", "relationship") 
	  userRelationships: org.apache.spark.sql.DataFrame = [src: bigint, dst: bigint, relationship: string] 
	  scala> val userGraph = GraphFrame(users, userRelationships) 
	  userGraph: org.graphframes.GraphFrame = GraphFrame(v:[id: bigint, name: string], e:[src: bigint, dst: bigint, relationship: string]) 
	  scala> // Vertices in the graph 
	  scala> userGraph.vertices.show() 
	  +---+------+ 
	  | id|  name| 
	  +---+------+ 
	  |  1|Thomas| 
	  |  2| Krish| 
	  |  3|Mathew| 
	  +---+------+ 
	  scala> // Edges in the graph 
	  scala> userGraph.edges.show() 
	  +---+---+------------+ 
	  |src|dst|relationship| 
	  +---+---+------------+ 
	  |  1|  2|     Follows| 
	  |  1|  2|         Son| 
	  |  2|  3|     Follows| 
	  +---+---+------------+ 
	  scala> //Number of edges in the graph 
	  scala> val edgeCount = userGraph.edges.count() 
	  edgeCount: Long = 3 
	  scala> //Number of vertices in the graph 
	  scala> val vertexCount = userGraph.vertices.count() 
	  vertexCount: Long = 3 
	  scala> //Number of edges coming to each of the vertex.  
	  scala> userGraph.inDegrees.show() 
	  +---+--------+ 
	  | id|inDegree| 
	  +---+--------+ 
	  |  2|       2| 
	  |  3|       1| 
	  +---+--------+ 
	  scala> //Number of edges going out of each of the vertex.  
	  scala> userGraph.outDegrees.show() 
	  +---+---------+ 
	  | id|outDegree| 
	  +---+---------+ 
	  |  1|        2| 
	  |  2|        1| 
	  +---+---------+ 
	  scala> //Total number of edges coming in and going out of each vertex.  
	  scala> userGraph.degrees.show() 
	  +---+------+ 
	  | id|degree| 
	  +---+------+ 
	  |  1|     2| 
	  |  2|     3| 
	  |  3|     1| 
	  +---+------+ 
	  scala> //Get the triplets of the graph 
	  scala> userGraph.triplets.show() 
	  +-------------+----------+----------+ 
	  |         edge|       src|       dst| 
	  +-------------+----------+----------+ 
	  |[1,2,Follows]|[1,Thomas]| [2,Krish]| 
	  |    [1,2,Son]|[1,Thomas]| [2,Krish]| 
	  |[2,3,Follows]| [2,Krish]|[3,Mathew]| 
	  +-------------+----------+----------+ 
	  scala> //Using the DataFrame API, apply filter and select only the needed edges 
	  scala> val numFollows = userGraph.edges.filter("relationship = 'Follows'").count() 
	  numFollows: Long = 2 
	  scala> //Create an RDD of users containing tuple values with a mandatory Long and another String type as the property of the vertex 
	  scala> val usersRDD: RDD[(Long, String)] = sc.parallelize(Array((1L, "Thomas"), (2L, "Krish"),(3L, "Mathew"))) 
	  usersRDD: org.apache.spark.rdd.RDD[(Long, String)] = ParallelCollectionRDD[54] at parallelize at <console>:35 
	  scala> //Created an RDD of Edge type with String type as the property of the edge 
	  scala> val userRelationshipsRDD: RDD[Edge[String]] = sc.parallelize(Array(Edge(1L, 2L, "Follows"),    Edge(1L, 2L, "Son"),Edge(2L, 3L, "Follows"))) 
	  userRelationshipsRDD: org.apache.spark.rdd.RDD[org.apache.spark.graphx.Edge[String]] = ParallelCollectionRDD[55] at parallelize at <console>:35 
	  scala> //Create a graph containing the vertex and edge RDDs as created before 
	  scala> val userGraphXFromRDD = Graph(usersRDD, userRelationshipsRDD) 
	  userGraphXFromRDD: org.apache.spark.graphx.Graph[String,String] = 
	  org.apache.spark.graphx.impl.GraphImpl@77a3c614 
	  scala> //Create the GraphFrame based graph from Spark GraphX based graph 
	  scala> val userGraphFrameFromGraphX: GraphFrame = GraphFrame.fromGraphX(userGraphXFromRDD) 
	  userGraphFrameFromGraphX: org.graphframes.GraphFrame = GraphFrame(v:[id: bigint, attr: string], e:[src: bigint, dst: bigint, attr: string]) 
	  scala> userGraphFrameFromGraphX.triplets.show() 
	  +-------------+----------+----------+
	  |         edge|       src|       dst| 
	  +-------------+----------+----------+ 
	  |[1,2,Follows]|[1,Thomas]| [2,Krish]| 
	  |    [1,2,Son]|[1,Thomas]| [2,Krish]| 
	  |[2,3,Follows]| [2,Krish]|[3,Mathew]| 
	  +-------------+----------+----------+ 
	  scala> // Convert the GraphFrame based graph to a Spark GraphX based graph 
	  scala> val userGraphXFromGraphFrame: Graph[Row, Row] = userGraphFrameFromGraphX.toGraphX 
	  userGraphXFromGraphFrame: org.apache.spark.graphx.Graph[org.apache.spark.sql.Row,org.apache.spark.sql.Row] = org.apache.spark.graphx.impl.GraphImpl@238d6aa2 

```

在为 GraphFrame 创建 DataFrames 时，唯一需要注意的是，对于顶点和边有一些强制性列。在顶点的 DataFrame 中，id 列是强制性的。在边的 DataFrame 中，src 和 dst 列是强制性的。除此之外，可以在 GraphFrame 的顶点和边上存储任意数量的任意列。在 Spark GraphX 库中，顶点标识符必须是长整型，但 GraphFrame 没有这样的限制，任何类型都可以作为顶点标识符。读者应该已经熟悉 DataFrames；任何可以在 DataFrame 上执行的操作都可以在 GraphFrame 的顶点和边上执行。

### 提示

所有 Spark GraphX 支持的图处理算法，GraphFrames 也同样支持。

GraphFrames 的 Python 版本功能较少。由于 Python 不是 Spark GraphX 库支持的编程语言，因此在 Python 中不支持 GraphFrame 与 GraphX 之间的转换。鉴于读者熟悉使用 Python 在 Spark 中创建 DataFrames，此处省略了 Python 示例。此外，GraphFrames API 的 Python 版本存在一些待解决的缺陷，并且在撰写本文时，并非所有之前在 Scala 中演示的功能都能在 Python 中正常工作。

# 理解 GraphFrames 查询

Spark GraphX 库是基于 RDD 的图处理库，而 GraphFrames 是作为外部包提供的基于 Spark DataFrame 的图处理库。Spark GraphX 支持多种图处理算法，但 GraphFrames 不仅支持图处理算法，还支持图查询。图处理算法与图查询之间的主要区别在于，图处理算法用于处理图数据结构中隐藏的数据，而图查询用于搜索图数据结构中隐藏的数据中的模式。在 GraphFrame 术语中，图查询也称为模式查找。这在涉及序列模式的遗传学和其他生物科学中具有巨大的应用价值。

从用例角度出发，以社交媒体应用中用户相互关注为例。用户之间存在关系。在前述章节中，这些关系被建模为图。在现实世界的用例中，此类图可能变得非常庞大，如果需要找到在两个方向上存在关系的用户，这可以通过图查询中的模式来表达，并使用简单的编程结构来找到这些关系。以下演示模型展示了用户间关系在 GraphFrame 中的表示，并利用该模型进行了模式搜索。

在 Spark 1.6 的 Scala REPL 提示符下，尝试以下语句：

```scala
 $ cd $SPARK_1.6_HOME 
	  $ ./bin/spark-shell --packages graphframes:graphframes:0.1.0-spark1.6 
	  Ivy Default Cache set to: /Users/RajT/.ivy2/cache 
	  The jars for the packages stored in: /Users/RajT/.ivy2/jars 
	  :: loading settings :: url = jar:file:/Users/RajT/source-code/spark-source/spark-1.6.1/assembly/target/scala-2.10/spark-assembly-1.6.2-SNAPSHOT-hadoop2.2.0.jar!/org/apache/ivy/core/settings/ivysettings.xml 
	  graphframes#graphframes added as a dependency 
	  :: resolving dependencies :: org.apache.spark#spark-submit-parent;1.0 
	  confs: [default] 
	  found graphframes#graphframes;0.1.0-spark1.6 in list 
	  :: resolution report :: resolve 145ms :: artifacts dl 2ms 
	  :: modules in use: 
	  graphframes#graphframes;0.1.0-spark1.6 from list in [default] 
	  --------------------------------------------------------------------- 
	  |                  |            modules            ||   artifacts   | 
	  |       conf       | number| search|dwnlded|evicted|| number|dwnlded| 
	  --------------------------------------------------------------------- 
	  |      default     |   1   |   0   |   0   |   0   ||   1   |   0   | 
	  --------------------------------------------------------------------- 
	  :: retrieving :: org.apache.spark#spark-submit-parent 
	  confs: [default] 
	  0 artifacts copied, 1 already retrieved (0kB/5ms) 
	  16/07/29 07:09:08 WARN NativeCodeLoader: 
	  Unable to load native-hadoop library for your platform... using builtin-java classes where applicable 
	  Welcome to 
      ____              __ 
     / __/__  ___ _____/ /__ 
    _\ \/ _ \/ _ `/ __/  '_/ 
   /___/ .__/\_,_/_/ /_/\_\   version 1.6.1 
      /_/ 

	  Using Scala version 2.10.5 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_66) 
	  Type in expressions to have them evaluated. 
	  Type :help for more information. 
	  Spark context available as sc. 
	  SQL context available as sqlContext. 
	  scala> import org.graphframes._ 
	  import org.graphframes._ 
	  scala> import org.apache.spark.rdd.RDD 
	  import org.apache.spark.rdd.RDD 
	  scala> import org.apache.spark.sql.Row 
	  import org.apache.spark.sql.Row 
	  scala> import org.apache.spark.graphx._ 
	  import org.apache.spark.graphx._ 
	  scala> //Create a DataFrame of users containing tuple values with a mandatory String field as id and another String type as the property of the vertex. Here it can be seen that the vertex identifier is no longer a long integer. 
	  scala> val users = sqlContext.createDataFrame(List(("1", "Thomas"),("2", "Krish"),("3", "Mathew"))).toDF("id", "name") 
	  users: org.apache.spark.sql.DataFrame = [id: string, name: string] 
	  scala> //Create a DataFrame for Edge with String type as the property of the edge 
	  scala> val userRelationships = sqlContext.createDataFrame(List(("1", "2", "Follows"),("2", "1", "Follows"),("2", "3", "Follows"))).toDF("src", "dst", "relationship") 
	  userRelationships: org.apache.spark.sql.DataFrame = [src: string, dst: string, relationship: string] 
	  scala> //Create the GraphFrame 
	  scala> val userGraph = GraphFrame(users, userRelationships) 
	  userGraph: org.graphframes.GraphFrame = GraphFrame(v:[id: string, name: string], e:[src: string, dst: string, relationship: string]) 
	  scala> // Search for pairs of users who are following each other 
	  scala> // In other words the query can be read like this. Find the list of users having a pattern such that user u1 is related to user u2 using the edge e1 and user u2 is related to the user u1 using the edge e2\. When a query is formed like this, the result will list with columns u1, u2, e1 and e2\. When modelling real-world use cases, more meaningful variables can be used suitable for the use case. 
	  scala> val graphQuery = userGraph.find("(u1)-[e1]->(u2); (u2)-[e2]->(u1)") 
	  graphQuery: org.apache.spark.sql.DataFrame = [e1: struct<src:string,dst:string,relationship:string>, u1: struct<
	  d:string,name:string>, u2: struct<id:string,name:string>, e2: struct<src:string,dst:string,relationship:string>] 
	  scala> graphQuery.show() 
	  +-------------+----------+----------+-------------+

	  |           e1|        u1|        u2|           e2| 
	  +-------------+----------+----------+-------------+ 
	  |[1,2,Follows]|[1,Thomas]| [2,Krish]|[2,1,Follows]| 
	  |[2,1,Follows]| [2,Krish]|[1,Thomas]|[1,2,Follows]| 
	  +-------------+----------+----------+-------------+

```

请注意，图查询结果中的列是由搜索模式中给出的元素构成的。形成模式的方式没有限制。

### 注意

注意图查询结果的数据类型。它是一个 DataFrame 对象。这为使用熟悉的 Spark SQL 库处理查询结果带来了极大的灵活性。

Spark GraphX 库的最大限制是其 API 目前不支持 Python 和 R 等编程语言。由于 GraphFrames 是基于 DataFrame 的库，一旦成熟，它将使所有支持 DataFrame 的编程语言都能进行图处理。这个 Spark 外部包无疑是未来可能被纳入 Spark 的一部分的有力候选。

# 参考文献

如需了解更多信息，请访问以下链接：

+   [`spark.apache.org/docs/1.5.2/graphx-programming-guide.html`](https://spark.apache.org/docs/1.5.2/graphx-programming-guide.html)

+   [`en.wikipedia.org/wiki/2015_ATP_World_Tour_Finals_%E2%80%93_Singles`](https://en.wikipedia.org/wiki/2015_ATP_World_Tour_Finals_%E2%80%93_Singles)

+   [`www.protennislive.com/posting/2015/605/mds.pdf`](http://www.protennislive.com/posting/2015/605/mds.pdf)

+   [`infolab.stanford.edu/~backrub/google.html`](http://infolab.stanford.edu/~backrub/google.html)

+   [`graphframes.github.io/index.html`](http://graphframes.github.io/index.html)

+   [`github.com/graphframes/graphframes`](https://github.com/graphframes/graphframes)

+   [`spark-packages.org/package/graphframes/graphframes`](https://spark-packages.org/package/graphframes/graphframes)

# 总结

图是一种非常有用的数据结构，具有广泛的应用潜力。尽管在大多数应用中不常使用，但在某些独特的应用场景中，使用图作为数据结构是必不可少的。只有当数据结构与经过充分测试和高度优化的算法结合使用时，才能有效地使用它。数学家和计算机科学家已经提出了许多处理图数据结构中数据的算法。Spark GraphX 库在 Spark 核心之上实现了大量此类算法。本章通过入门级别的用例对 Spark GraphX 库进行了快速概览，并介绍了一些基础知识。

基于 DataFrame 的图抽象名为 GraphFrames，它是 Spark 的一个外部包，可单独获取，在图处理和图查询方面具有巨大潜力。为了进行图查询以发现图中的模式，已提供了对该外部 Spark 包的简要介绍。

任何教授新技术的书籍都应以一个涵盖其显著特点的应用案例作为结尾。Spark 也不例外。到目前为止，本书已经介绍了 Spark 作为下一代数据处理平台的特性。现在是时候收尾并构建一个端到端应用了。下一章将涵盖使用 Spark 及其上层构建的库家族设计和开发数据处理应用的内容。
