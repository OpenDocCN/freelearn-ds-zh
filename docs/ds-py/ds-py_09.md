# 第九章：分析学习：图算法 - 美国国内航班数据分析

> “在没有数据之前进行理论推测是一个严重的错误。”  

*福尔摩斯*  

本章我们将重点介绍一种基本的计算机科学数据模型——图，以及常用于图的各种算法。作为数据科学家或开发者，熟悉图并能迅速识别何时使用图来解决特定数据问题是非常重要的。例如，图非常适合于基于 GPS 的应用，如 Google Maps，用于寻找从 A 点到 B 点的最佳路线，考虑到各种参数，包括用户是开车、步行还是乘坐公共交通工具，或者用户是否希望选择最短路径，还是最大化使用高速公路而不考虑总体距离。这些参数中的一些也可以是实时参数，如交通状况和天气。另一个使用图的应用重要类别是社交网络，如 Facebook 或 Twitter，其中顶点代表个人，边表示关系，例如*是朋友*和*关注*。  

本章将从图论和相关图算法的高层次介绍开始。然后我们将介绍`networkx`，这是一个 Python 库，可以轻松加载、操作和可视化图数据结构，并提供丰富的图算法集合。接下来，我们将通过构建样本分析，使用各种图算法分析美国航班数据，其中机场作为顶点，航班作为边。像往常一样，我们还将通过构建一个简单的仪表板 PixieApp 来操作这些分析。最后，我们将通过建立一个预测模型，应用在第八章中学到的时间序列技术，来分析历史航班数据。  

# 图论简介  

图论的引入及相关理论广泛归功于 1736 年莱昂哈德·欧拉（Leonhard Euler），当时他研究了*哥尼斯堡七桥问题*（[`en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg`](https://en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg)）。  

这座城市被普雷格尔河分开，河流形成了两座岛屿，并根据以下图示建造了七座桥。问题是找到一种方法，使一个人能够走过每座桥一次且仅一次，并回到起点。欧拉证明了这个问题没有解，并在此过程中创立了图论。其基本思路是将城市的布局转化为图，其中每块陆地是一个顶点，每座桥是连接两个顶点（即陆地）的边。问题被简化为找到一条路径，这条路径是一个包含每座桥仅一次的连续边和顶点的序列。

以下图示展示了欧拉是如何将 *哥尼斯堡七桥问题* 简化为图论问题的：

![图形介绍](img/B09699_09_01.jpg)

将哥尼斯堡七桥问题简化为图论问题

使用更正式的定义，**图**是表示对象之间成对关系（称为**边**）的数据结构（对象称为**顶点**或**节点**）。通常使用以下符号来表示图：*G = (V, E)*，其中 *V* 是顶点集，*E* 是边集。

图的表示主要有两大类：

+   **有向图（称为有向图）**：在成对关系中，顺序是重要的，即从顶点 **A** 到顶点 **B** 的边（A-B）与从顶点 **B** 到顶点 **A** 的边（B-A）是不同的。

+   **无向图**：在成对关系中，顺序无关，即边（A-B）与边（B-A）是相同的。

以下图示展示了一个样本图的表示方式，既有无向图（边没有箭头），也有有向图（边有箭头）：

![图形介绍](img/B09699_09_02.jpg)

## 图形表示

表示图的方式主要有两种：

+   **邻接矩阵**：使用 *n* 乘 *n* 矩阵（我们称之为 *A*）来表示图，其中 *n* 是图中顶点的数量。顶点通过 1 到 *n* 的整数进行索引。我们使用 ![图形表示](img/B09699_09_37.jpg) 来表示顶点 *i* 和顶点 *j* 之间存在边，使用 ![图形表示](img/B09699_09_38.jpg) 来表示顶点 *i* 和顶点 *j* 之间不存在边。在无向图的情况下，由于顺序无关，通常会有 ![图形表示](img/B09699_09_39.jpg)![图形表示](img/B09699_09_40.jpg)。然而，在有向图中，由于顺序重要，*A**[i,j]* 可能与 *A**[j,i]* 不同。以下示例展示了如何在邻接矩阵中表示一个样本图，既适用于有向图，也适用于无向图：![图形表示](img/B09699_09_03.jpg)

    图的邻接矩阵表示（包括有向和无向图）

    需要注意的是，邻接矩阵表示法具有恒定的空间复杂度，其为 ![图表示](img/B09699_09_41.jpg)，其中 *n* 是顶点的数量，但它具有 *O(1)* 的时间复杂度，即常数时间可以判断两个顶点是否通过边连接。当图是密集的（即边很多）时，高空间复杂度可能是可以接受的，但当图是稀疏的时，这种方式可能会浪费空间，在这种情况下，我们可能会更倾向于使用下面的邻接表表示法。

    ### 注意

    **注意**：大 O 符号（[`en.wikipedia.org/wiki/Big_O_notation`](https://en.wikipedia.org/wiki/Big_O_notation)）常用于代码分析中，通过评估算法在输入大小增长时的表现来表示算法的性能。它用于评估运行时间（运行算法所需的指令数量）和空间需求（随着时间推移所需的存储量）。

+   **邻接表**：对于每个顶点，我们维护一个与之相连的所有顶点的列表。在无向图中，每条边被表示两次，分别对应两个端点，而有向图则不同，边的顺序是有意义的。

    下图展示了图的邻接表表示方法，包括有向图和无向图：

    ![图表示](img/B09699_09_04.jpg)

    图的邻接表表示（包括有向图和无向图）

    与邻接矩阵表示法相反，邻接表表示法具有较小的空间复杂度，时间复杂度为 *O (m + n)*，其中 *m* 是边的数量，*n* 是顶点的数量。然而，时间复杂度相较于邻接矩阵的 *O(1)* 增加到 *O(m)*。因此，当图是稀疏连接时（即边较少），使用邻接表表示法更为合适。

正如前面讨论中所提示的，使用哪种图表示方法很大程度上依赖于图的密度，同时也取决于我们计划使用的算法类型。在接下来的章节中，我们将讨论最常用的图算法。

## 图算法

以下是最常用的图算法列表：

+   **搜索**：在图的上下文中，搜索意味着查找两个顶点之间的路径。路径被定义为一系列连续的边和顶点。图中搜索路径的动机可能有多种；它可能是你希望根据某些预定义的距离标准找到最短路径，例如最少的边数（例如，GPS 路线规划），或者你仅仅想知道两个顶点之间是否存在路径（例如，确保网络中每台机器都可以与其他机器互联）。一种通用的路径搜索算法是从给定的顶点开始，*发现*与其连接的所有顶点，标记已发现的顶点为已探索（以防止重复发现），并对每个已发现的顶点继续相同的探索，直到找到目标顶点，或者所有顶点都已被遍历。这个搜索算法有两种常用的变种：广度优先搜索（BFS）和深度优先搜索（DFS），它们各自有适用的场景，适合不同的使用情况。这两种算法的区别在于它们寻找未探索顶点的方式：

    +   **广度优先搜索（BFS）**：首先探索直接邻居的未探索节点。当一个层次的邻居已被探索完毕，开始探索每个节点的邻域，直到遍历完整个图。由于我们首先探索所有直接连接的顶点，这个算法保证能够找到最短路径，该路径与发现的邻居数量相对应。BFS 的一个扩展是著名的 Dijkstra 最短路径算法，其中每条边都与一个非负权重关联。在这种情况下，最短路径可能不是跳数最少的路径，而是最小化所有权重之和的路径。Dijkstra 最短路径的一个应用示例是查找地图上两点之间的最短路径。

    +   **深度优先搜索（DFS）**：对于每个直接邻居顶点，首先深入探索它的邻居，尽可能深入，直到没有更多邻居为止，然后开始回溯。DFS 的应用示例包括查找有向图的拓扑排序和强连通分量。作为参考，拓扑排序是顶点的线性排序，使得每个顶点在排序中按照边的方向依次排列（即，不会逆向）。更多信息请参见 [`en.wikipedia.org/wiki/Topological_sorting`](https://en.wikipedia.org/wiki/Topological_sorting)。

    下图展示了 BFS 和 DFS 在寻找未探索节点时的差异：

    ![图形算法](img/B09699_09_05.jpg)

    BFS 和 DFS 中未探索顶点的发现顺序

+   **连通分量和强连通分量**：图的连通分量是指一组顶点，其中任意两个顶点之间都有路径。请注意，定义只要求存在路径，这意味着两个顶点之间不必有边，只要有路径就行。对于有向图，连通分量被称为**强连通分量**，这是因为有额外的方向约束要求，不仅任何顶点 A 应该有路径通向任意其他顶点 B，而且 B 也必须有路径通向 A。

    以下图示展示了强连通分量或一个示例有向图：

    ![图算法](img/B09699_09_06.jpg)

    有向图的强连通分量

+   **中心性**：顶点的中心性指标提供了一个关于该顶点相对于图中其他顶点的重要性的衡量。中心性指数有多个重要应用。例如，识别社交网络中最有影响力的人，或根据页面的重要性对网页搜索结果进行排名，等等。

    中心性有多个指标，但我们将专注于本章稍后使用的以下四个：

    +   **度数**：顶点的度数是该顶点作为端点的边的数量。在有向图的情况下，它是该顶点作为源或目标的边的数量，我们称**入度**为该顶点作为目标的边的数量，**出度**为该顶点作为源的边的数量。

    +   **PageRank**：这是由谷歌创始人拉里·佩奇和谢尔盖·布林开发的著名算法。PageRank 用于通过提供一个衡量网站重要性的指标来排名搜索结果，其中包括计算指向该网站的其他网站的链接数量。它还考虑了这些链接的质量估计（即，指向你网站的链接站点的可信度）。

    +   **接近度**：接近度中心性与给定顶点与图中所有其他顶点之间最短路径的平均长度成反比。直观上，顶点距离其他所有节点越近，它就越重要。

        接近度中心性可以通过以下简单公式计算：

        ![图算法](img/B09699_09_42.jpg)

        (来源：https://en.wikipedia.org/wiki/Centrality#Closeness_centrality)

        其中 *d(y,x)* 是节点 *x* 和 *y* 之间的边的长度。

    +   **最短路径介入度**：这个度量是基于给定顶点在任意两节点之间最短路径上出现的次数。直观上，顶点对最短路径的贡献越大，它就越重要。最短路径介入度的数学公式如下所示：![图算法](img/B09699_09_46.jpg)

        (来源：https://en.wikipedia.org/wiki/Centrality#Betweenness_centrality)

        其中 ![图算法](img/B09699_09_43.jpg) 是从顶点 *s* 到顶点 *t* 的所有最短路径的总数，! 图算法 是经过 *v* 的 ![图算法](img/B09699_09_45.jpg) 的子集。

        ### 注意

        **注意**：更多关于中心性的详细信息可以在这里找到：

        [`en.wikipedia.org/wiki/Centrality`](https://en.wikipedia.org/wiki/Centrality)

## 图与大数据

目前我们的图讨论主要集中在可以适应单台机器的数据上，但当我们有包含数十亿个顶点和边的大图时，如何处理呢？如果将整个数据加载到内存中是不可能的，如何解决呢？一个自然的解决方案是将数据分布到多个节点的集群中，在这些节点上并行处理数据，并将各自的结果合并形成最终答案。幸运的是，已经有多个框架提供了这种图并行计算能力，并且它们几乎都实现了大多数常用图算法。流行的开源框架包括 Apache Spark GraphX（[`spark.apache.org/graphx`](https://spark.apache.org/graphx)）和 Apache Giraph（[`giraph.apache.org`](http://giraph.apache.org)），后者目前被 Facebook 用于分析其社交网络。

不深入细节，重要的是要知道，这些框架都受到**大规模同步并行**（**BSP**）分布式计算模型的启发（[`en.wikipedia.org/wiki/Bulk_synchronous_parallel`](https://en.wikipedia.org/wiki/Bulk_synchronous_parallel)），该模型通过机器间的消息传递来查找集群中的顶点。需要记住的关键点是，这些框架通常非常易于使用，例如，使用 Apache Spark GraphX 编写本章的分析将相对简单。

在本节中，我们只回顾了所有图算法中的一小部分，深入探讨将超出本书的范围。自己实现这些算法将花费大量时间，但幸运的是，现有许多开源库提供了相当完整的图算法实现，并且易于使用和集成到应用程序中。在本章的其余部分，我们将使用`networkx`开源 Python 库。

# 开始使用 networkx 图库

在我们开始之前，如果尚未完成，需要使用`pip`工具安装`networkx`库。请在单独的单元格中执行以下代码：

```py
!pip install networkx

```

### 注意

**注意**：和往常一样，安装完成后不要忘记重启内核。

`networkx` 提供的大多数算法可以直接从主模块调用。因此，用户只需要以下的`import`语句：

```py
import networkx as nx
```

## 创建图

作为起点，让我们回顾一下`networkx`支持的不同类型的图及其创建空图的构造函数：

+   `Graph`：一个无向图，顶点之间只允许有一条边。允许自环边。构造函数示例：

    ```py
    G = nx.Graph()
    ```

+   `Digraph`：`Graph` 的子类，表示有向图。构造函数示例：

    ```py
    G = nx.DiGraph()
    ```

+   `MultiGraph`：允许顶点之间有多条边的无向图。构造函数示例：

    ```py
    G = nx.MultiGraph()
    ```

+   `MultiDiGraph`：允许顶点之间有多条边的有向图。构造函数示例：

    ```py
    G = nx.MultiDiGraph()
    ```

`Graph` 类提供了许多方法用于添加和删除顶点及边。以下是可用方法的子集：

+   `add_edge(u_of_edge, v_of_edge, **attr)`：在顶点 `u` 和顶点 `v` 之间添加一条边，并可选地添加附加属性，这些属性将与边相关联。如果顶点 `u` 和 `v` 在图中不存在，它们会自动创建。

+   `remove_edge(u, v)`：移除顶点 `u` 和 `v` 之间的边。

+   `add_node(self, node_for_adding, **attr)`：向图中添加一个节点，并可选地添加额外属性。

+   `remove_node(n)`：移除由给定参数 `n` 标识的节点。

+   `add_edges_from(ebunch_to_add, **attr)`：批量添加多条边，并可选地添加额外属性。边必须以二元组 `(u,v)` 或三元组 `(u,v,d)` 的形式给出，其中 `d` 是包含边数据的字典。

+   `add_nodes_from(self, nodes_for_adding, **attr)`：批量添加多个节点，并可选地添加额外属性。节点可以作为列表、字典、集合、数组等提供。

作为练习，让我们从一开始就构建我们使用的有向图示例：

![创建图](img/B09699_09_07.jpg)

要通过编程方式使用 networkx 创建的示例图

以下代码通过创建一个 `DiGraph()` 对象开始，使用 `add_nodes_from()` 方法一次性添加所有节点，然后使用 `add_edge()` 和 `add_edges_from()` 的组合（作为示例）开始添加边：

```py
G = nx.DiGraph()
G.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
G.add_edge('A', 'B')
G.add_edge('B', 'B')
G.add_edges_from([('A', 'E'),('A', 'D'),('B', 'C'),('C', 'E'),('D', 'C')])
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode1.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode1.py)

`Graph` 类还通过变量类视图提供了方便访问其属性的方法。例如，你可以使用 `G.nodes` 和 `G.edges` 遍历图的顶点和边，但也可以通过以下表示法访问单个边：`G.edges[u,v]`。

以下代码遍历图的节点并打印它们：

```py
for n in G.nodes:
    print(n)
```

`networkx` 库还提供了一套丰富的预构建图生成器，这些生成器对于测试你的算法非常有用。例如，你可以通过 `complete_graph()` 生成器轻松生成一个完全图，如以下代码所示：

```py
G_complete = nx.complete_graph(10)
```

### 注意

你可以在这里找到所有可用图生成器的完整列表：

[`networkx.github.io/documentation/networkx-2.1/reference/generators.html#generators`](https://networkx.github.io/documentation/networkx-2.1/reference/generators.html#generators)

## 可视化一个图

NetworkX 支持多种渲染引擎，包括 Matplotlib、Graphviz AGraph ([`pygraphviz.github.io`](http://pygraphviz.github.io)) 和 Graphviz with pydot ([`github.com/erocarrera/pydot`](https://github.com/erocarrera/pydot))。尽管 Graphviz 提供了非常强大的绘图功能，但我发现它的安装非常困难。然而，Matplotlib 已经在 Jupyter Notebooks 中预装，可以让你快速开始。

核心绘图功能是 `draw_networkx`，它接受一个图作为参数，并有一组可选的关键字参数，允许你对图进行样式设置，例如颜色、宽度，以及节点和边的标签字体。图形绘制的整体布局通过 `pos` 关键字参数传递 `GraphLayout` 对象来配置。默认布局是 `spring_layout`（使用基于力的算法），但 NetworkX 支持许多其他布局，包括 `circular_layout`、`random_layout` 和 `spectral_layout`。你可以在这里查看所有可用布局的列表：[`networkx.github.io/documentation/networkx-2.1/reference/drawing.html#module-networkx.drawing.layout`](https://networkx.github.io/documentation/networkx-2.1/reference/drawing.html#module-networkx.drawing.layout)。

为了方便，`networkx` 将每个布局封装成其高层绘图方法，调用合理的默认值，这样调用者就不需要处理每个布局的复杂性。例如，`draw()` 方法将使用 `spring_layout` 绘制图形，`draw_circular()` 使用 `circular_layout`，`draw_random()` 使用 `random_layout`。

在下面的示例代码中，我们使用 `draw()` 方法来可视化我们之前创建的 `G_complete` 图：

```py
%matplotlib inline
import matplotlib.pyplot as plt
nx.draw(G_complete, with_labels=True)
plt.show()
```

### 注

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode2.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode2.py)

结果显示在以下输出中：

![可视化图形](img/B09699_09_08.jpg)

绘制一个包含 10 个节点的完整图

使用 `networkx` 绘制图形既简单又有趣，并且由于它使用了 Matplotlib，你可以进一步美化它们，利用 Matplotlib 的绘图功能。我鼓励读者进一步实验，通过在笔记本中可视化不同的图形。在下一部分中，我们将开始实现一个示例应用程序，使用图算法分析航班数据。

# 第一部分 – 将美国国内航班数据加载到图中

为了初始化笔记本，接下来我们运行以下代码，在它自己的单元格中导入本章中将频繁使用的包：

```py
import pixiedust
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
```

我们还将使用 *2015 年航班延误和取消* 数据集，该数据集可以在 Kaggle 网站上找到，位置为：[`www.kaggle.com/usdot/datasets`](https://www.kaggle.com/usdot/datasets)。该数据集由三个文件组成：

+   `airports.csv`：所有美国机场的列表，包括它们的**IATA**代码（**国际航空运输协会**：[`openflights.org/data.html`](https://openflights.org/data.html)）、城市、州、经度和纬度。

+   `airlines.csv`：美国航空公司的列表，包括它们的 IATA 代码。

+   `flights.csv`：2015 年发生的航班列表。该数据包括日期、起点和终点机场、计划和实际时间以及延误情况。

`flights.csv`文件包含接近 600 万条记录，需要清理，移除所有起点或终点机场没有 IATA 三字代码的航班。我们还需要删除`ELAPSED_TIME`列中缺失值的行。如果不这样做，在将数据加载到图形结构时会出现问题。另一个问题是数据集包含一些时间列，比如`DEPARTURE_TIME`和`ARRIVAL_TIME`，为了节省空间，这些列只存储`HHMM`格式的时间，实际日期存储在`YEAR`、`MONTH`和`DAY`列中。我们将在本章进行的一个分析中需要完整的`DEPARTURE_TIME`日期时间，因为进行这个转换是一个耗时的操作，所以我们现在就进行转换，并将处理后的版本存储在将上传到 GitHub 的`flights.csv`文件中。此操作使用 pandas 的`apply()`方法，调用`to_datetime()`函数并设置`axis=1`（表示该转换应用于每一行）。

另一个问题是我们想将文件存储在 GitHub 上，但 GitHub 对文件大小有 100MB 的最大限制。因此，为了将文件大小缩小到 100MB 以下，我们还删除了一些在我们构建分析中不需要的列，然后将文件压缩后再存储到 GitHub 上。当然，另一个好处是，较小的文件会使 DataFrame 加载得更快。

在从 Kaggle 网站下载文件后，我们运行以下代码，该代码首先将 CSV 文件加载到 pandas DataFrame 中，移除不需要的行和列，然后将数据写回到文件中：

### 注意

**注意**：原始数据存储在名为`flights.raw.csv`的文件中。

运行以下代码可能需要一些时间，因为文件非常大，包含 600 万条记录。

```py
import pandas as pd
import datetime
import numpy as np

# clean up the flights data in flights.csv
flights = pd.read_csv('flights.raw.csv', low_memory=False)

# select only the rows that have a 3 letter IATA code in the ORIGIN and DESTINATION airports
mask = (flights["ORIGIN_AIRPORT"].str.len() == 3) & (flights["DESTINATION_AIRPORT"].str.len() == 3)
flights = flights[ mask ]

# remove the unwanted columns
dropped_columns=["SCHEDULED_DEPARTURE","SCHEDULED_TIME",
"CANCELLATION_REASON","DIVERTED","DIVERTED","TAIL_NUMBER",
"TAXI_OUT","WHEELS_OFF","WHEELS_ON",
"TAXI_IN","SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "AIR_SYSTEM_DELAY","SECURITY_DELAY",
"AIRLINE_DELAY","LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
flights.drop(dropped_columns, axis=1, inplace=True)

# remove the row that have NA in the ELAPSED_TIME column
flights.dropna(subset=["ELAPSED_TIME"], inplace=True)

# remove the row that have NA in the DEPARTURE_TIME column
flights.dropna(subset=["ELAPSED_TIME"], inplace=True)

# Create a new DEPARTURE_TIME columns that has the actual datetime
def to_datetime(row):
    departure_time = str(int(row["DEPARTURE_TIME"])).zfill(4)
    hour = int(departure_time[0:2])
    return datetime.datetime(year=row["YEAR"], month=row["MONTH"],
                             day=row["DAY"],
                             hour = 0 if hour >= 24 else hour,
                             minute=int(departure_time[2:4])
                            )
flights["DEPARTURE_TIME"] = flights.apply(to_datetime, axis=1)

# write the data back to file without the index
flights.to_csv('flights.csv', index=False)

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode3.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode3.py)

### 注意

**注意**：根据`pandas.read_csv`文档（[`pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.read_csv.html`](http://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.read_csv.html)），我们使用关键字参数`low_memory=False`来确保数据不是按块加载，这样可以避免在类型推断时出现问题，特别是在处理非常大的文件时。

为了方便起见，三个文件存储在以下 GitHub 位置：[`github.com/DTAIEB/Thoughtful-Data-Science/tree/master/chapter%209/USFlightsAnalysis`](https://github.com/DTAIEB/Thoughtful-Data-Science/tree/master/chapter%209/USFlightsAnalysis)。

以下代码使用 `pixiedust.sampleData()` 方法将数据加载到三个 pandas DataFrame 中，分别对应 `airlines`、`airports` 和 `flights`：

```py
airports = pixiedust.sampleData("https://github.com/DTAIEB/Thoughtful-Data-Science/raw/master/chapter%209/USFlightsAnalysis/airports.csv")
airlines = pixiedust.sampleData("https://github.com/DTAIEB/Thoughtful-Data-Science/raw/master/chapter%209/USFlightsAnalysis/airlines.csv")
flights = pixiedust.sampleData("https://github.com/DTAIEB/Thoughtful-Data-Science/raw/master/chapter%209/USFlightsAnalysis/flights.zip")
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode4.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode4.py)

**注意**：GitHub URL 使用了 `/raw/` 部分，表示我们希望下载原始文件，而不是相应 GitHub 页面上的 HTML。

下一步是使用 `flights` DataFrame 作为 `edge` 列表，并将 `ELAPSED_TIME` 列的值作为权重，加载数据到 `networkx` 有向加权图对象中。我们首先通过使用 `pandas.groupby()` 方法按多重索引对所有起始地和目的地相同的航班进行去重，其中 `ORIGIN_AIRPORT` 和 `DESTINATION_AIRPORT` 为键。接着，我们从 `DataFrameGroupBy` 对象中选择 `ELAPSED_TIME` 列，并使用 `mean()` 方法对结果进行聚合。这将为我们提供一个新的 DataFrame，其中包含每个具有相同起始地和目的地机场的航班的平均 `ELAPSED_TIME`：

```py
edges = flights.groupby(["ORIGIN_AIRPORT","DESTINATION_AIRPORT"]) [["ELAPSED_TIME"]].mean()
edges
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode5.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode5.py)

结果如下截图所示：

![第一部分 – 将美国国内航班数据加载到图中](img/B09699_09_09.jpg)

按起始地和目的地分组的航班，以及平均的 ELAPSED_TIME

在使用这个 DataFrame 创建有向图之前，我们需要将索引从多重索引重置为常规的单一索引，并将索引列转换为常规列。为此，我们只需使用 `reset_index()` 方法，如下所示：

```py
edges = edges.reset_index()
edges
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode6.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode6.py)

我们现在有了一个形状正确的 DataFrame，准备用于创建有向图，如下截图所示：

![第一部分 – 将美国国内航班数据加载到图中](img/B09699_09_10.jpg)

按起始地和目的地分组的航班，计算平均 ELAPSED_TIME，并使用单一索引

为了创建有向加权图，我们使用 NetworkX 的 `from_pandas_edgelist()` 方法，该方法以 pandas DataFrame 作为输入源。我们还指定了源列和目标列，以及权重列（在我们的例子中是 `ELAPSED_TIME`）。最后，我们通过使用 `create_using` 关键字参数，传入 `DiGraph` 的实例，告诉 NetworkX 我们想要创建一个有向图。

以下代码展示了如何调用 `from_pandas_edgelist()` 方法：

```py
flight_graph = nx.from_pandas_edgelist(
    flights, "ORIGIN_AIRPORT","DESTINATION_AIRPORT",
    "ELAPSED_TIME",
    create_using = nx.DiGraph() )
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode7.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode7.py) **注意**：NetworkX 支持通过转换多种格式（包括字典、列表、NumPy 和 SciPy 矩阵以及 pandas）来创建图形。你可以在这里找到关于这些转换功能的更多信息：

[`networkx.github.io/documentation/networkx-2.1/reference/convert.html`](https://networkx.github.io/documentation/networkx-2.1/reference/convert.html)

我们可以通过直接打印图的节点和边来快速验证图的值是否正确：

```py
print("Nodes: {}".format(flight_graph.nodes))
print("Edges: {}".format(flight_graph.edges))
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode8.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode8.py)

这会生成以下输出（已截断）：

```py
Nodes: ['BOS', 'TYS', 'RKS', 'AMA', 'BUF', 'BHM', 'PPG', …, 'CWA', 'DAL', 'BFL']
Edges: [('BOS', 'LAX'), ('BOS', 'SJC'), ..., ('BFL', 'SFO'), ('BFL', 'IAH')]
```

我们还可以使用 `networkx` 中内置的绘图 API 来创建更好的可视化效果，这些 API 支持多个渲染引擎，包括 Matplotlib、Graphviz AGraph ([`pygraphviz.github.io`](http://pygraphviz.github.io)) 和带有 pydot 的 Graphviz ([`github.com/erocarrera/pydot`](https://github.com/erocarrera/pydot))。

为了简化，我们将使用 NetworkX 的 `draw()` 方法，该方法使用现成的 Matplotlib 引擎。为了美化可视化效果，我们配置了合适的宽度和高度 `(12, 12)`，并添加了一个色彩图，色彩鲜艳（我们使用了 `matplotlib.cm` 中的 `cool` 和 `spring` 色彩图，参见：[`matplotlib.org/2.0.2/examples/color/colormaps_reference.html`](https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html)）。

以下代码展示了图形可视化的实现：

```py
import matplotlib.cm as cm
fig = plt.figure(figsize = (12,12))
nx.draw(flight_graph, arrows=True, with_labels=True,
        width = 0.5,style="dotted",
        node_color=range(len(flight_graph)),
        cmap=cm.get_cmap(name="cool"),
        edge_color=range(len(flight_graph.edges)),
        edge_cmap=cm.get_cmap(name="spring")
       )
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode9.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode9.py)

这会生成以下结果：

![第一部分 – 将美国国内航班数据加载到图中](img/B09699_09_11.jpg)

使用 Matplotlib 快速可视化我们的有向图

在前面的图表中，节点是通过一种名为`spring_layout`的默认图布局进行定位的，这是一种力导向布局。这种布局的一个优点是它能够快速显示出图中连接最多边的节点，这些节点位于图的中心。我们可以通过在调用`draw()`方法时使用`pos`关键字参数来更改图的布局。`networkx`还支持其他类型的布局，包括`circular_layout`、`random_layout`、`shell_layout`和`spectral_layout`。

例如，使用`random_layout`：

```py
import matplotlib.cm as cm
fig = plt.figure(figsize = (12,12))
nx.draw(flight_graph, arrows=True, with_labels=True,
        width = 0.5,style="dotted",
        node_color=range(len(flight_graph)),
        cmap=cm.get_cmap(name="cool"),
        edge_color=range(len(flight_graph.edges)),
        edge_cmap=cm.get_cmap(name="spring"),
        pos = nx.random_layout(flight_graph)
       )
plt.show()

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode10.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode10.py)

我们得到以下结果：

![第一部分 - 将美国国内航班数据加载到图中](img/B09699_09_12.jpg)

使用`random_layout`的航班数据图

### 注意

注意：你可以在这里找到更多关于这些布局的信息：

[`networkx.github.io/documentation/networkx-2.1/reference/drawing.html`](https://networkx.github.io/documentation/networkx-2.1/reference/drawing.html)

## 图的中心性

接下来，图的一个有趣分析点是其中心性指标，这可以帮助我们发现哪些节点是最重要的顶点。作为练习，我们将计算四种类型的中心性指标：**度数**、**PageRank**、**接近度**和**最短路径中介中心性**。然后，我们将扩展机场数据框，添加每个中心性指标的列，并使用 PixieDust 的`display()`在 Mapbox 地图中可视化结果。

计算有向图的度数非常简单，只需使用`networkx`的`degree`属性，像这样：

```py
print(flight_graph.degree)
```

这将输出一个元组数组，每个元组包含机场代码和度数索引，如下所示：

```py
[('BMI', 14), ('RDM', 8), ('SBN', 13), ('PNS', 18), ………, ('JAC', 26), ('MEM', 46)]
```

现在我们想要在机场数据框中添加一个`DEGREE`列，其中包含前述数组中每个机场行的度数值。为了做到这一点，我们需要创建一个包含两个列的新数据框：`IATA_CODE`和`DEGREE`，并在`IATA_CODE`上执行 pandas 的`merge()`操作。

合并操作在下图中展示：

![图的中心性](img/B09699_09_13.jpg)

合并度数数据框到机场数据框

以下代码展示了如何实现上述步骤。我们首先通过遍历`flight_path.degree`输出创建一个 JSON 负载，并使用`pd.DataFrame()`构造函数创建数据框。然后我们使用`pd.merge()`，将`airports`和`degree_df`作为参数传入。我们还使用`on`参数，其值为`IATA_CODE`，这是我们要进行连接的键列：

```py
degree_df = pd.DataFrame([{"IATA_CODE":k, "DEGREE":v} for k,v in flight_graph.degree], columns=["IATA_CODE", "DEGREE"])
airports_centrality = pd.merge(airports, degree_df, on='IATA_CODE')
airports_centrality
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode11.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode11.py)

结果如下所示：

![图形中心性](img/B09699_09_14.jpg)

增加了 DEGREE 列的机场数据框

为了在 Mapbox 地图中可视化数据，我们只需对 `airport_centrality` 数据框使用 `PixieDust.display()`：

```py
display(airports_centrality)
```

以下截图显示了选项对话框：

![图形中心性](img/B09699_09_15.jpg)

Mapbox 显示机场的选项

在选项对话框点击 **OK** 后，我们得到以下结果：

![图形中心性](img/B09699_09_16.jpg)

显示具有度中心性的机场

对于其他中心性指标，我们可以注意到，相关的计算函数都返回一个 JSON 输出（与度属性的数组不同），其中 `IATA_CODE` 作为机场代码，中心性指数作为值。

例如，如果我们使用以下代码计算 PageRank：

```py
nx.pagerank(flight_graph)
```

我们得到以下结果：

```py
{'ABE': 0.0011522441195896051,
 'ABI': 0.0006671948649909588,
 ...
 'YAK': 0.001558809391270303,
 'YUM': 0.0006214341604372096}
```

考虑到这一点，我们可以实现一个通用的函数 `compute_centrality()`，而不是重复为 `degree` 所做的相同步骤。该函数接受计算中心性的函数和列名作为参数，创建一个包含计算中心性值的临时数据框，并将其与 `airports_centrality` 数据框合并。

以下代码展示了 `compute_centrality()` 的实现：

```py
from six import iteritems
def compute_centrality(g, centrality_df, compute_fn, col_name, *args, **kwargs):
    # create a temporary DataFrame that contains the computed centrality values
    temp_df = pd.DataFrame(
        [{"IATA_CODE":k, col_name:v} for k,v in iteritems(compute_fn(g, *args, **kwargs))],
        columns=["IATA_CODE", col_name]
    )
    # make sure to remove the col_name from the centrality_df is already there
    if col_name in centrality_df.columns:
        centrality_df.drop([col_name], axis=1, inplace=True)
    # merge the 2 DataFrame on the IATA_CODE column
    centrality_df = pd.merge(centrality_df, temp_df, on='IATA_CODE')
    return centrality_df

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode12.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode12.py)

我们现在可以简单地调用 `compute_centrality()` 方法，并使用三个计算函数 `nx.pagerank()`、`nx.closeness_centrality()` 和 `nx.betweenness_centrality()`，并分别将 `PAGE_RANK`、`CLOSENESS` 和 `BETWEENNESS` 作为列，如下所示的代码：

```py
airports_centrality = compute_centrality(flight_graph, airports_centrality, nx.pagerank, "PAGE_RANK")
airports_centrality = compute_centrality(flight_graph, airports_centrality, nx.closeness_centrality, "CLOSENESS")
airports_centrality = compute_centrality(
    flight_graph, airports_centrality, nx.betweenness_centrality, "BETWEENNESS", k=len(flight_graph))
airports_centrality
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode13.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode13.py)

`airports_centrality` 数据框现在增加了额外的列，如下所示的输出：

![图形中心性](img/B09699_09_17.jpg)

增加了 PAGE_RANK、CLOSENESS 和 BETWEENNESS 值的机场数据框

作为练习，我们可以验证四个中心性指数对排名前的机场提供一致的结果。使用 pandas 的 `nlargest()` 方法，我们可以获得四个指数的前 10 个机场，如下所示的代码：

```py
for col_name in ["DEGREE", "PAGE_RANK", "CLOSENESS", "BETWEENNESS"]:
    print("{} : {}".format(
        col_name,
        airports_centrality.nlargest(10, col_name)["IATA_CODE"].values)
    )
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode14.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode14.py)

这将产生以下结果：

```py
DEGREE : ['ATL' 'ORD' 'DFW' 'DEN' 'MSP' 'IAH' 'DTW' 'SLC' 'EWR' 'LAX']
PAGE_RANK : ['ATL' 'ORD' 'DFW' 'DEN' 'MSP' 'IAH' 'DTW' 'SLC' 'SFO' 'LAX']
CLOSENESS : ['ATL' 'ORD' 'DFW' 'DEN' 'MSP' 'IAH' 'DTW' 'SLC' 'EWR' 'LAX']
BETWEENNESS : ['ATL' 'DFW' 'ORD' 'DEN' 'MSP' 'SLC' 'DTW' 'ANC' 'IAH' 'SFO']
```

正如我们所看到的，亚特兰大机场在所有中心性指标中排名第一。作为一个练习，让我们创建一个通用方法 `visualize_neighbors()`，用来可视化给定节点的所有邻居，并通过亚特兰大节点调用它。在这个方法中，我们通过从父节点到所有邻居添加边，创建一个以父节点为中心的子图。我们使用 NetworkX 的 `neighbors()` 方法获取特定节点的所有邻居。

以下代码展示了 `visualize_neighbors()` 方法的实现：

```py
import matplotlib.cm as cm
def visualize_neighbors(parent_node):
    fig = plt.figure(figsize = (12,12))
    # Create a subgraph and add an edge from the parent node to all its neighbors
    graph = nx.DiGraph()
    for neighbor in flight_graph.neighbors(parent_node):
        graph.add_edge(parent_node, neighbor)
    # draw the subgraph
    nx.draw(graph, arrows=True, with_labels=True,
            width = 0.5,style="dotted",
            node_color=range(len(graph)),
            cmap=cm.get_cmap(name="cool"),
            edge_color=range(len(graph.edges)),
            edge_cmap=cm.get_cmap(name="spring"),
           )
    plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode15.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode15.py)

然后我们在 `ATL` 节点上调用 `visualize_neighbors()` 方法：

```py
visualize_neighbors("ATL")
```

该方法生成如下输出：

![图中心性](img/B09699_09_18.jpg)

可视化顶点 ATL 及其邻居

我们通过计算使用著名的 Dijkstra 算法在两个节点之间的最短路径来完成 *第一部分*，该算法的详细信息请参见（[`en.wikipedia.org/wiki/Dijkstra%27s_algorithm`](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)）。我们将尝试不同的权重属性，看看是否能得到不同的结果。

例如，让我们使用 NetworkX 的 `dijkstra_path()` 方法计算从马萨诸塞州波士顿洛根机场（`BOS`）到华盛顿帕斯科三城市机场（`PSC`）之间的最短路径（[`networkx.github.io/documentation/networkx-2.1/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html`](https://networkx.github.io/documentation/networkx-2.1/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html)）。

我们首先将 `ELAPSED_TIME` 列作为权重属性：

### 注意

**注意**：提醒一下，`ELAPSED_TIME` 是我们在本节早些时候计算的，从相同出发地和目的地机场的每个航班的平均飞行时间。

```py
nx.dijkstra_path(flight_graph, "BOS", "PSC", weight="ELAPSED_TIME")
```

该方法返回：

```py
['BOS', 'MSP', 'PSC']
```

不幸的是，我们之前计算的中心性指标不属于`flight_graph`数据框，因此将其用作`weight`属性的列名是行不通的。然而，`dijkstra_path()`也允许我们使用一个函数来动态计算权重。由于我们希望尝试不同的中心性指标，我们需要创建一个工厂方法（[`en.wikipedia.org/wiki/Factory_method_pattern`](https://en.wikipedia.org/wiki/Factory_method_pattern)），该方法会为传入的中心性指标创建一个函数。这个参数作为闭包传递给一个嵌套的包装函数，符合`dijkstra_path()`方法的`weight`参数要求。我们还使用了一个`cache`字典来记住计算出的某个机场的权重，因为算法会对同一个机场多次调用该函数。如果权重不在缓存中，我们会在`airports_centrality`数据框中使用`centrality_indice_col`参数查找。最终的权重通过获取中心性值的倒数来计算，因为 Dijkstra 算法偏向于选择较短路径。

以下代码展示了`compute_weight`工厂方法的实现：

```py
# use a cache so we don't recompute the weight for the same airport every time
cache = {}
def compute_weight(centrality_indice_col):
    # wrapper function that conform to the dijkstra weight argument
    def wrapper(source, target, attribute):
        # try the cache first and compute the weight if not there
        source_weight = cache.get(source, None)
        if source_weight is None:
            # look up the airports_centrality for the value
            source_weight = airports_centrality.loc[airports_centrality["IATA_CODE"] == source][centrality_indice_col].values[0]
            cache[source] = source_weight
        target_weight = cache.get(target, None)
        if target_weight is None:
            target_weight = airports_centrality.loc[airports_centrality["IATA_CODE"] == target][centrality_indice_col].values[0]
            cache[target] = target_weight
        # Return weight is inversely proportional to the computed weighted since
        # the Dijkstra algorithm give precedence to shorter distances
        return float(1/source_weight) + float(1/target_weight)
    return wrapper
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode16.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode16.py)

我们现在可以针对每个中心性指标调用 NetworkX 的`dijkstra_path()`方法。请注意，我们没有使用 BETWEENNESS，因为一些值为零，不能用作权重。在调用`dijkstra_path()`方法之前，我们还需要清除缓存，因为使用不同的中心性指标会为每个机场生成不同的值。

以下代码展示了如何计算每个中心性指标的最短路径：

```py
for col_name in ["DEGREE", "PAGE_RANK", "CLOSENESS"]:
    #clear the cache
    cache.clear()
    print("{} : {}".format(
        col_name,
        nx.dijkstra_path(flight_graph, "BOS", "PSC",
                         weight=compute_weight(col_name))
    ))
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode17.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode17.py)

下面是产生的结果：

```py
DEGREE : ['BOS', 'DEN', 'PSC']
PAGE_RANK : ['BOS', 'DEN', 'PSC']
CLOSENESS : ['BOS', 'DEN', 'PSC']
```

有趣的是，正如预期的那样，计算出的最短路径对于三个中心性指标是相同的，都经过丹佛机场，这是一座重要的枢纽机场。然而，它与使用`ELAPSED_TIME`权重计算的路径不同，后者会让我们经过明尼阿波利斯。

在本节中，我们展示了如何将美国航班数据加载到图数据结构中，计算不同的中心性指标，并利用这些指标计算机场之间的最短路径。我们还讨论了不同的图数据可视化方法。

### 注意

*第一部分*的完整笔记本可以在这里找到：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/USFlightsAnalysis/US%20Flight%20data%20analysis%20-%20Part%201.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/USFlightsAnalysis/US%20Flight%20data%20analysis%20-%20Part%201.ipynb)

在接下来的部分，我们将创建`USFlightsAnalysis` PixieApp，将这些分析功能实现并投入使用。

# 第二部分 – 创建 USFlightsAnalysis PixieApp

在我们`USFlightsAnalysis`的第一次迭代中，我们希望实现一个简单的用户故事，利用*第一部分*创建的分析功能：

+   欢迎屏幕将显示两个下拉控制，用于选择出发机场和目的地机场。

+   当选择一个机场时，我们会显示一个图表，展示所选机场及其邻近的机场。

+   当两个机场都被选择时，用户点击**分析**按钮，显示一个包含所有机场的 Mapbox 地图。

+   用户可以选择一个中心性指标（作为复选框）来根据选定的中心性显示最短的飞行路径。

首先让我们来看一下欢迎屏幕的实现，它是在`USFlightsAnalysis` PixieApp 的默认路由中实现的。以下代码定义了`USFlightsAnalysis`类，并用`@PixieApp`装饰器将其标记为 PixieApp。它包含一个用`@route()`装饰器装饰的`main_screen()`方法，将其设置为默认路由。该方法返回一个 HTML 片段，该片段将在 PixieApp 启动时作为欢迎屏幕使用。HTML 片段由两部分组成：一部分显示选择出发机场的下拉控制，另一部分包含选择目的地机场的下拉控制。我们使用 Jinja2 的`{%for...%}`循环遍历每个机场（由`get_airports()`方法返回），生成一组`<options>`元素。在每个控制下方，我们添加一个占位符`<div>`元素，当选择机场时，这个`<div>`将承载图表可视化。

### 注意

**注意**：和往常一样，我们使用`[[USFlightsAnalysis]]`符号来表示代码只显示了部分实现，因此读者在完整实现提供之前不应尝试直接运行。

我们稍后会解释为什么`USFlightsAnalysis`类继承自`MapboxBase`类。

```py
[[USFlightsAnalysis]]
from pixiedust.display.app import *
from pixiedust.apps.mapboxBase import MapboxBase
from collections import OrderedDict

@PixieApp
class USFlightsAnalysis(MapboxBase):
    ...
    @route()
    def main_screen(self):
        return """
<style>
    div.outer-wrapper {
        display: table;width:100%;height:300px;
    }
    div.inner-wrapper {
        display: table-cell;vertical-align: middle;height: 100%;width: 100%;
    }
</style>
<div class="outer-wrapper">
    <div class="inner-wrapper">
        <div class="col-sm-6">
            <div class="rendererOpt" style="font-weight:bold">
                 Select origin airport:
            </div>
            <div>
                <select id="origin_airport{{prefix}}"
                        pd_refresh="origin_graph{{prefix}}">
                    <option value="" selected></option>
                    {%for code, airport in this.get_airports() %}
 <option value="{{code}}">{{code}} - {{airport}}</option>
 {%endfor%}
                </select>
            </div>
            <div id="origin_graph{{prefix}}" pd_options="visualize_graph=$val(origin_airport{{prefix}})"></div>
        </div>
        <div class="input-group col-sm-6">
            <div class="rendererOpt" style="font-weight:bold">
                 Select destination airport:
            </div>
            <div>
                <select id="destination_airport{{prefix}}"
                        pd_refresh="destination_graph{{prefix}}">
                    <option value="" selected></option>
                    {%for code, airport in this.get_airports() %}
 <option value="{{code}}">{{code}} - {{airport}}</option>
 {%endfor%}
                </select>
            </div>
            <div id="destination_graph{{prefix}}"
pd_options="visualize_graph=$val(destination_airport{{prefix}})">
            </div>
        </div>
    </div>
</div>
<div style="text-align:center">
    <button class="btn btn-default" type="button"
pd_options="org_airport=$val(origin_airport{{prefix}});dest_airport=$val(destination_airport{{prefix}})">
        <pd_script type="preRun">
            if ($("#origin_airport{{prefix}}").val() == "" || $("#destination_airport{{prefix}}").val() == ""){
                alert("Please select an origin and destination airport");
                return false;
            }
            return true;
        </pd_script>
        Analyze
    </button>
</div>
"""

def get_airports(self):
    return [tuple(l) for l in airports_centrality[["IATA_CODE", "AIRPORT"]].values.tolist()]

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode18.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode18.py)

当用户选择出发机场时，会触发一个`pd_refresh`，该操作针对 ID 为`origin_graph{{prefix}}`的占位符`<div>`元素。进而，该`<div>`元素会触发一个路由，使用状态：`visualize_graph=$val(origin_airport{{prefix}})`。作为提醒，`$val()`指令在运行时解析，通过获取`origin_airport{{prefix}}`下拉菜单元素的机场值来实现。目的地机场的实现也类似。

`visualize_graph`路由的代码提供如下。它简单地调用了我们在*第一部分*中实现的`visualize_neighbors()`方法，并在*第二部分*中稍作修改，增加了一个可选的图形大小参数，以适应主机`<div>`元素的大小。作为提醒，我们还使用了`@captureOutput`装饰器，因为`visualize_neighbors()`方法直接写入选定单元格的输出：

```py
[[USFlightsAnalysis]]
@route(visualize_graph="*")
@captureOutput
def visualize_graph_screen(self, visualize_graph):
    visualize_neighbors(visualize_graph, (5,5))
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode19.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode19.py)

`Analyze`按钮触发了`compute_path_screen()`路由，该路由与`org_airport`和`dest_airport`状态参数相关联。我们还希望在允许`compute_path_screen()`路由继续之前，确保两个机场都已选择。为此，我们使用一个`<pd_script>`子元素，`type="preRun"`，其中包含将在触发路由之前执行的 JavaScript 代码。该代码的契约是：如果我们希望让路由继续执行，它返回布尔值`true`，否则返回`false`。

对于`Analyze`按钮，我们检查两个下拉菜单是否都有值，如果是，则返回`true`，否则抛出错误信息并返回`false`：

```py
<button class="btn btn-default" type="button" pd_options="org_airport=$val(origin_airport{{prefix}});dest_airport=$val(destination_airport{{prefix}})">
   <pd_script type="preRun">
 if ($("#origin_airport{{prefix}}").val() == "" || $("#destination_airport{{prefix}}").val() == ""){
 alert("Please select an origin and destination airport");
 return false;
 }
 return true;
   </pd_script>
      Analyze
   </button>
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode20.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode20.html)

以下输出显示了选择 BOS 作为出发机场，PSC 作为目的地时的最终结果：

![Part 2 – 创建 USFlightsAnalysis PixieApp](img/B09699_09_19.jpg)

欢迎界面，已选择两个机场

现在让我们看看`compute_path_screen()`路由的实现，它负责显示所有机场的 Mapbox 地图，以及基于所选的中心性指标作为图层的最短路径，这是一个附加的可视化层，叠加在整体地图上。

以下代码展示了其实现：

```py
[[USFlightsAnalysis]]
@route(org_airport="*", dest_airport="*")
def compute_path_screen(self, org_airport, dest_airport):
    return """
<div class="container-fluid">
    <div class="form-group col-sm-2" style="padding-right:10px;">
        <div><strong>Centrality Indices</strong></div>
 {% for centrality in this.centrality_indices.keys() %}
        <div class="rendererOpt checkbox checkbox-primary">
            <input type="checkbox"
                   pd_refresh="flight_map{{prefix}}"
pd_script="self.compute_toggle_centrality_layer('{{org_airport}}', '{{dest_airport}}', '{{centrality}}')">
            <label>{{centrality}}</label>
        </div>
 {%endfor%}
    </div>
    <div class="form-group col-sm-10">
        <h1 class="rendererOpt">Select a centrality index to show the shortest flight path
        </h1>
        <div id="flight_map{{prefix}}" pd_entity="self.airports_centrality" pd_render_onload>
            <pd_options>
 {
 "keyFields": "LATITUDE,LONGITUDE",
 "valueFields": "AIRPORT,DEGREE,PAGE_RANK,ELAPSED_TIME,CLOSENESS",
 "custombasecolorsecondary": "#fffb00",
 "colorrampname": "Light to Dark Red",
 "handlerId": "mapView",
 "quantiles": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
 "kind": "choropleth",
 "rowCount": "1000",
 "numbins": "5",
 "mapboxtoken": "pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4M29iazA2Z2gycXA4N2pmbDZmangifQ.-g_vE53SD2WrJ6tFX7QHmA",
 "custombasecolor": "#ffffff"
 }
            </pd_options>
        </div>
    </div>
</div>
"""
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode21.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode21.py)

该界面的中央`<div>`元素是 Mapbox 地图，默认显示所有机场的 Mapbox 地图。如上面的代码所示，`<pd_options>`子元素直接取自相应单元格的元数据，其中我们在*第一部分*中配置了地图。

在左侧，我们通过 Jinja2 的`{%for …%}`循环，针对每个中心性指数生成一组复选框，循环的目标是`centrality_indices`变量。我们在`USFlightsAnalysis` PixieApp 的`setup()`方法中初始化了这个变量，该方法在 PixieApp 启动时必定会被调用。此变量是一个 OrderedDict（[`docs.python.org/3/library/collections.html#collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict)），键是中心性指数，值是将用于 Mapbox 渲染的颜色方案：

```py
[[USFlightsAnalysis]]
def setup(self):
   self.centrality_indices = OrderedDict([
      ("ELAPSED_TIME","rgba(256,0,0,0.65)"),
      ("DEGREE", "rgba(0,256,0,0.65)"),
      ("PAGE_RANK", "rgba(0,0,256,0.65)"),
      ("CLOSENESS", "rgba(128,0,128,0.65)")
  ])
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode22.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode22.py)

以下输出显示了未选择中心性指数的分析界面：

![第二部分 – 创建 USFlightsAnalysis PixieApp](img/B09699_09_20.jpg)

未选择中心性指数的分析界面

我们现在到达了用户选择中心性指数以触发最短路径搜索的步骤。每个复选框都有一个`pd_script`属性，调用`compute_toggle_centrality_layer()`方法。该方法负责调用 NetworkX 的`dijkstra_path()`方法，并传入一个`weight`参数，该参数通过调用我们在*第一部分*中讨论的`compute_weight()`方法生成。此方法返回一个包含构成最短路径的每个机场的数组。利用该路径，我们创建一个包含 GeoJSON 有效负载的 JSON 对象，该有效负载作为一组要显示在地图上的线路。

此时，值得暂停讨论一下什么是图层。**图层**是使用 GeoJSON 格式（[`geojson.org`](http://geojson.org)）定义的，我们在第五章中简要讨论过，*Python 和 PixieDust 最佳实践与高级概念*。提醒一下，GeoJSON 有效负载是一个具有特定模式的 JSON 对象，其中包括定义绘制对象形状的`geometry`元素等内容。

例如，我们可以使用`LineString`类型和一个包含线路两端经纬度坐标的数组来定义一条线：

```py
{
    "geometry": {
        "type": "LineString",
        "coordinates": [
            [-93.21692, 44.88055],
 [-119.11903000000001, 46.26468]
        ]
    },
    "type": "Feature",
    "properties": {}
}
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode23.json`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode23.json)

假设我们能够从最短路径生成这个 GeoJSON 有效载荷，我们可能会想知道如何将其传递给 PixieDust Mapbox 渲染器，以便显示。其实，机制非常简单：Mapbox 渲染器会检查主机 PixieApp 中是否有符合特定格式的类变量，并利用它生成要显示的 Mapbox 图层。为了帮助遵循这一机制，我们使用了之前简要介绍的`MapboxBase`工具类。该类有一个`get_layer_index()`方法，接受一个唯一的名称（我们使用`centrality`索引）作为参数并返回其索引。它还接受一个额外的可选参数，在图层不存在时创建该图层。然后，我们调用`toggleLayer()`方法，传递图层索引作为参数来打开和关闭图层。

以下代码展示了`compute_toggle_centrality_layer()`方法的实现，该方法实现了上述步骤：

```py
[[USFlightsAnalysis]]
def compute_toggle_centrality_layer(self, org_airport, dest_airport, centrality):
    cache.clear()
    cities = nx.dijkstra_path(flight_graph, org_airport, dest_airport, weight=compute_weight(centrality))
    layer_index = self.get_layer_index(centrality, {
        "name": centrality,
        "geojson": {
            "type": "FeatureCollection",
            "features":[
                {"type":"Feature",
                 "properties":{"route":"{} to {}".format(cities[i], cities[i+1])},
                 "geometry":{
                     "type":"LineString",
                     "coordinates":[
                         self.get_airport_location(cities[i]),
 self.get_airport_location(cities[i+1])
                     ]
                 }
                } for i in range(len(cities) - 1)
            ]
        },
        "paint":{
 "line-width": 8,
 "line-color": self.centrality_indices[centrality]
 }
    })
 self.toggleLayer(layer_index)

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode24.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode24.py)

几何对象中的坐标是使用`get_airport_location()`方法计算的，该方法查询了我们在*第一部分*中创建的`airports_centrality`数据框，如下代码所示：

```py
[[USFlightsAnalysis]]
def get_airport_location(self, airport_code):
    row = airports_centrality.loc[airports["IATA_CODE"] == airport_code]
    if row is not None:
        return [row["LONGITUDE"].values[0], row["LATITUDE"].values[0]]
    return None
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode25.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode25.py)

传递给`get_layer_index()`方法的图层对象具有以下属性：

+   `name`：唯一标识图层的字符串。

+   `geojson`：GeoJSON 对象，定义了图层的特征和几何形状。

+   `url`：仅在`geojson`不存在时使用。指向一个返回 GeoJSON 有效载荷的 URL。

+   `paint`：Mapbox 规范中定义的可选附加属性，指定图层数据的样式，例如颜色、宽度和透明度。

+   `layout`：Mapbox 规范中定义的可选附加属性，指定图层数据的绘制方式，例如填充、可见性和符号。

### 注意

**注意**：你可以在这里找到更多关于 Mapbox 布局和绘制属性的信息：

[`www.mapbox.com/mapbox-gl-js/style-spec/#layers`](https://www.mapbox.com/mapbox-gl-js/style-spec/#layers)

在上面的代码中，我们指定了额外的`paint`属性来配置`line-width`和`line-color`，这些属性来自在`setup()`方法中定义的`centrality_indices` JSON 对象。

以下输出显示了从`BOS`到`PSC`的最短飞行路径，使用了**ELAPSED_TIME**（红色）和**DEGREE**（绿色）中心性指标：

![第二部分 – 创建 USFlightsAnalysis PixieApp](img/B09699_09_21.jpg)

使用 ELAPSED_TIME 和 DEGREE 中心性指标显示从 BOS 到 PSC 的最短路径

在这一部分中，我们构建了一个 PixieApp，它使用 PixieDust Mapbox 渲染器可视化两个机场之间的最短路径。我们展示了如何使用`MapboxBase`工具类创建新图层，以丰富地图信息。

### 注意

你可以在这里找到完成的 Notebook 文件，*第二部分*：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/USFlightsAnalysis/US%20Flight%20data%20analysis%20-%20Part%202.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/USFlightsAnalysis/US%20Flight%20data%20analysis%20-%20Part%202.ipynb)

在接下来的部分，我们将添加与航班延误和相关航空公司有关的额外数据探索内容。

# 第三部分 – 向 USFlightsAnalysis PixieApp 添加数据探索功能

在这一部分中，我们希望扩展`USFlightsAnalysis` PixieApp 的路线分析界面，添加两张图表，展示从选定起始机场起飞的每个航空公司的历史到达延误情况：一张显示所有从该机场起飞的航班，另一张则显示所有航班（不管机场在哪里）。这将使我们能够直观地比较特定机场的延误情况与其他机场的延误情况。

我们从实现一个方法开始，该方法选择给定航空公司的航班。我们还添加了一个可选的机场参数，可以用来控制是包括所有航班，还是仅包括从该机场起飞的航班。返回的 DataFrame 应该包含两列：`DATE`和`ARRIVAL_DELAY`。

以下代码展示了此方法的实现：

```py
def compute_delay_airline_df(airline, org_airport=None):
    # create a mask for selecting the data
    mask = (flights["AIRLINE"] == airline)
    if org_airport is not None:
        # Add the org_airport to the mask
        mask = mask & (flights["ORIGIN_AIRPORT"] == org_airport)
    # Apply the mask to the Pandas dataframe
    df = flights[mask]
    # Convert the YEAR, MONTH and DAY column into a DateTime
    df["DATE"] = pd.to_datetime(flights[['YEAR','MONTH', 'DAY']])
    # Select only the columns that we need
    return df[["DATE", "ARRIVAL_DELAY"]]
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode26.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode26.py)

我们可以通过使用它来测试前面的代码，选择波士顿的 Delta 航班。然后，我们可以调用 PixieDust 的`display()`方法，创建一个线性图表，在 PixieApp 中使用：

```py
bos_delay = compute_delay_airline_df("DL", "BOS")
display(bos_delay)
```

在 PixieDust 输出中，我们选择**折线图**菜单，并按如下方式配置选项对话框：

![第三部分 – 向 USFlightsAnalysis PixieApp 添加数据探索功能](img/B09699_09_22.jpg)

为生成波士顿出发的 Delta 航班到达延误线性图表的选项对话框

点击**确定**后，我们得到以下图表：

![第三部分 – 向 USFlightsAnalysis PixieApp 添加数据探索功能](img/B09699_09_23.jpg)

展示所有从波士顿起飞的 Delta 航班的延误情况

由于我们将在 PixieApp 中使用此图表，因此从**编辑单元格元数据**对话框复制 JSON 配置是个不错的主意：

![第三部分 – 向 USFlightsAnalysis PixieApp 添加数据探索功能](img/B09699_09_24.jpg)

PixieDust display() 配置，用于延误图表，需要复制到 PixieApp 中

现在我们知道如何生成延迟图表了，可以开始设计 PixieApp。我们首先通过更改主屏幕的布局，使用 `TemplateTabbedApp` 辅助类，这样就能免费得到标签式布局。整体分析屏幕现在由 `RouteAnalysisApp` 子类 PixieApp 驱动，包含两个标签：一个是与 `SearchShortestRouteApp` 子类 PixieApp 相关的 `Search Shortest Route` 标签，另一个是与 `AirlinesApp` 子类 PixieApp 相关的 `Explore Airlines` 标签。

以下图表提供了新布局中涉及的所有类的高层次流程：

![第三部分 – 向 USFlightsAnalysis PixieApp 添加数据探索](img/B09699_09_25.jpg)

新的标签式布局类图

`RouteAnalysisApp` 的实现非常直接，使用 `TemplateTabbedApp`，如下代码所示：

```py
from pixiedust.apps.template import TemplateTabbedApp

@PixieApp
class RouteAnalysisApp(TemplateTabbedApp):
    def setup(self):
        self.apps = [
            {"title": "Search Shortest Route",
             "app_class": "SearchShortestRouteApp"},
            {"title": "Explore Airlines",
             "app_class": "AirlinesApp"}
        ]

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode27.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode27.py)

`SearchShortestRouteApp` 子类 PixieApp 基本上是我们在*第二部分*中创建的主要 PixieApp 类的副本。唯一的不同是它是 `RouteAnalysisApp` 的子类 PixieApp，而 `RouteAnalysisApp` 本身又是 `USFlightsAnalysis` 主要 PixieApp 的子类。因此，我们需要一种机制将起始和目的地机场传递给各自的子类 PixieApp。为此，我们在实例化 `RouteAnalysisApp` 子类 PixieApp 时使用 `pd_options` 属性。

在 `USFlightAnalysis` 类中，我们将 `analyze_route` 方法更改为返回一个简单的 `<div>` 元素，该元素触发 `RouteAnalysisApp`。我们还添加了一个包含 `org_airport` 和 `dest_airport` 的 `pd_options` 属性，如以下代码所示：

```py
[[USFlightsAnalysis]]
@route(org_airport="*", dest_airport="*")
def analyze_route(self, org_airport, dest_airport):
    return """
<div pd_app="RouteAnalysisApp"
pd_options="org_airport={{org_airport}};dest_airport={{dest_airport}}"
     pd_render_onload>
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode28.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode28.py)

相反，在 `SearchShortestRouteApp` 子类 PixieApp 的 `setup()` 方法中，我们从 `parent_pixieapp` 的 options 字典中读取 `org_airport` 和 `dest_airport` 的值，如以下代码所示：

```py
[[SearchShortestRouteApp]]
from pixiedust.display.app import *
from pixiedust.apps.mapboxBase import MapboxBase
from collections import OrderedDict

@PixieApp
class SearchShortestRouteApp(MapboxBase):
    def setup(self):
 self.org_airport = self.parent_pixieapp.options.get("org_airport")
 self.dest_airport = self.parent_pixieapp.options.get("dest_airport")
        self.centrality_indices = OrderedDict([
            ("ELAPSED_TIME","rgba(256,0,0,0.65)"),
            ("DEGREE", "rgba(0,256,0,0.65)"),
            ("PAGE_RANK", "rgba(0,0,256,0.65)"),
            ("CLOSENESS", "rgba(128,0,128,0.65)")
        ])
        ...
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode29.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode29.py)

**注意**：由于 `SearchShortestRouteApp` 的其余实现与 *第二部分* 完全相同，因此已省略。若要访问实现，请参阅完整的 *第三部分* Notebook。

最后要实现的 PixieApp 类是`AirlinesApp`，它将显示所有的延误图表。与`SearchShortestRouteApp`类似，我们从`parent_pixieapp`选项字典中存储了`org_airport`和`dest_airport`。我们还计算了一个元组列表（代码和名称），列出了所有从给定`org_airport`起飞的航空公司。为了实现这一点，我们使用 pandas 的`groupby()`方法对`AIRLINE`列进行分组，并获取索引值列表，代码如下所示：

```py
[[AirlinesApp]]
@PixieApp
class AirlinesApp():
    def setup(self):
        self.org_airport = self.parent_pixieapp.options.get("org_airport")
 self.dest_airport = self.parent_pixieapp.options.get("dest_airport")
        self.airlines = flights[flights["ORIGIN_AIRPORT"] == self.org_airport].groupby("AIRLINE").size().index.values.tolist()
        self.airlines = [(a, airlines.loc[airlines["IATA_CODE"] == a]["AIRLINE"].values[0]) for a in self.airlines]
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode30.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode30.py)

在`AirlinesApp`的主屏幕上，我们使用 Jinja2 的`{%for...%}`循环为每个航空公司生成一组行。在每一行中，我们添加两个`<div>`元素，用于显示给定航空公司的延误折线图：一个用于显示从起始机场出发的航班，另一个用于显示该航空公司的所有航班。每个`<div>`元素都有一个`pd_options`属性，其中`org_airport`和`dest_airport`作为状态属性，触发`delay_airline_screen`路由。我们还添加了一个`delay_org_airport`布尔状态属性，用于表示我们想要显示哪种类型的延误图表。为了确保`<div>`元素能够立即渲染，我们还添加了`pd_render_onload`属性。

以下代码展示了`AirlinesApp`默认路由的实现：

```py
[[AirlinesApp]]
@route()
    def main_screen(self):
        return """
<div class="container-fluid">
    {%for airline_code, airline_name in this.airlines%}
    <div class="row" style="max-e">
        <h1 style="color:red">{{airline_name}}</h1>
        <div class="col-sm-6">
            <div pd_render_onload pd_options="delay_org_airport=true;airline_code={{airline_code}};airline_name={{airline_name}}"></div>
        </div>
        <div class="col-sm-6">
            <div pd_render_onload pd_options="delay_org_airport=false;airline_code={{airline_code}};airline_name={{airline_name}}"></div>
        </div>
    </div>
    {%endfor%}
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode31.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode31.py)

`delay_airline_screen()`路由有三个参数：

+   `delay_org_airport`：如果我们只想显示从起始机场出发的航班，则为`true`；如果我们想显示该航空公司的所有航班，则为`false`。我们使用这个标志来构建过滤航班 DataFrame 数据的掩码。

+   `airline_code`：给定航空公司的 IATA 代码。

+   `airline_name`：航空公司的全名。在构建 Jinja2 模板的 UI 时我们会使用这个值。

在`delay_airline_screen()`方法的主体中，我们还计算了所选数据的平均延误，并将结果保存在`average_delay`局部变量中。提醒一下，为了在 Jinja2 模板中使用此变量，我们使用了`@templateArgs`装饰器，它会自动使所有局部变量在 Jinja2 模板中可用。

承载图表的 `<div>` 元素具有一个 `pd_entity` 属性，使用我们在本节开头创建的 `compute_delay_airline_df()` 方法。然而，由于参数发生了变化，我们需要将此方法重写为类的成员：`org_airport` 现在是一个类变量，`delay_org_airport` 现在是一个字符串布尔值。我们还添加了一个 `<pd_options>` 子元素，其中包含我们从 **编辑单元格元数据** 对话框复制的 PixieDust `display()` JSON 配置。

以下代码展示了 `delay_airline_screen()` 路由的实现：

```py
[[AirlinesApp]]
@route(delay_org_airport="*",airline_code="*", airline_name="*")
    @templateArgs
    def delay_airline_screen(self, delay_org_airport, airline_code, airline_name):
        mask = (flights["AIRLINE"] == airline_code)
        if delay_org_airport == "true":
            mask = mask & (flights["ORIGIN_AIRPORT"] == self.org_airport)
        average_delay = round(flights[mask]["ARRIVAL_DELAY"].mean(), 2)
        return """
{%if delay_org_airport == "true" %}
<h4>Delay chart for all flights out of {{this.org_airport}}</h4>
{%else%}
<h4>Delay chart for all flights</h4>
{%endif%}
<h4 style="margin-top:5px">Average delay: {{average_delay}} minutes</h4>
<div pd_render_onload pd_entity="compute_delay_airline_df('{{airline_code}}', '{{delay_org_airport}}')">
    <pd_options>
    {
 "keyFields": "DATE",
 "handlerId": "lineChart",
 "valueFields": "ARRIVAL_DELAY",
 "noChartCache": "true"
 }
    </pd_options>
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode32.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode32.py)

`compute_delay_airline_df()` 方法有两个参数：对应 IATA 代码的航空公司和 `delay_org_airport` 字符串布尔值。我们已经介绍了该方法的实现，但这里提供了更新后的代码：

```py
[[AirlinesApp]]
def compute_delay_airline_df(self, airline, delay_org_airport):
        mask = (flights["AIRLINE"] == airline)
        if delay_org_airport == "true":
            mask = mask & (flights["ORIGIN_AIRPORT"] == self.org_airport)
        df = flights[mask]
        df["DATE"] = pd.to_datetime(flights[['YEAR','MONTH', 'DAY']])
        return df[["DATE", "ARRIVAL_DELAY"]]

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode33.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode33.py)

运行 `USFlightsAnalysis` PixieApp，将 BOS 和 PSC 分别设置为起点和终点机场时，我们点击 **探索航空公司** 标签。

结果如下图所示：

![Part 3 – 向 USFlightsAnalysis PixieApp 添加数据探索](img/B09699_09_26.jpg)

显示所有从波士顿机场提供服务的航空公司的延误线图

本节中，我们提供了另一个示例，展示如何使用 PixieApp 编程模型构建强大的仪表盘，提供可视化和分析结果的洞察，展示 Notebook 中开发的分析输出。

### 注意

完整的 *Part 3* Notebook 可在以下位置找到：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/USFlightsAnalysis/US%20Flight%20data%20analysis%20-%20Part%203.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/USFlightsAnalysis/US%20Flight%20data%20analysis%20-%20Part%203.ipynb)

在接下来的部分，我们将构建一个 ARIMA 模型，尝试预测航班延误。

# 第四部分 - 创建一个 ARIMA 模型来预测航班延误

在第八章，*分析研究：预测 - 财务时间序列分析与预测*中，我们使用时间序列分析构建了一个预测金融股票的预测模型。我们实际上可以使用相同的技术来分析航班延误，因为毕竟我们这里处理的也是时间序列，因此，在本节中，我们将遵循完全相同的步骤。对于每个目的地机场和可选航空公司，我们将构建一个包含匹配航班信息的 pandas DataFrame。

### 注意

**注意**：我们将再次使用`statsmodels`库。如果你还没有安装它，请确保先安装，并参考第八章，*分析学习：预测 - 金融时间序列分析与预测*，获取更多信息。

作为例子，让我们关注所有以`BOS`为目的地的 Delta（`DL`）航班：

```py
df = flights[(flights["AIRLINE"] == "DL") & (flights["ORIGIN_AIRPORT"] == "BOS")]
```

使用`ARRIVAL_DELAY`列作为时间序列的值，我们绘制 ACF 和 PACF 图来识别趋势和季节性，如下代码所示：

```py
import statsmodels.tsa.api as smt
smt.graphics.plot_acf(df['ARRIVAL_DELAY'], lags=100)
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode34.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode34.py)

结果如下所示：

![Part 4 – 创建 ARIMA 模型预测航班延误](img/B09699_09_27.jpg)

ARRIVAL_DELAY 数据的自相关函数

同样，我们使用以下代码绘制偏自相关函数：

```py
import statsmodels.tsa.api as smt
smt.graphics.plot_pacf(df['ARRIVAL_DELAY'], lags=50)
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode35.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode35.py)

结果如下所示：

![Part 4 – 创建 ARIMA 模型预测航班延误](img/B09699_09_28.jpg)

ARRIVAL_DELAY 数据的偏自相关

从前面的图表中，我们可以推测数据具有趋势和/或季节性，并且它不是平稳的。使用我们在第八章，*分析学习：预测 - 金融时间序列分析与预测*中解释的对数差分技术，我们转换该序列，并使用 PixieDust 的`display()`方法进行可视化，如以下代码所示：

### 注意

**注意**：我们还确保通过首先调用`replace()`方法将`np.inf`和`-np.inf`替换为`np.nan`，然后调用`dropna()`方法移除所有包含`np.nan`值的行，从而移除包含 NA 和无限值的行。

```py
import numpy as np
train_set, test_set = df[:-14], df[-14:]
train_set.index = train_set["DEPARTURE_TIME"]
test_set.index = test_set["DEPARTURE_TIME"]
logdf = np.log(train_set['ARRIVAL_DELAY'])
logdf.index = train_set['DEPARTURE_TIME']
logdf_diff = pd.DataFrame(logdf - logdf.shift()).reset_index()
logdf_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
logdf_diff.dropna(inplace=True)
display(logdf_diff)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode36.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode36.py)

以下截图显示了 PixieDust 选项对话框：

![Part 4 – 创建 ARIMA 模型预测航班延误](img/B09699_09_29.jpg)

ARRIVAL_DELAY 数据的对数差分选项对话框

点击**确定**后，我们得到以下结果：

### 注意

**注意**：在运行上述代码时，你可能不会得到与以下截图完全相同的图表。这是因为我们在选项对话框中将**显示行数**配置为`100`，这意味着 PixieDust 将在创建图表之前从中抽取 100 个样本。

![Part 4 – 创建用于预测航班延误的 ARIMA 模型](img/B09699_09_30.jpg)

ARRIVAL_DELAY 数据的对数差分折线图

前面的图看起来是平稳的；我们可以通过对对数差分重新绘制 ACF 和 PACF 来强化这个假设，如下代码所示：

```py
smt.graphics.plot_acf(logdf_diff["ARRIVAL_DELAY"], lags=100)
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode37.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode37.py)

结果如下所示：

![Part 4 – 创建用于预测航班延误的 ARIMA 模型](img/B09699_09_31.jpg)

ARRIVAL_DELAY 数据的对数差分 ACF 图

在以下代码中，我们对 PACF 进行了相同的操作：

```py
smt.graphics.plot_pacf(logdf_diff["ARRIVAL_DELAY"], lags=100)
plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode38.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode38.py)

结果如下所示：

![Part 4 – 创建用于预测航班延误的 ARIMA 模型](img/B09699_09_32.jpg)

ARRIVAL_DELAY 数据的对数差分 PACF 图

提醒一下，参考第八章，*分析研究：预测 - 财务时间序列分析与预测*，ARIMA 模型由三个阶数组成：*p*、*d*和*q*。从前面的两张图，我们可以推断出我们要构建的 ARIMA 模型的这些阶数：

+   **自回归阶数 p 为 1**：对应 ACF 首次穿越显著性水平的时刻

+   **差分阶数 d 为 1**：我们进行了 1 次对数差分

+   **移动平均阶数 q 为 1**：对应 PACF 首次穿越显著性水平的时刻

根据这些假设，我们可以使用`statsmodels`包构建 ARIMA 模型，并获取其残差误差信息，如下代码所示：

```py
from statsmodels.tsa.arima_model import ARIMA

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    arima_model_class = ARIMA(train_set['ARRIVAL_DELAY'],
                              dates=train_set['DEPARTURE_TIME'],
                              order=(1,1,1))
    arima_model = arima_model_class.fit(disp=0)
    print(arima_model.resid.describe())
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode39.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode39.py)

结果如下所示：

```py
count    13882.000000
mean         0.003116
std         48.932043
min       -235.439689
25%        -17.446822
50%         -5.902274
75%          6.746263
max       1035.104295
dtype: float64
```

如我们所见，均值误差仅为 0.003，非常好，因此我们可以准备使用`train_set`中的数据运行模型，并将结果与实际值的差异可视化。

以下代码使用 ARIMA 的`plot_predict()`方法创建图表：

```py
def plot_predict(model, dates_series, num_observations):
    fig,ax = plt.subplots(figsize = (12,8))
    model.plot_predict(
        start = dates_series[len(dates_series)-num_observations],
        end = dates_series[len(dates_series)-1],
        ax = ax
    )
    plt.show()
plot_predict(arima_model, train_set['DEPARTURE_TIME'], 100)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode40.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode40.py)

结果如下所示：

![Part 4 – 创建用于预测航班延误的 ARIMA 模型](img/B09699_09_33.jpg)

预测与实际值对比

在前面的图表中，我们可以清晰地看到，预测线比实际值平滑得多。这是有道理的，因为实际上，航班延误总是会有一些意外的原因，这些原因可能会被视为离群值，因此很难进行建模。

我们仍然需要使用`test_set`来验证模型，使用的是模型尚未见过的数据。以下代码创建了一个`compute_test_set_predictions()`方法，用于比较预测数据和测试数据，并通过 PixieDust 的`display()`方法可视化结果：

```py
def compute_test_set_predictions(train_set, test_set):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = train_set['ARRIVAL_DELAY'].values
        forecast = np.array([])
        for t in range(len(test_set)):
            prediction = ARIMA(history, order=(1,1,0)).fit(disp=0).forecast()
            history = np.append(history, test_set['ARRIVAL_DELAY'].iloc[t])
            forecast = np.append(forecast, prediction[0])
        return pd.DataFrame(
          {"forecast": forecast,
 "test": test_set['ARRIVAL_DELAY'],
 "Date": pd.date_range(start=test_set['DEPARTURE_TIME'].iloc[len(test_set)-1], periods = len(test_set))
 }
        )

results = compute_test_set_predictions(train_set, test_set)
display(results)
```

### 注释

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode41.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode41.py)

这里显示的是 PixieDust 选项对话框：

![第四部分 – 创建 ARIMA 模型以预测航班延误](img/B09699_09_34.jpg)

预测与测试对比折线图的选项对话框

点击**确定**后，我们得到以下结果：

![第四部分 – 创建 ARIMA 模型以预测航班延误](img/B09699_09_35.jpg)

预测值与测试数据的折线图

现在我们准备将这个模型集成到我们的`USFlightsAnalysis` PixieApp 中，通过在`RouteAnalysisApp`主界面添加一个新的选项卡，名为`航班延误预测`。该选项卡将由一个名为`PredictDelayApp`的新子 PixieApp 驱动，用户可以选择使用 Dijkstra 最短路径算法计算出的最短路径航班段，`DEGREE`作为中心性指标。用户还可以选择航空公司，在这种情况下，训练数据将仅限于由所选航空公司运营的航班。

在以下代码中，我们创建了`PredictDelayApp`子 PixieApp 并实现了`setup()`方法，计算所选起点和终点机场的 Dijkstra 最短路径：

```py
[[PredictDelayApp]]
import warnings
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

@PixieApp
class PredictDelayApp():
    def setup(self):
        self.org_airport = self.parent_pixieapp.options.get("org_airport")
        self.dest_airport = self.parent_pixieapp.options.get("dest_airport")
        self.airlines = flights[flights["ORIGIN_AIRPORT"] == self.org_airport].groupby("AIRLINE").size().index.values.tolist()
        self.airlines = [(a, airlines.loc[airlines["IATA_CODE"] == a]["AIRLINE"].values[0]) for a in self.airlines]
        path = nx.dijkstra_path(flight_graph, self.org_airport, self.dest_airport, weight=compute_weight("DEGREE"))
        self.paths = [(path[i], path[i+1]) for i in range(len(path) - 1)]
```

在`PredictDelayApp`的默认路由中，我们使用 Jinja2 的`{%for..%}`循环构建了两个下拉框，显示航班段和航空公司，如下所示：

```py
[[PredictDelayApp]]
@route()
    def main_screen(self):
        return """
<div class="container-fluid">
    <div class="row">
        <div class="col-sm-6">
            <div class="rendererOpt" style="font-weight:bold">
                Select a flight segment:
            </div>
            <div>
                <select id="segment{{prefix}}" pd_refresh="prediction_graph{{prefix}}">
                    <option value="" selected></option>
                    {%for start, end in this.paths %}
 <option value="{{start}}:{{end}}">{{start}} -> {{end}}</option>
 {%endfor%}
                </select>
            </div>
        </div>
        <div class="col-sm-6">
            <div class="rendererOpt" style="font-weight:bold">
                Select an airline:
            </div>
            <div>
                <select id="airline{{prefix}}" pd_refresh="prediction_graph{{prefix}}">
                    <option value="" selected></option>
                    {%for airline_code, airline_name in this.airlines%}
 <option value="{{airline_code}}">{{airline_name}}</option>
 {%endfor%}
                </select>
            </div>
        </div>
    </div>
    <div class="row">
        <div class="col-sm-12">
            <div id="prediction_graph{{prefix}}"
                pd_options="flight_segment=$val(segment{{prefix}});airline=$val(airline{{prefix}})">
            </div>
        </div>
    </div>
</div>
        """
```

### 注释

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode42.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode42.py)

这两个下拉框具有`pd_refresh`属性，该属性指向 ID 为`prediction_graph{{prefix}}`的`<div>`元素。当触发时，该`<div>`元素会调用`predict_screen()`路由，并使用`flight_segment`和`airline`的状态属性。

在`predict_screen()`路由中，我们使用`flight_segment`和`airline`参数来创建训练数据集，建立一个 ARIMA 模型来进行预测，并通过折线图展示预测值与实际值的对比。

### 注释

时间序列预测模型的限制在于只能预测接近实际数据的结果，由于我们只有 2015 年的数据，因此无法使用该模型预测更新的数据。当然，在生产应用中，假设我们有当前的航班数据，因此这不会是一个问题。

以下代码展示了`predict_screen()`路由的实现：

```py
[[PredictDelayApp]]
@route(flight_segment="*", airline="*")
 @captureOutput
    def predict_screen(self, flight_segment, airline):
        if flight_segment is None or flight_segment == "":
            return "<div>Please select a flight segment</div>"
 airport = flight_segment.split(":")[1]
 mask = (flights["DESTINATION_AIRPORT"] == airport)
        if airline is not None and airline != "":
 mask = mask & (flights["AIRLINE"] == airline)
        df = flights[mask]
        df.index = df["DEPARTURE_TIME"]
        df = df.tail(50000)
 df = df[~df.index.duplicated(keep='first')]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model_class = ARIMA(df["ARRIVAL_DELAY"], dates=df['DEPARTURE_TIME'], order=(1,1,1))
            arima_model = arima_model_class.fit(disp=0)
            fig, ax = plt.subplots(figsize = (12,8))
            num_observations = 100
            date_series = df["DEPARTURE_TIME"]
 arima_model.plot_predict(
 start = str(date_series[len(date_series)-num_observations]),
 end = str(date_series[len(date_series)-1]),
 ax = ax
 )
            plt.show()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode43.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode43.py)

在以下代码中，我们还希望确保数据集索引去重，以避免在绘制结果时发生错误。通过使用`df = df[~df.index.duplicated(keep='first')]`来过滤重复的索引。

最后要做的事情是将`PredictDelayApp`子应用 PixieApp 接入到`RouteAnalysisApp`，如下代码所示：

```py
from pixiedust.apps.template import TemplateTabbedApp

@PixieApp
class RouteAnalysisApp(TemplateTabbedApp):
    def setup(self):
        self.apps = [
            {"title": "Search Shortest Route",
             "app_class": "SearchShortestRouteApp"},
            {"title": "Explore Airlines",
             "app_class": "AirlinesApp"},
            {"title": "Flight Delay Prediction",
 "app_class": "PredictDelayApp"}
        ]
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode44.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/sampleCode44.py)

当我们运行`USFlightsAnalysis` PixieApp 并使用 BOS 和 PSC，如前几节所做的那样，在**航班延误预测**标签页中，选择**BOS->DEN**航班段。

结果如下所示：

![Part 4 – 创建一个 ARIMA 模型来预测航班延误](img/B09699_09_36.jpg)

波士顿到丹佛航段的预测

在本节中，我们展示了如何使用时间序列预测模型基于历史数据来预测航班延误。

### 注意

你可以在这里找到完整的 Notebook：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/USFlightsAnalysis/US%20Flight%20data%20analysis%20-%20Part%204.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%209/USFlightsAnalysis/US%20Flight%20data%20analysis%20-%20Part%204.ipynb)

提醒一下，虽然这只是一个示例应用，仍有很多改进空间，但使用 PixieApp 编程模型将数据分析转化为实际应用的技巧，在任何其他项目中也同样适用。

# 总结

在本章中，我们讨论了图形及其相关的图论，探索了图的结构和算法。我们还简要介绍了`networkx` Python 库，它提供了一套丰富的 API，用于操作和可视化图形。然后，我们将这些技巧应用于构建一个示例应用，该应用通过将航班数据视为图问题（机场为顶点，航班为边）来进行分析。像往常一样，我们还展示了如何将这些分析转化为一个简单而强大的仪表盘，该仪表盘可以直接在 Jupyter Notebook 中运行，然后可以选择性地作为 Web 分析应用通过 PixieGateway 微服务部署。

本章完成了一系列涵盖许多重要行业应用的示例。在下一章中，我将对本书的主题做一些最终的思考，主题是通过简化和使数据工作变得对所有人可及，架起数据科学与工程之间的桥梁。
