# 第六章：使用 SparkSQL 为您的大数据添加结构

在本章中，我们将学习如何使用 Spark SQL 模式操作数据框，并使用 Spark DSL 构建结构化数据操作的查询。到目前为止，我们已经学会了将大数据导入 Spark 环境使用 RDD，并对这些大数据进行多个操作。现在让我们看看如何操作我们的数据框并构建结构化数据操作的查询。

具体来说，我们将涵盖以下主题：

+   使用 Spark SQL 模式操作数据框

+   使用 Spark DSL 构建查询

# 使用 Spark SQL 模式操作数据框

在本节中，我们将学习更多关于数据框，并学习如何使用 Spark SQL。

Spark SQL 接口非常简单。因此，去除标签意味着我们处于无监督学习领域。此外，Spark 对聚类和降维算法有很好的支持。通过使用 Spark SQL 为大数据赋予结构，我们可以有效地解决学习问题。

让我们看一下我们将在 Jupyter Notebook 中使用的代码。为了保持一致，我们将使用相同的 KDD 杯数据：

1.  我们首先将`textFile`输入到`raw_data`变量中，如下所示：

```py
raw_data = sc.textFile("./kddcup.data.gz")
```

1.  新的是我们从`pyspark.sql`中导入了两个新包：

+   `Row`

+   `SQLContext`

1.  以下代码向我们展示了如何导入这些包：

```py
from pyspark.sql import Row, SQLContext
sql_context = SQLContext(sc)
csv = raw_data.map(lambda l: l.split(","))
```

使用`SQLContext`，我们创建一个新的`sql_context`变量，其中包含由 PySpark 创建的`SQLContext`变量的对象。由于我们使用`SparkContext`来启动这个`SQLContext`变量，我们需要将`sc`作为`SQLContext`创建者的第一个参数。之后，我们需要取出我们的`raw_data`变量，并使用`l.split`lambda 函数将其映射为一个包含我们的逗号分隔值（CSV）的对象。

1.  我们将利用我们的新重要`Row`对象来创建一个新对象，其中定义了标签。这是为了通过我们正在查看的特征对我们的数据集进行标记，如下所示：

```py
rows = csv.map(lambda p: Row(duration=int(p[0]), protocol=p[1], service=p[2]))
```

在上面的代码中，我们取出了我们的逗号分隔值（csv），并创建了一个`Row`对象，其中包含第一个特征称为`duration`，第二个特征称为`protocol`，第三个特征称为`service`。这直接对应于实际数据集中的标签。

1.  现在，我们可以通过在`sql_context`变量中调用`createDataFrame`函数来创建一个新的数据框。要创建这个数据框，我们需要提供我们的行数据对象，结果对象将是`df`中的数据框。之后，我们需要注册一个临时表。在这里，我们只是称之为`rdd`。通过这样做，我们现在可以使用普通的 SQL 语法来查询由我们的行构造的临时表中的内容：

```py
df = sql_context.createDataFrame(rows)
df.registerTempTable("rdd")
```

1.  在我们的示例中，我们需要从`rdd`中选择`duration`，这是一个临时表。我们在这里选择的协议等于`'tcp'`，而我们在一行中的第一个特征是大于`2000`的`duration`，如下面的代码片段所示：

```py
sql_context.sql("""SELECT duration FROM rdd WHERE protocol = 'tcp' AND duration > 2000""")
```

1.  现在，当我们调用`show`函数时，它会给我们每个符合这些条件的数据点：

```py
sql_context.sql("""SELECT duration FROM rdd WHERE protocol = 'tcp' AND duration > 2000""").show()
```

1.  然后我们将得到以下输出：

```py
+--------+
|duration|
+--------+
|   12454|
|   10774|
|   13368|
|   10350|
|   10409|
|   14918|
|   10039|
|   15127|
|   25602|
|   13120|
|    2399|
|    6155|
|   11155|
|   12169|
|   15239|
|   10901|
|   15182|
|    9494|
|    7895|
|   11084|
+--------+
only showing top 20 rows
```

使用前面的示例，我们可以推断出我们可以使用 PySpark 包中的`SQLContext`变量将数据打包成 SQL 友好格式。

因此，PySpark 不仅支持使用 SQL 语法查询数据，还可以使用 Spark 领域特定语言（DSL）构建结构化数据操作的查询。

# 使用 Spark DSL 构建查询

在本节中，我们将使用 Spark DSL 构建结构化数据操作的查询：

1.  在以下命令中，我们使用了与之前相同的查询；这次使用了 Spark DSL 来说明和比较使用 Spark DSL 与 SQL 的不同之处，但实现了与我们在前一节中展示的 SQL 相同的目标：

```py
df.select("duration").filter(df.duration>2000).filter(df.protocol=="tcp").show()
```

在这个命令中，我们首先取出了在上一节中创建的`df`对象。然后我们通过调用`select`函数并传入`duration`参数来选择持续时间。

1.  接下来，在前面的代码片段中，我们两次调用了`filter`函数，首先使用`df.duration`，第二次使用`df.protocol`。在第一种情况下，我们试图查看持续时间是否大于`2000`，在第二种情况下，我们试图查看协议是否等于`"tcp"`。我们还需要在命令的最后附加`show`函数，以获得与以下代码块中显示的相同结果。

```py
+--------+
|duration|
+--------+
|   12454|
|   10774|
|   13368|
|   10350|
|   10409|
|   14918|
|   10039|
|   15127|
|   25602|
|   13120|
|    2399|
|    6155|
|   11155|
|   12169|
|   15239|
|   10901|
|   15182|
|    9494|
|    7895|
|   11084|
+--------+
only showing top 20 rows
```

在这里，我们再次有了符合代码描述的前 20 行数据点的结果。

# 总结

在本章中，我们涵盖了 Spark DSL，并学习了如何构建查询。我们还学习了如何使用 Spark SQL 模式操纵 DataFrames，然后我们使用 Spark DSL 构建了结构化数据操作的查询。现在我们对 Spark 有了很好的了解，让我们在接下来的章节中看一些 Apache Spark 中的技巧和技术。

在下一章中，我们将看一下 Apache Spark 程序中的转换和操作。
