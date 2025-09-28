

# 第六章：Spark SQL 中的 SQL 查询

在本章中，我们将探索 Spark SQL 在结构化数据处理方面的广泛功能。我们将深入了解加载数据、操作数据、执行 SQL 查询、执行高级分析和将 Spark SQL 与外部系统集成。到本章结束时，您将深入了解 Spark SQL 的功能，并具备利用其在数据处理任务中发挥其强大功能的知识。

我们将涵盖以下主题：

+   什么是 Spark SQL？

+   Spark SQL 入门

+   高级 Spark SQL 操作

# 什么是 Spark SQL？

Spark SQL 是 Apache Spark 生态系统中的一个强大模块，它允许高效地处理和分析结构化数据。它提供了一个比 Apache Spark 传统的基于 RDD 的 API 更高级的接口来处理结构化数据。Spark SQL 结合了关系和过程处理的优势，使用户能够无缝地将 SQL 查询与复杂分析集成。通过利用 Spark 的分布式计算能力，Spark SQL 实现了可扩展和高效的数据处理。

它提供了一个编程接口，使用 SQL 查询、DataFrame API 和 Dataset API 来处理结构化数据。

它允许用户使用类似 SQL 的语法查询数据，并为在大型数据集上执行 SQL 查询提供强大的引擎。Spark SQL 还支持从各种结构化数据源（如 Hive 表、Parquet 文件和 JDBC 数据库）读取和写入数据。

## Spark SQL 的优势

Spark SQL 提供了几个关键优势，使其成为结构化数据处理的热门选择：

### 使用 Spark SQL 进行统一数据处理

使用 Spark SQL，用户可以使用单个引擎处理结构化和非结构化数据。这意味着用户可以使用相同的编程接口查询存储在不同格式（如 JSON、CSV 和 Parquet）中的数据。

用户可以在 SQL 查询、DataFrame 转换和 Spark 的机器学习 API 之间无缝切换。这种统一的数据处理方法使得在单个应用程序中集成不同的数据处理任务变得更加容易，从而降低了开发复杂性。

### 性能和可扩展性

Spark SQL 利用 Apache Spark 的分布式计算能力，在机器集群上处理大规模数据集。它使用高级查询优化技术，如 Catalyst 优化器（在第五章中详细讨论），来优化和加速查询执行。此外，Spark SQL 支持数据分区和缓存机制，进一步提高了性能和可扩展性。

Spark SQL 使用优化的执行引擎，可以比传统的 SQL 引擎更快地处理查询。它通过使用内存缓存和优化的查询执行计划来实现这一点。

Spark SQL 被设计为可以在机器集群上水平扩展。它可以通过将数据集分区到多台机器上并并行处理来处理大型数据集。

### 与现有基础设施的无缝集成

Spark SQL 与现有的 Apache Spark 基础设施和工具无缝集成。它与其他 Spark 组件（如 Spark Streaming 用于实时数据处理和 Spark MLlib 用于机器学习任务）提供互操作性。此外，Spark SQL 与流行的存储系统和数据格式（包括 Parquet、Avro、ORC 和 Hive）集成，使其与广泛的数据源兼容。

### 高级分析功能

Spark SQL 通过使用高级分析功能扩展了传统的 SQL 功能。它支持窗口函数，使用户能够执行复杂的分析操作，例如排名、滑动窗口上的聚合和累积聚合。与 Spark 中的机器学习库的集成允许预测分析和数据科学工作流程的无缝集成。

### 易用性

Spark SQL 提供了一个简单的编程接口，允许用户使用类似 SQL 的语法查询数据。这使得熟悉 SQL 的用户能够轻松开始使用 Spark SQL。

## 与 Apache Spark 的集成

Spark SQL 是 Apache Spark 框架的组成部分，与 Spark 的其他组件无缝协作。它利用 Spark 的核心功能，如容错、数据并行和分布式计算，以提供可扩展和高效的数据处理。Spark SQL 可以从各种来源读取数据，包括分布式文件系统（如 HDFS）、对象存储（如 Amazon S3）和关系数据库（通过 JDBC）。它还与外部系统（如 Hive）集成，使用户能够利用现有的 Hive 元数据和查询。

现在，让我们看看 Spark SQL 的一些基本结构。

## 关键概念 – DataFrame 和 Dataset

Spark SQL 引入了两个基本抽象，用于处理结构化数据：DataFrame 和 Dataset。

### DataFrame

DataFrame 表示组织成命名列的分布式数据集合。它们提供了高级接口，用于处理结构化数据，并提供了丰富的 API 用于数据操作、过滤、聚合和查询。DataFrame 是不可变的，并且是惰性评估的，通过 Spark 的 Catalyst 优化器实现优化执行计划。它们可以从各种数据源创建，包括结构化文件（CSV、JSON 和 Parquet）、Hive 表和现有的 RDD。

### Dataset

Dataset 是 DataFrame 的扩展，提供了一种类型安全、面向对象的编程接口。Dataset 结合了 Spark 的 RDD（强类型和用户定义函数）的优点与 DataFrame 的性能优化。Dataset 允许编译时类型检查，并且可以无缝转换为 DataFrame，从而实现灵活高效的数据处理。

现在我们已经知道了 DataFrame 和 Dataset 是什么，我们将在下一节中看到如何将这些结构应用于不同的 Spark SQL 操作。

# 开始使用 Spark SQL

要开始使用 Spark SQL 操作，我们首先需要将数据加载到 DataFrame 中。我们将在下一节中看到如何做到这一点。然后，我们将看到如何在不同数据之间切换 PySpark 和 Spark SQL，并对它应用不同的转换。

## 加载数据和保存数据

在本节中，我们将探索从不同来源将数据加载到 Spark SQL 中并保存为表的各种技术。我们将深入研究 Python 代码示例，演示如何有效地将数据加载到 Spark SQL 中，执行必要的转换，并将处理后的数据保存为表以供进一步分析。

在 Spark SQL 中执行 SQL 查询使我们能够利用熟悉的 SQL 语法并利用其表达力。让我们看看执行 SQL 查询的语法和示例：

在 Spark SQL 中执行 SQL 查询，我们使用 `spark.sql()` 方法，如下所示：

```py
results = spark.sql("SELECT * FROM tableName")
```

+   `spark.sql()` 方法用于在 Spark SQL 中执行 SQL 查询

+   在方法内部，我们提供 SQL 查询作为字符串参数。在这个例子中，我们从 `tableName` 表中选择所有列。

+   查询的结果存储在 `results` 变量中，可以进一步处理或按需显示。

为了开始本章的代码示例，我们将使用在 *第四章* 中创建的 DataFrame。

```py
salary_data_with_id = [(1, "John", "Field-eng", 3500, 40), \
    (2, "Robert", "Sales", 4000, 38), \
    (3, "Maria", "Finance", 3500, 28), \
    (4, "Michael", "Sales", 3000, 20), \
    (5, "Kelly", "Finance", 3500, 35), \
    (6, "Kate", "Finance", 3000, 45), \
    (7, "Martin", "Finance", 3500, 26), \
    (8, "Kiran", "Sales", 2200, 35), \
  ]
columns= ["ID", "Employee", "Department", "Salary", "Age"]
salary_data_with_id = spark.createDataFrame(data = salary_data_with_id, schema = columns)
salary_data_with_id.show()
```

输出结果如下：

```py
+---+--------+----------+------+---+
| ID|Employee|Department|Salary|Age|
+---+--------+----------+------+---+
|  1|    John| Field-eng|  3500| 40|
|  2|  Robert|     Sales|  4000| 38|
|  3|   Maria|   Finance|  3500| 28|
|  4| Michael|     Sales|  3000| 20|
|  5|   Kelly|   Finance|  3500| 35|
|  6|    Kate|   Finance|  3000| 45|
|  7|  Martin|   Finance|  3500| 26|
|  8|   Kiran|     Sales|  2200| 35|
+---+--------+----------+------+---+
```

我在这个 DataFrame 中添加了一个 `Age` 列以进行进一步处理。

记住，在 *第四章* 中，我们将这个 DataFrame 保存为 CSV 文件。我们用于读取 CSV 文件的代码片段可以在以下代码中看到。

如你所回忆的，我们使用以下代码行用 Spark 编写 CSV 文件：

```py
salary_data_with_id.write.format("csv").mode("overwrite").option("header", "true").save("salary_data.csv")
```

输出结果如下：

```py
+---+--------+----------+------+---+
| ID|Employee|Department|Salary|Age|
+---+--------+----------+------+---+
|  1|    John| Field-eng|  3500| 40|
|  2|  Robert|     Sales|  4000| 38|
|  3|   Maria|   Finance|  3500| 28|
|  4| Michael|     Sales|  3000| 20|
|  5|   Kelly|   Finance|  3500| 35|
|  6|    Kate|   Finance|  3000| 45|
|  7|  Martin|   Finance|  3500| 26|
|  8|   Kiran|     Sales|  2200| 35|
+---+--------+----------+------+---+
```

现在我们有了 DataFrame，我们可以使用 SQL 操作来处理它：

```py
# Perform transformations on the loaded data
processed_data = csv_data.filter(csv_data["Salary"] > 3000)
# Save the processed data as a table
processed_data.createOrReplaceTempView("high_salary_employees")
# Perform SQL queries on the saved table
results = spark.sql("SELECT * FROM high_salary_employees ")
results.show()
```

输出结果如下：

```py
+---+--------+----------+------+---+
| ID|Employee|Department|Salary|Age|
+---+--------+----------+------+---+
|  1|    John| Field-eng|  3500| 40|
|  2|  Robert|     Sales|  4000| 38|
|  3|   Maria|   Finance|  3500| 28|
|  5|   Kelly|   Finance|  3500| 35|
|  7|  Martin|   Finance|  3500| 26|
+---+--------+----------+------+---+
```

上述代码片段展示了如何对加载的数据执行转换。在这种情况下，我们过滤数据，只包括 `Salary` 列大于 3,000 的行。

通过使用 `filter()` 函数，我们可以应用特定条件来选择所需的数据子集。

转换后的数据将存储在 `results` 变量中，并准备好进行进一步分析。

### 将转换后的数据保存为视图

一旦我们完成了必要的转换，通常将处理后的数据保存为视图以便于访问和未来的分析是有用的。让我们看看如何在 Spark SQL 中实现这一点：

`createOrReplaceTempView()` 方法允许我们将处理后的数据作为 Spark SQL 中的视图保存。我们为视图提供名称，在本例中为 `high_salary_employees`。

通过给表赋予一个有意义的名称，我们可以在后续的操作和查询中轻松引用它。保存的表作为处理数据的结构化表示，便于进一步分析和探索。

将转换后的数据保存为表格后，我们可以利用 SQL 查询的功能来获取洞察力并提取有价值的信息。

通过使用`spark.sql()`方法，我们可以对保存的视图`high_salary_employees`执行 SQL 查询。

在前面的示例中，我们执行了一个简单的查询，根据过滤条件从视图中选择所有列。

`show()`函数显示了 SQL 查询的结果，使我们能够检查从数据集中提取的所需信息。

在 Spark SQL 中创建视图的另一种方法是`createTempView()`。与`createOrReplaceTempView()`方法相比，`createTempView()`只会尝试创建一个视图。如果该视图名称已存在于目录中，则会抛出`TempTableAlreadyExistsException`异常。

## 利用 Spark SQL 根据特定标准过滤和选择数据

在本节中，我们将探讨执行 SQL 查询和应用转换的语法和实际示例。

让我们考虑一个实际示例，其中我们执行一个 SQL 查询来过滤和选择表中的特定数据：

```py
# Save the processed data as a view
salary_data_with_id.createOrReplaceTempView("employees")
#Apply filtering on data
filtered_data = spark.sql("SELECT Employee, Department, Salary, Age FROM employees WHERE age > 30")
# Display the results
filtered_data.show()
```

输出结果如下：

```py
+--------+----------+------+---+
|Employee|Department|Salary|Age|
+--------+----------+------+---+
|    John| Field-eng|  3500| 40|
|  Robert|     Sales|  4000| 38|
|   Kelly|   Finance|  3500| 35|
|    Kate|   Finance|  3000| 45|
|   Kiran|     Sales|  2200| 35|
+--------+----------+------+---+
```

在本例中，我们创建一个临时视图，名为`employees`，并使用 Spark SQL 执行 SQL 查询以过滤和选择`employees`表中的特定列。

查询选择了`employee`、`department`、`salary`和`age`列，其中`age`大于 30。查询的结果存储在`filtered_data`变量中。

最后，我们调用`show()`方法来显示过滤后的数据。

## 探索使用 Spark SQL 进行排序和聚合操作

Spark SQL 提供了一组丰富的转换函数，可以应用于操作和转换数据。让我们探索一些 Spark SQL 中转换的实际示例：

### 聚合

在本例中，我们使用 Spark SQL 执行聚合操作，从`employees`表中计算平均工资。

```py
# Perform an aggregation to calculate the average salary
average_salary = spark.sql("SELECT AVG(Salary) AS average_salary FROM employees")
# Display the average salary
average_salary.show()
```

输出结果如下：

```py
+--------------+
|average_salary|
+--------------+
|        3275.0|
+--------------+
```

`AVG()`函数计算`salary`列的平均值。我们使用 AS 关键字将结果别名为`average_salary`。

结果存储在`average_salary`变量中，并使用`show()`方法显示：

### 排序

在本例中，我们使用 Spark SQL 对`employees`表应用排序转换。

```py
# Sort the data based on the salary column in descending order
sorted_data = spark.sql("SELECT * FROM employees ORDER BY Salary DESC")
# Display the sorted data
sorted_data.show()
```

输出结果如下：

```py
+---+--------+----------+------+---+
| ID|Employee|Department|Salary|Age|
+---+--------+----------+------+---+
|  2|  Robert|     Sales|  4000| 38|
|  1|    John| Field-eng|  3500| 40|
|  5|   Kelly|   Finance|  3500| 35|
|  3|   Maria|   Finance|  3500| 28|
|  7|  Martin|   Finance|  3500| 26|
|  6|    Kate|   Finance|  3000| 45|
|  4| Michael|     Sales|  3000| 20|
|  8|   Kiran|     Sales|  2200| 35|
+---+--------+----------+------+---+
```

使用`ORDER BY`子句来指定排序标准，在本例中是对`salary`列进行降序排序。

排序后的数据存储在`sorted_data`变量中，并使用`show()`方法进行显示。

### 结合聚合

我们还可以在一个 SQL 命令中结合不同的聚合操作，如下面的代码示例所示：

```py
# Sort the data based on the salary column in descending order
filtered_data = spark.sql("SELECT Employee, Department, Salary, Age FROM employees WHERE age > 30 AND Salary > 3000 ORDER BY Salary DESC")
# Display the results
filtered_data.show()
```

输出结果如下：

```py
+--------+----------+------+---+
|Employee|Department|Salary|Age|
+--------+----------+------+---+
|  Robert|     Sales|  4000| 38|
|   Kelly|   Finance|  3500| 35|
|    John| Field-eng|  3500| 40|
+--------+----------+------+---+
```

在这个例子中，我们使用 Spark SQL 对 `employees` 表进行不同的转换。首先，我们选择那些年龄大于 30 岁且工资大于 3,000 的员工。`ORDER BY` 子句用于指定排序标准；在这种情况下，按 `salary` 列降序排序。

结果数据存储在“`filtered_data`”变量中，并使用 `show()` 方法显示。

在本节中，我们探讨了使用 Spark SQL 执行 SQL 查询和应用转换的过程。我们学习了执行 SQL 查询的语法，并展示了执行查询、过滤数据、执行聚合和排序数据的实际示例。通过利用 SQL 的表达能力和 Spark SQL 的灵活性，您可以高效地分析和操作结构化数据，以完成各种数据分析任务。

## 根据特定列进行分组和聚合数据 – 基于特定列进行分组并执行聚合函数

在 Spark SQL 中，分组和聚合数据是常见的操作，用于从大型数据集中获取洞察和总结信息。本节将探讨如何使用 Spark SQL 根据特定列分组数据并执行各种聚合函数。我们将通过代码示例演示 Spark SQL 在此方面的功能。

### 数据分组

当我们想要根据特定列对数据进行分组时，可以利用 SQL 查询中的 `GROUP BY` 子句。让我们考虑一个例子，其中我们有一个包含 `department` 和 `salary` 列的员工 DataFrame。我们想要计算每个部门的平均工资：

```py
# Group the data based on the Department column and take average salary for each department
grouped_data = spark.sql("SELECT Department, avg(Salary) FROM employees GROUP BY Department")
# Display the results
grouped_data.show()
```

输出将如下所示：

```py
+----------+------------------+
|Department|       avg(Salary)|
+----------+------------------+
| Field-eng|            3500.0|
|     Sales|3066.6666666666665|
|   Finance|            3375.0|
+----------+------------------+
```

在这个例子中，我们使用 Spark SQL 对 `employees` 表的不同转换进行数据分组。首先，我们根据 `Department` 列对员工进行分组。我们从 `employees` 表中获取每个部门的平均工资。

结果数据存储在 `grouped_data` 变量中，并使用 `show()` 方法显示。

### 数据聚合

Spark SQL 提供了广泛的聚合函数，用于对分组数据进行汇总统计。让我们考虑另一个例子，其中我们想要计算每个部门的总工资和最高工资：

```py
# Perform grouping and multiple aggregations
aggregated_data = spark.sql("SELECT Department, sum(Salary) AS total_salary, max(Salary) AS max_salary FROM employees GROUP BY Department")
# Display the results
aggregated_data.show()
```

输出将如下所示：

```py
+----------+-----------+-----------+
|Department|sum(Salary)|max(Salary)|
+----------+-----------+-----------+
| Field-eng|       3500|       3500|
|     Sales|       9200|       4000|
|   Finance|      13500|       3500|
+----------+-----------+-----------+
```

在这个例子中，我们使用 Spark SQL 对 `employees` 表的不同转换进行合并和分组。首先，我们根据 `Department` 列对员工进行分组。我们从员工表中获取每个部门的总工资和最高工资。我们还为这些聚合列使用了别名。

结果数据存储在 `aggregated_data` 变量中，并使用 `show()` 方法显示。

在本节中，我们探讨了 Spark SQL 在数据分组和聚合方面的功能。我们看到了如何根据特定列分组数据并执行各种聚合函数的示例。Spark SQL 提供了广泛的聚合函数，并允许创建自定义聚合函数以满足特定需求。利用这些功能，您可以使用 Spark SQL 高效地总结和从大量数据集中获得洞察。

在下一节中，我们将探讨用于复杂数据操作的高级 Spark SQL 函数。

# 高级 Spark SQL 操作

让我们探索 Apache Spark 高级操作的关键功能。

## 利用窗口函数在 DataFrame 上执行高级分析操作

在本节中，我们将探讨 Spark SQL 中窗口函数的强大功能，用于在 DataFrame 上执行高级分析操作。窗口函数提供了一种在分区内对一组行进行计算的方法，使我们能够获得洞察并有效地执行复杂计算。在本节中，我们将深入研究窗口函数的主题，并通过展示 Spark SQL 查询中其使用的代码示例来展示其用法。

### 理解窗口函数

Spark SQL 中的窗口函数通过根据指定标准将数据集划分为组或分区，从而实现高级分析操作。这些函数在每个分区内部对滑动窗口中的行进行计算或聚合。

### 在 Spark SQL 中使用窗口函数的一般语法如下：

```py
function().over(Window.partitionBy("column1", "column2").orderBy("column3").rowsBetween(start, end))
```

`function()` 代表您想要应用的窗口函数，例如 `sum`、`avg`、`row_number` 或自定义定义的函数。`over()` 子句定义了函数应用的窗口。`Window.partitionBy()` 指定用于将数据集划分为分区的列。它还确定了每个分区内部行的顺序。`rowsBetween(start, end)` 指定窗口中包含的行范围。它可以是不限定的或相对于当前行的相对定义。

### 使用窗口函数计算累积和

让我们通过一个实际示例来探索窗口函数的使用，以计算 DataFrame 中某列的累积和：

```py
from pyspark.sql.window import Window
from pyspark.sql.functions import col, sum
# Define the window specification
window_spec = Window.partitionBy("Department").orderBy("Age")
# Calculate the cumulative sum using window function
df_with_cumulative_sum = salary_data_with_id.withColumn("cumulative_sum", sum(col("Salary")).over(window_spec))
# Display the result
df_with_cumulative_sum.show()
```

输出将如下所示：

```py
+---+--------+----------+------+---+--------------+
| ID|Employee|Department|Salary|Age|cumulative_sum|
+---+--------+----------+------+---+--------------+
|  1|    John| Field-eng|  3500| 40|          3500|
|  7|  Martin|   Finance|  3500| 26|          3500|
|  3|   Maria|   Finance|  3500| 28|          7000|
|  5|   Kelly|   Finance|  3500| 35|         10500|
|  6|    Kate|   Finance|  3000| 45|         13500|
|  4| Michael|     Sales|  3000| 20|          3000|
|  8|   Kiran|     Sales|  2200| 35|          5200|
|  2|  Robert|     Sales|  4000| 38|          9200|
+---+--------+----------+------+---+--------------+
```

在本例中，我们首先导入必要的库。我们使用与之前示例相同的 DataFrame：`salary_data_with_id`。

接下来，我们使用 `Window.partitionBy("Department").orderBy("Age")` 定义窗口规范，根据 `Department` 列对数据进行分区，并在每个分区内部根据 `Age` 列对行进行排序。

然后，我们将 `sum()` 函数用作窗口函数，在定义的窗口规范上应用，以计算 `Salary` 列的累积和。结果存储在一个名为 `cumulative_sum` 的新列中。

最后，我们调用 `show()` 方法来显示具有附加累积和列的 DataFrame。通过利用窗口函数，我们可以在 Spark SQL 中高效地计算累积和、滚动总和、滚动平均值以及其他复杂分析计算。

在本节中，我们探讨了 Spark SQL 中窗口函数的强大功能，用于高级分析。我们讨论了窗口函数的语法和用法，使我们能够在定义的分区和窗口内执行复杂计算和聚合。通过将窗口函数集成到 Spark SQL 查询中，您可以获得有价值的见解，并深入了解您的数据，以便进行高级分析操作。

在下一节中，我们将探讨 Spark 用户定义函数。

## 用户定义函数

在本节中，我们将深入探讨 Spark SQL 中的 **用户定义函数**（**UDFs**）主题。UDFs 允许我们通过定义自己的函数来扩展 Spark SQL 的功能，这些函数可以应用于 DataFrame 或 SQL 查询。在本节中，我们将探讨 UDF 的概念，并提供代码示例以展示它们在 Spark SQL 中的使用和优势。

Spark SQL 中的 UDF 允许我们创建自定义函数，以在 DataFrame 或 SQL 查询中对列进行转换或计算。当 Spark 的内置函数不能满足我们的特定要求时，UDF 特别有用。

### 要在 Spark SQL 中定义 UDF，我们使用 `pyspark.sql.functions` 模块中的 `udf()` 函数。其一般语法如下：

```py
from pyspark.sql.functions import udf
udf_name = udf(lambda_function, return_type)
```

首先，我们从 `pyspark.sql.functions` 模块导入 `udf()` 函数。接下来，我们通过提供 lambda 函数或常规 Python 函数作为 `lambda_function` 参数来定义 UDF。此函数封装了我们想要应用的自定义逻辑。

我们还指定了 UDF 的 `return_type`，它表示 UDF 将返回的数据类型。

### 将 UDF 应用到 DataFrame

让我们通过一个实际示例来探索 UDF 在 Spark SQL 中的使用，该示例通过将自定义函数应用于 DataFrame 来演示：

```py
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
# Define a UDF to capitalize a string
capitalize_udf = udf(lambda x: x.upper(), StringType())
# Apply the UDF to a column
df_with_capitalized_names = salary_data_with_id.withColumn("capitalized_name", capitalize_udf("Employee"))
# Display the result
df_with_capitalized_names.show()
```

输出将是以下内容：

```py
+---+--------+----------+------+---+----------------+
| ID|Employee|Department|Salary|Age|capitalized_name|
+---+--------+----------+------+---+----------------+
|  1|    John| Field-eng|  3500| 40|            JOHN|
|  2|  Robert|     Sales|  4000| 38|          ROBERT|
|  3|   Maria|   Finance|  3500| 28|           MARIA|
|  4| Michael|     Sales|  3000| 20|         MICHAEL|
|  5|   Kelly|   Finance|  3500| 35|           KELLY|
|  6|    Kate|   Finance|  3000| 45|            KATE|
|  7|  Martin|   Finance|  3500| 26|          MARTIN|
|  8|   Kiran|     Sales|  2200| 35|           KIRAN|
+---+--------+----------+------+---+----------------+
```

在本例中，我们首先使用 `udf()` 函数定义一个名为 `capitalize_udf` 的 UDF。它应用一个 lambda 函数，将输入字符串转换为大写。我们使用 `withColumn()` 方法将 UDF `capitalize_udf` 应用到 `name` 列，在结果 DataFrame 中创建一个名为 `capitalized_name` 的新列。

最后，我们调用 `show()` 方法来显示具有转换列的 DataFrame。

UDFs 允许我们对 DataFrame 中的列应用自定义逻辑和转换，使我们能够处理复杂计算、执行字符串操作或应用在 Spark 内置函数中不可用的特定领域操作。

在本节中，我们探讨了 Spark SQL 中 UDFs 的概念。我们讨论了定义 UDFs 的语法，并通过代码示例演示了其用法。UDFs 通过允许我们对 DataFrame 或 SQL 查询应用自定义转换和计算，提供了一个强大的机制来扩展 Spark SQL 的功能。通过将 UDFs 纳入您的 Spark SQL 工作流程，您可以处理复杂的数据操作，并定制数据处理管道以满足特定要求或特定领域的需求。

### 应用函数

PySpark 也支持各种 UDFs 和 API，允许用户在 Python 原生函数中直接使用这些 API。例如，以下示例允许用户在 Python 原生函数中直接使用 pandas 序列中的 API：

```py
import pandas as pd
from pyspark.sql.functions import pandas_udf
@pandas_udf('long')
def pandas_plus_one(series: pd.Series) -> pd.Series:
    # Simply plus one by using pandas Series.
    return series + 1
salary_data_with_id.select(pandas_plus_one(salary_data_with_id.Salary)).show()
```

输出将如下所示：

```py
+-----------------------+
|pandas_plus_one(Salary)|
+-----------------------+
|                   3501|
|                   4001|
|                   3501|
|                   3001|
|                   3501|
|                   3001|
|                   3501|
|                   2201|
+-----------------------+
```

在本例中，我们首先使用 `@pandas_udf()` 函数定义一个名为 `pandas_plus_one` 的 pandas UDF。我们定义此函数以便将其添加到 pandas 序列中。我们使用已创建的名为 `salary_data_with_id` 的 DataFrame，并调用 pandas UDF 将此函数应用于 DataFrame 的 `salary` 列。

最后，我们在同一语句中调用 `show()` 方法，以显示转换后的列的 DataFrame。

此外，UDFs 可以直接在 SQL 中注册和调用。以下是一个示例，说明我们如何实现这一点：

```py
@pandas_udf("integer")
def add_one(s: pd.Series) -> pd.Series:
    return s + 1
spark.udf.register("add_one", add_one)
spark.sql("SELECT add_one(Salary) FROM employees").show()
```

输出将如下所示：

```py
+---------------+
|add_one(Salary)|
+---------------+
|           3501|
|           4001|
|           3501|
|           3001|
|           3501|
|           3001|
|           3501|
|           2201|
+---------------+
```

在本例中，我们首先使用 `@pandas_udf()` 函数定义一个名为 `add_one` 的 pandas UDF。我们定义此函数以便将其添加到 pandas 序列中。然后，我们将此 UDF 注册以用于 SQL 函数。我们使用已创建的员工表，并调用 pandas UDF 将此函数应用于表的 `salary` 列。

最后，我们在同一语句中调用 `show()` 方法以显示结果。

在本节中，我们探讨了 UDFs 的强大功能以及我们如何在使用聚合计算中使用它们。

在下一节中，我们将探讨旋转和逆旋转函数。

## 处理复杂数据类型 – 旋转和逆旋转

旋转和逆旋转操作用于将数据从基于行的格式转换为基于列的格式，反之亦然。在 Spark SQL 中，可以使用旋转和逆旋转函数执行这些操作。

旋转函数用于将行转换为列。它接受三个参数：用作新列标题的列，用作新行标题的列，以及用作新表中值的列。结果表将具有行标题列中每个唯一值的一行，以及列标题列中每个唯一值的一列。

逆旋转函数用于将列转换为行。它接受两个参数：用作新行标题的列，以及用作新表中值的列。结果表将具有每个行标题列中值的唯一组合的一行，以及每个值的一列。

pivot 和 unpivot 操作的一些用例包括以下内容：

+   将数据从宽格式转换为长格式或反之亦然

+   通过多个维度聚合数据

+   创建汇总表或报告

+   准备数据以进行可视化或分析

总体而言，pivot 和 unpivot 操作是 Spark SQL 中转换数据的实用工具。

# 摘要

在本章中，我们探讨了在 Spark SQL 中转换和分析数据的过程。我们学习了如何过滤和操作加载的数据，将转换后的数据保存为表，并执行 SQL 查询以提取有意义的见解。通过遵循提供的 Python 代码示例，你可以将这些技术应用到自己的数据集中，释放 Spark SQL 在数据分析和解探中的潜力。

在介绍那些主题之后，我们探讨了 Spark SQL 中窗口函数的强大功能，用于高级分析。我们讨论了窗口函数的语法和用法，使我们能够在定义的分区和窗口内执行复杂的计算和聚合。通过将窗口函数纳入 Spark SQL 查询，你可以获得有价值的见解，并更深入地理解你的数据，以便进行高级分析操作。

然后，我们讨论了一些在 Spark 中使用 UDF 的方法以及它们如何在 DataFrame 的多行和多列的复杂聚合中变得有用。

最后，我们介绍了一些在 Spark SQL 中使用 pivot 和 unpivot 的方法。

# 样题

**问题 1**：

以下哪个代码片段会在 Spark SQL 中创建一个视图，如果已经存在则替换现有视图？

1.  `dataframe.createOrReplaceTempView()`

1.  `dataframe.createTempView()`

1.  `dataFrame.createTableView()`

1.  `dataFrame.createOrReplaceTableView()`

1.  `dataDF.write.path(filePath)`

**问题 2**：

我们使用哪个函数将两个 DataFrame 合并在一起？

1.  `DataFrame.filter()`

1.  `DataFrame.distinct()`

1.  `DataFrame.intersect()`

1.  `DataFrame.join()`

1.  `DataFrame.count()`

## 答案

1.  A

1.  D

# 第四部分：Spark 应用程序

在这部分，我们将介绍 Spark 的结构化流，重点关注使用事件时间处理、水印、触发器和输出模式等概念进行实时数据处理。实际示例将说明如何使用结构化流构建和部署流应用程序。此外，我们还将深入研究 Spark ML，Spark 的机器学习库，探索监督和非监督技术，模型构建、评估以及跨各种算法的超参数调整。实际示例将展示 Spark ML 在现实世界机器学习任务中的应用，这对于当代数据科学至关重要。虽然这些内容不包括在 Spark 认证考试中，但理解这些概念对于现代数据工程至关重要。

本部分包含以下章节：

+   *第七章*，*Spark 中的结构化流*

+   *第八章*，*使用 Spark ML 进行机器学习*
