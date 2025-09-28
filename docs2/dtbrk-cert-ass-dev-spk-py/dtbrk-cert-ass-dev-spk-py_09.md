

# 模拟测试 1

# 问题

尝试回答这些问题以测试你对 Apache Spark 的了解：

**问题 1**：

以下哪个陈述没有准确地描述 Spark 驱动程序的功能？

1.  Spark 驱动程序作为运行 Spark 应用程序主方法的节点，用于协调应用程序。

1.  Spark 驱动程序可以水平扩展以提高整体处理吞吐量。

1.  Spark 驱动程序包含 SparkContext 对象。

1.  Spark 驱动程序负责使用集群模式下的不同工作节点调度数据的执行。

1.  最佳性能要求 Spark 驱动程序应尽可能靠近工作节点。

**问题 2**：

以下哪个陈述准确地描述了阶段？

1.  阶段内的任务可以由多台机器同时执行。

1.  作业中的各个阶段可以并发运行。

1.  阶段由一个或多个作业组成。

1.  阶段在提交之前暂时存储事务。

**问题 3**：

以下哪个陈述准确地描述了 Spark 的集群执行模式？

1.  集群模式在网关节点上运行执行器进程。

1.  集群模式涉及驱动程序托管在网关机器上。

1.  在集群模式下，Spark 驱动程序和集群管理器不是位于同一位置的。

1.  集群模式下的驱动程序位于工作节点上。

**问题 4**：

以下哪个陈述准确地描述了 Spark 的客户端执行模式？

1.  客户端模式在网关节点上运行执行器进程。

1.  在客户端模式下，驱动程序与执行器位于同一位置。

1.  在客户端模式下，Spark 驱动程序和集群管理器是位于同一位置的。

1.  在客户端模式下，驱动程序位于边缘节点上。

**问题 5**：

以下哪个陈述准确地描述了 Spark 的独立部署模式？

1.  独立模式为每个应用程序在每个工作节点上使用一个执行器。

1.  在独立模式下，驱动程序位于工作节点上。

1.  在独立模式下，集群不需要驱动程序。

1.  在独立模式下，驱动程序位于边缘节点上。

**问题 6**：

Spark 中的任务是什么？

1.  每个数据分区在任务中执行的工作单元是槽。

1.  任务是 Spark 中可以执行的第二小实体。

1.  具有广泛依赖关系的任务可以合并为单个任务。

1.  任务是 Spark 中分区执行的单个工作单元。

**问题 7**：

以下哪个是 Spark 执行层次结构中的最高级别？

1.  任务

1.  任务

1.  执行器

1.  阶段

**问题 8**：

如何在 Spark 的上下文中准确描述槽的概念？

1.  槽的创建和终止与执行器的工作负载相一致。

1.  Spark 通过在各个槽之间策略性地存储数据来增强 I/O 性能。

1.  每个槽始终被限制在单个核心上。

1.  槽允许任务并行运行。

**问题 9**：

Spark 中执行器的角色是什么？

1.  执行器的角色是将操作请求转换为 DAG。

1.  Spark 环境中只能有一个执行器。

1.  执行器以优化和分布式的方式处理分区

1.  执行器安排查询以执行

**问题 10**:

Shuffle 在 Spark 中的作用是什么？

1.  Shuffle 将变量广播到不同的分区

1.  使用 shuffle，数据会被写入磁盘

1.  Shuffle 命令在 Spark 中转换数据

1.  Shuffle 是一种窄转换

**问题 11**:

Actions 在 Spark 中的作用是什么？

1.  Actions 只从磁盘读取数据

1.  Actions 用于修改现有的 RDD

1.  Actions 触发任务的执行

1.  Actions 用于建立阶段边界

**问题 12**:

以下哪项是 Spark 中集群管理器的一项任务？

1.  在执行器失败的情况下，集群管理器将与驱动器协作以启动一个新的执行器

1.  集群管理器可以将分区合并以增加复杂数据处理的速度

1.  集群管理器收集查询的运行时统计信息

1.  集群管理器创建查询计划

**问题 13**:

以下哪项是 Spark 中自适应查询执行的一项任务？

1.  自适应查询执行可以合并分区以增加复杂数据处理的速度

1.  在执行器失败的情况下，自适应查询执行功能将与驱动器协作以启动一个新的执行器

1.  自适应查询执行创建查询计划

1.  自适应查询执行负责在 Spark 中生成多个执行器以执行任务

**问题 14**:

以下哪项操作被认为是转换？

1.  `df.select()`

1.  `df.show()`

1.  `df.head()`

1.  `df.count()`

**问题 15**:

Spark 中懒加载评估的一个特性是什么？

1.  Spark 只在执行期间失败作业，而不是在定义期间

1.  Spark 只在定义期间失败作业

1.  Spark 在收到转换操作时会执行

1.  Spark 在收到操作时会失败

**问题 16**:

以下关于 Spark 执行层次结构的哪个陈述是正确的？

1.  在 Spark 的执行层次结构中，任务位于作业之上

1.  在 Spark 的执行层次结构中，多个作业包含在一个阶段中

1.  在 Spark 的执行层次结构中，一个作业可能跨越多个阶段边界

1.  在 Spark 的执行层次结构中，slot 是最小的单元

**问题 17**:

以下哪项是 Spark 驱动的特征？

1.  当驱动器发送命令时，工作节点负责将 Spark 操作转换为 DAG

1.  Spark 驱动负责执行任务并将结果返回给执行器

1.  Spark 驱动可以通过添加更多机器来扩展，从而提高 Spark 任务的性能

1.  Spark 驱动以优化和分布式的方式处理分区

**问题 18**:

以下关于广播变量的哪个陈述是准确的？

1.  广播变量仅存在于驱动节点上

1.  广播变量只能用于适合内存的表

1.  广播变量不是不可变的，这意味着它们可以在集群之间共享

1.  广播变量不会在工作节点之间共享

**问题 19**:

以下哪个代码块返回了 DataFrame `df` 中 `employee_state` 和 `employee_salary` 列的唯一值？

1.  `Df.select('employee_state').join(df.select('employee_salary'),` `col('employee_state')==col('employee_salary'), 'left').show()`

1.  `df.select(col('employee_state'),` `col('employee_salary')).agg({'*': 'count'}).show()`

1.  `df.select('employee_state', 'employee_salary').distinct().show()`

1.  `df.select('employee_state').union(df.select('employee_salary')).distinct().show()`

**问题 20**:

以下哪个代码块从 `my_fle_path` 位置读取名为 `my_file.parquet` 的 Parquet 文件到 DataFrame `df`？

1.  `df =` `spark.mode("parquet").read("my_fle_path/my_file.parquet")`

1.  `df =` `spark.read.path("my_fle_path/my_file.parquet")`

1.  `df =` `spark.read().parquet("my_fle_path/my_file.parquet")`

1.  `df =` `spark.read.parquet("/my_fle_path/my_file.parquet")`

**问题 21**:

以下哪个代码块对 `salarydf` 和 `employeedf` DataFrame 的 `employeeSalaryID` 和 `employeeID` 列执行了内连接？

1.  `salarydf.join(employeedf, salarydf.employeeID ==` `employeedf.employeeSalaryID)`

1.  1.  `Salarydf.createOrReplaceTempView(salarydf)`

    1.  `employeedf.createOrReplaceTempView('employeedf')`

    1.  `spark.sql("SELECT * FROM salarydf CROSS JOIN employeedf ON` `employeeSalaryID ==employeeID")`

1.  1.  `salarydf`

    1.  `.``join(employeedf, col(employeeID)==col(employeeSalaryID))`

1.  1.  `Salarydf.createOrReplaceTempView(salarydf)`

    1.  `employeedf.createOrReplaceTempView('employeedf')`

    1.  `SELECT *` `FROM salarydf`

    1.  `INNER` `JOIN employeedf`

    1.  `ON salarydf.employeeSalaryID ==` `employeedf. employeeID`

**问题 22**:

以下哪个代码块按列 salary 降序排序返回 `df` DataFrame，并显示最后的缺失值？

1.  `df.sort(nulls_last("salary"))`

1.  `df.orderBy("salary").nulls_last()`

1.  `df.sort("salary", ascending=False)`

1.  `df.nulls_last("salary")`

**问题 23**:

以下代码块包含一个错误。该代码块应该返回一个 `df` DataFrame 的副本，其中列名 `state` 被更改为 `stateID`。找出错误。

代码块：

```py
df.withColumn("stateID", "state")
```

1.  方法中的参数 `"stateID"` 和 `"state"` 应该交换

1.  应该将 `withColumn` 方法替换为 `withColumnRenamed` 方法

1.  应该将 `withColumn` 方法替换为 `withColumnRenamed` 方法，并且需要重新排序方法的参数

1.  没有这样的方法可以更改列名

**问题 24**:

以下哪个代码块在 `salarydf` 和 `employeedf` DataFrame 之间使用 `employeeID` 和 `salaryEmployeeID` 列作为连接键执行了内连接？

1.  `salarydf.join(employeedf, "inner", salarydf.employeedf ==` `employeeID.salaryEmployeeID)`

1.  `salarydf.join(employeedf, employeeID ==` `salaryEmployeeID)`

1.  `salarydf.join(employeedf, salarydf.salaryEmployeeID ==` `employeedf.employeeID, "inner")`

1.  `salarydf.join(employeedf, salarydf.employeeID ==` `employeedf.salaryEmployeeID`, "inner")`

**问题 25**:

以下代码块应返回一个`df` DataFrame，其中`employeeID`列被转换为整数。请选择正确填充代码块空白的答案以完成此操作：

```py
df.__1__(__2__.__3__(__4__))
```

1.  1.  `select`

    1.  `col("employeeID")`

    1.  `as`

    1.  `IntegerType`

1.  1.  `select`

    1.  `col("employeeID")`

    1.  `as`

    1.  `Integer`

1.  1.  `cast`

    1.  `"``employeeID"`

    1.  `as`

    1.  `IntegerType()`

1.  1.  `select`

    1.  `col("employeeID")`

    1.  `cast`

    1.  `IntegerType()`

**问题 26**:

查找在将`employeedf`和`salarydf` DataFrames 按`employeeID`和`employeeSalaryID`列分别连接后，结果 DataFrame 中列 department 不为空的记录数。以下哪些代码块（按顺序）应执行以实现此目的？

1. `.filter(col("department").isNotNull())`

2. `.count()`

3. `employeedf.join(salarydf, employeedf.employeeID ==` `salarydf.employeeSalaryID)`

4. `employeedf.join(salarydf, employeedf.employeeID ==salarydf.` `employeeSalaryID`, how='inner')`

5. `.filter(col(department).isnotnull())`

6. `.sum(col(department))`

1.  3, 1, 6

1.  3, 1, 2

1.  4, 1, 2

1.  3, 5, 2

**问题 27**:

以下哪个代码块返回了`df` DataFrame 中列 state 值唯一的那些行？

1.  `df.dropDuplicates(subset=["state"]).show()`

1.  `df.distinct(subset=["state"]).show()`

1.  `df.drop_duplicates(subset=["state"]).show()`

1.  `df.unique("state").show()`

**问题 28**:

以下代码块包含一个错误。该代码块应返回一个包含额外列`squared_number`的`df` DataFrame 副本，该列是列 number 的平方。请找出错误。

代码块：

```py
df.withColumnRenamed(col("number"), pow(col("number"), 0.2).alias("squared_number"))
```

1.  `withColumnRenamed`方法的参数需要重新排序

1.  应将`withColumnRenamed`方法替换为`withColumn`方法

1.  应将`withColumnRenamed`方法替换为`select`方法，并将`0.2`替换为`2`

1.  应将参数`0.2`替换为`2`

**问题 29**:

以下哪个代码块返回了一个新的 DataFrame，其中列 salary 被重命名为`new_salary`，employee 被重命名为`new_employee`在`df` DataFrame 中？

1.  `df.withColumnRenamed(salary,` `new_salary).withColumnRenamed(employee, new_employee)`

1.  `df.withColumnRenamed("salary", "new_salary")`

1.  `df.withColumnRenamed("employee", "new_employee")`

1.  `df.withColumn("salary", "``new_salary").withColumn("employee", "new_employee")`

1.  `df.withColumnRenamed("salary", "``new_salary").withColumnRenamed("employee", "new_employee")`

**问题 30**:

以下哪个代码块返回了一个`df` DataFrame 的副本，其中列 salary 已被重命名为`employeeSalary`？

1.  `df.withColumn(["salary", "employeeSalary"])`

1.  `df.withColumnRenamed("salary").alias("employeeSalary ")`

1.  `df.withColumnRenamed("salary", "``employeeSalary ")`

1.  `df.withColumn("salary", "``employeeSalary ")`

**问题 31**:

以下代码块包含一个错误。代码块应该将 `df` DataFrame 保存到 `my_file_path` 路径作为 Parquet 文件，并追加到任何现有的 Parquet 文件。找出错误。

```py
df.format("parquet").option("mode", "append").save(my_file_path)
```

1.  代码没有保存到正确的路径

1.  应该交换 `save()` 和 `format` 函数

1.  代码块缺少对 `DataFrameWriter` 的引用

1.  应该覆盖 `option` 模式以正确写入文件

**问题 32**:

我们如何将 `df` DataFrame 从 12 个分区减少到 6 个分区？

1.  `df.repartition(12)`

1.  `df.coalesce(6).shuffle()`

1.  `df.coalesce(6, shuffle=True)`

1.  `df.repartition(6)`

**问题 33**:

以下哪个代码块返回一个 DataFrame，其中时间戳列被转换为名为 `record_timestamp` 的新列，格式为日、月和年？

1.  `df.withColumn("record_timestamp", ` `from_unixtime(unix_timestamp(col("timestamp")), "dd-MM-yyyy"))`

1.  `df.withColumnRenamed("record_timestamp", ` `from_unixtime(unix_timestamp(col("timestamp")), "dd-MM-yyyy"))`

1.  `df.select ("record_timestamp", ` `from_unixtime(unix_timestamp(col("timestamp")), "dd-MM-yyyy"))`

1.  `df.withColumn("record_timestamp", ` `from_unixtime(unix_timestamp(col("timestamp")), "MM-dd-yyyy"))`

**问题 34**:

以下哪个代码块通过将 DataFrame `salaryDf` 的行追加到 DataFrame `employeeDf` 的行来创建一个新的 DataFrame，而不考虑两个 DataFrame 都有不同的列名？

1.  `salaryDf.join(employeeDf)`

1.  `salaryDf.union(employeeDf)`

1.  `salaryDf.concat(employeeDf)`

1.  `salaryDf.unionAll(employeeDf)`

**问题 35**:

以下代码块包含一个错误。代码块应该计算每个部门 `employee_salary` 列中所有工资的总和。找出错误。

```py
df.agg("department").sum("employee_salary")
```

1.  应该使用 `avg(col("value"))` 而不是 `avg("value")`

1.  所有列名都应该用 `col()` 运算符包裹

1.  `"storeId"` 和 “`value"` 应该交换

1.  `Agg` 应该替换为 `groupBy`

**问题 36**:

以下代码块包含一个错误。代码块旨在对 `salarydf` 和 `employeedf` DataFrame 的 `employeeSalaryID` 和 `employeeID` 列分别执行交叉连接。找出错误。

```py
employeedf.join(salarydf, [salarydf.employeeSalaryID, employeedf.employeeID], "cross")
```

1.  参数中的连接类型 `"cross"` 需要替换为 `crossJoin`

1.  `salarydf.employeeSalaryID, employeedf.employeeID` 应替换为 `salarydf.employeeSalaryID ==` `employeedf.employeeID`

1.  应该删除 `"cross"` 参数，因为 `"cross"` 是默认的连接类型

1.  应从调用中删除 `"cross"` 参数，并用 `crossJoin` 替换 `join`

**问题 37**:

以下代码块包含一个错误。代码块应该显示 `df` DataFrame 的模式。找出错误。

```py
df.rdd.printSchema()
```

1.  在 Spark 中，我们无法打印 DataFrame 的模式

1.  `printSchema` 不能通过 `df.rdd` 调用，而应该直接从 `df` 调用

1.  Spark 中没有名为 `printSchema()` 的方法

1.  应该使用 `print_schema()` 方法而不是 `printSchema()`

**问题 38**:

以下代码块应该将 `df` DataFrame 写入到 `filePath` 路径的 Parquet 文件中，替换任何现有文件。选择正确填充代码块空白处的答案以完成此操作：

```py
df.__1__.format("parquet").__2__(__3__).__4__(filePath)
```

1.  1.  `save`

    1.  `mode`

    1.  `"``ignore"`

    1.  `path`

1.  1.  `store`

    1.  `with`

    1.  `"``replace"`

    1.  `path`

1.  1.  `write`

    1.  `mode`

    1.  `"``overwrite"`

    1.  `save`

1.  1.  `save`

    1.  `mode`

    1.  `"``overwrite"`

    1.  `path`

**问题 39**:

以下代码块包含一个错误。代码块本应按薪资降序对 `df` DataFrame 进行排序。然后，它应该根据奖金列进行排序，将 `nulls` 放在最后。找出错误。

```py
df.orderBy ('salary', asc_nulls_first(col('bonus')))
transactionsDf.orderBy('value', asc_nulls_first(col('predError')))
```

1.  应该以降序对 `salary` 列进行排序。此外，它应该被包裹在 `col()` 操作符中

1.  应该用 `col()` 操作符将 `salary` 列包裹起来

1.  应该以降序对 `bonus` 列进行排序，将 `nulls` 放在最后

1.  应该使用 `desc_nulls_first()` 对 `bonus` 列进行排序

**问题 40**:

以下代码块包含一个错误。代码块应该使用 `square_root_method` Python 方法找到 `df` DataFrame 中 `salary` 列的平方根，并在新列 `sqrt_salary` 中返回它。找出错误。

```py
square_root_method_udf = udf(square_root_method)
df.withColumn("sqrt_salary", square_root_method("salary"))
```

1.  `square_root_method` 没有指定返回类型

1.  在第二行代码中，Spark 需要调用 `squre_root_method_udf` 而不是 `square_root_method`

1.  `udf` 未在 Spark 中注册

1.  需要添加一个新列

**问题 41**:

以下代码块包含一个错误。代码块应该返回将 `employeeID` 重命名为 `employeeIdColumn` 的 `df` DataFrame。找出错误。

```py
df.withColumn("employeeIdColumn", "employeeID")
```

1.  应该使用 `withColumnRenamed` 方法而不是 `withColumn`

1.  应该使用 `withColumnRenamed` 方法而不是 `withColumn`，并且参数 `"employeeIdColumn"` 应该与参数 `"employeeID"` 交换

1.  参数 `"employeeIdColumn"` 和 `"employeeID"` 应该交换

1.  应该将 `withColumn` 操作符替换为 `withColumnRenamed` 操作符

**问题 42**:

以下哪个代码块会返回一个新的 DataFrame，其列与 DataFrame `df` 相同，除了 `salary` 列？

1.  `df.drop("salary")`

1.  `df.drop(col(salary))`

1.  `df.drop(salary)`

1.  `df.delete("salary")`

**问题 43**:

以下哪个代码块返回一个 DataFrame，显示 `df` DataFrame 中 `salary` 列的平均值，按 `department` 列分组？

1.  `df.groupBy("department").agg(avg("salary"))`

1.  `df.groupBy(col(department).avg())`

1.  `df.groupBy("department").avg(col("salary"))`

1.  `df.groupBy("department").agg(average("salary"))`

**问题 44**:

以下哪个代码块创建了一个 DataFrame，显示基于部门和国家/地区列，年龄大于 35 的 `salaryDf` DataFrame 中 `salary` 列的平均值？

1.  `salaryDf.filter(col("age") >` `35)`

1.  `.``filter(col("employeeID")`

1.  `.``filter(col("employeeID").isNotNull())`

1.  `.``groupBy("department")`

1.  `.``groupBy("department", "state")`

1.  `.``agg(avg("salary").alias("mean_salary"))`

1.  `.``agg(average("salary").alias("mean_salary"))`

    1.  1,2,5,6

    1.  1,3,5,6

    1.  1,3,6,7

    1.  1,2,4,6

**问题 45**:

以下代码块包含一个错误。该代码块需要缓存`df` DataFrame，以便此 DataFrame 具有容错性。找出错误。

```py
df.persist(StorageLevel.MEMORY_AND_DISK_3)
```

1.  `persist()`不是 DataFrame API 的一个函数

1.  应将`df.write()`与`df.persist`结合使用以正确写入 DataFrame

1.  存储级别不正确，应为`MEMORY_AND_DISK_2`

1.  应使用`df.cache()`而不是`df.persist()`

**问题 46**:

以下哪个代码块在不重复的情况下连接了`salaryDf`和`employeeDf` DataFrame 的行（假设两个 DataFrame 的列相似）？

1.  `salaryDf.concat(employeeDf).unique()`

1.  `spark.union(salaryDf, employeeDf).distinct()`

1.  `salaryDf.union(employeeDf).unique()`

1.  `salaryDf.union(employeeDf).distinct()`

**问题 47**:

以下哪个代码块从`filePath`读取一个完整的 CSV 文件文件夹，包含列标题？

1.  `spark.option("header",True).csv(filePath)`

1.  `spark.read.load(filePath)`

1.  `spark.read().option("header",True).load(filePath)`

1.  `spark.read.format("csv").option("header",True).load(filePath)`

**问题 48**:

以下代码块包含一个错误。`df` DataFrame 包含列`[`employeeID`, `salary`, 和 `department`]。该代码块应返回一个仅包含`employeeID`和`salary`列的 DataFrame。找出错误。

```py
df.select(col(department))
```

1.  应在`select`参数中指定`df` DataFrame 的所有列名

1.  应将`select`运算符替换为`drop`运算符，并列出`df` DataFrame 中的所有列名作为列表

1.  应将`select`运算符替换为`drop`运算符

1.  列名`department`应列出为`col("department")`

**问题 49**:

以下代码块包含一个错误。该代码块应将 DataFrame `df`作为 Parquet 文件写入到`filePath`位置，在按`department`列分区后。找出错误。

```py
df.write.partition("department").parquet()
```

1.  应使用`partitionBy()`方法而不是`partition()`方法。

1.  应使用`partitionBy()`方法而不是`partition()`，并将`filePath`添加到`parquet`方法中

1.  在写入方法之前应调用`partition()`方法，并将`filePath`添加到`parquet`方法中

1.  应将`"department"`列用`col()`运算符包裹

**问题 50**:

以下哪个代码块从内存和磁盘中移除了缓存的`df` DataFrame？

1.  `df.unpersist()`

1.  `drop df`

1.  `df.clearCache()`

1.  `df.persist()`

**问题 51**:

以下代码块应该返回一个包含额外列：`test_column`，其值为`19`的`df` DataFrame 的副本。请选择正确填充代码块空白处的答案以完成此操作：

```py
df.__1__(__2__, __3__)
```

1.  1.  `withColumn`

    1.  `'``test_column'`

    1.  `19`

1.  1.  `withColumnRenamed`

    1.  `test_column`

    1.  `lit(19)`

1.  1.  `withColumn`

    1.  `'test_column'`

    1.  `lit(19)`

1.  1.  `withColumnRenamed`

    1.  `test_column`

    1.  `19`

**问题 52**:

以下代码块应该返回一个包含 `employeeId`、`salary`、`bonus` 和 `department` 列的 DataFrame，来自 `transactionsDf` DataFrame。选择正确填充空白的答案以完成此操作：

```py
df.__1__(__2__)
```

1.  1.  `drop`

    1.  `"employeeId", "salary", "bonus", "department"`

1.  1.  `filter`

    1.  `"employeeId, salary, bonus, department"`

1.  1.  `select`

    1.  `["employeeId", "salary", "bonus", "department"]`

1.  1.  `select`

    1.  `col(["employeeId", "salary", "bonus", "department"])`

**问题 53**:

以下哪个代码块返回了一个 DataFrame，其中 `salary` 列在 `df` DataFrame 中被转换为字符串？

1.  `df.withColumn("salary", `castString("salary", "string"))`

1.  `df.withColumn("salary", col("salary").cast("string"))`

1.  `df.select(cast("salary", "string"))`

1.  `df.withColumn("salary", col("salary").castString("string"))`

**问题 54**:

以下代码块包含错误。该代码块应该结合来自 `salaryDf` 和 `employeeDf` DataFrame 的数据，显示 `salaryDf` DataFrame 中所有与 `employeeDf` DataFrame 中 `employeeSalaryID` 列的值匹配的行。找出错误。

```py
employeeDf.join(salaryDf, employeeDf.employeeID==employeeSalaryID)
```

1.  `join` 语句缺少右侧的 DataFrame，其中列名为 `employeeSalaryID`

1.  应该使用 `union` 方法而不是 `join`

1.  应该使用 `innerJoin` 而不是 `join`

1.  `salaryDf` 应该替换 `employeeDf`

**问题 55**:

以下哪个代码块读取存储在 `my_file_path` 的 JSON 文件作为 DataFrame？

1.  `spark.read.json(my_file_path)`

1.  `spark.read(my_file_path, source="json")`

1.  `spark.read.path(my_file_path)`

1.  `spark.read().json(my_file_path)`

**问题 56**:

以下代码块包含错误。该代码块应该返回一个新的 DataFrame，通过过滤 `df` DataFrame 中 `salary` 列大于 2000 的行。找出错误。

```py
df.where("col(salary) >= 2000")
```

1.  应该使用 `filter()` 而不是 `where()`

1.  `where` 方法的参数应该是 `"col(salary) >` `2000"`

1.  应该使用 `>` 操作符而不是 `>=`

1.  `where` 方法的参数应该是 `"salary >` `2000"`

**问题 57**:

以下哪个代码块返回了一个 DataFrame，其中从 `df` DataFrame 中删除了 `salary` 和 `state` 列？

1.  `df.withColumn ("salary", "state")`

1.  `df.drop(["salary", "state"])`

1.  `df.drop("salary", "state")`

1.  `df.withColumnRenamed ("salary", "state")`

**问题 58**:

以下哪个代码块返回了一个包含 `df` DataFrame 中每个部门计数的两列 DataFrame？

1.  `df.count("department").distinct()`

1.  `df.count("department")`

1.  `df.groupBy("department").count()`

1.  `df.groupBy("department").agg(count("department"))`

**问题 59**:

以下哪个代码块打印了 DataFrame 的模式，并包含列名和类型？

1.  `print(df.columns)`

1.  `df.printSchema()`

1.  `df.rdd.printSchema()`

1.  `df.print_schema()`

**问题 60**:

以下哪个代码块创建了一个新的 DataFrame，包含三个列：`department`（部门），`age`（年龄）和`max_salary`（最高薪水），并且对于每个部门以及每个年龄组的每个员工都有最高的薪水？

1.  `df.max(salary)`

1.  `df.groupBy(["department", "age"]).agg(max("salary").alias("max_salary"))`

1.  `df.agg(max(salary).alias(max_salary')`

1.  `df.groupby(department).agg(max(salary).alias(max_salary)`

## 答案

1.  B

1.  A

1.  D

1.  D

1.  A

1.  D

1.  A

1.  D

1.  C

1.  B

1.  C

1.  A

1.  A

1.  A

1.  A

1.  C

1.  B

1.  B

1.  D

1.  D

1.  D

1.  C

1.  C

1.  D

1.  D

1.  C

1.  A

1.  C

1.  E

1.  C

1.  C

1.  D

1.  A

1.  B

1.  D

1.  B

1.  B

1.  C

1.  A

1.  B

1.  B

1.  A

1.  A

1.  A

1.  C

1.  D

1.  D

1.  C

1.  B

1.  A

1.  C

1.  C

1.  B

1.  A

1.  A

1.  D

1.  C

1.  C

1.  B

1.  B
