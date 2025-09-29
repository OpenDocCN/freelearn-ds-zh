# Hadoop 中的数据建模

到目前为止，我们已经学习了如何创建 Hadoop 集群以及如何将数据加载到其中。在上一章中，我们学习了各种数据摄取工具和技术。正如我们所知，市场上有很多开源工具，但并没有一个能够应对所有用例的万能工具。每个数据摄取工具都有其独特的功能；在典型用例中，它们可以证明是非常高效和有用的。例如，Sqoop 在用于从关系型数据库管理系统（RDBMS）导入和导出 Hadoop 数据时更为有用。

在本章中，我们将学习如何在 Hadoop 集群中存储和建模数据。与数据摄取工具类似，有各种数据存储可供选择。这些数据存储支持不同的数据模型——即列式数据存储、键值对等；并且支持各种文件格式，如 ORC、Parquet 和 AVRO 等。目前，有一些非常流行的数据存储，在生产环境中被广泛使用，例如 Hive、HBase、Cassandra 等。我们将更深入地了解以下两个数据存储和数据建模技术：

+   Apache Hive

+   Apache HBase

首先，我们将从基本概念开始，然后我们将学习如何应用现代数据建模技术以实现更快的数据访问。简而言之，本章将涵盖以下主题：

+   Apache Hive 和关系型数据库管理系统（RDBMS）

+   支持的数据类型

+   Hive 架构及其工作原理

# Apache Hive

Hive 是 Hadoop 中的一个数据处理工具。正如我们在上一章所学的，数据摄取工具在 Hadoop 中加载数据并生成 HDFS 文件；我们需要根据业务需求查询这些数据。我们可以使用 MapReduce 编程来访问数据。但是，使用 MapReduce 访问数据非常慢。为了访问 HDFS 文件中的一小部分数据，我们必须编写单独的 mapper、reducer 和 driver 代码。因此，为了避免这种复杂性，Apache 引入了 Hive。Hive 支持类似 SQL 的接口，它可以帮助使用 SQL 命令访问相同的 HDFS 文件行。Hive 最初由 Facebook 开发，但后来被 Apache 接管。

# Apache Hive 和 RDBMS

我提到 Hive 提供了一个类似 SQL 的接口。考虑到这一点，随之而来的问题是：*Hive 是否与 Hadoop 上的 RDBMS 相同？* 答案是*不是*。Hive 不是一个数据库。Hive 不存储任何数据。Hive 将表信息作为元数据的一部分存储，这被称为模式，并指向 HDFS 上的文件。Hive 使用一个称为**HiveQL**（**HQL**）的类似 SQL 的接口来访问存储在 HDFS 文件上的数据。Hive 支持 SQL 命令来访问和修改 HDFS 中的数据。Hive 不是一个 OLTP 工具。它不提供任何行级别的插入、更新或删除。当前版本的 Hive（版本 0.14）支持具有完整 ACID 属性的插入、更新和删除，但这个功能效率不高。此外，这个功能不支持所有文件格式。例如，更新只支持 ORC 文件格式。基本上，Hive 是为批量处理设计的，不支持像 RDBMS 那样的事务处理。因此，Hive 更适合数据仓库应用，提供数据汇总、查询和分析。内部，Hive SQL 查询由其编译器转换为 MapReduce。用户无需担心编写任何复杂的 mapper 和 reducer 代码。Hive 只支持查询结构化数据。使用 Hive SQL 访问非结构化数据非常复杂。你可能需要为该功能编写自己的自定义函数。Hive 支持各种文件格式，如文本文件、序列文件、ORC 和 Parquet，这些格式提供了显著的数据压缩。

# 支持的数据类型

Hive 版本 0.14 支持以下数据类型：

| **数据类型组** | **数据类型** | **格式** |
| --- | --- | --- |
| 字符串 | `STRING` | `column_name STRING` |
| `VARCHAR` | `column_name VARCHAR(max_length)` |
| `CHAR` | `column_name CHAR(length)` |
| 数值 | `TINYINT` | `column_name TINYINT` |
| `SMALLINT` | `column_name SMALLINT` |
| `INT` | `column_name INT` |
| `BIGINT` | `column_name BIGINT` |
| `FLOAT` | `column_name FLOAT` |
| `DOUBLE` | `column_name DOUBLE` |
| `DECIMAL` | `column_name DECIMAL[(precision[,scale])]` |
| 日期/时间类型 | `TIMESTAMP` | `column_name TIMESTAMP` |
| `DATE` | `column_name DATE` |
| `INTERVAL` | `column_name INTERVAL year to month` |
| 杂项类型 | `BOOLEAN` | `column_name BOOLEAN` |
| `BINARY` | `column_name BINARY` |
| 复杂数据类型 | `ARRAY` | `column_name ARRAY < type >` |
| `MAPS` | `column_name MAP < primitive_type, type >` |
| `STRUCT` | `column_name STRUCT < name : type [COMMENT 'comment_string'] >` |
| `UNION` | `column_name UNIONTYPE <int, double, array, string>` |

# Hive 的工作原理

Hive 数据库由表组成，这些表由分区构成。数据可以通过简单的查询语言访问，并且 Hive 支持数据的覆盖或追加。在特定数据库中，表中的数据是序列化的，每个表都有一个对应的 HDFS 目录。每个表可以进一步细分为分区，这些分区决定了数据如何在表目录的子目录中分布。分区内的数据可以进一步细分为桶。

# Hive 架构

以下是对 Hive 架构的表示：

![](img/7dcc7f8c-71f0-4b1d-b245-8e0e317aa65a.png)

上述图示显示 Hive 架构分为三个部分——即客户端、服务和元存储。Hive SQL 的执行方式如下：

+   **Hive SQL 查询**：可以使用以下方式之一将 Hive 查询提交给 Hive 服务器：WebUI、JDBC/ODBC 应用程序和 Hive CLI。对于基于 thrift 的应用程序，它将提供一个 thrift 客户端用于通信。

+   **查询执行**：一旦 Hive 服务器接收到查询，它就会被编译，转换为优化查询计划以提高性能，并转换为 MapReduce 作业。在这个过程中，Hive 服务器与元存储交互以获取查询元数据。

+   **作业执行**：MapReduce 作业在 Hadoop 集群上执行。

# Hive 数据模型管理

Hive 以以下四种方式处理数据：

+   Hive 表

+   Hive 表分区

+   Hive 分区桶

+   Hive 视图

我们将在接下来的几节中详细探讨每一个。

# Hive 表

Hive 表与任何 RDBMS 表非常相似。表被分为行和列。每个列（字段）都使用适当的名称和数据类型进行定义。我们已经在*支持的数据类型*部分中看到了 Hive 中所有可用的数据类型。Hive 表分为两种类型：

+   管理表

+   外部表

我们将在接下来的几节中学习这两种类型。

# 管理表

以下是一个定义 Hive 管理表的示例命令：

```py
Create Table < managed_table_name>  
   Column1 <data type>, 
   Column2 <data type>, 
   Column3 <data type> 
Row format delimited Fields Terminated by "t"; 
```

当执行前面的查询时，Hive 会创建表，并且元数据会相应地更新到元存储中。但是表是空的。因此，可以通过执行以下命令将数据加载到这个表中：

```py
Load data inpath <hdfs_folder_name> into table <managed_table_name>; 
```

执行前面的命令后，数据将从`<hdfs_folder_name>`移动到 Hive 表的默认位置`/user/hive/warehouse/<managed_table_name>`。这个默认文件夹`/user/hive/warehouse`在`hive-site.xml`中定义，可以被更改为任何文件夹。现在，如果我们决定删除表，可以通过以下命令执行：

```py
Drop table <managed_table_name>; 
```

`/user/hive/warehouse/<managed_table_name>`文件夹将被删除，并且存储在元存储中的元数据将被删除。

# 外部表

以下是一个定义 Hive 外部表的示例命令：

```py
Create Table < external_table_name>  
   Column1 <data type>, 
   Column2 <data type>, 
   Column3 <data type> 
Row format delimited Fields Terminated by "t" 
Location <hdfs_folder_name>; 
```

当执行前面的查询时，Hive 将创建表，并在 metastore 中相应地更新元数据。但是，表仍然是空的。因此，可以通过执行以下命令将数据加载到该表中：

```py
Load data inpath <hdfs_folder_name> into table <external_table_name>; 
```

此命令不会将任何文件移动到任何文件夹，而是创建一个指向文件夹位置的指针，并在 metastore 的元数据中进行更新。文件将保持在查询相同的位置（《hdfs_folder_name》）。现在，如果我们决定删除表，我们可以通过以下命令执行：

```py
Drop table <managed_table_name>;  
```

文件夹`/user/hive/warehouse/<managed_table_name>`不会被删除，只会删除存储在 metastore 中的元数据。文件将保持在相同的位置——`<hdfs_folder_name>`。

# Hive 表分区

表分区意味着根据分区键的值将表划分为不同的部分。分区键可以是任何列，例如日期、部门、国家等。由于数据存储在部分中，查询响应时间会更快。分区不是扫描整个表，而是在主表文件夹内创建子文件夹。Hive 将根据查询的`WHERE`子句仅扫描表的具体部分或部分。Hive 表分区类似于任何 RDBMS 表分区。其目的也是相同的。随着我们不断向表中插入数据，表的数据量会越来越大。假设我们创建了一个如下所示的`ORDERS`表：

```py
hive> create database if not exists ORDERS; 
OK 
Time taken: 0.036 seconds 

hive> use orders; 
OK 
Time taken: 0.262 seconds 

hive> CREATE TABLE if not exists ORDEERS_DATA 
    > (Ord_id INT, 
    > Ord_month INT, 
    > Ord_customer_id INT, 
    > Ord_city  STRING, 
    > Ord_zip   STRING, 
    > ORD_amt   FLOAT 
    > ) 
    > ROW FORMAT DELIMITED 
    > FIELDS TERMINATED BY  ',' 
    > ; 
OK 
Time taken: 0.426 seconds 
hive> 
```

我们将按照以下方式加载以下示例文件`ORDERS_DATA`表：

```py
101,1,100,'Los Angeles','90001',1200 
102,2,200,'Los Angeles','90002',1800 
103,3,300,'Austin','78701',6500 
104,4,400,'Phoenix','85001',7800 
105,5,500,'Beverly Hills','90209',7822 
106,6,600,'Gaylord','49734',8900 
107,7,700,'Los Angeles','90001',7002 
108,8,800,'Los Angeles','90002',8088 
109,9,900,'Reno','89501',6700 
110,10,1000,'Los Angeles','90001',8500 
111,10,1000,'Logan','84321',2300 
112,10,1000,'Fremont','94539',9500 
113,10,1000,'Omaha','96245',7500 
114,11,2000,'New York','10001',6700 
115,12,3000,'Los Angeles','90003',1000 
```

然后将`orders.txt`加载到`/tmp` HDFS 文件夹中：

```py
[root@sandbox order_data]# hadoop fs -put /root/order_data/orders.txt /tmp 

[root@sandbox order_data]# hadoop fs -ls /tmp 
Found 3 items 
-rw-r--r--   1 root      hdfs        530 2017-09-02 18:06 /tmp/orders.txt 
```

按照以下方式加载`ORDERS_DATA`表：

```py
hive> load data inpath '/tmp/orders.txt' into table ORDERS_DATA; 
Loading data to table orders.orders_data 
Table orders.orders_data stats: [numFiles=1, numRows=0, totalSize=530, rawDataSize=0] 
OK 
Time taken: 0.913 seconds 

hive> select * from ORDERS_DATA; 
OK 
101      1     100   'Los Angeles'     '90001'     1200.0 
102      2     200   'Los Angeles'     '90002'     1800.0 
103      3     300   'Austin'    '78701'     6500.0 
104      4     400   'Phoenix'   '85001'     7800.0 
105      5     500   'Beverly Hills'   '90209'     7822.0 
106      6     600   'Gaylord'   '49734'     8900.0 
107      7     700   'Los Angeles'     '90001'     7002.0 
108      8     800   'Los Angeles'     '90002'     8088.0 
109      9     900   'Reno'      '89501'     6700.0 
110      10    1000  'Los Angeles'     '90001'     8500.0 
111      10    1000  'Logan'     '84321'     2300.0 
112      10    1000  'Fremont'   '94539'     9500.0 
113      10    1000  'Omaha'     '96245'     7500.0 
114      11    2000  'New York'  '10001'     6700.0 
115      12    3000  'Los Angeles'     '90003'     1000.0 
Time taken: 0.331 seconds, Fetched: 15 row(s) 
```

假设我们想在`ORDERS_DATA`表中插入城市数据。每个城市的订单数据大小为 1 TB。因此，`ORDERS_DATA`表的总数据大小将为 15 TB（表中共有 15 个城市）。现在，如果我们编写以下查询以获取在`洛杉矶`预订的所有订单：

```py
hive>  select * from ORDERS where Ord_city = 'Los Angeles' ; 

```

查询将运行得非常慢，因为它必须扫描整个表。显然的想法是，我们可以为每个城市创建 10 个不同的`orders`表，并将`orders`数据存储在`ORDERS_DATA`表的相应城市中。但不是这样，我们可以按照以下方式对`ORDERS_PART`表进行分区：

```py
hive> use orders; 

hive> CREATE TABLE orders_part 
    > (Ord_id INT, 
    > Ord_month INT, 
    > Ord_customer_id INT, 
    > Ord_zip   STRING, 
    > ORD_amt   FLOAT 
    > ) 
    > PARTITIONED BY  (Ord_city INT) 
    > ROW FORMAT DELIMITED 
    > FIELDS TERMINATED BY  ',' 
    > ; 
OK 
Time taken: 0.305 seconds 
hive> 
```

现在，Hive 根据列或分区键将表组织成分区，以便将相似类型的数据分组在一起。假设我们为每个城市有 10 个`orders`文件，即`Orders1.txt`到`Orders10.txt`。以下示例显示了如何将每个月度文件加载到相应的分区中：

```py
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='Los Angeles'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='Austin'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='Phoenix'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='Beverly Hills'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='Gaylord'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city=Reno'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='Fremont'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='Omaha'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='New York'); 
load data inpath '/tmp/orders.txt' into table orders_part partition(Ord_city='Logan'); 

[root@sandbox order_data]# hadoop fs -ls /apps/hive/warehouse/orders.db/orders_part 
Found 10 items 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:32 /apps/hive/warehouse/orders.db/orders_part/ord_city=Austin 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:32 /apps/hive/warehouse/orders.db/orders_part/ord_city=Beverly Hills 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:32 /apps/hive/warehouse/orders.db/orders_part/ord_city=Fremont 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:32 /apps/hive/warehouse/orders.db/orders_part/ord_city=Gaylord 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:33 /apps/hive/warehouse/orders.db/orders_part/ord_city=Logan 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:32 /apps/hive/warehouse/orders.db/orders_part/ord_city=Los Angeles 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:32 /apps/hive/warehouse/orders.db/orders_part/ord_city=New York 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:32 /apps/hive/warehouse/orders.db/orders_part/ord_city=Omaha 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:32 /apps/hive/warehouse/orders.db/orders_part/ord_city=Phoenix 
drwxrwxrwx   - root hdfs          0 2017-09-02 18:33 /apps/hive/warehouse/orders.db/orders_part/ord_city=Reno 
[root@sandbox order_data]  
```

数据分区可以显著提高查询性能，因为数据已经根据列值分离到不同的文件中，这可以减少映射器的数量，并大大减少 MapReduce 作业中数据的洗牌和排序量。

# Hive 静态分区和动态分区

如果您想在 Hive 中使用静态分区，应设置以下属性：

```py
set hive.mapred.mode = strict;  
```

在前面的例子中，我们已经看到我们必须将每个月的订单文件分别插入到每个静态分区中。与动态分区相比，静态分区在加载数据时可以节省时间。我们必须单独向表中添加一个分区并将文件移动到表的分区中。如果我们有很多分区，编写一个查询来加载数据到每个分区可能会变得繁琐。我们可以通过动态分区来克服这一点。在动态分区中，我们可以使用单个 SQL 语句将数据插入到分区表中，但仍可以加载数据到每个分区。与静态分区相比，动态分区在加载数据时花费更多时间。当你有一个表中有大量数据存储时，动态分区是合适的。如果你想要对多个列进行分区，但你不知道它们有多少列，那么动态分区也是合适的。以下是你应该允许的 hive 动态分区属性：

```py
SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;  
```

以下是一个动态分区的示例。假设我们想要从`ORDERS_PART`表加载数据到一个名为`ORDERS_NEW`的新表中：

```py
hive> use orders; 
OK 
Time taken: 1.595 seconds 
hive> drop table orders_New; 
OK 
Time taken: 0.05 seconds 
hive> CREATE TABLE orders_New 
    > (Ord_id INT, 
    > Ord_month INT, 
    > Ord_customer_id INT, 
    > Ord_city  STRING, 
    > Ord_zip   STRING, 
    > ORD_amt   FLOAT 
    > ) 
    > ) 
    > PARTITIONED BY  (Ord_city STRING) 
    > ROW FORMAT DELIMITED 
    > FIELDS TERMINATED BY  ',' 
    > ; 
OK 
Time taken: 0.458 seconds 
hive> 
```

从`ORDERS_PART`表加载数据到`ORDER_NEW`表。在这里，Hive 将动态地加载数据到`ORDERS_NEW`表的全部分区：

```py
hive> SET hive.exec.dynamic.partition = true; 
hive> SET hive.exec.dynamic.partition.mode = nonstrict; 
hive>  
    > insert into table orders_new  partition(Ord_city) select * from orders_part; 
Query ID = root_20170902184354_2d409a56-7bfc-416e-913a-2323ea3b339a 
Total jobs = 1 
Launching Job 1 out of 1 
Status: Running (Executing on YARN cluster with App id application_1504299625945_0013) 

-------------------------------------------------------------------------------- 
        VERTICES      STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED 
-------------------------------------------------------------------------------- 
Map 1 ..........   SUCCEEDED      1          1        0        0       0       0 
-------------------------------------------------------------------------------- 
VERTICES: 01/01  [==========================>>] 100%  ELAPSED TIME: 3.66 s      
-------------------------------------------------------------------------------- 
Loading data to table orders.orders_new partition (ord_city=null) 
    Time taken to load dynamic partitions: 2.69 seconds 
   Loading partition {ord_city=Logan} 
   Loading partition {ord_city=Los Angeles} 
   Loading partition {ord_city=Beverly Hills} 
   Loading partition {ord_city=Reno} 
   Loading partition {ord_city=Fremont} 
   Loading partition {ord_city=Gaylord} 
   Loading partition {ord_city=Omaha} 
   Loading partition {ord_city=Austin} 
   Loading partition {ord_city=New York} 
   Loading partition {ord_city=Phoenix} 
    Time taken for adding to write entity : 3 
Partition orders.orders_new{ord_city=Austin} stats: [numFiles=1, numRows=1, totalSize=13, rawDataSize=12] 
Partition orders.orders_new{ord_city=Beverly Hills} stats: [numFiles=1, numRows=1, totalSize=13, rawDataSize=12] 
Partition orders.orders_new{ord_city=Fremont} stats: [numFiles=1, numRows=1, totalSize=15, rawDataSize=14] 
Partition orders.orders_new{ord_city=Gaylord} stats: [numFiles=1, numRows=1, totalSize=13, rawDataSize=12] 
Partition orders.orders_new{ord_city=Logan} stats: [numFiles=1, numRows=1, totalSize=15, rawDataSize=14] 
Partition orders.orders_new{ord_city=Los Angeles} stats: [numFiles=1, numRows=6, totalSize=82, rawDataSize=76] 
Partition orders.orders_new{ord_city=New York} stats: [numFiles=1, numRows=1, totalSize=15, rawDataSize=14] 
Partition orders.orders_new{ord_city=Omaha} stats: [numFiles=1, numRows=1, totalSize=15, rawDataSize=14] 
Partition orders.orders_new{ord_city=Phoenix} stats: [numFiles=1, numRows=1, totalSize=13, rawDataSize=12] 
Partition orders.orders_new{ord_city=Reno} stats: [numFiles=1, numRows=1, totalSize=13, rawDataSize=12] 
OK 
Time taken: 10.493 seconds 
hive>  
```

让我们看看`ORDERS_NEW`中创建了多少个分区：

```py
hive> show partitions ORDERS_NEW; 
OK 
ord_city=Austin 
ord_city=Beverly Hills 
ord_city=Fremont 
ord_city=Gaylord 
ord_city=Logan 
ord_city=Los Angeles 
ord_city=New York 
ord_city=Omaha 
ord_city=Phoenix 
ord_city=Reno 
Time taken: 0.59 seconds, Fetched: 10 row(s) 
hive>  
```

现在很清楚何时使用静态分区和动态分区。当在将数据加载到 Hive 表之前，分区列的值已知得很好时，可以使用静态分区。在动态分区的情况下，分区列的值仅在将数据加载到 Hive 表期间知道。

# Hive 分区桶

桶分区是一种将大型数据集分解成更易管理的小组的技术。桶分区基于哈希函数。当一个表被桶分区时，具有相同列值的所有表记录将进入同一个桶。在物理上，每个桶是表文件夹中的一个文件，就像分区一样。在一个分区表中，Hive 可以在多个文件夹中分组数据。但是，当分区数量有限且数据在它们之间均匀分布时，分区证明是有效的。如果有大量分区，那么它们的使用效果就会降低。因此，在这种情况下，我们可以使用桶分区。我们可以在创建表时显式创建多个桶。

# Hive 桶分区的工作原理

以下图表详细展示了 Hive 桶分区的原理：

![](img/9496a468-bbc4-4711-8027-95b31f9076d7.png)

如果我们决定在一个表中的列（在我们的例子中是`Ord_city`）有三个桶，那么 Hive 将创建三个编号为 0-2（*n-1*）的桶。在记录插入时，Hive 将对每个记录的`Ord_city`列应用哈希函数以决定哈希键。然后 Hive 将对每个哈希值应用模运算符。我们也可以在非分区表中使用桶分区。但是，当与分区表一起使用桶分区功能时，我们将获得最佳性能。桶分区有两个关键好处：

+   **改进查询性能**：在相同分桶列上的连接操作中，我们可以明确指定桶的数量。由于每个桶的数据大小相等，因此分桶表上的 map-side 连接比非分桶表上的连接性能更好。在 map-side 连接中，左侧表桶将确切知道右侧桶中的数据集，以便高效地执行表连接。

+   **改进采样**：因为数据已经被分割成更小的块。

让我们考虑我们的`ORDERS_DATA`表示例。它按`CITY`列进行分区。可能所有城市订单的分布并不均匀。一些城市可能比其他城市有更多的订单。在这种情况下，我们将有倾斜的分区。这将影响查询性能。对于订单较多的城市的查询将比订单较少的城市慢。我们可以通过分桶表来解决这个问题。表中的桶由表 DDL 中的`CLUSTER`子句定义。以下示例详细解释了分桶功能。

# 在非分区表中创建桶

首先，我们将创建一个`ORDERS_BUCK_non_partition`表：

```py
SET hive.exec.dynamic.partition = true; 
SET hive.exec.dynamic.partition.mode = nonstrict; 
SET hive.exec.mx_dynamic.partition=20000; 
SET hive.exec.mx_dynamic.partition.pernode=20000; 
SET hive.enforce.bucketing = true; 

hive> use orders; 
OK 
Time taken: 0.221 seconds 
hive>  
    > CREATE TABLE ORDERS_BUCKT_non_partition 
    > (Ord_id INT, 
    > Ord_month INT, 
    > Ord_customer_id INT, 
    > Ord_city  STRING, 
    > Ord_zip   STRING, 
    > ORD_amt   FLOAT 
    > ) 
    > CLUSTERED BY (Ord_city) into 4 buckets stored as textfile; 
OK 
Time taken: 0.269 seconds 
hive>  

```

要引用所有 Hive `SET`配置参数，请使用此 URL：

[`cwiki.apache.org/confluence/display/Hive/Configuration+Properties`](https://cwiki.apache.org/confluence/display/Hive/Configuration+Properties).

加载新创建的非分区桶表：

```py
hive> insert into ORDERS_BUCKT_non_partition select * from orders_data; 
Query ID = root_20170902190615_1f557644-48d6-4fa1-891d-2deb7729fa2a 
Total jobs = 1 
Launching Job 1 out of 1 
Tez session was closed. Reopening... 
Session re-established. 
Status: Running (Executing on YARN cluster with App id application_1504299625945_0014) 

-------------------------------------------------------------------------------- 
        VERTICES      STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED 
-------------------------------------------------------------------------------- 
Map 1 ..........   SUCCEEDED      1          1        0        0       0       0 
Reducer 2 ......   SUCCEEDED      4          4        0        0       0       0 
-------------------------------------------------------------------------------- 
VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 9.58 s      
-------------------------------------------------------------------------------- 
Loading data to table orders.orders_buckt_non_partition 
Table orders.orders_buckt_non_partition stats: [numFiles=4, numRows=15, totalSize=560, rawDataSize=545] 
OK 
Time taken: 15.55 seconds 
hive> 

```

以下命令显示 Hive 在表中创建了四个桶（文件夹），`00000[0-3]_0`：

```py

[root@sandbox order_data]# hadoop fs -ls /apps/hive/warehouse/orders.db/orders_buckt_non_partition 
Found 4 items 
-rwxrwxrwx   1 root hdfs         32 2017-09-02 19:06 /apps/hive/warehouse/orders.db/orders_buckt_non_partition/000000_0 
-rwxrwxrwx   1 root hdfs        110 2017-09-02 19:06 /apps/hive/warehouse/orders.db/orders_buckt_non_partition/000001_0 
-rwxrwxrwx   1 root hdfs        104 2017-09-02 19:06 /apps/hive/warehouse/orders.db/orders_buckt_non_partition/000002_0 
-rwxrwxrwx   1 root hdfs        314 2017-09-02 19:06 /apps/hive/warehouse/orders.db/orders_buckt_non_partition/000003_0 
[root@sandbox order_data]# 
```

# 在分区表中创建桶

首先，我们将创建一个分桶分区表。在这里，表按`Ord_city`列分为四个桶，但按`Ord_zip`列进一步细分：

```py
SET hive.exec.dynamic.partition = true; 
SET hive.exec.dynamic.partition.mode = nonstrict; 
SET hive.exec.mx_dynamic.partition=20000; 
SET hive.exec.mx_dynamic.partition.pernode=20000; 
SET hive.enforce.bucketing = true; 

hive> CREATE TABLE ORDERS_BUCKT_partition 
    > (Ord_id INT, 
    > Ord_month INT, 
    > Ord_customer_id INT, 
    > Ord_zip   STRING, 
    > ORD_amt   FLOAT 
    > ) 
    > PARTITIONED BY  (Ord_city STRING) 
    > CLUSTERED BY (Ord_zip) into 4 buckets stored as textfile; 
OK 
Time taken: 0.379 seconds 
```

使用动态分区将分桶分区表加载到另一个分桶分区表（`ORDERS_PART`）中：

```py
hive> SET hive.exec.dynamic.partition = true; 
hive> SET hive.exec.dynamic.partition.mode = nonstrict; 
hive> SET hive.exec.mx_dynamic.partition=20000; 
Query returned non-zero code: 1, cause: hive configuration hive.exec.mx_dynamic.partition does not exists. 
hive> SET hive.exec.mx_dynamic.partition.pernode=20000; 
Query returned non-zero code: 1, cause: hive configuration hive.exec.mx_dynamic.partition.pernode does not exists. 
hive> SET hive.enforce.bucketing = true; 
hive> insert into ORDERS_BUCKT_partition partition(Ord_city) select * from orders_part; 
Query ID = root_20170902194343_dd6a2938-6aa1-49f8-a31e-54dafbe8d62b 
Total jobs = 1 
Launching Job 1 out of 1 
Status: Running (Executing on YARN cluster with App id application_1504299625945_0017) 

-------------------------------------------------------------------------------- 
        VERTICES      STATUS  TOTAL  COMPLETED  RUNNING  PENDING  FAILED  KILLED 
-------------------------------------------------------------------------------- 
Map 1 ..........   SUCCEEDED      1          1        0        0       0       0 
Reducer 2 ......   SUCCEEDED      4          4        0        0       0       0 
-------------------------------------------------------------------------------- 
VERTICES: 02/02  [==========================>>] 100%  ELAPSED TIME: 7.13 s      
-------------------------------------------------------------------------------- 
Loading data to table orders.orders_buckt_partition partition (ord_city=null) 
    Time taken to load dynamic partitions: 2.568 seconds 
   Loading partition {ord_city=Phoenix} 
   Loading partition {ord_city=Logan} 
   Loading partition {ord_city=Austin} 
   Loading partition {ord_city=Fremont} 
   Loading partition {ord_city=Beverly Hills} 
   Loading partition {ord_city=Los Angeles} 
   Loading partition {ord_city=New York} 
   Loading partition {ord_city=Omaha} 
   Loading partition {ord_city=Reno} 
   Loading partition {ord_city=Gaylord} 
    Time taken for adding to write entity : 3 
Partition orders.orders_buckt_partition{ord_city=Austin} stats: [numFiles=1, numRows=1, totalSize=22, rawDataSize=21] 
Partition orders.orders_buckt_partition{ord_city=Beverly Hills} stats: [numFiles=1, numRows=1, totalSize=29, rawDataSize=28] 
Partition orders.orders_buckt_partition{ord_city=Fremont} stats: [numFiles=1, numRows=1, totalSize=23, rawDataSize=22] 
Partition orders.orders_buckt_partition{ord_city=Gaylord} stats: [numFiles=1, numRows=1, totalSize=23, rawDataSize=22] 
Partition orders.orders_buckt_partition{ord_city=Logan} stats: [numFiles=1, numRows=1, totalSize=26, rawDataSize=25] 
Partition orders.orders_buckt_partition{ord_city=Los Angeles} stats: [numFiles=1, numRows=6, totalSize=166, rawDataSize=160] 
Partition orders.orders_buckt_partition{ord_city=New York} stats: [numFiles=1, numRows=1, totalSize=23, rawDataSize=22] 
Partition orders.orders_buckt_partition{ord_city=Omaha} stats: [numFiles=1, numRows=1, totalSize=25, rawDataSize=24] 
Partition orders.orders_buckt_partition{ord_city=Phoenix} stats: [numFiles=1, numRows=1, totalSize=23, rawDataSize=22] 
Partition orders.orders_buckt_partition{ord_city=Reno} stats: [numFiles=1, numRows=1, totalSize=20, rawDataSize=19] 
OK 
Time taken: 13.672 seconds 
hive>  
```

# Hive 视图

Hive 视图是一个逻辑表。它就像任何 RDBMS 视图一样。概念是相同的。当创建视图时，Hive 不会将其中的任何数据存储进去。当创建视图时，Hive 会冻结元数据。Hive 不支持任何 RDBMS 的物化视图概念。视图的基本目的是隐藏查询复杂性。有时，HQL 包含复杂的连接、子查询或过滤器。借助视图，整个查询可以简化为一个虚拟表。

当在底层表上创建视图时，对该表的任何更改，甚至添加或删除该表，都会在视图中失效。此外，当创建视图时，它只更改元数据。但是，当查询访问该视图时，它将触发 MapReduce 作业。视图是一个纯粹的逻辑对象，没有关联的存储（Hive 中目前不支持物化视图）。当查询引用视图时，将评估视图的定义以生成一组行供查询进一步处理。（这是一个概念性描述。实际上，作为查询优化的部分，Hive 可能会将视图的定义与查询结合起来，例如，将查询中的过滤器推入视图。）

视图的模式在创建视图时被冻结；对底层表（例如，添加列）的后续更改不会反映在视图的模式中。如果底层表被删除或以不兼容的方式更改，后续尝试查询无效视图将失败。视图是只读的，不能用作`LOAD`/`INSERT`/`ALTER`更改元数据的目标。视图可能包含`ORDER BY`和`LIMIT`子句。如果引用查询也包含这些子句，则查询级别的子句将在视图子句（以及查询中的任何其他操作）之后评估。例如，如果视图指定`LIMIT 5`，并且引用查询以（`select * from v LIMIT 10`）执行，则最多返回五行。

# 视图的语法

让我们看看几个视图的示例：

```py
CREATE VIEW [IF NOT EXISTS] [db_name.]view_name [(column_name [COMMENT column_comment], ...) ] 
  [COMMENT view_comment] 
  [TBLPROPERTIES (property_name = property_value, ...)] 
  AS SELECT ...;
```

我将通过以下几个示例演示视图的优势。假设我们有两个表，`Table_X`和`Table_Y`，具有以下模式：`Table_XXCol_1`字符串，`XCol_2`字符串，`XCol_3`字符串，`Table_YYCol_1`字符串，`YCol_2`字符串，`YCol_3`字符串，和`YCol_4`字符串。要创建与基础表完全相同的视图，请使用以下代码：

```py
Create view table_x_view as select * from Table_X; 
```

要在基础表的选定列上创建视图，请使用以下语法：

```py
Create view table_x_view as select xcol_1,xcol_3  from Table_X; 
```

要创建一个用于筛选基础表列值的视图，我们可以使用：

```py
Create view table_x_view as select * from Table_X where XCol_3 > 40 and  XCol_2 is not null; 
```

要创建一个用于隐藏查询复杂性的视图：

```py
create view table_union_view  as select XCol_1, XCol_2, XCol_3,Null from Table_X 
   where XCol_2  = "AAA" 
   union all 
   select YCol_1, YCol_2, YCol_3, YCol_4 from Table_Y 
   where YCol_3 = "BBB"; 

   create view table_join_view as select * from Table_X 
   join Table_Y on Table_X. XCol_1 = Table_Y. YCol_1; 
```

# Hive 索引

索引的主要目的是便于快速搜索记录并加速查询。Hive 索引的目标是提高对表特定列的查询查找速度。如果没有索引，带有如`WHERE tab1.col1 = 10`这样的谓词的查询将加载整个表或分区并处理所有行。但是，如果为`col1`存在索引，则只需加载和处理文件的一部分。索引可以提供的查询速度提升是以创建索引的额外处理和存储索引的磁盘空间为代价的。有两种类型的索引：

+   紧凑索引

+   位图索引

主要区别在于存储不同块中行的映射值。

# 紧凑索引

在 HDFS 中，数据存储在块中。但是扫描存储在哪个块中的数据是耗时的。紧凑索引存储索引列的值及其 `blockId`。因此，查询将不会访问表。相反，查询将直接访问存储列值和 `blockId` 的紧凑索引。无需扫描所有块以找到数据！所以，在执行查询时，它将首先检查索引，然后直接进入该块。

# 位图索引

位图索引将索引列的值和行列表的组合存储为位图。位图索引通常用于具有不同值的列。让我们回顾几个示例：基础表，`Table_XXCol_1` 整数，`XCol_2` 字符串，`XCol_3` 整数，和 `XCol_4` 字符串。创建索引：

```py
CREATE INDEX table_x_idx_1 ON TABLE table_x (xcol_1) AS 'COMPACT';  
SHOW INDEX ON table_x_idx;  
DROP INDEX table_x_idx ON table_x; 

CREATE INDEX table_x_idx_2 ON TABLE table_x (xcol_1) AS 'COMPACT' WITH DEFERRED REBUILD;  
ALTER INDEX table_x_idx_2 ON table_x REBUILD;  
SHOW FORMATTED INDEX ON table_x; 
```

前面的索引为空，因为它使用了 `DEFERRED REBUILD` 子句创建，无论表是否包含任何数据。在此索引创建后，需要使用 `REBUILD` 命令来构建索引结构。在索引创建后，如果基础表中的数据发生变化，必须使用 `REBUILD` 命令来更新索引。创建索引并将其存储在文本文件中：

```py
CREATE INDEX table_x_idx_3 ON TABLE table_x (table_x) AS 'COMPACT' ROW FORMAT DELIMITED  
FIELDS TERMINATED BY 't'  
STORED AS TEXTFILE; 
```

创建位图索引：

```py
CREATE INDEX table_x_bitmap_idx_4 ON TABLE table_x (table_x) AS 'BITMAP' WITH DEFERRED REBUILD;  
ALTER INDEX table_x_bitmap_idx_4 ON table03 REBUILD;  
SHOW FORMATTED INDEX ON table_x; 
DROP INDEX table_x_bitmap_idx_4 ON table_x; 
```

# 使用 Hive 的 JSON 文档

JSON 是结构化数据的最小可读格式。它主要用于在服务器和 Web 应用程序之间传输数据，作为 XML 的替代方案。JSON 建立在两种结构之上：

+   名称/值对的集合。在不同的语言中，这通常实现为对象、记录、结构、字典、散列表、键列表或关联数组。

+   值的有序列表。在大多数语言中，这通常实现为数组、向量、列表或序列。

请在以下网址了解更多关于 JSON 的信息：[`www.json.org/`](http://www.json.org/). [](http://www.json.org/)

# 示例 1 – 使用 Hive 访问简单的 JSON 文档（Hive 0.14 及更高版本）

在本例中，我们将看到如何使用 HiveQL 查询简单的 JSON 文档。假设我们想访问以下 `Sample-Json-simple.json` 文件在 `HiveSample-Json-simple.json`：

```py
{"username":"abc","tweet":"Sun shine is bright.","time1": "1366150681" } 
{"username":"xyz","tweet":"Moon light is mild .","time1": "1366154481" } 
```

查看 `Sample-Json-simple.json` 文件：

```py
[root@sandbox ~]# cat Sample-Json-simple.json 
{"username":"abc","tweet":"Sun shine is bright.","timestamp": 1366150681 } 
{"username":"xyz","tweet":"Moon light is mild .","timestamp": 1366154481 } 
[root@sandbox ~]#  
```

将 `Sample-Json-simple.json` 加载到 HDFS：

```py
[root@sandbox ~]# hadoop fs -mkdir  /user/hive-simple-data/ 
[root@sandbox ~]# hadoop fs -put Sample-Json-simple.json /user/hive-simple-data/ 
```

创建一个外部 Hive 表，`simple_json_table`：

```py
hive> use orders; 
OK 
Time taken: 1.147 seconds 
hive>  
CREATE EXTERNAL TABLE simple_json_table ( 
username string, 
tweet string, 
time1 string) 
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe' 
LOCATION '/user/hive-simple-data/'; 
OK 
Time taken: 0.433 seconds 
hive>  
```

现在验证记录：

```py
hive> select * from simple_json_table ; 
OK 
abc      Sun shine is bright.    1366150681 
xyz      Moon light is mild .    1366154481 
Time taken: 0.146 seconds, Fetched: 2 row(s) 
hive>  
```

# 示例 2 – 使用 Hive 访问嵌套 JSON 文档（Hive 0.14 及更高版本）

我们将看到如何使用 HiveQL 查询嵌套 JSON 文档。假设我们想访问以下 `Sample-Json-complex.json` 文件在 `HiveSample-Json-complex.json`：

```py
{"DocId":"Doc1","User1":{"Id":9192,"Username":"u2452","ShippingAddress":{"Address1":"6373 Sun Street","Address2":"apt 12","City":"Foster City","State":"CA"},"Orders":[{"ItemId":5343,"OrderDate":"12/23/2017"},{"ItemId":7362,"OrderDate":"12/24/2017"}]}} 
```

将 `Sample-Json-simple.json` 加载到 HDFS：

```py
[root@sandbox ~]# hadoop fs -mkdir  /user/hive-complex-data/ 
[root@sandbox ~]# hadoop fs -put Sample-Json-complex.json /user/hive-complex-data/ 
```

创建一个外部 Hive 表，`json_nested_table`：

```py
hive>  
CREATE EXTERNAL TABLE json_nested_table( 
DocId string, 
user1 struct<Id: int, username: string, shippingaddress:struct<address1:string,address2:string,city:string,state:string>, orders:array<struct<ItemId:int,orderdate:string>>> 
) 
ROW FORMAT SERDE 
'org.apache.hive.hcatalog.data.JsonSerDe' 
LOCATION 
'/user/hive-complex-data/'; 
OK 
Time taken: 0.535 seconds 
hive>  
```

验证记录：

```py
hive> select DocId,user1.username,user1.orders FROM json_nested_table; 
OK 
Doc1     u2452   [{"itemid":5343,"orderdate":"12/23/2017"},{"itemid":7362,"orderdate":"12/24/2017"}] 
Time taken: 0.598 seconds, Fetched: 1 row(s) 
hive>  
```

# 示例 3 – 使用 Hive 和 Avro 进行模式演变（Hive 0.14 及更高版本）

在生产环境中，我们必须更改表结构以应对新的业务需求。表模式必须更改以添加/删除/重命名表列。这些更改中的任何一项都会对下游 ETL 作业产生不利影响。为了避免这些，我们必须对 ETL 作业和目标表进行相应的更改。

架构演变允许你在保持与旧数据架构向后兼容的同时更新用于写入新数据的架构。然后你可以将所有数据一起读取，就像所有数据具有一个架构一样。请阅读以下 URL 上的 Avro 序列化更多信息：[`avro.apache.org/`](https://avro.apache.org/)。在以下示例中，我将演示 Avro 和 Hive 表如何在不失败 ETL 作业的情况下吸收源表架构更改。我们将创建一个 MySQL 数据库中的客户表，并使用 Avro 文件将其加载到目标 Hive 外部表中。然后我们将向源表添加一个额外的列，以查看 Hive 表如何无错误地吸收该更改。连接到 MySQL 创建源表（`customer`）：

```py
mysql -u root -p 

GRANT ALL PRIVILEGES ON *.* TO 'sales'@'localhost' IDENTIFIED BY 'xxx';  

mysql -u sales  -p 

mysql> create database orders; 

mysql> use orders; 

CREATE TABLE customer( 
cust_id INT , 
cust_name  VARCHAR(20) NOT NULL, 
cust_city VARCHAR(20) NOT NULL, 
PRIMARY KEY ( cust_id ) 
); 
```

向 `customer` 表中插入记录：

```py
INSERT into customer (cust_id,cust_name,cust_city) values (1,'Sam James','Austin'); 
INSERT into customer (cust_id,cust_name,cust_city) values (2,'Peter Carter','Denver'); 
INSERT into customer (cust_id,cust_name,cust_city) values (3,'Doug Smith','Sunnyvale'); 
INSERT into customer (cust_id,cust_name,cust_city) values (4,'Harry Warner','Palo Alto'); 

```

在 Hadoop 上，运行以下 `sqoop` 命令以导入 `customer` 表并将数据存储在 Avro 文件中到 HDFS：

```py

hadoop fs -rmr /user/sqoop_data/avro 
sqoop import -Dmapreduce.job.user.classpath.first=true  
--connect jdbc:mysql://localhost:3306/orders   
--driver com.mysql.jdbc.Driver  
--username sales --password xxx  
--target-dir /user/sqoop_data/avro  
--table customer  
--as-avrodatafile  

```

验证目标 HDFS 文件夹：

```py
[root@sandbox ~]# hadoop fs -ls /user/sqoop_data/avro 
Found 7 items 
-rw-r--r--   1 root hdfs          0 2017-09-09 08:57 /user/sqoop_data/avro/_SUCCESS 
-rw-r--r--   1 root hdfs        472 2017-09-09 08:57 /user/sqoop_data/avro/part-m-00000.avro 
-rw-r--r--   1 root hdfs        475 2017-09-09 08:57 /user/sqoop_data/avro/part-m-00001.avro 
-rw-r--r--   1 root hdfs        476 2017-09-09 08:57 /user/sqoop_data/avro/part-m-00002.avro 
-rw-r--r--   1 root hdfs        478 2017-09-09 08:57 /user/sqoop_data/avro/part-m-00003.avro 
```

创建一个 Hive 外部表以访问 Avro 文件：

```py
use orders; 
drop table customer ; 
CREATE EXTERNAL TABLE customer  
( 
cust_id INT , 
cust_name  STRING , 
cust_city STRING   
) 
STORED AS AVRO 
location '/user/sqoop_data/avro/'; 
```

验证 Hive `customer` 表：

```py
hive> select * from customer; 
OK 
1  Sam James   Austin 
2  Peter Carter      Denver 
3  Doug Smith  Sunnyvale 
4  Harry Warner      Palo Alto 
Time taken: 0.143 seconds, Fetched: 4 row(s) 
hive>  

```

完美！我们没有错误。我们成功地将源 `customer` 表导入目标 Hive 表，使用了 Avro 序列化。现在，我们在源表中添加一个列，并再次导入以验证我们可以在没有任何架构更改的情况下访问目标 Hive 表。连接到 MySQL 并添加一个额外的列：

```py
mysql -u sales  -p 

mysql>  
ALTER TABLE customer 
ADD COLUMN cust_state VARCHAR(15) NOT NULL; 

mysql> desc customer; 
+------------+-------------+------+-----+---------+-------+ 
| Field      | Type        | Null | Key | Default | Extra | 
+------------+-------------+------+-----+---------+-------+ 
| cust_id    | int(11)     | NO   | PRI | 0       |       | 
| cust_name  | varchar(20) | NO   |     | NULL    |       | 
| cust_city  | varchar(20) | NO   |     | NULL    |       | 
| CUST_STATE | varchar(15) | YES  |     | NULL    |       | 
+------------+-------------+------+-----+---------+-------+ 
4 rows in set (0.01 sec) 

mysql>  
```

现在插入行：

```py
INSERT into customer (cust_id,cust_name,cust_city,cust_state) values (5,'Mark Slogan','Huston','TX'); 
INSERT into customer (cust_id,cust_name,cust_city,cust_state) values (6,'Jane Miller','Foster City','CA'); 
```

在 Hadoop 上，运行以下 `sqoop` 命令以导入 `customer` 表，以便追加新的地址列和数据。我使用了 `append` 和 `where "cust_id > 4"` 参数来仅导入新行：

```py
sqoop import -Dmapreduce.job.user.classpath.first=true  
--connect jdbc:mysql://localhost:3306/orders   
--driver com.mysql.jdbc.Driver  
--username sales --password xxx  
--table customer  
--append  
--target-dir /user/sqoop_data/avro  
--as-avrodatafile  
--where "cust_id > 4"  
```

验证 HDFS 文件夹：

```py
[root@sandbox ~]# hadoop fs -ls /user/sqoop_data/avro 
Found 7 items 
-rw-r--r--   1 root hdfs          0 2017-09-09 08:57 /user/sqoop_data/avro/_SUCCESS 
-rw-r--r--   1 root hdfs        472 2017-09-09 08:57 /user/sqoop_data/avro/part-m-00000.avro 
-rw-r--r--   1 root hdfs        475 2017-09-09 08:57 /user/sqoop_data/avro/part-m-00001.avro 
-rw-r--r--   1 root hdfs        476 2017-09-09 08:57 /user/sqoop_data/avro/part-m-00002.avro 
-rw-r--r--   1 root hdfs        478 2017-09-09 08:57 /user/sqoop_data/avro/part-m-00003.avro 
-rw-r--r--   1 root hdfs        581 2017-09-09 09:00 /user/sqoop_data/avro/part-m-00004.avro 
-rw-r--r--   1 root hdfs        586 2017-09-09 09:00 /user/sqoop_data/avro/part-m-00005.avro 
```

现在，让我们验证我们的目标 Hive 表是否仍然能够访问旧的和新的 Avro 文件：

```py
hive> select * from customer; 
OK 
1  Sam James   Austin 
2  Peter Carter      Denver 
3  Doug Smith  Sunnyvale 
4  Harry Warner      Palo Alto 
Time taken: 0.143 seconds, Fetched: 4 row(s 
```

太好了！没有错误。尽管如此，一切照旧；现在我们将向 Hive 表添加一个新列，以查看新添加的 Avro 文件：

```py

hive> use orders; 
hive> ALTER TABLE customer ADD COLUMNS (cust_state STRING); 
hive> desc customer; 
OK 
cust_id              int                                          
cust_name            string                                       
cust_city            string                                       
cust_state           string                                       
Time taken: 0.488 seconds, Fetched: 4 row(s 
```

验证 Hive 表中的新数据：

```py
hive> select * from customer; 
OK 
1  Sam James   Austin      NULL 
2  Peter Carter      Denver      NULL 
3  Doug Smith  Sunnyvale   NULL 
4  Harry Warner      Palo Alto   NULL 
5  Mark Slogan Huston      TX 
6  Jane Miller Foster City CA 
Time taken: 0.144 seconds, Fetched: 6 row(s) 
hive>  
```

太棒了！看看客户 ID `5` 和 `6`。我们可以看到新添加的列（`cust_state`）及其值。您可以使用相同的技术进行删除列和替换列的实验。现在我们相当清楚地了解了如何使用 Apache Hive 访问数据。在下一节中，我们将学习如何使用 HBase 访问数据，HBase 是一个 NoSQL 数据存储。

# Apache HBase

我们刚刚学习了 Hive，这是一个用户可以使用 SQL 命令访问数据的数据库。但是，也有一些数据库用户不能使用 SQL 命令。这些数据库被称为**NoSQL 数据存储**。HBase 就是一个 NoSQL 数据库。那么，NoSQL 实际上意味着什么呢？NoSQL 不仅仅意味着 SQL。在 HBase 这样的 NoSQL 数据存储中，关系型数据库管理系统（RDBMS）的主要特性，如验证和一致性，被放宽了。另外，RDBMS 或 SQL 数据库与 NoSQL 数据库之间的重要区别在于写入时模式与读取时模式。在写入时模式中，数据在写入表时进行验证，而读取时模式支持在读取数据时进行验证。通过放宽在写入数据时的基本数据验证，NoSQL 数据存储能够支持存储大量数据速度。目前市场上大约有 150 种 NoSQL 数据存储。每种 NoSQL 数据存储都提供一些独特的特性。一些流行的 NoSQL 数据存储包括 HBase、Cassandra、MongoDB、Druid、Apache Kudu 和 Accumulo 等。

您可以在[`nosql-database.org/`](http://nosql-database.org/)上获取所有类型 NoSQL 数据库的详细列表。

HBase 是一个被许多大型公司如 Facebook、Google 等广泛使用的流行 NoSQL 数据库。

# HDFS 与 HBase 之间的差异

以下解释了 HDFS 和 HBase 之间的关键差异。Hadoop 建立在 HDFS 之上，支持存储大量（PB 级）数据集。这些数据集通过批处理作业，使用 MapReduce 算法进行访问。为了在如此庞大的数据集中找到数据元素，需要扫描整个数据集。另一方面，HBase 建立在 HDFS 之上，并为大型表提供快速的记录查找（和更新）。HBase 内部将数据存储在 HDFS 上存在的索引 StoreFiles 中，以实现高速查找。

# Hive 与 HBase 之间的差异

HBase 是一个数据库管理系统；它支持事务处理和分析处理。Hive 是一个数据仓库系统，只能用于分析处理。HBase 支持低延迟和随机数据访问操作。Hive 仅支持批处理，这导致高延迟。HBase 不支持任何 SQL 接口与表数据进行交互。您可能需要编写 Java 代码来读取和写入 HBase 表中的数据。有时，处理涉及多个数据集连接的数据集的 Java 代码可能非常复杂。但 Hive 支持使用 SQL 进行非常容易的访问，这使得读取和写入其表中的数据变得非常简单。在 HBase 中，数据建模涉及灵活的数据模型和列式数据存储，必须支持数据反规范化。HBase 表的列在将数据写入表时确定。在 Hive 中，数据模型涉及具有固定模式（如 RDBMS 数据模型）的表。

# HBase 的关键特性

以下是一些 HBase 的关键特性：

+   **排序行键**：在 HBase 中，数据处理通过三个基本操作/API 进行：get、put 和 scan。这三个 API 都使用行键访问数据，以确保数据访问的流畅。由于扫描是在行键的范围内进行的，因此 HBase 根据行键的字典顺序对行进行排序。使用这些排序行键，可以从其起始和终止行键简单地定义扫描。这在单个数据库调用中获取所有相关数据方面非常强大。应用程序开发者可以设计一个系统，通过根据其时间戳查询最近的行来访问最近的数据集，因为所有行都是根据最新的时间戳在表中按顺序存储的。

+   **控制数据分片**：HBase 表的行键强烈影响数据分片。表数据按行键、列族和列键的升序排序。良好的行键设计对于确保数据在 Hadoop 集群中均匀分布非常重要。由于行键决定了表行的排序顺序，因此表中的每个区域最终负责行键空间的一部分的物理存储。

+   **强一致性**：HBase 更倾向于一致性而不是可用性。它还支持基于每行的 ACID 级语义。当然，这会影响写性能，可能会变慢。总体而言，权衡有利于应用程序开发者，他们将保证数据存储始终具有数据的正确值。

+   **低延迟处理**：HBase 支持对存储的所有数据进行快速、随机的读取和写入。

+   **灵活性**：HBase 支持任何类型——结构化、半结构化、非结构化。

+   **可靠性**：HBase 表数据块被复制多次，以确保防止数据丢失。HBase 还支持容错。即使在任何区域服务器故障的情况下，表数据始终可用于处理。

# HBase 数据模型

这些是 HBase 数据模型的关键组件：

+   **表**：在 HBase 中，数据存储在逻辑对象中，称为**表**，该表具有多个行。

+   **行**：HBase 中的行由一个行键和一个或多个列组成。行键用于排序行。目标是按这种方式存储数据，使得相关的行彼此靠近。行键可以是多个列的组合。行键类似于表的主键，必须是唯一的。HBase 使用行键在列中查找数据。例如，`customer_id`可以是`customer`表的行键。

+   **列**：HBase 中的列由一个列族和一个列限定符组成。

+   **列限定符**：它是表的列名。

+   **单元格**：这是行、列族和列限定符的组合，包含一个值和一个时间戳，该时间戳表示值的版本。

+   **列族**：它是一组位于同一位置并存储在一起的列的集合，通常出于性能考虑。每个列族都有一组存储属性，例如缓存、压缩和数据编码。

# RDBMS 表与列式数据存储之间的差异

我们都知道数据在任何关系型数据库管理系统（RDBMS）表中是如何存储的。它看起来像这样：

| **ID** | `Column_1` | `Column_2` | `Column_3` | `Column_4` |
| --- | --- | --- | --- | --- |
| 1 | A | 11 | P | XX |
| 2 | B | 12 | Q | YY |
| 3 | C | 13 | R | ZZ |
| 4 | D | 14 | S | XX1 |

列 ID 用作表的唯一/主键，以便从表的其它列访问数据。但在像 HBase 这样的列式数据存储中，相同的表被分为键和值，并按如下方式存储：

| **Key** | **Value** |
| --- | --- |
| **行** | **列** | **列值** |
| 1 | `Column_1` | `A` |
| 1 | `Column_2` | `11` |
| 1 | `Column_3` | `P` |
| 1 | `Column_4` | `XX` |
| 2 | `Column_1` | `B` |
| 2 | `Column_2` | `12` |
| 2 | `Column_3` | `Q` |
| 2 | `Column_4` | `YY` |
| 3 | `Column_1` | `C` |
| 3 | `Column_2` | `13` |
| 3 | `Column_3` | `R` |
| 3 | `Column_4` | `ZZ` |

在 HBase 中，每个表都是一个排序映射格式，其中每个键按升序排序。内部，每个键和值都被序列化并以字节数组格式存储在磁盘上。每个列值通过其对应键进行访问。因此，在前面的表中，我们定义了一个键，它是两个列的组合，*行 + 列*。例如，为了访问第 1 行的`Column_1`数据元素，我们必须使用一个键，即行 1 + `column_1`。这就是为什么行键设计在 HBase 中非常关键。在创建 HBase 表之前，我们必须为每个列决定一个列族。列族是一组列的集合，这些列位于同一位置并一起存储，通常出于性能原因。每个列族都有一组存储属性，例如缓存、压缩和数据编码。例如，在一个典型的`CUSTOMER`表中，我们可以定义两个列族，即`cust_profile`和`cust_address`。所有与客户地址相关的列都分配给列族`cust_address`；所有其他列，即`cust_id`、`cust_name`和`cust_age`，都分配给列族`cust_profile`。分配列族后，我们的示例表将如下所示：

| **Key** | **Value** |
| --- | --- |
| **行** | **列** | **列族** | **值** | **时间戳** |
| 1 | `Column_1` | `cf_1` | `A` | `1407755430` |
| 1 | `Column_2` | `cf_1` | `11` | `1407755430` |
| 1 | `Column_3` | `cf_1` | `P` | `1407755430` |
| 1 | `Column_4` | `cf_2` | `XX` | `1407755432` |
| 2 | `Column_1` | `cf_1` | `B` | `1407755430` |
| 2 | `Column_2` | `cf_1` | `12` | `1407755430` |
| 2 | `Column_3` | `cf_1` | `Q` | `1407755430` |
| 2 | `Column_4` | `cf_2` | `YY` | `1407755432` |
| 3 | `Column_1` | `cf_1` | `C` | `1407755430` |
| 3 | `Column_2` | `cf_1` | `13` | `1407755430` |
| 3 | `Column_3` | `cf_1` | `R` | `1407755430` |
| 3 | `Column_4` | `cf_2` | `ZZ` | `1407755432` |

当向表中插入数据时，HBase 将为每个单元格的每个版本自动添加时间戳。

# HBase 架构

如果我们要从 HBase 表中读取数据，我们必须提供一个合适的行 ID，HBase 将根据提供的行 ID 进行查找。HBase 使用以下排序嵌套映射来返回行 ID 的列值：行 ID、列族、列在时间戳和值。HBase 始终部署在 Hadoop 上。以下是一个典型的安装：

![](img/2b1fd55d-59df-48b3-b0a5-d5aacf0c836e.png)

它是 HBase 集群的主服务器，负责 RegionServer 的管理、监控和管理，例如将 region 分配给 RegionServer、region 拆分等。在分布式集群中，HMaster 通常运行在 Hadoop NameNode 上。

ZooKeeper 是 HBase 集群的协调器。HBase 使用 ZooKeeper 作为分布式协调服务来维护集群中的服务器状态。ZooKeeper 维护哪些服务器是活跃的和可用的，并提供服务器故障通知。RegionServer 负责 region 的管理。RegionServer 部署在 DataNode 上。它提供读写服务。RegionServer 由以下附加组件组成：

+   Regions：HBase 表通过行键范围水平分割成 region。一个 region 包含表中从 region 的起始键到结束键之间的所有行。**预写日志**（**WAL**）用于存储尚未存储在磁盘上的新数据。

+   MemStore 是一个写缓存。它存储尚未写入磁盘的新数据。在写入磁盘之前进行排序。每个 region 每个列族有一个 MemStore。Hfile 在磁盘/HDFS 上以排序的键/值形式存储行：

![](img/79ed20fa-ef32-4b86-9fc6-d0cb11127b8f.png)

# HBase 架构概述

+   HBase 集群由一个活动主服务器和一个或多个备份主服务器组成

+   该集群包含多个 RegionServer

+   HBase 表始终很大，行被分割成称为**region**的分区/碎片

+   每个 RegionServer 托管一个或多个 region

+   HBase 目录被称为 META 表，它存储表 region 的位置

+   ZooKeeper 存储 META 表的位置

+   在写入过程中，客户端将 put 请求发送到 HRegionServer

+   数据写入 WAL

+   然后将数据推入 MemStore 并向客户端发送确认

+   一旦 MemStore 中积累足够的数据，它就会将数据刷新到 HDFS 上的 Hfile

+   HBase 的压缩过程定期激活，将多个 HFile 合并成一个 Hfile（称为**压缩**）

# HBase 行键设计

行键设计是 HBase 表设计的一个非常关键的部分。在键设计期间，必须采取适当的措施以避免热点。在设计不良的键的情况下，所有数据都将被摄入到少数几个节点中，导致集群不平衡。然后，所有读取都必须指向这些少数节点，导致数据读取速度变慢。我们必须设计一个键，以帮助将数据均匀地加载到集群的所有节点上。可以通过以下技术避免热点：

+   **密钥加盐**：这意味着在密钥的开头添加一个任意值，以确保行在所有表区域中均匀分布。例如，`aa-customer_id`、`bb-customer_id`等。

+   **密钥哈希**：密钥可以被哈希，哈希值可以用作行键，例如，`HASH(customer_id)`。

+   **反向时间戳密钥**：在这种技术中，您必须定义一个常规密钥，然后将其与一个反向时间戳相关联。时间戳必须通过从任意最大值中减去它来反转，然后附加到密钥上。例如，如果`customer_id`是您的行 ID，则新的密钥将是`customer_id` + 反向时间戳。

设计 HBase 表时的以下是一些指南：

+   每个表定义不超过两个列族

+   尽可能保持列族名称尽可能小

+   尽可能保持列名尽可能小

+   尽可能保持行键长度尽可能小

+   不要将行版本设置在较高水平

+   表不应超过 100 个区域

# 示例 4 – 从 MySQL 表加载数据到 HBase 表

我们将使用我们之前创建的相同的`customer`表：

```py
mysql -u sales -p
mysql> use orders;
mysql> select * from customer;
+---------+--------------+--------------+------------+
| cust_id | cust_name | cust_city | InsUpd_on |
+---------+--------------+--------------+------------+
| 1 | Sam James | Austin | 1505095030 |
| 2 | Peter Carter | Denver | 1505095030 |
| 3 | Doug Smith | Sunnyvale | 1505095030 |
| 4 | Harry Warner | Palo Alto | 1505095032 |
| 5 | Jen Turner | Cupertino | 1505095036 |
| 6 | Emily Stone | Walnut Creek | 1505095038 |
| 7 | Bill Carter | Greenville | 1505095040 |
| 8 | Jeff Holder | Dallas | 1505095042 |
| 10 | Mark Fisher | Mil Valley | 1505095044 |
| 11 | Mark Fisher | Mil Valley | 1505095044 |
+---------+--------------+--------------+------------+
10 rows in set (0.00 sec)
```

启动 HBase 并在 HBase 中创建一个`customer`表：

```py
hbase shell
create 'customer','cf1'
```

使用 Sqoop 在 HBase 中加载数据库`customer`表数据：

```py
hbase
sqoop import --connect jdbc:mysql://localhost:3306/orders --driver com.mysql.jdbc.Driver --username sales --password sales1 --table customer --hbase-table customer --column-family cf1 --hbase-row-key cust_id
```

验证 HBase 表：

```py
hbase shell
scan 'customer'
```

您必须看到 HBase 表中的所有 11 行。

# 示例 5 – 从 MySQL 表增量加载数据到 HBase 表

```py
mysql -u sales -p
mysql> use orders;
```

插入一个新客户并更新现有客户：

```py
mysql> Update customer set cust_city = 'Dublin', InsUpd_on = '1505095065' where cust_id = 4;
Query OK, 1 row affected (0.00 sec)
Rows matched: 1 Changed: 1 Warnings: 0

mysql> INSERT into customer (cust_id,cust_name,cust_city,InsUpd_on) values (12,'Jane Turner','Glen Park',1505095075);
Query OK, 1 row affected (0.00 sec)

mysql> commit;
Query OK, 0 rows affected (0.00 sec)

mysql> select * from customer;
+---------+--------------+--------------+------------+
| cust_id | cust_name | cust_city | InsUpd_on |
+---------+--------------+--------------+------------+
| 1 | Sam James | Austin | 1505095030 |
| 2 | Peter Carter | Denver | 1505095030 |
| 3 | Doug Smith | Sunnyvale | 1505095030 |
| 4 | Harry Warner | Dublin | 1505095065 |
| 5 | Jen Turner | Cupertino | 1505095036 |
| 6 | Emily Stone | Walnut Creek | 1505095038 |
| 7 | Bill Carter | Greenville | 1505095040 |
| 8 | Jeff Holder | Dallas | 1505095042 |
| 10 | Mark Fisher | Mil Valley | 1505095044 |
| 11 | Mark Fisher | Mil Valley | 1505095044 |
| 12 | Jane Turner | Glen Park | 1505095075 | +---------+--------------+--------------+------------+
11 rows in set (0.00 sec)
mysql>
```

# 示例 6 – 将 MySQL 客户更改数据加载到 HBase 表中

在这里，我们使用了`InsUpd_on`列作为我们的 ETL 日期：

```py
sqoop import --connect jdbc:mysql://localhost:3306/orders --driver com.mysql.jdbc.Driver --username sales --password sales1 --table customer --hbase-table customer --column-family cf1 --hbase-row-key cust_id --append -- -m 1 --where "InsUpd_on > 1505095060"

hbase shell
hbase(main):010:0> get 'customer', '4'
COLUMN          CELL
cf1:InsUpd_on   timestamp=1511509774123, value=1505095065
cf1:cust_city   timestamp=1511509774123, value=Dublin cf1:cust_name   timestamp=1511509774123, value=Harry Warner
3 row(s) in 0.0200 seconds

hbase(main):011:0> get 'customer', '12'
COLUMN           CELL
cf1:InsUpd_on    timestamp=1511509776158, value=1505095075
cf1:cust_city    timestamp=1511509776158, value=Glen Park
cf1:cust_name    timestamp=1511509776158, value=Jane Turner
3 row(s) in 0.0050 seconds

hbase(main):012:0>
```

# 示例 7 – Hive 与 HBase 集成

现在，我们将使用 Hive 外部表访问 HBase `customer`表：

```py
create external table customer_hbase(cust_id string, cust_name string, cust_city string, InsUpd_on string)
STORED BY 'org.apache.hadoop.hive.hbase.HBaseStorageHandler'with serdeproperties ("hbase.columns.mapping"=":key,cf1:cust_name,cf1:cust_city,cf1:InsUpd_on")tblproperties("hbase.table.name"="customer");

hive> select * from customer_hbase;
OK
1 Sam James Austin 1505095030
10 Mark Fisher Mil Valley 1505095044
11 Mark Fisher Mil Valley 1505095044
12 Jane Turner Glen Park 1505095075
2 Peter Carter Denver 1505095030
3 Doug Smith Sunnyvale 1505095030
4 Harry Warner Dublin 1505095065
5 Jen Turner Cupertino 1505095036
6 Emily Stone Walnut Creek 1505095038
7 Bill Carter Greenville 1505095040
8 Jeff Holder Dallas 1505095042
Time taken: 0.159 seconds, Fetched: 11 row(s)
hive>
```

# 摘要

在本章中，我们看到了如何使用称为 Hadoop SQL 接口的 Hive 存储和访问数据。我们研究了 Hive 中的各种分区和索引策略。工作示例帮助我们理解了在 Hive 中使用 Avro 访问 JSON 数据和模式演变。在第二部分中，我们研究了名为 HBase 的 NoSQL 数据存储以及它与 RDBMS 的区别。HBase 表的行设计对于平衡读写以避免区域热点至关重要。必须记住本章中讨论的 HBase 表设计最佳实践。工作示例展示了数据导入 HBase 表和其与 Hive 集成的简单路径。

在下一章中，我们将探讨设计实时数据分析的工具和技术。
