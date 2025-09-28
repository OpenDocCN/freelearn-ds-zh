

# 第五章：从结构化和非结构化数据库中摄取数据

现在，我们可以从多个来源存储和检索数据，最佳存储方法取决于正在处理的信息类型。例如，大多数 API 以非结构化格式提供数据，因为这允许共享多种格式的数据（例如，音频、视频和图像），并且通过使用数据湖具有低存储成本。然而，如果我们想使定量数据可用于与多个工具一起支持分析，那么最可靠的选择可能是结构化数据。

最终，无论你是数据分析师、科学家还是工程师，了解如何管理结构化和非结构化数据都是至关重要的。

本章我们将介绍以下食谱：

+   配置 JDBC 连接

+   使用 SQL 从 JDBC 数据库中摄取数据

+   连接到 NoSQL 数据库（MongoDB）

+   在 MongoDB 中创建我们的 NoSQL 表

+   使用 PySpark 从 MongoDB 中摄取数据

# 技术要求

你可以从本章的 GitHub 仓库 [`github.com/PacktPublishing/Data-Ingestion-with-Python-Cookbook`](https://github.com/PacktPublishing/Data-Ingestion-with-Python-Cookbook) 中找到代码。

使用 **Jupyter Notebook** 不是强制性的，但它允许我们以交互式的方式探索代码。由于我们将执行 Python 和 PySpark 代码，Jupyter 可以帮助我们更好地理解脚本。一旦你安装了 Jupyter，你可以使用以下行来执行它：

```py
$ jupyter notebook
```

建议创建一个单独的文件夹来存储本章中将要介绍的 Python 文件或笔记本；然而，请随意以最适合你的方式组织它们。

# 配置 JDBC 连接

与不同的系统合作带来了找到一种有效方式连接系统的挑战。适配器或驱动程序是解决这种通信问题的解决方案，它创建了一个桥梁来翻译一个系统到另一个系统的信息。

**JDBC**，或 **Java 数据库连接**，用于促进基于 Java 的系统与数据库之间的通信。本食谱涵盖了在 SparkSession 中配置 JDBC 以连接到 PostgreSQL 数据库，一如既往地使用最佳实践。

## 准备工作

在配置 SparkSession 之前，我们需要下载 `.jars` 文件（Java 存档）。你可以在 PostgreSQL 官方网站 [`jdbc.postgresql.org/`](https://jdbc.postgresql.org/) 上完成此操作。

选择 **下载**，你将被重定向到另一个页面：

![图 5.1 – PostgreSQL JDBC 主页](img/Figure_5.1_B19453.jpg)

图 5.1 – PostgreSQL JDBC 主页

然后，选择 **Java 8** **下载** 按钮。

请将此 `.jar` 文件保存在安全的地方，因为你稍后需要它。我建议将其保存在你的代码所在的文件夹中。

对于 PostgreSQL 数据库，你可以使用 Docker 镜像或我们在第四章中创建的实例。如果你选择 Docker 镜像，请确保它已启动并运行。

此菜谱的最终准备步骤是导入一个要使用的数据集。我们将使用 `world_population.csv` 文件（您可以在本书的 GitHub 仓库中找到，位于 [`github.com/PacktPublishing/Data-Ingestion-with-Python-Cookbook/tree/main/Chapter_5/datasets`](https://github.com/PacktPublishing/Data-Ingestion-with-Python-Cookbook/tree/main/Chapter_5/datasets)）。使用 DBeaver 或您选择的任何其他 SQL IDE 导入它。我们将在本章后面的 *使用 SQL 从 JDBC 数据库中获取数据* 菜谱中使用此数据集。

要将数据导入 DBeaver，在 Postgres 数据库下创建一个您选择的表名。我选择给我的表取与 CSV 文件完全相同的名字。目前您不需要插入任何列。

然后，右键单击表并选择 **导入数据**，如下面的截图所示：

![图 5.2 – 使用 DBeaver 在表格中导入数据](img/Figure_5.2_B19453.jpg)

图 5.2 – 使用 DBeaver 在表格中导入数据

将打开一个新窗口，显示使用 CSV 文件或数据库表选项。选择 **CSV** 然后选择 **下一步**，如下所示：

![图 5.3 – 使用 DBeaver 将 CSV 文件导入到表格中](img/Figure_5.3_B19453.jpg)

图 5.3 – 使用 DBeaver 将 CSV 文件导入到表格中

将打开一个新窗口，您可以在其中选择文件。选择 `world_population.csv` 文件并选择 **下一步** 按钮，保留如下所示的默认设置：

![图 5.4 – CSV 文件成功导入到 world_population 表中](img/Figure_5.4_B19453.jpg)

图 5.4 – CSV 文件成功导入到 world_population 表中

如果一切顺利，您应该能够看到 `world_population` 表已填充了列和数据：

![图 5.5 – 已用 CSV 数据填充的 world_population 表](img/Figure_5.5_B19453.jpg)

图 5.5 – 已用 CSV 数据填充的 world_population 表

## 如何操作…

我将使用 Jupyter notebook 来插入和执行代码，使这个练习更加动态。以下是我们的操作方法：

1.  `SparkSession`，我们需要一个额外的类 `SparkConf` 来设置我们的新配置：

    ```py
    from pyspark.conf import SparkConf
    from pyspark.sql import SparkSession
    ```

1.  `SparkConf(`)，我们实例化后，我们可以使用 `spark.jars` 设置 `.jar` 路径：

    ```py
    conf = SparkConf()
    conf.set('spark.jars', /path/to/your/postgresql-42.5.1.jar')
    ```

您将看到创建了一个 `SparkConf` 对象，如下面的输出所示：

![图 5.6 – SparkConf 对象](img/Figure_5.6_B19453.jpg)

图 5.6 – SparkConf 对象

1.  创建 `SparkSession` 并初始化它：

    ```py
    spark = SparkSession.builder \
            .config(conf=conf) \
            .master("local") \
            .appName("Postgres Connection Test") \
            .getOrCreate()
    ```

如果出现如下截图所示的警告信息，您可以忽略它：

![图 5.7 – SparkSession 初始化警告信息](img/Figure_5.7_B19453.jpg)

图 5.7 – SparkSession 初始化警告信息

1.  **连接到我们的数据库**：最后，我们可以通过传递所需的凭据（包括主机、数据库名称、用户名和密码）来连接到 PostgreSQL 数据库，如下所示：

    ```py
    df= spark.read.format("jdbc") \
        .options(url="jdbc:postgresql://localhost:5432/postgres",
                 dbtable="world_population",
                 user="root",
                 password="root",
                 driver="org.postgresql.Driver") \
        .load()
    ```

如果凭据正确，我们应在此处不期望有任何输出。

1.  `.printSchema()`，现在可以查看表列：

    ```py
    df.printSchema()
    ```

执行代码将显示以下输出：

![图 5.8 – world_population 架构的 DataFrame](img/Figure_5.8_B19453.jpg)

图 5.8 – world_population 架构的 DataFrame

## 它是如何工作的…

我们可以观察到 PySpark（以及 Spark）需要额外的配置来与数据库建立连接。在这个配方中，使用 PostgreSQL 的`.jars`文件是使其工作所必需的。

让我们通过查看我们的代码来了解 Spark 需要什么样的配置：

```py
conf = SparkConf()
conf.set('spark.jars', '/path/to/your/postgresql-42.5.1.jar')
```

我们首先实例化了`SparkConf()`方法，该方法负责定义在 SparkSession 中使用的配置。实例化类后，我们使用`set()`方法传递一个键值对参数：`spark.jars`。如果使用了多个`.jars`文件，路径可以通过逗号分隔的值参数传递。也可以定义多个`conf.set()`方法；它们只需要依次包含即可。

如您在以下代码中看到的，配置的集合是在 SparkSession 的第二行传递的：

```py
spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
    (...)
```

然后，随着我们的 SparkSession 实例化，我们可以使用它来读取我们的数据库，如下面的代码所示：

```py
df= spark.read.format("jdbc") \
    .options(url="jdbc:postgresql://localhost:5432/postgres",
             dbtable="world_population",
             user="root",
             password="root",
             driver="org.postgresql.Driver") \
    .load()
```

由于我们正在处理第三方应用程序，我们必须使用`.format()`方法设置读取输出的格式。`.options()`方法将携带认证值和驱动程序。

注意

随着时间的推移，您将观察到声明`.options()`键值对有几种不同的方式。例如，另一种常用的格式是.`options("driver", "org.postgresql.Driver")`。这两种方式都是正确的，取决于开发者的*口味*。

## 还有更多…

这个配方涵盖了如何使用 JDBC 驱动程序，同样的逻辑也适用于**开放数据库连接**（**ODBC**）。然而，确定使用 JDBC 或 ODBC 的标准需要了解我们从哪个数据源摄取数据。

Spark 中的 ODBC 连接通常与 Spark Thrift Server 相关联，这是 Apache HiveServer2 的一个 Spark SQL 扩展，允许用户在**商业智能**（**BI**）工具（如 MS PowerBI 或 Tableau）中执行 SQL 查询。以下图表概述了这种关系：

![图 5.9 – Spark Thrift 架构，由 Cloudera 文档提供（https://docs.cloudera.com/HDPDocuments/HDP3/HDP-3.1.5/developing-spark-applications/content/using_spark_sql.xhtml）](img/Figure_5.9_B19453.jpg)

图 5.9 – Spark Thrift 架构，由 Cloudera 文档提供（[`docs.cloudera.com/HDPDocuments/HDP3/HDP-3.1.5/developing-spark-applications/content/using_spark_sql.xhtml`](https://docs.cloudera.com/HDPDocuments/HDP3/HDP-3.1.5/developing-spark-applications/content/using_spark_sql.xhtml)）

与 JDBC 相比，ODBC 在实际项目中使用较多，这些项目规模较小，且更具体于某些系统集成。它还需要使用另一个名为 `pyodbc` 的 Python 库。您可以在 [`kontext.tech/article/290/connect-to-sql-server-in-spark-pyspark`](https://kontext.tech/article/290/connect-to-sql-server-in-spark-pyspark) 上了解更多信息。

### 调试连接错误

PySpark 错误可能非常令人困惑，并可能导致误解。这是因为错误通常与 JVM 上的问题有关，而 Py4J（一个与 JVM 动态通信的 Python 解释器）将消息与其他可能发生的 Python 错误合并。

一些错误信息在管理数据库连接时很常见，并且很容易识别。让我们看看以下代码使用时发生的错误：

```py
df= spark.read.format("jdbc") \
    .options(url="jdbc:postgresql://localhost:5432/postgres",
             dbtable="world_population",
             user="root",
             password="root") \
    .load()
```

这里是产生的错误信息：

![图 5.10 – Py4JJavaError 信息](img/Figure_5.10_B19453.jpg)

图 5.10 – Py4JJavaError 信息

在第一行，我们看到 `Py4JJavaError` 通知我们在调用加载函数时出现错误。继续到第二行，我们可以看到消息：`java.sql.SQLException: No suitable driver`。它通知我们，尽管 `.jars` 文件已配置并设置，但 PySpark 仍然不知道使用哪个驱动程序来从 PostgreSQL 加载数据。这可以通过在 `.options()` 下添加 `driver` 参数来轻松解决。请参考以下代码：

```py
df= spark.read.format("jdbc") \
    .options(url="jdbc:postgresql://localhost:5432/postgres",
             dbtable="world_population",
             user="root",
             password="root",
             driver="org.postgresql.Driver") \
    .load()
```

## 参见

了解更多关于 Spark Thrift 服务器的信息，请访问 [`jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-thrift-server.xhtml`](https://jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-thrift-server.xhtml)。

# 使用 SQL 从 JDBC 数据库中摄取数据

在测试连接并配置 SparkSession 后，下一步是从 PostgreSQL 中摄取数据，过滤它，并将其保存为称为 Parquet 文件的分析格式。现在不用担心 Parquet 文件是如何工作的；我们将在接下来的章节中介绍。

这个菜谱旨在使用我们与 JDBC 数据库创建的连接，并从 `world_population` 表中摄取数据。

## 准备工作

这个菜谱将使用与 *配置 JDBC 连接* 菜谱相同的数据集和代码来连接到 PostgreSQL 数据库。请确保您的 Docker 容器正在运行或您的 PostgreSQL 服务器已启动。

这个菜谱从 *配置 JDBC 连接* 中继续。我们现在将学习如何摄取 Postgres 数据库内部的数据。

## 如何做到这一点...

在我们之前的代码基础上，让我们按照以下方式读取数据库中的数据：

1.  `world_population` 表：

    ```py
    df= spark.read.format("jdbc") \
        .options(url="jdbc:postgresql://localhost:5432/postgres",
                 dbtable="world_population",
                 user="root",
                 password="root",
                 driver="org.postgresql.Driver") \
        .load()
    ```

1.  **创建临时视图**：为了组织目的，我们使用表的准确名称，从 DataFrame 中在 Spark 默认数据库中创建一个临时视图：

    ```py
    df.createOrReplaceTempView("world_population")
    ```

这里不应该有输出。

1.  `spark` 变量：

    ```py
    spark.sql("select * from world_population").show(3)
    ```

根据您的显示器大小，输出可能看起来很混乱，如下所示：

![图 5.11 – 使用 Spark SQL 的 world_population 视图](img/Figure_5.11_B19453.jpg)

图 5.11 – 使用 Spark SQL 的 world_population 视图

1.  **过滤数据**：使用 SQL 语句，让我们仅过滤 DataFrame 中的南美国家：

    ```py
    south_america = spark.sql("select * from world_population where continent = 'South America' ")
    ```

由于我们将结果分配给一个变量，因此没有输出。

1.  `.toPandas()` 函数以更用户友好的方式查看：

    ```py
    south_america.toPandas()
    ```

这就是结果呈现的样子：

![图 5.12 – 使用 toPandas() 可视化的 south_america 国家](img/Figure_5.12_B19453.jpg)

图 5.12 – 使用 toPandas() 可视化的 south_america 国家

1.  **保存我们的工作**：现在，我们可以如下保存我们的过滤数据：

    ```py
    south_america.write.parquet('south_america_population')
    ```

查看您的脚本文件夹，您应该看到一个名为 `south_america_population` 的文件夹。在里面，您应该看到以下输出：

![图 5.13 – Parquet 文件中的 south_america 数据](img/Figure_5.13_B19453.jpg)

图 5.13 – Parquet 文件中的 south_america 数据

这是我们的过滤和导入的 DataFrame，以分析格式呈现。

## 工作原理...

与 Spark 一起工作的一个显著优势是使用 SQL 语句从 DataFrame 中过滤和查询数据。这允许数据分析和 BI 团队通过处理查询来帮助数据工程师。这有助于构建分析数据并将其插入数据仓库。

然而，我们需要注意一些事项来正确执行 SQL 语句。其中之一是使用 `.createOrReplaceTempView()`，如代码中的这一行所示：

```py
df.createOrReplaceTempView("world_population")
```

在幕后，这个临时视图将作为一个 SQL 表来工作，并从 DataFrame 中组织数据，而不需要物理文件。

然后，我们使用实例化的 `SparkSession` 变量来执行 SQL 语句。请注意，表名与临时视图的名称相同：

```py
spark.sql("select * from world_population").show(3)
```

在执行所需的 SQL 查询后，我们继续使用 `.write()` 方法保存我们的文件，如下所示：

```py
south_america.write.parquet('south_america_population')
```

`parquet()` 方法内部的参数定义了文件的路径和名称。在写入 Parquet 文件时，还有其他一些配置可用，我们将在后面的 *第七章* 中介绍。

## 更多内容...

虽然我们使用了临时视图来编写我们的 SQL 语句，但也可以使用 DataFrame 中的过滤和聚合函数。让我们通过仅过滤南美国家来使用本食谱中的示例：

```py
df.filter(df['continent'] == 'South America').show(10)
```

您应该看到以下输出：

![图 5.14 – 使用 DataFrame 操作过滤的南美国家](img/Figure_5.14_B19453.jpg)

图 5.14 – 使用 DataFrame 操作过滤的南美国家

重要的是要理解并非所有 SQL 函数都可以用作 DataFrame 操作。您可以在 [`spark.apache.org/docs/2.2.0/sql-programming-guide.xhtml`](https://spark.apache.org/docs/2.2.0/sql-programming-guide.xhtml) 找到更多使用 DataFrame 操作进行过滤和聚合函数的实用示例。

## 参见

*TowardsDataScience* 有关于使用 PySpark 的 SQL 函数的精彩博客文章，请参阅 [`towardsdatascience.com/pyspark-and-sparksql-basics-6cb4bf967e53`](https://towardsdatascience.com/pyspark-and-sparksql-basics-6cb4bf967e53)。

# 连接到 NoSQL 数据库（MongoDB）

MongoDB 是一个用 C++ 编写的开源、非结构化、面向文档的数据库。它在数据界因其可扩展性、灵活性和速度而闻名。

对于将处理数据（或者可能已经正在处理数据）的人来说，了解如何探索 MongoDB（或任何其他非结构化）数据库是至关重要的。MongoDB 有一些独特的特性，我们将在实践中进行探索。

在这个菜谱中，您将学习如何创建连接以通过 Studio 3T Free 访问 MongoDB 文档，这是一个 MongoDB 图形用户界面。

## 准备工作

为了开始使用这个强大的数据库，我们首先需要在本地机器上安装并创建一个 MongoDB 服务器。我们已经在 *第一章* 中配置了一个 MongoDB Docker 容器，所以让我们启动它。您可以使用 Docker Desktop 或通过以下命令在命令行中完成此操作：

```py
my-project/mongo-local$ docker run \
--name mongodb-local \
-p 27017:27017 \
-e MONGO_INITDB_ROOT_USERNAME=<your_username> \
-e MONGO_INITDB_ROOT_PASSWORD=<your_password>\
-d mongo:latest
```

不要忘记使用您选择的用户名和密码更改变量。

在 Docker Desktop 上，您应该看到以下内容：

![图 5.15 – 运行中的 MongoDB Docker 容器](img/Figure_5.15_B19453.jpg)

图 5.15 – 运行中的 MongoDB Docker 容器

下一步是下载并配置 Studio 3T Free，这是开发社区用来连接到 MongoDB 服务器的免费软件。您可以从 [`studio3t.com/download-studio3t-free`](https://studio3t.com/download-studio3t-free) 下载此软件，并按照安装程序为您的操作系统提供的步骤进行操作。

在安装过程中，可能会出现类似于以下图所示的提示信息。如果是这样，您可以留空字段。我们不需要为本地或测试目的进行密码加密。

![图 5.16 – Studio 3T Free 密码加密消息](img/Figure_5.16_B19453.jpg)

图 5.16 – Studio 3T Free 密码加密消息

安装过程完成后，您将看到以下窗口：

![图 5.17 – Studio 3T Free 连接窗口](img/Figure_5.17_B19453.jpg)

图 5.17 – Studio 3T Free 连接窗口

现在我们已准备好将我们的 MongoDB 实例连接到 IDE。

## 如何做到这一点…

现在我们已经安装了 Studio 3T，让我们连接到我们的本地 MongoDB 实例：

1.  **创建连接**：在您打开 Studio 3T 后，将出现一个窗口，要求您输入连接字符串或手动配置它。选择第二个选项，然后点击 **下一步**。

您将看到类似以下内容：

![图 5.18 – Studio 3T 新连接初始选项](img/Figure_5.18_B19453.jpg)

图 5.18 – Studio 3T 新连接初始选项

1.  `localhost` 和 `27017`。

目前您的屏幕应该看起来如下：

![图 5.19 – 新连接服务器信息](img/Figure_5.19_B19453.jpg)

图 5.19 – 新连接服务器信息

现在在**连接组**字段下选择**身份验证**选项卡，然后从**身份验证模式**下拉菜单中选择**基本**。

将出现三个字段——在**身份验证** **数据库**字段中为`admin`。

![图 5.20 – 新连接身份验证信息](img/Figure_5.20_B19453.jpg)

图 5.20 – 新连接身份验证信息

1.  **测试我们的连接**：使用此配置，我们应该能够测试我们的数据库连接。在左下角，选择**测试** **连接**按钮。

如果您提供的凭据正确，您将看到以下输出：

![图 5.21 – 测试连接成功](img/Figure_5.21_B19453.jpg)

图 5.21 – 测试连接成功

点击**保存**按钮，窗口将关闭。

1.  **连接到我们的数据库**：在保存我们的配置后，将出现一个包含可用连接的窗口，包括我们新创建的连接：

![图 5.22 – 创建连接后的连接管理器](img/Figure_5.22_B19453.jpg)

图 5.22 – 创建连接后的连接管理器

选择**连接**按钮，将出现三个默认数据库：**admin**、**config**和**local**，如下截图所示：

![图 5.23 – 服务器上默认数据库的本地 MongoDB 主页](img/Figure_5.23_B19453.jpg)

图 5.23 – 服务器上默认数据库的本地 MongoDB 主页

我们现在已经完成了 MongoDB 的配置，并准备好进行本章以及其他章节的后续食谱，包括**第六章**、**第十一章**和**第十二章**。

## 工作原理…

就像可用的数据库一样，通过 Docker 容器创建和运行 MongoDB 是直接了当的。检查以下命令：

```py
my-project/mongo-local$ docker run \
--name mongodb-local \
-p 27017:27017 \
-e MONGO_INITDB_ROOT_USERNAME=<your_username> \
-e MONGO_INITDB_ROOT_PASSWORD=<your_password>\
-d mongo:latest
```

正如我们在**准备就绪**部分所看到的，需要传递的最关键信息是用户名和密码（使用 `-e` 参数），连接的端口（使用 `-p` 参数），以及容器镜像版本，这是最新可用的。

连接到 Studio 3T Free 的 MongoDB 容器的架构甚至更加简单。一旦连接端口可用，我们就可以轻松访问数据库。您可以在以下架构表示中看到：

![图 5.24 – 使用 Docker 镜像连接到 Studio 3T Free 的 MongoDB](img/Figure_5.24_B19453.jpg)

图 5.24 – 使用 Docker 镜像连接到 Studio 3T Free 的 MongoDB

如本食谱开头所述，MongoDB 是一个面向文档的数据库。其结构类似于 JSON 文件，除了每一行都被解释为一个文档，并且每个文档都有自己的 `ObjectId`，如下所示：

![图 5.25 – MongoDB 文档格式](img/Figure_5.25_B19453.jpg)

图 5.25 – MongoDB 文档格式

文档的集合被称为**集合**，这可以更好地理解为结构化数据库中的表表示。您可以在以下架构中看到它是如何按层次结构组织的：

![图 5.26 – MongoDB 数据结构](img/Figure_5.26_B19453.jpg)

图 5.26 – MongoDB 数据结构

正如我们在使用 Studio 3T Free 登录 MongoDB 时观察到的，有三个默认数据库：`admin`、`config`和`local`。目前，让我们忽略最后两个，因为它们与数据的操作工作有关。`admin`数据库是由`root`用户创建的主要数据库。这就是为什么我们在*步骤 3*中提供了这个数据库作为**认证数据库**选项的原因。

创建一个用户以导入数据并访问特定的数据库或集合通常是被推荐的。然而，为了演示目的，我们在这里以及本书接下来的食谱中将继续使用 root 访问权限。

## 更多内容…

连接字符串将根据你的 MongoDB 服务器配置而有所不同。例如，当存在*副本集*或*分片集群*时，我们需要指定我们想要连接到哪些实例。

注意

分片集群是一个复杂且有趣的话题。你可以在 MongoDB 的官方文档中了解更多关于这个话题的内容，并深入探讨[`www.mongodb.com/docs/manual/core/sharded-cluster-components/`](https://www.mongodb.com/docs/manual/core/sharded-cluster-components/)。

让我们看看一个使用基本认证模式的独立服务器字符串连接的例子：

```py
mongodb://mongo-server-user:some_password@mongo-host01.example.com:27017/?authSource=admin
```

如你所见，它与其他数据库连接类似。如果我们想要连接到本地服务器，我们将主机更改为`localhost`。

现在，对于副本集或分片集群，字符串连接看起来是这样的：

```py
mongodb://mongo-server-user:some_password@mongo-host01.example.com:27017, mongo-host02.example.com:27017, mongo-hosta03.example.com:27017/?authSource=admin
```

在这个 URI 中的`authSource=admin`参数对于通知 MongoDB 我们想要使用数据库的管理员用户进行认证是必不可少的。没有它，将会引发错误或认证问题，如下面的输出所示：

```py
MongoError: Authentication failed
```

避免这种错误的另一种方式是创建一个特定的用户来访问数据库和集合。

### SRV URI 连接

MongoDB 引入了**域名系统**（**DNS**）种子列表连接，它通过 DNS 中的**服务记录**（**SRV**）规范构建数据，以尝试解决这种冗长的字符串。我们在本食谱的第一步中看到了使用 SRV URI 配置 MongoDB 连接的可能性。

下面是一个例子：

```py
mongodb+srv://my-server.example.com/
```

它与我们在之前看到的标准连接字符串格式相似。然而，我们需要在开头指示使用 SRV，然后提供 DNS 条目。

当处理副本或节点时，这种类型的连接具有优势，因为 SRV 为集群创建了一个单一的身份。你可以在 MongoDB 官方文档中找到更详细的解释，以及如何配置它的概述，请参阅[`www.mongodb.com/docs/manual/reference/connection-string/#dns-seed-list-connection-format`](https://www.mongodb.com/docs/manual/reference/connection-string/#dns-seed-list-connection-format)。

## 参见

如果您感兴趣，市场上还有其他 MongoDB 图形用户界面工具可供选择：[`www.guru99.com/top-20-mongodb-tools.xhtml`](https://www.guru99.com/top-20-mongodb-tools.xhtml)。

# 在 MongoDB 中创建我们的 NoSQL 表

在成功连接并了解 Studio 3T 的工作原理后，我们现在将导入一些 MongoDB 集合。我们在*连接到 NoSQL 数据库（MongoDB）*的配方中看到了如何开始使用 MongoDB，在这个配方中，我们将导入一个 MongoDB 数据库并了解其结构。尽管 MongoDB 有特定的格式来组织内部数据，但在处理数据摄取时了解 NoSQL 数据库的行为至关重要。

我们将在本章的以下配方中通过摄取导入的集合进行练习。

## 准备工作

对于这个配方，我们将使用一个名为`listingsAndReviews.json`的 Airbnb 评论样本数据集。您可以在本书的 GitHub 存储库中找到此数据集：[`github.com/PacktPublishing/Data-Ingestion-with-Python-Cookbook/tree/main/Chapter_5/datasets/sample_airbnb`](https://github.com/PacktPublishing/Data-Ingestion-with-Python-Cookbook/tree/main/Chapter_5/datasets/sample_airbnb)。下载后，将文件放入我们创建的`mongo-local`目录中，如*第一章*所示。

我将我的数据库放在`sample_airbnb`文件夹中，只是为了组织目的，如以下截图所示：

![图 5.27 – 带有 listingsAndReviews.json 的命令行](img/Figure_5.27_B19453.jpg)

图 5.27 – 带有 listingsAndReviews.json 的命令行

下载数据集后，我们需要安装`pymongo`，这是一个用于连接和管理 MongoDB 操作的 Python 库。要安装它，请使用以下命令：

```py
$ pip3 install pymongo
```

您可以自由地为这个安装创建`virtualenv`。

现在我们已经准备好开始向 MongoDB 中插入数据。在我们开始之前，别忘了检查您的 Docker 镜像是否正在运行。

## 如何操作...

执行此配方的步骤如下：

1.  使用`pymongo`，我们可以轻松地与 MongoDB 数据库建立连接。请参考以下代码：

    ```py
    import json
    import os
    from pymongo import MongoClient, InsertOne
    mongo_client = pymongo.MongoClient("mongodb://root:root@localhost:27017/")
    ```

1.  **定义我们的数据库和集合**：我们将使用实例化的客户端连接创建数据库和集合实例。

对于`json_collection`变量，插入您放置 Airbnb 样本数据集的路径：

```py
db_cookbook = mongo_client.db_airbnb
collection = db_cookbook.reviews
json_collection = "sample_airbnb/listingsAndReviews.json"
```

1.  使用`bulk_write`函数，我们将把 JSON 文件中的所有文档插入到我们创建的销售集合中，并关闭连接：

    ```py
    requesting_collection = []
    with open(json_collection) as f:
        for object in f:
            my_dict = json.loads(object)
            requesting.append(InsertOne(my_dict))
    result = collection.bulk_write(requesting_collection)
    mongo_client.close()
    ```

此操作不应有任何输出，但我们可以检查数据库以查看操作是否成功。

1.  **检查 MongoDB 数据库结果**：让我们检查我们的数据库，看看数据是否已正确插入。

打开 Studio 3T Free 并刷新连接（右键单击连接名称并选择**刷新所有**）。您应该会看到一个名为**db_airbnb**的新数据库已创建，其中包含一个**reviews**集合，如以下截图所示：

![图 5.28 – 在 MongoDB 上成功创建数据库和集合](img/Figure_5.28_B19453.jpg)

图 5.28 – 在 MongoDB 上成功创建数据库和集合

现在集合已经创建并包含了一些数据，让我们更深入地了解代码是如何工作的。

## 它是如何工作的...

如您所见，我们实现的代码非常简单，只用几行代码就能在我们的数据库中创建和插入数据。然而，由于 MongoDB 的特殊性，有一些重要的点需要注意。

现在我们逐行检查代码：

```py
mongo_client = pymongo.MongoClient("mongodb://root:root@localhost:27017/")
```

这行代码定义了与我们的 MongoDB 数据库的连接，并且从这个实例，我们可以创建一个新的数据库及其集合。

注意

注意 URI 连接包含用户名和密码的硬编码值。在实际应用中，甚至在开发服务器上都必须避免这样做。建议将这些值存储为环境变量或使用秘密管理器保险库。

接下来，我们定义数据库和集合的名称；您可能已经注意到我们之前没有在数据库中创建它们。在代码执行时，MongoDB 会检查数据库是否存在；如果不存在，MongoDB 将创建它。同样的规则适用于 **reviews** 集合。

注意集合是从 `db_cookbook` 实例派生的，这使得它明确地与 `db_airbnb` 数据库相关联：

```py
db_cookbook = mongo_client.db_airbnb
collection = db_cookbook.reviews
```

按照代码，下一步是打开 JSON 文件并解析每一行。这里我们开始看到 MongoDB 的一些棘手的特殊性：

```py
requesting_collection = []
with open(json_collection) as f:
    for object in f:
        my_dict = json.loads(object)
        requesting_collection.append(InsertOne(my_dict))
```

人们常常想知道为什么我们实际上需要解析 JSON 行，因为 MongoDB 接受这种格式。让我们检查下面的截图中的 `listingsAndReviews.json` 文件：

![图 5.29 – 包含 MongoDB 文档行的 JSON 文件](img/Figure_5.29_B19453.jpg)

图 5.29 – 包含 MongoDB 文档行的 JSON 文件

如果我们使用任何工具来验证这是有效的 JSON，它肯定会说这不是有效的格式。这是因为这个文件的每一行代表 MongoDB 集合中的一个文档。仅使用传统的 `open()` 和 `json.loads()` 方法尝试打开该文件将产生如下错误：

```py
json.decoder.JSONDecodeError: Extra data: line 2 column 1 (char 190)
```

为了使 Python 解释器能够接受，我们需要逐行打开和读取并追加到 `requesting_collection` 列表中。此外，`InsertOne()` 方法将确保每行单独插入。在插入特定行时出现的问题将更容易识别。

最后，`bulk_write()` 将文档列表插入 MongoDB 数据库：

```py
result = collection.bulk_write(requesting_collection)
```

如果一切正常，这个操作将完成而不会返回任何输出或错误信息。

## 还有更多...

我们已经看到创建一个 Python 脚本来将数据插入我们的 MongoDB 服务器是多么简单。尽管如此，MongoDB 提供了数据库工具来提供相同的结果，并且可以通过命令行执行。`mongoimport` 命令用于将数据插入我们的数据库，如下面的代码所示：

```py
mongoimport --host localhost --port 27017-d db_name -c collection_name --file path/to/file.json
```

如果你想了解更多关于其他数据库工具和命令的信息，请查看官方 MongoDB 文档，链接为[`www.mongodb.com/docs/database-tools/installation/installation/`](https://www.mongodb.com/docs/database-tools/installation/installation/).

### 字段名称的限制

当将数据加载到 MongoDB 时，一个主要问题是对字段名称中使用的字符的限制。由于 MongoDB 服务器版本或编程语言特定性，有时字段的键名会带有`$`前缀，并且默认情况下，MongoDB 与它不兼容，会创建如下错误输出：

```py
localhost:27017: $oid is not valid for storage.
```

在这种情况下，从 MongoDB 服务器导出了一个 JSON 文件，`ObjectID`的引用带有`$`前缀。尽管 MongoDB 的最新版本已经开始接受这些字符（见此处线程：[`jira.mongodb.org/browse/SERVER-41628?fbclid=IwAR1t5Ld58LwCi69SrMCcDbhPGf2EfBWe_AEurxGkEWHpZTHaEIde0_AZ-uM%5D`](https://jira.mongodb.org/browse/SERVER-41628?fbclid=IwAR1t5Ld58LwCi69SrMCcDbhPGf2EfBWe_AEurxGkEWHpZTHaEIde0_AZ-uM%5D)），但在可能的情况下避免使用它们是一个好的做法。在这种情况下，我们有两个主要选项：使用脚本删除所有受限字符，或者将 JSON 文件编码成**二进制 JavaScript 对象表示法**（**BSON**）文件。你可以在[`kb.objectrocket.com/mongo-db/how-to-use-python-to-encode-a-json-file-into-mongodb-bson-documents-545`](https://kb.objectrocket.com/mongo-db/how-to-use-python-to-encode-a-json-file-into-mongodb-bson-documents-545)了解更多关于将文件编码成 BSON 格式的信息。

## 参见

你可以在[`www.mongodb.com/docs/manual/reference/limits/#mongodb-limit-Restrictions-on-Field-Names`](https://www.mongodb.com/docs/manual/reference/limits/#mongodb-limit-Restrictions-on-Field-Names)了解更多关于 MongoDB 对字段名称的限制。

# 使用 PySpark 从 MongoDB 导入数据

虽然自己创建和导入数据似乎不太实际，但这个练习可以应用于实际项目。与数据打交道的人通常参与定义数据库类型的架构过程，帮助其他工程师将应用程序中的数据插入到数据库服务器中，然后仅导入仪表板或其他分析工具的相关信息。

到目前为止，我们已经创建并评估了我们的服务器，然后在 MongoDB 实例内部创建了集合。有了所有这些准备工作，我们现在可以使用 PySpark 导入我们的数据。

## 准备工作

这个菜谱需要执行*在 MongoDB 中创建我们的 NoSQL 表*菜谱，因为需要数据插入。然而，你可以创建并插入其他文档到 MongoDB 数据库中，并在此处使用它们。如果你这样做，确保设置合适的配置以使其正常运行。

同样，正如在*在 MongoDB 中创建我们的 NoSQL 表*食谱中一样，检查 Docker 容器是否正在运行，因为这是我们 MongoDB 实例的主要数据源。让我们继续进行数据摄取！

![图 5.30 – 运行中的 MongoDB Docker 容器](img/Figure_5.30._B19453.jpg)

图 5.30 – 运行中的 MongoDB Docker 容器

## 如何做…

要尝试这个食谱，你需要执行以下步骤：

1.  `SparkSession`，但这次传递读取我们的 MongoDB 数据库`db_airbnb`的特定配置，例如 URI 和`.jars`：

    ```py
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
          .master("local[1]") \
          .appName("MongoDB Ingest") \
          .config("spark.executor.memory", '3g') \
          .config("spark.executor.cores", '1') \
          .config("spark.cores.max", '1') \
          .config("spark.mongodb.input.uri", "mongodb://root:root@127.0.0.1/db_airbnb?authSource=admin&readPreference=primaryPreferred") \
          .config("spark.mongodb.input.collection", "reviews") \
          .config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
          .getOrCreate()
    ```

我们应该期望这里有一个显著的结果，因为 Spark 下载了包并设置了我们传递的其余配置：

![图 5.31 – 使用 MongoDB 配置初始化 SparkSession](img/Figure_5.31._B19453.jpg)

图 5.31 – 使用 MongoDB 配置初始化 SparkSession

1.  我们实例化的`SparkSession`。在这里，我们不应该期望有任何输出，因为`SparkSession`仅设置为在`WARN`级别发送日志：

    ```py
    df = spark.read.format("mongo").load()
    ```

1.  **获取我们的 DataFrame 模式**：我们可以通过在 DataFrame 上执行打印操作来查看集合的模式：

    ```py
    df.printSchema()
    ```

你应该观察到以下输出：

![图 5.32 – Reviews DataFrame 集合模式打印](img/Figure_5.32._B19453.jpg)

图 5.32 – Reviews DataFrame 集合模式打印

如你所见，结构类似于一个嵌套对象的 JSON 文件。非结构化数据通常以这种形式呈现，并且可以包含大量信息，以便创建一个 Python 脚本来将数据插入我们的数据中。现在，让我们更深入地了解我们的代码。

## 它是如何工作的…

MongoDB 在`SparkSession`中需要一些额外的配置来执行`.read`函数。理解为什么我们使用配置而不是仅仅使用文档中的代码是至关重要的。让我们来探索相关的代码：

```py
config("spark.mongodb.input.uri", "mongodb://root:root@127.0.0.1/db_airbnb?authSource=admin) \
```

注意使用`spark.mongodb.input.uri`，它告诉我们的`SparkSession`需要进行一个使用 MongoDB URI 的*读取*操作。如果我们想进行*写入*操作（或两者都进行），我们只需要添加`spark.mongodb.output.uri`配置。

接下来，我们传递包含用户和密码信息、数据库名称和认证源的 URI。由于我们使用 root 用户检索数据，因此最后一个参数设置为`admin`。

接下来，我们定义在读取操作中使用的集合名称：

```py
.config("spark.mongodb.input.collection", "reviews")\
```

注意

即使在 SparkSession 中定义这些参数看起来可能有些奇怪，并且可以设置数据库和集合，但这是一种社区在操作 MongoDB 连接时采用的良好实践。

```py
.config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")
```

在这里还有一个新的配置是`spark.jars.packages`。当使用这个键与`.config()`方法一起时，Spark 将搜索其可用的在线包，下载它们，并将它们放置在`.jar`文件夹中以供使用。虽然这是一种设置`.jar`连接器的有利方式，但这并不是所有数据库都支持的。

一旦建立连接，读取过程与 JDBC 非常相似：我们传递数据库的 `.format()`（这里为 `mongo`），由于数据库和集合名称已经设置，我们不需要配置 `.option()`：

```py
df = spark.read.format("mongo").load()
```

当执行 `.load()` 时，Spark 将验证连接是否有效，如果无效则抛出错误。在下面的截图中，你可以看到一个错误信息的示例，当凭证不正确时：

![图 5.33 – Py4JJavaError：MongoDB 连接认证错误](img/Figure_5.33._B19453.jpg)

图 5.33 – 当 MongoDB 包未在配置中设置时引发的 Py4JJavaError：MongoDB 连接认证错误

尽管我们正在处理非结构化数据格式，但一旦 PySpark 将我们的集合转换为 DataFrame，所有过滤、清理和操作数据的过程基本上与 PySpark 数据相同。

## 更多内容...

正如我们之前看到的，PySpark 错误信息可能会让人困惑，并且一开始看起来会让人不舒服。让我们探索在没有适当配置的情况下从 MongoDB 数据库中摄取数据时出现的其他常见错误。

在这个例子中，让我们不要在 `SparkSession` 配置中设置 `spark.jars.packages`：

```py
spark = SparkSession.builder \
      (...)
      .config("spark.mongodb.input.uri", "mongodb://root:root@127.0.0.1/db_aibnb?authSource=admin") \
      .config("spark.mongodb.input.collection", "reviews")
     .getOrCreate()
df = spark.read.format("mongo").load()
```

如果你尝试执行前面的代码（传递剩余的内存设置），你将得到以下输出：

![图 5.34 – 当 MongoDB 包未在配置中设置时引发的 java.lang.ClassNotFoundException 错误](img/Figure_5.34._B19453.jpg)

图 5.34 – 当 MongoDB 包未在配置中设置时引发的 java.lang.ClassNotFoundException 错误

仔细查看以 `java.lang.ClassNotFoundException` 开头的第二行，JVM 强调了一个需要搜索第三方存储库的缺失包或类。该包包含连接到我们的 JVM 的连接器代码，并建立与数据库服务器的通信。

另一个常见的错误信息是 `IllegalArgumentException`。此类错误向开发者表明，方法或类中传递了一个错误的参数。通常，当与数据库连接相关时，它指的是无效的字符串连接，如下面的截图所示：

![图 5.35 – 当 URI 无效时引发的 IllegalArgumentException 错误](img/Figure_5.35._B19453.jpg)

图 5.35 – 当 URI 无效时引发的 IllegalArgumentException 错误

虽然看起来不太清楚，但 URI 中存在一个拼写错误，其中 `db_aibnb/?` 包含一个额外的正斜杠。移除它并再次运行 `SparkSession` 将会使此错误消失。

注意

建议在重新定义 SparkSession 配置时关闭并重新启动内核进程，因为 SparkSession 倾向于向进程追加而不是替换它们。

## 参见

+   MongoDB Spark 连接器文档：[`www.mongodb.com/docs/spark-connector/current/configuration/`](https://www.mongodb.com/docs/spark-connector/current/configuration/)

+   您可以在 MongoDB 文档中找到 MongoDB 连接器与 PySpark 交互的完整说明：[`www.mongodb.com/docs/spark-connector/current/read-from-mongodb/`](https://www.mongodb.com/docs/spark-connector/current/read-from-mongodb/)

+   这里也有一些 MongoDB 的有趣用例：[`www.mongodb.com/use-cases`](https://www.mongodb.com/use-cases)

# 进一步阅读

+   [`www.talend.com/resources/structured-vs-unstructured-data/`](https://www.talend.com/resources/structured-vs-unstructured-data/)

+   [`careerfoundry.com/en/blog/data-analytics/structured-vs-unstructured-data/`](https://careerfoundry.com/en/blog/data-analytics/structured-vs-unstructured-data/)

+   [`www.dba-ninja.com/2022/04/is-mongodbsrv-necessary-for-a-mongodb-connection.xhtml`](https://www.dba-ninja.com/2022/04/is-mongodbsrv-necessary-for-a-mongodb-connection.xhtml)

+   [`www.mongodb.com/docs/manual/reference/connection-string/#connection-string-options`](https://www.mongodb.com/docs/manual/reference/connection-string/#connection-string-options)

+   [`sparkbyexamples.com/spark/spark-createorreplacetempview-explained/`](https://sparkbyexamples.com/spark/spark-createorreplacetempview-explained/)
