# 第三章：3 从数据库中读取时间序列数据

## 加入我们的书籍社区，加入 Discord

![](img/file0.png)

[`packt.link/zmkOY`](https://packt.link/zmkOY)

**数据库**扩展了你可以存储的内容，包括文本、图像和媒体文件，并且被设计用于在大规模下进行高效的读写操作。数据库能够存储数 TB 甚至 PB 的数据，并提供高效优化的数据检索功能，例如在执行分析操作时用于**数据仓库**和**数据湖**。数据仓库是一个专门设计用于存储大量结构化数据的数据库，数据大多来自多个源系统的集成，专为支持商业智能报告、仪表板和高级分析而构建。另一方面，数据湖以原始格式存储大量结构化、半结构化或非结构化数据。本章中，我们将继续使用**pandas**库从数据库中读取数据。我们将通过读取**关系型**（SQL）数据库和**非关系型**（NoSQL）数据库的数据来创建时间序列 DataFrame。

此外，你将探索如何与第三方数据提供商合作，从他们的数据库系统中拉取金融数据。

在本章中，你将创建带有 `DatetimeIndex` 数据类型的时间序列 DataFrame，并涉及以下食谱：

+   从关系型数据库中读取数据

+   从 Snowflake 中读取数据

+   从文档数据库中读取数据

+   从时间序列数据库中读取数据

## 技术要求

在本章中，我们将广泛使用 pandas 2.2.2（于 2024 年 4 月 10 日发布）。

你将与不同类型的数据库一起工作，例如 PostgreSQL、Amazon Redshift、MongoDB、InfluxDB 和 Snowflake。你需要安装额外的 Python 库以连接到这些数据库。

你还可以从本书的 GitHub 仓库下载 Jupyter 笔记本（[`github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook`](https://github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook)）来进行跟随练习。

作为一个良好的实践，你会将你的数据库凭据存储在 Python 脚本之外的 `database.cfg` 配置文件中。你可以使用 `configparser` 来读取并将值存储在 Python 变量中。你不希望凭据暴露或硬编码在代码中：

```py
# Example of configuration file "database.cfg file"
[SNOWFLAKE]
user=username
password=password
account=snowflakeaccount
warehouse=COMPUTE_WH
database=SNOWFLAKE_SAMPLE_DATA
schema=TPCH_SF1
role=somerole
[POSTGRESQL]
host: 127.0.0.1
dbname: postgres
user: postgres
password: password
[AWS]
host=<your_end_point.your_region.redshift.amazonaws.com>
port=5439
database=dev
username=<your_username>
password=<your_password>
```

你可以使用 `config.read()` 加载 `database.cfg` 文件：

```py
import configparser
config = configparser.ConfigParser()
config.read(database.cfg')
```

## 从关系型数据库中读取数据

在这个食谱中，你将从 PostgreSQL 读取数据，这是一个流行的开源关系型数据库。

你将探索两种连接并与 PostgreSQL 交互的方法。首先，你将使用 `psycopg`，这是一个 PostgreSQL Python 连接器，来连接并查询数据库，然后将结果解析到 pandas DataFrame 中。第二种方法是再次查询相同的数据库，但这次你将使用**SQLAlchemy**，一个与 pandas 集成良好的**对象关系映射器**（**ORM**）。

### 准备工作

在这个食谱中，假设你已经安装了最新版本的 PostgreSQL。在写这篇文章时，版本 16 是最新的稳定版本。

要在 Python 中连接和查询数据库，你需要安装 `psycopg`，它是一个流行的 PostgreSQL 数据库适配器。你还需要安装 `SQLAlchemy`，它提供了灵活性，可以根据你希望管理数据库的方式（无论是写入还是读取数据）来进行选择。

要使用 `conda` 安装库，运行以下命令：

```py
>>> conda install -c conda-forge psycopg sqlalchemy -y
```

要使用 `pip` 安装库，运行以下命令：

```py
>>> pip install sqlalchemy psycopg
```

如果你无法访问 PostgreSQL 数据库，最快的方式是通过 Docker ([`hub.docker.com/_/postgres`](https://hub.docker.com/_/postgres)) 来启动。以下是一个示例命令：

```py
docker run -d \
    --name postgres-ch3 \
        -p 5432:5432 \
    -e POSTGRES_PASSWORD=password \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    postgres:16.4-alpine
```

这将创建一个名为 `postgres-ch3` 的容器。`username` 为 `postgres`，密码是 `password`。创建的默认 `database` 名为 `postgres`。

一旦**postgres-ch3**容器启动并运行，你可以使用**DBeaver**进行连接，如下所示：

![图 3.1 – DBeaver PostgreSQL 连接设置](img/file22.png)

图 3.1 – DBeaver PostgreSQL 连接设置

你将使用存放在 `datasets/Ch3/MSFT.csv` 文件夹中的 MSFT 股票数据集。我已经通过**DBeaver Community Edition**上传了数据集到数据库，你可以在这里下载 [`dbeaver.io/download/`](https://dbeaver.io/download/)

你可以通过右键点击 `public` 模式下的 `tables` 来导入 CSV 文件，如下图所示：

![图 3.2 – 使用 DBeaver 导入数据到 PostgreSQL](img/file23.png)

图 3.2 – 使用 DBeaver 导入数据到 PostgreSQL

你可以确认所有记录已被写入 `postgres` 数据库中的 `msft` 表，如下所示。

![图 3.3 – 在 DBeaver 中使用 SQL 编辑器运行 SQL 查询，以查询 msft 表](img/file24.png)

图 3.3 – 在 DBeaver 中使用 SQL 编辑器运行 SQL 查询，以查询 msft 表

### 如何操作…

我们将首先连接到 PostgreSQL 实例，查询数据库，将结果集加载到内存中，并将数据解析为时间序列 DataFrame。

在这个食谱中，我将连接到一个本地运行的 PostgreSQL 实例，因此我的连接将是 `localhost (127.0.0.1)`。你需要根据你自己的 PostgreSQL 数据库设置来调整此连接。

#### 使用 psycopg

**Psycopg** 是一个 Python 库（也是数据库驱动程序），它在使用 PostgreSQL 数据库时提供了额外的功能和特性。请按照以下步骤操作：

1.  开始时导入必要的库。你将从 `database.cfg` 文件中导入所需的连接参数，正如*技术要求*部分中所突出显示的那样。你将创建一个 Python 字典来存储所有连接到数据库所需的参数值，比如 `host`、`database` 名称、`user` 名称和 `password`：

```py
import psycopg
import pandas as pd
import configparser
config = configparser.ConfigParser()
config.read('database.cfg')
params = dict(config['POSTGRESQL'])
```

1.  你可以通过将参数传递给 `connect()` 方法来建立连接。一旦连接成功，你可以创建一个**游标**对象，用来执行 SQL 查询：

```py
conn = psycopg.connect(**params)
cursor = conn.cursor()
```

1.  游标对象提供了多个属性和方法，包括 `execute`、`executemany`、`fetchall`、`fetchmany` 和 `fetchone`。以下代码使用游标对象传递 SQL 查询，然后使用 `rowcount` 属性检查该查询所产生的记录数：

```py
cursor.execute("""
SELECT date, close, volume
FROM msft
ORDER BY date;
""")
cursor.rowcount
>> 1259
```

1.  执行查询后返回的结果集将不包括标题（没有列名）。或者，你可以通过使用 `description` 属性从游标对象中获取列名，代码如下所示：

```py
cursor.description
>>
[<Column 'date', type: varchar(50) (oid: 1043)>,
 <Column 'close', type: float4 (oid: 700)>,
 <Column 'volume', type: int4 (oid: 23)>]
```

1.  你可以使用列表推导式从 `cursor.description` 中提取列名，并在创建 DataFrame 时将其作为列标题传入：

```py
columns = [col[0] for col in cursor.description]
columns
>>
['date', 'close', 'volume']
```

1.  要获取执行查询所产生的结果，你将使用 `fetchall` 方法。你将根据获取的结果集创建一个 DataFrame。确保传入你刚才捕获的列名：

```py
data = cursor.fetchall()
df = pd.DataFrame(data, columns=columns)
df.info()
>>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1259 entries, 0 to 1258
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   date    1259 non-null   object
 1   close   1259 non-null   float64
 2   volume  1259 non-null   int64 
dtypes: float64(1), int64(1), object(1)
memory usage: 29.6+ KB
```

注意，`date` 列作为 `object` 类型返回，而不是 `datetime` 类型。

1.  使用 `pd.to_datetime()` 解析 `date` 列，并将其设置为 DataFrame 的索引：

```py
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
print(df.tail(3))
>>
             close    volume
date                       
2024-08-30  417.14  24308300
2024-09-03  409.44  20285900
2024-09-04  408.84   9167942
```

在前面的代码中，游标返回了一个没有标题的元组列表**（没有列名）**。你可以通过运行以下代码来确认这一点：

```py
data = cursor.fetchall()
data[0:5]
>>
[('2019-09-04', 131.45726, 17995900),
 ('2019-09-05', 133.7687, 26101800),
 ('2019-09-06', 132.86136, 20824500),
 ('2019-09-09', 131.35222, 25773900),
 ('2019-09-10', 129.97684, 28903400)]
```

你可以指示游标返回一个 `dict_row` 类型，它将包括列名信息（即标题）。这在转换为 DataFrame 时更为方便。可以通过将 `dict_row` 类传递给 `row_factory` 参数来实现：

```py
from psycopg.rows import dict_row
conn = psycopg.connect(**params, row_factory=dict_row)
cursor = conn.cursor()
cursor.execute("SELECT * FROM msft;")
data = cursor.fetchall()
data[0:2]
>>
[{'date': '2019-09-04',
  'open': 131.14206,
  'high': 131.51457,
  'low': 130.35883,
  'close': 131.45726,
  'volume': 17995900},
 {'date': '2019-09-05',
  'open': 132.87086,
  'high': 134.08391,
  'low': 132.53656,
  'close': 133.7687,
  'volume': 26101800}]
```

注意列名已经可用。你现在可以像以下代码所示那样创建一个 DataFrame：

```py
df = pd.DataFrame(data)
print(df.head())
>>
        date       open       high        low      close    volume
0  2019-09-04  131.14206  131.51457  130.35883  131.45726  17995900
1  2019-09-05  132.87086  134.08391  132.53656  133.76870  26101800
2  2019-09-06  133.74963  133.89291  132.00171  132.86136  20824500
3  2019-09-09  133.32938  133.48220  130.33977  131.35222  25773900
4  2019-09-10  130.66455  130.75050  128.47725  129.97684  28903400
```

1.  关闭游标和数据库连接：

```py
cursor.close()
conn.close()
```

注意，`psycopg` 连接和游标可以在 Python 的 `with` 语句中用于处理异常，以便在提交事务时进行异常处理。游标对象提供了三种不同的获取函数；即 `fetchall`、`fetchmany` 和 `fetchone`。`fetchone` 方法返回一个单独的元组。以下示例展示了这一概念：

```py
with psycopg.connect(**params) as conn:
     with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM msft')
            data = cursor.fetchone()
print(data)
>>
('2019-09-04', 131.14206, 131.51457, 130.35883, 131.45726, 17995900)
```

#### 使用 pandas 和 SQLAlchemy

SQLAlchemy 是一个非常流行的开源库，用于在 Python 中操作关系型数据库。SQLAlchemy 可以被称为**对象关系映射器（ORM）**，它提供了一个抽象层（可以把它当作一个接口），让你可以使用面向对象编程与关系型数据库进行交互。

你将使用 SQLAlchemy，因为它与 pandas 的集成非常好，多个 pandas SQL 读写函数依赖于 SQLAlchemy 作为抽象层。SQLAlchemy 会为任何 pandas 的 SQL 读写请求做幕后翻译。这种翻译确保了 pandas 中的 SQL 语句会以适用于底层数据库类型（如 MySQL、Oracle、SQL Server 或 PostgreSQL 等）的正确语法/格式表示。

一些依赖于 SQLAlchemy 的 pandas SQL 读取函数包括 `pandas.read_sql`、`pandas.read_sql_query` 和 `pandas.read_sql_table`。让我们执行以下步骤：

1.  开始时导入必要的库。注意，在幕后，SQLAlchemy 将使用 **psycopg**（或任何其他已安装且 SQLAlchemy 支持的数据库驱动程序）：

```py
import pandas as pd
from sqlalchemy import create_engine
engine =\
    create_engine("postgresql+psycopg://postgres:password@localhost:5432/postgres")
query = "SELECT * FROM msft"
df = pd.read_sql(query,
                 engine,
                 index_col='date',
                 parse_dates={'date': '%Y-%m-%d'})
print(df.tail(3))
>>
              open    high     low   close    volume
date                                               
2024-08-30  415.60  417.49  412.13  417.14  24308300
2024-09-03  417.91  419.88  407.03  409.44  20285900
2024-09-04  405.63  411.24  404.37  408.84   9167942
```

在前面的示例中，对于 `parse_dates`，你传递了一个字典格式的参数 `{key: value}`，其中 `key` 是列名，`value` 是日期格式的字符串表示。与之前的 **psycopg** 方法不同，`pandas.read_sql` 更好地处理了数据类型的正确性。注意，我们的索引是 `DatetimeIndex` 类型：

```py
df.info()
>>
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1259 entries, 2019-09-04 to 2024-09-04
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   open    1259 non-null   float64
 1   high    1259 non-null   float64
 2   low     1259 non-null   float64
 3   close   1259 non-null   float64
 4   volume  1259 non-null   int64 
dtypes: float64(4), int64(1)
memory usage: 59.0 KB
```

1.  你也可以使用 `pandas.read_sql_query` 来完成相同的操作：

```py
df = pd.read_sql_query(query,
                       engine,
                       index_col='date',
                       parse_dates={'date':'%Y-%m-%d'})
```

1.  `pandas.read_sql_table` 是 pandas 提供的另一个 SQL 读取函数，它接受表名而不是 SQL 查询。可以把它看作是一个 `SELECT * FROM tablename` 查询：

```py
df = pd.read_sql_table('msft',
                        engine,
                        index_col='date',
                        parse_dates={'date':'%Y-%m-%d'})
```

> `read_sql` 读取函数更为通用，因为它是 `read_sql_query` 和 `read_sql_table` 的封装器。`read_sql` 函数既可以接受 SQL 查询，也可以接受表名。

### 它是如何工作的……

在本教程中，你探索了两种连接 PostgreSQL 数据库的方法：直接使用 psycopg 驱动程序或利用 pandas 和 SQLAlchemy。

使用**psycopg**连接 PostgreSQL 时，首先需要创建一个连接对象，然后创建一个游标对象。连接对象和游标的概念在 Python 中的不同数据库驱动程序之间是一致的。创建游标对象后，可以访问多个方法，包括：

+   `Execute()` – 执行 SQL 查询（CRUD）或命令到数据库

+   `Executemany()` – 使用一系列输入数据执行相同的数据库操作，例如，这对于批量插入的 `INSERT INTO` 操作非常有用。

+   `Fetchall()` – 返回当前查询结果集中的所有剩余记录

+   `Fetchone()` - 从当前查询结果集中返回下一条记录（单条记录）

+   `fetchmany(n)` – 从当前查询结果集中返回 `n` 条记录

+   `close()` - 关闭当前游标并释放关联的资源

另一方面，创建 **engine** 对象是使用 SQLAlchemy 时的第一步，因为它提供了关于正在使用的数据库的说明，这被称为 **dialect**（方言）。

当你使用 `create_engine` 创建引擎时，你传递了一个作为连接字符串的 URL。让我们检查一下 SQLAlchemy 的引擎连接字符串：

```py
create_engine("dialect+driver://username:password@host:port/database")
```

+   `dialect` – SQLAlchemy 方言（数据库类型）的名称，例如 postgresql、mysql、sqlite、oracle 或 mssql。

+   `driver` – 连接指定方言的已安装驱动程序（DBAPI）的名称，例如，PostgreSQL 数据库的 `psycopg` 或 `pg8000`。

+   `username` - 数据库身份验证的登录用户名

+   `password` - 为指定的用户名设置的密码

+   `host` - 数据库所在的服务器

+   `port` - 数据库的特定端口

+   `database` - 你要连接的具体数据库的名称

之前，你使用 `psycopg` 作为 PostgreSQL 的数据库驱动程序。`psycopg` 驱动程序被称为 **数据库 API (DBAPI)**，SQLAlchemy 支持多种基于 Python DBAPI 规范的 DBAPI 包装器，用于连接和与各种类型的关系型数据库进行交互。SQLAlchemy 已经内置了多种方言，以便与不同类型的关系数据库管理系统（RDBMS）协作，具体包括：

+   SQL Server

+   SQLite

+   PostgreSQL

+   MySQL

+   Oracle

+   Snowflake

当使用 SQLAlchemy 连接到数据库时，你需要指定要使用的 **方言** 和 **驱动程序**（DBAPI）。这是 PostgreSQL 的 URL 字符串格式：

```py
create_engine("postgresql+psycopg2://username:password@localhost:5432/dbname")
```

如果你使用 **MySQL** 数据库并配合 `PyMySQL` 驱动，连接 URL 将如下所示：

```py
create_engine("mysql+pymysql://username:password@localhost:3306/dbname")
```

在前面 *如何做……* 部分的代码示例中，你无需指定 `psycopg` 驱动程序，因为它是 SQLAlchemy 默认使用的 DBAPI。假设已经安装了 `psycopg`，此示例将可以正常工作：

```py
create_engine("postgresql://username:password@localhost:5432/dbname")
```

SQLAlchemy 支持其他 PostgreSQL 驱动程序（DBAPI），包括以下内容：

+   `psycopg`

+   `pg8000`

+   `asyncpg`

如需查看更全面的支持的方言和驱动程序列表，你可以访问官方文档页面 [`docs.sqlalchemy.org/en/20/dialects/index.html`](https://docs.sqlalchemy.org/en/20/dialects/index.html)。

使用 SQLAlchemy 的优势在于，它与 pandas 集成良好。如果你查阅官方 pandas 文档中的 `read_sql`、`read_sql_query`、`read_sql_table` 和 `to_sql`，你会注意到 `con` 参数期望一个 SQLAlchemy 连接对象（引擎）。

另一个优势是，你可以轻松更换后端引擎（数据库），例如从 PostgreSQL 切换到 MySQL，而无需修改其余代码。

### 还有更多……

在本节中，我们将探讨一些附加概念，帮助你更好地理解 SQLAlchemy 的多功能性，并将第二章 *处理大型数据文件* 中介绍的一些概念与本教程结合起来，这些概念讨论了如何从文件中读取时间序列数据。

具体来说，我们将讨论以下内容：

+   在 SQLAlchemy 中生成连接 URL

+   扩展到 Amazon Redshift 数据库

+   pandas 中的分块处理

#### 在 SQLAlchemy 中生成连接 URL

在这个教程中，你已经学习了 SQLAlchemy 库中的 `create_engine` 函数，它通过 URL 字符串来建立数据库连接。到目前为止，你一直是手动创建 URL 字符串，但有一种更方便的方式可以为你生成 URL。这可以通过 SQLAlchemy 中 `URL` 类的 `create` 方法来实现。

以下代码演示了这一点：

```py
from sqlalchemy import URL, create_engine
url = URL.create(
    drivername='postgresql+psycopg',
    host= '127.0.0.1',
    username='postgres',
    password='password',
    database='postgres',
    port= 5432
)
>>
postgresql+psycopg://postgres:***@127.0.0.1:5432/postgres
```

请注意，`drivername` 包含 *方言* 和 *驱动程序*，格式为 `dialct+driver`。

现在，你可以像以前一样将 `url` 传递给 `create_engine`。

```py
engine = create_engine(url)
df = pd.read_sql('select * from msft;', engine)
```

#### 扩展到 Amazon Redshift 数据库

我们讨论了 SQLAlchemy 的多功能性，它允许你更改数据库引擎（后端数据库），而保持其余代码不变。例如，使用 PostgreSQL 或 MySQL，或者 SQLAlchemy 支持的任何其他方言。我们还将探讨如何连接到像 Amazon Redshift 这样的云数据仓库。

值得一提的是，**Amazon Redshift**是一个基于 PostgreSQL 的云数据仓库。你将安装适用于 SQLAlchemy 的 Amazon Redshift 驱动程序（它使用 psycopg2 DBAPI）。

你也可以使用**conda**进行安装：

```py
conda install conda-forge::sqlalchemy-redshift
```

你也可以使用**pip**进行安装：

```py
pip install sqlalchemy-redshift
```

因为我们不希望在代码中暴露你的 AWS 凭证，你将更新在*技术要求*部分讨论的`database.cfg`文件，以包含你的 AWS Redshift 信息：

```py
[AWS]
host=<your_end_point.your_region.redshift.amazonaws.com>
port=5439
database=dev
username=<your_username>
password=<your_password>
```

你将使用`configparser`加载你的值：

```py
from configparser import ConfigParser
config = ConfigParser()
config.read('database.cfg')
config.sections()
params = dict(config['AWS'])
```

你将使用 URL.create 方法来生成你的 URL：

```py
url = URL.create('redshift+psycopg2', **params)
aws_engine = create_engine(url)
```

现在，你可以将之前代码中的引擎切换，从指向本地 PostgreSQL 实例的引擎，改为在 Amazon Redshift 上运行相同的查询。这假设你在 Amazon Redshift 中有一个`msft`表。

```py
df = pd.read_sql(query,
                 aws_engine,
                 index_col='date',
                 parse_dates=True)
```

要了解更多关于`sqlalchemy-redshift`的信息，可以访问该项目的仓库：[`github.com/sqlalchemy-redshift/sqlalchemy-redshift`](https://github.com/sqlalchemy-redshift/sqlalchemy-redshift)。

> Amazon Redshift 的例子可以扩展到其他数据库，例如 Google BigQuery、Teradata 或 Microsoft SQL Server，只要这些数据库有支持的 SQLAlchemy 方言即可。要查看完整的列表，请访问官方页面：[`docs.sqlalchemy.org/en/20/dialects/index.html`](https://docs.sqlalchemy.org/en/20/dialects/index.html)

#### 使用 pandas 进行 chunking

当你执行针对`msft`表的查询时，它返回了 1259 条记录。试想一下，如果处理一个更大的数据库，可能会返回数百万条记录，甚至更多。这就是`chunking`参数派上用场的地方。

`chunksize`参数允许你将一个大的数据集拆分成较小的、更易于管理的数据块，这些数据块能够适应本地内存。当执行`read_sql`函数时，只需将要检索的行数（每个数据块）传递给`chunksize`参数，之后它会返回一个`generator`对象。你可以循环遍历这个生成器对象，或者使用`next()`一次获取一个数据块，并进行所需的计算或处理。让我们看一个如何实现 chunking 的例子。你将每次请求`500`条记录（行）：

```py
df_gen = pd.read_sql(query,
                 engine,
                 index_col='date',
                 parse_dates=True,
                 chunksize=500)
```

上面的代码将生成三个（3）数据块。你可以像下面这样遍历`df_gen`生成器对象：

```py
for idx, data in enumerate(df_gen):
     print(idx, data.shape)
>>
0 (500, 5)
1 (500, 5)
2 (259, 5)
```

上面的代码展示了 chunking 如何工作。使用`chunksize`参数应该减少内存使用，因为每次加载的行数较少。

### 另见：

若要获取关于这些主题的更多信息，请查看以下链接：

+   对于**SQLAlchemy**，你可以访问[`www.sqlalchemy.org/`](https://www.sqlalchemy.org/)

+   关于`pandas.read_sql`函数，请访问[`pandas.pydata.org/docs/reference/api/pandas.read_sql_table.html`](https://pandas.pydata.org/docs/reference/api/pandas.read_sql_table.html)

+   关于`pandas.read_sql_query`函数，请访问[`pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html`](https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html)

+   关于`pandas.read_sql_table`函数，请访问[`pandas.pydata.org/docs/reference/api/pandas.read_sql_table.html`](https://pandas.pydata.org/docs/reference/api/pandas.read_sql_table.html)

## 从 Snowflake 读取数据

一个非常常见的数据分析提取来源通常是公司的*数据仓库*。数据仓库托管了大量的数据，这些数据大多是集成的，用来支持各种报告和分析需求，此外还包含来自不同源系统的历史数据。

云计算的发展为我们带来了云数据仓库，如**Amazon Redshift**、**Google BigQuery**、**Azure SQL Data Warehouse**和**Snowflake**。

在这个教程中，你将使用*Snowflake*，一个强大的**软件即服务**（**SaaS**）基于云的数据仓库平台，可以托管在不同的云平台上，例如**Amazon Web Services**（**AWS**）、**Google Cloud Platform**（**GCP**）和**Microsoft Azure**。你将学习如何使用 Python 连接到 Snowflake，提取时间序列数据，并将其加载到 pandas DataFrame 中。

### 准备工作

这个教程假设你有访问 Snowflake 的权限。你将探索三种（3）不同的方法来连接 Snowflake，因此你需要安装三种（3）不同的库。

推荐的`雪花连接器-python`库安装方法是使用**pip**，这样可以让你安装像`pandas`这样的*附加组件*，如下所示：

```py
pip install snowflake-sqlalchemy snowflake-snowpark-python
pip install "snowflake-connector-python[pandas]"
```

你也可以使用**conda**进行安装，但如果你想要将`snowflake-connector-python`与 pandas 一起使用，你需要使用 pip 安装。

```py
conda install -c conda-forge snowflake-sqlalchemy snowflake-snowpark-python
conda install -c conda-froge snowflake-connector-python
```

确保你在*技术要求*部分创建的配置文件`database.cfg`包含了你的**Snowflake**连接信息：

```py
[SNOWFLAKE]
user=username
password=password
account=snowflakeaccount
warehouse=COMPUTE_WH
database=SNOWFLAKE_SAMPLE_DATA
schema=TPCH_SF1
role=somerole
```

在这个教程中，你将使用`SNOWFLAKE_SAMPLE_DATA`数据库和 Snowflake 提供的`TPCH_SF1`模式。

> 获取正确的`account`值可能会让许多人感到困惑。为了确保你获得正确的格式，请使用 Snowflake 中的*复制帐户 URL*选项，它可能像这样`https://abc1234.us-east-1.snowflakecomputing.com`，其中`abc1234.us-east-1`部分是你将用作`account`值的部分。

### 如何操作...

我们将探索三种（3）方法和库来连接到 Snowflake 数据库。在第一种方法中，你将使用 Snowflake Python 连接器建立连接，并创建一个游标来查询和提取数据。在第二种方法中，你将使用 Snowflake **SQLAlchemy**。在第三种方法中，你将探索**Snowpark** Python API。让我们开始吧：

#### 使用 snowflake-connector-python

1.  我们将从导入必要的库开始：

```py
import pandas as pd
from snowflake import connector
from configparser import ConfigParser
```

1.  使用 `ConfigParser`，你将提取 `[SNOWFLAKE]` 部分下的内容，以避免暴露或硬编码你的凭据。你可以读取 `[SNOWFLAKE]` 部分的所有内容并将其转换为字典对象，如下所示：

```py
config = ConfigParser()
config.read(database.cfg')
params = dict(config['SNOWFLAKE'])
```

1.  你需要将参数传递给 `connector.connect()` 来与 Snowflake 建立连接。我们可以轻松地 *解包* 字典内容，因为字典的键与参数名匹配。一旦连接建立，我们可以创建我们的 *游标*：

```py
con = connector.connect(**params)
cursor = con.cursor()
```

1.  游标对象有许多方法，如 `execute`、`fetchall`、`fetchmany`、`fetchone`、`fetch_pandas_all` 和 `fetch_pandas_batches`。

你将从 `execute` 方法开始，向数据库传递 SQL 查询，然后使用任何可用的获取方法来检索数据。在以下示例中，你将查询 `ORDERS` 表，然后利用 `fetch_pandas_all` 方法将整个结果集作为 pandas DataFrame 检索：

```py
query = "SELECT * FROM ORDERS;"
cursor.execute(query)
df = cursor.fetch_pandas_all()
```

之前的代码可以写成如下：

```py
df = cursor.execute(query).fetch_pandas_all()
```

1.  使用 `df.info()` 检查 DataFrame：

```py
df.info()
>>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1500000 entries, 0 to 1499999
Data columns (total 9 columns):
 #   Column           Non-Null Count    Dtype 
---  ------           --------------    ----- 
 0   O_ORDERKEY       1500000 non-null  int32 
 1   O_CUSTKEY        1500000 non-null  int32 
 2   O_ORDERSTATUS    1500000 non-null  object
 3   O_TOTALPRICE     1500000 non-null  float64
 4   O_ORDERDATE      1500000 non-null  object
 5   O_ORDERPRIORITY  1500000 non-null  object
 6   O_CLERK          1500000 non-null  object
 7   O_SHIPPRIORITY   1500000 non-null  int8  
 8   O_COMMENT        1500000 non-null  object
dtypes: float64(1), int32(2), int8(1), object(5)
memory usage: 81.5+ MB
```

1.  从前面的输出中可以看到，DataFrame 的索引仅是一个数字序列，且 `O_ORDERDATE` 列不是一个 `Date` 类型的字段。你可以使用 `pandas.to_datetime()` 函数将 `O_ORDERDATE` 列解析为日期时间类型，然后使用 `DataFrame.set_index()` 方法将该列设置为 DataFrame 的索引：

```py
df_ts = (
    df.set_index(
                pd.to_datetime(df['O_ORDERDATE'])
                )
                .drop(columns='O_ORDERDATE'))
```

让我们显示 `df_ts` DataFrame 的前四（4）列和前五（5）行：

```py
print(df_ts.iloc[0:3, 1:5])
>>
             O_CUSTKEY O_ORDERSTATUS  O_TOTALPRICE  O_ORDERPRIORITY
O_ORDERDATE                                                       
1994-02-21       13726             F      99406.41         3-MEDIUM
1997-04-14      129376             O     256838.41  4-NOT SPECIFIED
1997-11-24      141613             O     150849.49  4-NOT SPECIFIED
```

1.  检查 DataFrame 的索引。打印前两个索引：

```py
df_ts.index[0:2]
>>
DatetimeIndex(['1994-02-21', '1997-04-14'], dtype='datetime64[ns]', name='O_ORDERDATE', freq=None)
```

1.  最后，你可以关闭游标和当前连接。

```py
Cursor.close()
con.close()
```

现在你拥有一个具有`DatetimeIndex`的时间序列 DataFrame。

#### 使用 SQLAlchemy

在之前的例子中，`从关系数据库读取数据`，你探索了 pandas 的 `read_sql`、`read_sql_query` 和 `read_sql_table` 函数。这是通过使用 SQLAlchemy 和安装一个支持的方言来完成的。在这里，我们将在安装 `snowflake-sqlalchemy` 驱动程序后使用 Snowflake 方言。

SQLAlchemy 与 pandas 更好地集成，正如你将在本节中体验的那样。

1.  开始时，导入必要的库，并从 `database.cfg` 文件中的 `[SNOWFLAKE]` 部分读取 Snowflake 连接参数。

```py
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import configparser
config = ConfigParser()
config.read('database.cfg')
params = dict(config['SNOWFLAKE'])
```

1.  你将使用 URL 类来生成 URL 连接字符串。我们将创建我们的引擎对象，然后使用 `engine.connect()` 打开连接：

```py
url = URL(**params)
engine = create_engine(url)
connection = engine.connect()
```

1.  现在，你可以使用 `read_sql` 或 `read_sql_query` 来对 SNOWFLAKE_SAMPLE_DATA 数据库中的 `ORDERS` 表执行查询：

```py
query = "SELECT * FROM ORDERS;"
df = pd.read_sql(query,
                 connection,
                 index_col='o_orderdate',
                 parse_dates='o_orderdate')
df.info()
>>
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1500000 entries, 1992-04-22 to 1994-03-19
Data columns (total 8 columns):
 #   Column           Non-Null Count    Dtype 
---  ------           --------------    ----- 
 0   o_orderkey       1500000 non-null  int64 
 1   o_custkey        1500000 non-null  int64 
 2   o_orderstatus    1500000 non-null  object
 3   o_totalprice     1500000 non-null  float64
 4   o_orderpriority  1500000 non-null  object
 5   o_clerk          1500000 non-null  object
 6   o_shippriority   1500000 non-null  int64 
 7   o_comment        1500000 non-null  object
dtypes: float64(1), int64(3), object(4)
memory usage: 103.0+ MB
```

注意与之前使用 Snowflake Python 连接器的方法相比，我们是如何在一个步骤中解析 `o_orderdate` 列并将其设置为索引的。

1.  最后，关闭数据库连接。

```py
connection.close()
engine.dispose()
```

通过使用 **上下文管理器**，可以进一步简化代码，以自动分配和释放资源。以下示例使用了 `with engine.connect()`：

```py
query = "SELECT * FROM ORDERS;"
url = URL(**params)
engine = create_engine(url)
with engine.connect() as connection:
    df = pd.read_sql(query,
                      connection,
                      index_col='o_orderdate',
                      parse_dates=['o_orderdate'])
df.info()
```

这样应该能够得到相同的结果，而无需关闭连接或处理引擎。

#### 使用 snowflake-snowpark-python

Snowpark API 支持 Java、Python 和 Scala。你已经按照本配方中 *准备工作* 部分的描述安装了 `snowflake-snowpark-python`。

1.  从导入必要的库并从 `database.cfg` 文件的 `[SNOWFLAKE]` 部分读取 Snowflake 连接参数开始

```py
from snowflake.snowpark import Session
from configparser import ConfigParser
config = ConfigParser()
config.read('database.cfg')
params = dict(config['SNOWFLAKE'])
```

1.  通过与 Snowflake 数据库建立连接来创建会话

```py
session = Session.builder.configs(params).create()
```

1.  会话有多个 `DataFrameReader` 方法，如 `read`、`table` 和 `sql`。这些方法中的任何一个都会返回一个 Snowpark DataFrame 对象。返回的 Snowpark DataFrame 对象可以使用 `to_pandas` 方法转换为更常见的 pandas DataFrame。你将探索 `read`、`table` 和 `sql` 方法，以返回相同的结果集。

从 `read` 方法开始。更具体地说，你将使用 `read.table` 并传入一个表名。这将返回该表的内容并通过 `to_pandas` 方法转换为 pandas DataFrame。可以将其视为等同于 `SELECT * FROM TABLE`。

```py
orders = session.read.table("ORDERS").to_pandas()
```

类似地，`table` 方法接受一个表名，返回的对象（Snowpark DataFrame）可以使用 `to_pandas` 方法：

```py
orders = session.table("ORDERS").to_pandas()
```

最后，`sql` 方法接受一个 SQL 查询：

```py
query = 'SELECT * FROM ORDERS'
orders = session.sql(query).to_pandas()
orders.info()
>>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1500000 entries, 0 to 1499999
Data columns (total 9 columns):
 #   Column           Non-Null Count    Dtype 
---  ------           --------------    ----- 
 0   O_ORDERKEY       1500000 non-null  int32 
 1   O_CUSTKEY        1500000 non-null  int32 
 2   O_ORDERSTATUS    1500000 non-null  object
 3   O_TOTALPRICE     1500000 non-null  float64
 4   O_ORDERDATE      1500000 non-null  object
 5   O_ORDERPRIORITY  1500000 non-null  object
 6   O_CLERK          1500000 non-null  object
 7   O_SHIPPRIORITY   1500000 non-null  int8  
 8   O_COMMENT        1500000 non-null  object
dtypes: float64(1), int32(2), int8(1), object(5)
memory usage: 81.5+ MB
```

这三种方法应该产生相同的 pandas DataFrame。

### 它是如何工作的...

Snowflake Python 连接器、Snowflake SQLAlchemy 驱动程序和 Snowpark Python 需要相同的输入变量来建立与 Snowflake 数据库的连接。这些包括以下内容：

![表 3.1 – Snowflake Python 连接器的输入变量](img/file25.jpg)

表 3.1 – Snowflake Python 连接器的输入变量

回想一下，在前一个活动中，你在 `database.cfg` 文件的 `[SNOWFLAKE]` 部分为所有三种方法使用了相同的配置。

#### Snowflake Python 连接器

当使用 Python 连接器时，你首先通过 `con = connector.connect(**params)` 来建立与数据库的连接。一旦连接被接受，你使用 `cursor = con.cursor()` 创建一个**游标**对象。

游标提供了执行和获取操作的方法，如 `describe()`、`execute()`、`execute_async()`、`executemany()`、`fetchone()`、`fetchall()`、`fetchmany()`、`fetch_pandas_all()` 和 `fetch_pandas_batches()`，每个游标还具有若干属性，包括 `description`、`rowcount`、`rownumber` 等。注意，当使用 Python 连接器 **psycopg** 时，之前的配方 *从关系数据库读取数据* 中讨论了熟悉的方法和属性。

+   `Execute()` – 执行 SQL 查询（CRUD）或命令到数据库

+   `executemany()` – 使用一系列输入数据执行相同的数据库操作，例如，这在使用 INSERT INTO 进行批量插入时非常有用。

+   `Fetchall()` – 返回当前查询结果集中的所有剩余记录

+   `fetchone()` - 从当前查询结果集中返回下一条记录（一条记录）

+   `fetchmany(n)` – 从当前查询结果集中返回`n`条记录

+   `fetch_pandas_all()` - 返回当前查询结果集中的所有剩余记录，并将它们加载到 pandas DataFrame 中

+   `fetch_pandas_batches()` - 返回当前查询结果集中的一部分剩余记录，并将它们加载到 pandas DataFrame 中

+   `close()` - 关闭当前游标并释放相关资源

+   `describe()` – 返回结果集的元数据，但不会执行查询。或者，你可以使用`execute()`，然后使用`description`属性来获取相同的元数据信息。

要查看完整的属性和方法列表，请参考官方文档：[`docs.snowflake.com/en/user-guide/python-connector-api.html#object-cursor`](https://docs.snowflake.com/en/user-guide/python-connector-api.html#object-cursor)。

#### SQLAlchemy API

当使用 SQLAlchemy 时，你可以利用`pandas.read_sql`、`pandas.read_sql_query`和`pandas.read_sql_query`读取函数，并利用许多可用参数在读取时转换和处理数据，如`index_col`和`parse_dates`。另一方面，当使用 Snowflake Python 连接器时，`fetch_pandas_all()`函数不接受任何参数，你需要在之后解析和调整 DataFrame。

Snowflake SQLAlchemy 库提供了一个方便的方法`URL`，帮助构建连接字符串以连接到 Snowflake 数据库。通常，SQLAlchemy 期望提供以下格式的 URL：

```py
'snowflake://<user>:<password>@<account>/<database>/<schema>
?warehouse=<warehouse>&role=<role>'
```

使用`URL`方法，我们传递了参数，方法会自动构造所需的连接字符串：

```py
engine = create_engine(URL(
    account = '<your_account>',
    user = '<your_username>',
    password = '<your_password>',
    database = '<your_database>',
    schema = '<your_schema>',
    warehouse = '<your_warehouse>',
    role='<your_role>',
))
```

或者，我们将 Snowflake 参数存储在配置文件 database.cfg 中，并以 Python 字典的形式存储。这样，你就不会在代码中暴露你的凭证。

```py
params = dict(config['SNOWFLAKE'])
url = create_engine(URL(**params))
```

> 如果你比较本食谱中的过程，使用**SQLAlchemy**连接 Snowflake，与之前食谱中的*从关系数据库读取数据*过程，你会发现两者在过程和代码上有许多相似之处。这就是使用 SQLAlchemy 的优势之一，它为各种数据库创建了一个标准流程，只要 SQLAlchemy 支持这些数据库。*SQLAlchemy 与 pandas 集成得很好，可以轻松地切换方言（后端数据库），而无需对代码做太多修改*。

#### Snowpark API

在之前的方法中，你只是使用了`snowflake-connector-python`和`snowflake-connector-python`库作为连接器连接到你的 Snowflake 数据库，然后提取数据以在本地处理。

Snowpark 不仅仅提供了一种连接数据库的机制。它允许您直接在 Snowflake 云环境中处理数据，而无需将数据移出或在本地处理。此外，Snowpark 还非常适合执行更复杂的任务，如构建复杂的数据管道或使用 Snowpark ML 处理机器学习模型，所有这些都在 Snowflake 云中完成。

在我们的方案中，类似于其他方法，我们需要与 Snowflake 建立连接。这是通过使用`Session`类来实现的。

```py
params = dict(config['SNOWFLAKE'])
session = Session.builder.configs(params).create()
```

Snowpark 和 PySpark（Spark）在 API 和概念上有很多相似之处。更具体地说，Snowpark DataFrame 被视为*延迟求值的关系数据集*。`to_pandas`方法执行了两件事：它执行查询并将结果加载到 pandas DataFrame 中（数据会被从 Snowflake 外部提取）。要将 pandas DataFrame 转换回 Snowpark DataFrame（在 Snowflake 内部），您可以使用如下的`create_dataframe`方法：

```py
df = session.create_dataframe(orders)
```

为了成功执行前面的代码，您需要具有*写权限*，因为 Snowflake 会创建一个**临时**表来存储 pandas DataFrame（在 Snowflake 中），然后返回一个指向该临时表的 Snowpark DataFrame。或者，如果您希望将 pandas DataFrame 持久化到一个表中，您可以使用如下的`write_pandas`方法：

```py
df = session.write_pandas(orders, table_name='temp_table')
```

在前面的代码中，您传递了 pandas DataFrame 和表名。

### 还有更多...

您可能已经注意到，当使用 Snowflake Python 连接器和 Snowpark 时，返回的 DataFrame 中的列名全部以大写字母显示，而在使用 Snowflake SQLAlchemy 时，它们则是小写的。

之所以如此，是因为 Snowflake 默认在创建对象时将未加引号的对象名称存储为大写。例如，在之前的代码中，我们的`Order Date`列被返回为`O_ORDERDATE`。

为了明确指出名称是区分大小写的，您在创建 Snowflake 对象时需要使用引号（例如，`'o_orderdate'` 或 `'OrderDate'`）。相对而言，使用 Snowflake SQLAlchemy 时，默认会将名称转换为小写。

### 另见

+   有关**Snowflake Python 连接器**的更多信息，您可以访问官方文档：[`docs.snowflake.com/en/user-guide/python-connector.html`](https://docs.snowflake.com/en/user-guide/python-connector.html)

+   有关**Snowflake SQLAlchemy**的更多信息，您可以访问官方文档：[`docs.snowflake.com/en/user-guide/sqlalchemy.html`](https://docs.snowflake.com/en/user-guide/sqlalchemy.html)

+   有关**Snowpark API**的更多信息，您可以访问官方文档：[`docs.snowflake.com/developer-guide/snowpark/reference/python/latest/snowpark/index`](https://docs.snowflake.com/developer-guide/snowpark/reference/python/latest/snowpark/index)

## 从文档数据库读取数据

**MongoDB** 是一个**NoSQL** 数据库，使用文档存储数据，并使用 BSON（一种类似 JSON 的结构）来存储无模式的数据。与关系型数据库不同，关系型数据库中的数据存储在由行和列组成的表中，而面向文档的数据库则将数据存储在集合和文档中。

文档代表存储数据的最低粒度，就像关系型数据库中的行一样。集合像关系型数据库中的表一样存储文档。与关系型数据库不同，集合可以存储不同模式和结构的文档。

### 准备工作

在本教程中，假设你已经有一个正在运行的 MongoDB 实例。为准备此教程，你需要安装 `PyMongo` Python 库来连接 MongoDB。

要使用 `conda` 安装 MongoDB，请运行以下命令：

```py
conda install -c conda-forge pymongo -y
```

要使用 `pip` 安装 MongoDB，请运行以下命令：

```py
python -m pip install pymongo
```

如果你无法访问 PostgreSQL 数据库，最快的启动方式是通过 Docker ([`hub.docker.com/_/mongo`](https://hub.docker.com/_/mongo))。以下是一个示例命令：

```py
docker run -d \
    --name mongo-ch3 \
    -p 27017:27017 \
    --env MARIADB_ROOT_PASSWORD=password \
    mongo:8.0-rc
```

另外，你可以尝试免费使用**MongoDB Atlas**，访问 [`www.mongodb.com/products/platform/atlas-database`](https://www.mongodb.com/products/platform/atlas-database)。**MongoDB Atlas** 是一个完全托管的云数据库，可以部署在你喜欢的云提供商上，如 AWS、Azure 和 GCP。

> 关于使用 MongoDB Atlas 的注意事项
> 
> > 如果你连接的是 MongoDB Atlas（云）免费版或其 M2/M5 共享版集群，则需要使用 `mongodb+srv` 协议。在这种情况下，你可以在 pip 安装时指定 `python -m pip install "pymongo[srv]"`

可选地，如果你希望通过图形界面访问 MongoDB，可以从这里安装**MongoDB Compass** [`www.mongodb.com/products/tools/compass`](https://www.mongodb.com/products/tools/compass)

我正在使用 MongoDB Compass 来创建数据库、集合并加载数据。在第五章，*将时间序列数据持久化到数据库*，你将学习如何使用 Python 创建数据库、集合并加载数据。

使用**Compass**选择**创建数据库**选项。对于数据库名称，可以输入 `stock_data`，对于**集合**名称，可以输入 `microsoft`。勾选**时间序列**复选框，并将 `date` 设置为**时间字段**。

![图 3.4 – MongoDB Compass 创建数据库界面](img/file26.png)

图 3.4 – MongoDB Compass 创建数据库界面

一旦**数据库**和**集合**创建完成，点击**导入数据**并选择 `datasets/Ch3/MSFT.csv` 文件夹中的 MSFT 股票数据集。

![图 3.5 – MongoDB Compass 导入数据界面](img/file27.png)

图 3.5 – MongoDB Compass 导入数据界面

![图 3.6 – MongoDB Compass 在导入前审核数据类型界面](img/file28.png)

图 3.6 – MongoDB Compass 在导入前审核数据类型界面

最终页面确认数据类型。最后，点击**导入**。

### 如何操作……

在本教程中，你将连接到你已设置的 MongoDB 实例。如果你使用的是本地安装（本地安装或 Docker 容器），则连接字符串可能类似于`mongodb://<username>:<password>@<host>:<port>/<DatabaseName>`。如果你使用的是 Atlas，连接字符串可能更像是`mongodb+srv://<username>:<password>@<clusterName>.mongodb.net/<DatabaseName>?retryWrites=true&w=majority`。

执行以下步骤：

1.  首先，让我们导入必要的库：

```py
import pandas as pd
from pymongo import MongoClient, uri_parser
```

建立与 MongoDB 的连接。对于自托管实例，例如本地安装，连接字符串可能是这样的：

```py
# connecting to a self-hosted instance
url = "mongodb://127.0.0.1:27017"
client = MongoClient(url)
>>
MongoClient(host=['localhost:27017'], document_class=dict, tz_aware=False, connect=True)
```

这等同于以下内容：

```py
client = MongoClient(host=['127.0.0.1:27017'],
                     password=None,
                     username=None,
                     document_class=dict,
                     tz_aware=False,
                     connect=True)
```

如果你的自托管 MongoDB 实例具有用户名和密码，你必须提供这些信息。

`uri_parser`是一个有用的实用函数，允许你*验证*MongoDB 的 URL，如下所示：

```py
uri_parser.parse_uri("mongodb://127.0.0.1:27107")
>>
{'nodelist': [('127.0.0.1', 27107)],
 'username': None,
 'password': None,
 'database': None,
 'collection': None,
 'options': {},
 'fqdn': None}
```

如果你连接的是**MongoDB** **Atlas**，那么你的连接字符串看起来应该像这样：

```py
# connecting to Atlas cloud Cluster
cluster = 'cluster0'
username = 'user'
password = 'password'
database = 'stock_data'
url = \
f"mongodb+srv://{username}:{password}@{cluster}.3rncb.mongodb.net/{database}"
client =  MongoClient(url)
client
>>
MongoClient(host=['cluster0-shard-00-00.3rncb.mongodb.net:27017', 'cluster0-shard-00-01.3rncb.mongodb.net:27017', 'cluster0-shard-00-02.3rncb.mongodb.net:27017'], document_class=dict, tz_aware=False, connect=True, authsource='somesource', replicaset='Cluster0-shard-0', ssl=True)
```

在本章之前的教程中，我们使用了一个配置文件，例如`database.cfg`文件，用来存储我们的连接信息并隐藏凭证。你也应该遵循这个建议。

> 如果你的用户名或密码包含特殊字符，包括空格字符（`:/?#[]@!$&'()* ,;=%`），你需要对它们进行编码。你可以使用`urllib` Python 库中的`quote_plus()`函数进行百分号编码（百分号转义）。
> 
> > 这是一个示例：
> > 
> > `username = urllib.parse.quote_plus('user!*@')`
> > 
> > `password = urllib.parse.quote_plus('pass/w@rd')`
> > 
> > 更多信息，请阅读[这里的文档](https://www.mongodb.com/docs/atlas/troubleshoot-connection/#std-label-special-pass-characters)

1.  一旦连接成功，你可以列出所有可用的数据库。在此示例中，我将数据库命名为`stock_data`，集合命名为`microsoft`：

```py
client.list_database_names()
>>
['admin', 'config', 'local', 'stock_data']
```

1.  你可以使用`list_collection_names`列出`stock_data`数据库下可用的集合：

```py
db = client['stock_data']
db.list_collection_names()
>>
['microsoft', 'system.buckets.microsoft', 'system.views']
```

1.  现在，你可以指定查询哪个集合。在这个例子中，我们感兴趣的是名为`microsoft`的集合：

```py
collection = db['microsoft']
```

1.  现在，使用`find`方法将数据库查询到一个 pandas DataFrame 中：

```py
results = collection.find({})
msft_df = (pd.DataFrame(results)
             .set_index('date')
             .drop(columns='_id'))
msft_df.head()
>>
               close         low    volume        high        open
date                                                               
2019-09-04  131.457260  130.358829  17995900  131.514567  131.142059
2019-09-05  133.768707  132.536556  26101800  134.083908  132.870864
2019-09-06  132.861359  132.001715  20824500  133.892908  133.749641
2019-09-09  131.352219  130.339762  25773900  133.482199  133.329371
2019-09-10  129.976837  128.477244  28903400  130.750506  130.664546
```

### 它是如何工作的……

第一步是连接到数据库，我们通过使用`MongoClient`创建 MongoDB 实例的客户端对象来实现。这将使你能够访问一组方法，如`list_databases_names()`、`list_databases()`，以及其他属性，如`address`和`HOST`。

`MongoClient()`接受一个连接字符串，该字符串应遵循 MongoDB 的 URI 格式，如下所示：

```py
client = MongoClient("mongodb://localhost:27017")
```

另外，也可以通过显式提供*host*（字符串）和*port*（数字）位置参数来完成相同的操作，如下所示：

```py
client = MongoClient('localhost', 27017)
```

主机字符串可以是主机名或 IP 地址，如下所示：

```py
client = MongoClient('127.0.0.1', 27017)
```

请注意，要连接到使用默认端口（`27017`）的 **localhost**，你可以在不提供任何参数的情况下建立连接，如以下代码所示：

```py
# using default values for host and port
client = MongoClient()
```

此外，你可以显式地提供命名参数，如下所示：

```py
client = MongoClient(host='127.0.0.1',
                     port=27017,
                     password=password,
                     username=username,
                     document_class=dict,
                     tz_aware=False,
                     connect=True)
```

让我们来探讨这些参数：

+   `host` – 这可以是主机名、IP 地址或 MongoDB URI。它也可以是一个包含主机名的 Python 列表。

+   `password` – 你分配的密码。请参阅 *Getting Ready* 部分中关于特殊字符的说明。

+   `username` – 你分配的用户名。请参阅 *Getting Ready* 部分中关于特殊字符的说明。

+   `document_class` – 指定用于查询结果中文档的类。默认值是 `dict`。

+   `tz_aware` – 指定 datetime 实例是否是时区感知的。默认值为 `False`，意味着它们是“天真”的（不具备时区感知）。

+   `connect` – 是否立即连接到 MongoDB 实例。默认值是 `True`。

若需要更多参数，你可以参考官方文档页面 [`pymongo.readthedocs.io/en/stable/api/pymongo/mongo_client.html`](https://pymongo.readthedocs.io/en/stable/api/pymongo/mongo_client.html)。

一旦与 MongoDB 实例建立连接，你可以指定使用的数据库，列出其集合，并查询任何可用的集合。在可以查询和检索文档之前的整体流程是：指定 **数据库**，选择你感兴趣的 **集合**，然后提交 **查询**。

在前面的示例中，我们的数据库名为 `stock_data`，其中包含一个名为 `microsoft` 的集合。一个数据库可以包含多个集合，而一个集合可以包含多个文档。如果从关系型数据库的角度考虑，集合就像是表格，而文档代表表格中的行。

在 PyMongo 中，你可以使用不同的语法来指定数据库，如以下代码所示。请记住，所有这些语句都会生成一个 `pymongo.database.Database` 对象：

```py
# Specifying the database
db = client['stock_data']
db = client.stock_data
db = client.get_database('stock_data')
```

在前面的代码中，`get_database()` 可以接受额外的参数，如 `codec_options`、`read_preference`、`write_concern` 和 `read_concern`，其中后两个参数更关注节点间的操作以及如何确定操作是否成功。

同样地，一旦你拥有了 `PyMongo` 数据库对象，你可以使用不同的语法来指定集合，如以下示例所示：

```py
# Specifying the collection
collection = db.microsoft
collection = db['microsoft']
collection = db.get_collection('microsoft')
```

`get_collection()` 方法提供了额外的参数，类似于 `get_database()`。

前面的三个语法变体返回一个 `pymongo.database.Collection` 对象，它带有额外的内建方法和属性，如 `find`、`find_one`、`find_one_and_delete`、`find_one_and_replace`、`find_one_and_update`、`update`、`update_one`、`update_many`、`delete_one` 和 `delete_many` 等。

让我们探索不同的*检索*集合方法：

+   `find()` – 基于提交的查询从集合中检索多个文档。

+   `find_one()` – 基于提交的查询，从集合中检索单个文档。如果多个文档匹配，则返回第一个匹配的文档。

+   `find_one_and_delete()` – 查找单个文档，类似于`find_one`，但它会从集合中删除该文档，并返回删除的文档。

+   `find_one_and_replace()` - 查找单个文档并用新文档替换它，返回原始文档或替换后的文档。

+   `find_one_and_update()` - 查找单个文档并更新它，返回原始文档或更新后的文档。与`find_one_and_replace`不同，它是更新现有文档，而不是替换整个文档。

一旦你到达集合级别，你可以开始查询数据。在本示例中，你使用了`find()`，它类似于 SQL 中的`SELECT`语句。

在`如何实现…`部分的`步骤 5`中，你查询了整个集合，通过以下代码检索所有文档：

```py
collection.find({})
```

空字典`{}`在`find()`中表示我们的过滤条件。当你传递一个空的过滤条件`{}`时，实际上是检索所有数据。这类似于 SQL 数据库中的`SELECT *`。另外，你也可以使用`collection.find()`来检索所有文档。

要在 MongoDB 中查询文档，你需要熟悉 MongoDB 查询语言（MQL）。通常，你会编写查询并将其传递给`find`方法，`find`方法类似于一个过滤器。

查询或过滤器采用**键值对**来返回匹配指定值的文档。以下是一个示例查询，查找收盘价大于 130 的股票：

```py
query = {"close": {"$gt": 130}}
results = collection.find(query)
```

结果对象实际上是一个**游标**，它还没有包含结果集。你可以遍历游标或将其转换为 DataFrame。通常，当执行`collection.find()`时，它返回一个**游标**（更具体地说，是一个`pymongo.cursor.Cursor`对象）。这个游标对象只是查询结果集的指针，允许你遍历结果。你可以使用`for`循环或`next()`方法（可以类比为 Python 中的迭代器）。然而，在这个示例中，我们没有直接循环遍历游标对象，而是将整个结果集便捷地转换成了 pandas DataFrame。

这是一个将结果集检索到 pandas DataFrame 的示例。

```py
df = pd.DataFrame(results)
df.info()
>>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1256 entries, 0 to 1255
Data columns (total 7 columns):
 #   Column  Non-Null Count  Dtype        
---  ------  --------------  -----        
 0   date    1256 non-null   datetime64[ns]
 1   close   1256 non-null   float64      
 2   _id     1256 non-null   object       
 3   low     1256 non-null   float64      
 4   volume  1256 non-null   int64        
 5   high    1256 non-null   float64      
 6   open    1256 non-null   float64      
dtypes: datetime64ns, float64(4), int64(1), object(1)
memory usage: 68.8+ KB
```

请注意，输出中添加了`_id`列，这是原始`MSFT.csv`文件中没有的。MongoDB 自动为集合中的每个文档添加了这个**唯一标识符**。

在前面的代码中，查询作为过滤器，仅检索`close`值大于`130`的数据。PyMongo 允许你传递一个字典（键值对），指定要检索的字段。下面是一个示例：

```py
query = {"close": {"$gt": 130}}
projection = {
   "_id": 0,
   "date":1,
   "close": 1,
   "volume": 1
}
results = collection.find(query, projection)
df = pd.DataFrame(results).set_index(keys='date')
print(df.head())
>>
               close    volume
date                           
2019-09-04  131.457260  17995900
2019-09-05  133.768707  26101800
2019-09-06  132.861359  20824500
2019-09-09  131.352219  25773900
2019-09-11  130.014984  24726100
```

在前面的代码中，我们指定了不返回`_id`，只返回`date`、`close`和`volume`字段。

最后，在我们前面的例子中，注意查询中使用的`$gt`。它表示“大于”，更具体地说，它翻译为*“大于 130”*。在 MQL 中，操作符以美元符号`$`开头。以下是 MQL 中常用操作符的示例列表：

+   `$eq` - 匹配等于指定值的值

    +   示例：查询`{"close": {"$eq": 130}}`查找`close`字段值恰好为 130 的文档。

+   `$gt` - 匹配大于指定值的值

    +   示例：查询`{"close": {"$gt": 130}}`查找收盘价大于 130 的文档。

+   `$gte` - 匹配大于或等于指定值的值

    +   示例：查询`{"close": {"$gte": 130}}`查找收盘价大于或等于 130 的文档。

+   `$lt` - 匹配小于指定值的值

    +   示例：查询`{"close": {"$lt": 130}}`查找收盘价小于 130 的文档。

+   `$lte` - 匹配小于或等于指定值的值

    +   示例：查询`{"close": {"$lt3": 130}}`查找收盘价小于或等于 130 的文档。

+   `$and` - 使用逻辑**AND**操作符连接查询子句，所有条件必须为真。

    +   示例：查询`{"$and": [{"close": {"$gt": 130}}, {"volume": {"$lt": 20000000}}]}`查找收盘价大于 130**且**交易量小于 20,000,000 的文档。

+   `$or` - 使用逻辑**OR**操作符连接查询子句，至少有一个条件必须为真。

    +   示例：查询`{"$or": [{"close": {"$gt": 135}}, {"volume": {"$gt": 30000000}}]}`查找收盘价大于 135**或**交易量大于 30,000,000 的文档。

+   `$in` - 匹配数组（列表）中指定的值

    +   示例：查询`{"date": {"$in": [datetime.datetime(2019, 9, 4), datetime.datetime(2019, 9, 5), datetime.datetime(2019, 9, 6)]}}`查找日期字段与以下指定日期之一匹配的文档：2019 年 9 月 4 日；2019 年 9 月 5 日；2019 年 9 月 6 日。

要查看 MQL 中操作符的完整列表，您可以访问官方文档：[`www.mongodb.com/docs/manual/reference/operator/query/`](https://www.mongodb.com/docs/manual/reference/operator/query/)

### 还有更多……

有多种方法可以使用`PyMongo`从 MongoDB 中检索数据。在前面的部分，我们使用了`db.collection.find()`，它总是返回一个游标。正如我们之前讨论的，`find()`返回的是指定集合中所有匹配的文档。如果您只想返回匹配文档的第一个实例，那么`db.collection.find_one()`将是最佳选择，它会返回一个**字典**对象，而不是游标。请记住，这只会返回一个文档，示例如下：

```py
db.microsoft.find_one()
>>>
{'date': datetime.datetime(2019, 9, 4, 0, 0),
 'close': 131.45726013183594,
 '_id': ObjectId('66e30c09a07d56b6db2f446e'),
 'low': 130.35882921332006,
 'volume': 17995900,
 'high': 131.5145667829114,
 'open': 131.14205897649285}
```

在处理游标时，您有多种方法可以遍历数据：

+   使用 `pd.DataFrame(cursor)` 将数据转换为 pandas DataFrame，如以下代码所示：

```py
cursor = db.microsoft.find()
df = pd.DataFrame(cursor)
```

+   转换为 Python **list** 或 **tuple**：

```py
data = list(db.microsoft.find())
```

您还可以将 **Cursor** 对象转换为 Python 列表，然后将其转换为 pandas DataFrame，像这样：

```py
data = list(db.microsoft.find())
df = pd.DataFrame(data)
```

+   使用 `next()` 将指针移动到结果集中的下一个项目：

```py
cursor = db.microsoft.find()
cursor.next()
```

+   通过对象进行**循环**，例如，使用 `for` 循环：

```py
cursor = db.microsoft.find()
for doc in cursor:
    print(doc)
```

前面的代码将遍历整个结果集。如果您想遍历前 5 条记录，可以使用以下代码：

```py
cursor = db.microsoft.find()
for doc in cursor[0:5]:
    print(doc)
```

+   指定 **index**。在这里，我们打印第一个值：

```py
cursor = db.microsoft.find()
cursor[0]
```

请注意，如果提供了一个切片，例如 `cursor[0:1]`，这是一个范围，那么它将返回一个游标对象（而不是文档）。

### 另见

欲了解更多有关 **PyMongo** API 的信息，请参考官方文档，您可以在此找到：`pymongo.readthedocs.io/en/stable/index.html`。

## 从时间序列数据库读取数据

时间序列数据库，一种 **NoSQL** 数据库，专为时间戳或时间序列数据进行了优化，并在处理包含 IoT 数据或传感器数据的大型数据集时提供了更好的性能。过去，时间序列数据库的常见使用场景主要与金融股票数据相关，但它们的应用领域已经扩展到其他学科和领域。在本教程中，您将探索三种流行的时间序列数据库：**InfluxDB**、**TimescaleDB** 和 **TDEngine**。

**InfluxDB** 是一个流行的开源时间序列数据库，拥有一个庞大的社区基础。在本教程中，我们将使用本文写作时的 InfluxDB 最新版本，即 2.7.10 版本。最近的 InfluxDB 版本引入了 Flux 数据脚本语言，您将通过 Python API 使用该语言查询我们的时间序列数据。

**TimescaleDB** 是 PostgreSQL 的扩展，专门为时间序列数据进行了优化。它利用 PostgreSQL 的强大功能和灵活性，同时提供了额外的功能，专门为高效处理带时间戳的信息而设计。使用 TimescaleDB 的一个优势是，您可以利用 SQL 查询数据。TimescaleDB 是一个开源的时间序列数据库，在本教程中，我们将使用 TimescaleDB 最新的版本 2.16.1。

**TDEngine** 是一个开源的时间序列数据库，专为物联网（IoT）、大数据和实时分析设计。与 TimescaleDB 类似，TDEngine 使用 SQL 查询数据。在本教程中，我们将使用 TDEngine 的最新版本 3.3.2.0。

### 准备工作

本食谱假设您可以访问正在运行的 InfluxDB、TimeseriesDB 或 TDEngine 实例。您将安装适当的库来使用 Python 连接和与这些数据库交互。对于**InfluxDB** V2，您需要安装 `influxdb-client`；对于**TimescaleDB**，您需要安装 PostgreSQL Python 库 `psycopg2`（回想在本章的*从关系数据库读取数据*食谱中，我们使用了 `psycopg3`）；最后，对于**TDEngine**，您需要安装 `taospy`。

您可以通过以下方式使用 `pip` 安装这些库：

```py
pip install 'influxdb-client[ciso]'
pip install 'taospy[ws]'
pip install psycopg2
```

要使用 **conda** 安装，请使用以下命令：

```py
conda install -c conda-forge influxdb-client
conda install -c conda-forge taospy taospyws
conda install -c conda-forge psycopg2
```

如果您没有访问这些数据库的权限，那么最快速的方式是通过 Docker。以下是 InlfuxDB、TimescaleDB 和 TDEngine 的示例命令：

#### InfluxDB Docker 容器

要创建 InfluxDB 容器，您需要运行以下命令：

```py
docker run -d\
    --name influxdb-ch3 \
    -p 8086:8086 \
    influxdb:2.7.9-alpine
```

欲了解更多信息，您可以访问官方 Docker Hub 页面 [`hub.docker.com/_/influxdb`](https://hub.docker.com/_/influxdb)

一旦**influxdb-ch3**容器启动并运行，您可以使用您喜欢的浏览器导航到 `http://localhost:8086`，并继续设置，例如用户名、密码、初始组织名称和初始存储桶名称。

对于本食谱，我们将使用**国家海洋和大气管理局**（**NOAA**）的水位样本数据，时间范围为 2019 年 8 月 17 日至 2019 年 9 月 17 日，数据来源为圣塔莫尼卡和科约特溪。

在数据浏览器 UI 中，您可以运行以下**Flux**查询来加载样本数据集：

```py
import "influxdata/influxdb/sample"
sample.data(set: "noaaWater")
    |> to(bucket: "tscookbook")
```

在之前的代码片段中，NOAA 数据集已加载到初始设置时创建的 `tscookbook` 存储桶中。

关于如何加载样本数据或其他提供的*样本数据集*的说明，请参阅 InfluxDB 官方文档 [`docs.influxdata.com/influxdb/v2/reference/sample-data/`](https://docs.influxdata.com/influxdb/v2/reference/sample-data/)

#### TimescaleDB Docker 容器

要创建 TimescaleDB 容器，您需要运行以下命令：

```py
docker run -d \
    --name timescaledb-ch3 \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=password \
    timescale/timescaledb:latest-pg16
```

欲了解更多信息，您可以访问官方 Docker Hub 页面 [`hub.docker.com/r/timescale/timescaledb`](https://hub.docker.com/r/timescale/timescaledb)

一旦**timescaledb-ch3**容器启动并运行，您可以使用与*从关系数据库读取数据*食谱中的`准备工作`部分相同的说明加载 `MSFT.csv` 文件。

注意，默认的用户名是 `postgres`，密码是您在 Docker 命令中设置的密码。

![图 3\. – DBeaver TimescaleDB/Postgres 连接设置（应该与图 3.1 类似）](img/file29.png)

图 3\. – DBeaver TimescaleDB/Postgres 连接设置（应该与图 3.1 类似）

> 由于 TimescaleDB 基于 PostgreSQL，它也默认使用 5432 端口。因此，如果你已经在本地运行一个默认使用 5432 端口的 PostgreSQL 数据库，你可能会遇到 TimescaleDB 的端口冲突问题。在这种情况下，你可以选择修改 Docker 运行配置并更改端口。

#### TDEngine Docker 容器 [待删除部分]

要创建一个 TDEngine 容器，你需要运行以下命令：

```py
docker run -d \
   --name tdengine-ch3 \
   -p 6030-6060:6030-6060 \
   -p 6030-6060:6030-6060/udp \
   tdengine/tdengine:3.3.2.0
```

更多信息可以访问官方的 Docker Hub 页面 [`hub.docker.com/r/tdengine/tdengine`](https://hub.docker.com/r/tdengine/tdengine)

一旦 **tdengine-ch3** 容器启动并运行，你可以通过在容器 shell 中运行 `taosBenchmark` 命令来创建一个演示数据集。以下是从正在运行的容器内部访问 shell 并运行所需命令来安装和设置演示数据集的步骤：

```py
docker exec -it tdengine-ch3 /bin/bash
>>
root@9999897cbeb4:~# taosBenchmark
```

一旦演示数据集创建完成，你可以退出终端。现在你可以使用 DBeaver 验证数据是否已创建。你可以使用与 *从关系数据库读取数据* 食谱中 `准备工作` 部分相同的说明。

请注意，默认用户名是 `root`，默认密码是 `taosdata`

![图 3\. – DBeaver TDEngine 连接设置](img/file30.png)

图 3\. – DBeaver TDEngine 连接设置

现在你应该能看到一个名为 `test` 的 **数据库** 被创建，并且一个名为 `meters` 的 **超级表**，它包含 10,000 个 **子表**，命名为 `d0` 到 `d9999`，每个表包含大约 10,000 行和四列（`ts`、`current`、`voltage` 和 `phase`）。你可能无法在 DBeaver 导航窗格中看到 `meters` 超级表，但如果你运行以下 SQL 查询 "`SELECT COUNT(*) FROM test.meters;`"，它应该会输出 100,000,000 行（10,000 个子表乘以每个子表的 10,000 行）。

### 如何做到这一点…

本食谱将演示如何连接并与三种流行的时序数据库系统进行交互。

#### InfluxDB

我们将利用 `Influxdb_client` Python SDK 来访问 InfluxDB 2.x，它支持 pandas DataFrame 进行读取和写入功能。让我们开始吧：

1.  首先，让我们导入必要的库：

```py
from influxdb_client import InfluxDBClient
import pandas as pd
```

1.  要使用 `InfluxDBClient(url="http://localhost:8086", token=token)` 建立连接，你需要定义 `token`、`org` 和 `bucket` 变量：

```py
token = "c5c0JUoz-\
joisPCttI6hy8aLccEyaflyfNj1S_Kff34N_4moiCQacH8BLbLzFu4qWTP8ibSk3JNYtv9zlUwxeA=="
org = "ts"
bucket = "tscookbook"
```

可以将桶看作是关系数据库中的数据库。

1.  现在，你可以通过将 `url`、`token` 和 `org` 参数传递给 `InlfuxDBClient()` 来建立连接：

```py
client = InfluxDBClient(url="http://localhost:8086",
                        token=token,
                        org=org)
```

1.  接下来，你将实例化 `query_api`：

```py
query_api = client.query_api()
```

1.  传递你的 Flux 查询，并使用 `query_data_frame` 方法请求以 pandas DataFrame 格式返回结果：

```py
query = f'''
from(bucket: "tscookbook")
  |> range(start: 2019-09-01T00:00:00Z)
  |> filter(fn: (r) =>
        r._measurement == "h2o_temperature" and
        r.location == "coyote_creek" and
        r._field == "degrees"
  )
  |> movingAverage(n: 120)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
'''
result = client.query_api().query_data_frame(org=org, query=query)
```

1.  在前面的 Flux 脚本中，选择了度量 `h2o_temparature`，并且位置是 `coyote_creek`。现在让我们检查一下 DataFrame。请注意以下输出中的数据类型：

```py
result.info()
>>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3885 entries, 0 to 3884
Data columns (total 8 columns):
 #   Column        Non-Null Count  Dtype             
---  ------        --------------  -----             
 0   result        3885 non-null   object            
 1   table         3885 non-null   int64             
 2   _start        3885 non-null   datetime64[ns, UTC]
 3   _stop         3885 non-null   datetime64[ns, UTC]
 4   _time         3885 non-null   datetime64[ns, UTC]
 5   _measurement  3885 non-null   object            
 6   location      3885 non-null   object            
 7   degrees       3885 non-null   float64           
dtypes: datetime64ns, UTC, float64(1), int64(1), object(3)
memory usage: 242.9+ KB
```

1.  如果你只想检索时间和温度列，你可以更新 Flux 查询，如下所示：

```py
query = f'''
from(bucket: "tscookbook")
  |> range(start: 2019-09-01T00:00:00Z)
  |> filter(fn: (r) =>
        r._measurement == "h2o_temperature" and
        r.location == "coyote_creek" and
        r._field == "degrees"
  )
  |> movingAverage(n: 120)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "degrees"])
'''
result = client.query_api().query_data_frame( query=query)
result.head()
>>
result  table                     _time    degrees
0  _result      0 2019-09-01 11:54:00+00:00  64.891667
1  _result      0 2019-09-01 12:00:00+00:00  64.891667
2  _result      0 2019-09-01 12:06:00+00:00  64.816667
3  _result      0 2019-09-01 12:12:00+00:00  64.841667
4  _result      0 2019-09-01 12:18:00+00:00  64.850000
```

原始数据集包含 15,258 条观察数据，数据每 6 分钟收集一次，来源于两个站点（位置）。移动平均是基于 120 个数据点计算的。理解数据集的渐进性非常重要。最终的 DataFrame 包含 3885 条记录。

#### TimescaleDB

由于 TimescaleDB 基于 PostgreSQL，并且我们已经安装了 **psycopg2**，因此检索和查询数据的方式应该与示例 *从关系数据库中读取数据* 中使用的方法类似。

这里简要说明如何使用 pandas 的 from_sql 来实现：

1.  导入 SQLAlchemy 和 pandas

```py
import pandas as pd
from sqlalchemy import create_engine
```

1.  使用正确的连接字符串创建 PostgreSQL 后端的引擎对象。

```py
engine =\
    create_engine("postgresql+psycopg2://postgres:password@localhost:5432/postgres")
```

1.  最后，使用 `read_sql` 方法将查询结果集检索到 pandas DataFrame 中：

```py
query = "SELECT * FROM msft"
df = pd.read_sql(query,
                 engine,
                 index_col='date',
                 parse_dates={'date': '%Y-%m-%d'})
print(df.head())
```

TimescaleDB 提供了比 PostgreSQL 更多的优势，你将在 *第五章* *持久化时间序列数据到数据库* 中探索其中的一些优势。然而，查询 TimescaleDB 的体验与熟悉 SQL 和 PostgreSQL 的用户类似。

#### TDEngine

对于这个示例，让我们更新配置文件 `database.cfg`，根据 *技术要求* 包括一个 [TDENGINE] 部分，如下所示：

```py
[TDENGINE]
user=root
password=taosdata
url=http://localhost:6041
```

你将首先建立与 TDEngine 服务器的连接，然后对 **taosBenchmark** 演示数据集进行查询，该数据集在 *Getting Read* 部分中有所描述。

1.  从导入所需的库开始。

```py
import taosrest
import pandas as pd
```

1.  你将创建一个 Python 字典，存储所有连接数据库所需的参数值，如 `url`、`user` 和 `password`。

```py
import configparser
config = configparser.ConfigParser()
config.read('database.cfg')
params = dict(config['TDENGINE'])
```

1.  建立与服务器的连接。

```py
conn = taosrest.connect(**params)
```

1.  运行以下查询，并使用连接对象 conn 的 `query` 方法执行该查询。

```py
query = """
SELECT *
FROM test.meters
WHERE location = 'California.LosAngles'
LIMIT 100000;
"""
results = conn.query(query)
```

1.  你可以验证结果集中的行数和列名。

```py
results.rows
>>
100000
results.fields
>>
[{'name': 'ts', 'type': 'TIMESTAMP', 'bytes': 8},
 {'name': 'current', 'type': 'FLOAT', 'bytes': 4},
 {'name': 'voltage', 'type': 'INT', 'bytes': 4},
 {'name': 'phase', 'type': 'FLOAT', 'bytes': 4},
 {'name': 'groupid', 'type': 'INT', 'bytes': 4},
 {'name': 'location', 'type': 'VARCHAR', 'bytes': 24}]
```

1.  `results.data` 包含结果集中的值，但没有列标题。在将结果集写入 pandas DataFrame 之前，我们需要从 `results.fields` 捕获列名列表：

```py
cols = [col['name'] for col in results.fields ]
df = pd.DataFrame(results.data, columns=cols)
df = df.set_index('ts')
df.info()
>>
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 100000 entries, 2017-07-14 05:40:00 to 2017-07-14 05:40:05.903000
Data columns (total 5 columns):
 #   Column    Non-Null Count   Dtype 
---  ------    --------------   ----- 
 0   current   100000 non-null  float64
 1   voltage   100000 non-null  int64 
 2   phase     100000 non-null  float64
 3   groupid   100000 non-null  int64 
 4   location  100000 non-null  object
dtypes: float64(2), int64(2), object(1)
memory usage: 4.6+ MB
```

### 它是如何工作的……

TimescaleDB 和 TDEngine 都使用 SQL 来查询数据，而 InfluxDB 使用他们的专有查询语言 Flux。

InfluxDB 1.8x 引入了 **Flux** 查询语言，作为 **InfluxQL** 的替代查询语言，后者与 SQL 更加相似。InfluxDB 2.0 引入了 **bucket** 的概念，数据存储在这里，而 InfluxDB 1.x 将数据存储在数据库中。

在这个示例中，我们从创建一个 `InfluxDbClient` 实例开始，这样我们就可以访问 `query_api` 方法，进而获得包括以下方法：

+   `query()` 返回结果作为 **FluxTable**。

+   `query_csv()` 返回结果作为 CSV 迭代器（CSV 读取器）。

+   `query_data_frame()` 返回结果作为 pandas DataFrame。

+   `query_data_frame_stream()` 返回一个 pandas DataFrame 流作为生成器。

+   `query_raw()` 返回原始未处理的数据，格式为 `s` 字符串。

+   `query_stream()`类似于`query_data_frame_stream`，但它返回的是一个生成器流，其中包含`FluxRecord`。

在这个示例中，你使用了`client.query_api()`来获取数据，如下所示：

```py
result = client.query_api().query_data_frame(org=org, query=query)
```

你使用了`query_data_frame`，它执行一个同步的 Flux 查询并返回一个你熟悉的 pandas DataFrame。

请注意，我们在 Flux 查询中必须使用`pivot`函数来将结果转换为适合 pandas DataFrame 的表格格式。

```py
pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
```

让我们逐行分析前面的代码：

`pivot()`用于重塑数据，并将其从长格式转换为宽格式。

`rowKey`参数指定了哪一列用作每行的唯一标识符。在我们的示例中，我们指定了`["_time"]`，因此每行将有一个唯一的时间戳。

`columnKey`参数指定了哪一列的值将用于在输出中创建新列。在我们的示例中，我们指定了`["_field"]`来从字段名称创建列。

`valueColumn`参数指定了哪个列包含数值，我们指定了`"_value"`来填充新列中的相应值。

### 还有更多…

在使用**InfluxDB**和`influxdb-client`时，有一个额外的参数可以用来创建 DataFrame 索引。在`query_data_frame()`中，你可以将一个列表作为`data_frame_index`参数的参数传入，如下面的示例所示：

```py
result =\
query_api.query_data_frame(query=query,                  
                                    data_frame_index=['_time'])
result['_value'].head()
>>
_time
2021-04-01 01:45:02.350669+00:00    64.983333
2021-04-01 01:51:02.350669+00:00    64.975000
2021-04-01 01:57:02.350669+00:00    64.916667
2021-04-01 02:03:02.350669+00:00    64.933333
2021-04-01 02:09:02.350669+00:00    64.958333
Name: _value, dtype: float64
```

这将返回一个带有`DatetimeIndex`（`_time`）的时间序列 DataFrame。

### 另见

+   如果你是 Flux 查询语言的新手，可以查看官方文档中的*Flux 入门指南*：[`docs.influxdata.com/influxdb/v2.0/query-data/get-started/`](https://docs.influxdata.com/influxdb/v2.0/query-data/get-started/).

+   请参考官方的**InfluxDB-Client** Python 库文档，地址为 GitHub：[`github.com/influxdata/influxdb-client-python`](https://github.com/influxdata/influxdb-client-python).

+   要了解更多关于**TDEngine** Python 库的信息，请参阅官方文档：[`docs.tdengine.com/cloud/programming/client-libraries/python/`](https://docs.tdengine.com/cloud/programming/client-libraries/python/)
