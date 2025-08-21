# 第五章：5 将时间序列数据持久化到数据库

## 加入我们在 Discord 上的书籍社区

![](img/file0.png)

[`packt.link/zmkOY`](https://packt.link/zmkOY)

在完成一个**数据分析**任务后，通常会从源系统提取数据，进行处理、转换并可能建模，最后将结果存储到数据库中以实现持久化。你总是可以将数据存储在平面文件中或导出为 CSV，但在处理大量企业数据（包括专有数据）时，你需要一种更强大且安全的存储方式。**数据库**提供了多个优势：安全性（静态加密）、并发性（允许多个用户查询数据库而不影响性能）、容错性、**ACID**合规性、优化的读写机制、分布式计算和分布式存储。

在企业环境中，一旦数据被存储在数据库中，它可以跨不同部门共享；例如，财务、市场营销、销售和产品开发部门现在可以根据自己的需求访问存储的数据。此外，数据现在可以实现民主化，供不同角色的组织人员应用于各种用例，如业务分析师、数据科学家、数据工程师、市场分析师和商业智能开发人员。

在本章中，你将把时间序列数据写入数据库系统以实现持久化。你将探索不同类型的数据库（关系型和非关系型），并使用**Python**推送你的数据。

更具体地说，你将使用**pandas**库，因为你会通过使用 pandas 的**DataFrame**进行大部分的分析。你将学习如何使用 pandas 库将你的时间序列 DataFrame 持久化到数据库存储系统中。许多数据库提供 Python API 和连接器，最近，许多数据库已经支持 pandas DataFrame（用于读取和写入），因为它们的流行和主流应用。在本章中，你将使用关系型数据库、文档数据库、云数据仓库和专门的时间序列数据库。

本章的目的是让你通过与不同方法连接到这些数据库系统，亲身体验如何持久化时间序列 DataFrame。

以下是本章将涵盖的内容列表：

+   将时间序列数据写入关系型数据库

+   将时间序列数据写入 MongoDB

+   将时间序列数据写入 InfluxDB

+   将时间序列数据写入 Snowflake

> 写入数据库和权限
> 
> > 记住，当你安装数据库实例或使用云服务时，写入数据是直接的，因为你是所有者/管理员角色。
> > 
> > 在任何公司中，这种情况在他们的数据库系统中并不适用。你必须与数据库的所有者、维护者以及可能的 IT 人员、数据库管理员或云管理员对接。在大多数情况下，他们可以允许你将数据写入沙盒或开发环境。然后，一旦完成，可能是同一个团队或另一个团队（如 DevOps 团队）会检查代码并评估性能，之后才会将代码迁移到质量保证（QA）/用户验收测试（UAT）环境。一旦进入该环境，业务部门可能会参与测试并验证数据，以便获得批准。最终，它可能会被推广到生产环境，以便所有人都可以开始使用数据。

## 技术要求

本章将广泛使用 pandas 2.2.2（于 2024 年 4 月 10 日发布）。

在我们的旅程中，你将安装多个 Python 库来与 pandas 一起使用。这些库在每个配方的准备部分中都有说明。你还可以从 GitHub 仓库下载 Jupyter notebooks，网址为 [`github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook`](https://github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook) 来跟着一起练习。

> 你应该参考 *第三章* 中的 *技术要求* 部分，*从数据库中读取时间序列数据*。这包括创建一个 **配置文件**，如 `database.cfg`。

本章中所有的配方将使用相同的数据集。该数据集基于 2019 年 1 月至 2023 年 12 月的亚马逊股票数据，通过 `yfinance` 库获取，并以 pandas DataFrame 的形式存储。

首先安装 `yfinance` 库，你可以通过 **conda** 安装，方法如下：

```py
conda install -c conda-forge yfinance
```

你还可以通过 **pip** 安装，方法如下：

```py
pip install yfinance
```

为了了解这个库的工作原理，你将从使用 `yfinance` 拉取亚马逊股票数据开始。

```py
import yfinance as yf
amzn = yf.Ticker("AMZN")
amzn_hist = amzn.history(start="2019-01-01", end="2023-12-31")
amzn_hist.info()
>>
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1258 entries, 2019-01-02 00:00:00-05:00 to 2023-12-29 00:00:00-05:00
Data columns (total 7 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   Open          1258 non-null   float64
 1   High          1258 non-null   float64
 2   Low           1258 non-null   float64
 3   Close         1258 non-null   float64
 4   Volume        1258 non-null   int64 
 5   Dividends     1258 non-null   float64
 6   Stock Splits  1258 non-null   float64
dtypes: float64(6), int64(1)
memory usage: 78.6 KB
```

生成的 DataFrame 有七（7）列和 **1258** 行。它包含一个 `DatetimeIndex`，格式为 `2019-01-02 00:00:00-05:00`。我们将重点关注几列（Open、High、Low、Close 和 Volume），并将 `DatetimeIndex` 的日期时间格式更改为 `YYYY-MM-DD`：

```py
amzn_hist.index = amzn_hist.index.strftime('%Y-%m-%d')
amzn_hist = amzn_hist[['Open', 'High', 'Low', 'Close', 'Volume']]
print(amzn_hist.head())
>>
                 Open       High        Low      Close     Volume
Date                                                            
2019-01-02  73.260002  77.667999  73.046501  76.956497  159662000
2019-01-03  76.000504  76.900002  74.855499  75.014000  139512000
2019-01-04  76.500000  79.699997  75.915497  78.769501  183652000
2019-01-07  80.115501  81.727997  79.459503  81.475502  159864000
2019-01-08  83.234497  83.830498  80.830498  82.829002  177628000
```

基于前面的示例，我们可以通过创建一个可以在本章中调用的函数来概括这种方法：

```py
import yfinance as yf
def get_stock_data(ticker, start, end):
    stock_data = yf.Ticker(ticker)
    stock_data = stock_data.history(start=start, end=end)
    stock_data.index = stock_data.index.strftime('%Y-%m-%d')
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return stock_data
```

`get_stock_data` 函数将返回一个包含选定列和格式化 DatetimeIndex 的 pandas DataFrame。它需要三个输入：一个 `ticker` 符号，一个 `start` 日期和一个 `end` 日期。如果你想获取从 2024 年 1 月 1 日到今天的股票数据，只需将 `end` 参数传递为 `None`。下面是一个示例：

```py
msft = get_stock_data('MSFT', '2024-01-01', None)
```

这将提供从 2024 年 1 月 1 日到请求时的最新数据的股票数据。

## 将时间序列数据写入关系型数据库（PostgreSQL 和 MySQL）

在这个配方中，你将把 DataFrame 写入 PostgreSQL 等关系型数据库。对于 `SQLAlchemy` Python 库支持的任何关系型数据库系统，这种方法都是一样的。你将体验到 SQLAlchemy 如何使得切换后端数据库（称为 `dialect`）变得简单，而无需更改代码。SQLAlchemy 提供的抽象层使得你能够使用相同的代码在任何支持的数据库之间切换，例如从 PostgreSQL 切换到 Amazon Redshift。

SQLAlchemy 支持的关系型数据库（方言）示例包括以下内容：

+   Microsoft SQL Server

+   MySQL/MariaDB

+   PostgreSQL

+   Oracle

+   SQLite

此外，还可以安装并使用外部方言与 SQLAlchemy 配合使用，以支持其他数据库（方言），如 `Snowflake`、`Microsoft SQL Server` 和 `Google BigQuery`。请访问 SQLAlchemy 的官方网站，查看可用的方言列表：[`docs.sqlalchemy.org/en/14/dialects/`](https://docs.sqlalchemy.org/en/14/dialects/)。

### 准备工作

> 你应参考第 *3 章* 中的配方“*从关系型数据库中读取数据*”，以回顾连接 PostgreSQL 的不同方式。

在此配方中，你将使用 `yfinance` Python 库来拉取股票数据。

要使用 **conda** 安装这些库，请运行以下命令：

```py
>> conda install sqlalchemy psycopg
```

要使用 `pip` 安装这些库，请运行以下命令：

```py
>> pip install sqlalchemy
>> pip install pyscopg
```

本书的 GitHub 仓库中提供了文件，你可以在这里找到：[`github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook`](https://github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook)。

### 如何操作……

在此配方中，你将使用 `yfnance` 库从 2019 年 1 月到 2023 年 12 月获取亚马逊的股票数据，并将其存入一个 pandas DataFrame 中，然后将该 DataFrame 写入 PostgreSQL 数据库中的表：

1.  从调用 *技术要求* 部分中创建的 `get_stock_data` 函数开始。

```py
amzn_hist = get_stock_data('AMZN', '2019-01-01', '2023-12-31')
```

1.  你需要创建一个 SQLAlchemy **engine** 对象。该引擎告诉 SQLAlchemy 和 pandas 我们计划与之交互的方言（后端数据库）以及运行中数据库实例的连接详情。利用 `URL.create()` 方法，通过提供必要的参数（`drivername`、`username`、`password`、`host`、`port` 和 `database`）来创建一个格式正确的 URL 对象。这些参数存储在 `database.cfg` 文件中。

```py
from sqlalchemy import create_engine, URL
from configparser import ConfigParser
config = ConfigParser()
config.read('database.cfg')
config.sections()
params = dict(config['POSTGRESQL'])
url = URL.create('postgresql+psycopg', **params)
print(url)
>>
postgresql+psycopg://postgres:***@127.0.0.1/postgres
```

现在，你可以将 `url` 对象传递给 `create_engine`：

```py
engine = create_engine(url)
print(engine)
>>
Engine(postgresql+psycopg://postgres:***@127.0.0.1/postgres)
```

1.  让我们将`amz_hist` DataFrame 写入 PostgreSQL 数据库实例中的新`amzn`表。这是通过使用`DataFrame.to_sql()`写入函数来实现的，该函数利用 SQLAlchemy 的功能将 DataFrame 转换为合适的表模式，并将数据转换为适当的 SQL 语句（如`CREATE TABLE`和`INSERT INTO`），这些语句特定于方言（后端数据库）。如果表不存在，在加载数据之前会创建一个新表；如果表已存在，你需要提供如何处理表的指令。这是通过`if_exists`参数来完成的，参数可以接受以下选项之一：`'fail'`、`'replace'`或`'append'`。

```py
amzn_hist.to_sql('amzn',
            engine,
            if_exists='replace')
```

完成与前面的代码相同任务的另一种方法是利用`with`语句，这样你就无需管理连接。这通常是一个更优的做法。

```py
with engine.connect() as connection:
    hist.to_sql('amzn',
                    connection,
                    if_exists='replace')
```

一旦执行上述代码，一个新的`amzn`表会在默认的`postgres`数据库的 public 模式下创建（默认）。

你可以通过以下方式验证数据库中的内容：

```py
from sqlalchemy import text
query = """
SELECT EXISTS (
   SELECT FROM information_schema.tables
   WHERE  table_schema = 'public'
   AND    table_name   = 'amzn'
   );"""
with engine.connect() as conn:
    result = conn.execute(text(query))
print(result.fetchone())
>>
(True,)
```

注意使用`text()`函数包裹我们的查询。`text()`构造了一个新的`TextClause`，用来表示文本 SQL 字符串。

1.  通过查询`amzn`表并统计记录数来确认数据已写入数据库：

```py
query = "select count(*) from amzn;"
with engine.connect() as conn:
    result = conn.execute(text(query))
result.fetchone()
>>
(1258,)
```

1.  接下来，使用`get_stock_data`请求额外的亚马逊股价数据，这次是 2024 年的数据（例如，2024 年 1 月 1 日到 2024 年 9 月 23 日），并将其附加到现有的`amzn`表中。在这里，你将利用`to_sql()`写入函数中的`if_exists`参数。

```py
amzn_hist_2024 = get_stock_data('AMZN', '2024-01-01', None)
print(amzn_hist_2024.shape)
>>
(182, 5)
```

确保将`append`传递给`if_exists`参数，如以下代码所示：

```py
with engine.connect() as connection:
    amzn_hist_2024.to_sql('amzn',
                    connection,
                    if_exists='append')
```

1.  统计记录的总数，以确保我们已经将 182 条记录附加到原来的 1258 条记录中。你将运行与之前相同的查询，如以下代码所示：

```py
query = "select count(*) from amzn;"
with engine.connect() as conn:
    result = conn.execute(text(query))
print(result.fetchone())
>>
(1440,)
```

确实，你可以观察到所有 1440 条记录都已写入`amzn`表。

### 它是如何工作的……

使用`DataFrame.to_sql()`写入函数，SQLAlchemy 在后台处理许多细节，比如创建表的模式、插入记录并提交到数据库。

使用 pandas 和 SQLAlchemy 向关系型数据库写入和读取数据非常相似。我们在*第三章*的*从关系型数据库读取数据*章节中讨论了使用 SQLAlchemy 读取数据。许多讨论的概念同样适用于这里。

我们总是从`create_engine`开始，并指定**方言**（后端数据库）。`to_sql()`函数会将 DataFrame 的数据类型映射到适当的 PostgreSQL 数据类型。使用**对象关系映射器（ORM）**如 SQLAlchemy 的优势在于，它提供了一个抽象层，使你无需担心将 DataFrame 模式转换为特定数据库模式。

在前面的例子中，你在 `DataFrame.to_sql()` 函数中使用了 `if_exists` 参数，并传递了两个不同的参数：

1.  最初，你将值设置为 `replace`，这会在表存在时覆盖该表。如果我们将此覆盖操作转换为 SQL 命令，它会执行 `DROP TABLE`，然后是 `CREATE TABLE`。如果你已经有一个包含数据的表并打算向其中添加记录，这可能会很危险。因此，如果不传递任何参数，默认值会设置为 `fail`。这种默认行为会在表已存在时抛出错误。

1.  在食谱的第二部分，计划是将额外的记录插入到现有表中，并且你将参数从 `replace` 更新为 `append`。

当你使用 `yfinance` 拉取股票数据时，它会自动将 `Date` 字段指定为 `DatetimeIndex`。换句话说，`Date` 不是一列，而是一个索引。在 `to_sql()` 中，默认行为是将 DataFrame 的索引作为数据库中的一列写入，这由 `index` 参数控制。这个参数是布尔类型，默认值为 `True`，表示将 DataFrame 索引作为列写入。

另一个非常有用的参数是 `chunksize`。默认值为 `None`，表示一次性将 DataFrame 中的所有行写入数据库。如果你的数据集非常庞大，可以使用 `chunksize` 参数批量写入数据库；例如，设置 `chunksize` 为 500 会一次性批量写入 500 行数据。

```py
with engine.connect() as connection:
    amzn_hist.to_sql('amzn',
                    connection,
                    chunksize=500,
                    if_exists='append')
```

### 还有更多…

使用 pandas 的 `read_sql`、`read_sql_table`、`read_sql_query` 和 `to_sql` I/O 函数时，它们需要一个 SQLAlchemy 连接对象（SQLAlchemy 引擎）。要使用 SQLAlchemy 连接到目标数据库，你需要为特定的数据库（例如 Amazon Redshift、Google BigQuery、MySQL、MariaDB、PostgreSQL 或 Oracle）安装相应的 Python DBAPI（驱动程序）。这样你可以一次编写脚本，并且仍然能够与 SQLAlchemy 支持的其他方言（后端数据库）一起使用。为了演示这一点，我们将扩展最后一个例子。

**Amazon Redshift** 是一款流行的云数据仓库数据库，基于 PostgreSQL 架构，并在其基础上做了多项增强，包括列存储以加速分析查询。你将探索 SQLAlchemy 的简便性，以及其他将 pandas DataFrame 写入 Amazon Redshift 的选项。

#### 使用 SQLAlchemy 向 Amazon Redshift 写入数据

这次你将使用相同的代码，但写入 Amazon Redshift 数据库。除了运行 MySQL 实例之外，唯一的要求是安装适用于的 Python DBAPI（驱动程序）。

Amazon Redshift。注意，`sqlalchemy-redshift` 需要 `psycopg2`。

要使用 **conda** 安装，请运行以下命令：

```py
conda install -c conda-forge psycopg2 sqlalchemy-redshift
```

要使用 **pip** 安装，请运行以下命令：

```py
pip install pip install psycopg2 sqlalchemy-redshift
```

你将使用相同的代码来操作 PostgreSQL；唯一的不同是 SQLAlchemy 引擎，它使用的是 Amazon Redshift 的 DBAPI。首先从配置文件中加载连接参数。在这个示例中，配置存储在 `database.cfg` 文件中。

```py
[AWS]
host=yourendpoint
port=5439
database=dev
username=username
password=password
```

使用 `ConfigParser` 和 `URL` 提取参数并构建 URL：

```py
from configparser import ConfigParser
config = ConfigParser()
config.read('database.cfg')
config.sections()
params = dict(config['AWS'])
from sqlalchemy import URL, create_engine
url = URL.create('redshift+psycopg2', **params)
print(url)
>>
redshift+psycopg2://awsuser:***@redshift-cluster-1.cltc17lacqp7.us-east-1.redshift.amazonaws.com:5439/dev
```

你现在可以使用以下代码创建引擎：

```py
aws_engine = create_engine(url)
```

使用 `yfinance` 库创建一个基于过去 **5 年** 股票数据的新 `amzn_hist` DataFrame：

```py
amzn = yf.Ticker("AMZN")
amzn_hist = amzn.history(period="5y")
amzn_hist = amzn_hist[['Open', 'High', 'Low', 'Close', 'Volume']]
```

在写入 DataFrame 之前，我们需要重置索引。这将使我们恢复 Date 列。我们这样做是因为 Amazon Redshift 不支持传统的索引，因为它是列式数据库（相反，你可以定义一个 **排序键**）。

```py
amzn_hist = amzn_hist.reset_index()
with aws_engine.connect() as conn:
    amzn_hist.to_sql('amzn',
                    conn,
                    if_exists='replace', index=False)
```

请注意前面的代码中的 `index=False`。这是因为 `to_sql` 会写入 DataFrame 中的索引对象，默认情况下 `index=True`。当你重置 DataFrame 索引时，它会将 DatetimeIndex 移动到 Date 列，并用 `RangeIndex`（从 0 到 1257）替换索引。使用 `index=False` 确保我们不会尝试将 `RangeIndex` 写入 Amazon Redshift。

最后，你可以验证写入的记录总数：

```py
from sqlalchemy import text
query = "select count(*) from amzn;"
with aws_engine.connect() as conn:
    result = conn.execute(text(query))
result.fetchone()
>>
(1258,)
```

#### 使用 redshift_connector 写入 Amazon Redshift

在这个示例中，你将使用一个不同的库，即 `redshift_connector`。你首先需要安装该库。

你可以使用 **conda** 安装：

```py
conda install -c conda-forge redshift_connector
```

你也可以使用 **pip** 安装它：

```py
pip install redshift_connector
```

请注意，`redshift_connector` 需要一个 `user` 参数，这与 SQLAlchemy 需要 `username` 参数不同。为此，你可以在配置文件中创建一个新部分。下面是一个示例：

```py
[AWS2]
host=yourendpoint
port=5439
database=dev
user=username
password=password
```

以下代码从 `database.cfg` 文件中读取参数，并将这些参数传递给 `redshift_connector.connect()` 来创建连接对象。

```py
import redshift_connector
from configparser import ConfigParser
config = ConfigParser()
config.read('database.cfg')
config.sections()
params = dict(config['AWS2'])
conn = redshift_connector.connect(**params)
```

你将创建一个游标对象，它提供对 `write_dataframe` 方法的访问。

```py
cursor = conn.cursor()
cursor.write_dataframe(amzn_hist, 'amzn')
```

最后，你将提交事务。

```py
conn.commit()
```

请注意，`write_dataframe` 方法没有提供指定追加、替换/覆盖或失败行为的参数，正如你在 SQLAlchemy 中所看到的那样。`write_dataframe` 方法期望在 Amazon Redshift 中已存在的表进行追加。

#### 使用 AWS SDK for pandas 写入 Amazon Redshift

**awswrangler** 库或 AWS SDK for pandas 可以轻松地与多个 AWS 服务（如 Athena、Glue、Redshift、Neptune、DynamoDB、EMR、S3 等）集成。

你可以使用 **conda** 安装该库：

```py
conda install -c conda-forge awswrangler
```

你也可以使用 **pip** 安装：

```py
pip install 'awswrangler[redshift]'
```

你可以利用在上一节中创建的 `conn` 对象，*使用 redshift_connector 写入 Amazon Redshift*。

```py
import awswrangler as wr
wr.redshift.to_sql(
    df=amzn_hist,
    table='amzn',
    schema='public',
    con=conn,
    mode='overwrite'
)
```

请注意，mode 参数支持三种（3）不同的选项：`overwrite`、`append` 或 `upsert`。

### 另请参见

这里有一些额外的资源：

+   若要了解更多关于 `DataFrame.to_sql()` 函数的信息，可以访问 [`pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html)。

+   要了解更多关于**SQLAlchemy**的功能，你可以先阅读它们的功能页面：[`www.sqlalchemy.org/features.html`](https://www.sqlalchemy.org/features.html)。

+   要了解`awswrangler`，你可以访问他们的 GitHub 仓库：[`github.com/aws/aws-sdk-pandas`](https://github.com/aws/aws-sdk-pandas)

## 将时间序列数据写入 MongoDB

**MongoDB**是一个文档数据库系统，它以**BSON**格式存储数据。当你从 MongoDB 查询数据时，数据将以 JSON 格式呈现。BSON 与 JSON 类似，它是 JSON 的二进制编码格式。不过，BSON 不像 JSON 那样是人类可读的格式。JSON 非常适合传输数据，且与系统无关，而 BSON 则专为存储数据并与 MongoDB 相关联。

在这个实例中，你将学习如何将一个 pandas DataFrame 写入 MongoDB。

### 准备工作

> 你应该参考*第三章*中“*从文档数据库读取数据*”这一实例，以便复习连接 MongoDB 的不同方法。

在*第三章*中“*从文档数据库读取数据*”这一实例中，我们安装了`pymongo`。在本实例中，你将再次使用该库。

要通过**conda**安装，请运行以下命令：

```py
$ conda install -c anaconda pymongo -y
```

要通过**pip**安装，请运行以下命令：

```py
$ python -m pip install pymongo
```

本书的 GitHub 仓库中提供了该文件，你可以在此找到：[`github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook`](https://github.com/PacktPublishing/Time-Series-Analysis-with-Python-Cookbook)。

### 如何操作…

要将数据存储到 MongoDB 中，你需要创建一个**数据库**和一个**集合**。一个数据库包含一个或多个集合，这些集合类似于关系数据库中的表。一旦创建集合，你将以文档形式写入数据。集合包含文档，文档相当于关系数据库中的行。

1.  首先，导入必要的库：

```py
import pandas as pd
from pymongo import MongoClient
```

1.  创建一个`MongoClient`实例以建立与数据库的连接：

```py
client = MongoClient('mongodb://localhost:27017')
```

1.  创建一个名为`stock_data`的新数据库，并创建一个名为`daily_stock`的**时间序列集合**。

首先，我们在 MongoDB 中创建一个**常规集合**：

```py
db = client['stock_data']
collection = db.create_collection('amazon')
```

这将创建一个名为`stock_data`的新数据库，并创建一个名为 amazon 的集合。如果`stock_data`已存在，它将把`amazon`集合添加到现有数据库中。

然而，由于我们处理的是时间序列数据，我们可以通过创建一个**时间序列集合**来更高效地存储和查询数据。从 MongoDB 5.0 版本开始，时间序列集合已针对带时间戳的数据进行了优化。我们可以修改之前的代码来创建一个`daily_stock`时间序列集合。

```py
db = client['stock_data']
ts = db.create_collection(
    name="daily_stock",
    timeseries={
        "timeField": "Date",
        "metaField": "symbol",
        "granularity": "hours"
    }
)
```

通过此次更新，我们现在使用时间序列集合，这提高了时间数据（如股票价格）的存储效率和查询性能。以后，我们将使用`ts`引用来与时间序列集合进行交互。

1.  你将利用在技术要求部分创建的`get_stock_data`函数，拉取从 2019 年 1 月 1 日到 2024 年 8 月 31 日的亚马逊股票数据：

```py
amzn_hist = get_stock_data('AMZN', '2019-01-01', '2024-8-31')
```

1.  在 pandas 中，我们以表格格式处理数据，其中每一列代表一个变量，每一行代表一个数据点。而 MongoDB 则将数据存储为类似 JSON 的格式（BSON），其中每个文档是一个独立的记录，可以包含时间戳、元数据和其他键值对。

在将数据插入 MongoDB 之前，你需要将 DataFrame 转换为一个字典列表，每个字典（或文档）表示一个股票数据点。每个字典将包含时间戳（`Date`）、股票信息（例如`High`、`Low`）和元数据（例如`"ticker": "AMZN"`）。

你将探索两个选项：第一个选项使用`to_dict()`方法，第二个选项是遍历 DataFrame。

让我们来探索第一个选项：

```py
metadata = {"ticker": "AMZN"}
amzn_hist['metadata'] = [metadata] * len(amzn_hist)
amzn_hist = amzn_hist.reset_index()
amzn_hist['Date'] = pd.to_datetime(amzn_hist['Date'])
amzn_records = amzn_hist.to_dict(orient='records')
amzn_records[0:2]
>>
[{'Date': Timestamp('2019-01-02 00:00:00'),
  'Open': 73.26000213623047,
  'High': 77.66799926757812,
  'Low': 73.04650115966797,
  'Close': 76.95649719238281,
  'Volume': 159662000,
  'metadata': {'ticker': 'AMZN'}},
 {'Date': Timestamp('2019-01-03 00:00:00'),
  'Open': 76.00050354003906,
  'High': 76.9000015258789,
  'Low': 74.85549926757812,
  'Close': 75.01399993896484,
  'Volume': 139512000,
  'metadata': {'ticker': 'AMZN'}}]
```

在这里，我们假设所有数据都有相同的元数据信息（例如`"ticker": "AMZN"`）。

> `to_dict()`方法中`orient`参数的默认值是`dict`，它会生成一个**字典**，其格式为`{column -> {index -> value}}`。
> 
> > 另一方面，使用`records`作为值，会生成一个**列表**，它遵循`[{column -> value}, … , {column -> value}]`的模式。

现在，让我们来探索第二个选项，它提供了更多的灵活性，可以为特定字段添加数据或对单个记录进行转换，例如，基于不同的股票代码值：

```py
amzn_hist = amzn_hist.reset_index()
amzn_records = []
for idx, row in amzn_hist.iterrows():
    doc = {
        "Date": pd.to_datetime(row['Date']),
        "metadata": {"ticker": "AMZN"},
        "High": row['High'],
        "Low": row['Low'],
        "Close": row['Close'],
        "Open": row['Open'],
        "Volume": row['Volume']
    }
    amzn_records.append(doc)
amzn_records[0:2]
>>
[{'Date': Timestamp('2019-01-02 00:00:00'),
  'metadata': {'ticker': 'AMZN'},
  'High': 77.66799926757812,
  'Low': 73.04650115966797,
  'Close': 76.95649719238281,
  'Open': 73.26000213623047,
  'Volume': 159662000},
 {'Date': Timestamp('2019-01-03 00:00:00'),
  'metadata': {'ticker': 'AMZN'},
  'High': 76.9000015258789,
  'Low': 74.85549926757812,
  'Close': 75.01399993896484,
  'Open': 76.00050354003906,
  'Volume': 139512000}]
```

现在，你有了一个长度为`1426`的 Python 列表（每个记录是一个字典）：

```py
len(amzn_records)
>>
1426
```

1.  现在，你已经准备好使用`insert_many()`方法将数据写入时间序列`daily_stock`集合中：

```py
result = ts.insert_many(amzn_records)
```

1.  你可以使用以下代码验证数据库和集合是否已创建：

```py
client.list_database_names()
>>
['admin', 'config', 'local', 'stock_data']
db.list_collection_names()
>>
['daily_stock', 'system.buckets.daily_stock', 'system.views']
```

1.  接下来，拉取微软的股票数据（MSFT）并将其添加到同一个`daily_stock`时间序列集合中。稍后，你将探索如何利用元数据在查询数据时区分不同的股票代码（如 AMZN 与 MSFT）。

```py
msft_hist = get_stock_data('MSFT', '2019-01-01', '2024-8-31')
metadata = {"ticker": "MSFT"}
msft_hist['metadata'] = [metadata] * len(msft_hist)
msft_hist = msft_hist.reset_index()
msft_hist['Date'] = pd.to_datetime(msft_hist['Date'])
msft_records = msft_hist.to_dict(orient='records')
result = ts.insert_many(msft_records)
```

你可以通过查询数据库来检查写入的文档总数，如下面的代码所示：

```py
ts.count_documents({})
>>
2852
```

1.  现在，集合中包含了两个股票代码的数据。你可以使用元数据在查询中为每个代码进行筛选。你将首先查询`daily_stock`集合，检索仅包含微软（MSFT）股票数据的记录。这时，`metadata`字段变得非常有用，允许你按股票代码进行筛选。让我们首先定义一个日期范围，然后仅查询 MSFT 的数据。

```py
from datetime import datetime
# Define date range
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 1, 31)
# Query for MSFT stock data within the date range
results = ts.find({
    "metadata.ticker": "MSFT",
    "Date": {"$gte": start_date, "$lte": end_date}
})
# Convert the query results to a DataFrame
msft_df = (pd.DataFrame(results)
             .set_index('Date')
             .drop(columns=['_id', 'metadata']))
print(msft_df.head())
>>
                Close       High       Open    Volume        Low
Date                                                           
2019-01-02  95.501335  96.096327  94.018571  35329300  93.442465
2019-01-03  91.988037  94.623014  94.538011  42579100  91.799146
2019-01-04  96.266327  96.814101  94.179125  44060600  93.433020
2019-01-07  96.389107  97.531873  95.992445  35656100  95.369122
2019-01-08  97.087975  98.192962  97.314637  31514400  96.058536
```

1.  你还可以进行聚合计算每个股票代码的平均`Close`价格：

```py
msft_avg_close = ts.aggregate([
    {"$group":
         {"_id": "$metadata.ticker",
          "avgClose":
                  {"$avg": "$Close"}}
    }
])
for doc in msft_avg_close:
    print(doc)
>>
{'_id': 'AMZN', 'avgClose': 133.4635473361022}
{'_id': 'MSFT', 'avgClose': 252.4193055419066}
```

### 它是如何工作的…

`PyMongo`提供了两个插入函数，用于将我们的记录作为文档写入集合。这些函数如下：

+   `insert_one()`会将一个文档插入到集合中。

+   `insert_many()`会将多个文档插入到集合中。

在前面的示例中，你使用了`insert_many()`并同时传入了要写入的数据作为文档。然而，在执行此操作之前，将 DataFrame 转换为符合`[{column -> value}, … , {column -> value}]`模式的字典列表格式是至关重要的。这是通过在`to_dict()` DataFrame 方法中使用`orient='records'`来完成的。

当文档插入数据库时，它们会被分配一个唯一的`_id`值。如果文档尚未拥有`_id`，MongoDB 会在插入操作中自动生成一个。你可以捕获生成的`_id`，因为插入函数会返回一个结果对象——对于单次插入是`InsertOneResult`，对于批量插入是`InsertManyResult`。以下代码演示了如何使用`insert_one`和`InsertOneResult`类来实现这一点：

```py
one_record = amzn_records[0]
one_record
>>
{'Date': Timestamp('2019-01-02 00:00:00'),
 'metadata': {'ticker': 'AMZN'},
 'High': 77.66799926757812,
 'Low': 73.04650115966797,
 'Close': 76.95649719238281,
 'Open': 73.26000213623047,
 'Volume': 159662000}
result_id = ts.insert_one(one_record)
result_id
>>
InsertOneResult(ObjectId('66f2ed5efad8cbd88968d02e'), acknowledged=True)
```

返回的对象是`InsertOneResult`的一个实例；要查看实际的值，你可以使用`insert_id`属性：

```py
result_id.inserted_id
>>
ObjectId('66f2ed5efad8cbd88968d02e')
```

如果你有分钟级的股票数据，你可以利用`granularity`属性，它可以是`seconds`、`minutes`或`hours`。

### 还有更多内容…

在前面的示例中，如果你运行以下代码查询数据库以列出可用的集合，你将看到三个集合：

```py
db = client['stock_data']
db.list_collection_names()
>>
['daily_stock', 'system.buckets.daily_stock', 'system.views']
```

你创建了`daily_stock`集合，那么另外两个集合是什么呢？让我们首先探讨一下 MongoDB 中的**桶模式**。

桶模式是一种数据建模技术，用来优化数据在数据库中的存储方式。默认情况下，当你将 DataFrame 转换为字典列表时，实际上是将每个 DataFrame 记录（数据点）作为一个独立的 MongoDB 文档插入。这会在记录和文档之间创建一对一的映射。

然而，**桶策略**允许你将相关的数据点分组到一个文档中。例如，如果你有每小时的数据，你可以将它们分组到一个桶中，比如 24 小时周期，并将该时间范围内的所有数据存储在一个文档中。同样，如果我们有来自多个设备的传感器数据，你可以使用桶模式将数据（例如按设备 ID 和时间范围）分组，并将它们作为一个文档插入。这将减少数据库中文档的数量，提高整体性能，并简化查询。

当你创建一个时间序列集合时，MongoDB 会自动应用桶模式，以高效的格式存储数据。让我们分解一下：

+   `daily_stock`：这是你创建的主要时间序列集合。它充当视图，允许你使用标准的 MongoDB 操作与时间序列数据进行交互。

+   `system.buckets.daily_stock`：这是一个内部集合，MongoDB 使用桶模式存储实际的时间序列数据。MongoDB 会自动为时间序列集合实现这一策略，以提高存储和查询性能。它是如何工作的呢：

    +   文档会根据时间戳和元数据字段（例如，股票符号）被分组到“桶”中。

    +   每个桶包含时间上接近的数据点，并共享相同的元数据值。

    +   这种分桶策略显著减少了存储的文档数量，提高了查询效率并减少了磁盘使用。

+   `system.views`：这是一个系统集合，MongoDB 用来存储数据库中所有视图的信息，包括你的时间序列集合的视图。

为了更好地理解桶模式的应用，我们将通过创建一个新的集合（常规集合，而非时间序列集合），并按年和月将每日股票数据分桶：

```py
db = client['stock_data']
bucket = db.create_collection(name='stock_bucket')
```

接下来，让我们创建一个新的 DataFrame，并添加两个额外的列：**month** 和 **year**：

```py
amzn = yf.Ticker("AMZN")
amzn_hist = amzn.history(period="5y")
amzn_hist = amzn_hist[['Open',
                       'High',
                       'Low',
                       'Close',
                       'Volume']].reset_index()
amzn_hist['Date'] = pd.to_datetime(amzn_hist['Date'])
amzn_hist['month'] = amzn_hist['Date'].dt.month
amzn_hist['year'] = amzn_hist['Date'].dt.year
```

在前面的代码中，你向 DataFrame 添加了 `month` 和 `year` 列，并创建了一个名为 `stocks_bucket` 的新集合。在接下来的代码段中，你将循环遍历数据，并将按年和月分组的数据作为一个单一文档写入：

```py
for year in amzn_hist['year'].unique():
    for month in amzn_hist['month'].unique():
        record = {}
        record['month'] = int(month)
        record['year'] = int(year)
        record['symbol'] = 'AMZN'
        try:
            prices = amzn_hist[(amzn_hist['month'] == month) & (amzn_hist['year'] == year)]['Close'].values
            record['price'] = [float(price) for price in prices]
        except Exception as e:
            print(f"Error processing data for {month}/{year}: {str(e)}")
            continue
        else:
            bucket.insert_one(record)
```

在代码中，你遍历了唯一的年和月组合，然后为每个组合创建一个包含月、年、符号和收盘价列表的记录字典。然后，该记录被插入到 `stock_bucket` 集合中，有效地按月和年对数据进行了分桶。

为了说明文档数量的差异，请运行以下代码：

```py
print('without bucketing: ',
      db.daily_stock.count_documents({}))
print('with bucketing: ',
      db.stock_bucket.count_documents({}))
>>
without bucketing:  2853
with bucketing:  72
```

请注意，`stock_bucket` 集合包含 72 个文档，代表按年和月分组的数据。

要查询 2024 年和 6 月的数据，并查看文档如何表示，请使用以下代码示例：

```py
results = pd.DataFrame(bucket.find({'year':2024, 'month': 6}))
results['price'].to_dict()[0]
>>
[178.33999633789062,
 179.33999633789062,
 181.27999877929688,
 185.0,
 184.3000030517578,
 187.05999755859375,
 187.22999572753906,
 186.88999938964844,
 183.8300018310547,
 183.66000366210938,
 184.05999755859375,
 182.80999755859375,
 186.10000610351562,
 189.0800018310547,
 185.57000732421875,
 186.33999633789062,
 193.61000061035156,
 197.85000610351562,
 193.25]
```

你也可以使用 MongoDB Compass 运行相同的查询，结果应与图示中显示的类似：

![图示 – 使用 MongoDB Compass 查询 stock_bucket 集合](img/file41.png)

图示 – 使用 MongoDB Compass 查询 stock_bucket 集合

### 另见

+   欲了解更多关于在 MongoDB 中存储时间序列数据和分桶的信息，你可以参考这篇 MongoDB 博客文章：[`www.mongodb.com/blog/post/time-series-data-and-mongodb-part-2-schema-design-best-practices`](https://www.mongodb.com/blog/post/time-series-data-and-mongodb-part-2-schema-design-best-practices)

+   查看 MongoDB 手册，了解更多关于**时间序列集合**的信息 [`www.mongodb.com/docs/manual/core/timeseries-collections/`](https://www.mongodb.com/docs/manual/core/timeseries-collections/)

## 写入时间序列数据到 InfluxDB

在处理大规模时间序列数据时，如传感器或**物联网**（**IoT**）数据，你需要一种更高效的方式来存储和查询这些数据，以便进行进一步的分析。这就是**时间序列数据库**的优势所在，因为它们专门为处理复杂且非常大的时间序列数据集而构建。

在这个示例中，我们将以**InfluxDB**为例，演示如何写入时间序列数据库。

### 准备工作

> 你应该参考第三章“*从数据库中读取时间序列数据*”中的教程“*从时间序列数据库中读取数据*”，以便复习连接 InfluxDB 的不同方式。

你将使用`ExtraSensory`数据集，这是由加利福尼亚大学圣地亚哥分校提供的一个移动感应数据集：*Vaizman, Y., Ellis, K., 和 Lanckriet, G. “从智能手机和智能手表识别复杂的人类背景”。IEEE Pervasive Computing, vol. 16, no. 4, 2017 年 10 月至 12 月, pp. 62-74\. doi:10.1109/MPRV.2017.3971131*

你可以在这里下载数据集：[`extrasensory.ucsd.edu/#download`](http://extrasensory.ucsd.edu/#download)

数据集包含 60 个文件，每个文件代表一个参与者，并通过唯一标识符（UUID）来识别。每个文件包含 278 列：225 列（特征）、51 列（标签）和 2 列（时间戳和标签来源）。

本教程的目标是演示如何将时间序列 DataFrame 写入 InfluxDB。在这个教程中，选择了两列：时间戳（日期范围从`2015-07-23`到`2016-06-02`，共覆盖 152 天）和手表加速度计读数（以毫 G 为单位测量）。

在你可以在 Python 中与 InfluxDB 交互之前，你需要安装 InfluxDB Python 库。

你可以通过运行以下命令，使用**pip**安装该库：

```py
$ pip install 'influxdb-client[ciso]'
```

要使用**conda**安装，请使用以下命令：

```py
conda install -c conda-forge influxdb-client
```

### 如何操作…

你将通过读取**ExtraSensory**数据集中的一个文件（针对特定的 UUID）来开始这个教程，重点关注一个特征列——手表加速度计。你将进行一些数据转换，为将时间序列 DataFrame 写入 InfluxDB 做准备：

1.  首先加载所需的库：

```py
from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from  pathlib import Path
```

1.  数据集由 60 个压缩 CSV 文件（`csv.gz`）组成，你可以使用`pandas.read_csv()`读取这些文件。`read_csv`的默认`compression`参数设置为`infer`，这意味着 pandas 会根据文件扩展名推断使用哪种压缩或解压协议。文件扩展名为（`gz`），pandas 会使用这个扩展名来推断需要使用的解压协议。或者，你也可以通过`compression='gzip'`明确指定使用哪种压缩协议。

在以下代码中，你将读取这些文件中的一个，选择`timestamp`和`watch_acceleration:magnitude_stats:mean`两列，重命名这些列，最后，针对所有**na**（缺失）值执行**回填**操作：

```py
path = Path('../../datasets/Ch5/ExtraSensory/')
file = '0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv.gz'
columns = ['timestamp',
           'watch_acceleration:magnitude_stats:mean']
df = pd.read_csv(path.joinpath(file),
                usecols=columns,
                compression='gzip')
df = df.bfill()
df.columns = ['timestamp','wacc']
df.shape
>>
(3960, 2)
```

从前面的输出中，你有来自那个文件的`3960`个传感器读数。

1.  为了将数据写入 InfluxDB，你需要至少一个`measurement`列和一个`timestamp`列。目前，时间戳是一个 Unix 时间戳（**epoch**），以秒为单位捕获，这是一个可接受的写入 InfluxDB 的数据格式。例如，`2015-12-08 7:06:37 PM`在数据集中以`1449601597`的形式存储。

**InfluxDB** 在磁盘上以纪元纳秒存储时间戳，但在查询数据时，InfluxDB 将数据显示为**RFC3339 UTC 格式**，以使其更易读。因此，在**RFC3339**中`1449601597`将表示为`2015-12-08T19:06:37+00:00.000Z`。请注意 InfluxDB 中的精度为*纳秒*。

在以下步骤中，您将把 Unix 时间戳转换为在**pandas**中更易读的格式，这也是 InfluxDB 中可接受的格式：

```py
df['timestamp'] = pd.to_datetime(df['timestamp'],
                                  origin='unix',
                                  unit='s',
                                  utc=True)
df.set_index('timestamp', inplace=True)
print(df.head())
>>
                                  wacc
timestamp                            
2015-12-08 19:06:37+00:00   995.369977
2015-12-08 19:07:37+00:00   995.369977
2015-12-08 19:08:37+00:00   995.369977
2015-12-08 19:09:37+00:00   996.406005
2015-12-08 19:10:55+00:00  1034.180063
```

在上述代码中，`unit`参数设置为`'s'`用于**秒**。这指示 pandas 基于起点计算秒数。`origin`参数默认设置为`unix`，因此转换将计算到 Unix 纪元开始的秒数。`utc`参数设置为`True`，这将返回一个**UTC**的`DatetimeIndex`类型。我们的 DataFrame 索引的`dtype`现在是`datetime64[ns, UTC]`。

> 您可以在*第六章*中的*Chapter 6*中的*Working with Unix epoch timestamps*中的食谱中了解有关 Unix 纪元时间戳的更多信息

1.  接下来，您需要建立与运行的 InfluxDB 数据库实例的连接。您只需传递您的 API 读/写令牌即可。在写入数据库时，您需要指定 bucket 和组织名称：

```py
bucket = "sensor"
org = "<yourorg>"
token = "<yourtoken>"
client = InfluxDBClient(url="http://localhost:8086",
                        token=token,
                        org=org)
```

1.  初始化`write_api`并配置`WriterOptions`。包括指定`writer_type`为`SYNCHRONOUS`，`batch_size`和`max_retries`，在失败之前：

```py
writer = client.write_api(WriteOptions(SYNCHRONOUS,
                     batch_size=500,
                     max_retries=5))
writer.write(bucket=bucket,
                record=df,
                write_precision='ns',
                data_frame_measurement_name='wacc',
                data_frame_tag_columns=[])
```

1.  要验证数据是否正确写入，您可以使用`query_data_frame`方法查询数据库，如以下代码所示：

```py
query = '''
         from(bucket: "sensor")
         |> range(start: 2015-12-08)
         |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
         '''
result = client.query_api()
influx_df = result.query_data_frame(
                             org=org,
                             query=query,
                             data_frame_index='_time')
```

检查返回的 DataFrame：

```py
Influx_df.info()
>>
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 3960 entries, 2015-12-08 19:06:37+00:00 to 2015-12-11 18:48:27+00:00
Data columns (total 6 columns):
 #   Column        Non-Null Count  Dtype             
---  ------        --------------  -----             
 0   result        3960 non-null   object            
 1   table         3960 non-null   int64             
 2   _start        3960 non-null   datetime64[ns, UTC]
 3   _stop         3960 non-null   datetime64[ns, UTC]
 4   _measurement  3960 non-null   object            
 5   wacc          3960 non-null   float64           
dtypes: datetime64ns, UTC, float64(1), int64(1), object(2)
memory usage: 216.6+ KB
```

请注意 DataFrame 有两个`datetime64[ns, UTC]`类型的列。

1.  现在您完成了，可以关闭您的写入对象并关闭客户端，如下所示：

```py
writer.close()
client.close()
```

### 工作原理…

在使用`write_api`将 pandas DataFrame 写入 InfluxDB 之前，您需要在 InfluxDB 中定义几个必需的事项。包括以下内容：

+   **测量**：这些是您要跟踪的值。InfluxDB 每个数据点接受一个测量。

+   **字段**：我们不需要明确指定字段，因为未在标签定义中的任何列将被标记为字段。字段作为键值对存储的元数据对象。与标签不同，字段不被索引。

+   **标签**（可选）：一个元数据对象，您可以指定要索引以提高查询性能的列。这也存储为键值对。

**WriteAPI** 支持 *同步* 和 *异步* 写入。此外，**WriteAPI** 在写入 InfluxDB 时还提供了多个选项（例如，行协议字符串、行协议字节、数据点结构、字典样式，以及对 `pandas` DataFrame 的支持）。在第三章的 *从时间序列数据库读取数据* 示例中，你使用了 `query_data_frame()` 方法，指定查询结果应该作为 **pandas DataFrame** 返回。

同样，`write_api` 在将 `pandas` DataFrame 写入 InfluxDB 时提供了额外的参数：

+   `data_frame_measurement_name`：用于写入 `pandas` DataFrame 的测量名称

+   `data_frame_tag_columns`：作为标签的 DataFrame 列表；其余列将作为字段

### 还有更多内容…

在前面的示例中，我们需要手动使用 `writer.close()` 刷新数据，并使用 `client.close()` 终止连接。为了更好的资源管理（例如，自动关闭连接）和异常处理，你可以使用 `with` 语句，享受它带来的便利。以下示例展示了如何以更清晰、更高效的格式重写相同的代码：

```py
with InfluxDBClient(url="http://localhost:8086", token=token) as client:
    with client.write_api(WriteOptions(SYNCHRONOUS,
                     batch_size=500,
                     max_retries=5_000)) as writer:

        writer.write(bucket=bucket,
                        org=org,
                        record=df,
                        write_precision='ns',
                        data_frame_measurement_name='wacc',
                        data_frame_tag_columns=[])
```

### 另见

+   要了解更多关于 InfluxDB 行协议的信息，请参阅他们的文档：[`docs.influxdata.com/influxdb/v2.0/reference/syntax/line-protocol/`](https://docs.influxdata.com/influxdb/v2.0/reference/syntax/line-protocol/)。

+   要了解更多关于 InfluxDB 2.x Python API 的信息，请参阅官方文档：[`docs.influxdata.com/influxdb/cloud/tools/client-libraries/python/`](https://docs.influxdata.com/influxdb/cloud/tools/client-libraries/python/)。

## 将时间序列数据写入 Snowflake

**Snowflake** 已成为构建大数据分析的热门云数据库选项，因其可扩展性、性能以及 SQL 导向（列存储关系数据库）。

Snowflake 的 Python 连接器简化了与数据库的交互，无论是读取数据还是写入数据，特别是对 `pandas` DataFrame 的内建支持。在这个示例中，你将使用在 *将时间序列数据写入 InfluxDB* 示例中准备的传感器物联网数据集。这项技术适用于任何你打算写入 Snowflake 的 `pandas` DataFrame。

### 准备工作

> 你可以参考 *第三章* 中的 *从 Snowflake 读取数据* 示例，作为回顾，了解连接到 Snowflake 的不同方式。

对于 `snowflake-connector-python` 库，推荐的安装方法是使用 **pip**，这允许你安装诸如 `pandas` 等 *额外* 组件，如下所示：

```py
pip install snowflake-sqlalchemy snowflake-snowpark-python
pip install "snowflake-connector-python[pandas]"
```

你也可以通过 **conda** 安装，但如果你想在使用 `snowflake-connector-python` 与 pandas 时，必须使用 pip 安装。

```py
conda install -c conda-forge snowflake-sqlalchemy snowflake-snowpark-python
conda install -c conda-froge snowflake-connector-python
```

创建一个配置文件，例如 `database.cfg`，用于存储你的 Snowflake 连接信息，如下所示：

```py
[SNOWFLAKE]
ACCOUNT=<YOURACCOUNT>
USER=<YOURUSERNAME>
PASSWORD= <YOURPASSWORD>
WAREHOUSE=COMPUTE_WH
DATABASE=TSCOOKBOOK
SCHEMA=CHAPTER5
ROLE=<YOURROLE>
```

使用 `ConfigParser`，提取 `[SNOWFLAKE]` 部分的内容，以避免暴露或硬编码你的凭证。读取 `[SNOWFLAKE]` 部分下的参数，并将其转换为 Python 字典，如下所示：

```py
config = ConfigParser()
config.read('database.cfg)
config.sections()
params = dict(config['SNOWFLAKE'])
```

你将利用在*技术要求*部分创建的 `get_stock_data` 函数，拉取 **Amazon** 从 2019 年 1 月 1 日到 2024 年 8 月 31 日的股票数据：

```py
amzn_hist = get_stock_data('AMZN', '2019-01-01', '2024-8-31')
```

`amzn_hist` DataFrame 没有 **Date** 列，而是有一个 `DatetimeIndex`。由于 API 不支持写入索引对象，你需要将索引转换为一列。

```py
amzn_hist = amzn_hist.reset_index()
amzn_hist.shape
>>
(1426, 6)
```

在本食谱中，你将引用 `amzn_hist` DataFrame 和对象 `params`。

### 如何操作…

我们将探索三种方法和库来连接到 Snowflake 数据库。你将首先使用 **Snowflake Python 连接器**，然后探索 **Snowflake** **SQLAlchemy**，最后探索 **Snowpark Python API**。让我们开始吧。

#### 使用 snowflake-connector-python（write_pandas）

本节中的食谱将利用 snowflake-connector-python 库来连接并将数据写入 Snowflake 数据库。

1.  导入本食谱所需的库：

```py
import pandas as pd
from snowflake import connector
from snowflake.connector.pandas_tools import pd_writer, write_pandas
```

`pands_tools` 模块提供了几个用于处理 pandas DataFrame 的函数，其中包括两个写入方法（`write_pandas` 和 `pd_writer`）。

`write_pandas` 是一个将 pandas DataFrame 写入 Snowflake 数据库的方法。背后，该函数会将数据存储为 **Parquet** 文件，将文件上传到 **临时阶段**，然后通过 `COPY INTO` 命令将数据从文件插入到指定的表中。

另一方面，`pd_writer` 方法是一个插入方法，用于通过 `DataFrame.to_sql()` 方法将数据插入到 Snowflake 数据库，并传递一个 SQLAlchemy 引擎。在本食谱后面的*使用 SQLAlchemy* 部分，你将探索 `pd_writer`。

1.  建立与 Snowflake 数据库实例的连接，并创建一个游标对象：

```py
con = connector.connect(**params)
cursor = con.cursor()
```

`cursor` 对象将用于执行 SQL 查询，以验证数据集是否已正确写入 Snowflake 数据库。

1.  使用 writer_pandas 方法将 amzn_hist DataFrame 写入 Snowflake。该方法接受连接对象 con、DataFrame、目标表名及其他可选参数，如 auto_create_table 和 table_type 等。

```py
success, nchunks, nrows, copy_into = write_pandas(
                                            con,
                                            amzn_hist,
                                            auto_create_table=True,
                                            table_name='AMAZON',
                                            table_type='temporary')
```

使用 write_pandas 时，它返回一个元组。在之前的代码中，我们将元组解包为：success、nchunks、nrows 和 copy_into。让我们查看这些对象内部的值：

```py
print('success: ', success)
print('number of chunks: ', nchunks)
print('number of rows: ', nrows)
print('COPY INTO output', copy_into)
>>
success:  True
number of chunks:  1
number of rows:  1426
COPY INTO output [('ntporcytgv/file0.txt', 'LOADED', 1426, 1426, 1, 0, None, None, None, None)]
```

`success` 对象是一个布尔值（True 或 False），用于指示函数是否成功将数据写入指定的表中。`nchunks` 表示写入过程中的块数，在本例中，整个数据作为一个块写入。`nrows` 表示函数插入的行数。最后，`copy_into` 对象包含 `COPY INTO` 命令的输出。

注意 `auto_create_table=True` 的使用，如果没有设置为 True 并且表 AMAZON 在 Snowflake 中不存在，`write_pandas` 会抛出一个错误。当设置为 `True` 时，我们明确要求 `write_pandas` 创建该表。此外，如果表已经存在，你可以使用 `overwrite=True` 参数指定是否希望覆盖现有的表。

`table_type` 支持 Snowflake 中的 **永久**、**临时** 和 **瞬态** 表类型。该参数可以采用以下值：`'temp'`、`'temporary'` 和 `'transient'`。如果传递空字符串 `table_type=''`，则会创建一个 **永久** 表（默认行为）。

1.  你可以进一步验证所有 1426 条记录是否已写入临时表：

```py
cursor.execute('SELECT count(*) FROM AMAZON;')
count = cursor.fetchone()[0]
print(count)
>>
1426
```

实际上，你已将所有 1426 条记录写入 Snowflake 中的 `AMAZON` 表。

#### 使用 SQLAlchemy

本节中的示例将使用 snowflake-sqlalchemy 和 snowflake-connector-python 库来连接和将数据写入 Snowflake 数据库。

1.  导入本示例所需的库：

```py
import pandas as pd
from snowflake.connector.pandas_tools import pd_writer
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
```

1.  你将使用 Snowflake SQLAlchemy 库中的 `URL` 函数来构建连接字符串并创建 SQLAlchemy **引擎**，以便与 Snowflake 实例建立连接：

```py
url = URL(**params)
engine = create_engine(url)
```

1.  使用 `to_sql()` 写入函数将数据框架写入 Snowflake 数据库。你需要传递一个 **插入方法**；在此情况下，你将传递 `pd_writer`：

```py
try:
    amzn_hist.to_sql(
    'amazon_alchemy',
    engine,
    index=False,
    if_exists='replace'
)
except:
    print('failed to write')
```

上述代码使用标准的 `SQL INSERT` 子句，每一行一个。Snowflake 连接器 API 提供了一个插入方法 `pd_writer`，你可以将其传递给 `to_sql` 方法中的方法参数，如下所示：

```py
try:
    amzn_hist.to_sql(
    'amazon_alchemy',
    engine,
    index=False,
    if_exists='replace',
    method=pd_writer
)
except:
    print('failed to write')
```

在幕后，`pd_writer` 函数将使用 `write_pandas` 函数将数据框架写入 Snowflake 数据库。

1.  要读取并验证数据是否已写入，你可以使用 `pandas.read_sql()` 来查询表：

```py
pd.read_sql_table('amazon_alchemy',
                  con=engine).info()
>>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1426 entries, 0 to 1425
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Date    1426 non-null   object
 1   Open    1426 non-null   float64
 2   High    1426 non-null   float64
 3   Low     1426 non-null   float64
 4   Close   1426 non-null   float64
 5   Volume  1426 non-null   float64
dtypes: float64(5), object(1)
memory usage: 67.0+ KB
```

新的数据框架包含所有 1426 条记录以及预期的确切列数。

#### 使用 snowflake-snowpark-python

本节中的示例将使用 Snowpark API 来写入 pandas 数据框架。

1.  导入本示例所需的库：

```py
from snowflake.snowpark import Session
import pandas as pd
```

1.  通过与 Snowflake 数据库建立连接来创建一个会话。

```py
session = Session.builder.configs(params).create()
```

1.  在写入数据框架之前，你必须将 pandas 数据框架转换为 Snowpark 数据框架。

```py
amzn_snowpark_df = session.create_dataframe(amzn_hist)
```

Snowpark 数据框架采用延迟计算，并提供了许多相对于 Panda 数据框架的优势。

1.  要写入 Snowpark 数据框架，你可以使用 `write` 和 `save_as_table` 方法：

```py
amzn_snowpark_df.write.mode("overwrite").save_as_table("amazon_snowpark")
```

1.  要读取并验证数据是否已写入，你可以使用 `session.table` 来查询表：

```py
amzn_df = session.table("amazon_snowpark")
```

如果你更习惯使用 pandas，你可以使用 `to_pandas` 方法将 Snowpark 数据框架转换为 pandas 数据框架，如下所示：

```py
df = amzn_df.to_pandas()
df.shape
>>
(1426, 6)
```

数据框架包含所有 1426 条记录和预期的六（6）列。

### 它是如何工作的…

Snowflake Python API 提供了两种将 pandas 数据框架写入 Snowflake 的机制，这些机制包含在 `pandas_tools` 模块中：

```py
from snowflake.connector.pandas_tools import pd_writer, write_pandas
```

在这个食谱中，你使用了 `pd_writer` 并将其作为**插入方法**传递给 `DataFrame.to_sql()` 写入函数。当在 `to_sql()` 中使用 `pd_writer` 时，你可以通过 `if_exists` 参数来改变插入行为，该参数有三个选项：

+   `fail`，如果表格已存在，则引发`ValueError`错误

+   `replace`，在插入新值之前会删除表格

+   `append`，将数据插入到现有表格中

如果表格不存在，SQLAlchemy 会为你创建表格，并将 pandas DataFrame 中的数据类型映射到 Snowflake 数据库中的相应数据类型。通过 `pandas.read_sql()` 使用 SQLAlchemy 引擎从 Snowflake 读取数据时也适用此规则。

请注意，`pd_writer` 在后台使用了 `write_pandas` 函数。它们的工作方式都是将 DataFrame 转储为 Parquet 文件，上传到临时阶段，最后通过 `COPY INTO` 将数据复制到表格中。

当你使用 Snowpark API 写入 DataFrame 时，你使用了 `write.mode()` 方法。`mode()` 方法接受不同的写入模式选项：

+   `append`：将 DataFrame 的数据追加到现有表格。如果表格不存在，它将被创建。

+   `overwrite`：通过**删除**旧表来覆盖现有表格。

+   `truncate`：通过**截断**旧表来覆盖现有表格。

+   `errorifexists`：如果表格已存在，则抛出异常**错误**。

+   `ignore`：如果表格已存在，则忽略该操作。

请记住，默认值是 `errorifexists`。

### 还有更多...

在 Snowpark 中有一个有用的方法，可以直接写入 pandas DataFrame，而无需将其转换为 Snowpark DataFrame：

```py
snowpark_df = session.write_pandas(amzn_hist,
                               table_name="amazon_temp",
                               auto_create_table=True,
                               table_type="temp")
```

`write_pandas` 函数将 pandas DataFrame 写入 Snowflake，并返回一个 Snowpark DataFrame 对象。

### 另见

+   访问 Snowflake 文档，了解更多关于 `write_pandas` 和 `pd_write` 方法的信息：[`docs.snowflake.com/en/user-guide/python-connector-api.html#write_pandas`](https://docs.snowflake.com/en/user-guide/python-connector-api.html#write_pandas)。

+   你可以在这里了解更多关于 `pandas DataFrame.to_sql()` 函数的信息：[`pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html)。

+   要了解更多关于 Snowpark API 中 `write_pandas` 方法的信息，请参阅官方文档：`docs.snowflake.com/en/developer-guide/snowpark/reference/python/1.22.1/snowpark/api/snowflake.snowpark.Session.write_pandas`
