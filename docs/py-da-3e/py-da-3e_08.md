# 第七章：检索、处理和存储数据

数据无处不在，呈现各种形态和形式。我们可以从网络、物联网传感器、电子邮件、FTP 和数据库中获取数据。我们还可以通过实验室实验、选举民调、市场调查和社会调查等方式收集数据。作为数据专业人员，你应该知道如何处理各种数据集，因为这是一个非常重要的技能。本章将讨论如何检索、处理和存储不同类型的数据。本章概述了如何获取不同格式的数据，如 CSV、Excel、JSON、HDF5、Parquet 和 `pickle`。

有时候，我们需要在数据分析之前或之后存储或保存数据。我们还将学习如何访问来自关系型和**NoSQL**（**非关系型数据库**）数据库的数据，如 `sqlite3`、MySQL、MongoDB、Cassandra 和 Redis。在 21 世纪的网络世界中，NoSQL 数据库在大数据和 Web 应用程序中正在经历快速增长。它们提供了更灵活、更快速、无需架构的数据库。NoSQL 数据库可以存储各种格式的数据，如文档风格、列式存储、对象、图形、元组或它们的组合。

本章涵盖的主题如下：

+   使用 NumPy 读取和写入 CSV 文件

+   使用 pandas 读取和写入 CSV 文件

+   从 Excel 读取和写入数据

+   从 JSON 读取和写入数据

+   从 HDF5 读取和写入数据

+   从 HTML 表格读取和写入数据

+   从 Parquet 读取和写入数据

+   从 `pickle pandas` 对象读取和写入数据

+   使用 `sqlite3` 进行轻量级访问

+   从 MySQL 读取和写入数据

+   从 MongoDB 读取和写入数据

+   从 Cassandra 读取和写入数据

+   从 Redis 读取和写入数据

+   PonyORM

# 技术要求

本章有以下技术要求：

+   你可以在以下 GitHub 链接中找到代码和数据集：[`github.com/PacktPublishing/Python-Data-Analysis-Third-Edition/tree/master/Chapter06`](https://github.com/PacktPublishing/Python-Data-Analysis-Third-Edition/tree/master/Chapter06)。

+   所有代码块都可以在 `ch6.ipynb` 文件中找到。

+   本章使用以下文件进行实践：CSV 文件（`demo.csv`、`product.csv`、`demo_sample_df.csv`、`my_first_demo.csv` 和 `employee.csv`）、Excel 文件（`employee.xlsx`、`employee_performance.xlsx` 和 `new_employee_details.xlsx`）、JSON 文件（`employee.json` 和 `employee_demo.json`）、HTML 文件（`country.html`）、`pickle` 文件（`demo_obj.pkl`）、HDF5 文件（`employee.h5`）和 Parquet 文件（`employee.parquet`）。

+   本章将使用以下 Python 库：`pandas`、`pickle`、`pyarrow`、`sqlite3`、`pymysql`、`mysql-connector`、`pymongo`、`cassandra-driver` 和 `redis`。

# 使用 NumPy 读取和写入 CSV 文件

在第二章，《*NumPy 和 pandas*》中，我们详细研究了 NumPy 库并探索了许多功能。NumPy 也有读取和写入 CSV 文件的功能，并能以 NumPy 数组的形式输出结果。`genfromtxt()`函数将帮助我们读取数据，而`savetxt()`函数将帮助我们将数据写入文件。由于其两阶段操作，`genfromtxt()`函数比其他函数慢。第一阶段，它将数据以字符串类型读取，第二阶段，它将字符串类型转换为适当的数据类型。`genfromtxt()`具有以下参数：

+   `fname`：字符串；文件名或文件路径。

+   `delimiter`：字符串；可选，分隔字符串值。默认情况下，它采用连续的空白字符。

+   `skip_header`：整数；可选，表示要跳过的文件开头的行数。

让我们看一个读取和写入 CSV 文件的例子：

```py
# import genfromtxt function
from numpy import genfromtxt

# Read comma separated file
product_data = genfromtxt('demo.csv', delimiter=',')

# display initial 5 records
print(product_data)
```

这将产生以下输出：

```py
[[14\. 32\. 33.]
 [24\. 45\. 26.]
 [27\. 38\. 39.]]
```

在前面的代码示例中，我们使用 NumPy 模块的`genfromtxt()`方法读取了`demo.csv`文件：

```py
# import numpy
import numpy as np

# Create a sample array
sample_array = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])

# Write sample array to CSV file
np.savetxt("my_first_demo.csv", sample_array, delimiter=",")
```

在前面的代码示例中，我们使用 NumPy 模块的`savetxt()`方法写入了`my_first_demo.csv`文件。

让我们在下一节中看看如何使用`pandas`模块读取 CSV 文件。

# 使用 pandas 读取和写入 CSV 文件

`pandas`库提供了多种文件读取和写入选项。在本节中，我们将学习如何读取和写入 CSV 文件。为了读取 CSV 文件，我们将使用`read_csv()`方法。让我们看一个例子：

```py
# import pandas
import pandas as pd

# Read CSV file
sample_df=pd.read_csv('demo.csv', sep=',' , header=None)

# display initial 5 records
sample_df.head()
```

这将产生以下输出：

![](img/2f3288ff-42bd-4c76-9674-b5c9d94c0888.png)

现在我们可以使用以下代码将数据框保存为 CSV 文件：

```py
# Save DataFrame to CSV file
sample_df.to_csv('demo_sample_df.csv')
```

在前面的示例代码中，我们使用`pandas`模块的`read_csv()`和`to_csv(0)`方法读取和保存了 CSV 文件。

`read_csv()`方法有以下重要参数：

+   `filepath_or_buffer`：提供作为字符串的文件路径或 URL，用于读取文件。

+   `sep`：提供字符串中的分隔符，例如逗号为`','`，分号为`';'`。默认分隔符是逗号`','`。

+   `delim_whitespace`：用于空白分隔符的备用参数。它是一个布尔变量。`delim_whitespace`的默认值为`False`。

+   `header`：用于标识列名。默认值为`infer`。

+   `names`：可以传递一个列名列表。`names`的默认值为`None`。

在`pandas`中，可以使用`to_csv()`方法将 DataFrame 导出为 CSV 文件。CSV 文件是逗号分隔值文件。此方法可以仅使用一个参数（文件名作为字符串）运行：

+   `path_or_buf`：文件将导出的文件路径或位置。

+   `sep`：用于输出文件的分隔符。

+   `header`：包含列名或列别名列表（默认值：`True`）。

+   `index`：将索引写入文件（默认值：`True`）。

更多参数和详细描述，请访问[`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html)。接下来，我们将看看如何使用`pandas`模块读取 Excel 文件。

# 从 Excel 读取和写入数据

Excel 文件是商业领域中广泛使用的文件类型。在 Python 中，Excel 文件可以通过`pandas`的`read_excel()`函数轻松读取。`read_excel()`函数需要传入文件路径和`sheet_name`参数来读取数据：

```py
# Read excel file
df=pd.read_excel('employee.xlsx',sheet_name='performance')

# display initial 5 records
df.head()
```

这将产生以下输出：

![](img/9458050f-19fc-47fd-95a9-0f411cfb3ef6.png)

DataFrame 对象可以写入 Excel 工作表中。我们可以使用`to_excel()`函数将 DataFrame 对象导出到 Excel 工作表中。通常，`to_excel()`函数的参数与`to_csv()`相同，除了`sheet_name`参数：

```py
df.to_excel('employee_performance.xlsx')
```

在前面的代码示例中，我们将一个单独的 DataFrame 导出到了 Excel 工作表中。我们还可以将多个 DataFrame 导出到一个文件中，并使用不同的工作表名称。我们还可以使用`ExcelWriter`在一个 Excel 文件中写入多个 DataFrame（每个 DataFrame 在不同的工作表上），如下所示：

```py
# Read excel file
emp_df=pd.read_excel('employee.xlsx',sheet_name='employee_details')

# write multiple dataframes to single excel file
with pd.ExcelWriter('new_employee_details.xlsx') as writer:
    emp_df.to_excel(writer, sheet_name='employee')
    df.to_excel(writer, sheet_name='perfromance')
```

在前面的代码示例中，我们将多个 DataFrame 写入了一个 Excel 文件。这里，每个 DataFrame 存储在不同的工作表上，使用了`ExcelWriter`函数。接下来，我们将看看如何使用`pandas`模块读取 JSON 文件。

# 从 JSON 读取和写入数据

**JSON**（**JavaScript 对象表示法**）文件是用于 Web 应用和服务器之间交换数据的广泛使用的格式。它作为数据交换格式，比 XML 更易读。`pandas`提供了`read_json`函数来读取 JSON 数据，以及`to_json()`函数来写入 JSON 数据：

```py
# Reading JSON file
df=pd.read_json('employee.json')

# display initial 5 records
df.head()
```

这将产生以下输出：

![](img/b694891d-1d89-4264-bf15-5412ba97b2fc.png)

在前面的代码示例中，我们使用`read_json()`方法读取了 JSON 文件。接下来，我们将看看如何写入一个 JSON 文件：

```py
# Writing DataFrame to JSON file
df.to_json('employee_demo.json',orient="columns")
```

在前面的代码示例中，我们使用`to_json()`方法写入了 JSON 文件。在`to_json()`方法中，`orient`参数用于处理输出字符串格式。`orient`提供了 record、column、index 和 value 等格式。你可以在官方网页上查看更多详细信息，网址是：[`pandas.pydata.org/pandas-docs/version/0.24.2/reference/api/pandas.DataFrame.to_json.html`](https://pandas.pydata.org/pandas-docs/version/0.24.2/reference/api/pandas.DataFrame.to_json.html)。接下来，我们将进入 HDF5 文件部分，在下一节中，我们将学习如何使用`pandas`模块读取和写入 HDF5 文件。

# 从 HDF5 读取和写入数据

**HDF** 代表 **层次化数据格式**。HDF 旨在以高性能存储和管理大量数据。它提供了快速的 I/O 处理和异构数据的存储。有多种 HDF 文件格式可用，如 HDF4 和 HDF5。HDF5 与读取和写入 `pandas` DataFrame 的字典对象相同。它使用 PyTables 库的 `read_hdf()` 函数来读取 HDF5 文件，使用 `to_hdf()` 函数来写入：

```py
# Write DataFrame to hdf5
df.to_hdf('employee.h5', 'table', append=True)
```

在前面的代码示例中，我们使用 `to_hdf()` 方法写入了 HDF 文件格式。`'table'` 是用于表格格式的格式参数。表格格式可能会较慢，但提供了更灵活的操作，例如搜索和选择。`append` 参数用于将输入数据追加到现有数据文件中：

```py
# Read a hdf5 file
df=pd.read_hdf('employee.h5', 'table')

# display initial 5 records
df.head()
```

这将生成以下输出：

![](img/3ea05ef4-fb90-4f8e-ab93-0d15496e2aef.png)

在前面的代码示例中，我们使用 `read_hdf()` 方法读取了 HDF 文件格式。接下来让我们看看如何从网站读取和写入 HTML 表格。

# 从 HTML 表格读取和写入数据

HTML 表格将行存储在 `<tr>...</tr>` 标签中，每行都有相应的 `<td>...</td>` 单元格来存储值。在 `pandas` 中，我们还可以从文件或 URL 中读取 HTML 表格。`read_html()` 函数从文件或 URL 中读取 HTML 表格，并将 HTML 表格返回为 `pandas` DataFrame 列表：

```py
# Reading HTML table from given URL
table_url = 'https://en.wikipedia.org/wiki/List_of_sovereign_states_and_dependent_territories_in_North_America'
df_list = pd.read_html(table_url)

print("Number of DataFrames:",len(df_list))
```

这将生成以下输出：

```py
Number of DataFrames: 7
```

在前面的代码示例中，我们使用 `read_html()` 方法从给定的网页读取了 HTML 表格。`read_html()` 将返回所有表格，作为 DataFrame 列表。接下来让我们检查列表中的一个 DataFrame：

```py
# Check first DataFrame
df_list[0].head()
```

这将生成以下输出：

![](img/12053630-a477-4f75-9354-f324711171b0.png)

在前面的代码示例中，我们展示了给定网页上第一个表格的前五条记录。类似地，我们还可以使用 `to_html()` 将 DataFrame 对象写入 HTML 表格。`to_html()` 将内容渲染为 HTML 表格：

```py
# Write DataFrame to raw HTML
df_list[1].to_html('country.html')
```

通过前面的代码示例，我们可以将任何 DataFrame 转换为一个包含 DataFrame 作为表格的 HTML 页面。

# 从 Parquet 读取和写入数据

Parquet 文件格式为 `pandas` DataFrame 提供了列式序列化。它在存储和性能方面高效地读取和写入 DataFrame，并且能够在分布式系统之间共享数据而不丢失信息。Parquet 文件格式不支持重复列和数值列。

`pandas` 使用两个引擎来读取和写入 Parquet 文件：`pyarrow` 和 `fastparquet` 引擎。`pandas` 的默认 Parquet 引擎是 `pyarrow`；如果 `pyarrow` 不可用，则使用 `fastparquet`。在我们的示例中，我们使用的是 `pyarrow`。让我们使用 `pip` 安装 `pyarrow`：

```py
pip install pyarrow
```

你还可以通过在 `pip` 关键词前加上 `!` 来在 Jupyter Notebook 中安装 `pyarrow` 引擎。以下是一个示例：

```py
!pip install pyarrow
```

让我们使用 `pyarrow` 引擎写入一个文件：

```py
# Write to a parquet file.
df.to_parquet('employee.parquet', engine='pyarrow')
```

在前面的代码示例中，我们使用`to_parquet()` Parquet 文件和`pyarrow`引擎进行写入：

```py
# Read parquet file
employee_df = pd.read_parquet('employee.parquet', engine='pyarrow')

# display initial 5 records
employee_df.head()
```

这将产生以下输出：

![](img/b44e908a-ac9e-43fc-9a94-86fbb6d1abb0.png)

在前面的代码示例中，我们使用`read_parquet()`和`pyarrow`引擎读取了 Parquet 文件。`read_parquet()`有助于读取 Parquet 文件格式。接下来，我们将在下一部分了解如何读取和写入`pickle`文件中的数据。

# 从 Pickle pandas 对象中读取和写入数据

在数据准备步骤中，我们将使用各种数据结构，如字典、列表、数组或 DataFrame。有时，我们可能希望将它们保存以备将来参考，或将它们发送给其他人。在这种情况下，`pickle`对象就派上用场了。`pickle`将对象序列化以保存它们，并可以随时加载。`pandas`提供了两个函数：`read_pickle()`用于加载`pandas`对象，`to_pickle()`用于保存 Python 对象：

```py
# import pandas
import pandas as pd

# Read CSV file
df=pd.read_csv('demo.csv', sep=',' , header=None)

# Save DataFrame object in pickle file
df.to_pickle('demo_obj.pkl')
```

在前面的代码中，我们使用`read_csv()`方法读取了`demo.csv`文件，并设置了`sep`和`header`参数。在这里，我们将`sep`设置为逗号，将`header`设置为`None`。最后，我们使用`to_pickle()`方法将数据集写入`pickle`对象。接下来，让我们看看如何使用`pandas`库读取`pickle`对象：

```py
#Read DataFrame object from pickle file
pickle_obj=pd.read_pickle('demo_obj.pkl')

# display initial 5 records
pickle_obj.head()
```

这将产生以下输出：

![](img/e6244c42-b825-4193-886c-eee9ae5d3c8e.png)

在前面的代码中，我们使用`read_pickle()`方法读取了`pickle`对象。

# 使用 sqllite3 轻量级访问

SQLite 是一个开源数据库引擎。它提供了多种特性，如更快的执行速度、轻量级处理、无服务器架构、ACID 兼容性、更少的管理工作、高稳定性和可靠的事务。它是移动设备和计算机世界中最流行、最广泛部署的数据库。它也被称为嵌入式关系型数据库，因为它作为应用程序的一部分运行。SQLite 是一个更轻量的数据库，功能上没有完整的功能。它主要用于存储和处理小型数据，比如移动应用和桌面应用。SQLite 的主要优点是易于使用、高效、轻便，并且可以嵌入到应用程序中。

我们可以使用`sqlite3`模块在 Python 中读取和写入数据。我们不需要下载和安装`sqlite3`，因为它已经包含在所有标准 Python 发行版中。使用`sqlite3`，我们可以将数据库存储在文件中或保存在内存中。`sqlite3`允许我们使用 SQL 编写任何数据库，而不需要第三方应用程序服务器。让我们通过以下示例了解数据库连接：

```py
# Import sqlite3
import sqlite3

# Create connection. This will create the connection with employee database. If the database does not exist it will create the database
conn = sqlite3.connect('employee.db')

# Create cursor
cur = conn.cursor()

# Execute SQL query and create the database table
cur.execute("create table emp(eid int,salary int)")

# Execute SQL query and Write the data into database
cur.execute("insert into emp values(105, 57000)")

# commit the transaction
con.commit()

# Execute SQL query and Read the data from the database
cur.execute('select * from emp')

# Fetch records
print(cur.fetchall())

# Close the Database connection
conn.close()

Output: [(105, 57000)]
```

这里，我们使用`sqlite3`模块。首先，我们导入该模块并使用`connect()`方法创建一个连接。`connect()`方法将接收数据库名称和路径；如果数据库不存在，它将使用给定的名称和路径创建该数据库。一旦与数据库建立连接，您需要创建`Cursor`对象，并使用`execute()`方法执行 SQL 查询。我们可以在`execute()`方法中创建表，例如在员工数据库中创建的`emp`表。类似地，我们可以使用`execute()`方法和`insert`查询参数写入数据，并使用`commit()`方法将数据提交到数据库。也可以通过`execute()`方法传递`select`查询作为参数来提取数据，使用`fetchall()`和`fetchone()`方法进行提取。`fetchone()`提取一条记录，`fetchall()`提取数据库表中的所有记录。

# 从 MySQL 读取和写入数据

MySQL 是一个快速、开源、易于使用的关系型数据库，适用于小型和大型商业应用。它与数据库驱动的 Web 开发应用非常兼容。Python 中有很多方式可以访问 MySQL 中的数据。像 MySQLdb、`mysqlconnector`和`pymysql`等连接器可用于 MySQL 数据库连接。为了实现这种连接，您需要安装 MySQL 关系型数据库和`mysql-python`连接器。MySQL 的安装详情可以在其官网上找到：[`www.mysql.com/downloads/`](https://www.mysql.com/downloads/)。

您可以使用`pymysql`连接器作为客户端库，可以通过`pip`安装：

```py
pip install pymysql
```

我们可以通过以下步骤建立连接：

1.  导入库。

1.  创建数据库连接。

1.  创建游标对象。

1.  执行 SQL 查询。

1.  获取记录或响应以更新或插入记录。

1.  关闭连接。

在我们的示例中，我们尝试使用`mysqlconnecter`和`pymysql`进行数据库连接。在运行数据库连接脚本之前，第一步是设计并创建一个数据库，然后在 MySQL 中创建一个表。

让我们使用以下查询来创建一个数据库：

```py
>> create database employee
```

将数据库更改为员工数据库：

```py
>> use employee
```

在数据库中创建表：

```py
>> create table emp(eid int, salary int);
```

现在我们可以插入和获取 MySQL 表中的记录。让我们看一下以下示例来理解数据库连接：

```py
# import pymysql connector module
import pymysql

# Create a connection object using connect() method 
connection = pymysql.connect(host='localhost', # IP address of the MySQL database server
                             user='root', # user name
                             password='root',# password
                             db='emp', # database name
                             charset='utf8mb4', # character set
                             cursorclass=pymysql.cursors.DictCursor) # cursor type

try:
    with connection.cursor() as cur:
        # Inject a record in database
        sql_query = "INSERT INTO `emp` (`eid`, `salary`) VALUES (%s, %s)"
        cur.execute(sql_query, (104,43000))

    # Commit the record insertion explicitly.
    connection.commit()

    with connection.cursor() as cur:
        # Read records from employee table
        sql_query = "SELECT * FROM `emp`"
        cur.execute(sql_query )
        table_data = cur.fetchall()
        print(table_data)
except:
    print("Exception Occurred")
finally:
    connection.close()
```

这里，我们使用`pymysql`模块。首先，我们导入该模块并创建一个连接。`connect()`函数将接收主机地址，在我们的案例中是`localhost`（我们也可以使用远程数据库的 IP 地址）、用户名、密码、数据库名称、字符集和游标类。

在建立连接之后，我们可以读取或写入数据。在我们的例子中，我们使用 `insert` SQL 查询写入数据，并使用 `select` 查询检索数据。在插入查询中，我们执行查询并传递要插入数据库的参数，然后使用 `commit()` 方法将结果提交到数据库。当我们使用选择查询读取记录时，将得到一些记录。我们可以使用 `fetchone()` 和 `fetchall()` 函数提取这些记录。`fetchone()` 方法仅提取单条记录，而 `fetchall()` 方法则提取数据库表中的多条记录。

还有一件事；在这里，所有的读写操作都在 `try` 块中执行，连接则在最终块中关闭。我们还可以尝试另一个模块 `mysql.connector` 来实现 MySQL 和 Python 的连接。它可以通过 `pip` 安装：

```py
pip install mysql-connector-python
```

让我们看一个例子来理解数据库连接：

```py
# Import the required connector
import mysql.connector
import pandas as pd

# Establish a database connection to mysql
connection=mysql.connector.connect(user='root',password='root',host='localhost',database='emp')

# Create a cursor
cur=connection.cursor()

# Running sql query
cur.execute("select * from emp")

# Fetch all the records and print it one by one
records=cur.fetchall()
for i in records:
    print(i)

# Create a DataFrame from fetched records.
df = pd.DataFrame(records)

# Assign column names to DataFrame
df.columns = [i[0] for i in cur.description]

# close the connection
connection.close()
```

在前面的代码示例中，我们使用 `mysql.connector` 模块连接 Python 和 MySQL 数据库，提取数据的方式和步骤与使用 `pymysql` 模块相同。我们还通过将提取的记录传递到 `pandas` DataFrame 对象中，并从游标描述中分配列名来写入数据。

## 将整个 DataFrame 插入数据库

在前面的程序中，使用 `insert` 命令插入单个记录。如果我们想要插入多个记录，则需要运行一个循环，将多个记录插入数据库。我们还可以使用 `to_sql()` 函数在一行代码中插入多个记录：

```py
# Import the sqlalchemy engine
from sqlalchemy import create_engine

# Instantiate engine object
en = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                .format(user="root", 
                        pw="root", 
                        db="emp"))

# Insert the whole dataframe into the database
df.to_sql('emp', con=en, if_exists='append',chunksize=1000, index= False)
```

在前面的代码示例中，我们将为数据库连接创建一个引擎，包含用户名、密码和数据库参数。`to_sql()` 函数将多个记录从 DataFrame 写入 SQL 数据库。它需要表名、`con` 参数（用于连接引擎对象）、`if_exists` 参数（检查数据是追加到新表还是替换为新表），以及 `chunksize`（用于批量写入数据）。

# 从 MongoDB 读取和写入数据

MongoDB 是一个面向文档的非关系型（NoSQL）数据库。它使用类似 JSON 的符号，**BSON**（**二进制对象符号**）来存储数据。MongoDB 提供以下功能：

+   它是一个免费的、开源的、跨平台的数据库软件。

+   它易于学习，能够构建更快的应用程序，支持灵活的模式，处理多种数据类型，并且具备在分布式环境中扩展的能力。

+   它基于文档的概念工作。

+   它包含数据库、集合、文档、字段和主键。

我们可以通过`pymongo`连接器在 Python 中从 MongoDB 读取和写入数据。为了实现这个连接，我们需要安装 MongoDB 和`pymongo`连接器。你可以从其官方网站下载 MongoDB：[`www.mongodb.com/download-center/community`](https://www.mongodb.com/download-center/community)。PyMongo 是一个纯 Python 的 MongoDB 客户端库，可以通过`pip`安装：

```py
pip install pymongo
```

让我们尝试使用`pymongo`进行数据库连接：

```py
# Import pymongo
import pymongo

# Create mongo client
client = pymongo.MongoClient()

# Get database
db = client.employee

# Get the collection from database
collection = db.emp

# Write the data using insert_one() method
employee_salary = {"eid":114, "salary":25000}
collection.insert_one(employee_salary)

# Create a dataframe with fetched data
data = pd.DataFrame(list(collection.find()))
```

在这里，我们通过创建 Mongo 客户端、插入数据、提取集合详情并将其分配给 DataFrame，尝试从 MongoDB 的数据库集合中提取数据。接下来，我们将展示如何在下一节中使用列式数据库 Cassandra 建立数据库连接。

# 从 Cassandra 读取和写入数据

Cassandra 是一个可扩展、高可用、耐久和容错的列式数据库，具有较低的管理开销、更快的读写速度，并且能够提供强大的弹性。它容易学习和配置，能够为复杂问题提供解决方案，并且支持跨多个数据中心的复制。许多大型公司，例如苹果、eBay 和 Netflix，都在使用 Cassandra。

我们可以通过`cassandra-driver`连接器在 Python 中从 Cassandra 读取和写入数据。为了实现这个连接，我们需要安装 Cassandra 和`cassandra-driver`连接器。你可以从其官方网站下载 Cassandra：[`cassandra.apache.org/download/`](http://cassandra.apache.org/download/)。`cassandra-driver`是一个纯 Python 的 Cassandra 客户端库，可以通过`pip`安装：

```py
pip install cassandra-driver
```

让我们尝试使用`cassandra-driver`进行数据库连接：

```py
# Import the cluster
from cassandra.cluster import Cluster

# Creating a cluster object
cluster = Cluster()

# Create connections by calling Cluster.connect():
conn = cluster.connect()

# Execute the insert query
conn.execute("""INSERT INTO employee.emp_details (eid, ename, age) VALUES (%(eid)s, %(ename)s, %(age)s)""", {'eid':101, 'ename': "Steve Smith", 'age': 42})

# Execute the select query
rows = conn.execute('SELECT * FROM employee.emp_details')

# Print the results
for emp_row in rows:
    print(emp_row.eid, emp_row.ename, emp_row.age)

# Create a dataframe with fetched data
data = pd.DataFrame(rows)
```

在这里，我们通过创建一个集群对象、使用`connect()`方法建立连接、执行插入操作以及选择查询数据，尝试从 Cassandra 数据库中提取数据。运行查询后，我们打印结果并将提取的记录分配给`pandas` DataFrame。接下来，让我们进入另一个 NoSQL 数据库：Redis。

# 从 Redis 读取和写入数据

Redis 是一个开源的 NoSQL 数据库。它是一个键值数据库，存储在内存中，速度极快，并且高可用。它还可以用作缓存或消息代理。内存存储意味着它使用 RAM 来存储数据，并通过虚拟内存处理更大的数据。Redis 提供缓存服务或永久存储。Redis 支持多种数据结构，如字符串、集合、列表、位图、地理空间索引和超日志。Redis 能够处理地理空间、流式数据和时间序列数据。它还可以与云服务如 AWS 和 Google Cloud 一起使用。

我们可以通过 Redis 连接器在 Python 中从 Redis 读取和写入数据。为了实现这个连接，我们需要安装 Redis 和 Redis 连接器。你可以通过以下链接下载 Redis：[`github.com/rgl/redis/downloads`](https://github.com/rgl/redis/downloads)。Redis 是一个纯 Python 的 Redis 客户端库，可以通过`pip`安装：

```py
pip install redis
```

让我们尝试使用 Redis 进行数据库连接：

```py
# Import module
import redis

# Create connection
r = redis.Redis(host='localhost', port=6379, db=0)

# Setting key-value pair
r.set('eid', '101')

# Get value for given key
value=r.get('eid')

# Print the value
print(value)
```

这里，我们尝试从 Redis 键值数据库中提取数据。首先，我们与数据库建立了连接。我们使用 `set()` 方法将键值对设置到 Redis 数据库中，并且通过给定的键参数使用 `get()` 方法提取值。

最后，进入本章的最后一个主题，即用于**对象关系映射**（**ORM**）的 PonyORM。

# PonyORM

PonyORM 是一个强大的 ORM 包，使用纯 Python 编写。它快速、易于使用，并且能以最小的努力执行操作。它提供自动查询优化和图形界面数据库模式编辑器。它还支持自动事务管理、自动缓存和复合键。PonyORM 使用 Python 生成器表达式，这些表达式会被翻译成 SQL。我们可以通过 `pip` 安装它：

```py
$ pip install pony
```

让我们来看一个使用 `pony` 的 ORM 示例：

```py
# Import pony module
from pony.orm import *

# Create database
db = Database()

# Define entities
class Emp(db.Entity):
    eid = PrimaryKey(int,auto=True)
    salary = Required(int)

# Check entity definition
show(Emp)

# Bind entities to MySQL database
db.bind('mysql', host='localhost', user='root', passwd='12345', db='employee')

# Generate required mappings for entities
db.generate_mapping(create_tables=True)

# turn on the debug mode
sql_debug(True)

# Select the records from Emp entities or emp table
select(e for e in Emp)[:]

# Show the values of all the attribute
select(e for e in Emp)[:].show()

Output:
eid|salary
---+------
104|43000 
104|43000 
```

在前面的代码示例中，我们正在执行 ORM。首先，我们创建了一个 `Database` 对象，并使用 `Emp` 类定义了实体。接着，我们通过 `db.bind()` 方法将实体绑定到数据库。我们可以将其绑定到四个数据库：`sqlite`、`mysql`、`postgresql` 和 `oracle`。在我们的示例中，我们使用 MySQL，并传递其凭证信息，如用户名、密码和数据库名称。我们可以使用 `generate_mapping()` 来执行实体与数据的映射。`create_tables=True` 参数在表格不存在时会创建表。`sql_debug(True)` 会启用调试模式。`select()` 函数将 Python 生成器翻译成 SQL 查询，并返回一个 `pony` 对象。这个 `pony` 对象将通过切片操作符（`[:]`）转换为实体列表，`show()` 函数将以表格形式显示所有记录。

# 总结

在这一章中，我们学习了如何以不同格式检索、处理和存储数据。我们查看了如何从各种文件格式和来源读取和写入数据，如 CSV、Excel、JSON、HDF5、HTML、`pickle`、表格和 Parquet 文件。我们还学习了如何从各种关系型和 NoSQL 数据库中读取和写入数据，如 SQLite3、MySQL、MongoDB、Cassandra 和 Redis。

下一章，第七章，*清理混乱数据*，讲述了数据预处理和特征工程在 Python 中的关键主题。本章从探索性数据分析开始，接着介绍了数据过滤、处理缺失值和异常值。清理之后，将重点讨论数据转换，如编码、缩放和拆分。
