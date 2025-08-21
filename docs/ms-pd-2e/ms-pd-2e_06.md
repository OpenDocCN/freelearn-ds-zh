# 第四章：不同数据格式的 I/O 操作与 pandas

数据科学家需要处理来自各种来源的数据，因此数据格式也各异。最常见的格式是无处不在的电子表格、Excel 表格、`CSV` 和文本文件。但也有许多其他格式，如 `URL`、`API`、`JSON`、`XML`、`HDF`、`Feather` 等，具体取决于数据访问的方式。本章我们将涉及以下几个主题：

+   数据源和 pandas 方法

+   CSV 和 TXT

+   URL 和 S3

+   JSON

+   读取 HDF 格式

开始吧！

# 数据源和 pandas 方法

数据科学项目的数据源可以分为以下几类：

+   **数据库**：大多数 CRM、ERP 及其他业务操作工具都将数据存储在数据库中。根据数据的体积、速度和种类，可能是传统的数据库或 NoSQL 数据库。为了连接到大多数流行的数据库，我们需要 Python 的 `JDBC/ODBC` 驱动程序。幸运的是，所有流行数据库的驱动程序都有提供。使用这些数据库中的数据需要通过 Python 连接到数据库，通过 Python 查询数据，然后使用 pandas 进行数据操作。我们将在本章后面查看如何实现这一过程的示例。

+   **Web 服务**：许多业务操作工具，尤其是 **软件即服务**（**SaaS**）工具，通过 **应用程序编程接口**（**API**）而非数据库提供其数据访问。这减少了永久托管数据库的基础设施成本。相反，数据作为服务按需提供。可以通过 Python 发起 `API` 调用，返回格式为 `JSON` 或 `XML` 的数据包。然后解析这些数据并使用 pandas 进行进一步处理。

+   **数据文件**：许多用于原型数据科学模型的数据以数据文件的形式存在。一个例子是来自物联网传感器的数据——这些传感器的数据通常存储在平面文件中，如 `.txt` 文件或 `.csv` 文件。另一个数据文件来源是从数据库中提取并存储在这些文件中的示例数据。许多数据科学和机器学习算法的输出也存储在此类文件中，如 CSV、Excel 和 `.txt` 文件。另一个例子是深度学习神经网络模型的训练权重矩阵，可以存储为 HDF 文件。

+   **网页和文档抓取**：另一个数据来源是网页上存在的表格和文本。通过使用如 BeautifulSoup 和 Scrapy 等 Python 包，从这些页面中提取数据，并将其放入数据文件或数据库中以供进一步使用。存在于其他非数据格式文件中的表格和数据，如 PDF 或 Docs，也是一个重要的数据来源。这些数据通过如 Tesseract 和 Tabula-py 等 Python 包进行提取。

在本章中，我们将研究如何使用 pandas 和附属库读取和写入这些格式/源的数据。我们还将讨论这些格式的简要介绍，它们的用途以及可以对它们执行的各种操作。

以下是 Python 中用于读取和写入某些数据格式的方法总结，这些格式将在本章中讨论：

![](img/8abf637c-9d15-45c5-b8e7-fdef295dd9d5.png)

pandas 中用于不同数据文件格式的读取和写入方法及其来源

这些部分标题意味着我们正在处理该文件类型的 I/O 操作。

# CSV 和 TXT

CSV 代表逗号分隔值，意味着逗号是这些文件的默认分隔符。然而，它们也接受其他分隔符。

CSV 由列和行组成，单元格值以表格形式排列。它们可以带有或不带有列名和行索引。CSV 文件存在的主要原因包括手动收集的数据、从数据库中提取并下载的数据、从工具或网站直接下载的数据、网页抓取的数据以及运行数据科学算法的结果。

# 读取 CSV 和 TXT 文件

`read_csv`是 pandas 中读取 CSV 文件的首选方法。它也可以用来读取`txt`文件。使用`read_csv`的语法如下：

```py
pd.read_csv(filepath, sep=', ', dtype=None, header=None, names=None, skiprows=None, index_col=None, skip_blank_lines=TRUE, na_filter=TRUE)
```

`read_csv`方法的参数如下：

+   `filepath`: 字符串或带有或不带有文件路径的文件名。

+   `dtype`: 可以作为字典传递，字典包含名称和类型的键值对。指定列名的数据类型。通常，pandas 会根据前几行来猜测列的类型。

+   `header`: True/False。指定数据的第一行是否为标题。

+   `names`: 列表。指定数据集所有列的列名。

+   `skiprows`: 列表。通过指定行索引跳过某些数据行。

+   `index_col`: Series/列表。指定可以作为行号/标识符的列。

+   `skip_blank_lines`: True/False。指定是否跳过空行。

+   `na_filter`: True/False。指定是否过滤 NA 值。

+   `usecols`: 列表。返回传递列表中的列的数据子集。

`read_csv`方法返回一个 DataFrame。以下是使用`read_csv`方法读取文件的一些示例。

# 读取 CSV 文件

我们可以使用以下代码读取 CSV 文件：

```py
import pandas as pd
import os
os.chdir(' ')
data=pd.read_csv('Hospital Cost.csv')
```

# 为数据集指定列名

以下代码将指定数据集的列名：

```py
column_names=pd.read_csv('Customer Churn Columns.csv')
column_names_list=column_names['Column Names'].tolist()
data=pd.read_csv('Customer Churn Model.txt',header=None,names=column_names_list)
```

请注意，列名是从文件中读取的，然后转换成列表，并传递给`read_csv`中的 names 参数。

# 从数据字符串中读取

下面是我们如何使用`read_csv`从字符串列表创建一个 DataFrame：

```py
from io import StringIO
data = 'col1,col2,col3\na,b,1\na,b,2\nc,d,3\nc,e,4\ng,f,5\ne,z,6'
pd.read_csv(StringIO(data))
```

# 跳过某些行

我们还可以跳过某些行。假设我们只想要索引是 3 的倍数的行：

```py
from io import StringIO
data = 'col1,col2,col3\na,b,1\na,b,2\nc,d,3\nc,e,4\ng,f,5\ne,z,6'
pd.read_csv(StringIO(data),skiprows=lambda x: x % 3 != 0)
```

我们得到以下输出：

![](img/7e5b5fa6-3e3b-4995-a8db-66caaf0727eb.png)

演示如何在`read_csv`中使用`skiprows`参数。右侧面板显示了通过`skiprows`筛选的数据（仅保留行号是 3 的倍数的行）

左侧图示显示了结果 DataFrame，没有跳过任何行，而右侧图示显示了过滤掉行索引不是 3 的倍数的同一 DataFrame。注意，这种方法是根据实际索引（从 1 开始的第 3 行和第 6 行）来过滤行，而不是根据 Python 索引（从 0 开始）。

# 行索引

如果文件的列数比列名的数量多一列数据，第一列将作为 DataFrame 的行名：

```py
data = 'a,b,c\n4,apple,bat,5.7\n8,orange,cow,10'
pd.read_csv(StringIO(data), index_col=0)
```

我们得到以下输出：

![](img/8e787128-6e43-4df6-a46d-728f1e129fb9.png)

具有值但没有对应列名的列被用作行索引。

# 阅读文本文件

`read_csv`也可以帮助读取文本文件。通常，数据存储在具有不同分隔符的`.txt`文件中。`sep`参数可用于指定特定文件的分隔符，如以下代码所示：

```py
data=pd.read_csv('Tab Customer Churn Model.txt',sep='/t')
```

上述文件的分隔符为`Tab`，通过`sep`参数指定。

# 阅读时的子集化

在读取时，可以使用`usecols`参数仅选择部分列进行子集化和加载：

```py
data=pd.read_csv('Tab Customer Churn Model.txt',sep='/t',usecols=[1,3,5])
data=pd.read_csv('Tab Customer Churn Model.txt',sep='/t',usecols=['VMail Plan','Area Code'])
```

数字列表以及带有列名的显式列表都可以使用。数字索引遵循 Python 索引规则，即从 0 开始。

# 将千位格式的数字作为数字读取

如果数据集中包含一个数字列，该列的千位数字以逗号或其他分隔符格式化，那么该列的默认数据类型将是字符串或对象。问题是它实际上是一个数字字段，需要作为数字字段进行读取，以便进一步使用：

```py
pd.read_csv('tmp.txt',sep='|')
```

我们得到以下输出：

![](img/a1ae239f-3e96-45a6-85a3-47f6f43ba54d.png)

带有千位格式数字的列数据

```py
data.level.dtype returns dtype('O')
```

为了克服这个问题，可以在读取时使用`thousands`参数：

```py
pd.read_csv('tmp.txt',sep='|',thousands=',')
data.level.dtype now returns dtype('int64')
```

# 索引和多重索引

`index_col`可用于指定某一列作为行索引。可以传递列的列表作为索引，这将导致多重索引。我们来看一个例子：

```py
pd.read_csv('mindex.txt')
pd.read_csv('mindex.txt',index_col=[0,1])
```

请看下面的截图：

![](img/c22b0911-c5a3-40af-9b87-92c6b3161b6d.png)

单一索引（左）和多重索引（右）在同一数据上的应用

这种多重索引使得基于某一索引或两个索引进行子集化变得容易：

```py
data.loc[1977]
data.loc[(1977,'A')]
```

我们得到以下输出：

![](img/4752a1fa-6e39-4f44-80b3-529f1880ba9e.png)

使用单一索引（左）和两个索引（右）对子集化多重索引数据

# 分块读取大文件

一次性将大文件读取到内存中可能会占用整个计算机的内存，并可能导致错误。在这种情况下，按块划分数据变得尤为重要。这些数据块可以按顺序读取并处理。通过使用`chunksize`参数，在`read_csv`中实现了这一点。

结果的块可以通过 `for` 循环进行迭代。在以下代码中，我们正在打印块的形状：

```py
for chunks in pd.read_csv('Chunk.txt',chunksize=500):
    print(chunks.shape)
```

然后，这些块可以使用 `concat` 方法连接在一起：

```py
data=pd.read_csv('Chunk.txt',chunksize=500)
data=pd.concat(data,ignore_index=True)
print(data.shape)
```

# 处理列数据中的分隔符字符

有时，列分隔符字符作为数据的一部分出现在某一列中。这会导致数据解析错误，因为它会将本应作为一个读取的列分割成两个。为了避免这种情况，应该在指定的列数据周围使用引号字符。这个引号字符强制 `read_csv` 忽略数据中引号字符内的分隔符，并且不将其分成两部分。

可以通过 `read_csv` 的 `quotechar` 参数指定引号字符。例如，考虑以下数据集。在这里，使用空格作为分隔符，并使用双引号作为分组元素：

![](img/c343c737-1b11-46a8-9dd4-f8dbb4d2f424.png)

使用 `quotechar` 关键字 1——输入数据集

要解析此数据，我们可以使用以下代码：

```py
d1 = 
pd.read_csv('t1.txt',index_col=0, delim_whitespace=True,quotechar="\"")
d1.head()
```

我们将得到以下输出：

![](img/5e92e0f9-5111-4605-bc3a-9110692a106d.png)

使用 `quotechar` 关键字 2——输出数据集

# 写入 CSV

DataFrame 是内存中的对象。通常，DataFrame 需要保存为物理文件以便后续使用。在这种情况下，DataFrame 可以写入为 `CSV` 或 `TXT` 文件。

让我们使用随机数创建一个合成的 DataFrame：

```py
import numpy as np
import pandas as pd
a=['Male','Female']
b=['Rich','Poor','Middle Class']
gender=[]
seb=[]

for i in range(1,101):
    gender.append(np.random.choice(a))
    seb.append(np.random.choice(b))
    height=30*np.random.randn(100)+155
    weight=20*np.random.randn(100)+60
    age=10*np.random.randn(100)+35
    income=1500*np.random.randn(100)+15000

df=pd.DataFrame({'Gender':gender,'Height':height,'Weight':weight,'Age':age,'Income':income,'Socio-Eco':seb})
```

这可以通过 `to_csv` 方法写入 `.csv` 或 `.txt` 文件，如以下代码所示：

```py
df.to_csv('data.csv')
df.to_csv('data.txt')
```

这些文件将被写入当前工作目录。

写入文件时可以提供自定义的分隔符：

```py
df.to_csv('data.csv',sep='|')
df.to_csv('data.txt',sep='/')
```

还有许多其他有用的选项，如下所示：

+   `index`: True/False。指示是否需要行索引。

+   `index_label`: 字符串/列名。用作行索引的列。

+   `header`: True/False。指定是否写入列名。

+   `na_rep`: 字符串。缺失值的字符串表示。

# Excel

Excel 文件与 `CSV` 文件相似，但不同之处在于它们可以包含多个工作表、格式化的数据和表格、图表以及公式。在许多情况下，需要从 Excel 文件中读取数据。

`xlrd` 是处理 Excel 工作表时首选的包。`xlrd` 包的一些主要功能总结在下表中：

| **代码片段** | **目标达成** |
| --- | --- |
| `import xlrd` | 导入 xlrd 库 |
| `book=xlrd.open_workbook('SRS Career.xlsx')` | 读取 Excel 工作簿 |
| `n=book.nsheets` | 查找工作簿中的工作表数量 |
| `book.sheet_names()` | 查找工作簿中工作表的名称 |
| `last_sheet=book.sheet_by_index(n-1)` | 通过工作表索引读取工作表 |
| `last_sheet.row_values(0)` | 获取工作表的第一行 |
| `last_sheet.cell(0,0)` | 获取工作表的第一个单元格 |
| `last_sheet.row_slice(rowx=0,start_colx=1,end_colx=5)` | 获取第一行的第 1 列到第 5 列 |

# URL 和 S3

有时，数据可以直接通过 URL 获取。在这种情况下，可以直接使用 `read_csv` 从这些 URL 中读取数据：

```py
pd.read_csv('http://bit.ly/2cLzoxH').head()
```

或者，为了通过 URL 获取数据，我们可以使用一些尚未使用的 Python 包，如 `.csv` 和 `.urllib`。只需知道 `.csv` 提供了一系列处理 `.csv` 文件的方法，而 `urllib` 用于访问和获取 URL 中的信息即可。以下是实现方法：

```py
import csv 
import urllib2 

url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' 
response=urllib2.urlopen(url) 
cr=csv.reader(response) 

for rows in cr: 
   print rows 

```

`AWS S3` 是一个流行的文件共享和存储库，许多企业将其业务运营数据以文件形式存储在 S3 上，这些数据需要直接读取和处理，或者被移入数据库。Python 允许我们直接从 S3 读取文件，如下所示的代码所示：

Python 3.4 及以上版本在使用 pandas 之外，还需要使用 `s3fs` 包来直接从 S3 读取文件。AWS 配置文件需要放置在当前工作目录中。需要传入存储桶名称以及路径和文件名来进行读取：

```py
import os 
import pandas as pd 
from s3fs.core import S3FileSystem 

os.environ['AWS_CONFIG_FILE'] = 'aws_config.ini' 

s3 = S3FileSystem(anon=False) 
key = 'path\to\your-csv.csv' 
bucket = 'your-bucket-name' 

df = pd.read_csv(s3.open('{}/{}'.format(bucket, key), 
                         mode='rb') 
                 ) 
```

可以将 DataFrame 写入 CSV 文件并直接保存在 S3 中，如下所示：

```py
 import s3fs 

bytes_to_write = df.to_csv(None).encode() 
fs = s3fs.S3FileSystem(key=key, secret=secret) 
with fs.open('s3://bucket/path/to/file.csv', 'wb') as f: 
    f.write(bytes_to_write) 

```

# HTML

HTML 是一种流行的文件格式，用于创建和包装网页元素和页面。有时，表格数据会存储在文件中。在这种情况下，可以直接使用 `read_html` 方法读取这些数据。此功能解析 HTML 文件中的表格元素，并将表格读取为 DataFrame：

```py
pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html') 
```

你可以通过以下代码查找包含特定匹配词的所有表格元素：

```py
 match = 'Malta National Bank' 
df_list = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html', match=match) 
```

可以将 DataFrame 转换为 HTML 表格元素，以便将其嵌入 HTML 文件中，如下所示：

```py
data=pd.read_csv('http://bit.ly/2cLzoxH')
print(data.to_html())
```

我们得到如下输出：

![](img/99e747e1-7272-4977-8aad-2b2550d77cb7.png)

从 DataFrame 创建的 HTML 表格元素

可以选择性地筛选和转换某些列为 HTML，如下所示：

```py
 print(data.to_html(columns=['country','year'])) 
```

# 写入 HTML 文件

HTML 文件可以保存为一个物理文件，如下所示：

```py
data=pd.read_csv('http://bit.ly/2cLzoxH') 
print(data.to_html('test.html')) 
```

看一下以下截图：

![](img/6f0d904f-7480-42a1-a071-7acbf2a42ad1.png)

使用一个索引（左）和两个索引（右）对子集化多重索引数据

默认情况下，行和列的名称是粗体的。可以通过以下代码进行更改：

```py
 data.to_html('test.html',bold_rows=False) 
```

# JSON

`JSON` 是一种流行的字典型、基于键值对的数据结构，适用于从 SaaS 工具暴露数据为 API。`address`、`postalCode`、`state`、`streetAddress`、`age`、`firstName`、`lastName` 和 `phoneNumber` 是键，它们的值显示在它们的右侧。`JSON` 文件也可以是嵌套的（键的值本身也是 JSON）。在这里，`address` 有嵌套值：

![](img/b14ed272-db57-4fba-9a57-9c9cea728544.png)

JSON 数据示例（字典；键值对）

DataFrame 可以使用 `to_json` 转换为 JSON 格式：

```py
import numpy as np 
pd.DataFrame(np.random.randn(5, 2), columns=list('AB')).to_json() "
```

看一下以下截图：

![](img/502b72f1-18c4-4307-ab86-8fd00f5b27d8.png)

将 DataFrame 转换为 JSON 格式

在将 DataFrame 转换为 JSON 文件时，可以设置方向。

如果我们想保持列名作为主要索引，行索引作为次要索引，那么我们可以选择将方向设置为 `columns`：

```py
dfjo = pd.DataFrame(dict(A=range(1, 4), B=range(4, 7), C=range(7, 10)),columns=list('ABC'), index=list('xyz')) 
dfjo.to_json(orient="columns")
```

我们得到以下输出：

![](img/016ceada-6767-443e-8442-efbebb525bb9.png)

将一个 DataFrame 转换为具有列方向的 JSON

如果我们想保持行索引作为主要索引，列名作为次要索引，那么我们可以选择将方向设置为 `index`：

```py
dfjo = pd.DataFrame(dict(A=range(1, 4), B=range(4, 7), C=range(7, 10)),columns=list('ABC'), index=list('xyz')) 
dfjo.to_json(orient="index")
```

我们得到以下输出：

![](img/9f1817dc-3dee-4536-8d37-613c445ee872.png)

将一个 DataFrame 转换为具有索引方向的 JSON

另一种选择是将一个 DataFrame 转换为一个 JSON 数组。这在将数据传递给可视化库时非常有用，如下所示：

```py
d3.js. dfjo = pd.DataFrame(dict(A=range(1, 4), B=range(4, 7), C=range(7, 10)),columns=list('ABC'), index=list('xyz')) dfjo.to_json(orient="records")
```

我们得到以下输出：

![](img/3f25c39c-f779-4804-8e0e-e03ee72f54ea.png)

将一个 DataFrame 转换为具有记录方向的 JSON

我们还可以将纯值作为值的列表包含，而不需要任何行或列索引：

```py
dfjo = pd.DataFrame(dict(A=range(1, 4), B=range(4, 7), C=range(7, 10)),columns=list('ABC'), index=list('xyz')) dfjo.to_json(orient="values") 
```

我们得到以下输出：

![](img/b3ea3f67-3fe1-4968-bfd3-302fcc8198d0.png)

将一个 DataFrame 转换为具有值方向的 JSON

最后，我们还可以调整转换后的 JSON，以便分离行索引、列名和数据值：

```py
dfjo = pd.DataFrame(dict(A=range(1, 4), B=range(4, 7), C=range(7, 10)),columns=list('ABC'), index=list('xyz')) dfjo.to_json(orient="split")
```

我们得到以下输出：

![](img/69ac8ae1-0481-40fa-a39f-dcbc19b84268.png)

将一个 DataFrame 转换为具有拆分方向的 JSON

# 将 JSON 写入文件

JSON 可以像这样写入物理文件：

```py
import json with open('jsonex.txt','w') as outfile: json.dump(dfjo.to_json(orient="columns"), outfile)
```

# 读取 JSON

`json_loads` 用于读取包含 JSON 的物理文件：

```py
f=open('usagov_bitly.txt','r').readline() json.loads(f) 
```

我们得到以下输出：

![](img/04b51282-c183-492b-a784-6eeeec586db7.png)

JSON 列表中的第一个记录

文件可以通过 `open` 和 `readline` 方法一次读取一个 JSON：

```py
records=[] f=open('usagov_bitly.txt','r') for i in range(1000): fiterline=f.readline() d=json.loads(fiterline) records.append(d) f.close()
```

现在，`records` 包含一个 JSON 列表，我们可以从中提取出某个特定键的所有值。例如，在这里，我们提取出所有具有非零值的 `latlong`（`'ll'` 列）：

```py
latlong=[rec['ll'] for rec in records if 'll' in rec]
```

# 将 JSON 写入 DataFrame

一个 JSON 对象列表可以转换为一个 DataFrame（就像字典一样）。我们之前创建的 `records` 元素是一个 JSON 列表（我们可以通过使用 `records[0:3]` 或 `type(records)` 来检查）：

```py
df=pd.DataFrame(records) 
df.head() 
df['tz'].value_counts() 
```

在最后一行，我们尝试查找 `'tz'` 列中包含的不同时区的数量。

# 对 JSON 进行子集筛选

让我们来看看一个新的 JSON 文件：

```py
 with open('product.json') as json_string: 
    d=json.load(json_string) 
d 
```

我们得到以下输出：

![](img/2d6b3b39-b011-49be-a8ed-edfd9191c8e3.png)

加载一个具有多个嵌套层级的 JSON 文件

这是一个具有多个嵌套层级的 JSON。`hits` 键包含一个值为 JSON 的键值对，其中该 JSON 的键值为 `hits`，该 JSON 的值是一个包含另一个 JSON 的列表。

假设我们想从这个 JSON 中找出分数值：

```py
 d['hits']['hits'][0]['_score'] 
```

同样，图像 URL 可以如下找到：

```py
 d['hits']['hits'][0]['_source']['r']['wayfair']['image_url']
```

# 遍历 JSON 键

可以遍历 JSON 数据的键和值：

```py
for keys,values in d['hits']['hits'][0].items(): 
print(keys)
```

我们得到以下输出：

![](img/5f5e5349-afc0-410e-845e-10fe1314ea0e.png)

通过遍历键和值来打印加载的 JSON 的键

我们也可以将键和值一起打印出来。

```py
for keys,values in d['hits']['hits'][0].items():
 print(keys,values) 
```

我们接收到以下输出：

![](img/bb4bd0e2-2b88-4120-a6e4-1bc8d5ef1f16.png)

通过遍历键和值来打印加载的 JSON 的键和值

现在，我们将学习如何使用读写操作处理不同的文件格式。

# 读取 HDF 格式

**层次数据格式**（**HDF**）在处理大规模和复杂数据模型时非常高效。HDF 在数据存储中的多功能性和灵活性使其成为存储科学数据的热门格式。事实上，HDF 被 NASA 选为地球观测系统中的标准数据与信息系统。HDF5 是当前由 HDF 文件格式使用的技术套件，并取代了较旧的 HDF4。

以下是 HDF5 的一些独特功能：

+   HDF5 对文件大小和文件中的对象没有设定限制。

+   HDF5 可以对文件中的对象进行分组和链接，从而为数据中的复杂关系和依赖提供支持机制。

+   HDF5 还支持元数据。

+   HDF5 在支持多种预定义和用户定义的数据类型的同时，还能够在 HDF 文件中存储和共享数据类型描述。

为了提高数据传输过程的效率，HDF5 集成了标准（Posix）、并行和网络 I/O 文件驱动程序。还可以开发并将其他文件驱动程序与 HDF5 集成，以满足任何自定义的数据传输和存储需求。HDF5 通过压缩、可扩展性和分块等技术优化了数据存储。能够在数据传输过程中执行数据转换、修改数据类型以及选择数据子集，使得读写过程更加高效。

现在，让我们使用 pandas 读取一个 HDF 文件：

```py
pd.read_hdf('stat_df.h5','table')
```

我们接收到以下输出：

![](img/d925a312-cffd-4dd2-8e60-44d2d1fad99c.png)

read_hdf 的输出

在读取过程中，可以通过索引参数提取数据的一个子集：

```py
pd.read_hdf('stat_df.h5', 'table', where=['index>=2'])
```

我们接收到以下输出：

![](img/743debf2-53b8-4f10-b761-dada60221e07.png)

带索引的 read_hdf 输出

# 读取 Feather 文件

Feather 格式是一种二进制文件格式，用于存储数据，采用 Apache Arrow 作为内存中的列式数据结构。它由 RStudio 的首席科学家 Wes Mckinney 和 Hadley Wickham 开发，旨在为 Python 和 R 之间的数据共享基础设施提供支持。Feather 文件中数据的列式序列化使得读写操作更加高效，比存储按记录方式排列的 CSV 和 JSON 文件要快得多。

Feather 文件具有以下特点：

+   快速的输入/输出操作。

+   Feather 文件可以在除 R 或 Python 之外的其他语言中进行读写，例如 Julia 和 Scala。

+   它们与所有 pandas 数据类型兼容，例如 Datetime 和 Categorical。

Feather 当前支持以下数据类型：

+   所有数值数据类型

+   逻辑数据

+   时间戳

+   分类数据

+   UTF-8 编码的字符串

+   二进制

由于 feather 只是 Arrow 的简化版，它有一些相关的限制。以下是使用 feather 文件的一些限制：

+   不推荐用于长期数据存储，因为它们在版本之间的稳定性无法保证。

+   Feather 格式不支持任何非默认索引或多重索引。

+   Python 数据类型如 Period 不被支持。

+   不支持列名重复。

在 pandas 中读取 feather 文件的方法如下：

```py
pd.read_feather("sample.feather")
```

这将产生以下输出：

![](img/b96d3b4f-a50f-4a52-91c6-a4165aaf499b.png)

read_feather 输出

# 读取 parquet 文件

Apache Parquet 是另一种文件格式，利用列式压缩进行高效的读写操作。它设计上与 Hadoop 等大数据生态系统兼容，并能处理嵌套数据结构和稀疏列。虽然 parquet 和 feather 格式有相似的基础，但 parquet 的压缩方法比 feather 更优。压缩后的文件在 parquet 中比在 feather 中小。具有相似数据类型的列使用相同的编码进行压缩。不同编码方案的使用使得 parquet 的压缩效率更高。像 feather 一样，parquet 是一种二进制文件格式，可以与所有 pandas 数据类型良好配合，并在多个语言中得到支持。Parquet 可以用于长期数据存储。

以下是 parquet 文件格式的一些限制：

+   虽然 parquet 可以接受多级索引，但它要求索引级别名称必须是字符串格式。

+   Python 数据类型如 Period 不被支持。

+   不支持列名重复。

+   当分类对象序列化到 parquet 文件时，它们会作为对象数据类型进行反序列化。

pandas 中的 parquet 文件的序列化或反序列化可以通过 `pyarrow` 和 `fastparquet` 引擎完成。这两个引擎有不同的依赖关系。Pyarrow 不支持 Timedelta。

让我们使用 `pyarrow` 引擎读取一个 parquet 文件：

```py
pd.read_parquet("sample.paraquet",engine='pyarrow')
```

这将产生以下输出：

![](img/1a4f82f0-be20-408d-805a-3cb1e7160638.png)

read_parquet 输出

Parquet 允许我们在读取文件时选择列，这可以节省时间：

```py
pd.read_parquet("sample.paraquet",engine='pyarrow',columns=["First_Name","Score"])
```

`fastparquet` 引擎也适用相同的方法：

```py
pd.read_parquet("sample.paraquet",engine='fastparquet')
```

# 读取 SQL 文件

通过 pandas 与 SQL 数据库交互需要安装 sqlalchemy 依赖。

首先，让我们定义获取连接参数的引擎：

```py
engine = create_engine('sqlite:///:memory:')
```

现在，让我们从 SQL 数据库中读取 `data_sql` 表：

```py
with engine.connect() as conn, conn.begin():
    print(pd.read_sql_table('data_sql', conn))
```

这将产生以下输出：

![](img/e991f948-810a-4d79-9894-cd22d844b593.png)

read_sql_table 输出

`read_sql_table()` 函数用于读取给定表名的整个表。读取时，可以将特定的列设置为索引：

```py
pd.read_sql_table('data_sql', engine, index_col='index')
```

这将产生以下输出：

![](img/357a68c7-d791-454a-8dfe-fdff32a5188c.png)

使用索引的 read_sql_table 输出

`columns` 参数允许我们在读取数据时选择特定的列，通过传递列名列表来实现。任何日期列都可以在读取过程中解析为特定格式，如下代码所示：

```py
pd.read_sql_table('data_sql', engine, parse_dates={'Entry_date': '%Y-%m-*%d*'})
pd.read_sql_table('data_sql', engine, parse_dates={'Entry_date': {'format': '%Y-%m-*%d* %H:%M:%S'}})
```

此函数中的 `schema` 参数有助于指定要从中提取表格的模式。

除了读取整个表格外，还可以使用 SQL 查询以所需格式获取数据。我们可以通过 `read_sql_query()` 函数来实现：

```py
pd.read_sql_query("SELECT Last_name, Score FROM data_sql", engine)
```

这将导致以下输出：

![](img/5276913a-81d8-4b1e-a138-bf3e32b90d67.png)

`read_sql_query` 的输出

要执行 `INSERT` 和 `CREATE` 查询，这些查询不会返回任何输出，可以使用 `sql.execute()` 函数。这需要导入一个 `pandas.io` 的 `sql` 文件：

```py
from pandas.io import sql
sql.execute("INSERT INTO tablename VALUES (90, 100, 171)", engine)
```

使用 `sqlite` 数据库时，必须按照如下方式定义与引擎的连接，以便可以在 `read_sql_table()` 或 `read_sql_query()` 函数中使用。必须先导入 `sqlite` 模块：

```py
import sqlite3
conn = sqlite3.connect(':memory:')
```

# 读取 SAS/Stata 文件

Pandas 可以读取来自 SAS 的两种文件格式——SAS xports（`.XPT`）和 SAS 数据文件（`.sas7bdat`）。

`read_sas()` 函数帮助读取 SAS 文件。这里已读取一个 SAS 数据文件，并作为 pandas 数据框显示：

```py
df = pd.read_sas('sample.sas7bdat')
df
```

这将导致以下输出：

![](img/83cb9f2b-0d6c-428a-bde7-f4e3ebaa063e.png)

`read_sas` 的输出

`chunksize` 和 `iterator` 参数有助于按相同大小的组读取 SAS 文件。如果之前使用的 SAS 数据文件以 chunksize 10 进行读取，则 51 条记录将被分为六组，如下代码所示：

```py
rdr = pd.read_sas('sample.sas7bdat', chunksize=10)
for chunk in rdr:
print(chunk.shape)
```

看一下以下输出：

![](img/b71ebc09-1929-443e-bdf7-1a75cf72d09c.png)

使用 chunksize 的 `read_sas` 输出

然而，这些 SAS 文件无法使用 pandas 写入。

Pandas 还支持读取和写入从 Stata 生成的文件。Stata 只支持有限的数据类型：`int8`、`int16`、`int32`、`float32`、`float64`，以及长度小于 244 的字符串。在通过 pandas 写入 Stata 数据文件时，会在适当的地方应用类型转换。

让我们使用 pandas 读取一个 Stata 数据文件：

```py
df = pd.read_stata('sample.dta')
df
```

看一下以下输出：

![](img/a760ba3a-248a-4b8d-9d52-6bb2a95e589c.png)

`read_stata` 的输出

`read_stata()` 函数还具有 `chunksize` 和 `iterator` 参数，用于按较小的组读取数据。以下是可用的 `stata` 读取函数参数：

+   `convert_categoricals`：将合适的列转换为分类数据类型

+   `index_col`：指定要定义为索引的列

+   `convert_missing`：指定是否将缺失值表示为 NaN 或 Stata 的缺失值对象

+   `columns`：要从数据集中选择的列

# 从 Google BigQuery 读取数据

BigQuery 是 Google 提供的一个极其强大的数据仓储解决方案。Pandas 可以直接连接到 BigQuery，并将数据带入 Python 环境进行进一步分析。

以下是从 BigQuery 读取数据集的示例：

```py
pd.read_gbq("SELECT urban_area_code, geo_code, name, area_type, area_land_meters 
FROM `bigquery-public-data.utility_us.us_cities_area` LIMIT 5", project_id, dialect = "standard")
```

看一下以下输出：

![](img/7131995c-654c-4c9f-a9a1-74c5225ac87a.png)

read_gbq 的输出

`read_gbq()` 函数接受查询和 Google Cloud 项目 ID（作为密钥）以便访问数据库并提取数据。dialect 参数处理 SQL 语法的使用：BigQuery 的旧版 SQL 方言或标准 SQL 方言。此外，还有允许设置索引列（`index_col`）、重新排序列（`col_order`）以及启用重新认证（`reauth`）的参数。

# 从剪贴板读取数据

这是 pandas 中一个非常有趣的功能。任何已经复制到剪贴板的表格数据都可以作为 DataFrame 在 pandas 中读取。

我们可以使用常规的 *ctrl + C* 键盘命令复制以下表格数据：

|  | **性别** | **入职日期** | **标记** |
| --- | --- | --- | --- |
| **A** | M | 2012-01-19 | True |
| **B** | F | 2012-12-30 | False |
| **C** | M | 2012-05-05 | False |

调用 `read_clipboard()` 函数会将剪贴板中的数据转换为 pandas DataFrame：

```py
pd.read_clipboard()
```

看一下以下输出：

![](img/9479c71d-b380-484b-9e00-c217d19a956f.png)

read_clipboard 的输出

该函数默认也会将 **Flag** 列识别为布尔类型，并将未命名的列作为索引：

![](img/8ab745a9-7b80-4af3-bd88-83f89e40fe90.png)

从剪贴板读取后的数据类型

# 管理稀疏数据

稀疏数据指的是像数组、系列、DataFrame 和面板等数据结构，其中有很大一部分数据缺失或为 NaN。

我们来创建一个稀疏的 DataFrame：

```py
df = pd.DataFrame(np.random.randn(100, 3))
df.iloc[:95] = np.nan
```

该 DataFrame 中 95% 的记录包含 NaN。可以通过以下代码估算该数据的内存使用情况：

```py
df.memory_usage()
```

看一下以下输出：

![](img/395b1fb1-ccac-4920-9d60-8879499c651a.png)

含有 95% NaN 的 DataFrame 的内存使用情况

如我们所见，每个元素消耗 8 字节的内存，无论它是实际数据还是 NaN。Pandas 提供了一种内存高效的解决方案来处理稀疏数据，如下代码所示：

```py
sparse_df = df.to_sparse()
sparse_df.memory_usage()
```

看一下以下输出：

![](img/006a584e-c19f-4aa2-b459-408b6d75f790.png)

稀疏数据的内存使用情况

现在，内存使用量已经减少，NaN 不再分配内存。这也可以通过定义一个 `fill_value` 来代替 NaN 实现：

```py
df.fillna(0).to_sparse(fill_value = 0)
df.fillna(0).to_sparse(fill_value = 0).memory_usage()
```

看一下以下输出：

![](img/e92501ca-3ad0-4c92-9cdb-c15e3cfb8a0c.png)

填充数据后的稀疏数据内存使用情况

稀疏数据也可以转换回原始的密集形式，如下代码所示：

```py
sparse_df.to_dense()
```

这种处理稀疏数据的方式也可以类似地应用于系列、面板和数组。

# 将 JSON 对象写入文件

`to_json()` 函数允许将任何 DataFrame 对象转换为 JSON 字符串，或者如果指定文件路径，则写入 JSON 文件：

```py
df = pd.DataFrame({"Col1": [1, 2, 3, 4, 5], "Col2": ["A", "B", "B", "A", "C"], "Col3": [True, False, False, True, True]})
df.to_json()
```

看一下以下输出：

![](img/3093bb79-a18a-40aa-9758-d5a8d340405b.png)

JSON 输出

JSON 数据的方向可以被更改。`to_json()` 函数具有一个 orient 参数，可以设置为以下模式：columns、index、record、value 和 split。默认的方向设置是列（columns）：

```py
df.to_json(orient="columns")
```

请看以下输出：

![](img/a98a5f64-dc5b-44bd-8e85-be398a6fc207.png)

JSON 输出 – 列方向

沿索引方向的设置就像是前一个情况的转置，JSON 字典中的行和列索引被反转：

```py
df.to_json(orient="index")
```

请看以下输出：

![](img/b257e514-0a3d-4c8e-b9cb-4e236929305f.png)

JSON 输出 – 索引方向

将 orient 设置为记录（records）会创建一个 JSON 结构，其中原 DataFrame 中的每个记录或行保留其结构形式：

```py
df.to_json(orient="records")
```

请看以下输出：

![](img/2238dc0e-150a-451d-866b-651b27862cb4.png)

JSON 输出 – 记录方向

当 orient 选项设置为值（values）时，行索引和列索引都会从图像中消失：

```py
df.to_json(orient="values")
```

请看以下输出：

![](img/a0d7aa85-911e-4556-94e7-6db105f4765f.png)

JSON 输出 – 值方向

拆分方向定义了一个由实体如列、索引和数据组成的 JSON：

```py
df.to_json(orient="split")
```

请看以下输出：

![](img/c909c169-02bf-4ebc-a2e7-40fb9626bad0.png)

JSON 输出 – 拆分方向

将 orient 设置为表格会展示出如架构和字段等方面：

```py
df.to_json(orient="table")
```

请看以下输出：

![](img/4214d637-f8be-4ec1-9a61-40762c7b7e15.png)

JSON 输出 – 表格方向

`to_json()` 的 `date_format` 参数允许将 DataFrame 中的时间戳转换为 `epoch` 格式或 `iso` 格式。

对于像 `complex` 这样的不支持的数据类型，可以通过 `default_handler` 参数指定要执行的类型转换来处理。

# 序列化/反序列化

序列化是将 [数据结构](https://en.wikipedia.org/wiki/Data_structure) 或 [对象](https://en.wikipedia.org/wiki/Object_(computer_science)) 的状态转换为一种可以存储（例如，在 [文件](https://en.wikipedia.org/wiki/Computer_file) 或内存 [缓冲区](https://en.wikipedia.org/wiki/Data_buffer) 中）或通过 [网络](https://en.wikipedia.org/wiki/Computer_network) 连接传输的格式，并在之后重新构建（可能在不同的计算环境中）。[[1]](https://en.wikipedia.org/wiki/Serialization#cite_note-1) 当根据序列化格式重新读取该位序列时，它可以用来创建一个语义上与原始对象完全相同的克隆。

数据结构如 JSON、数组、DataFrame 和 Series 有时需要作为物理文件存储或通过网络传输。这些序列化可以理解为数据的转储，数据可以以任何格式（文本、CSV 等）或结构存储，但所有重要的数据点可以通过加载/反序列化恢复。

这些的例子包括存储统计模型的训练模型对象的参数。这个包含训练参数的序列化文件可以被加载，并且测试数据可以通过它进行预测。这是将统计模型投入使用的一种流行方法。

序列化数据格式的其他用途包括通过网络传输数据、在数据库或硬盘中存储对象、进行远程过程调用，以及检测时间变化数据的变化。

让我们创建一个示例 DataFrame，来理解 Pandas 支持的各种文件格式的序列化：

```py
df = pd.DataFrame({"First_Name":["Mike","Val","George","Chris","Benjamin"],
"Last_name":["K.","K.","C.","B.","A."],
"Entry_date":pd.to_datetime(["June 23,1989","June 16,1995","June 20,1997","June 25,2005","March 25,2016"],format= "%B %d,%Y"),
"Score":np.random.random(5)})
df
```

看一下以下输出：

![](img/20da6e5b-781d-4b37-9945-e38614c41d24.png)

用于序列化的 DataFrame

# 写入特殊文件类型

存储数据结构或对象的格式有多种。让我们来看看其中一些格式。

# to_pickle()

当一个 Python 对象被 pickle 时，它会被保存到磁盘。Pickling 首先序列化对象，然后写入磁盘。它包括将列表、字典、DataFrame 和训练的机器学习模型等对象转换为字符流。

让我们将之前定义的 DataFrame 转换成 pickle 格式：

```py
df.to_pickle('pickle_filename.pkl')
```

还可以在写入之前压缩 pickle 文件。支持`gzip`、`bz2`和`xz`等压缩方案：

```py
df.to_pickle("pickle_filename.compress", compression="gzip")
```

默认情况下，压缩类型是根据提供的扩展名推断的：

```py
df.to_pickle("pickle_filename.gz")
```

`read_pickle()`函数将反序列化`pickle`文件。压缩仅支持读取单个文件，不支持写入。

# to_parquet()

正如我们在*读取 parquet 文件*部分中讨论的那样，也可以使用两种引擎进行反序列化：

```py
df.to_parquet('sample_pyarrow.parquet', engine='pyarrow')
df.to_parquet('sample_fastparquet.parquet', engine='fastparquet')
```

# to_hdf()

HDF 文件像字典一样，可以存储多个对象。`to_hdf()`函数将 Pandas 对象转换为 HDF 文件：

```py
df.to_hdf('store.h5', append = True, format='table')
```

当一行的所有列都是 NaN 时，它们不会自动删除。可以通过将`dropna`参数设置为`True`来在写入 HDF 时删除这些行。

# to_sql()

通过`sqlalchemy`包的支持，数据可以通过 pandas 传输到数据库：

```py
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df.to_sql('data_sql',engine)
```

还可以通过使用`chunksize`参数迭代地批量推送数据。

在将数据推送到数据库时，任何列的数据类型也可以更改，如以下代码所示：

```py
from sqlalchemy.types import String
df.to_sql('data_dtype', engine, dtype={'Score': String})
```

`Timedelta`数据类型（在数据库中不支持）会在存储到数据库之前，转换为等效的纳秒整数值。

# to_feather()

将 pandas 对象序列化为 feather 格式只需要调用`to_feather()`函数：

```py
df.to_feather('sample.feather')
```

# to_html()

`to_html()`函数将 DataFrame 转换为原始 HTML 格式：

```py
df.to_html()
```

这将产生以下输出：

![](img/91c4cda7-34b2-4cd7-bb92-f9548df02cf8.png)

HTML 格式的 DataFrame

`to_html()`函数中有一大堆选项，可以丰富原始的 HTML。通过使用`columns`和`escape`参数，可以选择列并控制转义序列。

# to_msgpack()

Msgpack 提供快速高效的二进制序列化。

单个对象可以像这样直接转换为`msgpack`格式：

```py
df.to_msgpack("sample.msg")
```

如果我们有多个对象，它们可以像这样序列化为一个`msgpack`文件：

```py
arr = np.random.randint(1,10,7)
lst = [10,20,40,60,60]
strg = "Data"
pd.to_msgpack("msg_all.msg",df,arr,lst,strg)
```

# to_latex()

`to_latex()`函数将 DataFrame 转换为兼容`latex`文档的美观表格结构：

```py
df.to_latex()
```

看一下以下输出：

![](img/f3f596fb-88e6-4ca5-99d7-6ef2497ef3a2.png)

LaTeX 格式的 DataFrame

# to_stata()

Pandas 可以帮助创建带有`.dta`扩展名的`stata`数据文件，如下代码所示：

```py
df.to_stata('stata_df.dta')
```

# to_clipboard()

`to_clipboard()`函数将 DataFrame 从 Python 环境转移到剪贴板。然后可以通过*ctrl* + *V*键盘命令将对象粘贴到其他地方：

```py
df.to_clipboard()
```

这个 DataFrame 也可以以更兼容 CSV 的格式复制到剪贴板，如下所示：

```py
df.to_clipboard(excel=True,sep=",")
```

# GeoPandas

GeoPandas 是一个基于 pandas 的 Python 包，用于处理地理空间数据。它旨在与现有工具兼容，如桌面 GIS、地理空间数据库、网页地图和 Python 数据工具。

GeoPandas 允许你轻松地在 Python 中执行本来需要空间数据库（如 PostGIS）才能完成的操作。

# 什么是地理空间数据？

空间数据、地理空间数据、GIS 数据和地理数据是指通过地理坐标系统标识物理对象（如建筑物、街道、城镇、城市、国家等）的地理位置的数值数据**。** 除了地理位置，地理空间数据通常还存储每个位置的社会经济数据、交易数据等。

# 安装与依赖

可以通过 pip 或 Anaconda，或直接通过 GitHub 安装 GeoPandas。最常见的安装方法是通过`pip`和 Anaconda 在终端窗口中进行：

```py
pip install geopandas 
conda install -c conda-forge geopandas 
```

GeoPandas 依赖于以下 Python 库：

+   `pandas`

+   `numpy`

+   `shapely`

+   `fiona`

+   `pyproj`

+   `six`

+   `rtree`

# 使用 GeoPandas

我们可以使用 GeoPandas 库通过`geopandas.read_file`函数读取许多 GIS 文件格式（依赖于`fiona`库，这是 GDAL/OGR 的接口）。

数据也可以通过 shapefiles 读取。在本节中，我们将通过一个示例演示如何使用 GeoPandas。我们将解释如何读取包含地理空间数据的 shapefile，对其进行聚合、排序，最后绘制所需的 Geo DataFrame。

使用以下代码调用所需的先决库：

```py
    import pandas as pd
    import geopandas
    import matplotlib as plt
```

使用以下代码读取包含地理空间信息数据的 shapefile：

```py
countries = geopandas.read_file("ne_110m_admin_0_countries.shp") 
```

使用以下代码访问数据集的前五行，就像我们使用 pandas 一样：

```py
countries.head(5) 
```

前面的代码片段将产生以下输出：

![](img/9ce0578f-c959-4a75-bd6b-e3c778dcbd1f.png)

作为 DataFrame 读取的地理空间 shapefile

让我们快速绘制数据的基本可视化：

```py
countries.plot() 
```

这将产生以下输出：

![](img/5db74b7f-59a8-46b4-bed9-66be5731c66d.png)

在地图上绘制的形状文件中的国家

要检查我们地理空间数据的类型，可以使用以下代码：

```py
type(countries) 
```

这将产生以下输出：

![](img/b7a1f2d4-33c6-46b5-b852-2b0e93eb46b6.png)

在转换后，断言国家形状文件的数据类型是 GeoDataFrame

在这里，我们可以看到该 DataFrame 是一个 GeoDataFrame。现在，让我们讨论一下 GeoDataFrame 是什么。

# GeoDataFrames

GeoDataFrame 包含一个地理空间数据集。它就像一个 pandas DataFrame，但具有一些额外的功能，用于处理地理空间数据。这些附加功能如下：

+   `.geometry` 属性始终返回包含几何信息的列（返回一个 GeoSeries）。列名本身不一定需要是 `.geometry`，但它将始终可以作为 `.geometry` 属性访问。

+   它有一些额外的方法来处理空间数据（如面积、距离、缓冲区、交集等），这些方法我们将在后面的章节中逐一探讨。

GeoDataFrame 仍然是一个 DataFrame，因此我们可以在 GeoDataFrame 中执行 DataFrame 中所有可用的功能，比如聚合、排序、过滤等。

使用以下代码执行 GeoPandas 的简单聚合：

```py
countries['POP_EST'].mean()  
```

`POP_EST` 是 `countries` GeoDataFrame 中的一列，数据类型为数字。 这将产生以下输出：

![](img/c9bc4306-c07b-4ef9-af33-5110ae805ff2.png)

对 GeoDataFrame 中的数字列进行聚合，证明它的工作方式与普通 DataFrame 完全相同

或者，我们可以使用布尔过滤来基于条件选择 DataFrame 的子集：

```py
africa = countries[countries['CONTINENT'] == 'Africa'] 
```

现在，我们将尝试使用 `plot()` 函数绘制过滤后的 GeoDataFrame：

```py
africa.plot() 
```

这将产生以下输出：

![](img/1812f806-00e9-48b4-83f9-553a54086457.png)

对一个大洲进行子集化并绘制以提高可视化效果

GeoPandas 还可以帮助将普通 DataFrame 转换为 GeoDataFrame，前提是你拥有 `Latitude` 和 `Longitude` 坐标。让我们来看一下这个例子。

假设我们有一个简单的 DataFrame，像这样：

```py
df = pd.DataFrame( 
{'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'], 
'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'], 
'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48], 
'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]}) 
```

上面的代码将产生以下输出：

![](img/33331910-d469-4863-9e25-20be419b7a83.png)

创建一个包含国家首都及其经纬度的普通 DataFrame

让我们添加一个新列，命名为 `'Coordinates'`，它将经纬度列连接起来：

```py
   df['Coordinates']  = list(zip(df.Longitude, df.Latitude)) 
```

这将产生以下输出：

![](img/d86862f8-a170-4390-97fd-193d8b46b2bb.png)

一个将纬度和经度合并在一列中的普通 DataFrame

使用 shapely 包中的 `Point` 函数，我们可以正确地将这些标识为位置坐标或点元组参数，它们是 GeoDataFrame 的骨架。

```py
from shapely.geometry import Point 
   df['Coordinates'] = df['Coordinates'].apply(Point) 
```

这将产生以下输出：

![](img/86e05119-7fff-4636-80a6-b3214ef99f72.png)

将压缩的经度和纬度转换为点，以便 GeoPandas 可以使用它将其转换为 GeoDataFrame

一切准备就绪后，让我们将其转换为 GeoDataFrame：

```py
gdf = geopandas.GeoDataFrame(df, geometry='Coordinates') 
```

让我们打印出`gdf`类型，看看它的 GeoDataFrame 类型：

```py
   type(gdf) 
```

这将产生以下输出：

![](img/ecdd1986-6e59-4459-9087-39a7fe552661.png)

确认新创建的 DataFrame 的类型是 GeoDataFrame

这为我们提供了关于 GeoPandas 的基本概念，以及它是如何工作的。它的应用范围广泛，你可以跨越其各种功能并从中获益。

# 开源 API – Quandl

Python 可以用来从开源和商业 API 获取数据。我们可以用它获取多种格式的数据。有些数据以 JSON 格式输出，有些以 XML 格式输出，还有一些以表格格式，如 CSV 和 DataFrame 输出。数据转换为 DataFrame 后，通常会在 pandas 中进行处理。

在本节中，我们将展示一个从 Quandl API 获取数据的示例，Quandl 是一个开源 API，包含各种主题的数据，如金融、经济和替代数据。你可以在这里查看这个著名的数据仓库：[`www.quandl.com/`](https://www.quandl.com/)。

`api`密钥是一种应用程序接口，充当开发者或任何希望通过计算机代码访问网站数据的用户之间的中介。`api`密钥是一段代码，用于识别用户及其关联的账户。

要开始这个示例，你需要注册一个免费的 Quandl 账号。注册后，你可以在账户设置选项中找到 API 密钥，该选项会显示在个人资料图片的下拉菜单中。

Python 通过提供一个可以与最新版本的 Quandl 数据仓库交互的包，使得使用 Quandl API 变得更加容易。这个包兼容 Python 2.7 及以上版本。

首先，你需要通过`pip`或`conda`安装 Quandl 包，使用以下命令：

```py
pip install quandl
conda install quandl
```

你可以获取任何你希望使用的数据集。在这里，我使用的是 2022 年 7 月的巴西雷亚尔期货数据集作为示例。你需要找到你想要下载的数据集的代码。这个代码可以从 Quandl 网站获取，如下图所示：

![](img/6b21ddba-b76c-4576-b7e6-109183f6a825.png)

在 Quandl 网站上查找数据集的数据代码

现在，让我们来看一下如何使用 Quandl API 来获取我们想要的数据：

```py
    # import quandl into your code
    import quandl

    # setting your api key
    quandl.ApiConfig.api_key = "[YOUR API KEY]"

    # BCCME/L6N2022 is the code for the dataset and you can see it on the right below the table
    data = quandl.get("BCCME/L6N2022", start_date="2015-12-31", end_date="2019-06-23")

    data.head()  
```

这将产生以下输出：

![](img/3b2db996-2bea-4c2c-9b41-18c247cd38f2.png)

通过 Quandl API 获取的巴西雷亚尔期货数据

API 还可以用来获取数据的子集，而不是一次性获取所有数据。例如，在这里，我们筛选了某个日期范围内的数据：

```py
    transform =  quandl.get("BCCME/L6N2022", start_date="2015-12-31", end_date="2019-06-23",transformation='diff')
    transform.head()
```

这将产生以下输出：

![](img/4377abc8-7f48-40fc-bd88-b142a4cb1b36.png)

通过 Quandl API 获取的巴西雷亚尔期货数据，带有日期范围过滤

一旦数据被读取，pandas 可以用来对数据进行所有转换，这将对进一步的分析非常有帮助。

数据集也可以通过提供数据集的网址来下载。通过下载一个文件列出可用的数据集，您可以检查这一点。让我们尝试通过 Python 的`urllib`包来下载一个文件，而不是按照`Quandl`包的方法。

要获取可以在此方法中使用的网址，请按照以下步骤操作：

1.  点击数据集的标题/链接（标记在红色框中）：

![](img/27f37797-063f-4bb8-bdd5-858dc4fb0b66.png)

数据主题链接以获取更多关于该主题的详细信息

1.  点击链接将带您到下一个页面，您将在该页面看到这些选项。在“用法”选项下选择 API，如下图所示：

![](img/3805a13b-b080-48cf-b9c3-48710a3fbe5f.png)

数据主题文档

1.  在此选择之后，您应该稍微向下滚动一下，找到以下网址，可以在代码中使用它们来获取数据：

![](img/bd0bf04a-881e-4eb0-af08-610d5189a8f4.png)

从数据主题文档中获取的 Quandl 数据的 API 链接

1.  由于一个主题可能包含多个数据集，主题会作为`.zip`文件下载，文件中包含所有数据集。它提供了一个元数据表，并且列出了每个数据集的详细信息（包括数据集键）：

```py
    import urllib.request
    print('Beginning file download with urllib...')
    url = 'https://www.quandl.com/api/v3/databases/EOD/metadata?api_key=[YOUR API KEY]'
    urllib.request.urlretrieve(url,'file location.zip')  

    # We will read the zip file contents through the zipfile package.
    import zipfile
    archive = zipfile.ZipFile('[Name of the zip file].zip', 'r')

    # lists the contents of the zip file
    archive.namelist()
    ['EOD_metadata.csv']
    df = pd.read_csv(archive.open('EOD_metadata.csv'))    
    df.head()

```

这会产生以下输出：

![](img/507b5a15-cb3c-4309-99b6-531fd6a82d48.png)

下载的数据主题的元数据表输出

# read_sql_query

Python 支持很多数据库操作，使用如`psycopg2`和`sqlalchemy`等库。它们在从 Python 接口操作数据库时非常全面和有用。然而，它们有自己的繁琐内容，有时会对简单的查询任务产生过多的信息。幸运的是，pandas 中有一个隐藏的宝石，那就是`read_sql_query`方法。它的功能如下：

+   运行包含 select、where 等的简单查询。

+   运行所有返回表格或其子集的查询，并以表格形式展示。

+   无法使用 INSERT、UPDATE 和 DELETE 语句。

+   输出是一个数据框，因此可以使用所有 pandas 方法进行进一步的数据处理。

让我们看看如何利用这个方法。为了说明这一点，我们将把一个数据集作为表格插入到数据库中。为此，您需要在本地目录中安装 PostgreSQL 或 SQL 数据库。如果您已经设置好了数据库，可以跳过表格创建过程，直接跳到查询过程。

让我们从 Kaggle 下载 2019 年世界幸福感数据集，将其推送到`db`，并对其进行各种 DB 和 pandas 操作：

```py
    import pandas as pd
    df = pd.read_csv('F:/world-happiness-report-2019.csv')
    df.head()
```

以下数据展示了世界幸福报告作为一个数据框：

![](img/b3bac349-5720-4957-9446-3d44cee6a066.png)

世界幸福报告作为一个数据框

由于我们将直接从之前生成的 DataFrame 创建一个表，因此需要将列名更改为 postgresql，因为它不支持带空格的列名：

```py
    # rename Country (region) to region
    df= df.rename(columns={'Country (region)':'region'})

    # push the dataframe to postgresql using sqlalchemy
    # Syntax:engine = db.create_engine('dialect+driver://user:pass@host:port/db')

    from sqlalchemy import create_engine
    engine = create_engine('postgresql://postgres:1128@127.0.0.1:5433/postgres')
    df.to_sql('happy', engine,index=False)      
```

为了使用`read_sql_query`方法，我们需要通过`psycopg2`或`sqlalchemy`与数据库建立连接。连接建立后，就可以完整地使用`read_sql_query`方法：

```py
    import psycopg2
    try:
       connection = psycopg2.connect(user="[db-user_name]",
                                      password="[db-pwd]",
                                      host="127.0.0.1",
                                      port="5433",
                                      database="postgres")

       happy= pd.read_sql_query("select * from happy;",con=connection).head()      
```

这将生成以下输出：

![](img/97849629-087f-4e41-a3f4-2c00306967da.png)

从 PostgreSQL 数据库中的表查询出的世界幸福报告数据作为 DataFrame

看一下下面的代码。它有助于运行 SQL 查询。

```py
posgrt40 = pd.read_sql_query('select * from happy where "Positive affect" > 40;',con=connection).head()

```

这将生成以下输出：

![](img/ce2363d9-2dc4-43f9-9fc5-31592f4d4844.png)

使用从 PostgreSQL 数据库中的表查询出的带过滤器的世界幸福报告数据作为 DataFrame

```py
except (Exception, psycopg2.Error) as error :
print ("Error while fetching data from PostgreSQL", error)  
```

`pd.read_sql_query()`方法将结果返回为 DataFrame，而无需程序员介入并将数据转换为所需格式。

# Pandas 绘图

一图胜千言。这就是为什么图表通常用于直观地展示数据之间的关系。图表的目的是呈现那些过于庞大或复杂，以至于无法通过文本充分描述的数据，并且能在更小的空间内展示。使用 Python 的**绘图函数**，只需要几行代码就可以创建出生产质量的图形。

我们将开始安装所需的包：

```py
import pandas as pd 
import numpy as np 
```

我们在这里使用`mtcars`数据来解释这些图表：

```py
mtcars = pd.DataFrame({ 
        'mpg':[21,21,22.8,21.4,18.7,18.1,18.3,24.4,22.8,19.2], 
        'cyl':[6,6,4,6,8,6,8,4,4,4], 
        'disp':[160,160,108,258,360,225,360,146.7,140.8,167.7], 
  'hp':[110,110,93,110,175,105,245,62,95,123],    
'category':['SUV','Sedan','Sedan','Hatchback','SUV','Sedan','SUV','Hatchback','SUV','Sedan'] 
        }) 
mtcars 
```

这将生成以下输出：

![](img/2d0b263d-0e95-49e8-9a1b-f32841cfce3b.png)

mtcars DataFrame

让我们详细讨论`pandas.plotting`中的各种绘图。

# Andrews 曲线

Andrews 曲线是一种用于可视化多维数据的方法。它通过将每个观察值映射到一个函数来实现。在这里，每种使用的颜色代表一个类别，我们可以很容易地看到，代表同一类别的样本线具有相似的曲线。这种曲线在分析时间序列和信号数据时非常有用。

基本上，每个数据点都根据傅里叶函数进行傅里叶变换。以下图表中的每条线代表一个单独的数据点。可以使用下面的代码片段绘制它。

```py
andrew = pd.plotting.andrews_curves(mtcars,'Category') 
```

以下是图表：

![](img/c372bbe4-daf0-4f94-946e-9706f991bbf0.png)

Andrews 曲线图

# 并行图

并行绘图最适合在需要比较每个点的多个变量并理解它们之间关系时使用，例如，当你需要比较一组具有相同属性但不同数值的变量时（例如，比较不同型号的摩托车规格）。

每条连接线代表一个数据点。垂直线表示已为每个数据点绘制的列或变量的值。拐点（用红色标记）表示这些变量在这些点上的值。可以通过以下代码非常轻松地绘制并行图：

```py
parallel = pd.plotting.parallel_coordinates(mtcars,'Category') 
```

这会生成以下输出：

![](img/d63564e2-d01a-4300-8c17-66bc6ddb8dd5.png)

并行图

# Radviz 图

Radviz 图允许探索多任务分类问题。它以二维投影展示三个或更多变量的数据。这个图像类似于一个圆形，数据点位于圆形内部。变量则位于圆圈的周边。

每个点的位置由所有构成它的变量值决定。创建一个虚拟圆圈，并将变量放置在这个圆圈上。点位于圆圈的边缘内。点的确切位置由各个变量施加的力的总和为零的位置决定。每个变量施加的力可以看作是弹簧力，并遵循胡克定律（F = kx）：

![](img/29eadd53-789c-4924-b7c2-96032063ac2c.png)

Radviz 图解释

上图可以通过运行下面的代码片段来获得。

```py
rad_viz = pd.plotting.radviz(mtcars, 'Category') 
```

这会生成以下输出：

![](img/0d7e97fb-4d6c-44ee-a265-6e41c7fd895c.png)

Radviz 图

# 散点矩阵图

散点矩阵图由多个变量的图表组成，所有这些图表都以矩阵格式呈现。基本上，会创建一个 2 x 2 的变量矩阵，其中每个单元格表示两个变量的组合。然后，为每个组合生成一个散点图。它可以用来确定变量之间的相关性，广泛应用于维度缩减的情况：

```py
scatter = pd.plotting.scatter_matrix(mtcars,alpha = 0.5) 
```

这会生成以下输出：

![](img/e32e1878-bc5c-4c40-a6f0-19eeb6f9c660.png)

散点矩阵图

# 延迟图

延迟图是一种特殊类型的散点图，包含变量（X、X-滞后等）。

X-滞后是由 X 衍生出的带时间滞后的变量。该图绘制了两个变量之间的关系，用于确定数据的随机性、模型适应性、异常值和序列相关性——特别是时间序列数据：

```py
s = pd.Series(np.random.uniform(size=100)) 
```

这会生成以下输出：

![](img/68f06f9e-8f57-4809-b3a3-1bd8cf53baf8.png)

延迟图

# 自助法图

自助法图用于确定统计量的统计不确定性，例如均值、中位数、最小-最大值等。它依赖于带有替换的随机抽样方法。通过从同一数据中随机抽取多次样本并计算每次抽样的结果平均值，这一过程称为自助抽样。自助法图基本上绘制了从每个随机样本中得到的所有结果值。它计算所有样本的均值、中位数和众数，并将它们绘制为条形图和折线图。

从数据中选择一个随机样本，重复该过程若干次，以获得所需的指标。最终得到的图表就是一个自助法图：

```py
fig = pd.plotting.bootstrap_plot(s) 
```

这将产生以下输出：

![](img/fdd433ce-4e14-47e1-ad5a-40d2c899fec3.png)

延迟图

# pandas-datareader

我们可以使用 pandas 不仅从本地 CSV 或文本文件中读取数据，还可以从各种流行的远程数据源中读取数据，如 Yahoo Finance、世界银行等。如果没有 pandas 的支持，这将变得十分繁琐，我们可能不得不依赖网页抓取。通过`pandas-datareader`，这个简单且强大的功能得到了提供。

它为我们提供了一种通过 pandas 生态系统直接连接到各种数据源的方式，无需深入 HTML/JavaScript 代码的复杂性，在这些代码中数据被嵌套。只需提供数据源名称和数据代码即可访问这些数据源，且仅能获取数据的一个子集。

让我们深入了解，看看如何使用它：

1.  通过以下命令使用`pip`安装`pandas-datareader`：

```py
pip install pandas-datareader  
```

您也可以通过以下命令集通过`conda`进行安装：

1.  首先，我们需要将`conda-forge`添加到我们的频道：

```py
conda config --add channels conda-forge  
```

1.  启用`pandas-datareader`后，可以使用以下代码进行安装：

```py
conda install pandas-datareader  
```

现在，让我们动手使用一些远程数据源，并通过 pandas 库提供的各种功能，了解它是如何工作的。

# Yahoo Finance

如果您对了解商业世界的趋势感兴趣，并渴望获取关于[股票](https://money.howstuffworks.com/personal-finance/financial-planning/stocks.htm)和[债券](https://money.howstuffworks.com/personal-finance/budgeting/bonds.htm)的最新动态，或者您是其中的投资者，您可能会渴望每分钟更新一次。Google Finance 是由 Google 开发的一个金融网站，它通过提供所需的信息并允许我们根据兴趣自定义需求，使这一过程变得更加简便。

由于 API 中存在较大的中断，Google 的 API 在 2017 年以后变得不那么可靠，并且由于没有稳定的替代品，已经变得高度弃用。

其替代方案是**Yahoo** Finance，它与 Google Finance 类似，并且因其强大的数据和一致性而在用户中颇受欢迎。

现在，让我们使用`pandas-datareader`通过 Google Finance API 获取与股票、共同基金以及任何与金融相关的信息。

```py
    import pandas as pd
    from pandas_datareader import data
    symbols=['AAPL','GOOGL','FB','TWTR']
    # initializing a dataframe
    get_data = pd.DataFrame()
    stock = pd.DataFrame()

    for ticker in symbols:
       get_data = get_data.append(data.DataReader(ticker, 
                           start='2015-1-1', 
                           end='2019-6-23', data_source='yahoo'))

       for line in get_data:
            get_data['symbol'] = ticker

       stock = stock.append(get_data)
       get_data = pd.DataFrame()

    stock.head()

```

上述代码将产生以下输出：

![](img/8c88926d-7f47-4230-b091-596d14be8e2e.png)

苹果、谷歌、Facebook 和 Twitter 的日期过滤股票数据

```py
stock.describe() 
```

这将产生以下输出：

![](img/0845aa70-e0bd-489d-9f36-6bfc68a14f7c.png)

股票数据的汇总统计

请查看以下代码：

```py
 # get the list of column names
    cols = [ col for col in stock.columns]
    cols                      
```

这将产生以下输出：

![](img/f1b34b5d-8aca-4956-9ebb-42e4a4861bb4.png)

请查看以下代码：

```py
 # returns the symbol of the highest traded value among the symbols
 stock.loc[stock['High']==stock['High'].max(), 'High']
```

这将得到以下输出：

![](img/d74262cf-8abb-48c4-8a9b-f2c91efecc78.png)

上述代码将返回一个 DataFrame，提供`Apple[AAPL]`、`Google[GOOGL]`、`FB[FB]` 和 `Twitter[TWTR]` 在这两个日期之间的每一天的股票价格详细信息。

在进行任何分析之前，了解你的数据非常重要。请考虑以下几点：

+   High 是该特定日期股票的最高交易价格。

+   Low 是该特定日期股票的最低交易价格。

+   Open 是该日期开始时股票的价格。

+   Close 是该日期市场收盘时的股票价格。

+   Volume 是该特定股票的交易量，即交易的实际股票数量。

+   Adj Close 是市场关闭后股票的价格。

# 世界银行

世界银行是一个提供财务咨询并在经济状况方面帮助各国的组织。它还提供各种数据，包括时间序列、地理空间数据、财务数据等，这些都对分析非常有帮助。

在开始从世界银行网站获取数据之前，我们必须先注册。这允许我们获得我们想要下载的数据集的指标代码。注册世界银行是免费的，且不会占用太多时间。

出于本示例的目的，我使用了世界发展指标数据集。你可以选择任何你喜欢的数据集并开始使用：

```py
from pandas_datareader import wb  
```

要获取指标代码，请选择 Databank 标签页，并从左侧面板中显示的列表中选择 Metadata Glossary。你可以在每个数据集下方的左侧面板找到该指标（在附图中用红色标出）：

![](img/cbf5065e-9268-4764-aff2-898397b13765.png)

数据集的指标代码，用于获取数据

```py
dat = wb.download(indicator='FP.CPI.TOTL.ZG', start=2005, end=2019)  
```

返回的 DataFrame 采用多重索引行格式：

```py
dat.head()  
```

这将得到以下输出：

![](img/d8bb985a-5374-4ce0-a358-e528b1090bd5.png)

世界银行指标数据的多重索引 DataFrame 输出

现在，我们来展示一个特定国家的数据，如下所示：

```py
    dat.loc['Canada']

```

这将得到以下输出：

![](img/35220b81-cb9a-4a59-a420-71a2dcadc30c.png)

世界银行指标数据的多重索引 DataFrame 输出，针对一个国家。它变为单重索引。

我们还可以仅返回一个特定年份的价格通货膨胀数据，如下所示：

```py
    dat.loc['Canada'].loc['2015']

```

这将得到以下输出：

![](img/40efc2f1-f406-4a72-aa41-2ec6a597c839.png)

按两个索引子集化的数据

# 总结

阅读完本章后，观察到以下几点：

+   `pandas` 提供了强大的方法，让我们可以从各种数据结构和多个数据源读取和写入数据。

+   pandas 中的 `read_csv` 方法可以用来读取 CSV 文件、TXT 文件和表格。此方法有许多参数，用于指定分隔符、跳过的行、分块读取文件等。

+   pandas 可以用来直接从 URL 或 S3 中读取数据。

+   DataFrame 可以转换为 JSON，反之亦然。JSON 可以存储在文本文件中，并且可以读取。

+   JSON 具有类似字典的结构，可以无限次嵌套。这些嵌套数据可以像字典一样通过键进行子集提取。

+   Pandas 提供了方法，允许我们从 HD5、HTML、SAS、SQL、parquet、feather 和 Google BigQuery 等数据格式中读取数据。

+   序列化有助于将数据结构或对象转储到物理文件中、存储到数据库，或通过消息进行传输。

在下一章中，我们将学习如何访问和选择 pandas 数据结构中的数据。我们还将详细讨论基本索引、标签索引、整数索引和混合索引。
