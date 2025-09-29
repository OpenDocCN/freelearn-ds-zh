# 第六章。增强版本

传统上，更改通常不受欢迎，并且关系型数据库开发人员会尽可能地避免更改。然而，业务每天都在变化，尤其是在当前这个快节奏的时代。使用关系型数据库的系统对业务变化的延迟响应会降低企业的敏捷性，甚至威胁到企业的生存。随着 NoSQL 和其他相关技术的发展，我们现在有替代方案来拥抱这样的业务变化。

通过继续增强在第五章“初步设计和实现”中开发的股票筛选器应用程序，将详细解释如何演进现有的 Cassandra 数据模型的技术。同时，还将展示查询建模技术。然后，将相应地修改股票筛选器应用程序的源代码。到本章结束时，将开发一个完整的股票技术分析应用程序。您可以用它作为快速开发您自己的基础。

# 优化数据模型

在第五章中创建的股票筛选器应用程序，*初步设计和实现*，足以一次检索和分析一只股票。然而，在实际应用中，仅扫描一只股票看起来非常有限。这里可以稍作改进；它可以处理一组股票而不是单个股票。这组股票将被存储在 Cassandra 数据库中的观察名单中。

因此，股票筛选器应用程序将被修改以分析观察名单中的股票，因此它将根据相同的筛选规则为每个被观察的股票生成警报。

对于产生的警报，将它们保存在 Cassandra 中将有利于回测交易策略和持续改进股票筛选器应用程序。它们可以不时地被审查，而无需即时审查。

### 注意

回测是一个术语，用于指用现有历史数据测试交易策略、投资策略或预测模型。它也是应用于时间序列数据的一种特殊类型的交叉验证。

此外，当观察名单中的股票数量增长到几百只时，股票筛选器应用程序的用户将很难仅通过参考它们的股票代码来回忆起这些股票。因此，最好将股票名称添加到生成的警报中，使它们更具描述性和用户友好性。

最后，我们可能对找出在特定时间段内特定股票上生成的警报数量以及特定日期上生成的警报数量感兴趣。我们将使用 CQL 编写查询来回答这两个问题。通过这样做，可以展示查询建模技术。

## 增强方法

增强方法总共包括四个变更请求。首先，我们将对数据模型进行更改，然后代码将增强以提供新功能。之后，我们将再次测试运行增强的股票筛选应用程序。以下图中突出显示了需要修改的股票筛选应用程序的部分。

值得注意的是，股票筛选应用程序中增加了两个新组件。第一个组件是 **观察列表**，它管理 **数据映射器和存档器**，以从 Yahoo! Finance 收集观察列表中股票的股票报价数据。第二个组件是 **查询**。它提供了两个针对 **警报列表**的查询，用于回测目的：

![增强方法](img/8884OS_06_01.jpg)

### 观察列表

**观察列表**是一个非常简单的表，仅存储其组成部分的股票代码。对于关系型数据库开发者来说，将股票代码定义为主键是非常直观的，对吧？然而，请记住，在 Cassandra 中，主键用于确定存储行的节点。由于观察列表预计不会非常长，将其所有行放在同一个节点上以实现更快的检索会更合适。但我们应该如何做到这一点呢？

我们可以为此特定目的创建一个额外的列，例如 `watch_list_code`。新表称为 `watchlist`，并将创建在 `packtcdma` 键空间中。CQL 语句显示在 `chapter06_001.py` 中：

```py
# -*- coding: utf-8 -*-
# program: chapter06_001.py

## import Cassandra driver library
from cassandra.cluster import Cluster

## function to create watchlist
def create_watchlist(ss):
    ## create watchlist table if not exists
    ss.execute('CREATE TABLE IF NOT EXISTS watchlist (' + \
               'watch_list_code varchar,' + \
               'symbol varchar,' + \
               'PRIMARY KEY (watch_list_code, symbol))')

    ## insert AAPL, AMZN, and GS into watchlist
    ss.execute("INSERT INTO watchlist (watch_list_code, " + \
               "symbol) VALUES ('WS01', 'AAPL')")
    ss.execute("INSERT INTO watchlist (watch_list_code, " + \
               "symbol) VALUES ('WS01', 'AMZN')")
    ss.execute("INSERT INTO watchlist (watch_list_code, " + \
               "symbol) VALUES ('WS01', 'GS')")

## create Cassandra instance
cluster = Cluster()

## establish Cassandra connection, using local default
session = cluster.connect()

## use packtcdma keyspace
session.set_keyspace('packtcdma')

## create watchlist table
create_watchlist(session)

## close Cassandra connection
cluster.shutdown()
```

`create_watchlist` 函数创建表。请注意，`watchlist` 表由 `watch_list_code` 和 `symbol` 组成的复合主键构成。还创建了一个名为 `WS01` 的观察列表，其中包含三只股票，`AAPL`、`AMZN` 和 `GS`。

### 警报列表

在 第五章，*初步设计和实现* 中，**警报列表**非常基础。它由一个 Python 程序生成，列出了收盘价高于其 10 日简单移动平均线的日期，即当时的信号和收盘价。请注意，当时没有股票代码和股票名称。

我们将创建一个名为 `alertlist` 的表来存储警报，包括股票的代码和名称。包含股票名称是为了满足使股票筛选应用程序更用户友好的要求。同时，请记住，不允许使用连接，并且在 Cassandra 中去规范化确实是最佳实践。这意味着我们不会介意在将要查询的表中重复存储（复制）股票名称。一个经验法则是 *一个表对应一个查询*；就这么简单。

`alertlist` 表是通过 CQL 语句创建的，如 `chapter06_002.py` 中所示：

```py
# -*- coding: utf-8 -*-
# program: chapter06_002.py

## import Cassandra driver library
from cassandra.cluster import Cluster

## function to create alertlist
def create_alertlist(ss):
    ## execute CQL statement to create alertlist table if not exists
    ss.execute('CREATE TABLE IF NOT EXISTS alertlist (' + \
               'symbol varchar,' + \
               'price_time timestamp,' + \
               'stock_name varchar,' + \
               'signal_price float,' + \
               'PRIMARY KEY (symbol, price_time))')

## create Cassandra instance
cluster = Cluster()

## establish Cassandra connection, using local default
session = cluster.connect()

## use packtcdma keyspace
session.set_keyspace('packtcdma')

## create alertlist table
create_alertlist(session)

## close Cassandra connection
cluster.shutdown()
```

主键也是一个复合主键，由 `symbol` 和 `price_time` 组成。

### 添加描述性股票名称

到目前为止，`packtcdma`键空间有三个表，分别是`alertlist`、`quote`和`watchlist`。为了添加描述性的股票名称，人们可能会想到只向`alertlist`添加一个股票名称列。如前所述，这已经完成了。那么，我们是否需要为`quote`和`watchlist`添加列？

实际上，这是一个设计决策，取决于这两个表是否将用于处理用户查询。用户查询的含义是，该表将用于检索用户提出的查询所需的行。如果用户想知道 2014 年 6 月 30 日苹果公司的收盘价，这便是一个用户查询。另一方面，如果股票筛选应用程序使用查询来检索其内部处理的行，那么这便不是用户查询。因此，如果我们想让`quote`和`watchlist`为用户查询返回行，它们就需要包含股票名称列；否则，它们不需要。

`watchlist`表仅用于当前设计的内部使用，因此它不需要包含股票名称列。当然，如果将来股票筛选应用程序允许用户维护股票观察列表，那么股票名称也应该添加到`watchlist`表中。

然而，对于`quote`来说，这有点棘手。因为股票名称应该从数据提供者那里检索，在我们的案例中是雅虎财经，最适合获取股票名称的时间是在检索相应的股票报价数据时。因此，在`quote`中添加了一个名为`stock_name`的新列，如`chapter06_003.py`所示：

```py
# -*- coding: utf-8 -*-
# program: chapter06_003.py

## import Cassandra driver library
from cassandra.cluster import Cluster

## function to add stock_name column
def add_stockname_to_quote(ss):
    ## add stock_name to quote
    ss.execute('ALTER TABLE quote ' + \
               'ADD stock_name varchar')

## create Cassandra instance
cluster = Cluster()

## establish Cassandra connection, using local default
session = cluster.connect()

## use packtcdma keyspace
session.set_keyspace('packtcdma')

## add stock_name column
add_stockname_to_quote(session)

## close Cassandra connection
cluster.shutdown()
```

这相当直观。在这里，我们使用`ALTER TABLE`语句向`quote`添加了`varchar`数据类型的`stock_name`列。

### 警报查询

如前所述，我们感兴趣的是两个问题：

+   在指定时间段内，针对某只股票生成了多少个警报？

+   在特定日期上生成了多少个警报？

对于第一个问题，`alertlist`足以提供答案。然而，`alertlist`无法回答第二个问题，因为它的主键由`symbol`和`price_time`组成。我们需要为这个问题创建另一个特定的表。这是一个通过查询建模的例子。

基本上，第二个问题的新表结构应该类似于`alertlist`的结构。我们给这个表起了一个名字，`alert_by_date`，并在`chapter06_004.py`中创建了它：

```py
# -*- coding: utf-8 -*-
# program: chapter06_004.py

## import Cassandra driver library
from cassandra.cluster import Cluster

## function to create alert_by_date table
def create_alertbydate(ss):
    ## create alert_by_date table if not exists
    ss.execute('CREATE TABLE IF NOT EXISTS alert_by_date (' + \
               'symbol varchar,' + \
               'price_time timestamp,' + \
               'stock_name varchar,' + \
               'signal_price float,' + \
               'PRIMARY KEY (price_time, symbol))')

## create Cassandra instance
cluster = Cluster()

## establish Cassandra connection, using local default
session = cluster.connect()

## use packtcdma keyspace
session.set_keyspace('packtcdma')

## create alert_by_date table
create_alertbydate(session)

## close Cassandra connection
cluster.shutdown()
```

与`chapter06_002.py`中的`alertlist`相比，`alert_by_date`只是在复合主键中交换了列的顺序。有人可能会认为可以在`alertlist`上创建一个二级索引来实现相同的效果。然而，在 Cassandra 中，不能在已经参与主键的列上创建二级索引。始终要注意这个限制。

我们现在完成了数据模型的修改。接下来，我们需要增强下一节中的应用逻辑。

# 代码增强

关于要纳入股票筛选应用程序的新要求，已创建观察列表，我们将在本节中继续实现剩余更改的代码。

## 数据映射器和归档器

数据映射器和归档器是数据馈送提供器模块的组件，其源代码文件是`chapter05_005.py`。大部分源代码可以保持不变；我们只需要添加代码到：

1.  为观察列表代码加载观察列表并检索基于该列表的数据馈送

1.  检索股票名称并将其存储在报价表中

修改后的源代码显示在`chapter06_005.py`中：

```py
# -*- coding: utf-8 -*-
# program: chapter06_005.py

## import Cassandra driver library
from cassandra.cluster import Cluster
from decimal import *

## web is the shorthand alias of pandas.io.data
import pandas.io.data as web
import datetime

## import BeautifulSoup and requests
from bs4 import BeautifulSoup
import requests

## function to insert historical data into table quote
## ss: Cassandra session
## sym: stock symbol
## d: standardized DataFrame containing historical data
## sn: stock name
def insert_quote(ss, sym, d, sn):
    ## CQL to insert data, ? is the placeholder for parameters
    insert_cql = "INSERT INTO quote (" + \
                 "symbol, price_time, open_price, high_price," + \
                 "low_price, close_price, volume, stock_name" + \
                 ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    ## prepare the insert CQL as it will run repeatedly
    insert_stmt = ss.prepare(insert_cql)

    ## set decimal places to 4 digits
    getcontext().prec = 4

    ## loop thru the DataFrame and insert records
    for index, row in d.iterrows():
        ss.execute(insert_stmt, \
                   [sym, index, \
                   Decimal(row['open_price']), \
                   Decimal(row['high_price']), \
                   Decimal(row['low_price']), \
                   Decimal(row['close_price']), \
                   Decimal(row['volume']), \
                   sn])
```

在这里，我们将`INSERT`语句修改为在`insert_quote`函数中将股票名称存储到`quote`中。然后我们添加一个名为`load_watchlist`的函数：

```py
## retrieve the historical daily stock quote from Yahoo! Finance
## Parameters
## sym: stock symbol
## sd: start date
## ed: end date
def collect_data(sym, sd, ed):
    ## data is a DataFrame holding the daily stock quote
    data = web.DataReader(sym, 'yahoo', sd, ed)
    return data

## transform received data into standardized format
## Parameter
## d: DataFrame containing Yahoo! Finance stock quote
def transform_yahoo(d):
    ## drop extra column 'Adj Close'
    d1 = d.drop(['Adj Close'], axis=1)

    ## standardize the column names
    ## rename index column to price_date
    d1.index.names=['price_date']

    ## rename the columns to match the respective columns
    d1 = d1.rename(columns={'Open':'open_price', \
                            'High':'high_price', \
                            'Low':'low_price', \
                            'Close':'close_price', \
                            'Volume':'volume'})
    return d1

## function to retrieve watchlist
## ss: Cassandra session
## ws: watchlist code
def load_watchlist(ss, ws):
    ## CQL to select data, ? is the placeholder for parameters
    select_cql = "SELECT symbol FROM watchlist " + \
                 "WHERE watch_list_code=?"

    ## prepare select CQL
    select_stmt = ss.prepare(select_cql)

    ## execute the select CQL
    result = ss.execute(select_stmt, [ws])

    ## initialize the stock array
    stw = []

    ## loop thru the query resultset to make up the DataFrame
    for r in result:
        stw.append(r.symbol)

    return stw
```

在这里，新函数`load_watchlist`对`watch_list`执行`SELECT`查询，以检索特定观察列表代码的观察股票；然后它返回一个`symbol`列表：

```py
## function to retrieve stock name from Yahoo!Finance
## sym: stock symbol
def get_stock_name(sym):
  url = 'http://finance.yahoo.com/q/hp?s=' + sym + \
  '+Historical+Prices'
  r = requests.get(url)
  soup = BeautifulSoup(r.text)
  data = soup.findAll('h2')
  return data[2].text

def testcase001():
    ## create Cassandra instance
    cluster = Cluster()

    ## establish Cassandra connection, using local default
    session = cluster.connect('packtcdma')

    start_date = datetime.datetime(2012, 1, 1)
    end_date = datetime.datetime(2014, 6, 28)

    ## load the watchlist
    stocks_watched = load_watchlist(session, "WS01")

    ## iterate the watchlist
    for symbol in stocks_watched:
        ## get stock name
        stock_name = get_stock_name(symbol)

        ## collect data
        data = collect_data(symbol, start_date, end_date)

        ## transform Yahoo! Finance data
        data = transform_yahoo(data)

        ## insert historical data
        insert_quote(session, symbol, data, stock_name)

    ## close Cassandra connection
    cluster.shutdown()

testcase001()
```

这里的更改是一个名为`get_stock_name`的新函数，它向 Yahoo! Finance 发送一个网络服务请求，并从返回的 HTML 页面中提取股票名称。我们使用一个名为`BeautifulSoup`的 Python 包来使从 HTML 页面中提取元素变得非常方便。然后`get_stock_name`函数返回股票名称。

### 注意

`BeautifulSoup`是一个为快速周转项目如屏幕抓取而设计的库。它主要解析它所接收的任何文本，并通过解析文本的树遍历找到所需的内容。更多信息可以在[`www.crummy.com/software/BeautifulSoup/`](http://www.crummy.com/software/BeautifulSoup/)找到。

使用`for`循环遍历观察列表以检索股票名称和股票报价数据。此外，由于我们需要将股票名称存储在`quote`表中，`insert_quote`函数接受股票名称作为新参数，并相应地对`INSERT`语句和`for`循环进行少量修改。

那就是关于数据映射器和归档器更改的全部内容。

### 股票筛选引擎

我们将使用第五章中 Stock Screener Engine 的源代码，*初步设计和实现*来包含增强功能；为此，我们将执行以下操作：

1.  与数据映射器和归档器类似，我们将为观察列表代码加载观察列表并在每个股票上扫描警报。

1.  从报价表中的股票名称列检索股票报价数据。

1.  将警报保存到`alertlist`中。

修改后的源代码显示在`chapter06_006.py`中：

```py
# -*- coding: utf-8 -*-
# program: chapter06_006.py

## import Cassandra driver library
from cassandra.cluster import Cluster

import pandas as pd
import numpy as np
import datetime

## import Cassandra BatchStatement library
from cassandra.query import BatchStatement
from decimal import *

## function to insert historical data into table quote
## ss: Cassandra session
## sym: stock symbol
## sd: start date
## ed: end date
## return a DataFrame of stock quote
def retrieve_data(ss, sym, sd, ed):
    ## CQL to select data, ? is the placeholder for parameters
    select_cql = "SELECT * FROM quote WHERE symbol=? " + \
                 "AND price_time >= ? AND price_time <= ?"

    ## prepare select CQL
    select_stmt = ss.prepare(select_cql)

    ## execute the select CQL
    result = ss.execute(select_stmt, [sym, sd, ed])

    ## initialize an index array
    idx = np.asarray([])

    ## initialize an array for columns
    cols = np.asarray([])

    ## loop thru the query resultset to make up the DataFrame
    for r in result:
        idx = np.append(idx, [r.price_time])
        cols = np.append(cols, [r.open_price, r.high_price, \
                         r.low_price, r.close_price, \
                         r.volume, r.stock_name])

    ## reshape the 1-D array into a 2-D array for each day
    cols = cols.reshape(idx.shape[0], 6)

    ## convert the arrays into a pandas DataFrame
    df = pd.DataFrame(cols, index=idx, \
                      columns=['open_price', 'high_price', \
                      'low_price', 'close_price', \
                      'volume', 'stock_name'])
    return df
```

由于我们已经将股票名称包含在查询结果集中，我们需要修改`retrieve_data`函数中的`SELECT`语句：

```py
## function to compute a Simple Moving Average on a DataFrame
## d: DataFrame
## prd: period of SMA
## return a DataFrame with an additional column of SMA
def sma(d, prd):
    d['sma'] = pd.rolling_mean(d.close_price, prd)
    return d

## function to apply screening rule to generate buy signals
## screening rule, Close > 10-Day SMA
## d: DataFrame
## return a DataFrame containing buy signals
def signal_close_higher_than_sma10(d):
    return d[d.close_price > d.sma]

## function to retrieve watchlist
## ss: Cassandra session
## ws: watchlist code
def load_watchlist(ss, ws):
    ## CQL to select data, ? is the placeholder for parameters
    select_cql = "SELECT symbol FROM watchlist " + \
                 "WHERE watch_list_code=?"

    ## prepare select CQL
    select_stmt = ss.prepare(select_cql)

    ## execute the select CQL
    result = ss.execute(select_stmt, [ws])

    ## initialize the stock array
    stw = []

    ## loop thru the query resultset to make up the DataFrame
    for r in result:
        stw.append(r.symbol)

    return stw

## function to insert historical data into table quote
## ss: Cassandra session
## sym: stock symbol
## d: standardized DataFrame containing historical data
## sn: stock name
def insert_alert(ss, sym, sd, cp, sn):
    ## CQL to insert data, ? is the placeholder for parameters
    insert_cql1 = "INSERT INTO alertlist (" + \
                 "symbol, price_time, signal_price, stock_name" +\
                 ") VALUES (?, ?, ?, ?)"

    ## CQL to insert data, ? is the placeholder for parameters
    insert_cql2 = "INSERT INTO alert_by_date (" + \
                 "symbol, price_time, signal_price, stock_name" +\
                 ") VALUES (?, ?, ?, ?)"

    ## prepare the insert CQL as it will run repeatedly
    insert_stmt1 = ss.prepare(insert_cql1)
    insert_stmt2 = ss.prepare(insert_cql2)

    ## set decimal places to 4 digits
    getcontext().prec = 4

    ## begin a batch
    batch = BatchStatement()

    ## add insert statements into the batch
    batch.add(insert_stmt1, [sym, sd, cp, sn])
    batch.add(insert_stmt2, [sym, sd, cp, sn])

    ## execute the batch
    ss.execute(batch)

def testcase002():
    ## create Cassandra instance
    cluster = Cluster()

    ## establish Cassandra connection, using local default
    session = cluster.connect('packtcdma')

    start_date = datetime.datetime(2012, 6, 28)
    end_date = datetime.datetime(2012, 7, 28)

    ## load the watch list
    stocks_watched = load_watchlist(session, "WS01")

    for symbol in stocks_watched:
        ## retrieve data
        data = retrieve_data(session, symbol, start_date, end_date)

        ## compute 10-Day SMA
        data = sma(data, 10)

        ## generate the buy-and-hold signals
        alerts = signal_close_higher_than_sma10(data)

        ## save the alert list
        for index, r in alerts.iterrows():
            insert_alert(session, symbol, index, \
                         Decimal(r['close_price']), \
                         r['stock_name'])

    ## close Cassandra connection
    cluster.shutdown()

testcase002()
```

在 `chapter06_006.py` 的底部，`for` 循环负责迭代由新的 `load_watchlist` 函数加载的 `watchlist`，这个函数与 `chapter06_005.py` 中的函数相同，不需要进一步解释。另一个 `for` 循环内部通过调用新的 `insert_alert` 函数将扫描到的警报保存到 `alertlist`。

在解释 `insert_alert` 函数之前，让我们跳转到顶部的 `retrieve_data` 函数。`retrieve_data` 函数被修改为同时返回股票名称，因此 `cols` 变量现在包含六个列。向下滚动一点到 `insert_alert`。

如其名称所示，`insert_alert` 函数将警报保存到 `alertlist` 和 `alert_by_date`。它为这两个表分别有两个 `INSERT` 语句。这两个 `INSERT` 语句几乎完全相同，只是表名不同。显然，它们是重复的，这就是反规范化的含义。我们在这里还应用了 Cassandra 2.0 的新特性，称为 *批处理*。批处理将多个 **数据修改语言** (**DML**) 语句组合成一个单一的逻辑、原子操作。DataStax 的 Cassandra Python 驱动程序通过 `BatchStatement` 包支持此功能。我们通过调用 `BatchStatement()` 函数创建一个批处理，然后将准备好的 `INSERT` 语句添加到批处理中，最后执行它。如果在提交过程中任一 `INSERT` 语句遇到错误，批处理中的所有 DML 语句将不会执行。因此，这与关系型数据库中的事务类似。

### 警报查询

Stock Screener 应用程序的最后一次修改是关于警报的查询功能，这些功能对回测和性能测量很有用。我们编写了两个查询来回答两个问题，如下所示：

+   在指定时间段内，一只股票产生了多少个警报？

+   在特定日期上产生了多少个警报？

由于我们在数据模型上使用了反规范化，因此执行起来非常容易。对于第一次查询，请参阅 `chapter06_007.py`：

```py
# -*- coding: utf-8 -*-
# program: chapter06_007.py

## import Cassandra driver library
from cassandra.cluster import Cluster

import pandas as pd
import numpy as np
import datetime

## execute CQL statement to retrieve rows of
## How many alerts were generated on a particular stock over
## a specified period of time?
def alert_over_daterange(ss, sym, sd, ed):
    ## CQL to select data, ? is the placeholder for parameters
    select_cql = "SELECT * FROM alertlist WHERE symbol=? " + \
                 "AND price_time >= ? AND price_time <= ?"

    ## prepare select CQL
    select_stmt = ss.prepare(select_cql)

    ## execute the select CQL
    result = ss.execute(select_stmt, [sym, sd, ed])

     ## initialize an index array
    idx = np.asarray([])

    ## initialize an array for columns
    cols = np.asarray([])

    ## loop thru the query resultset to make up the DataFrame
    for r in result:
        idx = np.append(idx, [r.price_time])
        cols = np.append(cols, [r.symbol, r.stock_name, \
                         r.signal_price])

    ## reshape the 1-D array into a 2-D array for each day
    cols = cols.reshape(idx.shape[0], 3)

    ## convert the arrays into a pandas DataFrame
    df = pd.DataFrame(cols, index=idx, \
                      columns=['symbol', 'stock_name', \
                      'signal_price'])
    return df

def testcase001():
    ## create Cassandra instance
    cluster = Cluster()

    ## establish Cassandra connection, using local default
    session = cluster.connect()

    ## use packtcdma keyspace
    session.set_keyspace('packtcdma')

    ## scan buy-and-hold signals for GS
    ## over 1 month since 28-Jun-2012
    symbol = 'GS'
    start_date = datetime.datetime(2012, 6, 28)
    end_date = datetime.datetime(2012, 7, 28)

    ## retrieve alerts
    alerts = alert_over_daterange(session, symbol, \
                                  start_date, end_date)

    for index, r in alerts.iterrows():
        print index.date(), '\t', \
            r['symbol'], '\t', \
            r['stock_name'], '\t', \
            r['signal_price']

    ## close Cassandra connection
    cluster.shutdown()

testcase001()
```

定义了一个名为 `alert_over_daterange` 的函数来检索与第一个问题相关的行，然后将其转换为 pandas DataFrame。

然后，我们可以根据 `chapter06_007.py` 中的相同逻辑来为第二个问题编写查询。源代码显示在 `chapter06_008.py`：

```py
# -*- coding: utf-8 -*-
# program: chapter06_008.py

## import Cassandra driver library
from cassandra.cluster import Cluster

import pandas as pd
import numpy as np
import datetime

## execute CQL statement to retrieve rows of
## How many alerts were generated on a particular stock over
## a specified period of time?
def alert_on_date(ss, dd):
    ## CQL to select data, ? is the placeholder for parameters
    select_cql = "SELECT * FROM alert_by_date WHERE " + \
                 "price_time=?"

    ## prepare select CQL
    select_stmt = ss.prepare(select_cql)

    ## execute the select CQL
    result = ss.execute(select_stmt, [dd])

     ## initialize an index array
    idx = np.asarray([])

    ## initialize an array for columns
    cols = np.asarray([])

    ## loop thru the query resultset to make up the DataFrame
    for r in result:
        idx = np.append(idx, [r.symbol])
        cols = np.append(cols, [r.stock_name, r.price_time, \
                         r.signal_price])

    ## reshape the 1-D array into a 2-D array for each day
    cols = cols.reshape(idx.shape[0], 3)

    ## convert the arrays into a pandas DataFrame
    df = pd.DataFrame(cols, index=idx, \
                      columns=['stock_name', 'price_time', \
                      'signal_price'])
    return df

def testcase001():
    ## create Cassandra instance
    cluster = Cluster()

    ## establish Cassandra connection, using local default
    session = cluster.connect()

    ## use packtcdma keyspace
    session.set_keyspace('packtcdma')

    ## scan buy-and-hold signals for GS over 1 month since 28-Jun-2012
    on_date = datetime.datetime(2012, 7, 13)

    ## retrieve alerts
    alerts = alert_on_date(session, on_date)

    ## print out alerts
    for index, r in alerts.iterrows():
        print index, '\t', \
              r['stock_name'], '\t', \
              r['signal_price']

    ## close Cassandra connection
    cluster.shutdown()

testcase001()
```

再次强调，反规范化是 Cassandra 的朋友。它不需要外键、引用完整性或表连接。

# 实现系统更改

我们现在可以逐个实现系统更改：

1.  首先，我们按顺序运行 `chapter06_001.py` 到 `chapter06_004.py`，以对数据模型进行修改。

1.  然后，我们执行`chapter06_005.py`以检索观察列表的股票报价数据。值得一提的是，UPSERT 是 Cassandra 的一个非常好的特性。当我们向表中插入相同的行时，我们不会遇到重复的主键。如果行已存在，它将简单地更新该行，否则将插入该行。这使得数据操作逻辑变得整洁且清晰。

1.  此外，我们运行`chatper06_006.py`通过扫描观察列表中每只股票的股票报价数据来存储警报。

1.  最后，我们执行`chapter06_007.py`和`chapter06_008.py`来查询`alertlist`和`alert_by_date`，其样本测试结果如下：![实施系统更改](img/8884OS_06_02.jpg)

# 摘要

本章通过一系列增强扩展了股票筛选应用程序。我们对数据模型进行了修改，以展示通过查询技术建模以及非规范化如何帮助我们实现高性能应用程序。我们还尝试了 Cassandra 2.0 提供的批量功能。

注意，本章中的源代码未经整理，可以进行某种重构。然而，由于页面数量的限制，它被留作读者的练习。

股票筛选应用程序现在运行在单个节点集群上。

在下一章中，我们将深入探讨将其扩展到更大集群的考虑和流程，这在现实生活中的生产系统中相当常见。
