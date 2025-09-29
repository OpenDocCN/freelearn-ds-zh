# 第三章：使用 Spark 处理数据

根据上一章中概述的批处理和流式架构，我们需要数据来为我们的应用程序提供动力。我们将从 Twitter 收集专注于 Apache Spark 的数据。本章的目标是为机器学习和流式应用程序准备数据。本章重点介绍如何在分布式网络中交换代码和数据。我们将获得关于序列化、持久化、打包和缓存的实用见解。我们将掌握 Spark SQL，这是 Spark 中用于交互式探索结构化和半结构化数据的关键模块。驱动 Spark SQL 的基本数据结构是 Spark 数据框。Spark 数据框受到 Python Pandas 数据框和 R 数据框的启发。它是一个强大的数据结构，被具有 R 或 Python 背景的数据科学家广泛理解和欣赏。

在本章中，我们将涵盖以下内容：

+   连接到 Twitter，收集相关数据，并将其以 JSON 和 CSV 等格式以及 MongoDB 等数据存储方式持久化

+   使用 Blaze 和 Odo（Blaze 的一个衍生库）分析数据，以便从各种来源和目的地连接和传输数据

+   介绍 Spark 数据框作为 Spark 模块之间数据交换的基础，并使用 Spark SQL 交互式地探索数据

# 重新审视数据密集型应用架构

让我们首先从数据密集型应用架构的角度来界定本章的重点。我们将集中注意力于集成层，并基本上运行数据获取、精炼和持久化的迭代周期。这个周期被称为五个 C，即*连接*、*收集*、*纠正*、*组合*和*消费*。它们是我们为了从 Twitter 获取正确质量和数量的数据而在集成层运行的基本过程。我们还将深入了解持久化层，并设置一个如 MongoDB 这样的数据存储来收集我们的数据以便后续处理。

我们将使用 Blaze（一个用于数据操作的 Python 库）和 Spark SQL（Spark 的交互式数据发现模块，由 Spark 数据框驱动）探索数据。数据框范式由 Python Pandas、Python Blaze 和 Spark SQL 共享。我们将了解三种数据框风味的细微差别。

以下图表设置了本章重点的上下文，突出了集成层和持久化层：

![重新审视数据密集型应用架构](img/B03986_03_01.jpg)

# 序列化和反序列化数据

由于我们在速率限制约束下从 Web API 收集数据，我们需要存储它们。由于数据在分布式集群上处理，我们需要一致的方式来保存状态并在以后使用时检索它。

现在让我们定义序列化、持久化、打包和缓存或记忆化。

序列化 Python 对象将其转换为字节流。Python 对象需要在程序关闭时从其存在范围之外检索。序列化的 Python 对象可以通过网络传输或存储在持久化存储中。反序列化是相反的过程，它将字节流转换为原始 Python 对象，以便程序可以从保存的状态继续执行。Python 中最流行的序列化库是 Pickle。实际上，PySpark 命令是通过序列化数据通过网络传输到工作节点的。

持久化将程序的状态数据保存到磁盘或内存中，以便在重启时从上次停止的地方继续。它将 Python 对象从内存保存到文件或数据库，并在稍后以相同的状态加载它。

Marshalling 在多核或分布式系统中通过 TCP 连接在网络中发送 Python 代码或数据。

缓存将 Python 对象转换为内存中的字符串，以便以后用作字典键。Spark 支持将数据集拉入集群范围内的内存缓存。当数据被重复访问时，例如查询小型参考数据集或运行迭代算法（如 Google PageRank）时，这非常有用。

缓存对于 Spark 来说是一个关键概念，因为它允许我们将 RDDs 存储在内存中或通过溢出到磁盘。可以根据数据的血缘关系或应用于 RDDs 的**DAG**（代表**有向无环图**）来选择缓存策略，以最小化洗牌或跨网络重数据交换。为了在 Spark 中获得良好的性能，请注意数据洗牌。良好的分区策略和 RDD 缓存的合理使用，以及避免不必要的行动操作，可以带来更好的 Spark 性能。

# 收集和存储数据

在深入研究数据库持久化存储，如 MongoDB 之前，我们将探讨一些广泛使用的有用文件存储：**CSV**（代表**逗号分隔值**）和**JSON**（代表**JavaScript 对象表示法**）文件存储。这两种文件格式的持久流行有几个关键原因：它们是可读的，简单，相对轻量级，且易于使用。

## 在 CSV 中持久化数据

CSV 格式轻量级，可读，易于使用。它具有带固有表格模式的分隔文本列。

Python 提供了一个强大的`csv`库，可以将`csv`文件序列化为 Python 字典。为了我们程序的目的，我们已经编写了一个`python`类，该类能够以 CSV 格式持久化数据并从给定的 CSV 中读取。

让我们运行`IO_csv`对象类的代码。类的`__init__`部分基本上实例化了文件路径、文件名和文件后缀（在这种情况下，为`.csv`）：

```py
class IO_csv(object):

    def __init__(self, filepath, filename, filesuffix='csv'):
        self.filepath = filepath       # /path/to/file without the /' at the end
        self.filename = filename       # FILE_NAME
        self.filesuffix = filesuffix
```

类的`save`方法使用 Python 命名元组和`csv`文件的标题字段来在持久化 CSV 行的同时赋予一个模式。如果`csv`文件已经存在，它将被追加而不是覆盖；否则，它将被创建：

```py
    def save(self, data, NTname, fields):
        # NTname = Name of the NamedTuple
        # fields = header of CSV - list of the fields name
        NTuple = namedtuple(NTname, fields)

        if os.path.isfile('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix)):
            # Append existing file
            with open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), 'ab') as f:
                writer = csv.writer(f)
                # writer.writerow(fields) # fields = header of CSV
                writer.writerows([row for row in map(NTuple._make, data)])
                # list comprehension using map on the NamedTuple._make() iterable and the data file to be saved
                # Notice writer.writerows and not writer.writerow (i.e. list of multiple rows sent to csv file
        else:
            # Create new file
            with open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(fields) # fields = header of CSV - list of the fields name
                writer.writerows([row for row in map(NTuple._make, data)])
                #  list comprehension using map on the NamedTuple._make() iterable and the data file to be saved
                # Notice writer.writerows and not writer.writerow (i.e. list of multiple rows sent to csv file
```

类的`load`方法也使用 Python 命名元组和`csv`文件的标题字段来使用一致的架构检索数据。`load`方法是一个内存高效的生成器，以避免在内存中加载一个巨大的文件：因此我们使用`yield`代替`return`：

```py
    def load(self, NTname, fields):
        # NTname = Name of the NamedTuple
        # fields = header of CSV - list of the fields name
        NTuple = namedtuple(NTname, fields)
        with open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix),'rU') as f:
            reader = csv.reader(f)
            for row in map(NTuple._make, reader):
                # Using map on the NamedTuple._make() iterable and the reader file to be loaded
                yield row 
```

这里是命名元组。我们使用它来解析推文，以便将其保存或从`csv`文件中检索：

```py
fields01 = ['id', 'created_at', 'user_id', 'user_name', 'tweet_text', 'url']
Tweet01 = namedtuple('Tweet01',fields01)

def parse_tweet(data):
    """
    Parse a ``tweet`` from the given response data.
    """
    return Tweet01(
        id=data.get('id', None),
        created_at=data.get('created_at', None),
        user_id=data.get('user_id', None),
        user_name=data.get('user_name', None),
        tweet_text=data.get('tweet_text', None),
        url=data.get('url')
    )
```

## 在 JSON 中持久化数据

JSON 是互联网应用程序中最流行的数据格式之一。我们处理的所有 API，包括 Twitter、GitHub 和 Meetup，都以 JSON 格式提供数据。与相对较重的 XML 相比，JSON 格式相对轻量级，且易于阅读，其模式嵌入在 JSON 中。与所有记录都遵循完全相同的表格结构的 CSV 格式不同，JSON 记录的结构可以不同。JSON 是半结构化的。一个 JSON 记录可以被映射到一个 Python 字典的字典。

让我们遍历`IO_json`类对象的代码。类的`__init__`部分基本上实例化了文件路径、文件名和文件后缀（在这种情况下，是`.json`）：

```py
class IO_json(object):
    def __init__(self, filepath, filename, filesuffix='json'):
        self.filepath = filepath        # /path/to/file without the /' at the end
        self.filename = filename        # FILE_NAME
        self.filesuffix = filesuffix
        # self.file_io = os.path.join(dir_name, .'.join((base_filename, filename_suffix)))
```

类的`save`方法使用`utf-8`编码以确保数据的读写兼容性。如果 JSON 文件已经存在，它将被追加而不是覆盖；否则，它将被创建：

```py
    def save(self, data):
        if os.path.isfile('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix)):
            # Append existing file
            with io.open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), 'a', encoding='utf-8') as f:
                f.write(unicode(json.dumps(data, ensure_ascii= False))) # In python 3, there is no "unicode" function 
                # f.write(json.dumps(data, ensure_ascii= False)) # create a \" escape char for " in the saved file        
        else:
            # Create new file
            with io.open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(data, ensure_ascii= False)))
                # f.write(json.dumps(data, ensure_ascii= False))
```

类的`load`方法仅返回已读取的文件。需要应用进一步的`json.loads`函数来从读取的文件中检索`json`：

```py
    def load(self):
        with io.open('{0}/{1}.{2}'.format(self.filepath, self.filename, self.filesuffix), encoding='utf-8') as f:
            return f.read()
```

## 设置 MongoDB

存储收集到的信息至关重要。因此，我们将 MongoDB 设置为我们的主要文档数据存储。由于所有收集到的信息都是 JSON 格式，而 MongoDB 以**BSON**（即**Binary JSON**）格式存储信息，因此这是一个自然的选择。

我们现在将执行以下步骤：

+   安装 MongoDB 服务器和客户端

+   运行 MongoDB 服务器

+   运行 Mongo 客户端

+   安装 PyMongo 驱动程序

+   创建 Python Mongo 客户端

### 安装 MongoDB 服务器和客户端

为了安装 MongoDB 软件包，执行以下步骤：

1.  导入包管理系统中使用的公钥（在我们的例子中，是 Ubuntu 的`apt`）。为了导入 MongoDB 的公钥，我们执行以下命令：

    ```py
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10

    ```

1.  为 MongoDB 创建一个列表文件。要创建列表文件，我们使用以下命令：

    ```py
    echo "deb http://repo.mongodb.org/apt/ubuntu "$("lsb_release -sc)"/ mongodb-org/3.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.0.list

    ```

1.  使用`sudo`更新本地软件包数据库：

    ```py
    sudo apt-get update

    ```

1.  安装 MongoDB 软件包。我们使用以下命令安装 MongoDB 的最新稳定版本：

    ```py
    sudo apt-get install -y mongodb-org

    ```

### 运行 MongoDB 服务器

让我们启动 MongoDB 服务器：

1.  要启动 MongoDB 服务器，我们执行以下命令以启动`mongod`：

    ```py
    sudo service mongodb start

    ```

1.  为了检查 `mongod` 是否已正确启动，我们发出以下命令：

    ```py
    an@an-VB:/usr/bin$ ps -ef | grep mongo
    mongodb    967     1  4 07:03 ?        00:02:02 /usr/bin/mongod --config /etc/mongod.conf
    an        3143  3085  0 07:45 pts/3    00:00:00 grep --color=auto mongo

    ```

    在这种情况下，我们看到 `mongodb` 正在进程 `967` 中运行。

1.  `mongod` 服务器发送一条消息，表明它正在等待连接到 `port 27017`。这是 MongoDB 的默认端口。它可以在配置文件中更改。

1.  我们可以检查 `/var/log/mongod/mongod.log` 中的日志文件内容：

    ```py
    an@an-VB:/var/lib/mongodb$ ls -lru
    total 81936
    drwxr-xr-x 2 mongodb nogroup     4096 Apr 25 11:19 _tmp
    -rw-r--r-- 1 mongodb nogroup       69 Apr 25 11:19 storage.bson
    -rwxr-xr-x 1 mongodb nogroup        5 Apr 25 11:19 mongod.lock
    -rw------- 1 mongodb nogroup 16777216 Apr 25 11:19 local.ns
    -rw------- 1 mongodb nogroup 67108864 Apr 25 11:19 local.0
    drwxr-xr-x 2 mongodb nogroup     4096 Apr 25 11:19 journal

    ```

1.  为了停止 `mongodb` 服务器，只需发出以下命令：

    ```py
    sudo service mongodb stop

    ```

### 运行 Mongo 客户端

在控制台中运行 Mongo 客户端就像调用 `mongo` 一样简单，如下所示：

```py
an@an-VB:/usr/bin$ mongo
MongoDB shell version: 3.0.2
connecting to: test
Server has startup warnings: 
2015-05-30T07:03:49.387+0200 I CONTROL  [initandlisten] 
2015-05-30T07:03:49.388+0200 I CONTROL  [initandlisten] 

```

在 mongo 客户端控制台提示符下，我们可以使用以下命令查看数据库：

```py
> show dbs
local  0.078GB
test   0.078GB
```

我们使用 `use test` 选择测试数据库：

```py
> use test
switched to db test
```

我们显示测试数据库中的集合：

```py
> show collections
restaurants
system.indexes
```

我们检查之前列出的餐厅集合中的一个样本记录：

```py
> db.restaurants.find()
{ "_id" : ObjectId("553b70055e82e7b824ae0e6f"), "address : { "building : "1007", "coord" : [ -73.856077, 40.848447 ], "street : "Morris Park Ave", "zipcode : "10462 }, "borough : "Bronx", "cuisine : "Bakery", "grades : [ { "grade : "A", "score" : 2, "date" : ISODate("2014-03-03T00:00:00Z") }, { "date" : ISODate("2013-09-11T00:00:00Z"), "grade : "A", "score" : 6 }, { "score" : 10, "date" : ISODate("2013-01-24T00:00:00Z"), "grade : "A }, { "date" : ISODate("2011-11-23T00:00:00Z"), "grade : "A", "score" : 9 }, { "date" : ISODate("2011-03-10T00:00:00Z"), "grade : "B", "score" : 14 } ], "name : "Morris Park Bake Shop", "restaurant_id : "30075445" }
```

### 安装 PyMongo 驱动程序

使用 anaconda 安装 Python 驱动程序非常简单。只需在终端运行以下命令：

```py
conda install pymongo
```

### 创建 MongoDB 的 Python 客户端

我们正在创建一个 `IO_mongo` 类，它将在我们的收集和处理程序中使用，以存储收集和检索保存的信息。为了创建 `mongo` 客户端，我们将从 `pymongo` 中导入 `MongoClient` 模块。我们连接到本地主机上的 `mongodb` 服务器，端口为 27017。命令如下：

```py
from pymongo import MongoClient as MCli

class IO_mongo(object):
    conn={'host':'localhost', 'ip':'27017'}
```

我们使用客户端连接、数据库（在这种情况下为 `twtr_db`）和要访问的集合（在这种情况下为 `twtr_coll`）来初始化我们的类：

```py
    def __init__(self, db='twtr_db', coll='twtr_coll', **conn ):
        # Connects to the MongoDB server 
        self.client = MCli(**conn)
        self.db = self.client[db]
        self.coll = self.db[coll]
```

`save` 方法将新记录插入到预先初始化的集合和数据库中：

```py
    def save(self, data):
        # Insert to collection in db  
        return self.coll.insert(data)
```

`load` 方法允许根据标准和投影检索特定记录。在大量数据的情况下，它返回一个游标：

```py
    def load(self, return_cursor=False, criteria=None, projection=None):

            if criteria is None:
                criteria = {}

            if projection is None:
                cursor = self.coll.find(criteria)
            else:
                cursor = self.coll.find(criteria, projection)

            # Return a cursor for large amounts of data
            if return_cursor:
                return cursor
            else:
                return [ item for item in cursor ]
```

## 从 Twitter 收集数据

每个社交网络都存在其局限性和挑战。在收集数据时，一个主要的障碍是施加的速率限制。在运行重复或长时间运行的连接时，我们必须要小心，以避免收集重复数据。

我们已经重新设计了上一章中概述的连接程序，以处理速率限制。

在这个 `TwitterAPI` 类中，我们连接并收集我们指定的搜索查询的推文，我们添加了以下内容：

+   使用 Python 日志库的日志功能，以收集程序失败时的任何错误或警告

+   使用 MongoDB 的持久性功能，包括之前公开的 `IO_mongo` 类以及使用 `IO_json` 类的 JSON 文件

+   API 速率限制和错误管理功能，这样我们就可以确保对 Twitter 的调用更加健壮，而不会因为访问数据流而被禁止。

让我们一步步来：

1.  我们通过实例化 Twitter API 并提供我们的凭据来初始化：

    ```py
    class TwitterAPI(object):
        """
        TwitterAPI class allows the Connection to Twitter via OAuth
        once you have registered with Twitter and receive the 
        necessary credentials 
        """

        def __init__(self): 
            consumer_key = 'get_your_credentials'
            consumer_secret = get your_credentials'
            access_token = 'get_your_credentials'
            access_secret = 'get your_credentials'
            self.consumer_key = consumer_key
            self.consumer_secret = consumer_secret
            self.access_token = access_token
            self.access_secret = access_secret
            self.retries = 3
            self.auth = twitter.oauth.OAuth(access_token, access_secret, consumer_key, consumer_secret)
            self.api = twitter.Twitter(auth=self.auth)
    ```

1.  我们通过提供日志级别来初始化日志记录器：

    +   `logger.debug`(调试信息)

    +   `logger.info`(信息消息)

    +   `logger.warn`(警告消息)

    +   `logger.error`(错误消息)

    +   `logger.critical`(关键信息)

1.  我们设置日志路径和消息格式：

    ```py
            # logger initialisation
            appName = 'twt150530'
            self.logger = logging.getLogger(appName)
            #self.logger.setLevel(logging.DEBUG)
            # create console handler and set level to debug
            logPath = '/home/an/spark/spark-1.3.0-bin-hadoop2.4/examples/AN_Spark/data'
            fileName = appName
            fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fileHandler.setFormatter(formatter)
            self.logger.addHandler(fileHandler) 
            self.logger.setLevel(logging.DEBUG)
    ```

1.  我们初始化 JSON 文件持久化指令：

    ```py
            # Save to JSON file initialisation
            jsonFpath = '/home/an/spark/spark-1.3.0-bin-hadoop2.4/examples/AN_Spark/data'
            jsonFname = 'twtr15053001'
            self.jsonSaver = IO_json(jsonFpath, jsonFname)
    ```

1.  我们初始化 MongoDB 数据库和集合以实现持久化：

    ```py
            # Save to MongoDB Intitialisation
            self.mongoSaver = IO_mongo(db='twtr01_db', coll='twtr01_coll')
    ```

1.  方法 `searchTwitter` 根据指定的查询启动搜索：

    ```py
        def searchTwitter(self, q, max_res=10,**kwargs):
            search_results = self.api.search.tweets(q=q, count=10, **kwargs)
            statuses = search_results['statuses']
            max_results = min(1000, max_res)

            for _ in range(10):
                try:
                    next_results = search_results['search_metadata']['next_results']
                    # self.logger.info('info' in searchTwitter - next_results:%s'% next_results[1:])
                except KeyError as e:
                    self.logger.error('error' in searchTwitter: %s', %(e))
                    break

                # next_results = urlparse.parse_qsl(next_results[1:]) # python 2.7
                next_results = urllib.parse.parse_qsl(next_results[1:])
                # self.logger.info('info' in searchTwitter - next_results[max_id]:', next_results[0:])
                kwargs = dict(next_results)
                # self.logger.info('info' in searchTwitter - next_results[max_id]:%s'% kwargs['max_id'])
                search_results = self.api.search.tweets(**kwargs)
                statuses += search_results['statuses']
                self.saveTweets(search_results['statuses'])

                if len(statuses) > max_results:
                    self.logger.info('info' in searchTwitter - got %i tweets - max: %i' %(len(statuses), max_results))
                    break
            return statuses
    ```

1.  `saveTweets` 方法实际上将收集到的推文保存为 JSON 和 MongoDB：

    ```py
        def saveTweets(self, statuses):
            # Saving to JSON File
            self.jsonSaver.save(statuses)

            # Saving to MongoDB
            for s in statuses:
                self.mongoSaver.save(s)
    ```

1.  `parseTweets` 方法允许我们从 Twitter API 提供的大量信息中提取关键推文信息：

    ```py
        def parseTweets(self, statuses):
            return [ (status['id'], 
                      status['created_at'], 
                      status['user']['id'],
                      status['user']['name'] 
                      status['text''text'], 
                      url['expanded_url']) 
                            for status in statuses 
                                for url in status['entities']['urls'] ]
    ```

1.  `getTweets` 方法调用前面描述的 `searchTwitter` 方法。`getTweets` 方法确保 API 调用可靠，同时尊重施加的速率限制。代码如下：

    ```py
        def getTweets(self, q,  max_res=10):
            """
            Make a Twitter API call whilst managing rate limit and errors.
            """
            def handleError(e, wait_period=2, sleep_when_rate_limited=True):
                if wait_period > 3600: # Seconds
                    self.logger.error('Too many retries in getTweets: %s', %(e))
                    raise e
                if e.e.code == 401:
                    self.logger.error('error 401 * Not Authorised * in getTweets: %s', %(e))
                    return None
                elif e.e.code == 404:
                    self.logger.error('error 404 * Not Found * in getTweets: %s', %(e))
                    return None
                elif e.e.code == 429: 
                    self.logger.error('error 429 * API Rate Limit Exceeded * in getTweets: %s', %(e))
                    if sleep_when_rate_limited:
                        self.logger.error('error 429 * Retrying in 15 minutes * in getTweets: %s', %(e))
                        sys.stderr.flush()
                        time.sleep(60*15 + 5)
                        self.logger.info('error 429 * Retrying now * in getTweets: %s', %(e))
                        return 2
                    else:
                        raise e # Caller must handle the rate limiting issue
                elif e.e.code in (500, 502, 503, 504):
                    self.logger.info('Encountered %i Error. Retrying in %i seconds' % (e.e.code, wait_period))
                    time.sleep(wait_period)
                    wait_period *= 1.5
                    return wait_period
                else:
                    self.logger.error('Exit - aborting - %s', %(e))
                    raise e
    ```

1.  在这里，我们根据指定的参数调用 `searchTwitter` API 的相关查询。如果遇到任何错误，例如来自提供者的速率限制，这将由 `handleError` 方法处理：

    ```py
            while True:
                try:
                    self.searchTwitter( q, max_res=10)
                except twitter.api.TwitterHTTPError as e:
                    error_count = 0 
                    wait_period = handleError(e, wait_period)
                    if wait_period is None:
                        return
    ```

# 使用 Blaze 探索数据

Blaze 是一个开源的 Python 库，主要由 Continuum.io 开发，利用 Python Numpy 数组和 Pandas dataframe。Blaze 扩展到离核计算，而 Pandas 和 Numpy 是单核。

Blaze 在各种后端提供可适应的、统一的和一致的用户界面。Blaze 协调以下操作：

+   **数据**：在 CSV、JSON、HDF5、HDFS 和 Bcolz 文件等存储之间无缝交换数据。

+   **计算**：使用相同的查询处理计算后端，如 Spark、MongoDB、Pandas 或 SQL Alchemy。

+   **符号表达式**：类似于 Pandas 的语法，但作用域有限的抽象表达式，如连接、分组、过滤、选择和投影。实现了 R 语言开创的拆分-应用-组合方法。

Blaze 表达式是惰性评估的，在这方面与 Spark RDDs 转换具有相似的处理范式。

让我们首先导入必要的库：`numpy`、`pandas`、`blaze` 和 `odo`。Odo 是 Blaze 的衍生品，确保从各种后端进行数据迁移。命令如下：

```py
import numpy as np
import pandas as pd
from blaze import Data, by, join, merge
from odo import odo
BokehJS successfully loaded.
```

我们通过读取保存在 CSV 文件 `twts_csv` 中的解析后的推文来创建 Pandas `Dataframe`：

```py
twts_pd_df = pd.DataFrame(twts_csv_read, columns=Tweet01._fields)
twts_pd_df.head()

Out[65]:
id    created_at    user_id    user_name    tweet_text    url
1   598831111406510082   2015-05-14 12:43:57   14755521 raulsaeztapia    RT @pacoid: Great recap of @StrataConf EU in L...   http://www.mango-solutions.com/wp/2015/05/the-...
2   598831111406510082   2015-05-14 12:43:57   14755521 raulsaeztapia    RT @pacoid: Great recap of @StrataConf EU in L...   http://www.mango-solutions.com/wp/2015/05/the-...
3   98808944719593472   2015-05-14 11:15:52   14755521 raulsaeztapia   RT @alvaroagea: Simply @ApacheSpark http://t.c...    http://www.webex.com/ciscospark/
4   598808944719593472   2015-05-14 11:15:52   14755521 raulsaeztapia   RT @alvaroagea: Simply @ApacheSpark http://t.c...   http://sparkjava.com/
```

我们将 Tweets Panda `Dataframe` 运行到 `describe()` 函数以获取有关数据集的一些总体信息：

```py
twts_pd_df.describe()
Out[66]:
id    created_at    user_id    user_name    tweet_text    url
count  19  19  19  19  19  19
unique    7  7   6   6     6   7
top    598808944719593472    2015-05-14 11:15:52    14755521 raulsaeztapia    RT @alvaroagea: Simply @ApacheSpark http://t.c...    http://bit.ly/1Hfd0Xm
freq    6    6    9    9    6    6
```

我们通过简单地通过 `Data()` 函数传递 Pandas `dataframe` 来将其转换为 Blaze `dataframe`：

```py
#
# Blaze dataframe
#
twts_bz_df = Data(twts_pd_df)
```

我们可以通过传递 `schema` 函数来检索 Blaze `dataframe` 的模式表示：

```py
twts_bz_df.schema
Out[73]:
dshape("""{
  id: ?string,
  created_at: ?string,
  user_id: ?string,
  user_name: ?string,
  tweet_text: ?string,
  url: ?string
  }""")
```

`.dshape` 函数提供了记录数和模式：

```py
twts_bz_df.dshape
Out[74]: 
dshape("""19 * {
  id: ?string,
  created_at: ?string,
  user_id: ?string,
  user_name: ?string,
  tweet_text: ?string,
  url: ?string
  }""")
```

我们可以打印 Blaze `dataframe` 的内容：

```py
twts_bz_df.data
Out[75]:
id    created_at    user_id    user_name    tweet_text    url
1    598831111406510082    2015-05-14 12:43:57   14755521 raulsaeztapia    RT @pacoid: Great recap of @StrataConf EU in L...    http://www.mango-solutions.com/wp/2015/05/the-...
2    598831111406510082    2015-05-14 12:43:57    14755521 raulsaeztapia    RT @pacoid: Great recap of @StrataConf EU in L...    http://www.mango-solutions.com/wp/2015/05/the-...
... 
18   598782970082807808    2015-05-14 09:32:39    1377652806 embeddedcomputer.nl    RT @BigDataTechCon: Moving Rating Prediction w...    http://buff.ly/1QBpk8J
19   598777933730160640     2015-05-14 09:12:38   294862170    Ellen Friedman   I'm still on Euro time. If you are too check o...http://bit.ly/1Hfd0Xm
```

我们提取列 `tweet_text` 并取其唯一值：

```py
twts_bz_df.tweet_text.distinct()
Out[76]:
    tweet_text
0   RT @pacoid: Great recap of @StrataConf EU in L...
1   RT @alvaroagea: Simply @ApacheSpark http://t.c...
2   RT @PrabhaGana: What exactly is @ApacheSpark a...
3   RT @Ellen_Friedman: I'm still on Euro time. If...
4   RT @BigDataTechCon: Moving Rating Prediction w...
5   I'm still on Euro time. If you are too check o...
```

我们从 `dataframe` 中提取多个列 `['id', 'user_name','tweet_text']` 并取其唯一记录：

```py
twts_bz_df[['id', 'user_name','tweet_text']].distinct()
Out[78]:
  id   user_name   tweet_text
0   598831111406510082   raulsaeztapia   RT @pacoid: Great recap of @StrataConf EU in L...
1   598808944719593472   raulsaeztapia   RT @alvaroagea: Simply @ApacheSpark http://t.c...
2   598796205091500032   John Humphreys   RT @PrabhaGana: What exactly is @ApacheSpark a...
3   598788561127735296   Leonardo D'Ambrosi   RT @Ellen_Friedman: I'm still on Euro time. If...
4   598785545557438464   Alexey Kosenkov   RT @Ellen_Friedman: I'm still on Euro time. If...
5   598782970082807808   embeddedcomputer.nl   RT @BigDataTechCon: Moving Rating Prediction w...
6   598777933730160640   Ellen Friedman   I'm still on Euro time. If you are too check o...
```

## 使用 Odo 转移数据

Odo 是 Blaze 的一个衍生项目。Odo 允许数据交换。Odo 确保数据在不同格式（CSV、JSON、HDFS 等）和不同数据库（SQL 数据库、MongoDB 等）之间迁移，使用一个非常简单的谓词：

```py
Odo(source, target)
```

要传输到数据库，使用 URL 指定地址。例如，对于 MongoDB 数据库，它看起来会是这样：

```py
mongodb://username:password@hostname:port/database_name::collection_name
```

让我们运行一些使用 Odo 的示例。在这里，我们通过读取 CSV 文件并创建 Blaze `dataframe` 来展示 `odo`：

```py
filepath   = csvFpath
filename   = csvFname
filesuffix = csvSuffix
twts_odo_df = Data('{0}/{1}.{2}'.format(filepath, filename, filesuffix))
```

计算数据框中的记录数：

```py
twts_odo_df.count()
Out[81]:
19
```

显示 `dataframe` 的前五个记录：

```py
twts_odo_df.head(5)
Out[82]:
  id   created_at   user_id   user_name   tweet_text   url
0   598831111406510082   2015-05-14 12:43:57   14755521   raulsaeztapia   RT @pacoid: Great recap of @StrataConf EU in L...   http://www.mango-solutions.com/wp/2015/05/the-...
1   598831111406510082   2015-05-14 12:43:57   14755521   raulsaeztapia   RT @pacoid: Great recap of @StrataConf EU in L...   http://www.mango-solutions.com/wp/2015/05/the-...
2   598808944719593472   2015-05-14 11:15:52   14755521   raulsaeztapia   RT @alvaroagea: Simply @ApacheSpark http://t.c...   http://www.webex.com/ciscospark/
3   598808944719593472   2015-05-14 11:15:52   14755521   raulsaeztapia   RT @alvaroagea: Simply @ApacheSpark http://t.c...   http://sparkjava.com/
4   598808944719593472   2015-05-14 11:15:52   14755521   raulsaeztapia   RT @alvaroagea: Simply @ApacheSpark http://t.c...   https://www.sparkfun.com/
```

从 `dataframe` 获取 `dshape` 信息，它给我们记录数和 schema：

```py
twts_odo_df.dshape
Out[83]:
dshape("var * {
  id: int64,
  created_at: ?datetime,
  user_id: int64,
  user_name: ?string,
  tweet_text: ?string,
  url: ?string
  }""")
```

将处理后的 Blaze `dataframe` 保存到 JSON：

```py
odo(twts_odo_distinct_df, '{0}/{1}.{2}'.format(jsonFpath, jsonFname, jsonSuffix))
Out[92]:
<odo.backends.json.JSONLines at 0x7f77f0abfc50>
```

将 JSON 文件转换为 CSV 文件：

```py
odo('{0}/{1}.{2}'.format(jsonFpath, jsonFname, jsonSuffix), '{0}/{1}.{2}'.format(csvFpath, csvFname, csvSuffix))
Out[94]:
<odo.backends.csv.CSV at 0x7f77f0abfe10>
```

# 使用 Spark SQL 探索数据

Spark SQL 是建立在 Spark Core 之上的关系型查询引擎。Spark SQL 使用一个名为 **Catalyst** 的查询优化器。

关系型查询可以使用 SQL 或 HiveQL 表达，并针对 JSON、CSV 和各种数据库执行。Spark SQL 给我们在 RDD 函数式编程的基础上，提供了 Spark 数据框的声明性编程的完整表达能力。

## 理解 Spark 数据框

这里是一条来自 `@bigdata` 的推文，宣布 Spark 1.3.0 的发布，Spark SQL 和数据框的出现。它还突出了图中下方的各种数据源。在上部，我们可以注意到 R 作为将在 Scala、Java 和 Python 之上逐步支持的新语言。最终，数据框哲学在 R、Python 和 Spark 之间无处不在。

![理解 Spark 数据框](img/B03986_03_02.jpg)

Spark 数据框起源于 SchemaRDDs。它结合了 RDD 和 Spark 可以推断的 schema，如果需要，在注册 dataframe 时可以请求。它允许我们使用普通的 SQL 查询复杂的嵌套 JSON 数据。延迟评估、血缘、分区和持久化适用于数据框。

让我们使用 Spark SQL 查询数据，首先导入 `SparkContext` 和 `SQLContext`：

```py
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
In [95]:
sc
Out[95]:
<pyspark.context.SparkContext at 0x7f7829581890>
In [96]:
sc.master
Out[96]:
u'local[*]'
''In [98]:
# Instantiate Spark  SQL context
sqlc =  SQLContext(sc)
```

我们读取使用 Odo 保存的 JSON 文件：

```py
twts_sql_df_01 = sqlc.jsonFile ("/home/an/spark/spark-1.3.0-bin-hadoop2.4/examples/AN_Spark/data/twtr15051401_distinct.json")
In [101]:
twts_sql_df_01.show()
created_at           id                 tweet_text           user_id    user_name          
2015-05-14T12:43:57Z 598831111406510082 RT @pacoid: Great... 14755521   raulsaeztapia      
2015-05-14T11:15:52Z 598808944719593472 RT @alvaroagea: S... 14755521   raulsaeztapia      
2015-05-14T10:25:15Z 598796205091500032 RT @PrabhaGana: W... 48695135   John Humphreys     
2015-05-14T09:54:52Z 598788561127735296 RT @Ellen_Friedma... 2385931712 Leonardo D'Ambrosi
2015-05-14T09:42:53Z 598785545557438464 RT @Ellen_Friedma... 461020977  Alexey Kosenkov    
2015-05-14T09:32:39Z 598782970082807808 RT @BigDataTechCo... 1377652806 embeddedcomputer.nl
2015-05-14T09:12:38Z 598777933730160640 I'm still on Euro... 294862170  Ellen Friedman     
```

我们打印 Spark 数据框的模式：

```py
twts_sql_df_01.printSchema()
root
 |-- created_at: string (nullable = true)
 |-- id: long (nullable = true)
 |-- tweet_text: string (nullable = true)
 |-- user_id: long (nullable = true)
 |-- user_name: string (nullable = true)
```

我们从 dataframe 中选择 `user_name` 列：

```py
twts_sql_df_01.select('user_name').show()
user_name          
raulsaeztapia      
raulsaeztapia      
John Humphreys     
Leonardo D'Ambrosi
Alexey Kosenkov    
embeddedcomputer.nl
Ellen Friedman     
```

我们将 dataframe 注册为表，因此可以在其上执行 SQL 查询：

```py
twts_sql_df_01.registerAsTable('tweets_01')
```

我们在 dataframe 上执行 SQL 语句：

```py
twts_sql_df_01_selection = sqlc.sql("SELECT * FROM tweets_01 WHERE user_name = 'raulsaeztapia'")
In [109]:
twts_sql_df_01_selection.show()
created_at           id                 tweet_text           user_id  user_name    
2015-05-14T12:43:57Z 598831111406510082 RT @pacoid: Great... 14755521 raulsaeztapia
2015-05-14T11:15:52Z 598808944719593472 RT @alvaroagea: S... 14755521 raulsaeztapia
```

让我们处理一些更复杂的 JSON；我们读取原始的 Twitter JSON 文件：

```py
tweets_sqlc_inf = sqlc.jsonFile(infile)
```

Spark SQL 能够推断复杂嵌套 JSON 文件的 schema：

```py
tweets_sqlc_inf.printSchema()
root
 |-- contributors: string (nullable = true)
 |-- coordinates: string (nullable = true)
 |-- created_at: string (nullable = true)
 |-- entities: struct (nullable = true)
 |    |-- hashtags: array (nullable = true)
 |    |    |-- element: struct (containsNull = true)
 |    |    |    |-- indices: array (nullable = true)
 |    |    |    |    |-- element: long (containsNull = true)
 |    |    |    |-- text: string (nullable = true)
 |    |-- media: array (nullable = true)
 |    |    |-- element: struct (containsNull = true)
 |    |    |    |-- display_url: string (nullable = true)
 |    |    |    |-- expanded_url: string (nullable = true)
 |    |    |    |-- id: long (nullable = true)
 |    |    |    |-- id_str: string (nullable = true)
 |    |    |    |-- indices: array (nullable = true)
... (snip) ...
|    |-- statuses_count: long (nullable = true)
 |    |-- time_zone: string (nullable = true)
 |    |-- url: string (nullable = true)
 |    |-- utc_offset: long (nullable = true)
 |    |-- verified: boolean (nullable = true)
```

我们通过在 dataframe 中选择特定的列（在这种情况下，`['created_at', 'id', 'text', 'user.id', 'user.name', 'entities.urls.expanded_url']`）从数据墙中提取感兴趣的关键信息：

```py
tweets_extract_sqlc = tweets_sqlc_inf[['created_at', 'id', 'text', 'user.id', 'user.name', 'entities.urls.expanded_url']].distinct()
In [145]:
tweets_extract_sqlc.show()
created_at           id                 text                 id         name                expanded_url        
Thu May 14 09:32:... 598782970082807808 RT @BigDataTechCo... 1377652806 embeddedcomputer.nl ArrayBuffer(http:...
Thu May 14 12:43:... 598831111406510082 RT @pacoid: Great... 14755521   raulsaeztapia       ArrayBuffer(http:...
Thu May 14 12:18:... 598824733086523393 @rabbitonweb spea... 

...   
Thu May 14 12:28:... 598827171168264192 RT @baandrzejczak... 20909005   Paweł Szulc         ArrayBuffer()       
```

## 理解 Spark SQL 查询优化器

我们在 dataframe 上执行 SQL 语句：

```py
tweets_extract_sqlc_sel = sqlc.sql("SELECT * from Tweets_xtr_001 WHERE name='raulsaeztapia'")
```

我们可以详细查看 Spark SQL 执行的查询计划：

+   解析的逻辑计划

+   分析的逻辑计划

+   优化的逻辑计划

+   物理计划

查询计划使用 Spark SQL 的 Catalyst 优化器。为了从查询部分生成编译后的字节码，Catalyst 优化器会运行逻辑计划解析和优化，然后基于成本进行物理计划评估和优化。

这在下述推文中得到了说明：

![理解 Spark SQL 查询优化器](img/B03986_03_03.jpg)

回顾我们的代码，我们在刚刚执行的 Spark SQL 查询上调用 `.explain` 函数，它提供了 Catalyst 优化器在评估和优化逻辑计划和物理计划以及得到结果 RDD 所采取的步骤的完整细节：

```py
tweets_extract_sqlc_sel.explain(extended = True)
== Parsed Logical Plan ==
'Project [*]
 'Filter ('name = raulsaeztapia)'name'  'UnresolvedRelation' [Tweets_xtr_001], None
== Analyzed Logical Plan ==
Project [created_at#7,id#12L,text#27,id#80L,name#81,expanded_url#82]
 Filter (name#81 = raulsaeztapia)
  Distinct 
   Project [created_at#7,id#12L,text#27,user#29.id AS id#80L,user#29.name AS name#81,entities#8.urls.expanded_url AS expanded_url#82]
    Relation[contributors#5,coordinates#6,created_at#7,entities#8,favorite_count#9L,favorited#10,geo#11,id#12L,id_str#13,in_reply_to_screen_name#14,in_reply_to_status_id#15,in_reply_to_status_id_str#16,in_reply_to_user_id#17L,in_reply_to_user_id_str#18,lang#19,metadata#20,place#21,possibly_sensitive#22,retweet_count#23L,retweeted#24,retweeted_status#25,source#26,text#27,truncated#28,user#29] JSONRelation(/home/an/spark/spark-1.3.0-bin-hadoop2.4/examples/AN_Spark/data/twtr15051401.json,1.0,None)
== Optimized Logical Plan ==
Filter (name#81 = raulsaeztapia)
 Distinct 
  Project [created_at#7,id#12L,text#27,user#29.id AS id#80L,user#29.name AS name#81,entities#8.urls.expanded_url AS expanded_url#82]
   Relation[contributors#5,coordinates#6,created_at#7,entities#8,favorite_count#9L,favorited#10,geo#11,id#12L,id_str#13,in_reply_to_screen_name#14,in_reply_to_status_id#15,in_reply_to_status_id_str#16,in_reply_to_user_id#17L,in_reply_to_user_id_str#18,lang#19,metadata#20,place#21,possibly_sensitive#22,retweet_count#23L,retweeted#24,retweeted_status#25,source#26,text#27,truncated#28,user#29] JSONRelation(/home/an/spark/spark-1.3.0-bin-hadoop2.4/examples/AN_Spark/data/twtr15051401.json,1.0,None)
== Physical Plan ==
Filter (name#81 = raulsaeztapia)
 Distinct false
  Exchange (HashPartitioning [created_at#7,id#12L,text#27,id#80L,name#81,expanded_url#82], 200)
   Distinct true
    Project [created_at#7,id#12L,text#27,user#29.id AS id#80L,user#29.name AS name#81,entities#8.urls.expanded_url AS expanded_url#82]
     PhysicalRDD [contributors#5,coordinates#6,created_at#7,entities#8,favorite_count#9L,favorited#10,geo#11,id#12L,id_str#13,in_reply_to_screen_name#14,in_reply_to_status_id#15,in_reply_to_status_id_str#16,in_reply_to_user_id#17L,in_reply_to_user_id_str#18,lang#19,metadata#20,place#21,possibly_sensitive#22,retweet_count#23L,retweeted#24,retweeted_status#25,source#26,text#27,truncated#28,user#29], MapPartitionsRDD[165] at map at JsonRDD.scala:41
Code Generation: false
== RDD ==
```

最后，这是查询的结果：

```py
tweets_extract_sqlc_sel.show()
created_at           id                 text                 id       name          expanded_url        
Thu May 14 12:43:... 598831111406510082 RT @pacoid: Great... 14755521 raulsaeztapia ArrayBuffer(http:...
Thu May 14 11:15:... 598808944719593472 RT @alvaroagea: S... 14755521 raulsaeztapia ArrayBuffer(http:...
In [148]:
```

## 使用 Spark SQL 加载和处理 CSV 文件

我们将使用 Spark 包 `spark-csv_2.11:1.2.0`。用于启动 PySpark 并使用 `spark-csv` 包的命令应明确指定 `–packages` 参数：

```py
$ IPYTHON_OPTS='notebook' /home/an/spark/spark-1.5.0-bin-hadoop2.6/bin/pyspark --packages com.databricks:spark-csv_2.11:1.2.0

```

这将触发以下输出；我们可以看到 `spark-csv` 包及其所有依赖项已安装：

```py
an@an-VB:~/spark/spark-1.5.0-bin-hadoop2.6/examples/AN_Spark$ IPYTHON_OPTS='notebook' /home/an/spark/spark-1.5.0-bin-hadoop2.6/bin/pyspark --packages com.databricks:spark-csv_2.11:1.2.0

```

```py
... (snip) ...
Ivy Default Cache set to: /home/an/.ivy2/cache
The jars for the packages stored in: /home/an/.ivy2/jars
:: loading settings :: url = jar:file:/home/an/spark/spark-1.5.0-bin-hadoop2.6/lib/spark-assembly-1.5.0-hadoop2.6.0.jar!/org/apache/ivy/core/settings/ivysettings.xml
com.databricks#spark-csv_2.11 added as a dependency
:: resolving dependencies :: org.apache.spark#spark-submit-parent;1.0
  confs: [default]
  found com.databricks#spark-csv_2.11;1.2.0 in central
  found org.apache.commons#commons-csv;1.1 in central
  found com.univocity#univocity-parsers;1.5.1 in central
:: resolution report :: resolve 835ms :: artifacts dl 48ms
  :: modules in use:
  com.databricks#spark-csv_2.11;1.2.0 from central in [default]
  com.univocity#univocity-parsers;1.5.1 from central in [default]
  org.apache.commons#commons-csv;1.1 from central in [default]
  ----------------------------------------------------------------
  |               |          modules            ||   artifacts   |
  |    conf     | number| search|dwnlded|evicted|| number|dwnlded|
  ----------------------------------------------------------------
  |    default     |   3   |   0   |   0   |   0   ||   3   |   0   
  ----------------------------------------------------------------
:: retrieving :: org.apache.spark#spark-submit-parent
  confs: [default]
  0 artifacts copied, 3 already retrieved (0kB/45ms)
```

我们现在准备加载我们的 `csv` 文件并处理它。首先，我们导入 `SQLContext`：

```py
#
# Read csv in a Spark DF
#
sqlContext = SQLContext(sc)
spdf_in = sqlContext.read.format('com.databricks.spark.csv')\
                                    .options(delimiter=";").options(header="true")\
                                    .options(header='true').load(csv_in)
```

我们访问从加载的 `csv` 创建的数据框的架构：

```py
In [10]:
spdf_in.printSchema()
root
 |-- : string (nullable = true)
 |-- id: string (nullable = true)
 |-- created_at: string (nullable = true)
 |-- user_id: string (nullable = true)
 |-- user_name: string (nullable = true)
 |-- tweet_text: string (nullable = true)
```

我们检查数据框的列：

```py
In [12]:
spdf_in.columns
Out[12]:
['', 'id', 'created_at', 'user_id', 'user_name', 'tweet_text']
```

我们检查数据框的内容：

```py
In [13]:
spdf_in.show()
+---+------------------+--------------------+----------+------------------+--------------------+
|   |                id|          created_at|   user_id|         user_name|          tweet_text|
+---+------------------+--------------------+----------+------------------+--------------------+
|  0|638830426971181057|Tue Sep 01 21:46:...|3276255125|     True Equality|ernestsgantt: Bey...|
|  1|638830426727911424|Tue Sep 01 21:46:...|3276255125|     True Equality|ernestsgantt: Bey...|
|  2|638830425402556417|Tue Sep 01 21:46:...|3276255125|     True Equality|ernestsgantt: Bey...|
... (snip) ...
| 41|638830280988426250|Tue Sep 01 21:46:...| 951081582|      Jack Baldwin|RT @cloudaus: We ...|
| 42|638830276626399232|Tue Sep 01 21:46:...|   6525302|Masayoshi Nakamura|PynamoDB 使いやすいです  |
+---+------------------+--------------------+----------+------------------+--------------------+
only showing top 20 rows
```

## 从 Spark SQL 查询 MongoDB

与 MongoDB 交互有两种主要方式：第一种是通过 Hadoop MongoDB 连接器，第二种是直接从 Spark 到 MongoDB。

从 Spark 交互 MongoDB 的第一种方法是设置 Hadoop 环境，并通过 Hadoop MongoDB 连接器进行查询。连接器详细信息托管在 GitHub 上，网址为 [`github.com/mongodb/mongo-hadoop/wiki/Spark-Usage`](https://github.com/mongodb/mongo-hadoop/wiki/Spark-Usage)。MongoDB 系列博客文章中描述了实际用例：

+   *使用 Hadoop 和 Spark 与 MongoDB：第一部分 - 简介与设置* ([`www.mongodb.com/blog/post/using-mongodb-hadoop-spark-part-1-introduction-setup`](https://www.mongodb.com/blog/post/using-mongodb-hadoop-spark-part-1-introduction-setup))

+   *使用 Hadoop 和 Spark 与 MongoDB：第二部分 - Hive 示例* ([`www.mongodb.com/blog/post/using-mongodb-hadoop-spark-part-2-hive-example`](https://www.mongodb.com/blog/post/using-mongodb-hadoop-spark-part-2-hive-example))

+   *使用 Hadoop 和 Spark 与 MongoDB：第三部分 - Spark 示例与关键要点* ([`www.mongodb.com/blog/post/using-mongodb-hadoop-spark-part-3-spark-example-key-takeaways`](https://www.mongodb.com/blog/post/using-mongodb-hadoop-spark-part-3-spark-example-key-takeaways))

设置完整的 Hadoop 环境有些复杂。我们将优先考虑第二种方法。我们将使用由 Stratio 开发和维护的 `spark-mongodb` 连接器。我们使用托管在 `spark.packages.org` 的 `Stratio spark-mongodb` 包。包信息和版本可以在 `spark.packages.org` 中找到：

### 注意

**版本**

版本：0.10.1 ( 8263c8 | zip | jar ) / 日期：2015-11-18 / 许可证：Apache-2.0 / Scala 版本：2.10

([`spark-packages.org/package/Stratio/spark-mongodb`](http://spark-packages.org/package/Stratio/spark-mongodb))

启动 PySpark 与 IPython Notebook 以及`spark-mongodb`包的命令应明确指定包参数：

```py
$ IPYTHON_OPTS='notebook' /home/an/spark/spark-1.5.0-bin-hadoop2.6/bin/pyspark --packages com.stratio.datasource:spark-mongodb_2.10:0.10.1

```

这将触发以下输出；我们可以看到`spark-mongodb`包及其所有依赖项已安装：

```py
an@an-VB:~/spark/spark-1.5.0-bin-hadoop2.6/examples/AN_Spark$ IPYTHON_OPTS='notebook' /home/an/spark/spark-1.5.0-bin-hadoop2.6/bin/pyspark --packages com.stratio.datasource:spark-mongodb_2.10:0.10.1
... (snip) ... 
Ivy Default Cache set to: /home/an/.ivy2/cache
The jars for the packages stored in: /home/an/.ivy2/jars
:: loading settings :: url = jar:file:/home/an/spark/spark-1.5.0-bin-hadoop2.6/lib/spark-assembly-1.5.0-hadoop2.6.0.jar!/org/apache/ivy/core/settings/ivysettings.xml
com.stratio.datasource#spark-mongodb_2.10 added as a dependency
:: resolving dependencies :: org.apache.spark#spark-submit-parent;1.0
  confs: [default]
  found com.stratio.datasource#spark-mongodb_2.10;0.10.1 in central
[W 22:10:50.910 NotebookApp] Timeout waiting for kernel_info reply from 764081d3-baf9-4978-ad89-7735e6323cb6
  found org.mongodb#casbah-commons_2.10;2.8.0 in central
  found com.github.nscala-time#nscala-time_2.10;1.0.0 in central
  found joda-time#joda-time;2.3 in central
  found org.joda#joda-convert;1.2 in central
  found org.slf4j#slf4j-api;1.6.0 in central
  found org.mongodb#mongo-java-driver;2.13.0 in central
  found org.mongodb#casbah-query_2.10;2.8.0 in central
  found org.mongodb#casbah-core_2.10;2.8.0 in central
downloading https://repo1.maven.org/maven2/com/stratio/datasource/spark-mongodb_2.10/0.10.1/spark-mongodb_2.10-0.10.1.jar ...
  [SUCCESSFUL ] com.stratio.datasource#spark-mongodb_2.10;0.10.1!spark-mongodb_2.10.jar (3130ms)
downloading https://repo1.maven.org/maven2/org/mongodb/casbah-commons_2.10/2.8.0/casbah-commons_2.10-2.8.0.jar ...
  [SUCCESSFUL ] org.mongodb#casbah-commons_2.10;2.8.0!casbah-commons_2.10.jar (2812ms)
downloading https://repo1.maven.org/maven2/org/mongodb/casbah-query_2.10/2.8.0/casbah-query_2.10-2.8.0.jar ...
  [SUCCESSFUL ] org.mongodb#casbah-query_2.10;2.8.0!casbah-query_2.10.jar (1432ms)
downloading https://repo1.maven.org/maven2/org/mongodb/casbah-core_2.10/2.8.0/casbah-core_2.10-2.8.0.jar ...
  [SUCCESSFUL ] org.mongodb#casbah-core_2.10;2.8.0!casbah-core_2.10.jar (2785ms)
downloading https://repo1.maven.org/maven2/com/github/nscala-time/nscala-time_2.10/1.0.0/nscala-time_2.10-1.0.0.jar ...
  [SUCCESSFUL ] com.github.nscala-time#nscala-time_2.10;1.0.0!nscala-time_2.10.jar (2725ms)
downloading https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.6.0/slf4j-api-1.6.0.jar ...
  [SUCCESSFUL ] org.slf4j#slf4j-api;1.6.0!slf4j-api.jar (371ms)
downloading https://repo1.maven.org/maven2/org/mongodb/mongo-java-driver/2.13.0/mongo-java-driver-2.13.0.jar ...
  [SUCCESSFUL ] org.mongodb#mongo-java-driver;2.13.0!mongo-java-driver.jar (5259ms)
downloading https://repo1.maven.org/maven2/joda-time/joda-time/2.3/joda-time-2.3.jar ...
  [SUCCESSFUL ] joda-time#joda-time;2.3!joda-time.jar (6949ms)
downloading https://repo1.maven.org/maven2/org/joda/joda-convert/1.2/joda-convert-1.2.jar ...
  [SUCCESSFUL ] org.joda#joda-convert;1.2!joda-convert.jar (548ms)
:: resolution report :: resolve 11850ms :: artifacts dl 26075ms
  :: modules in use:
  com.github.nscala-time#nscala-time_2.10;1.0.0 from central in [default]
  com.stratio.datasource#spark-mongodb_2.10;0.10.1 from central in [default]
  joda-time#joda-time;2.3 from central in [default]
  org.joda#joda-convert;1.2 from central in [default]
  org.mongodb#casbah-commons_2.10;2.8.0 from central in [default]
  org.mongodb#casbah-core_2.10;2.8.0 from central in [default]
  org.mongodb#casbah-query_2.10;2.8.0 from central in [default]
  org.mongodb#mongo-java-driver;2.13.0 from central in [default]
  org.slf4j#slf4j-api;1.6.0 from central in [default]
  ---------------------------------------------------------------------
  |                  |            modules            ||   artifacts   |
  |       conf       | number| search|dwnlded|evicted|| number|dwnlded|
  ---------------------------------------------------------------------
  |      default     |   9   |   9   |   9   |   0   ||   9   |   9   |
  ---------------------------------------------------------------------
:: retrieving :: org.apache.spark#spark-submit-parent
  confs: [default]
  9 artifacts copied, 0 already retrieved (2335kB/51ms)
... (snip) ... 
```

我们现在已准备好从数据库`twtr01_db`中的集合`twtr01_coll`在`localhost:27017`上查询 MongoDB：

我们首先导入`SQLContext`：

```py
In [5]:
from pyspark.sql import SQLContext
sqlContext.sql("CREATE TEMPORARY TABLE tweet_table USING com.stratio.datasource.mongodb OPTIONS (host 'localhost:27017', database 'twtr01_db', collection 'twtr01_coll')")
sqlContext.sql("SELECT * FROM tweet_table where id=598830778269769728 ").collect()
```

这是我们的查询输出：

```py
Out[5]:
[Row(text=u'@spark_io is now @particle - awesome news - now I can enjoy my Particle Cores/Photons + @sparkfun sensors + @ApacheSpark analytics :-)', _id=u'55aa640fd770871cba74cb88', contributors=None, retweeted=False, user=Row(contributors_enabled=False, created_at=u'Mon Aug 25 14:01:26 +0000 2008', default_profile=True, default_profile_image=False, description=u'Building open source tools for and teaching enterprise software developers', entities=Row(description=Row(urls=[]), url=Row(urls=[Row(url=u'http://t.co/TSHp13EWeu', indices=[0, 22], 

... (snip) ...

 9], name=u'Spark is Particle', screen_name=u'spark_io'), Row(id=487010011, id_str=u'487010011', indices=[17, 26], name=u'Particle', screen_name=u'particle'), Row(id=17877351, id_str=u'17877351', indices=[88, 97], name=u'SparkFun Electronics', screen_name=u'sparkfun'), Row(id=1551361069, id_str=u'1551361069', indices=[108, 120], name=u'Apache Spark', screen_name=u'ApacheSpark')]), is_quote_status=None, lang=u'en', quoted_status_id_str=None, quoted_status_id=None, created_at=u'Thu May 14 12:42:37 +0000 2015', retweeted_status=None, truncated=False, place=None, id=598830778269769728, in_reply_to_user_id=3187046084, retweet_count=0, in_reply_to_status_id=None, in_reply_to_screen_name=u'spark_io', in_reply_to_user_id_str=u'3187046084', source=u'<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>', id_str=u'598830778269769728', coordinates=None, metadata=Row(iso_language_code=u'en', result_type=u'recent'), quoted_status=None)]
#
```

# 摘要

在本章中，我们从 Twitter 收集了数据。一旦数据被获取，我们就使用`Continuum.io's`的 Blaze 和 Odo 库来探索这些信息。Spark SQL 是交互式数据探索、分析和转换的重要模块，它利用了 Spark dataframe 数据结构。dataframe 概念起源于 R，然后被 Python Pandas 成功采用。dataframe 是数据科学家的得力助手。Spark SQL 和 dataframe 的结合为数据处理创建了一个强大的引擎。

我们现在正在准备使用 Spark MLlib 的机器学习从数据集中提取洞察：
