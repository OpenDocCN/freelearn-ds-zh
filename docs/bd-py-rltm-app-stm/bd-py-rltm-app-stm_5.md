# 第五章：使用 Redis 和 MongoDB 进行持久化

通常有必要将元组存储在持久数据存储中，如 NoSQL 数据库或快速键值缓存，以执行额外的分析。在本章中，我们将借助两种流行的持久化媒体：Redis 和 MongoDB，回顾第四章中的 Twitter 趋势分析拓扑，*示例拓扑 - Twitter*。

Redis ([`redis.io/`](http://redis.io/)) 是一个开源的 BSD 许可的高级键值缓存和存储。MongoDB 是一个跨平台的文档型数据库 ([`www.mongodb.org/`](https://www.mongodb.org/))。

本章我们将解决以下两个问题：

+   使用 Redis 查找最热门的推文主题

+   使用 MongoDB 计算城市提及的小时聚合

# 使用 Redis 查找排名前 n 的热门主题

拓扑将计算过去 5 分钟内最受欢迎的单词的滚动排名。词频存储在长度为 60 秒的单独窗口中。它由以下组件组成：

+   Twitter 流源（`twitterstream.py`）：此源从 Twitter 样本流中读取推文。此源与第四章中的相同，*示例拓扑 - Twitter*。

+   分词 bolt（`splitsentence.py`）：此 bolt 接收推文并将它们分割成单词。这也与第四章中的相同，*示例拓扑 - Twitter*。

+   滚动词频 bolt（`rollingcount.py`）：此 bolt 接收单词并计算它们的出现次数。Redis 键看起来像`twitter_word_count:<当前窗口开始时间（秒）>`，值以以下简单格式存储：

    ```py
    {
        "word1": 5,
        "word2", 3,
    }
    ```

    此 bolt 使用 Redis 的`expireat`命令在 5 分钟后丢弃旧数据。以下代码行执行关键工作：

    ```py
          self.conn.zincrby(name, word)
          self.conn.expireat(name, expires)
          Total rankings bolt (totalrankings.py)
    ```

在此 bolt 中，以下代码执行最重要的工作：

```py
self.conn.zunionstore(
    'twitter_word_count',
    ['twitter_word_count:%s' % t for t in xrange(
        first_window, now_floor)])
for t in self.conn.zrevrange('twitter_word_count', 0, self.maxSize, withscores=True):
    log.info('Emitting: %s', repr(t))
    storm.emit(t)
```

此 bolt 计算过去 num_windows 个周期内的前`maxSize`个单词。`zunionstore()`将周期内的词频合并。`zrevrange()`对合并的计数进行排序，返回前`maxSize`个单词。

在原始的 Twitter 示例中，大致相同的逻辑在`rollingcount.py`、`intermediaterankings.py`和`totalrankings.py`中实现。使用 Redis，我们只需几行代码就能实现相同的计算。设计将大部分工作委托给了 Redis。根据您的数据量，这可能不如上一章中的拓扑扩展性好。然而，它展示了 Redis 的能力远不止简单地存储数据。

## 拓扑配置文件 - Redis 案例

接下来是拓扑配置文件。根据您的 Redis 安装情况，您可能需要更改`redis_url`的值。

在`topology.yaml`中输入以下代码：

```py
nimbus.host: "localhost"
topology.workers: 1
oauth.consumer_key: "your-key-for-oauth-blah"
oauth.consumer_secret: "your-secret-for-oauth-blah"
oauth.access_token: "your-access-token-blah"
oauth.access_token_secret: "your-access-secret-blah"
twitter_word_count.redis_url: "redis://localhost:6379"
twitter_word_count.num_windows: 5
twitter_word_count.window_duration: 60
```

## 滚动词频 bolt - Redis 案例

滚动词计数螺栓与第三章中的词计数螺栓类似。早期章节中的螺栓简单地无限期地累积词计数。这不利于分析 Twitter 上的热门话题，因为热门话题可能在一瞬间发生变化。相反，我们希望计数反映最新信息。如前所述，滚动词计数螺栓将数据存储在基于时间的桶中。然后，它定期丢弃超过 5 分钟年龄的桶。因此，此螺栓的词计数只考虑最后 5 分钟的数据。

在`rollingcount.py`中输入以下代码：

```py
import math
import time
from collections import defaultdict

import redis

from petrel import storm
from petrel.emitter import BasicBolt

class RollingCountBolt(BasicBolt):
    def __init__(self):
        super(RollingCountBolt, self).__init__(script=__file__)

    def initialize(self, conf, context):
        self.conf = conf
        self.num_windows = self.conf['twitter_word_count.num_windows']
        self.window_duration = self.conf['twitter_word_count.window_duration']
        self.conn = redis.from_url(conf['twitter_word_count.redis_url'])

    @classmethod
    def declareOutputFields(cls):
        return ['word', 'count']

    def process(self, tup):
        word = tup.values[0]
        now = time.time()
        now_floor = int(math.floor(now / self.window_duration) * self.window_duration)
        expires = int(now_floor + self.num_windows * self.window_duration)
        name = 'twitter_word_count:%s' % now_floor
        self.conn.zincrby(name, word)
        self.conn.expireat(name, expires)

    def run():
        RollingCountBolt().run()
```

## 总排名螺栓 – Redis 案例

在`totalrankings.py`中输入以下代码：

```py
import logging
import math
import time
import redis

from petrel import storm
from petrel.emitter import BasicBolt

log = logging.getLogger('totalrankings')

class TotalRankingsBolt(BasicBolt):
    emitFrequencyInSeconds = 15
    maxSize = 10

    def __init__(self):
        super(TotalRankingsBolt, self).__init__(script=__file__)
        self.rankedItems = {}

    def initialize(self, conf, context):
        self.conf = conf
          self.num_windows = \
            self.conf['twitter_word_count.num_windows']
        self.window_duration = \
            self.conf['twitter_word_count.window_duration']
        self.conn = redis.from_url(
            conf['twitter_word_count.redis_url'])

    def declareOutputFields(self):
        return ['word', 'count']

    def process(self, tup):
        if tup.is_tick_tuple():
            now = time.time()
            now_floor = int(math.floor(now / self.window_duration) *
                self.window_duration)
            first_window = int(now_floor - self.num_windows *
                self.window_duration)
            self.conn.zunionstore(
                'twitter_word_count',
                ['twitter_word_count:%s' % t for t in xrange(first_window, now_floor)])
            for t in self.conn.zrevrange('
                'twitter_word_count', 0,
               self.maxSize, withScores=True):
                log.info('Emitting: %s', repr(t))
                storm.emit(t)
    def getComponentConfiguration(self):
          return {"topology.tick.tuple.freq.secs":
            self.emitFrequencyInSeconds}

   def run():
       TotalRankingsBolt().run()
```

## 定义拓扑 – Redis 案例

这里是定义拓扑结构的`create.py`脚本：

```py
from twitterstream import TwitterStreamSpout
from splitsentence import SplitSentenceBolt
from rollingcount import RollingCountBolt
from totalrankings import TotalRankingsBolt

def create(builder):
    spoutId = "spout"
    splitterId = "splitter"
    counterId = "counter"
    totalRankerId = "finalRanker"
    builder.setSpout(spoutId, TwitterStreamSpout(), 1)
    builder.setBolt(
        splitterId, SplitSentenceBolt(), 1).shuffleGrouping("spout")
    builder.setBolt(
        counterId, RollingCountBolt(), 4).fieldsGrouping(
            splitterId, ["word"])
    builder.setBolt(
        totalRankerId, TotalRankingsBolt()).globalGrouping(
            counterId)
```

# 运行拓扑 – Redis 案例

在我们运行拓扑之前，还有一些小事情要处理：

1.  将第三章中的第二个示例`logconfig.ini`文件复制到这个拓扑目录中。

1.  创建一个名为`setup.sh`的文件。Petrel 将与此拓扑一起打包此脚本并在启动时运行它。此脚本安装拓扑使用的第三方 Python 库。文件看起来像这样：

    ```py
    pip install -U pip
    pip install nltk==3.0.1 oauthlib==0.7.2
    tweepy==3.2.0
    ```

1.  创建一个包含以下两行的`manifest.txt`文件：

    ```py
    logconfig.ini
    setup.sh
    ```

1.  在一个知名节点上安装 Redis 服务器。所有工作节点都将在此处存储状态：

    ```py
     sudo apt-get install redis-server
    ```

1.  在所有 Storm 工作机器上安装 Python Redis 客户端：

    ```py
     sudo apt-get install python-redis
    ```

1.  在运行拓扑之前，让我们回顾一下我们创建的文件列表。确保你已经正确创建了这些文件：

    +   `topology.yaml`

    +   `twitterstream.py`

    +   `splitsentence.py`

    +   `rollingcount.py`

    +   `totalrankings.py`

    +   `manifest.txt`

    +   `setup.sh`

1.  使用以下命令运行拓扑：

    ```py
    petrel submit --config topology.yaml --logdir `pwd`
    ```

一旦拓扑开始运行，在拓扑目录中打开另一个终端。输入以下命令以查看总排名螺栓的日志文件，按时间顺序从旧到新排序：

```py
ls -ltr petrel*totalrankings.log
```

如果你第一次运行拓扑，将只列出一个日志文件。每次运行都会创建一个新的文件。如果有几个列出，请选择最新的一个。输入以下命令以监控日志文件的内容（在您的系统上，确切的文件名可能不同）：

```py
tail -f petrel24748_totalrankings.log
```

定期，你会看到如下输出，列出按流行度降序排列的前 5 个单词：

`totalrankings`的示例输出：

```py
[2015-08-10 21:30:01,691][totalrankings][INFO]Emitting: ('love', 74.0)
[2015-08-10 21:30:01,691][totalrankings][INFO]Emitting: ('amp', 68.0)
[2015-08-10 21:30:01,691][totalrankings][INFO]Emitting: ('like', 67.0)
[2015-08-10 21:30:01,692][totalrankings][INFO]Emitting: ('zaynmalik', 61.0)
[2015-08-10 21:30:01,692][totalrankings][INFO]Emitting: ('mtvhottest', 61.0)
[2015-08-10 21:30:01,692][totalrankings][INFO]Emitting: ('get', 58.0)
[2015-08-10 21:30:01,692][totalrankings][INFO]Emitting: ('one', 49.0)
[2015-08-10 21:30:01,692][totalrankings][INFO]Emitting: ('follow', 46.0)
[2015-08-10 21:30:01,692][totalrankings][INFO]Emitting: ('u', 44.0)
[2015-08-10 21:30:01,692][totalrankings][INFO]Emitting: ('new', 38.0)
[2015-08-10 21:30:01,692][totalrankings][INFO]Emitting: ('much', 37.0)
```

## 使用 MongoDB 按城市名称查找推文的每小时计数

MongoDB 是一个用于存储大量数据的流行数据库。它设计用于跨多个节点轻松扩展。

要运行此拓扑，你首先需要安装 MongoDB 并配置一些数据库特定的设置。此示例使用名为 `cities` 的 MongoDB 数据库，其集合名为 `minute`。为了按城市和分钟计算计数，我们必须在 `cities.minute` 集合上创建一个唯一索引。为此，启动 MongoDB 命令行客户端：

```py
mongo
```

在 `cities.minute` 集合上创建一个唯一索引：

```py
use cities
db.minute.createIndex( { minute: 1, city: 1 }, { unique: true } )
```

此索引在 MongoDB 中存储每分钟城市计数的时序数据。在运行示例拓扑以捕获一些数据后，我们将运行一个独立的命令行脚本 (`city_report.py`) 来按小时和城市汇总每分钟的计数。

这是之前 Twitter 拓扑的一个变体。此示例使用 Python geotext 库 ([`pypi.python.org/pypi/geotext`](https://pypi.python.org/pypi/geotext)) 在推文中查找城市名称。

下面是拓扑的概述：

+   读取推文。

+   将它们拆分成单词并找出城市名称。

+   在 MongoDB 中，计算每分钟提及一个城市的次数。

+   Twitter 流源 (`twitterstream.py`)：此代码从 Twitter 样本流中读取推文。

+   城市计数 Bolt (`citycount.py`)：此代码查找城市名称并将其写入 MongoDB。它与 Twitter 样本的 `SplitSentenceBolt` 类似，但在按单词拆分后，它会查找城市名称。

这里的 `_get_words()` 函数与之前的示例略有不同。这是因为 geotext 不识别小写字符串为城市名称。

它创建或更新 MongoDB 记录，利用分钟和城市上的唯一索引来累积每分钟的计数。

这是表示 MongoDB 中时间序列数据的一种常见模式。每条记录还包括一个 `hour` 字段。`city_report.py` 脚本使用此字段来计算每小时计数。

在 `citycount.py` 文件中输入以下代码：

```py
Import datetime
import logging
import geotext
import nltk.corpus
import pymongo

from petrel import storm
from petrel.emitter import BasicBolt

log = logging.getLogger('citycount')

class CityCountBolt(BasicBolt):
    def __init__(self):
        super(CityCountBolt, self).__init__(script=__file__)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stop_words.update(['http', 'https', 'rt'])
        self.stop_cities = set([
            'bay', 'best', 'deal', 'man', 'metro', 'of', 'un'])

    def initialize(self, conf, context):
        self.db = pymongo.MongoClient()

    def declareOutputFields(self):
        return []

    def process(self, tup):
        clean_text = ' '.join(w for w in self._get_words(tup.values[0]))
        places = geotext.GeoText(clean_text)
        now_minute = self._get_minute()
        now_hour = now_minute.replace(minute=0)
        for city in places.cities:
            city = city.lower()
            if city in self.stop_cities:
                continue
            log.info('Updating count: %s, %s, %s', now_hour, now_minute, city)
            self.db.cities.minute.update(
                {
                    'hour': now_hour,
                    'minute': now_minute,
                    'city': city
                },
                {'$inc': { 'count' : 1 } },
                upsert=True)

    @staticmethod
    def _get_minute():
        return datetime.datetime.now().replace(second=0, microsecond=0)

    def _get_words(self, sentence):
        for w in nltk.word_tokenize(sentence):
            wl = w.lower()
            if wl.isalpha() and wl not in self.stop_words:
                yield w

def run():
    CityCountBolt().run()
```

## 定义拓扑 – MongoDB 的情况

在 `create.py` 文件中输入以下代码：

```py
from twitterstream import TwitterStreamSpout
from citycount import CityCountBolt

def create(builder):
    spoutId = "spout"
    cityCountId = "citycount"
    builder.setSpout(spoutId, TwitterStreamSpout(), 1)
    builder.setBolt(cityCountId, CityCountBolt(), 1).shuffleGrouping("spout")
```

# 运行拓扑 – MongoDB 的情况

在我们运行拓扑之前，还有一些小事情需要处理：

1.  将 第三章 中的第二个示例中的 `logconfig.ini` 文件复制到该拓扑目录。

1.  创建一个名为 `setup.sh` 的文件：

    ```py
    pip install -U pip
    pip install nltk==3.0.1 oauthlib==0.7.2 tweepy==3.2.0 geotext==0.1.0 pymongo==3.0.3
    ```

1.  接下来，创建一个名为 `manifest.txt` 的文件。这与 Redis 示例相同。

    安装 MongoDB 服务器。在 Ubuntu 上，你可以使用 [`docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/`](http://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/) 中给出的说明。

1.  在所有 Storm 工作机上都安装 Python MongoDB 客户端：

    ```py
    pip install pymongo==3.0.3
    ```

1.  为了验证 `pymongo` 是否已安装且索引已正确创建，通过运行 `python` 启动一个交互式 Python 会话。然后使用此代码：

    ```py
    import pymongo
    from pymongo import MongoClient
    db = MongoClient()
    for index in db.cities.minute.list_indexes():
        print index
    ```

    你应该看到以下输出。第二行是我们添加的索引：

    ```py
    SON([(u'v', 1), (u'key', SON([(u'_id', 1)])), (u'name', u'_id_'), (u'ns', u'cities.minute')])
    SON([(u'v', 1), (u'unique', True), (u'key', SON([(u'minute', 1.0), (u'city', 1.0)])), (u'name', u'minute_1_city_1'), (u'ns', u'cities.minute')])
    ```

1.  接下来，安装 `geotext`：

    ```py
    pip install geotext==0.1.0
    ```

1.  在运行拓扑之前，让我们回顾一下我们创建的文件列表。确保你已经正确创建了这些文件：

    +   `topology.yaml`

    +   `twitterstream.py`

    +   `citycount.py`

    +   `manifest.txt`

    +   `setup.sh`

1.  使用以下命令运行拓扑：

    ```py
    petrel submit --config topology.yaml --logdir `pwd`
    ```

`city_report.py` 文件是一个独立的脚本，它从拓扑插入的数据生成简单的每小时报告。此脚本使用 MongoDB 聚合来计算每小时的总数。如前所述，报告依赖于`hour`字段的存。

在 `city_report.py` 中输入以下代码：

```py
import pymongo

def main():
    db = pymongo.MongoClient()
    pipeline = [{
        '$group': { 
          '_id':   { 'hour': '$hour', 'city': '$city' },
          'count': { '$sum': '$count' } 
        } 
      }]
    for r in db.cities.command('aggregate', 'minute', pipeline=pipeline)['result']:
        print '%s,%s,%s' % (r['_id']['city'], r['_id']['hour'], r['count'])

if __name__ == '__main__':
    main()
```

# 摘要

在本章中，我们展示了如何使用两个流行的 NoSQL 存储引擎（Redis 和 MongoDB）与 Storm 结合使用。我们还向您展示了如何在拓扑中创建数据并从其他应用程序访问它，证明了 Storm 可以成为 ETL 管道的有效部分。
