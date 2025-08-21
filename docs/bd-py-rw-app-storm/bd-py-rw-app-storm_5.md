# 第五章：使用 Redis 和 MongoDB 进行持久化

通常需要将元组存储在持久性数据存储中，例如 NoSQL 数据库或快速键值缓存，以进行额外的分析。在本章中，我们将借助两种流行的持久性媒体 Redis 和 MongoDB，重新访问来自第四章的 Twitter 趋势分析拓扑，*示例拓扑-推特*。

Redis（[`redis.io/`](http://redis.io/)）是一个开源的 BSD 许可高级键值缓存和存储。MongoDB 是一个跨平台的面向文档的数据库（[`www.mongodb.org/`](https://www.mongodb.org/)）。

在本章中，我们将解决以下两个问题：

+   使用 Redis 查找热门推文话题

+   使用 MongoDB 计算城市提及的每小时聚合

# 使用 Redis 查找排名前 n 的话题

拓扑将计算过去 5 分钟内最受欢迎的单词的滚动排名。单词计数存储在长度为 60 秒的各个窗口中。它包括以下组件：

+   Twitter 流喷口（`twitterstream.py`）：这从 Twitter 样本流中读取推文。这个喷口与第四章中的相同，*示例拓扑-推特*。

+   分割器螺栓（`splitsentence.py`）：这接收推文并将它们分割成单词。这也与第四章中的相同，*示例拓扑-推特*。

+   滚动字数计数螺栓（`rollingcount.py`）：这接收单词并计算出现次数。 Redis 键看起来像`twitter_word_count：<当前窗口开始时间（以秒为单位）>`，值存储在哈希中，格式如下：

```scala
{
    "word1": 5,
    "word2", 3,
}
```

这个螺栓使用 Redis 的`expireat`命令在 5 分钟后丢弃旧数据。这些代码行执行关键工作：

```scala
      self.conn.zincrby(name, word)
      self.conn.expireat(name, expires)
      Total rankings bolt (totalrankings.py)
```

在这个螺栓中，以下代码完成了最重要的工作：

```scala
self.conn.zunionstore(
    'twitter_word_count',
    ['twitter_word_count:%s' % t for t in xrange(
        first_window, now_floor)])
for t in self.conn.zrevrange('twitter_word_count', 0, self.maxSize, withscores=True):
    log.info('Emitting: %s', repr(t))
    storm.emit(t)
```

这个螺栓计算了在过去的 num_windows 周期内的前`maxSize`个单词。`zunionstore()`组合了各个时期的单词计数。`zrevrange()`对组合计数进行排序，返回前`maxSize`个单词。

在原始的 Twitter 示例中，`rollingcount.py`，`intermediaterankings.py`和`totalrankings.py`中实现了大致相同的逻辑。使用 Redis，我们可以用几行代码实现相同的计算。设计将大部分工作委托给了 Redis。根据您的数据量，这可能不如前一章中的拓扑那样具有规模。但是，这表明了 Redis 的能力远远不止于简单存储数据。

## 拓扑配置文件-Redis 案例

接下来是拓扑配置文件。根据您的 Redis 安装，您可能需要更改`redis_url`的值。

在`topology.yaml`中输入以下代码：

```scala
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

## 滚动字数计数螺栓-Redis 案例

滚动字数计数螺栓类似于第三章中的字数计数螺栓，*介绍 Petrel*。早期章节中的螺栓只是无限累积了单词计数。这对于分析 Twitter 上的热门话题并不好，因为热门话题可能在下一刻就会改变。相反，我们希望计数反映最新的信息。如前所述，滚动字数计数螺栓将数据存储在基于时间的存储桶中。然后，定期丢弃超过 5 分钟的存储桶。因此，这个螺栓的单词计数只考虑最近 5 分钟的数据。

在`rollingcount.py`中输入以下代码：

```scala
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

## 总排名螺栓-Redis 案例

在`totalrankings.py`中输入以下代码：

```scala
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

## 定义拓扑-Redis 案例

这是定义拓扑结构的`create.py`脚本：

```scala
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

# 运行拓扑-Redis 案例

在运行拓扑之前，我们还有一些小事情要处理：

1.  从第三章的第二个例子中复制`logconfig.ini`文件，*Petrel 介绍*到这个拓扑的目录。

1.  创建一个名为`setup.sh`的文件。Petrel 将会把这个脚本和拓扑一起打包，并在启动时运行它。这个脚本安装了拓扑使用的第三方 Python 库。文件看起来是这样的：

```scala
pip install -U pip
pip install nltk==3.0.1 oauthlib==0.7.2
tweepy==3.2.0
```

1.  创建一个名为`manifest.txt`的文件，包含以下两行：

```scala
logconfig.ini
setup.sh
```

1.  在一个已知的节点上安装 Redis 服务器。所有的工作节点都会在这里存储状态：

```scala
 sudo apt-get install redis-server
```

1.  在所有 Storm 工作节点上安装 Python Redis 客户端：

```scala
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

```scala
petrel submit --config topology.yaml --logdir `pwd`
```

拓扑运行后，在拓扑目录中打开另一个终端。输入以下命令来查看总排名 bolt 的日志文件，从最旧到最新排序：

```scala
ls -ltr petrel*totalrankings.log
```

如果这是你第一次运行这个拓扑，那么只会列出一个日志文件。每次运行都会创建一个新文件。如果列出了几个文件，选择最近的一个。输入以下命令来监视日志文件的内容（确切的文件名在你的系统上会有所不同）：

```scala
tail -f petrel24748_totalrankings.log
```

定期地，你会看到类似以下的输出，按照流行度降序列出前 5 个单词：

`totalrankings`的示例输出：

```scala
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

## 使用 MongoDB 按城市名称查找每小时推文数量

MongoDB 是一个用于存储大量数据的流行数据库。它被设计为在许多节点之间轻松扩展。

要运行这个拓扑，首先需要安装 MongoDB 并配置一些特定于数据库的设置。这个例子使用一个名为`cities`的 MongoDB 数据库，其中包含一个名为`minute`的集合。为了计算每个城市和分钟的计数，我们必须在`cities.minute`集合上创建一个唯一索引。为此，启动 MongoDB 命令行客户端：

```scala
mongo
```

在`cities.minute`集合上创建一个唯一索引：

```scala
use cities
db.minute.createIndex( { minute: 1, city: 1 }, { unique: true } )
```

这个索引在 MongoDB 中存储了每分钟城市计数的时间序列。在运行示例拓扑捕获一些数据后，我们将运行一个独立的命令行脚本（`city_report.py`）来按小时和城市汇总每分钟的城市计数。

这是之前 Twitter 拓扑的一个变种。这个例子使用了 Python 的 geotext 库（[`pypi.python.org/pypi/geotext`](https://pypi.python.org/pypi/geotext)）来查找推文中的城市名称。

以下是拓扑的概述：

+   阅读推文。

+   将它们拆分成单词并找到城市名称。

+   在 MongoDB 中，计算每分钟提到一个城市的次数。

+   Twitter 流 spout（`twitterstream.py`）：从 Twitter 样本流中读取推文。

+   城市计数 bolt（`citycount.py`）：这个模块找到城市名称并写入 MongoDB。它类似于 Twitter 样本中的`SplitSentenceBolt`，但在拆分单词后，它会寻找城市名称。

这里的`_get_words()`函数与之前的例子略有不同。这是因为 geotext 不会将小写字符串识别为城市名称。

它创建或更新 MongoDB 记录，利用了分钟和城市的唯一索引来累积每分钟的计数。

这是在 MongoDB 中表示时间序列数据的常见模式。每条记录还包括一个`hour`字段。`city_report.py`脚本使用这个字段来计算每小时的计数。

在`citycount.py`中输入以下代码：

```scala
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

## 定义拓扑 - MongoDB 案例

在`create.py`中输入以下代码：

```scala
from twitterstream import TwitterStreamSpout
from citycount import CityCountBolt

def create(builder):
    spoutId = "spout"
    cityCountId = "citycount"
    builder.setSpout(spoutId, TwitterStreamSpout(), 1)
    builder.setBolt(cityCountId, CityCountBolt(), 1).shuffleGrouping("spout")
```

# 运行拓扑 - MongoDB 案例

在我们运行拓扑之前，我们还有一些小事情要处理：

1.  从第三章的第二个例子中复制`logconfig.ini`文件，*Petrel 介绍*到这个拓扑的目录。

1.  创建一个名为`setup.sh`的文件：

```scala
pip install -U pip
pip install nltk==3.0.1 oauthlib==0.7.2 tweepy==3.2.0 geotext==0.1.0 pymongo==3.0.3
```

1.  接下来，创建一个名为`manifest.txt`的文件。这与 Redis 示例相同。

安装 MongoDB 服务器。在 Ubuntu 上，您可以使用[`docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/`](http://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/)中提供的说明。 

1.  在所有 Storm 工作机器上安装 Python MongoDB 客户端：

```scala
pip install pymongo==3.0.3
```

1.  要验证`pymongo`是否已安装并且索引已正确创建，请运行`python`启动交互式 Python 会话，然后使用此代码：

```scala
import pymongo
from pymongo import MongoClient
db = MongoClient()
for index in db.cities.minute.list_indexes():
    print index
```

您应该看到以下输出。第二行是我们添加的索引：

```scala
SON([(u'v', 1), (u'key', SON([(u'_id', 1)])), (u'name', u'_id_'), (u'ns', u'cities.minute')])
SON([(u'v', 1), (u'unique', True), (u'key', SON([(u'minute', 1.0), (u'city', 1.0)])), (u'name', u'minute_1_city_1'), (u'ns', u'cities.minute')])
```

1.  接下来，安装`geotext`：

```scala
pip install geotext==0.1.0
```

1.  在运行拓扑之前，让我们回顾一下我们创建的文件列表。确保您已正确创建这些文件：

+   `topology.yaml`

+   `twitterstream.py`

+   `citycount.py`

+   `manifest.txt`

+   `setup.sh`

1.  使用以下命令运行拓扑：

```scala
petrel submit --config topology.yaml --logdir `pwd`
```

`city_report.py`文件是一个独立的脚本，它从拓扑插入的数据中生成一个简单的每小时报告。此脚本使用 MongoDB 聚合来计算每小时的总数。正如前面所述，报告取决于是否存在`hour`字段。

在`city_report.py`中输入此代码：

```scala
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

在本章中，我们看到如何将两种流行的 NoSQL 存储引擎（Redis 和 MongoDB）与 Storm 一起使用。我们还向您展示了如何在拓扑中创建数据并从其他应用程序访问它，证明了 Storm 可以成为 ETL 管道的有效部分。
