# 第十章。推荐系统

在本章中，我们将介绍以下食谱：

+   基于显式反馈的协同过滤

+   基于隐式反馈的协同过滤

# 简介

以下是对推荐系统的维基百科定义：

> *"推荐系统是信息过滤系统的一个子类，旨在预测用户对项目的'评分'或'偏好'。"*

近年来，推荐系统获得了巨大的普及。亚马逊使用它们推荐书籍，Netflix 推荐电影，谷歌新闻推荐新闻故事。正如俗话所说，“实践是检验真理的唯一标准”，以下是一些推荐可能产生的影响的例子（来源：Celma, Lamere, 2008）：

+   净 flix 上观看的影片中有三分之二是通过推荐观看的

+   38% 的谷歌新闻点击是通过推荐实现的

+   亚马逊 35%的销售额是通过推荐实现的

如前几章所见，特征和特征选择在机器学习算法的有效性中起着重要作用。推荐引擎算法会自动发现这些特征，称为**潜在特征**。简而言之，存在一些潜在特征，使一个用户喜欢一部电影而讨厌另一部。如果另一个用户有相应的潜在特征，那么这个人也很可能对电影有相似的品味。

为了更好地理解这一点，让我们看看一些示例电影评分：

| 电影 | Rich | Bob | Peter | Chris |
| --- | --- | --- | --- | --- |
| *Titanic* | 5 | 3 | 5 | ? |
| *GoldenEye* | 3 | 2 | 1 | 5 |
| *Toy Story* | 1 | ? | 2 | 2 |
| *Disclosure* | 4 | 4 | ? | 4 |
| *Ace Ventura* | 4 | ? | 4 | ? |

我们的目标是预测带有?符号的缺失条目。让我们看看我们是否能找到与电影相关的某些特征。一开始，你将查看类型，如下所示：

| 电影 | 类型 |
| --- | --- |
| *Titanic* | 动作，浪漫 |
| *GoldenEye* | 动作，冒险，惊悚 |
| *Toy Story* | 动画，儿童，喜剧 |
| *Disclosure* | 剧情，惊悚 |
| *Ace Ventura* | 喜剧 |

现在每部电影都可以对每个类型从 0 到 1 进行评分。例如，*GoldenEye* 并非主要是一部浪漫电影，因此它可能对浪漫的评分为 0.1，但对动作的评分为 0.98。因此，每部电影都可以表示为一个特征向量。

### 注意

在本章中，我们将使用来自 [grouplens.org/datasets/movielens/](http://grouplens.org/datasets/movielens/) 的 MovieLens 数据集。

InfoObjects 大数据沙盒预装了 10 万条电影评分。从 GroupLens，你还可以下载 100 万或甚至高达 1000 万的评分，如果你想要分析更大的数据集以获得更好的预测。

我们将使用这个数据集的两个文件：

+   `u.data`：这是一个制表符分隔的电影评分列表，其格式如下：

    ```py
    user id | item id | rating | epoch time
    ```

    由于我们不需要时间戳，我们将从我们的食谱中过滤掉数据中的时间戳。

+   `u.item`：这是一个制表符分隔的电影列表，其格式如下：

    ```py
    movie id | movie title | release date | video release date |               IMDb URL | unknown | Action | Adventure | Animation |               Children's | Comedy | Crime | Documentary | Drama | Fantasy |               Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |               Thriller | War | Western |
    ```

本章将介绍如何使用 MLlib（Spark 的机器学习库）进行推荐。

# 使用显式反馈的协同过滤

协同过滤是推荐系统中最常用的技术。它有一个有趣的特性——它能够自主学习特征。因此，在电影评分的情况下，我们不需要提供关于电影是否浪漫或动作的实际人类反馈。

正如我们在*简介*部分所看到的，电影有一些潜在特征，例如类型，同样用户也有一些潜在特征，例如年龄、性别等。协同过滤不需要这些特征，并且能够自主学习潜在特征。

在这个例子中，我们将使用一个名为**交替最小二乘法（ALS**）的算法。这个算法基于少数潜在特征解释电影与用户之间的关联。它使用三个训练参数：rank、迭代次数和 lambda（本章后面将解释）。确定这三个参数最佳值的方法是尝试不同的值，并查看哪个值具有最小的**均方根误差（RMSE**）。这个误差类似于标准差，但它基于模型结果而不是实际数据。

## 准备中

将从 GroupLens 下载的`moviedata`上传到`hdfs`中的`moviedata`文件夹：

```py
$ hdfs dfs -put moviedata moviedata

```

我们将向这个数据库添加一些个性化评分，以便我们可以测试推荐的准确性。

你可以通过查看`u.item`来挑选一些电影并对其进行评分。以下是我选择的一些电影及其评分。请随意选择你想要评分的电影并提供你自己的评分。

| 电影 ID | 电影名称 | 评分（1-5） |
| --- | --- | --- |
| 313 | *泰坦尼克号* | 5 |
| 2 | *黄金眼* | 3 |
| 1 | *玩具总动员* | 1 |
| 43 | *披露* | 4 |
| 67 | *猫鼠游戏* | 4 |
| 82 | *侏罗纪公园* | 5 |
| 96 | *终结者 2* | 5 |
| 121 | *独立日* | 4 |
| 148 | *鬼影与黑暗* | 4 |

最大的用户 ID 是 943，因此我们将新用户添加为 944。让我们创建一个新的以逗号分隔的文件`p.data`，其中包含以下数据：

```py
944,313,5
944,2,3
944,1,1
944,43,4
944,67,4
944,82,5
944,96,5
944,121,4
944,148,4
```

## 如何做到这一点...

1.  将个性化电影数据上传到`hdfs`：

    ```py
    $ hdfs dfs -put p.data p.data

    ```

1.  导入 ALS 和评分类：

    ```py
    scala> import org.apache.spark.mllib.recommendation.ALS
    scala> import org.apache.spark.mllib.recommendation.Rating

    ```

1.  将评分数据加载到 RDD 中：

    ```py
    scala> val data = sc.textFile("moviedata/u.data")

    ```

1.  将`val data`转换为评分 RDD：

    ```py
    scala> val ratings = data.map { line => 
     val Array(userId, itemId, rating, _) = line.split("\t") 
     Rating(userId.toInt, itemId.toInt, rating.toDouble) 
    }

    ```

1.  将个性化评分数据加载到 RDD 中：

    ```py
    scala> val pdata = sc.textFile("p.data")

    ```

1.  将数据转换为个性化评分 RDD：

    ```py
    scala> val pratings = pdata.map { line => 
     val Array(userId, itemId, rating) = line.split(",")
     Rating(userId.toInt, itemId.toInt, rating.toDouble) 
    }

    ```

1.  将评分与个性化评分合并：

    ```py
    scala> val movieratings = ratings.union(pratings)

    ```

1.  使用具有 5 个 rank 和 10 次迭代以及 0.01 作为 lambda 的 ALS 构建模型：

    ```py
    scala> val model = ALS.train(movieratings, 10, 10, 0.01)

    ```

1.  让我们根据这个模型预测我对给定电影的评分。

1.  让我们从原始的电影 ID 为 195 的*终结者*开始：

    ```py
    scala> model.predict(sc.parallelize(Array((944,195)))).collect.foreach(println)
    Rating(944,195,4.198642954004738)

    ```

    由于我给*终结者 2*评了 5 分，这是一个合理的预测。

1.  让我们尝试电影 ID 为 402 的*鬼影*：

    ```py
    scala> model.predict(sc.parallelize(Array((944,402)))).collect.foreach(println)
    Rating(944,402,2.982213836456829)

    ```

    这是一个合理的猜测。

1.  让我们尝试*《鬼影与黑暗》*这部电影，我已经对其进行了评分，ID 为 148：

    ```py
    scala> model.predict(sc.parallelize(Array((944,402)))).collect.foreach(println)
    Rating(944,148,3.8629938805450035)

    ```

    预测非常接近，知道我给这部电影评了 4 分。

您可以将更多电影添加到`train`数据集中。还有 100 万和 1000 万评分数据集可用，这将进一步优化算法。

# 使用隐式反馈进行协同过滤

有时可用的反馈不是评分的形式，而是播放的音频轨道、观看的电影等形式。乍一看，这些数据可能不如用户明确给出的评分看起来那么好，但它们的信息量要大得多。

## 准备工作

我们将使用来自[`www.kaggle.com/c/msdchallenge/data`](http://www.kaggle.com/c/msdchallenge/data)的百万歌曲数据。您需要下载三个文件：

+   `kaggle_visible_evaluation_triplets`

+   `kaggle_users.txt`

+   `kaggle_songs.txt`

现在执行以下步骤：

1.  在`hdfs`中创建一个`songdata`文件夹，并将所有三个文件放在这里：

    ```py
    $ hdfs dfs -mkdir songdata

    ```

1.  将歌曲数据上传到`hdfs`：

    ```py
    $ hdfs dfs -put kaggle_visible_evaluation_triplets.txt songdata/
    $ hdfs dfs -put kaggle_users.txt songdata/
    $ hdfs dfs -put kaggle_songs.txt songdata/

    ```

我们还需要做一些更多的预处理。MLlib 中的 ALS 接受用户和产品 ID 作为整数。`Kaggle_songs.txt`文件中包含歌曲 ID 和序列号，而`Kaggle_users.txt`文件没有。我们的目标是替换`triplets`数据中的`userid`和`songid`为相应的整数序列号。为此，请按照以下步骤操作：

1.  将`kaggle_songs`数据加载为 RDD：

    ```py
    scala> val songs = sc.textFile("songdata/kaggle_songs.txt")

    ```

1.  将用户数据加载为 RDD：

    ```py
    scala> val users = sc.textFile("songdata/kaggle_users.txt")

    ```

1.  将三元组（用户，歌曲，播放次数）数据加载为 RDD：

    ```py
    scala> val triplets = sc.textFile("songdata/kaggle_visible_evaluation_triplets.txt")

    ```

1.  将歌曲数据转换为`PairRDD`：

    ```py
    scala> val songIndex = songs.map(_.split("\\W+")).map(v => (v(0),v(1).toInt))

    ```

1.  收集`songIndex`为 Map：

    ```py
    scala> val songMap = songIndex.collectAsMap

    ```

1.  将用户数据转换为`PairRDD`：

    ```py
    scala> val userIndex = users.zipWithIndex.map( t => (t._1,t._2.toInt))

    ```

1.  收集`userIndex`为 Map：

    ```py
    scala> val userMap = userIndex.collectAsMap

    ```

我们将需要`songMap`和`userMap`来替换`userId`和`songId`在三元组中的值。Spark 会自动在集群上根据需要提供这两个 map。这工作得很好，但每次需要时在集群间发送它都很昂贵。

一个更好的方法是使用 Spark 的一个名为`broadcast`变量的功能。`broadcast`变量允许 Spark 作业在每个机器上缓存变量的只读副本，而不是在每个任务中发送副本。Spark 使用高效的广播算法分发广播变量，因此网络通信成本可以忽略不计。

如你所猜，`songMap`和`userMap`都是很好的候选变量，可以围绕`broadcast`变量进行包装。执行以下步骤：

1.  广播`userMap`：

    ```py
    scala> val broadcastUserMap = sc.broadcast(userMap)

    ```

1.  广播`songMap`：

    ```py
    scala> val broadcastSongMap = sc.broadcast(songMap)

    ```

1.  将`triplet`转换为数组：

    ```py
    scala> val tripArray = triplets.map(_.split("\\W+"))

    ```

1.  导入评分：

    ```py
    scala> import org.apache.spark.mllib.recommendation.Rating

    ```

1.  将`triplet`数组转换为评分对象的 RDD：

    ```py
    scala> val ratings = tripArray.map { case Array(user, song, plays) =>
     val userId = broadcastUserMap.value.getOrElse(user, 0)
     val songId = broadcastUserMap.value.getOrElse(song, 0)
     Rating(userId, songId, plays.toDouble)
    }

    ```

现在，我们的数据已经准备好进行建模和预测。

## 如何做到这一点…

1.  导入 ALS：

    ```py
    scala> import org.apache.spark.mllib.recommendation.ALS

    ```

1.  使用排名 10 和 10 次迭代的 ALS 构建模型：

    ```py
    scala> val model = ALS.trainImplicit(ratings, 10, 10)

    ```

1.  从三元组中提取用户和歌曲元组：

    ```py
    scala> val usersSongs = ratings.map( r => (r.user, r.product) )

    ```

1.  为用户和歌曲元组进行预测：

    ```py
    scala> val predictions = model.predict(usersSongs)

    ```

## 它是如何工作的…

我们的模式需要四个参数来工作，如下所示：

| 参数名称 | 描述 |
| --- | --- |
| 排名 | 模型中的潜在特征数量 |
| 迭代次数 | 此分解运行的迭代次数 |
| Lambda | 过拟合参数 |
| Alpha | 观察到的交互的相对权重 |

正如你在梯度下降的例子中所看到的，这些参数需要手动设置。我们可以尝试不同的值，但最佳值是 rank=50，iterations=30，lambda=0.00001，以及 alpha= 40。

## 还有更多...

快速测试不同参数的一种方法是在 Amazon EC2 上启动一个 Spark 集群。这让你有灵活性，可以选择一个强大的实例来快速测试这些参数。我已经创建了一个公共的 s3 存储桶 `com.infoobjects.songdata`，用于将数据拉入 Spark。

你需要遵循以下步骤从 S3 加载数据并运行 ALS：

```py
sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", "<your access key>")
sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey","<your secret key>")
val songs = sc.textFile("s3n://com.infoobjects.songdata/kaggle_songs.txt")
val users = sc.textFile("s3n://com.infoobjects.songdata/kaggle_users.txt")
val triplets = sc.textFile("s3n://com.infoobjects.songdata/kaggle_visible_evaluation_triplets.txt")
val songIndex = songs.map(_.split("\\W+")).map(v => (v(0),v(1).toInt))
val songMap = songIndex.collectAsMap
val userIndex = users.zipWithIndex.map( t => (t._1,t._2.toInt))
val userMap = userIndex.collectAsMap
val broadcastUserMap = sc.broadcast(userMap)
val broadcastSongMap = sc.broadcast(songMap)
val tripArray = triplets.map(_.split("\\W+"))
import org.apache.spark.mllib.recommendation.Rating
val ratings = tripArray.map{ v =>
 val userId: Int = broadcastUserMap.value.get(v(0)).fold(0)(num => num)
 val songId: Int = broadcastSongMap.value.get(v(1)).fold(0)(num => num)
 Rating(userId,songId,v(2).toDouble)
 }
import org.apache.spark.mllib.recommendation.ALS
val model = ALS.trainImplicit(ratings, 50, 30, 0.000001, 40)
val usersSongs = ratings.map( r => (r.user, r.product) )
val predictions =model.predict(usersSongs)

```

这些是基于 `usersSongs` 矩阵做出的预测。
