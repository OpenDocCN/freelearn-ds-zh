# 第十章：推荐系统

在本章中，我们将介绍以下内容：

+   使用显式反馈的协同过滤

+   使用隐式反馈的协同过滤

# 介绍

以下是维基百科对推荐系统的定义：

> “推荐系统是信息过滤系统的一个子类，旨在预测用户对物品的‘评分’或‘偏好’。”

推荐系统近年来变得非常受欢迎。亚马逊用它们来推荐书籍，Netflix 用来推荐电影，Google 新闻用来推荐新闻故事。以下是一些推荐的影响的例子（来源：Celma，Lamere，2008）：

+   Netflix 上观看的电影有三分之二是推荐的

+   谷歌新闻点击量的 38%是推荐的

+   亚马逊销售额的 35%是推荐的结果

正如我们在前几章中看到的，特征和特征选择在机器学习算法的有效性中起着重要作用。推荐引擎算法会自动发现这些特征，称为**潜在特征**。简而言之，有一些潜在特征决定了用户喜欢一部电影而不喜欢另一部电影。如果另一个用户具有相应的潜在特征，那么这个人也很可能对电影有相似的口味。

为了更好地理解这一点，让我们看一些样本电影评分：

| 电影 | Rich | Bob | Peter | Chris |
| --- | --- | --- | --- | --- |
| *Titanic* | 5 | 3 | 5 | ? |
| *GoldenEye* | 3 | 2 | 1 | 5 |
| *Toy Story* | 1 | ? | 2 | 2 |
| *Disclosure* | 4 | 4 | ? | 4 |
| *Ace Ventura* | 4 | ? | 4 | ? |

我们的目标是预测用?符号表示的缺失条目。让我们看看是否能找到一些与电影相关的特征。首先，您将查看电影类型，如下所示：

| 电影 | 类型 |
| --- | --- |
| *Titanic* | 动作，爱情 |
| *GoldenEye* | 动作，冒险，惊悚 |
| *Toy Story* | 动画，儿童，喜剧 |
| *Disclosure* | 戏剧，惊悚 |
| *Ace Ventura* | 喜剧 |

现在每部电影可以根据每种类型进行评分，评分范围从 0 到 1。例如，*GoldenEye*不是一部主要的爱情片，所以它可能在爱情方面的评分为 0.1，但在动作方面的评分为 0.98。因此，每部电影可以被表示为一个特征向量。

### 注意

在本章中，我们将使用[grouplens.org/datasets/movielens/](http://grouplens.org/datasets/movielens/)的 MovieLens 数据集。

InfoObjects 大数据沙箱中加载了 100k 部电影评分。您还可以从 GroupLens 下载 100 万甚至高达 1000 万的评分，以便分析更大的数据集以获得更好的预测。

我们将使用这个数据集中的两个文件：

+   `u.data`：这是一个以制表符分隔的电影评分列表，格式如下：

```scala
user id | item id | rating | epoch time
```

由于我们不需要时间戳，我们将从我们的配方数据中将其过滤掉

+   `u.item`：这是一个以制表符分隔的电影列表，格式如下：

```scala
movie id | movie title | release date | video release date |               IMDb URL | unknown | Action | Adventure | Animation |               Children's | Comedy | Crime | Documentary | Drama | Fantasy |               Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |               Thriller | War | Western |
```

本章将介绍如何使用 MLlib 进行推荐，MLlib 是 Spark 的机器学习库。

# 使用显式反馈的协同过滤

协同过滤是推荐系统中最常用的技术。它有一个有趣的特性——它自己学习特征。因此，在电影评分的情况下，我们不需要提供有关电影是浪漫还是动作的实际人类反馈。

正如我们在*介绍*部分看到的，电影有一些潜在特征，比如类型，同样用户也有一些潜在特征，比如年龄，性别等。协同过滤不需要它们，并且自己找出潜在特征。

在这个例子中，我们将使用一种名为**交替最小二乘法**（**ALS**）的算法。该算法基于少量潜在特征解释电影和用户之间的关联。它使用三个训练参数：秩、迭代次数和 lambda（在本章后面解释）。找出这三个参数的最佳值的最佳方法是尝试不同的值，看哪个值的**均方根误差**（**RMSE**）最小。这个误差类似于标准差，但是它是基于模型结果而不是实际数据的。

## 准备工作

将从 GroupLens 下载的`moviedata`上传到`hdfs`中的`moviedata`文件夹：

```scala
$ hdfs dfs -put moviedata moviedata

```

我们将向这个数据库添加一些个性化评分，以便测试推荐的准确性。

你可以查看`u.item`来挑选一些电影并对其进行评分。以下是我选择的一些电影，以及我的评分。随意选择你想评分的电影并提供你自己的评分。

| 电影 ID | 电影名称 | 评分（1-5） |
| --- | --- | --- |
| 313 | *泰坦尼克号* | 5 |
| 2 | *黄金眼* | 3 |
| 1 | *玩具总动员* | 1 |
| 43 | *揭秘* | 4 |
| 67 | *玩具总动员* | 4 |
| 82 | *侏罗纪公园* | 5 |
| 96 | *终结者 2* | 5 |
| 121 | *独立日* | 4 |
| 148 | *鬼与黑暗* | 4 |

最高的用户 ID 是 943，所以我们将把新用户添加为 944。让我们创建一个新的逗号分隔的文件`p.data`，其中包含以下数据：

```scala
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

## 如何做…

1.  将个性化电影数据上传到`hdfs`：

```scala
$ hdfs dfs -put p.data p.data

```

1.  导入 ALS 和评分类：

```scala
scala> import org.apache.spark.mllib.recommendation.ALS
scala> import org.apache.spark.mllib.recommendation.Rating

```

1.  将评分数据加载到 RDD 中：

```scala
scala> val data = sc.textFile("moviedata/u.data")

```

1.  将`val data`转换为评分的 RDD：

```scala
scala> val ratings = data.map { line => 
 val Array(userId, itemId, rating, _) = line.split("\t") 
 Rating(userId.toInt, itemId.toInt, rating.toDouble) 
}

```

1.  将个性化评分数据加载到 RDD 中：

```scala
scala> val pdata = sc.textFile("p.data")

```

1.  将数据转换为个性化评分的 RDD：

```scala
scala> val pratings = pdata.map { line => 
 val Array(userId, itemId, rating) = line.split(",")
 Rating(userId.toInt, itemId.toInt, rating.toDouble) 
}

```

1.  将评分与个性化评分结合：

```scala
scala> val movieratings = ratings.union(pratings)

```

1.  使用秩为 5 和 10 次迭代以及 0.01 作为 lambda 构建 ALS 模型：

```scala
scala> val model = ALS.train(movieratings, 10, 10, 0.01)

```

1.  让我们根据这个模型预测我对给定电影的评分会是多少。

1.  让我们从原始的*终结者*开始，电影 ID 为 195：

```scala
scala> model.predict(sc.parallelize(Array((944,195)))).collect.foreach(println)
Rating(944,195,4.198642954004738)

```

由于我给*终结者 2*评了 5 分，这是一个合理的预测。

1.  让我们尝试一下*鬼*，电影 ID 为 402：

```scala
scala> model.predict(sc.parallelize(Array((944,402)))).collect.foreach(println)
Rating(944,402,2.982213836456829)

```

这是一个合理的猜测。

1.  让我们尝试一下*鬼与黑暗*，这是我已经评分的电影，ID 为 148：

```scala
scala> model.predict(sc.parallelize(Array((944,402)))).collect.foreach(println)
Rating(944,148,3.8629938805450035)

```

非常接近的预测，知道我给这部电影评了 4 分。

你可以将更多电影添加到`train`数据集中。还有 100 万和 1000 万的评分数据集可用，这将进一步完善算法。

# 使用隐式反馈的协同过滤

有时，可用的反馈不是评分的形式，而是音轨播放、观看的电影等形式。这些数据乍一看可能不如用户的明确评分好，但这更加详尽。

## 准备工作

我们将使用来自[`www.kaggle.com/c/msdchallenge/data`](http://www.kaggle.com/c/msdchallenge/data)的百万首歌数据。你需要下载三个文件：

+   `kaggle_visible_evaluation_triplets`

+   `kaggle_users.txt`

+   `kaggle_songs.txt`

现在执行以下步骤：

1.  在`hdfs`中创建一个`songdata`文件夹，并将所有三个文件放在这里：

```scala
$ hdfs dfs -mkdir songdata

```

1.  将歌曲数据上传到`hdfs`：

```scala
$ hdfs dfs -put kaggle_visible_evaluation_triplets.txt songdata/
$ hdfs dfs -put kaggle_users.txt songdata/
$ hdfs dfs -put kaggle_songs.txt songdata/

```

我们仍然需要做一些预处理。MLlib 中的 ALS 需要用户和产品 ID 都是整数。`Kaggle_songs.txt`文件有歌曲 ID 和其后的序列号，而`Kaggle_users.txt`文件没有。我们的目标是用相应的整数序列号替换`triplets`数据中的`userid`和`songid`。为此，请按照以下步骤操作：

1.  将`kaggle_songs`数据加载为 RDD：

```scala
scala> val songs = sc.textFile("songdata/kaggle_songs.txt")

```

1.  将用户数据加载为 RDD：

```scala
scala> val users = sc.textFile("songdata/kaggle_users.txt")

```

1.  将三元组（用户、歌曲、播放次数）数据加载为 RDD：

```scala
scala> val triplets = sc.textFile("songdata/kaggle_visible_evaluation_triplets.txt")

```

1.  将歌曲数据转换为`PairRDD`：

```scala
scala> val songIndex = songs.map(_.split("\\W+")).map(v => (v(0),v(1).toInt))

```

1.  收集`songIndex`作为 Map：

```scala
scala> val songMap = songIndex.collectAsMap

```

1.  将用户数据转换为`PairRDD`：

```scala
scala> val userIndex = users.zipWithIndex.map( t => (t._1,t._2.toInt))

```

1.  收集`userIndex`作为 Map：

```scala
scala> val userMap = userIndex.collectAsMap

```

我们需要`songMap`和`userMap`来替换三元组中的`userId`和`songId`。Spark 会根据需要自动在集群上提供这两个映射。这样做效果很好，但每次需要发送到集群时都很昂贵。

更好的方法是使用 Spark 的一个特性叫做`broadcast`变量。`broadcast`变量允许 Spark 作业在每台机器上保留一个只读副本的变量缓存，而不是在每个任务中传输一个副本。Spark 使用高效的广播算法来分发广播变量，因此网络上的通信成本可以忽略不计。

正如你可以猜到的，`songMap`和`userMap`都是很好的候选对象，可以包装在`broadcast`变量周围。执行以下步骤：

1.  广播`userMap`：

```scala
scala> val broadcastUserMap = sc.broadcast(userMap)

```

1.  广播`songMap`：

```scala
scala> val broadcastSongMap = sc.broadcast(songMap)

```

1.  将`triplet`转换为数组：

```scala
scala> val tripArray = triplets.map(_.split("\\W+"))

```

1.  导入评分：

```scala
scala> import org.apache.spark.mllib.recommendation.Rating

```

1.  将`triplet`数组转换为评分对象的 RDD：

```scala
scala> val ratings = tripArray.map { case Array(user, song, plays) =>
 val userId = broadcastUserMap.value.getOrElse(user, 0)
 val songId = broadcastUserMap.value.getOrElse(song, 0)
 Rating(userId, songId, plays.toDouble)
}

```

现在，我们的数据已经准备好进行建模和预测。

## 如何做…

1.  导入 ALS：

```scala
scala> import org.apache.spark.mllib.recommendation.ALS

```

1.  使用 ALS 构建一个具有 rank 10 和 10 次迭代的模型：

```scala
scala> val model = ALS.trainImplicit(ratings, 10, 10)

```

1.  从三元组中提取用户和歌曲元组：

```scala
scala> val usersSongs = ratings.map( r => (r.user, r.product) )

```

1.  为用户和歌曲元组做出预测：

```scala
scala> val predictions = model.predict(usersSongs)

```

## 它是如何工作的…

我们的模型需要四个参数才能工作，如下所示：

| 参数名称 | 描述 |
| --- | --- |
| Rank | 模型中的潜在特征数 |
| Iterations | 用于运行此因子分解的迭代次数 |
| Lambda | 过拟合参数 |
| Alpha | 观察交互的相对权重 |

正如你在梯度下降的情况下看到的，这些参数需要手动设置。我们可以尝试不同的值，但最好的值是 rank=50，iterations=30，lambda=0.00001，alpha=40。

## 还有更多…

快速测试不同参数的一种方法是在 Amazon EC2 上生成一个 Spark 集群。这样可以灵活地选择一个强大的实例来快速测试这些参数。我已经创建了一个名为`com.infoobjects.songdata`的公共 s3 存储桶，以便将数据传输到 Spark。

以下是您需要遵循的步骤，从 S3 加载数据并运行 ALS：

```scala
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

这些是在`usersSongs`矩阵上做出的预测。
