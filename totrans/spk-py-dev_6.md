# 第六章：可视化洞察和趋势

到目前为止，我们专注于从 Twitter 收集、分析和处理数据。我们已经为使用我们的数据进行可视化渲染和提取洞察与趋势做好了准备。我们将简要介绍 Python 生态系统中的可视化工具。我们将强调 Bokeh 作为渲染和查看大型数据集的强大工具。Bokeh 是 Python Anaconda Distribution 生态系统的一部分。

在本章中，我们将涵盖以下要点：

+   使用图表和词云衡量社交网络社区中的关键词和流行语

+   将社区围绕特定主题或话题增长最活跃的位置进行映射

# 重新审视数据密集型应用架构

我们已经到达了数据密集型应用架构的最后一层：参与层。这一层专注于如何综合、强调和可视化对数据消费者相关的关键背景信息。仅仅在控制台中显示一堆数字是不够与最终用户互动的。以快速、易于消化和吸引人的方式呈现大量信息至关重要。

以下图表设置了本章重点的上下文，突出了参与层。

![重新审视数据密集型应用架构](img/B03968_06_01.jpg)

对于 Python 的绘图和可视化，我们有很多工具和库。对我们目的来说最有趣和相关的如下：

+   **Matplotlib**是 Python 绘图库的鼻祖。Matplotlib 最初是*John Hunter*的创意，他是一位开源软件倡导者，并将 Matplotlib 确立为学术和数据科学社区中最普遍的绘图库之一。Matplotlib 允许生成图表、直方图、功率谱、条形图、误差图、散点图等。示例可以在 Matplotlib 的专用网站上找到，网址为[`matplotlib.org/examples/index.html`](http://matplotlib.org/examples/index.html)。

+   由*Michael Waskom*开发的**Seaborn**是一个快速可视化统计信息的优秀库。它建立在 Matplotlib 之上，并与 Pandas 和 Python 数据堆栈（包括 Numpy）无缝集成。Seaborn 的图形画廊可以在[`stanford.edu/~mwaskom/software/seaborn/examples/index.html`](http://stanford.edu/~mwaskom/software/seaborn/examples/index.html)上查看，展示了该库的潜力。

+   **ggplot**相对较新，旨在为 Python 数据整理者提供 R 生态系统中著名的 ggplot2 的等效功能。它具有与 ggplot2 相同的视觉和感觉，并使用 Hadley Wickham 阐述的相同的图形语法。ggplot 的 Python 端口由`yhat`团队开发。更多信息可以在[`ggplot.yhathq.com`](http://ggplot.yhathq.com)找到。

+   **D3.js**是一个非常流行的、由*Mike Bostock*开发的 JavaScript 库。**D3**代表**数据驱动文档**，它利用 HTML、SVG 和 CSS 在任何现代浏览器中将数据生动化。它通过操作 DOM（文档对象模型）提供动态、强大、交互式的可视化。Python 社区迫不及待地想要将 D3 与 Matplotlib 集成。在 Jake Vanderplas 的推动下，mpld3 被创建出来，旨在将`matplotlib`带到浏览器中。示例图形托管在以下地址：[`mpld3.github.io/index.html`](http://mpld3.github.io/index.html)。

+   **Bokeh**旨在在非常大的或流式数据集上提供高性能交互性，同时利用`D3.js`的许多概念，而不必承担编写一些令人畏惧的`javascript`和`css`代码的负担。Bokeh 在浏览器中提供动态可视化，无论是否有服务器。它与 Matplotlib、Seaborn 和 ggplot 无缝集成，并在 IPython 笔记本或 Jupyter 笔记本中渲染得非常漂亮。Bokeh 由 Continuum.io 团队积极开发，是 Anaconda Python 数据栈的一个组成部分。

Bokeh 服务器提供了一个完整的、动态的绘图引擎，它从 JSON 中生成一个反应性场景图。它使用 Web 套接字来保持状态并更新 HTML5 画布，背后使用 Backbone.js 和 Coffee-script。由于 Bokeh 由 JSON 中的数据驱动，因此它为其他语言如 R、Scala 和 Julia 提供了简单的绑定。

这提供了主要绘图和可视化库的高级概述。它并不详尽。让我们转到具体可视化示例。

# 预处理数据以进行可视化

在跳入可视化之前，我们将对收集到的数据进行一些准备工作：

```py
In [16]:
# Read harvested data stored in csv in a Panda DF
import pandas as pd
csv_in = '/home/an/spark/spark-1.5.0-bin-hadoop2.6/examples/AN_Spark/data/unq_tweetstxt.csv'
pddf_in = pd.read_csv(csv_in, index_col=None, header=0, sep=';', encoding='utf-8')
In [20]:
print('tweets pandas dataframe - count:', pddf_in.count())
print('tweets pandas dataframe - shape:', pddf_in.shape)
print('tweets pandas dataframe - colns:', pddf_in.columns)
('tweets pandas dataframe - count:', Unnamed: 0    7540
id            7540
created_at    7540
user_id       7540
user_name     7538
tweet_text    7540
dtype: int64)
('tweets pandas dataframe - shape:', (7540, 6))
('tweets pandas dataframe - colns:', Index([u'Unnamed: 0', u'id', u'created_at', u'user_id', u'user_name', u'tweet_text'], dtype='object'))
```

为了我们的可视化活动，我们将使用包含 7,540 条推文的数据库。关键信息存储在`tweet_text`列中。我们通过在数据框上调用`head()`函数来预览存储在数据框中的数据：

```py
In [21]:
pddf_in.head()
Out[21]:
  Unnamed: 0   id   created_at   user_id   user_name   tweet_text
0   0   638830426971181057   Tue Sep 01 21:46:57 +0000 2015   3276255125   True Equality   ernestsgantt: BeyHiveInFrance: 9_A_6: dreamint...
1   1   638830426727911424   Tue Sep 01 21:46:57 +0000 2015   3276255125   True Equality   ernestsgantt: BeyHiveInFrance: PhuketDailyNews...
2   2   638830425402556417   Tue Sep 01 21:46:56 +0000 2015   3276255125   True Equality   ernestsgantt: BeyHiveInFrance: 9_A_6: ernestsg...
3   3   638830424563716097   Tue Sep 01 21:46:56 +0000 2015   3276255125   True Equality   ernestsgantt: BeyHiveInFrance: PhuketDailyNews...
4   4   638830422256816132   Tue Sep 01 21:46:56 +0000 2015   3276255125   True Equality   ernestsgantt: elsahel12: 9_A_6: dreamintention...
```

现在，我们将创建一些实用函数来清理推文文本并解析 Twitter 日期。首先，我们导入 Python 正则表达式库`re`和用于解析日期和时间的`time`库：

```py
In [72]:
import re
import time
```

我们创建一个正则表达式字典，该字典将被编译，然后作为函数传递：

+   **RT**：第一个以键`RT`为关键字的正则表达式在推文文本的开头寻找关键字`RT`：

    ```py
    re.compile(r'^RT'),
    ```

+   **ALNUM**：第二个以键`ALNUM`为关键字的正则表达式在推文文本中寻找以`@`符号开头的包含字母数字字符和下划线符号的单词：

    ```py
    re.compile(r'(@[a-zA-Z0-9_]+)'),
    ```

+   **HASHTAG**：第三个以键`HASHTAG`为关键字的正则表达式在推文文本中寻找以`#`符号开头的包含字母数字字符的单词：

    ```py
    re.compile(r'(#[\w\d]+)'),
    ```

+   **SPACES**：第四个以键`SPACES`为关键字的正则表达式在推文文本中寻找空白或行空间字符：

    ```py
    re.compile(r'\s+'), 
    ```

+   **URL**：第五个以键`URL`为关键字的正则表达式在推文文本中寻找以`https://`或`http://`标记开头的包含字母数字字符的`url`地址：

    ```py
    re.compile(r'([https://|http://]?[a-zA-Z\d\/]+[\.]+[a-zA-Z\d\/\.]+)')
    In [24]:
    regexp = {"RT": "^RT", "ALNUM": r"(@[a-zA-Z0-9_]+)",
              "HASHTAG": r"(#[\w\d]+)", "URL": r"([https://|http://]?[a-zA-Z\d\/]+[\.]+[a-zA-Z\d\/\.]+)",
              "SPACES":r"\s+"}
    regexp = dict((key, re.compile(value)) for key, value in regexp.items())
    In [25]:
    regexp
    Out[25]:
    {'ALNUM': re.compile(r'(@[a-zA-Z0-9_]+)'),
     'HASHTAG': re.compile(r'(#[\w\d]+)'),
     'RT': re.compile(r'^RT'),
     'SPACES': re.compile(r'\s+'),
     'URL': re.compile(r'([https://|http://]?[a-zA-Z\d\/]+[\.]+[a-zA-Z\d\/\.]+)')}
    ```

我们创建一个实用函数来识别推文是转发推文还是原始推文：

```py
In [77]:
def getAttributeRT(tweet):
    """ see if tweet is a RT """
    return re.search(regexp["RT"], tweet.strip()) != None
```

然后，我们提取推文中的所有用户名：

```py
def getUserHandles(tweet):
    """ given a tweet we try and extract all user handles"""
    return re.findall(regexp["ALNUM"], tweet)
```

我们还提取推文中的所有标签：

```py
def getHashtags(tweet):
    """ return all hashtags"""
    return re.findall(regexp["HASHTAG"], tweet)
```

按如下方式提取推文中的所有 URL 链接：

```py
def getURLs(tweet):
    """ URL : [http://]?[\w\.?/]+"""
    return re.findall(regexp["URL"], tweet)
```

我们从推文文本中剥离所有以`@`符号开头的 URL 链接和用户名。这个函数将成为我们即将构建的词云的基础：

```py
def getTextNoURLsUsers(tweet):
    """ return parsed text terms stripped of URLs and User Names in tweet text
        ' '.join(re.sub("(@[A-Za-z0-9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()) """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)"," ", tweet).lower().split())
```

我们标记数据，以便我们可以为词云创建数据集组：

```py
def setTag(tweet):
    """ set tags to tweet_text based on search terms from tags_list"""
    tags_list = ['spark', 'python', 'clinton', 'trump', 'gaga', 'bieber']
    lower_text = tweet.lower()
    return filter(lambda x:x.lower() in lower_text,tags_list)
```

我们以`yyyy-mm-dd hh:mm:ss`格式解析推文的日期：

```py
def decode_date(s):
    """ parse Twitter date into format yyyy-mm-dd hh:mm:ss"""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(s,'%a %b %d %H:%M:%S +0000 %Y'))
```

在处理之前预览数据：

```py
In [43]:
pddf_in.columns
Out[43]:
Index([u'Unnamed: 0', u'id', u'created_at', u'user_id', u'user_name', u'tweet_text'], dtype='object')
In [45]:
# df.drop([Column Name or list],inplace=True,axis=1)
pddf_in.drop(['Unnamed: 0'], inplace=True, axis=1)
In [46]:
pddf_in.head()
Out[46]:
  id   created_at   user_id   user_name   tweet_text
0   638830426971181057   Tue Sep 01 21:46:57 +0000 2015   3276255125   True Equality   ernestsgantt: BeyHiveInFrance: 9_A_6: dreamint...
1   638830426727911424   Tue Sep 01 21:46:57 +0000 2015   3276255125   True Equality   ernestsgantt: BeyHiveInFrance: PhuketDailyNews...
2   638830425402556417   Tue Sep 01 21:46:56 +0000 2015   3276255125   True Equality   ernestsgantt: BeyHiveInFrance: 9_A_6: ernestsg...
3   638830424563716097   Tue Sep 01 21:46:56 +0000 2015   3276255125   True Equality   ernestsgantt: BeyHiveInFrance: PhuketDailyNews...
4   638830422256816132   Tue Sep 01 21:46:56 +0000 2015   3276255125   True Equality   ernestsgantt: elsahel12: 9_A_6: dreamintention...
```

我们通过应用描述的实用函数创建新的 dataframe 列，创建一个新列用于`htag`、用户名、URL、从 URL 中剥离的文本术语、不需要的字符和标签。最后解析日期：

```py
In [82]:
pddf_in['htag'] = pddf_in.tweet_text.apply(getHashtags)
pddf_in['user_handles'] = pddf_in.tweet_text.apply(getUserHandles)
pddf_in['urls'] = pddf_in.tweet_text.apply(getURLs)
pddf_in['txt_terms'] = pddf_in.tweet_text.apply(getTextNoURLsUsers)
pddf_in['search_grp'] = pddf_in.tweet_text.apply(setTag)
pddf_in['date'] = pddf_in.created_at.apply(decode_date)
```

以下代码提供了新生成的 dataframe 的快速快照：

```py
In [83]:
pddf_in[2200:2210]
Out[83]:
  id   created_at   user_id   user_name   tweet_text   htag   urls   ptxt   tgrp   date   user_handles   txt_terms   search_grp
2200   638242693374681088   Mon Aug 31 06:51:30 +0000 2015   19525954   CENATIC   El impacto de @ApacheSpark en el procesamiento...   [#sparkSpecial]   [://t.co/4PQmJNuEJB]   el impacto de en el procesamiento de datos y e...   [spark]   2015-08-31 06:51:30   [@ApacheSpark]   el impacto de en el procesamiento de datos y e...   [spark]
2201   638238014695575552   Mon Aug 31 06:32:55 +0000 2015   51115854   Nawfal   Real Time Streaming with Apache Spark\nhttp://...   [#IoT, #SmartMelboune, #BigData, #Apachespark]   [://t.co/GW5PaqwVab]   real time streaming with apache spark iot smar...   [spark]   2015-08-31 06:32:55   []   real time streaming with apache spark iot smar...   [spark]
2202   638236084124516352   Mon Aug 31 06:25:14 +0000 2015   62885987   Mithun Katti   RT @differentsachin: Spark the flame of digita...   [#IBMHackathon, #SparkHackathon, #ISLconnectIN...   []   spark the flame of digital india ibmhackathon ...   [spark]   2015-08-31 06:25:14   [@differentsachin, @ApacheSpark]   spark the flame of digital india ibmhackathon ...   [spark]
2203   638234734649176064   Mon Aug 31 06:19:53 +0000 2015   140462395   solaimurugan v   Installing @ApacheMahout with @ApacheSpark 1.4...   []   [1.4.1, ://t.co/3c5dGbfaZe.]   installing with 1 4 1 got many more issue whil...   [spark]   2015-08-31 06:19:53   [@ApacheMahout, @ApacheSpark]   installing with 1 4 1 got many more issue whil...   [spark]
2204   638233517307072512   Mon Aug 31 06:15:02 +0000 2015   2428473836   Ralf Heineke   RT @RomeoKienzler: Join me @velocityconf on #m...   [#machinelearning, #devOps, #Bl]   [://t.co/U5xL7pYEmF]   join me on machinelearning based devops operat...   [spark]   2015-08-31 06:15:02   [@RomeoKienzler, @velocityconf, @ApacheSpark]   join me on machinelearning based devops operat...   [spark]
2205   638230184848687106   Mon Aug 31 06:01:48 +0000 2015   289355748   Akim Boyko   RT @databricks: Watch live today at 10am PT is...   []   [1.5, ://t.co/16cix6ASti]   watch live today at 10am pt is 1 5 presented b...   [spark]   2015-08-31 06:01:48   [@databricks, @ApacheSpark, @databricks, @pwen...   watch live today at 10am pt is 1 5 presented b...   [spark]
2206   638227830443110400   Mon Aug 31 05:52:27 +0000 2015   145001241   sachin aggarwal   Spark the flame of digital India @ #IBMHackath...   [#IBMHackathon, #SparkHackathon, #ISLconnectIN...   [://t.co/C1AO3uNexe]   spark the flame of digital india ibmhackathon ...   [spark]   2015-08-31 05:52:27   [@ApacheSpark]   spark the flame of digital india ibmhackathon ...   [spark]
2207   638227031268810752   Mon Aug 31 05:49:16 +0000 2015   145001241   sachin aggarwal   RT @pravin_gadakh: Imagine, innovate and Igni...   [#IBMHackathon, #ISLconnectIN2015]   []   gadakh imagine innovate and ignite digital ind...   [spark]   2015-08-31 05:49:16   [@pravin_gadakh, @ApacheSpark]   gadakh imagine innovate and ignite digital ind...   [spark]
2208   638224591920336896   Mon Aug 31 05:39:35 +0000 2015   494725634   IBM Asia Pacific   RT @sachinparmar: Passionate about Spark?? Hav...   [#IBMHackathon, #ISLconnectIN]   [India..]   passionate about spark have dreams of clean sa...   [spark]   2015-08-31 05:39:35   [@sachinparmar]   passionate about spark have dreams of clean sa...   [spark]
2209   638223327467692032   Mon Aug 31 05:34:33 +0000 2015   3158070968   Open Source India   "Game Changer" #ApacheSpark speeds up #bigdata...   [#ApacheSpark, #bigdata]   [://t.co/ieTQ9ocMim]   game changer apachespark speeds up bigdata pro...   [spark]   2015-08-31 05:34:33   []   game changer apachespark speeds up bigdata pro...   [spark]
```

我们将处理后的信息保存为 CSV 格式。我们有 7,540 条记录和 13 列。在你的情况下，输出将根据你选择的数据集而变化：

```py
In [84]:
f_name = '/home/an/spark/spark-1.5.0-bin-hadoop2.6/examples/AN_Spark/data/unq_tweets_processed.csv'
pddf_in.to_csv(f_name, sep=';', encoding='utf-8', index=False)
In [85]:
pddf_in.shape
Out[85]:
(7540, 13)
```

# 一眼就能判断词汇、情绪和梗

我们现在已准备好开始构建词云，这将让我们感受到这些推文中携带的重要词汇。我们将为收集到的数据集创建词云。词云提取单词列表中的顶级单词，并创建一个散点图，其中单词的大小与其频率相关。在数据集中单词越频繁，词云渲染中的字体大小就越大。它们包括三个非常不同的主题和两个竞争或类似实体。我们的第一个主题显然是数据处理和分析，Apache Spark 和 Python 是我们的实体。我们的第二个主题是 2016 年总统选举活动，两位竞争者是希拉里·克林顿和唐纳德·特朗普。我们的最后一个主题是流行音乐界，两位代表是贾斯汀·比伯和 Lady Gaga。

## 设置词云

我们将通过分析与 Spark 相关的推文来展示编程步骤。我们加载数据并预览 dataframe：

```py
In [21]:
import pandas as pd
csv_in = '/home/an/spark/spark-1.5.0-bin-hadoop2.6/examples/AN_Spark/data/spark_tweets.csv'
tspark_df = pd.read_csv(csv_in, index_col=None, header=0, sep=',', encoding='utf-8')
In [3]:
tspark_df.head(3)
Out[3]:
  id   created_at   user_id   user_name   tweet_text   htag   urls   ptxt   tgrp   date   user_handles   txt_terms   search_grp
0   638818911773856000   Tue Sep 01 21:01:11 +0000 2015   2511247075   Noor Din   RT @kdnuggets: R leads RapidMiner, Python catc...   [#KDN]   [://t.co/3bsaTT7eUs]   r leads rapidminer python catches up big data ...   [spark, python]   2015-09-01 21:01:11   [@kdnuggets]   r leads rapidminer python catches up big data ...   [spark, python]
1   622142176768737000   Fri Jul 17 20:33:48 +0000 2015   24537879   IBM Cloudant   Be one of the first to sign-up for IBM Analyti...   [#ApacheSpark, #SparkInsight]   [://t.co/C5TZpetVA6, ://t.co/R1L29DePaQ]   be one of the first to sign up for ibm analyti...   [spark]   2015-07-17 20:33:48   []   be one of the first to sign up for ibm analyti...   [spark]
2   622140453069169000   Fri Jul 17 20:26:57 +0000 2015   515145898   Arno Candel   Nice article on #apachespark, #hadoop and #dat...   [#apachespark, #hadoop, #datascience]   [://t.co/IyF44pV0f3]   nice article on apachespark hadoop and datasci...   [spark]   2015-07-17 20:26:57   [@h2oai]   nice article on apachespark hadoop and datasci...   [spark]
```

### 注意

我们将使用的词云库是由 Andreas Mueller 开发的，托管在他的 GitHub 账户上，网址为[`github.com/amueller/word_cloud`](https://github.com/amueller/word_cloud)。

该库需要**PIL**（即**Python Imaging Library**）。可以通过调用`conda install pil`轻松安装 PIL。PIL 是一个复杂的库，安装起来比较麻烦，并且尚未移植到 Python 3.4，因此我们需要运行 Python 2.7+环境才能看到我们的词云：

```py
#
# Install PIL (does not work with Python 3.4)
#
an@an-VB:~$ conda install pil

Fetching package metadata: ....
Solving package specifications: ..................
Package plan for installation in environment /home/an/anaconda:
```

以下包将被下载：

```py
    package                    |            build
    ---------------------------|-----------------
    libpng-1.6.17              |                0         214 KB
    freetype-2.5.5             |                0         2.2 MB
    conda-env-2.4.4            |           py27_0          24 KB
    pil-1.1.7                  |           py27_2         650 KB
    ------------------------------------------------------------
                                           Total:         3.0 MB
```

以下包将被更新：

```py
    conda-env: 2.4.2-py27_0 --> 2.4.4-py27_0
    freetype:  2.5.2-0      --> 2.5.5-0     
    libpng:    1.5.13-1     --> 1.6.17-0    
    pil:       1.1.7-py27_1 --> 1.1.7-py27_2

Proceed ([y]/n)? y
```

接下来，我们安装词云库：

```py
#
# Install wordcloud
# Andreas Mueller
# https://github.com/amueller/word_cloud/blob/master/wordcloud/wordcloud.py
#

an@an-VB:~$ pip install wordcloud
Collecting wordcloud
  Downloading wordcloud-1.1.3.tar.gz (163kB)
    100% |████████████████████████████████| 163kB 548kB/s 
Building wheels for collected packages: wordcloud
  Running setup.py bdist_wheel for wordcloud
  Stored in directory: /home/an/.cache/pip/wheels/32/a9/74/58e379e5dc614bfd9dd9832d67608faac9b2bc6c194d6f6df5
Successfully built wordcloud
Installing collected packages: wordcloud
Successfully installed wordcloud-1.1.3
```

## 创建词云

在此阶段，我们已准备好使用从推文文本生成的术语列表调用词云程序。

让我们从调用`%matplotlib` inline 开始，以便在我们的笔记本中显示词云：

```py
In [4]:
%matplotlib inline
In [11]:
```

我们将 dataframe `txt_terms` 列转换为单词列表。我们确保它全部转换为 `str` 类型，以避免任何意外，并检查列表的前四条记录：

```py
len(tspark_df['txt_terms'].tolist())
Out[11]:
2024
In [22]:
tspark_ls_str = [str(t) for t in tspark_df['txt_terms'].tolist()]
In [14]:
len(tspark_ls_str)
Out[14]:
2024
In [15]:
tspark_ls_str[:4]
Out[15]:
['r leads rapidminer python catches up big data tools grow spark ignites kdn',
 'be one of the first to sign up for ibm analytics for apachespark today sparkinsight',
 'nice article on apachespark hadoop and datascience',
 'spark 101 running spark and mapreduce together in production hadoopsummit2015 apachespark altiscale']
```

我们首先调用 Matplotlib 和 wordcloud 库：

```py
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
```

从输入的术语列表中，我们创建一个由空格分隔的统一字符串，作为词云程序的输入。词云程序会移除停用词：

```py
# join tweets to a single string
words = ' '.join(tspark_ls_str)

# create wordcloud 
wordcloud = WordCloud(
                      # remove stopwords
                      stopwords=STOPWORDS,
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(words)

# render wordcloud image
plt.imshow(wordcloud)
plt.axis('off')

# save wordcloud image on disk
plt.savefig('./spark_tweets_wordcloud_1.png', dpi=300)

# display image in Jupyter notebook
plt.show()
```

在这里，我们可以可视化 Apache Spark 和 Python 的词云。显然，在 Spark 的情况下，*Hadoop*、*big data* 和 *analytics* 是热门话题，而 Python 则回忆起其名称的根源 Monty Python，重点在于 *developer*、*apache spark* 和编程，并有一些关于 java 和 ruby 的提示。

![创建词云](img/B03968_06_02.jpg)

我们还可以从以下词云中一瞥占据北美 2016 年总统选举候选人注意力的词汇：希拉里·克林顿和唐纳德·特朗普。显然，希拉里·克林顿被她的对手唐纳德·特朗普和伯尼·桑德斯的存在所掩盖，而特朗普则主要集中在他自己身上：

![创建词云](img/B03968_06_03.jpg)

有趣的是，在贾斯汀·比伯和 Lady Gaga 的情况下，出现了单词 *love*。在比伯的情况下，*follow* 和 *belieber* 是关键词，而 *diet*、*weight loss* 和 *fashion* 是 Lady Gaga 粉丝团的关注点。

![创建词云](img/B03968_06_04.jpg)

# 定位推文和绘制聚会地图

现在，我们将深入探讨使用 Bokeh 创建交互式地图的过程。首先，我们创建一个世界地图，在这个地图上我们定位样本推文，并将鼠标移动到这些位置时，我们可以在悬停框中看到用户及其相应的推文。

第二个地图专注于伦敦即将举行的聚会。这可能是一个交互式地图，它将作为特定城市即将举行的聚会的日期、时间和地点的提醒。

## 定位推文

目标是创建一个世界地图散点图，显示地图上重要推文的位置，并在悬停这些点时显示推文和作者。我们将通过三个步骤来构建这个交互式可视化：

1.  首先加载一个包含所有世界国家边界及其相应经纬度的字典，以创建背景世界地图。

1.  加载我们希望定位的推文及其相应的坐标和作者。

1.  最后，在世界地图上散点图推文的坐标，并激活悬停工具，以交互式地可视化地图上突出显示的点上的推文和作者。

在第一步中，我们创建一个名为 data 的 Python 列表，它将包含所有世界国家边界及其相应的纬度和经度：

```py
In [4]:
#
# This module exposes geometry data for World Country Boundaries.
#
import csv
import codecs
import gzip
import xml.etree.cElementTree as et
import os
from os.path import dirname, join

nan = float('NaN')
__file__ = os.getcwd()

data = {}
with gzip.open(join(dirname(__file__), 'AN_Spark/data/World_Country_Boundaries.csv.gz')) as f:
    decoded = codecs.iterdecode(f, "utf-8")
    next(decoded)
    reader = csv.reader(decoded, delimiter=',', quotechar='"')
    for row in reader:
        geometry, code, name = row
        xml = et.fromstring(geometry)
        lats = []
        lons = []
        for i, poly in enumerate(xml.findall('.//outerBoundaryIs/LinearRing/coordinates')):
            if i > 0:
                lats.append(nan)
                lons.append(nan)
            coords = (c.split(',')[:2] for c in poly.text.split())
            lat, lon = list(zip(*[(float(lat), float(lon)) for lon, lat in
                coords]))
            lats.extend(lat)
            lons.extend(lon)
        data[code] = {
            'name'   : name,
            'lats'   : lats,
            'lons'   : lons,
        }
In [5]:
len(data)
Out[5]:
235
```

在第二步中，我们加载一组重要的样本推文，我们希望用它们各自的地理定位信息来可视化：

```py
In [69]:
# data
#
#
In [8]:
import pandas as pd
csv_in = '/home/an/spark/spark-1.5.0-bin-hadoop2.6/examples/AN_Spark/data/spark_tweets_20.csv'
t20_df = pd.read_csv(csv_in, index_col=None, header=0, sep=',', encoding='utf-8')
In [9]:
t20_df.head(3)
Out[9]:
    id  created_at  user_id     user_name   tweet_text  htag    urls    ptxt    tgrp    date    user_handles    txt_terms   search_grp  lat     lon
0   638818911773856000  Tue Sep 01 21:01:11 +0000 2015  2511247075  Noor Din    RT @kdnuggets: R leads RapidMiner, Python catc...   [#KDN]  [://t.co/3bsaTT7eUs]    r leads rapidminer python catches up big data ...   [spark, python]     2015-09-01 21:01:11     [@kdnuggets]    r leads rapidminer python catches up big data ...   [spark, python]     37.279518   -121.867905
1   622142176768737000  Fri Jul 17 20:33:48 +0000 2015  24537879    IBM Cloudant    Be one of the first to sign-up for IBM Analyti...   [#ApacheSpark, #SparkInsight]   [://t.co/C5TZpetVA6, ://t.co/R1L29DePaQ]    be one of the first to sign up for ibm analyti...   [spark]     2015-07-17 20:33:48     []  be one of the first to sign up for ibm analyti...   [spark]     37.774930   -122.419420
2   622140453069169000  Fri Jul 17 20:26:57 +0000 2015  515145898   Arno Candel     Nice article on #apachespark, #hadoop and #dat...   [#apachespark, #hadoop, #datascience]   [://t.co/IyF44pV0f3]    nice article on apachespark hadoop and datasci...   [spark]     2015-07-17 20:26:57     [@h2oai]    nice article on apachespark hadoop and datasci...   [spark]     51.500130   -0.126305
In [98]:
len(t20_df.user_id.unique())
Out[98]:
19
In [17]:
t20_geo = t20_df[['date', 'lat', 'lon', 'user_name', 'tweet_text']]
In [24]:
# 
t20_geo.rename(columns={'user_name':'user', 'tweet_text':'text' }, inplace=True)
In [25]:
t20_geo.head(4)
Out[25]:
    date    lat     lon     user    text
0   2015-09-01 21:01:11     37.279518   -121.867905     Noor Din    RT @kdnuggets: R leads RapidMiner, Python catc...
1   2015-07-17 20:33:48     37.774930   -122.419420     IBM Cloudant    Be one of the first to sign-up for IBM Analyti...
2   2015-07-17 20:26:57     51.500130   -0.126305   Arno Candel     Nice article on #apachespark, #hadoop and #dat...
3   2015-07-17 19:35:31     51.500130   -0.126305   Ira Michael Blonder     Spark 101: Running Spark and #MapReduce togeth...
In [22]:
df = t20_geo
#
```

在第三步中，我们首先导入了所有必要的 Bokeh 库。我们将在 Jupyter Notebook 中实例化输出。我们加载了世界国家边界信息。我们获取了地理定位的推文数据。我们实例化了 Bokeh 交互式工具，如滚轮和框缩放以及悬停工具。

```py
In [29]:
#
# Bokeh Visualization of tweets on world map
#
from bokeh.plotting import *
from bokeh.models import HoverTool, ColumnDataSource
from collections import OrderedDict

# Output in Jupiter Notebook
output_notebook()

# Get the world map
world_countries = data.copy()

# Get the tweet data
tweets_source = ColumnDataSource(df)

# Create world map 
countries_source = ColumnDataSource(data= dict(
    countries_xs=[world_countries[code]['lons'] for code in world_countries],
    countries_ys=[world_countries[code]['lats'] for code in world_countries],
    country = [world_countries[code]['name'] for code in world_countries],
))

# Instantiate the bokeh interactive tools 
TOOLS="pan,wheel_zoom,box_zoom,reset,resize,hover,save"
```

我们现在准备好将收集到的各种元素层叠到一个名为**p**的对象中。定义**p**的标题、宽度和高度。附加工具。通过带有浅色背景色和边框的补丁创建世界地图背景。根据各自的地理坐标散点图绘制推文。然后，激活带有用户及其相应推文的悬停工具。最后，在浏览器上渲染图片。代码如下：

```py
# Instantiante the figure object
p = figure(
    title="%s tweets " %(str(len(df.index))),
    title_text_font_size="20pt",
    plot_width=1000,
    plot_height=600,
    tools=TOOLS)

# Create world patches background
p.patches(xs="countries_xs", ys="countries_ys", source = countries_source, fill_color="#F1EEF6", fill_alpha=0.3,
        line_color="#999999", line_width=0.5)

# Scatter plots by longitude and latitude
p.scatter(x="lon", y="lat", source=tweets_source, fill_color="#FF0000", line_color="#FF0000")
# 

# Activate hover tool with user and corresponding tweet information
hover = p.select(dict(type=HoverTool))
hover.point_policy = "follow_mouse"
hover.tooltips = OrderedDict([
    ("user", "@user"),
   ("tweet", "@text"),
])

# Render the figure on the browser
show(p)
BokehJS successfully loaded.

inspect

#
#
```

以下代码给出了带有红色点的世界地图概览，代表推文来源的位置：

![地理定位推文](img/B03968_06_05.jpg)

我们可以悬停在特定的点上，以揭示该位置上的推文：

![地理定位推文](img/B03968_06_06.jpg)

我们可以放大到特定位置：

![地理定位推文](img/B03968_06_07.jpg)

最后，我们可以揭示给定放大位置上的推文：

![地理定位推文](img/B03968_06_08.jpg)

## 在谷歌地图上显示即将举行的聚会

现在，我们的目标是关注伦敦的即将举行的聚会。我们正在绘制三个聚会**数据科学伦敦**、**Apache Spark**和**机器学习**。我们在一个 Bokeh 可视化中嵌入谷歌地图，并根据它们的坐标地理定位这三个聚会，并使用悬停工具获取每个聚会的即将举行活动的信息。

首先，导入所有必要的 Bokeh 库：

```py
In [ ]:
#
# Bokeh Google Map Visualization of London with hover on specific points
#
#
from __future__ import print_function

from bokeh.browserlib import view
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.models.glyphs import Circle
from bokeh.models import (
    GMapPlot, Range1d, ColumnDataSource,
    PanTool, WheelZoomTool, BoxSelectTool,
    HoverTool, ResetTool,
    BoxSelectionOverlay, GMapOptions)
from bokeh.resources import INLINE

x_range = Range1d()
y_range = Range1d()
```

我们将实例化作为我们的 Bokeh 可视化底层的谷歌地图：

```py
# JSON style string taken from: https://snazzymaps.com/style/1/pale-dawn
map_options = GMapOptions(lat=51.50013, lng=-0.126305, map_type="roadmap", zoom=13, styles="""
[{"featureType":"administrative","elementType":"all","stylers":[{"visibility":"on"},{"lightness":33}]},
 {"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2e5d4"}]},
 {"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#c5dac6"}]},
 {"featureType":"poi.park","elementType":"labels","stylers":[{"visibility":"on"},{"lightness":20}]},
 {"featureType":"road","elementType":"all","stylers":[{"lightness":20}]},
 {"featureType":"road.highway","elementType":"geometry","stylers":[{"color":"#c5c6c6"}]},
 {"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#e4d7c6"}]},
 {"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#fbfaf7"}]},
 {"featureType":"water","elementType":"all","stylers":[{"visibility":"on"},{"color":"#acbcc9"}]}]
""")
```

使用前一步骤中的尺寸和地图选项，从`GMapPlot`类实例化 Bokeh 对象`plot`：

```py
# Instantiate Google Map Plot
plot = GMapPlot(
    x_range=x_range, y_range=y_range,
    map_options=map_options,
    title="London Meetups"
)
```

引入我们希望绘制的三个聚会的信息，并通过悬停在相应的坐标上方来获取信息：

```py
source = ColumnDataSource(
    data=dict(
        lat=[51.49013, 51.50013, 51.51013],
        lon=[-0.130305, -0.126305, -0.120305],
        fill=['orange', 'blue', 'green'],
        name=['LondonDataScience', 'Spark', 'MachineLearning'],
        text=['Graph Data & Algorithms','Spark Internals','Deep Learning on Spark']
    )
)
```

定义要在谷歌地图上绘制的点：

```py
circle = Circle(x="lon", y="lat", size=15, fill_color="fill", line_color=None)
plot.add_glyph(source, circle)
```

定义用于此可视化的 Bokeh 工具的字符串：

```py
# TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"
pan = PanTool()
wheel_zoom = WheelZoomTool()
box_select = BoxSelectTool()
reset = ResetTool()
hover = HoverTool()
# save = SaveTool()

plot.add_tools(pan, wheel_zoom, box_select, reset, hover)
overlay = BoxSelectionOverlay(tool=box_select)
plot.add_layout(overlay)
```

激活携带信息的`hover`工具：

```py
hover = plot.select(dict(type=HoverTool))
hover.point_policy = "follow_mouse"
hover.tooltips = OrderedDict([
    ("Name", "@name"),
    ("Text", "@text"),
    ("(Long, Lat)", "(@lon, @lat)"),
])

show(plot)
```

渲染出伦敦的视图：

![在谷歌地图上显示即将举行的聚会](img/B03968_06_09.jpg)

一旦我们悬停在突出显示的点上，我们就可以获取给定聚会的信息：

![在谷歌地图上显示即将举行的聚会](img/B03968_06_10.jpg)

如以下截图所示，保留了完整的平滑缩放功能：

![在谷歌地图上显示即将举行的聚会](img/B03968_06_11.jpg)

# 摘要

在本章中，我们关注了几种可视化技术。我们看到了如何构建词云及其直观的强大功能，可以一眼揭示成千上万推文中携带的关键词、情绪和梗。

我们随后讨论了使用 Bokeh 的交互式地图可视化。我们从零开始构建了一个世界地图，并创建了一个关键推文的散点图。一旦地图在浏览器上渲染，我们就可以交互式地从一点移动到另一点，揭示来自世界各地不同部分的推文。

我们的最终可视化集中在伦敦即将举行的 Spark、数据科学和机器学习聚会及其相应主题的映射上，使用实际的谷歌地图制作了一个美丽的交互式可视化。
