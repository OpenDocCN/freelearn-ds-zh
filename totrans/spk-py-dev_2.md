# 第二章. 使用 Spark 构建批处理和流式应用程序

本书的目标是通过构建一个分析社交网络上 Spark 社区互动的应用程序来向您介绍 PySpark 和 PyData 库。我们将从 GitHub 收集有关 Apache Spark 的信息，检查 Twitter 上的相关推文，并使用**Meetup**来感受更广泛的开源软件社区中 Spark 的热潮。

在本章中，我们将概述各种数据和信息来源。我们将了解它们的结构。我们将概述数据处理管道，从收集到批处理和流处理。

在本节中，我们将涵盖以下内容：

+   从收集到批处理和流处理概述数据处理管道，有效地描绘我们计划构建的应用程序的架构。

+   检查各种数据源（GitHub、Twitter 和 Meetup），它们的数据结构（JSON、结构化信息、非结构化文本、地理位置、时间序列数据等），以及它们的复杂性。我们还讨论了连接到三个不同 API 的工具，以便您可以构建自己的数据混合。本书将在以下章节中专注于 Twitter。

# 构建数据密集型应用程序的架构

我们在上一章中定义了数据密集型应用框架架构蓝图。现在，让我们将本书中将要使用的各种软件组件放回到我们的原始框架中。以下是数据密集型架构框架中映射的软件各种组件的示意图：

![构建数据密集型应用程序的架构](img/B03986_02_01.jpg)

Spark 是一个极其高效的分布式计算框架。为了充分利用其全部功能，我们需要相应地构建我们的解决方案。出于性能原因，整体解决方案还需要了解其在 CPU、存储和网络方面的使用情况。

这些必要性驱动着我们解决方案的架构：

+   **延迟**：此架构结合了慢速和快速处理。慢速处理在批处理模式下对历史数据进行。这也被称为静态数据。此阶段构建的预计算模型和数据模式将在实时连续数据输入系统后由快速处理臂使用。数据处理或流数据的实时分析指的是运动中的数据。静态数据基本上是以较长的延迟在批处理模式下处理的数据。运动中的数据指的是实时摄入数据的流计算。

+   **可扩展性**：Spark 通过其分布式内存计算框架原生动地线性可扩展。与 Spark 交互的数据库和数据存储也需要能够随着数据量的增长而线性扩展。

+   **容错性**：当由于硬件、软件或网络原因发生故障时，架构应该足够弹性，并始终提供可用性。

+   **灵活性**：在这个架构中实施的数据管道可以根据用例快速适应和改造。

Spark 独特之处在于它允许在同一统一平台上进行批量处理和流式分析。

我们将考虑两个数据处理管道：

+   第一个处理静态数据，专注于构建数据的批量分析管道

+   第二个是动态数据，它针对实时数据摄取，并基于预计算模型和数据模式提供洞察

## 静态数据处理

让我们了解静态数据或批量处理管道。在这个管道中的目标是摄取来自 Twitter、GitHub 和 Meetup 的各种数据集；为 Spark MLlib 机器学习引擎准备数据；并推导出将在批量模式或实时模式下应用的基本模型，以生成洞察。

以下图表说明了数据管道，以实现静态数据的处理：

![静态数据处理](img/B03986_02_02.jpg)

## 动态数据处理

动态数据处理引入了新的复杂性级别，因为我们引入了新的失败可能性。如果我们想要扩展，我们需要考虑引入分布式消息队列系统，如 Kafka。我们将在下一章中专门讨论理解流式分析。

以下图表描述了处理动态数据的管道：

![动态数据处理](img/B03986_02_03.jpg)

## 交互式探索数据

构建一个数据密集型应用并不像将数据库暴露给网络界面那样简单。在设置静态数据和动态数据处理的过程中，我们将利用 Spark 分析数据交互式的能力，以及优化机器学习和流式活动所需的数据丰富性和质量。在这里，我们将通过数据收集、精炼和调查的迭代周期，以获取我们应用感兴趣的数据集。

# 连接到社交网络

让我们深入了解数据密集型应用架构集成层的第一步。我们将专注于收集数据，确保其完整性，并为 Spark 在下一阶段进行批量和流式数据处理做准备。这一阶段在以下五个处理步骤中描述：*连接*、*纠正*、*收集*、*组合*和*消费*。这些是数据探索的迭代步骤，将使我们熟悉数据，并帮助我们优化数据结构以进行进一步处理。

以下图表描述了数据获取和精炼的迭代过程，以便进行消费：

![连接到社交网络](img/B03986_02_04.jpg)

我们将连接到感兴趣的社交网络：Twitter、GitHub 和 Meetup。我们将讨论访问 **API**（即 **应用程序编程接口**）的方式，以及如何在遵守社交网络施加的速率限制的同时，与这些服务建立 RESTful 连接。**REST**（即 **表示状态转移**）是互联网上最广泛采用的架构风格，旨在实现可扩展的 Web 服务。它依赖于主要在 **JSON**（即 **JavaScript 对象表示法**）中交换消息。RESTful API 和 Web 服务实现了四个最常用的动词 `GET`、`PUT`、`POST` 和 `DELETE`。`GET` 用于从给定的 `URI` 中检索元素或集合。`PUT` 更新集合为新集合。`POST` 允许创建新条目，而 `DELETE` 则删除集合。

## 获取 Twitter 数据

Twitter 允许在 OAuth 授权协议下访问其搜索和流推文服务，OAuth 允许 API 应用程序代表用户安全地执行操作。为了创建连接，第一步是在 Twitter 上创建一个应用程序，网址为 [`apps.twitter.com/app/new`](https://apps.twitter.com/app/new)。

![获取 Twitter 数据](img/B03986_02_05.jpg)

一旦应用程序创建完成，Twitter 将会发放四个代码，允许它接入 Twitter 流。

```py
CONSUMER_KEY = 'GetYourKey@Twitter'
CONSUMER_SECRET = ' GetYourKey@Twitter'
OAUTH_TOKEN = ' GetYourToken@Twitter'
OAUTH_TOKEN_SECRET = ' GetYourToken@Twitter'
```

如果您想了解提供的各种 RESTful 查询，您可以在 [`dev.twitter.com/rest/tools/console`](https://dev.twitter.com/rest/tools/console) 的开发者控制台中探索 Twitter API。

![获取 Twitter 数据](img/B03986_02_06.jpg)

我们将使用以下代码在 Twitter 上建立程序连接，这将激活我们的 OAuth 访问权限，并允许我们在速率限制下接入 Twitter API。在流模式中，限制是针对 GET 请求的。

## 获取 GitHub 数据

GitHub 使用与 Twitter 类似的认证过程。前往开发者网站，在 GitHub 上注册后获取您的凭证，注册地址为 [`developer.github.com/v3/`](https://developer.github.com/v3/)：

![获取 GitHub 数据](img/B03986_02_07.jpg)

## 获取 Meetup 数据

Meetup 可以通过 Meetup.com 开发者资源中发放的令牌访问。获取 Meetup API 访问所需的令牌或 OAuth 凭证，可以在他们的开发者网站上找到，网址为 [`secure.meetup.com/meetup_api`](https://secure.meetup.com/meetup_api)：

![获取 Meetup 数据](img/B03986_02_08.jpg)

# 分析数据

让我们先感受一下从每个社交网络中提取的数据，并了解这些来源的数据结构。

## 探索推文的解剖结构

在本节中，我们将与 Twitter API 建立连接。Twitter 提供两种连接模式：REST API，允许我们搜索给定搜索词或标签的历史推文，以及流式 API，在现有速率限制下提供实时推文。

为了更好地理解如何操作 Twitter API，我们将进行以下步骤：

1.  安装 Twitter Python 库。

1.  通过 OAuth 以编程方式建立连接，这是 Twitter 所需的认证。

1.  搜索查询词 *Apache Spark* 的最近推文并探索获得的结果。

1.  确定感兴趣的关键属性并从 JSON 输出中检索信息。

让我们一步一步来：

1.  安装 Python Twitter 库。为了安装它，您需要从命令行写入 `pip install twitter`：

    ```py
    $ pip install twitter

    ```

1.  创建 Python Twitter API 类及其用于认证、搜索和解析结果的基类方法。`self.auth` 从 Twitter 获取凭证。然后创建一个注册的 API 作为 `self.api`。我们实现了两个方法：第一个是使用给定的查询搜索 Twitter，第二个是解析输出以检索相关信息，如推文 ID、推文文本和推文作者。代码如下：

    ```py
    import twitter
    import urlparse
    from pprint import pprint as pp

    class TwitterAPI(object):
        """
        TwitterAPI class allows the Connection to Twitter via OAuth
        once you have registered with Twitter and receive the 
        necessary credentiials 
        """

    # initialize and get the twitter credentials
         def __init__(self): 
            consumer_key = 'Provide your credentials'
            consumer_secret = 'Provide your credentials'
            access_token = 'Provide your credentials'
            access_secret = 'Provide your credentials'

            self.consumer_key = consumer_key
            self.consumer_secret = consumer_secret
            self.access_token = access_token
            self.access_secret = access_secret

    #
    # authenticate credentials with Twitter using OAuth
            self.auth = twitter.oauth.OAuth(access_token, access_secret, consumer_key, consumer_secret)
        # creates registered Twitter API
            self.api = twitter.Twitter(auth=self.auth)
    #
    # search Twitter with query q (i.e. "ApacheSpark") and max. result
        def searchTwitter(self, q, max_res=10,**kwargs):
            search_results = self.api.search.tweets(q=q, count=10, **kwargs)
            statuses = search_results['statuses']
            max_results = min(1000, max_res)

            for _ in range(10): 
                try:
                    next_results = search_results['search_metadata']['next_results']
                except KeyError as e: 
                    break

                next_results = urlparse.parse_qsl(next_results[1:])
                kwargs = dict(next_results)
                search_results = self.api.search.tweets(**kwargs)
                statuses += search_results['statuses']

                if len(statuses) > max_results: 
                    break
            return statuses
    #
    # parse tweets as it is collected to extract id, creation 
    # date, user id, tweet text
        def parseTweets(self, statuses):
            return [ (status['id'], 
                      status['created_at'], 
                      status['user']['id'],
                      status['user']['name'], 
                      status['text'], url['expanded_url']) 
                            for status in statuses 
                                for url in status['entities']['urls'] ]
    ```

1.  使用所需的认证实例化类：

    ```py
    t= TwitterAPI()
    ```

1.  对查询词 *Apache Spark* 进行搜索：

    ```py
    q="ApacheSpark"
    tsearch = t.searchTwitter(q)
    ```

1.  分析 JSON 输出：

    ```py
    pp(tsearch[1])

    {u'contributors': None,
     u'coordinates': None,
     u'created_at': u'Sat Apr 25 14:50:57 +0000 2015',
     u'entities': {u'hashtags': [{u'indices': [74, 86], u'text': u'sparksummit'}],
                   u'media': [{u'display_url': u'pic.twitter.com/WKUMRXxIWZ',
                               u'expanded_url': u'http://twitter.com/bigdata/status/591976255831969792/photo/1',
                               u'id': 591976255156715520,
                               u'id_str': u'591976255156715520',
                               u'indices': [143, 144],
                               u'media_url': 
    ...(snip)... 
     u'text': u'RT @bigdata: Enjoyed catching up with @ApacheSpark users &amp; leaders at #sparksummit NYC: video clips are out http://t.co/qrqpP6cG9s http://t\u2026',
     u'truncated': False,
     u'user': {u'contributors_enabled': False,
               u'created_at': u'Sat Apr 04 14:44:31 +0000 2015',
               u'default_profile': True,
               u'default_profile_image': True,
               u'description': u'',
               u'entities': {u'description': {u'urls': []}},
               u'favourites_count': 0,
               u'follow_request_sent': False,
               u'followers_count': 586,
               u'following': False,
               u'friends_count': 2,
               u'geo_enabled': False,
               u'id': 3139047660,
               u'id_str': u'3139047660',
               u'is_translation_enabled': False,
               u'is_translator': False,
               u'lang': u'zh-cn',
               u'listed_count': 749,
               u'location': u'',
               u'name': u'Mega Data Mama',
               u'notifications': False,
               u'profile_background_color': u'C0DEED',
               u'profile_background_image_url': u'http://abs.twimg.com/images/themes/theme1/bg.png',
               u'profile_background_image_url_https': u'https://abs.twimg.com/images/themes/theme1/bg.png',
               ...(snip)... 
               u'screen_name': u'MegaDataMama',
               u'statuses_count': 26673,
               u'time_zone': None,
               u'url': None,
               u'utc_offset': None,
               u'verified': False}}
    ```

1.  解析 Twitter 输出以检索感兴趣的关键信息：

    ```py
    tparsed = t.parseTweets(tsearch)
    pp(tparsed)

    [(591980327784046592,
      u'Sat Apr 25 15:01:23 +0000 2015',
      63407360,
      u'Jos\xe9 Carlos Baquero',
      u'Big Data systems are making a difference in the fight against cancer. #BigData #ApacheSpark http://t.co/pnOLmsKdL9',
      u'http://tmblr.co/ZqTggs1jHytN0'),
     (591977704464875520,
      u'Sat Apr 25 14:50:57 +0000 2015',
      3139047660,
      u'Mega Data Mama',
      u'RT @bigdata: Enjoyed catching up with @ApacheSpark users &amp; leaders at #sparksummit NYC: video clips are out http://t.co/qrqpP6cG9s http://t\u2026',
      u'http://goo.gl/eF5xwK'),
     (591977172589539328,
      u'Sat Apr 25 14:48:51 +0000 2015',
      2997608763,
      u'Emma Clark',
      u'RT @bigdata: Enjoyed catching up with @ApacheSpark users &amp; leaders at #sparksummit NYC: video clips are out http://t.co/qrqpP6cG9s http://t\u2026',
      u'http://goo.gl/eF5xwK'),
     ... (snip)...  
     (591879098349268992,
      u'Sat Apr 25 08:19:08 +0000 2015',
      331263208,
      u'Mario Molina',
      u'#ApacheSpark speeds up big data decision-making http://t.co/8hdEXreNfN',
      u'http://www.computerweekly.com/feature/Apache-Spark-speeds-up-big-data-decision-making')]
    ```

# 探索 GitHub 世界

为了更好地理解如何操作 GitHub API，我们将进行以下步骤：

1.  安装 GitHub Python 库。

1.  通过我们在开发者网站上注册时提供的令牌访问 API。

1.  检索托管 spark 仓库的 Apache 基金会的相关关键事实。

让我们一步一步来：

1.  安装 Python PyGithub 库。为了安装它，您需要从命令行运行 `pip install PyGithub`：

    ```py
    pip install PyGithub
    ```

1.  以编程方式创建客户端以实例化 GitHub API：

    ```py
    from github import Github

    # Get your own access token

    ACCESS_TOKEN = 'Get_Your_Own_Access_Token'

    # We are focusing our attention to User = apache and Repo = spark

    USER = 'apache'
    REPO = 'spark'

    g = Github(ACCESS_TOKEN, per_page=100)
    user = g.get_user(USER)
    repo = user.get_repo(REPO)
    ```

1.  从 Apache 用户中检索关键事实。GitHub 上有 640 个活跃的 Apache 仓库：

    ```py
    repos_apache = [repo.name for repo in g.get_user('apache').get_repos()]
    len(repos_apache)
    640
    ```

1.  从 Spark 仓库中检索关键事实，Spark 仓库中使用的编程语言如下所示：

    ```py
    pp(repo.get_languages())

    {u'C': 1493,
     u'CSS': 4472,
     u'Groff': 5379,
     u'Java': 1054894,
     u'JavaScript': 21569,
     u'Makefile': 7771,
     u'Python': 1091048,
     u'R': 339201,
     u'Scala': 10249122,
     u'Shell': 172244}
    ```

1.  获取 Spark GitHub 仓库网络的一些关键参与者。在撰写本文时，Apache Spark 仓库有 3,738 个 star。网络非常庞大。第一个 star 是 *Matei Zaharia*，当时他在伯克利大学攻读博士学位时是 Spark 项目的共同创始人。

    ```py
    stargazers = [ s for s in repo.get_stargazers() ]
    print "Number of stargazers", len(stargazers)
    Number of stargazers 3738

    [stargazers[i].login for i in range (0,20)]
    [u'mateiz',
     u'beyang',
     u'abo',
     u'CodingCat',
     u'andy327',
     u'CrazyJvm',
     u'jyotiska',
     u'BaiGang',
     u'sundstei',
     u'dianacarroll',
     u'ybotco',
     u'xelax',
     u'prabeesh',
     u'invkrh',
     u'bedla',
     u'nadesai',
     u'pcpratts',
     u'narkisr',
     u'Honghe',
     u'Jacke']
    ```

## 通过 Meetup 了解社区

为了更好地理解如何操作 Meetup API，我们将进行以下步骤：

1.  创建一个 Python 程序，使用认证令牌调用 Meetup API。

1.  检索类似 *London Data Science* 的 Meetup 群组过去活动的信息。

1.  检索 Meetup 成员的资料，以便分析他们在类似 Meetup 群组中的参与情况。

让我们一步一步地通过这个过程：

1.  由于没有可靠的 Meetup API Python 库，我们将程序化创建一个客户端来实例化 Meetup API：

    ```py
    import json
    import mimeparse
    import requests
    import urllib
    from pprint import pprint as pp

    MEETUP_API_HOST = 'https://api.meetup.com'
    EVENTS_URL = MEETUP_API_HOST + '/2/events.json'
    MEMBERS_URL = MEETUP_API_HOST + '/2/members.json'
    GROUPS_URL = MEETUP_API_HOST + '/2/groups.json'
    RSVPS_URL = MEETUP_API_HOST + '/2/rsvps.json'
    PHOTOS_URL = MEETUP_API_HOST + '/2/photos.json'
    GROUP_URLNAME = 'London-Machine-Learning-Meetup'
    # GROUP_URLNAME = 'London-Machine-Learning-Meetup' # 'Data-Science-London'

    class Mee
    tupAPI(object):
        """
        Retrieves information about meetup.com
        """
        def __init__(self, api_key, num_past_events=10, http_timeout=1,
                     http_retries=2):
            """
            Create a new instance of MeetupAPI
            """
            self._api_key = api_key
            self._http_timeout = http_timeout
            self._http_retries = http_retries
            self._num_past_events = num_past_events

        def get_past_events(self):
            """
            Get past meetup events for a given meetup group
            """
            params = {'key': self._api_key,
                      'group_urlname': GROUP_URLNAME,
                      'status': 'past',
                      'desc': 'true'}
            if self._num_past_events:
                params['page'] = str(self._num_past_events)

            query = urllib.urlencode(params)
            url = '{0}?{1}'.format(EVENTS_URL, query)
            response = requests.get(url, timeout=self._http_timeout)
            data = response.json()['results']
            return data

        def get_members(self):
            """
            Get meetup members for a given meetup group
            """
            params = {'key': self._api_key,
                      'group_urlname': GROUP_URLNAME,
                      'offset': '0',
                      'format': 'json',
                      'page': '100',
                      'order': 'name'}
            query = urllib.urlencode(params)
            url = '{0}?{1}'.format(MEMBERS_URL, query)
            response = requests.get(url, timeout=self._http_timeout)
            data = response.json()['results']
            return data

        def get_groups_by_member(self, member_id='38680722'):
            """
            Get meetup groups for a given meetup member
            """
            params = {'key': self._api_key,
                      'member_id': member_id,
                      'offset': '0',
                      'format': 'json',
                      'page': '100',
                      'order': 'id'}
            query = urllib.urlencode(params)
            url = '{0}?{1}'.format(GROUPS_URL, query)
            response = requests.get(url, timeout=self._http_timeout)
            data = response.json()['results']
            return data
    ```

1.  然后，我们将从给定的 Meetup 群组中检索过去的事件：

    ```py
    m = MeetupAPI(api_key='Get_Your_Own_Key')
    last_meetups = m.get_past_events()
    pp(last_meetups[5])

    {u'created': 1401809093000,
     u'description': u"<p>We are hosting a joint meetup between Spark London and Machine Learning London. Given the excitement in the machine learning community around Spark at the moment a joint meetup is in order!</p> <p>Michael Armbrust from the Apache Spark core team will be flying over from the States to give us a talk in person.\xa0Thanks to our sponsors, Cloudera, MapR and Databricks for helping make this happen.</p> <p>The first part of the talk will be about MLlib, the machine learning library for Spark,\xa0and the second part, on\xa0Spark SQL.</p> <p>Don't sign up if you have already signed up on the Spark London page though!</p> <p>\n\n\nAbstract for part one:</p> <p>In this talk, we\u2019ll introduce Spark and show how to use it to build fast, end-to-end machine learning workflows. Using Spark\u2019s high-level API, we can process raw data with familiar libraries in Java, Scala or Python (e.g. NumPy) to extract the features for machine learning. Then, using MLlib, its built-in machine learning library, we can run scalable versions of popular algorithms. We\u2019ll also cover upcoming development work including new built-in algorithms and R bindings.</p> <p>\n\n\n\nAbstract for part two:\xa0</p> <p>In this talk, we'll examine Spark SQL, a new Alpha component that is part of the Apache Spark 1.0 release. Spark SQL lets developers natively query data stored in both existing RDDs and external sources such as Apache Hive. A key feature of Spark SQL is the ability to blur the lines between relational tables and RDDs, making it easy for developers to intermix SQL commands that query external data with complex analytics. In addition to Spark SQL, we'll explore the Catalyst optimizer framework, which allows Spark SQL to automatically rewrite query plans to execute more efficiently.</p>",
     u'event_url': u'http://www.meetup.com/London-Machine-Learning-Meetup/events/186883262/',
     u'group': {u'created': 1322826414000,
                u'group_lat': 51.52000045776367,
                u'group_lon': -0.18000000715255737,
                u'id': 2894492,
                u'join_mode': u'open',
                u'name': u'London Machine Learning Meetup',
                u'urlname': u'London-Machine-Learning-Meetup',
                u'who': u'Machine Learning Enthusiasts'},
     u'headcount': 0,
     u'id': u'186883262',
     u'maybe_rsvp_count': 0,
     u'name': u'Joint Spark London and Machine Learning Meetup',
     u'rating': {u'average': 4.800000190734863, u'count': 5},
     u'rsvp_limit': 70,
     u'status': u'past',
     u'time': 1403200800000,
     u'updated': 1403450844000,
     u'utc_offset': 3600000,
     u'venue': {u'address_1': u'12 Errol St, London',
                u'city': u'EC1Y 8LX',
                u'country': u'gb',
                u'id': 19504802,
                u'lat': 51.522533,
                u'lon': -0.090934,
                u'name': u'Royal Statistical Society',
                u'repinned': False},
     u'visibility': u'public',
     u'waitlist_count': 84,
     u'yes_rsvp_count': 70}
    ```

1.  获取 Meetup 成员的信息：

    ```py
    members = m.get_members()

    {u'city': u'London',
      u'country': u'gb',
      u'hometown': u'London',
      u'id': 11337881,
      u'joined': 1421418896000,
      u'lat': 51.53,
      u'link': u'http://www.meetup.com/members/11337881',
      u'lon': -0.09,
      u'name': u'Abhishek Shivkumar',
      u'other_services': {u'twitter': {u'identifier': u'@abhisemweb'}},
      u'photo': {u'highres_link': u'http://photos3.meetupstatic.com/photos/member/9/6/f/3/highres_10898643.jpeg',
                 u'photo_id': 10898643,
                 u'photo_link': u'http://photos3.meetupstatic.com/photos/member/9/6/f/3/member_10898643.jpeg',
                 u'thumb_link': u'http://photos3.meetupstatic.com/photos/member/9/6/f/3/thumb_10898643.jpeg'},
      u'self': {u'common': {}},
      u'state': u'17',
      u'status': u'active',
      u'topics': [{u'id': 1372, u'name': u'Semantic Web', u'urlkey': u'semweb'},
                  {u'id': 1512, u'name': u'XML', u'urlkey': u'xml'},
                  {u'id': 49585,
                   u'name': u'Semantic Social Networks',
                   u'urlkey': u'semantic-social-networks'},
                  {u'id': 24553,
                   u'name': u'Natural Language Processing',
    ...(snip)...
                   u'name': u'Android Development',
                   u'urlkey': u'android-developers'}],
      u'visited': 1429281599000}
    ```

# 预览我们的应用程序

我们的挑战是从这些社交网络中提取数据，找到关键关系并得出见解。以下是一些感兴趣的元素：

+   可视化顶级影响者：发现社区中的顶级影响者：

    +   高频 Twitter 用户在*Apache Spark*上

    +   GitHub 的提交者

    +   领先的 Meetup 演讲

+   理解网络：GitHub 提交者、观察者和星标者的网络图

+   确定热点位置：定位 Spark 最活跃的位置

以下截图提供了我们应用程序的预览：

![预览我们的应用程序](img/B03986_02_09.jpg)

# 摘要

在本章中，我们概述了我们的应用程序的整体架构。我们解释了处理数据的主要范式：批处理，也称为静态数据，以及流分析，称为动态数据。我们继续建立与三个感兴趣的社会网络的连接：Twitter、GitHub 和 Meetup。我们采样了数据，并提供了我们旨在构建的预览。本书的其余部分将专注于 Twitter 数据集。我们提供了访问三个社交网络的工具和 API，以便您可以在以后阶段创建自己的数据混合。我们现在准备调查收集到的数据，这将是下一章的主题。

在下一章中，我们将更深入地探讨数据分析，提取我们目的的关键属性，并管理批处理和流处理的信息存储。
