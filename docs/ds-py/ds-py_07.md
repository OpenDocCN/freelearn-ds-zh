# 第七章：分析研究：Twitter 情感分析与 NLP 及大数据

|   | *“数据是新的石油。”* |   |
| --- | --- | --- |
|   | --*未知* |

本章我们将探讨人工智能和数据科学的两个重要领域：**自然语言处理**（**NLP**）和大数据分析。为了支持示例应用程序，我们重新实现了第一章中描述的*Twitter 标签情感分析*项目，*编程与数据科学——一种新工具集*，但这次我们利用 Jupyter Notebooks 和 PixieDust 构建实时仪表盘，分析来自与特定实体（例如公司提供的某个产品）相关的推文流中的数据，提供情感信息以及从相同推文中提取的其他趋势实体的信息。在本章的结尾，读者将学习如何将基于云的 NLP 服务，如*IBM Watson 自然语言理解*，集成到他们的应用程序中，并使用像 Apache Spark 这样的框架在（Twitter）规模上执行数据分析。

一如既往，我们将展示如何通过实现一个作为 PixieApp 的实时仪表盘，直接在 Jupyter Notebook 中运行来使分析工作可操作化。

# 开始使用 Apache Spark

*大数据*这一术语常常给人模糊不清和不准确的感觉。什么样的数据集才算是大数据呢？是 10 GB、100 GB、1 TB 还是更多？我喜欢的一个定义是：大数据是当数据无法完全装入单个机器的内存时。多年来，数据科学家被迫对大数据集进行抽样处理，以便能够在单台机器上处理，但随着并行计算框架的出现，这些框架能够将数据分布到多台机器的集群中，使得可以在整个数据集上进行工作，当然，前提是集群有足够的机器。与此同时，云技术的进步使得可以按需提供适合数据集大小的机器集群。

目前，有多种框架（大多数通常以开源形式提供）可以提供强大且灵活的并行计算能力。最受欢迎的一些包括 Apache Hadoop ([`hadoop.apache.org`](http://hadoop.apache.org))、Apache Spark ([`spark.apache.org`](https://spark.apache.org)) 和 Dask ([`dask.pydata.org`](https://dask.pydata.org))。对于我们的*Twitter 情感分析*应用程序，我们将使用 Apache Spark，它在可扩展性、可编程性和速度方面表现出色。此外，许多云服务提供商提供了某种形式的 Spark 即服务，能够在几分钟内按需创建一个合适大小的 Spark 集群。

一些 Spark 即服务的云服务提供商包括：

+   Microsoft Azure: [`azure.microsoft.com/en-us/services/hdinsight/apache-spark`](https://azure.microsoft.com/en-us/services/hdinsight/apache-spark)

+   亚马逊网络服务：[`aws.amazon.com/emr/details/spark`](https://aws.amazon.com/emr/details/spark)

+   Google Cloud：[`cloud.google.com/dataproc`](https://cloud.google.com/dataproc)

+   Databricks：[`databricks.com`](https://databricks.com)

+   IBM Cloud：[`www.ibm.com/cloud/analytics-engine`](https://www.ibm.com/cloud/analytics-engine)

### 注意

**注意**：Apache Spark 也可以轻松地在本地机器上安装用于测试，在这种情况下，集群节点通过线程进行模拟。

## Apache Spark 架构

下图展示了 Apache Spark 框架的主要组件：

![Apache Spark 架构](img/B09699_07_01.jpg)

Spark 高层架构

+   **Spark SQL**：该组件的核心数据结构是 Spark DataFrame，使得熟悉 SQL 语言的用户能够轻松地处理结构化数据。

+   **Spark Streaming**：用于处理流式数据的模块。正如我们稍后所看到的，我们将在示例应用中使用该模块，特别是 Spark 2.0 引入的结构化流处理（Structured Streaming）。

+   **MLlib**：提供一个功能丰富的机器学习库，在 Spark 规模上运行。

+   **GraphX**：用于执行图并行计算的模块。

如下图所示，主要有两种方式可以与 Spark 集群工作：

![Apache Spark 架构](img/B09699_07_02.jpg)

与 Spark 集群工作的两种方式

+   **spark-submit**：用于在集群上启动 Spark 应用的 Shell 脚本

+   **Notebooks**：与 Spark 集群交互式执行代码语句

本书不涵盖` spark-submit` shell 脚本的内容，但可以在以下网址找到官方文档：[`spark.apache.org/docs/latest/submitting-applications.html`](https://spark.apache.org/docs/latest/submitting-applications.html)。在本章的其余部分，我们将重点介绍通过 Jupyter Notebooks 与 Spark 集群进行交互。

## 配置 Notebooks 以便与 Spark 一起使用

本节中的说明仅涵盖在本地安装 Spark 用于开发和测试。手动在集群中安装 Spark 超出了本书的范围。如果需要真正的集群，强烈建议使用基于云的服务。

默认情况下，本地 Jupyter Notebooks 会安装普通的 Python 内核。为了与 Spark 一起使用，用户必须执行以下步骤：

1.  从[`spark.apache.org/downloads.html`](https://spark.apache.org/downloads.html)下载二进制分发包，安装 Spark 到本地。

1.  使用以下命令在临时目录中生成内核规范：

    ```py
    ipython kernel install --prefix /tmp

    ```

    ### 注意

    **注意**：上述命令可能会生成警告消息，只要显示以下信息，这些警告可以安全忽略：

    `已在/tmp/share/jupyter/kernels/python3 中安装 kernelspec python3`

1.  转到 `/tmp/share/jupyter/kernels/python3`，编辑 `kernel.json` 文件，向 JSON 对象中添加以下键（将 `<<spark_root_path>>` 替换为你安装 Spark 的目录路径，将 `<<py4j_version>>` 替换为你系统上安装的版本）：

    ```py
    "env": {
        "PYTHONPATH": "<<spark_root_path>>/python/:<<spark_root_path>>/python/lib/py4j-<<py4j_version>>-src.zip",
        "SPARK_HOME": "<<spark_root_path>>",
        "PYSPARK_SUBMIT_ARGS": "--master local[10] pyspark-shell",
        "SPARK_DRIVER_MEMORY": "10G",
        "SPARK_LOCAL_IP": "127.0.0.1",
        "PYTHONSTARTUP": "<<spark_root_path>>/python/pyspark/shell.py"
    }
    ```

1.  你可能还想自定义 `display_name` 键，以使其在 Juptyer 界面中独特且易于识别。如果你需要查看现有内核的列表，可以使用以下命令：

    ```py
    jupyter kernelspec list

    ```

    前述命令将为你提供内核名称和相关路径的列表。从路径中，你可以打开 `kernel.json` 文件，访问 `display_name` 值。例如：

    ```py
     Available kernels:
     pixiedustspark16
     /Users/dtaieb/Library/Jupyter/kernels/pixiedustspark16
     pixiedustspark21
     /Users/dtaieb/Library/Jupyter/kernels/pixiedustspark21
     pixiedustspark22
     /Users/dtaieb/Library/Jupyter/kernels/pixiedustspark22
     pixiedustspark23
     /Users/dtaieb/Library/Jupyter/kernels/pixiedustspark23

    ```

1.  使用以下命令安装带有编辑文件的内核：

    ```py
    jupyter kernelspec install /tmp/share/jupyter/kernels/python3

    ```

    ### 注意

    注意：根据环境不同，你可能在运行前述命令时会遇到“权限拒绝”的错误。在这种情况下，你可能需要使用管理员权限运行该命令，使用 `sudo` 或者按如下方式使用 `--user` 开关：

    `jupyter kernelspec install --user /tmp/share/jupyter/kernels/python3`

    如需了解更多安装选项的信息，可以使用 `-h` 开关。例如：

    ```py
     jupyter kernelspec install -h

    ```

1.  重启 Notebook 服务器并开始使用新的 PySpark 内核。

幸运的是，PixieDust 提供了一个 `install` 脚本来自动化前述的手动步骤。

### 注意

你可以在这里找到该脚本的详细文档：

[`pixiedust.github.io/pixiedust/install.html`](https://pixiedust.github.io/pixiedust/install.html)

简而言之，使用自动化 PixieDust `install` 脚本需要发出以下命令并按照屏幕上的说明操作：

```py
jupyter pixiedust install

```

本章稍后会深入探讨 Spark 编程模型，但现在让我们在下一节定义我们 *Twitter 情感分析* 应用的 MVP 要求。

# Twitter 情感分析应用

和往常一样，我们首先定义 MVP 版本的要求：

+   连接 Twitter，获取由用户提供的查询字符串过滤的实时推文流

+   丰富推文，添加情感信息和从文本中提取的相关实体

+   使用实时图表显示有关数据的各种统计信息，并在指定的时间间隔内更新图表

+   系统应该能够扩展到 Twitter 数据规模

以下图示展示了我们应用架构的第一个版本：

![Twitter 情感分析应用](img/B09699_07_03.jpg)

Twitter 情感分析架构版本 1

对于第一个版本，应用将完全在一个 Python Notebook 中实现，并调用外部服务处理 NLP 部分。为了能够扩展，我们肯定需要将一些处理外部化，但对于开发和测试，我发现能够将整个应用封装在一个 Notebook 中显著提高了生产力。

至于库和框架，我们将使用 Tweepy（[`www.tweepy.org`](http://www.tweepy.org)）连接到 Twitter，使用 Apache Spark 结构化流处理（[`spark.apache.org/streaming`](https://spark.apache.org/streaming)）处理分布式集群中的流数据，使用 Watson Developer Cloud Python SDK（[`github.com/watson-developer-cloud/python-sdk`](https://github.com/watson-developer-cloud/python-sdk)）访问 IBM Watson 自然语言理解（[`www.ibm.com/watson/services/natural-language-understanding`](https://www.ibm.com/watson/services/natural-language-understanding)）服务。

# 第一部分 – 使用 Spark 结构化流处理获取数据

为了获取数据，我们使用 Tweepy，它提供了一个优雅的 Python 客户端库来访问 Twitter API。Tweepy 支持的 API 非常广泛，详细介绍超出了本书的范围，但你可以在 Tweepy 官方网站找到完整的 API 参考：[`tweepy.readthedocs.io/en/v3.6.0/cursor_tutorial.html`](http://tweepy.readthedocs.io/en/v3.6.0/cursor_tutorial.html)。

你可以直接通过 PyPi 安装 Tweepy 库，使用`pip install`命令。以下命令展示了如何通过 Notebook 使用`!`指令安装：

```py
!pip install tweepy

```

### 注意

**注意**：当前使用的 Tweepy 版本是 3.6.0。安装完库后，别忘了重启内核。

## 数据管道架构图

在深入了解数据管道的每个组件之前，最好先了解其整体架构并理解计算流。

如下图所示，我们首先创建一个 Tweepy 流，将原始数据写入 CSV 文件。然后，我们创建一个 Spark Streaming 数据框，读取 CSV 文件并定期更新新数据。从 Spark Streaming 数据框中，我们使用 SQL 创建一个 Spark 结构化查询，并将其结果存储在 Parquet 数据库中：

![数据管道架构图](img/B09699_07_04.jpg)

流计算流程

## Twitter 身份验证

在使用任何 Twitter API 之前，建议先进行身份验证。最常用的身份验证机制之一是 OAuth 2.0 协议（[`oauth.net`](https://oauth.net)），该协议使第三方应用程序能够访问网络服务。你需要做的第一件事是获取一组密钥字符串，这些字符串由 OAuth 协议用于对你进行身份验证：

+   **消费者密钥**：唯一标识客户端应用程序的字符串（即 API 密钥）。

+   **消费者密钥**：仅应用程序和 Twitter OAuth 服务器知道的密钥字符串。可以将其视为密码。

+   **访问令牌**：用于验证请求的字符串。该令牌也在授权阶段用于确定应用程序的访问级别。

+   **访问令牌密钥**：与消费者密钥类似，这是与访问令牌一起发送的密码字符串，用作密码。

要生成前面的密钥字符串，您需要访问[`apps.twitter.com`](http://apps.twitter.com)，使用您的常规 Twitter 用户 ID 和密码进行身份验证，并按照以下步骤操作：

1.  使用**创建新应用**按钮创建一个新的 Twitter 应用。

1.  填写应用程序详情，同意开发者协议，然后点击**创建您的 Twitter 应用**按钮。

    ### 提示

    **注意**：确保您的手机号码已添加到个人资料中，否则在创建 Twitter 应用时会出现错误。

    您可以为必填项**网站**输入提供一个随机 URL，并将**URL**输入留空，因为这是一个可选的回调 URL。

1.  点击**密钥和访问令牌**标签以获取消费者和访问令牌。您可以随时使用页面上提供的按钮重新生成这些令牌。如果您这么做，您还需要在应用程序代码中更新这些值。

为了更容易进行代码维护，我们将把这些令牌放在 Notebook 顶部的单独变量中，并创建我们稍后将使用的`tweepy.OAuthHandler`类：

```py
from tweepy import OAuthHandler
# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key="XXXX"
consumer_secret="XXXX"

# After the step above, you will be redirected to your app's page.
# Create an access token under the "Your access token" section
access_token="XXXX"
access_token_secret="XXXX"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

```

## 创建 Twitter 流

为了实现我们的应用程序，我们只需要使用这里文档化的 Twitter 流 API：[`tweepy.readthedocs.io/en/v3.5.0/streaming_how_to.html`](http://tweepy.readthedocs.io/en/v3.5.0/streaming_how_to.html)。在此步骤中，我们创建一个 Twitter 流，将传入的数据存储到本地文件系统中的 CSV 文件中。通过继承自`tweepy.streaming.StreamListener`的自定义`RawTweetsListener`类完成此操作。通过重写`on_data`方法来处理传入数据的自定义处理。

在我们的案例中，我们希望使用标准 Python `csv`模块中的`DictWriter`将传入的 JSON 数据转换为 CSV 格式。由于 Spark Streaming 文件输入源仅在输入目录中创建新文件时触发，因此我们不能简单地将数据追加到现有文件中。相反，我们将数据缓冲到一个数组中，并在缓冲区达到容量时将其写入磁盘。

### 注意

为了简化，实施中没有包括处理完文件后的清理工作。另一个小的限制是，我们目前等待缓冲区填满后再写入文件，理论上如果没有新推文出现，这可能需要很长时间。

`RawTweetsListener`的代码如下所示：

```py
from six import iteritems
import json
import csv
from tweepy.streaming import StreamListener
class RawTweetsListener(StreamListener):
    def __init__(self):
        self.buffered_data = []
        self.counter = 0

    def flush_buffer_if_needed(self):
        "Check the buffer capacity and write to a new file if needed"
        length = len(self.buffered_data)
        if length > 0 and length % 10 == 0:
            with open(os.path.join( output_dir,
                "tweets{}.csv".format(self.counter)), "w") as fs:
                self.counter += 1
                csv_writer = csv.DictWriter( fs,
                    fieldnames = fieldnames)
                for data in self.buffered_data:
 csv_writer.writerow(data)
            self.buffered_data = []

    def on_data(self, data):
        def transform(key, value):
            return transformskey if key in transforms else value

        self.buffered_data.append(
            {key:transform(key,value) \
                 for key,value in iteritems(json.loads(data)) \
                 if key in fieldnames}
        )
        self.flush_buffer_if_needed()
        return True

    def on_error(self, status):
        print("An error occured while receiving streaming data: {}".format(status))
        return False
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode1.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode1.py)

从前面的代码中有几个重要的地方需要注意：

+   每条来自 Twitter API 的推文都包含大量数据，我们使用`field_metadata`变量选择保留的字段。我们还定义了一个全局变量`fieldnames`，它保存了要从流中捕获的字段列表，以及一个`transforms`变量，它包含一个字典，字典的键是所有具有变换函数的字段名，值是变换函数本身：

    ```py
    from pyspark.sql.types import StringType, DateType
    from bs4 import BeautifulSoup as BS
    fieldnames = [f["name"] for f in field_metadata]
    transforms = {
        item['name']:item['transform'] for item in field_metadata if "transform" in item
    }
    field_metadata = [
        {"name": "created_at","type": DateType()},
        {"name": "text", "type": StringType()},
        {"name": "source", "type": StringType(),
             "transform": lambda s: BS(s, "html.parser").text.strip()
        }
    ]
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode2.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode2.py)

+   CSV 文件被写入定义在自己的变量中的`output_dir`目录。在启动时，我们首先删除该目录及其内容：

    ```py
    import shutil
    def ensure_dir(dir, delete_tree = False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        elif delete_tree:
            shutil.rmtree(dir)
            os.makedirs(dir)
        return os.path.abspath(dir)

    root_dir = ensure_dir("output", delete_tree = True)
    output_dir = ensure_dir(os.path.join(root_dir, "raw"))

    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode3.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode3.py)

+   `field_metadata`包含了 Spark DataType，我们稍后将在创建 Spark 流查询时使用它来构建模式。

+   `field_metadata`还包含一个可选的变换`lambda`函数，用于在将值写入磁盘之前清理数据。作为参考，Python 中的 lambda 函数是一个内联定义的匿名函数（请参见[`docs.python.org/3/tutorial/controlflow.html#lambda-expressions`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions)）。我们在此使用它来处理常常以 HTML 片段形式返回的源字段。在这个 lambda 函数中，我们使用了 BeautifulSoup 库（它也在上一章中使用过）来提取只有文本的内容，如以下代码片段所示：

    ```py
    lambda s: BS(s, "html.parser").text.strip()
    ```

现在，`RawTweetsListener`已经创建，我们定义了一个`start_stream`函数，稍后将在 PixieApp 中使用。此函数接受一个搜索词数组作为输入，并使用`filter`方法启动一个新的流：

```py
from tweepy import Stream
def start_stream(queries):
    "Asynchronously start a new Twitter stream"
    stream = Stream(auth, RawTweetsListener())
 stream.filter(track=queries, async=True)
    return stream
```

### 注意

注意到传递给`stream.filter`的`async=True`参数。这是必要的，确保该函数不会阻塞，这样我们就可以在 Notebook 中运行其他代码。

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode4.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode4.py)

以下代码启动了一个流，它将接收包含单词`baseball`的推文：

```py
stream = start_stream(["baseball"])
```

当运行上述代码时，Notebook 中不会生成任何输出。然而，你可以在输出目录（即`../output/raw`）中看到生成的文件（如`tweets0.csv`、`tweets1.csv`等），这些文件位于 Notebook 运行的路径下。

要停止流，我们只需调用`disconnect`方法，如下所示：

```py
stream.disconnect()
```

## 创建一个 Spark Streaming DataFrame

根据架构图，下一步是创建一个 Spark Streaming DataFrame `tweets_sdf`，该 DataFrame 使用 `output_dir` 作为源文件输入。我们可以把 Streaming DataFrame 看作一个没有边界的表格，随着新数据从流中到达，新的行会不断被添加进来。

### 注意

**注意**：Spark Structured Streaming 支持多种类型的输入源，包括文件、Kafka、Socket 和 Rate。（Socket 和 Rate 仅用于测试。）

以下图表摘自 Spark 网站，能够很好地解释新数据是如何被添加到 Streaming DataFrame 中的：

![创建 Spark Streaming DataFrame](img/B09699_07_05.jpg)

Streaming DataFrame 流程

来源: [`spark.apache.org/docs/latest/img/structured-streaming-stream-as-a-table.png`](https://spark.apache.org/docs/latest/img/structured-streaming-stream-as-a-table.png)

Spark Streaming Python API 提供了一种优雅的方式来使用 `spark.readStream` 属性创建 Streaming DataFrame，该属性会创建一个新的 `pyspark.sql.streamingreamReader` 对象，方便你链式调用方法，并能让代码更加清晰（有关此模式的更多细节，请参见 [`en.wikipedia.org/wiki/Method_chaining`](https://en.wikipedia.org/wiki/Method_chaining)）。

例如，要创建一个 CSV 文件流，我们调用 `format` 方法并传入 `csv`，接着链式调用适用的选项，并通过指定目录路径调用 `load` 方法：

```py
schema = StructType(
[StructField(f["name"], f["type"], True) for f in field_metadata]
)
csv_sdf = spark.readStream\
    .format("csv")\
 .option("schema", schema)\
    .option("multiline", True)\
 .option("dateFormat", 'EEE MMM dd kk:mm:ss Z y')\
    .option("ignoreTrailingWhiteSpace", True)\
    .option("ignoreLeadingWhiteSpace", True)\
    .load(output_dir)
```

### 注意

你可以在此处找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode5.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode5.py)

`spark.readStream` 还提供了一个方便的高阶 `csv` 方法，它将路径作为第一个参数，并为选项提供关键字参数：

```py
csv_sdf = spark.readStream \
    .csv(
        output_dir,
        schema=schema,
        multiLine = True,
        dateFormat = 'EEE MMM dd kk:mm:ss Z y',
        ignoreTrailingWhiteSpace = True,
        ignoreLeadingWhiteSpace = True
    )
```

### 注意

你可以在此处找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode6.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode6.py)

你可以通过调用 `isStreaming` 方法来验证 `csv_sdf` DataFrame 是否真的是一个 Streaming DataFrame，返回值应为 `true`。以下代码还添加了 `printSchema` 方法的调用，以验证 schema 是否按照 `field_metadata` 配置如预期那样：

```py
print(csv_sdf.isStreaming)
csv_sdf.printSchema()
```

返回值：

```py
root
 |-- created_at: date (nullable = true)
 |-- text: string (nullable = true)
 |-- source: string (nullable = true)
```

在继续下一步之前，理解`csv_sdf`流数据框如何适应结构化流编程模型及其局限性非常重要。从本质上讲，Spark 的低级 API 定义了**弹性分布式数据集**（**RDD**）数据结构，它封装了管理分布式数据的所有底层复杂性。像容错（集群节点因任何原因崩溃时，框架会自动重启节点，无需开发者干预）等特性都由框架自动处理。RDD 操作有两种类型：转换和动作。**转换**是对现有 RDD 的逻辑操作，直到调用动作操作时，转换才会在集群上立即执行（懒执行）。转换的输出是一个新的 RDD。内部，Spark 维护一个 RDD 有向无环图（DAG），记录所有生成 RDD 的血统，这在从服务器故障恢复时非常有用。常见的转换操作包括`map`、`flatMap`、`filter`、`sample`和`distinct`。对数据框的转换（数据框在内部由 RDD 支持）也适用，且它们具有包括 SQL 查询的优点。另一方面，**动作**不会生成其他 RDD，而是对实际分布式数据执行操作，返回非 RDD 值。常见的动作操作包括`reduce`、`collect`、`count`和`take`。

如前所述，`csv_sdf`是一个流式数据框（Streaming DataFrame），这意味着数据会持续被添加到其中，因此我们只能对其应用转换，而不能执行操作。为了解决这个问题，我们必须先使用`csv_sdf.writeStream`创建一个流查询，这是一个`pyspark.sql.streaming.DataStreamWriter`对象。流查询负责将结果发送到输出接收器。然后，我们可以通过`start()`方法运行流查询。

Spark Streaming 支持多种输出接收器类型：

+   **文件**：支持所有经典文件格式，包括 JSON、CSV 和 Parquet

+   **Kafka**：直接写入一个或多个 Kafka 主题

+   **Foreach**：对集合中的每个元素执行任意计算

+   **控制台**：将输出打印到系统控制台（主要用于调试）

+   **内存**：输出存储在内存中

在下一节中，我们将创建并运行一个结构化查询，针对`csv_sdf`使用输出接收器将结果存储为 Parquet 格式。

## 创建并运行结构化查询

使用`tweets_sdf`流数据框，我们创建一个流查询`tweet_streaming_query`，该查询将数据以*append*输出模式写入 Parquet 格式。

### 注意

**注意**：Spark 流查询支持三种输出模式：**complete**，每次触发时写入整个表；**append**，只写入自上次触发以来的增量行；以及**update**，只写入已修改的行。

Parquet 是一种列式数据库格式，提供了高效、可扩展的分布式分析存储。你可以在这里找到有关 Parquet 格式的更多信息：[`parquet.apache.org`](https://parquet.apache.org)。

以下代码创建并启动 `tweet_streaming_query` 流式查询：

```py
tweet_streaming_query = csv_sdf \
    .writeStream \
    .format("parquet") \
 .option("path", os.path.join(root_dir, "output_parquet")) \
 .trigger(processingTime="2 seconds") \
 .option("checkpointLocation", os.path.join(root_dir, "output_chkpt")) \
    .start()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode7.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode7.py)

类似地，你可以使用 `stop()` 方法来停止流式查询，如下所示：

```py
tweet_streaming_query.stop()
```

在上述代码中，我们使用 `path` 选项指定 Parquet 文件的位置，并使用 `checkpointLocation` 指定在服务器故障时用于恢复的数据位置。我们还指定了从流中读取新数据并将新行添加到 Parquet 数据库的触发间隔。

出于测试目的，你也可以使用 `console` sink 来查看每次生成新原始 CSV 文件时从 `output_dir` 目录读取的新行：

```py
tweet_streaming_query = csv_sdf.writeStream\
    .outputMode("append")\
    .format("console")\
    .trigger(processingTime='2 seconds')\
    .start()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode8.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode8.py)

你可以在 Spark 集群主节点的系统输出中查看结果（你需要物理访问主节点机器并查看日志文件，因为不幸的是，由于操作在不同的进程中执行，输出不会显示在笔记本中。日志文件的位置取决于集群管理软件；有关更多信息，请参阅具体的文档）。

以下是特定批次显示的示例结果（标识符已被屏蔽）：

```py
-------------------------------------------
Batch: 17
-------------------------------------------
+----------+--------------------+-------------------+
|created_at|                text|             source|
+----------+--------------------+-------------------+
|2018-04-12|RT @XXXXXXXXXXXXX...|Twitter for Android|
|2018-04-12|RT @XXXXXXX: Base...| Twitter for iPhone|
|2018-04-12|That's my roommat...| Twitter for iPhone|
|2018-04-12|He's come a long ...| Twitter for iPhone|
|2018-04-12|RT @XXXXXXXX: U s...| Twitter for iPhone|
|2018-04-12|Baseball: Enid 10...|   PushScoreUpdates|
|2018-04-12|Cubs and Sox aren...| Twitter for iPhone|
|2018-04-12|RT @XXXXXXXXXX: T...|          RoundTeam|
|2018-04-12|@XXXXXXXX that ri...| Twitter for iPhone|
|2018-04-12|RT @XXXXXXXXXX: S...| Twitter for iPhone|
+----------+--------------------+-------------------+
```

## 监控活动流式查询

当流式查询启动时，Spark 会分配集群资源。因此，管理和监控这些查询非常重要，以确保你不会耗尽集群资源。随时可以通过以下代码获取所有正在运行的查询列表：

```py
print(spark.streams.active)
```

结果：

```py
[<pyspark.sql.streaming.StreamingQuery object at 0x12d7db6a0>, <pyspark.sql.streaming.StreamingQuery object at 0x12d269c18>]
```

然后，你可以通过使用以下查询监控属性来深入了解每个查询的细节：

+   `id`：返回查询的唯一标识符，该标识符在重启时仍会保留（从检查点数据恢复）

+   `runId`：返回为当前会话生成的唯一 ID

+   `explain()`：打印查询的详细解释

+   `recentProgress`：返回最近的进度更新数组

+   `lastProgress`：返回最新的进度

以下代码打印每个活动查询的最新进度：

```py
import json
for query in spark.streams.active:
    print("-----------")
    print("id: {}".format(query.id))
    print(json.dumps(query.lastProgress, indent=2, sort_keys=True))
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode9.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode9.py)

第一个查询的结果显示如下：

```py
-----------
id: b621e268-f21d-4eef-b6cd-cb0bc66e53c4
{
  "batchId": 18,
  "durationMs": {
    "getOffset": 4,
    "triggerExecution": 4
  },
  "id": "b621e268-f21d-4eef-b6cd-cb0bc66e53c4",
  "inputRowsPerSecond": 0.0,
  "name": null,
  "numInputRows": 0,
  "processedRowsPerSecond": 0.0,
  "runId": "d2459446-bfad-4648-ae3b-b30c1f21be04",
  "sink": {
    "description": "org.apache.spark.sql.execution.streaming.ConsoleSinkProvider@586d2ad5"
  },
  "sources": [
    {
      "description": "FileStreamSource[file:/Users/dtaieb/cdsdev/notebookdev/Pixiedust/book/Chapter7/output/raw]",
      "endOffset": {
        "logOffset": 17
      },
      "inputRowsPerSecond": 0.0,
      "numInputRows": 0,
      "processedRowsPerSecond": 0.0,
      "startOffset": {
        "logOffset": 17
      }
    }
  ],
  "stateOperators": [],
  "timestamp": "2018-04-12T21:40:10.004Z"
}
```

作为读者的练习，构建一个 PixieApp，它提供一个实时仪表盘，显示每个活跃流查询的更新详情，会很有帮助。

### 注意

**注意**：我们将在*第三部分 – 创建实时仪表盘 PixieApp*中展示如何构建这个 PixieApp。

## 从 Parquet 文件创建批处理 DataFrame

### 注意

**注意**：在本章的其余部分，我们将批处理 Spark DataFrame 定义为经典 Spark DataFrame，即非流式的。

这个流计算流程的最后一步是创建一个或多个批处理 DataFrame，我们可以用来构建分析和数据可视化。我们可以将这最后一步视为对数据进行快照，以便进行更深层次的分析。

有两种方法可以通过编程方式从 Parquet 文件加载批处理 DataFrame：

+   使用`spark.read`（注意，我们不再像之前那样使用`spark.readStream`）：

    ```py
    parquet_batch_df = spark.read.parquet(os.path.join(root_dir, "output_parquet"))
    ```

+   使用`spark.sql`：

    ```py
    parquet_batch_df = spark.sql(
    "select * from parquet.'{}'".format(
    os.path.join(root_dir, "output_parquet")
    )
    )
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode10.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode10.py)

这种方法的好处是，我们可以使用任何 ANSI SQL 查询来加载数据，而不必像第一种方法那样使用等效的低级 DataFrame API。

然后，我们可以通过重新运行前面的代码并重新创建 DataFrame 来定期刷新数据。我们现在可以为数据创建进一步的分析，例如，通过在数据上运行 PixieDust 的`display()`方法来生成可视化图表：

```py
import pixiedust
display(parquet_batch_df)
```

我们选择**条形图**菜单，并将`source`字段拖到**Keys**字段区域。由于我们只想显示前 10 条推文，因此我们在**要显示的行数**字段中设置这个值。下图显示了 PixieDust 选项对话框：

![从 Parquet 文件创建批处理 DataFrame](img/B09699_07_06.jpg)

显示前 10 个推文来源的选项对话框

点击**确定**后，我们会看到以下结果：

![从 Parquet 文件创建批处理 DataFrame](img/B09699_07_07.jpg)

展示与棒球相关的推文数量按来源分类的图表

在这一部分中，我们已经展示了如何使用 Tweepy 库创建 Twitter 流，清洗原始数据并将其存储在 CSV 文件中，创建 Spark Streaming DataFrame，在其上运行流查询并将输出存储在 Parquet 数据库中，从 Parquet 文件创建批处理 DataFrame，并使用 PixieDust 的`display()`方法进行数据可视化。

### 注意

*第一部分 – 使用 Spark 结构化流获取数据*的完整笔记本可以在这里找到：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/Twitter%20Sentiment%20Analysis%20-%20Part%201.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/Twitter%20Sentiment%20Analysis%20-%20Part%201.ipynb)

在下一部分中，我们将探讨如何使用 IBM Watson 自然语言理解服务，丰富数据中的情感分析和实体提取。

# 第二部分 - 使用情感和最相关的提取实体丰富数据

在这一部分，我们将推特数据与情感信息进行丰富处理，例如，*正面*，*负面*和*中性*。我们还希望从推文中提取出最相关的实体，例如，运动，组织和地点。这些额外的信息将通过我们在下一部分构建的实时仪表板进行分析和可视化。从非结构化文本中提取情感和实体所使用的算法属于计算机科学和人工智能领域，称为**自然语言处理**（**NLP**）。网上有许多教程提供了提取情感的算法示例。例如，您可以在 scikit-learn 的 GitHub 仓库找到一个全面的文本分析教程，链接为[`github.com/scikit-learn/scikit-learn/blob/master/doc/tutorial/text_analytics/working_with_text_data.rst`](https://github.com/scikit-learn/scikit-learn/blob/master/doc/tutorial/text_analytics/working_with_text_data.rst)。

然而，对于这个示例应用程序，我们不会构建自己的 NLP 算法。而是选择一个提供文本分析（如情感和实体提取）的云服务。当您的需求比较通用，不需要训练自定义模型时，这种方法效果很好，尽管即便如此，许多服务提供商现在也提供了相关工具来完成此类任务。使用云服务提供商相比自己创建模型具有显著优势，比如节省开发时间、提高准确性和性能。通过简单的 REST 调用，我们可以生成所需数据并将其集成到应用程序流程中。如果需要，切换服务提供商也非常容易，因为与服务接口的代码已经很好地隔离。

对于这个示例应用程序，我们将使用**IBM Watson 自然语言理解**（**NLU**）服务，它是 IBM Watson 认知服务家族的一部分，并且可以在 IBM Cloud 上使用。

## 开始使用 IBM Watson 自然语言理解服务

为新服务提供资源的过程对于每个云服务提供商通常都是相同的。登录后，您将进入服务目录页面，在那里可以搜索特定的服务。

要登录到 IBM Cloud，只需访问[`console.bluemix.net`](https://console.bluemix.net)，如果还没有 IBM 账户，可以创建一个免费的账户。进入仪表板后，有多种方式可以搜索 IBM Watson NLU 服务：

+   点击左上角菜单，选择**Watson**，点击**浏览服务**，然后在服务列表中找到**自然语言理解**条目。

+   点击右上角的**创建资源**按钮进入目录。一旦进入目录，你可以在搜索栏中搜索`Natural Language Understanding`，如以下截图所示：![开始使用 IBM Watson 自然语言理解服务](img/B09699_07_08.jpg)

    在服务目录中搜索 Watson NLU

然后，你可以点击**自然语言理解**来配置一个新实例。云服务提供商通常会为一些服务提供免费的或基于试用的计划，幸运的是，Watson NLU 也提供了这样的计划，但有限制，你只能训练一个自定义模型，每月最多处理 30,000 个 NLU 项目（对于我们的示例应用足够了）。选择**Lite**（免费）计划并点击**创建**按钮后，新配置的实例将出现在仪表盘上，并准备好接受请求。

### 注意

**注意**：创建服务后，你可能会被重定向到 NLU 服务的*入门文档*。如果是这种情况，只需返回仪表盘，应该能看到新创建的服务实例。

下一步是通过在笔记本中发出 REST 调用来测试服务。每个服务都会提供详细的文档，说明如何使用，包括 API 参考。在笔记本中，我们可以使用 requests 包根据 API 参考发出 GET、POST、PUT 或 DELETE 请求，但强烈建议检查服务是否提供具有高级编程访问功能的 SDK。

幸运的是，IBM Watson 提供了`watson_developer_cloud`开源库，其中包含多个支持流行编程语言（包括 Java、Python 和 Node.js）的开源 SDK。对于本项目，我们将使用 Python SDK，源代码和示例代码可以在此找到：[`github.com/watson-developer-cloud/python-sdk`](https://github.com/watson-developer-cloud/python-sdk)。

以下`pip`命令直接从 Jupyter Notebook 安装`watson_developer_cloud`包：

```py
!pip install Watson_developer_cloud

```

### 注意

请注意命令前的`!`，它表示这是一个 shell 命令。

**注意**：安装完成后，别忘了重新启动内核。

大多数云服务提供商使用一种通用模式，允许用户通过服务控制台仪表盘生成一组凭证，然后将其嵌入到客户端应用程序中。要生成凭证，只需点击 Watson NLU 实例的**服务凭证**标签，然后点击**新建凭证**按钮。

这将生成一组新的凭证，格式为 JSON，如下截图所示：

![开始使用 IBM Watson 自然语言理解服务](img/B09699_07_09.jpg)

为 Watson NLU 服务生成新凭证

现在我们已经拥有了服务的凭据，我们可以创建一个 `NaturalLanguageUnderstandingV1` 对象，它将提供对 REST API 的编程访问，如下所示的代码所示：

```py
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, SentimentOptions, EntitiesOptions

nlu = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='XXXX',
    password='XXXX'
)
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode11.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode11.py)

**注意**：在前面的代码中，将 `XXXX` 文本替换为服务凭据中的适当用户名和密码。

`version` 参数指的是 API 的特定版本。要了解最新版本，请访问此处的官方文档页面：

[`www.ibm.com/watson/developercloud/natural-language-understanding/api/v1`](https://www.ibm.com/watson/developercloud/natural-language-understanding/api/v1)

在继续构建应用程序之前，让我们花点时间了解 Watson 自然语言服务所提供的文本分析功能，包括：

+   情感

+   实体

+   概念

+   类别

+   情感

+   关键词

+   关系

+   语义角色

在我们的应用程序中，Twitter 数据的丰富化发生在 `RawTweetsListener` 中，我们在其中创建了一个 `enrich` 方法，该方法将从 `on_data` 处理程序方法中调用。在这个方法中，我们使用 Twitter 数据和仅包含情感和实体的特征列表调用 `nlu.analyze` 方法，如下所示的代码所示：

### 注意

**注意**：`[[RawTweetsListener]]` 符号表示以下代码是一个名为 `RawTweetsListener` 的类的一部分，用户不应尝试在没有完整类的情况下直接运行代码。像往常一样，您可以参考完整的笔记本进行查看。

```py
[[RawTweetsListener]]
def enrich(self, data):
    try:
        response = nlu.analyze(
 text = data['text'],
 features = Features(
 sentiment=SentimentOptions(),
 entities=EntitiesOptions()
 )
 )
        data["sentiment"] = response["sentiment"]["document"]["label"]
        top_entity = response["entities"][0] if len(response["entities"]) > 0 else None
        data["entity"] = top_entity["text"] if top_entity is not None else ""
        data["entity_type"] = top_entity["type"] if top_entity is not None else ""
        return data
    except Exception as e:
 self.warn("Error from Watson service while enriching data: {}".format(e))

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode12.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode12.py)

结果将存储在 `data` 对象中，随后会写入 CSV 文件。我们还会防范意外异常，跳过当前推文并记录警告信息，而不是让异常冒泡，从而停止 Twitter 流。

### 注意

**注意**：最常见的异常发生在推文数据使用该服务不支持的语言时。

我们使用在第五章中描述的 `@Logger` 装饰器，*Python 和 PixieDust 最佳实践与高级概念*，通过 PixieDust 日志框架记录日志消息。提醒一下，您可以使用来自另一个单元的 `%pixiedustLog` 魔法命令来查看日志消息。

我们仍然需要更改模式元数据以包括新的字段，如下所示：

```py
field_metadata = [
    {"name": "created_at", "type": DateType()},
    {"name": "text", "type": StringType()},
    {"name": "source", "type": StringType(),
         "transform": lambda s: BS(s, "html.parser").text.strip()
    },
 {"name": "sentiment", "type": StringType()},
 {"name": "entity", "type": StringType()},
 {"name": "entity_type", "type": StringType()}
]
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode13.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode13.py)

最后，我们更新`on_data`处理程序以调用`enrich`方法，如下所示：

```py
def on_data(self, data):
    def transform(key, value):
        return transformskey if key in transforms else value
    data = self.enrich(json.loads(data))
 if data is not None:
        self.buffered_data.append(
            {key:transform(key,value) \
                for key,value in iteritems(data) \
                if key in fieldnames}
        )
        self.flush_buffer_if_needed()
    return True

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode14.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode14.py)

当我们重新启动 Twitter 流并创建 Spark Streaming DataFrame 时，我们可以通过以下代码验证我们是否有正确的模式：

```py
schema = StructType(
    [StructField(f["name"], f["type"], True) for f in field_metadata]
)
csv_sdf = spark.readStream \
    .csv(
        output_dir,
        schema=schema,
        multiLine = True,
        dateFormat = 'EEE MMM dd kk:mm:ss Z y',
        ignoreTrailingWhiteSpace = True,
        ignoreLeadingWhiteSpace = True
    )
csv_sdf.printSchema()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode15.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode15.py)

这将显示如下结果，如预期：

```py
root
 |-- created_at: date (nullable = true)
 |-- text: string (nullable = true)
 |-- source: string (nullable = true)
 |-- sentiment: string (nullable = true)
 |-- entity: string (nullable = true)
 |-- entity_type: string (nullable = true)

```

类似地，当我们使用`console`接收器运行结构化查询时，数据将按批次显示在 Spark 主节点的控制台中，如下所示：

```py
-------------------------------------------
Batch: 2
-------------------------------------------
+----------+---------------+---------------+---------+------------+-------------+
|created_at|           text|         source|sentiment|      entity|  entity_type|
+----------+---------------+---------------+---------+------------+-------------+
|2018-04-14|Some little ...| Twitter iPhone| positive|        Drew|       Person|d
|2018-04-14|RT @XXXXXXXX...| Twitter iPhone|  neutral| @XXXXXXXXXX|TwitterHandle|
|2018-04-14|RT @XXXXXXXX...| Twitter iPhone|  neutral|    baseball|        Sport|
|2018-04-14|RT @XXXXXXXX...| Twitter Client|  neutral| @XXXXXXXXXX|TwitterHandle|
|2018-04-14|RT @XXXXXXXX...| Twitter Client| positive| @XXXXXXXXXX|TwitterHandle|
|2018-04-14|RT @XXXXX: I...|Twitter Android| positive| Greg XXXXXX|       Person|
|2018-04-14|RT @XXXXXXXX...| Twitter iPhone| positive| @XXXXXXXXXX|TwitterHandle|
|2018-04-14|RT @XXXXX: I...|Twitter Android| positive| Greg XXXXXX|       Person|
|2018-04-14|Congrats to ...|Twitter Android| positive|    softball|        Sport|
|2018-04-14|translation:...| Twitter iPhone|  neutral|        null|         null|
+----------+---------------+---------------+---------+------------+-------------+
```

最后，我们使用 Parquet 的`output`接收器运行结构化查询，创建一个批量 DataFrame，并使用 PixieDust 的`display()`探索数据，展示例如按情感（`positive`，`negative`，`neutral`）聚类的推文计数，如下图所示：

![开始使用 IBM Watson 自然语言理解服务](img/B09699_07_10.jpg)

显示按情感分类的推文数的条形图，按实体聚类

### 注意

完整的笔记本《*第二部分——通过情感和最相关的提取实体丰富数据*》位于此处：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/Twitter%20Sentiment%20Analysis%20-%20Part%202.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/Twitter%20Sentiment%20Analysis%20-%20Part%202.ipynb)

如果你正在运行它，我鼓励你通过向模式添加更多字段、运行不同的 SQL 查询，并使用 PixieDust 的`display()`来可视化数据进行实验。

在接下来的部分，我们将构建一个展示 Twitter 数据多个指标的仪表盘。

# 第三部分——创建实时仪表盘 PixieApp

一如既往，我们首先需要定义 MVP 版本仪表盘的需求。这次我们将借用敏捷方法中的一个工具，称为**用户故事**，它从用户的角度描述我们希望构建的功能。敏捷方法还要求我们通过将不同的用户分类为角色，充分理解与软件互动的用户的背景。在我们的案例中，我们只使用一个角色：*Frank，市场营销总监，想要实时了解消费者在社交媒体上讨论的内容*。

用户故事是这样的：

+   Frank 输入类似产品名称的搜索查询

+   然后，展示一个仪表板，显示一组图表，展示有关用户情绪（正面、负面、中立）的度量

+   仪表板还包含一个展示所有在推文中提到的实体的词云

+   此外，仪表板还提供了一个选项，可以显示当前所有活跃的 Spark Streaming 查询的实时进度

### 注意

**注意**：最后一个功能对于 Frank 来说并不是必需的，但我们还是在这里展示它，作为之前练习的示例实现。

## 将分析功能重构为独立的方法

在开始之前，我们需要将启动 Twitter 流和创建 Spark Streaming 数据框的代码重构为独立的方法，并在 PixieApp 中调用这些方法。

`start_stream,` `start_streaming_dataframe` 和 `start_parquet_streaming_query` 方法如下：

```py
def start_stream(queries):
    "Asynchronously start a new Twitter stream"
    stream = Stream(auth, RawTweetsListener())
    stream.filter(track=queries, languages=["en"], async=True)
    return stream
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode16.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode16.py)

```py
def start_streaming_dataframe(output_dir):
    "Start a Spark Streaming DataFrame from a file source"
    schema = StructType(
        [StructField(f["name"], f["type"], True) for f in field_metadata]
    )
    return spark.readStream \
        .csv(
            output_dir,
            schema=schema,
            multiLine = True,
            timestampFormat = 'EEE MMM dd kk:mm:ss Z yyyy',
            ignoreTrailingWhiteSpace = True,
            ignoreLeadingWhiteSpace = True
        )
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode17.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode17.py)

```py
def start_parquet_streaming_query(csv_sdf):
    """
    Create and run a streaming query from a Structured DataFrame
    outputing the results into a parquet database
    """
    streaming_query = csv_sdf \
      .writeStream \
      .format("parquet") \
      .option("path", os.path.join(root_dir, "output_parquet")) \
      .trigger(processingTime="2 seconds") \
      .option("checkpointLocation", os.path.join(root_dir, "output_chkpt")) \
      .start()
    return streaming_query
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode18.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode18.py)

作为准备工作的一部分，我们还需要管理 PixieApp 将要创建的不同流的生命周期，并确保在用户重新启动仪表板时，底层资源被正确停止。为此，我们创建了一个`StreamsManager`类，封装了 Tweepy 的`twitter_stream`和 CSV 流数据框。这个类有一个`reset`方法，它会停止`twitter_stream`，停止所有活动的流查询，删除先前查询创建的所有输出文件，并使用新的查询字符串启动一个新的流。如果`reset`方法在没有查询字符串的情况下被调用，我们将不会启动新的流。

我们还创建了一个全局的`streams_manager`实例，它将跟踪当前状态，即使仪表板被重新启动。由于用户可以重新运行包含全局`streams_manager`的单元，我们需要确保在当前全局实例被删除时，`reset`方法会自动调用。为此，我们重写了对象的`__del__`方法，这是 Python 实现析构函数的一种方式，并调用`reset`。

`StreamsManager`的代码如下：

```py
class StreamsManager():
    def __init__(self):
        self.twitter_stream = None
        self.csv_sdf = None

    def reset(self, search_query = None):
        if self.twitter_stream is not None:
            self.twitter_stream.disconnect()
        #stop all the active streaming queries and re_initialize the directories
        for query in spark.streams.active:
            query.stop()
        # initialize the directories
        self.root_dir, self.output_dir = init_output_dirs()
        # start the tweepy stream
        self.twitter_stream = start_stream([search_query]) if search_query is not None else None
        # start the spark streaming stream
        self.csv_sdf = start_streaming_dataframe(output_dir) if search_query is not None else None

 def __del__(self):
 # Automatically called when the class is garbage collected
 self.reset()

streams_manager = StreamsManager()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode19.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode19.py)

## 创建 PixieApp

如同在第六章，*分析研究：TensorFlow 下的 AI 和图像识别*，我们再次使用 `TemplateTabbedApp` 类来创建一个包含两个 PixieApp 的标签布局：

+   `TweetInsightApp`：允许用户指定查询字符串并显示与之关联的实时仪表盘

+   `StreamingQueriesApp`：监控活动结构化查询的进度

在 `TweetInsightApp` 的默认路由中，我们返回一个片段，提示用户输入查询字符串，如下所示：

```py
from pixiedust.display.app import *
@PixieApp
class TweetInsightApp():
    @route()
    def main_screen(self):
        return """
<style>
    div.outer-wrapper {
        display: table;width:100%;height:300px;
    }
    div.inner-wrapper {
        display: table-cell;vertical-align: middle;height: 100%;width: 100%;
    }
</style>
<div class="outer-wrapper">
    <div class="inner-wrapper">
        <div class="col-sm-3"></div>
        <div class="input-group col-sm-6">
          <input id="query{{prefix}}" type="text" class="form-control"
              value=""
              placeholder="Enter a search query (e.g. baseball)">
          <span class="input-group-btn">
            <button class="btn btn-default" type="button"
 pd_options="search_query=$val(query{{prefix}})">
                Go
            </button>
          </span>
        </div>
    </div>
</div>
        """

TweetInsightApp().run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode20.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode20.py)

以下截图显示了运行上述代码后的结果：

### 注意

**注意**：稍后我们会创建主 `TwitterSentimentApp` PixieApp，它具有标签布局，并包含此类。在此之前，我们只展示 `TweetInsightApp` 子应用程序的独立功能。

![创建 PixieApp](img/B09699_07_11.jpg)

Twitter 情感仪表盘的欢迎界面

在 `Go` 按钮中，我们通过用户提供的查询字符串调用 `search_query` 路由。在这个路由中，我们首先启动各种流并创建一个批量数据框，该数据框从 Parquet 数据库所在的输出目录中存储为一个类变量，命名为 `parquet_df`。然后，我们返回由三个小部件组成的 HTML 片段，展示以下指标：

+   按照实体分组的三种情感的柱状图

+   显示推文情感分布的折线图子图

+   用于实体的词云

每个小部件都在使用 `pd_refresh_rate` 属性定期调用特定的路由，相关文档可以参考第五章，*Python 和 PixieDust 最佳实践与高级概念*。我们还确保重新加载 `parquet_df` 变量，以获取自上次加载以来到达的新数据。该变量随后在 `pd_entity` 属性中引用，用于显示图表。

以下代码展示了 `search_query` 路由的实现：

```py
import time
[[TweetInsightApp]]
@route(search_query="*")
    def do_search_query(self, search_query):
        streams_manager.reset(search_query)
        start_parquet_streaming_query(streams_manager.csv_sdf)
 while True:
 try:
 parquet_dir = os.path.join(root_dir,
 "output_parquet")
 self.parquet_df = spark.sql("select * from parquet.'{}'".format(parquet_dir))
 break
 except:
 time.sleep(5)
        return """
<div class="container">
 <div id="header{{prefix}}" class="row no_loading_msg"
 pd_refresh_rate="5000" pd_target="header{{prefix}}">
 <pd_script>
print("Number of tweets received: {}".format(streams_manager.twitter_stream.listener.tweet_count))
 </pd_script>
 </div>
    <div class="row" style="min-height:300px">
        <div class="col-sm-5">
            <div id="metric1{{prefix}}" pd_refresh_rate="10000"
                class="no_loading_msg"
                pd_options="display_metric1=true"
                pd_target="metric1{{prefix}}">
            </div>
        </div>
        <div class="col-sm-5">
            <div id="metric2{{prefix}}" pd_refresh_rate="12000"
                class="no_loading_msg"
                pd_options="display_metric2=true"
                pd_target="metric2{{prefix}}">
            </div>
        </div>
    </div>

    <div class="row" style="min-height:400px">
        <div class="col-sm-offset-1 col-sm-10">
            <div id="word_cloud{{prefix}}" pd_refresh_rate="20000"
                class="no_loading_msg"
                pd_options="display_wc=true"
                pd_target="word_cloud{{prefix}}">
            </div>
        </div>
    </div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode21.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode21.py)

从上述代码中有多个需要注意的地方：

+   当我们尝试加载 `parquet_df` 批数据框时，Parquet 文件的输出目录可能尚未准备好，这会导致异常。为了解决这个时序问题，我们将代码包裹在 `try...except` 语句中，并使用 `time.sleep(5)` 等待 5 秒钟。

+   我们还在页头显示当前推文的数量。为此，我们添加了一个每 5 秒刷新一次的`<div>`元素，并且在该元素中使用 `<pd_script>` 来打印当前的推文数量，使用 `streams_manager.twitter_stream.listener.tweet_count` 变量，它是我们在 `RawTweetsListener` 类中添加的变量。我们还更新了 `on_data()` 方法，以便每次新推文到达时增加 `tweet_count` 变量，以下代码展示了这一过程：

    ```py
    [[TweetInsightApp]]
    def on_data(self, data):
            def transform(key, value):
                return transformskey if key in transforms else value
            data = self.enrich(json.loads(data))
            if data is not None:
     self.tweet_count += 1
                self.buffered_data.append(
                    {key:transform(key,value) \
                         for key,value in iteritems(data) \
                         if key in fieldnames}
                )
                self.flush_buffer_if_needed()
            return True
    ```

    同时，为了避免闪烁，我们通过在 `<div>` 元素中使用 `class="no_loading_msg"` 来阻止显示 *加载旋转图标* 图像。

+   我们调用了三个不同的路由（`display_metric1`，`display_metric2` 和 `display_wc`），它们分别负责显示三个小部件。

    `display_metric1` 和 `display_metric2` 路由非常相似。它们返回一个包含`parquet_df`作为`pd_entity`的`div`，以及一个自定义的 `<pd_options>` 子元素，该元素包含传递给 PixieDust `display()` 层的 JSON 配置。

以下代码展示了 `display_metric1` 路由的实现：

```py
[[TweetInsightApp]]
@route(display_metric1="*")
    def do_display_metric1(self, display_metric1):
        parquet_dir = os.path.join(root_dir, "output_parquet")
        self.parquet_df = spark.sql("select * from parquet.'{}'".format(parquet_dir))
        return """
<div class="no_loading_msg" pd_render_onload pd_entity="parquet_df">
    <pd_options>
    {
      "legend": "true",
      "keyFields": "sentiment",
      "clusterby": "entity_type",
      "handlerId": "barChart",
      "rendererId": "bokeh",
      "rowCount": "10",
      "sortby": "Values DESC",
      "noChartCache": "true"
    }
    </pd_options>
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode22.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode22.py)

`display_metric2` 路由遵循类似的模式，但使用了不同的一组 `pd_options` 属性。

最后一条路由是 `display_wc`，负责显示实体的词云。该路由使用 `wordcloud` Python 库，你可以通过以下命令安装它：

```py
!pip install wordcloud

```

### 注意

**注意**：一如既往，安装完成后不要忘记重启内核。

我们使用了在第五章中记录的 `@captureOutput` 装饰器，*Python 和 PixieDust 最佳实践与高级概念*，如以下所示：

```py
import matplotlib.pyplot as plt
from wordcloud import WordCloud

[[TweetInsightApp]]
@route(display_wc="*")
@captureOutput
def do_display_wc(self):
    text = "\n".join(
 [r['entity'] for r in self.parquet_df.select("entity").collect() if r['entity'] is not None]
 )
    plt.figure( figsize=(13,7) )
    plt.axis("off")
    plt.imshow(
        WordCloud(width=750, height=350).generate(text),
        interpolation='bilinear'
    )
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode23.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode23.py)

传递给 `WordCloud` 类的文本是通过收集 `parquet_df` 批处理 DataFrame 中的所有实体生成的。

以下截图展示了在使用搜索查询 `baseball` 创建的 Twitter 流运行一段时间后的仪表盘：

![创建 PixieApp](img/B09699_07_12.jpg)

用于搜索查询“baseball”的 Twitter 情感仪表盘

第二个 PixieApp 用于监控正在积极运行的流查询。主路由返回一个 HTML 片段，该片段包含一个 `<div>` 元素，该元素定期（每 5000 毫秒）调用 `show_progress` 路由，如以下代码所示：

```py
@PixieApp
class StreamingQueriesApp():
    @route()
    def main_screen(self):
        return """
<div class="no_loading_msg" pd_refresh_rate="5000" pd_options="show_progress=true">
</div>
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode24.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode24.py)

在`show_progress`路由中，我们使用了本章之前描述的`query.lastProgress`监控 API，通过 Jinja2 `{%for%}` 循环遍历 JSON 对象，并如以下代码所示在表格中显示结果：

```py
@route(show_progress="true")
    def do_show_progress(self):
        return """
{%for query in this.spark.streams.active%}
    <div>
    <div class="page-header">
        <h1>Progress Report for Spark Stream: {{query.id}}</h1>
    <div>
    <table>
        <thead>
          <tr>
             <th>metric</th>
             <th>value</th>
          </tr>
        </thead>
        <tbody>
 {%for key, value in query.lastProgress.items()%}
 <tr>
 <td>{{key}}</td>
 <td>{{value}}</td>
 </tr>
 {%endfor%}
        </tbody>
    </table>
{%endfor%}
        """
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode25.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode25.py)

以下截图显示了 PixieApp 的流查询监控：

![创建 PixieApp](img/B09699_07_13.jpg)

实时监控活动的 Spark 流查询

最后一步是使用`TemplateTabbedApp`类来集成完整的应用程序，如下所示的代码：

```py
from pixiedust.display.app import *
from pixiedust.apps.template import TemplateTabbedApp

@PixieApp
class TwitterSentimentApp(TemplateTabbedApp):
    def setup(self):
 self.apps = [
 {"title": "Tweets Insights", "app_class": "TweetInsightApp"},
 {"title": "Streaming Queries", "app_class": "StreamingQueriesApp"}
 ]

app = TwitterSentimentApp()
app.run()
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode26.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode26.py)

我们的示例应用程序第三部分现已完成；您可以在这里找到完整的 Notebook：

### 注意

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/Twitter%20Sentiment%20Analysis%20-%20Part%203.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/Twitter%20Sentiment%20Analysis%20-%20Part%203.ipynb)

在下一部分，我们将讨论如何通过使用 Apache Kafka 进行事件流处理和 IBM Streams Designer 对流数据进行数据增强来使应用程序的数据管道更加可扩展。

# 第四部分 – 使用 Apache Kafka 和 IBM Streams Designer 增加可扩展性

### 注意

**注意**：本节为可选部分。它演示了如何通过使用基于云的流服务重新实现数据管道的部分，以实现更大的可扩展性。

在单个 Notebook 中实现整个数据管道使我们在开发和测试过程中具有很高的生产力。我们可以快速实验代码并测试更改，且占用的资源非常小。由于我们使用的是相对较小的数据量，性能也很合理。然而，显然我们不会在生产环境中使用这种架构，接下来我们需要问自己的是，随着来自 Twitter 的流数据量急剧增加，哪些瓶颈会阻碍应用程序的扩展。

在本节中，我们确定了两个改进的方向：

+   在 Tweepy 流中，传入的数据会通过 `on_data` 方法发送到 `RawTweetsListener` 实例进行处理。我们需要确保在此方法中尽量减少时间消耗，否则随着传入数据量的增加，系统将会落后。在当前的实现中，数据是通过外部调用 Watson NLU 服务同步丰富的，然后将数据缓冲，最终写入磁盘。为了解决这个问题，我们将数据发送到 Kafka 服务，这是一个高可扩展性、容错的流平台，使用发布/订阅模式来处理大量数据。我们还使用了 Streaming Analytics 服务，它将从 Kafka 消费数据并通过调用 Watson NLU 服务来丰富数据。两个服务都可以在 IBM Cloud 上使用。

    ### 注意

    **注意**：我们可以使用其他开源框架来处理流数据，例如 Apache Flink（[`flink.apache.org`](https://flink.apache.org)）或 Apache Storm（[`storm.apache.org`](http://storm.apache.org)）。

+   在当前实现中，数据以 CSV 文件形式存储，我们使用输出目录作为源创建一个 Spark Streaming DataFrame。这个步骤会消耗 Notebook 和本地环境的时间和资源。相反，我们可以让 Streaming Analytics 将丰富后的事件写回到不同的主题，并创建一个以 Message Hub 服务作为 Kafka 输入源的 Spark Streaming DataFrame。

下图展示了我们示例应用程序的更新架构：

![Part 4 – 使用 Apache Kafka 和 IBM Streams Designer 添加可扩展性](img/B09699_07_14.jpg)

使用 Kafka 和 Streams Designer 扩展架构

在接下来的几个部分中，我们将实现更新后的架构，首先将推文流式传输到 Kafka。

## 将原始推文流式传输到 Kafka

在 IBM Cloud 上配置 Kafka / Message Hub 服务实例的过程与我们配置 Watson NLU 服务时的步骤相同。首先，我们在目录中找到并选择该服务，选择定价计划后点击 **创建**。然后，我们打开服务仪表板，选择 **服务凭证** 标签以创建新的凭证，如下图所示：

![将原始推文流式传输到 Kafka](img/B09699_07_15.jpg)

为 Message Hub 服务创建新的凭证

与 IBM Cloud 上的所有服务一样，凭证以 JSON 对象的形式提供，我们需要将其存储在 Notebook 中的一个变量里，代码如下所示（同样，别忘了将 `XXXX` 替换为您的用户名和服务凭证中的密码）：

```py
message_hub_creds = {
  "instance_id": "XXXXX",
  "mqlight_lookup_url": "https://mqlight-lookup-prod02.messagehub.services.us-south.bluemix.net/Lookup?serviceId=XXXX",
  "api_key": "XXXX",
  "kafka_admin_url": "https://kafka-admin-prod02.messagehub.services.us-south.bluemix.net:443",
  "kafka_rest_url": "https://kafka-rest-prod02.messagehub.services.us-south.bluemix.net:443",
  "kafka_brokers_sasl": [
    "kafka03-prod02.messagehub.services.us-south.bluemix.net:9093",
    "kafka01-prod02.messagehub.services.us-south.bluemix.net:9093",
    "kafka02-prod02.messagehub.services.us-south.bluemix.net:9093",
    "kafka05-prod02.messagehub.services.us-south.bluemix.net:9093",
    "kafka04-prod02.messagehub.services.us-south.bluemix.net:9093"
  ],
  "user": "XXXX",
  "password": "XXXX"
}
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode27.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode27.py)

关于与 Kafka 的接口，我们可以选择多个优秀的客户端库。我尝试了很多，但最终我使用得最多的是`kafka-python`（[`github.com/dpkp/kafka-python`](https://github.com/dpkp/kafka-python)），它的优势是纯 Python 实现，因此更容易安装。

要从 Notebook 安装它，请使用以下命令：

```py
!pip install kafka-python

```

### 注

**注**：像往常一样，在安装任何库之后，不要忘记重启内核。

`kafka-python`库提供了一个`KafkaProducer`类，用于将数据作为消息写入服务，我们需要用之前创建的凭证来配置它。Kafka 有多个配置选项，涵盖所有选项超出了本书的范围。所需的选项与身份验证、主机服务器和 API 版本相关。

以下代码实现了`RawTweetsListener`类的`__init__`构造函数。它创建了一个`KafkaProducer`实例并将其存储为类变量：

```py
[[RawTweetsListener]]
context = ssl.create_default_context()
context.options &= ssl.OP_NO_TLSv1
context.options &= ssl.OP_NO_TLSv1_1
kafka_conf = {
    'sasl_mechanism': 'PLAIN',
    'security_protocol': 'SASL_SSL',
    'ssl_context': context,
    "bootstrap_servers": message_hub_creds["kafka_brokers_sasl"],
    "sasl_plain_username": message_hub_creds["user"],
    "sasl_plain_password": message_hub_creds["password"],
    "api_version":(0, 10, 1),
    "value_serializer" : lambda v: json.dumps(v).encode('utf-8')
}
self.producer = KafkaProducer(**kafka_conf)

```

### 注

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode28.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode28.py)

我们为`value_serializer`键配置了一个 lambda 函数，用于序列化 JSON 对象，这是我们将用于数据的格式。

### 注

**注**：我们需要指定`api_version`键，否则库会尝试自动发现其值，这会导致由于`kafka-python`库中的一个 bug（只在 Mac 上可复现）引发`NoBrokerAvailable`异常。编写本书时，尚未提供该 bug 的修复。

现在，我们需要更新`on_data`方法，通过使用`tweets`主题将推文数据发送到 Kafka。Kafka 主题就像一个频道，应用程序可以发布或订阅它。在尝试向主题写入之前，确保该主题已经创建，否则会引发异常。此操作在以下`ensure_topic_exists`方法中完成：

```py
import requests
import json

def ensure_topic_exists(topic_name):
    response = requests.post(
 message_hub_creds["kafka_rest_url"] +
 "/admin/topics",
 data = json.dumps({"name": topic_name}),
 headers={"X-Auth-Token": message_hub_creds["api_key"]}
 )
    if response.status_code != 200 and \
       response.status_code != 202 and \
       response.status_code != 422 and \
       response.status_code != 403:
        raise Exception(response.json())
```

### 注

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode29.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode29.py)

在前面的代码中，我们向路径`/admin/topic`发出了一个 POST 请求，载荷为包含我们想要创建的主题名称的 JSON 数据。请求必须使用凭证中提供的 API 密钥和`X-Auth-Token`头进行身份验证。我们还确保忽略 HTTP 错误码 422 和 403，它们表示该主题已经存在。

`on_data`方法的代码现在看起来简单得多，如下所示：

```py
[[RawTweetsListener]]
def on_data(self, data):
    self.tweet_count += 1
 self.producer.send(
 self.topic,
 {key:transform(key,value) \
 for key,value in iteritems(json.loads(data)) \
 if key in fieldnames}
 )
    return True
```

### 注

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode30.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode30.py)

如我们所见，通过这段新代码，我们在`on_data`方法中所花费的时间最少，这是我们想要实现的目标。推文数据现在正在流入 Kafka 的`tweets`主题，准备通过我们将在下一节讨论的流式分析服务进行丰富化。

## 使用流式分析服务丰富推文数据

在这一步，我们需要使用 Watson Studio，这是一个集成的基于云的 IDE，提供多种数据处理工具，包括机器学习/深度学习模型、Jupyter Notebooks、流式数据流等。Watson Studio 是 IBM Cloud 的一个配套工具，可以通过[`datascience.ibm.com`](https://datascience.ibm.com)访问，因此无需额外注册。

登录到 Watson Studio 后，我们创建一个新的项目，命名为`Thoughtful Data Science`。

### 注意

**注意**：创建项目时，选择默认选项是可以的。

然后，我们进入**设置**标签页创建一个流式分析服务，它将成为驱动我们丰富化过程的引擎，并将其与项目关联。请注意，我们也可以像为本章中其他服务一样，在 IBM Cloud 目录中创建该服务，但由于我们仍然需要将其与项目关联，最好也在 Watson Studio 中进行创建。

在**设置**标签页中，我们向下滚动到**关联服务**部分，点击**添加服务**下拉菜单，选择**流式分析**。在接下来的页面中，您可以选择**现有**和**新建**。选择**新建**并按照步骤创建服务。创建完成后，新创建的服务应已与项目关联，如下图所示：

### 注意

**注意**：如果有多个免费选项，可以任选其一。

![使用流式分析服务丰富推文数据](img/B09699_07_16.jpg)

将流式分析服务与项目关联

现在我们准备创建定义推文数据丰富处理的流式数据流。

我们进入**资源**标签页，向下滚动到**流式数据流**部分，点击**新建流式数据流**按钮。在接下来的页面中，我们为其命名，选择流式分析服务，选择**手动**并点击**创建**按钮。

现在我们在流式设计器中，它由左侧的操作符调色板和一个可以用来图形化构建流式数据流的画布组成。对于我们的示例应用程序，我们需要从调色板中选择三个操作符并将它们拖放到画布上：

+   **调色板中的源部分的消息中心**：我们数据的输入源。进入画布后，我们将其重命名为`Source Message Hub`（通过双击进入编辑模式）。

+   **处理和分析部分的代码**：它将包含调用 Watson NLU 服务的数据丰富化 Python 代码。我们将操作符重命名为`Enrichment`。

+   **来自调色板的目标部分中的 Message Hub**：丰富数据的输出源。我们将其重命名为`目标 Message Hub`。

接下来，我们创建**源 Message Hub**与**丰富**之间，以及**丰富**与**目标 Message Hub**之间的连接。要创建两个操作符之间的连接，只需将第一个操作符末尾的输出端口拖动到另一个操作符的输入端口。请注意，源操作符右侧只有一个输出端口，表示它仅支持外部连接，而目标操作符左侧只有一个输入端口，表示它仅支持内部连接。**处理与分析**部分的任何操作符都有左右两个端口，因为它们同时接受和发送连接。

以下截图显示了完整的画布：

![使用流分析服务丰富推文数据](img/B09699_07_17.jpg)

推文丰富流处理

现在让我们看一下这三个操作符的配置。

### 注意

**注意**：要完成此部分，请确保运行生成主题的代码，并将其发送到我们在前一部分讨论过的 Message Hub 实例。否则，Message Hub 实例将为空，且无法检测到任何模式。

点击源 Message Hub。右侧会出现一个动画窗格，提供选择包含推文的 Message Hub 实例的选项。第一次使用时，您需要创建与 Message Hub 实例的连接。选择`tweets`作为主题。点击**编辑输出模式**，然后点击**检测模式**，以从数据中自动填充模式。您还可以使用**显示预览**按钮预览实时流数据，如下图所示：

![使用流分析服务丰富推文数据](img/B09699_07_18.jpg)

设置模式并预览实时流数据

现在选择**代码**操作符，执行调用 Watson NLU 的代码。右侧的动画上下文窗格包含一个 Python 代码编辑器，其中包含所需实现的模板代码，分别是`init(state)`和`process(event, state)`函数。

在`init`方法中，我们实例化了`NaturalLanguageUnderstandingV1`实例，如下代码所示：

```py
import sys
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, SentimentOptions, EntitiesOptions

# init() function will be called once on pipeline initialization
# @state a Python dictionary object for keeping state. The state object is passed to the process function
def init(state):
    # do something once on pipeline initialization and save in the state object
 state["nlu"] = NaturalLanguageUnderstandingV1(
 version='2017-02-27',
 username='XXXX',
 password='XXXX'
 )

```

### 注意

您可以在此处找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode31.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode31.py)

**注意**：我们需要通过位于 Python 编辑器窗口上方的**Python 包**链接安装`Watson_developer_cloud`库，如下图所示：

![使用流分析服务丰富推文数据](img/B09699_07_19.jpg)

将 watson_cloud_developer 包添加到流处理中

每次事件数据都会调用该过程方法。我们使用它来调用 Watson NLU，并将额外的信息添加到事件对象中，如下代码所示：

```py
# @event a Python dictionary object representing the input event tuple as defined by the input schema
# @state a Python dictionary object for keeping state over subsequent function calls
# return must be a Python dictionary object. It will be the output of this operator.
# Returning None results in not submitting an output tuple for this invocation.
# You must declare all output attributes in the Edit Schema window.
def process(event, state):
    # Enrich the event, such as by:
    # event['wordCount'] = len(event['phrase'].split())
    try:
        event['text'] = event['text'].replace('"', "'")
 response = state["nlu"].analyze(
 text = event['text'],
 features=Features(sentiment=SentimentOptions(), entities=EntitiesOptions())
 )
        event["sentiment"] = response["sentiment"]["document"]["label"]
        top_entity = response["entities"][0] if len(response["entities"]) > 0 else None
        event["entity"] = top_entity["text"] if top_entity is not None else ""
        event["entity_type"] = top_entity["type"] if top_entity is not None else ""
    except Exception as e:
        return None
 return event

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode32.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode32.py)

**注意**：我们还必须通过使用**编辑输出架构**链接声明所有输出变量，如下截图所示：

![通过流分析服务丰富推文数据](img/B09699_07_20.jpg)

声明所有输出变量用于代码操作符

最后，我们配置目标 Message Hub 以使用`enriched_tweets`主题。请注意，首次需要手动创建该主题，方法是进入 IBM Cloud 上的 Message Hub 实例的仪表板并点击**添加主题**按钮。

然后我们使用主工具栏中的**保存**按钮保存流。流中的任何错误，无论是代码中的编译错误、服务配置错误还是其他任何错误，都将在通知面板中显示。在确保没有错误后，我们可以使用**运行**按钮运行流，该按钮将带我们进入流数据监控屏幕。此屏幕由多个面板组成。主面板显示不同的操作符，数据以小球的形式流动在操作符之间的虚拟管道中。我们可以点击管道，在右侧面板中显示事件负载。这对于调试非常有用，因为我们可以可视化数据如何在每个操作符中进行转换。

### 注意

**注意**：Streams Designer 还支持在代码操作符中添加 Python 日志消息，然后可以将其下载到本地机器进行分析。你可以在这里了解更多关于此功能的信息：

[`dataplatform.cloud.ibm.com/docs/content/streaming-pipelines/downloading_logs.html`](https://dataplatform.cloud.ibm.com/docs/content/streaming-pipelines/downloading_logs.html)

下图显示了流式数据监控屏幕：

![通过流分析服务丰富推文数据](img/B09699_07_21.jpg)

Twitter 情感分析流数据的实时监控屏幕

现在，我们的丰富推文数据已经通过`enriched_tweets`主题流入 Message Hub 实例。在下一节中，我们将展示如何使用 Message Hub 实例作为输入源创建 Spark Streaming DataFrame。

## 使用 Kafka 输入源创建 Spark Streaming DataFrame

在最后一步中，我们创建一个 Spark Streaming DataFrame，它从 `enriched_tweets` Kafka 主题中消费经过增强的推文，这个主题属于 Message Hub 服务。为此，我们使用内置的 Spark Kafka 连接器，并在 `subscribe` 选项中指定我们想要订阅的主题。同时，我们还需要在 `kafka.bootstrap.servers` 选项中指定 Kafka 服务器的列表，这些信息通过读取我们之前创建的全局 `message_hub_creds` 变量来获取。

### 注意

**注意**：你可能已经注意到，不同的系统为此选项使用不同的名称，这使得它更容易出错。幸运的是，如果拼写错误，异常将显示一个明确的根本原因信息。

上述选项是针对 Spark Streaming 的，我们仍然需要配置 Kafka 凭证，以便较低级别的 Kafka 消费者可以与 Message Hub 服务进行正确的身份验证。为了正确地将这些消费者属性传递给 Kafka，我们不使用 `.option` 方法，而是创建一个 `kafka_options` 字典，并将其作为参数传递给加载方法，代码如下所示：

```py
def start_streaming_dataframe():
    "Start a Spark Streaming DataFrame from a Kafka Input source"
    schema = StructType(
        [StructField(f["name"], f["type"], True) for f in field_metadata]
    )
 kafka_options = {
 "kafka.ssl.protocol":"TLSv1.2",
 "kafka.ssl.enabled.protocols":"TLSv1.2",
 "kafka.ssl.endpoint.identification.algorithm":"HTTPS",
 'kafka.sasl.mechanism': 'PLAIN',
 'kafka.security.protocol': 'SASL_SSL'
 }
    return spark.readStream \
        .format("kafka") \
 .option("kafka.bootstrap.servers", ",".join(message_hub_creds["kafka_brokers_sasl"])) \
 .option("subscribe", "enriched_tweets") \
 .load(**kafka_options)

```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode33.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode33.py)

你可能认为代码到此为止就完成了，因为 Notebook 的其他部分应该与 *第三部分 – 创建实时仪表板 PixieApp* 一致。这个想法是正确的，直到我们运行 Notebook 并开始看到 Spark 抛出异常，提示 Kafka 连接器无法找到。这是因为 Kafka 连接器并不包含在 Spark 的核心发行版中，必须单独安装。

不幸的是，这类基础设施层面的问题并不直接与手头的任务相关，然而它们经常发生，我们最终花费大量时间去修复它们。在 Stack Overflow 或其他技术网站搜索通常能够快速找到解决方案，但有时答案并不显而易见。在这种情况下，由于我们是在 Notebook 中运行，而不是在 `spark-submit` 脚本中运行，因此没有太多现成的帮助，我们只能自己尝试直到找到解决方法。要安装 `spark-sql-kafka`，我们需要编辑本章前面讨论过的 `kernel.json` 文件，并将以下选项添加到 `"PYSPARK_SUBMIT_ARGS"` 项中：

```py
--packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.3.0
```

当内核重启时，这个配置将自动下载依赖并将其缓存到本地。

现在应该可以正常工作了吧？嗯，暂时还不行。我们仍然需要配置 Kafka 的安全性，以使用我们 Message Hub 服务的凭证，而该服务使用 SASL 作为安全协议。为此，我们需要提供一个**JAAS**（即**Java 认证和授权服务**）配置文件，其中包含服务的用户名和密码。Kafka 的最新版本提供了一种灵活的机制，允许使用名为`sasl.jaas.config`的消费者属性以编程方式配置安全性。不幸的是，Spark 的最新版本（截至写作时为 2.3.0）尚未更新为 Kafka 的最新版本。因此，我们必须退回到另一种配置 JAAS 的方式，即设置一个名为`java.security.auth.login.config`的 JVM 系统属性，并指向一个`jaas.conf`配置文件的路径。

我们首先在选择的目录中创建`jaas.conf`文件，并将以下内容添加到其中：

```py
KafkaClient {
    org.apache.kafka.common.security.plain.PlainLoginModule required
 username="XXXX"
 password="XXXX";
};
```

在上述内容中，将`XXXX`替换为从 Message Hub 服务凭证中获得的用户名和密码。

然后，我们将以下配置添加到`kernel.json`中的`"PYSPARK_SUBMIT_ARGS"`条目：

```py
--driver-java-options=-Djava.security.auth.login.config=<<jaas.conf path>>
```

作为参考，这里是一个包含这些配置的示例`kernel.json`：

```py
{
 "language": "python",
 "env": {
  "SCALA_HOME": "/Users/dtaieb/pixiedust/bin/scala/scala-2.11.8",
  "PYTHONPATH": "/Users/dtaieb/pixiedust/bin/spark/spark-2.3.0-bin-hadoop2.7/python/:/Users/dtaieb/pixiedust/bin/spark/spark-2.3.0-bin-hadoop2.7/python/lib/py4j-0.10.6-src.zip",
  "SPARK_HOME": "/Users/dtaieb/pixiedust/bin/spark/spark-2.3.0-bin-hadoop2.7",
  "PYSPARK_SUBMIT_ARGS": "--driver-java-options=-Djava.security.auth.login.config=/Users/dtaieb/pixiedust/jaas.conf --jars /Users/dtaieb/pixiedust/bin/cloudant-spark-v2.0.0-185.jar --driver-class-path /Users/dtaieb/pixiedust/data/libs/* --master local[10] --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.3.0 pyspark-shell",
  "PIXIEDUST_HOME": "/Users/dtaieb/pixiedust",
  "SPARK_DRIVER_MEMORY": "10G",
  "SPARK_LOCAL_IP": "127.0.0.1",
  "PYTHONSTARTUP": "/Users/dtaieb/pixiedust/bin/spark/spark-2.3.0-bin-hadoop2.7/python/pyspark/shell.py"
 },
 "display_name": "Python with Pixiedust (Spark 2.3)",
 "argv": [
  "python",
  "-m",
  "ipykernel",
  "-f",
  "{connection_file}"
 ]
}

```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode34.json`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/sampleCode34.json)

**注意**：我们在修改`kernel.json`时，应该始终重启 Notebook 服务器，以确保所有新配置能够正确重新加载。

其余的 Notebook 代码没有变化，PixieApp 仪表盘应该依然能够正常工作。

### 注意

我们现在已经完成了示例应用的第四部分；您可以在这里找到完整的笔记本：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/Twitter%20Sentiment%20Analysis%20-%20Part%204.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%207/Twitter%20Sentiment%20Analysis%20-%20Part%204.ipynb)

在本节末尾我们编写的额外代码提醒我们，与数据打交道的旅程永远不会是一条直线。我们必须准备好应对不同性质的障碍：可能是依赖库中的错误，或外部服务的限制。克服这些障碍不必让项目停滞太久。由于我们主要使用开源组件，我们可以借助像 Stack Overflow 这样的社交网站上志同道合的开发者社区，获取新的想法和代码示例，并在 Jupyter Notebook 中快速实验。

# 总结

在本章中，我们构建了一个数据管道，用于分析包含非结构化文本的大量流数据，并应用来自外部云服务的 NLP 算法来提取情感和文本中发现的其他重要实体。我们还构建了一个 PixieApp 仪表板，显示从推文中提取的实时指标和洞察。我们还讨论了多种分析大规模数据的技术，包括 Apache Spark 结构化流处理、Apache Kafka 和 IBM Streaming Analytics。像往常一样，这些示例应用程序的目标是展示如何构建数据管道，特别关注如何利用现有框架、库和云服务的可能性。

在下一章中，我们将讨论时间序列分析，这是另一个具有广泛行业应用的数据科学话题，我们将通过构建一个*金融投资组合*分析应用程序来说明这一点。
