# 第9章. 使用Blaze的多语言持久化

我们的世界是复杂的，没有一种单一的方法可以解决所有问题。同样，在数据世界中，也不能用一种技术来解决所有问题。

现在，任何大型科技公司（以某种形式）都使用MapReduce范式来筛选每天收集的数以千计（甚至数以万计）的数据。另一方面，在文档型数据库（如MongoDB）中存储、检索、扩展和更新产品信息比在关系型数据库中要容易得多。然而，在关系型数据库中持久化交易记录有助于后续的数据汇总和报告。

即使这些简单的例子也表明，解决各种商业问题需要适应不同的技术。这意味着，如果你作为数据库管理员、数据科学家或数据工程师，想要用设计来轻松解决这些问题的工具来解决这些问题，你就必须分别学习所有这些技术。然而，这并不使你的公司变得敏捷，并且容易出错，需要对你的系统进行大量的调整和破解。

Blaze抽象了大多数技术，并暴露了一个简单而优雅的数据结构和API。

在本章中，你将学习：

+   如何安装Blaze

+   多语言持久化的含义

+   如何抽象存储在文件、pandas DataFrame或NumPy数组中的数据

+   如何处理归档（GZip）

+   如何使用Blaze连接到SQL（PostgreSQL和SQLite）和No-SQL（MongoDB）数据库

+   如何查询、连接、排序、转换数据，并执行简单的汇总统计

# 安装Blaze

如果你运行Anaconda，安装Blaze很容易。只需在你的CLI（如果你不知道CLI是什么，请参阅奖励章节[第1章](ch01.html "第1章. 理解Spark")，*安装Spark*）中发出以下命令：

[PRE0]

一旦发出命令，你将看到类似于以下截图的屏幕：

![安装Blaze](img/B05793_09_01.jpg)

我们将稍后使用Blaze连接到PostgreSQL和MongoDB数据库，因此我们需要安装一些Blaze在后台使用的附加包。

我们将安装SQLAlchemy和PyMongo，它们都是Anaconda的一部分：

[PRE1]

现在剩下的只是在我们笔记本中导入Blaze本身：

[PRE2]

# 多语言持久化

Neal Ford于2006年引入了“多语言编程”这一概念，用以说明没有一种万能的解决方案，并提倡使用多种更适合特定问题的编程语言。

在数据并行的世界中，任何想要保持竞争力的企业都需要适应一系列技术，以便在尽可能短的时间内解决问题，从而最小化成本。

在Hadoop文件中存储事务数据是可能的，但意义不大。另一方面，使用**关系数据库管理系统**（**RDBMS**）处理PB级的互联网日志也是不明智的。这些工具是为了解决特定类型的任务而设计的；尽管它们可以被用来解决其他问题，但适应这些工具以解决这些问题的成本将是巨大的。这就像试图将方钉塞入圆孔一样。

例如，考虑一家在线销售乐器和配件的公司（以及一系列商店）。从高层次来看，公司需要解决许多问题才能成功：

1.  吸引客户到其商店（无论是虚拟的还是实体的）。

1.  向他们展示相关的产品（你不会试图向钢琴家卖鼓组，对吧？！）。

1.  一旦他们决定购买，处理付款并安排运输。

为了解决这些问题，公司可能会选择一系列旨在解决这些问题的技术：

1.  将所有产品存储在文档数据库中，如MongoDB、Cassandra、DynamoDB或DocumentDB。文档数据库具有多个优点：灵活的模式、分片（将更大的数据库分解为一系列更小、更易于管理的数据库）、高可用性和复制等。

1.  使用基于图数据库（如Neo4j、Tinkerpop/Gremlin或Spark的GraphFrames）来建模推荐：此类数据库反映了客户及其偏好的事实和抽象关系。挖掘这样的图非常有价值，可以为客户提供更个性化的服务。

1.  对于搜索，公司可能会使用定制的搜索解决方案，如Apache Solr或ElasticSearch。此类解决方案提供快速、索引的文本搜索功能。

1.  一旦产品售出，交易通常有一个结构良好的模式（例如产品名称、价格等）。为了存储此类数据（以及稍后对其进行处理和报告），关系数据库是最适合的。

使用多语言持久性，公司总是选择最适合的工具来完成工作，而不是试图将单一技术强加于解决所有问题。

如我们所见，Blaze将这些技术抽象化，并引入了一个简单的API来与之交互，因此您不需要学习每个想要使用的技术的API。本质上，它是一个多语言持久性的优秀工作示例。

### 备注

要了解其他人如何做，请查看[http://www.slideshare.net/Couchbase/couchbase-at-ebay-2014](http://www.slideshare.net/Couchbase/couchbase-at-ebay-2014)

或者

[http://www.slideshare.net/bijoor1/case-study-polyglotpersistence-in-pharmaceutical-industry](https://www.slideshare.net/bijoor1/case-study-polyglot-persistence-in-pharmaceutical-industry).

# 抽象化数据

Blaze 可以抽象许多不同的数据结构，并暴露一个单一、易于使用的 API。这有助于获得一致的行为并减少学习多个接口来处理数据的需要。如果你熟悉 pandas，实际上没有多少东西需要学习，因为语法上的差异是细微的。我们将通过一些示例来说明这一点。

## 使用 NumPy 数组

将数据从 NumPy 数组放入 Blaze 的 DataShape 对象中非常容易。首先，让我们创建一个简单的 NumPy 数组：我们首先加载 NumPy，然后创建一个两行三列的矩阵：

[PRE3]

现在我们有了数组，我们可以使用 Blaze 的 DataShape 结构来抽象它：

[PRE4]

就这样！简单得令人难以置信。

为了窥视结构，你可以使用 .`peek()` 方法：

[PRE5]

你应该看到以下截图所示的类似输出：

![使用 NumPy 数组](img/B05793_09_02.jpg)

你也可以使用（对于那些熟悉 pandas 语法的人来说很熟悉）的 .`head(...)` 方法。

### 注意

`.peek()` 和 `.head(...)` 之间的区别在于 `.head(...)` 允许指定行数作为其唯一参数，而 `.peek()` 不允许这样做，并且总是打印前 10 条记录。

如果你想要检索你的 DataShape 的第一列，你可以使用索引：

[PRE6]

你应该看到一个表格，就像这里所示：

![使用 NumPy 数组](img/B05793_09_03.jpg)

另一方面，如果你对检索一行感兴趣，你所要做的（就像在 NumPy 中一样）就是转置你的 DataShape：

[PRE7]

你将得到的结果如下所示：

![使用 NumPy 数组](img/B05793_09_04.jpg)

注意到列的名称是 `None`。DataShapes，就像 pandas 的 DataFrames 一样，支持命名列。因此，让我们指定我们字段的名称：

[PRE8]

现在，你可以通过调用列的名称来简单地检索数据：

[PRE9]

作为回报，你将得到以下输出：

![使用 NumPy 数组](img/B05793_09_05.jpg)

如你所见，定义字段会转置 NumPy 数组，现在，数组的每个元素都形成一个 *行*，这与我们最初创建的 `simpleData_np` 不同。

## 使用 pandas 的 DataFrame

由于 pandas 的 DataFrame 内部使用 NumPy 数据结构，将 DataFrame 转换为 DataShape 是轻而易举的。

首先，让我们创建一个简单的 DataFrame。我们首先导入 pandas：

[PRE10]

接下来，我们创建一个 DataFrame：

[PRE11]

然后，我们将它转换成一个 DataShape：

[PRE12]

你可以使用与从 NumPy 数组创建的 DataShape 相同的方式检索数据。使用以下命令：

[PRE13]

然后，它将产生以下输出：

![使用 pandas 的 DataFrame](img/B05793_09_06.jpg)

## 使用文件

DataShape 对象可以直接从 `.csv` 文件创建。在这个例子中，我们将使用一个包含 404,536 蒙哥马利县马里兰州发生的交通违规行为的数据集。

### 注意

我们于 2016 年 8 月 23 日从 [https://catalog.data.gov/dataset/traffic-violations-56dda](https://catalog.data.gov/dataset/traffic-violations-56dda) 下载了数据；数据集每日更新，因此如果你在稍后的日期检索数据集，交通违规的数量可能会有所不同。

我们将数据集存储在本地 `../Data` 文件夹中。然而，我们稍微修改了数据集，以便我们可以将其存储在 MongoDB 中：在其原始形式中，带有日期列，从 MongoDB 中读取数据会导致错误。我们向 Blaze 提交了一个错误报告 [https://github.com/blaze/blaze/issues/1580](https://github.com/blaze/blaze/issues/1580)：

[PRE14]

如果你不知道任何数据集的列名，你可以从 DataShape 中获取这些信息。要获取所有字段的列表，可以使用以下命令：

[PRE15]

![处理文件](img/B05793_09_07.jpg)

### 小贴士

对于熟悉 pandas 的你们来说，很容易识别 `.fields` 和 `.columns` 属性之间的相似性，因为它们基本上以相同的方式工作——它们都返回列的列表（在 pandas DataFrame 的情况下），或者称为 Blaze DataShape 中的字段列表。

Blaze 还可以直接从 `GZipped` 归档中读取，节省空间：

[PRE16]

为了验证我们得到的确切相同的数据，让我们从每个结构中检索前两个记录。你可以调用以下之一：

[PRE17]

或者，你也可以选择调用：

[PRE18]

它产生相同的结果（此处省略列名）：

![处理文件](img/B05793_09_08.jpg)

然而，很容易注意到从归档文件中检索数据需要显著更多的时间，因为 Blaze 需要解压缩数据。

你也可以一次从多个文件中读取并创建一个大数据集。为了说明这一点，我们将原始数据集按违规年份分割成四个 `GZipped` 数据集（这些存储在 `../Data/Years` 文件夹中）。

Blaze 使用 `odo` 来处理将 DataShape 保存到各种格式。要保存按年份划分的交通违规数据，你可以这样调用 `odo`：

[PRE19]

上述指令将数据保存到 `GZip` 归档中，但你也可以将其保存到前面提到的任何格式。`.odo(...)` 方法的第一个参数指定输入对象（在我们的例子中，是2013年发生的交通违规的 DataShape），第二个参数是输出对象——我们想要保存数据的文件路径。正如我们即将学习的——存储数据不仅限于文件。

要从多个文件中读取，可以使用星号字符 `*`：

[PRE20]

上述代码片段，再次，将生成一个熟悉的表格：

![处理文件](img/B05793_09_09.jpg)

Blaze 的读取能力不仅限于 `.csv` 或 `GZip` 文件：你可以从 JSON 或 Excel 文件（`.xls` 和 `.xlsx`）、HDFS 或 bcolz 格式的文件中读取数据。

### 小贴士

要了解更多关于 bcolz 格式的信息，请查看其文档 [https://github.com/Blosc/bcolz](https://github.com/Blosc/bcolz)。

## 处理数据库

Blaze 也可以轻松地从 SQL 数据库（如 PostgreSQL 或 SQLite）中读取。虽然 SQLite 通常是一个本地数据库，但 PostgreSQL 可以在本地或服务器上运行。

如前所述，Blaze 在后台使用 `odo` 来处理与数据库的通信。

### 注意

`odo` 是 Blaze 的一个要求，它将与包一起安装。在此处查看 [https://github.com/blaze/odo](https://github.com/blaze/odo)。

为了执行本节中的代码，您需要两样东西：一个运行中的本地 PostgreSQL 数据库实例，以及一个本地运行的 MongoDB 数据库。

### 小贴士

为了安装 PostgreSQL，从 [http://www.postgresql.org/download/](http://www.postgresql.org/download/) 下载包，并遵循那里找到的适用于您操作系统的安装说明。

要安装 MongoDB，请访问 [https://www.mongodb.org/downloads](https://www.mongodb.org/downloads) 并下载包；安装说明可以在 [http://docs.mongodb.org/manual/installation/](http://docs.mongodb.org/manual/installation/) 找到。

在您继续之前，我们假设您已经在 `http://localhost:5432/` 上运行了一个 PostgreSQL 数据库，并且 MongoDB 数据库在 `http://localhost:27017` 上运行。

我们已经将交通数据加载到两个数据库中，并将它们存储在 `traffic` 表（PostgreSQL）或 `traffic` 集合（MongoDB）中。

### 小贴士

如果您不知道如何上传数据，我在我的另一本书中解释了这一点 [https://www.packtpub.com/big-data-and-business-intelligence/practical-data-analysis-cookbook](https://www.packtpub.com/big-data-and-business-intelligence/practical-data-analysis-cookbook)。

### 与关系数据库交互

现在我们从 PostgreSQL 数据库中读取数据。访问 PostgreSQL 数据库的 **统一资源标识符**（**URI**）具有以下语法 `postgresql://<user_name>:<password>@<server>:<port>/<database>::<table>`。

要从 PostgreSQL 读取数据，只需将 URI 包裹在 `.Data(...)` 中 - Blaze 将处理其余部分：

[PRE21]

我们使用 Python 的 `.format(...)` 方法来填充字符串以包含适当的数据。

### 小贴士

在前面的示例中替换您的凭据以访问 PostgreSQL 数据库。如果您想了解更多关于 `.format(...)` 方法的知识，可以查看 Python 3.5 文档 [https://docs.python.org/3/library/string.html#format-string-syntax](https://docs.python.org/3/library/string.html#format-string-syntax)。

将数据输出到 PostgreSQL 或 SQLite 数据库相当简单。在下面的示例中，我们将输出涉及 2016 年生产的汽车的交通违规数据到 PostgreSQL 和 SQLite 数据库。如前所述，我们将使用 `odo` 来管理传输：

[PRE22]

与pandas类似，为了过滤数据，我们实际上选择了`Year`列（第一行的`traffic_psql['Year']`部分）并创建一个布尔标志，通过检查该列中的每个记录是否等于`2016`。通过将`traffic_psql`对象索引为这样的真值向量，我们提取了对应值等于`True`的记录。

如果您数据库中已经存在`traffic2016`表，则应取消注释以下两行；否则`odo`将数据追加到表末尾。

SQLite的URI与PostgreSQL略有不同；它的语法如下`sqlite://</relative/path/to/db.sqlite>::<table_name>`。

到现在为止，从SQLite数据库读取数据应该对您来说很简单：

[PRE23]

### 与MongoDB数据库交互

MongoDB多年来获得了大量的流行度。它是一个简单、快速且灵活的文档型数据库。对于使用`MEAN.js`堆栈的所有全栈开发者来说，该数据库是一个首选的存储解决方案：这里的M代表Mongo（见[http://meanjs.org](http://meanjs.org)）。

由于Blaze旨在以非常熟悉的方式工作，无论数据源如何，从MongoDB读取与从PostgreSQL或SQLite数据库读取非常相似：

[PRE24]

# 数据操作

我们已经介绍了一些您将使用DataShapes（例如，`.peek()`）的最常见方法，以及根据列值过滤数据的方式。Blaze实现了许多使处理任何数据变得极其容易的方法。

在本节中，我们将回顾许多其他常用的数据处理方式和与之相关的方法。对于来自`pandas`和/或SQL的您，我们将提供相应的语法，如果存在等效项。

## 访问列

有两种访问列的方式：您可以一次获取一个列，就像访问DataShape属性一样：

[PRE25]

上述脚本将产生以下输出：

![访问列](img/B05793_09_10.jpg)

您还可以使用索引，允许一次选择多个列：

[PRE26]

这将生成以下输出：

![访问列](img/B05793_09_11.jpg)

上述语法对于pandas DataFrames也是相同的。对于不熟悉Python和pandas API的您，请注意以下三点：

1.  要指定多个列，您需要将它们放在另一个列表中：注意双括号`[[`和`]]`。

1.  如果所有方法的链不适合一行（或者您想为了更好的可读性而断开链），您有两个选择：要么将整个方法链用括号`(...)`括起来，其中`...`是所有方法的链，或者，在换行前，在每个方法链的行末放置反斜杠字符`\`。我们更喜欢后者，并将在我们的示例中继续使用它。

1.  注意，等效的SQL代码将是：

    [PRE27]

## 符号变换

Blaze的美丽之处在于它可以**符号化**地操作。这意味着您可以在数据上指定变换、过滤器或其他操作，并将它们作为对象存储。然后，您可以用几乎任何符合原始模式的数据形式**提供**这样的对象，Blaze将返回变换后的数据。

例如，让我们选择所有发生在2013年的交通违规行为，并仅返回`'Arrest_Type'`、`'Color'`和`'Charge'`列。首先，如果我们不能从一个已存在的对象中反映模式，我们就必须手动指定模式。为此，我们将使用`.symbol(...)`方法来实现；该方法的第一参数指定了变换的符号名称（我们倾向于保持与对象名称相同，但可以是任何名称），第二个参数是一个长字符串，以`<column_name>: <column_type>`的形式指定模式，用逗号分隔：

[PRE28]

现在，您可以使用`schema_example`对象并指定一些变换。然而，由于我们已经有了一个现有的`traffic`数据集，我们可以通过使用`traffic.dshape`并指定我们的变换来**重用**该模式：

[PRE29]

为了展示这是如何工作的，让我们将原始数据集读入pandas的`DataFrame`：

[PRE30]

一旦读取，我们直接将数据集传递给`traffic_2013`对象，并使用Blaze的`.compute(...)`方法进行计算；该方法的第一参数指定了变换对象（我们的对象是`traffic_2013`），第二个参数是要对变换执行的数据：

[PRE31]

以下是前一个代码片段的输出：

![符号化变换](img/B05793_09_12.jpg)

您也可以传递一个列表的列表或一个NumPy数组的列表。在这里，我们使用DataFrame的`.values`属性来访问构成DataFrame的底层NumPy数组列表：

[PRE32]

此代码将产生我们预期的精确结果：

![符号化变换](img/B05793_09_13.jpg)

## 列操作

Blaze允许对数值列进行简单的数学运算。数据集中引用的所有交通违规行为都发生在2013年至2016年之间。您可以通过使用`.distinct()`方法获取`Stop_year`列的所有不同值来检查这一点。`.sort()`方法按升序排序结果：

[PRE33]

上述代码生成了以下输出表：

![列操作](img/B05793_09_14.jpg)

对于pandas，等效的语法如下：

[PRE34]

对于SQL，请使用以下代码：

[PRE35]

您也可以对列进行一些数学变换/算术运算。由于所有交通违规行为都发生在2000年之后，我们可以从`Stop_year`列中减去`2000`，而不会丢失任何精度：

[PRE36]

这是您应该得到的结果：

![列操作](img/B05793_09_15.jpg)

使用pandas `DataFrame`可以通过相同的语法达到相同的效果（假设`traffic`是pandas `DataFrame`类型）。对于SQL，等效的代码如下：

[PRE37]

然而，如果你想要进行一些更复杂的数学运算（例如，`log` 或 `pow`），那么你首先需要使用Blaze提供的（在后台，它将你的命令转换为NumPy、math或pandas的合适方法）。

例如，如果你想要对`Stop_year`进行对数转换，你需要使用以下代码：

[PRE38]

这将产生以下输出：

![列操作](img/B05793_09_16.jpg)

## 减少数据

一些减少方法也是可用的，例如`.mean()`（计算平均值）、`.std`（计算标准差）或`.max()`（从列表中返回最大值）。执行以下代码：

[PRE39]

它将返回以下输出：

![减少数据](img/B05793_09_17.jpg)

如果你有一个pandas DataFrame，你可以使用相同的语法，而对于SQL，可以使用以下代码完成相同操作：

[PRE40]

向你的数据集中添加更多列也非常简单。比如说，你想要计算在违规发生时汽车的年龄（以年为单位）。首先，你会从`Stop_year`中减去制造年份的`Year`。

在下面的代码片段中，`.transform(...)`方法的第一个参数是要执行转换的DataShape，其他参数会是转换列表。

[PRE41]

### 注意

在`.transform(...)`方法的源代码中，这样的列表会被表示为`*args`，因为你可以一次指定多个要创建的列。任何方法的`*args`参数可以接受任意数量的后续参数，并将其视为列表。

上述代码产生以下表格：

![减少数据](img/B05793_09_18.jpg)

在pandas中，可以通过以下代码实现等效操作：

[PRE42]

对于SQL，你可以使用以下代码：

[PRE43]

如果你想要计算涉及致命交通事故的汽车的平均年龄并计算发生次数，你可以使用`.by(...)`操作执行`group by`操作：

[PRE44]

`.by(...)`的第一个参数指定了DataShape中要执行聚合的列，后面跟着一系列我们想要得到的聚合。在这个例子中，我们选择了`Age_of_car`列，并计算了每个`'Fatal'`列值的平均值和行数。

前面的脚本生成了以下聚合结果：

![减少数据](img/B05793_09_19.jpg)

对于pandas，等效的代码如下：

[PRE45]

对于SQL，代码如下：

[PRE46]

## 连接

连接两个`DataShapes`同样简单。为了展示如何进行这一操作，尽管可以通过不同的方式得到相同的结果，我们首先通过违规类型（`violation`对象）和涉及安全带的违规（`belts`对象）选择所有交通违规：

[PRE47]

现在，我们将两个对象在六个日期和时间列上连接起来。

### 注意

如果我们只是简单地一次性选择了两个列：`Violation_type` 和 `Belts`，同样可以达到相同的效果。然而，这个例子是为了展示 `.join(...)` 方法的机制，所以请耐心等待。

`.join(...)` 方法的第一个参数是我们想要连接的第一个 DataShape，第二个参数是第二个 DataShape，而第三个参数可以是单个列或列的列表，用于执行连接操作：

[PRE48]

一旦我们有了完整的数据集，让我们检查有多少交通违规涉及安全带，以及司机受到了什么样的惩罚：

[PRE49]

这是前面脚本的输出：

![Joins](img/B05793_09_20.jpg)

使用以下代码在 pandas 中可以实现相同的效果：

[PRE50]

使用 SQL，您将使用以下片段：

[PRE51]

# 概述

本章中介绍的概念只是使用 Blaze 的道路起点。还有许多其他的使用方式和可以连接的数据源。将其视为构建你对多语言持久性的理解的基础。

然而，请注意，如今本章中解释的大多数概念都可以在 Spark 中原生获得，因为您可以直接在 Spark 中使用 SQLAlchemy，这使得与各种数据源一起工作变得容易。尽管需要投入学习 SQLAlchemy API 的初始成本，但这样做的好处是返回的数据将存储在 Spark DataFrame 中，您将能够访问 PySpark 提供的一切。这绝对不意味着您永远不应该使用 Blaze：选择，一如既往，是您的。

在下一章中，您将学习关于流式处理以及如何使用 Spark 进行流式处理。流式处理在当今已经成为一个越来越重要的主题，因为，每天（截至 2016 年为真），世界大约产生约 2.5 兆字节的数据（来源：[http://www.northeastern.edu/levelblog/2016/05/13/how-much-data-produced-every-day/](http://www.northeastern.edu/levelblog/2016/05/13/how-much-data-produced-every-day/))，这些数据需要被摄取、处理并赋予意义。
