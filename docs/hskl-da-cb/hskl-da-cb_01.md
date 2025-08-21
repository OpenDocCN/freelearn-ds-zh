# 第一章 数据的猎寻

本章将涵盖以下食谱：

+   利用来自不同来源的数据

+   从文件路径积累文本数据

+   捕捉 I/O 代码故障

+   保存和表示来自 CSV 文件的数据

+   使用 aeson 包检查 JSON 文件

+   使用 HXT 包读取 XML 文件

+   捕获 HTML 页面中的表格行

+   理解如何执行 HTTP GET 请求

+   学习如何执行 HTTP POST 请求

+   遍历在线目录以获取数据

+   在 Haskell 中使用 MongoDB 查询

+   从远程 MongoDB 服务器读取数据

+   探索来自 SQLite 数据库的数据

# 介绍

![介绍](img/ch01.jpg)

数据无处不在，日志记录便宜，分析是不可避免的。本章的一个最基本的概念就是收集有用的数据。在建立了一个大规模的可用文本集合后，我们称之为语料库，我们必须学会在代码中表示这些内容。主要关注将首先是获取数据，随后是列举表示数据的各种方式。

收集数据在某种程度上与分析数据一样重要，以便推断结果并形成有效的普遍性结论。这是一个科学的追求；因此，必须且将会非常小心地确保采样无偏且具有代表性。我们建议在本章中密切跟随，因为本书的其余部分都依赖于有数据源可供操作。如果没有数据，就几乎没有什么可以分析的，所以我们应该仔细观察在本章中提出的技术，以便构建我们自己的强大语料库。

第一条食谱列举了多种在线收集数据的来源。接下来的几个食谱涉及使用不同文件格式的本地数据。然后我们学习如何使用 Haskell 代码从互联网上下载数据。最后，我们以几个使用 Haskell 数据库的食谱结束本章。

# 利用来自不同来源的数据

信息可以被描述为结构化、非结构化，或有时是两者的混合——半结构化。

从广义上讲，结构化数据是指任何可以被算法解析的数据。常见的例子包括 JSON、CSV 和 XML。如果提供了结构化数据，我们可以设计一段代码来分析其底层格式，并轻松地生成有用的结果。由于挖掘结构化数据是一个确定性的过程，这使得我们可以自动化解析，从而让我们收集更多的输入来喂养我们的数据分析算法。

非结构化数据是指其他所有的数据。它是没有按照特定方式定义的数据。像英语这样的书面语言通常被视为非结构化数据，因为从自然句子中解析出数据模型非常困难。

在寻找好数据的过程中，我们常常会发现结构化和非结构化文本的混合。这就是所谓的半结构化文本。

本食谱将主要关注从以下来源获取结构化和半结构化数据。

### 提示

与本书中的大多数食谱不同，这个食谱不包含任何代码。阅读本书的最佳方式是跳到那些你感兴趣的食谱。

## 如何实现…

我们将通过以下章节提供的链接浏览，以建立一个源列表，利用可用格式的有趣数据。然而，这个列表并不详尽。

其中一些数据源提供**应用程序编程接口**（**API**），允许更复杂地访问有趣的数据。API 指定了交互方式并定义了数据如何传输。

### 新闻

《纽约时报》拥有最精炼的 API 文档之一，能够访问从房地产数据到文章搜索结果的各种内容。该文档可以在[`developer.nytimes.com`](http://developer.nytimes.com)找到。

《卫报》还提供了一个包含超过一百万篇文章的大型数据存储库，网址为[`www.theguardian.com/data`](http://www.theguardian.com/data)。

《今日美国》提供有关书籍、电影和音乐评论的一些有趣资源。技术文档可以在[`developer.usatoday.com`](http://developer.usatoday.com)找到。

BBC 提供一些有趣的 API 端点，包括 BBC 节目和音乐信息，网址为[`www.bbc.co.uk/developer/technology/apis.html`](http://www.bbc.co.uk/developer/technology/apis.html)。

### 私人

Facebook、Twitter、Instagram、Foursquare、Tumblr、SoundCloud、Meetup 等许多社交网络网站支持 API 来访问一定程度的社交信息。

对于特定的 API，如天气或体育，Mashape 是一个集中式搜索引擎，可以缩小搜索范围到一些较不为人知的来源。Mashape 的网址是[`www.mashape.com/`](https://www.mashape.com/)

大多数数据源可以通过位于[`www.google.com/publicdata`](http://www.google.com/publicdata)的 Google 公共数据搜索进行可视化。

要查看包含各种数据格式的所有国家列表，请参考位于[`github.com/umpirsky/country-list`](https://github.com/umpirsky/country-list)的代码库。

### 学术

一些数据源由世界各地的大学公开托管，用于研究目的。

为了分析医疗数据，华盛顿大学已发布健康指标与评估研究所（IHME），以收集世界上最重要的健康问题的严格且可比较的测量数据。更多信息请访问[`www.healthdata.org`](http://www.healthdata.org)。

来自纽约大学、谷歌实验室和微软研究院的 MNIST 手写数字数据库，是一个用于手写数字的标准化和居中样本的训练集。可以从[`yann.lecun.com/exdb/mnist`](http://yann.lecun.com/exdb/mnist)下载数据。

### 非营利组织

《人类发展报告》每年更新，涵盖从成人识字率到拥有个人电脑人数的国际数据。它自称拥有多种国际公共资源，并代表了这些指标的最新统计数据。更多信息请访问[`hdr.undp.org/en/statistics`](http://hdr.undp.org/en/statistics)。

世界银行是贫困和全球发展数据的来源。它自认为是一个自由来源，旨在提供全球各国发展的开放数据访问。更多信息请访问[`data.worldbank.org/`](http://data.worldbank.org/)。

世界卫生组织提供全球健康状况监测的数据和分析。更多信息请访问[`www.who.int/research/en`](http://www.who.int/research/en)。

联合国儿童基金会（UNICEF）还发布了有趣的统计数据，正如其网站上的引用所示：

> *“联合国儿童基金会数据库包含儿童死亡率、疾病、水卫生等方面的统计表。联合国儿童基金会声称在监测儿童和妇女状况方面发挥着核心作用——帮助各国收集和分析数据，协助他们制定方法论和指标，维护全球数据库，传播和发布数据。可以在[`www.unicef.org/statistics`](http://www.unicef.org/statistics)找到相关资源。”*

联合国在[`www.un.org/en/databases`](http://www.un.org/en/databases)发布有趣的公开政治统计数据。

### 美国政府

如果我们像尼古拉斯·凯奇在电影《国家宝藏》（2004 年）中所做的那样，渴望发现美国政府中的模式，那么[`www.data.gov/`](http://www.data.gov/)将是我们的首选来源。它是美国政府积极提供有用数据的努力，旨在“增加公众对联邦政府执行部门生成的高价值、机器可读数据集的访问。”更多信息请访问[`www.data.gov`](http://www.data.gov)。

美国人口普查局发布人口统计、住房统计、区域测量等数据。这些数据可以在[`www.census.gov`](http://www.census.gov)找到。

# 从文件路径累积文本数据

开始处理输入的最简单方法之一是从本地文件读取原始文本。在这个例子中，我们将从特定的文件路径提取所有文本。此外，为了对数据做些有趣的事情，我们将统计每行的单词数。

### 提示

Haskell 是一种纯粹的函数式编程语言，对吗？没错，但从代码外部获取输入会引入不纯净性。为了优雅性和可重用性，我们必须仔细区分纯净代码和不纯净代码。

## 准备开始

我们首先创建一个`input.txt`文本文件，文件中有几行文本供程序读取。我们将此文件保存在一个容易访问的目录中，因为稍后会用到。比如，我们正在处理的文本文件包含了一段柏拉图的七行引用。以下是我们执行以下命令时终端的输出：

```py
$ cat input.txt

And how will you inquire, Socrates,
into that which you know not? 
What will you put forth as the subject of inquiry? 
And if you find what you want, 
how will you ever know that 
this is what you did not know?

```

### 小贴士

**下载示例代码**

你可以从你的账户中下载所有购买的 Packt 书籍的示例代码文件，网址是[`www.packtpub.com`](http://www.packtpub.com)。如果你是在其他地方购买了本书，你可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册以便直接将文件通过电子邮件发送给你。代码也将托管在 GitHub 上，网址是[`github.com/BinRoot/Haskell-Data-Analysis-Cookbook`](https://github.com/BinRoot/Haskell-Data-Analysis-Cookbook)。

## 如何操作...

创建一个新文件开始编写代码。我们将文件命名为 Main.hs。

1.  与所有可执行的 Haskell 程序一样，首先定义并实现`main`函数，如下所示：

    ```py
    main :: IO ()
    main = do

    ```

1.  使用 Haskell 的`readFile :: FilePath -> IO String`函数来从`input.txt`文件路径中提取数据。请注意，文件路径实际上只是`String`的同义词。将字符串加载到内存后，将其传递给`countWords`函数，以便计算每行的单词数，如下所示：

    ```py
    input <- readFile "input.txt"
    print $ countWords input

    ```

1.  最后，定义我们的纯函数`countWords`，如下所示：

    ```py
    countWords :: String -> [Int]
    countWords input = map (length.words) (lines input)

    ```

1.  程序将打印出每行的单词数，并以数字列表的形式呈现，具体如下：

    ```py
    $ runhaskell Main.hs

    [6,6,10,7,6,7]

    ```

## 它是如何工作的...

Haskell 提供了有用的输入和输出（I/O）功能，可以以不同方式读取输入和写入输出。在我们的例子中，我们使用`readFile`来指定要读取的文件路径。使用`main`中的`do`关键字意味着我们将多个 I/O 操作连接在一起。`readFile`的输出是一个 I/O 字符串，这意味着它是一个返回`String`类型的 I/O 操作。

现在我们要进入一些技术细节，请注意。或者，你可以微笑并点头表示理解。在 Haskell 中，I/O 数据类型是名为 Monad 的实例。这允许我们使用`<-`符号从这个 I/O 操作中提取字符串。然后，我们通过将字符串传递给`countWords`函数来使用它，从而计算每行的单词数。请注意，我们将`countWords`函数与不纯粹的`main`函数分开。

最后，我们打印出`countWords`的输出。`$`符号表示我们使用函数应用来避免在代码中使用过多的括号。如果没有它，`main`的最后一行将是`print (countWords input)`。

## 另见

为了简便起见，这段代码易于阅读，但非常脆弱。如果`input.txt`文件不存在，运行代码将立即使程序崩溃。例如，以下命令将生成错误信息：

```py
$ runhaskell Main.hs

Main.hs: input.txt: openFile: does not exist…

```

为了使这段代码具有容错性，请参考 *捕获 I/O 代码错误* 的做法。

# 捕获 I/O 代码错误

确保我们的代码在数据挖掘或分析过程中不会崩溃是一个非常重要的考虑因素。某些计算可能需要几个小时，甚至几天。Haskell 提供了类型安全和强类型检查，以帮助确保程序不会失败，但我们也必须小心，仔细检查可能发生故障的边缘情况。

例如，如果没有找到本地文件路径，程序可能会异常崩溃。在前面的例子中，我们的代码强烈依赖于 `input.txt` 的存在。如果程序无法找到该文件，它将产生以下错误：

```py
mycode: input.txt: openFile: does not exist (No such file or directory)

```

自然地，我们应该通过允许用户指定文件路径以及在文件未找到时不让程序崩溃，从而解耦文件路径的依赖关系。

考虑对源代码进行以下修改。

## 如何做到……

创建一个新文件，命名为 `Main.hs`，并执行以下步骤：

1.  首先，导入一个库来捕获致命错误，如下所示：

    ```py
    import Control.Exception (catch, SomeException)
    ```

1.  接下来，导入一个库来获取命令行参数，使文件路径动态化。我们使用以下代码行来实现：

    ```py
    import System.Environment (getArgs)
    ```

1.  按照之前的方式，定义并实现 `main` 如下：

    ```py
    main :: IO ()
    main = do
    ```

1.  根据用户提供的参数定义一个 `fileName` 字符串，如果没有参数则默认为 `input.txt`。该参数通过从库函数 `getArgs :: IO [String]` 中获取字符串数组来获取，如以下步骤所示：

    ```py
    args <- getArgs
      let filename = case args of
        (a:_) -> a
            _ -> "input.txt"
    ```

1.  现在在这个路径上应用 `readFile`，但使用库的 `catch :: Exception e => IO a -> (e -> IO a) -> IO a` 函数来捕获任何错误。`catch` 的第一个参数是要运行的计算，第二个参数是如果出现异常时要调用的处理程序，如以下命令所示：

    ```py
      input <- catch (readFile fileName)
        $ \err -> print (err::SomeException) >> return ""
    ```

1.  如果读取文件时出现错误，`input` 字符串将为空。我们现在可以使用 `input` 来执行任何操作，如下所示：

    ```py
      print $ countWords input
    ```

1.  别忘了定义 `countWords` 函数，如下所示：

    ```py
    countWords input = map (length.words) (lines input)
    ```

## 它是如何工作的……

这个例子展示了两种捕获错误的方法，如下所示：

+   首先，我们使用一个模式匹配的 `case` 表达式来匹配传入的任何参数。因此，如果没有传入参数，`args` 列表为空，最后的模式 `"_"` 会被捕获，从而得到默认的文件名 `input.txt`。

+   其次，我们使用 `catch` 函数来处理错误，如果出现问题。在读取文件时遇到麻烦时，我们通过将 `input` 设置为空字符串来允许代码继续运行。

## 还有更多……

方便的是，Haskell 还提供了一个来自 `System.Directory` 模块的 `doesFileExist :: FilePath -> IO Bool` 函数。我们可以通过修改 `input <- …` 这一行来简化之前的代码。它可以被以下代码片段替换：

```py
exists <- doesFileExist filename
input <- if exists then readFile filename else return ""
```

在这种情况下，代码只有在文件存在时才会将其作为输入读取。不要忘记在源代码的顶部添加以下 `import` 语句：

```py
import System.Directory (doesFileExist)
```

# 保留和表示来自 CSV 文件的数据

**逗号分隔值**（**CSV**）是一种以纯文本表示数值表格的格式。它通常用于与电子表格中的数据进行交互。CSV 的规格在 RFC 4180 中有描述，可以在[`tools.ietf.org/html/rfc4180`](http://tools.ietf.org/html/rfc4180)找到。

在这个例子中，我们将读取一个名为`input.csv`的本地 CSV 文件，里面包含各种姓名及其对应的年龄。然后，为了对数据做一些有意义的操作，我们将找到最年长的人。

## 准备工作

准备一个简单的 CSV 文件，列出姓名及其对应的年龄。可以使用文本编辑器完成此操作，或通过电子表格导出，如下图所示：

![准备工作](img/6331OS_01_01.jpg)

原始的`input.csv`文件包含以下文本：

```py
$ cat input.csv 

name,age
Alex,22
Anish,22
Becca,23
Jasdev,22
John,21
Jonathon,21
Kelvin,22
Marisa,19
Shiv,22
Vinay,22

```

该代码还依赖于`csv`库。我们可以使用以下命令通过 Cabal 安装该库：

```py
$ cabal install csv

```

## 如何操作...

1.  使用以下代码行导入`csv`库：

    ```py
    import Text.CSV
    ```

1.  定义并实现`main`，在这里我们将读取并解析 CSV 文件，如以下代码所示：

    ```py
    main :: IO ()
    main = do
      let fileName = "input.csv"
      input <- readFile fileName
    ```

1.  将`parseCSV`应用于文件名，以获得一系列行，表示表格数据。`parseCSV`的输出是`Either ParseError CSV`，因此确保我们考虑`Left`和`Right`两种情况：

    ```py
      let csv = parseCSV fileName input
      either handleError doWork csv
    handleError csv = putStrLn "error parsing"
    doWork csv = (print.findOldest.tail) csv
    ```

1.  现在我们可以处理 CSV 数据了。在这个例子中，我们找到并打印包含最年长的人的行，如下面的代码片段所示：

    ```py
    findOldest :: [Record] -> Record
    findOldest [] = []
    findOldest xs = foldl1
              (\a x -> if age x > age a then x else a) xs

    age [a,b] = toInt a

    toInt :: String -> Int                               
    toInt = read
    ```

1.  运行`main`后，代码应该产生以下输出：

    ```py
    $ runhaskell Main.hs

    ["Becca", "23"]

    ```

    ### 提示

    我们也可以使用`parseCSVFromFile`函数直接从文件名获取 CSV 表示，而不是使用`readFile`后接`parseCSV`。

## 如何操作...

在 Haskell 中，CSV 数据结构表示为一个记录列表。`Record`仅仅是`Fields`的列表，`Field`是`String`的类型别名。换句话说，它是表示表格的行的集合，如下图所示：

![如何操作...](img/6331OS_01_02.jpg)

`parseCSV`库函数返回一个`Either`类型，`Left`侧是一个`ParseError`，`Right`侧是一个列表的列表。`Either l r`数据类型与`Maybe a`类型非常相似，后者有`Just a`或`Nothing`构造器。

我们使用`either`函数来处理`Left`和`Right`的情况。`Left`情况处理错误，`Right`情况处理数据上的实际操作。在这个例子中，`Right`侧是一个`Record`。`Record`中的字段可以通过任何列表操作进行访问，例如`head`、`last`、`!!`等。

# 使用 aeson 包检查 JSON 文件

**JavaScript 对象表示法**（**JSON**）是一种以纯文本表示键值对的方式。该格式在 RFC 4627 中有广泛描述（[`www.ietf.org/rfc/rfc4627`](http://www.ietf.org/rfc/rfc4627)）。

在这个例子中，我们将解析一个关于某人的 JSON 描述。我们常在来自 Web 应用程序的 API 中遇到 JSON 格式。

## 准备工作

使用 Cabal 从 hackage 安装`aeson`库。

准备一个代表数学家的`input.json`文件，如下代码片段所示：

```py
$ cat input.json

{"name":"Gauss", "nationality":"German", "born":1777, "died":1855}

```

我们将解析这个 JSON 并将其表示为 Haskell 中的可用数据类型。

## 如何操作...

1.  使用`OverloadedStrings`语言扩展将字符串表示为`ByteString`，如下代码行所示：

    ```py
    {-# LANGUAGE OverloadedStrings #-}
    ```

1.  如下所示导入`aeson`及一些辅助函数：

    ```py
    import Data.Aeson
    import Control.Applicative
    import qualified Data.ByteString.Lazy as B
    ```

1.  创建与 JSON 结构对应的数据类型，如下代码所示：

    ```py
    data Mathematician = Mathematician 
                         { name :: String
                         , nationality :: String
                         , born :: Int
                         , died :: Maybe Int
                         } 
    ```

1.  如下代码片段所示，为`parseJSON`函数提供一个实例：

    ```py
    instance FromJSON Mathematician where
      parseJSON (Object v) = Mathematician
                             <$> (v .: "name")
                             <*> (v .: "nationality")
                             <*> (v .: "born")
                             <*> (v .:? "died")
    ```

1.  如下所示定义并实现`main`：

    ```py
    main :: IO ()
    main = do
    ```

1.  阅读输入并解码 JSON，如下代码片段所示：

    ```py
      input <- B.readFile "input.json"

      let mm = decode input :: Maybe Mathematician

      case mm of
        Nothing -> print "error parsing JSON"
        Just m -> (putStrLn.greet) m
    ```

1.  现在我们将对数据做一些有趣的操作，如下所示：

    ```py
    greet m = (show.name) m ++ 
              " was born in the year " ++ 
              (show.born) m
    ```

1.  我们可以运行代码以查看以下输出：

    ```py
    $ runhaskell Main.hs

    "Gauss" was born in the year 1777

    ```

## 它是如何工作的...

Aeson 处理表示 JSON 的复杂性。它将结构化文本转换为本地可用的数据。在本食谱中，我们使用`Data.Aeson`模块提供的`.:`和`.:?`函数。

由于`Aeson`包使用`ByteStrings`而非`Strings`，因此很有帮助的是告诉编译器引号中的字符应该被当作正确的数据类型处理。这是在代码的第一行通过调用`OverloadedStrings`语言扩展来实现的。

### 提示

如今，`OverloadedStrings`等语言扩展目前仅**Glasgow Haskell Compiler**（**GHC**）支持。

我们使用 Aeson 提供的`decode`函数将字符串转换为数据类型。它的类型为`FromJSON a => B.ByteString -> Maybe a`。我们的`Mathematician`数据类型必须实现`FromJSON`类型类的实例才能正确使用此函数。幸运的是，实现`FromJSON`所需的唯一函数是`parseJSON`。本食谱中用于实现`parseJSON`的语法有些奇怪，但这是因为我们正在利用应用函数和镜头，这是更高级的 Haskell 主题。

`.:`函数有两个参数，`Object`和`Text`，并返回一个`Parser a`数据类型。根据文档，它用于检索与给定键相关联的对象中的值。如果 JSON 文档中存在该键和值，则使用此函数。`:?`函数也从给定键的对象中检索关联值，但键和值的存在不是必需的。因此，对于 JSON 文档中的可选键值对，我们使用`.:?`。

## 还有更多...

如果`FromJSON`类型类的实现过于复杂，我们可以轻松地让 GHC 通过`DeriveGeneric`语言扩展自动填充它。以下是代码的简化重写：

```py
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
import Data.Aeson
import qualified Data.ByteString.Lazy as B
import GHC.Generics

data Mathematician = Mathematician { name :: String
                                   , nationality :: String
                                   , born :: Int
                                   , died :: Maybe Int
                                   } deriving Generic

instance FromJSON Mathematician

main = do
  input <- B.readFile "input.json"
  let mm = decode input :: Maybe Mathematician
  case mm of
    Nothing -> print "error parsing JSON"
    Just m -> (putStrLn.greet) m

greet m = (show.name) m ++" was born in the year "++ (show.born) m
```

尽管 Aeson 功能强大且具有通用性，但对于一些简单的 JSON 交互，它可能显得过于复杂。作为替代，如果我们希望使用一个非常简洁的 JSON 解析器和打印器，可以使用 Yocto，它可以从[`hackage.haskell.org/package/yocto`](http://hackage.haskell.org/package/yocto)下载。

# 使用 HXT 包读取 XML 文件

**可扩展标记语言**（**XML**）是对纯文本的编码，旨在为文档提供机器可读的注释。该标准由 W3C 指定（[`www.w3.org/TR/2008/REC-xml-20081126/`](http://www.w3.org/TR/2008/REC-xml-20081126/)）。

在这个示例中，我们将解析一个表示电子邮件对话的 XML 文档，并提取所有日期。

## 准备就绪

我们首先设置一个名为 `input.xml` 的 XML 文件，包含以下值，表示 2014 年 12 月 18 日 Databender 和 Princess 之间的电子邮件对话，如下所示：

```py
$ cat input.xml

<thread>
    <email>
        <to>Databender</to>
        <from>Princess</from>
        <date>Thu Dec 18 15:03:23 EST 2014</date>
        <subject>Joke</subject>
        <body>Why did you divide sin by tan?</body>
    </email>
    <email>
        <to>Princess</to>
        <from>Databender</from>
        <date>Fri Dec 19 3:12:00 EST 2014</date>
        <subject>RE: Joke</subject>
        <body>Just cos.</body>
    </email>
</thread>
```

使用 Cabal 安装 HXT 库，我们用它来处理 XML 文档：

```py
$ cabal install hxt

```

## 它是如何做的...

1.  我们只需要一个导入，用于解析 XML，代码如下：

    ```py
    import Text.XML.HXT.Core
    ```

1.  定义并实现 `main` 函数并指定 XML 的位置。对于此示例，文件从 `input.xml` 获取。参考以下代码：

    ```py
    main :: IO ()
    main = do
        input <- readFile "input.xml"
    ```

1.  将 `readString` 函数应用于输入并提取所有日期文档。我们使用 `hasName :: String -> a XmlTree XmlTree` 函数筛选具有特定名称的项目。同时，我们使用 `getText :: a XmlTree String` 函数提取文本，如下所示的代码片段：

    ```py
        dates <- runX $ readString [withValidate no] input 
            //> hasName "date" 
            //> getText
    ```

1.  我们现在可以按如下方式使用提取的日期列表：

    ```py
        print dates
    ```

1.  运行代码后，我们打印出以下输出：

    ```py
     $ runhaskell Main.hs

    ["Thu Dec 18 15:03:23 EST 2014", "Fri Dec 19 3:12:00 EST 2014"]

    ```

## 它是如何工作的...

库函数 `runX` 接受一个 **Arrow**。可以将 Arrow 看作是比 Monad 更强大的版本。Arrows 允许进行有状态的全局 XML 处理。具体来说，本示例中的 `runX` 函数接受 `IOSArrow XmlTree String`，并返回一个类型为 `String` 的 `IO` 动作。我们使用 `readString` 函数生成此 `IOSArrow` 对象，它对 XML 数据执行一系列操作。

对于深入了解 XML 文档，应该使用 `//>`，而 `/>` 只查看当前级别。我们使用 `//>` 函数查找日期属性并显示所有关联文本。

如文档中所定义，`hasName` 函数用于测试一个节点是否具有特定名称，`getText` 函数用于选择文本节点的文本。还有其他一些函数，包括：

+   `isText`：用于测试文本节点

+   `isAttr`：用于测试属性树

+   `hasAttr`：用于测试一个元素节点是否具有特定名称的属性节点

+   `getElemName`：用于选择元素节点的名称

所有的箭头函数都可以在 `Text.XML.HXT.Arrow.XmlArrow` 文档中找到，链接：[`hackage.haskell.org/package/hxt/docs/Text-XML-HXT-Arrow-XmlArrow.html`](http://hackage.haskell.org/package/hxt/docs/Text-XML-HXT-Arrow-XmlArrow.html)。

# 从 HTML 页面捕获表格行

挖掘**超文本标记语言**（**HTML**）通常是一项识别和解析其结构化部分的工作。并非 HTML 文件中的所有文本都是有用的，所以我们往往只关注特定的子集。例如，HTML 表格和列表提供了一种强大且常用的结构来提取数据，而文章中的段落可能过于无结构和复杂，不易处理。

在本配方中，我们将找到网页上的一个表格，并收集所有行以供程序使用。

## 准备工作

我们将从 HTML 表格中提取值，所以首先创建一个包含表格的`input.html`文件，如下图所示：

![准备工作](img/6331OS_01_03.jpg)

该表格背后的 HTML 如下所示：

```py
$ cat input.html

<!DOCTYPE html>
<html>
    <body>
        <h1>Course Listing</h1>
        <table>
            <tr>
                <th>Course</th>
                <th>Time</th>
                <th>Capacity</th>
            </tr>
            <tr>
                <td>CS 1501</td>
                <td>17:00</td>
                <td>60</td>
            </tr>
            <tr>
                <td>MATH 7600</td>
                <td>14:00</td>
                <td>25</td>
            </tr>
            <tr>
                <td>PHIL 1000</td>
                <td>9:30</td>
                <td>120</td>
            </tr>
        </table>
    </body>
</html>
```

如果尚未安装，请使用 Cabal 来设置 HXT 库和 split 库，如下命令所示：

```py
$ cabal install hxt
$ cabal install split

```

## 如何实现...

1.  我们将需要`htx`包用于 XML 操作，以及来自 split 包的`chunksOf`函数，如以下代码片段所示：

    ```py
    import Text.XML.HXT.Core
    import Data.List.Split (chunksOf)
    ```

1.  定义并实现`main`来读取`input.html`文件。

    ```py
    main :: IO ()
    main = do
      input <- readFile "input.html"
    ```

1.  将 HTML 数据传递给`readString`，设置`withParseHTML`为`yes`，并可选择关闭警告。提取所有`td`标签并获取剩余文本，如以下代码所示：

    ```py
      texts <- runX $ readString 
               [withParseHTML yes, withWarnings no] input 
        //> hasName "td"
        //> getText
    ```

1.  现在数据可以作为字符串列表使用。它可以像之前 CSV 配方中展示的那样，转换为列表的列表，如以下代码所示：

    ```py
      let rows = chunksOf 3 texts
      print $ findBiggest rows
    ```

1.  通过折叠数据，使用以下代码片段识别容量最大课程：

    ```py
    findBiggest :: [[String]] -> [String]
    findBiggest [] = []
    findBiggest items = foldl1 
                        (\a x -> if capacity x > capacity a 
                                 then x else a) items

    capacity [a,b,c] = toInt c
    capacity _ = -1

    toInt :: String -> Int
    toInt = read
    ```

1.  运行代码将显示容量最大的课程，如下所示：

    ```py
    $ runhaskell Main.hs

    {"PHIL 1000", "9:30", "120"}

    ```

## 它是如何工作的...

这与 XML 解析非常相似，只是我们调整了`readString`的选项为`[withParseHTML yes, withWarnings no]`。

# 了解如何执行 HTTP GET 请求

寻找好数据的最有资源的地方之一就是在线。**GET 请求**是与 HTTP 网页服务器通信的常用方法。在这个配方中，我们将抓取维基百科文章中的所有链接并将其打印到终端。为了方便地抓取所有链接，我们将使用一个叫做`HandsomeSoup`的有用库，它可以让我们通过 CSS 选择器轻松地操作和遍历网页。

## 准备工作

我们将从一个维基百科网页中收集所有链接。在运行此配方之前，请确保已连接互联网。

安装`HandsomeSoup` CSS 选择器包，如果尚未安装 HXT 库，请安装它。为此，请使用以下命令：

```py
$ cabal install HandsomeSoup
$ cabal install hxt

```

## 如何实现...

1.  此配方需要`hxt`来解析 HTML，并需要`HandsomeSoup`来提供易于使用的 CSS 选择器，如以下代码片段所示：

    ```py
    import Text.XML.HXT.Core
    import Text.HandsomeSoup
    ```

1.  按如下方式定义并实现`main`：

    ```py
    main :: IO ()
    main = do
    ```

1.  将 URL 作为字符串传递给 HandsomeSoup 的`fromUrl`函数：

    ```py
        let doc = fromUrl "http://en.wikipedia.org/wiki/Narwhal"
    ```

1.  按如下方式选择维基百科页面中`bodyContent`字段内的所有链接：

    ```py
        links <- runX $ doc >>> css "#bodyContent a" ! "href"
        print links
    ```

## 它是如何工作的…

`HandsomeSoup` 包允许使用简易的 CSS 选择器。在此配方中，我们在 Wikipedia 文章网页上运行 `#bodyContent a` 选择器。这将找到所有作为`bodyContent` ID 元素后代的链接标签。

## 另见…

另一种常见的在线获取数据的方法是通过 POST 请求。要了解更多信息，请参考*学习如何执行 HTTP POST 请求*的配方。

# 学习如何执行 HTTP POST 请求

**POST 请求**是另一种非常常见的 HTTP 服务器请求，许多 API 都在使用它。我们将挖掘弗吉尼亚大学的目录搜索。当发送一个用于搜索查询的 POST 请求时，**轻量级目录访问协议**（**LDAP**）服务器会返回一个包含搜索结果的网页。

## 准备就绪

此配方需要访问互联网。

安装`HandsomeSoup` CSS 选择器包，如果尚未安装，还需要安装 HXT 库：

```py
$ cabal install HandsomeSoup
$ cabal install hxt

```

## 如何操作...

1.  导入以下库：

    ```py
    import Network.HTTP
    import Network.URI (parseURI)
    import Text.XML.HXT.Core
    import Text.HandsomeSoup
    import Data.Maybe (fromJust)
    ```

1.  定义目录搜索网站指定的 POST 请求。根据服务器的不同，以下 POST 请求的细节会有所不同。请参考以下代码片段：

    ```py
    myRequestURL = "http://www.virginia.edu/cgi-local/ldapweb"

    myRequest :: String -> Request_String
    myRequest query = Request { 
        rqURI = fromJust $ parseURI myRequestURL
      , rqMethod = POST
      , rqHeaders = [ mkHeader HdrContentType "text/html"
                    , mkHeader HdrContentLength $ show $ length body ]
      , rqBody = body
      }
      where body = "whitepages=" ++ query
    ```

1.  定义并实现 `main` 来运行如下的 POST 请求：

    ```py
    main :: IO ()
    main = do
      response <- simpleHTTP $ myRequest "poon"
    ```

1.  收集 HTML 并进行解析：

    ```py
      html <- getResponseBody response
      let doc = readString [withParseHTML yes, withWarnings no] html
    ```

1.  查找表格行并使用以下代码打印输出：

    ```py
      rows <- runX $ doc >>> css "td" //> getText
      print rows
    ```

运行代码将显示与`"poon"`相关的所有搜索结果，如“Poonam”或“Witherspoon”。

## 它是如何工作的...

POST 请求需要指定的 URI、头信息和主体。通过填写`Request`数据类型，可以用来建立服务器请求。

## 另见

请参考*理解如何执行 HTTP GET 请求*的配方，了解如何执行 GET 请求的详细信息。

# 遍历在线目录以获取数据

目录搜索通常会根据查询提供姓名和联系方式。通过强行进行多个搜索查询，我们可以获取目录列表数据库中存储的所有数据。此配方仅作为学习工具，用于展示 Haskell 数据收集的强大和简便性。

## 准备就绪

确保拥有强劲的互联网连接。

使用 Cabal 安装`hxt`和`HandsomeSoup`包：

```py
$ cabal install hxt
$ cabal install HandsomeSoup

```

## 如何操作...

1.  设置以下依赖项：

    ```py
    import Network.HTTP
    import Network.URI
    import Text.XML.HXT.Core
    import Text.HandsomeSoup
    ```

1.  定义一个 `SearchResult` 类型，它可能会失败并返回错误，或返回成功，如以下代码所示：

    ```py
    type SearchResult = Either SearchResultErr [String]
    data SearchResultErr = NoResultsErr 
                         | TooManyResultsErr 
                         | UnknownErr     
                         deriving (Show, Eq)
    ```

1.  定义目录搜索网站指定的 POST 请求。根据服务器的不同，POST 请求会有所不同。为了避免重写代码，我们使用在上一个配方中定义的 `myRequest` 函数。

1.  编写一个辅助函数来获取 HTTP POST 请求的文档，如下所示：

    ```py
    getDoc query = do  
        rsp <- simpleHTTP $ myRequest query
        html <- getResponseBody rsp
        return $ readString [withParseHTML yes, withWarnings no] html
    ```

1.  扫描 HTML 文档并返回是否有错误，或者提供结果数据。此函数中的代码依赖于网页生成的错误消息。在我们的案例中，错误消息如下：

    ```py
    scanDoc doc = do
        errMsg <- runX $ doc >>> css "h3" //> getText

        case errMsg of 
            [] -> do 
                text <- runX $ doc >>> css "td" //> getText 
                return $ Right text
            "Error: Sizelimit exceeded":_ -> 
                return $ Left TooManyResultsErr
            "Too many matching entries were found":_ -> 
                return $ Left TooManyResultsErr
            "No matching entries were found":_ -> 
                return $ Left NoResultsErr
            _ -> return $ Left UnknownErr
    ```

1.  定义并实现 `main`。我们将使用一个辅助函数 `main'`，如下所示的代码片段中，将递归地强行列出目录：

    ```py
    main :: IO ()
    main = main' "a"
    ```

1.  执行查询搜索，然后在下一个查询中递归执行：

    ```py
    main' query = do
        print query
        doc <- getDoc query
        searchResult <- scanDoc doc
        print searchResult
        case searchResult of
            Left TooManyResultsErr -> 
                main' (nextDeepQuery query)
            _ -> if (nextQuery query) >= endQuery 
                  then print "done!" else main' (nextQuery query)
    ```

1.  编写辅助函数来定义下一个逻辑查询，如下所示：

    ```py
    nextDeepQuery query = query ++ "a"

    nextQuery "z" = endQuery
    nextQuery query = if last query == 'z'
                      then nextQuery $ init query
                      else init query ++ [succ $ last query]
    endQuery = [succ 'z']
    ```

## 它是如何工作的……

代码开始时会在目录查找中搜索 "a"。这很可能会由于结果过多而发生错误。因此，在下一次迭代中，代码会通过查询 "aa" 来细化搜索，再接着是 "aaa"，直到不再出现 `TooManyResultsErr :: SearchResultErr`。

然后，它将枚举到下一个逻辑搜索查询 "aab"，如果没有结果，它将搜索 "aac"，依此类推。这个强制前缀搜索将获取数据库中的所有项目。我们可以收集大量数据，比如姓名和部门类型，稍后进行有趣的聚类或分析。下图展示了程序的启动方式：

![它是如何工作的...](img/6331OS_01_04.jpg)

# 在 Haskell 中使用 MongoDB 查询

MongoDB 是一个非关系型的无模式数据库。在这个方法中，我们将把所有数据从 MongoDB 获取到 Haskell 中。

## 准备工作

我们需要在本地机器上安装 MongoDB，并在运行此方法中的代码时，确保后台有一个数据库实例在运行。

MongoDB 安装说明位于 [`www.mongodb.org`](http://www.mongodb.org)。在基于 Debian 的操作系统中，我们可以使用 `apt-get` 安装 MongoDB，命令如下：

```py
$ sudo apt-get install mongodb

```

通过指定数据库文件路径，运行数据库守护进程，方法如下：

```py
$ mkdir ~/db
$ mongod --dbpath ~/db

```

填充一个名为 `"people"` 的集合，插入虚拟数据，方法如下：

```py
$ mongo
> db.people.insert( {first: "Joe", last: "Shmoe"} )

```

使用以下命令从 Cabal 安装 MongoDB 包：

```py
$ cabal install mongoDB

```

## 如何做……

1.  使用 `OverloadedString` 和 `ExtendedDefaultRules` 语言扩展来使 MongoDB 库更容易使用：

    ```py
    {-# LANGUAGE OverloadedStrings, ExtendedDefaultRules #-}
    import Database.MongoDB
    ```

1.  定义并实现 `main` 来设置与本地托管数据库的连接。运行 `run` 函数中定义的 MongoDB 查询，方法如下：

    ```py
    main :: IO ()
    main = do
        let db = "test"
        pipe <- runIOE $ connect (host "127.0.0.1")
        e <- access pipe master db run
        close pipe
        print e
    ```

1.  在 `run` 中，我们可以结合多个操作。对于这个方法，`run` 将只执行一个任务，即从 `"people"` 集合中收集数据：

    ```py
    run = getData

    getData = rest =<< find (select [] "people") {sort=[]}
    ```

## 它是如何工作的……

驱动程序在运行的程序与数据库之间建立了管道。这使得运行 MongoDB 操作能够将程序与数据库连接起来。`find` 函数接收一个查询，我们通过调用 `select :: Selector -> Collection -> aQueryOrSelection` 函数来构建查询。

其他函数可以在文档中找到：[`hackage.haskell.org/package/mongoDB/docs/Database-MongoDB-Query.html`](http://hackage.haskell.org/package/mongoDB/docs/Database-MongoDB-Query.html)

## 另见

如果 MongoDB 数据库在远程服务器上，请参考*从远程 MongoDB 服务器读取数据*这一方法，来设置与远程数据库的连接。

# 从远程 MongoDB 服务器读取数据

在许多情况下，可能在远程计算机上设置 MongoDB 实例更加可行。本做法将介绍如何从远程托管的 MongoDB 获取数据。

## 准备工作

我们应创建一个远程数据库。MongoLab（[`mongolab.com`](https://mongolab.com)）和 MongoHQ（[`www.mongohq.com`](http://www.mongohq.com)）提供作为服务的 MongoDB，并且有免费的选项来设置一个小型开发数据库。

### 提示

这些服务要求我们接受其条款和条件。对某些人来说，将数据库托管在我们自己的远程服务器上可能是最好的选择。

按如下方式从 Cabal 安装 MongoDB 包：

```py
$ cabal install mongoDB

```

还需安装以下辅助库：

```py
$ cabal install split
$ cabal install uri

```

## 如何做……

1.  使用库所需的`OverloadedString`和`ExtendedDefaultRules`语言扩展。按如下方式导入辅助函数：

    ```py
    {-# LANGUAGE OverloadedStrings, ExtendedDefaultRules #-}
    import Database.MongoDB
    import Text.URI
    import Data.Maybe
    import qualified Data.Text as T
    import Data.List.Split
    ```

1.  按如下方式指定数据库连接的远程 URI：

    ```py
    mongoURI = "mongodb://user:pass@ds12345.mongolab.com:53788/mydb"
    ```

1.  用户名、密码、主机名、端口地址和数据库名称必须从 URI 中提取，如下代码片段所示：

    ```py
    uri = fromJust $ parseURI mongoURI

    getUser = head $ splitOn ":" $ fromJust $ uriUserInfo uri

    getPass = last $ splitOn ":" $ fromJust $ uriUserInfo uri

    getHost = fromJust $ uriRegName uri

    getPort = case uriPort uri of 
        Just port -> show port 
        Nothing -> (last.words.show) defaultPort

    getDb = T.pack $ tail $ uriPath uri
    ```

1.  通过读取远程 URI 的主机端口来创建数据库连接，如下所示：

    ```py
    main :: IO ()
    main = do
        let hostport = getHost ++ ":" ++ getPort
        pipe <- runIOE $ connect (readHostPort hostport)
        e <- access pipe master getDb run
        close pipe
        print e
    ```

1.  可选地，对数据库进行身份验证并按如下方式从`"people"`集合中获取数据：

    ```py
    run = do
      auth (T.pack getUser) (T.pack getPass)
      getData

    getData = rest =<< find (select [] "people") {sort=[]}
    ```

## 另见

如果数据库在本地计算机上，请参阅*在 Haskell 中使用 MongoDB 查询*这一做法。

# 探索 SQLite 数据库中的数据

SQLite 是一个关系型数据库，它执行严格的模式。它仅仅是机器上的一个文件，我们可以通过**结构化查询语言**（**SQL**）与之交互。Haskell 有一个易于使用的库来将这些 SQL 命令发送到我们的数据库。

在本做法中，我们将使用这样的库来提取 SQLite 数据库中的所有数据。

## 准备工作

如果 SQLite 数据库尚未设置，我们需要先安装它。可以从[`www.sqlite.org`](http://www.sqlite.org)获取。在 Debian 系统中，我们可以通过以下命令从`apt-get`获取：

```py
$ sudo apt-get install sqlite3

```

现在创建一个简单的数据库来测试我们的代码，使用以下命令：

```py
$ sqlite3 test.db "CREATE TABLE test \
(id INTEGER PRIMARY KEY, str text); \
INSERT INTO test (str) VALUES ('test string');"

```

我们还必须按如下方式从 Cabal 安装 SQLite Haskell 包：

```py
$ cabal install sqlite-simple

```

本做法将详细分析库文档页面上展示的示例代码，页面地址为[`hackage.haskell.org/package/sqlite-simple/docs/Database-SQLite-Simple.html`](http://hackage.haskell.org/package/sqlite-simple/docs/Database-SQLite-Simple.html)。

## 如何做……

1.  使用`OverloadedStrings`语言扩展并导入相关库，如下代码所示：

    ```py
    {-# LANGUAGE OverloadedStrings #-}

    import Control.Applicative
    import Database.SQLite.Simple
    import Database.SQLite.Simple.FromRow
    ```

1.  为每个 SQLite 表字段定义一个数据类型。为它提供`FromRow`类型类的实例，以便我们可以轻松地从表中解析它，如下代码片段所示：

    ```py
    data TestField = TestField Int String deriving (Show)

    instance FromRow TestField where
      fromRow = TestField <$> field <*> field
    ```

1.  最后，按如下方式打开数据库并导入所有内容：

    ```py
    main :: IO ()
    main = do
      conn <- open "test.db"
      r <- query_ conn "SELECT * from test" :: IO [TestField]
      mapM_ print r
      close conn
    ```
