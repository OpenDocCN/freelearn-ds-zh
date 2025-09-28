# 第九章. 解析特定数据

在本章中，我们将涵盖：

+   使用 Dateutil 解析日期和时间

+   时区查找和转换

+   使用 Timex 标记时间表达式

+   使用 lxml 从 HTML 中提取 URL

+   清理和剥离 HTML

+   使用 BeautifulSoup 转换 HTML 实体

+   检测和转换字符编码

# 引言

本章涵盖了解析特定类型的数据，主要关注日期、时间和 HTML。幸运的是，有多个有用的库可以完成这项任务，所以我们不必深入研究复杂且过于复杂的正则表达式。这些库可以很好地补充 NLTK：

+   `dateutil`：提供日期/时间解析和时区转换

+   `timex`：可以在文本中识别时间词

+   `lxml`和`BeautifulSoup`：可以解析、清理和转换 HTML

+   `chardet`：检测文本的字符编码

这些库在将文本传递给 NLTK 对象之前进行预处理，或者在 NLTK 处理和提取后的文本进行后处理时非常有用。以下是一个将许多这些工具结合在一起的示例。

假设你需要解析一篇关于餐厅的博客文章。你可以使用`lxml`或`BeautifulSoup`提取文章文本、外链以及文章写作的日期和时间。然后，可以使用`dateutil`将这些日期和时间解析为 Python 的`datetime`对象。一旦你有了文章文本，你可以使用`chardet`确保它是 UTF-8 编码，然后在清理 HTML 并通过基于 NLTK 的词性标注、分块提取和/或文本分类进行处理之前，创建关于文章的额外元数据。如果你在餐厅有活动，你可能可以通过查看`timex`识别的时间词来发现这一点。这个示例的目的是说明现实世界的文本处理往往需要比基于 NLTK 的自然语言处理更多，而本章涵盖的功能可以帮助满足这些额外需求。

# 使用 Dateutil 解析日期和时间

如果您需要在 Python 中解析日期和时间，没有比`dateutil`更好的库了。`parser`模块可以解析比这里展示的更多格式的`datetime`字符串，而`tz`模块提供了查找时区所需的一切。这两个模块结合起来，使得将字符串解析为时区感知的`datetime`对象变得相当容易。

## 准备工作

您可以使用`pip`或`easy_install`安装`dateutil`，即`sudo pip install dateutil`或`sudo easy_install dateutil`。完整的文档可以在[`labix.org/python-dateutil`](http://labix.org/python-dateutil)找到。

## 如何做到这一点...

让我们深入几个解析示例：

```py
>>> from dateutil import parser
>>> parser.parse('Thu Sep 25 10:36:28 2010')
datetime.datetime(2010, 9, 25, 10, 36, 28)
>>> parser.parse('Thursday, 25\. September 2010 10:36AM')
datetime.datetime(2010, 9, 25, 10, 36)
>>> parser.parse('9/25/2010 10:36:28')
datetime.datetime(2010, 9, 25, 10, 36, 28)
>>> parser.parse('9/25/2010')
datetime.datetime(2010, 9, 25, 0, 0)
>>> parser.parse('2010-09-25T10:36:28Z')
datetime.datetime(2010, 9, 25, 10, 36, 28, tzinfo=tzutc())
```

如您所见，只需导入`parser`模块，并使用`datetime`字符串调用`parse()`函数即可。解析器将尽力返回一个合理的`datetime`对象，但如果它无法解析字符串，它将引发`ValueError`。

## 它是如何工作的...

解析器不使用正则表达式。相反，它寻找可识别的标记，并尽力猜测这些标记代表什么。这些标记的顺序很重要，例如，一些文化使用看起来像 *Month/Day/Year*（默认顺序）的日期格式，而其他文化使用 *Day/Month/Year* 格式。为了处理这个问题，`parse()` 函数接受一个可选的关键字参数 `dayfirst`，默认为 `False`。如果你将其设置为 `True`，它可以正确解析后者的日期格式。

```py
>>> parser.parse('25/9/2010', dayfirst=True)
datetime.datetime(2010, 9, 25, 0, 0)
```

还可能发生与两位数年份相关的排序问题。例如，`'10-9-25'` 是模糊的。由于 `dateutil` 默认为 *Month-Day-Year* 格式，`'10-9-25'` 被解析为 2025 年。但如果你在 `parse()` 中传递 `yearfirst=True`，它将被解析为 2010 年。

```py
>>> parser.parse('10-9-25')
datetime.datetime(2025, 10, 9, 0, 0)
>>> parser.parse('10-9-25', yearfirst=True)
datetime.datetime(2010, 9, 25, 0, 0)
```

## 还有更多...

`dateutil` 解析器还可以进行 **模糊解析**，这允许它忽略 `datetime` 字符串中的无关字符。默认值为 `False` 时，`parse()` 遇到未知标记时会引发 `ValueError`。但如果 `fuzzy=True`，则通常可以返回 `datetime` 对象。

```py
>>> try:
...    parser.parse('9/25/2010 at about 10:36AM')
... except ValueError:
...    'cannot parse'
'cannot parse'
>>> parser.parse('9/25/2010 at about 10:36AM', fuzzy=True)
datetime.datetime(2010, 9, 25, 10, 36)
```

## 参见

在下一个配方中，我们将使用 `dateutil` 的 `tz` 模块来进行时区查找和转换。

# 时区查找和转换

从 `dateutil` 解析器返回的大多数 `datetime` 对象是 *naive*，这意味着它们没有显式的 `tzinfo`，它指定了时区和 UTC 偏移量。在先前的配方中，只有一个示例有 `tzinfo`，这是因为它是 UTC 日期和时间字符串的标准 ISO 格式。**UTC** 是协调世界时，它与 GMT 相同。**ISO** 是 **国际标准化组织**，它规定了标准日期和时间格式。

Python 的 `datetime` 对象可以是 *naive* 或 *aware*。如果一个 `datetime` 对象有 `tzinfo`，则它是 aware 的。否则，`datetime` 是 naive 的。要使 naive `datetime` 对象时区感知，你必须给它一个显式的 `tzinfo`。然而，Python 的 `datetime` 库只定义了一个 `tzinfo` 的抽象基类，并将其留给其他人来实现 `tzinfo` 的创建。这就是 `dateutil` 的 `tz` 模块发挥作用的地方——它提供了从你的操作系统时区数据中查找时区所需的一切。

## 准备工作

应使用 `pip` 或 `easy_install` 安装 `dateutil`。你还应该确保你的操作系统有时区数据。在 Linux 上，这通常位于 `/usr/share/zoneinfo`，Ubuntu 软件包称为 `tzdata`。如果你在 `/usr/share/zoneinfo` 中有多个文件和目录，例如 `America/`、`Europe/` 等，那么你应该准备好继续。以下示例显示了 Ubuntu Linux 的目录路径。

## 如何操作...

让我们从获取一个 UTC `tzinfo` 对象开始。这可以通过调用 `tz.tzutc()` 来完成，你可以通过调用带有 UTC `datetime` 对象的 `utcoffset()` 方法来检查偏移量是否为 **0**。

```py
>>> from dateutil import tz
>>> tz.tzutc()
tzutc()
>>> import datetime
>>> tz.tzutc().utcoffset(datetime.datetime.utcnow())
datetime.timedelta(0)
```

要获取其他时区的 `tzinfo` 对象，你可以将时区文件路径传递给 `gettz()` 函数。

```py
>>> tz.gettz('US/Pacific')
tzfile('/usr/share/zoneinfo/US/Pacific')
>>> tz.gettz('US/Pacific').utcoffset(datetime.datetime.utcnow())
datetime.timedelta(-1, 61200)
>>> tz.gettz('Europe/Paris')
tzfile('/usr/share/zoneinfo/Europe/Paris')
>>> tz.gettz('Europe/Paris').utcoffset(datetime.datetime.utcnow())
datetime.timedelta(0, 7200)
```

你可以看到 UTC 偏移是 `timedelta` 对象，其中第一个数字是**天**，第二个数字是**秒**。

### 小贴士

如果你将 `datetimes` 存储在数据库中，将它们全部存储在 UTC 中以消除任何时区歧义是个好主意。即使数据库可以识别时区，这也是一个好习惯。

要将非 UTC 的 `datetime` 对象转换为 UTC，它必须成为时区感知的。如果你尝试将无知的 `datetime` 转换为 UTC，你会得到一个 `ValueError` 异常。要使无知的 `datetime` 时区感知，你只需使用正确的 `tzinfo` 调用 `replace()` 方法。一旦 `datetime` 对象有了 `tzinfo`，就可以通过调用 `astimezone()` 方法并传递 `tz.tzutc()` 来执行 UTC 转换。

```py
>>> pst = tz.gettz('US/Pacific')
>>> dt = datetime.datetime(2010, 9, 25, 10, 36)
>>> dt.tzinfo
>>> dt.astimezone(tz.tzutc())
Traceback (most recent call last):
  File "/usr/lib/python2.6/doctest.py", line 1248, in __run
  compileflags, 1) in test.globs
  File "<doctest __main__[22]>", line 1, in <module>
  dt.astimezone(tz.tzutc())
ValueError: astimezone() cannot be applied to a naive datetime
>>> dt.replace(tzinfo=pst)
datetime.datetime(2010, 9, 25, 10, 36, tzinfo=tzfile('/usr/share/zoneinfo/US/Pacific'))
>>> dt.replace(tzinfo=pst).astimezone(tz.tzutc())
datetime.datetime(2010, 9, 25, 17, 36, tzinfo=tzutc())
```

## 它是如何工作的...

`tzutc` 和 `tzfile` 对象都是 `tzinfo` 的子类。因此，它们知道时区转换的正确 UTC 偏移（对于 `tzutc` 是 0）。`tzfile` 对象知道如何读取操作系统的 `zoneinfo` 文件以获取必要的偏移数据。`datetime` 对象的 `replace()` 方法做的是它的名字所暗示的——替换属性。一旦 `datetime` 有了一个 `tzinfo`，`astimezone()` 方法将能够使用 UTC 偏移转换时间，然后使用新的 `tzinfo` 替换当前的 `tzinfo`。

### 注意

注意，`replace()` 和 `astimezone()` 都返回**新**的 `datetime` 对象。它们不会修改当前对象。

## 更多...

你可以将 `tzinfos` 关键字参数传递给 `dateutil` 解析器以检测其他未被识别的时间区域。

```py
>>> parser.parse('Wednesday, Aug 4, 2010 at 6:30 p.m. (CDT)', fuzzy=True)
datetime.datetime(2010, 8, 4, 18, 30)
>>> tzinfos = {'CDT': tz.gettz('US/Central')}
>>> parser.parse('Wednesday, Aug 4, 2010 at 6:30 p.m. (CDT)', fuzzy=True, tzinfos=tzinfos)
datetime.datetime(2010, 8, 4, 18, 30, tzinfo=tzfile('/usr/share/zoneinfo/US/Central'))
```

在第一种情况下，我们得到一个无知的 `datetime`，因为时区没有被识别。然而，当我们传递 `tzinfos` 映射时，我们得到一个时区感知的 `datetime`。

### 本地时区

如果你想要查找你的本地时区，你可以调用 `tz.tzlocal()`，这将使用操作系统认为的本地时区。在 Ubuntu Linux 中，这通常在 `/etc/timezone` 文件中指定。

### 自定义偏移

你可以使用 `tzoffset` 对象创建具有自定义 UTC 偏移的 `tzinfo` 对象。可以创建一个一小时的自定义偏移，如下所示：

```py
>>> tz.tzoffset('custom', 3600)
tzoffset('custom', 3600)
```

你必须提供名称作为第一个参数，以及以秒为单位的偏移时间作为第二个参数。

## 参见

之前的配方涵盖了使用 `dateutil.parser` 解析 `datetime` 字符串。

# 使用 Timex 标记时间表达式

NLTK 项目有一个鲜为人知的 `contrib` 仓库，其中包含许多其他模块，包括一个名为 `timex.py` 的模块，它可以标记时间表达式。**时间表达式**只是一些时间词，如“本周”或“下个月”。这些是相对于某个其他时间点的模糊表达式，比如文本编写的时间。`timex` 模块提供了一种注释文本的方法，以便可以从文本中提取这些表达式进行进一步分析。更多关于 TIMEX 的信息可以在 [`timex2.mitre.org/`](http://timex2.mitre.org/) 找到。

## 准备工作

`timex.py` 模块是 `nltk_contrib` 包的一部分，它独立于 NLTK 的当前版本。这意味着你需要自己安装它，或者使用书中代码下载中包含的 `timex.py` 模块。你也可以直接从 [`code.google.com/p/nltk/source/browse/trunk/nltk_contrib/nltk_contrib/timex.py`](http://code.google.com/p/nltk/source/browse/trunk/nltk_contrib/nltk_contrib/timex.py) 下载 `timex.py`。

如果你想要安装整个 `nltk_contrib` 包，你可以从 [`nltk.googlecode.com/svn/trunk/`](http://nltk.googlecode.com/svn/trunk/) 检出源代码，并在 `nltk_contrib` 文件夹中执行 `sudo python setup.py install`。如果你这样做，你需要执行 `from nltk_contrib import timex` 而不是在下面的 *如何操作* 部分中直接执行 `import timex`。

对于这个配方，你必须将 `timex.py` 模块下载到与代码其余部分相同的文件夹中，这样 `import timex` 就不会引发 `ImportError`。

你还需要安装 `egenix-mx-base` 包。这是一个用于 Python 的 C 扩展库，所以如果你已经安装了所有正确的 Python 开发头文件，你应该能够执行 `sudo pip install egenix-mx-base` 或 `sudo easy_install egenix-mx-base`。如果你正在运行 Ubuntu Linux，你可以改为执行 `sudo apt-get install python-egenix-mxdatetime`。如果这些都不起作用，你可以访问 [`www.egenix.com/products/python/mxBase/`](http://www.egenix.com/products/python/mxBase/) 下载该包并找到安装说明。

## 如何操作...

使用 `timex` 非常简单：将一个字符串传递给 `timex.tag()` 函数，并返回一个带有注释的字符串。这些注释将是围绕每个时间表达式的 XML `TIMEX` 标签。

```py
>>> import timex
>>> timex.tag("Let's go sometime this week")
"Let's go sometime <TIMEX2>this week</TIMEX2>"
>>> timex.tag("Tomorrow I'm going to the park.")
"<TIMEX2>Tomorrow</TIMEX2> I'm going to the park."
```

## 它是如何工作的...

`timex.py` 的实现基本上是超过 300 行的条件正则表达式匹配。当其中一个已知表达式匹配时，它创建一个 `RelativeDateTime` 对象（来自 `mx.DateTime` 模块）。然后，这个 `RelativeDateTime` 被转换回一个带有周围 `TIMEX` 标签的字符串，并替换文本中的原始匹配字符串。

## 还有更多...

`timex` 非常智能，不会对已经标记的表达式再次进行标记，因此可以将标记过的 `TIMEX` 文本传递给 `tag()` 函数。

```py
>>> timex.tag("Let's go sometime <TIMEX2>this week</TIMEX2>")
"Let's go sometime <TIMEX2>this week</TIMEX2>"
```

## 相关内容

在下一个菜谱中，我们将从 HTML 中提取 URL，但可以使用相同的模块和技术来提取用于进一步处理的`TIMEX`标记表达式。

# 使用 lxml 从 HTML 中提取 URL

解析 HTML 时的一项常见任务是提取链接。这是每个通用网络爬虫的核心功能之一。有多个 Python 库用于解析 HTML，`lxml`是其中之一。正如你将看到的，它包含一些专门针对链接提取的出色辅助函数。

## 准备工作

`lxml`是 C 库`libxml2`和`libxslt`的 Python 绑定。这使得它成为一个非常快速的 XML 和 HTML 解析库，同时仍然保持*pythonic*。然而，这也意味着你需要安装 C 库才能使其工作。安装说明请参阅[`codespe`](http://codespe) [ak.net/lxml/installation.html](http://ak.net/lxml/installation.html)。然而，如果你正在运行 Ubuntu Linux，安装就像`sudo apt-get install python-lxml`一样简单。

## 如何做...

`lxml`包含一个专门用于解析 HTML 的`html`模块。使用`fromstring()`函数，我们可以解析一个 HTML 字符串，然后获取所有链接的列表。`iterlinks()`方法生成形式为`(element, attr, link, pos)`的四元组：

+   `element`：这是从锚标签中提取的`link`的解析节点。如果你只对`link`感兴趣，可以忽略这个。

+   `attr`：这是`link`的来源属性，通常是`href`。

+   `link`：这是从锚标签中提取的实际 URL。

+   `pos`：这是文档中锚标签的数字索引。第一个标签的`pos`为`0`，第二个标签的`pos`为`1`，依此类推。

以下是一些演示代码：

```py
>>> from lxml import html
>>> doc = html.fromstring('Hello <a href="/world">world</a>')
>>> links = list(doc.iterlinks())
>>> len(links)
1
>>> (el, attr, link, pos) = links[0]
>>> attr
'href'
>>> link
'/world'
>>> pos
0
```

## 它是如何工作的...

`lxml`将 HTML 解析为`ElementTree`。这是一个由父节点和子节点组成的树结构，其中每个节点代表一个 HTML 标签，并包含该标签的所有相应属性。一旦创建了树，就可以迭代以查找元素，例如`a`或**锚标签**。核心树处理代码位于`lxml.etree`模块中，而`lxml.html`模块只包含创建和迭代树的 HTML 特定函数。有关完整文档，请参阅 lxml 教程：[`codespeak.net/lxml/tutorial.html`](http://codespeak.net/lxml/tutorial.html)。

## 更多内容...

你会注意到在之前的代码中，链接是**相对的**，这意味着它不是一个绝对 URL。在提取链接之前，我们可以通过调用带有基本 URL 的`make_links_absolute()`方法来将其转换为**绝对 URL**。

```py
>>> doc.make_links_absolute('http://hello')
>>> abslinks = list(doc.iterlinks())
>>> (el, attr, link, pos) = abslinks[0]
>>> link
'http://hello/world'
```

### 直接提取链接

如果你只想提取链接而不做其他任何事情，你可以使用 HTML 字符串调用`iterlinks()`函数。

```py
>>> links = list(html.iterlinks('Hello <a href="/world">world</a>'))
>>> links[0][2]
'/world'
```

### 从 URL 或文件解析 HTML

你可以使用 `parse()` 函数而不是使用 `fromstring()` 函数来解析 HTML 字符串，通过传递一个 URL 或文件名。例如，`html.parse("http://my/url")` 或 `html.parse("/path/to/file")`。结果将与你自己将 URL 或文件加载到字符串中然后调用 `fromstring()` 一样。

### 使用 XPaths 提取链接

你也可以使用 `xpath()` 方法来获取链接，而不是使用 `iterlinks()` 方法，这是一个从 HTML 或 XML 解析树中提取任何所需内容的一般方法。

```py
>>> doc.xpath('//a/@href')[0]
'http://hello/world'
```

关于 XPath 语法，请参阅 [`www.w3schools.com/XPath/xpath_syntax.asp`](http://www.w3schools.com/XPath/xpath_syntax.asp)。

## 参见

在下一个配方中，我们将介绍清理和剥离 HTML。

# 清理和剥离 HTML

清理文本是文本处理中不幸但完全必要的方面之一。当涉及到解析 HTML 时，你可能不想处理任何嵌入的 JavaScript 或 CSS，你只对标签和文本感兴趣。或者你可能想完全移除 HTML，只处理文本。这个配方涵盖了如何进行这两种预处理操作。

## 准备工作

你需要安装 `lxml`。请参阅前面的配方或 [`codespeak.net/lxml/installation.html`](http://codespeak.net/lxml/installation.html) 以获取安装说明。你还需要安装 NLTK 以剥离 HTML。

## 如何操作...

我们可以使用 `lxml.html.clean` 模块中的 `clean_html()` 函数来从 HTML 字符串中移除不必要的 HTML 标签和嵌入的 JavaScript。

```py
>>> import lxml.html.clean
>>> lxml.html.clean.clean_html('<html><head></head><body onload=loadfunc()>my text</body></html>')
'<div><body>my text</body></div>'
```

结果会更加干净，更容易处理。使用 `clean_html()` 函数的完整模块路径是因为 `nltk.util` 模块中也有一个 `clean_html()` 函数，但它的用途不同。当你只想得到文本时，`nltk.util.clean_html()` 函数会移除所有 HTML 标签。

```py
>>> import nltk.util
>>> nltk.util.clean_html('<div><body>my text</body></div>')
'my text'
```

## 它是如何工作的...

`lxml.html.clean_html()` 函数将 HTML 字符串解析成树，然后迭代并移除所有应该被移除的节点。它还使用正则表达式匹配和替换来清理节点的非必要属性（例如嵌入的 JavaScript）。

`nltk.util.clean_html()` 函数执行一系列正则表达式替换来移除 HTML 标签。为了安全起见，最好在清理后剥离 HTML，以确保正则表达式能够匹配。

## 还有更多...

`lxml.html.clean` 模块定义了一个默认的 `Cleaner` 类，当你调用 `clean_html()` 时会使用它。你可以通过创建自己的实例并调用其 `clean_html()` 方法来自定义这个类的行为。有关这个类的更多详细信息，请参阅 [`codespeak.net/lxml/lxmlhtml.html`](http://codespeak.net/lxml/lxmlhtml.html)。

## 参见

在前面的配方中介绍了 `lxml.html` 模块，用于解析 HTML 和提取链接。在下一个配方中，我们将介绍取消转义 HTML 实体。

# 使用 BeautifulSoup 转换 HTML 实体

HTML 实体是像 `&amp;` 或 `&lt;` 这样的字符串。这些是具有特殊用途于 HTML 的正常 ASCII 字符的编码。例如，`&lt;` 是 `<` 的实体。你无法在 HTML 标签内直接使用 `<`，因为它是一个 HTML 标签的开始字符，因此需要转义它并定义 `&lt;` 实体。& 的实体代码是 `&amp;`；正如我们刚才看到的，这是实体代码的开始字符。如果你需要处理 HTML 文档中的文本，那么你将想要将这些实体转换回它们的正常字符，这样你就可以识别并适当地处理它们。

## 准备工作

你需要安装 `BeautifulSoup`，你可以使用 `sudo pip install BeautifulSoup` 或 `sudo easy_install BeautifulSoup` 来完成。你可以在 [`www.crummy.com/software/BeautifulSoup/`](http://www.crummy.com/software/BeautifulSoup/) 上了解更多关于 `BeautifulSoup` 的信息。

## 如何做到这一点...

`BeautifulSoup` 是一个 HTML 解析库，它还包含一个名为 `BeautifulStoneSoup` 的 XML 解析器。这是我们用于实体转换可以使用的。这很简单：给定一个包含 HTML 实体的字符串创建 `BeautifulStoneSoup` 的实例，并指定关键字参数 `convertEntities='html'`。将这个实例转换成字符串，你将得到 HTML 实体的 ASCII 表示。

```py
>>> from BeautifulSoup import BeautifulStoneSoup
>>> unicode(BeautifulStoneSoup('&lt;', convertEntities='html'))
u'<'
>>> unicode(BeautifulStoneSoup('&amp;', convertEntities='html'))
u'&'
```

将字符串多次运行是可以的，只要 ASCII 字符不是单独出现。如果你的字符串只是一个用于 HTML 实体的单个 ASCII 字符，那么这个字符将会丢失。

```py
>>> unicode(BeautifulStoneSoup('<', convertEntities='html'))
u''
>>> unicode(BeautifulStoneSoup('< ', convertEntities='html'))
u'< '
```

为了确保字符不会丢失，只需要在字符串中有一个不是实体代码部分的字符。

## 它是如何工作的...

为了转换 HTML 实体，`BeautifulStoneSoup` 会寻找看起来像实体的标记，并用 Python 标准库中的 `htmlentitydefs.name2codepoint` 字典中对应的值来替换它们。如果实体标记在 HTML 标签内，或者它在一个普通字符串中，它都可以这样做。

## 还有更多...

`BeautifulSoup` 是一个优秀的 HTML 和 XML 解析器，并且可以是一个很好的 `lxml` 的替代品。它在处理格式不正确的 HTML 方面特别出色。你可以在 [`www.crummy.com/software/BeautifulSoup/documentation.html`](http://www.crummy.com/software/BeautifulSoup/documentation.html) 上了解更多关于如何使用它的信息。

### 使用 BeautifulSoup 提取 URL

这里有一个使用 `BeautifulSoup` 提取 URL 的例子，就像我们在 *使用 lxml 从 HTML 中提取 URL* 的配方中所做的那样。你首先使用 HTML 字符串创建 `soup`，然后调用 `findAll()` 方法并传入 `'a'` 以获取所有锚标签，并提取 `'href'` 属性以获取 URL。

```py
>>> from BeautifulSoup import BeautifulSoup
>>> soup = BeautifulSoup('Hello <a href="/world">world</a>')
>>> [a['href'] for a in soup.findAll('a')]
[u'/world']
```

## 参见

在 *使用 lxml 从 HTML 中提取 URL* 的配方中，我们介绍了如何使用 `lxml` 从 HTML 字符串中提取 URL，并在该配方之后介绍了 *清理和去除 HTML*。

# 检测和转换字符编码

在文本处理中，一个常见的情况是找到具有非标准字符编码的文本。理想情况下，所有文本都应该是 ASCII 或 UTF-8，但这只是现实。在您有非 ASCII 或非 UTF-8 文本且不知道字符编码的情况下，您需要检测它并将文本转换为标准编码，然后再进一步处理。

## 准备工作

您需要安装`chardet`模块，使用`sudo pip install chardet`或`sudo easy_install chardet`。您可以在[`chardet.feedparser.org/`](http://chardet.feedparser.org/)了解更多关于`chardet`的信息。

## 如何实现...

`encoding.py`中提供了编码检测和转换函数。这些是围绕`chardet`模块的简单包装函数。要检测字符串的编码，请调用`encoding.detect()`。您将得到一个包含两个属性的`dict`：`confidence`和`encoding`。`confidence`是`chardet`对`encoding`值正确性的置信度概率。

```py
# -*- coding: utf-8 -*-
import chardet

def detect(s):
  try:
    return chardet.detect(s)
  except UnicodeDecodeError:
    return chardet.detect(s.encode('utf-8'))

  def convert(s):
    encoding = detect(s)['encoding']

    if encoding == 'utf-8':
      return unicode(s)
    else:
      return unicode(s, encoding)
```

下面是一个使用`detect()`来确定字符编码的示例代码：

```py
>>> import encoding
>>> encoding.detect('ascii')
{'confidence': 1.0, 'encoding': 'ascii'}
>>> encoding.detect(u'abcdé')
{'confidence': 0.75249999999999995, 'encoding': 'utf-8'}
>>> encoding.detect('\222\222\223\225')
{'confidence': 0.5, 'encoding': 'windows-1252'}
```

要将字符串转换为标准的`unicode`编码，请调用`encoding.convert()`。这将解码字符串的原始编码，然后将其重新编码为 UTF-8。

```py
>>> encoding.convert('ascii')
u'ascii'	
>>> encoding.convert(u'abcdé')
u'abcd\\xc3\\xa9'
>>> encoding.convert('\222\222\223\225')
u'\u2019\u2019\u201c\u2022'
```

## 它是如何工作的...

`detect()`函数是`chardet.detect()`的包装器，可以处理`UnicodeDecodeError`异常。在这些情况下，在尝试检测编码之前，字符串被编码为 UTF-8。

`convert()`函数首先调用`detect()`以获取`encoding`，然后返回一个带有`encoding`作为第二个参数的`unicode`字符串。通过将`encoding`传递给`unicode()`，字符串从原始编码解码，允许它被重新编码为标准编码。

## 更多内容...

模块顶部的注释`# -*- coding: utf-8 -*-`是给 Python 解释器的提示，告诉它代码中字符串应使用哪种编码。这对于您源代码中有非 ASCII 字符串时很有帮助，并在[`www.python.org/dev/peps/pep-0263/`](http://www.python.org/dev/peps/pep-0263/)中详细记录。

### 转换为 ASCII

如果您想要纯 ASCII 文本，将非 ASCII 字符转换为 ASCII 等效字符，或者在没有等效字符的情况下删除，那么您可以使用`unicodedata.normalize()`函数。

```py
>>> import unicodedata
>>> unicodedata.normalize('NFKD', u'abcd\xe9').encode('ascii', 'ignore')
'abcde'
```

将第一个参数指定为`'NFKD'`确保非 ASCII 字符被替换为其等效的 ASCII 版本，并且最终调用`encode()`时使用`'ignore'`作为第二个参数将移除任何多余的 Unicode 字符。

## 参见

在使用`lxml`或`BeautifulSoup`进行 HTML 处理之前，编码检测和转换是推荐的第一步，这包括在*使用 lxml 从 HTML 中提取 URL*和*使用 BeautifulSoup 转换 HTML 实体*的食谱中。
