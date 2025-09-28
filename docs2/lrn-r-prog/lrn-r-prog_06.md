# 第六章. 处理字符串

在上一章中，你学习了多个类别中的许多内置函数，用于处理基本对象。你学习了如何访问对象类、类型和维度；如何进行逻辑、数学和基本统计计算；以及如何执行简单的分析任务，如求根。这些函数是我们解决特定问题的基石。

与字符串相关的函数是一个非常重要的函数类别。它们将在本章中介绍。在 R 中，文本存储在字符向量中，许多函数和技术对于操作和分析文本非常有用。在本章中，你将学习处理字符串的基本和有用技术，包括以下主题：

+   字符向量的基本操作

+   在日期/时间对象与其字符串表示之间进行转换

+   使用正则表达式从文本中提取信息

# 字符串入门

R 中的字符向量用于存储文本数据。你之前了解到，与许多其他编程语言相比，字符向量不是一个由单个字符、字母或字母符号（如 a、b、c）组成的向量。相反，它是一个字符串的向量。

R 还提供了一系列内置函数来处理字符向量。其中许多也执行向量化操作，因此它们可以一步处理多个字符串值。

在本节中，你将学习更多关于打印、组合和转换存储在字符向量中的文本的内容。

## 打印文本

也许我们可以用文本做的最基本的事情就是查看它们。R 提供了几种在控制台中查看文本的方法。

最简单的方法是直接在引号内键入字符串：

```py
"Hello"
## [1] "Hello"
```

就像浮点数的数值向量一样，字符向量是一个字符值或字符串的向量。"Hello"位于第一个位置，是我们之前创建的字符向量的唯一元素。

我们也可以通过简单地评估它来打印存储在变量中的字符串值：

```py
str1 <- "Hello" 
str1
## [1] "Hello"
```

然而，仅仅在循环中写入字符值并不会迭代地打印它。它根本不会打印任何内容：

```py
for (i in 1:3) {
  "Hello"
}
```

这是因为 R 只会在控制台中键入表达式时自动打印表达式的*值*。for 循环不会显式地返回一个值。这种行为也解释了当调用以下两个函数时打印行为之间的差异：

```py
test1 <- function(x) {
  "Hello"
  x
}
test1("World")
## [1] "World"
```

在前面的输出中，`test1`没有打印`Hello`，而是打印了`World`，因为`test1("World")`返回最后一个表达式`x`的值，即`World`，这是函数调用的值，R 自动打印了这个值。让我们假设我们按照以下方式从函数中移除`x`：

```py
test2 <- function(x) {
  "Hello" 
}
test2("World")
## [1] "Hello"
```

然后，`test2`无论`x`取什么值，总是返回`Hello`。因此，R 自动打印表达式`test2("World")`的值，即`Hello`。

如果我们想显式地打印一个对象，我们应该使用`print()`：

```py
print(str1)
## [1] "Hello"
```

然后，字符向量以位置 `[1]` 打印出来。这同样适用于循环：

```py
for (i in 1:3) {
  print(str1) 
}
## [1] "Hello" 
## [1] "Hello" 
## [1] "Hello"
```

它也可以在函数中使用：

```py
test3 <- function(x) {
 print("Hello")
  x
}
test3("World")
## [1] "Hello"
## [1] "World"
```

在某些情况下，我们希望文本以消息的形式出现，而不是带有索引的字符向量。在这种情况下，我们可以调用 `cat()` 或 `message()` 函数：

```py
cat("Hello")
## Hello
```

我们可以更灵活地构建消息：

```py
name <- "Ken"
language <- "R"
cat("Hello,", name, "- a user of", language)
## Hello, Ken - a user of R
```

我们更改输入以打印一个更正式的句子：

```py
cat("Hello, ", name, ", a user of ", language, ".")
## Hello, Ken , a user of R .
```

看起来连接的字符串似乎在不同的参数之间使用了不必要的空格。这是因为默认情况下，空格字符被用作输入字符串之间的分隔符。我们可以通过指定 `sep=` 参数来更改它。在下面的例子中，我们将避免默认的空格分隔符，并在输入字符串中手动写入空格以创建正确的版本：

```py
cat("Hello, ", name, ", a user of ", language, ".", sep = "")
## Hello, Ken, a user of R.
```

另一个函数是 `message()`，它通常用于重要事件等严肃场合。输出文本具有更明显的显示效果。它与 `cat()` 不同，因为它在连接输入字符串时不会自动使用空格分隔符：

```py
message("Hello, ", name, ", a user of ", language, ".")
## Hello, Ken, a user of R.
```

使用 `message()`，我们需要手动写入分隔符，以便显示与之前相同的文本。

`cat()` 和 `message()` 之间的另一个行为差异是，`message()` 会自动在文本末尾添加一个新行，而 `cat()` 则不会。

以下两个例子演示了差异。我们想要打印相同的内容，但得到不同的结果：

```py
for (i in 1:3) {
  cat(letters[[i]]) 
}
## abc
for (i in 1:3) {
  message(letters[[i]]) 
}
## a
## b
## c
```

很明显，每次调用 `cat()` 时，它都会打印输入字符串而不添加新行。结果是三个字母显示在同一行上。相比之下，每次调用 `message()` 时，它都会在输入字符串后添加一个新行。因此，三个字母显示在三个行上。要使用 `cat()` 在新的一行上打印每个字母，我们需要在输入中显式添加一个换行符。下面的代码打印了与上一个例子中 `message()` 相同的内容：

```py
for (i in 1:3) {
  cat(letters[[i]], "\n", sep = "") 
}
## a 
## b 
## c
```

## 字符串连接

在实际应用中，我们经常需要连接几个字符串来构建一个新的字符串。`paste()` 函数用于将几个字符向量连接在一起。此函数也使用空格作为默认的分隔符：

```py
paste("Hello", "world")
## [1] "Hello world"
paste("Hello", "world", sep = "-")
## [1] "Hello-world"
```

如果我们不想使用分隔符，可以设置 `sep=""` 或者使用 `paste0()` 函数：

```py
paste0("Hello", "world")
## [1] "Helloworld"
```

可能你会对 `paste()` 和 `cat()` 混淆，因为它们都能连接字符串。但它们有什么区别呢？尽管这两个函数都能连接字符串，但区别在于 `cat()` 只会将字符串打印到控制台，而 `paste()` 则返回字符串以供进一步使用。以下代码演示了 `cat()` 打印连接的字符串但返回 `NULL`：

```py
value1 <- cat("Hello", "world")
## Hello world
value1
## NULL
```

换句话说，`cat()` 只打印字符串，而 `paste()` 创建一个新的字符向量。

之前的例子展示了 `paste()` 函数与单元素字符向量一起工作的行为。那么，与多元素向量一起工作又是怎样的呢？让我们看看这是如何实现的：

```py
paste(c("A", "B"), c("C", "D"))
## [1] "A C" "B D"
```

我们可以看到`paste()`是按元素工作的，即首先`paste("A", "C")`，然后`paste("B", "D")`，最后将结果收集起来构建一个包含两个元素的字符向量。

如果我们想将结果合并成一个字符串，我们可以通过设置`collapse=`来指定这两个元素再次连接的方式：

```py
paste(c("A", "B"), c("C", "D"),collapse = ", ")
## [1] "A C, B D"
```

如果我们想将它们放在两行中，可以将`collapse`设置为`\n`（新行）：

```py
result <- paste(c("A", "B"), c("C", "D"), collapse = "\n") result
## [1] "A C\nB D"
```

新的字符向量`result`是一个两行字符串，但它的文本表示仍然写在一行中。新行由我们指定的`\n`表示。要查看我们创建的文本，我们需要调用`cat()`：

```py
cat(result)
## A C ## B D
```

现在，两行字符串按照预期格式打印到控制台。同样，`paste0()`也适用。

## 文本转换

将文本转换为另一种形式在许多情况下都很有用。对文本执行多种基本类型的转换很容易。

### 改变大小写

当我们处理文本数据时，输入可能不符合我们的标准。例如，我们期望所有产品都按大写字母评级，从 A 到 F，但实际输入可能包含这些字母的大小写形式。改变大小写有助于确保输入字符串在大小写上的一致性。

`tolower()`函数将文本转换为小写字母，而`toupper()`则相反：

```py
tolower("Hello")
## [1] "hello"
toupper("Hello")
## [1] "HELLO"
```

这在函数接受字符输入时尤其有用。例如，我们可以定义一个函数，在所有可能的情况下，当`type`为`add`时返回`x + y`，当`type`为`times`时返回`x * y`。最好的做法是无论输入值如何，总是将`type`转换为小写或大写：

```py
calc <- function(type, x, y) {
  type <- tolower(type)
  if (type == "add") {
    x + y 
  }else if (type == "times") {
    x * y
  } else {
    stop("Not supported type of command")
  }
}
c(calc("add", 2, 3), calc("Add", 2, 3), calc("TIMES", 2, 3))
## [1] 5 5 6
```

这使得对相似输入（仅在大小写不同的情况下）有更多的容错性，从而使`type`不区分大小写。

此外，这两个函数是向量化的，也就是说，它会改变给定字符向量中每个字符串元素的大小写：

```py
toupper(c("Hello", "world"))
## [1] "HELLO" "WORLD"
```

### 计数字符

另一个有用的函数是`nchar()`，它简单地计算字符向量中每个元素的字符数：

```py
nchar("Hello")
## [1] 5
```

与`toupper()`和`tolower()`类似，`nchar()`也是向量化的：

```py
nchar(c("Hello", "R", "User"))
## [1] 5 1 4
```

这个函数通常用于检查是否向其提供了有效的字符串参数。例如，以下函数接受学生的某些个人信息并将其存储到数据库中：

```py
store_student <- function(name, age) {
 stopifnot(length(name) == 1, nchar(name) >= 2,
 is.numeric(age), age > 0) 
  # store the information in the database 
}
```

在将信息存储到数据库之前，函数使用`stopifnot()`来检查`name`和`age`是否提供了有效的值。如果用户没有提供有意义的名称（例如，不少于两个字母），函数将停止并显示错误：

```py
store_student("James", 20)
store_student("P", 23)
## Error: nchar(name) >= 2 is not TRUE
```

注意，`nchar(x) == 0`等价于`x == ""`。要检查空字符串，两种方法都适用。

### 去除前后空白字符

在前面的例子中，我们使用了 `nchar()` 来检查 `name` 是否有效。然而，有时输入数据会包含无用的空白字符。这增加了数据的噪声，并需要仔细检查字符串参数。例如，上一节中的 `store_student()` 函数对 ``"P"`` 这样的名字进行了处理，它和直接的 ``"P"`` 参数一样无效，但 `nchar(" P")` 返回 `3`：

```py
store_student(" P", 23)
```

为了考虑可能性，我们需要改进 `store_student()` 函数。在 R 3.2.0 中，引入了 `trimws()` 函数，用于去除给定字符串的前导和/或尾随空白字符：

```py
store_student2 <- function(name, age) {
 stopifnot(length(name) == 1, nchar(trimws(name)) >= 2,
 is.numeric(age), age > 0) 
  # store the information in the database 
}
```

现在，函数对噪声数据更加鲁棒：

```py
store_student2(" P", 23)
## Error: nchar(trimws(name)) >= 2 is not TRUE
```

默认情况下，该函数会去除前导和尾随空白字符，包括空格和制表符。您可以指定“left”或“right”以仅去除字符串的一侧：

```py
trimws(c(" Hello", "World "), which = "left")
## [1] "Hello" "World "
```

### 子字符串

在前面的章节中，你学习了如何对向量和列表进行子集操作。我们还可以通过调用 `substr()` 来对字符向量中的文本进行子集操作。假设我们有以下形式的几个日期：

```py
dates <- c("Jan 3", "Feb 10", "Nov 15")
```

所有月份都由三个字母的缩写表示。我们可以使用 `substr()` 来提取月份：

```py
substr(dates, 1, 3)
## [1] "Jan" "Feb" "Nov"
```

要提取日期，我们需要同时使用 `substr()` 和 `nchar()`：

```py
substr(dates, 5, nchar(dates))
## [1] "3" "10" "15"
```

现在我们可以从输入字符串中提取月份和日期，因此编写一个函数将此类格式的字符串转换为数值表示相同的日期是有用的。以下函数使用了你之前学到的许多函数和思想：

```py
get_month_day <- function(x) {
  months <- vapply(substr(tolower(x), 1, 3), function(md) { 
    switch(md, jan = 1, feb = 2, mar = 3, apr = 4, may = 5,
    jun = 6, jul = 7, aug = 8, sep = 9, oct = 10, nov = 11, dec = 12) 
  }, numeric(1), USE.NAMES = FALSE) 
  days <- as.numeric(substr(x, 5,nchar(x)))
 data.frame(month = months, day = days) 
}
get_month_day(dates)
##   month day 
## 1   1    3 
## 2   2   10 
## 3  11   15
```

`substr()` 函数还有一个对应的函数，用于用给定的字符向量替换子字符串：

```py
substr(dates, 1, 3) <- c("Feb", "Dec", "Mar") dates
## [1] "Feb 3" "Dec 10" "Mar 15"
```

### 文本分割

在许多情况下，要提取的字符串部分的长度不是固定的。例如，人名如 "Mary Johnson" 或 "Jack Smiths" 首名和姓氏的长度没有固定值。使用 `substr()`，如你在上一节中学到的，来分离和提取这两部分会更困难。此类格式的文本有一个常规分隔符，如空格或逗号。为了提取有用的部分，我们需要分割文本并使每个部分可访问。`strsplit()` 函数用于通过指定的字符向量分隔符分割文本：

```py
strsplit("a,bb,ccc", split = ",")
## [[1]] 
## [1] "a" "bb" "ccc"
```

该函数返回一个列表。列表中的每个元素都是从原始字符向量中分割该元素产生的字符向量。这是因为 `strsplit()`，就像我们之前介绍的所有字符串函数一样，也是向量化的，即它返回一个分割的字符向量列表作为结果：

```py
students <- strsplit(c("Tony, 26, Physics", "James, 25, Economics"),
split = ", ") 
students
## [[1]] 
## [1] "Tony" "26" "Physics" 
## 
## [[2]] 
## [1] "James" "25" "Economics"
```

`strsplit()` 函数通过逐元素工作返回一个包含分割部分的字符向量列表。在实践中，分割只是提取或重新组织数据的第一步。为了继续，我们可以使用 `rbind` 将数据放入矩阵中，并为列赋予适当的名称：

```py
students_matrix <- do.call(rbind, students)colnames(students_matrix) <- c("name", "age", "major")students_matrix
##       name   age   major 
## [1,] "Tony"  "26"  "Physics" 
## [2,] "James" "25"  "Economics"
```

然后，我们将矩阵转换为数据框，这样我们就可以将每一列转换为更合适的类型：

```py
students_df <- data.frame(students_matrix, stringsAsFactors = FALSE)students_df$age <- as.numeric(students_df$age)students_df
##   name  age major
## 1 Tony  26  Physics
## 2 James 25  Economics
```

现在，原始字符串输入的学生已经转换为更组织化和更有用的数据框 `students_df`。

一个将整个字符串拆分为单个字符的小技巧是使用一个空的 `split` 参数：

```py
strsplit(c("hello", "world"), split = "")
## [[1]] 
## [1] "h" "e" "l" "l" "o" 
## 
## [[2]] 
## [1] "w" "o" "r" "l" "d"
```

实际上，`strsplit()` 的功能比显示的还要强大。它还支持 *正则表达式*，这是一种非常强大的文本数据处理框架。我们将在本章的最后部分介绍这个主题。

## 格式化文本

使用 `paste()` 连接文本有时不是一个好主意，因为文本需要被分割成多个部分，随着格式的变长，阅读起来会变得更加困难。

例如，假设我们需要以下格式打印 `students_df` 中的每条记录：

```py
#1, name: Tony, age: 26, major: Physics
```

在这种情况下，使用 `paste()` 将会非常麻烦：

```py
cat(paste("#", 1:nrow(students_df), ", name: ", students_df$name, ", age: ", students_df$age, ", major: ", students_df$major, sep = ""), sep = "\n")
## #1, name: Tony, age: 26, major: Physics 
## #2, name: James, age: 25, major: Economics
```

代码看起来很杂乱，初次看很难一眼看出通用模板。相比之下，`sprintf()` 支持格式化模板，并以一种优雅的方式解决了这个问题：

```py
cat(sprintf("#%d, name: %s, age: %d, major: %s", 
  1:nrow(students_df), students_df$name, students_df$age, 
students_df$major), sep = "\n")
#1, name: Tony, age: 26, major: Physics
## #2, name: James, age: 25, major: Economics
```

在前面的代码中，`#%d, name: %s, age: %d, major: %s` 是格式化模板，其中 `%d` 和 `%s` 是占位符，用于表示字符串中要出现的输入参数。`sprintf()` 函数特别易于使用，因为它防止模板字符串被拆分，并且每个要替换的部分都指定为函数参数。实际上，这个函数使用的是在 [`en.wikipedia.org/wiki/Printf_format_string`](https://en.wikipedia.org/wiki/Printf_format_string) 中详细描述的 C 风格格式化规则。

在前面的例子中，`%s` 表示字符串，`%d` 表示数字（整数）。此外，`sprintf()` 在使用 `%f` 格式化数值时也非常灵活。例如，`%.1f` 表示将数字四舍五入到 0.1：

```py
sprintf("The length of the line is approximately %.1fmm", 12.295)
## [1] "The length of the line is approximately 12.3mm"
```

实际上，存在不同类型值的格式化语法。以下表格显示了最常用的语法：

| **格式** | **输出** |
| --- | --- |
| `sprintf("%s", "A")` | `A` |
| `sprintf("%d", 10)` | `10` |
| `sprintf("%04d", 10)` | `0010` |
| `sprintf("%f", pi)` | `3.141593` |
| `sprintf("%.2f", pi)` | `3.14` |
| `sprintf("%1.0f", pi)` | `3` |
| `sprintf("%8.2f", pi)` | `3.14` |
| `sprintf("%08.2f", pi)` | `00003.14` |
| `sprintf("%+f", pi)` | `+3.141593` |
| `sprintf("%e", pi)` | `3.141593e+00` |
| `sprintf("%E", pi)` | `3.141593E+00` |

### 注意

官方文档（[`stat.ethz.ch/R-manual/R-devel/library/base/html/sprintf.html`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/sprintf.html)）提供了对支持格式的完整描述。

注意，格式文本中的 `%` 是一个特殊字符，将被解释为占位符的初始字符。如果我们真的想在字符串中表示 `%`，为了避免格式化解释，我们需要使用 `%%` 来表示字面量 `%`。以下代码是一个示例：

```py
sprintf("The ratio is %d%%", 10)
## [1] "The ratio is 10%"
```

### 在 R 中使用 Python 字符串函数

`sprintf()` 函数功能强大，但并非所有用例都完美。例如，如果模板中某些部分需要多次出现，您将需要多次编写相同的参数。这通常会使代码更加冗余，并且修改起来有些困难：

```py
sprintf("%s, %d years old, majors in %s and loves %s.", "James", 25, "Physics", "Physics")
## [1] "James, 25 years old, majors in Physics and loves Physics."
```

表示占位符还有其他方法。`pystr` 包提供了 `pystr_format()` 函数，用于使用数字或命名的占位符以 Python 格式化风格格式化字符串。前面的例子可以用这个函数以两种方式重写：

一种方法是使用数字占位符：

```py
# install.packages("pystr")
library(pystr)
pystr_format("{1}, {2} years old, majors in {3} and loves {3}.", "James", 25, "Physics", "Physics")
## [1] "James, 25 years old, majors in Physics and loves Physics."
```

另一种方法是使用命名的占位符：

```py
pystr_format("{name}, {age} years old, majors in {major} and loves {major}.", 
name = "James", age = 25, major = "Physics")
## [1] "James, 25 years old, majors in Physics and loves Physics."
```

在这两种情况下，不需要重复任何参数，输入出现的位置可以轻松地移动到模板字符串的其他地方。

# 日期/时间格式化

在数据分析中，经常会遇到日期和时间数据类型。也许，与日期最相关的最简单函数是 `Sys.Date()`，它返回当前日期，以及 `Sys.time()`，它返回当前时间。

当书籍正在渲染时，日期的打印方式如下：

```py
Sys.Date() ## [1] "2016-02-26"
```

时间如下：

```py
Sys.time() ## [1] "2016-02-26 22:12:25 CST"
```

从输出来看，日期和时间看起来像字符向量，但实际上它们不是：

```py
current_date <- Sys.Date()
as.numeric(current_date)
## [1] 16857
current_time <- Sys.time()
as.numeric(current_time)
## [1] 1456495945
```

它们本质上是以原点为基准的数值，并且具有执行日期/时间计算的特殊方法。对于一个日期，它的数值表示自 1970-01-01 之后的总天数。对于一个时间，它的数值表示自 1970-01-01 00:00.00 UTC 之后的总秒数。

## 将文本解析为日期/时间

我们可以创建一个相对于自定义原点的日期：

```py
as.Date(1000, "1970-01-01")
## [1] "1972-09-27"
```

然而，在更多情况下，我们从一个标准的文本表示形式创建日期和时间：

```py
my_date <- as.Date("2016-02-10") 
my_date
## [1] "2016-02-10"
```

但如果我们可以在字符串中代表时间，如 2016-02-10，那么为什么我们需要创建一个像之前那样创建的 `Date` 对象呢？这是因为日期具有更多功能：我们可以用它们进行日期计算。假设我们有一个日期对象，我们可以添加或减去一定数量的天数来得到一个新的日期：

```py
my_date + 3
## [1] "2016-02-13"
my_date + 80
## [1] "2016-04-30"
my_date - 65
## [1] "2015-12-07"
```

我们可以直接从另一个日期中减去一个日期，以得到两个日期之间天数差：

```py
date1 <- as.Date("2014-09-28") 
date2 <- as.Date("2015-10-20") 
date2 - date1
## Time difference of 387 days
```

`date2 - date1` 的输出看起来像一条消息，但实际上是一个数值。我们可以使用 `as.numeric()` 来使其更明确：

```py
as.numeric(date2 - date1)
## [1] 387
```

时间类似，但没有名为 `as.Time()` 的函数。要从文本表示形式创建日期时间，我们可以使用 `as.POSIXct()` 或 `as.POSIXlt()`。这两个函数是 POSIX 标准下日期/时间对象的两种不同实现。在以下示例中，我们使用 `as.POSIXlt` 来创建日期/时间对象：

```py
my_time <- as.POSIXlt("2016-02-10 10:25:31") 
my_time
## [1] "2016-02-10 10:25:31 CST"
```

这种类型的对象还定义了 `+` 和 `-` 用于简单的时计算。与日期对象不同，它以秒为单位而不是以天为单位：

```py
my_time + 10
## [1] "2016-02-10 10:25:41 CST"
my_time + 12345
## [1] "2016-02-10 13:51:16 CST"
my_time - 1234567
## [1] "2016-01-27 03:29:24 CST"
```

在数据中给定日期或时间的字符串表示形式时，我们必须将其转换为日期或日期/时间对象，这样我们就可以进行计算。然而，在原始数据中，我们得到的内容并不总是 `as.Date()` 或 `as.POSIXlt()` 可以直接识别的格式。在这种情况下，我们需要使用一组特殊字母作为占位符来表示日期或时间的某些部分，就像我们使用 `sprintf()` 一样。

例如，对于输入 `2015.07.25`，如果没有提供格式字符串，`as.Date()` 将产生错误：

```py
as.Date("2015.07.25")
## Error in charToDate(x): character string is not in a standard unambiguous format
```

我们可以使用格式字符串作为模板来告诉 `as.Date()` 如何将字符串解析为日期：

```py
as.Date("2015.07.25", format = "%Y.%m.%d")
## [1] "2015-07-25"
```

类似地，对于非标准的日期/时间字符串，我们还需要指定一个模板字符串来告诉`as.POSIXlt()`如何处理它：

```py
as.POSIXlt("7/25/2015 09:30:25", format = "%m/%d/%Y %H:%M:%S")
## [1] "2015-07-25 09:30:25 CST"
```

将字符串转换为日期/时间的另一种（更直接）的函数是`strptime()`：

```py
strptime("7/25/2015 09:30:25", "%m/%d/%Y %H:%M:%S")
## [1] "2015-07-25 09:30:25 CST"
```

实际上，`as.POSIXlt()`只是字符输入的`strptime()`的一个包装器，但`strptime()`始终要求你提供格式字符串，而`as.POSIXlt()`在没有提供模板的情况下适用于标准格式。

就像数值向量一样，日期和日期/时间也是向量。你可以向`as.Date()`提供一个字符向量，并得到一个日期向量：

```py
as.Date(c("2015-05-01", "2016-02-12"))
## [1] "2015-05-01" "2016-02-12"
```

数学运算也是向量化的。在以下代码中，我们将一些连续的整数添加到日期上，并得到预期的连续日期：

```py
as.Date("2015-01-01") + 0:2
## [1] "2015-01-01" "2015-01-02" "2015-01-03"
```

同样的功能也适用于日期/时间对象：

```py
strptime("7/25/2015 09:30:25", "%m/%d/%Y %H:%M:%S") + 1:3
## [1] "2015-07-25 09:30:26 CST" "2015-07-25 09:30:27 CST" ## [3] "2015-07-25 09:30:28 CST"
```

有时，数据使用日期和时间的整数表示。这使得解析日期和时间变得更加复杂。例如，为了解析`20150610`，我们将运行以下代码：

```py
as.Date("20150610", format = "%Y%m%d")
## [1] "2015-06-10"
```

为了解析`20150610093215`，我们可以指定模板来描述这种格式：

```py
strptime("20150610093215", "%Y%m%d%H%M%S")
## [1] "2015-06-10 09:32:15 CST"
```

一个更复杂一点的例子是解析以下数据框中的日期/时间：

```py
datetimes <- data.frame(
date = c(20150601, 20150603), 
time = c(92325, 150621))
```

如果我们在`datetimes`的列上使用`paste0()`，并直接使用之前示例中使用的模板调用`strptime()`，我们将得到一个缺失值，这表明第一个元素与格式不一致：

```py
dt_text <- paste0(datetimes$date, datetimes$time)dt_text
## [1] "2015060192325" "20150603150621"
strptime(dt_text, "%Y%m%d%H%M%S")
## [1] NA "2015-06-03 15:06:21 CST"
```

问题出在`92325`上，它应该是`092325`。我们需要使用`sprintf()`来确保在必要时存在前导零：

```py
dt_text2 <- paste0(datetimes$date, sprintf("%06d", datetimes$time))dt_text2
## [1] "20150601092325" "20150603150621"
strptime(dt_text2, "%Y%m%d%H%M%S")
## [1] "2015-06-01 09:23:25 CST" "2015-06-03 15:06:21 CST"
```

最后，转换工作如预期的那样进行。

## 格式化日期/时间为字符串

在上一节中，你学习了如何将字符串转换为日期和日期/时间对象。在本节中，你将学习相反的操作：根据特定的模板将日期和日期/时间对象转换回字符串。

一旦创建了一个日期对象，每次我们打印它时，它总是以标准格式表示：

```py
my_date
## [1] "2016-02-10"
```

我们可以使用`as.character()`将日期转换为标准表示：

```py
date_text <- as.character(my_date) 
date_text
## [1] "2016-02-10"
```

从输出来看，`my_date`看起来相同，但现在字符串只是一个纯文本，不再支持日期计算：

```py
date_text + 1
## Error in date_text + 1: non-numeric argument to binary operator
```

有时，我们需要以非标准的方式格式化日期：

```py
as.character(my_date, format = "%Y.%m.%d")
## [1] "2016.02.10"
```

实际上，`as.character()`在幕后直接调用`format()`。我们将得到完全相同的结果，这在大多数情况下是推荐的：

```py
format(my_date, "%Y.%m.%d")
## [1] "2016.02.10"
```

同样，这也适用于日期/时间对象。我们可以进一步自定义模板，以包含除了占位符之外的其他文本：

```py
my_time
## [1] "2016-02-10 10:25:31 CST"
format(my_time, "date: %Y-%m-%d, time: %H:%M:%S")
## [1] "date: 2016-02-10, time: 10:25:31"
```

### 注意

格式占位符远不止我们提到的那么多。通过输入`?strptime`来阅读文档，以获取详细信息。

有许多包可以使处理日期和时间变得更容易。我推荐使用`lubridate`包（[`cran.r-project.org/web/packages/lubridate`](https://cran.r-project.org/web/packages/lubridate)），因为它提供了几乎所有你需要用来处理日期和时间对象的函数。

在前面的章节中，你学习了处理字符串和日期/时间对象的一些基本函数。这些函数很有用，但与正则表达式相比，它们的灵活性要小得多。你将在下一节学习这种非常强大的技术。

# 使用正则表达式

对于研究，你可能需要从开放获取网站或需要认证的数据库下载数据。这些数据源提供的数据格式多种多样，而且大部分提供的数据很可能组织得很好。例如，许多经济和金融数据库提供 CSV 格式的数据，这是一种广泛支持的文本格式，用于表示表格数据。典型的 CSV 格式如下所示：

```py
id,name,score 
1,A,20 
2,B,30 
3,C,25
```

在 R 中，使用`read.csv()`将 CSV 文件导入为具有正确标题和数据类型的 data frame 非常方便，因为这种格式是 data frame 的自然表示形式。

然而，并非所有数据文件都组织得很好，处理组织不良的数据非常费力。内置函数如`read.table()`和`read.csv()`在许多情况下都有效，但它们可能对这种无格式数据根本无济于事。

例如，如果你需要分析如下所示以 CSV 格式组织的原始数据（`messages.txt`），在调用`read.csv()`时你最好要小心：

```py
2014-02-01,09:20:25,James,Ken,Hey, Ken! 
2014-02-01,09:20:29,Ken,James,Hey, how are you? 
2014-02-01,09:20:41,James,Ken, I'm ok, what about you? 
2014-02-01,09:21:03,Ken,James,I'm feeling excited! 
2014-02-01,09:21:26,James,Ken,What happens?
```

假设你想要以以下格式将此文件导入为 data frame，该格式组织得很好：

```py
      Date      Time     Sender   Receiver   Message 
1  2014-02-01  09:20:25  James    Ken        Hey, Ken! 
2  2014-02-01  09:20:29  Ken      James      Hey, how are you? 
3  2014-02-01  09:20:41  James    Ken        I'm ok, what about you? 
4  2014-02-01  09:21:03  Ken      James      I'm feeling excited! 
5  2014-02-01  09:21:26  James    Ken        What happens?
```

然而，如果你盲目地调用`read.csv()`，你会发现它并没有正确工作。这个数据集在消息列中有些特殊。存在额外的逗号，这些逗号会被错误地解释为 CSV 文件中的分隔符。以下是翻译自原始文本的数据框：

```py
read.csv("data/messages.txt", header = FALSE)
## V1V2V3V4V5V6 
## 1 2014-02-01 09:20:25 James Ken Hey Ken! 
## 2 2014-02-01 09:20:29 Ken James Hey how are you? 
## 3 2014-02-01 09:20:41 James Ken I'm ok what about you?
## 4 2014-02-01 09:21:03 Ken James I'm feeling excited! 
## 5 2014-02-01 09:21:26 James Ken What happens?
```

解决这个问题有多种方法。你可能考虑对每一行使用`strsplit()`，手动取出前几个元素，并将其他部分粘贴到每一行分割成多个部分。但其中最简单、最稳健的方法是使用所谓的正则表达式（[`en.wikipedia.org/wiki/Regular_expression`](https://en.wikipedia.org/wiki/Regular_expression)）。如果你对术语感到陌生，不要担心。它的用法非常简单：描述匹配文本的模式，并从该文本中提取所需的部分。

在我们应用技术之前，我们需要一些基本知识。最好的激励自己的方式是看看一个更简单的问题，并考虑解决该问题需要什么。

假设我们正在处理以下文本（`fruits.txt`），它描述了一些水果的数量或状态：

```py
apple: 20 
orange: missing 
banana: 30 
pear: sent to Jerry 
watermelon: 2 
blueberry: 12 
strawberry: sent to James
```

现在，我们想要挑选出带有数字而不是状态信息的所有水果。虽然我们可以轻松地通过视觉完成这项任务，但对于计算机来说并不那么容易。如果行数超过两千行，使用适当的技术处理对于计算机来说可能很容易，而相比之下，对于人类来说则可能很困难、耗时且容易出错。

我们首先应该想到的是，我们需要区分带数字和不带数字的水果。一般来说，我们需要区分匹配特定模式和不匹配模式的文本。在这里，正则表达式无疑是处理这个问题的正确技术。

正则表达式通过两个步骤解决问题：第一步是找到一个匹配文本的模式，第二步是分组模式以提取所需的信息。

## 寻找字符串模式

为了解决问题，我们的计算机不需要理解水果实际上是什么。我们只需要找出一个描述我们想要的模式的规律。字面上，我们想要获取所有以下列单词、分号和空格开始的行，并以整数结束而不是单词或其他符号。

正则表达式提供了一套符号来表示模式。前面的模式可以用 `^\w+:\s\d+$` 来描述，其中元符号用于表示符号类：

+   `^`：这个符号用于行的开头

+   `\w`：这个符号代表一个单词字符

+   `\s`：这个符号是一个空格字符

+   `\d`：这个符号是一个数字字符

+   `$`：这个符号用于行的末尾

此外，`\w+` 表示一个或多个单词字符，`:` 是我们期望在单词后面看到的符号，而 `\d+` 表示一个或多个数字字符。看，这个模式如此神奇，它代表了所有我们想要的案例，并排除了所有我们不想要的案例。

更具体地说，这个模式匹配如 `abc: 123` 这样的行，但不包括其他行。为了在 R 中挑选出所需的案例，我们使用 `grep()` 来获取哪些字符串匹配该模式：

```py
fruits <- readLines("data/fruits.txt") fruits
## [1] "apple: 20" "orange: missing" 
## [3] "banana: 30" "pear: sent to Jerry" 
## [5] "watermelon: 2" "blueberry: 12" 
## [7] "strawberry: sent to James"
matches <- grep("^\\w+:\\s\\d+$", fruits) 
matches
## [1] 1 3 5 6
```

注意，在 R 中，`\` 应该写成 `\\` 以避免转义。然后，我们可以通过 `matches` 过滤 `fruits`：

```py
fruits[matches]
## [1] "apple: 20" "banana: 30" "watermelon: 2" "blueberry: 12"
```

现在，我们已经成功地区分了所需的行和不需要的行。匹配模式的行被选中，不匹配模式的行被省略。

注意，我们指定了一个以 `^` 开头以 `$` 结尾的模式，因为我们不希望进行部分匹配。实际上，正则表达式默认执行部分匹配，即如果字符串的任何部分匹配模式，整个字符串就被认为是匹配模式的。例如，以下代码尝试分别找出哪些字符串匹配两个模式：

```py
grep("\\d", c("abc", "a12", "123", "1"))
## [1] 2 3 4
grep("^\\d$", c("abc", "a12", "123", "1"))
## [1] 4
```

第一个模式匹配包含任何数字的字符串（部分匹配），而第二个模式使用 `^` 和 `$` 匹配只有单个数字的字符串。

一旦模式正确工作，我们就进行下一步：使用组来提取数据。

## 使用组提取数据

在模式字符串中，我们可以用括号标记出我们想要从文本中提取的部分。在这个问题中，我们可以将模式修改为 `(\w+):\s(\d+)`，其中标记了两组：一组是匹配 `\w+` 的水果名称，另一组是匹配 `\d+` 的水果数量。

现在，我们可以使用这个修改后的模式来提取我们想要的信息。虽然使用 R 的内置函数做这个工作是完全可能的，但我强烈建议使用`stringr`包中的函数。这个包使得使用正则表达式变得容易得多。我们使用修改后的模式并带有组的`str_match()`：

```py
library(stringr)
matches <- str_match(fruits, "^(\\w+):\\s(\\d+)$")
matches
##      [,1]            [,2]         [,3]
## [1,] "apple: 20"     "apple"      "20"
## [2,] NA              NA           NA  
## [3,] "banana: 30"    "banana"     "30"
## [4,] NA              NA           NA  
## [5,] "watermelon: 2" "watermelon" "2" 
## [6,] "blueberry: 12" "blueberry"  "12"
## [7,] NA              NA           NA
```

这次匹配的是一个有多列的矩阵。括号内的组是从文本中提取出来的，并放置在第二列和第三列。现在，我们可以轻松地将这个字符矩阵转换成带有正确标题和数据类型的 data frame：

```py
# transform to data frame
fruits_df <- data.frame(na.omit(matches[, -1]), stringsAsFactors =FALSE)
# add a header
colnames(fruits_df) <- c("fruit","quantity")
# convert type of quantity from character to integer
fruits_df$quantity <- as.integer(fruits_df$quantity)
```

现在，`fruits_df`是一个带有正确标题和数据类型的 data frame：

```py
fruits_df
##    fruit  quantity
## 1  apple      20
## 2  banana     30
## 3  watermelon  2
## 4  blueberry  12
```

如果你不确定前面代码的中间结果，你可以逐行运行代码，看看每一步发生了什么。最后，这个问题完全可以用正则表达式解决。

从前面的例子中，我们看到正则表达式的魔力只是一组标识符，用于表示不同类型的字符和符号。除了我们提到的元符号外，以下也是很有用的：

+   `[0-9]`：这个符号代表从 0 到 9 的单个整数

+   `[a-z]`：这个符号代表从 a 到 z 的单个小写字母

+   `[A-Z]`：这个符号代表从 A 到 Z 的单个大写字母

+   `.`：这个符号代表任何单个符号

+   `*`：这个符号代表一个模式，可能出现零次、一次或多次

+   `+`：这是一个模式，出现一次或多次

+   `{n}`：这是一个出现`n`次的模式

+   `{m,n}`：这是一个至少出现`m`次且最多出现`n`次的模式

使用这些元符号，我们可以轻松地检查或过滤字符串数据。例如，假设我们有一些来自两个国家的电话号码混合在一起。如果一个国家的电话号码模式与另一个国家不同，正则表达式可以帮助将它们分成两类：

```py
telephone <- readLines("data/telephone.txt") 
telephone
## [1] "123-23451" "1225-3123" "121-45672" "1332-1231" "1212-3212" "123456789"
```

注意，数据中有一个例外。数字中间没有`-`。对于非例外情况，应该很容易找出两种电话号码的模式：

```py
telephone[grep("^\\d{3}-\\d{5}$", telephone)]
## [1] "123-23451" "121-45672"
telephone[grep("^\\d{4}-\\d{4}$", telephone)]
## [1] "1225-3123" "1332-1231" "1212-3212"
```

要找出例外情况，`grepl()`更有用，因为它返回一个逻辑向量，指示每个元素是否与给定的模式匹配。因此，我们可以使用这个函数来选择所有不匹配给定模式的记录：

```py
telephone[!grepl("^\\d{3}-\\d{5}$", telephone) & !grepl("^\\d{4}-\\d{4}$", telephone)]
## [1] "123456789"
```

上述代码基本上表示所有不匹配两种模式的记录都被认为是例外。想象一下，我们有一百万条记录要检查。例外情况可能以任何格式存在，因此使用这种方法（排除所有有效记录以找出无效记录）更为稳健。

## 以可定制的方式读取数据

现在，让我们回到本节开头遇到的问题。这个程序与水果示例中的程序完全相同：找到模式并分组。

首先，让我们看看原始数据中的一条典型行：

```py
2014-02-01,09:20:29,Ken,James,Hey, how are you?
```

显然，所有行都是基于相同的格式，即日期、时间、发件人、收件人和消息由逗号分隔。唯一特殊的是，消息中可能包含逗号，但我们不希望我们的代码将其解释为分隔符。

注意，正则表达式与前面的例子一样完美地适用于这个目的。要表示一个或多个遵循相同模式的符号，只需在符号标识符后放置一个加号（`+`）。例如，`\d+` 表示由一个或多个介于 "0" 和 "9" 之间的数字字符组成的字符串。例如，“1”，“23”，和“456”都匹配这个模式，而“word”则不匹配。也存在某些模式可能或可能不出现的情况。然后，我们需要在符号标识符后放置一个 `*` 来标记这个特定模式可能出现一次或多次，或者可能不出现，以便匹配广泛的文本。

现在，让我们回到我们的问题。我们需要识别典型行的足够通用的模式。以下是我们应该解决的分组模式：

```py
(\d+-\d+-\d+),(\d+:\d+:\d+),(\w+),(\w+),\s*(.+)
```

现在，我们需要以与我们在水果示例中使用 `readLines()` 的相同方式导入原始文本：

```py
messages <- readLines("data/messages.txt")
```

然后，我们需要找出代表我们想要从文本中提取的文本和信息模式的模式：

```py
pattern <- "^(\\d+-\\d+-\\d+),(\\d+:\\d+:\\d+),(\\w+),(\\w+),\\s*(.+)$"
matches <- str_match(messages, pattern)
messages_df <- data.frame(matches[, -1]) colnames(messages_df) <- c("Date", "Time", "Sender", "Receiver", "Message")
```

这里的模式看起来像某种秘密代码。别担心，这正是正则表达式的工作方式，如果你看过了前面的例子，现在应该能理解一些了。

正则表达式工作得非常完美。`messages_df` 文件看起来像以下结构：

```py
messages_df
##      Date        Time    Sender   Receiver    Message 
## 1 2014-02-01   09:20:25  James    Ken         Hey, Ken! 
## 2 2014-02-01   09:20:29  Ken      James       Hey, how are you? 
## 3 2014-02-01   09:20:41  James    Ken         I'm ok, what about you? 
## 4 2014-02-01   09:21:03  Ken      James       I'm feeling excited! 
## 5 2014-02-01   09:21:26  James    Ken         What happens?
```

我们使用的模式可以比作一把钥匙。任何正则表达式应用的难点在于找到这把钥匙。一旦我们找到了它，我们就能打开门，并从混乱的文本中提取我们想要的信息。一般来说，找到这把钥匙的难度很大程度上取决于正例和反例之间的差异。如果差异非常明显，几个符号就能解决问题。如果差异微妙且涉及许多特殊情况，就像大多数现实世界问题一样，你需要更多的经验，更深入的思考，以及许多尝试和错误来找到解决方案。

通过前面提到的激励性例子，你现在应该已经掌握了正则表达式的概念。你不需要理解它是如何内部工作的，但熟悉相关的函数非常有用，无论是内置的还是某些包提供的。

如果你想了解更多，RegexOne ([`regexone.com/`](http://regexone.com/)) 是一个非常好的地方，可以以交互式的方式学习基础知识。要了解更多具体的示例和完整的标识符集合，这个网站 ([`www.regular-expressions.info/`](http://www.regular-expressions.info/)) 是一个很好的参考。为了找到解决你问题的良好模式，你可以访问 RegExr ([`www.regexr.com/`](http://www.regexr.com/)) 以在线交互式地测试你的模式。

# 摘要

在本章中，你学习了关于操作字符向量以及在不同日期/时间对象及其字符串表示之间进行转换的许多内置函数。你还了解了正则表达式的基本概念，这是一个非常强大的工具，用于检查和过滤字符串数据，并从原始文本中提取信息。

通过在本章和前几章中构建的词汇，我们现在能够处理基本的数据结构。在下一章中，你将学习一些用于处理数据的工具和技术。我们将从读取和写入简单的数据文件开始，生成各种类型的图形，对简单数据集应用基本的统计分析和数据挖掘模型，以及使用数值方法来解决根求解和优化问题。
