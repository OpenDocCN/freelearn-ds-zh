

# 第二章：选择和赋值

在上一章中，我们学习了如何创建 `pd.Series` 和 `pd.DataFrame`，并且还了解了它们与 `pd.Index` 的关系。在掌握了 *构造函数* 的基础后，我们现在转向选择和赋值这一关键过程。选择，也称为 *索引*，被认为是 *getter*；即它用于从 pandas 对象中检索值。相比之下，赋值是 *setter*，用于更新值。

本章的内容从教你如何从 `pd.Series` 和 `pd.DataFrame` 对象中检索值开始，逐步增加复杂性。我们最终将介绍 `pd.MultiIndex`，它可以用于层次选择数据，最后我们会介绍赋值运算符。pandas API 在选择和赋值方面非常注重复用相同的方法，这使得你可以以非常表达的方式与数据交互。

到本章结束时，你将熟练掌握如何高效地从 pandas 对象中检索数据并更新其中的值。本章我们将涵盖以下内容：

+   从 Series 中的基本选择

+   从 DataFrame 中的基本选择

+   从 Series 中的按位置选择

+   DataFrame 的位置选择

+   基于标签的选择从 Series 中选择

+   从 DataFrame 中基于标签的选择

+   混合基于位置和基于标签的选择

+   DataFrame.filter

+   按数据类型选择

+   通过布尔数组进行选择/过滤

+   使用多级索引选择 – 单级别

+   使用多级索引选择 – 多级别

+   使用多级索引选择 – DataFrame

+   使用 .loc 和 .iloc 进行项赋值

+   DataFrame 列赋值

# 从 Series 中的基本选择

从 `pd.Series` 中选择涉及通过其位置或标签访问元素。这类似于通过索引访问列表中的元素或通过键访问字典中的元素。`pd.Series` 对象的多功能性使得数据检索直观且简单，是数据操作的基础工具。

`pd.Series` 被认为是 Python 中的 *容器*，就像内建的 `list`、`tuple` 和 `dict` 对象。因此，对于简单的选择操作，用户首先使用的是 Python 索引运算符，即 `[]` 语法。

## 如何做到这一点

为了介绍选择的基础知识，我们从一个非常简单的 `pd.Series` 开始：

```py
`ser = pd.Series(list("abc") * 3) ser` 
```

```py
`0    a 1    b 2    c 3    a 4    b 5    c 6    a 7    b 8    c dtype: object` 
```

在 Python 中，你已经发现 `[]` 运算符可以用来从 *容器* 中选择元素；即，`some_dictionary[0]` 会返回与键为 0 相关联的值。对于 `pd.Series`，基本选择的行为类似：

```py
`ser[3]` 
```

```py
`a` 
```

使用表达式`ser[3]`时，pandas 会尝试在`pd.Series`的索引中找到标签 3，并且假设只有一个匹配项，它会返回与该标签相关联的值。

如果你不想从`pd.Series`中选择相关值，而是希望返回一个`pd.Series`，这样可以保持标签 3 与数据元素“a”关联。在 pandas 中，你可以通过提供包含单个元素的列表参数来实现这一点：

```py
`ser[[3]]` 
```

```py
`3    a dtype: object` 
```

在扩展列表参数的用法时，如果列表包含多个元素，你可以从`pd.Series`中选择多个值：

```py
`ser[[0, 2]]` 
```

```py
`0    a 2    c dtype: object` 
```

假设你使用默认的索引，你可以使用类似于 Python 列表切片的切片参数。例如，要获取一个`pd.Series`中位置 3 之前的元素（但不包括位置 3），可以使用：

```py
`ser[:3]` 
```

```py
`0    a 1    b 2    c dtype: object` 
```

负数切片索引对 pandas 来说不是问题。以下代码将选择`pd.Series`的最后四个元素：

```py
`ser[-4:]` 
```

```py
`5    c 6    a 7    b 8    c dtype: object` 
```

你甚至可以提供带有*start*和*stop*参数的切片。以下代码将检索`pd.Series`中从位置 2 开始，并直到（但不包括）位置 6 的所有元素：

```py
`ser[2:6]` 
```

```py
`2    c 3    a 4    b 5    c dtype: object` 
```

这个关于切片的最终示例使用了*start*、*stop*和*step*参数，从位置 1 开始，抓取每隔一个元素，直到遇到位置 8：

```py
`ser[1:8:3]` 
```

```py
`1    b 4    b 7    b dtype: object` 
```

当提供你自己的`pd.Index`值时，选择仍然有效。让我们创建一个带有字符串索引标签的小型`pd.Series`来说明：

```py
`ser = pd.Series(range(3), index=["Jack", "Jill", "Jayne"]) ser` 
```

```py
`Jack     0 Jill     1 Jayne    2 dtype: int64` 
```

通过`ser["Jill"]`选择时，将扫描索引查找字符串`Jill`并返回相应的元素：

```py
`ser["Jill"]` 
```

```py
`1` 
```

再次提供单元素列表参数将确保你收到一个`pd.Series`作为返回值，而不是单一值：

```py
`ser[["Jill"]]` 
```

```py
`Jill    1 dtype: int64` 
```

## 还有更多…

使用`[]`运算符时，一个常见的陷阱是错误地认为使用整数参数的选择方式与从 Python 列表中选择的方式相同。这*仅*在使用默认的`pd.Index`时才成立，默认的`pd.Index`是自动编号的，从 0 开始（这在技术上称为`pd.RangeIndex`）。

在不使用`pd.RangeIndex`时，必须特别注意行为。为了说明这一点，让我们从一个小型`pd.Series`开始，它仍然使用整数作为`pd.Index`，但没有使用从 0 开始的自动递增序列：

```py
`ser = pd.Series(list("abc"), index=[2, 42, 21]) ser` 
```

```py
`2     a 42    b 21    c dtype: object` 
```

需要注意的是，整数参数是按*标签*选择，而不是按*位置*选择；也就是说，以下代码将返回与标签 2 关联的值，而不是位置 2 的值：

```py
`ser[2]` 
```

```py
`a` 
```

尽管整数参数按标签匹配，而不是按位置匹配，但切片仍然按位置工作。以下示例在遇到数字 2 时不会停止，而是返回前两个元素：

```py
`ser[:2]` 
```

```py
`2     a 42    b dtype: object` 
```

用户还应该了解在处理非唯一的`pd.Index`时的选择行为。让我们创建一个小型`pd.Series`，其中数字 1 在行索引中出现两次：

```py
`ser = pd.Series(["apple", "banana", "orange"], index=[0, 1, 1]) ser` 
```

```py
`0     apple 1    banana 1    orange dtype: object` 
```

对于这个`pd.Series`，尝试选择数字 1 将*不会*返回单一值，而是返回另一个`pd.Series`：

```py
`ser[1]` 
```

```py
`1    banana 1    orange dtype: object` 
```

由于使用默认的`pd.RangeIndex`时，像`ser[1]`这样的选择可以被认为是通过位置或标签进行互换的，但实际上，在使用其他`pd.Index`类型时是通过标签进行选择的，这可能是用户程序中细微 bug 的来源。许多用户可能*认为*他们在选择第*n*个元素，但当数据发生变化时，这个假设会被打破。为了消除通过整数参数选择*标签*或*位置*的歧义，**强烈建议**利用本章稍后介绍的`.loc`和`.iloc`方法。

# 从 DataFrame 中进行基本选择

使用`[]`运算符与`pd.DataFrame`时，简单的选择通常涉及从*列索引*中选择数据，而不是从*行索引*中选择数据。这个区别对于有效的数据操作和分析至关重要。`pd.DataFrame`中的列可以通过它们的标签进行访问，这使得处理来自更大`pd.DataFrame`结构中的`pd.Series`的命名数据变得容易。

理解这种选择行为的基本差异是充分利用 pandas 中`pd.DataFrame`功能的关键。通过利用`[]`运算符，你可以高效地访问和操作特定的列数据，为更高级的操作和分析奠定基础。

## 如何实现

让我们从创建一个简单的 3x3`pd.DataFrame`开始。`pd.DataFrame`的值并不重要，但我们故意提供自己的列标签，而不是让 pandas 为我们创建自动编号的列索引：

```py
`df = pd.DataFrame(np.arange(9).reshape(3, -1), columns=["a", "b", "c"]) df` 
```

```py
 `a     b     c 0    0     1     2 1    3     4     5 2    6     7     8` 
```

要选择单个列，使用带有标量参数的`[]`运算符：

```py
`df["a"]` 
```

```py
`0    0 1    3 2    6 Name: a, dtype: int64` 
```

要选择单个列，但仍然返回`pd.DataFrame`而不是`pd.Series`，请传递一个单元素列表：

```py
`df[["a"]]` 
```

```py
 `a 0    0 1    3 2    6` 
```

可以使用列表选择多个列：

```py
`df[["a", "b"]]` 
```

```py
 `a     b 0    0     1 1    3     4 2    6     7` 
```

在所有这些示例中，`[]`的参数是从列中选择的，但是提供切片参数会表现出不同的行为，实际上会从行中进行选择。请注意，以下示例选择了所有列和前两行数据，而不是反过来：

```py
`df[:2]` 
```

```py
 `a     b     c 0    0     1     2 1    3     4     5` 
```

## 还有更多…

当使用列表参数时，`[]`运算符使你可以灵活指定输出中列的顺序。这允许你根据需要定制`pd.DataFrame`。输出中列的顺序将完全匹配作为输入提供的标签顺序。例如：

```py
`df[["a", "b"]]` 
```

```py
 `a     b 0    0     1 1    3     4 2    6     7` 
```

将列表中的元素顺序交换作为`[]`运算符的参数时，将会交换结果`pd.DataFrame`中列的顺序：

```py
`df[["b", "a"]]` 
```

```py
 `b     a 0    1     0 1    4     3 2    7     6` 
```

这个功能在需要为展示目的重新排序列时特别有用，或者在准备导出到 CSV 或 Excel 格式时需要特定列顺序时（有关 pandas I/O 系统的更多信息，请参见*第四章*，*pandas I/O 系统*）。

# 基于位置的系列选择

如同在*DataFrame 的基本选择*一节中讨论的那样，使用`[]`作为选择机制并没有明确表达意图，有时甚至可能让人感到困惑。`ser[42]`选择的是与数字 42 匹配的*标签*，而不是`pd.Series`的第 42 行，这是新用户常见的错误，而当你开始尝试使用`[]`操作符从`pd.DataFrame`中选择两个维度时，这种模糊性可能会变得更加复杂。

为了明确表明你是在按*位置*选择而不是按*标签*选择，你应该使用`pd.Series.iloc`。

## 如何操作

让我们创建一个`pd.Series`，其索引使用的是整数标签，并且这些标签不唯一：

```py
`ser = pd.Series(["apple", "banana", "orange"], index=[0, 1, 1]) ser` 
```

```py
`0     apple 1    banana 1    orange dtype: object` 
```

要选择一个标量，你可以使用带有整数参数的`pd.Series.iloc`：

```py
`ser.iloc[1]` 
```

```py
`banana` 
```

按照我们之前看到的模式，将整数参数转换为包含单一元素的列表将返回一个`pd.Series`，而不是标量：

```py
`ser.iloc[[1]]` 
```

```py
`1    banana dtype: object` 
```

在列表参数中使用多个整数将按位置选择`pd.Series`的多个元素：

```py
`ser.iloc[[0, 2]]` 
```

```py
`0     apple 1    orange dtype: object` 
```

切片是表达你想选择的元素范围的自然方式，并且它们与`pd.Series.iloc`的参数非常匹配：

```py
`ser.iloc[:2]` 
```

```py
`0     apple 1    banana dtype: object` 
```

# 基于位置的 DataFrame 选择

与`pd.Series`类似，整数、整数列表和切片对象都是`DataFrame.iloc`的有效参数。然而，对于`pd.DataFrame`，需要两个参数。第一个参数负责选择*行*，第二个参数负责选择*列*。

在大多数使用场景中，用户在获取行时会选择基于位置的选择，而在获取列时会选择基于标签的选择。我们将在*基于标签的 DataFrame 选择*一节中讲解后者，并且在*混合位置选择和标签选择*一节中展示如何将两者结合使用。然而，当你的行索引使用默认的`pd.RangeIndex`并且列的顺序很重要时，本节中展示的技巧将非常有价值。

## 如何操作

让我们创建一个包含五行四列的`pd.DataFrame`：

```py
`df = pd.DataFrame(np.arange(20).reshape(5, -1), columns=list("abcd")) df` 
```

```py
 `a     b     c     d 0    0     1     2     3 1    4     5     6     7 2    8     9     10    11 3    12    13    14    15 4    16    17    18    19` 
```

将两个整数参数传递给`pd.DataFrame.iloc`将返回该行和列位置的标量：

```py
`df.iloc[2, 2]` 
```

```py
`10` 
```

在某些情况下，你可能不想选择特定轴上的单个值，而是希望选择该轴上的所有内容。一个空切片对象`:`可以让你做到这一点；例如，如果你想选择`pd.DataFrame`第一列的*所有*数据行，你可以使用：

```py
`df.iloc[:, 0]` 
```

```py
`0     0 1     4 2     8 3    12 4    16 Name: a, dtype: int64` 
```

翻转`pd.DataFrame.iloc`的参数顺序会改变行为。下面的代码不是选择第一列的所有行，而是选择所有列并且只选择第一行的数据：

```py
`df.iloc[0, :]` 
```

```py
`a    0 b    1 c    2 d    3 Name: 0, dtype: int64` 
```

因为前述示例仅返回数据的一个维度，它们隐含地尝试将`pd.DataFrame`的返回值“*挤压*”为`pd.Series`。根据本章中我们已经多次看到的模式，您可以通过为轴传递单元素列表参数来防止隐式降维，而不是空切片。例如，要选择第一列的所有行但仍然返回`pd.DataFrame`，您可以选择：

```py
`df.iloc[:, [0]]` 
```

```py
 `a 0    0 1    4 2    8 3    12 4    16` 
```

反转这些参数会给我们返回一个`pd.DataFrame`中的第一行和所有列：

```py
`df.iloc[[0], :]` 
```

```py
 `a    b    c    d 0    0    1    2    3` 
```

列表可以用来从行和列中选择多个元素。如果我们想要`pd.DataFrame`的第一行和第二行与最后一列和倒数第二列配对，您可以选择一个表达式如下：

```py
`df.iloc[[0, 1], [-1, -2]]` 
```

```py
 `d    c 0    3    2 1    7    6` 
```

## 还有更多……

空切片是`.iloc`的有效参数。`ser.iloc[:]`和`df.iloc[:, :]`都将返回每个轴上的所有内容，从本质上来说，给您一个对象的副本。

# 从 Series 中基于标签的选择

在 pandas 中，`pd.Series.loc`用于根据标签而不是位置进行选择。当您考虑您的`pd.Series`的`pd.Index`包含查找值时，此方法特别有用，这些值类似于 Python 字典中的键，而不是给出数据在`pd.Series`中的顺序或位置的重要性。

## 如何做到这一点

让我们创建一个`pd.Series`，其中我们使用整数标签作为行索引，这些标签也是非唯一的：

```py
`ser = pd.Series(["apple", "banana", "orange"], index=[0, 1, 1]) ser` 
```

```py
`0     apple 1    banana 1    orange dtype: object` 
```

`pd.Series.loc`将选择所有具有标签 1 的行：

```py
`ser.loc[1]` 
```

```py
`1    banana 1    orange dtype: object` 
```

当然，在 pandas 中，您并不局限于整数标签。让我们看看使用由字符串值组成的`pd.Index`的情况：

```py
`ser = pd.Series([2, 2, 4], index=["dog", "cat", "human"], name="num_legs") ser` 
```

```py
`dog      2 cat      2 human    4 Name: num_legs, dtype: int64` 
```

`pd.Series.loc`可以选择所有具有`"dog"`标签的行：

```py
`ser.loc["dog"]` 
```

```py
`2` 
```

要选择所有具有`"dog"`或`"cat"`标签的行：

```py
`ser.loc[["dog", "cat"]]` 
```

```py
`dog    2 cat    2 Name: num_legs, dtype: int64` 
```

最后，要选择直到包括标签`"cat"`的所有行：

```py
`ser.loc[:"cat"]` 
```

```py
`dog    2 cat    2 Name: num_legs, dtype: int64` 
```

## 还有更多……

使用`pd.Series.loc`进行基于标签的选择提供了强大的功能，用于访问和操作`pd.Series`中的数据。虽然这种方法可能看起来很简单，但它提供了重要的细微差别和行为，对于有效的数据处理是很重要的。

对于所有经验水平的 pandas 用户来说，一个非常常见的错误是忽视`pd.Series.loc`切片行为与标准 Python 和`pd.Series.iloc`情况下切片行为之间的差异。

要逐步进行这一点，让我们创建一个小 Python 列表和一个具有相同数据的`pd.Series`：

```py
`values = ["Jack", "Jill", "Jayne"] ser = pd.Series(values) ser` 
```

```py
`0     Jack 1     Jill 2    Jayne dtype: object` 
```

正如您已经看到的那样，与 Python 语言内置的列表和其他容器一样，切片返回值直到提供的位置，但不包括：

```py
`values[:2]` 
```

```py
`Jack    Jill` 
```

使用`pd.Series.iloc`进行切片与此行为相匹配，返回一个与 Python 列表具有相同长度和元素的`pd.Series`：

```py
`ser.iloc[:2]` 
```

```py
`0    Jack 1    Jill dtype: object` 
```

但是，使用`pd.Series.loc`进行切片实际上产生了不同的结果：

```py
`ser.loc[:2]` 
```

```py
`0     Jack 1     Jill 2    Jayne dtype: object` 
```

这里发生了什么？为了更好理解这一点，需要记住 `pd.Series.loc` 是通过标签进行匹配，而非位置。pandas 库会对每个 `pd.Series` 和其对应的 `pd.Index` 元素进行类似循环的操作，直到它在索引中找到值为 2 的元素。然而，pandas 无法保证 `pd.Index` 中只有一个值为 2 的元素，因此它必须继续搜索直到找到*其他的东西*。如果你尝试对一个重复索引标签为 2 的 `pd.Series` 进行相同的选择，你将看到这一点的实际表现：

```py
`repeats_2 = pd.Series(range(5), index=[0, 1, 2, 2, 0]) repeats_2.loc[:2]` 
```

```py
`0    0 1    1 2    2 2    3 dtype: int64` 
```

如果你预期行索引包含整数，可能会觉得这有些狡猾，但`pd.Series.loc`的主要用例是用于处理 `pd.Index`，其中位置/顺序不重要（对于此情况，可以使用 `pd.Series.iloc`）。以字符串标签作为一个更实际的例子，`pd.Series.loc` 的切片行为变得更加自然。以下代码可以基本理解为在请求 pandas 遍历 `pd.Series`，直到行索引中找到标签 `"xxx"`，并继续直到找到另一个标签：

```py
`ser = pd.Series(range(4), index=["zzz", "xxx", "xxx", "yyy"]) ser.loc[:"xxx"]` 
```

```py
`zzz    0 xxx    1 xxx    2 dtype: int64` 
```

在某些情况下，当你尝试使用 `pd.Series.loc` 切片，但索引标签没有确定的顺序时，pandas 将会抛出错误：

```py
`ser = pd.Series(range(4), index=["zzz", "xxx", "yyy", "xxx"]) ser.loc[:"xxx"]` 
```

```py
`KeyError: "Cannot get right slice bound for non-unique label: 'xxx'"` 
```

# 基于标签的 DataFrame 选择

正如我们在*基于位置的 DataFrame 选择*部分讨论过的，`pd.DataFrame` 最常见的用例是，在引用列时使用基于标签的选择，而在引用行时使用基于位置的选择。然而，这并不是绝对要求，pandas 允许你从行和列中使用基于标签的选择。

与其他数据分析工具相比，从 `pd.DataFrame` 的行中通过标签进行选择是 pandas 独有的优势。对于熟悉 SQL 的用户，SQL 并没有提供真正相当的功能；在 `SELECT` 子句中选择列非常容易，但只能通过 `WHERE` 子句过滤行。对于擅长使用 Microsoft Excel 的用户，你可以通过数据透视表创建具有行标签和列标签的二维结构，但你在该透视表内选择或引用数据的能力是有限的。

目前，我们将介绍如何为非常小的 `pd.DataFrame` 对象进行选择，以便熟悉语法。在*第八章*，*数据框重塑*中，我们将探索如何创建有意义的 `pd.DataFrame` 对象，其中行和列标签是重要的。结合本节介绍的知识，你将会意识到这种选择方式是 pandas 独有的，并且它如何帮助你以其他工具无法表达的方式探索数据。

## 如何进行选择

让我们创建一个 `pd.DataFrame`，其行列索引均由字符串组成：

```py
`df = pd.DataFrame([     [24, 180, "blue"],     [42, 166, "brown"],     [22, 160, "green"], ], columns=["age", "height_cm", "eye_color"], index=["Jack", "Jill", "Jayne"]) df` 
```

```py
 `age    height_cm    eye_color Jack    24     180          blue Jill    42     166          brown Jayne   22     160          green` 
```

`pd.DataFrame.loc` 可以通过行和列标签进行选择：

```py
`df.loc["Jayne", "eye_color"]` 
```

```py
`green` 
```

要选择来自标签为 `"age"` 列的所有行：

```py
`df.loc[:, "age"]` 
```

```py
`Jack     24 Jill     42 Jayne    22 Name: age, dtype: int64` 
```

要从标签为 `"Jack"` 的行中选择所有列：

```py
`df.loc["Jack", :]` 
```

```py
`age            24 height_cm     180 eye_color    blue Name: Jack, dtype: object` 
```

要从标签为 `"age"` 的列中选择所有行，并保持 `pd.DataFrame` 的形状：

```py
`df.loc[:, ["age"]]` 
```

```py
 `age Jack     24 Jill     42 Jayne    22` 
```

要从标签为 `"Jack"` 的行中选择所有列，并保持 `pd.DataFrame` 的形状：

```py
`df.loc[["Jack"], :]` 
```

```py
 `age   height_cm    eye_color Jack    24    180          blue` 
```

使用标签列表选择行和列：

```py
`df.loc[["Jack", "Jill"], ["age", "eye_color"]]` 
```

```py
 `age   eye_color Jack    24    blue Jill    42    brown` 
```

# 混合基于位置和标签的选择

由于 `pd.DataFrame.iloc` 用于基于位置的选择，而 `pd.DataFrame.loc` 用于基于标签的选择，当用户尝试在一个维度上按标签选择，另一个维度上按位置选择时，必须额外采取一步措施。如前所述，大多数构造的 `pd.DataFrame` 对象会非常重视用于列的标签，而对这些列的顺序关注较少。行的情况正好相反，因此能够有效地混合和匹配这两种风格是非常有价值的。

## 如何操作

让我们从一个 `pd.DataFrame` 开始，该数据框的行使用默认的自动编号 `pd.RangeIndex`，但列使用自定义的字符串标签：

```py
`df = pd.DataFrame([     [24, 180, "blue"],     [42, 166, "brown"],     [22, 160, "green"], ], columns=["age", "height_cm", "eye_color"]) df` 
```

```py
 `age   height_cm    eye_color 0    24    180          blue 1    42    166          brown 2    22    160          green` 
```

`pd.Index.get_indexer` 方法可以帮助我们将标签或标签列表转换为它们在 `pd.Index` 中对应的位置：

```py
`col_idxer = df.columns.get_indexer(["age", "eye_color"]) col_idxer` 
```

```py
`array([0, 2])` 
```

这随后可以作为参数传递给 `.iloc`，确保你在行和列上都使用基于位置的选择：

```py
`df.iloc[[0, 1], col_idxer]` 
```

```py
 `age    eye_color 0    24     blue 1    42     brown` 
```

## 还有更多……

你可以不使用 `pd.Index.get_indexer`，将这个表达式拆分成几个步骤，其中一个步骤进行基于索引的选择，另一个步骤进行基于标签的选择。如果你这样做，你最终会得到与上面相同的结果：

```py
`df[["age", "eye_color"]].iloc[[0, 1]]` 
```

```py
 `age    eye_color 0    24     blue 1    42     brown` 
```

有强有力的理由认为，这比使用 `pd.Index.get_indexer` 更具表达力，所有 pandas 用户的开发者都会同意这一点。那么，为什么还要使用 `pd.Index.get_indexer` 呢？

尽管这些在表面上看起来一样，但 pandas 计算结果的方式却有很大不同。为各种方法添加一些计时基准应该能突出这一点。尽管准确的数字会因你的机器而异，但可以比较本节中描述的惯用方法的计时输出：

```py
`import timeit def get_indexer_approach():   col_idxer = df.columns.get_indexer(["age", "eye_color"])   df.iloc[[0, 1], col_idxer] timeit.timeit(get_indexer_approach, number=10_000)` 
```

```py
`1.8184850879988517` 
```

使用分步方法先按标签选择，再按位置选择：

```py
`two_step_approach = lambda: df[["age", "eye_color"]].iloc[[0, 1]] timeit.timeit(two_step_approach, number=10_000` 
```

```py
`2.027099569000711` 
```

`pd.Index.get_indexer` 方法速度更快，并且应该能更好地扩展到更大的数据集。之所以如此，是因为 pandas 采用的是 *贪婪* 计算方式，或者更具体地说，它会按你说的来做。表达式 `df[["age", "eye_color"]].iloc[[0, 1]]` 首先运行 `df[["age", "eye_color"]]`，这会创建一个中间的 `pd.DataFrame`，然后 `.iloc[[0, 1]]` 被应用于这个数据框。相比之下，表达式 `df.iloc[[0, 1], col_idxer]` 一次性执行了标签和位置的选择，避免了创建任何中间的 `pd.DataFrame`。

与 pandas 采取的*急切执行*方法相对的方式通常被称为*延迟执行*。如果你以前使用过 SQL，后者就是一个很好的例子；你通常不会精确指示 SQL 引擎应该采取什么步骤来产生期望的结果。相反，你*声明*你希望结果是什么样的，然后将优化和执行查询的任务交给 SQL 数据库。

pandas 是否会支持延迟评估和优化？我认为会，因其有助于 pandas 扩展到更大的数据集，并减轻最终用户编写优化查询的负担。然而，这种功能目前还不存在，因此作为库的用户，你仍然需要了解你编写的代码是否会高效或低效地处理。

在决定是否值得尝试将基于位置/标签的选择合并为一步操作时，也值得考虑你数据分析的上下文，或者它们是否可以作为单独的步骤。在我们这个简单的示例中，`df.iloc[[0, 1], col_idxer]`与`df[["age", "eye_color"]].iloc[[0, 1]]`之间的运行时差异在整体上可能不值得关注，但如果你处理的是更大的数据集，并且受到性能瓶颈的限制，前一种方法可能是救命稻草。

# DataFrame.filter

`pd.DataFrame.filter`是一个专门的方法，允许你从`pd.DataFrame`的行或列中进行选择。

## 如何操作

让我们创建一个`pd.DataFrame`，其中行和列的索引都是由字符串组成的：

```py
`df = pd.DataFrame([     [24, 180, "blue"],     [42, 166, "brown"],     [22, 160, "green"], ], columns=[     "age",     "height_cm",     "eye_color" ], index=["Jack", "Jill", "Jayne"]) df` 
```

```py
 `age   height_cm   eye_color Jack    24    180         blue Jill    42    166         brown Jayne   22    160         green` 
```

默认情况下，`pd.DataFrame.filter`会选择与标签参数匹配的列，类似于`pd.DataFrame[]`：

```py
`df.filter(["age", "eye_color"])` 
```

```py
 `age   eye_color Jack   24    blue Jill   42    brown Jayne  22    green` 
```

然而，`pd.DataFrame.filter`也接受一个`axis=`参数，它允许你改变所选择的轴。若要选择行而不是列，传递`axis=0`：

```py
`df.filter(["Jack", "Jill"], axis=0)` 
```

```py
 `age   height_cm   eye_color Jack   24    180         blue Jill   42    166         brown` 
```

你不局限于与标签进行精确字符串匹配。如果你想选择包含某个字符串的标签，可以使用`like=`参数。此示例将选择任何包含下划线的列：

```py
`df.filter(like="_")` 
```

```py
 `height_cm   eye_color Jack   180         blue Jill   166         brown Jayne  160         green` 
```

如果简单的字符串包含检查不够，你也可以使用正则表达式通过`regex=`参数匹配索引标签。以下示例将选择任何以`"Ja"`开头，但不以`"e"`结尾的行标签：

```py
`df.filter(regex=r"^Ja.*(?<!e)$", axis=0)` 
```

```py
 `age   height_cm   eye_color Jack   24    180         blue` 
```

# 按数据类型选择

到目前为止，在这本食谱中，我们*已经看过*数据类型，但我们并没有深入讨论它们到底是什么。我们还没有完全深入探讨；pandas 的类型系统将在*第三章*，*数据类型*中进行深入讨论。不过，目前你应该意识到，列类型提供了元数据，`pd.DataFrame.select_dtypes`可以用它来进行选择。

## 如何操作

让我们从一个包含整数、浮点数和字符串列的`pd.DataFrame`开始：

```py
`df = pd.DataFrame([     [0, 1.0, "2"],     [4, 8.0, "16"], ], columns=["int_col", "float_col", "string_col"]) df` 
```

```py
 `int_col   float_col   string_col 0   0         1.0         2 1   4         8.0         16` 
```

使用`pd.DataFrame.select_dtypes`仅选择整数列：

```py
`df.select_dtypes("int")` 
```

```py
 `int_col 0   0 1   4` 
```

如果你传递一个列表参数，多个类型可以被选择：

```py
`df.select_dtypes(include=["int", "float"])` 
```

```py
 `int_col   float_col 0   0         1.0 1   4         8.0` 
```

默认行为是包括你作为参数传递的数据类型。要排除它们，请使用 `exclude=` 参数：

```py
`df.select_dtypes(exclude=["int", "float"])` 
```

```py
 `string_col 0   2 1   16` 
```

# 通过布尔数组进行选择/过滤

使用布尔列表/数组（也称为 *遮罩*）是选择子集行的常见方法。

## 如何做到这一点

让我们创建一个 `True=/=False` 值的遮罩，并与一个简单的 `pd.Series` 配对：

```py
`mask = [True, False, True] ser = pd.Series(range(3)) ser` 
```

```py
`0    0 1    1 2    2 dtype: int64` 
```

将遮罩作为参数传递给 `pd.Series[]` 将返回每一行，其中相应的遮罩条目为 `True`：

```py
`ser[mask]` 
```

```py
`0    0 2    2 dtype: int64` 
```

`pd.Series.loc` 在这种情况下将与 `pd.Series[]` 的行为完全一致：

```py
`ser.loc[mask]` 
```

```py
`0    0 2    2 dtype: int64` 
```

有趣的是，尽管 `pd.DataFrame[]` 通常在提供列表参数时尝试从列中选择，但在使用布尔值序列时，它的行为有所不同。使用我们已经创建的遮罩，`df[mask]` 实际上会沿行匹配，而不是列：

```py
`df = pd.DataFrame(np.arange(6).reshape(3, -1)) df[mask]` 
```

```py
 `0   1 0   0   1 2   4   5` 
```

如果你需要同时遮蔽行和列，`pd.DataFrame.loc` 将接受两个遮罩参数：

```py
`col_mask = [True, False] df.loc[mask, col_mask]` 
```

```py
 `0 0   0 2   4` 
```

## 还有更多内容……

通常，你会使用 OR、AND 或 INVERT 运算符的组合来操作你的遮罩。为了演示这些操作，我们从一个稍微复杂的 `pd.DataFrame` 开始：

```py
`df = pd.DataFrame([     [24, 180, "blue"],     [42, 166, "brown"],     [22, 160, "green"], ], columns=["age", "height_cm", "eye_color"], index=["Jack", "Jill", "Jayne"]) df` 
```

```py
 `age   height_cm   eye_color Jack   24    180         blue Jill   42    166         brown Jayne  22    160         green` 
```

如果我们的目标是只筛选出有蓝眼睛或绿眼睛的用户，我们可以先识别哪些用户有蓝眼睛：

```py
`blue_eyes = df["eye_color"] == "blue" blue_eyes` 
```

```py
`Jack      True Jill     False Jayne    False Name: eye_color, dtype: bool` 
```

然后，我们找出哪些人有绿色眼睛：

```py
`green_eyes = df["eye_color"] == "green" green_eyes` 
```

```py
`Jack     False Jill     False Jayne     True Name: eye_color, dtype: bool` 
```

并将这些结合在一起，使用 OR 运算符 `|` 形成一个布尔 *遮罩*：

```py
`mask = blue_eyes | green_eyes mask` 
```

```py
`Jack      True Jill     False Jayne     True Name: eye_color, dtype: bool` 
```

在将该遮罩作为索引器传递给我们的 `pd.DataFrame` 之前：

```py
`df[mask]` 
```

```py
 `age   height_cm   eye_color Jack   24    180         blue Jayne  22    160         green` 
```

与使用 OR 运算符 `|` 不同，你通常会使用 AND 运算符 `&`。例如，让我们为年龄小于 40 的记录创建一个筛选器：

```py
`age_lt_40 = df["age"] < 40 age_lt_40` 
```

```py
`Jack      True Jill     False Jayne     True Name: age, dtype: bool` 
```

还要高度大于 170：

```py
`height_gt_170 = df["height_cm"] > 170 height_gt_170` 
```

```py
`Jack      True Jill     False Jayne    False Name: height_cm, dtype: bool` 
```

这些可以通过 AND 运算符组合在一起，只选择满足两个条件的记录：

```py
`df[age_lt_40 & height_gt_170]` 
```

```py
 `age   height_cm   eye_color Jack   24    180         blue` 
```

INVERT 运算符可以视为 NOT 运算符；也就是说，在遮罩的上下文中，它将使任何 `True` 值变为 `False`，任何 `False` 值变为 `True`。继续我们上面的例子，如果我们想找到那些没有满足年龄低于 40 且身高超过 170 的条件的记录，我们只需使用 `~` 反转遮罩：

```py
`df[~(age_lt_40 & height_gt_170)]` 
```

```py
 `age   height_cm   eye_color Jill   42    166         brown Jayne  22    160         green` 
```

# 使用 MultiIndex 进行选择 – 单一层级

`pd.MultiIndex` 是 `pd.Index` 的一个子类，支持层级标签。根据询问的人不同，这可能是 pandas 最好的特性之一，或者是最差的特性之一。看完这本手册后，我希望你把它视为最好的特性之一。

对 `pd.MultiIndex` 的大部分贬低来自于这样一个事实：使用它选择时的语法很容易变得模糊，尤其是在使用 `pd.DataFrame[]` 时。以下示例仅使用 `pd.DataFrame.loc` 方法，避免使用 `pd.DataFrame[]` 以减少混淆。

## 如何做到这一点

`pd.MultiIndex.from_tuples` 可以用来从元组列表构建 `pd.MultiIndex`。在以下示例中，我们创建一个具有两个层级的 `pd.MultiIndex` – `first_name` 和 `last_name`，依次排列。我们将其与一个非常简单的 `pd.Series` 配对：

```py
`index = pd.MultiIndex.from_tuples([     ("John", "Smith"),     ("John", "Doe"),     ("Jane", "Doe"),     ("Stephen", "Smith"), ], names=["first_name", "last_name"]) ser = pd.Series(range(4), index=index) ser` 
```

```py
`first_name  last_name John        Smith        0             Doe          1 Jane        Doe          2 Stephen     Smith        3 dtype: int64` 
```

使用`pd.Series.loc`与`pd.MultiIndex`以及标量参数将匹配`pd.MultiIndex`的第一个级别。输出将不包含这个第一个级别的结果：

```py
`ser.loc["John"]` 
```

```py
`last_name Smith    0 Doe      1 dtype: int64` 
```

上面示例中删除`pd.MultiIndex`第一个级别的行为也被称为*部分切片*。这个概念类似于我们在前几节看到的`.loc`和`.iloc`中的维度压缩，唯一的不同是，pandas 在这里试图减少`pd.MultiIndex`中的*级别*数量，而不是减少*维度*。

为了防止发生这种隐式的级别减少，我们可以再次提供一个包含单一元素的列表参数：

```py
`ser.loc[["John"]]` 
```

```py
`first_name  last_name John        Smith        0             Doe          1 dtype: int64` 
```

# 使用 MultiIndex 选择 – 多个级别

如果你只能从`pd.MultiIndex`的第一个级别进行选择，那么事情就不那么有趣了。幸运的是，`pd.DataFrame.loc`通过巧妙地使用元组参数可以扩展到不仅仅是第一个级别。

## 如何执行

让我们重新创建前面一节中的`pd.Series`：

```py
`index = pd.MultiIndex.from_tuples([     ("John", "Smith"),     ("John", "Doe"),     ("Jane", "Doe"),     ("Stephen", "Smith"), ], names=["first_name", "last_name"]) ser = pd.Series(range(4), index=index) ser` 
```

```py
`first_name  last_name John        Smith        0             Doe          1 Jane        Doe          2 Stephen     Smith        3 dtype: int64` 
```

要选择所有第一个索引级别使用标签`"Jane"`且第二个索引级别使用标签`"Doe"`的记录，请传递以下元组：

```py
`ser.loc[("Jane", "Doe")]` 
```

```py
`2` 
```

要选择所有第一个索引级别使用标签`"Jane"`且第二个索引级别使用标签`"Doe"`的记录，同时保持`pd.MultiIndex`的形状，请将一个单一元素的列表放入元组中：

```py
`ser.loc[(["Jane"], "Doe")]` 
```

```py
`first_name  last_name Jane        Doe          2 dtype: int64` 
```

要选择所有第一个索引级别使用标签`"John"`且第二个索引级别使用标签`"Smith"`，或者第一个索引级别是`"Jane"`且第二个索引级别是`"Doe"`的记录：

```py
`ser.loc[[("John", "Smith"), ("Jane", "Doe")]]` 
```

```py
`first_name  last_name John        Smith        0 Jane        Doe          2 dtype: int64` 
```

要选择所有第二个索引级别为`"Doe"`的记录，请将一个空切片作为元组的第一个元素。注意，这会删除第二个索引级别，并从剩下的第一个索引级别重建结果，形成一个简单的`pd.Index`：

```py
`ser.loc[(slice(None), "Doe")]` 
```

```py
`first_name John    1 Jane    2 dtype: int64` 
```

要选择所有第二个索引级别为`"Doe"`的记录，同时保持`pd.MultiIndex`的形状，请将一个单元素列表作为第二个元组元素：

```py
`ser.loc[(slice(None), ["Doe"])]` 
```

```py
`first_name  last_name John        Doe          1 Jane        Doe          2 dtype: int64` 
```

在这一点上，你可能会问，`slice(None)`到底是什么意思？这个相当隐晦的表达式实际上创建了一个没有*起始*、*停止*或*步长*值的切片对象，这在用更简单的 Python 列表来说明时会更容易理解——注意，这里的行为：

```py
`alist = list("abc") alist[:]` 
```

```py
`['a', 'b', 'c']` 
```

结果与使用`slice(None)`时完全相同：

```py
`alist[slice(None)]` 
```

```py
`['a', 'b', 'c']` 
```

当`pd.MultiIndex`期待一个元组参数却没有得到时，这个问题通常是由元组中的切片引起的，类似于 Python 中`(:,)`的语法错误。更明确的写法`(slice(None),)`可以修复这个问题。

## 还有更多…

如果你觉得`slice(None)`语法太繁琐，pandas 提供了一个方便的对象叫做`pd.IndexSlice`，它像元组一样工作，但允许你使用更自然的`:`符号进行切片。

```py
`ser.loc[(slice(None), ["Doe"])]` 
```

```py
`first_name  last_name John        Doe          1 Jane        Doe          2 dtype: int64` 
```

这样可以变成：

```py
`ixsl = pd.IndexSlice ser.loc[ixsl[:, ["Doe"]]]` 
```

```py
`first_name  last_name John        Doe          1 Jane        Doe          2 dtype: int64` 
```

# 使用 MultiIndex 选择 – 一个 DataFrame

`pd.MultiIndex`可以同时作为行索引和列索引使用，并且通过`pd.DataFrame.loc`的选择方式在两者中都可以工作。

## 如何执行

让我们创建一个既在行也在列使用`pd.MultiIndex`的`pd.DataFrame`：

```py
`row_index = pd.MultiIndex.from_tuples([     ("John", "Smith"),     ("John", "Doe"),     ("Jane", "Doe"),     ("Stephen", "Smith"), ], names=["first_name", "last_name"]) col_index = pd.MultiIndex.from_tuples([     ("music", "favorite"),     ("music", "last_seen_live"),     ("art", "favorite"), ], names=["art_type", "category"]) df = pd.DataFrame([    ["Swift", "Swift", "Matisse"],    ["Mozart", "T. Swift", "Van Gogh"],    ["Beatles", "Wonder", "Warhol"],    ["Jackson", "Dylan", "Picasso"], ], index=row_index, columns=col_index) df` 
```

```py
 `art_type              music           art              category   favorite   last_seen_live  favorite first_name   last_name John         Smith      Swift      Swift           Matisse              Doe        Mozart     T. Swift        Van Gogh Jane         Doe        Beatles    Wonder          Warhol Stephen      Smith      Jackson    Dylan           Picasso` 
```

要选择所有第二级为 `"Smith"` 的行以及所有第二级为 `"favorite"` 的列，你需要传递两个元组，其中每个元组的第二个元素是所需的标签：

```py
`row_idxer = (slice(None), "Smith") col_idxer = (slice(None), "favorite") df.loc[row_idxer, col_idxer]` 
```

```py
 `art_type   music      art              category   favorite   favorite first_name   last_name John         Smith      Swift      Matisse Stephen      Smith      Jackson    Picasso` 
```

`pd.DataFrame.loc` 总是需要两个参数——第一个指定如何对行进行索引，第二个指定如何对列进行索引。当你有一个 `pd.DataFrame`，其行列都使用 `pd.MultiIndex` 时，你可能会发现，将索引器分开定义为变量在风格上更为清晰。上面的代码也可以写成：

```py
`df.loc[(slice(None), "Smith"), (slice(None), "favorite")]` 
```

```py
 `art_type   music      art              category   favorite   favorite first_name   last_name John         Smith      Swift      Matisse Stephen      Smith      Jackson    Picasso` 
```

尽管你可以说这更难以解读。正如古老的说法所说，美在于观者的眼中。

# 使用 .loc 和 .iloc 进行项目赋值

pandas 库针对读取、探索和评估数据进行了优化。试图 *修改* 或改变数据的操作要低效得多。

然而，当你必须修改数据时，可以使用 `.loc` 和 `.iloc` 来实现。

## 如何做

让我们从一个非常小的 `pd.Series` 开始：

```py
`ser = pd.Series(range(3), index=list("abc"))` 
```

`pd.Series.loc` 在你想通过匹配索引的标签来赋值时非常有用。例如，如果我们想在行索引包含 `"b"` 的位置存储值 `42`，我们可以写成：

```py
`ser.loc["b"] = 42 ser` 
```

```py
`a     0 b    42 c     2 dtype: int64` 
```

`pd.Series.iloc` 用于在你想按位置赋值时。例如，为了将值 `-42` 赋给我们 `pd.Series` 中的第二个元素，我们可以写成：

```py
`ser.iloc[2] = -42 ser` 
```

```py
`a     0 b    42 c   -42 dtype: int64` 
```

## 还有更多内容…

通过 pandas 修改数据的成本在很大程度上取决于两个因素：

+   pandas `pd.Series` 支持的数组类型（*第三章*，*数据类型*，将在后续章节中详细讲解数据类型）

+   有多少个对象引用了 `pd.Series`

对这些因素的深入探讨远超本书的范围。对于上面的第一个要点，我的普遍建议是，*数组类型越简单*，你就越有可能在不复制数组内容的情况下修改它，对于大型数据集来说，复制可能会非常昂贵。

对于第二个要点，在 pandas 2.x 系列中涉及了大量的 **写时复制** (**CoW**) 工作。CoW 是 pandas 3.0 中的默认行为，它试图使得在修改数据时，哪些内容被复制，哪些内容没有被复制变得更加可预测。对于高级用户，我强烈建议阅读 pandas CoW 文档。

# DataFrame 列赋值

在 pandas 中，对 *数据* 的赋值操作可能相对昂贵，但为 `pd.DataFrame` 分配列是常见操作。

## 如何做

让我们创建一个非常简单的 `pd.DataFrame`：

```py
`df = pd.DataFrame({"col1": [1, 2, 3]}) df` 
```

```py
 `col1 0   1 1   2 2   3` 
```

新列可以使用 `pd.DataFrame[]` 操作符进行赋值。最简单的赋值类型可以是一个标量值，并将其 *广播* 到 `pd.DataFrame` 的每一行：

```py
`df["new_column1"] = 42 df` 
```

```py
 `col1   new_column1 0   1      42 1   2      42 2   3      42` 
```

你还可以赋值一个 `pd.Series` 或序列，只要元素的数量与 `pd.DataFrame` 中的行数匹配：

```py
`df["new_column2"] = list("abc") df` 
```

```py
 `col1   new_column1   new_column2 0   1      42            a 1   2      42            b 2   3      42            c` 
```

```py
`df["new_column3"] = pd.Series(["dog", "cat", "human"]) df` 
```

```py
 `col1   new_column1   new_column2   new_column3 0   1      42            a             dog 1   2      42            b             cat 2   3      42            c             human` 
```

如果新序列的行数与现有 `pd.DataFrame` 的行数不匹配，赋值将失败：

```py
`df["should_fail"] = ["too few", "rows"]` 
```

```py
`ValueError: Length of values (2) does not match length of index (3)` 
```

对于具有`pd.MultiIndex`列的`pd.DataFrame`，也可以进行赋值操作。让我们来看一个这样的`pd.DataFrame`：

```py
`row_index = pd.MultiIndex.from_tuples([     ("John", "Smith"),     ("John", "Doe"),     ("Jane", "Doe"),     ("Stephen", "Smith"), ], names=["first_name", "last_name"]) col_index = pd.MultiIndex.from_tuples([     ("music", "favorite"),     ("music", "last_seen_live"),     ("art", "favorite"), ], names=["art_type", "category"]) df = pd.DataFrame([    ["Swift", "Swift", "Matisse"],    ["Mozart", "T. Swift", "Van Gogh"],    ["Beatles", "Wonder", "Warhol"],    ["Jackson", "Dylan", "Picasso"], ], index=row_index, columns=col_index) df` 
```

```py
 `art_type   music            art              category   favorite   last_seen_live   favorite first_name   last_name John         Smith      Swift      Swift            Matisse              Doe        Mozart     T. Swift         Van Gogh Jane         Doe        Beatles    Wonder           Warhol Stephen      Smith      Jackson    Dylan            Picasso` 
```

要在`"art"`层次下为看到的博物馆数量赋值，可以将元组作为参数传递给`pd.DataFrame.loc`：

```py
`df.loc[:, ("art", "museuems_seen")] = [1, 2, 4, 8] df` 
```

```py
 `art_type    music            art              category    favorite   last_seen_live   favorite   museuems_seen first_name   last_name John         Smith       Swift       Swift           Matisse    1              Doe         Mozart      T. Swift        Van Gogh   2 Jane         Doe         Beatles     Wonder          Warhol     4 Stephen      Smith       Jackson     Dylan           Picasso    8` 
```

使用`pd.DataFrame`进行赋值时遵循与使用`pd.DataFrame[]`和`pd.DataFrame.loc[]`选择值时相同的模式。主要的区别在于，在选择时，你会在表达式的右侧使用`pd.DataFrame[]`和`pd.DataFrame.loc[]`，而在赋值时，它们出现在左侧。

## 还有更多…

`pd.DataFrame.assign`方法可用于在赋值时允许*方法链*。我们从一个简单的`pd.DataFrame`开始，来展示这种方法的用法：

```py
`df = pd.DataFrame([[0, 1], [2, 4]], columns=list("ab")) df` 
```

```py
 `a   b 0   0   1 1   2   4` 
```

*方法链*指的是 pandas 能够将多个算法连续应用于 pandas 数据结构的能力（算法及其应用方式将在*第五章*，*算法及其应用*中详细讨论）。所以，要将我们的`pd.DataFrame`进行加倍并为每个元素加上 42，我们可以做类似如下的操作：

```py
`(     df     .mul(2)     .add(42) )` 
```

```py
 `a    b 0   42   44 1   46   50` 
```

但是，如果我们想在这个链条中添加一个新列会发生什么呢？不幸的是，使用标准赋值运算符时，你需要打破这个链条，通常需要为新变量赋值：

```py
`df2 = (     df     .mul(2)     .add(42) ) df2["assigned_c"] = df2["b"] - 3 df2` 
```

```py
 `a   b   assigned_c 0   42  44  41 1   46  50  47` 
```

但是通过`pd.DataFrame.assign`，你可以继续链式操作。只需将所需的列标签作为关键字传递给`pd.DataFrame.assign`，其参数是你希望在新的`pd.DataFrame`中看到的值：

```py
`(     df     .mul(2)     .add(42)     .assign(chained_c=lambda df: df["b"] - 3) )` 
```

```py
 `a    b    chained_c 0   42   44   41 1   46   50   47` 
```

在这种情况下，你只能使用符合 Python 语法要求的标签作为参数名，而不幸的是，这在`pd.MultiIndex`中无法使用。有些用户认为方法链使调试变得更困难，而另一些人则认为像这样的方法链使代码更容易阅读。归根结底，没有对错之分，我现在能给出的最佳建议是使用你最舒服的形式。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/pandas`](https://packt.link/pandas)

![](img/QR_Code5040900042138312.png)
