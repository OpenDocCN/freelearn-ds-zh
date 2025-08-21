

# 第三章：数据类型

`pd.Series`的数据类型允许你指定可以或不可以存储的元素类型。数据类型对于确保数据质量以及在代码中启用高性能算法至关重要。如果你有数据库工作背景，你很可能已经熟悉数据类型及其好处；你将在 pandas 中找到像`TEXT`、`INTEGER`和`DOUBLE PRECISION`这样的类型，就像在数据库中一样，尽管它们的名称不同。

然而，与数据库不同，pandas 提供了多种实现方式，来处理`TEXT`、`INTEGER`和`DOUBLE PRECISION`类型。不幸的是，这意味着作为最终用户，你至少应该了解不同数据类型的实现方式，以便为你的应用选择最佳的选项。

关于 pandas 中类型的简短历史可以帮助解释这一可用性上的怪癖。最初，pandas 是建立在 NumPy 类型系统之上的。这种方法在一段时间内是有效的，但存在重大缺陷。首先，pandas 构建的 NumPy 类型不支持缺失值，因此 pandas 创造了一种“弗兰肯斯坦的怪物”方法来支持这些值。由于 NumPy 专注于*数值*计算，它也没有提供一流的字符串数据类型，导致 pandas 中的字符串处理非常差。

从 pandas 版本 0.23 开始，pandas 努力超越了 NumPy 类型系统，该版本引入了直接内置在 pandas 中的新数据类型，这些类型虽然仍然使用 NumPy 实现，但实际上能够处理缺失值。在版本 1.0 中，pandas 实现了自己的字符串数据类型。当时，这些类型被称为`numpy_nullable`数据类型，但随着时间的推移，它们被称为 pandas 扩展类型。

在这一切发生的同时，pandas 的原始创建者 Wes McKinney 正致力于 Apache Arrow 项目。完全解释 Arrow 项目超出了本书的范围，但它帮助的一个主要方面是定义一组可以在不同工具和编程语言之间使用的标准化数据类型。这些数据类型也受到数据库的启发；如果使用数据库已经是你分析旅程的一部分，那么 Arrow 类型对你来说可能非常熟悉。从版本 2.0 开始，pandas 允许你使用 Arrow 作为数据类型。

尽管支持 pandas 扩展类型和 Arrow 数据类型，但 pandas 的默认类型从未改变，在大多数情况下仍然使用 NumPy。作者认为这是非常遗憾的；本章将介绍一种较为主观的观点，如何最好地管理类型的领域，通常的指导原则如下：

+   在可用时，使用 pandas 扩展类型

+   当 pandas 扩展类型不足时，使用 Arrow 数据类型

+   使用基于 NumPy 的数据类型

这条指南可能会引发争议，并且在极端情况下可能会受到质疑，但对于刚接触 pandas 的人来说，我认为这种优先级设定为用户提供了最佳的可用性与性能平衡，无需深入了解 pandas 背后如何工作。

本章的总体结构将首先介绍 pandas 扩展系统的常规使用方法，然后再深入探讨 Arrow 类型系统以应对更复杂的使用案例。在我们介绍这些类型时，还将突出展示可以通过*访问器*解锁的特殊行为。最后，我们将讨论历史上的 NumPy 支持的数据类型，并深入探讨它们的一些致命缺陷，我希望这能说服你为什么应当限制使用这些类型。

本章将涵盖以下几个实例：

+   整数类型

+   浮点类型

+   布尔类型

+   字符串类型

+   缺失值处理

+   分类类型

+   时间类型 – 日期时间

+   时间类型 – 时间差

+   时间类型 PyArrow

+   PyArrow 列表类型

+   PyArrow 十进制类型

+   NumPy 类型系统、对象类型及其陷阱

# 整数类型

整数类型是最基本的类型类别。类似于 Python 中的`int`类型或数据库中的`INTEGER`数据类型，这些类型仅能表示整数。尽管有这一限制，整数在各种应用中非常有用，包括但不限于算术运算、索引、计数和枚举。

整数类型经过高度优化，性能得到了极大的提升，从 pandas 一直追踪到你电脑上的硬件。pandas 提供的整数类型比 Python 标准库中的`int`类型要快得多，正确使用整数类型通常是实现高性能、可扩展报告的关键。

## 如何实现

任何有效的整数序列都可以作为参数传递给`pd.Series`构造函数。搭配`dtype=pd.Int64Dtype()`参数，你将得到一个 64 位整数数据类型：

```py
`pd.Series(range(3), dtype=pd.Int64Dtype())` 
```

```py
`0    0 1    1 2    2 dtype: Int64` 
```

当存储和计算资源不成问题时，用户通常会选择 64 位整数，但在我们的示例中，我们也可以选择一个更小的数据类型：

```py
`pd.Series(range(3), dtype=pd.Int8Dtype())` 
```

```py
`0    0 1    1 2    2 dtype: Int8` 
```

关于缺失值，pandas 使用`pd.NA`作为指示符，类似于数据库使用`NULL`：

```py
`pd.Series([1, pd.NA, 2], dtype=pd.Int64Dtype())` 
```

```py
`0       1 1    <NA> 2       2 dtype: Int64` 
```

为了方便，`pd.Series`构造函数会自动将 Python 中的`None`值转换为`pd.NA`：

```py
`pd.Series([1, None, 2], dtype=pd.Int64Dtype())` 
```

```py
`0       1 1    <NA> 2       2 dtype: Int64` 
```

## 还有更多……

对于科学计算的新手用户来说，重要的是要知道，与 Python 的`int`类型不同，后者没有理论上的大小限制，pandas 中的整数有上下限。这些限制由整数的*宽度*和*符号*决定。

在大多数计算环境中，用户拥有的整数宽度为 8、16、32 和 64。符号性可以是*有符号*（即，数字可以是正数或负数）或*无符号*（即，数字不得为负）。每种整数类型的限制总结在下表中：

| **类型** | **下限** | **上限** |
| --- | --- | --- |
| 8 位宽度，有符号 | -128 | 127 |
| 8 位宽度，无符号 | 0 | 255 |
| 16 位宽度，有符号 | -32769 | 32767 |
| 16 位宽度，无符号 | 0 | 65535 |
| 32 位宽度，有符号 | -2147483648 | 2147483647 |
| 32 位宽度，无符号 | 0 | 4294967295 |
| 64 位宽度，有符号 | -(2**63) | 2**63-1 |
| 64 位宽度，无符号 | 0 | 2**64-1 |

表 3.1：按符号性和宽度的整数极限

这些类型的权衡是容量与内存使用之间的平衡——64 位整数类型需要的内存是 8 位整数类型的 8 倍。是否会成为问题完全取决于你的数据集的大小以及你执行分析的系统。

在 pandas 扩展类型系统中，每种类型的`dtype=`参数遵循`pd.IntXXDtype()`形式的有符号整数和`pd.UIntXXDtype()`形式的无符号整数，其中`XX`表示位宽：

```py
`pd.Series(range(555, 558), dtype=pd.Int16Dtype())` 
```

```py
`0    555 1    556 2    557 dtype: Int16` 
```

```py
`pd.Series(range(3), dtype=pd.UInt8Dtype())` 
```

```py
`0    0 1    1 2    2 dtype: UInt8` 
```

# 浮点类型

浮点类型允许你表示实数，而不仅仅是整数。这使得你可以在计算中处理一个连续的、*理论上*无限的值集。浮点计算几乎出现在每一个科学计算、宏观金融分析、机器学习算法等中，这一点并不令人惊讶。

然而，单词*理论上*的重点是故意强调的，并且对于理解非常重要。浮点类型仍然有边界，真实的限制是由你的计算机硬件所强加的。本质上，能够表示任何数字的概念是一种错觉。浮点类型容易失去精度并引入舍入误差，尤其是在处理极端值时。因此，当你需要绝对精度时，浮点类型并不适用（对于这种情况，你可以参考本章后面介绍的 PyArrow 十进制类型）。

尽管存在这些限制，但实际上你很少需要绝对精度，因此浮点类型是最常用的数据类型，通常用于表示分数。

## 如何操作

要构建浮点数据，请使用`dtype=pd.Float64Dtype()`：

```py
`pd.Series([3.14, .333333333, -123.456], dtype=pd.Float64Dtype())` 
```

```py
`0        3.14 1    0.333333 2    -123.456 dtype: Float64` 
```

就像我们在整数类型中看到的那样，缺失值指示符是`pd.NA`。Python 对象`None`会被隐式地转换为此，以便于使用：

```py
`pd.Series([3.14, None, pd.NA], dtype=pd.Float64Dtype())` 
```

```py
`0    3.14 1    <NA> 2    <NA> dtype: Float64` 
```

## 还有更多内容…

由于其设计的性质，浮点值是*不精确*的，且浮点值的算术运算比整数运算要慢。深入探讨浮点算术超出了本书的范围，但有兴趣的人可以在 Python 文档中找到更多信息。

Python 有一个内建的 `float` 类型，这个名字有些误导，因为它实际上是一个 IEEE 754 `double`。该标准和其他像 C/C++ 这样的语言有独立的 `float` 和 `double` 类型，前者占用 32 位，后者占用 64 位。为了澄清这些位宽的差异，同时保持与 Python 术语的一致性，pandas 提供了 `pd.Float64Dtype()`（有些人认为它是 `double`）和 `pd.Float32Dtype()`（有些人认为它是 `float`）。

通常，除非你的系统资源有限，否则建议用户使用 64 位浮动点类型。32 位浮动点类型丢失精度的概率远高于对应的 64 位类型。事实上，32 位浮动点数仅提供 6 到 9 位小数的精度，因此，尽管我们可以很清楚地看到数字并不相同，下面的表达式仍然可能返回 `True` 作为相等比较的结果：

```py
`ser1 = pd.Series([1_000_000.123], dtype=pd.Float32Dtype()) ser2 = pd.Series([1_000_000.124], dtype=pd.Float32Dtype()) ser1.eq(ser2)` 
```

```py
`0    True dtype: boolean` 
```

使用 64 位浮动点数时，你至少能获得 15 到 17 位小数的精度，因此四舍五入误差发生的数值范围要远大于 32 位浮动点数。

# 布尔类型

布尔类型表示一个值，值只能是 `True` 或 `False`。布尔数据类型用于简单地回答是/否式的问题，也广泛用于机器学习算法中，将分类值转换为计算机可以更容易处理的 1 和 0（分别代表 `True` 和 `False`）（参见《第五章，算法及其应用》中关于 *One-hot 编码与 pd.get_dummies* 的部分）。

## 如何实现

对于布尔类型，适当的 `dtype=` 参数是 `pd.BooleanDtype`：

```py
`pd.Series([True, False, True], dtype=pd.BooleanDtype())` 
```

```py
`0     True 1    False 2     True dtype: boolean` 
```

pandas 库会自动为你处理值到布尔值的隐式转换。通常，`False` 和 `True` 分别用 0 和 1 来代替：

```py
`pd.Series([1, 0, 1], dtype=pd.BooleanDtype())` 
```

```py
`0     True 1    False 2     True dtype: boolean` 
```

再次强调，`pd.NA` 是标准的缺失值指示符，尽管 pandas 会自动将 `None` 转换为缺失值：

```py
`pd.Series([1, pd.NA, None], dtype=pd.BooleanDtype())` 
```

```py
`0    True 1    <NA> 2    <NA> dtype: boolean` 
```

# 字符串类型

字符串数据类型是表示文本数据的合适选择。除非你在纯粹的科学领域工作，否则字符串类型的值很可能会广泛存在于你使用的数据中。

在本教程中，我们将重点介绍 pandas 在处理字符串数据时提供的一些附加功能，特别是通过 `pd.Series.str` 访问器。这个访问器有助于改变大小写、提取子字符串、匹配模式等等。

作为一个技术注解，在我们进入具体内容之前，从 pandas 3.0 开始，字符串类型将会在幕后进行重大改造，启用一种更符合类型的实现，速度更快，内存需求也比 pandas 2.x 系列要低得多。为了在 3.0 及更高版本中实现这一点，强烈建议用户在安装 pandas 时同时安装 PyArrow。对于那些想了解 pandas 3.0 中字符串处理的权威参考，可以查看 PDEP-14 专门针对字符串数据类型的文档。

## 如何做到这一点

字符串数据应该使用 `dtype=pd.StringDtype()` 构造：

```py
`pd.Series(["foo", "bar", "baz"], dtype=pd.StringDtype())` 
```

```py
`0    foo 1    bar 2    baz dtype: string` 
```

你可能已经发现，`pd.NA` 是用于表示缺失值的标识符，但 pandas 会自动将 `None` 转换为 `pd.NA`：

```py
`pd.Series(["foo", pd.NA, None], dtype=pd.StringDtype())` 
```

```py
`0     foo 1    <NA> 2    <NA> dtype: string` 
```

在处理包含字符串数据的 `pd.Series` 时，pandas 会创建一个称为字符串 *访问器* 的工具，帮助你解锁适用于字符串的新方法。字符串访问器通过 `pd.Series.str` 使用，帮助你做诸如通过 `pd.Series.str.len` 获取每个字符串的长度等操作：

```py
`ser = pd.Series(["xx", "YyY", "zZzZ"], dtype=pd.StringDtype()) ser.str.len()` 
```

```py
`0    2 1    3 2    4 dtype: Int64` 
```

它也可以用于强制将所有内容转换为特定的格式，例如大写：

```py
`ser.str.upper()` 
```

```py
`0      XX 1     YYY 2    ZZZZ dtype: string` 
```

它也可以用于强制将所有内容转换为小写：

```py
`ser.str.lower()` 
```

```py
`0      xx 1     yyy 2    zzzz dtype: string` 
```

甚至可以是“标题大小写”（即只有第一个字母大写，其他字母小写）：

```py
`ser.str.title()` 
```

```py
`0      Xx 1     Yyy 2    Zzzz dtype: string` 
```

`pd.Series.str.contains` 可用于检查字符串是否包含特定内容：

```py
`ser = pd.Series(["foo", "bar", "baz"], dtype=pd.StringDtype()) ser.str.contains("o")` 
```

```py
`0     True 1    False 2    False dtype: boolean` 
```

它还具有使用 `regex=True` 测试正则表达式的灵活性，类似于标准库中的 `re.search`。`case=False` 参数还会将匹配操作转换为不区分大小写的比较：

```py
`ser.str.contains(r"^ba[rz]$", case=False, regex=True)` 
```

```py
`0    False 1     True 2     True dtype: boolean` 
```

# 缺失值处理

在我们继续讨论更多数据类型之前，我们必须回过头来谈谈 pandas 如何处理缺失值。到目前为止，事情都很简单（我们只看到了 `pd.NA`），但随着我们探索更多类型，会发现 pandas 处理缺失值的方式并不一致，这主要源于该库的开发历史。虽然能够挥动魔杖让任何不一致消失会很棒，但实际上，它们一直存在，并将在未来几年继续出现在生产代码库中。对这种发展过程有一个高层次的理解将帮助你编写更好的 pandas 代码，并希望能将那些不了解的人引导到我们在本书中提倡的习惯用法中。

## 如何做到这一点

pandas 库最初是建立在 NumPy 之上的，而 NumPy 的默认数据类型不支持缺失值。因此，pandas 必须从零开始构建自己的缺失值处理解决方案，并且，无论好坏，它决定使用 `np.nan` 哨兵值（代表“非数字”）作为其处理缺失值的工具。

`np.nan` 本身是 IEEE 754 标准中的“非数字”哨兵值的实现，这一规范仅与浮点运算有关。对于整数数据来说并不存在“非数字”这一概念，这也是为什么 pandas 会隐式地将像这样的 `pd.Series` 转换为：

```py
`ser = pd.Series(range(3)) ser` 
```

```py
`0    0 1    1 2    2 dtype: int64` 
```

在分配缺失值后，将数据转换为浮点数据类型：

```py
`ser.iloc[1] = None ser` 
```

```py
`0    0.0 1    NaN 2    2.0 dtype: float64` 
```

正如我们在 *浮动点类型* 配方中讨论的那样，浮动点值的计算速度比整型值要慢。虽然整数可以使用 8 位和 16 位宽度表示，但浮动点类型至少需要 32 位。即使你使用的是 32 位宽度的整数，使用 32 位浮动点值可能会因为精度损失而不可行，而使用 64 位整数时，转换可能只能牺牲精度。一般来说，从整型到浮动点类型的转换，必须牺牲一定的性能、内存使用和/或精度，因此这类转换并不是理想的选择。

当然，pandas 不仅提供了整型和浮动点类型，因此其他类型也必须附加自定义的缺失值解决方案。默认的布尔类型会被转换为 `object` 类型，这一问题将在本章后面的一个配方中讨论。对于日期时间类型，我们很快会讨论，pandas 必须创建一个完全不同的 `pd.NaT` 哨兵，因为 `np.nan` 在技术上并不适用于该数据类型。实质上，pandas 中的每个数据类型都有可能有自己的指示符和隐式类型转换规则，这对于初学者和有经验的 pandas 开发者来说都很难解释清楚。

pandas 库通过引入 *pandas 扩展类型* 在 0.24 版本中尝试解决这些问题，正如我们迄今为止所看到的示例，它们在缺失值出现时仅使用 `pd.NA` 并且没有进行隐式类型转换，做得相当出色。然而，*pandas 扩展类型* 被作为可选类型引入，而不是默认类型，因此 pandas 为处理缺失值所开发的自定义解决方案在代码中仍然占主导地位。遗憾的是，由于这些不一致性从未得到纠正，用户必须理解他们所选择的数据类型以及不同数据类型如何处理缺失值。

尽管存在这些不一致之处，幸运的是 pandas 提供了一个 `pd.isna` 函数，可以告诉你数组中的某个元素是否缺失。它适用于默认数据类型：

```py
`pd.isna(pd.Series([1, np.nan, 2]))` 
```

```py
`0    False 1     True 2    False dtype: bool` 
```

它与 *pandas 扩展类型* 同样有效：

```py
`pd.isna(pd.Series([1, pd.NA, 2], dtype=pd.Int64Dtype()))` 
```

```py
`0    False 1     True 2    False dtype: bool` 
```

## 还有更多内容…

用户应当注意，与 `np.nan` 和 `pd.NA` 进行比较时，它们的行为是不同的。例如，`np.nan == np.nan` 返回 `False`，而 `pd.NA == pd.NA` 返回 `pd.NA`。前者的比较遵循 IEEE 757 标准，而 `pd.NA` 哨兵遵循 Kleene 逻辑。

`pd.NA` 的工作方式允许在 pandas 中进行更具表现力的掩码/选择。例如，如果你想创建一个也包含缺失值的布尔掩码并用它来选择值，`pd.BooleanDtype` 使你能够做到这一点，并且自然地只会选择掩码为 `True` 的记录：

```py
`ser = pd.Series(range(3), dtype=pd.Int64Dtype()) mask = pd.Series([True, pd.NA, False], dtype=pd.BooleanDtype()) ser[mask]` 
```

```py
`0    0 dtype: Int64` 
```

如果没有布尔扩展类型，相应的操作将引发错误：

```py
`mask = pd.Series([True, None, False]) ser[mask]` 
```

```py
`ValueError: Cannot mask with non-boolean array containing NA / NaN values` 
```

因此，在不使用`pd.BooleanDtype`的代码中，你可能会看到许多方法调用，将“缺失”值替换为`False`，然后使用`pd.Series.astype`尝试在填充后将其转换回布尔数据类型：

```py
`mask = pd.Series([True, None, False]) mask = mask.fillna(False).astype(bool) ser[mask]` 
```

```py
``/tmp/ipykernel_45649/2987852505.py:2: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`  mask = mask.fillna(False).astype(bool) 0    0 dtype: Int64`` 
```

这种方法不必要地复杂且低效。使用`pd.BooleanDtype`能更简洁地表达你的操作意图，让你更少担心 pandas 的细微差别。

# 分类类型

分类数据类型的主要目的是定义你的`pd.Series`可以包含的可接受域值集合。*第四章*中的*CSV - 读取大文件的策略*食谱将向你展示一个例子，其中这可能导致显著的内存节省，但通常这里的使用案例是让 pandas 将诸如`foo`、`bar`和`baz`等字符串值转换为代码`0`、`1`和`2`，这些代码可以更高效地存储。

## 如何做到

到目前为止，我们总是选择`pd.XXDtype()`作为`dtype=`参数，这在分类数据类型的情况下仍然*可能*有效，但不幸的是，它没有一致地处理缺失值（详见*还有更多……*，深入探讨这个问题）。因此，我们必须选择两种替代方法中的一种，使用`pd.NA`缺失值指示符来创建`pd.CategoricalDtype`。

无论哪种方法，你都需要从一个使用`pd.StringDtype`的数据`pd.Series`开始：

```py
`values = ["foo", "bar", "baz"] values_ser = pd.Series(values, dtype=pd.StringDtype())` 
```

从那里，你可以使用`pd.DataFrame.astype`将其转换为分类类型：

```py
`ser = values_ser.astype(pd.CategoricalDtype()) ser` 
```

```py
`0    foo 1    bar 2    baz dtype: category Categories (3, string): [bar, baz, foo]` 
```

或者，如果你需要对分类类型的行为有更多控制，你可以从你的`pd.Series`值构造`pd.CategoricalDtype`，并随后将其用作`dtype=`参数：

```py
`cat = pd.CategoricalDtype(values_ser) ser = pd.Series(values, dtype=cat) ser` 
```

```py
`0    foo 1    bar 2    baz dtype: category Categories (3, string): [foo, bar, baz]` 
```

两种方法最终都能达到相同的目的，尽管第二种方法在构造`pd.CategoricalDtype`时牺牲了一些冗长性，以换取对其行为的更细致控制，正如你将在本食谱的其余部分看到的那样。

无论你采用何种方法，都应该注意，在构造分类类型`pd.Series`时所使用的值定义了可以使用的可接受域值的集合。鉴于我们用`["foo", "bar", "baz"]`创建了我们的分类类型，随后使用这些值进行赋值并没有问题：

```py
`ser.iloc[2] = "foo" ser` 
```

```py
`0    foo 1    bar 2    foo dtype: category Categories (3, string): [foo, bar, baz]` 
```

然而，赋值超出该域范围会引发错误：

```py
`ser.iloc[2] = "qux"` 
```

```py
`TypeError: Cannot setitem on a Categorical with a new category (qux), set the categories first` 
```

当显式构造`pd.CategoricalDtype`时，可以通过`ordered=`参数为值分配非字典序的顺序。这在处理*顺序*数据时非常宝贵，因为这些数据的值并不是按你想要的方式由计算机算法自然排序的。

作为一个实际例子，我们来考虑一下服装尺码的使用案例。自然，小号服装比中号服装小，中号服装比大号服装小，依此类推。通过按顺序构造`pd.CategoricalDtype`并使用`ordered=True`，pandas 使得比较尺码变得非常自然：

```py
`shirt_sizes = pd.Series(["S", "M", "L", "XL"], dtype=pd.StringDtype()) cat = pd.CategoricalDtype(shirt_sizes, ordered=True) ser = pd.Series(["XL", "L", "S", "L", "S", "M"], dtype=cat) ser < "L"` 
```

```py
`0    False 1    False 2     True 3    False 4     True 5     True dtype: bool` 
```

那么，pandas 是如何做到这么简单高效的呢？pandas 库暴露了一个类别访问器`pd.Series.cat`，它可以帮助你更深入地理解这一点。为了进一步探索，让我们首先创建一个`pd.Series`类别数据，其中某一类别被多次使用：

```py
`accepted_values = pd.Series(["foo", "bar"], dtype=pd.StringDtype()) cat = pd.CategoricalDtype(accepted_values) ser = pd.Series(["foo", "bar", "foo"], dtype=cat) ser` 
```

```py
`0    foo 1    bar 2    foo dtype: category Categories (2, string): [foo, bar]` 
```

如果你检查`pd.Series.cat.codes`，你会看到一个大小相同的`pd.Series`，但值`foo`被替换为数字`0`，值`bar`被替换为数字`1`：

```py
`ser.cat.codes` 
```

```py
`0    0 1    1 2    0 dtype: int8` 
```

另外，`pd.Series.cat.categories`将包含每个类别的值，按顺序排列：

```py
`ser.cat.categories` 
```

```py
`Index(['foo', 'bar'], dtype='string')` 
```

除去一些内部细节，你可以将 pandas 视为创建了一个形式为`{0: "foo", 1: "bar"}`的字典。虽然它内部存储着一个值为`[0, 1, 0]`的`pd.Series`，但当需要显示或以任何方式访问这些值时，这些值会像字典中的键一样被用来访问最终用户想要使用的真实值。因此，你会经常看到类别数据类型被描述为`字典`类型（例如，Apache Arrow 就使用了“字典”这个术语）。

那么，为什么要费心呢？将标签*编码*成非常小的整数查找值的过程，可能会对内存使用产生显著影响。请注意与普通字符串类型之间的内存使用差异：

```py
`pd.Series(["foo", "bar", "baz"] * 100, dtype=pd.StringDtype()).memory_usage()` 
```

```py
`2528` 
```

与等效的类别类型相比，如下所示：

```py
`pd.Series(["foo", "bar", "baz"] * 100, dtype=cat).memory_usage()` 
```

```py
`552` 
```

你的数字可能和`.memory_usage()`的输出完全不一致，但至少你应该会看到，在使用类别数据类型时，内存使用量有明显的减少。

## 还有更多…

如果直接使用`dtype=pd.CategoricalDtype()`有效，为什么用户不想使用它呢？不幸的是，pandas API 中存在一个较大的空白，导致缺失值无法在类别类型之间传播，这可能会意外地引入我们在*缺失值处理*方法中警告过的`np.nan`缺失值指示符。这可能会导致非常令人惊讶的行为，即使你认为自己已经正确使用了`pd.NA`哨兵值：

```py
`pd.Series(["foo", "bar", pd.NA], dtype=pd.CategoricalDtype())` 
```

```py
`0    foo 1    bar 2    NaN dtype: category Categories (2, object): ['bar', 'foo']` 
```

请注意，在前面的示例中，我们尝试提供`pd.NA`但*仍然*返回了`np.nan`？从`dtype=pd.StringDtype()`构造的`pd.Series`显式构建`pd.CategoricalDtype`帮助我们避免了这种令人惊讶的行为：

```py
`values = pd.Series(["foo", "bar"], dtype=pd.StringDtype()) cat = pd.CategoricalDtype(values) pd.Series(["foo", "bar", pd.NA], dtype=cat)` 
```

```py
`0     foo 1     bar 2    <NA> dtype: category Categories (2, string): [foo, bar]` 
```

如果你发现这种行为令人困惑或麻烦，相信你并不孤单。隧道尽头的曙光可能是 PDEP-16，它旨在让`pd.NA`仅作为缺失值指示符使用。这意味着你可以直接使用`pd.CategoricalDtype()`构造函数，并遵循直到此时为止所看到的所有相同模式。

不幸的是，这本书是在 pandas 3.0 发布的时候发布的，而且在 PDEP-16 被正式接受之前，因此很难预测这些 API 中的不一致何时会消失。如果你是在本书出版几年后阅读的，请务必查看 PDEP-16 的状态，因为它可能会改变构造分类数据的正确方式（以及其他数据类型）。

# 时间类型 – 日期时间

“时间”一词通常包含那些涉及日期和时间的数据类型，既包括绝对时间，也包括衡量两点之间持续时间的情况。时间类型是基于时间序列分析的关键支持，它对趋势检测和预测模型至关重要。事实上，pandas 最初是在一家资本管理公司开发的，随后才开源。pandas 内置的许多时间序列处理功能，受到金融和经济行业实际报告需求的影响。

尽管*分类类型*部分开始展示了 pandas 类型系统 API 中的一些不一致，时间类型更是将这些问题推向了一个新的层次。合理的预期是，`pd.DatetimeDtype()`应该作为构造函数存在，但不幸的是，至少在写作时并非如此。此外，正如*缺失值处理*一节所提到的，时间类型是在 pandas 类型扩展系统之前实现的，使用了不同的缺失值指示符`pd.NaT`（即，“不是一个时间”）。

尽管存在这些问题，pandas 仍然提供了令人惊讶的高级功能来处理时间数据。*第九章*，*时间数据类型与算法*，将深入探讨这些数据类型的应用；现在，我们只提供一个快速概述。

## 如何操作

与许多数据库系统提供单独的`DATE`和`DATETIME`或`TIMESTAMP`数据类型不同，pandas 只有一种“日期时间”类型，可以通过`dtype=`参数的`"datetime64[<unit>]"`形式进行构造。

在 pandas 的历史大部分时间里，`ns`是唯一被接受的`<unit>`值，因此我们暂时从它开始（但请查看*还有更多……*，了解不同值的详细解释）：

```py
`ser = pd.Series([     "2024-01-01 00:00:00",     "2024-01-02 00:00:01",     "2024-01-03 00:00:02" ], dtype="datetime64[ns]") ser` 
```

```py
`0   2024-01-01 00:00:00 1   2024-01-02 00:00:01 2   2024-01-03 00:00:02 dtype: datetime64[ns]` 
```

你也可以使用不包含时间组件的字符串参数构造一个`pd.Series`数据类型：

```py
`ser = pd.Series([     "2024-01-01",     "2024-01-02",     "2024-01-03" ], dtype="datetime64[ns]") ser` 
```

```py
`0   2024-01-01 1   2024-01-02 2   2024-01-03 dtype: datetime64[ns]` 
```

上述构造的输出略显误导；虽然时间戳没有显示，pandas 仍然将这些值内部存储为日期时间，而不是日期。这可能是一个问题，因为没有办法阻止后续的时间戳被存储在那个`pd.Series`中：

```py
`ser.iloc[1] = "2024-01-04 00:00:42" ser` 
```

```py
`0   2024-01-01 00:00:00 1   2024-01-04 00:00:42 2   2024-01-03 00:00:00 dtype: datetime64[ns]` 
```

如果保留日期很重要，请务必稍后阅读本章中的*时间 PyArrow 类型*一节。

就像我们在字符串类型中看到的那样，包含日期时间数据的`pd.Series`会有一个*访问器*，它解锁了处理日期和时间的灵活功能。在这种情况下，访问器是`pd.Series.dt`。

我们可以使用这个访问器来确定`pd.Series`中每个元素的年份：

```py
`ser.dt.year` 
```

```py
`0    2024 1    2024 2    2024 dtype: int32` 
```

`pd.Series.dt.month`将返回月份：

```py
`ser.dt.month` 
```

```py
`0    1 1    1 2    1 dtype: int32` 
```

`pd.Series.dt.day`提取日期所在的月日：

```py
`ser.dt.day` 
```

```py
`0    1 1    4 2    3 dtype: int32` 
```

还有一个`pd.Series.dt.day_of_week`函数，它会告诉你一个日期是星期几。星期一为`0`，依此类推，直到`6`，表示星期日：

```py
`ser.dt.day_of_week` 
```

```py
`0    0 1    3 2    2 dtype: int32` 
```

如果你曾经处理过时间戳（尤其是在全球化组织中），你可能会问这些值代表的是哪个时间。2024-01-03 00:00:00 在纽约市发生的时间并不会与 2024-01-03 00:00:00 在伦敦或上海同时发生。那么，我们如何获得*真正*的时间表示呢？

我们之前看到的时间戳被视为*无时区感知*的（即，它们并未清楚地表示地球上任何一个时刻）。相比之下，你可以通过在`dtype=`参数中指定时区，使你的时间戳变为*有时区感知*。

奇怪的是，pandas 确实有一个`pd.DatetimeTZDtype()`，因此我们可以结合`tz=`参数来指定假定事件发生的时区。例如，要使你的时间戳表示为 UTC，你可以执行以下操作：

```py
`pd.Series([     "2024-01-01 00:00:01",     "2024-01-02 00:00:01",     "2024-01-03 00:00:01" ], dtype=pd.DatetimeTZDtype(tz="UTC"))` 
```

```py
`0   2024-01-01 00:00:01+00:00 1   2024-01-02 00:00:01+00:00 2   2024-01-03 00:00:01+00:00 dtype: datetime64[ns, UTC]` 
```

字符串 UTC 表示的是**互联网号码分配局**（**IANA**）的时区标识符。你可以使用任何这些标识符作为`tz=`参数，如`America/New_York`：

```py
`pd.Series([     "2024-01-01 00:00:01",     "2024-01-02 00:00:01",     "2024-01-03 00:00:01" ], dtype=pd.DatetimeTZDtype(tz="America/New_York"))` 
```

```py
`0   2024-01-01 00:00:01-05:00 1   2024-01-02 00:00:01-05:00 2   2024-01-03 00:00:01-05:00 dtype: datetime64[ns, America/New_York]` 
```

如果你不想使用时区标识符，也可以选择指定一个 UTC 偏移量：

```py
`pd.Series([     "2024-01-01 00:00:01",     "2024-01-02 00:00:01",     "2024-01-03 00:00:01" ], dtype=pd.DatetimeTZDtype(tz="-05:00"))` 
```

```py
`0   2024-01-01 00:00:01-05:00 1   2024-01-02 00:00:01-05:00 2   2024-01-03 00:00:01-05:00 dtype: datetime64[ns, UTC-05:00]` 
```

我们在本节中介绍的`pd.Series.dt`访问器也具有一些非常适用于时区操作的功能。例如，如果你正在处理的数据技术上没有时区信息，但你知道这些时间实际上代表的是美国东部时间，`pd.Series.dt.tz_localize`可以帮助你表达这一点：

```py
`ser_no_tz = pd.Series([     "2024-01-01 00:00:00",     "2024-01-01 00:01:10",     "2024-01-01 00:02:42" ], dtype="datetime64[ns]") ser_et = ser_no_tz.dt.tz_localize("America/New_York") ser_et` 
```

```py
`0   2024-01-01 00:00:00-05:00 1   2024-01-01 00:01:10-05:00 2   2024-01-01 00:02:42-05:00 dtype: datetime64[ns, America/New_York]` 
```

你还可以使用`pd.Series.dt.tz_convert`将时间转换为其他时区：

```py
`ser_pt = ser_et.dt.tz_convert("America/Los_Angeles") ser_pt` 
```

```py
`0   2023-12-31 21:00:00-08:00 1   2023-12-31 21:01:10-08:00 2   2023-12-31 21:02:42-08:00 dtype: datetime64[ns, America/Los_Angeles]` 
```

你甚至可以使用`pd.Series.dt.normalize`将所有的日期时间数据设定为所在时区的午夜。如果你根本不关心日期时间的时间部分，只想将其视为日期，这会很有用，尽管 pandas 并没有提供一等的`DATE`类型：

```py
`ser_pt.dt.normalize()` 
```

```py
`0   2023-12-31 00:00:00-08:00 1   2023-12-31 00:00:00-08:00 2   2023-12-31 00:00:00-08:00 dtype: datetime64[ns, America/Los_Angeles]` 
```

尽管我们迄今为止提到的 pandas 在处理日期时间数据时有许多很棒的功能，但我们也应该看看其中一些并不那么出色的方面。在*缺失值处理*部分，我们讨论了如何使用`np.nan`作为 pandas 中的缺失值指示符，尽管更现代的数据类型使用`pd.NA`。对于日期时间数据类型，还有一个额外的缺失值指示符`pd.NaT`：

```py
`ser = pd.Series([     "2024-01-01",     None,     "2024-01-03" ], dtype="datetime64[ns]") ser` 
```

```py
`0   2024-01-01 1          NaT 2   2024-01-03 dtype: datetime64[ns]` 
```

再次强调，这个差异源于时间类型在 pandas 引入扩展类型之前就已经存在，而推动统一缺失值指示符的进展尚未完全实现。幸运的是，像`pd.isna`这样的函数仍然能够正确识别`pd.NaT`作为缺失值：

```py
`pd.isna(ser)` 
```

```py
`0    False 1     True 2    False dtype: bool` 
```

## 还有更多内容……

历史上的`ns`精度限制了 pandas 中的时间戳范围，从稍早于 1677-09-21 开始，到稍晚于 2264-04-11 结束。尝试分配超出这些边界之外的日期时间值将引发`OutOfBoundsDatetime`异常：

```py
`pd.Series([     "1500-01-01 00:00:01",     "2500-01-01 00:00:01", ], dtype="datetime64[ns]")` 
```

```py
`OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 1500-01-01 00:00:01, at position 0` 
```

从 pandas 的 3.0 系列开始，您可以指定低精度，如`s`、`ms`或`us`，以扩展您的时间范围超出这些窗口：

```py
`pd.Series([     "1500-01-01 00:00:01",     "2500-01-01 00:00:01", ], dtype="datetime64[us]")` 
```

```py
`0   1500-01-01 00:00:01 1   2500-01-01 00:00:01 dtype: datetime64[us]` 
```

# 时间类型 – timedelta

Timedelta 非常有用，用于测量两个时间点之间的*持续时间*。这可以用于测量诸如“平均来看，事件 X 和事件 Y 之间经过了多少时间”，这对于监控和预测组织内某些过程或系统的周转时间非常有帮助。此外，timedelta 可以用于操作您的日期时间，轻松实现“添加 X 天”或“减去 Y 秒”，而无需深入了解日期时间对象在内部存储的细节。

## 如何操作

到目前为止，我们已经介绍了每种数据类型通过直接构建它。然而，手工构造 timedelta `pd.Series`的用例非常罕见。更常见的情况是，您会遇到这种类型作为从一个日期时间减去另一个日期时间的表达式的结果：

```py
`ser = pd.Series([     "2024-01-01",     "2024-01-02",     "2024-01-03" ], dtype="datetime64[ns]") ser - pd.Timestamp("2023-12-31 12:00:00")` 
```

```py
`0   0 days 12:00:00 1   1 days 12:00:00 2   2 days 12:00:00 dtype: timedelta64[ns]` 
```

在 pandas 中，还有`pd.Timedelta`标量，可以在表达式中用来添加或减去一个持续时间到日期时间。例如，以下代码展示了如何在`pd.Series`中的每个日期时间上添加 3 天：

```py
`ser + pd.Timedelta("3 days")` 
```

```py
`0   2024-01-04 1   2024-01-05 2   2024-01-06 dtype: datetime64[ns]` 
```

## 更多信息…

虽然不是一个常见的模式，如果您曾经需要手动构建一个 timedelta 对象的`pd.Series`，您可以使用`dtype="timedelta[ns]"`来实现：

```py
`pd.Series([     "-1 days",     "6 hours",     "42 minutes",     "12 seconds",     "8 milliseconds",     "4 microseconds",     "300 nanoseconds", ], dtype="timedelta64[ns]")` 
```

```py
`0           -1 days +00:00:00 1             0 days 06:00:00 2             0 days 00:42:00 3             0 days 00:00:12 4      0 days 00:00:00.008000 5      0 days 00:00:00.000004 6   0 days 00:00:00.000000300 dtype: timedelta64[ns]` 
```

如果我们尝试创建一个以月为单位的 timedelta 呢？我们来看看：

```py
`pd.Series([     "1 months", ], dtype="timedelta64[ns]")` 
```

```py
`ValueError: invalid unit abbreviation: months` 
```

pandas 不允许这样做的原因是，timedelta 表示一个一致可测量的*持续时间*。尽管微秒中始终有 1,000 纳秒，毫秒中始终有 1,000 微秒，秒中始终有 1,000 毫秒，等等，但一个月中的天数并不一致，从 28 到 31 不等。说两个事件*相隔一个月*并不能满足 timedelta 测量时间段的严格要求。

如果您需要按照日历而不是有限的持续时间来移动日期，您仍然可以使用我们在*第九章* *时间数据类型和算法*中将介绍的`pd.DateOffset`对象。虽然这本章节中没有相关的数据类型介绍，但该对象本身可以作为 timedelta 类型的一个很好的补充或增强，用于那些不严格将时间视为有限持续时间的分析。

# PyArrow 的时间类型

到目前为止，我们已经回顾了内置到 pandas 中的许多“一流”数据类型，同时也突出了困扰它们的一些粗糙边缘和不一致性。尽管存在这些问题，内置到 pandas 中的这些类型可以在数据旅程中为您提供很长一段路。

但仍然有一些情况下，pandas 类型不适用，常见的情况是与数据库的互操作性。大多数数据库有独立的`DATE`和`DATETIME`类型，因此，pandas 只提供`DATETIME`类型可能让熟悉 SQL 的用户感到失望。

幸运的是，Apache Arrow 项目定义了一个真正的`DATE`类型。从 2.0 版本开始，pandas 用户可以开始利用通过 PyArrow 库暴露的 Arrow 类型。

## 如何实现

要在 pandas 中直接构造 PyArrow 类型，你总是需要提供`dtype=`参数，格式为`pd.ArrowDtype(XXX)`，并用适当的 PyArrow 类型替换`XXX`。PyArrow 中的 DATE 类型是`pa.date32()`：

```py
`ser = pd.Series([     "2024-01-01",     "2024-01-02",     "2024-01-03", ], dtype=pd.ArrowDtype(pa.date32())) ser` 
```

```py
`0    2024-01-01 1    2024-01-02 2    2024-01-03 dtype: date32[day][pyarrow]` 
```

`pa.date32()`类型可以表示更广泛的日期范围，而不需要切换精度：

```py
`ser = pd.Series([     "9999-12-29",     "9999-12-30",     "9999-12-31", ], dtype=pd.ArrowDtype(pa.date32())) ser` 
```

```py
`0    9999-12-29 1    9999-12-30 2    9999-12-31 dtype: date32[day][pyarrow]` 
```

PyArrow 库提供了一个时间戳类型；然而，它的功能几乎与您已经看到的 datetime 类型完全相同，因此我建议使用 pandas 内置的 datetime 类型。

# PyArrow 列表类型

如果你遇到的每一项数据都恰好适合并整齐地放在`pd.DataFrame`的单个位置，生活将会变得如此简单，但不可避免地，你会遇到那些情况，数据并不总是这样。让我们先假设尝试分析一家公司中工作的员工：

```py
`df = pd.DataFrame({     "name": ["Alice", "Bob", "Janice", "Jim", "Michael"],     "years_exp": [10, 2, 4, 8, 6], }) df` 
```

```py
 `name      years_exp 0    Alice     10 1    Bob       2 2    Janice    4 3    Jim       8 4    Michael   6` 
```

这种类型的数据相对容易处理——你可以轻松地计算出每个员工的工作经验年数总和或平均值。但如果我们还想知道，Bob 和 Michael 向 Alice 汇报，而 Janice 向 Jim 汇报怎么办？

我们对世界的美好视角突然崩塌——我们怎么可能在`pd.DataFrame`中表达这个呢？如果你来自 Microsoft Excel 或 SQL 背景，你可能会想你需要创建一个单独的`pd.DataFrame`来存储直接汇报信息。但在 pandas 中，我们可以通过使用 PyArrow 的`pa.list_()`数据类型，更自然地表达这一点。

## 如何实现

在处理`pa.list_()`类型时，您必须*参数化*它，指定它将包含的元素的数据类型。在我们的例子中，我们希望列表包含类似`Bob`和`Janice`这样的值，因此我们将使用`pa.string()`类型对`pa.list_()`类型进行参数化：

```py
`ser = pd.Series([     ["Bob", "Michael"],     None,     None,     ["Janice"],     None, ], dtype=pd.ArrowDtype(pa.list_(pa.string()))) df["direct_reports"] = ser df` 
```

```py
 `name      years_exp    direct_reports 0    Alice     10           ['Bob' 'Michael'] 1    Bob       2            <NA> 2    Janice    4            <NA> 3    Jim       8            ['Janice'] 4    Michael   6            <NA>` 
```

## 还有更多……

在处理具有 PyArrow 列表类型的`pd.Series`时，你可以通过使用`.list`访问器解锁更多`pd.Series`的功能。例如，要查看列表中包含多少项，可以调用`ser.list.len()`：

```py
`ser.list.len()` 
```

```py
`0       2 1    <NA> 2    <NA> 3       1 4    <NA> dtype: int32[pyarrow]` 
```

你可以使用`.list[]`语法访问列表中给定位置的项：

```py
`ser.list[0]` 
```

```py
`0       Bob 1      <NA> 2      <NA> 3    Janice 4      <NA> dtype: string[pyarrow]` 
```

还有一个`.list.flatten`访问器，它可以帮助你识别所有向某人汇报的员工：

```py
`ser.list.flatten()` 
```

```py
`0        Bob 1    Michael 2     Janice dtype: string[pyarrow]` 
```

# PyArrow 十进制类型

当我们在本章早些时候查看 *浮点数类型* 示例时，我们提到的一个重要内容是浮点数类型是 *不精确的*。大多数计算机软件的用户可能一生都不会知道这个事实，在许多情况下，精度的缺失可能是为了获得浮点数类型所提供的性能而可接受的折衷。然而，在某些领域，**精确**的计算是至关重要的。

举一个简单的例子，假设一个电影推荐系统使用浮点算术计算某部电影的评分为 4.3334（满分 5 星），而实际应为 4.33337。即使这个四舍五入误差重复了一百万次，可能也不会对文明产生很大负面影响。相反，一个每天处理数十亿交易的金融系统会认为这种四舍五入误差是无法接受的。随着时间的推移，这个误差会积累成一个相当大的数值。

十进制数据类型是解决这些问题的方案。通过放弃一些浮点计算带来的性能，十进制值允许你实现更精确的计算。

## 如何实现

`pa.decimal128()` 数据类型需要两个参数，这两个参数定义了你希望表示的数字的 *精度* 和 *小数位*。精度决定了可以安全存储多少位小数，而小数位则表示这些小数位中有多少位出现在小数点后。

例如，当 *精度* 为 5 和 *小数位* 为 2 时，你可以准确表示 -999.99 到 999.99 之间的数字，而精度为 5 且小数位为 0 时，表示的范围是 -99999 到 99999。实际上，你选择的精度通常会更高。

下面是如何在 `pd.Series` 中表示这一点的示例：

```py
`pd.Series([     "123456789.123456789",     "-987654321.987654321",     "99999999.9999999999", ], dtype=pd.ArrowDtype(pa.decimal128(19, 10)))` 
```

```py
`0     123456789.1234567890 1    -987654321.9876543210 2      99999999.9999999999 dtype: decimal128(19, 10)[pyarrow]` 
```

特别需要注意的是，我们将数据提供为字符串。如果我们一开始尝试提供浮点数数据，我们会立即看到精度丢失：

```py
`pd.Series([     123456789.123456789,     -987654321.987654321,     99999999.9999999999, ], dtype=pd.ArrowDtype(pa.decimal128(19, 10)))` 
```

```py
`0     123456789.1234567910 1    -987654321.9876543283 2     100000000.0000000000 dtype: decimal128(19, 10)[pyarrow]` 
```

这发生是因为 Python 默认使用浮点数存储实数，因此当语言运行时尝试解释你提供的数字时，四舍五入误差就会出现。根据你的平台，你甚至可能会发现 `99999999.9999999999 == 100000000.0` 返回 `True`。对于人类读者来说，这显然不是真的，但计算机存储的限制使得语言无法辨别这一点。

Python 解决这个问题的方法是使用 `decimal` 模块，确保不会发生四舍五入误差：

```py
`import decimal decimal.Decimal("99999999.9999999999") == decimal.Decimal("100000000.0")` 
```

```py
`False` 
```

同时，你仍然可以进行正确的算术运算，如下所示：

```py
`decimal.Decimal("99999999.9999999999") + decimal.Decimal("100000000.0")` 
```

```py
`Decimal('199999999.9999999999')` 
```

`decimal.Decimal` 对象在构建 PyArrow 十进制类型时也是有效的参数：

```py
`pd.Series([     decimal.Decimal("123456789.123456789"),     decimal.Decimal("-987654321.987654321"),     decimal.Decimal("99999999.9999999999"), ], dtype=pd.ArrowDtype(pa.decimal128(19, 10)))` 
```

```py
`0     123456789.1234567890 1    -987654321.9876543210 2      99999999.9999999999 dtype: decimal128(19, 10)[pyarrow]` 
```

## 还有更多内容……

`pa.decimal128` 数据类型最多支持 38 位有效数字。如果你需要更高的精度，Arrow 生态系统还提供了 `pa.decimal256` 数据类型：

```py
`ser = pd.Series([     "123456789123456789123456789123456789.123456789" ], dtype=pd.ArrowDtype(pa.decimal256(76, 10))) ser` 
```

```py
`0    123456789123456789123456789123456789.1234567890 dtype: decimal256(76, 10)[pyarrow]` 
```

只需注意，这将消耗是 `pa.decimal128` 数据类型两倍的内存，且可能会有更慢的计算时间。

# NumPy 类型系统、对象类型和陷阱

如本章介绍所提到的，至少在 2.x 和 3.x 系列中，pandas 仍然默认使用对一般数据分析并不理想的数据类型。然而，你无疑会在同伴的代码或在线代码片段中遇到它们，因此理解它们的工作原理、潜在的陷阱以及如何避免它们，将在未来几年中变得非常重要。

## 如何做到这一点

让我们看一下从整数序列构造 `pd.Series` 的默认方式：

```py
`pd.Series([0, 1, 2])` 
```

```py
`0    0 1    1 2    2 dtype: int64` 
```

从这个参数开始，pandas 给我们返回了一个 `pd.Series`，它的 `dtype` 是 `int64`。这看起来很正常，那到底有什么问题呢？好吧，让我们看看引入缺失值时会发生什么：

```py
`pd.Series([0, None, 2])` 
```

```py
`0    0.0 1    NaN 2    2.0 dtype: float64` 
```

嗯？我们提供了整数数据，但现在却得到了浮点类型。指定 `dtype=` 参数肯定能帮我们解决这个问题吧：

```py
`pd.Series([0, None, 2], dtype=int)` 
```

```py
`TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'` 
```

无论你多么努力，你就是*不能*将缺失值与 pandas 默认返回的 NumPy 整数数据类型混合。解决这个模式的常见方法是，在使用 `pd.Series.astype` 转回实际整数数据类型之前，先用另一个值（如 `0`）填充缺失值：

```py
`ser = pd.Series([0, None, 2]) ser.fillna(0).astype(int)` 
```

```py
`0    0 1    0 2    2 dtype: int64` 
```

这解决了让我们得到正确整数类型的问题，但它必须改变数据才能做到这一点。是否在意这个变化是一个依赖于上下文的问题；有些用户如果只是想对这一列进行*求和*，可能会接受将缺失值当作 0 处理，但同样的用户可能不满意这种数据所产生的新*计数*和*平均值*。

请注意此`fillna`方法与本章开始时介绍的 pandas 扩展类型之间的区别：

```py
`pd.Series([0, None, 2]).fillna(0).astype(int).mean()` 
```

```py
`0.6666666666666666` 
```

```py
`pd.Series([0, None, 2], dtype=pd.Int64Dtype()).mean()` 
```

```py
`1.0` 
```

不仅得到了不同的结果，而且我们不使用 `dtype=pd.Int64Dtype()` 的方法需要更长时间来计算：

```py
`import timeit func = lambda: pd.Series([0, None, 2]).fillna(0).astype(int).mean() timeit.timeit(func, number=10_000)` 
```

```py
`0.9819313539992436` 
```

```py
`func = lambda: pd.Series([0, None, 2], dtype=pd.Int64Dtype()).mean() timeit.timeit(func, number=10_000)` 
```

```py
`0.6182142379984725` 
```

当你考虑到你必须经历的步骤，只为了获得整数而非浮点数时，这也许并不令人惊讶。

当你查看 pandas 中历史上的布尔数据类型时，事情变得更加怪异。让我们再次从看似合理的基本案例开始：

```py
`pd.Series([True, False])` 
```

```py
`0     True 1    False dtype: bool` 
```

让我们通过引入缺失值来打乱一些事情：

```py
`pd.Series([True, False, None])` 
```

```py
`0     True 1    False 2     None dtype: object` 
```

这是我们第一次看到 `object` 数据类型。撇开一些技术细节，你应该相信 `object` 数据类型是 pandas 中最差的数据类型之一。基本上，`object` 数据类型几乎可以存储任何内容；它完全禁止类型系统对你的数据进行任何强制要求。即使我们只是想存储 `True` 和 `False` 的值，其中一些可能是缺失的，实际上任何有效的值都可以与这些值一起存储：

```py
`pd.Series([True, False, None, "one of these things", ["is not like"], ["the other"]])` 
```

```py
`0                   True 1                  False 2                   None 3    one of these things 4          [is not like] 5            [the other] dtype: object` 
```

所有这些混乱都可以通过使用 `pd.BooleanDtype` 来避免：

```py
`pd.Series([True, False, None], dtype=pd.BooleanDtype())` 
```

```py
`0     True 1    False 2     <NA> dtype: boolean` 
```

另一个相当不幸的事实是，默认的 pandas 实现（至少在 2.x 系列中）将 `object` 数据类型用于字符串：

```py
`pd.Series(["foo", "bar", "baz"])` 
```

```py
`0    foo 1    bar 2    baz dtype: object` 
```

再次强调，这里并没有严格强制要求我们必须拥有字符串数据：

```py
`ser = pd.Series(["foo", "bar", "baz"]) ser.iloc[2] = 42 ser` 
```

```py
`0    foo 1    bar 2     42 dtype: object` 
```

使用`pd.StringDtype()`时，这种类型的赋值将会引发错误：

```py
`ser = pd.Series(["foo", "bar", "baz"], dtype=pd.StringDtype()) ser.iloc[2] = 42` 
```

```py
`TypeError: Cannot set non-string value '42' into a StringArray.` 
```

## 还有更多……

在本章中，我们详细讨论了`object`数据类型缺乏类型强制的问题。另一方面，在某些使用场景中，拥有这种灵活性可能是有帮助的，尤其是在与 Python 对象交互时，在这种情况下，您无法事先对数据做出断言：

```py
`alist = [42, "foo", ["sub", "list"], {"key": "value"}] ser = pd.Series(alist) ser` 
```

```py
`0                  42 1                 foo 2         [sub, list] 3    {'key': 'value'} dtype: object` 
```

如果您曾经使用过类似 Microsoft Excel 的工具，您可能会觉得将任何值以几乎任何格式放入任何地方的想法并不新奇。另一方面，如果您的经验更多来自于使用 SQL 数据库，您可能会觉得将*任何*数据加载进来是一个陌生的概念。

在数据处理领域，主要有两种方法：**提取、转换、加载**（**ETL**）和**提取、加载、转换**（**ELT**）。ETL 要求您在将数据加载到数据分析工具之前先进行*转换*，这意味着所有清理工作必须提前在其他工具中完成。

ELT 方法允许您先加载数据，稍后再处理清理工作；`object`数据类型使您能够在 pandas 中使用 ELT 方法，如果您选择这样做的话。

话虽如此，我通常建议您将`object`数据类型严格用作`staging`数据类型，然后再将其转换为更具体的类型。通过避免使用`object`数据类型，您将获得更高的性能，更好地理解您的数据，并能够编写更简洁的代码。

在本章的最后一点，当您直接使用`pd.Series`构造函数并指定`dtype=`参数时，控制数据类型是相当容易的。虽然`pd.DataFrame`也有`dtype=`参数，但它不允许您为每一列指定类型，这意味着您通常会在创建`pd.DataFrame`时使用历史的 NumPy 数据类型：

```py
`df = pd.DataFrame([     ["foo", 1, 123.45],     ["bar", 2, 333.33],     ["baz", 3, 999.99], ], columns=list("abc")) df` 
```

```py
 `a     b   c 0   foo   1   123.45 1   bar   2   333.33 2   baz   3   999.99` 
```

检查`pd.DataFrame.dtypes`将帮助我们确认这一点：

```py
`df.dtypes` 
```

```py
`a     object b      int64 c    float64 dtype: object` 
```

为了让我们开始使用更理想的 pandas 扩展类型，我们可以显式使用`pd.DataFrame.astype`方法：

```py
`df.astype({     "a": pd.StringDtype(),     "b": pd.Int64Dtype(),     "c": pd.Float64Dtype(), }).dtypes` 
```

```py
`a    string[python] b             Int64 c           Float64 dtype: object` 
```

或者，我们可以使用`pd.DataFrame.convert_dtypes`方法并设置`dtype_backend="numpy_nullable"`：

```py
`df.convert_dtypes(dtype_backend="numpy_nullable").dtypes` 
```

```py
`a    string[python] b             Int64 c           Float64 dtype: object` 
```

`numpy_nullable`这个术语在 pandas 的历史上有些不准确，但正如我们在介绍中提到的，它是后来被称为 pandas 扩展类型系统的原始名称。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/pandas`](https://packt.link/pandas)

![](img/QR_Code5040900042138312.png)

# 留下评论！

感谢您从 Packt 出版购买本书——我们希望您喜欢它！您的反馈非常宝贵，能够帮助我们改进和成长。读完本书后，请花一点时间在亚马逊上留下评论；这只需要一分钟，但对像您这样的读者来说，意义重大。

扫描下面的二维码，领取您选择的免费电子书。

`packt.link/NzOWQ`

![](img/QR_Code1474021820358918656.png)
