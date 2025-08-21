

# 第十章：一般使用和性能优化建议

到目前为止，我们已经覆盖了 pandas 库的相当大一部分，同时通过示例应用来强化良好的使用习惯。掌握了这些知识后，你现在已经做好准备，踏入实际工作，并将所学的内容应用到数据分析问题中。

本章将提供一些你在独立工作时应牢记的小窍门和建议。本章介绍的食谱是我在多种经验水平的 pandas 用户中经常看到的常见错误。尽管这些做法出发点良好，但不当使用 pandas 构造会浪费很多性能。当数据集较小时，这可能不是大问题，但数据通常是增长的，而不是缩小。使用正确的惯用法并避免低效代码带来的维护负担，可以为你的组织节省大量时间和金钱。

本章将涵盖以下食谱：

+   避免 `dtype=object`

+   注意数据大小

+   使用矢量化函数代替循环

+   避免修改数据

+   使用字典编码低基数数据

+   测试驱动开发功能

# 避免使用 dtype=object

在 pandas 中使用 `dtype=object` 来存储字符串是最容易出错且效率低下的做法之一。不幸的是，在很长一段时间里，`dtype=object` 是处理字符串数据的唯一方法；直到 1.0 版本发布之前，这个问题才“得到解决”。

我故意将“解决”放在引号中，因为尽管 pandas 1.0 确实引入了 `pd.StringDtype()`，但直到 3.0 版本发布之前，许多构造和 I/O 方法默认并未使用它。实际上，除非你明确告诉 pandas 否则，在 2.x 版本中，你所有的字符串数据都会使用 `dtype=object`。值得一提的是，1.0 引入的 `pd.StringDtype()` 帮助确保你*仅*存储字符串，但直到 3.0 版本发布之前，它并未进行性能优化。

如果你使用的是 pandas 3.0 版本及更高版本，可能仍然会遇到一些旧代码，如 `ser = ser.astype(object)`。通常情况下，这类调用应该被替换为 `ser = ser.astype(pd.StringDtype())`，除非你确实需要在 `pd.Series` 中存储 Python 对象。不幸的是，我们无法真正了解原始意图，所以作为开发者，你应该意识到使用 `dtype=object` 的潜在问题，并学会判断是否可以用 `pd.StringDtype()` 适当地替代它。

## 如何实现

我们在*第三章*《数据类型》中已经讨论过使用 `dtype=object` 的一些问题，但在此重申并扩展这些问题是值得的。

为了做一个简单的比较，我们来创建两个 `pd.Series` 对象，它们的数据相同，一个使用 `object` 数据类型，另一个使用 `pd.StringDtype`：

```py
`ser_obj = pd.Series(["foo", "bar", "baz"] * 10_000, dtype=object) ser_str = pd.Series(["foo", "bar", "baz"] * 10_000, dtype=pd.StringDtype())` 
```

尝试为 `ser_str` 分配一个非字符串值将会失败：

```py
`ser_str.iloc[0] = False` 
```

```py
`TypeError: Cannot set non-string value 'False' into a StringArray.` 
```

相比之下，`pd.Series` 类型的对象会很乐意接受我们的 `Boolean` 值：

```py
`ser_obj.iloc[0] = False` 
```

反过来，这只会让你更难发现数据中的问题。当我们尝试分配非字符串数据时，使用`pd.StringDtype`时，失败的地方是非常明显的。而在使用对象数据类型时，直到你在代码的后面尝试做一些字符串操作（比如大小写转换）时，才可能发现问题：

```py
`ser_obj.str.capitalize().head()` 
```

```py
`0    NaN 1    Bar 2    Baz 3    Foo 4    Bar dtype: object` 
```

pandas 并没有抛出错误，而是决定将我们在第一行的`False`条目设置为缺失值。这样的默认行为可能并不是你想要的，但使用对象数据类型时，你对数据质量的控制会大大减弱。

如果你使用的是 pandas 3.0 及更高版本，当安装了 PyArrow 时，你还会发现`pd.StringDtype`变得显著更快。让我们重新创建我们的`pd.Series`对象来测量这一点：

```py
`ser_obj = pd.Series(["foo", "bar", "baz"] * 10_000, dtype=object) ser_str = pd.Series(["foo", "bar", "baz"] * 10_000, dtype=pd.StringDtype())` 
```

为了快速比较执行时间，让我们使用标准库中的`timeit`模块：

```py
`import timeit timeit.timeit(ser_obj.str.upper, number=1000)` 
```

```py
`2.2286621460007154` 
```

将该运行时间与使用正确的`pd.StringDtype`的情况进行比较：

```py
`timeit.timeit(ser_str.str.upper, number=1000)` 
```

```py
`2.7227514309997787` 
```

不幸的是，3.0 版本之前的用户无法看到任何性能差异，但单单是数据验证的改进就足以让你远离`dtype=object`。

那么，避免使用`dtype=object`的最简单方法是什么？如果你有幸使用 pandas 3.0 及以上版本，你会发现这个数据类型不再那么常见，因为这是该库的自然发展。即便如此，对于仍然使用 pandas 2.x 系列的用户，我建议在 I/O 方法中使用`dtype_backend="numpy_nullable"`参数：

```py
`import io data = io.StringIO("int_col,string_col\n0,foo\n1,bar\n2,baz") data.seek(0) pd.read_csv(data, dtype_backend="numpy_nullable").dtypes` 
```

```py
`int_col                Int64 string_col    string[python] dtype: object` 
```

如果你手动构建一个`pd.DataFrame`，可以使用`pd.DataFrame.convert_dtypes`方法，并配合使用相同的`dtype_backend="numpy_nullable"`参数：

```py
`df = pd.DataFrame([     [0, "foo"],     [1, "bar"],     [2, "baz"], ], columns=["int_col", "string_col"]) df.convert_dtypes(dtype_backend="numpy_nullable").dtypes` 
```

```py
`int_col                Int64 string_col    string[python] dtype: object` 
```

请注意，`numpy_nullable`这个术语有些误导。这个参数本来应该被命名为`pandas_nullable`，甚至直接命名为`pandas`或`nullable`也更合适，但当它首次引入时，它与 NumPy 系统密切相关。随着时间的推移，`numpy_nullable`这个术语保留下来了，但类型已经不再依赖于 NumPy。在本书出版后，可能会有更合适的值来实现相同的行为，基本上是为了找到支持缺失值的 pandas 最佳数据类型。

虽然`dtype=object`最常被*误用*来处理字符串，但它在处理日期时间时也暴露了一些问题。我常常看到新用户写出这样的代码，试图创建一个他们认为是日期的`pd.Series`：

```py
`import datetime ser = pd.Series([     datetime.date(2024, 1, 1),     datetime.date(2024, 1, 2),     datetime.date(2024, 1, 3), ]) ser` 
```

```py
`0    2024-01-01 1    2024-01-02 2    2024-01-03 dtype: object` 
```

虽然这是一种逻辑上可行的方法，但问题在于 pandas 并没有一个真正的*日期*类型。相反，这些数据会使用 Python 标准库中的`datetime.date`类型存储在一个`object`数据类型的数组中。Python 对象的这种不幸用法掩盖了你正在处理日期的事实，因此，尝试使用`pd.Series.dt`访问器时会抛出错误：

```py
`ser.dt.year` 
```

```py
`AttributeError: Can only use .dt accessor with datetimelike values` 
```

在*第三章*，*数据类型*中，我们简要讨论了 PyArrow 的`date32`类型，它会是解决这个问题的一个更原生的方案：

```py
`import datetime ser = pd.Series([     datetime.date(2024, 1, 1),     datetime.date(2024, 1, 2),     datetime.date(2024, 1, 3), ], dtype=pd.ArrowDtype(pa.date32())) ser` 
```

```py
`0    2024-01-01 1    2024-01-02 2    2024-01-03 dtype: date32[day][pyarrow]` 
```

这样就解锁了`pd.Series.dt`属性，可以使用了：

```py
`ser.dt.year` 
```

```py
`0    2024 1    2024 2    2024 dtype: int64[pyarrow]` 
```

我觉得这个细节有些遗憾，希望未来的 pandas 版本能够抽象化这些问题，但无论如何，它们在当前版本中存在，并且可能会存在一段时间。

尽管我已经指出了`dtype=object`的一些缺点，但在处理凌乱数据时，它仍然有其用处。有时，你可能对数据一无所知，需要先检查它，才能做出进一步的决策。对象数据类型为你提供了加载几乎任何数据的灵活性，并且可以应用相同的 pandas 算法。虽然这些算法可能效率不高，但它们仍然为你提供了一种一致的方式来与数据交互和探索数据，最终为你争取了时间，帮助你找出如何最好地清理数据并将其存储为更合适的格式。因此，我认为`dtype=object`最好作为一个暂存区——我不建议将类型保存在其中，但它为你争取时间，以便对数据类型做出断言，可能是一个资产。

# 留意数据大小

随着数据集的增长，你可能会发现必须选择更合适的数据类型，以确保`pd.DataFrame`仍然能适应内存。

在*第三章*，*数据类型*中，我们讨论了不同的整数类型，以及它们在内存使用和容量之间的权衡。当处理像 CSV 和 Excel 文件这样的无类型数据源时，pandas 会倾向于使用过多的内存，而不是选择错误的容量。这种保守的做法可能导致系统内存的低效使用，因此，了解如何优化内存使用可能是加载文件和收到`OutOfMemory`错误之间的关键差异。

## 如何做

为了说明选择合适的数据类型的影响，我们从一个相对较大的`pd.DataFrame`开始，这个 DataFrame 由 Python 整数组成：

```py
`df = pd.DataFrame({     "a": [0] * 100_000,     "b": [2 ** 8] * 100_000,     "c": [2 ** 16] * 100_000,     "d": [2 ** 32] * 100_000, }) df = df.convert_dtypes(dtype_backend="numpy_nullable") df.head()` 
```

```py
 `a    b       c          d 0   0  256  65536  4294967296 1   0  256  65536  4294967296 2   0  256  65536  4294967296 3   0  256  65536  4294967296 4   0  256  65536  4294967296` 
```

对于整数类型，确定每个`pd.Series`需要多少内存是一个相对简单的过程。对于`pd.Int64Dtype`，每条记录是一个 64 位整数，需要 8 个字节的内存。每条记录旁边，`pd.Series`还会关联一个字节，值为 0 或 1，用来表示记录是否缺失。因此，每条记录总共需要 9 个字节，对于每个`pd.Series`中的 100,000 条记录，我们的内存使用量应该为 900,000 字节。`pd.DataFrame.memory_usage`确认这个计算是正确的：

```py
`df.memory_usage()` 
```

```py
`Index       128 a        900000 b        900000 c        900000 d        900000 dtype: int64` 
```

如果你知道应该使用什么类型，可以通过`.astype`显式地为`pd.DataFrame`的列选择更合适的大小：

```py
`df.assign(     a=lambda x: x["a"].astype(pd.Int8Dtype()),     b=lambda x: x["b"].astype(pd.Int16Dtype()),     c=lambda x: x["c"].astype(pd.Int32Dtype()), ).memory_usage()` 
```

```py
`Index       128 a        200000 b        300000 c        500000 d        900000 dtype: int64` 
```

作为一种便捷方式，pandas 可以通过调用 `pd.to_numeric` 来推断更合适的大小。传递 `downcast="signed"` 参数将确保我们继续使用带符号整数，并且我们将继续传递 `dtype_backend="numpy_nullable"` 来确保我们获得适当的缺失值支持：

```py
`df.select_dtypes("number").assign(     **{x: pd.to_numeric(          y, downcast="signed", dtype_backend="numpy_nullable"     ) for x, y in df.items()} ).memory_usage()` 
```

```py
`Index       128 a        200000 b        300000 c        500000 d        900000 dtype: int64` 
```

# 使用向量化函数代替循环

Python 作为一门语言，以其强大的循环能力而著称。无论你是在处理列表还是字典，循环遍历 Python 对象通常是一个相对容易完成的任务，并且能够编写出非常简洁、清晰的代码。

尽管 pandas 是一个 Python 库，但这些相同的循环结构反而成为编写符合 Python 编程习惯且高效代码的障碍。与循环相比，pandas 提供了*向量化计算*，即对 `pd.Series` 中的所有元素进行计算，而无需显式地循环。

## 如何实现

让我们从一个简单的 `pd.Series` 开始，这个 `pd.Series` 是由一个范围构造的：

```py
`ser = pd.Series(range(100_000), dtype=pd.Int64Dtype())` 
```

我们可以使用内置的 `pd.Series.sum` 方法轻松计算求和：

```py
`ser.sum()` 
```

```py
`4999950000` 
```

遍历 `pd.Series` 并积累自己的结果会得到相同的数字：

```py
`result = 0 for x in ser:     result += x result` 
```

```py
`4999950000` 
```

然而，两个代码示例完全不同。使用 `pd.Series.sum` 时，pandas 在低级语言（如 C）中执行元素求和，避免了与 Python 运行时的任何交互。在 pandas 中，我们称之为*向量化*函数。

相反，`for` 循环由 Python 运行时处理，正如你可能知道的那样，Python 比 C 慢得多。

为了提供一些具体的数字，我们可以使用 Python 的 `timeit` 模块进行简单的时间基准测试。我们先从 `pd.Series.sum` 开始：

```py
`timeit.timeit(ser.sum, number=1000)` 
```

```py
`0.04479526499926578` 
```

我们可以将其与 Python 循环进行比较：

```py
`def loop_sum():     result = 0     for x in ser:         result += x timeit.timeit(loop_sum, number=1000)` 
```

```py
`5.392715779991704` 
```

使用循环会导致巨大的性能下降！

通常情况下，你应该使用 pandas 内置的向量化函数来满足大多数分析需求。对于更复杂的应用，使用 `.agg`、`.transform`、`.map` 和 `.apply` 方法，这些方法已经在*第五章，算法及其应用*中讲解过。你应该能避免在 99.99%的分析中使用 `for` 循环；如果你发现自己更频繁地使用它们，可能需要重新考虑你的设计，通常可以通过仔细重读*第五章，算法及其应用*来解决。

这个规则的唯一例外情况是当处理 `pd.GroupBy` 对象时，使用 `for` 循环可能更合适，因为它可以像字典一样高效地进行迭代：

```py
`df = pd.DataFrame({     "column": ["a", "a", "b", "a", "b"],     "value": [0, 1, 2, 4, 8], }) df = df.convert_dtypes(dtype_backend="numpy_nullable") for label, group in df.groupby("column"):     print(f"The group for label {label} is:\n{group}\n")` 
```

```py
`The group for label a is:  column  value 0      a      0 1      a      1 3      a      4 The group for label b is:  column  value 2      b      2 4      b      8` 
```

# 避免修改数据

尽管 pandas 允许你修改数据，但修改的成本会根据数据类型有所不同。在某些情况下，这可能代价高昂，因此你应该尽可能地减少任何必须执行的修改操作。

## 如何实现

在考虑数据变更时，应该尽量在加载数据到 pandas 结构之前进行变更。我们可以通过比较加载到 `pd.Series` 后修改记录所需的时间，轻松地说明性能差异：

```py
`def mutate_after():     data = ["foo", "bar", "baz"]     ser = pd.Series(data, dtype=pd.StringDtype())     ser.iloc[1] = "BAR" timeit.timeit(mutate_after, number=1000)` 
```

```py
`0.041951814011554234` 
```

如果变异事先执行，所需的时间：

```py
`def mutate_before():     data = ["foo", "bar", "baz"]     data[1] = "BAR"     ser = pd.Series(data, dtype=pd.StringDtype()) timeit.timeit(mutate_before, number=1000)` 
```

```py
`0.019495725005981512` 
```

## 还有更多...

你可能会陷入一个技术性的深坑，试图解读在不同版本的 pandas 中变异不同数据类型的影响。然而，从 pandas 3.0 开始，行为变得更加一致，这是由于引入了按需写入（Copy-on-Write），这一点是 PDEP-07 提案的一部分。简单来说，每次你尝试变异`pd.Series`或`pd.DataFrame`时，都会得到原始数据的一个副本。

尽管这种行为现在更容易预测，但也意味着变异可能非常昂贵，特别是如果你尝试变异一个大的`pd.Series`或`pd.DataFrame`。

# 字典编码低基数数据

在*第三章*，*数据类型*中，我们讨论了分类数据类型，它可以通过将字符串（或任何数据类型）替换为更小的整数代码来减少内存使用。虽然*第三章*，*数据类型*提供了一个很好的技术深度讲解，但考虑到在处理*低基数*数据时，这可以带来显著的节省，因此在这里再强调一次作为最佳实践是值得的。*低基数*数据是指唯一值与总记录数的比率相对较低的数据。

## 如何做

为了进一步强调内存节省的观点，假设我们创建一个*低基数*的`pd.Series`。我们的`pd.Series`将有 300,000 行数据，但只有三个唯一值`"foo"`，`"bar"`和`"baz"`：

```py
`values = ["foo", "bar", "baz"] ser = pd.Series(values * 100_000, dtype=pd.StringDtype()) ser.memory_usage()` 
```

```py
`2400128` 
```

仅仅将其更改为分类数据类型，就能大幅提升内存性能：

```py
`cat = pd.CategoricalDtype(values) ser = pd.Series(values * 100_000, dtype=cat) ser.memory_usage()` 
```

```py
`300260` 
```

# 测试驱动开发特点

**测试驱动开发**（简称**TDD**）是一种流行的软件开发实践，旨在提高代码质量和可维护性。总体上，TDD 从开发者编写测试开始，测试描述了对变更的预期功能。测试从失败状态开始，开发者在测试最终通过时，可以确信他们的实现是正确的。

测试通常是代码评审者在考虑代码变更时首先查看的内容（在贡献 pandas 时，测试是必须的！）。在接受了一个有测试的变更后，后续的任何代码变更都会重新运行该测试，确保你的代码库随着时间的推移继续按预期工作。通常，正确构造的测试可以帮助你的代码库扩展，同时在开发新特性时减轻回归的风险。

pandas 库提供了工具，使得你可以通过`pd.testing`模块为你的`pd.Series`和`pd.DataFrame`对象编写测试，我们将在本教程中进行回顾。

## 它是如何工作的

Python 标准库提供了`unittest`模块，用于声明和自动执行测试。创建测试时，通常会创建一个继承自`unittest.TestCase`的类，并在该类中创建方法来对程序行为进行断言。

在下面的代码示例中，`MyTests.test_42` 方法将调用 `unittest.TestCase.assertEqual`，并传入两个参数，`21 * 2` 和 `42`。由于这两个参数在逻辑上是相等的，测试执行将通过：

```py
`import unittest class MyTests(unittest.TestCase):     def test_42(self):         self.assertEqual(21 * 2, 42) def suite():     suite = unittest.TestSuite()     suite.addTest(MyTests("test_42"))     return suite runner = unittest.TextTestRunner() runner.run(suite())` 
```

```py
`. ---------------------------------------------------------------------- Ran 1 test in 0.001s OK <unittest.runner.TextTestResult run=1 errors=0 failures=0>` 
```

现在让我们尝试使用 pandas 遵循相同的执行框架，不过这次我们不是比较 `21 * 2` 和 `42`，而是尝试比较两个 `pd.Series` 对象：

```py
`def some_cool_numbers():     return pd.Series([42, 555, pd.NA], dtype=pd.Int64Dtype()) class MyTests(unittest.TestCase):     def test_cool_numbers(self):         result = some_cool_numbers()         expected = pd.Series([42, 555, pd.NA], dtype=pd.Int64Dtype())         self.assertEqual(result, expected) def suite():     suite = unittest.TestSuite()     suite.addTest(MyTests("test_cool_numbers"))     return suite runner = unittest.TextTestRunner() runner.run(suite())` 
```

```py
`E ====================================================================== ERROR: test_cool_numbers (__main__.MyTests) ---------------------------------------------------------------------- Traceback (most recent call last):  File "/tmp/ipykernel_79586/2361126771.py", line 9, in test_cool_numbers    self.assertEqual(result, expected)  File "/usr/lib/python3.9/unittest/case.py", line 837, in assertEqual    assertion_func(first, second, msg=msg)  File "/usr/lib/python3.9/unittest/case.py", line 827, in _baseAssertEqual    if not first == second:  File "/home/willayd/clones/Pandas-Cookbook-Third-Edition/lib/python3.9/site-packages/pandas/core/generic.py", line 1577, in __nonzero__    raise ValueError( ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all(). ---------------------------------------------------------------------- Ran 1 test in 0.004s FAILED (errors=1) <unittest.runner.TextTestResult run=1 errors=1 failures=0>` 
```

哦……这真是让人吃惊！

这里的根本问题是调用 `self.assertEqual(result, expected)` 执行表达式 `result == expected`。如果该表达式的结果是 `True`，测试将通过；返回 `False` 的表达式将使测试失败。

然而，pandas 重载了 `pd.Series` 的等于运算符，因此它不会返回 `True` 或 `False`，而是返回另一个进行逐元素比较的 `pd.Series`：

```py
`result = some_cool_numbers() expected = pd.Series([42, 555, pd.NA], dtype=pd.Int64Dtype()) result == expected` 
```

```py
`0    True 1    True 2    <NA> dtype: boolean` 
```

由于测试框架不知道如何处理这个问题，你需要使用 `pd.testing` 命名空间中的自定义函数。对于 `pd.Series` 的比较，`pd.testing.assert_series_equal` 是最合适的工具：

```py
`import pandas.testing as tm def some_cool_numbers():     return pd.Series([42, 555, pd.NA], dtype=pd.Int64Dtype()) class MyTests(unittest.TestCase):     def test_cool_numbers(self):         result = some_cool_numbers()         expected = pd.Series([42, 555, pd.NA], dtype=pd.Int64Dtype())         tm.assert_series_equal(result, expected) def suite():     suite = unittest.TestSuite()     suite.addTest(MyTests("test_cool_numbers"))     return suite runner = unittest.TextTestRunner() runner.run(suite())` 
```

```py
`. ---------------------------------------------------------------------- Ran 1 test in 0.001s   OK <unittest.runner.TextTestResult run=1 errors=0 failures=0>` 
```

为了完整性，让我们触发一个故意的失败并查看输出：

```py
`def some_cool_numbers():     return pd.Series([42, 555, pd.NA], dtype=pd.Int64Dtype()) class MyTests(unittest.TestCase):     def test_cool_numbers(self):         result = some_cool_numbers()         expected = pd.Series([42, 555, pd.NA], dtype=pd.Int32Dtype())         tm.assert_series_equal(result, expected) def suite():     suite = unittest.TestSuite()     suite.addTest(MyTests("test_cool_numbers"))     return suite runner = unittest.TextTestRunner() runner.run(suite())` 
```

```py
`F ====================================================================== FAIL: test_cool_numbers (__main__.MyTests) ---------------------------------------------------------------------- Traceback (most recent call last):  File "/tmp/ipykernel_79586/2197259517.py", line 9, in test_cool_numbers    tm.assert_series_equal(result, expected)  File "/home/willayd/clones/Pandas-Cookbook-Third-Edition/lib/python3.9/site-packages/pandas/_testing/asserters.py", line 975, in assert_series_equal    assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")  File "/home/willayd/clones/Pandas-Cookbook-Third-Edition/lib/python3.9/site-packages/pandas/_testing/asserters.py", line 421, in assert_attr_equal    raise_assert_detail(obj, msg, left_attr, right_attr)  File "/home/willayd/clones/Pandas-Cookbook-Third-Edition/lib/python3.9/site-packages/pandas/_testing/asserters.py", line 614, in raise_assert_detail    raise AssertionError(msg) AssertionError: Attributes of Series are different Attribute "dtype" are different [left]:  Int64 [right]: Int32 ---------------------------------------------------------------------- Ran 1 test in 0.003s FAILED (failures=1) <unittest.runner.TextTestResult run=1 errors=0 failures=1>` 
```

在测试失败的追踪信息中，pandas 告诉我们，比较对象的数据类型不同。调用 `some_cool_numbers` 返回一个带有 `pd.Int64Dtype` 的 `pd.Series`，而我们的期望是 `pd.Int32Dtype`。

虽然这些示例集中在使用 `pd.testing.assert_series_equal`，但对于 `pd.DataFrame`，等效的方法是 `pd.testing.assert_frame_equal`。这两个函数知道如何处理可能不同的行索引、列索引、值和缺失值语义，并且如果期望不符合，它们会向测试运行器报告有用的错误信息。

## 还有更多……

这个示例使用了 `unittest` 模块，因为它是 Python 语言自带的。然而，许多大型 Python 项目，特别是在科学 Python 领域，使用 `pytest` 库来编写和执行单元测试。

与 `unittest` 不同，`pytest` 放弃了基于类的测试结构（包括 `setUp` 和 `tearDown` 方法），转而采用基于测试夹具的方法。关于这两种不同测试范式的比较，可以在 *pytest* 文档中找到。

`pytest` 库还提供了一套丰富的插件。有些插件可能旨在改善与第三方库的集成（比如 `pytest-django` 和 `pytest-sqlalchemy`），而其他插件则可能专注于扩展测试套件，利用系统的所有资源（例如 `pytest-xdist`）。介于两者之间，还有无数的插件使用场景，因此我强烈建议你了解 `pytest` 及其插件生态系统，以便测试你的 Python 代码库。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者一起讨论：

[`packt.link/pandas`](https://packt.link/pandas)

![](img/QR_Code5040900042138312.png)
