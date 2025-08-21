

# 第四章：pandas I/O 系统

到目前为止，我们一直在用数据创建`pd.Series`和`pd.DataFrame`对象，*内联*处理数据。虽然这样做有助于建立理论基础，但在生产代码中，很少有用户会这样做。相反，用户会使用 pandas 的 I/O 函数来从各种格式读取/写入数据。

I/O，指的是**输入/输出**，通常指从常见的数据格式（如 CSV、Microsoft Excel、JSON 等）中读取和写入数据的过程。当然，数据存储并不是只有一种格式，许多选项在性能、存储大小、第三方集成、可访问性和/或普及性之间进行权衡。有些格式假设数据是结构化且严格定义的（SQL 可能是最极端的例子），而其他格式则可以用于表示半结构化数据，这些数据不局限于二维结构（JSON 就是一个很好的例子）。

pandas 能够与多种数据格式进行交互，这也是它的最大优势之一，使得 pandas 成为数据分析工具中的瑞士军刀。无论是与 SQL 数据库、Microsoft Excel 文件集、HTML 网页，还是通过 JSON 传输数据的 REST API 端点交互，pandas 都能够胜任帮助你构建数据的统一视图。因此，pandas 被认为是 ETL 领域中的一个流行工具。

在本章中，我们将介绍以下操作方法：

+   CSV – 基本的读写操作

+   CSV – 读取大型文件的策略

+   Microsoft Excel – 基本的读写操作

+   Microsoft Excel – 在非默认位置查找表格

+   Microsoft Excel – 层次化数据

+   使用 SQLAlchemy 的 SQL

+   使用 ADBC 的 SQL

+   Apache Parquet

+   JSON

+   HTML

+   Pickle

+   第三方 I/O 库

# CSV – 基本的读写操作

CSV，代表*逗号分隔值*，是最常见的数据交换格式之一。虽然没有正式的标准来定义什么是 CSV 文件，但大多数开发者和用户通常认为它是一个纯文本文件，其中文件中的每一行表示一条数据记录，每条记录的字段之间有*分隔符*，用于表示一条记录的结束和下一条记录的开始。最常用的*分隔符*是逗号（因此叫做*逗号分隔值*），但这并不是硬性要求；有时我们也会看到使用管道符（`|`）、波浪符（`~`）或反引号（`` ` ``）作为分隔符的 CSV 文件。如果期望分隔符字符出现在某条记录内，通常会对单个记录（或所有记录）加上引号，以确保正确解析。

例如，假设一个 CSV 文件使用管道分隔符，其内容如下：

```py
`column1|column2 a|b|c` 
```

第一行将只读取两列数据，而第二行将包含三列数据。假设我们希望记录 `["a|b", "c"]` 出现在第二行，就需要进行适当的引号处理：

```py
`column1|column2 "a|b"|c` 
```

上述规则相对简单，可以轻松地写入 CSV 文件，但反过来这也使得读取 CSV 文件变得更加困难。CSV 格式没有提供元数据（例如，什么分隔符、引号规则等），也没有提供关于数据类型的任何信息（例如，哪些数据应位于 X 列）。这使得 CSV 读取器必须自己搞清楚这些内容，从而增加了性能开销，并且很容易导致数据误解。作为一种基于文本的格式，与像 Apache Parquet 这样的二进制格式相比，CSV 也是一种低效的数据存储方式。通过压缩 CSV 文件（以牺牲读/写性能为代价），可以在一定程度上弥补这些问题，但通常来说，CSV 在 CPU 效率、内存使用和无损性方面是最差的格式之一。

尽管存在这些缺点，CSV 格式已经存在很长时间，并且不会很快消失，因此了解如何使用 pandas 读取和写入此类文件是很有帮助的。

## 如何做到这一点

让我们从一个简单的 `pd.DataFrame` 开始。基于我们在*第三章*中学到的*数据类型*，我们知道 pandas 默认使用的数据类型并不理想，因此我们将使用 `pd.DataFrame.convert_dtypes` 方法，并使用 `dtype_backend="numpy_nullable"` 参数来构建这个以及以后所有的 `pd.DataFrame` 对象。

```py
`df = pd.DataFrame([     ["Paul", "McCartney", 1942],     ["John", "Lennon", 1940],     ["Richard", "Starkey", 1940],     ["George", "Harrison", 1943], ], columns=["first", "last", "birth"]) df = df.convert_dtypes(dtype_backend="numpy_nullable") df` 
```

```py
 `first   last       birth 0       Paul    McCartney  1942 1       John    Lennon     1940 2       Richard Starkey    1940 3       George  Harrison   1943` 
```

要将这个 `pd.DataFrame` 写入到 CSV 文件中，我们可以使用 `pd.DataFrame.to_csv` 方法。通常，您提供的第一个参数是文件名，但在这个例子中，我们将使用 `io.StringIO` 对象来代替。`io.StringIO` 对象类似于一个文件，但不会将任何内容保存到磁盘上。相反，它完全在内存中管理文件内容，无需清理，也不会在文件系统中留下任何东西：

```py
`import io buf = io.StringIO() df.to_csv(buf) print(buf.getvalue())` 
```

```py
`,first,last,birth 0,Paul,McCartney,1942 1,John,Lennon,1940 2,Richard,Starkey,1940 3,George,Harrison,1943` 
```

现在我们有了一个包含 CSV 数据的“文件”，我们可以使用 `pd.read_csv` 函数将这些数据读回来。然而，默认情况下，pandas 中的 I/O 函数将使用与 `pd.DataFrame` 构造函数相同的默认数据类型。幸运的是，我们仍然可以使用 `dtype_backend="numpy_nullable"` 参数与 I/O 读取函数一起使用，从而避免这个问题：

```py
`buf.seek(0) pd.read_csv(buf, dtype_backend="numpy_nullable")` 
```

```py
 `Unnamed: 0   first    last       birth 0    0            Paul     McCartney  1942 1    1            John     Lennon     1940 2    2            Richard  Starkey    1940 3    3            George   Harrison   1943` 
```

有趣的是，`pd.read_csv` 的结果并不完全与我们最初的 `pd.DataFrame` 匹配，因为它包含了一个新增的 `Unnamed: 0` 列。当你调用 `pd.DataFrame.to_csv` 时，它会将行索引和列一起写入到 CSV 文件中。CSV 格式不允许你存储任何额外的元数据来指示哪些列应映射到行索引，哪些列应表示 `pd.DataFrame` 中的列，因此 `pd.read_csv` 假设所有内容都是列。

您可以通过让 `pd.read_csv` 知道 CSV 文件中的第一列数据应该形成行索引，并使用 `index_col=0` 参数来纠正这种情况：

```py
`buf.seek(0) pd.read_csv(buf, dtype_backend="numpy_nullable", index_col=0)` 
```

```py
 `first    last       birth 0     Paul     McCartney  1942 1     John     Lennon     1940 2     Richard  Starkey    1940 3     George   Harrison   1943` 
```

或者，您可以通过 `pd.DataFrame.to_csv` 的 `index=False` 参数避免一开始就写入索引：

```py
`buf = io.StringIO() df.to_csv(buf, index=False) print(buf.getvalue())` 
```

```py
`first,last,birth Paul,McCartney,1942 John,Lennon,1940 Richard,Starkey,1940 George,Harrison,1943` 
```

## 还有更多……

正如本节开头提到的，CSV 文件使用引号来防止字段中出现的*分隔符*与其预期用途（即指示新记录的开始）混淆。幸运的是，pandas 默认以一种相当理智的方式处理这一点，我们可以通过一些新示例数据来看到这一点：

```py
`df = pd.DataFrame([     ["McCartney, Paul", 1942],     ["Lennon, John", 1940],     ["Starkey, Richard", 1940],     ["Harrison, George", 1943], ], columns=["name", "birth"]) df = df.convert_dtypes(dtype_backend="numpy_nullable") df` 
```

```py
 `name               birth 0    McCartney, Paul    1942 1    Lennon, John       1940 2    Starkey, Richard   1940 3    Harrison, George   1943` 
```

现在我们只有一个包含逗号的`name`列，可以看到 pandas 会对该字段加上引号，表示逗号的使用是数据的一部分，而不是新记录的开始：

```py
`buf = io.StringIO() df.to_csv(buf, index=False) print(buf.getvalue())` 
```

```py
`name,birth "McCartney, Paul",1942 "Lennon, John",1940 "Starkey, Richard",1940 "Harrison, George",1943` 
```

我们也可以选择使用不同的*分隔符*，这可以通过 `sep=` 参数进行切换：

```py
`buf = io.StringIO() df.to_csv(buf, index=False, sep="|") print(buf.getvalue())` 
```

```py
`name|birth McCartney, Paul|1942 Lennon, John|1940 Starkey, Richard|1940 Harrison, George|1943` 
```

我们还提到，尽管 CSV 文件本质上是纯文本格式，您也可以通过压缩它们来节省存储空间。最简单的方法是提供带有常见压缩文件扩展名的文件名参数，例如通过 `df.to_csv("data.csv.zip")`。如果需要更明确的控制，您可以使用 `compression=` 参数。

为了看到这一点的实际效果，让我们使用一个更大的 `pd.DataFrame`：

```py
`df = pd.DataFrame({     "col1": ["a"] * 1_000,     "col2": ["b"] * 1_000,     "col3": ["c"] * 1_000, }) df = df.convert_dtypes(dtype_backend="numpy_nullable") df.head()` 
```

```py
 `col1     col2    col3 0      a        b       c 1      a        b       c 2      a        b       c 3      a        b       c 4      a        b       c` 
```

请注意将文件写出为纯文本 CSV 文件时所使用的字节数：

```py
`buf = io.StringIO() df.to_csv(buf, index=False) len(buf.getvalue())` 
```

```py
`6015` 
```

使用 `compression="gzip"`，我们可以生成一个占用存储空间更少的文件：

```py
`buf = io.BytesIO() df.to_csv(buf, index=False, compression="gzip") len(buf.getvalue())` 
```

```py
`69` 
```

这里的权衡是，虽然压缩文件需要更少的磁盘存储空间，但它们需要更多的 CPU 工作来压缩或解压缩文件内容。

# CSV – 读取大文件的策略

处理大型 CSV 文件可能具有挑战性，尤其是在它们耗尽计算机内存时。在许多现实世界的数据分析场景中，您可能会遇到无法通过单次读取操作处理的数据集。这可能导致性能瓶颈和 `MemoryError` 异常，使分析变得困难。不过，不必担心！有很多方法可以提高处理文件的效率。

在这个例子中，我们将展示如何使用 pandas 查看 CSV 文件的部分内容，以了解正在推断的数据类型。通过这个理解，我们可以指示 `pd.read_csv` 使用更高效的数据类型，从而大大提高内存使用效率。

## 如何操作

在这个例子中，我们将查看*钻石*数据集。这个数据集对于现代计算机来说并不算特别大，但让我们假设这个文件比实际要大得多，或者假设我们的机器内存有限，以至于正常的 `read_csv` 调用会引发 `MemoryError`。

首先，我们将查看数据集中的前 1,000 行，通过 `nrows=1_000` 来了解文件中的内容：

```py
`df = pd.read_csv("data/diamonds.csv", dtype_backend="numpy_nullable", nrows=1_000) df` 
```

```py
 `carat  cut      color  clarity  depth  table  price  x      y      z 0  0.23   Ideal    E      SI2      61.5   55.0   326    3.95   3.98   2.43 1  0.21   Premium  E      SI1      59.8   61.0   326    3.89   3.84   2.31 2  0.23   Good     E      VS1      56.9   65.0   327    4.05   4.07   2.31 3  0.29   Premium  I      VS2      62.4   58.0   334    4.2    4.23   2.63 4  0.31   Good     J      SI2      63.3   58.0   335    4.34   4.35   2.75 …  …      …        …      …        …      …      …      …      …      … 995  0.54   Ideal    D    VVS2    61.4    52.0   2897   5.3    5.34   3.26 996  0.72   Ideal    E    SI1     62.5    55.0   2897   5.69   5.74   3.57 997  0.72   Good     F    VS1     59.4    61.0   2897   5.82   5.89   3.48 998  0.74   Premium  D    VS2     61.8    58.0   2897   5.81   5.77   3.58 999  1.12   Premium  J    SI2     60.6    59.0   2898   6.68   6.61   4.03 1000 rows × 10 columns` 
```

`pd.DataFrame.info` 方法应该能让我们了解这个子集使用了多少内存：

```py
`df.info()` 
```

```py
`<class 'pandas.core.frame.DataFrame'> RangeIndex: 1000 entries, 0 to 999 Data columns (total 10 columns): #   Column   Non-Null Count  Dtype  ---  ------   --------------  -----  0   carat    1000 non-null   Float64 1   cut      1000 non-null   string 2   color    1000 non-null   string 3   clarity  1000 non-null   string 4   depth    1000 non-null   Float64 5   table    1000 non-null   Float64 6   price    1000 non-null   Int64  7   x        1000 non-null   Float64 8   y        1000 non-null   Float64 9   z        1000 non-null   Float64 dtypes: Float64(6), Int64(1), string(3) memory usage: 85.1 KB` 
```

您看到的具体内存使用量可能取决于您使用的 pandas 版本和操作系统，但假设我们使用的 `pd.DataFrame` 大约需要 85 KB 的内存。如果我们有 10 亿行数据，而不是只有 1,000 行，那仅仅存储这个 `pd.DataFrame` 就需要 85 GB 的内存。

那么我们如何解决这个问题呢？首先，值得更仔细地查看已经推断出的数据类型。`price`列可能是一个立即引起我们注意的列；它被推断为`pd.Int64Dtype()`，但我们很可能不需要 64 位来存储这些信息。关于汇总统计的更多细节将在*第五章*中讨论，*算法及其应用*，但现在，让我们先看看`pd.Series.describe`，看看 pandas 可以为我们提供关于这个列的信息：

```py
`df["price"].describe()` 
```

```py
`count       1000.0 mean       2476.54 std      839.57562 min          326.0 25%         2777.0 50%         2818.0 75%         2856.0 max         2898.0 Name: price, dtype: Float64` 
```

最小值为 326，最大值为 2,898。这些值可以安全地适配`pd.Int16Dtype()`，与`pd.Int64Dtype()`相比，这将节省大量内存。

让我们还来看看一些浮点类型，从*指数*开始：

```py
`df["carat"].describe()` 
```

```py
`count      1000.0 mean      0.68928 std      0.195291 min           0.2 25%           0.7 50%          0.71 75%          0.79 max          1.27 Name: carat, dtype: Float64` 
```

这些值的范围从 0.2 到 1.27，除非我们预计要进行许多小数点计算，否则 32 位浮点数据类型提供的 6 到 9 位小数精度应该足够使用。

对于这个配方，我们假设 32 位浮动类型可以用于所有其他浮动类型。告诉`pd.read_csv`我们希望使用更小的数据类型的一个方法是使用`dtype=`参数，并通过字典将列名映射到所需的数据类型。由于我们的`dtype=`参数将覆盖所有列，因此我们也可以省略`dtype_backend="numpy_nullable"`，因为它是多余的：

```py
`df2 = pd.read_csv(     "data/diamonds.csv",     nrows=1_000,     dtype={         "carat": pd.Float32Dtype(),         "cut": pd.StringDtype(),         "color": pd.StringDtype(),         "clarity": pd.StringDtype(),         "depth": pd.Float32Dtype(),         "table": pd.Float32Dtype(),         "price": pd.Int16Dtype(),         "x": pd.Float32Dtype(),         "y": pd.Float32Dtype(),         "z": pd.Float32Dtype(),     } ) df2.info()` 
```

```py
`<class 'pandas.core.frame.DataFrame'> RangeIndex: 1000 entries, 0 to 999 Data columns (total 10 columns): #   Column   Non-Null Count  Dtype  ---  ------   --------------  -----  0   carat    1000 non-null   Float32 1   cut      1000 non-null   string 2   color    1000 non-null   string 3   clarity  1000 non-null   string 4   depth    1000 non-null   Float32 5   table    1000 non-null   Float32 6   price    1000 non-null   Int16  7   x        1000 non-null   Float32 8   y        1000 non-null   Float32 9   z        1000 non-null   Float32 dtypes: Float32(6), Int16(1), string(3) memory usage: 55.8 KB` 
```

仅仅这些步骤可能就会将内存使用量减少到大约 55KB，相较于最初的 85KB，已经是一个不错的减少！为了更安全起见，我们可以使用`pd.DataFrame.describe()`方法获取汇总统计信息，并确保这两个`pd.DataFrame`对象相似。如果这两个`pd.DataFrame`对象的数字相同，那是一个很好的迹象，表明我们的转换没有实质性地改变数据：

```py
`df.describe()` 
```

```py
 `carat    depth    table     price      x         y         z count   1000.0   1000.0	  1000.0    1000.0     1000.0    1000.0    1000.0 mean    0.68928  61.7228  57.7347   2476.54    5.60594   5.59918   3.45753 std     0.195291 1.758879 2.467946  839.57562  0.625173  0.611974  0.389819 min     0.2      53.0     52.0      326.0      3.79      3.75      2.27 25%     0.7      60.9     56.0      2777.0     5.64      5.63      3.45 50%     0.71     61.8     57.0      2818.0     5.77      5.76      3.55 75%     0.79     62.6     59.0      2856.0     5.92      5.91      3.64 max     1.27     69.5     70.0      2898.0     7.12      7.05      4.33` 
```

```py
`df2.describe()` 
```

```py
 `carat    depth     table      price      x        y        z count  1000.0   1000.0    1000.0     1000.0     1000.0   1000.0   1000.0 mean   0.68928  61.722801 57.734699  2476.54    5.60594  5.59918  3.45753 std    0.195291 1.758879  2.467946   839.57562  0.625173 0.611974 0.389819 min    0.2      53.0      52.0       326.0      3.79     3.75     2.27 25%    0.7      60.900002 56.0       2777.0     5.64     5.63     3.45 50%    0.71     61.799999 57.0       2818.0     5.77     5.76     3.55 75%    0.79     62.599998 59.0       2856.0     5.92     5.91     3.64 max    1.27     69.5      70.0       2898.0     7.12     7.05     4.33` 
```

到目前为止，一切看起来不错，但我们仍然可以做得更好。首先，似乎`cut`列有相对较少的唯一值：

```py
`df2["cut"].unique()` 
```

```py
`<StringArray> ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'] Length: 5, dtype: string` 
```

对`color`列也可以说同样的事情：

```py
`df2["color"].unique()` 
```

```py
`<StringArray> ['E', 'I', 'J', 'H', 'F', 'G', 'D'] Length: 7, dtype: string` 
```

以及`clarity`列：

```py
`df2["clarity"].unique()` 
```

```py
`<StringArray> ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF'] Length: 8, dtype: string` 
```

从 1,000 行抽样数据来看，`cut`列有 5 个不同的值，`color`列有 7 个不同的值，`clarity`列有 8 个不同的值。我们认为这些列具有*低基数*，即与行数相比，独特值的数量非常少。

这使得这些列非常适合使用分类类型。然而，我建议不要将`pd.CategoricalDtype()`作为`dtype=`的参数，因为默认情况下，它使用`np.nan`作为缺失值指示符（如果你需要回顾这个警告，可以重新查看在*第三章*中提到的*分类类型*配方）。相反，最佳的做法是首先将列读取为`pd.StringDtype()`，然后在适当的列上使用`pd.DataFrame.astype`：

```py
`df3 = pd.read_csv(     "data/diamonds.csv",     nrows=1_000,     dtype={         "carat": pd.Float32Dtype(),         "cut": pd.StringDtype(),         "color": pd.StringDtype(),         "clarity": pd.StringDtype(),         "depth": pd.Float32Dtype(),         "table": pd.Float32Dtype(),         "price": pd.Int16Dtype(),         "x": pd.Float32Dtype(),         "y": pd.Float32Dtype(),         "z": pd.Float32Dtype(),     } ) cat_cols = ["cut", "color", "clarity"] df3[cat_cols] = df3[cat_cols].astype(pd.CategoricalDtype()) df3.info()` 
```

```py
`<class 'pandas.core.frame.DataFrame'> RangeIndex: 1000 entries, 0 to 999 Data columns (total 10 columns): #   Column   Non-Null Count  Dtype   ---  ------   --------------  -----   0   carat    1000 non-null   Float32 1   cut      1000 non-null   category 2   color    1000 non-null   category 3   clarity  1000 non-null   category 4   depth    1000 non-null   Float32 5   table    1000 non-null   Float32 6   price    1000 non-null   Int16   7   x        1000 non-null   Float32 8   y        1000 non-null   Float32 9   z        1000 non-null   Float32 dtypes: Float32(6), Int16(1), category(3) memory usage: 36.2 KB` 
```

为了进一步节省内存，我们可能会决定不读取 CSV 文件中的某些列。如果希望 pandas 跳过这些数据以节省更多内存，可以使用`usecols=`参数：

```py
`dtypes = {  # does not include x, y, or z     "carat": pd.Float32Dtype(),     "cut": pd.StringDtype(),     "color": pd.StringDtype(),     "clarity": pd.StringDtype(),     "depth": pd.Float32Dtype(),     "table": pd.Float32Dtype(),     "price": pd.Int16Dtype(), } df4 = pd.read_csv(     "data/diamonds.csv",     nrows=1_000,     dtype=dtypes,     usecols=dtypes.keys(), ) cat_cols = ["cut", "color", "clarity"] df4[cat_cols] = df4[cat_cols].astype(pd.CategoricalDtype()) df4.info()` 
```

```py
`<class 'pandas.core.frame.DataFrame'> RangeIndex: 1000 entries, 0 to 999 Data columns (total 7 columns): #   Column   Non-Null Count  Dtype   ---  ------   --------------  -----   0   carat    1000 non-null   Float32 1   cut      1000 non-null   category 2   color    1000 non-null   category 3   clarity  1000 non-null   category 4   depth    1000 non-null   Float32 5   table    1000 non-null   Float32 6   price    1000 non-null   Int16   dtypes: Float32(3), Int16(1), category(3) memory usage: 21.5 KB` 
```

如果前面的步骤不足以创建足够小的`pd.DataFrame`，你可能还是有机会的。如果你可以一次处理一部分数据，而不需要将所有数据都加载到内存中，你可以使用`chunksize=`参数来控制从文件中读取数据块的大小：

```py
`dtypes = {  # does not include x, y, or z     "carat": pd.Float32Dtype(),     "cut": pd.StringDtype(),     "color": pd.StringDtype(),     "clarity": pd.StringDtype(),     "depth": pd.Float32Dtype(),     "table": pd.Float32Dtype(),     "price": pd.Int16Dtype(), } df_iter = pd.read_csv(     "data/diamonds.csv",     nrows=1_000,     dtype=dtypes,     usecols=dtypes.keys(),     chunksize=200 ) for df in df_iter:     cat_cols = ["cut", "color", "clarity"]     df[cat_cols] = df[cat_cols].astype(pd.CategoricalDtype())     print(f"processed chunk of shape {df.shape}")` 
```

```py
`processed chunk of shape (200, 7) processed chunk of shape (200, 7) processed chunk of shape (200, 7) processed chunk of shape (200, 7) processed chunk of shape (200, 7)` 
```

## 还有更多...

这里介绍的`usecols`参数也可以接受一个可调用对象，当它在每个遇到的列标签上求值时，如果该列应被读取，则返回`True`，如果应跳过，则返回`False`。如果我们只想读取`carat`、`cut`、`color`和`clarity`列，可能会像这样：

```py
`def startswith_c(column_name: str) -> bool:     return column_name.startswith("c") pd.read_csv(     "data/diamonds.csv",     dtype_backend="numpy_nullable",     usecols=startswith_c, )` 
```

```py
 `carat   cut       color  clarity 0      0.23    Ideal     E      SI2 1      0.21    Premium   E      SI1 2      0.23    Good      E      VS1 3      0.29    Premium   I      VS2 4      0.31    Good      J      SI2 …      …       …         …      … 53935  0.72    Ideal     D      SI1 53936  0.72    Good      D      SI1 53937  0.7     Very Good D      SI1 53938  0.86    Premium   H      SI2 53939  0.75    Ideal     D      SI2 53940 rows × 4 columns` 
```

# Microsoft Excel – 基本的读写操作

Microsoft Excel 是一个极其流行的数据分析工具，因其易用性和普及性。Microsoft Excel 提供了一个相当强大的工具包，帮助清洗、转换、存储和可视化数据，而无需了解编程语言。许多成功的分析师可能会认为它是他们永远需要的*唯一*工具。尽管如此，Microsoft Excel 在性能和可扩展性上确实存在困难，作为存储介质时，它甚至可能在意想不到的方式上改变你的数据。

如果你以前使用过 Microsoft Excel，现在开始学习 pandas，你会发现 pandas 是一个互补工具。使用 pandas 时，你将放弃 Microsoft Excel 的点选操作便捷性，但你将轻松解锁性能，超越 Microsoft Excel 的限制。

在我们进入这个方法之前，值得注意的是，Microsoft Excel 支持并不是 pandas 的一部分，因此你需要安装第三方包来使这些方法生效。虽然这不是唯一的选择，但鼓励用户安装`openpyxl`，因为它非常适合读取和写入各种 Microsoft Excel 格式。如果你还没有安装`openpyxl`，可以通过以下命令进行安装：

```py
`python -m pip install openpyxl` 
```

## 如何做到这一点

让我们再次从一个简单的`pd.DataFrame`开始：

```py
`df = pd.DataFrame([     ["Paul", "McCartney", 1942],     ["John", "Lennon", 1940],     ["Richard", "Starkey", 1940],     ["George", "Harrison", 1943], ], columns=["first", "last", "birth"]) df = df.convert_dtypes(dtype_backend="numpy_nullable") df` 
```

```py
 `first     last       birth 0    Paul      McCartney  1942 1    John      Lennon     1940 2    Richard   Starkey    1940 3    George    Harrison   1943` 
```

你可以使用`pd.DataFrame.to_excel`方法将其写入文件，通常第一个参数是文件名，例如`myfile.xlsx`，但在这里，我们将再次使用`io.BytesIO`，它像文件一样工作，但将二进制数据存储在内存中，而不是磁盘上：

```py
`import io buf = io.BytesIO() df.to_excel(buf)` 
```

对于读取操作，使用`pd.read_excel`函数。我们将继续使用`dtype_backend="numpy_nullable"`，以防止 pandas 执行默认的类型推断：

```py
`buf.seek(0) pd.read_excel(buf, dtype_backend="numpy_nullable")` 
```

```py
 `Unnamed: 0   first   last       birth 0    0            Paul    McCartney  1942 1    1            John    Lennon     1940 2    2            Richard Starkey    1940 3    3            George  Harrison   1943` 
```

许多函数参数与 CSV 共享。为了去除上面提到的`Unnamed: 0`列，我们可以指定`index_col=`参数：

```py
`buf.seek(0) pd.read_excel(buf, dtype_backend="numpy_nullable", index_col=0)` 
```

```py
 `first    last       birth 0    Paul     McCartney  1942 1    John     Lennon     1940 2    Richard  Starkey    1940 3    George   Harrison   1943` 
```

或者选择根本不写索引：

```py
`buf = io.BytesIO() df.to_excel(buf, index=False) buf.seek(0) pd.read_excel(buf, dtype_backend="numpy_nullable")` 
```

```py
 `first     last       birth 0    Paul      McCartney  1942 1    John      Lennon     1940 2    Richard   Starkey    1940 3    George    Harrison   1943` 
```

数据类型可以通过`dtype=`参数进行控制：

```py
`buf.seek(0) dtypes = {     "first": pd.StringDtype(),     "last": pd.StringDtype(),     "birth": pd.Int16Dtype(), } df = pd.read_excel(buf, dtype=dtypes) df.dtypes` 
```

```py
`first    string[python] last     string[python] birth             Int16 dtype: object` 
```

# Microsoft Excel – 在非默认位置查找表格

在上一个教程中，Microsoft Excel – 基本读写，我们使用了 Microsoft Excel 的 I/O 函数，而没有考虑数据在工作表中的位置。默认情况下，pandas 会从/写入数据的第一个工作表的第一个单元格开始读取，但通常会收到数据位于文档其他位置的 Microsoft Excel 文件。

对于这个示例，我们有一个 Microsoft Excel 工作簿，其中第一个选项卡**Sheet1**用作封面页：

![计算机的屏幕截图  自动生成描述](img/B31091_04_01.png)

图 4.1：Sheet1 中不包含有用数据的工作簿

第二个选项卡是我们有用的信息所在：

![计算机的屏幕截图](img/B31091_04_02.png)

图 4.2：另一个工作表包含相关数据的工作簿

## 如何操作

为了仍然能够读取这些数据，您可以使用`pd.read_excel`的`sheet_name=`、`skiprows=`和`usecols=`参数的组合：

```py
`pd.read_excel(     "data/beatles.xlsx",     dtype_backend="numpy_nullable",     sheet_name="the_data",     skiprows=4,     usecols="C:E", )` 
```

```py
 `first     last       birth 0    Paul      McCartney  1942 1    John      Lennon     1940 2    Richard   Starkey    1940 3    George    Harrison   1943` 
```

通过传递`sheet_name="the_data"`，`pd.read_excel`函数能够准确定位 Microsoft Excel 文件中要开始查找数据的特定工作表。或者，我们也可以使用`sheet_name=1`按选项卡位置搜索。在找到正确的工作表后，pandas 查看`skiprows=`参数，并知道要忽略工作表上的第 1-4 行。然后查看`usecols=`参数，仅选择 C 到 E 列。

## 还有更多…

我们可以提供我们想要的标签，而不是`usecols="C:E"`：

```py
`pd.read_excel(     "data/beatles.xlsx",     dtype_backend="numpy_nullable",     sheet_name="the_data",     skiprows=4,     usecols=["first", "last", "birth"], )` 
```

```py
 `first    last      birth 0    Paul     McCartney 1942 1    John     Lennon    1940 2    Richard  Starkey   1940 3    George   Harrison  1943` 
```

将这样的参数传递给`usecols=`是在处理 CSV 格式时选择文件中特定列的要求。然而，pandas 在读取 Microsoft Excel 文件时提供了特殊行为，允许像`"C:E"`或`"C,D,E"`这样的字符串引用列。

# 微软 Excel – 分层数据

数据分析的主要任务之一是将非常详细的信息汇总为易于消化的摘要。大多数公司的高管不想要翻阅成千上万的订单，他们只想知道，“过去 X 个季度我的销售情况如何？”

使用 Microsoft Excel，用户通常会在类似*图 4.3*所示的视图中总结这些信息，该视图代表了行上的`区域`/`子区域`层次结构和列上的`年份`/`季度`：

![电子表格的屏幕截图](img/B31091_04_03.png)

图 4.3：具有分层数据的工作簿 – 按区域和季度销售

尽管这个总结似乎并不太离谱，但许多分析工具很难正确呈现这种类型的信息。以传统的 SQL 数据库为例，没有直接的方法来在表中表示这种`Year`/`Quarter`层次结构 - 您唯一的选择是将所有层次结构字段连接在一起，并生成像`2024`/`Q1`、`2024`/`Q2`、`2025`/`Q1`和`2025`/`Q2`这样的列。虽然这样可以轻松选择任何单独的列，但您失去了轻松选择诸如“所有 2024 年销售额”之类的内容而不需要额外努力的能力。

幸运的是，pandas 可以比 SQL 数据库更理智地处理这个问题，直接支持行和列索引中的这种层次关系。如果您回忆起*第二章*，*选择和赋值*，我们介绍了`pd.MultiIndex`；能够保持这些关系使用户能够高效地从任何层次的层次结构中进行选择。

## 如何做到这一点

仔细检查*图 4.3*，您会看到第 1 行和第 2 行包含标签`Year`和`Quarter`，这些标签可以形成我们想要在`pd.DataFrame`的列中形成的`pd.MultiIndex`的级别。Microsoft Excel 使用每行基于 1 的编号，因此行`[1, 2]`转换为 Python 实际上是`[0, 1]`；我们将使用这个作为我们的`header=`参数，以确立我们希望前两行形成我们的列`pd.MultiIndex`。

将我们的注意力转向 Microsoft Excel 中的 A 列和 B 列，我们现在可以看到标签`Region`和`Sub-Region`，这将帮助我们在行中塑造`pd.MultiIndex`。回到*CSV - 基本读取/写入*部分，我们介绍了`index_col=`参数，它可以告诉 pandas 实际上应该使用哪些列数据来生成行索引。Microsoft Excel 文件中的 A 列和 B 列代表第一列和第二列，因此我们可以再次使用`[0, 1]`来告诉 pandas 我们的意图：

```py
`df = pd.read_excel(     "data/hierarchical.xlsx",     dtype_backend="numpy_nullable",     index_col=[0, 1],     header=[0, 1], ) df` 
```

```py
 `Year    2024            2025         Quarter Q1      Q2      Q1      Q2 Region  Sub-Region America East    1       2       4       8         West    16      32      64      128         South   256     512     1024    4096 Europe  West    8192    16384   32768   65536         East    131072  262144  524288  1048576` 
```

大功告成！我们成功读取了数据并保持了行和列的层次结构，这使我们可以使用所有原生 pandas 功能从这些数据中进行选择，甚至回答诸如“每个东部子区域的 Q2 业绩在年度上看起来如何？”这样的问题。

```py
`df.loc[(slice(None), "East"), (slice(None), "Q2")]` 
```

```py
 `Year    2024    2025         Quarter Q2      Q2 Region  Sub-Region America East    2       8 Europe  East    262144  1048576` 
```

# 使用 SQLAlchemy 的 SQL

pandas 库提供了与 SQL 数据库交互的强大功能，使您可以直接在关系数据库中存储的数据上进行数据分析。

当然，存在着无数的数据库（而且还会有更多！），每个数据库都有自己的特点、认证方案、方言和怪癖。为了与它们交互，pandas 依赖于另一个伟大的 Python 库 SQLAlchemy，它在核心上充当 Python 和数据库世界之间的桥梁。理论上，pandas 可以与 SQLAlchemy 可以连接的任何数据库一起使用。

要开始，您应该首先将 SQLAlchemy 安装到您的环境中：

```py
`python -m pip install sqlalchemy` 
```

SQLAlchemy 支持所有主要的数据库，如 MySQL、PostgreSQL、MS SQL Server 等，但设置和正确配置这些数据库本身需要不少努力，这部分内容超出了本书的范围。为了尽可能简化，我们将专注于使用 SQLite 作为我们的数据库，因为它不需要任何设置，且可以完全在计算机内存中运行。一旦你熟悉了 SQLite 的使用，你只需更改连接的凭据来指向目标数据库；否则，所有功能保持不变。

## 操作步骤

我们需要做的第一件事是创建一个 SQLAlchemy 引擎，使用`sa.create_engine`。这个函数的参数是一个 URL，取决于你试图连接的数据库（更多信息请参见 SQLAlchemy 文档）。在这些示例中，我们将使用内存中的 SQLite：

```py
`import sqlalchemy as sa engine = sa.create_engine("sqlite:///:memory:")` 
```

使用`pd.DataFrame.to_sql`方法，你可以将一个现有的`pd.DataFrame`写入数据库表中。第一个参数是你想创建的表的名称，第二个参数是引擎/可连接对象：

```py
`df = pd.DataFrame([     ["dog", 4],     ["cat", 4], ], columns=["animal", "num_legs"]) df = df.convert_dtypes(dtype_backend="numpy_nullable") df.to_sql("table_name", engine, index=False)` 
```

```py
`2` 
```

`pd.read_sql`函数可以用于反向操作，从数据库表中读取数据：

```py
`pd.read_sql("table_name", engine, dtype_backend="numpy_nullable")` 
```

```py
 `animal    num_legs 0    dog       4 1    cat       4` 
```

另外，如果你想要的不是仅仅复制表，而是某个不同的内容，你可以将 SQL 查询传递给`pd.read_sql`：

```py
`pd.read_sql(     "SELECT SUM(num_legs) AS total_legs FROM table_name",     engine,     dtype_backend="numpy_nullable" )` 
```

```py
 `total_legs 0     8` 
```

当数据库中已经存在表时，再次向同一表写入数据将会引发错误。你可以传递`if_exists="replace"`来覆盖此行为并替换表：

```py
`df = pd.DataFrame([     ["dog", 4],     ["cat", 4],     ["human", 2], ], columns=["animal", "num_legs"]) df = df.convert_dtypes(dtype_backend="numpy_nullable") df.to_sql("table_name", engine, index=False, if_exists="replace")` 
```

```py
`3` 
```

你还可以使用`if_exists="append"`将数据添加到表中：

```py
`new_data = pd.DataFrame([["centipede", 100]], columns=["animal", "num_legs"]) new_data.to_sql("table_name", engine, index=False, if_exists="append") pd.read_sql("table_name", engine, dtype_backend="numpy_nullable")` 
```

```py
 `animal     num_legs 0    dog        4 1    cat        4 2    human      2 3    centipede  100` 
```

绝大部分繁重的工作都由 SQLAlchemy 引擎在幕后完成，该引擎使用`dialect+driver://username:password@host:port/database`形式的 URL 构建。并非所有的字段都是必需的——该字符串会根据你使用的数据库及其配置有所不同。

在我们的具体示例中，`sa.create_engine("sqlite:///:memory:")`在我们计算机的内存空间中创建并连接到一个 SQLite 数据库。这个特性是 SQLite 特有的；我们也可以传递一个文件路径，比如`sa.create_engine("sqlite:///tmp/adatabase.sql")`，而不是使用`:memory:`。

欲了解更多关于 SQLAlchemy URL 的信息，以及如何搭配其他数据库使用驱动程序，请参考 SQLAlchemy 的后端特定 URL 文档。

# 使用 ADBC 的 SQL

尽管使用 SQLAlchemy 连接数据库是一种可行的选择，并且多年来一直帮助着 pandas 的用户，但来自 Apache Arrow 项目的一项新技术已经出现，它能进一步扩展 SQL 交互。这项新技术被称为**Arrow 数据库连接**，简称**ADBC**。从 2.2 版本开始，pandas 增加了对使用 ADBC 驱动程序与数据库交互的支持。

使用 ADBC 在与 SQL 数据库交互时，比上述基于 SQLAlchemy 的方法能提供更好的性能和类型安全。权衡之下，SQLAlchemy 支持更多的数据库，因此根据你的数据库，SQLAlchemy 可能是唯一的选择。ADBC 会记录其驱动实现状态；我建议在回退到 SQLAlchemy 之前，首先查看该记录，以确保你所使用的数据库有一个稳定的驱动实现。

和上一节一样，我们将使用 SQLite 作为数据库，因为它易于设置和配置。请确保为 SQLite 安装适当的 ADBC Python 包：

```py
`python -m pip install adbc-driver-sqlite` 
```

## 如何实现

首先，我们从我们的 SQLite ADBC 驱动中导入 `dbapi` 对象，并创建一些示例数据：

```py
`from adbc_driver_sqlite import dbapi df = pd.DataFrame([     ["dog", 4],     ["cat", 4],     ["human", 2], ], columns=["animal", "num_legs"]) df = df.convert_dtypes(dtype_backend="numpy_nullable") df` 
```

```py
 `animal   num_legs 0     dog      4 1     cat      4 2     human    2` 
```

术语 `dbapi` 来自 Python 数据库 API 规范（PEP-249），该规范标准化了 Python 模块和库如何与数据库交互。调用 `.connect` 方法并提供凭据是 Python 中打开数据库连接的标准化方式。我们将再次通过 `dbapi.connect("file::memory:")` 使用内存中的 SQLite 应用程序。

通过在 Python 中使用 `with ... as:` 语法来使用上下文管理器，我们可以连接到一个数据库，并将其赋值给一个变量，这样 Python 就会在块执行完毕后自动清理连接。在块内连接处于打开状态时，我们可以使用 `pd.DataFrame.to_sql` / `pd.read_sql` 分别向数据库写入和从数据库读取数据：

```py
`with dbapi.connect("file::memory:") as conn:     df.to_sql("table_name", conn, index=False, if_exists="replace")     df = pd.read_sql(         "SELECT * FROM table_name",         conn,         dtype_backend="numpy_nullable",     ) df` 
```

```py
 `animal    num_legs 0    dog       4 1    cat       4 2    human     2` 
```

对于较小的数据集，你可能看不出太大差别，但在较大的数据集上，ADBC 的性能提升会非常明显。让我们比较一下使用 SQLAlchemy 写入一个 10,000 行、10 列的 `pd.DataFrame` 所需的时间：

```py
`import timeit import sqlalchemy as sa np.random.seed(42) df = pd.DataFrame(     np.random.randn(10_000, 10),     columns=list("abcdefghij") ) with sa.create_engine("sqlite:///:memory:").connect() as conn:     func = lambda: df.to_sql("test_table", conn, if_exists="replace")     print(timeit.timeit(func, number=100))` 
```

```py
`4.898935955003253` 
```

使用 ADBC 的等效代码：

```py
`from adbc_driver_sqlite import dbapi with dbapi.connect("file::memory:") as conn:     func = lambda: df.to_sql("test_table", conn, if_exists="replace")     print(timeit.timeit(func, number=100))` 
```

```py
`0.7935214300014195` 
```

你的结果会有所不同，具体取决于你的数据和数据库，但通常情况下，ADBC 应该会表现得更快。

## 还有更多…

为了理解 ADBC 的作用以及它为何重要，首先值得简单回顾一下数据库标准的历史以及它们如何发展。在 1990 年代，**开放数据库连接**（**ODBC**）和 **Java 数据库连接**（**JDBC**）标准被引入，这帮助标准化了不同客户端如何与各种数据库进行通信。在这些标准引入之前，如果你开发了一个需要与两个或更多不同数据库交互的应用程序，那么你的应用程序就必须使用每个数据库能理解的语言来与其交互。

假设这个应用程序只想获取每个数据库中可用表格的列表。PostgreSQL 数据库将这些信息存储在名为 `pg_catalog.pg_tables` 的表中，而 SQLite 数据库则将其存储在一个 `sqlite_schema` 表中，条件是 `type='table'`。这个应用程序需要根据这些特定的信息来开发，并且每当数据库更改了存储这些信息的方式，或者当应用程序想要支持新数据库时，都需要重新发布。

使用像 ODBC 这样的标准时，应用程序只需要与驱动程序进行通信，告知驱动程序它需要系统中的所有表格。这将数据库交互的责任从应用程序转移到驱动程序，给应用程序提供了一个抽象层。随着新数据库或新版本的发布，应用程序本身不再需要改变；它只需与新的 ODBC/JDBC 驱动程序配合工作，继续正常运作。事实上，SQLAlchemy 就像这个理论中的应用程序；它通过 ODBC/JDBC 驱动程序与数据库交互，而不是试图独立管理无尽的数据库交互。

尽管这些标准在许多用途上非常出色，但值得注意的是，1990 年代的数据库与今天的数据库差异很大。许多这些标准试图解决的问题是针对当时盛行的行式数据库的。列式数据库是在十多年后才出现的，并且它们已经主导了数据分析领域。不幸的是，在没有列式数据传输标准的情况下，许多数据库不得不重新设计，使其兼容 ODBC/JDBC。这使得它们能够与今天存在的无数数据库客户端工具兼容，但也需要在性能和效率上做出一定的妥协。

ADBC 是解决这个问题的列式规范。pandas 库以及许多类似的产品在设计上明确地（或至少非常接近）采用了列式设计。当与列式数据库（如 BigQuery、Redshift 或 Snowflake）交互时，拥有一个列式驱动程序来交换信息可以带来数量级更好的性能。即使你没有与列式数据库交互，ADBC 驱动程序也经过精细优化，专为与 Apache Arrow 配合的分析工作，因此它 *仍然* 比 SQLAlchemy 使用的任何 ODBC/JDBC 驱动程序都要好。

对于想要了解更多 ADBC 的用户，我建议观看我在 PyData NYC 2023 上的演讲，标题为《使用 pandas 和 Apache Arrow 加速 SQL》，可以在 YouTube 上观看 ([`youtu.be/XhnfybpWOgA?si=RBrM7UUvpNFyct0L`](https://youtu.be/XhnfybpWOgA?si=RBrM7UUvpNFyct0L))。

# Apache Parquet

就 `pd.DataFrame` 的通用存储格式而言，Apache Parquet 是最佳选择。Apache Parquet 允许：

+   元数据存储 —— 这使得格式能够追踪数据类型等特性

+   分区 – 不是所有数据都需要在一个文件中

+   查询支持 – Parquet 文件可以在磁盘上进行查询，因此你不必将所有数据都加载到内存中

+   并行化 – 读取数据可以并行化以提高吞吐量

+   紧凑性 – 数据被压缩并以高效的方式存储

除非你在处理遗留系统，否则 Apache Parquet 格式应该取代在工作流中使用 CSV 文件的方式，从本地持久化数据、与其他团队成员共享，到跨系统交换数据。

## 如何操作

读取/写入 Apache Parquet 的 API 与我们到目前为止看到的所有 pandas API 一致；读取使用 `pd.read_parquet`，写入使用 `pd.DataFrame.to_parquet` 方法。

让我们从一些示例数据和 `io.BytesIO` 对象开始：

```py
`import io buf = io.BytesIO() df = pd.DataFrame([     ["Paul", "McCartney", 1942],     ["John", "Lennon", 1940],     ["Richard", "Starkey", 1940],     ["George", "Harrison", 1943], ], columns=["first", "last", "birth"]) df = df.convert_dtypes(dtype_backend="numpy_nullable") df` 
```

```py
 `first    last       birth 0     Paul     McCartney  1942 1     John     Lennon     1940 2     Richard  Starkey    1940 3     George   Harrison   1943` 
```

这是如何写入文件句柄的方式：

```py
`df.to_parquet(buf, index=False)` 
```

下面是如何从文件句柄中读取数据。注意我们故意没有提供 `dtype_backend="numpy_nullable"`：

```py
`buf.seek(0) pd.read_parquet(buf)` 
```

```py
 `first    last       birth 0     Paul     McCartney  1942 1     John     Lennon     1940 2     Richard  Starkey    1940 3     George   Harrison   1943` 
```

为什么我们在 `pd.read_parquet` 中不需要 `dtype_backend=` 参数？与像 CSV 这样的格式只存储数据不同，Apache Parquet 格式同时存储数据和元数据。在元数据中，Apache Parquet 能够追踪正在使用的数据类型，所以你写入的数据类型应该和你读取的数据类型完全一致。

你可以通过将 `birth` 列的数据类型更改为不同类型来进行测试：

```py
`df["birth"] = df["birth"].astype(pd.UInt16Dtype()) df.dtypes` 
```

```py
`first    string[python] last     string[python] birth            UInt16 dtype: object` 
```

通过 Apache Parquet 格式进行回环操作会返回你最初使用的相同数据类型：

```py
`buf = io.BytesIO() df.to_parquet(buf, index=False) buf.seek(0) pd.read_parquet(buf).dtypes` 
```

```py
`first    string[python] last     string[python] birth            UInt16 dtype: object` 
```

当然，如果你想更加防守型，在这里使用 `dtype_backend="numpy_nullable"` 也没有坏处。我们一开始故意没有使用它，是为了展示 Apache Parquet 格式的强大，但如果你从其他来源和开发者那里接收文件，他们没有使用我们在*第三章*《数据类型》中推荐的类型系统，那么确保使用 pandas 提供的最佳类型可能会对你有帮助：

```py
`suboptimal_df = pd.DataFrame([     [0, "foo"],     [1, "bar"],     [2, "baz"], ], columns=["int_col", "str_col"]) buf = io.BytesIO() suboptimal_df.to_parquet(buf, index=False) buf.seek(0) pd.read_parquet(buf, dtype_backend="numpy_nullable").dtypes` 
```

```py
`int_col             Int64 str_col    string[python] dtype: object` 
```

Apache Parquet 格式的另一个伟大特点是它支持 *分区*，这打破了所有数据必须位于一个文件中的要求。通过能够将数据分割到不同的目录和文件中，分区使得用户可以轻松地组织内容，同时也使程序能够更高效地优化可能需要或不需要读取的文件来解决分析查询。

有许多方法可以对数据进行分区，每种方法都有实际的空间/时间权衡。为了演示，我们假设使用 *基于时间的分区*，即为不同的时间段生成单独的文件。考虑到这一点，让我们使用以下数据布局，其中我们为每个年份创建不同的目录，并在每个年份内，为每个销售季度创建单独的文件：

```py
`Partitions  2022/    q1_sales.parquet    q2_sales.parquet  2023/    q1_sales.parquet    q2_sales.parquet` 
```

本书附带的每个示例 Apache Parquet 文件都已经使用我们在*第三章*（数据类型）中推荐的 pandas 扩展类型创建，因此我们进行的 `pd.read_parquet` 调用故意不包括 `dtype_backend="numpy_nullable"` 参数。在任何文件中，你都会看到我们存储了关于 `year`（年份）、`quarter`（季度）、`region`（地区）和总的 `sales`（销售额）等信息：

```py
`pd.read_parquet(     "data/partitions/2022/q1_sales.parquet", )` 
```

```py
 `year   quarter   region   sales 0     2022   Q1        America  1 1     2022   Q1        Europe   2` 
```

如果我们想查看所有数据的汇总，最直接的方法是遍历每个文件并累积结果。然而，使用 Apache Parquet 格式，pandas 可以本地有效地处理这个问题。与其将单个文件名传递给 `pd.read_parquet`，不如传递目录路径：

```py
`pd.read_parquet("data/partitions/")` 
```

```py
 `year    quarter  region    sales 0    2022    Q1       America   1 1    2022    Q1       Europe    2 2    2022    Q2       America   4 3    2022    Q2       Europe    8 4    2023    Q1       America   16 5    2023    Q1       Europe    32 6    2023    Q2       America   64 7    2023    Q2       Europe    128` 
```

由于我们的示例数据非常小，我们没有问题先将所有数据读取到 `pd.DataFrame` 中，然后从那里进行操作。然而，在生产环境中，你可能会遇到存储量达到吉字节或太字节的 Apache Parquet 文件。试图将所有数据读取到 `pd.DataFrame` 中可能会导致 `MemoryError` 错误。

幸运的是，Apache Parquet 格式使你能够在读取文件时动态过滤记录。在 pandas 中，你可以通过传递 `filters=` 参数来启用此功能，方法是使用 `pd.read_parquet`。该参数应该是一个列表，其中每个列表元素是一个包含三个元素的元组：

+   列名

+   逻辑运算符

+   值

例如，如果我们只想读取 `region` 列值为 `Europe` 的数据，可以这样写：

```py
`pd.read_parquet(     "data/partitions/",     filters=[("region", "==", "Europe")], )` 
```

```py
 `year    quarter   region   sales 0    2022    Q1        Europe   2 1    2022    Q2        Europe   8 2    2023    Q1        Europe   32 3    2023    Q2        Europe   128` 
```

# JSON

**JavaScript 对象表示法**（**JSON**）是一种常用于通过互联网传输数据的格式。JSON 规范可以在 [`www.json.org`](https://www.json.org/json-en.html) 上找到。尽管名称中有 JavaScript，但它不需要 JavaScript 来读取或创建。

Python 标准库附带了 `json` 库，它可以将 Python 对象序列化为 JSON，或者从 JSON 反序列化回 Python 对象：

```py
`import json beatles = {     "first": ["Paul", "John", "Richard", "George",],     "last": ["McCartney", "Lennon", "Starkey", "Harrison",],     "birth": [1942, 1940, 1940, 1943], } serialized = json.dumps(beatles) print(f"serialized values are: {serialized}") deserialized = json.loads(serialized) print(f"deserialized values are: {deserialized}")` 
```

```py
`serialized values are: {"first": ["Paul", "John", "Richard", "George"], "last": ["McCartney", "Lennon", "Starkey", "Harrison"], "birth": [1942, 1940, 1940, 1943]} deserialized values are: {'first': ['Paul', 'John', 'Richard', 'George'], 'last': ['McCartney', 'Lennon', 'Starkey', 'Harrison'], 'birth': [1942, 1940, 1940, 1943]}` 
```

然而，标准库并不知道如何处理 pandas 对象，因此 pandas 提供了自己的 I/O 函数，专门用于处理 JSON。

## 如何实现

在最简单的形式下，`pd.read_json` 可以用于读取 JSON 数据：

```py
`import io data = io.StringIO(serialized) pd.read_json(data, dtype_backend="numpy_nullable")` 
```

```py
 `first    last       birth 0    Paul     McCartney  1942 1    John     Lennon     1940 2    Richard  Starkey    1940 3    George   Harrison   1943` 
```

`pd.DataFrame.to_json` 方法可以用于写入：

```py
`df = pd.DataFrame(beatles) print(df.to_json())` 
```

```py
`{"first":{"0":"Paul","1":"John","2":"Richard","3":"George"},"last":{"0":"McCartney","1":"Lennon","2":"Starkey","3":"Harrison"},"birth":{"0":1942,"1":1940,"2":1940,"3":1943}}` 
```

然而，在实际应用中，存在无数种将表格数据表示为 JSON 的方式。有些用户可能希望看到 `pd.DataFrame` 中的每一行作为 JSON 数组，而另一些用户可能希望看到每一列作为数组。还有一些用户可能希望查看行索引、列索引和数据作为独立的 JSON 对象列出，而其他用户可能根本不关心是否看到行或列标签。

对于这些用例以及更多用例，pandas 允许你传递一个参数给 `orient=`，其值决定了要读取或写入的 JSON 布局：

+   `columns`（默认值）：生成 JSON 对象，其中键是列标签，值是另一个对象，该对象将行标签映射到数据点。

+   `records`：`pd.DataFrame` 的每一行都表示为一个 JSON 数组，其中包含将列名映射到数据点的对象。

+   `split`：映射到 `{"columns": […], "index": […], "data": […]}`。列/索引值是标签的数组，数据包含数组的数组。

+   `index`：与列类似，不同之处在于行和列标签作为键的使用被反转。

+   `values`：将 `pd.DataFrame` 的数据映射到数组的数组中。行/列标签被丢弃。

+   `table`：遵循 JSON 表格模式。

JSON 是一种*有损*的数据交换格式，因此上述每种 `orient` 都是在损失、冗长性和最终用户需求之间的权衡。`orient="table"` 会最少损失数据，但会产生最大负载，而 `orient="values"` 完全位于该范围的另一端。

为了突出每种 `orient` 之间的差异，让我们从一个相当简单的 `pd.DataFrame` 开始：

```py
`df = pd.DataFrame(beatles, index=["row 0", "row 1", "row 2", "row 3"]) df = df.convert_dtypes(dtype_backend="numpy_nullable") df` 
```

```py
 `first    last       birth row 0  Paul     McCartney  1942 row 1  John     Lennon     1940 row 2  Richard  Starkey    1940 row 3  George   Harrison   1943` 
```

传递 `orient="columns"` 会生成使用以下模式的数据：`{"column":{"row": value, "row": value, ...}, ...}`：

```py
`serialized = df.to_json(orient="columns") print(f'Length of orient="columns": {len(serialized)}') serialized[:100]` 
```

```py
`Length of orient="columns": 221 {"first":{"row 0":"Paul","row 1":"John","row 2":"Richard","row 3":"George"},"last":{"row 0":"McCartn` 
```

这是一种相当冗长的存储数据方式，因为它会为每一列重复行索引标签。优点是，pandas 可以相对较好地从这种 `orient` 中重建正确的 `pd.DataFrame`：

```py
`pd.read_json(     io.StringIO(serialized),     orient="columns",     dtype_backend="numpy_nullable" )` 
```

```py
 `first    last       birth row 0  Paul     McCartney  1942 row 1  John     Lennon     1940 row 2  Richard  Starkey    1940 row 3  George   Harrison   1943` 
```

使用 `orient="records"` 时，你会得到每一行 `pd.DataFrame` 的表示，不带行索引标签，形成一个模式 `[{"col": value, "col": value, ...}, ...]`：

```py
`serialized = df.to_json(orient="records") print(f'Length of orient="records": {len(serialized)}') serialized[:100]` 
```

```py
`Length of orient="records": 196 [{"first":"Paul","last":"McCartney","birth":1942},{"first":"John","last":"Lennon","birth":1940},{"fi` 
```

尽管这种表示方式比 `orient="columns"` 更紧凑，但它不存储任何行标签，因此在重建时，你将得到一个带有新生成 `pd.RangeIndex` 的 `pd.DataFrame`：

```py
`pd.read_json(     io.StringIO(serialized),     orient="orient",     dtype_backend="numpy_nullable" )` 
```

```py
 `first    last       birth 0     Paul     McCartney  1942 1     John     Lennon     1940 2     Richard  Starkey    1940 3     George   Harrison   1943` 
```

使用 `orient="split"` 时，行索引标签、列索引标签和数据会分别存储：

```py
`serialized = df.to_json(orient="split") print(f'Length of orient="split": {len(serialized)}') serialized[:100]` 
```

```py
`Length of orient="split": 190 {"columns":["first","last","birth"],"index":["row 0","row 1","row 2","row 3"],"data":[["Paul","McCar` 
```

这种格式使用的字符比 `orient="columns"` 少，而且你仍然可以相对较好地重建一个 `pd.DataFrame`，因为它与你使用构造函数（如 `pd.DataFrame(data, index=index, columns=columns)`）构建 `pd.DataFrame` 的方式相似：

```py
`pd.read_json(     io.StringIO(serialized),     orient="split",     dtype_backend="numpy_nullable", )` 
```

```py
 `first    last       birth row 0  Paul     McCartney  1942 row 1  John     Lennon     1940 row 2  Richard  Starkey    1940 row 3  George   Harrison   1943` 
```

虽然这是一个很好的格式，用于双向转换 `pd.DataFrame`，但与其他格式相比，在“野外”遇到这种 JSON 格式的可能性要低得多。

`orient="index"` 与 `orient="columns"` 非常相似，但它反转了行和列标签的角色：

```py
`serialized = df.to_json(orient="index") print(f'Length of orient="index": {len(serialized)}') serialized[:100]` 
```

```py
`Length of orient="index": 228 {"row 0":{"first":"Paul","last":"McCartney","birth":1942},"row 1":{"first":"John","last":"Lennon","b` 
```

再次强调，你可以合理地重建你的 `pd.DataFrame`：

```py
`pd.read_json(     io.StringIO(serialized),     orient="index",     dtype_backend="numpy_nullable", )` 
```

```py
 `first    last       birth row 0  Paul     McCartney  1942 row 1  John     Lennon     1940 row 2  Richard  Starkey    1940 row 3  George   Harrison   1943` 
```

一般来说，`orient="index"` 会比 `orient="columns"` 占用更多空间，因为大多数 `pd.DataFrame` 对象使用的列标签比索引标签更冗长。我建议仅在列标签不那么冗长，或者有其他系统强制要求特定格式的情况下使用此格式。

对于最简洁的表示方式，你可以选择 `orient="values"`。使用此 `orient`，既不保存行标签，也不保存列标签：

```py
`serialized = df.to_json(orient="values") print(f'Length of orient="values": {len(serialized)}') serialized[:100]` 
```

```py
`Length of orient="values": 104 [["Paul","McCartney",1942],["John","Lennon",1940],["Richard","Starkey",1940],["George","Harrison",19` 
```

当然，由于它们没有在 JSON 数据中表示，当使用 `orient="values"` 读取时，你将无法保留行/列标签：

```py
`pd.read_json(     io.StringIO(serialized),     orient="values",     dtype_backend="numpy_nullable", )` 
```

```py
 `0        1          2 0    Paul     McCartney  1942 1    John     Lennon     1940 2    Richard  Starkey    1940 3    George   Harrison   1943` 
```

最后，我们有了 `orient="table"`。这是所有输出中最冗长的一种，但它是唯一一个有实际标准支持的输出，标准叫做 JSON 表格模式：

```py
`serialized = df.to_json(orient="table") print(f'Length of orient="table": {len(serialized)}') serialized[:100]` 
```

```py
`Length of orient="table": 524 {"schema":{"fields":[{"name":"index","type":"string"},{"name":"first","type":"any","extDtype":"strin` 
```

表格模式更冗长，因为它存储了关于数据序列化的元数据，类似于我们在 Apache Parquet 格式中看到的内容（尽管功能不如 Apache Parquet）。对于所有其他的 `orient=` 参数，pandas 必须在读取时推断数据的类型，但 JSON 表格格式会为你保留这些信息。因此，假设你一开始就使用了 pandas 扩展类型，你甚至不需要 `dtype_backend="numpy_nullable"` 参数：

```py
`df["birth"] = df["birth"].astype(pd.UInt16Dtype()) serialized = df.to_json(orient="table") pd.read_json(     io.StringIO(serialized),     orient="table", ).dtypes` 
```

```py
`first    string[python] last     string[python] birth            UInt16 dtype: object` 
```

## 还有更多……

当尝试读取 JSON 时，你可能会发现以上格式仍然无法充分表达你想要实现的目标。幸运的是，仍然有 `pd.json_normalize`，它可以作为一个功能强大的函数，将你的 JSON 数据转换为表格格式。

想象一下，正在处理来自一个理论上的带有分页的 REST API 的以下 JSON 数据：

```py
`data = {     "records": [{         "name": "human",         "characteristics": {             "num_leg": 2,             "num_eyes": 2         }     }, {         "name": "dog",         "characteristics": {             "num_leg": 4,             "num_eyes": 2         }     }, {         "name": "horseshoe crab",         "characteristics": {             "num_leg": 10,             "num_eyes": 10         }     }],     "type": "animal",     "pagination": {         "next": "23978sdlkusdf97234u2io",         "has_more": 1     } }` 
```

虽然 `"pagination"` 键对于导航 API 很有用，但对于我们来说报告价值不大，而且它可能会导致 JSON 序列化器出现问题。我们真正关心的是与 `"records"` 键相关的数组。你可以指示 `pd.json_normalize` 只查看这些数据，使用 `record_path=` 参数。请注意，`pd.json_normalize` 不是一个真正的 I/O 函数，因为它处理的是 Python 对象而不是文件句柄，因此没有 `dtype_backend=` 参数；相反，我们将链接一个 `pd.DataFrame.convert_dtypes` 调用，以获得所需的 pandas 扩展类型：

```py
`pd.json_normalize(     data,     record_path="records" ).convert_dtypes(dtype_backend="numpy_nullable")` 
```

```py
 `name            characteristics.num_leg  characteristics.num_eyes 0   human           2                        2 1   dog             4                        2 2   horseshoe crab  10                       10` 
```

通过提供 `record_path=` 参数，我们能够忽略不需要的 `"pagination"` 键，但不幸的是，我们现在有了一个副作用，就是丢失了包含每条记录重要元数据的 `"type"` 键。为了保留这些信息，你可以使用 `meta=` 参数：

```py
`pd.json_normalize(     data,     record_path="records",     meta="type" ).convert_dtypes(dtype_backend="numpy_nullable")` 
```

```py
 `name    characteristics.num_leg  characteristics.num_eyes  type 0   human   2                        2                         animal 1   dog     4                        2                         animal 2   horseshoe     crab    10                       10                        animal` 
```

# HTML

你可以使用 pandas 从网站读取 HTML 表格。这使得获取像 Wikipedia 上的表格变得容易。

在这个教程中，我们将从 Wikipedia 上关于 *The Beatles Discography* 的页面抓取表格 ([`en.wikipedia.org/wiki/The_Beatles_discography`](https://en.wikipedia.org/wiki/The_Beatles_discography))。特别是，我们想要抓取 2024 年 Wikipedia 上图片中展示的表格：

![一张图表的截图](img/B31091_04_04.png)

图 4.4：The Beatles Discography 的 Wikipedia 页面

在尝试读取 HTML 之前，用户需要安装一个第三方库。在本节的示例中，我们将使用 `lxml`：

```py
`python -m pip install lxml` 
```

## 如何做到这一点

`pd.read_html` 允许你从网站读取表格：

```py
`url = "https://en.wikipedia.org/wiki/The_Beatles_discography" dfs = pd.read_html(url, dtype_backend="numpy_nullable") len(dfs)` 
```

```py
`60` 
```

与我们到目前为止看到的其他 I/O 方法不同，`pd.read_html`不会返回一个`pd.DataFrame`，而是返回一个`pd.DataFrame`对象的列表。让我们看看第一个列表元素是什么样子的：

```py
`dfs[0]` 
```

```py
 `The Beatles discography   The Beatles discography.1 0   The Beatles in 1965       The Beatles in 1965 1   Studio albums             12 (UK), 17 (US) 2   Live albums               5 3   Compilation albums        51 4   Video albums              22 5   Music videos              53 6   EPs                       36 7   Singles                   63 8   Mash-ups                  2 9   Box sets                  17` 
```

上述表格是工作室专辑、现场专辑、合辑专辑等的统计摘要。这不是我们想要的表格。我们可以循环遍历`pd.read_html`创建的每个表格，或者我们可以给它一个提示，以找到特定的表格。

获取我们想要的表格的一种方法是利用`pd.read_html`的`attrs=`参数。此参数接受一个将 HTML 属性映射到值的字典。因为 HTML 中的`id`属性应该在文档中是唯一的，尝试使用`attrs={"id": ...}`来查找表格通常是一个安全的方法。让我们看看我们能否在这里做到这一点。

使用你的网络浏览器检查网页的 HTML（如果你不知道如何做到这一点，请在网上搜索诸如*Firefox inspector*、*Safari Web Inspector*或*Google Chrome DevTools*之类的术语；不幸的是，术语并不标准化）。寻找任何`id`字段、唯一字符串或帮助我们识别所需表格的表格元素属性。

这是原始 HTML 的一部分：

```py
`<table class="wikipedia plainrowheaders" style="text-align:center;">   <caption>List of studio albums, with selected chart positions and certification   </caption>   <tbody>     <tr>       <th rowspan="2" scope="col" style="width:20em;">Title</th>       <th rowspan="2" scope="col" style="width:20em;">Album details<sup id="cite_ref-8" class="reference"><a href="#cite_note-8">[A]</a></sup></th>       ...     </tr>   </tbody>` 
```

不幸的是，我们正在寻找的表格没有`id`属性。我们可以尝试使用上述 HTML 片段中看到的`class`或`style`属性，但这些可能不会是唯一的。

我们可以尝试的另一个参数是`match=`，它可以是一个字符串或正则表达式，用来匹配表格内容。在上述 HTML 的`<caption>`标签中，你会看到文本`"List of studio albums"`；让我们将其作为一个参数试一试。为了提高可读性，我们只需查看每张专辑及其在英国、澳大利亚和加拿大的表现：

```py
`url = "https://en.wikipedia.org/wiki/The_Beatles_discography" dfs = pd.read_html(     url,     match=r"List of studio albums",     dtype_backend="numpy_nullable", ) print(f"Number of tables returned was: {len(dfs)}") dfs[0].filter(regex=r"Title|UK|AUS|CAN").head()` 
```

```py
 `Title                  Peak chart positions                       Title       UK [8][9]   AUS [10]    CAN [11] 0          Please Please Me               1          —          — 1       With the Beatles[B]               1          —          — 2        A Hard Day's Night               1          1          — 3          Beatles for Sale               1          1          — 4                     Help!               1          1          —` 
```

虽然我们现在能够找到表格，但列名不太理想。如果你仔细观察维基百科的表格，你会注意到它在`Peak chart positions`文本和下面国家名称之间部分创建了一个层次结构，这些内容会被 pandas 转换为`pd.MultiIndex`。为了使我们的表格更易读，我们可以传递`header=1`来忽略生成的`pd.MultiIndex`的第一个级别：

```py
`url = "https://en.wikipedia.org/wiki/The_Beatles_discography" dfs = pd.read_html(     url,     match="List of studio albums",     header=1,     dtype_backend="numpy_nullable", ) dfs[0].filter(regex=r"Title|UK|AUS|CAN").head()` 
```

```py
 `Title      UK [8][9]   AUS [10]   CAN [11] 0          Please Please Me              1         —           — 1       With the Beatles[B]              1         —           — 2        A Hard Day's Night              1         1           — 3          Beatles for Sale              1         1           — 4                     Help!              1         1           —` 
```

当我们更仔细地查看数据时，我们可以看到维基百科使用`—`来表示缺失值。如果我们将其作为参数传递给`pd.read_html`的`na_values=`参数，我们将看到`=—=`值被转换为缺失值：

```py
`url = "https://en.wikipedia.org/wiki/The_Beatles_discography" dfs = pd.read_html(     url,     match="List of studio albums",     header=1,     na_values=["—"],     dtype_backend="numpy_nullable", ) dfs[0].filter(regex=r"Title|UK|AUS|CAN").head()` 
```

```py
 `Title                 UK [8][9]   AUS [10]   CAN [11] 0       Please Please Me              1       <NA>       <NA> 1       With the Beatles[B]           1       <NA>       <NA> 2       A Hard Day's Night            1          1       <NA> 3       Beatles for Sale              1          1       <NA> 4       Help!                         1          1       <NA>` 
```

# Pickle

Pickle 格式是 Python 的内置序列化格式。Pickle 文件通常以`.pkl`扩展名结尾。

与之前遇到的其他格式不同，pickle 格式不应该用于在机器间传输数据。它的主要使用场景是将*包含 Python 对象*的 pandas 对象保存在本地机器上，并在以后重新加载。如果你不确定是否应该使用这个格式，我建议你首先尝试 Apache Parquet 格式，它涵盖了更广泛的使用场景。

**不要从不可信来源加载 pickle 文件**。我通常只建议将 pickle 用于自己的分析；不要分享数据，也不要指望从别人那里收到 pickle 格式的数据。

## 如何操作

为了强调 pickle 格式应该仅在 pandas 对象包含 Python 对象时使用，假设我们决定将 Beatles 数据存储为一个包含 `namedtuple` 类型的 `pd.Series`。人们可能会质疑*为什么*要这么做，因为它更适合表示为 `pd.DataFrame`……但不论如何，这样做是有效的：

```py
`from collections import namedtuple Member = namedtuple("Member", ["first", "last", "birth"]) ser = pd.Series([     Member("Paul", "McCartney", 1942),     Member("John", "Lennon", 1940),     Member("Richard", "Starkey", 1940),     Member("George", "Harrison", 1943), ]) ser` 
```

```py
`0     (Paul, McCartney, 1942) 1        (John, Lennon, 1940) 2    (Richard, Starkey, 1940) 3    (George, Harrison, 1943) dtype: object` 
```

我们在本章讨论的其他 I/O 方法都无法准确表示`namedtuple`，因为它是纯 Python 构造。然而，`pd.Series.to_pickle`却能够顺利写出这个内容：

```py
`import io buf = io.BytesIO() ser.to_pickle(buf)` 
```

当你调用 `pd.read_pickle` 时，你将获得你开始时所使用的确切表示：

```py
`buf.seek(0) ser = pd.read_pickle(buf) ser` 
```

```py
`0     (Paul, McCartney, 1942) 1        (John, Lennon, 1940) 2    (Richard, Starkey, 1940) 3    (George, Harrison, 1943) dtype: object` 
```

你可以通过检查单个元素进一步验证这一点：

```py
`ser.iloc[0]` 
```

```py
`Member(first='Paul', last='McCartney', birth=1942)` 
```

再次强调，Apache Parquet 格式应优先于 pickle，只有在 `pd.Series` 或 `pd.DataFrame` 中包含 Python 特定对象且需要回传时，才应作为最后的选择使用。务必**不要从不可信来源加载 pickle 文件**；除非你自己创建了该 pickle 文件，否则强烈建议不要尝试处理它。

# 第三方 I/O 库

虽然 pandas 支持大量格式，但它无法涵盖所有*重要*格式。为此，第三方库应运而生，填补了这个空白。

以下是一些你可能感兴趣的库——它们的工作原理超出了本书的范围，但它们通常遵循的模式是：具有读取函数返回`pd.DataFrame`对象，并且写入方法接受`pd.DataFrame`参数：

+   pandas-gbq 让你与 Google BigQuery 交换数据

+   AWS SDK for pandas 可以与 Redshift 以及更广泛的 AWS 生态系统一起使用

+   Snowflake Connector for Python 帮助与 Snowflake 数据库交换数据

+   pantab 让你将 `pd.DataFrame` 对象在 Tableau 的 Hyper 数据库格式中进出（注意：我也是 pantab 的作者）

# 加入我们社区的 Discord

加入我们社区的 Discord 讨论区，与作者和其他读者互动：

[`packt.link/pandas`](https://packt.link/pandas)

![](img/QR_Code5040900042138312.png)
