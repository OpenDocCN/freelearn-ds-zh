# 第三章 简单调整以使 R 运行更快

提高 R 代码的速度并不一定涉及高级优化技术，如并行化代码或使其在数据库中运行。实际上，有一些简单的调整，虽然并不总是显而易见，但可以使 R 运行得更快。在本章中，描述了其中的一些调整。这些调整绝对不能涵盖所有可能的简单优化方法。然而，它们构成了获得一些速度提升的最基本、也最常遇到的机会。

本章按照递减的普遍性顺序介绍了这些调整——更普遍的调整几乎可以在所有 R 代码中找到，无论其应用如何。每个调整都伴随着一个示例代码，这些代码故意保持简单，以免不必要的应用特定知识掩盖了对预期概念的说明。在这些所有示例中，都是使用 R 中的随机函数生成人工数据集。

本章涵盖了以下主题：

+   向量化

+   使用内置函数

+   预分配内存

+   使用更简单的数据结构

+   在大型数据上使用哈希表进行频繁查找

+   在 CRAN 中寻找快速替代包

# 向量化

大多数 R 用户都应该遇到过这个第一个调整。本质上，向量化允许 R 运算符将向量作为参数进行快速处理多个值。这与 C、C++和 Java 等一些其他编程语言不同，在这些语言中，多个值的处理通常是通过遍历并应用运算符到向量（或数组）的每个元素来完成的。R 是一种灵活的语言，允许用户使用迭代或向量化进行编程。然而，大多数时候，迭代会带来显著且不必要的计算成本，因为 R 是一种解释型语言，而不是编译型语言。

以以下简单的代码为例。其目的是简单地计算随机向量`data`中每个元素的平方。第一种方法是设置一个`for`循环遍历`data`中的每个元素并单独平方。许多人可能会倾向于采取这种方法，因为这是在其他编程语言中通常的做法。然而，在 R 中，一个更优化的方法是直接在`data`向量上应用平方运算符。这会得到与`for`循环完全相同的结果，但速度要快得多：

```py
N <- 1E5
data <- sample(1:30, size=N, replace=T)
system.time({ 
  data_sq1 <- numeric(N)
  for(j in 1:N) {
    data_sq1[j] <- data[j]²
  } 
})
##  user  system elapsed 
## 0.144   0.011   0.156 
system.time(data_sq2 <- data²)
##  user  system elapsed 
##     0       0       0
```

以下表格显示了随着向量大小（以对数尺度）从 100,000 增加到 100,000,000 时的性能提升。请注意，非向量化方法的计算时间大约是向量化方法的 200 倍，无论向量大小如何。

| **向量大小** | 100,000 | 1,000,000 | 10,000,000 | 100,000,000 |
| --- | --- | --- | --- | --- |
| **非向量化** | 120 ms | 1.19 s | 11.9 s | 117 s |
| **向量化** | 508 μs | 5.67 ms | 52.5 ms | 583 ms |

当 R 执行代码时，它需要在幕后执行许多步骤。一个例子是类型检查。R 对象，如向量，不需要严格定义为特定类型，如整数或字符。可以在整数向量中添加一个字符而不会触发任何错误——R 会自动将向量转换为字符向量。每次在向量上应用运算符时，R 只需要检查一次向量的类型，但使用迭代方法，这种类型检查会像迭代次数一样多次发生，这会产生一些计算成本。

# 内置函数的使用

作为一种编程语言，R 包含底层运算符，例如基本算术运算符，可以用来构建更复杂的运算符或函数。虽然 R 提供了定义函数的灵活性，但与编译语言中的等效函数相比，性能比较几乎总是偏向后者。然而，R 和一些 CRAN 软件包提供了一组丰富的函数，这些函数是用 C/C++ 等编译语言实现的。通常，使用这些函数而不是编写自定义 R 函数来完成相同任务更为可取。

考虑以下随机矩阵 `data` 行求和的简单示例。可以通过调用 `apply()` 函数并设置边距为 1（表示行操作）以及将 `FUN`（或函数）参数设置为 `sum` 来构建执行这些功能的代码。或者，R 提供了一个用于此目的的内置函数，称为 `rowSums`。通过 `system.time` 测量的前者方法的计算时间比后者方法长 11 倍，后者是一个优化并预编译的 C 函数：

```py
data <- rnorm(1E4*1000)
dim(data) <- c(1E4,1000)
system.time(data_sum1 <- apply(data, 1, sum)) 
##  user  system elapsed 
## 0.241   0.053   0.294 
system.time(data_sum2 <- rowSums(data))
##  user  system elapsed 
## 0.026   0.000   0.026
```

说到优化函数，我们提高 R 代码速度的努力不应仅限于 R 伴随的预编译函数。多年来，开源社区已经开发了一系列优化库，R 可以利用这些库。以基本线性代数子程序（BLAS）为例（更多信息请参阅 [`www.netlib.org/blas/`](http://www.netlib.org/blas/)）。它在 20 世纪 70 年代为 Fortran 开发，此后由于矩阵运算构成了许多领域算法的构建块，因此被其他语言（包括 R）广泛使用。现在有许多 BLAS 的实现，其中一些包括以多线程方式执行矩阵运算的能力。

例如，Mac OS X 版本的 R 默认启用了 BLAS。使用的 BLAS 实现是 R 中称为 `libRblas.0.dylib` 的参考 BLAS。然而，Mac OS X 自带其自己的 BLAS 版本，`libBLAS.dylib`，该版本针对其硬件进行了优化。R 可以配置为使用优化的 BLAS，通过在终端执行以下命令：

```py
$ cd /Library/Frameworks/R.framework/Resources/lib
$ ln -sf /System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/Current/libBLAS.dylib libRblas.dylib
```

为了测试使用不同 BLAS 库的效果，下面的 R 代码在一个大随机矩阵上执行简单的矩阵乘法。使用 R 的默认 BLAS 库，我们完成这项任务大约需要 7 秒钟。在将 R 指向优化的 BLAS 之后，同样的任务在约十分之一的时间内完成：

```py
data <- rnorm(1E7)
dim(data) <- c(1E4, 1E3)
system.time(data_mul <- t(data) %*% data)
##  user  system elapsed 
## 7.123   0.015   7.136
system.time(data_mul <- t(data) %*% data) # with optimized BLAS
##  user  system elapsed 
## 1.304   0.005   0.726
```

可供您下载的 BLAS 版本有 Windows 和 Linux。如果 R 是用启用了 BLAS 编译的，即通过在从 R 的源代码编译 R 时设置配置选项为`--enable-BLAS-shlib`，那么在 BLAS 版本之间切换的方式与 Mac OS X 类似：通过用新的库文件替换默认的 BLAS 库文件。在 Windows 中，默认库位于`R_HOME\bin\x64\Rblas.dll`；而在 Linux 中，它位于`R_HOME/lib/libRblas.so`。

# 预分配内存

大多数强类型编程语言，如 C、C++和 Java，通常要求在向其应用任何操作之前声明一个向量（或数组）。这种声明实际上预分配了向量所需的内存空间。在某些特殊情况下会使用动态内存分配，但这很少是首选，主要是因为动态内存分配会减慢程序的速度。每次调整向量大小时，程序都需要执行额外的步骤，包括将向量复制到更大的或更小的内存块中，并删除旧向量。如果预分配了内存，这些步骤就不需要了。

当涉及到预分配内存时，R 与其他编程语言没有区别。然而，作为一个解释型语言，它施加的控制较少，因此用户很容易忽略这一点——如果向量的内存没有预分配，R 不会抛出任何编译错误。尽管如此，在 R 中没有预分配内存可能会导致执行时间显著变长，尤其是在向量很大时。

为了演示这一点，让我们看一下下面的 R 代码。它展示了两种生成一系列随机数的方法，其中每个向量元素定义为前一个元素的值加减一个介于-5 到 5 之间的随机整数。第一种方法（将结果存储在`data_series1`中）绕过了向量内存的预分配，即它从一个只有一个元素的向量开始，并在每次迭代中添加一个新元素。第二种方法（结果在`data_series2`中）通过声明一个大小为`N`的数值向量来预分配内存。预分配的空间，由向量的索引表示，在每次迭代中填充。通过预分配内存，一个包含 10,000 个元素的向量的计算时间比动态分配快 10 倍。通过改变向量大小进行的基准测试，如即将到来的表格所示，显示当预分配内存时，计算时间呈线性增长，而当动态分配内存时，增长呈超线性。因此，避免在 R 中不必要的动态内存分配对于性能至关重要：

```py
N <- 1E4
data_series1 <- 1
system.time({
  for (j in 2:N) {
    data_series1 <- c(data_series1,
                      data_series1[j-1]+sample(-5:5, size=1))
  }
})
##  user  system elapsed 
## 0.254   0.004   0.257 
data_series2 <- numeric(N)
data_series2[1] <- 1
system.time({
  for (j in 2:N) {
    data_series2[j] <- data_series2[j-1]+sample(-5:5, size=1)
  }
})
##  user  system elapsed 
## 0.066   0.003   0.068
```

| **向量大小** | 10 | 100 | 1000 | 10,000 |
| --- | --- | --- | --- | --- |
| **动态分配** | 0 | 0.006 | 0.288 | 25.373 |
| **预分配** | 0.001 | 0.006 | 0.062 | 0.577 |

在这一点上，比较 R 中的 `apply` 函数族与循环很有趣。大多数 R 用户都熟悉 `apply()` 函数及其变体，包括 `lapply()`、`sapply()` 和 `tapply()`。它们提供了对集合（例如 `data.frame`、`list` 或 `vector`/`matrix`）中的单个元素重复执行相同操作的手段。实际上，`apply` 函数族可以作为 R 中循环的可能替代品，前提是迭代之间没有依赖关系。除了简化表达式（通常可以将多行 `for` 循环表达为单行的 `apply()` 调用）之外，`apply` 函数族还提供了自动处理内存预分配和其他家务活动（如删除循环索引）的好处。

但 `apply` 是否比循环有性能优势？以下代码提供了一个答案。使用了两种不同的方法来生成一个大小在 1 到 30 之间随机设定的正态分布随机向量的列表。第一种方法使用 `for` 循环，而第二种使用 `lapply()`。对这两种方法应用 `system.time()` 显示，`lapply()` 比循环快得多：

```py
N <- 1E5
data <- sample(1:30, size=N, replace=T)
data_rand1 <- list()
system.time(for(i in 1:N) data_rand1[[i]] <- rnorm(data[i]))
##   user  system elapsed 
## 33.891   1.241  35.120 
system.time(data_rand2 <- lapply(data, rnorm))
##  user  system elapsed 
## 0.597   0.037   0.633
```

但请注意，`for` 循环是天真地实现的，没有预先分配内存。以下代码现在使用预分配的内存对其进行修改。其计算时间显著减少，比 `lapply()` 慢不到十分之一秒：

```py
data_rand3 <- vector("list", N)
system.time(for(i in 1:N) data_rand3[[i]] <- rnorm(data[i]))
##  user  system elapsed 
## 0.737   0.036   0.773
```

为了更有说服力地建立这一点，使用 `microbenchmark()` 重复进行了比较，每次表达式运行 100 次。结果表明，`lapply()` 比循环有轻微的性能优势：

```py
microbenchmark(data_rand2 <- lapply(data, rnorm),
               for(i in 1:N) data_rand3[[i]] <- rnorm(data[i]))
## Unit: milliseconds
##                                              expr      min
##                 data_rand2 <- lapply(data, rnorm) 441.1108
##  for (i in 1:N) data_rand3[[i]] <- rnorm(data[i]) 531.1212
##       lq     mean   median       uq      max neval
## 459.9666 498.1296 477.4583 517.4329 634.7849   100
## 555.8512 603.7997 581.5236 662.2536 745.4247   100
```

基于此，在 R 中尽可能用 `apply` 替换 `for` 循环的一般观点是有效的，但性能提升可能不会非常显著。在第六章“减少 RAM 使用量的简单技巧”中，将讨论 `apply` 的另一个好处——它揭示了 R 代码中可以并行化的部分。

# 简单数据结构的使用

许多 R 用户会同意，`data.frame` 作为一种数据结构是 R 中数据分析的得力工具。它提供了一种直观的方式来表示典型的结构化数据集，其中行和列分别代表观测值和变量。`data.frame` 对象也比矩阵提供了更多的灵活性，因为它允许不同类型的变量（例如，单个 `data.frame` 中的字符和数值变量）。此外，在 `data.frame` 仅存储相同类型变量的情况下，基本的矩阵运算可以方便地应用于它，而无需任何显式的强制转换。然而，这种便利性可能会带来性能下降。

在`data.frame`上执行矩阵运算比在矩阵上慢。其中一个原因是大多数矩阵运算首先将`data.frame`强制转换为`matrix`，然后再进行计算。因此，在可能的情况下，应该使用`matrix`代替`data.frame`。下面的代码演示了这一点。目标是简单地在一个矩阵及其等效的`data.frame`表示上执行行求和。使用`matrix`表示比使用`data.frame`表示快约 3 倍：

```py
data <- rnorm(1E4*1000)
dim(data) <- c(1E4,1000)
system.time(data_rs1 <- rowSums(data))
##  user  system elapsed 
## 0.026   0.000   0.026 
data_df <- data.frame(data)
system.time(data_rs2 <- rowSums(data_df))
##  user  system elapsed 
## 0.060   0.015   0.076
```

然而，在许多 R 的案例中，使用`data.frame`是不可避免的，例如，当数据集包含混合变量类型时。在这种情况下，也有一个简单的调整可以改善`data.frame`上最常用的操作之一，即子集操作的速度。通常通过以下代码通过逻辑测试对`data.frame`的行（或列）进行条件化来实现子集操作：

```py
data <- rnorm(1E5*1000)
dim(data) <- c(1E5,1000)
data_df <- data.frame(data)
system.time(data_df[data_df$X100>0 & data_df$X200<0,])
##  user  system elapsed 
## 2.436   0.221   2.656
```

一种替代方法是使用`which`函数包装条件。如以下所示，速度显著提高：

```py
system.time(data_df[which(data_df$X100>0 & data_df$X200<0),])
##  user  system elapsed 
## 0.245   0.086   0.331
```

# 在大数据上频繁查找时使用哈希表

数据分析中的一个常见任务是数据查找，这通常通过 R 中的列表实现。例如，为了查找客户的年龄，我们可以定义一个列表，比如`cust_age`，将值设置为客户的年龄，将名称设置为相应的客户名称（或 ID），即`names(cust_age) <- cust_name`。在这种情况下，要查找 John Doe 的年龄，可以调用以下内容：`cust_age[["John_Doe"]]`。然而，R 中列表的实现并没有针对查找进行优化；在包含*N*个元素的列表上进行查找需要*O(N)*的时间复杂度。这意味着列表中索引较晚的值需要更多的时间来查找。随着*N*的增长，这种影响会变得更强。当程序需要频繁查找时，累积效应可能会非常显著。提供更优化数据查找的列表的替代方案是哈希表。在 R 中，这可以通过 CRAN 包*hash*获得。哈希表的查找需要*O(1)*的时间复杂度。

下面的代码演示了在哈希表中进行查找比在列表中查找的优势。它模拟了从随机列表及其等效哈希表表示中进行的 1,000 次查找。列表所需的总计算时间为 6.14 秒，而哈希表为 0.31 秒。一个权衡是生成哈希表比列表需要更多的时间。但对于需要在大数据上频繁查找的程序来说，这种开销可以忽略不计：

```py
data <- rnorm(1E6)
data_ls <- as.list(data)
names(data_ls) <- paste("V", c(1:1E6), sep="")
index_rand <- sample(1:1E6, size=1000, replace=T)
index <- paste("V", index_rand, sep="")
list_comptime <- sapply(index, FUN=function(x){
  system.time(data_ls[[x]])[3]})
sum(list_comptime)
## [1] 6.144
library(hash)
data_h <- hash(names(data_ls), data)
hash_comptime <- sapply(index, FUN=function(x){
  system.time(data_h[[x]])[3]})
sum(hash_comptime)
## [1] 0.308
```

# 在 CRAN 中寻找快速替代包

R 的一个关键优势是其丰富且活跃的开源社区，CRAN。截至编写本文时，CRAN 上有超过 6,000 个 R 包。鉴于这一点，多个包提供相同的功能是很常见的。其中一些替代方案专门设计用来提高基础包或现有 CRAN 包的性能。其他替代方案虽然没有明确针对性能提升，但作为副产品实现了这一点。

为了实现性能提升而开发的替代快速包的例子是 `fastcluster` 包。它是为了通过 `hclust` 函数提高基础包提供的层次聚类速度而开发的。根据层次聚类过程中每次分支合并后距离矩阵的更新方式，其时间复杂度可能会有显著变化。`fastcluster` 包是使用优化的 C++ 代码开发的，与 `hclust` 中实现的例程相比，速度显著提高。以下 R 代码比较了两个函数在具有 10,000 行和 100 列的随机矩阵上的性能：

```py
data <- rnorm(1E4*100)
dim(data) <- c(1E4,100)
dist_data <- dist(data)
system.time(hc_data <- hclust(dist_data))
##  user  system elapsed 
## 3.488   0.200   4.081 
library(fastcluster)
system.time(hc_data <- hclust(dist_data))
##  user  system elapsed 
## 1.972   0.123   2.127
```

一个具有多个实现且其中一个实现比其他实现更快的函数的例子是**主成分分析**（**PCA**）。PCA 是一种降维技术，通过将数据集投影到正交轴（称为主成分）上以最大化数据集的方差来实现其目标。PCA 最常见的方法是通过数据集协方差矩阵的特征值分解。但还有其他方法。在 R 中，这些替代方法体现在两个 PCA 函数 `prcomp` 和 `princomp` 中（都是 `stats` 包的一部分）。以下代码在具有 100,000 行和 100 列的随机矩阵上的快速比较表明，`princomp` 比 `prcomp` 快近 2 倍：

```py
data <- rnorm(1E5*100)
dim(data) <- c(1E5,100)
system.time(prcomp_data <- prcomp(data))
##  user  system elapsed 
## 4.101   0.091   4.190 
system.time(princomp_data <- princomp(data))
##  user  system elapsed 
## 2.505   0.071   2.576 
```

还有其他快速包的例子，既有明确的也有隐含的。它们包括：

+   `fastmatch`: 这提供了 R 的基础 `match` 函数的更快版本

+   `RcppEigen`: 这包括线性建模 `lm` 的更快版本

+   `data.table`: 这提供了比标准 `data.frame` 操作更快的数据处理操作

+   `dplyr` : 这提供了一套高效操作数据框对象（data frame-like objects）的工具

# 摘要

本章描述了一些简单的调整来提高 R 代码的速度。其中一些调整是众所周知的，但在实践中经常被忽视；其他一些则不那么明显。无论它们的性质如何，尽管它们很简单，这些低垂的果实可以提供显著的性能提升，有时甚至比后续章节中讨论的高级优化还要多。因此，这些调整应被视为优化 R 代码的第一步。

在下一章中，我们将看到如何通过使用编译代码来进一步提升 R 的性能。
