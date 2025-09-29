# 第二章. 分析 – 测量代码的性能

提高 R 程序性能的第一步是确定性能瓶颈发生的位置。为此，我们**分析**或测量 R 程序在运行时的性能，相对于各种指标，如执行时间、内存利用率、CPU 利用率和磁盘 I/O。这让我们对程序及其各部分的表现有一个很好的了解，因此我们可以首先解决最大的瓶颈。本章将向您展示如何使用一些简单的工具来测量 R 程序的性能。

这里应用了 80/20 法则。通常，通过解决 20%的最大性能问题，可以实现 80%的可能性能提升。我们将探讨如何确定先解决哪些问题，以便在最少的时间和精力下获得最大的改进。

本章涵盖以下主题：

+   测量总执行时间

+   分析执行时间

+   分析内存利用率

+   使用操作系统工具监控内存利用率、CPU 利用率和磁盘 I/O

+   识别和解决瓶颈

# 测量总执行时间

当人们说他们的程序表现不佳时，他们通常指的是**执行时间**或程序完成执行所需的时间。在许多情况下，执行时间可能是最重要的性能指标，因为它对人和过程有直接影响。较短的执行时间意味着 R 程序员可以更快地完成他的或她的分析，从而更快地得出见解。

结果表明，执行时间也是可以准确且详细测量的最易测量的性能特征（尽管不一定最容易解决）。因此，我们将通过学习如何测量 R 程序的执行时间来学习如何分析 R 代码。我们将学习三种不同的工具来完成这项任务：`system.time()`、`benchmark()` 和 `microbenchmark()`。

## 使用 system.time() 测量执行时间

我们将要了解的第一个分析工具是 `system.time()`。这是一个非常有用的工具，我们可以用它来测量任何 R 表达式的执行时间。

假设我们想知道生成 1000 万个均匀随机变量需要多长时间。看看以下语句及其在 R 控制台运行时的输出：

```py
system.time(runif(1e8))
##  user  system elapsed 
## 2.969   0.166   3.138
```

`runif(1e8)` 表达式生成 0 到 1 之间的 1000 万个随机值。为了测量运行此命令所需的时间，我们只需将此表达式传递给 `system.time()`。

输出包含三个元素，所有元素均以秒为单位：

+   **用户时间**：此元素是指给定表达式执行用户指令所花费的 CPU 时间，例如，遍历一个数组。它不包括其他进程使用的 CPU 时间（例如，如果计算机在后台运行病毒扫描，它所消耗的 CPU 时间不计入）。

+   **系统时间**：系统时间是指为给定表达式执行系统指令所收取的 CPU 时间，例如，打开和关闭文件，或分配和释放内存。这不包括其他进程使用的 CPU 时间。

+   **耗时**：耗时是指执行给定表达式的总时钟时间。它包括 CPU 在处理其他进程上花费的时间以及等待时间（例如，等待文件被打开以供读取）。有时，耗时可能长于用户时间和系统时间的总和，因为 CPU 正在处理其他进程的多任务，或者它必须等待资源（如文件和网络连接）可用。在其他时候，耗时可能短于用户时间和系统时间的总和。这可能会发生在使用多个线程或 CPU 执行表达式时。例如，一个需要 10 秒用户时间的任务，如果有两个 CPU 分担负载，可以在 5 秒内完成。

大多数时候，我们感兴趣的是执行给定表达式的总耗时。当表达式在单个线程上执行（R 的默认设置）时，耗时通常非常接近用户时间和系统时间的总和。如果不是这种情况，则表达式可能花费时间等待资源可用，或者系统上存在许多其他进程正在竞争 CPU 时间。

### 小贴士

在运行 `system.time()` 之前，最好关闭系统上所有不必要的程序和进程，以减少对 CPU 时间的竞争，并获得准确的测量结果。当然，不应关闭防病毒软件或其他任何关键系统软件。

### 注意

`system.time()` 声明实际上返回一个包含五个元素的向量，但它的 `print()` 函数只显示前三个。要查看所有五个元素，我们可以调用 `print(unclass(system.time(expr)))`。其他两个元素是 `expr` 所产生的任何子进程的执行的系统时间和用户时间。在 Windows 机器上，这些不可用，始终会显示为 `NA`。

这是我们多次运行 `system.time()` 并使用相同表达式时发生的情况：

```py
system.time(runif(1e8))
##  user  system elapsed 
## 2.963   0.160   3.128 
system.time(runif(1e8))
##  user  system elapsed 
## 2.971   0.162   3.136 
system.time(runif(1e8))
##  user  system elapsed 
## 2.944   0.161   3.106
```

通过重复运行 `system.time()`，我们每次都会得到略微不同的结果，因为 R 的开销、操作系统缓存机制、其他正在运行的进程以及许多其他因素可能会对执行时间产生轻微影响。

## 使用 rbenchmark 重复时间测量

有时多次运行相同的表达式并获取平均执行时间，甚至获取多次运行中执行时间的分布，这很有帮助。`rbenchmark` CRAN 软件包让我们可以轻松地做到这一点。

首先，安装并加载 `rbenchmark` 软件包：

```py
install.packages("rbenchmark")
library(rbenchmark)
```

接下来，使用 `benchmark()` 运行相同的随机数生成任务 10 次，通过指定 `replications=10`：

```py
bench1 <- benchmark(runif(1e8), replications=10)
bench1
##           test replications elapsed relative user.self
## 1 runif(1e+08)           10   32.38        1    29.781
##   sys.self user.child sys.child
## 1    2.565          0         0
```

结果显示了在 10 次重复中生成 1 亿个均匀随机变量所花费的总系统时间和用户时间。我们可以使用 `within()` 来将时间测量值除以重复次数，从而找到每次重复的平均时间：

```py
within(bench1, {
       elapsed.mean <- elapsed/replications
       user.self.mean <- user.self/replications
       sys.self.mean <- sys.self/replications
       })
##           test replications elapsed relative user.self
## 1 runif(1e+08)           10   32.38        1    29.781
##   sys.self user.child sys.child sys.self.mean user.self.mean
## 1    2.565          0         0        0.2565         2.9781
##   elapsed.mean
## 1        3.238
```

如果我们想知道每次重复的执行时间，或者执行时间在重复中的分布情况呢？我们可以将一个向量而不是单个数字作为 `replications` 参数传递。对于这个向量的每个元素，`benchmark()` 将执行指定的表达式。因此，我们可以得到随机数生成执行一次的 10 个样本，如下面的代码所示。除了用户和系统时间外，`benchmark()` 还返回一个额外的列，`relative`，它表示每次重复的经过时间与最快的一次相比如何。例如，第一次重复花费了最快重复（第四次）1.011 倍的时间，或者运行时间长了 1.1%：

```py
benchmark(runif(1e8), replications=rep.int(1, 10))
##            test replications elapsed relative user.self
## 1  runif(1e+08)            1   3.162    1.011     2.971
## 2  runif(1e+08)            1   3.145    1.005     2.951
## 3  runif(1e+08)            1   3.141    1.004     2.949
## 4  runif(1e+08)            1   3.128    1.000     2.937
## 5  runif(1e+08)            1   3.261    1.043     3.021
## 6  runif(1e+08)            1   3.207    1.025     2.993
## 7  runif(1e+08)            1   3.274    1.047     3.035
## 8  runif(1e+08)            1   3.174    1.015     2.966
## 9  runif(1e+08)            1   3.172    1.014     2.970
## 10 runif(1e+08)            1   3.230    1.033     3.004
##    sys.self user.child sys.child
## 1     0.187          0         0
## 2     0.191          0         0
## 3     0.189          0         0
## 4     0.190          0         0
## 5     0.228          0         0
## 6     0.210          0         0
## 7     0.230          0         0
## 8     0.207          0         0
## 9     0.201          0         0
## 10    0.224          0         0
```

## 使用 microbenchmark 测量执行时间的分布

CRAN 包 `microbenchmark` 提供了另一种测量 R 表达式执行时间的方法。尽管它的 `microbenchmark()` 函数只测量经过时间，而不是用户时间或系统时间，但它可以给出重复运行中执行时间分布的概览。它还自动纠正了与执行时间测试相关的开销。如果你不需要测量用户或系统时间，`microbenchmark()` 函数非常方便，可以用来测量多次重复的短运行任务。我们将在本书中多次使用这个工具。

安装并加载 `microbenchmark` 包：

```py
install.packages("microbenchmark")
library(microbenchmark)
```

现在，使用 `microbenchmark()` 运行相同的随机数生成任务 10 次：

```py
microbenchmark(runif(1e8), times=10)
## Unit: seconds
##          expr      min       lq  median       uq      max
##  runif(1e+08) 3.170571 3.193331 3.25089 3.299966 3.314355
##  neval
##     10
```

统计数据显示了 10 次重复中经过时间的最小值、下四分位数、中位数、上四分位数和最大值。这让我们对相同表达式的不同重复中的经过时间分布有了概念。

# 分析执行时间

到目前为止，我们已经看到了如何测量整个 R 表达式的执行时间。对于包含对其他函数调用等多个部分的更复杂表达式呢？是否有方法可以深入挖掘并分析构成表达式的每个部分的执行时间？R 内置了 `Rprof()` 分析工具，允许我们做到这一点。让我们看看它是如何工作的。

## 使用 Rprof() 分析函数

在这个例子中，我们编写以下 `sampvar()` 函数来计算数值向量的无偏样本方差。这显然不是编写此函数的最佳方式（实际上 R 提供了 `var()` 函数来完成此操作），但它有助于说明代码分析是如何工作的：

```py
# Compute sample variance of numeric vector x
sampvar <- function(x) {
    # Compute sum of vector x
    my.sum <- function(x) {
        sum <- 0
        for (i in x) {
            sum <- sum + i
        }
        sum
    }

    # Compute sum of squared variances of the elements of x from
    # the mean mu
    sq.var <- function(x, mu) {
        sum <- 0
        for (i in x) {
            sum <- sum + (i - mu) ^ 2
        }
        sum
    }

    mu <- my.sum(x) / length(x)
    sq <- sq.var(x, mu)
    sq / (length(x) - 1)
}
```

在 `sampvar()` 函数中，我们定义了两个实用函数：

+   `my.sum()`：通过遍历向量的元素来计算向量的和。

+   `sq.var()`：通过遍历向量的元素，计算向量与给定均值平方偏差的总和。

`sampvar()` 函数首先计算样本均值，然后计算从该均值到平方偏差的总和，最后通过除以 *n-1* 来计算样本方差。

我们可以这样分析 `sampvar()` 函数：

```py
x <- runif(1e7)
Rprof("Rprof.out")
y <- sampvar(x)
Rprof(NULL)
summaryRprof("Rprof.out")
## $by.self
##          self.time self.pct total.time total.pct
## "sq.var"      4.38    58.24       5.28     70.21
## "my.sum"      1.88    25.00       2.24     29.79
## "^"           0.46     6.12       0.46      6.12
## "+"           0.44     5.85       0.44      5.85
## "-"           0.28     3.72       0.28      3.72
## "("           0.08     1.06       0.08      1.06
##
## $by.total
##           total.time total.pct self.time self.pct
## "sampvar"       7.52    100.00      0.00     0.00
## "sq.var"        5.28     70.21      4.38    58.24
## "my.sum"        2.24     29.79      1.88    25.00
## "^"             0.46      6.12      0.46     6.12
## "+"             0.44      5.85      0.44     5.85
## "-"             0.28      3.72      0.28     3.72
## "("             0.08      1.06      0.08     1.06
##
## $sample.interval
## [1] 0.02
##
## $sampling.time
## [1] 7.52
```

这就是代码的工作方式：

1.  `runif(1e7)` 表达式生成一个包含 1000 万个随机数的随机样本。

1.  `Rprof("Rprof.out")` 表达式告诉 R 开始分析。`Rprof.out` 是一个文件的名称，其中存储了分析数据。除非指定了另一个文件路径，否则它将存储在 R 的当前工作目录中。

1.  `sampvar(x)` 表达式调用我们刚刚创建的函数。

1.  `Rprof(NULL)` 表达式告诉 R 停止分析。否则，它将继续分析我们运行的其他 R 语句，但这些语句我们并不打算分析。

1.  `summaryRprof("Rprof.out")` 表达式打印了分析结果。

## 分析结果

结果被分解为几个度量：

+   `self.time` 和 `self.pct` 列表示每个函数的耗时，不包括被函数调用的其他函数的耗时。

+   `total.time` 和 `total.pct` 列表示每个函数的总耗时，包括函数调用内部花费的时间。

从分析数据中，我们得到一些有趣的观察：

+   `sampvar()` 函数的 `self.time` 可以忽略不计（报告为零），这表明运行 `sampvar` 所花费的大部分时间是由它调用的函数贡献的。

+   虽然 `sampvar()` 总共花费了 7.52 秒，但其中 5.28 秒是由 `sq.var()` 贡献的，2.24 秒是由 `my.sum()` 贡献的（参见 `sq.var()` 和 `my.sum()` 的 `total.time`）。

+   `sq.var()` 函数执行所需时间最长（70.21%），看起来是一个开始提高性能的好地方。

+   R 操作符 `-`、`+` 和 `*` 非常快，每个操作的总耗时不超过 0.46 秒，尽管它们被执行了数百万次。

`Rprof()` 通过观察 R 表达式运行时的调用栈来工作，并在固定的时间间隔（默认为每 0.02 秒）对调用栈进行快照，以查看当前正在执行哪个函数。从这些快照中，`summaryRprof()` 可以计算出每个函数花费了多少时间。

为了更直观地查看分析数据，我们可以使用 `proftools` 包。我们还需要从 Bioconductor 仓库安装 `graph` 和 `Rgraphviz` 包：

```py
install.packages("proftools")
source("http://bioconductor.org/biocLite.R")
biocLite(c("graph", "Rgraphviz"))
library(proftools)
p <- readProfileData(filename="Rprof.out")
plotProfileCallGraph(p, style=google.style, score="total")
```

`plotProfileCallGraph()` 函数生成一个直观的视觉分析图。我们使用 `google.style` 模板，该模板使用更大的框来显示 `self.time` 较长的函数。我们还指定 `score="total"` 以根据 `total.time` 为框着色。以下图显示了相同分析数据的输出：

![分析结果](img/9263OS_02_01.jpg)

由 `plotProfileCallGrah()` 生成的 `sampvar()` 的分析数据

从 `sampvar()` 我们可以看到，它有最长的 `total.time` 为 100%。这是预期的，因为它是被分析的功能。运行时间第二长的函数是 `sq.var()`，占用了 70.21% 的运行时间。`sq.var()` 也恰好有最长的 `self.time`，这可以从其框的大小中看出。因此，`sq.var()` 似乎是在解决性能问题时的一个很好的候选者。

`Rprof()` 函数是一个有用的工具，可以帮助我们了解 R 程序不同部分的表现，并快速找到我们可以解决以改进 R 代码整体性能的瓶颈。

# 分析内存利用率

接下来，让我们考虑如何分析 R 代码的内存使用情况。

一种方法是使用 `Rprof()`，通过设置 `memory.profiling` 参数和相应的 `memory` 参数到 `summaryRprof()`：

```py
Rprof("Rprof-mem.out", memory.profiling=TRUE)
y <- sampvar(x)
Rprof(NULL)
summaryRprof("Rprof-mem.out", memory="both")
## $by.self
##          self.time self.pct total.time total.pct mem.total
## "sq.var"      4.16    54.88       5.40     71.24    1129.4
## "my.sum"      1.82    24.01       2.18     28.76     526.9
## "^"           0.56     7.39       0.56      7.39     171.0
## "+"           0.44     5.80       0.44      5.80     129.2
## "-"           0.40     5.28       0.40      5.28     140.2
## "("           0.20     2.64       0.20      2.64      49.7
##
## $by.total
##           total.time total.pct mem.total self.time self.pct
## "sampvar"       7.58    100.00    1656.2      0.00     0.00
## "sq.var"        5.40     71.24    1129.4      4.16    54.88
## "my.sum"        2.18     28.76     526.9      1.82    24.01
## "^"             0.56      7.39     171.0      0.56     7.39
## "+"             0.44      5.80     129.2      0.44     5.80
## "-"             0.40      5.28     140.2      0.40     5.28
## "("             0.20      2.64      49.7      0.20     2.64
##
## $sample.interval
## [1] 0.02
##
## $sampling.time
## [1] 7.58
```

现在的输出显示了一个额外的列 `mem.total`，报告每个函数的内存利用率。对于这个例子，似乎运行 `sampvar()` 需要 1,656 MB 的内存！这对于对包含 1000 万个元素的数值向量进行计算来说似乎异常高，这将在内存中测量为只有 76.3 MB（你可以通过运行 `print(object.size(x), units="auto")` 来检查这一点）。

不幸的是，`mem.total` 是一个误导性的度量，因为 `Rprof()` 将内存使用量归因于在它进行快照时恰好正在运行的函数，但内存可能已经被其他函数使用且尚未释放。此外，R 的垃圾回收器定期将未使用的内存释放给操作系统，因此任何给定时间实际使用的内存可能与 `Rprof()` 报告的内存大相径庭。换句话说，`Rprof()` 提供了在运行 R 代码时分配的总内存量的指示，但没有考虑到垃圾回收器释放的内存。

要查看垃圾回收如何影响内存利用率，我们可以运行以下代码：

```py
> gcinfo(TRUE)
y <- sampvar(x)
## Garbage collection 945 = 886+43+16 (level 0) ... 
## 31.1 Mbytes of cons cells used (59%)
## 82.8 Mbytes of vectors used (66%)
## Garbage collection 946 = 887+43+16 (level 0) ... 
## 31.1 Mbytes of cons cells used (59%)
## 82.8 Mbytes of vectors used (66%)
##... (truncated for brevity) ...
gcinfo(FALSE)
```

`gcinfo(TRUE)` 表达式告诉 R 在垃圾回收器释放内存时通知我们。在我们的机器上，垃圾回收器在运行 `sampvar()` 时被激活了 272 次！尽管 `Rprof()` 报告说总共分配了 1.7 GB 的内存，但垃圾回收器一直在努力释放未使用的内存，以便 R 的总内存消耗保持在可管理的 113.9 MB 左右（*31.1 MB + 82.8 MB*）。

由于 `Rprof()` 测量的是累积分配的内存，而不考虑垃圾回收，因此它不适合确定 R 程序是否会超出系统上的可用内存。`gcinfo()` 通过在每次垃圾回收间隔提供内存消耗的快照，提供了一个更清晰的画面，尽管仍然是一个近似值。

### 注意

在这种情况下，`gcinfo()`和`gc()`函数给出了相当好的内存利用率估计，因为我们的代码只使用标准的 R 操作。一些 R 包使用自定义内存分配器，而`gcinfo()`和`gc()`无法跟踪，因此内存利用率可能会被低估。

# 使用操作系统工具监控内存利用率、CPU 利用率和磁盘 I/O

与执行时间不同，R 没有提供任何好的工具来分析 CPU 利用率和磁盘 I/O。即使是 R 中的内存分析工具也可能无法提供完整或准确的图像。这就是我们转向操作系统提供的系统监控工具来监控 R 程序运行时的计算资源的原因。在 Windows 中是任务管理器或资源监视器，在 Mac OS X 中是活动监视器，在 Linux 中是`top`。运行这些工具时，寻找代表 R 的进程（通常称为`R`或`rsession`）。

我们得到的信息取决于操作系统，但以下是关注 R 资源利用率的几个关键指标：

+   **% CPU 或 CPU 使用率**：R 占用的系统 CPU 时间的百分比

+   **% 内存，驻留内存或工作集**：R 占用的系统物理内存的百分比

+   **交换空间大小或页面输出**：存储在操作系统交换空间中的 R 使用的内存大小

+   **每秒读取或写入的字节数**：R 从/向磁盘读取或写入数据的速率

此外，我们还可能想要监控以下系统级资源利用率指标：

+   **% 可用内存**：系统物理内存中可用于使用的百分比

+   **交换空间大小或页面输出**：存储在操作系统交换空间中的内存总大小

前述指标有助于解决 R 的性能问题：

+   **高 CPU 利用率**：CPU 很可能是 R 性能的主要瓶颈。使用本章中介绍的分析技术来识别代码中占用 CPU 时间最多的部分。

+   **低 CPU 利用率，低可用系统内存，但交换空间大小大，以及高磁盘 I/O**：系统可能正在耗尽物理内存，因此将内存交换到磁盘上。使用第六章中介绍的内存管理技术，*减少 RAM 使用的简单调整*和第七章中介绍的*使用有限 RAM 处理大型数据集*，以减少 R 程序所需的内存。

+   **足够的可用系统内存和高磁盘 I/O**：程序非常频繁地写入/读取磁盘。检查是否有不必要的 I/O 操作，并在有足够可用内存的情况下将中间数据存储在内存中。

# 识别和解决瓶颈

现在我们已经介绍了分析 R 代码的基本技术，我们应该首先尝试解决哪些性能瓶颈？

作为一项经验法则，我们首先尝试改进导致最大性能瓶颈的代码片段，无论是执行时间、内存利用率还是其他指标。这些可以通过前面提到的分析技术来识别。然后我们逐步解决最大的瓶颈，直到程序的整体性能足够好。

如您所回忆的，我们使用`Rprof()`对`varsamp()`函数进行了性能分析。具有最高`self.time`的函数是`sq.var()`。我们如何使这个函数运行得更快呢？我们可以将其写成向量操作的形式`my.sum((x - mu) ^ 2)`，而不是通过遍历`x`的每个元素。正如我们将在下一章中看到的，将循环转换为向量操作是加快许多 R 操作的好方法。实际上，我们甚至可以完全删除该函数，因为新的向量表达式只需要一行：

```py
# Compute sample variance of numeric vector x
sampvar <- function(x) {
    # Compute sum of vector x
    my.sum <- function(x) {
        sum <- 0
        for (i in x) {
            sum <- sum + i
        }
        sum
    }

    mu <- my.sum(x) / length(x)
    sq <- my.sum((x - mu) ^ 2)
    sq / (length(x) - 1)
}

x <- runif(1e7)
Rprof("Rprof-mem.out", memory.profiling=TRUE)
y <- sampvar(x)
Rprof(NULL)
summaryRprof("Rprof-mem.out", memory="both")
## $by.self
##          self.time self.pct total.time total.pct mem.total
## "my.sum"      3.92    85.22       4.60    100.00    1180.6
## "+"           0.66    14.35       0.66     14.35     104.2
## "-"           0.02     0.43       0.02      0.43      83.1
##
## $by.total
##               total.time total.pct mem.total self.time self.pct
## "my.sum"            4.60    100.00    1180.6      3.92    85.22
## "eval"              4.60    100.00    1180.6      0.00     0.00
## "sampvar"           4.60    100.00    1180.6      0.00     0.00
## "source"            4.60    100.00    1180.6      0.00     0.00
## "withVisible"       4.60    100.00    1180.6      0.00     0.00
## "+"                 0.66     14.35     104.2      0.66    14.35
## "-"                 0.02      0.43      83.1      0.02     0.43
##
## $sample.interval
## [1] 0.02
##
## $sampling.time
## [1] 4.6
```

这个改动将运行时间减少了 2.98 秒，并将函数运行时分配的总内存减少了 477 MB。

现在，`my.sum()`函数对总运行时间的贡献达到了显著的 85%。让我们用 R 中的`sum()`函数替换它，该函数运行得更快：

```py
# Compute sample variance of numeric vector x
sampvar <- function(x) {
    mu <- sum(x) / length(x)
    sq <- sum((x - mu) ^ 2)
    sq / (length(x) - 1)
}

x <- runif(1e7)
Rprof("Rprof-mem.out", memory.profiling=TRUE)
y <- sampvar(x)
Rprof(NULL)
summaryRprof("Rprof-mem.out", memory="both")
## $by.self
##     self.time self.pct total.time total.pct mem.total
## "-"      0.08      100       0.08       100      76.2
##
## $by.total
##           total.time total.pct mem.total self.time self.pct
## "-"             0.08       100      76.2      0.08      100
## "sampvar"       0.08       100      76.2      0.00        0
##
## $sample.interval
## [1] 0.02
##
## $sampling.time
## [1] 0.08
```

哇！通过两个简单的步骤，我们将`sampvar()`的运行时间从 7.58 秒减少到 0.08 秒（减少了 99%）。此外，`Rprof()`报告的内存利用率也从超过 1.6 GB 减少到仅有 76.2 MB（减少了 95.4%）。这种内存分配和垃圾回收的减少也在加快我们的代码中发挥了重要作用。

让我们比较我们的代码与 R 函数`var()`的运行速度，后者是用 C 编写的以实现最佳性能（我们将在第四章，*使用编译代码以获得更高的速度*）：

```py
library(microbenchmark)
microbenchmark(sampvar(x), var(x))
## Unit: milliseconds
##        expr      min       lq   median       uq      max neval
##  sampvar(x) 44.31072 44.90836 50.38668 62.14281 74.93704   100
##      var(x) 35.62815 36.60720 37.04430 37.88039 42.85260   100
```

我们的函数的中位运行时间为 50 毫秒，比具有中位数为 37 毫秒的优化 C 版本多出 36%的时间。

前面的练习说明了如何将代码分析作为工作流程的一部分，用于识别、优先排序和修复 R 程序中的性能问题。本书的其余部分将介绍我们可以用来解决特定性能问题的技术。

# 摘要

在本章中，我们学习了如何使用`system.time()`、`benchmark()`（来自`rbenchmark`包）和`microbenchmark()`（来自`microbenchmark`包）来测量 R 表达式的执行时间。我们探讨了如何使用`Rprof()`和`summaryRprof()`来分析 R 程序不同部分的执行时间和内存使用情况，并使用`proftools`包以直观的视觉形式显示结果。

我们还看到了操作系统提供的监控工具在理解 R 程序整体性能以及这些系统度量如何提供关于我们 R 程序可能遇到性能瓶颈的线索方面的作用。

最后，我们学习了如何在实践中应用配置文件技术，通过迭代工作流程来识别、优先排序和解决 R 代码中与性能相关的问题。

在下一章中，我们将学习一些简单的调整来提高 R 代码的运行速度。
