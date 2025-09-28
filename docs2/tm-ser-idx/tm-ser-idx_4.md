

# iSAX – 实现

在继续本章并开始编写代码之前，请确保您已经很好地理解了上一章中涵盖的信息，因为本章全部关于在 Python 中实现 iSAX。作为一个一般原则，如果您不能手动执行一项任务，那么您将无法在计算机的帮助下执行该任务——同样的原则适用于构建和使用 iSAX 索引。

在阅读本章时，请记住，我们正在创建一个适合内存的 iSAX 索引，并且不使用任何外部文件来存储每个终端节点的子序列。原始的 iSAX 论文建议使用外部文件来存储每个终端节点的子序列，主要是因为当时与今天相比，RAM 的限制更大，而今天我们可以轻松地拥有具有许多 CPU 核心和超过 64 GB RAM 的计算机。因此，使用 RAM 使得整个过程比使用磁盘文件要快得多。然而，如果您系统上的 RAM 不多，并且正在处理大型时间序列，您可能会使用交换空间，这会减慢处理速度。

在本章中，我们将涵盖以下主要主题：

+   快速浏览 `isax` Python 包

+   存储子序列的类

+   iSAX 节点的类

+   整个 iSAX 索引的类

+   解释缺失的部分

+   探索剩余的文件

+   使用 iSAX Python 包

# 技术要求

代码的 GitHub 仓库可以在 [`github.com/PacktPublishing/Time-Series-Indexing`](https://github.com/PacktPublishing/Time-Series-Indexing) 找到。每个章节的代码都在其自己的目录中。因此，本章的代码可以在 `ch04` 文件夹及其子文件夹中找到。

第一部分简要介绍了本章为特定目的开发的 Python 包，这个包奇怪地被命名为 `isax`，然后再详细介绍。

那么，错误怎么办？

我们已经尽力提供无错误的代码。然而，任何程序都可能出现错误，尤其是当程序长度超过 100 行时！这就是为什么理解 iSAX 索引的操作和构建原理以及 SAX 表示至关重要，以便能够理解代码中可能存在的小或大问题，或者能够将现有实现移植到不同的编程语言。我使用同事提供的 Java 实现作为起点编写了 iSAX 的 Python 版本。

# 快速浏览 iSAX Python 包

在本节中，我们将首次了解 iSAX Python 包，以更好地了解其支持的功能。虽然我们将从我们在 *第二章* 中开发的 `sax` 包的代码开始，但我们打算将那个包重命名为 `isax` 并创建额外的源代码，该代码命名为 `isax.py`。

`isax` 目录中 Python 文件的结构如下：

```py
$ tree isax/
isax/
├── SAXalphabet
├── __init__.py
├── isax.py
├── sax.py
├── tools.py
└── variables.py
1 directory, 6 files
```

因此，总共有六个文件。您已经从 `sax` 包中知道了其中五个。唯一的新一个是 `isax.py` 源代码文件，这是本章的核心文件。此外，我们还将向 `variables.py` 文件添加更多全局变量，并将一些函数添加到 `tools.py`。

在 `isax.py` 文件中找到的方法列表（不包括 `__init__()` 函数）如下：

```py
$ grep -w def isax/isax.py | grep -v __init__
    def insert(self, ts, ISAX):
    def nTimeSeries(self):
    def insert(self, ts_node):
```

我们之所以谈论方法而不是函数，是因为每个函数都附加到一个 Python 类上，这使其自动成为该类的方法。

此外，如果我们还包括 `__init__()` 函数，那么我们可能会对 Python 文件中找到的类数量有一个很好的预测。在这种情况下，您可能想运行 `grep -w def -n1` `isax/isax.py`：

```py
$ grep -w def -n1 isax/isax.py
5-class TS:
6:    def __init__(self, ts, segments):
7-        self.index = 0
--
11-class Node:
12:    def __init__(self, sax_word):
13-        self.left = None
--
19-    # Follow algorithm from iSAX paper
20:    def insert(self, ts, ISAX):
21-        # Accessing a subsequence
--
127-
128:    def nTimeSeries(self):
129-        if self.terminalNode == False:
--
141-class iSAX:
142:    def __init__(self):
143-        # This is now a hash table
--
148-
149:    def insert(self, ts_node):
150-        # Array with number of segments
```

因此，我们有三个类，分别命名为 `TS`、`Node` 和 `iSAX`。

下一节将讨论 `isax.py` 的方法及其所属的类。

## 存储子序列的类

在本小节中，我们将解释用于 `TS` 的 Python 类。类的定义如下：

```py
class TS:
    def __init__(self, ts, segments):
        self.index = 0
        self.ts = ts
        self.maxCard = sax.createPAA(ts,
            variables.maximumCardinality, segments)
```

定义该类的对象时，我们需要提供 `ts` 参数，它是一个存储为 NumPy 数组的子序列，以及使用 `segments` 参数指定的段数。之后，`maxCard` 字段将自动初始化为具有最大基数该子序列的 SAX 表示。`index` 参数是可选的，并保持子序列在原始时间序列中的位置。iSAX 不使用 `index` 参数，但有一个这样的字段是好的。

此类没有附加任何方法，这与下一个将要介绍的 `Node` 类不同。

## 表示 iSAX 节点的类

在本小节中，我们将解释用于*保持内部和终端节点*的 Python 结构。这是该包的一个重要部分及其功能：

```py
class Node:
    def __init__(self, sax_word):
        self.left = None
        self.right = None
        self.terminalNode = False
        self.word = sax_word
        self.children = [TS] * variables.threshold
```

如果我们处理的是内部节点，则 `terminalNode` 字段设置为 `False`。然而，如果 `terminalNode` 字段的布尔值设置为 `True`，则我们正在处理一个终端节点。

`word` 字段包含节点的 SAX 表示。最后，`left` 和 `right` 字段是内部节点的两个子节点的链接，而 `children` 字段是一个包含终端节点子序列的列表。

`Node` 类有两个方法：

+   `insert()`: 此方法用于向节点添加子序列

+   `nTimeSeries()`: 此方法用于计算终端节点中存储的子序列数量

接下来，让我们谈谈表示整个 iSAX 索引的类。

## 整个 iSAX 索引的类

`isax` 包的最后一个类用于表示整个 iSAX 索引：

```py
class iSAX:
    def __init__(self):
        # This is now a hash table
        self.children = {}
        # HashTable for storing Nodes
        self.ht = {}
        self.length = 0
```

`children` 字段包含根节点的子节点——实际上，iSAX 类的实例是 iSAX 索引的根节点。

`ht` 字段，它是一个字典，包含了 iSAX 索引中的所有节点。每个键是节点的 SAX 表示，它是*唯一的*，每个值是一个 `Node` 实例。最后，`length` 字段包含了存储在 iSAX 索引中的子序列数量，这是一个可选字段。

`iSAX` 类只有一个方法，称为 `insert()`，用于将子序列插入到 iSAX 索引中。

我们为什么要使用这三个类？

iSAX 索引的实现包含三个不同的实体：子序列、节点以及 iSAX 索引本身，它由索引的根节点表示。iSAX 包含节点，而节点包含其他节点或子序列。这些实体中的每一个都有自己的类。

到目前为止，我们已经了解了我们包中使用的 Python 类的详细信息。下一节将介绍实现缺失的部分。

# 解释缺失的部分

在本节中，我们将展示类方法的实现。我们首先从 `iSAX` 类的 `insert()` 函数开始，这个函数不应与 `Node` 类的 `insert()` 函数混淆。在 Python 以及许多其他编程语言中，类是独立的实体，这意味着只要在类命名空间内是唯一的，它们就可以有相同名称的方法。

我们将分八部分展示 `Node.insert()` 的代码。该方法接受两个参数——除了 `self`，表示当前的 `Node` 对象之外——这两个参数是我们试图插入的子序列以及 `Node` 实例所属的 iSAX 索引。

为什么需要一个 iSAX 实例作为参数？我们需要它以便能够通过访问 `iSAX.ht` 来向 iSAX 索引添加新节点。

`insert()` 的第一部分如下：

```py
    # Follow algorithm from iSAX paper
    def insert(self, ts, ISAX):
        # Accessing a subsequence
        variables.nSubsequences += 1
        if self.terminalNode:
            if self.nTimeSeries() == variables.threshold:
                variables.nSplits += 1
                # Going to duplicate self Node
                temp = Node(self.word)
                temp.children = self.children
                temp.terminalNode = True
                # The current Terminal node becomes
                # an inner node
                self.terminalNode = False
                self.children = None
```

`insert()` 的第一件事是检查我们是否正在处理一个终端节点。这是因为如果我们正在处理终端节点，我们将尝试将给定的子序列存储在终端节点中，而不会有任何延迟。第二个检查是终端节点是否已满。如果已满，那么*我们将进行分裂*。首先，我们使用 `temp = Node(self.word)` 语句复制当前节点，并将当前终端节点通过将 `terminalNode` 的值更改为 `False` 变成内部节点。在此阶段，我们必须创建两个新的空节点，它们将成为当前节点的两个子节点——前者将在接下来的代码摘录中实现。

`insert()` 函数的第二部分如下：

```py
                # Create TWO new Terminal nodes
                new1 = Node(temp.word)
                new1.terminalNode = True
                new2 = Node(temp.word)
                new2.terminalNode = True
                n1Segs = new1.word.split('_')
                n2Segs = new2.word.split('_')
```

在前面的代码中，我们创建了两个新的终端节点，这两个节点将成为即将分裂的节点的子节点。这两个新节点目前具有与即将分裂的节点相同的 SAX 表示，并成为内部节点。它们 SAX 表示的改变，即分裂的标志，将在接下来的代码中实现。

第三部分包含以下代码：

```py
                # This is where the promotion
                # strategy is selected
                if variables.defaultPromotion:
                    tools.round_robin_promotion(n1Segs)
                else:
                    tools.shorter_first_promotion(n1Segs)
                # New SAX_WORD 1
                n1Segs[variables.promote] =
                    n1Segs[variables.promote] + "0"
                # CONVERT it to string
                new1.word = "_".join(n1Segs)
                # New SAX_WORD 2
                n2Segs[variables.promote] =
                    n2Segs[variables.promote] + "1"
                # CONVERT it to string
                new2.word = "_".join(n2Segs)
                # The inner node has the same
                # SAX word as before but this is
                # not true for the two
                # NEW Terminal nodes, which should
                # be added to the Hash Table
                ISAX.ht[new1.word] = new1
                ISAX.ht[new2.word] = new2
                # Associate the 2 new Nodes with the
                # Node that is being splitted
                self.left = new1
                self.right = new2
```

在代码摘录的开头，我们处理提升策略，该策略在`tools.py`文件中实现，这在*The tools.py file*部分有解释，并且与定义将要提升的 SAX 词（段）有关。

之后，代码通过两个字符串操作创建了分割的两个 SAX 表示——这是我们将 SAX 词（段）存储为字符串以及我们使用列表来保存整个 SAX 表示的主要原因。之后，我们将 SAX 表示转换为存储在`new1.word`和`new2.word`中的字符串，然后使用`ISAX.ht[new1.word] = new1`和`ISAX.ht[new2.word] = new2`将这些相应的节点放入 iSAX 索引中。在`iSAX.ht` Python 字典中找到这两个节点的键是它们的 SAX 表示。代码的最后两条语句通过定义内部节点的`left`和`right`字段将两个新的终端节点与内部节点关联起来，从而表示其两个子节点。

`Node.insert()`方法的第四部分代码如下：

```py
                # Check all TS in original node
                # and put them
                # in one of the two children
                #
                # This is where the actual
                # SPLITTING takes place
                #
                for i in range(variables.threshold):
                    # Accessing a subsequence
                    variables.nSubsequences += 1
                    # Decrease TS.maxCard to
                    # current Cardinality
                    tempCard =
                        tools.promote(temp.children[i],
                        n1Segs)
                    if tempCard == new1.word:
                        new1.insert(temp.children[i], ISAX)
                    elif tempCard == new2.word:
                        new2.insert(temp.children[i], ISAX)
                    else:
                        if variables.overflow == 0:
                            print("OVERFLOW:", tempCard)
                        variables.overflow =
                            variables.overflow + 1
                # Now insert the INITIAL TS node!
                # self is now an INNER node
                self.insert(ts, ISAX)
                if variables.defaultPromotion:
                    # Next time, promote the next segment
                    Variables.promote = (variables.promote
                        + 1) % variables.segments
```

我们可以肯定的是，在将之前存储在成为内部节点的终端节点中的子序列分割后，我们不会出现溢出。然而，我们仍然需要调用`self.insert(ts, ISAX)`来插入之前造成溢出的子序列，并查看会发生什么。

最后的`if`检查我们是否使用默认的提升策略，即轮询策略，如果是这样，它将提升段切换到下一个。

但我们如何知道是否存在溢出情况？如果将子序列提升到比其当前基数（`tempCard`）更高的基数，而这个子序列不能分配给两个新创建的终端节点（`new1.word`或`new2.word`）中的任何一个，那么我们知道它没有被提升。因此，我们有一个溢出条件。这体现在`if tempCard == new1.word:`块的`else:`分支中。

`Node.insert()`的第五部分如下：

```py
            else:
                # TS is added if we have a Terminal node
                self.children[self.nTimeSeries()] = ts
```

当我们处理一个非满的终端节点时，会执行之前的`else`代码。因此，我们将给定的子序列存储在`children`列表中——这是向 iSAX 索引添加新子序列的理想方式。

`insert()`函数的第六部分代码如下：

```py
        else:
            # Otherwise, we are dealing with an INNER node
            # and we should add it to the
            # INNER node by trying
            # to find an existing terminal node
            # or create a new one
            # See whether it is going to be
            # included in the left
            # or the right child
            left = self.left
            right = self.right
```

如果我们正在处理一个内部节点，我们必须根据子序列的 SAX 表示来决定它将进入左子节点还是右子节点，以便最终找到将存储该子序列的终端节点。这就是过程的开始。

第七部分包含以下代码：

```py
            leftSegs = left.word.split('_')
            # Promote
            tempCard = tools.promote(ts, leftSegs)
```

在前面的代码中，我们改变（减少）子序列的最大基数以适应左节点的基数——我们本可以使用右节点，因为两个节点使用相同的基数。持有新基数的`tempCard`变量将被用来决定子序列在树中要遵循的路径，直到找到适当的终端节点。

`Node.insert()`的最后一部分如下：

```py
            if tempCard == left.word:
                left.insert(ts, ISAX)
            elif tempCard == right.word:
                right.insert(ts, ISAX)
            else:
                if variables.overflow == 0:
                    print("OVERFLOW:", tempCard, left.word,
                        right.word)
                variables.overflow = variables.overflow + 1
        return
```

如果`tempCard`与左节点或右节点的 SAX 表示不匹配，那么我们知道*它没有被提升*，这意味着我们有一个溢出条件。

这是`Node.insert()`实现背后的逻辑——代码中存在许多你可以阅读的注释，并且你可以添加自己的`print()`语句以更好地理解流程。

为什么我们要在子序列中存储最大基数？

存储这个子序列最大基数的原因是我们可以轻松地降低这个最大基数，而无需进行诸如从头开始计算新的 SAX 表示等困难的计算。这种小的优化使得分割操作变得更快。

`Node`类的另一个方法称为`nTimeSeries()`，其实现如下：

```py
    def nTimeSeries(self):
        if self.terminalNode == False:
            print("Not a terminal node!")
            return
        n = 0
        for i in range(0, variables.threshold):
            if type(self.children[n]) == TS:
                n = n + 1
        return n
```

所展示的函数返回存储在终端节点中的子序列数量。首先，`nTimeSeries()`确保我们在遍历`children`列表的内容之前正在处理一个终端节点。如果存储值的类型是`TS`，那么我们有一个子序列。

之后，我们将讨论并解释`iSAX`类的`insert()`方法，该方法分为三个部分。当我们要向 iSAX 索引添加子序列时，会调用`iSAX.insert()`方法。

`iSAX.insert()`的第一个部分如下：

```py
    def insert(self, ts_node):
        # Array with number of segments
        # For cardinality 1
        segs = [1] * variables.segments
        # Get cardinality 1 from ts_node
        # in order to find its main subtree
        lower_cardinality = tools.lowerCardinality(segs,
            ts_node)
        lower_cardinality_str = ""
        for i in lower_cardinality:
            lower_cardinality_str = lower_cardinality_str +
                "_" + i
        # Remove _ at the beginning
        lower_cardinality_str = lower_cardinality_str[
            1:len(lower_cardinality_str)]
```

代码的这一部分找到根节点中将要放置给定子序列的子节点。`lower_cardinality_str`值用作查找根节点相关子节点的键——`tools.lowerCardinality()`函数将在稍后解释。

`iSAX.insert()`的第二部分包含以下代码：

```py
        # Check whether the SAX word with CARDINALITY 1
        # exists in the Hash Table.
        # If not, create it and update Hash Table
        if self.ht.get(lower_cardinality_str) == None:
            n = Node(lower_cardinality_str)
            n.terminalNode = True
            # Add it to the hash table
            self.children[lower_cardinality_str] = n
            self.ht[lower_cardinality_str] = n
            n.insert(ts_node, self)
```

如果具有`lower_cardinality_str` SAX 表示的根节点的子节点找不到，我们创建相应的根子节点并将其添加到`self.children`哈希表（字典）中，并调用`insert()`将给定的子序列放在那里。

`iSAX.insert()`的最后一部分如下：

```py
        else:
            n = self.ht.get(lower_cardinality_str)
            n.insert(ts_node, self)
        return
```

如果具有`lower_cardinality_str` SAX 表示的根节点的子节点存在，那么我们尝试插入该子序列，从而调用`insert()`。

在这一点上，我们从`iSAX`类级别转到`Node`类级别。

但`isax.py`并不是唯一包含新代码的文件。下一节将展示对剩余包文件的添加和更改，以完成实现。

# 探索剩余的文件

除了 `isax.py` 文件外，`isax` Python 包由更多的源代码文件组成，主要是因为它基于 `sax` 包。我们将从 `tools.py` 文件开始。

## `tools.py` 文件

与我们最初在 *第二章* 中看到的 `tools.py` 源代码文件相比，有一些新增内容，这主要与提升策略有关。如前所述，我们支持两种提升策略：轮询和从左到右。

轮询策略在这里实现：

```py
def round_robin_promotion(nSegs):
    # Check if there is a promotion overflow
    n = power_of_two(variables.maximumCardinality)
    t = 0
    while len(nSegs[variables.promote]) == n:
        # Go to the next SAX word and promote it
        Variables.promote = (variables.promote + 1) %
            variables.segments
        t += 1
        if t == variables.segments:
            if variables.overflow == 0:
                print("Non recoverable Promotion overflow!")
            return
```

在轮询情况下，我们试图找到比指定最大基数（一个不满的段）的数字更少的右侧段。如果前一次提升发生在最后一个段，那么我们就回到第一个段并从头开始。为了计算最大基数（SAX 单词的长度）的二进制位数，我们使用 `power_of_two()` 函数，该函数对于基数 `8` 返回 `3`，对于基数 `16` 返回 `4`，依此类推。如果我们遍历给定 SAX 表示的所有段（`nSegs`）并且所有段都具有最大长度，我们知道存在溢出条件。

也称为 **最短优先** 的从左到右策略在这里实现：

```py
def shorter_first_promotion(nSegs):
    length = len(nSegs)
    pos = 0
    min = len(nSegs[pos])
    for i in range(1,length):
        if min > len(nSegs[i]):
            min = len(nSegs[i])
            pos = i
    variables.promote = pos
```

从左到右的提升策略遍历给定 SAX 表示变量（`nSegs`）的所有段，从左到右，并找到最左边的最小长度段。因此，如果第二和第三段具有相同的最小长度，该策略将选择第二个，因为它是最左边的可用段。之后，它将 `variables.promote` 设置为所选段值。

接下来，我们将讨论 `tools.py` 中驻留的两个附加函数，它们被称为 `promote()` 和 `lowerCardinality()`。

`promote()` 函数的实现如下：

```py
def promote(node, segments):
    new_sax_word = ""
    max_array = node.maxCard.split("_")[
        0:variables.segments]
    # segments is an array
    #
    for i in range(variables.segments):
        t = len(segments[i])
        new_sax_word = new_sax_word + "_" +
            max_array[i][0:t]
    # Remove _ from the beginning of the new_sax_word
    new_sax_word = new_sax_word[1:len(new_sax_word)]
    return new_sax_word
```

`promote()` 函数将现有 SAX 表示（`node`）的段的数字长度复制到给定的子序列（`s`）中，以便它们在所有 SAX 单词中都具有相同的基数。这 *允许我们比较* 这两个 SAX 表示。

`lowerCardinality()` 的实现如下：

```py
def lowerCardinality(segs, ts_node):
    # Get Maximum Cardinality
    max = ts_node.maxCard
    lowerCardinality = [""] * variables.segments
    # Because max is a string, we need to split.
    # The max string has an
    # underscore character at the end.
    max_array = max.split("_")[0:variables.segments]
    for i in range(variables.segments):
        t = segs[i]
        lowerCardinality[i] = max_array[i][0:t]
    return lowerCardinality
```

`lowerCardinality()` 函数降低了一个节点在其所有 SAX 单词（段）中的基数。这主要是由 `iSAX.insert()` 函数需要的，以便将子序列放入根的适当子节点中。在我们将子序列放入根的适当子节点之后，我们一次提升子序列 SAX 表示的单个段，以找出它在 iSAX 索引中的位置。记住，所有 iSAX 节点的键都是 SAX 表示，通常它们的段有不同的基数。

如何测试单个函数

个人而言，我更喜欢创建小的命令行实用工具来测试复杂的函数，理解其操作，并可能发现错误！

让我们创建两个小的命令行实用工具来更详细地展示`promote()`和`lowerCardinality()`的使用。

首先，我们在`usePromote.py`实用工具中演示了`promote()`函数，该实用工具包含以下代码：

```py
#!/usr/bin/env python3
from isax import variables
from isax import isax
import numpy as np
variablesPromote = 0
maximumCardinality = 8
segments = 4
def promote(node, s):
    global segments
    new_sax_word = ""
    max_array = node.maxCard.split("_")[0:segments]
    for i in range(segments):
        t = len(s[i])
        new_sax_word = new_sax_word + "_" +
            max_array[i][0:t]
    new_sax_word = new_sax_word[1:len(new_sax_word)]
    return new_sax_word
```

重要的是要记住，`promote()`函数通过将子序列的最大 SAX 表示（`s`）降低以匹配存储在`node`参数中的给定 SAX 表示，来模拟现有 SAX 表示的段长度。

`usePromote.py`的其余部分如下：

```py
def main():
    global variablesPromote
    global maximumCardinality
    global segments
    variables.maximumCardinality = maximumCardinality
    ts = np.array([1, 2, 3, 4])
    t = isax.TS(ts, segments)
    SAX_WORD = "0_0_1_1_"
    Segs = SAX_WORD.split('_')
    print("Max cardinality:", t.maxCard)
    SAX_WORD = "00_0_1_1_"
    Segs = SAX_WORD.split('_')
    print("P1:", promote(t, Segs))
    SAX_WORD = "000_0_1_1_"
    Segs = SAX_WORD.split('_')
    print("P2:", promote(t, Segs))
    SAX_WORD = "000_01_1_1_"
    Segs = SAX_WORD.split('_')
    print("P3:", promote(t, Segs))
    SAX_WORD = "000_011_1_100_"
    Segs = SAX_WORD.split('_')
    print("P4:", promote(t, Segs))
if __name__ == '__main__':
    main()
```

在`usePromote.py`文件中，所有内容都是硬编码的，因为我们只想更多地了解`promote()`函数的使用，而不想了解其他内容。然而，由于`promote()`函数在`isax`包中有许多依赖项，我们必须将其整个实现放入我们的脚本中，并对 Python 代码进行必要的修改。

给定一个子序列`ts`和一个`TS`类实例`t`，我们可以使用最大基数来计算`ts`的 SAX 表示，然后将其降低以匹配其他 SAX 词的基数。

运行`usePromote.py`生成以下输出：

```py
$ ./usePromote.py
Max cardinality: 000_010_101_111_
P1: 00_0_1_1
P2: 000_0_1_1
P3: 000_01_1_1
P4: 000_010_1_111
```

输出显示，给定子序列的最大基数（`000_010_101_111`）已被降低以匹配四个其他 SAX 词的基数。

之后，我们在`useLCard.py`实用工具中演示了`lowerCardinality()`函数，该实用工具包含以下代码：

```py
#!/usr/bin/env python3
from isax import variables
from isax import tools
from isax import isax
import numpy as np
def main():
    global maximumCardinality
    global segments
    # Used by isax.TS()
    variables.maximumCardinality = 8
    variables.segments = 4
    ts = np.array([1, 2, 3, 4])
    t = isax.TS(ts, variables.segments)
    Segs = [1] * variables.segments
    print(tools.lowerCardinality(Segs ,t))
    Segs = [2] * variables.segments
    print(tools.lowerCardinality(Segs ,t))
    Segs = [3] * variables.segments
    print(tools.lowerCardinality(Segs ,t))
if __name__ == '__main__':
    main()
```

这次，我们没有在我们的代码中放置`lowerCardinality()`的实现，因为它有较少的依赖项，可以直接从`tools.py`文件中使用。我们传递给`lowerCardinality()`的参数是*我们想要在每一个 SAX 词中得到的数字位数*。所以，`1`表示一位数字，这意味着基数是 2¹，而`3`表示三位数字，计算出的基数是 2³。

再次强调，在`useLCard.py`中，所有内容都是硬编码的，因为我们只想更多地了解`lowerCardinality()`函数的使用，而不想了解其他内容。运行`useLCard.py`生成以下输出：

```py
$ ./useLCard.py
['0', '0', '1', '1']
['00', '01', '10', '11']
['000', '010', '101', '111']
```

因此，给定一个 SAX 表示为`000_010_101_111`的子序列，我们计算其基数分别为`2`、`4`和`8`的 SAX 表示。

接下来，我们将展示对`variables.py`文件的修改，该文件包含全局变量，这些变量可以被包中的所有文件或使用`isax`包的实用工具访问。

## `variables.py`文件

本小节展示了更新后的`variables.py`文件的内容，其中包含在代码的任何地方都可以访问的变量。

需要多少功能才算足够？

请记住，有时我们可能需要包含有助于调试或未来可能需要的功能，因此，我们可能需要包含不会立即或总是使用的变量或实现函数。只需记住，在想要支持一切和取悦每个人（这是不可能的）以及想要支持绝对最小功能（这通常缺乏灵活性）之间保持良好的平衡。

`variables.py` 文件的内容如下：

```py
# This file includes all variables for the isax package
#
maximumCardinality = 32
breakpointsFile = "SAXalphabet"
# Breakpoints in breakpointsFile
elements = ""
slidingWindowSize = 16
segments = 0
# Maximum number of time series in a terminal node
threshold = 100
# Keeps number of splits
nSplits = 0
# Keep number of accesses of subsequences
nSubsequences = 0
# Currently supporting TWO promotion strategies
defaultPromotion = True
# Number of overflows
overflow = 0
# Floating point precision
precision = 5
# Segment to promote
promote = 0
```

`variables.promote`变量定义了如果需要，将要提升的 SAX 词。简单来说，我们根据`variables.promote`的值创建一个分割的两个节点的 SAX 表示——我们提升由`variables.promote`值定义的段。每次我们有一个分割时，`variables.promote`都会根据提升（分割）策略更新，并准备好下一次分割。

如果你希望查看同一文件两个版本之间的更改，可以使用`diff(1)`实用程序。在我们的情况下，`ch03`目录中找到的`variables.py`文件与当前版本之间的差异如下：

```py
2c2
< # This file includes all variables for the sax package
---
> # This file includes all variables for the isax package
13a14,16
> # Breakpoints in breakpointsFile
> elements = ""
>
20,21c23,24
< # Breakpoints in breakpointsFile
< elements = ""
---
> # Maximum number of time series in a terminal node
> threshold = 100
22a26,37
> # Keeps number of splits
> nSplits = 0
>
> # Keeps number of accesses of subsequences
> nSubsequences = 0
>
> # Currently supporting TWO promotion strategies
> defaultPromotion = True
>
> # Number of overflows
> overflow = 0
>
24a40,42
>
> # Segment to promote
> promote = 0
```

以`>`开头的行显示了`ch04/isax/variables.py`文件的内容，而以`<`开头的行显示了`ch03/sax/variables.py`文件中的语句。

下一个小节将讨论`sax.py`，它并没有发生太多变化。

## `sax.py` 文件

`sax.py` 文件并没有任何实际上的改动。然而，我们应该修改它的`import`语句，因为它不再是一个独立的包，而是另一个不同名称的包的一部分。因此，我们需要修改以下两个语句：

```py
from sax import sax
from sax import variables
```

我们用以下这些语句来替换它们：

```py
from isax import sax
from isax import variables
```

除了这些，不需要进行额外的更改。

现在我们已经知道了`isax`包的源代码，是时候看看这个代码的实际应用了。

# 使用 iSAX Python 包

在本节中，我们将使用`isax` Python 包来开发实用的命令行工具。但首先，我们将学习如何从用户那里读取 iSAX 参数。

## 读取 iSAX 参数

本小节说明了如何读取 iSAX 参数，包括时间序列的文件名，以及如何为其中的一些参数设置默认值。尽管我们在*第二章*中看到了相关的代码，但这次我们将更详细地解释这个过程。此外，代码还将展示我们如何使用这些输入参数来设置位于`./isax/variables.py`文件内的相关变量。提醒一下，存储在`./isax/variables.py`或类似文件中的变量——碰巧我们使用的是`./isax/variables.py`——只要我们成功导入了相关文件，就可以在我们的代码的任何地方访问。

我们需要创建一个 iSAX 索引

作为提醒，要创建一个 iSAX 索引，我们需要一个时间序列和一个阈值值，这是终端节点可以持有的最大子序列数，以及一个段值和一个基数值。最后，我们还需要一个滑动窗口大小。

作为一条经验法则，当处理全局变量时，最好使用长且描述性的名称。此外，为全局参数提供默认值也是一个好的实践。

这里展示了 `parameters.py` 的 Python 代码：

```py
#!/usr/bin/env python3
import argparse
from isax import variables
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--segments",
        dest = "segments", default = "4",
        help="Number of Segments", type=int)
    parser.add_argument("-c", "--cardinality",
        dest = "cardinality", default = "32",
        help="Cardinality", type=int)
    parser.add_argument("-w", "--window", dest = "window",
        default = "16", help="Sliding Window Size",
        type=int)
    parser.add_argument("TS1")
    args = parser.parse_args()
    variables.segments = args.segments
    variables.maximumCardinality = args.cardinality
    variables.slidingWindowSize = args.window
    windowSize = variables.slidingWindowSize
    maxCardinality = variables.maximumCardinality
    f1 = args.TS1
    print("Time Series:", f1, "Window Size:", windowSize)
    print("Maximum Cardinality:", maxCardinality,
        "Segments:", variables.segments)
if __name__ == '__main__':
    main()
```

所有工作都是由 `argparse` 包和用于定义命令行参数和选项的 `parser.add_argument()` 语句完成的。`dest` 参数定义了参数的名称——这个名称将在以后用于读取参数的值。

`parser.add_argument()` 的另一个参数被称为 `type`，它允许我们定义参数的数据类型。这可以避免许多问题，并减少将字符串转换为实际值所需的代码，因此尽可能使用 `type`。

之后，我们调用 `parser.parse_args()`，然后我们就可以读取任何想要的 `rgparse` 参数。

运行 `parameters.py` 会生成以下输出：

```py
$ ./parameters.py -s 2 -c 32 -w 16 ts1.gz
Time Series: ts1.gz Window Size: 16
Maximum Cardinality: 32 Segments: 2
```

如果发生错误，`parameters.py` 会生成以下输出：

```py
$ ./parameters.py -s 1 -c cardinality ts1.gz
usage: parameters.py [-h] [-s SEGMENTS] [-c CARDINALITY] [-w WINDOW] TS1
parameters.py: error: argument -c/--cardinality: invalid int value: 'cardinality'
```

在这种情况下，错误是 `cardinality` 参数是一个字符串，而我们是期望一个整数值。错误输出非常具有信息性。

如果缺少必要的参数，`parameters.py` 会生成以下输出：

```py
$ ./parameters.py
usage: parameters.py [-h] [-s SEGMENTS] [-c CARDINALITY] [-w WINDOW] TS1
parameters.py: error: the following arguments are required: TS1
```

下一个部分展示了我们如何处理时间序列的子序列以创建一个 iSAX 索引。

## 如何处理子序列以创建 iSAX 索引

这是一个非常重要的子部分，因为在这里，我们解释了用于存储 iSAX 索引每个子序列数据的 Python 结构。

代码不会说谎！

如果你对每个子序列的字段和数据存储有疑问，请查看 Python 代码以了解更多信息。文档可能会说谎，但代码永远不会。

`subsequences.py` 脚本展示了我们如何创建子序列，如何将它们存储在 Python 数据结构中，以及我们如何处理它们：

```py
#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from isax import sax
from isax import variables
class TS:
    def __init__(self, ts, index):
        self.ts = ts
        self.sax = sax.createPAA(ts,
            variables.maximumCardinality,
            variables.segments)
        self.index = index
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--window", dest = "window",
        default = "16", help="Sliding Window Size",
        type=int)
    parser.add_argument("-s", "--segments",
        dest = "segments", default = "4",
        help="Number of Segments", type=int)
    parser.add_argument("-c", "--cardinality",
        dest = "cardinality", default = "32",
        help="Cardinality", type=int)
    parser.add_argument("TS")
    args = parser.parse_args()
    windowSize = args.window
    variables.segments = args.segments
    variables.maximumCardinality = args.cardinality
    file = args.TS
```

我们再次定义 `TS` 类，并在 `subsequences.py` 中使用这个版本，以便能够在不改变 `isax` 包代码的情况下对 `TS` 类进行更多修改。在此之前，我们已经读取了程序的参数，并且准备读取时间序列：

```py
    ts = pd.read_csv(file, names=['values'],
        compression='gzip', header = None)
    ts_numpy = ts.to_numpy()
    length = len(ts_numpy)
```

目前，我们使用 `ts_numpy` 变量将时间序列存储为 NumPy 数组：

```py
    # Split sequence into subsequences
    n = 0
    for i in range(length - windowSize + 1):
        # Get the actual subsequence
        ts = ts_numpy[i:i+windowSize]
        # Create new TS node based on ts
        ts_node = TS(sax.normalize(ts), i)
        n = n + n
```

`for` 循环根据滑动窗口大小将时间序列分割成子序列。每个子序列的归一化版本存储在具有三个成员的 `TS()` 结构中：子序列的归一化版本（`ts`）、子序列的 SAX 表示（`sax`）以及子序列在时间序列中的位置（`index`）。`TS()` 结构的最后一个成员允许我们在需要时找到子序列的原始版本。

现在，检查以下代码：

```py
    print("Created", n, "TS() nodes")
if __name__ == '__main__':
    main()
```

完成后，脚本会打印出已处理的子序列数量。

只有在我们将子序列的 SAX 表示存储在其基于最大基数构建的 Python 结构中之后，我们才准备好将该子序列放入 iSAX 索引中。因此，下一个步骤（此处未展示）是将每个`TS()`节点放入 iSAX 索引中。

`subsequences.py`的输出会告诉你已经处理了多少个子序列：

```py
$ ./subsequences.py ts1.gz
Created 35 TS() nodes
```

总结来说，这是我们处理子序列以便将它们添加到 iSAX 索引中的方法。在下一小节中，我们将创建我们的第一个 iSAX 索引！

## 创建我们的第一个 iSAX 索引

在本节中，我们将首次创建一个 iSAX 索引。但首先，我们将展示用于此目的的 Python 实用工具。`createiSAX.py`的 Python 代码分为四个部分。第一部分如下：

```py
#!/usr/bin/env python3
from isax import variables
from isax import isax
from isax import tools
from isax import sax
import sys
import pandas as pd
import numpy as np
import time
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--segments",
        dest = "segments", default = "16",
        help="Number of Segments", type=int)
    parser.add_argument("-c", "--cardinality",
        dest = "cardinality", default = "16",
        help="Cardinality", type=int)
    parser.add_argument("-w", "--windows", dest = "window",
        default = "16", help="Sliding Window Size",
        type=int)
    parser.add_argument("-t", "--threshold",
        dest = "threshold", default = "1000",
        help="Threshold for split", type=int)
    parser.add_argument("-p", "--promotion",
        action='store_true',
        help="Define Promotion Strategy")
    parser.add_argument("TSfile")
    args = parser.parse_args()
```

这第一部分是关于`import`语句和通过`argparse`读取所需参数。

`createiSAX.py`的第二部分如下：

```py
    variables.segments = args.segments
    variables.maximumCardinality = args.cardinality
    variables.slidingWindowSize = args.window
    variables.threshold = args.threshold
    variables.defaultPromotion = args.promotion
    file = args.TSfile
    maxCardinality = variables.maximumCardinality
    segments = variables.segments
    windowSize = variables.slidingWindowSize
    if tools.power_of_two(maxCardinality) == -1:
        print("Not a power of 2:", maxCardinality)
        sys.exit()
    if variables.segments > variables.slidingWindowSize:
        print("Segments:", variables.segments,
            "Sliding window:", variables.slidingWindowSize)
        print("Sliding window size should be bigger than #
            of segments.")
        sys.exit()
    print("Max Cardinality:", maxCardinality, "Segments:",
        variables.segments,
        "Sliding Window:", variables.slidingWindowSize,
        "Threshold:", variables.threshold,
        "Default Promotion:", variables.defaultPromotion)
```

在`createiSAX.py`的这一部分中，我们将参数分配给相关的局部和全局变量，并进行一些测试以确保参数是有意义的。使用局部变量的原因是有更小的变量名可以工作。`print()`语句将参数输出到屏幕上。

`createiSAX.py`的第三部分包含以下代码：

```py
    ts = pd.read_csv(file, names=['values'],
        compression='gzip')
    ts_numpy = ts.to_numpy()
    length = len(ts_numpy)
    #
    # Initialize iSAX index
    #
    ISAX = isax.iSAX()
```

在这部分中，我们读取压缩的时间序列文件，创建一个 NumPy 变量来存储整个时间序列。之后，我们初始化一个变量来存储 iSAX 索引。由于类的名称是`iSAX`，相关变量被初始化为`isax.iSAX()`类的实例。

`createiSAX.py`的最后部分包含以下代码：

```py
    # Split sequence into subsequences
    for i in range(length - windowSize + 1):
        # Get the subsequence
        ts = ts_numpy[i:i+windowSize]
        # Create new TS node based on ts
        ts_node = isax.TS(ts, segments)
        ISAX.insert(ts_node)
if __name__ == '__main__':
    main()
```

这最后部分根据滑动窗口大小分割时间序列，创建`TS()`对象，并使用`ISAX`变量通过`iSAX`类的`insert()`方法将它们插入到 iSAX 索引中——记住，是`iSAX.insert()`调用`Node.insert()`。

运行`createiSAX.py`会产生以下输出：

```py
$ ./createiSAX.py ts1.gz
Max Cardinality: 16 Segments: 16 Sliding Window: 16 Threshold: 1000 Default Promotion: False
$ ./createiSAX.py
usage: createiSAX.py [-h] [-s SEGMENTS] [-c CARDINALITY] [-w WINDOW] [-t THRESHOLD] [-p] TSfile
createiSAX.py: error: the following arguments are required: TSfile
```

好处在于`createiSAX.py`为所有 iSAX 参数提供了默认值。然而，提供包含时间序列的文件的路径是必需的。

在下一小节中，我们将开发一个命令行实用工具，用于计算 iSAX 索引中的子序列总数。

## 计算 iSAX 索引的子序列

这是一个非常实用的工具，它不仅展示了如何遍历整个 iSAX 索引，还允许你计算 iSAX 索引的所有子序列，并确保在过程中没有遗漏任何子序列，这可以用于测试目的。

执行计数的`countSub.py`代码如下——其余的实现与`createiSAX.py`相同：

```py
    # Visit all entries in Dictionary
    # Count TS in Terminal Nodes
    sum = 0
    for k in ISAX.ht:
        t = ISAX.ht[k]
        if t.terminalNode:
            sum += t.nTimeSeries()
    print(length - windowSize + 1, sum)
```

代码访问 iSAX 类的 `ISAX.ht` 字段，因为这是 iSAX 索引中所有节点都保存的地方。如果我们正在处理一个终端节点，那么我们调用 `nTimeSeries()` 方法来找到存储在该终端节点中的子序列数量。我们对所有终端节点都这样做，然后我们就完成了。最后一条语句打印出理论上的子序列数量以及实际在 iSAX 索引中找到的子序列数量。只要这两个值相同，我们就没问题。

在 `ch04` 目录中运行 `countSub.py` 在一个短时间序列上，生成以下类型的输出：

```py
$ ./countSub.py ts1.gz
Max Cardinality: 16 Segments: 16 Sliding Window: 16 Threshold: 1000 Default Promotion: False
35 35
```

下一个子节显示了构建 iSAX 索引所需的时间。

## 创建 iSAX 索引需要多长时间？

在本小节中，我们将计算计算机创建 iSAX 索引所需的时间。iSAX 构建阶段中任何延迟的主要原因是对节点的分割和子序列的重新排列。我们分割得越广泛，生成索引所需的时间就越长。

计算 iSAX 索引创建所需时间的 `howMuchTime.py` 代码如下——其余的实现与 `createiSAX.py` 相同：

```py
    start_time = time.time()
    print("--- %.5f seconds ---" % (time.time() –
        start_time))
```

第一条语句位于我们使用 `pd.read_csv()` 开始读取时间序列文件之前，第二条语句位于我们完成分割并将时间序列插入 iSAX 索引之后。

`howMuchTime.py` 处理 `ts1.gz` 的输出类似于以下内容：

```py
$ ./howMuchTime.py -w 2 -s 2 ts1.gz
Max Cardinality: 16 Segments: 2 Sliding Window: 2 Threshold: 1000 Default Promotion: False
--- 0.00833 seconds ---
```

由于 `ts1.gz` 是一个包含 50 个元素的短时间序列，输出并不那么有趣。因此，让我们尝试使用 `howMuchTime.py` 在更大的时间序列上。

以下输出显示了 `howMuchTime.py` 在 macOS 机器上创建包含 500,000 个元素的时间序列 iSAX 索引所需的时间——你可以在自己的机器上创建相同长度的时间序列，并尝试相同的命令或使用提供的文件，该文件名为 `500k.gz`：

```py
$ ./howMuchTime.py 500k.gz
Max Cardinality: 16 Segments: 16 Sliding Window: 16 Threshold: 1000 Default Promotion: False
--- 114.80277 seconds ---
```

使用以下命令创建了 `500k.gz` 文件：

```py
$ ./ch01/synthetic_data.py 500000 -1 1 > 500k
$ gzip 500k
```

以下输出显示了 `howMuchTime.py` 在 macOS 机器上创建包含 2,000,000 个元素的时间序列 iSAX 索引所需的时间——你可以在自己的机器上创建相同长度或更长的时序，并尝试相同的命令或使用提供的文件，该文件名为 `2M.gz`：

```py
$ ./howMuchTime.py 2M.gz
Max Cardinality: 16 Segments: 16 Sliding Window: 16 Threshold: 1000 Default Promotion: False
--- 450.37358 seconds ---
```

使用以下命令创建了 `2M.gz` 文件：

```py
$ ./ch01/synthetic_data.py 2000000 -10 10 > 2M
$ gzip 2M
```

我们可以得出的一个有趣的结论是，对于四倍大的时间序列，我们的程序构建它大约需要四倍的时间。然而，情况并不总是如此。

此外，创建 iSAX 索引所需的时间并不能完全说明问题，尤其是在繁忙的机器或内存较少的慢速机器上进行测试时。更重要的是节点分割的数量以及访问子序列的次数。访问子序列的最小次数等于时间序列的子序列数量。然而，当发生分割时，我们必须重新访问涉及到的子序列，以便根据新创建的 SAX 表示和终端节点进行分配。分割和重新访问子序列会增加 iSAX 的构建时间。

因此，我们将创建 `howMuchTime.py` 的修改版本来打印节点分割的数量以及子序列访问的总数。新工具的名称是 `accessSplit.py`。执行分割和子序列访问计数的语句已经在 `isax/isax.py` 中存在，我们只需要访问两个全局变量，即 `variables.nSplits` 和 `variables.nSubsequences`，以获取结果。

使用默认参数在 `500k.gz` 上运行 `accessSplit.py` 会产生以下类型的输出：

```py
$ ./accessSplit.py 500k.gz
Max Cardinality: 16 Segments: 16 Sliding Window: 16 Threshold: 1000 Default Promotion: False
Number of splits: 0
Number of subsequence accesses: 499985
```

这个输出告诉我们什么？它告诉我们没有发生分割！在实践中，这意味着该特定 iSAX 索引的根节点只有终端节点作为子节点。这是好是坏？一般来说，这意味着*索引像哈希表一样工作*，其中哈希函数是计算 SAX 表示的函数。大多数情况下，这不是我们希望索引具有的理想形式，因为我们一开始就可以使用哈希表！

如果我们使用不同的参数在相同的时间序列上运行 `accessSplit.py`，我们将得到关于 iSAX 索引构建的完全不同的输出：

```py
$ ./accessSplit.py -w 1024 -s 8 -c 32 500k.gz
Max Cardinality: 32 Segments: 8 Sliding Window: 1024 Threshold: 1000 Default Promotion: False
Number of splits: 4733
Number of subsequence accesses: 16370018
```

这个输出告诉我们什么？它告诉我们即使在相对较小的时间序列上，iSAX 参数在 iSAX 索引创建时间中起着巨大的作用。然而，子序列访问的次数大约是时间序列子序列总数的 33 倍，这相当大，因此效率不高。

让我们现在尝试在更大的时间序列 `2M.gz` 上运行 `accessSplit.py`，看看会发生什么：

```py
$ ./accessSplit.py 2M.gz
Max Cardinality: 16 Segments: 16 Sliding Window: 16 Threshold: 1000 Default Promotion: False
Number of splits: 0
Number of subsequence accesses: 1999985
```

如前所述，我们正在使用 iSAX 索引作为哈希表，这不是我们期望的行为。让我们尝试使用不同的参数：

```py
$ ./accessSplit.py -s 8 -c 32 2M.gz
Max Cardinality: 32 Segments: 8 Sliding Window: 16 Threshold: 1000 Default Promotion: False
Number of splits: 3039
Number of subsequence accesses: 13694075
```

这次，访问子序列的次数大约是时间序列长度的七倍，这比我们处理 `500k.gz` 文件时更为现实。

我们将在 *第五章* 中再次使用 `accessSplit.py`。但到目前为止，我们将更多地了解 iSAX 索引的溢出问题。

## 处理 iSAX 溢出

在本小节中，我们将对溢出情况进行实验。请记住，在`variables.py`中存在一个专门的全局参数，它保存了由于溢出而被忽略的子序列数量。除此之外，这还有助于你更快地修复 iSAX 参数，因为你知道溢出有多严重。通常，修复溢出最简单的方法是增加阈值值，但这样做在搜索 iSAX 索引或比较两个 iSAX 索引时可能会产生严重影响。

与`createiSAX.py`相比，`overflow.py`的 Python 代码只有一个变化，那就是以下语句，因为导致溢出的 SAX 表示默认情况下会被打印出来：

```py
print("Number of overflows:", variables.overflow)
```

这主要是因为功能内置在`isax`包中，当第一次发生溢出时会自动打印一条消息，我们只需访问`variables.overflow`变量来找出总的溢出次数。

使用`500k.gz`时间序列处理时，`overflow.py`的输出包括以下信息：

```py
$ ./overflow.py -w 1024 -s 8 500k.gz
Max Cardinality: 16 Segments: 8 Sliding Window: 1024 Threshold: 1000 Default Promotion: False
OVERFLOW: 1000_0111_0111_1000_1000_0111_0111_1000
Number of overflows: 303084
```

之前的输出告诉我们，导致溢出的第一个 SAX 表示是`1000_0111_0111_1000_1000_0111_0111_1000`，总共发生了`303084`次溢出——我们可能还有更多导致溢出的 SAX 表示，但我们决定只打印第一个。这意味着有`303084`个子序列没有被插入到 iSAX 索引中，与时间序列的长度相比，这是一个非常大的数字。

现在我们尝试使用其他提升策略执行相同的命令，看看会发生什么：

```py
$ ./overflow.py -w 1024 -s 8 500k.gz -p
Max Cardinality: 16 Segments: 8 Sliding Window: 1024 Threshold: 1000 Default Promotion: True
Non recoverable Promotion overflow!
OVERFLOW: 1000_0111_0111_1000_1000_0111_0111_1000
Number of overflows: 303084
```

结果显示，我们得到了相同类型的溢出和完全相同的总溢出次数。这完全合理，因为溢出情况与提升策略无关，而是与 SAX 表示有关。*不同的提升策略可能会稍微改变 iSAX 索引的形状，但它与* *溢出情况* *无关*。

由于`303084`是一个很大的数字，我们可能需要大幅增加 iSAX 索引的容量，但又不能创建一个不必要的大的 iSAX 索引。因此，考虑到这一点，我们可以尝试通过改变 iSAX 索引的参数来解决溢出问题。那么，让我们尝试通过增加阈值值来这样做：

```py
$ ./overflow.py -w 1024 -s 8 -c 16 -t 1500 500k.gz
Max Cardinality: 16 Segments: 8 Sliding Window: 1024 Threshold: 1500 Default Promotion: False
OVERFLOW: 0111_1000_1000_1000_1000_0111_0111_0111
Number of overflows: 176454
```

因此，看起来我们减少了一半的溢出次数，这对开始来说是个好事。然而，尽管我们使用了与之前相同的基数，但这次导致第一次溢出的却是不同的 SAX 表示（`0111_1000_1000_1000_1000_0111_0111_0111`），这意味着增加的阈值值解决了之前由`1000_0111_0111_1000_1000_0111_0111_1000` SAX 表示引起的溢出条件。

让我们通过增加`cardinality`值并同时降低阈值值来再试一次：

```py
$ ./overflow.py -w 1024 -s 8 -c 32 -t 500 500k.gz
Max Cardinality: 32 Segments: 8 Sliding Window: 1024 Threshold: 500 Default Promotion: False
Number of overflows: 0
```

因此，我们最终找到了一组适用于`1024`滑动窗口大小和`500k.gz`数据集的参数组合。

有没有找到哪些参数有效和无效的配方？没有，因为这主要取决于数据集的值和滑动窗口大小。你越使用并实验 iSAX 索引，你就越会了解哪些参数对于给定的数据集和滑动窗口大小效果最好。

因此，在本节的最后，我们了解了 iSAX 溢出，并介绍了一种解决这种情况的技术。

# 摘要

在本章中，我们看到了`isax` Python 包的实现细节，该包允许我们创建 iSAX 索引。请确保你理解代码，最重要的是知道如何使用代码。

此外，我们还实现了许多命令行工具，使我们能够创建 iSAX 索引，并了解在分割和子序列访问以及溢出条件方面幕后发生了什么。更好地理解 iSAX 索引的结构使我们能够选择更好的索引，并避免使用较差的索引。

下一章将通过展示如何搜索和连接 iSAX 索引来将 iSAX 索引应用于实践。

# 有用链接

+   `argparse`包：[`docs.python.org/3/library/argparse.xhtml`](https://docs.python.org/3/library/argparse.xhtml)

+   NumPy Python 包：[`numpy.org/`](https://numpy.org/)

# 练习

尝试完成以下练习：

+   创建一个包含 100,000 个元素、值从*-10 到 10*的合成数据集，并构建一个具有 4 个段、基数 64 和阈值`1000`的 iSAX 索引。你的机器创建 iSAX 索引花费了多长时间？是否有溢出？

+   创建一个包含 100,000 个元素、值从*-1 到 1*的合成数据集，并构建一个具有 4 个段、基数 64 和阈值`1000`的 iSAX 索引。你的机器创建该 iSAX 索引花费了多长时间？

+   创建一个包含 500,000 个元素、值从*0 到 10*的合成数据集，并构建一个具有 4 个段、基数 64 和阈值`1000`的 iSAX 索引。你的机器创建 iSAX 索引花费了多长时间？

+   创建一个包含 500,000 个元素、值从*0 到 10*的合成数据集，并构建一个具有 4 个段、基数 64 和阈值`1000`的 iSAX 索引。发生了多少次分割和子序列访问？如果你将阈值值增加到`1500`会发生什么？

+   创建一个包含 150,000 个元素、值从*-1 到 1*的合成数据集，并构建一个具有 4 个段、基数 64 和阈值`1000`的 iSAX 索引。是否有溢出？构建 iSAX 索引时执行了多少次分割？

+   在`2M.gz`上使用不同的 iSAX 参数进行`accessSplit.py`实验。哪些参数似乎效果最好？不要忘记，高阈值值对搜索有很大影响；因此，通常不要使用非常大的阈值值以降低分割次数。

+   在`500k.gz`文件上使用各种 iSAX 参数对`accessSplit.py`进行实验。哪些参数看起来效果最好？
