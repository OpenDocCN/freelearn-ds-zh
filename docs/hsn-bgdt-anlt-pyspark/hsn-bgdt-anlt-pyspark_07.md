# 第七章：转换和操作

转换和操作是 Apache Spark 程序的主要构建模块。在本章中，我们将看一下 Spark 转换来推迟计算，然后看一下应该避免哪些转换。然后，我们将使用`reduce`和`reduceByKey`方法对数据集进行计算。然后，我们将执行触发实际计算的操作。在本章结束时，我们还将学习如何重用相同的`rdd`进行不同的操作。

在本章中，我们将涵盖以下主题：

+   使用 Spark 转换来推迟计算到以后的时间

+   避免转换

+   使用`reduce`和`reduceByKey`方法来计算结果

+   执行触发实际计算我们的**有向无环图**（**DAG**）的操作

+   重用相同的`rdd`进行不同的操作

# 使用 Spark 转换来推迟计算到以后的时间

让我们首先了解 Spark DAG 的创建。我们将通过发出操作来执行 DAG，并推迟关于启动作业的决定，直到最后一刻来检查这种可能性给我们带来了什么。

让我们看一下我们将在本节中使用的代码。

首先，我们需要初始化 Spark。我们进行的每个测试都是相同的。在开始使用之前，我们需要初始化它，如下例所示：

```py
class DeferComputations extends FunSuite {
val spark: SparkContext = SparkSession.builder().master("local[2]").getOrCreate().sparkContext
```

然后，我们将进行实际测试。在这里，`test`被称为`should defer computation`。它很简单，但展示了 Spark 的一个非常强大的抽象。我们首先创建一个`InputRecord`的`rdd`，如下例所示：

```py
test("should defer computations") {
 //given
    val input = spark.makeRDD(
        List(InputRecord(userId = "A"),
            InputRecord(userId = "B")))
```

`InputRecord`是一个具有可选参数的唯一标识符的案例类。

如果我们没有提供它和必需的参数`userId`，它可以是一个随机的`uuid`。`InputRecord`将在本书中用于测试目的。我们已经创建了两条`InputRecord`的记录，我们将对其应用转换，如下例所示：

```py
//when apply transformation
val rdd = input
    .filter(_.userId.contains("A"))
    .keyBy(_.userId)
.map(_._2.userId.toLowerCase)
//.... built processing graph lazy
```

我们只会过滤`userId`字段中包含`A`的记录。然后我们将其转换为`keyBy(_.userId)`，然后从值中提取`userId`并将其映射为小写。这就是我们的`rdd`。所以，在这里，我们只创建了 DAG，但还没有执行。假设我们有一个复杂的程序，在实际逻辑之前创建了许多这样的无环图。

Spark 的优点是直到发出操作之前不会执行，但我们可以有一些条件逻辑。例如，我们可以得到一个快速路径的执行。假设我们有`shouldExecutePartOfCode()`，它可以检查配置开关，或者去 REST 服务计算`rdd`计算是否仍然相关，如下例所示：

```py
if (shouldExecutePartOfCode()) {
     //rdd.saveAsTextFile("") ||
     rdd.collect().toList
  } else {
    //condition changed - don't need to evaluate DAG
 }
}
```

我们已经使用了简单的方法进行测试，我们只是返回`true`，但在现实生活中，这可能是复杂的逻辑：

```py
private def shouldExecutePartOfCode(): Boolean = {
    //domain logic that decide if we still need to calculate
    true
    }
}
```

在它返回`true`之后，我们可以决定是否要执行 DAG。如果要执行，我们可以调用`rdd.collect().toList`或`saveAsTextFile`来执行`rdd`。否则，我们可以有一个快速路径，并决定我们不再对输入的`rdd`感兴趣。通过这样做，只会创建图。

当我们开始测试时，它将花费一些时间来完成，并返回以下输出：

```py
"C:\Program Files\Java\jdk-12\bin\java.exe" "-javaagent:C:\Program Files\JetBrains\IntelliJ IDEA 2018.3.5\lib\idea_rt.jar=50627:C:\Program Files\JetBrains\IntelliJ IDEA 2018.3.5\bin" -Dfile.encoding=UTF-8 -classpath C:\Users\Sneha\IdeaProjects\Chapter07\out\production\Chapter07 com.company.Main

Process finished with exit code 0
```

我们可以看到我们的测试通过了，我们可以得出它按预期工作的结论。现在，让我们看一些应该避免的转换。

# 避免转换

在本节中，我们将看一下应该避免的转换。在这里，我们将专注于一个特定的转换。

我们将从理解`groupBy`API 开始。然后，我们将研究在使用`groupBy`时的数据分区，然后我们将看一下什么是 skew 分区以及为什么应该避免 skew 分区。

在这里，我们正在创建一个交易列表。`UserTransaction`是另一个模型类，包括`userId`和`amount`。以下代码块显示了一个典型的交易，我们正在创建一个包含五个交易的列表：

```py
test("should trigger computations using actions") {
 //given
 val input = spark.makeRDD(
     List(
         UserTransaction(userId = "A", amount = 1001),
         UserTransaction(userId = "A", amount = 100),
         UserTransaction(userId = "A", amount = 102),
         UserTransaction(userId = "A", amount = 1),
         UserTransaction(userId = "B", amount = 13)))
```

我们已经为`userId = "A"`创建了四笔交易，为`userId = "B"`创建了一笔交易。

现在，让我们考虑我们想要合并特定`userId`的交易以获得交易列表。我们有一个`input`，我们正在按`userId`分组，如下例所示：

```py
//when apply transformation
val rdd = input
    .groupBy(_.userId)
    .map(x => (x._1,x._2.toList))
    .collect()
    .toList
```

对于每个`x`元素，我们将创建一个元组。元组的第一个元素是一个 ID，而第二个元素是该特定 ID 的每个交易的迭代器。我们将使用`toList`将其转换为列表。然后，我们将收集所有内容并将其分配给`toList`以获得我们的结果。让我们断言结果。`rdd`应该包含与`B`相同的元素，即键和一个交易，以及`A`，其中有四个交易，如下面的代码所示：

```py
//then
rdd should contain theSameElementsAs List(
    ("B", List(UserTransaction("B", 13))),
    ("A", List(
        UserTransaction("A", 1001),
        UserTransaction("A", 100),
        UserTransaction("A", 102),
        UserTransaction("A", 1))
    )
  )
 }
}
```

让我们开始这个测试，并检查它是否按预期行为。我们得到以下输出：

```py
"C:\Program Files\Java\jdk-12\bin\java.exe" "-javaagent:C:\Program Files\JetBrains\IntelliJ IDEA 2018.3.5\lib\idea_rt.jar=50822:C:\Program Files\JetBrains\IntelliJ IDEA 2018.3.5\bin" -Dfile.encoding=UTF-8 -classpath C:\Users\Sneha\IdeaProjects\Chapter07\out\production\Chapter07 com.company.Main

Process finished with exit code 0
```

乍一看，它已经通过了，并且按预期工作。但是，为什么我们要对它进行分组的问题就出现了。我们想要对它进行分组以将其保存到文件系统或进行一些进一步的操作，例如连接所有金额。

我们可以看到我们的输入不是正常分布的，因为几乎所有的交易都是针对`userId = "A"`。因此，我们有一个偏斜的键。这意味着一个键包含大部分数据，而其他键包含较少的数据。当我们在 Spark 中使用`groupBy`时，它会获取所有具有相同分组的元素，例如在这个例子中是`userId`，并将这些值发送到完全相同的执行者。

例如，如果我们的执行者有 5GB 的内存，我们有一个非常大的数据集，有数百 GB，其中一个键有 90%的数据，这意味着所有数据都将传输到一个执行者，其余的执行者将获取少数数据。因此，数据将不会正常分布，并且由于非均匀分布，处理效率将不会尽可能高。

因此，当我们使用`groupBy`键时，我们必须首先回答为什么要对其进行分组的问题。也许我们可以在`groupBy`之前对其进行过滤或聚合，然后我们只会对结果进行分组，或者根本不进行分组。我们将在以下部分中研究如何使用 Spark API 解决这个问题。

# 使用 reduce 和 reduceByKey 方法来计算结果

在本节中，我们将使用`reduce`和`reduceBykey`函数来计算我们的结果，并了解`reduce`的行为。然后，我们将比较`reduce`和`reduceBykey`函数，以确定在特定用例中应该使用哪个函数。

我们将首先关注`reduce`API。首先，我们需要创建一个`UserTransaction`的输入。我们有用户交易`A`，金额为`10`，`B`的金额为`1`，`A`的金额为`101`。假设我们想找出全局最大值。我们对特定键的数据不感兴趣，而是对全局数据感兴趣。我们想要扫描它，取最大值，并返回它，如下例所示：

```py
test("should use reduce API") {
    //given
    val input = spark.makeRDD(List(
    UserTransaction("A", 10),
    UserTransaction("B", 1),
    UserTransaction("A", 101)
    ))
```

因此，这是减少使用情况。现在，让我们看看如何实现它，如下例所示：

```py
//when
val result = input
    .map(_.amount)
    .reduce((a, b) => if (a > b) a else b)

//then
assert(result == 101)
}
```

对于`input`，我们需要首先映射我们感兴趣的字段。在这种情况下，我们对`amount`感兴趣。我们将取`amount`，然后取最大值。

在前面的代码示例中，`reduce`有两个参数，`a`和`b`。一个参数将是我们正在传递的特定 Lambda 中的当前最大值，而第二个参数将是我们现在正在调查的实际值。如果该值高于到目前为止的最大状态，我们将返回`a`；如果不是，它将返回`b`。我们将遍历所有元素，最终结果将只是一个长数字。

因此，让我们测试一下，检查结果是否确实是`101`，如以下代码输出所示。这意味着我们的测试通过了。

```py
"C:\Program Files\Java\jdk-12\bin\java.exe" "-javaagent:C:\Program Files\JetBrains\IntelliJ IDEA 2018.3.5\lib\idea_rt.jar=50894:C:\Program Files\JetBrains\IntelliJ IDEA 2018.3.5\bin" -Dfile.encoding=UTF-8 -classpath C:\Users\Sneha\IdeaProjects\Chapter07\out\production\Chapter07 com.company.Main

Process finished with exit code 0
```

现在，让我们考虑一个不同的情况。我们想找到最大的交易金额，但这次我们想根据用户来做。我们不仅想找出用户`A`的最大交易，还想找出用户`B`的最大交易，但我们希望这些事情是独立的。因此，对于相同的每个键，我们只想从我们的数据中取出最大值，如以下示例所示：

```py
test("should use reduceByKey API") {
    //given
    val input = spark.makeRDD(
    List(
        UserTransaction("A", 10),
        UserTransaction("B", 1),
        UserTransaction("A", 101)
    )
)
```

要实现这一点，`reduce`不是一个好选择，因为它将遍历所有的值并给出全局最大值。我们在 Spark 中有关键操作，但首先，我们要为特定的元素组做这件事。我们需要使用`keyBy`告诉 Spark 应该将哪个 ID 作为唯一的，并且它将仅在特定的键内执行`reduce`函数。因此，我们使用`keyBy(_.userId)`，然后得到`reducedByKey`函数。`reduceByKey`函数类似于`reduce`，但它按键工作，因此在 Lambda 内，我们只会得到特定键的值，如以下示例所示：

```py
    //when
    val result = input
      .keyBy(_.userId)
      .reduceByKey((firstTransaction, secondTransaction) =>
        TransactionChecker.higherTransactionAmount(firstTransaction, secondTransaction))
      .collect()
      .toList
```

通过这样做，我们得到第一笔交易，然后是第二笔。第一笔将是当前的最大值，第二笔将是我们正在调查的交易。我们将创建一个辅助函数，它接受这些交易并称之为`higherTransactionAmount`。

`higherTransactionAmount`函数用于获取`firstTransaction`和`secondTransaction`。请注意，对于`UserTransaction`类型，我们需要传递该类型。它还需要返回`UserTransaction`，我们不能返回不同的类型。

如果您正在使用 Spark 的`reduceByKey`方法，我们需要返回与`input`参数相同的类型。如果`firstTransaction.amount`高于`secondTransaction.amount`，我们将返回`firstTransaction`，因为我们返回的是`secondTransaction`，所以是交易对象而不是总金额。这在以下示例中显示：

```py
object TransactionChecker {
    def higherTransactionAmount(firstTransaction: UserTransaction, secondTransaction: UserTransaction): UserTransaction = {
        if (firstTransaction.amount > secondTransaction.amount) firstTransaction else     secondTransaction
    }
}
```

现在，我们将收集、添加和测试交易。在我们的测试之后，我们得到了输出，对于键`B`，我们应该得到交易`("B", 1)`，对于键`A`，交易`("A", 101)`。没有交易`("A", 10)`，因为我们已经过滤掉了它，但我们可以看到对于每个键，我们都能找到最大值。这在以下示例中显示：

```py
    //then
    result should contain theSameElementsAs
      List(("B", UserTransaction("B", 1)), ("A", UserTransaction("A", 101)))
  }

}
```

我们可以看到测试通过了，一切都如预期的那样，如以下输出所示：

```py
"C:\Program Files\Java\jdk-12\bin\java.exe" "-javaagent:C:\Program Files\JetBrains\IntelliJ IDEA 2018.3.5\lib\idea_rt.jar=50909:C:\Program Files\JetBrains\IntelliJ IDEA 2018.3.5\bin" -Dfile.encoding=UTF-8 -classpath C:\Users\Sneha\IdeaProjects\Chapter07\out\production\Chapter07 com.company.Main

Process finished with exit code 0
```

在下一节中，我们将执行触发数据计算的操作。

# 执行触发计算的操作

Spark 有更多触发 DAG 的操作，我们应该了解所有这些，因为它们非常重要。在本节中，我们将了解 Spark 中可以成为操作的内容，对操作进行一次遍历，并测试这些操作是否符合预期。

我们已经涵盖的第一个操作是`collect`。除此之外，我们还涵盖了两个操作——在上一节中我们都涵盖了`reduce`和`reduceByKey`。这两种方法都是操作，因为它们返回单个结果。

首先，我们将创建我们的交易的`input`，然后应用一些转换，仅用于测试目的。我们将只取包含`A`的用户，使用`keyBy_.userId`，然后只取所需交易的金额，如以下示例所示：

```py
test("should trigger computations using actions") {
     //given
     val input = spark.makeRDD(
     List(
         UserTransaction(userId = "A", amount = 1001),
         UserTransaction(userId = "A", amount = 100),
         UserTransaction(userId = "A", amount = 102),
         UserTransaction(userId = "A", amount = 1),
         UserTransaction(userId = "B", amount = 13)))

//when apply transformation
 val rdd = input
     .filter(_.userId.contains("A"))
     .keyBy(_.userId)
     .map(_._2.amount)
```

我们已经知道的第一个操作是`rdd.collect().toList`。接下来是`count()`，它需要获取所有的值并计算`rdd`中有多少值。没有办法在不触发转换的情况下执行`count()`。此外，Spark 中还有不同的方法，如`countApprox`、`countApproxDistinct`、`countByValue`和`countByValueApprox`。以下示例显示了`rdd.collect().toList`的代码：

```py
//then
 println(rdd.collect().toList)
 println(rdd.count()) //and all count*
```

如果我们有一个庞大的数据集，并且近似计数就足够了，你可以使用`countApprox`，因为它会快得多。然后我们使用`rdd.first()`，但这个选项有点不同，因为它只需要取第一个元素。有时，如果你想取第一个元素并执行我们 DAG 中的所有操作，我们需要专注于这一点，并以以下方式检查它：

```py
println(rdd.first())
```

此外，在`rdd`上，我们有`foreach()`，这是一个循环，我们可以传递任何函数。假定 Scala 函数或 Java 函数是 Lambda，但要执行我们结果`rdd`的元素，需要计算 DAG，因为从这里开始，它就是一个操作。`foreach()`方法的另一个变体是`foreachPartition()`，它获取每个分区并返回分区的迭代器。在其中，我们有一个迭代器再次进行迭代并打印我们的元素。我们还有我们的`max()`和`min()`方法，预期的是，`max()`取最大值，`min()`取最小值。但这些方法都需要隐式排序。

如果我们有一个简单的原始类型的`rdd`，比如`Long`，我们不需要在这里传递它。但如果我们不使用`map()`，我们需要为 Spark 定义`UserTransaction`的排序，以便找出哪个元素是`max`，哪个元素是`min`。这两件事需要执行 DAG，因此它们被视为操作，如下面的例子所示：

```py
 rdd.foreach(println(_))
 rdd.foreachPartition(t => t.foreach(println(_)))
 println(rdd.max())
 println(rdd.min())
```

然后我们有`takeOrdered()`，这是一个比`first()`更耗时的操作，因为`first()`取一个随机元素。`takeOrdered()`需要执行 DAG 并对所有内容进行排序。当一切都排序好后，它才取出顶部的元素。

在我们的例子中，我们取`num = 1`。但有时，出于测试或监控的目的，我们需要只取数据的样本。为了取样，我们使用`takeSample()`方法并传递一个元素数量，如下面的代码所示：

```py
 println(rdd.takeOrdered(1).toList)
 println(rdd.takeSample(false, 2).toList)
 }
}
```

现在，让我们开始测试并查看实现前面操作的输出，如下面的屏幕截图所示：

```py
List(1001, 100, 102 ,1)
4
1001
1001
100
102
1
```

第一个操作返回所有值。第二个操作返回`4`作为计数。我们将考虑第一个元素`1001`，但这是一个随机值，它是无序的。然后我们在循环中打印所有的元素，如下面的输出所示：

```py
102
1
1001
1
List(1)
List(100, 1)
```

然后我们得到`max`和`min`值，如`1001`和`1`，这与`first()`类似。之后，我们得到一个有序列表`List(1)`，和一个样本`List(100, 1)`，这是随机的。因此，在样本中，我们从输入数据和应用的转换中得到随机值。

在下一节中，我们将学习如何重用`rdd`进行不同的操作。

# 重用相同的 rdd 进行不同的操作

在这一部分，我们将重用相同的`rdd`进行不同的操作。首先，我们将通过重用`rdd`来最小化执行时间。然后，我们将查看缓存和我们代码的性能测试。

下面的例子是前面部分的测试，但稍作修改，这里我们通过`currentTimeMillis()`取`start`和`result`。因此，我们只是测量执行的所有操作的`result`：

```py
//then every call to action means that we are going up to the RDD chain
//if we are loading data from external file-system (I.E.: HDFS), every action means
//that we need to load it from FS.
    val start = System.currentTimeMillis()
    println(rdd.collect().toList)
    println(rdd.count())
    println(rdd.first())
    rdd.foreach(println(_))
    rdd.foreachPartition(t => t.foreach(println(_)))
    println(rdd.max())
    println(rdd.min())
    println(rdd.takeOrdered(1).toList)
    println(rdd.takeSample(false, 2).toList)
    val result = System.currentTimeMillis() - start

    println(s"time taken (no-cache): $result")

}
```

如果有人对 Spark 不太了解，他们会认为所有操作都被巧妙地执行了。我们知道每个操作都意味着我们要上升到链中的`rdd`，这意味着我们要对所有的转换进行加载数据。在生产系统中，加载数据将来自外部的 PI 系统，比如 HDFS。这意味着每个操作都会导致对文件系统的调用，这将检索所有数据，然后应用转换，如下例所示：

```py
//when apply transformation
val rdd = input
    .filter(_.userId.contains("A"))
    .keyBy(_.userId)
    .map(_._2.amount)
```

这是一个非常昂贵的操作，因为每个操作都非常昂贵。当我们开始这个测试时，我们可以看到没有缓存的时间为 632 毫秒，如下面的输出所示：

```py
List(1)
List(100, 1)
time taken (no-cache): 632
Process finished with exit code 0
```

让我们将这与缓存使用进行比较。乍一看，我们的测试看起来非常相似，但这并不相同，因为您正在使用`cache()`，而我们正在返回`rdd`。因此，`rdd`将已经被缓存，对`rdd`的每个后续调用都将经过`cache`，如下例所示：

```py
//when apply transformation
val rdd = input
    .filter(_.userId.contains("A"))
    .keyBy(_.userId)
    .map(_._2.amount)
    .cache()
```

第一个操作将执行 DAG，将数据保存到我们的缓存中，然后后续的操作将根据从内存中调用的方法来检索特定的内容。不会有 HDFS 查找，所以让我们按照以下示例开始这个测试，看看需要多长时间：

```py
//then every call to action means that we are going up to the RDD chain
//if we are loading data from external file-system (I.E.: HDFS), every action means
//that we need to load it from FS.
    val start = System.currentTimeMillis()
    println(rdd.collect().toList)
    println(rdd.count())
    println(rdd.first())
    rdd.foreach(println(_))
    rdd.foreachPartition(t => t.foreach(println(_)))
    println(rdd.max())
    println(rdd.min())
    println(rdd.takeOrdered(1).toList)
    println(rdd.takeSample(false, 2).toList)
    val result = System.currentTimeMillis() - start

    println(s"time taken(cache): $result")

    }
}
```

第一个输出将如下所示：

```py
List(1)
List(100, 102)
time taken (no-cache): 585
List(1001, 100, 102, 1)
4
```

第二个输出将如下所示：

```py
1
List(1)
List(102, 1)
time taken(cache): 336
Process finished with exit code 0
```

没有缓存，值为`585`毫秒，有缓存时，值为`336`。这个差异并不大，因为我们只是在测试中创建数据。然而，在真实的生产系统中，这将是一个很大的差异，因为我们需要从外部文件系统中查找数据。

# 总结

因此，让我们总结一下这一章节。首先，我们使用 Spark 转换来推迟计算到以后的时间，然后我们学习了哪些转换应该避免。接下来，我们看了如何使用`reduceByKey`和`reduce`来计算我们的全局结果和特定键的结果。之后，我们执行了触发计算的操作，然后了解到每个操作都意味着加载数据的调用。为了缓解这个问题，我们学习了如何为不同的操作减少相同的`rdd`。

在下一章中，我们将看一下 Spark 引擎的不可变设计。
