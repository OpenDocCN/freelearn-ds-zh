# 附录 A. 几分钟内构建交互式分析

在本附录中，我们将涵盖以下内容：

+   crossfilter.js 库

+   维度图表 – dc.js

# 简介

恭喜！你已经完成了一本关于数据可视化的 D3 整本书。我们一起探讨了各种主题和技术。此时，你可能会同意，即使有了像 D3 这样强大的库的帮助，构建交互式、准确且美观的数据可视化也不是一件简单的事情。即使不考虑通常在后台所需的工作量，完成一个专业的数据可视化项目通常也需要几天甚至几周的时间。那么，如果你需要快速构建交互式分析，或者在一个完整可视化项目开始之前进行一个概念验证，而你需要在几分钟内完成这项工作，那会怎样呢？在本附录中，我们将向你介绍两个 JavaScript 库，它们允许你在几分钟内完成这些工作：在浏览器中快速构建多维数据交互式分析。

# crossfilter.js 库

Crossfilter 也是一个由 D3 的作者 *Mike Bostock* 创建的库，最初用于为 Square Register 提供分析功能。

> Crossfilter 是一个用于在浏览器中探索大型多变量数据的 JavaScript 库。Crossfilter 支持与协调视图进行极快的（<30ms）交互，即使是在包含百万或更多记录的数据集中也是如此。
> 
> -Crossfilter Wiki (2013年8月)

换句话说，Crossfilter 是一个库，你可以用它在大型的通常平坦的多变量数据集上生成数据维度。那么，什么是数据维度呢？数据维度可以被视为一种数据分组或分类，而每个维度的数据元素是一个分类变量。由于这仍然是一个相当抽象的概念，让我们看一下以下 JSON 数据集，看看它是如何通过 Crossfilter 转换为维度数据集的。假设我们有一个以下扁平的 JSON 数据集，描述了酒吧中的支付交易：

[PRE0]

### 注意

从 Crossfilter Wiki 借用的样本数据集：[https://github.com/square/crossfilter/wiki/API-Reference](https://github.com/square/crossfilter/wiki/API-Reference)。

在这个样本数据集中，我们看到了多少维度？答案是：它有与你可以对数据进行分类的不同方式一样多的维度。例如，由于这些数据是关于客户支付的，这是一种时间序列的观察，显然“日期”是一个维度。其次，支付类型是自然地对数据进行分类的方式；因此，“类型”也是一个维度。下一个维度有点棘手，因为从技术上讲，我们可以将数据集中的任何字段建模为维度或其导数；然而，我们不想将任何不帮助我们更有效地切片数据或提供更多洞察数据试图表达的内容的东西作为维度。总计和小费字段具有非常高的基数，这通常是一个维度较差的指标（尽管小费/总计，即小费百分比可能是一个有趣的维度）；然而，“数量”字段可能具有相对较小的基数，假设人们不会在这个酒吧购买成千上万杯饮料，因此，我们选择使用数量作为我们的第三个维度。现在，这就是维度逻辑模型看起来像什么：

![The crossfilter.js library](img/2162OS_Appendix_01.jpg)

维度数据集

这些维度使我们能够从不同的角度观察数据，如果结合使用，将允许我们提出一些相当有趣的问题，例如：

+   使用账单支付的客户更有可能购买大量商品吗？

+   客户在周五晚上更有可能购买大量商品吗？

+   与使用现金相比，客户使用账单支付时更有可能给小费吗？

现在，你可以看到为什么维度数据集是一个如此强大的想法。本质上，每个维度都为你提供了一个不同的视角来观察你的数据，当它们结合在一起时，它们可以迅速将原始数据转化为知识。一个好的分析师可以快速使用这种工具来制定假设，从而从数据中获得知识。

## 如何做到这一点...

现在，我们理解了为什么我们想要使用我们的数据集建立维度；让我们看看如何使用Crossfilter来实现这一点：

[PRE1]

## 它是如何工作的...

如前所述，在Crossfilter中创建维度和组相当直接。在我们能够创建任何内容之前的第一步是，通过调用`crossfilter`函数将使用D3加载的JSON数据集通过Crossfilter进行传递（行A）。一旦完成，你可以通过调用`dimension`函数并传入一个访问器函数来创建你的维度，该函数将检索用于定义维度的数据元素。对于`type`，我们只需传入`function(d){return d.type;}`。你还可以在维度函数中执行数据格式化或其他任务（例如，行B上的日期格式化）。在创建维度之后，我们可以使用维度进行分类或分组，因此`totalByHour`是对每个小时的销售额进行求和的分组，而`salesByQuantity`是对按数量计数的交易进行分组的分组。为了更好地理解`group`的工作方式，我们将查看组对象的外观。如果你在`transactionsByType`组上调用`all`函数，你将得到以下对象：

![如何工作...](img/2162OS_Appendix_05.jpg)

Crossfilter组对象

我们可以清楚地看到，`transactionByType`组本质上是对数据元素按其类型进行分组，并在每个组内计数数据元素的总数，因为我们创建组时调用了`reduceCount`函数。

以下是我们在这个示例中使用的函数的描述：

+   `crossfilter`：如果指定，创建一个新的带有给定记录的crossfilter。记录可以是任何对象数组或原始数据类型。

+   `dimension`：使用给定的值访问器函数创建一个新的维度。该函数必须返回自然排序的值，即，与JavaScript的<、<=、>=和>运算符正确行为的值。这通常意味着原始数据类型：布尔值、数字或字符串。

+   `dimension.group`：基于给定的`groupValue`函数创建给定维度的新的分组，该函数接受维度值作为输入并返回相应的舍入值。

+   `group.all`：按键的自然顺序返回所有组。

+   `group.reduceCount`：一个用于计数记录的快捷函数；返回此组。

+   `group.reduceSum`：一个用于使用指定的值访问器函数求和记录的快捷函数。

在这个阶段，我们已经拥有了想要分析的所有内容。现在，让我们看看如何能在几分钟内而不是几小时或几天内完成这项工作。

## 还有更多...

我们只接触了Crossfilter函数的一小部分。当涉及到如何创建维度和组时，Crossfilter提供了更多的功能；更多信息请查看其API参考：[https://github.com/square/crossfilter/wiki/API-Reference](https://github.com/square/crossfilter/wiki/API-Reference)。

## 参见

+   数据维度：[http://en.wikipedia.org/wiki/Dimension_(data_warehouse)](http://en.wikipedia.org/wiki/Dimension_(data_warehouse))

+   基数：[http://en.wikipedia.org/wiki/Cardinality](http://en.wikipedia.org/wiki/Cardinality)

# 维度图表 – dc.js

可视化 Crossfilter 维度和组正是 `dc.js` 被创建的原因。这个方便的 JavaScript 库是由你的谦逊作者创建的，旨在让你轻松快速地可视化 Crossfilter 维度数据集。

## 准备工作

打开以下文件的本地副本作为参考：

[https://github.com/NickQiZhu/d3-cookbook/blob/master/src/appendix-a/dc.html](https://github.com/NickQiZhu/d3-cookbook/blob/master/src/appendix-a/dc.html)

## 如何做...

在这个例子中，我们将创建三个图表：

+   用于可视化时间序列上交易总量的折线图

+   用于可视化按支付类型交易数量的饼图

+   展示按购买数量销售数量的条形图

以下是代码的样子：

[PRE2]

这会生成一组协调的交互式图表：

![如何做...](img/2162OS_Appendix_02.jpg)

交互式 dc.js 图表

当你点击或拖动鼠标穿过这些图表时，你将看到所有图表上相应的 Crossfilter 维度被相应地过滤：

![如何做...](img/2162OS_Appendix_03.jpg)

过滤后的 dc.js 图表

## 它是如何工作的...

如我们通过这个例子所看到的，`dc.js` 是设计在 Crossfilter 上生成标准图表可视化工具的。每个 `dc.js` 图表都设计为交互式的，因此用户可以通过与图表交互来简单地应用维度过滤器。`dc.js` 完全基于 D3 构建，因此，它的 API 非常类似于 D3，我相信，通过这本书你获得的知识，你会在使用 `dc.js` 时感到非常熟悉。图表通常按照以下步骤创建。

1.  第一步是通过调用一个图表创建函数并传入其锚点元素的 D3 选择来创建一个图表对象，在我们的例子中是用于托管图表的 `div` 元素：

    [PRE3]

1.  然后我们为每个图表设置 `width`、`height`、`dimension` 和 `group`：

    [PRE4]

    对于在笛卡尔平面上渲染的坐标图表，你还需要设置 `x` 和 `y` 尺度：

    [PRE5]

    在这个第一种情况下，我们明确设置了 x 轴尺度，同时让图表自动为我们计算 y 尺度。而在下一个例子中，我们明确设置了 x 和 y 尺度。

    [PRE6]

## 还有更多...

不同的图表有不同的自定义外观和感觉的功能，你可以在 [https://github.com/NickQiZhu/dc.js/wiki/API](https://github.com/NickQiZhu/dc.js/wiki/API) 查看完整的 API 参考文档。

利用 `crossfilter.js` 和 `dc.js` 可以让你快速构建复杂的数据分析仪表板。以下是对过去 20 年 NASDAQ 100 指数进行分析的演示仪表板 [http://nickqizhu.github.io/dc.js/](http://nickqizhu.github.io/dc.js/)：

![还有更多...](img/2162OS_Appendix_04.jpg)

dc.js NASDAQ 演示

在撰写这本书的时候，`dc.js` 支持以下图表类型：

+   可堆叠条形图

+   可堆叠折线图

+   面积图（可堆叠）

+   饼图

+   气泡图

+   组合图

+   着色地图

+   气泡叠加图

关于`dc.js`库的更多信息，请查看我们的Wiki页面 [https://github.com/NickQiZhu/dc.js/wiki](https://github.com/NickQiZhu/dc.js/wiki)。

## 参考阅读

以下是一些其他有用的基于D3的可重用图表库。尽管，与`dc.js`不同，它们不是原生设计用于与Crossfilter一起工作，但它们在应对一般的可视化挑战时往往更加丰富和灵活：

+   NVD3: [http://nvd3.org/](http://nvd3.org/)

+   手推车: [http://code.shutterstock.com/rickshaw/](http://code.shutterstock.com/rickshaw/)
