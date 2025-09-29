# 第八章。样式化你的地图

这是让开发者感到恐惧的一个词：样式。他们认为这是一个以右脑为主的艺术家领域。他们将色彩理论、排版和空白视为外语。他们渴望将这些任务推给网页设计师，并专注于算法和 API 调用。

许多小型公司、政府机构和部门没有在他们的员工中配备网页设计师的奢侈，甚至没有可随时调用的人。这些小型组织往往由一或多个因技术分析和技能而被雇佣的人组成，而设计技能则被当作次要考虑。也许你现在就在为这样的组织工作。

虽然这一章可能无法让你立刻成为一名网页设计师，但它将帮助你有效地使用 CSS 来布局你的网络地图应用程序。

在本章中，我们将涵盖以下主题：

+   CSS 如何应用于 HTML 文档

+   在页面上定位地图的不同方式

+   如何使用 Dojo 的`dijit/layout`模块来样式化你的页面

+   如何将 Bootstrap 添加到页面的布局中

# CSS 的内部工作原理

正如我们在第一章中提到的，*你的第一个地图应用程序*，**层叠样式表**（**CSS**）告诉浏览器如何渲染 HTML 页面。当浏览器扫描 HTML 时，它会扫描所有适用的 CSS 样式，包括 CSS 文件中的样式以及 HTML 中的任何覆盖样式，以确定如何渲染元素。CSS 描述，如颜色和字体大小，通常从父元素级联到其子元素，除非有特定的覆盖。例如，应用于`div`标签的样式也将应用于其内部的`p`标签，如下面的代码所示：

```py
<div>
  <p>I'm not red.</p>
 <div style="color:red;">
 <p>You caught me red-handed.</p>
 <p>Me too.</p>
 </div>
  <p>I'm not red</p>
</div>
```

CSS 与大多数编程语言的工作方式略有不同。在 JavaScript 中，当你程序的一部分有错误时，你可以在浏览器中逐步执行代码，直到找到出错的部分。然而，在 CSS 中，定义元素外观的规则可以跨越多个样式表，甚至可以写在元素内部。元素的外观也可能受到内部和外部元素的影响。

## 选择器特定性

浏览器通过比较用于定义样式的选择器类型来决定元素应该如何被样式化。它根据选择器的类型和数量应用权重。浏览器有五个基本的 CSS 选择器等级，根据它们的特定性来划分。技术上，当你使用`*`选择页面上的所有元素时，有一个零级选择器，但与其他选择器相比，它没有价值。选择器等级如下：

+   通过元素（例如`h1`、`div`或`p`）

+   通过元素类、属性和伪选择器。一些例子包括：

    +   `.email`或`.form`（类）

    +   `input[type='text']`（属性）

    +   `a:hover`或`p:first-child`（伪选择器）

+   通过 ID（例如`#main`或`#map-div`）

+   通过内联样式（`<p style=""></p>`）

+   通过标记`!important`的样式

记录特定性的一个常见方法是对每个类别的选择器数量进行计数，并用逗号分隔它们。一个`p`选择器得到特定性`1`，而`p > a`得到特定性`2`。然而，`p.special > a`得到特定性`1,2`，因为类属于一个单独的、更高等级的分类。一个`#main`选择器的特定性为`1,0,0`，而`p`标签的内联样式得到特定性`1,0,0,0`。强大的`!important`子句是唯一可以覆盖内联选择器的因素，它获得特定性`1,0,0,0,0`。

当比较选择器特定性时，一个高等级选择器胜过任何数量低等级选择器。在出现平局的情况下，比较下一个最低等级的选择器。例如，让我们看看以下 HTML 片段：

```py
<style>
  .addr { background: red; }
  .start.blank { background: orange; }
  .help.blank.addr { background: yellow; }
  #citytxt { background: green; }
  input[type='text'] { background: blue; }
  input { background: purple; }
</style>
…
<input type="text" id="citytxt" class="addr start blank help" />
```

你认为背景颜色会是什么？正确答案是绿色。`#citytxt`规则是一个第三等级的选择器，因为它指向页面上的单个元素。如果我们查看选择器及其特定性等级，它们看起来如下：

```py
<style>
  .addr { background: red; }               /* 0,1,0 */
  .start.blank { background: orange; }     /* 0,2,0 */
  .help.blank.addr { background: yellow; } /* 0,3,0 */
  #citytxt { background: green; }          /* 1,0,0 */
  input[type='text'] { background: blue; } /* 0,1,1 */
  input { background: purple; }            /* 0,0,1 */
</style>
```

那么，当其他一切都相等时会发生什么？

## 等级选择器特定性

当两个或更多规则具有相同的选择器特定性时，最后列出的规则获胜。这是 CSS 级联效果的另一个特性。我们总是在我们的应用程序中将自定义 CSS 放在 Dojo 和 ArcGIS JavaScript API 样式表之后。我们做出的任何样式更改都不会被其他样式表覆盖。

使用“最后者胜出”规则，我们可以撤销应用于我们的小部件的 CSS 规则可能产生的任何意外副作用。我们不必总是求助于使用`!important`标签或编写丢失在代码审查中的内联样式。只要我们将它放在旧规则之后，我们就可以使用相同的选择器特定性强度，并得到我们想要的结果。

# 样式技巧和窍门

现在我们已经了解了 CSS 的工作原理，我们可以在此基础上构建一些适用于我们应用程序的工作样式：

+   我们将首先研究一些我们需要避免的不良模式。我们将看看它们如何对样式和进一步的开发产生负面影响。

+   接下来，我们将探讨一些良好的实践，例如使用响应式设计和标准化样式表。

+   我们将探讨如何组织你的样式表，使它们更容易扩展和调试。

+   我们将介绍如何将地图定位在应用程序需要的任何位置。

## 样式禁忌

在我告诉你该做什么之前，让我们先了解一下你应该避免的一些事情。这些不良的设计习惯通常是在完成初学者教程和从互联网上复制粘贴单页应用程序时形成的：

+   内联样式化元素：试图逐个更改 20 个或更多段落的样式是一件痛苦的事情。

+   使一切变得重要：`important`子句允许你覆盖其他小部件和导入的样式表施加的样式，但不要过分沉迷。

+   对单个 ID 的引用过多：一些元素 ID 引用是可以接受的，但如果你想在其他页面或项目中重用你的 CSS 文件，你希望它们尽可能通用。并不是每个人都会使用你的 `#pink_and_purple_striped_address2_input` 元素。

+   在页面底部添加新的 CSS 文件更改：我们都知道“后到先得”，但如果每次都将新的更新直接添加到页面底部，文件就会变成一个杂乱无章的 CSS 规则抽屉。

就像任何硬性规则一样，有时打破它们是合适的。但，在这些规则的范围内工作，会使你自己和他人更容易维护你的应用程序。

## 响应式设计

**响应式设计**运动在网站开发中占据了稳固的地位。响应式设计围绕的理念是，网站应该在各种屏幕尺寸上可用，从大屏幕到手机屏幕。这减少了为桌面和移动用户维护多个网站的成本。

表面上看，响应式设计涉及分配百分比宽度和高度而不是固定大小，但还有更多。流体网格布局在较宽的屏幕上支持多列，而在较窄的屏幕上则折叠成单列。可以为平板电脑和具有视网膜显示的屏幕提供不同分辨率的图像，以获得更清晰的图像。CSS 媒体查询可以根据不同大小或不同媒体更改元素的显示方式。

使用 ArcGIS JavaScript API 创建的地图与响应式设计布局配合良好。地图会跟踪其 HTML 容器元素的尺寸和尺寸变化，同时更新其内容。当地图比例保持不变时，范围会重新计算，并为之前未存储在内存中的位置请求新的地图瓦片。这些更改可以通过在桌面浏览器中调整浏览器窗口大小或是在移动浏览器中将平板电脑侧转来触发。

## Normalize.css

在浏览器中使你的应用程序看起来良好可能会令人沮丧。引入更多浏览器会加剧这个问题。许多浏览器在不同的设备上都有独特的渲染引擎，这使得相同的 HTML 元素在每个设备上看起来都不同。难道他们不能就 HTML 元素的样式达成一致吗？

开发者和设计师经常使用一个名为 `normalize.css` 的 CSS 文件（[`necolas.github.io/normalize.css/`](http://necolas.github.io/normalize.css/)）。这个样式表为 HTML 元素设置样式，使它们在不同浏览器和设备上看起来相似。当你关心页面外观时，这减少了猜测的工作量。

`normalize.css` 文件样式通常作为 HTML 文档头部的第一个样式表插入。你对网站样式所做的任何更改都将在此应用 normalize 规则之后进行，并且不太可能被覆盖。一些 CSS 框架，如 Twitter Bootstrap，已经在其样式表中集成了`normalize.css`。

## 组织你的 CSS

如前所述，在“不要用你的样式做这些事情”的列表中，最大的违规行为包括使用比所需更高的选择器特定性，以及将样式表当作一个杂物抽屉。通过正确组织你的样式表，你可以减少这两种违规行为，并使你的应用程序更容易进行样式化和维护。让我们来探讨一些最佳实践。

### 按选择器特定性组织

当前网页设计趋势要求 CSS 选择器从最低的选择器特定性组织到最高的。你所有的`div`、`h1`和`p`标签可能都放在样式表的最顶部。在定义 HTML 元素的外观之后，你可以添加各种类、选择器和伪选择器来描述它们如何改变元素的外观。最后，你可以通过引用其`id`来分配单个元素的外观。

### 按模块分组

使用 ArcGIS JavaScript API 编写的应用程序可以很容易地按小部件组织，那么为什么不按小部件组织样式表呢？你可以在定义页面样式之后，在应用程序中定义单个小部件的样式。你可以使用`/* comments */`在模块和小部件样式之间分隔你的 CSS 为逻辑部分。

### 一切皆可分类

在组织选择器特定性代码时，一个常见的做法是尽可能多地分配 CSS 类。你的地图小部件可能有一个`.open`类来设置`width`和`height`。同一个小部件可能有一个`.closed`类来隐藏小部件。使用`dojo/dom-class`模块，你可以按任何方式添加、删除和切换你定义的类：

```py
.open {
  width: 30%;
  height: 100%;
  display: block;
}
.closed {
  width: 0%;
  display: none;
}
```

使用描述性类可以使你更容易看到你的应用程序正在做什么，尤其是在查看页面源代码时。描述性类在你的样式表中更容易引用。

### 媒体查询

媒体查询提供了在不同屏幕上创建自定义外观和响应式网格的有效方法。你可以根据媒体类型（屏幕、打印、投影仪等）、屏幕宽度和甚至像素深度（视网膜显示屏与标准桌面屏幕）来改变你网站的外观。

在组织代码时，有一件事需要考虑的是，媒体查询应该放在正常选择器之后。这样，你可以利用“后入先得”的原则，当屏幕尺寸变化时，使用相同的选择器显示不同的结果。我曾经犯过没有注意我的媒体查询放置位置的错误，浪费了时间排查为什么我的过渡效果没有发生。后来我才发现在 CSS 的混乱中，我在媒体查询之后应用了一条规则，这取消了效果。

## 定位你的地图

我们不能总是依赖其他框架和样式表来正确定位我们的地图。有时候，我们必须自己动手用 CSS 来做。我们将讨论一些地图的样式场景，并查看我们需要应用到地图元素上的 CSS 规则以正确定位它。所有示例都假设你正在创建一个 ID 为"map"的`div`元素上的地图。

## 固定宽度的地图

默认情况下，地图会创建一个特定的宽度和高度。宽度和高度可以是任何非负数，从全屏到狭窄的列。如果没有指定高度，地图元素将分配一个默认的`height`为`400px`。你可以在以下位置看到地图的简单、非响应式 CSS 样式：

```py
#map {
  width: 600px;
  height: 400px;
}
```

## 将地图拉伸到全屏

有时候，你的地图不仅仅是页面上的一个重要部分。有时候，地图需要占据整个页面。这就是全屏尺寸所代表的含义。假设 HTML 和 body 标签的宽度和高度也是`100%`，这种样式是可行的。这种全屏样式也可以分配给应该填充另一个元素`100%`区域的地图：

```py
#map {
  width: 100%;
  height: 100%;
}
```

## 将地图浮动到侧面

有时候你不需要全图。有时候你只需要在屏幕的一侧显示一个小地图，显示与它共享页面的内容的地理位置。然后你可以将内容浮动到一边。将元素浮动到右边或左边可以让其他内容填充在其周围。这种技术通常用于照片和文本，其中文本围绕照片流动。这也适用于地图：

```py
#map {
  width: 30%;
  height: 30%;
  float: right;
}
```

## 定位地图顶部和居中

有时候你需要将地图在你的布局中居中。你有一些文本，你只想让地图在中间整齐排列。使用这个居中技巧，你可以水平居中页面上的任何块级元素。在这里，你设置位置为相对，并分配一个自动的左右边距。浏览器会自动将相同的数值分配给左右边距，从而实现居中。但请记住，这必须在具有相对定位的块级元素（如地图）上执行。移除这些条件中的任何一个，这个技巧就不起作用，如下面的代码所示：

```py
#map {
/* just in case something sets the display to something else */
  display: block; 
  position: relative;
  margin: 0 auto;
}
```

## 使用地图覆盖页面的大部分区域

如果你需要一个几乎满页的效果，需要为标题栏或左侧或右侧的列留出空间，你可以使用绝对定位来拉伸地图。绝对定位将元素移出正常布局，并允许你将其定位在任何你想要的位置。

一旦你为地图分配了绝对定位，你可以使用 top、bottom、left 和 right 值来拉伸地图。将 `bottom` 的值设置为 `0`，告诉浏览器将元素的底部边缘设置为页面底部。将 `top` 的值设置为 `40px`，告诉浏览器将地图元素的顶部设置为页面顶部下方 40 像素处。通过分配左右两个值，你将地图元素在两个方向上拉伸。

作为注意事项，请记住，绝对定位的元素会超出其位置范围，将位于整个页面内，或者位于具有相对定位的第一个父元素内：

```py
#map {
  position: absolute;
  top: 40px; /* for a 40px tall titlebar */
  bottom: 0;
  left: 0; 
  right: 0;
}
```

## 居中已知宽度和高度的地图

有时，你需要将地图放在页面中心。你想要创建一个模态效果，地图在垂直和水平方向上居中，有点像模态对话框。如果你知道地图的宽度和高度，你可以轻松实现这一点。你将绝对定位分配给地图，并将地图的 `top` 和 `left` 边缘设置为页面宽度的 `50%`。一开始这看起来可能不太对，直到你分配了边距。技巧是分配负的 top 和 left 边距，其值分别为地图元素高度和宽度的一半。你得到的是一个垂直和水平居中的地图，同时在旧浏览器中也有效：

```py
#map {
  width: 640px;
  height: 480px;
  position: absolute;
  top: 50%;
  left: 50%;
  margin: -240px 0 0 -320px;
}
```

## 居中未知宽度和高度的地图

如果你将百分比或其他单位应用于地图的样式，你可能不知道在任何时候地图有多宽。我们可以使用绝对定位将元素的左上角放置在页面中间，但如何将元素移动，使其位于页面中间？当宽度和高度可变时，有一个简单的方法来垂直和水平居中地图。这需要 CSS3 变换。

我们可以通过使用 CSS3 变换来在任何方向上翻译或移动元素。第一个值将其移动到右侧或左侧，而第二个值将其移动到上方或下方。负值表示向左和向上移动。我们可以用像素为单位应用宽度和高度，或者我们可以应用元素宽度的百分比来居中元素：

```py
#map {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 60%;  /* any width is okay */
  height: 60%; /* any height is okay */
  -webkit-transform: translate(-50%, -50%);
 -moz-transform: translate(-50%, -50%);
 -ms-transform: translate(-50%, -50%);
 -o-transform: translate(-50%, -50%);
 transform: translate(-50%, -50%); 
 /* transform shifts it halfway back*/
}
```

### 小贴士

CSS3 变换在大多数现代浏览器中可用，一些稍微旧一点的浏览器需要供应商前缀。Internet Explorer 8 及更早版本不支持这些变换。有关更多详细信息，请参阅 [`caniuse.com/#feat=transforms2d`](http://caniuse.com/#feat=transforms2d)。

# 故障排除

随着浏览器端 Web 开发工具的日益流行，您的浏览器是排查页面样式问题的最佳工具。Mozilla Firefox 最初拥有最先进的检查工具，这些工具是通过一个名为**Firebug**的免费第三方插件实现的。后来，Chrome 发布了自己的开发工具，而 Firefox 和 Internet Explorer 最终也构建并改进了自己的工具。现在所有现代桌面浏览器都为桌面和移动设备提供了高级 JavaScript 和 CSS 信息。

大多数浏览器开发者工具都可以使用相同的键盘快捷键打开。Internet Explorer 从版本 8 开始就响应*F12*键。Chrome 和 Firefox 也响应*F12*，键盘组合为*Ctrl* + *Shift* + *I*（Mac 上的*Cmd* + *Opt* + *I*）。

## 响应式调整大小

所有桌面浏览器在最大化或缩小它们时都会缩小和增长。然而，许多现代浏览器都有额外的功能和插件，可以帮助您测试应用程序，就像它们是移动浏览器一样。Firefox 开发者版的最新版本有一个**响应式设计视图**，可以根据移动设备调整浏览器的大小。当用户旋转手机时，它可以旋转屏幕，甚至触发触摸事件。Google Chrome 有一个**设备模式**，允许您从流行的智能手机和平板电脑中选择，可以模拟较慢的网络速度，并通过更改请求中发送的用户代理来模拟移动浏览器。最新版本的 Internet Explorer 在其开发者工具中也提供了这些功能。

现在我们已经回顾了可以用来测试布局的工具，让我们看看 ArcGIS JavaScript API 为我们提供布局应用程序的工具。

# Dojo 布局

Dojo 使用自己的框架来控制应用程序的布局。Dojo 的布局可以在`dijit/layout`模块中找到。这些模块可以通过使用 Dojo 框架来实现具有所有功能的完整页面应用程序。

使用`dijit/layout`模块创建的 Dijits 可以直接编码在 HTML 中。这些编码使用`data-dojo-type`属性。包括样式和行为在内的属性编码在元素的`data-dojo-props`属性中。这些 Dijits 可以通过使用`dojo/parser`模块解析 HTML 来从 HTML 中加载。

### 小贴士

在通过 HTML 加载`dijit/layout`元素的应用中，如果`dojo/parser`无法访问`layout`模块，通常会出错。请确保所有用于 HTML 中的布局元素模块都已加载在调用`dojo/parser`模块的`parse()`方法的`require()`或`define()`语句中。检查模块加载器或 HTML 中的`data-dojo-type`属性中的拼写错误。

使用`dijit/layout`模块创建的布局可以分为两类：**容器**和**面板**。面板元素通常位于容器内部。

## 容器

容器是父元素，用于控制其中分配的子窗格的位置和可见性。容器有多种形状和功能。有些可以同时显示多个窗格，而有些则一次显示一个或几个。在 JavaScript 中，如果您可以访问`dijit`容器，您可以通过调用容器的`getChildren()`方法来访问其内部的窗格元素。

让我们看看一些常见的容器。

### LayoutContainer

`LayoutContainer`允许其他窗格围绕中心窗格进行定位。在`LayoutContainer`中的中心窗格被分配了一个中心区域属性。围绕它的窗格被分配了`top`、`bottom`、`right`、`left`、`leading`或`trailing`区域值来定义它们相对于中心窗格的位置。多个窗格可以具有相同的区域属性，例如两个或三个左侧窗格。这些窗格将并排堆叠在中心窗格的左侧。

### BorderContainer

`BorderContainer`是从`LayoutContainer`派生出来的。正如其名称所暗示的，它为应用程序中的不同窗格添加了边框以分隔它们。`BorderContainers`还可以提供`livesplitters`，这些是可拖动的元素，允许用户根据需要调整窗格的大小。

### AccordionContainer

`AccordionContainer`使用手风琴效果来排列和切换窗格。在这种安排中，窗格标题堆叠在一起，任何给定时间只有一个窗格可见。其他窗格的内容通过手风琴效果被隐藏。当用户在`AccordionContainer`中选择另一个窗格时，窗格将以滑动动画的方式切换，隐藏当前窗格并显示所选窗格。

### TabContainer

`TabContainer`提供了一个基于标签的内容组织。包含`ContentPane`标题的标签描述了内容，点击这些标签将移除内容的可见性。效果类似于旋转文件柜或文件夹，您可以通过翻动标签来查看所需的内容。

## 窗格

`dijit/layout`模块中的窗格为您的应用程序提供了一个容器，用于放置用户控件和小部件。您可以编写 HTML 内容或添加其他 dijit。窗格可以用 HTML 编码，或用 JavaScript 创建并附加到其父容器。让我们看看使用 ArcGIS JavaScript API 在 Dojo 中可用的几个窗格。

### ContentPane

`ContentPane`瓦片是在容器中插入的最常见的窗格。它可以作为除`AccordionContainer`之外所有其他容器中的窗格插入。表面上，它们看起来像是被赋予了跟踪大小和与其他 dijit 之间关系的`div`元素。但`ContentPane`瓦片还可以从其他网页下载并显示内容。设置`ContentPane`瓦片的`href`属性将在您的应用程序的单个窗格中下载并显示另一个网页的 HTML 内容。

### AccordionPane

在`AccordionContainer`内添加一个或多个`AccordionPane`面板，以以可折叠的格式显示其内容。`AccordionPane`标题堆叠在一起，当你点击标题时，内容滑入视图，覆盖之前打开的`AccordionPane`。否则，`AccordionPane`表现出与`ContentPane`相同的职能行为。

现在我们已经回顾了 Dojo 框架如何处理应用程序布局，让我们看看如何使用替代样式框架。

# Bootstrap

如果你正在寻找 Dojo 布局方式的替代方案，你可能需要考虑 Bootstrap。Bootstrap 是一个由 Twitter 的开发者最初创建的流行样式框架。据说，开发者需要一种快速发布网站的方法，因此他们制定了一套样式表作为项目起点。这些样式模板因其易用性和满足大多数网络开发者的需求而非常受欢迎。最初命名为 Twitter Blueprint 的模板后来于 2011 年 8 月作为 Bootstrap 发布。

Bootstrap 为开发者提供了适用于桌面和移动浏览器的响应式设计样式。响应式网格可以根据需要调整，以提供多列布局，在较小的浏览器窗口中折叠成更小的尺寸。Bootstrap 提供了看起来很时尚的表单元素和足够大的按钮，以便在手机浏览器上方便操作。该框架提供了易于理解的 CSS 类，文档和样式表提供了如何使用框架的指导。从可以跨越语言障碍理解的照片图标，到创建模态对话框、标签页、轮播图和其他我们在网站上习惯使用的元素，JavaScript 插件都可以实现。可以使用 Bootstrap 创建整个应用程序。

虽然 Bootstrap 样式不需要 JavaScript 库，但所有 JavaScript 插件都需要 jQuery 才能运行。这对那些使用 Dojo 的人来说并不太有帮助，但我们确实有一个替代方案。

# ESRI-Bootstrap

结合 Bootstrap 的易用性和与 ArcGIS JavaScript API 的兼容性，ESRI 创建了 ESRI-Bootstrap 库([`github.com/Esri/bootstrap-map-js`](https://github.com/Esri/bootstrap-map-js))。该样式框架可以像 Bootstrap 一样调整地图和其他元素的大小，同时许多元素保留了 Bootstrap 网站相同的视觉和感觉。对话框不会跑出屏幕，元素对鼠标点击和触摸都有响应。最后，ESRI Bootstrap 可以与 Dojo 和 jQuery 的组合或仅使用 Dojo 一起使用。

我们将在我们的 Y2K 应用程序上添加 ESRI-Bootstrap。你可以使用我们在第七章中编写的`Dojo/jQuery`应用程序，*与其他人相处融洽*。我们将使用与 jQuery 并行编写的纯 Dojo 应用程序来展示如何在不使用 jQuery 的情况下将 ESRI-Bootstrap 添加到应用程序中。

# 重新设计我们的应用程序

我们最近让我们的实习生使用其他框架创建了我们的应用程序的多个副本。当他们忙于做这件事的时候，我们决定只用 ArcGIS JavaScript API 编写更新。Y2K 协会检查了我们的应用程序并批准了我们的更改，但这并不是他们要说的全部。

当我们与 Y2K 协会董事会会面时，我们发现董事会的新成员中有一位批评者。他们认为看起来还可以，但需要更现代的外观。当被要求详细说明时，他们向我们展示了他们认为看起来不错的网站。我们看到的所有网站都有一个共同点，它们都是使用 Bootstrap 构建的。他设法说服董事会上的其他人，我们的应用程序需要采用这种新风格。

回到原点，我们研究了 ESRI 能提供什么。我们决定将 ESRI-Bootstrap 整合到我们的应用程序中。它提供了现代感。

## 将 ESRI-Bootstrap 添加到我们的应用程序中

让我们先在我们的应用程序中添加对 ESRI-Bootstrap 的引用。ESRI-Bootstrap 的**入门**页面告诉我们下载 GitHub 上发布的最新版本的 ZIP 文件。点击 GitHub 页面右侧的**下载 ZIP**按钮。下载完成后，解压文件。

我们主要对`src/js/`和`src/css/`文件夹中的`bootstrapmap.js`和`bootstrapmap.css`文件感兴趣。其余的文件包含模板和示例，您可以查看它们如何使用 Bootstrap 与地图结合。将这些文件复制到`js`和`css`文件夹中。

接下来，我们将在`index.html`文件的头部标签中添加必要的引用。由于我们不再使用 Dojo 小部件来布局我们的应用程序，我们可以移除`nihilo.css`外部样式表，并添加以下内容：

```py
…
<link href="http://netdna.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css" rel="stylesheet" media="screen" />
<link rel="stylesheet" href="https://js.arcgis.com/3.13/esri/css/esri.css" />
<link rel="stylesheet" type="text/css" href="css/bootstrapmap.css" >
<link rel="stylesheet" href="css/y2k.css" />
…
```

现在我们已经添加了样式表，我们需要在`dojoConfig`变量中添加对 Dojo 版 Bootstrap 的引用。覆盖通常由 jQuery 处理的特性的 Dojo Bootstrap 库可以在[`github.com/xsokev/Dojo-Bootstrap`](https://github.com/xsokev/Dojo-Bootstrap)找到，但我们将从[`rawgit.com`](https://rawgit.com)拉取它，它直接提供 GitHub 代码。它应该看起来像这样：

```py
dojoConfig = {
 async: true,
  packages: [
    …
    {
      name: "d3",
      location: "http://cdnjs.cloudflare.com/ajax/libs/d3/3.4.12/"
    },
 {
 name: "bootstrap",
 location: "http://rawgit.com/xsokev/Dojo-Bootstrap/master"
 }
  ]
};
```

现在头部已经处理好了，是时候重新设计主体了。

## 启动我们的 HTML

由于 ESRI-Bootstrap 提供了大量的 CSS 样式和 JavaScript，大部分工作涉及重新设计我们的 HTML。我们可以从移除主 HTML 页面主体中的 Dojo 布局小部件引用开始。您可以移除`body`元素的`class="nihilo"`，`#mainwindow div`中的`dijit/layout/BorderContainer`，以及所有`dijit/layout/ContentPane`引用。

### 用导航栏替换标题

我们的第一项任务是替换应用程序中的头部。你可以用很多种方式开始这样的页面，但我们将使用 Bootstrap `NavBar` 替换头部，因为页面的初始布局是一个没有滚动的全页应用程序。你可以在 [`getbootstrap.com`](http://getbootstrap.com) 看到如何实现 HTML 中的 `navbar` 的几个示例。

我们将在头部中使用具有 `navbar` 和 `navbar-default` 类的 `nav` 元素替换 `header` `div`。我们将在其中添加一个具有 `container-fluid` 类的 `div`，因为其内容的大小和外观会随着不同的浏览器而变化。我们将在 `container-fluid` div 元素中添加两个 `div` 元素。第一个将具有 `navbar-header` 类，第二个将具有折叠和 `navbar-collapse` 类。

我们将在 `navbar-header` 中添加一个按钮，该按钮仅在浏览器足够窄时才会出现。该按钮将具有 `navbar-toggle` 和 `collapsed` 类，以及 `data-toggle` 属性为折叠。当可见时，该按钮将浮动到最右侧。地图的标题将放在与切换按钮相邻的 `span` 中。这个 `span` 将被赋予 `navbar-brand` 类。

我们将把显示我们的 `Census` 小部件的按钮添加到 `navbar-collapse` div 元素中。我们将添加 `btn` 类（使其成为一个按钮），`btn-default`（使其具有默认颜色），以及 `navbar-btn`（使按钮样式适合 `navbar`）。完成之后，HTML 应该看起来如下所示：

```py
…
<body>
  <nav class="navbar navbar-default">
       <div class="container-fluid">
      <div class="navbar-header">
        <button type="button" class="navbar-toggle collapsed" 
         data-toggle="collapse" 
         data-target="#bs-example-navbar-collapse-1">
          <span class="sr-only">Toggle navigation</span>
          <span class="icon-bar"></span>
          <span class="icon-bar"></span>
          <span class="icon-bar"></span>
        </button>
        <span class="navbar-brand" >Year 2000 Map</span>
      </div>
     <div class="collapse navbar-collapse" 
        id="bs-example-navbar-collapse-1">
        <button id="census-btn" 
           class="btn btn-default navbar-btn">Census</button>
      </div>
       </div>
     </nav>
  …
```

### 重新排列地图

现在头部已经重新设计，是时候看看页面的其余部分了。我们将想要将人口普查小部件的位置从地图中提取出来，以便包含一个 Bootstrap 模态对话框。我们可以将页脚移动到地图内部以填补之前版本中丢失的空间。我们可以在头部下方定义地图和相关项目，如下所示：

```py
…
<div class="modal fade" id="census-widget"></div>
<div id="map">
 <span id="footer">Courtesy of the Y2K Society</span>
</div>
<script type="text/javascript" src="img/app.js"></script>
…
```

## 重新设计我们的应用程序

由于我们正在移除 Dojo 布局系统的所有痕迹，我们需要手动重新设计页面。我们可以移除大部分样式引用，因为它们将与 Bootstrap 样式冲突。我们将保留 HTML 和 body 标签的基本样式，因为它们使应用程序成为一个全页应用程序。我们还将保留 `D3.js` 图表的样式，但可以删除其余的样式。

我们需要将地图从上到下、从左到右拉伸。如果不这样做，地图的宽度和高度将受到限制。我们可以使用绝对定位将地图拉伸到页面。我们将使用之前提到的几乎全页样式。由于页面上的工具栏高度为 50 像素（当你尝试应用程序时你会看到），我们将地图的顶部设置为从顶部 `50px`。底部、右侧和左侧可以定位在屏幕的边缘：

```py
#map {
  position: absolute;
  top: 50px; /* 50 pixels from the top */
  bottom: 0; /* all the way to the bottom */
  left: 0; /* left side 0 units from the left edge */
  right: 0; /* right side 0 units from the right edge */
}
```

我们还需要重新设计另一个元素，即页脚元素。我们可以使用与地图相同的技巧将其定位在页面底部。我们还可以使背景半透明，以达到很好的效果：

```py
#footer {
  position: absolute;
  bottom: 5px; /* 5 pixels from the bottom */
  left: 5px; /* left side 5 pixels from the left edge */
  padding: 3px 5px;
  background: rgba(255,255,255,0.5); /* white, semitransparent */
}
```

一旦应用了这些样式，我们就可以看到我们地图的工作示例。您可以在以下图片中看到地图的示例：

![重新设计我们的应用](img/6459OT_08_01.jpg)

## 使我们的 Census dijit 模态化

现在我们已经将页面转换成了 Bootstrap 应用，我们需要将相同的样式应用到我们的`Census` dijit 上。我们需要利用 Bootstrap 的模态小部件来模仿浮动对话框的效果。

在`js/templates/`文件夹中打开`Census.html`文件。在基础`div`元素中，添加`modal`和`fade`类。`modal`类告诉 Bootstrap 这将是一个模态对话框，而`fade`描述了元素如何隐藏和显示。我们还将向元素添加一个`data-backdrop`属性并将其设置为`static`。这将创建一个通用的模态对话框，在打开时阻止点击页面的其余部分。在这种情况下，我们将放弃关闭小部件将关闭地图点击事件的想法：

```py
<div class="${baseClass} modal fade" 
  style="display: none;" data-backdrop="static">
…
</div>
```

我们将在基础`div`中添加几个更多的`div`元素来定义模态和标题。在我们的模态类内部的一个级别，我们将添加一个具有`modal-dialog`和`modal-sm`类的`div`元素。`modal-dialog`类定义了模态的样式，而`modal-sm`使模态更小。移除`modal-sm`将创建一个跨越更大屏幕的对话框。

我们将在具有`modal-dialog`类的`div`中创建一个具有`modal-content`类的`div`，并在其中添加两个`div`元素，分别具有`modal-header`和`modal-body`类。我们将把我们的关闭按钮和标题添加到`modal-header` div 中。我们将把剩余的`dijit`文本和选择下拉列表添加到`modal-body` div 中：

```py
<div class="${baseClass} modal fade" 
  style="display: none;" data-backdrop="static">
  <div class="modal-dialog modal-sm">
    <div class="modal-content">
      <div class="modal-header">
      …
      </div>
      <div class="modal-body">
      …
      </div>
    </div>
  </div>
</div>
```

我们将在`modal-header` div 中将`dijit`关闭事件替换为 Bootstrap 的模态关闭事件。我们将向按钮添加一个`close`类，并添加一个模态的`data-dismiss`属性。大多数 Bootstrap 示例还包含 ARIA 属性来处理屏幕阅读器和其他辅助工具，因此我们将添加一个`aria-hidden`值为`true`，这样屏幕阅读器就不会大声朗读我们放置在那里的*X*。对于标题，我们将用具有`modal-title`类的`span`包围`Census`数据。它应该看起来像以下代码：

```py
…
<div class="modal-header">
  <button type="button" class="close" 
    data-dismiss="modal" aria-hidden="true">x</button>
  <span class="modal-title">Census Data</span>
</div>
…
```

我们将把我们的描述段落添加到`modal-body` div 中，并格式化选择元素，使它们看起来像表单元素。我们将向围绕我们的选择下拉列表的`div`元素添加`form-group`类。这将使内容对齐并添加适当的间距和格式。我们将用`label`标签替换`b`标签，并添加`control-label`类。我们将向选择元素添加`form-control`类。这将使`select`下拉列表扩展到对话框的宽度。我们的 HTML 应该如下所示：

```py
…
<div class="modal-body">
  <p>
    Click on a location in the United States to view 
    the census data for that region. You can also use 
    the dropdown tools to select a State, County, or 
    Blockgroup.
  </p>
  <div class="form-group" >
    <label class="control-label">State Selector: </label>
    <select class="form-control"
      data-dojo-attach-point='stateSelect' 
      data-dojo-attach-event='change:_stateSelectChanged'>
    </select>
  </div>
  <div class="form-group">
    <label class="control-label">County Selector: </label>
     <select class="form-control" 
       data-dojo-attach-point='countySelect' 
      data-dojo-attach-event='change:_countySelectChanged'>
     </select>
  </div>
  <div class="form-group">
    <label class="control-label">
      Block Group Selector: </label>
    <select class="form-control" 
      data-dojo-attach-point='blockGroupSelect' 
      data-dojo-attach-event='change:_blockGroupSelectChanged'>
    </select>
  </div>
</div>
…
```

我们将在`modal-body` div 中添加一个带有`modal-footer`类的`div`。在这里，我们可以添加另一个按钮来关闭对话框，以防用户没有注意到上角的小`x`。我们将通过添加`btn`和`btn-default`类来格式化关闭按钮，这将影响形状和颜色，顺序如下。我们还将添加`data-dismiss`属性并将其设置为`modal`。它应该看起来如下：

```py
  …
    <div class="modal-footer">
 <button id="btnDismiss" type="button" 
 class="btn btn-default" data-dismiss="modal">
 Dismiss
 </button>
 </div>
  </div>
  …
```

一旦正确格式化小部件 HTML，应用程序应该看起来像以下图像。注意下拉菜单和宽间距的按钮，这使得在较小的移动设备上点击它们更容易：

![制作我们的 Census dijit 模式](img/6459OT_08_02.jpg)

现在应用程序功能已经完善，我们可以修改其部分以创建我们自己的外观和感觉。记住，Bootstrap 旨在作为创建网站的起点。它不一定是最终结果。我们仍然可以更改颜色和其他功能，使应用程序成为我们自己的。

# 摘要

在本章中，我们讨论了您可以使用不同方式来设置您的 ArcGIS JavaScript API 应用程序的样式。我们探讨了 CSS 的工作原理，规则如何相互影响，以及浏览器如何决定遵循哪些 CSS 规则。我们探讨了 Dojo 布局模块，以及如何使用这些模块来处理应用程序的外观和功能。我们还探讨了 ESRI-Bootstrap，这是一个可以与 ArcGIS JavaScript API 一起运行的 Bootstrap 版本。最后，我们将 ESRI-Bootstrap 的外观添加到我们的应用程序中。

在下一章中，我们将转向移动端。我们将创建一个适用于大多数平板电脑和手机浏览器的移动应用程序。
