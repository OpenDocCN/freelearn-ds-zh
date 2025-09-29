# 第三章. Dojo 小部件系统

Esri 的开发者使用 Dojo 框架创建了 ArcGIS JavaScript API。Dojo 提供了大量工具、库和 UI 控件，这些控件可以在多个浏览器中工作。任何开发者都可以使用 Dojo 和 ArcGIS JavaScript API 创建具有良好协作的 UI 元素的自定义应用程序。此外，Dojo 还提供了开发自己的自定义小部件、库和控件的 AMD 工具。

在前面的章节中，我们已经回顾了 ArcGIS API for JavaScript，并使用该 API 编写了一个小型应用程序。我们甚至将 AMD 的基本原理整合到了一个单页应用程序中。到目前为止我们所做的一切对于一个小型应用程序来说都非常适用。

但是当应用程序变得更大时会发生什么？我们是否要实现一个单独的脚本以加载更大应用程序所需的全部组件？我们将如何扩展我们网站的功能？如果我们更新了库而某些东西出了问题怎么办？我们是否要在单个 JavaScript 文件中的数千行代码中查找需要更改的部分？

在本章中，我们将利用 Dojo 框架为我们的应用程序创建一个自定义小部件。通过这个过程，我们将涵盖以下内容：

+   Dojo 框架的背景

+   `dojo`、`dijit` 和 `dojox` 模块包提供的内容

+   如何创建和使用我们自己的自定义模块

+   如何创建小部件（具有 UI 组件的模块），以及如何扩展它们

# Dojo 框架的简要历史

Dojo 框架始于 2004 年，最初在 Informatica 进行工作。Alex Russell、David Schontzler 和 Dylan Schiemann 为该项目贡献了第一行代码。随着代码工作的继续，其他开发者被引入并提供了对框架方向的反馈。项目变得如此之大，以至于创始人创建了 Dojo 基金会来监督代码库及其知识产权。从那时起，超过 60 名开发者为框架做出了贡献，IBM 和 SitePen 等公司至今仍在使用它。更多信息，请访问 [`dojotoolkit.org/reference-guide/1.10/quickstart/introduction/history.html`](http://dojotoolkit.org/reference-guide/1.10/quickstart/introduction/history.html)。

那么，是什么让 Dojo 成为一个框架，而不是一个库呢？当这个问题被提出给 Stack Overflow 的人时 ([`stackoverflow.com/questions/3057526/framework-vs-toolkit-vs-library`](http://stackoverflow.com/questions/3057526/framework-vs-toolkit-vs-library))，最一致的答案集中在控制反转上。当我们使用库中的工具时，我们的代码控制逻辑和活动的流程。当我们计算一个值时，我们通过我们的代码在需要该值的所有位置更新该值。然而，在框架中，框架控制应用程序的行为。当框架中的值被更新时，框架会在网页上绑定该值的任何位置更新该值。

Dojo 框架提供了一些基于 HTML 的控件，我们可以加载和使用。CSS 外观和 JavaScript 行为的大部分内容都预先集成到控件中。在初始设置之后，我们的代码指示框架在特定控件事件发生时将数据发送到何处。当我们的函数被调用时，我们没有控制权，只有调用之后发生的事情。

### 注意

如果您想了解更多关于框架和控制反转的信息，Martin Fowler 在他的博客上提供了对这个主题的良好解释，请访问 [`martinfowler.com/bliki/InversionOfControl.html`](http://martinfowler.com/bliki/InversionOfControl.html)。

# 介绍 dojo、dijit 和 dojox

对 Dojo 框架的组织非常用心。当 Dojo 集成 AMD 模块风格时，许多对象、方法和属性都被重新组织到逻辑文件夹和模块中。Dojo 被分解为三个主要包：**dojo**、**dijit** 和 **dojox**。让我们了解这三个包为框架带来了什么。

## The dojo package

`dojo` 包提供了加载、运行和卸载我们应用程序中各种模块所需的大部分基本功能。这些模块提供了跨多个浏览器（包括令人讨厌的旧版 Internet Explorer）工作的函数和工具。例如，开发者不需要通过尝试 `addEventListener()`（用于 Chrome）和 `attachEvent()`（用于旧版 IE）来处理事件，因为 `dojo/on` 在幕后处理这些。有了这个框架，您可以摆脱所有的浏览器黑客技术，专注于创建一个好的应用程序。

`dojo` 包不包含小部件，但它包含操作小部件内事物所需的工具。您需要处理鼠标点击和触摸事件吗？请使用 `dojo/mouse` 和 `dojo/touch` 模块。您需要一个类似 jQuery 的选择器来选择您的 HTML 吗？请加载 `dojo/query` 模块。`dojo` 包提供了用于 HTML DOM 操作、数组、AJAX、日期、事件和 i18n 的模块，仅举几例。

## The dijit package

`dijit` 包提供了与 `dojo` 包良好集成的视觉元素（称为 **dijits**）。使用 `dijit` 包创建的元素已在多个浏览器上进行了测试。它们基于库中加载的 CSS 样式表提供了一致的用户界面。

由于 Dojo 框架被许多公司和开发者使用和贡献，因此有大量的用户控件可供选择。无论你是在创建提交表单、事件日历还是执行仪表板，你都可以以适合你需求的方式组合这个包中的 dijits。一些更受欢迎的包包括：

+   `dijit/Calendar`：这提供了一个在桌面和平板电脑上都能工作的交互式 HTML 日历控件。

+   `dijit/form`：一个包含按钮、复选框、下拉菜单、文本框和滑块等样式化表单元素的包。这些表单元素在旧版和新版浏览器中具有一致的样式。

+   `dijit/layout`：一个包含处理布局的控件的包。你可以创建简单的容器、标签容器、手风琴容器，甚至可以控制其他容器位置的容器。

+   `dijit/Tree`：一个创建可折叠树菜单的模块，例如，可以用来显示文件夹布局。

`dijit` 包不仅包含用户控件，还提供了创建自定义 dijits 所需的工具。使用 `dijit/_WidgetBase` 模块和混合，你可以将 HTML 元素和现有的 dijits 集成到自己的自定义 `dijit` 中。与 `dijit` 组件一起工作可以为用户提供在整个应用程序中一致的使用体验。

## dojox 包

根据关于 Dojo 框架的文档（[`dojotoolkit.org/reference-guide/1.10/dojox/info.html`](http://dojotoolkit.org/reference-guide/1.10/dojox/info.html)），`dojox` 为 Dojo 提供了扩展。这部分库处理更多实验性功能、用户界面和测试功能。由于广泛的使用，库的许多部分已经成熟，而其他部分已被弃用，并且不再处于积极开发中。

`dojox` 包中的一个有用子包是 `dojox/mobile`。`Dojox/mobile` 包提供了可用于移动应用程序的 UI 元素和控制。它们已在各种移动浏览器上进行了测试，其样式甚至可以模仿原生智能手机和平板电脑应用程序的样式。

### 注意

如需了解 Dojo 框架中 `dojo`、`dijit` 和 `dojox` 包的更多信息，您可以访问教程文档：[`dojotoolkit.org/documentation/tutorials/1.10/beyond_dojo/`](http://dojotoolkit.org/documentation/tutorials/1.10/beyond_dojo/)。

# DojoConfig 包

使用内置的 Dojo 包是很好，但关于创建自己的自定义包怎么办？你可以在本地文件夹中创建自定义包，并通过`dojoConfig`对象引用它们。在`dojoConfig`对象中，你可以添加一个包含 JavaScript 对象数组的`packages`参数。这些包对象应该包含一个`name`属性，它是您包的引用，以及一个`location`属性，它是文件夹位置的引用。以下是一个包含对自定义包引用的`dojoConfig`对象示例：

```py
<script>
  dojoConfig = {
    async: true,
    isDebug: true,
    packages: [
 {
 name: "myapp",
 location: location.pathname.replace(/\/[^/]+$/, '') + "/myapp"
 }
 ]
  };
</script>
```

在前面的示例中，引用了`myapp`包，并且该包的所有文件都被加载到当前页面的`myapp`文件夹下。因此，如果这个页面在`http://www.example.com/testmap/`显示，那么`myapp`包可以在`http://www.example.com/testmap/myapp`找到。当在您的`myapp`包中引用`Something`模块时，您将按如下方式加载模块：

```py
require(["myapp/Something", "dojo/domReady!"], function (Something) { … });
```

# 定义您的部件

使用 Dojo 的 AMD 风格，有两种主要方式来使用 AMD 组件。使用`require()`函数只播放一次脚本，然后完成。但如果你想要创建一个可以重复使用的模块，你将想要`define()`该模块。`define()`函数创建一个或多个自定义模块，供应用程序使用。

`define()`函数接受一个单一参数，它可以是任何 JavaScript 对象。即使是`define("Hello World!")`也是一个有效的模块定义，尽管不是那么有用。你可以通过传递执行应用程序任务的函数或对象构造函数来创建更有用的模块。请参阅以下示例：

```py
define(function () {
  var mysteryNumber = Math.floor((Math.random() * 10) + 1);
  return {
    guess: function (num) {
      if (num === mysteryNumber) {
        alert("You guessed the mystery number!");
      } else if (num < mysteryNumber) {
        alert("Guess higher.");
      } else {
        alert("Guess lower.");
      }
    }
  };
});
```

在前面的示例中，该模块从一到十中随机选择一个数字，然后返回一个带有`guess()`方法的模块，该方法接受一个数值。调用`guess()`方法时，它会向用户提示猜测是否正确。

## 声明模块

从更传统的**面向对象**（**OO**）语言进入 JavaScript 的开发者可能难以接受这种语言基于原型的继承机制。在大多数面向对象的语言中，类是在编译时定义的，有时会从其他基类继承属性和方法。在 JavaScript 对象中，类类型的对象存储在对象的原型中，这些原型为多个相关对象持有相同的共享属性和方法。

当这些开发者使用`dojo/_base/declare`模块时，Dojo 可以帮助他们更容易地过渡。该模块创建基于 Dojo 的类，这些类可以被其他应用程序使用。在幕后，它将类似类的 JavaScript 对象转换为基于原型的构造函数。此模块与`define()`函数很好地配合，用于创建自定义的`dojo`模块，如下所示：

```py
define(["dojo/_base/declare", …], function (declare, …) {
  return declare(null, {…});
});
```

`declare()` 函数接受三个参数：类名、父类和类属性对象。类名是一个可选的字符串，可以用来引用类。`declare()` 函数会将该名称转换为全局变量，以便当你在 `data-dojo-type` 属性中引用 `dijit` 包时，`dojo/parser` 可以将其写入 HTML。如果你不打算使用 `dojo/parser` 将你的小部件写入 HTML，强烈建议你不要使用第一个参数。

### 通过 Dojo 进行类继承

`declare()` 函数的第二个参数指的是父类。定义父类有三种可能的方式。如果你创建的类没有父类，则称其没有继承。在这种情况下，父类参数为 null，如下面的语句所示：

```py
define(["dojo/_base/declare", …], function (declare, …) {
  return declare(null, {…});
});
```

在第二种场景中，你创建的类有一个父类。这个父类可以是另一个 AMD 模块，或者另一个已声明的项。以下示例通过扩展 `dijit/form/Button` 模块来展示这一点：

```py
define(["dojo/_base/declare", "dijit/form/Button", …],
  function (declare, Button, …) {
    return declare(Button, {…});
  }
);
```

第三种可能的情况涉及多重继承，其中你的类从多个类继承属性和方法。为此，父类参数也将接受一个对象数组。数组中的第一个项目被认为是基类，它提供了主要的构造参数。列表中的其余项目被称为“混合”。它们不一定提供对象构造功能，但它们添加了属性和方法，这些属性和方法要么补充，要么覆盖现有的基类。以下示例展示了使用我们稍后将要讨论的几个库进行多重继承：

```py
define(["dojo/_base/declare", "dijit/_WidgetBase", "dijit/_OnDijitClickMixin, "dijit/_TemplatedMixin"], 
  function (declare, _WidgetBase, _OnDijitClickMixin, _TemplatedMixin) {
    return declare([_WidgetBase, _OnDijitClickMixin, _TemplatedMixin], {…});
});
```

### 处理类和继承

在 `declare()` 语句的最后一个参数中，开发者定义了类中包含的不同属性和方法。对象属性可以是字符串、数字、列表、对象或函数。请查阅以下代码示例：

```py
declare("myApp/Fibonacci", null, {
  list: [],
  contructor: function () {
    var len = this.list.length;
    if (len < 2) {
      this.myNumber = 1;
    } else {
      this.myNumber = this.list[len – 1] + this.list[len-2];
    }
    this.list.push(this.myNumber);
  },
  showNumber: function () { alert(this.myNumber); }
});
```

在使用 `declare()` 语句创建的对象中，你既有静态属性也有实例属性。**静态**属性是所有以相同方式创建的对象共享的属性。**实例**属性是创建的对象独有的属性。在类对象内部定义的属性被认为是静态的，但最初在构造函数或另一个方法中分配的每个属性都被认为是实例属性。

在前面的例子中，`showNumber()` 方法和 `list` 属性是静态的，而 `myNumber` 属性是一个实例。这意味着每个 `myapp/Fibonacci` 对象将共享 `showNumber()` 方法和 `list` 数组，但将具有独特的 `myNumber` 属性。例如，如果创建了五个 `myapp/Fibonacci` 对象，每个对象都应该包含列表值 `[1, 1, 2, 3, 5]`。向列表中添加一个新元素将添加到每个斐波那契列表中。`myNumber` 属性是在构造函数中创建的，因此每个对象都将具有该属性的独特值。

当从父类创建一个类时，它可以访问其父类的属性和方法。新类也可以通过在其构造函数中拥有同名的新方法来覆盖这些方法。例如，假设一个 `HouseCat` 对象继承自 `Feline` 对象，并且它们都有自己的 `pounce()` 方法版本，如果你调用 `HouseCat.pounce()`，它将只运行 `HouseCat` 中描述的方法，而不是 `Feline` 中的方法。如果你想在 `HouseCat.pounce()` 调用中运行 `Feline.pounce()` 方法，你可以在 `HouseCat.pounce()` 方法中添加 `this.inherited(arguments)`，以显示你想要运行父类方法的时间。

# 与 Evented 模块一起工作

`Evented` 模块允许你的小部件发布事件。当你的模块以 `Evented` 作为父类声明时，它提供了一个 `emit()` 方法来广播事件已发生，以及一个 `on()` 方法来监听事件。在 ArcGIS API 中可以找到一个例子，那就是绘图工具栏。它不显示信息，但它有发布事件的必要工具。

`emit()` 方法接受两个参数。第一个是一个描述事件的字符串名称，例如 `map-loaded` 或 `records-received`。第二个是与事件一起创建和传递的对象。该对象可以是你在 JavaScript 中创建的任何内容，但请记住保持返回内容相似，这样监听事件发生的方法就不会错过它。

# `_WidgetBase` 模块的概述

`_WidgetBase` 模块提供了创建 `dijit` 模块所需的基类。它本身不是一个小部件，但你可以很容易地在它上面构建一个小部件。使用 `_WidgetBase` 模块创建的小部件绑定到 HTML DOM 中的一个元素，可以通过其 `domNode` 属性来引用。

`_WidgetBase` 模块还引入了小部件的生命周期。生命周期指的是小部件的创建、构建、使用和销毁的方式。此模块加载以下在生命周期中发生的方法和事件：

+   `constructor`: 这个方法在创建小部件时被调用。无法访问模板，但可以分配值和访问数组。

+   `混合到小部件实例中的参数`：你传递给对象构造函数的参数，如按钮标签，被添加到小部件中，以覆盖任何现有的值。

+   `postMixinProperties`：这是渲染小部件 HTML 之前的一个步骤。如果您需要更改或纠正传递给参数的任何值，或者进行其他操作，这将是一个进行这些操作的好时机。

+   `buildRendering`：如果正在使用模板，这是在现有节点中添加 HTML 模板的地方。如果您正在使用 `_TemplatedMixin` 模块，模板字符串将被加载到浏览器中，渲染成 HTML，并插入到现有的 `domNode` 位置。绑定在 HTML 数据-* 属性中的事件也在这里分配。

+   `Custom setters are called`：如果您添加了自定义设置函数，这些函数将在此时被调用。

+   `postCreate`：现在 HTML 已经渲染，这个函数可以对其进行更多操作。但请注意，小部件的 HTML 可能尚未连接到文档。此外，如果这个小部件包含子小部件，它们可能尚未渲染。

+   `startup`：这个函数可以在所有解析和子小部件渲染完成后被调用。它通常用于小部件需要通过 `resize()` 方法调整大小时。

+   `destroy`：当这个小部件被拆解并移除时，无论是关闭应用程序还是调用此函数时，都会调用此函数。父类通常自行处理拆解事件，但如果您在销毁小部件之前需要执行任何独特操作，这将是一个扩展函数的好时机。

`_WidgetBase` 模块提供了专门的获取和设置函数，允许您在设置小部件属性时执行特定任务。您可以使用 `get()` 函数检索值，并使用 `set()` 函数设置这些属性。`get()` 和 `set()` 的第一个参数是属性的名称，而 `set()` 方法中的第二个参数是值。因此，如果您想设置小部件的 `name`，您将调用 `widget.set("name", "New Name")`。还有一个 `watch()` 方法，可以在通过 `set()` 改变值时执行函数。

# 与其他 `_Mixins` 一起工作

`_WidgetBase` 模块可能无法提供您应用程序所需的所有小部件功能。`dijit` 包为您的提供了混合，或 JavaScript 对象扩展，用于小部件。这些混合可以提供 HTML 模板、点击、触摸、聚焦以及无聚焦的事件处理，例如。使用正确的混合与您的应用程序可以节省大量的行为编码。让我们看看我们可能会使用的一些混合。

## 添加 `_TemplatedMixin`

`_TemplatedMixin` 模块允许模块用字符串模板或来自另一个来源的 HTML 替换其现有的 HTML 内容。小部件的模板字符串通过 `templateString` 属性分配。这允许小部件跳过可能复杂的 `buildRendering` 步骤。您可以在以下代码片段中看到一个 `_TemplatedMixin` 模块的调用示例：

```py
require(["dojo/_base/declare", "dijit/_WidgetBase", "dijit/_TemplatedMixin"], 
function (declare, _WidgetBase, _TemplatedMixin) {
  declare("ShoutingWidget, [_WidgetBase, _TemplatedMixin], {
    templateString: "<strong>I AM NOT SHOUTING!</strong>"
  });
});
```

随着 `templateString` 属性的渲染，属性可以在模板和小部件之间来回传递。在 `buildRendering` 阶段，可以通过在组件中引用属性名并在其周围使用 `${}` 包装器来将小部件属性写入模板。你还可以使用 HTML 数据属性分配节点引用并附加小部件事件。`data-dojo-attach-point` 属性允许小部件有一个命名属性，该属性连接到模板。`data-dojo-attach-event` 属性在小部件中附加一个当事件发生时被触发的 `callback` 方法。你可以查看以下示例：

```py
<div class="${baseClass}">
  <span data-dojo-attach-point="messageNode"></span>
  <button data-dojo-attach-event="onclick:doSomething">
    ${label}
  </button>
</div>
```

### 提示

开发者应该如何布局一个模板小部件？小部件的模板 HTML 应该全部包含在一个单独的 HTML 元素中。你可以使用 `<div>` 标签、`<table>` 标签，或者你计划将模板包围在内的元素标签。如果模板在基础层包含多个元素，小部件将无法渲染并抛出错误。

## 添加 _OnDijitClickMixin

`_OnDijitClickMixin` 模块允许你的模板中的元素能够“被点击”。这不仅适用于使用鼠标的点击事件，也适用于触摸和键盘事件。除了点击和触摸一个元素外，用户可以通过按制表符键直到元素被突出显示，然后按 *Enter* 键或空格键来激活它。

如果加载了 `_OnDijitClickMixin` 模块，开发者可以通过 `data-dojo-attach-event` 属性向 `dijit` 模板添加事件处理器。在此数据属性文本值中，添加 `ondijitclick:` 后跟你的点击事件处理器名称。你必须确保这个事件处理器指向一个有效的函数，否则整个小部件将无法加载。以下是一个使用 `clickMe(event) {}` 函数的 `dijit` 模板示例：

```py
<div class="clickable" data-dojo-attach- event="ondijitclick:clickMe"> I like to be clicked.</div>
```

作为旁注，点击事件处理器中的函数应该准备好接受一个单次点击事件参数，没有其他。

## 添加 _FocusMixin

`_FocusMixin` 模块为你的小部件及其组件提供聚焦和失焦事件。例如，如果你有一个在聚焦时占用更多空间的小部件，你可以在对象中添加一个 `_onFocus()` 事件处理器，如下所示：

```py
_onFocus: function () { domClass.add(this.domNode, "i-am-huge");}
```

另一方面，当你想让小部件缩小回正常大小时，你将添加一个 `_onBlur()` 事件处理器：

```py
_onBlur: function() { domClass.remove(this.domNode, "i-am-huge");}
```

# 事件系统

JavaScript 中的事件系统是该语言的重要组成部分。JavaScript 被设计为响应浏览器中的事件和用户引起的事件。一个网站可能需要响应用户的点击，或服务器的 AJAX 响应。使用该语言，你可以将称为事件处理器的函数附加到页面和浏览器中某些元素上，以监听事件。

在 Dojo 框架中，使用 `dojo/on` 模块来监听事件。在分配事件监听器时，模块函数调用返回一个对象，您可以通过调用该对象的 `remove()` 方法来停止监听。此外，`dojo/on` 有一个 `once()` 方法，当事件发生时，它会触发事件一次，然后自动删除该事件。

### 小贴士

一些较老的 ArcGIS API 示例仍然使用 Dojo 的旧事件处理器，`dojo.connect()`。事件处理器将通过 `dojo.connect()` 进行附加，并通过 `dojo.disconnect()` 移除。目前，Dojo 基金会正在弃用 `dojo.connect()`，并在 Dojo 升级到 2.0 版本时将其从代码中删除。如果您正在维护旧代码，请开始将所有 `dojo.connect()` 迁移到 `dojo/on`。在使用 ArcGIS API 时，请注意事件名称和返回的结果。名称可能从驼峰式命名法变为以破折号分隔，虽然 `dojo.connect()` 可以在其回调中返回多个项目，但 `dojo/on` 只返回一个 JavaScript 对象。

事件是通过 `emit()` 创建的。此函数接受事件字符串名称和一个您希望传递给事件监听器的 JavaScript 对象。`emit()` 方法适用于使用 `dojo/Evented` 模块的小部件，以及使用 `dijit/_WidgetBase` 模块构建的小部件。它也通过 `dojo/on` 模块提供，但 `dojo/on.emit()` 在事件名称之前需要一个 HTML DOM。

现在我们已经掌握了 Dojo 框架，让我们用它来构建一些东西。

# 创建我们自己的小部件

现在我们已经了解了创建自定义小部件的基础，也就是说，为我们的网络应用创建 dijit，让我们将我们的知识付诸实践。在本章的这一部分，我们将把我们在 第一章 中编写的单页应用程序代码，*您的第一个映射应用*，转换为一个可以与任何地图一起使用的小部件。然后，我们将使用这个小部件在可以扩展以包含其他小部件的地图应用程序中。

## 从我们上次停止的地方继续…

我们最终收到了 Y2K 社会对我们在 第一章 中创建的人口普查地图的反馈。他们喜欢地图的工作方式，但觉得它需要更多的润色。在与一些小组成员会议后，以下是他们的反馈列表：

+   应用程序应该在顶部和底部有部分来显示标题和关于社会的简要信息

+   颜色应该是温暖和愉快的，例如黄色

+   摆脱人口普查图层数据的普通黑色线条

+   应用程序应该让我们能够做的不仅仅是查看人口普查数据。它应该让我们点击其他按钮来做其他事情

+   人口普查信息需要更加有组织和逻辑地分组在表格中

为了满足他们的要求，我们需要从 第一章 *您的第一个映射应用程序* 重新组织我们的应用程序。为了解决这个问题，我们需要执行以下操作：

1.  我们将创建一个新的全页应用程序，包含页眉和页脚。

1.  我们将修改应用程序的样式。

1.  我们将使用自定义符号表示人口普查地图服务。

1.  我们将在页眉中添加按钮以启动普查弹出窗口。

1.  我们将把 `infoTemplate` 数据移动到单独的 HTML 文件中。

### 我们应用程序的文件结构

为了提供一个不仅能显示人口普查数据的地图，我们将使用 Dojo 框架提供的工具。我们将使用 Dojo 的元素来布局应用程序，并将我们之前的普查地图工具转换为其自己的小部件。而不是将 HTML、CSS 和 JavaScript 混合在同一文件中的单页应用程序，我们将将这些文件分离成独立的组件。

让我们从组织项目文件夹开始。将样式和脚本分别放入不同的文件夹是一个好主意。在我们的项目文件夹中，我们将向项目文件夹添加两个子文件夹。我们将它们命名为 `css` 和 `js`，分别用于样式表和脚本文件。在 `js` 文件夹内，我们将添加另一个文件夹并命名为 `templates`。这是我们的小部件模板将放置的地方。

在主项目文件夹中，我们将创建一个名为 `index.html` 的 HTML 页面。接下来，在 `css` 文件夹中，我们将创建一个名为 `y2k.css` 的文件，这将是我们的应用程序样式表。在我们的 `js` 文件夹中，我们将创建两个 JavaScript 文件：一个名为 `app.js` 的用于应用程序，另一个名为 `Census.js` 的用于小部件。我们将在 `js`/`templates` 文件夹中为普查小部件（`Census.html`）创建模板文件，以及一些弹出窗口的模板。文件结构应类似于以下：

![我们应用程序的文件结构](img/6459OT_03_01.jpg)

## 使用 Dojo 定义布局

为了处理应用程序的布局，我们将使用 `dijit/layout` 包内的几个模块。这个包包含许多具有样式外观和内置 UI 行为的 UI 容器，我们不需要重新发明。我们将添加布局元素并重新设计它们以满足客户的需求。让我们看看 `dijit` 包内找到的布局模块。

### 使用 BorderContainer 框架布局页面

`BorderContainer` 容器提供了 Dojo 中常用的布局工具。`BorderContainer` 容器是从 `LayoutContainer` 模块扩展而来的，它创建了一个填充整个浏览器窗口而不需要滚动的布局。`LayoutContainer` 模块（以及通过扩展的 `BorderContainer` 模块）允许使用区域属性放置内容元素。`BorderContainer` 容器添加了边框、间距和可调整大小的元素，称为分隔符，可以拖动以调整内容项的大小。

可以分配给 `BorderContainer` 容器内部的内容 dijits 的区域如下：

+   **中心**：将元素定位在页面中心，相对于其他项目。必须将此值分配给一个元素。

+   **顶部**：将元素定位在中心元素的上方。

+   **底部**：将元素定位在中心元素的下方。

+   **右侧**：将元素定位在中心元素的右侧。

+   **左侧**：将元素定位在中心元素的左侧。

+   **leading**：如果页面的 `dir` 属性设置为 "ltr"（从左到右），则 leading 元素放置在中心元素的左侧。如果 `dir` 属性为 "rtl"（从右到左），则 leading 元素放置在右侧。

+   **尾部**：布局与 leading 元素相反，"ltr" 在右侧，"rtl" 在左侧。

`BorderContainer` 模块及其父模块 `LayoutContainer` 有一个设计参数会影响围绕中心元素的容器。当 `design` 参数设置为默认值 `headline` 时，顶部和底部面板占据容器的整个宽度。右侧、左侧、leading 和 trailing 面板缩小以适应顶部和底部面板之间。或者，您可以将 `design` 参数设置为 `sidebar`，侧面板将占据整个高度，而顶部和底部面板则被挤压在它们之间。以下图显示了使用 headline 和 sidebar 设计的示例页面。示例中只更改了 `design` 参数：

![使用 BorderContainer 边框容器框定页面](img/6459OT_03_02.jpg)

具有带有 'headline'（左侧）和 'sidebar'（右侧）设计属性的 BorderContainer

### 插入内容面板

`ContentPane` 面板提供了在大多数桌面 Dojo 页面中使用的通用容器。`ContentPane` 面板可以调整以适应其分配的位置，并且可以根据用户调整浏览器大小而扩展和收缩。`ContentPane` 面板可以容纳 HTML 元素，例如其他 Dojo 小部件和控制。它们还有一个 `href` 属性，当设置时，将加载并渲染另一个 HTML 页面的内容。由于跨域问题，页面通过 **XMLHttpRequest** （**XHR**） 加载，因此加载的 HTML 应该在同一域上。

### 修改我们应用程序的布局

观察需求后，我们将使用 `BorderContainer` 容器和一些 `ContentPane` 面板来创建应用程序的标题、页脚和内容。我们将使用标准的 HTML 模板获取必要的库，然后添加 `dijit` 内容。

在你的应用程序文件夹中，首先创建一个名为`index.html`的 HTML 文件。我们将将其命名为`Y2K Map`，并添加必要的元标签和脚本以加载 ArcGIS API for JavaScript。我们还将加载`dojo` CSS 样式表`nihilo.css`，以处理应用程序中`dojo`元素的基本样式。为了将样式应用到我们的应用程序中，我们还将向我们的 HTML 文档的`body`元素添加一个`nihilo`类。最后，我们将添加一个链接到我们即将创建的`y2k.css`文件。你的 HTML 文档应该看起来像以下这样：

```py
<!DOCTYPE HTML>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=Edge" />
    <meta name="viewport" content="initial-scale=1, maximum-scale=1,user-scalable=no" />
    <title>Y2K Map</title>
    <link rel="stylesheet" href="http://js.arcgis.com/3.13/dijit/themes/nihilo/nihilo.css">
    <link rel="stylesheet" href="https://js.arcgis.com/3.13/esri/css/esri.css" />
    <link rel="stylesheet" href="css/y2k.css" />
    <script src="img/"></script>
  </head>
  <body class="nihilo">
  </body>
</html>
```

接下来，我们将在`<body>`标签内添加`BorderContainer`容器和`ContentPane`瓷砖。这些将基于基本的`<div>`元素构建。我们将给`BorderContainer`容器一个`id`为`mainwindow`，顶部一个`ContentPane`瓷砖，其`id`为`header`，另一个`ContentPane`瓷砖，其`id`为`map`，我们的地图将放在这里，还有一个`ContentPane`瓷砖在底部，其`id`为`footer`。我们还会在标题和页脚中添加一些内容，使其看起来更美观。以下是一个示例：

```py
<body class="nihilo">
 <div id="mainwindow" data-dojo-type="dijit/layout/BorderContainer" data-dojo-props="design:'headline',gutter:false,liveSplitters: true" style="width: 100%; height: 100%; margin: 0;">
    <div id="header" data-dojo-type="dijit/layout/ContentPane"
 data-dojo-props="region:'top',splitter:true">
      <h1>Year 2000 Map</h1>
 </div>
    <div id="map" data-dojo-type="dijit/layout/ContentPane" 
 data-dojo-props="region:'center',splitter:true">
 </div>
    <div id="footer" data-dojo-type="dijit/layout/ContentPane" 
 data-dojo-props="region:'bottom',splitter:true"
 style="height:21px;">
      <span>Courtesy of the Y2K Society</span>
    </div>
  </div>
</body>
```

由于客户需要一个地方来添加多个功能，包括我们的普查搜索，我们将在右上角添加一个按钮位置。我们将创建一个包含`<div>`的按钮，并将我们的普查按钮作为`dijit/form/Button`模块插入。使用`dijit`按钮将确保该部分的样式将与小部件的样式保持一致。以下是一个示例：

```py
<div id="header" 
  data-dojo-type="dijit/layout/ContentPane"
  data-dojo-props="region:'top',splitter:true">
    <h1>Year 2000 Map</h1>
    <div id="buttonbar">
 <button data-dojo-type="dijit/form/Button"
 id="census-btn" >Census</button>
 </div>
</div>
```

为了使文件正常工作，我们需要添加一个链接到我们将为应用程序运行的 main 脚本文件。我们将把`app.js`文件作为`<body>`标签内的最后一个元素插入，这样加载文件就不会阻止浏览器下载其他内容。在以下代码中，你可以看到我们插入的位置：

```py
      <span>Courtesy of the Y2K Society</span>
    </div>
  </div>
  <script type="text/javascript" src="img/app.js"></script>
</body>
```

在`app.js`文件中，我们将只做足够的事情来获得视觉效果。我们将随着工作的进行向此文件添加内容。我们将从正常的`require()`语句开始，加载`BorderContainer`、`ContentPane`和`Button`元素的模块。我们还将引入一个名为`dojo/parser`的模块，该模块将解析 HTML 中的 Dojo 数据标记并将其转换为应用程序小部件。代码如下：

```py
require([
  "dojo/parser",
  "dijit/layout/ContentPane", 
  "dijit/layout/BorderContainer", 
  "dijit/form/Button",
  "dojo/domReady!"
], function(
  parser
) {
  parser.parse();
});
```

经过所有这些工作后，我们的 HTML 应该看起来像以下这样：

```py
<!DOCTYPE HTML>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=Edge" />
    <meta name="viewport" content="initial-scale=1, maximum-scale=1,user-scalable=no" />
    <title>Y2K Map</title>
    <link rel="stylesheet" href="http://js.arcgis.com/3.13/dijit/themes/nihilo/nihilo.css">
    <link rel="stylesheet" href="https://js.arcgis.com/3.13/esri/css/esri.css" />
    <link rel="stylesheet" href="css/y2k.css" />
    <script src="img/"></script>
  </head>
  <body class="nihilo">
    <div id="mainwindow" data-dojo-type="dijit/layout/BorderContainer"
      data-dojo-props="design:'headline',gutter:false, liveSplitters: true"
      style="width: 100%; height: 100%; margin: 0;">
        <div id="header" 
          data-dojo-type="dijit/layout/ContentPane"
          data-dojo-props="region:'top',splitter:true">
          <h1>Year 2000 Map</h1>
          <div id="buttonbar">
            <button id="census-btn" data-dojo-type="dijit/form/Button">Census</button>
          </div>
        </div>
        <div id="map" data-dojo-type="dijit/layout/ContentPane" data-dojo-props="region:'center',splitter:true">
          <div id="census-widget"></div>
        </div>
        <div id="footer"
          data-dojo-type="dijit/layout/ContentPane" 
          data-dojo-props="region:'bottom',splitter:true"
             style="height:21px;">
        <span>Courtesy of the Y2K Society</span>
      </div>
    </div>
    <script type="text/javascript" src="img/app.js"></script>
  </body>
</html>
```

## 应用程序样式

如果我们现在运行应用程序，它看起来可能不会像预期的那样。事实上，它看起来就像一个空白屏幕。这是因为我们还没有为我们的页面分配大小。在这种情况下，`ContentPane`瓷砖的样式创建出定位外观的面板，绝对定位将内容从计算页面流程中移除。由于没有其他内容来填充`body`的高度，它就塌陷到零高度。

解决这个问题的快速方法是更新 HTML 和`<body>`标签的样式。在你的文本编辑器中打开`y2k.css`，并添加以下 CSS 行：

```py
html, body {
  border: 0;
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
  font-size: 14px;
}
```

我们应用的这个 CSS 使页面填充了浏览器窗口的 100%。添加边框、边距和填充移除了不同浏览器可能插入页面的任何可能的格式化。我们添加了字体大小，因为在 body 级别设置像素字体大小是一种良好的实践。进一步的字体大小分配可以使用`em`单位相对于这个值进行。

如果你现在播放你的页面，你应该看到以下输出：

![我们的应用样式](img/6459OT_03_03.jpg)

这无疑是进步，但我们还需要满足应用的其它要求。首先，如果我们将`#buttonbar`元素精确地定位在右上角，并为**人口普查**按钮留出一些垂直居中的空间，它会看起来更好。接下来，我们将在不同的面板上添加一些黄色色调，并使页眉和页脚的角落变得圆润。我们的应用最终将具有以下风格化的外壳：

![我们的应用样式](img/6459OT_03_04.jpg)

要实现这一点，这是我们将添加到`y2k.css`中的 CSS：

```py
h1 {
  margin: 2px 8px;
  display: inline-block;
  *display: inline; /* IE 7 compatible */
  zoom: 1; /* IE 7 compatible */
}

#buttonbar {
  position: absolute;
  top: 10px;
  right: 15px;
  width: auto;
  height: auto;
}

#mainwindow,
#mainwindow .dijitSplitter {
  background: #fffaa9; /* paler yellow */
}

#header, #footer {
  -moz-border-radius: 5px;
  -webkit-border-radius: 5px;
  border-radius: 5px;
  border: 1px solid #6f6222; 
}

#header { background: #ffec50; /* bold yellow */ }

#footer { background: #d0a921; /* darker yellow */ }
```

## 添加我们的自定义包

在我们能够使用自定义包之前，我们需要告诉 Dojo 它在哪。为此，我们将在加载 ArcGIS JavaScript API 之前添加一个`dojoConfig`对象。在`dojoConfig`对象中，我们将添加一个包数组对象，并告诉它`y2k`模块位于`js`文件夹中。脚本看起来可能像这样：

```py
<link rel="stylesheet" href="css/y2k.css" />
<script type="text/javascript">
  dojoConfig = {
    packages: [
      {
        name: 'y2k',
        location: location.pathname.replace(/\/[^\/]*$/, '') + 
          '/js'
      }
    ]
  };
</script>
<script src="img/"></script>
```

## 设置我们的应用

现在我们需要设置我们的应用脚本并开始移动。我们将打开我们的`app.js`文件，并添加地图和即将到来的`Census`小部件的功能。我们将像在第一章中做的那样添加对`esri/map`模块和我们的`y2k/Census`小部件的引用。我们将初始化地图，就像我们在第一章中做的那样，并创建一个新的`Census`小部件。遵循许多 dojo 小部件的模式，`Census`小部件构造函数将接受一个选项对象作为第一个参数，以及一个指向`<div>`地图内 HTML 节点的字符串引用作为第二个参数。我们将在稍后填写选项：

```py
require([
  "dojo/parser",
 "esri/map", "y2k/Census",
  "dijit/layout/ContentPane", 
   …
], function(
  parser, Map, Census
) {
  parser.parse();

  var map = new Map("map", {
 basemap: "national-geographic",
 center: [-95, 45],
 zoom: 3
 });

 var census = new Census({ }, "census-widget");

});
```

### 注意

可能你会想知道为什么我们代码中的一些模块以大写字母开头，而另一些则不是。常见的 JavaScript 编码约定指出，对象构造函数应以大写字母开头。`Map`和`Census`模块创建地图和人口普查小部件，因此它们应该大写。为什么对`esri/map`模块的引用没有大写仍然是个谜，如果你弄错了，这可能会成为错误的一个来源。

## 编写我们的小部件

我们需要开始组装我们的人口普查小部件。为此，我们将使用本章前面学到的创建自定义小部件的知识。在`Census.js`文件中，我们将使用`define()`、`declare()`和`_WidgetBase`模块创建一个外壳小部件。它应该看起来像以下代码：

```py
define([
  "dojo/_base/declare", 
  "dijit/_WidgetBase"
], function ( declare, _WidgetBase ) {
  return declare([_WidgetBase], { });
});
```

对于我们的小部件，我们希望有一个模板来指导用户如何使用工具。也许让用户关闭工具也是一个好主意，因为我们不希望地图被多个工具所杂乱：

```py
define([
  "dojo/_base/declare", 
  "dijit/_WidgetBase", "dijit/_TemplatedMixin", 
 "dijit/_OnDijitClickMixin"], 
function (declare, _WidgetBase, _TemplatedMixin, 
 _OnDijitClickMixin) {
  return declare([_WidgetBase, _TemplatedMixin, _OnDijitClickMixin], { });
});
```

在这一点上，我们需要将我们的模板加载到我们的小部件中。为了做到这一点，我们将实现另一个名为`dojo/text`的 dojo 模块。

### 添加一些`dojo/text`！

`dojo/text`模块允许模块以字符串的形式下载任何类型的文本文件。内容可以是 HTML、文本、CSV 或任何相关的基于文本的文件。当使用 AMD 加载文件时，格式如下：

```py
require([…, "dojo/text!path/filename.extension", …], 
function (…, textString, …) {…});
```

在前面的例子中，`filename.extension`描述了文件名，例如`report.txt`。路径显示了文件相对于脚本的定位。因此，路径`./templates/file.txt`意味着该文件位于`templates`文件夹中，它是包含此小部件脚本文件夹的子文件夹。

我们声明中的感叹号表示该模块有一个插件属性，可以在加载时自动调用。否则，我们不得不等待并在我们的脚本中模块加载后调用它。我们看到这种情况的另一个模块是`dojo/domReady`。感叹号激活了该模块，暂停我们的应用程序，直到 HTML DOM 也准备好。

回到我们的应用程序，现在是加载我们的 dijit 模板的时候了。`_TemplatedMixin`模块提供了一个名为`templateString`的属性，它从中读取以构建`dijit`包的 HTML 部分。我们将使用`dojo/text`模块从我们的`Census.html`模板中加载 HTML，然后将从该字符串创建的字符串插入到`templateString`属性中。它看起来可能像这样：

```py
define([
  …
  "dijit/_OnDijitClickMixin",
  "dojo/text!./templates/Census.html"
], function (…, _OnDijitClickMixin, dijitTemplate) {
  return declare([…], {
    templateString: dijitTemplate 
  });
});
```

### 我们的小部件模板

对于我们的模板，我们将利用 Dojo 的一些酷技巧。第一个技巧是我们可以通过与我们在第一章中学习的相同替换模板，将属性值混合到我们的小部件中，*您的第一个映射应用程序*。其次，我们将利用`_OnDijitClickMixin`模块来处理点击事件。

对于我们的模板，我们正在考虑创建一个带有标题（例如“人口普查”），一些说明和一个`close`按钮的东西。我们将通过`data-dojo-attach-event`属性使用`ondijitclick`分配一个关闭按钮的事件处理程序。我们还将从小部件分配一个`baseClass`属性到小部件中的 CSS 类。如果你还没有在你的`template`文件夹中创建`Census.html`文件，现在就创建它。然后，输入以下 HTML 内容：

```py
<div class="${baseClass}" style="display: none;">
  <span class="${baseClass}-close" 
    data-dojo-attach-event="ondijitclick:hide">X</span>
  <b>Census Data</b><br />
  <p>
    Click on a location in the United States to view the census 
    data for that region.
  </p>
</div>
```

### 小贴士

模板中可能存在哪些错误会导致小部件无法加载？如果你通过`data-dojo-attach-event`参数合并事件，请确保模板中的回调函数与你的 dijit 中的回调函数名称匹配。否则，dijit 将无法加载。

在我们的代码中，我们将分配一个 `baseClass` 属性，以及 `hide()` 和 `show()` 函数。许多 dijit 使用这些函数来控制它们的可见性。通过这些函数，我们将显示样式属性设置为 `none` 或 `block`，如下面的代码所示：

```py
define([…, "dojo/dom-style"], 
function (…, domStyle) { …
{
  baseClass: "y2k-census",

  show: function () {
    domStyle.set(this.domNode, "display", "block");
  },

  hide: function () {
    domStyle.set(this.domNode, "display", "none");
  }
});
```

### 使用 dijit 构造函数

我们需要扩展 `dijit` 构造函数功能，但首先我们需要考虑这个 dijit 需要什么。我们最初创建 dijit 的目的是让用户点击地图并识别人口普查地点。因此，我们需要一个地图，由于我们要识别某些东西，我们需要向 `IdentifyTask` 对象的构造函数提供地图服务 URL。

跟随大多数 `dijit` 构造函数的趋势，我们的 `dijit` 构造函数将接受一个选项对象，以及一个 HTML DOM 节点或该节点的字符串 `id`。在选项对象中，我们将寻找一个映射和一个 `mapService` URL。构造函数中的第二个参数将被分配为 dijit 的 `domNode`，如果它是一个字符串，则根据该字符串找到相应的节点。

对于我们的构造函数，我们将地图集成到小部件中，并将 `mapService` 转换为 `IdentifyTask`。我们还将添加 `dojo/dom` 模块，以提供对常见 JavaScript 操作 `document.getElementById()` 的快捷方式，并使用它将字符串 `id` 转换为 DOM 元素：

```py
define([…, "dojo/dom", "esri/tasks/IdentifyTask", …], 
function (…, dom, IdentifyTask, …) {
  …
  constructor: function (options, srcRefNode) {

      if (typeof srcRefNode === "string") {
        srcRefNode = dom.byId(srcRefNode)
      }

      This.identifyTask = new IdentifyTask(options.mapService);
      this.map = options.map || null;
      this.domNode = srcRefNode;
    },
  …
});
```

### 重复使用我们的旧代码

回顾 第一章，*您的第一个映射应用程序*，我们能够查看地图，点击它，并获得一些结果。如果我们可以重用一些代码会怎么样？好吧，通过一些修改，我们可以重用大部分的代码。

我们需要解决的第一件事是分配我们的地图点击事件。由于我们并不总是有机会这样做，因此最合理的时间是在我们的 dijit 可见时分配地图点击事件。我们将修改我们的显示和隐藏函数，以便它们分别分配和移除点击处理程序。我们将命名地图点击事件处理程序为 `_onMapClick()`。我们还将加载模块 `dojo/_base/lang`，它帮助我们处理对象。我们将使用 `lang.hitch()` 函数重新分配函数的 `this` 语句：

```py
define([…, "dojo/on", "dojo/_base/lang", …], 
function (…, dojoOn, lang, …) {
  …
    show: function () {
      domStyle.set(this.domNode, "display", "block");
      this._mapClickHandler = this.map.on("click", lang.hitch(this, this._onMapClick));
    },

    hide: function () {
      domStyle.set(this.domNode, "display", "none");
      if (this._mapClickHandler && this._mapClickHandler.remove) {
 this._mapClickHandler.remove();
 }
    },

    _mapOnClick: function () {}
  …
```

作为旁注，虽然 JavaScript 不支持私有变量，但 JavaScript 对象的典型命名约定表明，如果一个属性或方法以下划线 (`_`) 字符开头，它被认为是开发者认为的私有。

在我们的`_onMapClick()`方法中，我们将重用第一章中“您的第一个映射应用程序”的点击事件代码，但有几点值得注意。现在请记住，我们不是将地图作为变量来引用，而是将其作为小部件的属性。对于`IdentifyTask`以及我们可能在这个 dijit 中调用的任何其他方法也是如此。要引用方法内的属性和方法，变量前面必须加上`this`。在这种情况下，`this`将引用小部件，我们在调用`_onMapClick()`时使用了`dojo/_base/lang`库来确保这一点。如果你的应用程序在地图点击事件失败，可能是因为你没有正确地分配带有正确`this`上下文的变量：

```py
define([…, "esri/tasks/IdentifyParameters", …], 
function (…,IdentifyParameters, …) {
  …
    _onMapClick: function (event) {
      var params = new IdentifyParameters(),
        defResults;

      params.geometry = event.mapPoint;
      params.layerOption = IdentifyParameters.LAYER_OPTION_ALL;
      params.mapExtent = this.map.extent;
      params.returnGeometry = true;
      params.width = this.map.width;
      params.height= this.map.height;
      params.spatialReference = this.map.spatialReference;
      params.tolerance = 3;

      this.map.graphics.clear();
      defResults = this.identifyTask.execute(params).addCallback (lang.hitch(this, this._onIdentifyComplete));
      this.map.infoWindow.setFeatures([defResults]);
      this.map.infoWindow.show(event.mapPoint);
    },
  …
```

### 加载更多模板

Y2K 协会要求我们以更有组织和逻辑的方式显示弹出窗口。在我们的网络开发者心中，没有什么比表格更能体现组织良好的数据了。我们可以为我们的模板设置和组织表格，但创建那些用于组织长列表的长字符串会令人望而却步。在我们的 JavaScript 代码中包含 HTML 字符串将会很痛苦，因为大多数 IDE 和语法高亮文本编辑器都不会在我们的 JavaScript 中高亮显示 HTML。

但然后我们记得我们可以使用`dojo/text`模块将 HTML 加载到我们的 JavaScript 中。如果我们使用小的 HTML 片段定义我们的`InfoTemplates`内容，并使用`dojo/text`加载它们，这个过程将会更加流畅：

```py
define([…, 
  "dojo/_base/array", 
 "esri/InfoTemplate",
 "dojo/text!./templates/StateCensus.html",
 "dojo/text!./templates/CountyCensus.html",
 "dojo/text!./templates/BlockGroupCensus.html",
 "dojo/text!./templates/BlockCensus.html"
], 
function (…,
  arrayUtils, InfoTemplate, StateTemplate, CountyTemplate, BlockGroupTemplate, BlockTemplate) {

    _onIdentifyComplete: function (results) {

      return arrayUtils.map(results, function (result) {
        var feature = result.feature,
          title = result.layerName,
          content;

        switch(title) {
          case "Census Block Points":
            content = BlockTemplate;
            break;
          case "Census Block Group":
            content = BlockGroupTemplate;
            break;
          case "Detailed Counties":
            content = CountyTemplate;
            break;
          case "states":
            content = StateTemplate;
            break;
          default:
            content = "${*}";
        }

        feature.infoTemplate = new InfoTemplate(title, content);
        return feature;
      });
    }
  });
});
```

我会将 HTML `infoTemplates`的内容留给你作为练习。

### 回到我们的 app.js

我们没有忘记我们的`app.js`文件。我们为加载地图和`Census` dijit 编写了代码框架，但没有分配任何其他内容。我们需要为`Census`按钮分配一个点击事件处理程序来切换`Census`小部件的可见性。为此，我们将使用`dijit/registry`模块从其`id`加载`dijit`按钮。我们将加载地图和`Census` dijit 的`mapService`，并添加一个事件监听器到`Census`按钮的点击事件，这将显示`Census`小部件。我们将再次使用`dojo/_base/lang`模块来确保`Census.show()`函数被正确地应用于`Census` dijit：

```py
require([
  …
 "dojo/_base/lang",
 "dijit/registry",
   …
], function(
   … lang, registry, …
) {
  …
  var census = new Census({
    map: map,
 mapService: "http://sampleserver6.arcgisonline.com/arcgis/rest/services/Census/MapServer/"
  }, "census-widget");

  var censusBtn = registry.byId("census-btn");

 censusBtn.on("click", lang.hitch(census, census.show));

});
```

当我们运行我们的代码时，它应该看起来像以下这样：

![回到我们的 app.js](img/6459OT_03_05.jpg)

# 摘要

在本章的整个过程中，我们学习了 ArcGIS API for JavaScript 中包含的 Dojo 框架提供的许多功能。我们了解了 Dojo 的三个主要包，以及如何使用它们来创建应用程序。我们学习了如何创建自己的模块，并且修改了一个现有的应用程序来创建一个可以被任何应用程序导入的模块。

在下一章中，我们将学习 ArcGIS API for JavaScript 如何通过 REST 服务与 ArcGIS Server 通信。
