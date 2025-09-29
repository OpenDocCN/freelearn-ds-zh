# 第二章：选择之道

在本章中，我们将涵盖：

+   选择单个元素

+   选择多个元素

+   遍历选择集

+   执行子选择

+   函数链式调用

+   操作原始选择集

# 简介

在使用 D3 的任何数据可视化项目中，你需要执行的最基本任务之一就是选择。选择可以帮助你定位页面上的某些视觉元素。如果你已经熟悉 W3C 标准化的 CSS 选择器或由流行的 JavaScript 库（如 jQuery 和 Zepto.js）提供的其他类似选择器 API，那么你将发现自己在 D3 的选择器 API 中如鱼得水。如果你之前没有使用选择器 API，请不要担心，本章旨在通过一些非常直观的食谱分步骤介绍这个主题；它将涵盖你数据可视化需求的所有常见用例。

**引入选择集**：选择器支持已被 W3C 标准化，因此所有现代网络浏览器都内置了对选择器 API 的支持。然而，当涉及到网络开发时，尤其是在数据可视化领域，基本的 W3C 选择器 API 存在局限性。标准的 W3C 选择器 API 只提供选择器，而不提供选择集。这意味着选择器 API 帮助你在文档中选择元素，但是，要操作所选元素，你仍然需要遍历每个元素，以便操作所选元素。考虑以下使用标准选择器 API 的代码片段：

```py
var i = document.querySelectorAll("p").iterator();
var e;
while(e = i.next()){
  // do something with each element selected
  console.log(e);
}
```

上述代码实际上选择了文档中的所有 `<p>` 元素，然后遍历每个元素以执行某些任务。这显然会很快变得繁琐，尤其是在你需要在页面上不断操作许多不同元素时，这是我们通常在数据可视化项目中做的事情。这就是为什么 D3 引入了它自己的选择器 API，使得开发工作不再那么枯燥。在本章的剩余部分，我们将介绍 D3 的选择器 API 的工作原理以及一些其强大的功能。

**CSS3 选择器基础**：在我们深入 D3 的选择器 API 之前，需要对 W3C 第 3 级选择器 API 进行一些基本介绍。如果你已经熟悉 CSS3 选择器，请随意跳过本节。D3 的选择器 API 是基于第 3 级选择器构建的，或者更常见的是 CSS3 选择器支持。在本节中，我们计划介绍一些必须了解 D3 选择器 API 的最常见 CSS3 选择器语法。

+   `#foo`：选择 `id` 值为 `foo` 的元素

    ```py
    <div id="foo">
    ```

+   `foo`：选择元素 `foo`

    ```py
    <foo>
    ```

+   `.foo`：选择 `class` 值为 `foo` 的元素

    ```py
    <div class="foo">
    ```

+   `[foo=goo]`：选择具有 `foo` 属性值并将其设置为 `goo` 的元素

    ```py
    <div foo="goo">
    ```

+   `foo goo`：选择 `foo` 元素内部的 `goo` 元素

    ```py
    <foo><goo></foo>
    ```

+   `foo#goo`：选择 `id` 值为 `goo` 的 `foo` 元素

    ```py
    <foo id="goo">
    ```

+   `foo.goo`：选择 `class` 值为 `goo` 的 `foo` 元素

    ```py
    <foo class="goo">
    ```

+   `foo:first-child`：选择 `foo` 元素的第一个子元素

    ```py
    <foo> // <-- this one
    <foo>
    <foo> 
    ```

+   `foo:nth-child(n)`: 选择`foo`元素的第 n 个子元素

    ```py
    <foo>
    <foo> // <-- foo:nth-child(2)
    <foo> // <-- foo:nth-child(3)
    ```

CSS3 选择器是一个相当复杂的话题。在这里，我们只列出了一些你需要理解和掌握的、在用 D3 工作时最常用的选择器。有关这个主题的更多信息，请访问 W3C 的 3 级选择器 API 文档[`www.w3.org/TR/css3-selectors/`](http://www.w3.org/TR/css3-selectors/)。

### 小贴士

如果你正在针对不支持选择器的旧浏览器，你可以在 D3 之前包含 Sizzle 以实现向后兼容性。你可以在[`sizzlejs.com/`](http://sizzlejs.com/)找到 Sizzle。

目前，下一代选择器 API 的 4 级在 W3C 处于草案阶段。你可以在[`dev.w3.org/csswg/selectors4/`](http://dev.w3.org/csswg/selectors4/)查看它提供的内容及其当前草案。

主要浏览器厂商已经开始实现一些 4 级选择器，如果你想知道浏览器支持的程度，可以尝试这个实用的网站[`css4-selectors.com/browser-selector-test/`](http://css4-selectors.com/browser-selector-test/)。

# 选择单个元素

有时，你可能需要在一个页面上选择单个元素以执行一些视觉操作。这个示例将向你展示如何使用 CSS 选择器在 D3 中执行有针对性的单个元素选择。

## 准备工作

在你的网络浏览器中打开以下文件的本地副本：

[`github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/single-selection.html`](https://github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/single-selection.html)

## 如何操作...

让我们选择一些内容（比如一个段落元素）并在屏幕上生成经典的“hello world”。

```py
<p id="target"></p> <!-- A -->

<script type="text/javascript">
 d3.select("p#target") // <-- B
 .text("Hello world!"); // <-- C
</script>
```

这个示例简单地在你的屏幕上显示一个**Hello world!**。

## 工作原理...

`d3.select`命令用于在 D3 中执行单个元素选择。此方法接受一个表示有效 CSS3 选择器的字符串，或者如果你已经有了要选择的元素的引用，则可以接受一个元素对象。`d3.select`命令返回一个 D3 选择对象，你可以在此对象上链式调用修改函数来操作此元素的属性、内容或内部 HTML。

### 小贴士

使用提供的选择器可以选择多个元素，但只返回选择集中的第一个元素。

在这个例子中，我们在 B 行选择具有`target`作为`id`值的段落元素，然后在 C 行将其文本内容设置为`Hello world!`。所有 D3 选择都支持一组标准修改函数。我们在这里展示的`text`函数就是其中之一。以下是在本书中你将遇到的一些最常见的修改函数：

+   `selection.attr`函数：此函数允许你检索或修改所选元素上的给定属性。

    ```py
    // set foo attribute to goo on p element
    d3.select("p").attr("foo", "goo"); 
    // get foo attribute on p element
    d3.select("p").attr("foo");
    ```

+   `selection.classed`函数：此函数允许你在所选元素上添加或删除 CSS 类。

    ```py
    // test to see if p element has CSS class goo
    d3.select("p").classed("goo");
    // add CSS class goo to p element
    d3.select("p").classed("goo", true);
    // remove CSS class goo from p element. classed function
    // also accepts a function as the value so the decision 
    // of adding and removing can be made dynamically
    d3.select("p").classed("goo", function(){return false;});
    ```

+   `selection.style` 函数：此函数允许你将具有特定名称的 CSS 样式设置为选定的元素（们）的特定值。

    ```py
    // get p element's style for font-size
    d3.select("p").style("font-size");
    // set font-size style for p to 10px
    d3.select("p").style("font-size", "10px");
    // set font-size style for p to the result of some 
    // calculation. style function also accepts a function as // the value can be produced dynamically
    d3.select("p"). style("font-size", function(){return normalFontSize + 10;});
    ```

+   `selection.text` 函数：此函数允许你访问和设置选定的元素（们）的文本内容。

    ```py
    // get p element's text content
    d3.select("p").text();
    // set p element's text content to "Hello"
    d3.select("p").text("Hello");
    // text function also accepts a function as the value, 
    // thus allowing setting text content to some dynamically 
    // produced message
    d3.select("p").text(function(){
      var model = retrieveModel();
      return model.message;
    });
    ```

+   `selection.html` 函数：此函数允许你修改元素的内部 HTML 内容。

    ```py
    // get p element's inner html content
    d3.select("p").html();
    // set p element's inner html content to "<b>Hello</b>"
    d3.select("p").text("<b>Hello</b>");
    // html function also accepts a function as the value, 
    // thus allowing setting html content to some dynamically 
    // produced message
    d3.select("p").text(function(){
      var template = compileTemplate();
      return template();
    });
    ```

    这些修饰函数适用于单元素和多元素选择结果。当应用于多元素选择时，这些修改将应用于每个选定的元素。我们将在本章剩余部分的其他更复杂的食谱中看到它们的应用。

    ### 注意

    当一个函数在这些修饰函数中用作值时，实际上有一些内置参数被传递给这些函数，以启用数据驱动的计算。这种数据驱动的方法赋予了 D3 其力量和其名称（数据驱动文档），将在下一章中详细讨论。

# 选择多个元素

通常选择单个元素是不够的，你更希望同时对页面上的元素集应用某些更改。在这个食谱中，我们将使用 D3 多元素选择器和其选择 API 进行操作。

## 准备工作

在你的网络浏览器中打开以下文件的本地副本：

[`github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/multiple-selection.html`](https://github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/multiple-selection.html)

## 如何做到这一点...

这正是 `d3.selectAll` 函数的设计目的。在这个代码片段中，我们将选择三个不同的 `div` 元素，并使用一些 CSS 类增强它们。

```py
<div></div>
<div></div>
<div></div>

<script type="text/javascript">
 d3.selectAll("div") // <-- A
 .attr("class", "red box"); // <-- B
</script>
```

这段代码片段生成了以下视觉效果：

![如何做到这一点...](img/2162OS_02_01.jpg)

多元素选择

## 它是如何工作的...

在这个例子中，你可能会首先注意到 D3 选择 API 的使用与单元素版本是多么相似。这是 D3 选择 API 强大的设计选择之一。无论你针对多少个元素，无论是单个还是多个，修饰函数都是一样的。我们之前章节中提到的所有修饰函数都可以直接应用于多元素选择，换句话说，D3 选择是基于集合的。

现在既然已经说了这些，让我们更仔细地看看本节中展示的代码示例，尽管它通常很简单且具有自我描述性。在第 A 行，使用了`d3.selectAll`函数来选择页面上的所有`div`元素。这个函数调用的返回值是一个包含所有三个`div`元素的 D3 选择对象。紧接着，在第 B 行，对这个选择对象调用了`attr`函数，将所有三个`div`元素的`class`属性设置为`red box`。正如这个例子所示，选择和操作代码非常通用，如果现在页面上有超过三个`div`元素，代码也不会改变。这看起来现在似乎是一个微不足道的便利，但在后面的章节中，我们将展示这种便利如何使您的可视化代码更简单、更容易维护。

# 迭代选择

有时能够迭代选择中的每个元素并根据它们的位置不同地修改每个元素是非常方便的。在这个菜谱中，我们将向您展示如何使用 D3 选择迭代 API 实现这一点。

## 准备工作

在您的网络浏览器中打开以下文件的本地副本：

[`github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/selection-iteration.html`](https://github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/selection-iteration.html)

## 如何操作...

D3 选择对象提供了一个简单的迭代接口，以类似于迭代 JavaScript 数组的方式执行迭代。在这个例子中，我们将迭代我们在前一个菜谱中使用的三个选中的`div`元素，并用索引号标注它们。

```py
<div></div>
<div></div>
<div></div>

<script type="text/javascript">
d3.selectAll("div") // <-- A
 .attr("class", "red box") // <-- B
 .each(function (d, i) { // <-- C
 d3.select(this).append("h1").text(i); // <-- D
 });
</script>
```

### 小贴士

选择本质上是一种数组，尽管有一些增强。我们将在后面的章节中探索原始选择及其数组形式，以及如何处理它。

前面的代码片段产生了以下视觉效果：

![如何操作...](img/2162OS_02_02.jpg)

选择迭代

## 它是如何工作的...

这个例子是在我们之前章节中看到的内容的基础上构建的。除了在第 A 行选择页面上的所有`div`元素并在第 B 行设置它们的类属性之外，在这个例子中我们还对选择调用了`each`函数，以展示您如何迭代一个多元素选择并分别处理每个元素。

### 注意

这种在另一个函数的返回值上调用函数的形式被称为**函数链式调用**。如果您不熟悉这种调用模式，请参阅第一章 *使用 D3.js 入门*，其中解释了该主题。

**选择器.each(function) 函数**：`each` 函数接受一个迭代器函数作为其参数。给定的迭代器函数可以接收两个可选参数 `d` 和 `i`，以及一个作为 `this` 引用传递的隐藏参数，该引用指向当前 DOM 元素对象。第一个参数 `d` 代表绑定到该特定元素的数值（如果您觉得这很困惑，不要担心，我们将在下一章深入讲解数据绑定）。第二个参数 `i` 是正在迭代的当前元素对象的索引号。这个索引是从零开始的，意味着它从零开始，每次遇到新元素时增加。

**选择器.append(name) 函数**：在本例中引入的另一个新函数是 `append` 函数。该函数创建一个具有给定名称的新元素，并将其追加到当前选择中每个元素的最后一个子元素。它返回一个包含新追加元素的新选择。现在，有了这些知识，让我们更仔细地看看本例中的代码示例。

```py
d3.selectAll("div") // <-- A
    .attr("class", "red box") // <-- B
    .each(function (d, i) { // <-- C
        d3.select(this).append("h1").text(i); // <-- D
    });
```

迭代器函数定义在第 C 行，包含 `d` 和 `i` 参数。第 D 行稍微有趣一些。在第 D 行的开始处，`this` 引用被 `d3.select` 函数包裹。这种包裹实际上产生了一个包含当前 DOM 元素的单个元素选择。一旦被包裹，就可以在 `d3.select(this)` 上使用标准的 D3 选择操作 API。之后，在当前元素选择上调用 `append("h1")` 函数，将新创建的 `h1` 元素追加到当前元素。然后它简单地设置这个新创建的 `h1` 元素的文本内容为当前元素的索引号。这产生了如图所示编号框的视觉效果。再次提醒，索引从 0 开始，每次遇到新元素时增加 1。

### 小贴士

DOM 元素对象本身具有非常丰富的接口。如果您想了解在迭代器函数中它能做什么，请参阅 [`developer.mozilla.org/en-US/docs/DOM/element`](https://developer.mozilla.org/en-US/docs/DOM/element) 的 DOM 元素 API。

# 执行子选择

在处理可视化时，执行范围选择是很常见的。例如，选择特定 `section` 元素内的所有 `div` 元素是这种范围选择的一个用例。在本例中，我们将展示如何通过不同的方法和它们的优缺点来实现这一点。

## 准备工作

在您的网页浏览器中打开以下文件的本地副本：

[`github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/sub-selection.html`](https://github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/sub-selection.html)

## 如何实现...

以下代码示例使用 D3 支持的两种不同的子选择样式选择了两个不同的 `div` 元素。

```py
<section id="section1">
    <div>
        <p>blue box</p>
    </div>
</section>
<section id="section2">
    <div>
        <p>red box</p>
    </div>
</section>

<script type="text/javascript">
 d3.select("#section1 > div") // <-- A
            .attr("class", "blue box");

 d3.select("#section2") // <-- B
 .select("div") // <-- C
            .attr("class", "red box"); 
</script>
```

这段代码生成了以下视觉输出：

![如何做到这一点...](img/2162OS_02_03.jpg)

子选择

## 它是如何工作的...

虽然产生相同的效果，但这个例子展示了两种非常不同的子选择技术。我们将分别在这里讨论它们，以便您了解它们的优缺点以及何时使用一种而不是另一种。

**选择器第三级组合器**：在行 A 中，`d3.select` 使用了一个看起来特殊的字符串，该字符串由一个标签名通过大于号（U+003E，>）连接到另一个标签名。这种语法被称为**组合器**（这里的大于号表示它是一个子组合器）。第三级选择器支持几种不同的结构组合器。在这里，我们将快速介绍其中最常见的一些。

**后代组合器**：这个组合器的语法类似于`selector selector`。

如其名所示，后代组合器用于描述两个选择器之间松散的父子关系。之所以称为松散的父子关系，是因为后代组合器不关心第二个选择器是否是父选择器的子、孙子或曾孙。让我们通过一些例子来说明这种松散关系概念。

```py
<div>
<span>
The quick <em>red</em> fox jumps over the lazy brown dog
   </span>
</div>
```

使用以下选择器：

```py
div em
```

它将选择`em`元素，因为`div`是`em`元素的祖先，而`em`是`div`元素的子代。

**子组合器**：这个组合器的语法类似于`selector > selector`。

子组合器提供了一种更严格的方式来描述两个元素之间的父子关系。子组合器是通过使用大于号（U+003E，>）字符分隔两个选择器来定义的。

以下选择器：

```py
span > em
```

它将选择`em`元素，因为在我们例子中`em`是`span`元素的直接子代。而选择器`div > em`将不会产生任何有效选择，因为`em`不是`div`元素的直接子代。

### 注意

第三级选择器也支持兄弟组合器，但由于它不太常见，我们在这里不进行介绍；感兴趣的读者可以参考 W3C 第三级选择器文档[`www.w3.org/TR/css3-selectors/#sibling-combinators`](http://www.w3.org/TR/css3-selectors/#sibling-combinators)。

W3C 第四级选择器提供了一些有趣的附加组合器，即跟随兄弟和引用组合器，这些组合器可以提供一些非常强大的目标选择能力；更多详情请参阅[`dev.w3.org/csswg/selectors4/#combinators`](http://dev.w3.org/csswg/selectors4/#combinators)。

**D3 子选择**：在行 B 和 C 上，使用了不同类型的子选择技术。在这种情况下，首先在行 B 上对`section #section2`元素进行了简单的 D3 选择。紧接着，另一个`select`被链式调用，以选择行 C 上的`div`元素。这种链式选择定义了一个范围选择。用简单的话说，这基本上意味着选择一个嵌套在`#section2`下的`div`元素。在语义上，这本质上类似于使用后代组合器`#section2 div`。然而，这种子选择形式的优势在于，由于父元素是单独选择的，因此它允许你在选择子元素之前处理父元素。为了演示这一点，让我们看一下以下代码片段：

```py
d3.select("#section2") // <-- B
    .style("font-size", "2em") // <-- B-1
    .select("div") // <-- C
    .attr("class", "red box");
```

如前述代码片段所示，现在在我们选择`div`元素之前，我们可以在行 B-1 上对`#section2`应用一个修改器函数。这种灵活性将在下一节进一步探讨。

# 函数链

如我们所见，D3 API 完全围绕函数链的概念设计。因此，它几乎形成了一个用于动态构建 HTML/SVG 元素的领域特定语言（DSL）。在这个代码示例中，我们将看看如何仅使用 D3 构建上一个示例的整个主体结构。

### 注意

如果领域特定语言（DSL）对你来说是一个新概念，我强烈推荐查看 Martin Fowler 在其书籍《领域特定语言》（*Domain-Specific Languages*）中的精彩解释摘录。摘录可以在[`www.informit.com/articles/article.aspx?p=1592379`](http://www.informit.com/articles/article.aspx?p=1592379)找到。

## 准备工作

在你的网络浏览器中打开以下文件的本地副本：

[`github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/function-chain.html`](https://github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/function-chain.html)

## 如何操作...

让我们看看如何使用函数链来生成简洁且易于阅读的代码，从而产生动态视觉内容。

```py
<script type="text/javascript">
 var body = d3.select("body"); // <-- A

 body.append("section") // <-- B
 .attr("id", "section1") // <-- C
 .append("div") // <-- D
 .attr("class", "blue box") // <-- E
 .append("p") // <-- F
 .text("dynamic blue box"); // <-- G

  body.append("section")
      .attr("id", "section2")
    .append("div")
      .attr("class", "red box")
    .append("p")
      .text("dynamic red box");
</script>
```

此代码生成以下视觉输出（与我们之前章节中看到的内容相似）：

![如何操作...](img/2162OS_02_04.jpg)

函数链

## 工作原理...

尽管与上一个示例在视觉上相似，但在这个示例中 DOM 元素的构建过程与上一个示例有显著不同。正如代码示例所示，与上一个食谱中存在的`section`和`div`元素不同，页面上没有静态 HTML 元素。

让我们仔细看看这些元素是如何动态创建的。在第 A 行，对顶级 `body` 元素进行了通用选择。使用名为 `body` 的局部变量缓存了 `body` 选择结果。然后在第 B 行，将一个新的 `section` 元素附加到 `body` 上。记住，`append` 函数返回一个包含新附加元素的新选择，因此在第 C 行，可以将新创建的 `section` 元素的 `id` 属性设置为 `section1`。之后在第 D 行，创建了一个新的 `div` 元素并将其附加到 `#section1` 上，其 CSS 类在第 E 行设置为 `blue box`。下一步，同样在第 F 行，将一个 `paragraph` 元素附加到 `div` 元素上，其文本内容在第 G 行设置为 `dynamic blue box`。

如此示例所示，此链式过程可以继续创建任意复杂度的结构。实际上，这就是通常基于 D3 的数据可视化结构是如何创建的。许多可视化项目仅包含一个 HTML 骨架，而依赖于 D3 创建其余部分。如果你想要高效地使用 D3 库，熟悉这种函数链式方法至关重要。

### 小贴士

一些 D3 的修改函数返回一个新的选择，例如 `select`、`append` 和 `insert` 函数。使用不同级别的缩进区分函数链应用于哪个选择是一个好习惯。

# 操作原始选择

有时，尽管不经常，访问 D3 原始选择数组在开发中可能有益，无论是用于调试目的还是与其他需要访问原始 DOM 元素的 JavaScript 库集成；在这个配方中，我们将向您展示如何做到这一点。我们还将看到 D3 选择对象的一些内部结构。

## 准备工作

在您的网络浏览器中打开以下文件的本地副本：

[`github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/raw-selection.html`](https://github.com/NickQiZhu/d3-cookbook/blob/master/src/chapter2/raw-selection.html)

## 如何操作...

当然，您可以通过使用 `nth-child` 选择器或通过 `each` 的选择器迭代函数来实现这一点，但在某些情况下，这些选项可能过于繁琐和不方便。这就是您可能会发现处理原始选择数组作为更方便的方法的时候。在这个例子中，我们将看到如何访问和利用原始选择数组。

```py
<table class="table">
    <thead>
    <tr>
        <th>Time</th>
        <th>Type</th>
        <th>Amount</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>10:22</td>
        <td>Purchase</td>
        <td>$10.00</td>
    </tr>
    <tr>
        <td>12:12</td>
        <td>Purchase</td>
        <td>$12.50</td>
    </tr>
    <tr>
        <td>14:11</td>
        <td>Expense</td>
        <td>$9.70</td>
    </tr>
    </tbody>
</table>

<script type="text/javascript">
 var rows = d3.selectAll("tr");// <-- A

 var headerElement = rows[0][0];// <-- B

 d3.select(headerElement).attr("class","table-header");// <--C

 d3.select(rows[0][1]).attr("class","table-row-odd"); //<-- D
 d3.select(rows[0][2]).attr("class","table-row-even"); //<-- E
 d3.select(rows[0][3]).attr("class","table-row-odd"); //<-- F
</script>
```

此配方生成以下视觉输出：

![如何操作...](img/2162OS_02_05.jpg)

原始选择操作

## 工作原理...

在这个例子中，我们通过现有的 HTML 表格来着色表格。这并不是一个很好的例子，说明您如何使用 D3 着色表格的奇数和偶数行。相反，这个例子旨在展示如何访问原始选择数组。

### 小贴士

在表格中为奇数和偶数行着色的一个更好的方法是使用 `each` 函数，然后依赖于索引参数 `i` 来完成这项工作。

在行 A 中，我们选择了所有行并将选择存储在 `rows` 变量中。D3 选择存储在一个二维 JavaScript 数组中。选中的元素存储在一个数组中，然后被包裹在一个单元素数组中。因此，为了访问第一个选中的元素，你需要使用 `rows[0][0]`，而第二个元素可以通过 `rows[0][1]` 访问。正如我们在行 B 中看到的，表头元素可以通过 `rows[0][0]` 访问，这将返回一个 DOM 元素对象。同样，正如我们在前面的章节中演示的那样，任何 DOM 元素都可以通过 `d3.select` 直接选择，如行 C 所示。行 D、E 和 F 展示了如何直接索引和访问选择中的每个元素。

在某些情况下，原始选择访问可能很有用；然而，由于它依赖于直接访问 D3 选择数组，它会在你的代码中创建一个结构依赖。换句话说，如果在 D3 的未来版本中这个结构发生了变化，那么依赖于它的代码将会被破坏。因此，除非绝对必要，否则建议避免使用原始选择操作。

### 小贴士

这种方法通常不是必要的，但在某些情况下可能会很有用，例如在你的单元测试用例中，当你需要快速知道每个元素的绝对索引并获得它们的引用时。我们将在后面的章节中更详细地介绍单元测试。
