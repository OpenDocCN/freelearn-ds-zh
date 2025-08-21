# 第三章：使用 Python 库加速数据分析

> “每一个愿景都是笑话，直到第一个人实现它；一旦实现，它就变得平凡。”

– *罗伯特·戈达德*

在本章中，我们将深入探讨 PixieApp 框架的技术细节。你将能够将以下信息既作为*入门教程*，也作为 PixieApp 编程模型的参考文档。

我们将从 PixieApp 的基本结构入手，然后再深入探讨其基础概念，如路由和请求。为了帮助大家跟进，我们将逐步构建一个*GitHub 跟踪*示例应用程序，应用介绍的功能和最佳实践，从构建数据分析开始，到将其集成到 PixieApp 中。

到本章结束时，你应该能够将学到的知识应用到自己的实际案例中，包括编写自己的 PixieApp。

# PixieApp 的结构

### 注意

**注意**：PixieApp 编程模型不要求有 JavaScript 的先验经验，但期望读者熟悉以下内容：

+   Python ([`www.python.org`](https://www.python.org))

+   HTML5 ([`www.w3schools.com/html`](https://www.w3schools.com/html))

+   CSS3 ([`www.w3schools.com/css`](https://www.w3schools.com/css))

**PixieApp**一词代表**Pixie 应用程序**，旨在强调其与 PixieDust 功能的紧密集成，特别是`display()` API。其主要目标是使开发者能够轻松构建可以调用 Jupyter Notebook 中实现的数据分析的用户界面。

一个 PixieApp 遵循**单页应用程序**（**SPA**）设计模式 ([`en.wikipedia.org/wiki/Single-page_application`](https://en.wikipedia.org/wiki/Single-page_application))，用户会看到一个欢迎页面，并根据用户的交互动态更新。更新可以是部分刷新，例如用户点击控件后更新图表，或是完全刷新，比如在多步骤过程中显示新页面。在每种情况下，更新都由服务器端的路由控制，路由通过特定机制触发，我们将在后面讨论。当触发时，路由会执行代码处理请求，然后返回一个 HTML 片段，该片段会应用到客户端的目标 DOM 元素上 ([`www.w3schools.com/js/js_htmldom.asp`](https://www.w3schools.com/js/js_htmldom.asp))。

以下时序图展示了在运行 PixieApp 时，客户端和服务器端是如何相互交互的：

![PixieApp 的结构](img/B09699_03_01.jpg)

显示 PixieApp 信息流的时序图

当 PixieApp 启动时（通过调用`run`方法），默认路由会被调用，并返回相应的 HTML 片段。当用户与应用交互时，会执行更多请求，触发关联的路由，并相应地刷新 UI。

从实现的角度来看，PixieApp 只是一个常规的 Python 类，已经使用了`@PixieApp`装饰器。在幕后，`PixieApp`装饰器为类添加了运行应用所需的方法和字段，例如 `run` 方法。

### 注意

更多关于 Python 装饰器的信息可以在这里找到：

[`wiki.python.org/moin/PythonDecorators`](https://wiki.python.org/moin/PythonDecorators)

为了开始，以下代码展示了一个简单的 *Hello World* PixieApp：

```py
#import the pixieapp decorators
from pixiedust.display.app import *

@PixieApp   #decorator for making the class a PixieApp
class HelloWorldApp():
    @route()  #decorator for making a method a route (no arguments means default route)
    def main_screen(self):
        return """<div>Hello World</div>"""

#Instantiate the application and run it
app = HelloWorldApp()
app.run()
```

### 注意

你可以在这里找到代码：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode1.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode1.py)

上述代码展示了 PixieApp 的结构，如何定义路由，以及如何实例化并运行应用。由于 PixieApps 是常规的 Python 类，因此它们可以继承其他类，包括其他 PixieApp，这对于大型项目来说，能够使代码模块化和可重用。

## 路由

路由用于动态更新客户端屏幕的全部或部分内容。可以通过在任何类方法上使用`@route`装饰器来轻松定义路由，依据以下规则：

+   路由方法需要返回一个字符串，表示用于更新的 HTML 片段。

    ### 注意

    **注意**：在片段中允许使用 CSS 和 JavaScript。

+   `@route`装饰器可以有一个或多个关键字参数，这些参数必须是字符串类型。可以将这些关键字参数视为请求参数，PixieApp 框架内部使用它们将请求分发到最匹配的路由，依据以下规则：

    +   参数最多的路由总是会被优先评估。

    +   所有参数必须匹配，才能选择一个路由。

    +   如果未找到路由，则会选择默认路由作为备用。

    +   路由可以使用通配符`*`进行配置，在这种情况下，任何状态参数的值都会匹配。

        以下是一个示例：

        ```py
               @route(state1="value1", state2="value2")
        ```

+   PixieApp 必须有且仅有一个默认路由，即没有参数的路由，即`@route()`。

配置路由时避免冲突非常重要，特别是当你的应用有层次化状态时。例如，关联 `state1="load"` 的路由可能负责加载数据，然后关联 `(state1="load", state2="graph")` 的第二个路由可能负责绘制数据。在这种情况下，带有 `state1` 和 `state2` 两个指定参数的请求将匹配第二个路由，因为路由评估是从最具体到最不具体的顺序进行的，直到找到第一个匹配的路由为止。

为了更清楚，以下图示展示了请求与路由是如何匹配的：

![Routes](img/B09699_03_02.jpg)

请求与路由的匹配

定义为路由的方法的预期合同是返回一个 HTML 片段，其中可以包含 Jinja2 模板构造。Jinja2 是一个强大的 Python 模板引擎，提供了一套丰富的功能来动态生成文本，包括访问 Python 变量、方法和控制结构，如 `if...else`、`for` 循环等。覆盖所有功能超出了本书的范围，但我们将讨论一些常用的重要构造：

### 注意

**注意**：如果你想了解更多关于 Jinja2 的内容，可以在这里阅读完整文档：

[`jinja.pocoo.org/docs/templates`](http://jinja.pocoo.org/docs/templates)

+   **变量**：你可以使用双花括号来访问作用域中的变量，例如，`"<div>这是我的变量 {{my_var}}</div>"`。在渲染时，`my_var` 变量将被其实际值替换。你还可以使用 `.`（点）符号来访问复杂对象，例如，`"<div>这是一个嵌套的值 {{my_var.sub_value}}</div>"`。

+   **for 循环**：你可以使用 `{%for ...%}...{%endfor%}` 语法来通过迭代一系列项目（如列表、元组、字典等）动态生成文本，如下例所示：

    ```py
    {%for message in messages%}
    <li>{{message}}</li>
    {%endfor%}
    ```

+   **if 语句**：你可以使用 `{%if ...%}...{%elif ...%}...{%else%}…{%endif%}` 语法来有条件地输出文本，如下例所示：

    ```py
    {%if status.error%}
    <div class="error">{{status.error}}</div>
    {%elif status.warning%}
    <div class="warning">{{status.warning}}</div>
    {%else%}
    <div class="ok">{{status.message}}</div>
    {%endif%}
    ```

了解变量和方法如何进入 Jinja2 模板字符串的作用域也非常重要。PixieApp 会自动提供三种类型的变量和方法供你访问：

+   **类变量和方法**：可以使用 `this` 关键字访问这些内容。

    ### 注意

    **注意**：我们没有使用更符合 Python 风格的 `self` 关键字，因为不幸的是，Jinja2 本身已经占用了这个关键字。

+   **方法参数**：当路由参数使用 `*` 值，并且你希望在运行时访问该值时，这个功能非常有用。在这种情况下，你可以在方法中添加与路由参数中定义的名称相同的参数，PixieApp 框架会自动传递正确的值。

    ### 注意

    **注意**：参数的顺序实际上无关紧要。你也不必使用路由中定义的每个参数，这样如果你只关心使用其中的一部分参数会更加方便。

    该变量也会在 Jinja2 模板字符串的作用域内，如下例所示：

    ```py
    @route(state1="*", state2="*")
    def my_method(self, state1, state2):
        return "<div>State1 is {{state1}}. State2 is {{state2}}</div>"
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode2.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode2.py)

+   **方法的局部变量**：PixieApp 将自动把在方法中定义的所有局部变量放入 Jinja2 模板字符串的作用域中，前提是你在方法中添加了`@templateArgs`装饰器，如下例所示：

    ```py
    @route()
    @templateArgs
    def main_screen(self):
        var1 = self.compute_something()
        var2 = self.compute_something_else()
        return "<div>var1 is {{var1}}. var2 is {{var2}}</div>"
    ```

    ### 注意

    你可以在这里找到代码：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode3.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode3.py)

## 生成请求到路由

如前所述，PixieApp 遵循 SPA 设计模式。在第一次加载屏幕后，所有与服务器的后续交互都是通过动态请求进行的，而不是像多页面 Web 应用程序那样使用 URL 链接。有三种方法可以生成到路由的内核请求：

+   使用`pd_options`自定义属性定义要传递给服务器的状态列表，如下例所示：

    ```py
    pd_options="state1=value1;state2=value2;..;staten=valuen"
    ```

+   如果你已经有一个包含`pd_options`值的 JSON 对象——例如在调用`display()`时——你需要将其转换为`pd_options` HTML 属性所期望的格式，这可能会非常耗时。在这种情况下，更方便的方法是将`pd_options`指定为子元素，这样就能直接将选项作为 JSON 对象传递（避免了转换数据的额外工作），如下例所示：

    ```py
    <div>
        <pd_options>
            {"state1":"value1","state2":"value2",...,
            "staten":"valuen"}
        </pd_options>
    </div>
    ```

+   通过调用`invoke_route`方法以编程方式，如下例所示：

    ```py
    self.invoke_route(self.route_method, state1='value1', state2='value2')
    ```

### 注意

**注意**：记得使用`this`，而不是`self`，如果你是从 Jinja2 模板字符串中调用此方法，因为`self`已经被 Jinja2 本身使用。

当`pd_options`中传递的状态值需要根据用户选择动态计算时，你需要使用`$val(arg)`特殊指令，该指令作为宏，在内核请求执行时解析。

`$val(arg)`指令接受一个参数，该参数可以是以下之一：

+   页面上 HTML 元素的 ID，例如输入框或组合框，如下例所示：

    ```py
    <div>
        <pd_options>
            {"state1":"$val(my_element_id)","state2":"value2"}
        <pd_options>
    </div>
    ```

+   一个必须返回期望值的 JavaScript 函数，如下例所示：

    ```py
    <script>
        function resValue(){
                return "my_query";
        }
    </script>
    ...
    <div pd_options="state1=$val(resValue)"></div>
    ```

### 注意

**注意**：使用`$val`指令的动态值被大多数 PixieDust 自定义属性支持。

## 一个 GitHub 项目追踪示例应用程序

让我们将迄今为止学到的内容应用到实现示例应用程序中。为了试验，我们将使用 GitHub Rest API ([`developer.github.com/v3`](https://developer.github.com/v3)) 来搜索项目并将结果加载到 pandas DataFrame 中进行分析。

初始代码显示欢迎界面，包含一个简单的输入框用于输入 GitHub 查询，并有一个按钮提交请求：

```py
from pixiedust.display.app import *

@PixieApp
class GitHubTracking():
    @route()
    def main_screen(self):
        return """
<style>
    div.outer-wrapper {
        display: table;width:100%;height:300px;
    }
    div.inner-wrapper {
        display: table-cell;vertical-align: middle;height: 100%;width: 100%;
    }
</style>
<div class="outer-wrapper">
    <div class="inner-wrapper">
        <div class="col-sm-3"></div>
        <div class="input-group col-sm-6">
            <input id="query{{prefix}}" type="text" class="form-control" placeholder="Search projects on GitHub">
            <span class="input-group-btn">
                <button class="btn btn-default" type="button">Submit Query</button>
            </span>
        </div>
    </div>
</div>
"""

app = GitHubTracking()
app.run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode4.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode4.py)

从上面的代码中有几点需要注意：

+   Bootstrap CSS 框架 ([`getbootstrap.com/docs/3.3`](https://getbootstrap.com/docs/3.3)) 和 jQuery JS 框架 ([`jquery.com`](https://jquery.com)) 已由 Jupyter Notebook 提供。我们可以直接在代码中使用它们，无需安装。

+   Font Awesome 图标 ([`fontawesome.com`](https://fontawesome.com)) 在 Notebook 中默认可用。

+   PixieApp 代码可以在 Notebook 的多个单元格中执行。由于我们依赖于 DOM 元素 ID，因此确保两个元素没有相同的 ID 是很重要的，否则会导致不希望出现的副作用。为此，建议始终包括由 PixieDust 框架提供的唯一标识符`{{prefix}}`，例如 `"query{{prefix}}"`。

结果显示在以下截图中：

![A GitHub project tracking sample application](img/B09699_03_03.jpg)

我们 GitHub 跟踪应用的欢迎界面

下一步是创建一个新路由，接收用户输入并返回结果。此路由将通过 **提交查询** 按钮调用。

为了简化，以下代码没有使用与 GitHub 交互的 Python 库，如 PyGithub ([`pygithub.readthedocs.io/en/latest`](http://pygithub.readthedocs.io/en/latest))，而是直接调用 GitHub 网站中文档化的 REST API：

### 注意

**注意**：当你看到以下符号 `[[GitHubTracking]]` 时，表示该代码应添加到 `GitHubTracking` PixieApp 类中，为了避免重复代码，已省略周围的代码。在有疑问时，你可以参考本节末尾指定的完整 Notebook。

```py
import requests
import pandas
[[GitHubTracking]]
@route(query="*")
@templateArgs
def do_search(self, query):
    response = requests.get( "https://api.github.com/search/repositories?q={}".format(query))
    frames = [pandas.DataFrame(response.json()['items'])]
    while response.ok and "next" in response.links:
        response = requests.get(response.links['next']['url'])
        frames.append(pandas.DataFrame(response.json()['items']))

    pdf = pandas.concat(frames)
    response = requests.get( "https://api.github.com/search/repositories?q={}".format(query))
    if not response.ok:
        return "<div>An Error occurred: {{response.text}}</div>"
    return """<h1><center>{{pdf|length}} repositories were found</center></h1>"""
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode5.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode5.py)

在上述代码中，我们创建了一个名为`do_search`的路由，它接受一个名为`query`的参数，我们用这个参数来构建 GitHub 的 API URL。使用`requests` Python 模块（[`docs.python-requests.org`](http://docs.python-requests.org)）向这个 URL 发出 GET 请求，我们得到一个 JSON 数据包，并将其转换为一个 pandas DataFrame。根据 GitHub 文档，搜索 API 会进行分页，下一页的链接会存储在响应头中。代码使用`while`循环遍历每个链接，并将下一页加载到新的 DataFrame 中。然后，我们将所有的 DataFrame 合并为一个名为`pdf`的 DataFrame。接下来，我们只需构建一个 HTML 片段来显示结果。该片段使用 Jinja2 语法`{{...}}`来访问定义为局部变量的`pdf`，这只因为我们在`do_search`方法中使用了`@templateArgs`装饰器。请注意，我们还使用了一个 Jinja2 过滤器`length`来显示找到的仓库数量：`{{pdf|length}}`。

### 注意

有关过滤器的更多信息，请访问以下链接：

[`jinja.pocoo.org/docs/templates/#filters`](http://jinja.pocoo.org/docs/templates/#filters)

当用户点击**提交查询**按钮时，我们仍然需要调用`do_search`路由。为此，我们在`<button>`元素中添加`pd_options`属性，如下所示：

```py
<div class="input-group col-sm-6">
    <input id="query{{prefix}}" type="text"
     class="form-control"
     placeholder="Search projects on GitHub">
    <span class="input-group-btn">
        <button class="btn btn-default" type="button" pd_options="query=$val(query{{prefix}})">
            Submit Query
        </button>
    </span>
</div>
```

我们在`pd_options`属性中使用了`$val()`指令，动态获取 ID 为`"query{{prefix}}"`的输入框的值，并将其存储在`query`参数中。

## 在表格中显示搜索结果

上述代码一次性加载所有数据，这并不推荐，因为我们可能会有大量数据。类似地，一次性展示所有数据会导致界面变得缓慢且不实用。幸运的是，我们可以通过以下步骤轻松构建一个分页表格，而无需太多努力：

1.  创建一个名为`do_retrieve_page`的路由，接受一个 URL 作为参数，并返回表格主体的 HTML 片段

1.  将`first`、`previous`、`next`和`last`的 URL 作为字段保存在 PixieApp 类中

1.  创建一个分页控件（我们将使用 Bootstrap，因为它是现成的），包括`First`、`Prev`、`Next`和`Last`按钮

1.  创建一个表格占位符，显示需要显示的列标题

现在，我们将更新`do_search`方法，如下所示：

### 注意

**注意**：以下代码引用了`do_retrieve_page`方法，我们稍后会定义它。在你添加`do_retrieve_page`方法之前，请不要尝试运行此代码。

```py
[[GitHubTracking]]
@route(query="*")
@templateArgs
def do_search(self, query):
    self.first_url = "https://api.github.com/search/repositories?q={}".format(query)
    self.prev_url = None
    self.next_url = None
    self.last_url = None

    response = requests.get(self.first_url)
    if not response.ok:
        return "<div>An Error occurred: {{response.text}}</div>"

    total_count = response.json()['total_count']
    self.next_url = response.links.get('next', {}).get('url', None)
    self.last_url = response.links.get('last', {}).get('url', None)
    return """
<h1><center>{{total_count}} repositories were found</center></h1>
<ul class="pagination">
    <li><a href="#" pd_options="page=first_url" pd_target="body{{prefix}}">First</a></li>
    <li><a href="#" pd_options="page=prev_url" pd_target="body{{prefix}}">Prev</a></li>
    <li><a href="#" pd_options="page=next_url" pd_target="body{{prefix}}">Next</a></li>
    <li><a href="#" pd_options="page=last_url" pd_target="body{{prefix}}">Last</a></li>
</ul>
<table class="table">
    <thead>
        <tr>
            <th>Repo Name</th>
            <th>Lastname</th>
            <th>URL</th>
            <th>Stars</th>
        </tr>
    </thead>
    <tbody id="body{{prefix}}">
        {{this.invoke_route(this.do_retrieve_page, page='first_url')}}
    </tbody>
</table>
"""
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode6.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode6.py)

上面的代码示例展示了 PixieApps 的一个非常重要的特性，那就是你可以通过简单地将数据存储到类变量中来保持应用程序生命周期中的状态。在这个例子中，我们使用了 `self.first_url`、`self.prev_url`、`self.next_url` 和 `self.last_url` 变量。这些变量为每个分页控件中的按钮使用 `pd_options` 属性，并在每次调用 `do_retrieve_page` 路由时更新。`do_search` 返回的片段现在返回一个表格，其中包含一个由 `body{{prefix}}` 标识的占位符，成为每个按钮的 `pd_target`。我们还使用 `invoke_route` 方法来确保在首次显示表格时获取第一页。

我们之前看到，路由返回的 HTML 片段用于替换整个页面，但在前面的代码中，我们使用了 `pd_target="body{{prefix}}"` 属性来表示 HTML 片段将被注入到具有 `body{{prefix}}` ID 的表格主体元素中。如果需要，你还可以通过创建一个或多个 `<target>` 元素作为可点击源元素的子元素，来定义多个目标以响应用户操作。每个 `<target>` 元素本身可以使用所有 PixieApp 自定义属性来配置内核请求。

这是一个示例：

```py
<button type="button">Multiple Targets
    <target pd_target="elementid1" pd_options="state1=value1"></target>
    <target pd_target="elementid2" pd_options="state2=value2"></target>
</button>
```

回到我们的 GitHub 示例应用程序，`do_retrieve_page` 方法现在看起来像这样：

```py
[[GitHubTracking]]
@route(page="*")
@templateArgs
def do_retrieve_page(self, page):
    url = getattr(self, page)
    if url is None:
        return "<div>No more rows</div>"
    response = requests.get(url)
    self.prev_url = response.links.get('prev', {}).get('url', None)
    self.next_url = response.links.get('next', {}).get('url', None)
    items = response.json()['items']
    return """
{%for row in items%}
<tr>
    <td>{{row['name']}}</td>
    <td>{{row.get('owner',{}).get('login', 'N/A')}}</td>
    <td><a href="{{row['html_url']}}" target="_blank">{{row['html_url']}}</a></td>
    <td>{{row['stargazers_count']}}</td>
</tr>
{%endfor%}
        """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode7.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode7.py)

`page` 参数是一个字符串，包含我们想要显示的 `url` 类变量的名称。我们使用标准的 `getattr` Python 函数 ([`docs.python.org/2/library/functions.html#getattr`](https://docs.python.org/2/library/functions.html#getattr)) 来从页面中获取 `url` 值。然后，我们对 GitHub API 的 `url` 发出 GET 请求，获取以 JSON 格式返回的有效载荷，并将其传递给 Jinja2 模板，以生成将注入到表格中的一组行。为此，我们使用 Jinja2 中的 `{%for…%}` 循环控制结构 ([`jinja.pocoo.org/docs/templates/#for`](http://jinja.pocoo.org/docs/templates/#for)) 来生成一系列 `<tr>` 和 `<td>` HTML 标签。

以下截图展示了查询 `pixiedust` 的搜索结果：

![在表格中显示搜索结果](img/B09699_03_04.jpg)

显示来自查询的 GitHub 仓库列表的屏幕

### 注意

在第一部分中，我们展示了如何创建 `GitHubTracking` PixieApp，调用 GitHub 查询 REST API，并使用分页在表格中显示结果。你可以在这里找到包含源代码的完整笔记本：

`https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%201.ipynb`

在接下来的部分，我们将探索更多 PixieApp 的功能，这些功能将帮助我们通过让用户深入某个特定的仓库，并可视化该仓库的各种统计数据，来改进应用程序。

第一步是为搜索结果表格的每一行添加一个按钮，该按钮触发一个新路由，用于可视化所选仓库的统计信息。

以下代码是 `do_search` 函数的一部分，并在表头中添加了一个新列：

```py
<thead>
    <tr>
        <th>Repo Name</th>
        <th>Lastname</th>
        <th>URL</th>
        <th>Stars</th>
 <th>Actions</th>
    </tr>
</thead>
```

为了完成表格，我们更新了 `do_retrieve_page` 方法，添加了一个新单元格，其中包含一个 `<button>` 元素，带有与新路由匹配的 `pd_options` 参数：`analyse_repo_owner` 和 `analyse_repo_name`。这些参数的值是从用于遍历从 GitHub 请求中接收到的有效负载的 `row` 元素中提取的：

```py
{%for row in items%}
<tr>
    <td>{{row['name']}}</td>
    <td>{{row.get('owner',{}).get('login', 'N/A')}}</td>
    <td><a href="{{row['html_url']}}" target="_blank">{{row['html_url']}}</a></td>
    <td>{{row['stargazers_count']}}</td>
 <td>
 <button pd_options=
 "analyse_repo_owner={{row["owner"]["login"]}};
 analyse_repo_name={{row['name']}}"
 class="btn btn-default btn-sm" title="Analyze Repo">
 <i class="fa fa-line-chart"></i>
 </button>
 </td>
</tr>
{%endfor%}
```

在这个简单的代码更改完成后，通过重新运行单元格来重启 PixieApp，现在我们可以看到每个仓库的按钮，尽管我们还没有实现相应的路由，接下来我们将实现该路由。提醒一下，当没有找到匹配的路由时，默认路由将被触发。

以下截图展示了添加按钮后的表格：

![在表格中显示搜索结果](img/B09699_03_05.jpg)

为每一行添加操作按钮

下一步是创建与仓库可视化页面关联的路由。该页面的设计相当简单：用户从下拉框中选择他们想要在页面上可视化的数据类型。GitHub REST API 提供了多种类型的数据访问，但对于这个示例应用程序，我们将使用提交活动数据，这是统计类别的一部分（请参阅 [`developer.github.com/v3/repos/statistics/#get-the-last-year-of-commit-activity-data`](https://developer.github.com/v3/repos/statistics/#get-the-last-year-of-commit-activity-data) 以获取此 API 的详细说明）。

### 提示

作为练习，您可以通过添加其他类型 API 的可视化效果来改进这个示例应用程序，例如流量 API（[`developer.github.com/v3/repos/traffic`](https://developer.github.com/v3/repos/traffic)）。

同时需要注意的是，尽管大多数 GitHub API 可以在没有认证的情况下工作，但如果未提供凭证，服务器可能会限制响应。要进行请求认证，您需要使用 GitHub 密码或通过在 GitHub **设置**页面中选择**开发者设置**菜单，点击**个人访问令牌**菜单，然后点击**生成新令牌按钮**来生成个人访问令牌。

在一个单独的 Notebook 单元格中，我们将为 GitHub 用户 ID 和令牌创建两个变量：

```py
github_user = "dtaieb"
github_token = "XXXXXXXXXX"
```

这些变量稍后将在请求认证中使用。请注意，尽管这些变量是在各自的单元格中创建的，但它们对整个 Notebook 可见，包括 PixieApp 代码。

为了提供良好的代码模块化和复用性，我们将在一个新类中实现 Repo Visualization 页面，并让主 PixieApp 类继承该类并自动复用它的路由。这是一个模式，当你开始处理大型项目并希望将其拆分为多个类时，需要记住。

Repo Visualization 页面的主路由返回一个包含下拉菜单和 `<div>` 占位符的 HTML 片段。下拉菜单使用 Bootstrap `dropdown` 类创建（[`www.w3schools.com/bootstrap/bootstrap_dropdowns.asp`](https://www.w3schools.com/bootstrap/bootstrap_dropdowns.asp)）。为了使代码更易于维护，菜单项是通过 Jinja2 `{%for..` `%}` 循环生成的，该循环遍历一个包含元组的数组（[`docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences`](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences)），数组名为 `analyses`，它包含一个描述和一个用于将数据加载到 pandas DataFrame 中的函数。这里，我们再次在自己的单元格中创建这个数组，并将在 PixieApp 类中引用它：

```py
analyses = [("Commit Activity", load_commit_activity)]
```

### 注意

**注意**：`load_commit_activity` 函数将在本节后面讨论。

对于这个示例应用程序，数组仅包含一个与提交活动相关的元素，但将来你添加的任何元素都会自动被 UI 捕捉到。

`do_analyse_repo` 路由有两个参数：`analyse_repo_owner` 和 `analyse_repo_name`，这些应该足以访问 GitHub APIs。我们还需要将这些参数保存为类变量，因为它们将在生成可视化的路由中使用：

```py
@PixieApp
class RepoAnalysis():
    @route(analyse_repo_owner="*", analyse_repo_name="*")
    @templateArgs
    def do_analyse_repo(self, analyse_repo_owner, analyse_repo_name):
        self._analyse_repo_owner = analyse_repo_owner
        self._analyse_repo_name = analyse_repo_name
        return """
<div class="container-fluid">
    <div class="dropdown center-block col-sm-2">
        <button class="btn btn-primary dropdown-toggle" type="button" data-toggle="dropdown">
            Select Repo Data Set
            <span class="caret"></span>
        </button>
        <ul class="dropdown-menu" style="list-style:none;margin:0px;padding:0px">
            {%for analysis,_ in this.analyses%}
                <li>
                    <a href="#" pd_options="analyse_type={{analysis}}" pd_target="analyse_vis{{prefix}}"
                     style="text-decoration: none;background-color:transparent">
                        {{analysis}}
                    </a>
                </li>
            {%endfor%}
        </ul>
    </div>
    <div id="analyse_vis{{prefix}}" class="col-sm-10"></div>
</div>
"""
```

### 注意

你可以在此找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode8.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode8.py)

### 注意

上述代码中需要注意两件事：

+   Jinja2 模板使用 `this` 关键字引用 `analyses` 数组，尽管 `analyses` 变量并未定义为类变量。之所以能够这样工作，是因为 PixieApp 的另一个重要特性：在 Notebook 中定义的任何变量都可以像类变量一样引用。

+   我将 `analyse_repo_owner` 和 `analyse_repo_name` 作为类变量存储，并使用不同的名称，例如 `_analyse_repo_owner` 和 `_analyse_repo_name`。这很重要，因为使用相同的名称会对路由匹配算法产生副作用，算法也会查看类变量来查找参数。使用相同的名称会导致该路由始终被找到，这显然不是我们想要的效果。

动作按钮链接由 `<a>` 标签定义，并使用 `pd_options` 访问一个包含名为 `analyse_type` 的参数的路由，此外 `pd_target` 指向同一 HTML 片段下方定义的 `"analyse_vis{{prefix}}"` 占位符 `<div>`。

## 使用`pd_entity`属性调用 PixieDust 的 display() API

当使用`pd_options`属性创建内核请求时，PixieApp 框架会将当前的 PixieApp 类作为目标。但是，你可以通过指定`pd_entity`属性来更改这个目标。例如，你可以指向另一个 PixieApp，或者更有趣的是，指向`display()` API 支持的数据结构，比如 pandas 或 Spark DataFrame。在这种情况下，只要你包含`display()` API 所期望的正确选项，生成的输出将是图表本身（在 Matplotlib 的情况下是图像，在 Mapbox 的情况下是 Iframe，在 Bokeh 的情况下是 SVG）。获取正确选项的一种简单方法是，在自己的单元格中调用`display()` API，使用菜单配置所需的图表，然后复制通过点击**编辑元数据**按钮获得的单元格元数据 JSON 片段。（你可能需要先通过菜单**视图** | **单元格工具栏** | **编辑元数据**来启用该按钮）。

你也可以在不指定任何值的情况下指定`pd_entity`。在这种情况下，PixieApp 框架将使用传递给`run`方法的第一个参数作为实体，该方法用于启动 PixieApp 应用程序。例如，`my_pixieapp.run(cars)`，其中`cars`是通过`pixiedust.sampleData()`方法创建的 pandas 或 Spark DataFrame。`pd_entity`的值也可以是返回实体的函数调用。当你想在渲染实体之前动态计算实体时，这很有用。与其他变量一样，`pd_entity`的作用范围可以是 PixieApp 类，也可以是 Notebook 中声明的任何变量。

例如，我们可以在一个单元格中创建一个函数，该函数以前缀作为参数并返回一个 pandas DataFrame。然后我们将其用作我的 PixieApp 中的`pd_entity`值，如下代码所示：

```py
def compute_pdf(key):
    return pandas.DataFrame([
        {"col{}".format(i): "{}{}-{}".format(key,i,j) for i in range(4)} for j in range(10)
    ])
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode9.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode9.py)

在上述代码中，我们使用了 Python 列表推导式（[`docs.python.org/2/tutorial/datastructures.html#list-comprehensions`](https://docs.python.org/2/tutorial/datastructures.html#list-comprehensions)）来快速生成基于`key`参数的模拟数据。

### 注意

Python 列表推导式是我最喜欢的 Python 语言特性之一，因为它让你能够以简洁而富有表现力的语法创建、转换和提取数据。

然后，我可以创建一个 PixieApp，使用`compute_pdf`函数作为`pd_entity`来将数据呈现为表格：

```py
from pixiedust.display.app import *
@PixieApp
class TestEntity():
    @route()
    def main_screen(self):
        return """
        <h1><center>
            Simple PixieApp with dynamically computed dataframe
        </center></h1>
        <div pd_entity="compute_pdf('prefix')" pd_options="handlerId=dataframe" pd_render_onload></div>
        """
test = TestEntity()
test.run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode10.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode10.py)

在前面的代码中，为了简便起见，我将键硬编码为`'prefix'`，接下来留给您作为练习，使用输入控件和`$val()`指令使其可以由用户定义。

另一个需要注意的重要事项是，在显示图表的 div 中使用了`pd_render_onload`属性。这个属性告诉 PixieApp，在元素加载到浏览器 DOM 中后，立即执行由该元素定义的内核请求。

以下截图展示了前面 PixieApp 的结果：

![使用 pd_entity 属性调用 PixieDust display() API](img/B09699_03_06.jpg)

在 PixieApp 中动态创建 DataFrame

回到我们的*GitHub 跟踪*应用程序，现在让我们将`pd_entity`值应用到从 GitHub 统计 API 加载的 DataFrame 中。我们创建一个名为`load_commit_activity`的方法，负责将数据加载到 pandas DataFrame 并返回它和显示图表所需的`pd_options`：

```py
from datetime import datetime
import requests
import pandas
def load_commit_activity(owner, repo_name):
    response = requests.get(
        "https://api.github.com/repos/{}/{}/stats/commit_activity".format(owner, repo_name),
        auth=(github_user, github_token)
    ).json()
    pdf = pandas.DataFrame([
        {"total": item["total"], "week":datetime.fromtimestamp(item["week"])} for item in response
    ])

    return {
        "pdf":pdf,
        "chart_options": {
          "handlerId": "lineChart",
          "keyFields": "week",
          "valueFields": "total",
          "aggregation": "SUM",
          "rendererId": "bokeh"
        }
    }
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode11.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode11.py)

前面的代码向 GitHub 发送了一个 GET 请求，并使用在笔记本开始时设置的`github_user`和`github_token`变量进行身份验证。响应是一个 JSON 负载，我们将使用它来创建一个 pandas DataFrame。在创建 DataFrame 之前，我们需要将 JSON 负载转换为正确的格式。目前，负载看起来像这样：

```py
[
{"days":[0,0,0,0,0,0,0],"total":0,"week":1485046800},
{"days":[0,0,0,0,0,0,0],"total":0,"week":1485651600},
{"days":[0,0,0,0,0,0,0],"total":0,"week":1486256400},
{"days":[0,0,0,0,0,0,0],"total":0,"week":1486861200}
...
]
```

我们需要删除`days`键，因为它不需要用于显示图表，而且为了正确显示图表，我们需要将`week`键的值（Unix 时间戳）转换为 Python 的`datetime`对象。此转换通过 Python 列表推导和一行简单代码完成：

```py
[{"total": item["total"], "week":datetime.fromtimestamp(item["week"])} for item in response]
```

在当前实现中，`load_commit_activity`函数定义在它自己的单元格中，但我们也可以将其定义为 PixieApp 的成员方法。作为最佳实践，使用自己的单元格非常方便，因为我们可以对函数进行单元测试并快速迭代，而无需每次都运行整个应用程序，避免了额外的开销。

要获取`pd_options`值，我们只需运行该函数，使用示例仓库信息，然后在单独的单元格中调用`display()` API：

![使用 pd_entity 属性调用 PixieDust display() API](img/B09699_03_07.jpg)

在单独的单元格中使用 display()获取可视化配置

要获取前面的图表，您需要选择**折线图**，然后在**选项**对话框中，将`week`列拖到**键**框中，将`total`列拖到**值**框中。您还需要选择 Bokeh 作为渲染器。完成后，注意到 PixieDust 会自动检测到*x*轴是一个日期时间，并相应地调整渲染。

使用**编辑元数据**按钮，我们现在可以复制图表选项的 JSON 片段：

![使用 pd_entity 属性调用 PixieDust 的 display() API](img/B09699_03_08.jpg)

捕获 display() JSON 配置

并将其返回到`load_commit_activity`数据负载中：

```py
return {
        "pdf":pdf,
        "chart_options": {
          "handlerId": "lineChart",
          "keyFields": "week",
          "valueFields": "total",
          "aggregation": "SUM",
          "rendererId": "bokeh"
        }
    }
```

我们现在准备在`RepoAnalysis`类中实现`do_analyse_type`路由，如以下代码所示：

```py
[[RepoAnalysis]]
@route(analyse_type="*")
@templateArgs
def do_analyse_type(self, analyse_type):
    fn = [analysis_fn for a_type,analysis_fn in analyses if a_type == analyse_type]
    if len(fn) == 0:
        return "No loader function found for {{analyse_type}}"
    vis_info = fn0
    self.pdf = vis_info["pdf"]
    return """
    <div pd_entity="pdf" pd_render_onload>
        <pd_options>{{vis_info["chart_options"] | tojson}}</pd_options>
    </div>
    """
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode12.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode12.py)

该路由有一个名为`analyse_type`的参数，我们用它作为键来查找`analyses`数组中的 load 函数（注意，我再次使用列表推导式来快速执行查找）。然后，我们调用这个函数，并传入仓库的所有者和名称，以获取`vis_info` JSON 数据包，并将 pandas DataFrame 存储在一个名为`pdf`的类变量中。返回的 HTML 片段将使用`pdf`作为`pd_entity`的值，并使用`vis_info["chart_options"]`作为`pd_options`。在这里，我使用了`tojson` Jinja2 过滤器（[`jinja.pocoo.org/docs/templates/#list-of-builtin-filters`](http://jinja.pocoo.org/docs/templates/#list-of-builtin-filters)）以确保它在生成的 HTML 中被正确转义。尽管`vis_info`变量在栈上声明，但由于我为该函数使用了`@templateArgs`装饰器，因此仍然可以使用它。

在测试我们改进后的应用程序之前，最后一步是确保主`GitHubTracking` PixieApp 类继承自`RepoAnalysis` PixieApp：

```py
@PixieApp
class GitHubTracking(RepoAnalysis):
    @route()
    def main_screen(self):
        <<Code omitted here>>

    @route(query="*")
    @templateArgs
    def do_search(self, query):
        <<Code omitted here>>

    @route(page="*")
    @templateArgs
    def do_retrieve_page(self, page):
        <<Code omitted here>>

app = GitHubTracking()
app.run()
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode13.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode13.py)

下面显示的是 Repo 分析页面的截图：

![使用 pd_entity 属性调用 PixieDust 的 display() API](img/B09699_03_09.jpg)

GitHub 仓库提交活动可视化

### 注意

如果你想进一步实验，可以在这里找到完整的 Notebook，针对*GitHub 跟踪应用程序*第二部分：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%202.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%202.ipynb)

## 使用 pd_script 调用任意 Python 代码

在这一部分，我们将查看`pd_script`自定义属性，它可以让你在每次触发内核请求时运行任意的 Python 代码。执行 Python 代码时有几个规则需要遵循：

+   代码可以通过`self`关键字访问 PixieApp 类，以及 Notebook 中定义的任何变量、函数和类，如以下示例所示：

    ```py
    <button type="submit" pd_script="self.state='value'">Click me</button>
    ```

+   如果指定了`pd_target`，则任何使用`print`函数的语句都会输出到`target`元素中。如果没有`pd_target`，则不会这样做。换句话说，你不能使用`pd_script`来进行页面的完全刷新（你需要使用`pd_options`属性），例如以下示例：

    ```py
    from pixiedust.display.app import *

    def call_me():
        print("Hello from call_me")

    @PixieApp
    class Test():
        @route()
        def main_screen(self):
            return """
            <button type="submit" pd_script="call_me()" pd_target="target{{prefix}}">Click me</button>

            <div id="target{{prefix}}"></div>
            """
    Test().run()
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode14.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode14.py)

+   如果代码包含多行，建议使用`pd_script`子元素，这样可以使用多行编写 Python 代码。在使用这种形式时，确保代码遵循 Python 语言的缩进规则，如以下示例所示：

    ```py
    @PixieApp
    class Test():
        @route()
        def main_screen(self):
            return """
            <button type="submit" pd_script="call_me()" pd_target="target{{prefix}}">
                <pd_script>
                    self.name="some value"
                    print("This is a multi-line pd_script")
                </pd_script>
                Click me
            </button>

            <div id="target{{prefix}}"></div>
            """
    Test().run()
    ```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode15.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode15.py)

`pd_script`的一个常见使用场景是在触发内核请求之前更新服务器上的某些状态。让我们通过添加一个复选框，在我们的*GitHub Tracking*应用中切换数据的可视化方式，从线图切换为统计数据摘要。

在`do_analyse_repo`返回的 HTML 片段中，我们添加了一个复选框元素，用于在图表和统计摘要之间切换：

```py
[[RepoAnalysis]]
...
return """
<div class="container-fluid">
    <div class="col-sm-2">
        <div class="dropdown center-block">
            <button class="btn btn-primary
             dropdown-toggle" type="button"
             data-toggle="dropdown">
                Select Repo Data Set
                <span class="caret"></span>
            </button>
            <ul class="dropdown-menu"
             style="list-style:none;margin:0px;padding:0px">
                {%for analysis,_ in this.analyses%}
                    <li>
                        <a href="#"
                        pd_options="analyse_type={{analysis}}"
                        pd_target="analyse_vis{{prefix}}"
                        style="text-decoration: none;background-color:transparent">
                            {{analysis}}
                        </a>
                    </li>
                {%endfor%}
            </ul>
        </div>
        <div class="checkbox">
            <label>
                <input id="show_stats{{prefix}}" type="checkbox"
                  pd_script="self.show_stats=('$val(show_stats{{prefix}})' == 'true')">
                Show Statistics
            </label>
        </div>
    </div>
    <div id="analyse_vis{{prefix}}" class="col-sm-10"></div>
</div>
"""
```

在`checkbox`元素中，我们包含了一个`pd_script`属性，该属性根据`checkbox`元素的状态修改服务器上的变量状态。我们使用`$val()`指令来获取`show_stats_{{prefix}}`元素的值，并将其与`true string`进行比较。当用户点击复选框时，状态会立即在服务器上改变，下一次用户点击菜单时，数据显示的是统计数据而不是图表。

现在我们需要更改`do_analyse_type`路由，以动态配置`pd_entity`和`chart_options`：

```py
[[RepoAnalysis]]
@route(analyse_type="*")
@templateArgs
def do_analyse_type(self, analyse_type):
    fn = [analysis_fn for a_type,analysis_fn in analyses if a_type == analyse_type]
    if len(fn) == 0:
        return "No loader function found for {{analyse_type}}"
    vis_info = fn0
    self.pdf = vis_info["pdf"]
    chart_options = {"handlerId":"dataframe"} if self.show_stats else vis_info["chart_options"]
    return """
    <div pd_entity="get_pdf()" pd_render_onload>
        <pd_options>{{chart_options | tojson}}</pd_options>
    </div>
    """
```

### 注意

你可以在这里找到文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode16.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode16.py)

`chart_options`现在是一个本地变量，包含了在`show_stats`为`true`时以表格形式显示的选项，以及在不为`true`时作为常规折线图选项的显示方式。

`pd_entity`现在被设置为`get_pdf()`方法，该方法负责根据`show_stats`变量返回相应的 DataFrame：

```py
def get_pdf(self):
    if self.show_stats:
        summary = self.pdf.describe()
        summary.insert(0, "Stat", summary.index)
        return summary
    return self.pdf
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode17.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode17.py)

我们使用 pandas 的`describe()`方法（[`pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)），该方法返回一个包含统计摘要的 DataFrame，诸如计数、均值、标准差等。我们还确保这个 DataFrame 的第一列包含统计数据的名称。

我们需要进行的最后一次修改是初始化`show_stats`变量，因为如果我们不这样做，那么第一次检查它时会得到`AttributeError`异常。

由于使用`@PixieApp`装饰器的内部机制，你不能使用`__init__`方法来初始化变量；相反，PixieApp 编程模型要求你使用一个名为`setup`的方法，该方法在应用程序启动时保证会被调用：

```py
@PixieApp
class RepoAnalysis():
    def setup(self):
        self.show_stats = False
    ...
```

### 注意

**注意**：如果你有一个类继承自其他 PixieApps，那么 PixieApp 框架将自动按其出现顺序调用所有基类的`setup`函数。

以下截图显示了统计摘要的展示：

![使用 pd_script 调用任意 Python 代码](img/B09699_03_10.jpg)

GitHub 仓库的统计摘要

### 注意

你可以在这里找到完整的*GitHub 跟踪*应用程序第三部分的笔记本：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%203.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%203.ipynb)

## 使用`pd_refresh`使应用程序更具响应性

我们希望通过让**显示统计**按钮直接展示统计表格来改善用户体验，而不是让用户再次点击菜单。类似于加载**提交活动**的菜单，我们可以为复选框添加`pd_options`属性，并将`pd_target`属性指向`analyse_vis{{prefix}}`元素。我们可以将`pd_options`添加到`analyse_vis{{prefix}}`一次，而不是在每个触发新显示的控件中重复添加，然后通过`pd_refresh`属性让它自行更新。

以下图示显示了两种设计之间的差异：

![使用 pd_refresh 使应用程序更具响应性](img/B09699_03_11.jpg)

带有和不带`pd_refresh`的时序图

在两种情况下，第一步都是更新服务器端的某些状态。在第 2 步中由**控件**调用的路由的情况下，请求规范存储在控件本身中，触发第 3 步，即生成 HTML 片段并将其注入目标元素。使用`pd_refresh`时，控件不知道`pd_options`来调用路由，相反，它只是使用`pd_refresh`来通知目标元素，而后者将调用路由。在这种设计中，我们只需要在目标元素中指定一次请求，用户控件只需在触发刷新之前更新状态。这使得实现更容易维护。

为了更好地理解这两种设计之间的差异，我们来对比`RepoAnalysis`类中的两种实现。

对于**分析**菜单，变更如下：

之前，控件触发了`analyse_type`路由，将`{{analysis}}`选择作为内核请求的一部分，目标是`analyse_vis{{prefix}}`：

```py
<a href="#" pd_options="analyse_type={{analysis}}"
            pd_target="analyse_vis{{prefix}}"
            style="text-decoration: none;background-color:transparent">
      {{analysis}}
</a>
```

之后，控件现在将选择状态存储为类字段，并请求`analyse_vis{{prefix}}`元素刷新自身：

```py
<a href="#" pd_script="self.analyse_type='{{analysis}}'"
 pd_refresh="analyse_vis{{prefix}}"
 style="text-decoration: none;background-color:transparent">
    {{analysis}}
</a>
```

同样，**显示统计信息**复选框的变更如下：

之前，复选框只是简单地在类中设置`show_stats`状态；用户必须再次点击菜单才能获取可视化：

```py
<div class="checkbox">
    <label>
        <input type="checkbox"
         id="show_stats{{prefix}}"
pd_script="self.show_stats='$val(show_stats{{prefix}})'=='true'">
        Show Statistics
    </label>
</div>
```

之后，感谢`pd_refresh`属性，复选框一旦被选中，可视化就会立即更新：

```py
<div class="checkbox">
    <label>
        <input type="checkbox"
         id="show_stats{{prefix}}"
  pd_script="self.show_stats='$val(show_stats{{prefix}})'=='true'"
         pd_refresh="analyse_vis{{prefix}}">
         Show Statistics
    </label>
</div>
```

最后，`analyse_vis{{prefix}}`元素的变更如下：

之前，元素不知道如何更新自己，它依赖其他控件将请求定向到合适的路由：

```py
<div id="analyse_vis{{prefix}}" class="col-sm-10"></div>
```

之后，元素携带内核配置以更新自身；现在，任何控件都可以更改状态并调用刷新：

```py
<div id="analyse_vis{{prefix}}" class="col-sm-10"
     pd_options="display_analysis=true"
     pd_target="analyse_vis{{prefix}}">
</div>
```

### 注意

你可以在此处找到本节的完整 Notebook，适用于*GitHub 跟踪*应用程序第四部分：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%204.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%204.ipynb)

## 创建可重用的小部件

PixieApp 编程模型提供了一种机制，将复杂 UI 构造的 HTML 和逻辑封装成一个小部件，可以轻松地从其他 PixieApp 中调用。创建小部件的步骤如下：

1.  创建一个包含小部件的 PixieApp 类。

1.  创建一个带有特殊`widget`属性的路由，如示例所示：

    ```py
    @route(widget="my_widget")
    ```

    它将是小部件的起始路由。

1.  创建一个消费者 PixieApp 类，继承自小部件 PixieApp 类。

1.  通过使用`pd_widget`属性从`<div>`元素调用小部件。

下面是创建小部件和消费者 PixieApp 类的示例：

```py
from pixiedust.display.app import *

@PixieApp
class WidgetApp():
    @route(widget="my_widget")
    def widget_main_screen(self):
        return "<div>Hello World Widget</div>"

@PixieApp
class ConsumerApp(WidgetApp):
    @route()
    def main_screen(self):
        return """<div pd_widget="my_widget"></div>"""

ConsumerApp.run()
```

### 注意

你可以在此处找到代码：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode18.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/sampleCode18.py)

# 总结

在本章中，我们涵盖了 PixieApp 编程模型的基础构建模块，使您能够直接在 Notebook 中创建强大的工具和仪表板。

我们还通过展示如何构建*GitHub Tracking*示例应用程序（包括详细的代码示例）来说明了 PixieApp 的概念和技术。关于最佳实践和更高级的 PixieApp 概念将在第五章中介绍，*Python and PixieDust Best Practices and Advanced Concepts*，包括事件、流处理和调试。

现在，您应该已经对 Jupyter Notebooks、PixieDust 和 PixieApps 如何通过使数据科学家和开发人员能够在单一工具（如 Jupyter Notebook）中进行协作有了一个良好的理解。

在下一章中，我们将展示如何将 PixieApp 从 Notebook 中解放出来，并使用 PixieGateway 微服务服务器将其发布为 Web 应用程序。
