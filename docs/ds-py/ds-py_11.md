# 附录 A. PixieApp 快速参考

本附录是开发者的快速参考指南，提供了所有 PixieApp 属性的汇总。

# 注解

+   `@PixieApp`：类注解，必须添加到任何 PixieApp 类上。

    参数：无

    示例：

    ```py
    from pixiedust.display.app import *
    @PixieApp
    class MyApp():
        pass
    ```

+   `@route`：方法注解，必须加到一个方法上，以表示该方法（方法名可以随意）与一个路由相关联。

    参数：`**kwargs`。表示路由定义的关键字参数（键值对）。PixieApp 调度程序会根据以下规则将当前内核请求与路由进行匹配：

    +   参数数量最多的路由将最先被评估。

    +   所有参数必须匹配，路由才会被选中。参数值可以使用 `*` 来表示匹配任何值。

    +   如果找不到路由，则会选择默认路由（没有参数的那个）。

    +   路由参数的每个键可以是临时状态（由 `pd_options` 属性定义）或持久化状态（PixieApp 类的字段，直到明确更改之前都会保留）。

    +   方法可以有任意数量的参数。在调用方法时，PixieApp 调度程序将尝试根据参数名将方法参数与路由参数进行匹配。

    返回值：该方法必须返回一个 HTML 片段（除非使用了 `@captureOutput` 注解），该片段将被注入到前端。方法可以利用 Jinja2 模板语法生成 HTML。HTML 模板可以访问一些特定的变量：

    +   **this**：引用 PixieApp 类（注意，我们使用 `this` 而不是 `self`，因为 `self` 已经被 Jinja2 框架本身使用）

    +   **prefix**：一个字符串 ID，唯一标识 PixieApp 实例

    +   **entity**：请求的当前数据实体

    +   **方法参数**：方法的所有参数都可以作为变量在 Jinja2 模板中访问。

        ```py
        from pixiedust.display.app import *
        @PixieApp
        class MyApp():
            @route(key1=”value1”, key2=”*”)
            def myroute_screen(self, key1, key2):
                return “<div>fragment: Key1 = {{key1}} - Key2 = {{key2}}”
        ```

    示例：

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode25.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode25.py)

+   `@templateArgs`：注解，允许在 Jinja2 模板中使用任何本地变量。注意，`@templateArgs` 不能与 `@captureOutput` 一起使用：

    参数：无

    示例：

    ```py
    from pixiedust.display.app import *
    @PixieApp
    class MyApp():
        @route(key1=”value1”, key2=”*”)
        @templateArgs
        def myroute_screen(self, key1, key2):
            local_var = “some value”
            return “<div>fragment: local_var = {{local_var}}”
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode26.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode26.py)

+   `@captureOutput`：注解用于改变路由方法的契约，使得方法不再需要返回 HTML 片段。相反，方法体可以像在 Notebook 单元中那样直接输出结果。框架会捕获输出并以 HTML 形式返回。注意，在这种情况下你不能使用 Jinja2 模板。

    参数：无

    示例：

    ```py
    from pixiedust.display.app import *
    import matplotlib.pyplot as plt
    @PixieApp
    class MyApp():
        @route()
        @captureOutput
        def main_screen(self):
            plt.plot([1,2,3,4])
            plt.show()
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode27.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode27.py)

+   `@Logger`：通过向类添加日志方法来添加日志功能：`debug`、`warn`、`info`、`error`、`critical`、`exception`。

    参数：无

    示例：

    ```py
    from pixiedust.display.app import *
    from pixiedust.utils import Logger
    @PixieApp
    @Logger()
    class MyApp():
        @route()
        def main_screen(self):
            self.debug(“In main_screen”)
            return “<div>Hello World</div>”
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode28.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode28.py)

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode28.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode28.py)

# 自定义 HTML 属性

这些可以与任何常规 HTML 元素一起使用，用于配置内核请求。当元素接收到点击或更改事件时，或者在 HTML 片段完成加载后，PixieApp 框架可以触发这些请求。

+   `pd_options`：定义内核请求瞬态状态的键值对列表，格式如下：`pd_options=”key1=value1;key2=value2;...”`。当与`pd_entity`属性结合使用时，`pd_options`属性会调用 PixieDust 的`display()` API。在这种情况下，你可以从另一个 Notebook 单元格的元数据中获取值，在该单元格中你已经使用了`display()` API。建议在`display()`模式下使用`pd_options`时，为了方便起见，可以通过创建一个名为`<pd_options>`的子元素，并将 JSON 值作为文本包含在其中，来使用`pd_options`的 JSON 表示法。

    使用`pd_options`作为子元素调用`display()`的示例：

    ```py
    <div pd_entity>
        <pd_options>
            {
                “mapboxtoken”: “XXXXX”,
                “chartsize”: “90”,
                “aggregation”: “SUM”,
                “rowCount”: “500”,
                “handlerId”: “mapView”,
                “rendererId”: “mapbox”,
                “valueFields”: “IncidntNum”,
                “keyFields”: “X,Y”,
                “basemap”: “light-v9”
            }
        </pd_options>
    </div>
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode29.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode29.html)

    使用`pd_options`作为 HTML 属性的示例：

    ```py
    <!-- Invoke a route that displays a chart -->
    <button type=”submit” pd_options=”showChart=true” pd_target=”chart{{prefix}}”>
        Show Chart
    </button>
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode30.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode30.html)

+   `pd_entity`：仅用于在特定数据上调用`display()` API。必须与`pd_options`一起使用，其中的键值对将作为参数传递给`display()`。如果没有为`pd_entity`属性指定值，则假定它是传递给启动 PixieApp 的`run`方法的实体。`pd_entity`的值可以是 Notebook 中定义的变量，也可以是 PixieApp 的字段（例如，`pd_entity=”df”`），或者是使用点符号表示法指向对象的字段（例如，`pd_entity=”obj_instance.df”`）。

+   `pd_target`：默认情况下，内核请求的输出会被注入到整体输出单元格或对话框中（如果你将`runInDialog="true"`作为`run`方法的参数）。但是，你可以使用`pd_target="elementId"`来指定一个接收输出的目标元素。（请注意，`elementId`必须在当前视图中存在。）

    示例：

    ```py
    <div id=”chart{{prefix}}”>
    <button type=”submit” pd_options=”showChart=true” pd_target=”chart{{prefix}}”>
        Show Chart
    </button>
    </div>
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode31.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode31.html)

+   `pd_script`：此属性调用任意的 Python 代码作为内核请求的一部分。可以与其他属性如`pd_entity`和`pd_options`结合使用。需要注意的是，必须遵守 Python 的缩进规则（[`docs.python.org/2.0/ref/indentation.html`](https://docs.python.org/2.0/ref/indentation.html)），以避免运行时错误。

    如果 Python 代码包含多行，建议将`pd_script`作为子元素使用，并将代码存储为文本。

    示例：

    ```py
    <!-- Invoke a method to load a dataframe before visualizing it -->
    <div id=”chart{{prefix}}”>
    <button type=”submit”
        pd_entity=”df”
        pd_script=”self.df = self.load_df()”
        pd_options=”handlerId=dataframe”
        pd_target=”chart{{prefix}}”>
        Show Chart
    </button>
    </div>
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode32.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode32.html)

+   `pd_app`：此属性通过其完全限定类名动态调用一个独立的 PixieApp。可以使用`pd_options`属性传递路由参数，以调用 PixieApp 的特定路由。

    示例：

    ```py
    <div pd_render_onload
         pd_option=”show_route_X=true”
         pd_app=”some.package.RemoteApp”>
    </div>
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode33.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode33.html)

+   `pd_render_onload`：此属性应在页面加载时触发内核请求，而不是在用户点击某个元素或发生变化事件时触发。`pd_render_onload`属性可以与定义请求的其他属性结合使用，如`pd_options`或`pd_script`。请注意，此属性只能与 div 元素一起使用。

    示例：

    ```py
    <div pd_render_onload>
        <pd_script>
    print(‘hello world rendered on load’)
        </pd_script>
    </div>
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode34.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode34.html)

+   `pd_refresh`：此属性用于强制 HTML 元素执行内核请求，即使没有发生任何事件（点击或变化事件）。如果没有指定值，则刷新当前元素；否则，刷新值中指定 ID 的元素。

    示例：

    ```py
    <!-- Update state before refreshing a chart -->
    <button type=”submit”
        pd_script=”self.show_line_chart()”
        pd_refresh=”chart{{prefix}}”>
        Show line chart
    </button>
    ```

    ### 注意

    你可以在这里找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode35.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode35.html)

+   `pd_event_payload`：此属性用于发出一个带有指定负载内容的 PixieApp 事件。该属性遵循与`pd_options`相同的规则：

    +   每个键值对必须使用`key=value`表示法进行编码

    +   事件将在点击或变化事件时触发

    +   支持使用`$val()`指令动态注入用户输入

    +   使用`<pd_event_payload>`子元素输入原始 JSON。

        ```py
        <button type=”submit” pd_event_payload=”type=topicA;message=Button clicked”>
            Send event A
        </button>
        <button type=”submit”>
            <pd_event_payload>
            {
                “type”:”topicA”,
                “message”:”Button Clicked”
            }
            </pd_event_payload>
            Send event A
        </button>
        ```

    示例：

    ### 注意

    你可以在此处找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode36.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode36.html)

+   `pd_event_handler`：订阅者可以通过声明`<pd_event_handler>`子元素来监听事件，该元素可以接受任何 PixieApp 内核执行属性，如`pd_options`和`pd_script`。该元素必须使用`pd_source`属性来过滤其想要处理的事件。`pd_source`属性可以包含以下值之一：

    +   `targetDivId`：只有来自指定 ID 元素的事件才会被接受

    +   `type`：只有指定类型的事件才会被接受

        ```py
        <div class=”col-sm-6” id=”listenerA{{prefix}}”>
            Listening to button event
            <pd_event_handler
                pd_source=”topicA”
                pd_script=”print(eventInfo)”
                pd_target=”listenerA{{prefix}}”>
            </pd_event_handler>
        </div>
        ```

    示例：

    ### 注意

    你可以在此处找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode37.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode37.html)

    **注意**：使用`*`作为`pd_source`表示将接受所有事件。

+   `pd_refresh_rate`：用于在指定的时间间隔内（以毫秒为单位）重复执行一个元素。这对于需要轮询特定变量状态并在 UI 中显示结果的场景非常有用。

    示例：

    ```py
    <div pd_refresh_rate=”3000”
        pd_script=”print(self.get_status())”>
    </div>
    ```

    ### 注意

    你可以在此处找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode38.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode38.html)

# 方法

+   `setup`：这是 PixieApp 实现的可选方法，用于初始化其状态。在 PixieApp 运行之前会自动调用。

    参数：无

    示例：

    ```py
    def setup(self):
        self.var1 = “some initial value”
        self.pandas_dataframe = pandas.DataFrame(data)
    ```

    ### 注意

    你可以在此处找到代码文件：

    [`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode39.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%205/sampleCode39.py)

+   `run`：用于启动 PixieApp。

    参数：

    +   **实体**：[可选] 作为输入传递给 PixieApp 的数据集。可以通过`pd_entity`属性或直接作为名为`pixieapp_entity`的字段引用。

    +   ****kwargs**：要传递给 PixieApp 的关键字参数。例如，使用`runInDialog="true"`将以对话框方式启动 PixieApp。

        ```py
        app = MyPixieApp()
        app.run(runInDialog=”true”)
        ```

    示例：

+   `invoke_route`：用于以编程方式调用路由。

    参数：

    +   **路由方法**：需要调用的方法。

    +   ****kwargs**：要传递给路由方法的关键字参数。

        ```py
        app.invoke_route(app.route_method, arg1 = “value1”, arg2 = “value2”)
        ```

    示例：

+   `getPixieAppEntity`：用于检索调用`run()`方法时传递的当前 PixieApp 实体（可能为 None）。`getPixieAppEntity()`通常在 PixieApp 内部调用，即：

    ```py
    self.getPixieAppEntity()
    ```
