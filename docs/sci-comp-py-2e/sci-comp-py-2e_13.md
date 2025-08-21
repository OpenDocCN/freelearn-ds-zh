命名空间、作用域和模块

在本章中，我们将介绍 Python 模块。模块是包含函数和类定义的文件。本章还解释了命名空间和跨函数和模块的变量作用域的概念。

本章将涵盖以下主题：

+   命名空间

+   变量的作用域

+   模块

# 第十四章：13.1 命名空间

Python 对象的名称，如变量、类、函数和模块的名称，都集中在命名空间中。模块和类具有它们自己的命名空间，与这些对象的名称相同。这些命名空间在导入模块或实例化类时创建。模块的命名空间的生存期与当前 Python 会话一样长。类实例的命名空间的生存期是直到实例被删除。

当函数被执行（调用）时，函数会创建一个局部命名空间。当函数通过常规返回或异常停止执行时，局部命名空间将被删除。局部命名空间是无名的。

命名空间的概念将变量名放置在其上下文中。例如，有几个名为`sin`的函数，它们通过所属的命名空间进行区分，如下面的代码所示：

```py
import math
import numpy
math.sin
numpy.sin
```

它们确实不同，因为`numpy.sin`是一个通用函数，接受列表或数组作为输入，而`math.sin`仅接受浮点数。可以使用命令`dir(<name of the namespace>)`获取特定命名空间中所有名称的列表。它包含两个特殊名称，`__name__`指的是模块的名称，`__doc__`指的是其文档字符串：

```py
math.__name__ # returns math
math.__doc__ # returns 'This module provides access to .....'
```

有一个特殊的命名空间，`__builtin__`，其中包含在 Python 中无需任何导入即可使用的名称。它是一个命名空间，但是在引用内置对象时不需要给出其名称：

```py
'float' in dir(__builtin__) # returns True 
float is __builtin__.float # returns True
```

让我们在下一节学习变量的作用域。

# 13.2 变量的作用域

程序的一部分定义的变量不需要在其他部分中知道。已知某个变量的所有程序单元被称为该变量的*作用域*。我们先举一个例子。让我们考虑两个嵌套函数：

```py
e = 3
def my_function(in1):
    a = 2 * e
    b = 3
    in1 = 5
    def other_function():
       c = a
       d = e
       return dir()
    print(f"""
          my_function's namespace: {dir()} 
          other_function's namespace: {other_function()}
          """)
    return a
```

执行`my_function(3)`的结果是：

```py
my_function's namespace: ['a', 'b', 'in1', 'other_function'] 
other_function's namespace: ['a', 'c', 'd']
```

变量`e`位于包围函数`my_function`的程序单元的命名空间中。变量`a`位于该函数的命名空间中，该函数本身包围最内层的函数`other_function`。对于这两个函数，`e`是一个全局变量，即它不在本地命名空间中，也不会被`dir()`列出，但其值是可用的。

通过参数列表将信息传递给函数，而不使用前面示例中的构造是一种良好的实践。一个例外可以在第 7.7 节找到：*匿名函数*，在这里全局变量用于闭包。

通过为其分配一个值，变量自动成为局部变量：

```py
e = 3
def my_function():
    e = 4
    a = 2
    print(f"my_function's namespace: {dir()}")
```

执行以下代码块时可以看到这一点：

```py
e = 3
my_function()
e # has the value 3
```

上述代码的输出显示了 `my_function` 的局部变量：

```py
my_function's namespace: ['a', 'e']
```

现在，`e` 变成了一个局部变量。事实上，这段代码现在有两个 `e` 变量，分别属于不同的命名空间。

通过使用 `global` 声明语句，可以将函数中定义的变量变为全局变量，也就是说，它的值即使在函数外部也可以访问。以下是使用 `global` 声明的示例：

```py
def fun():
    def fun1():
        global a
        a = 3
    def fun2():
        global b
        b = 2
        print(a)
    fun1()
    fun2() # prints a
    print(b)
```

建议避免使用这种构造以及 `global` 的使用。使用 `global` 的代码难以调试和维护。类的使用基本上使得 `global` 变得过时。

# 13.3 模块

在 Python 中，模块只是一个包含类和函数的文件。通过在会话或脚本中导入该文件，函数和类就可以被使用。

## 13.3.1 介绍

Python 默认带有许多不同的库。你可能还希望为特定目的安装更多的库，如优化、绘图、读写文件格式、图像处理等。NumPy 和 SciPy 是这类库的重要例子，Matplotlib 是用于绘图的另一个例子。在本章结束时，我们将列出一些有用的库。

使用库的方法有两种：

+   只从库中加载某些对象，例如，从 NumPy 中：

```py
from numpy import array, vander
```

+   加载整个库：

```py
from numpy import *
```

+   或者通过创建一个与库名相同的命名空间来访问整个库：

```py
import numpy
...
numpy.array(...)
```

在库中的函数前加上命名空间，可以访问该函数，并将其与其他同名对象区分开来。

此外，可以在 `import` 命令中指定命名空间的名称：

```py
import numpy as np
...
np.array(...)
```

你选择使用这些替代方式的方式会影响代码的可读性以及出错的可能性。一个常见的错误是变量覆盖（shadowing）：

```py
from scipy.linalg import eig
A = array([[1,2],[3,4]])
(eig, eigvec) = eig(A)
...
(c, d) = eig(B) # raises an error
```

避免这种无意的效果的一种方法是使用 `import` 而不是 `from`，然后通过引用命名空间来访问命令，例如 `sl`：

```py
import scipy.linalg as sl
A = array([[1,2],[3,4]])
(eig, eigvec) = sl.eig(A) # eig and sl.eig are different objects
...
(c, d) = sl.eig(B)
```

本书中，我们使用了许多命令、对象和函数。这些通过类似以下语句被导入到本地命名空间中：

```py
from scipy import *
```

以这种方式导入对象不会显现它们来自的模块。以下表格给出了几个例子（*表 13.1*）：

| **库** | **方法** |
| --- | --- |
| `numpy` | `array`、`arange`、`linspace`、`vstack`、`hstack`、`dot`、`eye`、`identity` 和 `zeros`。 |
| `scipy.linalg` | `solve`、`lstsq`、`eig` 和 `det`。 |
| `matplotlib.pyplot` | `plot`、`legend` 和 `cla`。 |
| `scipy.integrate` | `quad`。 |
| `copy` | `copy` 和 `deepcopy`。 |

表 13.1：模块及其对应导入函数的示例

## 13.3.2 IPython 中的模块

IPython 用于代码开发。一个典型的场景是，你在一个文件中工作，文件中有一些函数或类定义，你在开发周期中对其进行更改。为了将该文件的内容加载到 Shell 中，你可以使用 `import`，但文件只会加载一次。更改文件对后续的导入没有影响。这时，IPython 的 *魔法命令* `run` 就显得非常有用。

### IPython 魔法命令 – run

IPython 有一个特殊的 *魔法命令* `run`，它会像直接在 Python 中运行一样执行文件。这意味着文件会独立于 IPython 中已经定义的内容执行。这是推荐的在 IPython 中执行文件的方法，特别是当你想要测试作为独立程序的脚本时。你必须像从命令行执行文件一样，在被执行的文件中导入所有需要的内容。运行 `myfile.py` 文件的典型示例如下：

```py
from numpy import array
...
a = array(...)
```

该脚本文件在 Python 中通过 `exec(open('myfile.py').read())` 执行。或者，在 IPython 中可以使用 *魔法命令* `run myfile`，如果你想确保脚本独立于之前的导入运行。文件中定义的所有内容都会被导入到 IPython 工作空间中。

## 13.3.3 变量 `__name__`

在任何模块中，特殊变量 `__name__` 被定义为当前模块的名称。在命令行（在 IPython 中）中，此变量被设置为 `__main__`。这个特性使得以下技巧成为可能：

```py
# module
import ...

class ...

if __name__ == "__main__":
   # perform some tests here
```

测试只有在文件直接运行时才会执行，*而不是*在被导入时执行，因为当被导入时，变量 `__name__` 会取模块名，而不是 `__main__`。

## 13.3.4 一些有用的模块

有用的 Python 模块列表非常庞大。下表展示了这样一个简短的列表，专注于与数学和工程应用相关的模块 (*表 13.2)*：

| **模块** | **描述** |
| --- | --- |
| `scipy` | 科学计算中使用的函数 |
| `numpy` | 支持数组及相关方法 |
| `matplotlib` | 绘图和可视化 |
| `functools` | 函数的部分应用 |
| `itertools` | 提供特殊功能的迭代器工具，例如切片生成器 |
| `re` | 用于高级字符串处理的正则表达式 |
| `sys` | 系统特定函数 |
| `os` | 操作系统接口，如目录列表和文件处理 |
| `datetime` | 表示日期及日期增量 |
| `time` | 返回壁钟时间 |
| `timeit` | 测量执行时间 |
| `sympy` | 计算机算术包（符号计算） |
| `pickle` | Pickling，一种特殊的文件输入输出格式 |
| `shelves` | Shelves，一种特殊的文件输入输出格式 |
| `contextlib` | 用于上下文管理器的工具 |

表 13.2：用于工程应用的有用 Python 包的非详尽列表

我们建议不要使用数学模块`math`，而是推荐使用`numpy`。原因是 NumPy 的许多函数，例如`sin`，是作用于数组的，而`math`中的对应函数则不支持。

# 13.4 总结

我们从告诉你需要导入 SciPy 和其他有用的模块开始。现在你已经完全理解了导入的含义。我们介绍了命名空间，并讨论了`import`和`from ... import *`之间的区别。变量的作用域已在第 7.2.3 节中介绍：*访问定义在局部之外的变量*

*命名空间*，但现在你对该概念的重要性有了更完整的理解。
