# 第十一章。命名空间、作用域和模块

在本章中，我们将介绍 Python 模块。模块是包含函数和类定义的文件。本章还解释了命名空间的概念以及函数和模块之间变量的作用域。

# 命名空间

Python 对象的名称，如变量、类、函数和模块的名称，收集在命名空间中。模块和类有自己的命名空间，其名称与这些对象相同。这些命名空间在导入模块或实例化类时创建。模块的命名空间生命周期与当前的 Python 会话一样长。类实例的命名空间生命周期直到实例被删除。

函数在执行（调用）时创建一个局部命名空间。当函数通过常规返回或异常停止执行时，它会被删除。局部命名空间是无名的。

命名空间的概念将变量名称放入其上下文中。例如，有几个名为 `sin` 的函数，它们通过所属的命名空间来区分，如下面的代码所示：

```py
import math
import scipy
math.sin
scipy.sin
```

它们确实是不同的，因为 `scipy.sin` 是一个接受列表或数组作为输入的通用函数，而 `math.sin` 只接受浮点数。可以通过命令 `dir(<命名空间名称>)` 获取特定命名空间中所有名称的列表。它包含两个特殊名称 `__name__` 和 `__doc__`。前者指的是模块的名称，后者指的是其文档字符串：

```py
math.__name__ # returns math
math.__doc__ # returns 'This module is always ...'
```

存在一个特殊的命名空间，`__builtin__`，它包含在 Python 中无需任何`import`即可使用的名称。它是一个命名命名空间，但在引用内置对象时不需要给出其名称：

```py
'float' in dir(__builtin__) # returns True
float is __builtin__.float # returns True
```

# 变量的作用域

在程序的一部分中定义的变量不需要在其他部分中为人所知。所有知道某个变量的程序单元都称为该变量的作用域。我们首先给出一个例子；让我们考虑两个嵌套函数：

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
    print("""
          my_function's namespace: {} 
          other_function's namespace: {}
          """.format(dir(),other_function()))
    return a
```

执行 `my_function(3)` 的结果如下：

```py
my_function's namespace: ['a', 'b', 'in1', 'other_function'] 
other_function's namespace: ['a', 'c', 'd']
```

变量 `e` 位于包含函数 `my_function` 的程序单元的命名空间中。变量 `a` 位于该函数的命名空间中，该函数本身又包含最内层的函数 `other_function`。对于这两个函数，`e` 是一个全局变量。

将信息通过参数列表传递给函数是一种良好的实践，而不是使用前面示例中的构造。在第七章的*匿名函数*部分中可以找到一个例外，在那里全局变量被用于闭包。通过给它赋值，一个变量自动成为局部变量：

```py
e = 3
def my_function():
    e = 4
    a = 2
    print("my_function's namespace: {}".format(dir()))
```

执行

```py
e = 3
my_function()
e # has the value 3
```

给出：

```py
my_function's namespace: ['a', 'e']
```

其中 `e` 成为一个局部变量。实际上，现在这段代码有两个属于不同命名空间的变量 `e`。

通过使用 `global` 声明语句，可以在函数中定义的变量被设置为全局变量，即其值即使在函数外部也可以访问。以下是如何使用 `global` 声明进行演示：

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

### 小贴士

**避免使用全局变量**

建议避免使用这种构造和 `global` 的使用。这类代码难以调试和维护。使用类（请参阅第八章类，以获取更多信息）使得 `global` 主要变得过时。

# 模块

在 Python 中，模块只是一个包含类和函数的文件。通过在你的会话或脚本中导入该文件，函数和类变得可用。

## 简介

Python 默认附带了许多不同的库。你也可能想要安装更多用于特定目的的库，例如优化、绘图、读取/写入文件格式、图像处理等。NumPy 和 SciPy 是这类库的重要例子，matplotlib 用于绘图是另一个例子。在本章末尾，我们将列出一些有用的库。

要使用库，你可以：

+   只从库中加载某些对象，例如从 NumPy：

    ```py
            from numpy import array, vander
    ```

+   或者加载整个库：

    ```py
            from numpy import *
    ```

+   或者通过创建一个以库名为名的命名空间来访问整个库：

    ```py
            import numpy
            ...
            numpy.array(...)
    ```

    在库函数前加上命名空间可以访问此函数，并区分具有相同名称的其他对象。

此外，命名空间的名字可以与 `import` 命令一起指定：

```py
import numpy as np
...
np.array(...)
```

你选择哪种选项会影响你代码的可读性以及出错的可能性。一个常见的错误是遮蔽：

```py
from scipy.linalg import eig
A = array([[1,2],[3,4]])
(eig, eigvec) = eig(A)
...
(c, d) = eig(B) # raises an error
```

避免这种意外效果的一种方法是使用 `import`：

```py
import scipy.linalg as sl
A = array([[1,2],[3,4]])
(eig, eigvec) = sl.eig(A) # eig and sl.eig are different objects
...
(c, d) = sl.eig(B)
```

在本书中，我们使用了许多命令、对象和函数。这些是通过以下语句导入到局部命名空间的：

```py
from scipy import *
```

以这种方式导入对象不会使导入它们的模块变得明显。以下表格中给出了几个例子（*表 11.1*）：

| **库** | **方法** |
| --- | --- |
| `numpy` | `array`, `arange`, `linspace`, `vstack`, `hstack`, `dot`, `eye`, `identity`, 和 `zeros`. |
| `numpy.linalg` | `solve`, `lstsq`, `eig`, 和 `det`. |
| `matplotlib.pyplot` | `plot`, `legend`, 和 `cla`. |
| `scipy.integrate` | `quad`. |
| `copy` | `copy` 和 `deepcopy`. |

表 11.1：导入对象的示例

## IPython 中的模块

IPython 用于代码开发。一个典型的场景是你在开发周期内修改一些函数或类定义的文件上工作。为了将此类文件的 内容加载到 shell 中，你可以使用 `import`，但文件只加载一次。修改文件对后续导入没有影响。这就是 IPyhthon 的魔法命令 `run` 出现的原因。

### IPython 魔法命令

IPython 有一个名为 `run` 的特殊魔法命令，它将文件作为直接在 Python 中运行一样执行。这意味着文件是独立于 IPython 中已定义的内容执行的。当你想要测试一个作为独立程序设计的脚本时，这是在 IPython 内部执行文件推荐的方法。你必须以与从命令行执行相同的方式在执行文件中导入所有需要的内容。在 `myfile.py` 中运行代码的典型示例是：

```py
from numpy import array
...
a = array(...)
```

此脚本文件通过 `exec(open('myfile.py').read())` 在 Python 中执行。或者，如果你想在脚本独立于之前的导入运行时，在 IPython 中可以使用魔法命令 `run myfile`。文件中定义的所有内容都导入到 IPython 工作区。

## 变量 __name__

在任何模块中，特殊变量 `__name__` 被定义为当前模块的名称。在命令行（在 IPython 中），此变量设置为 `__main__`，这允许以下技巧：

```py
# module
import ...

class ...

if __name__ == "__main__":
   # perform some tests here
```

测试仅在文件直接运行时执行，而不是在导入时执行。

## 一些有用的模块

有用的 Python 模块列表非常庞大。在下面的表中，我们给出了一份非常简短的此类列表片段，专注于与数学和工程应用相关的模块（*表 11.2*）：

| **模块** | **描述** |
| --- | --- |
| `scipy` | 科学计算中使用的函数 |
| `numpy` | 支持数组和相关方法 |
| `matplotlib` | 使用导入子模块 pyplot 进行绘图和可视化 |
| `functools` | 函数的偏应用 |
| `itertools` | 提供特殊功能的迭代器工具，如切片到生成器 |
| `re` | 高级字符串处理的正则表达式 |
| `sys` | 系统特定函数 |
| `os` | 操作系统接口，如目录列表和文件处理 |
| `datetime` | 表示日期和日期增量 |
| `time` | 返回系统时钟时间 |
| `timeit` | 测量执行时间 |
| `sympy` | 计算机算术包（符号计算） |
| `pickle` |  Pickling，特殊的文件输入和输出格式 |
| `shelves` |  Shelves，特殊的文件输入和输出格式 |
| `contextlib` | 上下文管理器工具 |

表 11.2：工程应用的非详尽 Python 包列表

# 摘要

我们在本书开头告诉您，您必须导入 SciPy 和其他有用的模块。现在您完全理解了导入的含义。我们介绍了命名空间，并讨论了 `import` 和 `from ... import *` 之间的区别。变量的作用域在早期的 第七章 中已经介绍，但现在您对这个概念的重要性有了更完整的了解。
