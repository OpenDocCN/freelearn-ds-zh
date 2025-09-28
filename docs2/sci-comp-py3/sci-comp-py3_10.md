# 第十章. 错误处理

在本章中，我们将介绍错误、异常以及如何查找和修复它们。处理异常是编写可靠和可用代码的重要部分。我们将介绍基本的内置异常，并展示如何使用和处理异常。我们将介绍调试，并展示如何使用内置的 Python 调试器。

# 异常是什么？

程序员（即使是经验丰富的程序员）发现的一个错误是代码有错误的语法，这意味着代码指令格式不正确。

考虑一个语法错误的例子：

```py
>>> for i in range(10)
  File “<stdin>”, line 1
    for i in range(10)
                      ^
SyntaxError: invalid syntax
```

错误发生是因为 `for` 声明末尾缺少冒号。这是一个异常被引发的例子。在 `SyntaxError` 的情况下，它告诉程序员代码有错误的语法，并打印出错误发生的行，其中有一个箭头指向该行中的问题所在。

Python 中的异常是从一个称为 `Exception` 的基类派生（继承）的。Python 内置了许多异常。一些常见的异常类型列在 *表 10.1* 中（有关内置异常的完整列表，请参阅 *[[38]](apa.html "附录 . 参考文献")*）。

这里有两个常见的异常示例。正如你所预期的，当你尝试除以零时，会引发 `ZeroDivisionError`。

```py
def f(x):
    return 1/x

>>> f(2.5)
0.4 
>>> f(0)

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "exception_tests.py", line 3, in f
    return 1/x
ZeroDivisionError: integer division or modulo by zero
```

| **异常** | **描述** |
| --- | --- |
| `IndexError` | 索引超出范围，例如，当 v 只有 5 个元素时 `v[10]` |
| `KeyError` | 引用未定义的字典键 |
| `NameError` | 未找到名称，例如，未定义的变量 |
| `LinAlgError` | `linalg` 模块中的错误，例如，当使用奇异矩阵求解系统时 |
| `ValueError` | 数据值不兼容，例如，使用 `dot` 与不兼容的数组 |
| `IOError` | I/O 操作失败，例如，“文件未找到” |
| `ImportError` | 导入模块或名称时未找到 |

表 10.1：一些常用内置异常及其含义

除以零会引发 `ZeroDivisionError` 并打印出错误发生的文件、行和函数名称。

如我们之前所见，数组只能包含相同数据类型的元素。如果你尝试分配一个不兼容类型的值，则会引发 `ValueError`。一个值错误的例子：

```py
>>> a = arange(8.0) 
>>> a 
array([ 0., 1., 2., 3., 4., 5., 6., 7.]) 
>>> a[3] = 'string'
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module>
ValueError: could not convert string to float: string
```

这里，`ValueError` 被引发，因为数组包含浮点数，而一个元素不能被分配一个字符串值。

## 基本原则

让我们看看如何通过使用 `raise` 语句引发异常和 `try` 语句捕获异常来使用异常的基本原则。

### 抛出异常

创建错误称为引发异常。你已经在上一节中看到了一些异常的示例。你也可以定义自己的异常，可以是预定义类型或无类型。引发异常使用如下命令：

```py
raise Exception("Something went wrong")
```

当出错时，可能会诱使你打印出错误消息，例如，如下所示：

```py
print("The algorithm did not converge.")
```

由于多种原因，这并不推荐。首先，打印输出很容易被忽略，尤其是如果消息被埋藏在许多其他打印到控制台的消息中。其次，更重要的是，它使得其他代码无法使用你的代码。调用代码将无法知道发生了错误，因此无法处理它。

由于这些原因，总是抛出异常而不是捕获异常要好。异常应该总是包含一个描述性的消息，例如：

```py
raise Exception("The algorithm did not converge.")
```

这条消息将清楚地显示给用户。它还给了调用代码知道发生了错误的机会，并可能找到补救措施。

这里是一个典型的例子，检查函数内部的输入以确保在继续之前它是可用的。例如，对负值和正确数据类型进行简单检查确保了计算阶乘函数的预期输入：

```py
def factorial(n):
  if not (n >=0 and isinstance(n,(int,int32,int64))):
    raise ValueError("A positive integer is expected")
    ...
```

如果输入不正确，函数的用户将立即知道错误是什么，并且处理异常是用户的责任。注意在抛出预定义的异常类型时使用异常名称，在这种情况下是 `ValueError` 后跟消息。通过指定异常类型，调用代码可以决定根据抛出的错误类型以不同的方式处理错误。

总结来说，总是抛出异常比打印错误消息要好。

### 捕获异常

处理异常被称为捕获异常。检查异常使用 `try` 和 `except` 命令进行。

异常会停止程序执行流程，并寻找最近的包含 `try` 的包围块。如果异常没有被捕获，程序单元将被留下，并且它将继续在调用栈中更高层的程序单元中搜索下一个包围的 `try` 块。如果在没有找到任何块且异常未被处理的情况下，执行将完全停止；将显示标准的回溯信息。

让我们看看 `try` 语句的一个例子：

```py
try:
    <some code that might raise an exception>
except ValueError:
    print("Oops, a ValueError occurred")
```

在这种情况下，如果 `try` 块内部抛出一个 `ValueError` 类型的错误，异常将被捕获，并在 `except` 块中打印出消息。如果 `try` 块内部没有发生异常，则 `except` 块将被完全跳过，并且执行继续。

`except` 语句可以捕获多个异常。这是通过简单地将它们分组在一个元组中实现的，如下所示：

```py
except (RuntimeError, ValueError, IOError):
```

`try` 块也可以有多个 `except` 语句。这使得根据异常类型的不同来处理异常成为可能。让我们看看多个异常类型的例子：

```py
try:
    f = open('data.txt', 'r')
    data = f.readline()
    value = float(data)
except OSError as oe:
    print("{}:  {}".format(oe.strerror, oe.filename))
except ValueError:
    print("Could not convert data to float.")
```

例如，如果文件不存在，将会捕获一个 `OSError`；如果文件的第一行中的数据与浮点数据类型不兼容，将会捕获一个 `ValueError`。

在此示例中，我们通过关键字`as`将`OSError`赋值给变量`oe`。这允许在处理此异常时访问更多详细信息。在这里，我们打印了错误字符串`oe.strerror`和相关的文件名`oe.filename`。每种错误类型都可以根据类型有自己的变量集。如果文件不存在，在先前的示例中，消息将是：

```py
I/O error(2): No such file or directory
```

另一方面，如果文件存在但您没有权限打开它，消息将是：

```py
I/O error(13): Permission denied
```

这是在捕获异常时格式化输出的有用方式。

`try` - `except`组合可以通过可选的`else`和`finally`块进行扩展。使用`else`的一个示例可以在第十三章的*测试二分查找算法*部分中看到，*测试*。将`try`与`finally`结合使用，当需要在结束时执行清理工作时有用的构造：

确保文件正确关闭的一个示例：

```py
try:
    f = open('data.txt', 'r')
    # some function that does something with the file
    process_file_data(f) 
except: 
    ... 
finally:
    f.close()
```

这将确保无论在处理文件数据时抛出什么异常，文件最终都会被关闭。在`try`语句内部未处理的异常将被保存，并在`finally`块之后抛出。这种组合在`with`语句中使用；请参阅*上下文管理器 - with 语句*部分。

## 用户自定义异常

除了内置的 Python 异常之外，还可以定义自己的异常。这样的用户定义异常应该从`Exception`基类继承。当您定义自己的类，如第十四章*综合示例*部分的*多项式*中的多项式类时，这可能很有用。

看一下这个简单的用户定义异常的例子：

```py
class MyError(Exception):
    def __init__(self, expr):
        self.expr = expr
    def __str__(self):
        return str(self.expr)

try:
   x = random.rand()
   if x < 0.5:
      raise MyError(x)
except MyError as e:
   print("Random number too small", e.expr)
else:
   print(x)
```

生成一个随机数。如果数字小于 0.5，将抛出异常并打印出值过小的消息。如果没有抛出异常，则打印该数字。

在此示例中，您还看到了在`try`语句中使用`else`的一个案例。如果没有发生异常，则将执行`else`下的块。

建议您使用以`Error`结尾的名称定义您的异常，就像标准内置异常的命名一样。

## 上下文管理器 - with 语句

在 Python 中，有一个非常有用的构造，用于简化在处理上下文（如文件或数据库）时的异常处理。该语句将`try ... finally`结构封装在一个简单的命令中。以下是一个使用`with`读取文件的示例：

```py
with open('data.txt', 'r') as f:
    process_file_data(f)
```

这将尝试打开文件，在文件上运行指定的操作（例如，读取），然后关闭文件。如果在执行`process_file_data`过程中出现任何错误，文件将正确关闭，然后抛出异常。这相当于：

```py
f = open('data.txt', 'r')
try: 
    # some function that does something with the file 
    process_file_data(f) 
except:
    ... 
finally:
    f.close()
```

我们将在第十二章*输入和输出*的*文件处理*部分中使用此选项来读取和写入文件。

上述文件读取示例是使用上下文管理器的示例。上下文管理器是具有两个特殊方法的 Python 对象，`_enter_` 和 `_exit_`。任何实现了这两个方法的类对象都可以用作上下文管理器。在这个例子中，文件对象 `f` 是一个上下文管理器，因为它有 `f._enter_` 和 `f._exit_` 方法。

`_enter_` 方法应该实现初始化指令，例如，打开文件或数据库连接。如果此方法有返回语句，则使用 `as` 构造来访问返回的对象。否则，省略 `as` 关键字。`_exit_` 方法包含清理指令，例如，关闭文件或提交事务并关闭数据库连接。有关更多解释和自定义上下文管理器的示例，请参阅第十三章 *使用上下文管理器计时* 的 *测试* 部分，第十三章。

有一些 NumPy 函数可以用作上下文管理器。例如，`load` 函数支持某些文件格式的上下文管理器。NumPy 的函数 `errstate` 可以用作上下文管理器来指定代码块内的浮点错误处理行为。

这是一个使用 errstate 和上下文管理器的示例：

```py
import numpy as np      # note, sqrt in NumPy and SciPy 
                        # behave differently in that example
with errstate(invalid='ignore'):
    print(np.sqrt(-1)) # prints 'nan'

with errstate(invalid='warn'):
    print(np.sqrt(-1)) # prints 'nan' and 
                   # 'RuntimeWarning: invalid value encountered in sqrt'

with errstate(invalid='raise'):
    print(np.sqrt(-1)) # prints nothing and raises FloatingPointError
```

有关此示例的更多详细信息，请参阅第二章 *变量和基本类型* 的 *无限和非数字* 部分，以及第十三章 *使用上下文管理器计时* 的 *测试* 部分，第十三章。

# 查找错误：调试

软件代码中的错误有时被称为 *bug*。调试是查找和修复代码中错误的过程。这个过程可以在不同程度上进行。最有效的方法是使用一个名为调试器的工具。设置单元测试是一个很好的方法来早期识别错误，请参阅第十三章 *使用单元测试* 的 *测试* 部分，第十三章。当问题不明显时，调试器非常有用。

## 错误

通常有两种类型的错误：

+   发生异常但没有被捕获。

+   代码无法正常工作。

第一种情况通常更容易修复。第二种情况可能更困难，因为问题可能是一个错误的想法或解决方案，一个错误的实现，或者两者的组合。

在以下内容中，我们只关注第一种情况，但可以使用相同的工具来帮助找出代码为什么没有按预期执行。

## 栈

当发生异常时，你会看到调用栈。调用栈包含所有调用异常发生代码的函数的跟踪。

一个简单的栈示例：

```py
def f():
   g()
def g():
   h()
def h():
   1//0

f()
```

在这种情况下，栈是 `f`、`g` 和 `h`。运行此段代码生成的输出如下：

```py
Traceback (most recent call last):
  File "stack_example.py", line 11, in <module>
    f() 
  File "stack_example.py", line 3, in f
    g() 
  File "stack_example.py", line 6, in g
    h() File "stack_example.py", line 9, in h
    1//0 
ZeroDivisionError: integer division or modulo by zero
```

错误被打印出来。显示导致错误的函数调用序列。第 11 行上的函数 `f` 被调用，它反过来调用 `g`，然后是 `h`。这导致了 `ZeroDivisionError`。

调试堆栈报告了程序执行中某个特定点的活动堆栈。调试堆栈让您能够追踪到给定点的函数调用序列。通常这发生在未捕获的异常被抛出之后。这有时被称为事后分析，此时调试堆栈点就是异常发生的地方。另一种选择是手动调用调试堆栈来分析您怀疑存在错误的代码片段，可能是在异常发生之前。

## Python 调试器

Python 自带一个名为 pdb 的内置调试器。一些开发环境将调试器集成其中。以下过程在大多数情况下仍然适用。

使用调试器的最简单方法是启用您想要调查的代码点的堆栈跟踪。以下是基于第七章中 *返回值* 部分 *函数* 示例触发调试器的简单示例：

```py
import pdb

def complex_to_polar(z):
    pdb.set_trace() 
    r = sqrt(z.real ** 2 + z.imag ** 2)
    phi = arctan2(z.imag, z.real)
    return (r,phi)
z = 3 + 5j 
r,phi = complex_to_polar(z)

print(r,phi)
```

`pdb.set_trace()` 命令启动调试器并启用后续命令的跟踪。前面的代码将显示如下：

```py
> debugging_example.py(7)complex_to_polar()
-> r = sqrt(z.real ** 2 + z.imag ** 2) 
(Pdb)
```

调试器提示符用 (Pdb) 表示。调试器停止程序执行并提供一个提示，让您可以检查变量、修改变量、单步执行命令等。

每一步都会打印当前行，因此您可以跟踪您所在的位置以及接下来会发生什么。使用命令 `n` (next) 进行单步执行：

```py
> debugging_example.py(7)complex_to_polar() 
-> r = sqrt(z.real ** 2 + z.imag ** 2) 
(Pdb) n 
> debugging_example.py(8)complex_to_polar() 
-> phi = arctan2(z.imag, z.real) 
(Pdb) n 
> debugging_example.py(9)complex_to_polar() 
-> return (r,phi) 
(Pdb) 
...
```

命令 `n` (next) 将继续到下一行并打印该行。如果您需要同时查看多行，列表命令 `l` (list) 将显示当前行及其周围的代码：

在调试器中列出周围的代码：

```py
> debugging_example.py(7)complex_to_polar() 
-> r = sqrt(z.real ** 2 + z.imag ** 2) 
(Pdb) l
  2
  3 import pdb
  4
  5 def complex_to_polar(z):
  6 pdb.set_trace()
  7 -> r = sqrt(z.real ** 2 + z.imag ** 2)
  8 phi = arctan2(z.imag, z.real)
  9 return (r,phi)
 10
 11 z = 3 + 5j
 12 r,phi = complex_to_polar(z) 
(Pdb)
```

通过使用命令 `p` (print) 后跟变量名称，可以将变量的值打印到控制台以进行检查。打印变量的示例：

```py
> debugging_example.py(7)complex_to_polar() 
-> r = sqrt(z.real ** 2 + z.imag ** 2) 
(Pdb) p z 
(3+5j) (Pdb) n 
> debugging_example.py(8)complex_to_polar() 
-> phi = arctan2(z.imag, z.real) 
(Pdb) p r 
5.8309518948453007 
(Pdb) c 
(5.8309518948453007, 1.0303768265243125)
```

`p` (print) 命令将打印变量；命令 `c` (continue) 继续执行。

在执行过程中更改变量是有用的。只需在调试器提示符下分配新值，然后单步执行或继续执行。

变量更改的示例：

```py
> debugging_example.py(7)complex_to_polar() 
-> r = sqrt(z.real ** 2 + z.imag ** 2) 
(Pdb) z = 2j 
(Pdb) z 
2j 
(Pdb) c 
(2.0, 1.5707963267948966)
```

在这里，变量 `z` 被分配了一个新值，该值将在剩余的代码中使用。请注意，最终的打印输出已更改。

## 概述 - 调试命令

在 *表 10.2* 中，显示了最常见的调试命令。有关命令的完整列表和描述，(请参阅文档 [[25]](apa.html "附录 . 参考文献") 以获取更多信息)。请注意，任何 Python 命令也都适用，例如，将值分配给变量。

### 小贴士

**短变量名**

如果您想检查与调试器的任何简短命令名称相同的变量，例如 `h`，您必须使用 `!h` 来显示该变量。

| **命令** | **动作** |
| --- | --- |
| `h` | 帮助（没有参数时，它打印可用的命令） |
| `l` | 列出当前行周围的代码 |
| `q` | 退出（退出调试器并停止执行） |
| `c` | 继续执行 |
| `r` | 继续执行直到当前函数返回 |
| `n` | 继续执行直到下一行 |
| `p <expression>` | 在当前上下文中评估并打印表达式 |

表 10.2：调试器中最常见的调试命令。

## IPython 中的调试

IPython 自带一个名为`ipdb`的调试器版本。在撰写本书时，这些差异非常微小，但可能会发生变化。

IPython 中有一个命令，在发生异常时自动打开调试器。这在尝试新想法或代码时非常有用。以下是如何在 IPython 中自动打开调试器的示例：

```py
In [1]: %pdb # this is a so - called IPython magic command 
Automatic pdb calling has been turned ON

In [2]: a = 10

In [3]: b = 0

In [4]: c = a/b
___________________________________________________________________
ZeroDivisionError                  Traceback (most recent call last) 
<ipython-input-4-72278c42f391> in <module>() 
—-> 1 c = a/b

ZeroDivisionError: integer division or modulo by zero 
> <ipython-input-4-72278c42f391>(1)<module>()
      -1 c = a/b
ipdb>
```

在 IPython 提示符下的 IPython 魔法命令`%pdb`在抛出异常时自动启用调试器。在这里，调试器提示符显示`ipdb`以指示调试器正在运行。

# 摘要

本章的关键概念是异常和错误。我们展示了如何抛出异常以便稍后在另一个程序单元中捕获。你可以定义自己的异常，并为其配备消息和给定变量的当前值。

代码可能会在没有抛出异常的情况下返回意外的结果。定位错误结果来源的技术称为调试。我们介绍了调试方法，并希望鼓励你训练这些方法，以便在需要时能够随时使用。严重调试的需求可能比你预期的要早。
