错误和异常处理

在这一章中，我们将讨论错误和异常，以及如何查找和修复它们。处理异常是编写可靠且易用代码的重要部分。我们将介绍基本的内置异常，并展示如何使用和处理异常。我们还将介绍调试，并展示如何使用内置的 Python 调试器。

在本章中，我们将讨论以下主题：

+   什么是异常？

+   查找错误：调试

# 第十三章：12.1 什么是异常？

程序员（即使是有经验的程序员）最先遇到的错误是代码语法不正确，即代码指令格式不正确。

考虑这个语法错误的示例：

```py
>>> for i in range(10)
  File “<stdin>”, line 1
    for i in range(10)
                      ^
SyntaxError: invalid syntax
```

错误发生是因为 `for` 声明末尾缺少冒号。这是引发异常的一个示例。在 `SyntaxError` 的情况下，它告诉程序员代码语法错误，并且还会打印出发生错误的行，箭头指向该行中问题所在的位置。

Python 中的异常是从一个基类 `Exception` 派生（继承）而来的。Python 提供了许多内置异常。一些常见的异常类型列在 *表 12.1* 中。

这里有两个常见的异常示例。如你所料，`ZeroDivisionError` 是在尝试除以零时引发的：

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
| `IndexError` | 索引超出范围，例如，当 `v` 只有五个元素时，尝试访问 `v[10]`。 |
| `KeyError` | 引用未定义的字典键。 |
| `NameError` | 未找到名称，例如，未定义的变量。 |
| `LinAlgError` | `linalg` 模块中的错误，例如，在求解含有奇异矩阵的系统时。 |
| `ValueError` | 不兼容的数据值，例如，使用不兼容的数组进行 `dot` 运算。 |
| `IOError` | I/O 操作失败，例如，`文件未找到`。 |
| `ImportError` | 模块或名称在导入时未找到。 |

表 12.1：一些常用的内置异常及其含义

除以零时会引发 `ZeroDivisionError` 并打印出文件名、行号和发生错误的函数名。

如我们之前所见，数组只能包含相同数据类型的元素。如果你尝试赋值为不兼容的类型，将引发 `ValueError`。一个值错误的示例是：

```py
>>> a = arange(8.0) 
>>> a 
array([ 0., 1., 2., 3., 4., 5., 6., 7.]) 
>>> a[3] = 'string'
Traceback (most recent call last): 
  File "<stdin>", line 1, in <module>
ValueError: could not convert string to float: string
```

这里，`ValueError` 被引发，因为数组包含浮点数，而元素不能赋值为字符串。

## 12.1.1 基本原则

让我们看看如何通过使用 `raise` 引发异常并通过 `try` 语句捕获异常的基本原则。

### 引发异常

创建错误称为引发异常。你在前一部分中看到了异常的一些示例。你也可以定义自己的异常，使用预定义类型的异常或使用未指定类型的异常。引发异常的命令如下：

```py
raise Exception("Something went wrong")
```

这里引发了一个未指定类型的异常。

当某些事情出错时，可能会诱使你打印出错误信息，例如，像这样：

```py
print("The algorithm did not converge.")
```

这不推荐使用，原因有多个。首先，打印输出很容易被忽视，特别是当信息被埋在控制台打印的其他许多消息中时。其次，更重要的是，这会让你的代码无法被其他代码使用。调用代码不会*读取*你打印的内容，并且无法知道发生了错误，因此也无法处理它。

由于这些原因，最好改为引发异常。异常应该总是包含描述性消息，例如：

```py
raise Exception("The algorithm did not converge.")
```

这个消息会清晰地显示给用户。它还为调用代码提供了一个机会，使其知道发生了错误，并可能找到解决方法。

这里是一个典型的示例，用于在函数内检查输入，确保它在继续之前是可用的。简单地检查负值和正确的数据类型，确保函数的输入符合计算阶乘的预期：

```py
def factorial(n):
    if not (isinstance(n, (int, int32, int64))):
       raise TypeError("An integer is expected")
    if not (n >=0): 
       raise ValueError("A positive number is expected")
```

如果给定了不正确的输入，函数的用户会立即知道是什么错误，而用户有责任处理该异常。请注意，在抛出预定义异常类型时，使用异常名称，在这个例子中是`ValueError`，后面跟着消息。通过指定异常类型，调用代码可以根据引发的错误类型决定如何不同地处理错误。

总结来说，抛出异常总比打印错误信息更好。

### 捕获异常

处理异常被称为*捕获异常*。检查异常是通过`try`和`except`命令来完成的。

异常会停止程序的执行流程，并查找最近的`try`封闭块。如果异常没有被捕获，程序单元会被跳出，并继续向调用栈中更高层的程序单元查找下一个封闭的`try`块。如果没有找到任何块并且异常没有被处理，程序执行会完全停止，并显示标准的回溯信息。

让我们看看之前的阶乘示例，并用`try`语句来使用它：

```py
n=-3
try:
    print(factorial(n))
except ValueError:
    print(factorial(-n))   # Here we catch the error

```

在这种情况下，如果`try`块中的代码引发了`ValueError`类型的错误，异常将会被捕获，并且执行`except`块中的操作。如果`try`块中没有发生异常，`except`块会完全跳过，程序继续执行。

`except`语句可以捕获多个异常。这是通过将它们简单地组合成一个元组来完成的，例如：

```py
except (RuntimeError, ValueError, IOError):
```

`try`块也可以有多个`except`语句。这使得可以根据异常类型不同来分别处理异常。让我们看一下另一个多个异常类型的示例：

```py
try: 
    f = open('data.txt', 'r') 
    data = f.readline() 
    value = float(data) 
except FileNotFoundError as FnF: 
    print(f'{FnF.strerror}: {FnF.filename}') 
except ValueError: 
    print("Could not convert data to float.")
```

在这里，如果文件不存在，`FileNotFoundError`会被捕获；如果文件的第一行数据与浮动数据类型不兼容，`ValueError`会被捕获。

在这个示例中，我们通过关键字`as`将`FileNotFoundError`赋值给变量`FnF`。这允许在处理此异常时访问更多的详细信息。在这里，我们打印了错误字符串`FnF.strerror`和相关文件的名称`FnF.filename`。每种错误类型可以根据类型有自己的属性集。如果名为`data.txt`的文件不存在，在上面的示例中，消息将是：

```py
No such file or directory: data.txt
```

这是在捕获异常时格式化输出的一个有用方法。

`try`-`except`组合可以通过可选的`else`和`finally`块进行扩展。

使用`else`的一个示例可以在第 15.2.1 节中看到：*测试二分法算法*。将`try`与`finally`结合使用，在需要在结束时进行清理工作的情况下，提供了一个有用的结构。通过一个确保文件正确关闭的示例来说明：

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

这将确保无论在处理文件数据时抛出什么异常，文件都会在结束时关闭。`try`语句内部未处理的异常会在`finally`块之后保存并抛出。这个组合在`with`语句中使用；参见第 12.1.3 节：*上下文管理器——`with`语句*。

## 12.1.2 用户定义异常

除了内置的 Python 异常外，还可以定义自己的异常。这样的用户定义异常应继承自基类`Exception`。当你定义自己的类时，这会非常有用，例如在第 19.1 节中定义的多项式类。

看看这个简单的用户定义异常的小示例：

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

生成一个随机数。如果该数字小于`0.5`，则会抛出一个异常，并打印一个值太小的消息。如果没有抛出异常，则打印该数字。

在这个示例中，你还看到了在`try`语句中使用`else`的一个例子。如果没有发生异常，`else`下的代码块将会被执行。

建议你为你的异常定义以`Error`结尾的名称，就像标准内置异常的命名一样。

## 12.1.3 上下文管理器——`with`语句

Python 中有一个非常有用的结构，在处理文件或数据库等上下文时简化异常处理。该语句将`try ... finally`结构封装为一个简单的命令。以下是使用`with`读取文件的示例：

```py
with open('data.txt', 'w') as f:
    process_file_data(f)
```

这将尝试打开文件，在文件上运行指定的操作（例如，读取），然后关闭文件。如果在执行`process_file_data`期间出现任何问题，文件将被正确关闭，然后抛出异常。这等同于：

```py
f = open('data.txt', 'w')
try: 
    # some function that does something with the file 
    process_file_data(f) 
except:
    ... 
finally:
    f.close()
```

我们将在第 14.1 节中使用此选项：*文件处理*，在读取和写入文件时使用。

前面的文件读取示例是使用上下文管理器的一个例子。上下文管理器是具有两个特殊方法`__enter__`和`__exit__`的 Python 对象。任何实现了这两个方法的类的对象都可以用作上下文管理器。在此示例中，文件对象`f`是一个上下文管理器，因为它具有方法`f.__enter__`和`f.__exit__`。

方法`__enter__`应该实现初始化指令，例如打开文件或数据库连接。如果此方法包含返回语句，则通过构造`as`来访问返回的对象。否则，省略关键字`as`。方法`__exit__`包含清理指令，例如关闭文件或提交事务并关闭数据库连接。有关更多解释和自定义上下文管理器的示例，请参见第 15.3.3 节：*使用上下文管理器进行计时*。

有一些 NumPy 函数可以用作上下文管理器。例如，函数`load`支持某些文件格式的上下文管理器。NumPy 的函数`errstate`可以作为上下文管理器，用于在代码块中指定浮点错误处理行为。

下面是使用`errstate`和上下文管理器的示例：

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

请参见第 2.2.2 节：*浮动点数*，了解更多此示例的详细信息，并查看第 15.3.3 节：*使用上下文管理器进行计时*以获得另一个示例。

# 12.2 查找错误：调试

软件代码中的错误有时被称为 bug。调试是找到并修复代码中的 bug 的过程。这个过程可以在不同的复杂度下进行。最有效的方式是使用名为调试器的工具。提前编写单元测试是识别错误的好方法；请参见第 15.2.2 节：*使用 unittest 包*。当问题所在和问题是什么不明显时，调试器非常有用。

## 12.2.1 Bugs

通常有两种类型的 bug：

+   异常被引发，但未被捕获。

+   代码无法正常运行。

第一个情况通常比较容易修复。第二种情况可能更难，因为问题可能是一个错误的想法或解决方案、错误的实现，或两者的结合。

接下来我们只关注第一个情况，但同样的工具也可以帮助找出为什么代码没有按预期执行。

## 12.2.2 栈

当异常被引发时，你会看到调用栈。调用栈包含所有调用异常发生代码的函数的追踪信息。

一个简单的栈示例是：

```py
def f():
   g()
def g():
   h()
def h():
   1//0

f()
```

在这种情况下，栈是`f`，`g`和`h`。运行这段代码生成的输出如下所示：

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

错误已打印。导致错误的函数序列已显示。`line 11`上的函数`f`被调用，接着调用了`g`，然后是`h`。这导致了`ZeroDivisionError`。

堆栈跟踪报告程序执行某一时刻的活动堆栈。堆栈跟踪可以让你追踪到某一时刻调用的函数序列。通常这是在抛出未捕获的异常之后。这有时被称为事后分析，堆栈跟踪点就是异常发生的位置。另一种选择是手动调用堆栈跟踪来分析你怀疑有错误的代码片段，可能是在异常发生之前。

在以下示例中，引发异常以引发堆栈跟踪的生成：

```py
def f(a):
   g(a)
def g(a):
   h(a)
def h(a):
   raise Exception(f'An exception just to provoke a strack trace and a value a={a}')

f(23)
```

这将返回以下输出：

```py
Traceback (most recent call last):

  File ".../Python_experiments/manual_trace.py", line 17, in <module>
    f(23)

  File "../Python_experiments/manual_trace.py", line 11, in f
    g(a)

  File "../Python_experiments/manual_trace.py", line 13, in g
    h(a)

  File "/home/claus/Python_experiments/manual_trace.py", line 15, in h
    raise Exception(f'An exception just to provoke a strack trace and a value a={a}')

Exception: An exception just to provoke a strack trace and a value a=23
```

## 12.2.3 Python 调试器

Python 自带有一个内置调试器，叫做`pdb`。一些开发环境中集成了调试器。即使在这些情况下，以下过程依然适用。

使用调试器的最简单方法是在代码中你想调查的地方启用堆栈跟踪。这里是一个基于第 7.3 节中的示例触发调试器的简单示例：*返回值*：

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

命令`pdb.set_trace()`启动调试器并启用后续命令的跟踪。前面的代码将显示如下：

```py
> debugging_example.py(7)complex_to_polar()
-> r = sqrt(z.real ** 2 + z.imag ** 2) 
(Pdb)
```

调试器提示符由`(Pdb)`表示。调试器暂停程序执行，并提供一个提示符，允许你检查变量、修改变量、逐步执行命令等。

每一步都会打印当前行，因此你可以跟踪当前所在的位置以及接下来会发生什么。逐步执行命令可以通过命令`n`（下一个）完成，像这样：

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

命令`n`（下一个）将继续到下一行并打印该行。如果你需要同时查看多行，命令`l`（列出）将显示当前行及其周围的代码：

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

变量的检查可以通过使用命令`p`（打印）后跟变量名，将其值打印到控制台来完成。打印变量的示例是：

```py
> debugging_example.py(7)complex_to_polar() 
-> r = sqrt(z.real ** 2 + z.imag ** 2) 
(Pdb) p z 
(3+5j) 
(Pdb) n 
> debugging_example.py(8)complex_to_polar() 
-> phi = arctan2(z.imag, z.real) 
(Pdb) p r 
5.8309518948453007 
(Pdb) c 
(5.8309518948453007, 1.0303768265243125)
```

命令`p`（打印）将打印变量；命令`c`（继续）将继续执行。

在执行过程中修改变量是很有用的。只需在调试器提示符下分配新值，并逐步执行或继续执行：

```py
> debugging_example.py(7)complex_to_polar() 
-> r = sqrt(z.real ** 2 + z.imag ** 2) 
(Pdb) z = 2j 
(Pdb) z 
2j 
(Pdb) c 
(2.0, 1.5707963267948966)
```

在这里，变量`z`被赋予一个新值，并在剩余的代码中使用。请注意，最终的打印输出已发生变化。

## 12.2.4 概述 – 调试命令

在*表 12.2*中，显示了最常用的调试命令。有关完整的命令列表和描述，请参阅文档以获取更多信息[24]。请注意，任何 Python 命令也都有效，例如，为变量赋值。

如果你想检查一个与调试器短命令重名的变量，例如`h`，你必须使用`!h`来显示该变量。

| **命令** | **操作** |
| --- | --- |
| `h` | 帮助（不带参数时，显示可用的命令） |
| `l` | 列出当前行周围的代码 |
| `q` | 退出（退出调试器，停止执行） |
| `c` | 继续执行 |
| `r` | 继续执行，直到当前函数返回 |
| `n` | 继续执行，直到下一行 |
| `p <expression>` | 计算并打印当前上下文中的表达式 |

表 12.2：调试器中最常用的调试命令

## 12.2.5 IPython 中的调试

IPython 自带一个调试器版本，称为 `ipdb`。在撰写本书时，`ipdb` 与 `pdb` 之间的差异非常小，但这可能会发生变化。

在 IPython 中有一个命令 `%pdb`，在出现异常时自动启动调试器。当你在实验新想法或代码时，这非常有用。如何在 IPython 中自动启动调试器的一个示例如下：

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

在 IPython 提示符下，IPython 魔法命令 `%pdb` 会在抛出异常时自动启用调试器。在此，调试器提示符会显示 `ipdb`，以表明调试器正在运行。

# 12.3 总结

本章的关键概念是异常和错误。我们展示了如何抛出异常并在另一个程序单元中捕获它。你可以定义自己的异常，并为它们配上消息和当前变量的值。

代码可能会返回意外结果而没有抛出异常。定位错误结果来源的技巧叫做调试。我们介绍了调试方法，并希望能鼓励你训练这些技巧，以便在需要时随时使用。严重的调试需求可能比你预想的更早出现。
