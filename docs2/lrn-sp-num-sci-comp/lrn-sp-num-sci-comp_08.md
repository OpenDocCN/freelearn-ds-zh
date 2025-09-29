# 第八章。与其他语言的交互

我们经常需要将不同语言的代码整合到我们的工作流程中；主要是 C/C++ 或 Fortran，以及来自 R、MATLAB 或 Octave 的代码。Python 优秀地允许所有这些其他来源的代码在内部运行；必须注意将不同的数值类型转换为 Python 可以理解的形式，但这几乎是我们遇到唯一的难题。

如果您正在使用 SciPy，那是因为您的 Python 生态系统中有可用的 C 和 Fortran 程序编译器。否则，SciPy 就无法安装到您的系统上。鉴于其流行程度，您的计算机环境很可能有 MATLAB/Octave。因此，这导致了本章后面列出主题的选择。我们将留给感兴趣的读者去了解如何与 R 和许多其他软件进行接口，这些软件可用于数值计算。使用 R 的两种替代方案是包 **PypeR** ([`bioinfo.ihb.ac.cn/softwares/PypeR/`](http://bioinfo.ihb.ac.cn/softwares/PypeR/)) 和 **rpy2** ([`rpy.sourceforge.net/`](http://rpy.sourceforge.net/))。其他替代方案可以在 [`stackoverflow.com/questions/11716923/python-interface-for-r-programming-language`](http://stackoverflow.com/questions/11716923/python-interface-for-r-programming-language) 找到。

在本章中，我们将涵盖以下内容：

+   简要讨论如何使用 Python 运行 Fortran、C/C++ 和 MATLAB/Octave 的代码

+   我们将首先了解实用程序 `f2py` 的基本功能，以通过 SciPy 处理在 Python 中包含 Fortran 代码。

+   使用 `scipy.weave` 模块提供的工具在 Python 代码中包含 C/C++ 代码的基本用法

通过简单的示例来展示这些例程，您可以通过修改与本章对应的 IPython Notebook 来丰富这些示例。

# 与 Fortran 的交互

SciPy 提供了一种简单的方法来包含 Fortran 代码——`f2py`。这是一个与 NumPy 库一起提供的实用程序，当 SciPy 的 `distutils` 可用时才会生效。当我们安装 SciPy 时，这总是成立的。

`f2py` 实用程序应该在 Python 之外运行，它用于从任何 Fortran 文件创建一个可以在我们的会话中轻松调用的 Python 模块。在任何 `*nix` 系统中，我们从终端调用它。在 Windows 上，我们建议您在原生终端中运行它，或者更好的是，通过 `cygwin` 会话运行。

在使用 `f2py` 编译之前，任何 Fortran 代码都需要进行以下三个基本更改：

+   移除所有分配

+   将整个程序转换为一个子程序

+   如果需要将任何特殊内容传递给 `f2py`，我们必须使用注释字符串 `"!f2py"` 或 `"cf2py"` 来添加它。

让我们用一个简单的例子来说明这个过程。以下存储在 `primefactors.f90` 文件中的简单子程序，对任何给定的整数进行质因数分解：

```py
SUBROUTINE PRIMEFACTORS(num, factors, f)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: num  !input number
  INTEGER,INTENT(OUT), DIMENSION((num/2))::factors 
  INTEGER, INTENT(INOUT) :: f
  INTEGER :: i, n
  i = 2  
  f = 1  
  n = num
  DO
    IF (MOD(n,i) == 0) THEN 
      factors(f) = i
      f = f+1
      n = n/i
    ELSE
      i = i+1
    END IF
    IF (n == 1) THEN    
      f = f-1    
      EXIT
    END IF
  END DO
```

由于代码中没有进行任何分配，并且我们直接收到一个子例程，我们可以跳到第三步，但暂时我们不会修改 `f2py` 命令，我们满足于尝试从它创建一个 Python 模块。将此 `primefactors` 子例程包装起来的最快方式是发出以下命令（在由 `%` 指示的 shell 或终端提示符处）：

```py
% f2py –c primefactors.f90 –m primefactors

```

如果一切正常，将创建一个名为 `primefactors.so` 的扩展模块。然后我们可以从 `primefactors` 模块中访问 `primefactors` 例程：

```py
>>> import primefactors
>>> primefactors.primefactors(6,1)

```

输出如下所示：

```py
array([2, 3, 0], dtype=int32)

```

# 与 C/C++ 的交互

技术上，`f2py` 也可以为我们包装 C 代码，但还有更有效的方法来完成这项任务。例如，如果我们需要接口一个非常大的 C 函数库，完成此任务的首选方法是 **简化包装和接口生成器**（**SWIG**）（[`www.swig.org/`](http://www.swig.org/)）。要包装 C++ 代码，根据所需功能和与 Python 交互的方法，我们有几种方法，如 SWIG 或再次使用 `f2py`，但还有 **PyCXX**、**Boost.Python**、**Cython** 或 SciPy 模块：`weave`。当 C 编译器不可用（因此无法以通常的方式链接大量库）时，我们使用 `ctypes`。每当我们将使用 NumPy/SciPy 代码，并且想要快速解决我们的包装/绑定问题时，与 C/C++ 交互的两种最常见方式通常是 Python/C API 和 `weave` 包。

这里简要列举的所有方法都需要一本专著来详细描述，具体来说，就是根据系统和需求绑定包装的繁琐之处的方法，以及它们实现时的注意事项。在本章中，我们想更详细地介绍的是 `weave` 包，更具体地说，是通过 `inline` 例程。此命令接收一个包含一系列命令的字符串（原始或非原始），并通过调用您的 C/C++ 编译器在 Python 中运行它。语法如下：

```py
inline(code, arg_names, local_dict=None, global_dict=None,
           force = 0,
           compiler='',
           verbose = 0,
support_code = None,
           customize=None,
type_factories = None,
auto_downcast=1,
           **kw)
```

让我们来看看不同的参数：

+   `code` 参数是包含要运行的代码的字符串。请注意，此代码不得指定任何类型的 `return` 语句。相反，它应分配一些可以返回给 Python 的结果。

+   `arg_names` 参数是一个包含要发送到 C/C++ 代码的 Python 变量名的字符串列表。

+   `local_dict` 参数是可选的，它必须是一个包含用作 C/C++ 代码局部作用域的值的 Python 字典。

+   `global_dict` 参数也是可选的，它必须是一个包含应作为 C/C++ 代码全局作用域的值的另一个 Python 字典。

+   `force` 参数仅用于调试目的。它也是可选的，只能取两个值——0（默认值）或 1。如果其值设置为 1，则每次调用 `inline` 时都会编译 C/C++ 代码。

+   我们可以使用`compiler`选项指定接管 C/C++代码的编译器。它必须是一个包含 C/C++编译器名称的字符串。

让我们以`inline`例程为例，其中我们使用以下方法来使用`cout`进行文本显示：

```py
>>> import scipy.weave
>>> name = 'Francisco'
>>> pin = 1234
>>> code = 'std::cout << name << "---PIN: " '
>>> code+= '<<std::hex << pin <<std::endl;'
>>> arg_names = ['name','pin']
>>> scipy.weave.inline(code, arg_names)

```

输出如下所示：

```py
Francisco---PIN: 4d2

```

这是一个非常简单的例子，其中不需要外部头文件声明。如果我们希望这样做，那些将放入`support_code`选项中。例如，如果我们希望在 C/C++代码中包含 R 的数学函数并通过`inline`传递，我们需要执行以下步骤：

1.  将 C 函数配置为共享库。在终端会话中，在包含 R 发布的文件夹中，输入以下命令：

    ```py
    % ./configure --enable-R-static-lib --enable-static --with-readline=no

    ```

1.  切换到`src/nmath`中的`standalone`文件夹，完成库的安装。最后，我们应该有一个名为`libRmath.so`的文件，需要从`libpath`字符串指向我们的 Python 会话：

    ```py
    % cd src/nmath/standalone
    % make

    ```

1.  在我们的 Python 会话中，我们使用适当的选项准备`inline`调用。例如，如果我们想调用 R 例程`pbinom`，我们按以下步骤进行：

    ```py
    >>> import scipy.weave 
    >>> support_code= 'extern "C" double pbinom(double x,\ 
     double n, double p, int lower_tail, int log_p);' 
    >>> libpath='/opt/Rlib' #IS THE LOCATION OF LIBRARY libRmath.so
    >>> library_dirs=[libpath] 
    >>> libraries=['Rmath'] 
    >>> runtime_library_dirs=[libpath] 
    >>> code='return_val=pbinom(100,20000,100./20000.,0,1);' 
    >>> res=scipy.weave.inline(code, support_code=support_code, \ 
     library_dirs=library_dirs, libraries=libraries, \ 
     runtime_library_dirs=runtime_library_dirs) 
    >>> print(res) 

    ```

    输出如下所示：

    ```py
    -0.747734910363 

    ```

    ### 注意

    注意函数声明是在`support_code`中传递的，而不是在代码中。还要注意，每次我们不使用 C++时，此选项都需要以`extern "C"`开头。

1.  如果需要传递额外的头文件，我们使用`header`选项，而不是`support_code`或`code`：

    ```py
    >>> headers = ['<math.h>']

    ```

我们有一些建议。在将不同变量类型从其原始 C/C++格式转换为 Python 理解的形式时，必须小心谨慎。在某些情况下，这需要修改原始的 C/C++代码。但默认情况下，我们不必担心以下 C/C++类型，因为 SciPy 会自动将它们转换为以下表所示的 Python 格式：

| **Python** | `int` | `float` | `complex` | `string` | `list` | `dict` | `tuple` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **C/C++** | `int` | `double` | `std::complex` | `py::string` | `py::list` | `py:dict` | `py::tuple` |

文件类型`FILE*`被发送到 Python 文件。Python 的可调用对象和实例都来自`py::object`。NumPy ndarrays 是从`PyArrayObject*`构建的。对于任何其他要使用的 Python 类型，相应的 C/C++类型必须仔细转换为前面的组合。

这样就足够了。要超越内联函数的简单使用，我们通常创建扩展模块，并将其中的函数编目以供将来使用。

# 与 MATLAB/Octave 的交互

由于这两个数值计算环境都提供了第四代编程语言，我们不建议直接包含这两个环境中的任何代码。在速度、资源使用或编码能力方面都没有任何优势。在极端且罕见的情况下，如果 SciPy 中没有特定的例程，将例程带到我们的会话中的首选方式是从 MATLAB/Octave 代码生成 C 代码，然后使用本章中*与 C/C++交互*部分中建议的任何方法进行封装。

当我们接收由 MATLAB 或 Octave 创建的数据时，情况会有所不同。SciPy 有一个专门的模块来处理这种情况——`scipy.io`。

让我们通过示例来展示。我们从 Octave 开始，在平面上生成一个由 10 个随机点组成的**Delaunay 三角剖分**。

我们将这些点的坐标以及三角剖分中三角形的指针保存到一个名为 data 的 MATLAB 风格文件（版本 7）中：

```py
octave:1> x=rand(1,10);
octave:2> y=rand(size(x));
octave:3> T=delaunay(x,y);
octave:4> save -v7 data x y T

```

我们在这里完成了。然后我们转到我们的 Python 会话，在那里我们恢复文件数据：

```py
>>> from scipy.io import loadmat
>>> datadict = loadmat("data")

```

`datadict`变量包含一个 Python 字典，其中变量的名称作为`keys`，加载的矩阵作为它们对应的值：

```py
>>> datadict.keys()

```

输出如下所示：

```py
['__header__', '__globals__', 'T', 'y', 'x', '__version__']

```

让我们发出`datadict`命令：

```py
>>> datadict['x']

```

输出如下所示：

```py
array([[0.81222999,0.51836246,0.60425982,0.23660352,0.01305779,
 0.0875166,0.77873049,0.70505801,0.51406693,0.65760987]])

```

让我们看看下面的`datadict`命令：

```py
>>> datadict['__header__']

```

输出如下所示：

```py
'MATLAB 5.0 MAT-file, written by Octave 3.2.4, 2012-11-27
 15:45:20 UTC'

```

我们可以将会话中的数据保存为 MATLAB 和 Octave 可以理解的格式。我们使用来自同一模块的`savemat`命令来完成此操作。其语法如下：

```py
savemat(file_name, mdict, appendmat=True, format='5', 
long_field_names=False, do_compression=False,
oned_as=None)
```

`file_name`参数包含将要写入数据的 MATLAB 类型文件的名称。Python 字典`mdict`包含变量名称（作为键）及其对应的数组值。

如果我们希望在文件末尾附加`.mat`扩展名，我们可以在`file_name`变量中这样做，或者通过将`appendmat`设置为`True`。如果我们需要为文件提供长名称（并非所有 MATLAB 版本都接受），我们需要通过将`long_field_names`选项设置为`True`来表示这一点。

我们可以使用`format`选项来指定 MATLAB 的版本。对于 5.0 及以后的版本，我们将其设置为字符串`'5'`，对于 4.0 版本，则设置为字符串`'4'`。

我们可以压缩我们发送的矩阵，并且通过将`do_compression`选项设置为`True`来表示这一点。

最后一个选项非常有趣。它允许我们向 MATLAB/Octave 指示我们的数组是按列读取还是按行读取。将`oned_as`参数设置为字符串`'column'`将我们的数据发送到一列向量集合中。如果我们将其设置为字符串`'row'`，它将数据作为行向量集合发送。如果设置为`None`，则尊重数据写入的格式。

# 摘要

本章介绍了 SciPy 的一个主要优势——能够与其他语言如 C/C++、Fortran、R 和 MATLAB/Octave 交互。要深入了解 Python 与其他语言的接口，您可能需要阅读更多专门的文献，如 *《Cython 编程学习》*，作者 *Philip Herron*，出版社 *Packt Publishing*，或者深入了解 F2PY 的资料，可在 [`docs.scipy.org/doc/numpy/f2py/`](http://docs.scipy.org/doc/numpy/f2py/) 和 [`www.f2py.com/home/references`](http://www.f2py.com/home/references) 找到。更多帮助信息可在 [`wiki.python.org/moin/IntegratingPythonWithOtherLanguages`](https://wiki.python.org/moin/IntegratingPythonWithOtherLanguages) 找到。

如果您已经阅读到这一章，并且是从第一章开始阅读的，那么您应该知道在本章关于 SciPy 的简介中省略了许多主题。本书已经为您提供了足够的背景知识，以进一步强化您使用 SciPy 的技能和能力。要继续学习，请参考 SciPy 参考指南 ([`docs.scipy.org/doc/scipy/reference/`](http://docs.scipy.org/doc/scipy/reference/)) 和其他可用的文档指南 ([`docs.scipy.org/doc/`](http://docs.scipy.org/doc/))。

此外，我们建议您定期阅读并订阅 SciPy 邮件列表 ([`mail.scipy.org/mailman/listinfo/scipy-user`](http://mail.scipy.org/mailman/listinfo/scipy-user))，在那里您可以与世界各地的 SciPy 用户互动，不仅可以通过提问/回答有关 SciPy 的问题，还可以了解 SciPy 的当前趋势，甚至找到与之相关的职位。

您可以浏览该列表历史存档的帖子集合，[`mail.scipy.org/pipermail/scipy-user/`](http://mail.scipy.org/pipermail/scipy-user/)。此外，您应该知道每年都会举办 SciPy 会议 ([`conference.scipy.org/`](http://conference.scipy.org/))，正如他们所说，这允许来自学术、商业和政府机构的参与者展示他们最新的科学 Python 项目，从熟练的用户和开发者那里学习，并在代码开发上进行合作。
