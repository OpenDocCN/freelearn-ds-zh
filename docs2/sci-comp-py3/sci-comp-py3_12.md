# 第十二章 输入和输出

在本章中，我们将介绍处理数据文件的一些选项。根据数据和所需的格式，有几种读取和写入的选项。我们将展示一些最有用的替代方案。

# 文件处理

文件输入输出（输入和输出）在许多场景中是必不可少的。例如：

+   处理测量或扫描的数据。测量数据存储在需要读取以进行分析的文件中。

+   与其他程序交互。将结果保存到文件中，以便可以在其他应用程序中导入，反之亦然。

+   存储信息以供将来参考或比较。

+   与他人共享数据和结果，可能是在其他平台使用其他软件的情况下。

在本节中，我们将介绍如何在 Python 中处理文件输入输出。

## 与文件交互

在 Python 中，类型为`file`的对象代表存储在磁盘上的物理文件的 内容。可以使用以下语法创建一个新的`file`对象：

```py
myfile = open('measurement.dat','r') # creating a new file object from an existing file
```

可以使用以下方式访问文件内容，例如：

```py
print(myfile.read())
```

使用文件对象需要小心。问题是文件在可以重新读取或被其他应用程序使用之前必须关闭，这可以通过以下语法完成：

```py
myfile.close() # closes the file object
```

然而，这并不简单，因为可能在执行`close`调用之前触发异常，这将跳过关闭代码（考虑以下示例）。确保文件正确关闭的简单方法是用上下文管理器。这种使用`with`关键字的构造在第十章的*异常*部分有更详细的解释，*错误处理*。以下是它与文件一起使用的方式：

```py
with open('measurement.dat','r') as myfile: 
     ... # use myfile here
```

这确保了当退出`with`块时文件将被关闭，即使块内抛出异常。该命令与上下文管理器对象一起工作。我们建议您阅读第十章的*异常*部分了解更多关于上下文管理器的信息，*错误处理*。以下是一个示例，说明为什么`with`构造是可取的：

```py
myfile = open(name,'w')
myfile.write('some data')
a = 1/0
myfile.write('other data')
myfile.close()
```

在关闭文件之前抛出异常。文件保持打开状态，无法保证文件中写入的数据或写入的时间。因此，实现相同结果的正确方式是：

```py
with open(name,'w') as myfile:
    myfile.write('some data')
    a = 1/0
    myfile.write('other data')
```

在这种情况下，文件在异常（这里为`ZeroDivisionError`）被引发后立即干净地关闭。注意，也没有必要显式关闭文件。

## 文件是可迭代的

一个文件特别地是可迭代的（参考第九章的*迭代器*部分第九章, *迭代*）。文件通过迭代其行：

```py
with open(name,'r') as myfile:
    for line in myfile:
        data = line.split(';')
        print('time {} sec temperature {} C'.format(data[0],data[1]))
```

文件行被返回为字符串。字符串方法`split`是将其转换为字符串列表的可能工具。例如：

```py
data = 'aa;bb;cc;dd;ee;ff;gg'
data.split(';') # ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg']

data = 'aa bb cc dd ee ff gg'
data.split(' ') # ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg']
```

由于`myfile`对象是可迭代的，我们也可以直接将其提取到列表中，如下所示：

```py
data = list(myfile)
```

## 文件模式

如您在这些文件处理示例中所见，`open` 函数至少需要两个参数。第一个显然是文件名，第二个是一个字符串，描述了文件将被如何使用。有几种这样的模式用于打开文件；基本的有：

```py
with open('file1.dat','r') as ...  # read only
with open('file2.dat','r+') as ...  # read/write
with open('file3.dat','rb') as ...  # read in byte mode  
with open('file4.dat','a') as ...  # append (write to the end of the file)
with open('file5.dat','w') as ... # (over-)write the file
with open('file6.dat','wb') as ... # (over-)write the file in byte mode
```

`'r'`、`'r+'` 和 `'a'` 模式要求文件存在，而 `'w'` 如果不存在具有该名称的文件，则会创建一个新文件。使用 `'r'` 和 `'w'` 进行读取和写入是最常见的，正如您在之前的例子中所看到的。

考虑一个使用 append `'a'` 模式打开文件并在文件末尾添加数据而不修改现有内容的例子。注意换行符，`\n` :

```py
with open('file3.dat','a') as myfile:
    myfile.write('something new\n')
```

# NumPy 方法

NumPy 有内置的方法用于将 NumPy 数组数据读取和写入文本文件。这些是 `numpy.loadtxt` 和 `numpy.savetxt`。

## savetxt

将数组写入文本文件很简单：

```py
savetxt(filename,data)
```

有两个作为字符串给出的有用参数，`fmt` 和 `delimiter`，它们控制列之间的格式和分隔符。默认的分隔符是空格，格式是 `%.18e`，这对应于包含所有数字的指数格式。格式化参数的使用方法如下：

```py
x = range(100) # 100 integers
savetxt('test.txt',x,delimiter=',')   # use comma instead of space
savetxt('test.txt',x,fmt='%d') # integer format instead of float with e
```

## loadtxt

从文本文件读取到数组是通过以下语法完成的：

```py
filename = 'test.txt'
data = loadtxt(filename)
```

由于数组中的每一行必须具有相同的长度，因此文本文件中的每一行必须具有相同数量的元素。类似于 `savetxt`，默认值是 `float` 和分隔符是 `space`。这些可以通过 `dtype` 和 `delimiter` 参数设置。另一个有用的参数是 `comments`，它可以用来标记数据文件中使用的注释符号。以下是一个使用格式化参数的例子：

```py
data = loadtxt('test.txt',delimiter=';')    # data separated by semicolons
data = loadtxt('test.txt',dtype=int,comments='#') # read to integer type, 
                                               #comments in file begin with a hash character
```

# Pickling

您刚才看到的读取和写入方法在写入之前将数据转换为字符串。复杂类型（如对象和类）不能这样写入。使用 Python 的 pickle 模块，您可以保存任何对象，也可以将多个对象保存到文件中。

数据可以以纯文本（ASCII）格式或使用稍微更有效的二进制格式保存。有两种主要方法：`dump`，它将 Python 对象的 pickled 表示保存到文件中，以及 `load`，它从文件中检索 pickled 对象。基本用法如下：

```py
import pickle
with open('file.dat','wb') as myfile:
    a = random.rand(20,20)
    b = 'hello world'
    pickle.dump(a,myfile)    # first call: first object
    pickle.dump(b,myfile)    # second call: second object

import pickle
with open('file.dat','rb') as myfile:
    numbers = pickle.load(myfile) # restores the array
    text = pickle.load(myfile)    # restores the string
```

注意返回的两个对象的顺序。除了两种主要方法外，有时将 Python 对象序列化为字符串而不是文件也是有用的。这是通过 `dumps` 和 `load` 来实现的。考虑一个序列化数组和字典的例子：

```py
a = [1,2,3,4]
pickle.dumps(a) # returns a bytes object
b = {'a':1,'b':2}
pickle.dumps(b) # returns a bytes object
```

使用`dumps`的一个好例子是当你需要将 Python 对象或 NumPy 数组写入数据库时。这些通常支持存储字符串，这使得在没有特殊模块的情况下写入和读取复杂的数据和对象变得容易。除了 pickle 模块外，还有一个称为`cPickle`的优化版本。它是用 C 编写的，如果你需要快速读写，这是一个选项。pickle 和*cPickle*生成相同的数据，可以互换使用。

# Shelving

字典中的对象可以通过键访问。有一种类似的方法可以通过首先分配一个键来访问文件中的特定数据。这可以通过使用 shelve 模块来实现：

```py
from contextlib import closing
import shelve as sv
# opens a data file (creates it before if necessary)
with closing(sv.open('datafile')) as data:
    A = array([[1,2,3],[4,5,6]])     
    data['my_matrix'] = A  # here we created a key
```

在*文件处理*部分，我们看到了内置的`open`命令生成上下文管理器，并看到了为什么这对于处理外部资源（如文件）很重要。与此命令相反，`sv.open`本身不会创建上下文管理器。需要`contextlib`模块的`closing`命令将其转换为适当的上下文管理器。考虑以下恢复文件的示例：

```py
from contextlib import closing
import shelve as sv
with closing(sv.open('datafile')) as data: # opens a data file
    A = data['my_matrix']  # here we used the key
    ...
```

shelving 对象具有所有字典方法，例如键和值，并且可以像字典一样使用。请注意，只有在调用`close`或`sync`方法后，更改才会写入文件。

# 读取和写入 Matlab 数据文件

SciPy 模块具有使用模块读取和写入 Matlab 的`.mat`文件格式的功能。命令是`loadmat`和`savemat`。要加载数据，请使用以下语法：

```py
import scipy.io
data = scipy.io.loadmat('datafile.mat')
```

变量 data 现在包含一个字典，键对应于`.mat`文件中保存的变量名称。变量以 NumPy 数组格式存在。保存到`.mat`文件涉及创建一个包含您想要保存的所有变量（变量名和值）的字典。然后命令是`savemat`：

```py
data = {}
data['x'] = x
data['y'] = y
scipy.io.savemat('datafile.mat',data)
```

这将使用相同的名称将 NumPy 数组`x`和`y`保存到 Matlab 中。

# 读取和写入图像

SciPy 附带了一些处理图像的基本函数。模块函数将图像读取到 NumPy 数组中。该函数将数组保存为图像。以下将读取*JPEG*图像到数组，打印形状和类型，然后创建一个新的具有调整大小图像的新数组，并将新图像写入文件：

```py
import scipy.misc as sm

# read image to array
im = sm.imread("test.jpg") 
print(im.shape)   # (128, 128, 3)
print(im.dtype)   # uint8

# resize image
im_small = sm.imresize(im, (64,64))
print(im_small.shape)   # (64, 64, 3)

# write result to new image file
sm.imsave("test_small.jpg", im_small)
```

注意数据类型。图像几乎总是以*0...255*范围内的像素值存储为 8 位无符号整数。第三个形状值显示了图像有多少个颜色通道。在这种情况下，*3*表示它是一个具有以下顺序存储值的彩色图像：红色`im[0]`，绿色`im[1]`，蓝色`im[2]`。灰度图像将只有一个通道。

对于图像处理，SciPy 模块的`scipy.misc`包含许多有用的基本图像处理函数，例如滤波、转换和测量。

# 概述

在处理测量和其他大量数据来源时，文件处理是不可避免的。与其他程序和工具的通信也是通过文件处理完成的。

你学会了将文件视为一个像其他对象一样的 Python 对象，它具有诸如`readlines`和`write`等重要方法。我们展示了如何通过特殊属性来保护文件，这些属性可能只允许读取或只允许写入访问。

你写入文件的方式通常会影响到处理速度。我们看到了数据是如何通过序列化（pickling）或使用`shelve`方法进行存储的。
