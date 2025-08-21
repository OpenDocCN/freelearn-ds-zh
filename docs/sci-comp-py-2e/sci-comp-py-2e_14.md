输入与输出

在本章中，我们将介绍一些处理数据文件的选项。根据数据和所需的格式，有几种读取和写入的选项。我们将展示一些最有用的替代方案。

本章将涵盖以下主题：

+   文件处理

+   NumPy 方法

+   序列化

+   保持存储

+   读取和写入 Matlab 数据文件

+   读取和写入图像

# 第十五章：14.1 文件处理

文件**输入与输出**（**I/O**）在许多场景中是至关重要的，例如：

+   处理测量或扫描数据。测量结果存储在文件中，需要读取这些文件以进行分析。

+   与其他程序的交互。将结果保存到文件中，以便可以导入到其他应用程序中，反之亦然。

+   存储信息以备将来参考或比较。

+   与他人共享数据和结果，可能是在其他平台上使用其他软件。

本节将介绍如何在 Python 中处理文件 I/O。

## 14.1.1 与文件的交互

在 Python 中，`file` 类型的对象表示存储在磁盘上的物理文件的内容。可以使用以下语法创建一个新的 `file` 对象：

```py
# creating a new file object from an existing file
myfile = open('measurement.dat','r')
```

文件内容可以通过以下命令访问：

```py
print(myfile.read())
```

使用文件对象需要小心。问题在于，文件必须在重新读取或由其他应用程序使用之前关闭，这是通过以下语法完成的：

```py
myfile.close() # closes the file object
```

事情并没有那么简单，因为在执行 `close` 调用之前可能会触发异常，这将跳过关闭代码（考虑以下示例）。确保文件正确关闭的简单方法是使用上下文管理器。使用 `with` 关键字的这种结构将在第 12.1.3 节：*上下文管理器 – with 语句*中进行更详细的说明。以下是如何与文件一起使用它：

```py
with open('measurement.dat','r') as myfile: 
     ... # use myfile here
```

这确保了即使在块内引发异常时，文件也会在退出 `with` 块时关闭。该命令适用于上下文管理器对象。我们建议您阅读更多关于上下文管理器的内容，见第 12.1.3 节：*上下文*

*管理器 – with 语句*。以下是一个示例，展示了为什么 `with` 结构是值得推荐的：

```py
myfile = open(name,'w')
myfile.write('some data')
a = 1/0
myfile.write('other data')
myfile.close()
```

在文件关闭之前引发了异常。文件保持打开状态，并且无法保证数据何时以及如何写入文件。因此，确保达到相同结果的正确方法是：

```py
with open(name,'w') as myfile:
    myfile.write('some data')
    a = 1/0
    myfile.write('other data')
```

在这种情况下，文件会在异常（此处为`ZeroDivisionError`）被触发后干净地关闭。还需要注意的是，无需显式地关闭文件。

## 14.1.2 文件是可迭代的

文件尤其是可迭代的（见第 9.3 节：*可迭代对象*）。文件会迭代它们的每一行：

```py
with open(name,'r') as myfile:
    for line in myfile:
        data = line.split(';')
        print(f'time {data[0]} sec temperature {data[1]} C')
```

文件的每一行会作为字符串返回。`split` 字符串方法是将字符串转换为字符串列表的一个可能工具，例如：

```py
data = 'aa;bb;cc;dd;ee;ff;gg'
data.split(';') # ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg']

data = 'aa bb cc dd ee ff gg'
data.split(' ') # ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg']
```

由于对象 `myfile` 是可迭代的，我们也可以直接提取到列表中，示例如下：

```py
data = list(myfile)
```

## 14.1.3 文件模式

如你在这些文件处理的示例中看到的，函数 `open` 至少需要两个参数。第一个显然是文件名，第二个是描述文件使用方式的字符串。打开文件有几种模式，基本模式如下：

```py
with open('file1.dat','r') as ...  # read only
with open('file2.dat','r+') as ...  # read/write
with open('file3.dat','rb') as ...  # read in byte mode  
with open('file4.dat','a') as ...  # append (write to the end of the file)
with open('file5.dat','w') as ... # (over-)write the file
with open('file6.dat','wb') as ... # (over-)write the file in byte mode
```

模式 `'r'`、`'r+'` 和 `'a'` 要求文件已存在，而 `'w'` 会在没有该文件的情况下创建一个新文件。使用 `'r'` 和 `'w'` 进行读写是最常见的，正如你在前面的示例中看到的那样。

这里是一个例子，展示了如何使用追加模式 `'a'` 打开文件并在文件末尾添加数据，而不修改文件中已存在的内容。注意换行符 `\n`：

```py
_with open('file3.dat','a') as myfile:
    myfile.write('something new\n')
```

# 14.2 NumPy 方法

NumPy 提供了用于将 NumPy 数组数据读取和写入文本文件的内置方法。这些方法是 `numpy.loadtxt` 和 `numpy.savetxt`。

## 14.2.1 savetxt

将一个数组写入文本文件非常简单：

```py
savetxt(filename,data)
```

有两个有用的参数作为字符串给出，`fmt` 和 `delimiter`，它们控制列之间的格式和分隔符。默认值为分隔符为空格，格式为`%.18e`，即对应于具有所有数字的指数格式。格式化参数的使用方式如下：

```py
x = range(100) # 100 integers
savetxt('test.txt',x,delimiter=',') # use comma instead of space
savetxt('test.txt',x,fmt='%d') # integer format instead of float with e
```

## 14.2.3 loadtxt

从文本文件读取到数组使用以下语法：

```py
filename = 'test.txt'
data = loadtxt(filename)
```

由于数组中的每一行必须具有相同的长度，因此文本文件中的每一行必须有相同数量的元素。与 `savetxt` 类似，默认值为 `float`，分隔符为空格。可以使用 `dtype` 和 `delimiter` 参数进行设置。另一个有用的参数是 `comments`，可以用来标记数据文件中用于注释的符号。使用格式化参数的示例如下：

```py
data = loadtxt('test.txt',delimiter=';')    # data separated by semicolons

# read to integer type, comments in file begin with a hash character
data = loadtxt('test.txt',dtype=int,comments='#')
```

# 14.3 Pickling

你刚刚看到的读写方法会在写入之前将数据转换为字符串。复杂类型（如对象和类）不能以这种方式写入。使用 Python 的模块 `pickle`，你可以将任何对象以及多个对象保存到文件中。

数据可以保存为纯文本（ASCII）格式或使用稍微高效一些的二进制格式。主要有两种方法：`dump`，它将一个 Python 对象的 pickle 表示保存到文件中，和 `load`，它从文件中检索一个 pickle 对象。基本用法如下：

```py
import pickle
with open('file.dat','wb') as myfile:
    a = random.rand(20,20)
    b = 'hello world'
    pickle.dump(a,myfile)    # first call: first object
    pickle.dump(b,myfile)    # second call: second object

import pickle
with open('file.dat','rb') as myfile:
    numbers = pickle.load(myfile) # restores the array
    text = pickle.load(myfile)    # restores the string
```

注意返回的两个对象的顺序。除了这两种主要方法，有时将 Python 对象序列化为字符串而不是文件也是很有用的。这可以通过 `dumps` 和 `loads` 来实现。以下是序列化数组和字典的一个例子：

```py
a = [1,2,3,4]
pickle.dumps(a) # returns a bytes object
b = {'a':1,'b':2}
pickle.dumps(b) # returns a bytes object
```

使用`dumps`的一个好例子是当你需要将 Python 对象或 NumPy 数组写入数据库时。数据库通常支持存储字符串，这使得无需特殊模块就可以轻松地写入和读取复杂数据和对象。除了`pickle`模块，还有一个优化版，称为`cPickle`。它是用 C 语言编写的，若需要快速的读写操作，可以使用它。`pickle`和`*cPickle*`产生的数据是相同的，可以互换使用。

# 14.4 文件架构

字典中的对象可以通过键来访问。类似地，可以通过先为文件分配一个键来访问特定的数据。这可以通过使用`shelve`模块来实现：

```py
from contextlib import closing
import shelve as sv
# opens a data file (creates it before if necessary)
with closing(sv.open('datafile')) as data:
    A = array([[1,2,3],[4,5,6]])     
    data['my_matrix'] = A  # here we created a key
```

在第 14.1.1 节：*与文件交互*中，我们看到内置命令`open`会生成一个上下文管理器，我们也了解了这对于处理外部资源（如文件）为什么很重要。与此命令不同，`sv.open`本身不会创建上下文管理器。`contextlib`模块中的命令`closing`需要将其转变为合适的上下文管理器。

考虑以下恢复文件的示例：

```py
from contextlib import closing
import shelve as sv
with closing(sv.open('datafile')) as data: # opens a data file
    A = data['my_matrix']  # here we used the key
    ...
```

`shelve`对象具有所有字典方法，例如键和值，可以像字典一样使用。请注意，只有在调用了`close`或`sync`等方法后，文件中的更改才会被写入。

# 14.5 读取和写入 Matlab 数据文件

SciPy 能够使用模块\pyth!scipy.io!读取和写入 Matlab 的`.mat`文件格式。相关命令是`loadmat`和`savemat`。

要加载数据，请使用以下语法：

```py
import scipy.io
data = scipy.io.loadmat('datafile.mat')
```

变量数据现在包含一个字典，字典的键对应于保存在`.mat`文件中的变量名。变量以 NumPy 数组格式存储。保存到`.mat`文件时，需要创建一个包含所有要保存的变量（变量名和对应的值）的字典。然后使用命令`savemat`：

```py
data = {}
data['x'] = x
data['y'] = y
scipy.io.savemat('datafile.mat',data)
```

这将`x`和`y`这两个 NumPy 数组保存为 Matlab 的内部文件格式，从而保留变量名。

# 14.6 读取和写入图像

`PIL.Image`模块提供了一些用于处理图像的函数。以下代码将读取一张*JPEG*图像，打印其形状和类型，然后创建一张调整过大小的图像，并将新图像写入文件：

```py
import PIL.Image as pil   # imports the Pillow module

# read image to array
im=pil.open("test.jpg")
print(im.size)   # (275, 183)  
                 # Number of pixels in horizontal and vertical directions
# resize image
im_big = im.resize((550, 366))
im_big_gray = im_big.convert("L") # Convert to grayscale

im_array=array(im)

print(im_array.shape)
print(im_array.dtype)   # unint 8
# write result to new image file
im_big_gray.save("newimage.jpg")
```

PIL 创建了一个可以轻松转换为 NumPy 数组的图像对象。作为数组对象，图像以 8 位无符号整数（`unint8`）的形式存储像素值，范围为*0...255*。第三个形状值表示图像的颜色通道数。在此情况下，*3*表示这是一个彩色图像，其值按照以下顺序存储：红色`im_array[:,:,0]`，绿色`im_array[:,:,1]`，蓝色`im_array[:,:,2]`。灰度图像只有一个通道。

对于处理图像，`PIL`模块包含许多有用的基本图像处理功能，包括滤波、变换、度量以及从 NumPy 数组转换为`PIL`图像对象：

```py
new_image = pil.from_array(ima_array)
```

# 14.7 总结

文件处理在处理测量数据和其他大量数据源时是不可避免的。此外，与其他程序和工具的通信也是通过文件处理完成的。

你已经学会将文件视为一个 Python 对象，就像其他对象一样，具有重要的方法，如`readlines`和`write`。我们展示了如何通过特殊属性保护文件，这些属性可能只允许读取或写入访问。

你写入文件的方式往往会影响处理的速度。我们看到数据是通过序列化（pickling）或使用`shelve`方法来存储的。
