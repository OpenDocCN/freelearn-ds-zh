# 第八章：Python 参考

# 介绍

Python 是一种通用编程和脚本语言。其简洁性和丰富的库使得开发出快速符合现代技术需求的应用成为可能。

Python 代码写在以`.py`为后缀的文件中，可以通过`python`命令执行。

# Python Hello World 示例

Python 中最简单的程序会打印一行文本。

**输入**：

```py
source_code/appendix_c_python/example00_helloworld.py
print "Hello World!"
```

**输出**：

```py
$ python example00_helloworld.py
Hello World!
```

# 注释

注释在 Python 中不会被执行。它们以`#`字符开头，并以行尾结束。

**输入**：

```py
# source_code/appendix_c_python/example01_comments.py print "This text will be printed because the print statement is executed."
#This is just a comment and will not be executed.
#print "Even commented statements are not executed."
print "But the comment finished with the end of the line."
print "So the 4th and 5th line of the code are executed again."
```

**输出**：

```py
$ python example01_comments.py This text will be printed because the print statement is executed
But the comment finished with the end of the line.
So the 4th and 5th line of the code are executed again.
```

# 数据类型

Python 中可用的一些数据类型如下：

+   **数值数据类型**：`int`和`float`

+   **文本数据类型**：`str`

+   **复合数据类型**：`tuple`、`list`、`set`和`dictionary`

# 整型

`int`数据类型只能存储整数值。

**输入**：

```py
# source_code/appendix_c_python/example02_int.py rectangle_side_a = 10
rectangle_side_b = 5
rectangle_area = rectangle_side_a * rectangle_side_b
rectangle_perimeter = 2*(rectangle_side_a + rectangle_side_b)
print "Let there be a rectangle with the sides of lengths:"
print rectangle_side_a, "and", rectangle_side_b, "cm."
print "Then the area of the rectangle is", rectangle_area, "cm squared."
print "The perimeter of the rectangle is", rectangle_perimeter, "cm."
```

**输出**：

```py
$ python example02_int.py Let there be a rectangle with the sides of lengths: 10 and 5 cm.
Then the area of the rectangle is 50 cm squared.
The perimeter of the rectangle is 30 cm.
```

# 浮点数

`float`数据类型也可以存储非整数的有理数值。

**输入**：

```py
# source_code/appendix_c_python/example03_float.py pi = 3.14159
circle_radius = 10.2
circle_perimeter = 2 * pi * circle_radius
circle_area = pi * circle_radius * circle_radius
print "Let there be a circle with the radius", circle_radius, "cm."
print "Then the perimeter of the circle is", circle_perimeter, "cm."
print "The area of the circle is", circle_area, "cm squared."
```

**输出**：

```py
$ python example03_float.py Let there be a circle with the radius 10.2 cm.
Then the perimeter of the circle is 64.088436 cm.
The area of the circle is 326.8510236 cm squared.
```

# 字符串

字符串变量可以用来存储文本。

**输入**：

```py
# source_code/appendix_c_python/example04_string.py first_name = "Satoshi"
last_name = "Nakamoto"
full_name = first_name + " " + last_name
print "The inventor of Bitcoin is", full_name, "."
```

**输出**：

```py
$ python example04_string.py The inventor of Bitcoin is Satoshi Nakamoto .
```

# 元组

`tuple`数据类型类似于数学中的向量；例如，`tuple = (整数, 浮点数)`。

**输入**：

```py
# source_code/appendix_c_python/example05_tuple.py import math

point_a = (1.2,2.5)
point_b = (5.7,4.8)
#math.sqrt computes the square root of a float number.
#math.pow computes the power of a float number.
segment_length = math.sqrt(
        math.pow(point_a[0] - point_b[0], 2) +
        math.pow(point_a[1] - point_b[1], 2))
print "Let the point A have the coordinates", point_a, "cm."
print "Let the point B have the coordinates", point_b, "cm."
print "Then the length of the line segment AB is", segment_length, "cm."
```

**输出**：

```py
$ python example05_tuple.py Let the point A have the coordinates (1.2, 2.5) cm.
Let the point B have the coordinates (5.7, 4.8) cm.
Then the length of the line segment AB is 5.0537115074 cm.
```

# 列表

Python 中的列表是一个有序的值集合。

**输入**：

```py
# source_code/appendix_c_python/example06_list.py some_primes = [2, 3]
some_primes.append(5)
some_primes.append(7)
print "The primes less than 10 are:", some_primes
```

**输出**：

```py
$ python example06_list.py The primes less than 10 are: [2, 3, 5, 7]
```

# 集合

`Set`是 Python 中的一个无序的数学集合类型。

**输入**：

```py
# source_code/appendix_c_python/example07_set.py from sets import Set

boys = Set(['Adam', 'Samuel', 'Benjamin'])
girls = Set(['Eva', 'Mary'])
teenagers = Set(['Samuel', 'Benjamin', 'Mary'])
print 'Adam' in boys
print 'Jane' in girls
girls.add('Jane')
print 'Jane' in girls
teenage_girls = teenagers & girls #intersection
mixed = boys | girls #union
non_teenage_girls = girls - teenage_girls #difference
print teenage_girls
print mixed
print non_teenage_girls
```

**输出**：

```py
$ python example07_set.py True
False
True
Set(['Mary'])
Set(['Benjamin', 'Adam', 'Jane', 'Eva', 'Samuel', 'Mary'])
Set(['Jane', 'Eva'])
```

# 字典

`dictionary`是一种数据结构，可以通过键存储对应的值。

**输入**：

```py
# source_code/appendix_c_python/example08_dictionary.py dictionary_names_heights = {}
dictionary_names_heights['Adam'] = 180.
dictionary_names_heights['Benjamin'] = 187
dictionary_names_heights['Eva'] = 169
print 'The height of Eva is', dictionary_names_heights['Eva'], 'cm.'
```

**输出**：

```py
$ python example08_dictionary.py The height of Eva is 169 cm.
```

# 流程控制

Python 编程语言通过使用条件语句、`for`循环（包括`break`和`continue`语句）以及函数来控制程序执行流程。

# 条件语句

如果满足特定条件，可以使用`if`语句执行一定的代码。如果条件不满足，则可以执行`else`语句后的代码。如果第一个条件不满足，我们可以设置下一个条件，通过`elif`语句来执行代码。

**输入**：

```py
# source_code/appendix_c_python/example09_if_else_elif.py x = 10
if x == 10:
        print 'The variable x is equal to 10.'

if x > 20:
        print 'The variable x is greater than 20.'
else:
        print 'The variable x is not greater than 20.'

if x > 10:
        print 'The variable x is greater than 10.'
elif x > 5:
        print 'The variable x is not greater than 10, but greater ' +
              'than 5.'
else:
        print 'The variable x is not greater than 5 or 10.'
```

**输出**：

```py
$ python example09_if_else_elif.py The variable x is equal to 10.
The variable x is not greater than 20.
The variable x is not greater than 10, but greater than 5.
```

# `for`循环

`for`循环可以遍历一组元素中的每个元素，例如，`range`、`python set`和`list`。

# 范围上的`for`循环

**输入**：

```py
source_code/appendix_c_python/example10_for_loop_range.py print "The first 5 positive integers are:"
for i in range(1,6):
        print i
```

**输出**：

```py
$ python example10_for_loop_range.py The first 5 positive integers are:
1
2
3
4
5
```

# 列表上的`for`循环

**输入**：

```py
source_code/appendix_c_python/example11_for_loop_list.py primes = [2, 3, 5, 7, 11, 13]
print 'The first', len(primes), 'primes are:'
for prime in primes:
        print prime
```

**输出**：

```py
$ python example11_for_loop_list.py The first 6 primes are:
2
3
5
7
11
13
```

# Break 和 continue

`for`循环可以通过`break`语句提前退出。使用`continue`语句可以跳过`for`循环中的剩余部分。

**输入**：

```py
source_code/appendix_c_python/example12_break_continue.py for i in range(0,10):
        if i % 2 == 1: #remainder from the division by 2
                continue
        print 'The number', i, 'is divisible by 2.'

for j in range(20,100):
        print j
        if j > 22:
                break;
```

**输出**：

```py
$ python example12_break_continue.py The number 0 is divisible by 2.
The number 2 is divisible by 2.
The number 4 is divisible by 2.
The number 6 is divisible by 2.
The number 8 is divisible by 2.
20
21
22
23
```

# 函数

Python 支持函数的定义，这是一种很好的方式，可以在程序的多个地方执行相同的代码。函数是使用`def`关键字定义的。

**输入**：

```py
source_code/appendix_c_python/example13_function.py def rectangle_perimeter(a, b):
        return 2 * (a + b)

print 'Let a rectangle have its sides 2 and 3 units long.'
print 'Then its perimeter is', rectangle_perimeter(2, 3), 'units.'
print 'Let a rectangle have its sides 4 and 5 units long.'
print 'Then its perimeter is', rectangle_perimeter(4, 5), 'units.'
```

**输出**：

```py
$ python example13_function.py Let a rectangle have its sides 2 and 3 units long.
Then its perimeter is 10 units.
Let a rectangle have its sides 4 and 5 units long.
Then its perimeter is 18 units.
```

# 输入与输出

让我们看看如何向 Python 程序传递参数，以及如何读写文件。

# 程序参数

我们可以通过命令行向程序传递参数。

**输入**：

```py
source_code/appendix_c_python/example14_arguments.py #Import the system library in order to use the argument list.
import sys

print 'The number of the arguments given is', len(sys.argv),'arguments.'
print 'The argument list is ', sys.argv, '.'
```

**输出**：

```py
$ python example14_arguments.py arg1 110 The number of the arguments given is 3 arguments.
The argument list is  ['example14_arguments.py', 'arg1', '110'] .
```

# 读写文件

以下程序将写入两行到`test.txt`文件，然后读取它们，最后将其打印到输出。

**输入**：

```py
# source_code/appendix_c_python/example15_file.py #write to the file with the name "test.txt"
file = open("test.txt","w") 
file.write("first line\n")
file.write("second line")
file.close()

#read the file
file = open("test.txt","r")
print file.read()
```

**输出**：

```py
$ python example15_file.py first line
second line
```
