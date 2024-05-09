# 第九章：理解输入和输出以设计解决方案算法

在本章中，我们将深入研究问题，以确定设计问题算法所需的输入和输出。我们将使用我们在*第八章*中学到的概念，*Python 简介*，在那里我们讨论了面向对象的编程、字典、列表等。当您练习获取输入并在算法中使用它时，您将能够看到算法的输出取决于输入信息。

在本章中，我们将涵盖以下主题：

+   定义输入和输出

+   理解计算思维中的输入和输出

在本章中，我们将重点关注理解不同类型的输入以及如何使用**Python**编程语言中的输出。为了更好地理解这些主题，我们首先要看一下它们的定义。

# 技术要求

对于本章，您需要安装最新的 Python 版本。

您可以在此章节中找到使用的源代码：[`github.com/PacktPublishing/Applied-Computational-Thinking-with-Python/tree/master/Chapter09`](https://github.com/PacktPublishing/Applied-Computational-Thinking-with-Python/tree/master/Chapter09)

# 定义输入和输出

我们将从研究输入及其定义开始，这反过来又用于提供结果或输出。

**输入**是放入系统或机器中的东西。在计算机中，我们有输入设备，计算机会解释这些设备以提供结果或**输出**。当我们看不同类型的输入以及如何在算法中使用输入时，您将看到我们从这些算法中获得的输出。让我们看一些输入设备的例子：

+   **键盘**：当我们使用键盘时，计算机会解释按键，并在各种程序中使用该解释，比如文档编写和搜索栏。

+   **鼠标**：鼠标用于在我们的物理屏幕上导航，帮助我们点击内容和滚动页面。

+   **操纵杆**：操纵杆可用于玩电脑游戏。

+   **麦克风**：在更现代的机器中，麦克风不仅用于通过电话和视频应用进行通信，还用于接受**人工智能**（**AI**）助手如**Cortana**和**Siri**的口头命令，以及语音转文字命令。

所有上述输入都被编程到我们的计算机中以便使用。在 Python 中，我们编写使用这些输入的算法。**用户**输入是指程序询问用户将用于生成输出的信息。请记住，输入和输出可以在算法中的任何时间发生。我们还可以提供需要用户额外输入的输出反馈，依此类推。让我们看看如何从 Python 获取用户输入。

Python 有一个主要的用户提示，即`input()`函数。此函数用于接受用户输入，然后自动将输入转换为用户期望的输出。

我们之前在多个程序中使用过这些主要提示，比如*第五章*中的商店示例，*探索问题分析*。让我们看看算法中一些输入命令的样子。看看以下代码片段：

ch9_input1.py

```py
name = input("What is your name? ")
print("Nice to meet you " + name + ".")
```

片段要求用户输入他们的名字，然后使用输入的信息打印一个声明。这段代码的结果如下：

```py
What is your name? Mikayla
Nice to meet you Mikayla.
```

现在，我们可以只要求输入而不使用任何提示问题，但用户将不知道正在被问及什么或者它将如何被使用。看看没有提示的片段，如下所示：

ch9_input2.py

```py
name = input()
print("Nice to meet you " + name + ".")
```

当我们运行上述程序时，在我们的 shell 中没有打印任何内容。然而，我知道我需要输入一些东西，所以看看当我只输入一个字母时发生了什么：

```py
d
Nice to meet you d.
```

正如你所看到的，用户不会知道该怎么做，因为窗口没有询问任何事情；它只是给了一个空白的空间，用户必须假设那是输入的地方。但这可能会导致很多混乱，因为没有告诉用户是否需要输入，或者他们需要输入什么样的东西。

我们可以通过使用 `print` 语句来减轻一些混乱，例如，通过向用户提供关于我们将要询问的输入的信息。我们还可以在 `input()` 函数中包含一条语句。例如，我们可以使用这样的一行代码 `name = input('你叫什么名字？')`，这样首先会显示带引号的语句，这样用户就知道我们在问什么。无论哪种方式，提供一条语句让用户知道需要什么是我们设计算法时非常重要的。

正如你在前面的例子中看到的，`input()` 函数接受一个字符串，然后在 `print()` 语句中使用该字符串。让我们看看使用 `input` 命令时列表会是什么样子：

ch9_input3.py

```py
#Create the list
names = []

#Ask user how many names will be added
name = int(input("How many names will be in the list? ")) 

#Iterate to add each name to the list
for i in range(0, name): 
    people = str(input()) 

    names.append(people) 

print(names)
```

在上面的片段中，我们首先创建了列表，这不会显示给用户。然后询问用户将有多少个名字添加到列表中。之后，我们遍历列表，以便用户可以分别输入每个名字。最后，名字被添加到列表中并打印出来。看一下这个算法的输出：

```py
How many names will be in the list? 4
Manny
Lisa
John
Katya
['Manny', 'Lisa', 'John', 'Katya']
```

注意，名字没有提示，所以算法假设用户在输入值 `4` 后会知道该怎么做。这可以通过在迭代中简单添加来减轻。看一下下面的代码片段：

ch9_input4.py

```py
#Create the list
names = []

#Ask user how many names will be added
name = int(input("How many names will be in the list? ")) 

#Iterate to add each name to the list
for i in range(0, name): 
    people = input("Type the next name on the list. ") 

    names.append(people) 

print(names)
```

正如你所看到的，每个名字现在都有一个提示，要求输入下一个名字，当算法运行时，输出看起来像下面的样子：

```py
How many names will be in the list? 4
Type the next name on the list. Milo
Type the next name on the list. Sonya
Type the next name on the list. Gabriel
Type the next name on the list. Maxine
['Milo', 'Sonya', 'Gabriel', 'Maxine']
```

上面的输出显示了完成的列表以及添加到该列表中的每个名字的提示。在算法中对这些提示进行简单的添加可以减轻用户在输入时的混乱。

正如你从本节中前面的片段中看到的，输入被读取为整数、字符串和列表。Python 自动转换了信息，以便算法可以使用它。

在 Python 中，还有一种方法可以使用一行代码从多个输入中定义多个变量。看一下下面的代码片段：

ch9_input5.py

```py
name1, name2 = input("Enter First Name: "), input("Enter Last Name: ")
print(name1 + " " + name2)
```

正如你在前面的代码中所看到的，`print()` 命令中的引号(`" "`) 用于分隔输入。看一下这个算法的输出：

```py
Enter First Name: John
Enter Last Name: Doe
John Doe
```

正如你所看到的，程序要求输入名字，这些名字在算法的第一行中被调用。这绝不是我们从用户那里获取输入的唯一方式，也不是我们将使用的唯一输入。

在本节中，我们看了一些在算法中定义输入的方法。我们还学会了如何从用户那里获取输入以便稍后在算法中使用。现在让我们转到下一节，我们将在其中看一些计算思维问题中的输入和输出。

# 理解计算思维中的输入和输出

为了更好地理解输入和输出，我们将在本节中看一些计算思维问题。当我们设计算法的过程中，我们将专注于确定算法所需的输入以及我们从算法中需要的输出。让我们来看看我们的第一个问题。

## 问题 1 - 构建凯撒密码

**凯撒密码**是一种用于编码消息的密码系统。密码学用于保护信息，以便只有预期的用户可以阅读消息。凯撒密码使用字母的移位来编码消息。例如，字母 *a* 移动 3 个位置将变成 *d*。为了构建一个可以为我们做到这一点的算法，我们需要一些东西：

+   将要编码的消息

+   我们将每个字母移动多少

+   一个打印的编码消息

让我们思考一下前面提到的要点意味着什么。消息将需要用户输入。移位也将是一个输入，因为我们不希望程序总是使用相同的代码。否则，一旦知道了移位，原始消息就很容易被破解。打印的编码消息是我们的输出。以下是一些步骤，以帮助创建算法：

1.  首先，我们需要输入消息。

1.  然后我们需要定义输入移位。

1.  之后，我们需要遍历每个字母。

1.  迭代后，我们根据定义的移位调整每个字母。

1.  最后，我们打印出新的编码消息。

我们将通过以下密码算法片段来举例说明前面的步骤：

ch9_problem1.py

```py
#Print initial message for the user
print("This program will take your message and encode it.")
#Ask for the message
msg = input("What message would you like to code? ")
#Ask for shift
shift = int(input("How many places will you shift your message? "))
msgCipher = ""
#Iterate through the letters, adjusting for shift
for letter in msg:
  k = ord(letter)
  if 48 <= k <= 57:
    newk = (k - 48 + shift)%10 + 48
  elif 65 <= k <= 90:
    newk = (k - 65 + shift)%26 + 65
  elif 97 <= k <=122:
    newk = (k - 97 + shift)%26 + 97
  else:
    newk = k
  msgCipher += chr(newk)
print("Your coded message is below.")
print(msgCipher)
```

请注意，在迭代中，我们正在进行一些数学运算，以找出字母表中每个字母的值。我们使用一些条件语句来定义在使用用户定义的移位值后，每个字母的值是什么。

让我们看看当我们运行此算法时产生的输出：

```py
This program will take your message and encode it.
What message would you like to code? Code this message
How many places will you shift your message? 2
Your coded message is below.
Eqfg vjku oguucig
```

正如您可以从前面的输出中看到的，注意*消息*和*编码消息*中的第一个单词 - `Code`：

+   `C`，向后移动两个字母，现在是`E`。

+   `o`现在是`q`，依此类推。

对于消息中的每个字母，字母都向后移动了两个位置。输出是我们的`print()`语句加上编码消息，由行`print(msgCipher)`给出。我们的输出包括两个语句以便清晰。*第一个消息是必要的吗？* 不。但是，在某些算法中，最好有一些行，让用户知道算法的运行情况。

让我们看看另一个问题。

## 问题 2 - 查找最大值

您被要求创建一个程序，该程序从数字列表中找到最大值。数字列表由用户提供。因此，为此问题创建一个算法。

首先，我们需要确定此特定问题的输入和输出。它们列如下：

1.  列表中的项目数（输入）

1.  列表中的数字（输入）

1.  列表中的最大值（输出）

回想一下本章前面，在*定义输入和输出*部分，我们可以定义一个空列表，然后要求用户让程序知道将有多少项目输入列表。以下程序举例说明了相同的情况：

ch9_problem2.py

```py
#Define the list name
maxList = []
#Ask user how many numbers will be entered
quant = int(input("How many data points are you entering? "))
#Iterate, append points, and find maximum
for i in range(0, quant):
    dataPoint = int(input("Enter number: "))
    maxList.append(dataPoint)
#Print maximum value
print("The maximum value is " + str(max(maxList)) + ".")
```

请注意，从前面的代码中，我们使用了`max()`函数来找到列表中的最大值。此外，我们还必须添加`int()`类型，以便算法正确识别值为数字。代码遍历从`0`到我们从用户输入中收到的数据点数，我们将其定义为`quant`变量。当算法遍历数字时，它会比较它们并找到最大值。

让我们看看输出是什么样子的：

```py
How many data points are you entering? 6
Enter number: 1
Enter number: 2
Enter number: 8
Enter number: 4
Enter number: 5
Enter number: 7
The maximum value is 8.
```

正如您可以从前面的输出中看到的，用户声明将输入`6`个数字。然后算法提示用户输入每个值。最后，识别出最大值。

现在让我们看看如何构建一个猜数字游戏。

## 问题 3 - 构建一个猜数字游戏

您被要求构建一个游戏，用户可以根据需要获得尽可能多的机会来按正确顺序识别四位数的数字。当用户输入猜测时，算法将指出是否有任何数字是正确的。

算法将确定数字是否在正确的位置，但不会确定哪些数字是正确的，哪些在正确的位置。如果玩家猜对了数字，游戏就结束了。您可能会发现这个游戏类似于一个名为**Mastermind**的流行游戏，其中玩家将四个颜色代码插入游戏，第二个玩家猜测。

在这种情况下，我们使用数字而不是颜色，计算机程序将帮助我们确定是否有任何数字是正确的，以及它们是否在正确的位置。

为了构建这个游戏，让我们看看我们需要的输入和输出：

+   随机生成的 4 位数（输入）

+   用户的猜测（输入）

+   来自算法的数字和位置的反馈（输出）

+   在数字被猜中后，结束游戏的消息（输出）

当我们生成数字时，我们需要找到一种比较该数字的数字与用户输入的数字的方法。请注意，我们需要为这个特定的算法导入`random`库，以便我们可以生成随机的四位数。让我们看一下代码片段：

ch9_problem3.py

```py
import random as rand
number = rand.randint(1000,10000)
guess = int(input("What's your first guess? ")) 
#Algorithm checks if number is correct. 
if (guess == number): 
	print("That's right! You win!") 
else: 
        i = 0
```

正如你所看到的，这是我们需要用户输入的一个例子。然而，这只是第一次猜测，所以我们需要从用户那里获取更多的输入，并提供一些提供反馈的输出。看一下来自同一算法的以下代码片段：

```py
#Condition so that user keeps guessing until they win.
        while (guess != number):
            i = i + 1
            #Remember you can also write as i += 1
            j = 0
            guess = str(guess) 
            number = str(number) 
            #Check which numbers are correct and mark incorrect with 'N'
            guessY = ['N']*4
            #Make sure you check each digit, so run loop 4 times 
            for q in range(0, 4):
                    if (guess[q] == number[q]): 
                            j += 1 
                            guessY[q] = guess[q] 
                    else: 
                            continue
            #If only some digits are correct, run next condition 
            if (j < 4) and (j != 0): 
                    print("You have " + str(j) + " digit(s) right.") 
                    print("These numbers were correct.") 
                    for m in guessY: 
                            print(m, end = " ") 
                    #Ask for next input
                    guess = int(input("What is your next guess? ")) 
```

注意，我们需要在每次猜测失败后询问每次猜测。算法需要知道下一次猜测是什么，以便在用户没有识别出正确的四位数时继续运行。这就是为什么猜测输入值在算法中出现在多个地方。

一旦用户猜对了，程序需要打印最终的声明，在这种情况下，我们会让用户知道猜测花了多少次：

```py
            #If only some digits are correct, run next condition 
            if (j < 4) and (j != 0): 
                    print("You have " + str(j) + " digit(s) right.") 
                    print("These numbers were correct.") 
                    for m in guessY: 
                            print(m, end = " ") 
                    #Ask for next input
                    guess = int(input("What is your next guess? ")) 
            #No digits correct
            elif (j == 0): 
                    print("None of these digits are correct.") 
                    guess = int(input("What is your next guess? ")) 
        if guess == number: 
            print("It took you " + str(i)+ " tries to win!") 
```

正如你所看到的，这个算法在多个地方基于满足条件的输入，以及输出。输出包括`print`语句和对正确或错误数字的反馈。让我们看看程序执行时输出的样子：

```py
What's your first guess? 1111
None of these digits are correct.
What is your next guess? 2222
None of these digits are correct.
What is your next guess? 3333
You have 1 digit(s) right.
These numbers were correct.
3 N N N What is your next guess? 3444
You have 3 digit(s) right.
These numbers were correct.
3 4 N 4 What is your next guess? 3454
You have 3 digit(s) right.
These numbers were correct.
3 4 N 4 What is your next guess? 3464
You won in 6 attempts.
```

请注意，在第三次尝试后，我们有一个正确的数字，表示为`3` `N N N`。这意味着第一位数字是 3。然后我们需要猜测其余的数字。在程序中提供反馈，或输出，使用户能够继续这个游戏。

正如你所看到的，理解输入和输出使我们能够创建能够提供反馈、根据条件进行迭代等等的程序。

在这一部分，我们学习了如何在解决问题时使用输入和输出。我们提供了一些场景，让我们可以探索一些输入类型，比如用户定义的输入和算法定义的输入。我们还学习了如何编写算法，为用户提供清晰的输出。

# 总结

在本章中，我们讨论了 Python 算法中的输入和输出。输入是我们获取算法响应的信息的方式。例如，用户可以为需要提示信息的算法提供输入，就像我们在示例问题中看到的那样。Python 和其他编程语言也可以从鼠标、扫描仪、摄像头和程序交互的其他设备中获取输入。

为了编写我们的算法，重要的是要确定我们需要什么输入，我们在设计中何时需要它们，以及我们需要程序输出什么。在[*第三章*]（B15413_03_Final_SK_ePub.xhtml#_idTextAnchor056），*理解算法和算法思维*中，我们使用用户输入来找到午餐的成本。我们创建了一个字典来帮助我们找到这些信息。同样，在阅读完本章之后，你现在有了构建算法来解决一些额外问题的技能，比如我们的数字猜测游戏和我们的最大查找器。

在设计算法之前，确定输入和输出是至关重要的，因为我们在算法中做出的条件和决策取决于输入和我们需要程序提供的输出。

在下一章中，我们将讨论控制流，这在理解算法的阅读方式和指令执行方式上至关重要。我们还将在探索控制流中更仔细地查看函数。
