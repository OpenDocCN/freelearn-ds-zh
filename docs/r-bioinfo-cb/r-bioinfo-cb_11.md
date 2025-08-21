# 第十一章：为了代码重用构建对象和包

在最后一章中，我们将探讨如何将我们的代码从自己的机器中带出去，并与世界分享。我们最常分享的对象将是我们自己！因此，为了使我们的编程生活更加轻松和流畅，我们将学习如何创建对象和类来简化我们的工作流程，并如何将它们打包成可在其他项目中重用的包。我们将查看如何在 GitHub 等网站上共享代码的工具，以及如何检查代码是否按预期工作。

本章将涵盖以下食谱：

+   创建简单的 S3 对象来简化代码

+   利用 S3 类的通用对象函数

+   使用 S4 系统创建结构化和正式的对象

+   简单的代码打包方法，用于共享和重用

+   使用 `devtools` 从 GitHub 托管代码

+   构建单元测试套件，以确保函数按预期工作

+   使用 Travis 进行持续集成，以确保代码经过测试并保持最新

# 技术要求

您需要的示例数据可以从本书的 GitHub 仓库获取：[`github.com/PacktPublishing/R-Bioinformatics-Cookbook`](https://github.com/PacktPublishing/R-Bioinformatics-Cookbook)[。](https://github.com/danmaclean/R_Bioinformatics_Cookbook) 如果您想按照书中的代码示例使用，您需要确保这些数据位于您的工作目录的子目录中。

这里是您需要的 R 包。通常，您可以通过 `install.packages("package_name")` 来安装这些包。列在 `Bioconductor` 下的包需要通过专用的安装程序安装。如果您需要做其他安装操作，会在包使用的食谱中描述：

+   `devtools`

+   `usethis`

对于一些后续的食谱，我们还需要安装 `git` 版本控制系统。请访问官方网页以获取适用于您系统的最新版本：[`git-scm.com/downloads`](https://git-scm.com/downloads)。您还可以在 GitHub 网站上找到一个有用的 GitHub 账户。如果您还没有 GitHub 账户，请访问[`github.com/`](https://github.com/)。

通常，在 R 中，用户会加载一个库并直接按名称使用函数。这在交互式会话中非常方便，但当加载了多个包时，可能会引起混乱。为了明确在某一时刻我正在使用哪个包和函数，我偶尔会使用 `packageName::functionName()` 这种惯例。

有时，在食谱的中途，我会中断代码，以便您可以看到一些中间输出或了解对象的结构，这对理解非常重要。每当发生这种情况时，您将看到一个代码块，其中每一行都以`##`（双哈希符号）开头。请考虑以下命令：

`letters[1:5]` 这将给我们以下输出：

`## a b c d e` 请注意，输出行前面有`##`。

# 创建简单的 S3 对象以简化代码

创建你自己的对象可以大大简化代码和工作流，使它们更容易被你复现和重用，并且将程序的内部逻辑抽象掉，减少你作为程序员的认知负担，让你能够更多地集中精力在项目的生物信息学和分析方面。R 实际上有多种方法来创建对象和类。在这个教程中，我们将重点看它最简单、最临时的方法——S3。这是一种相当非正式的创建对象和类的方式，但在很多情况下都足够使用。

# 准备工作

在这个教程中，我们只需要基础的 R 函数，因此无需安装任何额外的工具。

# 如何实现...

创建简单的 S3 对象以简化代码可以通过以下步骤完成：

1.  创建一个构造函数：

```py
SimpleGenome <- function( nchr=NA, lengths = NA){

 genome <- list(
 chromosome_count = nchr,
 chromosome_lengths = lengths
 )
 class(genome) <- append(class(genome), "SimpleGenome")
 return(genome)
}
```

1.  调用构造函数来创建新对象：

```py
ecoli <- SimpleGenome(nchr = 1, lengths = c(4600000) )
bakers_yeast <- SimpleGenome(nchr = 1, lengths=c(12100000))

```

# 它是如何工作的...

*第 1 步* 是所有工作的起点。这就是我们创建 S3 对象所需的全部内容。如你所见，这是非常简洁的代码。我们只是创建了一个生成并返回数据结构的函数。我们的类应该代表一个简化的基因组，并且我们希望它能保存一些关于基因组的基本信息。`SimpleGenome()` 函数是我们创建对象的构造函数。`SimpleGenome` 创建的基因组列表是构成最终对象主体的数据结构。这个列表的成员就是对象的槽，因此我们创建了名为 `chromosome_count` 和 `chromosome_length` 的成员，以表示基因组的一些特征。完成这一步后，我们进行一个重要的步骤——我们将类名（`SimpleGenome`）附加到基因组列表的类属性上。正是这一点使 R 能够识别这个对象属于 `SimpleGenome` 类。现在我们可以返回创建的 S3 对象。

在*第 2 步*中，我们只是简单地使用构造函数来创建类的实例。检查得到的对象如下所示：

```py
> ecoli 
$chromosome_count 
[1] 1 
$chromosome_lengths 
[1] 4600000 
attr(,"class") 
[1] "list" "SimpleGenome"

> bakers_yeast 
$chromosome_count 
[1] 1 
$chromosome_lengths 
[1] 12100000 
attr(,"class") 
[1] "list" "SimpleGenome" 
```

我们可以看到对象的槽、对象之间的差异，以及包含新 `SimpleGenome` 对象的类。这就是我们创建 S3 对象的方式；它是一个简单但有效的做法。与仅仅创建一个普通数据结构（如列表）相比，优势并不是立刻显现出来，但当我们查看如何在下一个教程中创建方法时，原因将会更加明确。

# 利用 S3 类的通用对象函数

一旦我们有了一个 S3 对象，我们需要创建与之配套的函数。这些函数实际上是使得长期使用这些对象变得更容易的关键。正是在这些函数中，我们可以抽象化对象数据的处理，从而减少每次使用时所需的工作量。R 的对象系统基于通用函数。这些函数是具有相同基础名称但类特定名称扩展的分组函数。每个分组称为方法，R 会根据方法调用的对象的类来决定调用属于该方法的具体哪个函数。这意味着我们可以在`A`类的对象上调用`plot()`，并得到与在`B`类对象上调用时完全不同的图形。在本食谱中，我们将了解它是如何工作的。

# 准备工作

对于本食谱，我们将使用基础的 R 函数，因此无需安装任何包，但我们将使用内置的`iris`数据。

# 如何实现...

利用通用对象函数与 S3 类一起使用，可以通过以下步骤完成：

1.  在`plot()`方法中创建一个通用函数：

```py
plot.SimpleGenome <- function(x){
 barplot(x$chromosome_lengths, main = "Chromosome Lengths")
}
```

1.  创建一个对象并在`plot()`中使用它：

```py
athal <- SimpleGenome(nchr = 5, lengths = c(34964571, 22037565, 25499034, 20862711, 31270811 ) )
plot(athal)
```

1.  首先创建一个新方法：

```py
genomeLength <- function(x){
 UseMethod("genomeLength", x)
}

genomeLength.SimpleGenome <- function(x){
 return(sum(x$chromosome_lengths))
}
genomeLength(athal)
```

1.  修改现有对象的类：

```py
some_data <- iris
summary(some_data)
class(some_data) <- c("my_new_class", class(some_data) )
class(some_data)
```

1.  为新类创建通用函数：

```py
summary.my_new_class <- function(x){
 col_types <- sapply(x, class)
 return(paste0("object contains ", length(col_types), " columns of classes:", paste (col_types, sep =",", collapse = "," )))
}
summary(some_data)
```

# 它是如何工作的...

在*第 1 步*中，我们创建了一个名为`plot.SimpleGenome()`的通用函数。这里的特殊命名约定标志着该函数是属于专门针对`SimpleGenome`类对象的通用绘图函数组。约定是`method.class`。这就是使通用绘图方法工作的所有必要步骤。

在*第 2 步*中，我们实际上创建一个`SimpleGenome`对象，就像在本章中的*创建简单的 S3 对象以简化代码*的食谱中所做的那样（你需要确保该食谱的*第 1 步*在当前会话中已经执行，否则这一步无法正常工作），然后调用`plot()`方法。`plot`方法查找适用于`SimpleGenome`对象的通用函数并运行该对象，从而生成我们预期的条形图，如下图所示：

![](img/a0f08b74-f112-4fdb-a319-9c4823670dd8.png)

在*第 3 步*中，我们深入了一些。在这一步中，我们想使用一个不存在的方法名（你可以使用`methods()`函数查看已经存在的方法），因此我们必须首先创建方法组。我们通过创建一个调用`UseMethod()`函数的函数来实现这一点，方法的名称作为封闭函数名称和第一个参数。完成后，我们就可以为`SimpleGenome`类创建通用函数，并通过简单地调用`genomeLength()`在对象上使用它。由于我们的通用函数只是将`chromosome_lengths`向量相加，我们得到如下结果：

```py
> genomeLength(athal)
[1] 134634692
```

*第 4 步*展示了类查找系统的机制。我们首先复制`iris`数据，然后在其上使用`summary()`方法，得到数据框的标准结果：

```py
> summary(some_data) 
Sepal.Length Sepal.Width Petal.Length Petal.Width Species 
Min. :4.300 Min. :2.000 Min. :1.000 Min. :0.100 setosa :50 
1st Qu.:5.100 1st Qu.:2.800 1st Qu.:1.600 1st Qu.:0.300 versicolor:50
```

接下来，在*第 4 步*中，我们使用`class()`函数为`some_data`对象添加了一个新类。注意，我们将其作为向量的第一个元素添加。我们可以看到，`data.frame`类仍然存在，但排在我们添加的类之后：

```py
> class(some_data) 
[1] "my_new_class" "data.frame"
```

然后，在*第 5 步*中，我们为`my_new_class`创建了一个通用的`summary()`函数，使其返回一个完全不同类型的摘要。我们可以看到，当我们调用它时：

```py
> summary(some_data) 
[1] "object contains 5 columns of classes:numeric,numeric,numeric,numeric,factor"
```

需要注意的一点是，尽管对象有多个类，但默认情况下会选择第一个与类匹配的通用函数。如果你想测试这一点，可以尝试交换`class`属性的顺序。

# 使用 S4 系统创建结构化和正式的对象

S4 是 S3 的更正式的对应版本，特别是因为它具有正式的类定义，所以不能随意使用，但它的工作方式与 S3 非常相似，因此我们已经学到的内容通常适用。在这个教程中，我们将快速介绍如何使用 S4 系统创建一个类似于本章前两节中`SimpleGenome`对象的类。了解 S4 会对你扩展`Bioconductor`（因为它是用 S4 编写的）有所帮助。

# 准备工作

再次，我们只使用基础 R，因此无需安装任何东西。

# 如何做到这一点...

使用 S4 系统创建结构化和正式的对象可以按照以下步骤进行：

1.  编写类定义：

```py
S4genome <- setClass("S4genome", slots = list(chromosome_count = "numeric", chromosome_lengths = "numeric" ))
```

1.  创建通用函数：

```py
setGeneric( "chromosome_count", 
 function(x){ standardGeneric("chromosome_count") }
)
```

1.  创建方法：

```py
setMethod( "chromosome_count", "S4genome", function(x){ slot(x, "chromosome_count")} )
```

# 它是如何工作的

这里的大纲与前两个教程非常相似。在*第 1 步*中，我们使用`setClass()`函数创建了类定义；第一个参数是类的名称，`slots`参数是一个适当的插槽名称列表以及每个插槽的类型。S4 类需要定义类型。使用中的对象可以像 S3 那样实例化：

```py
## > ecoli <- S4genome(chromosome_count = 1, chromosome_lengths = c(4600000) ) 
## > ecoli An object of class "S4genome" 
## Slot "chromosome_count": [1] 1 
## Slot "chromosome_lengths": [1] 4600000 
```

在*第 2 步*中，我们使用`setGeneric()`函数创建了一个通用函数`chromosome_count`，传入函数的名称和一个调用`standardGeneric()`函数的函数。这基本上是模板代码，因此请按照这个步骤进行，并在需要更多细节时查阅文档。

在*第 3 步*中，我们创建了方法。我们使用`setMethod()`函数创建了一个`chromosome_count`方法。第二个参数是这个方法将被调用的类，最后，我们传递了我们想要的代码。匿名函数仅仅调用了传递给它的对象上的`slot()`函数。`slot()`返回第二个参数中指定的槽的内容。

# 另请参见

如果你希望进一步了解 S4 来扩展 Bioconductor 类，请查看 Bioconductor 自己提供的教程：[`www.bioconductor.org/help/course-materials/2017/Zurich/S4-classes-and-methods.html`](https://www.bioconductor.org/help/course-materials/2017/Zurich/S4-classes-and-methods.html)。

# 分享和重用代码的简单方法

不可避免地，总会有这么一个时刻，你希望能够重用一些函数或类，而不必每次都重新输入（或者——更糟——复制粘贴）它们。将功能的唯一可靠版本放在一个地方，可以轻松管理，及时发现并解决代码中的错误和变化。因此，在这个教程中，我们将探讨两种简单的封装代码以便重用的方法。我们将讨论包创建的基础知识，尽管我们将创建的包将非常简陋，并且在考虑发布之前，需要进行大量的完善——特别是文档和测试方面。然而，以这种方式创建的包，将在你开发代码时提供帮助。

# 准备工作

为此，我们需要`devtools`和`usethis`包，以及本书仓库中`datasets/ch11`文件夹下的源代码文件`my_source_file.R`。

# 如何操作...

封装代码以便共享和重用可以通过以下步骤完成：

1.  加载现有的源代码文件：

```py
source(file.path(getwd(), "datasets", "ch11", "my_source_file.R"))
my_sourced_function()
```

1.  创建一个包骨架：

```py
usethis::create_package("newpackage")
```

1.  编写代码：

```py
my_package_function <- function(x){
 return( c("I come from a package!") )
}
```

1.  将包代码加载到内存中：

```py
devtools::load_all()
```

1.  将包安装到当前的 R 安装环境中：

```py
devtools::install()
library(newpackage)
```

# 它是如何工作的...

这段代码的第一步展示了一种非常有效但非常基础的加载外部代码的方法。我们使用 `source()` 函数将 R 代码文件加载到当前的命名空间中。这里的特定文件包含普通的 R 函数，除此之外没有其他内容。`source()` 函数仅仅是读取外部文件中的代码并将其执行，就像它是直接在当前控制台中输入的一样。由于文件中仅包含函数，因此你必须将这些函数加载到内存中以便立即使用。

*步骤 2* 更进一步，使用 `usethis::create_package()` 函数创建了一个简陋的包。该函数会创建一个你提供名称的新文件夹（在这个例子中是 `newpackage`），并将包所需的所有基本文件和文件夹放入其中。现在你可以在包的 `R/` 子文件夹中填充 R 代码，这些代码将在你加载包时加载。尝试使用 *步骤 3* 中的函数；将此函数添加到 `R/` 文件夹中的 `my_functions.R` 文件中。`R/` 文件夹中的文件名并不太重要，你可以有很多文件——但一定要确保它们以 `.R` 结尾。

*步骤 4* 将会使用 `devtools::load_all()` 函数加载你的源代码包到内存中。这大致模拟了我们调用 `library()` 函数时发生的情况，但并没有真正安装包。通过使用 `devtools::load_all()`，我们可以快速加载代码进行测试，而不必先安装它，这样如果我们需要更改代码，就不会有破损的版本被安装。我们没有提供任何参数，因此它会加载当前目录中的包（如果提供路径作为第一个参数，它会加载路径中找到的包）。

在 *第 5 步* 中，我们实际上将代码正确安装到 R 中。我们使用 `devtools::install()` 函数，它会构建包并将构建后的版本复制到 R 中的常规位置。现在，我们可以像加载任何其他包一样加载该构建版本 (`newpackage`)。请注意，这意味着我们有两个包的副本—一个是我们安装的版本，另一个是我们正在开发的版本。在开发更多代码并将其添加到包中时，你需要根据需要重复步骤四和五。

# 使用 devtools 从 GitHub 托管代码

良好的代码开发实践意味着将代码保存在某种版本控制系统中。Git 和 Git 共享网站 GitHub 是其中一种非常流行的系统。在本教程中，我们将使用 `usethis` 包添加一些有用的非代码文件，帮助描述其他用户如何重用我们的代码以及当前开发状态，并添加机制确保下游用户拥有与你的代码相关依赖的其他包。接下来，我们将看看如何将包上传到 GitHub 以及如何直接从 GitHub 安装。

# 准备工作

我们需要 `usethis` 和 `devtools` 包。

# 如何操作...

使用 `devtools` 从 GitHub 托管代码可以通过以下步骤完成：

1.  向包中添加一些有用的元数据和许可证文件：

```py
usethis::use_mit_license(name = "Dan MacLean")
usethis::use_readme_rmd()
usethis::use_lifecycle_badge("Experimental")
usethis::use_version()
```

1.  向依赖项列表中添加将在安装包时自动安装的内容：

```py
usethis::use_package("ggplot2")
```

1.  自动设置本地 Git 仓库并获取 GitHub 凭证：

```py
usethis::use_git()
usethis::browse_github_token() 
usethis::use_github()
```

1.  从 GitHub 安装包：

```py
devtools::install_github("user/repo")
```

# 工作原理...

*第 1 步* 中的代码非常简单，但它为包添加了很多内容。`usethis::use_mit_license()` 函数会添加一个名为 `LICENSE` 的文本文件，文件中包含 MIT 许可协议的内容。没有许可证文件，其他人很难知道在什么条件下可以使用该软件。MIT 许可协议是一种简单且宽松的许可，非常适合一般的开源软件，但也有其他替代方案；有关如何选择适合你的许可证，参见此网站：[`choosealicense.com/`](https://choosealicense.com/)。查看 `usethis` 文档，了解有关许可证的相关函数，允许你添加其他许可证类型。所有这些函数的参数名称允许你指定软件的版权持有者—如果你为公司或机构工作，法律版权可能属于他们，值得检查这一点。

`usethis::use_readme_rmd()` 函数会添加一个空的 `.Rmd` 文件，你可以在其中添加代码和文本，该文件将被构建成一个常规的 markdown 文件，并作为 `README` 文件显示在你仓库的 GitHub 前端页面上。最少在此文件中添加描述你的包的目标、基本用法和安装说明。

向文档中添加一个有用的信息是指明开发的阶段。`usethis::use_lifecycle_badge()`函数可以让你创建一个漂亮的小图形徽章，显示你的包的开发进度。你可以作为第一个参数使用的术语可以在这里查看：[`www.tidyverse.org/lifecycle/`](https://www.tidyverse.org/lifecycle/)。与此相关的是`usethis::use_version()`函数，它将帮助你递增软件的主要、次要或修复版本。

在*步骤 2*中，我们管理包所需的依赖项。当用户安装你的包时，包管理软件应会自动安装这些依赖项；R 要求它们放置在包元数据描述文件中的特定位置。`usethis::use_package()`函数会为你完成这项工作。

在*步骤 3*中，我们使用`usethis::use_git()`函数在当前目录创建一个本地的`git`仓库；它还会对当前代码进行初始提交。`usethis::browse_github_token()`函数会打开一个浏览器窗口，并导航到 GitHub 页面，允许你获取 GitHub 访问令牌，以便你的 R 会话能够与 GitHub 交互。获得令牌后，`usethis::use_github()`将把本地的`git`仓库上传至 GitHub，设置远程仓库并推送代码。你只需要执行一次。当`git`和 GitHub 仓库存在时，你需要手动管理版本，使用 RStudio 的`git`面板或`git`的命令行版本。

在*步骤 4*中，我们看到远程用户如何安装你的包，只需要使用`devtools::install_github()`函数，并提供适当的用户名和仓库名。

# 构建单元测试套件以确保函数按预期工作

大多数程序员会过度测试代码，单元测试的实践应运而生，让我们有了一种可以自动化并帮助减少构建中等复杂代码项目所需时间的正式测试方法。一个设计良好并维护的软件包，会为尽可能多的组件函数提供单元测试套件。在本食谱中，我们将学习如何使用`usethis`包添加组件文件和文件夹，以便创建一个使用`testthat`包的自动化测试套件。本书的范围不包括详细探讨为什么以及如何编写测试的哲学，但你可以查看`testthat`包的文档，网址是：[`testthat.r-lib.org/`](https://testthat.r-lib.org/)，作为一个很好的入门指南。

# 准备工作

我们需要`usethis`、`testthat`和`devtools`包。

# 如何操作...

使用以下步骤构建单元测试套件，以确保函数按预期工作：

1.  创建测试文件夹结构：

```py
usethis::use_testthat()
```

1.  添加新的测试：

```py
usethis::use_test("adds")

test_that("addition works", {
 expect_equal(1 + 1, 2)
})
```

1.  运行实际的测试：

```py
devtools::test()
```

# 它是如何工作的...

*第 1 步* 是一个典型的 `usethis` 风格函数，创建一些你的包所需的常见文件系统组件——`use_testthat()` 只是构建了 `testthat` 测试引擎所需要的文件夹结构。

*第 2 步* 中，`usethis::use_test()` 函数开始工作，创建一个测试文件——它使用函数参数的值作为文件名的后缀，因此在此情况下，使用 `adds` 作为参数，会在 `tests/testthat` 文件夹中创建一个名为 `test-adds.R` 的文件。然后，我们可以将 `tests` 添加到该文件中。每个测试将遵循此步骤第二行所示的基本模式。调用 `test_that()` 函数；其第一个参数是打印到控制台的文本，显示当前正在进行的测试。第二个参数是来自 `testthat` 包的断言块，比较函数的输出与预期值。如果两者匹配，则测试通过；否则，测试失败。`testthat` 中有许多断言，可以测试多种类型的输出和对象。你可以在文档中查看：[`testthat.r-lib.org/`](https://testthat.r-lib.org/)。请注意，测试应该在测试文件中完成并保存，而不是直接在控制台中输入。

在 *第 3 步* 中，我们在控制台使用 `devtools::test()` 函数自动运行测试套件。测试结果会打印到控制台，你可以根据需要修改函数，然后重新运行此步骤。

# 使用 Travis 的持续集成来保持代码经过测试并保持最新

**持续集成**（**CI**）是一种团队编程实践，旨在帮助大团队在同一个项目中保持代码、依赖关系和测试的最佳协作。为此开发的工具同样可以帮助我们管理自己的软件项目，解决由于自己更新、所依赖包的更新，甚至在某些情况下，R 和操作系统更新引发的问题。`Travis.CI` 是 `devtools` 包支持的一个 CI 服务。通过将 `Travis.CI` 集成到你的项目中，Travis 服务器将构建一个新的虚拟计算机，安装操作系统，安装 R 及你的包所需的所有依赖包，然后安装你的包并运行测试套件。Travis 会将结果发送给你。这个过程会周期性地重复，尤其是每次你向 GitHub 推送代码时，以便你及时了解哪些代码出现了问题，并能及早解决问题。在这个食谱中，我们将介绍如何为你的包项目设置 Travis。

# 准备工作

对于这个食谱，你需要 `usethis` 包和一个托管在 GitHub 上的包项目。如果你还没有设置这些，前面章节的食谱将帮助你完成这些步骤。

# 如何实现...

为了使用 Travis 的持续集成来保持代码经过测试并保持最新，我们需要创建一个 `.travis.yml` 文件：

```py
usethis::use_travis()
```

# 它是如何工作的...

这段代码中的唯一一行会在你的包目录的根目录中创建一个名为`.travis.yml`的文件。这个文件在 GitHub 上作为钩子使用，因此，一旦仓库被更新，`Travis.CI`服务器将会执行新的虚拟服务器构建和打包，并运行测试，然后将结果通过电子邮件发送到与你的 GitHub 账户关联的邮箱地址。尽管这只有一行代码，但这可能是本书中最具影响力的单行代码！`.travis.yml`文件包含了 Travis 构建的配置选项，并且可以添加很多内容来自定义输出。一个常见的添加内容如下：

```py
warnings_are_errors: false
```

这将告诉 Travis，R 代码中的警告不应视为错误，也不会导致构建失败。

构建可能需要一些时间；即便是简单的代码也可能需要 15 分钟。更复杂的项目将需要更长的时间。
