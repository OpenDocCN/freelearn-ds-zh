# 第十章：使用 Tidyverse 和 Bioconductor 编程

R 是一种适合交互式使用的优秀语言；然而，这也意味着许多用户没有体验过将其作为编程语言使用——也就是说，在自动化分析和节省重复工作所需时间和精力方面的应用。在本章中，我们将介绍一些实现这一目标的技术——特别是，我们将探讨如何将基础 R 对象集成到 `tidyverse` 工作流中，如何扩展 `Bioconductor` 类以满足我们自己的需求，以及如何使用文献编程和笔记本风格的编码来保持我们的工作记录具有表达性和可读性。

本章将介绍以下食谱：

+   使基础 R 对象整洁

+   使用嵌套数据框

+   为 `mutate` 编写函数

+   使用编程方式操作 Bioconductor 类

+   开发可重用的工作流和报告

+   使用 apply 系列函数

# 技术要求

你需要的示例数据可以从本书的 GitHub 仓库获取，链接为 [`github.com/PacktPublishing/R-Bioinformatics-Cookbook`](https://github.com/PacktPublishing/R-Bioinformatics-Cookbook)[.](https://github.com/danmaclean/R_Bioinformatics_Cookbook) 如果你希望按原样使用代码示例，你需要确保这些数据位于你的工作目录的子目录中。

以下是你需要的 R 包。通常，你可以使用 `install.packages("package_name")` 来安装这些包。列在 `Bioconductor` 下的包需要使用专用的安装程序进行安装。如果需要做其他操作，安装过程将在使用这些包的食谱中描述：

+   `Bioconductor`：

    +   `Biobase`

    +   `biobroom`

    +   `SummarizedExperiment`

+   `broom`

+   `dplyr`

+   `ggplot2`

+   `knitr`

+   `magrittr`

+   `purrr`

+   `rmarkdown`

+   `tidyr`

`Bioconductor` 非常庞大，并且有自己的安装管理器。你可以通过以下代码安装该管理器：

```py
if (!requireNamespace("BiocManager"))
    install.packages("BiocManager")
```

然后，你可以使用以下代码安装这些包：

```py
BiocManager::install("package_name")
```

更多信息请参见 [`www.bioconductor.org/install/`](https://www.bioconductor.org/install/)。

通常，在 R 中，用户会加载一个库并直接使用函数名。这在交互式会话中很方便，但当加载了许多包时，可能会导致混淆。为了明确当前使用的是哪个包和函数，我偶尔会使用 `packageName::functionName()` 这种约定。

有时，在某个代码片段中，我会暂停代码执行，以便你能够看到一些中间输出或理解某个重要对象的结构。每当发生这种情况时，你会看到一个代码块，其中每行前面都以 `##`（双井号）符号开头。考虑以下命令：

`letters[1:5]`

这将给我们以下输出：

`## a b c d e`

请注意，输出行前缀为 `##`。

# 使基础 R 对象整洁

`tidyverse`软件包集（包括`dplyr`、`tidyr`和`magrittr`）通过应用整洁的工作方式，在数据处理和分析中对 R 语言产生了巨大影响。本质上，这意味着数据保持在一种特定的整洁格式中，每一行包含一个单独的观测值，每一列包含一个变量的所有观测值。这样的结构意味着分析步骤有可预测的输入和输出，可以构建成流畅且富有表现力的管道。然而，大多数基础 R 对象并不整洁，往往需要大量的编程工作来提取所需的部分，以便下游使用。在这个教程中，我们将介绍一些函数，用于自动将一些常见的基础 R 对象转换为整洁的数据框。

# 准备工作

我们将需要`tidyr`、`broom`和`biobroom`包。我们将使用内置的`mtcars`数据和来自本书仓库`datasets/ch1`文件夹中的`modencodefly_eset.RData`。

# 如何操作...

使基础 R 对象整洁可以通过以下步骤完成：

1.  整理一个`lm`对象：

```py
library(broom)
model <- lm(mpg ~ cyl + qsec, data = mtcars)
tidy(model) 
augment(model)
glance(model)
```

1.  整理一个`t_test`对象：

```py
t_test_result <- t.test(x = rnorm(20), y = rnorm(20) )
tidy(t_test_result)
```

1.  整理一个 ANOVA 对象：

```py
anova_result <- aov(Petal.Length ~ Species, data = iris)
tidy(anova_result)
post_hoc <- TukeyHSD(anova_result)
tidy(post_hoc)
```

1.  整理一个`Bioconductor` `ExpressionSet`对象：

```py
library(biobroom)
library(Biobase)
load(file.path(getwd(), "datasets", "ch1", "modencodefly_eset.RData") ) 
tidy(modencodefly.eset, addPheno = TRUE)
```

# 它是如何工作的...

*步骤 1* 展示了使用`lm()`函数整理`lm`对象的一些函数。第一步是创建对象。在这里，我们使用`mtcars`数据执行一个多元回归模型。然后，我们对该模型使用`tidy()`函数，返回模型组件的对象摘要，例如系数，以整洁的数据框形式。`augment()`函数返回额外的逐观测数据，如果需要的话——同样是整洁格式。`glance()`函数检查模型本身，并返回关于模型的摘要——自然地，以整洁格式显示。`glance()`对于比较模型非常有用。

*步骤 2* 展示了相同的过程，适用于`t.test`对象。首先，我们对两个随机数向量进行 t 检验。`tidy()`函数会将所有细节以整洁的数据框形式返回。

在*步骤 3*中，我们对`iris`数据运行方差分析（ANOVA）。我们使用`aov()`函数查看`Species`对`Petal.Length`的影响。我们可以再次对结果使用`tidy()`，但它会给出模型组件的摘要。实际上，我们更感兴趣的可能是来自后验检验的比较结果，该检验使用`TukeyHSD()`函数进行，它也可以在`tidy()`中使用。

在*步骤 4*中，我们使用`biobroom`版本的`tidy()`对`ExpressionSet`对象进行处理。这将表达值的方阵转化为整洁的数据框，并包含样本和其他数据类型的列。额外的参数`addPheno`是特定于此类对象的，并将`ExpressionSet`元数据容器中的表型元数据插入其中。请注意，结果数据框超过 200 万行——生物学数据集可能很大，并且生成非常大的数据框。

# 使用嵌套数据框

数据框（dataframe）是整洁工作方式的核心，我们通常将其视为一个类似电子表格的矩形数据容器，每个单元格中仅包含一个值。实际上，数据框可以嵌套——即它们可以在特定的单个单元格中包含其他数据框。这是通过将数据框的向量列替换为列表列来实现的。每个单元格实际上是列表的一个成员，因此任何类型的对象都可以保存在外部数据框的概念性单元格中。在本教程中，我们将介绍创建嵌套数据框的方式以及与之合作的不同方法。

# 准备工作

我们将需要`tidyr`、`dplyr`、`purrr`和`magrittr`库。我们还将使用来自`ggplot2`包的`diamonds`数据，但我们不会使用任何函数。

# 工作原理...

使用嵌套数据框可以通过以下步骤完成：

1.  创建一个嵌套数据框：

```py
library(tidyr)
library(dplyr)
library(purrr)
library(magrittr)
library(ggplot2)

nested_mt <- nest(mtcars, -cyl)
```

1.  添加一个新列表列，包含`lm()`的结果：

```py
nested_mt_list_cols <- nested_mt %>% mutate(
 model = map(data, ~ lm(mpg ~ wt, data = .x))
)
```

1.  添加一个新列表列，包含`tidy()`的结果：

```py
nested_mt_list_cols <- nested_mt_list_cols %>% mutate(
 tidy_model = map(model, tidy)
)
```

1.  解开整个数据框：

```py
unnest(nested_mt_list_cols, tidy_model)
```

1.  在单个步骤中运行管道：

```py
models_df <- nest(mtcars, -cyl) %>%
 mutate(
 model = map(data, ~ lm(mpg ~ wt, data = .x)),
 tidy_model = map(model, tidy)
 ) %>%
 unnest(tidy_model)
```

# 工作原理...

在*步骤 1*中，我们使用`nest()`函数将`mtcars`数据框进行嵌套。`-`选项告诉函数哪些列在嵌套时要排除；将`cyl`列转换为因子，用于创建不同的子集。从概念上讲，这类似于`dplyr::group_by()`函数。检查该对象会得到以下内容：

```py
 A tibble: 3 x 2
## cyl   data 
## <dbl> <list> 
## 1 6   <tibble [7 × 10]> 
## 2 4   <tibble [11 × 10]>
## 3 8   <tibble [14 × 10]>
```

嵌套数据框包含一个名为`data`的新数据框列，以及被简化的`cyl`列。

在*步骤 2*中，我们通过使用`mutate()`在数据框中创建一个新列。在此过程中，我们使用`purrr`包中的`map()`函数，它遍历作为第一个参数提供的列表项（即我们的数据框列），并在作为第二个参数提供的代码中使用它们。在这里，我们对嵌套数据使用`lm()`函数，一次处理一个元素——注意，`.x`变量表示*我们当前正在处理的内容*——也就是列表中的当前项。运行后，列表看起来是这样的：

```py
##  cyl   data              model 
##  <dbl> <list> <list>
## 1 6   <tibble [7 × 10]>  <lm> 
## 2 4   <tibble [11 × 10]> <lm> 
## 3 8   <tibble [14 × 10]> <lm>
```

新的`model`列表列包含我们的`lm`对象。

在确认添加新列表列的模式是使用`mutate()`并在其中嵌入`map()`后，我们可以同样整理`lm`对象。这就是在*步骤 3*中发生的情况。结果给我们带来了以下嵌套数据框：

```py
##  cyl   data              model  tidy_model 
## <dbl> <list>            <list>  <list> 
## 1 6   <tibble [7 × 10]>  <lm>   <tibble [2 × 5]>
## 2 4   <tibble [11 × 10]> <lm>   <tibble [2 × 5]>
## 3 8   <tibble [14 × 10]> <lm>   <tibble [2 × 5]>
```

*步骤 4*使用`unnest()`函数将所有内容恢复为单个数据框；第二个参数`tidy_model`是需要解包的列。

*步骤 5*将*步骤 1*到*步骤 4*的整个过程合并为一个管道，突出显示这些只是常规的`tidyverse`函数，并且可以链式调用，无需保存中间步骤。

# 还有更多内容...

`unnest()`函数只有在嵌套列表列成员兼容且可以根据正常规则合理对齐和回收时才有效。在许多情况下，这种情况并不成立，因此你需要手动操作输出。以下示例展示了我们如何做到这一点。工作流程基本上与之前的示例相同，唯一的变化是在早期步骤中，我们使用`dplyr::group_by()`来创建`nest()`的分组。在`mutate()`中，我们传递自定义函数来分析数据，但其他步骤保持不变。最后一步是最大的变化，利用`transmute()`来删除不需要的列，并创建一个新的列，它是`map_dbl()`和自定义汇总函数的结果。`map_dbl()`类似于`map()`，但只返回双精度数值向量。其他的`map_**`函数也存在。

# 为`dplyr::mutate()`编写函数

`dplyr`的`mutate()`函数非常有用，可以根据现有列的计算结果向数据框中添加新列。它是一个矢量化函数，但通常被误解为按行工作，实际上它是按列工作的，也就是说，它对整个向量进行操作，并利用 R 内置的回收机制。这种行为往往会让那些想在复杂例子或自定义函数中使用`mutate()`的人感到困惑，因此，在本教程中，我们将探讨`mutate()`在某些情况下的实际行为，希望能够带来启示。

# 准备工作

为此，我们需要`dplyr`包和内置的`iris`数据。

# 如何做……

为`dplyr::mutate()`编写函数可以通过以下步骤完成：

1.  使用返回单个值的函数：

```py
return_single_value <- function(x){
 sum(x) 
}
iris %>% mutate(
 result = return_single_value(Petal.Length)
)
```

1.  使用返回与给定值相同数量值的函数：

```py
return_length_values <- function(x){
 paste0("result_", 1:length(x))
}
iris %>% mutate(
 result = return_length_values(Petal.Length)
)
```

1.  使用返回既不是单个值也不是与给定值相同数量值的函数：

```py
return_three_values <- function(x){
 c("A", "b", "C")
}
iris %>% mutate(
 result = return_three_values(Petal.Length)
)
```

1.  强制函数的重复以适应向量的长度：

```py
rep_until <- function(x){
 rep(c("A", "b", "C"), length.out = length(x))
}
iris %>% mutate(
 result = rep_until(Petal.Length)
)
```

# 它是如何工作的……

在*步骤 1*中，我们创建一个函数，给定一个向量，返回一个单一的值（长度为 1 的向量）。然后我们在`mutate()`中使用它，添加一个名为`result`的列，并得到如下结果：

```py
## Sepal.Length Sepal.Width Petal.Length Petal.Width Species result 
## 1 5.1 3.5 1.4 0.2 setosa 563.7 
## 2 4.9 3.0 1.4 0.2 setosa 563.7 
## 3 4.7 3.2 1.3 0.2 setosa 563.7 
## 4 4.6 3.1 1.5 0.2 setosa 563.7
```

请注意，函数在`result`列中返回的单个值是反复出现的。对于`length == 1`的向量，R 会回收结果并将其放置到每个位置。

在*步骤 2*中，我们走到对立面，创建一个函数，给定一个向量，返回一个相同长度的向量（具体来说，它返回一个将`result_`与向量中的位置数字拼接在一起的向量）。当我们运行它时，我们得到如下结果：

```py
## Sepal.Length Sepal.Width Petal.Length Petal.Width Species result 
## 1 5.1 3.5 1.4 0.2 setosa result_1 
## 2 4.9 3.0 1.4 0.2 setosa result_2 
## 3 4.7 3.2 1.3 0.2 setosa result_3 
## 4 4.6 3.1 1.5 0.2 setosa result_4
```

由于它的长度与数据框中其余列完全相同，R 会接受它并将其作为新列应用。

在*步骤 3*中，我们创建一个返回三个元素的向量的函数。由于长度既不是 1，也不是数据框中其他列的长度，代码会失败。

在*第 4 步*中，我们将探讨如何重复一个不兼容长度的向量，以便在需要时使其适应。`rep_until()`函数与`length.out`参数会重复其输入，直到向量的长度为`length.out`。通过这种方式，我们可以得到以下列，这就是我们在*第 3 步*中使用该函数时所期望看到的结果：

```py
## Sepal.Length Sepal.Width Petal.Length Petal.Width Species result 
## 1 5.1 3.5 1.4 0.2 setosa A 
## 2 4.9 3.0 1.4 0.2 setosa b 
## 3 4.7 3.2 1.3 0.2 setosa C 
## 4 4.6 3.1 1.5 0.2 setosa A 
## 5 5.0 3.6 1.4 0.2 setosa b
```

# 使用 Bioconductor 类进行编程操作

`Bioconductor`的广泛应用意味着有大量的类和方法可以完成几乎任何你想要的生物信息学工作流程。不过，有时候，额外的数据槽或一些工具上的调整会帮助简化我们的工作。在本教程中，我们将探讨如何扩展一个现有的类，以包含一些特定于我们数据的额外信息。我们将扩展`SummarizedExperiment`类，以添加假设的条形码信息——一种元数据，表示一些核苷酸标签，这些标签可以标识包含在序列读取中的样本。

# 准备工作

对于本教程，我们只需要`Bioconductor`的`SummarizedExperiment`包。

# 如何操作...

使用以下步骤可以通过编程方式与`Bioconductor`类进行交互：

1.  创建一个继承自`SummarizedExperiment`的新类：

```py
setClass("BarcodedSummarizedExperiment",
   contains = "SummarizedExperiment",
   slots = c(barcode_id = "character", barcode_sequence = "character")
 )
```

1.  创建构造函数：

```py
BarcodedSummarizedExperiment <- function(assays, rowRanges, colData, barcode_id, barcode_sequence){
   new("BarcodedSummarizedExperiment", 
       SummarizedExperiment(assays=assays, rowRanges=rowRanges, colData=colData),
       barcode_id = barcode_id,
       barcode_sequence = barcode_sequence
   )
}
```

1.  向类中添加必需的方法：

```py
setGeneric("barcode_id", function(x) standardGeneric("barcode_id"))
setMethod("barcode_id", "BarcodedSummarizedExperiment", function(x) x@barcode_id )
```

1.  构建新类的实例：

```py
nrows <- 200
ncols <- 6
counts <- matrix(runif(nrows * ncols, 1, 1e4), nrows)
assays <- list(counts = counts)
rowRanges <- GRanges(    rep(c("chr1", "chr2"), c(50, 150)),
                         IRanges(floor(runif(200, 1e5, 1e6)), width=100),
                         strand=sample(c("+", "-"), 200, TRUE),
                         feature_id=sprintf("ID%03d", 1:200)
)
colData <- DataFrame(
                Treatment=rep(c("ChIP", "Input"), 3),
                row.names=LETTERS[1:6]
)

my_new_barcoded_experiment <- BarcodedSummarizedExperiment(
        assays = assays, 
        rowRanges = rowRanges, 
        colData = colData, 
        barcode_id = letters[1:6], 
        barcode_sequence = c("AT", "GC", "TA", "CG","GA", "TC") 
)
```

1.  调用新的方法：

```py
barcode_id(my_new_barcoded_experiment)
```

# 它是如何工作的...

在*第 1 步*中，我们使用`setClass()`函数创建一个新的 S4 类。这个函数的第一个参数是新类的名称。`contains`参数指定我们希望继承的现有类（这样我们的新类就会包含这个类的所有功能以及我们创建的任何新功能）。`slots`参数指定我们希望添加的新数据槽，并要求我们为其指定类型。在这里，我们添加了文本数据槽，用于新的`barcode_id`和`barcode_sequence`槽，因此我们为这两个槽都使用`character`类型。

在*第 2 步*中，我们创建一个构造函数。该函数的名称必须与类名相同，并且我们在调用`function()`时指定创建新对象所需的参数。在函数体内，我们使用`new()`函数，其第一个参数是要实例化的类的名称。其余的部分用于填充实例数据；我们调用继承自`SummarizedExperiment`的构造函数，以填充新对象的那一部分，然后手动填充新的条形码槽。每次运行`BarcodedSummarizedExperiment`时，我们都会获得该类的一个新对象。

在*步骤 3*中，我们添加了一个新函数（严格来说，在 R 中，它被称为方法）。如果我们选择一个在 R 中尚不存在的`Generic`函数名称，我们必须使用`setGeneric()`注册该函数名，`setGeneric()`的第一个参数是函数名，第二个参数是一个模板函数。`Generic`函数设置完毕后，我们可以使用`setMethod()`函数添加实际的函数。新函数的名称是第一个参数，它将附加到的类是第二个参数，而代码本身是第三个参数。请注意，我们这里只是在创建一个访问器（`getter`）函数，它返回当前对象中`barcode_id`槽的数据。

在*步骤 4*中，我们的准备工作已经完成，可以构建类的实例。在这一步的前六行中，我们仅仅创建了构建对象所需的数据。这部分数据进入一个普通的`SummarizedExperiment`对象；你可以在文档中看到更多关于这里具体发生了什么的细节。然后，我们可以通过调用`BarcodedSummarizedExperiment`函数，并传入我们创建的数据以及新`barcode_id`和`barcode_sequence`槽的特定数据，实际创建`my_new_barcoded_experiment`。

现在，创建了对象后，在*步骤 5*中，我们可以像调用其他函数一样使用我们的方法，传入我们新创建的对象作为参数。

# 开发可重用的工作流程和报告

在生物信息学中，一个非常常见的任务是撰写结果，以便与同事交流，或者仅仅为了在实验记录中留有一份完整的记录（无论是电子版还是纸质版）。一个关键技能是尽可能让工作可重复，以便当我们需要回顾它时，或者有其他人对我们所做的工作感兴趣时，他们可以复制整个过程。一个日益流行的解决方案是使用文献编程技术和可执行笔记本，这些笔记本结合了可读的文本、分析代码和计算输出，打包成一个单一文档。在 R 中，`rmarkdown`包使我们能够以这种方式将代码和文本结合在一起，并生成多种格式的输出文档。

在这个食谱中，我们将查看一个可以通过`rmarkdown`编译的文档的大规模结构。RStudio 应用程序使得这个过程非常简便，所以我们将通过这个工具来查看编译过程。

# 准备工作

对于这个食谱，我们需要[RStudio 应用程序](https://www.rstudio.com/)和`rmarkdown`包。这个食谱的示例代码可以在本书的`datasets/ch10/`文件夹中的`example_rmarkdown.rmd`文件中找到。

# 如何做到这一点...

开发可重用的工作流程和报告可以通过以下步骤完成：

1.  在外部文件中，添加一个`YAML`头部：

```py
---
title: "R Markdown Report"
author: "R Bioinformatics Cookbook"
date: "`r format(Sys.time(), '%d %B, %Y')`"
output:
 html_document:
 df_print: paged
 bookdown::html_document2:
 fig_caption: yes
 keep_md: yes
 toc: yes
---
```

1.  然后，添加一些文本和代码进行解释：

```py
We can include text and create code blocks, the code gets executed and the result passed in

```{r}

x <- iris$Sepal.Width

y <- iris$Sepal.Length

lm(y ~ x, data = iris)

```py
```

1.  文本可以使用最小化标记格式进行格式化：

```py
## We can format text using Markdown
We can create many text formats including *italics* and **bold**,
We can make lists 
 1\. First item
 2\. Second item
```

1.  在块内应用更多选项并传递变量：

```py
The whole document acts as a single R session - so variables created in earlier blocks can still be used later.
Plots are presented within the document. Options for blocks can be set in the header

```{r, fig.width=12 }

plot(x, y)

```py
```

# 它是如何工作的…

这里的代码很特殊，因为它必须从外部文档中运行；在 R 控制台中无法运行。运行文档的编译步骤有几种方法。在 RStudio 中，一旦安装了`rmarkdown`并且正在编辑一个`.Rmd`扩展名的文档，您会看到一个`knit`按钮。或者，您可以通过控制台使用`rmarkdown::render()`函数编译文档，尽管我建议使用 RStudio IDE 来完成此操作。

在*步骤 1*中，我们创建了一个`YAML`头部，描述了文档的渲染方式，包括输出格式、动态日期插入、作者和标题。这些内容将自动添加到您的文档中。

在*步骤 2*中，我们实际上创建了一些内容——第一行是纯文本，将作为段落文本传递到最终文档中，不做任何修改。块中的部分由```py` ``` ```py` is code to be interpreted. Options for the block go inside the curly brackets—here, `{r}` means this should be an R code block (some other languages are supported too). The code in this block is run in a new R session, its output captured; and inserted immediately after the code block.

In *Step 3*, we create some plaintext with the `Markdown` tags. `##` gives us a line with a second-level heading, the `**starred**` text gives us different formatting options, and we can also create lists. Valid `Markdown` is interpreted and the reformatted text is passed into the eventual document.

In *Step 4*, we start with some more plaintext and follow with a new code block. The options for the code block are set in the curly brackets again—here, we set a width for figures in the plot. Note that the code in this block refers to variables created in an earlier block. Although the document creates a new R session without access to variables already in the usual console, the document itself is a single session so blocks can access earlier block's variables, allowing the code and text to be mixed up at whatever resolution the author requires. Finally, the resulting figure is inserted into the document just like code. 

# Making use of the apply family of functions

Programming in R can sometimes seem a bit tricky; the control flow and looping structures it has, are a bit more basic than in other languages. As many R functions are vectorized, the language actually has some features and functions; that mean we don't need to take the same low-level approach we may have learned in Python or other places. Instead, base R provides the `apply` functions to do the job of common looping tasks. These functions all have a loop inside them, meaning we don't need to specify the loop manually. In this recipe, we'll look at using some `apply` family functions with common data structures to loop over them and get a result. The common thread in all of the `apply` functions is that we have an input data structure that we're going to iterate over and some code (often wrapped in a function definition) that we're going to apply to each item of the structure.

# Getting ready

We will only need base R functions and data for this recipe, so you are good to go!

# How to do it...

Making use of the `apply` family of functions can be done using the following steps:

1.  Create a matrix and use `apply` to work on it:

```限定

m <- matrix(rep(1:10, 10, replace = TRUE), nrow = 10)

apply(m, 1, sum)

apply(m, 2, sum)

```py

2.  Use `lapply` over the vector:

```

numbers <- 1:3

number_of_numbers <- function(x){

rnorm(x)

}

my_list <- lapply(numbers, number_of_numbers)

```py

3.  Use `lapply` and `sapply` over the list:

```

summary_function <- function(x){

mean(x)

}

lapply(my_list, summary_function)

sapply(my_list, summary_function)

```py

4.  Use `lapply` over a dataframe:

```

list_from_data_frame <- lapply(iris, mean, trim = 0.1, na.rm = TRUE )

unlist(list_from_data_frame)

```py

# How it works...

*Step 1* begins with the creation of a 10 x 10 matrix, with rows holding the same number and columns running from 1 to 10\. Inspecting it makes it clear, as partly shown in the following output:

```

## > m

## [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]

## [1,] 1 1 1 1 1 1 1 1 1 1

## [2,] 2 2 2 2 2 2 2 2 2 2

## [3,] 3 3 3 3 3 3 3 3 3 3

```py

We then use `apply()`: the first argument is the object to loop over, the second is the direction to loop in (or margin, 1 = rows, and 2 = columns), and the third is the code to apply. Here, it's the name of a built-in function, but it could be a custom one. Note it's the margin argument that affects the amount of data that is taken each time. Contrast the two `apply()` calls:

```

> apply(m, 1, sum)

[1] 10 20 30 40 50 60 70 80 90 100

> apply(m, 2, sum)

[1] 55 55 55 55 55 55 55 55 55 55

```py

Clearly, `margin = 1` is taking each row at a time, whereas `margin = 2` is taking the columns. In any case, `apply()` returns a vector of results, meaning the results must be of the same type each time. It is not the same shape as the input data.

With *Step 2*, we move onto using `lapply()`, which can loop over many types of data structures, but always returns a list with one member for each iteration. Because it's a list, each member can be of a different type. We start by creating a simple vector containing the integers 1 to 3 and a custom function that just creates a vector of random numbers of a given length. Then, we use `lapply()` to apply that function over the vector; the first argument to `lapply()` is the thing to iterate over, and the second is the code to apply. Note that the current value of the vector we're looping over is passed automatically to the called function as the argument. Inspecting the resulting list, we see the following:

```

>my_list

[[1]] [1] -0.3069078

[[2]] [1] 0.9207697 1.8198781

[[3]] [1] 0.3801964 -1.3022340 -0.8660626

```py

We get a list of one random number, then two, then three, reflecting the change in the original vector.

In *Step 3*, we see the difference between `lapply()` and `sapply()` when running over the same object. Recall `lapply()` always returns a list but `sapply()` can return a vector (`s` can be thought of as standing for *simplify*). We create a simple summary function to ensure we only get a single value back and `sapply()` can be used. Inspecting the results, we see the following:

```

>lapply(my_list, summary_function)

[[1]] [1] -0.3069078

[[2]] [1] 1.370324

[[3]] [1] -0.5960334

>sapply(my_list, summary_function)

[1] -0.3069078 1.3703239 -0.5960334

```py

Finally, in *Step 4*, we use `lapply()` over a dataframe, namely, the built-in `iris` data. By default, it applies to columns on a dataframe, applying the `mean()` function to each one in turn. Note the last two arguments (`trim` and `na.rm`) are not arguments for `lapply()`, though, it does look like it. In all of these functions, the arguments after the vector to iterate over and the code (in other words, argument positions 1 and 2) are all passed to the code being run—here, our `mean()` function. The column names of the dataframe are used as the member names for the list. You may recall that one of the columns in `iris` is categorical, so `mean()` doesn't make much sense. Inspect the result to see what `lapply()` has done in this case:

```

> lapply(iris, mean, trim = 0.1, na.rm = TRUE )

$Sepal.Length [1] 5.808333

$Sepal.Width [1] 3.043333

$Petal.Length [1] 3.76

$Petal.Width [1] 1.184167

$Species [1] NA

```py

It has returned `NA`. Also, it has generated a warning but not failed. This can be a source of bugs in later analyses.

With a simple list like this, we can also use `unlist()` to get a vector of the results:

```

> unlist(list_from_data_frame)

Sepal.Length Sepal.Width Petal.Length Petal.Width Species

5.808333 3.043333 3.760000 1.184167 NA

```

如果存在名称，向量将被命名。
