

# 第八章：机器学习模型的超参数调整

本章介绍了如何通过调整模型的超参数，使用遗传算法来提高**监督学习模型**的性能。本章将首先简要介绍机器学习中的**超参数调整**，然后介绍**网格搜索**的概念。在介绍 Wine 数据集和自适应提升分类器后，二者将在本章中反复使用，我们将展示如何通过传统的网格搜索和遗传算法驱动的网格搜索进行超参数调整。最后，我们将尝试通过直接的遗传算法方法来优化超参数调整结果，从而提升性能。

到本章结束时，你将能够完成以下任务：

+   演示对机器学习中超参数调整概念的熟悉度

+   演示对 Wine 数据集和自适应提升分类器的熟悉度

+   使用超参数网格搜索提升分类器的性能

+   使用遗传算法驱动的超参数网格搜索提升分类器的性能

+   使用直接的遗传算法方法提升分类器的性能，以进行超参数调整

本章将以快速概述机器学习中的超参数开始。如果你是经验丰富的数据科学家，可以跳过引言部分。

# 技术要求

本章将使用 Python 3，并配备以下支持库：

+   **deap**

+   **numpy**

+   **pandas**

+   **matplotlib**

+   **seaborn**

+   **scikit-learn**

重要说明

如果你使用我们提供的**requirements.txt**文件（参见*第三章*），这些库已经包含在你的环境中。

此外，我们将使用 UCI Wine 数据集：[`archive.ics.uci.edu/ml/datasets/Wine`](https://archive.ics.uci.edu/ml/datasets/Wine)

本章中使用的程序可以在本书的 GitHub 仓库中找到：

[`github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/tree/main/chapter_08`](https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/tree/main/chapter_08)

查看以下视频，观看代码演示：[`packt.link/OEBOd`](https://packt.link/OEBOd)

# 机器学习中的超参数

在*第七章*《使用特征选择提升机器学习模型》中，我们将*监督学习*描述为调整（或调整）模型内部参数的程序化过程，以便在给定输入时产生期望的输出。为了实现这一目标，每种类型的监督学习模型都配有一个学习算法，在*学习*（或训练）阶段反复调整其内部参数。

然而，大多数模型还有另一组参数是在学习发生之前设置的。这些参数被称为 **超参数**，并且它们影响学习的方式。以下图示了这两类参数：

![图 8.1：机器学习模型的超参数调优](img/B20851_08_001.jpg)

图 8.1：机器学习模型的超参数调优

通常，超参数有默认值，如果我们没有特别设置，它们将会生效。例如，如果我们查看 `scikit-learn` 库中 **决策树分类器** 的实现（[`scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)），我们会看到几个超参数及其默认值。

以下表格描述了一些超参数：

| **名称** | **类型** | **描述** | **默认值** |
| --- | --- | --- | --- |
| `max_depth` | 整数 | 树的最大深度 | 无 |
| `splitter` | 枚举型 | 用于选择每个最佳节点分裂的策略：`{'best', 'random'}` | `'best'` |
| `min_samples_split` | 整数或浮动型 | 分裂内部节点所需的最小样本数 | `2` |

表 8.1：超参数及其详细信息

这些参数每个都会影响决策树在学习过程中构建的方式，它们对学习过程结果的综合影响——从而对模型的表现——可能是显著的。

由于超参数的选择对机器学习模型的性能有着重要影响，数据科学家通常会花费大量时间寻找最佳超参数组合，这个过程称为 **超参数调优**。一些用于超参数调优的方法将在下一小节中介绍。

## 超参数调优

寻找超参数良好组合的常见方法是使用 `{2, 5, 10}` 来设置 `max_depth` 参数，而对于 `splitter` 参数，我们选择两个可能的值——`{"best", "random"}`。然后，我们尝试所有六种可能的组合。对于每个组合，分类器会根据某个性能标准（例如准确度）进行训练和评估。在过程结束时，我们选择出表现最好的超参数组合。

网格搜索的主要缺点是它对所有可能的组合进行穷举搜索，这可能非常耗时。生成良好组合的常见方法之一是 **随机搜索**，它通过选择和测试随机组合的超参数来加速过程。

对我们特别有意义的一个更好选择是在进行网格搜索时，利用遗传算法来寻找在预定义网格中超参数的最佳组合。这种方法比原始的全面网格搜索在更短时间内找到最佳组合的潜力更大。

虽然`scikit-learn`库支持网格搜索和随机搜索，但`sklearn-deap`提供了一个遗传算法驱动的网格搜索选项。这个小型库基于 DEAP 遗传算法的能力，并结合了`scikit-learn`现有的功能。在撰写本书时，这个库与`scikit-learn`的最新版本不兼容，因此我们在*第八章*的文件中包含了一个稍作修改的版本，并将使用该版本。

在接下来的章节中，我们将比较两种网格搜索方法——全面搜索和遗传算法驱动的搜索。但首先，我们将快速了解一下我们将在实验中使用的数据集——**UCI** **葡萄酒数据集**。

## 葡萄酒数据集

一个常用的数据集来自*UCI 机器学习库*（[`archive.ics.uci.edu/`](https://archive.ics.uci.edu/)），葡萄酒数据集（[`archive.ics.uci.edu/ml/datasets/Wine`](https://archive.ics.uci.edu/ml/datasets/Wine)）包含对 178 种在意大利同一地区种植的葡萄酒进行的化学分析结果。这些葡萄酒被分为三种类型之一。

化学分析由 13 个不同的测量组成，表示每种葡萄酒中以下成分的含量：

+   酒精

+   苹果酸

+   灰分

+   灰分的碱度

+   镁

+   总酚

+   类黄酮

+   非类黄酮酚

+   原花青素

+   色度

+   色调

+   稀释葡萄酒的 OD280/OD315

+   脯氨酸

数据集的`2`到`14`列包含前述测量值，而分类结果——即葡萄酒类型本身（`1`、`2`或`3`）——则位于第一列。

接下来，让我们看看我们选择的分类器，用于对这个数据集进行分类。

## 自适应提升分类器

**自适应提升算法**，简称**AdaBoost**，是一种强大的机器学习模型，通过加权求和结合多个简单学习算法（**弱学习器**）的输出。AdaBoost 在学习过程中逐步添加弱学习器实例，每个实例都会调整以改进先前分类错误的输入。

`scikit-learn`库实现的此模型——AdaBoost 分类器（[`scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)）——使用了多个超参数，其中一些如下：

| **名称** | **类型** | **描述** | **默认值** |
| --- | --- | --- | --- |
| `n_estimators` | 整数类型 | 最大估算器数量 | `50` |
| `learning_rate` | 浮动类型 | 每次提升迭代中应用于每个分类器的权重；较高的学习率增加每个分类器的贡献 | `1.0` |
| `algorithm` | 枚举类型 | 使用的提升算法：`{'SAMME' , 'SAMME.R'}` | `'SAMME.R'` |

表 8.1：超参数及其详细信息

有趣的是，这三个超参数各自具有不同的类型——一个是整数类型，一个是浮动类型，一个是枚举（或分类）类型。稍后我们将探讨每种调优方法如何处理这些不同类型的参数。我们将从两种网格搜索形式开始，下一节将描述这两种形式。

# 使用传统的与遗传网格搜索相比，调整超参数

为了封装通过网格搜索调优 AdaBoost 分类器的超参数，我们创建了一个名为 `HyperparameterTuningGrid` 的 Python 类，专门用于 Wine 数据集。此类位于 `01_hyperparameter_tuning_grid.py` 文件中，具体位置为：

[`github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_08/01_hyperparameter_tuning_grid.py`](https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_08/01_hyperparameter_tuning_grid.py)。

该类的主要部分如下所示：

1.  类的 **__init__()** 方法初始化葡萄酒数据集、AdaBoost 分类器、k 折交叉验证指标和网格参数：

    ```py
    self.initWineDataset()
    self.initClassifier()
    self.initKfold()
    self.initGridParams()
    ```

1.  **initGridParams()** 方法通过设置前一节中提到的三个超参数的测试值来初始化网格搜索：

    +   **n_estimators** 参数在 10 个值之间进行了测试，这些值在 10 和 100 之间均匀分布。

    +   **learning_rate** 参数在 100 个值之间进行了测试，这些值在 0.1 (10^−2) 和 1 (10⁰) 之间对数均匀分布。

    +   **algorithm** 参数的两种可能值，**'SAMME'** 和 **'SAMME.R'**，都进行了测试。

    此设置覆盖了 200 种不同的网格参数组合（10×10×2）：

    ```py
    self.gridParams = {
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'learning_rate': np.logspace(-2, 0, num=10, base=10),
        'algorithm': ['SAMME', 'SAMME.R'],
    }
    ```

1.  **getDefaultAccuracy()** 方法使用 **'****准确度'** 指标的均值评估分类器在其默认超参数值下的准确度：

    ```py
    cv_results = model_selection.cross_val_score(
        self.classifier,
        self.X,
        self.y,
        cv=self.kfold,
        scoring='accuracy')
    return cv_results.mean()
    ```

1.  **gridTest()** 方法在我们之前定义的测试超参数值集合上执行传统网格搜索。最优的参数组合是基于 k 折交叉验证的平均 **'****准确度'** 指标来确定的：

    ```py
    gridSearch = GridSearchCV(
        estimator=self.classifier,
        param_grid=self.gridParams,
        cv=self.kfold,
        scoring='accuracy')
    gridSearch.fit(self.X, self.y)
    ```

1.  **geneticGridTest()** 方法执行基于遗传算法的网格搜索。它使用 **sklearn-deap** 库的 **EvolutionaryAlgorithmSearchCV()** 方法，该方法的调用方式与传统网格搜索非常相似。我们所需要做的只是添加一些遗传算法参数——种群大小、变异概率、比赛大小和代数：

    ```py
    gridSearch = EvolutionaryAlgorithmSearchCV(
        estimator=self.classifier,
        params=self.gridParams,
        cv=self.kfold,
        scoring='accuracy',
        verbose=True,
        population_size=20,
        gene_mutation_prob=0.50,
        tournament_size=2,
        generations_number=5)
    gridSearch.fit(self.X, self.y)
    ```

1.  最后，类的**main()**方法首先评估分类器使用默认超参数值时的性能。然后，它进行常规的全面网格搜索，接着进行基于基因算法的网格搜索，同时记录每次搜索的时间。

运行该类的主方法的结果将在下一小节中描述。

## 测试分类器的默认性能

运行结果表明，使用默认参数值`n_estimators = 50`、`learning_rate = 1.0`和`algorithm = 'SAMME.R'`时，分类器的准确率约为 66.4%：

```py
Default Classifier Hyperparameter values:
{'algorithm': 'SAMME.R', 'base_estimator': 'deprecated', 'estimator': None, 'learning_rate': 1.0, 'n_estimators': 50, 'random_state': 42}
score with default values =  0.6636507936507937
```

这不是一个特别好的准确率。希望通过网格搜索可以通过找到更好的超参数组合来改进这个结果。

## 运行常规的网格搜索

接下来执行常规的全面网格搜索，覆盖所有 200 种可能的组合。搜索结果表明，在这个网格内，最佳组合是`n_estimators = 50`、`learning_rate ≈ 0.5995`和`algorithm = 'SAMME.R'`。

使用这些值时，我们获得的分类准确率约为 92.7%，这是对原始 66.4%的大幅改进。搜索的运行时间大约是 131 秒，使用的是一台相对较旧的计算机：

```py
performing grid search...
best parameters:  {'algorithm': 'SAMME.R', 'learning_rate': 0.5994842503189409, 'n_estimators': 50}
best score:  0.9266666666666667
Time Elapsed =  131.01380705833435
```

接下来是基于基因算法的网格搜索。它能匹配这些结果吗？让我们来看看。

## 运行基于基因算法的网格搜索

运行的最后部分描述了基于基因算法的网格搜索，它与相同的网格参数一起执行。搜索的冗长输出从一个稍显晦涩的打印输出开始：

```py
performing Genetic grid search...
Types [1, 2, 1] and maxint [9, 9, 1] detected
```

该打印输出描述了我们正在搜索的网格——一个包含 10 个整数（`n_estimators`值）的列表，一个包含 10 个元素（`learning_rate`值）的 ndarray，以及一个包含两个字符串（`algorithm`值）的列表——如下所示：

+   **Types [1, 2, 1]**表示**[list, ndarray, list]**的网格类型

+   **maxint [9, 9, 1]**对应于**[10, 10, 2]**的列表/数组大小

下一行打印的是可能的网格组合的总数（10×10×2）：

```py
--- Evolve in 200 possible combinations ---
```

剩余的打印输出看起来非常熟悉，因为它使用了我们一直在使用的基于 DEAP 的基因算法工具，详细描述了进化代的过程，并为每一代打印统计信息：

```py
gen  nevals    avg        min       max        std
0     20    0.708146   0.117978   0.910112   0.265811
1     13    0.870787   0.662921   0.910112   0.0701235
2     10    0.857865   0.662921   0.91573    0.0735955
3     12    0.87809    0.679775   0.904494   0.0473746
4     12    0.878933   0.662921   0.910112   0.0524153
5     7     0.864045   0.162921   0.926966   0.161174
```

在过程结束时，打印出最佳组合、得分值和所用时间：

```py
Best individual is: {'n_estimators': 50, 'learning_rate': 0.5994842503189409, 'algorithm': 'SAMME.R'}
with fitness: 0.9269662921348315
Time Elapsed =  21.147947072982788
```

这些结果表明，基于基因算法的网格搜索能够在较短时间内找到与全面搜索相同的最佳结果。

请注意，这是一个运行非常快的简单示例。在实际情况中，我们通常会遇到大型数据集、复杂模型和庞大的超参数网格。在这些情况下，执行全面的网格搜索可能需要极长的时间，而基于基因算法的网格搜索在合理的时间内有可能获得不错的结果。

但是，所有网格搜索，无论是否由遗传算法驱动，都仅限于由网格定义的超参数值子集。如果我们希望在不受预定义值子集限制的情况下搜索网格外的内容呢？下节将描述一个可能的解决方案。

# 使用直接遗传方法调优超参数

除了提供高效的网格搜索选项外，遗传算法还可以直接搜索整个参数空间，正如我们在本书中用于搜索许多问题的输入空间一样。每个超参数可以表示为一个参与搜索的变量，染色体可以是所有这些变量的组合。

由于超参数可能有不同的类型——例如，我们的 AdaBoost 分类器中的 float、int 和枚举类型——我们可能希望对它们进行不同的编码，然后将遗传操作定义为适应每种类型的独立操作符的组合。然而，我们也可以使用一种懒惰的方法，将它们都作为浮动参数来简化算法的实现，正如我们接下来将看到的那样。

## 超参数表示

在*第六章*，《优化连续函数》中，我们使用遗传算法优化了实值参数的函数。这些参数被表示为一个浮动数字列表：*[1.23, 7.2134, -25.309]*。

因此，我们使用的遗传操作符是专门为处理浮动点数字列表而设计的。

为了调整这种方法，使其能够调优超参数，我们将把每个超参数表示为一个浮动点数，而不管其实际类型是什么。为了使其有效，我们需要找到一种方法将每个参数转换为浮动点数，并从浮动点数转换回其原始表示。我们将实现以下转换：

+   **n_estimators**，最初是一个整数，将表示为一个特定范围内的浮动值；例如，**[1, 100]**。为了将浮动值转换回整数，我们将使用 Python 的**round()**函数，它会将值四舍五入为最接近的整数。

+   **learning_rate** 已经是一个浮动点数，因此无需转换。它将绑定在**[0.01, 1.0]**范围内。

+   **algorithm** 可以有两个值，**'SAMME'** 或 **'SAMME.R'**，并将由一个位于**[0, 1]**范围内的浮动数表示。为了转换该浮动值，我们将其四舍五入为最接近的整数——**0** 或 **1**。然后，我们将**0**替换为**'SAMME'**，将**1**替换为**'SAMME.R'**。

这些转换将由两个 Python 文件执行，接下来的小节中将描述这两个文件。

## 评估分类器准确度

我们从一个 Python 类开始，该类封装了分类器的*准确度*评估，称为`HyperparameterTuningGenetic`。该类可以在`hyperparameter_tuning_genetic_test.py`文件中找到，该文件位于

[`github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_08/hyperparameter_tuning_genetic_test.py`](https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_08/hyperparameter_tuning_genetic_test.py)。

该类的主要功能如下所示：

1.  该类的**convertParam()**方法接受一个名为**params**的列表，包含表示超参数的浮动值，并将其转换为实际值（如前一小节所讨论）：

    ```py
    n_estimators = round(params[0])
    learning_rate = params[1]
    algorithm = ['SAMME', 'SAMME.R'][round(params[2])]
    ```

1.  **getAccuracy()**方法接受一个浮动数字的列表，表示超参数值，使用**convertParam()**方法将其转化为实际值，并用这些值初始化 AdaBoost 分类器：

    ```py
    n_estimators, learning_rate, algorithm = \
        self.convertParams(params)
    self.classifier =  AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm)
    ```

1.  然后，它通过我们为葡萄酒数据集创建的 k 折交叉验证代码来找到分类器的准确度：

    ```py
    cv_results = model_selection.cross_val_score(
        self.classifier,
        self.X,
        self.y,
        cv=self.kfold,
        scoring='accuracy')
    return cv_results.mean()
    ```

该类被实现超参数调优遗传算法的程序所使用，具体内容将在下一节中描述。

## 使用遗传算法调整超参数

基于遗传算法的最佳超参数值搜索由 Python 程序`02_hyperparameter_tuning_genetic.py`实现，该程序位于

[`github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_08/02_hyperparameter_tuning_genetic.py`](https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_08/02_hyperparameter_tuning_genetic.py)。

以下步骤描述了该程序的主要部分：

1.  我们首先为表示超参数的每个浮动值设置下界和上界，如前一小节所述——**[1, 100]**用于**n_estimators**，**[0.01, 1]**用于**learning_rate**，**[0, 1]**用于**algorithm**：

    ```py
    # [n_estimators, learning_rate, algorithm]:
    BOUNDS_LOW =  [  1, 0.01, 0]
    BOUNDS_HIGH = [100, 1.00, 1]
    ```

1.  然后，我们创建了一个**HyperparameterTuningGenetic**类的实例，这将允许我们测试不同的超参数组合：

    ```py
    test = HyperparameterTuningGenetic(RANDOM_SEED)
    ```

1.  由于我们的目标是最大化分类器的准确率，我们定义了一个单一目标——最大化适应度策略：

    ```py
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    ```

1.  现在进入一个特别有趣的部分——由于解的表示是一个浮动值列表，每个值的范围不同，我们使用以下循环遍历所有的下界和上界值对。对于每个超参数，我们创建一个单独的工具箱操作符，用来在适当的范围内生成随机浮动值：

    ```py
    for i in range(NUM_OF_PARAMS):
        # "hyperparameter_0", "hyperparameter_1", ...
        toolbox.register("hyperparameter_" + str(i),
                          random.uniform,
                          BOUNDS_LOW[i],
                          BOUNDS_HIGH[i])
    ```

1.  然后，我们创建了超参数元组，包含我们刚刚为每个超参数创建的具体浮动数字生成器：

    ```py
    hyperparameters = ()
    for i in range(NUM_OF_PARAMS):
        hyperparameters = hyperparameters + \
            (toolbox.__getattribute__("hyperparameter_" + str(i)),)
    ```

1.  现在，我们可以使用这个超参数元组，结合 DEAP 内置的 **initCycle()** 操作符，创建一个新的 **individualCreator** 操作符，该操作符通过随机生成的超参数值的组合填充一个个体实例：

    ```py
    toolbox.register("individualCreator",
                      tools.initCycle,
                      creator.Individual,
                      hyperparameters,
                      n=1)
    ```

1.  然后，我们指示遗传算法使用 **HyperparameterTuningGenetic** 实例的 **getAccuracy()** 方法进行适应度评估。作为提醒，**getAccuracy()** 方法（我们在前一小节中描述过）将给定的个体——一个包含三个浮点数的列表——转换回它们所表示的分类器超参数值，用这些值训练分类器，并通过 k 折交叉验证评估其准确性：

    ```py
    def classificationAccuracy(individual):
        return test.getAccuracy(individual),
    toolbox.register("evaluate", classificationAccuracy)
    ```

1.  现在，我们需要定义遗传操作符。对于 **selection** 操作符，我们使用常见的锦标赛选择，锦标赛大小为 **2**，我们选择专门为有界浮点列表染色体设计的 **crossover** 和 **mutation** 操作符，并为它们提供我们为每个超参数定义的边界：

    ```py
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate",
                     tools.cxSimulatedBinaryBounded,
                     low=BOUNDS_LOW,
                     up=BOUNDS_HIGH,
                     eta=CROWDING_FACTOR)
    toolbox.register("mutate",
                     tools.mutPolynomialBounded,
                     low=BOUNDS_LOW,
                     up=BOUNDS_HIGH,
                     eta=CROWDING_FACTOR,
                     indpb=1.0 / NUM_OF_PARAMS)
    ```

1.  此外，我们继续使用精英策略，即 HOF 成员——当前最佳个体——始终不受影响地传递到下一代：

    ```py
    population, logbook = elitism.eaSimpleWithElitism(
        population,
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=True)
    ```

通过用一个种群大小为 30 的算法运行五代，我们得到了以下结果：

```py
gen nevals max avg
0       30      0.927143        0.831439
1       22      0.93254         0.902741
2       23      0.93254         0.907847
3       25      0.943651        0.916566
4       24      0.943651        0.921106
5       24      0.943651        0.921751
- Best solution is:
params =  'n_estimators'= 30, 'learning_rate'=0.613, 'algorithm'=SAMME.R
Accuracy = n_estimators = 30, learning_rate = 0.613, and algorithm = 'SAMME.R'.
The classification accuracy that we achieved with these values is about 94.4%—a worthy improvement over the accuracy we achieved with the grid search. Interestingly, the best value that was found for `learning_rate` is just outside the grid values we searched on.
Dedicated libraries
In recent years, several genetic-algorithm-based libraries have been developed that are dedicated to optimizing machine learning model development. One of them is `sklearn-genetic-opt` ([`sklearn-genetic-opt.readthedocs.io/en/stable/index.html`](https://sklearn-genetic-opt.readthedocs.io/en/stable/index.html)); it supports both hyperparameters tuning and feature selection. Another more elaborate library is `TPOT`([`epistasislab.github.io/tpot/`](https://epistasislab.github.io/tpot/)); this library provides optimization for the end-to-end machine learning development process, also called the **pipeline**. You are encouraged to try out these libraries in your own projects.
Summary
In this chapter, you were introduced to the concept of hyperparameter tuning in machine learning. After getting acquainted with the Wine dataset and the AdaBoost classifier, both of which we used for testing throughout this chapter, you were presented with the hyperparameter tuning methods of an exhaustive grid search and its genetic-algorithm-driven counterpart. These two methods were then compared using our test scenario. Finally, we tried out a direct genetic algorithm approach, where all the hyperparameters were represented as float values. This approach allowed us to improve the results of the grid search.
In the next chapter, we will look into the fascinating machine learning models of **neural networks** and **deep learning** and apply genetic algorithms to improve their performance.
Further reading
For more information on the topics that were covered in this chapter, please refer to the following resources:

*   Cross-validation and Parameter Tuning, from the book *Mastering Predictive Analytics with scikit-learn and TensorFlow*, Alan Fontaine, September 2018:
*   [`subscription.packtpub.com/book/big_data_and_business_intelligence/9781789617740/2/ch02lvl1sec16/introduction-to-hyperparameter-tuning`](https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789617740/2/ch02lvl1sec16/introduction-to-hyperparameter-tuning)
*   *sklearn-deap* at GitHub: [`github.com/rsteca/sklearn-deap`](https://github.com/rsteca/sklearn-deap)
*   *Scikit-learn* AdaBoost Classifier: [`scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
*   *UCI Machine Learning* *Repository*: [`archive.ics.uci.edu/`](https://archive.ics.uci.edu/)

```
