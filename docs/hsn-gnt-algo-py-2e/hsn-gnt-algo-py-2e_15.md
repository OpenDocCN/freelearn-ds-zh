

# 第十二章：可解释人工智能、因果关系与基因算法中的反事实

本章探讨了基因算法在生成“假设”场景中的应用，提供了对数据集和相关机器学习模型分析的宝贵见解，并为实现预期结果提供了可操作的洞察。

本章开始时介绍了 **可解释人工智能**（**XAI**）和 **因果关系** 领域，随后解释了 **反事实** 的概念。我们将使用这一技术探索无处不在的 *德国信用风险* 数据集，并利用基因算法对其进行反事实分析，从中发现有价值的洞察。

到本章结束时，你将能够做到以下几点：

+   熟悉 XAI 和因果关系领域及其应用

+   理解反事实的概念及其重要性

+   熟悉德国信用风险数据集及其缺点

+   实现一个应用程序，创建反事实的“假设”场景，为这个数据集提供可操作的洞察，并揭示其相关机器学习模型的操作。

本章将从对 XAI 和因果关系的简要概述开始。如果你是经验丰富的数据科学家，可以跳过这一介绍部分。

# 技术要求

在本章中，我们将使用 Python 3 和以下支持库：

+   **deap**

+   **numpy**

+   **pandas**

+   **scikit-learn**

重要提示

如果你使用的是我们提供的 **requirements.txt** 文件（见 *第三章*），这些库已经包含在你的环境中。

本章中将使用的程序可以在本书的 GitHub 仓库中找到，网址是 [`github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/tree/main/chapter_12`](https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/tree/main/chapter_12)。

查看以下视频，观看代码演示：

[`packt.link/OEBOd`](https://packt.link/OEBOd)。

# 解锁黑盒 – XAI

**XAI** 是 **人工智能**（**AI**）领域的一个关键元素，旨在揭示机器学习模型复杂的工作原理。随着人工智能应用的不断增长，理解模型的决策过程变得至关重要，以建立信任并确保负责任的部署。

XAI 旨在解决此类模型固有的复杂性和不透明性问题，并提供清晰且易于理解的预测解释。这种透明性不仅增强了 AI 模型的可解释性，还使用户、利益相关者和监管机构能够审视和理解这些过程。在医疗保健和金融等关键领域，决策具有现实世界的后果，XAI 显得尤为重要。例如，在医学诊断中，可解释的模型不仅提供准确的预测，还揭示了影响诊断的医学图像或患者记录中的具体特征，从而建立信任并符合伦理标准。

实现可解释性的有效方法之一是通过*模型无关*技术。这些方法为任何机器学习模型提供**事后**（“事后解释”）的解释，不论其架构如何。像*SHAP 值*和*LIME*等技术，通过对输入数据或模型参数进行小幅、受控的调整，生成解释，从而揭示对预测贡献最大的特征。

在 XAI 的基础上，**因果关系**通过探索“是什么背后的‘为什么’”为模型提供了更深层次的解释，正如下节所述。

# 揭示因果关系——AI 中的因果性

不仅仅是了解 AI 的预测结果，还要理解这些预测背后的因果关系，这在决策具有重大影响的领域尤为重要。

在 AI 中，因果性探索数据的各个方面的变化是否会影响模型的预测或决策。例如，在医疗保健领域，了解患者参数与预测结果之间的因果关系有助于更有效地量身定制治疗方案。目标不仅是准确的预测，还要理解这些预测背后的机制，以便为数据提供更加细致且可操作的洞察。

## 假设情景——反事实

**反事实**通过探索“假如”情景并考虑替代结果，进一步增强了 AI 系统的可解释性。反事实解释帮助我们了解如何通过调整输入来影响模型预测，通过微调这些输入并观察模型决策中的变化（或没有变化）。这一过程本质上提出了“*如果呢？*”的问题，并使我们能够获得有关 AI 模型敏感性和鲁棒性的有价值的洞察。

例如，假设一个 AI 驱动的汽车决定避开行人。通过反事实分析，我们可以揭示在不同条件下这一决策如何发生变化，从而为模型的行为提供有价值的洞察。另一个例子是推荐系统。反事实分析可以帮助我们理解如何调整某些用户偏好，可能会改变推荐的项目，从而为用户提供更清晰的系统工作原理，并使开发者能够提高用户满意度。

除了提供对模型行为的更深理解外，反事实分析还可以用于模型改进和调试。通过探索替代场景并观察变化如何传播，开发人员可以发现潜在的弱点、偏差或优化空间。

正如我们在以下章节中所示，探索“假设”场景也可以使用户预测和解读 AI 系统的响应。

## 遗传算法在反事实分析中的应用——导航替代场景

遗传算法作为执行反事实分析的有用工具，提供了一种灵活的方式来修改模型输入，从而达到预期的结果。在这里，遗传算法中的每个解代表一个独特的输入组合。优化目标取决于模型的输出，并可以结合与输入值相关的条件，例如限制变化或最大化某个特定输入值。

在接下来的章节中，我们将利用遗传算法对一个机器学习模型进行反事实分析，该模型的任务是确定贷款申请人的信用风险。通过这种探索，我们旨在回答有关特定申请人的各种问题，深入了解模型的内在运作。此外，这一分析还可以提供可操作的信息，帮助申请人提高获得贷款的机会。

让我们首先熟悉将用于训练模型的数据集。

## 德国信用风险数据集

在本章的实验中，我们将使用经过修改的*德国信用风险*数据集，该数据集在机器学习和统计学领域的研究和基准测试中被广泛使用。原始数据集可以从*UCI 机器学习库*访问，包含 1,000 个实例，每个实例有 20 个属性。该数据集旨在进行二分类任务，目的是预测贷款申请人是否值得信贷或存在信用风险。

按照现代标准，数据集中某些原始属性被认为是*受保护*的，特别是代表候选人性别和年龄的属性。在我们修改后的版本中，这些属性已被排除。此外，其他一些特征要么被删除，要么其值已被转换为数值格式，以便简化处理。

修改后的数据集可以在`chapter_12/data/credit_risk_data.csv`文件中找到，包含以下列：

1.  **checking**: 申请人支票账户的状态：

    +   **0**: 没有支票账户

    +   **1**: 余额 < 100 德国马克

    +   **2**: 100 <= 余额 < 200 德国马克

    +   **3**: 余额 >= 200 德国马克

1.  **duration**: 申请贷款的时长（月数）

1.  **credit_history**: 申请人的信用历史信息：

    +   **0**: 没有贷款/所有贷款已按时偿还

    +   **1**: 现有贷款已按时偿还

    +   **2**: 本银行的所有贷款均已按时还清

    +   **3**: 过去曾有还款延迟

    +   **4**: 存在重要账户/其他贷款

1.  **金额**: 申请贷款的金额

1.  **储蓄**: 申请人的储蓄账户状态：

    +   **0**: 未知/没有储蓄账户

    +   **1**: 余额 < 100 德国马克

    +   **2**: 100 <= 余额 < 500 德国马克

    +   **3**: 500 <= 余额 < 1000 德国马克

    +   **4**: 余额 >= 1000 德国马克

1.  **就业时长**:

    +   **0**: 失业

    +   **1**: 时长 < 1 年

    +   **2**: 1 <= 时长 < 4 年

    +   **3**: 4 <= 时长 < 7 年

    +   **4**: 时长 >= 7 年

1.  **其他债务人**: 除主要申请人外，任何可能是共同债务人或共同承担贷款财务责任的个人：

    +   **无**

    +   **担保人**

    +   **共同申请人**

1.  **现住址**: 申请人在当前地址的居住时长，用 1 到 4 之间的整数表示

1.  **住房**: 申请人的住房情况：

    +   **免费**

    +   **自有**

    +   **租赁**

1.  **信用账户数**: 在同一银行持有的信用账户数量

1.  **负担人**: 申请人经济上依赖的人的数量

1.  **电话**: 申请人是否有电话（**1** = 是，**0** = 否）

1.  **信用风险**: 要预测的值：

    +   **1**: 高风险（表示违约或信用问题的可能性较高）

    +   **0**: 低风险

为了说明，我们提供了数据的前 10 行：

```py
1,6,4,1169,0,4,none,4,own,2,1,1,1
2,48,1,5951,1,2,none,2,own,1,1,0,0
0,12,4,2096,1,3,none,3,own,1,2,0,1
1,42,1,7882,1,3,guarantor,4,for free,1,2,0,1
1,24,3,4870,1,2,none,4,for free,2,2,0,0
0,36,1,9055,0,2,none,4,for free,1,2,1,1
0,24,1,2835,3,4,none,4,own,1,1,0,1
2,36,1,6948,1,2,none,2,rent,1,1,1,1
0,12,1,3059,4,3,none,4,own,1,1,0,1
2,30,4,5234,1,0,none,2,own,2,1,0,0
```

虽然在以往的数据集工作中，我们的主要目标是开发一个机器学习模型，以对新数据做出精准的预测，但现在我们将使用反事实情境来扭转局面，并识别出与期望预测匹配的数据。

# 探索用于信用风险预测的反事实情境

从数据中可以看出，许多申请人被认为是信用风险（最后的值为`1`），导致贷款被拒。对于这些申请人，提出以下问题：他们能采取什么措施改变这一决定，并被视为有信用？（结果为`0`）。这里所说的*措施*是指更改他们的某些属性状态，例如他们申请的借款金额。

在检查数据集时，一些属性对于申请人来说是困难或甚至不可能改变的，比如就业时长、抚养人数或当前住房。对于我们的例子，我们将重点关注以下四个属性，它们都有一定的灵活性：

+   **金额**: 申请贷款的金额

+   **时长**: 申请贷款的时长（以月为单位）

+   **支票账户**: 申请人的支票账户状态

+   **储蓄**: 申请人的储蓄账户状态

现在问题可以这样表述：对于一个目前被标记为信用风险的候选人，我们可以对这四个属性（或其中一些）进行哪些最小的变化，使结果变为信用良好？

为了回答这个问题，以及其他相关问题，我们将创建以下内容：

+   一个已经在我们的数据集上训练的机器学习模型。该模型将被用来提供预测，从而在修改申请人数据时测试潜在的结果。

+   基于遗传算法的解决方案，寻找新的属性值以回答我们的问题。

这些组件将使用 Python 实现，如下文所述。

## Applicant 类

`Applicant`类表示数据集中的一个申请人；换句话说，就是 CSV 文件中的一行数据。该类还允许我们修改`amount`、`duration`、`checking`和`savings`字段的值，这些字段代表了申请人的相应属性。此类可以在`applicant.py`文件中找到，文件位于[`github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_12/applicant.py`](https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_12/applicant.py)。

该类的主要功能如下所示：

+   该类的**__init__()**方法使用**dataset_row**参数的值从对应的数据集行中复制值，并创建一个代表申请人的实例。

+   除了先前提到的四个属性的设置器和获取器外，**get_values()**方法返回四个属性的当前值，而**with_values()**方法则创建原始申请人实例的副本，并随之修改这四个属性的复制值。这两个方法都使用整数列表来表示四个属性的值，因为它们将被遗传算法直接使用，遗传算法将潜在的申请人表示为四个整数的列表。

## CreditRiskData 类

`CreditRiskData`类封装了信用风险数据集及其上训练的机器学习模型。该类位于`credit_risk_data.py`文件中，可以在[`github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_12/credit_risk_data.py`](https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_12/credit_risk_data.py)找到。

该类的主要功能在以下步骤中体现：

1.  该类的**__init__()**方法初始化随机种子；然后调用**read_dataset()**方法，该方法从 CSV 文件中读取数据集：

    ```py
    self.randomSeed = randomSeed
    self.dataset = self.read_dataset()
    ```

1.  接下来，它检查是否已创建并保存了训练好的模型文件。如果模型文件存在，则加载它。否则，调用**train_model()**方法。

1.  **train_model()**方法创建了一个*随机森林分类器*，该分类器首先通过 5 折交叉验证程序进行评估，以验证其泛化能力：

    ```py
    classifier = RandomForestClassifier(
        random_state=self.randomSeed)
    kfold = model_selection.KFold(n_splits=NUM_FOLDS)
    cv_results = model_selection.cross_val_score(
        classifier, X, y, cv=kfold, scoring='accuracy')
    print(f"Model's Mean k-fold accuracy = {cv_results.mean()}")
    ```

1.  接下来，使用整个数据集训练模型并对其进行评估：

    ```py
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    print(f"Model's Training Accuracy = {accuracy_score(y, 
        y_pred)}")
    ```

1.  一旦训练完成，*随机森林* 模型可以为数据集的各个属性分配 *特征重要性* 值，表示每个属性对模型预测的贡献。虽然这些值提供了对影响模型决策的因素的见解，但我们将在这里使用它们来验证我们假设的四个属性是否能产生不同的结果：

    ```py
    feature_importances = dict(zip(X.columns, 
        classifier.feature_importances_))
    print(dict(sorted(feature_importances.items(), 
        key=lambda item: -item[1])))
    ```

1.  **is_credit_risk()** 方法利用训练过的模型，通过 *Scikit-learn* 的 **predict()** 方法预测给定申请者数据的结果，当候选者被认为是信用风险时返回 **True**。

1.  此外，**risk_probability()** 方法提供一个介于 0 和 1 之间的浮动值，表示申请者被认为是信用风险的程度。它利用模型的 **predict_proba()** 方法，在应用阈值将其转换为离散值 0 或 1 之前，捕获连续的输出值。

1.  方便的方法 **get_applicant()** 允许我们从数据集中选择一个申请者的行并打印其数据。

1.  最后，**main()** 函数首先通过创建 **CreditRiskData** 类的实例来启动，如果需要，它会第一次训练模型。接着，它从数据集中获取第 25 个申请者的信息并打印出来。之后，它修改四个可变属性的值，并打印修改后的申请者信息。

1.  当第一次执行 **main()** 函数时，交叉验证测试评估的结果，以及训练精度，将会被打印出来：

    ```py
    Loading the dataset...
    Model's Mean k-fold accuracy = 0.7620000000000001
    Model's Training Accuracy = 1.0
    ```

    这些结果表明，虽然训练过的模型能够完全再现数据集的结果，但在对未见过的样本进行预测时，模型的准确率约为 76%——对于这个数据集来说，这是一个合理的结果。

1.  接下来，特征重要性值按降序打印。值得注意的是，列表中的前几个属性就是我们选择修改的四个：

    ```py
    ------- Feature Importance values:
    {
        "amount": 0.2357488244229738,
        "duration": 0.15326057481242433,
        "checking": 0.1323559111404014,
        "employment_duration": 0.08332785367394725,
        "credit_history": 0.07824885834794511,
        "savings": 0.06956484835261427,
        "present_residence": 0.06271797270697153,
         …
    }
    ```

1.  数据集中第 25 行申请者的属性信息和预测结果现在被打印出来。值得注意的是，在文件中，这对应于第 27 行，因为第一行包含标题，数据行从 0 开始计数：

    ```py
    applicant = credit_data.get_applicant(25)
    ```

    输出结果如下：

    ```py
    Before modifications: -------------
    Applicant 25:
    checking                        1
    duration                        6
    credit_history                  1
    amount                       1374
    savings                         1
    employment_duration             2
    present_residence               2
    …
    => Credit risk = True
    ```

    如输出所示，该申请者被认为是信用风险。

1.  程序现在通过 **with_values()** 方法修改所有四个值：

    ```py
    modified_applicant = applicant.with_values([1000, 20, 2, 0])
    ```

    然后，它重复打印，反映变化：

    ```py
    After modifications: -------------
    Applicant 25:
    checking                        2
    duration                       20
    credit_history                  1
    amount                       1000
    savings                         0
    employment_duration             2
    present_residence               2
    …
    => Credit risk = False
    ```

正如前面的输出所示，当使用新值时，申请者不再被认为是信用风险。虽然这些修改的值是通过反复试验手动选择的，但现在是时候使用遗传算法来自动化这个过程了。

## 使用遗传算法进行反事实分析

为了演示遗传算法如何与反事实一起工作，我们将从第 25 行的相同申请者开始，该申请者最初被认为是信用风险，然后寻找使其预测信用可接受的最小变化集。如前所述，我们将考虑对`amount`、`duration`、`checking`和`savings`属性进行更改。

### 解的表示

在处理这个问题时，表示候选解的一种简单方法是使用一个包含四个整数值的列表，每个值对应我们希望修改的四个属性之一：

[amount, duration, checking, savings]

例如，我们在`credit_risk_data.py`程序的主函数中使用的修改值将表示如下：

```py
[1000, 20, 2, 0]
```

正如我们在前几章中所做的那样，我们将利用浮点数来表示整数。这样，遗传算法可以使用行之有效的实数操作符进行交叉和变异，并且无论每个项的范围如何，都使用相同的表示。在评估之前，实数将通过`int()`函数转换为整数。

我们将在下一小节中评估每个解。

### 评估解

由于我们的目标是找到使预测结果反转的*最小*变化程度，因此一个问题出现了：我们如何衡量所做变化的程度？一种可能的方法是使用当前解的值与原始值之间的绝对差值之和，每个差值除以该值的范围，如下所示：

∑ i=1 4  |current valu e i − original valu e i|  _______________________  range of valu e i

现在我们已经建立了候选解的表示和评估方法，我们准备展示遗传算法的 Python 实现。

## 遗传算法解

基于遗传算法的反事实搜索实现于名为`01_counterfactual_search.py`的 Python 程序中，该程序位于[`github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_12/01_counterfactual_search.py`](https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_12/01_counterfactual_search.py)。

以下步骤描述了该程序的主要部分：

1.  我们首先定义几个常量。然后，我们创建**CreditRiskData**类的一个实例：

    ```py
    credit_data = CreditRiskData(randomSeed=RANDOM_SEED)
    ```

1.  接下来的几个代码片段用于设置将用作解变量的四个属性的范围。我们首先声明占位符，如下所示：

    ```py
    bounds_low = []
    bounds_high = []
    ranges = []
    ```

    第一个列表包含四个属性的下限，第二个包含上限，第三个包含它们之间的差值。

1.  接下来是**set_ranges()**方法，该方法接受四个属性的上下限，并相应地填充占位符。由于我们使用的是将转换为整数的实数，我们将增加每个范围的值，以确保结果整数的均匀分布：

    ```py
    bounds_low = [amount_low, duration_low, checking_low, 
        savings_low]
    bounds_high = [amount_high, duration_high, checking_high, 
        savings_high]
    bounds_high = [high + 1 for high in bounds_high]
    ranges = [high - low for high, low in zip(bounds_high, 
        bounds_low)]
    ```

1.  然后，我们将使用**set_ranges()**方法为当前问题设置范围。我们选择了以下值：

    +   **amount**：100..5000

    +   **duration**：2..72

    +   **checking**：0..3

    +   **savings**：0..4：

        ```py
        bounds_low, bounds_high, ranges =
        set_ranges(100, 5000, 2, 72, 0, 3, 0, 4)
        ```

1.  现在，我们必须从数据集的第 25 行选择申请人（与之前使用的一样），并将其原始的四个值保存在单独的变量**applicant_values**中：

    ```py
    applicant = credit_data.get_applicant(25)
    applicant_values = applicant.get_values()
    ```

1.  **get_score()**函数用于通过计算需要最小化的代价来评估每个解决方案的适应度。代价由两部分组成：首先，如*评估解决方案*部分所述，我们计算该解决方案表示的四个属性值与候选人匹配的原始值之间的距离——距离越大，代价越大：

    ```py
    cost = sum(
        [
            abs(int(individual[i]) - applicant_values[i])/ranges[i]
            for i in range(NUM_OF_PARAMS)
        ]
    )
    ```

1.  由于我们希望解决方案能够代表一个有信用的候选人，因此代价的第二部分（可选）用于惩罚被视为信用风险的解决方案。在这里，我们将使用**is_credit_risk()**和**risk_probability()**方法，这样当前者表明解决方案没有信用时，后者将用于确定增加的惩罚程度：

    ```py
    if credit_data.is_credit_risk(
        applicant.with_values(individual)
    ):
        cost += PENALTY * credit_data.risk_probability(
            applicant.with_values(individual))
    ```

1.  该程序的其余部分与我们之前看到的非常相似，当时我们使用实数列表来表示个体——例如，*第九章*，*深度学习网络架构优化*。我们将开始使用单目标策略来最小化适应度，因为我们的目标是最小化由先前定义的代价函数计算出的值：

    ```py
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    ```

1.  由于解决方案由四个浮动值的列表表示，每个值对应我们可以修改的一个属性，并且每个属性有其自己的范围，我们必须为它们定义独立的工具箱*creator*操作符，使用相应的**bounds_low**和**bounds_high**值：

    ```py
    toolbox.register("amount", random.uniform, \
        bounds_low[0], bounds_high[0])
    toolbox.register("duration", random.uniform, \
        bounds_low[1], bounds_high[1])
    toolbox.register("checking", random.uniform, \
        bounds_low[2], bounds_high[2])
    toolbox.register("savings", random.uniform, \
        bounds_low[3], bounds_high[3])
    ```

1.  这四个操作符接着在**individualCreator**的定义中使用：

    ```py
    toolbox.register("individualCreator",
        tools.initCycle,
        creator.Individual,
        (toolbox.amount, toolbox.duration,
            toolbox.checking, toolbox.savings),
        n=1)
    ```

1.  在将*selection*操作符分配给通常的*tournament selection*（锦标赛选择），并设置锦标赛大小为**2**后，我们将为其分配*crossovers*和*mutation*操作符，这些操作符专门用于有界浮点数列表染色体，并提供我们之前定义的范围：

    ```py
    toolbox.register("select",
                      tools.selTournament,
                      tournsize=2)
    toolbox.register("mate",
                     tools.cxSimulatedBinaryBounded,
                     low=bounds_low,
                     up= bounds_high,
                     eta=CROWDING_FACTOR)
    toolbox.register("mutate",
                     tools.mutPolynomialBounded,
                     low= bounds_low,
                     up=bounds_high,
                     eta=CROWDING_FACTOR,
                     indpb=1.0 / NUM_OF_PARAMS)
    ```

1.  此外，我们将继续使用*elitist*方法，其中**hall-of-fame**（**HOF**）成员——当前最佳个体——始终不加修改地传递到下一代：

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

最后，我们打印出找到的最佳解决方案以及该解决方案的预测值。

现在是时候试用该程序并查看结果了！输出从打印所选申请人的原始属性和状态开始：

```py
Loading the dataset...
Applicant 25:
checking                        1
duration                        6
credit_history                  1
amount                       1374
savings                         1
employment_duration             2
present_residence               2
...
=> Credit risk = amount, duration, checking, and savings:

```

gen     nevals  min             avg

0       50      0.450063        51.7213

1       42      0.450063        30.2695

2       44      0.393725        14.2223

3       37      0.38311         7.62647

...

28      40      0.141661        0.169646

29      40      0.141661        0.175401

30      44      0.141661        0.172197

-- 最佳方案：金额 = 1370，期限 = 16，检查 = 1，储蓄 = 1

-- 预测：是 _ 风险 = 检查和储蓄账户无需更改。

尽管金额被调整为`1,374`，它本来可以保持不变为 1374。通过直接调用`is_credit_risk()`函数，并使用值[1374, 16, 1, 1]，可以验证这一点。根据我们的成本函数定义，1370 与 1374 之间的差异除以 4900 的范围是微小的，可能导致算法需要更多的世代才能识别出 1374 比 1370 更好。通过将金额的范围缩小到 1000..2000，同样的程序能够在规定的 30 代内很好地识别出 1374 的值。

更多的“假设”场景

我们发现，将申请人 25 的贷款期限从 6 个月调整为 16 个月，可以使得申请获得信用。但是，如果申请人希望更短的期限，或想最大化贷款金额呢？这些正是反事实所探索的“假设”场景，我们编写的代码可以模拟不同场景并解决这些问题，如下文小节所示。

减少期限

让我们从同一申请人希望将期限设置为比之前找到的 16 个月更短的情况开始——其他变化能否弥补这一点？

根据前次运行的经验，缩小四个属性的允许范围可能是有益的。让我们尝试采取更保守的方式，并使用以下范围：

+   **金额**: 1000..2000

+   **期限**: 2..12

+   **检查**: 0..1

+   **储蓄**: 0..1

在这里，我们将期限限制为 12 个月，并且力求避免增加当前`检查`或`储蓄`账户的余额。这可以通过修改`set_ranges()`的调用来实现，如下所示：

```py
bounds_low, bounds_high, ranges = set_ranges(1000, 2000, 2, 12, 0, 1, 0, 1)
```

当我们运行修改后的程序时，结果如下：

```py
-- Best solution: Amount = 1249, Duration = 12, checking = 1, savings = 1
-- Prediction: is_risk = False
```

这表明，如果申请人愿意稍微降低所请求的贷款金额，则可以实现 12 个月的缩短期限。

如果我们想进一步减少期限呢？比如将期限范围更改为 1..10。这将得到以下结果：

```py
-- Best solution: Amount = 1003, Duration = 10, checking = 1, savings = 0
-- Prediction: is_risk = True
```

这表明，算法未能在这些范围内找到一个申请人是可信贷的解决方案。请注意，这并不一定意味着不存在这样的解决方案，但似乎不太可能。

即使我们回去并允许`checking`和`savings`账户的原始范围（0..3 和 0..4），如果期限限制在 10 个月或更少，仍然找不到解决方案。然而，允许金额低于 1,000 似乎能解决问题。让我们使用以下范围：

+   **金额**: 100..2000

+   **期限**: 2..10

+   **checking**: 0..1

+   **储蓄**: 0..1

在这里，我们得到了以下解决方案：

```py
-- Best solution: Amount = 971, Duration = 10, checking = 1, savings = 0
-- Prediction: is_risk = False
```

这意味着，如果申请者将贷款金额减少到 971，则申请将按所需的 10 个月期限获得批准。

更令人惊讶的是，`savings`属性的值为 0，低于原始的 1。如你所记得，这个属性的值解释如下：

+   **0**: 未知/没有储蓄账户

+   **1**: 余额 < 100 DM

+   **2**: 100 <= 余额 < 500 DM

+   **3**: 500 <= 余额 < 1000 DM

+   **4**: 余额 >= 1000 DM

看起来，在申请贷款时没有储蓄是不利的。而且，如果我们尝试所有除 0 以外的可能值，将范围设置为 1..3，则没有找到解决方案。这表明，根据使用的模型，没有储蓄账户比拥有储蓄账户更为优越，即使储蓄余额较高。这可能是模型存在缺陷的表现，或者是数据集本身存在问题，例如数据偏见或不完整。这样的发现是反事实推理的一种使用场景。

最大化贷款金额

到目前为止，我们已经操作了一个最初被认为是信用风险的申请者的结果。然而，我们可以对*任何*申请者进行这种“假设”游戏，包括那些已经被批准的申请者。让我们考虑第 68 行的申请者（文件中的第 70 行）。当打印出申请者信息时，我们看到以下内容：

```py
Applicant 68:
checking                        0
duration                       36
credit_history                  1
amount                       1819
savings                         1
employment_duration             2
present_residence               4
...
=> Credit risk = 02_counterfactual_search.py, which is located at https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python-Second-Edition/blob/main/chapter_12/02_counterfactual_search.py.
This program is identical to the previous one, except for three small changes. The first change is the use of this particular applicant:

```

applicant = credit_data.get_applicant(68)

```py

 The second change is to the range values:

```

bounds_low, bounds_high, ranges = set_ranges(2000, 50000, 36, 36, 0,

0, 1, 1)

```py

 The amount range is modified to allow up to a sum of 50,000, while the other ranges have been fixed to the existing values of the candidate. This will enable the genetic algorithm to only modify the amount.
But how do we instruct the genetic algorithm to *maximize* the loan amount? As you may recall, the cost function was initially designed to minimize the distance between the modified individual and the original one within the given range. However, in this scenario, we want the loan amount to be as large as possible compared to the original amount. One approach to address this is to replace the cost function with a new one. However, we’ll explore a somewhat simpler solution: we’ll set the original loan amount value to the same value we use for the upper end of the range, which is 50,000 in this case. By doing this, when the algorithm aims to find the closest possible solution, it will work inherently to maximize the amount toward this upper limit. This can be done by adding a single line of code that overrides the original amount value of the applicant. The line is placed immediately following the one that stores the original attribute values to be used by the cost function:

```

applicant_values = applicant.get_values()

applicant_values[0] 被使用，因为金额属性是程序所用的四个值中的第一个。

运行这个程序会得到以下输出：

```py
-- Best solution: Amount = 14165, Duration = 36, checking = 0, savings = 1
-- Prediction: is_risk = False
```

上述输出表明，这位申请者在增加贷款金额的同时，保持了原有的信用良好状态，而无需对其他属性做出任何更改。

接下来的问题是，是否可以通过允许更改`checking`和/或`savings`属性，*进一步*增加贷款金额。为此，我们将修改边界，使这两个属性可以调整为任何有效值：

```py
bounds_low, bounds_high, ranges = set_ranges(2000, 50000, 36, 36, 0, 
    3, 0, 4)
```

修改后的程序结果有些令人惊讶：

```py
-- Best solution: Amount = 50000, Duration = 36, checking = 1, savings = 1
-- Prediction: is_risk = False
```

这个结果表明，将支票账户的状态从 0（没有支票账户）改为 1（余额<100 DM）就足以获得显著更高的贷款金额。如果我们用更高的金额（例如 500,000，替换程序中两个不同位置的值）重复这个实验，结果将类似——只要将支票状态从 0 改为 1，候选人就能获得这笔高额贷款。

这个观察结果同样适用于其他申请人，表明模型可能存在潜在的漏洞。

鼓励你对程序进行额外修改，探索自己的“如果…会怎样”场景。除了提供有价值的洞察和更深入地理解模型行为外，实验“如果…会怎样”场景还可以非常有趣！

扩展到其他数据集

本章演示的相同过程可以应用于其他数据集。例如，考虑一个用于预测租赁公寓预期价格的数据集。在这个场景中，你可以使用类似的反事实分析来确定房东可以对公寓进行哪些修改，以实现某个租金。使用类似本章介绍的程序，你可以应用遗传算法来探索输入特征变化对模型预测的敏感性，并确定可行的洞察，以实现期望的结果。

总结

在本章中，我们介绍了**XAI**、**因果关系**和**反事实**的概念。在熟悉了*German Credis Risk* 数据集后，我们创建了一个机器学习模型，用于预测申请人是否具备信用资格。接下来，我们应用了基于遗传算法的反事实分析，将该数据集应用于训练好的模型，探索了几个“如果…会怎样”场景，并获得了宝贵的洞察。

在接下来的两章中，我们将转向加速基于遗传算法的程序的执行，例如我们在本书中开发的程序，通过探索应用并发的不同策略。

进一步阅读

如需了解本章中涉及的更多内容，请参考以下资源：

+   *Python 中的可解释 AI（XAI）实践*，作者：Denis Rothman，2020 年 7 月

+   *Python 中的因果推理与发现*，作者：Aleksander Molak，2023 年 5 月

+   *企业中的负责任 AI*，作者：Adnan Masood，2023 年 7 月

+   *Scikit-learn* *RandomForestClassifier*：[`scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

```py

```

```py

```
