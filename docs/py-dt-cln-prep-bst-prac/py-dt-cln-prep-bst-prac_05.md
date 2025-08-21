

# 第五章：数据转换 – 合并和拼接

理解如何转换和处理数据对于挖掘有价值的洞察至关重要。技术如连接、合并和附加使我们能够将来自不同来源的信息融合在一起，并组织和分析数据的子集。在本章中，我们将学习如何将多个数据集合并成一个单一的数据集，并探索可以使用的各种技术。我们将理解如何在合并数据集时避免重复值，并学习一些提升数据合并过程的技巧。

本章将涵盖以下主题：

+   连接数据集

+   处理数据合并中的重复项

+   合并时的性能优化技巧

+   拼接 DataFrame

# 技术要求

你可以在以下链接找到本章的所有代码：[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/tree/main/chapter05`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/tree/main/chapter05)。

每个章节后面都有一个具有类似命名约定的脚本，欢迎你执行脚本或通过阅读本章来跟随学习。

# 连接数据集

在数据分析项目中，常常会遇到数据分散在多个来源或数据集中的情况。每个数据集可能包含与某个共同实体或主题相关的不同信息片段。**数据合并**，也称为数据连接或数据拼接，是将这些独立的数据集合并成一个统一的数据集的过程。在数据分析项目中，常常会遇到某个特定主题或实体的信息分布在多个数据集中的情况。例如，假设你正在为一个零售企业分析客户数据。你可能有一个数据集包含客户的人口统计信息，如姓名、年龄和地址，另一个数据集包含他们的购买历史，如交易日期、购买的商品和总支出。每个数据集都提供了有价值的见解，但单独来看，它们无法完整展现客户的行为。为了获得全面的理解，你需要将这些数据集合并。通过根据一个共同的标识符（如客户 ID）将客户的人口统计信息与购买历史合并，你可以创建一个单一的数据集，从而进行更丰富的分析。例如，你可以识别出哪些年龄组购买了特定的产品，或支出习惯如何因地域而异。

## 选择正确的合并策略

选择正确的连接类型至关重要，因为它决定了输入 DataFrame 中哪些行会被包含在连接后的输出中。Python 的 pandas 库提供了几种连接类型，每种类型具有不同的行为。我们将介绍本章将要使用的用例示例，然后深入探讨不同类型的连接。

在本章中，我们的使用案例涉及员工数据和项目分配，适用于一个管理其员工和项目的公司。你可以执行以下脚本，详细查看数据框：[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/1.use_case.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/1.use_case.py)。

`employee_data` 数据框表示员工的详细信息，例如他们的姓名和部门，内容如下所示：

```py
   employee_id     name department
0            1    Alice         HR
1            2      Bob         IT
```

`project_data` 数据框包含项目分配的信息，包括项目名称：

```py
   employee_id project_name
0            2     ProjectA
1            3     ProjectB
```

在接下来的章节中，我们将讨论不同的数据框合并选项，从内连接开始。

### 内连接

内连接只返回在指定的连接列中，两个数据框中都有匹配值的行。需要特别注意以下几点：

+   任何一个数据框中具有不匹配键的行将会被排除在合并结果之外

+   具有缺失值的键列中的行将被排除在合并结果之外

内连接的结果展示在下图中：

![图 5.1 – 内连接](img/B19801_05_1.jpg)

图 5.1 – 内连接

让我们来看看如何使用 pandas `merge` 函数实现上述结果，参考前一节中的示例：

```py
merged_data = pd.merge(employee_data, project_data, on='employee_id', how='inner')
```

正如我们在前面的代码片段中看到的，`pd.merge()` 函数用于合并两个数据框。`on='employee_id'` 参数指定应使用 `employee_id` 列作为连接数据框的键。`how='inner'` 参数指定执行内连接。这种连接类型只返回在两个数据框中都有匹配值的行，在本例中就是 `employee_id` 在 `employee_data` 和 `project_data` 中匹配的行。以下表格展示了两个数据框内连接的结果：

```py
   employee_id     name department project_name
0            2      Bob         IT     ProjectA
1            3  Charlie  Marketing     ProjectB
2            4    David    Finance     ProjectC
3            5      Eva         IT     ProjectD
```

这种方法确保了两个数据框的数据是基于**公共键**合并的，只有在两个数据框中都有匹配时，才会包含对应的行，从而遵循内连接的原则。

如果仍然不清楚，以下列表展示了在数据世界中，内连接至关重要的具体示例：

+   **匹配表格**：当你需要匹配来自不同表的数据时，内连接是理想的选择。例如，如果你有一个员工表和一个部门名称表，你可以使用内连接将每个员工与他们相应的部门匹配。

+   **数据过滤**：内连接可以作为过滤器，排除那些在两个表中没有对应条目的行。这在你只希望考虑那些在多个表中有完整数据的记录时非常有用。例如，只有在客户订单和产品详情都有记录的情况下，才匹配这两者。

+   **查询执行效率**：由于内部连接只返回两个表中具有匹配值的行，因此在查询执行时间方面可能比需要检查并处理非匹配条目的外部连接更有效。

+   **减少数据重复**：内部连接通过仅返回匹配的行来帮助减少数据重复，从而确保结果集中的数据是相关的，而不是冗余的。

+   **简化复杂查询**：在处理多个表格时，内部连接可用于通过减少需要检查和处理的行数来简化查询。这在复杂的数据库模式中特别有用，其中多个表格相互关联。

从内部连接转向外部连接扩展了合并数据的范围，合并所有可用的两个数据集的行，即使它们之间没有对应的匹配项。

### 外部合并

外部合并（也称为完全外部连接）返回两个数据帧的所有行，结合匹配的行以及不匹配的行。完全外部连接确保不会丢失来自任一数据帧的数据，但在其中一个数据帧中存在不匹配行时，可能会引入 NaN 值。

外部合并的结果如下图所示：

![图 5.2 – 外部合并](img/B19801_05_2.jpg)

图 5.2 – 外部合并

让我们看看如何使用 pandas 的 `merge` 函数来实现前述结果，在上一节中提供的示例中：

```py
full_outer_merged_data = pd.merge(employee_data, project_data, on='employee_id', how='outer')
```

正如我们在前面的代码片段中看到的那样，`pd.merge()` 函数用于合并这两个数据帧。参数 `on='employee_id'` 指定了应该使用 `employee_id` 列作为合并数据帧的键。参数 `how='outer'` 指定执行完全外部连接。这种连接类型返回两个数据帧中的所有行，并在没有匹配项的地方填充 `NaN`。在以下表格中，您可以看到这两个数据帧进行外部连接的输出：

```py
   employee_id     name  department  project_name
0            1    Alice          HR           NaN
1            2      Bob          IT      ProjectA
2            3  Charlie   Marketing      ProjectB
3            4    David     Finance      ProjectC
4            5      Eva          IT      ProjectD
5            6      NaN         NaN      ProjectE
```

该方法确保合并来自两个数据帧的数据，允许全面查看所有可用数据，即使由于数据帧之间的不匹配导致部分数据不完整。

在以下列表中，我们提供了数据领域中外部合并至关重要的具体示例：

+   **包含可选数据**：当您希望包含另一个表格中具有可选数据的行时，外部连接是理想的选择。例如，如果您有一个用户表和一个单独的地址表，不是所有用户都可能有地址。外部连接允许您列出所有用户，并显示那些有地址的用户的地址，而不排除没有地址的用户。

+   **数据完整性和完整性**：在需要一个包含两张表中所有记录的全面数据集的场景中，无论是否在连接表中有匹配记录，外连接都是必不可少的。这确保了你能全面查看数据，特别是在需要展示所有实体的报告中，比如列出所有客户及其购买情况的报告，其中包括那些没有购买的客户。

+   **数据不匹配分析**：外连接可以用来识别表之间的差异或不匹配。例如，如果你在比较注册用户列表与事件参与者列表，外连接可以帮助识别未参与的用户和未注册的参与者。

+   **复杂数据合并**：在合并来自多个来源的数据时，这些数据无法完美对齐，外连接可以确保在合并过程中没有数据丢失。这在数据完整性至关重要的复杂数据环境中尤为有用。

从外连接过渡到右连接，缩小了合并数据的关注范围，强调包含右侧 DataFrame 中的所有行，同时保持左侧 DataFrame 中的匹配行。

### 右连接

右连接（也称为右外连接）返回右侧 DataFrame 中的所有行，以及左侧 DataFrame 中的匹配行。右连接的结果如下图所示：

![图 5.3 – 右连接](img/B19801_05_3.jpg)

图 5.3 – 右连接

让我们来看一下如何使用 pandas 的`merge`函数实现前述结果，参考上一节中提供的示例：

```py
right_merged_data = pd.merge(employee_data, project_data, on='employee_id', how='right')
```

`how='right'` 参数指定执行右外连接。此类型的连接返回右侧 DataFrame（`project_data`）中的所有行，以及左侧 DataFrame（`employee_data`）中的匹配行。如果没有匹配，则结果中左侧 DataFrame 的列会显示为 `NaN`。在下表中，你可以看到前述两个 DataFrame 合并的输出结果：

```py
   employee_id     name department project_name
0            2      Bob         IT     ProjectA
1            3  Charlie  Marketing     ProjectB
2            4    David    Finance     ProjectC
3            5      Eva         IT     ProjectD
4            6      NaN        NaN     ProjectE
```

在以下列表中，我们展示了数据领域中右连接至关重要的具体示例：

+   **完成数据**：当你需要确保保留右侧 DataFrame 中的所有条目时，右连接非常有用，这在右侧 DataFrame 包含必须保留的重要数据时尤其重要。

+   **数据增强**：这种类型的连接可用于通过从另一个数据集（左侧 DataFrame）中获取附加属性来丰富数据集（右侧 DataFrame），同时确保保留主数据集中的所有记录。

+   **数据不匹配分析**：与外连接类似，右连接可以帮助识别右侧 DataFrame 中哪些条目没有对应的左侧 DataFrame 条目，这对于数据清洗和验证过程至关重要。

从右连接转为左连接，改变了合并数据的视角，优先考虑包括左侧数据框的所有行，同时保持右侧数据框的匹配行。

### 左连接

左连接（也称为左外连接）返回左侧数据框的所有行以及右侧数据框的匹配行。左连接的结果如以下图所示：

![图 5.4 – 左连接](img/B19801_05_4.jpg)

图 5.4 – 左连接

让我们看看如何使用 pandas 的`merge`函数来实现前述结果，使用上一节中提供的示例：

```py
left_merged_data = pd.merge(employee_data, project_data, on='employee_id', how='left')
```

`how='left'`参数指定应执行左外连接。这种类型的连接返回左侧数据框（`employee_data`）的所有行，以及右侧数据框（`project_data`）的匹配行。如果没有匹配项，结果将会在右侧数据框的列中显示`NaN`。在以下表格中，您可以看到前述两数据框合并的结果：

```py
   employee_id     name department project_name
0            1    Alice         HR          NaN
1            2      Bob         IT     ProjectA
2            3  Charlie  Marketing     ProjectB
3            4    David    Finance     ProjectC
4            5      Eva         IT     ProjectD
```

如果你想知道何时使用左连接，那么之前关于右连接的考虑同样适用于左连接。现在我们已经讨论了合并操作，接下来我们来讨论在合并过程中可能出现的重复项如何处理。

# 合并数据集时处理重复项

在执行合并操作之前处理重复键非常重要，因为重复项可能导致意外结果，例如笛卡尔积，行数会根据匹配条目的数量而增加。这不仅会扭曲数据分析，还会因为结果数据框的大小增加而显著影响性能。

## 为什么要处理行和列中的重复项？

重复的键可能会导致一系列问题，这些问题可能会影响结果的准确性和数据处理的效率。让我们来探讨一下为什么在合并数据之前处理重复键是一个好主意：

+   如果任一表格中存在重复键，合并这些表格可能会导致**笛卡尔积**，即一个表格中的每个重复键与另一个表格中相同键的每个出现匹配，从而导致行数呈指数增长。

+   重复的键可能表示数据错误或不一致，这可能导致错误的分析或结论。

+   通过删除重复项来减少数据集的大小，可以加速合并操作的处理时间。

在理解了处理重复键的重要性后，让我们来看看在进行合并操作之前，有哪些有效的策略可以管理这些重复项。

## 删除重复行

在数据集中删除重复项涉及识别并删除基于特定键列的重复行，以确保每个条目都是唯一的。这一步不仅简化了后续的数据合并，还通过消除由重复数据引起的潜在错误来源，提高了分析的可靠性。为了展示删除重复项，我们将扩展我们一直在使用的示例，在每个 DataFrame 中添加更多的重复行。像往常一样，您可以在此查看完整代码：[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/6a.manage_duplicates.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/6a.manage_duplicates.py)。

让我们首先创建一些具有重复 `employee_id` 键的示例员工数据：

```py
employee_data = pd.DataFrame({
    'employee_id': [1, 2, 2, 3, 4, 5, 5],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'Eva', 'Eva'],
    'department': ['HR', 'IT', 'IT', 'Marketing', 'Finance', 'IT', 'IT']
})
```

让我们还创建一些具有重复 `employee_id` 键的示例项目数据：

```py
project_data = pd.DataFrame({
    'employee_id': [2, 3, 4, 5, 5, 6],
    'project_name': ['ProjectA', 'ProjectB', 'ProjectC', 'ProjectD', 'ProjectD', 'ProjectE']
})
```

现在，我们要合并这些数据集。但首先，我们将删除所有重复项，以使合并操作尽可能轻便。删除重复项后的合并操作在以下代码片段中展示：

```py
employee_data = employee_data.drop_duplicates(subset='employee_id', keep='first')
project_data = project_data.drop_duplicates(subset='employee_id', keep='first')
```

如代码所示，`drop_duplicates()` 用于根据 `employee_id` 删除重复行。`keep='first'` 参数确保仅保留首次出现的记录，其他记录将被删除。

删除重复项后，您可以继续进行合并操作，如以下代码所示：

```py
merged_data = pd.merge(employee_data, project_data, on='employee_id', how='inner')
```

合并后的数据集如下所示：

```py
   employee_id     name department project_name
0            2      Bob         IT     ProjectA
1            3  Charlie  Marketing     ProjectB
2            4    David    Finance     ProjectC
3            5      Eva         IT     ProjectD
```

`merged_data` DataFrame 包含了来自 `employee_data` 和 `project_data` 两个 DataFrame 的列，显示了每个在两个数据集中都存在的员工的 `employee_id`、`name`、`department` 和 `project_name` 的值。重复项被删除，确保每个员工在最终合并的数据集中仅出现一次。`drop_duplicates` 操作对避免数据冗余和合并过程中可能出现的冲突至关重要。接下来，我们将讨论如何确保合并操作尊重键的唯一性并遵守特定的约束条件。

## 合并前验证数据

在合并数据集时，尤其是处理大型和复杂数据集时，确保合并操作的完整性和有效性至关重要。pandas 在 `merge()` 函数中提供了 `validate` 参数，用于强制执行合并键之间的特定条件和关系。这有助于识别并防止可能影响分析的无意重复或数据不匹配。

以下代码演示了如何使用`validate`参数来强制执行`merge()`约束，并在这些约束未满足时处理异常。你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/6b.manage_duplicates_validate.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/6b.manage_duplicates_validate.py)查看完整代码：

```py
try:
    merged_data = pd.merge(employee_data, project_data, on='employee_id', how='inner', validate='one_to_many')
    print("Merged Data Result:")
    print(merged_data)
except ValueError as e:
    print("Merge failed:", e)
```

在前面的代码片段中，合并操作被包装在`try-except`代码块中。这是一种处理异常的方式，异常是指程序执行过程中发生的错误。`try`代码块包含可能引发异常的代码，在这种情况下是合并操作。如果发生异常，代码执行将跳转到`except`代码块。

如果合并操作未通过验证检查（在我们的例子中，如果左侧 DataFrame 中存在重复的键，而这些键应该是唯一的），将引发`ValueError`异常，并执行`except`代码块。`except`代码块捕获`ValueError`异常并打印`Merge failed:`消息，后跟 pandas 提供的错误信息。

执行上述代码后，你将看到以下错误消息：

```py
Merge failed: Merge keys are not unique in left dataset; not a one-to-many merge
```

`validate='one_to_many'`参数包含在合并操作中。该参数告诉 pandas 检查合并操作是否符合指定类型。在这种情况下，`one_to_many`表示合并键在左侧 DataFrame（`employee_data`）中应唯一，但在右侧 DataFrame（`project_data`）中可以有重复项。如果验证检查失败，pandas 将引发`ValueError`异常。

何时使用哪种方法

当你需要精细控制重复项的识别和处理方式，或者当重复项需要特殊处理（例如基于其他列值的聚合或转换）时，使用**手动删除重复项**。

当你希望直接在合并操作中确保数据模型的结构完整性时，使用**合并验证**，尤其是在表之间的关系明确定义并且根据业务逻辑或数据模型不应包含重复键的简单情况。

如果数据中存在重复项是有充分理由的，我们可以考虑在合并过程中采用聚合方法，以合并冗余信息。

## 聚合

聚合是管理数据集重复项的强大技术，特别是在处理应唯一但包含多个条目的关键列时。通过在这些关键列上分组数据并应用聚合函数，我们可以将重复条目合并为单一的汇总记录。可以使用求和、平均值或最大值等聚合函数，以与分析目标对齐的方式来合并或汇总数据。

让我们看看如何利用聚合来有效地处理数据重复问题。为了帮助展示这个例子，我们稍微扩展一下数据集，具体如下所示。你可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/6c.merge_and_aggregate.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/6c.merge_and_aggregate.py)看到完整示例：

```py
employee_data = pd.DataFrame({
    'employee_id': [1, 2, 2, 3, 4, 5, 5],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'Eva', 'Eva'],
    'department': ['HR', 'IT', 'IT', 'Marketing', 'Finance', 'IT', 'IT'],
    'salary': [50000, 60000, 60000, 55000, 65000, 70000, 70000]
})
# Sample project assignment data with potential duplicate keys
project_data = pd.DataFrame({
    'employee_id': [2, 3, 4, 5, 7, 6],
    'project_name': ['ProjectA', 'ProjectB', 'ProjectC', 'ProjectD', 'ProjectD', 'ProjectE']
})
```

现在，让我们进行聚合步骤：

```py
aggregated_employee_data = employee_data.groupby('employee_id').agg({
    'name': 'first', # Keep the first name encountered
    'department': 'first', # Keep the first department encountered
    'salary': 'sum' # Sum the salaries in case of duplicates
}).reset_index()
```

`groupby()`方法在`employee_data`上使用，`employee_id`作为键。这将 DataFrame 按`employee_id`分组，因为存在重复的`employee_id`值。

然后，`agg()`方法被应用于对不同列进行特定的聚合操作：

+   `'name': 'first'`和`'department': 'first'`确保在分组数据中保留这些列的首次遇到的值。

+   `'salary': 'sum'`对每个`employee_id`值的薪资进行求和，如果重复数据表示累计数据的拆分记录，这将非常有用。

在最后一步，使用`pd.merge()`函数通过在`employee_id`列上进行内连接，将`aggregated_employee_data`与`project_data`合并：

```py
merged_data = pd.merge(aggregated_employee_data, project_data, on='employee_id', how='inner')
```

这确保了只有有项目分配的员工会被包含在结果中。合并后的结果如下所示：

```py
   employee_id     name department  salary project_name
0            2      Bob         IT  120000     ProjectA
1            3  Charlie  Marketing   55000     ProjectB
2            4    David    Finance   65000     ProjectC
3            5      Eva         IT  140000     ProjectD
```

pandas 中的`agg()`方法非常灵活，提供了许多超出简单“保留首个”方法的选项。这个方法可以应用各种聚合函数来汇总数据，比如对数值进行求和、求平均值，或选择最大值或最小值。我们将在下一章深入探讨`agg()`方法的多种功能，探索如何运用这些不同的选项来提升数据准备和分析的质量。

让我们从使用聚合来处理重复数据过渡到拼接重复行，这在处理文本或类别数据时非常有效。

## 拼接

将重复行的值拼接成一行是一种有用的技巧，特别是在处理可能包含多个有效条目的文本或类别数据时。这种方法允许你保留重复数据中的所有信息，而不会丢失数据。

让我们看看如何通过拼接行来有效地处理数据重复问题，在合并数据之前。为了展示这一方法，我们将使用以下 DataFrame：

```py
employee_data = pd.DataFrame({
    'employee_id': [1, 2, 2, 3, 4, 5, 5],
    'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'Eva', 'Eva'],
    'department': ['HR', 'IT', 'Marketing', 'Marketing', 'Finance', 'IT', 'HR']
})
```

现在，让我们进行拼接步骤，如下面的代码片段所示：

```py
employee_data['department'] = employee_data.groupby('employee_id')['department'].transform(lambda x: ', '.join(x))
```

在拼接步骤中，`groupby('employee_id')`方法按`employee_id`对数据进行分组。然后，`transform(lambda x: ', '.join(x))`方法应用于`department`列。此时，使用`lambda`函数通过逗号将每个组（即`employee_id`）的`department`列的所有条目合并成一个字符串。

此操作的结果替换了`employee_data`中原始的`department`列，现在每个`employee_id`都有一个包含所有原始部门数据合并为一个字符串的单一`department`条目，如下表所示：

```py
   employee_id     name     department
0            1    Alice             HR
1            2      Bob  Marketing, IT
3            3  Charlie      Marketing
4            4    David        Finance
5            5      Eva         IT, HR
```

当你需要在重复条目中保留所有类别或文本数据，而不偏向某一条目时，可以使用连接。

这种方法有助于以可读且信息丰富的方式总结文本数据，特别是在处理可能具有多个有效值的属性时（例如，一个员工属于多个部门）。

一旦解决了每个数据框中的重复行，注意力就转向识别和解决跨数据框的重复列问题。

## 处理列中的重复

在合并来自不同来源的数据时，遇到列名重叠的数据框并不罕见。这种情况通常发生在合并类似数据集时。

扩展我们迄今为止使用的示例数据，我们将调整数据框（DataFrame）以帮助展示在处理多个数据框中共有列时可用的选项。数据可以在此查看：

```py
employee_data_1 = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'department': ['HR', 'IT', 'Marketing', 'Finance', 'IT']
})
employee_data_2 = pd.DataFrame({
    'employee_id': [6, 7, 8, 9, 10],
    'name': ['Frank', 'Grace', 'Hannah', 'Ian', 'Jill'],
    'department': ['Logistics', 'Marketing', 'IT', 'Marketing', 'Finance']
})
```

让我们看看如何通过应用不同的技巧来合并这些数据集，而不会破坏合并操作。

### 合并时处理重复列

前面展示的两个数据框中的列名相同，可能表示相同的数据。然而，我们决定在合并的数据框中保留两组列。这个决定基于这样的怀疑：尽管列名相同，但条目并不完全相同，这表明它们可能是相同数据的不同表示形式。这个问题我们可以在合并操作后再处理。

保持两个列集的最佳方法是使用`merge()`函数中的`suffixes`参数。这将允许你区分来自每个数据框的列，而不会丢失任何数据。以下是在 Python 中使用 pandas 实现这一点的方法：

```py
merged_data = pd.merge(employee_data_1, employee_data_2, on='employee_id', how='outer', suffixes=('_1', '_2'))
```

`pd.merge()`函数用于在`employee_id`上合并两个数据框。`how='outer'`参数确保包括来自两个数据框的所有记录，即使没有匹配的`employee_id`值。`suffixes=('_1', '_2')`参数为每个数据框的列添加后缀，以便在合并后的数据框中区分它们。当列名相同但来自不同数据源时，这一点尤为重要。让我们回顾一下输出数据框：

```py
   employee_id   name_1 department_1  name_2 department_2
0            1    Alice           HR     NaN          NaN
1            2      Bob           IT     NaN          NaN
2            3  Charlie    Marketing     NaN          NaN
3            4    David      Finance     NaN          NaN
4            5      Eva           IT     NaN          NaN
5            6      NaN          NaN   Frank    Logistics
6            7      NaN          NaN   Grace    Marketing
7            8      NaN          NaN  Hannah           IT
8            9      NaN          NaN     Ian    Marketing
9           10      NaN          NaN    Jill      Finance
```

这种方法在从不同来源合并数据时尤其有用，尤其是当涉及到列名重叠的情况，但同时也需要保留并清晰地区分这些列。另一个需要考虑的点是，后缀可以帮助识别数据来源的数据框，这在涉及多个来源的数据分析中非常有用。

在下一节中，我们将解释如何通过在合并之前删除列来处理重复列。

### 在合并前删除重复列

如果我们发现要合并的两个 DataFrame 中有相同列的副本，而且其中一个 DataFrame 中的列比另一个更可靠或足够使用，那么在合并操作之前删除其中一个重复列可能更为实际，而不是保留两个副本。做出这一决策的原因可能是简化数据集、减少冗余，或者当某一列对分析没有额外价值时。让我们看一下这个示例的数据：

```py
employee_data_1 = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'department': ['HR', 'IT', 'Marketing', 'Finance', 'IT']
})
employee_data_2 = pd.DataFrame({
    'employee_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'department': ['Human Resources', 'Information Technology', 'Sales', 'Financial', 'Technical']
})
```

如果我们仔细查看这些数据，可以发现两个 DataFrame 中的 `department` 列捕获了相同的信息，但格式不同。为了简化我们的示例，假设我们知道 HR 系统以第一个 DataFrame 中呈现的格式跟踪每个员工的部门。这就是为什么我们会更信任第一个 DataFrame 中的列，而不是第二个 DataFrame 中的列。因此，我们将在合并操作之前删除第二个列。下面是如何在合并之前删除列的操作：

```py
employee_data_2.drop(columns=['department'], inplace=True)
```

在合并之前，`employee_data_2` 中的 `department` 列被删除，因为它被认为不够可靠。这是通过 `drop(columns=['department'], inplace=True)` 方法完成的。在删除了不需要的列之后，我们可以继续进行合并：

```py
merged_data = pd.merge(employee_data_1, employee_data_2, on=['employee_id', 'name'], how='inner')
```

使用 `pd.merge()` 函数，以 `employee_id` 和 `name` 列作为键合并 DataFrame。使用 `how='inner'` 参数来执行内连接，只包含在两个 DataFrame 中具有匹配值的行。

为了优化合并过程并提高性能，通常在执行合并操作之前删除不必要的列是有益的，原因如下：

+   通过显著减少合并操作时的内存占用，这种做法可以提高性能，因为它最小化了需要处理和合并的数据量，从而加快了处理速度。

+   结果 DataFrame 变得更加简洁清晰，便于数据管理和后续分析。这种复杂度的减少不仅简化了合并操作，还减少了出错的可能性。

+   在资源受限的环境中，例如计算资源有限的情况，减少数据集大小在进行如合并等密集型操作之前，可以提高资源效率，并确保更顺畅的执行。

如果在 DataFrame 中存在相同的列，另一种选择是考虑是否可以将它们作为合并操作的键。

### 重复键

当遇到跨多个 DataFrame 的相同键时，一种智能的做法是基于这些共同列进行合并。让我们回顾一下前一节中提供的示例：

```py
merged_data = pd.merge(employee_data_1, employee_data_2, on=['employee_id', 'name'], how='inner')
```

我们可以看到，这里我们使用了 `['employee_id', 'name']` 作为合并的键。如果 `employee_id` 和 `name` 是可靠的标识符，能够确保在 DataFrame 之间准确匹配记录，那么它们应该作为合并的键。这确保了合并后的数据准确地代表了两个来源的结合记录。

随着数据量和复杂性的不断增长，高效地合并数据集变得至关重要，正如我们在接下来的部分中将要学习的那样。

# 合并的性能技巧

在处理大型数据集时，合并操作的性能可能会显著影响数据处理任务的整体效率。合并是数据分析中常见且常常必需的步骤，但它可能是计算密集型的，尤其是在处理大数据时。因此，采用性能优化技术对于确保合并操作尽可能快速高效地执行至关重要。

优化合并操作可以减少执行时间，降低内存消耗，并带来更加流畅的数据处理体验。在接下来的部分，我们将探讨一些可以应用于 pandas 合并操作的性能技巧，如使用索引、排序索引、选择合适的合并方法以及减少内存使用。

## 设置索引

在 pandas 中使用索引是数据处理和分析中的一个关键方面，尤其是在处理大型数据集或进行频繁的数据检索操作时。索引既是标识工具，也是高效数据访问的工具，提供了多种好处，能够显著提高性能。具体来说，在合并 DataFrame 时，使用索引能够带来性能上的提升。与基于列的合并相比，基于索引的合并通常更快，因为 pandas 可以使用优化的基于索引的连接方法来执行合并操作，这比基于列的合并更高效。让我们回顾一下员工示例来证明这一概念。此示例的完整代码可以在[`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/8a.perfomance_benchmark_set_index.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/8a.perfomance_benchmark_set_index.py)中找到。

首先，让我们导入必要的库：

```py
import pandas as pd
import numpy as np
from time import time
```

为每个 DataFrame 选择基准示例的行数：

```py
num_rows = 5
```

让我们创建示例所需的 DataFrame，这些 DataFrame 的行数将由 `num_rows` 变量定义。这里定义了第一个员工 DataFrame：

```py
employee_data_1 = pd.DataFrame({
  'employee_id': np.arange(num_rows),
  'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
  'department': ['HR', 'IT', 'Marketing', 'Finance', 'IT'],
  'salary': [50000, 60000, 70000, 80000, 90000]
})
```

第二个 DataFrame 如下所示：

```py
employee_data_2 = pd.DataFrame({
  'employee_id': np.arange(num_rows),
  'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
  'department': ['HR', 'IT', 'Sales', 'Finance', 'Operations'],
  'bonus': [3000, 4000, 5000, 6000, 7000]
})
```

为了展示我们所应用的性能技巧的效果，我们最初将执行不利用索引的合并操作。我们将计算此操作所需的时间。接着，我们会在两个 DataFrame 中设置索引并重新执行合并操作，重新计算时间。最后，我们将展示结果。希望这个方法能产生预期的效果！开始计时：

```py
start_time = time()
```

让我们在不使用索引的情况下执行合并操作，只通过`['employee_id', 'name']`进行内连接：

```py
merged_data = pd.merge(employee_data_1, employee_data_2, on=['employee_id', 'name'], how='inner', suffixes=('_1', '_2'))
```

让我们计算执行合并所花费的时间：

```py
end_time = time()
merge_time = end_time - start_time
Merge operation took around 0.00289 seconds
```

注意

执行程序的计算机可能会导致时间有所不同。这个想法是，优化后的版本比原始合并操作所需的时间更短。

通过将`employee_id`作为两个 DataFrame（`employee_data_1`和`employee_data_2`）的索引，我们让 pandas 使用基于索引的优化连接方法。这尤其有效，因为 pandas 中的索引是通过哈希表或 B 树实现的，具体取决于数据类型和索引的排序性，这有助于加速查找：

```py
employee_data_1.set_index('employee_id', inplace=True)
employee_data_2.set_index('employee_id', inplace=True)
```

在设置索引后，我们再执行一次合并操作，并重新计算时间：

```py
start_time = time()
merged_data_reduced = pd.merge(employee_data_1, employee_data_2, left_index=True, right_index=True, suffixes=('_1', '_2'))
end_time = time()
merge_reduced_time = end_time - start_time
Merge operation with reduced memory took around 0.00036 seconds
```

现在，如果我们计算从初始时间到最终时间的百分比差异，我们发现仅仅通过设置索引，我们就将时间缩短了约 88.5%。这看起来很令人印象深刻，但我们也需要讨论一些设置索引时的注意事项。

### 索引注意事项

选择合适的列进行索引设置非常重要，应基于查询模式。过度索引可能导致不必要的磁盘空间占用，并且由于维护索引的开销，可能会降低写操作性能。

**重建**或**重组索引**对于优化性能至关重要。这些任务解决了索引碎片问题，并确保随着时间推移性能的一致性。

虽然索引可以显著提高读取性能，但它们也可能影响写入性能。在优化读取操作（如搜索和连接）与保持高效的写入操作（如插入和更新）之间找到平衡至关重要。

多列索引或连接索引在多个字段经常一起用于查询时可能是有益的。然而，索引定义中字段的顺序非常重要，应反映出最常见的查询模式。

在证明了设置索引的重要性后，我们进一步讨论在合并前对索引进行排序的选项。

## 排序索引

在 pandas 中排序索引在你经常对大规模 DataFrame 进行合并或连接操作的场景中尤其有利。当索引被排序时，pandas 可以利用更高效的算法来对齐和连接数据，这可能会显著提升性能。让我们在继续代码示例之前深入探讨这一点：

+   当索引已排序时，pandas 可以使用二分查找算法来定位 DataFrame 之间的匹配行。二分查找的时间复杂度是 *O(log n)*，这比未排序索引所需的线性查找要快得多，特别是当 DataFrame 的大小增加时。

+   排序索引有助于更快地对齐数据。这是因为 pandas 可以对数据的顺序做出一些假设，从而简化在合并时查找每个 DataFrame 中对应行的过程。

+   使用排序后的索引，pandas 可以避免进行不必要的比较，这些比较是当索引未排序时所必需的。这样可以减少计算开销，加速合并过程。

让我们回到代码示例，加入索引排序的步骤。原始数据保持不变；但是在本实验中，我们比较的是在设置索引后执行合并操作的时间与在设置并排序索引后执行合并操作的时间。以下代码展示了主要的代码组件，但和往常一样，你可以通过 [`github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/8b.performance_benchmark_sort_indexes.py`](https://github.com/PacktPublishing/Python-Data-Cleaning-and-Preparation-Best-Practices/blob/main/chapter05/8b.performance_benchmark_sort_indexes.py) 跟进完整示例：

```py
employee_data_1.set_index('employee_id', inplace=True)
employee_data_2.set_index('employee_id', inplace=True)
```

让我们在不排序索引的情况下执行合并操作：

```py
merged_data = pd.merge(employee_data_1, employee_data_2, left_index=True, right_index=True, suffixes=('_1', '_2'))
Merge operation with setting index took around 0.00036 seconds
```

让我们在排序索引后重复合并操作，并再次计算时间：

```py
employee_data_1.sort_index(inplace=True)
employee_data_2.sort_index(inplace=True)
merged_data_reduced = pd.merge(employee_data_1, employee_data_2, left_index=True, right_index=True, suffixes=('_1', '_2'))
Merge operation after sorting took around 0.00028 seconds.
```

现在，如果我们计算从初始时间到最终时间的百分比差异，我们会发现通过排序索引，我们成功地将时间减少了大约 ~22%。这看起来很不错，但我们也需要讨论设置索引时的一些注意事项。

### 排序索引的注意事项

排序 DataFrame 的索引并不是没有计算成本的。初始的排序操作本身需要时间，因此当排序后的 DataFrame 在多个合并或连接操作中被使用时，这种做法最为有利，可以通过这些操作摊销排序的成本。

排序有时会增加内存开销，因为 pandas 可能会创建 DataFrame 索引的排序副本。在处理非常大的数据集时，若内存是一个限制因素，应该考虑这一点。

排序索引最有利的情况是，合并所用的键不仅是唯一的，而且具有一定的逻辑顺序，例如时间序列数据或有序的分类数据。

索引管理和维护是你在处理 pandas DataFrame 时需要考虑的关键因素，尤其是在处理大型数据集时。维护一个良好的索引需要谨慎考虑。例如，定期更新或重新索引 DataFrame 可能会引入计算成本，类似于排序操作。每次修改索引（通过排序、重新索引或重置）时，可能会导致额外的内存使用和处理时间，尤其是在大型数据集上。

索引需要以平衡性能和资源使用的方式进行维护。例如，如果你经常合并或连接 DataFrame，确保索引已正确排序并且是唯一的，可以显著加速这些操作。然而，持续维护一个已排序的索引可能会消耗大量资源，因此当 DataFrame 需要进行多次操作并利用已排序的索引时，这样做最为有利。

此外，选择合适的索引类型——无论是基于整数的简单索引、用于时间序列数据的日期时间索引，还是用于层次数据的多级索引——都可能影响 pandas 处理数据的效率。索引的选择应与数据的结构和访问模式相匹配，以最小化不必要的开销。

在接下来的部分，我们将讨论使用 `join` 函数而非 `merge` 如何影响性能。

## 合并与连接

虽然合并是根据特定条件或键来合并数据集的常用方法，但还有另一种方法：`join` 函数。这个函数提供了一种简化的方式，主要通过索引执行合并，为更通用的合并函数提供了一个更简单的替代方案。当涉及的 DataFrame 已经将索引设置为用于连接的键时，pandas 中的 `join` 方法特别有用，它能够高效、直接地进行数据组合，而无需指定复杂的连接条件。

使用 `join` 函数代替 `merge` 可能会以多种方式影响性能，主要是因为这两个函数的底层机制和默认行为：

+   pandas 中的 `join` 函数针对基于索引的连接进行了优化，意味着它在通过索引连接 DataFrame 时被设计得更为高效。如果你的 DataFrame 已经按你想要连接的键进行了索引，那么使用 `join` 可以更高效，因为它利用了优化过的索引结构[2][6][7]。

+   Join 是 merge 的简化版本，默认按索引进行连接。这种简化可能带来性能上的优势，尤其是对于那些连接任务简单、合并复杂性不必要的场景。通过避免对非索引列的对齐开销，在这些情况下，join 可以更快速地执行[2][6]。

+   从底层实现来看，join 使用的是 merge[2][6]。

+   在连接大型 DataFrame 时，join 和 merge 处理内存的方式会影响性能。通过专注于基于索引的连接，join 可能在某些场景下更高效地管理内存使用，尤其是当 DataFrame 具有 pandas 可优化的索引时 [1][3][4]。

+   虽然 merge 提供了更大的灵活性，允许在任意列上进行连接，但这种灵活性带来了性能上的代价，尤其是在涉及多个列或非索引连接的复杂连接时。由于其更具体的使用场景，join 在较简单的基于索引的连接中具有性能优势 [2][6]。

总结来说，选择 `join` 还是 `merge` 取决于任务的具体需求。如果连接操作主要基于索引，join 可以因其针对基于索引的连接进行优化而提供性能优势，且其接口更为简洁。然而，对于涉及特定列或多个键的更复杂连接需求，merge 提供了必要的灵活性，尽管这可能会对性能产生影响。

# 连接 DataFrame

当你有多个结构相似（列相同或行相同）的 DataFrame，且想将它们合并成一个 DataFrame 时，连接操作非常适用。连接过程可以沿特定轴进行，按行（`axis=0`）或按列（`axis=1`）连接。

让我们深入了解按行连接，也称为附加（append）。

## 按行连接

按行连接用于沿 `axis=0` 将一个 DataFrame 连接到另一个 DataFrame。为了展示这个操作，可以看到两个结构相同但数据不同的 DataFrame，`employee_data_1` 和 `employee_data_2`：

```py
employee_data_1 = pd.DataFrame({
  'employee_id': np.arange(1, 6),
  'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
  'department': ['HR', 'IT', 'Marketing', 'Finance', 'IT']
})
employee_data_2 = pd.DataFrame({
  'employee_id': np.arange(6, 11),
  'name': ['Frank', 'Grace', 'Hannah', 'Ian', 'Jill'],
  'department': ['Logistics', 'HR', 'IT', 'Marketing', 'Finance']
})
```

让我们执行按行连接，如以下代码片段所示：

```py
concatenated_data = pd.concat([employee_data_1, employee_data_2], axis=0)
```

`pd.concat()` 函数用于连接两个 DataFrame。第一个参数是要连接的 DataFrame 列表，`axis=0` 参数指定连接应按行进行，将 DataFrame 堆叠在一起。

结果可以在这里看到：

```py
   employee_id     name department
0            1    Alice         HR
1            2      Bob         IT
2            3  Charlie  Marketing
3            4    David    Finance
4            5      Eva         IT
0            6    Frank  Logistics
1            7    Grace         HR
2            8   Hannah         IT
3            9      Ian  Marketing
4           10     Jill    Finance
```

执行按行连接时，需要考虑的几点如下：

+   确保你要连接的列对齐正确。pandas 会自动按列名对齐列，并用 `NaN` 填充任何缺失的列。

+   连接后，你可能希望重置结果 DataFrame 的索引，以避免重复的索引值，尤其是当原始 DataFrame 各自有自己的索引范围时。请在执行 `reset` 操作之前，观察以下示例中的索引：

    ```py
       employee_id     name  department
    0            1    Alice          HR
    1            2      Bob          IT
    2            3  Charlie   Marketing
    3            4    David     Finance
    4            5      Eva          IT
    0            6    Frank   Logistics
    1            7    Grace          HR
    2            8   Hannah          IT
    3            9      Ian   Marketing
    4           10     Jill     Finance
    concatenated_data_reset = concatenated_data.reset_index(drop=True)
    ```

    让我们再次查看输出：

    ```py
       employee_id     name  department
    0            1    Alice          HR
    1            2      Bob          IT
    2            3  Charlie   Marketing
    3            4    David     Finance
    4            5      Eva          IT
    5            6    Frank   Logistics
    6            7    Grace          HR
    7            8   Hannah          IT
    8            9      Ian   Marketing
    9           10     Jill     Finance
    ```

    重置索引会为连接后的数据框创建一个新的连续索引。使用`drop=True`参数可以避免将旧索引作为列添加到新数据框中。这个步骤对于保持数据框的整洁至关重要，特别是当索引本身不携带有意义的数据时。一个连续的索引通常更容易操作，尤其是在索引、切片以及未来的合并或连接操作中。

+   连接操作可能会增加程序的内存使用，特别是当处理大型数据框时。需要注意可用的内存资源。

在下一节中，我们将讨论按列连接。

## 按列连接

在 pandas 中，按列连接数据框涉及将两个或更多的数据框并排组合，通过索引对齐它们。为了展示这个操作，我们将使用之前的两个数据框，`employee_data_1`和`employee_data_2`，操作可以像这样进行：

```py
concatenated_data = pd.concat([employee_data_1, employee_performance], axis=1)
```

`pd.concat()`函数与`axis=1`参数一起使用，用于并排连接数据框。这通过索引对齐数据框，有效地将`employee_performance`中的新列添加到`employee_data_1`中。输出将显示如下：

```py
  employee_id     name department employee_id performance_rating
0           1    Alice         HR           1            3
1           2      Bob         IT           2            4
2           3  Charlie  Marketing           3            5
3           4    David    Finance           4            3
4           5      Eva         IT           5            4
```

在进行按列连接时，你需要考虑的几个事项如下：

+   要连接的数据框的索引会被正确对齐。在按列连接数据框时，结果数据框中的每一行应理想地代表来自同一实体的数据（例如，同一员工）。索引未对齐可能导致来自不同实体的数据被错误地组合，从而产生不准确和误导性的结果。例如，如果索引表示员工 ID，未对齐可能导致某个员工的详细信息与另一个员工的表现数据错误地配对。

+   如果数据框中包含相同名称的列，但这些列打算是不同的，考虑在连接之前重命名这些列，以避免在结果数据框中产生混淆或错误。

+   虽然按列连接通常不像按行连接那样显著增加内存使用，但仍然需要监控内存使用，尤其是对于大型数据框。

连接与连接操作的比较

**连接**主要用于沿轴（行或列）组合数据框，而不考虑其中的值。它适用于那些你只是想根据顺序将数据框堆叠在一起或通过附加列扩展它们的情况。

**连接**用于根据一个或多个键（每个数据框中的公共标识符）组合数据框。这更多是基于共享数据点合并数据集，允许更复杂和有条件的数据组合。

在探讨了 pandas 中拼接操作的细节之后，包括它在对齐索引方面的重要性，以及它与连接操作的对比，我们现在总结讨论的关键点，概括我们的理解，并突出我们在探索 pandas 中 DataFrame 操作时的关键收获。

# 总结

在本章中，我们探讨了 pandas 中 DataFrame 操作的各个方面，重点讨论了拼接、合并以及索引管理的重要性。

我们讨论了合并操作，它适用于基于共享键的复杂组合，并通过内连接、外连接、左连接和右连接等多种连接类型提供灵活性。我们还讨论了如何使用拼接操作在特定轴上（按行或按列）合并 DataFrame，这对于追加数据集或为数据添加新维度尤其有用。我们还讨论了这些操作的性能影响，强调了正确的索引管理可以显著提升这些操作的效率，特别是在处理大数据集时。

在接下来的章节中，我们将深入探讨如何利用`groupby`函数与各种聚合函数结合，从复杂的数据结构中提取有意义的洞察。

# 参考资料

1.  [`github.com/pandas-dev/pandas/issues/38418`](https://github.com/pandas-dev/pandas/issues/38418)

1.  [`realpython.com/pandas-merge-join-and-concat/`](https://realpython.com/pandas-merge-join-and-concat/)

1.  [`datascience.stackexchange.com/questions/44476/merging-dataframes-in-pandas-is-taking-a-surprisingly-long-time`](https://datascience.stackexchange.com/questions/44476/merging-dataframes-in-pandas-is-taking-a-surprisingly-long-time)

1.  [`stackoverflow.com/questions/40860457/improve-pandas-merge-performance`](https://stackoverflow.com/questions/40860457/improve-pandas-merge-performance)

1.  [`www.youtube.com/watch?v=P6hSBrxs0Eg`](https://www.youtube.com/watch?v=P6hSBrxs0Eg)

1.  [`pandas.pydata.org/pandas-docs/version/1.5.1/user_guide/merging.html`](https://pandas.pydata.org/pandas-docs/version/1.5.1/user_guide/merging.html)

1.  [`pandas.pydata.org/pandas-docs/version/0.20/merging.html`](https://pandas.pydata.org/pandas-docs/version/0.20/merging.html)
