# 第十章：使用 PySpark 扩展单节点机器学习

在*第五章**，使用 PySpark 进行可扩展的机器学习*中，你学习了如何利用**Apache** **Spark**的分布式计算框架进行大规模的**机器学习**（**ML**）模型训练和评分。Spark 的本地 ML 库涵盖了数据科学家通常执行的标准任务；然而，还有许多标准的单节点**Python**库提供了丰富的功能，这些库并不是为分布式工作方式设计的。本章讨论了如何将标准 Python 数据处理和 ML 库（如**pandas**、**scikit-learn**、**XGBoost**等）水平扩展到分布式环境。它还涵盖了典型数据科学任务的扩展，如**探索性数据分析**（**EDA**）、**模型训练**、**模型推断**，最后，还介绍了一种名为**Koalas**的可扩展 Python 库，它允许你使用熟悉且易于使用的 pandas 类似语法编写**PySpark**代码。

本章将涵盖以下主要主题：

+   扩展 EDA

+   扩展模型推断

+   分布式超参数调优

+   使用**极易并行计算**进行模型训练

+   使用 Koalas 将 pandas 升级到 PySpark

本章中获得的一些技能包括大规模执行 EDA、大规模执行模型推断和评分、超参数调优，以及单节点模型的最佳模型选择。你还将学习如何水平扩展几乎所有的单节点 ML 模型，最后使用 Koalas，它让我们能够使用类似 pandas 的 API 编写可扩展的 PySpark 代码。

# 技术要求

+   在本章中，我们将使用 Databricks 社区版运行代码：

    [`community.cloud.databricks.com`](https://community.cloud.databricks.com)

    注册说明可以在[`databricks.com/try-databricks`](https://databricks.com/try-databricks)找到。

+   本章中使用的代码和数据可以从[`github.com/PacktPublishing/Essential-PySpark-for-Scalable-Data-Analytics/tree/main/Chapter10`](https://github.com/PacktPublishing/Essential-PySpark-for-Scalable-Data-Analytics/tree/main/Chapter10)下载。

# 扩展 EDA

EDA 是一种数据科学过程，涉及对给定数据集的分析，以了解其主要特征，有时通过可视化图表，有时通过数据聚合和切片。你已经在*第十一章**，使用 PySpark 进行数据可视化*中学习了一些可视化的 EDA 技术。在这一节中，我们将探索使用 pandas 进行非图形化的 EDA，并将其与使用 PySpark 和 Koalas 执行相同过程进行比较。

## 使用 pandas 进行 EDA

标准 Python 中的典型 EDA 涉及使用 pandas 进行数据处理，使用 `matplotlib` 进行数据可视化。我们以一个来自 scikit-learn 的示例数据集为例，执行一些基本的 EDA 步骤，如以下代码示例所示：

```py
import pandas as pd
from sklearn.datasets import load_boston
boston_data = datasets.load_boston()
boston_pd = pd.DataFrame(boston_data.data, 
                         columns=boston_data.feature_names)
boston_pd.info()
boston_pd.head()
boston_pd.shape
boston_pd.isnull().sum()
boston_pd.describe()
```

在前面的代码示例中，我们执行了以下步骤：

1.  我们导入 pandas 库并导入 scikit-learn 提供的示例数据集 `load_boston`。

1.  然后，我们使用 `pd.DataFrame()` 方法将 scikit-learn 数据集转换为 pandas DataFrame。

1.  现在我们有了一个 pandas DataFrame，可以对其进行分析，首先使用 `info()` 方法，打印关于 pandas DataFrame 的信息，如列名及其数据类型。

1.  pandas DataFrame 上的 `head()` 函数打印出实际 DataFrame 的几行几列，并帮助我们从 DataFrame 中直观地检查一些示例数据。

1.  pandas DataFrame 上的 `shape` 属性打印出行和列的数量。

1.  `isnull()` 方法显示 DataFrame 中每一列的 NULL 值数量。

1.  最后，`describe()` 方法打印出每一列的统计数据，如均值、中位数和标准差。

这段代码展示了使用 Python pandas 数据处理库执行的一些典型 EDA 步骤。现在，让我们看看如何使用 PySpark 执行类似的 EDA 步骤。

## 使用 PySpark 进行 EDA

PySpark 也有类似于 pandas DataFrame 的 DataFrame 构造，你可以使用 PySpark 执行 EDA，如以下代码示例所示：

```py
boston_df = spark.createDataFrame(boston_pd)
boston_df.show()
print((boston_df.count(), len(boston_df.columns)))
boston_df.where(boston_df.AGE.isNull()).count()
boston_df.describe().display()
```

在前面的代码示例中，我们执行了以下步骤：

1.  我们首先使用 `createDataFrame()` 函数将前一部分创建的 pandas DataFrame 转换为 Spark DataFrame。

1.  然后，我们使用 `show()` 函数展示 Spark DataFrame 中的一小部分数据。虽然也有 `head()` 函数，但 `show()` 能以更好的格式和更易读的方式展示数据。

1.  Spark DataFrame 没有内置的函数来显示 Spark DataFrame 的形状。相反，我们使用 `count()` 函数计算行数，使用 `len()` 方法计算列数，以实现相同的功能。

1.  同样，Spark DataFrame 也不支持类似 pandas 的 `isnull()` 函数来统计所有列中的 NULL 值。相反，我们使用 `isNull()` 和 `where()` 的组合，逐列过滤掉 NULL 值并进行计数。

1.  Spark DataFrame 确实支持 `describe()` 函数，可以在分布式模式下计算每列的基本统计数据，通过后台运行一个 Spark 作业来实现。对于小型数据集这可能看起来不太有用，但对于描述非常大的数据集来说，它非常有用。

通过使用 Spark DataFrame 提供的内置函数和操作，你可以轻松地扩展你的 EDA。由于 Spark DataFrame 本身支持**Spark** **SQL**，你可以在使用 DataFrame API 进行 EDA 的同时，也通过 Spark SQL 执行可扩展的 EDA。

# 扩展模型推理

除了数据清洗、模型训练和调优外，整个 ML 过程的另一个重要方面是模型的生产化。尽管有大量的数据可供使用，但有时将数据下采样，并在较小的子集上训练模型是有用的。这可能是由于信噪比低等原因。在这种情况下，不需要扩展模型训练过程本身。然而，由于原始数据集的大小非常庞大，因此有必要扩展实际的模型推理过程，以跟上生成的大量原始数据。

Apache Spark 与**MLflow**一起，可以用来对使用标准非分布式 Python 库（如 scikit-learn）训练的模型进行评分。以下代码示例展示了一个使用 scikit-learn 训练的模型，随后使用 Spark 进行大规模生产化的示例：

```py
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = boston_pd[features]
y = boston_pd['MEDV']
with mlflow.start_run() as run1:
  lr = LinearRegression()
  lr_model = lr.fit(X_train,y_train)
  mlflow.sklearn.log_model(lr_model, "model")
```

在上一个代码示例中，我们执行了以下步骤：

1.  我们打算使用 scikit-learn 训练一个线性回归模型，该模型在给定一组特征的情况下预测波士顿房价数据集中的中位数房价。

1.  首先，我们导入所有需要的 scikit-learn 模块，同时我们还导入了 MLflow，因为我们打算将训练好的模型记录到**MLflow** **Tracking** **Server**。

1.  然后，我们将特征列定义为变量`X`，标签列定义为`y`。

1.  然后，我们使用`with mlflow.start_run()`方法调用一个 MLflow 实验。

1.  然后，我们使用`LinearRegression`类训练实际的线性回归模型，并在训练的 pandas DataFrame 上调用`fit()`方法。

1.  然后，我们使用`mlflow.sklearn.log_model()`方法将结果模型记录到 MLflow 追踪服务器。`sklearn`限定符指定记录的模型是 scikit-learn 类型。

一旦我们将训练好的线性回归模型记录到 MLflow 追踪服务器，我们需要将其转换为 PySpark 的**用户定义函数**（**UDF**），以便能够以分布式方式进行推理。实现这一目标所需的代码如下所示：

```py
import mlflow.pyfunc
from pyspark.sql.functions import struct
model_uri = "runs:/" + run1.info.run_id + "/model"
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
predicted_df = boston_df.withColumn("prediction", pyfunc_udf(struct('CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO', 'B', 'LSTAT')))
predicted_df.show()
```

在上一个代码示例中，我们执行了以下步骤：

1.  我们从 mlflow 库中导入`pyfunc`方法，用于将 mlflow 模型转换为 PySpark UDF。

1.  然后，我们通过`run_id`实验从 MLflow 构建`model_uri`。

1.  一旦我们拥有了`model_uri`，我们使用`mlflow.pyfunc()`方法将模型注册为 PySpark UDF。我们指定模型类型为`spark`，因为我们打算在 Spark DataFrame 中使用它。

1.  现在，模型已经作为 PySpark 的 UDF 注册，我们可以用它对 Spark DataFrame 进行预测。我们通过使用该模型创建一个新的 Spark DataFrame 列，并将所有特征列作为输入。结果是一个包含每一行预测值的新列的数据框。

1.  需要注意的是，当调用`show`操作时，它会启动一个 Spark 作业，并以分布式方式执行模型评分。

通过这种方式，结合使用 MLflow 的`pyfunc`方法和 Spark DataFrame 操作，使用像 scikit-learn 这样的标准单节点 Python 机器学习库构建的模型，也可以以分布式方式进行推理推导，从而实现大规模推理。此外，推理的 Spark 作业可以被配置为将预测结果写入持久化存储方法，如数据库、数据仓库或数据湖，且该作业本身可以定期运行。这也可以通过使用**结构化流处理**轻松扩展，以近实时方式通过流式 DataFrame 进行预测。

# 使用令人尴尬的并行计算进行模型训练

如你之前所学，Apache Spark 遵循**数据并行处理**的**分布式计算**范式。在数据并行处理中，数据处理代码被移动到数据所在的地方。然而，在传统的计算模型中，如标准 Python 和单节点机器学习库所使用的，数据是在单台机器上处理的，并且期望数据存在于本地。为单节点计算设计的算法可以通过多进程和多线程技术利用本地 CPU 来实现某种程度的并行计算。然而，这些算法本身并不具备分布式能力，需要完全重写才能支持分布式计算。**Spark** **ML** **库**就是一个例子，传统的机器学习算法已被完全重新设计，以便在分布式计算环境中工作。然而，重新设计每个现有的算法将是非常耗时且不切实际的。此外，已经存在丰富的基于标准的 Python 机器学习和数据处理库，如果能够在分布式计算环境中利用这些库，将会非常有用。这就是令人尴尬的并行计算范式发挥作用的地方。

在分布式计算中，同一计算过程在不同机器上执行数据的不同部分，这些计算过程需要相互通信，以完成整体计算任务。然而，在令人尴尬的并行计算中，算法不需要各个进程之间的通信，它们可以完全独立地运行。在 Apache Spark 框架内，有两种方式可以利用令人尴尬的并行计算进行机器学习训练，接下来的部分将介绍这两种方式。

## 分布式超参数调优

机器学习过程中的一个关键步骤是模型调优，数据科学家通过调整模型的超参数来训练多个模型。这种技术通常被称为超参数调优。超参数调优的常见方法叫做**网格搜索**，它是一种寻找能够产生最佳性能模型的超参数组合的方法。网格搜索通过**交叉验证**选择最佳模型，交叉验证将数据分为训练集和测试集，并通过测试数据集评估训练模型的表现。在网格搜索中，由于多个模型是在相同数据集上训练的，它们可以独立地进行训练，这使得它成为一个适合显式并行计算的候选方法。

使用标准 scikit-learn 进行网格搜索的典型实现，通过以下代码示例进行了说明：

```py
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
digits_pd = load_digits()
X = digits_pd.data 
y = digits_pd.target
parameter_grid = {"max_depth": [2, None],
              "max_features": [1, 2, 5],
              "min_samples_split": [2, 3, 5],
              "min_samples_leaf": [1, 2, 5],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [5, 10, 15, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), 
                           param_grid=parameter_grid)
grid_search.fit(X, y) 
```

在前面的代码示例中，我们执行了以下步骤：

1.  首先，我们导入`GridSearchCV`模块、`load_digits`样本数据集，以及 scikit-learn 中与`RandomForestClassifier`相关的模块。

1.  然后，我们从 scikit-learn 样本数据集中加载`load_digits`数据，并将特征映射到`X`变量，将标签列映射到`y`变量。

1.  然后，我们通过指定`RandomForestClassifier`算法使用的各种超参数的值，如`max_depth`、`max_features`等，定义需要搜索的参数网格空间。

1.  然后，我们通过调用`GridSearchCV()`方法来启动网格搜索交叉验证，并使用`fit()`方法执行实际的网格搜索。

通过使用 scikit-learn 的内置网格搜索和交叉验证器方法，你可以执行模型超参数调优，并从多个训练模型中识别出最佳模型。然而，这个过程是在单台机器上运行的，因此模型是一个接一个地训练，而不是并行训练。使用 Apache Spark 和名为`spark_sklearn`的第三方 Spark 包，你可以轻松实现网格搜索的显式并行实现，以下代码示例演示了这一点：

```py
from sklearn import grid_search
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from spark_sklearn import GridSearchCV
digits_pd = load_digits()
X = digits_pd.data 
y = digits_pd.target
parameter_grid = {"max_depth": [2, None],
              "max_features": [1, 2, 5],
              "min_samples_split": [2, 3, 5],
              "min_samples_leaf": [1, 2, 5],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [5, 10, 15, 20]}
grid_search = grid_search.GridSearchCV(RandomForestClassifier(), 
             param_grid=parameter_grid)
grid_search.fit(X, y)
```

使用`grid_sklearn`进行网格搜索的前面代码片段与使用标准 scikit-learn 进行网格搜索的代码几乎相同。然而，我们不再使用 scikit-learn 的网格搜索和交叉验证器，而是使用`grid_sklearn`包中的网格搜索和交叉验证器。这有助于以分布式方式运行网格搜索，在不同的机器上对相同数据集进行不同超参数组合的模型训练。这显著加快了模型调优过程，使你能够从比仅使用单台机器时更大的训练模型池中选择模型。通过这种方式，利用 Apache Spark 上的显式并行计算概念，你可以在仍然使用 Python 的标准单节点机器库的情况下，扩展模型调优任务。

在接下来的部分中，我们将看到如何使用 Apache Spark 的 pandas UDF 来扩展实际的模型训练，而不仅仅是模型调优部分。

## 使用 pandas UDF 扩展任意 Python 代码

一般来说，UDF 允许你在 Spark 的执行器上执行任意代码。因此，UDF 可以用于扩展任意 Python 代码，包括特征工程和模型训练，适用于数据科学工作流。它们还可以用于使用标准 Python 扩展数据工程任务。然而，UDF 每次只能执行一行代码，并且在 JVM 与运行在 Spark 执行器上的 Python 进程之间会产生**序列化**和**反序列化**的开销。这个限制使得 UDF 在将任意 Python 代码扩展到 Spark 执行器时不太具备吸引力。

使用`groupby`操作符，将 UDF 应用于每个组，最终将每个组生成的单独 DataFrame 合并成一个新的 Spark DataFrame 并返回。标量以及分组的 pandas UDF 示例可以在 Apache Spark 的公开文档中找到：

[`spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.pandas_udf.html`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.pandas_udf.html)

到目前为止，你已经看到如何使用 Apache Spark 支持的不同技术来扩展 EDA 过程、模型调优过程，或者扩展任意 Python 函数。在接下来的部分中，我们将探索一个建立在 Apache Spark 之上的库，它让我们可以使用类似 pandas 的 API 来编写 PySpark 代码。

# 使用 Koalas 将 pandas 升级到 PySpark

pandas 是标准 Python 中数据处理的事实标准，Spark 则成为了分布式数据处理的事实标准。pandas API 是与 Python 相关的，并利用 Python 独特的特性编写可读且优雅的代码。然而，Spark 是基于 JVM 的，即使是 PySpark 也深受 Java 语言的影响，包括命名约定和函数名称。因此，pandas 用户转向 PySpark 并不容易或直观，且涉及到相当的学习曲线。此外，PySpark 以分布式方式执行代码，用户需要理解如何在将 PySpark 代码与标准单节点 Python 代码混合时，分布式代码的工作原理。这对普通的 pandas 用户来说，是使用 PySpark 的一大障碍。为了解决这个问题，Apache Spark 开发者社区基于 PySpark 推出了另一个开源库，叫做 Koalas。

Koalas 项目是基于 Apache Spark 实现的 pandas API。Koalas 帮助数据科学家能够立即使用 Spark，而不需要完全学习新的 API 集。此外，Koalas 帮助开发人员在不需要在两个框架之间切换的情况下，维护一个同时适用于 pandas 和 Spark 的单一代码库。Koalas 随`pip`一起打包发布。

让我们看几个代码示例，了解 Koalas 如何提供类似 pandas 的 API 来与 Spark 一起使用：

```py
import koalas as ks
boston_data = load_boston()
boston_pd = ks.DataFrame(boston_data.data, columns=boston_data.feature_names)
features = boston_data.feature_names
boston_pd['MEDV'] = boston_data.target
boston_pd.info()
boston_pd.head()
boston_pd.isnull().sum()
boston_pd.describe()
```

在前面的代码片段中，我们执行了本章早些时候所进行的相同的基本 EDA 步骤。唯一的不同是，这里我们没有直接从 scikit-learn 数据集创建 pandas DataFrame，而是导入 Koalas 库后创建了一个 Koalas DataFrame。你可以看到，代码和之前写的 pandas 代码完全相同，然而，在幕后，Koalas 会将这段代码转换为 PySpark 代码，并在集群上以分布式方式执行。Koalas 还支持使用 `DataFrame.plot()` 方法进行可视化，就像 pandas 一样。通过这种方式，你可以利用 Koalas 扩展任何现有的基于 pandas 的机器学习代码，比如特征工程或自定义机器学习代码，而无需先用 PySpark 重写代码。

Koalas 是一个活跃的开源项目，得到了良好的社区支持。然而，Koalas 仍处于初期阶段，并且有一些局限性。目前，只有大约 *70%* 的 pandas API 在 Koalas 中可用，这意味着一些 pandas 代码可能无法直接使用 Koalas 实现。Koalas 和 pandas 之间存在一些实现差异，并且在 Koalas 中实现某些 pandas API 并不合适。处理缺失的 Koalas 功能的常见方法是将 Koalas DataFrame 转换为 pandas 或 PySpark DataFrame，然后使用 pandas 或 PySpark 代码来解决问题。Koalas DataFrame 可以通过 `DataFrame.to_pandas()` 和 `DataFrame.to_spark()` 函数分别轻松转换为 pandas 和 PySpark DataFrame。然而，需注意的是，Koalas 在幕后使用的是 Spark DataFrame，Koalas DataFrame 可能过大，无法在单台机器上转换为 pandas DataFrame，从而导致内存溢出错误。

# 摘要

在本章中，你学习了一些技术来水平扩展基于 Python 的标准机器学习库，如 scikit-learn、XGBoost 等。首先，介绍了使用 PySpark DataFrame API 扩展 EDA（探索性数据分析）的方法，并通过代码示例进行了展示。接着，介绍了结合使用 MLflow pyfunc 功能和 Spark DataFrame 来分布式处理机器学习模型的推断和评分技术。还介绍了使用 Apache Spark 进行令人尴尬的并行计算技术扩展机器学习模型的方法。此外，还介绍了使用名为 `spark_sklearn` 的第三方包对标准 Python 机器学习库训练的模型进行分布式调优的方法。然后，介绍了 pandas UDF（用户定义函数），它可以以向量化的方式扩展任意 Python 代码，用于在 PySpark 中创建高性能、低开销的 Python 用户定义函数。最后，Koalas 被介绍为一种让 pandas 开发者无需首先学习 PySpark API，就能使用类似 pandas 的 API，同时仍能利用 Apache Spark 在大规模数据处理中的强大性能和效率的方法。
