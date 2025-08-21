

# 第四章：使用 Streamlit 进行机器学习与人工智能

数据科学家常见的一个情况是，在模型创建过程的最后，无法确定如何说服非数据科学家相信他们的模型是有价值的。他们可能有模型的性能指标或一些静态可视化，但没有一个简单的方式让其他人与他们的模型进行互动。

在 Streamlit 之前，有几个其他选项，其中最流行的是在 Flask 或 Django 中创建一个完整的应用程序，或者甚至将模型转化为**应用程序编程接口**（**API**），并引导开发人员使用它。这些都是不错的选择，但往往需要耗费时间，并且对于像应用原型开发这样的宝贵用例来说并不理想。

团队的激励机制在这里有些不一致。数据科学家希望为他们的团队创建最好的模型，但如果他们需要花一天或两天的时间（或者，如果有经验的话，几个小时）将模型转化为 Flask 或 Django 应用，直到他们认为模型过程几乎完成时才去做这个，似乎没有太大意义。理想情况下，数据科学家应该早期并且经常地与利益相关者进行沟通，这样他们才能构建人们真正需要的东西！

Streamlit 的好处在于，它帮助我们将这个繁琐的过程转化为一个无缝的应用程序创建体验。在本章中，我们将介绍如何在 Streamlit 中创建**机器学习**（**ML**）原型，如何为你的机器学习应用添加用户互动，以及如何理解机器学习结果。我们将使用包括 PyTorch、Hugging Face、OpenAI 和 scikit-learn 在内的最流行的机器学习库来完成这一切。

本章将涵盖以下主题：

+   标准机器学习工作流程

+   预测企鹅物种

+   利用预训练的机器学习模型

+   在 Streamlit 应用中训练模型

+   理解机器学习结果

+   集成外部机器学习库——Hugging Face 示例

+   集成外部人工智能库——OpenAI 示例

# 技术要求

本章我们将需要一个 OpenAI 账户。要创建账户，请前往([`platform.openai.com/`](https://platform.openai.com/))并按照页面上的说明操作。

# 标准机器学习工作流程

创建一个使用机器学习（ML）的应用程序的第一步是创建机器学习模型本身。创建自己的机器学习模型有许多流行的工作流程。你可能已经有自己的方法了！这个过程有两个部分需要考虑：

+   机器学习模型的生成

+   在生产环境中使用机器学习模型

如果计划是一次训练一个模型，然后在 Streamlit 应用中使用该模型，最好的方法是先在 Streamlit 外部创建这个模型（例如，在 Jupyter 笔记本或标准 Python 文件中），然后在应用中使用该模型。

如果计划是使用用户输入来训练我们应用中的模型，那么我们就不能再在 Streamlit 外部创建模型，而需要在 Streamlit 应用内进行模型训练。

我们将首先在 Streamlit 之外构建我们的机器学习模型，然后再将模型训练过程移入 Streamlit 应用中。

# 预测企鹅物种

本章中我们主要使用的数据集是我们在*第一章*《Streamlit 入门》中使用的 Palmer Penguins 数据集。按照惯例，我们将创建一个新文件夹来存放我们的新 Streamlit 应用和相关代码。

以下代码会在我们的`streamlit_apps`文件夹内创建一个新文件夹，并将数据从我们的`penguin_app`文件夹复制过来。如果你还没有下载 Palmer Penguins 数据集，请按照*第二章*《上传、下载与数据处理*中的*The setup: Palmer Penguins*部分的说明操作：

```py
mkdir penguin_ml
cp penguin_app/penguins.csv penguin_ml
cd penguin_ml
touch penguins_ml.py
touch penguins_streamlit.py 
```

正如你在前面的代码中看到的，这里有两个 Python 文件，一个用于创建机器学习模型（`penguins_ml.py`），另一个用于创建 Streamlit 应用（`penguins_streamlit.py`）。我们将从`penguins_ml.py`文件开始，一旦我们有了一个满意的模型，我们将继续处理`penguins_streamlit.py`文件。

你也可以选择在 Jupyter Notebook 中创建模型，尽管它的可复现性较差（因为单元格可以乱序运行），但它仍然非常流行。

让我们重新熟悉一下`penguins.csv`数据集。以下代码将读取数据集并打印出前五行：

```py
import pandas as pd
penguin_df = pd.read_csv('penguins.csv')
print(penguin_df.head()) 
```

当我们在终端中运行我们的 Python 文件`penguins_ml.py`时，前面的代码输出将类似于以下截图：

![图 4.1 – 前五只企鹅](img/B18444_04_01.png)

图 4.1：前五只企鹅

对于这个应用程序，我们将尝试创建一个帮助野外研究人员识别企鹅物种的应用。它将根据企鹅的喙、鳍和体重的测量数据，以及企鹅的性别和位置，预测企鹅的物种。

下一节并不是要尝试制作最好的机器学习模型，而只是为了创建一个快速原型供我们的 Streamlit 应用迭代。在这种情况下，我们将丢弃一些带有空值的行，并且不在特征中使用`year`变量，因为它与我们的用例不匹配。我们需要定义我们的特征和输出变量，对特征进行独热编码（或称*Pandas*中的虚拟变量处理），并对输出变量进行因子化（将其从字符串转换为数字）。以下代码将把我们的数据集转换成适合分类算法的状态：

```py
import pandas as pd
penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
      'flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)
print('Here are our output variables')
print(output.head())
print('Here are our feature variables')
print(features.head() 
```

现在，当我们再次运行 Python 文件`penguins_ml.py`时，我们会看到输出和特征变量已被分开，如下图所示：

![图 4.2 – 输出变量](img/B18444_04_02.png)

图 4.2：输出变量

现在，我们想使用数据的一个子集（在本例中为 80%）来创建一个分类模型，并获取该模型的准确度。以下代码通过使用随机森林模型来执行这些步骤，但如果你愿意，也可以使用其他分类算法。再次强调，这里主要是为了快速构建一个原型，向企鹅研究人员展示以便获取反馈！

```py
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                       'flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)
x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)
score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score)) 
```

现在我们已经有了一个相当不错的模型来预测企鹅的物种！在模型生成过程中的最后一步是保存我们最需要的两个部分——模型本身和 `uniques` 变量，它将因子化的输出变量映射到我们识别的物种名称。在之前的代码基础上，我们将添加几行代码，用于将这些对象保存为 pickle 文件（这是一种将 Python 对象转化为可直接保存并可以轻松从另一个 Python 文件（如我们的 Streamlit 应用）中导入的格式）。更具体地说，`open()` 函数创建了两个 pickle 文件，`pickle.dump()` 函数将我们的 Python 文件写入这些文件，而 `close()` 函数则关闭文件。`open()` 函数中的 `wb` 表示**写入字节**，它告诉 Python 我们要写入而不是读取该文件：

```py
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True)
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                       'flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)
x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)
score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close() 
```

现在我们在 `penguin_ml` 文件夹中有了两个文件：一个名为 `random_forest_penguin.pickle` 的文件，其中包含我们的模型，另一个是 `output_penguin_.pickle`，它包含企鹅物种与模型输出之间的映射。这就是 `penguins_ml.py` 函数的内容！接下来我们可以开始创建我们的 Streamlit 应用，使用我们刚刚创建的机器学习模型。

# 在 Streamlit 中使用预训练的机器学习模型

现在我们有了模型，我们想将它（以及我们的映射函数）加载到 Streamlit 中。在我们之前创建的文件 `penguins_streamlit.py` 中，我们将再次使用 `pickle` 库通过以下代码加载我们的文件。我们使用与之前相同的函数，但这次我们使用 `rb` 参数，而不是 `wb`，`rb` 表示**读取字节**。为了确保这些是我们之前使用的相同的 Python 对象，我们将使用我们非常熟悉的 `st.write()` 函数来进行检查：

```py
import streamlit as st
import pickle
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
st.write(rfc)
st.write(unique_penguin_mapping) 
```

与我们之前的 Streamlit 应用一样，我们在终端运行以下代码来启动我们的应用：

```py
streamlit run penguins_streamlit.py 
```

我们现在有了随机森林分类器，并且完成了企鹅的映射！接下来的步骤是添加 Streamlit 函数以获取用户输入。在我们的应用程序中，我们使用了岛屿、嘴长、嘴深、鳍肢长度、体重和性别来预测企鹅的物种，所以我们需要从用户那里获取这些信息。对于岛屿和性别，我们知道这些选项已经在我们的数据集中，并且希望避免解析用户输入的文本，因此我们将使用 `st.selectbox()`。对于其他数据，我们只需要确保用户输入的是一个正数，所以我们将使用 `st.number_input()` 函数并设置最小值为 `0`。以下代码会接收这些输入并在我们的 Streamlit 应用程序中显示它们：

```py
import pickle
import streamlit as st
rf_pickle = open("random_forest_penguin.pickle", "rb")
map_pickle = open("output_penguin.pickle", "rb")
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()
island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
sex = st.selectbox("Sex", options=["Female", "Male"])
bill_length = st.number_input("Bill Length (mm)", min_value=0)
bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
body_mass = st.number_input("Body Mass (g)", min_value=0)
user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]
st.write(f"""the user inputs are {user_inputs}""".format()) 
```

上述代码应当生成如下的应用程序。尝试一下，看看通过更改值，输出是否也发生变化。

Streamlit 设计时默认情况下，每次更改值时，整个应用程序都会重新运行。以下截图显示了应用程序的实时效果，并展示了我更改过的一些值。我们可以通过右侧的 **+** 和 **-** 按钮来更改数值，或者直接手动输入值：

![](img/B18444_04_03.png)

图 4.3：模型输入

现在我们已经准备好了所有输入和模型，下一步是将数据格式化为与我们预处理数据相同的格式。例如，我们的模型没有一个名为 `sex` 的变量，而是有两个名为 `sex_female` 和 `sex_male` 的变量。一旦数据的格式正确，我们就可以调用 `predict` 函数，并将预测结果映射到我们的原始物种列表中，以查看我们的模型如何工作。以下代码正是完成这一任务，同时还向应用程序添加了一些基本标题和说明，使其更易于使用。这个应用程序比较长，因此我会将其分成多个部分，以便阅读。我们首先为应用程序添加说明和标题：

```py
import streamlit as st
import pickle
st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using"
         "a model built on the Palmer Penguins dataset. Use the form below"
         " to get started!")
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close() 
```

我们现在已经有了一个带有标题和用户说明的应用程序。下一步是像之前一样获取用户输入。我们还需要将 `sex` 和 `island` 变量转换为正确的格式，如之前所述：

```py
island = st.selectbox('Penguin Island', options=[
                      'Biscoe', 'Dream', 'Torgerson'])
sex = st.selectbox('Sex', options=['Female', 'Male'])
bill_length = st.number_input('Bill Length (mm)', min_value=0)
bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
body_mass = st.number_input('Body Mass (g)', min_value=0)
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1
sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1 
```

我们的所有数据格式正确！最后一步是使用 `predict()` 函数处理我们的模型和新数据，这一部分代码完成了这一任务：

```py
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,
                               body_mass, island_biscoe, island_dream,
                               island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f"We predict your penguin is of the {prediction_species} species") 
```

现在我们的应用程序应该像下面的截图一样。

我已为输入添加了一些示例值，但你应该尝试更改数据，看看能否使物种发生变化！

![](img/B18444_04_04.png)

图 4.4：用于预测的完整 Streamlit 应用

我们现在拥有一个完整的 Streamlit 应用程序，利用我们预训练的机器学习模型，获取用户输入并输出预测结果。接下来，我们将讨论如何直接在 Streamlit 应用程序中训练模型！

# 在 Streamlit 应用程序中训练模型

经常情况下，我们可能希望用户输入的数据能够影响模型的训练。我们可能希望接受用户的数据，或者询问用户他们希望使用哪些特征，甚至允许用户选择他们希望使用的机器学习算法类型。所有这些选项在 Streamlit 中都是可行的，在本节中，我们将介绍如何使用用户输入来影响训练过程的基础知识。如上节所讨论的那样，如果模型只需要训练一次，最好在 Streamlit 外部训练模型，并将模型导入到 Streamlit 中。但如果在我们的示例中，企鹅研究人员将数据保存在本地，或者不知道如何重新训练模型，但已有正确格式的数据呢？在这种情况下，我们可以添加`st.file_uploader()`选项，并为这些用户提供一种方法，让他们输入自己的数据，并部署一个定制的模型，而无需编写任何代码。以下代码将添加一个用户选项来接受数据，并使用我们最初在`penguins_ml.py`中的预处理/训练代码为该用户创建一个独特的模型。需要注意的是，这只有在用户的数据格式与我们使用的完全相同时才会有效，这可能性较小。这里的另一个潜在补充是，向用户展示数据需要具备的格式，以便该应用能够正确地训练模型，按预期输出！

```py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.title('Penguin Classifier')
st.write(
    """This app uses 6 inputs to predict the species of penguin using
    a model built on the Palmer Penguins dataset. Use the form below
    to get started!"""
)
penguin_file = st.file_uploader('Upload your own penguin data') 
```

这一部分首先导入了我们需要的库，添加了标题——如我们之前所用——并添加了`file_uploader()`函数。然而，当用户还没有上传文件时会发生什么呢？如果没有企鹅文件，我们可以将默认设置为加载我们的随机森林模型，如下一个代码块所示：

```py
if penguin_file is None:
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close() 
```

接下来我们需要解决的问题是如何接收用户的数据，清理数据，并基于这些数据训练一个模型。幸运的是，我们可以重用已经创建的模型训练代码，并将其放入下一个代码块中的`else`语句中：

```py
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                           'flipper_length_mm', 'body_mass_g', 'sex']]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(
        features, output, test_size=.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values, y_train)
    y_pred = rfc.predict(x_test.values)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write(
        f"""We trained a Random Forest model on these
        data, it has a score of {score}! Use the
        inputs below to try out the model"""
    ) 
```

我们现在已经在应用中创建了模型，并且需要从用户处获取输入来进行预测。然而，这次我们可以在之前的基础上进行改进。到目前为止，每当用户在应用中更改输入时，整个 Streamlit 应用都会重新运行。我们可以使用`st.form()`和`st.submit_form_button()`函数将其余的用户输入包裹起来，并允许用户一次性更改所有输入并提交整个表单，而不是多次提交：

```py
with st.form('user_inputs'):
island = st.selectbox('Penguin Island', options=		['Biscoe', 'Dream', 'Torgerson'])
sex = st.selectbox('Sex', options=['Female', 'Male'])
bill_length = st.number_input('Bill Length (mm)', min_value=0)
bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
body_mass = st.number_input('Body Mass (g)', min_value=0)
st.form_submit_button()
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1
sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1 
```

现在我们有了新的表单输入，需要创建预测并将预测结果写入用户界面，如下一个代码块所示：

```py
new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male,
        ]
    ]
)
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f"We predict your penguin is of the {prediction_species} species") 
```

就这样！我们现在拥有一个 Streamlit 应用，允许用户输入自己的数据，基于这些数据训练模型，并输出结果，如下图所示：

![](img/B18444_04_05.png)

图 4.5：企鹅分类器应用

这里可以做一些潜在的改进，比如使用缓存函数（在*第二章*，*上传、下载和处理数据*中有介绍）作为一个例子。像这种让用户提供自己数据的应用，通常要比直接在应用内构建的应用更难开发，尤其是没有统一的数据格式。根据写作时的情况，更常见的是看到展示令人印象深刻的 ML 模型和用例的 Streamlit 应用，而不是直接在应用内构建这些模型的应用（尤其是当模型训练的计算开销较大时）。如前所述，Streamlit 开发者通常会在要求用户输入数据集之前提供所需数据格式的参考。然而，允许用户提供自己数据的选项仍然可用且实用，特别是为了快速迭代模型构建。

# 理解 ML 结果

到目前为止，我们的应用可能是有用的，但仅仅显示一个结果通常不足以满足数据应用的需求。我们应该展示一些对结果的解释。为了做到这一点，我们可以在我们已经创建的应用输出中包含一个部分，帮助用户更好地理解模型。

首先，随机森林模型已经内建了一个特征重要性方法，这个方法是从构成随机森林的每个单独决策树中得出的。我们可以编辑我们的`penguins_ml.py`文件来绘制这个特征重要性图，并从 Streamlit 应用中调用该图像。我们也可以直接在 Streamlit 应用中绘制这个图，但是一次性在`penguins_ml.py`中生成这个图比每次 Streamlit 应用重新加载时（即每次用户更改输入时）都生成图像更高效。以下代码编辑了我们的`penguins_ml.py`文件，添加了特征重要性图，并将其保存到我们的文件夹中。我们还调用了`tight_layout()`功能，帮助我们更好地格式化图表，确保不会有标签被截断。这段代码较长，文件的上半部分没有更改，因此只省略了库导入和数据清理部分。还有一点需要说明的是，我们将尝试使用其他图表库，如 Seaborn 和 Matplotlib，只是为了在使用的图形库上增加一些多样性。

```py
x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()
fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title('Which features are the most important for species prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png') 
```

现在，当我们重新运行`penguins_ml.py`时，我们应该能看到一个名为`feature_importance.png`的文件，我们可以从我们的 Streamlit 应用中调用这个文件。现在就让我们来做吧！我们可以使用`st.image()`函数从`.png`文件中加载图像，并将其打印到我们的企鹅应用中。以下代码将把我们的图像添加到 Streamlit 应用中，并改进我们对预测结果的解释。由于代码块较长，我们只展示从开始使用用户数据进行预测的代码部分：

```py
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,
                               body_mass, island_biscoe, island_dream,
                               island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
st.subheader("Predicting Your Penguin's Species:")
st.write(f"We predict your penguin is of the {prediction_species} species")
st.write(
    """We used a machine learning (Random Forest)
    model to predict the species, the features
    used in this prediction are ranked by 
    relative importance below."""
)
st.image('feature_importance.png') 
```

现在，您的 Streamlit 应用底部应该看起来像下面的截图（注意您的字符串可能会根据输入有所不同）：

![图 4.6 – 特征重要性截图](img/B18444_04_06.png)

图 4.6：特征重要性截图

如我们所见，鸟喙长度、鸟喙深度和鳍长是根据我们的随机森林模型得出的最重要变量。解释我们模型工作原理的另一个最终选项是绘制每个变量按物种分布的图表，并绘制一些表示用户输入的垂直线。理想情况下，用户可以开始从整体上理解底层数据，因此也能理解模型所做的预测。为了实现这一点，我们需要将数据实际导入到 Streamlit 应用中，这是我们之前没有做的。以下代码导入了我们用来构建模型的企鹅数据，并绘制了三个直方图（*鸟喙长度*、*鸟喙深度*和*鳍长*），同时将用户输入作为一条垂直线展示，从模型解释部分开始：

```py
st.subheader("Predicting Your Penguin's Species:")
st.write(f"We predict your penguin is of the {prediction_species} species")
st.write(
    """We used a machine learning (Random Forest)
    model to predict the species, the features
    used in this prediction are ranked by 
    relative importance below."""
)
st.image('feature_importance.png')
st.write(
    """Below are the histograms for each 
    continuous variable separated by penguin 
    species. The vertical line represents 
    your the inputted value."""
) 
```

现在我们已经设置好应用程序来显示直方图，我们可以使用 Seaborn 可视化库中的`displot()`函数来创建我们最重要变量的三个直方图：

```py
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper Length by Species')
st.pyplot(ax) 
```

上述代码应当生成如下图所示的应用程序，这是我们应用程序的最终形式。为了方便查看，我们只展示第一个直方图：

![](img/B18444_04_07.png)

图 4.7：按物种分类的鸟喙长度

如同往常一样，完整的最终代码可以在[`github.com/tylerjrichards/Streamlit-for-Data-Science`](https://github.com/tylerjrichards/Streamlit-for-Data-Science)找到。这部分到此为止。我们现在已经创建了一个完全构建的 Streamlit 应用程序，它可以接受一个预先构建的模型和用户输入，并输出预测结果以及对输出的解释。接下来，让我们探讨如何将你最喜爱的其他机器学习库集成到 Streamlit 中！

# 集成外部机器学习库——以 Hugging Face 为例

在过去几年里，由初创公司和机构创建的机器学习模型数量大幅增加。在我看来，有一个公司因其优先考虑开源和分享模型及方法而脱颖而出，那就是 Hugging Face。Hugging Face 使得使用一些领域内最优秀的研究人员创建的机器学习模型变得异常简单，你可以将这些模型应用到自己的用例中，在这一部分中，我们将快速展示如何将 Hugging Face 集成到 Streamlit 中。

在本书的初始设置中，我们已经下载了所需的两个库：PyTorch（最流行的深度学习 Python 框架）和 transformers（Hugging Face 的库，简化了使用其预训练模型的过程）。因此，对于我们的应用，让我们尝试进行自然语言处理中的一项基础任务：获取一段文本的情感！Hugging Face 通过其 pipeline 函数使这变得异常简单，该函数让我们按名称请求模型。接下来的代码片段会从用户那里获取文本输入，然后从 Hugging Face 检索情感分析模型：

```py
import streamlit as st
from transformers import pipeline

st.title("Hugging Face Demo")
text = st.text_input("Enter text to analyze")
model = pipeline("sentiment-analysis")
if text:
    result = model(text)
    st.write("Sentiment:", result[0]["label"])
    st.write("Confidence:", result[0]["score"]) 
```

当我们运行这个时，应该会看到如下结果。

![](img/B18444_04_08.png)

图 4.8：Hugging Face 演示

我在应用中放入了一个随机句子，但你可以随意尝试！试着给模型一些信心较低的文本（我试过“streamlit is a pizza pie”，并成功地让模型困惑）。想了解更多这里使用的模型，Hugging Face 提供了丰富的文档（[`huggingface.co/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)）。

当你玩这个应用时，你会发现应用经常加载较慢。这是因为每次运行应用时，transformers 库会从 Hugging Face 获取模型，并在应用中使用它。我们已经学习了如何缓存数据，但 Streamlit 有一个类似的缓存功能，叫做`st.cache_resource`，它可以让我们缓存像 ML 模型和数据库连接这样的对象。我们可以在这里使用它来加速应用：

```py
import streamlit as st
from transformers import pipeline
st.title("Hugging Face Demo")
text = st.text_input("Enter text to analyze")
@st.cache_resource()
def get_model():
    return pipeline("sentiment-analysis")
model = get_model()
if text:
    result = model(text)
    st.write("Sentiment:", result[0]["label"])
    st.write("Confidence:", result[0]["score"]) 
```

现在，我们的应用在多次使用时应该会运行得更快。这个应用并不完美，但它向我们展示了将一些顶尖库集成到 Streamlit 中的简便性。在本书的后续章节中，我们将讨论如何将 Streamlit 应用直接免费部署在 Hugging Face 上，但我鼓励你浏览 Hugging Face 网站（[`huggingface.co/`](https://huggingface.co/)），看看他们提供的所有资源。

# 集成外部 AI 库——OpenAI 示例

2023 年无疑是生成式 AI 的年份，ChatGPT 以其强大的影响力席卷了全球和开发者社区。像 ChatGPT 这样的服务背后生成模型的可用性也急剧增加，各大科技公司纷纷推出了自己的版本（例如 Meta 的[`ai.meta.com/llama/`](https://ai.meta.com/llama/)和 Google 的[`bard.google.com/`](https://bard.google.com/)）。其中最受欢迎的生成模型系列是 OpenAI 的**GPT**（**生成预训练变换器**）。本节将向你展示如何使用 OpenAI API 将生成式 AI 添加到你的 Streamlit 应用中！

## 使用 OpenAI 进行身份验证

我们的第一步是创建一个 OpenAI 账户并获取 API 密钥。为此，请访问[`platform.openai.com`](https://platform.openai.com)并创建一个账户。创建账户后，进入**API 密钥**部分（[`platform.openai.com/account/api-keys`](https://platform.openai.com/account/api-keys)），点击**创建新的秘密密钥**按钮。创建密钥后，务必将其保存在安全的地方，因为 OpenAI 不会再向你显示密钥！我将它保存在我的密码管理器中，以确保不会丢失（[`1password.com/`](https://1password.com/)），但你可以将它保存在任何你想要的地方。

## OpenAI API 费用

OpenAI API 不是免费的，但我们将使用的这一款（GPT-3.5 turbo）目前的费用为每千个 tokens（大约 750 个单词）输入$0.0015，输出每千个 tokens$0.002（有关最新信息，请参见[`openai.com/pricing`](https://openai.com/pricing)）。你也可以在[`platform.openai.com/account/billing/limits`](https://platform.openai.com/account/billing/limits)设置一个硬性限制，限制你在该 API 上的最大消费。如果设置了硬性限制，OpenAI 将不允许你超过此限制。我强烈建议设置限制。对这个示例部分设置 1 美元的限制；我们应该完全在这个范围内！一旦你开始创建并公开分享你自己的生成 AI 应用，这个功能将变得更加有用（通常，开发者要么要求用户输入自己的 API 密钥，要么通过像[`github.com/tylerjrichards/st-paywall`](https://github.com/tylerjrichards/st-paywall)这样的库收取访问 Streamlit 应用的费用，以避免支付过多）。

## Streamlit 和 OpenAI

在这个示例中，我们将重新创建我们 Hugging Face 示例中的情感分析，但使用的是 GPT-3.5 turbo。随着你尝试这些模型，你会发现它们通常非常智能，可以用来完成几乎所有你能想到的任务，而无需额外的训练。我来证明给你看！

现在我们有了 API，将其添加到一个 Secrets 文件中（我们将在*第五章*的*Streamlit Secrets*部分更详细地介绍 Secrets，*部署 Streamlit 到 Streamlit Community Cloud*）。创建一个名为`.streamlit`的文件夹，并在其中创建一个`secrets.toml`文件，然后将你的 API 密钥放入其中，并将其分配给名为`OPENAI_API_KEY`的变量，使其变成`OPENAI_API_KEY="sk-xxxxxxxxxxxx"`。

让我们打开现有的 Streamlit 应用，并在底部添加一个标题、一个按钮，让用户点击分析文本，以及我们的认证密钥：

```py
import openai
st.title("OpenAI Version")
analyze_button = st.button("Analyze Text")
openai.api_key = st.secrets["OPENAI_API_KEY"] 
```

OpenAI Python 库（我们通过初始的 `requirements.txt` 文件安装的）提供了一种方便的方式，以 Python 与 OpenAI API 进行交互，这真是一个非常有用的资源。我们要调用的端点叫做聊天完成端点 ([`platform.openai.com/docs/api-reference/chat/create`](https://platform.openai.com/docs/api-reference/chat/create))，它接受一个系统消息（这是我们告诉 OpenAI 模型如何响应的方式，在我们的案例中是一个有帮助的情感分析助手）以及关于我们要调用的底层模型的其他参数。虽然有比我们将要使用的模型更现代和昂贵的版本，但我发现 GPT 3.5 非常出色且速度很快。

我们可以像这样调用 API，并将响应写回到我们的应用程序：

```py
if analyze_button:
    messages = [
        {"role": "system", "content": """You are a helpful sentiment analysis assistant.
            You always respond with the sentiment of the text you are given and the confidence of your sentiment analysis with a number between 0 and 1"""},
        {"role": "user", 
    "content": f"Sentiment analysis of the following text: {text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    sentiment = response.choices[0].message['content'].strip()
    st.write(sentiment) 
```

让我们来测试一下！我们可以使用与 Hugging Face 示例中相同的文本输入来对比这两个分析器：

![A screenshot of a computer  Description automatically generated](img/B18444_04_09.png)

图 4.9：Hugging Face 和 OpenAI 情感分析器的比较

看起来两个版本都认为这个情感是积极的，且置信度相当高。这很了不起！Hugging Face 模型是专门为情感分析训练的，但 OpenAI 的模型并不是。对于这个简单的例子，它们似乎都能正常工作。如果我们尝试只给每个模型一个单词，比如 “Streamlit”，会怎么样呢？

![A screenshot of a computer  Description automatically generated](img/B18444_04_10.png)

图 4.10：测试“Streamlit”的情感

在这种情况下，两种方法得出的结论不同。OpenAI 认为这是中性情感，置信度适中，而 Hugging Face 认为情感是积极的，并且置信度非常高。我认为 OpenAI 在这里可能是对的，这真是令人着迷。显然，这种模型有着广泛的应用场景。

通过 Streamlit 小部件，我们可以让用户更改 API 调用的任何部分。我们只需添加正确的小部件类型并将用户的输入传递给 OpenAI 函数，然后就可以了！再试一次吧。如果我们让用户更改我们最初的系统消息会怎样？为此，我们需要添加一个新的文本输入。我们将使用一个叫做 `st.text_area` 的 Streamlit 输入小部件，它与我们熟悉的 `st.text_input` 相同，但允许多行输入，以便处理更长的文本段落：

```py
openai.api_key = st.secrets["OPENAI_API_KEY"]
system_message_default = """You are a helpful sentiment analysis assistant. You always respond with the sentiment of the text you are given and the confidence of your sentiment analysis with a number between 0 and 1"""

system_message = st.text_area(
    "Enter a System Message to instruct OpenAI", system_message_default
)
analyze_button = st.button("Analyze Text")
if analyze_button:
    messages = [
        {
            "role": "system",
            "content": f"{system_message}",
        },
        {
            "role": "user",
            "content": f"Sentiment analysis of the following text: {text}",
        },
    ] 
```

用户现在可以更改系统消息，但我们的默认消息保持不变。我已经将系统消息改成了一些荒谬的内容。我要求模型成为一个糟糕的情感分析助手，总是把输入的情感分析弄错：

![A screenshot of a computer  Description automatically generated](img/B18444_04_11.png)

图 4.11：更改 OpenAI 文本分析器的系统消息

如你所见，模型按照我要求的做法，错误地进行了 **streamlit is awesome** 的情感分析，结果显示情感是负面的。

快速警告：当你允许用户输入到大型语言模型中时，用户可能会尝试将不良的提示注入到你的应用程序中。这里有一个使用同一个应用程序的例子，我要求模型忽略所有其他指令，而改写一个海盗主题的故事：

![计算机截图 自动生成的描述](img/B18444_04_12.png)

图 4.12：OpenAI 与海盗

这个故事还继续了很多行，但你可以看到，越是给予用户更多控制输入的权力，就越有可能让他们以我没有预料到的方式使用我的应用程序。对此有许多创新的解决方法，包括将提示传递给另一个 API 调用，这次询问模型它是否认为提示不真诚，或是防止一些常见的注入，如“忽略之前的提示”。

也有像 Rebuff 这样的开源库（[`github.com/protectai/rebuff`](https://github.com/protectai/rebuff)），它们也非常有用！由于生成性 AI 领域发展极其迅速，我不太敢给出具体建议，但谨慎的原则和有意的用户输入应该是非常有帮助的。

如果你对更多生成性 AI 的 Streamlit 应用感兴趣，Streamlit 团队已经制作了一个网页，汇集了所有最新的信息和示例，网址是[`streamlit.io/generative-ai`](https://streamlit.io/generative-ai)。

# 总结

在本章中，我们学习了一些机器学习的基础知识：如何在 Streamlit 中使用预构建的机器学习模型，如何从 Streamlit 内部创建自己的模型，如何利用用户输入理解并迭代机器学习模型，甚至如何使用 Hugging Face 和 OpenAI 的模型。希望到本章结束时，你能对这些内容感到自如。接下来，我们将深入探讨如何使用 Streamlit 社区云部署 Streamlit 应用！

# 在 Discord 上了解更多

加入本书的 Discord 社区——在这里你可以分享反馈、向作者提问，并了解新版本的发布——请扫描下方的二维码：

[`packt.link/sl`](https://packt.link/sl)

![](img/QR_Code13440134443835796.png)
