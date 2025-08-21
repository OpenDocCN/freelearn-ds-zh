

# 第十二章：使用 FastAPI 创建高效的预测 API 端点

在上一章中，我们介绍了 Python 社区中广泛使用的最常见的数据科学技术和库。得益于这些工具，我们现在可以构建能够高效预测和分类数据的机器学习模型。当然，我们现在需要考虑一个便捷的接口，以便能够充分利用它们的智能。这样，微服务或前端应用程序就可以请求我们的模型进行预测，从而改善用户体验或业务运营。在本章中，我们将学习如何使用 FastAPI 实现这一点。

正如我们在本书中看到的，FastAPI 允许我们使用清晰而轻量的语法实现非常高效的 REST API。在本章中，你将学习如何以最有效的方式使用它们，以便处理成千上万的预测请求。为了帮助我们完成这项任务，我们将引入另一个库——Joblib，它提供了帮助我们序列化已训练模型和缓存预测结果的工具。

本章我们将涵盖以下主要内容：

+   使用 Joblib 持久化已训练的模型

+   实现高效的预测端点

+   使用 Joblib 缓存结果

# 技术要求

本章需要你设置一个 Python 虚拟环境，就像我们在*第一章*中设置的那样，*Python 开发环境* *设置*。

你可以在专门的 GitHub 仓库中找到本章的所有代码示例，地址为[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12)。

# 使用 Joblib 持久化已训练的模型

在上一章中，你学习了如何使用 scikit-learn 训练一个估计器。当构建这样的模型时，你可能会获得一个相当复杂的 Python 脚本来加载训练数据、进行预处理，并用最佳的参数集来训练模型。然而，在将模型部署到像 FastAPI 这样的 Web 应用程序时，你不希望在服务器启动时重复执行这些脚本和所有操作。相反，你需要一个现成的已训练模型表示，只需要加载并使用它即可。

这就是 Joblib 的作用。这个库旨在提供高效保存 Python 对象到磁盘的工具，例如大型数据数组或函数结果：这个操作通常被称为**持久化**。Joblib 已经是 scikit-learn 的一个依赖，因此我们甚至不需要安装它。实际上，scikit-learn 本身也在内部使用它来加载打包的示例数据集。

如我们所见，使用 Joblib 持久化已训练的模型只需要一行代码。

## 持久化已训练的模型

在这个示例中，我们使用了我们在*第十一章*中看到的 newsgroups 示例，*Python 中的数据科学入门*一节的*链式预处理器和估算器*部分。为了提醒一下，我们加载了`newsgroups`数据集中 20 个类别中的 4 个，并构建了一个模型，将新闻文章自动分类到这些类别中。一旦完成，我们将模型导出到一个名为`newsgroups_model.joblib`的文件中：

chapter12_dump_joblib.py

```py

# Make the pipelinemodel = make_pipeline(
           TfidfVectorizer(),
           MultinomialNB(),
)
# Train the model
model.fit(newsgroups_training.data, newsgroups_training.target)
# Serialize the model and the target names
model_file = "newsgroups_model.joblib"
model_targets_tuple = (model, newsgroups_training.target_names)
joblib.dump(model_targets_tuple, model_file)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_dump_joblib.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_dump_joblib.py)

如你所见，Joblib 提供了一个名为`dump`的函数，它仅需要两个参数：要保存的 Python 对象和文件路径。

请注意，我们并没有单独导出`model`变量：相反，我们将它和类别名称`target_names`一起封装在一个元组中。这使得我们能够在预测完成后检索类别的实际名称，而不必重新加载训练数据集。

如果你运行这个脚本，你会看到`newsgroups_model.joblib`文件已被创建：

```py

(venv) $ python chapter12/chapter12_dump_joblib.py$ ls -lh *.joblib
-rw-r--r--    1 fvoron    staff       3,0M 10 jan 08:27 newsgroups_model.joblib
```

注意，这个文件相当大：它超过了 3 MB！它存储了每个词在每个类别中的概率，这些概率是通过多项式朴素贝叶斯模型计算得到的。

这就是我们需要做的。这个文件现在包含了我们 Python 模型的静态表示，它将易于存储、共享和加载。现在，让我们学习如何加载它并检查我们是否可以对其进行预测。

## 加载导出的模型

现在我们已经有了导出的模型文件，让我们学习如何使用 Joblib 再次加载它，并检查一切是否正常工作。在下面的示例中，我们将加载位于`chapter12`目录下的 Joblib 导出文件，并进行预测：

chapter12_load_joblib.py

```py

import osimport joblib
from sklearn.pipeline import Pipeline
# Load the model
model_file = os.path.join(os.path.dirname(__file__), "newsgroups_model.joblib")
loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
model, targets = loaded_model
# Run a prediction
p = model.predict(["computer cpu memory ram"])
print(targets[p[0]])
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_load_joblib.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_load_joblib.py)

在这里，我们只需要调用 Joblib 的`load`函数，并将导出文件的有效路径传递给它。这个函数的结果是我们导出的相同 Python 对象。在这里，它是一个包含 scikit-learn 估算器和类别列表的元组。

注意，我们添加了一些类型提示：虽然这不是必须的，但它帮助 mypy 或你使用的任何 IDE 识别加载对象的类型，并受益于类型检查和自动补全功能。

最后，我们对模型进行了预测：它是一个真正的 scikit-learn 估算器，包含所有必要的训练参数。

就这样！如你所见，Joblib 的使用非常直接。尽管如此，它是一个重要工具，用于导出你的 scikit-learn 模型，并能够在外部服务中使用这些模型，而无需重复训练阶段。现在，我们可以在 FastAPI 项目中使用这些已保存的文件。

# 实现一个高效的预测端点

现在我们已经有了保存和加载机器学习模型的方法，是时候在 FastAPI 项目中使用它们了。正如你所看到的，如果你跟随本书的内容进行操作，实施过程应该不会太令你惊讶。实现的主要部分是类依赖，它将处理加载模型并进行预测。如果你需要复习类依赖的内容，可以查看*第五章*，*FastAPI 中的依赖注入*。

开始吧！我们的示例将基于上一节中提到的`newgroups`模型。我们将从展示如何实现类依赖开始，这将处理加载模型并进行预测：

chapter12_prediction_endpoint.py

```py

class PredictionInput(BaseModel):           text: str
class PredictionOutput(BaseModel):
           category: str
class NewsgroupsModel:
           model: Pipeline | None = None
           targets: list[str] | None = None
           def load_model(self) -> None:
                         """Loads the model"""
                         model_file = os.path.join(os.path.dirname(__file__), "newsgroups_model.joblib")
                         loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
                         model, targets = loaded_model
                         self.model = model
                         self.targets = targets
           async def predict(self, input: PredictionInput) -> PredictionOutput:
                         """Runs a prediction"""
                         if not self.model or not self.targets:
                                       raise RuntimeError("Model is not loaded")
                         prediction = self.model.predict([input.text])
                         category = self.targets[prediction[0]]
                         return PredictionOutput(category=category)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_prediction_endpoint.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_prediction_endpoint.py)

首先，我们定义了两个 Pydantic 模型：`PredictionInput`和`PredictionOutput`。按照纯 FastAPI 的理念，它们将帮助我们验证请求负载并返回结构化的 JSON 响应。在这里，作为输入，我们仅期望一个包含我们想要分类的文本的`text`属性。作为输出，我们期望一个包含预测类别的`category`属性。

这个代码片段中最有趣的部分是`NewsgroupsModel`类。它实现了两个方法：`load_model`和`predict`。

`load_model`方法使用 Joblib 加载模型，如我们在上一节中所见，并将模型和目标存储在类的属性中。因此，它们将可以在`predict`方法中使用。

另一方面，`predict`方法将被注入到路径操作函数中。如你所见，它直接接受`PredictionInput`，这个输入将由 FastAPI 注入。在这个方法中，我们进行预测，就像我们通常在 scikit-learn 中做的那样。我们返回一个`PredictionOutput`对象，包含我们预测的类别。

你可能已经注意到，首先我们在进行预测之前，会检查模型及其目标是否在类属性中被分配。当然，我们需要确保在进行预测之前，`load_model`已经被调用。你可能会想，为什么我们不把这个逻辑放在初始化函数`__init__`中，这样我们可以确保模型在类实例化时就加载。这种做法完全可行，但也会带来一些问题。正如我们所看到的，我们在 FastAPI 之后立即实例化了一个`NewsgroupsModel`实例，以便可以在路由中使用它。如果加载逻辑放在`__init__`中，那么每次我们从这个文件中导入一些变量（比如`app`实例）时，模型都会被加载，例如在单元测试中。在大多数情况下，这样会导致不必要的 I/O 操作和内存消耗。正如我们所看到的，最好使用 FastAPI 的生命周期处理器，在应用运行时加载模型。

以下摘录展示了其余的实现，并包含处理预测的 FastAPI 路由：

chapter12_prediction_endpoint.py

```py

newgroups_model = NewsgroupsModel()@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
           newgroups_model.load_model()
           yield
app = FastAPI(lifespan=lifespan)
@app.post("/prediction")
async def prediction(
           output: PredictionOutput = Depends(newgroups_model.predict),
) -> PredictionOutput:
           return output
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_prediction_endpoint.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_prediction_endpoint.py)

正如我们之前提到的，我们创建了一个`NewsgroupsModel`实例，以便将其注入到路径操作函数中。此外，我们正在实现一个生命周期处理器来调用`load_model`。通过这种方式，我们确保在应用程序启动时加载模型，并使其随时可用。

预测端点非常简单：正如你所看到的，我们直接依赖于`predict`方法，它会处理注入和验证负载。我们只需要返回输出即可。

就这样！再次感谢 FastAPI，让我们的生活变得更加轻松，它让我们能够编写简单且可读的代码，即使是面对复杂的任务。我们可以像往常一样使用 Uvicorn 来运行这个应用：

```py

(venv) $ uvicorn chapter12.chapter12_prediction_endpoint:app
```

现在，我们可以尝试使用 HTTPie 进行一些预测：

```py

$ http POST http://localhost:8000/prediction text="computer cpu memory ram"HTTP/1.1 200 OK
content-length: 36
content-type: application/json
date: Tue, 10 Jan 2023 07:37:22 GMT
server: uvicorn
{
           "category": "comp.sys.mac.hardware"
}
```

我们的机器学习分类器已经启动！为了进一步推动这一点，让我们看看如何使用 Joblib 实现一个简单的缓存机制。

# 使用 Joblib 缓存结果

如果你的模型需要一定时间才能进行预测，缓存结果可能会非常有用：如果某个特定输入的预测已经完成，那么返回我们保存在磁盘上的相同结果比再次运行计算更有意义。在本节中，我们将学习如何借助 Joblib 来实现这一点。

Joblib 为我们提供了一个非常方便且易于使用的工具，因此实现起来非常简单。主要的关注点是我们应该选择标准函数还是异步函数来实现端点和依赖关系。这样，我们可以更详细地解释 FastAPI 的一些技术细节。

我们将在前一节中提供的示例基础上进行构建。我们必须做的第一件事是初始化一个 Joblib 的`Memory`类，它是缓存函数结果的辅助工具。然后，我们可以为想要缓存的函数添加一个装饰器。你可以在以下示例中看到这一点：

chapter12_caching.py

```py

memory = joblib.Memory(location="cache.joblib")@memory.cache(ignore=["model"])
def predict(model: Pipeline, text: str) -> int:
           prediction = model.predict([text])
           return prediction[0]
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_caching.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_caching.py)

在初始化`memory`时，主要参数是`location`，它是 Joblib 存储结果的目录路径。Joblib 会自动将缓存结果保存在硬盘上。

然后，你可以看到我们实现了一个`predict`函数，它接受我们的 scikit-learn 模型和一些文本输入，并返回预测的类别索引。这与我们之前看到的预测操作相同。在这里，我们将其从`NewsgroupsModel`依赖类中提取出来，因为 Joblib 缓存主要是为了与常规函数一起使用的。缓存类方法并不推荐。正如你所看到的，我们只需在这个函数上方添加一个`@memory.cache`装饰器来启用 Joblib 缓存。

每当这个函数被调用时，Joblib 会检查它是否已经有相同参数的结果保存在磁盘上。如果有，它会直接返回该结果。否则，它会继续进行常规的函数调用。

正如你所看到的，我们为装饰器添加了一个`ignore`参数，这允许我们告诉 Joblib 在缓存机制中忽略某些参数。这里，我们排除了`model`参数。Joblib 无法存储复杂对象，比如 scikit-learn 估算器。但这不是问题：因为模型在多个预测之间是不会变化的，所以我们不关心是否缓存它。如果我们对模型进行了改进并部署了一个新的模型，我们只需要清除整个缓存，这样较早的预测就会使用新的模型重新计算。

现在，我们可以调整`NewsgroupsModel`依赖类，使其与这个新的`predict`函数兼容。你可以在以下示例中看到这一点：

chapter12_caching.py

```py

class NewsgroupsModel:           model: Pipeline | None = None
           targets: list[str] | None = None
           def load_model(self) -> None:
                         """Loads the model"""
                         model_file = os.path.join(os.path.dirname(__file__), "newsgroups_model.joblib")
                         loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
                         model, targets = loaded_model
                         self.model = model
                         self.targets = targets
           def predict(self, input: PredictionInput) -> PredictionOutput:
                         """Runs a prediction"""
                         if not self.model or not self.targets:
                                       raise RuntimeError("Model is not loaded")
                         prediction = predict(self.model, input.text)
                         category = self.targets[prediction]
                         return PredictionOutput(category=category)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_caching.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_caching.py)

在`predict`方法中，我们调用了外部的`predict`函数，而不是直接在方法内部调用，并且注意将模型和输入文本作为参数传递。之后我们只需要做的就是获取对应的类别名称并构建一个`PredictionOutput`对象。

最后，我们有了 REST API 端点。在这里，我们添加了一个`delete/cache`路由，以便通过 HTTP 请求清除整个 Joblib 缓存。你可以在以下示例中看到这一点：

chapter12_caching.py

```py

@app.post("/prediction")def prediction(
           output: PredictionOutput = Depends(newgroups_model.predict),
) -> PredictionOutput:
           return output
@app.delete("/cache", status_code=status.HTTP_204_NO_CONTENT)
def delete_cache():
           memory.clear()
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_caching.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_caching.py)

`memory`对象上的`clear`方法会删除硬盘上所有的 Joblib 缓存文件。

我们的 FastAPI 应用程序现在正在缓存预测结果。如果你使用相同的输入发出两次请求，第二次的响应将显示缓存结果。在这个例子中，我们的模型非常快，所以你不会注意到执行时间上的差异；然而，对于更复杂的模型，这可能会变得很有趣。

## 选择标准函数或异步函数

你可能注意到我们已经修改了`predict`方法以及`prediction`和`delete_cache`路径操作函数，使它们成为*标准的*、*非异步的*函数。

自从本书开始以来，我们已经向你展示了 FastAPI 如何完全拥抱异步 I/O，以及这对应用程序性能的好处。我们还推荐了能够异步工作的库，例如数据库驱动程序，以便利用这一优势。

然而，在某些情况下，这是不可能的。在这种情况下，Joblib 被实现为同步工作。然而，它执行的是长时间的 I/O 操作：它在硬盘上读取和写入缓存文件。因此，它会阻塞进程，在此期间无法响应其他请求，正如我们在*第二章*的*异步 I/O*部分中所解释的那样，*Python* *编程特性*。

为了解决这个问题，FastAPI 实现了一个巧妙的机制：*如果你将路径操作函数或依赖项定义为标准的、非异步的函数，它将在一个单独的线程中运行*。这意味着阻塞操作，例如同步文件读取，不会阻塞主进程。从某种意义上说，我们可以说它模仿了一个异步操作。

为了理解这一点，我们将进行一个简单的实验。在以下示例中，我们构建了一个包含三个端点的虚拟 FastAPI 应用程序：

+   `/fast`，它直接返回响应

+   `/slow-async`，一个定义为`async`的路径操作，创建一个同步阻塞操作，需要 10 秒钟才能完成

+   `/slow-sync`，一个作为标准方法定义的路径操作，创建一个同步阻塞操作，需要 10 秒钟才能完成

你可以在这里查看相应的代码：

chapter12_async_not_async.py

```py

import timefrom fastapi import FastAPI
app = FastAPI()
@app.get("/fast")
async def fast():
           return {"endpoint": "fast"}
@app.get("/slow-async")
async def slow_async():
           """Runs in the main process"""
           time.sleep(10)    # Blocking sync operation
           return {"endpoint": "slow-async"}
@app.get("/slow-sync")
def slow_sync():
           """Runs in a thread"""
           time.sleep(10)    # Blocking sync operation
           return {"endpoint": "slow-sync"}
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_async_not_async.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter12/chapter12_async_not_async.py)

通过这个简单的应用程序，我们的目标是观察那些阻塞操作如何阻塞主进程。让我们使用 Uvicorn 运行这个应用程序：

```py

(venv) $ uvicorn chapter12.chapter12_async_not_async:app
```

接下来，打开两个新的终端。在第一个终端中，向`/``slow-async`端点发出请求：

```py

$ http GET http://localhost:8000/slow-async
```

在没有等待响应的情况下，在第二个终端向`/``fast`端点发出请求：

```py

$ http GET http://localhost:8000/fast
```

您会看到，您必须等待 10 秒钟才能收到`/fast`端点的响应。这意味着`/slow-async`阻塞了进程，导致服务器无法在此期间响应其他请求。

现在，让我们在`/``slow-sync`端点进行相同的实验：

```py

$ http GET http://localhost:8000/slow-sync
```

再次运行以下命令：

```py

$ http GET http://localhost:8000/fast
```

您将立即收到`/fast`的响应，而无需等待`/slow-sync`完成。由于它被定义为标准的非异步函数，FastAPI 会在一个线程中运行它，以避免阻塞。但是请记住，将任务发送到单独的线程会带来一些开销，因此在解决当前问题时，需要考虑最佳的处理方式。

那么，在使用 FastAPI 进行开发时，如何在路径操作和依赖之间选择标准函数和异步函数呢？以下是一些经验法则：

+   如果函数不涉及长时间的 I/O 操作（例如文件读取、网络请求等），请将其定义为`async`。

+   如果涉及 I/O 操作，请参考以下内容：

    +   尝试选择与异步 I/O 兼容的库，正如我们在数据库或 HTTP 客户端中看到的那样。在这种情况下，您的函数将是`async`。

    +   如果不可能，如 Joblib 缓存的情况，请将它们定义为标准函数。FastAPI 将会在单独的线程中运行它们。

由于 Joblib 在进行 I/O 操作时是完全同步的，我们将路径操作和依赖方法切换为同步的标准方法。

在这个示例中，差异不太明显，因为 I/O 操作较小且快速。然而，如果您需要实现更慢的操作，例如将文件上传到云存储时，记得考虑这个问题。

# 总结

恭喜！您现在可以构建一个快速高效的 REST API 来服务您的机器学习模型。感谢 Joblib，您学会了如何将训练好的 scikit-learn 估计器保存到一个文件中，以便轻松加载并在您的应用程序中使用。我们还展示了使用 Joblib 缓存预测结果的方法。最后，我们讨论了 FastAPI 如何通过将同步操作发送到单独的线程来处理，以避免阻塞。虽然这有点技术性，但在处理阻塞 I/O 操作时，牢记这一点是很重要的。

我们的 FastAPI 之旅接近尾声。在让您独立构建令人惊叹的数据科学应用之前，我们将提供三个章节，进一步推动这一进程并研究更复杂的使用案例。我们将从一个可以执行实时物体检测的应用开始，得益于 WebSocket 和计算机视觉模型。
