

# 第十四章：使用 Stable Diffusion 模型创建分布式文本到图像 AI 系统

到目前为止，在这本书中，我们构建的 API 中所有操作都是在请求处理内部计算的。换句话说，用户必须等待服务器完成我们定义的所有操作（如请求验证、数据库查询、ML 预测等），才能收到他们的响应。然而，并非总是希望或可能要求这种行为。

典型例子是电子邮件通知。在 Web 应用程序中，我们经常需要向用户发送电子邮件，因为他们刚刚注册或执行了特定操作。为了做到这一点，服务器需要向电子邮件服务器发送请求，以便发送电子邮件。此操作可能需要几毫秒时间。如果我们在请求处理中执行此操作，响应将延迟直到我们发送电子邮件。这不是一个很好的体验，因为用户并不真正关心电子邮件是如何何时发送的。这个例子是我们通常所说的**后台操作**的典型例子：需要在我们的应用程序中完成的事情，但不需要直接用户交互。

另一种情况是当用户请求一个耗时的操作，在合理的时间内无法完成。这通常是复杂数据导出或重型 AI 模型的情况。在这种情况下，用户希望直接获取结果，但如果在请求处理程序中执行此操作，将会阻塞服务器进程，直到完成。如果大量用户请求这种操作，会迅速使我们的服务器无响应。此外，某些网络基础设施，如代理或 Web 客户端（如浏览器），具有非常严格的超时设置，这意味着如果响应时间过长，它们通常会取消操作。

为了解决这个问题，我们将引入一个典型的 Web 应用程序架构：**web-queue-worker**。正如我们将在本章中看到的，我们将把最昂贵、耗时最长的操作推迟到后台进程，即**worker**。为了展示这种架构的运行方式，我们将建立我们自己的 AI 系统，使用**Stable Diffusion**模型根据文本提示生成图像。

本章中，我们将涵盖以下主要话题：

+   使用 Stable Diffusion 模型与 Hugging Face Diffusers 生成图像的文本提示

+   使用 Dramatiq 实现工作进程和图像生成任务

+   存储和服务于对象存储中的文件

# 技术要求

对于本章，您将需要一个 Python 虚拟环境，就像我们在*第一章*中设置的那样，*Python 开发环境设置*。

为了正确运行 Stable Diffusion 模型，我们建议你使用配备至少 16 GB RAM 的最新计算机，理想情况下还应配备 8 GB VRAM 的专用 GPU。对于 Mac 用户，配备 M1 Pro 或 M2 Pro 芯片的最新型号也非常适合。如果你没有这种机器，也不用担心：我们会告诉你如何以其他方式运行系统——唯一的缺点是图像生成会变慢并且效果较差。

要运行工作程序，你需要在本地计算机上运行**Redis 服务器**。最简单的方法是将其作为 Docker 容器运行。如果你以前从未使用过 Docker，我们建议你阅读官方文档中的*入门教程*，网址为[`docs.docker.com/get-started/`](https://docs.docker.com/get-started/)。完成后，你将能够通过以下简单命令运行 Redis 服务器：

```py

$ docker run -d --name worker-redis -p 6379:6379 redis
```

你可以在专用的 GitHub 仓库中找到本章的所有代码示例，地址为[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14)。

# 使用 Stable Diffusion 从文本提示生成图像

最近，一种新一代的 AI 工具引起了全世界的关注：图像生成模型，例如 DALL-E 或 Midjourney。这些模型是在大量图像数据上进行训练的，能够从简单的文本提示中生成全新的图像。这些 AI 模型非常适合作为后台工作程序：它们的处理时间为几秒钟甚至几分钟，并且需要大量的 CPU、RAM 甚至 GPU 资源。

为了构建我们的系统，我们将依赖于 Stable Diffusion，这是一种非常流行的图像生成模型，发布于 2022 年。该模型是公开的，可以在现代游戏计算机上运行。正如我们在上一章中所做的，我们将依赖 Hugging Face 工具来下载和运行该模型。

首先，让我们安装所需的工具：

```py

(venv) $ pip install accelerate diffusers
```

现在，我们已经准备好通过 Hugging Face 使用扩散模型了。

## 在 Python 脚本中实现模型

在下面的示例中，我们将展示一个能够实例化模型并运行图像生成的类的实现。再次提醒，我们将应用懒加载模式，使用单独的 `load_model` 和 `generate` 方法。首先，让我们专注于 `load_model`：

text_to_image.py

```py

class TextToImage:    pipe: StableDiffusionPipeline | None = None
    def load_model(self) -> None:
        # Enable CUDA GPU
        if torch.cuda.is_available():
            device = "cuda"
        # Enable Apple Silicon (M1) GPU
        elif torch.backends.mps.is_available():
            device = "mps"
        # Fallback to CPU
        else:
            device = "cpu"
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        pipe.to(device)
        self.pipe = pipe
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/text_to_image.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/text_to_image.py)

该方法的第一部分旨在根据你的计算机找到最有效的运行模型的方式。当在 GPU 上运行时，这些扩散模型的速度更快——这就是为什么我们首先检查是否有 CUDA（NVIDIA GPU）或 MPS（Apple Silicon）设备可用。如果没有，我们将退回到 CPU。

然后，我们只需创建一个由 Hugging Face 提供的`StableDiffusionPipeline`管道。我们只需要设置我们想要从 Hub 下载的模型。对于这个例子，我们选择了`runwayml/stable-diffusion-v1-5`。你可以在 Hugging Face 上找到它的详细信息：[`huggingface.co/runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)。

我们现在可以专注于`generate`方法：

text_to_image.py

```py

    def generate(        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        num_steps: int = 50,
        callback: Callable[[int, int, torch.FloatTensor], None] | None = None,
    )    Image.Image:
        if not self.pipe:
            raise RuntimeError("Pipeline is not loaded")
        return self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=9.0,
            callback=callback,
        ).images[0]
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/text_to_image.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/text_to_image.py)

你可以看到它接受四个参数：

+   `prompt`，当然，这是描述我们想要生成的图像的文本提示。

+   `negative_prompt`，这是一个可选的提示，用于告诉模型我们绝对不希望出现的内容。

+   `num_steps`，即模型应执行的推理步骤数。更多步骤会导致更好的图像，但每次迭代都会延迟推理。默认值`50`应该在速度和质量之间提供良好的平衡。

+   `callback`，这是一个可选的函数，它将在每次迭代步骤中被调用。这对于了解生成进度并可能执行更多逻辑（如将进度保存到数据库中）非常有用。

方法签名中的星号（*）是什么意思？

你可能已经注意到方法签名中的星号（`*`）。它告诉 Python，星号后面的参数应该仅作为关键字参数处理。换句话说，你只能像这样调用它们：`.generate("PROMPT",` `negative_prompt="NEGATIVE", num_steps=10)`。

尽管不是必须的，但这是一种保持函数清晰且自解释的方式。如果你开发的是供其他开发者使用的类或函数，这尤其重要。

还有一种语法可以强制参数仅作为位置参数传递，方法是使用斜杠（`/`）符号。你可以在这里阅读更多相关内容：[`docs.python.org/3/whatsnew/3.8.html#positional-only-parameters`](https://docs.python.org/3/whatsnew/3.8.html#positional-only-parameters)。

然后，我们只需要将这些参数传递给`pipe`。如果需要的话，还有更多的参数可以调节，但默认的参数应该会给你不错的结果。你可以在 Hugging Face 文档中找到完整的参数列表：[`huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__)。这个`pipe`对象能够为每个提示生成多张图像，因此该操作的结果是一个 Pillow 图像列表。这里的默认行为是生成一张图像，所以我们直接返回第一张。

就这些！再次感谢 Hugging Face，通过允许我们在几十行代码内运行最前沿的模型，真的是让我们的生活变得更轻松！

## 执行 Python 脚本

我们敢打赌你急于自己试一试——所以我们在示例的底部添加了一个小的`main`脚本：

text_to_image.py

```py

if __name__ == "__main__":    text_to_image = TextToImage()
    text_to_image.load_model()
    def callback(step: int, _timestep, _tensor):
        print(f"🚀 Step {step}")
    image = text_to_image.generate(
        "A Renaissance castle in the Loire Valley",
        negative_prompt="low quality, ugly",
        callback=callback,
    )
    image.save("output.png")
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/text_to_image.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/text_to_image.py)

这个小脚本实例化了我们的`TextToImage`类，加载了模型，并在保存到磁盘之前生成了图像。我们还定义了一个虚拟回调函数，让你能看到它是如何工作的。

当你第一次运行这个脚本时，你会注意到 Hugging Face 会将几个 GB 的文件下载到你的计算机上：那就是稳定扩散模型，确实相当庞大！

然后，推理开始了。你会看到一个进度条，显示剩余的推理步骤数，并显示我们回调函数中的`print`语句，如*图 14.1*所示。

![图 14.1 – 稳定扩散生成图像](img/Figure_14.1_B19528.jpg)

图 14.1 – 稳定扩散生成图像

生成一张图像需要多长时间？

我们在不同类型的计算机上进行了多次测试。在配备 8 GB RAM 的现代 NVIDIA GPU 或 M1 Pro 芯片的 Mac 上，模型能够在*大约一分钟*内生成一张图像，并且内存使用合理。而在 CPU 上运行时，大约需要*5 到 10 分钟*，并且会占用多达 16 GB 的内存。

如果你的计算机上推理速度确实太慢，你可以尝试减少`num_steps`参数。

当推理完成后，你会在磁盘上找到生成的图像和你的脚本。*图 14.2*展示了这种结果的一个例子。不错吧？

![图 14.2 – 稳定扩散图像生成结果](img/Figure_14.2_B19528.jpg)

图 14.2 – 稳定扩散图像生成结果

现在，我们已经拥有了我们 AI 系统的基础构件。接下来，我们需要构建一个 API，供用户生成自己的图像。正如我们刚刚看到的，生成一张图像需要一些时间。正如我们在介绍中所说的，我们需要引入一个 Web 队列工作进程架构，使得这个系统既可靠又具有可扩展性。

# 创建 Dramatiq 工作进程并定义图像生成任务

正如我们在本章的介绍中提到的，直接在我们的 REST API 服务器上运行图像生成模型是不可行的。正如我们在上一节所见，这一操作可能需要几分钟，并消耗大量内存。为了解决这个问题，我们将定义一个独立于服务器进程的其他进程来处理图像生成任务：**工作进程**。本质上，工作进程可以是任何一个在后台执行任务的程序。

在 Web 开发中，这个概念通常意味着比这更多的内容。工作进程是一个持续运行在后台的进程，等待接收任务。这些任务通常由 Web 服务器发送，服务器会根据用户的操作请求执行特定的操作。

因此，我们可以看到，我们需要一个通信通道来连接 Web 服务器和工作进程。这就是**队列**的作用。队列会接收并堆积来自 Web 服务器的消息，然后将这些消息提供给工作进程读取。这就是 Web 队列工作进程架构。为了更好地理解这一点，*图 14.4* 展示了这种架构的示意图。

![图 14.3 – Web 队列工作进程架构示意图](img/Figure_14.3_B19528.jpg)

图 14.3 – Web 队列工作进程架构示意图

这是不是让你想起了什么？是的，这与我们在*第八章*中看到的非常相似，在*处理多个 WebSocket 连接并广播消息*这一节。实际上，这是同一个原理：我们通过一个中央数据源来解决有多个进程的问题。

这种架构的一个伟大特性是它非常容易扩展。试想你的应用程序取得了巨大成功，成千上万的用户想要生成图像：单个工作进程根本无法满足这种需求。事实上，我们所需要做的就是启动更多的工作进程。由于架构中有一个单独的消息代理，每个工作进程会在收到消息时进行拉取，从而实现任务的并行处理。它们甚至不需要位于同一台物理机器上。*图 14.4* 展示了这一点。

![图 14.4 – 带有多个工作进程的 Web 队列工作进程架构](img/Figure_14.4_B19528.jpg)

图 14.4 – 带有多个工作进程的 Web 队列工作进程架构

在 Python 中，有多种库可以帮助实现工作进程。它们提供了定义任务、将任务调度到队列中并运行进程、拉取并执行任务所需的工具。在本书中，我们将使用 Dramatiq，一个轻量级但强大且现代的后台任务处理库。正如我们在 *第八章* 中所做的，我们将使用 Redis 作为消息代理。

## 实现一个工作进程

和往常一样，我们首先安装所需的依赖项。运行以下命令：

```py

(venv) $ pip install "dramatiq[redis]"
```

这将安装 Dramatiq，并安装与 Redis 代理通信所需的依赖项。

在一个最小的示例中，设置 Dramatiq 工作进程涉及两件事：

1.  设置代理类型和 URL。

1.  通过使用 `@``dramatiq.actor` 装饰器来定义任务。

它非常适合绝大多数任务，比如发送电子邮件或生成导出文件。

然而，在我们的案例中，我们需要加载庞大的 Stable Diffusion 模型。正如我们通常在 FastAPI 服务器中通过 `startup` 事件做的那样，我们希望只有在进程实际启动时才执行这一操作。为了使用 Dramatiq 实现这一点，我们需要实现一个*中间件*。它们允许我们在工作进程生命周期中的几个关键事件插入自定义逻辑，包括当工作进程启动时。

你可以在以下示例中看到我们自定义中间件的实现：

worker.py

```py

class TextToImageMiddleware(Middleware):    def __init__(self) -> None:
        super().__init__()
        self.text_to_image = TextToImage()
    def after_process_boot(self, broker):
        self.text_to_image.load_model()
        return super().after_process_boot(broker)
text_to_image_middleware = TextToImageMiddleware()
redis_broker = RedisBroker(host="localhost")
redis_broker.add_middleware(text_to_image_middleware)
dramatiq.set_broker(redis_broker)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/worker.py)

我们定义了一个 `TextToImageMiddleware` 类，它的作用是承载 `TextToImage` 的实例，这是我们在上一节中定义的图像生成服务。它继承自 Dramatiq 的 `Middleware` 类。这里的关键是 `after_process_boot` 方法。它是 Dramatiq 提供的事件钩子之一，允许我们插入自定义逻辑。在这里，我们告诉它在工作进程启动后加载 Stable Diffusion 模型。你可以在官方文档中查看支持的钩子列表：[`dramatiq.io/reference.html#middleware`](https://dramatiq.io/reference.html#middleware)。

接下来的几行代码让我们可以配置我们的工作进程。我们首先实例化我们自定义中间件的一个实例。然后，我们创建一个与我们选择的技术相对应的代理类；在我们的案例中是 Redis。在告诉 Dramatiq 使用它之前，我们需要将中间件添加到这个代理中。我们的工作进程现在已经完全配置好，可以连接到 Redis 代理，并在启动时加载我们的模型。

现在，让我们来看一下如何定义一个任务来生成图像：

worker.py

```py

@dramatiq.actor()def text_to_image_task(
    prompt: str, *, negative_prompt: str | None = None, num_steps: int = 50
):
    image = text_to_image_middleware.text_to_image.generate(
        prompt, negative_prompt=negative_prompt, num_steps=num_steps
    )
    image.save(f"{uuid.uuid4()}.png")
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/worker.py)

实现是直接的：Dramatiq 任务实际上是我们用 `@dramatiq.actor` 装饰的普通函数。我们可以像定义其他函数一样定义参数。然而，这里有一个重要的陷阱需要避免：当我们从服务器调度任务时，参数将必须存储在队列存储中。因此，*Dramatiq 会将参数内部序列化为 JSON*。这意味着你的任务参数必须是可序列化的数据——你不能有任意的 Python 对象，比如类实例或函数。

函数体在将图像保存到磁盘之前，会调用我们在 `text_to_image_middleware` 中加载的 `TextToImage` 实例。为了避免文件覆盖，我们选择在这里生成一个**UUID**，即**通用唯一标识符**。它是一个大的随机字符串，保证每次生成时都是唯一的。凭借这个，我们可以安全地将其作为文件名，并确保它不会在磁盘上已存在。

这就是 worker 实现的内容。

### 启动 worker

我们还没有代码来调用它，但我们可以手动尝试。首先，确保你已经启动了一个 Redis 服务器，正如在*技术要求*部分中所解释的那样。然后，我们可以使用以下命令启动 Dramatiq worker：

```py

(venv) $ dramatiq -p 1 -t 1 chapter14.basic.worker
```

Dramatiq 提供了命令行工具来启动 worker 进程。主要的位置参数是 worker 模块的点路径。这类似于我们在使用 Uvicorn 时的操作。我们还设置了两个可选参数，`-p` 和 `-t`。它们控制 Dramatiq 启动的进程和线程的数量。默认情况下，它启动 10 个进程，每个进程有 8 个线程。这意味着将会有 80 个 worker 来拉取并执行任务。虽然这个默认配置适合常见需求，但由于两个原因，它不适用于我们的 Stable Diffusion 模型：

+   进程中的每个线程共享相同的内存空间。这意味着，如果两个（或更多）线程尝试生成图像，它们将对内存中的同一对象进行读写操作。对于我们的模型来说，这会导致并发问题。我们说它是*非线程安全的*。因此，每个进程应该仅启动一个线程：这就是`-t` `1`选项的意义所在。

+   每个进程都应该将模型加载到内存中。这意味着，如果我们启动 8 个进程，我们将加载 8 次模型。正如我们之前所看到的，它需要相当大的内存，所以这样做可能会使你的计算机内存爆炸。为了安全起见，我们仅启动一个进程，使用`-p 1`选项。如果你想尝试并行化并查看我们的 worker 能否并行生成两张图像，你可以尝试`-p 2`来启动两个进程。但要确保你的计算机能够处理！

如果你运行前面的命令，你应该会看到类似这样的输出：

```py

[2023-02-02 08:52:11,479] [PID 44348] [MainThread] [dramatiq.MainProcess] [INFO] Dramatiq '1.13.0' is booting up.Fetching 19 files:   0%|          | 0/19 [00:00<?, ?it/s]
Fetching 19 files: 100%|██████████| 19/19 [00:00<00:00, 13990.83it/s]
[2023-02-02 08:52:11,477] [PID 44350] [MainThread] [dramatiq.WorkerProcess(0)] [INFO] Worker process is ready for action.
[2023-02-02 08:52:11,578] [PID 44355] [MainThread] [dramatiq.ForkProcess(0)] [INFO] Fork process 'dramatiq.middleware.prometheus:_run_exposition_server' is ready for action.
```

你可以通过查看 Stable Diffusion 流水线的输出，检查模型文件是否已经下载，直到 worker 完全启动。这意味着它已经正确加载。

### 在 worker 中调度任务

现在我们可以尝试在工作线程中调度任务了。为此，我们可以启动一个 Python 交互式 Shell 并导入`task`函数。打开一个新的命令行并运行以下命令（确保你已启用 Python 虚拟环境）：

```py

(venv) $ python>>> from chapter14.basic.worker import text_to_image_task
>>> text_to_image_task.send("A Renaissance castle in the Loire Valley")
Message(queue_name='default', actor_name='text_to_image_task', args=('A Renaissance castle in the Loire Valley',), kwargs={}, options={'redis_message_id': '663df44a-cfc1-4f13-8457-05d8181290c1'}, message_id='bf57d112-6c20-49bc-a926-682ca43ea7ea', message_timestamp=1675324585644)
```

就是这样——我们在工作线程中安排了一个任务！注意我们在`task`函数上使用了`send`方法，而不是直接调用它：这是告诉 Dramatiq 将其发送到队列中的方式。

如果你回到工作线程终端，你会看到 Stable Diffusion 正在生成图像。过一会儿，你的图像将保存在磁盘上。你还可以尝试在短时间内连续发送两个任务。你会发现 Dramatiq 会一个接一个地处理它们。

干得好！我们的后台进程已经准备好，甚至能够在其中调度任务。下一步就是实现 REST API，以便用户可以自己请求图像生成。

## 实现 REST API

要在工作线程中调度任务，我们需要一个用户可以交互的安全接口。REST API 是一个不错的选择，因为它可以轻松集成到任何软件中，如网站或移动应用。在这一节中，我们将快速回顾一下我们实现的简单 API 端点，用于将图像生成任务发送到队列中。以下是实现代码：

api.py

```py

class ImageGenerationInput(BaseModel):    prompt: str
    negative_prompt: str | None
    num_steps: int = Field(50, gt=0, le=50)
class ImageGenerationOutput(BaseModel):
    task_id: UUID4
app = FastAPI()
@app.post(
    "/image-generation",
    response_model=ImageGenerationOutput,
    status_code=status.HTTP_202_ACCEPTED,
)
async def post_image_generation(input: ImageGenerationInput) -> ImageGenerationOutput:
    task: Message = text_to_image_task.send(
        input.prompt, negative_prompt=input.negative_prompt, num_steps=input.num_steps
    )
    return ImageGenerationOutput(task_id=task.message_id)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/api.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/basic/api.py)

如果你从这本书的开头一直跟到现在，这不应该让你感到惊讶。我们已经妥善地定义了合适的 Pydantic 模型来构建和验证端点负载。然后，这些数据会直接用于发送任务到 Dramatiq，正如我们在前一节看到的那样。

在这个简单的实现中，输出仅包含消息 ID，Dramatiq 会自动为每个任务分配这个 ID。注意我们将 HTTP 状态码设置为`202`，表示*已接受*。从语义上讲，这意味着服务器已理解并接受了请求，但处理尚未完成，甚至可能还没有开始。它专门用于处理在后台进行的情况，这正是我们在这里的情况。

如果你同时启动工作线程和这个 API，你将能够通过 HTTP 调用触发图像生成。

你可能在想：*这不错……但是用户怎么才能获取结果呢？他们怎么知道任务是否完成？* 你说得对——我们完全没有讨论这个问题！实际上，这里有两个方面需要解决：我们如何跟踪待处理任务及其执行情况？我们如何存储并提供生成的图像？这就是下一节的内容。

# 将结果存储在数据库和对象存储中

在上一节中，我们展示了如何实现一个后台工作程序来执行繁重的计算，以及一个 API 来调度任务给这个工作程序。然而，我们仍然缺少两个重要方面：用户没有任何方式了解任务的进度，也无法获取最终结果。让我们来解决这个问题！

## 在工作程序和 API 之间共享数据

正如我们所见，工作程序是一个在后台运行的程序，执行 API 请求它做的计算。然而，工作程序并没有与 API 服务器通信的任何方式。这是预期中的：因为可能有任意数量的服务器进程，且它们甚至可能运行在不同的物理服务器上，因此进程之间不能直接通信。始终是同样的问题：需要有一个中央数据源，供进程写入和读取数据。

事实上，解决 API 和工作程序之间缺乏通信的第一种方法是使用我们用来调度任务的相同代理：工作程序可以将结果写入代理，API 可以从中读取。这在大多数后台任务库中都是可能的，包括 Dramatiq。然而，这个解决方案有一些局限性，其中最主要的是我们能保留数据的时间有限。像 Redis 这样的代理并不适合长时间可靠地存储数据。在某些时候，我们需要删除最古老的数据以限制内存使用。

然而，我们已经知道有一些东西能够高效地存储结构化数据：当然是数据库！这就是我们在这里展示的方法。通过拥有一个中央数据库，我们可以在其中存储图像生成请求和结果，这样就能在工作程序和 API 之间共享信息。为此，我们将重用我们在《第六章》的*使用 SQLAlchemy ORM 与 SQL 数据库通信*部分中展示的很多技巧。我们开始吧！

### 定义一个 SQLAlchemy 模型

第一步是定义一个 SQLAlchemy 模型来存储单个图像生成任务。你可以如下所示查看它：

models.py

```py

class GeneratedImage(Base):    __tablename__ = "generated_images"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.now
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    num_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/models.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/models.py)

像往常一样，我们定义一个自增的 ID 作为主键。我们还添加了 `prompt`、`negative_prompt` 和 `num_steps` 列，这些列对应我们传递给工作程序任务的参数。这样，我们就可以直接将 ID 传递给工作程序，它会直接从对象中获取参数。此外，这还允许我们存储并记住用于特定生成的参数。

`progress` 列是一个整数，用来存储当前生成任务的进度。

最后，`file_name` 将存储我们在系统中保存的实际文件名。我们将在下一节中关于对象存储的部分看到如何使用它。

### 将 API 调整为在数据库中保存图像生成任务

有了这个模型后，我们对 API 中图像生成任务的调度方式稍微做了些调整。我们不再直接将任务发送给工作进程，而是首先在数据库中创建一行数据，并将该对象的 ID 作为输入传递给工作进程任务。端点的实现如下所示：

api.py

```py

@app.post(    "/generated-images",
    response_model=schemas.GeneratedImageRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_generated_image(
    generated_image_create: schemas.GeneratedImageCreate,
    session: AsyncSession = Depends(get_async_session),
)    GeneratedImage:
    image = GeneratedImage(**generated_image_create.dict())
    session.add(image)
    await session.commit()
    text_to_image_task.send(image.id)
    return image
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/api.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/api.py)

我们不会深入讨论如何使用 SQLAlchemy ORM 在数据库中创建对象。如果你需要复习，可以参考《第六章》中的*使用 SQLAlchemy ORM 与 SQL 数据库通信*部分。

在这个代码片段中，主要需要注意的是我们将新创建对象的 ID 作为`text_to_image_task`的参数传递。正如我们稍后看到的，工作进程会从数据库中重新读取这个 ID，以检索生成参数。

该端点的响应仅仅是我们`GeneratedImage`模型的表示，使用了 Pydantic 架构`GeneratedImageRead`。因此，用户将会收到类似这样的响应：

```py

{    "created_at": "2023-02-07T10:17:50.992822",
    "file_name": null,
    "id": 6,
    "negative_prompt": null,
    "num_steps": 50,
    "progress": 0,
    "prompt": "a sunset over a beach"
}
```

它展示了我们在请求中提供的提示，最重要的是，*它给了一个 ID*。这意味着用户将能够再次查询此特定请求以检索数据，并查看是否完成。这就是下面定义的`get_generated_image`端点的目的。我们不会在这里展示它，但你可以在示例仓库中阅读到它。

### 将工作进程调整为从数据库中读取和更新图像生成任务

你可能已经猜到，我们需要改变任务的实现，以便它能从数据库中检索对象，而不是直接读取参数。让我们一步步来进行调整。

我们做的第一件事是使用在任务参数中获得的 ID 从数据库中检索一个`GeneratedImage`。

worker.py

```py

@dramatiq.actor()def text_to_image_task(image_id: int):
    image = get_image(image_id)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py)

为了实现这一点，你会看到我们使用了一个名为`get_image`的辅助函数。它定义在任务的上方。让我们来看一下：

worker.py

```py

def get_image(id: int) -> GeneratedImage:    async def _get_image(id: int) -> GeneratedImage:
        async with async_session_maker() as session:
            select_query = select(GeneratedImage).where(GeneratedImage.id == id)
            result = await session.execute(select_query)
            image = result.scalar_one_or_none()
            if image is None:
                raise Exception("Image does not exist")
            return image
    return asyncio.run(_get_image(id))
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py)

它看起来可能有些奇怪，但实际上，你已经对其大部分逻辑非常熟悉了。如果你仔细观察，你会发现它定义了一个嵌套的私有函数，在其中我们定义了实际的逻辑来使用 SQLAlchemy ORM 获取和保存对象。请注意，它是*异步的*，并且我们在其中大量使用了异步 I/O 模式，正如本书中所展示的那样。

这正是我们需要像这样的辅助函数的原因。事实上，Dramatiq 并未原生设计为运行异步函数，因此我们需要手动使用`asyncio.run`来调度其执行。我们已经在*第二章*中看到过这个函数，那里介绍了异步 I/O。它的作用是运行异步函数并返回其结果。这就是我们如何在任务中同步调用包装函数而不出现任何问题。

其他方法也可以解决异步 I/O 问题。

我们在这里展示的方法是解决异步工作者问题最直接且稳健的方法。

另一种方法可能是为 Dramatiq 设置装饰器或中间件，使其能够原生支持运行异步函数，但这种方法复杂且容易出现 BUG。

我们也可以考虑拥有另一个同步工作的 SQLAlchemy 引擎和会话生成器。然而，这会导致代码中出现大量重复的内容。而且，如果我们有除了 SQLAlchemy 之外的其他异步函数，这也无法提供帮助。

现在，让我们回到`text_to_image_task`的实现：

worker.py

```py

@dramatiq.actor()def text_to_image_task(image_id: int):
    image = get_image(image_id)
    def callback(step: int, _timestep, _tensor):
        update_progress(image, step)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py)

我们为 Stable Diffusion 管道定义了一个`callback`函数。它的作用是将当前的进度保存到数据库中，针对当前的`GeneratedImage`。为此，我们再次使用了一个辅助函数`update_progress`：

worker.py

```py

def update_progress(image: GeneratedImage, step: int):    async def _update_progress(image: GeneratedImage, step: int):
        async with async_session_maker() as session:
            image.progress = int((step / image.num_steps) * 100)
            session.add(image)
            await session.commit()
    asyncio.run(_update_progress(image, step))
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py)

我们使用与`get_image`相同的方法来包装异步函数。

回到`text_to_image_task`，我们现在可以调用我们的`TextToImage`模型来生成图像。这与前一节中展示的调用完全相同。唯一的区别是，我们从`image`对象中获取参数。我们还使用 UUID 生成一个随机的文件名：

worker.py

```py

    image_output = text_to_image_middleware.text_to_image.generate(        image.prompt,
        negative_prompt=image.negative_prompt,
        num_steps=image.num_steps,
        callback=callback,
    )
    file_name = f"{uuid.uuid4()}.png"
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py)

以下部分用于将图像上传到对象存储。我们将在下一部分中更详细地解释这一点：

worker.py

```py

    storage = Storage()    storage.upload_image(image_output, file_name, settings.storage_bucket)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py)

最后，我们调用另一个辅助函数`update_file_name`，将随机文件名保存到数据库中。它将允许我们为用户检索该文件：

worker.py

```py

    update_file_name(image, file_name)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py)

如你所见，这个实现的重点是我们从数据库中读取和写入`GeneratedImage`的信息。这就是我们如何在 API 服务器和工作进程之间进行*同步*。工作进程的部分就到此为止！有了这个逻辑，我们就可以从 API 调度一个图像生成任务，而工作进程则能够在设置最终文件名之前定期更新任务进度。因此，通过 API，一个简单的`GET`请求就能让我们看到任务的状态。

## 在对象存储中存储和服务文件

我们必须解决的最后一个挑战是关于存储生成的图像。我们需要一种可靠的方式来存储它们，同时让用户能够轻松地从互联网中检索它们。

传统上，Web 应用程序的处理方式非常简单。它们将文件直接存储在服务器的硬盘中，在指定的目录下，并配置其 Web 服务器，当访问某个 URL 时提供这些文件。这实际上是我们在*第十三章*中的 WebSocket 示例中做的：我们使用了 `StaticFiles` 中间件来静态地提供我们磁盘上的 JavaScript 脚本。

虽然这种方式适用于静态文件，比如每个服务器都有自己副本的 JavaScript 或 CSS 文件，但对于用户上传或后台生成的动态文件来说并不合适，尤其是在多个进程运行在不同物理机器上的复杂架构中。问题再次出现，即不同进程读取的中央数据源问题。在前面的部分，我们看到消息代理和数据库可以在多个场景中解决这个问题。而对于任意的二进制文件，无论是图像、视频还是简单的文本文件，我们需要其他解决方案。让我们来介绍**对象存储**。

对象存储与我们日常在计算机中使用的标准文件存储有所不同，后者中的磁盘是以目录和文件的层次结构组织的。而对象存储将每个文件作为一个对象进行存储，其中包含实际数据及其所有元数据，如文件名、大小、类型和唯一 ID。这种概念化的主要好处是，它更容易将这些文件分布到多个物理机器上：*我们可以将数十亿个文件存储在同一个对象存储中*。从用户的角度来看，我们只需请求一个特定的文件，存储系统会负责从实际的物理磁盘加载该文件。

在云时代，这种方法显然获得了广泛的关注。2006 年，**亚马逊网络服务**（**AWS**）推出了其自有实现的对象存储——Amazon S3。它为开发人员提供了几乎无限的磁盘空间，允许通过一个简单的 API 存储文件，并且价格非常低廉。Amazon S3 因其广泛的流行，其 API 成为行业事实上的标准。如今，大多数云对象存储，包括微软 Azure 或 Google Cloud 等竞争对手的存储，都与 S3 API 兼容。开源实现也应运而生，如 MinIO。这个通用的 S3 API 的主要好处是，您可以在项目中使用相同的代码和库与任何对象存储提供商进行交互，并在需要时轻松切换。

总结一下，对象存储是一种非常方便的方式，用于大规模存储和提供文件，无论有多少个进程需要访问这些数据。在本节结束时，我们项目的全球架构将像*图 14.5*中所示。

![图 14.5 – Web-队列-工作者架构和对象存储](img/Figure_14.5_B19528.jpg)

图 14.5 – Web-队列-工作者架构和对象存储

值得注意的是，*对象存储会直接将文件提供给用户*。不会有一个端点，服务器在从对象存储下载文件后再将其发送给用户。以这种方式操作并没有太大好处，即使在认证方面也是如此。我们将看到，兼容 S3 的存储具有内建的机制来保护文件不被未授权访问。

### 实现一个对象存储助手

那么我们开始写代码吧！我们将使用 MinIO 的 Python 客户端库，这是一个与任何兼容 S3 的存储进行交互的库。让我们先安装它：

```py

(venv) $ pip install minio
```

我们现在可以实现一个类，以便手头有我们需要的所有操作。我们先从初始化器开始：

storage.py

```py

class Storage:    def __init__(self) -> None:
        self.client = Minio(
            settings.storage_endpoint,
            access_key=settings.storage_access_key,
            secret_key=settings.storage_secret_key,
        )
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/storage.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/storage.py)

在该类的初始化函数中，我们创建了一个 `Minio` 客户端实例。你会看到我们使用一个 `settings` 对象来提取存储 URL 和凭证。因此，使用环境变量就能非常轻松地切换它们。

然后我们将实现一些方法，帮助我们处理对象存储。第一个方法是 `ensure_bucket`：

storage.py

```py

    def ensure_bucket(self, bucket_name: str):        bucket_exists = self.client.bucket_exists(bucket_name)
        if not bucket_exists:
            self.client.make_bucket(bucket_name)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/storage.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/storage.py)

该方法的作用是确保在我们的对象存储中创建了正确的存储桶。在 S3 实现中，**存储桶**就像是你拥有的文件夹，你可以将文件存储在其中。你上传的每个文件都必须放入一个现有的存储桶中。

然后，我们定义了 `upload_image`：

storage.py

```py

    def upload_image(self, image: Image, object_name: str, bucket_name: str):        self.ensure_bucket(bucket_name)
        image_data = io.BytesIO()
        image.save(image_data, format="PNG")
        image_data.seek(0)
        image_data_length = len(image_data.getvalue())
        self.client.put_object(
            bucket_name,
            object_name,
            image_data,
            length=image_data_length,
            content_type="image/png",
        )
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/storage.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/storage.py)

这是用于将图像上传到存储的。为了简化操作，该方法接受一个 Pillow `Image` 对象，因为这是我们在 Stable Diffusion 流水线的最后得到的结果。我们实现了一些逻辑，将这个 `Image` 对象转换为适合 S3 上传的原始字节流。该方法还期望接收 `object_name`，即存储中实际的文件名，以及 `bucket_name`。请注意，我们首先确保存储桶已经正确创建，然后再尝试上传文件。

最后，我们添加了 `get_presigned_url` 方法：

storage.py

```py

    def get_presigned_url(        self,
        object_name: str,
        bucket_name: str,
        *,
        expires: timedelta = timedelta(days=7)
    )    str:
        return self.client.presigned_get_object(
            bucket_name, object_name, expires=expires
        )
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/storage.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/storage.py)

这种方法将帮助我们安全地将文件提供给用户。出于安全原因，S3 存储中的文件默认对互联网用户不可访问。为了给予文件访问权限，我们可以执行以下任一操作：

+   将文件设置为公开状态，这样任何拥有 URL 的人都能访问它。这个适合公开文件，但对于私密的用户文件则不适用。

+   生成一个带有临时访问密钥的 URL。这样，我们就可以将文件访问权限提供给用户，即使 URL 被窃取，访问也会在一段时间后被撤销。这带来的巨大好处是，URL 生成发生在我们的 API 服务器上，使用 S3 客户端。因此，在生成文件 URL 之前，我们可以根据自己的逻辑检查用户是否通过身份验证，并且是否有权访问特定的文件。这就是我们在这里采用的方法，并且此方法会在特定存储桶中的特定文件上生成预签名 URL，且有效期为一定时间。

如你所见，我们的类只是 MinIO 客户端的一个薄包装。现在我们要做的就是用它来上传图像并从 API 获取预签名 URL。

### 在工作者中使用对象存储助手

在上一节中，我们展示了任务实现中的以下几行代码：

worker.py

```py

    storage = Storage()    storage.upload_image(image_output, file_name, settings.storage_bucket)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/worker.py)

现在我们已经谈到了 `Storage` 类，你应该能猜到我们在这里做的事情：我们获取生成的图像及其随机名称，并将其上传到 `settings` 中定义的存储桶。就这样！

### 在服务器上生成预签名 URL

在 API 端，我们实现了一个新端点，角色是返回给定 `GeneratedImage` 的预签名 URL：

server.py

```py

@app.get("/generated-images/{id}/url")async def get_generated_image_url(
    image: GeneratedImage = Depends(get_generated_image_or_404),
    storage: Storage = Depends(get_storage),
)    schemas.GeneratedImageURL:
    if image.file_name is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image is not available yet. Please try again later.",
        )
    url = storage.get_presigned_url(image.file_name, settings.storage_bucket)
    return schemas.GeneratedImageURL(url=url)
```

[`github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/server.py`](https://github.com/PacktPublishing/Building-Data-Science-Applications-with-FastAPI-Second-Edition/tree/main/chapter14/complete/server.py)

在生成 URL 之前，我们首先检查 `GeneratedImage` 对象上是否设置了 `file_name` 属性。如果没有，意味着工作者任务尚未完成。如果有，我们就可以继续调用 `Storage` 类的 `get_presigned_url` 方法。

请注意，我们已经定义了依赖注入来获取 `Storage` 实例。正如本书中所展示的那样，在处理外部服务时，FastAPI 中使用依赖是一个非常好的实践。

好的，看来我们一切准备就绪！让我们看看它如何运行。

### 运行图像生成系统

首先，我们需要为项目填充环境变量，特别是数据库 URL 和 S3 凭据。为了简化，我们将使用一个简单的 SQLite 数据库和 MinIO 的示例平台作为 S3 存储。MinIO 是一个免费的开源对象存储平台，非常适合示例和玩具项目。当进入生产环境时，你可以轻松切换到任何兼容 S3 的提供商。让我们在项目根目录下创建一个 `.env` 文件：

```py

DATABASE_URL=sqlite+aiosqlite:///chapter14.dbSTORAGE_ENDPOINT=play.min.io
STORAGE_ACCESS_KEY=Q3AM3UQ867SPQQA43P2F
STORAGE_SECRET_KEY=zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG
STORAGE_BUCKET=fastapi-book-text-to-image
```

存储端点、访问密钥和秘钥是 MinIO 演示环境的参数。确保查看它们的官方文档，以了解自我们编写本书以来是否有所更改：[`min.io/docs/minio/linux/developers/python/minio-py.html#id5`](https://min.io/docs/minio/linux/developers/python/minio-py.html#id5)。

我们的`Settings`类将自动加载此文件，以填充我们在代码中使用的设置。如果你需要复习这一概念，确保查看*第十章*中的*设置和使用环境变量*部分。

现在我们可以运行系统了。确保你的 Redis 服务器仍在运行，正如在*技术要求*部分所解释的那样。首先，让我们启动 FastAPI 服务器：

```py

(venv) $ uvicorn chapter14.complete.api:app
```

然后，启动工作进程：

```py

(venv) $ dramatiq -p 1 -t 1 chapter14.complete.worker
```

堆栈现在已准备好生成图像。让我们使用 HTTPie 发起请求，开始一个新的任务：

```py

$ http POST http://localhost:8000/generated-images prompt="a sunset over a beach"HTTP/1.1 201 Created
content-length: 151
content-type: application/json
date: Mon, 13 Feb 2023 07:24:44 GMT
server: uvicorn
{
    "created_at": "2023-02-13T08:24:45.954240",
    "file_name": null,
    "id": 1,
    "negative_prompt": null,
    "num_steps": 50,
    "progress": 0,
    "prompt": "a sunset over a beach"
}
```

一个新的`GeneratedImage`已在数据库中创建，分配的 ID 为`1`。进度为*0%*；处理尚未开始。让我们尝试通过 API 查询它：

```py

http GET http://localhost:8000/generated-images/1HTTP/1.1 200 OK
content-length: 152
content-type: application/json
date: Mon, 13 Feb 2023 07:25:04 GMT
server: uvicorn
{
    "created_at": "2023-02-13T08:24:45.954240",
    "file_name": null,
    "id": 1,
    "negative_prompt": null,
    "num_steps": 50,
    "progress": 36,
    "prompt": "a sunset over a beach"
}
```

API 返回相同的对象及其所有属性。注意，进度已更新，现在为*36%*。过一会儿，我们可以再次尝试相同的请求：

```py

$ http GET http://localhost:8000/generated-images/1HTTP/1.1 200 OK
content-length: 191
content-type: application/json
date: Mon, 13 Feb 2023 07:25:34 GMT
server: uvicorn
{
    "created_at": "2023-02-13T08:24:45.954240",
    "file_name": "affeec65-5d9b-480e-ac08-000c74e22dc9.png",
    "id": 1,
    "negative_prompt": null,
    "num_steps": 50,
    "progress": 100,
    "prompt": "a sunset over a beach"
}
```

这次，进度为*100%*，文件名已经填写。图像准备好了！现在我们可以请求 API 为该图像生成一个预签名 URL：

```py

$ http GET http://localhost:8000/generated-images/1/urlHTTP/1.1 200 OK
content-length: 366
content-type: application/json
date: Mon, 13 Feb 2023 07:29:53 GMT
server: uvicorn
{
    "url": "https://play.min.io/fastapi-book-text-to-image/affeec65-5d9b-480e-ac08-000c74e22dc9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=Q3AM3UQ867SPQQA43P2F%2F20230213%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230213T072954Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=6ffddb81702bed6aac50786578eb75af3c1f6a3db28e4990467c973cb3b457a9"
}
```

我们在 MinIO 服务器上得到了一个非常长的 URL。如果你在浏览器中打开它，你会看到刚刚由我们的系统生成的图像，如*图 14.6*所示。

![图 14.6 – 生成的图像托管在对象存储中](img/Figure_14.6_B19528.jpg)

图 14.6 – 生成的图像托管在对象存储中

很不错，不是吗？我们现在拥有一个功能齐全的系统，用户能够执行以下操作：

+   请求根据他们自己的提示和参数生成图像

+   获取请求进度的信息

+   从可靠存储中获取生成的图像

我们在这里看到的架构已经可以在具有多台机器的云环境中部署。通常，我们可能会有一台标准的便宜服务器来提供 API 服务，而另一台则是更昂贵的服务器，配有专用 GPU 和充足的 RAM 来运行工作进程。代码无需更改就可以处理这种部署，因为进程间的通信是由中央元素——消息代理、数据库和对象存储来处理的。

# 总结

太棒了！你可能还没有意识到，但在这一章中，你已经学习了如何架构和实现一个非常复杂的机器学习系统，它能与你在外面看到的现有图像生成服务相媲美。我们在这里展示的概念是至关重要的，且是所有你能想象的分布式系统的核心，无论它们是设计用来运行机器学习模型、提取管道，还是数学计算。通过使用像 FastAPI 和 Dramatiq 这样的现代工具，你将能够在短时间内用最少的代码实现这种架构，最终得到一个非常快速且稳健的结果。

我们的旅程即将结束。在让你用 FastAPI 开始自己的冒险之前，我们将研究构建数据科学应用程序时的最后一个重要方面：日志记录和监控。
