# 第四章：将您的数据分析发布到 Web - PixieApp 工具

> "我认为数据是讲故事最强大的机制之一。我拿一大堆数据，试图让它讲述故事。"

– *史蒂文·列维特*，《怪诞经济学》合著者

在上一章中，我们讨论了如何通过将 PixieDust 与 Jupyter Notebooks 结合使用，借助简单的 API 加速您的数据科学项目，这些 API 可以让您加载、清理和可视化数据，而无需编写大量代码，还能通过 PixieApps 实现数据科学家和开发人员之间的协作。在本章中，我们将展示如何通过使用 PixieGateway 服务器，将您的 PixieApps 和相关的数据分析从 Jupyter Notebook 中“解放”出来，发布为 Web 应用程序。这种 Notebook 的操作化特别吸引那些业务用户（如商业分析师、高层管理人员等）使用，他们希望使用 PixieApps，但与数据科学家或开发人员不同，他们可能不太愿意使用 Jupyter Notebooks。相反，他们更愿意将其作为经典的 Web 应用程序访问，或者像 YouTube 视频一样，将其嵌入到博客文章或 GitHub 页面中。通过网站或博客文章，您可以更轻松地传达从数据分析中提取的有价值的见解和其他结果。

到本章结束时，您将能够在本地安装和配置 PixieGateway 服务器实例进行测试，或者在云端的 Kubernetes 容器中进行生产部署。对于那些不熟悉 Kubernetes 的读者，我们将在下一节中介绍其基础知识。

本章将介绍 PixieGateway 服务器的另一个主要功能，即轻松共享使用 PixieDust `display()` API 创建的图表。我们将展示如何通过单击一个按钮，将其发布为一个网页，供您的团队访问。最后，我们将介绍 PixieGateway 管理控制台，它可以让您管理应用程序、图表、内核、服务器日志，并在 Python 控制台中执行对内核的即席代码请求。

### 注意

**注意**：PixieGateway 服务器是 PixieDust 的一个子组件，其源代码可以在此找到：

[`github.com/pixiedust/pixiegateway`](https://github.com/pixiedust/pixiegateway)

# Kubernetes 概述

Kubernetes ([`kubernetes.io`](https://kubernetes.io)) 是一个可扩展的开源系统，用于自动化和编排容器化应用程序的部署和管理，这些应用程序在云服务提供商中非常流行。它通常与 Docker 容器（[`www.docker.com`](https://www.docker.com)）一起使用，尽管也支持其他类型的容器。在开始之前，您需要访问一组已经配置为 Kubernetes 集群的计算机；您可以在此处找到如何创建该集群的教程：[`kubernetes.io/docs/tutorials/kubernetes-basics`](https://kubernetes.io/docs/tutorials/kubernetes-basics)。

如果您没有足够的计算资源，一个不错的解决方案是使用提供 Kubernetes 服务的公共云服务商，例如 Amazon AWS EKS（[`aws.amazon.com/eks`](https://aws.amazon.com/eks)）、Microsoft Azure（[`azure.microsoft.com/en-us/services/container-service/kubernetes`](https://azure.microsoft.com/en-us/services/container-service/kubernetes)）或 IBM Cloud Kubernetes Service（[`www.ibm.com/cloud/container-service`](https://www.ibm.com/cloud/container-service)）。

为了更好地理解 Kubernetes 集群是如何工作的，让我们看看以下图示的高层架构：

![Kubernetes 概览](img/B09699_04_01.jpg)

Kubernetes 高层架构

在架构的顶部，我们有`kubectl`命令行工具，它允许用户通过向**Kubernetes 主节点**发送命令来管理 Kubernetes 集群。`kubectl`命令使用以下语法：

```py
kubectl [command] [TYPE] [NAME] [flags]

```

其中：

+   `command`：指定操作，例如`create`、`get`、`describe`和`delete`。

+   `TYPE`：指定资源类型，例如`pods`、`nodes`和`services`。

+   `NAME`：指定资源的名称。

+   `flags`：指定特定操作的可选标志。

### 注意

欲了解如何使用`kubectl`，请访问以下链接：

[`kubernetes.io/docs/reference/kubectl/overview`](https://kubernetes.io/docs/reference/kubectl/overview)

工作节点中的另一个重要组件是**kubelet**，它通过从**kube API Server**读取 pod 配置来控制 pod 的生命周期。它还负责与主节点的通信。kube-proxy 根据主节点指定的策略提供所有 pods 之间的负载均衡功能，从而确保整个应用程序的高可用性。

在下一部分，我们将讨论安装和配置 PixieGateway 服务器的不同方法，包括使用 Kubernetes 集群的其中一种方法。

# 安装和配置 PixieGateway 服务器

在我们深入技术细节之前，部署一个 PixieGateway 服务器实例来进行尝试会是一个好主意。

主要有两种安装类型：本地安装和服务器安装。

**本地安装**：使用这种方法进行测试和开发。

对于这一部分，我强烈推荐使用 Anaconda 虚拟环境（[`conda.io/docs/user-guide/tasks/manage-environments.html`](https://conda.io/docs/user-guide/tasks/manage-environments.html)），因为它们提供了良好的环境隔离，允许您在不同版本和配置的 Python 包之间进行实验。

如果您管理多个环境，您可以使用以下命令获取所有可用环境的列表：

```py
conda env list

```

首先，通过在终端使用以下命令选择您选择的环境：

```py
source activate <<my_env>>

```

你应该能在终端中看到你的环境名称，这表明你已正确激活它。

接下来，通过运行以下命令从 PyPi 安装`pixiegateway`包：

```py
pip install pixiegateway

```

### 注意

**注意**：你可以在 PyPi 上找到更多关于`pixiegateway`包的信息：

[`pypi.python.org/pypi/pixiegateway`](https://pypi.python.org/pypi/pixiegateway)

安装完所有依赖项后，你可以开始启动服务器。假设你想使用`8899 端口`，可以通过以下命令启动 PixieGateway 服务器：

```py
jupyter pixiegateway --port=8899

```

示例输出应如下所示：

```py
(dashboard) davids-mbp-8:pixiegateway dtaieb$ jupyter pixiegateway --port=8899
Pixiedust database opened successfully
Pixiedust version 1.1.10
[PixieGatewayApp] Jupyter Kernel Gateway at http://127.0.0.1:8899

```

### 注意

**注意**：要停止 PixieGateway 服务器，只需在终端中按*Ctrl* + *C*。

现在，你可以通过以下 URL 打开 PixieGateway 管理控制台：`http://localhost:8899/admin`。

### 注意

**注意**：在遇到挑战时，使用`admin`作为用户名，密码留空（即没有密码）。我们将在本章稍后通过*PixieGateway 服务器配置*部分介绍如何配置安全性和其他属性。

**使用 Kubernetes 和 Docker 安装服务器**：如果你需要在生产环境中运行 PixieGateway，并且希望通过 Web 向多个用户提供已部署的 PixieApps 访问权限，请使用此安装方法。

以下说明将使用 IBM Cloud Kubernetes 服务，但也可以很容易地适应其他提供商：

1.  如果你还没有 IBM Cloud 账户，请创建一个，并从目录中创建一个容器服务实例。

    ### 注意

    **注意**：有一个适用于测试的免费轻量版计划。

1.  下载并安装 Kubernetes CLI（[`kubernetes.io/docs/tasks/tools/install-kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl)）和 IBM Cloud CLI（[`console.bluemix.net/docs/cli/reference/bluemix_cli/get_started.html#getting-started`](https://console.bluemix.net/docs/cli/reference/bluemix_cli/get_started.html#getting-started)）。

    ### 注意

    **注意**：关于 Kubernetes 容器的其他入门文章可以在此找到：

    [`console.bluemix.net/docs/containers/container_index.html#container_index`](https://console.bluemix.net/docs/containers/container_index.html#container_index)

1.  登录到 IBM Cloud 后，定向到你的 Kubernetes 实例所在的组织和空间。安装并初始化`container-service`插件：

    ```py
    bx login -a https://api.ng.bluemix.net
    bx target -o <YOUR_ORG> -s <YOUR_SPACE></YOUR_SPACE>
    bx plugin install container-service -r Bluemix
    bx cs init

    ```

1.  检查你的集群是否已创建，如果没有，创建一个：

    ```py
    bx cs clusters
    bx cs cluster-create --name my-cluster

    ```

1.  下载稍后将用于`kubectl`命令的集群配置文件，该命令将在你的本地计算机上执行：

    ```py
    bx cs cluster-config my-cluster

    ```

    上述命令将生成一个临时的 YML 文件，包含集群信息和一个环境变量导出语句，你必须在开始使用`kubectl`命令之前运行，如示例所示：

    ```py
     export KUBECONFIG=/Users/dtaieb/.bluemix/plugins/container-
     service/clusters/davidcluster/kube-config-hou02-davidcluster.yml

    ```

    ### 注意

    **注意**：YAML 是一种非常流行的数据序列化格式，通常用于系统配置。你可以在此找到更多信息：

    [`www.yaml.org/start.html`](http://www.yaml.org/start.html)

1.  现在，你可以使用`kubectl`为 PixieGateway 服务器创建部署和服务。为了方便，PixieGateway GitHub 仓库中已经有一个通用版本的`deployment.yml`和`service.yml`，你可以直接引用。我们将在本章后面*PixieGateway 服务器配置*部分回顾如何为 Kubernetes 配置这些文件：

    ```py
    kubectl create -f https://github.com/ibm-watson-data-lab/pixiegateway/raw/master/etc/deployment.yml
    kubectl create -f https://github.com/ibm-watson-data-lab/pixiegateway/raw/master/etc/service.yml

    ```

1.  使用`kubectl get`命令验证集群的状态是一个不错的主意：

    ```py
    kubectl get pods
    kubectl get nodes
    kubectl get services

    ```

1.  最后，你需要服务器的公共 IP 地址，你可以通过在终端使用以下命令获取的输出中的`Public IP`列来找到它：

    ```py
    bx cs workers my-cluster

    ```

1.  如果一切顺利，你现在可以通过在`http://<server_ip>>:32222/admin`打开管理员控制台来测试你的部署。这次，管理员控制台的默认凭据是`admin`/`changeme`，我们将在下一节中演示如何更改这些凭据。

Kubernetes 安装说明中使用的`deployment.yml`文件引用了一个 Docker 镜像，该镜像已预安装并配置了 PixieGateway 二进制文件及其所有依赖项。PixieGateway Docker 镜像可以通过以下链接获取：[`hub.docker.com/r/dtaieb/pixiegateway-python35`](https://hub.docker.com/r/dtaieb/pixiegateway-python35)。

在本地工作时，推荐的方法是遵循之前描述的本地安装步骤。然而，对于那些喜欢使用 Docker 镜像的读者，可以尝试在没有 Kubernetes 的情况下直接在本地笔记本电脑上使用简单的 Docker 命令安装 PixieGateway Docker 镜像。

```py
docker run -p 9999:8888 dtaieb/pixiegateway-python35

```

上述命令假设你已经安装了 Docker，并且它当前正在本地机器上运行。如果没有，你可以从以下链接下载安装程序：[`docs.docker.com/engine/installation`](https://docs.docker.com/engine/installation)。

如果本地没有现成的 Docker 镜像，它将自动拉取该镜像，容器将启动，并在本地端口`8888`上启动 PixieGateway 服务器。命令中的`-p`选项将本地的`8888 端口`映射到主机机器的`9999 端口`。通过这个配置，你可以通过以下 URL 访问 PixieGateway 服务器的 Docker 实例：`http://localhost:9999/admin`。

### 注意

你可以在这里找到更多关于 Docker 命令行的信息：

[`docs.docker.com/engine/reference/commandline/cli`](https://docs.docker.com/engine/reference/commandline/cli)

### 注意

**注意**：使用这种方法的另一个原因是为 PixieGateway 服务器提供你自定义的 Docker 镜像。如果你为 PixieGateway 构建了扩展，并希望将其作为已经配置好的 Docker 镜像提供给用户，这会很有用。如何从基础镜像构建 Docker 镜像的讨论超出了本书的范围，但你可以在这里找到详细信息：

[`docs.docker.com/engine/reference/commandline/image_build`](https://docs.docker.com/engine/reference/commandline/image_build)

## PixieGateway 服务器配置

配置 PixieGateway 服务器与配置 Jupyter Kernel Gateway 非常相似。大多数选项都是通过 Python 配置文件进行配置的；为了开始，你可以使用以下命令生成一个模板配置文件：

```py
jupyter kernelgateway --generate-config

```

`jupyter_kernel_gateway_config.py`模板文件将在`~/.jupyter`目录下生成（`~`表示用户主目录）。你可以在这里找到更多关于标准 Jupyter Kernel Gateway 选项的信息：[`jupyter-kernel-gateway.readthedocs.io/en/latest/config-options.html`](http://jupyter-kernel-gateway.readthedocs.io/en/latest/config-options.html)。

当你在本地工作并且可以轻松访问文件系统时，使用`jupyter_kernel_gateway_config.py`文件是可以的。对于使用 Kubernetes 安装时，建议将选项配置为环境变量，你可以通过在`deployment.yml`文件中使用预定义的`env`类别直接设置这些变量。

现在让我们来看一下 PixieGateway 服务器的每个配置选项。这里提供了一个列表，包含了 Python 方法和环境方法：

### 注意

**注意**：提醒一下，Python 方法是指在`jupyter_kernel_gateway_config.py` Python 配置文件中设置参数，而环境方法是指在 Kubernetes 的`deployment.yml`文件中设置参数。

+   **管理员控制台凭证**：配置管理员控制台的用户 ID/密码：

    +   **Python**: `PixieGatewayApp.admin_user_id`, `PixieGatewayApp.admin_password`

    +   **环境**: `ADMIN_USERID` 和 `ADMIN_PASSWORD`

+   **存储连接器**: 配置一个持久化存储以保存各种资源，如图表和笔记本。默认情况下，PixieGateway 使用本地文件系统；例如，它会将发布的笔记本保存在`~/pixiedust/gateway`目录下。使用本地文件系统对于本地测试环境可能是可以的，但当使用 Kubernetes 安装时，你将需要明确地使用持久卷（[`kubernetes.io/docs/concepts/storage/persistent-volumes`](https://kubernetes.io/docs/concepts/storage/persistent-volumes)），这可能会比较难以使用。如果没有配置持久化策略，当容器重启时，已持久化的文件将被删除，所有已发布的图表和 PixieApps 也会消失。PixieGateway 提供了另一种选项，可以配置一个存储连接器，让你使用你选择的机制和后端来持久化数据。

    要配置图表的存储连接器，你必须在以下配置变量中的任何一个中指定一个完全限定的类名：

    +   **Python**: `SingletonChartStorage.chart_storage_class`

    +   **环境**: `PG_CHART_STORAGE`

    引用的连接器类必须继承自`pixiegateway.chartsManager`包中定义的`ChartStorage`抽象类（实现可以在这里找到：[`github.com/ibm-watson-data-lab/pixiegateway/blob/master/pixiegateway/chartsManager.py`](https://github.com/ibm-watson-data-lab/pixiegateway/blob/master/pixiegateway/chartsManager.py)）。

    PixieGateway 提供一个开箱即用的连接器，用于连接 Cloudant/CouchDB NoSQL 数据库 ([`couchdb.apache.org`](http://couchdb.apache.org))。要使用此连接器，你需要将连接器类设置为`pixiegateway.chartsManager.CloudantChartStorage`。你还需要指定其他配置变量来指定服务器和凭证信息（我们展示了 Python/环境变量形式）：

    +   `CloudantConfig.host / PG_CLOUDANT_HOST`

    +   `CloudantConfig.port / PG_CLOUDANT_PORT`

    +   `CloudantConfig.protocol / PG_CLOUDANT_PROTOCOL`

    +   `CloudantConfig.username / PG_CLOUDANT_USERNAME`

    +   `CloudantConfig.password / PG_CLOUDANT_PASSWORD`

+   **远程内核**：指定远程 Jupyter Kernel Gateway 的配置。

    目前，此配置选项仅在 Python 模式下受支持。你需要使用的变量名是`ManagedClientPool.remote_gateway_config`。预期的值是一个包含服务器信息的 JSON 对象，可以通过两种方式指定：

    +   `protocol`，`host`，和`port`

    +   `notebook_gateway`指定服务器的完全限定 URL

    根据内核配置，安全性也可以通过两种方式提供：

    +   `auth_token`

    +   `user`和`password`

    这可以在以下示例中看到：

    ```py
    c.ManagedClientPool.remote_gateway_config={
        'protocol': 'http',
        'host': 'localhost',
        'port': 9000,
        'auth_token':'XXXXXXXXXX'
    }

    c.ManagedClientPool.remote_gateway_config={
        'notebook_gateway': 'https://YYYYY.us-south.bluemix.net:8443/gateway/default/jkg/',
        'user': 'clsadmin',
        'password': 'XXXXXXXXXXX'
    }
    ```

    ### 注意

    注意，在前面的示例中，你需要在变量前加上`c.`。这是来自底层 Jupyter/IPython 配置机制的要求。

    作为参考，以下是使用 Python 和 Kubernetes 环境变量格式的完整配置示例文件：

+   以下是`jupyter_kernel_gateway_config.py`的内容：

    ```py
    c.PixieGatewayApp.admin_password = "password"

    c.SingletonChartStorage.chart_storage_class = "pixiegateway.chartsManager.CloudantChartStorage"
    c.CloudantConfig.host="localhost"
    c.CloudantConfig.port=5984
    c.CloudantConfig.protocol="http"
    c.CloudantConfig.username="admin"
    c.CloudantConfig.password="password"

    c.ManagedClientPool.remote_gateway_config={
        'protocol': 'http',
        'host': 'localhost',
        'port': 9000,
        'auth_token':'XXXXXXXXXX'
    }
    ```

+   以下是`deployment.yml`的内容：

    ```py
    apiVersion: extensions/v1beta1
    kind: Deployment 
    metadata:
      name: pixiegateway-deployment
    spec:
      replicas: 1
      template:
        metadata:
          labels:
            app: pixiegateway
        spec:
          containers:
            - name: pixiegateway
              image: dtaieb/pixiegateway-python35
              imagePullPolicy: Always
              env:
                - name: ADMIN_USERID
                  value: admin
                - name: ADMIN_PASSWORD
                  value: changeme
                - name: PG_CHART_STORAGE
                  value: pixiegateway.chartsManager.CloudantChartStorage
                - name: PG_CLOUDANT_HOST
                  value: XXXXXXXX-bluemix.cloudant.com
                - name: PG_CLOUDANT_PORT
                  value: "443"
                - name: PG_CLOUDANT_PROTOCOL
                  value: https
                - name: PG_CLOUDANT_USERNAME
                  value: YYYYYYYYYYY-bluemix
                - name: PG_CLOUDANT_PASSWORD
                  value: ZZZZZZZZZZZZZ
    ```

## PixieGateway 架构图

现在是时候再次查看在第二章中展示的 PixieGateway 架构图，*使用 Python 和 Jupyter Notebooks 为你的数据分析提供动力*。服务器作为 Jupyter Kernel Gateway 的自定义扩展（称为 Personality）实现 ([`github.com/jupyter/kernel_gateway`](https://github.com/jupyter/kernel_gateway))。

反过来，PixieGateway 服务器提供了扩展点，以自定义一些行为，我们将在本章稍后讨论。

PixieGateway 服务器的高级架构图如下所示：

![PixieGateway 架构图](img/B09699_04_02.jpg)

PixieGateway 架构图

如图所示，PixieGateway 为三种类型的客户端提供 REST 接口：

+   **Jupyter Notebook 服务器**：调用一组专门的 REST API 来共享图表和发布 PixieApps 作为 Web 应用程序

+   **浏览器客户端运行 PixieApp**：一个特殊的 REST API 管理着与之关联的内核中 Python 代码的执行

+   **浏览器客户端运行管理控制台**：一组专门的 REST API 用于管理各种服务器资源和统计信息，例如 PixieApps 和内核实例

在后台，PixieGateway 服务器管理一个或多个负责运行 PixieApps 的 Jupyter 内核实例的生命周期。在运行时，每个 PixieApp 都会使用一组特定的步骤在内核实例上进行部署。下图展示了在服务器上运行的所有 PixieApp 用户实例的典型拓扑结构：

![PixieGateway 架构](img/B09699_04_03.jpg)

运行中的 PixieApp 实例拓扑结构

当 PixieApp 部署在服务器上时，Jupyter Notebook 中每个单元的代码都会被分析并分为两部分：

+   **预热代码**：这是在所有主 PixieApp 定义之上的所有单元中定义的代码。此代码仅在 PixieApp 应用程序第一次在内核上启动时运行一次，除非内核重新启动，或者显式从运行代码中调用，否则不会再次运行。这一点很重要，因为它有助于优化性能；例如，您应该始终将那些加载大量数据且变化不大或者初始化可能需要较长时间的代码放在预热部分。

+   **运行代码**：这是每个用户会话中在其独立实例中运行的代码。运行代码通常是从包含 PixieApp 类声明的单元中提取的。发布者通过对 Python 代码进行静态分析，自动发现这个单元，并特别查找以下两个条件，这两个条件必须同时满足：

    +   该单元包含一个带有 `@PixieApp` 注解的类

    +   该单元实例化该类并调用其 `run()` 方法

        ```py
        @PixieApp
        class MyApp():
            @route()
            def main_screen(self):
            return "<div>Hello World</div>"

        app = MyApp()
        app.run()
        ```

    例如，以下代码必须放在单独的单元中，才能符合作为运行代码的条件：

    正如我们在 第三章 *加速使用 Python 库进行数据分析* 中看到的那样，可以在同一个笔记本中声明多个 PixieApps，作为子 PixieApp 或作为主 PixieApp 的基类。在这种情况下，我们需要确保它们定义在自己的单元中，并且不要尝试实例化它们并调用其 `run()` 方法。

    规则是，只有一个主 PixieApp 类可以调用 `run()` 方法，包含此代码的单元会被 PixieGateway 视为运行代码。

    ### 注意

    **注意**：在 PixieGateway 服务器进行静态分析时，未标记为代码的单元（如 Markdown、Raw NBConvert 或 Heading）会被忽略。因此，您可以安全地将它们保留在您的笔记本中。

对于每个客户端会话，PixieGateway 将使用运行代码（如上图中的彩色六边形表示）实例化主 PixieApp 类的一个实例。根据当前负载，PixieGateway 将决定在特定内核实例中应该运行多少个 PixieApp，如果需要，自动创建一个新的内核来服务额外的用户。例如，如果五个用户正在使用同一个 PixieApp，三个实例可能在特定的内核实例中运行，而另外两个将会在另一个内核实例中运行。PixieGateway 会不断监控使用模式，通过负载均衡将 PixieApp 实例分配到多个内核之间，从而优化工作负载分配。

为了帮助理解笔记本代码是如何被分解的，下面的图表展示了如何从笔记本中提取热身代码和运行代码，并将其转化，以确保多个实例能够在同一内核中和平共存：

### 注意

提醒一下，包含主 PixieApp 的单元格也必须有代码来实例化它并调用`run()`方法。

![PixieGateway 架构](img/B09699_04_04.jpg)

PixieApp 生命周期：热身代码与运行代码

因为一个给定的内核实例可以托管多个包含主 PixieApp 的笔记本，我们需要确保在执行两个主 PixieApp 的热身代码时不会发生意外的名称冲突。例如，`title`变量可能在两个 PixieApp 中都有使用，如果不加以处理，第二个 PixieApp 的值将覆盖第一个的值。为了避免这种冲突，所有热身代码中的变量名都会通过注入命名空间来使其唯一。

`title = 'some string'`语句在发布后会变成`ns1_title = 'some string'`。PixieGateway 发布程序还会更新代码中所有`title`的引用，以反映新的名称。所有这些重命名操作都是在运行时自动完成的，开发者无需做任何特别的事情。

我们稍后将在介绍*PixieApp 详细信息*页面时展示真实的代码示例。

### 提示

如果你已经将主 PixieApp 的代码打包为一个在笔记本中导入的 Python 模块，你仍然需要声明一个封装的 PixieApp 代码，该代码继承自主 PixieApp。这是因为 PixieGateway 会进行静态代码分析，查找`@PixieApp`注释，如果没有找到，主 PixieApp 将无法被正确识别。

例如，假设你有一个名为`AwesomePixieApp`的 PixieApp，它是从`awesome package`中导入的。在这种情况下，你需要将以下代码放入它自己的单元格中：

```py
from awesome import AwesomePixieApp
@PixieApp
class WrapperAwesome(AwesomePixieApp):
    pass
app = WrapperAwesome()
app.run()
```

## 发布应用程序

在本节中，我们将发布我们在第三章中创建的*GitHub 跟踪*应用程序，*使用 Python 库加速数据分析*，并将其发布到 PixieGateway 实例中。

### 注意

你可以从这个 GitHub 位置使用已完成的笔记本：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%204.ipynb`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%203/GitHub%20Tracking%20Application/GitHub%20Sample%20Application%20-%20Part%204.ipynb)

从笔记本中，像往常一样运行应用程序，并使用位于单元输出左上方的发布按钮，开始该过程：

![发布应用程序](img/B09699_04_05.jpg)

调用发布对话框

发布对话框包含多个标签菜单：

+   **选项**：

    +   **PixieGateway 服务器**：例如，`http://localhost:8899`

    +   **页面标题**：一个简短的描述，将作为浏览器中显示时的页面标题

+   **安全性**：通过网页访问时，请配置 PixieApp 的安全性：

    +   **无安全性**

    +   **令牌**：必须将安全令牌作为查询参数添加到 URL 中，例如，`http://localhost:8899/GitHubTracking?token=941b3990d5c0464586d67e48705b9deb`。

    ### 注意

    **注意**：此时，PixieGateway 并未提供任何身份验证/授权机制。第三方授权，如 OAuth 2.0（[`oauth.net/2`](https://oauth.net/2)）、JWT（[`jwt.io`](https://jwt.io)）等，将在未来添加。

+   **导入**：显示 PixieDust 发布器自动检测到的 Python 包依赖列表。这些导入的包将自动安装，如果目标内核中没有这些包的话。当检测到某个特定的依赖项时，PixieDust 会查看当前系统的版本和安装位置，例如，PyPi 或者自定义的安装 URL（如 GitHub 仓库等）。

+   **内核规格**：在这里，您可以为 PixieApp 选择内核规格。默认情况下，PixieDust 会选择 PixieGateway 服务器上可用的默认内核，但如果您的笔记本依赖于 Apache Spark，例如，您应该能够选择一个支持该功能的内核。此选项也可以在 PixieApp 部署后，通过管理员控制台进行更改。

以下是 PixieApp 发布对话框的示例截图：

![发布应用程序](img/B09699_04_06.jpg)

PixieApp 发布对话框

单击 **发布** 按钮将启动发布过程。完成后（根据笔记本的大小，通常非常快速），您将看到以下屏幕：

![发布应用程序](img/B09699_04_07.jpg)

成功发布屏幕

然后，您可以通过单击提供的链接来测试该应用程序，您可以复制该链接并与团队中的其他用户共享。以下截图显示了 *GitHub Tracking* 应用程序作为 Web 应用程序在 PixieGateway 上运行的三个主要屏幕：

![发布应用程序](img/B09699_04_08.jpg)

PixieApp 作为 Web 应用程序运行

现在你已经知道如何发布 PixieApp，接下来我们来回顾一些开发者最佳实践和规则，这些规则将帮助你优化那些打算发布为 Web 应用的 PixieApp：

+   每个用户会话都会创建一个 PixieApp 实例，因此为了提高性能，确保其中不包含长时间运行的代码或加载大量静态数据（不常更改的数据）。相反，将其放在热身代码部分，并根据需要从 PixieApp 中引用。

+   不要忘记在同一个单元格中添加运行 PixieApp 的代码。如果没有这样做，运行时会在网页上显示一个空白页面。作为一种良好的实践，建议将 PixieApp 实例分配到一个单独的变量中。例如，可以这样做：

    ```py
    app = GitHubTracking()
    app.run()
    ```

    这是替代以下代码的做法

    ```py
    GitHubTracking().run()
    ```

+   你可以在同一个 Notebook 中声明多个 PixieApp 类，如果你使用子 PixieApp 或 PixieApp 继承，这会是必要的。但是，只有其中一个可以是主 PixieApp，它是 PixieGateway 运行的那个。这个类包含了额外的代码，用于实例化并运行 PixieApp。

+   为你的 PixieApp 类添加文档字符串（[`www.python.org/dev/peps/pep-0257`](https://www.python.org/dev/peps/pep-0257)）是一个好主意，它简要描述应用程序的功能。正如我们在本章稍后的*PixieGateway 管理控制台*部分中所看到的，这个文档字符串将在 PixieGateway 管理控制台中显示，如以下示例所示：

    ```py
    @PixieApp
    class GitHubTracking(RepoAnalysis):
        """
        GitHub Tracking Sample Application
        """
        @route()
        def main_screen(self):
            return """
        ...
    ```

## 在 PixieApp URL 中编码状态

在某些情况下，你可能希望将 PixieApp 的状态捕捉到 URL 中作为查询参数，以便可以书签化和/或与其他人共享。这个思路是，当使用查询参数时，PixieApp 不会从主屏幕开始，而是自动激活与参数相对应的路由。例如，在*GitHub Tracking*应用中，你可以使用`http://localhost:8899/pixieapp/GitHubTracking?query=pixiedust`来跳过初始屏幕，直接进入显示与给定查询匹配的仓库列表的表格。

你可以通过在路由中添加`persist_args`特殊参数，使查询参数在路由激活时自动添加到 URL 中。

对于`do_search()`路由，它看起来应该是这样的：

```py
@route(query="*", persist_args='true')
@templateArgs
def do_search(self, query):
    self.first_url = "https://api.github.com/search/repositories?q={}".format(query)
    self.prev_url = None
    self.next_url = None
    self.last_url = None
    ...
```

### 注意

你可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%204/sampleCode1.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%204/sampleCode1.py)

`persist_args`关键字参数不会影响路由的激活方式。它只是为了在路由激活时自动将适当的查询参数添加到 URL 中。你可以尝试在 Notebook 中做出这个简单的更改，重新发布 PixieApp 到 PixieGateway 服务器并进行测试。当你在第一个屏幕上点击提交按钮时，你会发现 URL 会自动更新，包含查询参数。

### 注意

**注意**：`persist_args` 参数在 Notebook 中运行时也能工作，尽管实现方式不同，因为我们没有 URL。相反，参数会通过 `pixieapp` 键添加到单元格元数据中，如下图所示：

![PixieApp URL 中的编码状态](img/B09699_04_09.jpg)

显示 PixieApp 参数的单元格元数据

如果您使用 `persist_args` 功能，可能会发现，在进行迭代开发时，每次都去单元格元数据中移除参数会变得很麻烦。作为快捷方式，PixieApp 框架在右上角的工具栏中添加了一个主页按钮，单击即可重置参数。

作为替代方案，您也可以完全避免在 Notebook 中运行时将路由参数保存到单元格元数据中（但在 Web 上运行时仍然保存）。为此，您需要使用 `web` 作为 `persist_args` 参数的值，而不是 `true`：

```py
@route(query="*", persist_args='web')
…
```

## 通过将图表发布为网页进行分享

在本节中，我们展示了如何轻松分享通过 `display()` API 创建的图表，并将其发布为网页。

使用 第二章中的示例，*使用 Python 和 Jupyter Notebooks 来增强数据分析*，让我们加载汽车性能数据集并使用 `display()` 创建一个图表：

```py
import pixiedust
cars = pixiedust.sampleData(1, forcePandas=True) #car performance data
display(cars)
```

### 注意

您可以在这里找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%204/sampleCode2.py`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%204/sampleCode2.py)

在 PixieDust 输出界面中，选择 **柱状图** 菜单，然后在选项对话框中选择 `horsepower` 作为 **Keys** 和 `mpg` 作为 **Values**，如图所示：

![通过将图表发布为网页进行分享](img/B09699_04_10.jpg)

PixieDust 图表选项

然后，我们使用**分享**按钮来调用图表分享对话框，如下图所示，使用 Bokeh 作为渲染器：

### 注意

**注意**：图表分享适用于任何渲染器，我鼓励您尝试使用其他渲染器，如 Matplotlib 和 Mapbox。

![通过将图表发布为网页进行分享](img/B09699_04_11.jpg)

调用分享图表对话框

在**分享图表**对话框中，您可以指定 PixieGateway 服务器和图表的可选描述：

### 注意

请注意，作为一种便捷功能，PixieDust 会自动记住上次使用的设置。

![通过将图表发布为网页进行分享](img/B09699_04_12.jpg)

分享图表对话框

单击 **分享** 按钮将启动发布过程，图表内容会传送到 PixieGateway，然后返回一个指向网页的唯一 URL。与 PixieApp 类似，您可以将此 URL 分享给团队：

![通过将图表发布为网页进行分享](img/B09699_04_13.jpg)

图表分享确认对话框

确认对话框包含图表的唯一 URL 以及一个 HTML 片段，让您将图表嵌入到自己的网页中，比如博客文章或仪表板。

点击链接将显示以下 PixieGateway 页面：

![通过将图表发布为网页来共享](img/B09699_04_14.jpg)

图表页面

前面的页面显示了图表的元数据，例如**作者**、**描述**和**日期**，以及嵌入的 HTML 片段。请注意，如果图表具有交互性（如 Bokeh、Brunel 或 Mapbox），则它会在 PixieGateway 页面中得到保留。

例如，在前面的截图中，用户仍然可以使用滚轮缩放、框选缩放和拖动来探索图表或将图表下载为 PNG 文件。

将图表嵌入到您自己的页面中也非常简单。只需将嵌入的 HTML 片段复制到 HTML 的任何位置，如下例所示：

```py
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Example page with embedded chart</title>
    </head>
    <body>
        <h1> Embedded a PixieDust Chart in a custom HTML Page</h1>
        <div>
            <object type="text/html" width="600" height="400"
                data="http://localhost:8899/embed/04089782-7543-42a6-8dd1-e4d1cb06596a/600/400"> 
                <a href="http://localhost:8899/embed/04089782-7543-42a6-8dd1-e4d1cb06596a">View Chart</a>
            </object>
        </div>
    </body>
</html>
```

### 注意

您可以在此处找到代码文件：

[`github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%204/sampleCode3.html`](https://github.com/DTAIEB/Thoughtful-Data-Science/blob/master/chapter%204/sampleCode3.html)

### 提示

嵌入的图表对象必须使用与浏览器相同级别或更高的安全性。如果不符合，浏览器将抛出混合内容错误。例如，如果主机页面是通过 HTTPS 加载的，则嵌入的图表也必须通过 HTTPS 加载，这意味着您需要在 PixieGateway 服务器上启用 HTTPS。您还可以访问[`jupyter-kernel-gateway.readthedocs.io/en/latest/config-options.html`](http://jupyter-kernel-gateway.readthedocs.io/en/latest/config-options.html)为 PixieGateway 服务器配置 SSL/TLS 证书。另一种更容易维护的解决方案是为提供 TLS 终止的 Kubernetes 集群配置 Ingress 服务。

为了方便起见，我们在此提供了一个 PixieGateway 服务的模板入口 YAML 文件：[`github.com/ibm-watson-data-lab/pixiegateway/blob/master/etc/ingress.yml`](https://github.com/ibm-watson-data-lab/pixiegateway/blob/master/etc/ingress.yml)。您需要使用提供的 TLS 主机和密钥更新此文件。例如，如果您使用的是 IBM Cloud Kubernetes 服务，只需在`<your cluster name>`占位符中输入集群名称。有关如何将 HTTP 重定向到 HTTPS 的更多信息，请参阅：[`console.bluemix.net/docs/containers/cs_annotations.html#redirect-to-https`](https://console.bluemix.net/docs/containers/cs_annotations.html#redirect-to-https)。入口服务是提高安全性、可靠性并防止 DDOS 攻击的好方法。例如，您可以设置各种限制，例如允许每个唯一 IP 地址每秒的请求/连接次数或最大带宽限制。有关更多信息，请参见[`kubernetes.io/docs/concepts/services-networking/ingress`](https://kubernetes.io/docs/concepts/services-networking/ingress)。

## PixieGateway 管理控制台

管理控制台是管理资源和进行故障排除的好工具。你可以通过`/admin`网址访问它。请注意，你需要使用你配置的用户名/密码进行身份验证（关于如何配置用户名/密码的说明，请参见本章中的*PixieGateway 服务器配置*部分；默认情况下，用户名是`admin`，密码为空）。

管理控制台的用户界面由多个菜单组成，集中在特定任务上。让我们逐一查看：

+   **PixieApps**：

    +   关于所有已部署 PixieApps 的信息：网址、描述等

    +   安全管理

    +   操作，例如，删除和下载

    ![PixieGateway 管理控制台](img/B09699_04_15.jpg)

    管理控制台 PixieApp 管理页面

+   **图表**：

    +   关于所有已发布图表的信息：链接、预览等

    +   操作，例如，删除、下载和嵌入片段

    ![PixieGateway 管理控制台](img/B09699_04_16.jpg)

    管理控制台图表管理页面

+   **内核统计**：

    以下截图显示了**内核统计**屏幕：

    ![PixieGateway 管理控制台](img/B09699_04_17.jpg)

    管理控制台内核统计页面

    此屏幕显示 PixieGateway 中当前运行的所有内核的实时表格。每一行包含以下信息：

    +   **内核名称**：这是带有深入链接的内核名称，点击后会显示**内核规格**、**日志**和**Python 控制台**。

    +   **状态**：这显示状态为`空闲`或`忙碌`。

    +   **忙碌比率**：这是一个介于 0 和 100%之间的值，表示自启动以来内核的利用率。

    +   **正在运行的应用**：这是一个正在运行的 PixieApps 列表。每个 PixieApp 都是一个深入链接，显示该 PixieApp 的预热代码并运行代码。这对于故障排除非常有用，因为你可以查看 PixieGateway 正在运行的代码。

    +   **用户数量**：这是在该内核中有活动会话的用户数量。

+   **服务器日志**：

    完整访问龙卷风服务器日志以进行故障排除

    ![PixieGateway 管理控制台](img/B09699_04_18.jpg)

    管理控制台服务器日志页面

## Python 控制台

通过点击**内核统计**屏幕中的内核链接来调用 Python 控制台。管理员可以使用它来执行任何针对内核的代码，这对于故障排除非常有用。

例如，以下截图展示了如何调用 PixieDust 日志：

![Python 控制台](img/B09699_04_19.jpg)

从 PixieGateway 管理员 Python 控制台显示 PixieDust 日志

## 显示 PixieApp 的预热和运行代码

当加载页面时发生执行错误时，PixieGateway 将在浏览器中显示完整的 Python 回溯。然而，错误可能很难找到，因为其根本原因可能在于 PixieApp 启动时执行一次的预热代码。一项重要的调试技巧是查看 PixieGateway 执行的预热和运行代码，以发现任何异常。

如果错误仍然不明显，你可以例如将热身和运行代码复制到一个临时笔记本中，然后尝试从那里运行，期望能够重现错误并找出问题。

你可以通过点击**Kernel Stats**屏幕上的 PixieApp 链接来访问热身和运行代码，点击后会进入如下屏幕：

![显示 PixieApp 的热身和运行代码](img/B09699_04_20.jpg)

显示热身和运行代码

请注意，热身和运行代码没有原始的代码格式，因此可能较难阅读。你可以通过将其复制并粘贴到临时笔记本中，再重新格式化来缓解这个问题。

# 总结

阅读完本章后，你应该能够安装、配置和管理 PixieGateway 微服务服务器，将图表发布为网页，并将 PixieApp 从笔记本部署到网页应用程序。无论你是从事 Jupyter 笔记本中分析的数据显示科学家，还是为业务用户编写和部署应用程序的开发者，本章展示了 PixieDust 如何帮助你更高效地完成任务，减少将分析运用到生产中的时间。

在下一章中，我们将讨论与 PixieDust 和 PixieApp 编程模型相关的高级主题和最佳实践，这些内容在后续章节中讨论行业用例和示例数据管道时将非常有用。
