# 第十三章：地理空间 REST API

在网络上发布数据以供消费是现代 GIS 的一个重要组成部分。为了将数据从远程服务器传输到远程客户端，大多数地理空间发布软件堆栈都使用 **表示状态转移**（**REST**）Web 服务。对于针对特定数据资源的 Web 请求，REST 服务返回 **JavaScript 对象表示法**（**JSON**）编码的数据给请求的客户端机器。这些 Web 服务组合成一个应用程序编程接口（API），其中将包含代表每个可查询数据资源的端点。

通过结合 Python 网络框架、**对象关系映射**（**ORM**）和 PostGIS 后端，我们可以创建一个自定义的 REST API，该 API 将以 JSON 格式响应用户的 Web 请求。在这个练习中，我们将使用 Flask 网络框架和 SQLAlchemy 模块，结合 GeoAlchemy2 提供的空间 ORM 功能。

在本章中，我们将学习以下内容：

+   REST API 组件

+   JSON 响应格式化

+   如何处理 `GET`、`POST`、`PUT` 和 `DELETE` 请求方法

+   使用 API 执行地理空间操作

+   如何使用 IIS 部署 Flask 网站

# 使用 Python 编写 REST API

为了理解具有 JSON 响应的 REST API 的组件，我们将使用 Flask 网络框架、PostgreSQL/PostGIS 数据库，以及 SQLAlchemy 和 GeoAlchemy2 进行 ORM 查询。Flask 将用于创建 API 的 URL 端点。PostGIS 将将数据存储在由 SQLAlchemy 模型定义的表中，这些模型定义了所有列的类型，除了几何列，这些列由 GeoAlchemy2 列类型定义。

# REST

REST 是一个用于 Web 服务的标准，旨在接受请求和参数，并返回数据的表示，通常以 JSON 格式，但有时以 XML 或 HTML 格式。使用 REST 架构的 API 必须满足以下架构约束：

+   客户端-服务器交互

+   无状态

+   缓存能力

+   统一接口

+   分层系统

客户端（一个网络浏览器或远程计算机）将向指定 URL 端点的服务器发送请求。请求可以包含参数，这些参数限制了返回的数据对象，类似于 SQL 语句中的条件。它是无状态的，这意味着每个请求都必须包含请求参数，并且不能引用另一个请求的结果。返回的数据必须明确标记为可缓存或不可缓存，以便客户端决定数据是否可以存储，或者是否在需要时请求。当请求数据时，所有与数据相关的可用 API 端点（包括添加或删除数据的链接，如果有的话）都作为链接与数据表示一起返回。API 不揭示服务器的基础架构，可以在不改变 API 结构的情况下对其进行操作（添加或删除机器）。

# JSON

JSON 被设计成既能被人类理解，也能被机器解析。Python 字典可以轻松地生成 JavaScript 数据对象，因为它们使用相同的键值结构和花括号表示法。Python 内置了一个用于生成 JSON 的库（`json`模块），而像 Flask 这样的 Web 框架也包含了生成 JSON 响应的代码。

地理空间数据存在多个 JSON 标准，包括 GeoJSON 和 Esri JSON。在本章中，REST API 将使用 GeoJSON 格式来响应请求。

在这里了解更多关于 GeoJSON 的信息：[`geojson.org/`](http://geojson.org/)。

# Python 用于 REST API

Python 是编写 REST API 的绝佳语言。它包含允许进行数据库查询的模块，以及其他将 HTTP 网络请求处理成 URL 和参数组件的模块。使用这些模块，可以从数据库中检索请求的资源，并使用在 Python 字典和 JSON 对象之间进行转换的模块将数据作为 JSON 返回。

虽然可以使用标准库构建基于 Python 的 API，但使用 Web 框架构建 API 将加快开发速度，并允许根据需要添加组件模块。

# Flask

Flask 是 Python Web 框架中用于 REST API 的一个好选择。与 SQLAlchemy 和 GeoAlchemy2（见第十一章，*Flask 和 GeoAlchemy2*，了解更多关于这两个库的信息）配合使用，它允许将 REST URL 端点与一个视图（一个 Python 函数）配对，该视图将根据请求方法（例如`GET`和`POST`，仅举两个例子）以不同的方式处理请求并返回 JSON 数据。

# REST 模块

由于 Flask 被设计成可扩展的，因此有许多附加模块旨在简化 REST API 的创建。这些包括：

+   Flask-RESTful ([`flask-restful.readthedocs.io/en/latest/`](https://flask-restful.readthedocs.io/en/latest/))

+   Eve ([`python-eve.org/`](http://python-eve.org/))，它建立在 Flask 和 Cerberus 之上

+   Flask-REST-JSONAPI ([`github.com/miLibris/flask-rest-jsonapi`](https://github.com/miLibris/flask-rest-jsonapi))

本章将使用纯 Flask 功能，结合 SQLAlchemy 和 GeoAlchemy2 进行数据库查询，来展示 API 创建的基本原理。

# 其他框架

Django 和 GeoDjango（在第十二章*GeoDjango*中介绍）被广泛用于 REST API 的创建。Django 以其“内置电池”的设计理念，使得 API 开发变得容易。Django REST 框架为代码库添加了简单的 API 发布功能。

在这里探索 Django REST 框架：[`www.django-rest-framework.org/`](http://www.django-rest-framework.org/)。

# Flask URL 中的变量

当使用 Flask 进行 URL 处理时，了解如何将变量添加到 URL 中是有用的，因为每个资源都可以使用 ID 或字符串标识符（例如，一个州名）来请求。Flask URL 使用占位符将数据传递到函数参数中，并在每个端点的视图中将其用作变量。使用转换器，可以在占位符内为数值数据分配类型；默认类型是字符串类型。

# 数字转换器

在此示例中，在 URL 末尾添加了一个带有转换器的占位符，用于整数 ID。通过在占位符变量（`arena_id`）之前添加 `int:`，可以使用 `get(id)` 方法查询 `Arena` 模型/数据库表，该方法期望一个整数。如果占位符中没有指定数据类型转换器，`arena_id` 变量将包含字符串字符，并且不会被 `get(id)` 方法使用：

```py
@app.route('/nba/api/v0.1/arena/<int:arena_id>', methods=['GET'])
def get_arena(arena_id):
  arena = session.query(Arena).get(arena_id)
```

指定参数数据类型后，ORM 查询返回请求的 `arena` 对象，可以对其进行处理以生成响应。

# 其他数据转换器

除了整数，使用 `int` 转换器外，浮点数据可以使用 `float` 转换，URL 数据可以使用 `path` 转换。字符串，使用 `string` 转换器，是默认类型。在这种情况下，捕获的 `float` 值用于与 `county` 几何区域进行比较。由于此数据的 SRID 在 WKID 中，区域格式有些奇怪，但此查询将有效：

```py
@app.route('/nba/api/v0.1/county/query/size/<float:size>', methods=['GET'])
def get_county_size(size):
  counties = session.query(County).filter(County.geom.ST_Area() > size).all()
  data = [{"type": "Feature", 
  "properties":{"name":county.name,"id":county.id ,"state":county.state.name}, 
  "geometry":{"type":"MultiPolygon", 
  "coordinates":[shapely.geometry.geo.mapping(to_shape(county.geom))["coordinates"]]},
  } for county in counties]
  return jsonify({"type": "FeatureCollection","features":data})
```

在此示例中，从 URL 变量捕获的值使用 `ST_Area` 函数与 `county` 几何进行比较，该函数借鉴了 PostGIS 空间 SQL。

在这里了解更多关于 GeoAlchemy2 空间功能及其使用空间 SQL 的信息：[`geoalchemy-2.readthedocs.io/en/latest/spatial_functions.html`](http://geoalchemy-2.readthedocs.io/en/latest/spatial_functions.html)。

# 请求方法

当使用 REST API 时，可以利用多种 HTTP 请求方法。`GET` 方法用于请求数据，`POST` 方法用于添加新数据，`PUT` 方法用于更新数据，而 `DELETE` 方法用于从数据库中删除数据。

# GET

对于 Flask URL 端点，使用 `GET` 方法指定 `GET` 请求。数据可以作为参数传递，并使用 `request.args` 访问：

```py
from flask import requests, jsonify
@app.route('/nba/api/v0.1/arenas', methods=['GET'])
def get_arenas():
  if 'name' in request.args:
       arenas = session.query(Arena).filter(name=request.args['name'])
  else:
       arenas = session.query(Arena).all()
  data = [{"type": "Feature",  "properties":{"name":arena.name, "id":arena.id}, 
  "geometry":{"type":"Point","coordinates":[round(arena.longitude,6), round(arena.latitude,6)]},
  } for arena in arenas]
  return jsonify({"type": "FeatureCollection","features":data})
```

响应数据，通过列表推导式处理成 Python 字典列表，然后添加到另一个 Python 字典中，并使用 Flask 的 `jsonify` 函数转换为 JSON。

# POST

`POST` 请求携带的数据可以处理并添加到数据库中。为了区分 `POST` 请求，Flask 请求对象有一个 `method` 属性，可以检查请求方法是否为 `GET` 或 `POST`。如果我们创建一个 `form`（称为 `AddForm`）来向 `Arenas` 表添加新的 `arenas`，我们可以处理提交的 `POST` 请求数据，并使用会话管理器将其添加到数据库中：

```py
from flask import request
from .forms import AddForm
@app.route('/nba/api/v0.1/arena/add', methods=['GET', 'POST'])
def add_arenas():
  form = AddForm(request.form)
  form.name.data = "New Arena"
  form.longitude.data = -121.5
  form.latitude.data = 37.8
  if request.method == "POST":
    arena = Arena()
    arena.name = request.form['name']
    arena.longitude =float(request.form['longitude'])
    arena.latitude = float(request.form['latitude'])
    arena.geom = 'SRID=4326;POINT({0} {1})'.format(arena.longitude, arena.latitude)
    session.add(arena)
    data = [{"type": "Feature", "properties":{"name":arena.name}, 
    "geometry":{"type":"Point", 
    "coordinates":[round(arena.longitude,6), round(arena.latitude,6)]},}]
    return jsonify({'added':'success',"type": "FeatureCollection","features":data})
  return render_template('addarena.html', form=form)
```

由于此方法将接受 `GET` 和 `POST` 请求，因此根据每个请求方法发送不同的响应。

# 其他可用的请求方法

虽然 `GET` 和 `POST` 是主要请求方法，但还有其他方法可用于处理数据。对于示例 API，我们只会使用 `GET` 和 `POST` 以及 `DELETE`。

# PUT

与 `POST` 请求类似，`PUT` 请求将携带数据以更新或添加到数据库。它将尝试多次更新数据以确保更新完整传输。

# DELETE

`DELETE` 方法将从指定的端点删除资源，例如，从 `Arenas` 表中删除 `arena`。它需要一个记录标识符来指定要删除的资源：

```py
@app.route('/nba/api/v0.1/arena/delete/<int:arena_id>', methods=['DELETE'])
def delete_arena(arena_id):
  arena = session.query(Arena).delete(arena_id)
```

# REST API 应用程序

为了启用访问 NBA 体育馆、美国州、美国县和美国国会选区的数据库，我们将构建一个 REST API。该 API 将允许查询表格和特定表格资源，即数据行。它还将允许进行地理空间查询。

# 应用程序组件

此应用程序的组件包括：

+   在 第十一章 中创建的数据库，*Flask 和 GeoAlchemy2*，其中包含 NBA 体育馆、美国州、美国县和美国国会选区的表格

+   `app.py` 文件，当被 Python 可执行文件调用时，用于启动应用程序

+   `application` 文件夹，其中包含应用程序代码和文件夹

+   `__init__.py` 文件，它使 `application` 文件夹成为一个模块，定义 Flask 对象并连接到数据库

+   `views.py` 文件，它定义 API 端点、视图函数和返回的响应

+   `models.py` 文件，它定义数据库表模型为从 SQLAlchemy 继承的 Python 类

+   `forms.py` 文件，它定义 HTML 表单

+   `static` 和 `templates` 文件夹，其中包含模板和数据

# 应用程序文件夹和文件结构

示例 REST API 需要创建特定的文件和文件夹。外部文件夹称为 `arenaapp`，将包含 `app.py` 文件和名为 `application` 的文件夹。创建名为 `arenaapp` 的文件夹。在其内部，创建名为 `application` 的文件夹。在 `application` 内部，创建 `static` 和 `templates` 文件夹：

![图片](img/0ac1faae-182b-48c2-8d4b-36df91aa8089.png)

其他文件，`views.py`、`models.py` 和 `forms.py`，将位于 `application` 内部。两个文件夹 `static` 和 `templates` 将存储应用程序数据和 HTML 表单：

![图片](img/0c2891c3-592b-4374-8a1d-c5a6e57bc6ed.png)

# app.py

使用 IDE 或文本编辑器，在 `arenaapp` 中创建一个名为 `app.py` 的文件。打开此文件并添加以下行；此文件将由 Python 可执行文件运行以启动 REST API 应用程序：

```py
from application import app
app.run()
```

`__init__.py` 文件允许 `app.py` 导入 `application` 文件夹，允许调用 Flask 对象 `app` 和其 `app.run()` 方法。

# __init__.py

在 `application` 文件夹中，创建一个名为 `__init__.py` 的文件。在文件内部，添加以下代码（同时调整用户名和密码以匹配您的特定数据库凭据：

```py
import flask
app = flask.Flask(__name__)
conn_string = 'postgresql://{user}:{password}@localhost:5432/chapter11'
app.config['SQLALCHEMY_DATABASE_URI'] = conn_string 
app.config['SECRET_KEY'] = "SECRET_KEY"
import application.views
```

在此文件中，创建了 Flask 对象 `app` 并进行了配置。为了连接到数据库，使用连接字符串并将其存储在 `app.config` 字典中的 `'SQLALCHEMY_DATABASE_URI'`。记住将用户名和密码添加到连接字符串中。

# 数据库

这将连接到在 第十一章 中创建的数据库，*Flask 和 GeoAlchemy2*。它是由导入并结构化以匹配我们逐步描述的模型的 shapefiles 生成的。为了确保应用程序能够工作，请确保数据库已创建，并且 shapefiles 已导入。

# models.py

在 `models.py` 中，导入了 SQLAlchemy 和 GeoAlchemy2 模块，并初始化了数据库会话。数据库模型以其 Python 类的形式定义了其模式，允许查询和数据更新。

# 导入所需模块

这些模块使应用程序能够定义模型并连接到数据库：

```py
# The database connections and session management are managed with SQLAlchemy functions
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, relationship
# The Geometry columns of the data tables are added to the ORM using the Geometry data type
from geoalchemy2 import Geometry
```

# 声明会话

从 `app.config` 字典中，将数据库连接字符串传递给 `create_engine` 函数。一旦 `engine` 与 `sessionmaker` 绑定，就可以初始化 `session`：

```py
from application import app
# Connect to the database called chapter11 using SQLAlchemy functions
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()
```

从 `declarative_base()` 函数创建了一个名为 `Base` 的 Python 类。然后使用 `Base` 类来子类化所有应用程序类。

# 声明模型

对于模型，所有字段类型（例如，`Integer`、`String`和`Float`）都使用 SQLAlchemy ORM 列类定义，除了几何列，它们使用 GeoAlchemy2 的 `Geometry` 类。`Geometry` 类需要一个几何类型和 SRID：

```py
# Define the Arena class, which will model the Arena database table
class Arena(Base):
    __tablename__ = 'arena'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    longitude = Column(Float)
    latitude = Column(Float)
    geom = Column(Geometry(geometry_type='POINT', srid=4326))
```

`County` 类有一个主键字段和一个 `name` 字段，以及定义与 `State` 类的多对一关系的字段。它使用 `MULTIPOLYGON` 而不是 `POINT` 几何类型：

```py
# Define the County class, which will model the County database table
class County(Base):
    __tablename__ = 'county'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    state_id = Column(Integer, ForeignKey('state.id'))
    state_ref = relationship("State",backref='county')
    geom = Column(Geometry(geometry_type='MULTIPOLYGON', srid=4326))
```

`District` 类代表美国国会选区。以 `MULTIPOLYGON` 几何类型和 SRID `4326` 存储，它与 `State` 类有一个多对一关系。每个存储的 `district` 都与它所在的州相关联：

```py
# Define the District class, which will model the District database table
class District(Base):
    __tablename__ = 'district'
    id = Column(Integer, primary_key=True)
    district = Column(String)
    name = Column(String)
    state_id = Column(Integer, ForeignKey('state.id'))
    state_ref = relationship("State",backref='district')
    geom = Column(Geometry(geometry_type='MULTIPOLYGON', srid=4326))
```

`State` 类分别与 `County` 和 `District` 类有一对多关系，使用 `relationship` 函数定义。它还有一个 SRID 为 `4326` 的 `MULTIPOLYGON` 几何列：

```py
# Define the State class, which will model the State database table
class State(Base):
    __tablename__ = 'state'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    statefips = Column(String)
    stpostal = Column(String)
    counties = relationship('County', backref='state')
    districts = relationship('District', backref='state')
    geom = Column(Geometry(geometry_type='MULTIPOLYGON', srid=4326))
```

在定义了字段和关系之后，下一步是创建 REST API 端点并编写查询数据库并返回 GeoJSON 响应的视图。

# forms.py

为了捕获用户数据，例如一个新的 `arena`，将使用表单。在 `application` 文件夹内创建一个名为 `forms.py` 的文件，并添加以下代码：

```py
from flask_wtf import FlaskForm
from wtforms import TextField, FloatField
class AddForm(FlaskForm):
  name = TextField('Arena Name')
  longitude = FloatField('Longitude')
  latitude = FloatField('Latitude')
```

此代码将字段添加到模板中，这将在使用`POST`方法的章节中讨论。它将允许从 HTML 模板中输入代码并将其传递到服务器以添加新的`arena`。

# views.py

API 端点和处理包含在`views.py`中。视图在`__init__.py`中被导入，以便它们对`app`对象可用。打开一个 IDE，并在`application`文件夹中保存一个名为`views.py`的文件。

# 导入模块

为了启用 Web 请求的处理，我们需要从 Flask、GeoAlchemy2 和 Shapely（一个用于创建和处理地理空间数据的 Python 模块）导入功能。我们还将导入模型和表单：

```py
from application import app
from flask import render_template,jsonify, redirect, url_for, request, Markup
from .forms import * 
from .models import *
import geoalchemy2,shapely
from geoalchemy2.shape import to_shape
```

# 基本 URL

每个 API 模式可能不同，但通常应包括一个基本 URL，该 URL 指示 API 版本，并应链接到 API 中可用的其他端点。此应用程序将使用基本 URL 模式`nba`*`/*`api`*`/*`v0.1`。在这种情况下，主页 URL（`'/'`）将重定向到 API 的基本 URL：

```py
@app.route('/', methods=['GET'])
def get_api():
  return redirect('/nba/api/v0.1')

@app.route('/nba/api/v0.1', methods=['GET'])
def get_endpoints():
  data= [{'name':"Arena", "endpoint":"/arena"},
  {'name':"State", "endpoint":"/state"},
  {'name':"County", "endpoint":"/county"},
  {'name':"District", "endpoint":"/district"},]
  return jsonify({"endpoints":data})
```

下述各节中的端点都可通过基本 URL 获取。每个资源 URL 可以通过将资源特定的端点添加到基本 URL 来构建。

# 竞技场

要从`Arenas`表中请求数据，我们将定义 API 端点并使用视图函数查询`Arenas`模型。每个响应都将作为一个 GeoJSON 包。此端点（`'/arena'`）将返回一个 GeoJSON 响应，其内容将根据 URL 中添加的变量而变化。这些变量包括竞技场 ID 和名称。

# 获取所有竞技场

要生成包含所有`arenas`表示的响应，将使用 SQLAlchemy ORM 进行查询。为了将查询结果转换为 GeoJSON，使用列表推导式生成一个字典列表，该列表描述了从 ORM 查询返回的每个`arena`。然后，将生成的列表（`data`）添加到一个字典中，该字典使用`jsonify`函数从 Python 字典转换为 JSON 对象：

```py
@app.route('/nba/api/v0.1/arena', methods=['GET'])
def get_arenas():
  arenas = session.query(Arena).all()
  data = [{"type": "Feature", "properties":{"name":arena.name, "id":arena.id}, 
  "geometry":{"type":"Point", "coordinates":[round(arena.longitude,6),               round(arena.latitude,6)]},
  } for arena in arenas]
  return jsonify({"type": "FeatureCollection","features":data})
```

返回的字段包括`name`和`id`，以及`longitude`和`latitude`。为了限制传输的数据量，`latitude`和`longitude`被四舍五入到`6`位小数。描述`arena`位置所需的精度较低，这使得这种限制是合理的。由于点数据类型只包含两个点，因此返回的数据较少，而多边形和多线数据则更大，需要更高的精度。

与循环相比，列表推导式可以减少迭代列表所需的处理时间。了解更多关于列表推导式的内容：

[`docs.python.org/3/tutorial/datastructures.html#list-comprehensions. `](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions)

# 通过 ID 获取竞技场

通过向`arena`端点添加一个数字 ID，可以定位并返回特定的`arena`。使用`session.query`方法的`get`来检索请求的`arena`对象：

```py
@app.route('/nba/api/v0.1/arena/<int:arena_id>', methods=['GET'])
def get_arena(arena_id):
  arena = session.query(Arena).get(arena_id)
  data = [{"type": "Feature",  "properties":{"name":arena.name, "id":arena.id},  "geometry":{"type":"Point", "coordinates":[round(arena.longitude,6), round(arena.latitude,6)]}, 
  return jsonify({"type": "FeatureCollection","features":data})
```

选定的`arena`被添加到列表中的一个字典中，然后该字典被添加到另一个字典中，并作为 JSON `data`返回。

# 通过名称获取竞技场

可以通过`name`在此端点请求`arena`。通过利用称为`filter`的查询条件，可以检索与提供的`name`匹配的`arena`。为了增加灵活性，使用`like`运算符（以及`"%"`通配符运算符）使输入的`arena` `name`完整。相反，输入的字符串将用于`filter`查询，并仅返回名称以输入的字符串开头的`arena`对象：

```py
@app.route('/nba/api/v0.1/arena/<arena_name>', methods=['GET'])
def get_arena_name(arena_name):
  arenas = session.query(Arena).filter(Arena.name.like(arena_name+"%")).all()
  data = [{"type": "Feature",  "properties":{"name":arena.name,"id":arena.id}, 
  "geometry":{"type":"Point",  "coordinates":[round(arena.longitude,6), round(arena.latitude,6)]}, 
  } for arena in arenas]
  return jsonify({"type": "FeatureCollection","features":data})
```

使用列表推导式生成`arena`字典。以下是对`arena`端点进行字符串查询的响应示例：

![](img/5c9a2ed6-0822-4ca2-a9a5-336bedb5c3c5.png)

# 地理空间查询

通过添加一个额外的 URL 组件，API 被启用空间查询。传递一个`arena` ID 并添加`"/intersect"`将使用空间查询来查找描述请求的 NBA Arena 的数据。在此视图函数中，使用`intersect` `filter`（即，使用点在多边形函数中识别包含`arena`的`county`）查询`County`和`District`表。使用`county`和`state`之间的表关系检索基础州。返回所有几何形状和选定字段：

```py
@app.route('/nba/api/v0.1/arena/<int:arena_id>/intersect', methods=['GET'])
def arena_intersect(arena_id):
  arena = session.query(Arena).get(arena_id)
  county = session.query(County).filter(County.geom.ST_Intersects(arena.geom)).first()
  district=session.query(District).filter(District.geom.ST_Intersects(arena.geom))
  district = district.first()
  if county != None:
    data = [{"type": "Feature", "properties": {"name":arena.name, "id":arena.id,} ,
    "geometry":{"type":"Point", "coordinates":[round(arena.longitude,6), round(arena.latitude,6)]}, 
    },{"type": "Feature", "properties": {"name":county.name, "id":county.id,} ,
    "geometry":{"type":"MultiPolygon", 
    "coordinates":[shapely.geometry.geo.mapping(to_shape(county.geom))]}, 
    },{"type": "Feature", "properties": {"name":district.district, "id":district.id,},
    "geometry":{"type":"MultiPolygon", 
    "coordinates":[shapely.geometry.geo.mapping(to_shape(district.geom))]}, 
    },{"type": "Feature", "properties": {"name":county.state_ref.name, "id":county.state_ref.id,}, "geometry":{"type":"MultiPolygon", 
    "coordinates":[shapely.geometry.geo.mapping(to_shape(county.state_ref.geom))]}, 
    }]
    return jsonify({"type": "FeatureCollection","features":data})
  else:
    return redirect('/nba/api/v0.1/arena/' + str(arena_id))
```

为了确保函数有效，`if`条件检查`arena`是否在美`county`内；如果不是，则不使用`county`、`district`和`state`对象。相反，请求被重定向到非地理空间查询视图函数。

# 州

由于每个`state`由许多顶点组成，因此美国州数据可能很大。在`states`端点的端点中，我们将添加一些 URL 参数，使我们能够决定是否返回每个请求的`state`的几何形状。

# 获取所有州

通过在`request.args`字典中检查 URL 参数，然后检查该参数是否评估为真，我们可以确定是否应返回所有`state`几何形状。GeoJSON 响应是通过使用`to_shape`函数和`shapely.geometry.geo.mapping`函数（简称为`smapping`）从州的几何形状生成的：

```py
@app.route('/nba/api/v0.1/state', methods=['GET'])
def get_states():
  smapping = shapely.geometry.geo.mapping
  states = session.query(State).all()
  data = [{"type": "Feature", 
  "properties":{"state":state.name,"id":state.id}, 
  "geometry":{"type":"MultiPolygon", 
  "coordinates":"[Truncated]"},
  } for state in states]
  if "geometry" in request.args.keys():
    if request.args["geometry"]=='1' or request.args["geometry"]=='True':
      data = [{"type": "Feature", 
      "properties":{"state":state.name,"id":state.id}, 
      "geometry":{"type":"MultiPolygon", 
      "coordinates":[smapping(to_shape(state.geom))["coordinates"]]},
      } for state in states]
  return jsonify({"type": "FeatureCollection","features":data})
```

如果未包含`geometry`参数或参数，则几何形状将表示为截断。

# 通过 ID 获取州

要使用`state`的 ID 作为主键来获取特定的`state`，我们可以添加一个 URL 变量，该变量将检查整数 ID。它将`geometry`作为`geojson`返回：

```py
@app.route('/nba/api/v0.1/state/<int:state_id>', methods=['GET'])
def get_state(state_id):
  state = session.query(State).get(state_id)
  geojson = shapely.geometry.geo.mapping(to_shape(state.geom))
  data = [{"type": "Feature",  "properties":{"name":state.name}, 
  "geometry":{"type":"MultiPolygon",  "coordinates":[geojson["coordinates"]]},
  }]
  return jsonify({"type": "FeatureCollection","features":data})
```

# 通过名称获取州

使用`filter`将允许将 URL 变量用作`query` `filter`。字符串变量将与数据库表中的状态`name`字段进行比较，并使用`like`运算符进行模糊比较（即，如果`state_name`变量为`'M'`，则将获取所有以`'M'`开头的`states`）：

```py
@app.route('/nba/api/v0.1/state/<state_name>', methods=['GET'])
def get_state_name(state_name):
  states = session.query(State).filter(State.name.like(state_name+"%")).all()
  geoms = {state.id:smapping(to_shape(state.geom)) for state in states}
  data = [{"type": "Feature", "properties":{"state":state.name}, 
  "geometry":{"type":"MultiPolygon", 
  "coordinates":[shapely.geometry.geo.mapping(to_shape(state.geom)["coordinates"]]},
  } for state in states]
  return jsonify({"type": "FeatureCollection","features":data})
```

此函数没有 URL 参数，并将返回指定字段和选定州的`geometry`。

# 通过州获取区域

此函数使用空间分析来查找所有被`state`包含的`arenas`。`state`通过 ID 识别，并且 URL 组件将选择所有`geometry`在`state` `geometry`内的`arenas`：

```py
@app.route('/nba/api/v0.1/state/<int:state_id>/contains', methods=['GET'])
def get_state_arenas(state_id):
  state = session.query(State).get(state_id)
  shp = to_shape(state.geom)
  geojson = shapely.geometry.geo.mapping(shp)
  data = [{"type": "Feature", "properties":{"name":state.name}, 
  "geometry":{"type":"MultiPolygon", "coordinates":[geojson]},
  }]
  arenas = session.query(Arena).filter(state.geom.ST_Contains(arena.geom))
  data_arenas =[{"type": "Feature",
  "properties":{"name":arena.name}, "geometry":{"type":"Point", 
  "coordinates":[round(arena.longitude,6), round(arena.latitude,6)]}, 
  } for arena in arenas]
  data.extend(data_arenas)
  return jsonify({"type": "FeatureCollection","features":data})
```

返回的数据将包括州`data`和所有`arenas`的`data`，因为 GeoJSON 允许将多个数据类型打包为特征集合。

# 县

与`State`数据库表类似，这将检索所有的`county`数据。它接受一个`geometry`参数来决定是否返回每个`county`的`geometry`：

```py
@app.route('/nba/api/v0.1/county', methods=['GET'])
def get_counties():
  counties = session.query(County).all()
  geoms = {county.id:smapping(to_shape(county.geom)) for county in counties}
 if 'geometry' in request.args.keys():
      data = [{"type": "Feature", 
      "properties":{"name":county.name, "state":county.state.name}, 
      "geometry":{"type":"MultiPolygon", 
  "coordinates":[shapely.geometry.geo.mapping(to_shape(state.geom)["coordinates"]]},
       } for county in counties]
 else:
      data = [{"type": "Feature", 
      "properties":{"name":county.name, "state":county.state.name}, 
      "geometry":{"type":"MultiPolygon", 
  "coordinates":["Truncated"]},
       } for county in counties]
  return jsonify({"type": "FeatureCollection","features":data})
```

# 通过 ID 获取县

在使用`get_counties`函数检索所有县之后，可以将特定`county`的 ID 传递给此函数。使用`session.query.County).get(county_id)`可以检索感兴趣的`county`：

```py
@app.route('/nba/api/v0.1/county/<int:county_id>', methods=['GET'])
def get_county(county_id):
  county = session.query(County).get(county_id)
  shp = to_shape(county.geom)
  geojson = shapely.geometry.geo.mapping(shp)
  data = [{"type": "Feature",
  "properties":{"name":county.name, "state":county.state.name}, 
  "geometry":{"type":"MultiPolygon", 
  "coordinates":[geojson]},
  }]
  return jsonify({"type": "FeatureCollection","features":data})
```

# 通过名称获取县

再次，我们可以使用 URL 变量收集一个字符串，并使用提供的字符串作为查询过滤器。如果使用`Wash`作为 URL 变量`county_name`，查询将找到所有以`Wash`开头的`counties`：

```py
@app.route('/nba/api/v0.1/county/<county_name>', methods=['GET'])
def get_county_name(county_name):
  counties = session.query(County).filter(County.name.like(county_name+"%")).all()
  data = [{"type": "Feature", 
  "properties":{"name":county.name, "state":county.state.name}, 
  "geometry":{"type":"MultiPolygon", 
  "coordinates":[shapely.geometry.geo.mapping(to_shape(county.geom))["coordinates"]]},
  } for county in counties]
  return jsonify({"type": "FeatureCollection","features":data})
```

可以在空间字段以及非空间字段上使用`filter`方法。

# 区域

可以将区域类似地添加到 API 中。在这种情况下，我们将添加一个几何参数来决定是否返回几何形状。这允许请求的机器或浏览器获取所有区域及其 ID，这些 ID 可以在下一节中用于获取单个区域，或者根据需要一次性获取所有数据。

# 获取所有区域

此端点（`'/district'`），将使用`session.query(District).all()`查询`District`模型：

```py
@app.route('/nba/api/v0.1/district', methods=['GET'])
def get_districts():
  districts = session.query(District).all()
  if 'geometry' in request.args.keys() and request.args['geometry'] in ('1','True'):
    data = [{"type": "Feature", 
    "properties":{"representative":district.name, "district":district.district,
 "state": district.state_ref.name, "id":district.id}, 
    "geometry":{"type":"MultiPolygon", 
    "coordinates":shapely.geometry.geo.mapping(to_shape(district.geom))["coordinates"]},
    } for district in districts]
  else:
    data = [{"type": "Feature", 
    "properties":{"representative":district.name, "district":district.district,
    "state": district.state_ref.name, "id":district.id}, 
    "geometry":{"type":"MultiPolygon", 
    "coordinates":["Truncated"]},
    } for district in districts]
  return jsonify({"type": "FeatureCollection","features":data})
```

# 通过 ID 获取区域

传递整数`district` ID 将仅返回请求的`district`表示。`geometry`使用`shapely`和`geoalchemy2.shape`中的`to_shape`方法转换为 GeoJSON 格式：

```py
@app.route('/nba/api/v0.1/district/<int:district_id>', methods=['GET'])
def get_district(district_id):
  district = session.query(District).get(district_id)
 shp = to_shape(district.geom)
 geojson = shapely.geometry.geo.mapping(shp)
  data = [{"type": "Feature",
  "properties":{"district":district.district,"id":district.id}, 
  "geometry":{"type":"MultiPolygon", 
  "coordinates":[geojson['coordinates']]},
  }]
  return jsonify({"type": "FeatureCollection","features":data})
```

# 通过名称获取区域

在这种情况下，区域的`name`是国会区域编号。有一个`name`字段，但它包含该区域的当选代表的姓名：

```py
@app.route('/nba/api/v0.1/district/<dist>', methods=['GET'])
def get_district_name(dist):
  districts = session.query(District).filter(District.district.like(dist+"%")).all()
  data = [{"type": "Feature", 
  "properties":{"district":district.district,"id":district.id, 
  "representative":district.name},   "geometry":{"type":"MultiPolygon", 
  "coordinates":shapely.geometry.geo.mapping(to_shape(district.geom))["coordinates"]},
  } for district in districts]
  return jsonify({"type": "FeatureCollection","features":data})
```

所有这些方法都可以调整以包含更多参数。尝试添加检查返回字段的条件或另一个条件。所有 URL 参数参数都添加在查询后的问号（`'?'`）之后。

# API POST 端点

使用 JSON 数据和 HTML 表单都可以添加`arena`。在本节中，我们将创建一个 HTML 模板，使用`forms.py`中的`AddForm`，并使用它从第十二章，*GeoDjango*代码包中包含的`Leaflet.js`地图收集数据。它还使用 jQuery 库允许用户点击地图上的任何位置，从而更新地图的`longitude`和`latitude`数据：

![图片](img/3d47507c-fec2-4274-b8c2-21b86bfeff40.png)

# 新区域

要将新的 `arena` 添加到数据库的 `Arena` 表中，将创建一个用于处理请求的视图函数和一个 Jinja2 HTML 模板，并将其使用。该函数将确定请求方法，并将适当的响应发送给请求。如果是一个 `GET` 请求，它将发送一个包含 `AddForm` 表单的 HTML 模板。从 HTML 模板中，填写数据并点击按钮将提交一个 `POST` 请求，该请求将转到相同的视图函数，并使用提交的数据在 `Arena` 表中添加一行新数据。

# 视图函数

处理请求的视图函数接受 `GET` 和 `POST` 请求方法。在这种情况下使用的是端点 `'/add'`，尽管它可以是任何与 `arena` 端点区分开来的内容：

```py
@app.route('/nba/api/v0.1/arena/add', methods=['GET', 'POST'])
def add_arenas():
  form = AddForm(request.form)
  form.name.data = "New Arena"
  form.longitude.data = -121.5
  form.latitude.data = 37.8
  if request.method == "POST":
    arena = Arena()
    arena.name = request.form['name']
    arena.latitude = float(request.form['latitude'])    
    arena.longitude = float(request.form['longitude'])
    arena.geom = 'SRID=4326;POINT({0} {1})'.format(arena.longitude, arena.latitude)
    session.add(arena)
    data = [{"type": "Feature", "properties":{"name":arena.name}, 
    "geometry":{"type":"Point", 
    "coordinates":[round(arena.longitude,6), round(arena.latitude,6)]},}]
    return jsonify({'added':'success',"type": "FeatureCollection","features":data})
  return render_template('addarena.html', form=form)
```

一旦按下按钮，数据就会被提交。视图函数将根据请求方法确定要执行的操作——如果是一个 `POST` 请求，则 `form` 中提交的数据将用于创建一个新的 `arena` 对象，并且会话管理器将保存该对象，并将其添加到数据库中。

# addarena.html 的头部

接下来，让我们创建一个名为 `addarena.html` 的模板，该模板将被添加到 `application` 文件夹内的 `templates` 文件夹中。在 HTML 文件的顶部，在头部部分，添加 CSS、JavaScript 和 jQuery 库：

```py
<!DOCTYPE html>
<html>
<head>
  <title>Arena Map</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <link rel="stylesheet" href="https://unpkg.com/leaflet@1.2.0/dist/leaflet.css" />
 <script src="img/leaflet.js"></script>
    <script src="img/jquery.min.js"></script>
</head>
```

# addarena.html 的脚本

创建地图 `<div>` 部分，并添加将启用地图交互的 JavaScript。如果地图被点击，JavaScript 函数 `showMapClick`（它接受一个事件 `*e*` 作为参数）将移动标记。在函数内部，使用 jQuery 设置 `latitude` 和 `longitude` `form` 元素的值，从事件参数的 `e.latlng` 方法获取值：

```py
<body>
<div id="map" style="width: 600px; height: 400px;"></div>
<script>
  var themap = L.map('map').setView([ {{form.latitude.data}},{{form.longitude.data}}], 13);
  L.tileLayer('https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.{ext}', {
  subdomains: 'abcd',
  minZoom: 1,
  maxZoom: 18,
  ext: 'png'
  }).addTo(themap);
  marker = L.marker([ {{form.latitude.data}},{{form.longitude.data}}]).addTo(themap)
    .bindPopup("Click to locate the new arena").openPopup();
  var popup = L.popup();
  function showMapClick(e) {
 $('#longitude').val(e.latlng.lng);
 $('#latitude').val(e.latlng.lat);
    marker
      .setLatLng(e.latlng)
      .bindPopup("You added a new arena at " + e.latlng.toString())
      .openPopup();
  }
  themap.on('click', showMapClick);
</script>
```

# addarena.html 表单

`form` 数据将以 `POST` 方法提交。一旦按下添加 `arena` 按钮，表单中的数据就会被提交：

```py
  <form method="post" class="form">
    Name: {{form.name}}<br>
    Longitude: {{ form.longitude(class_ = 'form-control first-input last-input', placeholder = form.longitude.data, ) }} <br>
    Latitude: {{ form.latitude(class_ = 'form-control first-input last-input', placeholder = form.latitude.data, ) }} <br>
    <input type="submit" value="Add Arena">
  </form>
</body>
</html>
```

点击按钮将数据提交给视图函数。数据将被处理，并返回一个成功的 JSON 消息：

![图片](img/b9892c1f-af5c-4d01-9343-aef4bc325fc3.png)

# 使用 requests 库发送 POST 请求

可以使用网络请求添加一个新的 `arena`，从而避免使用 HTML 模板。以下是一个使用 `requests` 库进行请求的演示：

```py
>>> form = {'longitude':'-109.5', 'latitude':'40.7', 'name':'Test Arena'}
>>> requests.post('http://127.0.0.1:5000/nba/api/v0.1/arena/add', form)
<Response [200]>
```

将 `POST` 请求发送到 `'/add'` 端点，同时附带所需的 `form` 参数，作为一个 Python 字典。

# 删除 arena

删除一个 `arena`（或另一个资源）也可以使用视图函数和特定的端点来完成：

```py
@app.route('/nba/api/v0.1/arena/delete/<int:arena_id>', methods=['DELETE'])
def delete_arena(arena_id):
  arena = session.query(Arena).delete(arena_id)
  return jsonify({"deleted":"success"})
```

要删除一个 `arena`，请使用 `delete` 方法发送请求：

```py
>>> import requests
>>>requests.delete('http://127.0.0.1:5000/nba/api/v0.1/arena/delete/30')
```

# 在本地运行 REST API

要在本地运行此 API 应用程序，将 `app.py` 脚本传递给 Python 可执行文件。这将启动本地机器上的内置 Web 服务器：

```py
C:\Projects\Chapter13\arenaapp>python app.py
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

一旦服务器启动，导航到 API 端点以从视图函数获取响应。然而，如果应用程序是完整的，本地的服务器将不足以处理 API 请求。因此，需要在生产网络服务器上进行部署。

# 将 Flask 部署到 IIS

要在安装了 **Internet Information Services** (**IIS**) 的 Microsoft 服务器上部署新的 API 应用程序，我们需要下载一些 Python 代码，以及一个名为 **FastCGI** 的 IIS 模块。一旦配置完成，应用程序将能够响应来自任何允许的机器的网页请求。

# Flask 和网络服务器

虽然 Flask 包含一个用于测试目的的本地网络服务器，但它不是为生产部署设计的。Flask 与 Apache 或 IIS 等网络服务器配合使用效果最佳。虽然关于如何使用 Apache 部署 Flask 的文献很多，但关于如何使用 IIS 部署它的良好说明却很少见。由于大多数 GIS 专业人士使用 Windows 服务器或可以访问它们，因此这些说明将侧重于 IIS 7 的部署。

# WSGI

**Web 服务器网关接口** (**WSGI**) 是一个 Python 规范，它允许使用 Python 可执行文件来响应网页请求。WSGI 集成在 Flask 和 Django 等网络框架中。

要启用 Flask 网络框架来服务网页，需要对 IIS 进行一些配置，包括安装一个名为 FastCGI 的 IIS **通用网关接口** (**CGI**) 模块，以及安装一个名为 **WFastCGI** 的 Python 模块。有了这两个添加项，IIS 网络服务器将连接到 API 应用程序背后的代码。

# 安装 WFastCGI 模块和 FastCGI

使用可在此处找到的 Web 平台安装程序：[`www.microsoft.com/web/downloads/platform.aspx`](http://www.microsoft.com/web/downloads/platform.aspx)（如果尚未安装）。使用右上角的搜索栏，输入 `WFastCGI`。搜索结果将出现，并列出适用于 Python 2.x 和 Python 3.x 的可用 WFastCGI 版本。选择 Python 3.6 的版本并运行安装程序。

此安装向所需技术堆栈添加了两个重要组件。FastCGI 模块被添加到 IIS 中，WFastCGI Python 代码被添加到一个新的 Python 安装中。这个新安装将位于 `C:\Python36`，除非该位置已存在版本（不包括 ArcGIS10.X Python 安装中的 Python 版本）。

在这个新安装中，`C:\Python36\Scripts`（或等效）文件夹中添加了一个名为 `wfastcgi.py` 的文件。此文件应复制到站点文件夹中，紧挨着 `app.py` 文件。

# 配置 FastCGI

打开 IIS，点击默认网站。在内容窗格的功能视图中，选择处理器映射图标。双击打开它。从右侧窗格中选择添加模块映射。

+   在请求路径条目中添加一个星号 (`*`)。

+   从模块选择列表中选择 FastCGI 模块。

+   如果您已将 `wfastcgi.py` 文件复制到代码路径，并且代码位于 `C:\website`，请在可执行输入框中输入以下内容：`C:\Python36\python.exe|C:\website\wfastcgi.py`。

+   可选地，可以使用 `Scripts` 文件夹中的 `wfastcgi.py` 文件。设置如下：`C:\Python36\python.exe|C:\Python36\Scripts\wfastcgi.py`。

+   点击请求限制并取消选中仅当请求映射到时调用处理器的复选框。点击确定。

+   在添加模块映射界面中点击确定。

+   在确认对话框中点击是。

# 根服务器设置和环境变量

前往根服务器设置并点击 FastCGI 设置图标。双击与上一节中添加的路径匹配的参数。将打开编辑 FastCGI 应用程序界面。

+   点击环境变量（集合）条目。将出现省略号（...）。双击省略号以编辑环境变量。

+   点击添加按钮以添加一个新变量。

+   在名称输入框中添加 `PYTHONPATH`。

+   将网站代码的路径（例如 `C:\website\`）添加到值输入框中。

+   点击添加按钮以添加第二个变量。

+   在名称输入框中添加 `WSGI_HANDLER`。

+   如果网站由名为 `app.py` 的文件控制，请在值输入框中添加 `app.app`（将 `.py` 替换为 `.app`）。

+   变量添加完成后，点击确定。在编辑 FastCGI 应用程序中点击确定。

网站现在应该已经上线。使用浏览器导航到 REST 端点以确认网站按预期加载。

# 摘要

使用 Python Web 框架创建具有 REST 规范的 API 很容易。Flask 使得协调 URL 端点和请求方法以及响应类型变得简单。凭借内置的 JSON 功能，以及使用 SQLAlchemy 和 GeoAlchemy2 ORM，Flask 是创建地理空间 REST API 的完美框架。

在下一章中，我们将介绍使用 CARTOframes 模块进行地理空间数据的云可视化。
