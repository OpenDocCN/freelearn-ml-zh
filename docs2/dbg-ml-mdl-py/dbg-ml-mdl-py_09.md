

# 第九章：生产测试和调试

您可能对训练和测试机器学习模型感到兴奋，而没有考虑到模型在生产中的意外行为以及您的模型如何融入更大的技术。大多数学术课程不会详细介绍测试模型、评估其质量以及在生产前和生产中监控其性能的策略。在本章中，我们将回顾测试和调试生产中模型的重要概念和技术。

在本章中，我们将涵盖以下主题：

+   基础设施测试

+   机器学习管道的集成测试

+   监控和验证实时性能

+   模型断言

到本章结束时，您将了解基础设施和集成测试的重要性，以及模型监控和断言。您还将了解如何使用 Python 库，以便在项目中从中受益。

# 技术要求

在本章中，以下要求应予以考虑，因为它们将帮助您更好地理解概念，在项目中使用它们，并使用提供的代码进行实践：

+   Python 库要求：

    +   `sklearn` >= 1.2.2

    +   `numpy` >= 1.22.4

    +   `pytest` >= 7.2.2

+   您还必须具备机器学习生命周期的基础知识。

您可以在 GitHub 上找到本章的代码文件，地址为[`github.com/PacktPublishing/Debugging-Machine-Learning-Models-with-Python/tree/main/Chapter09`](https://github.com/PacktPublishing/Debugging-Machine-Learning-Models-with-Python/tree/main/Chapter09)。

# 基础设施测试

基础设施测试是指验证和验证部署、管理和扩展机器学习模型所涉及的各个组件和系统的过程。这包括测试软件、硬件和其他构成支持机器学习工作流程的基础设施的资源。机器学习中的基础设施测试有助于您确保模型得到有效训练、部署和维护。它为您在生产环境中提供可靠的模型。定期的基础设施测试可以帮助您及早发现和修复问题，并降低部署和生产阶段失败的风险。

这里是机器学习基础设施测试的一些重要方面：

+   **数据管道测试**：这确保了负责数据收集、选择和整理的数据管道正在正确且高效地工作。这有助于保持训练、测试和部署机器学习模型的数据质量和一致性。

+   **模型训练和评估**：这验证了模型训练过程的功能，例如超参数调整和模型评估。这个过程消除了训练和评估中的意外问题，以实现可靠和负责任的模式。

+   **模型部署和托管**：这项测试用于检查在生产环境中部署训练好的模型的过程，确保服务基础设施，如 API 端点，能够正确工作并能处理预期的请求负载。

+   **监控和可观察性**：这项测试用于检查提供对机器学习基础设施性能和行为洞察的监控和日志系统。

+   **集成测试**：这项测试验证机器学习基础设施的所有组件，如数据管道、模型训练系统和部署平台，是否能够无缝且无冲突地协同工作。

+   **可伸缩性测试**：这项测试评估基础设施根据不断变化的需求（如增加的数据量、更高的用户流量或更复杂的模型）进行扩展或缩减的能力。

+   **安全和合规性测试**：这项测试确保机器学习基础设施满足必要的网络安全要求、数据保护法规和隐私标准。

现在你已经了解了基础设施测试的重要性和好处，你准备好学习相关的工具，这些工具可以帮助你在模型部署和基础设施管理中。

## 基础设施即代码工具

**基础设施即代码**（**IaC**）和配置管理工具，如**Chef**、**Puppet**和**Ansible**，可以用于自动化软件和硬件基础设施的部署、配置和管理。这些工具可以帮助我们确保在不同环境中的一致性和可靠性。让我们了解 Chef、Puppet 和 Ansible 是如何工作的，以及它们如何能帮助你在项目中：

+   **Chef** ([`www.chef.io/products/chef-infrastructure-management`](https://www.chef.io/products/chef-infrastructure-management))：Chef 是一个开源配置管理工具，它依赖于客户端-服务器模型，其中 Chef 服务器存储所需的配置，Chef 客户端将其应用于节点。

+   **Puppet** ([`www.puppet.com/`](https://www.puppet.com/))：Puppet 是另一个开源配置管理工具，它以客户端-服务器模式或作为独立应用程序工作。Puppet 通过定期从 Puppet 主服务器拉取配置来在节点上强制执行所需的配置。

+   **Ansible** ([`www.ansible.com/`](https://www.ansible.com/))：Ansible 是一个开源且易于使用的配置管理、编排和自动化工具，它与节点通信并应用配置。

这些工具主要关注基础设施管理和自动化，但它们也具有模块或插件，可以执行基础设施的基本测试和验证。

## 基础设施测试工具

Test Kitchen、ServerSpec 和 InSpec 是我们可以使用的基础设施测试工具，以验证和验证我们基础设施所需配置和行为：

+   **Test Kitchen** ([`github.com/test-kitchen/test-kitchen`](https://github.com/test-kitchen/test-kitchen))：Test Kitchen 是一个主要用于与 Chef 一起使用的集成测试框架，但也可以与其他 IaC 工具（如 Ansible 和 Puppet）一起工作。它允许你在不同的平台和配置上测试你的基础设施代码。Test Kitchen 在各种平台上创建临时实例（使用 Docker 或云提供商等驱动程序），合并你的基础设施代码，并对配置的实例运行测试。你可以使用 Test Kitchen 与不同的测试框架（如 ServerSpec 或 InSpec）一起定义你的测试。

+   **ServerSpec** ([`serverspec.org/`](https://serverspec.org/))：ServerSpec 是一个用于基础设施的 **行为驱动开发**（**BDD**）测试框架。它允许你用人类可读的语言编写测试。ServerSpec 通过在目标系统上执行命令并检查输出与预期结果是否一致来测试你基础设施的期望状态。你可以使用 ServerSpec 与 Test Kitchen 或其他 IaC 工具一起确保你的基础设施配置正确。

+   **InSpec** ([`github.com/inspec/inspec`](https://github.com/inspec/inspec))：InSpec 是由 Chef 开发的开源基础设施测试框架。它以人类可读的语言定义测试和合规性规则。你可以独立运行 InSpec 测试，或者与 Test Kitchen、Chef 或其他 IaC 平台等工具一起运行。

这些工具确保我们的 IaC 和配置管理设置在部署前按预期工作，以实现跨不同环境的一致性和可靠性。

## 使用 Pytest 进行基础设施测试

我们还可以使用 Pytest，这是我们上一章中用于单元测试的工具，也可以用于基础设施测试。假设我们编写了应该在名为 `test_infrastructure.py` 的 Python 文件中以 `test_` 前缀开始的测试函数。我们可以使用 Python 库（如 `paramiko`、`requests` 或 `socket`）与我们的基础设施交互（例如，进行 API 调用、连接到服务器等）。例如，我们可以测试 Web 服务器是否以状态码 200 响应：

```py
import requestsdef test_web_server_response():
    url = "http://your-web-server-url.com"
    response = requests.get(url)
    assert response.status_code == 200,
        f"Expected status code 200,
        but got {response.status_code}"
```

然后，我们可以运行上一章中解释的测试。

除了基础设施测试之外，其他技术可以帮助你为成功部署模型做准备，例如集成测试，我们将在下一章中介绍。

# 机器学习管道的集成测试

当我们训练一个机器学习模型时，我们需要评估它与其所属的更大系统其他组件的交互效果。集成测试帮助我们验证模型在整体应用程序或基础设施中是否正确工作，并满足预期的性能标准。在我们的机器学习项目中，以下是一些重要的集成测试组件：

+   **测试数据管道**：我们需要评估在模型训练之前的数据预处理组件（如数据整理）在训练和部署阶段之间的一致性。

+   **测试 API**：如果我们的机器学习模型通过 API 公开，我们可以测试 API 端点以确保它正确处理请求和响应。

+   **测试模型部署**：我们可以使用集成测试来评估模型的部署过程，无论它是作为独立服务、容器内还是嵌入在应用程序中部署。这个过程有助于我们确保部署环境提供必要的资源，例如 CPU、内存和存储，并且模型在需要时可以更新。

+   **测试与其他组件的交互**：我们需要验证我们的机器学习模型与数据库、用户界面或第三方服务无缝工作。这可能包括测试模型预测在应用程序中的存储、显示或使用方式。

+   **测试端到端功能**：我们可以使用模拟真实场景和用户交互的端到端测试来验证模型的预测在整体应用程序的上下文中是准确的、可靠的和有用的。

我们可以从集成测试中受益，以确保在实际应用程序中的平稳部署和可靠运行。我们可以使用几个工具和库来为我们的 Python 机器学习模型创建健壮的集成测试。*表 9.1*显示了集成测试的一些流行工具：

| **工具** | **简要描述** | **URL** |
| --- | --- | --- |
| Pytest | 一个在 Python 中广泛用于单元和集成测试的框架 | [`docs.pytest.org/en/7.2.x/`](https://docs.pytest.org/en/7.2.x/) |
| Postman | 一个 API 测试工具，用于测试机器学习模型与 RESTful API 之间的交互 | [`www.postman.com/`](https://www.postman.com/) |
| Requests | 一个 Python 库，通过发送 HTTP 请求来测试 API 和服务 | [`requests.readthedocs.io/en/latest/`](https://requests.readthedocs.io/en/latest/) |
| Locust | 一个允许你模拟用户行为并测试机器学习模型在各种负载条件下的性能和可扩展性的负载测试工具 | [`locust.io/`](https://locust.io/) |
| Selenium | 一个浏览器自动化工具，你可以用它来测试利用机器学习模型的 Web 应用程序的端到端功能 | [`www.selenium.dev/`](https://www.selenium.dev/) |

表 9.1 – 集成测试的流行工具

## 使用 pytest 进行集成测试

在这里，我们想要练习使用`pytest`对一个具有两个组件的简单 Python 应用程序进行集成测试：一个数据库和一个服务，它们都从数据库中检索数据。让我们假设我们有`database.py`和`service.py`脚本文件：

database.py:

```py
class Database:    def __init__(self):
        self.data = {"users": [{"id": 1,
            "name": "John Doe"},
            {"id": 2, "name": "Jane Doe"}]}
    def get_user(self, user_id):
        for user in self.data["users"]:
            if user["id"] == user_id:
                return user
            return None
```

service.py:

```py
from database import Databaseclass UserService:
    def __init__(self, db):
        self.db = db
    def get_user_name(self, user_id):
        user = self.db.get_user(user_id)
        if user:
            return user["name"]
        return None
```

现在，我们将使用 `pytest` 编写一个集成测试，以确保 `UserService` 组件与 `Database` 组件正确工作。首先，我们需要在名为 `test_integration.py` 的测试脚本文件中编写我们的测试，如下所示：

```py
import pytestfrom database import Database
from service import UserService
@pytest.fixture
def db():
    return Database()
@pytest.fixture
def user_service(db):
    return UserService(db)
def test_get_user_name(user_service):
    assert user_service.get_user_name(1) == "John Doe"
    assert user_service.get_user_name(2) == "Jane Doe"
    assert user_service.get_user_name(3) is None
```

定义好的 `test_get_user_name` 函数通过检查 `get_user_name` 方法是否为不同的用户 ID 返回正确的用户名来测试 `UserService` 和 `Database` 组件之间的交互。

要运行测试，我们可以在终端中执行以下命令：

```py
pytest test_integration.py
```

### 使用 pytest 和 requests 进行集成测试

我们可以将 `requests` 和 `pytest` Python 库结合起来，对我们的机器学习 API 进行集成测试。我们可以使用 `requests` 库发送 HTTP 请求，并使用 `pytest` 库编写测试用例。假设我们有一个机器学习 API，其端点如下：

```py
POST http://mldebugging.com/api/v1/predict
```

在这里，API 接受一个包含输入数据的 JSON 有效负载：

```py
{    "rooms": 3,
    "square_footage": 1500,
    "location": "suburban"
}
```

这将返回一个包含预测价格的 JSON 响应：

```py
{    "predicted_price": 700000
}
```

现在，我们需要创建一个名为 `test_integration.py` 的测试脚本文件：

```py
import requestsimport pytest
API_URL = "http://mldebugging.com/api/v1/predict"
def test_predict_house_price():
    payload = {
        "rooms": 3,
        "square_footage": 1500,
        "location": "suburban"
    }
    response = requests.post(API_URL, json=payload)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"
    json_data = response.json()
    assert "predicted_price" in json_data
    assert isinstance(json_data["predicted_price"],
        (int, float))
```

要运行测试，我们可以在终端中执行以下命令：

```py
pytest test_integration.py
```

在这个例子中，我们定义了一个名为 `test_predict_house_price` 的测试函数，该函数向 API 发送 POST 请求（即用于将数据提交到服务器以创建或更新资源的 HTTP 方法），并将输入数据作为 JSON 有效负载。然后，测试函数检查 API 响应的状态码、内容类型和预测价格值。如果您想尝试使用您拥有的真实 API，请将示例 URL 替换为实际的 API 端点。

除了本章中提到的测试策略外，您还可以通过模型监控和断言来确保在生产环境中成功部署和可靠的模型。

# 监控和验证实时性能

在部署期间，我们可以使用监控和日志记录机制来跟踪模型的性能并检测潜在问题。我们可以定期评估已部署的模型，以确保它继续满足性能标准或其他标准，例如无偏见，这是我们为其定义的。我们还可以利用模型监控的信息，根据需要更新或重新训练模型。以下是关于部署前和在生产中建模之间差异的三个重要概念：

+   **数据方差**：用于模型训练和测试的数据会经过数据整理和所有必要的清理和重新格式化步骤。然而，提供给已部署模型的（即从用户到模型的数据）可能不会经过相同的数据处理过程，这会导致生产中模型结果出现差异。

+   **数据漂移**：如果生产中特征或独立变量的特征和意义与建模阶段的不同，就会发生数据漂移。想象一下，你使用第三方工具为人们的健康或财务状况生成分数。该工具背后的算法可能会随时间改变，当你的模型在生产中使用时，其范围和意义将不会相同。如果你没有相应地更新你的模型，那么你的模型将不会按预期工作，因为特征值的含义在用于训练的数据和部署后的用户数据之间将不同。

+   **概念漂移**：概念漂移是指输出变量定义的任何变化。例如，由于概念漂移，训练数据和生产之间的实际决策边界可能不同，这意味着在训练中付出的努力可能导致在生产中远离现实的决策边界。

除了上一章中介绍的 MLflow 之外，还有 Python 和库工具（如 *表 9.2* 中所示），你可以使用这些工具来监控机器学习模型的表现、I/O 数据和基础设施，帮助你维护生产环境中的模型质量和可靠性：

| **工具** | **简要描述** | **URL** |
| --- | --- | --- |
| Alibi Detect | 一个专注于异常值、对抗性和漂移检测的开源 Python 库 | [`github.com/SeldonIO/alibi-detect`](https://github.com/SeldonIO/alibi-detect) |
| Evidently | 一个开源 Python 库，用于分析和监控机器学习模型，提供各种模型评估技术，如数据漂移检测和模型性能监控 | [`github.com/evidentlyai/evidently`](https://github.com/evidentlyai/evidently) |
| ELK Stack | **Elasticsearch**, **Logstash**, **and Kibana** (**ELK**) 是一个流行的用于收集、处理和可视化来自各种来源（包括机器学习模型）的日志和指标的工具栈 | [`www.elastic.co/elk-stack`](https://www.elastic.co/elk-stack) |
| WhyLabs | 一个为机器学习模型提供可观察性和监控的平台 | [`whylabs.ai/`](https://whylabs.ai/) |

表 9.2 – 机器学习模型监控和漂移检测的流行工具

我们还可以从一些统计和可视化技术中受益，用于检测和解决数据和概念漂移。以下是一些用于数据漂移评估的方法示例：

+   **统计测试**：我们可以使用假设检验，如 *Kolmogorov-Smirnov 测试*、*卡方测试* 或 *Mann-Whitney U 测试*，来确定输入数据的分布是否在时间上发生了显著变化。

+   **分布指标**：我们可以使用分布指标，如均值、标准差、分位数和其他汇总统计量，来比较训练数据和生产中的新数据。这些指标中的显著差异可能表明数据漂移。

+   **可视化**：我们可以使用直方图、箱线图或散点图等可视化技术来展示训练数据和生产中新数据的输入特征，以帮助识别数据分布的变化。

+   **特征重要性**：我们可以监控特征重要性值的变化。如果新数据中的特征重要性与训练数据中的特征重要性有显著差异，这可能表明数据漂移。

+   **距离度量**：我们可以使用诸如*Kullback-Leibler 散度*或*Jensen-Shannon 散度*等距离度量来衡量训练数据与新数据分布之间的差异。

模型断言是另一种技术，正如你接下来将要学习的，它可以帮助你构建和部署可靠的机器学习模型。

# 模型断言

我们可以在机器学习建模中使用传统的编程断言来确保模型按预期行为。模型断言可以帮助我们早期发现问题，例如输入数据漂移或其他可能影响模型性能的意外行为。我们可以将模型断言视为一组在模型训练、验证甚至部署期间进行检查的规则，以确保模型的预测满足预定义的条件。模型断言可以从许多方面帮助我们，例如检测模型或输入数据的问题，使我们能够在它们影响模型性能之前解决它们。它们还可以帮助我们保持模型性能。以下是一些模型断言的示例：

+   **输入数据断言**：这些可以检查输入特征是否在预期的范围内或具有正确的数据类型。例如，如果一个模型根据房间数量预测房价，你可能会断言房间数量始终是正整数。

+   **输出数据断言**：这些可以检查模型的预测是否满足某些条件或约束。例如，在二元分类问题中，你可能会断言预测的概率在 0 到 1 之间。

让我们通过一个简单的 Python 中模型断言的例子来了解。在这个例子中，我们将使用`scikit-learn`中的简单线性回归模型，使用玩具数据集根据房间数量预测房价。首先，让我们创建一个玩具数据集并训练线性回归模型：

```py
import numpy as npfrom sklearn.linear_model import LinearRegression
# Toy dataset with number of rooms and corresponding house prices
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([100000, 150000, 200000, 250000, 300000])
# Train the linear regression model
model = LinearRegression()
model.fit(X, y)
```

现在，让我们定义我们的模型断言，以便它们执行以下操作：

1.  检查输入（房间数量）是否为正整数。

1.  检查预测的房价是否在预期范围内。

下面是执行这些操作的代码：

```py
def assert_input(input_data):    assert isinstance(input_data, int),
        "Input data must be an integer"
    assert input_data > 0, "Number of rooms must be positive"
def assert_output(predicted_price, min_price, max_price):
    assert min_price <= predicted_price <= max_price,
        f"Predicted price should be between {min_price} and 
        {max_price}"
```

现在，我们可以使用定义好的模型断言函数，如下所示：

```py
# Test the assertions with example input and output datainput_data = 3
assert_input(input_data)
predicted_price = model.predict([[input_data]])[0]
assert_output(predicted_price, 50000, 350000)
```

`assert_input`函数检查输入数据（即房间数量）是否为整数且为正数。`assert_output`函数检查预测的房价是否在指定的范围内（例如，在本例中为 50,000 至 350,000）。前面的代码没有给出任何`AssertionError`断言，因为它符合模型断言函数中定义的标准。假设我们不是使用整数`3`，而是使用一个字符串，如下所示：

```py
input_data = '3'assert_input(input_data)
```

这里，我们得到以下`AssertionError`：

```py
AssertionError: Input data must be an integer
```

假设我们定义了`assert_output`的输出范围，使其在`50000`和`150000`之间，并使用具有`3`个卧室的房屋模型预测，如下所示：

```py
input_data = 3predicted_price = model.predict([[input_data]])[0]
assert_output(predicted_price, 50000, 150000)
```

我们将得到以下`AssertionError`：

```py
AssertionError: Predicted price should be between 50000 and 150000
```

模型断言是另一种技术，与模型监控并列，有助于确保我们模型的可靠性。

有了这些，我们就结束了这一章。

# 摘要

在本章中，你学习了测试驱动开发的重要概念，包括基础设施和集成测试。你学习了实现这两种类型测试的可用工具和库。我们还通过示例学习了如何使用`pytest`库进行基础设施和集成测试。你还学习了模型监控和模型断言作为评估我们模型在生产和生产环境中的行为之前和之后的两个其他重要主题。这些技术和工具帮助你设计策略，以便你在生产环境中成功部署并拥有可靠的模型。

在下一章中，你将了解可重复性，这是正确机器学习建模中的一个重要概念，以及你如何可以使用数据和模型版本控制来实现可重复性。

# 问题

1.  你能解释数据漂移和概念漂移之间的区别吗？

1.  模型断言如何帮助你开发可靠的机器学习模型？

1.  集成测试的组件有哪些例子？

1.  我们如何使用 Chef、Puppet 和 Ansible？

# 参考文献

+   Kang, Daniel, et al. *模型监控与改进的模型断言*. 机器学习与系统会议论文集 2 (2020): 481-496.
