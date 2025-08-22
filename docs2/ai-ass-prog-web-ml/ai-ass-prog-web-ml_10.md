

# 第十章：维护现有代码库

# 简介

**棕色地带**是另一种与现有代码一起工作的说法。在我的开发生涯中，大部分工作都是在现有代码上完成的。棕色地带的对立面是**绿色地带**，这是一个没有现有代码的新项目。

因此，了解如何与现有代码库一起工作非常重要，并且在棕色地带环境中与像 GitHub Copilot 这样的 AI 助手一起工作时，有很多令人兴奋的事情。

在本章中，我们将：

+   了解不同类型的维护。

+   了解我们如何通过流程进行维护以降低引入更改的风险。

+   使用 GitHub Copilot 来帮助我们进行维护。

# 提示策略

本章与其他章节的书籍略有不同。重点是描述您可能在现有代码库空间中遇到的各种问题。建议您使用最舒适的提示建议方法，无论是提示注释还是聊天界面。至于模式，鼓励您尝试所有三种主要模式，即第二章中描述的 PIC、TAG 或探索性模式。然而，本章的重点是使用“探索性模式”。

# 不同类型的维护

有不同类型的维护，了解它们之间的区别非常重要。以下是一些您可能会遇到的不同类型：

+   **纠正性维护**：这是我们修复错误的时候。

+   **适应性维护**：在这种情况下，我们更改代码以适应新的需求。

+   **改进性维护**：当我们不改变功能的情况下改进代码。这可能是重构或提高代码性能的例子。

+   **预防性维护**：更改代码以防止未来的错误或问题。

# 维护流程

每次您更改代码时，都会引入风险。例如，一个错误修复可能会引入新的错误。为了减轻这种风险，我们需要遵循一个流程。一个建议的流程可能是以下步骤：

1.  **识别**：识别问题或需要做出的更改。

1.  **检查**：检查测试覆盖率以及您的代码被测试覆盖的情况。覆盖得越好，您发现引入的任何错误或其他问题的可能性就越大。

1.  **计划**：计划更改。您将如何进行？您将编写哪些测试？您将运行哪些测试？

1.  **实施**：实施更改。

1.  **验证**：验证更改是否按预期工作。运行测试，运行应用程序，检查日志等。

1.  **集成**：这是确保您在分支中做出的任何更改都能合并到主分支中。

1.  **发布/部署更改**：您希望确保最终客户能够利用这次更改带来的好处。为了实现这一点，您需要部署它。

我们是否需要为每次更改都涵盖所有这些步骤？不，这取决于更改；一些更改可能非常小，我们可能希望在发布我们希望最终客户拥有的版本之前将它们捆绑在一起。好消息是，我们有一个在 GitHub Copilot 中的优秀工具可以帮助我们处理这些小步骤，使我们能够专注于更重要步骤。

# 解决错误

有些错误比其他错误更严重。在最佳情况下，你的错误很少遇到，或者即使发生，也不是什么大问题。在最坏的情况下，你的错误会让你损失金钱。让我们关注一个在运行电子商务网站的场景中，错误让你损失金钱的情况。

希望这段代码从未在生产环境中发布过，但它是一个对电子商务网站可能造成灾难性后果的错误的好例子：

```py
def process_cart():
    # get cart from database
    cart = get_cart()
    card = cart.get_card()
    process_payment(card, cart)
    ship_order(cart) 
```

你可能遇到的问题是，即使`process_payment`失败，也会调用`ship_order`，这意味着你发货了但没有收到付款，损失了金钱。

看看这段代码，如果你在`process_payment`抛出错误而你甚至没有到达`ship_order`，你可能不会有问题。

如果它没有，而是返回一个依赖于支付结果的数值代码，会怎样？

```py
def process_payment(card, cart): 
    if card.is_valid():
        if card.has_funds(cart.total):
            charge_card(card, cart.total)
            return 0
        else:
            return 1 
```

如果你没有读取错误代码，你可能不会注意到支付失败，订单已经发货。那么，我们的方法是什么？

## 1. 识别问题

首先，我们需要识别问题；在这种情况下，你可以做两件事，即：

+   **阅读代码**：我们的代码看起来是什么样子？是否抛出错误，返回一个代码，或者完全是其他东西？

+   **检查测试**：我们是否有覆盖这个场景的测试？如果没有，我们需要编写一个。这个测试应该显示我们有一个错误。

让我们使用 Copilot 来完成这个任务，使用如下提示（这里，我们模拟`process_payment`返回错误代码而不是抛出错误）。在主窗口中选择`ship_order`代码，然后在聊天窗口中输入以下提示：

**[提示]**

为代码编写一个测试；如果`process_payment`返回 1，则不应调用`ship_order`。

**[提示结束]**

我们正在使用 GitHub Copilot 聊天功能，并在我们的文本编辑器中选择`process_cart`代码。

你应该看到类似于以下代码的响应：

**[提示响应]**

```py
```python

import unittest

from unittest.mock import Mock, patch

from your_module import process_cart  # 替换为实际的模块名称

class TestProcessCart(unittest.TestCase):

    @patch('your_module.get_cart')  # 替换为实际的模块名称

    @patch('your_module.process_payment')

    @patch('your_module.ship_order')

    def test_order_not_shipped_if_payment_fails(self, mock_ship_order, mock_process_payment, mock_get_cart):

        # 安排

        mock_cart = Mock()

        mock_cart.get_card.return_value = 'test_card'

        mock_get_cart.return_value = mock_cart

        mock_process_payment.return_value = 1

        # 行动

        process_cart()

        # 断言

        mock_ship_order.assert_not_called()

if __name__ == '__main__':

    unittest.main()

```py 
```

**[响应结束]**

上述代码包含一个测试，测试如果从`process_payment`得到错误代码 1 会发生什么。我们希望发生的事情是`ship_order`不被调用。

这个测试将失败，因为我们的代码中有一个错误，这是好事，因为我们已经识别出一个错误，并且可以通过测试证明它的存在。

## 2. 实施更改

现在我们有一个显示我们存在错误的测试，我们可以实施更改。我们可以自信地做出代码更改，因为我们有一个显示我们存在错误的测试，并且我们希望测试覆盖率也得到了足够的提高。

除非你有良好的测试覆盖率，或者变化很小，你确信不会引入错误，否则不要开始重构。

我们现在如何改进代码？简单——我们可以使用 GitHub Copilot 帮助我们实现。让我们使用以下提示来帮助我们进行所需的代码更改：

**[提示]**

实施更改；如果`process_payment`返回 1，则不应调用`ship_order`。

**[提示结束]**

你为`process_cart`编写的代码现在应该看起来像这样：

**[快速响应]**

```py
```python

def process_cart():

    # 从数据库获取购物车

    cart = get_cart()

    card = cart.get_card()

    payment_status = process_payment(card, cart)

    # 只有在支付成功时才发货

    if payment_status != 1:

        ship_order(cart)

```py 
```

**[响应结束]**

上述代码是一个正确的步骤，因为它现在包含了一个检查支付状态的`if`语句。

为了进一步改进代码，我们可以使用 Copilot 帮助我们使代码更健壮、更易读。让我们使用以下提示进行额外的改进：

**[提示]**

重构代码，使其更容易维护。

**[提示结束]**

这可能会给你以下代码：

**[提示响应]**

```py
def get_payment_status(cart):
    card = cart.get_card()
    return process_payment(card, cart)
def process_cart():
    # get cart from database
    cart = get_cart()
    payment_status = get_payment_status(cart)
    # Only ship the order if the payment was successful
    if payment_status != 1:
        ship_order(cart) 
```

**[响应结束]**

在这里，Copilot 建议我们将支付状态提取到一个单独的函数中。这是一个好建议，因为它使代码更容易阅读和维护。

# 添加新功能

在这个例子中，我们将查看现有的代码库并添加一个新功能。代码库类似于我们之前使用的例子，但它更复杂一些。这是因为它涉及更多涉及不同类型支付方式的函数。以下是代码：

```py
```python

import re

def validate_card(card):

    # 信用卡以 4 开头，有 13 或 16 位数字。

    # 信用卡以 51 至 55 开头，有 16 位数字。

    visa_pattern = r'⁴[0-9]{12}(?:[0-9]{3})?$'

    mastercard_pattern = r'⁵[1-5][0-9]{14}$'

    if re.match(visa_pattern, card):

        return 'Visa'

    elif re.match(mastercard_pattern, card):

        return 'MasterCard'

    else:

        return None

def process_payment(card, cart):

    card_type = validate_card(card)

    if card_type is None:

        return 1  # 无效卡

    else:

        # 在这里处理支付

        # 如果支付成功，则返回 0，否则返回 1

        pass

def process_cart():

    # 从数据库获取购物车

    cart = get_cart()

    card = cart.get_card()

    payment_status = process_payment(card, cart)

    # 只有在支付成功时才发货

    if payment_status == 0:

        ship_order(cart)

```py 
```

在上述代码中，我们有以下内容：

+   `validate_card` 验证卡片号码，如果有效则返回卡片类型；否则返回 `None`。

+   `process_payment` 处理支付，如果支付成功则返回 `0`；否则返回 `1`。

+   `process_cart` 处理购物车，获取卡片，处理支付，如果支付成功，则发货。

## 1. 识别问题并找到需要更改的函数/函数

我们的任务是使我们的代码也支持美国运通卡。实现新的卡片意味着我们需要更新 `validate_card` 函数以支持美国运通卡。

现在，我们知道哪些函数需要更改，但 GitHub Copilot 知道吗？

让我们用提示来询问 GitHub Copilot。

在识别步骤中，我们利用我们的专业知识来识别需要更改的内容和位置。一个好的做法是依靠你自己的技能和 Copilot 工具的组合来帮助分析和识别问题。

## 2. 实施更改，并添加新功能和测试

让我们使用 GitHub Copilot 来帮助我们。我们可以使用以下提示：

**[提示**]

添加对美国运通卡的支持。

**[提示结束**]

你将得到类似以下的代码：

[**快速响应**]

```py
import re
def validate_card(card):
    # Visa cards start with 4 and have 13 or 16 digits.
    # MasterCard cards start with 51 through 55 and have 16 digits.
    # American Express cards start with 34 or 37 and have 15 digits.
    visa_pattern = r'⁴[0-9]{12}(?:[0-9]{3})?$'
    mastercard_pattern = r'⁵[1-5][0-9]{14}$'
    amex_pattern = r'³[47][0-9]{13}$'
    if re.match(visa_pattern, card):
        return 'Visa'
    elif re.match(mastercard_pattern, card):
        return 'MasterCard'
    elif re.match(amex_pattern, card):
        return 'American Express'
    else:
        return None 
```

**[响应结束**]

在上述代码中，Copilot 做了以下更改：

+   在 `validate_card` 函数中创建了一个新的正则表达式以支持美国运通卡。

+   添加了一个新的 `elif` 语句，如果卡片与新的正则表达式匹配，则返回美国运通。

当然，我们还没有完成；我们需要添加一个测试来验证代码是否按预期工作。让我们使用以下提示：

**[提示**]

编写一个测试用例；`validate_card` 应该对有效卡返回“美国运通”。

**[提示结束**]

你的测试应该看起来像下面这样：

**[提示响应**]

```py
```python

import unittest

from your_module import validate_card  # 用实际的模块名替换

class TestValidateCard(unittest.TestCase):

    def test_validate_card_returns_amex_for_valid_amex_card(self):

        # 安排

        valid_amex_card = '378282246310005'

        # 行动

        result = validate_card(valid_amex_card)

        # 断言

        self.assertEqual(result, '美国运通')

if __name__ == '__main__':

    unittest.main()

```py 
```

**[响应结束**]

新的测试用例 `test_validate_card_returns_amex_for_valid_amex_card` 由 Copilot 添加，并测试 `validate_card` 函数对有效的美国运通卡返回 `American Express`。

在这一点上，继续迭代是个好主意；你不仅想测试代码对有效卡片的处理，还要测试无效卡片等。

你可以用 Copilot 作为挑战来编写更多的测试。

# 提高性能

提高性能是一个常见的任务，但可能很难实现。通常，你需要特定的工具来衡量性能，并且你需要知道要衡量什么。了解如何以这种方式改进你的代码的一个好方法是使用装饰器来 *分析* 方法执行所需的时间。然后，你可以衡量特定的方法，找出哪些方法需要改进。

这里是一个在睡眠 2 秒的函数上使用装饰器的例子：

```py
import time
def profile(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end - start} seconds')
        return result
    return wrapper
@profile
def sleep_two_seconds():
    time.sleep(2)
sleep_two_seconds() 
```

`profile` 函数是一个装饰器，它接受一个函数作为输入，并返回一个包装输入函数的函数。`wrapper` 函数测量输入函数的执行时间，并打印结果。

如果你通过终端运行此代码，你会看到如下类似的输出：

```py
sleep_two_seconds took 2.000000238418579 seconds 
```

好的，那么我们如何利用这个来提高性能呢？让我们使用 GitHub Copilot，看看我们如何获得关于改进代码性能的建议。

这是我们要改进的代码。它处理记录并查找重复项：

```py
def process_records(records):
    for i in range(len(records)):
        for j in range(len(records)):
            if i != j and records[i] == records[j]:
                print(f'Duplicate record found: {records[i]}') 
```

大多数看到这个的开发者会回想起他们的算法和数据结构书籍，并开始思考像 *大 O 符号* 这样的事情，这是一种衡量算法性能的方法。主要关注的是减少昂贵的操作次数以提高性能，同时保持相同的功能。

所以，我们应该向 GitHub Copilot 提出的问题是如何改进这段代码？让我们使用以下提示：

**[提示]**

提高代码的性能。

**[提示结束]**

我们得到如下反馈：

**[快速响应]**

```py
def process_records(records):
    record_set = set()
    for record in records:
        if record in record_set:
            print(f'Duplicate record found: {record}')
        else:
            record_set.add(record) 
```

**[响应结束]**

那么，为什么这段代码更好呢？让我们来证明它：

1.  首先，通过计算原始代码和新代码的大 O 符号。

1.  第二，通过使用我们之前创建的装饰器。

## 大 O 符号计算

尝试这个提示：

[**提示**]

这段代码的大 O 复杂度是多少？也请展示你是如何计算它的。

**[提示结束]**

你应该得到如下反馈：

**[快速响应]**

提供的代码的时间复杂度是 *O(n²)*，其中 n 是记录数。

这是它的计算方式：

+   外循环运行 n 次，其中 n 是记录数。这是因为它遍历列表中的每个记录。

+   在外循环内部，还有一个内循环，由于同样的原因，它也会运行 n 次。

+   由于内循环在每次外循环迭代时都会运行 n 次，所以总的迭代次数是 *n * n*，即 *n²*。

因此，时间复杂度是 *O(n²)*。

这意味着如果你将记录数加倍，处理这些记录所需的时间将增加大约四倍。这是二次时间复杂度的特征。

**[响应结束]**

好的，*O(n²)* 不是很好，但新代码怎么样呢？

在新代码上运行相同的提示：

**[快速响应]**

提供的代码的时间复杂度是 *O(n)*，其中 n 是记录数。

这是它的计算方式：

+   循环运行 n 次，其中 n 是记录的数量。这是因为它遍历列表中的每个记录。

+   在循环内部，操作（检查记录是否在集合中以及可能将其添加到集合中）是常数时间，*O(1)*，因为在 Python 中，集合操作通常平均是常数时间。

因此，时间复杂度是 *O(n)* ** O(1)*，这简化为 *O(n)*。

这意味着如果你将记录的数量加倍，处理这些记录所需的时间也将加倍，这是线性时间复杂度的特征。

**[响应结束]**

*O(n)* 比 *O(n²)* 更好，但好多少？让我们在下一节中衡量这一点。

## 测量性能

好的，所以从算法上讲，我们知道新代码更好，但好多少呢？让我们通过使用装饰器和调用代码来衡量。让我们使用 1,000 条记录和 10,000 条记录的输入，并比较执行代码所需的时间。

好的，让我们看看新代码是否比旧代码更快。让我们用 10,000 条记录来尝试：

```py
# old code
@profile
def process_records(records):
    for i in range(len(records)):
        for j in range(len(records)):
            if i != j and records[i] == records[j]:
                print(f'Duplicate record found: {records[i]}')
records_10000 = [i for i in range(10000)]
process_records(records_10000) 
```

运行这段代码，你应该看到以下输出：

```py
process_records took 5.193912506103516 seconds 
```

现在，让我们运行新的代码：

```py
# new code
@profile
def process_records(records):
    record_set = set()
    for record in records:
        if record in record_set:
            print(f'Duplicate record found: {record}')
        else:
            record_set.add(record)
records_10000 = [i for i in range(10000)]
process_records(records_10000) 
```

运行这段代码，你应该看到以下输出：

```py
process_records took 0.0011200904846191406 seconds 
```

如你所见，通过结合你的知识和 GitHub Copilot，你可以改进你的代码。

你的代码并不总是这么明显，你可能需要做更多的工作来提高性能。建议你使用性能分析器来测量性能，然后使用 GitHub Copilot 来帮助你改进代码。

# 提高可维护性

另一个有趣的使用案例是使用 GitHub Copilot 来帮助你提高代码的可维护性。那么，你可以做些什么来提高代码的可维护性呢？以下是一个列表：

+   **改进命名**变量、函数、类等。

+   **分离关注点**：例如，将业务逻辑与展示逻辑分开。

+   **移除重复**：特别是在大型代码库中，你很可能会发现重复。

+   **提高可读性**：例如，你可以通过使用注释、文档字符串、事件测试等方式来提高可读性。

让我们从代码库开始，看看我们如何可以改进它。以下是代码：

```py
def calculate_total(cart, discounts):
    # Define discount functions
    def three_for_two(items):
        total = 0
        for item in items:
            total += item.price * (item.quantity - item.quantity // 3)
        return total
    def christmas_discount(items):
        total = 0
        for item in items:
            total += item.price * item.quantity * 0.5
        return total
    def cheapest_free(items):
        items_sorted = sorted(items, key=lambda x: x.price)
        total = 0
        for item in items_sorted[1:]:
            total += item.price * item.quantity
        return total
    # Calculate totals for each discount
    totals = []
    for discount in discounts:
        if discount == '3 for 2':
            totals.append((discount, three_for_two(cart)))
        elif discount == 'Christmas Discount':
            totals.append((discount, christmas_discount(cart)))
        elif discount == 'Cheapest Free':
            totals.append((discount, cheapest_free(cart)))
    # Sort totals and return the lowest one
    totals_sorted = sorted(totals, key=lambda x: x[1])
    return totals_sorted[0][1] if totals_sorted else sum(item.price * item.quantity for item in cart)
# Define discounts
discounts = ['3 for 2', 'Christmas Discount', 'Cheapest Free']
# Define cart
cart = [
    {'name': 'item1', 'price': 10, 'quantity': 3},
    {'name': 'item2', 'price': 20, 'quantity': 1},
    {'name': 'item3', 'price': 30, 'quantity': 1}
]
# Calculate total
total = calculate_total(cart, discounts)
print(f'Total: {total}') 
```

上述代码库在高级别上描述了以下内容：

+   一个函数，`calculate_total`，用于计算购物车和一系列折扣的总计

+   一系列折扣

+   一个购物车

从这个代码库中，你可以看到它不太容易维护，因为它很难理解发生了什么。

## 1. 识别问题。你看到了哪些问题？

让我们先问问 GitHub Copilot 对这段代码的看法，并决定下一步要做什么。让我们使用以下提示：

**[提示]**

你对这段代码的看法是什么？你会怎么做来改进它？

**[提示结束]**

你会收到一个包含项目符号列表和改进代码的详细响应。让我们先看看项目符号列表：

**[快速响应]**

+   **使用类来表示商品和折扣**：目前，商品和折扣分别用字典和字符串表示。使用类来表示这些实体会使代码更易读和灵活。

+   **分离折扣函数**：当前的折扣函数定义在`calculate_total`函数内部。最好将这些函数单独定义，以提高可读性并使代码更容易测试和维护。

+   **使用更可扩展的折扣应用方法**：当前应用折扣的方法涉及一系列`if-elif`语句。通过将折扣函数存储在字典或`Discount`对象列表中，然后遍历该集合来应用折扣，可以使这种方法更具可扩展性。

**[回复结束]**

上述识别出的问题都似乎是合理的——使用类、分离函数和使用字典。

## 2. 添加测试并降低风险

在我们查看建议的代码更改之前，我们需要最小化引入错误的风险。我们首先应该做的事情是检查我们的测试覆盖率。在这种情况下，我们没有测试，所以我们需要编写一些。让我们使用以下提示：

**[提示]**

为此代码编写测试，覆盖所有可能的执行路径。

**[提示结束]**

**[提示回复]**

```py
import unittest
class TestCalculateTotal(unittest.TestCase):
    def test_no_discounts(self):
        cart = [
            {'name': 'item1', 'price': 10, 'quantity': 3},
            {'name': 'item2', 'price': 20, 'quantity': 1},
            {'name': 'item3', 'price': 30, 'quantity': 1}
        ]
        discounts = []
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 80)
    def test_three_for_two_discount(self):
        cart = [
            {'name': 'item1', 'price': 10, 'quantity': 3},
            {'name': 'item2', 'price': 20, 'quantity': 1},
            {'name': 'item3', 'price': 30, 'quantity': 1}
        ]
        discounts = ['3 for 2']
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 70)
    def test_christmas_discount(self):
        cart = [
            {'name': 'item1', 'price': 10, 'quantity': 3},
            {'name': 'item2', 'price': 20, 'quantity': 1},
            {'name': 'item3', 'price': 30, 'quantity': 1}
        ]
        discounts = ['Christmas Discount']
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 40)
    def test_cheapest_free_discount(self):
        cart = [
            {'name': 'item1', 'price': 10, 'quantity': 3},
            {'name': 'item2', 'price': 20, 'quantity': 1},
            {'name': 'item3', 'price': 30, 'quantity': 1}
        ]
        discounts = ['Cheapest Free']
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 70)
    def test_multiple_discounts(self):
        cart = [
            {'name': 'item1', 'price': 10, 'quantity': 3},
            {'name': 'item2', 'price': 20, 'quantity': 1},
            {'name': 'item3', 'price': 30, 'quantity': 1}
        ]
        discounts = ['3 for 2', 'Christmas Discount', 'Cheapest Free']
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 40)
if __name__ == '__main__':
    unittest.main() 
```

**[回复结束]**

现在，你永远不应该相信 AI 生成的结果，所以请花时间审查代码，运行测试覆盖率工具，确保你在进行下一步之前对代码感到满意，下一步是改进代码。

为了改进 Copilot 的第一个建议的代码，你想要确保它能够运行，并在添加商品到购物车等操作中使用`Item`类。以下是这些修改的结果：

```py
import unittest
from discount_old import calculate_total, Item
from item import Item
class TestCalculateTotal(unittest.TestCase):
    def test_no_discounts(self):
        cart = [
            Item('item1', 10, 3),
            Item('item2', 20, 1),
            Item('item3', 30, 1)
        ]
        discounts = []
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 80)
    def test_three_for_two_discount(self):
        cart = [
            Item('item1', 10, 3),
            Item('item2', 20, 1),
            Item('item3', 30, 1)
        ]
        discounts = ['3 for 2']
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 70)
    def test_christmas_discount(self):
        cart = [
            Item('item1', 10, 3),
            Item('item2', 20, 1),
            Item('item3', 30, 1)
        ]
        discounts = ['Christmas Discount']
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 40)
    def test_cheapest_free_discount(self):
        cart = [
            Item('item1', 10, 3), #30
            Item('item2', 20, 1), # 20
            Item('item3', 30, 1) # 30
        ]
        discounts = ['Cheapest Free']
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 60)
    def test_multiple_discounts(self):
        cart = [
            Item('item1', 10, 3),
            Item('item2', 20, 1),
            Item('item3', 30, 1)
        ]
        discounts = ['3 for 2', 'Christmas Discount', 'Cheapest Free']
        total = calculate_total(cart, discounts)
        self.assertEqual(total, 40)
if __name__ == '__main__':
    unittest.main() 
```

对于应用程序代码：

```py
from item import Item
def calculate_total(cart, discounts):
    # Define discount functions
    def three_for_two(items):
        total = 0
        for item in items:
            total += item.price * (item.quantity - item.quantity // 3)
        return total
    def christmas_discount(items):
        total = 0
        for item in items:
            total += item.price * item.quantity * 0.5
        return total
    def cheapest_free(items):
        if not items:
            return 0
        items_sorted = sorted(items, key=lambda x: x.price * x.quantity)
        cheapest_item = items_sorted[0]
        total = 0
        for item in items:
            total += item.price * item.quantity
        total -= cheapest_item.price * cheapest_item.quantity if cheapest_item.quantity > 0 else 0
        return total
    # Calculate totals for each discount
    totals = []
    for discount in discounts:
        if discount == '3 for 2':
            totals.append((discount, three_for_two(cart)))
        elif discount == 'Christmas Discount':
            totals.append((discount, christmas_discount(cart)))
        elif discount == 'Cheapest Free':
            totals.append((discount, cheapest_free(cart)))
    # Sort totals and return the lowest one
    totals_sorted = sorted(totals, key=lambda x: x[1])
    return totals_sorted[0][1] if totals_sorted else sum(item.price * item.quantity for item in cart)
# if run as script,
if __name__ == '__main__':
    # Define discounts
    discounts = ['3 for 2', 'Christmas Discount', 'Cheapest Free']
    # Define cart
    cart = [
        Item('item1', 10, 3),
        Item('item2', 20, 1),
        Item('item3', 30, 1)
    ]
    # Calculate total
    total = calculate_total(cart, discounts)
    print(f'Total: {total}') 
```

好的，现在我们有一个我们满意的代码库，并且通过了测试。让我们现在看看 GitHub Copilot 建议的代码更改。

## 3. 实施更改并提高可维护性

好的，我们对我们有良好的测试覆盖率感到自信，我们准备改进代码。让我们使用以下提示：

**[提示]**

提高代码的可维护性。

**[提示结束]**

你应该得到类似以下的结果：

**[提示回复]**

为了提高代码的可维护性，我们可以进行以下更改：

+   将折扣函数从`calculate_total`中移出，使其成为独立的函数。这提高了可读性，并使单独测试这些函数更容易。

+   使用字典将折扣名称映射到相应的函数。这消除了在应用折扣时需要多个`if/elif`语句的需求。

```py
from item import Item
# Define discount functions
def three_for_two(items):
    total = 0
    for item in items:
        total += item.price * (item.quantity - item.quantity // 3)
    return total
def christmas_discount(items):
    total = 0
    for item in items:
        total += item.price * item.quantity * 0.5
    return total
def cheapest_free(items):
    if not items:
        return 0
    items_sorted = sorted(items, key=lambda x: x.price * x.quantity)
    cheapest_item = items_sorted[0]
    total = 0
    for item in items:
        total += item.price * item.quantity
    total -= cheapest_item.price * cheapest_item.quantity if cheapest_item.quantity > 0 else 0
    return total
def calculate_total(cart, discounts):
    # Map discount names to functions
    discount_functions = {
        '3 for 2': three_for_two,
        'Christmas Discount': christmas_discount,
        'Cheapest Free': cheapest_free
    }
    # Calculate totals for each discount
    totals = []
    for discount in discounts:
        if discount in discount_functions:
            totals.append((discount, discount_functionsdiscount))
    # Sort totals and return the lowest one
    totals_sorted = sorted(totals, key=lambda x: x[1])
    return totals_sorted[0][1] if totals_sorted else sum(item.price * item.quantity for item in cart)
# if run as script,
if __name__ == '__main__':
    # Define discounts
    discounts = ['3 for 2', 'Christmas Discount', 'Cheapest Free']
    # Define cart
    cart = [
        Item('item1', 10, 3),
        Item('item2', 20, 1),
        Item('item3', 30, 1)
    ]
    # Calculate total
    total = calculate_total(cart, discounts)
    print(f'Total: {total}') 
```

**[回复结束]**

通过引入以下更改，代码现在更容易维护：

+   `calculate_total`函数变得大大缩小，其中它曾经包含的每个函数都已移出

+   提到的相同方法现在也使用字典将名称映射到函数

有效使用 Copilot 的关键是持续迭代，持续进行小的改动，并查看 Copilot 提出的建议。稍微改变提示，看看会发生什么，并选择最适合您需求的输出。

# 挑战

尝试使用提示和 GitHub Copilot 进一步改进代码。如果支付出现问题，应该怎么办？您应该记录它，也许抛出一个错误等？

# **更新现有的电子商务网站**

让我们继续改进我们在前几章中介绍的电子商务网站。在本章中，我们将专注于改进代码库和添加新功能。

为了参考，让我们展示我们开始创建的 `basket.html` 文件：

```py
<!-- a page showing a list of items in a basket, each item should have title, price, quantity, sum and buttons to increase or decrease quantity and the page should have a link to "checkout" at the bottom  -->
<html>
<head>
    <title>Basket</title>
    <link rel="stylesheet" href="css/basket.css">

    <!-- add bootstrap -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <!-- add 3  basket items with each item having id, name, price, quantity, use card css class -->

<!-- 
    <div class="container">
        <div id="basket" class="basket">
        </div>
    </div> -->
    <!-- add app.js  -->
    <!-- add app.js, type javascript -->

    <div id="basket" class="basket">
        <!-- render basket from Vue app, use Boostrap  -->
        <div v-for="(item, index) in basket" class="basket-item">
            <div class="basket-item-text">
                <h2>{{ item.name }}</h2>
                <p>Price: {{ item.price }}</p>
                <p>Quantity: {{ item.quantity }}</p>
                <p>Sum: {{ item.price * item.quantity }}</p>
            </div>
            <div class="basket-item-buttons">
                <button type="submit" class="btn btn-primary btn-block btn-large" @click="increaseQuantity(index)">+</button>
                <button type="submit" class="btn btn-primary btn-block btn-large" @click="decreaseQuantity(index)">-</button>
            </div>
        </div>
    </div>
     <!-- add Vue 3  -->
     <script src="img/vue@next"></script>
    <!-- add boostrap code for Vue 3 -->
    <script>
        let app = Vue.createApp({
            data() {
                return {
                    basket: [
                        {
                            id: 1,
                            name: 'Product 1',
                            price: 100,
                            quantity: 1
                        },
                        {
                            id: 2,
                            name: 'Product 2',
                            price: 200,
                            quantity: 2
                        },
                        {
                            id: 3,
                            name: 'Product 3',
                            price: 300,
                            quantity: 3
                        },
                        {
                            id: 4,
                            name: 'Product 4',
                            price: 400,
                            quantity: 4
                        }
                    ]
                }
            },
            methods: {
                increaseQuantity(index) {
                    this.basket[index].quantity++;
                },
                decreaseQuantity(index) {
                    if (this.basket[index].quantity > 0) {
                        this.basket[index].quantity--;
                    }
                }
            },
            computed: {
                total() {
                    let total = 0;
                    for (let i = 0; i < this.basket.length; i++) {
                        total += this.basket[i].price * this.basket[i].quantity;
                    }

     return total;
                }
            }
        });
        app.mount('#basket');
    </script>
    <!-- <script src="img/app.js"></script> -->
</body> 
```

这是一个原型早期版本；它不是指用于生产的代码。然而，我们可以使用此代码来改进它并使其适用于生产。

现在我们知道这段代码并不好，让我们使用 Copilot 和以下提示来改进它：

**[提示]**

改进代码，使其适用于生产环境。

**[提示结束]**

这个提示突出了代码中的一些问题，特别是：

**[提示响应]**

+   **将 JavaScript 和 CSS 分离到外部文件**：这使代码更易于维护，并允许缓存这些资源。

+   **使用构建工具或模块打包器**：像 Webpack 或 Parcel 这样的工具可以帮助您管理依赖项，将现代 JavaScript 转译为旧版本以获得更好的浏览器兼容性，并压缩您的代码以用于生产。

+   **错误处理**：向您的函数添加错误处理以确保您的应用程序可以优雅地处理意外情况。

+   **Vue 组件结构**：不要在单个脚本中创建整个 `Vue 应用程序`，而是将其分解成可重用的组件。

+   **可访问性**：通过使用适当的 ARIA 角色和属性确保您的 HTML 可访问。

**[响应结束]**

那是一个我们可以询问的很好的列表，但让我们看看它建议的代码更改。它建议将代码分成两个不同的文件，一个用于 HTML，另一个用于 JavaScript：

```py
<!-- index.html -->
<html>
<head>
    <title>Basket</title>
    <link rel="stylesheet" href="css/basket.css">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div id="basket" class="basket"></div>
    <script src="img/vue@next"></script>
    <script src="img/app.js"></script>
</body>
</html> 
```

现在，让我们看看 JavaScript 是如何得到改进的：

```py
// app.js
let app = Vue.createApp({
    data() {
        return {
            basket: [
                { id: 1, name: 'Product 1', price: 100, quantity: 1 },
                { id: 2, name: 'Product 2', price: 200, quantity: 2 },
                { id: 3, name: 'Product 3', price: 300, quantity: 3 },
                { id: 4, name: 'Product 4', price: 400, quantity: 4 }
            ]
        }
    },
    methods: {
        increaseQuantity(index) {
            this.basket[index].quantity++;
        },
        decreaseQuantity(index) {
            if (this.basket[index].quantity > 0) {
                this.basket[index].quantity--;
            }
        }
    },
    computed: {
        total() {
            return this.basket.reduce((total, item) => total + item.price * item.quantity, 0);
        }
    }
});
app.mount('#basket'); 
```

这是一个巨大的进步。作为一名开发者，您可以看到 JavaScript 代码使用了静态数据，所以让我们用以下提示来改变它：

**[提示]**

改进代码，通过调用 API 使用动态数据，并生成一个具有路由 /api/basket 的单独 Python API，其中包含购物车数据。

**[提示结束]**

现在，我们有两个文件，一个是 API，另一个是 JavaScript 代码。让我们先看看 API：

**[提示响应]**

```py
# app.py
from flask import Flask, jsonify
app = Flask(__name__)
@app.route('/api/basket')
def get_basket():
    basket = [
        { 'id': 1, 'name': 'Product 1', 'price': 100, 'quantity': 1 },
        { 'id': 2, 'name': 'Product 2', 'price': 200, 'quantity': 2 },
        { 'id': 3, 'name': 'Product 3', 'price': 300, 'quantity': 3 },
        { 'id': 4, 'name': 'Product 4', 'price': 400, 'quantity': 4 }
    ]
    return jsonify(basket)
if __name__ == '__main__':
    app.run(debug=True) 
```

在您上面得到一个返回购物车数据的路由 `/api/basket`。让我们看看改进后的 JavaScript 代码：

```py
let app = Vue.createApp({
    data() {
        return {
            basket: []
        }
    },
    methods: {
        increaseQuantity(index) {
            this.basket[index].quantity++;
        },
        decreaseQuantity(index) {
            if (this.basket[index].quantity > 0) {
                this.basket[index].quantity--;
            }
        }
    },
    computed: {
        total() {
            return this.basket.reduce((total, item) => total + item.price * item.quantity, 0);
        }
    },
    mounted() {
        fetch('/api/basket')
            .then(response => response.json())
            .then(data => {
                this.basket = data;
            });
    }
});
app.mount('#basket'); 
```

**[响应结束]**

太好了！我们现在从 API 获取数据。我们如何改进代码？如果您之前构建过 Vue 应用程序，您将知道它涉及到使用 Vue CLI 并有一个构建步骤。让我们用以下提示询问 GitHub Copilot：

**[提示]**

改进代码，使用 Vue CLI，并添加一个构建步骤。

**[提示结束]**

现在，我们以如下方式将命令和文件更改作为响应返回：

**[提示响应]**

在您的机器上全局安装`Vue` CLI：

```py
npm install -g @vue/cli 
```

创建一个新的`Vue`项目

```py
vue create my-project
cd my-project 
```

现在，让我们创建一个新的`Vue`组件用于篮子。在`src/components`目录中创建一个新的文件`Basket.vue`：

```py
<!-- src/components/Basket.vue -->
<template>
  <div id="basket">
    <!-- Your basket template goes here -->
  </div>
</template>
<script>
export default {
  data() {
    return {
      basket: []
    };
  },
  methods: {
    increaseQuantity(index) {
      this.basket[index].quantity++;
    },
    decreaseQuantity(index) {
      if (this.basket[index].quantity > 0) {
        this.basket[index].quantity--;
      }
    }
  },
  computed: {
    total() {
      return this.basket.reduce((total, item) => total + item.price * item.quantity, 0);
    }
  },
  mounted() {
    fetch('/api/basket')
      .then(response => response.json())
      .then(data => {
        this.basket = data;
      });
  }
};
</script>
<style scoped>
/* Your CSS goes here */
</style> 
```

然后，在主`App.vue`中导入并使用此组件：

```py
<!-- src/App.vue -->
<template>
  <div id="app">
    <Basket />
  </div>
</template>
<script>
import Basket from './components/Basket.vue';
export default {
  name: 'App',
  components: {
    Basket
  }
};
</script>
<style>
/* Your CSS goes here */
</style> 
```

现在，您可以使用`Vue` CLI 的构建命令来创建一个生产就绪版本的应用程序：

```py
npm run build 
```

这将创建一个包含应用程序生产就绪版本的`dist`目录。

**[响应结束]**

这看起来像是正确的步骤。然而，您应该始终测试代码以确保它按预期工作。

作为挑战，看看上述代码是否工作，以及您是否可以进一步改进它。

代码可能在框架的不同版本之间发生变化。在这种情况下，我们使用 Vue.js，所以请确保查阅您使用的版本的文档。

# 作业

找到您想要改进的代码片段。按照本章概述的过程进行：

1.  识别问题。您看到了什么问题？

1.  添加测试，降低变更风险。

1.  实施更改并提高可维护性。

如果您没有想要改进的代码片段，请尝试使用本章中的代码或 Kata（Gilded Rose）GitHub 页面上的代码：[`github.com/emilybache/GildedRose-Refactoring-Kata`](https://github.com/emilybache/GildedRose-Refactoring-Kata)。

# 知识检查

1.  绿色田野开发和棕色田野开发有什么区别？

A：绿色田野开发是指从头开始编码；棕色田野开发是指更新现有代码。

1.  更新现有代码的最佳方式是什么？

A：最好的方法是进行小改动，并确保有足够的测试。

# 摘要

在本章中，我们确立了编写代码的一个重要方面是更新现有代码，这被称为棕色田野开发。我们还探讨了 GitHub Copilot 如何帮助您完成这项任务。

从本章中可以得出的最重要的信息是确保您有一个更新代码的方法，以降低即将进行的变更风险。多次进行小改动比一次性进行大改动要好。在开始更改代码之前，强烈建议您有足够的测试。

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/aicode`](https://packt.link/aicode)

![](img/QR_Code510410532445718281.png)
