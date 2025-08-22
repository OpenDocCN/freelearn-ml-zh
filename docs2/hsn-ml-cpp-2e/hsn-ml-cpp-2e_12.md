

# 第十二章：导出和导入模型

在本章中，我们将讨论如何在训练期间和之后保存和加载模型参数。这很重要，因为模型训练可能需要几天甚至几周。保存中间结果允许我们在以后进行评估或生产使用时加载它们。

这种常规的保存操作在随机应用程序崩溃的情况下可能有益。任何 **机器学习**（**ML**）框架的另一个重要特性是它导出模型架构的能力，这使我们能够在框架之间共享模型，并使模型部署更加容易。本章的主要内容是展示如何使用不同的 C++ 库导出和导入模型参数，如权重和偏置值。本章的第二部分全部关于 **开放神经网络交换**（**ONNX**）格式，该格式目前在不同的 ML 框架中越来越受欢迎，可以用于共享训练模型。此格式适用于共享模型架构以及模型参数。

本章将涵盖以下主题：

+   C++ 库中的 ML 模型序列化 API

+   深入了解 ONNX 格式

# 技术要求

以下为本章的技术要求：

+   `Dlib` 库

+   `mlpack` 库

+   `F``lashlight` 库

+   `pytorch` 库

+   `onnxruntime` 框架

+   支持 C++20 的现代 C++ 编译器

+   CMake 构建系统版本 >= 3.8

本章的代码文件可以在本书的 GitHub 仓库中找到：[`github.com/PacktPublishing/Hands-on-Machine-learning-with-C-Second-Edition/tree/main/Chapter12`](https://github.com/PacktPublishing/Hands-on-Machine-learning-with-C-Second-Edition/tree/main/Chapter12)。

# C++ 库中的 ML 模型序列化 API

在本节中，我们将讨论 `Dlib`、`F``lashlight`、`mlpack` 和 `pytorch` 库中可用的 ML 模型共享 API。在不同的 C++ 库之间共享 ML 模型主要有三种类型：

+   共享模型参数（权重）

+   共享整个模型的架构

+   共享模型架构及其训练参数

在以下各节中，我们将查看每个库中可用的 API，并强调它支持哪种类型的共享。

## 使用 Dlib 进行模型序列化

`Dlib` 库使用 `decision_function` 和神经网络对象的序列化 API。让我们通过实现一个真实示例来学习如何使用它。

首先，我们将定义神经网络、回归核和训练样本的类型：

```py
using namespace Dlib;
using NetworkType = loss_mean_squared<fc<1, input<matrix<double>>>>;
using SampleType = matrix<double, 1, 1>;
using KernelType = linear_kernel<SampleType>;
```

然后，我们将使用以下代码生成训练数据：

```py
size_t n = 1000;
std::vector<matrix<double>> x(n);
std::vector<float> y(n);
std::random_device rd;
std::mt19937 re(rd());
std::uniform_real_distribution<float> dist(-1.5, 1.5);
// generate data
for (size_t i = 0; i < n; ++i) {
  xi = i;
  y[i] = func(i) + dist(re);
}
```

在这里，`x` 代表预测变量，而 `y` 代表目标变量。目标变量 `y` 被均匀随机噪声盐化，以模拟真实数据。这些变量具有线性依赖关系，该关系由以下函数定义：

```py
double func(double x) {
  return 4\. + 0.3 * x;
}
```

生成数据后，我们使用`vector_normalizer`类型的对象对其进行归一化。这种类型的对象在训练后可以重复使用，以使用学习到的均值和标准差对数据进行归一化。以下代码片段展示了其实现方式：

```py
vector_normalizer<matrix<double>> normalizer_x;
normalizer_x.train(x);
for (size_t i = 0; i < x.size(); ++i) {
  x[i] = normalizer_x(x[i]);
}
```

最后，我们使用`krr_trainer`类型的对象训练核岭回归的`decision_function`对象：

```py
void TrainAndSaveKRR(const std::vector<matrix<double>>& x,
                     const std::vector<float>& y) {
  krr_trainer<KernelType> trainer;
  trainer.set_kernel(KernelType());
  decision_function<KernelType> df = trainer.train(x, y);
  serialize("Dlib-krr.dat") << df;
}
```

注意，我们使用`KernelType`对象的实例初始化了训练器对象。

现在我们有了训练好的`decision_function`对象，我们可以使用`serialize`函数返回的流对象将其序列化到文件中：

```py
serialize("Dlib-krr.dat") << df;
```

此函数将文件存储的名称作为输入参数，并返回一个输出流对象。我们使用了`<<`运算符将回归模型学习到的权重放入文件。在先前的代码示例中使用的序列化方法仅保存模型参数。

同样的方法可以用来序列化`Dlib`库中的几乎所有机器学习模型。以下代码展示了如何使用它来序列化神经网络的参数：

```py
void TrainAndSaveNetwork(
    const std::vector<matrix<double>>& x,
    const std::vector<float>& y) {
  NetworkType network;
  sgd solver;
  dnn_trainer<NetworkType> trainer(network, solver);
  trainer.set_learning_rate(0.0001);
  trainer.set_mini_batch_size(50);
  trainer.set_max_num_epochs(300);
  trainer.be_verbose();
  trainer.train(x, y);
  network.clean();
  serialize("Dlib-net.dat") << network;
  net_to_xml(network, "net.xml");
}
```

对于神经网络，还有一个`net_to_xml`函数，它保存模型结构。然而，库 API 中没有函数可以将保存的结构加载到我们的程序中。这是用户的责任来实现加载函数。

如果我们希望在不同框架之间共享模型，可以使用`net_to_xml`函数，如`Dlib`文档中所示。

为了检查参数序列化是否按预期工作，我们可以生成新的测试数据来评估加载的模型：

```py
std::cout << "Target values \n";
std::vector<matrix<double>> new_x(5);
for (size_t i = 0; i < 5; ++i) {
  new_x[i].set_size(1, 1);
  new_xi = i;
  new_x[i] = normalizer_x(new_x[i]);
  std::cout << func(i) << std::endl;
}
```

注意，我们已经重用了`normalizer`对象。一般来说，`normalizer`对象的参数也应该进行序列化和加载，因为在评估过程中，我们需要将新数据转换为我们用于训练数据的相同统计特性。

要在`Dlib`库中加载序列化的对象，我们可以使用`deserialize`函数。此函数接受文件名并返回一个输入流对象：

```py
void LoadAndPredictKRR(
    const std::vector<matrix<double>>& x) {
  decision_function<KernelType> df;
  deserialize("Dlib-krr.dat") >> df;
  // Predict
  std::cout << "KRR predictions \n";
  for (auto& v : x) {
    auto p = df(v);
    std::cout << static_cast<double>(p) << std::endl;
  }
}
```

如前所述，在`Dlib`库中，序列化仅存储模型参数。因此，要加载它们，我们需要使用在序列化之前具有相同属性的模型对象。

对于回归模型，这意味着我们应该实例化一个与相同核类型相对应的决策函数对象。

对于神经网络模型，这意味着我们应该实例化一个与序列化时使用的相同类型的网络对象，如下面的代码块所示：

```py
void LoadAndPredictNetwork(
    const std::vector<matrix<double>>& x) {
  NetworkType network;
  deserialize("Dlib-net.dat") >> network;
  // Predict
  auto predictions = network(x);
  std::cout << "Net predictions \n";
  for (auto p : predictions) {
    std::cout << static_cast<double>(p) << std::endl;
  }
}
```

在本节中，我们了解到`Dlib`序列化 API 允许我们保存和加载机器学习模型参数，但在序列化和加载模型架构方面选项有限。在下一节中，我们将探讨`Shogun`库模型序列化 API。

## 使用 Flashlight 进行模型序列化

`Flashlight`库可以将模型和参数保存和加载到二进制格式中。它内部使用`Cereal` C++库进行序列化。以下示例展示了这一功能。

如前例所示，我们首先创建一些示例训练数据：

```py
int64_t n = 10000;
auto x = fl::randn({n});
auto y = x * 0.3f + 0.4f;
// Define dataset
std::vector<fl::Tensor> fields{x, y};
auto dataset = std::make_shared<fl::TensorDataset>(fields);
fl::BatchDataset batch_dataset(dataset, /*batch_size=*/64);
```

在这里，我们创建了一个包含随机数据的向量`x`，并通过应用线性依赖公式来创建我们的目标变量`y`。我们将独立和目标向量包装到一个名为`batch_dataset`的`BatchDataset`对象中，我们将使用它来训练一个示例神经网络。

以下代码展示了我们的神经网络定义：

```py
fl::Sequential model;
model.add(fl::View({1, 1, 1, -1}));
model.add(fl::Linear(1, 8));
model.add(fl::ReLU());
model.add(fl::Linear(8, 16));
model.add(fl::ReLU());
model.add(fl::Linear(16, 32));
model.add(fl::ReLU());
model.add(fl::Linear(32, 1));
```

如您所见，这是我们之前示例中使用的相同的正向传播网络，但这次是为 Flashlight 设计的。

以下代码示例展示了如何训练模型：

```py
auto loss = fl::MeanSquaredError();
float learning_rate = 0.01;
float momentum = 0.5;
auto sgd = fl::SGDOptimizer(model.params(), 
                            learning_rate,
                            momentum);
const int epochs = 5;
for (int epoch_i = 0; epoch_i < epochs; ++epoch_i) {
  for (auto& batch : batch_dataset) {
    sgd.zeroGrad();
    auto predicted = model(fl::input(batch[0]));
    auto local_batch_size = batch[0].shape().dim(0);
    auto target =
        fl::reshape(batch[1], {1, 1, 1, local_batch_size});
    auto loss_value = loss(predicted, fl::noGrad(target));
    loss_value.backward();
    sgd.step();
  }
}
```

在这里，我们使用了之前使用的相同训练方法。首先，我们定义了`loss`对象和`sgd`优化器对象。然后，我们使用两个循环来训练模型：一个循环遍历 epoch，另一个循环遍历批次。在内循环中，我们将模型应用于训练批次数据以获取新的预测值。然后，我们使用`loss`对象使用批次目标值计算 MSE 值。我们还使用了损失值变量的`backward`方法来计算梯度。最后，我们使用`sgd`优化器对象的`step`方法更新模型参数。

现在我们有了训练好的模型，我们有两种方法可以在`Flashlight`库中保存它：

1.  序列化整个模型及其架构和权重。

1.  仅序列化模型权重。

对于第一种选项——即序列化整个模型及其架构——我们可以这样做：

```py
fl::save("model.dat", model);
```

在这里，`model.dat`是我们将保存模型的文件名。要加载此类文件，我们可以使用以下代码：

```py
fl::Sequential model_loaded;
fl::load("model.dat", model_loaded);
```

在这种情况下，我们创建了一个名为`model_loaded`的新空对象。这个新对象只是一个没有特定层的`fl::Sequential`容器对象。所有层和参数值都是通过`fl::load`函数加载的。一旦我们加载了模型，我们就可以这样使用它：

```py
auto predicted = model_loaded(fl::noGrad(new_x));
```

在这里，`new_x`是我们用于评估目的的一些新数据。

当您存储整个模型时，这种方法对于包含不同模型但具有相同输入和输出接口的应用程序可能很有用，因为它可以帮助您轻松地在生产中更改或升级模型，例如。

第二种选项，仅保存网络的参数（权重）值，如果我们需要定期重新训练模型，或者如果我们只想共享或重用模型或其参数的某些部分，这可能是有用的。为此，我们可以使用以下代码：

```py
fl::save("model_params.dat", model.params());
```

在这里，我们使用了`model`对象的`params`方法来获取所有模型参数。此方法返回所有模型子模块的参数的`std::vector`序列。因此，您只能管理其中的一些。要加载已保存的参数，我们可以使用以下代码：

```py
std::vector<fl::Variable> params;
fl::load("model_params.dat", params);
for (int i = 0; i < static_cast<int>(params.size()); ++i) {
  model.setParams(params[i], i);
}
```

首先，我们创建了空的 `params` 容器。然后，使用 `fl::load` 函数将参数值加载到其中。为了能够更新特定子模块的参数值，我们使用了 `setParams` 方法。`'setParams'` 方法接受一个值和一个整数位置，我们想要设置这个值。我们保存了所有模型参数，以便我们可以按顺序将它们放回模型中。

不幸的是，没有方法可以将其他格式的模型和权重加载到 `Flashlight` 库中。因此，如果您需要从其他格式加载，您必须编写一个转换器并使用 `setParams` 方法设置特定值。在下一节中，我们将深入了解 `mlpack` 库的序列化 API。

## 使用 mlpack 进行模型序列化

`mlpack` 库仅实现了模型参数序列化。这种序列化基于存在于 Armadillo 数学库中的功能，该库被用作 mlpack 的后端。这意味着我们可以使用 mlpack API 以不同的文件格式保存参数值。具体如下：

+   `.csv`，或者可选的 `.txt`

+   `.txt`

+   `.txt`

+   `.pgm`

+   `.ppm`

+   `.bin`

+   `.bin`

+   `.hdf5`、`.hdf`、`.h5` 或 `.he5`

让我们看看使用 mlpack 创建模型和参数管理的最小示例。首先，我们需要一个模型。以下代码展示了我们可以使用的创建模型的功能：

```py
using ModelType = FFN<MeanSquaredError, ConstInitialization>;
ModelType make_model() {
  MeanSquaredError loss;
  ConstInitialization init(0.);
  ModelType model(loss, init);
  model.Add<Linear>(8);
  model.Add<ReLU>();
  model.Add<Linear>(16);
  model.Add<ReLU>();
  model.Add<Linear>(32);
  model.Add<ReLU>();
  model.Add<Linear>(1);
  return model;
}
```

`create_model` 函数创建了一个具有多个线性层的前馈网络。请注意，我们使此模型使用 `MSE` 作为损失函数并添加了零参数初始化器。现在我们有了模型，我们需要一些数据来训练它。以下代码展示了如何创建线性相关数据：

```py
size_t n = 10000;
arma::mat x = arma::randn(n).t();
arma::mat y = x * 0.3f + 0.4f;
```

在这里，我们创建了两个单维向量，类似于我们在 `Flashlight` 示例中所做的，但使用了 Armadillo 矩阵 API。请注意，我们使用了 `t()` 转置方法对 `x` 向量进行操作，因为 mlpack 使用列维度作为其训练特征。

现在，我们可以连接所有组件并执行模型训练：

```py
ens::Adam optimizer;
auto model = make_model();
model.Train(x, y, optimizer);
```

在这里，我们创建了 `Adam` 算法优化器对象，并在模型的 `Train` 方法中使用我们之前创建的两个数据向量。现在，我们有了训练好的模型，准备保存其参数。这可以按以下方式完成：

```py
data::Save("model.bin", model.Parameters(), true);
```

默认情况下，`data::Save` 函数会根据提供的文件名扩展名自动确定要保存的文件格式。在这里，我们使用了模型对象的 `Parameters` 方法来获取参数值。此方法返回一个包含所有值的矩阵。我们还传递了 `true` 作为第三个参数，以便在失败的情况下 `save` 函数抛出异常。默认情况下，它将只返回 `false`；这是您必须手动检查的事情。

我们可以使用 `mlpack::data::Load` 函数来加载参数值，如下所示：

```py
auto new_model = make_model();
data::Load("model.bin", new_model.Parameters());
```

在这里，我们创建了`new_model`对象；这是一个与之前相同的模型，但参数初始化为零。然后，我们使用`mlpack::data::Load`函数从文件中加载参数值。再次使用`Parameters`方法获取内部参数值矩阵的引用，并将其传递给`load`函数。我们将`load`函数的第三个参数设置为`true`，以便在出现错误时可以抛出异常。

现在我们已经初始化了模型，我们可以用它来进行预测：

```py
arma::mat predictions;
new_model.Predict(new_x, predictions);
```

在这里，我们创建了一个输出矩阵`prediction`，并使用`new_model`对象的`Predict`方法进行模型评估。请注意，`new_x`是我们希望对其获取预测的一些新数据。

注意，你不能将其他框架的文件格式加载到 mlpack 中，因此如果你需要，你必须创建转换器。在下一节中，我们将查看`pytorch`库的序列化 API。

## 使用 PyTorch 进行模型序列化

在本节中，我们将讨论在`pytorch` C++库中可用的两种网络参数序列化方法：

+   `torch::save`函数

+   一个`torch::serialize::OutputArchive`类型的对象，用于将参数写入`OutputArchive`对象

让我们从准备神经网络开始。

### 初始化神经网络

让我们从生成训练数据开始。以下代码片段显示了我们可以如何做到这一点：

```py
torch::DeviceType device = torch::cuda::is_available()
? torch::DeviceType::CUDA
: torch::DeviceType::CPU;
```

通常，我们希望尽可能利用硬件资源。因此，首先，我们通过使用`torch::cuda::is_available()`调用检查系统中是否有带有 CUDA 技术的 GPU 可用：

```py
std::random_device rd;
std::mt19937 re(rd());
std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
```

我们定义了`dist`对象，以便我们可以在`-1`到`1`的范围内生成均匀分布的实数：

```py
size_t n = 1000;
torch::Tensor x;
torch::Tensor y;
{
  std::vector<float> values(n);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), re);
  std::vector<torch::Tensor> x_vec(n);
  std::vector<torch::Tensor> y_vec(n);
  for (size_t i = 0; i < n; ++i) {
    x_vec[i] = torch::tensor(
        values[i],
        torch::dtype(torch::kFloat).device(
                                    device).requires_grad(false));
    y_vec[i] = torch::tensor(
        (func(values[i]) + dist(re)),
        torch::dtype(torch::kFloat).device(
                                    device).requires_grad(false));
  }
  x = torch::stack(x_vec);
  y = torch::stack(y_vec);
}
```

然后，我们生成了 1,000 个预测变量值并将它们打乱。对于每个值，我们使用在之前的示例中使用的线性函数计算目标值——即`func`。下面是这个过程的示例：

```py
float func(float x) {
  return 4.f + 0.3f * x;
}
```

然后，所有值都通过`torch::tensor`函数调用移动到`torch::Tensor`对象中。请注意，我们使用了之前检测到的设备来创建张量。一旦我们将所有值移动到张量中，我们就使用`torch::stack`函数将预测值和目标值连接到两个不同的单张量中。这是必要的，以便我们可以使用`pytorch`库的线性代数例程进行数据归一化：

```py
auto x_mean = torch::mean(x, /*dim*/ 0);
auto x_std = torch::std(x, /*dim*/ 0);
x = (x - x_mean) / x_std;
```

最后，我们使用了`torch::mean`和`torch::std`函数来计算预测值的平均值和标准差，并将它们进行了归一化处理。

在以下代码中，我们定义了`NetImpl`类，该类实现了我们的神经网络：

```py
class NetImpl : public torch::nn::Module {
 public:
  NetImpl() {
    l1_ = torch::nn::Linear(torch::nn::LinearOptions(
                                1, 8).with_bias(true));
    register_module("l1", l1_);
    l2_ = torch::nn::Linear(torch::nn::LinearOptions(
                                8, 4).with_bias(true));
    register_module("l2", l2_);
    l3_ = torch::nn::Linear(torch::nn::LinearOptions(
                                4, 1).with_bias(true));
    register_module("l3", l3_);
    // initialize weights
    for (auto m : modules(false)) {
      if (m->name().find("Linear") != std::string::npos) {
        for (auto& p : m->named_parameters()) {
          if (p.key().find("weight") != std::string::npos) {
            torch::nn::init::normal_(p.value(), 0, 0.01);
                    }
          if (p.key().find("bias") != std::string::npos) {
            torch::nn::init::zeros_(p.value());
          }
        }
      }
    }
  }
torch::Tensor forward(torch::Tensor x) {
  auto y = l1_(x);
  y = l2_(y);
  y = l3_(y);
  return y;
}
private:
  torch::nn::Linear l1_{nullptr};
  torch::nn::Linear l2_{nullptr};
  torch::nn::Linear l3_{nullptr};
}
TORCH_MODULE(Net);
```

在这里，我们将我们的神经网络模型定义为一个具有三个全连接神经元层和线性激活函数的网络。每个层都是`torch::nn::Linear`类型。

在我们模型的构造函数中，我们使用小的随机值初始化了所有网络参数。我们通过遍历所有网络模块（参见`modules`方法调用）并应用`torch::nn::init::normal_`函数到由`named_parameters()`模块方法返回的参数来实现这一点。偏差使用`torch::nn::init::zeros_`函数初始化为零。`named_parameters()`方法返回由字符串名称和张量值组成的对象，因此对于初始化，我们使用了它的`value`方法。

现在，我们可以使用我们生成的训练数据来训练模型。以下代码展示了我们如何训练我们的模型：

```py
Net model;
model->to(device);
// initialize optimizer -----------------------------------
double learning_rate = 0.01;
torch::optim::Adam optimizer(model->parameters(),
torch::optim::AdamOptions(learning_rate).weight_decay(0.00001));
// training
int64_t batch_size = 10;
int64_t batches_num = static_cast<int64_t>(n) / batch_size;
int epochs = 10;
for (int epoch = 0; epoch < epochs; ++epoch) {
  // train the model
  // -----------------------------------------------
  model->train();  // switch to the training mode
  // Iterate the data
  double epoch_loss = 0;
  for (int64_t batch_index = 0; batch_index < batches_num;
       ++batch_index) {
    auto batch_x =
        x.narrow(0, batch_index * batch_size, batch_size)
            .unsqueeze(1);
    auto batch_y =
        y.narrow(0, batch_index * batch_size, batch_size)
            .unsqueeze(1);
    // Clear gradients
    optimizer.zero_grad();
    // Execute the model on the input data
    torch::Tensor prediction = model->forward(batch_x);
    torch::Tensor loss =
        torch::mse_loss(prediction, batch_y);
    // Compute gradients of the loss and parameters of
    // our model
    loss.backward();
    // Update the parameters based on the calculated
    // gradients.
    optimizer.step();
  }
}
```

为了利用所有我们的硬件资源，我们将模型移动到选定的计算设备。然后，我们初始化了一个优化器。在我们的例子中，优化器使用了`Adam`算法。之后，我们在每个 epoch 上运行了一个标准的训练循环，其中对于每个 epoch，我们取训练批次，清除优化器的梯度，执行前向传递，计算损失，执行反向传递，并使用优化器步骤更新模型权重。

从数据集中选择一批训练数据，我们使用了张量的`narrow`方法，该方法返回了一个维度减少的新张量。此函数接受新的维度数量作为第一个参数，起始位置作为第二个参数，以及要保留的元素数量作为第三个参数。我们还使用了`unsqueeze`方法来添加一个批次维度；这是 PyTorch API 进行前向传递所必需的。

如我们之前提到的，我们可以使用两种方法在 C++ API 中的`pytorch`序列化模型参数（Python API 提供了更多的功能）。让我们来看看它们。

### 使用 torch::save 和 torch::load 函数

我们可以采取的第一种保存模型参数的方法是使用`torch::save`函数，该函数递归地保存传递的模块的参数：

```py
torch::save(model, "pytorch_net.pt");
```

为了正确地与我们的自定义模块一起使用，我们需要使用`register_module`模块的方法将所有子模块在父模块中注册。

要加载保存的参数，我们可以使用`torch::load`函数：

```py
Net model_loaded;
torch::load(model_loaded, "pytorch_net.pt");
```

该函数将读取自文件的值填充到传递的模块参数中。

### 使用 PyTorch 存档对象

第二种方法是用`torch::serialize::OutputArchive`类型的对象，并将我们想要保存的参数写入其中。以下代码展示了如何实现我们模型的`SaveWeights`方法。此方法将我们模块中存在的所有参数和缓冲区写入`archive`对象，然后它使用`save_to`方法将它们写入文件：

```py
void NetImpl::SaveWeights(const std::string& file_name) {
  torch::serialize::OutputArchive archive;
  auto parameters = named_parameters(true /*recurse*/);
  auto buffers = named_buffers(true /*recurse*/);
  for (const auto& param : parameters) {
    if (param.value().defined()) {
      archive.write(param.key(), param.value());
    }
  }
  for (const auto& buffer : buffers) {
    if (buffer.value().defined()) {
      archive.write(buffer.key(), buffer.value(),
                    /*is_buffer*/ true);
    }
  }
  archive.save_to(file_name);
}
```

保存缓冲区张量也很重要。可以使用 `named_buffers` 模块的 `named_buffers` 方法从模块中检索缓冲区。这些对象代表用于评估不同模块的中间值。例如，我们可以是批归一化模块的运行均值和标准差值。在这种情况下，如果我们使用序列化来保存中间步骤并且由于某种原因训练过程停止，我们需要它们继续训练。

要加载以这种方式保存的参数，我们可以使用 `torch::serialize::InputArchive` 对象。以下代码展示了如何为我们的模型实现 `LoadWeights` 方法：

```py
void NetImpl::LoadWeights(const std::string& file_name) {
  torch::serialize::InputArchive archive;
  archive.load_from(file_name);
  torch::NoGradGuard no_grad;
  auto parameters = named_parameters(true /*recurse*/);
  auto buffers = named_buffers(true /*recurse*/);
  for (auto& param : parameters) {
      archive.read(param.key(), param.value());
  }
  for (auto& buffer : buffers) {
      archive.read(buffer.key(), buffer.value(),
          /*is_buffer*/ true);
  }
}
```

在这里，`LoadWeights` 方法使用 `archive` 对象的 `load_from` 方法从文件中加载参数。首先，我们使用 `named_parameters` 和 `named_buffers` 方法从我们的模块中获取参数和缓冲区，并使用 `archive` 对象的 `read` 方法逐步填充它们的值。

注意，我们使用 `torch::NoGradGuard` 类的实例来告诉 `pytorch` 库我们不会执行任何模型计算或图相关操作。这样做是必要的，因为 `pytorch` 库构建计算图和任何无关操作都可能导致错误。

现在，我们可以使用新的 `model_loaded` 模型实例，并带有 `load` 参数来评估一些测试数据上的模型。请注意，我们需要使用 `eval` 方法将模型切换到评估模式。生成的测试数据值也应使用 `torch::tensor` 函数转换为张量对象，并将其移动到与我们的模型使用的相同计算设备上。以下代码展示了我们如何实现这一点：

```py
model_loaded->to(device);
model_loaded->eval();
std::cout << "Test:\n";
for (int i = 0; i < 5; ++i) {
  auto x_val = static_cast<float>(i) + 0.1f;
  auto tx = torch::tensor(
      x_val, torch::dtype(torch::kFloat).device(device));
  tx = (tx - x_mean) / x_std;
  auto ty = torch::tensor(
      func(x_val),
      torch::dtype(torch::kFloat).device(device));
  torch::Tensor prediction = model_loaded->forward(tx);
  std::cout << "Target:" << ty << std::endl;
  std::cout << "Prediction:" << prediction << std::endl;
}
```

在本节中，我们探讨了 `pytorch` 库中的两种序列化类型。第一种方法涉及使用 `torch::save` 和 `torch::load` 函数，分别轻松保存和加载所有模型参数。第二种方法涉及使用 `torch::serialize::InputArchive` 和 `torch::serialize::OutputArchive` 类型的对象，这样我们就可以选择我们想要保存和加载的参数。

在下一节中，我们将讨论 ONNX 文件格式，它允许我们在不同的框架之间共享我们的 ML 模型架构和模型参数。

# 深入探讨 ONNX 格式

ONNX 格式是一种特殊的文件格式，用于在不同框架之间共享神经网络架构和参数。它基于 Google 的 Protobuf 格式和库。这种格式存在的原因是测试和在不同的环境和设备上运行相同的神经网络模型。

通常，研究人员会使用他们熟悉的编程框架来开发模型，然后在不同环境中运行这个模型，用于生产目的或者他们想要与其他研究人员或开发者共享模型。这种格式得到了所有主流框架的支持，包括 PyTorch、TensorFlow、MXNet 以及其他。然而，这些框架的 C++ API 对这种格式的支持不足，在撰写本文时，它们只为处理 ONNX 格式提供了 Python 接口。尽管如此，微软提供了`onnxruntime`框架，可以直接使用不同的后端，如 CUDA、CPU 或甚至 NVIDIA TensorRT 来运行推理。

在深入探讨使用框架解决我们具体用例的细节之前，考虑某些限制因素是很重要的，这样我们可以全面地处理问题陈述。有时，由于缺少某些操作符或函数，导出为 ONNX 格式可能会出现问题，这可能会限制可以导出的模型类型。此外，对张量的动态维度和条件操作符的支持可能有限，这限制了使用具有动态计算图和实现复杂算法的模型的能力。这些限制取决于目标硬件。你会发现嵌入式设备有最多的限制，而且其中一些问题只能在推理运行时发现。然而，使用 ONNX 有一个很大的优势——通常，这样的模型可以在各种不同的张量数学加速硬件上运行。

与 ONNX 相比，TorchScript 对模型操作符和结构的限制更少。通常，可以导出具有所有所需分支的动态计算图模型。然而，在您必须推断模型的地方可能会有硬件限制。例如，通常无法使用移动 GPU 或 NPUs 进行 TorchScript 推理。ExecTorch 应该在将来解决这个问题。

为了尽可能多地利用可用硬件，我们可以使用特定供应商的不同推理引擎。通常，可以将 ONNX 格式或使用其他方法的模型转换为内部格式，以在特定的 GPU 或 NPU 上进行推理。此类引擎的例子包括 Intel 硬件的 OpenVINO、NVIDIA 的 TensorRT、基于 ARM 处理器的 ArmNN 以及 Qualcomm NPUs 的 QNN。

现在我们已经了解了如何最好地利用这个框架，接下来让我们了解如何使用 ResNet 神经网络架构进行图像分类。

## 使用 ResNet 架构进行图像分类

通常，作为开发者，我们不需要了解 ONNX 格式内部是如何工作的，因为我们只对保存模型的文件感兴趣。如前所述，内部上，ONNX 格式是一个 Protobuf 格式的文件。以下代码展示了 ONNX 文件的第一部分，它描述了如何使用 ResNet 神经网络架构进行图像分类：

```py
ir_version: 3
graph {
  node {
  input: "data"
  input: "resnetv24_batchnorm0_gamma"
  input: "resnetv24_batchnorm0_beta"
  input: "resnetv24_batchnorm0_running_mean"
  input: "resnetv24_batchnorm0_running_var"
  output: "resnetv24_batchnorm0_fwd"
  name: "resnetv24_batchnorm0_fwd"
  op_type: "BatchNormalization"
  attribute {
      name: "epsilon"
      f: 1e-05
      type: FLOAT
  }
  attribute {
      name: "momentum"
      f: 0.9
      type: FLOAT
  }
  attribute {
      name: "spatial"
      i: 1
      type: INT
  }
}
node {
  input: "resnetv24_batchnorm0_fwd"
  input: "resnetv24_conv0_weight"
  output: "resnetv24_conv0_fwd"
  name: "resnetv24_conv0_fwd"
  op_type: "Conv"
  attribute {
      name: "dilations"
      ints: 1
      ints: 1
      type: INTS
  }
  attribute {
      name: "group"
      i: 1
      type: INT
  }
  attribute {
      name: "kernel_shape"
      ints: 7
      ints: 7
      type: INTS
  }
  attribute {
      name: "pads"
      ints: 3
      ints: 3
      ints: 3
      ints: 3
      type: INTS
  }
  attribute {
      name: "strides"
      ints: 2
      ints: 2
      type: INTS
  }
}
...
}
```

通常，ONNX 文件以二进制格式提供，以减少文件大小并提高加载速度。

现在，让我们学习如何使用`onnxruntime` API 加载和运行 ONNX 模型。ONNX 社区为公开可用的模型库中最流行的神经网络架构提供了预训练模型([`github.com/onnx/models`](https://github.com/onnx/models))。

有许多现成的模型可以用于解决不同的机器学习任务。例如，我们可以使用`ResNet-50`模型来进行图像分类任务([`github.com/onnx/models/tree/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx`](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx))。

对于这个模型，我们必须下载相应的包含图像类别描述的`synset`文件，以便能够以人类可读的方式返回分类结果。您可以在[`github.com/onnx/models/blob/main/validated/vision/classification/synset.txt`](https://github.com/onnx/models/blob/main/validated/vision/classification/synset.txt)找到该文件。

为了能够使用`onnxruntime` C++ API，我们必须使用以下头文件：

```py
#include <onnxruntime_cxx_api.h>
```

然后，我们必须创建全局共享的`onnxruntime`环境和模型评估会话，如下所示：

```py
Ort::Env env;
Ort::Session session(env,
                     "resnet50-v1-7.onnx",
                     Ort::SessionOptions{nullptr});
```

`session`对象将模型的文件名作为其输入参数，并自动加载它。在这里，我们传递了下载的模型的名称。最后一个参数是`SessionOptions`类型的对象，它可以用来指定特定的设备执行器，例如 CUDA。`env`对象包含一些共享的运行时状态。最有价值的状态是日志数据和日志级别，这些可以通过构造函数参数进行配置。

一旦我们加载了一个模型，我们可以访问其参数，例如模型输入的数量、模型输出的数量和参数名称。如果您事先不知道这些信息，这些信息将非常有用，因为您需要输入参数名称来运行推理。我们可以按照以下方式发现此类模型信息：

```py
void show_model_info(const Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
```

在这里，我们创建了一个函数头并初始化了字符串内存分配器。现在，我们可以打印输入参数信息：

```py
  auto num_inputs = session.GetInputCount();
  for (size_t i = 0; i < num_inputs; ++i) {
    auto input_name = session.GetInputNameAllocated(i,
                                           allocator);
    std::cout << "Input name " << i << " : " << input_name
              << std::endl;
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info =
        type_info.GetTensorTypeAndShapeInfo();
    auto tensor_shape = tensor_info.GetShape();
    std::cout << "Input shape " << i << " : ";
    for (size_t j = 0; j < tensor_shape.size(); ++j)
      std::cout << tensor_shape[j] << " ";
    std::cout << std::endl;
  }
```

一旦我们发现了输入参数，我们可以按照以下方式打印输出参数信息：

```py
 auto num_outputs = session.GetOutputCount();
  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_name = session.GetOutputNameAllocated(i,
                                             allocator);
  std::cout << "Output name " << i << " : " <<
                       output_name << std::endl;
  Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto tensor_shape = tensor_info.GetShape();
  std::cout << "Output shape " << i << " : ";
  for (size_t j = 0; j < tensor_shape.size(); ++j)
    std::cout << tensor_shape[j] << " ";
  std::cout << std::endl;
  }
}
```

在这里，我们使用了`session`对象来发现模型属性。通过使用`GetInputCount`和`GetOutputCount`方法，我们得到了相应的输入和输出参数的数量。然后，我们使用`GetInputNameAllocated`和`GetOutputNameAllocated`方法通过它们的索引来获取参数名称。请注意，这些方法需要`allocator`对象。在这里，我们使用了在`show_model_info`函数顶部初始化的默认对象。

我们可以通过使用相应的参数索引，使用`GetInputTypeInfo`和`GetOutputTypeInfo`方法获取额外的参数类型信息。然后，通过使用这些参数类型信息对象，我们可以使用`GetTensorTypeAndShapeInfo`方法获取张量信息。这里最重要的信息是使用`tensor_onfo`对象的`GetShape`方法获取的张量形状。它很重要，因为我们需要为模型输入和输出张量使用特定的形状。形状表示为整数向量。现在，使用`show_model_info`函数，我们可以获取模型输入和输出参数信息，创建相应的张量，并将数据填充到它们中。

在我们的案例中，输入是一个大小为`1 x 3 x 224 x 224`的张量，它代表了用于分类的 RGB 图像。`onnxruntime`会话对象接受`Ort::Value`类型对象作为输入并将它们作为输出填充。

下面的代码片段展示了如何为模型准备输入张量：

```py
constexpr const int width = 224;
constexpr const int height = 224;
std::array<int64_t, 4> input_shape{1, 3, width, height};
std::vector<float> input_image(3 * width * height);
read_image(argv[3], width, height, input_image);
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,
                                              OrtMemTypeCPU);
Ort::Value input_tensor =
    Ort::Value::CreateTensor<float>(memory_ info,
                                    input_image.data(),
                                    input_image.size(),
                                    input_shape.data(),
                                    input_shape.size());
```

首先，我们定义了代表输入图像宽度和高度的常量。然后，我们创建了`input_shape`对象，它定义了张量的完整形状，包括其批次维度。有了形状，我们创建了`input_image`向量来保存确切的图像数据。这个数据容器被`read_image`函数填充，我们将在稍后对其进行详细探讨。最后，我们使用`Ort::Value::CreateTensor`函数创建了`input_tensor`对象，它接受`memory_info`对象和数据以及形状容器的引用。`memory_info`对象使用分配输入张量在主机 CPU 设备上的参数创建。输出张量也可以用同样的方式创建：

```py
std::array<int64_t, 2> output_shape{1, 1000};
std::vector<float> result(1000);
Ort::Value output_tensor =
    Ort::Value::CreateTensor<float>(memory_ info,
                                    result.data(),
                                    result.size(),
                                    output_shape.data(),
                                    output_shape.size());
```

注意到`onnxruntime` API 允许我们创建一个空的输出张量，它将被自动初始化。我们可以这样做：

```py
Ort::Value output_tensor{nullptr};
```

现在，我们可以使用`Run`方法进行评估：

```py
const char* input_names[] = {"data"};
const char* output_names[] = {"resnetv17_dense0_fwd"};
Ort::RunOptions run_options;
session.Run(run_options,
            input_names,
            &input_tensor,
            1,
            output_names,
            &output_tensor,
            1);

```

在这里，我们定义了输入和输出参数的名称和常量，并使用默认初始化创建了`run_options`对象。`run_options`对象可以用来配置日志的详细程度，而`Run`方法可以用来评估模型。请注意，输入和输出张量作为指针传递到数组中，并指定了相应的元素数量。在我们的案例中，我们指定了单个输入和输出元素。

该模型的输出是针对`ImageNet`数据集的 1,000 个类别的图像得分（概率），该数据集用于训练模型。以下代码展示了如何解码模型的输出：

```py
std::map<size_t, std::string> classes = read_classes("synset.txt");
std::vector<std::pair<float, size_t>> pairs;
for (size_t i = 0; i < result.size(); i++) {
  if (result[i] > 0.01f) {  // threshold check
    pairs.push_back(std::make_pair(
        output[i], i + 1));  // 0 –//background
  }
}
std::sort(pairs.begin(), pairs.end());
std::reverse(pairs.begin(), pairs.end());
pairs.resize(std::min(5UL, pairs.size()));
for (auto& p : pairs) {
  std::cout << "Class " << p.second << " Label "
            << classes.at(p.second) << « Prob « << p.first
            << std::endl;
}
```

在这里，我们遍历了结果张量数据中的每个元素——即我们之前初始化的`result`向量对象。在模型评估期间，这个`result`对象被填充了实际的数据值。然后，我们将得分值和类别索引放入相应的对向量中。这个向量按得分降序排序。然后，我们打印了得分最高的五个类别。

在本节中，我们通过`onnxruntime`框架的示例了解了如何处理 ONNX 格式。然而，我们仍需要学习如何将输入图像加载到张量对象中，这是我们用于模型输入的部分。

## 将图像加载到 onnxruntime 张量中

让我们学习如何根据模型的输入要求和内存布局加载图像数据。之前，我们初始化了一个相应大小的`input_image`向量。模型期望输入图像是归一化的，并且是三个通道的 RGB 图像，其形状为`N x 3 x H x W`，其中*N*是批处理大小，*H*和*W*至少应为 224 像素宽。归一化假设图像被加载到`[0, 1]`范围内，然后使用均值`[0.485, 0.456, 0.406]`和标准差`[0.229, 0.224, 0.225]`进行归一化。

假设我们有一个以下函数定义来加载图像：

```py
void read_image(const std::string& file_name,
                       int width,
                       int height,
                       std::vector<float>& image_data)
...
}
```

让我们编写它的实现。为了加载图像，我们将使用`OpenCV`库：

```py
// load image
auto image = cv::imread(file_name, cv::IMREAD_COLOR);
if (!image.cols || !image.rows) {
  return {};
}
if (image.cols != width || image.rows != height) {
  // scale image to fit
  cv::Size scaled(
      std::max(height * image.cols / image.rows, width),
      std::max(height, width * image.rows / image.cols));
  cv::resize(image, image, scaled);
  // crop image to fit
  cv::Rect crop((image.cols - width) / 2,
                (image.rows - height) / 2, width, height);
  image = image(crop);
}
```

在这里，我们使用`cv::imread`函数从文件中读取图像。如果图像的尺寸不等于已指定的尺寸，我们需要使用`cv::resize`函数调整图像大小，然后如果图像的尺寸超过指定的尺寸，还需要裁剪图像。

然后，我们必须将图像转换为浮点类型和 RGB 格式：

```py
image.convertTo(image, CV_32FC3);
cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
```

格式化完成后，我们可以将图像分成三个单独的通道，分别是红色、绿色和蓝色。我们还应该对颜色值进行归一化。以下代码展示了如何进行这一操作：

```py
std::vector<cv::Mat> channels(3);
cv::split(image, channels);
std::vector<double> mean = {0.485, 0.456, 0.406};
std::vector<double> stddev = {0.229, 0.224, 0.225};
size_t i = 0;
for (auto& c : channels) {
  c = ((c / 255) - mean[i]) / stddev[i];
  ++i;
}
```

在这里，每个通道都被减去相应的均值，并除以相应的标准差，以进行归一化处理。

然后，我们应该将通道连接起来：

```py
cv::vconcat(channels[0], channels[1], image);
cv::vconcat(image, channels[2], image);
assert(image.isContinuous());
```

在这种情况下，归一化后的通道被`cv::vconcat`函数连接成一个连续的图像。

以下代码展示了如何将 OpenCV 图像复制到`image_data`向量中：

```py
std::vector<int64_t> dims = {1, 3, height, width};
std::copy_n(reinterpret_cast<float*>(image.data),
image.size().area(),
image_data.begin());
```

在这里，图像数据被复制到一个由指定维度初始化的浮点向量中。使用`cv::Mat::data`类型成员访问 OpenCV 图像数据。我们将图像数据转换为浮点类型，因为该成员变量是`unsigned char *`类型。使用标准的`std::copy_n`函数复制像素数据。这个函数被用来填充`input_image`向量中的实际图像数据。然后，使用`input_image`向量数据的引用在`CreateTensor`函数中初始化`Ort::Value`对象。

在 ONNX 格式示例中，还使用了一个可以从`synset`文件中读取类定义的函数。我们将在下一节中查看这个函数。

## 读取类定义文件

在这个例子中，我们使用了`read_classes`函数来加载对象映射。在这里，键是一个图像类索引，值是一个文本类描述。这个函数很简单，逐行读取`synset`文件。在这样的文件中，每一行包含一个数字和一个由空格分隔的类描述字符串。以下代码展示了其定义：

```py
using Classes = std::map<size_t, std::string>;
Classes read_classes(const std::string& file_name) {
  Classes classes;
  std::ifstream file(file_name);
  if (file) {
    std::string line;
    std::string id;
    std::string label;
    std::string token;
    size_t idx = 1;
   while (std::getline(file, line)) {
      std::stringstream line_stream(line);
      size_t i = 0;
      while (std::getline(line_stream, token, ' ')) {
        switch (i) {
          case 0:
            id = token;
            break;
          case 1:
            label = token;
            break;
        }
        token.clear();
        ++i;
      }
      classes.insert({idx, label});
      ++idx;
    }
  }
  return classes;
```

注意，我们在内部`while`循环中使用了`std::getline`函数来对单行字符串进行分词。我们通过指定定义分隔符字符值的第三个参数来实现这一点。

在本节中，我们学习了如何加载`synset`文件，该文件表示类名与它们 ID 之间的对应关系。我们使用这些信息将作为分类结果得到的类 ID 映射到其字符串表示形式，并将其展示给用户。

# 摘要

在本章中，我们学习了如何在不同的机器学习框架中保存和加载模型参数。我们了解到，我们在`Flashlight`、`mlpack`、`Dlib`和`pytorch`库中使用的所有框架都有一个用于模型参数序列化的 API。通常，这些函数很简单，与模型对象和一些输入输出流一起工作。我们还讨论了可以用于保存和加载整体模型架构的序列化 API。在撰写本文时，我们使用的某些框架并不完全支持此类功能。例如，`Dlib`库可以以 XML 格式导出神经网络，但不能加载它们。PyTorch C++ API 缺少导出功能，但它可以加载和评估从 Python API 导出并使用 TorchScript 功能加载的模型架构。然而，`pytorch`库确实提供了对库 API 的访问，这允许我们从 C++中加载和评估保存为 ONNX 格式的模型。然而，请注意，您可以从之前导出为 TorchScript 并加载的 PyTorch Python API 中导出模型到 ONNX 格式。

我们还简要地了解了 ONNX 格式，并意识到它是一种在不同的机器学习框架之间共享模型非常流行的格式。它支持几乎所有用于有效地序列化复杂神经网络模型的操作和对象。在撰写本文时，它得到了所有流行的机器学习框架的支持，包括 TensorFlow、PyTorch、MXNet 和其他框架。此外，微软提供了 ONNX 运行时实现，这使得我们可以在不依赖任何其他框架的情况下运行 ONNX 模型的推理。

在本章末尾，我们开发了一个 C++应用程序，可以用来在 ResNet-50 模型上进行推理，该模型是在 ONNX 格式下训练和导出的。这个应用程序是用 onnxruntime C++ API 制作的，这样我们就可以加载模型并在加载的图像上进行分类评估。

在下一章中，我们将讨论如何将使用 C++库开发的机器学习模型部署到移动设备上。

# 进一步阅读

+   Dlib 文档：[`Dlib.net/`](http://dlib.net/)

+   PyTorch C++ API：[`pytorch.org/cppdocs/`](https://pytorch.org/cppdocs/)

+   ONNX 官方页面：[`onnx.ai/`](https://onnx.ai/)

+   ONNX 模型库：[`github.com/onnx/models`](https://github.com/onnx/models)

+   ONNX ResNet 模型用于图像分类：[`github.com/onnx/models/blob/main/validated/vision/classification/resnet`](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet)

+   `onnxruntime` C++示例：[`github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx`](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx)

+   Flashlight 文档：[`fl.readthedocs.io/en/stable/index.html`](https://fl.readthedocs.io/en/stable/index.html)

+   mlpack 文档：[`rcppmlpack.github.io/mlpack-doxygen/`](https://rcppmlpack.github.io/mlpack-doxygen/)
