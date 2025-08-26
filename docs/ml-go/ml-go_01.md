# 第一章：收集和组织数据

调查显示，90%或更多的数据科学家时间花在收集数据、组织数据和清洗数据上，而不是在训练/调整复杂的机器学习模型上。这是为什么？机器学习部分不是最有意思的部分吗？为什么我们需要如此关注我们数据的状态？首先，没有数据，我们的机器学习模型就无法学习。这看起来可能很明显。然而，我们需要意识到我们构建的模型的部分优势在于我们提供给它们的那些数据。正如常见的说法，“垃圾输入，垃圾输出”。我们需要确保收集相关、干净的数据来为我们的机器学习模型提供动力，这样它们才能按预期操作并产生有价值的结果。

并非所有类型的数据都适用于使用某些类型的模型。例如，当我们有高维数据（例如文本数据）时，某些模型的表现不佳，而其他模型则假设变量是正态分布的，这显然并不总是如此。因此，我们必须小心收集适合我们用例的数据，并确保我们理解我们的数据和模型将如何交互。

收集和组织数据消耗了数据科学家大量时间的原因之一是数据通常很混乱且难以聚合。在大多数组织中，数据可能存储在不同的系统和格式中，并具有不同的访问控制策略。我们不能假设向我们的模型提供训练集就像指定一个文件路径那样简单；这通常并非如此。

为了形成训练/测试集或向模型提供预测变量，我们可能需要处理各种数据格式，如 CSV、JSON、数据库表等，并且我们可能需要转换单个值。常见的转换包括解析日期时间、将分类数据转换为数值数据、归一化值以及应用一些函数到值上。然而，我们并不能总是假设某个变量的所有值都存在或能够以类似的方式进行解析。

通常数据中包含缺失值、混合类型或损坏值。我们如何处理这些情况将直接影响我们构建的模型的质量，因此，我们必须愿意仔细收集、组织和理解我们的数据。

尽管这本书的大部分内容将专注于各种建模技术，但你应始终将数据收集、解析和组织视为成功数据科学项目的关键组成部分（或可能是最重要的部分）。如果你的项目这部分没有经过精心开发且具有高度诚信，那么你将给自己在长远发展中埋下隐患。

# 处理数据 - Gopher 风格

与许多用于数据科学/分析的其它语言相比，Go 为数据操作和解析提供了一个非常强大的基础。尽管其他语言（例如 Python 或 R）可能允许用户快速交互式地探索数据，但它们通常促进破坏完整性的便利性，即动态和交互式数据探索通常会导致在更广泛的应用中行为异常的代码。

以这个简单的 CSV 文件为例：

```py
1,blah1
2,blah2
3,blah3
```

诚然，我们很快就能编写一些 Python 代码来解析这个 CSV 文件，并从整数列中输出最大值，即使我们不知道数据中有什么类型：

```py
import pandas as pd

# Define column names.
cols = [
 'integercolumn',
 'stringcolumn'
 ]

# Read in the CSV with pandas.
data = pd.read_csv('myfile.csv', names=cols)

# Print out the maximum value in the integer column.
print(data['integercolumn'].max())
```

这个简单的程序将打印出正确的结果：

```py
$ python myprogram.py
3
```

我们现在删除一个整数值以产生一个缺失值，如下所示：

```py
1,blah1
2,blah2
,blah3
```

Python 程序因此完全失去了完整性；具体来说，程序仍然运行，没有告诉我们任何事情有所不同，仍然产生了一个值，并且产生了一个不同类型的值：

```py
$ python myprogram.py
2.0
```

这是不可接受的。除了一个整数值外，我们的所有整数值都可能消失，而我们不会对变化有任何洞察。这可能会对我们的建模产生深远的影响，但它们将非常难以追踪。通常，当我们选择动态类型和抽象的便利性时，我们正在接受这种行为的变化性。

这里重要的是，你并不是不能在 Python 中处理这种行为，因为专家会很快认识到你可以正确处理这种行为。关键是这种便利性并不默认促进完整性，因此很容易自食其果。

另一方面，我们可以利用 Go 的静态类型和显式错误处理来确保我们的数据以预期的方式被解析。在这个小例子中，我们也可以编写一些 Go 代码，而不会遇到太多麻烦来解析我们的 CSV（现在不用担心细节）：

```py
// Open the CSV.
f, err := os.Open("myfile.csv")
if err != nil {
    log.Fatal(err)
}

// Read in the CSV records.
r := csv.NewReader(f)
records, err := r.ReadAll()
if err != nil {
    log.Fatal(err)
}

// Get the maximum value in the integer column.
var intMax int
for _, record := range records {

    // Parse the integer value.
    intVal, err := strconv.Atoi(record[0])
    if err != nil {
        log.Fatal(err)
    }

    // Replace the maximum value if appropriate.
    if intVal > intMax {
        intMax = intVal
    }
}

// Print the maximum value.
fmt.Println(intMax)
```

这将产生一个正确的结果，对于所有整数值都存在的 CSV 文件：

```py
$ go build
$ ./myprogram
3
```

但与之前的 Python 代码相比，我们的 Go 代码将在我们遇到输入 CSV 中不期望遇到的内容时通知我们（对于删除值 3 的情况）：

```py
$ go build
$ ./myprogram
2017/04/29 12:29:45 strconv.ParseInt: parsing "": invalid syntax
```

在这里，我们保持了完整性，并且我们可以确保我们可以以适合我们用例的方式处理缺失值。

# 使用 Go 收集和组织数据的最佳实践

如前所述部分所示，Go 本身为我们提供了在数据收集、解析和组织中保持高完整性水平的机会。我们希望确保在为机器学习工作流程准备数据时，我们能够利用 Go 的独特属性。

通常，Go 数据科学家/分析师在收集和组织数据时应遵循以下最佳实践。这些最佳实践旨在帮助您在应用程序中保持完整性，并能够重现任何分析：

1.  **检查并强制执行预期的类型**：这看起来可能很显然，但在使用动态类型语言时，它往往被忽视。尽管这稍微有些冗长，但将数据显式解析为预期类型并处理相关错误可以在将来为你节省很多麻烦。

1.  **标准化和简化你的数据输入/输出**：有许多第三方包用于处理某些类型的数据或与某些数据源交互（其中一些我们将在本书中介绍）。然而，如果你标准化与数据源交互的方式，特别是围绕使用 `stdlib` 的使用，你可以开发可预测的模式并在团队内部保持一致性。一个很好的例子是选择使用 `database/sql` 进行数据库交互，而不是使用各种第三方 API 和 DSL。

1.  **版本化你的数据**：机器学习模型产生的结果极其不同，这取决于你使用的训练数据、参数选择和输入数据。因此，如果不版本化你的代码和数据，就无法重现结果。我们将在本章后面讨论数据版本化的适当技术。

如果你开始偏离这些基本原则，你应该立即停止。你可能会为了方便而牺牲完整性，这是一条危险的道路。我们将让这些原则引导我们在本书中的学习，并在下一节考虑各种数据格式/来源时，我们将遵循这些原则。

# CSV 文件

CSV 文件可能不是大数据的首选格式，但作为一名机器学习领域的数据科学家或开发者，你肯定会遇到这种格式。你可能需要将邮政编码映射到经纬度，并在互联网上找到这个 CSV 文件，或者你的销售团队可能会以 CSV 格式提供销售数据。无论如何，我们需要了解如何解析这些文件。

我们将在解析 CSV 文件时利用的主要包是 Go 标准库中的 `encoding/csv`。然而，我们还将讨论几个允许我们快速操作或转换 CSV 数据的包--`github.com/kniren/gota/dataframe` 和 `go-hep.org/x/hep/csvutil`。

# 从文件中读取 CSV 数据

让我们考虑一个简单的 CSV 文件，我们将在稍后返回，命名为 `iris.csv`（可在以下链接找到：[`archive.ics.uci.edu/ml/datasets/iris`](https://archive.ics.uci.edu/ml/datasets/iris))。这个 CSV 文件包括四个表示花朵测量的浮点列和一个表示相应花朵种类的字符串列：

```py
$ head iris.csv 
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
```

导入 `encoding/csv` 后，我们首先打开 CSV 文件并创建一个 CSV 读取器值：

```py
// Open the iris dataset file.
f, err := os.Open("../data/iris.csv")
if err != nil {
    log.Fatal(err)
}
defer f.Close()

// Create a new CSV reader reading from the opened file.
reader := csv.NewReader(f)
```

然后，我们可以读取 CSV 文件的所有记录（对应于行），这些记录被导入为 `[][]string`：

```py
// Assume we don't know the number of fields per line. By setting
// FieldsPerRecord negative, each row may have a variable
// number of fields.
reader.FieldsPerRecord = -1

// Read in all of the CSV records.
rawCSVData, err := reader.ReadAll()
if err != nil {
    log.Fatal(err)
}
```

我们也可以通过无限循环逐个读取记录。只需确保检查文件末尾（`io.EOF`），以便在读取所有数据后循环结束：

```py
// Create a new CSV reader reading from the opened file.
reader := csv.NewReader(f)
reader.FieldsPerRecord = -1

// rawCSVData will hold our successfully parsed rows.
var rawCSVData [][]string

// Read in the records one by one.
for {

    // Read in a row. Check if we are at the end of the file.
    record, err := reader.Read()
    if err == io.EOF {
        break
    }

    // Append the record to our dataset.
    rawCSVData = append(rawCSVData, record)
}
```

如果你的 CSV 文件不是以逗号分隔的，或者如果你的 CSV 文件包含注释行，你可以利用`csv.Reader.Comma`和`csv.Reader.Comment`字段来正确处理格式独特的 CSV 文件。在字段在 CSV 文件中用单引号包围的情况下，你可能需要添加一个辅助函数来删除单引号并解析值。

# 处理意外的字段

前面的方法对干净的 CSV 数据工作得很好，但通常我们不会遇到干净的数据。我们必须解析混乱的数据。例如，你可能会在你的 CSV 记录中找到意外的字段或字段数量。这就是为什么`reader.FieldsPerRecord`存在的原因。这个读取值字段让我们能够轻松地处理混乱的数据，如下所示：

```py
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,blah,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
```

这个版本的`iris.csv`文件在一行中有一个额外的字段。我们知道每个记录应该有五个字段，所以让我们将我们的`reader.FieldsPerRecord`值设置为`5`：

```py
// We should have 5 fields per line. By setting
// FieldsPerRecord to 5, we can validate that each of the
// rows in our CSV has the correct number of fields.
reader.FieldsPerRecord = 5
```

那么当我们从 CSV 文件中读取记录时，我们可以检查意外的字段并保持我们数据的一致性：

```py
// rawCSVData will hold our successfully parsed rows.
var rawCSVData [][]string

// Read in the records looking for unexpected numbers of fields.
for {

    // Read in a row. Check if we are at the end of the file.
    record, err := reader.Read()
    if err == io.EOF {
        break
    }

    // If we had a parsing error, log the error and move on.
    if err != nil {
        log.Println(err)
        continue
    }

    // Append the record to our dataset, if it has the expected
    // number of fields.
    rawCSVData = append(rawCSVData, record)
}
```

在这里，我们选择通过记录错误来处理错误，并且我们只将成功解析的记录收集到`rawCSVData`中。读者会注意到这种错误可以以许多不同的方式处理。重要的是我们正在强迫自己检查数据的一个预期属性，并提高我们应用程序的完整性。

# 处理意外的类型

我们刚刚看到 CSV 数据被读取为`[][]string`。然而，Go 是静态类型的，这允许我们对每个 CSV 字段执行严格的检查。我们可以在解析每个字段以进行进一步处理时这样做。考虑一些混乱的数据，其中包含与列中其他值类型不匹配的随机字段：

```py
4.6,3.1,1.5,0.2,Iris-setosa
5.0,string,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
4.9,3.1,1.5,0.1,Iris-setosa
5.0,3.2,1.2,string,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
```

为了检查我们 CSV 记录中字段的类型，让我们创建一个`struct`变量来保存成功解析的值：

```py
// CSVRecord contains a successfully parsed row of the CSV file.
type CSVRecord struct {
    SepalLength  float64
    SepalWidth   float64
    PetalLength  float64
    PetalWidth   float64
    Species      string
    ParseError   error
}
```

然后，在我们遍历记录之前，让我们初始化这些值的一个切片：

```py
// Create a slice value that will hold all of the successfully parsed
// records from the CSV.
var csvData []CSVRecord
```

现在我们遍历记录时，我们可以解析为该记录的相关类型，捕获任何错误，并按需记录：

```py

// Read in the records looking for unexpected types.
for {

    // Read in a row. Check if we are at the end of the file.
    record, err := reader.Read()
    if err == io.EOF {
        break
    }

    // Create a CSVRecord value for the row.
    var csvRecord CSVRecord

    // Parse each of the values in the record based on an expected type.
    for idx, value := range record {

        // Parse the value in the record as a string for the string column.
        if idx == 4 {

            // Validate that the value is not an empty string. If the
            // value is an empty string break the parsing loop.
            if value == "" {
                log.Printf("Unexpected type in column %d\n", idx)
                csvRecord.ParseError = fmt.Errorf("Empty string value")
                break
            }

            // Add the string value to the CSVRecord.
            csvRecord.Species = value
            continue
        }

        // Otherwise, parse the value in the record as a float64.
        var floatValue float64

        // If the value can not be parsed as a float, log and break the
        // parsing loop.
        if floatValue, err = strconv.ParseFloat(value, 64); err != nil {
            log.Printf("Unexpected type in column %d\n", idx)
            csvRecord.ParseError = fmt.Errorf("Could not parse float")
            break
        }

        // Add the float value to the respective field in the CSVRecord.
        switch idx {
        case 0:
            csvRecord.SepalLength = floatValue
        case 1:
            csvRecord.SepalWidth = floatValue
        case 2:
            csvRecord.PetalLength = floatValue
        case 3:
            csvRecord.PetalWidth = floatValue
        }
    }

    // Append successfully parsed records to the slice defined above.
    if csvRecord.ParseError == nil {
        csvData = append(csvData, csvRecord)
    }
}
```

# 使用数据框操作 CSV 数据

正如你所见，手动解析许多不同的字段并逐行执行操作可能会相当冗长且繁琐。这绝对不是增加复杂性和导入大量非标准功能的借口。在大多数情况下，你应该仍然默认使用`encoding/csv`。

然而，数据框的操作已被证明是处理表格数据的一种成功且相对标准化的方式（在数据科学社区中）。因此，在某些情况下，使用一些第三方功能来操作表格数据，如 CSV 数据，是值得的。例如，数据框及其对应的功能在你尝试过滤、子集化和选择表格数据集的部分时非常有用。在本节中，我们将介绍`github.com/kniren/gota/dataframe`，这是一个为 Go 语言提供的优秀的`dataframe`包：

```py
import "github.com/kniren/gota/dataframe" 
```

要从 CSV 文件创建数据框，我们使用`os.Open()`打开一个文件，然后将返回的指针提供给`dataframe.ReadCSV()`函数：

```py
// Open the CSV file.
irisFile, err := os.Open("iris.csv")
if err != nil {
    log.Fatal(err)
}
defer irisFile.Close()

// Create a dataframe from the CSV file.
// The types of the columns will be inferred.
irisDF := dataframe.ReadCSV(irisFile)

// As a sanity check, display the records to stdout.
// Gota will format the dataframe for pretty printing.
fmt.Println(irisDF)
```

如果我们编译并运行这个 Go 程序，我们将看到一个漂亮的、格式化的数据版本，其中包含了在解析过程中推断出的类型：

```py
$ go build
$ ./myprogram
[150x5] DataFrame

 sepal_length sepal_width petal_length petal_width species 
 0: 5.100000 3.500000 1.400000 0.200000 Iris-setosa
 1: 4.900000 3.000000 1.400000 0.200000 Iris-setosa
 2: 4.700000 3.200000 1.300000 0.200000 Iris-setosa
 3: 4.600000 3.100000 1.500000 0.200000 Iris-setosa
 4: 5.000000 3.600000 1.400000 0.200000 Iris-setosa
 5: 5.400000 3.900000 1.700000 0.400000 Iris-setosa
 6: 4.600000 3.400000 1.400000 0.300000 Iris-setosa
 7: 5.000000 3.400000 1.500000 0.200000 Iris-setosa
 8: 4.400000 2.900000 1.400000 0.200000 Iris-setosa
 9: 4.900000 3.100000 1.500000 0.100000 Iris-setosa
 ... ... ... ... ... 
 <float> <float> <float> <float> <string>
```

一旦我们将数据解析到`dataframe`中，我们就可以轻松地进行过滤、子集化和选择我们的数据：

```py
// Create a filter for the dataframe.
filter := dataframe.F{
    Colname: "species",
    Comparator: "==",
    Comparando: "Iris-versicolor",
}

// Filter the dataframe to see only the rows where
// the iris species is "Iris-versicolor".
versicolorDF := irisDF.Filter(filter)
if versicolorDF.Err != nil {
    log.Fatal(versicolorDF.Err)
}

// Filter the dataframe again, but only select out the
// sepal_width and species columns.
versicolorDF = irisDF.Filter(filter).Select([]string{"sepal_width", "species"})

// Filter and select the dataframe again, but only display
// the first three results.
versicolorDF = irisDF.Filter(filter).Select([]string{"sepal_width", "species"}).Subset([]int{0, 1, 2})
```

这实际上只是对`github.com/kniren/gota/dataframe`包表面的探索。你可以合并数据集，输出到其他格式，甚至处理 JSON 数据。关于这个包的更多信息，你应该访问自动生成的 GoDocs，网址为[`godoc.org/github.com/kniren/gota/dataframe`](https://godoc.org/github.com/kniren/gota/dataframe)，这在一般情况下，对于我们在书中讨论的任何包来说都是好的实践。

# JSON

在一个大多数数据都是通过网络访问的世界里，大多数工程组织实施了一定数量的微服务，我们将非常频繁地遇到 JSON 格式的数据。我们可能只需要在从 API 中拉取一些随机数据时处理它，或者它实际上可能是驱动我们的分析和机器学习工作流程的主要数据格式。

通常，当易用性是数据交换的主要目标时，会使用 JSON。由于 JSON 是可读的，如果出现问题，它很容易调试。记住，我们希望在用 Go 处理数据时保持我们数据处理的一致性，这个过程的一部分是确保，当可能时，我们的数据是可解释和可读的。JSON 在实现这些目标方面非常有用（这也是为什么它也常用于日志记录）。

Go 在其标准库中提供了非常好的 JSON 功能，使用`encoding/json`。我们将在整个书中利用这个标准库功能。

# 解析 JSON

要了解如何在 Go 中解析（即反序列化）JSON 数据，我们将使用来自 Citi Bike API（[`www.citibikenyc.com/system-data`](https://www.citibikenyc.com/system-data)）的一些数据，这是一个在纽约市运营的自行车共享服务。Citi Bike 以 JSON 格式提供其自行车共享站点的频繁更新的运营信息，网址为[`gbfs.citibikenyc.com/gbfs/en/station_status.json`](https://gbfs.citibikenyc.com/gbfs/en/station_status.json)：

```py
{
  "last_updated": 1495252868,
  "ttl": 10,
  "data": {
    "stations": [
      {
        "station_id": "72",
        "num_bikes_available": 10,
        "num_bikes_disabled": 3,
        "num_docks_available": 26,
        "num_docks_disabled": 0,
        "is_installed": 1,
        "is_renting": 1,
        "is_returning": 1,
        "last_reported": 1495249679,
        "eightd_has_available_keys": false
      },
      {
        "station_id": "79",
        "num_bikes_available": 0,
        "num_bikes_disabled": 0,
        "num_docks_available": 33,
        "num_docks_disabled": 0,
        "is_installed": 1,
        "is_renting": 1,
        "is_returning": 1,
        "last_reported": 1495248017,
        "eightd_has_available_keys": false
      },

      etc...

      {
        "station_id": "3464",
        "num_bikes_available": 1,
        "num_bikes_disabled": 3,
        "num_docks_available": 53,
        "num_docks_disabled": 0,
        "is_installed": 1,
        "is_renting": 1,
        "is_returning": 1,
        "last_reported": 1495250340,
        "eightd_has_available_keys": false
      }
    ]
  }
}
```

在 Go 中解析导入和这种类型的数据时，我们首先需要导入`encoding/json`（以及从标准库中的一些其他东西，如`net/http`，因为我们将从之前提到的网站上拉取这些数据）。我们还将定义`struct`，它模仿了前面代码中显示的 JSON 结构：

```py
import (
    "encoding/json"
    "fmt"
    "io/ioutil"
    "log"
    "net/http"
)

// citiBikeURL provides the station statuses of CitiBike bike sharing stations.
const citiBikeURL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

// stationData is used to unmarshal the JSON document returned form citiBikeURL.
type stationData struct {
    LastUpdated int `json:"last_updated"`
    TTL int `json:"ttl"`
    Data struct {
        Stations []station `json:"stations"`
    } `json:"data"`
}

// station is used to unmarshal each of the station documents in stationData.
type station struct {
    ID string `json:"station_id"`
    NumBikesAvailable int `json:"num_bikes_available"`
    NumBikesDisabled int `json:"num_bike_disabled"`
    NumDocksAvailable int `json:"num_docks_available"`
    NumDocksDisabled int `json:"num_docks_disabled"`
    IsInstalled int `json:"is_installed"`
    IsRenting int `json:"is_renting"`
    IsReturning int `json:"is_returning"`
    LastReported int `json:"last_reported"`
    HasAvailableKeys bool `json:"eightd_has_available_keys"`
}
```

注意这里的一些事情：（i）我们遵循了 Go 的惯例，避免了使用下划线的`struct`字段名，但（ii）我们使用了`json`结构标签来标记`struct`字段，以对应 JSON 数据中的预期字段。

注意，为了正确解析 JSON 数据，结构体字段必须是导出字段。也就是说，字段需要以大写字母开头。`encoding/json`无法使用反射查看未导出的字段。

现在，我们可以从 URL 获取 JSON 数据并将其反序列化到一个新的`stationData`值中。这将产生一个`struct`变量，其相应字段填充了标记的 JSON 数据字段中的数据。我们可以通过打印与某个站点相关的一些数据来检查它：

```py
// Get the JSON response from the URL.
response, err := http.Get(citiBikeURL)
if err != nil {
    log.Fatal(err)
}
defer response.Body.Close()

// Read the body of the response into []byte.
body, err := ioutil.ReadAll(response.Body)
if err != nil {
    log.Fatal(err)
}

// Declare a variable of type stationData.
var sd stationData

// Unmarshal the JSON data into the variable.
if err := json.Unmarshal(body, &sd); err != nil {
    log.Fatal(err)
}

// Print the first station.
fmt.Printf("%+v\n\n", sd.Data.Stations[0])
```

当我们运行此操作时，我们可以看到我们的`struct`包含了从 URL 解析的数据：

```py
$ go build
$ ./myprogram 
{ID:72 NumBikesAvailable:11 NumBikesDisabled:0 NumDocksAvailable:25 NumDocksDisabled:0 IsInstalled:1 IsRenting:1 IsReturning:1 LastReported:1495252934 HasAvailableKeys:false}
```

# JSON 输出

现在假设我们已经在`stationData`结构体值中有了 Citi Bike 站点的数据，并希望将数据保存到文件中。我们可以使用`json.marshal`来完成此操作：

```py
// Marshal the data.
outputData, err := json.Marshal(sd)
if err != nil {
    log.Fatal(err)
}

// Save the marshalled data to a file.
if err := ioutil.WriteFile("citibike.json", outputData, 0644); err != nil {
    log.Fatal(err)
}
```

# 类似 SQL 的数据库

尽管围绕有趣的 NoSQL 数据库和键值存储有很多炒作，但类似 SQL 的数据库仍然无处不在。每个数据科学家在某个时候都会处理来自类似 SQL 的数据库的数据，例如 Postgres、MySQL 或 SQLite。

例如，我们可能需要查询 Postgres 数据库中的一个或多个表来生成用于模型训练的一组特征。在用该模型进行预测或识别异常之后，我们可能将结果发送到另一个数据库表，该表驱动仪表板或其他报告工具。

当然，Go 与所有流行的数据存储都很好地交互，例如 SQL、NoSQL、键值存储等，但在这里，我们将专注于类似 SQL 的交互。在整个书中，我们将使用`database/sql`进行这些交互。

# 连接到 SQL 数据库

在连接类似 SQL 的数据库之前，我们需要做的第一件事是确定我们将与之交互的特定数据库，并导入相应的驱动程序。在以下示例中，我们将连接到 Postgres 数据库，并将使用`github.com/lib/pq`数据库驱动程序来处理`database/sql`。此驱动程序可以通过空导入（带有相应的注释）来加载：

```py
import (
    "database/sql"
    "fmt"
    "log"
    "os"

    // pq is the library that allows us to connect
    // to postgres with databases/sql.
    _ "github.com/lib/pq"
)
```

现在假设您已经将 Postgres 连接字符串导出到环境变量`PGURL`中。我们可以通过以下代码轻松地为我们的连接创建一个`sql.DB`值：

```py
// Get the postgres connection URL. I have it stored in
// an environmental variable.
pgURL := os.Getenv("PGURL")
if pgURL == "" {
    log.Fatal("PGURL empty")
}

// Open a database value. Specify the postgres driver
// for databases/sql.
db, err := sql.Open("postgres", pgURL)
if err != nil {
    log.Fatal(err)
}
defer db.Close()
```

注意，我们需要延迟此值的`close`方法。另外，请注意，创建此值并不意味着您已成功连接到数据库。这只是一个由`database/sql`在触发某些操作（如查询）时用于连接数据库的值。

为了确保我们可以成功连接到数据库，我们可以使用`Ping`方法：

```py
if err := db.Ping(); err != nil {
    log.Fatal(err)
}
```

# 查询数据库

现在我们知道了如何连接到数据库，让我们看看我们如何从数据库中获取数据。在这本书中，我们不会涵盖 SQL 查询和语句的细节。如果您不熟悉 SQL，我强烈建议您学习如何查询、插入等，但就我们这里的目的而言，您应该知道我们基本上有两种类型的操作与 SQL 数据库相关：

+   `Query`操作在数据库中选取、分组或聚合数据，并将数据行返回给我们

+   `Exec`操作更新、插入或以其他方式修改数据库的状态，而不期望数据库中存储的数据的部分应该被返回

如您所预期的那样，为了从我们的数据库中获取数据，我们将使用`Query`操作。为此，我们需要使用 SQL 语句字符串查询数据库。例如，假设我们有一个存储大量鸢尾花测量数据（花瓣长度、花瓣宽度等）的数据库，我们可以查询与特定鸢尾花物种相关的数据如下：

```py
// Query the database.
rows, err := db.Query(`
    SELECT 
        sepal_length as sLength, 
        sepal_width as sWidth, 
        petal_length as pLength, 
        petal_width as pWidth 
    FROM iris
    WHERE species = $1`, "Iris-setosa")
if err != nil {
    log.Fatal(err)
}
defer rows.Close()
```

注意，这返回了一个指向`sql.Rows`值的指针，我们需要延迟关闭这个行值。然后我们可以遍历我们的行并将数据解析为预期的类型。我们利用`Scan`方法在行上解析 SQL 查询返回的列并将它们打印到标准输出：

```py
// Iterate over the rows, sending the results to
// standard out.
for rows.Next() {

    var (
        sLength float64
        sWidth float64
        pLength float64
        pWidth float64
    )

    if err := rows.Scan(&sLength, &sWidth, &pLength, &pWidth); err != nil {
        log.Fatal(err)
    }

    fmt.Printf("%.2f, %.2f, %.2f, %.2f\n", sLength, sWidth, pLength, pWidth)
}
```

最后，我们需要检查在处理我们的行时可能发生的任何错误。我们希望保持我们数据处理的一致性，我们不能假设我们在没有遇到错误的情况下遍历了所有的行：

```py
// Check for errors after we are done iterating over rows.
if err := rows.Err(); err != nil {
    log.Fatal(err)
}
```

# 修改数据库

如前所述，还有另一种与数据库的交互方式称为`Exec`。使用这些类型的语句，我们关注的是更新、添加或以其他方式修改数据库中的一个或多个表的状态。我们使用相同类型的数据库连接，但不是调用`db.Query`，我们将调用`db.Exec`。

例如，假设我们想要更新我们 iris 数据库表中的某些值：

```py
// Update some values.
res, err := db.Exec("UPDATE iris SET species = 'setosa' WHERE species = 'Iris-setosa'")
if err != nil {
    log.Fatal(err)
}
```

但我们如何知道我们是否成功并改变了某些内容呢？嗯，这里返回的`res`函数允许我们查看我们的表中有多少行受到了我们更新的影响：

```py
// See how many rows where updated.
rowCount, err := res.RowsAffected()
if err != nil {
    log.Fatal(err)
}

// Output the number of rows to standard out.
log.Printf("affected = %d\n", rowCount)
```

# 缓存

有时，我们的机器学习算法将通过外部来源（例如，API）的数据进行训练和/或提供预测输入，即不是运行我们的建模或分析的应用程序本地的数据。此外，我们可能有一些经常访问的数据集，可能很快会再次访问，或者可能需要在应用程序运行时提供。

在至少这些情况中，缓存数据在内存中或嵌入到应用程序运行的地方可能是合理的。例如，如果你经常访问政府 API（通常具有高延迟）以获取人口普查数据，你可能会考虑维护一个本地或内存中的缓存，以便你可以避免不断调用 API。

# 在内存中缓存数据

要在内存中缓存一系列值，我们将使用 `github.com/patrickmn/go-cache`。使用这个包，我们可以创建一个包含键和相应值的内存缓存。我们甚至可以指定缓存中特定键值对的时间生存期。

要创建一个新的内存缓存并在缓存中设置键值对，我们执行以下操作：

```py
// Create a cache with a default expiration time of 5 minutes, and which
// purges expired items every 30 seconds
c := cache.New(5*time.Minute, 30*time.Second)

// Put a key and value into the cache.
c.Set("mykey", "myvalue", cache.DefaultExpiration)
```

要从缓存中检索 `mykey` 的值，我们只需使用 `Get` 方法：

```py
v, found := c.Get("mykey")
if found {
    fmt.Printf("key: mykey, value: %s\n", v)
}
```

# 在磁盘上本地缓存数据

我们刚才看到的缓存是在内存中的。也就是说，缓存的数据在应用程序运行时存在并可访问，但一旦应用程序退出，数据就会消失。在某些情况下，你可能希望当你的应用程序重新启动或退出时，缓存的数据仍然保留。你也可能想要备份你的缓存，这样你就不需要在没有相关数据缓存的情况下从头开始启动应用程序。

在这些情况下，你可能考虑使用本地的嵌入式缓存，例如 `github.com/boltdb/bolt`。BoltDB，正如其名，是这类应用中非常受欢迎的项目，基本上由一个本地的键值存储组成。要初始化这些本地键值存储之一，请执行以下操作：

```py
// Open an embedded.db data file in your current directory.
// It will be created if it doesn't exist.
db, err := bolt.Open("embedded.db", 0600, nil)
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// Create a "bucket" in the boltdb file for our data.
if err := db.Update(func(tx *bolt.Tx) error {
    _, err := tx.CreateBucket([]byte("MyBucket"))
    if err != nil {
        return fmt.Errorf("create bucket: %s", err)
    }
    return nil
}); err != nil {
    log.Fatal(err)
}
```

当然，你可以在 BoltDB 中拥有多个不同的数据桶，并使用除 `embedded.db` 之外的其他文件名。

接下来，假设你有一个内存中的字符串值映射，你需要将其缓存到 BoltDB 中。为此，你需要遍历映射中的键和值，更新你的 BoltDB：

```py
// Put the map keys and values into the BoltDB file.
if err := db.Update(func(tx *bolt.Tx) error {
    b := tx.Bucket([]byte("MyBucket"))
    err := b.Put([]byte("mykey"), []byte("myvalue"))
    return err
}); err != nil {
    log.Fatal(err)
}
```

然后，要从 BoltDB 中获取值，你可以查看你的数据：

```py
// Output the keys and values in the embedded
// BoltDB file to standard out.
if err := db.View(func(tx *bolt.Tx) error {
    b := tx.Bucket([]byte("MyBucket"))
    c := b.Cursor()
    for k, v := c.First(); k != nil; k, v = c.Next() {
        fmt.Printf("key: %s, value: %s\n", k, v)
    }
    return nil
}); err != nil {
    log.Fatal(err)
}
```

# 数据版本控制

如前所述，机器学习模型产生的结果极其不同，这取决于你使用的训练数据、参数的选择和输入数据。为了协作、创造性和合规性原因，能够重现结果是至关重要的：

+   **协作**：尽管你在社交媒体上看到的是，没有数据科学和机器学习独角兽（即在每个数据科学和机器学习领域都有知识和能力的人）。我们需要同事的审查并改进我们的工作，而如果他们无法重现我们的模型结果和分析，这是不可能的。

+   **创造力**：我不知道你怎么样，但即使是我也难以记住昨天做了什么。我们无法信任自己总是能记住我们的推理和逻辑，尤其是在处理机器学习工作流程时。我们需要精确跟踪我们使用的数据、我们创建的结果以及我们是如何创建它们的。这是我们能够不断改进我们的模型和技术的方式。

+   **合规性**：最后，我们可能很快就会在机器学习中没有选择地进行数据版本化和可重现性。世界各地正在通过法律（例如，欧盟的**通用数据保护条例**（GDPR））赋予用户对算法决策的解释权。如果我们没有一种稳健的方式来跟踪我们正在处理的数据和产生的结果，我们根本无法希望遵守这些裁决。

存在多个开源数据版本控制项目。其中一些专注于数据的安全性和对等分布式存储。其他一些则专注于数据科学工作流程。在这本书中，我们将重点关注并利用 Pachyderm ([`pachyderm.io/`](http://pachyderm.io/))，这是一个开源的数据版本控制和数据管道框架。其中一些原因将在本书后面关于生产部署和管理 ML 管道时变得清晰。现在，我将仅总结一些使 Pachyderm 成为基于 Go（和其他）ML 项目数据版本控制吸引力的特性：

+   它有一个方便的 Go 客户端，`github.com/pachyderm/pachyderm/src/client`

+   能够对任何类型和格式的数据进行版本控制

+   为版本化数据提供灵活的对象存储后端

+   与数据管道系统集成以驱动版本化的 ML 工作流程

# Pachyderm 术语

将 Pachyderm 中的数据版本化想象成在 Git 中版本化代码。基本原理是相似的：

+   **仓库**：这些是版本化的数据集合，类似于在 Git 仓库中拥有版本化的代码集合

+   **提交**：在 Pachyderm 中，通过将数据提交到数据仓库来对数据进行版本控制

+   **分支**：这些轻量级指针指向特定的提交或一系列提交（例如，master 指向最新的 HEAD 提交）

+   **文件**：在 Pachyderm 中，数据在文件级别进行版本控制，并且 Pachyderm 自动采用去重等策略来保持你的版本化数据空间高效

尽管使用 Pachyderm 对数据进行版本控制的感觉与使用 Git 对代码进行版本控制相似，但也有一些主要区别。例如，合并数据并不完全有意义。如果存在数 PB（皮字节）数据的合并冲突，没有人能够解决这些问题。此外，Git 协议在处理大量数据时通常不会很节省空间。Pachyderm 使用其自身的内部逻辑来执行版本控制和处理版本化数据，这种逻辑在缓存方面既节省空间又高效。

# 部署/安装 Pachyderm

我们将在本书的多个地方使用 Pachyderm 来对数据进行版本控制并创建分布式机器学习工作流程。Pachyderm 本身是一个运行在 Kubernetes([`kubernetes.io/`](https://kubernetes.io/))之上的应用程序，并支持你选择的任何对象存储。为了本书的开发和实验目的，你可以轻松地安装并本地运行 Pachyderm。安装应该需要 5-10 分钟，并且不需要太多努力。本地安装的说明可以在 Pachyderm 文档中找到，网址为[`docs.pachyderm.io`](http://docs.pachyderm.io)。

当你准备好在生产环境中运行你的工作流程或部署模型时，你可以轻松地部署一个生产就绪的 Pachyderm 集群，该集群将与你本地安装的行为完全相同。Pachyderm 可以部署到任何云中，甚至可以在本地部署。

如前所述，Pachyderm 是一个开源项目，并且有一个活跃的用户群体。如果你有问题或需要帮助，你可以通过访问[`slack.pachyderm.io/`](http://slack.pachyderm.io/)加入公共 Pachyderm Slack 频道。活跃的 Pachyderm 用户和 Pachyderm 团队本身将能够快速回答你的问题。

# 为数据版本化创建数据仓库

如果你遵循了 Pachyderm 文档中指定的本地安装说明，你应该有以下内容：

+   在你的机器上的 Minikube VM 上运行的 Kubernetes

+   已安装并连接到你的 Pachyderm 集群的`pachctl`命令行工具

当然，如果你在云中运行一个生产集群，以下步骤仍然适用。你的`pachctl`将连接到远程集群。

我们将在下面的示例中使用`pachctl` **命令行界面**（**CLI**）（这是一个 Go 程序）来演示数据版本化功能。然而，如上所述，Pachyderm 有一个完整的 Go 客户端。你可以直接从你的 Go 程序中创建仓库、提交数据等等。这一功能将在第九章*部署和分发分析和模型*中演示。

要创建一个名为`myrepo`的数据仓库，你可以运行以下代码：

```py
$ pachctl create-repo myrepo
```

你可以使用`list-repo`来确认仓库是否存在：

```py
$ pachctl list-repo
NAME CREATED SIZE 
myrepo 2 seconds ago 0 B
```

这个`myrepo`仓库是我们定义的数据集合，已准备好存放版本化的数据。目前，仓库中没有数据，因为我们还没有放入任何数据。

# 将数据放入数据仓库

假设我们有一个简单的文本文件：

```py
$ cat blah.txt 
This is an example file.
```

如果这个文件是我们正在利用的机器学习工作流程中的数据的一部分，我们应该对其进行版本控制。要在我们的仓库`myrepo`中对此文件进行版本控制，我们只需将其提交到该仓库：

```py
$ pachctl put-file myrepo master -c -f blah.txt 
```

`-c`标志指定我们希望 Pachyderm 打开一个新提交，插入我们引用的文件，然后一次性关闭提交。`-f`标志指定我们提供了一个文件。

注意，我们在这里是将单个文件提交到单个仓库的 master 分支。然而，Pachyderm API 非常灵活。我们可以在单个提交或多个提交中提交、删除或以其他方式修改许多版本化文件。此外，这些文件可以通过 URL、对象存储链接、数据库转储等方式进行版本化。

作为一种合理性检查，我们可以确认我们的文件已在仓库中进行了版本化：

```py
$ pachctl list-repo
NAME CREATED SIZE 
myrepo 10 minutes ago 25 B 
$ pachctl list-file myrepo master
NAME TYPE SIZE 
blah.txt file 25 B
```

# 从版本化数据仓库中获取数据

现在我们已经在 Pachyderm 中有了版本化的数据，我们可能想知道如何与这些数据交互。主要的方式是通过 Pachyderm 数据管道（本书后面将讨论）。在管道中使用时与版本化数据交互的机制是一个简单的文件 I/O。

然而，如果我们想手动从 Pachyderm 中提取某些版本的版本化数据，进行交互式分析，那么我们可以使用`pachctl` CLI 来获取数据：

```py
$ pachctl get-file myrepo master blah.txt
This is an example file.
```

# 参考文献

CSV 数据：

+   `encoding/csv` 文档：[`golang.org/pkg/encoding/csv/`](https://golang.org/pkg/encoding/csv/)

+   `github.com/kniren/gota/dataframe` 文档：[`godoc.org/github.com/kniren/gota/dataframe`](https://godoc.org/github.com/kniren/gota/dataframe)

JSON 数据：

+   `encoding/json` 文档：[`golang.org/pkg/encoding/json/`](https://golang.org/pkg/encoding/json/)

+   Bill Kennedy 的博客文章 JSON 解码：[`www.goinggo.net/2014/01/decode-json-documents-in-go.html`](https://www.goinggo.net/2014/01/decode-json-documents-in-go.html)

+   Ben Johnson 的博客文章 Go Walkthrough：`encoding/json`包：[`medium.com/go-walkthrough/go-walkthrough-encoding-json-package-9681d1d37a8f`](https://medium.com/go-walkthrough/go-walkthrough-encoding-json-package-9681d1d37a8f)

缓存：

+   `github.com/patrickmn/go-cache` 文档：[`godoc.org/github.com/patrickmn/go-cache`](https://godoc.org/github.com/patrickmn/go-cache)

+   `github.com/boltdb/bolt` 文档：[`godoc.org/github.com/boltdb/bolt`](https://godoc.org/github.com/boltdb/bolt)

+   BoltDB 的相关信息和动机：[`npf.io/2014/07/intro-to-boltdb-painless-performant-persistence/`](https://npf.io/2014/07/intro-to-boltdb-painless-performant-persistence/)

Pachyderm：

+   通用文档：[`docs.pachyderm.io`](http://docs.pachyderm.io)

+   Go 客户端文档：[`godoc.org/github.com/pachyderm/pachyderm/src/client`](https://godoc.org/github.com/pachyderm/pachyderm/src/client)

+   公共用户 Slack 团队注册：[`docs.pachyderm.io`](http://docs.pachyderm.io)

# 摘要

在本章中，你学习了如何收集、组织和解析数据。这是开发机器学习模型的第一步，也是最重要的一步，但如果我们不对数据进行一些直观的理解并将其放入标准形式进行处理，那么拥有数据也不会让我们走得很远。接下来，我们将探讨一些进一步结构化我们的数据（矩阵）和理解我们的数据（统计学和概率）的技术。
