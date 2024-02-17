# 列表和数组

> 原文：[`allendowney.github.io/ElementsOfDataScience/03_arrays.html`](https://allendowney.github.io/ElementsOfDataScience/03_arrays.html)

[单击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/03_arrays.ipynb) 或 [单击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/03_arrays.ipynb)。

在上一章中，我们使用元组表示纬度和经度。在本章中，您将看到如何更一般地使用元组表示一系列值。我们还将看到另外两种表示序列的方法：列表和数组。

您可能会想知道为什么我们需要三种方式来表示相同的事物。大多数情况下我们不需要，但它们每个都有不同的功能。在处理数据时，我们大多数时候会使用数组。

例如，我们将使用《经济学人》一篇关于三明治价格的文章中的一个小数据集。这是一个愚蠢的例子，但我会用它来介绍相对差异和不同的总结方法。

## 元组

元组是一系列元素。当我们使用元组表示纬度和经度时，序列只包含两个元素，它们都是浮点数。但一般来说，元组可以包含任意数量的元素，并且元素可以是任何类型的值。以下是一个包含三个整数的元组：

```py
1, 2, 3 
```

```py
(1, 2, 3) 
```

请注意，当 Python 显示一个元组时，它会将元素放在括号中。当您输入一个元组时，如果您认为这样更容易阅读，可以将其放在括号中，但不是必须的。

元素可以是任何类型。这是一个字符串元组：

```py
'Data', 'Science' 
```

```py
('Data', 'Science') 
```

元素不必是相同的类型。这是一个包含字符串、整数和浮点数的元组。

```py
'one', 2, 3.14159 
```

```py
('one', 2, 3.14159) 
```

如果您有一个字符串，可以使用`tuple`函数将其转换为元组：

```py
tuple('DataScience') 
```

```py
('D', 'a', 't', 'a', 'S', 'c', 'i', 'e', 'n', 'c', 'e') 
```

结果是一个单字符字符串的元组。

创建元组时，括号是可选的，但逗号是必需的。那么您如何创建一个只有一个元素的元组呢？您可能会想要写：

```py
x = (5)
x 
```

```py
5 
```

但您会发现结果只是一个数字，而不是一个元组。要创建一个只有一个元素的元组，您需要一个逗号：

```py
t = 5,
t 
```

```py
(5,) 
```

这可能看起来有点滑稽，但它确实能胜任。

## 列表

Python 提供了另一种存储一系列元素的方式：**列表**。

要创建一个列表，您需要将一系列元素放在方括号中。

```py
[1, 2, 3] 
```

```py
[1, 2, 3] 
```

列表和元组非常相似。它们可以包含任意数量的元素，元素可以是任何类型，并且元素不必是相同的类型。不同之处在于您可以修改列表；元组是**不可变的**（无法修改）。这种差异以后会很重要，但现在我们可以忽略它。

当您创建一个列表时，方括号是必需的，但如果只有一个元素，您不需要逗号。所以您可以创建一个像这样的列表：

```py
single = [5] 
```

也可以创建一个没有元素的列表，就像这样：

```py
empty = [] 
```

`len`函数返回列表或元组中的长度（元素数量）。

```py
len([1, 2, 3]), len(single), len(empty) 
```

```py
(3, 1, 0) 
```

**练习：**创建一个包含 4 个元素的列表；然后使用`type`确认它是一个列表，使用`len`确认它有 4 个元素。

我们可以用列表做更多的事情，但这已经足够开始了。在下一节中，我们将使用列表来存储有关三明治价格的数据。

## 三明治价格

2019 年 9 月，《经济学人》发表了一篇比较波士顿和伦敦三明治价格的文章：“[为什么美国人午餐比英国人贵](https://www.economist.com/finance-and-economics/2019/09/07/why-americans-pay-more-for-lunch-than-britons-do)” 。其中包括这张图，显示了两个城市的几种三明治的价格：

![](img/5feb03c573c0b858be44a006b68b3324.png)

这是图表中的三明治名称，作为字符串列表。

```py
name_list = [
    'Lobster roll',
    'Chicken caesar',
    'Bang bang chicken',
    'Ham and cheese',
    'Tuna and cucumber',
    'Egg'
] 
```

我联系了《经济学人》，要求他们提供用于创建该图表的数据，他们很友好地与我分享了。以下是波士顿的三明治价格：

```py
boston_price_list = [9.99, 7.99, 7.49, 7.00, 6.29, 4.99] 
```

以下是伦敦的价格，以 1.25 美元/英镑的汇率转换成美元。

```py
london_price_list = [7.5, 5, 4.4, 5, 3.75, 2.25] 
```

列表提供了一些算术运算符，但它们可能不会得到你想要的结果。例如，你可以“相加”两个列表：

```py
boston_price_list + london_price_list 
```

```py
[9.99, 7.99, 7.49, 7.0, 6.29, 4.99, 7.5, 5, 4.4, 5, 3.75, 2.25] 
```

但它连接了这两个列表，在这个例子中并不是很有用。要计算价格之间的差异，你可以尝试减去列表，但会得到一个错误。

我们可以用 NumPy 解决这个问题。

## NumPy 数组

我们已经看到 NumPy 库提供了数学函数。它还提供了一种称为**数组**的序列类型。你可以使用`np.array`函数创建一个新数组，从一个列表或元组开始。

```py
import numpy as np

boston_price_array = np.array(boston_price_list)
london_price_array = np.array(london_price_list) 
```

结果的类型是`numpy.ndarray`。

```py
type(boston_price_array) 
```

```py
numpy.ndarray 
```

“nd”代表“n 维”，这表明 NumPy 数组可以有任意数量的维度。但现在我们将使用一维序列。如果你显示一个数组，Python 会显示元素：

```py
boston_price_array 
```

```py
array([9.99, 7.99, 7.49, 7\.  , 6.29, 4.99]) 
```

你还可以显示数组的**数据类型**，即元素的类型：

```py
boston_price_array.dtype 
```

```py
dtype('float64') 
```

`float64`表示元素是占据 64 位的浮点数。你不需要了解这些数字的存储格式，但如果你感兴趣，可以在[`en.wikipedia.org/wiki/Floating-point_arithmetic#Internal_representation`](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Internal_representation)上阅读相关内容。

NumPy 数组的元素可以是任何类型，但它们都必须是相同的类型。最常见的情况是元素是数字，但你也可以创建一个字符串数组。

```py
name_array = np.array(name_list)
name_array 
```

```py
array(['Lobster roll', 'Chicken caesar', 'Bang bang chicken',
       'Ham and cheese', 'Tuna and cucumber', 'Egg'], dtype='<U17') 
```

在这个例子中，`dtype`是`<U17`。`U`表示元素是 Unicode 字符串；Unicode 是 Python 用来表示字符串的标准。数字`17`是数组中最长字符串的长度。

现在，这就是为什么 NumPy 数组很有用的原因：它们可以进行算术运算。例如，要计算波士顿和伦敦价格之间的差异，我们可以这样写：

```py
differences = boston_price_array - london_price_array
differences 
```

```py
array([2.49, 2.99, 3.09, 2\.  , 2.54, 2.74]) 
```

减法是**逐元素**进行的；也就是说，NumPy 将两个数组对齐并相减对应的元素。结果是一个新数组。

## 统计摘要

NumPy 提供了计算统计摘要的函数，比如均值：

```py
np.mean(differences) 
```

```py
2.6416666666666666 
```

因此，我们可以这样描述价格的差异：“波士顿的三明治平均更贵 2.64 美元”。我们也可以先计算均值，然后计算它们的差异：

```py
np.mean(boston_price_array) - np.mean(london_price_array) 
```

```py
2.6416666666666675 
```

结果是一样的：均值的差异与差异的均值是一样的。顺便说一句，许多 NumPy 函数也适用于列表，所以我们也可以这样做：

```py
np.mean(boston_price_list) - np.mean(london_price_list) 
```

```py
2.6416666666666675 
```

**练习：**标准差是衡量一组数字变异性的方法。计算标准差的 NumPy 函数是`np.std`。

计算波士顿和伦敦三明治价格的标准差。按照这个度量，哪组价格更具变异性？

**练习：**均值的定义，在数学符号中，是

\(\mu = \frac{1}{N} \sum_i x_i\)

其中\(x\)是一个元素序列，\(x_i\)是索引为\(i\)的元素，\(N\)是元素的数量。标准差的定义是

\(\sigma = \sqrt{\frac{1}{N} \sum_i (x_i - \mu)²}\)

使用 NumPy 函数`np.mean`和`np.sqrt`计算`boston_price_list`的标准差，看看是否得到与`np.std`相同的结果。

你可以（也应该）只使用我们到目前为止讨论过的功能来完成这个练习。

这种标准差的定义有时被称为“总体标准差”。你可能见过分母中有\(N-1\)的另一种定义；那是“样本标准差”。我们现在使用总体标准差，稍后再回到这个问题。

## 相对差异

在前面的部分中，我们计算了价格之间的差异。但通常当我们进行这种比较时，我们对**相对差异**感兴趣，这是以某个数量的分数或百分比表示的差异。

以龙虾卷为例，价格的差异是：

```py
9.99 - 7.5 
```

```py
2.49 
```

我们可以将这种差异表示为伦敦价格的一部分，就像这样：

```py
(9.99 - 7.5) / 7.5 
```

```py
0.332 
```

或者作为伦敦价格的*百分比*，就像这样：

```py
(9.99 - 7.5) / 7.5 * 100 
```

```py
33.2 
```

因此，我们可以说波士顿的龙虾卷贵了 33%。但将伦敦放在分母是一个任意的选择。我们也可以将差异计算为波士顿价格的百分比：

```py
(9.99 - 7.5) / 9.99 * 100 
```

```py
24.924924924924927 
```

如果我们进行这样的计算，我们可能会说在伦敦龙虾卷要便宜 25%。当你阅读这种比较时，你应该确保你理解了分母是哪个量，并且你可能想想为什么会做出这样的选择。在这个例子中，如果你想让差异看起来更大，你可能会把伦敦的价格放在分母中。

如果我们用`boston_price_array`和`boston_price_array`数组进行相同的计算，我们可以计算所有三明治的相对差异：

```py
differences = boston_price_array - london_price_array
relative_differences = differences / london_price_array
relative_differences 
```

```py
array([0.332     , 0.598     , 0.70227273, 0.4       , 0.67733333,
       1.21777778]) 
```

以及百分差异。

```py
percent_differences = relative_differences * 100
percent_differences 
```

```py
array([ 33.2       ,  59.8       ,  70.22727273,  40\.        ,
        67.73333333, 121.77777778]) 
```

## 总结相对差异

现在让我们考虑如何总结百分差异的数组。一个选择是报告范围，我们可以用`np.min`和`np.max`来计算。

```py
np.min(percent_differences), np.max(percent_differences) 
```

```py
(33.2, 121.77777777777779) 
```

在波士顿，龙虾卷只贵了 33%；而鸡蛋三明治贵了 121%（也就是说，价格是原来的两倍多）。

**练习：**如果我们把波士顿的价格放在分母，百分差异是多少？这些差异的范围是多少？写一句话总结结果。

总结百分差异的另一种方法是报告平均值。

```py
np.mean(percent_differences) 
```

```py
65.4563973063973 
```

因此，我们可以说，平均而言，波士顿的三明治价格要贵 65%。但另一种总结数据的方法是计算每个城市的平均价格，然后计算平均数的百分比差异：

```py
boston_mean = np.mean(boston_price_array)
london_mean = np.mean(london_price_array)

(boston_mean - london_mean) / london_mean * 100 
```

```py
56.81003584229393 
```

因此，我们可以说波士顿的三明治平均价格要高 56%。正如这个例子所示：

+   对于相对和百分差异，差异的平均值与平均数的差异不同。

+   当你报告这样的数据时，你应该考虑不同的总结数据的方法。

+   当你阅读这样的数据总结时，确保你理解选择了什么总结以及它的含义。

在这个例子中，我认为第二个选择（平均数的相对差异）更有意义，因为它反映了包括每种三明治的“商品篮子”之间的价格差异。

## 调试

到目前为止，大多数练习只需要几行代码。如果你在这个过程中犯了错误，你可能会很快发现它们。

随着练习的进行，难度会更大，你可能会花更多时间进行调试。以下是一些建议，可以帮助你快速找到错误，并在一开始就避免错误。

+   最重要的是，你应该逐步开发代码；也就是说，你应该编写少量的代码并进行测试。如果它有效，就添加更多的代码；否则，调试你已经有的代码。

+   相反，如果你写了太多的代码，而且很难进行调试，可以将其分成较小的块，并分别进行调试。

例如，假设你想计算三明治列表中每个三明治的波士顿和伦敦价格的中点。作为初稿，你可能会写出类似这样的东西：

```py
boston_price_list = [9.99, 7.99, 7.49, 7, 6.29, 4.99]
london_price_list = [7.5, 5, 4.4, 5, 3.75, 2.25]

midpoint_price = np.mean(boston_price_list + london_price_list)
midpoint_price 
```

```py
5.970833333333334 
```

这段代码运行了，并且得到了一个答案，但这个答案是一个单一的数字，而不是我们期望的列表。

你可能已经发现了错误，但假设你没有。要调试这段代码，我会从将计算分成较小的步骤并显示中间结果开始。例如，我们可以将两个列表相加并显示结果，就像这样。

```py
total_price = boston_price_list + london_price_list
total_price 
```

```py
[9.99, 7.99, 7.49, 7, 6.29, 4.99, 7.5, 5, 4.4, 5, 3.75, 2.25] 
```

看结果，我们发现它没有按我们的意图对三明治的价格进行逐个元素相加。因为参数是列表，`+`运算符连接它们而不是将元素相加。

我们可以通过将列表转换为数组来解决这个问题。

```py
boston_price_array = np.array(boston_price_list)
london_price_array = np.array(london_price_list)

total_price_array = boston_price_array + london_price_array
total_price_array 
```

```py
array([17.49, 12.99, 11.89, 12\.  , 10.04,  7.24]) 
```

然后计算每对价格的中点，就像这样：

```py
midpoint_price_array = total_price_array / 2
midpoint_price_array 
```

```py
array([8.745, 6.495, 5.945, 6\.   , 5.02 , 3.62 ]) 
```

随着经验的积累，你将能够在测试之前编写更多的代码。但在开始阶段，保持简单！一般规则是，每行代码应执行少量操作，每个单元格应包含少量语句。在开始阶段，这个数量应该是一个。

## 总结

本章介绍了表示一系列值的三种方式：元组、列表和 Numpy 数组。在处理数据时，我们主要会使用数组。

它还介绍了表示差异的三种方式：绝对值、相对值和百分比；以及总结一组值的几种方式：最小值、最大值、平均值和标准差。

在下一章中，我们将开始处理数据文件，并使用循环处理字母和单词。
