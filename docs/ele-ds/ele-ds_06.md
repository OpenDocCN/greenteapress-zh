# 循环和文件

> 原文：[`allendowney.github.io/ElementsOfDataScience/04_loops.html`](https://allendowney.github.io/ElementsOfDataScience/04_loops.html)

[点击这里在 Colab 上运行这个笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/04_loops.ipynb) 或 [点击这里下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/04_loops.ipynb)。

本章介绍了循环，用于表示重复计算，以及文件，用于存储数据。例如，我们将从 Project Gutenberg 下载著名的书籍*战争与和平*，并编写一个循环来读取这本书并计算单词数。这个例子介绍了一些新的计算工具；也是处理文本数据的入门。

## 循环

计算中最重要的元素之一是重复，表示重复的最常见方式是`for`循环。举个简单的例子，假设我们想显示一个元组的元素。这是一个包含三个整数的元组：

```py
t = 1, 2, 3 
```

这是一个打印元素的`for`循环。

```py
for x in t:
    print(x) 
```

```py
1
2
3 
```

循环的第一行是一个**标题**，指定了元组`t`和一个变量名`x`。元组已经存在，但`x`不存在；循环将创建它。请注意，标题以冒号`:`结束。

循环内有一个`print`语句，显示`x`的值。

所以发生了什么：

1.  当循环开始时，它获取`t`的第一个元素，即`1`，并将其赋给`x`。它执行`print`语句，显示值`1`。

1.  然后它获取`t`的第二个元素，即`2`，并显示它。

1.  然后它获取`t`的第三个元素，即`3`，并显示它。

打印元组的最后一个元素后，循环结束。

我们也可以循环遍历字符串中的字母：

```py
word = 'Data'

for letter in word:
    print(letter) 
```

```py
D
a
t
a 
```

当循环开始时，`word`已经存在，但`letter`不存在。同样，循环创建`letter`并为其赋值。

循环创建的变量称为**循环变量**。你可以给它任何你喜欢的名字；在这个例子中，我选择了`letter`，以便提醒我它包含的是什么类型的值。

循环结束后，循环变量包含最后一个值。

```py
letter 
```

```py
'a' 
```

**练习：** 创建一个名为`sequence`的列表，其中包含四个任意类型的元素。编写一个`for`循环，打印这些元素。将循环变量命名为`element`。

你可能会想为什么我没有把列表称为`list`。我避免这样做是因为 Python 有一个名为`list`的函数，用于创建新的列表。例如，如果你有一个字符串，你可以创建一个字母列表，就像这样：

```py
list('string') 
```

```py
['s', 't', 'r', 'i', 'n', 'g'] 
```

如果你创建一个名为`list`的变量，你就不能再使用这个函数了。

## 循环计数

*战争与和平*是一本非常长的书；让我们看看它有多长。要计算单词数，我们需要两个元素：循环遍历文本中的单词和计数。我们将从计数开始。

我们已经看到你可以创建一个变量并给它一个值，就像这样：

```py
count = 0
count 
```

```py
0 
```

如果你给同一个变量赋予不同的值，新值会替换旧值。

```py
count = 1
count 
```

```py
1 
```

你可以通过读取旧值，加`1`，并将结果赋回原始变量来增加变量的值。

```py
count = count + 1
count 
```

```py
2 
```

增加变量的值称为**递增**；减少值称为**递减**。这些操作是如此常见，以至于有专门的运算符。

```py
count += 1
count 
```

```py
3 
```

在这个例子中，`+=`运算符读取`count`的值，加上`1`，并将结果赋回给`count`。Python 还提供`-=`和其他更新运算符，如`*=`和`/=`。

**练习：** 以下是来自*Learn With Math Games*的数字技巧[`www.learn-with-math-games.com/math-number-tricks.html`](https://www.learn-with-math-games.com/math-number-tricks.html)：

> *找到某人的年龄*
> 
> +   让这个人把他们的年龄的第一个数字乘以 5。
> +   
> +   告诉他们加 3。
> +   
> +   现在告诉他们把这个数字翻一番。
> +   
> +   最后，让这个人把他们的年龄的第二个数字加到这个数字上，并让他们告诉你答案。
> +   
> +   减去 6，你就得到他们的年龄。

使用你的年龄测试这个算法。使用一个变量，并使用`+=`和其他更新运算符来更新它。

## 文件

现在我们知道如何计数了，让我们看看如何从文件中读取单词。我们可以从 Project Gutenberg 下载《战争与和平》，这是一个免费图书的存储库，网址是[`www.gutenberg.org`](https://www.gutenberg.org)。

为了读取文件的内容，你必须**打开**它，你可以使用`open`函数来做到这一点。

```py
fp = open('2600-0.txt')
fp 
```

```py
<_io.TextIOWrapper name='2600-0.txt' mode='r' encoding='UTF-8'> 
```

结果是`TextIOWrapper`，这是一种**文件指针**类型。它包含文件名，模式（`r`表示“读取”）和编码（`UTF`表示“Unicode 转换格式”）。文件指针就像书签一样；它跟踪你已经读取了文件的哪些部分。

如果你在`for`循环中使用文件指针，它会循环遍历文件中的行。所以我们可以这样计算行数：

```py
fp = open('2600-0.txt')
count = 0
for line in fp:
    count += 1 
```

然后显示结果。

```py
count 
```

```py
66054 
```

这个文件中大约有 66,000 行。

## if 语句

我们已经看到比较运算符，比如`>`和`<`，它们比较值并产生一个布尔结果，`True`或`False`。例如，我们可以将`count`的最终值与一个数字进行比较：

```py
count > 60000 
```

```py
True 
```

我们可以在`if`语句中使用比较运算符来检查条件并相应地采取行动。

```py
if count > 60000:
    print('Long book!') 
```

```py
Long book! 
```

`if`语句的第一行指定了我们要检查的条件。就像`for`语句的头部一样，`if`语句的第一行必须以冒号结尾。

如果条件为真，则缩进的语句运行；否则，不运行。在前面的例子中，条件为真，所以`print`语句运行。在下面的例子中，条件为假，所以`print`语句不运行。

```py
if count < 1000:
    print('Short book!') 
```

我们可以在`for`循环内放置一个`print`语句。在这个例子中，当`count`为`1`时，我们只打印书中的一行。其他行被读取，但没有显示。

```py
fp = open('2600-0.txt')
count = 0
for line in fp:
    if count == 1:
        print(line)
    count += 1 
```

```py
The Project Gutenberg EBook of War and Peace, by Leo Tolstoy 
```

注意这个例子中的缩进：

+   `for`循环内的语句是缩进的。

+   `if`语句内的语句是缩进的。

+   语句`count += 1`从上一行**取消缩进**，所以它结束了`if`语句。但它仍然在`for`循环内。

在 Python 中使用空格或制表符进行缩进是合法的，但最常见的约定是使用四个空格，永远不使用制表符。这就是我在我的代码中要做的，我强烈建议你遵循这个约定。

## `break`语句

如果我们显示`count`的最终值，我们会看到循环读取了整个文件，但只打印了一行：

```py
count 
```

```py
66054 
```

我们可以使用`break`语句避免读取整个文件，就像这样：

```py
fp = open('2600-0.txt')
count = 0
for line in fp:
    if count == 1:
        print(line)
        break
    count += 1 
```

```py
The Project Gutenberg EBook of War and Peace, by Leo Tolstoy 
```

`break`语句立即结束循环，跳过文件的其余部分。我们可以通过检查`count`的最后一个值来确认：

```py
count 
```

```py
1 
```

**练习：**编写一个循环，打印文件的前 5 行，然后跳出循环。

## 空白

如果我们再次运行循环并显示`line`的最终值，我们会看到特殊序列`\n`在末尾。

```py
fp = open('2600-0.txt')
count = 0
for line in fp:
    if count == 1:
        break
    count += 1

line 
```

```py
'The Project Gutenberg EBook of War and Peace, by Leo Tolstoy\n' 
```

这个序列代表一个称为**换行符**的单个字符，它在行之间放置垂直空间。如果我们使用`print`语句来显示`line`，我们看不到特殊序列，但是我们会看到行后面有额外的空间。

```py
print(line) 
```

```py
The Project Gutenberg EBook of War and Peace, by Leo Tolstoy 
```

在其他字符串中，你可能会看到序列`\t`，它代表“制表符”字符。当你打印一个制表符字符时，它会添加足够的空间，使下一个字符出现在 8 的倍数列中。

```py
print('01234567' * 6)
print('a\tbc\tdef\tghij\tklmno\tpqrstu') 
```

```py
012345670123456701234567012345670123456701234567
a	bc	def	ghij	klmno	pqrstu 
```

换行字符、制表符和空格被称为**空白字符**，因为当它们被打印时，它们在页面上留下空白（假设背景颜色是白色）。

## 计算单词

到目前为止，我们已经成功计算了文件中的行数，但每行包含多个单词。为了将一行分割成单词，我们可以使用一个名为`split`的函数，它返回一个单词列表。更准确地说，`split`实际上并不知道什么是一个单词；它只是在有空格或其他空白字符的地方分割行。

```py
line.split() 
```

```py
['The',
 'Project',
 'Gutenberg',
 'EBook',
 'of',
 'War',
 'and',
 'Peace,',
 'by',
 'Leo',
 'Tolstoy'] 
```

请注意，`split`的语法与我们见过的其他函数不同。通常当我们调用一个函数时，我们会命名函数并在括号中提供值。所以你可能期望写成`split(line)`。遗憾的是，这样不起作用。

问题在于`split`函数属于字符串`line`；在某种意义上，该函数附加到字符串上，因此我们只能使用字符串和**点运算符**（`line`和`split`之间的句号）来引用它。出于历史原因，这样的函数被称为**方法**。

现在我们可以将一行拆分成一个单词列表，我们可以使用`len`来获取每个列表中的单词数，并相应地增加`count`。

```py
fp = open('2600-0.txt')
count = 0
for line in fp:
    count += len(line.split()) 
```

```py
count 
```

```py
566317 
```

按照这个计算，*战争与和平*中有超过 50 万个单词。

实际上，并不是有那么多单词，因为我们从古腾堡计划得到的文件在文本之前有一些介绍性文字和目录。并且在结尾有一些许可信息。为了跳过这些“前言”，我们可以使用一个循环读取行，直到我们到达`CHAPTER I`，然后使用第二个循环计算剩余行中的单词数。

文件指针`fp`跟踪文件中的位置，因此第二个循环从第一个循环结束的地方开始。在第二个循环中，我们检查书的结尾并停止，因此我们忽略文件末尾的“后事”。

```py
first_line = "CHAPTER I\n"
last_line = "End of the Project Gutenberg EBook of War and Peace, by Leo Tolstoy\n"

fp = open('2600-0.txt')
for line in fp:
    if line == first_line:
        break

count = 0
for line in fp:
    if line == last_line:
        print(line)
        break
    count += len(line.split()) 
```

```py
End of the Project Gutenberg EBook of War and Peace, by Leo Tolstoy 
```

```py
count 
```

```py
562482 
```

关于这个程序有两件事需要注意：

+   当我们比较两个值是否相等时，我们使用`==`运算符，不要与赋值运算符`=`混淆。

+   我们将`line`与之进行比较的字符串末尾有一个换行符。如果我们去掉它，程序就无法正常工作。

**练习：**

1.  在前一个程序中，用`=`替换`==`，看看会发生什么。这是一个常见的错误，所以看看错误消息是什么样子是很好的。

1.  纠正前面的错误，然后删除`CHAPTER I`后面的换行符，看看会发生什么。

第一个错误是**语法错误**，这意味着程序违反了 Python 的规则。如果程序有语法错误，Python 解释器会打印错误消息，程序将无法运行。

第二个错误是**逻辑错误**，这意味着程序的逻辑有问题。语法是合法的，程序可以运行，但它并不符合我们的预期。逻辑错误很难找到，因为我们不会收到任何错误消息。

如果你有逻辑错误，以下是两种调试策略：

1.  在程序运行时添加打印语句，以便显示额外的信息。

1.  简化程序，直到它符合预期，然后逐渐添加更多代码，一边测试一边进行。

## 总结

本章介绍了循环、`if`语句和`break`语句。它还介绍了处理字母和单词的工具，以及一种简单的文本分析方法，即单词计数。

在下一章中，我们将继续这个例子，统计文本中独特单词的数量以及每个单词出现的次数。我们还将看到另一种表示值集合的方法，即 Python 字典。
