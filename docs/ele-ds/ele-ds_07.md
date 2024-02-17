# 字典

> [`allendowney.github.io/ElementsOfDataScience/05_dictionaries.html`](https://allendowney.github.io/ElementsOfDataScience/05_dictionaries.html)

[单击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/05_dictionaries.ipynb) 或 [单击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/05_dictionaries.ipynb)。

在上一章中，我们使用`for`循环来读取文件并计算单词的数量。在本章中，您将学习一种称为**字典**的新类型，并将用它来计算唯一单词的数量以及每个单词出现的次数。

您还将学习如何从序列（元组、列表或数组）中选择一个元素。您还将学习一些关于 Unicode 的知识，它用于表示世界上几乎每种语言的字母、数字和标点符号。

## 索引

假设您有一个名为`t`的变量，它引用一个列表或元组。您可以使用**方括号运算符**`[]`选择一个元素。例如，这是一个字符串元组：

```py
t = 'zero', 'one', 'two' 
```

要选择第一个元素，我们将`0`放在括号中：

```py
t[0] 
```

```py
'zero' 
```

要选择第二个元素，我们将`1`放在括号中：

```py
t[1] 
```

```py
'one' 
```

要选择第三个元素，我们将`2`放在括号中：

```py
t[2] 
```

```py
'two' 
```

括号中的数字称为**索引**，因为它表示我们想要的元素。元组和列表使用从零开始的编号；也就是说，第一个元素的索引是 0。其他一些编程语言使用从一开始的编号。这两种系统都有利弊（参见[`en.wikipedia.org/wiki/Zero-based_numbering`](https://en.wikipedia.org/wiki/Zero-based_numbering)）。

括号中的索引也可以是一个变量：

```py
i = 1
t[i] 
```

```py
'one' 
```

或者是一个包含变量、值和操作符的表达式：

```py
t[i+1] 
```

```py
'two' 
```

但是，如果索引超出了列表或元组的末尾，就会出错。

此外，索引必须是整数；如果是其他类型，就会出错。

**练习：**您可以使用负整数作为索引。尝试使用`-1`和`-2`作为索引，看看能否弄清它们的作用。

## 字典

字典类似于元组或列表，但在字典中，索引可以是几乎任何类型，而不仅仅是整数。我们可以这样创建一个空字典：

```py
d = {} 
```

然后我们可以这样添加元素：

```py
d['one'] = 1
d['two'] = 2 
```

在这个例子中，索引是字符串`'one'`和`'two'`。如果显示字典，它会显示每个索引和相应的值。

```py
d 
```

```py
{'one': 1, 'two': 2} 
```

与其创建一个空字典，然后添加元素，不如创建一个字典并同时指定元素：

```py
d = {'one': 1, 'two': 2, 'three': 3}
d 
```

```py
{'one': 1, 'two': 2, 'three': 3} 
```

当我们谈论字典时，索引通常被称为**键**。在这个例子中，键是字符串，相应的值是整数。

字典也被称为**映射**，因为它表示键和值之间的对应关系或“映射”。因此，我们可以说这个字典将英文数字名称映射到相应的整数。

您可以使用方括号运算符从字典中选择一个元素，就像这样：

```py
d['two'] 
```

```py
2 
```

但不要忘记引号。没有引号，Python 会寻找一个名为`two`的变量，但找不到。

要检查特定的键是否在字典中，可以使用特殊单词`in`：

```py
'one' in d 
```

```py
True 
```

```py
'zero' in d 
```

```py
False 
```

因为单词`in`是 Python 中的一个操作符，所以不能将其用作变量名。

如果一个键已经在字典中，再次添加它不会产生任何效果：

```py
d 
```

```py
{'one': 1, 'two': 2, 'three': 3} 
```

```py
d['one'] = 1
d 
```

```py
{'one': 1, 'two': 2, 'three': 3} 
```

但是可以更改与键关联的值：

```py
d['one'] = 100
d 
```

```py
{'one': 100, 'two': 2, 'three': 3} 
```

您可以这样遍历字典中的键：

```py
for key in d:
    print(key) 
```

```py
one
two
three 
```

如果要获取键和值，一种方法是遍历键并查找值：

```py
for key in d:
    print(key, d[key]) 
```

```py
one 100
two 2
three 3 
```

或者您可以同时循环遍历两者，就像这样：

```py
for key, value in d.items():
    print(key, value) 
```

```py
one 100
two 2
three 3 
```

`items`方法遍历字典中的键-值对；每次循环时，它们被分配给`key`和`value`。

**练习：**创建一个字典，其中整数`1`、`2`和`3`作为键，字符串作为值。字符串应该是“one”、“two”和“three”或您所知道的任何其他语言中的等价词。

编写一个循环，仅打印字典中的值。

## 计算唯一单词

在上一章中，我们从 Project Gutenberg 下载了*战争与和平*并计算了行数和单词数。现在我们有了字典，我们还可以计算唯一单词的数量以及每个单词出现的次数。

就像我们在上一章中所做的那样，我们可以读取*战争与和平*的文本并计算单词的数量。

```py
fp = open('2600-0.txt')
count = 0
for line in fp:
    count += len(line.split())

count 
```

```py
566317 
```

为了计算唯一单词的数量，我们将循环遍历每行中的单词，并将它们作为字典中的键添加：

```py
fp = open('2600-0.txt')
unique_words = {}
for line in fp:
    for word in line.split():
        unique_words[word] = 1 
```

这是我们看到的第一个例子，一个循环**嵌套**在另一个循环中。

+   外部循环遍历文件中的行。

+   内部循环遍历每行中的单词。

每次内部循环时，我们将一个单词作为字典中的键添加，值为 1。如果同一个单词出现多次，它会再次添加到字典中，但不会产生影响。因此，字典中只包含文件中每个唯一单词的一个副本。

在循环结束时，我们可以显示前 10 个键：

```py
i = 0
for key in unique_words:
    print(key)
    i += 1
    if i == 10:
        break 
```

```py
The
Project
Gutenberg
EBook
of
War
and
Peace,
by 
```

字典按照单词在文件中出现的顺序包含了所有单词。但是每个单词只出现一次，所以键的数量就是唯一单词的数量：

```py
len(unique_words) 
```

```py
41991 
```

看起来这本书大约有 42,000 个不同的单词，这远少于总单词数，大约为 560,000。但这并不完全正确，因为我们没有考虑大小写和标点符号。

**练习：**在解决这个问题之前，让我们练习嵌套循环，也就是一个循环嵌套在另一个循环中。假设你有一个单词列表，就像这样：

```py
line = ['War', 'and', 'Peace'] 
```

编写一个嵌套循环，遍历列表中的每个单词和每个单词中的每个字母，并将字母打印在单独的行上。

## 处理大小写

当我们计算唯一单词时，我们可能希望将`The`和`the`视为相同的单词。我们可以通过使用`lower`函数将所有单词转换为小写来实现：

```py
word = 'The'
word.lower() 
```

```py
'the' 
```

`lower`创建一个新字符串；它不会修改原始字符串。

```py
word 
```

```py
'The' 
```

但是，你可以将新字符串赋回给现有变量，就像这样：

```py
word = word.lower() 
```

现在，如果我们可以显示`word`的新值，我们会得到小写版本：

```py
word 
```

```py
'the' 
```

**练习：**修改前面的循环，使其在将单词添加到字典之前制作单词的小写版本。如果我们忽略大写和小写之间的差异，有多少个唯一单词？

## 删除标点符号

为了从单词中删除标点符号，我们可以使用`strip`，它会从字符串的开头和结尾删除指定的字符。这里有一个例子：

```py
word = 'abracadabra'
word.strip('ab') 
```

```py
'racadabr' 
```

在这个例子中，`strip`会从单词的开头和结尾删除所有`a`和`b`的实例，但不会删除中间的。但请注意，它会生成一个新单词；它不会修改原始单词：

```py
word 
```

```py
'abracadabra' 
```

为了删除标点符号，我们可以使用`string`库，它提供了一个名为`punctuation`的变量。

```py
import string

string.punctuation 
```

```py
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' 
```

`string.punctuation`包含最常见的标点符号，但正如我们将看到的那样，并非所有标点符号都包含在内。尽管如此，我们可以使用它来处理大多数情况。这里有一个例子：

```py
line = "It's not given to people to judge what's right or wrong."

for word in line.split():
    word = word.strip(string.punctuation)
    print(word) 
```

```py
It's
not
given
to
people
to
judge
what's
right
or
wrong 
```

`strip`会删除`wrong`末尾的句号，但不会删除`It's`、`don't`和`what's`中的撇号。所以这很好，但我们还有一个问题要解决。这是书中的另一行。

```py
line = 'anyone, and so you don’t deserve to have them.”' 
```

当我们尝试删除标点符号时会发生什么。

```py
for word in line.split():
    word = word.strip(string.punctuation)
    print(word) 
```

```py
anyone
and
so
you
don’t
deserve
to
have
them.” 
```

它删除了`anyone`后面的逗号，但没有删除`them`后面的句号和引号。问题在于这种引号不在`string.punctuation`中。

为了解决这个问题，我们将使用以下循环

1.  读取文件并构建一个包含书中所有标点符号的字典，然后

1.  它使用`join`函数将字典的键连接成一个字符串。

你不必理解它的所有工作原理，但你应该阅读它并看看你能理解多少。你可以在[`docs.python.org/3/library/unicodedata.html`](https://docs.python.org/3/library/unicodedata.html)这里阅读`unicodedata`库的文档。

```py
import unicodedata

fp = open('2600-0.txt')
punc_marks = {}
for line in fp:
    for x in line:
        category = unicodedata.category(x)
        if category[0] == 'P':
            punc_marks[x] = 1

all_punctuation = ''.join(punc_marks)
print(all_punctuation) 
```

```py
,.-:[#]*/“’—‘!?”;()%@ 
```

结果是一个包含文档中出现的所有标点字符的字符串，按它们首次出现的顺序排列。

**练习：** 修改前一节中的单词计数循环，将单词转换为小写，并在将其添加到字典之前去除标点。现在有多少个唯一单词？

可选：你可能想跳过前言，直接从第一章的文本开始，跳过结尾的许可证，就像我们在上一章中做的那样。

## 计算单词频率

在前一节中，我们计算了唯一单词的数量，但我们可能还想知道每个单词出现的频率。然后我们可以找到书中最常见和最不常见的单词。为了计算每个单词的频率，我们将创建一个将每个单词映射到其出现次数的字典。

以下是一个循环遍历字符串并计算每个字母出现次数的示例。

```py
word = 'Mississippi'

letter_counts = {}
for x in word:
    if x in letter_counts:
        letter_counts[x] += 1
    else:
        letter_counts[x] = 1

letter_counts 
```

```py
{'M': 1, 'i': 4, 's': 4, 'p': 2} 
```

这里的`if`语句使用了一个我们以前没有见过的特性，即`else`子句。以下是它的工作原理。

1.  首先，它检查字母`x`是否已经是字典`letter_counts`中的一个键。

1.  如果是这样，它会运行第一个语句`letter_counts[x] += 1`，这会增加与该字母关联的值。

1.  否则，它会运行第二个语句`letter_counts[x] = 1`，这会将`x`作为一个新的键，值为`1`，表示我们已经看到了这个新字母一次。

结果是一个将每个字母映射到其出现次数的字典。

要获取最常见的字母，我们可以使用`Counter`，它类似于字典。要使用它，我们必须导入一个名为`collections`的库：

```py
import collections 
```

然后我们使用`collections.Counter`将字典转换为`Counter`：

```py
counter = collections.Counter(letter_counts)
type(counter) 
```

```py
collections.Counter 
```

`Counter`提供了一个名为`most_common`的函数，我们可以使用它来获取最常见的字符：

```py
counter.most_common(3) 
```

```py
[('i', 4), ('s', 4), ('p', 2)] 
```

结果是一个元组列表，其中每个元组包含一个字符和一个整数。

**练习：** 修改前一个练习中的循环，计算《战争与和平》中单词的频率；然后打印出最常见的 20 个单词以及每个单词出现的次数。

**练习：** 你可以像这样在括号中不加值地运行`most_common`：

```py
word_freq_pairs = counter.most_common() 
```

结果是一个元组列表，其中每个唯一单词都有一个元组。将结果分配给一个变量，这样它就不会被显示出来。然后回答以下问题：

1.  排名第 1 的单词出现了多少次（即列表的第一个元素）？

1.  排名第 10 的单词出现了多少次？

1.  排名第 100 的单词出现了多少次？

1.  排名第 1000 的单词出现了多少次？

1.  排名第 10000 的单词出现了多少次？

你在结果中看到了模式吗？我们将在下一章中更多地探索这个模式。

**练习：** 编写一个循环，计算出现 200 次的单词有多少个。它们是什么？出现 100 次、50 次和 20 次的单词有多少个？

**可选：** 如果你知道如何定义一个函数，可以编写一个函数，该函数接受`Counter`和频率作为参数，打印出所有具有该频率的单词，并返回具有该频率的单词数量。

## 总结

本章介绍了字典，它表示键和值的集合。我们使用字典来计算文件中唯一单词的数量以及每个单词出现的次数。

它还介绍了括号运算符，它选择列表或元组中的一个元素，或者在字典中查找一个键并找到相应的值。

我们学习了一些处理字符串的新方法，包括`lower`和`strip`。最后，我们使用`unicodedata`库来识别被视为标点符号的字符。
