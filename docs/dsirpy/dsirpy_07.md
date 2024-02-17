# 第七章：集合

> 原文：[`allendowney.github.io/DSIRP/set.html`](https://allendowney.github.io/DSIRP/set.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


[点击这里在 Colab 上运行这一章节](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/set.ipynb)

## 集合运算符和方法

以下示例基于 Luciano Ramalho 的演讲，[集合实践：从 Python 的集合类型中学习](https://www.youtube.com/watch?v=tGAngdU_8D8)。

```py
def fibonacci(stop):
    a, b = 0, 1
    while a < stop:
        yield a
        a, b = b, a + b 
```

```py
f = {n for n in fibonacci(10)}
f 
```

```py
{0, 1, 2, 3, 5, 8} 
```

```py
def primes(stop):
    m = {}
    q = 2
    while q < stop:
        if q not in m:
            yield q 
            m[q*q] = [q]
        else:
            for p in m[q]:
                m.setdefault(p+q, []).append(p)
            del m[q]
        q += 1 
```

```py
p = {n for n in primes(10)}
p 
```

```py
{2, 3, 5, 7} 
```

检查成员资格是常数时间。

```py
8 in f 
```

```py
True 
```

```py
8 in p 
```

```py
False 
```

交集就像 AND：它返回 f 和 p 中的元素。

```py
f & p 
```

```py
{2, 3, 5} 
```

并集就像 OR：它返回 f 或 p 中的元素。

```py
f | p 
```

```py
{0, 1, 2, 3, 5, 7, 8} 
```

对称差异就像 XOR：来自`f`或`p`但不是两者的元素。

```py
f ^ p 
```

```py
{0, 1, 7, 8} 
```

以下是不是素数的斐波那契数。

```py
f - p 
```

```py
{0, 1, 8} 
```

和不是斐波那契数的素数。

```py
p - f 
```

```py
{7} 
```

比较运算符检查子集和超集关系。

斐波那契数不是素数的超集。

```py
f >= p 
```

```py
False 
```

而且素数不是斐波那契数的超集。

```py
p >= f 
```

```py
False 
```

在这个意义上，集合不像数字：它们只是[部分有序](https://en.wikipedia.org/wiki/Partially_ordered_set)。

`f`是`{1, 2, 3}`的超集

```py
f >= {1, 2, 3} 
```

```py
True 
```

```py
p >= {1, 2, 3} 
```

```py
False 
```

集合提供方法以及运算符。为什么？

首先，传递给方法的参数可以是任何可迭代对象，而不仅仅是一个集合。

```py
try:
    f >= [1, 2, 3]
except TypeError as e:
    print(e) 
```

```py
'>=' not supported between instances of 'set' and 'list' 
```

```py
f.issuperset([1,2,3]) 
```

```py
True 
```

方法也接受多个参数：

```py
f.union([1,2,3], (3,4,5), {5,6,7}, {7:'a', 8:'b'}) 
```

```py
{0, 1, 2, 3, 4, 5, 6, 7, 8} 
```

如果你没有一个集合可以开始，你可以使用一个空集。

```py
set().union([1,2,3], (3,4,5), {5,6,7}, {7:'a', 8:'b'}) 
```

```py
{1, 2, 3, 4, 5, 6, 7, 8} 
```

一个小的语法麻烦：`{1, 2, 3}`是一个集合，但`{}`是一个空字典。

## 拼字比赛

[纽约时报拼字比赛](https://www.nytimes.com/puzzles/spelling-bee)是一个每日谜题，目标是仅使用给定的七个字母拼写尽可能多的单词。例如，在最近的拼字比赛中，可用的字母是`dehiklo`，所以你可以拼写“like”和“hold”。

你可以多次使用每个字母，所以“hook”和“deed”也是允许的。

为了使它更有趣，其中一个字母是特殊的，必须包含在每个单词中。在这个例子中，特殊字母是`o`，所以“hood”是允许的，但“like”不是。

你找到的每个单词都根据它的长度得分，长度至少为四个字母。使用所有字母的单词称为“全字母句”，并且得到额外的分数。

我们将使用这个谜题来探索 Python 集合的使用。

假设我们有一个单词，我们想知道它是否可以只用给定的字母集拼写。以下函数使用字符串操作解决了这个问题。

```py
def uses_only(word, letters):
    for letter in word:
        if letter not in letters:
            return False
    return True 
```

如果我们在`word`中找到任何不在字母列表中的字母，我们可以立即返回`False`。如果我们在单词中没有找到任何不可用的字母，我们可以返回`True`。

让我们用一些例子来试试。在最近的拼字比赛中，可用的字母是`dehiklo`。让我们看看我们能用它们拼出什么。

```py
letters = "dehiklo"
uses_only('lode', letters) 
```

```py
True 
```

```py
uses_only('implode', letters) 
```

```py
False 
```

练习：可以使用集合操作而不是列表操作更简洁地实现`uses_only`。[阅读`set`类的文档](https://docs.python.org/3/tutorial/datastructures.html#sets)并使用集合重写`uses_only`。

```py
uses_only('lode', letters) 
```

```py
True 
```

```py
uses_only('implode', letters) 
```

```py
False 
```

## 单词列表

以下函数下载了大约 10 万个美国英语单词的列表。

```py
from os.path import basename, exists

def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)

download('https://github.com/AllenDowney/DSIRP/raw/main/american-english') 
```

文件每行包含一个单词，所以我们可以读取文件并将其拆分成一个单词列表，就像这样：

```py
word_list = open('american-english').read().split()
len(word_list) 
```

```py
102401 
```

练习：编写一个循环，遍历这个单词列表，并只打印单词

+   至少有四个字母，

+   只能使用字母`dehiklo`拼写，和

+   包括字母`o`。

练习：现在让我们检查全字母句。编写一个名为`uses_all`的函数，它接受两个字符串，并在第一个字符串使用第二个字符串的所有字母时返回`True`。思考如何使用集合操作来表达这个计算。

用至少一个返回`True`和一个返回`False`的案例测试你的函数。

练习：修改前面的循环，使用`uses_only`和`uses_all`在单词列表中搜索全字母句。

或者，作为奖励，编写一个名为`uses_all_and_only`的函数，使用单个`set`操作检查这两个条件。

## 剩余部分

到目前为止，我们一直在编写测试特定条件的布尔函数，但如果它们返回`False`，它们不会解释原因。作为`uses_only`的替代方案，我们可以编写一个名为`bad_letters`的函数，该函数接受一个单词和一组字母，并返回一个新字符串，其中包含单词中不可用的字母。让我们称它为`bad_letters`。

```py
def bad_letters(word, letters):
    return set(word) - set(letters) 
```

现在，如果我们用一个非法单词运行这个函数，它会告诉我们单词中哪些字母是不可用的。

```py
bad_letters('oilfield', letters) 
```

```py
{'f'} 
```

**练习：**编写一个名为`unused_letters`的函数，该函数接受一个单词和一组字母，并返回在`word`中未使用的字母的子集。

**练习：**编写一个名为`no_duplicates`的函数，该函数接受一个字符串并返回`True`，如果每个字母只出现一次。

Python 中的数据结构和信息检索

版权所有 2021 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
