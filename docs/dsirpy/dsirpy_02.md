# 第二章：算法

> 原文：[`allendowney.github.io/DSIRP/algorithms.html`](https://allendowney.github.io/DSIRP/algorithms.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


[点击这里在 Colab 上运行本章](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/algorithms.ipynb)

## 搜索变位词

在这本笔记本中，我们将实现两个任务的算法：

+   测试一对单词，看它们是否是彼此的变位词，也就是说，你是否可以重新排列一个单词的字母来拼写另一个单词。

+   在单词列表中搜索所有彼此是变位词的对。

这些例子有一个要点，我会在最后解释。

**练习 1：**编写一个函数，它接受两个单词，并在它们是变位词时返回`True`。用下面的示例测试你的函数。

```py
def is_anagram(word1, word2):
    return False 
```

```py
is_anagram('tachymetric', 'mccarthyite') # True 
```

```py
False 
```

```py
is_anagram('post', 'top') # False, letter not present 
```

```py
False 
```

```py
is_anagram('pott', 'top') # False, letter present but not enough copies 
```

```py
False 
```

```py
is_anagram('top', 'post') # False, letters left over at the end 
```

```py
False 
```

```py
is_anagram('topss', 'postt') # False 
```

```py
False 
```

**练习 2：**使用`timeit`来查看你的函数在这些示例中有多快：

```py
%timeit is_anagram('tops', 'spot') 
```

```py
50.8 ns ± 0.779 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each) 
```

```py
%timeit is_anagram('tachymetric', 'mccarthyite') 
```

```py
49.9 ns ± 3.99 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each) 
```

我们如何比较在不同计算机上运行的算法？

## 搜索变位词对

**练习 3：**编写一个函数，它接受一个单词列表，并返回所有变位词对的列表。

```py
short_word_list = ['proudest', 'stop', 'pots', 'tops', 'sprouted'] 
```

```py
def all_anagram_pairs(word_list):
    return [] 
```

```py
all_anagram_pairs(short_word_list) 
```

```py
[] 
```

以下单元格下载一个包含英语单词列表的文件。

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

以下函数读取一个文件并返回一组单词（我使用了一个集合，因为在我们将单词转换为小写后，有一些重复。）

```py
def read_words(filename):
  """Read lines from a file and split them into words."""
    res = set()
    for line in open(filename):
        for word in line.split():
            res.add(word.strip().lower())
    return res 
```

```py
word_list = read_words('american-english')
len(word_list) 
```

```py
100781 
```

**练习 4：**循环遍历单词列表，并打印所有与`stop`是变位词的单词。

现在用完整的`word_list`运行`all_anagram_pairs`：

```py
# pairs = all_anagram_pairs(word_list) 
```

**练习 5：**当它正在运行时，让我们估计它要花多长时间。

## 更好的算法

**练习 6：**编写一个更好的算法！提示：制作一个字典。你的算法快多少？

```py
def all_anagram_lists(word_list):
  """Finds all anagrams in a list of words.

 word_list: sequence of strings
 """
    return {} 
```

```py
%time anagram_map = all_anagram_lists(word_list) 
```

```py
CPU times: user 173 ms, sys: 8.02 ms, total: 181 ms
Wall time: 180 ms 
```

```py
len(anagram_map) 
```

```py
93406 
```

## 总结

这本笔记本中的例子的要点是什么？

+   `is_anagram`的不同版本表明，当输入很小时，很难说哪个算法会最快。这往往取决于实现的细节。无论如何，差异往往很小，所以在实践中可能并不重要。

+   我们用来搜索变位词对的不同算法表明，当输入很大时，我们通常可以知道哪个算法会最快。快速算法和慢算法之间的差异可能是巨大的！

## 练习

在你做这些练习之前，你可能想阅读 Python 的[排序指南](https://docs.python.org/3/howto/sorting.html)。它使用`lambda`来定义一个匿名函数，你可以在[这里](https://www.w3schools.com/python/python_lambda.asp)了解它。

**练习 7：**制作一个像`anagram_map`一样的字典，其中只包含映射到多于一个元素的列表的键。你可以使用一个`for`循环来制作一个新字典，或者使用[字典推导](https://www.freecodecamp.org/news/dictionary-comprehension-in-python-explained-with-examples/)。

**练习 8：**找到至少有一个变位词的最长单词。建议：使用`sort`或`sorted`的`key`参数（[见这里](https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda)）。

**练习 9：**找到最大的单词列表，这些单词是彼此的变位词。

**练习 10：**编写一个函数，它接受一个整数`word_length`，并找到具有给定长度的最长单词列表，这些单词是彼此的变位词。

**练习 11：**到目前为止，我们有一个包含变位词列表的数据结构，但我们实际上还没有枚举所有的对。编写一个函数，它接受`anagram_map`并返回所有变位词对的列表。有多少个？

*Python 中的数据结构和信息检索*

版权所有 2021 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
