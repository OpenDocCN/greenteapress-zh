# 第五章：测验 1

> 原文：[`allendowney.github.io/DSIRP/quiz01.html`](https://allendowney.github.io/DSIRP/quiz01.html)

在开始此测验之前：

1.  单击“复制到驱动器”以复制测验，

1.  单击“分享”,

1.  单击“更改”，然后选择“任何人都可以编辑此链接”

1.  点击“复制链接”

1.  将链接粘贴到[此 Canvas 作业](https://canvas.olin.edu/courses/313/assignments/4866)中。

版权所有 2021 年 Allen Downey，[MIT 许可证](http://opensource.org/licenses/MIT)

## 设置

以下单元格下载一个包含单词列表的文件，读取单词，并将它们存储在一个`set`中。

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

## 问题 1

以下函数接受一个字符串，并在字符串中的字母按字母顺序出现时返回`True`。

```py
def is_alphabetical(word):
    return list(word) == sorted(word) 
```

```py
is_alphabetical('almost') # True 
```

```py
True 
```

```py
is_alphabetical('alphabetical') # False 
```

```py
False 
```

创建一个名为`alpha_words`的新列表，其中仅包含`word_list`中按字母顺序的单词，并显示列表的长度。

## 问题 2

找到并显示`alpha_words`中最长的单词。如果有多个具有最大长度的单词，则可以显示其中任何一个（但只能显示一个）。

注意：即使您对上一个问题的答案不起作用，您也可以为此问题编写代码。我会评估代码，而不是输出。

## 问题 3

编写一个名为`encompasses`的函数，该函数接受两个单词并返回`True`，如果第一个单词包含第二个单词，但不在开头或结尾（否则返回`False`）。例如，`hippopotomus`包含单词`pot`。

提示：您可能会发现字符串方法`find`有用。

```py
'hippopotomus'.find('pot') 
```

```py
5 
```

```py
'hippopotomus'.find('potato') 
```

```py
-1 
```

```py
# WRITE YOUR FUNCTION HERE 
```

您可以使用以下示例来测试您的函数。

```py
word1 = 'hippopotamus'
word2 = 'pot'
word3 = 'hippo'
word4 = 'mus'
word5 = 'potato' 
```

```py
encompasses(word1, word2) # True 
```

```py
True 
```

```py
encompasses(word1, word3) # False because word3 is at the beginning 
```

```py
False 
```

```py
encompasses(word1, word4) # False because word4 is at the end 
```

```py
False 
```

```py
encompasses(word1, word5) # False because word5 is not in word1 
```

```py
False 
```

## 问题 4

如果其中一个单词是另一个单词的倒转，则两个单词构成“倒转对”。例如，`pots`和`stop`是一个倒转对。

倒转对中的单词必须不同，因此`gag`和`gag`不构成倒转对。

制作`word_list`中所有倒转对的列表。每对单词应该只出现一次，因此如果列表中有`('tons', 'snot')`，则不应该有`('snot', 'tons')`。

仅供娱乐的奖励问题：这个单词列表中最长的倒转对是什么？
