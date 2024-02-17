# 第九章：测验 2

> 原文：[`allendowney.github.io/DSIRP/quiz02.html`](https://allendowney.github.io/DSIRP/quiz02.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


在开始此测验之前：

1.  单击“复制到驱动器”以复制测验，

1.  单击“分享”，

1.  单击“更改”，然后选择“任何拥有此链接的人都可以编辑”

1.  单击“复制链接”和

1.  将链接粘贴到[此 Canvas 作业](https://canvas.olin.edu/courses/313/assignments/4929)中。

此测验是开放笔记，开放互联网。您唯一不能做的事情就是寻求帮助。

版权所有 2021 年 Allen Downey，[MIT 许可证](http://opensource.org/licenses/MIT)

## 问题 1

假设您有一个接受许多选项的函数；有些是必需的，有些是可选的。

在运行函数之前，您可能需要检查：

1.  提供了所有必需的选项，和

1.  没有提供非法选项。

例如，假设此字典包含提供的选项及其值：

```py
options = dict(a=1, b=2)
options 
```

假设只有`a`是必需的。

```py
required = ['a'] 
```

可选参数是`b`和`c`：

```py
optional = ['b', 'c'] 
```

如果选项是必需的或可选的，则选项是合法的。所有其他选项都是非法的。

编写一个名为`check_options`的函数，该函数接受选项及其值的字典，一系列必需选项和一系列合法但不是必需的选项。

1.  应检查是否提供了所有必需的选项，如果没有，则打印一个错误消息，列出缺少的选项。

1.  应检查所有提供的选项是否合法，如果不合法，则打印一个错误消息，列出不合法的选项。

为了获得全额学分，您必须在适当的情况下使用集合操作，而不是编写`for`循环。

以下测试不应显示任何内容，因为字典包含所有必需的选项和没有非法选项。

```py
options = dict(a=1, b=2)
check_options(options, required, optional) 
```

以下测试应打印错误消息，因为字典缺少一个必需的选项。

```py
options = dict(b=2, c=3)
check_options(options, required, optional) 
```

以下测试应显示错误消息，因为字典包含一个非法选项。

```py
options = dict(a=1, b=2, d=4)
check_options(options, required, optional) 
```

## 问题 2

集合方法`symmetric_difference`作用于两个集合，并计算出现在任一集合中但不同时出现的元素的集合。

```py
s1 = {1, 2}
s2 = {2, 3}

s1.symmetric_difference(s2) 
```

对于两个以上的集合，对称差操作也有定义。它计算**出现在奇数个集合中的元素的集合**。

`symmetric_difference`方法只能处理两个集合（不像其他一些集合方法），但您可以像这样链接该方法：

```py
s3 = {3, 4}
s1.symmetric_difference(s2).symmetric_difference(s3) 
```

但是，为了练习，假设我们没有集合方法`symmetric_difference`的等效`^`运算符。

编写一个函数，该函数以列表的形式作为参数，计算它们的对称差，并将结果作为`set`返回。

使用以下测试来检查您的函数。

```py
symmetric_difference([s1, s2])    # should be {1, 3} 
```

```py
symmetric_difference([s2, s3])     # should be {2, 4} 
```

```py
symmetric_difference([s1, s2, s3]) # should be {1, 4} 
```

## 问题 3

编写一个名为`evens_and_odds`的生成器函数，该函数接受一个整数列表并产生：

+   列表中的所有偶数元素，然后

+   列表中的所有奇数元素。

例如，如果列表是`[1, 2, 4, 7]`，则生成的值序列应为`2, 4, 1, 7`。

使用此示例来测试您的函数。

```py
t = [1, 2, 4, 7]

for x in evens_and_odds(t):
    print(x) 
```

作为挑战，仅供娱乐，编写此函数的版本，如果参数是只能迭代一次的迭代器，则该函数有效。

## 问题 4

以下字符串包含[一首著名歌曲](https://youtu.be/dQw4w9WgXcQ?t=43)的歌词。

```py
lyrics = """
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you 
""" 
```

以下生成器函数逐个生成`lyrics`中的单词。

```py
def generate_lyrics(lyrics):
    for word in lyrics.split():
        yield word 
```

编写几行代码，使用`generate_lyrics`一次迭代单词，并构建一个从每个单词到其后跟的单词集的字典。

例如，字典中的前两个条目应为

```py
{'Never': {'gonna'},
 'gonna': {'give', 'let', 'make', 'run', 'say', 'tell'},
 ... 
```

因为在`lyrics`中，“Never”一词总是后跟“gonna”，而“gonna”一词后跟六个不同的词。
