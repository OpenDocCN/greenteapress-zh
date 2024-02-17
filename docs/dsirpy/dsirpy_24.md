# 第二十四章：层次遍历

> 原文：[`allendowney.github.io/DSIRP/level_order.html`](https://allendowney.github.io/DSIRP/level_order.html)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)


[点击这里在 Colab 上运行本章](https://colab.research.google.com/github/AllenDowney/DSIRP/blob/main/notebooks/level_order.ipynb)

## 更多的树遍历

在以前的笔记本中，我们编写了树中深度优先搜索的两个版本。现在我们正在朝着深度优先搜索前进，但我们将在途中停下来：层次遍历。

层次遍历的一个应用是在文件系统中搜索目录（也称为文件夹）。由于目录可以包含其他目录，其他目录可以包含其他目录，依此类推，我们可以将文件系统视为一棵树。

在本笔记本中，我们将首先创建一个目录和假数据文件的树。然后我们将以几种方式遍历它。

而且，我们将学习`os`模块，它提供了与操作系统交互的函数，特别是文件系统。

`os`模块提供了`mkdir`，它创建一个目录。如果目录存在，它会引发一个异常，所以我将它包装在一个`try`语句中。

```py
import os

def mkdir(dirname):
    try:
        os.mkdir(dirname)
        print('made', dirname)
    except FileExistsError:
        print(dirname, 'exists') 
```

现在我将创建一个目录，我们将在其中放置假数据。

```py
mkdir('level_data') 
```

```py
level_data exists 
```

在`level_data`中，我想创建一个名为`2021`的子目录。很诱人地写一些像：

```py
year_dir = `level_data/2021` 
```

这条路径适用于 Unix 操作系统（包括 MacOS），但不适用于 Windows，Windows 在路径中使用`\`而不是`/`。

我们可以通过使用`os.path.join`来避免这个问题，它使用操作系统想要的任何字符来连接路径中的名称。

```py
year_dir = os.path.join('level_data', '2021')
mkdir(year_dir) 
```

```py
level_data/2021 exists 
```

为了制作假数据文件，我将使用以下函数，它打开一个文件进行写入，并把单词`data`放入其中。

```py
def make_datafile(dirname, filename):
    filename = os.path.join(dirname, filename)
    open(filename, 'w').write('data\n')
    print('made', filename) 
```

所以让我们从把一个数据文件放在`year_dir`开始，假设这个文件包含了整年的汇总数据。

```py
make_datafile(year_dir, 'year.csv') 
```

```py
made level_data/2021/year.csv 
```

以下函数

1.  创建一个代表一年中一个月的子目录，

1.  创建一个我们想象中包含整月汇总数据的数据文件，并

1.  调用`make_day`（下面）来创建每个月的每一天的子目录（在一个所有月份都有 30 天的世界中）。

```py
def make_month(i, year_dir):
    month = '%.2d' % i
    month_dir = os.path.join(year_dir, month) 
    mkdir(month_dir)
    make_datafile(month_dir, 'month.csv')

    for j in range(1, 31):
        make_day(j, month_dir) 
```

`make_day`为一个给定的日期创建一个子目录，并在其中放置一个数据文件。

```py
def make_day(j, month_dir):
    day = '%.2d' % j
    day_dir = os.path.join(month_dir, day) 
    mkdir(day_dir)
    make_datafile(day_dir, 'day.csv') 
```

以下循环为每个月创建一个目录。

```py
for i in range(1, 13):
    make_month(i, year_dir) 
```

```py
level_data/2021/01 exists
made level_data/2021/01/month.csv
level_data/2021/01/01 exists
made level_data/2021/01/01/day.csv
level_data/2021/01/02 exists
made level_data/2021/01/02/day.csv
level_data/2021/01/03 exists
made level_data/2021/01/03/day.csv
level_data/2021/01/04 exists
made level_data/2021/01/04/day.csv
level_data/2021/01/05 exists
made level_data/2021/01/05/day.csv
level_data/2021/01/06 exists
made level_data/2021/01/06/day.csv
level_data/2021/01/07 exists
made level_data/2021/01/07/day.csv
level_data/2021/01/08 exists
made level_data/2021/01/08/day.csv
level_data/2021/01/09 exists
made level_data/2021/01/09/day.csv
level_data/2021/01/10 exists
made level_data/2021/01/10/day.csv
level_data/2021/01/11 exists
made level_data/2021/01/11/day.csv
level_data/2021/01/12 exists
made level_data/2021/01/12/day.csv
level_data/2021/01/13 exists
made level_data/2021/01/13/day.csv
level_data/2021/01/14 exists
made level_data/2021/01/14/day.csv
level_data/2021/01/15 exists
made level_data/2021/01/15/day.csv
level_data/2021/01/16 exists
made level_data/2021/01/16/day.csv
level_data/2021/01/17 exists
made level_data/2021/01/17/day.csv
level_data/2021/01/18 exists
made level_data/2021/01/18/day.csv
level_data/2021/01/19 exists
made level_data/2021/01/19/day.csv
level_data/2021/01/20 exists
made level_data/2021/01/20/day.csv
level_data/2021/01/21 exists
made level_data/2021/01/21/day.csv
level_data/2021/01/22 exists
made level_data/2021/01/22/day.csv
level_data/2021/01/23 exists
made level_data/2021/01/23/day.csv
level_data/2021/01/24 exists
made level_data/2021/01/24/day.csv
level_data/2021/01/25 exists
made level_data/2021/01/25/day.csv
level_data/2021/01/26 exists
made level_data/2021/01/26/day.csv
level_data/2021/01/27 exists
made level_data/2021/01/27/day.csv
level_data/2021/01/28 exists
made level_data/2021/01/28/day.csv
level_data/2021/01/29 exists
made level_data/2021/01/29/day.csv
level_data/2021/01/30 exists
made level_data/2021/01/30/day.csv
level_data/2021/02 exists
made level_data/2021/02/month.csv
level_data/2021/02/01 exists
made level_data/2021/02/01/day.csv
level_data/2021/02/02 exists
made level_data/2021/02/02/day.csv
level_data/2021/02/03 exists
made level_data/2021/02/03/day.csv
level_data/2021/02/04 exists
made level_data/2021/02/04/day.csv
level_data/2021/02/05 exists
made level_data/2021/02/05/day.csv
level_data/2021/02/06 exists
made level_data/2021/02/06/day.csv
level_data/2021/02/07 exists
made level_data/2021/02/07/day.csv
level_data/2021/02/08 exists
made level_data/2021/02/08/day.csv
level_data/2021/02/09 exists
made level_data/2021/02/09/day.csv
level_data/2021/02/10 exists
made level_data/2021/02/10/day.csv
level_data/2021/02/11 exists
made level_data/2021/02/11/day.csv
level_data/2021/02/12 exists
made level_data/2021/02/12/day.csv
level_data/2021/02/13 exists
made level_data/2021/02/13/day.csv
level_data/2021/02/14 exists
made level_data/2021/02/14/day.csv
level_data/2021/02/15 exists
made level_data/2021/02/15/day.csv
level_data/2021/02/16 exists
made level_data/2021/02/16/day.csv
level_data/2021/02/17 exists
made level_data/2021/02/17/day.csv
level_data/2021/02/18 exists
made level_data/2021/02/18/day.csv
level_data/2021/02/19 exists
made level_data/2021/02/19/day.csv
level_data/2021/02/20 exists
made level_data/2021/02/20/day.csv
level_data/2021/02/21 exists
made level_data/2021/02/21/day.csv
level_data/2021/02/22 exists
made level_data/2021/02/22/day.csv
level_data/2021/02/23 exists
made level_data/2021/02/23/day.csv
level_data/2021/02/24 exists
made level_data/2021/02/24/day.csv
level_data/2021/02/25 exists
made level_data/2021/02/25/day.csv
level_data/2021/02/26 exists
made level_data/2021/02/26/day.csv
level_data/2021/02/27 exists
made level_data/2021/02/27/day.csv
level_data/2021/02/28 exists
made level_data/2021/02/28/day.csv
level_data/2021/02/29 exists
made level_data/2021/02/29/day.csv
level_data/2021/02/30 exists
made level_data/2021/02/30/day.csv
level_data/2021/03 exists
made level_data/2021/03/month.csv
level_data/2021/03/01 exists
made level_data/2021/03/01/day.csv
level_data/2021/03/02 exists
made level_data/2021/03/02/day.csv
level_data/2021/03/03 exists
made level_data/2021/03/03/day.csv
level_data/2021/03/04 exists
made level_data/2021/03/04/day.csv
level_data/2021/03/05 exists
made level_data/2021/03/05/day.csv
level_data/2021/03/06 exists
made level_data/2021/03/06/day.csv
level_data/2021/03/07 exists
made level_data/2021/03/07/day.csv
level_data/2021/03/08 exists
made level_data/2021/03/08/day.csv
level_data/2021/03/09 exists
made level_data/2021/03/09/day.csv
level_data/2021/03/10 exists
made level_data/2021/03/10/day.csv
level_data/2021/03/11 exists
made level_data/2021/03/11/day.csv
level_data/2021/03/12 exists
made level_data/2021/03/12/day.csv
level_data/2021/03/13 exists
made level_data/2021/03/13/day.csv
level_data/2021/03/14 exists
made level_data/2021/03/14/day.csv
level_data/2021/03/15 exists
made level_data/2021/03/15/day.csv
level_data/2021/03/16 exists
made level_data/2021/03/16/day.csv
level_data/2021/03/17 exists
made level_data/2021/03/17/day.csv
level_data/2021/03/18 exists
made level_data/2021/03/18/day.csv
level_data/2021/03/19 exists
made level_data/2021/03/19/day.csv
level_data/2021/03/20 exists
made level_data/2021/03/20/day.csv
level_data/2021/03/21 exists
made level_data/2021/03/21/day.csv
level_data/2021/03/22 exists
made level_data/2021/03/22/day.csv
level_data/2021/03/23 exists
made level_data/2021/03/23/day.csv
level_data/2021/03/24 exists
made level_data/2021/03/24/day.csv
level_data/2021/03/25 exists
made level_data/2021/03/25/day.csv
level_data/2021/03/26 exists
made level_data/2021/03/26/day.csv
level_data/2021/03/27 exists
made level_data/2021/03/27/day.csv
level_data/2021/03/28 exists
made level_data/2021/03/28/day.csv
level_data/2021/03/29 exists
made level_data/2021/03/29/day.csv
level_data/2021/03/30 exists
made level_data/2021/03/30/day.csv
level_data/2021/04 exists
made level_data/2021/04/month.csv
level_data/2021/04/01 exists
made level_data/2021/04/01/day.csv
level_data/2021/04/02 exists
made level_data/2021/04/02/day.csv
level_data/2021/04/03 exists
made level_data/2021/04/03/day.csv
level_data/2021/04/04 exists
made level_data/2021/04/04/day.csv
level_data/2021/04/05 exists
made level_data/2021/04/05/day.csv
level_data/2021/04/06 exists
made level_data/2021/04/06/day.csv
level_data/2021/04/07 exists
made level_data/2021/04/07/day.csv
level_data/2021/04/08 exists
made level_data/2021/04/08/day.csv
level_data/2021/04/09 exists
made level_data/2021/04/09/day.csv
level_data/2021/04/10 exists
made level_data/2021/04/10/day.csv
level_data/2021/04/11 exists
made level_data/2021/04/11/day.csv
level_data/2021/04/12 exists
made level_data/2021/04/12/day.csv
level_data/2021/04/13 exists
made level_data/2021/04/13/day.csv
level_data/2021/04/14 exists
made level_data/2021/04/14/day.csv
level_data/2021/04/15 exists
made level_data/2021/04/15/day.csv
level_data/2021/04/16 exists
made level_data/2021/04/16/day.csv
level_data/2021/04/17 exists
made level_data/2021/04/17/day.csv
level_data/2021/04/18 exists
made level_data/2021/04/18/day.csv
level_data/2021/04/19 exists
made level_data/2021/04/19/day.csv
level_data/2021/04/20 exists
made level_data/2021/04/20/day.csv
level_data/2021/04/21 exists
made level_data/2021/04/21/day.csv
level_data/2021/04/22 exists
made level_data/2021/04/22/day.csv
level_data/2021/04/23 exists
made level_data/2021/04/23/day.csv
level_data/2021/04/24 exists
made level_data/2021/04/24/day.csv
level_data/2021/04/25 exists
made level_data/2021/04/25/day.csv
level_data/2021/04/26 exists
made level_data/2021/04/26/day.csv
level_data/2021/04/27 exists
made level_data/2021/04/27/day.csv
level_data/2021/04/28 exists
made level_data/2021/04/28/day.csv
level_data/2021/04/29 exists
made level_data/2021/04/29/day.csv
level_data/2021/04/30 exists
made level_data/2021/04/30/day.csv
level_data/2021/05 exists
made level_data/2021/05/month.csv
level_data/2021/05/01 exists
made level_data/2021/05/01/day.csv
level_data/2021/05/02 exists
made level_data/2021/05/02/day.csv
level_data/2021/05/03 exists
made level_data/2021/05/03/day.csv
level_data/2021/05/04 exists
made level_data/2021/05/04/day.csv
level_data/2021/05/05 exists
made level_data/2021/05/05/day.csv
level_data/2021/05/06 exists
made level_data/2021/05/06/day.csv
level_data/2021/05/07 exists
made level_data/2021/05/07/day.csv
level_data/2021/05/08 exists
made level_data/2021/05/08/day.csv
level_data/2021/05/09 exists
made level_data/2021/05/09/day.csv
level_data/2021/05/10 exists
made level_data/2021/05/10/day.csv
level_data/2021/05/11 exists
made level_data/2021/05/11/day.csv
level_data/2021/05/12 exists
made level_data/2021/05/12/day.csv
level_data/2021/05/13 exists
made level_data/2021/05/13/day.csv
level_data/2021/05/14 exists
made level_data/2021/05/14/day.csv
level_data/2021/05/15 exists
made level_data/2021/05/15/day.csv
level_data/2021/05/16 exists
made level_data/2021/05/16/day.csv
level_data/2021/05/17 exists
made level_data/2021/05/17/day.csv
level_data/2021/05/18 exists
made level_data/2021/05/18/day.csv
level_data/2021/05/19 exists
made level_data/2021/05/19/day.csv
level_data/2021/05/20 exists
made level_data/2021/05/20/day.csv
level_data/2021/05/21 exists
made level_data/2021/05/21/day.csv
level_data/2021/05/22 exists
made level_data/2021/05/22/day.csv
level_data/2021/05/23 exists
made level_data/2021/05/23/day.csv
level_data/2021/05/24 exists
made level_data/2021/05/24/day.csv
level_data/2021/05/25 exists
made level_data/2021/05/25/day.csv
level_data/2021/05/26 exists
made level_data/2021/05/26/day.csv
level_data/2021/05/27 exists
made level_data/2021/05/27/day.csv
level_data/2021/05/28 exists
made level_data/2021/05/28/day.csv
level_data/2021/05/29 exists
made level_data/2021/05/29/day.csv
level_data/2021/05/30 exists
made level_data/2021/05/30/day.csv
level_data/2021/06 exists
made level_data/2021/06/month.csv
level_data/2021/06/01 exists
made level_data/2021/06/01/day.csv
level_data/2021/06/02 exists
made level_data/2021/06/02/day.csv
level_data/2021/06/03 exists
made level_data/2021/06/03/day.csv
level_data/2021/06/04 exists
made level_data/2021/06/04/day.csv
level_data/2021/06/05 exists
made level_data/2021/06/05/day.csv
level_data/2021/06/06 exists
made level_data/2021/06/06/day.csv
level_data/2021/06/07 exists
made level_data/2021/06/07/day.csv
level_data/2021/06/08 exists
made level_data/2021/06/08/day.csv
level_data/2021/06/09 exists
made level_data/2021/06/09/day.csv
level_data/2021/06/10 exists
made level_data/2021/06/10/day.csv
level_data/2021/06/11 exists
made level_data/2021/06/11/day.csv
level_data/2021/06/12 exists
made level_data/2021/06/12/day.csv
level_data/2021/06/13 exists
made level_data/2021/06/13/day.csv
level_data/2021/06/14 exists
made level_data/2021/06/14/day.csv
level_data/2021/06/15 exists
made level_data/2021/06/15/day.csv
level_data/2021/06/16 exists
made level_data/2021/06/16/day.csv
level_data/2021/06/17 exists
made level_data/2021/06/17/day.csv
level_data/2021/06/18 exists
made level_data/2021/06/18/day.csv
level_data/2021/06/19 exists
made level_data/2021/06/19/day.csv
level_data/2021/06/20 exists
made level_data/2021/06/20/day.csv
level_data/2021/06/21 exists
made level_data/2021/06/21/day.csv
level_data/2021/06/22 exists
made level_data/2021/06/22/day.csv
level_data/2021/06/23 exists
made level_data/2021/06/23/day.csv
level_data/2021/06/24 exists
made level_data/2021/06/24/day.csv
level_data/2021/06/25 exists
made level_data/2021/06/25/day.csv
level_data/2021/06/26 exists
made level_data/2021/06/26/day.csv
level_data/2021/06/27 exists
made level_data/2021/06/27/day.csv
level_data/2021/06/28 exists
made level_data/2021/06/28/day.csv
level_data/2021/06/29 exists
made level_data/2021/06/29/day.csv
level_data/2021/06/30 exists
made level_data/2021/06/30/day.csv
level_data/2021/07 exists
made level_data/2021/07/month.csv
level_data/2021/07/01 exists
made level_data/2021/07/01/day.csv
level_data/2021/07/02 exists
made level_data/2021/07/02/day.csv
level_data/2021/07/03 exists
made level_data/2021/07/03/day.csv
level_data/2021/07/04 exists
made level_data/2021/07/04/day.csv
level_data/2021/07/05 exists
made level_data/2021/07/05/day.csv
level_data/2021/07/06 exists
made level_data/2021/07/06/day.csv
level_data/2021/07/07 exists
made level_data/2021/07/07/day.csv
level_data/2021/07/08 exists
made level_data/2021/07/08/day.csv
level_data/2021/07/09 exists
made level_data/2021/07/09/day.csv
level_data/2021/07/10 exists
made level_data/2021/07/10/day.csv
level_data/2021/07/11 exists
made level_data/2021/07/11/day.csv
level_data/2021/07/12 exists
made level_data/2021/07/12/day.csv
level_data/2021/07/13 exists
made level_data/2021/07/13/day.csv
level_data/2021/07/14 exists
made level_data/2021/07/14/day.csv
level_data/2021/07/15 exists
made level_data/2021/07/15/day.csv
level_data/2021/07/16 exists
made level_data/2021/07/16/day.csv
level_data/2021/07/17 exists
made level_data/2021/07/17/day.csv
level_data/2021/07/18 exists
made level_data/2021/07/18/day.csv
level_data/2021/07/19 exists
made level_data/2021/07/19/day.csv
level_data/2021/07/20 exists
made level_data/2021/07/20/day.csv
level_data/2021/07/21 exists
made level_data/2021/07/21/day.csv
level_data/2021/07/22 exists
made level_data/2021/07/22/day.csv
level_data/2021/07/23 exists
made level_data/2021/07/23/day.csv
level_data/2021/07/24 exists
made level_data/2021/07/24/day.csv
level_data/2021/07/25 exists
made level_data/2021/07/25/day.csv
level_data/2021/07/26 exists
made level_data/2021/07/26/day.csv
level_data/2021/07/27 exists
made level_data/2021/07/27/day.csv
level_data/2021/07/28 exists
made level_data/2021/07/28/day.csv
level_data/2021/07/29 exists
made level_data/2021/07/29/day.csv
level_data/2021/07/30 exists
made level_data/2021/07/30/day.csv
level_data/2021/08 exists
made level_data/2021/08/month.csv
level_data/2021/08/01 exists
made level_data/2021/08/01/day.csv
level_data/2021/08/02 exists
made level_data/2021/08/02/day.csv
level_data/2021/08/03 exists
made level_data/2021/08/03/day.csv
level_data/2021/08/04 exists
made level_data/2021/08/04/day.csv
level_data/2021/08/05 exists
made level_data/2021/08/05/day.csv
level_data/2021/08/06 exists
made level_data/2021/08/06/day.csv
level_data/2021/08/07 exists
made level_data/2021/08/07/day.csv
level_data/2021/08/08 exists
made level_data/2021/08/08/day.csv
level_data/2021/08/09 exists
made level_data/2021/08/09/day.csv
level_data/2021/08/10 exists
made level_data/2021/08/10/day.csv
level_data/2021/08/11 exists
made level_data/2021/08/11/day.csv
level_data/2021/08/12 exists
made level_data/2021/08/12/day.csv
level_data/2021/08/13 exists
made level_data/2021/08/13/day.csv
level_data/2021/08/14 exists
made level_data/2021/08/14/day.csv
level_data/2021/08/15 exists
made level_data/2021/08/15/day.csv
level_data/2021/08/16 exists
made level_data/2021/08/16/day.csv
level_data/2021/08/17 exists
made level_data/2021/08/17/day.csv
level_data/2021/08/18 exists
made level_data/2021/08/18/day.csv
level_data/2021/08/19 exists
made level_data/2021/08/19/day.csv
level_data/2021/08/20 exists
made level_data/2021/08/20/day.csv
level_data/2021/08/21 exists
made level_data/2021/08/21/day.csv
level_data/2021/08/22 exists
made level_data/2021/08/22/day.csv
level_data/2021/08/23 exists
made level_data/2021/08/23/day.csv
level_data/2021/08/24 exists
made level_data/2021/08/24/day.csv
level_data/2021/08/25 exists
made level_data/2021/08/25/day.csv
level_data/2021/08/26 exists
made level_data/2021/08/26/day.csv
level_data/2021/08/27 exists
made level_data/2021/08/27/day.csv
level_data/2021/08/28 exists
made level_data/2021/08/28/day.csv
level_data/2021/08/29 exists
made level_data/2021/08/29/day.csv
level_data/2021/08/30 exists
made level_data/2021/08/30/day.csv
level_data/2021/09 exists
made level_data/2021/09/month.csv
level_data/2021/09/01 exists
made level_data/2021/09/01/day.csv
level_data/2021/09/02 exists
made level_data/2021/09/02/day.csv
level_data/2021/09/03 exists
made level_data/2021/09/03/day.csv
level_data/2021/09/04 exists
made level_data/2021/09/04/day.csv
level_data/2021/09/05 exists
made level_data/2021/09/05/day.csv
level_data/2021/09/06 exists
made level_data/2021/09/06/day.csv
level_data/2021/09/07 exists
made level_data/2021/09/07/day.csv
level_data/2021/09/08 exists
made level_data/2021/09/08/day.csv
level_data/2021/09/09 exists
made level_data/2021/09/09/day.csv
level_data/2021/09/10 exists
made level_data/2021/09/10/day.csv
level_data/2021/09/11 exists
made level_data/2021/09/11/day.csv
level_data/2021/09/12 exists
made level_data/2021/09/12/day.csv
level_data/2021/09/13 exists
made level_data/2021/09/13/day.csv
level_data/2021/09/14 exists
made level_data/2021/09/14/day.csv
level_data/2021/09/15 exists
made level_data/2021/09/15/day.csv
level_data/2021/09/16 exists
made level_data/2021/09/16/day.csv
level_data/2021/09/17 exists
made level_data/2021/09/17/day.csv
level_data/2021/09/18 exists
made level_data/2021/09/18/day.csv
level_data/2021/09/19 exists
made level_data/2021/09/19/day.csv
level_data/2021/09/20 exists
made level_data/2021/09/20/day.csv
level_data/2021/09/21 exists
made level_data/2021/09/21/day.csv
level_data/2021/09/22 exists
made level_data/2021/09/22/day.csv
level_data/2021/09/23 exists
made level_data/2021/09/23/day.csv
level_data/2021/09/24 exists
made level_data/2021/09/24/day.csv
level_data/2021/09/25 exists
made level_data/2021/09/25/day.csv
level_data/2021/09/26 exists
made level_data/2021/09/26/day.csv
level_data/2021/09/27 exists
made level_data/2021/09/27/day.csv
level_data/2021/09/28 exists
made level_data/2021/09/28/day.csv
level_data/2021/09/29 exists
made level_data/2021/09/29/day.csv
level_data/2021/09/30 exists
made level_data/2021/09/30/day.csv
level_data/2021/10 exists
made level_data/2021/10/month.csv
level_data/2021/10/01 exists
made level_data/2021/10/01/day.csv
level_data/2021/10/02 exists
made level_data/2021/10/02/day.csv
level_data/2021/10/03 exists
made level_data/2021/10/03/day.csv
level_data/2021/10/04 exists
made level_data/2021/10/04/day.csv
level_data/2021/10/05 exists
made level_data/2021/10/05/day.csv
level_data/2021/10/06 exists
made level_data/2021/10/06/day.csv
level_data/2021/10/07 exists
made level_data/2021/10/07/day.csv
level_data/2021/10/08 exists
made level_data/2021/10/08/day.csv
level_data/2021/10/09 exists
made level_data/2021/10/09/day.csv
level_data/2021/10/10 exists
made level_data/2021/10/10/day.csv
level_data/2021/10/11 exists
made level_data/2021/10/11/day.csv
level_data/2021/10/12 exists
made level_data/2021/10/12/day.csv
level_data/2021/10/13 exists
made level_data/2021/10/13/day.csv
level_data/2021/10/14 exists
made level_data/2021/10/14/day.csv
level_data/2021/10/15 exists
made level_data/2021/10/15/day.csv
level_data/2021/10/16 exists
made level_data/2021/10/16/day.csv
level_data/2021/10/17 exists
made level_data/2021/10/17/day.csv
level_data/2021/10/18 exists
made level_data/2021/10/18/day.csv
level_data/2021/10/19 exists
made level_data/2021/10/19/day.csv
level_data/2021/10/20 exists
made level_data/2021/10/20/day.csv
level_data/2021/10/21 exists
made level_data/2021/10/21/day.csv
level_data/2021/10/22 exists
made level_data/2021/10/22/day.csv
level_data/2021/10/23 exists
made level_data/2021/10/23/day.csv
level_data/2021/10/24 exists
made level_data/2021/10/24/day.csv
level_data/2021/10/25 exists
made level_data/2021/10/25/day.csv
level_data/2021/10/26 exists
made level_data/2021/10/26/day.csv
level_data/2021/10/27 exists
made level_data/2021/10/27/day.csv
level_data/2021/10/28 exists
made level_data/2021/10/28/day.csv
level_data/2021/10/29 exists
made level_data/2021/10/29/day.csv
level_data/2021/10/30 exists
made level_data/2021/10/30/day.csv
level_data/2021/11 exists
made level_data/2021/11/month.csv
level_data/2021/11/01 exists
made level_data/2021/11/01/day.csv
level_data/2021/11/02 exists
made level_data/2021/11/02/day.csv
level_data/2021/11/03 exists
made level_data/2021/11/03/day.csv
level_data/2021/11/04 exists
made level_data/2021/11/04/day.csv
level_data/2021/11/05 exists
made level_data/2021/11/05/day.csv
level_data/2021/11/06 exists
made level_data/2021/11/06/day.csv
level_data/2021/11/07 exists
made level_data/2021/11/07/day.csv
level_data/2021/11/08 exists
made level_data/2021/11/08/day.csv
level_data/2021/11/09 exists
made level_data/2021/11/09/day.csv
level_data/2021/11/10 exists
made level_data/2021/11/10/day.csv
level_data/2021/11/11 exists
made level_data/2021/11/11/day.csv
level_data/2021/11/12 exists
made level_data/2021/11/12/day.csv
level_data/2021/11/13 exists
made level_data/2021/11/13/day.csv
level_data/2021/11/14 exists
made level_data/2021/11/14/day.csv
level_data/2021/11/15 exists
made level_data/2021/11/15/day.csv
level_data/2021/11/16 exists
made level_data/2021/11/16/day.csv
level_data/2021/11/17 exists
made level_data/2021/11/17/day.csv
level_data/2021/11/18 exists
made level_data/2021/11/18/day.csv
level_data/2021/11/19 exists
made level_data/2021/11/19/day.csv
level_data/2021/11/20 exists
made level_data/2021/11/20/day.csv
level_data/2021/11/21 exists
made level_data/2021/11/21/day.csv
level_data/2021/11/22 exists
made level_data/2021/11/22/day.csv
level_data/2021/11/23 exists
made level_data/2021/11/23/day.csv
level_data/2021/11/24 exists
made level_data/2021/11/24/day.csv
level_data/2021/11/25 exists
made level_data/2021/11/25/day.csv
level_data/2021/11/26 exists
made level_data/2021/11/26/day.csv
level_data/2021/11/27 exists
made level_data/2021/11/27/day.csv
level_data/2021/11/28 exists
made level_data/2021/11/28/day.csv
level_data/2021/11/29 exists
made level_data/2021/11/29/day.csv
level_data/2021/11/30 exists
made level_data/2021/11/30/day.csv
level_data/2021/12 exists
made level_data/2021/12/month.csv
level_data/2021/12/01 exists
made level_data/2021/12/01/day.csv
level_data/2021/12/02 exists
made level_data/2021/12/02/day.csv
level_data/2021/12/03 exists
made level_data/2021/12/03/day.csv
level_data/2021/12/04 exists
made level_data/2021/12/04/day.csv
level_data/2021/12/05 exists
made level_data/2021/12/05/day.csv
level_data/2021/12/06 exists
made level_data/2021/12/06/day.csv
level_data/2021/12/07 exists
made level_data/2021/12/07/day.csv
level_data/2021/12/08 exists
made level_data/2021/12/08/day.csv
level_data/2021/12/09 exists
made level_data/2021/12/09/day.csv
level_data/2021/12/10 exists
made level_data/2021/12/10/day.csv
level_data/2021/12/11 exists
made level_data/2021/12/11/day.csv
level_data/2021/12/12 exists
made level_data/2021/12/12/day.csv
level_data/2021/12/13 exists
made level_data/2021/12/13/day.csv
level_data/2021/12/14 exists
made level_data/2021/12/14/day.csv
level_data/2021/12/15 exists
made level_data/2021/12/15/day.csv
level_data/2021/12/16 exists
made level_data/2021/12/16/day.csv
level_data/2021/12/17 exists
made level_data/2021/12/17/day.csv
level_data/2021/12/18 exists
made level_data/2021/12/18/day.csv
level_data/2021/12/19 exists
made level_data/2021/12/19/day.csv
level_data/2021/12/20 exists
made level_data/2021/12/20/day.csv
level_data/2021/12/21 exists
made level_data/2021/12/21/day.csv
level_data/2021/12/22 exists
made level_data/2021/12/22/day.csv
level_data/2021/12/23 exists
made level_data/2021/12/23/day.csv
level_data/2021/12/24 exists
made level_data/2021/12/24/day.csv
level_data/2021/12/25 exists
made level_data/2021/12/25/day.csv
level_data/2021/12/26 exists
made level_data/2021/12/26/day.csv
level_data/2021/12/27 exists
made level_data/2021/12/27/day.csv
level_data/2021/12/28 exists
made level_data/2021/12/28/day.csv
level_data/2021/12/29 exists
made level_data/2021/12/29/day.csv
level_data/2021/12/30 exists
made level_data/2021/12/30/day.csv 
```

## 遍历目录

`os`模块提供了`walk`，它是一个生成器函数，遍历一个目录及其所有子目录，以及它们的所有子目录，依此类推。

对于每个目录，它产生：

+   dirpath，它是目录的名称。

+   目录名，它是包含的子目录列表，以及

+   文件名，它是包含的文件列表。

这是我们如何使用它来打印我们创建的目录中所有文件的路径。

```py
for dirpath, dirnames, filenames in os.walk('level_data'):
    for filename in filenames:
        path = os.path.join(dirpath, filename)
        print(path) 
```

```py
level_data/2021/year.csv
level_data/2021/04/month.csv
level_data/2021/04/04/day.csv
level_data/2021/04/24/day.csv
level_data/2021/04/09/day.csv
level_data/2021/04/02/day.csv
level_data/2021/04/27/day.csv
level_data/2021/04/17/day.csv
level_data/2021/04/21/day.csv
level_data/2021/04/18/day.csv
level_data/2021/04/14/day.csv
level_data/2021/04/23/day.csv
level_data/2021/04/01/day.csv
level_data/2021/04/10/day.csv
level_data/2021/04/30/day.csv
level_data/2021/04/15/day.csv
level_data/2021/04/12/day.csv
level_data/2021/04/06/day.csv
level_data/2021/04/05/day.csv
level_data/2021/04/29/day.csv
level_data/2021/04/07/day.csv
level_data/2021/04/28/day.csv
level_data/2021/04/08/day.csv
level_data/2021/04/03/day.csv
level_data/2021/04/20/day.csv
level_data/2021/04/22/day.csv
level_data/2021/04/11/day.csv
level_data/2021/04/25/day.csv
level_data/2021/04/13/day.csv
level_data/2021/04/16/day.csv
level_data/2021/04/26/day.csv
level_data/2021/04/19/day.csv
level_data/2021/09/month.csv
level_data/2021/09/04/day.csv
level_data/2021/09/24/day.csv
level_data/2021/09/09/day.csv
level_data/2021/09/02/day.csv
level_data/2021/09/27/day.csv
level_data/2021/09/17/day.csv
level_data/2021/09/21/day.csv
level_data/2021/09/18/day.csv
level_data/2021/09/14/day.csv
level_data/2021/09/23/day.csv
level_data/2021/09/01/day.csv
level_data/2021/09/10/day.csv
level_data/2021/09/30/day.csv
level_data/2021/09/15/day.csv
level_data/2021/09/12/day.csv
level_data/2021/09/06/day.csv
level_data/2021/09/05/day.csv
level_data/2021/09/29/day.csv
level_data/2021/09/07/day.csv
level_data/2021/09/28/day.csv
level_data/2021/09/08/day.csv
level_data/2021/09/03/day.csv
level_data/2021/09/20/day.csv
level_data/2021/09/22/day.csv
level_data/2021/09/11/day.csv
level_data/2021/09/25/day.csv
level_data/2021/09/13/day.csv
level_data/2021/09/16/day.csv
level_data/2021/09/26/day.csv
level_data/2021/09/19/day.csv
level_data/2021/02/month.csv
level_data/2021/02/04/day.csv
level_data/2021/02/24/day.csv
level_data/2021/02/09/day.csv
level_data/2021/02/02/day.csv
level_data/2021/02/27/day.csv
level_data/2021/02/17/day.csv
level_data/2021/02/21/day.csv
level_data/2021/02/18/day.csv
level_data/2021/02/14/day.csv
level_data/2021/02/23/day.csv
level_data/2021/02/01/day.csv
level_data/2021/02/10/day.csv
level_data/2021/02/30/day.csv
level_data/2021/02/15/day.csv
level_data/2021/02/12/day.csv
level_data/2021/02/06/day.csv
level_data/2021/02/05/day.csv
level_data/2021/02/29/day.csv
level_data/2021/02/07/day.csv
level_data/2021/02/28/day.csv
level_data/2021/02/08/day.csv
level_data/2021/02/03/day.csv
level_data/2021/02/20/day.csv
level_data/2021/02/22/day.csv
level_data/2021/02/11/day.csv
level_data/2021/02/25/day.csv
level_data/2021/02/13/day.csv
level_data/2021/02/16/day.csv
level_data/2021/02/26/day.csv
level_data/2021/02/19/day.csv
level_data/2021/01/month.csv
level_data/2021/01/04/day.csv
level_data/2021/01/24/day.csv
level_data/2021/01/09/day.csv
level_data/2021/01/02/day.csv
level_data/2021/01/27/day.csv
level_data/2021/01/17/day.csv
level_data/2021/01/21/day.csv
level_data/2021/01/18/day.csv
level_data/2021/01/14/day.csv
level_data/2021/01/23/day.csv
level_data/2021/01/01/day.csv
level_data/2021/01/10/day.csv
level_data/2021/01/30/day.csv
level_data/2021/01/15/day.csv
level_data/2021/01/12/day.csv
level_data/2021/01/06/day.csv
level_data/2021/01/05/day.csv
level_data/2021/01/29/day.csv
level_data/2021/01/07/day.csv
level_data/2021/01/28/day.csv
level_data/2021/01/08/day.csv
level_data/2021/01/03/day.csv
level_data/2021/01/20/day.csv
level_data/2021/01/22/day.csv
level_data/2021/01/11/day.csv
level_data/2021/01/25/day.csv
level_data/2021/01/13/day.csv
level_data/2021/01/16/day.csv
level_data/2021/01/26/day.csv
level_data/2021/01/19/day.csv
level_data/2021/10/month.csv
level_data/2021/10/04/day.csv
level_data/2021/10/24/day.csv
level_data/2021/10/09/day.csv
level_data/2021/10/02/day.csv
level_data/2021/10/27/day.csv
level_data/2021/10/17/day.csv
level_data/2021/10/21/day.csv
level_data/2021/10/18/day.csv
level_data/2021/10/14/day.csv
level_data/2021/10/23/day.csv
level_data/2021/10/01/day.csv
level_data/2021/10/10/day.csv
level_data/2021/10/30/day.csv
level_data/2021/10/15/day.csv
level_data/2021/10/12/day.csv
level_data/2021/10/06/day.csv
level_data/2021/10/05/day.csv
level_data/2021/10/29/day.csv
level_data/2021/10/07/day.csv
level_data/2021/10/28/day.csv
level_data/2021/10/08/day.csv
level_data/2021/10/03/day.csv
level_data/2021/10/20/day.csv
level_data/2021/10/22/day.csv
level_data/2021/10/11/day.csv
level_data/2021/10/25/day.csv
level_data/2021/10/13/day.csv
level_data/2021/10/16/day.csv
level_data/2021/10/26/day.csv
level_data/2021/10/19/day.csv
level_data/2021/12/month.csv
level_data/2021/12/04/day.csv
level_data/2021/12/24/day.csv
level_data/2021/12/09/day.csv
level_data/2021/12/02/day.csv
level_data/2021/12/27/day.csv
level_data/2021/12/17/day.csv
level_data/2021/12/21/day.csv
level_data/2021/12/18/day.csv
level_data/2021/12/14/day.csv
level_data/2021/12/23/day.csv
level_data/2021/12/01/day.csv
level_data/2021/12/10/day.csv
level_data/2021/12/30/day.csv
level_data/2021/12/15/day.csv
level_data/2021/12/12/day.csv
level_data/2021/12/06/day.csv
level_data/2021/12/05/day.csv
level_data/2021/12/29/day.csv
level_data/2021/12/07/day.csv
level_data/2021/12/28/day.csv
level_data/2021/12/08/day.csv
level_data/2021/12/03/day.csv
level_data/2021/12/20/day.csv
level_data/2021/12/22/day.csv
level_data/2021/12/11/day.csv
level_data/2021/12/25/day.csv
level_data/2021/12/13/day.csv
level_data/2021/12/16/day.csv
level_data/2021/12/26/day.csv
level_data/2021/12/19/day.csv
level_data/2021/06/month.csv
level_data/2021/06/04/day.csv
level_data/2021/06/24/day.csv
level_data/2021/06/09/day.csv
level_data/2021/06/02/day.csv
level_data/2021/06/27/day.csv
level_data/2021/06/17/day.csv
level_data/2021/06/21/day.csv
level_data/2021/06/18/day.csv
level_data/2021/06/14/day.csv
level_data/2021/06/23/day.csv
level_data/2021/06/01/day.csv
level_data/2021/06/10/day.csv
level_data/2021/06/30/day.csv
level_data/2021/06/15/day.csv
level_data/2021/06/12/day.csv
level_data/2021/06/06/day.csv
level_data/2021/06/05/day.csv
level_data/2021/06/29/day.csv
level_data/2021/06/07/day.csv
level_data/2021/06/28/day.csv
level_data/2021/06/08/day.csv
level_data/2021/06/03/day.csv
level_data/2021/06/20/day.csv
level_data/2021/06/22/day.csv
level_data/2021/06/11/day.csv
level_data/2021/06/25/day.csv
level_data/2021/06/13/day.csv
level_data/2021/06/16/day.csv
level_data/2021/06/26/day.csv
level_data/2021/06/19/day.csv
level_data/2021/05/month.csv
level_data/2021/05/04/day.csv
level_data/2021/05/24/day.csv
level_data/2021/05/09/day.csv
level_data/2021/05/02/day.csv
level_data/2021/05/27/day.csv
level_data/2021/05/17/day.csv
level_data/2021/05/21/day.csv
level_data/2021/05/18/day.csv
level_data/2021/05/14/day.csv
level_data/2021/05/23/day.csv
level_data/2021/05/01/day.csv
level_data/2021/05/10/day.csv
level_data/2021/05/30/day.csv
level_data/2021/05/15/day.csv
level_data/2021/05/12/day.csv
level_data/2021/05/06/day.csv
level_data/2021/05/05/day.csv
level_data/2021/05/29/day.csv
level_data/2021/05/07/day.csv
level_data/2021/05/28/day.csv
level_data/2021/05/08/day.csv
level_data/2021/05/03/day.csv
level_data/2021/05/20/day.csv
level_data/2021/05/22/day.csv
level_data/2021/05/11/day.csv
level_data/2021/05/25/day.csv
level_data/2021/05/13/day.csv
level_data/2021/05/16/day.csv
level_data/2021/05/26/day.csv
level_data/2021/05/19/day.csv
level_data/2021/07/month.csv
level_data/2021/07/04/day.csv
level_data/2021/07/24/day.csv
level_data/2021/07/09/day.csv
level_data/2021/07/02/day.csv
level_data/2021/07/27/day.csv
level_data/2021/07/17/day.csv
level_data/2021/07/21/day.csv
level_data/2021/07/18/day.csv
level_data/2021/07/14/day.csv
level_data/2021/07/23/day.csv
level_data/2021/07/01/day.csv
level_data/2021/07/10/day.csv
level_data/2021/07/30/day.csv
level_data/2021/07/15/day.csv
level_data/2021/07/12/day.csv
level_data/2021/07/06/day.csv
level_data/2021/07/05/day.csv
level_data/2021/07/29/day.csv
level_data/2021/07/07/day.csv
level_data/2021/07/28/day.csv
level_data/2021/07/08/day.csv
level_data/2021/07/03/day.csv
level_data/2021/07/20/day.csv
level_data/2021/07/22/day.csv
level_data/2021/07/11/day.csv
level_data/2021/07/25/day.csv
level_data/2021/07/13/day.csv
level_data/2021/07/16/day.csv
level_data/2021/07/26/day.csv
level_data/2021/07/19/day.csv
level_data/2021/08/month.csv
level_data/2021/08/04/day.csv
level_data/2021/08/24/day.csv
level_data/2021/08/09/day.csv
level_data/2021/08/02/day.csv
level_data/2021/08/27/day.csv
level_data/2021/08/17/day.csv
level_data/2021/08/21/day.csv
level_data/2021/08/18/day.csv
level_data/2021/08/14/day.csv
level_data/2021/08/23/day.csv
level_data/2021/08/01/day.csv
level_data/2021/08/10/day.csv
level_data/2021/08/30/day.csv
level_data/2021/08/15/day.csv
level_data/2021/08/12/day.csv
level_data/2021/08/06/day.csv
level_data/2021/08/05/day.csv
level_data/2021/08/29/day.csv
level_data/2021/08/07/day.csv
level_data/2021/08/28/day.csv
level_data/2021/08/08/day.csv
level_data/2021/08/03/day.csv
level_data/2021/08/20/day.csv
level_data/2021/08/22/day.csv
level_data/2021/08/11/day.csv
level_data/2021/08/25/day.csv
level_data/2021/08/13/day.csv
level_data/2021/08/16/day.csv
level_data/2021/08/26/day.csv
level_data/2021/08/19/day.csv
level_data/2021/03/month.csv
level_data/2021/03/04/day.csv
level_data/2021/03/24/day.csv
level_data/2021/03/09/day.csv
level_data/2021/03/02/day.csv
level_data/2021/03/27/day.csv
level_data/2021/03/17/day.csv
level_data/2021/03/21/day.csv
level_data/2021/03/18/day.csv
level_data/2021/03/14/day.csv
level_data/2021/03/23/day.csv
level_data/2021/03/01/day.csv
level_data/2021/03/10/day.csv
level_data/2021/03/30/day.csv
level_data/2021/03/15/day.csv
level_data/2021/03/12/day.csv
level_data/2021/03/06/day.csv
level_data/2021/03/05/day.csv
level_data/2021/03/29/day.csv
level_data/2021/03/07/day.csv
level_data/2021/03/28/day.csv
level_data/2021/03/08/day.csv
level_data/2021/03/03/day.csv
level_data/2021/03/20/day.csv
level_data/2021/03/22/day.csv
level_data/2021/03/11/day.csv
level_data/2021/03/25/day.csv
level_data/2021/03/13/day.csv
level_data/2021/03/16/day.csv
level_data/2021/03/26/day.csv
level_data/2021/03/19/day.csv
level_data/2021/11/month.csv
level_data/2021/11/04/day.csv
level_data/2021/11/24/day.csv
level_data/2021/11/09/day.csv
level_data/2021/11/02/day.csv
level_data/2021/11/27/day.csv
level_data/2021/11/17/day.csv
level_data/2021/11/21/day.csv
level_data/2021/11/18/day.csv
level_data/2021/11/14/day.csv
level_data/2021/11/23/day.csv
level_data/2021/11/01/day.csv
level_data/2021/11/10/day.csv
level_data/2021/11/30/day.csv
level_data/2021/11/15/day.csv
level_data/2021/11/12/day.csv
level_data/2021/11/06/day.csv
level_data/2021/11/05/day.csv
level_data/2021/11/29/day.csv
level_data/2021/11/07/day.csv
level_data/2021/11/28/day.csv
level_data/2021/11/08/day.csv
level_data/2021/11/03/day.csv
level_data/2021/11/20/day.csv
level_data/2021/11/22/day.csv
level_data/2021/11/11/day.csv
level_data/2021/11/25/day.csv
level_data/2021/11/13/day.csv
level_data/2021/11/16/day.csv
level_data/2021/11/26/day.csv
level_data/2021/11/19/day.csv
level_data/2021/00/month.csv
level_data/2021/00/04/day.csv
level_data/2021/00/24/day.csv
level_data/2021/00/09/day.csv
level_data/2021/00/02/day.csv
level_data/2021/00/27/day.csv
level_data/2021/00/17/day.csv
level_data/2021/00/21/day.csv
level_data/2021/00/18/day.csv
level_data/2021/00/14/day.csv
level_data/2021/00/23/day.csv
level_data/2021/00/01/day.csv
level_data/2021/00/10/day.csv
level_data/2021/00/30/day.csv
level_data/2021/00/15/day.csv
level_data/2021/00/12/day.csv
level_data/2021/00/06/day.csv
level_data/2021/00/05/day.csv
level_data/2021/00/29/day.csv
level_data/2021/00/07/day.csv
level_data/2021/00/28/day.csv
level_data/2021/00/08/day.csv
level_data/2021/00/03/day.csv
level_data/2021/00/20/day.csv
level_data/2021/00/22/day.csv
level_data/2021/00/11/day.csv
level_data/2021/00/25/day.csv
level_data/2021/00/13/day.csv
level_data/2021/00/16/day.csv
level_data/2021/00/26/day.csv
level_data/2021/00/19/day.csv 
```

`os.walk`的一个特点是目录和文件不会按任何特定顺序出现。当然，我们可以存储结果并按我们想要的任何顺序对其进行排序。

但作为练习，我们可以编写我们自己的`walk`版本。我们需要两个函数：

+   `os.listdir`，它接受一个目录并列出它包含的目录和文件，以及

+   `os.path.isfile`，它接受一个路径并返回`True`，如果它是一个文件，则返回`False`，如果它是一个目录或其他东西。

你可能会注意到一些与文件相关的函数在子模块`os.path`中。这种组织有一些逻辑，但并不总是明显为什么特定的函数在这个子模块中或者不在其中。

无论如何，这是`walk`的递归版本：

```py
def walk(dirname):
    for name in sorted(os.listdir(dirname)):
        path = os.path.join(dirname, name)
        if os.path.isfile(path):
            print(path)
        else:
            walk(path) 
```

```py
walk(year_dir) 
```

```py
level_data/2021/00/01/day.csv
level_data/2021/00/02/day.csv
level_data/2021/00/03/day.csv
level_data/2021/00/04/day.csv
level_data/2021/00/05/day.csv
level_data/2021/00/06/day.csv
level_data/2021/00/07/day.csv
level_data/2021/00/08/day.csv
level_data/2021/00/09/day.csv
level_data/2021/00/10/day.csv
level_data/2021/00/11/day.csv
level_data/2021/00/12/day.csv
level_data/2021/00/13/day.csv
level_data/2021/00/14/day.csv
level_data/2021/00/15/day.csv
level_data/2021/00/16/day.csv
level_data/2021/00/17/day.csv
level_data/2021/00/18/day.csv
level_data/2021/00/19/day.csv
level_data/2021/00/20/day.csv
level_data/2021/00/21/day.csv
level_data/2021/00/22/day.csv
level_data/2021/00/23/day.csv
level_data/2021/00/24/day.csv
level_data/2021/00/25/day.csv
level_data/2021/00/26/day.csv
level_data/2021/00/27/day.csv
level_data/2021/00/28/day.csv
level_data/2021/00/29/day.csv
level_data/2021/00/30/day.csv
level_data/2021/00/month.csv
level_data/2021/01/01/day.csv
level_data/2021/01/02/day.csv
level_data/2021/01/03/day.csv
level_data/2021/01/04/day.csv
level_data/2021/01/05/day.csv
level_data/2021/01/06/day.csv
level_data/2021/01/07/day.csv
level_data/2021/01/08/day.csv
level_data/2021/01/09/day.csv
level_data/2021/01/10/day.csv
level_data/2021/01/11/day.csv
level_data/2021/01/12/day.csv
level_data/2021/01/13/day.csv
level_data/2021/01/14/day.csv
level_data/2021/01/15/day.csv
level_data/2021/01/16/day.csv
level_data/2021/01/17/day.csv
level_data/2021/01/18/day.csv
level_data/2021/01/19/day.csv
level_data/2021/01/20/day.csv
level_data/2021/01/21/day.csv
level_data/2021/01/22/day.csv
level_data/2021/01/23/day.csv
level_data/2021/01/24/day.csv
level_data/2021/01/25/day.csv
level_data/2021/01/26/day.csv
level_data/2021/01/27/day.csv
level_data/2021/01/28/day.csv
level_data/2021/01/29/day.csv
level_data/2021/01/30/day.csv
level_data/2021/01/month.csv
level_data/2021/02/01/day.csv
level_data/2021/02/02/day.csv
level_data/2021/02/03/day.csv
level_data/2021/02/04/day.csv
level_data/2021/02/05/day.csv
level_data/2021/02/06/day.csv
level_data/2021/02/07/day.csv
level_data/2021/02/08/day.csv
level_data/2021/02/09/day.csv
level_data/2021/02/10/day.csv
level_data/2021/02/11/day.csv
level_data/2021/02/12/day.csv
level_data/2021/02/13/day.csv
level_data/2021/02/14/day.csv
level_data/2021/02/15/day.csv
level_data/2021/02/16/day.csv
level_data/2021/02/17/day.csv
level_data/2021/02/18/day.csv
level_data/2021/02/19/day.csv
level_data/2021/02/20/day.csv
level_data/2021/02/21/day.csv
level_data/2021/02/22/day.csv
level_data/2021/02/23/day.csv
level_data/2021/02/24/day.csv
level_data/2021/02/25/day.csv
level_data/2021/02/26/day.csv
level_data/2021/02/27/day.csv
level_data/2021/02/28/day.csv
level_data/2021/02/29/day.csv
level_data/2021/02/30/day.csv
level_data/2021/02/month.csv
level_data/2021/03/01/day.csv
level_data/2021/03/02/day.csv
level_data/2021/03/03/day.csv
level_data/2021/03/04/day.csv
level_data/2021/03/05/day.csv
level_data/2021/03/06/day.csv
level_data/2021/03/07/day.csv
level_data/2021/03/08/day.csv
level_data/2021/03/09/day.csv
level_data/2021/03/10/day.csv
level_data/2021/03/11/day.csv
level_data/2021/03/12/day.csv
level_data/2021/03/13/day.csv
level_data/2021/03/14/day.csv
level_data/2021/03/15/day.csv
level_data/2021/03/16/day.csv
level_data/2021/03/17/day.csv
level_data/2021/03/18/day.csv
level_data/2021/03/19/day.csv
level_data/2021/03/20/day.csv
level_data/2021/03/21/day.csv
level_data/2021/03/22/day.csv
level_data/2021/03/23/day.csv
level_data/2021/03/24/day.csv
level_data/2021/03/25/day.csv
level_data/2021/03/26/day.csv
level_data/2021/03/27/day.csv
level_data/2021/03/28/day.csv
level_data/2021/03/29/day.csv
level_data/2021/03/30/day.csv
level_data/2021/03/month.csv
level_data/2021/04/01/day.csv
level_data/2021/04/02/day.csv
level_data/2021/04/03/day.csv
level_data/2021/04/04/day.csv
level_data/2021/04/05/day.csv
level_data/2021/04/06/day.csv
level_data/2021/04/07/day.csv
level_data/2021/04/08/day.csv
level_data/2021/04/09/day.csv
level_data/2021/04/10/day.csv
level_data/2021/04/11/day.csv
level_data/2021/04/12/day.csv
level_data/2021/04/13/day.csv
level_data/2021/04/14/day.csv
level_data/2021/04/15/day.csv
level_data/2021/04/16/day.csv
level_data/2021/04/17/day.csv
level_data/2021/04/18/day.csv
level_data/2021/04/19/day.csv
level_data/2021/04/20/day.csv
level_data/2021/04/21/day.csv
level_data/2021/04/22/day.csv
level_data/2021/04/23/day.csv
level_data/2021/04/24/day.csv
level_data/2021/04/25/day.csv
level_data/2021/04/26/day.csv
level_data/2021/04/27/day.csv
level_data/2021/04/28/day.csv
level_data/2021/04/29/day.csv
level_data/2021/04/30/day.csv
level_data/2021/04/month.csv
level_data/2021/05/01/day.csv
level_data/2021/05/02/day.csv
level_data/2021/05/03/day.csv
level_data/2021/05/04/day.csv
level_data/2021/05/05/day.csv
level_data/2021/05/06/day.csv
level_data/2021/05/07/day.csv
level_data/2021/05/08/day.csv
level_data/2021/05/09/day.csv
level_data/2021/05/10/day.csv
level_data/2021/05/11/day.csv
level_data/2021/05/12/day.csv
level_data/2021/05/13/day.csv
level_data/2021/05/14/day.csv
level_data/2021/05/15/day.csv
level_data/2021/05/16/day.csv
level_data/2021/05/17/day.csv
level_data/2021/05/18/day.csv
level_data/2021/05/19/day.csv
level_data/2021/05/20/day.csv
level_data/2021/05/21/day.csv
level_data/2021/05/22/day.csv
level_data/2021/05/23/day.csv
level_data/2021/05/24/day.csv
level_data/2021/05/25/day.csv
level_data/2021/05/26/day.csv
level_data/2021/05/27/day.csv
level_data/2021/05/28/day.csv
level_data/2021/05/29/day.csv
level_data/2021/05/30/day.csv
level_data/2021/05/month.csv
level_data/2021/06/01/day.csv
level_data/2021/06/02/day.csv
level_data/2021/06/03/day.csv
level_data/2021/06/04/day.csv
level_data/2021/06/05/day.csv
level_data/2021/06/06/day.csv
level_data/2021/06/07/day.csv
level_data/2021/06/08/day.csv
level_data/2021/06/09/day.csv
level_data/2021/06/10/day.csv
level_data/2021/06/11/day.csv
level_data/2021/06/12/day.csv
level_data/2021/06/13/day.csv
level_data/2021/06/14/day.csv
level_data/2021/06/15/day.csv
level_data/2021/06/16/day.csv
level_data/2021/06/17/day.csv
level_data/2021/06/18/day.csv
level_data/2021/06/19/day.csv
level_data/2021/06/20/day.csv
level_data/2021/06/21/day.csv
level_data/2021/06/22/day.csv
level_data/2021/06/23/day.csv
level_data/2021/06/24/day.csv
level_data/2021/06/25/day.csv
level_data/2021/06/26/day.csv
level_data/2021/06/27/day.csv
level_data/2021/06/28/day.csv
level_data/2021/06/29/day.csv
level_data/2021/06/30/day.csv
level_data/2021/06/month.csv
level_data/2021/07/01/day.csv
level_data/2021/07/02/day.csv
level_data/2021/07/03/day.csv
level_data/2021/07/04/day.csv
level_data/2021/07/05/day.csv
level_data/2021/07/06/day.csv
level_data/2021/07/07/day.csv
level_data/2021/07/08/day.csv
level_data/2021/07/09/day.csv
level_data/2021/07/10/day.csv
level_data/2021/07/11/day.csv
level_data/2021/07/12/day.csv
level_data/2021/07/13/day.csv
level_data/2021/07/14/day.csv
level_data/2021/07/15/day.csv
level_data/2021/07/16/day.csv
level_data/2021/07/17/day.csv
level_data/2021/07/18/day.csv
level_data/2021/07/19/day.csv
level_data/2021/07/20/day.csv
level_data/2021/07/21/day.csv
level_data/2021/07/22/day.csv
level_data/2021/07/23/day.csv
level_data/2021/07/24/day.csv
level_data/2021/07/25/day.csv
level_data/2021/07/26/day.csv
level_data/2021/07/27/day.csv
level_data/2021/07/28/day.csv
level_data/2021/07/29/day.csv
level_data/2021/07/30/day.csv
level_data/2021/07/month.csv
level_data/2021/08/01/day.csv
level_data/2021/08/02/day.csv
level_data/2021/08/03/day.csv
level_data/2021/08/04/day.csv
level_data/2021/08/05/day.csv
level_data/2021/08/06/day.csv
level_data/2021/08/07/day.csv
level_data/2021/08/08/day.csv
level_data/2021/08/09/day.csv
level_data/2021/08/10/day.csv
level_data/2021/08/11/day.csv
level_data/2021/08/12/day.csv
level_data/2021/08/13/day.csv
level_data/2021/08/14/day.csv
level_data/2021/08/15/day.csv
level_data/2021/08/16/day.csv
level_data/2021/08/17/day.csv
level_data/2021/08/18/day.csv
level_data/2021/08/19/day.csv
level_data/2021/08/20/day.csv
level_data/2021/08/21/day.csv
level_data/2021/08/22/day.csv
level_data/2021/08/23/day.csv
level_data/2021/08/24/day.csv
level_data/2021/08/25/day.csv
level_data/2021/08/26/day.csv
level_data/2021/08/27/day.csv
level_data/2021/08/28/day.csv
level_data/2021/08/29/day.csv
level_data/2021/08/30/day.csv
level_data/2021/08/month.csv
level_data/2021/09/01/day.csv
level_data/2021/09/02/day.csv
level_data/2021/09/03/day.csv
level_data/2021/09/04/day.csv
level_data/2021/09/05/day.csv
level_data/2021/09/06/day.csv
level_data/2021/09/07/day.csv
level_data/2021/09/08/day.csv
level_data/2021/09/09/day.csv 
```

```py
level_data/2021/09/10/day.csv
level_data/2021/09/11/day.csv
level_data/2021/09/12/day.csv
level_data/2021/09/13/day.csv
level_data/2021/09/14/day.csv
level_data/2021/09/15/day.csv
level_data/2021/09/16/day.csv
level_data/2021/09/17/day.csv
level_data/2021/09/18/day.csv
level_data/2021/09/19/day.csv
level_data/2021/09/20/day.csv
level_data/2021/09/21/day.csv
level_data/2021/09/22/day.csv
level_data/2021/09/23/day.csv
level_data/2021/09/24/day.csv
level_data/2021/09/25/day.csv
level_data/2021/09/26/day.csv
level_data/2021/09/27/day.csv
level_data/2021/09/28/day.csv
level_data/2021/09/29/day.csv
level_data/2021/09/30/day.csv
level_data/2021/09/month.csv
level_data/2021/10/01/day.csv
level_data/2021/10/02/day.csv
level_data/2021/10/03/day.csv
level_data/2021/10/04/day.csv
level_data/2021/10/05/day.csv
level_data/2021/10/06/day.csv
level_data/2021/10/07/day.csv
level_data/2021/10/08/day.csv
level_data/2021/10/09/day.csv
level_data/2021/10/10/day.csv
level_data/2021/10/11/day.csv
level_data/2021/10/12/day.csv
level_data/2021/10/13/day.csv
level_data/2021/10/14/day.csv
level_data/2021/10/15/day.csv
level_data/2021/10/16/day.csv
level_data/2021/10/17/day.csv
level_data/2021/10/18/day.csv
level_data/2021/10/19/day.csv
level_data/2021/10/20/day.csv
level_data/2021/10/21/day.csv
level_data/2021/10/22/day.csv
level_data/2021/10/23/day.csv
level_data/2021/10/24/day.csv
level_data/2021/10/25/day.csv
level_data/2021/10/26/day.csv
level_data/2021/10/27/day.csv
level_data/2021/10/28/day.csv
level_data/2021/10/29/day.csv
level_data/2021/10/30/day.csv
level_data/2021/10/month.csv
level_data/2021/11/01/day.csv
level_data/2021/11/02/day.csv
level_data/2021/11/03/day.csv
level_data/2021/11/04/day.csv
level_data/2021/11/05/day.csv
level_data/2021/11/06/day.csv
level_data/2021/11/07/day.csv
level_data/2021/11/08/day.csv
level_data/2021/11/09/day.csv
level_data/2021/11/10/day.csv
level_data/2021/11/11/day.csv
level_data/2021/11/12/day.csv
level_data/2021/11/13/day.csv
level_data/2021/11/14/day.csv
level_data/2021/11/15/day.csv
level_data/2021/11/16/day.csv
level_data/2021/11/17/day.csv
level_data/2021/11/18/day.csv
level_data/2021/11/19/day.csv
level_data/2021/11/20/day.csv
level_data/2021/11/21/day.csv
level_data/2021/11/22/day.csv
level_data/2021/11/23/day.csv
level_data/2021/11/24/day.csv
level_data/2021/11/25/day.csv
level_data/2021/11/26/day.csv
level_data/2021/11/27/day.csv
level_data/2021/11/28/day.csv
level_data/2021/11/29/day.csv
level_data/2021/11/30/day.csv
level_data/2021/11/month.csv
level_data/2021/12/01/day.csv
level_data/2021/12/02/day.csv
level_data/2021/12/03/day.csv
level_data/2021/12/04/day.csv
level_data/2021/12/05/day.csv
level_data/2021/12/06/day.csv
level_data/2021/12/07/day.csv
level_data/2021/12/08/day.csv
level_data/2021/12/09/day.csv
level_data/2021/12/10/day.csv
level_data/2021/12/11/day.csv
level_data/2021/12/12/day.csv
level_data/2021/12/13/day.csv
level_data/2021/12/14/day.csv
level_data/2021/12/15/day.csv
level_data/2021/12/16/day.csv
level_data/2021/12/17/day.csv
level_data/2021/12/18/day.csv
level_data/2021/12/19/day.csv
level_data/2021/12/20/day.csv
level_data/2021/12/21/day.csv
level_data/2021/12/22/day.csv
level_data/2021/12/23/day.csv
level_data/2021/12/24/day.csv
level_data/2021/12/25/day.csv
level_data/2021/12/26/day.csv
level_data/2021/12/27/day.csv
level_data/2021/12/28/day.csv
level_data/2021/12/29/day.csv
level_data/2021/12/30/day.csv
level_data/2021/12/month.csv
level_data/2021/year.csv 
```

**练习：**编写一个名为`walk_gen`的`walk`版本，它是一个生成器函数；也就是说，它应该产生它找到的路径，而不是打印它们。

您可以使用以下循环来测试您的代码。

```py
for path in walk_gen(year_dir):
    print(path) 
```

```py
level_data/2021/00/01/day.csv
level_data/2021/00/02/day.csv
level_data/2021/00/03/day.csv
level_data/2021/00/04/day.csv
level_data/2021/00/05/day.csv
level_data/2021/00/06/day.csv
level_data/2021/00/07/day.csv
level_data/2021/00/08/day.csv
level_data/2021/00/09/day.csv
level_data/2021/00/10/day.csv
level_data/2021/00/11/day.csv
level_data/2021/00/12/day.csv
level_data/2021/00/13/day.csv
level_data/2021/00/14/day.csv
level_data/2021/00/15/day.csv
level_data/2021/00/16/day.csv
level_data/2021/00/17/day.csv
level_data/2021/00/18/day.csv
level_data/2021/00/19/day.csv
level_data/2021/00/20/day.csv
level_data/2021/00/21/day.csv
level_data/2021/00/22/day.csv
level_data/2021/00/23/day.csv
level_data/2021/00/24/day.csv
level_data/2021/00/25/day.csv
level_data/2021/00/26/day.csv
level_data/2021/00/27/day.csv
level_data/2021/00/28/day.csv
level_data/2021/00/29/day.csv
level_data/2021/00/30/day.csv
level_data/2021/00/month.csv
level_data/2021/01/01/day.csv
level_data/2021/01/02/day.csv
level_data/2021/01/03/day.csv
level_data/2021/01/04/day.csv
level_data/2021/01/05/day.csv
level_data/2021/01/06/day.csv
level_data/2021/01/07/day.csv
level_data/2021/01/08/day.csv
level_data/2021/01/09/day.csv
level_data/2021/01/10/day.csv
level_data/2021/01/11/day.csv
level_data/2021/01/12/day.csv
level_data/2021/01/13/day.csv
level_data/2021/01/14/day.csv
level_data/2021/01/15/day.csv
level_data/2021/01/16/day.csv
level_data/2021/01/17/day.csv
level_data/2021/01/18/day.csv
level_data/2021/01/19/day.csv
level_data/2021/01/20/day.csv
level_data/2021/01/21/day.csv
level_data/2021/01/22/day.csv
level_data/2021/01/23/day.csv
level_data/2021/01/24/day.csv
level_data/2021/01/25/day.csv
level_data/2021/01/26/day.csv
level_data/2021/01/27/day.csv
level_data/2021/01/28/day.csv
level_data/2021/01/29/day.csv
level_data/2021/01/30/day.csv
level_data/2021/01/month.csv
level_data/2021/02/01/day.csv
level_data/2021/02/02/day.csv
level_data/2021/02/03/day.csv
level_data/2021/02/04/day.csv
level_data/2021/02/05/day.csv
level_data/2021/02/06/day.csv
level_data/2021/02/07/day.csv
level_data/2021/02/08/day.csv
level_data/2021/02/09/day.csv
level_data/2021/02/10/day.csv
level_data/2021/02/11/day.csv
level_data/2021/02/12/day.csv
level_data/2021/02/13/day.csv
level_data/2021/02/14/day.csv
level_data/2021/02/15/day.csv
level_data/2021/02/16/day.csv
level_data/2021/02/17/day.csv
level_data/2021/02/18/day.csv
level_data/2021/02/19/day.csv
level_data/2021/02/20/day.csv
level_data/2021/02/21/day.csv
level_data/2021/02/22/day.csv
level_data/2021/02/23/day.csv
level_data/2021/02/24/day.csv
level_data/2021/02/25/day.csv
level_data/2021/02/26/day.csv
level_data/2021/02/27/day.csv
level_data/2021/02/28/day.csv
level_data/2021/02/29/day.csv
level_data/2021/02/30/day.csv
level_data/2021/02/month.csv
level_data/2021/03/01/day.csv
level_data/2021/03/02/day.csv
level_data/2021/03/03/day.csv
level_data/2021/03/04/day.csv
level_data/2021/03/05/day.csv
level_data/2021/03/06/day.csv
level_data/2021/03/07/day.csv
level_data/2021/03/08/day.csv
level_data/2021/03/09/day.csv
level_data/2021/03/10/day.csv
level_data/2021/03/11/day.csv
level_data/2021/03/12/day.csv
level_data/2021/03/13/day.csv
level_data/2021/03/14/day.csv
level_data/2021/03/15/day.csv
level_data/2021/03/16/day.csv
level_data/2021/03/17/day.csv
level_data/2021/03/18/day.csv
level_data/2021/03/19/day.csv
level_data/2021/03/20/day.csv
level_data/2021/03/21/day.csv
level_data/2021/03/22/day.csv
level_data/2021/03/23/day.csv
level_data/2021/03/24/day.csv
level_data/2021/03/25/day.csv
level_data/2021/03/26/day.csv
level_data/2021/03/27/day.csv
level_data/2021/03/28/day.csv
level_data/2021/03/29/day.csv
level_data/2021/03/30/day.csv
level_data/2021/03/month.csv
level_data/2021/04/01/day.csv
level_data/2021/04/02/day.csv
level_data/2021/04/03/day.csv
level_data/2021/04/04/day.csv
level_data/2021/04/05/day.csv
level_data/2021/04/06/day.csv
level_data/2021/04/07/day.csv
level_data/2021/04/08/day.csv
level_data/2021/04/09/day.csv
level_data/2021/04/10/day.csv
level_data/2021/04/11/day.csv
level_data/2021/04/12/day.csv
level_data/2021/04/13/day.csv
level_data/2021/04/14/day.csv
level_data/2021/04/15/day.csv
level_data/2021/04/16/day.csv
level_data/2021/04/17/day.csv
level_data/2021/04/18/day.csv
level_data/2021/04/19/day.csv
level_data/2021/04/20/day.csv
level_data/2021/04/21/day.csv
level_data/2021/04/22/day.csv
level_data/2021/04/23/day.csv
level_data/2021/04/24/day.csv
level_data/2021/04/25/day.csv
level_data/2021/04/26/day.csv
level_data/2021/04/27/day.csv
level_data/2021/04/28/day.csv
level_data/2021/04/29/day.csv
level_data/2021/04/30/day.csv
level_data/2021/04/month.csv
level_data/2021/05/01/day.csv
level_data/2021/05/02/day.csv
level_data/2021/05/03/day.csv
level_data/2021/05/04/day.csv
level_data/2021/05/05/day.csv
level_data/2021/05/06/day.csv
level_data/2021/05/07/day.csv
level_data/2021/05/08/day.csv
level_data/2021/05/09/day.csv
level_data/2021/05/10/day.csv
level_data/2021/05/11/day.csv
level_data/2021/05/12/day.csv
level_data/2021/05/13/day.csv
level_data/2021/05/14/day.csv
level_data/2021/05/15/day.csv
level_data/2021/05/16/day.csv
level_data/2021/05/17/day.csv
level_data/2021/05/18/day.csv
level_data/2021/05/19/day.csv
level_data/2021/05/20/day.csv
level_data/2021/05/21/day.csv
level_data/2021/05/22/day.csv
level_data/2021/05/23/day.csv
level_data/2021/05/24/day.csv
level_data/2021/05/25/day.csv
level_data/2021/05/26/day.csv
level_data/2021/05/27/day.csv
level_data/2021/05/28/day.csv
level_data/2021/05/29/day.csv
level_data/2021/05/30/day.csv
level_data/2021/05/month.csv
level_data/2021/06/01/day.csv
level_data/2021/06/02/day.csv
level_data/2021/06/03/day.csv
level_data/2021/06/04/day.csv
level_data/2021/06/05/day.csv
level_data/2021/06/06/day.csv
level_data/2021/06/07/day.csv
level_data/2021/06/08/day.csv
level_data/2021/06/09/day.csv
level_data/2021/06/10/day.csv
level_data/2021/06/11/day.csv
level_data/2021/06/12/day.csv
level_data/2021/06/13/day.csv
level_data/2021/06/14/day.csv
level_data/2021/06/15/day.csv
level_data/2021/06/16/day.csv
level_data/2021/06/17/day.csv
level_data/2021/06/18/day.csv
level_data/2021/06/19/day.csv
level_data/2021/06/20/day.csv
level_data/2021/06/21/day.csv
level_data/2021/06/22/day.csv
level_data/2021/06/23/day.csv
level_data/2021/06/24/day.csv
level_data/2021/06/25/day.csv
level_data/2021/06/26/day.csv
level_data/2021/06/27/day.csv
level_data/2021/06/28/day.csv
level_data/2021/06/29/day.csv
level_data/2021/06/30/day.csv
level_data/2021/06/month.csv
level_data/2021/07/01/day.csv
level_data/2021/07/02/day.csv
level_data/2021/07/03/day.csv
level_data/2021/07/04/day.csv
level_data/2021/07/05/day.csv
level_data/2021/07/06/day.csv
level_data/2021/07/07/day.csv
level_data/2021/07/08/day.csv
level_data/2021/07/09/day.csv
level_data/2021/07/10/day.csv
level_data/2021/07/11/day.csv
level_data/2021/07/12/day.csv
level_data/2021/07/13/day.csv
level_data/2021/07/14/day.csv
level_data/2021/07/15/day.csv
level_data/2021/07/16/day.csv
level_data/2021/07/17/day.csv
level_data/2021/07/18/day.csv
level_data/2021/07/19/day.csv
level_data/2021/07/20/day.csv
level_data/2021/07/21/day.csv
level_data/2021/07/22/day.csv
level_data/2021/07/23/day.csv
level_data/2021/07/24/day.csv
level_data/2021/07/25/day.csv
level_data/2021/07/26/day.csv
level_data/2021/07/27/day.csv
level_data/2021/07/28/day.csv
level_data/2021/07/29/day.csv
level_data/2021/07/30/day.csv
level_data/2021/07/month.csv
level_data/2021/08/01/day.csv
level_data/2021/08/02/day.csv
level_data/2021/08/03/day.csv
level_data/2021/08/04/day.csv
level_data/2021/08/05/day.csv
level_data/2021/08/06/day.csv
level_data/2021/08/07/day.csv
level_data/2021/08/08/day.csv
level_data/2021/08/09/day.csv
level_data/2021/08/10/day.csv
level_data/2021/08/11/day.csv
level_data/2021/08/12/day.csv
level_data/2021/08/13/day.csv
level_data/2021/08/14/day.csv
level_data/2021/08/15/day.csv
level_data/2021/08/16/day.csv
level_data/2021/08/17/day.csv
level_data/2021/08/18/day.csv
level_data/2021/08/19/day.csv
level_data/2021/08/20/day.csv
level_data/2021/08/21/day.csv
level_data/2021/08/22/day.csv
level_data/2021/08/23/day.csv
level_data/2021/08/24/day.csv
level_data/2021/08/25/day.csv
level_data/2021/08/26/day.csv
level_data/2021/08/27/day.csv
level_data/2021/08/28/day.csv
level_data/2021/08/29/day.csv
level_data/2021/08/30/day.csv
level_data/2021/08/month.csv
level_data/2021/09/01/day.csv
level_data/2021/09/02/day.csv
level_data/2021/09/03/day.csv
level_data/2021/09/04/day.csv
level_data/2021/09/05/day.csv
level_data/2021/09/06/day.csv
level_data/2021/09/07/day.csv
level_data/2021/09/08/day.csv
level_data/2021/09/09/day.csv
level_data/2021/09/10/day.csv
level_data/2021/09/11/day.csv
level_data/2021/09/12/day.csv
level_data/2021/09/13/day.csv
level_data/2021/09/14/day.csv
level_data/2021/09/15/day.csv
level_data/2021/09/16/day.csv
level_data/2021/09/17/day.csv
level_data/2021/09/18/day.csv
level_data/2021/09/19/day.csv
level_data/2021/09/20/day.csv
level_data/2021/09/21/day.csv
level_data/2021/09/22/day.csv
level_data/2021/09/23/day.csv
level_data/2021/09/24/day.csv
level_data/2021/09/25/day.csv
level_data/2021/09/26/day.csv
level_data/2021/09/27/day.csv
level_data/2021/09/28/day.csv
level_data/2021/09/29/day.csv
level_data/2021/09/30/day.csv
level_data/2021/09/month.csv
level_data/2021/10/01/day.csv
level_data/2021/10/02/day.csv
level_data/2021/10/03/day.csv
level_data/2021/10/04/day.csv
level_data/2021/10/05/day.csv
level_data/2021/10/06/day.csv
level_data/2021/10/07/day.csv
level_data/2021/10/08/day.csv
level_data/2021/10/09/day.csv
level_data/2021/10/10/day.csv
level_data/2021/10/11/day.csv
level_data/2021/10/12/day.csv
level_data/2021/10/13/day.csv
level_data/2021/10/14/day.csv
level_data/2021/10/15/day.csv
level_data/2021/10/16/day.csv
level_data/2021/10/17/day.csv
level_data/2021/10/18/day.csv
level_data/2021/10/19/day.csv
level_data/2021/10/20/day.csv
level_data/2021/10/21/day.csv
level_data/2021/10/22/day.csv
level_data/2021/10/23/day.csv
level_data/2021/10/24/day.csv
level_data/2021/10/25/day.csv
level_data/2021/10/26/day.csv
level_data/2021/10/27/day.csv
level_data/2021/10/28/day.csv
level_data/2021/10/29/day.csv
level_data/2021/10/30/day.csv
level_data/2021/10/month.csv
level_data/2021/11/01/day.csv
level_data/2021/11/02/day.csv
level_data/2021/11/03/day.csv
level_data/2021/11/04/day.csv
level_data/2021/11/05/day.csv
level_data/2021/11/06/day.csv
level_data/2021/11/07/day.csv
level_data/2021/11/08/day.csv
level_data/2021/11/09/day.csv
level_data/2021/11/10/day.csv
level_data/2021/11/11/day.csv
level_data/2021/11/12/day.csv
level_data/2021/11/13/day.csv
level_data/2021/11/14/day.csv
level_data/2021/11/15/day.csv
level_data/2021/11/16/day.csv
level_data/2021/11/17/day.csv
level_data/2021/11/18/day.csv
level_data/2021/11/19/day.csv
level_data/2021/11/20/day.csv
level_data/2021/11/21/day.csv
level_data/2021/11/22/day.csv
level_data/2021/11/23/day.csv
level_data/2021/11/24/day.csv
level_data/2021/11/25/day.csv
level_data/2021/11/26/day.csv
level_data/2021/11/27/day.csv
level_data/2021/11/28/day.csv
level_data/2021/11/29/day.csv
level_data/2021/11/30/day.csv
level_data/2021/11/month.csv
level_data/2021/12/01/day.csv
level_data/2021/12/02/day.csv
level_data/2021/12/03/day.csv
level_data/2021/12/04/day.csv
level_data/2021/12/05/day.csv
level_data/2021/12/06/day.csv
level_data/2021/12/07/day.csv
level_data/2021/12/08/day.csv
level_data/2021/12/09/day.csv
level_data/2021/12/10/day.csv
level_data/2021/12/11/day.csv
level_data/2021/12/12/day.csv
level_data/2021/12/13/day.csv
level_data/2021/12/14/day.csv
level_data/2021/12/15/day.csv
level_data/2021/12/16/day.csv
level_data/2021/12/17/day.csv
level_data/2021/12/18/day.csv
level_data/2021/12/19/day.csv
level_data/2021/12/20/day.csv
level_data/2021/12/21/day.csv
level_data/2021/12/22/day.csv
level_data/2021/12/23/day.csv
level_data/2021/12/24/day.csv
level_data/2021/12/25/day.csv
level_data/2021/12/26/day.csv
level_data/2021/12/27/day.csv
level_data/2021/12/28/day.csv
level_data/2021/12/29/day.csv
level_data/2021/12/30/day.csv
level_data/2021/12/month.csv
level_data/2021/year.csv 
```

**练习：**编写一个名为`walk_dfs`的`walk_gen`版本，它遍历给定的目录并产生它包含的文件，但它应该使用堆栈并以迭代方式运行，而不是递归方式。

您可以使用以下循环来测试您的代码。

```py
for path in walk_dfs(year_dir):
    print(path) 
```

```py
level_data/2021/year.csv
level_data/2021/12/month.csv
level_data/2021/12/30/day.csv
level_data/2021/12/29/day.csv
level_data/2021/12/28/day.csv
level_data/2021/12/27/day.csv
level_data/2021/12/26/day.csv
level_data/2021/12/25/day.csv
level_data/2021/12/24/day.csv
level_data/2021/12/23/day.csv
level_data/2021/12/22/day.csv
level_data/2021/12/21/day.csv
level_data/2021/12/20/day.csv
level_data/2021/12/19/day.csv
level_data/2021/12/18/day.csv
level_data/2021/12/17/day.csv
level_data/2021/12/16/day.csv
level_data/2021/12/15/day.csv
level_data/2021/12/14/day.csv
level_data/2021/12/13/day.csv
level_data/2021/12/12/day.csv
level_data/2021/12/11/day.csv
level_data/2021/12/10/day.csv
level_data/2021/12/09/day.csv
level_data/2021/12/08/day.csv
level_data/2021/12/07/day.csv
level_data/2021/12/06/day.csv
level_data/2021/12/05/day.csv
level_data/2021/12/04/day.csv
level_data/2021/12/03/day.csv
level_data/2021/12/02/day.csv
level_data/2021/12/01/day.csv
level_data/2021/11/month.csv
level_data/2021/11/30/day.csv
level_data/2021/11/29/day.csv
level_data/2021/11/28/day.csv
level_data/2021/11/27/day.csv
level_data/2021/11/26/day.csv
level_data/2021/11/25/day.csv
level_data/2021/11/24/day.csv
level_data/2021/11/23/day.csv
level_data/2021/11/22/day.csv
level_data/2021/11/21/day.csv
level_data/2021/11/20/day.csv
level_data/2021/11/19/day.csv
level_data/2021/11/18/day.csv
level_data/2021/11/17/day.csv
level_data/2021/11/16/day.csv
level_data/2021/11/15/day.csv
level_data/2021/11/14/day.csv
level_data/2021/11/13/day.csv
level_data/2021/11/12/day.csv
level_data/2021/11/11/day.csv
level_data/2021/11/10/day.csv
level_data/2021/11/09/day.csv
level_data/2021/11/08/day.csv
level_data/2021/11/07/day.csv
level_data/2021/11/06/day.csv
level_data/2021/11/05/day.csv
level_data/2021/11/04/day.csv
level_data/2021/11/03/day.csv
level_data/2021/11/02/day.csv
level_data/2021/11/01/day.csv
level_data/2021/10/month.csv
level_data/2021/10/30/day.csv
level_data/2021/10/29/day.csv
level_data/2021/10/28/day.csv
level_data/2021/10/27/day.csv
level_data/2021/10/26/day.csv
level_data/2021/10/25/day.csv
level_data/2021/10/24/day.csv
level_data/2021/10/23/day.csv
level_data/2021/10/22/day.csv
level_data/2021/10/21/day.csv
level_data/2021/10/20/day.csv
level_data/2021/10/19/day.csv
level_data/2021/10/18/day.csv
level_data/2021/10/17/day.csv
level_data/2021/10/16/day.csv
level_data/2021/10/15/day.csv
level_data/2021/10/14/day.csv
level_data/2021/10/13/day.csv
level_data/2021/10/12/day.csv
level_data/2021/10/11/day.csv
level_data/2021/10/10/day.csv
level_data/2021/10/09/day.csv
level_data/2021/10/08/day.csv
level_data/2021/10/07/day.csv
level_data/2021/10/06/day.csv
level_data/2021/10/05/day.csv
level_data/2021/10/04/day.csv
level_data/2021/10/03/day.csv
level_data/2021/10/02/day.csv
level_data/2021/10/01/day.csv
level_data/2021/09/month.csv
level_data/2021/09/30/day.csv
level_data/2021/09/29/day.csv
level_data/2021/09/28/day.csv
level_data/2021/09/27/day.csv
level_data/2021/09/26/day.csv
level_data/2021/09/25/day.csv
level_data/2021/09/24/day.csv
level_data/2021/09/23/day.csv
level_data/2021/09/22/day.csv
level_data/2021/09/21/day.csv
level_data/2021/09/20/day.csv
level_data/2021/09/19/day.csv
level_data/2021/09/18/day.csv
level_data/2021/09/17/day.csv
level_data/2021/09/16/day.csv
level_data/2021/09/15/day.csv
level_data/2021/09/14/day.csv
level_data/2021/09/13/day.csv
level_data/2021/09/12/day.csv
level_data/2021/09/11/day.csv
level_data/2021/09/10/day.csv
level_data/2021/09/09/day.csv
level_data/2021/09/08/day.csv
level_data/2021/09/07/day.csv
level_data/2021/09/06/day.csv
level_data/2021/09/05/day.csv
level_data/2021/09/04/day.csv
level_data/2021/09/03/day.csv
level_data/2021/09/02/day.csv
level_data/2021/09/01/day.csv
level_data/2021/08/month.csv
level_data/2021/08/30/day.csv
level_data/2021/08/29/day.csv
level_data/2021/08/28/day.csv
level_data/2021/08/27/day.csv
level_data/2021/08/26/day.csv
level_data/2021/08/25/day.csv
level_data/2021/08/24/day.csv
level_data/2021/08/23/day.csv
level_data/2021/08/22/day.csv
level_data/2021/08/21/day.csv
level_data/2021/08/20/day.csv
level_data/2021/08/19/day.csv
level_data/2021/08/18/day.csv
level_data/2021/08/17/day.csv
level_data/2021/08/16/day.csv
level_data/2021/08/15/day.csv
level_data/2021/08/14/day.csv
level_data/2021/08/13/day.csv
level_data/2021/08/12/day.csv
level_data/2021/08/11/day.csv
level_data/2021/08/10/day.csv
level_data/2021/08/09/day.csv
level_data/2021/08/08/day.csv
level_data/2021/08/07/day.csv
level_data/2021/08/06/day.csv
level_data/2021/08/05/day.csv
level_data/2021/08/04/day.csv
level_data/2021/08/03/day.csv
level_data/2021/08/02/day.csv
level_data/2021/08/01/day.csv
level_data/2021/07/month.csv
level_data/2021/07/30/day.csv
level_data/2021/07/29/day.csv
level_data/2021/07/28/day.csv
level_data/2021/07/27/day.csv
level_data/2021/07/26/day.csv
level_data/2021/07/25/day.csv
level_data/2021/07/24/day.csv
level_data/2021/07/23/day.csv
level_data/2021/07/22/day.csv
level_data/2021/07/21/day.csv
level_data/2021/07/20/day.csv
level_data/2021/07/19/day.csv
level_data/2021/07/18/day.csv
level_data/2021/07/17/day.csv
level_data/2021/07/16/day.csv
level_data/2021/07/15/day.csv
level_data/2021/07/14/day.csv
level_data/2021/07/13/day.csv
level_data/2021/07/12/day.csv
level_data/2021/07/11/day.csv
level_data/2021/07/10/day.csv
level_data/2021/07/09/day.csv
level_data/2021/07/08/day.csv
level_data/2021/07/07/day.csv
level_data/2021/07/06/day.csv
level_data/2021/07/05/day.csv
level_data/2021/07/04/day.csv
level_data/2021/07/03/day.csv
level_data/2021/07/02/day.csv
level_data/2021/07/01/day.csv
level_data/2021/06/month.csv
level_data/2021/06/30/day.csv
level_data/2021/06/29/day.csv
level_data/2021/06/28/day.csv
level_data/2021/06/27/day.csv
level_data/2021/06/26/day.csv
level_data/2021/06/25/day.csv
level_data/2021/06/24/day.csv
level_data/2021/06/23/day.csv
level_data/2021/06/22/day.csv
level_data/2021/06/21/day.csv
level_data/2021/06/20/day.csv
level_data/2021/06/19/day.csv
level_data/2021/06/18/day.csv
level_data/2021/06/17/day.csv
level_data/2021/06/16/day.csv
level_data/2021/06/15/day.csv
level_data/2021/06/14/day.csv
level_data/2021/06/13/day.csv
level_data/2021/06/12/day.csv
level_data/2021/06/11/day.csv
level_data/2021/06/10/day.csv
level_data/2021/06/09/day.csv
level_data/2021/06/08/day.csv
level_data/2021/06/07/day.csv
level_data/2021/06/06/day.csv
level_data/2021/06/05/day.csv
level_data/2021/06/04/day.csv
level_data/2021/06/03/day.csv
level_data/2021/06/02/day.csv
level_data/2021/06/01/day.csv
level_data/2021/05/month.csv
level_data/2021/05/30/day.csv
level_data/2021/05/29/day.csv
level_data/2021/05/28/day.csv
level_data/2021/05/27/day.csv
level_data/2021/05/26/day.csv
level_data/2021/05/25/day.csv
level_data/2021/05/24/day.csv
level_data/2021/05/23/day.csv
level_data/2021/05/22/day.csv
level_data/2021/05/21/day.csv
level_data/2021/05/20/day.csv
level_data/2021/05/19/day.csv
level_data/2021/05/18/day.csv
level_data/2021/05/17/day.csv
level_data/2021/05/16/day.csv
level_data/2021/05/15/day.csv
level_data/2021/05/14/day.csv
level_data/2021/05/13/day.csv
level_data/2021/05/12/day.csv
level_data/2021/05/11/day.csv
level_data/2021/05/10/day.csv
level_data/2021/05/09/day.csv
level_data/2021/05/08/day.csv
level_data/2021/05/07/day.csv
level_data/2021/05/06/day.csv
level_data/2021/05/05/day.csv
level_data/2021/05/04/day.csv
level_data/2021/05/03/day.csv
level_data/2021/05/02/day.csv
level_data/2021/05/01/day.csv
level_data/2021/04/month.csv
level_data/2021/04/30/day.csv
level_data/2021/04/29/day.csv
level_data/2021/04/28/day.csv
level_data/2021/04/27/day.csv
level_data/2021/04/26/day.csv
level_data/2021/04/25/day.csv
level_data/2021/04/24/day.csv
level_data/2021/04/23/day.csv
level_data/2021/04/22/day.csv
level_data/2021/04/21/day.csv
level_data/2021/04/20/day.csv
level_data/2021/04/19/day.csv
level_data/2021/04/18/day.csv
level_data/2021/04/17/day.csv
level_data/2021/04/16/day.csv
level_data/2021/04/15/day.csv
level_data/2021/04/14/day.csv
level_data/2021/04/13/day.csv
level_data/2021/04/12/day.csv
level_data/2021/04/11/day.csv
level_data/2021/04/10/day.csv
level_data/2021/04/09/day.csv
level_data/2021/04/08/day.csv
level_data/2021/04/07/day.csv
level_data/2021/04/06/day.csv
level_data/2021/04/05/day.csv
level_data/2021/04/04/day.csv
level_data/2021/04/03/day.csv
level_data/2021/04/02/day.csv
level_data/2021/04/01/day.csv
level_data/2021/03/month.csv
level_data/2021/03/30/day.csv
level_data/2021/03/29/day.csv
level_data/2021/03/28/day.csv
level_data/2021/03/27/day.csv
level_data/2021/03/26/day.csv
level_data/2021/03/25/day.csv
level_data/2021/03/24/day.csv
level_data/2021/03/23/day.csv
level_data/2021/03/22/day.csv
level_data/2021/03/21/day.csv
level_data/2021/03/20/day.csv
level_data/2021/03/19/day.csv
level_data/2021/03/18/day.csv
level_data/2021/03/17/day.csv
level_data/2021/03/16/day.csv
level_data/2021/03/15/day.csv
level_data/2021/03/14/day.csv
level_data/2021/03/13/day.csv
level_data/2021/03/12/day.csv
level_data/2021/03/11/day.csv
level_data/2021/03/10/day.csv
level_data/2021/03/09/day.csv
level_data/2021/03/08/day.csv
level_data/2021/03/07/day.csv
level_data/2021/03/06/day.csv
level_data/2021/03/05/day.csv
level_data/2021/03/04/day.csv
level_data/2021/03/03/day.csv
level_data/2021/03/02/day.csv
level_data/2021/03/01/day.csv
level_data/2021/02/month.csv
level_data/2021/02/30/day.csv
level_data/2021/02/29/day.csv
level_data/2021/02/28/day.csv
level_data/2021/02/27/day.csv
level_data/2021/02/26/day.csv
level_data/2021/02/25/day.csv
level_data/2021/02/24/day.csv
level_data/2021/02/23/day.csv
level_data/2021/02/22/day.csv
level_data/2021/02/21/day.csv
level_data/2021/02/20/day.csv
level_data/2021/02/19/day.csv
level_data/2021/02/18/day.csv
level_data/2021/02/17/day.csv
level_data/2021/02/16/day.csv
level_data/2021/02/15/day.csv
level_data/2021/02/14/day.csv
level_data/2021/02/13/day.csv
level_data/2021/02/12/day.csv
level_data/2021/02/11/day.csv
level_data/2021/02/10/day.csv
level_data/2021/02/09/day.csv
level_data/2021/02/08/day.csv
level_data/2021/02/07/day.csv
level_data/2021/02/06/day.csv
level_data/2021/02/05/day.csv
level_data/2021/02/04/day.csv
level_data/2021/02/03/day.csv
level_data/2021/02/02/day.csv
level_data/2021/02/01/day.csv
level_data/2021/01/month.csv
level_data/2021/01/30/day.csv
level_data/2021/01/29/day.csv
level_data/2021/01/28/day.csv
level_data/2021/01/27/day.csv
level_data/2021/01/26/day.csv
level_data/2021/01/25/day.csv
level_data/2021/01/24/day.csv
level_data/2021/01/23/day.csv
level_data/2021/01/22/day.csv
level_data/2021/01/21/day.csv
level_data/2021/01/20/day.csv
level_data/2021/01/19/day.csv
level_data/2021/01/18/day.csv
level_data/2021/01/17/day.csv
level_data/2021/01/16/day.csv
level_data/2021/01/15/day.csv
level_data/2021/01/14/day.csv
level_data/2021/01/13/day.csv
level_data/2021/01/12/day.csv
level_data/2021/01/11/day.csv
level_data/2021/01/10/day.csv
level_data/2021/01/09/day.csv
level_data/2021/01/08/day.csv
level_data/2021/01/07/day.csv
level_data/2021/01/06/day.csv
level_data/2021/01/05/day.csv
level_data/2021/01/04/day.csv
level_data/2021/01/03/day.csv
level_data/2021/01/02/day.csv
level_data/2021/01/01/day.csv
level_data/2021/00/month.csv
level_data/2021/00/30/day.csv
level_data/2021/00/29/day.csv
level_data/2021/00/28/day.csv
level_data/2021/00/27/day.csv
level_data/2021/00/26/day.csv
level_data/2021/00/25/day.csv
level_data/2021/00/24/day.csv
level_data/2021/00/23/day.csv
level_data/2021/00/22/day.csv
level_data/2021/00/21/day.csv
level_data/2021/00/20/day.csv
level_data/2021/00/19/day.csv
level_data/2021/00/18/day.csv
level_data/2021/00/17/day.csv
level_data/2021/00/16/day.csv
level_data/2021/00/15/day.csv
level_data/2021/00/14/day.csv
level_data/2021/00/13/day.csv
level_data/2021/00/12/day.csv
level_data/2021/00/11/day.csv
level_data/2021/00/10/day.csv
level_data/2021/00/09/day.csv
level_data/2021/00/08/day.csv
level_data/2021/00/07/day.csv
level_data/2021/00/06/day.csv
level_data/2021/00/05/day.csv
level_data/2021/00/04/day.csv
level_data/2021/00/03/day.csv
level_data/2021/00/02/day.csv
level_data/2021/00/01/day.csv 
```

请注意，文件被发现的顺序是“深度优先”的。例如，它在任何第二个月的文件之前产生了第一个月的所有文件。

另一种选择是按级别顺序遍历，它首先产生第一级（年度摘要）的所有文件，然后产生第二级（月度摘要）的所有文件，然后是第三级的文件。

要实现按级别顺序遍历，我们可以对`walk_dfs`进行最小的更改：用 FIFO 队列替换堆栈。为了有效地实现队列，我们可以使用`collections.deque`。

**练习：**编写一个名为`walk_level`的生成器函数，它接受一个目录并按级别顺序产生其文件。

使用以下循环来测试您的代码。

```py
for path in walk_level(year_dir):
    print(path) 
```

```py
level_data/2021/year.csv
level_data/2021/00/month.csv
level_data/2021/01/month.csv
level_data/2021/02/month.csv
level_data/2021/03/month.csv
level_data/2021/04/month.csv
level_data/2021/05/month.csv
level_data/2021/06/month.csv
level_data/2021/07/month.csv
level_data/2021/08/month.csv
level_data/2021/09/month.csv
level_data/2021/10/month.csv
level_data/2021/11/month.csv
level_data/2021/12/month.csv
level_data/2021/00/01/day.csv
level_data/2021/00/02/day.csv
level_data/2021/00/03/day.csv
level_data/2021/00/04/day.csv
level_data/2021/00/05/day.csv
level_data/2021/00/06/day.csv
level_data/2021/00/07/day.csv
level_data/2021/00/08/day.csv
level_data/2021/00/09/day.csv
level_data/2021/00/10/day.csv
level_data/2021/00/11/day.csv
level_data/2021/00/12/day.csv
level_data/2021/00/13/day.csv
level_data/2021/00/14/day.csv
level_data/2021/00/15/day.csv
level_data/2021/00/16/day.csv
level_data/2021/00/17/day.csv
level_data/2021/00/18/day.csv
level_data/2021/00/19/day.csv
level_data/2021/00/20/day.csv
level_data/2021/00/21/day.csv
level_data/2021/00/22/day.csv
level_data/2021/00/23/day.csv
level_data/2021/00/24/day.csv
level_data/2021/00/25/day.csv
level_data/2021/00/26/day.csv
level_data/2021/00/27/day.csv
level_data/2021/00/28/day.csv
level_data/2021/00/29/day.csv
level_data/2021/00/30/day.csv
level_data/2021/01/01/day.csv
level_data/2021/01/02/day.csv
level_data/2021/01/03/day.csv
level_data/2021/01/04/day.csv
level_data/2021/01/05/day.csv
level_data/2021/01/06/day.csv
level_data/2021/01/07/day.csv
level_data/2021/01/08/day.csv
level_data/2021/01/09/day.csv
level_data/2021/01/10/day.csv
level_data/2021/01/11/day.csv
level_data/2021/01/12/day.csv
level_data/2021/01/13/day.csv
level_data/2021/01/14/day.csv
level_data/2021/01/15/day.csv
level_data/2021/01/16/day.csv
level_data/2021/01/17/day.csv
level_data/2021/01/18/day.csv
level_data/2021/01/19/day.csv
level_data/2021/01/20/day.csv
level_data/2021/01/21/day.csv
level_data/2021/01/22/day.csv
level_data/2021/01/23/day.csv
level_data/2021/01/24/day.csv
level_data/2021/01/25/day.csv
level_data/2021/01/26/day.csv
level_data/2021/01/27/day.csv
level_data/2021/01/28/day.csv
level_data/2021/01/29/day.csv
level_data/2021/01/30/day.csv
level_data/2021/02/01/day.csv
level_data/2021/02/02/day.csv
level_data/2021/02/03/day.csv
level_data/2021/02/04/day.csv
level_data/2021/02/05/day.csv
level_data/2021/02/06/day.csv
level_data/2021/02/07/day.csv
level_data/2021/02/08/day.csv
level_data/2021/02/09/day.csv
level_data/2021/02/10/day.csv
level_data/2021/02/11/day.csv
level_data/2021/02/12/day.csv
level_data/2021/02/13/day.csv
level_data/2021/02/14/day.csv
level_data/2021/02/15/day.csv
level_data/2021/02/16/day.csv
level_data/2021/02/17/day.csv
level_data/2021/02/18/day.csv
level_data/2021/02/19/day.csv
level_data/2021/02/20/day.csv
level_data/2021/02/21/day.csv
level_data/2021/02/22/day.csv
level_data/2021/02/23/day.csv
level_data/2021/02/24/day.csv
level_data/2021/02/25/day.csv
level_data/2021/02/26/day.csv
level_data/2021/02/27/day.csv
level_data/2021/02/28/day.csv
level_data/2021/02/29/day.csv
level_data/2021/02/30/day.csv
level_data/2021/03/01/day.csv
level_data/2021/03/02/day.csv
level_data/2021/03/03/day.csv
level_data/2021/03/04/day.csv
level_data/2021/03/05/day.csv
level_data/2021/03/06/day.csv
level_data/2021/03/07/day.csv
level_data/2021/03/08/day.csv
level_data/2021/03/09/day.csv
level_data/2021/03/10/day.csv
level_data/2021/03/11/day.csv
level_data/2021/03/12/day.csv
level_data/2021/03/13/day.csv
level_data/2021/03/14/day.csv
level_data/2021/03/15/day.csv
level_data/2021/03/16/day.csv
level_data/2021/03/17/day.csv
level_data/2021/03/18/day.csv
level_data/2021/03/19/day.csv
level_data/2021/03/20/day.csv
level_data/2021/03/21/day.csv
level_data/2021/03/22/day.csv
level_data/2021/03/23/day.csv
level_data/2021/03/24/day.csv
level_data/2021/03/25/day.csv
level_data/2021/03/26/day.csv
level_data/2021/03/27/day.csv
level_data/2021/03/28/day.csv
level_data/2021/03/29/day.csv
level_data/2021/03/30/day.csv
level_data/2021/04/01/day.csv
level_data/2021/04/02/day.csv
level_data/2021/04/03/day.csv
level_data/2021/04/04/day.csv
level_data/2021/04/05/day.csv
level_data/2021/04/06/day.csv
level_data/2021/04/07/day.csv
level_data/2021/04/08/day.csv
level_data/2021/04/09/day.csv
level_data/2021/04/10/day.csv
level_data/2021/04/11/day.csv
level_data/2021/04/12/day.csv
level_data/2021/04/13/day.csv
level_data/2021/04/14/day.csv
level_data/2021/04/15/day.csv
level_data/2021/04/16/day.csv
level_data/2021/04/17/day.csv
level_data/2021/04/18/day.csv
level_data/2021/04/19/day.csv
level_data/2021/04/20/day.csv
level_data/2021/04/21/day.csv
level_data/2021/04/22/day.csv
level_data/2021/04/23/day.csv
level_data/2021/04/24/day.csv
level_data/2021/04/25/day.csv
level_data/2021/04/26/day.csv
level_data/2021/04/27/day.csv
level_data/2021/04/28/day.csv
level_data/2021/04/29/day.csv
level_data/2021/04/30/day.csv
level_data/2021/05/01/day.csv
level_data/2021/05/02/day.csv
level_data/2021/05/03/day.csv
level_data/2021/05/04/day.csv
level_data/2021/05/05/day.csv
level_data/2021/05/06/day.csv
level_data/2021/05/07/day.csv
level_data/2021/05/08/day.csv
level_data/2021/05/09/day.csv
level_data/2021/05/10/day.csv
level_data/2021/05/11/day.csv
level_data/2021/05/12/day.csv
level_data/2021/05/13/day.csv
level_data/2021/05/14/day.csv
level_data/2021/05/15/day.csv
level_data/2021/05/16/day.csv
level_data/2021/05/17/day.csv
level_data/2021/05/18/day.csv
level_data/2021/05/19/day.csv
level_data/2021/05/20/day.csv
level_data/2021/05/21/day.csv
level_data/2021/05/22/day.csv
level_data/2021/05/23/day.csv
level_data/2021/05/24/day.csv
level_data/2021/05/25/day.csv
level_data/2021/05/26/day.csv
level_data/2021/05/27/day.csv
level_data/2021/05/28/day.csv
level_data/2021/05/29/day.csv
level_data/2021/05/30/day.csv
level_data/2021/06/01/day.csv
level_data/2021/06/02/day.csv
level_data/2021/06/03/day.csv
level_data/2021/06/04/day.csv
level_data/2021/06/05/day.csv
level_data/2021/06/06/day.csv
level_data/2021/06/07/day.csv
level_data/2021/06/08/day.csv
level_data/2021/06/09/day.csv
level_data/2021/06/10/day.csv
level_data/2021/06/11/day.csv
level_data/2021/06/12/day.csv
level_data/2021/06/13/day.csv
level_data/2021/06/14/day.csv
level_data/2021/06/15/day.csv
level_data/2021/06/16/day.csv
level_data/2021/06/17/day.csv
level_data/2021/06/18/day.csv
level_data/2021/06/19/day.csv
level_data/2021/06/20/day.csv
level_data/2021/06/21/day.csv
level_data/2021/06/22/day.csv
level_data/2021/06/23/day.csv
level_data/2021/06/24/day.csv
level_data/2021/06/25/day.csv
level_data/2021/06/26/day.csv
level_data/2021/06/27/day.csv
level_data/2021/06/28/day.csv
level_data/2021/06/29/day.csv
level_data/2021/06/30/day.csv
level_data/2021/07/01/day.csv
level_data/2021/07/02/day.csv
level_data/2021/07/03/day.csv
level_data/2021/07/04/day.csv
level_data/2021/07/05/day.csv
level_data/2021/07/06/day.csv
level_data/2021/07/07/day.csv
level_data/2021/07/08/day.csv
level_data/2021/07/09/day.csv
level_data/2021/07/10/day.csv
level_data/2021/07/11/day.csv
level_data/2021/07/12/day.csv
level_data/2021/07/13/day.csv
level_data/2021/07/14/day.csv
level_data/2021/07/15/day.csv
level_data/2021/07/16/day.csv
level_data/2021/07/17/day.csv
level_data/2021/07/18/day.csv
level_data/2021/07/19/day.csv
level_data/2021/07/20/day.csv
level_data/2021/07/21/day.csv
level_data/2021/07/22/day.csv
level_data/2021/07/23/day.csv
level_data/2021/07/24/day.csv
level_data/2021/07/25/day.csv
level_data/2021/07/26/day.csv
level_data/2021/07/27/day.csv
level_data/2021/07/28/day.csv
level_data/2021/07/29/day.csv
level_data/2021/07/30/day.csv
level_data/2021/08/01/day.csv
level_data/2021/08/02/day.csv
level_data/2021/08/03/day.csv
level_data/2021/08/04/day.csv
level_data/2021/08/05/day.csv
level_data/2021/08/06/day.csv
level_data/2021/08/07/day.csv
level_data/2021/08/08/day.csv
level_data/2021/08/09/day.csv
level_data/2021/08/10/day.csv
level_data/2021/08/11/day.csv
level_data/2021/08/12/day.csv
level_data/2021/08/13/day.csv
level_data/2021/08/14/day.csv
level_data/2021/08/15/day.csv
level_data/2021/08/16/day.csv
level_data/2021/08/17/day.csv
level_data/2021/08/18/day.csv
level_data/2021/08/19/day.csv
level_data/2021/08/20/day.csv
level_data/2021/08/21/day.csv
level_data/2021/08/22/day.csv
level_data/2021/08/23/day.csv
level_data/2021/08/24/day.csv
level_data/2021/08/25/day.csv
level_data/2021/08/26/day.csv
level_data/2021/08/27/day.csv
level_data/2021/08/28/day.csv
level_data/2021/08/29/day.csv
level_data/2021/08/30/day.csv
level_data/2021/09/01/day.csv
level_data/2021/09/02/day.csv
level_data/2021/09/03/day.csv
level_data/2021/09/04/day.csv
level_data/2021/09/05/day.csv
level_data/2021/09/06/day.csv
level_data/2021/09/07/day.csv
level_data/2021/09/08/day.csv
level_data/2021/09/09/day.csv
level_data/2021/09/10/day.csv
level_data/2021/09/11/day.csv
level_data/2021/09/12/day.csv
level_data/2021/09/13/day.csv
level_data/2021/09/14/day.csv
level_data/2021/09/15/day.csv
level_data/2021/09/16/day.csv
level_data/2021/09/17/day.csv
level_data/2021/09/18/day.csv
level_data/2021/09/19/day.csv
level_data/2021/09/20/day.csv
level_data/2021/09/21/day.csv
level_data/2021/09/22/day.csv
level_data/2021/09/23/day.csv
level_data/2021/09/24/day.csv
level_data/2021/09/25/day.csv
level_data/2021/09/26/day.csv
level_data/2021/09/27/day.csv
level_data/2021/09/28/day.csv
level_data/2021/09/29/day.csv
level_data/2021/09/30/day.csv
level_data/2021/10/01/day.csv
level_data/2021/10/02/day.csv
level_data/2021/10/03/day.csv 
```

```py
level_data/2021/10/04/day.csv
level_data/2021/10/05/day.csv
level_data/2021/10/06/day.csv
level_data/2021/10/07/day.csv
level_data/2021/10/08/day.csv
level_data/2021/10/09/day.csv
level_data/2021/10/10/day.csv
level_data/2021/10/11/day.csv
level_data/2021/10/12/day.csv
level_data/2021/10/13/day.csv
level_data/2021/10/14/day.csv
level_data/2021/10/15/day.csv
level_data/2021/10/16/day.csv
level_data/2021/10/17/day.csv
level_data/2021/10/18/day.csv
level_data/2021/10/19/day.csv
level_data/2021/10/20/day.csv
level_data/2021/10/21/day.csv
level_data/2021/10/22/day.csv
level_data/2021/10/23/day.csv
level_data/2021/10/24/day.csv
level_data/2021/10/25/day.csv
level_data/2021/10/26/day.csv
level_data/2021/10/27/day.csv
level_data/2021/10/28/day.csv
level_data/2021/10/29/day.csv
level_data/2021/10/30/day.csv
level_data/2021/11/01/day.csv
level_data/2021/11/02/day.csv
level_data/2021/11/03/day.csv
level_data/2021/11/04/day.csv
level_data/2021/11/05/day.csv
level_data/2021/11/06/day.csv
level_data/2021/11/07/day.csv
level_data/2021/11/08/day.csv
level_data/2021/11/09/day.csv
level_data/2021/11/10/day.csv
level_data/2021/11/11/day.csv
level_data/2021/11/12/day.csv
level_data/2021/11/13/day.csv
level_data/2021/11/14/day.csv
level_data/2021/11/15/day.csv
level_data/2021/11/16/day.csv
level_data/2021/11/17/day.csv
level_data/2021/11/18/day.csv
level_data/2021/11/19/day.csv
level_data/2021/11/20/day.csv
level_data/2021/11/21/day.csv
level_data/2021/11/22/day.csv
level_data/2021/11/23/day.csv
level_data/2021/11/24/day.csv
level_data/2021/11/25/day.csv
level_data/2021/11/26/day.csv
level_data/2021/11/27/day.csv
level_data/2021/11/28/day.csv
level_data/2021/11/29/day.csv
level_data/2021/11/30/day.csv
level_data/2021/12/01/day.csv
level_data/2021/12/02/day.csv
level_data/2021/12/03/day.csv
level_data/2021/12/04/day.csv
level_data/2021/12/05/day.csv
level_data/2021/12/06/day.csv
level_data/2021/12/07/day.csv
level_data/2021/12/08/day.csv
level_data/2021/12/09/day.csv
level_data/2021/12/10/day.csv
level_data/2021/12/11/day.csv
level_data/2021/12/12/day.csv
level_data/2021/12/13/day.csv
level_data/2021/12/14/day.csv
level_data/2021/12/15/day.csv
level_data/2021/12/16/day.csv
level_data/2021/12/17/day.csv
level_data/2021/12/18/day.csv
level_data/2021/12/19/day.csv
level_data/2021/12/20/day.csv
level_data/2021/12/21/day.csv
level_data/2021/12/22/day.csv
level_data/2021/12/23/day.csv
level_data/2021/12/24/day.csv
level_data/2021/12/25/day.csv
level_data/2021/12/26/day.csv
level_data/2021/12/27/day.csv
level_data/2021/12/28/day.csv
level_data/2021/12/29/day.csv
level_data/2021/12/30/day.csv 
```

如果您在大型文件系统中寻找文件，如果您认为文件更可能靠近根目录而不是深入嵌套的子目录中，按级别顺序搜索可能会有用。

*Python 中的数据结构和信息检索*

版权所有 2021 Allen Downey

许可证：[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
