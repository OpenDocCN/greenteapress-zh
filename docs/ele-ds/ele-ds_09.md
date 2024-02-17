# 数据框和系列

> [`allendowney.github.io/ElementsOfDataScience/07_dataframes.html`](https://allendowney.github.io/ElementsOfDataScience/07_dataframes.html)

[单击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ElementsOfDataScience/blob/master/07_dataframes.ipynb)或[单击此处下载](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/07_dataframes.ipynb)。

本章介绍了 Pandas，这是一个提供读取和写入数据文件、探索和分析数据以及生成可视化的 Python 库。它还提供了两种用于处理数据的新类型，`DataFrame`和`Series`。

我们将使用这些工具来回答一个数据问题：美国婴儿的平均出生体重是多少？这个例子将演示几乎任何数据科学项目中的重要步骤：

1.  确定可以回答问题的数据。

1.  在 Python 中获取数据并加载数据。

1.  检查数据并处理错误。

1.  从数据中选择相关的子集。

1.  使用直方图来可视化值的分布。

1.  使用摘要统计数据描述数据以最佳方式回答问题。

1.  考虑我们结论中可能的错误来源和限制。

让我们开始获取数据。

## 读取数据

我们将使用来自国家家庭增长调查（NSFG）的数据，该数据可从国家卫生统计中心的[`www.cdc.gov/nchs/nsfg/index.htm`](https://www.cdc.gov/nchs/nsfg/index.htm)获取。

要下载数据，您必须同意[`www.cdc.gov/nchs/data_access/ftp_dua.htm`](https://www.cdc.gov/nchs/data_access/ftp_dua.htm)上的数据使用协议。您应该仔细阅读这些条款，但让我提醒您我认为最重要的一点：

> 不要试图了解数据中包含的任何个人或机构的身份。

NSFG 受访者诚实地回答了最私人性质的问题，期望他们的身份不会被揭示。作为道德数据科学家，我们应该尊重他们的隐私并遵守使用条款。

NSFG 的受访者提供有关自己的一般信息，这些信息存储在受访者文件中，以及有关他们每次怀孕的信息，这些信息存储在怀孕文件中。

我们将使用怀孕文件，其中每次怀孕都包含一行，每个变量都有 248 个。每个变量代表对 NSFG 问卷上的问题的回答。

数据以固定宽度格式存储，这意味着每一行的长度相同，每个变量跨越固定的字符范围（参见[`www.ibm.com/docs/en/baw/19.x?topic=formats-fixed-width-format`](https://www.ibm.com/docs/en/baw/19.x?topic=formats-fixed-width-format)）。例如，每行的前六个字符代表一个名为`CASEID`的变量，它是每个受访者的唯一标识符；接下来的两个字符代表`PREGORDR`，它表示受访者的怀孕次数。

要读取这些数据，我们需要一个**数据字典**，它指定变量的名称和每个变量出现的字符范围。数据和数据字典可在单独的文件中找到。

```py
dict_file = '2015_2017_FemPregSetup.dct'
data_file = '2015_2017_FemPregData.dat' 
```

Pandas 可以读取大多数常见格式的数据，包括 CSV、Excel 和固定宽度格式，但它无法读取 Stata 格式的数据字典。为此，我们将使用一个名为`statadict`的 Python 库。

从`statadict`中，我们将导入`parse_stata_dict`，它读取数据字典。

```py
from statadict import parse_stata_dict

stata_dict = parse_stata_dict(dict_file)
stata_dict 
```

```py
<statadict.base.StataDict at 0x7f26b4428e50> 
```

结果是一个包含数据的对象

+   `names`，这是一个变量名列表，和

+   `colspecs`，这是一个元组列表。

`colspecs`中的每个元组指定变量出现的第一列和最后一列。

这些值正是我们需要使用`read_fwf`的参数，它是 Pandas 函数，用于读取固定宽度格式的文件。

```py
import pandas as pd

nsfg = pd.read_fwf(data_file, 
                   names=stata_dict.names, 
                   colspecs=stata_dict.colspecs)
type(nsfg) 
```

```py
pandas.core.frame.DataFrame 
```

`read_fwf()`的结果是一个`DataFrame`，这是 Pandas 用于存储数据的主要类型。`DataFrame`有一个名为`head()`的方法，显示前 5 行：

```py
nsfg.head() 
```

|  | CASEID | PREGORDR | HOWPREG_N | HOWPREG_P | MOSCURRP | NOWPRGDK | PREGEND1 | PREGEND2 | HOWENDDK | NBRNALIV | ... | SECU | SEST | CMINTVW | CMLSTYR | CMJAN3YR | CMJAN4YR | CMJAN5YR | QUARTER | PHASE | INTVWYEAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 70627 | 1 | NaN | NaN | NaN | NaN | 6.0 | NaN | NaN | 1.0 | ... | 3 | 322 | 1394 | 1382 | 1357 | 1345 | 1333 | 18 | 1 | 2016 |
| 1 | 70627 | 2 | NaN | NaN | NaN | NaN | 1.0 | NaN | NaN | NaN | ... | 3 | 322 | 1394 | 1382 | 1357 | 1345 | 1333 | 18 | 1 | 2016 |
| 2 | 70627 | 3 | NaN | NaN | NaN | NaN | 6.0 | NaN | NaN | 1.0 | ... | 3 | 322 | 1394 | 1382 | 1357 | 1345 | 1333 | 18 | 1 | 2016 |
| 3 | 70628 | 1 | NaN | NaN | NaN | NaN | 6.0 | NaN | NaN | 1.0 | ... | 2 | 366 | 1409 | 1397 | 1369 | 1357 | 1345 | 23 | 1 | 2017 |
| 4 | 70628 | 2 | NaN | NaN | NaN | NaN | 6.0 | NaN | NaN | 1.0 | ... | 2 | 366 | 1409 | 1397 | 1369 | 1357 | 1345 | 23 | 1 | 2017 |

5 行×248 列

```py
# NOTE: For the printed version of the book, 
# I'm using iloc to show
# the first 5 rows and first 9 columns 
nsfg.iloc[:5,:9] 
```

|  | CASEID | PREGORDR | HOWPREG_N | HOWPREG_P | MOSCURRP | NOWPRGDK | PREGEND1 | PREGEND2 | HOWENDDK |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 70627 | 1 | NaN | NaN | NaN | NaN | 6.0 | NaN | NaN |
| 1 | 70627 | 2 | NaN | NaN | NaN | NaN | 1.0 | NaN | NaN |
| 2 | 70627 | 3 | NaN | NaN | NaN | NaN | 6.0 | NaN | NaN |
| 3 | 70628 | 1 | NaN | NaN | NaN | NaN | 6.0 | NaN | NaN |
| 4 | 70628 | 2 | NaN | NaN | NaN | NaN | 6.0 | NaN | NaN |

前两列是`CASEID`和`PREGORDR`，我之前提到过。前三行具有相同的`CASEID`，因此此受访者报告了三次怀孕；`PREGORDR`的值表明它们是按顺序的第一、第二和第三次怀孕。

随着我们的学习，我们将了解更多关于其他变量的信息。

除了像`head`这样的方法，`Dataframe`对象还有几个**属性**，这些属性是与对象相关联的变量。例如，`nsfg`有一个名为`shape`的属性，它是一个包含行数和列数的元组：

```py
nsfg.shape 
```

```py
(9553, 248) 
```

此数据集中有 9553 行，每次怀孕一行，以及 248 列，每个变量一列。

`nsfg`还有一个名为`columns`的属性，其中包含列名：

```py
nsfg.columns 
```

```py
Index(['CASEID', 'PREGORDR', 'HOWPREG_N', 'HOWPREG_P', 'MOSCURRP', 'NOWPRGDK',
       'PREGEND1', 'PREGEND2', 'HOWENDDK', 'NBRNALIV',
       ...
       'SECU', 'SEST', 'CMINTVW', 'CMLSTYR', 'CMJAN3YR', 'CMJAN4YR',
       'CMJAN5YR', 'QUARTER', 'PHASE', 'INTVWYEAR'],
      dtype='object', length=248) 
```

列名存储在`Index`中，这是 Pandas 的另一种类型，类似于列表。

根据列名，您可能能够猜出一些变量是什么，但通常您需要阅读文档。

当您使用 NSFG 等数据集时，重要的是要仔细阅读文档。如果您错误地解释了一个变量，就会产生无意义的结果，而您却从未意识到。因此，在我们开始查看数据之前，让我们先熟悉一下 NSFG 代码簿，其中描述了每个变量。您可以从[`github.com/AllenDowney/ElementsOfDataScience/raw/master/data/2015-2017_NSFG_FemPregFile_Codebook-508.pdf`](https://github.com/AllenDowney/ElementsOfDataScience/raw/master/data/2015-2017_NSFG_FemPregFile_Codebook-508.pdf)下载此数据集的代码簿。

如果您在文档中搜索“出生体重”，您应该会找到与出生体重相关的这些变量。

+   `BIRTHWGT_LB1`：第一胎出生时的体重（磅）

+   `BIRTHWGT_OZ1`：第一胎出生时的体重（盎司）

在双胞胎或三胞胎的情况下，还有类似的变量用于第 2 个或第 3 个宝宝。现在我们将专注于每次怀孕的第一个宝宝，并且我们将回到多胞胎的问题。

## 系列

在许多方面，`DataFrame`类似于 Python 字典，其中列名是键，列是值。您可以使用方括号运算符从`DataFrame`中选择列，键是字符串。 

```py
pounds = nsfg['BIRTHWGT_LB1']
type(pounds) 
```

```py
pandas.core.series.Series 
```

结果是一个`Series`，它是 Pandas 类型，表示单列数据。在这种情况下，`Series`包含每个活产的出生体重（以磅为单位）。

`head`显示了`Series`中的前五个值，`Series`的名称和数据类型：

```py
pounds.head() 
```

```py
0    7.0
1    NaN
2    9.0
3    6.0
4    7.0
Name: BIRTHWGT_LB1, dtype: float64 
```

其中一个值是`NaN`，代表“不是一个数字”。`NaN`是一个特殊值，用于指示无效或缺失的数据。在这个例子中，怀孕没有以活产结束，所以出生体重是不适用的。

**练习：** 变量`BIRTHWGT_OZ1`包含出生体重的盎司部分。

从`nsfg` `DataFrame`中选择列`'BIRTHWGT_OZ1'`，并将其赋值给一个名为`ounces`的新变量。然后显示`ounces`的前 5 个元素。

**练习：** 到目前为止，我们看到的 Pandas 类型有`DataFrame`、`Index`和`Series`。您可以在以下找到这些类型的文档：

+   `DataFrame`：[`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

+   `Index`: [`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.html`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.html)

+   `Series`：[`pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)

这份文档可能会让人不知所措；我不建议现在就试图阅读全部。但您可能想略读一下，这样您就知道以后要查找的地方。

## 验证

此时，我们已经确定了回答问题所需的列，并将它们分配给名为`pounds`和`ounces`的变量。

```py
pounds = nsfg['BIRTHWGT_LB1']
ounces = nsfg['BIRTHWGT_OZ1'] 
```

在我们对这些数据进行任何操作之前，我们必须对其进行验证。验证的一部分是确认我们是否正确地解释了数据。

我们可以使用`value_counts`方法查看`pounds`中出现的值以及每个值出现的次数。

```py
pounds.value_counts() 
```

默认情况下，结果按最频繁的值排序，但我们可以使用`sort_index`按值进行排序，最轻的婴儿排在最前面，最重的婴儿排在最后面。

```py
pounds.value_counts().sort_index() 
```

```py
0.0        2
1.0       28
2.0       46
3.0       76
4.0      179
5.0      570
6.0     1644
7.0     2268
8.0     1287
9.0      396
10.0      82
11.0      17
12.0       2
13.0       1
14.0       1
98.0       2
99.0      89
Name: BIRTHWGT_LB1, dtype: int64 
```

正如我们所预期的那样，最频繁的值是 6-8 磅，但也有一些非常轻的婴儿，一些非常重的婴儿，以及两个特殊值，98 和 99。根据代码簿，这些值表示受访者拒绝回答问题（98）或不知道（99）。

我们可以通过将结果与代码簿进行比较来验证结果，代码簿列出了数值及其频率。

| value | label | 总数 |
| --- | --- | --- |
| . | 不适用 | 2863 |
| 0-5 | 6 磅以下 | 901 |
| 6 | 6 磅 | 1644 |
| 7 | 7 磅 | 2268 |
| 8 | 8 磅 | 1287 |
| 9-95 | 9 磅或更多 | 499 |
| 98 | 拒绝 | 2 |
| 99 | 不知道 | 89 |
|  | 总数 | 9553 |

`value_counts`的结果与代码簿一致，因此我们有一些信心，我们正在正确地阅读和解释数据。

**练习：** 在`nsfg` `DataFrame`中，列`'OUTCOME'`编码了每次怀孕的结果，如下所示：

| 值 | 含义 |
| --- | --- |
| 1 | 活产 |
| 2 | 人工流产 |
| 3 | 死胎 |
| 4 | 流产 |
| 5 | 异位妊娠 |
| 6 | 当前怀孕 |

使用`value_counts`来显示此列中的值以及每个值出现的次数。结果是否与代码簿一致？

## 总结统计

另一种验证数据的方法是使用`describe`，它计算总结数据的统计数据，如均值、标准差、最小值和最大值。

以下是`pounds`的结果。

```py
pounds.describe() 
```

```py
count    6690.000000
mean        8.008819
std        10.771360
min         0.000000
25%         6.000000
50%         7.000000
75%         8.000000
max        99.000000
Name: BIRTHWGT_LB1, dtype: float64 
```

`count`是数值的数量，不包括`NaN`。对于这个变量，有 6690 个数值不是`NaN`。

`mean`和`std`是均值和标准差。`min`和`max`是最小值和最大值，中间是 25th、50th 和 75th 百分位数。50th 百分位数是中位数。

平均值约为`8.05`，但这并不重要，因为它包括特殊值 98 和 99。在我们真正计算平均值之前，我们必须用`NaN`替换这些值以识别它们为缺失数据。

`replace()`方法做了我们想要的事情：

```py
import numpy as np

pounds_clean = pounds.replace([98, 99], np.nan) 
```

`replace`接受一个我们想要替换的值的列表和我们想要替换它们的值。`np.nan`表示我们从 NumPy 库中获得特殊值`NaN`，它被导入为`np`。

`replace()`的结果是一个新的`Series`，我将其赋给`pounds_clean`。如果我们再次运行`describe`，我们会看到`count`现在更小了，因为它只包括有效值。

```py
pounds_clean.describe() 
```

```py
count    6599.000000
mean        6.754357
std         1.383268
min         0.000000
25%         6.000000
50%         7.000000
75%         8.000000
max        14.000000
Name: BIRTHWGT_LB1, dtype: float64 
```

新`Series`的平均值约为 6.7 磅。请记住，原始`Series`的平均值超过了 8 磅。当你移除几个重达 99 磅的婴儿时，这会产生很大的差异！

**练习：**使用`describe`来总结`ounces`。

然后使用`replace`将特殊值 98 和 99 替换为 NaN，并将结果赋给`ounces_clean`。再次运行`describe`。这种清理对结果有多大影响？

## 系列算术

现在我们想要将`pounds`和`ounces`组合成一个包含总出生体重的单个`Series`。算术运算符可以用于`Series`对象；因此，例如，要将`pounds`转换为 ounces，我们可以这样写：

`pounds * 16`

然后我们可以这样添加`ounces`

`pounds * 16 + ounces`

**练习：**使用`pounds_clean`和`ounces_clean`来计算以千克表示的总出生体重（大约每千克有 2.2 磅）。平均出生体重是多少？

**练习：**对于 NSFG 数据集中的每次怀孕，变量`'AGECON'`编码了受访者受孕时的年龄，`'AGEPREG'`编码了受访者怀孕结束时的年龄。

+   阅读这些变量的文档。我们是否有任何特殊值需要处理？

+   选择`'AGECON'`和`'AGEPREG'`，并将它们分配给名为`agecon`和`agepreg`的变量。

+   计算差值，这是对怀孕持续时间的估计。

+   使用`.describe()`来计算平均持续时间和其他总结统计量。

根据结果，似乎这可能不是估计怀孕持续时间的好方法。为什么？

## 直方图

让我们回到最初的问题：美国婴儿的平均出生体重是多少？

作为答案，我们*可以*使用前一节的结果来计算平均值：

```py
pounds_clean = pounds.replace([98, 99], np.nan)
ounces_clean = ounces.replace([98, 99], np.nan)

birth_weight = pounds_clean + ounces_clean / 16
birth_weight.mean() 
```

```py
7.180217889908257 
```

但在查看值的整个分布之前计算一个总结统计量，如均值，是有风险的。

**分布**是一组可能的值及其频率。可视化分布的一种方法是**直方图**，它显示了`x`轴上的值及其在`y`轴上的频率。

`Series`提供了一个`hist`方法，用于制作直方图。我们可以使用 Matplotlib 来标记轴。

```py
import matplotlib.pyplot as plt

birth_weight.hist(bins=30)
plt.xlabel('Birth weight in pounds')
plt.ylabel('Number of live births')
plt.title('Distribution of U.S. birth weight'); 
```

![_images/07_dataframes_66_0.png](img/b0744351975b411829eb928c8e0df35a.png)

关键字参数`bins`告诉`hist`将重量范围分成 30 个间隔，称为**bins**，并计算每个 bin 中有多少值。`x`轴是出生体重（磅）；`y`轴是每个 bin 中的出生次数。

分布看起来有点像钟形曲线，但左边的尾巴比右边的长；也就是说，轻的婴儿比重的婴儿更多。这是有道理的，因为分布包括一些早产婴儿。

**练习：**`hist`接受关键字参数，指定直方图的类型和外观。找到`hist`的文档，看看是否能弄清楚如何绘制直方图作为一个未填充的线，并且背景没有网格线。

**练习：**正如我们在之前的练习中看到的，NSFG 数据集包括一个名为`AGECON`的列，记录了每次怀孕的受孕年龄。

+   从`DataFrame`中选择这一列，并用 20 个 bin 绘制值的直方图。

+   适当地标记`x`轴和`y`轴。

## 布尔 Series

我们已经看到出生体重的分布向左**偏斜**；也就是说，轻婴儿比重婴儿更多，它们离均值更远。这是因为早产婴儿往往体重较轻。怀孕的最常见持续时间是 39 周，这是“足月”；“早产”通常定义为少于 37 周。

要查看哪些婴儿为早产，我们可以使用`PRGLNGTH`，它记录了怀孕周数，并将其计算为`37`。

```py
preterm = (nsfg['PRGLNGTH'] < 37)
preterm.dtype 
```

```py
dtype('bool') 
```

当您将`Series`与一个值进行比较时，结果是一个布尔`Series`；也就是说，每个元素都是布尔值，`True`或`False`。在这种情况下，每个早产婴儿为`True`，否则为`False`。我们可以使用`head`来查看前 5 个元素。

```py
preterm.head() 
```

```py
0    False
1     True
2    False
3    False
4    False
Name: PRGLNGTH, dtype: bool 
```

如果计算布尔`Series`的总和，它将`True`视为 1，`False`视为 0，因此总和是`True`值的数量，即早产婴儿的数量。

```py
preterm.sum() 
```

```py
3675 
```

如果计算布尔`Series`的均值，您将得到`True`值的*比例*。在这种情况下，约为 0.38；也就是说，大约 38%的怀孕少于 37 周。

```py
preterm.mean() 
```

```py
0.38469590704490736 
```

然而，这个结果可能会误导，因为它包括了所有的怀孕结果，而不仅仅是活产。我们可以创建另一个布尔`Series`来指示哪些怀孕以活产结束：

```py
live = (nsfg['OUTCOME'] == 1)
live.mean() 
```

```py
0.7006176070344394 
```

现在我们可以使用运算符`&`，它表示逻辑与操作，来识别结果为活产和早产的怀孕：

```py
live_preterm = (live & preterm)
live_preterm.mean() 
```

```py
0.08929132209777034 
```

**练习：**在所有活产中，早产的比例是多少？

其他常见的逻辑运算符是：

+   `|`，表示逻辑或操作；例如`live | preterm`如果`live`为真，或`preterm`为真，或两者都为真，则为真。

+   `~`，表示逻辑非操作；例如`~live`如果`live`为假或`NaN`，则为真。

逻辑运算符将`NaN`视为`False`，因此在使用带有`NaN`值的`Series`的 NOT 运算符时，您应该小心。例如，`~preterm`将包括不仅仅是足月怀孕，还包括怀孕持续时间未知的怀孕。

**练习：**在所有怀孕中，足月的比例是多少，也就是说，37 周或更长？在所有活产中，足月的比例是多少？

## 数据过滤

我们可以使用布尔`Series`作为过滤器；也就是说，我们可以仅选择满足条件或符合某些标准的行。例如，我们可以使用`preterm`和括号运算符从`birth_weight`中选择值，因此`preterm_weight`得到早产婴儿的出生体重。

```py
preterm_weight = birth_weight[preterm]
preterm_weight.mean() 
```

```py
5.480958781362007 
```

要选择足月婴儿，我们可以创建一个布尔`Series`，如下所示：

```py
fullterm = (nsfg['PRGLNGTH'] >= 37) 
```

并使用它来选择足月婴儿的出生体重：

```py
full_term_weight = birth_weight[fullterm]
full_term_weight.mean() 
```

```py
7.429609416096791 
```

预期地，足月婴儿的平均体重要比早产婴儿重。更明确地说，我们也可以将结果限制为活产，如下所示：

```py
full_term_weight = birth_weight[live & fullterm]
full_term_weight.mean() 
```

```py
7.429609416096791 
```

但在这种情况下，我们得到了相同的结果，因为`birth_weight`仅对活产有效。

**练习：**让我们看看单胎和多胎（双胞胎，三胞胎等）之间的体重差异。变量`NBRNALIV`表示单次怀孕中活产的婴儿数量。

```py
nbrnaliv = nsfg['NBRNALIV']
nbrnaliv.value_counts() 
```

```py
1.0    6573
2.0     111
3.0       6
Name: NBRNALIV, dtype: int64 
```

使用`nbrnaliv`和`live`创建一个名为`multiple`的布尔系列，该系列对于多个活产为真。在所有活产中，多胎出生的比例是多少？

**练习：**创建一个名为`single`的布尔系列，对于单个活产为真。在所有单胎中，早产的比例是多少？在所有多胎中，早产的比例是多少？

**练习：**活产、单胎、足月出生的平均出生体重是多少？

## 加权平均值

我们几乎完成了，但还有一个问题需要解决：过度抽样。

NSFG 并不完全代表美国人口。按设计，一些群体比其他群体更有可能出现在样本中；也就是说，它们被**过度抽样**了。过度抽样有助于确保每个子组中有足够的人数以获得可靠的统计数据，但它使数据分析变得更加复杂。

数据集中的每次怀孕都有一个表示其代表多少怀孕的**抽样权重**。在`nsfg`中，抽样权重存储在名为`wgt2015_2017`的列中。它看起来是这样的。

```py
sampling_weight = nsfg['WGT2015_2017']
sampling_weight.describe() 
```

```py
count      9553.000000
mean      13337.425944
std       16138.878271
min        1924.916000
25%        4575.221221
50%        7292.490835
75%       15724.902673
max      106774.400000
Name: WGT2015_2017, dtype: float64 
```

该列的中位数（第 50 百分位数）约为 7292，这意味着体重为该值的怀孕代表人口中的 7292 个总怀孕。但数值范围很广，因此有些行代表的怀孕数量要比其他行多得多。

为了考虑这些权重，我们可以计算**加权平均值**。以下是步骤：

1.  将每次怀孕的出生体重乘以抽样权重并将乘积相加。

1.  将抽样权重相加。

1.  将第一个总和除以第二个总和。

为了做到正确，我们必须小心处理缺失数据。为了帮助处理，我们将使用两个`Series`方法，`isna`和`notna`。

`isna`返回一个布尔`Series`，其中相应的值为`NaN`时为`True`。

```py
missing = birth_weight.isna()
missing.sum() 
```

```py
3013 
```

在`birth_weight`中有 3013 个缺失值（主要是指没有以活产结束的怀孕）。

`notna`返回一个布尔`Series`，其中相应的值*不是*`NaN`时为`True`。

```py
valid = birth_weight.notna()
valid.sum() 
```

```py
6540 
```

我们可以将`valid`与我们计算出的其他布尔`Series`结合起来，以识别具有有效出生体重的单胎、足月、活产。

```py
single = (nbrnaliv == 1)
selected = valid & live & single & fullterm
selected.sum() 
```

```py
5648 
```

你可以把这个计算完成作为练习。

**练习：**使用`selected`，`birth_weight`和`sampling_weight`来计算活产、单胎、足月婴儿出生体重的加权平均值。

你会发现，加权平均值比我们在上一节计算的非加权平均值略高。这是因为在 NSFG 中被过度抽样的群体平均来说婴儿体重较轻。

## 总结

这一章提出了一个看似简单的问题：美国婴儿的平均出生体重是多少？

为了回答这个问题，我们找到了一个合适的数据集并读取了文件。然后我们验证了数据并处理了特殊值、缺失数据和错误。为了探索数据，我们使用了`value_counts`，`hist`，`describe`和其他 Pandas 方法。为了选择相关数据，我们使用了布尔`Series`。

在这个过程中，我们不得不更多地思考这个问题。我们所说的“平均”，以及应该包括哪些婴儿？应该包括所有活产还是排除早产婴儿或多胞胎？

我们还必须考虑抽样过程。按设计，NSFG 受访者并不代表美国人口，但我们可以使用抽样权重来纠正这种影响。

即使是一个简单的问题也可能是一个具有挑战性的数据科学项目。
