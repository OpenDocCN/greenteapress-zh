# 探索性数据分析

> 原文：[`allendowney.github.io/ThinkStats/chap01.html`](https://allendowney.github.io/ThinkStats/chap01.html)

这本书的论点是我们可以使用数据来回答问题、解决辩论和做出更好的决策。

本章介绍了我们将要使用的步骤：加载数据和验证数据、探索数据，并选择衡量我们感兴趣内容的统计方法。作为一个例子，我们将使用国家家庭成长调查（NSFG）的数据来回答当我和妻子期待我们第一个孩子时听到的问题：第一个孩子是否倾向于晚出生？

[点击此处运行此笔记本在 Colab 上](https://colab.research.google.com/github/AllenDowney/ThinkStats/blob/v3/nb/chap01.ipynb)。

```py
from  os.path  import basename, exists

def  download(url):
    filename = basename(url)
    if not exists(filename):
        from  urllib.request  import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)

download("https://github.com/AllenDowney/ThinkStats/raw/v3/nb/thinkstats.py") 
```

```py
try:
    import  empiricaldist
except ImportError:
    %pip install empiricaldist 
```

```py
import  numpy  as  np
import  pandas  as  pd
import  matplotlib.pyplot  as  plt

from  IPython.display  import HTML
from  thinkstats  import decorate 
```

## 证据

你可能听说过第一个孩子更有可能晚出生。如果你用这个问题在网络上搜索，你会找到很多讨论。有些人声称这是真的，有些人说这是一个神话，还有一些人说情况正好相反：第一个孩子会提前出生。

在许多这样的讨论中，人们提供数据来支持他们的主张。我发现了很多这样的例子：

> “我最近有两个朋友生了第一个孩子，他们两个在分娩或被诱导分娩前都几乎迟到了两周。”
> 
> “我的第一个孩子迟到了两周，现在我觉得第二个孩子可能会提前两周出生！！！”
> 
> “我不认为这是真的，因为我的姐姐是我母亲第一个孩子，她很早就出生了，就像我的许多表亲一样。”

这样的报告被称为**轶事证据**，因为它们基于未发表的数据，通常是个人数据。在闲聊中，轶事并没有什么不妥，所以我并不是要批评我引用的人。

但我们可能需要更有说服力的证据和更可靠的答案。按照这些标准，轶事证据通常是不够的，因为：

+   观察数量少：如果第一个孩子的怀孕时间更长，那么与自然变异相比，差异可能很小。在这种情况下，我们可能需要比较大量的怀孕案例来了解是否存在差异。

+   选择偏误：加入关于这个问题讨论的人可能是因为他们的第一个孩子晚出生而感兴趣的。在这种情况下，选择数据的过程可能会使结果产生偏差。

+   确认偏误：相信这个说法的人可能更倾向于提供证实它的例子。怀疑这个说法的人更有可能引用反例。

+   不准确性：轶事通常是个人故事，可能会被错误地记住、错误地表述、不准确地重复等。

为了解决轶事的局限性，我们将使用统计工具，这包括：

+   数据收集：我们将使用来自一个大型国家调查的数据，该调查明确旨在生成关于美国人口的统计有效推断。

+   描述性统计：我们将生成总结数据的统计量，并评估不同的数据可视化方法。

+   探索性数据分析：我们将寻找模式、差异和其他特征，以解决我们感兴趣的问题。同时，我们将检查不一致性并确定局限性。

+   估计：我们将使用样本数据来估计总体特征。

+   假设检验：当我们看到明显的效应，如两组之间的差异时，我们将评估这种效应是否可能偶然发生。

通过小心执行这些步骤以避免陷阱，我们可以得出更加合理且更有可能正确的结论。

## 全国家庭成长调查

自 1973 年以来，美国疾病控制与预防中心（CDC）一直在进行全国家庭成长调查（NSFG），旨在收集有关“家庭生活、婚姻与离婚、怀孕、不孕、避孕药具的使用以及男女健康”的信息。调查结果被用于……规划健康服务和健康教育计划，以及进行家庭、生育和健康方面的统计分析。

你可以在[`cdc.gov/nchs/nsfg.htm`](http://cdc.gov/nchs/nsfg.htm)了解更多关于 NSFG 的信息。

我们将使用此调查收集的数据来调查是否第一个孩子倾向于晚出生，以及其他问题。为了有效地使用这些数据，我们必须了解研究的设计。

通常，统计研究的目标是关于**总体**的结论。在 NSFG 中，目标总体是美国 15-44 岁的人群。

理想情况下，调查将收集总体中每个成员的数据，但这很少可能实现。相反，我们收集来自称为**样本**的总体子集的数据。参与调查的人被称为**受访者**。

NSFG 是一项**横断面**研究，这意味着它捕捉了在某一时间点的总体快照。NSFG 已经进行了几次；每次部署被称为**周期**。我们将使用第 6 周期的数据，该周期从 2002 年 1 月持续到 2003 年 3 月。

通常，横断面研究旨在是**代表性的**，这意味着样本在所有对研究目的重要方面都与目标总体相似。在实践中实现这一理想很困难，但进行调查的人会尽可能接近。

美国国家家庭生育调查（NSFG）不具有代表性；相反，它是**分层**的，这意味着它故意**过度抽样**某些群体。该研究的制定者招募了三个群体——西班牙裔、非裔美国人和青少年——其在美国人口中的代表性高于其比例，以确保每个群体中受访者的数量足够多，以便得出有效的结论。过度抽样的缺点是，基于样本的统计数据，基于人口得出结论并不那么容易。我们稍后会回到这个点。

当处理这类数据时，熟悉**代码簿**非常重要，它记录了研究的设计、调查问题和响应的编码。

NSFG 数据的代码簿和用户指南可在[`www.cdc.gov/nchs/nsfg/nsfg_cycle6.htm`](http://www.cdc.gov/nchs/nsfg/nsfg_cycle6.htm)获取。

## 数据读取

在下载 NSFG 数据之前，您必须同意使用条款：

> 任何对个人或机构的故意识别或披露都违反了向信息提供者提供的保密保证。因此，用户将：
> 
> +   仅使用此数据集中的数据进行统计分析。
> +   
> +   不要试图了解这些数据中包含的任何个人或机构的身份。
> +   
> +   不要将此数据集与来自其他 NCHS 或非 NCHS 数据集的个人可识别数据链接。
> +   
> +   不要参与评估用于保护个人和机构的披露方法或任何关于个人和机构重新识别方法的研究。

如果您同意遵守这些条款，本章笔记本中提供了下载数据的说明。

数据文件可以直接从 NSFG 网站获取，网址为[`www.cdc.gov/nchs/data_access/ftp_dua.htm?url_redirect=ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NSFG`](https://www.cdc.gov/nchs/data_access/ftp_dua.htm?url_redirect=ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NSFG)，但我们将从本书的存储库中下载它们，该存储库提供了数据文件的压缩版本。

以下单元格下载数据文件并安装`statadict`，这是我们读取数据所需的。

```py
download("https://github.com/AllenDowney/ThinkStats/raw/v3/data/2002FemPreg.dct")
download("https://github.com/AllenDowney/ThinkStats/raw/v3/data/2002FemPreg.dat.gz") 
```

```py
try:
    import  statadict
except ImportError:
    %pip install statadict 
```

数据存储在两个文件中，一个“字典”描述了数据的格式，一个数据文件。

```py
dct_file = "2002FemPreg.dct"
dat_file = "2002FemPreg.dat.gz" 
```

本章笔记本定义了一个函数，用于读取这些文件。它被称为`read_stata`，因为这种数据格式与名为 Stata 的统计软件包兼容。

以下函数将这些文件名作为参数，读取字典，并使用结果读取数据文件。

```py
from  statadict  import parse_stata_dict

def  read_stata(dct_file, dat_file):
    stata_dict = parse_stata_dict(dct_file)
    resp = pd.read_fwf(
        dat_file,
        names=stata_dict.names,
        colspecs=stata_dict.colspecs,
        compression="gzip",
    )
    return resp 
```

这是我们的使用方法。

```py
preg = read_stata(dct_file, dat_file) 
```

结果是一个`DataFrame`，这是 Pandas 数据结构，用于表示行和列的表格数据。这个`DataFrame`为每个受访者报告的怀孕包含一行，为每个**变量**包含一列。一个变量可以包含对调查问题的回答或基于一个或多个问题的回答计算出的值。

除了数据之外，`DataFrame`还包含变量名及其类型，并提供访问和修改数据的方法。`DataFrame`有一个名为`shape`的属性，其中包含行数和列数。

```py
preg.shape 
```

```py
(13593, 243) 
```

该数据集包含 243 个变量，涉及 13,593 次怀孕的信息。`DataFrame`提供了一个名为`head`的方法，用于显示前几行。

```py
preg.head() 
```

|  | caseid | pregordr | howpreg_n | howpreg_p | moscurrp | nowprgdk | pregend1 | pregend2 | nbrnaliv | multbrth | ... | poverty_i | laborfor_i | religion_i | metro_i | basewgt | adj_mod_basewgt | finalwgt | secu_p | sest | cmintvw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | NaN | NaN | NaN | NaN | 6.0 | NaN | 1.0 | NaN | ... | 0 | 0 | 0 | 0 | 3410.389399 | 3869.349602 | 6448.271112 | 2 | 9 | 1231 |
| 1 | 1 | 2 | NaN | NaN | NaN | NaN | 6.0 | NaN | 1.0 | NaN | ... | 0 | 0 | 0 | 0 | 3410.389399 | 3869.349602 | 6448.271112 | 2 | 9 | 1231 |
| 2 | 2 | 1 | NaN | NaN | NaN | NaN | 5.0 | NaN | 3.0 | 5.0 | ... | 0 | 0 | 0 | 0 | 7226.301740 | 8567.549110 | 12999.542264 | 2 | 12 | 1231 |
| 3 | 2 | 2 | NaN | NaN | NaN | NaN | 6.0 | NaN | 1.0 | NaN | ... | 0 | 0 | 0 | 0 | 7226.301740 | 8567.549110 | 12999.542264 | 2 | 12 | 1231 |
| 4 | 2 | 3 | NaN | NaN | NaN | NaN | 6.0 | NaN | 1.0 | NaN | ... | 0 | 0 | 0 | 0 | 7226.301740 | 8567.549110 | 12999.542264 | 2 | 12 | 1231 |

5 行 × 243 列

左侧列是`DataFrame`的索引，其中包含每行的标签。在这种情况下，标签是从 0 开始的整数，但它们也可以是字符串和其他类型。

`DataFrame`有一个名为`columns`的属性，其中包含变量的名称。

```py
preg.columns 
```

```py
Index(['caseid', 'pregordr', 'howpreg_n', 'howpreg_p', 'moscurrp', 'nowprgdk',
       'pregend1', 'pregend2', 'nbrnaliv', 'multbrth',
       ...
       'poverty_i', 'laborfor_i', 'religion_i', 'metro_i', 'basewgt',
       'adj_mod_basewgt', 'finalwgt', 'secu_p', 'sest', 'cmintvw'],
      dtype='object', length=243) 
```

列名包含在一个`Index`对象中，这是另一个 Pandas 数据结构。要从`DataFrame`中访问列，可以使用列名作为键。

```py
pregordr = preg["pregordr"]
type(pregordr) 
```

```py
pandas.core.series.Series 
```

结果是 Pandas 的`Series`，它表示一系列值。`Series`也提供了`head`方法，用于显示前几个值及其标签。

```py
pregordr.head() 
```

```py
0    1
1    2
2    1
3    2
4    3
Name: pregordr, dtype: int64 
```

最后一行包含`Series`的名称和`dtype`，即值的类型。在这个例子中，`int64`表示值是 64 位整数。

NSFG 数据集总共包含 243 个变量。以下是本书探索中将使用的一些变量。

+   `caseid`是受访者的整数 ID。

+   `pregordr`是怀孕序列号：受访者第一次怀孕的代码是 1，第二次怀孕是 2，依此类推。

+   `prglngth`是怀孕的整数持续时间（周）。

+   `outcome` 是妊娠结果的整数代码。代码 1 表示活产。

+   `birthord` 是活产的序列号：受访者的第一个孩子的代码是 1，依此类推。对于除活产之外的结果，此字段为空。

+   `birthwgt_lb` 和 `birthwgt_oz` 包含婴儿出生体重的磅和盎司部分。

+   `agepreg` 是妊娠结束时的母亲年龄。

+   `finalwgt` 是与受访者相关的统计权重。它是一个浮点值，表示该受访者代表美国人口中的多少人。

如果仔细阅读代码簿，你会看到许多变量是**重新编码**，这意味着它们不是调查收集的**原始数据**的一部分——它们是使用原始数据计算得出的。

例如，对于活产，如果可用，`prglngth` 等于原始变量 `wksgest`（妊娠周数）；否则，它使用 `mosgest * 4.33`（妊娠月数乘以每月平均周数）来估算。

重新编码通常基于检查数据的完整性和准确性的逻辑。一般来说，当可用时使用重新编码是一个好主意，除非有充分的理由自行处理原始数据。

## 验证

当数据从一个软件环境导出并导入到另一个软件环境时，可能会引入错误。当你熟悉新的数据集时，你可能会错误地解码数据或误解其含义。如果你投入时间验证数据，你可以在以后节省时间并避免错误。

验证数据的一种方法是通过计算基本统计量并将它们与已发布的结果进行比较。例如，NSFG 代码簿包括总结每个变量的表格。以下是 `outcome` 的表格，它编码了每个妊娠的结果。

| 值 | 标签 | 总计 |
| --- | --- | --- |
| 1 | 活产 | 9148 |
| 2 | 人工流产 | 1862 |
| 3 | 死产 | 120 |
| 4 | 流产 | 1921 |
| 5 | 宫外孕 | 190 |
| 6 | 当前妊娠 | 352 |
| 总计 |  | 13593 |

“总计”列表示具有每种结果的妊娠数量。为了检查这些总计，我们将使用 `value_counts` 方法，该方法计算每个值出现的次数，以及 `sort_index`，该方法根据 `Index`（左侧列）中的值对结果进行排序。

```py
preg["outcome"].value_counts().sort_index() 
```

```py
outcome
1    9148
2    1862
3     120
4    1921
5     190
6     352
Name: count, dtype: int64 
```

将结果与已发布的表格进行比较，我们可以确认 `outcome` 中的值是正确的。同样，以下是 `birthwgt_lb` 的已发布表格。

| 值 | 标签 | 总计 |
| --- | --- | --- |
| . | 不适用 | 4449 |
| 0-5 | 低于 6 磅 | 1125 |
| 6 | 6 磅 | 2223 |
| 7 | 7 磅 | 3049 |
| 8 | 8 磅 | 1889 |
| 9-95 | 9 磅或以上 | 799 |
| 97 | 未确定 | 1 |
| 98 | 拒绝 | 1 |
| 99 | 未知 | 57 |
| 总计 |  | 13593 |

出生重量仅记录在以活产结束的怀孕中。表格表明，有 4449 个案例中这个变量不适用。此外，还有一个案例中未提问，一个案例中受访者未回答，以及 57 个案例中他们不知道。

再次，我们可以使用`value_counts`来比较数据集中的计数与代码簿中的计数。

```py
counts = preg["birthwgt_lb"].value_counts(dropna=False).sort_index()
counts 
```

```py
birthwgt_lb
0.0        8
1.0       40
2.0       53
3.0       98
4.0      229
5.0      697
6.0     2223
7.0     3049
8.0     1889
9.0      623
10.0     132
11.0      26
12.0      10
13.0       3
14.0       3
15.0       1
51.0       1
97.0       1
98.0       1
99.0      57
NaN     4449
Name: count, dtype: int64 
```

参数`dropna=False`意味着`value_counts`不会忽略“NA”或“不适用”的值。这些值在结果中显示为`NaN`，代表“不是一个数字”——并且这些值的计数与代码簿中不适用的案例计数一致。

6、7 和 8 磅的计数与代码簿一致。为了检查 0 至 5 磅范围内的计数，我们可以使用一个名为`loc`的属性——它是“位置”的缩写——以及一个切片索引来选择计数的一个子集。

```py
counts.loc[0:5] 
```

```py
birthwgt_lb
0.0      8
1.0     40
2.0     53
3.0     98
4.0    229
5.0    697
Name: count, dtype: int64 
```

我们还可以使用`sum`方法将它们加起来。

```py
counts.loc[0:5].sum() 
```

```py
np.int64(1125) 
```

总数与代码簿一致。

值 97、98 和 99 代表出生重量未知的情况。我们可能有几种处理缺失数据的方法。一个简单的选项是将这些值替换为`NaN`。同时，我们还将替换一个明显错误的值，51 磅。

我们可以使用这种方法使用`replace`方法：

```py
preg["birthwgt_lb"] = preg["birthwgt_lb"].replace([51, 97, 98, 99], np.nan) 
```

第一个参数是要替换的值的列表。第二个参数`np.nan`从 NumPy 获取`NaN`值。

当你以这种方式读取数据时，你通常需要检查错误并处理特殊值。这种操作被称为**数据清洗**。

## 转换

作为另一种数据清洗方式，有时我们必须将数据转换为不同的格式，并执行其他计算。

例如，`agepreg`包含怀孕结束时的母亲年龄。根据代码簿，它是一个以百分之一年（百分之一年）为单位的整数，正如我们可以通过使用`mean`方法计算其平均值来得知。

```py
preg["agepreg"].mean() 
```

```py
np.float64(2468.8151197039497) 
```

为了将其转换为年，我们可以将其除以 100。

```py
preg["agepreg"] /= 100.0
preg["agepreg"].mean() 
```

```py
np.float64(24.6881511970395) 
```

现在平均值更可信了。

作为另一个例子，`birthwgt_lb`和`birthwgt_oz`包含出生重量，磅和盎司分别在不同的列中。将它们合并成一个包含磅和磅分数的单列将更方便。

首先，我们将像处理`birthwgt_lb`一样清洗`birthwgt_oz`。

```py
preg["birthwgt_oz"].value_counts(dropna=False).sort_index() 
```

```py
birthwgt_oz
0.0     1037
1.0      408
2.0      603
3.0      533
4.0      525
5.0      535
6.0      709
7.0      501
8.0      756
9.0      505
10.0     475
11.0     557
12.0     555
13.0     487
14.0     475
15.0     378
97.0       1
98.0       1
99.0      46
NaN     4506
Name: count, dtype: int64 
```

```py
preg["birthwgt_oz"] = preg["birthwgt_oz"].replace([97, 98, 99], np.nan) 
```

现在我们可以使用清洗后的值来创建一个新的列，该列将磅和盎司合并成一个单一的数量。

```py
preg["totalwgt_lb"] = preg["birthwgt_lb"] + preg["birthwgt_oz"] / 16.0
preg["totalwgt_lb"].mean() 
```

```py
np.float64(7.265628457623368) 
```

结果的平均值看起来是合理的。

## 摘要统计

**统计量**是从数据集中派生出的一个数字，通常旨在量化数据的某个方面。例如包括计数、平均值、方差和标准差。

`Series`对象有一个`count`方法，它返回非`nan`值的数量。

```py
weights = preg["totalwgt_lb"]
n = weights.count()
n 
```

```py
np.int64(9038) 
```

它还提供了一个返回值总和的`sum`方法——我们可以用它来计算平均值，如下所示。

```py
mean = weights.sum() / n
mean 
```

```py
np.float64(7.265628457623368) 
```

但正如我们已经看到的，还有一个 `mean` 方法可以做到同样的事情。

```py
weights.mean() 
```

```py
np.float64(7.265628457623368) 
```

在这个数据集中，平均出生体重约为 7.3 磅。

方差是一种统计量，用于量化一组值的分布。它是平方偏差的平均值，即每个点与平均值之间的距离。

```py
squared_deviations = (weights - mean) ** 2 
```

我们可以这样计算平方偏差的平均值。

```py
var = squared_deviations.sum() / n
var 
```

```py
np.float64(1.983070989750022) 
```

如您所预期，`Series` 提供了一个 `var` 方法，它几乎做的是同样的事情。

```py
weights.var() 
```

```py
np.float64(1.9832904288326545) 
```

结果略有不同，因为当 `var` 方法计算平方偏差的平均值时，它除以 `n-1` 而不是 `n`。这是因为根据你试图做什么，有两种计算样本方差的方法。我将在 第八章 中解释这种差异——但在实践中通常无关紧要。如果你更喜欢分母中有 `n` 的版本，你可以通过将 `ddof=0` 作为关键字参数传递给 `var` 方法来获得它。

```py
weights.var(ddof=0) 
```

```py
np.float64(1.983070989750022) 
```

在这个数据集中，出生体重的方差大约是 1.98，但这个值很难解释——一方面，它是以磅平方为单位的。方差在某些计算中很有用，但不是描述数据集的好方法。更好的选择是 **标准差**，它是方差的平方根。我们可以这样计算它。

```py
std = np.sqrt(var)
std 
```

```py
np.float64(1.40821553384062) 
```

或者，我们可以使用 `std` 方法。

```py
weights.std(ddof=0) 
```

```py
np.float64(1.40821553384062) 
```

在这个数据集中，出生体重的标准差约为 1.4 磅。非正式地说，距离平均值一个或两个标准差的值很常见——距离平均值更远的值很少见。

## 解释

为了有效地处理数据，你必须同时从两个层面思考：统计层面和背景层面。例如，让我们选择怀孕文件中 `caseid` 为 10229 的行。`query` 方法接受一个字符串，可以包含列名、比较运算符和数字等。

```py
subset = preg.query("caseid == 10229")
subset.shape 
```

```py
(7, 244) 
```

结果是一个只包含查询为 `True` 的行的 `DataFrame`。这位受访者报告了七次怀孕——以下是他们的结果，这些结果按时间顺序记录。

```py
subset["outcome"].values 
```

```py
array([4, 4, 4, 4, 4, 4, 1]) 
```

结果代码 `1` 表示活产。代码 `4` 表示流产——即怀孕丢失，通常没有已知的医学原因。

从统计学的角度来看，这位受访者并不异常。怀孕丢失很常见，还有其他受访者报告了同样多的实例。但考虑到背景，这些数据讲述了一个女人怀孕六次，每次都以流产告终的故事。她的第七次也是最最近的一次怀孕以活产结束。如果我们带着同情心考虑这些数据，那么这个故事自然会触动人心。

NSFG 数据集中的每一行代表一个对许多个人和困难问题给出诚实回答的人。我们可以使用这些数据来回答关于家庭生活、生育和健康的统计问题。同时，我们有责任考虑数据所代表的人，并给予他们尊重和感激。

## 术语表

每章的结尾提供了一个定义在章节中的词汇表。

+   **轶事证据**：从少量个体案例中非正式收集的数据，通常没有系统抽样。

+   **横断面研究**：在某一时间点或时间间隔内从代表性样本中收集数据的调查。

+   **周期**：在一个在多个时间间隔收集数据的调查中，一个数据收集间隔。

+   **总体**：研究主题的整个个体或项目群体。

+   **样本**：总体的一部分，通常随机选择。

+   **受访者**：参与调查并回答问题的人。

+   **代表性**：如果一个样本在研究目的上重要的方面与总体相似，则该样本是具有代表性的。

+   **分层抽样**：如果一个样本故意对某些群体进行过抽样，通常是为了确保包含足够的成员以支持有效的结论，则该样本是分层的。

+   **过抽样**：如果一个群体的成员在样本中出现的概率更高，则该群体是过抽样的。

+   **变量**：在调查数据中，变量是对问题的回答或从回答中计算出的值的集合。

+   **代码簿**：描述数据集中变量的文档，并提供有关数据的其他信息。

+   **重新编码**：基于数据集中其他变量的变量。

+   **原始数据**：收集后未经处理的原始数据。

+   **数据清洗**：识别和纠正数据集中错误的过程，处理缺失值，并计算重新编码。

+   **统计量**：描述或总结样本属性的值。

+   **标准差**：一个统计量，用于量化数据围绕平均值的分布。

## 练习

本章的练习基于 NSFG 怀孕文件。

### 练习 1.1

从`preg`中选择`birthord`列，打印值计数，并与[`ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NSFG/Cycle6Codebook-Pregnancy.pdf`](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NSFG/Cycle6Codebook-Pregnancy.pdf)中发布的代码簿结果进行比较。

### 练习 1.2

创建一个名为`totalwgt_kg`的新列，其中包含以千克为单位的出生体重（每千克大约有 2.2 磅）。计算新列的平均值和标准差。

### 练习 1.3

对于`caseid`为 2298 的受访者，其怀孕时长是多少？

`caseid`为 5013 的受访者第一个孩子的出生体重是多少？提示：您可以使用`and`在查询中检查多个条件。

[Think Stats: Python 中的探索性数据分析，第 3 版](https://allendowney.github.io/ThinkStats/index.html)

版权所有 2024 [艾伦·B·唐尼](https://allendowney.com)

代码许可：[MIT 许可证](https://mit-license.org/)

文本许可：[Creative Commons 知识共享署名-非商业性使用-相同方式共享 4.0 国际](https://creativecommons.org/licenses/by-nc-sa/4.0/)
