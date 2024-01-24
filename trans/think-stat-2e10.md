# 第九章 假设检验

> 原文：[`greenteapress.com/thinkstats2/html/thinkstats2010.html`](https://greenteapress.com/thinkstats2/html/thinkstats2010.html)

本章的代码在`hypothesis.py`中。有关下载和使用此代码的信息，请参阅第 0.2 节。

## 9.1 经典假设检验

通过研究 NSFG 的数据，我们看到了几个“表面效应”，包括初生婴儿和其他婴儿之间的差异。到目前为止，我们已经直接接受了这些效应；在本章中，我们将对它们进行测试。

我们要解决的基本问题是，我们在样本中看到的效应是否可能出现在更大的人群中。例如，在 NSFG 样本中，我们看到初生婴儿和其他婴儿的怀孕期长度有差异。我们想知道这种效应是否反映了美国妇女之间的真实差异，或者是否可能仅仅是样本中偶然出现的。

我们可以用几种方法来制定这个问题，包括费舍尔零假设检验、内曼-皮尔逊决策理论和贝叶斯推断^(1)。我在这里介绍的是这三种方法的一个子集，它包括了大多数人在实践中使用的内容，我将其称为经典假设检验。

经典假设检验的目标是回答这样一个问题：“在给定一个样本和一个表面效应的情况下，看到这样的效应的概率是多少？”我们是这样回答这个问题的：

+   第一步是通过选择一个检验统计量来量化表面效应的大小。在 NSFG 的例子中，表面效应是初生婴儿和其他婴儿的怀孕期长度差异，因此检验统计量的一个自然选择是两组之间的均值差异。

+   第二步是定义一个零假设，这是一个基于表面效应不是真实的假设系统模型。在 NSFG 的例子中，零假设是初生婴儿和其他婴儿之间没有差异；也就是说，两组的怀孕期长度具有相同的分布。

+   第三步是计算 p 值，即在零假设成立的情况下看到表面效应的概率。在 NSFG 的例子中，我们将计算实际的均值差异，然后计算在零假设下看到一个与之一样大或更大的差异的概率。

+   最后一步是解释结果。如果 p 值很低，效应被认为是统计显著的，这意味着它不太可能是偶然发生的。在这种情况下，我们推断这种效应更可能出现在更大的人群中。

这个过程的逻辑类似于反证法。为了证明一个数学命题 A，你暂时假设 A 是假的。如果这个假设导致矛盾，那么你就得出结论 A 实际上是真的。

类似地，为了测试“这种效应是真实的”这样的假设，我们暂时假设它不是。这就是零假设。基于这一假设，我们计算表面效应的概率。这就是 p 值。如果 p 值很低，我们得出结论，零假设不太可能成立。

## 9.2 假设检验

`thinkstats2`提供了`HypothesisTest`类，表示经典假设检验的结构。以下是定义：

```py
class HypothesisTest(object):

    def __init__(self, data):
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)

    def PValue(self, iters=1000):
        self.test_stats = [self.TestStatistic(self.RunModel())
                           for _ in range(iters)]

        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def TestStatistic(self, data):
        raise UnimplementedMethodException()

    def MakeModel(self):
        pass

    def RunModel(self):
        raise UnimplementedMethodException() 
```

`HypothesisTest`是一个抽象的父类，为一些方法提供完整的定义，并为其他方法提供占位符。基于`HypothesisTest`的子类继承`__init__`和`PValue`，并提供`TestStatistic`，`RunModel`，以及可选的`MakeModel`。

`__init__`以适当的形式接受数据。它调用`MakeModel`，建立零假设的表示，然后将数据传递给`TestStatistic`，计算样本中效应的大小。

`PValue`计算在零假设下显著效应的概率。它以`iters`作为参数，这是要运行的模拟次数。第一行生成模拟数据，计算检验统计量，并将它们存储在`test_stats`中。结果是`test_stats`中的元素比观察到的检验统计量`self.actual`大或等于的比例。

作为一个简单的例子^(2)，假设我们抛一枚硬币 250 次，看到 140 次正面和 110 次反面。基于这个结果，我们可能怀疑硬币是有偏的；也就是说，更有可能落在正面。为了测试这个假设，我们计算如果硬币实际上是公平的，看到这样的差异的概率：

```py
class CoinTest(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        heads, tails = data
        test_stat = abs(heads - tails)
        return test_stat

    def RunModel(self):
        heads, tails = self.data
        n = heads + tails
        sample = [random.choice('HT') for _ in range(n)]
        hist = thinkstats2.Hist(sample)
        data = hist['H'], hist['T']
        return data 
```

参数`data`是一对整数：正反面的次数。检验统计量是它们之间的绝对差异，因此`self.actual`是 30。

`RunModel`模拟假设硬币实际上是公平的。它生成 250 次抛硬币的样本，使用 Hist 来计算正反面的次数，并返回一对整数。

现在我们只需要实例化`CoinTest`并调用`PValue`：

```py
 ct = CoinTest((140, 110))
    pvalue = ct.PValue() 
```

结果约为 0.07，这意味着如果硬币是公平的，我们预计大约有 7%的时间会看到 30 这么大的差异。

我们应该如何解释这个结果？按照惯例，5%是统计显著性的阈值。如果 p 值小于 5%，则认为效应是显著的；否则不是。

但是 5%的选择是任意的，而且（正如我们将在后面看到的）p 值取决于检验统计量和零假设模型的选择。因此，p 值不应被视为精确的测量。

我建议根据 p 值的数量级来解释 p 值：如果 p 值小于 1%，效应不太可能是由于偶然性引起的; 如果大于 10%，效应可能可以用偶然性解释。1%到 10%之间的 p 值应被视为边际的。因此，在这个例子中，我得出结论，数据并没有提供强有力的证据表明硬币是有偏的还是没有。

## 9.3  测试均值的差异

最常见的效应之一是测试两组之间的均值差异。在 NSFG 数据中，我们看到第一个婴儿的平均怀孕期略长，出生体重略小。现在我们将看看这些效应是否在统计上显著。

对于这些例子，零假设是两组的分布相同。建模零假设的一种方法是通过排列; 也就是说，我们可以取第一个婴儿和其他人的值并对它们进行洗牌，将这两组视为一个大组：

```py
class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data 
```

`data`是一对序列，每组一个。检验统计量是均值的绝对差。

`MakeModel`记录了组的大小，`n`和`m`，并将组合成一个 NumPy 数组`self.pool`。

`RunModel`通过对汇总值进行洗牌并将其分成大小为`n`和`m`的两组来模拟零假设。与往常一样，`RunModel`的返回值与观察数据的格式相同。

为了测试怀孕长度的差异，我们运行：

```py
 live, firsts, others = first.MakeFrames()
    data = firsts.prglngth.values, others.prglngth.values
    ht = DiffMeansPermute(data)
    pvalue = ht.PValue() 
```

`MakeFrames`读取 NSFG 数据并返回表示所有活产、第一个婴儿和其他人的数据框。我们将怀孕长度提取为 NumPy 数组，将它们作为数据传递给`DiffMeansPermute`，并计算 p 值。结果约为 0.17，这意味着我们预计大约有 17%的时间会看到观察到的效果一样大的差异。因此，这种效应在统计上并不显著。

> * * *
> 
> ![](img/bc93e63f2becf3d667655bd41f74599f.png)
> 
> | 图 9.1：零假设下怀孕期长度均值差异的 CDF。 |
> | --- |
> 
> * * *

`HypothesisTest`提供了`PlotCdf`，它绘制了检验统计量的分布和一个灰色线，表示观察到的效应大小：

```py
 ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF') 
```

图 9.1 显示了结果。CDF 在 0.83 处与观察到的差异相交，这是 p 值的补数，0.17。

如果我们用出生体重进行相同的分析，计算得到的 p 值为 0；经过 1000 次尝试，模拟从未产生与观察到的差异 0.12 磅一样大的效应。因此，我们会报告 p < 0.001，并得出结论，出生体重的差异在统计上是显著的。

## 9.4  其他检验统计量

选择最佳的检验统计量取决于你试图解决的问题。例如，如果相关问题是第一个宝宝的怀孕期是否不同，那么测试均值的绝对差异是有意义的，就像我们在前一节中所做的那样。

如果我们有理由认为第一个宝宝可能会迟到，那么我们就不会取差值的绝对值；相反，我们会使用这个检验统计量：

```py
class DiffMeansOneSided(DiffMeansPermute):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.mean() - group2.mean()
        return test_stat 
```

`DiffMeansOneSided`继承了`DiffMeansPermute`的`MakeModel`和`RunModel`；唯一的区别是`TestStatistic`不取差值的绝对值。这种测试被称为单侧，因为它只计算差异分布的一侧。之前的测试，使用了两侧，是双侧的。

对于这个版本的检验统计量，p 值为 0.09。通常，单侧检验的 p 值约为双侧检验的一半，这取决于分布的形状。

单侧假设，即第一个宝宝出生较晚，比双侧假设更具体，因此 p 值更小。但即使对于更强的假设，差异也不具有统计学意义。

我们可以使用相同的框架来测试标准差的差异。在第 3.3 节中，我们看到一些证据表明第一个宝宝更有可能早产或晚产，而不太可能准时出生。因此，我们可能会假设标准差更高。下面是我们如何测试的方法：

```py
class DiffStdPermute(DiffMeansPermute):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.std() - group2.std()
        return test_stat 
```

这是一个单侧检验，因为假设是第一个宝宝的标准差更高，而不仅仅是不同。p 值为 0.09，这在统计上不显著。

## 9.5  测试相关性

这个框架也可以测试相关性。例如，在 NSFG 数据集中，出生体重和母亲年龄之间的相关性约为 0.07。似乎年龄较大的母亲会生下更重的宝宝。但这种效应可能是由于偶然造成的吗？

对于检验统计量，我使用 Pearson 相关性，但 Spearman 相关性也可以。如果我们有理由期望正相关，我们将进行单侧检验。但由于我们没有这样的理由，我将使用绝对值相关性进行双侧检验。

零假设是母亲年龄和新生儿体重之间没有相关性。通过洗牌观察到的值，我们可以模拟一个世界，在这个世界中，年龄和出生体重的分布是相同的，但变量是不相关的。

```py
class CorrelationPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys 
```

`data`是一对序列。`TestStatistic`计算 Pearson 相关性的绝对值。`RunModel`洗牌`xs`并返回模拟数据。

这是读取数据并运行测试的代码：

```py
 live, firsts, others = first.MakeFrames()
    live = live.dropna(subset=['agepreg', 'totalwgt_lb'])
    data = live.agepreg.values, live.totalwgt_lb.values
    ht = CorrelationPermute(data)
    pvalue = ht.PValue() 
```

我使用`dropna`和`subset`参数删除缺少我们需要的变量的行。

实际相关性为 0.07。计算得到的 p 值为 0；经过 1000 次迭代，最大的模拟相关性为 0.04。因此，尽管观察到的相关性很小，但在统计上是显著的。

这个例子提醒我们，“统计上显著”并不总是意味着一个效应是重要的，或者在实践中是显著的。它只意味着它不太可能是偶然发生的。

## 9.6  测试比例

假设你经营一家赌场，你怀疑一位顾客正在使用一个不正当的骰子；也就是说，一个面比其他面更有可能出现。你逮捕了这名涉嫌作弊的人，并没收了骰子，但现在你必须证明它是不正当的。你掷了 60 次骰子，得到了以下结果：

| 值 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| 频率 | 8 | 9 | 19 | 5 | 8 | 11 |

平均而言，你期望每个值出现 10 次。在这个数据集中，值 3 出现的次数比预期的要多，而值 4 出现的次数比预期的要少。但这些差异在统计上显著吗？

为了检验这个假设，我们可以计算每个值的期望频率，期望频率和观察频率之间的差异，以及总的绝对差异。在这个例子中，我们期望每一面出现 60 次中的 10 次；与这个期望的偏差是-2，-1，9，-5，-2 和 1；因此总的绝对差异是 20。我们会有多少次看到这样的差异？

这是一个回答这个问题的`HypothesisTest`版本：

```py
class DiceTest(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        observed = data
        n = sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = sum(abs(observed - expected))
        return test_stat

    def RunModel(self):
        n = sum(self.data)
        values = [1, 2, 3, 4, 5, 6]
        rolls = np.random.choice(values, n, replace=True)
        hist = thinkstats2.Hist(rolls)
        freqs = hist.Freqs(values)
        return freqs 
```

数据表示为频率列表：观察值为`[8, 9, 19, 5, 8, 11]`；期望频率都是 10。检验统计量是绝对差异的总和。

零假设是骰子是公平的，因此我们通过从`values`中随机抽取样本来模拟。`RunModel`使用`Hist`来计算并返回频率列表。

这些数据的 p 值为 0.13，这意味着如果骰子是公平的，我们预计会有观察到的总偏差，或者更多，大约 13%的时间。因此，明显的效应在统计上并不显著。

## 9.7  卡方检验

在前一节中，我们使用总偏差作为检验统计量。但是对于测试比例，更常见的是使用卡方统计量：

| χ² =  |
| --- |

&#124;   &#124;

&#124; ∑ &#124;

&#124; i &#124;

|   |
| --- |

&#124; (O[i] − E[i])² &#124;

&#124;  &#124;

&#124; E[i] &#124;

|   |
| --- |

其中 O[i]是观察频率，E[i]是期望频率。以下是 Python 代码：

```py
class DiceChiTest(DiceTest):

    def TestStatistic(self, data):
        observed = data
        n = sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = sum((observed - expected)**2 / expected)
        return test_stat 
```

对偏差进行平方（而不是取绝对值）会更加重视大的偏差。通过`expected`除以标准化偏差，尽管在这种情况下没有影响，因为期望频率都是相等的。

使用卡方统计量的 p 值为 0.04，远小于我们使用总偏差得到的 0.13。如果我们认真对待 5%的阈值，我们会认为这种效应在统计上是显著的。但考虑到这两个检验，我会说结果是边缘的。我不会排除骰子是歪的可能性，但我也不会定罪被告作弊者。

这个例子证明了一个重要的观点：p 值取决于检验统计量的选择和零假设模型，有时这些选择决定了效应是否在统计上显著。

## 9.8  再次第一个宝宝

在本章的前面，我们看过第一个宝宝和其他宝宝的怀孕时长，并得出结论，平均值和标准差的明显差异在统计上并不显著。但在第 3.3 节中，我们看到了怀孕时长分布中的几个明显差异，特别是在 35 到 43 周的范围内。为了确定这些差异是否在统计上显著，我们可以使用基于卡方统计量的检验。

代码结合了以前示例的元素：

```py
class PregLengthTest(thinkstats2.HypothesisTest):

    def MakeModel(self):
        firsts, others = self.data
        self.n = len(firsts)
        self.pool = np.hstack((firsts, others))

        pmf = thinkstats2.Pmf(self.pool)
        self.values = range(35, 44)
        self.expected_probs = np.array(pmf.Probs(self.values))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data 
```

数据表示为两个怀孕时长的列表。零假设是两个样本都是从相同的分布中抽取的。`MakeModel`通过使用`hstack`汇总两个样本来模拟该分布。然后`RunModel`通过对汇总样本进行洗牌并将其分成两部分来生成模拟数据。

`MakeModel`还定义了`values`，这是我们将使用的周数范围，以及`expected_probs`，这是汇总分布中每个值的概率。

这是计算检验统计量的代码：

```py
# class PregLengthTest:

    def TestStatistic(self, data):
        firsts, others = data
        stat = self.ChiSquared(firsts) + self.ChiSquared(others)
        return stat

    def ChiSquared(self, lengths):
        hist = thinkstats2.Hist(lengths)
        observed = np.array(hist.Freqs(self.values))
        expected = self.expected_probs * len(lengths)
        stat = sum((observed - expected)**2 / expected)
        return stat 
```

`TestStatistic`计算了第一个宝宝和其他宝宝的卡方统计量，并将它们相加。

`ChiSquared`接受怀孕长度的序列，计算其直方图，并计算`observed`，它是与`self.values`对应的频率列表。为了计算预期频率列表，它将预先计算的概率`expected_probs`乘以样本大小。它返回卡方统计量`stat`。

对于 NSFG 数据，总卡方统计量为 102，这本身并没有太多意义。但经过 1000 次迭代后，在零假设下生成的最大检验统计量为 32。我们得出结论，观察到的卡方统计量在零假设下是不太可能的，因此这个明显的效应具有统计学意义。

这个例子展示了卡方检验的一个局限性：它们表明两组之间存在差异，但并不具体说明差异是什么。

## 9.9 错误

在经典假设检验中，如果 p 值低于某个阈值，通常是 5%，则认为效应具有统计学意义。这个过程引发了两个问题：

+   如果效应实际上是由于偶然发生的，那么我们错误地认为它具有统计学意义的概率是多少？这个概率就是假阳性率。

+   如果效应是真实的，那么假设检验失败的概率是多少？这个概率就是假阴性率。

假阳性率相对容易计算：如果阈值为 5%，假阳性率为 5%。原因如下：

+   如果没有真实效应，那么零假设是成立的，因此我们可以通过模拟零假设来计算检验统计量的分布。将这个分布称为 CDF [T]。

+   每次运行实验，我们得到一个检验统计量 t，它是从 CDF [T]中抽取的。然后我们计算一个 p 值，即从 CDF [T]中抽取的随机值超过`t`的概率，即 1 − CDF T。

+   如果 p 值小于 5%，那么 CDF T 大于 95%；也就是说，t 超过了 95th 百分位数。那么从 CDF [T]中选择的值超过 95th 百分位数的概率有多大？是 5%。

因此，如果使用 5%的阈值进行一次假设检验，你期望 20 次中有 1 次是假阳性。

## 9.10 功效

假阴性率更难计算，因为它取决于实际效应大小，而通常我们并不知道。一种选择是计算基于假设效应大小的率。

例如，如果我们假设两组之间的观察差异是准确的，我们可以使用观察样本作为总体的模型，并使用模拟数据进行假设检验：

```py
def FalseNegRate(data, num_runs=100):
    group1, group2 = data
    count = 0

    for i in range(num_runs):
        sample1 = thinkstats2.Resample(group1)
        sample2 = thinkstats2.Resample(group2)

        ht = DiffMeansPermute((sample1, sample2))
        pvalue = ht.PValue(iters=101)
        if pvalue > 0.05:
            count += 1

    return count / num_runs 
```

`FalseNegRate`接受两个序列的数据，每次循环时，它通过从每组中抽取随机样本并运行假设检验来模拟实验。然后它检查结果并计算假阴性的数量。

`Resample`接受一个序列并抽取一个与之相同长度的样本，可以进行替换：

```py
def Resample(xs):
    return np.random.choice(xs, len(xs), replace=True) 
```

以下是测试怀孕长度的代码：

```py
 live, firsts, others = first.MakeFrames()
    data = firsts.prglngth.values, others.prglngth.values
    neg_rate = FalseNegRate(data) 
```

结果约为 70%，这意味着如果实际的怀孕期长度的平均差异为 0.078 周，我们期望使用这个样本大小进行的实验有 70%的可能性产生负面测试结果。

这个结果通常是反过来呈现的：如果实际差异为 0.078 周，我们只有 30%的几率期望得到积极的测试结果。这个“正确的积极率”被称为测试的功效，有时也称为“敏感性”。它反映了测试检测特定大小效应的能力。

在这个例子中，测试只有 30%的几率产生积极的结果（同样，假设差异为 0.078 周）。一般来说，80%的功效被认为是可以接受的，因此我们会说这个测试“功效不足”。

一般来说，负假设检验并不意味着两组之间没有差异；相反，它表明如果有差异，那么它太小以至于无法用这个样本大小检测出来。

## 9.11 复制

我在本章中演示的假设检验过程严格来说并不是一个好的做法。

首先，我进行了多次测试。如果您进行一次假设检验，那么发生假阳性的几率约为 20 分之 1，这可能是可以接受的。但是如果您进行 20 次测试，您应该至少期望有一个假阳性，大多数情况下都会发生。

其次，我在探索和测试中使用了相同的数据集。如果您探索一个大型数据集，发现了一个令人惊讶的效应，然后测试它是否显著，您很有可能产生假阳性。

为了补偿多重检验，您可以调整 p 值阈值（参见[`en.wikipedia.org/wiki/Holm-Bonferroni_method`](https://en.wikipedia.org/wiki/Holm-Bonferroni_method)）。或者您可以通过将数据分区，使用一个集合进行探索，另一个集合进行测试来解决这两个问题。

在某些领域，这些做法是必需的，或者至少是受鼓励的。但是通过复制已发表的结果来隐式解决这些问题也很常见。通常，第一篇报告新结果的论文被认为是探索性的。随后使用新数据复制结果的论文被认为是验证性的。

恰好，我们有机会复制本章中的结果。本书的第一版基于 2002 年发布的 NSFG 第 6 周期。2011 年 10 月，CDC 发布了基于 2006-2010 年进行的访谈的额外数据。`nsfg2.py`包含用于读取和清理这些数据的代码。在新数据集中：

+   妊娠期长度的平均差异为 0.16 周，p <0.001，具有统计学意义（与原始数据集中的 0.078 周相比）。

+   出生体重的差异为 0.17 磅，p <0.001（与原始数据集中的 0.12 磅相比）。

+   出生体重与母亲年龄之间的相关性为 0.08，p <0.001（与 0.07 相比）。

+   卡方检验具有统计学意义，p <0.001（与原始数据集中的情况一样）。

总之，在新数据集中，所有在原始数据集中具有统计学意义的效应都在新数据集中得到了复制，而在原始数据集中不显著的妊娠期长度差异在新数据集中更大且显著。

## 9.12 练习

这些练习的解决方案在`chap09soln.py`中。

练习 1  随着样本大小的增加，假设检验的功效会增加，这意味着如果效应是真实的，它更有可能是积极的。相反，随着样本大小的减小，即使效应是真实的，测试也不太可能是积极的。

*为了调查这种行为，使用 NSFG 数据的不同子集运行本章中的测试。您可以使用`thinkstats2.SampleRows`来选择 DataFrame 中行的随机子集。*

*随着样本大小的减小，这些检验的 p 值会发生什么变化？能够产生积极测试的最小样本量是多少？*

练习 2

*在第 9.3 节中，我们通过置换来模拟零假设；也就是说，我们将观察到的值视为代表整个人口，并随机将人口的成员分配到两个组中。*

*另一种方法是使用样本来估计人口的分布，然后从该分布中随机抽取一个样本。这个过程称为重新采样。有几种实现重新采样的方法，但其中一种最简单的方法是从观察到的值中有放回地抽取样本，就像第 9.10 节中那样。*

*编写一个名为`DiffMeansResample`的类，该类继承自`DiffMeansPermute`并覆盖`RunModel`以实现重新采样，而不是置换。*

*使用此模型测试妊娠期长度和出生体重的差异。模型对结果有多大影响？*

## 9.13 术语表

+   假设检验：确定明显效应是否具有统计学意义的过程。

+   检验统计量：用于量化效应大小的统计量。

+   零假设：基于假设明显效应是由偶然引起的系统模型。

+   p 值：一个效应可能是偶然发生的概率。

+   统计学上显著：如果一个效应不太可能是偶然发生的，那么它就是统计学上显著的。

+   排列检验：通过生成观察到的数据集的排列来计算 p 值的一种方法。

+   重抽样检验：通过从观察到的数据集中生成带有替换的样本来计算 p 值的一种方法。

+   双侧检验：一个检验，问：“观察到的效应有多大几率是正的或负的？”

+   单侧检验：一个检验，问：“观察到的效应有多大几率和同样的符号一样大？”

+   卡方检验：使用卡方统计量作为检验统计量的一种检验方法。

+   假阳性：当一个效应并不存在时得出结论说它是真实的。

+   假阴性：当一个效应并非由偶然引起时得出结论说它是由偶然引起的。

+   功效：如果零假设是错误的，进行正检验的概率。

* * *

1

有关贝叶斯推断的更多信息，请参阅本书的续集《Bayes 思维》。

2

改编自 MacKay，《信息论、推断和学习算法》，2003 年。
