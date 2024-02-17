# 红线问题

> 原文：[`allendowney.github.io/ThinkBayes2/redline.html`](https://allendowney.github.io/ThinkBayes2/redline.html)

红线是马萨诸塞州剑桥和波士顿之间的地铁。当我在剑桥工作时，我从肯德尔广场乘坐红线到南站，然后搭乘通勤列车到尼德姆。在高峰时间，红线列车平均每 7-8 分钟运行一次。

当我到达地铁站时，我可以根据站台上的乘客数量估计下一班火车的时间。如果只有几个人，我推断我刚错过了一班火车，预计要等大约 7 分钟。如果有更多的乘客，我预计火车会更快到达。但如果有很多乘客，我怀疑火车没有按时运行，所以我会离开地铁站，打车。

当我在等火车的时候，我想到了贝叶斯估计如何帮助预测我的等待时间，并决定何时放弃，乘坐出租车。本章介绍了我想出的分析。

这个例子是基于布兰登·里特和凯·奥斯汀的一个项目，他们在奥林学院和我一起上课。

[单击此处在 Colab 上运行此笔记本](https://colab.research.google.com/github/AllenDowney/ThinkBayes2/blob/master/notebooks/redline.ipynb)

在我们进行分析之前，我们必须做出一些建模决策。首先，我将把乘客到达视为泊松过程，这意味着我假设乘客在任何时间到达的可能性都是相等的，并且以每分钟到达的速率λ来到达。由于我在短时间内观察到乘客，并且每天都在同一时间观察到乘客，我假设λ是恒定的。

另一方面，火车的到达过程不是泊松过程。到波士顿的火车应该在高峰时段每 7-8 分钟从终点站（阿尔维夫站）出发，但是当它们到达肯德尔广场时，列车之间的时间在 3 到 12 分钟之间变化。

为了收集有关列车之间的时间的数据，我编写了一个脚本，从[MBTA](http://www.mbta.com/rider_tools/developers/)下载实时数据，选择到达肯德尔广场的南行列车，并记录它们的到达时间在数据库中。我每个工作日下午 4 点到 6 点运行脚本 5 天，并记录每天约 15 次到达。然后我计算了连续到达之间的时间间隔。这是我记录的间隔时间，以秒为单位。

```py
observed_gap_times = [
    428.0, 705.0, 407.0, 465.0, 433.0, 425.0, 204.0, 506.0, 143.0, 351.0, 
    450.0, 598.0, 464.0, 749.0, 341.0, 586.0, 754.0, 256.0, 378.0, 435.0, 
    176.0, 405.0, 360.0, 519.0, 648.0, 374.0, 483.0, 537.0, 578.0, 534.0, 
    577.0, 619.0, 538.0, 331.0, 186.0, 629.0, 193.0, 360.0, 660.0, 484.0, 
    512.0, 315.0, 457.0, 404.0, 740.0, 388.0, 357.0, 485.0, 567.0, 160.0, 
    428.0, 387.0, 901.0, 187.0, 622.0, 616.0, 585.0, 474.0, 442.0, 499.0, 
    437.0, 620.0, 351.0, 286.0, 373.0, 232.0, 393.0, 745.0, 636.0, 758.0,
] 
```

我将把它们转换成分钟，并使用`kde_from_sample`来估计分布。

```py
import numpy as np

zs = np.array(observed_gap_times) / 60 
```

```py
from utils import kde_from_sample

qs = np.linspace(0, 20, 101)
pmf_z = kde_from_sample(zs, qs) 
```

这就是它的样子。

```py
from utils import decorate

pmf_z.plot()

decorate(xlabel='Time (min)',
         ylabel='PDF',
         title='Distribution of time between trains') 
```

![图片](img/7e5c60a70c0bba8a0b8e1436228eb7b6.png)

## 更新

在这一点上，我们已经对列车之间的时间分布有了估计。现在假设我到达车站，看到站台上有 10 名乘客。我应该期望什么等待时间的分布？

我们将分两步回答这个问题。

+   首先，我们将推导出作为随机到达（我）观察到的间隔时间的分布。

+   然后我们将推导出等待时间的分布，条件是乘客的数量。

当我到达车站时，我更有可能在较长的间隔期间到达，而不是在较短的间隔期间到达。实际上，我在任何间隔期间到达的概率与其持续时间成正比。

如果我们把`pmf_z`看作是间隔时间的先验分布，我们可以进行贝叶斯更新来计算后验分布。我在每个间隔期间到达的可能性是间隔的持续时间：

```py
likelihood = pmf_z.qs 
```

所以这是第一个更新。

```py
posterior_z = pmf_z * pmf_z.qs
posterior_z.normalize() 
```

```py
7.772927524715933 
```

这就是后验分布的样子。

```py
pmf_z.plot(label='prior', color='C5')
posterior_z.plot(label='posterior', color='C4')

decorate(xlabel='Time (min)',
         ylabel='PDF',
         title='Distribution of time between trains') 
```

![图片](img/f151fe5e310d26189b4c0ccf0015adea.png)

因为我更有可能在较长的间隔期间到达，所以分布向右移。先验均值约为 7.8 分钟；后验均值约为 8.9 分钟。

```py
pmf_z.mean(), posterior_z.mean() 
```

```py
(7.772927524715933, 8.89677416786441) 
```

这种转变是“检验悖论”的一个例子，[我写了一篇文章](https://towardsdatascience.com/the-inspection-paradox-is-everywhere-2ef1c2e9d709)。

顺便说一句，红线的时间表报告称高峰时段每 9 分钟有一班火车。这接近后验均值，但高于先验均值。我和 MBTA 的一位代表交换了电子邮件，他确认报告的火车之间的时间故意保守，以应对变化。

## 经过的时间

经过的时间，我称之为`x`，是上一班火车到达和乘客到达之间的时间。等待时间，我称之为`y`，是乘客到达和下一班火车到达之间的时间。我选择这种符号是为了

```py
z = x + y. 
```

根据`z`的分布，我们可以计算`x`的分布。我将从一个简单的情况开始，然后推广。假设两列火车之间的间隙是 5 分钟或 10 分钟，概率相等。

如果我们在随机时间到达，有 1/3 的概率在 5 分钟的间隙到达，有 2/3 的概率在 10 分钟的间隙到达。

如果我们在 5 分钟的间隙到达，`x`在 0 到 5 分钟之间是均匀分布的。如果我们在 10 分钟的间隙到达，`x`在 0 到 10 分钟之间是均匀分布的。因此等待时间的分布是两个均匀分布的加权混合。

更一般地，如果我们有`z`的后验分布，我们可以通过制作均匀分布的混合来计算`x`的分布。我们将使用以下函数来制作均匀分布。

```py
from empiricaldist import Pmf

def make_elapsed_dist(gap, qs):
    qs = qs[qs <= gap]
    n = len(qs)
    return Pmf(1/n, qs) 
```

`make_elapsed_dist`接受一个假设的间隙和一系列可能的时间。它选择小于或等于`gap`的经过时间，并将它们放入代表均匀分布的`Pmf`中。

我将使用这个函数来生成`posterior_z`中每个间隙的`Pmf`对象序列。

```py
qs = posterior_z.qs
pmf_seq = [make_elapsed_dist(gap, qs) for gap in qs] 
```

这是一个代表从 0 到 0.6 分钟的均匀分布的例子。

```py
pmf_seq[3] 
```

|  | 概率 |
| --- | --- |
| 0.0 | 0.25 |
| 0.2 | 0.25 |
| 0.4 | 0.25 |
| 0.6 | 0.25 |

序列的最后一个元素是从 0 到 20 分钟的均匀分布。

```py
pmf_seq[-1].plot()

decorate(xlabel='Time (min)',
         ylabel='PDF',
         title='Distribution of wait time in 20 min gap') 
```

![_images/e04c1eca3ff2aea714e04a33a17b2de9c6cc50b1c57b7713bc9cae27a7752c73.png](img/5cd0ae13b62935fcd0aa0d8aab31d774.png)

现在我们可以使用`make_mixture`来制作均匀分布的加权混合，其中权重是`posterior_z`的概率。

```py
from utils import make_mixture

pmf_x = make_mixture(posterior_z, pmf_seq) 
```

```py
pmf_z.plot(label='prior gap', color='C5')
posterior_z.plot(label='posterior gap', color='C4')
pmf_x.plot(label='elapsed time', color='C1')

decorate(xlabel='Time (min)',
         ylabel='PDF',
         title='Distribution of gap and elapsed times') 
```

![_images/6dea62010873c6070d43c669b7d6fed2488b72f85a7a92175e24eae25cedba65.png](img/b5912e612f0d114e052c5984a1af004b.png)

```py
posterior_z.mean(), pmf_x.mean() 
```

```py
(8.89677416786441, 4.448387083932206) 
```

平均经过时间是 4.4 分钟，是`z`的后验均值的一半。这是有道理的，因为我们期望平均来说在间隙的中间到达。

## 计算乘客

现在让我们考虑站台上等待的乘客数量。假设乘客在任何时间到达的可能性相等，并且以已知的速率`λ`到达，该速率为每分钟 2 名乘客。

在这些假设下，到达`x`分钟的乘客数量遵循参数为`λ x`的泊松分布。因此，我们可以使用 SciPy 函数`poisson`来计算每个可能的`x`值的 10 名乘客的似然。

```py
from scipy.stats import poisson

lam = 2
num_passengers = 10
likelihood = poisson(lam * pmf_x.qs).pmf(num_passengers) 
```

有了这个似然，我们可以计算`x`的后验分布。

```py
posterior_x = pmf_x * likelihood
posterior_x.normalize() 
```

```py
0.04757676716097805 
```

看起来是这样的：

```py
pmf_x.plot(label='prior', color='C1')
posterior_x.plot(label='posterior', color='C2')

decorate(xlabel='Time (min)',
         ylabel='PDF',
         title='Distribution of time since last train') 
```

![_images/32a1f7d633afb2a26c83c036ec21abe5835f2647577f4f38059cc02c3083bbe9.png](img/a724e2ebba97e3d3b3149c940de6d5b5.png)

根据乘客数量，我们认为距离上一班火车已经大约 5 分钟。

```py
pmf_x.mean(), posterior_x.mean() 
```

```py
(4.448387083932206, 5.1439350761797495) 
```

## 等待时间

现在我们认为下一班火车还要多久？根据我们目前所知，`z`的分布是`posterior_z`，`x`的分布是`posterior_x`。记住我们定义了

```py
z = x + y 
```

如果我们知道`x`和`z`，我们可以计算

```py
y = z - x 
```

所以我们可以使用`sub_dist`来计算`y`的分布。

```py
posterior_y = Pmf.sub_dist(posterior_z, posterior_x) 
```

嗯，几乎。该分布包含一些负值，这是不可能的。但我们可以移除它们并重新归一化，就像这样：

```py
nonneg = (posterior_y.qs >= 0)
posterior_y = Pmf(posterior_y[nonneg])
posterior_y.normalize() 
```

```py
0.8900343090047254 
```

根据目前的信息，这里是`x`，`y`和`z`的分布，显示为 CDFs。

```py
posterior_x.make_cdf().plot(label='posterior of x', color='C2')
posterior_y.make_cdf().plot(label='posterior of y', color='C3')
posterior_z.make_cdf().plot(label='posterior of z', color='C4')

decorate(xlabel='Time (min)',
         ylabel='PDF',
         title='Distribution of elapsed time, wait time, gap') 
```

![_images/61d9287b6df210c1961fdd80866bb33aed5bd0f21854fe0ae10d4fbf5e6cda7d.png](img/75e81c84847b0f9a0cbfd30dae478be9.png)

由于四舍五入误差，`posterior_y`包含`posterior_x`和`posterior_z`中没有的数量；这就是为什么我将其绘制为 CDF，以及为什么它看起来不平滑。

## 决策分析

在这一点上，我们可以使用站台上的乘客数量来预测等待时间的分布。现在让我们来回答问题的第二部分：我应该在什么时候停止等待火车，转而打车呢？

请记住，在原始情景中，我试图去南站赶通勤列车。假设我离开办公室的时间足够长，可以等待 15 分钟，仍然能在南站换乘。

在这种情况下，我想知道`y`超过 15 分钟的概率作为`num_passengers`的函数。为了回答这个问题，我们可以使用`num_passengers`的范围运行上一节的分析。

但是有一个问题。分析对长时间延迟的频率很敏感，因为长时间延迟很少，很难估计它们的频率。

我只有一周的数据，我观察到的最长延迟是 15 分钟。因此，我无法准确估计更长延迟的频率。

然而，我可以使用以前的观察来至少粗略估计。当我每天乘坐红线通勤一年时，我看到了由信号问题、停电和另一个站点的“警方活动”引起的三次长时间延迟。因此，我估计每年大约有 3 次重大延误。

但请记住，我的观察是有偏见的。我更有可能观察到长时间延迟，因为它们影响了大量的乘客。因此，我们应该将我的观察视为`posterior_z`的样本，而不是`pmf_z`的样本。

这是我们如何使用一些关于长时间延迟的假设来增加间隔时间的观察分布。从`posterior_z`中，我将抽取 260 个值的样本（大约是一年的工作日数）。然后我将添加 30、40 和 50 分钟的延迟（我一年观察到的长时间延迟的数量）。

```py
sample = posterior_z.sample(260)
delays = [30, 40, 50]
augmented_sample = np.append(sample, delays) 
```

我将使用这个增强的样本来对`z`的后验分布进行新的估计。

```py
qs = np.linspace(0, 60, 101)
augmented_posterior_z = kde_from_sample(augmented_sample, qs) 
```

这就是它的样子。

```py
augmented_posterior_z.plot(label='augmented posterior of z', color='C4')

decorate(xlabel='Time (min)',
         ylabel='PDF',
         title='Distribution of time between trains') 
```

![_images/7bfbd95ae9bd4887c57bc32400c1dec1dbeb8652fa9241dc818eacb979846646.png](img/f95937bcdf0834cd93a8fc34d62d2008.png)

现在让我们把前面章节的分析封装成一个函数。

```py
qs = augmented_posterior_z.qs
pmf_seq = [make_elapsed_dist(gap, qs) for gap in qs]
pmf_x = make_mixture(augmented_posterior_z, pmf_seq)
lam = 2
num_passengers = 10

def compute_posterior_y(num_passengers):   
  """Distribution of wait time based on `num_passengers`."""
    likelihood = poisson(lam * qs).pmf(num_passengers)
    posterior_x = pmf_x * likelihood
    posterior_x.normalize()
    posterior_y = Pmf.sub_dist(augmented_posterior_z, posterior_x)
    nonneg = (posterior_y.qs >= 0)
    posterior_y = Pmf(posterior_y[nonneg])
    posterior_y.normalize()
    return posterior_y 
```

根据我们到达车站时的乘客数量，计算`y`的后验分布。例如，如果我们看到 10 名乘客，这是等待时间的分布。

```py
posterior_y = compute_posterior_y(10) 
```

我们可以用它来计算平均等待时间和等待时间超过 15 分钟的概率。

```py
posterior_y.mean() 
```

```py
4.774817797206827 
```

```py
1 - posterior_y.make_cdf()(15) 
```

```py
0.014549512746375837 
```

如果我们看到 10 名乘客，我们预计等待时间会略少于 5 分钟，等待超过 15 分钟的几率约为 1%。

让我们看看如果我们扫描`num_passengers`的一系列值会发生什么。

```py
nums = np.arange(0, 37, 3)
posteriors = [compute_posterior_y(num) for num in nums] 
```

这是等待时间的平均值作为乘客数量的函数。

```py
mean_wait = [posterior_y.mean()
             for posterior_y in posteriors] 
```

```py
import matplotlib.pyplot as plt

plt.plot(nums, mean_wait)

decorate(xlabel='Number of passengers',
         ylabel='Expected time until next train',
         title='Expected wait time based on number of passengers') 
```

![_images/de8465ec97e738535f8ac65691bea6648336280ca606ad9c716453477a6eb069.png](img/5e37984f60418e47375174ccc8603cdf.png)

如果我到达站台时没有乘客，我推断我刚错过了一班火车；在这种情况下，期望的等待时间是`augmented_posterior_z`的均值。

我看到的乘客越多，我认为自上一班火车以来的时间越长，下一班火车到达的可能性就越大。

但只有到一定程度。如果站台上的乘客超过 30 人，这表明有长时间延迟，预期的等待时间开始增加。

现在这里是等待时间超过 15 分钟的概率。

```py
prob_late = [1 - posterior_y.make_cdf()(15) 
             for posterior_y in posteriors] 
```

```py
plt.plot(nums, prob_late)

decorate(xlabel='Number of passengers',
         ylabel='Probability of being late',
         title='Probability of being late based on number of passengers') 
```

![_images/a87493d6a5550a6e781319f383c1cbe9de96056c8d7a72a31c5b2d216b9adac1.png](img/92079aa07aadc5383a5fa67cef4c049b.png)

当乘客数量少于 20 时，我们推断系统正常运行，因此长时间延迟的概率很小。如果有 30 名乘客，我们怀疑出了问题，并且预计会有更长的延迟。

如果我们愿意接受在南站错过连接的概率为 5%，我们应该等待，直到乘客少于 30 人，如果超过这个数量就应该乘坐出租车。

或者，为了进一步分析，我们可以量化错过连接的成本和乘坐出租车的成本，然后选择最小化预期成本的阈值。

这种分析是基于到达率`lam`已知的假设。如果它不是精确已知的，而是从数据中估计出来的，我们可以用分布表示对`lam`的不确定性，计算每个`lam`值的`y`的分布，并进行混合以表示`y`的分布。我在《Bayes 思维》第一版中的这个问题的版本中做过这样的处理；我在这里没有提到，因为这不是问题的重点。
