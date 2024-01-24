# 分层模型的网格算法

> 原文：[`allendowney.github.io/ThinkBayes2/hospital.html`](https://allendowney.github.io/ThinkBayes2/hospital.html)

版权所有 2021 Allen B. Downey

许可证：[署名-非商业性使用-相同方式共享 4.0 国际 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

人们普遍认为网格算法只适用于具有 1-3 个参数的模型，或者如果你小心的话，可能是 4-5 个参数。[我自己也这么说过](https://allendowney.github.io/ThinkBayes2/chap19.html)。

但最近，我使用了一个网格算法来解决[发射器-探测器问题](https://www.allendowney.com/blog/2021/09/05/emitter-detector-redux/)，在解决问题的过程中，我注意到了问题的结构：尽管模型有两个参数，但数据只依赖于其中一个。这使得能够非常有效地评估似然函数并更新模型。

许多分层模型具有类似的结构：数据依赖于少量参数，这些参数依赖于少量超参数。我想知道是否相同的方法会推广到更复杂的模型，事实上确实如此。

例如，在这个笔记本中，我将使用一个 logitnormal-二项式分层模型来解决一个具有两个超参数和 13 个参数的问题。网格算法不仅实用；而且比 MCMC 快得多。

以下是我将要使用的一些实用函数。

```py
import matplotlib.pyplot as plt

def legend(**options):
  """Make a legend only if there are labels."""
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels):
        plt.legend(**options) 
```

```py
def decorate(**options):
    plt.gca().set(**options)
    legend()
    plt.tight_layout() 
```

```py
from empiricaldist import Cdf

def compare_cdf(pmf, sample):
    pmf.make_cdf().step(label='grid')
    Cdf.from_seq(sample).plot(label='mcmc')
    print(pmf.mean(), sample.mean())
    decorate() 
```

```py
from empiricaldist import Pmf

def make_pmf(ps, qs, name):
    pmf = Pmf(ps, qs)
    pmf.normalize()
    pmf.index.name = name
    return pmf 
```

## 心脏病发作数据

我将要解决的问题基于[《概率与贝叶斯建模》第十章](https://bayesball.github.io/BOOK/bayesian-hierarchical-modeling.html#example-deaths-after-heart-attack)；它使用了纽约市各医院治疗患者心脏病发作死亡率的数据。

我们可以使用 Pandas 将数据读入 `DataFrame`。

```py
import os

filename = 'DeathHeartAttackManhattan.csv'
if not os.path.exists(filename):
    !wget  https://github.com/AllenDowney/BayesianInferencePyMC/raw/main/DeathHeartAttackManhattan.csv 
```

```py
import pandas as pd

df = pd.read_csv(filename)
df 
```

|  | 医院 | 病例 | 死亡 | 死亡率 |
| --- | --- | --- | --- | --- |
| 0 | Bellevue 医院中心 | 129 | 4 | 3.101 |
| 1 | 哈莱姆医院中心 | 35 | 1 | 2.857 |
| 2 | 莱诺克斯山医院 | 228 | 18 | 7.894 |
| 3 | 大都会医院中心 | 84 | 7 | 8.333 |
| 4 | 山西贝斯以色列 | 291 | 24 | 8.247 |
| 5 | 山西医院 | 270 | 16 | 5.926 |
| 6 | 罗斯福山西医院 | 46 | 6 | 13.043 |
| 7 | 圣卢克斯山西医院 | 293 | 19 | 6.485 |
| 8 | 纽约大学医院中心 | 241 | 15 | 6.224 |
| 9 | NYP 医院 - 艾伦医院 | 105 | 13 | 12.381 |
| 10 | NYP 医院 - 哥伦比亚长老中心 | 353 | 25 | 7.082 |
| 11 | NYP 医院 - 纽约威尔康奈尔中心 | 250 | 11 | 4.400 |
| 12 | NYP/曼哈顿医院 | 41 | 4 | 9.756 |

我们需要的列是 `病例`，即每家医院治疗的患者数量，以及 `死亡`，即这些患者中死亡的数量。

```py
data_ns = df['Cases'].values
data_ks = df['Deaths'].values 
```

## PyMC 解决方案

这是一个分层模型，它估计了每家医院的死亡率，并同时估计了各家医院的死亡率分布。

```py
import pymc3 as pm

def make_model():
    with pm.Model() as model:
        mu = pm.Normal('mu', 0, 2)
        sigma = pm.HalfNormal('sigma', sigma=1)
        xs = pm.LogitNormal('xs', mu=mu, sigma=sigma, shape=len(data_ns))
        ks = pm.Binomial('ks', n=data_ns, p=xs, observed=data_ks)
    return model 
```

```py
%time model = make_model()
pm.model_to_graphviz(model) 
```

```py
CPU times: user 875 ms, sys: 51.7 ms, total: 927 ms
Wall time: 2.22 s 
```

![_images/4aa477637d76c73e22c5ba5569dcd0f6ab56abcc71deb97593454b7f77dbc822.svg](img/cee02df9a8687ea1e007d9785e5ee05e.png)

```py
with model:
    pred = pm.sample_prior_predictive(1000)
    %time trace = pm.sample(500, target_accept=0.97) 
```

```py
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [xs, sigma, mu] 
```

<progress value="6000" class="" max="6000" style="width:300px; height:20px; vertical-align: middle;">100.00% [6000/6000 00:07<00:00 Sampling 4 chains, 10 divergences]</progress>

```py
Sampling 4 chains for 1_000 tune and 500 draw iterations (4_000 + 2_000 draws total) took 8 seconds.
There were 2 divergences after tuning. Increase `target_accept` or reparameterize.
The acceptance probability does not match the target. It is 0.9060171753417431, but should be close to 0.97\. Try to increase the number of tuning steps.
There was 1 divergence after tuning. Increase `target_accept` or reparameterize.
There were 7 divergences after tuning. Increase `target_accept` or reparameterize.
The acceptance probability does not match the target. It is 0.9337619072936738, but should be close to 0.97\. Try to increase the number of tuning steps.
The estimated number of effective samples is smaller than 200 for some parameters. 
```

```py
CPU times: user 5.12 s, sys: 153 ms, total: 5.27 s
Wall time: 12.3 s 
```

公平地说，PyMC 对这种参数化不太喜欢（尽管我不确定为什么）。在大多数运行中，有一定数量的发散。即便如此，结果还是足够好的。

这是超参数的后验分布。

```py
import arviz as az

with model:
    az.plot_posterior(trace, var_names=['mu', 'sigma']) 
```

![_images/802eebbd477f71df4cbd9a340d2b5613eb821486aecd59aa0aa52a13b50925b4.png](img/12fd7003a8a0f8644da27de2aabe7a95.png)

我们可以提取 x 的后验分布。

```py
trace_xs = trace['xs'].transpose()
trace_xs.shape 
```

```py
(13, 2000) 
```

例如，这是第一个医院 x 的后验分布。

```py
with model:
    az.plot_posterior(trace_xs[0]) 
```

![_images/548f75d0a6ec0b23bac76b1316b5db52bedb7da62330a04f61a98e13bdf8ef4a.png](img/ae8b4180c3aa20fb7981df69ff63edce.png)

## 网格先验

现在让我们使用网格算法解决同样的问题。我将使用相同的超参数先验，用每个维度约 100 个元素的网格来近似。

```py
import numpy as np
from scipy.stats import norm

mus = np.linspace(-6, 6, 101)
ps = norm.pdf(mus, 0, 2)
prior_mu = make_pmf(ps, mus, 'mu')

prior_mu.plot()
decorate(title='Prior distribution of mu') 
```

![_images/4d36a111de94173800e762bcb4e11959161323a032fb787907ef801163972d21.png](img/b1f2ea3c74c546eb75cd9df07609763d.png)

```py
from scipy.stats import logistic

sigmas = np.linspace(0.03, 3.6, 90)
ps = norm.pdf(sigmas, 0, 1)
prior_sigma = make_pmf(ps, sigmas, 'sigma')

prior_sigma.plot()
decorate(title='Prior distribution of sigma') 
```

![_images/c5a4ef41048db2671496eab57fce6d329cb8f96b1be3d92a1916c1150e1a9737.png](img/f2dd9988c740ab93a9771e1c21d328d9.png)

以下单元格证实这些先验与 PyMC 的先验样本一致。

```py
compare_cdf(prior_mu, pred['mu'])
decorate(title='Prior distribution of mu') 
```

```py
2.6020852139652106e-18 -0.06372282505953483 
```

![_images/87280efa676b32352a19b7704ee06b02e34ee045e7ef7f4831fdb67fa9c344d0.png](img/bf43b5d007ea8420fe02af58fa6c5eb7.png)

```py
compare_cdf(prior_sigma, pred['sigma'])
decorate(title='Prior distribution of sigma') 
```

```py
0.8033718951689776 0.8244605687886865 
```

![_images/ce51b8b82a93033e1b1d67375bedb99b23fa5283788710bd0205144e236ed498.png](img/d38b70b734ab5e74690ed941974b4d5e.png)

## 超参数的联合分布

我将使用`make_joint`来创建一个表示超参数联合先验分布的数组。

```py
def make_joint(prior_x, prior_y):
    X, Y = np.meshgrid(prior_x.ps, prior_y.ps, indexing='ij')
    hyper = X * Y
    return hyper 
```

```py
prior_hyper = make_joint(prior_mu, prior_sigma)
prior_hyper.shape 
```

```py
(101, 90) 
```

这就是它的样子。

```py
import pandas as pd
from utils import plot_contour

plot_contour(pd.DataFrame(prior_hyper, index=mus, columns=sigmas))
decorate(title="Joint prior of mu and sigma") 
```

![_images/958b2a9ea3b4c8c4b4f3da5eb4ab7ec508f51a23ee621cafca7af2836e950632.png](img/00ac59083dbac51dc0a419beb53d318e.png)

## 超参数和 x 的联合先验

现在我们准备好为 x 布置网格，这是我们将为每个医院估计的比例。

```py
xs = np.linspace(0.01, 0.99, 295) 
```

对于每对超参数，我们将计算`x`的分布。

```py
from scipy.special import logit

M, S, X = np.meshgrid(mus, sigmas, xs, indexing='ij')
LO = logit(X)
LO.sum() 
```

```py
-6.440927791118156e-10 
```

```py
from scipy.stats import norm

%time normpdf = norm.pdf(LO, M, S)
normpdf.sum() 
```

```py
CPU times: user 69.6 ms, sys: 16.5 ms, total: 86.1 ms
Wall time: 84.9 ms 
```

```py
214125.5678798693 
```

我们可以通过计算不依赖于 x 的项来加快速度

```py
%%time

z = (LO-M) / S
normpdf = np.exp(-z**2/2) 
```

```py
CPU times: user 26 ms, sys: 10.6 ms, total: 36.6 ms
Wall time: 35.1 ms 
```

结果是一个带有 mu、sigma 和 x 轴的 3D 数组。

现在我们需要对每个`x`的分布进行归一化。

```py
totals = normpdf.sum(axis=2)
totals.shape 
```

```py
(101, 90) 
```

为了归一化，我们必须使用`divide`的安全版本，其中`0/0`是`0`。

```py
def divide(x, y):
    out = np.zeros_like(x)
    return np.divide(x, y, out=out, where=(y!=0)) 
```

```py
shape = totals.shape + (1,)
normpdf = divide(normpdf, totals.reshape(shape))
normpdf.shape 
```

```py
(101, 90, 295) 
```

结果是一个包含每对超参数的`x`分布的数组。

现在，为了得到先验分布，我们需要通过超参数的联合分布进行乘法。

```py
def make_prior(hyper):

    # reshape hyper so we can multiply along axis 0
    shape = hyper.shape + (1,)
    prior = normpdf * hyper.reshape(shape)

    return prior 
```

```py
%time prior = make_prior(prior_hyper)
prior.sum() 
```

```py
CPU times: user 5.57 ms, sys: 0 ns, total: 5.57 ms
Wall time: 4.87 ms 
```

```py
0.999937781278039 
```

结果是一个表示`mu`、`sigma`和`x`的联合先验分布的 3D 数组。

为了检查它是否正确，我将提取边缘分布并将其与先验进行比较。

```py
def marginal(joint, axis):
    axes = [i for i in range(3) if i != axis]
    return joint.sum(axis=tuple(axes)) 
```

```py
prior_mu.plot()
marginal_mu = Pmf(marginal(prior, 0), mus)
marginal_mu.plot()
decorate(title='Checking the marginal distribution of mu') 
```

![_images/34c43829d056d3ea02a3b60bb7ae8a85900c7622aad347608b98ca80caa33779.png](img/e5b39fcfba081fb44539bcc089de8761.png)

```py
prior_sigma.plot()
marginal_sigma = Pmf(marginal(prior, 1), sigmas)
marginal_sigma.plot()
decorate(title='Checking the marginal distribution of sigma') 
```

![_images/d0686bec19a94e62e2430a4f3992ecb72ff02e44030ef23a4d6856232389b10f.png](img/7735daeec9faf0628ff06fd7a96bac42.png)

我们没有明确计算`x`的先验分布；它是由超参数的分布得出的。但我们可以从联合先验中提取`x`的先验边缘。

```py
marginal_x = Pmf(marginal(prior, 2), xs)
marginal_x.plot()
decorate(title='Checking the marginal distribution of x',
         ylim=[0, np.max(marginal_x) * 1.05]) 
```

![_images/bdf43d9b32661a9cbb2dba60731aee477114f535ff9dde513a8f60a814568cd9.png](img/8b344dcdb4ee3fc57ed2b4d87590458a.png)

并将其与 PyMC 的先验样本进行比较。

```py
pred_xs = pred['xs'].transpose()
pred_xs.shape 
```

```py
(13, 1000) 
```

```py
compare_cdf(marginal_x, pred_xs[0])
decorate(title='Prior distribution of x') 
```

```py
0.49996889063901967 0.4879934000104224 
```

![_images/4a08494ee22a1cd5637087377ca5d42275a09b72ebe2a0bcab09d4743c7880bf.png](img/3cca26cd857d171029edd0e7c4a4e10b.png)

我从网格中得到的`x`的先验分布与我从 PyMC 中得到的有点不同。我不确定为什么，但似乎并不影响结果太多。

除了边缘之外，我们还会发现从超参数的联合边缘分布中提取是有用的。

```py
def get_hyper(joint):
    return joint.sum(axis=2) 
```

```py
hyper = get_hyper(prior) 
```

```py
plot_contour(pd.DataFrame(hyper, 
                          index=mus, 
                          columns=sigmas))
decorate(title="Joint prior of mu and sigma") 
```

![_images/ac19e03323b4eb0f5f7faff979e4b93a985cd85a2344b9596e03357d0b38d5fd.png](img/e4a58fb5c6bad02db959c56ff3784715.png)

## 更新

数据的似然性只取决于`x`，所以我们可以这样计算。

```py
from scipy.stats import binom

data_k = data_ks[0]
data_n = data_ns[0]

like_x = binom.pmf(data_k, data_n, xs)
like_x.shape 
```

```py
(295,) 
```

```py
plt.plot(xs, like_x)
decorate(title='Likelihood of the data') 
```

![_images/1d1840c79bc6a6af27158c4e2a15b8e4f57e29e8cef2496426af8782521d9b97.png](img/60671e80e622b1a3551d6e67df9ceb00.png)

这是更新。

```py
def update(prior, data):
    n, k = data
    like_x = binom.pmf(k, n, xs)
    posterior = prior * like_x
    posterior /= posterior.sum()
    return posterior 
```

```py
data = data_n, data_k
%time posterior = update(prior, data) 
```

```py
CPU times: user 11.6 ms, sys: 11.9 ms, total: 23.5 ms
Wall time: 7.66 ms 
```

## 串行更新

在这一点上，我们可以根据单个医院进行更新，但如何根据所有医院进行更新呢？

作为得到正确答案的一步，我将从错误答案开始，即逐个更新。

每次更新后，我们提取超参数的后验分布并用它来创建下一个更新的先验。

最后，超参数的后验分布是正确的，最后一个医院的`x`的边际后验是正确的，但其他边缘是错误的，因为它们没有考虑来自后续医院的数据。

```py
def multiple_updates(prior, ns, ks):
    for data in zip(ns, ks):
        print(data)
        posterior = update(prior, data)
        hyper = get_hyper(posterior)
        prior = make_prior(hyper)
    return posterior 
```

```py
%time posterior = multiple_updates(prior, data_ns, data_ks) 
```

```py
(129, 4)
(35, 1)
(228, 18)
(84, 7)
(291, 24)
(270, 16)
(46, 6)
(293, 19)
(241, 15)
(105, 13)
(353, 25)
(250, 11)
(41, 4)
CPU times: user 185 ms, sys: 35.4 ms, total: 220 ms
Wall time: 172 ms 
```

以下是超参数的后验分布，与 PyMC 的结果进行比较。

```py
marginal_mu = Pmf(marginal(posterior, 0), mus)
compare_cdf(marginal_mu, trace['mu']) 
```

```py
-2.6478808810110768 -2.5956645549514694 
```

![_images/34b5f3aae609642555c9a66934a9f59842750d058cd635e3775a6350d7035c40.png](img/b79b5761da1d1849da0c955187b9f5b9.png)

```py
marginal_sigma = Pmf(marginal(posterior, 1), sigmas)
compare_cdf(marginal_sigma, trace['sigma']) 
```

```py
0.19272226451430116 0.18501785022543282 
```

![_images/3761044539e7fc0be684821dbb5b0ea13c2ebcb0327a8a976943b7d8ef5cd7fe.png](img/37314e08f0517a5a6d7c9a9e027ee566.png)

```py
marginal_x = Pmf(marginal(posterior, 2), xs)
compare_cdf(marginal_x, trace_xs[-1]) 
```

```py
0.07330826956150183 0.07297933578329886 
```

![_images/a542fa2d10fed87e7ed4b8eaf3608ebe1f6b56281ed16d700cd0f002bf301bd9.png](img/a4c8b46af5695de7ebe28e6536959548.png)

## 并行更新

逐个更新并不完全正确，但它给了我们一个洞察。

假设我们从超参数的均匀分布开始，并使用来自一家医院的数据进行更新。如果我们提取超参数的后验联合分布，我们得到的是与一个数据集相关联的似然函数。

以下函数计算这些似然函数并将它们保存在一个名为`hyper_likelihood`的数组中。

```py
def compute_hyper_likelihood(ns, ks):
    shape = ns.shape + mus.shape + sigmas.shape
    hyper_likelihood = np.empty(shape)

    for i, data in enumerate(zip(ns, ks)):
        print(data)
        n, k = data
        like_x = binom.pmf(k, n, xs)
        posterior = normpdf * like_x
        hyper_likelihood[i] = get_hyper(posterior)
    return hyper_likelihood 
```

```py
%time hyper_likelihood = compute_hyper_likelihood(data_ns, data_ks) 
```

```py
(129, 4)
(35, 1)
(228, 18)
(84, 7)
(291, 24)
(270, 16)
(46, 6)
(293, 19)
(241, 15)
(105, 13)
(353, 25)
(250, 11)
(41, 4)
CPU times: user 82 ms, sys: 55.2 ms, total: 137 ms
Wall time: 75.5 ms 
```

我们可以将其相乘以得到似然的乘积。

```py
%time hyper_likelihood_all = hyper_likelihood.prod(axis=0)
hyper_likelihood_all.sum() 
```

```py
CPU times: user 279 µs, sys: 0 ns, total: 279 µs
Wall time: 158 µs 
```

```py
1.685854062633571e-14 
```

这很有用，因为它提供了一种有效的方法来计算任何医院的`x`的边际后验分布。以下是一个例子。

```py
i = 3
data = data_ns[i], data_ks[i]
data 
```

```py
(84, 7) 
```

假设我们按顺序进行更新，并将这家医院保存到最后。最终更新的先验分布将反映出所有先前医院的更新，我们可以通过除以`hyper_likelihood[i]`来计算。

```py
%time hyper_i = divide(prior_hyper * hyper_likelihood_all, hyper_likelihood[i])
hyper_i.sum() 
```

```py
CPU times: user 310 µs, sys: 147 µs, total: 457 µs
Wall time: 342 µs 
```

```py
4.3344287278716945e-17 
```

我们可以使用`hyper_i`来制作最后更新的先验。

```py
prior_i = make_prior(hyper_i) 
```

然后进行更新。

```py
posterior_i = update(prior_i, data) 
```

我们可以确认结果与 PyMC 的结果相似。

```py
marginal_mu = Pmf(marginal(posterior_i, 0), mus)
marginal_sigma = Pmf(marginal(posterior_i, 1), sigmas)
marginal_x = Pmf(marginal(posterior_i, 2), xs) 
```

```py
compare_cdf(marginal_mu, trace['mu']) 
```

```py
-2.647880881011078 -2.5956645549514694 
```

![_images/34b5f3aae609642555c9a66934a9f59842750d058cd635e3775a6350d7035c40.png](img/b79b5761da1d1849da0c955187b9f5b9.png)

```py
compare_cdf(marginal_sigma, trace['sigma']) 
```

```py
0.19272226451430124 0.18501785022543282 
```

![_images/3761044539e7fc0be684821dbb5b0ea13c2ebcb0327a8a976943b7d8ef5cd7fe.png](img/37314e08f0517a5a6d7c9a9e027ee566.png)

```py
compare_cdf(marginal_x, trace_xs[i]) 
```

```py
0.07245354421667904 0.07224440565018131 
```

![_images/a9d4fdfb10f48b1637dd49502a40eaa7edd8fbde43581c654b938fa0020c4977.png](img/a8eeefee425f21107c32bce6286b1094.png)

## 计算所有边际

以下函数计算所有医院的边际并将结果存储在一个数组中。

```py
def compute_all_marginals(ns, ks):
    shape = len(ns), len(xs)
    marginal_xs = np.zeros(shape)
    numerator = prior_hyper * hyper_likelihood_all

    for i, data in enumerate(zip(ns, ks)):
        hyper_i = divide(numerator, hyper_likelihood[i])
        prior_i = make_prior(hyper_i) 
        posterior_i = update(prior_i, data)
        marginal_xs[i] = marginal(posterior_i, 2)

    return marginal_xs 
```

```py
%time marginal_xs = compute_all_marginals(data_ns, data_ks) 
```

```py
CPU times: user 184 ms, sys: 49.8 ms, total: 234 ms
Wall time: 173 ms 
```

以下是结果的样子，与 PyMC 的结果进行比较。

```py
for i, ps in enumerate(marginal_xs):
    pmf = Pmf(ps, xs)
    plt.figure()
    compare_cdf(pmf, trace_xs[i])
    decorate(title=f'Posterior marginal of x for Hospital {i}',
             xlabel='Death rate',
             ylabel='CDF',
             xlim=[trace_xs[i].min(), trace_xs[i].max()]) 
```

```py
0.06123636407822421 0.0617519291444324
0.06653003152551518 0.06643868288267936
0.07267383211481376 0.07250041300148316
0.07245354421667904 0.07224440565018131
0.07430385699796423 0.07433369435815212
0.06606326919655045 0.06646020352443961
0.07774639529896528 0.07776805141855801
0.06788483681522386 0.06807113157490664
0.06723306224279789 0.06735326167909643
0.08183332535205982 0.08115900598539395
0.07003760661997555 0.0704088595242495
0.06136130741477605 0.06159674913422137
0.07330826956150185 0.07297933578329886 
```

![_images/d0edcb50f337ffa8fb1aa3438f6d6b3e834fec9120c52d0b0e7e6a084ee2e721.png](img/daf5c629fbde95f4d9f95cab77cacad4.png) ![_images/9f9b3321de2136f0646b22cb95a31ee10e7d6fb91f1aeb5879646fa92de663fa.png](img/7f1657ec693bfe615b3224818c87d9ea.png) ![_images/dd45e7a3b4c455b669c74c577c44986221d25ce2af5489c2735d21ec7ac1fbac.png](img/6c55a70f2feb1ea5d6148e5cbb5171b7.png) ![_images/1c2aeadaebdcc8c134743788e207023c8cd7d72f976460fbcd4c75d5e12ea2e5.png](img/bc572287debc3b9654705c55a87d3176.png) ![_images/8f723b526054d2ea3763233fe110e90964d903b48cfb6f4bc9dddb1c761ae0de.png](img/0ec18c9a451fcc08bc77fb5d0c5c808b.png) ![_images/4d8b18ec6de53ad0b15bb01a9db2eaa08692b0304a89adcdd98cb2454a56cd7f.png](img/9438b827d107bf0c2da0a578dbe9c6d0.png) ![_images/4d0b50ac9cd3057c970c57df8010f7500d3689cf99266d83a94913c1451bf50c.png](img/d50a9d9a3998a1c867f27d7e6a95d4e1.png) ![_images/c65d3bde651709a62e0b7a1f12e8cfabef2d99890b4b722dfadb090b572c9112.png](img/16d48694973aa93f1d5a39ab67eab690.png) ![_images/5e010692b56d2e899d5e37cfc93ef9d517b6d2d0434f5daac9f6dd4c969c4b0c.png](img/2e7aa725bbc93091d2c4daacd9b28299.png) ![_images/97f4239fe527906abe1e81365268c5abdb7ceee3fe173e0fafde560df3aa03c9.png](img/47643ee2ff73e24930532b10835a5c96.png) ![_images/4b1e0fea1bea326560809ea4054d54b76b1b6071a85fcd46203200293a708086.png](img/7bba56a4e32abae7a8fda5b85fb16658.png) ![_images/1aea59ec4b6b828f78566aa465c4c4c9c5f80b3391f07dccc183314b7ab43781.png](img/b22db57b14186121c858ff10f2036fb0.png) ![_images/c4b8c68ee0b70f97fdf8ae4c88b880c00ac5b27a77ddd157d95953c8307b1bc0.png](img/3b60d3e1a1173a8473c26ff0871015c1.png)

以下是网格算法和 PyMC 结果之间的百分比差异。其中大部分小于 1%。

```py
for i, ps in enumerate(marginal_xs):
    pmf = Pmf(ps, xs)
    diff = abs(pmf.mean() - trace_xs[i].mean()) / pmf.mean()
    print(diff * 100) 
```

```py
0.841926319383687
0.13730437329010417
0.23862662568368032
0.28865194761527047
0.04015586995533174
0.6008396688759207
0.027854821447936134
0.274427646029194
0.17878024931315142
0.8240155997142278
0.5300765148763152
0.38369736461746806
0.44869941709241024 
```

所有这些计算所需的总时间约为 300 毫秒，而构建和运行 PyMC 模型需要超过 10 秒。而且 PyMC 使用了 4 个核心，而我只用了一个。

网格算法易于并行化，并且是增量式的。如果你从新医院获取数据，或者为现有医院获取新数据，你可以：

1.  计算更新后医院`x`的后验分布，使用现有的其他医院的`hyper_likelihoods`。

1.  更新其他医院的`hyper_likelihoods`，并再次运行它们的更新。

总时间大约是从头开始所需时间的一半，而且很容易并行化。

网格算法的一个缺点是它生成了每家医院的边际分布，而不是它们所有的联合分布的样本。因此，很难看出它们之间的相关性。

另一个一般的缺点是设置网格算法需要更多的工作。如果我们切换到另一个参数化，改变 PyMC 模型会更容易。
