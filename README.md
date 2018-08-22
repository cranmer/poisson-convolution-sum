# An infinite sum of Poisson-weighted convolutions

Kyle Cranmer, Aug 2018

If viewing on GitHub, this looks better with nbviewer: [click here](http://nbviewer.jupyter.org/github/cranmer/poisson-convolution-sum/blob/master/Poisson-weighted-convolutions.ipynb)

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/cranmer/poisson-convolution-sum/master?filepath=Poisson-weighted-convolutions.ipynb)

Consider a variable $x = \sum_{i=1}^n z_i$, where the $z_i \sim p(z|\theta)$ are iid and $n \sim Pois(\nu)$. For each value of $n$, the distribution $p(x|n)$ is given by repeated convolutions
\begin{equation}
{p(x|n, \theta)} = p(z|\theta)\underbrace{\star \dots \star}_\text{n times} p(z|\theta)
\end{equation}
and for a Poisson mean of $\nu$ the expected distribution for $x$ is  
\begin{equation}
{p(x|\nu, \theta)} = \sum_{n=0}^\infty e^{-\nu} \frac{\nu^n}{n!} p(x|n, \theta)
\end{equation}

The infinite tower of Poisson-weighted convolutions can be computed efficiently with a nice trick documented in *Analytic Confidence Level Calculations using the Likelihood Ratio and Fourier Transform* by Hongbo Hu and Jason Nielsen https://arxiv.org/pdf/physics/9906010.pdf. See also [this old paper](https://arxiv.org/abs/physics/0312050) and this [code](http://phystat.org/phystat/packages/0703002.1.html) for a C++ implementation.

First we take advantage of the convolution theorem relating convolutions to multiplication in the Fourier domain, denoted with a bar.
\begin{equation}
\overline{p(x|n, \theta)} = \overline{p(z|\theta)\underbrace{\star \dots \star}_\text{n times} z|\theta} = \left[\overline{p(z|\theta)} \right]^n 
\end{equation}
and then we can compress the infinite sum into the following exponential
\begin{equation}
\overline{p(x|\nu, \theta)} = \sum_{n=0}^\infty e^{-\nu} \frac{\nu^n}{n!}  \overline{p(x|n)}  =  \exp\left( \nu \left[ \overline{p(z|\theta)}-1 \right] \right)
\end{equation}

In the code below, we will reimplement the technique using pytorch. An interesting feature of implementing this in pytorch is that we can backprop through the inverse FFT, the exponentiation, the multplication, the subtraction, and forward FFT to calculate the gradient
\begin{equation}
\nabla_\theta\, p(x|\nu, \theta)
\end{equation}
and 
\begin{equation}
\nabla_\nu \, p(x|\nu, \theta)
\end{equation}.