# An infinite sum of Poisson-weighted convolutions

Kyle Cranmer, Aug 2018

If viewing on GitHub, this looks better with nbviewer: [click here](http://nbviewer.jupyter.org/github/cranmer/poisson-convolution-sum/blob/master/Poisson-weighted-convolutions.ipynb)

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/cranmer/poisson-convolution-sum/master?filepath=Poisson-weighted-convolutions.ipynb)

Consider a variable x that comes from a sum of n iid samples of z, where n is Poisson distributed. 
The distribution of x is given by an infinite sum of Poisson-weighted convolutions, which can be computed efficiently with a nice trick documented in *Analytic Confidence Level Calculations using the Likelihood Ratio and Fourier Transform* by Hongbo Hu and Jason Nielsen https://arxiv.org/pdf/physics/9906010.pdf. See also [this old paper](https://arxiv.org/abs/physics/0312050) and this [code](http://phystat.org/phystat/packages/0703002.1.html) for a C++ implementation.

First we take advantage of the convolution theorem relating convolutions to multiplication in the Fourier domain
and then we can compress the infinite sum into an exponential.

The notebook in this repository implements the technique using pytorch. An interesting feature of implementing this in pytorch is that we can backprop through the inverse FFT, the exponentiation, the multplication, the subtraction, and forward FFT to calculate the gradient with respect to the Poisson mean and the parameters for the distribution of z. The notebook demonstrates such a fit.