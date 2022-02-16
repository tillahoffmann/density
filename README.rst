(Conditional) density estimation in torch
=========================================

.. image:: https://github.com/tillahoffmann/density/actions/workflows/main.yml/badge.svg
  :target: https://github.com/tillahoffmann/density/actions/workflows/main.yml

Background
----------

A flow :math:`f:\mathcal{U}\rightarrow\mathcal{Y}` is a (parameterized) bijective transformation from a latent space :math:`\mathcal{U}` to a target space :math:`\mathcal{Y}`. We assume that the distribution :math:`p_\mathcal{U}` of the latent variable can be sampled from and that its density can be evaluated. The flow thus induces a distribution :math:`p_\mathcal{Y}` in the target space.

On the one hand, the flow allows us to sample from an "interesting" distribution by first sampling :math:`u\in\mathcal{U}` from the latent distribution :math:`p_\mathcal{U}` and then applying the transformation. We can thus evaluate the Kullback-Leibler divergence

.. math::
  :label: forward-kl

  D_\mathrm{KL}\left\{p_\mathcal{Y}\vert\vert p^*_\mathcal{Y}\right\}=\int dy\,p_\mathcal{Y}(y)\log\frac{p_\mathcal{Y}(y)}{p^*_\mathcal{Y}(y)}

from :math:`p_\mathcal{Y}` to some distribution of interest :math:`p^*_\mathcal{Y}`. In particular, the flow allows us to change variables of integration from :math:`y` to :math:`u`, and we note

.. math::
  :label: log-volume-change

  \log p_\mathcal{Y}(y) = \log p_\mathcal{U}(u) - \log\left\vert J\right\vert,

where :math:`J_{ij}=\frac{\partial f_i(u)}{\partial u_j}` is the Jacobian matrix of the flow. Substituting into the above expression yields

.. math::

  D_\mathrm{KL}\left\{p_\mathcal{Y}\vert\vert p^*_\mathcal{Y}\right\} &= \int du\,p_\mathcal{U}(u) \left[\log p_\mathcal{U}(u) + \log\left\vert\frac{\partial u}{\partial y}\right\vert -\log p_\mathcal{Y}^*(f(u))\right]\\
  &=\left\langle\log\left\vert\frac{\partial u}{\partial y}\right\vert -\log p_\mathcal{Y}^*(f(u))\right\rangle_{u\sim p_\mathcal{U}} - H\left\{p_\mathcal{U}\right\},

where :math:`H\left\{p_\mathcal{U}\right\}` is the entropy of the latent distribution and is independent of the flow for optimization purposes. This type of flow is useful, for example, for black-box variational inference by setting the target distribution to the joint distribution of data and parameters.

On the other hand, we can evaluate the density in the target space using the inverse flow :math:`f^{-1}:\mathcal{Y}\rightarrow\mathcal{U}` by substituting into :eq:`log-volume-change`. This allows us to evaluate the Kullback-Leibler divergence from a target distribution :math:`p^*_\mathcal{Y}` to :math:`p_\mathcal{Y}`, i.e. the reverse of :eq:`forward-kl`. In particular,

.. math::
  :label: reverse-kl

  D_\mathrm{KL}\left\{p^*_\mathcal{Y}\vert\vert p_\mathcal{Y}\right\}&=\int dy\,p^*_\mathcal{Y}(y)\log\frac{p^*_\mathcal{Y}(y)}{p_\mathcal{Y}(y)}\\
  &= \left\langle\log\left\vert J\right\vert - \log p_\mathcal{U}\left(f^{-1}(y)\right)\right\rangle_{y\sim p^*_\mathcal{Y}} -H\left\{p^*_\mathcal{Y}\right\},

where the entropy of the target distribution is a constant that does not affect the optimization of the flow. This type of flow is useful, for example, if we want to estimate a density given samples from the target distribution. In practice the expectations in :eq:`forward-kl` and :eq:`reverse-kl` are evaluated using a Monte Carlo estimate by sampling from the flow and using available samples from the target distribution, respectively.
