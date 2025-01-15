---
layout: post
title: Variational Autoencoder (VAE) Model - A Simplified Math Elaboration for VAE's Loss  
date: 2025-01-13 13:02:00-0400
inline: false
related_posts: false
giscus_comments: true 

---

In this post, I will elaborate the mathemtatics behind VAE's loss. This math proof is often conveniently ignored to faciliate the model adaptation. I, however, think it's helpful to fully understand VAE model and other laten variable models.

---
First, In addition to the [original VAE paper](https://arxiv.org/pdf/1312.6114), I would like to recommend this amazing lecture video from Stanford ([Variational Inference and Generative Models](https://www.youtube.com/watch?v=iL1c1KmYPM0&list=PLoROMvodv4rNjRoawgt72BBNwL2V7doGI&index=11). The course about meta-learning is also very well explained.

# VAE

VAE tries to model the distribution $$p(x)$$ of example data $$x$$ by assuming the Gaussian distribution of the data and then training a model (parameterized by $$\theta$$ ($$p_\theta(x)$$) to approximate the parameters (mean $$\mu_\theta$$  and variance $$\sigma_\theta$$) of the distribution. VAE however does not approximate these parameters directly from the training examples. It instead first learns an intermediate laten variable $$z$$ with prior Gaussian distribution $$p(z)$$ (also approximated as a Gaussian $$q_\phi(z)$$ parameterized by $$\mu_\phi$$ and $$\sigma_\phi$$) to represent the data (encoding step) and then learns $$\mu_\theta$$ and $$\sigma_\theta$$ from this $$z$$  (decoding process). This processs elaborates its name and why VAE is categorized as a Gaussian laten variable model.

$$
\begin{aligned}
  p_\theta(x) & = \int_z p_\theta(x|z).p(z)dz, \text(1*) \\
	 & = \int_z p_\theta(x|z).p(z).\frac{q_\phi(z)}{q_\phi(z)}dz, \text(2*) \\ 
	 & = E_{z\sim q_\phi(z)}\frac{p_\theta(x|z).p(z)}{q_\phi(z)}, \text( 3*) \\ 
 \end{aligned}
$$

>  * (1*): the general formula to compose a more complicated distribution $$p_\theta(x)$$ from two different distributions ($$p_\theta(x\|z)$$ and $$p(z)$$). 
>  * (2*): introduce another distribution to the equation by multiplying and dividing by the same distribution
>  * (3*): converting from integral to expectation using the relationship between integral and expectation, from which we can compute the distrubution $$p_\theta(x)$$ by sampling instead of evaluating the integral.

Applying log to both sides of equation (3*)

$$
\begin{aligned}
log(p_\theta(x)) & = log(E_{z\sim q_\phi(z)}\frac{p_\theta(x|z).p(z)}{q_\phi(z)}) \\
&    >= E_{z\sim q_\phi(z)}[log(p_\theta(x|z)) + log(p(z)) - log(q_\phi(z))], \text(4*)\\
& = ELBO
\end{aligned}
$$

> * (4*): Applying Jenshen Inequality $$log(E(y)) >= E(log(y))$$ and then applying $$log(a*b/c) = log(a)+ log(b) - log(c)$$
 
VAE trains a model to maximize $$log(p_\theta(x))$$ by maximizing the lower bound ELBO. Directly optimizing for $$log(p_\theta(x))$$ is extremely difficult because we need to sample lots of $$z$$. The intuition of VAE is to learn $$z$$ that most likely produces training examples $$x$$. Hence, to further reduce the search space for the ELBO optimization process, it's better to learn $$q_\phi(z)$$ conditioned on $$x$$, $$q_\phi(z\|x)$$. 

$$
 \begin{aligned}
 ELBO & =  E_{z\sim q_\phi(z|x)}[log(p_\theta(x|z)) + log(p(z)) - log(q_\phi(z|x))] \\
 & =E_{z\sim q_\phi(z|x)}[log(p_\theta(x|z))] - E_{z\sim q_\phi(z|x)}[log(\frac{q_\phi(z|x)}{p(z)}] (5*)\\ 
 & = E_{z\sim q_\phi(z|x)}[log(p_\theta(x|z))] - \sum_zq_\phi(z|x)*log(\frac{q_\phi(z|x)}{p(z)}) (6*)\\
 & =E_{z\sim q_\phi(z|x)}[log(p_\theta(x|z))] - D_{KL}[q_\phi(z|x)||p(z)] (7*)
 \end{aligned}
 $$
 
 > * (5*): $$E(x+y) = E(x) + E(y)$$
 > * (6*): Applying the definition of expected value. $$E(f(x)) = \sum_xf(x).p(x)$$, where $$p(x)$$ is the probability density function (pdf) of $$x$$.
 > * 7(*): KL divergence definition: $$D_{KL} (P\|Q) = \sum_xP(x)log(\frac{P(x)}{Q(x)})$$
 
One issue of this ELBO is the sampling of laten variable $$z$$ for the decoding process $$log(p_\theta(x\|z))$$ is not differential. To make ELBO optimizable using gradient descent, VAE applies the parameterization trick to the sampling process

$$
\begin{aligned}
z = \mu_\phi(x) + \epsilon *\sigma_\phi(x), \epsilon \sim N(0,1)
\end{aligned}
$$
 
$$\epsilon$$ is not dependent on $$\phi$$, so now we can compute gradient of ELBO with respect to $$\theta$$ and $$\phi$$ 

