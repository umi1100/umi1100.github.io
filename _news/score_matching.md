---
layout: post
title: Score-based Generative Models, Part 1 Score Matching
date: 2025-04-29 16:02:00-0400
inline: false
related_posts: true 
giscus_comments: true
---

Score-based Generative Models (SGM) try to learn the data distribution $$p_{data}(x)$$ of training samples x by learning a model parameterized by $$\theta$$, $$s_\theta$$ to approximate the derivative of log of $$p_{data}(x)$$ with respect to the data $$x$$, $$\nabla_x{log(p_{data}(x))}$$. This derivative is referred to as the score of the probability density $$p_{data}(x)$$, which elaborates the name of the model. 

[Score matching (SM)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) is the most basic approach to learn $$s_\theta$$. Other variants are then proposed such as Denoising Score matching (DSM), Sliced Score Matching (SSM) to overcome the computational difficulty of SM. As usual, I prefer to step back to explore this basic approach before delving into more recent score-based generative models. In this post, I re-present elaborately the math encountered in developing SM based on my understanding with a simplified set of notations.

* [Original Score Matching paper](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
* [Denoising Score Matching paper](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)
* [Score Matching using stochastic differential equation](https://arxiv.org/pdf/2011.13456)
 
## Score matching
SM learns $$s_\theta$$ by minimizing the expected squared distance of the score of $$p_{data}(x)$$ called $$s_x$$ and the score given by the model $$s_\theta$$.

$$
\begin{aligned}
L(\theta) &= \frac{1}{2}\int_x p_{data}(x)(||s_x - s_\theta||_2^2)dx\\
&= \frac{1}{2}E_{x \sim p_{data}}(||s_x - s_\theta||_2^2), \\
\theta^* &= \arg\min_{\theta} L(\theta)
\end{aligned}
$$

where $$ s_x = \nabla_x{log(p_{data}(x))}$$, $$s_\theta = \nabla_x{log(p_{\theta}(x))}$$. $$p_\theta(x)$$ refers to the normalized probability density function approximated by the model. The difference between SM or other score-based generative models and log-likelihood-based models is SGMs do not need to learn $$p_\theta$$, which is very challenging because we need to ensure $$p_\theta(x)$$ represents a probability function, whose value range is between 0 and 1 and values are added up to 1 over all high-dimensonal x. Thanks to the definition of scores, SM only needs to trains an unnormalized $$\tilde{p}_\theta(x)$$, which can be implemented by any neural network. Below is the proof elaborating this.

$$
\begin{aligned}
L(\theta) &= \frac{1}{2}E_{x \sim p_{data}}(||s_x - s_\theta||_2^2), \\
&= \frac{1}{2}E_{x \sim p_{data}}(||\nabla_x{log(p_{data}(x))} - \nabla_x{log(p_{\theta}(x))}||_2^2 \\
&= \frac{1}{2}E_{x \sim p_{data}}(||\nabla_x{log(p_{data}(x))} - \nabla_x{log(\frac{\tilde{p}_{\theta}(x)}{   
\int_x\tilde{p}_{\theta}(x)dx
})}||_2^2\\
&= \frac{1}{2}E_{x \sim p_{data}}(||\nabla_x{log(p_{data}(x))} - \nabla_x{log(\frac{\tilde{p}_{\theta}(x)}{Z(\theta)})}||_2^2 \\
&= \frac{1}{2}E_{x \sim p_{data}}(||\nabla_x{log(p_{data}(x))} - \nabla_x{log(\tilde{p}_{\theta}(x)) - \nabla_xlog(Z(\theta))}||_2^2 \\
& \stackrel{(1)}{=} \frac{1}{2}E_{x \sim p_{data}}(||\nabla_x{log(p_{data}(x))} - \nabla_x{log(\tilde{p}_{\theta}(x))}||_2^2, \\ 
\end{aligned}
$$

(1) holds because $$Z(\theta)$$ does not depend on $$x$$ so the derivative $$\nabla_x{log(Z(\theta))} = 0$$

If we use a neural network to model $$\tilde{p}_{\theta}(x)$$, then $$\nabla_x{log(\tilde{p}_{\theta}(x))}$$ can be computed straightforwardly. The problem now is $$p_{data}(x)$$ is unknown. We are only given a set of training examples i.i.d from $$p_{data}(x)$$. SM resolves this issue by algebraically transforming $$L(\theta)$$ to make it undependable on $$p_{data}(x)$$. 

$$
\begin{aligned}
L(\theta) &= \int_{x}p_{data}(x) [Tr(\nabla_x{s_\theta}) + \frac{1}{2}||s_{\theta}||_2^2]dx + const\\
&=E_{p_{data}}[Tr(\nabla_x{s_\theta}) + \frac{1}{2}||s_{\theta}||_2^2] + const \\
\end{aligned}
$$

Before delving into the derivation of this training objective for SM, let us recal the integration by part formula 

$$
\begin{aligned}
\int_a^bu(x)v'(x)dx = u(b)v(b)-u(a)v(a) - \int_a^bu'(x)v(x)dx
\end{aligned}
$$

Applying $$p_{data}(x)$$ to $$u(x)$$ and $$s_\theta(x)$$ to $$v(x)$$. Also, without loosing the generality for simplicity, let us consider only 1 dimension of $$x$$ , say $$x_1$$, and fix the other dimensions. If $$a \to -\infty$$ and $$b \to \infty$$, and if we assume the derivative of $$log(p_{data}(x))$$ approaches zero at infinity, we have $$\lim_{a\to -\infty, b\to\infty}[p_{data}(b, x_2, x_3...)s_\theta(b, x_2, x_3,...) - p_{data}(a, x_2, x_3...)s_\theta(a, x_2, x_3,...)] = 0$$. This assumption is necessary in score matching to ensure the stability of the model. If this assumption holds, we will have

$$
\begin{aligned}
\int_xp_{data}(x)\nabla_xs_\theta(x) dx = - \int_x\nabla_xp_{data}(x)s_\theta(x)dx \text{ (*)}
\end{aligned}
$$

Now, let's delve into proving the objective $$L(\theta)$$ for training $$\tilde{p}_\theta(x)$$
$$
\begin{aligned}
L(\theta) &= \frac{1}{2}\int_x p_{data}(x)(||s_x - s_\theta||_2^2)dx\\
&= \int_x p_{data}(x)[\frac{1}{2}||s_x||_2^2 + \frac{1}{2}||s_\theta||_2^2 - s_x^Ts_\theta]dx\\
&= \int_x p_{data}(x)\frac{1}{2}||s_\theta||_2^2dx - \int_xp_{data}(x)\sum_i s_x^{(i)} s_\theta^{(i)}]dx + const\\
&= \int_x p_{data}(x)\frac{1}{2}||s_\theta||_2^2dx - \sum_i\int_xp_{data}(x)[s_x^{(i)} s_\theta^{(i)}]dx + const\\
&\stackrel{(2)}{=}\int_x p_{data}(x)\frac{1}{2}||s_\theta||_2^2dx - \sum_i\int_xp_{data}(x)[\nabla_{x_i}{log(p_{data}(x))} s_\theta^{(i)}]dx + const \text{ (2): replace the formula to compute } s_{x}^{(i)} \\
&\stackrel{(3)}{=}\int_x p_{data}(x)\frac{1}{2}||s_\theta||_2^2dx - \sum_i\int_xp_{data}(x)[\frac{1}{p_{data}(x)}\nabla_{x_i}{p_{data}(x)} s_\theta^{(i)}]dx + const \text{ (3): apply the formula to compute } \nabla_{x_i}{log(p_{data}(x))}\\
&=\int_x p_{data}(x)\frac{1}{2}||s_\theta||_2^2dx - \sum_i\int_x\nabla_{x_i}{p_{data}(x)} s_\theta^{(i)}dx + const \\
&\stackrel{(4)}{=}\int_x p_{data}(x)\frac{1}{2}||s_\theta||_2^2dx + \sum_i\int_xp_{data}(x) \nabla_{x_i}s_\theta dx + const \text{ (4): applying formula (*) mentioned above} \\
&=\int_x p_{data}(x)\frac{1}{2}[||s_\theta||_2^2 + \sum_i \nabla_{x_i}s_\theta]dx + const \\
&=\int_x p_{data}(x)[\frac{1}{2}||s_\theta||_2^2 + Tr( \nabla_{x_i}s_\theta)]dx + const \\
\end{aligned}
$$
