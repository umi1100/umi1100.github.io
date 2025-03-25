---
layout: post
title: Diffusion Model - A Math-based Simplified Understanding 
date: 2025-03-24 12:02:00-0400
inline: false
related_posts: true 
giscus_comments: true
---

It took me quite some time to comprehend the denoising process of DDPM (Denoising Diffusion Probabilistic Model), one of the most basic variant of diffusion models. In this post, I will elaborate the math behind deriving the objective of DDPM training. I think this math understanding will faciliate the adaptation of other DM variants.  

As usual, before delving into DDPM, I would like to recommend amazing knowledge sources that helped deepen my understanding. 

* [Original Diffusion model](https://arxiv.org/pdf/1503.03585) 
* [DDPM paper](https://arxiv.org/pdf/2006.11239)
* [Survey of Diffusion models](https://arxiv.org/pdf/2209.00796)
* [Lilian Weng's blog post about diffusion model](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* [Berkeley's lecture video about diffusion model](https://www.youtube.com/watch?v=687zEGODmHA&t=11s)

---

# Diffusion Model - DDPM
Diffusion Models (DM) tries to match the training data distribution by learning to reverse/denoise the diffusion process. Here, diffussion process is the one that gradually destructs the data structure by noise. My focus in this post is on DDPM, the most basic variant of DM that leverages Markov Chain (MC) for both the diffusion and reverse process. 

## Discrete MC Diffusion Process
In the diffusion (also called forward) processs, the training data $$x$$ is pertubed by a fixed MC through T discrete time steps. At time step 0, let $$x_0$$ be an original training sample following distribution ($$q_{data}$$). Then at an arbitrary time step $$t$$, $$x_t$$ is $$x_{t-1}$$ plus a small Gaussian noise characterized by variance $$\beta_t$$ such that 
 
$$
\begin{aligned}
& t=0,  & x_0 & \sim q_{data}(x) \\
& t = 1:T, & q(x_t|x_{t-1})& \sim N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_tI) (1*)\\
& t=T & x_T & \sim N(0, I) (2*)
\end{aligned}
$$

>* (1*): the conditional probability of $$x_t$$ given $$x_{t-1}$$ ($$q(x_t\|x_{t-1})$$) follows a Gaussian distribution with mean $$\sqrt{1-\beta_t}x_{t-1}$$ and variance $$\beta_tI$$ ($$I$$ is an identity matrix), from which we can sample $$x_t$$ given $$x_{t-1}$$.
>* (2*): T should be sufficiently large so that the final generated sample x_T of the diffusion process is close to Gaussian noise with mean of 0 and unit variance (denoted by $$I$$ because the data is usually high-dimensional).

After T time steps, for each sample $$x_0$$, we have a sequence of T noisy samples $$x_1, x_2,...,x_T$$. The joint distribution conditioned on x_0 is $$q(x_{1:T}\|x_0) = \prod_{t=1}^{T}q(x_t\|x_{t-1})$$

This sampling process is stochastic, we can't backprograte the gradient for model training. Applying the reparameterization trick as in VAE, we have:

$$
\begin{aligned}
q(x_t|x_{t-1}) & = \sqrt{1-\beta_t}x_{t-1} + \sqrt\beta_t\epsilon_{t-1}; \; \text{where}\: \epsilon_{t-1}, \epsilon_{t-2},... \sim N(0,I) \\
& = \sqrt\alpha_tx_{t-1} + \sqrt{1- \alpha_t}\epsilon_{t-1}; \; \text{where}\: \alpha_t = \beta_t - 1 \\
& = \sqrt\alpha_t(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2})  +  \sqrt{1- \alpha_t}\epsilon_{t-1}; \\
& \text{where} \: x_{t-1} \: \text{is computed recurvely by} \; x_{t-2} \\
& = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1- \alpha_t}\epsilon_{t-1}\\
& = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2}  + \sqrt{\alpha_t(1-\alpha_{t-1}) + 1-\alpha_t}\bar\epsilon_{t-2}; \; (3*)\\
& = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2}  + \sqrt{1-\alpha_t\alpha_{t-1} }\bar\epsilon_{t-2}\\
& = \sqrt{\alpha_t\alpha_{t-1}\alpha_{t-2}}x_{t-3}  + \sqrt{1-\alpha_t\alpha_{t-1}\alpha_{t-2} }\bar\epsilon_{t-3}\\
& = \sqrt{\alpha_t\alpha_{t-1}\alpha_{t-2}...\alpha_1}x_{0} + \sqrt{1-\alpha_t\alpha_{t-1}\alpha_{t-2}...\alpha_{1} }\epsilon\\
& = \cdots\\
& = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon; \; (4*)
\end{aligned}
$$

>* (3*) Combining two Gaussian distributions: $$\epsilon_{t-2} \sim N(0, \alpha_t(1-\alpha_{t-1})I)$$ and $$\epsilon_{t-1} \sim N(0, (1-\alpha_t)I)$$. The merged distribution is also Gaussian with  mean of 0 and variance of the sum of two individual variances: $$\bar \epsilon_{t-2} \sim N(0, (1-\alpha_t + \alpha_t-\alpha_t\alpha_{t-1})I$$

From Eq.(4*), we can sample any ($$x_t$$) directly from $$x_0$$: $$q(x_t|x_0) \sim N(x_t; \sqrt{\bar\alpha_t}x_0, \sqrt{1-\bar\alpha_t}I)$$
## Reverse/Denoising Diffusion Process

In the reverse process, DDPM tries to recover $$x_0$$ from $$x_T$$ by approximating the posterior probability $$q(x_{t-1}\|x_{t}$$ to reverse the diffusion process $$q(x_t\|x_{t-1})$$. However, it's very challenging to approximate $$q(x_{t-1}\|x_{t}$$ directly because we need the entire training dataset. We only know that $$q(x_{t-1}\|x_{t}$$ also follows Gaussion distribution because the noise added at each time step t $$\beta_t$$ is small. 

DDPM instead trains a model (parameterized by $$\theta$$) to estimate $$q(x_{t-1}\|x_t)$$, says $$p_\theta(x_{t-1}\|x_t)$$.

$$
\begin{aligned}
p_\theta(x_{0..T}) = p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}|x_t)
\end{aligned}
$$

Because $$q(x_{t-1}\|x_t)$$ follows Gaussian, so does $$p_\theta(x_{t-1}\|x_t)$$. Therefore, we can train the model $$p_\theta$$ to estimate the mean and variance of the Gaussian $$p_\theta(x_{t-1}\|x_t) \sim N(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_\theta(x_t, t))$$

This setup is similar to VAE, where we want to reconstruct the data $$x_0$$ from a Gaussian laten variable $$z$$. In DDPM, we reconstruct x_0 from a a set of latents or noisy samples $${x_{1..T}}$$ and we ccan utilize ELBO to maximize the log-likelihood of the data $$p_\theta(x_0)$$. 

$$
\begin{aligned}
  p_\theta(x_0) & = \int p_\theta(x_{0:T})dx_{1:T}, \text(1*) \\
	 & = \int p_\theta(x_{0:T}) \frac{q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)}dx_{1:T}, \\ 
	 & = E_{x\sim q(x_{1:T}|x_0)}\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\ 
\end{aligned}
$$

Apply the log operator to both sides of the equation

$$
\begin{aligned}
  -log(p_\theta(x)) & = -log(E_{x\sim q(x_{1:T}|x_0)}\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\
  & \le - E_q(log(\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)})) \\
  & = E_q(log(\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})})) =  L_{LVB}
\end{aligned}
$$

$$
\begin{aligned}
  L_{LVB} & =  E_q(log(\frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})})) \\
  & = E_q[log\frac{\prod_{t=1}^T{q(x_{t}|x_{t-1})}}{p_\theta(x_T)\prod_{t=1}^T{p_\theta(x_{t-1}|x_{t})}}] \\
  & = E_q[-log(p_\theta(x_T) + \sum_{t=1}^Tlog(\frac{q(x_{t}|x_{t-1})}{p_\theta(x_{t-1}|x_{t})})]\\
  & = E_q[-log(p_\theta(x_T) + \sum_{t=2}^Tlog(\frac{q(x_{t}|x_{t-1})}{p_\theta(x_{t-1}|x_{t})}) + log(\frac{q(x_{1}|x_{0})}{p_\theta(x_{0}|x_{1})})] \;\; (5*)\\
  & = E_q[-log(p_\theta(x_T) + \sum_{t=2}^Tlog(\frac{q(x_{t-1}|x_{t}, x_0)}{p_\theta(x_{t-1}|x_{t})}) +  \sum_{t=2}^Tlog(\frac{q(x_{t}|x_{0})}{q(x_{t-1}|x_{0})}) + log(\frac{q(x_{1}|x_{0})}{p_\theta(x_{0}|x_{1})})] \;\; (6*) \\
  & = E_q[-log(p_\theta(x_T) + \sum_{t=2}^Tlog(\frac{q(x_{t-1}|x_{t}, x_0)}{p_\theta(x_{t-1}|x_{t})}) +  log(\frac{q(x_{T}|x_{0})}{q(x_{1}|x_{0})}) + log(\frac{q(x_{1}|x_{0})}{p_\theta(x_{0}|x_{1})})] \\
  & = E_q[-log(p_\theta(x_T) + \sum_{t=2}^Tlog(\frac{q(x_{t-1}|x_{t}, x_0)}{p_\theta(x_{t-1}|x_{t})}) +  log(\frac{q(x_{T}|x_{0})}{p_\theta(x_{0}|x_{1})})] \\
  & = E_q[log(\frac{q(x_T|x_0)}{p_\theta(x_T)}) +\sum_{t=2}^Tlog(\frac{q(x_{t-1}|x_{t}, x_0)}{p_\theta(x_{t-1}|x_{t})}) - log(p_\theta(x_0|x_1))] \\
  & = E_q[D_{KL}(q(x_T|x_0)||p_\theta(x_T))+ \sum_{t=2}^TD_{KL}(q(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t)) - log(p_\theta(x_0|x_1))]
\end{aligned}
$$

Let us denote:

$$
\begin{aligned}
& L_T = D_{KL}(q(x_T|x_0)||p_\theta(x_T))\\
& L_{t} = D_{KL}(q(x_{t}|x_{t+1}, x_0)||p_\theta(x_{t}|x_{t+1}))\; \text{t=1,...,T-1}\\
& L_0 = - log(p_\theta(x_0|x_1))
\end{aligned}
$$

According to the forward process, $$x_T$$ is a Gaussian noise and can be computed directly from $$x_0$$. Hence, $$q(x_T\|x_0)$$ does not have trainable parameters and can be ignored. $$L_0$$ can be optimized by a decoder with input $$x_1$$ and output $$x_0$$. We only need to focus on $$L_t$$ 

$$L_t$$ compares two Guassions $$p_\theta$$ and $$q$$. It's notable that $$q(x_t\|x_{t+1}, x_0)$$ is tractable and we can derive the Gaussion function for $$q(x_t\|x_{t+1}, x_0) \sim N(x_t; \tilde \mu(x_{t+1}, x_0), \tilde \beta_{t+1}I)$$, from which we can modify L_t for a more stable training. Let us delve into this derivation. 

According to Bayes's rule,

$$
\begin{aligned}
q(x_{t+1}, x_{t}, x_0) & = q(x_{t+1}|x_t, x_0)q(x_{t}|x_0)q(x_0)\\
& = q(x_{t}|x_{t+1}, x_0)q(x_{t+1}|x_0)q(x_0) \\
q(x_t|x_{t+1}, x_0) & = q(x_{t+1}|x_t, x_0)\frac{q(x_{t}|x_0)}{q(x_{t+1}|x_0)}\\
& \propto \exp(-\frac{1}{2}(\frac{(x_{t+1}-\sqrt{\alpha_{t+1}}x_t)^2}{\beta_{t+1}}+\frac{(x_t-\sqrt{\bar\alpha_t}x_0)^2}{1-\bar\alpha_t} - \frac{(x_{t+1}-\sqrt{\bar\alpha_{t+1}}x_0)^2}{1-\bar\alpha_{t+1}} )) \\
& \propto \exp(-\frac{1}{2}(\frac{(x_{t+1}^2-2\sqrt{\alpha_{t+1}}x_tx_{t+1} + \alpha_{t+1}x_t^2)}{\beta_{t+1}}+\frac{(x_t^2-2\sqrt{\bar\alpha_t}x_0x_t + \bar\alpha_tx_0^2)}{1-\bar\alpha_t} - \frac{(x_{t+1}-\sqrt{\bar\alpha_{t+1}}x_0)^2}{1-\bar\alpha_{t+1}} )) \\
\end{aligned}
$$
Now we group terms with $$x_t^2$$ and $$x_t$$ and combine other terms to a constant function C. We call $$C$$ a constant function because it's not dependent on $$x_t$$

$$
\begin{aligned}
q(x_t|x_{t+1}, x_0) & \propto \exp[-\frac{1}{2}((\frac{\alpha_{t+1}}{\beta_{t+1}}+\frac{1}{1-\bar\alpha_t})x_t^2 -2(\frac{\sqrt{\alpha_{t+1}}x_{t+1}}{\beta_{t+1}}+\frac{\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t})x_t ) + C(x_{t+1}, x_0)] \\
\end{aligned}
$$

Now let's looking at the standard Gaussian density function $$f(x) \propto \exp[-\frac{1}{2}(\frac{x^2}{\sigma^2}-\frac{2\mu x}{\sigma^2}+\frac{\mu^2}{\sigma^2})]$$

Recall $$\alpha_t = 1-\beta_t$$ and $$\bar\alpha_t = \alpha_1\alpha_2\dots\alpha_t$$. We can parameterize
$$
\begin{aligned}
\tilde \beta_{t+1} & = 1/(\frac{\alpha_{t+1}}{\beta_{t+1}}+\frac{1}{1-\bar\alpha_t})\\
& = 1/(\frac{\alpha_{t+1} - \alpha_{t+1}\bar\alpha_t + \beta_{t+1}}{\beta_{t+1}(1-\bar\alpha_t)}) \\
& = \frac{1 - \bar\alpha_{t}}{1-\bar\alpha_{t+1}}\beta_{t+1}\\
\tilde\mu(x_{t+1}, x_0) & = (\frac{\sqrt{\alpha_{t+1}}x_{t+1}}{\beta_{t+1}}+\frac{\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t})/(\frac{\alpha_{t+1}}{\beta_{t+1}}+\frac{1}{1-\bar\alpha_t}) \\
& = (\frac{\sqrt{\alpha_{t+1}}x_{t+1}}{\beta_{t+1}}+\frac{\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t})\frac{1 - \bar\alpha_{t}}{1-\bar\alpha_{t+1}}\beta_{t+1} \\
&= \frac{\sqrt{\alpha_{t+1}}(1-\bar\alpha_t)}{1-\bar\alpha_{t+1}}x_{t+1} + \frac{\sqrt{\bar\alpha_t}\beta_{t+1}}{(1-\bar\alpha_{t+1})}x_0
\end{aligned}
$$

From the forward process, we have 
$$
x_0 = \frac{1}{\sqrt{\bar\alpha_{t+1}}}(x_{t+1} - \sqrt{1-\bar\alpha_{t+1}}\epsilon_{t+1})
$$

$$
\begin{aligned}
\tilde\mu(x_{t+1}, x_0) & = \frac{\sqrt{\alpha_{t+1}}(1-\bar\alpha_t)}{1-\bar\alpha_{t+1}}x_{t+1} + \frac{\sqrt{\bar\alpha_t}\beta_{t+1}}{(1-\bar\alpha_{t+1})}(\frac{1}{\sqrt{\bar\alpha_{t+1}}}(x_{t+1} - \sqrt{1-\bar\alpha_{t+1}}\epsilon_{t+1}))\\
& = \frac{\sqrt{\alpha_{t+1}}(1-\bar\alpha_t)}{1-\bar\alpha_{t+1}}x_{t+1}  + \frac{\beta_{t+1}}{(1-\bar\alpha_{t+1})\sqrt{\alpha_{t+1}}}(x_{t+1}-\sqrt{1-\bar\alpha_{t+1}}\epsilon_{t+1})\\
& = \frac{\alpha_{t+1}(1-\bar\alpha_t) + \beta_{t+1}}{(1-\bar\alpha_{t+1})\sqrt{\alpha_{t+1}}}x_{t+1} - \frac{1- \alpha_{t+1}}{\sqrt{1-\bar\alpha_{t+1}}\sqrt{\alpha_{t+1}}}\epsilon_{t+1}\\
& = \frac{1-\bar\alpha_{t+1}}{(1-\bar\alpha_{t+1})\sqrt{\alpha_{t+1}}}x_{t+1} - \frac{1- \alpha_{t+1}}{\sqrt{1-\bar\alpha_{t+1}}\sqrt{\alpha_{t+1}}}\epsilon_{t+1}\\
& = \frac{1}{\sqrt{\alpha_{t+1}}}(x_{t+1}- \frac{1-\alpha_{t+1}}{\sqrt{1-\bar\alpha_{t+1}}}\epsilon_{t+1})
\end{aligned}
$$
 
Recall we want to train a neural network to learn 
$$
p_\theta(x_t|x_{t+1})$$ and $$p_\theta(x_t|x_{t+1}) \sim N(x_t; \mu_\theta(x_{t+1}, t), \Sigma_\theta(x_{t+1},t))
$$. 
To train the network, we need to minimize the KL divergence between 
$$
q(x_t|x_{t+1}, x_0)
$$ 
and 
$$
p_\theta(x_{t}|x_{t+1})
$$. 
From the density function of $$q(x_t|x_{t+1}, x_0)$$ derived above, we can try to predict $$\tilde\mu =  \frac{1}{\sqrt{\alpha_{t+1}}}(x_{t+1}- \frac{1-\alpha_{t+1}}{\sqrt{1-\bar\alpha_{t+1}}}\epsilon_{t+1}) $$. Because $$x_{t+1}$$ is provided as input in the reverse process, we can parameterize the network to predict $$\epsilon_{t+1}$$: $$p_\theta(x_t|x_{t+1}) \sim N(x_t; \frac{1}{\sqrt{\alpha_{t+1}}}(x_{t+1}- \frac{1-\alpha_{t+1}}{\sqrt{1-\bar\alpha_{t+1}}}\epsilon_{\theta}(x_{t+1}, t)), \Sigma_\theta(x_{t+1},t))$$

The KL divergence between $$q(x_t|x_{t+1}, x_0)$$ and $$p_\theta(x_{t}|x_{t+1})$$ is the difference between 
 two mean $$\tilde\mu$$ and $$\mu_\theta$$

$$
\begin{aligned}
L_t &= D_{KL}(q(x_t|x_{t+1}, x_0)|p_\theta(x_{t}|x_{t+1})) \\
& = E_{x_0, \epsilon}(\frac{1}{2\|\Sigma_\theta(x_{t+1}, t)\|_2^2}\|(\tilde\mu_t(x_{t+1}, x_0)-\mu_\theta(x_{t+1}, t))\|^2\\
&=E_{x_0, \epsilon}(\frac{1}{2\|\Sigma_\theta\|_2^2}\|(\frac{1}{\sqrt{\alpha_{t+1}}}(x_{t+1}- \frac{1-\alpha_{t+1}}{\sqrt{1-\bar\alpha_{t+1}}}\epsilon_{t+1})-\frac{1}{\sqrt{\alpha_{t+1}}}(x_{t+1}- \frac{1-\alpha_{t+1}}{\sqrt{1-\bar\alpha_{t+1}}}\epsilon_{\theta}(x_{t+1}, t))\|^2\\
&=E_{x_0, \epsilon}(\frac{1-\alpha_{t+1}}{2\alpha_{t+1}\|\Sigma_\theta\|_2^2(1-\bar\alpha_{t+1})}\|\epsilon_{t+1}-\epsilon_{\theta}(x_{t+1}, t)\|^2\\
\end{aligned}
$$
 
With this $$L_t$$, we can train a network to predict the injected noise at each time step $$t$$. 
 



