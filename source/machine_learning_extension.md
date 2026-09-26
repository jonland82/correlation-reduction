# Potential extension: dependence and work in ReLU classifiers

**Status:** Working note for a possible application of [*The Work of Reducing Dependence*](paper.tex). This proposes a measurement framework, not a thermodynamic law for neural networks.

## Question and observables

Let $X=(A,U)$, where $A=X_1$ is a feature of interest and $U=(X_2,\ldots,X_d)$ contains the other features. A classifier with parameters $\theta$ produces a class $C_\theta\in\{1,\ldots,K\}$ to predict a target $Y$. The population distribution $p(a,u,y)$ and the classifier together determine the joint distribution of $A$ and $C_\theta$.

The paper's dependence ratio and mutual information apply directly:

$$
R_\theta(a,k)
=\frac{p_\theta(a,k)}{p_A(a)p_\theta(k)}
=\frac{q_{\theta,k}(a)}{q_{\theta,k}},
\qquad
I_\theta(A;C)=\int p_A(a)\sum_{k=1}^K q_{\theta,k}(a)
\log\frac{q_{\theta,k}(a)}{q_{\theta,k}}\,da,
$$

where $q_{\theta,k}(a)=P(C_\theta=k\mid A=a)$ and $q_{\theta,k}=P(C_\theta=k)$. For discrete $A$, replace the integral by a sum. Terms with $q_{\theta,k}(a)=0$ contribute zero to $I$; classes with $q_{\theta,k}=0$ can be omitted. A proposed reduction is

$$
\Delta I_A=I_{\theta_0}(A;C)-I_{\theta_1}(A;C)>0.
$$

This is a precise reduction in *statistical dependence of predictions on $A$*. If $A$ is a protected attribute, $I(A;C)=0$ is demographic parity. Another possible criterion is equalized odds, $I(A;C\mid Y)=0$. These criteria answer different questions; neither alone measures every kind of bias or establishes whether a particular decision causally used $A$. If $A$ is an ordinary predictive feature, dependence on it can be useful.

## Exact form for a ReLU network

Consider a feedforward network with $L$ ReLU hidden layers and logits $z_\theta(x)\in\mathbb R^K$:

$$
h^{(0)}=x,
\qquad t^{(\ell)}=W_\ell h^{(\ell-1)}+b_\ell,
\qquad h^{(\ell)}=\operatorname{ReLU}(t^{(\ell)})
\quad(1\leq\ell\leq L),
\qquad z_\theta(x)=W_{L+1}h^{(L)}+b_{L+1}.
$$

First take the reported class to be the deterministic argmax, $C_\theta(x)=\arg\max_k z_{\theta,k}(x)$, with a fixed tie rule. Let $s=(s_1,\ldots,s_L)$ specify which units are active and let $D_\ell(s)=\operatorname{diag}(s_\ell)$. Within the activation region $P_s$, initialize $M_0=\mathbf I$, $c_0=0$, and recursively define

$$
M_\ell=D_\ell(s)W_\ell M_{\ell-1},
\qquad
c_\ell=D_\ell(s)(W_\ell c_{\ell-1}+b_\ell).
$$

Assign zero preactivations to the inactive branch. The region is specified by the sign conditions

$$
P_s=
\bigcap_{\ell,j:s_{\ell j}=1}\{x:t_{\ell j,s}(x)>0\}
\cap
\bigcap_{\ell,j:s_{\ell j}=0}\{x:t_{\ell j,s}(x)\leq0\},
\qquad
t_{\ell,s}(x)=W_\ell(M_{\ell-1}x+c_{\ell-1})+b_\ell,
$$

so the activation regions are disjoint even when the input distribution has mass on a boundary. Thus, on $P_s$, the logits are affine:

$$
z_\theta(x)=M_sx+c_s^*,
\qquad
M_s=W_{L+1}M_L,
\qquad
c_s^*=W_{L+1}c_L+b_{L+1}.
$$

The portion of that region classified as $k$ is

$$
B_{s,k}=P_s\cap\bigcap_{j\ne k}
\left\{x:[M_sx+c_s^*]_k\geq[M_sx+c_s^*]_j\right\}.
$$

Assign equality in the class comparisons according to the fixed argmax tie rule. These are polyhedral decision regions. Conditioning on $A=a$ gives the full class probabilities:

$$
\boxed{q_{\theta,k}(a)=
\sum_s\int_{\{u:(a,u)\in B_{s,k}\}}p(u\mid a)\,du},
\qquad
q_{\theta,k}=\int p_A(a)q_{\theta,k}(a)\,da.
$$

Substituting these probabilities into the first equations gives $R_\theta(a,k)$ and $I_\theta(A;C)$ exactly. For a one-hidden-layer network, the logits inside the decision-region integral reduce to

$$
z_{\theta,k}(a,u)=b^{(2)}_k+
\sum_{r=1}^{m}V_{kr}
\max\!\left(0,w_{rA}a+w_{rU}^{\mathsf T}u+b^{(1)}_r\right).
$$

If a class is **sampled** from the softmax distribution, replace each hard decision indicator with

$$
P(C_\theta=k\mid x)=
\frac{\exp z_{\theta,k}(x)}{\sum_{j=1}^K\exp z_{\theta,j}(x)}.
$$

The same $R$ and $I$ equations then apply after averaging that probability over $p(u\mid a)$. If the measured output is instead a continuous score or probability vector, its joint distribution with $A$ must be specified separately; mutual information can even be infinite for a deterministic continuous mapping. The finite-class formulation avoids that issue.

Two limits help interpret the equations:

1. **Only $A$ enters a deterministic classifier.** Then $C=f_\theta(A)$ and $I(A;C)=H(C)$. Independence requires a constant class prediction. Other features or randomization are needed for a nontrivial dependence reduction.
2. **Binary illustration.** If $A$ is a fair bit and a randomized binary output agrees with it with probability $(1+\lambda)/2$, then $R(a,c)=1+\lambda$ when $a=c$ and $R(a,c)=1-\lambda$ otherwise. Its mutual information is

   $$
   I(A;C)=\frac{1+\lambda}{2}\log(1+\lambda)
   +\frac{1-\lambda}{2}\log(1-\lambda).
   $$

   This gives a simple dependence axis for a first experiment; it does not assign an energy cost to changing $\lambda$.

## A measurable energy-versus-dependence question

Fix a baseline classifier $\theta_0$, an input population, a loss function $\mathcal L(\theta)=\mathbb E[\ell(Y,C_\theta)]$, specified hardware, and an allowed set of update procedures $\pi:\theta_0\to\theta_\pi$. Let $E_{\rm update}(\pi)$ be the electrical energy measured while applying an update, with a declared measurement boundary. A constrained energy frontier is

$$
\boxed{
E^*_{\rm update}(\varepsilon,\delta)
=\inf_\pi\mathbb E[E_{\rm update}(\pi)]
\quad\text{subject to}\quad
I_{\theta_\pi}(A;C)\leq\varepsilon,
\qquad
\mathcal L(\theta_\pi)\leq\mathcal L(\theta_0)+\delta.
}
$$

The loss constraint prevents a constant-output classifier from being treated as a satisfactory solution. A deployment study can separately report $E_{\rm update}(\pi)+nE_{\rm infer}(\theta_\pi)$ over $n$ inferences. Comparing threshold changes, post-processing, and retraining under the same population and accuracy constraint would produce an empirical curve in joules versus nats. A local slope, when it exists, is an implementation-specific energy cost per nat of tightening the dependence constraint.

The mutual information above is computed across *examples and their predictions*. It is not, by itself, the thermodynamic mutual information between two physical parts of the computer. Consequently, the oscillator identity $W_{\rm rev}=k_BT\Delta I$ in the main paper cannot be substituted for $E^*_{\rm update}$. A physical work bound would require an explicit model of the hardware states, baths, energy landscape, and update protocol. Two procedures can achieve the same $\Delta I_A$ while using very different electrical energy.

## Proposed first test

1. Choose an attribute $A$, an outcome $Y$, and either demographic parity $I(A;C)$ or equalized odds $I(A;C\mid Y)$. State why that criterion is appropriate.
2. Train a small ReLU classifier and evaluate $q_{\theta,k}(a)$, $R_\theta(a,k)$, $I_\theta$, and predictive loss on held-out data. For a small synthetic distribution, compare these estimates with the decision-region integrals above.
3. Apply several specified update procedures at several dependence targets. Measure update energy on the same hardware and record predictive loss, uncertainty, and achieved dependence at each endpoint.
4. Plot the attainable energy–dependence–loss frontier. Only after specifying a physical hardware model should the measured electrical energy be compared with a thermodynamic work bound.

## Related work

- [Hanin and Rolnick, *Complexity of Linear Regions in Deep Networks* (2019)](https://proceedings.mlr.press/v97/hanin19a.html): ReLU decision-region structure.
- [Agarwal et al., *A Reductions Approach to Fair Classification* (2018)](https://proceedings.mlr.press/v80/agarwal18a.html): constrained demographic parity and equalized odds.
- [Hardt, Price, and Srebro, *Equality of Opportunity in Supervised Learning* (2016)](https://papers.neurips.cc/paper_files/paper/2016/hash/6a9659feb1216f14f7384ba499518b38-Abstract.html): conditional fairness criteria.
- [Kolchinsky and Wolpert, *Work, Entropy Production, and Thermodynamics of Information under Protocol Constraints* (2021)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.041024): why a physical work bound depends on allowed controls.
