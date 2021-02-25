# Interpolator AutoEncoder (IAE)

This is the first version of a learning-based interpolatory scheme. It basically allows to build an non-linear interpolatory process from anchor points.

It goes with a simple jupyter notebook, which illustrates the way it works on simple Gaussian-shaped 1D signals.

```math
{\bf b}(\{\lambda\}) = \mbox{argmin}_{\bf z} \sum_{a=1}^d \lambda_a \phi_{\theta}({\bf z},{\varphi}_a),
```
