# MetricLearning

This is the first version of the NN-based metric learning

It goes with a simple jupyter notebook, which illustrates the way it works on simple Gaussian-shaped 1D signals

A large number of questions are still open:
- The minimization can be improved using standard learning heuristics (see later if we need to improve the robustness of the learning stage)
- Propagation of noise/robustness to noise ? Include it in the learning stage in the spirit of the denoising autoencoder ?
- To get an interpolated value, do we use a simple forward-backprojection or a least-squares minimization scheme ? Especially when the input is noisy.
- Choosing the examples/anchor points ?
- Fine-tuning the trade-off coefficient or relaxing it starting from a large to small value ?
- As usual, how many layers, architecture of the network ?
- How many training samples ?
- ...
