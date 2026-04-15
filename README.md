# monge-constrained-flow

This is the official repository for the paper *[Learning Monge maps with constrained drifting models](https://hal.science/hal-05566583v2/document)*.

**Authors:** Théo Dumont

**Contact:** `theo.dumont@univ-eiffel.fr`


## Dependencies

- `numpy`
- `pytorch`
- `matplotlib`

## Reproducing the results of the article

After installing the required dependencies, one has to run `main.py` (e.g. from terminal). This will store three `.npy` arrays in the repository and produce a figure afterward. 

It runs the implicit and explicit constrained flows as described in the article, along with the Euclidean gradient flow in the space of parameter has a benchmark.

Beware that running the code can take some time (few dozen of minutes on a laptop). 


## Organization and comments

- `main.py` is the main file. It relies on the function `expe()` implemented `utils_benchmark.py`, calling it with a series of hyper-parameters. 
- `explicitConstrainedFlow.py` (resp. `implicitConstrainedFlow.py`) implement the explicit (resp. implicit) drifting scheme we introduce in the referenced paper. Similarly, `oneshotFlow.py` refers to the Euclidean gradient flow. 
- `mongeMap.py` implements our parametrization of Monge Maps as gradients of ICNN. 
- `dynamics.py` implement the Langevin dynamic, giving access to Euclidean and Wasserstein gradient of the functionals.
- `sampling.py`, `MMD.py`, `Sinkhorn.py` are utilitary files. 

**Comment:** This code provides a minimal working example to reproduce the results proposed in our paper. On demand, we can provide a (messier) repository with more options availables (other architectures for Monge maps, other dynamics, other samplers, etc.).