'''
authors: Theo Dumont, Theo Lacombe
date: 2026-04
contact: theo.dumont@univ-eiffel.fr, theo.lacombe@univ-eiffel.fr

Main file to reproduce the benchmark of the paper "Learning Monge maps with constrained drifting models", as a MWE.

More options available on demand (see Readme.md).
'''

import numpy as np
import torch
import random
import matplotlib.pyplot as plt

import sampling
from utils_benchmark import expe
from dynamics import *
from explicitConstrainedFlow import explicitDynamic
from oneshotFlow import oneshotDynamic
from implicitConstrainedFlow import implicitDyanmic

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

torch.set_num_threads(1)


# Parameters of the mixture model
means = 2 * torch.tensor([[-1.,-1.0],[-0.5,1.0],[1.0,-1.50],[1.0,1.0],])
covariances = torch.tensor([[[0.10,0.06],[0.06,0.10]],[[0.1,0.01],[0.01,0.5]],[[0.10,0.03],[0.03,0.10]],[[0.10,0.0],[0.0,0.10]],]) #/ 5.
weights = torch.tensor([0.3, 0.2, 0.3, 0.2])


n_seed_max = 5  # param paper: 100
n_batch = 100     # param paper: 100


if __name__ == "__main__":
    list_mmd_euclidean      = []
    list_mmd_explicit = []
    list_mmd_implicit = []

    for seed in range(n_seed_max):
        
        torch.manual_seed(seed)          # Set seed for PyTorch
        np.random.seed(seed)             # Set seed for NumPy
        random.seed(seed)                   # Set seed for Python's built-in random module

        # Sampler from the source distribution
        gen_rho = sampling.GenerateGaussianMixture(n_batch=n_batch, means=means, covariances=covariances, weights=weights)

        # Sampler from the target distribution (used to compute MMD at the end)
        gen_gamma = sampling.GenerateGaussian(n_batch=n_batch)

        X_0, gamma_true = gen_rho.next(), gen_gamma.next()
        X_0, gamma_true = X_0.detach(), gamma_true.detach()

        param_shared = {
        'generator':gen_rho,           # random generator for rho0 (needed for generalization metrics)
        'X_0':X_0,                     # impose rho0 point cloud
        'generator_target':gen_gamma,  # random generator for gamma
        'n_batch':n_batch,             # number of points
        'ground_truth':None,           # optimal map is not known
        'verbose':0,                   
        'epsilon':None,                # epsilon for score estimation is None so done with jax-ott heuristic
        'seed':seed,             
        'device':device,
        'target': None,                # if MMD: samples of gamma. if langevin: potential V (None default to N(0,1))
        'n_batch_big_MMD': 10000,      # for MMD to target with big batch (MMD_ed_big)  # param paper: 10000
        'overfit': False               # whether sampling once for all or at each step
        }

        model_params_icnn = {'hidden_dim': 20, 'depth':2}           # parameters of ICNN architecture


        nstep_osflow = 3000        # Param paper: 3000
        param_oneshot = {              # parameters of one shot learning
        'dynamic':oneshotDynamic,
        'nb_grad_step_oneshot':nstep_osflow,  # total nb of gradient descent steps
        'dynamic_to_correct': langevinDynamic}

        param_explicit = {                # parameters of constrained flow
        'dynamic':explicitDynamic,
        'nb_grad_step':100,            # nb steps in inner optimization problem. # param paper: 100
        'dynamic_to_correct': langevinDynamic}

        param_implicit = {
            'dynamic':implicitDyanmic,
            'nb_grad_step':100,   # param paper: 100
            'dynamic_to_correct': langevinDynamic
        }

        ##### oneshot
        try:  
            lr = 1e-3   # Param paper: 1e-3
            mmd_std = expe(mapClass='ICNNMap', lr=lr, **param_shared, **param_oneshot, **model_params_icnn)
            list_mmd_euclidean.append(mmd_std)
        except:
            # Explicit Euclidean flow can be numerically unstable for large learning rates, so we catch any exceptions and skip those cases.
            # The value used in this benchmark is lr=1e-3, which is stable for the vast majority of seeds.
            # This is a small value, somewhat compensated by doing 3000 steps for the Euclidean GD (heuristically, seems to reach convergence
            # i.e. doing 2000 or 3000 steps gives similar results.
            # Taking a larger learning rate (e.g. 1e-2) could help to reduce the number of steps required to reach convergence, but is unstable for many seeds.
            # Note that the learning rate for the one shot method is not directly comparable to the learning rates used for the constrained flows
            pass

        ##### explicit constrained flow
        mmd_explicit_flow = expe(mapClass='ICNNMap', lr_inner_explicit=1e-1, **param_shared, **param_explicit, final_time=4, n_steps=int(10), **model_params_icnn)
        list_mmd_explicit.append(mmd_explicit_flow)

        # Reset batch sizes
        gen_rho.n_batch = n_batch
        gen_gamma.n_batch = n_batch
        ##### Implicit constrained flow
        mmd_implicit = expe(mapClass='ICNNMap', lr_inner_implicit=1e-1, **param_shared, **param_implicit, final_time=4, n_steps=int(10), **model_params_icnn)
        list_mmd_implicit.append(mmd_implicit)



    # save results in npy file
    hist_euclidean = np.array(list_mmd_euclidean)
    hist_explicit = np.array(list_mmd_explicit)
    hist_implicit = np.array(list_mmd_implicit)

    fig, axs = plt.subplots(1,3, figsize=(24, 6))
    ax = axs[0]
    ax.hist(hist_euclidean, bins=30, range=(0, 0.1), color='blue', alpha=0.3, density=True, label='std GD')
    ax = axs[1]
    ax.hist(hist_explicit, bins=30, range=(0, 0.1), color='orange', alpha=0.3, density=True, label='natural GD (our)')
    ax = axs[2]
    ax.hist(hist_implicit, bins=30, range=(0, 0.1), color='green', alpha=0.3, density=True, label='implicit GD (our)')
    [ax.legend() for ax in axs]

    plt.show()