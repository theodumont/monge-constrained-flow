"""
Implementation of implicit flow. Similar to explicit flow, but with another inner implementation problem.

In contrast to Modin: only implemented for relative entropy.
"""

import torch
from mongeMap import ICNNMap
from dynamics import estimateGradKL

    
FUNCTIONS = {'ICNNMap': ICNNMap}

def solveJKO(map, t_to_learn, mapClass, nb_grad_step, generator, **kwargs):
    """
    Solve the following optimization problem:

    \min_theta KL(map_theta * rho0 | target) + 1/(2 * tau) * |map_theta - map_previous|^2.

    Because KL is not autodiff friendly on samples, we implement the gradient by hand.
    Once the gradient is computed, we use pytorch autodiff to compute the second gradient (squared norm of the map - map_previous).
    and optimize using Adam. 
    """
    tau = map.dt

    lr_implicit = kwargs['lr_inner_implicit']
    target = kwargs['target']
    
    # set pytorch optimizer, a standard gradient descent on the parameters of the map, with learning rate lr_implicit
    # optimizer = torch.optim.SGD(map.parameters(), lr=lr_implicit)
    optimizer = torch.optim.Adam(map.parameters(), lr=lr_implicit)

    # Store previous map
    if mapClass == 'ICNNMap':
        map_previous = FUNCTIONS[mapClass](map.hidden_dim, map.depth, n_timesteps=map.n_timesteps, final_time=map.final_time)
    map_previous.load_state_dict(map.state_dict())
    map_previous.current_time = map.current_time

    # Perform optimization
    for i in range(nb_grad_step):
        optimizer.zero_grad()
        X_0 = generator.next()
        if type(X_0) is tuple: X_0 = X_0[0]
        # Pushforward of rho0 by map_previous
        X_t = map_previous.forward(X_0, t_to_learn)
        T_X_0 = map.forward(X_0, t_to_learn)

        # Compute gradient of KL with respect to map.parameters, using hand-crafted formula (see estimateGradKL in dynamics.py)
        grad_kl = estimateGradKL(map, T_X_0, target)

        # Compute gradient of squared norm using autodiff, 
        squared_norm = torch.mean(torch.norm(T_X_0 - X_t, dim=1)**2) / (2 * tau)
        squared_norm.backward()
        grad_squared_norm = []

        for param in map.parameters():
            if param.grad is None:
                grad_squared_norm.append(torch.zeros_like(param))
            else:
                grad_squared_norm.append(param.grad.clone())
        
        for p, g_kl, g_sn in zip(map.parameters(), grad_kl, grad_squared_norm):
            p.grad = g_kl + g_sn

        optimizer.step()

    return map


def implicitDyanmic(generator, n_steps, final_time, nb_grad_step, X_0=None, mapClass='RKHS', **kwargs):
    device = kwargs['device']

    if X_0 is None:
        X_0 = generator.next()
    else:
        assert generator.n_batch == X_0.shape[0]

    map = FUNCTIONS[mapClass](n_timesteps=n_steps,final_time=final_time, **kwargs).to(device)

    for j in range(n_steps):  # Outer loop
        map.current_time += 1
        # Solving JKO, updating map in place
        _ = solveJKO(map=map, t_to_learn=j+1, 
                    nb_grad_step=nb_grad_step, generator=generator, mapClass=mapClass, **kwargs)
    return map
