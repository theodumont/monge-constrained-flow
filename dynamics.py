import torch

import Sinkhorn as ud


def langevinDynamic(X, 
                    target=None, 
                    epsilon=None, 
                    **kwargs):
    """
    Langevin dynamic for a given sample X. 

    The score is estimated using Sinkhorn self entropic term, hence the epsilon in the parameters.

    :param X: torch.tensor of shape (n_batch, dimension), current state of the measure.
    :param target: the target potential, as a pytorch object that can be called on a sample and has a gradient. 
    :param epsilon: the regularization parameter for Sinkhorn. If None, set to be the median heuristic.
    :returns: a function v that takes a (possibly new) sample and returns the dynamic at this sample.
    """
    device = kwargs['device']
    X = X.to(device)
    if epsilon is None:
        X_t_square = torch.sum(X ** 2, dim=1).view(-1, 1)
        pairwise_distance_matrix = X_t_square + X_t_square.t() - 2 * torch.mm(X, X.t())
        epsilon = 0.05 * torch.median(pairwise_distance_matrix)  # median heuristic

    S = ud.Sinkhorn(X.detach().clone().to(device), epsilon=epsilon, iterations=5000, verbose=False).to(device)
    S.iterateSinkhorn()

    if target is None:
        def v(new_X):
            nabla_log_rho_new_X = S.computeScore_(new_X)
            return -new_X - nabla_log_rho_new_X
    else:
        def v(new_X):
            XX = new_X.detach().clone().requires_grad_(True).to(device)
            nabla_log_rho_new_X = S.computeScore_(new_X)
            tmp = target(XX)
            nabla_target = torch.autograd.grad(outputs=tmp, inputs=XX, grad_outputs=torch.ones_like(tmp), create_graph=True)[0]
            return -nabla_target - nabla_log_rho_new_X
    
    return v(X)



# additional function to estimate the gradient of KL divergence.

def estimateGradKL(map, X_theta, target, epsilon=None, wasserstein_gradient=False):
    """
    Estimate the gradient of divergence J(theta) = KL(map(gen_source) || target)
    If wasserstein_gradient is False, returns the gradient w.r.t. theta.
    If wasserstein_gradient is True, returns the Wasserstein gradient (w.r.t. rho_theta).
    """

    # Estimate epsilon if not given
    if epsilon is None:
        X_t_square = torch.sum(X_theta ** 2, dim=1).view(-1, 1)
        pairwise_distance_matrix = X_t_square + X_t_square.t() - 2 * torch.mm(X_theta, X_theta.t())
        epsilon = 0.05 * torch.median(pairwise_distance_matrix)

    # Estimate score function ∇ log ρ_θ
    S = ud.Sinkhorn(X_theta.detach().clone(), epsilon=epsilon, iterations=5000, verbose=False)
    S.iterateSinkhorn()
    nabla_log_rho_theta = S.computeScore_(X_theta)

    # Compute ∇V if target is provided
    if target is None:
        direction = X_theta + nabla_log_rho_theta
    else:
        XX = X_theta.detach().clone().requires_grad_(True)
        tmp = target(XX)
        nabla_target = torch.autograd.grad(outputs=tmp, inputs=XX, grad_outputs=torch.ones_like(tmp), create_graph=True)[0]
        direction = nabla_target + nabla_log_rho_theta  # shape: (n_samples, d)

    # if we want the Wasserstein gradient
    if wasserstein_gradient:
        return direction

    # if we want the gradient wrt theta
    else:
        # Compute gradient of output w.r.t. parameters, vector-Jacobian product with direction
        grads = torch.autograd.grad(
            outputs=X_theta,
            inputs=list(map.parameters()),
            grad_outputs=direction,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )

        # Replace grad_J with actual computed gradients
        grad_J = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, map.parameters())]

        return grad_J
