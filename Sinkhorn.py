'''
Sinkhorn for score estimation.
'''
import torch
from torch import nn
import matplotlib.pyplot as plt


class Sinkhorn(nn.Module):
    """
    Class for Sinkhorn algorithm and various computations related to it.

    Note: for score estimation, we only need the self potential, so we can use the self_EOT option.
            But for transport cost computation, we need to compute the potentials f and g.

            TODO: provide a way for a "warm start" of the potentials f and g.
    """

    def __init__(self, mu, nu=None,
                 epsilon=None, 
                 iterations=1000, C=None, display_skip=5, verbose=True):
        super(Sinkhorn, self).__init__()
        """
        Inputs:
        mu is a pytorch array N * (1 + dimension)
        nu is a pytorch array M * (1 + dimension)
        mu[:,0] is a probability vector, same for nu[:,0]
        """
        self.n = mu.shape[0]
        if nu is not None:
            self.self_EOT = False
        else:
            self.self_EOT = True
            nu = mu.clone()
        self.mu_m = torch.ones_like(mu[:,0])/self.n  # 1/n (1,...,1)
        self.nu_m = torch.ones_like(nu[:,0])/self.n  # 1/n (1,...,1)
        self.X, self.Y = mu, nu                      # μ, ν
        self.X.requires_grad_(True)
        self.Y.requires_grad_(True)
        self.f, self.g = None, None                  # f, g (potentials)
        if epsilon is None:
            X_t_square = torch.sum(self.X ** 2, dim=1).view(-1, 1)
            pairwise_distance_matrix = X_t_square + X_t_square.t() - 2 * torch.mm(self.X, self.X.t())
            epsilon = 0.05 * torch.median(pairwise_distance_matrix)  # median heuristic
        self.epsilon = epsilon                       # ε
        self.iterations = iterations
        self.C = C                                   # C (can be None)
        self.display_skip = display_skip             # just for display
        self.verbose = verbose

    def constructCostMatrix(self):
        """Computes the matrix C(x,y)=∥x-y∥² / 2 is not done yet."""
        if self.C is None:
            self.C = self.constructCost(self.X)
    
    def constructCost(self,X):
        """Compute the matrix C(x,y)=∥x-y∥² / 2 for specified x."""
        C = torch.sum(X ** 2, 1).view(-1, 1) - 2 * torch.sum(X.unsqueeze(1) * torch.transpose(self.Y.unsqueeze(1), 0, 1), 2) + torch.sum(self.Y ** 2,1).view(1, -1)
        return 0.5 * C

    # Note : a.logsumexp(1) is a vector of size a.size()[0] containing log(sum(exp(a),axis=1))
    def soft_i(self, g):
        """Return -εlog∫exp[(g-c)/ε]dν."""
        return -self.epsilon * ((g.view(1, -1) - self.C) / self.epsilon + (self.nu_m.log()).view(1, -1)).logsumexp(1)

    def soft_j(self, f):
        """Return -εlog∫exp[(f-c)/ε]dμ."""
        return -self.epsilon * ((f.view(-1, 1) - self.C) / self.epsilon + (self.mu_m.log()).view(-1, 1)).logsumexp(0)

    def soft_selfEOT(self, h):
        """Return h/2 - ε/2 log∫exp[(h-c)/ε]dμ."""
        return h/2 - self.epsilon/2 * ((h.view(-1, 1) - self.C) / self.epsilon + (self.mu_m.log()).view(-1, 1)).logsumexp(0)

    def iterateSinkhorn(self):
        """Iterate Sinkhorn algorithm and return potentials."""
        if self.f is None:
            self.constructCostMatrix()
            self.list_df, self.list_dg, self.list_fmg = [], [], []
            self.list_marginal_1, self.list_marginal_2, = [], []
            f = torch.zeros(self.mu_m.size()[0],requires_grad = True)
            g = torch.zeros(self.nu_m.size()[0], dtype=torch.float32,requires_grad = True)
            for i in range(self.iterations):
                if i>=1 and i%self.display_skip==0: f_old = f.clone(); g_old = g.clone()
                # classical Sinkhorn iterations
                if not self.self_EOT:
                    f = self.soft_i(g)              # f = -εlog∫exp[(g-c)/ε]dν
                    g = self.soft_j(f)              # g = -εlog∫exp[(f-c)/ε]dμ
                # self EOT Sinkhorn iterations
                else:
                    f = self.soft_selfEOT(f)        # f = f/2-ε/2log∫exp[(f-c)/ε]dμ
                    g = f.clone()                   # g = f
                self.f, self.g = f, g 
                # display -----------------------
                if i>=1 and i%self.display_skip==0:
                    delta_f = ((f-f_old)**2/(f**2)).detach().clone().abs().mean()
                    delta_g = ((g-g_old)**2/(g**2)).detach().clone().abs().mean()
                    self.list_df.append(delta_f)
                    self.list_dg.append(delta_g)
                    plan = self.computePlan()
                    marginal_1 = ((plan.sum(axis=0)-self.mu_m)**2).mean()
                    marginal_2 = ((plan.sum(axis=1)-self.nu_m)**2).mean()
                    self.list_marginal_1.append(marginal_1)
                    self.list_marginal_2.append(marginal_2)
                    if (marginal_1 <= 1e-8 and marginal_2 <= 1e-8) or (delta_f <= 1e-12 and delta_g <= 1e-12): break
                # end display -------------------
            if self.verbose: print('Sinkhorn done in', i+1, 'iterations')
        return self.f, self.g

    def computePlan(self):
        """Compute π=exp[(f+g-C)/ε]d(μ⊗ν)."""
        plan = (self.g.view(1, -1) + self.f.view(-1, 1) - self.C) / self.epsilon + (self.nu_m.log()).view(1, -1) + (
            self.mu_m.log()).view(-1, 1)
        self.plan = plan.exp()
        return self.plan.detach().clone()

    def computeKL(self):
        """Compute KL=1/ε[⟨f,μ⟩+⟨g,ν⟩-⟨π,C⟩]"""
        self.iterateSinkhorn()
        return (self.computeDivergence() - self.computeSharp()) / self.epsilon

    def computeDivergence(self):
        """Compute ⟨f,μ⟩+⟨g,ν⟩."""
        self.iterateSinkhorn()
        return torch.dot(self.f, self.mu_m) + torch.dot(self.g, self.nu_m)

    def computeSharp(self):
        """Compute ⟨π,C⟩."""
        self.iterateSinkhorn()
        self.computePlan()
        return torch.sum(self.plan * self.C)

    def computeGradientOfPotential(self,x):
        """Computes ∇f_ε(x)."""
        self.iterateSinkhorn()
        C = self.constructCost(x)
        K = (self.g - C) / self.epsilon
        gammaz = -torch.amax(K, dim=1)
        K = K + gammaz.view(-1,1)
        Z = torch.exp(K)
        return x - torch.matmul(Z,self.Y) / torch.sum(Z,axis = 1).unsqueeze(1)

    def computeScore_(self,x):
        """Compute the score ∇logρ(x) = -2/ε ∇f_ε(x)."""
        assert self.self_EOT
        return -2./self.epsilon * self.computeGradientOfPotential(x)
    
    def plot_iterations(self):
        """Plot the iterations of some values along the iterations of Sinkhorn."""
        self.iterateSinkhorn()
        it = range(len(self.list_df)*self.display_skip)[::self.display_skip]
        if not self.self_EOT:
            for thing, label in zip(
                [self.list_df, self.list_dg, self.list_marginal_1, self.list_marginal_2],
                ['$\\delta f_n$','$\\delta g_n$','$1$','$2$']
                ):
                plt.plot(it, thing, label=label)
        else:
            for thing, label in zip(
                [self.list_df, self.list_marginal_1, self.list_marginal_2],
                ['$\\delta f_n$','$1$','$2$']
                ):
                plt.plot(it, thing, label=label)

        plt.yscale('log')
        plt.legend()
        plt.show()
