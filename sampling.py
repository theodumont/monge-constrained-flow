'''
Element in this files are Generator object. They should be equipped with a .next() function that enables sampling. 
Initialization specify the batch size through a n_batch argument. 
'''
import torch

class GenerateGaussian(object):
    def __init__(self, mean=None, covariance=None, n_batch=200, device='cpu'):
        if mean is None:
            self.mean = torch.tensor([0.0, 0.0])
        else:
            self.mean = mean
        if covariance is None:
            self.covariance = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        else:
            self.covariance = covariance
        self.n_batch = n_batch
        self.device = device

    def next(self):
        """
        Sample from a Gaussian distribution.

        Args:
            mean (torch.Tensor): Tensor of shape (d,) representing the mean of the Gaussian.
            covariance (torch.Tensor): Tensor of shape (d, d) representing the covariance matrix.
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Samples of shape (num_samples, d) drawn from the Gaussian.
        """
        return torch.distributions.MultivariateNormal(self.mean, self.covariance).sample((self.n_batch,)).to(self.device)



class GenerateGaussianMixture(object):
    def __init__(self,means=torch.tensor([[0.0,0.0]]),
                 covariances=torch.tensor([[[1.0,0.0],[0.0,1.0]]]),
                 weights=torch.tensor([1.0]),
                 n_batch=200,
                 pushforward=None, 
                 return_also_before_push=False, device='cpu'):
        self.n_batch = n_batch
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.pushforward = pushforward
        self.return_also_before_push = return_also_before_push
        self.device = device
        
        
    def next(self):
        """
        Sample from a multidimensional mixture of Gaussians.

        Args:
            means (torch.Tensor): Tensor of shape (k, d) representing the means of the Gaussians.
            covariances (torch.Tensor): Tensor of shape (k, d, d) representing the covariance matrices.
            weights (torch.Tensor): Tensor of shape (k,) representing the mixture weights (should sum to 1).
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Samples of shape (num_samples, d) drawn from the mixture.
        """
        k, d = self.means.shape

        # Ensure weights sum to 1
        self.weights = self.weights / self.weights.sum()

        # Sample from the categorical distribution to choose components
        components = torch.multinomial(self.weights, self.n_batch, replacement=True)

        # Allocate space for samples
        samples = torch.empty(self.n_batch, d)
        
        #print("samples",samples.shape)
        # Sample from the chosen Gaussian components
        for i in range(k):
            mask = components == i
            num_samples_i = mask.sum()

            if num_samples_i > 0:
                # Sample from a multivariate normal
                samples[mask] = torch.distributions.MultivariateNormal(
                    self.means[i], self.covariances[i]
                ).sample((self.n_batch,))[mask]

        
        if self.pushforward is not None:
            samples_before = samples.clone()
            # samples -= self.pushforward(samples)
            samples = samples - self.pushforward(samples)  # change due to autograd error
            if self.return_also_before_push:
                return samples, samples_before

        return samples.to(self.device)