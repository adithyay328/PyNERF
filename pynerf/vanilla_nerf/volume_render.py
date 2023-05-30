"""
Implements logic related to volume
rendering from this paper.

Specifically, this includes the sampler
for sampling points along a ray, the
quadrature algorithm for approximating
the color at a viewing pixel, the sampling
of points to run through the fine network,
and also logic for computing the viewing
ray from a given camera pose + pixel.

Basically, the skeleton for all of vanilla
NeRF except the training loop.
"""
import torch
from torch import nn

from pynerf.vanilla_nerf.mlps import VanillaNeRFCoarseMLP, VanillaNeRFFineMLP

class CoarseDistanceSampling(nn.Module):
    """
    This class implements the coarse sampling
    algorithm from the paper. This algorithm
    is used to sample points along a ray
    given a segment across the ray and a number of
    samples to take.

    After getting the sampled distances, the associated
    world coordinate can be determined, allowing us
    to get coarse volume density from the coarse net.
    We then use the coarse volume density to determine
    the fine sampling points, allowing better performance.
    """
    def __init__(self):
        super(CoarseDistanceSampling, self).__init__()

    def forward(self, n : int, t_n : float, t_f : float) -> torch.Tensor:
        """
        Computes a set of n distances along a ray
        to sample it.

        As far as details go, we want to to sample
        points such that
        t_i ~ U[t_n + (i - 1) / N * (t_f - t_n), t_n + (i / N) * (t_f - t_n)]

        We can start simplifying this by creating a Tensor
        where all values are t_n, and instead add on a random
        Tensor where the ith entry is sampled with the law
        ith entry ~ U[(i - 1) / N * (t_f - t_n), (i / N) * (t_f - t_n)]
         = (t_f - t_n) / N * U[i - 1, i]
         = (t_f - t_n) / N * ( U[0, 1] + i - 1 )

        :param n: The number of distances to sample.
        :param t_n: The near plane of the camera.
        :param t_f: The far plane of the camera.

        :return: A tensor of all distances sampled.
        """
        multiplier = (t_f - t_n) / n
        randomTensor = torch.rand(n) + torch.linspace(1, n, n) - 1
        return torch.ones(n) * t_n + multiplier * randomTensor

def stratifiedSamplingDistancesToNeighbourDistance(sampledDistances : torch.Tensor) -> torch.Tensor:
    """
    Converts a set of sampled distances to
    the distance between each sampled distance
    and its neighbour. By default we use the right neighbour,
    but for the last point we just use the left neighbour

    :param sampledDistances: The sampled distances.

    :return: The distance between each sampled distance
        and its neighbour. By default we use the right neighbour,
        but for the last point we just use the left neighbour
    """
    neighbourDistances = torch.zeros_like(sampledDistances, requires_grad=False)
    neighbourDistances[:-1] =  sampledDistances[1:] - sampledDistances[:-1]
    neighbourDistances[-1] = torch.abs(sampledDistances[-1] - sampledDistances[-2])
    
    return neighbourDistances

class AccumulatedTransmittanceApproximation(nn.Module):
    """
    This module implements the quadrature approximation
    of the accumulated transmittance equation from the
    paper. This is used to approximate the color of a
    viewing pixel.
    """
    def __init__(self):
        super(AccumulatedTransmittanceApproximation, self).__init__()
    
    def forward(self, sampledDistances : torch.Tensor, volumeDensity : torch.Tensor) -> torch.Tensor:
        """
        Computes the accumulated transmittance approximation
        for a set of distances and volume densities.

        :param distances: The distances to sample at.
        :param volumeDensity: The volume density at each distance.

        :return: The accumulated transmittance approximation.
        """
        result = torch.zeros_like(sampledDistances, requires_grad=False)
        # First element of result is 1, since exp(0) = 1
        result[0] = 1

        neighbourDistances = stratifiedSamplingDistancesToNeighbourDistance(sampledDistances)
        product = neighbourDistances * volumeDensity
        # Cumulative sum while respecting batch dimension
        cumsum = torch.cumsum(product, dim=-1)

        result[1:] = torch.exp(-cumsum)

        return result

class ColorApproximation(nn.Module):
    """
    This module implements the quadrature approximation
    of the color equation from the paper. This is used
    to approximate the color of a viewing pixel.
    """
    def __init__(self):
        super(ColorApproximation, self).__init__()

        self.accumulatedTransmittanceApproximation = AccumulatedTransmittanceApproximation()
    
    def forward(self, sampledDistances : torch.Tensor, volumeDensity : torch.Tensor, volumeColor : torch.Tensor) -> torch.Tensor:
        """
        Computes the color approximation
        for a set of distances, volume densities,
        and volume colors.

        :param distances: The distances to sample at.
        :param volumeDensity: The volume density at each distance.
        :param volumeColor: The volume color at each distance.

        :return: The color approximation.
        """
        neighbourDists = stratifiedSamplingDistancesToNeighbourDistance(sampledDistances)
        accumulatedTransmittance = self.accumulatedTransmittanceApproximation(sampledDistances, volumeDensity)
        
        resultVector = accumulatedTransmittance * ( 1 - torch.exp(-1 * volumeDensity * neighbourDists) ) * volumeColor

        # Return sum on last dimension
        return resultVector.sum(dim=-1)
