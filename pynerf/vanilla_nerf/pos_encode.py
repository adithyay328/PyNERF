"""
Implements the positional encoding shceme introduced
in the NeRF paper. As a recap, the original NeRF paper
suggested that for each dimension of X and Theta,
the single dimensional input should be mapped
to a higher dimensional representation where each
dimension encodes the original data at a varying
frequency, allowing for MLPs, which learn low frequency
signals by default, to deal with high-frequency textures
in the underlying surface.

Specifically, the encoding function, which we will call
R(X) for some scalar value X, takes on the form R(X) = 
Tensor[sin(2^0 * math.pi * X), cos(2^0 * math.pi * X), 
sin(2^1 * math.pi * X), cos(2^1 * math.pi * X), ...
sin(2^(N-1) * math.pi * X), cos(2^(N-1) * math.pi * X)] 
where N is the dimension of the output that we desire.

In the paper, R(X) is applied elementwise for each
dimension of X and Theta, with N = 10 for X and
N = 4 for Theta.
"""
import math

import torch
from torch import nn

class VanillaNeRFEncoder(nn.Module):
    """
    Encoder module to encode
    the input X and Theta from
    the NeRF paper into a higher
    dimensional space, high frequency
    space for better representation
    and rendering results.

    :param N: The dimension of the output
        of the encoding function. Look at
        the module docstring for more
        details on the encoding scheme
        presented in this paper.
    """
    def __init__(self, N : int):
        super().__init__()
        self.N = N
    
    #TODO vectorize this
    def forward(self, X : torch.Tensor):
        # We can easily compute the
        # inside of the sinusoids with
        # a linspace and a multiply by PI * X,
        # and then follow up with some stacks
        # and reshapes to efficiently implement
        # this

        # If X is not batched, add an outer dimension
        if len(X.shape) == 1:
            X = X.unsqueeze(0)

        # Compute general linspaces for all dimensions
        linspaceTillN = torch.linspace(0, self.N - 1, self.N)
        raisedWithPi = (2 ** linspaceTillN) * math.pi

        # Now, multiply each element of each batch vector
        # to get the encoding
        encodingList = []

        for batch in range( X.shape[0] ):
            batchEncoding = []
            for dim in range( X.shape[1] ):
                batchEncoding.append(
                    raisedWithPi * X[batch][dim]
                )
            
            encodingList.append(
                torch.hstack( batchEncoding )
            )
        
        insideOfSinusoids = torch.vstack( encodingList )

        # Now, we need to apply sin and cos to all
        # of them, and then interleave those
        # 2 tensors together
        sined = torch.sin(insideOfSinusoids)
        cosined = torch.cos(insideOfSinusoids)

        catted = torch.cat((sined, cosined), dim=-1)
        return catted

        # # Now, we need to interleave them
        # # i.e. the first element is sined[0],
        # # second is cosined[0], third is sined[1],
        # # etc.
        # interleaved = torch.stack((sined, cosined)).mT.flatten()