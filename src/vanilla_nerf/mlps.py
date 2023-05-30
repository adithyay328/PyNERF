"""
This module implements both MLPs
used in the Vanilla NERF
paper, specifically the coarse
and the fine MLP.

The coarse MLP is used to
predict the density of a
point, while the fine MLP
is used to predict the
RGB color of a point. To re-use
some of the original compute
from the coarse MLP, it outputs
an embedding which we attach to the
input of the fine MLP
"""
from typing import Tuple

import torch
from torch import nn

class VanillaNeRFCoarseMLP(nn.Module):
    """
    An implementation of the Coarse MLP
    used in the Vanilla NERF paper. This
    MLP is used to predict the density
    of a point, and also outputs an embedding
    that can be re-used later. For the sake
    of similarity to the original paper, we
    will use a similar architecture to the
    coarse MLP proposed, which has N layers
    of D-wide linear layers with ReLU activations,
    followed by a concatenation of the input
    with the Nth layer. We then pass through 
    N_1 more layers of D_1-wide linear layers,
    which output the final density and embedding.

    These can then be used in the fine MLP.

    :param N: The number of layers in the first
        block of the coarse MLP.
    :param D: The width of each layer in the first
        block of the coarse MLP.
    :param N_1: The number of layers in the second
        block of the coarse MLP.
    :param D_1: The width of each layer in the second
        block of the coarse MLP.
    :param input_dim: The dimension of the input
        to the coarse MLP. This is the dimension
        of the input point after encoding.

    """
    def __init__(self, N : int, D: int, N_1 : int, D_1 : int, input_dim : int):
        self.N = N
        self.D = D
        self.N_1 = N_1
        self.D_1 = D_1
        self.input_dim = input_dim

        # We will use a list to store
        # all of the layers
        self.first_block = nn.ModuleList()
        self.second_block = nn.ModuleList()
        self.volume_density_layer = nn.Sequential(
            nn.Linear(D_1, 1),
            nn.ReLU()
        )

        self.first_block.append(nn.Linear(input_dim, D))
        self.first_block.append(nn.ReLU())

        for i in range(N - 1):
            self.first_block.append(nn.Linear(D, D))
            self.first_block.append(nn.ReLU())
        
        self.second_block.append(nn.Linear(D + input_dim, D_1))
        self.second_block.append(nn.ReLU())

        for i in range(N_1 - 1):
            self.second_block.append(nn.Linear(D_1, D_1))
            self.second_block.append(nn.ReLU())
        
    def forward(self, X : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation through the coarse MLP;
        returns a tuple of (volume_density, embedding of size D_1).
        """
        firstBlockOut = self.first_block(X)
        secondBlockIn = torch.cat([firstBlockOut, X], dim=-1)
        secondBlockOut = self.second_block(secondBlockIn)

        volume_density = self.volume_density_layer(secondBlockOut)

        return volume_density, secondBlockOut

class VanillaNeRFFineMLP(nn.Module):
    """
    Implements the fine MLP used in the
    Vanilla NERF paper. This MLP is used
    to predict the RGB color of a point,
    and takes in as input the embedding
    generated by the coarse MLP, as well
    as the viewing direction of the camera
    at the point.

    Like with before, we will use a similar
    architecture to the one proposed in the
    paper, which has 2 linear layers; the first
    one takes in the embedding and the viewing
    direction, while the second one takes in
    the output of the first layer and outputs
    the final RGB color. All are ReLU activated.

    :param D_1: The dimension of the embedding
        generated by the coarse MLP.
    :param input_dim: The dimension of the
        viewing direction vector.
    :param hidden_dim: The dimension of the
        hidden layer in the fine MLP.
    """
    def __init__(self, D_1 : int, input_dim : int, hidden_dim : int):
        self.D_1 = D_1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.first_layer = nn.Sequential(
            nn.Linear(D_1 + input_dim, hidden_dim),
            nn.ReLU()
        )

        self.second_layer = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.ReLU()
        )
    
    def forward(self, embedding : torch.Tensor, viewing_direction : torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the fine MLP;
        returns a tensor of size 3, which represents
        the RGB color of the point.
        """
        first_layer_in = torch.cat([embedding, viewing_direction], dim=-1)
        first_layer_out = self.first_layer(first_layer_in)
        second_layer_out = self.second_layer(first_layer_out)

        return second_layer_out