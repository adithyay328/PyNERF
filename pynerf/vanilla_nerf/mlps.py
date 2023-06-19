"""
This module implements both MLPs
used in the Vanilla NERF
paper, specifically the coarse
and the fine MLP.

The coarse MLP is used to
predict the density of a
point, while the fine MLP
is used to predict the
RGB color of a point.
"""
from typing import Tuple

import torch
from torch import nn

from pos_encode import VanillaNeRFEncoder

class VanillaNeRFMLP(nn.Module):
    """
    In the original NeRF paper, 2 different
    neural nets, one coarse and one fine,
    are used in tandem to produce high-quality
    renders of the scene. While not explicitly
    mentioned in the paper, it is implied
    that both neural nets actually share the same
    architecture, and as such we only need one
    module, with configurable parameters, to
    represent both the coarse and fine MLPs.

    Internally, there are are 2 first blocks of
    FC layers, at the end of which we are given a volume
    density and an embedding. Optionally, we can
    pass the embedding, along with the viewing
    direction, to a third block of MLPs, which
    outputs a color. During training, both the
    coarse and fine MLPs are trained by using
    a color loss; during inference, the coarse
    MLP is only evaluated up to the volume density,
    and the fine MLP is used to predict the color.
    
    My intuition for why we need both is that
    the coarse network spends more time learning
    the representation of the entire scene, while
    the fine network spends more time learning the
    detail of the portions of the scene with
    high volume density. This should allow better
    rendering quality, but I also want to play around
    with using different network layouts i.e.
    less layers in the coarse net, more in the fine net,
    etc.

    :param N: The number of layers in the first
        block of the MLP.
    :param D: The width of each layer in the first
        block of the MLP.
    :param N_1: The number of layers in the second
        block of the MLP, after the input
        position is concatenated with the
        output of the first block.
    :param D_1: The width of each layer in the second
        block of the MLP.
    :param X_Encoding_N: The order of sinusoid
        encoding to use for the input position.
        If N = 1 we use 2 dimensions, if N = 2
        we use 4 dimensions, etc.
    :param theta_Encoding_N: The order of sinusoid
        encoding to use for the viewing direction.
        Same as X_Encoding_N, but for the viewing
        direction.
    :param N_2: The number of layers in the third
        block of the MLP, after the viewing
        direction is concatenated with the
        output of the second block.
    :param D_2: The width of each layer in the third
        block of the MLP.
    """
    def __init__(self, N : int, D: int, N_1 : int,
                 D_1 : int, X_Encoding_N : int,
                 theta_Encoding_N : int, N_2 : int,
                 D_2 : int):
        self.N = N
        self.D = D
        self.N_1 = N_1
        self.D_1 = D_1
        self.X_Encoding_N = X_Encoding_N
        self.theta_Encoding_N = theta_Encoding_N
        self.N_2 = N_2
        self.D_2 = D_2

        super(VanillaNeRFMLP, self).__init__()

        # Storing both of our encoders, one
        # for the position and one for the
        # viewing direction
        self.position_encoder = VanillaNeRFEncoder(
            X_Encoding_N
        )
        self.theta_encoder = VanillaNeRFEncoder(
            theta_Encoding_N
        )

        # Computing output dimensions
        # for encoding
        self.x_Encoded_Size = 3 * (2 * X_Encoding_N)
        self.theta_Encoded_Size = 2 * (2 * theta_Encoding_N)

        # The first block of the MLP
        self.block_1 = nn.ModuleList()
        self.block_1.append(
            nn.Linear(self.x_Encoded_Size, D)
        )
        self.block_1.append(
            nn.ReLU()
        )

        for i in range(N - 1):
            self.block_1.append(
                nn.Linear(D, D)
            )
            self.block_1.append(
                nn.ReLU()
            )
        
        # The second block of the MLP
        self.block_2 = nn.ModuleList()
        self.block_2.append(
            nn.Linear(D + self.x_Encoded_Size, D_1)
        )
        self.block_2.append(
            nn.ReLU()
        )

        for i in range(N_1 - 2):
            self.block_2.append(
                nn.Linear(D_1, D_1)
            )
            self.block_2.append(
                nn.ReLU()
            )
        
        # We add one more linear layer
        # that outputs the embedding for
        # this block
        self.block_2.append(
            nn.Linear(D_1, D_1)
        )
        
        # We will add one more linear, ReLU
        # rectified layer
        # that computes the volume density
        # from the block 2 embedding
        self.density_layer = nn.Sequential(
            nn.Linear(D_1, 1),
            nn.ReLU()
        )

        # The third block of the MLP
        self.block_3 = nn.ModuleList()
        self.block_3.append(
            nn.Linear(D_1 + self.theta_Encoded_Size, D_2)
        )
        self.block_3.append(
            nn.ReLU()
        )

        for i in range(N_2 - 2):
            self.block_3.append(
                nn.Linear(D_2, D_2)
            )
            self.block_3.append(
                nn.ReLU()
            )
        
        # The last layer has a sigmoid
        # activation
        self.block_3.append(
            nn.Linear(D_2, 3)
        )
        self.block_3.append(
            nn.Sigmoid()
        )
    
    def getVolumeDensityAndEmbedding(self, X : torch.Tensor):
        """
        Given X input, computes the associated volume density
        and the embedding out of the second block of MLPs

        :param X: A tensor of shape (N, 3) representing
            the position of the point.

        :return: A tuple (volume_density, embedding) where
            volume_density is a tensor of shape (N, 1)
            representing the volume density of the point,
            and embedding is a tensor of shape (N, D_1)
            representing the embedding of the point.
        """
        X_Encoded = self.position_encoder(X)

        # Pass through the first block
        # of the MLP
        X = X_Encoded
        for layer in self.block_1:
            X = layer(X)
        
        # Now, concatenate X with the
        # output of the first block
        X = torch.cat([X, X_Encoded], dim = -1)

        # Now, pass through the second
        # block of the MLP
        for layer in self.block_2:
            X = layer(X)
        
        # At this point, X is the embedding,
        # and we can compute the volume density
        # from it
        volume_density = self.density_layer(X)

        return volume_density, X
    
    def forward(self, X : torch.Tensor, theta : torch.Tensor):
        """
        Forward function for the MLP. This
        function takes in a position and
        viewing direction, and outputs
        the volume density and color
        of the point. If you only want
        the volume density, call compute_density.

        :param X: A tensor of shape (N, 3)
            representing the position of
            the point.
        :param theta: A tensor of shape (N, 2)
            representing the viewing direction
            of the point.
        :return: A tuple of tensors, the first
            of which is a tensor of shape (N, 1)
            representing the volume density of
            the point, and the second of which
            is a tensor of shape (N, 3) representing
            the color of the point.
        """
        # Encode theta
        theta_Encoded = self.theta_encoder(theta)

        # Get the volume density and embedding
        volume_density, embedding = self.getVolumeDensityAndEmbedding(X)

        # Now, concatenate theta with the
        # embedding
        X = torch.cat([embedding, theta_Encoded], dim = -1)

        # Pass through the third block to get
        # the color
        for layer in self.block_3:
            X = layer(X)
        
        return volume_density, X

# class VanillaNeRFCoarseMLP(nn.Module):
#     """
#     An implementation of the Coarse MLP
#     used in the Vanilla NERF paper. This
#     MLP is used to predict the density
#     of a point, and also outputs an embedding
#     that can be re-used later. For the sake
#     of similarity to the original paper, we
#     will use a similar architecture to the
#     coarse MLP proposed, which has N layers
#     of D-wide linear layers with ReLU activations,
#     followed by a concatenation of the input
#     with the Nth layer. We then pass through 
#     N_1 more layers of D_1-wide linear layers,
#     which output the final density and embedding.

#     These can then be used in the fine MLP.

#     :param N: The number of layers in the first
#         block of the coarse MLP.
#     :param D: The width of each layer in the first
#         block of the coarse MLP.
#     :param N_1: The number of layers in the second
#         block of the coarse MLP.
#     :param D_1: The width of each layer in the second
#         block of the coarse MLP.
#     :param input_dim: The dimension of the input
#         to the coarse MLP. This is the dimension
#         of the input point after encoding.

#     """
#     def __init__(self, N : int, D: int, N_1 : int, D_1 : int, input_dim : int):
#         self.N = N
#         self.D = D
#         self.N_1 = N_1
#         self.D_1 = D_1
#         self.input_dim = input_dim

#         # We will use a list to store
#         # all of the layers
#         self.first_block = nn.ModuleList()
#         self.second_block = nn.ModuleList()
#         self.volume_density_layer = nn.Sequential(
#             nn.Linear(D_1, 1),
#             nn.ReLU()
#         )

#         self.first_block.append(nn.Linear(input_dim, D))
#         self.first_block.append(nn.ReLU())

#         for i in range(N - 1):
#             self.first_block.append(nn.Linear(D, D))
#             self.first_block.append(nn.ReLU())
        
#         self.second_block.append(nn.Linear(D + input_dim, D_1))
#         self.second_block.append(nn.ReLU())

#         for i in range(N_1 - 1):
#             self.second_block.append(nn.Linear(D_1, D_1))
#             self.second_block.append(nn.ReLU())
        
#     def forward(self, X : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward propagation through the coarse MLP;
#         returns a tuple of (volume_density, embedding of size D_1).
#         """
#         firstBlockOut = self.first_block(X)
#         secondBlockIn = torch.cat([firstBlockOut, X], dim=-1)
#         secondBlockOut = self.second_block(secondBlockIn)

#         volume_density = self.volume_density_layer(secondBlockOut)

#         return volume_density, secondBlockOut

# class VanillaNeRFFineMLP(nn.Module):
#     """
#     Implements the fine MLP used in the
#     Vanilla NERF paper. This MLP is used
#     to predict the RGB color of a point,
#     and takes in as input the embedding
#     generated by the coarse MLP, as well
#     as the viewing direction of the camera
#     at the point.

#     Like with before, we will use a similar
#     architecture to the one proposed in the
#     paper, which has 2 linear layers; the first
#     one takes in the embedding and the viewing
#     direction, while the second one takes in
#     the output of the first layer and outputs
#     the final RGB color. All are ReLU activated.

#     :param D_1: The dimension of the embedding
#         generated by the coarse MLP.
#     :param input_dim: The dimension of the
#         viewing direction vector.
#     :param hidden_dim: The dimension of the
#         hidden layer in the fine MLP.
#     """
#     def __init__(self, D_1 : int, input_dim : int, hidden_dim : int):
#         self.D_1 = D_1
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         self.first_layer = nn.Sequential(
#             nn.Linear(D_1 + input_dim, hidden_dim),
#             nn.ReLU()
#         )

#         self.second_layer = nn.Sequential(
#             nn.Linear(hidden_dim, 3),
#             nn.ReLU()
#         )
    
#     def forward(self, embedding : torch.Tensor, viewing_direction : torch.Tensor) -> torch.Tensor:
#         """
#         Forward propagation through the fine MLP;
#         returns a tensor of size 3, which represents
#         the RGB color of the point.
#         """
#         first_layer_in = torch.cat([embedding, viewing_direction], dim=-1)
#         first_layer_out = self.first_layer(first_layer_in)
#         second_layer_out = self.second_layer(first_layer_out)

#         return second_layer_out