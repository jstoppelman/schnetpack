import torch
import torch.nn as nn

from schnetpack.nn.base import Dense
from schnetpack import Properties
from schnetpack.nn.cfconv import CFConv
from schnetpack.nn.cutoff import HardCutoff
from schnetpack.nn.cutoff import FermiDirac
from schnetpack.nn.acsf import GaussianSmearing
from schnetpack.nn.neighbors import AtomDistances
from schnetpack.nn.activations import shifted_softplus


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_spatial_basis (int): number of input features of filter-generating networks.
        n_filters (int): number of filters used in continuous-filter convolution.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_spatial_basis,
        n_filters,
        cutoff,
        cutoff_network=HardCutoff,
        normalize_filter=False,
    ):
        super(SchNetInteraction, self).__init__()
        # filter block used in interaction block
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters),
        )
        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)
        # interaction block
        self.cfconv = CFConv(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            self.filter_network,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        return v


class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems.

    Args:
        n_atom_basis (int, optional): number of features to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        n_filters (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis=128,
        n_filters=128,
        n_interactions=3,
        cutoff=5.0,
        n_gaussians=25,
        normalize_filter=False,
        coupled_interactions=False,
        return_intermediate=False,
        max_z=100,
        cutoff_network=HardCutoff,
        trainable_gaussians=False,
        distance_expansion=None,
        charged_systems=False,
    ):
        super(SchNet, self).__init__()

        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size n_atom_basis
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # layer for computing interatomic distances
        self.distances = AtomDistances()

        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion = distance_expansion

        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # set attributes
        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)

    def forward(self, inputs):
        """Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)

        if False and self.charged_systems and Properties.charge in inputs.keys():
            n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
            charge = inputs[Properties.charge] / n_atoms  # B
            charge = charge[:, None] * self.charge  # B x F
            x = x + charge

        # compute interatomic distance of every atom to its neighbors
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v
            if self.return_intermediate:
                xs.append(x)

        if self.return_intermediate:
            return x, xs
        return x

class SchNet_mod(SchNet):
    """SchNet architecture for learning representations of atomistic systems.

    Args:
        n_atom_basis (int, optional): number of features to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        n_filters (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis=128,
        n_filters=128,
        n_interactions=3,
        cutoff=5.0,
        n_gaussians=25,
        normalize_filter=False,
        coupled_interactions=False,
        return_intermediate=False,
        max_z=100,
        cutoff_network=HardCutoff,
        trainable_gaussians=False,
        distance_expansion=None,
        charged_systems=False,
        morse_bond=None,
        morse_mu=None,
        atom_t=None,
        atom_react=None,
        atom_prod=None,
        cutoff_react=None,
        cutoff_prod=None,
        morse_beta=100.0,
    ):
        super(SchNet_mod, self).__init__(n_atom_basis=n_atom_basis, n_filters=n_filters, n_gaussians=n_gaussians, n_interactions=n_interactions, cutoff=cutoff, cutoff_network=cutoff_network)

        if morse_bond is not None:
            g1 = [i for i in morse_bond if isinstance(i, int)]
            if len(g1) < len(morse_bond):
                g2 = [i for i in morse_bond if isinstance(i, list)][0]
                self.morse1 = torch.tensor(g1).reshape(-1)
                self.morse2 = torch.tensor(g2)
                #inc = -torch.ones_like(self.morse2)
                #zero = torch.zeros_like(self.morse2)
                #zero[self.morse2 > self.morse1] = inc[self.morse2 > self.morse1]
                #self.morse2 = self.morse2 + zero
            else: 
                morse1 = g1[0]
                morse2 = g1[1]
                #if morse2 > morse1: morse2 -= 1
                self.morse1 = torch.tensor(morse1).reshape(-1)
                self.morse2 = torch.tensor(morse2).reshape(-1)

            self.fermi_dirac = FermiDirac(mu=morse_mu, beta=morse_beta)
        
        if atom_t is not None:
            self.atom_t = torch.tensor(atom_t).reshape(-1)
            if isinstance(atom_react, int):
                if atom_react < self.atom_t:
                    self.atom_react = torch.tensor(atom_react).reshape(-1)
                else:
                    self.atom_react = torch.tensor(atom_react - 1).reshape(-1)
            else:
                atom_react = torch.tensor(atom_react).reshape(-1)
                inc = -torch.ones_like(atom_react)
                zero = torch.zeros_like(atom_react)
                zero[atom_react > self.atom_t] = inc[atom_react > self.atom_t]
                self.atom_react = atom_react + zero
            if isinstance(atom_prod, int):
                if atom_prod < self.atom_t:
                    self.atom_prod = torch.tensor(atom_prod).reshape(-1)
                else:
                    self.atom_prod = torch.tensor(atom_prod - 1).reshape(-1)
            else:
                atom_prod = torch.tensor(atom_prod).reshape(-1)
                inc = -torch.ones_like(atom_prod)
                zero = torch.zeros_like(atom_prod)
                zero[atom_prod > self.atom_t] = inc[atom_prod > self.atom_t]
                self.atom_prod = atom_prod + zero
            self.fd_react = FermiDirac(mu=cutoff_react, beta=morse_beta)
            self.fd_prod = FermiDirac(mu=cutoff_prod, beta=morse_beta)

    def forward(self, inputs):
        """Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]

        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)

        if False and self.charged_systems and Properties.charge in inputs.keys():
            n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
            charge = inputs[Properties.charge] / n_atoms  # B
            charge = charge[:, None] * self.charge  # B x F
            x = x + charge

        # compute interatomic distance of every atom to its neighbors
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
       
        if hasattr(self, 'fermi_dirac'):
            if neighbors.get_device() == -1:
                self.morse2 = self.morse2.cuda()
            #This will only work if only one type of molecule is in the batch, and all the atoms
            #are ordered the same
            inds = [torch.where(neighbors[0, self.morse1, :] == i)[1] for i in self.morse2]
            #print(inds)
            inds_check = [False for i in inds if len(i)>1]
            if False not in inds_check:
                morse_bond = r_ij[:, self.morse1, inds].reshape(r_ij.shape[0], self.morse2.shape[0])
                morse_bond = torch.min(morse_bond, dim=1, keepdim=True)[0]
                fd = self.fermi_dirac(morse_bond)
                #print(morse_bond)
                #sys.exit()
            else:
                fd = 0
        
        if hasattr(self, 'fd_react'):
            react_dist = r_ij[:, self.atom_t, self.atom_react].reshape(r_ij.shape[0], self.atom_react.shape[0])
            prod_dist = r_ij[:, self.atom_t, self.atom_prod].reshape(r_ij.shape[0], self.atom_prod.shape[0])
            react_dists = torch.min(react_dist, dim=1, keepdim=True)[0]
            prod_dists = torch.min(prod_dist, dim=1, keepdim=True)[0]
            #trans_dist = torch.cat((react_dists, prod_dists), dim=-1)

            #cut_dist = torch.max(trans_dist, dim=1, keepdim=True)[0]
            fd_react = self.fd_react(react_dists)
            fd_react = fd_react.unsqueeze(1)
            fd_prod = self.fd_prod(prod_dists)
            fd_prod = fd_prod.unsqueeze(1)
            fd = torch.mul(fd_react, fd_prod)

        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v
            if self.return_intermediate:
                xs.append(x)

        if self.return_intermediate:
            return x, xs
        if hasattr(self, 'fermi_dirac'):
            return x, fd
        if hasattr(self, 'fd_react'):
            return x, fd
        return x
