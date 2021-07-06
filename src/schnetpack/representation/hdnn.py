import torch
import torch.nn as nn
import sys
import schnetpack.nn as snn
from schnetpack.data import StatisticsAccumulator
from schnetpack import Properties
import numpy as np
from schnetpack.nn.cutoff import FermiDirac
import time

class HDNNException(Exception):
    pass


class SymmetryFunctions(nn.Module):
    """
    Compute atom-centered symmetry functions [#acsf1]_ and weighted variant thereof as described
    in Reference [#wacsf1]_.
    By default, the atomic number is used as element depended weight. However, by specifying the
    trainz=True, a more general elemental embedding is learned instead.

    Args:
        n_radial (int):  Number of radial functions
        n_angular (int): Number of angular functions
        zetas (set of int): Set of exponents used to compute the angular term, default is zetas={1}
        cutoff (callable): Cutoff function, default is the typical cosine cutoff function
        cutoff_radius (float): Cutoff radius, default are 5 Angstrom
        centered (bool): Whether centered Gaussians should be used for radial functions. Angular functions use centered Gaussians by default.
        crossterms (bool): Whether cutoff and exponential terms of the distance r_jk between both neighbors should be included in the angular functions. Default is False
        elements (set of int): Nuclear charge present in the molecules, default is {1,6,7,8,9} (H,C,N,O and F).
        sharez (bool): Whether angular and radial functions should use the same elemental weighting. The default is true.
        trainz (bool): If set to true, elemental weights are initialized at random and learned during training. (default is False)
        initz (str): How elemental weights are initialized. Allowed are (default='weighted'):
                        weighted: Weigh symmetry functions with nuclear charges (wACSF)
                        onehot: Represent elements by onehot vectors. This can be used in combination with pairwise_elements
                                in order to emulate the behavior of classic Behler symmetry functions.
                        embedding: Use random embeddings, which can e.g. be trained. Embedding length is modified via
                                   len_embedding (default=False).
        len_embedding (int): Number of elemental weights, default is 1. If more are used, embedding vectors similar to SchNet can be obtained.
        pairwise_elements (bool): Recombine elemental embedding vectors in the angular functions via an outer product. If e.g. one-hot encoding
                                  is used for the elements, this is equivalent to standard Behler functions
                                  (default=False).

    References
    ----------
        .. [#acsf1] Behler:
           Atom-centered symmetry functions for constructing high-dimensional neural network potentials.
           The Journal of Chemical Physics 134. 074106. 2011.
        .. [#wacsf1] Gastegger, Schwiedrzik, Bittermann, Berzsenyi, Marquetand:
           wACSF -- Weighted atom-centered symmetry functions as descriptors in machine learning potentials.
           The Journal of Chemical Physics 148 (24), 241709. 2018.
    """

    def __init__(
        self,
        n_radial=22,
        n_angular=5,
        zetas={1},
        cutoff=snn.CosineCutoff,
        cutoff_radius=5.0,
        centered=False,
        crossterms=False,
        elements=frozenset((1, 6, 7, 8, 9)),
        sharez=True,
        trainz=False,
        initz="weighted",
        len_embedding=5,
        pairwise_elements=False,
    ):

        super(SymmetryFunctions, self).__init__()

        self.n_radial = n_radial
        self.n_angular = n_angular
        self.len_embedding = len_embedding
        self.n_elements = None
        self.n_theta = 2 * len(zetas)

        # Initialize cutoff function
        self.cutoff_radius = cutoff_radius
        self.cutoff = cutoff(cutoff=self.cutoff_radius)

        # Check for general stupidity:
        if self.n_angular < 1 and self.n_radial < 1:
            raise ValueError("At least one type of SF required")

        if self.n_angular > 0:
            # Get basic filters
            self.theta_filter = snn.BehlerAngular(zetas=zetas)
            self.angular_filter = snn.GaussianSmearing(
                start=1.0,
                stop=self.cutoff_radius - 0.5,
                n_gaussians=n_angular,
                centered=True,
            )
            self.ADF = snn.AngularDistribution(
                self.angular_filter,
                self.theta_filter,
                cutoff_functions=self.cutoff,
                crossterms=crossterms,
                pairwise_elements=pairwise_elements,
            )
        else:
            self.ADF = None

        if self.n_radial > 0:
            # Get basic filters (if centered Gaussians are requested, start is set to 0.5
            if centered:
                radial_start = 1.0
            else:
                radial_start = 0.5
            self.radial_filter = snn.GaussianSmearing(
                start=radial_start,
                stop=self.cutoff_radius - 0.5,
                n_gaussians=n_radial,
                centered=centered,
            )
            self.RDF = snn.RadialDistribution(
                self.radial_filter, cutoff_function=self.cutoff
            )
        else:
            self.RDF = None

        # Initialize the atomtype embeddings
        self.radial_Z = self.initz(initz, elements)

        # check whether angular functions should use the same embedding
        if sharez:
            self.angular_Z = self.radial_Z
        else:
            self.angular_Z = self.initz(initz, elements)

        # Turn of training of embeddings unless requested explicitly
        if not trainz:
            # Turn off gradients
            self.radial_Z.weight.requires_grad = False
            self.angular_Z.weight.requires_grad = False

        # Compute total number of symmetry functions
        if not pairwise_elements:
            self.n_symfuncs = (
                self.n_radial + self.n_angular * self.n_theta
            ) * self.n_elements
        else:
            # if the outer product is used, all unique pairs of elements are considered, leading to the factor of
            # (N+1)/2
            self.n_symfuncs = (
                self.n_radial
                + self.n_angular * self.n_theta * (self.n_elements + 1) // 2
            ) * self.n_elements

    def initz(self, mode, elements):
        """
        Subroutine to initialize the element dependent weights.

        Args:
            mode (str): Manner in which the weights are initialized. Possible are:
                weighted: Weigh symmetry functions with nuclear charges (wACSF)
                onehot: Represent elements by onehot vectors. This can be used in combination with pairwise_elements
                        in order to emulate the behavior of classic Behler symmetry functions.
                embedding: Use random embeddings, which can e.g. be trained. Embedding length is modified via
                           len_embedding (default=False).
            elements (set of int): List of elements present in the molecule.

        Returns:
            torch.nn.Embedding: Embedding layer of the initialized elemental weights.

        """

        maxelements = max(elements)
        nelements = len(elements)

        if mode == "weighted":
            weights = torch.arange(maxelements + 1)[:, None]
            z_weights = nn.Embedding(maxelements + 1, 1)
            z_weights.weight.data = weights
            self.n_elements = 1
        elif mode == "onehot":
            weights = torch.zeros(maxelements + 1, nelements)
            for idx, Z in enumerate(elements):
                weights[Z, idx] = 1.0
            z_weights = nn.Embedding(maxelements + 1, nelements)
            z_weights.weight.data = weights
            self.n_elements = nelements
        elif mode == "embedding":
            z_weights = nn.Embedding(maxelements + 1, self.len_embedding)
            self.n_elements = self.len_embedding
        else:
            raise NotImplementedError(
                "Unregognized option {:s} for initializing elemental weights. Use 'weighted', 'onehot' or 'embedding'.".format(
                    mode
                )
            )

        return z_weights

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Nbatch x Natoms x Nsymmetry_functions Tensor containing ACSFs or wACSFs.

        """
        positions = inputs[Properties.R]
        Z = inputs[Properties.Z]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]

        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]

        # Compute radial functions
        if self.RDF is not None:
            # Get atom type embeddings
            Z_rad = self.radial_Z(Z)
            # Get atom types of neighbors
            Z_ij = snn.neighbor_elements(Z_rad, neighbors)
            # Compute distances
            distances = snn.atom_distances(
                positions,
                neighbors,
                neighbor_mask=neighbor_mask,
                cell=cell,
                cell_offsets=cell_offset,
            )
            radial_sf = self.RDF(
                distances, elemental_weights=Z_ij, neighbor_mask=neighbor_mask
            )

        else:
            radial_sf = None

        if self.ADF is not None:
            # Get pair indices
            try:
                idx_j = inputs[Properties.neighbor_pairs_j]
                idx_k = inputs[Properties.neighbor_pairs_k]

            except KeyError as e:
                raise HDNNException(
                    "Angular symmetry functions require "
                    + "`collect_triples=True` in AtomsData."
                )
            neighbor_pairs_mask = inputs[Properties.neighbor_pairs_mask]

            # Get element contributions of the pairs
            Z_angular = self.angular_Z(Z)
            Z_ij = snn.neighbor_elements(Z_angular, idx_j)
            Z_ik = snn.neighbor_elements(Z_angular, idx_k)

            # Offset indices
            offset_idx_j = inputs[Properties.neighbor_offsets_j]
            offset_idx_k = inputs[Properties.neighbor_offsets_k]
            
            # Compute triple distances
            r_ij, r_ik, r_jk = snn.triple_distances(
                positions,
                idx_j,
                idx_k,
                offset_idx_j=offset_idx_j,
                offset_idx_k=offset_idx_k,
                cell=cell,
                cell_offsets=cell_offset,
            )

            angular_sf = self.ADF(
                r_ij,
                r_ik,
                r_jk,
                elemental_weights=(Z_ij, Z_ik),
                triple_masks=neighbor_pairs_mask,
            )
        else:
            angular_sf = None

        # Concatenate and return symmetry functions
        if self.RDF is None:
            symmetry_functions = angular_sf
        elif self.ADF is None:
            symmetry_functions = radial_sf
        else:
            symmetry_functions = torch.cat((radial_sf, angular_sf), 2)

        return symmetry_functions


class BehlerSFBlock(SymmetryFunctions):
    """
    Utility layer for fast initialisation of ACSFs and wACSFs.

    Args:
        n_radial (int):  Number of radial functions
        n_angular (int): Number of angular functions
        zetas (set of int): Set of exponents used to compute the angular term, default is zetas={1}
        cutoff_radius (float): Cutoff radius, default are 5 Angstrom
        elements (set of int): Nuclear charge present in the molecules, default is {1,6,7,8,9} (H,C,N,O and F).
        centered (bool): Whether centered Gaussians should be used for radial functions. Angular functions use centered Gaussians by default.
        crossterms (bool): Whether cutoff and exponential terms of the distance r_jk between both neighbors should be included in the angular functions. Default is False
        mode (str): Manner in which the weights are initialized. Possible are:
            weighted: Weigh symmetry functions with nuclear charges (wACSF)
            onehot: Represent elements by onehot vectors. This can be used in combination with pairwise_elements
                    in order to emulate the behavior of classic Behler symmetry functions (ACSF).
            embedding: Use random embeddings, which can e.g. be trained. Embedding length is modified via
                       len_embedding (default=False).
    """

    def __init__(
        self,
        n_radial=22,
        n_angular=5,
        zetas={1},
        cutoff_radius=5.0,
        elements=frozenset((1, 6, 7, 8, 9)),
        centered=False,
        crossterms=False,
        mode="weighted",
    ):
        # Determine mode.
        if mode == "weighted":
            initz = "weighted"
            pairwise_elements = False
        elif mode == "Behler":
            initz = "onehot"
            pairwise_elements = True
        else:
            raise NotImplementedError("Unrecognized symmetry function %s" % mode)

        # Construct symmetry functions.
        super(BehlerSFBlock, self).__init__(
            n_radial=n_radial,
            n_angular=n_angular,
            zetas=zetas,
            cutoff_radius=cutoff_radius,
            centered=centered,
            crossterms=crossterms,
            elements=elements,
            initz=initz,
            pairwise_elements=pairwise_elements,
        )

class APNet(nn.Module):
    def __init__(
        self,
        n_radial=43,
        n_ap=21,
        cutoff_radius=8.0,
        cutoff_radius2=None,
        sym_start=0.8,
        sym_cut=5.5,
        width_adjust=(0.5)**(0.5),
        width_adjust_ap=(2)**(0.5),
        elements=frozenset((1, 6, 7, 8, 9)),
        centered=False,
        sharez=True,
        trainz=False,
        mode="Behler",
        cutoff=snn.CosineCutoff,
        len_embedding=1,
        morse_bond=None,
        morse_mu=None,
        morse_res=None,
        morse_beta=10.0,
    ):
        super(APNet, self).__init__() 

         # Determine mode.
        if mode == "weighted":
            initz = "weighted"
            pairwise_elements = False
        elif mode == "Behler":
            initz = "onehot"
            pairwise_elements = True
        else:
            raise NotImplementedError("Unrecognized symmetry function %s" % mode)

        self.n_radial = n_radial
        self.n_ap = n_ap
        self.len_embedding = len_embedding
        self.n_elements = None

        self.cutoff_radius = cutoff_radius
        self.cutoff_radius2 = cutoff_radius2
        self.cutoff = cutoff(cutoff=self.cutoff_radius)
        if self.cutoff_radius2 is not None: self.cutoff2 = cutoff(cutoff=self.cutoff_radius2)
        else: self.cutoff2 = None
        self.sym_cut = sym_cut

        if morse_bond is not None:
            g1 = [i for i in morse_bond if isinstance(i, int)]
            if len(g1) < len(morse_bond):
                g2 = [i for i in morse_bond if isinstance(i, list)][0]
                self.morse1 = torch.tensor(g1).reshape(-1)
                self.morse2 = torch.tensor(g2)
                inc = -torch.ones_like(self.morse2)
                zero = torch.zeros_like(self.morse2)
                zero[self.morse2 > self.morse1] = inc[self.morse2 > self.morse1]
                self.morse2 = self.morse2 + zero
                self.morse1 += morse_res[0]
            else:
                morse1 = g1[0]
                morse2 = g1[1]
                if morse2 > morse1: morse2 -= 1
                self.morse1 = torch.tensor(morse1+morse_res[0]).reshape(-1)
                self.morse2 = torch.tensor(morse2).reshape(-1)
            self.fermi_dirac = FermiDirac(mu=morse_mu, beta=morse_beta)

        # Check for general stupidity:
        if self.n_ap < 1 and self.n_radial < 1:
            raise ValueError("At least one type of SF required")

        if self.n_radial > 0:
            # Get basic filters (if centered Gaussians are requested, start is set to 0.5
            if centered:
                radial_start = 1.0
            else:
                radial_start = 0.5
            self.radial_filter = snn.GaussianSmearing(
                start=sym_start,
                stop=self.sym_cut - 0.5,
                n_gaussians=n_radial,
                centered=centered,
                width_adjust=width_adjust
            )
            self.RDF = snn.RadialDistribution(
                self.radial_filter, cutoff_function=self.cutoff
            )
        else:
            self.RDF = None

        if self.n_ap > 0:
            self.radial_filter_ap = snn.GaussianSmearing(
                start=-1.0,
                stop=1.0,
                n_gaussians=n_ap,
                centered=centered,
                width_adjust=width_adjust_ap
            )
            self.APF = snn.APDistribution(
                self.radial_filter_ap, cutoff_functions=self.cutoff
            )
        
        else:
            self.APF = None

        self.radial_Z = self.initz(initz, elements)

        # check whether angular functions should use the same embedding
        if sharez:
            self.ap_Z = self.radial_Z
        else:
            self.ap_Z = self.initz(initz, elements)

        # Turn of training of embeddings unless requested explicitly
        if not trainz:
            # Turn off gradients
            self.radial_Z.weight.requires_grad = False
            self.ap_Z.weight.requires_grad = False
    
        # Compute total number of symmetry functions
        self.n_symfuncs = (
                self.n_radial + self.n_ap 
            ) * self.n_elements

    def initz(self, mode, elements):
        maxelements = max(elements)
        nelements = len(elements)

        if mode == "weighted":
            weights = torch.arange(maxelements + 1)[:, None]
            z_weights = nn.Embedding(maxelements + 1, 1)
            z_weights.weight.data = weights
            self.n_elements = 1
        elif mode == "onehot":
            weights = torch.zeros(maxelements + 1, nelements)
            for idx, Z in enumerate(elements):
                weights[Z, idx] = 1.0
            z_weights = nn.Embedding(maxelements + 1, nelements)
            z_weights.weight.data = weights
            self.n_elements = nelements
        elif mode == "embedding":
            z_weights = nn.Embedding(maxelements + 1, self.len_embedding)
            self.n_elements = self.len_embedding
        else:
            raise NotImplementedError(
                "Unregognized option {:s} for initializing elemental weights. Use 'weighted', 'onehot' or 'embedding'.".format(
                    mode
                )
            )

        return z_weights

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Nbatch x Natoms x Nsymmetry_functions Tensor containing ACSFs or wACSFs.

        """
        
        positions = inputs[Properties.R]
        Z = inputs[Properties.Z]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]

        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        # Compute radial functions
        if self.RDF is not None:
            ZA = inputs['ZA']
            ZB = inputs['ZB']
            # Get atom type embeddings
            Z_rad = self.radial_Z(Z)
            # Get atom types of neighbors
            Z_ij = snn.neighbor_elements(Z_rad, neighbors)
            # Compute distances
            
            distances = snn.atom_distances(
                positions,
                neighbors,
                neighbor_mask=neighbor_mask,
                cell=cell,
                cell_offsets=cell_offset,
            )
            
            radial_sf = self.RDF(
                distances, elemental_weights=Z_ij, neighbor_mask=neighbor_mask
            )
            
            mon_A = torch.arange(0, ZA.shape[1], 1, device=cell_offset.device)
            mon_B = torch.arange(ZA.shape[1], ZA.shape[1]+ZB.shape[1], 1, device=cell_offset.device)

            radial_sf_A = radial_sf[:, mon_A, :]
            radial_sf_B = radial_sf[:, mon_B, :]
        
        else:
            radial_sf = None

        if self.APF is not None:
            try:
                idx_j = inputs[Properties.neighbor_pairs_j]
                idx_k = inputs[Properties.neighbor_pairs_k]

            except KeyError as e:
                raise HDNNException(
                    "Angular symmetry functions require "
                    + "`collect_triples=True` in AtomsData."
                )

            ZA = inputs['ZA']
            ZB = inputs['ZB']

            neighbor_pairs_mask = inputs[Properties.neighbor_pairs_mask]

            neighbor_inter = inputs[Properties.neighbor_inter]
            offset_inter = inputs[Properties.neighbor_offset_inter]
            offset_intra = inputs[Properties.cell_offset_intra]
            neighbor_inter_mask = inputs[Properties.neighbor_inter_mask]

            distances_inter = snn.atom_distances(
                positions,
                neighbor_inter,
                neighbor_mask=neighbor_inter_mask,
                cell=cell,
                cell_offsets=offset_inter,
            )
            
            pair_dist = torch.ones_like(distances_inter, device=cell_offset.device)
            pair_dist[neighbor_inter_mask != 0] = distances_inter[neighbor_inter_mask != 0]
            inv_dist = 1/pair_dist[:, :, :]
            mon_A = torch.arange(0, ZA.shape[1], 1, device=cell_offset.device)
            mon_B = torch.arange(0, ZB.shape[1], 1, device=cell_offset.device)
            
            pair_dist = pair_dist[:, mon_A, :][:, :, mon_B]
            inv_dist = inv_dist[:, mon_A, :][:, :, mon_B]
            pair_dist = torch.reshape(pair_dist, (pair_dist.shape[0], pair_dist.shape[1]*pair_dist.shape[2]))
            inv_dist = torch.reshape(inv_dist, (inv_dist.shape[0], inv_dist.shape[1]*inv_dist.shape[2]))
            dists = torch.stack((pair_dist, inv_dist), -1)

            # Get element contributions of the pairs
            Z_ap = self.ap_Z(Z)
            Z_ik = snn.neighbor_elements(Z_ap, idx_k)

            # Offset indices
            offset_idx_j = inputs[Properties.neighbor_offsets_j]
            offset_idx_k = inputs[Properties.neighbor_offsets_k]
            
            # Compute triple distances
            r_ij, r_ik, r_jk = snn.triple_distances_apnet(
                positions,
                idx_j,
                idx_k,
                offset_idx_j=offset_idx_j,
                offset_idx_k=offset_idx_k,
                cell=cell,
                cell_offsets_intra=offset_intra,
                cell_offsets_inter=offset_inter,
            )
            
            ap_sf = self.APF(
                r_ij,
                r_ik,
                r_jk,
                elemental_weights=Z_ik,
                triple_masks=neighbor_pairs_mask,
            )
            
            mon_A = torch.arange(0, ZA.shape[1], 1, device=cell_offset.device)
            mon_B = torch.arange(ZA.shape[1], ZA.shape[1]+ZB.shape[1], 1, device=cell_offset.device)
            smon_B = torch.arange(0, ZB.shape[1], 1, device=cell_offset.device)
            ap_sf_A = ap_sf[:, mon_A, :, :][:, :, smon_B, :]
            ap_sf_B = ap_sf[:, mon_B, :]

            if self.cutoff2 is not None:
                cutoffs = self.cutoff2(pair_dist)
                
                cutoffs = cutoffs.unsqueeze(-1)
                #test_zero_A = test_zero_A.unsqueeze(-1)
                #test_zero_B = test_zero_B.unsqueeze(-1)

                dists = torch.mul(dists, cutoffs)
                #dists = torch.mul(dists, test_zero_A)
                #dists = torch.mul(dists, test_zero_B)
                if hasattr(self, 'fermi_dirac'):
                    morse_bond = distances[:, self.morse1, self.morse2].reshape(distances.shape[0], self.morse2.shape[0])
                    morse_bond = torch.min(morse_bond, dim=1, keepdim=True)[0]
                    fd = self.fermi_dirac(morse_bond)
                    fd = torch.repeat_interleave(fd, repeats=pair_dist.shape[1], dim=1)
                    fd = fd.unsqueeze(-1)
                    dists = torch.mul(dists, fd)
        
        else:
            ap_sf = None
        return radial_sf_A, radial_sf_B, ap_sf_A, ap_sf_B, dists

class APNet_mod(nn.Module):
    """
    Modified version of APNet which uses all pairs of atoms
    """
    def __init__(
        self,
        n_radial=43,
        n_ap=21,
        cutoff_radius=8.0,
        cutoff_react=None,
        cutoff_prod=None,
        sym_start=0.8,
        sym_cut=5.5,
        width_adjust=(0.5)**(0.5),
        width_adjust_ap=(2)**(0.5),
        elements=frozenset((1, 6, 7, 8, 9)),
        centered=False,
        sharez=True,
        trainz=False,
        mode="Behler",
        cutoff=snn.CosineCutoff,
        len_embedding=1,
        atom_t=None,
        atom_react=None,
        atom_prod=None,
        morse_beta=10.0,
    ):
        super(APNet_mod, self).__init__()

        # Determine mode.
        if mode == "weighted":
            initz = "weighted"
            pairwise_elements = False
        elif mode == "Behler":
            initz = "onehot"
            pairwise_elements = True
        else:
            raise NotImplementedError("Unrecognized symmetry function %s" % mode)

        self.n_radial = n_radial
        self.n_ap = n_ap
        self.len_embedding = len_embedding
        self.n_elements = None

        self.cutoff_radius = cutoff_radius
        self.cutoff = cutoff(cutoff=self.cutoff_radius)
        if atom_t is not None: 
            self.atom_t = torch.tensor(atom_t).reshape(-1)
            if isinstance(atom_react, int): 
                if atom_react < self.atom_t:
                    self.atom_react = torch.tensor(atom_react).reshape(-1)
                else:
                    self.atom_react = torch.tensor(atom_react - 1).reshape(-1)
                    #self.atom_react = torch.tensor(atom_react).reshape(-1)
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
                    #self.atom_prod = torch.tensor(atom_prod).reshape(-1)
            else:
                atom_prod = torch.tensor(atom_prod).reshape(-1)
                inc = -torch.ones_like(atom_prod)
                zero = torch.zeros_like(atom_prod)
                zero[atom_prod > self.atom_t] = inc[atom_prod > self.atom_t]
                self.atom_prod = atom_prod + zero
            self.fd_react = FermiDirac(mu=cutoff_react, beta=morse_beta)
            self.fd_prod = FermiDirac(mu=cutoff_prod, beta=morse_beta)

        else: self.cutoff2 = None
        self.sym_cut = sym_cut

        # Check for general stupidity:
        if self.n_ap < 1 and self.n_radial < 1:
            raise ValueError("At least one type of SF required")

        if self.n_radial > 0:
            # Get basic filters (if centered Gaussians are requested, start is set to 0.5
            if centered:
                radial_start = 1.0
            else:
                radial_start = 0.5
            self.radial_filter = snn.GaussianSmearing(
                start=sym_start,
                stop=self.sym_cut - 0.5,
                n_gaussians=n_radial,
                centered=centered,
                width_adjust=width_adjust
            )
            self.RDF = snn.RadialDistribution(
                self.radial_filter, cutoff_function=self.cutoff
            )
        else:
            self.RDF = None

        if self.n_ap > 0:
            self.radial_filter_ap = snn.GaussianSmearing(
                start=-1.0,
                stop=1.0,
                n_gaussians=n_ap,
                centered=centered,
                width_adjust=width_adjust_ap
            )
            self.APF = snn.APDistribution_Mod(
                self.radial_filter_ap, cutoff_functions=self.cutoff
            )

        else:
            self.APF = None

        self.radial_Z = self.initz(initz, elements)

        # check whether angular functions should use the same embedding
        if sharez:
            self.ap_Z = self.radial_Z
        else:
            self.ap_Z = self.initz(initz, elements)

        # Turn of training of embeddings unless requested explicitly
        if not trainz:
            # Turn off gradients
            self.radial_Z.weight.requires_grad = False
            self.ap_Z.weight.requires_grad = False

        # Compute total number of symmetry functions
        self.n_symfuncs = (
                self.n_radial + self.n_ap
            ) * self.n_elements

    def initz(self, mode, elements):
        maxelements = max(elements)
        nelements = len(elements)

        if mode == "weighted":
            weights = torch.arange(maxelements + 1)[:, None]
            z_weights = nn.Embedding(maxelements + 1, 1)
            z_weights.weight.data = weights
            self.n_elements = 1
        elif mode == "onehot":
            weights = torch.zeros(maxelements + 1, nelements)
            for idx, Z in enumerate(elements):
                weights[Z, idx] = 1.0
            z_weights = nn.Embedding(maxelements + 1, nelements)
            z_weights.weight.data = weights
            self.n_elements = nelements
        elif mode == "embedding":
            z_weights = nn.Embedding(maxelements + 1, self.len_embedding)
            self.n_elements = self.len_embedding
        else:
            raise NotImplementedError(
                "Unregognized option {:s} for initializing elemental weights. Use 'weighted', 'onehot' or 'embedding'.".format(
                    mode
                )
            )

        return z_weights

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Nbatch x Natoms x Nsymmetry_functions Tensor containing ACSFs or wACSFs.

        """
        positions = inputs[Properties.R]
        Z = inputs[Properties.Z]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]

        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        # Compute radial functions
        if self.RDF is not None:
            # Get atom type embeddings
            Z_rad = self.radial_Z(Z)
            # Get atom types of neighbors
            Z_ij = snn.neighbor_elements(Z_rad, neighbors)
            # Compute distances
            distances = snn.atom_distances(
                positions,
                neighbors,
                neighbor_mask=neighbor_mask,
                cell=cell,
                cell_offsets=cell_offset,
            )
            radial_sf = self.RDF(
                distances, elemental_weights=Z_ij, neighbor_mask=neighbor_mask
            )

        else:
            radial_sf = None

        if self.APF is not None:
            try:
                idx_j = inputs[Properties.neighbor_pairs_j]
                idx_k = inputs[Properties.neighbor_pairs_k]

            except KeyError as e:
                raise HDNNException(
                    "Angular symmetry functions require "
                    + "`collect_triples=True` in AtomsData."
                )

            neighbor_pairs_mask = inputs[Properties.neighbor_pairs_mask]

            pair_dist = distances
            inv_dist = 1/pair_dist[:, :, :]
            pair_dist = torch.reshape(pair_dist, (pair_dist.shape[0], pair_dist.shape[1]*pair_dist.shape[2]))
            inv_dist = torch.reshape(inv_dist, (inv_dist.shape[0], inv_dist.shape[1]*inv_dist.shape[2]))
            dists = torch.stack((pair_dist, inv_dist), -1)
            
            # Get element contributions of the pairs
            Z_ap = self.ap_Z(Z)
            Z_ik = snn.neighbor_elements(Z_ap, idx_k)

            # Offset indices
            offset_idx_j = inputs[Properties.neighbor_offsets_j]
            offset_idx_k = inputs[Properties.neighbor_offsets_k]

            # Compute triple distances
            r_ij, r_ik, r_jk = snn.triple_distances(
                positions,
                idx_j,
                idx_k,
                offset_idx_j=offset_idx_j,
                offset_idx_k=offset_idx_k,
                cell=cell,
                cell_offsets=cell_offset,
            )

            ap_sf = self.APF(
                r_ij,
                r_ik,
                r_jk,
                elemental_weights=Z_ik,
                triple_masks=neighbor_pairs_mask,
            )

            if self.fd_react is not None:
                atom_t_ind = self.atom_t * (positions.shape[1] - 1)
                sel_atom_react = atom_t_ind + self.atom_react
                react_dists = pair_dist[:, sel_atom_react].reshape(pair_dist.shape[0], sel_atom_react.shape[0])
                sel_atom_prod = atom_t_ind + self.atom_prod
                prod_dists = pair_dist[:, sel_atom_prod].reshape(pair_dist.shape[0], sel_atom_prod.shape[0])

                react_dists = torch.min(react_dists, dim=1, keepdim=True)[0]
                prod_dists = torch.min(prod_dists, dim=1, keepdim=True)[0]
                #trans_dist = torch.cat((react_dists, prod_dists), dim=-1)

                #cut_dist = torch.max(trans_dist, dim=1, keepdim=True)[0]
                #cutoffs = self.cutoff2(cut_dist)
                fd_react = self.fd_react(react_dists)
                fd_prod = self.fd_prod(prod_dists)
                fd = torch.mul(fd_react, fd_prod)
                #fd_react = torch.repeat_interleave(fd_react, repeats=pair_dist.shape[1], dim=1)
                #fd_react = fd_react.unsqueeze(-1)
                #fd_prod = torch.repeat_interleave(fd_prod, repeats=pair_dist.shape[1], dim=1)
                #fd_prod = fd_prod.unsqueeze(-1)
                #dists = torch.mul(dists, fd_react)
                #dists = torch.mul(dists, fd_prod)

        else:
            ap_sf = None

        return radial_sf, ap_sf, dists, fd

class StandardizeSF(nn.Module):
    """
    Compute mean and standard deviation of all symmetry functions computed for the molecules in the data loader
    and use them to standardize the descriptor vectors,

    Args:
        SFBlock (callable): Object for computing the descriptor vectors
        data_loader (object): DataLoader containing the molecules used for computing the statistics. If None, dummy
                              vectors are generated instead
        cuda (bool): Cuda flag
    """

    def __init__(self, SFBlock, data_loader=None, cuda=False):

        super(StandardizeSF, self).__init__()

        device = torch.device("cuda" if cuda else "cpu")

        self.n_symfuncs = SFBlock.n_symfuncs

        if data_loader is not None:
            symfunc_statistics = StatisticsAccumulator(batch=True, atomistic=True)
            SFBlock = SFBlock.to(device)

            for sample in data_loader:
                if cuda:
                    sample = {k: v.to(device) for k, v in sample.items()}
                symfunc_values = SFBlock.forward(sample)
                symfunc_statistics.add_sample(symfunc_values.detach())

            SF_mean, SF_stddev = symfunc_statistics.get_statistics()

        else:
            SF_mean = torch.zeros(self.n_symfuncs)
            SF_stddev = torch.ones(self.n_symfuncs)

        self.SFBlock = SFBlock
        self.standardize = snn.Standardize(SF_mean, SF_stddev)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Standardized representations.
        """
        representation = self.SFBlock(inputs)
        return self.standardize(representation)
