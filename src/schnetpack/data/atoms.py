"""
This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import os, sys
import warnings
from base64 import b64encode, b64decode

import numpy as np
import torch
from ase.db import connect
from torch.utils.data import Dataset

from schnetpack import Properties
from schnetpack.environment import SimpleEnvironmentProvider, collect_atom_triples
from .partitioning import train_test_split
from schnetpack.nn.neighbors import atom_distances
logger = logging.getLogger(__name__)

__all__ = [
    "AtomsData",
    "AtomsDataError",
    "AtomsConverter",
    "get_center_of_mass",
    "get_center_of_geometry",
]


def get_center_of_mass(atoms):
    """
    Computes center of mass.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of mass
    """
    masses = atoms.get_masses()
    return np.dot(masses, atoms.arrays["positions"]) / masses.sum()


def get_center_of_geometry(atoms):
    """
    Computes center of geometry.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of geometry
    """
    return atoms.arrays["positions"].mean(0)


class AtomsDataError(Exception):
    pass


class AtomsData(Dataset):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database. Use together with schnetpack.data.AtomsLoader to feed data
    to your model.

    To improve the performance, the data is not stored in string format,
    as usual in the ASE database. Instead, it is encoded as binary before being written
    to the database. Reading work both with binary-encoded as well as
    standard ASE files.

    Args:
        dbpath (str): path to directory containing database.
        subset (list, optional): indices to subset. Set to None for entire database.
        available_properties (list, optional): complete set of physical properties
            that are contained in the database.
        load_only (list, optional): reduced set of properties to be loaded
        units (list, optional): definition of units for all available properties
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
    """

    ENCODING = "utf-8"

    def __init__(
        self,
        dbpath,
        subset=None,
        available_properties=None,
        load_only=None,
        units=None,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        centering_function=get_center_of_mass,
    ):
        if not dbpath.endswith(".db"):
            raise AtomsDataError(
                "Invalid dbpath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        self.dbpath = dbpath
        self.subset = subset
        self.load_only = load_only
        self.available_properties = self.get_available_properties(available_properties)
        if load_only is None:
            self.load_only = self.available_properties
        if units is None:
            units = [1.0] * len(self.available_properties)
        self.units = dict(zip(self.available_properties, units))
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.centering_function = centering_function

    def get_available_properties(self, available_properties):
        """
        Get available properties from argument or database.

        Args:
            available_properties (list or None): all properties of the dataset

        Returns:
            (list): all properties of the dataset
        """
        # use the provided list
        if not os.path.exists(self.dbpath):
            if available_properties is None:
                raise AtomsDataError(
                    "Please define available_properties or set "
                    "db_path to an existing database!"
                )
            return available_properties
        # read database properties
        with connect(self.dbpath) as conn:
            atmsrw = conn.get(1)
            db_properties = list(atmsrw.data.keys())
            db_properties = [prop for prop in db_properties if not prop.startswith("_")]
        # check if properties match
        if available_properties is None or set(db_properties) == set(
            available_properties
        ):
            return db_properties

        raise AtomsDataError(
            "The available_properties {} do not match the "
            "properties in the database {}!".format(available_properties, db_properties)
        )

    def create_splits(self, num_train=None, num_val=None, split_file=None):
        warnings.warn(
            "create_splits is deprecated, "
            + "use schnetpack.data.train_test_split instead",
            DeprecationWarning,
        )
        return train_test_split(self, num_train, num_val, split_file)

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.
        Args:
            idx (numpy.ndarray): subset indices

        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = (
            idx if self.subset is None or len(idx) == 0 else np.array(self.subset)[idx]
        )
        return type(self)(
            dbpath=self.dbpath,
            subset=subidx,
            load_only=self.load_only,
            environment_provider=self.environment_provider,
            collect_triples=self.collect_triples,
            centering_function=self.centering_function,
            available_properties=self.available_properties,
        )

    def __len__(self):
        if self.subset is None:
            with connect(self.dbpath) as conn:
                return conn.count()
        return len(self.subset)

    def __getitem__(self, idx):
        at, properties = self.get_properties(idx)
        properties["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))

        return properties

    def _subset_index(self, idx):
        # get row
        if self.subset is None:
            idx = int(idx)
        else:
            idx = int(self.subset[idx])
        return idx

    def get_atoms(self, idx):
        """
        Return atoms of provided index.

        Args:
            idx (int): atoms index

        Returns:
            ase.Atoms: atoms data

        """
        idx = self._subset_index(idx)
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()
        return at

    def get_metadata(self, key=None):
        """
        Returns an entry from the metadata dictionary of the ASE db.

        Args:
            key: Name of metadata entry. Return full dict if `None`.

        Returns:
            value: Value of metadata entry or full metadata dict, if key is `None`.

        """
        with connect(self.dbpath) as conn:
            if key is None:
                return conn.metadata
            if key in conn.metadata.keys():
                return conn.metadata[key]
        return None

    def set_metadata(self, metadata=None, **kwargs):
        """
        Sets the metadata dictionary of the ASE db.

        Args:
            metadata (dict): dictionary of metadata for the ASE db
            kwargs: further key-value pairs for convenience
        """

        # merge all metadata
        if metadata is not None:
            kwargs.update(metadata)

        with connect(self.dbpath) as conn:
            conn.metadata = kwargs

    def update_metadata(self, data):
        with connect(self.dbpath) as conn:
            metadata = conn.metadata
        metadata.update(data)
        self.set_metadata(metadata)

    def _add_system(self, conn, atoms, **properties):
        data = {}
        
        for pname in self.available_properties:
            try:
                prop = properties[pname]
            except:
                raise AtomsDataError("Required property missing:" + pname)

            try:
                pshape = prop.shape
                ptype = prop.dtype
            except:
                raise AtomsDataError(
                    "Required property `" + pname + "` has to be `numpy.ndarray`."
                )

            base64_bytes = b64encode(prop.tobytes())
            base64_string = base64_bytes.decode(AtomsData.ENCODING)
            data[pname] = base64_string
            data["_shape_" + pname] = pshape
            data["_dtype_" + pname] = str(ptype)

        conn.write(atoms, data=data)

    def add_system(self, atoms, **properties):
        """
        Add atoms data to the dataset.

        Args:
            atoms (ase.Atoms): system composition and geometry
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        with connect(self.dbpath) as conn:
            self._add_system(conn, atoms, **properties)

    def add_systems(self, atoms_list, property_list):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list (list of ase.Atoms): system composition and geometry
            property_list (list): Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset.

        """
        with connect(self.dbpath) as conn:
            for at, prop in zip(atoms_list, property_list):
                self._add_system(conn, at, **prop)

    def get_properties(self, idx):
        """
        Return property dictionary at given index.

        Args:
            idx (int): data index

        Returns:

        """
        idx = self._subset_index(idx)
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()

        # extract properties
        properties = {}
        for pname in self.load_only:
            # new data format
            try:
                shape = row.data["_shape_" + pname]
                dtype = row.data["_dtype_" + pname]
                prop = np.frombuffer(b64decode(row.data[pname]), dtype=dtype)
                prop = prop.reshape(shape)
            except:
                # fallback for properties stored directly
                # in the row
                if pname in row:
                    prop = row[pname]
                else:
                    prop = row.data[pname]

                try:
                    prop.shape
                except AttributeError as e:
                    prop = np.array([prop], dtype=np.float32)

            properties[pname] = torch.FloatTensor(prop)

        # extract/calculate structure
        properties = _convert_atoms(
            at,
            environment_provider=self.environment_provider,
            collect_triples=self.collect_triples,
            centering_function=self.centering_function,
            output=properties,
        )

        return at, properties

    def _get_atomref(self, property):
        """
        Returns single atom reference values for specified `property`.

        Args:
            property (str): property name

        Returns:
            list: list of atomrefs
        """
        labels = self.get_metadata("atref_labels")
        if labels is None:
            return None

        col = [i for i, l in enumerate(labels) if l == property]
        assert len(col) <= 1

        if len(col) == 1:
            col = col[0]
            atomref = np.array(self.get_metadata("atomrefs"))[:, col : col + 1]
        else:
            atomref = None

        return atomref

    def get_atomref(self, properties):
        """
        Return multiple single atom reference values as a dictionary.

        Args:
            properties (list or str): Desired properties for which the atomrefs are
                calculated.

        Returns:
            dict: atomic references
        """
        if type(properties) is not list:
            properties = [properties]
        return {p: self._get_atomref(p) for p in properties}


def _convert_atoms(
    atoms,
    environment_provider=SimpleEnvironmentProvider(),
    collect_triples=False,
    centering_function=get_center_of_mass,
    output=None,
    res_list=None,
):
    """
        Helper function to convert ASE atoms object to SchNetPack input format.

        Args:
            atoms (ase.Atoms): Atoms object of molecule
            environment_provider (callable): Neighbor list provider.
            collect_triples (bool, optional): Set to True if angular features are needed.
            centering_function (callable or None): Function for calculating center of
                molecule (center of mass/geometry/...). Center will be subtracted from
                positions.
            output (dict): Destination for converted atoms, if not None

    Returns:
        dict of torch.Tensor: Properties including neighbor lists and masks
            reformated into SchNetPack input format.
    """
    if output is None:
        inputs = {}
    else:
        inputs = output

    # Elemental composition
    cell = np.array(atoms.cell.array, dtype=np.float32)  # get cell array

    inputs[Properties.Z] = torch.LongTensor(atoms.numbers.astype(np.int))
    positions = atoms.positions.astype(np.float32)
    if centering_function:
        positions -= centering_function(atoms)
    inputs[Properties.R] = torch.FloatTensor(positions)
    inputs[Properties.cell] = torch.FloatTensor(cell)

    if "AP" not in type(environment_provider).__name__:

        # get atom environment
        nbh_idx, offsets = environment_provider.get_environment(atoms)

        # Get neighbors and neighbor mask
        inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))

        # Get cells
        inputs[Properties.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))

        # If requested get neighbor lists for triples
        if collect_triples:
            # Construct possible permutations
            nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)

            inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
            inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

            inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
                offset_idx_j.astype(np.int)
            )
            inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
                offset_idx_k.astype(np.int)
            )

    elif type(environment_provider).__name__ == "APModEnvironmentProvider":
        # get atom environment
        nbh_idx, offsets = environment_provider.get_environment(atoms)
        
        # Get neighbors and neighbor mask
        inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))

        # Get cells
        inputs[Properties.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))
        ZB = atoms.numbers.astype(np.int)
        natoms, nneigh = nbh_idx.shape
        ZB = np.tile(
                ZB[np.newaxis], (natoms, 1)
            )

        ZB = ZB[
                ~np.eye(natoms, dtype=np.bool)
            ].reshape(natoms, natoms - 1)

        inputs["ZB"] = torch.LongTensor(ZB)

        # If requested get neighbor lists for triples
        if collect_triples:

            # Construct possible permutations
            nbh_idx_j = np.tile(nbh_idx, nneigh)
            nbh_idx_k = np.repeat(nbh_idx, nneigh).reshape((natoms, -1))

            nbh_idx_j_tmp = nbh_idx_j[nbh_idx_j != nbh_idx_k].reshape(natoms, nneigh*(nneigh-1))
            nbh_idx_k_tmp = nbh_idx_k[nbh_idx_j != nbh_idx_k].reshape(natoms, nneigh*(nneigh-1))

            # Keep track of periodic images
            offset_idx = np.tile(np.arange(nneigh), (natoms, 1))

            # Construct indices for pairs of offsets
            offset_idx_j = np.tile(offset_idx, nneigh)
            offset_idx_k = np.repeat(offset_idx, nneigh).reshape((natoms, -1))

            offset_idx_j_tmp = offset_idx_j[nbh_idx_j != nbh_idx_k].reshape(natoms, nneigh*(nneigh-1))
            offset_idx_k_tmp = offset_idx_k[nbh_idx_j != nbh_idx_k].reshape(natoms, nneigh*(nneigh-1))

            inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j_tmp.astype(np.int))
            inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k_tmp.astype(np.int))

            inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
                offset_idx_j_tmp.astype(np.int)
            )
            inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
                offset_idx_k_tmp.astype(np.int)
            )
        
        """
        # If requested get neighbor lists for triples
        if collect_triples:
            # Construct possible permutations
            nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)

            inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
            inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

            inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
                offset_idx_j.astype(np.int)
            )
            inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
                offset_idx_k.astype(np.int)
            )
        """
    elif type(environment_provider).__name__ == "APModPBCEnvironmentProvider":
        # get atom environment
        nbh_idx, offsets = environment_provider.get_environment(atoms)

        # Get neighbors and neighbor mask
        inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))

        neighbor_test = inputs[Properties.neighbors].unsqueeze(0)
        pos_test = inputs[Properties.R].unsqueeze(0)
        cell_test = inputs[Properties.cell].unsqueeze(0)

        # Get cells
        inputs[Properties.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))
        offset_test = torch.FloatTensor(offsets.astype(np.float32)).unsqueeze(0)

        distances, displacement = atom_distances(pos_test, neighbor_test, cell_test, offset_test, return_vecs=True)
        displacement = displacement.squeeze(0)

        #only works for orthorhombic box
        box_shift = -torch.round(displacement/cell_test[0, 0, 0])
        inputs[Properties.cell_offset] = torch.FloatTensor(box_shift)

        ZB = atoms.numbers.astype(np.int)
        natoms, nneigh = nbh_idx.shape
        ZB = np.tile(
                ZB[np.newaxis], (natoms, 1)
            )

        ZB = ZB[
                ~np.eye(natoms, dtype=np.bool)
            ].reshape(natoms, natoms - 1)

        inputs["ZB"] = torch.LongTensor(ZB)

        # If requested get neighbor lists for triples
        if collect_triples:

            # Construct possible permutations
            nbh_idx_j = np.tile(nbh_idx, nneigh)
            nbh_idx_k = np.repeat(nbh_idx, nneigh).reshape((natoms, -1))

            nbh_idx_j_tmp = nbh_idx_j[nbh_idx_j != nbh_idx_k].reshape(natoms, nneigh*(nneigh-1))
            nbh_idx_k_tmp = nbh_idx_k[nbh_idx_j != nbh_idx_k].reshape(natoms, nneigh*(nneigh-1))

            # Keep track of periodic images
            offset_idx = np.tile(np.arange(nneigh), (natoms, 1))

            # Construct indices for pairs of offsets
            offset_idx_j = np.tile(offset_idx, nneigh)
            offset_idx_k = np.repeat(offset_idx, nneigh).reshape((natoms, -1))

            offset_idx_j_tmp = offset_idx_j[nbh_idx_j != nbh_idx_k].reshape(natoms, nneigh*(nneigh-1))
            offset_idx_k_tmp = offset_idx_k[nbh_idx_j != nbh_idx_k].reshape(natoms, nneigh*(nneigh-1))

            inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j_tmp.astype(np.int))
            inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k_tmp.astype(np.int))

            inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
                offset_idx_j_tmp.astype(np.int)
            )
            inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
                offset_idx_k_tmp.astype(np.int)
            )


    elif type(environment_provider).__name__ == "APNetPBCEnvironmentProvider":
        if res_list is not None:
            monA = len(res_list[0])
            monB = len(res_list[1])
            inputs['ZA'] = torch.LongTensor(atoms.numbers[0:monA].astype(np.int))
            inputs['ZB'] = torch.LongTensor(atoms.numbers[monA:monA+monB].astype(np.int))

        inputs['ZA'], inputs['ZB'] = inputs['ZA'].long(), inputs['ZB'].long()

        # get atom environment
        nbh_idx_intra, offset_intra, nbh_idx_inter, offsets_inter = environment_provider.get_environment(atoms, inputs)
        
        # Get neighbors and neighbor mask
        inputs[Properties.neighbor_inter] = torch.LongTensor(nbh_idx_inter.astype(np.int))

        mask = inputs[Properties.neighbor_inter] >= 0
        inputs[Properties.neighbor_inter_mask] = mask.float()
        inputs[Properties.neighbor_inter] = (
            inputs[Properties.neighbor_inter] * inputs[Properties.neighbor_inter_mask].long()
        )

        neighbor_test = inputs[Properties.neighbor_inter].unsqueeze(0)
        pos_test = inputs[Properties.R].unsqueeze(0)
        cell_test = inputs[Properties.cell].unsqueeze(0)

        # Get cells
        offset_test = torch.FloatTensor(offsets_inter.astype(np.float32)).unsqueeze(0)

        distances, displacement = atom_distances(pos_test, neighbor_test, cell_test, offset_test, return_vecs=True)
        displacement = displacement.squeeze(0)

        #only works for orthorhombic box
        box_shift = -torch.round(displacement/cell_test[0, 0, 0])
        distances = atom_distances(pos_test, neighbor_test, cell_test, box_shift.unsqueeze(0))
        inputs[Properties.neighbor_offset_inter] = torch.FloatTensor(box_shift)

        natoms, nneigh = nbh_idx_inter.shape
        nbh_idx_k = np.tile(nbh_idx_intra, nneigh)
        nbh_idx_j = np.repeat(nbh_idx_inter, nneigh).reshape((natoms, -1))

        offset_idx = np.tile(np.arange(nneigh), (natoms, 1))
        offset_idx_k = np.tile(offset_idx, nneigh)
        offset_idx_j = np.repeat(offset_idx, nneigh).reshape((natoms, -1))

        inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
        inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

        inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
            offset_idx_j.astype(np.int)
        )
        inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
            offset_idx_k.astype(np.int)
        )

        mask_triples = np.ones_like(inputs[Properties.neighbor_pairs_j].numpy())
        mask_triples[inputs[Properties.neighbor_pairs_j].numpy() < 0] = 0
        mask_triples[inputs[Properties.neighbor_pairs_k].numpy() < 0] = 0

        neighbor_test = torch.LongTensor(nbh_idx_intra.astype(np.int)).unsqueeze(0)
        pos_test = inputs[Properties.R].unsqueeze(0)
        cell_test = inputs[Properties.cell].unsqueeze(0)

        # Get cells
        offset_test = torch.FloatTensor(offset_intra.astype(np.float32)).unsqueeze(0)

        distances, displacement = atom_distances(pos_test, neighbor_test, cell_test, offset_test, return_vecs=True)
        displacement = displacement.squeeze(0)

        #only works for orthorhombic box
        box_shift = -torch.round(displacement/cell_test[0, 0, 0])
        inputs[Properties.cell_offset_intra] = torch.FloatTensor(box_shift)

        mask_self = np.repeat(np.arange(0, nbh_idx_k.shape[0]), nbh_idx_k.shape[1]).reshape(nbh_idx_k.shape[0], nbh_idx_k.shape[1])
        mask_triples[mask_self == nbh_idx_k] = 0
        inputs[Properties.neighbor_pairs_mask] = torch.LongTensor(mask_triples.astype(np.float))
        
        mask_self = np.repeat(np.arange(0, nbh_idx_intra.shape[0]), nbh_idx_intra.shape[1]).reshape(nbh_idx_intra.shape[0], nbh_idx_intra.shape[1])
        neighborhood_idx = nbh_idx_intra[mask_self != nbh_idx_intra].reshape(nbh_idx_intra.shape[0], nbh_idx_intra.shape[1] - 1)
        inputs[Properties.neighbors] = torch.LongTensor(neighborhood_idx.astype(np.int))

        box_shift = box_shift[mask_self != nbh_idx_intra, :].reshape(nbh_idx_intra.shape[0], nbh_idx_intra.shape[1] - 1, 3)
        inputs[Properties.cell_offset] = torch.FloatTensor(box_shift)

    else:
        
        if res_list is not None:
            monA = len(res_list[0])
            monB = len(res_list[1])
            inputs['ZA'] = torch.LongTensor(atoms.numbers[0:monA].astype(np.int))
            inputs['ZB'] = torch.LongTensor(atoms.numbers[monA:monA+monB].astype(np.int))
      
        inputs['ZA'], inputs['ZB'] = inputs['ZA'].long(), inputs['ZB'].long()

        # get atom environment
        nbh_idx_intra, offset_intra, nbh_idx_inter, offsets_inter = environment_provider.get_environment(atoms, inputs)
        nbh_idx_test, offset_test = SimpleEnvironmentProvider().get_environment(atoms)
        #print(nbh_idx_test[0, :])
        #sys.exit()
        # Get neighbors and neighbor mask
        inputs[Properties.neighbor_inter] = torch.LongTensor(nbh_idx_inter.astype(np.int))
        
        mask = inputs[Properties.neighbor_inter] >= 0
        inputs[Properties.neighbor_inter_mask] = mask.float()
        inputs[Properties.neighbor_inter] = (
            inputs[Properties.neighbor_inter] * inputs[Properties.neighbor_inter_mask].long()
        )

        # Get cells
        inputs[Properties.neighbor_offset_inter] = torch.FloatTensor(offsets_inter.astype(np.float32))

        natoms, nneigh = nbh_idx_inter.shape
        nbh_idx_k = np.tile(nbh_idx_intra, nneigh)
        nbh_idx_j = np.repeat(nbh_idx_inter, nneigh).reshape((natoms, -1))

        offset_idx = np.tile(np.arange(nneigh), (natoms, 1))
        offset_idx_k = np.tile(offset_idx, nneigh)
        offset_idx_j = np.repeat(offset_idx, nneigh).reshape((natoms, -1))

        inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
        inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

        inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
            offset_idx_j.astype(np.int)
        )
        inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
            offset_idx_k.astype(np.int)
        )

        mask_triples = np.ones_like(inputs[Properties.neighbor_pairs_j].numpy())
        mask_triples[inputs[Properties.neighbor_pairs_j].numpy() < 0] = 0
        mask_triples[inputs[Properties.neighbor_pairs_k].numpy() < 0] = 0

        mask_self = np.repeat(np.arange(0, nbh_idx_k.shape[0]), nbh_idx_k.shape[1]).reshape(nbh_idx_k.shape[0], nbh_idx_k.shape[1])
        mask_triples[mask_self == nbh_idx_k] = 0
        inputs[Properties.neighbor_pairs_mask] = torch.LongTensor(mask_triples.astype(np.float))

        mask_self = np.repeat(np.arange(0, nbh_idx_intra.shape[0]), nbh_idx_intra.shape[1]).reshape(nbh_idx_intra.shape[0], nbh_idx_intra.shape[1])     
        neighborhood_idx = nbh_idx_intra[mask_self != nbh_idx_intra].reshape(nbh_idx_intra.shape[0], nbh_idx_intra.shape[1] - 1)
        inputs[Properties.neighbors] = torch.LongTensor(neighborhood_idx.astype(np.int))

        inputs[Properties.cell_offset_intra] = torch.FloatTensor(offset_intra.astype(np.float32))

        offset_intra = offset_intra[mask_self != nbh_idx_intra, :].reshape(nbh_idx_intra.shape[0], nbh_idx_intra.shape[1] - 1, 3)
        inputs[Properties.cell_offset] = torch.FloatTensor(offset_intra.astype(np.float32))
    return inputs

class AtomsConverter:
    """
    Convert ASE atoms object to an input suitable for the SchNetPack
    ML models.

    Args:
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        device (str): Device for computation (default='cpu')
    """

    def __init__(
        self,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        device=torch.device("cpu"),
        res_list=None,
    ):
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.res_list = res_list
        # Get device
        self.device = device

    def __call__(self, atoms):
        """
        Args:
            atoms (ase.Atoms): Atoms object of molecule

        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks
                reformated into SchNetPack input format.
        """
        inputs = _convert_atoms(atoms, self.environment_provider, self.collect_triples, res_list=self.res_list)
        
        # Calculate masks
        inputs[Properties.atom_mask] = torch.ones_like(inputs[Properties.Z]).float()
        mask = inputs[Properties.neighbors] >= 0
        inputs[Properties.neighbor_mask] = mask.float()
        inputs[Properties.neighbors] = (
            inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].long()
        )

        if self.collect_triples:
            mask_triples = torch.ones_like(inputs[Properties.neighbor_pairs_j])
            mask_triples[inputs[Properties.neighbor_pairs_j] < 0] = 0
            mask_triples[inputs[Properties.neighbor_pairs_k] < 0] = 0
            inputs[Properties.neighbor_pairs_mask] = mask_triples.float()
        
        # Add batch dimension and move to CPU/GPU
        for key, value in inputs.items():
            inputs[key] = value.unsqueeze(0).to(self.device)

        return inputs
