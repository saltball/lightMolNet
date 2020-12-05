# -*- coding: utf-8 -*-
from ase.units import eV, Bohr, Debye
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'lightMolNet'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


class Properties:
    """
    Keys to access structure properties in `lightMolNet.data.AtomsData`
    """

    # common
    UNIT = "_Unit"

    # geometry
    Z = "_atomic_numbers"
    charge = "_charge"
    atom_mask = "_atom_mask"
    position = "_positions"
    R = position
    cell = "_cell"
    pbc = "_pbc"
    neighbors = "_neighbors"
    neighbor_mask = "_neighbor_mask"
    cell_offset = "_cell_offset"
    neighbor_pairs_j = "_neighbor_pairs_j"
    neighbor_pairs_k = "_neighbor_pairs_k"
    neighbor_pairs_mask = "_neighbor_pairs_mask"
    neighbor_offsets_j = "_neighbor_offsets_j"
    neighbor_offsets_k = "_neighbor_offsets_k"

    # chemical properties
    energy = "energy"
    energy_U0 = "energy_U0"
    forces = "forces"
    stress = "stress"
    dipole_moment = "dipole_moment"
    total_dipole_moment = "total_dipole_moment"
    polarizability = "polarizability"
    iso_polarizability = "iso_polarizability"
    at_polarizability = "at_polarizability"
    charges = "charges"
    energy_contributions = "energy_contributions"
    shielding = "shielding"
    hessian = "hessian"
    dipole_derivatives = "dipole_derivatives"
    polarizability_derivatives = "polarizability_derivatives"
    electric_field = "electric_field"
    magnetic_field = "magnetic_field"
    dielectric_constant = "dielectric_constant"
    magnetic_moments = "magnetic_moments"

    properties = [
        energy,
        forces,
        stress,
        dipole_moment,
        polarizability,
        shielding,
        hessian,
        dipole_derivatives,
        polarizability_derivatives,
        electric_field,
        magnetic_field,
    ]

    properties_unit = {
        energy: eV,
        energy_U0: eV,
        dipole_moment: Debye,
        iso_polarizability: Bohr ** 3
    }

    external_fields = [electric_field, magnetic_field]

    electric_properties = [
        dipole_moment,
        dipole_derivatives,
        dipole_derivatives,
        polarizability_derivatives,
        polarizability,
    ]
    magnetic_properties = [shielding]

    required_grad = {
        energy: [],
        forces: [position],
        hessian: [position],
        dipole_moment: [electric_field],
        polarizability: [electric_field],
        dipole_derivatives: [electric_field, position],
        polarizability_derivatives: [electric_field, position],
        shielding: [magnetic_field, magnetic_moments],
    }
