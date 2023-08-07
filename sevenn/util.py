from ase.data import atomic_numbers

import sevenn._keys as KEY
import sevenn._const
from sevenn.nn.node_embedding import get_type_mapper_from_specie


def chemical_species_preprocess(input_chem):
    config = {}
    chemical_specie = sorted([x.strip() for x in input_chem])
    config[KEY.CHEMICAL_SPECIES] = chemical_specie
    config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER] = \
        [atomic_numbers[x] for x in chemical_specie]
    config[KEY.NUM_SPECIES] = len(chemical_specie)
    config[KEY.TYPE_MAP] = get_type_mapper_from_specie(chemical_specie)
    #print(config[KEY.TYPE_MAP])
    #print(config[KEY.NUM_SPECIES])  # why
    #print(config[KEY.CHEMICAL_SPECIES])  # we need
    #print(config[KEY.CHEMICAL_SPECIES_BY_ATOMIC_NUMBER])  # all of this?
    return config

