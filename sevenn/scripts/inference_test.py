import torch

from ase import io, Atoms
from sevenn.atom_graph_data import AtomGraphData
from nn.node_embedding import get_type_mapper_from_specie
import _keys as KEY


def test_parallel_model_inference(deployed_files, outcar):



# for E3
def test_model_inference(deployed_file, outcar):
    extra = {"chemical_symbols_to_index": "", "cutoff": "", "num_species": "",
             "model_type": "", "version": "", "dtype": "", "time": ""}

    atoms = io.read(outcar, format='vasp-out')
    io.write('res_from_ase.dat', atoms, format='lammps-data')
    model = torch.jit.load(deployed_file, torch.device("cpu"), extra)
    chem = str(extra["chemical_symbols_to_index"], "utf-8").strip().split(' ')
    cutoff = float(extra["cutoff"])
    type_map = get_type_mapper_from_specie(chem)
    data = AtomGraphData.data_for_E3_equivariant_model(atoms, cutoff, type_map)
    print(data[KEY.POS])
    # overwrite x from one_hot to index list (see deploy)
    atomic_numbers = atoms.get_atomic_numbers()
    primitive_x = torch.LongTensor([type_map[num] for num in atomic_numbers])
    data[KEY.NODE_FEATURE] = primitive_x
    #print(data[KEY.ENERGY])
    #print(data[KEY.FORCE])
    infered = model(data.to_dict())
    print(infered[KEY.PRED_TOTAL_ENERGY])
    print(infered[KEY.PRED_FORCE])


def main():
    test_model_inference('/home/parkyutack/sevenn/working_dir/deployed_model.pt',
                         '/home/parkyutack/sevenn/working_dir/OUTCAR')


if __name__ == "__main__":
    main()
