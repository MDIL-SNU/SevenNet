import torch

from ase import io, Atoms
from sevenn.atom_graph_data import AtomGraphData
from sevenn.nn.node_embedding import get_type_mapper_from_specie
import sevenn._keys as KEY


#def test_parallel_model_inference(deployed_files, outcar):



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
    #print(data[KEY.POS])
    # overwrite x from one_hot to index list (see deploy)
    atomic_numbers = atoms.get_atomic_numbers()
    primitive_x = torch.LongTensor([type_map[num] for num in atomic_numbers])
    data[KEY.NODE_FEATURE] = primitive_x
    #print(data[KEY.ENERGY])
    #print(data[KEY.FORCE])

    data = data.to_dict()
    data = {k: v for k, v in data.items()}

    data[KEY.EDGE_VEC].requires_grad = True

    # Maybe neccessary for certain torch versions
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            continue
        data[k] = torch.tensor([v])
    """

    #data: Dict[str, torch.Tensor]
    infered = model(data)

    #print(infered[KEY.PRED_TOTAL_ENERGY])
    #print(infered[KEY.PRED_FORCE])


def pure_python_model(checkpoint, outcar):
    from sevenn.model_build import build_E3_equivariant_model
    # load chechkpoint
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    config = checkpoint['config']
    model_stct = checkpoint['model_state_dict']

    model = build_E3_equivariant_model(config)
    model.load_state_dict(model_stct)
    model.set_is_batch_data(False)
    model.eval()

    cutoff = config[KEY.CUTOFF]
    type_map = config[KEY.TYPE_MAP]

    atoms = io.read(outcar, format='vasp-out')
    data = AtomGraphData.data_for_E3_equivariant_model(atoms, cutoff, type_map)
    data = {k: v for k, v in data.items()}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            continue
        print(k)
        data[k] = torch.tensor([v])

    data[KEY.EDGE_VEC].requires_grad = True
    output = model(data)
    print(output)


def main():
    pure_python_model('/home/parkyutack/SEVENNet/example_inputs/valid_lmp/checkpoint_best.pth',
                      '/home/parkyutack/Odin/valid_lmp/OUTCAR_3')
    test_model_inference('/home/parkyutack/Odin/valid_lmp/deployed_model_best.pt',
                         '/home/parkyutack/Odin/valid_lmp/OUTCAR_3')


if __name__ == "__main__":
    main()
