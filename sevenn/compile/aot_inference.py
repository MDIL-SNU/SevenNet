import warnings
from functools import partial
from typing import Any, Dict, List, Optional

import packaging.version
import torch
from e3nn.util.jit import prepare

from sevenn.atom_graph_data import AtomGraphData
from sevenn.checkpoint import SevenNetCheckpoint
from sevenn.train.dataload import unlabeled_atoms_to_graph

from ._nequip_fx import nequip_make_fx
from ._wrappers import (
    DictInputOutputWrapper,
    ListInputOutputWrapper,
    inputs_list,
    outputs_list,
)

_node_dim = torch.export.dynamic_shapes.Dim('node', min=2, max=torch.inf)
_edge_dim = torch.export.dynamic_shapes.Dim('edge', min=2, max=torch.inf)


_dyn_shapes = {
    'atom_type': {
        0: _node_dim,
        1: torch.export.Dim.STATIC,
    },
    'cell_volume': {},
    'edge_index': {
        0: torch.export.Dim.STATIC,
        1: _edge_dim,
    },
    'edge_vec': {
        0: _edge_dim,
        1: torch.export.Dim.STATIC,
    },
    'num_atoms': {},
}


def _dummy_data_factory(symbol: str, cutoff: float) -> AtomGraphData:
    from ase.build import bulk

    atoms = bulk(symbol, 'fcc', a=3.5).repeat((2, 2, 2))
    atoms.rattle()
    return AtomGraphData.from_numpy_dict(unlabeled_atoms_to_graph(atoms, cutoff))


def aot_export_compile_and_package(
    checkpoint: SevenNetCheckpoint,
    package_path: str = 'aot_compiled.pt2',
    test_data: Optional[AtomGraphData] = None,
    modal: Optional[str] = None,
    build_model_kwargs: Optional[Dict[str, Any]] = None,
    write_fx_code: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Export, compile, and package given SevenNet checkpoint with AOT.
    """
    cp = checkpoint  # just alias

    if packaging.version.parse(torch.__version__) < packaging.version.parse('2.6.0'):
        warnings.warn(f'Torch version: {torch.__version__}<2.6.0. Code will crash!')

    if not modal and cp.config.get('use_modality', False):
        raise ValueError('Modal is not given')

    if not package_path.endswith('.pt2'):
        raise ValueError('Path must be ends with .pt2')

    if not torch.cuda.is_available():
        raise ValueError('CUDA not available')

    sym = cp.config['chemical_species'][0]
    data = test_data or _dummy_data_factory(sym, float(cp.config['cutoff']))
    data = data.to('cuda')
    data_clone = data.clone()

    build_model_kwargs = build_model_kwargs or {}
    build_model_fn = partial(cp.build_model, **build_model_kwargs)
    build_model_fn = prepare(build_model_fn)  # remove jit things from legacy e3nn

    if verbose:
        print('[aot_inference] model build')

    model_ref = build_model_fn()

    if modal:
        model_ref.prepare_modal_deploy(modal)  # type: ignore

    model_ref.set_is_batch_data(False)  # type: ignore
    model_ref.to('cuda')
    model_ref.eval()

    outputs_ref = model_ref(data)

    data_preprocessed = model_ref._preprocess(data_clone)  # type: ignore

    model_to_trace = ListInputOutputWrapper(model_ref)  # type: ignore
    assert all([key in data_preprocessed for key in model_to_trace.input_keys])
    input_keys = model_to_trace.input_keys
    output_keys = model_to_trace.output_keys

    input_in_list = [data_preprocessed[k] for k in input_keys]

    if verbose:
        print(f'[aot_inference] input keys: {input_keys}')

    for param in model_to_trace.parameters():
        param.requires_grad_(False)
    model_to_trace.eval()
    model_to_trace = model_to_trace.to('cuda')

    if verbose:
        print('[aot_inference] make fx')

    fx_model = nequip_make_fx(model_to_trace, input_in_list)

    if write_fx_code:  # mostly debugging purpose
        with open(write_fx_code, 'w') as f:
            f.write(fx_model.code)

        if verbose:
            print(f'[aot_inference] fx code saved to {write_fx_code}')

    if verbose:
        print('[aot_inference] start export')

    exported = torch.export.export(
        fx_model,
        (*input_in_list,),
        dynamic_shapes=[_dyn_shapes.copy()[k] for k in input_keys],
    )

    if verbose:
        print('[aot_inference] start compiling')

    out_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=package_path,
    )

    if verbose:
        print('[aot_inference] compiled model saved. start santy check')

    # sanity check step
    aot_model = torch._inductor.aoti_load_package(out_path)
    aot_model = DictInputOutputWrapper(aot_model, input_keys, output_keys)

    outputs_aot = aot_model(data_preprocessed)

    atol = 1e-5
    rtol = 1e-5
    for key in output_keys:
        if not torch.allclose(
            outputs_ref[key], outputs_aot[key], rtol=rtol, atol=atol
        ):
            warnings.warn(
                (
                    f'{key} output is not within the tolerance (rtol={rtol}, '
                    + f'atol={atol}). \nReference: \n{outputs_ref[key]}. '
                    + f'\nAOT: \n{outputs_aot[key]}'
                )
            )

    if verbose:
        print('[aot_inference] santy check passed.')
