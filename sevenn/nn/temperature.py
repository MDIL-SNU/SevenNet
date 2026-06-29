from typing import Dict, List, Optional
import warnings

import math
from scipy.special import lambertw

import torch
import torch.nn as nn
import torch.nn.functional
from e3nn.o3 import Linear, Irreps
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


class BaseBasis(nn.Module):
    """
    f : T (*, 1) -> [g(T/softplus(T0)-relu(shift))] (*, num_basis)
    """
    def __init__(
        self,
        num_basis: int=8,
        initial_T0: Optional[List]=None,
        initial_shift: Optional[List]=None,
        trainable_coeff: bool=True,
        softplus_params: Optional[Dict]={},
        as_gate: bool=False,
    ) -> None:
        super().__init__()
        if initial_T0 is None:
            self.initial_T0 = torch.ones(dtype=torch.float32) * 200.
        elif isinstance(initial_T0, list):
            assert len(initial_T0) == num_basis
            self.initial_T0 = torch.tensor(initial_T0, dtype=torch.float32)
        elif isinstance(initial_T0, (float, int)):
            self.initial_T0 = torch.tensor(
                [initial_T0 for _ in range(num_basis)], dtype=torch.float32
            )
        else:
            raise ValueError('Initial T0 value should be list of float or float/int')

        if initial_shift is None:
            self.initial_shift = torch.linspace(0, 7, num_basis, dtype=torch.float32)
        elif isinstance(initial_shift, list):
            assert len(initial_shift) == num_basis
            self.initial_shift = torch.tensor(initial_shift, dtype=torch.float32)
        elif isinstance(initial_shift, (float, int)):
            self.initial_shift = torch.tensor(
                [initial_shift for _ in range(num_basis)], dtype=torch.float32
            )
        else:
            raise ValueError('Initial shift value should be list of float or float/int')

        self.num_basis = num_basis
        self.initial_T0 = nn.Parameter(self.initial_T0, requires_grad=trainable_coeff)
        self.initial_shift = nn.Parameter(self.initial_shift, requires_grad=trainable_coeff)

        self.softplus = nn.Softplus(**softplus_params)
        self.relu = nn.ReLU()
        self.enc_function = None
        self.as_gate = as_gate

    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        t = temperature.unsqueeze(-1)
        val = self.enc_function(t / self.softplus(self.initial_T0) - self.relu(self.initial_shift))
        val[torch.isinf(t).repeat((1, self.num_basis))] = 0.
        if self.as_gate:
            return 1. - val
        return val

"""
class SigmoidBasis(BaseBasis):
    def __init__(
        self,
        num_basis: int=8,
        initial_T0: Optional[List]=None,
        initial_shift: Optional[List]=None,
        trainable_coeff: bool=True,
        softplus_params: Optional[Dict]={},
        **kwargs,
    ) -> None:
        super().__init__(
            num_basis=num_basis,
            initial_T0=initial_T0,
            initial_shift=initial_shift,
            trainable_coeff=trainable_coeff,
            softplus_params=softplus_params,
        )
        self.enc_function = nn.Sigmoid(**kwargs)
        

class TanhBasis(BaseBasis):
    def __init__(
        self,
        num_basis: int=8,
        initial_T0: Optional[List]=None,
        initial_shift: Optional[List]=None,
        trainable_coeff: bool=True,
        softplus_params: Optional[Dict]={},
        **kwargs,
    ) -> None:
        super().__init__(
            num_basis=num_basis,
            initial_T0=initial_T0,
            initial_shift=initial_shift,
            trainable_coeff=trainable_coeff,
            softplus_params=softplus_params,
        )
        self.enc_function = nn.Tanh(**kwargs)
        

class AvramiBasis(BaseBasis):
    def __init__(
        self,
        num_basis: int=8,
        initial_T0: Optional[List]=None,
        initial_shift: Optional[List]=None,
        trainable_coeff: bool=True,
        softplus_params: Optional[Dict]={},
        initial_exp: Optional[List]=None,
        trainable_exp: bool=True,
        **kwargs,
    ) -> None:
        super().__init__(
            num_basis=num_basis,
            initial_T0=initial_T0,
            initial_shift=initial_shift,
            trainable_coeff=trainable_coeff,
            softplus_params=softplus_params,
        )
        if initial_exp is None:
            self.initial_exp = torch.ones(num_basis)
        elif isinstance(initial_exp, list):
            assert len(initial_exp) == num_basis
            self.initial_exp = torch.tensor(initial_exp, dtype=torch.float32)
        elif isinstance(initial_exp, (float, int)):
            self.initial_exp = torch.tensor(
                [initial_exp for _ in range(num_basis)], dtype=torch.float32
            )
        else:
            raise ValueError('Initial exp should be one of list of float, float or int')

        if trainable_exp:
            self.initial_exp = nn.Parameter(self.initial_exp)

        self.relu = nn.ReLU()
        self.enc_function = lambda x: 1 - torch.exp(-torch.pow(x, self.relu(self.initial_exp)))
"""

class GaussianEncoding(BaseBasis):
    def __init__(
        self,
        num_basis: int=8,
        initial_T0: Optional[List]=None,
        initial_shift: Optional[List]=None,
        trainable_coeff: bool=True,
        softplus_params: Optional[Dict]={},
        as_gate: bool=False,
        decay_x0: Optional[float] = None,
        pow_p: Optional[int] = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            num_basis=num_basis,
            initial_T0=initial_T0,
            initial_shift=initial_shift,
            trainable_coeff=trainable_coeff,
            softplus_params=softplus_params,
            as_gate=as_gate,
        )
        #self.enc_function = lambda x: torch.where(x < 0, 1., torch.exp(-torch.pow(x/decay_x0, 2)))
        if decay_x0 is None:
            w = lambertw(-(2/pow_p) * (0.01)**(2/pow_p), k=-1)
            decay_x0 = 1 / math.sqrt(-(pow_p/2) * w.real)

        m = pow_p // 2
        if pow_p % 2 == 1:
            coeffs = torch.tensor([1/math.perm(m-k, m-k) for k in range(m)])
            powers = torch.tensor([2*(m-k) for k in range(m)])
            self.enc_function = lambda x: torch.where(
                x < 0,
                1.,
                torch.exp(-torch.pow((x/decay_x0),2)) * (torch.sum(coeffs.to(x.device) * (x/decay_x0).unsqueeze(-1)**powers.to(x.device), dim=-1) + 1)
            )

        else:
            coeffs = torch.tensor([4**(m-k)/math.perm(2*(m-k), m-k) for k in range(m)])
            powers = torch.tensor([2*(m-k)-1 for k in range(m)])
            self.enc_function = lambda x: torch.where(
                x < 0,
                torch.ones_like(x),
                #torch.erfc(x/decay_x0) + torch.exp(-torch.pow((x/decay_x0),2)) * (torch.sum(coeffs.to(x.device) * (x/decay_x0).unsqueeze(-1)**powers.to(x.device), dim=-1)) / torch.pi**0.5
                #torch.erfc(x/decay_x0) + torch.exp(-torch.pow((x/decay_x0), 2)) * torch.stack([(x/decay_x0).pow(p) for p in powers], dim=-1).mul(torch.tensor(coeffs, dtype=x.dtype, device=x.device)).sum(-1) / torch.pi**0.5
                torch.erfc(x/decay_x0) + torch.exp(-torch.pow(x/decay_x0, 2)) * 2 * x/decay_x0 / torch.pi ** 0.5
            )


class SineEncoding(BaseBasis):
    def __init__(
        self,
        num_basis: int=8,
        initial_T0: Optional[List]=None,
        initial_shift: Optional[List]=None,
        trainable_coeff: bool=True,
        softplus_params: Optional[Dict]={},
        as_gate: bool=False,
        decay_x0: Optional[float] = 1,
        pow_p: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            num_basis=num_basis,
            initial_T0=initial_T0,
            initial_shift=initial_shift,
            trainable_coeff=trainable_coeff,
            softplus_params=softplus_params,
            as_gate=as_gate,
        )
        ks = torch.arange(1, pow_p+1)
        coeffs = torch.tensor([(-1)**(k+1)/torch.pi/k * math.comb(pow_p, pow_p-k)/math.comb(pow_p+k, pow_p) for k in ks])
        self.enc_function = lambda x: torch.where(
            x > decay_x0,
            0,
            torch.where(
                x < 0,
                1.,
                1. - x/decay_x0 + torch.sum(coeffs.to(x.device) * torch.sin(2*ks.to(x.device)*torch.pi*x/decay_x0), dim=1).unsqueeze(-1)
            ),
        )


class CosineEncoding(BaseBasis):
    def __init__(
        self,
        num_basis: int=8,
        initial_T0: Optional[List]=None,
        initial_shift: Optional[List]=None,
        trainable_coeff: bool=True,
        softplus_params: Optional[Dict]={},
        as_gate: bool=False,
        decay_x0: Optional[float] = 1,
        pow_p: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            num_basis=num_basis,
            initial_T0=initial_T0,
            initial_shift=initial_shift,
            trainable_coeff=trainable_coeff,
            softplus_params=softplus_params,
            as_gate=as_gate,
        )
        ks = torch.arange(1, pow_p+1)
        coeffs = torch.tensor([(-1)**(k)/torch.pi/(2*k+1) * math.comb(pow_p, pow_p-k)/math.comb(pow_p+k+1, pow_p+1) for k in ks])
        scale = 1 + torch.sum(coeffs).item()
        self.enc_function = lambda x: torch.where(
            x > decay_x0,
            0,
            torch.where(
                x < 0,
                1.,
                1/2/scale*(torch.cos(torch.pi*x/decay_x0) + torch.sum(coeffs.to(x.device) * torch.cos((2*ks.to(x.device)+1)*torch.pi*x/decay_x0), dim=1).unsqueeze(-1)) + 0.5
            ),
        )


class PolynomialEncoding(BaseBasis):
    def __init__(
        self,
        num_basis: int=8,
        initial_T0: Optional[List]=None,
        initial_shift: Optional[List]=None,
        trainable_coeff: bool=True,
        softplus_params: Optional[Dict]={},
        as_gate: bool=False,
        decay_x0: Optional[float] = 1,
        poly_p: Optional[int] = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            num_basis=num_basis,
            initial_T0=initial_T0,
            initial_shift=initial_shift,
            trainable_coeff=trainable_coeff,
            softplus_params=softplus_params,
            as_gate=as_gate,
        )
        if poly_p < 2:
            warnings.warn('Polynomial with p < 2 will give discontinuous 2nd derivative')

        coeffs = torch.tensor([math.comb(n+poly_p, n) for n in range(1, poly_p+1)])
        powers = torch.arange(len(coeffs)) + 1
        self.enc_function = lambda x: torch.where(
            x > decay_x0,
            0,
            torch.where(
                x < 0,
                1.,
                torch.pow((1-x/decay_x0), poly_p+1) * (torch.sum(coeffs.to(x.device) * x**powers.to(x.device).unsqueeze(-1), dim=-1) + 1.)
            ),
        )


@compile_mode('script')
class TemperatureEncoding(nn.Module):
    def __init__(
        self,
        temperature_key_in: str=KEY.TEMPERATURE,
        temperature_key_out: str=KEY.TEMPERATURE_ENC,
        temperature_encoding_function: str='gaussian',
        temperature_encoding_params: Optional[Dict]={},
    ) -> None:
        super().__init__()
        self.temperature_key_in = temperature_key_in
        self.temperature_key_out = temperature_key_out
        basis_cls = {
            #'sigmoid': SigmoidBasis,
            #'tanh': TanhBasis,
            #'avrami': AvramiBasis,
            'gaussian': GaussianEncoding,
            'sine': SineEncoding,
            'cosine': CosineEncoding,
            'polynomial': PolynomialEncoding,
        }[temperature_encoding_function.lower()]
        self.basis_func = basis_cls(**temperature_encoding_params)


    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        data[self.temperature_key_out] = self.basis_func(data[self.temperature_key_in])
        return data


class BaseTemperatureCoeff(nn.Module):
    def __init__(
        self,
        initial_min_coeff: float,
        original_min_coeff_val: float=0.,
        original_max_coeff_val: float=1.,
        trainable_coeff: bool=True,
    ):
        super().__init__()
        self.min_val = original_min_coeff_val
        self.max_val = original_max_coeff_val
        self.scale = self.max_val - self.min_val
        self.shift = nn.Parameter(torch.tensor(initial_min_coeff), requires_grad=trainable_coeff)
        self.relu = nn.ReLU()
        self.act = None


    def forward(self, tensor):
        current_shift = self.relu(torch.min(self.shift, torch.ones_like(self.shift)))
        return ((1.-current_shift)/self.scale * (self.act(tensor)-self.min_val) + current_shift).squeeze()


class UniformTemperatureCoeff(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


    def forward(self, tensor):
        return 1.


class SigmoidTemperatureCoeff(BaseTemperatureCoeff):
    def __init__(
        self,
        initial_min_coeff: float=0.25,
        trainable_coeff: bool=True,
    ):
        super().__init__(
            initial_min_coeff=initial_min_coeff,
            original_min_coeff_val=0.,
            original_max_coeff_val=1.,
            trainable_coeff=trainable_coeff,
        )

        self.act = nn.Sigmoid()


class TanhTemperatureCoeff(BaseTemperatureCoeff):
    def __init__(
        self,
        initial_min_coeff: float=0.25,
        trainable_coeff: bool=True,
    ):
        super().__init__(
            initial_min_coeff=initial_min_coeff,
            original_min_coeff_val=-1.,
            original_max_coeff_val=1.,
            trainable_coeff=trainable_coeff,
        )

        self.act = nn.Sigmoid()


@compile_mode('script')
class TemperatureGate(nn.Module):
    def __init__(
        self,
        irreps_hidden: Irreps,
        temperature_key_in: str=KEY.TEMPERATURE,
        hidden_key_in: str=KEY.NODE_FEATURE,
        data_energy_key_in: str=KEY.ATOMIC_ENERGY,
        data_energy_key_out: Optional[str]=None,
        data_entropy_key_in: str=KEY.ATOMIC_ENTROPY,
        data_entropy_key_out: Optional[str]=None,
        data_asymptot_key_in: str=KEY.ATOMIC_ASYMPTOT,
        data_asymptot_key_out: Optional[str]=None,
        temperature_gate_function: str='gaussian',
        temperature_gate_params: Optional[Dict]={},
        temperature_coeff_function: Optional[str]='uniform',
        temperature_coeff_params: Optional[Dict]={},
    ) -> None:
        super().__init__()
        self.temperature_key_in = temperature_key_in
        self.hidden_key_in = hidden_key_in
        self.data_energy_key_in = data_energy_key_in
        self.data_energy_key_out = data_energy_key_in if data_energy_key_out is None else data_energy_key_out
        self.data_entropy_key_in = data_entropy_key_in
        self.data_entropy_key_out = data_entropy_key_in if data_entropy_key_out is None else data_entropy_key_out
        self.data_asymptot_key_in = data_asymptot_key_in
        self.data_asymptot_key_out = data_asymptot_key_in if data_asymptot_key_out is None else data_asymptot_key_out
        gate_cls = {
            'gaussian': GaussianEncoding,
            'sine': SineEncoding,
            'cosine': CosineEncoding,
            'polynomial': PolynomialEncoding,
        }[temperature_gate_function.lower()]
        temperature_gate_params.update({'num_basis': 1, 'as_gate': True})
        self.gate = gate_cls(**temperature_gate_params)
        temperature_coeff_cls = {
            'uniform': UniformTemperatureCoeff,
            'sigmoid': SigmoidTemperatureCoeff,
            'tanh': TanhTemperatureCoeff,
        }[temperature_coeff_function.lower()]
        self.temperature_coeff = temperature_coeff_cls(**temperature_coeff_params)

        if temperature_coeff_function.lower() != 'uniform':
            self.head = Linear(
                irreps_hidden, Irreps([(1, (0, 1))])
            )
        else:
            self.head = lambda x: 1.  # null function


    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        embedding = self.head(data[self.hidden_key_in])
        coeff = self.temperature_coeff(embedding)
        if self._is_batch_data:
            gate_input = data[self.temperature_key_in][data[KEY.BATCH]] * coeff
        else:
            gate_input = data[self.temperature_key_in] * coeff
        data['_gate'] = self.gate(gate_input)
        data[self.data_energy_key_out] = data[self.data_energy_key_in] * (1. - data['_gate'])
        data[self.data_entropy_key_out] = data[self.data_entropy_key_in] * data['_gate']
        data[self.data_asymptot_key_out] = data[self.data_asymptot_key_in] * data['_gate']
        return data


@compile_mode('script')
class TemperatureBlock(nn.Module):
    def __init__(
        self,
        data_key_in: str,
        irreps_node: Irreps,
        data_key_out: Optional[str]=None,
        temperature_enc_key: str=KEY.TEMPERATURE_ENC,
        enc_dimension: int=8,
        **linear_kwargs,
    ) -> None:
        super().__init__()
        self.temperature_enc_key = temperature_enc_key
        self.irreps_node = irreps_node
        self.key_in = data_key_in
        self.key_out = data_key_in if data_key_out is None else data_key_out
        self.irreps_in = Irreps(f'{enc_dimension}x0e')
        self.linear_kwargs = linear_kwargs

        self.linear = Linear(
            self.irreps_in, self.irreps_node, **self.linear_kwargs
        )

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        temperature_embedding = self.linear(data[self.temperature_enc_key])
        if self._is_batch_data:
            data[self.key_out] = data[self.key_in] + temperature_embedding[data[KEY.BATCH]]
        else:
            data[self.key_out] = data[self.key_in] + temperature_embedding
        return data


@compile_mode('script')
class TemperatureOutputBlock(nn.Module):
    def __init__(
        self,
        data_energy_key: str=KEY.PRED_TOTAL_ENERGY,
        data_head_entropy_key: str=KEY.PRED_TOTAL_HEAD_ENTROPY,
        data_head_asymptot_key: str=KEY.PRED_TOTAL_HEAD_ASYMPTOT,
        data_free_energy_key: str=KEY.PRED_TOTAL_FREE_ENERGY,
        data_grad_entropy_key: str=KEY.PRED_TOTAL_ENTROPY,
        data_internal_energy_key: str=KEY.PRED_TOTAL_INTERNAL_ENERGY,
        data_heat_capacity_key: str=KEY.PRED_TOTAL_HEAT_CAPACITY,
        data_temperature_key: str=KEY.TEMPERATURE,
        infer_grad_entropy: bool=True,         # requires back in inference, double-back in training
        infer_grad_heat_capacity: bool=False,  # requires double-back in inference, triple-back in training (not supported for flash, oeq)
    ) -> None:
        super().__init__()
        self.key_temperature = data_temperature_key
        self.key_energy = data_energy_key
        self.key_head_entropy = data_head_entropy_key
        self.key_head_asymptot = data_head_asymptot_key
        self.key_free_energy = data_free_energy_key
        self.key_grad_entropy = data_grad_entropy_key
        self.key_internal_energy = data_internal_energy_key
        self.key_heat_capacity = data_heat_capacity_key
        self.infer_grad_entropy = infer_grad_entropy
        self.infer_grad_heat_capacity = infer_grad_heat_capacity


    def get_grad_key(self) -> str:
        return self.key_temperature


    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        safe_temperature = torch.where(data[self.key_temperature] == 0., -1, data[self.key_temperature])
        safe_asymptot = torch.where(safe_temperature < 0., 0., data[self.key_head_asymptot] / safe_temperature)
        data[self.key_free_energy] = data[self.key_energy] - data[self.key_head_entropy] * data[self.key_temperature] - safe_asymptot
        if self.infer_grad_entropy:
            grad = torch.autograd.grad(
                [data[self.key_free_energy].sum()],
                [data[self.key_temperature]],
                create_graph=self.training or self.infer_grad_heat_capacity,
                allow_unused=True,
            )[0]

            if grad is not None:
                data[self.key_grad_entropy] = torch.neg(grad)  # dF/dT = -S
                data[self.key_internal_energy] = data[self.key_free_energy] + data[self.key_grad_entropy] * data[self.key_temperature]

            if self.infer_grad_heat_capacity:
                grad = torch.autograd.grad(
                    [data[self.key_grad_entropy].sum()],
                    [data[self.key_temperature]],
                    create_graph=self.training,
                    allow_unused=True,
                )[0]
                heat_capacity = grad * data[self.key_temperature]  # dS/dT = C_v/T
                data[self.key_heat_capacity] = heat_capacity

        return data

