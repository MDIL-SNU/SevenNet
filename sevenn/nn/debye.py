from ase.units import kB

import torch
import torch.nn as nn

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType, INTEGRAL_FTDT, INTEGRAL_FTDT_TAYLOR_DIV_X3, DEBYE_F, DEBYE_X, DEBYE_FT


def zpe(Td):
    return 9/8*kB*Td


def debye_integral(x):
    # return ((1/x)**3) * (integral 0 to x dt 3t^3/(e^t-1))
    # at too low temperature, integral converge to pi**4/5/x**3
    # also, F ~ ZPE + kB*T*(3L(T;Td) - debye_integral) -> at low T, numerical accuracy of debye_integral less matters
    integral_ftdt = INTEGRAL_FTDT.to(x.device)
    debye_x = DEBYE_X.to(x.device)
    debye_ft = DEBYE_FT.to(x.device)

    fx = torch.where(x > 0, DEBYE_F(x), torch.zeros_like(x))
    is_sampled = debye_x.unsqueeze(0) < x.unsqueeze(-1)
    last_idx = torch.sum(is_sampled, dim = 1) - 1
    x_last = debye_x[last_idx]
    fx_last = debye_ft[last_idx]
    integral = integral_ftdt[last_idx] + (fx + fx_last)/2 * (x - x_last)
    integral_div_x3 = torch.where(last_idx == -1, INTEGRAL_FTDT_TAYLOR_DIV_X3(x), integral / x**3)

    return torch.where(x < 0, 0., integral_div_x3)


def debye_log(x):
    # return log(1-exp(-x))
    val = torch.log(1-torch.exp(-x))
    return torch.where(x < 0, 0., val)


class DebyeBlock(nn.Module):
    def __init__(
        self,
        debye_temperature: float,
        trainable_coeff: bool = True,
        infer_heat_capacity: bool = False,
        infer_debye: bool = True,
    ) -> None:
        super().__init__()
        self.debye_temperature = torch.tensor(debye_temperature, dtype=torch.float32)
        self.softplus = nn.Softplus()
        if trainable_coeff:
            self.debye_temperature = nn.Parameter(self.debye_temperature)
        self.infer_heat_capacity = infer_heat_capacity
        self.infer_debye = infer_debye

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        if not self.infer_debye:
            return data

        size = int(data[KEY.BATCH].max()) + 1 if self._is_batch_data else 1
        data[KEY.DEBYE_ZPE] = zpe(self.softplus(self.debye_temperature)).repeat(size)
        safe_temperature = torch.where(data[KEY.TEMPERATURE] == 0, -1, data[KEY.TEMPERATURE])
        x = torch.where(data[KEY.TEMPERATURE] == 0, -1, self.softplus(self.debye_temperature) / safe_temperature)
        debye_integral_value = debye_integral(x)
        debye_log_value = debye_log(x)
        data[KEY.DEBYE_FREE_ENERGY] = data[KEY.DEBYE_ZPE] + kB * data[KEY.TEMPERATURE] * (3 * debye_log_value - debye_integral_value)
        data[KEY.DEBYE_INTERNAL_ENERGY] = data[KEY.DEBYE_ZPE] + 3 * kB * data[KEY.TEMPERATURE] * debye_integral_value
        data[KEY.DEBYE_ENTROPY] = kB * (4 * debye_integral_value - 3 * debye_log_value)
        data[KEY.DEBYE_ASYMPTOT] = - (kB * self.softplus(self.debye_temperature) ** 2 * 3 / 40).repeat(size)

        if self.infer_heat_capacity:
            heat_capacity = 3 * kB * (4 * debye_integral_value - 3 * x / (torch.exp(x) - 1))
            data[KEY.DEBYE_HEAT_CAPACITY] = torch.where(x < 0, 0, heat_capacity)

        return data


if __name__=='__main__':
    import math
    import numpy as np
    #from sevenn._const import INTEGRAL_FTDT, DEBYE_F, DEBYE_X, DEBYE_FT
    #print(INTEGRAL_FTDT)
    #inp = torch.tensor([20/10000*8, 20/10000*16, 20/10000*32, 64/10000*32])
    import sys
    import matplotlib.pyplot as plt

    debye = 1000
    temperature = torch.tensor([0, 100, 200, 300])
    x = debye / temperature
    print(debye_integral(x))
    print(debye_log(x))

    """
    deb_x = torch.linspace(0, 30, 10000)
    t = torch.linspace(0, 30, 10000)
    ft = DEBYE_F(t)
    x = deb_x.unsqueeze(1)
    t = t.unsqueeze(0)
    ft = DEBYE_FT.unsqueeze(0)
    mask = t <= x
    mask_ft = ft * mask
    last_ft = ft * (t==x)
    integral_ftdt = torch.trapz(mask_ft, t, dim=1) - torch.sum(last_ft, dim=1) * (30)/10000/2
    integral_ftdt_div_x3 = integral_ftdt / deb_x**3  # integral from zero

    #inp = torch.tensor([float(sys.argv[1])])
    inp = torch.linspace(0, 30, 10000)
    custom = debye_integral(inp)
    taylor = 1-3*inp/8+inp**2/20-inp**4/7/240
    asymp = torch.pi**4/5/inp**3

    plt.figure(figsize=(12/2.54, 10/2.54))
    plt.plot(inp, custom, color='b')
    plt.plot(inp, integral_ftdt_div_x3, color='r')
    plt.plot(inp, asymp, color='gray', linestyle='dashed')
    plt.plot(inp, taylor, color='gray', linestyle='dashed')
    plt.scatter([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1., 2., 3., 4., 5., 10., 15., 20., 25., 30.], [0.981375, 0.963, 0.944875, 0.926999, 0.909373, 0.891995, 0.874866, 0.857985, 0.841351, 0.824963, 0.674416, 0.441128, 0.28358, 0.1817369, 0.117597, 0.0192957, 0.0057712, 0.00243522, 0.00124684, 0.0007215488], c='b', edgecolor='k', s=15, zorder=1000)
    plt.xlim(0, 30)
    plt.ylim(0, 1.2)
    plt.savefig('plot_debye.png')

    plt.figure(figsize=(12/2.54, 10/2.54))
    plt.plot(inp, custom, color='b')
    plt.plot(inp, integral_ftdt_div_x3, color='r')
    plt.plot(inp, asymp, color='gray', linestyle='dashed')
    plt.plot(inp, taylor, color='gray', linestyle='dashed')
    plt.scatter([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1., 2., 3., 4., 5., 10., 15., 20., 25., 30.], [0.981375, 0.963, 0.944875, 0.926999, 0.909373, 0.891995, 0.874866, 0.857985, 0.841351, 0.824963, 0.674416, 0.441128, 0.28358, 0.1817369, 0.117597, 0.0192957, 0.0057712, 0.00243522, 0.00124684, 0.0007215488], c='b', edgecolor='k', s=15, zorder=1000)
    plt.xlim(0, 5)
    plt.ylim(0.1, 1.1)
    plt.savefig('plot_debye_small.png')

    plt.figure(figsize=(12/2.54, 10/2.54))
    plt.plot(inp, custom, color='b')
    plt.plot(inp, integral_ftdt_div_x3, color='r')
    plt.plot(inp, asymp, color='gray', linestyle='dashed')
    plt.plot(inp, taylor, color='gray', linestyle='dashed')
    plt.scatter([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1., 2., 3., 4., 5., 10., 15., 20., 25., 30.], [0.981375, 0.963, 0.944875, 0.926999, 0.909373, 0.891995, 0.874866, 0.857985, 0.841351, 0.824963, 0.674416, 0.441128, 0.28358, 0.1817369, 0.117597, 0.0192957, 0.0057712, 0.00243522, 0.00124684, 0.0007215488], c='b', edgecolor='k', s=15, zorder=1000)
    plt.xlim(5, 30)
    plt.ylim(0, 0.12)
    plt.savefig('plot_debye_large.png')
    """
