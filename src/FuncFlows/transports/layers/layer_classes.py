
import torch
from ..abstract_transformation import Transformation
import math


# very much a skeleton class
class DiscreteLayer(Transformation):

    def __init__(self, base_measure, num_modes):
        super().__init__(base_measure)
        self.num_modes = num_modes


class HouseholderLayer(DiscreteLayer):

    def __init__(self, base_measure, num_modes):
        super().__init__(base_measure, num_modes)

        # randomly initializing vn
        self.direction_raw = torch.nn.Parameter(0.1 * torch.randn(num_modes, dtype=base_measure.dtype))

        # randomly initializing bn
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=base_measure.dtype))

    def _map(self, coeffs):

        # gotta normalize according to Theorem 2.4 of https://arxiv.org/pdf/2411.13277
            # i.e. equation after 2.7 if you're Ctrl+F-ing
        direction = self.direction_raw / self.direction_raw.norm()

        head = coeffs[..., :self.num_modes]

        # first equation in section 2.3.1 of 2411.13277. They use 0.5, I use 0.5
        head = head - 0.5 * direction * (head @ direction + self.bias)[..., None]

        coeffs_out = torch.cat([head, coeffs[..., self.num_modes:]], dim=-1)


        return coeffs_out, torch.full(coeffs.shape[:-1], math.log(0.5), dtype=coeffs.dtype)