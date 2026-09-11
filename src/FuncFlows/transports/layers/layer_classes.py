
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

    def pull_back(self, coeffs_out):
        direction = self.direction_raw / self.direction_raw.norm()
        head = coeffs_out[..., :self.num_modes]
        head = head + direction * (head @ direction + self.bias)[..., None]
        return torch.cat([head, coeffs_out[..., self.num_modes:]], dim=-1)



class SylvesterLayer(DiscreteLayer):

    def __init__(self, base_measure, num_modes, init_scale=0.1):
        super().__init__(base_measure, num_modes)
        dtype = base_measure.dtype
        self.blo_inner = torch.nn.Parameter(init_scale * torch.randn(num_modes, num_modes, dtype=dtype))   # R_B
        self.alo_outer = torch.nn.Parameter(init_scale * torch.randn(num_modes, num_modes, dtype=dtype))    # R_A
        
        self.diag_after_raw = torch.nn.Parameter(init_scale * torch.randn(num_modes, dtype=dtype))

        self.bias = torch.nn.Parameter(torch.zeros(num_modes, dtype=dtype))
        self.eye = torch.eye(num_modes, dtype=self.blo_inner.dtype)
        
    def matrices(self):
        before = torch.triu(self.blo_inner, diagonal=1) + self.eye                                   # R_B
        diag_after = torch.tanh(self.diag_after_raw)                                               # in (-1, 1)
        after = torch.triu(self.alo_outer, diagonal=1) + torch.diag(diag_after)                  # R_A

        return before, after, diag_after

    def _map(self, coeffs):
        before, after, diag_after = self.matrices()

        head, tail = coeffs[..., :self.num_modes], coeffs[..., self.num_modes:]

        hidden = torch.tanh(head @ before.T + self.bias)
        head = head + hidden @ after.T

        log_det = torch.log1p((1 - hidden ** 2) * diag_after).sum(-1)

        return torch.cat([head, tail], dim=-1), log_det

    def pull_back(self, coeffs_out, num_iterations=30):
        before, after, _ = self.matrices()

        head_out, tail = coeffs_out[..., :self.num_modes], coeffs_out[..., self.num_modes:]

        head = head_out
        for _ in range(num_iterations):
            head = head_out - torch.tanh(head @ before.T + self.bias) @ after.T

        return torch.cat([head, tail], dim=-1)