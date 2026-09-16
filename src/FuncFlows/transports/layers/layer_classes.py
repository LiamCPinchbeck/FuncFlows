import math

import torch

from ..abstract_transformation import Transformation


class DiscreteLayer(Transformation):
    """One invertible layer acting on the leading num_modes coefficients; the tail passes through."""

    def __init__(self, base_measure, num_modes):
        super().__init__(base_measure)
        if num_modes > base_measure.num_functions:
            raise ValueError(f"num_modes {num_modes} exceeds the measure's {base_measure.num_functions}")
        self.num_modes = num_modes


class HouseholderLayer(DiscreteLayer):
    """f(u) = u - 0.5 v (v.u + b), unit v. P1 §2.3.1; Jacobian I - 0.5 v v^T has det 0.5 exactly."""

    def __init__(self, base_measure, num_modes):
        super().__init__(base_measure, num_modes)
        self.direction_raw = torch.nn.Parameter(0.1 * torch.randn(num_modes, dtype=base_measure.dtype))
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=base_measure.dtype))

    def _map(self, coeffs, context=None):
        direction = self.direction_raw / self.direction_raw.norm()          # P1 Thm 2.4 normalisation
        head = coeffs[..., :self.num_modes]
        head = head - 0.5 * direction * (head @ direction + self.bias)[..., None]
        coeffs_out = torch.cat([head, coeffs[..., self.num_modes:]], dim=-1)
        log_det = torch.full_like(coeffs[..., 0], math.log(0.5))         # inherits device
        return coeffs_out, log_det

    def pull_back(self, coeffs_out, context=None):
        direction = self.direction_raw / self.direction_raw.norm()
        head = coeffs_out[..., :self.num_modes]
        head = head + direction * (head @ direction + self.bias)[..., None]
        return torch.cat([head, coeffs_out[..., self.num_modes:]], dim=-1)


class SylvesterLayer(DiscreteLayer):
    """f(u) = u + A tanh(B u + b) with A, B upper-triangular (P1 §2.3.2, van den Berg et al. 2018).

    diag(B) = 1 and diag(A) = tanh(.) in (-1, 1), so by Sylvester's identity
    log|det Df| = sum_j log(1 + (1 - h_j^2) diag(A)_j) is exact and O(num_modes).
    """

    def __init__(self, base_measure, num_modes, init_scale=0.1):
        super().__init__(base_measure, num_modes)
        dtype = base_measure.dtype
        self.upper_before = torch.nn.Parameter(init_scale * torch.randn(num_modes, num_modes, dtype=dtype))   # R_B
        self.upper_after = torch.nn.Parameter(init_scale * torch.randn(num_modes, num_modes, dtype=dtype))    # R_A
        self.diag_after_raw = torch.nn.Parameter(init_scale * torch.randn(num_modes, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(num_modes, dtype=dtype))
        self.register_buffer("identity", torch.eye(num_modes, dtype=dtype))   # a buffer, so .to() moves it

    def matrices(self):
        before = torch.triu(self.upper_before, diagonal=1) + self.identity                  # unit diagonal
        diag_after = torch.tanh(self.diag_after_raw)
        after = torch.triu(self.upper_after, diagonal=1) + torch.diag(diag_after)
        return before, after, diag_after

    def _map(self, coeffs, context=None):
        before, after, diag_after = self.matrices()
        head, tail = coeffs[..., :self.num_modes], coeffs[..., self.num_modes:]
        hidden = torch.tanh(head @ before.T + self.bias)
        head = head + hidden @ after.T
        log_det = torch.log1p((1 - hidden ** 2) * diag_after).sum(-1)
        return torch.cat([head, tail], dim=-1), log_det

    def pull_back(self, coeffs_out, context=None, num_iterations=30):
        """Fixed-point inverse. Converges when the layer is a contraction, which the small
        init_scale and |diag(A)| < 1 encourage but do not guarantee. No residual check inside
        the loop: on a GPU that would be a sync per iteration. Check once outside if in doubt:
        (layer._map(layer.pull_back(x))[0] - x).abs().max()."""
        before, after, _ = self.matrices()
        head_out, tail = coeffs_out[..., :self.num_modes], coeffs_out[..., self.num_modes:]
        head = head_out
        for _ in range(num_iterations):
            head = head_out - torch.tanh(head @ before.T + self.bias) @ after.T
        return torch.cat([head, tail], dim=-1)
