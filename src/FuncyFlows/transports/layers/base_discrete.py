import torch

from ..abstract_transformation import Transformation
from .layer_classes import DiscreteLayer
from FuncyFlows.base_measures import ReferenceMeasure


class DiscreteTransformation(Transformation):
    """A stack of DiscreteLayers. Pass either a list of layers, or a layer class + count + its kwargs."""

    def __init__(self, base_measure: ReferenceMeasure,
                 layers=None, layer_class: type[DiscreteLayer] = None,
                 num_layers: int = 1, **layer_kwargs):
        super().__init__(base_measure)
        if layers is None:
            layers = [layer_class(base_measure, **layer_kwargs) for _ in range(num_layers)]
        self.layers = torch.nn.ModuleList(layers)

    def _map(self, coeffs, context=None):
        log_det_term = torch.zeros_like(coeffs[..., 0])           # inherits device and dtype
        for layer in self.layers:
            coeffs, layer_log_det = layer._map(coeffs, context)
            log_det_term = log_det_term + layer_log_det
        return coeffs, log_det_term

    def pull_back(self, coeffs_out, context=None):
        for layer in reversed(self.layers):
            coeffs_out = layer.pull_back(coeffs_out, context)
        return coeffs_out
