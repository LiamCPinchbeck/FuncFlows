import torch


from FuncFlows.base_measures import ReferenceMeasure


class Transformation(torch.nn.Module):

    def __init__(self, base_measure:ReferenceMeasure):
        super().__init__()
        self.base_measure = base_measure


    def _map(self, coeffs):
        """coeffs [..., num_functions] -> (coeffs_out [..., num_functions], log_det_term [...])
        log_det_term = log|det Df| for a discrete map, integral of Tr Dh for an ODE."""
        raise NotImplementedError

    def pull_back(self, coeffs_out):
        """Inverse map. Optional; needed only by NLL / learned-prior uses."""
        raise NotImplementedError

    def push_forward(self, coeffs):
        """-> (coeffs_out, log_rn_weight) with log_rn_weight = log d(f#mu)/d(mu) at coeffs_out."""
        coeffs_out, log_det_term = self._map(coeffs)
        return coeffs_out, self.base_measure.log_density_diff(coeffs, coeffs_out) - log_det_term

    def log_rn_at(self, coeffs_out):
        return self.push_forward(self.pull_back(coeffs_out))[1]