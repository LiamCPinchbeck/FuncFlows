import torch


class NegativeLogL:
    """P2 Alg. 1 (direct data): -mean_i log d(f#mu0)/dmu0 (u_i) over a minibatch of dataset coefficients."""

    def __init__(self, transformation, dataset_coeffs, batch_size=64, context=None):
        self.transformation, self.dataset_coeffs, self.batch_size = transformation, dataset_coeffs, batch_size
        self.context = context

    def __call__(self):
        batch_indices = torch.randint(len(self.dataset_coeffs), (self.batch_size,),
                                      device=self.dataset_coeffs.device)      # no host round trip
        context = None if self.context is None else self.context[batch_indices]
        return -self.transformation.log_rn_at(self.dataset_coeffs[batch_indices], context).mean()
