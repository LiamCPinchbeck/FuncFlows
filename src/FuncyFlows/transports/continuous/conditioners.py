import torch

from .vector_fields import TimeCosines

# base class
class Conditioner(torch.nn.Module):
    """(time, context) -> (output_matrix [num_functions, num_terms],
                           input_matrix  [num_terms, num_functions],
                           biases        [num_terms] or [batch, num_terms])"""


    def __init__(self, num_functions, num_terms):
        super().__init__()
        self.num_functions, self.num_terms = num_functions, num_terms


    def forward(self, time_value, context=None):
        raise NotImplementedError



# Just a class for conditioners in the velocity fields script
    # that handles unknown parameters that are functions of just time
class TimeBasisConditioner(Conditioner):
    """A(t) = sum_m cos(m pi t / T) A_m, same for B and the biases. Genuine time dependence
    for num_time_modes * 2 * L * r parameters, instead of a net emitting that many per step.

    The contraction over m is a tensordot, which never materialises the [T, r, L] product the
    broadcast-and-sum form did — that intermediate was rebuilt four times per RK4 step.
    """

    def __init__(self, num_functions, num_terms, num_time_modes=4, total_time=1.0,
                 init_scale=0.05, dtype=torch.float64):
        super().__init__(num_functions, num_terms)
        scale = init_scale / (num_functions * num_time_modes) ** 0.5
        self.total_time, self.num_time_modes = total_time, num_time_modes

        self.output_stack = torch.nn.Parameter(scale * torch.randn(num_time_modes, num_functions, num_terms, dtype=dtype))
        self.input_stack = torch.nn.Parameter(scale * torch.randn(num_time_modes, num_terms, num_functions, dtype=dtype))
        self.bias_stack = torch.nn.Parameter(torch.zeros(num_time_modes, num_terms, dtype=dtype))

        self.cosines = TimeCosines(num_time_modes, total_time, dtype)     # cached per scalar time


    def time_weights(self, time_value):
        return self.cosines(time_value)

    def forward(self, time_value, context=None):
        weights = self.time_weights(time_value)
        return (torch.tensordot(weights, self.output_stack, dims=1),
                torch.tensordot(weights, self.input_stack, dims=1),
                weights @ self.bias_stack)



# Just a class for conditioners in the velocity fields script
    # that handles unknown parameters that are functions of time AND data embeddings
class DataConditioner(TimeBasisConditioner):
    """Same field, but the data shifts the bias: b(t, c) = sum_m w_m(t) [bias_m + c @ context_m.T].

    The context does not depend on v, so dh/dv is unchanged and MatrixField.trace still holds.
    """

    def __init__(self, num_functions, num_terms, context_dim, num_time_modes=4,
                 total_time=1.0, init_scale=0.05, dtype=torch.float64):
        super().__init__(num_functions, num_terms, num_time_modes, total_time, init_scale, dtype)

        self.context_stack = torch.nn.Parameter(
            init_scale / (context_dim * num_time_modes) ** 0.5
            * torch.randn(num_time_modes, num_terms, context_dim, dtype=dtype))



    def forward(self, time_value, context=None):
        output_matrix, input_matrix, biases = super().forward(time_value)


        if context is not None:
            weights = self.time_weights(time_value)
            biases = biases + context @ torch.tensordot(weights, self.context_stack, dims=1).T


        return output_matrix, input_matrix, biases
