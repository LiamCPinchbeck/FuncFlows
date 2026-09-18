import torch




def combine_context(stack, context, weights):
    """sum_t w_t (stack_t @ context) without forming the [batch, out, context] tensor.    """

    partial = torch.einsum("toc,bc->bto", stack, context)

    return (weights.reshape(-1, stack.shape[0], 1) * partial).sum(1)


################################################################################################################
################################################################################################################
################################################################################################################


class VectorField(torch.nn.Module):
    """dv/dt = velocity(coeffs, time). Must also expose the trace of its Jacobian.   
    
    'v' is used to refer to the function space coefficient vectors 
    
    """

    supports_batched_time = False           # can `time_value` carry one entry per sample?
    trace_samples = 1                       # Hutchinson probes, where the trace is estimated


    def velocity(self, coeffs, time_value, context=None):
        raise NotImplementedError

    def trace(self, coeffs, time_value, context=None):
        return self.estimated_trace(coeffs, time_value, context)

    def estimated_trace(self, coeffs, time_value, context=None):
        return self.estimated_velocity_and_trace(coeffs, time_value, context)[1]

    def estimated_velocity_and_trace(self, coeffs, time_value, context=None):
        """Hutchinson (FFJORD): Tr J = E[probe . J probe] for any probe with unit covariance."""
        needs_graph = torch.is_grad_enabled()

        with torch.enable_grad():
            state = coeffs if coeffs.requires_grad else coeffs.detach().requires_grad_(True)
            velocity = self.velocity(state, time_value, context)
            total = torch.zeros_like(coeffs[..., 0])

            for index in range(self.trace_samples):

                probe = 2.0 * torch.randint(0, 2, coeffs.shape, device=coeffs.device).to(coeffs) - 1.0
                slope, = torch.autograd.grad(velocity, state, grad_outputs=probe,
                                             create_graph=needs_graph,
                                             retain_graph=index + 1 < self.trace_samples or needs_graph)
                total = total + (probe * slope).sum(-1)
        total = total / self.trace_samples

        return (velocity, total) if needs_graph else (velocity.detach(), total.detach())


    def velocity_and_trace(self, coeffs, time_value, context=None):
        """Both at once. Subclasses that share work between them should override this."""

        return self.velocity(coeffs, time_value, context), self.trace(coeffs, time_value, context)






################################################################################################################
################################################################################################################
################################################################################################################

class TimeCosines(torch.nn.Module):
    """cosine_time_weights with the order vector held as a buffer instead of rebuilt per call.    """

    def __init__(self, num_time_modes, total_time, dtype, time_pair=False):
        super().__init__()
        self.total_time, self.time_pair = total_time, time_pair
        self.register_buffer("orders", torch.arange(num_time_modes, dtype=dtype), persistent=False)

        per_axis = int(num_time_modes ** 0.5 - 1e-9) + 1                    # ceil, so per_axis^2 >= n
        pairs = torch.cartesian_prod(torch.arange(per_axis, dtype=dtype),
                                     torch.arange(per_axis, dtype=dtype))
        
        chosen = pairs[torch.argsort(pairs.sum(1), stable=True)[:num_time_modes]]

        self.register_buffer("end_orders", chosen[:, 0].contiguous(), persistent=False)
        self.register_buffer("span_orders", chosen[:, 1].contiguous(), persistent=False)

        self.cached = {}

    def as_times(self, time_value):
        """-> a tensor, or a (start, end) tuple of tensors, on this module's device and dtype."""

        def cast(value):
            return torch.as_tensor(value, dtype=self.orders.dtype, device=self.orders.device)
        
        if isinstance(time_value, tuple):
            return cast(time_value[0]), cast(time_value[1])
        
        return (cast(time_value), cast(time_value)) if self.time_pair else cast(time_value)


    def weights(self, times):
        if not self.time_pair:
            return torch.cos(torch.pi * self.orders * (times.unsqueeze(-1) / self.total_time))
        start_time, end_time = times
        span = (end_time - start_time).unsqueeze(-1) / self.total_time
        return (torch.cos(torch.pi * self.end_orders * (end_time.unsqueeze(-1) / self.total_time))
                * torch.cos(torch.pi * self.span_orders * span))

    def forward(self, time_value):
        compiling = getattr(torch.compiler, "is_compiling", lambda: False)()
        tensor_time = (torch.is_tensor(time_value)
                       or (isinstance(time_value, tuple) and torch.is_tensor(time_value[0])))

        
        if tensor_time or compiling:
            return self.weights(self.as_times(time_value))
        
        label = (time_value if isinstance(time_value, tuple) else (time_value,),
                 self.orders.device.type, self.orders.dtype)
        
        if label not in self.cached:
            if len(self.cached) > 512:
                self.cached.clear()
            self.cached[label] = self.weights(self.as_times(time_value))
        
        return self.cached[label]




################################################################################################################
################################################################################################################
################################################################################################################


class LinearField(VectorField):
    """h(v,t,c) = gain_k(t) \odot v + s \odot W(t) c    """

    def __init__(self, num_functions, context_dim=None, num_time_modes=4, total_time=1.0, init_scale=0.0,
                 time_pair=False, dtype=torch.float64, mode_scale=None, rank=None, ):
        super().__init__()

        self.num_time_modes, self.total_time, self.dtype = num_time_modes, total_time, dtype

        self.gain_stack = torch.nn.Parameter(
                                            init_scale * torch.randn(num_time_modes, num_functions, dtype=dtype)
                                            )
        
        self.cosines = TimeCosines(num_time_modes, total_time, dtype, time_pair)

        self.mode_scale, self.rank = mode_scale, rank

        self.has_drift = bool(context_dim)


        if self.has_drift:
            if rank:
                self.left = torch.nn.Parameter(
                    init_scale / rank ** 0.5 * torch.randn(num_functions, rank, dtype=dtype))
                self.right = torch.nn.Parameter(
                    init_scale / (context_dim * num_time_modes) ** 0.5
                    * torch.randn(num_time_modes, rank, context_dim, dtype=dtype))
            else:
                self.drift_stack = torch.nn.Parameter(
                    init_scale / (context_dim * num_time_modes) ** 0.5
                    * torch.randn(num_time_modes, num_functions, context_dim, dtype=dtype))


    def _gains(self, time_value):
        return self.cosines(time_value) @ self.gain_stack

    def velocity(self, coeffs, time_value, context=None):

        if self.has_drift:
            if context is None:
                return self._gains(time_value) * coeffs
            
            weights = self.cosines(time_value)

            if self.rank:
                update = combine_context(self.right, context, weights) @ self.left.T
            else:
                update = combine_context(self.drift_stack, context, weights)
            drift_update = update if self.mode_scale is None else update * self.mode_scale

            return self._gains(time_value) * coeffs + drift_update

        else:

            return self._gains(time_value) * coeffs


    def trace(self, coeffs, time_value, context=None):
        return self._gains(time_value).sum(-1).expand(coeffs.shape[:-1])


    def velocity_and_trace(self, coeffs, time_value, context=None):

        gains = self._gains(time_value) 

        return self.velocity(coeffs, time_value, context=context), gains.sum(-1).expand(coeffs.shape[:-1])






################################################################################################################
################################################################################################################
################################################################################################################


class MatrixField(VectorField):
    """Based off of arXiv:1803.05649 Sylvester flows

    """

    supports_batched_time = False

    def __init__(self, conditioner, activation="tanh", mode_scale=None, num_active=None):
        super().__init__()
        self.conditioner = conditioner
        self.activation = activation
        self.mode_scale = mode_scale
        self.num_active = num_active or conditioner.num_functions



    def _parts(self, coeffs, time_value, context=None):

        output_matrix, input_matrix, biases = self.conditioner(time_value, context)
        active = coeffs[..., :self.num_active]

        if self.mode_scale is not None:
            active = active / self.mode_scale[:self.num_active]
        pre_activation = active @ input_matrix.T + biases

        if self.activation == "identity":
            return output_matrix, input_matrix, pre_activation, torch.ones_like(pre_activation)
        hidden = torch.tanh(pre_activation)

        return output_matrix, input_matrix, hidden, 1 - hidden ** 2



    def _update(self, coeffs, output_matrix, hidden):
        update = hidden @ output_matrix.T
        if self.mode_scale is not None:
            update = update * self.mode_scale[:self.num_active]

        return torch.nn.functional.pad(update, (0, coeffs.shape[-1] - self.num_active))


    def velocity(self, coeffs, time_value, context=None):
        output_matrix, _, hidden, _ = self._parts(coeffs, time_value, context)

        return self._update(coeffs, output_matrix, hidden)


    def trace(self, coeffs, time_value, context=None):
        output_matrix, input_matrix, _, derivative = self._parts(coeffs, time_value, context)

        return (derivative * (input_matrix * output_matrix.T).sum(1)).sum(-1)


    def velocity_and_trace(self, coeffs, time_value, context=None):
        output_matrix, input_matrix, hidden, derivative = self._parts(coeffs, time_value, context)

        return (self._update(coeffs, output_matrix, hidden),
                (derivative * (input_matrix * output_matrix.T).sum(1)).sum(-1))






################################################################################################################
################################################################################################################
################################################################################################################


class SumField(VectorField):
    """h = sum of the given fields. Velocities add; traces add."""

    def __init__(self, *fields):
        super().__init__()
        self.fields = torch.nn.ModuleList(fields)

    @property
    def supports_batched_time(self):
        return all(field.supports_batched_time for field in self.fields)

    def velocity(self, coeffs, time_value, context=None):
        return sum(field.velocity(coeffs, time_value, context) for field in self.fields)

    def trace(self, coeffs, time_value, context=None):
        return sum(field.trace(coeffs, time_value, context) for field in self.fields)

    def velocity_and_trace(self, coeffs, time_value, context=None):
        parts = [field.velocity_and_trace(coeffs, time_value, context) for field in self.fields]
        return sum(part[0] for part in parts), sum(part[1] for part in parts)

