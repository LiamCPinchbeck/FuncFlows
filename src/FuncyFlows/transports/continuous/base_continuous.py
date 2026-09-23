import torch

from ..abstract_transformation import Transformation


class ContinuousTransformation(Transformation):
    """f = the time-T flow of dv/dt = h(v,t). log_det_term is the integral of Tr Dh (P2 Thm 5).

    with_trace=False skips the log-determinant, which the transport (sampling) and pull_back need; 
    with_trace=True uses velocity_and_trace so the trace shares the field evaluation instead of 
    repeating it.
    """

    def __init__(self, base_measure, vector_field, total_time=1.0, num_steps=20, method="rk4"):
        super().__init__(base_measure)
        self.vector_field, self.total_time, self.num_steps, self.method = vector_field, total_time, num_steps, method

    def _stage(self, coeffs, time_value, context, with_trace):
        if with_trace:
            return self.vector_field.velocity_and_trace(coeffs, time_value, context)
        return self.vector_field.velocity(coeffs, time_value, context), None

    def _integrate(self, coeffs, forward=True, context=None, with_trace=True):
        step = (self.total_time / self.num_steps) * (1 if forward else -1)
        time_value = 0.0 if forward else self.total_time
        trace_total = torch.zeros_like(coeffs[..., 0]) if with_trace else None
        for _ in range(self.num_steps):
            velocity_1, trace_1 = self._stage(coeffs, time_value, context, with_trace)
            if self.method == "euler":
                velocity, trace = velocity_1, trace_1
            else:
                velocity_2, trace_2 = self._stage(coeffs + 0.5 * step * velocity_1,
                                                  time_value + 0.5 * step, context, with_trace)
                velocity_3, trace_3 = self._stage(coeffs + 0.5 * step * velocity_2,
                                                  time_value + 0.5 * step, context, with_trace)
                velocity_4, trace_4 = self._stage(coeffs + step * velocity_3,
                                                  time_value + step, context, with_trace)
                velocity = (velocity_1 + 2 * velocity_2 + 2 * velocity_3 + velocity_4) / 6
                trace = ((trace_1 + 2 * trace_2 + 2 * trace_3 + trace_4) / 6) if with_trace else None
            coeffs = coeffs + step * velocity
            if with_trace:
                trace_total = trace_total + step * trace
            time_value = time_value + step
        return coeffs, trace_total

    def _map(self, coeffs, context=None):
        return self._integrate(coeffs, forward=True, context=context)

    def pull_back(self, coeffs_out, context=None):
        return self._integrate(coeffs_out, forward=False, context=context, with_trace=False)[0]

    def log_rn_at(self, coeffs_out, context=None):
        """One backward solve instead of pull_back + push_forward (P2 Supp Thm 2.1)."""
        coeffs, trace_total = self._integrate(coeffs_out, forward=False, context=context)
        return self.base_measure.log_density_diff(coeffs, coeffs_out) + trace_total

    def transport(self, coeffs, context=None):
        """Integrate the field only. No trace, no log-determinant — for sampling."""
        return self._integrate(coeffs, forward=True, context=context, with_trace=False)[0]



