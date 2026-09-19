import torch


class FlowMatching:
    """Regress the field onto the straight-path velocity from the reference measure to the data.

    coupling: 'independent' pairs each base draw with a random target (Lipman et al. 2023;
    FFM). 'optimal' re-pairs the minibatch by exact optimal transport in the training metric
    (Tong et al. 2023 OT-CFM; in function space FOT-CFM, Li et al. 2026): straighter flows,
    lower-variance targets, and fewer integration steps at sampling time.
    """

    def __init__(self, transformation, dataset_coeffs, batch_size, weights=None, coupling="independent"):
        self.field, self.measure = transformation.vector_field, transformation.base_measure
        self.total_time = transformation.total_time
        self.dataset_coeffs, self.batch_size, self.weights = dataset_coeffs, batch_size, weights
        self.coupling = coupling

    def __call__(self):
        if callable(self.dataset_coeffs):
            target_coeffs = self.dataset_coeffs(self.batch_size)
        else:
            index = torch.randint(len(self.dataset_coeffs), (self.batch_size,),
                                  device=self.dataset_coeffs.device)
            target_coeffs = self.dataset_coeffs[index]
        start_coeffs = self.measure.sample(self.batch_size)
        return straight_path_loss(self.field, start_coeffs, target_coeffs, self.total_time, None,
                                  self.weights, self.coupling)


class PairedFlowMatching:
    """Reflow (Liu et al. 2022): regress on the model's OWN (base, endpoint) pairs.

    A trained flow defines a coupling whose marginals are the base measure and the model, and
    whose paths cross. Refitting a field to those exact pairs -- never redrawing the base --
    removes the crossings without changing either marginal, so the flow gets straighter and the
    same sample needs fewer integration steps. Repeatable; each round compounds, and each round
    also compounds whatever error the previous model had, so two or three is the usual limit.
    """

    def __init__(self, transformation, base_coeffs, target_coeffs, batch_size, weights=None):
        self.field, self.total_time = transformation.vector_field, transformation.total_time
        self.base_coeffs, self.target_coeffs = base_coeffs, target_coeffs
        self.batch_size, self.weights = batch_size, weights

    def __call__(self):
        index = torch.randint(len(self.base_coeffs), (self.batch_size,),
                              device=self.base_coeffs.device)
        return straight_path_loss(self.field, self.base_coeffs[index], self.target_coeffs[index],
                                  self.total_time, None, self.weights, "independent")


def reflow_pairs(flow, count, batch_size=512):
    """Base draws and where the flow sends them: the dataset PairedFlowMatching refits."""
    base, target = [], []
    with torch.no_grad():
        for done in range(0, count, batch_size):
            draws = flow.base_measure.sample(min(batch_size, count - done))
            base.append(draws)
            target.append(flow.transport(draws))
    return torch.cat(base), torch.cat(target)


class ConditionalFlowMatching:
    """Same, but base and target come from a simulator and the field sees the context.

    coupling='optimal' pairs base draws with targets ACROSS contexts, which biases which base
    draw reaches each context, so the conditional flow no longer starts from the base measure.
    context_weight is the fix (Chemseddine et al. 2024; Kerrigan et al. 2024): the transport cost
    carries the squared context distance too, so a pair is only swapped when the contexts nearly
    agree. Large weight -> only near-identical contexts are re-paired, and the conditional is
    preserved; zero -> the biased coupling. It only does anything when the simulator repeats or
    nearly repeats contexts within a batch.
    """

    def __init__(self, transformation, simulate, draw_base, batch_size, weights=None,
                 coupling="independent", context_weight=0.0):
        self.field, self.total_time = transformation.vector_field, transformation.total_time
        self.simulate, self.draw_base = simulate, draw_base
        self.batch_size, self.weights, self.coupling = batch_size, weights, coupling
        self.context_weight = context_weight

    def __call__(self):
        target_coeffs, context = self.simulate(self.batch_size)
        start_coeffs = self.draw_base(self.batch_size)
        return straight_path_loss(self.field, start_coeffs, target_coeffs, self.total_time, context,
                                  self.weights, self.coupling, self.context_weight)


def squared_distances(left, right):
    """[n, m] of squared distances, by expansion rather than an [n, m, dim] difference."""
    return ((left ** 2).sum(-1)[:, None] + (right ** 2).sum(-1)[None, :] - 2 * left @ right.T)


def optimal_pairing(start_coeffs, target_coeffs, weights, context=None, context_weight=0.0):
    """Minibatch optimal transport: the permutation of the targets that minimises the summed
    squared distance to the base draws, in the metric the loss uses (weights = the same per-mode
    weights as the loss, so 'cameron-martin' and 'l2' stay consistent).

    For uniform equal-size batches the OT plan is a permutation, so an exact assignment is the
    whole solver: O(batch^3) on the CPU, about 2 ms at batch 128. One device sync per step.
    """
    from scipy.optimize import linear_sum_assignment       # only needed for this coupling
    scaled_start = start_coeffs if weights is None else start_coeffs * weights
    scaled_target = target_coeffs if weights is None else target_coeffs * weights
    cost = squared_distances(scaled_start, scaled_target)
    if context is not None and context_weight:
        cost = cost + context_weight * squared_distances(context, context)
    _, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(columns, device=target_coeffs.device)


def straight_path_loss(field, start_coeffs, target_coeffs, total_time, context, weights,
                       coupling="independent", context_weight=0.0):
    """One time per sample when the field allows it, otherwise one time for the whole batch."""
    if coupling == "optimal":
        order = optimal_pairing(start_coeffs, target_coeffs, weights, context, context_weight)
        target_coeffs = target_coeffs[order]
        context = None if context is None else context[order]     # the context stays with its target
    elif coupling != "independent":
        raise ValueError(f"coupling must be 'independent' or 'optimal', not {coupling!r}")
    batch = len(start_coeffs)
    if getattr(field, "supports_batched_time", False):
        times = torch.rand(batch, dtype=start_coeffs.dtype, device=start_coeffs.device) * total_time
    else:
        times = torch.as_tensor(torch.rand(()).item() * total_time, dtype=start_coeffs.dtype,
                                device=start_coeffs.device)
    path_coeffs = start_coeffs + times.reshape(-1, 1) / total_time * (target_coeffs - start_coeffs)
    residual = field.velocity(path_coeffs, times, context) - (target_coeffs - start_coeffs) / total_time
    if weights is not None:
        residual = residual * weights
    return residual.pow(2).sum(-1).mean()
