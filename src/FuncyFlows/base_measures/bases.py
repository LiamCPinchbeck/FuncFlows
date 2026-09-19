import torch


class Basis:
    """Abstract basis class.

    Args:
        num_functions (int): Number of functions you want to play around with.
        physical_dim (int, optional): Number of physical dimensions the functions will be over. Defaults to 1.
        dtype (optional): Data type. If you're reading this and don't know what that is. God help you. Defaults to torch.float64.
    """

    def __init__(self, num_functions: int, physical_dim=1, dtype=torch.float64):
        self.num_functions = num_functions
        self.physical_dim = physical_dim
        self.dtype = dtype

    def evaluate(self, points):
        """points [num_points, physical_dim] -> [num_points, num_functions], L2-orthonormal columns on [0,1]^d."""
        raise NotImplementedError("You are calling an abstract eval method on a Basis class.")

    def quadrature(self, num_points_per_axis):
        quad_points = torch.linspace(0, 1, num_points_per_axis, dtype=self.dtype)
        quad_weights = torch.full_like(quad_points, 1 / (num_points_per_axis - 1))
        quad_weights[[0, -1]] /= 2
        points = torch.cartesian_prod(*[quad_points] * self.physical_dim).reshape(-1, self.physical_dim)
        weights = torch.cartesian_prod(*[quad_weights] * self.physical_dim).reshape(-1, self.physical_dim).prod(1)
        return points, weights


def _leading_modes(num_functions, physical_dim, per_axis, sort_key):
    """The first num_functions multi-indices from a per_axis^d grid, ordered by sort_key.

    per_axis only has to reach past the ball that holds num_functions modes. The n-th mode in d
    dimensions sits at radius ~ (n / volume of the unit d-ball)^(1/d), so a bound of order
    n^(1/d) is enough; arange(2n) per axis — the previous choice — built a (2n)^d grid, which at
    2000 modes in 2-D is 16M entries sorted on every construction (seconds), and in 3-D cannot be
    allocated at all. The selection is only right if the ball fits inside the grid, so the first
    point outside the grid, (per_axis, 0, ...), must sort strictly after the last mode kept.
    """
    axes = [torch.arange(per_axis)] * physical_dim
    modes = torch.cartesian_prod(*axes).reshape(-1, physical_dim) if physical_dim > 1 else axes[0][:, None]
    order = torch.argsort(sort_key(modes), stable=True)[:num_functions]
    outside = torch.zeros(1, physical_dim, dtype=modes.dtype)
    outside[0, 0] = per_axis
    if len(order) < num_functions or sort_key(outside) <= sort_key(modes[order[-1:]]):
        raise ValueError(f"per-axis bound {per_axis} too small for {num_functions} modes in {physical_dim}-D")
    return modes[order]


class CosineBasis(Basis):
    """Neumann Laplacian eigenfunctions on [0,1]^d: prod_axis norm * cos(pi m x).

    Args:
        num_functions (int): Number of functions you want to play around with.
        physical_dim (int, optional): Number of physical dimensions the functions will be over. Defaults to 1.
        dtype (optional): Data type. If you're reading this and don't know what that is. God help you. Defaults to torch.float64.
    """

    def __init__(self, num_functions: int, physical_dim: int = 1, dtype=torch.float64):
        super().__init__(num_functions, physical_dim, dtype=dtype)
        per_axis = num_functions + 1 if physical_dim == 1 else int(2 * num_functions ** (1 / physical_dim)) + 3
        self.modes = _leading_modes(num_functions, physical_dim, per_axis,
                                    lambda modes: (modes ** 2).sum(1)).to(self.dtype)
        self.eval_norm = 1 + (2 ** 0.5 - 1) * (self.modes != 0).to(self.dtype)
        self.laplacian_eigenvalues = ((torch.pi * self.modes) ** 2).sum(1)

    def evaluate(self, points):
        # one axis at a time: peak memory is num_points x num_functions, not x physical_dim as
        # well. .to(points): the mode tables follow the points' device and dtype.
        points = points.reshape(-1, self.physical_dim)
        modes, norms = self.modes.to(points), self.eval_norm.to(points)
        values = None
        for axis in range(self.physical_dim):
            factor = norms[:, axis] * torch.cos(torch.pi * points[:, axis, None] * modes[:, axis])
            values = factor if values is None else values * factor
        return values


class FourierBasis(Basis):
    """Periodic Laplacian eigenfunctions on [0,1]^d: 1, sqrt2 cos(2 pi k t), sqrt2 sin(2 pi k t), ...

    Per axis, mode index m -> wavenumber ceil(m/2); odd m is a cosine, even m > 0 a sine.
    Columns are ordered by Laplacian eigenvalue (2 pi)^2 * sum_axis wavenumber^2.

    Args:
        num_functions (int): Number of functions you want to play around with.
        physical_dim (int, optional): Number of physical dimensions the functions will be over. Defaults to 1.
        dtype (optional): Data type. If you're reading this and don't know what that is. God help you. Defaults to torch.float64.
    """

    def __init__(self, num_functions: int, physical_dim=1, dtype=torch.float64):
        super().__init__(num_functions, physical_dim, dtype)
        # two mode indices per wavenumber, so twice the cosine bound, plus slack
        per_axis = num_functions + 1 if physical_dim == 1 else 2 * int(2 * num_functions ** (1 / physical_dim)) + 3
        modes = _leading_modes(num_functions, physical_dim, per_axis,
                               lambda modes: (((modes + 1) // 2) ** 2).sum(1))
        self.wavenumbers = ((modes + 1) // 2).to(dtype)
        self.phases = -torch.pi / 2 * ((modes % 2 == 0) & (modes > 0)).to(dtype)    # sine = shifted cosine
        self.norm_factors = 1 + (2 ** 0.5 - 1) * (modes > 0).to(dtype)
        self.laplacian_eigenvalues = ((2 * torch.pi * self.wavenumbers) ** 2).sum(1)

    def evaluate(self, points):
        points = points.reshape(-1, self.physical_dim)
        wavenumbers, phases, norms = (self.wavenumbers.to(points), self.phases.to(points),
                                      self.norm_factors.to(points))
        values = None
        for axis in range(self.physical_dim):                  # same axis-by-axis product
            angles = 2 * torch.pi * points[:, axis, None] * wavenumbers[:, axis] + phases[:, axis]
            factor = norms[:, axis] * torch.cos(angles)
            values = factor if values is None else values * factor
        return values
