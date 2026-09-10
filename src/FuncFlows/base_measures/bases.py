import torch


class Basis:
    """Abstract basis class.

    Args:
        num_functions (int): Number of functions you want to play around with.
        physical_dim (int, optional): Number of physical dimensions the functions will be over. Defaults to 1.
        dtype (optional): Data type. If you're reading this and don't know what that is. God help you. Defaults to torch.float64.
    """



    def __init__(self, num_functions, physical_dim=1, dtype=torch.float64):
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




class CosineBasis(Basis):
    """Class containing information and stuff for evaluation of Cosine basis functions.

    Args:
        num_functions (int): Number of functions you want to play around with.
        physical_dim (int, optional): Number of physical dimensions the functions will be over. Defaults to 1.
        dtype (optional): Data type. If you're reading this and don't know what that is. God help you. Defaults to torch.float64.
    """



    def __init__(self, num_functions, physical_dim=1, dtype=torch.float64):
        super().__init__(num_functions, physical_dim, dtype=dtype)


        if physical_dim == 1:
            max_mode = num_functions  
        else: 
            max_mode = int((2 * num_functions)**0.5) + 2

        modes = torch.cartesian_prod(*[torch.arange(max_mode, dtype=self.dtype)] * physical_dim).reshape(-1, physical_dim)

        self.modes = modes[torch.argsort((modes ** 2).sum(1), stable=True)][:num_functions]

        self.eval_norm = 1 + (2**0.5 - 1) * (self.modes != 0).to(self.dtype)

        self.laplacian_eigenvalues = ((torch.pi * self.modes) ** 2).sum(1)


    def evaluate(self, points):

        cosines = torch.cos(torch.pi * points.reshape(-1, 1, self.physical_dim) * self.modes)

        return (cosines * self.eval_norm).prod(2)



class FourierBasis(Basis):
    """Class containing information and stuff for evaluation of Fourier basis functions.
    
    Periodic Laplacian eigenfunctions on [0,1]^d: 1, sqrt2 cos(2 pi k t), sqrt2 sin(2 pi k t), ...
 
    Per axis, mode index m -> wavenumber ceil(m/2); odd m is a cosine, even m > 0 a sine.
    Columns are ordered by Laplacian eigenvalue (2 pi)^2 * sum_axis wavenumber^2.
        Args:
            num_functions (int): Number of functions you want to play around with.
            physical_dim (int, optional): Number of physical dimensions the functions will be over. Defaults to 1.
            dtype (optional): Data type. If you're reading this and don't know what that is. God help you. Defaults to torch.float64.
        """

    def __init__(self, num_functions, physical_dim=1, dtype=torch.float64):

        super().__init__(num_functions, physical_dim, dtype)
        modes = torch.cartesian_prod(*[torch.arange(2 * num_functions)] * physical_dim).reshape(-1, physical_dim)
        wavenumbers = (modes + 1) // 2
        keep = torch.argsort((wavenumbers ** 2).sum(1), stable=True)[:num_functions]
        self.wavenumbers = wavenumbers[keep].to(dtype)
        self.phases = -torch.pi / 2 * ((modes[keep] % 2 == 0) & (modes[keep] > 0)).to(dtype)
        self.norm_factors = 1 + (2 ** 0.5 - 1) * (modes[keep] > 0).to(dtype)
        self.laplacian_eigenvalues = ((2 * torch.pi * self.wavenumbers) ** 2).sum(1)


    def evaluate(self, points):
        angles = 2 * torch.pi * points.reshape(-1, 1, self.physical_dim) * self.wavenumbers + self.phases
        return (torch.cos(angles) * self.norm_factors).prod(2)