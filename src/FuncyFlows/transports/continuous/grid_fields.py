
import torch
from .vector_fields import VectorField, TimeCosines, combine_context
import warnings

################################################################################################################
################################################################################################################
################################################################################################################


class GridTransform(torch.nn.Module):
    r"""Coefficients <-> values on a uniform grid, for one fixed basis truncation.

    Dense matrix product that works for arbitrary bases. 

    An FFT that works only for a separable Fourier basis and is picked automatically when it agrees
    with the dense one to basic round-off.

    Quick Ref
    ---------------
    The two maps::

      to_grid(v)   = Φ v        [.., M]  -> [.., Gᵈ]    synthesis: evaluate the function
      from_grid(y) = w Φᵀ y     [.., Gᵈ] -> [.., M]     analysis: integrate against each mode
      w            = G⁻ᵈ                                cell volume (``quadrature_weight``)

      from_grid(to_grid(v)) == v   iff  w ΦᵀΦ = I   (Fourier: needs G ≥ 2·k_max + 2)

    Note the asymmetry: **w appears in one direction only**. Evaluating a function at a point is a
    sum over modes and needs no weight; recovering a coefficient is an integral over space, which
    the rectangle rule turns into a sum times the cell volume.

    **Mathematical Formulation**

    (annoyingly is not readable in VS code natively. The notes bit right after this *is* readable.)

    With :math:`\Phi \in \mathbb{R}^{G^d \times M}` the basis evaluated on the grid points
    :math:`x_p` and :math:`w = G^{-d}`:

    .. math::

        \begin{aligned}
        u_p    &= \sum_{k=1}^{M} \Phi_{pk}\, v_k          && \text{to\_grid} \\
        v_k    &= w \sum_{p=1}^{G^d} \Phi_{pk}\, u_p
                  \;\approx\; \int_{[0,1]^d} u(x)\,\phi_k(x)\,\mathrm{d}x   && \text{from\_grid}
        \end{aligned}

    The round trip is the identity exactly when the columns of :math:`\Phi` are orthonormal *under
    that quadrature*, :math:`w\,\Phi^{\!\top}\Phi = I` -- discrete orthogonality, not just the
    continuous kind.

    Notes
    -----
    1. The grid is **cell-edge**: xₚ = p/G for p = 0 … G-1.

       - Not cell-centred, so the FFT needs no half-sample phase factor.
       - On a periodic uniform grid the rectangle rule is spectrally accurate for band-limited
         functions, so nothing fancier (trapezium, Gauss) buys anything for a Fourier basis.

    2. Dense path. Store Φ as a [Gᵈ, M] buffer and matrix-multiply.

       - Works for any basis, any grid size.
       - Cost O(B·M·Gᵈ) per direction, and O(Gᵈ·M) of memory.

    3. FFT path. Separable Fourier bases only.

       - One rfft/irfft per axis, the d axes done in turn with ``movedim``.
       - Cost O(d·B·Gᵈ·log G), no dense matrix touched.
       - The packed [M] coefficient vector is scattered into a [max_axis_mode]ᵈ tensor-product
         block by ``flat_index``, transformed axis by axis, then gathered back.
       - Per-axis mode index m: 0 -> constant, odd -> cos with k=(m+1)//2, even>0 -> sin with
         k=m//2. The √2 in the basis normalisation is why the constants are G and G/√2 going out,
         1/G and √2/G coming back.

    4. Which path, via ``mode``:

       - ``"auto"`` (default): build the FFT tables if the basis exposes ``wavenumbers``, keep the
         FFT only if ``_agrees`` passes, otherwise fall back to dense.
       - ``"dense"``: never try the FFT.
       - ``"fft"``: demand it, and raise if the basis or the check says no.

    5. ``_agrees`` is a numerical self-test: the FFT path is a fixed reindexing of
       the dense one, so with random probes the two must match to 1e-9 (float64) or 1e-4 (float32).
       Lil check for a basis whose mode ordering isn't the one the tables assume.

    Attributes
    ----------
    basis_grid : Tensor [Gᵈ, M]
        Φ itself, always kept. Clients build their own spatial quantities from it --
        ``PointwiseField`` uses it for the smooth a(x), g(x), c(x) profiles and the Σⱼ Φₚⱼ² sums.
    basis_values : Tensor [Gᵈ, M] or None
        The copy the dense path multiplies by; None when ``use_fft`` is True.
    quadrature_weight : float
        w = G⁻ᵈ.
    use_fft : bool
        Which path ``to_grid`` / ``from_grid`` actually take.

    ***DANGER DANGER*** 
    
    The Nyquist check lives in the FFT table builder: it refuses a grid with k_max >= G/2 and tells 
    you to use at least 2·k_max + 2. (Good behaviour)

    That check does **not** run on the *dense* path or on a *non-Fourier basis*. In this case, 
    too few grid points silently breaks discrete orthogonality (NOT good behaviour), the round trip stops 
    being the identity, and every trace computed on top of it (e.g. ``PointwiseField``) is wrong. 
    
    Nyquist is only the floor for a *linear* round trip; anything with a nonlinearity on the grid wants ~6·k_max.
    """

    def __init__(self, basis, num_active, grid_size, mode="auto", check=True, dtype=torch.float64):
        super().__init__()

        self.num_active, self.grid_size, self.dtype = num_active, grid_size, dtype
        self.physical_dim = basis.physical_dim
        self.quadrature_weight = grid_size ** (-basis.physical_dim)
        device = basis.laplacian_eigenvalues.device


        # Fourier is orthogonal cell-edge and the FFT tables assume no half-sample phase;
        # a cosine basis is orthogonal only cell-centred (edge leaks 2/G into every odd pair).
        offset = 0.0 if hasattr(basis, "wavenumbers") else 0.5
        axis = (torch.arange(grid_size, dtype=dtype, device=device) + offset) / grid_size

        points = torch.cartesian_prod(*[axis] * basis.physical_dim).reshape(-1, basis.physical_dim)
        values = basis.evaluate(points)[:, :num_active]

        self.register_buffer("basis_grid", values)                 # [grid^d, num_active]

        self.use_fft = False

        if mode != "dense" and hasattr(basis, "wavenumbers"):      # separable Fourier basis only
            self._build_tables(basis, num_active, grid_size, device)
            self.use_fft = self._agrees(values)


        if mode == "fft" and not self.use_fft:
            raise RuntimeError("the FFT transform needs a FourierBasis and must pass its check; "
                               "pass transform='dense'")
        
        self.register_buffer("basis_values", None if self.use_fft else values)

        if check:
            error = self._round_trip_error(values)
            tolerance = 1e-4 if dtype is torch.float32 else 1e-9
            if error > tolerance:
                raise ValueError(
                    f"from_grid(to_grid(v)) is off by {error:.2e} (tolerance {tolerance:.0e}): the basis "
                    f"is not orthogonal on this grid, so every projection and every trace built on it is "
                    f"wrong. grid_size={grid_size}, num_active={num_active}, dim={self.physical_dim}. "
                    f"Use a larger grid, or a grid that matches the basis (Fourier: cell-edge; "
                    f"cosine: cell-centred). check=False to override.")



    def _build_tables(self, basis, num_active, grid_size, device):
        """Per-axis mode index m: 0 -> constant, odd -> cos with k=(m+1)//2, even>0 -> sin with k=m//2."""
        wavenumbers = basis.wavenumbers[:num_active].round().long()        # [num_active, d]

        is_sine = basis.phases[:num_active] != 0

        modes = torch.where(wavenumbers == 0, torch.zeros_like(wavenumbers),
                            2 * wavenumbers - 1 + is_sine.long())
        
        self.max_axis_mode = int(modes.max().item()) + 1
        highest = int(wavenumbers.max().item())


        if highest >= grid_size // 2:
            raise ValueError(f"grid_size {grid_size} is below Nyquist for wavenumber {highest}; "
                             f"use at least {2 * highest + 2}")
        flat = torch.zeros(num_active, dtype=torch.long, device=device)


        for axis in range(self.physical_dim):
            flat = flat * self.max_axis_mode + modes[:, axis]


        
        self.register_buffer("flat_index", flat)
        axis_modes = torch.arange(1, self.max_axis_mode, device=device)
        cosine, sine = axis_modes[axis_modes % 2 == 1], axis_modes[axis_modes % 2 == 0]


        self.register_buffer("cosine_mode", cosine)
        self.register_buffer("cosine_wave", (cosine + 1) // 2)
        self.register_buffer("sine_mode", sine)
        self.register_buffer("sine_wave", sine // 2)
        self.register_buffer("zero_index", torch.zeros(1, dtype=torch.long, device=device))



    def _synthesise_axis(self, coefficients):
        """[..., max_axis_mode] -> [..., grid_size] along the last axis."""
        shape = coefficients.shape[:-1] + (self.grid_size // 2 + 1,)
        scale = self.grid_size / 2 ** 0.5

        real = torch.zeros(shape, dtype=coefficients.dtype, device=coefficients.device)
        real = real.index_add(-1, self.zero_index, self.grid_size * coefficients[..., :1])
        real = real.index_add(-1, self.cosine_wave, scale * coefficients[..., self.cosine_mode])

        imaginary = torch.zeros(shape, dtype=coefficients.dtype, device=coefficients.device)
        imaginary = imaginary.index_add(-1, self.sine_wave, -scale * coefficients[..., self.sine_mode])

        return torch.fft.irfft(torch.complex(real, imaginary), n=self.grid_size, dim=-1)

    def _analyse_axis(self, values):
        spectrum = torch.fft.rfft(values, dim=-1)
        scale = 2 ** 0.5 / self.grid_size

        coefficients = torch.zeros(values.shape[:-1] + (self.max_axis_mode,),
                                   dtype=values.dtype, device=values.device)
        
        coefficients = coefficients.index_add(-1, self.zero_index, spectrum.real[..., :1] / self.grid_size)
        coefficients = coefficients.index_add(-1, self.cosine_mode, scale * spectrum.real[..., self.cosine_wave])

        return coefficients.index_add(-1, self.sine_mode, -scale * spectrum.imag[..., self.sine_wave])

    def to_grid(self, coeffs):
        """[..., num_active] -> [..., grid_size^d]."""
        if not self.use_fft:
            return coeffs @ self.basis_values.T
        
        block = torch.zeros(coeffs.shape[:-1] + (self.max_axis_mode ** self.physical_dim,),
                            dtype=coeffs.dtype, device=coeffs.device)
        block = block.index_add(-1, self.flat_index, coeffs)
        block = block.reshape(coeffs.shape[:-1] + (self.max_axis_mode,) * self.physical_dim)

        for _ in range(self.physical_dim):
            block = self._synthesise_axis(block).movedim(-1, -self.physical_dim)

        return block.reshape(coeffs.shape[:-1] + (self.grid_size ** self.physical_dim,))

    def from_grid(self, values):
        """[..., grid_size^d] -> [..., num_active] projection."""
        if not self.use_fft:
            return (values @ self.basis_values) * self.quadrature_weight
        
        block = values.reshape(values.shape[:-1] + (self.grid_size,) * self.physical_dim)

        for _ in range(self.physical_dim):
            block = self._analyse_axis(block).movedim(-1, -self.physical_dim)

        return block.reshape(values.shape[:-1]
                             + (self.max_axis_mode ** self.physical_dim,))[..., self.flat_index]

    def _agrees(self, values, tolerance=None):
        """The FFT path is a fixed reindexing of the dense one, so it should agree to round-off."""
        tolerance = tolerance or (1e-4 if self.dtype is torch.float32 else 1e-9)

        probe = torch.randn(3, self.num_active, dtype=self.dtype, device=values.device)
        grid_probe = torch.randn(3, values.shape[0], dtype=self.dtype, device=values.device)
        self.use_fft = True

        try:
            forward = (self.to_grid(probe) - probe @ values.T).abs().max().item()
            backward = (self.from_grid(grid_probe)
                        - (grid_probe @ values) * self.quadrature_weight).abs().max().item()
        except Exception:
            self.use_fft = False
            return False
        
        self.use_fft = False
        scale = max(1.0, probe.abs().max().item())
        return bool(forward < tolerance * scale and backward < tolerance * scale)


    def _round_trip_error(self, values):
        """w Phi^T Phi = I is the condition every projection and every trace on this grid relies on.
        Random probes plus the highest mode alone, which is the first to fail."""
        probe = torch.randn(4, self.num_active, dtype=self.dtype, device=values.device)
        probe[0] = 0
        probe[0, -1] = 1.0
        returned = self.from_grid(self.to_grid(probe))
        return (returned - probe).abs().max().item() / max(1.0, probe.abs().max().item())



################################################################################################################
################################################################################################################
################################################################################################################

# took me 30 minutes to figure out how to do the math formatting, you better appreciate it!!
    # but yes I got Gemini to do the unicode function version. Ain't nobody got time for that.
    # but yes the docstring is huge coz if it were me, I'd have no idea what was going on
    # even with my attempt to make informative variable names

class PointwiseField(VectorField):
    r"""One Fourier-neural-operator layer (arXiv:2010.08895).

    Evaluates the function on a grid, applies a point-wise nonlinearity, 
    and projects back into coefficient space.

    Quick Reference
    ---------------
    Step-by-step transformation:
      ṽ      = S⁻¹ v                       (whiten)
      u      = Φ ṽ                         (to grid)
      ū      = Φ (κ(t) ⊙ ṽ)                (low-pass copy)
      z      = g(x,t) ⊙ u + ū + c(x,t)     (pre-activation)
      h̃      = w Φᵀ [a(x,t) ⊙ tanh z]      (gate -> back to coefs)
      h(v,t) = S h̃                         (un-whiten)

    Expanded single line (sum over grid points G):
      hⱼ(v,t) = σⱼ·w ∑ₚ₌₁ Φₚⱼ · aₚ(t) · tanh( gₚ(t)·(Φ S⁻¹ v)ₚ + (Φ·(κ(t) ⊙ S⁻¹ v))ₚ + cₚ(t) )
          
    (Sum should be \sum_{p=1}^G but I couldn't figure out how to do that without it looking terrible)

    **Layer Diagram**

    ::

    
                    g(x,t)      ū = Φ(κ(t) ⊙ ṽ)      c(x,t)              a(x,t)
                       └──────────────┬──────────────────┘                  │
                                      ▼                                     ▼
              ┌───────┐    ┌──────────────────┐    ┌──────┐    ╭───╮    ┌─────────┐
     v ──▶ S⁻¹│  Φ ·  │──▶ │  z = g⊙u + ū + c │──▶ │ tanh │──▶ │ × │──▶ │  w Φᵀ · │──▶ S ──▶ h(v,t)
              └───────┘    └──────────────────┘    └──────┘    ╰───╯    └─────────┘
               to grid                                            │      to coefs
               [B, Gᵈ]           [B, Gᵈ]                          │       [B, M]
                                                                  │
                                            h = tanh z            │  1 − h²
                                                                  ▼
                                              ┌───────────────────────────────┐
                                              │  w Σₚ a(x)·g(x)·(1−h²)·Σⱼ Φ²ₚⱼ │──▶ Tr ∂h/∂v
                                              └───────────────────────────────┘
                                                exact, from the same forward pass

    
    
    Every arrow entering from above is a *function of position* -- a short series in the first
    ``num_field_modes`` basis functions -- and none of them depends on v. That is the whole
    reason the trace tap on the right exists: the grid map is applied point by point, so
    ∂u_out/∂u_in is diagonal and the divergence collapses to a weighted count over grid points.

    Contrast ``OperatorField``, where the gates are constants in x, there are L of these in
    series, and a spectral operator between them couples the grid points -- at which point the
    Jacobian is a product of dense factors and the tap on the right is replaced by probes.



    
    The transformation in order is...

    1. Whiten. Divide each coefficient by its base-measure standard deviation, so every mode is
        order one before anything nonlinear sees it. (Undone at the end; the divergence is unchanged.)
        This is done with the `mode_scale` attributes.

    2. Go to the grid. Evaluate the whitened function at G uniformly spaced points.

        - Also build a low-pass copy: keep only the first few modes, each scaled by a learned number, and evaluate
          that too. This gives each grid point a local average to compare itself against.
          Giving more context on the 'neighbourhoods' of the points.

        - The grid transformations are handled by, you guessed it, `GridTransform`. 
        - Do you ever worry that you make variable names __too__ obvious? Just me? Neat

    3. Form the pre-activation at every grid point

        - The operation in english is

          - a *learned* gain times the function value,
          - plus the low-pass copy,
          - plus a learned offset.

        - Gain and offset are smooth functions of position (a handful of low spatial modes each), so neighbouring points are treated alike.
        - They don't depend on the functions/function coefficients being transformed. Keeping the derivative tractable.
    
    4. Apply `tanh`, point by point.

        - This is the only place a non-linearity comes in for this class by itself. 
        - This simplicity keeps the divergence closed-form.
        - Use `SumField` would stack this, but not compose the non-linearity. There should be enough seeing as the transformation
          is done at every integration step though.

    5. Multiply by a learned amplitude, also a smooth function of position.

        - This decides how much velocity the layer is allowed to produce in each region.
        - Starts small so the flow starts near the identity (controlled by init_scale) so the flow starts near the identity

    6. Transform back to coefficients

        - integrate against each basis function over the grid (a sum over grid points weighted by the cell area), then un-whiten.

    Every learned quantity (gain, offset, amplitude, low-pass scalings) is a short cosine series in 'flow time', 
    so the layer changes smoothly along the ODE without any time input to the tanh. Handled by `TimeCosines`.

    None of them depends on the function itself; that is the condition under which the trace of the
    Jacobian collapses to a weighted count, over grid points, of how much of the tanh is still in its
    linear regime -- computed from the same forward pass, no probes, no probs.

    DANGER DANGER || Grid resolution: the tanh generates harmonics, so the grid must resolve about three times the
    highest wavenumber in the basis (grid_size >= 6 * k_max) or the projection back aliases and the trace is wrong.

    


    ---

    **Mathematical Formulation**

    With :math:`S = \mathrm{diag}(\sigma_k)` as the base-measure scales, 
    :math:`\Phi \in \mathbb{R}^{G \times M}` as the basis on the grid, 
    and :math:`w = G^{-d}` as the cell volume:

    .. math::

        \begin{aligned}
        \tilde v &= S^{-1} v                                         && \text{whiten} \\
        u        &= \Phi \tilde v                                    && \text{to the grid} \\
        \bar u   &= \Phi\,(\kappa(t) \odot \tilde v)                 && \text{low-pass copy} \\
        z        &= g(x,t) \odot u + \bar u + c(x,t)                 && \text{pre-activation} \\
        \tilde h &= w\,\Phi^{\!\top}\big[\,a(x,t) \odot \tanh z\,\big] && \text{gate, back to coefficients} \\
        h(v,t)   &= S\,\tilde h                                      && \text{un-whiten}
        \end{aligned}

    Single-line expanded form:

    .. math::

        h_j(v,t) = \sigma_j\, w \sum_{p=1}^{G} \Phi_{pj}\; a_p(t)\,
               \tanh\!\Big( g_p(t)\,(\Phi S^{-1} v)_p + \big(\Phi\,(\kappa(t) \odot S^{-1} v)\big)_p + c_p(t) \Big).
    



    """

    supports_batched_time = True

    def __init__(self, basis, num_active, grid_size, num_time_modes=4, num_field_modes=32,
                 num_spectral_modes=0, context_dim=0, total_time=1.0, mode_scale=None,
                 init_scale=0.05, transform="auto", time_pair=False, dtype=torch.float64):
        super().__init__()
        self.num_active, self.num_time_modes = num_active, num_time_modes
        self.total_time, self.mode_scale, self.dtype = total_time, mode_scale, dtype
        self.physical_dim, self.grid_size = basis.physical_dim, grid_size
        self.quadrature_weight = grid_size ** (-basis.physical_dim)

        self.cosines = TimeCosines(num_time_modes, total_time, dtype, time_pair)

        self.transform = GridTransform(basis, num_active, grid_size, mode=transform, dtype=dtype)

        self.quadrature_weight = self.transform.quadrature_weight
        device = basis.laplacian_eigenvalues.device
        values = self.transform.basis_grid

        self.register_buffer("squared_sum", (values ** 2).sum(1))          # S(x), [grid^d]
        self.register_buffer("field_values", values[:, :num_field_modes])  # a, g, c stay dense

        gain_scale = init_scale / num_field_modes ** 0.5
        self.gain_stack = torch.nn.Parameter(
            gain_scale * torch.randn(num_time_modes, num_field_modes, dtype=dtype))
        
        self.slope_stack = torch.nn.Parameter(torch.zeros(num_time_modes, num_field_modes, dtype=dtype))
        self.shift_stack = torch.nn.Parameter(torch.zeros(num_time_modes, num_field_modes, dtype=dtype))

        with torch.no_grad():                       # phi_0 = 1, so this makes g(x) a constant.
            # whitened coefficients are O(1) each, so the whitened field is O(sqrt(num_active))
            self.slope_stack[0, 0] = 1.0 if mode_scale is None else num_active ** -0.5

        self.context_stack = None
        if context_dim:
            self.context_stack = torch.nn.Parameter(
                init_scale / context_dim ** 0.5
                * torch.randn(num_time_modes, num_field_modes, context_dim, dtype=dtype))


        self.kappa_stack = None
        if num_spectral_modes:
            logs = torch.log1p(basis.laplacian_eigenvalues[:num_active].to(dtype))
            span = (logs.max() - logs.min()).clamp(min=1e-12)
            position = ((logs - logs.min()) / span).clamp(0, 1)
            orders = torch.arange(num_spectral_modes, dtype=dtype, device=device)
            self.register_buffer("spectral_features", torch.cos(torch.pi * orders * position[:, None]))
            self.register_buffer("squared_basis", values ** 2)             # [grid^d, num_active]
            # zero, not init_scale * randn: the slope path is normalised by num_active^-1/2 so the
            # pre-activation is O(1), but kappa multiplies whitened coefficients that sum over all
            # num_active modes. At 0.05 * randn that term is O(0.17 * sqrt(2000)) ~ 7.6, tanh is
            # flat, (1 - hidden^2) ~ 0, and neither the trace nor any gradient gets through.
            self.kappa_stack = torch.nn.Parameter(
                torch.zeros(num_time_modes, num_spectral_modes, dtype=dtype))


        if hasattr(basis, "wavenumbers"):
            k_max = int(basis.wavenumbers[:num_active].abs().max().item())
        else:                                   # cosine: lambda_k = (k pi)^2
            k_max = int(round(basis.laplacian_eigenvalues[:num_active].max().sqrt().item() / torch.pi))


        if grid_size < 6 * k_max:
            warnings.warn(f"grid_size {grid_size} < 6 * k_max = {6 * k_max}: tanh harmonics alias back "
                          f"onto the retained modes. The velocity tolerates this; the closed-form trace, "
                          f"log_rn_at and ImportanceCorrection do not.", stacklevel=2)

    ################################################################################
    ################################################################################
    # field

    def _profiles(self, time_value, context):
        weights = self.cosines(time_value)
        gain = (weights @ self.gain_stack) @ self.field_values.T
        slope = (weights @ self.slope_stack) @ self.field_values.T
        shift_coeffs = weights @ self.shift_stack


        if context is not None and self.context_stack is not None:
            shift_coeffs = shift_coeffs + combine_context(self.context_stack, context, weights)
        
        multiplier = None
        if self.kappa_stack is not None:
            multiplier = (weights @ self.kappa_stack) @ self.spectral_features.T
        
        return gain, slope, shift_coeffs @ self.field_values.T, multiplier



    def _hidden(self, coeffs, time_value, context):
        gain, slope, shift, multiplier = self._profiles(time_value, context)
        active = coeffs[..., :self.num_active]

        if self.mode_scale is not None:
            active = active / self.mode_scale[:self.num_active]
        
        if multiplier is None:
            pre_activation = slope * self.transform.to_grid(active) + shift
        else:                                       # one transform for both fields, not two
            both = self.transform.to_grid(torch.stack([active, active * multiplier]))
            pre_activation = slope * both[0] + both[1] + shift                      # + (K v)(x)
        
        return gain, slope, multiplier, torch.tanh(pre_activation)



    def _update(self, coeffs, gain, hidden):
        update = self.transform.from_grid(gain * hidden)

        if self.mode_scale is not None:
            update = update * self.mode_scale[:self.num_active]
        
        return torch.nn.functional.pad(update, (0, coeffs.shape[-1] - self.num_active))



    def _trace(self, gain, slope, multiplier, hidden):
        local = slope * self.squared_sum
        if multiplier is not None:
            local = local + multiplier @ self.squared_basis.T         # T(x) = sum_j kappa_j phi_j^2
        return ((gain * local) * (1 - hidden ** 2)).sum(-1) * self.quadrature_weight



    @property
    def use_fft(self):
        """Lives on the transform now. Kept here so callers written before it was factored out
        (the scripts' preflight, among others) don't explode."""
        return self.transform.use_fft



    def velocity(self, coeffs, time_value, context=None):

        gain, _, _, hidden = self._hidden(coeffs, time_value, context)

        return self._update(coeffs, gain, hidden)



    def trace(self, coeffs, time_value, context=None):
        gain, slope, multiplier, hidden = self._hidden(coeffs, time_value, context)

        return self._trace(gain, slope, multiplier, hidden)



    def velocity_and_trace(self, coeffs, time_value, context=None):

        gain, slope, multiplier, hidden = self._hidden(coeffs, time_value, context)

        return self._update(coeffs, gain, hidden), self._trace(gain, slope, multiplier, hidden)









################################################################################################################
################################################################################################################
################################################################################################################

class OperatorField(VectorField):
    r"""A stack of Fourier-neural-operator layers as the velocity (arXiv:2010.08895).

    The multi-layer sibling of ``PointwiseField``. Where that class applies one ``tanh`` on the
    grid and keeps a closed-form divergence, this one alternates local nonlinearity with global
    spectral mixing ``L`` times over and pays for it with an *estimated* trace. But who needs
    densities anyway...

    Choose this over ``PointwiseField`` only when you need mixing *between* nonlinearities. The
    ODE already applies the field at every integration step, so depth in t is free; depth in the
    layer costs the closed-form trace.

    **Quick Reference**

    Step-by-step transformation::

      ṽ        = S⁻¹ v                                  (whiten)
      z₀       = (Φ ṽ) ⊗ ℓ                              (to grid, lift to C channels)
      z_l      = tanh( z_{l-1} Wₗ(t)ᵀ + Kₗ(t) z_{l-1} + bₗ(t,c) )      l = 1 … L
      h̃        = w Φᵀ ( z_L · π(t) )                    (project channels, back to coefficients)
      h(v,t)   = S h̃                                    (un-whiten)

    The spectral operator, per output channel::

      (Kₗ z)[p, i] = ∑ⱼ Φ_p · ( κₗ,ᵢⱼ(t) ⊙ w Φᵀ z[:, j] )
      κₗ,ᵢⱼ(t)_k   = ∑ₛ θₗ,ᵢⱼₛ(t) · cos( π s ρ_k ),   ρ_k = normalised log(1 + λ_k)

    so ``Kₗ`` is **diagonal in the eigenbasis** (a per-mode multiplier) but **dense across
    channels**. It is smooth in log-wavenumber by construction, which is what makes the learned
    multipliers independent of where the basis happens to be truncated.

    Expanded single line (sums over grid points Gᵈ, channels C, modes M):

      hₘ(v,t) = σₘ · w ∑ₚ₌₁ Φₚₘ ∑ᵢ₌₁ πᵢ(t) · z_L[p,i]

    where the layers recurse as

      z₀[p,i]  = ℓ ∑ₖ₌₁ Φₚₖ σₖ⁻¹ vₖ

      z_l[p,i] = tanh( ∑ⱼ₌₁ Wₗ,ᵢⱼ(t)·z_{l-1}[p,j]
                     + ∑ⱼ₌₁ ∑ₖ₌₁ Φₚₖ · κₗ,ᵢⱼ(t)ₖ · w ∑_q₌₁ Φ_qk · z_{l-1}[q,j]
                     + bₗ,ᵢ(t,c) )

    Collapsed at L = 1, C = 1 (where w ΦᵀΦ = I lets the spectral term fold back):

      hₘ(v,t) = σₘ · w ∑ₚ₌₁ Φₚₘ · π(t) · tanh( ℓ·W(t)·(Φ S⁻¹ v)ₚ
                                              + ℓ·(Φ·(κ(t) ⊙ S⁻¹ v))ₚ
                                              + b(t,c) )

    Compare ``PointwiseField``: identical in shape, but there ``π``, ``ℓ·W`` and ``b`` are aₚ(t),
    gₚ(t) and cₚ(t) -- functions of position. Here they are constants. That is the whole
    difference at L = 1: this class is translation-equivariant, that one is not.

    

    **Layer Diagram**

    ::

    
                                                               b_l(t,c)
                                                                   │
                        ┌──────────────┐                           ▼
                  ┌────▶│   W_l(t) ·   │────────────────────────▶ ╭───╮     ┌──────┐
                  │     └──────────────┘  local: channels only    │ + │────▶│ tanh │───▶ z_l
       z_{l-1} ───┤                                               ╰───╯     └──────┘
      [B, Gᵈ, C]  │     ┌──────────┐  ┌─────────────┐  ┌────────┐   ▲
                  └────▶│  w Φᵀ ·  │─▶│  κ_l(t) ⊙   │─▶│  Φ ·   │───┘
                        └──────────┘  └─────────────┘  └────────┘
                         to coefs       per mode         to grid
                         [B, C, M]      [C, C, M]       [B, C, Gᵈ]

                        global: the only path between grid points, and the reason
                        the Jacobian stops being diagonal

    The upper branch never moves information across the domain -- it is the same ``[C, C]``
    matrix at every grid point. The lower branch is why the class exists: project each channel
    onto the basis, scale each mode, synthesise back. Strip it from layers 2…L and the grid map
    becomes pointwise, the Jacobian becomes diagonal, and the trace closes in one line.
    



    **Notation**

    Every symbol in this docstring, and the tensor that carries it.

    ========= ======================= =============== ========================================
    Symbol    Code                    Shape           Meaning
    ========= ======================= =============== ========================================
    v         `coeffs[..., :M]`         [B, M]          coefficient vector, the ODE state
    ṽ         `active` / `mode_scale`     [B, M]          whitened coefficients
    M         `num_active`              int             retained modes
    Gᵈ        `grid_size ** dim`        int             grid points
    Φ         `transform.basis_grid`    [Gᵈ, M]         basis functions evaluated on the grid
    Φ ·       `transform.to_grid`       --              synthesis, coefficients -> values
    wΦᵀ·      `transform.from_grid`     --              analysis, values -> coefficients
    w         `quadrature_weight`       float           cell volume G⁻ᵈ; w ΦᵀΦ = I
    S         `mode_scale`              [M]             diag(σ_k), base-measure scales
    C         `num_channels`            int             channels carried on the grid
    L         `num_layers`              int             operator layers
    N_s       `num_spectral_modes`      int             terms in each spectral multiplier
    T_m       `num_time_modes`          int             cosine terms in flow time
    λ_k       `laplacian_eigenvalues`   [M]             eigenvalue of mode k
    ρ_k       `spectral_features`       [M, N_s]        normalised log(1+λ_k), as cos(π s ρ_k)
    ℓ         `lift`                    [C]             fixed channel scale; buffer, not learned
    z_l       `state`                   [B, Gᵈ, C]      layer state on the grid
    W_l(t)    `pointwise_stack[l]`      [T_m, C, C]     local channel mixing
    κ_l(t)    `spectral_stack[l]`       [T_m,C,C,N_s]   per-mode multiplier, dense in channels
    κ_l(t)_k  `multipliers`             [C, C, M]       the same, after both contractions
    b_l(t,c)  `bias_stack[l]`           [T_m, C]        per-channel offset; the context entry point
    π(t)      `project_stack`           [T_m, C]        channel readout
    ========= ======================= =============== ========================================

    Indices: k, m over modes; p, q over grid points; i, j over channels; s over spectral
    orders; l over layers. B is the batch. Quantities written θ(t) are ``weights @ θ_stack``
    with ``weights = cosines(time_value)``.


    **Mathematical Formulation**

    With :math:`S = \mathrm{diag}(\sigma_k)` the base-measure scales,
    :math:`\Phi \in \mathbb{R}^{G^d \times M}` the basis on the grid, :math:`w = G^{-d}` the cell
    volume, :math:`C` channels, :math:`L` layers and :math:`N_s` spectral modes:

    .. math::

        \begin{aligned}
        \tilde v   &= S^{-1} v                                          && \text{whiten} \\
        z_0        &= \ell \, (\Phi \tilde v) \otimes \mathbf{1}_C       && \text{lift} \\
        z_{l}      &= \tanh\!\big( W_l(t)\, z_{l-1} + K_l(t)\, z_{l-1} + b_l(t,c) \big)
                                                                        && l = 1,\dots,L \\
        h(v,t)     &= S\, w\, \Phi^{\!\top} \big( \pi(t)^{\!\top} z_L \big)
                                                                        && \text{project}
        \end{aligned}

    where :math:`W_l(t) \in \mathbb{R}^{C \times C}` acts point by point, and

    .. math::

        \big(K_l(t)\, z\big)_{:,i} \;=\; \sum_{j=1}^{C}
            \Phi \Big[ \kappa_{l,ij}(t) \odot w \Phi^{\!\top} z_{:,j} \Big],
        \qquad
        \kappa_{l,ij}(t)_k = \sum_{s=0}^{N_s - 1} \theta_{l,ijs}(t)\, \cos(\pi s \rho_k).

    Single-line expanded form:

    .. math::

        h_m(v,t) = \sigma_m\, w \sum_{p=1}^{G^d} \Phi_{pm}
                   \sum_{i=1}^{C} \pi_i(t)\, z_L[p,i],

    with the layer recursion

    .. math::

        \begin{aligned}
        z_0[p,i] &= \ell \sum_{k=1}^{M} \Phi_{pk}\, \sigma_k^{-1} v_k \\
        z_l[p,i] &= \tanh\!\Big(
            \sum_{j=1}^{C} W_{l,ij}(t)\, z_{l-1}[p,j]
            \;+\; \sum_{j=1}^{C} \sum_{k=1}^{M} \Phi_{pk}\, \kappa_{l,ij}(t)_k\,
                  w \sum_{q=1}^{G^d} \Phi_{qk}\, z_{l-1}[q,j]
            \;+\; b_{l,i}(t,c) \Big).
        \end{aligned}

    At :math:`L = 1`, :math:`C = 1` the inner projection collapses through
    :math:`w\Phi^{\!\top}\Phi = I` and this reduces to

    .. math::

        h_m(v,t) = \sigma_m\, w \sum_{p=1}^{G^d} \Phi_{pm}\, \pi(t)\,
            \tanh\!\Big( \ell\,W(t)\,(\Phi S^{-1} v)_p
                       + \ell\,\big(\Phi\,(\kappa(t) \odot S^{-1} v)\big)_p
                       + b(t,c) \Big).

    That collapse holds only because :math:`w\Phi^{\!\top}\Phi = I`, the round-trip condition
    ``GridTransform`` now checks at construction. Break it and the two classes stop agreeing,
    silently.

    The transformation in order is...

    1. Whiten and lift. Divide by the base-measure scales, evaluate on the grid, and copy the
       single field into ``num_channels`` identical channels scaled by ``lift``.

       - ``lift`` is a **buffer, not a parameter**: a whitened field is O(√num_active) on the grid,
         so without the ``num_active^-1/2`` factor every ``tanh`` starts saturated and nothing --
         neither signal nor gradient -- reaches layer 2. Same trap as ``PointwiseField``'s ``κ``.
       - All channels start identical; they only differentiate through ``pointwise_stack``.

    2. Each layer does three things and then a ``tanh``:

       - ``W_l z``: mixes channels *at each grid point*, local.
       - ``K_l z``: projects to coefficients, multiplies each mode by a learned number, and comes
         back. Global, and the only way information crosses the grid inside the stack.
       - ``b_l``: a per-channel offset, the only place ``context`` enters.

    3. Project. A learned time-dependent weight per channel collapses ``C`` back to one field,
       which is integrated against the basis and un-whitened.

    4. Every learned quantity is a short cosine series in flow time (``TimeCosines``), as in
       ``PointwiseField``. Unlike ``PointwiseField``, they are *not* smooth functions of position:
       ``W``, ``b`` and ``π`` are constant in x, and all spatial structure comes from ``K``.




    **Why the trace is not exact**

    In ``PointwiseField`` no gate depends on ``v``, the grid map is point by point, and
    :math:`\partial u_{\text{out}} / \partial u_{\text{in}}` is diagonal -- so the divergence
    collapses to :math:`w \sum_p d_p \sum_j \Phi_{pj}^2`.

    Here layer ``l``'s input is layer ``l-1``'s ``tanh`` output, and ``K_l`` couples grid points.
    The Jacobian is a product of ``L`` dense factors
    :math:`D_l (W_l + \Phi \kappa_l w\Phi^{\!\top})` and no longer has a closed-form trace. Hence
    ``velocity_and_trace`` forwards to ``estimated_velocity_and_trace`` (Hutchinson,
    ``trace_samples`` probes).

    An exact trace *is* available at :math:`O(L M^2 G^d)` by propagating the full
    :math:`M \times M` Jacobian instead of probes -- viable at M ≈ 100, not at M ≈ 1000. Not
    implemented.

    ***DANGER DANGER***

    1. **The estimated trace is unbiased in log, but biased in density.**

       - Hutchinson gives an unbiased :math:`\mathrm{Tr}\,J`, so ``log_rn_at`` is unbiased,
       - BUT importance weights exponentiate it:
         :math:`\mathbb{E}[e^{\epsilon}] \neq e^{\mathbb{E}[\epsilon]}`.
       - ``ImportanceCorrection`` efficiency and log-evidence come out biased *upward*, not merely
         noisy.

    2. **``latent_pcn`` is invalid with this field.**

       - Its acceptance ratio needs the exact ``log_rn_at``: a noisy one makes the chain target
         the wrong measure rather than a noisy version of the right one.

    3. **Aliasing.** ``PointwiseField`` wants ``grid_size >= 6 * k_max`` because one ``tanh``
       reaches 3·k_max.

       - Composing L of them reaches further; the harmonic amplitudes decay, so ``2 * 3^L * k_max``
         is pessimistic.
       - Treat ``6 * k_max`` as a floor here, not a rule -- and note that unlike ``PointwiseField``
         there is no warning that fires.

    4. **No residual connections.** ``z_l = tanh(...)`` overwrites rather than adds, so the signal
       passes through L saturating nonlinearities in series.

       - Combined with the zero-init of ``spectral_stack``, layers 2…L start as near-copies of a
         single ``tanh``
       - depth only appears once training has *moved* the weights.

    **Cost**

    ``2L + 2`` grid transforms per evaluation -- one pair per layer for ``K``, plus the initial
    lift and the final projection -- each carrying ``C`` channels, versus ``2`` total for
    ``PointwiseField`` at any depth. Plus one vjp per Hutchinson probe. At M=240, G=1024, C=4, L=3
    that is roughly an order of magnitude on the dominant term.

    Attributes
    ----------
    transform : GridTransform
        Owns Φ, the quadrature weight and the dense/FFT choice.
    cosines : TimeCosines
        Contracted with every ``*_stack`` to give that quantity's value at time t.
    lift : Tensor [C], buffer
        Fixed channel scaling, ``num_active^-1/2`` when whitening, else 1. Not learnable on
        purpose -- see step 1.
    spectral_features : Tensor [num_active, num_spectral_modes]
        cos(π s ρ_k), smooth in log wavenumber so the multipliers do not depend on the truncation.
    pointwise_stack, spectral_stack, bias_stack, project_stack, context_stack : Parameter
        Time-mode-first stacks contracted with ``cosines(t)``; ``spectral_stack`` starts at zero.
    trace_samples : int
        Hutchinson probes per evaluation.
    """

    supports_batched_time = True
    """``time_value`` may carry a leading batch dimension."""

    def __init__(self, basis, num_active, grid_size, num_channels=4, num_layers=3,
                 num_time_modes=4, num_spectral_modes=8, context_dim=0, total_time=1.0,
                 mode_scale=None, init_scale=0.5, transform="auto", time_pair=False,
                 trace_samples=1, dtype=torch.float64):
        super().__init__()

        self.num_active, self.num_layers, self.num_channels = num_active, num_layers, num_channels
        self.total_time, self.mode_scale, self.dtype = total_time, mode_scale, dtype
        self.trace_samples = trace_samples

        self.cosines = TimeCosines(num_time_modes, total_time, dtype, time_pair)
        self.transform = GridTransform(basis, num_active, grid_size, mode=transform, dtype=dtype)

        logs = torch.log1p(basis.laplacian_eigenvalues[:num_active].to(dtype))

        span = (logs.max() - logs.min()).clamp(min=1e-12)
        position = ((logs - logs.min()) / span).clamp(0, 1)

        orders = torch.arange(num_spectral_modes, dtype=dtype, device=logs.device)


        # smooth in log wavenumber, so the multipliers do not depend on where the basis is cut
        self.register_buffer("spectral_features", torch.cos(torch.pi * orders * position[:, None]))
        # a whitened v(x) is O(sqrt(num_active)), so the lift has to undo that or every tanh
        # starts saturated and nothing propagates (the same trap as the kappa init)
        self.register_buffer("lift", torch.full((num_channels,),
                                                1.0 if mode_scale is None else num_active ** -0.5,
                                                dtype=dtype))
    
        shape = (num_layers, num_time_modes, num_channels, num_channels)
    
        self.pointwise_stack = torch.nn.Parameter(
            init_scale / (num_channels * num_time_modes) ** 0.5 * torch.randn(shape, dtype=dtype))
        self.spectral_stack = torch.nn.Parameter(torch.zeros(shape + (num_spectral_modes,), dtype=dtype))
    
        self.bias_stack = torch.nn.Parameter(
            torch.zeros(num_layers, num_time_modes, num_channels, dtype=dtype))
        self.project_stack = torch.nn.Parameter(
            init_scale / (num_channels * num_time_modes) ** 0.5
            * torch.randn(num_time_modes, num_channels, dtype=dtype))
    
        self.context_stack = None
        if context_dim:
            self.context_stack = torch.nn.Parameter(
                init_scale / (context_dim * num_time_modes) ** 0.5
                * torch.randn(num_layers, num_time_modes, num_channels, context_dim, dtype=dtype))


    def velocity(self, coeffs, time_value, context=None):
    
        weights = self.cosines(time_value)
        active = coeffs[..., :self.num_active]
    
        if self.mode_scale is not None:
            active = active / self.mode_scale[:self.num_active]
    
        state = self.transform.to_grid(active).unsqueeze(-1) * self.lift       # [..., points, channels]
        for layer in range(self.num_layers):

            pointwise = torch.tensordot(weights, self.pointwise_stack[layer], dims=1)

            multipliers = (torch.tensordot(weights, self.spectral_stack[layer], dims=1)
                           @ self.spectral_features.T)              # [..., out, in, num_active]
            
            modes = self.transform.from_grid(state.transpose(-1, -2))

            spectral = self.transform.to_grid((multipliers * modes.unsqueeze(-3)).sum(-2))

            biases = torch.tensordot(weights, self.bias_stack[layer], dims=1)
            if context is not None and self.context_stack is not None:
                biases = biases + combine_context(self.context_stack[layer], context, weights)

        
            state = torch.tanh(torch.matmul(state, pointwise.transpose(-1, -2))
                               + spectral.transpose(-1, -2) + biases.unsqueeze(-2))


        projection = (weights @ self.project_stack).unsqueeze(-2)
        update = self.transform.from_grid((state * projection).sum(-1))
    
        if self.mode_scale is not None:
            update = update * self.mode_scale[:self.num_active]
    
        return torch.nn.functional.pad(update, (0, coeffs.shape[-1] - self.num_active))



    def velocity_and_trace(self, coeffs, time_value, context=None):

        return self.estimated_velocity_and_trace(coeffs, time_value, context)




class DeepPointwiseField:
    r"""DeepPointwiseField — L residual pointwise layers; Jacobian stays diagonal in x
    """


    def __init__(self, basis, num_active, grid_size, num_time_modes=4, num_field_modes=32,
                 num_spectral_modes=0, context_dim=0, total_time=1.0, mode_scale=None,
                 init_scale=0.05, transform="auto", time_pair=False, dtype=torch.float64,):

        super().__init__()
        self.num_active, self.num_time_modes = num_active, num_time_modes
        self.total_time, self.mode_scale, self.dtype = total_time, mode_scale, dtype
        self.physical_dim, self.grid_size = basis.physical_dim, grid_size
        self.quadrature_weight = grid_size ** (-basis.physical_dim)

        self.cosines = TimeCosines(num_time_modes, total_time, dtype, time_pair)

        self.transform = GridTransform(basis, num_active, grid_size, mode=transform, dtype=dtype)

        self.quadrature_weight = self.transform.quadrature_weight
        device = basis.laplacian_eigenvalues.device
        values = self.transform.basis_grid

        self.register_buffer("squared_sum", (values ** 2).sum(1))          # S(x), [grid^d]
        self.register_buffer("field_values", values[:, :num_field_modes])  # a, g, c stay dense

        gain_scale = init_scale / num_field_modes ** 0.5
        self.gain_stack = torch.nn.Parameter(
            gain_scale * torch.randn(num_time_modes, num_field_modes, dtype=dtype))
        
        self.slope_stack = torch.nn.Parameter(torch.zeros(num_time_modes, num_field_modes, dtype=dtype))
        self.shift_stack = torch.nn.Parameter(torch.zeros(num_time_modes, num_field_modes, dtype=dtype))

        with torch.no_grad():                       # phi_0 = 1, so this makes g(x) a constant.
            # whitened coefficients are O(1) each, so the whitened field is O(sqrt(num_active))
            self.slope_stack[0, 0] = 1.0 if mode_scale is None else num_active ** -0.5

        self.context_stack = None
        if context_dim:
            self.context_stack = torch.nn.Parameter(
                init_scale / context_dim ** 0.5
                * torch.randn(num_time_modes, num_field_modes, context_dim, dtype=dtype))


        self.kappa_stack = None
        if num_spectral_modes:
            logs = torch.log1p(basis.laplacian_eigenvalues[:num_active].to(dtype))
            span = (logs.max() - logs.min()).clamp(min=1e-12)
            position = ((logs - logs.min()) / span).clamp(0, 1)
            orders = torch.arange(num_spectral_modes, dtype=dtype, device=device)
            self.register_buffer("spectral_features", torch.cos(torch.pi * orders * position[:, None]))
            self.register_buffer("squared_basis", values ** 2)             # [grid^d, num_active]
            # zero, not init_scale * randn: the slope path is normalised by num_active^-1/2 so the
            # pre-activation is O(1), but kappa multiplies whitened coefficients that sum over all
            # num_active modes. At 0.05 * randn that term is O(0.17 * sqrt(2000)) ~ 7.6, tanh is
            # flat, (1 - hidden^2) ~ 0, and neither the trace nor any gradient gets through.
            self.kappa_stack = torch.nn.Parameter(
                torch.zeros(num_time_modes, num_spectral_modes, dtype=dtype))


        if hasattr(basis, "wavenumbers"):
            k_max = int(basis.wavenumbers[:num_active].abs().max().item())
        else:                                   # cosine: lambda_k = (k pi)^2
            k_max = int(round(basis.laplacian_eigenvalues[:num_active].max().sqrt().item() / torch.pi))


        if grid_size < 6 * k_max:
            warnings.warn(f"grid_size {grid_size} < 6 * k_max = {6 * k_max}: tanh harmonics alias back "
                          f"onto the retained modes. The velocity tolerates this; the closed-form trace, "
                          f"log_rn_at and ImportanceCorrection do not.", stacklevel=2)






class SpectralStackField:
    r"""SpectralStackField — L residual layers, each = spectral filter ∘ pointwise nonlinearity
    """

    pass