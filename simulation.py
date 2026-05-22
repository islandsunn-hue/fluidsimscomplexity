"""
Two-fluid mixing simulation: alternating sine-flow map + molecular diffusion.

Physics
-------
At each time step the concentration field C[i,j] ∈ [0,1] is:
  1. Advected by an alternating sinusoidal shear (semi-Lagrangian, periodic BC).
     Even steps:  horizontal shear  Δj = A·sin(2π·i/N)
     Odd  steps:  vertical   shear  Δi = A·sin(2π·j/N)
  2. Diffused with an explicit finite-difference Laplacian.

The alternating sine map is a textbook model for Lagrangian chaos — it creates
the exponential stretching of material lines that produces the thin lamellae
visible during intermediate mixing, before diffusion smooths them away.

Boltzmann entropy
-----------------
For an ideal binary mixture on a lattice, the mixing entropy per site is

    s_i = −[ x_i ln x_i + (1−x_i) ln(1−x_i) ]          (nats)

Summed and normalised by N·ln 2 this gives S ∈ [0,1]:
  S = 0  →  fluids completely separated (pure sites everywhere)
  S = 1  →  every site is exactly 50/50 (maximum disorder)

Because we work with a continuous concentration field, this is exact within
the lattice model — no pixel-colour approximation is required.
"""

import numpy as np
from scipy.ndimage import map_coordinates

_LN2 = np.log(2.0)
_EPS = 1e-10


class Simulation:
    def __init__(self, grid_size: int = 64, diffusion: float = 0.08,
                 flow_speed: float = 2.0, seed: int = 42):
        """
        Parameters
        ----------
        grid_size  : lattice side length N  (N×N sites)
        diffusion  : diffusion coefficient D  (must be < 0.25 for stability)
        flow_speed : peak displacement A in grid units per shear step
        seed       : RNG seed for the interface perturbation
        """
        self.N = grid_size
        self.D = min(diffusion, 0.24)   # clamp for explicit-method stability
        self.A = flow_speed
        self.t = 0

        # Initial condition: top half = fluid 1 (C=1), bottom = fluid 2 (C=0)
        # A small random perturbation seeds the Kelvin-Helmholtz-like instability.
        rng = np.random.default_rng(seed)
        C = np.zeros((self.N, self.N), dtype=float)
        C[: self.N // 2, :] = 1.0
        C += 0.02 * rng.standard_normal((self.N, self.N))
        self.C = np.clip(C, 0.0, 1.0)

    # ── Simulation steps ─────────────────────────────────────────────────────

    def step(self):
        self._advect()
        self._diffuse()
        self.t += 1

    def _advect(self):
        N = self.N
        ii, jj = np.mgrid[0:N, 0:N].astype(float)
        if self.t % 2 == 0:
            # Horizontal shear: column displacement depends on row
            di = np.zeros((N, N))
            dj = self.A * np.sin(2.0 * np.pi * ii / N)
        else:
            # Vertical shear: row displacement depends on column
            di = self.A * np.sin(2.0 * np.pi * jj / N)
            dj = np.zeros((N, N))

        # Back-trace source coordinates (periodic)
        src_i = (ii - di) % N
        src_j = (jj - dj) % N
        self.C = map_coordinates(self.C, [src_i, src_j], order=1, mode='wrap')
        np.clip(self.C, 0.0, 1.0, out=self.C)

    def _diffuse(self):
        lap = (
            np.roll(self.C,  1, axis=0) + np.roll(self.C, -1, axis=0) +
            np.roll(self.C,  1, axis=1) + np.roll(self.C, -1, axis=1) -
            4.0 * self.C
        )
        self.C = np.clip(self.C + self.D * lap, 0.0, 1.0)

    # ── Measurements ─────────────────────────────────────────────────────────

    def boltzmann_entropy(self) -> float:
        """
        Exact normalised Boltzmann mixing entropy S ∈ [0, 1].

            S = ⟨ −[x ln x + (1−x) ln(1−x)] ⟩_sites / ln 2

        Computed directly from the concentration field — no colour
        approximation involved.
        """
        x = self.C
        s = -(x * np.log(x + _EPS) + (1.0 - x) * np.log(1.0 - x + _EPS))
        return float(s.mean() / _LN2)

    def to_rgb(self) -> np.ndarray:
        """Return (N, N, 3) uint8 array. Fluid 1 = blue, fluid 2 = red."""
        c = self.C
        rgb = np.stack([
            220.0 - 190.0 * c,   # R: 220 (fluid 2) → 30 (fluid 1)
            100.0 -  20.0 * c,   # G: nearly flat
             30.0 + 190.0 * c,   # B:  30 (fluid 2) → 220 (fluid 1)
        ], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)
