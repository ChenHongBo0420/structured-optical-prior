"""Optical layer building blocks."""

from __future__ import annotations

import numpy as np
import neuroptica as neu  # noqa: F401
from neuroptica.layers import OpticalMeshNetworkLayer, StaticMatrix
from neuroptica.component_layers import OpticalMesh, PhaseShifterLayer, MZILayer


class MDCPhaseDiag(OpticalMeshNetworkLayer):
    """Diagonal phase-gating layer implemented by N phase shifters."""

    def __init__(self, N: int, name: str = "MDCPhaseDiag"):
        super().__init__(N, N)
        self.mesh = OpticalMesh(N, [PhaseShifterLayer(N)])
        self.__name__ = name

    def forward_pass(self, X, cache_fields=False, use_partial_vectors=False):
        self.input_prev = X
        self.output_prev = self.mesh.get_transfer_matrix() @ X
        return self.output_prev

    def backward_pass(self, delta, cache_fields=False, use_partial_vectors=False):
        return self.mesh.get_transfer_matrix().T @ delta


class GroupedClementsLayer(OpticalMeshNetworkLayer):
    """Grouped local Clements-like mesh used as the Ug block."""

    def __init__(self, N: int, group_size: int = 4, include_phase_shifter_layer: bool = True,
                 name: str = "GroupedClements"):
        super().__init__(N, N)
        if N % group_size != 0:
            raise ValueError("N must be divisible by group_size.")
        self.N = N
        self.G = group_size
        layers = []
        if include_phase_shifter_layer:
            layers.append(PhaseShifterLayer(N))
        M = self.G
        for col in range(M):
            if col % 2 == 0:
                for b in range(0, N, self.G):
                    idx = list(range(b, b + self.G))
                    layers.append(MZILayer.from_waveguide_indices(N, idx))
            else:
                for b in range(0, N, self.G):
                    idx = list(range(b + 1, b + self.G - 1))
                    if idx:
                        layers.append(MZILayer.from_waveguide_indices(N, idx))
        self.mesh = OpticalMesh(N, layers)
        self.__name__ = name

    def forward_pass(self, X, cache_fields=False, use_partial_vectors=False):
        self.input_prev = X
        self.output_prev = self.mesh.get_transfer_matrix() @ X
        return self.output_prev

    def backward_pass(self, delta, cache_fields=False, use_partial_vectors=False):
        return self.mesh.get_transfer_matrix().T @ delta


def build_spg_layers(N: int, S: np.ndarray, prefix: str):
    """Build S^H diag(exp(i phi)) S layers."""
    S_H = StaticMatrix(S.conj().T)
    S_H.__name__ = f"{prefix}_S_H"
    phase_diag = MDCPhaseDiag(N, name=f"{prefix}_PhaseDiag")
    S_layer = StaticMatrix(S)
    S_layer.__name__ = f"{prefix}_S"
    return [S_H, phase_diag, S_layer]
