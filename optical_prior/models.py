"""Model definitions for the coherent dual-branch optical prior."""

from __future__ import annotations

import numpy as np
import neuroptica as neu
from neuroptica.models import BaseModel
from neuroptica.layers import OpticalMeshNetworkLayer
from .layers import GroupedClementsLayer, build_spg_layers


class AddBeforeDetectModel(BaseModel):
    """Two optical branches combined either coherently or incoherently before readout."""

    def __init__(self, branch1_layers, branch2_layers, tail_layers, combine: str = "coherent"):
        self.branch1_layers = branch1_layers
        self.branch2_layers = branch2_layers
        self.tail_layers = tail_layers
        self.combine = combine
        self.layers = branch1_layers + branch2_layers + tail_layers

    @staticmethod
    def _forward_branch(X, layers, cache_fields=False, use_partial_vectors=False):
        out = X
        for layer in layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                out = layer.forward_pass(out, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
            else:
                out = layer.forward_pass(out)
        return out

    def forward_pass(self, X, cache_fields=False, use_partial_vectors=False):
        self.input_prev = X
        out1 = self._forward_branch(X, self.branch1_layers, cache_fields, use_partial_vectors)
        out2 = self._forward_branch(X, self.branch2_layers, cache_fields, use_partial_vectors)
        self.out_branch1, self.out_branch2 = out1, out2
        if self.combine == "coherent":
            out = out1 + out2
            self.pre_detect = out
            for layer in self.tail_layers:
                out = layer.forward_pass(out)
            self.output_prev = out
            return out
        if self.combine == "incoherent":
            out = np.abs(out1) ** 2 + np.abs(out2) ** 2
            self.pre_detect = out
            for layer in self.tail_layers:
                out = layer.forward_pass(out)
            self.output_prev = out
            return out
        raise ValueError(f"Unknown combine: {self.combine}")

    def backward_pass(self, d_loss, cache_fields=False, use_partial_vectors=False):
        deltas = {}
        back = d_loss
        for layer in reversed(self.tail_layers):
            back = layer.backward_pass(back)
            deltas[layer.__name__] = back

        if self.combine == "coherent":
            d1 = back
            d2 = back
        elif self.combine == "incoherent":
            # Consistent with the previous codebase's Wirtinger-style approximation.
            d1 = 2 * self.out_branch1 * back
            d2 = 2 * self.out_branch2 * back
        else:
            raise ValueError(self.combine)

        for layer in reversed(self.branch1_layers):
            if isinstance(layer, OpticalMeshNetworkLayer):
                d1 = layer.backward_pass(d1, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
            else:
                d1 = layer.backward_pass(d1)
            deltas[layer.__name__] = d1

        for layer in reversed(self.branch2_layers):
            if isinstance(layer, OpticalMeshNetworkLayer):
                d2 = layer.backward_pass(d2, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
            else:
                d2 = layer.backward_pass(d2)
            deltas[layer.__name__] = d2
        return deltas


class SingleBranchModel(BaseModel):
    def __init__(self, branch_layers, tail_layers):
        self.branch_layers = branch_layers
        self.tail_layers = tail_layers
        self.layers = branch_layers + tail_layers

    def forward_pass(self, X, cache_fields=False, use_partial_vectors=False):
        self.input_prev = X
        out = X
        for layer in self.branch_layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                out = layer.forward_pass(out, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
            else:
                out = layer.forward_pass(out)
        self.pre_detect = out
        for layer in self.tail_layers:
            out = layer.forward_pass(out)
        self.output_prev = out
        return out

    def backward_pass(self, d_loss, cache_fields=False, use_partial_vectors=False):
        deltas = {}
        back = d_loss
        for layer in reversed(self.tail_layers):
            back = layer.backward_pass(back)
            deltas[layer.__name__] = back
        for layer in reversed(self.branch_layers):
            if isinstance(layer, OpticalMeshNetworkLayer):
                back = layer.backward_pass(back, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
            else:
                back = layer.backward_pass(back)
            deltas[layer.__name__] = back
        return deltas


def _tail_layers(N: int, out_ports: int, include_abs2: bool = True):
    tail = []
    if include_abs2:
        abs2 = neu.Activation(neu.AbsSquared(N))
        abs2.__name__ = "AbsSquared"
        tail.append(abs2)
    drop = neu.DropMask(N, keep_ports=list(range(out_ports)))
    drop.__name__ = "DropMask"
    tail.append(drop)
    return tail


def build_single_model(N=16, out_ports=10, gsize=4, S=None):
    if S is None:
        S = np.eye(N, dtype=np.complex128)
    ug = GroupedClementsLayer(N, group_size=gsize, name="Ug")
    branch = [ug] + build_spg_layers(N, S, prefix="SingleSPG")
    return SingleBranchModel(branch, _tail_layers(N, out_ports, include_abs2=True))


def build_dual_model(N=16, out_ports=10, gsize=4, S=None, combine: str = "coherent"):
    if S is None:
        S = np.eye(N, dtype=np.complex128)
    ug1 = GroupedClementsLayer(N, group_size=gsize, name="Ug1")
    ug2 = GroupedClementsLayer(N, group_size=gsize, name="Ug2")
    b1 = [ug1] + build_spg_layers(N, S, prefix="Branch1SPG")
    b2 = [ug2] + build_spg_layers(N, S, prefix="Branch2SPG")
    return AddBeforeDetectModel(b1, b2, _tail_layers(N, out_ports, include_abs2=(combine == "coherent")), combine=combine)


def count_params(architecture: str, N: int, gsize: int):
    """Approximate component counts used for reporting and ablation tables."""
    groups = N // gsize
    # Consistent with previous N=16, gsize=4 summaries.
    per_branch_phase_shifters = 2 * N  # Ug phase layer + SPG diag phase layer
    if N == 16 and gsize == 4:
        per_branch_mzi = 12
        per_branch_tunable = 48
    else:
        per_branch_mzi = groups * max(gsize - 1, 1) * max(gsize // 2, 1)
        per_branch_tunable = 2 * per_branch_mzi + N
    mult = 1 if architecture == "single" else 2
    return {
        "mzi": int(per_branch_mzi * mult),
        "phase_shifters": int(per_branch_phase_shifters * mult),
        "tunable_phases": int(per_branch_tunable * mult),
    }
