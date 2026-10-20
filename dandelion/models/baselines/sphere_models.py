"""Hydra-friendly adapters for the torch-harmonics reference architectures.

The wrappers in this codefile wrap the call signatures to match the
Dandelion signature. Fno is derived from SphericalFourierNeuralOperator by
swapping the SHTs for 2D FFT wrappers.
"""
from __future__ import annotations

import torch
from torch_harmonics.examples.models import (
    SphericalFourierNeuralOperator,
    SphericalTransformer,
)
from torch_harmonics.examples.models._layers import RealFFT2, InverseRealFFT2

from dandelion.models.baselines.transformer import Transformer


def _tuplify_shapes(kwargs):
    for k in ("encoder_kernel_shape", "attn_kernel_shape"):
        if k in kwargs and isinstance(kwargs[k], list):
            kwargs[k] = tuple(kwargs[k])
    return kwargs


class LocalS2Transformer(SphericalTransformer):
    """SphericalTransformer with local NeighborhoodAttentionS2."""

    def __init__(self, inp_shape, inp_chans, out_chans, **kwargs):
        kwargs.setdefault("attention_mode", "neighborhood")
        super().__init__(
            img_size=tuple(inp_shape),
            in_chans=inp_chans,
            out_chans=out_chans,
            **_tuplify_shapes(kwargs),
        )


def _enable_natten_flex_compile():
    # On envs without a prebuilt libnatten (e.g. torch 2.11+cu130), NATTEN
    # falls back to its flex backend. Without torch.compile, flex_attention
    # materialises the full BHN^2 scores matrix and OOMs; with it, NATTEN
    # emits a fused block-sparse kernel. NeighborhoodAttention2D doesn't
    # expose `torch_compile`, so we wrap the symbol it imported by name.
    import natten
    import natten.modules
    from natten.functional import neighborhood_attention_generic as _orig

    natten.allow_flex_compile(mode=True, backprop=True)

    if getattr(_orig, "_dandelion_compiled_wrap", False):
        return

    def _wrapped(q, k, v, *args, **kwargs):
        kwargs.setdefault("torch_compile", True)
        return _orig(q, k, v, *args, **kwargs)

    _wrapped._dandelion_compiled_wrap = True
    natten.modules.neighborhood_attention_generic = _wrapped


class LocalRTransformer(Transformer):
    """Lat/lon transformer with NATTEN NeighborhoodAttention2D (rectangular local attn).

    The forward pass runs under bf16 autocast: NATTEN's compiled flex
    backend (the only one available without libnatten) rejects fp32.
    """

    def __init__(self, inp_shape, inp_chans, out_chans, **kwargs):
        _enable_natten_flex_compile()

        kwargs.setdefault("attention_mode", "neighborhood")
        super().__init__(
            img_size=tuple(inp_shape),
            in_chans=inp_chans,
            out_chans=out_chans,
            **_tuplify_shapes(kwargs),
        )

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = super().forward(x)
        return out.to(x.dtype)


class Sfno(SphericalFourierNeuralOperator):
    """SphericalFourierNeuralOperator with the project's standard kwargs."""

    def __init__(self, inp_shape, inp_chans, out_chans, **kwargs):
        super().__init__(
            img_size=tuple(inp_shape),
            in_chans=inp_chans,
            out_chans=out_chans,
            **kwargs,
        )


class Fno(SphericalFourierNeuralOperator):
    """FNO = SFNO with SHTs swapped for 2D FFTs on the periodic domain."""

    def __init__(self, inp_shape, inp_chans, out_chans, **kwargs):
        import torch_harmonics.examples.models.sfno as _sfno_mod

        # RealFFT2 takes (nlat, nlon, lmax, mmax); SFNO calls its forward
        # transform with a `grid` kwarg which we silently swallow here.
        class _FwdFFT(RealFFT2):
            def __init__(self, nlat, nlon, lmax=None, mmax=None, grid=None):
                super().__init__(nlat, nlon, lmax=lmax, mmax=mmax)

        class _InvFFT(InverseRealFFT2):
            def __init__(self, nlat, nlon, lmax=None, mmax=None, grid=None):
                super().__init__(nlat, nlon, lmax=lmax, mmax=mmax)

        orig_f, orig_i = _sfno_mod.RealSHT, _sfno_mod.InverseRealSHT
        _sfno_mod.RealSHT, _sfno_mod.InverseRealSHT = _FwdFFT, _InvFFT
        try:
            super().__init__(
                img_size=tuple(inp_shape),
                in_chans=inp_chans,
                out_chans=out_chans,
                **kwargs,
            )
        finally:
            _sfno_mod.RealSHT, _sfno_mod.InverseRealSHT = orig_f, orig_i


__all__ = ["LocalS2Transformer", "LocalRTransformer", "Sfno", "Fno"]
