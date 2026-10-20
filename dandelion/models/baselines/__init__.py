"""Baselines from the paper: FNO, SFNO, local S²/R² transformers, GAOT.

The Local-R transformer imports `natten` on construction. GAOT needs
`torch-scatter` and `rotary-embedding-torch` plus the upstream model code from
`scripts/fetch_gaot.sh`; without them a stub raises ImportError on instantiation.
"""

from dandelion.models.baselines.sphere_models import (
    Fno,
    LocalRTransformer,
    LocalS2Transformer,
    Sfno,
)

try:
    from dandelion.models.baselines.gaot import GAOTWrapper
except ImportError as _e:  # pragma: no cover - runtime optional
    _gaot_import_error = _e

    class GAOTWrapper:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GAOTWrapper requires `uv sync --extra baselines` and "
                "`bash scripts/fetch_gaot.sh`. "
                f"Original error: {_gaot_import_error}"
            )

__all__ = ["Fno", "Sfno", "LocalS2Transformer", "LocalRTransformer", "GAOTWrapper"]
