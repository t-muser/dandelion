"""Download the benchmark datasets into ``$DANDELION_DATA`` (default ``./data``).

The six spherical datasets come from the ``sada-group`` HuggingFace org; each
lands in ``<root>/<name>/`` with the same layout as the repo
(``data/{train,valid,test}/run_*.h5``, ``stats.yaml``, ``grid.h5`` for
global-ocean-3D), which is exactly what ``configs/data/<name>.yaml`` expects.
``planetswe`` comes from The Well and lands in ``<root>/the_well/``.

Examples:
    uv run scripts/download_data.py shock-caps
    uv run scripts/download_data.py active-turbulence cahn-hilliard --splits test
    uv run scripts/download_data.py all --root /scratch/data
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

HF_ORG = "sada-group"
SIZES = {
    "shock-caps": "63 GB",
    "cahn-hilliard": "15 GB",
    "active-turbulence": "18 GB",
    "barotropic-jets": "210 GB",
    "global-ocean-3D": "248 GB",
    "dry-atmosphere-3D": "667 GB",
    "planetswe": "187 GB",
}
SPLITS = ("train", "valid", "test")


def download_hf(name: str, root: Path, splits: list[str]) -> None:
    from huggingface_hub import snapshot_download

    target = root / name
    patterns = [f"data/{s}/*" for s in splits] + ["stats.yaml", "grid.h5", "README.md"]
    print(f"[{name}] sada-group/{name} -> {target} (splits: {', '.join(splits)})")
    snapshot_download(
        f"{HF_ORG}/{name}",
        repo_type="dataset",
        local_dir=target,
        allow_patterns=patterns,
    )


def download_planetswe(root: Path, splits: list[str]) -> None:
    from the_well.utils.download import well_download

    base = root / "the_well"
    print(f"[planetswe] The Well -> {base} (splits: {', '.join(splits)})")
    for split in splits:
        well_download(base_path=str(base), dataset="planetswe", split=split)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "datasets", nargs="+", choices=[*SIZES, "all"], metavar="DATASET",
        help=f"one or more of: {', '.join(SIZES)}, or 'all'",
    )
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get("DANDELION_DATA", "data")),
        help="download root (default: $DANDELION_DATA or ./data)",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
        help="splits to download (default: all)",
    )
    args = parser.parse_args()

    names = list(SIZES) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    print("Full sizes: " + ", ".join(f"{n} ~{SIZES[n]}" for n in names))
    for name in names:
        if name == "planetswe":
            download_planetswe(args.root, args.splits)
        else:
            download_hf(name, args.root, args.splits)


if __name__ == "__main__":
    main()
