# Dandelion

Code for **Dandelion: A Spherical Flower for Neural Simulation of Planetary Dynamics**.


## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/t-muser/dandelion.git
cd dandelion
uv sync                      # Dandelion, Flower2D, FNO, SFNO
uv sync --extra baselines    # + Local-R/S² transformers and GAOT dependencies
```

**GAOT.** For license reasons, we provide a script to download GAOT instead of including the source code. Separately install a version of `torch-scatter` from the PyG wheel index that matches your torch/CUDA build:

```bash
bash scripts/fetch_gaot.sh
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu130.html
```

**Local transformers on GPU.** The PyPI `torch-harmonics` wheel ships without
CUDA kernels for the DISCO convolutions used by the Local-S² transformer. Rebuild
it from source on a GPU machine with `nvcc` matching torch's CUDA runtime:

```bash
uv pip install setuptools wheel setuptools-scm
git clone --depth 1 --branch v0.9.0 https://github.com/NVIDIA/torch-harmonics.git third_party/torch-harmonics
FORCE_CUDA_EXTENSION=1 uv pip install --no-deps --no-build-isolation --reinstall third_party/torch-harmonics
```


## Data

Datasets generated for the paper are available on HuggingFace
([collection](https://huggingface.co/collections/sada-group/spherical-pde-datasets-6ad5da71b329cace0daf04d9));
`planetswe` via [The Well](https://polymathic-ai.org/the_well/).

| Dataset | Grid | Runs (train/valid/test) | Size | Source |
| --- | --- | --- | --- | --- |
| `barotropic-jets` | 256×512 | 768 / 96 / 96 | 210 GB | [sada-group/barotropic-jets](https://huggingface.co/datasets/sada-group/barotropic-jets) |
| `active-turbulence` | 128×256 | 608 / 76 / 76 | 18 GB | [sada-group/active-turbulence](https://huggingface.co/datasets/sada-group/active-turbulence) |
| `cahn-hilliard` | 256×512 | 461 / 58 / 57 | 15 GB | [sada-group/cahn-hilliard](https://huggingface.co/datasets/sada-group/cahn-hilliard) |
| `shock-caps` | 256×512 | 400 / 50 / 50 | 63 GB | [sada-group/shock-caps](https://huggingface.co/datasets/sada-group/shock-caps) |
| `dry-atmosphere-3D` | 8×144×288 | 130 / 16 / 16 | 667 GB | [sada-group/dry-atmosphere-3D](https://huggingface.co/datasets/sada-group/dry-atmosphere-3D) |
| `global-ocean-3D` | cs32×15 → 64×128 | 194 / 24 / 25 | 248 GB | [sada-group/global-ocean-3D](https://huggingface.co/datasets/sada-group/global-ocean-3D) |
| `planetswe` | 256×512 | (The Well) | 187 GB | [The Well](https://polymathic-ai.org/the_well/datasets/planetswe/) |

Download into `$DANDELION_DATA` (default `./data`), whole datasets or individual splits:

```bash
uv run scripts/download_data.py active-turbulence
uv run scripts/download_data.py shock-caps --splits test
DANDELION_DATA=/scratch/data uv run scripts/download_data.py all
```

## Training and testing

A run combines a data config, a model config and the training recipe:

```bash
uv run python -m dandelion.train \
    --data configs/data/active-turbulence.yaml \
    --model configs/models/dandelion.yaml \
    --train configs/train.yaml \
    optimizer.lr=5e-4 data.batch_size=42
```

Multi-GPU training uses DDP via `torchrun`. `data.batch_size` is per GPU:

```bash
uv run torchrun --standalone --nproc_per_node=4 -m dandelion.train \
    --data configs/data/active-turbulence.yaml \
    --model configs/models/dandelion.yaml \
    --train configs/train.yaml \
    optimizer.lr=5e-4 data.batch_size=42
```

Runs are written to `runs/<dataset>-paper-<Model>-<lr>/<n>/`. `scripts/train.sbatch` is a SLURM template that trains and then tests.

**Testing.** Re-run the same command with `test_mode=true auto_resume=true`. This
picks the latest run folder for that configuration, loads its final checkpoint
(`checkpoints/recent.pt`) and reports next-step and 20-step rollout metrics on
the test split. To test a specific folder, use `folder_override=runs/.../0`
instead of `auto_resume=true`.

```bash
uv run python -m dandelion.train \
    --data configs/data/active-turbulence.yaml \
    --model configs/models/dandelion.yaml \
    --train configs/train.yaml \
    optimizer.lr=5e-4 data.batch_size=42 test_mode=true auto_resume=true
```


## Acknowledgments

### Model and Code
```
@misc{muser2026dandelionsphericalflowerneural,
      title={Dandelion: A Spherical Flower for Neural Simulation of Planetary Dynamics}, 
      author={Till Muser and Giovanni Abati and Ivan Dokmanić},
      year={2026},
      eprint={2608.27521},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.27521}, 
}
```
If you use this repo, please cite the appropriate publications:
- If you use Dandelion or any of the six datasets generated for this paper, please cite our paper using the bibtex above.
- If you use `planetswe` please cite [The Well](https://polymathic-ai.org/the_well/#citation) and [the paper associated with the dataset](https://arxiv.org/abs/2306.10619).
- If you use any of the other models please cite the appropriate publication, as listed in our paper.
