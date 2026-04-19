# Cardiac Volume Reconstruction

End-to-end experiments for reconstructing 3D cardiac shape from sparse 2D views using implicit neural representations. The repository contains a working prototype pipeline, ablation runs, sparse-view reconstruction, and interactive 3D comparison viewers built around the MITEA dataset.

Start with:

- [Setup](#setup)
- [Recommended commands](#recommended-commands)
- [Dataset layout](#dataset-layout)
- [Usage guide](docs/USAGE.md)

## What’s here

- `src/minimal_starter_5.py` is the main single-subject reconstruction pipeline and the shared utility module used by most other scripts.
- `src/run_all.py` runs the main stages sequentially.
- `src/ablation_studies_7.py` runs a small, self-contained ablation sweep.
- `src/sparse_reconstruction_2.py` performs sparse-view reconstruction and saves mesh/grid outputs.
- `src/complete_pipeline.py` combines sparse reconstruction and 3D viewer generation.
- `src/viewer_3d_reconstruction_2.py` builds interactive HTML viewers for existing reconstructions.
- `src/data_inspector.py` checks the dataset layout and prints scan statistics.
- `old/` contains earlier iterations and implementation notes.

## Requirements

The code is written for Python 3.11+ and uses PyTorch with GPU acceleration when available.

Core dependencies used across the active scripts:

- `torch`
- `numpy`
- `matplotlib`
- `nibabel`
- `scipy`
- `plotly`
- `scikit-image`

Optional dependencies used by some historical or experimental scripts:

- `wandb`
- `trimesh`
- `pandas`
- `seaborn`
- `h5py`
- `pytorch3d`

## Setup

If you prefer a pinned dependency file, use `requirements.txt` for pip or `environment.yml` for Conda.

```bash
conda env create -f environment.yml
conda activate cardiac-3d
```

If you prefer pip, create an environment first and then run `pip install -r requirements.txt`.

If you need a specific CUDA build of PyTorch, install that build first and then install the rest of the dependencies.

If you plan to use scripts that rely on `pytorch3d`, install it separately following the official PyTorch3D instructions for your CUDA / PyTorch version.

## Dataset layout

The active code expects the MITEA dataset to be organized as:

```text
<data_path>/
  images/
    *.nii or *.nii.gz
  labels/
    *.nii or *.nii.gz
```

By default, `src/minimal_starter_5.py` points to:

```python
Path("/home/sofa/host_dir/cap-mitea/mitea")
```

If your dataset lives elsewhere, update `CONFIG['data_path']` in `src/minimal_starter_5.py`. Most other scripts import that same config.

## Recommended commands

1. Inspect the dataset before running anything else.

```bash
python src/data_inspector.py
```

2. Run the main reconstruction pipeline.

```bash
python src/minimal_starter_5.py
```

3. Run the full staged workflow.

```bash
python src/run_all.py
```

`src/run_all.py` will ask for confirmation and then run the staged experiments in sequence.

## Output directories

The repository creates several on-disk result folders:

- `checkpoints/`
- `ablation_results/`
- `sparse_reconstruction_results/`
- `3d_comparison_viewers/`
- `3d_comparison_viewers_v2/`
- `pipeline_results/`
- `wandb/` if Weights & Biases logging is enabled

These paths are ignored by git.

## Versioned Results

Text-based result snapshots from the current runs are kept in git so the documented outputs stay visible:

- `sparse_reconstruction_results/*.json` for per-subject reconstruction metadata
- `3d_comparison_viewers_v2/*.html` for interactive comparison viewers

The corresponding NIfTI volumes remain untracked because they are large binary artifacts.

## Notes on the codebase

- The project went through several iterations; the active scripts in `src/` are the ones to use first.
- `old/` preserves earlier versions and implementation notes, which are useful for archaeology but not the recommended execution path.
- Files with suffixes like `_fixed`, `_optimized`, `_multicore`, or `FINAL_*` are alternative experiment variants and not the canonical starting point.
- The code is optimized for experimentation, not for a polished research release. Expect hard-coded paths and dataset assumptions in some scripts.
- See [docs/USAGE.md](docs/USAGE.md) for a more complete script-by-script usage guide.

## Troubleshooting

- If a script says no subjects were found, check that your dataset has `images/` and `labels/` folders and that filenames match by stem.
- If you get CUDA errors, the code will fall back to CPU where possible, but runs will be much slower.
- If `plotly` HTML viewers are generated but do not open correctly, try opening the `.html` files directly in a browser rather than through an IDE preview.
- If you change the dataset location, update the config in `src/minimal_starter_5.py` first.

## License

MIT License. See [LICENSE](LICENSE).
