# Usage Guide

This repository is organized around a few runnable scripts in `src/`. The code is experimental, but there is a clear default path for getting started.

## 1. Set up the environment

If you use Conda:

```bash
conda env create -f environment.yml
conda activate cardiac-3d
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install that first and then install the rest of the packages.
The Conda file uses the `pytorch` channel for the core torch packages.

## 2. Prepare the data

The active scripts expect:

```text
<data_path>/
  images/
    *.nii or *.nii.gz
  labels/
    *.nii or *.nii.gz
```

The default path used by `src/minimal_starter_5.py` is:

```python
Path("/home/sofa/host_dir/cap-mitea/mitea")
```

If your dataset lives elsewhere, change `CONFIG['data_path']` in `src/minimal_starter_5.py`.

## 3. Run the scripts

### `python src/data_inspector.py`

Use this first. It verifies the image/label structure and prints:

- number of scans and subjects
- volume shapes
- voxel spacing
- segmentation foreground statistics

### `python src/minimal_starter_5.py`

This is the main entry point for a single reconstruction run. It:

- loads image/label pairs from the dataset
- extracts synthetic 2D cardiac views
- optimizes an implicit neural representation
- reports Dice and IoU

Typical defaults:

- `num_views = 3`
- `image_size = 256`
- `hidden_dim = 64`
- `num_inr_layers = 4`
- `learning_rate = 1e-4`
- `num_optimization_steps = 1500`

### `python src/run_all.py`

This is the highest-level orchestrator in the repository. It runs:

- main training
- ablation studies
- sparse reconstruction
- integrated pipeline evaluation

It prompts before execution because it can take a long time.

### `python src/ablation_studies_7.py`

Runs a compact ablation sweep over:

- `num_views`
- `hidden_dim`
- `num_inr_layers`
- `learning_rate`

Results are written to `ablation_results/`.

### `python src/sparse_reconstruction_2.py`

Runs sparse-view reconstruction and writes:

- occupancy grids as NIfTI files
- vertices and faces as `.npy`
- a 12-panel visualization image
- metadata JSON

Results are written to `sparse_reconstruction_results/`.

### `python src/complete_pipeline.py`

Runs sparse reconstruction and then generates interactive 3D comparison viewers. Use this when you want both reconstruction outputs and browser-based visualization in one pass.

Outputs go to:

- `sparse_reconstruction_results/`
- `3d_comparison_viewers_v2/`

### `python src/viewer_3d_reconstruction_2.py`

Builds HTML viewers from existing reconstruction outputs and computes comparison metrics. Use this after reconstruction if you only want visualization.

Outputs go to:

- `3d_comparison_viewers/`
- `comparison_metrics.json`

### `python src/integrated_pipeline.py`

Runs a broader multi-stage experiment suite and writes summary JSON into `pipeline_results/`.

## 4. Output locations

Common output directories:

- `checkpoints/`
- `ablation_results/`
- `sparse_reconstruction_results/`
- `3d_comparison_viewers/`
- `3d_comparison_viewers_v2/`
- `pipeline_results/`
- `wandb/` if W&B logging is enabled

## 5. Practical workflow

Recommended order:

1. Run `src/data_inspector.py` to confirm the dataset is readable.
2. Run `src/minimal_starter_5.py` to validate the main reconstruction loop.
3. Run `src/sparse_reconstruction_2.py` or `src/complete_pipeline.py` to produce meshes and viewer files.
4. Run `src/ablation_studies_7.py` only after the baseline is working.
5. Use `src/run_all.py` when you want the whole sequence.

## 6. Notes

- Most scripts import configuration from `src/minimal_starter_5.py`.
- Some files in `src/` are alternate experiment branches and not the main path.
- Files in `old/` are preserved for reference and are not the primary workflow.

## 7. Common issues

- No subjects found: verify the dataset has `images/` and `labels/` folders and matching filename stems.
- Slow execution: the code is GPU-friendly but will still run on CPU if CUDA is unavailable.
- Viewer not opening: open the generated `.html` file directly in a browser.
- Path mismatch: update `CONFIG['data_path']` in `src/minimal_starter_5.py`.
