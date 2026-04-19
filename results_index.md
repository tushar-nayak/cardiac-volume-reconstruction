# Results Index

This repository tracks a small set of text-based result snapshots so the latest experiment outputs are visible in git without checking in large binary volumes.

## Sparse Reconstruction Metadata

Files in `sparse_reconstruction_results/` contain per-subject reconstruction metadata for the current tracked runs:

- `sparse_reconstruction_results/MITEA_001_scan1_ED.nii_metadata.json`
- `sparse_reconstruction_results/MITEA_001_scan1_ES.nii_metadata.json`
- `sparse_reconstruction_results/MITEA_001_scan2_ED.nii_metadata.json`
- `sparse_reconstruction_results/MITEA_001_scan2_ES.nii_metadata.json`
- `sparse_reconstruction_results/MITEA_002_scan1_ED.nii_metadata.json`

Each metadata file stores:

- subject ID
- occupancy grid shape and summary statistics
- mesh counts when available
- reconstruction config used for the run

## 3D Comparison Viewers

Files in `3d_comparison_viewers_v2/` are interactive HTML viewers for the same tracked runs:

- `3d_comparison_viewers_v2/MITEA_001_scan1_ED.nii_comparison.html`
- `3d_comparison_viewers_v2/MITEA_001_scan1_ED.nii_overlay.html`
- `3d_comparison_viewers_v2/MITEA_001_scan1_ES.nii_comparison.html`
- `3d_comparison_viewers_v2/MITEA_001_scan1_ES.nii_overlay.html`
- `3d_comparison_viewers_v2/MITEA_001_scan2_ED.nii_comparison.html`
- `3d_comparison_viewers_v2/MITEA_001_scan2_ED.nii_overlay.html`
- `3d_comparison_viewers_v2/MITEA_001_scan2_ES.nii_comparison.html`
- `3d_comparison_viewers_v2/MITEA_001_scan2_ES.nii_overlay.html`
- `3d_comparison_viewers_v2/MITEA_002_scan1_ED.nii_comparison.html`
- `3d_comparison_viewers_v2/MITEA_002_scan1_ED.nii_overlay.html`

Open the HTML files in a browser to inspect the reconstructed surfaces and overlays.

## Not Tracked

The corresponding `.nii.gz` occupancy grids and other large binary artifacts are intentionally not versioned.
