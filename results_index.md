# Results Index

This repository tracks a small set of text-based result snapshots so the latest experiment outputs are visible in git without checking in large binary volumes.

For a browser-friendly landing page, switch to the `gh-pages` branch or the published GitHub Pages site.

## Reported Headline Results

The final write-up in [report.md](report.md) reports the following headline numbers on the MITEA healthy end-diastole setting:

| Method | 2D Dice | 2D IoU | 3D Dice | 3D IoU |
| --- | --- | --- | --- | --- |
| Mixed stratified ED-healthy | 0.9458 | 0.9007 | 0.8491 | 0.7422 |
| Meta / Reptile, after refinement | 0.9540 | 0.9143 | 0.8638 | 0.7649 |
| Mixed, no stratifiers | 0.9505 | 0.9085 | 0.8643 | 0.7658 |

The best full-volume overlap in the saved summaries is the mixed run without stratifiers.

The tracked HTML viewers and metadata below correspond to the documented reconstruction runs that support those results.

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

## Featured Visualization Set

The published GitHub Pages site highlights the following viewer files:

- `checkpoints/html_visualizations/MITEA_107_scan1_ED.nii_meta_3d_mesh.html`
- `checkpoints/html_visualizations/MITEA_107_scan1_ED.nii_mixed_3d_mesh.html`
- `checkpoints3/html_visualizations3/MITEA_107_scan1_ES.nii_mixed_refined_3d_mesh.html`
- `3d_comparison_viewers_v2/MITEA_001_scan1_ED.nii_comparison.html`
- `3d_comparison_viewers_v2/MITEA_001_scan1_ED.nii_overlay.html`

## Not Tracked

The corresponding `.nii.gz` occupancy grids and other large binary artifacts are intentionally not versioned.
