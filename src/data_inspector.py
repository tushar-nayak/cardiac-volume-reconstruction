#!/usr/bin/env python3
"""
Quick MITEA dataset inspector.

- Finds all image / label pairs under:
    <repo>/cap-mitea/mitea/images
    <repo>/cap-mitea/mitea/labels

- For each scan, prints:
    * subject ID
    * scan ID (filename stem)
    * image shape (D, H, W)  -> D = # slices
    * voxel spacing (dz, dy, dx) in mm
    * segmentation foreground voxel count / fraction

- At the end, prints global stats over all scans.
"""

from collections import defaultdict
import os
from pathlib import Path

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------
# CONFIG: adjust if your path is different
# ---------------------------------------------------------------------
DATA_PATH = Path(
    os.environ.get(
        "CARDIAC_DATA_PATH",
        str(Path(__file__).resolve().parents[1] / "cap-mitea" / "mitea"),
    )
).expanduser()
IMAGES_DIR = DATA_PATH / "images"
LABELS_DIR = DATA_PATH / "labels"


def find_mitea_image_files(data_path: Path):
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected 'images' and 'labels' under {data_path}, "
            f"found images_dir={images_dir.exists()}, labels_dir={labels_dir.exists()}"
        )

    image_files = sorted(images_dir.glob("*.nii*"))
    pairs = []

    for img in image_files:
        stem = img.stem  # e.g. "MITEA_001_scan2_ED.nii"
        # try direct match
        candidates = list(labels_dir.glob(f"{stem}*"))
        if not candidates and stem.endswith(".nii"):
            # handle .nii vs .nii.gz mismatch
            alt = stem[:-4]
            candidates = list(labels_dir.glob(f"{alt}*"))

        if candidates:
            pairs.append((img, candidates[0]))
        else:
            print(f"WARNING: no label found for image {img.name}")

    return pairs


def get_subject_id(img_path: Path) -> str:
    """Extract subject ID from filename.
    Example: MITEA_001_scan2_ED.nii.gz -> MITEA_001
    """
    stem = img_path.stem  # "MITEA_001_scan2_ED.nii"
    if "_scan" in stem:
        return stem.split("_scan")[0]
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def main():
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"  images: {IMAGES_DIR.exists()}, labels: {LABELS_DIR.exists()}")

    pairs = find_mitea_image_files(DATA_PATH)
    print(f"\nFound {len(pairs)} image/label pairs.")

    # group by subject
    subjects = defaultdict(list)
    for img, lab in pairs:
        sid = get_subject_id(img)
        subjects[sid].append((img, lab))

    print(f"Discovered {len(subjects)} subjects.")
    scan_counts = [len(v) for v in subjects.values()]
    print(
        f"  Scans per subject: min={min(scan_counts)}, "
        f"max={max(scan_counts)}, mean={np.mean(scan_counts):.2f}"
    )

    all_D, all_H, all_W = [], [], []
    all_dz, all_dy, all_dx = [], [], []

    print("\nPer-scan details:\n" + "-" * 60)

    for subj_id, scans in sorted(subjects.items()):
        print(f"\nSubject: {subj_id} (num scans = {len(scans)})")
        for img_file, label_file in sorted(scans):
            scan_id = img_file.stem  # includes scan + phase info

            img_nii = nib.load(str(img_file))
            seg_nii = nib.load(str(label_file))

            img_data = img_nii.get_fdata()
            seg_data = seg_nii.get_fdata()

            # shape: (D, H, W)
            D, H, W = img_data.shape
            all_D.append(D)
            all_H.append(H)
            all_W.append(W)

            # voxel spacing (dz, dy, dx) in mm
            img_zooms = img_nii.header.get_zooms()[:3]
            seg_zooms = seg_nii.header.get_zooms()[:3]
            dz, dy, dx = img_zooms
            all_dz.append(dz)
            all_dy.append(dy)
            all_dx.append(dx)

            # segmentation stats
            fg_voxels = np.count_nonzero(seg_data > 0)
            total_voxels = seg_data.size
            fg_frac = fg_voxels / total_voxels if total_voxels > 0 else 0.0

            print(f"  Scan: {scan_id}")
            print(f"    Image file: {img_file.name}")
            print(f"    Label file: {label_file.name}")
            print(f"    Shape (D,H,W): ({D}, {H}, {W})  -> #slices D = {D}")
            print(
                f"    Spacing (dz,dy,dx) [mm]: "
                f"({dz:.3f}, {dy:.3f}, {dx:.3f})"
            )
            if img_zooms != seg_zooms:
                print(f"    WARNING: image and label zooms differ: "
                      f"img={img_zooms}, seg={seg_zooms}")
            print(
                f"    Seg foreground voxels: {fg_voxels} "
                f"({fg_frac*100:.3f}% of volume)"
            )

    # -----------------------------------------------------------------
    # Global summary
    # -----------------------------------------------------------------
    if not all_D:
        print("\nNo valid scans found.")
        return

    def summarize(name, arr):
        arr = np.array(arr, dtype=float)
        print(
            f"  {name}: min={arr.min():.3f}, "
            f"max={arr.max():.3f}, mean={arr.mean():.3f}"
        )

    print("\n" + "=" * 60)
    print("GLOBAL IMAGE SHAPE STATS (over all scans):")
    summarize("D (num slices)", all_D)
    summarize("H (height)", all_H)
    summarize("W (width)", all_W)

    print("\nGLOBAL VOXEL SPACING STATS (from image headers):")
    summarize("dz [mm]", all_dz)
    summarize("dy [mm]", all_dy)
    summarize("dx [mm]", all_dx)


if __name__ == "__main__":
    main()
