# Reproducibility

This repository is now split into:

- `main` for source code and documentation
- `gh-pages` for static HTML viewers and presentation artifacts

The results in the report were produced from the source code in `main`. The HTML outputs are only for inspection.

## Data Location

The code expects the MITEA dataset under:

```text
<data_path>/
  images/
  labels/
```

By default, the canonical scripts use:

```bash
<repo>/cap-mitea/mitea
```

If your dataset lives elsewhere, override it without editing code:

```bash
export CARDIAC_DATA_PATH=/path/to/mitea
```

## Recommended Run Order

1. Verify the dataset:

```bash
python -m cardiac_reconstruction inspect
```

2. Run the main reconstruction baseline:

```bash
python -m cardiac_reconstruction baseline
```

3. Run the sparse reconstruction / viewer pipeline:

```bash
python -m cardiac_reconstruction sparse
python -m cardiac_reconstruction viewer
```

4. Run the full staged workflow if you want every saved artifact:

```bash
python -m cardiac_reconstruction all
```

## Reported Headline Results

The saved summaries in this repository report:

- Mixed stratified ED-healthy: 3D Dice `0.8491`, 3D IoU `0.7422`
- Meta / Reptile after refinement: 3D Dice `0.8638`, 3D IoU `0.7649`
- Mixed, no stratifiers: 3D Dice `0.8643`, 3D IoU `0.7658`

The best full-volume overlap is the mixed run without stratifiers.

## Notes

- The code is experimental, but the canonical entrypoint is `python -m cardiac_reconstruction ...`.
- The tracked HTML files were moved to `gh-pages` so `main` stays source-focused.
- If a command fails, first confirm `CARDIAC_DATA_PATH` points at a valid MITEA directory.
