# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a research repository, not the Ultralytics library itself. It trains and evaluates Ultralytics YOLO11 / YOLO26 instance-segmentation models to detect fruit (apples, peaches, avocados, pears, oranges) and studies how detection/segmentation quality degrades with occlusion (visibility %), comparing against Mask2Former. The work is written up as "Occlusion-Aware Fruit Instance Segmentation for Agricultural Robotics: A Comparative Study of YOLO11, YOLO26, and Mask2Former" for the ARTIIS 2026 venue — manuscript source lives in `paper/` (see below); `referencias/` holds third-party background papers (Mask2Former, etc.) and the blank Springer template, not the manuscript itself.

The repo is a Google-Drive-synced folder used as a git repo: raw datasets, label projects, trained weights, and reference PDFs are gitignored (kept only in Drive/local disk); notebooks, conversion scripts, and generated figures/CSVs for the paper are versioned.

## Data & experiment pipeline

The end-to-end flow, in order:

1. **Label in MATLAB Image Labeler** — projects `ImageLabelingProject/` and `FruitLabelingProject_v2/` (`.prj` + `GroundTruthProject/*.mat`, gitignored). Produces a `groundTruth` object saved as a `.mat`.
2. **Flatten the MATLAB groundTruth** — `export_gtruth_for_python.m` (run in MATLAB) converts the `groundTruth` object into a flat, SciPy-readable `.mat` with `imageFiles` (cellstr), `labelNames` (cellstr), `labelPolys` (N×C cell of Nx2 pixel polygons). Handles `polyshape`, `images.roi.Polygon`, and struct/ROI variants.
3. **Filter empty rows** — `filter_gtruth_flat_no_empty.py --in_mat --out_mat` drops images that have no polygon in any class, producing e.g. `gTruth_py_flat_filtered.mat`.
4. **Convert to YOLO segmentation format** — `gtruth_flat_to_yolo.py --mat --labels_out --yaml_out [--dataset_root --train_rel --val_rel]` normalizes polygon pixel coords to `0..1`, writes one `.txt` label per image (class-id + flattened polygon), and emits a `data.yaml` for Ultralytics.
5. **(Optional) merge in an external Roboflow-style dataset** — `filter_remap_external_yolo.py --external_root --out_root [--tag --drop_empty]` reads an external dataset's `data.yaml`, remaps only its `green_apple`/`red_apple` classes onto this project's `apple_green`(0)/`apple_red`(1) ids, copies/renames files with a tag prefix to avoid collisions, and writes a merged `data.yaml`.
6. **Train / validate / benchmark** — done interactively from the root notebooks via `from ultralytics import YOLO`. See `Manzana/yolo_reduced/data.yaml` for the live 6-class dataset config (`apple_green, apple_red, peach, avocado, pear, orange`), path `Manzana/yolo_reduced`, splits `images/train` / `images/val`.
7. **Occlusion / visibility analysis** — validation is re-run against per-visibility-bucket dataset YAMLs (`validation_by_visibility/data_val_visibility_{25,50,70,75,100}pct.yaml`, and the `epoch_visibility_analysis/val_*pct.yaml` variants) to measure how mAP/recall change as fruit visibility decreases; results land in the CSV/XLSX files alongside those YAMLs and in `epoch_visibility_analysis/*.csv`.
8. **Figures for the paper** — comparison plots/tables are generated into `figures_quant/`, `figures_training_compare/`, and `epoch_visibility_analysis/*.png|pdf`.

The three root notebooks are successive iterations of the same pipeline:
- `InstanceSeg_Code.ipynb` — current/most complete version (train → val → sample predictions → comparison plots → benchmark tables incl. TensorRT and F1 → per-class metrics → per-visibility validation).
- `Instance_segV2.ipynb`, `Instance_segV1.ipynb` — earlier versions, kept for history; prefer `InstanceSeg_Code.ipynb` for new work unless comparing against a specific past run.

## Environment

Python env is managed with conda; the spec is `yolo.yml` (despite the name, this is a **conda environment export**, not an Ultralytics data YAML — don't confuse it with `Manzana/yolo_reduced/data.yaml` or the `validation_by_visibility/*.yaml` dataset configs). It targets Windows, Python 3.11, env name `yolo`, and pins `ultralytics==8.3.230`. `torch`/`torchvision`/`torchaudio` are commented out in the file (installed separately, matched to the local CUDA version — the notebooks pick `device="cuda" if torch.cuda.is_available() else "cpu"`).

```powershell
conda env create -f yolo.yml
conda activate yolo
# torch/torchvision are commented out in yolo.yml — install matching your CUDA version separately, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

There is no test suite, linter, or build step in this repo — work happens by running the conversion scripts and executing notebook cells.

## Running the conversion scripts

```powershell
python export_gtruth_for_python.m   # run inside MATLAB, not from the shell
python filter_gtruth_flat_no_empty.py --in_mat gTruth_py_flat.mat --out_mat gTruth_py_flat_filtered.mat
python gtruth_flat_to_yolo.py --mat gTruth_py_flat_filtered.mat --labels_out Manzana/yolo_reduced/labels --yaml_out Manzana/yolo_reduced/data.yaml --dataset_root Manzana/yolo_reduced
python filter_remap_external_yolo.py --external_root <path_to_external_dataset> --out_root <merged_dataset_root> --tag external --drop_empty
```

## Paper manuscript (`paper/`)

The LaTeX source for the paper (currently untracked — not yet added to git):

- **Entry point**: `paper/Articulo/sn-articleOK.tex`, compiled with `paper/` as the working directory (class `llncs.cls` and bib style `splncs04.bst` sit at `paper/` root, e.g. `latexmk -pdf Articulo/sn-articleOK.tex` run from inside `paper/`). Build artifacts (`sn-articleOK.{aux,bbl,blg,fls,fdb_latexmk,log,pdf}`) also land at `paper/` root, alongside the sources — there's no `.gitignore` for them yet.
- **`paper/Articulo/`** holds the figures/tables actually `\includegraphics`'d by the manuscript (per-class `*_train/*_yolo11/*_yolo26/*_mask2former` comparison grids, `Arquitectura poster.pdf`, `quant_comparison_bar.*`, `Ref.bib`/`referencesOK.bib`) plus `Occlusion/` (copies of the `epoch_visibility_analysis/` curve/heatmap figures). Many other files sitting directly in `Articulo/` (NASA-TLX/SUS/UEQ/robustness/times mosaics, `Task1Diagram.pdf`, `Task2Diagram.pdf`, `Robotic-Arm.png`) are **not** referenced anywhere in `sn-articleOK.tex` — they're leftovers from a different manuscript/user-study, just not moved into `_unused/`.
- **`paper/_unused/`** is explicitly set-aside material: `llncs_template_docs/` is the stock LLNCS template's own sample/doc files (not project-specific), `other_project_fuzzy_robot/` is assets from an unrelated fuzzy-logic robot-control project, and `maybe_relevant_not_referenced/` is dataset-tooling screenshots that may or may not get used.

## Repository layout conventions

- **Versioned** (paper-relevant, reproducible from the pipeline above): root notebooks, `*.py` conversion scripts, `epoch_visibility_analysis/`, `validation_by_visibility/`, `figures_quant/`, `figures_training_compare/`, top-level metrics mosaics.
- **Gitignored** (bulk data / local artifacts, see `.gitignore`): `Manzana/` (raw captures + YOLO dataset), `runs/` (all Ultralytics train/val outputs), `ImageLabelingProject/`, `FruitLabelingProject_v2/`, `Imagenes_Output/`, `imagenes_comparacion/`, `referencias/` (third-party papers/PDFs), and the `.pt` weight files at the repo root.
- **Run naming** under `runs/segment/Manzana/`: `V{11|26}{n|s|m|l|x}_640` for training runs and the same name + `_val` for the corresponding validation run (e.g. `V26m_640`, `V26m_640_val`), matching `imgsz=640` and the YOLO major version/size letter used. `_comparisons/` and `_iou_curves/` hold cross-run comparison artifacts.
- `Manzana/csv/` holds raw robot/capture telemetry (`depth_metrics.csv`, `end_pose.csv`, `joint_states.csv`, `data_capture_status.csv`) associated with the image captures, not model metrics.

## Known issues in the conversion scripts

- `gtruth_flat_to_yolo.py` and `filter_gtruth_flat_no_empty.py` both call `loadmat(..., squeeze_me=True)` and then assert `labelPolys.ndim == 2`. `squeeze_me` collapses singleton dimensions, so a `.mat` with exactly one image or one class column loads `labelPolys` as 1D and the scripts abort with `RuntimeError` even though the data is valid.
- `filter_remap_external_yolo.py`: `lbl_dir1` and `lbl_dir2` (around the labels-folder lookup) are the same expression (`img_dir.parent / "labels"`), so the intended two-path fallback never tries an alternate location — an external dataset with labels anywhere else silently has its split skipped.
- `filter_remap_external_yolo.py`'s `filter_and_remap_label` swallows malformed class-id lines with a bare `except: continue` and no logging, so a corrupted label line disappears silently instead of surfacing as a warning.
