# AGENTS

## Scope and source of truth
- This repo has two active codebases in separate directories:
  - `cropland_extraction/` (segmentation training + vectorization, formerly `farm-u-net/`)
  - `crop_classification/crop_segmentation/` (parcel-level crop classification, formerly `crop_segmentation/`)
- Treat `过程文件/` as historical/experimental copies; use scripts under the two project directories first.

## Environment and install
- Use Python 3.8+.
- Each project has its own `requirements.txt`:
  - `cropland_extraction/requirements.txt` for U-Net
  - `crop_classification/requirements.txt` for crop classification
- Main install: `pip install -r requirements.txt` (root; includes geospatial + ML stack).

## High-value commands
- U-Net V7 train: `python cropland_extraction/u-net--CBAMV7.py --data_root <dataset_root> --output_dir <out_dir> --encoder_name resnet50 --batch_size 8 --epochs 80 --lr 5e-4 --num_workers 4`
- U-Net V8 train (multispectral): `python cropland_extraction/u-net--CBAMV8.py --data_root <dataset_root> --output_dir <out_dir> --in_channels 4`
- U-Net V8.4090 train (RTX 4090 optimized): `python cropland_extraction/u-net--CBAMV8_4090.py --data_root ./dataset --output_dir ./v8_4090 --encoder_name resnet50 --batch_size 32 --epochs 100 --lr 5e-4 --in_channels 4 --num_workers 8`
- U-Net inference + vectorization V2: `python cropland_extraction/u-net矢量化V2.py --model <pth> --input <image_or_dir> --output <out_dir> --in_channels 4 --save_shp`
- Crop classification smoke check: `python crop_classification/crop_segmentation/check_model_switch_smoke.py`
- Crop module import sanity check: `python crop_classification/crop_segmentation/test_module_import.py`

## OpenCode Skills
This project includes reusable skills in `.opencode/skills/`. Load with `/load <name>`:

| Skill | Purpose |
|-------|---------|
| `farm-training` | U-Net training pipeline: data prep, V7/V8 config, tuning, vectorization |
| `python-debug` | Python traceback analysis and fix suggestions |
| `pr-review` | Code quality, security, performance review checklist |
| `git-release` | Generate release notes and version bumps from git history |
| `docker-optimize` | Dockerfile/docker-compose best practices |

## Data and path conventions that break runs
- V7/V8 training expects dataset layout `.../train/img`, `.../train/lbl`, `.../val/img`, `.../val/lbl`.
- Most scripts assume local absolute Windows paths with spaces; always quote paths in commands.
- `cropland_extraction/run_pipeline.bat` is interactive and uses hardcoded defaults; edit its config block before use.

## Important implementation quirks
- In `cropland_extraction/u-net--CBAMV7.py`, `config['resume_path']` is hardcoded to `autodl-tmp/...`; usually missing locally and only prints a warning.
- In `cropland_extraction/u-net矢量化V2.py`, use `--save_shp` for ArcGIS compatibility (GeoJSON is unreliable for non-WGS84 data).
- `crop_classification/crop_segmentation/core/data_loader_optimized.py` may duplicate parcels across chunks (no global dedup) and does not actually window-read in `process_chunk` (still masks full raster source).

## crop_classification/crop_segmentation execution flow
- Programmatic API entrypoints are `crop_classification/crop_segmentation/interfaces/train_interface.py:22` (`train_pipeline`) and `crop_classification/crop_segmentation/interfaces/infer_interface.py:11` (`predict_pipeline`).
- `train_pipeline` auto-switches between full-memory and chunked loading via `should_load_in_memory(...)`; use `force_memory=True` only when RAM is known sufficient.
- Model bundles are saved as `<model_type>_model_bundle.joblib` and include scaler + label encoder; inference expects this bundle format.

## Git/outputs gotcha
- `.gitignore` ignores `*.json`, `*.png`, `*.jpg`, `*.tif`, `dataset/`, `results/`, and `models/`; many generated evaluation artifacts will not appear in `git status` unless force-added.
