---
name: farm-training
description: U-Net training pipeline: data prep, V7/V8 config, tuning, vectorization
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: training
---

## What I do

- Guide through the U-Net training pipeline from data prep to vectorization
- Distinguish between V7 (RGB-only) and V8 (multispectral + spectral attention)
- Suggest hyperparameter tuning strategies for farmland segmentation

## Pipeline overview

```
prepare_unet_patches.py  →  images/*.tif + masks/*.tif
split_dataset_for_unet.py  →  train/img/ train/lbl/ val/img/ val/lbl/
u-net--CBAMV8.py  →  best_model.pth + best_boundary_model.pth
u-net矢量化V2.py  →  GeoJSON + Shapefile + visualization
```

## V7 vs V8 comparison

| Aspect | V7 | V8 |
|--------|----|----|
| Input channels | 3 (RGB) | 4+ (multispectral) |
| Spectral attention | No | Yes (SpectralAttention) |
| Encoder | hardcoded resnet34 | configurable (resnet50 default) |
| Loss | DiceBCE | LogCoshDice (PRUE 2025) |
| TTA/CRF | No | Yes (in V2 vectorization) |
| Resume | manual | supported |
| AMP | No | Yes |

## Training parameters (RTX 4090)

```powershell
python u-net--CBAMV8_4090.py ^
  --data_root "./dataset" --output_dir "./v8_4090" ^
  --encoder_name resnet50 --batch_size 32 --epochs 100 --lr 5e-4 ^
  --in_channels 4 --num_workers 8
```

### Key tuning knobs
- **batch_size**: 4090 24GB → 32 (resnet50 512×512)
- **lr**: larger batch → higher lr (5e-4 for bs=32, 3e-4 for bs=16)
- **encoder_name**: resnet50 for accuracy, resnet34 for speed
- **loss_type**: log_cosh_dice (default), tversky (for imbalanced data)
- **warmup_epochs**: 3-5 for large batch, 2 for small batch

## Data preparation tips

- Masks in `prepare_unet_patches.py` can use `--parcel_gap_width 2` to preserve field boundaries
- CRS in GeoTIFF may be broken — check with `rasterio.open().crs`
- If PROJ has database issues: `conda install -c conda-forge proj-data`
- split_dataset script needs `.tif` support (modify `glob("*.png")` to `iterdir()`)

## Vectorization

```powershell
python u-net矢量化V2.py ^
  --model best_model.pth --input val/img --output results ^
  --in_channels 4 --save_shp
```

Use `--source_epsg` for WGS84 conversion if PROJ works.
Add `--save_shp` for ArcGIS-compatible Shapefile output.

## FAQ

**Q**: First epoch IoU is low (0.28 vs 0.57)?
**A**: Check if ImageNet pretrained weights loaded. V8 loads resnet50-0676ba61.pth.

**Q**: Boundary IoU stays at 0.01?
**A**: Normal — decoder is randomly initialized. Improves after ~10 epochs.

**Q**: GeoJSON empty in ArcGIS?
**A**: Use Shapefile output (`--save_shp`). ArcGIS has issues with GeoJSON in non-WGS84.

**Q**: Training loss not decreasing?
**A**: Check learning rate, data normalization (values in [0,1]), and teed batch normalization layers.
