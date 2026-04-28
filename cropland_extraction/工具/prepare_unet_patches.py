"""遥感影像 U-Net 切片预处理脚本。

功能：
1. 对大幅 GeoTIFF 影像按滑窗方式切片。
2. 对矢量边界进行 CRS 对齐后栅格化，生成同窗口 mask。
3. 保存 image / mask 切片时完全保留影像原始波段数与 dtype，不做拉伸、归一化或强制类型转换。
4. mask 使用带地理空间元数据的 GeoTIFF 保存，便于训练脚本直接读取。
5. 通过空间索引加速多边形筛选，并对 NoData / NaN 比例过高的窗口进行跳过。

示例：
python prepare_unet_patches.py ^
  --tif_path "D:\\Work space\\DeepLearning\\farm\\data\\遥感图像\\HE017008_2024.tif" ^
  --shp_path "D:\\Work space\\DeepLearning\\farm\\data\\edge\\HE017008_2024-3\\HE017008_2024-3.shp" ^
  --output_dir "D:\\Work space\\DeepLearning\\farm\\dataset" ^
  --crop_size 512 ^
  --stride 256 ^
  --background_keep_ratio 0.1
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from shapely.geometry import box
from shapely.validation import make_valid
from tqdm import tqdm


def parse_args(defaults: Optional[dict] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description="遥感影像滑窗切片预处理，保留原始波段与 dtype，并生成 GeoTIFF mask。"
    )
    parser.add_argument("--tif_path", default=defaults.get("tif_path"), help="输入遥感影像 GeoTIFF 路径")
    parser.add_argument("--shp_path", default=defaults.get("shp_path"), help="输入矢量边界 SHP 路径")
    parser.add_argument("--output_dir", default=defaults.get("output_dir"), help="输出目录")
    parser.add_argument("--crop_size", type=int, default=defaults.get("crop_size", 512), help="切片尺寸，默认 512")
    parser.add_argument("--stride", type=int, default=defaults.get("stride", 256), help="滑窗步长，默认 256")
    parser.add_argument(
        "--background_keep_ratio",
        type=float,
        default=defaults.get("background_keep_ratio", 0.1),
        help="纯背景切片保留比例，默认 0.1",
    )

    parser.add_argument("--class_field", default=defaults.get("class_field", "class"), help="SHP 中的类别字段名")
    parser.add_argument("--positive_class_value", type=int, default=defaults.get("positive_class_value", 1), help="正类取值，默认 1")
    parser.add_argument(
        "--use_all_classes_for_mask",
        action="store_true",
        help="不筛选正类，使用全部矢量要素生成 mask",
    )
    parser.add_argument(
        "--selected_bands",
        default=defaults.get("selected_bands"),
        help="可选，仅读取指定波段，如 '1,2,3,4'。默认读取全部波段",
    )
    parser.add_argument(
        "--all_touched",
        action="store_true",
        help="栅格化时将接触到的像素都置为前景，默认关闭",
    )
    parser.add_argument(
        "--boundary_buffer",
        type=float,
        default=defaults.get("boundary_buffer", 0.0),
        help="边界缓冲距离，负值收缩、正值外扩，单位与影像 CRS 一致，默认 0",
    )
    parser.add_argument(
        "--parcel_gap_width",
        type=int,
        default=defaults.get("parcel_gap_width", 2),
        help="相邻地块之间插入的间隔宽度（像素），0 表示不插入间隔，默认 2",
    )
    parser.add_argument(
        "--max_nodata_ratio",
        type=float,
        default=defaults.get("max_nodata_ratio", 0.6),
        help="窗口中 NoData / NaN 比例超过该阈值则跳过，默认 0.6",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.get("seed", 42),
        help="随机种子，仅用于背景切片抽样",
    )
    parser.add_argument(
        "--audit_sample_count",
        type=int,
        default=defaults.get("audit_sample_count", 10),
        help="切片完成后抽样检查的样本数，默认 10",
    )
    parser.add_argument(
        "--allow_missing_shp_crs",
        action="store_true",
        default=defaults.get("allow_missing_shp_crs", True),
        help="当 SHP 缺少 CRS 时，默认按影像 CRS 视为已对齐",
    )
    parser.add_argument(
        "--allow_missing_tif_crs",
        action="store_true",
        default=defaults.get("allow_missing_tif_crs", False),
        help="当 TIF 缺少 CRS 时允许继续",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_band_list(text: Optional[str]) -> Optional[List[int]]:
    """把形如 '1,2,3,4' 的字符串解析为波段列表。"""
    if text is None or str(text).strip() == "":
        return None
    bands: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("selected_bands 中的波段编号必须从 1 开始。")
        bands.append(value)
    if not bands:
        return None
    return bands


def safe_fix_geometry(geom):
    """尽量修复无效几何，失败则返回 None。"""
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    try:
        fixed = make_valid(geom)
        if fixed is not None and not fixed.is_empty:
            return fixed
    except Exception:
        pass
    try:
        fixed = geom.buffer(0)
        if fixed is not None and not fixed.is_empty:
            return fixed
    except Exception:
        pass
    return None


def load_and_prepare_geometries(
    shp_path: Path,
    class_field: str,
    positive_class_value: int,
    use_all_classes_for_mask: bool,
) -> gpd.GeoDataFrame:
    """读取矢量并筛选出用于 mask 的几何。"""
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"矢量文件为空: {shp_path}")

    if class_field not in gdf.columns and not use_all_classes_for_mask:
        raise ValueError(f"SHP 中不存在字段: {class_field}")

    if use_all_classes_for_mask:
        filtered = gdf.copy()
    else:
        filtered = gdf[gdf[class_field] == positive_class_value].copy()

    filtered = filtered[filtered.geometry.notnull()].copy()
    filtered = filtered[~filtered.geometry.is_empty].copy()
    filtered["geometry"] = filtered.geometry.apply(safe_fix_geometry)
    filtered = filtered[filtered.geometry.notnull()].copy()
    filtered = filtered[~filtered.geometry.is_empty].copy()

    if filtered.empty:
        raise ValueError("筛选后没有可用于栅格化的有效几何。")

    return filtered


def apply_boundary_buffer(gdf: gpd.GeoDataFrame, boundary_buffer: float) -> gpd.GeoDataFrame:
    """对矢量边界做可选缓冲，用于控制 mask 的边界宽紧。"""
    if boundary_buffer == 0:
        return gdf

    buffered = gdf.copy()

    def _buffer_geom(geom):
        if geom is None or geom.is_empty:
            return None
        try:
            return safe_fix_geometry(geom.buffer(boundary_buffer))
        except Exception:
            return None

    buffered["geometry"] = buffered.geometry.apply(_buffer_geom)
    buffered = buffered[buffered.geometry.notnull()].copy()
    buffered = buffered[~buffered.geometry.is_empty].copy()
    if buffered.empty:
        raise ValueError("boundary_buffer 处理后几何为空，请减小缓冲距离。")
    return buffered


def align_geometries_to_raster(
    gdf: gpd.GeoDataFrame,
    raster_crs,
    allow_missing_shp_crs: bool = True,
    allow_missing_tif_crs: bool = False,
) -> gpd.GeoDataFrame:
    """对齐矢量 CRS 到影像 CRS。"""
    if gdf.crs is None:
        if raster_crs is None:
            print("警告：SHP 和 TIF 都缺少 CRS 信息，将按原始坐标直接继续处理。")
            return gdf
        if allow_missing_shp_crs:
            # 默认认为矢量与影像已经对齐，直接把影像 CRS 赋给矢量。
            return gdf.set_crs(raster_crs, allow_override=True)
        raise ValueError("SHP 缺少 CRS 信息，请先补齐或显式允许继续。")

    if raster_crs is None:
        if not allow_missing_tif_crs:
            print("警告：TIF 缺少 CRS 信息，将按原始坐标直接继续处理。")
            return gdf
        return gdf

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    return gdf


def get_fill_value(src: rasterio.io.DatasetReader):
    """为越界读取提供填充值，尽量贴近原始 nodata。"""
    if src.nodata is None:
        return 0
    if isinstance(src.nodata, float) and np.isnan(src.nodata):
        return np.nan
    return src.nodata


def get_locked_window(src: rasterio.io.DatasetReader, gdf: gpd.GeoDataFrame) -> Tuple[Window, int, int, int, int]:
    """根据矢量总外接框锁定需要扫描的像素范围。"""
    minx, miny, maxx, maxy = gdf.total_bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

    row_start = max(0, int(math.floor(win.row_off)))
    col_start = max(0, int(math.floor(win.col_off)))
    row_stop = min(src.height, int(math.ceil(win.row_off + win.height)))
    col_stop = min(src.width, int(math.ceil(win.col_off + win.width)))

    if row_stop <= row_start or col_stop <= col_start:
        raise ValueError("矢量范围与影像范围没有有效交集。")

    locked_window = Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )
    return locked_window, row_start, row_stop, col_start, col_stop


def generate_positions(start: int, stop: int, crop_size: int, stride: int) -> List[int]:
    """生成滑窗起点，保留边缘最后一个窗口。"""
    if stop <= start:
        return []
    if stop - start <= crop_size:
        return [start]

    positions = list(range(start, stop - crop_size + 1, stride))
    if not positions:
        positions = [start]

    edge_start = stop - crop_size
    if positions[-1] != edge_start:
        positions.append(edge_start)

    return sorted(set(positions))


def query_candidate_indices(sindex, bbox_geom, total_count: int) -> List[int]:
    """通过空间索引快速获取候选要素索引。"""
    if sindex is None:
        return list(range(total_count))

    try:
        result = sindex.query(bbox_geom, predicate="intersects")
        return list(result)
    except TypeError:
        pass
    except Exception:
        pass

    try:
        return list(sindex.intersection(bbox_geom.bounds))
    except Exception:
        return list(range(total_count))


def rasterize_mask_for_patch(
    farmland: gpd.GeoDataFrame,
    farmland_sindex,
    patch_window: Window,
    src: rasterio.io.DatasetReader,
    all_touched: bool = False,
    parcel_gap_width: int = 0,
) -> np.ndarray:
    """对当前窗口实时栅格化 mask。

    当 parcel_gap_width > 0 时，先用唯一 ID 逐地块栅格化，再检测相邻地块
    之间的边界，沿边界插入 0 值缝隙，从而在二值 mask 中保留地块轮廓。
    """
    out_height = int(patch_window.height)
    out_width = int(patch_window.width)
    patch_transform = src.window_transform(patch_window)

    left, bottom, right, top = window_bounds(patch_window, src.transform)
    patch_bbox = box(left, bottom, right, top)

    candidate_idx = query_candidate_indices(farmland_sindex, patch_bbox, len(farmland))
    if not candidate_idx:
        return np.zeros((out_height, out_width), dtype=np.uint8)

    shapes = []
    for idx in candidate_idx:
        geom = farmland.iloc[idx].geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.intersects(patch_bbox):
            continue
        shapes.append((geom, 1))

    if not shapes:
        return np.zeros((out_height, out_width), dtype=np.uint8)

    if parcel_gap_width <= 0:
        return rasterize(
            shapes=shapes,
            out_shape=(out_height, out_width),
            transform=patch_transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )

    instance_map = rasterize(
        shapes=[(g, i + 1) for i, (g, _) in enumerate(shapes)],
        out_shape=(out_height, out_width),
        transform=patch_transform,
        fill=0,
        dtype="uint16",
        all_touched=all_touched,
    )

    boundary_mask = np.zeros((out_height, out_width), dtype=bool)

    # 水平相邻（左右）
    mask_left = instance_map[:, :-1]
    mask_right = instance_map[:, 1:]
    diff_h = (mask_left != mask_right) & (mask_left > 0) & (mask_right > 0)
    boundary_mask[:, :-1] |= diff_h
    boundary_mask[:, 1:] |= diff_h

    # 垂直相邻（上下）
    mask_up = instance_map[:-1, :]
    mask_down = instance_map[1:, :]
    diff_v = (mask_up != mask_down) & (mask_up > 0) & (mask_down > 0)
    boundary_mask[:-1, :] |= diff_v
    boundary_mask[1:, :] |= diff_v

    # 对角相邻
    mask_ul = instance_map[:-1, :-1]
    mask_dr = instance_map[1:, 1:]
    diff_d1 = (mask_ul != mask_dr) & (mask_ul > 0) & (mask_dr > 0)
    boundary_mask[:-1, :-1] |= diff_d1
    boundary_mask[1:, 1:] |= diff_d1

    mask_ur = instance_map[:-1, 1:]
    mask_dl = instance_map[1:, :-1]
    diff_d2 = (mask_ur != mask_dl) & (mask_ur > 0) & (mask_dl > 0)
    boundary_mask[:-1, 1:] |= diff_d2
    boundary_mask[1:, :-1] |= diff_d2

    # 膨胀边界到指定宽度
    if parcel_gap_width > 1:
        kernel = np.ones((parcel_gap_width, parcel_gap_width), dtype=np.uint8)
        boundary_mask = cv2.dilate(boundary_mask.astype(np.uint8), kernel) > 0

    binary_mask = (instance_map > 0).astype(np.uint8)
    binary_mask[boundary_mask] = 0

    return binary_mask


def read_image_patch(
    src: rasterio.io.DatasetReader,
    patch_window: Window,
    selected_bands: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """读取影像窗口，完整保留原始波段和 dtype。"""
    fill_value = get_fill_value(src)
    if selected_bands is None:
        return src.read(
            window=patch_window,
            boundless=True,
            fill_value=fill_value,
            masked=False,
        )
    return src.read(
        indexes=list(selected_bands),
        window=patch_window,
        boundless=True,
        fill_value=fill_value,
        masked=False,
    )


def nodata_ratio(image_patch: np.ndarray, src: rasterio.io.DatasetReader) -> float:
    """估算当前窗口中的 NoData / NaN 比例。"""
    if image_patch.size == 0:
        return 1.0

    invalid = np.zeros(image_patch.shape, dtype=bool)
    if np.issubdtype(image_patch.dtype, np.floating):
        invalid |= ~np.isfinite(image_patch)

    if src.nodata is not None:
        nodata = src.nodata
        if not (isinstance(nodata, float) and np.isnan(nodata)):
            invalid |= image_patch == nodata

    return float(invalid.mean())


def should_keep_background(mask_patch: np.ndarray, background_keep_ratio: float) -> bool:
    """按比例保留纯背景切片。"""
    if background_keep_ratio >= 1.0:
        return True
    if background_keep_ratio <= 0.0:
        return False
    return random.random() < background_keep_ratio


def save_patch_pair(
    src: rasterio.io.DatasetReader,
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    patch_window: Window,
    image_path: Path,
    mask_path: Path,
) -> None:
    """保存影像和 mask 切片，并写入正确的空间元数据。"""
    image_profile = src.profile.copy()
    image_profile.update(
        driver="GTiff",
        height=int(image_patch.shape[1]),
        width=int(image_patch.shape[2]),
        count=int(image_patch.shape[0]),
        dtype=str(image_patch.dtype),
        crs=src.crs,
        transform=src.window_transform(patch_window),
        compress="lzw",
        tiled=True,
    )

    mask_profile = src.profile.copy()
    mask_profile.update(
        driver="GTiff",
        height=int(mask_patch.shape[0]),
        width=int(mask_patch.shape[1]),
        count=1,
        dtype="uint8",
        crs=src.crs,
        transform=src.window_transform(patch_window),
        compress="lzw",
        tiled=True,
        nodata=0,
    )

    ensure_dir(image_path.parent)
    ensure_dir(mask_path.parent)

    with rasterio.open(image_path, "w", **image_profile) as dst:
        dst.write(image_patch)

    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write((mask_patch.astype(np.uint8, copy=False) * 255), 1)


def stretch_to_uint8(image: np.ndarray) -> np.ndarray:
    """把单通道数组线性拉伸到 0-255，便于可视化。"""
    image = np.asarray(image)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)
    out = (image - min_val) / (max_val - min_val) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def make_rgb_preview(image_patch: np.ndarray) -> np.ndarray:
    """从多波段切片生成用于抽查的 RGB 预览图。"""
    if image_patch.ndim != 3:
        raise ValueError("image_patch 必须是 CxHxW 格式。")

    band_count = image_patch.shape[0]
    if band_count >= 3:
        rgb = image_patch[:3]
    elif band_count == 2:
        rgb = np.concatenate([image_patch, image_patch[:1]], axis=0)
    else:
        rgb = np.repeat(image_patch, 3, axis=0)

    rgb_hw = np.transpose(rgb, (1, 2, 0))
    preview = np.zeros_like(rgb_hw, dtype=np.uint8)
    for i in range(3):
        preview[:, :, i] = stretch_to_uint8(rgb_hw[:, :, i])
    return preview


def save_audit_triplet(audit_dir: Path, stem: str, image_patch: np.ndarray, mask_patch: np.ndarray) -> Tuple[float, Path]:
    """保存抽查样本的影像、mask 和叠加图，并返回前景比例。"""
    ensure_dir(audit_dir)
    fg_ratio = float((mask_patch > 0).mean()) if mask_patch.size else 0.0

    preview = make_rgb_preview(image_patch)
    mask_u8 = (mask_patch > 0).astype(np.uint8) * 255
    mask_color = np.zeros_like(preview)
    mask_color[:, :, 1] = mask_u8  # 绿色显示前景
    overlay = cv2.addWeighted(preview, 0.75, mask_color, 0.25, 0)

    cv2.imwrite(str(audit_dir / f"{stem}_image.png"), preview[:, :, ::-1])
    cv2.imwrite(str(audit_dir / f"{stem}_mask.png"), mask_u8)
    cv2.imwrite(str(audit_dir / f"{stem}_overlay.png"), overlay[:, :, ::-1])
    return fg_ratio, audit_dir / f"{stem}_overlay.png"


def process_dataset(args: argparse.Namespace) -> None:
    """主处理流程。"""
    tif_path = Path(args.tif_path)
    shp_path = Path(args.shp_path)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    ensure_dir(image_dir)
    ensure_dir(mask_dir)

    selected_bands = parse_band_list(args.selected_bands)
    random.seed(args.seed)

    farmland = load_and_prepare_geometries(
        shp_path=shp_path,
        class_field=args.class_field,
        positive_class_value=args.positive_class_value,
        use_all_classes_for_mask=args.use_all_classes_for_mask,
    )

    with rasterio.open(tif_path) as src:
        farmland = align_geometries_to_raster(
            farmland,
            raster_crs=src.crs,
            allow_missing_shp_crs=getattr(args, "allow_missing_shp_crs", True),
            allow_missing_tif_crs=getattr(args, "allow_missing_tif_crs", False),
        )
        farmland = apply_boundary_buffer(farmland, getattr(args, "boundary_buffer", 0.0))

        if selected_bands is not None:
            band_count = src.count
            invalid_bands = [b for b in selected_bands if b < 1 or b > band_count]
            if invalid_bands:
                raise ValueError(f"selected_bands 超出影像波段范围: {invalid_bands}, 影像总波段数: {band_count}")

        farmland_sindex = None
        try:
            farmland_sindex = farmland.sindex
        except Exception:
            farmland_sindex = None

        _, row_start, row_stop, col_start, col_stop = get_locked_window(src, farmland)
        row_positions = generate_positions(row_start, row_stop, args.crop_size, args.stride)
        col_positions = generate_positions(col_start, col_stop, args.crop_size, args.stride)
        total = len(row_positions) * len(col_positions)
        if total == 0:
            raise ValueError("有效区域不足以生成任何切片。")

        saved = 0
        skipped_nodata = 0
        skipped_background = 0
        skipped_empty_mask = 0
        saved_entries = []
        patch_stem = tif_path.stem

        pbar = tqdm(total=total, desc="切片进度", unit="patch")
        try:
            for row in row_positions:
                for col in col_positions:
                    patch_window = Window(
                        col_off=col,
                        row_off=row,
                        width=args.crop_size,
                        height=args.crop_size,
                    )

                    image_patch = read_image_patch(src, patch_window, selected_bands=selected_bands)
                    if image_patch.ndim != 3:
                        pbar.update(1)
                        continue

                    ratio = nodata_ratio(image_patch, src)
                    if ratio >= args.max_nodata_ratio:
                        skipped_nodata += 1
                        pbar.update(1)
                        continue

                    mask_patch = rasterize_mask_for_patch(
                        farmland=farmland,
                        farmland_sindex=farmland_sindex,
                        patch_window=patch_window,
                        src=src,
                        all_touched=args.all_touched,
                        parcel_gap_width=args.parcel_gap_width,
                    )

                    if mask_patch.size == 0:
                        skipped_empty_mask += 1
                        pbar.update(1)
                        continue

                    if np.all(mask_patch == 0):
                        if not should_keep_background(mask_patch, args.background_keep_ratio):
                            skipped_background += 1
                            pbar.update(1)
                            continue

                    patch_name = f"{patch_stem}_r{row}_c{col}.tif"
                    image_path = image_dir / patch_name
                    mask_path = mask_dir / patch_name

                    save_patch_pair(
                        src=src,
                        image_patch=image_patch,
                        mask_patch=mask_patch,
                        patch_window=patch_window,
                        image_path=image_path,
                        mask_path=mask_path,
                    )

                    saved += 1
                    saved_entries.append((patch_name, image_path, mask_path))
                    pbar.update(1)
        finally:
            pbar.close()

    audit_dir = output_dir / "audits"
    audit_report = audit_dir / "audit_report.txt"
    if saved_entries and args.audit_sample_count > 0:
        ensure_dir(audit_dir)
        sample_count = min(args.audit_sample_count, len(saved_entries))
        sampled = random.sample(saved_entries, k=sample_count)
        lines = ["样本抽查结果\n"]
        for idx, (patch_name, image_path, mask_path) in enumerate(sampled, start=1):
            with rasterio.open(image_path) as img_ds:
                image_patch = img_ds.read()
            with rasterio.open(mask_path) as mask_ds:
                mask_patch = mask_ds.read(1)
            fg_ratio, overlay_path = save_audit_triplet(audit_dir, f"audit_{idx:02d}_{Path(patch_name).stem}", image_patch, mask_patch)
            lines.append(f"{patch_name}\t前景比例={fg_ratio:.4f}\t叠加图={overlay_path.name}\n")
        audit_report.write_text("".join(lines), encoding="utf-8")

    print("\n处理完成")
    print(f"输入影像: {tif_path}")
    print(f"输入矢量: {shp_path}")
    print(f"输出目录: {output_dir}")
    print(f"成功保存切片: {saved}")
    print(f"因 NoData / NaN 过高跳过: {skipped_nodata}")
    print(f"因纯背景丢弃: {skipped_background}")
    print(f"因空 mask 跳过: {skipped_empty_mask}")
    if saved_entries and args.audit_sample_count > 0:
        print(f"抽查结果已保存: {audit_dir}")


def main() -> None:
    # 直接在这里修改默认参数，免去每次手写命令行。
    defaults = {
        "tif_path": r"D:\Work space\DeepLearning\farm\data\遥感图像\HE015009_2024.tif",
        "shp_path": r"D:\Work space\DeepLearning\farm\data\edge\HE015009_2024-12\HE015009_2024-12.shp",
        "output_dir": r"D:\Work space\DeepLearning\farm\dataset",
        "crop_size": 512,
        "stride": 256,
        "background_keep_ratio": 0.1,
        "parcel_gap_width": 2,
        "class_field": "class",
        "positive_class_value": 1,
        "selected_bands": None,
        "boundary_buffer": 0.0,
        "max_nodata_ratio": 0.6,
        "seed": 42,
        "allow_missing_shp_crs": True,
        "allow_missing_tif_crs": False,
    }
    args = parse_args(defaults)
    process_dataset(args)


if __name__ == "__main__":
    main()
