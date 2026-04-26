import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

spec = importlib.util.spec_from_file_location("model", "u-net--CBAMV8_4090.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MultiTaskUNet = mod.MultiTaskUNet


def load_model(model_path, device, encoder_name="resnet50", in_channels=4):
    model = MultiTaskUNet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        use_cbam=True,
        use_spectral_attention=True,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    iou = ckpt.get("best_iou", "?")
    print(f"  加载: {model_path} (best IoU={iou})")
    return model


def read_image(path, in_channels=4):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".tif", ".tiff"]:
        import rasterio
        with rasterio.open(path) as ds:
            img = ds.read().astype(np.float32)
        img = np.transpose(img, (1, 2, 0))
    else:
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"无法读取: {path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if img.shape[2] < in_channels:
        pad = in_channels - img.shape[2]
        img = np.concatenate([img, np.zeros((*img.shape[:2], pad))], dtype=img.dtype)
    elif img.shape[2] > in_channels:
        img = img[:, :, :in_channels]
    img = np.nan_to_num(img.astype(np.float32), nan=0.0)
    if suffix in [".tif", ".tiff"]:
        max_val = img.max()
        if max_val > 2000:
            img = img / 10000.0
        elif max_val > 1.5:
            img = img / 255.0
    return np.clip(img, 0, 1)


def predict(model, img, device):
    img_size = 512
    h, w = img.shape[:2]
    if h > img_size or w > img_size:
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_logit, _, _ = model(tensor)
        seg = torch.sigmoid(seg_logit).squeeze().cpu().numpy()
    seg = (seg > 0.5).astype(np.uint8)
    if h != seg.shape[0] or w != seg.shape[1]:
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
    return seg


def compare_direct(img_a, img_b, name_a, name_b, out_path):
    """直接比较两张RGB图"""
    if img_a.shape[:2] != img_b.shape[:2]:
        h = min(img_a.shape[0], img_b.shape[0])
        w = min(img_a.shape[1], img_b.shape[1])
        img_a = cv2.resize(img_a, (w, h))
        img_b = cv2.resize(img_b, (w, h))

    rgb_a = img_a[:, :, :3]
    rgb_b = img_b[:, :, :3]

    gray_a = cv2.cvtColor((rgb_a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor((rgb_b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    diff = np.zeros((*gray_a.shape, 3), dtype=np.uint8)
    diff[(gray_a > 128) & (gray_b <= 128)] = [255, 0, 0]
    diff[(gray_a <= 128) & (gray_b > 128)] = [0, 0, 255]
    diff[(gray_a > 128) & (gray_b > 128)] = [0, 255, 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(rgb_a)
    axes[0, 0].set_title(name_a)
    axes[0, 0].axis("off")
    axes[0, 1].imshow(rgb_b)
    axes[0, 1].set_title(name_b)
    axes[0, 1].axis("off")
    axes[1, 0].imshow(diff)
    axes[1, 0].set_title("差异 (红=A独有 蓝=B独有 绿=共有)")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="对比两个模型的预测结果")
    parser.add_argument("--model_a", help="模型 A (.pth)")
    parser.add_argument("--model_b", help="模型 B (.pth)")
    parser.add_argument("--input", help="输入图像文件或目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--encoder_name", default="resnet50")
    parser.add_argument("--name_a", default="Model A")
    parser.add_argument("--name_b", default="Model B")
    parser.add_argument("--direct", nargs=2, metavar=("IMG_A", "IMG_B"),
                        help="直接比较两张图片（跳过推理）")
    args = parser.parse_args()

    if args.direct:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_a = read_image(args.direct[0], 3)
        img_b = read_image(args.direct[1], 3)
        stem_a = Path(args.direct[0]).stem
        compare_direct(img_a, img_b, args.name_a, args.name_b,
                       out_dir / f"{stem_a}_vs_{Path(args.direct[1]).stem}.png")
        print(f"对比结果: {out_dir}/")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model_a = load_model(args.model_a, device, args.encoder_name, args.in_channels)
    model_b = load_model(args.model_b, device, args.encoder_name, args.in_channels)

    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.tif")) + sorted(input_path.glob("*.tiff")) + sorted(input_path.glob("*.png"))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(files, desc="对比"):
        img = read_image(str(f), args.in_channels)
        seg_a = predict(model_a, img, device)
        seg_b = predict(model_b, img, device)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        rgb = img[:, :, :3] if img.shape[2] >= 3 else np.stack([img[:, :, 0]] * 3, axis=2)
        rgb = np.clip(rgb, 0, 1)

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("输入影像")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(seg_a, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title(f"{args.name_a} (预测)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(seg_b, cmap="gray", vmin=0, vmax=1)
        axes[0, 2].set_title(f"{args.name_b} (预测)")
        axes[0, 2].axis("off")

        overlay_a = rgb.copy()
        overlay_a[seg_a == 1] = [0, 1, 0]
        axes[1, 0].imshow(overlay_a)
        axes[1, 0].set_title(f"{args.name_a} 叠加")
        axes[1, 0].axis("off")

        overlay_b = rgb.copy()
        overlay_b[seg_b == 1] = [0, 1, 0]
        axes[1, 1].imshow(overlay_b)
        axes[1, 1].set_title(f"{args.name_b} 叠加")
        axes[1, 1].axis("off")

        diff = np.zeros((*seg_a.shape, 3), dtype=np.uint8)
        diff[(seg_a == 1) & (seg_b == 0)] = [255, 0, 0]
        diff[(seg_a == 0) & (seg_b == 1)] = [0, 0, 255]
        diff[(seg_a == 1) & (seg_b == 1)] = [0, 255, 0]
        axes[1, 2].imshow(diff)
        axes[1, 2].set_title(f"差异 (红=A独有 蓝=B独有 绿=共有)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"{f.stem}_compare.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"\n对比结果已保存到: {out_dir}/")


if __name__ == "__main__":
    main()
