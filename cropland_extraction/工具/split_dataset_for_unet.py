import argparse
import random
import shutil
from pathlib import Path


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clear_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="将 images/masks 划分为 train/val 的 img/lbl 结构")
    parser.add_argument("--data-root", type=str, default=r"D:\Work space\DeepLearning\farm\dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move"])
    parser.add_argument("--clean", action="store_true", help="清空已有 train/val 目录后再写入")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"缺少目录: {images_dir} 或 {masks_dir}")

    train_img = data_root / "train" / "img"
    train_lbl = data_root / "train" / "lbl"
    val_img = data_root / "val" / "img"
    val_lbl = data_root / "val" / "lbl"

    if args.clean:
        clear_dir(train_img.parent)
        clear_dir(val_img.parent)

    ensure_dir(train_img)
    ensure_dir(train_lbl)
    ensure_dir(val_img)
    ensure_dir(val_lbl)

    image_files = sorted([
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".tif", ".tiff"]
    ])
    if len(image_files) == 0:
        raise ValueError(f"{images_dir} 下没有图像文件 (支持 .png/.tif/.tiff)")

    paired = []
    missing_mask = 0
    for img_path in image_files:
        msk_path = masks_dir / img_path.name
        if msk_path.exists():
            paired.append((img_path, msk_path))
        else:
            missing_mask += 1

    if len(paired) == 0:
        raise ValueError("没有找到同名 image/mask 配对文件")

    random.seed(args.seed)
    random.shuffle(paired)

    val_count = int(len(paired) * args.val_ratio)
    val_pairs = paired[:val_count]
    train_pairs = paired[val_count:]

    op = shutil.copy2 if args.mode == "copy" else shutil.move

    for img_path, msk_path in train_pairs:
        op(img_path, train_img / img_path.name)
        op(msk_path, train_lbl / msk_path.name)

    for img_path, msk_path in val_pairs:
        op(img_path, val_img / img_path.name)
        op(msk_path, val_lbl / msk_path.name)

    print(f"总图片: {len(image_files)}")
    print(f"成功配对: {len(paired)}")
    print(f"缺少掩膜: {missing_mask}")
    print(f"训练集: {len(train_pairs)}")
    print(f"验证集: {len(val_pairs)}")
    print(f"模式: {args.mode}")
    print("完成。")


if __name__ == "__main__":
    main()