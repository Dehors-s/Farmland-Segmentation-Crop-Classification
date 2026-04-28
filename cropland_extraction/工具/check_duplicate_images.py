import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(folder: Path):
    hash_map = defaultdict(list)
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

    for p in files:
        try:
            digest = file_sha256(p)
            hash_map[digest].append(p)
        except Exception as e:
            print(f"[跳过] {p}，原因: {e}")

    dup_groups = [paths for paths in hash_map.values() if len(paths) > 1]
    dup_groups.sort(key=len, reverse=True)
    return files, dup_groups


def main():
    parser = argparse.ArgumentParser(description="检测图片重复（按文件内容哈希）")
    parser.add_argument(
        "--folder",
        type=str,
        default=r"D:\Work space\DeepLearning\耕地分割\dataset\images",
        help="待检测图片目录",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default="",
        help="可选：保存重复报告到 txt 文件路径",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"目录不存在: {folder}")

    files, dup_groups = find_duplicates(folder)

    print(f"\n扫描目录: {folder}")
    print(f"图片总数: {len(files)}")
    print(f"重复组数: {len(dup_groups)}")

    duplicate_file_count = sum(len(g) for g in dup_groups)
    redundant_count = sum(len(g) - 1 for g in dup_groups)
    print(f"重复文件总数(含每组首个): {duplicate_file_count}")
    print(f"可删除冗余数量: {redundant_count}\n")

    lines = []
    for i, group in enumerate(dup_groups, 1):
        lines.append(f"=== 重复组 #{i} (共 {len(group)} 张) ===")
        print(lines[-1])
        for p in group:
            rel = p.relative_to(folder)
            line = f"  {rel}"
            print(line)
            lines.append(line)
        print("")
        lines.append("")

    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()