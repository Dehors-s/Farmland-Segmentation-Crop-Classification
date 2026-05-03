"""
模型训练记录导出工具
=====================
从 .pth checkpoint 中提取训练指标并导出为可分析的格式。

用法:
  # 导出单个模型
  python export_model_metrics.py --model path/to/best_model.pth --output ./analysis

  # 批量导出多个模型并对比
  python export_model_metrics.py ^
      --model V8=./results/v8/best_model.pth ^
      --model V8.5=./results/v8.5/best_model.pth ^
      --model V9=./results/v9/best_model.pth ^
      --output ./analysis

  # 只导出 CSV（不画图）
  python export_model_metrics.py --model path/to/best_model.pth --output ./analysis --no-plot

输出:
  - {model_name}_metrics.csv       # 逐 epoch 指标表
  - {model_name}_config.json       # 训练配置
  - comparison_metrics.csv         # 多模型关键指标对比
  - training_curves_comparison.png # 训练曲线对比图（多模型时）
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def extract_metrics(ckpt: dict) -> dict:
    """从 checkpoint 字典中提取所有指标和配置。"""
    metrics = {
        "epoch": ckpt.get("epoch"),
        "best_iou": ckpt.get("best_iou"),
        "best_bdy_iou": ckpt.get("best_bdy_iou"),
        "best_threshold": ckpt.get("best_threshold", None),
    }

    # 逐 epoch 序列
    for key in ["train_losses", "val_ious", "val_bdy_ious",
                 "boundary_losses", "distance_losses"]:
        arr = ckpt.get(key, [])
        metrics[key] = list(arr) if isinstance(arr, np.ndarray) else arr

    # 配置
    config = ckpt.get("config", {})
    if isinstance(config, dict):
        # 序列化配置（过滤不可 JSON 序列化的值）
        serializable = {}
        for k, v in config.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, OverflowError):
                serializable[k] = str(v)
        metrics["config"] = serializable
    else:
        metrics["config"] = {}

    return metrics


def export_csv(metrics: dict, output_path: str, model_name: str = "model"):
    """将逐 epoch 指标导出为 CSV。"""
    # 确定有效长度（所有序列长度一致）
    arrays = {k: v for k, v in metrics.items()
              if isinstance(v, list) and len(v) > 0}
    if not arrays:
        print(f"  [WARN] {model_name}: 无逐 epoch 数据，跳过 CSV")
        return

    n_epochs = max(len(v) for v in arrays.values())
    fieldnames = ["epoch"] + sorted(arrays.keys())
    # 按逻辑顺序重排
    order = ["train_losses", "val_ious", "val_bdy_ious",
             "boundary_losses", "distance_losses"]
    fieldnames = ["epoch"] + [k for k in order if k in arrays]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_epochs):
            row = {"epoch": i + 1}
            for key in arrays:
                arr = arrays[key]
                row[key] = arr[i] if i < len(arr) else ""
            writer.writerow(row)

    print(f"  [OK] {model_name}: CSV -> {output_path} ({n_epochs} epochs)")


def export_config(metrics: dict, output_path: str, model_name: str = "model"):
    """导出训练配置为 JSON。"""
    config = metrics.get("config", {})
    if not config:
        print(f"  [WARN] {model_name}: 无配置数据")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {model_name}: Config -> {output_path}")


def compare_models(all_metrics: dict, output_dir: str):
    """生成多模型对比表 CSV。"""
    rows = []
    for model_name, metrics in all_metrics.items():
        config = metrics.get("config", {})
        val_ious = metrics.get("val_ious", [])
        val_bdy = metrics.get("val_bdy_ious", [])
        bdy_loss = metrics.get("boundary_losses", [])

        row = {
            "model": model_name,
            "best_iou": metrics.get("best_iou", ""),
            "best_bdy_iou": metrics.get("best_bdy_iou", ""),
            "epochs_trained": len(val_ious) if val_ious else 0,
            "first_iou": f"{val_ious[0]:.4f}" if val_ious else "",
            "last_iou": f"{val_ious[-1]:.4f}" if val_ious else "",
            "max_iou": f"{max(val_ious):.4f}" if val_ious else "",
            "first_bdy_iou": f"{val_bdy[0]:.4f}" if val_bdy else "",
            "last_bdy_iou": f"{val_bdy[-1]:.4f}" if val_bdy else "",
            "bdy_loss_min": f"{min(bdy_loss):.4f}" if bdy_loss else "",
            "lr": config.get("lr", ""),
            "batch_size": config.get("batch_size", ""),
            "encoder": config.get("encoder_name", ""),
            "scheduler": config.get("scheduler_type", ""),
            "norm_mode": config.get("norm_mode", "legacy"),
            "min_bdy_weight": config.get("min_boundary_weight", ""),
            "dropout": config.get("dropout_rate", ""),
            "weight_decay": config.get("weight_decay", ""),
            "loss_type": config.get("loss_type", ""),
        }
        # 里程碑 (到达特定 IoU 的 epoch)
        if val_ious:
            for target in [0.60, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76, 0.77]:
                ep = next((i + 1 for i, v in enumerate(val_ious) if v >= target), "")
                row[f"epoch_to_{target:.2f}"] = ep
        rows.append(row)

    if not rows:
        return

    # 写出对比 CSV
    out_path = os.path.join(output_dir, "comparison_metrics.csv")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  [OK] 对比表 -> {out_path}")

    # 打印对比摘要
    print("\n" + "=" * 70)
    print("模型对比摘要")
    print("=" * 70)
    print(f"{'模型':<10} {'Best IoU':<12} {'Bdy IoU':<12} {'Epochs':<8} {'架构':<30}")
    print("-" * 70)
    for r in rows:
        name = r["model"]
        enc = r.get("encoder", "?")
        norm = r.get("norm_mode", "?")
        arch = f"{enc}+{norm}" if norm != "legacy" else enc
        print(f"{name:<10} {str(r['best_iou']):<12} {str(r['best_bdy_iou']):<12} {str(r['epochs_trained']):<8} {arch:<30}")
    print("=" * 70)


def plot_curves(all_metrics: dict, output_dir: str):
    """绘制多模型训练曲线对比图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib 未安装，跳过绘图")
        return

    if len(all_metrics) == 0:
        return

    # 找到最大 epoch 数以对齐 x 轴
    do_plot = False
    for m in all_metrics.values():
        if len(m.get("val_ious", [])) > 0:
            do_plot = True
            break
    if not do_plot:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    plot_config = [
        ("val_ious", "Validation IoU", "IoU", "lower right"),
        ("val_bdy_ious", "Validation Boundary IoU", "Boundary IoU", "lower right"),
        ("train_losses", "Train Loss", "Loss", "upper right"),
        ("boundary_losses", "Boundary Loss", "Loss", "upper right"),
        ("distance_losses", "Distance Loss", "Loss", "upper right"),
    ]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    markers = ["o", "s", "^", "D", "v"]

    for ax_idx, (key, title, ylabel, legend_loc) in enumerate(plot_config):
        ax = axes_flat[ax_idx]
        for i, (model_name, metrics) in enumerate(all_metrics.items()):
            arr = metrics.get(key, [])
            if not arr:
                continue
            epochs = list(range(1, len(arr) + 1))
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            ax.plot(epochs, arr, label=model_name, color=color,
                    marker=marker, markersize=3, linewidth=1.5, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc=legend_loc, fontsize=9)
        ax.grid(True, alpha=0.3)

    # 最后一个子图显示对比表
    ax = axes_flat[5]
    ax.axis("off")
    table_data = []
    col_labels = ["Model", "Best IoU", "Bdy IoU", "Epochs"]
    for name, m in all_metrics.items():
        vi = m.get("val_ious", [])
        vb = m.get("val_bdy_ious", [])
        table_data.append([
            name,
            f"{max(vi):.4f}" if vi else "-",
            f"{max(vb):.4f}" if vb else "-",
            str(len(vi)) if vi else "0"
        ])
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.6)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_fontsize(13)
            cell.get_text().set_fontweight("bold")
        elif key[0] % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    plt.suptitle("Training Curves Comparison", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] 对比图 -> {out_path}")


def process_model(model_path: str, model_name: str, output_dir: str, no_plot: bool):
    """处理单个模型文件。"""
    print(f"\n{'='*60}")
    print(f"处理: {model_name} <- {model_path}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  [ERROR] 文件不存在: {model_path}")
        return None

    # 加载 checkpoint
    print(f"  加载中...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    metrics = extract_metrics(ckpt)
    del ckpt  # 释放内存

    # 导出 CSV
    csv_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    export_csv(metrics, csv_path, model_name)

    # 导出配置
    config_path = os.path.join(output_dir, f"{model_name}_config.json")
    export_config(metrics, config_path, model_name)

    # 打印摘要
    val_ious = metrics.get("val_ious", [])
    val_bdy = metrics.get("val_bdy_ious", [])
    print(f"\n  摘要:")
    print(f"    训练轮数: {len(val_ious)}")
    print(f"    Best IoU: {metrics.get('best_iou', '?'):.4f}" if isinstance(metrics.get('best_iou'), float) else f"    Best IoU: {metrics.get('best_iou', '?')}")
    print(f"    Best Bdy IoU: {metrics.get('best_bdy_iou', '?'):.4f}" if isinstance(metrics.get('best_bdy_iou'), float) else f"    Best Bdy IoU: {metrics.get('best_bdy_iou', '?')}")
    if val_ious:
        print(f"    Max IoU: {max(val_ious):.4f} @ epoch {val_ious.index(max(val_ious)) + 1}")
    if val_bdy:
        print(f"    Max Bdy IoU: {max(val_bdy):.4f} @ epoch {val_bdy.index(max(val_bdy)) + 1}")

    # 里程碑
    if val_ious:
        milestones = []
        for t in [0.60, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76, 0.77]:
            for i, v in enumerate(val_ious):
                if v >= t:
                    milestones.append(f"    IoU ≥ {t:.2f} → epoch {i+1}")
                    break
        if milestones:
            print(f"  收敛里程碑:")
            for line in milestones[:6]:  # 只显示前6个
                print(line)
            if len(milestones) > 6:
                print(f"    ... (共 {len(milestones)} 个里程碑)")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="从 .pth checkpoint 导出训练指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型
  python export_model_metrics.py --model results/v8/best_model.pth --output ./analysis

  # 多模型对比（推荐: name=path 格式）
  python export_model_metrics.py ^
      --model V8=results/v8/best_model.pth ^
      --model V8.5=results/v8.5/best_model.pth ^
      --model V9=results/v9/best_model.pth ^
      --output ./analysis

  # 无绘图（仅 CSV + JSON）
  python export_model_metrics.py --model best_model.pth --output ./analysis --no-plot
        """
    )
    parser.add_argument(
        "--model", "-m",
        action="append",
        help="模型路径，格式: 名称=路径 (例如 V8=./results/v8/best_model.pth)。可多次使用。"
    )
    parser.add_argument(
        "--output", "-o",
        default="./model_analysis",
        help="输出目录 (默认: ./model_analysis)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="跳过生成对比图"
    )

    args = parser.parse_args()

    if not args.model:
        parser.print_help()
        print("\n[ERROR] 请至少指定一个 --model")
        sys.exit(1)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 解析模型参数
    all_metrics = {}
    for m in args.model:
        if "=" in m:
            name, path = m.split("=", 1)
        else:
            name = os.path.splitext(os.path.basename(m))[0]
            path = m
        metrics = process_model(path.strip(), name.strip(), output_dir, args.no_plot)
        if metrics is not None:
            all_metrics[name.strip()] = metrics

    # 多模型对比
    if len(all_metrics) >= 2:
        print(f"\n\n{'='*60}")
        print("生成多模型对比...")
        print(f"{'='*60}")
        compare_models(all_metrics, output_dir)
        if not args.no_plot:
            plot_curves(all_metrics, output_dir)

    print(f"\n所有输出已保存至: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
