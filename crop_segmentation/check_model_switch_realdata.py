import os
import sys
import json
import traceback
from datetime import datetime

import geopandas as gpd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(CURRENT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from crop_segmentation.interfaces.train_interface import train_pipeline
from crop_segmentation.interfaces.infer_interface import predict_pipeline


def main():
    shp_full = r"D:\Work space\DeepLearning\耕地分割\data\完整作物边界\作物分布.shp"
    tif_train = r"D:\Work space\DeepLearning\耕地分割\data\Sentinel2_TimeSeries_Stack_2023-0000000000-0000000000.tif"
    tif_infer = r"D:\Work space\DeepLearning\耕地分割\data\Sentinel2_TimeSeries_Stack_2023-0000000000-0000006400.tif"

    sample_size = 300
    random_seed = 42

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(WORKSPACE_ROOT, "results", f"model_switch_realcheck_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("[1/4] 读取并抽样真实地块...")
    gdf = gpd.read_file(shp_full)
    n = min(sample_size, len(gdf))
    gdf_sample = gdf.sample(n=n, random_state=random_seed)

    sampled_shp = os.path.join(run_dir, "sampled_parcels.shp")
    gdf_sample.to_file(sampled_shp, encoding="utf-8")
    print(f"  抽样完成: {len(gdf_sample)} 条 -> {sampled_shp}")

    model_types = ["rf", "svm", "xgboost", "lgbm", "cnn"]
    summary = {
        "sample_size": int(len(gdf_sample)),
        "sampled_shp": sampled_shp,
        "train_tif": tif_train,
        "infer_tif": tif_infer,
        "results": {}
    }

    print("[2/4] 逐模型训练+推理...")
    for model_type in model_types:
        print("\n" + "=" * 90)
        print(f"模型: {model_type}")
        print("=" * 90)

        model_out_dir = os.path.join(run_dir, "models")
        os.makedirs(model_out_dir, exist_ok=True)

        infer_out = os.path.join(run_dir, f"pred_{model_type}.shp")

        model_kwargs = {}
        if model_type == "cnn":
            model_kwargs.update({"epochs": 2, "batch_size": 16, "learning_rate": 0.001})

        try:
            model_path = train_pipeline(
                model_type=model_type,
                shp_path=sampled_shp,
                tif_path=tif_train,
                output_dir=model_out_dir,
                label_column="value",
                use_optimized_loader=True,
                max_workers=4,
                chunk_size=1024,
                force_memory=False,
                grid_search=False,
                eval_mode=False,
                **model_kwargs,
            )

            out_shp = predict_pipeline(
                model_path=model_path,
                tif_path=tif_infer,
                shp_path=sampled_shp,
                output_shp=infer_out,
                force_memory=False,
                max_workers=4,
                chunk_size=1024,
            )

            summary["results"][model_type] = {
                "status": "PASS",
                "model_path": model_path,
                "output_shp": out_shp,
            }
            print(f"[PASS] {model_type} -> {out_shp}")
        except Exception as e:
            summary["results"][model_type] = {
                "status": "FAIL",
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"[FAIL] {model_type}: {type(e).__name__}: {e}")

    print("\n[3/4] 保存汇总...")
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  汇总文件: {summary_path}")

    print("[4/4] 最终结果")
    for m, info in summary["results"].items():
        print(f"  - {m}: {info['status']}")


if __name__ == "__main__":
    main()
