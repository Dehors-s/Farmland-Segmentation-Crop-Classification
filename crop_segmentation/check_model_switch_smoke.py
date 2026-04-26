import os
import sys
import tempfile
import numpy as np
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(CURRENT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from crop_segmentation.interfaces import train_interface as ti


def main():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 16)).astype('float32')
    y = np.array([0, 1, 2] * 40)
    feature_names = [f'f{i}' for i in range(X.shape[1])]

    ti.load_data = lambda shp, tif, label: (X, y, feature_names)
    ti.load_data_optimized = lambda shp, tif, label, chunk_size=1024, max_workers=4: (X, y, feature_names)

    results = {}
    outdir = tempfile.mkdtemp(prefix='model_switch_smoke_')

    for mt in ['rf', 'svm', 'xgboost', 'lgbm', 'cnn']:
        try:
            kwargs = {'epochs': 1, 'batch_size': 16} if mt == 'cnn' else {}
            model_path = ti.train_pipeline(
                model_type=mt,
                shp_path='dummy.shp',
                tif_path='dummy.tif',
                output_dir=outdir,
                label_column='value',
                use_optimized_loader=True,
                eval_mode=False,
                grid_search=False,
                **kwargs,
            )
            bundle = joblib.load(model_path)
            model = bundle['model']
            scaler = bundle['scaler']
            test_x = scaler.transform(X[:1])

            proba_ok = True
            proba_error = ''
            try:
                _ = model.predict_proba(test_x)
            except Exception as pe:
                proba_ok = False
                proba_error = f"{type(pe).__name__}: {pe}"

            results[mt] = (
                f"OK | saved={os.path.exists(model_path)} | file={os.path.basename(model_path)}"
                f" | predict_proba={proba_ok}"
                + (f" | proba_error={proba_error}" if not proba_ok else "")
            )
        except Exception as e:
            results[mt] = f"FAIL | {type(e).__name__}: {e}"

    print('--- smoke results ---')
    for k, v in results.items():
        print(f'{k}: {v}')
    print('outdir=' + outdir)


if __name__ == '__main__':
    main()
