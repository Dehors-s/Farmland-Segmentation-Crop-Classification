code = open(r"D:\Work space\DeepLearning\farm\cropland_extraction\模型推理\u-net--CBAMV8.5.py", encoding="utf-8").read()
checks = [
    ("_normalize_percentile", "_normalize_percentile" in code),
    ("RandomScale aug", "RandomScale" in code),
    ("find_best_threshold", "find_best_threshold" in code),
    ("validate_with_threshold", "validate_with_threshold" in code),
    ("auto_find_threshold", "auto_find_threshold" in code),
    ("header V8.5", "V8.5" in code),
    ("NO decoder0", "decoder0" not in code),
    ("NO decoder0_up", "decoder0_up" not in code),
    ("percentile norm enabled", "norm_mode" in code),
]
for name, ok in checks:
    print(f"  [{ 'OK' if ok else 'FAIL' }] {name}")
print(f"  Lines: {len(code.splitlines())}")
