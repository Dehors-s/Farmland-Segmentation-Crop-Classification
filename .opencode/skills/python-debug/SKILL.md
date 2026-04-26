---
name: python-debug
description: Debug Python errors with stack trace analysis and fix suggestions
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: debugging
---

## What I do

- Analyze Python tracebacks to identify root cause
- Distinguish between: coding errors, type mismatches, env issues, dependency conflicts
- Suggest targeted fixes with code examples
- Recommend debugging tools (pdb, logging, pytest)

## Common error patterns

### Traceback analysis
1. Read the error message — it often tells you exactly what's wrong
2. Check the last lines first (the actual exception)
3. Trace back through the call stack to find the origin
4. Identify: is it a logic bug, type error, import issue, or external dependency?

### Fix templates

**ImportError / ModuleNotFoundError**
```bash
pip list | grep <package>
pip install <package>          # missing package
pip install --upgrade <package>  # version conflict
```

**TypeError / ValueError**
- Check function signatures and argument types
- Add type hints and runtime assertions
- Use `isinstance()` guards for union types

**IndexError / KeyError**
- Verify data structure contents before access
- Use `.get(key, default)` for dicts
- Check list boundaries with `len()`

## Diagnostic commands

```bash
# Reproduce with verbose logging
python -u -W error script.py

# Check dependency tree
pip check
pipdeptree --warn silence | grep <problem_package>

# Profile memory usage
python -m memory_profiler script.py
```
