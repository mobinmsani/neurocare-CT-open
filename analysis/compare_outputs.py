"""Compare regenerated outputs with the delivered ones: JSON values (numbers to a relative tolerance) and CSV tables.
Usage: python compare_outputs.py <regenerated out/> <out_delivered/>"""
import json, sys, math
from pathlib import Path
import pandas as pd

A, B = Path(sys.argv[1]), Path(sys.argv[2])
TOL = 1e-9
bad = []


def cmp(x, y, path):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in set(x) | set(y):
            if k not in x or k not in y:
                bad.append(f'{path}.{k}: missing in one file'); continue
            cmp(x[k], y[k], f'{path}.{k}')
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            bad.append(f'{path}: list length {len(x)} vs {len(y)}'); return
        for i, (u, v) in enumerate(zip(x, y)):
            cmp(u, v, f'{path}[{i}]')
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        if not (math.isclose(x, y, rel_tol=1e-6, abs_tol=TOL) or (math.isnan(x) and math.isnan(y))):
            bad.append(f'{path}: {x} vs {y}')
    elif x != y:
        bad.append(f'{path}: {str(x)[:60]} vs {str(y)[:60]}')


n = 0
for f in sorted(B.glob('*.json')):
    g = A / f.name
    if not g.exists():
        bad.append(f'{f.name}: not regenerated'); continue
    cmp(json.load(open(g)), json.load(open(f)), f.name); n += 1
for f in sorted((B / 'tables').glob('*.csv')):
    g = A / 'tables' / f.name
    if not g.exists():
        bad.append(f'tables/{f.name}: not regenerated'); continue
    a, b = pd.read_csv(g), pd.read_csv(f)
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6, atol=1e-9)
    except AssertionError as e:
        bad.append(f'tables/{f.name}: {str(e).splitlines()[0][:120]}')
    n += 1
for f in sorted(B.glob('*.txt')):
    g = A / f.name
    if not g.exists() or g.read_text() != f.read_text():
        bad.append(f'{f.name}: differs or not regenerated')
    n += 1
print(f'compared {n} files; {len(bad)} differences')
for b in bad[:200]:
    print(' ', b)
