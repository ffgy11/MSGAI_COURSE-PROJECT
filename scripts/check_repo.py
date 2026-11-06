#!/usr/bin/env python3
"""Static pre-upload sanity check; does not require ML dependencies."""
from __future__ import annotations
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / 'README.md',
    ROOT / 'pyproject.toml',
    ROOT / 'requirements.txt',
    ROOT / 'scripts/generate/generate_llada_trajectories.py',
    ROOT / 'scripts/generate/generate_dream_trajectories.py',
    ROOT / 'scripts/final/test_pcs_selection_compact.py',
    ROOT / 'scripts/final/test_rcc_all_configs.py',
    ROOT / 'scripts/final/analyze_rcc_selective_answering.py',
    ROOT / 'scripts/final/analyze_csqa64_rcc_selective.py',
    ROOT / 'scripts/final/label_csqa_manifest.py',
    ROOT / 'scripts/final/merge_hallucination_labels.py',
    ROOT / 'results/main/tracedet_main_table_with_rcc.csv',
    ROOT / 'results/main/csqa64_vs_128_selective.csv',
]

errors=[]
for p in REQUIRED:
    if not p.exists(): errors.append(f'missing: {p.relative_to(ROOT)}')

for base in [ROOT/'src', ROOT/'scripts/final', ROOT/'scripts/generate']:
    for p in base.rglob('*.py'):
        try:
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        except Exception as e:
            errors.append(f'syntax: {p.relative_to(ROOT)}: {e}')

bad=[]
for p in ROOT.rglob('*'):
    if p.resolve() == Path(__file__).resolve():
        continue
    if not p.is_file() or p.suffix.lower() in {'.png','.pdf','.zip','.npz','.npy','.pyc'}:
        continue
    try: s=p.read_text(encoding='utf-8')
    except Exception: continue
    if '/storage/homefs/' in s or 'pw25i008' in s:
        bad.append(str(p.relative_to(ROOT)))
if bad:
    errors.append('personal cluster paths remain in: ' + ', '.join(bad))

if errors:
    print('FAILED')
    for e in errors: print(' -',e)
    sys.exit(1)
print('OK: repository structure, Python syntax, and path sanitization checks passed.')
