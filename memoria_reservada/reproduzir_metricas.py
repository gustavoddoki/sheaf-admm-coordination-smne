"""Reconstruct policy means for both stages from saved family-level metrics, without training."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parent
manifest = json.loads((base / 'MANIFEST_SHA256.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, name


def stage(gate_name, metrics_dir):
    gate = json.loads((base / 'protocolos' / gate_name).read_text())
    profiles = [p['name'] for p in gate['profiles']]
    families = gate['family_ids']
    shape = (len(gate['seeds']), len(profiles), len(gate['conditions']), 20, len(families))
    correct = np.full(shape, np.nan)
    disagreement = np.full(shape, np.nan)
    for si, seed in enumerate(gate['seeds']):
        with (base / metrics_dir / f'seed_{seed}.csv').open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                idx = (si, profiles.index(r['profile']), gate['conditions'].index(r['condition']),
                       int(r['iteration']) - 1, families.index(int(r['family'])))
                assert np.isnan(correct[idx])
                correct[idx] = float(r['correct_agreement'])
                disagreement[idx] = float(r['disagreement'])
    assert np.isfinite(correct).all() and np.isfinite(disagreement).all()
    ci = gate['conditions'].index('full')
    fault = [i for i in range(len(gate['conditions'])) if i != ci]
    report = {}
    for pi, p in enumerate(profiles):
        c = correct[:, pi, fault, -1, :].mean() * 100
        ex = (disagreement[:, pi][:, fault, 5:, :] - disagreement[:, pi, ci, 5:, :][:, None, :, :]).mean() * 100
        report[p] = {'correct_agreement_pct': float(c), 'excess_disagreement_pp': float(ex)}
    return report


stage1 = stage('gate_original.json', 'metricas')
stage2 = stage('gate_m5.json', 'metricas_m5')
assert abs(stage1['projection_on__retained_dual']['correct_agreement_pct'] - 75.14) < 0.01
assert abs(stage1['projection_on__standard_dual']['correct_agreement_pct'] - 65.15) < 0.01
assert abs(stage2['projection_on__retained_dual']['correct_agreement_pct'] - 75.91) < 0.01
assert abs(stage2['projection_on__standard_dual']['correct_agreement_pct'] - 65.60) < 0.01
print(json.dumps({'primeira_etapa_48_familias': stage1, 'segunda_etapa_96_familias': stage2}, indent=2))
