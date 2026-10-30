"""Shared data access for the figure set. Every number shown in a figure comes from these files
(per-scan outputs of the locked pipeline and the JSON outputs of the re-analysis scripts)."""
import json, ast, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

W = Path(os.environ.get('NEUROCARE_WORK', Path(__file__).resolve().parents[1]))
D = W / 'data'
E = D / 'evidence'
T = D / 'tables'
OUT = W / 'out'

REG = json.load(open(OUT / 'registry.json'))
N = json.load(open(OUT / 'new_analyses_cq500.json'))
S = json.load(open(OUT / 'new_analyses_seg.json'))
VO = json.load(open(OUT / 'verify_original.json'))
TC = json.load(open(OUT / 'thresholds_clinical.json'))
X = json.load(open(OUT / 'extra_checks.json'))
X2 = json.load(open(OUT / 'extra_checks2.json'))
X3 = json.load(open(OUT / 'extra_checks3.json'))
CQ = json.load(open(E / 'cq500_external/cq500_results.json'))
V2R = json.load(open(E / 'neurocare_v2_20261021/v2_results.json'))
V3R = json.load(open(E / 'v3_subtypes/v3b_results.json'))
R3 = json.load(open(OUT / 'posthoc.json'))                # post hoc analyses (ambiguity by extent, Holm, PPV counterfactual, BHX discordance)
CAPR = json.load(open(OUT / 'capacity_rubric.json'))       # model-independent validation-capacity rubric (17 endpoints) + sensitivity
LED = json.load(open(OUT / 'ledger.json'))              # integrity, discordance and convention ledger
CALLM = json.load(open(OUT / 'cq500_call_matrix_colneg.json'))

QCID = 'ID_a429fd9a_ID_2eabc43f3d'
SUB = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH']


def load_cq500():
    p = pd.read_csv(E / 'cq500_external/cq500_per_scan.csv')
    v = pd.read_csv(E / 'v3_subtypes/cq500_v3b_per_scan.csv')
    r = pd.read_csv(E / 'cq500_external/reads.csv')
    v['dz'] = v.spacing.apply(lambda s: ast.literal_eval(s)[2])
    m = p.merge(v[['name', 'EDH_ml', 'IPH_ml', 'IVH_ml', 'SAH_ml', 'SDH_ml', 'ANY_ml', 'dz']], on='name', how='left')
    m = m.merge(r, on='name', how='left')
    assert len(m) == 489 and m.ANY_ml.notna().all()
    return m


def votes(m, lab):
    return m[[f'R1:{lab}', f'R2:{lab}', f'R3:{lab}']].sum(axis=1).astype(int).values


def bhsd():
    v1 = pd.read_csv(E / 'neurocare_neuroimaging_advanced_20261021/per_scan_quantitative_metrics.csv').set_index('case_id')
    nm = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/ablation_no_postmask.csv').set_index('case_id')
    v2 = pd.read_csv(E / 'neurocare_v2_20261021/hemorrhage_v2/v2_metrics.csv')
    v2b = v2[v2.dataset == 'BHSD_test'].set_index('case_id')
    v3b = pd.read_csv(E / 'v3_subtypes/v3b/v3b_bhsd_test_metrics.csv').set_index('case_id')
    v3 = pd.read_csv(E / 'v3_subtypes/v3/v3_bhsd_test_metrics.csv').set_index('case_id')
    k = sorted(v1.index)
    return k, v1.loc[k], nm.loc[k], v2b.loc[k], v3b.loc[k], v3.loc[k]


def rr_pairs():
    return pd.read_csv(T / 'TableS4_Reference_reference_agreement.csv')


def mr():
    return pd.read_csv(T / 'MRBHSD_full43_compact_model_rater_metrics.csv')
