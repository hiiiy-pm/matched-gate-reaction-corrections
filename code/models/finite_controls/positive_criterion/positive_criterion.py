"""Independent exact finite-state checks of a classical capacity consequence.

Uses Python Fraction for exact algebra and NumPy/SciPy only for a separately
labelled floating-point finite-rate scan. Does not import project solvers.
All new output is written beside this script, inside the model directory.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.linalg import solve as floating_solve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]

def transpose(a):
    return [list(v) for v in zip(*a)]

def mm(a, b):
    bt = transpose(b)
    return [[sum((x*y for x, y in zip(row, col)), F(0)) for col in bt] for row in a]

def mv(a, x):
    return [sum((v*w for v, w in zip(row, x)), F(0)) for row in a]

def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))

def minus(a, b):
    return [x-y for x, y in zip(a, b)]

def inv(a):
    """Fraction Gauss-Jordan, no symbolic/numerical linear-algebra library."""
    n = len(a)
    b = [row[:] + e for row, e in zip(a, eye(n))]
    for j in range(n):
        p = next((i for i in range(j, n) if b[i][j]), None)
        if p is None:
            raise ValueError("singular matrix")
        b[j], b[p] = b[p], b[j]
        s = b[j][j]
        b[j] = [v/s for v in b[j]]
        for i in range(n):
            if i != j and b[i][j]:
                q = b[i][j]
                b[i] = [x-q*y for x, y in zip(b[i], b[j])]
    return [row[n:] for row in b]

def norm_inf(a):
    return max(sum(abs(v) for v in row) for row in a)

@lru_cache(None)
def gate_green(n, gamma=F(1)):
    """Four-site formula, verified using exact cycle Poisson residual."""
    H = eye(n)
    for i in range(n):
        H[i][(i-1) % n] -= F(1, 2)
        H[i][(i+1) % n] -= F(1, 2)
    Hp = [[F(n*n-1, 6*n) - F(((i-j) % n)*(n-(i-j) % n), n)
           for j in range(n)] for i in range(n)]
    assert all(sum(row) == 0 for row in Hp)
    assert mm(H, Hp) == [[F(i == j)-F(1, n) for j in range(n)] for i in range(n)]
    A1 = inv([[F(i == j)+gamma*H[i][j] for j in range(n)] for i in range(n)])
    A2 = inv([[2*F(i == j)+gamma*H[i][j] for j in range(n)] for i in range(n)])
    return [[Hp[i][j]/(4*gamma)+A1[i][j]/2+A2[i][j]/4
             for j in range(n)] for i in range(n)]

def exact_model(n, states, reference=None, gamma=F(1)):
    r, m = len(states), 4*n
    G = gate_green(n, gamma)
    M = [[G[i][j] for j in states] for i in states]
    u = [F(1, r)]*r
    P = [[F(i == j)-F(1, r) for j in range(r)] for i in range(r)]
    b = mv(P, mv(M, u))
    W = [[F(i == j) if i < r-1 else F(-1) for j in range(r-1)] for i in range(r)]
    C = mm(mm(transpose(W), M), W)
    Ci = inv(C)
    d = mv(transpose(W), mv(M, u))
    x = mv(W, mv(Ci, d))
    eta = minus(u, x)
    tau_stat = m*dot(u, mv(M, u))
    energy = dot(b, x)
    tau = tau_stat-m*energy
    assert sum(eta) == 1 and all(v >= 0 for v in eta)
    assert mv(M, eta) == [tau/m]*r
    assert mv(P, mv(M, x)) == b
    assert energy == dot(x, mv(M, x))
    assert tau == m*dot(eta, mv(M, eta))
    assert (energy == 0) == all(v == 0 for v in b) == (eta == u)
    row_sums = [sum(row) for row in M]
    assert (len(set(row_sums)) == 1) == (eta == u)
    if reference is not None:
        assert tau == F(reference['mfpt'])
        assert eta == [F(reference['eta_full'][j]) for j in states]
        assert tau_stat == F(reference['stationary_substitution_baseline']['tau_stat'])
    # Rational certified eigenvalue enclosure on V=1-perp. S acts as T on V
    # and as identity on constants; symmetric spectral radius <= infinity norm.
    Tfull = mm(mm(P, M), P)
    S = [[Tfull[i][j]+F(1, r) for j in range(r)] for i in range(r)]
    Si = inv(S)
    trace_T = sum(Tfull[i][i] for i in range(r))
    trace_Tinv = sum(Si[i][i] for i in range(r))-1
    lam_lower = max(1/norm_inf(Si), 1/trace_Tinv)
    lam_upper = min(norm_inf(S), trace_T)
    b2 = dot(b, b)
    bias_lower, bias_upper = m*b2/lam_upper, m*b2/lam_lower
    assert bias_lower <= m*energy <= bias_upper
    # One-dimensional energy minimization along b. This uses M and b only.
    if b2:
        omega = b2/dot(b, mv(M, b))
        x1 = [omega*v for v in b]
    else:
        omega = F(0)
        x1 = [F(0)]*r
    rho1 = minus(u, x1)
    e1 = minus(b, mv(Tfull, x1))
    tau1 = m*dot(rho1, mv(M, rho1))
    remainder = tau1-tau
    assert remainder == m*dot(e1, mv(Si, e1))
    residual_lower = m*dot(e1, e1)/lam_upper
    residual_upper = m*dot(e1, e1)/lam_lower
    assert residual_lower <= remainder <= residual_upper
    mean_interval = [tau1-residual_upper, tau1-residual_lower]
    assert mean_interval[0] <= tau <= mean_interval[1]
    Mf = np.array([[float(v) for v in row] for row in M])
    # Orthonormal Helmert basis: stable numerical spectral description only.
    Z = np.zeros((r, r-1))
    for j in range(r-1):
        Z[:j+1, j] = 1/np.sqrt((j+1)*(j+2))
        Z[j+1, j] = -(j+1)/np.sqrt((j+1)*(j+2))
    eigs = np.linalg.eigvalsh(Z.T @ Mf @ Z)
    assert float(lam_lower) <= eigs[0]*(1+1e-12)
    assert eigs[-1] <= float(lam_upper)*(1+1e-12)
    return {
        'n': n, 'gamma': str(gamma), 'open_states': states,
        'row_sums': [str(v) for v in row_sums],
        'row_sum_residual_b': [str(v) for v in b],
        'uniformity_defect_x': [str(v) for v in x],
        'eta': [str(v) for v in eta],
        'row_sums_constant': len(set(row_sums)) == 1,
        'eta_uniform': eta == u, 'stationary_substitution_exact': energy == 0,
        'tau': str(tau), 'tau_stat': str(tau_stat),
        'correction_energy': str(energy), 'bias': str(m*energy),
        'bias_relative_to_exact_mean': str(m*energy/tau),
        'certified_eigenvalue_lower': str(lam_lower),
        'certified_eigenvalue_upper': str(lam_upper),
        'certified_bias_interval': [str(bias_lower), str(bias_upper)],
        'one_direction_correction': {
            'omega': str(omega), 'candidate_rho': [str(v) for v in rho1],
            'candidate_is_probability': all(v >= 0 for v in rho1),
            'tau_candidate': str(tau1),
            'residual': [str(v) for v in e1],
            'exact_remaining_error': str(remainder),
            'certified_remaining_error_interval': [str(residual_lower), str(residual_upper)],
            'certified_mean_interval': [str(v) for v in mean_interval],
            'bias_fraction_removed': str((tau_stat-tau1)/(m*energy)) if energy else None,
        },
        'floating_spectral_description': {'lambda_min': float(eigs[0]), 'lambda_max': float(eigs[-1])},
        'checks': {'all_exact_identities': True, 'certified_bounds_contain_exact_values': True,
                   'reference_exact_mean_and_eta_match': reference is not None},
    }

def float_model(n, states, gamma):
    """Independent direct full killed matrix; no Green matrix used here."""
    m = 4*n
    L = np.zeros((m, m))
    for x in range(4):
        for j in range(n):
            a = x*n+j
            L[a, a] = -1-gamma
            for delta in (-1, 1):
                L[a, ((x+delta) % 4)*n+j] += .5
                L[a, x*n+(j+delta) % n] += gamma*.5
    R = np.array(states)
    U = np.array([i for i in range(m) if i not in states])
    K = -L[np.ix_(U, U)]
    h = floating_solve(K, np.ones(len(U)), assume_a='pos')
    eta = np.ones(len(R))/m + L[np.ix_(U, R)].T @ h/m
    tau = h.sum()/m
    g = np.zeros(n)
    g[R] = 1
    Qg = -.5*(2*g-np.roll(g, 1)-np.roll(g, -1))
    return {'tau': float(tau),
            'selected_return_derivative': float(gamma*eta @ Qg[R]),
            'killed_equation_residual_max': float(np.max(np.abs(K @ h-1))),
            'eta_normalization_residual': float(abs(eta.sum()-1)),
            'eta_min': float(eta.min())}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-scan', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=HERE,
                        help='Directory for fresh output; use this to preserve frozen results.')
    args = parser.parse_args()
    if not __debug__:
        raise SystemExit('Do not use Python -O: exact assertions must remain active.')
    started = time.monotonic()
    witnesses_path = ROOT/'reproducible'/'witnesses.json'
    reference_path = ROOT/'reaction_selection'/'results'/'exact_reaction_selection.json'
    witnesses = json.loads(witnesses_path.read_text(encoding='utf-8'))
    refs = {v['id']: v for v in json.loads(reference_path.read_text(encoding='utf-8'))['pairs']}
    out = {'schema': 'positive-criterion-v1',
           'scope': 'Finite symmetric product CTMC; exact gamma=1; finite-grid floating scan explicitly separate.',
           'exact_arithmetic': 'Python standard-library Fraction; no SymPy or project solver imported.',
           'provenance': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (witnesses_path, reference_path)}},
           'pairs': []}
    for witness in witnesses:
        wid, n = witness['id'], witness['n']
        models = {s: exact_model(n, witness[s], refs[wid][s]) for s in ('A', 'B')}
        a, b = models['A'], models['B']
        assert a['tau_stat'] == b['tau_stat']
        gap = F(a['tau'])-F(b['tau'])
        correction_gap = F(b['correction_energy'])-F(a['correction_energy'])
        assert gap == 4*n*correction_gap
        assert (gap == 0) == (correction_gap == 0)
        out['pairs'].append({'id': wid, **models, 'tau_A_minus_B': str(gap),
                             'correction_energy_B_minus_A': str(correction_gap),
                             'common_stationary_prediction_exact': True,
                             'response_equality_iff_correction_energy_equal': True,
                             'relative_half_gap_over_pair_mean': str(abs(gap)/(F(a['tau'])+F(b['tau'])))})
        print(wid, 'exact PASS', 'gap', float(gap), 'one-mode fractions',
              *[float(F(models[s]['one_direction_correction']['bias_fraction_removed'])) for s in ('A', 'B')], flush=True)
    out['alternating_control'] = exact_model(4, [0, 2])
    assert out['alternating_control']['stationary_substitution_exact']
    assert F(out['alternating_control']['tau']) == F(47, 12)
    if not args.skip_scan:
        rates = [.01, .03, .1, .3, 1., 3., 10., 30., 100.]
        out['floating_finite_rate_scan'] = {'rates': rates,
            'scope': 'Exploratory finite gamma grid only: no interval certification, no asymptotic or between-grid statement.',
            'pairs': []}
        for witness, exact in zip(witnesses, out['pairs']):
            rows = []
            for gamma in rates:
                a = float_model(witness['n'], witness['A'], gamma)
                b = float_model(witness['n'], witness['B'], gamma)
                gap = a['tau']-b['tau']
                row = {'gamma': gamma, 'A': a, 'B': b, 'tau_A_minus_B': gap,
                       'relative_half_gap_over_pair_mean': abs(gap)/(a['tau']+b['tau']),
                       'selected_return_derivative_A_minus_B': a['selected_return_derivative']-b['selected_return_derivative']}
                if gamma == 1.:
                    assert abs(a['tau']-float(F(exact['A']['tau']))) < 1e-10
                    assert abs(b['tau']-float(F(exact['B']['tau']))) < 1e-10
                rows.append(row)
            out['floating_finite_rate_scan']['pairs'].append({'id': witness['id'], 'rows': rows})
    out['summary'] = {'exact_gates': 14, 'exact_controls': 1, 'all_exact_assertions_pass': True,
                      'seconds': time.monotonic()-started}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir/'positive_criterion_evidence.json'
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('WROTE', output, 'seconds', round(out['summary']['seconds'], 3), flush=True)

if __name__ == '__main__':
    main()
