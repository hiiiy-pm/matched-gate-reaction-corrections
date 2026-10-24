"""Exact bounded interpolation certificate for a finite CTMC response identity.

For C18_0220, K(gamma) is 63x63 and affine in gamma. Hence the cross-multiplied
difference of two mean hitting-time rational functions has degree at most125.
Equality at126 distinct positive integer rates proves equality for all positive
rates, since both killed matrices are positive definite there. The degree bound
is deliberately crude and does not rely on numerical polynomial fitting.
"""
from __future__ import annotations

from fractions import Fraction as F
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from positive_criterion import ROOT, HERE, gate_green, inv, mv, dot

def mean_and_law(n, states, gamma):
    G = gate_green(n, gamma)
    M = [[G[i][j] for j in states] for i in states]
    h = mv(inv(M), [F(1)]*len(states))
    c = sum(h)
    eta = [v/c for v in h]
    tau = F(4*n)/c
    assert mv(M, eta) == [tau/(4*n)]*len(states)
    assert sum(eta) == 1 and all(v >= 0 for v in eta)
    g = [F(j in states) for j in range(n)]
    Qg = [-g[j]+(g[(j-1) % n]+g[(j+1) % n])/2 for j in range(n)]
    derivative = gamma*dot(eta, [Qg[j] for j in states])
    return tau, derivative

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=HERE,
                        help='Directory for fresh output; use this to preserve frozen results.')
    args = parser.parse_args()
    if not __debug__:
        raise SystemExit('Run without -O so exact checks remain enabled.')
    started = time.monotonic()
    witness_path = ROOT/'reproducible'/'witnesses.json'
    witness = next(v for v in json.loads(witness_path.read_text(encoding='utf-8')) if v['id'] == 'C18_0220')
    n = witness['n']
    d = 4*n-len(witness['A'])
    assert len(witness['A']) == len(witness['B'])
    degree_bound = 2*d-1
    rows = []
    for gamma in range(1, degree_bound+2):
        a, da = mean_and_law(n, witness['A'], F(gamma))
        b, db = mean_and_law(n, witness['B'], F(gamma))
        assert a == b, (gamma, a, b)
        rows.append({'gamma': str(gamma), 'tau_A': str(a), 'tau_B': str(b),
                     'exact_difference': str(a-b),
                     'selected_return_first_derivative_A_minus_B': str(da-db)})
        if gamma % 20 == 0:
            print('exact interpolation nodes verified', gamma, flush=True)
    assert F(rows[0]['selected_return_first_derivative_A_minus_B']) != 0
    out = {'schema': 'finite-equal-mean-family-certificate-v1',
           'claim': 'For the finite four-site spatial control and C18_0220, tau_A(gamma)=tau_B(gamma) for every gamma>0.',
           'scope': 'Finite CTMC only, uniform full initial law including immediate absorption; not the periodic continuum model.',
           'witness': witness, 'transient_dimension': d,
           'degree_argument': {'matrix_entries_degree_at_most': 1,
                               'mean_numerator_degree_at_most': d-1,
                               'mean_denominator_degree_at_most': d,
                               'cross_product_difference_degree_at_most': degree_bound,
                               'required_distinct_nodes': degree_bound+1,
                               'all_nodes_positive_and_distinct': True,
                               'denominators_nonzero': 'K(gamma) is SPD for every gamma>0 by irreducibility and nonempty absorbing set.'},
           'rows': rows, 'all_exact_equalities_pass': True,
           'return_laws_differ_at_gamma_one': True,
           'seconds': time.monotonic()-started,
           'provenance': {'python': sys.version, 'arithmetic': 'Python Fraction',
                          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'helper_sha256': hashlib.sha256((HERE/'positive_criterion.py').read_bytes()).hexdigest(),
                          'witnesses_sha256': hashlib.sha256(witness_path.read_bytes()).hexdigest()}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir/'equal_mean_family_certificate.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('WROTE', path, 'seconds', round(out['seconds'], 3), flush=True)

if __name__ == '__main__':
    main()
