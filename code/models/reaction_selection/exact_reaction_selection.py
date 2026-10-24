"""Exact finite-model diagnostic of reaction-selected gate return probabilities.

Independent implementation: no imports from the existing scientific solvers.
Only rational SymPy arithmetic is used.  No continuum/fast-gating claim follows
from these finite gamma=1 calculations.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import sympy as sp

ROOT = Path(__file__).resolve().parent

def rational_text(value):
    return str(sp.cancel(value))

def exact_vector(values):
    return [rational_text(v) for v in values]

def cycle_generator(n):
    if n < 3:
        raise ValueError("This implementation requires an ordinary cycle n>=3.")
    q = sp.zeros(n)
    for j in range(n):
        q[j, j] = -1
        q[j, (j - 1) % n] += sp.Rational(1, 2)
        q[j, (j + 1) % n] += sp.Rational(1, 2)
    return q

def product_generator(n):
    """Assemble row generator explicitly, independently of Kronecker code."""
    L = sp.zeros(4 * n)
    for x in range(4):
        for j in range(n):
            i = x * n + j
            L[i, i] = -2
            L[i, ((x - 1) % 4) * n + j] += sp.Rational(1, 2)
            L[i, ((x + 1) % 4) * n + j] += sp.Rational(1, 2)
            L[i, x * n + (j - 1) % n] += sp.Rational(1, 2)
            L[i, x * n + (j + 1) % n] += sp.Rational(1, 2)
    assert L == L.T
    assert L * sp.ones(4 * n, 1) == sp.zeros(4 * n, 1)
    return L

@lru_cache(maxsize=None)
def stationary_green(n):
    """Classical finite symmetric-chain Green matrix, shared by each n."""
    L = product_generator(n)
    N = 4 * n
    P = sp.ones(N) / N
    G = (-L + P).inv(method="DM") - P
    assert G == G.T
    assert G * sp.ones(N, 1) == sp.zeros(N, 1)
    assert (-L) * G == sp.eye(N) - P
    return G

def first_nonzero(rows, field):
    for row in rows:
        if sp.Rational(row[field]) != 0:
            return {"k": row["k"], "gap": row[field]}
    return None

def model(n, open_states):
    R = tuple(sorted(open_states))
    if not R or len(R) == n or len(set(R)) != len(R):
        raise ValueError("Use a nonconstant binary mask with distinct states.")
    Q = cycle_generator(n)
    L = product_generator(n)
    absorbing = [j for j in R]  # Index x*n+j, with x=0.
    keep = [i for i in range(4 * n) if i not in set(absorbing)]
    K = -L.extract(keep, keep)
    aT = sp.ones(len(keep), 1) / (4 * n)
    # Exact domain-matrix inverse; forward and backward residuals below are exact.
    inverse = K.inv(method="DM")
    occupation = inverse.T * aT
    h = inverse * sp.ones(len(keep), 1)
    assert K.T * occupation == aT
    assert K * h == sp.ones(len(keep), 1)
    assert occupation == h / (4 * n)
    assert all(z >= 0 for z in occupation)
    mfpt = sp.cancel(sum(occupation))
    assert mfpt == sp.cancel(sum(h) / (4 * n))
    flux = L.extract(keep, absorbing).T * occupation
    initial_atom = sp.Rational(len(R), 4 * n)
    eta = sp.zeros(n, 1)
    eta_plus = sp.zeros(n, 1)
    u = sp.zeros(n, 1)
    for col, j in enumerate(R):
        eta[j] = sp.cancel(sp.Rational(1, 4 * n) + flux[col])
        eta_plus[j] = sp.cancel(flux[col] / (1 - initial_atom))
        u[j] = sp.Rational(1, len(R))
    assert sum(eta) == 1
    assert sum(eta_plus) == 1
    assert all(z >= 0 for z in eta)
    assert all(z >= 0 for z in eta_plus)
    assert eta == initial_atom * u + (1 - initial_atom) * eta_plus
    # Cross-check the occupation solution using the independent full-chain
    # stationary Green matrix. eta is not solved from this capacity identity.
    M = stationary_green(n).extract(absorbing, absorbing)
    eta_R = eta.extract(R, [0])
    u_R = u.extract(R, [0])
    N = 4 * n
    assert M * eta_R == (mfpt / N) * sp.ones(len(R), 1)
    capacity_mfpt = sp.cancel(N * (eta_R.T * M * eta_R)[0])
    tau_stat = sp.cancel(N * (u_R.T * M * u_R)[0])
    bias = sp.cancel(tau_stat - mfpt)
    d_R = u_R - eta_R
    bias_square = sp.cancel(N * (d_R.T * M * d_R)[0])
    assert capacity_mfpt == mfpt
    assert bias == bias_square
    assert bias >= 0
    assert (bias == 0) == (eta == u)
    g = sp.Matrix([int(j in R) for j in range(n)])
    power_g = g
    derivatives = []
    for k in range(n):
        values = [(vec.T * power_g)[0] for vec in [eta, eta_plus, u]]
        derivatives.append({
            "k": k,
            "eta": rational_text(values[0]),
            "eta_positive_time": rational_text(values[1]),
            "uniform_open": rational_text(values[2]),
            "eta_minus_uniform": rational_text(values[0] - values[2]),
            "eta_positive_time_minus_uniform": rational_text(values[1] - values[2]),
        })
        power_g = Q * power_g
    first = first_nonzero(derivatives, "eta_minus_uniform")
    first_plus = first_nonzero(derivatives, "eta_positive_time_minus_uniform")
    # Since the initial atom has the uniform-open gate distribution, it cannot
    # create an eta-vs-u difference and only rescales every return discrepancy.
    for row in derivatives:
        assert sp.Rational(row["eta_minus_uniform"]) == (
            (1 - initial_atom) * sp.Rational(row["eta_positive_time_minus_uniform"])
        )
    assert (first is None) == (first_plus is None)
    if first is not None:
        assert first["k"] == first_plus["k"]
    full_occupation = sp.zeros(4 * n, 1)
    full_h = sp.zeros(4 * n, 1)
    for i, state in enumerate(keep):
        full_occupation[state] = occupation[i]
        full_h[state] = h[i]
    result = {
        "n": n,
        "open_states": list(R),
        "p": rational_text(sp.Rational(len(R), n)),
        "initial_absorption_mass": rational_text(initial_atom),
        "positive_time_mass": rational_text(1 - initial_atom),
        "eta_full": exact_vector(eta),
        "eta_positive_time_full": exact_vector(eta_plus),
        "uniform_open_full": exact_vector(u),
        "positive_time_flux_on_R": exact_vector(flux),
        "transient_indices_xn_plus_j": keep,
        "occupation_transient": exact_vector(occupation),
        "backward_mfpt_transient": exact_vector(h),
        "mfpt": rational_text(mfpt),
        "stationary_substitution_baseline": {
            "tau_stat": rational_text(tau_stat),
            "bias": rational_text(bias),
            "relative_bias": rational_text(bias / mfpt),
            "bias_tau_stat_minus_exact": rational_text(bias),
            "relative_bias_over_exact_mfpt": rational_text(bias / mfpt),
            "capacity_mfpt": rational_text(capacity_mfpt),
            "bias_square_value": rational_text(bias_square),
            "capacity_vector_identity": True,
            "capacity_identity": True,
            "bias_square_identity": True,
            "bias_is_zero_iff_eta_is_uniform_open": True,
            "green_normalization": "N=4n; P=11^T/N; G=(-L+P)^(-1)-P; M=G_BB",
            "scope": "Classical finite-chain identity; stationary substitution diagnostic, not a new capacity formula.",
        },
        "future_open_derivatives": derivatives,
        "first_eta_vs_uniform_difference": first,
        "first_eta_positive_time_vs_uniform_difference": first_plus,
        "eta_is_uniform_open": eta == u,
        "eta_positive_time_is_uniform_open": eta_plus == u,
        "same_future_open_return_as_uniform_open": first is None,
        "checks": {
            "generator_symmetric": True,
            "generator_row_sums_zero": True,
            "forward_equation_exact": True,
            "backward_equation_exact": True,
            "occupation_equals_backward_vector_divided_by_4n": True,
            "occupation_nonnegative": True,
            "eta_nonnegative_normalized": True,
            "eta_positive_time_nonnegative_normalized": True,
            "initial_atom_mixture_exact": True,
            "eta_gap_is_positive_time_gap_times_positive_time_mass": True,
        },
    }
    return result, {"eta": eta, "eta_plus": eta_plus, "occupation": full_occupation,
                    "h": full_h, "mfpt": mfpt}

def compare_models(a, b):
    assert a["n"] == b["n"]
    rows = []
    for da, db in zip(a["future_open_derivatives"], b["future_open_derivatives"]):
        row = {"k": da["k"]}
        for key in ["eta", "eta_positive_time", "uniform_open"]:
            row[f"{key}_A_minus_B"] = rational_text(
                sp.Rational(da[key]) - sp.Rational(db[key]))
        rows.append(row)
    return {
        "mfpt_A_minus_B": rational_text(sp.Rational(a["mfpt"]) - sp.Rational(b["mfpt"])),
        "between_gate_future_return_derivatives": rows,
        "first_between_gate_eta_return_difference": first_nonzero(rows, "eta_A_minus_B"),
        "first_between_gate_eta_positive_time_return_difference":
            first_nonzero(rows, "eta_positive_time_A_minus_B"),
        "first_between_gate_uniform_open_return_difference":
            first_nonzero(rows, "uniform_open_A_minus_B"),
        "identical_stat_prediction": (
            a["stationary_substitution_baseline"]["tau_stat"]
            == b["stationary_substitution_baseline"]["tau_stat"]),
    }

def rotation_control(n, R, base, raw, shift=3):
    rotated_R = sorted((j + shift) % n for j in R)
    rotated, rotated_raw = model(n, rotated_R)
    eta_ok = all(rotated_raw["eta"][(j + shift) % n] == raw["eta"][j] for j in range(n))
    plus_ok = all(rotated_raw["eta_plus"][(j + shift) % n] == raw["eta_plus"][j] for j in range(n))
    occupation_ok = all(
        rotated_raw["occupation"][x * n + (j + shift) % n]
        == raw["occupation"][x * n + j]
        for x in range(4) for j in range(n))
    h_ok = all(
        rotated_raw["h"][x * n + (j + shift) % n] == raw["h"][x * n + j]
        for x in range(4) for j in range(n))
    checks = {
        "eta_relabel_exact": eta_ok,
        "eta_positive_time_relabel_exact": plus_ok,
        "occupation_relabel_exact": occupation_ok,
        "backward_mfpt_vector_relabel_exact": h_ok,
        "mfpt_invariant": rotated["mfpt"] == base["mfpt"],
        "all_return_derivatives_invariant": rotated["future_open_derivatives"] == base["future_open_derivatives"],
        "stationary_substitution_baseline_invariant":
            rotated["stationary_substitution_baseline"] == base["stationary_substitution_baseline"],
    }
    assert all(checks.values())
    return {"shift": shift, "rotated_open_states": rotated_R, "checks": checks}

def main():
    if not __debug__:
        raise SystemExit("Run without -O so all exact validation assertions remain enabled.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witnesses", type=Path,
                        default=ROOT.parent / "reproducible" / "witnesses.json")
    parser.add_argument("--reference-exact", type=Path,
                        default=ROOT.parents[2] / "data" / "figures" / "raw" / "round1" / "exact.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "exact_reaction_selection.json")
    args = parser.parse_args()
    started = time.monotonic()
    witnesses = json.loads(args.witnesses.read_text(encoding="utf-8"))
    references = {r["id"]: r for r in json.loads(args.reference_exact.read_text(encoding="utf-8"))}
    output = {
        "schema": "exact-reaction-selection-v1",
        "purpose": "Finite gamma=1 mechanism diagnostic; not a continuum ell or fast-gating proof.",
        "model": {
            "spatial_states": 4,
            "space_generator": "4-cycle, each neighboring jump has rate 1/2",
            "hidden_generator": "n-cycle Q, each neighboring jump has rate 1/2",
            "gamma": "1",
            "absorbing_set": "{(0,j): j in R}",
            "initial_law": "uniform on all 4n states, including immediate absorption",
            "row_generator_ordering": "x*n+j",
            "occupation_formula": "v=K^(-T) a_T; K=-L_TT",
            "eta_formula": "eta_j=1/(4n)+sum_i v_i L[i,(0,j)] for j in R; zero otherwise",
            "eta_positive_time_formula": "eta_plus_j=flux_j/(1-|R|/(4n))",
            "derivative_convention": "q_rho^(k)(0)=rho^T Q^k g, k=0,...,n-1",
        },
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "witnesses": str(args.witnesses.resolve()),
            "witnesses_sha256": hashlib.sha256(args.witnesses.read_bytes()).hexdigest(),
            "reference_exact": str(args.reference_exact.resolve()),
            "reference_exact_sha256": hashlib.sha256(args.reference_exact.read_bytes()).hexdigest(),
            "python": sys.version,
            "sympy": sp.__version__,
            "platform": platform.platform(),
            "exact_arithmetic": True,
            "existing_scientific_solver_imported_or_executed": False,
        },
        "pairs": [],
    }
    for witness in witnesses:
        wid = witness["id"]
        n = witness["n"]
        a, ar = model(n, witness["A"])
        b, br = model(n, witness["B"])
        ref = references[wid]
        assert a["open_states"] == ref["A"]
        assert b["open_states"] == ref["B"]
        assert [a["mfpt"], b["mfpt"]] == ref["finite_means"]
        pair = {"id": wid, "A": a, "B": b, **compare_models(a, b)}
        assert pair["mfpt_A_minus_B"] == ref["finite_gap"]
        assert pair["first_between_gate_uniform_open_return_difference"] is None
        assert pair["identical_stat_prediction"]
        pair["reference_comparison"] = {
            "both_mfpts_equal_existing_four_point_exact": True,
            "gap_equals_existing_four_point_exact": True,
            "reference_has_occupation_vector": False,
            "occupation_validation": "K^T v=a_T exactly and v=h/(4n), Kh=1; old exact.json stores only averaged MFPT",
            "reference_C2_C3_dwell_match_flags": [ref[k] for k in ["C2_equal", "C3_equal", "dwell_equal"]],
        }
        zero = compare_models(a, a)
        assert zero["mfpt_A_minus_B"] == "0"
        assert all(v == "0" for row in zero["between_gate_future_return_derivatives"]
                   for k, v in row.items() if k != "k")
        swapped = compare_models(b, a)
        assert sp.Rational(swapped["mfpt_A_minus_B"]) == -sp.Rational(pair["mfpt_A_minus_B"])
        for normal_row, swap_row in zip(pair["between_gate_future_return_derivatives"],
                                        swapped["between_gate_future_return_derivatives"]):
            for key in normal_row:
                if key != "k":
                    assert sp.Rational(swap_row[key]) == -sp.Rational(normal_row[key])
        pair["controls"] = {
            "zero_identical_gate_comparison_all_exact_zero": True,
            "swap_comparison_all_signs_reverse_exactly": True,
            "rotation_A": rotation_control(n, witness["A"], a, ar),
            "rotation_B": rotation_control(n, witness["B"], b, br),
        }
        output["pairs"].append(pair)
        print(wid, "PASS", "A first", a["first_eta_vs_uniform_difference"],
              "B first", b["first_eta_vs_uniform_difference"], flush=True)
    positive, _ = model(4, [0, 2])
    assert positive["eta_is_uniform_open"]
    assert positive["eta_positive_time_is_uniform_open"]
    assert positive["same_future_open_return_as_uniform_open"]
    output["multiple_open_valid_factor_control"] = {
        "description": "Alternating open mask on C4 is strongly lumpable; two open states.",
        "open_to_closed_rate_each_open_state": "1",
        "closed_to_open_rate_each_closed_state": "1",
        "result": positive,
    }
    output["summary"] = {
        "pair_count": len(output["pairs"]),
        "gate_count": 2 * len(output["pairs"]),
        "all_reference_mfpts_match": True,
        "all_normalization_residual_mixture_and_symmetry_checks_pass": True,
        "gate_eta_vs_uniform_first_k": {
            p["id"]: {s: p[s]["first_eta_vs_uniform_difference"] for s in ["A", "B"]}
            for p in output["pairs"]
        },
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", args.output, "seconds", round(output["summary"]["seconds"], 3), flush=True)

if __name__ == "__main__":
    main()
