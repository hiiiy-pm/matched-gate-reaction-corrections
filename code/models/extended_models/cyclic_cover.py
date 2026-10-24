"""Cyclic-cover construction for C18_0064 on Z_36.

Lift each 18-bit mask by repeating it twice.  Verify exact p, C2, C3 and
run-length matching with integer arithmetic.  The local-coefficient
certificate is run separately with interval_engine_arb.py.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
w = json.load(open(ROOT / "reproducible/witnesses.json", encoding="utf-8"))
base = [z for z in w if z["id"] == "C18_0064"][0]
n = base["n"]
A = list(base["A"])
B = list(base["B"])
A36 = sorted(set(A) | {i + n for i in A})
B36 = sorted(set(B) | {i + n for i in B})

def runs(R, n):
    R = set(R)
    out = []
    for i in range(n):
        if i in R and (i - 1) % n not in R:
            m = 0
            j = i
            while j in R:
                m += 1
                j = (j + 1) % n
            out.append(m)
    return sorted(out)

def signature(n, R):
    g = [1 if i in R else 0 for i in range(n)]
    # M = 2Q has integer entries: -2 on the diagonal, 1 on the neighbours.
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = -2
        M[i][(i - 1) % n] = 1
        M[i][(i + 1) % n] = 1
    # u_k = M^k g
    u = [g]
    for _ in range(1, n):
        prev = u[-1]
        u.append([sum(M[i][j] * prev[j] for j in range(n)) for i in range(n)])
    c2 = [sum(g[i] * u[k][i] for i in range(n)) for k in range(n)]
    c3 = [[sum(g[i] * u[k][i] * u[l][i] for i in range(n)) for l in range(n)] for k in range(n)]
    return c2, c3

c2A, c3A = signature(36, A36)
c2B, c3B = signature(36, B36)
print("A36", A36)
print("B36", B36)
print("p", len(A36) / 36, len(B36) / 36)
print("run lengths A", runs(A36, 36), "B", runs(B36, 36))
print("C2 equal", c2A == c2B)
print("C3 equal", c3A == c3B)
print("first C2", c2A[:6])
print("C2 diff nonzero", sum(1 for a, b in zip(c2A, c2B) if a != b))
print("C3 diff nonzero", sum(1 for a, b in zip(c3A, c3B) for x, y in zip(a, b) if x != y))
out = {
    "n": 36,
    "parent": "C18_0064",
    "A36": A36,
    "B36": B36,
    "p": [len(A36) / 36, len(B36) / 36],
    "run_lengths_A": runs(A36, 36),
    "run_lengths_B": runs(B36, 36),
    "C2_equal": c2A == c2B,
    "C3_equal": c3A == c3B,
    "C2_first": c2A[:6],
    "C3_nonzero_differences": sum(1 for a, b in zip(c3A, c3B) for x, y in zip(a, b) if x != y),
}
(ROOT / "extended_models/cyclic_cover_matching.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote", ROOT / "extended_models/cyclic_cover_matching.json")
