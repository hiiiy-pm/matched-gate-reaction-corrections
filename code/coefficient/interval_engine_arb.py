"""Arb (python-flint) re-implementation of the local-coefficient interval engine.

This is an independent arithmetic backend for the 698/360/500-segment
certificate.  It follows the same analytic formulas as
``interval_engine.py`` but evaluates them with ``arb`` balls instead of
``mpmath.iv`` intervals.  It is a cross-check, not a new proof of the
analytic inequalities.
"""
from __future__ import annotations
import hashlib, json, math, platform, sys, time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import flint
from flint import arb, arb_mat, ctx

ROOT = Path(__file__).resolve().parent
CERTIFICATE_SCHEMA = "arb-local-v14-outward-1"


def endpoint_fraction(endpoint):
    """The exact dyadic value returned by Arb.lower/upper (no decimal roundtrip)."""
    if not endpoint.is_exact() or not endpoint.is_finite():
        raise ValueError("a finite exact Arb endpoint is required")
    mantissa, exponent = map(int, endpoint.man_exp())
    return (Fraction(mantissa << exponent) if exponent >= 0
            else Fraction(mantissa, 1 << -exponent))


def outward_float(x, upper=False):
    """Enclose x in both binary64 and the decimal JSON spelling of that float.

    Endpoints are extracted by directed Arb rounding, converted via an exact
    dyadic Fraction, and moved one binary64 ulp outward.  The two exact-rational
    assertions cover the in-memory binary64 and json.dumps' shortest-decimal
    representation separately.  Float conversion is never used in arithmetic.
    """
    q = endpoint_fraction(x.upper() if upper else x.lower())
    f = math.nextafter(float(q), math.inf if upper else -math.inf)
    if not math.isfinite(f):
        raise ArithmeticError("nonfinite serialized interval endpoint")
    binary = Fraction.from_float(f)
    decimal = Fraction(str(f))
    if upper:
        assert binary >= q and decimal >= q
    else:
        assert binary <= q and decimal <= q
    return f


def bounds(x):
    return [outward_float(x), outward_float(x, upper=True)]


def interval_hull(lower, upper):
    """Arb ball enclosing two exact or enclosed endpoints."""
    return arb(lower).union(arb(upper))


def signed_quotient(numerator, denominator):
    """Full interval division; endpoint pairing must depend on numerator sign."""
    if not denominator > 0:
        raise ValueError("the coefficient denominator must be strictly positive")
    return numerator / denominator


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def isum(xs):
    return sum(xs, arb(0))


def dot(x, y):
    return isum(a * b for a, b in zip(x, y))


def hi(x):
    return arb(x.upper())


def lo(x):
    return arb(x.lower())


def upper_abs(x):
    return arb(abs(x).upper())


def runs(C, n):
    out = []
    cs = set(C)
    for j in C:
        if (j - 1) % n not in cs:
            r = []
            k = j
            while k in cs:
                r.append(k)
                k = (k + 1) % n
            out.append(r)
    assert sorted(j for r in out for j in r) == C
    return out


class Model:
    def __init__(self, R, n, prec=200):
        ctx.prec = prec
        self.R = R
        self.n = n
        self.C = [j for j in range(n) if j not in R]
        nc = len(self.C)
        self.runs = runs(self.C, n)
        self.ring = []
        self.lam = []
        self.ra = []
        for k in range(1, n // 2 + 1):
            angle = Fraction(2 * k, n)
            for kind in ([0] if 2 * k == n else [0, 1]):
                norm = (arb(1) / arb(n)).sqrt() if 2 * k == n else (arb(2) / arb(n)).sqrt()
                fn = arb.cos if kind == 0 else arb.sin
                self.ring.append([
                    norm * fn(arb.pi() * arb(2 * k * j) / arb(n))
                    for j in range(n)
                ])
                self.ra.append(angle)
                self.lam.append(
                    (1 - arb.cos(arb.pi() * arb(angle.numerator) / arb(angle.denominator))).sqrt()
                )
        self.path = []
        self.mu = []
        self.pa = []
        for block in self.runs:
            for k in range(1, len(block) + 1):
                a = Fraction(k, len(block) + 1)
                v = [arb(0) for _ in range(n)]
                for pos, j in enumerate(block, 1):
                    v[j] = (arb(2) / arb(len(block) + 1)).sqrt() * arb.sin(
                        arb.pi() * arb(k * pos) / arb(len(block) + 1)
                    )
                self.path.append(v)
                self.pa.append(a)
                self.mu.append(
                    (1 - arb.cos(arb.pi() * arb(a.numerator) / arb(a.denominator))).sqrt()
                )

        mat = []
        for i in self.C:
            row = []
            for j in self.C:
                row.append(
                    isum(l * v[i] * v[j] for l, v in zip(self.lam, self.ring))
                    + isum(l * v[i] * v[j] for l, v in zip(self.mu, self.path))
                )
            mat.append(row)
        A = arb_mat(mat)
        rhs = arb_mat([[arb(1)] for _ in self.C])
        xc = A.solve(rhs)

        self.b = [arb(0) for _ in range(n)]
        for idx, j in enumerate(self.C):
            self.b[j] = xc[idx, 0]

        self.beta = [dot(v, self.b) for v in self.ring]
        self.a = [dot(v, self.b) for v in self.path]
        self.cross = [[dot(u, v) for v in self.path] for u in self.ring]
        self.lam0 = 1 - arb.cos(arb.pi() / arb(max(map(len, self.runs)) + 1))
        self.q = arb(nc) / arb(n)

        grp = {}
        for k, a in enumerate(self.ra):
            grp.setdefault(("r", a), []).append(k)
        for k, a in enumerate(self.pa):
            grp.setdefault(("p", a), []).append(k)
        U = arb(0)
        V = arb(0)
        for (typ, a), ks in grp.items():
            coeff = self.beta if typ == "r" else self.a
            lam = self.lam if typ == "r" else self.mu
            s = isum(abs(coeff[k]) ** 2 for k in ks)
            amp = arb(s.upper()).sqrt()
            U += amp
            V += lam[ks[0]] * amp
        jump = []
        for j in range(n):
            gp = -isum(l * b * v[j] for l, b, v in zip(self.lam, self.beta, self.ring))
            gm = isum(l * a * v[j] for l, a, v in zip(self.mu, self.a, self.path))
            jump.append(gp - gm)
        s = isum(abs(x) ** 2 for x in jump)
        V += arb(s.upper()).sqrt()
        self.U = hi(U)
        self.V = hi(V)
        self.gate = (
            isum(b * b * l for b, l in zip(self.beta, self.lam))
            + isum(a * a * m for a, m in zip(self.a, self.mu))
        ) / (2 * n)
        self.brownian = isum(self.b[j] for j in self.C) / n
        self.meta = {
            "closed_runs": self.runs,
            "U": str(self.U),
            "V": str(self.V),
            "brownian": [str(self.brownian.lower()), str(self.brownian.upper())],
        }

    def trial(self, s):
        rho = 2 * s - 1
        diagonal = isum(b * b * l ** rho for b, l in zip(self.beta, self.lam)) + isum(
            a * a * m ** rho for a, m in zip(self.a, self.mu)
        )
        mixed = arb(0)
        for i, l in enumerate(self.lam):
            for j, m in enumerate(self.mu):
                if self.ra[i] == self.pa[j]:
                    div = (1 - rho) * l ** (rho - 2)
                else:
                    div = (l ** (rho - 1) - m ** (rho - 1)) / (m - l)
                mixed += l * m * self.beta[i] * self.a[j] * self.cross[i][j] * div
        kinetic = (diagonal - 2 * mixed) / (2 * self.n * (-arb.cos(arb.pi() * s)))
        source = arb.gamma(1 + s) * isum(
            isum(v) * a * m ** (s - 1) for v, a, m in zip(self.path, self.a, self.mu)
        ) / self.n
        return 2 * source - kinetic - self.gate

    def residue_square(self, s, smin):
        delta = 1 - smin
        ar = arb.gamma(1 + s) / arb.gamma(2 - s)
        Bs = 1 / (arb.gamma(2 * s) * (-arb.cos(arb.pi() * s)))
        bu = hi(Bs)
        lam = lo(self.lam0)
        f = self.q.sqrt() * (
            upper_abs(ar - 1) * (1 / lam.sqrt() + bu.sqrt() / (smin + 0.5))
            + delta * hi(ar) * (2 * bu.sqrt() + 1 / (lam * (2 * smin - 1)).sqrt())
        )
        op = ((self.U ** 2 + self.V ** 2) / (self.n * arb.pi())
              * 8 * delta ** 2 / (1 - 4 * delta ** 2)).sqrt()
        return hi((f + op) ** 2)


def certify(al, ah, A, B):
    mid = (al + ah) / 2
    rad = (ah - al) / 2
    s = arb(str(mid), str(rad)) / 2
    # Decimal alpha endpoints enter Arb directly, with no binary64 narrowing.
    smin = arb(str(al)) / 2
    ja = A.trial(s)
    jb = B.trial(s)
    ra = A.residue_square(s, smin)
    rb = B.residue_square(s, smin)
    den = arb.gamma(1 + s) ** 2
    numerator = ja - jb + interval_hull(-rb.upper(), ra.upper())
    d = signed_quotient(numerator, den)
    gap = bounds(d)
    return {
        "alpha_lo": str(al),
        "alpha_hi": str(ah),
        "gap": gap,
        "positive": gap[0] > 0,
        "negative": gap[1] < 0,
        "residual_A": outward_float(ra, upper=True),
        "residual_B": outward_float(rb, upper=True),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--rb", required=True)
    ap.add_argument("--lower", required=True)
    ap.add_argument("--segments", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prec", type=int, default=200)
    ap.add_argument("--maxseg", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    help="new output file; existing files are never overwritten")
    args = ap.parse_args()
    output = (args.out.resolve() if args.out else
              ROOT / "results" / "arb" / f"arb_{args.name}.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite certificate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    RA = [int(x) for x in args.ra.split(",") if x != ""]
    RB = [int(x) for x in args.rb.split(",") if x != ""]
    t0 = time.time()
    A = Model(RA, args.n, prec=args.prec)
    B = Model(RB, args.n, prec=args.prec)
    al = Decimal(args.lower)
    step = (Decimal(2) - al) / args.segments
    rows = []
    count = args.segments if args.maxseg <= 0 else min(args.segments, args.maxseg)
    for k in range(count):
        lo_a = al + k * step
        hi_a = lo_a + step
        rows.append(certify(lo_a, hi_a, A, B))
        if k % 50 == 0:
            print(k, rows[-1]["gap"], flush=True)
    env = [min(r["gap"][0] for r in rows), max(r["gap"][1] for r in rows)]
    out = {
        "schema": CERTIFICATE_SCHEMA,
        "status": "valid" if count == args.segments else "partial_not_certificate",
        "name": args.name,
        "n": args.n,
        "RA": RA,
        "RB": RB,
        "lower": args.lower,
        "segments": args.segments,
        "rows_done": count,
        "rows": rows,
        "envelope": env,
        "all_positive": all(r["positive"] for r in rows),
        "all_negative": all(r["negative"] for r in rows),
        "seconds": time.time() - t0,
        "backend": "python-flint arb",
        "prec": args.prec,
        "source_sha256": sha256(__file__),
        "source_path": str(Path(__file__).resolve()),
        "output_path": str(output),
        "command": [sys.executable, *sys.argv],
        "environment": {"python": sys.version, "python_executable": sys.executable,
                        "python_flint": flint.__version__,
                        "platform": platform.platform()},
        "serialization": "Arb directed exact dyadic endpoints; binary64 nextafter; exact rational checks of binary and JSON decimal",
        "scope": "local coefficient conditional on analytic inequalities; not a new analytic proof",
    }
    print(json.dumps({k: out[k] for k in ("name", "envelope", "all_positive", "all_negative", "rows_done", "seconds")}, indent=1))
    with output.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
