"""Exact cyclotomic chain sums and Arb enclosures for the Brownian mechanism."""
from pathlib import Path
import json,time
from flint import fmpq,fmpq_poly,arb,arb_mat,ctx
from interval_engine_arb import bounds
R=Path(__file__).resolve().parents[1];start=time.perf_counter();n=18
masks=[[0,1,4,5,6,8,11,12,14],[0,1,4,6,7,10,11,12,14]]
# x = exp(i*pi/36). Phi_72(x) = x^24 - x^12 + 1.
P=fmpq_poly([1]+[0]*11+[-1]+[0]*11+[1]);one=fmpq_poly([1])
def xp(k):return fmpq_poly([0]*(k%72)+[1])%P
def mul(a,b):return a*b%P
root2=xp(9)+xp(-9)
ev=[mul(root2,mul(xp(2*k)-xp(-2*k),-xp(18)))/2 for k in range(18)]
kd=[sum((mul(ev[k],xp(4*k*d)+xp(-4*k*d))/36 for k in range(1,18)),fmpq_poly([]))%P for d in range(18)]
exact=[]
for mask in masks:
    c=[j for j in range(n) if j not in mask];v=[one for j in c];mom=[]
    for power in range(1,4):
        v=[sum((mul(kd[(i-j)%n],v[t]) for t,j in enumerate(c)),fmpq_poly([]))%P for i in c]
        mom.append(sum(v,fmpq_poly([]))%P)
    exact.append(mom)
assert exact[0][0]==exact[1][0]
assert exact[0][1]==exact[1][1]==fmpq_poly([fmpq(5,2)])
diff=exact[0][2]-exact[1][2];assert diff!=0
def polylist(p):return [str(z) for z in p.coeffs()]
records=[]
for prec in (160,256):
    ctx.prec=prec
    kval=[sum(((arb(2).sqrt()*(arb.pi()*k/18).sin()*(2*arb.pi()*k*d/18).cos()/18) for k in range(1,18)),arb(0)) for d in range(18)]
    models=[]
    for mask in masks:
        c=[j for j in range(n) if j not in mask]
        K=arb_mat([[kval[(i-j)%n] for j in c] for i in c]);v=arb_mat([[1] for j in c])
        sums=[]
        for power in range(1,4):
            v=K*v;sums.append(sum((v[j,0] for j in range(9)),arb(0)))
        traces=[];T=K
        for power in range(1,10):
            traces.append(sum((T[j,j] for j in range(9)),arb(0)));T=T*K
        # H_CC eigenvalues depend only on closed run inventory, verified separately.
        models.append({'sums':sums,'traces':traces})
    gaps=[models[0]['traces'][j]-models[1]['traces'][j] for j in range(9)]
    distinguish=next(j+1 for j,gap in enumerate(gaps) if not gap.contains(0))
    records.append({'precision':prec,'chain_sums':[[bounds(v) for v in m['sums']] for m in models],
                    'third_chain_gap':bounds(models[0]['sums'][2]-models[1]['sums'][2]),
                    'spectral_power_trace_first_resolved_order':distinguish,
                    'spectral_trace_gap':bounds(gaps[distinguish-1])})
out={'all_pass':True,'exact_field':'Q[x]/(x^24-x^12+1), x=exp(i*pi/36)',
     'first_chain_equal':True,'second_chain_exact':'5/2','third_chain_gap_polynomial_ascending':polylist(diff),
     'chain_polynomials':[[polylist(v) for v in row] for row in exact],
     'arb_checks':records,'seconds':time.perf_counter()-start,
     'scope':'Chain order three uses four observation factors; it is a mechanism diagnostic, not by itself a proof of a nonzero inverse-response gap.'}
(R/'results/MECHANISM.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
