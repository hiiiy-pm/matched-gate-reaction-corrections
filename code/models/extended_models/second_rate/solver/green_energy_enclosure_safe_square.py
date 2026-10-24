"""Arb enclosure of the exact continuous Green energy of a V9 trial charge.

The common scalar G0 kernel is integrated analytically for cells and atoms.
Only the smoother negative gate-resolvent correction is Fourier truncated;
its omitted tail has a proved nonpositive interval.  Candidate CG errors are
irrelevant: all input coordinates are arbitrary exact binary64 trial numbers.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import flint
from flint import arb, acb, ctx

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
V9=WORK/'extended_models/second_rate/continuum'

def outward(x):
    if not x.is_finite():
        raise ArithmeticError('Nonfinite Arb result rejected before endpoint conversion')
    return [math.nextafter(float(x.lower()),-math.inf),
            math.nextafter(float(x.upper()),math.inf)]

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def scalar_energy(cell,atoms,N,r,alpha,bits):
    """Return analytic G0 energy; all |x-y| <= w=1/32."""
    ctx.prec=bits; m=N//32; pi=arb.pi(); beta=alpha-1; w=arb(1)/32
    C=2*(1-alpha).gamma()*(pi*alpha/2).sin()*(2*pi)**beta
    J=10
    poly=[2*(alpha-2*j).zeta()*((-1)**j)*(2*pi)**(2*j)/arb(math.factorial(2*j)) for j in range(J+1)]
    cfe=2**(alpha+1)*pi**(alpha-1)*(pi*alpha/2).sin()
    rem=abs(cfe)*arb(3).zeta()*w**(2*J+2)/(1-w*w)
    # H''=G0, F'=G0, where H is even and F odd.
    def H(t):
        x=arb(abs(t))/N
        val=C*x**(beta+2)/((beta+1)*(beta+2)) if t else arb(0)
        mon=x*x
        for j,p in enumerate(poly):
            val+=p*mon/((2*j+1)*(2*j+2)); mon*=x*x
        return val
    def F(t):
        x=arb(abs(t))/N
        val=C*x**(beta+1)/(beta+1) if t else arb(0)
        mon=x
        for j,p in enumerate(poly):
            val+=p*mon/(2*j+1);mon*=x*x
        return val if t>=0 else -val
    hs=[H(t) for t in range(m+1)]
    kern=[N*N*(hs[1]-2*hs[0]+hs[1])]
    kern.extend(N*N*(hs[t+1]-2*hs[t]+hs[t-1]) for t in range(1,m))
    fs=[F(t) for t in range(m+1)]
    point=[N*(fs[t+1]-fs[t]) for t in range(m)]
    gw=C*w**beta;mon=arb(1)
    for p in poly:gw+=p*mon;mon*=w*w
    L=1<<math.ceil(math.log2(2*m-1))
    emb=[acb(0)]*L
    for t in range(m):emb[t]=acb(kern[t])
    for t in range(1,m):emb[L-t]=acb(kern[t])
    fk=acb.dft(emb); result=arb(0); max_imag=arb(0)
    for j in range(r):
        q=[cell[t*r+j] for t in range(m)]
        fq=acb.dft([acb(z) for z in q]+[acb(0)]*(L-m))
        # Parseval avoids one inverse transform and is real by symmetry.
        ec=sum((fk[k].real*(fq[k].real*fq[k].real+fq[k].imag*fq[k].imag) for k in range(L)),arb(0))/L
        a,b=atoms[j],atoms[r+j]
        ec+=2*a*sum((q[t]*point[t] for t in range(m)),arb(0))
        ec+=2*b*sum((q[t]*point[m-1-t] for t in range(m)),arb(0))
        ec+=(a*a+b*b)*poly[0]+2*a*b*gw
        result+=ec
    # A uniform kernel remainder contributes at most total variation squared.
    btot=sum(map(abs,cell+atoms),arb(0))
    error=rem*btot*btot
    return result+arb(0,error.abs_upper()), {'series_J':J,'uniform_G0_remainder_bound':outward(rem),
            'G0_energy_remainder_bound':outward(error),'target_convolution_length':L}

def run(data,tr,K,bits,trial_path):
    started=time.perf_counter();ctx.prec=bits
    N=data['N']; M=data['witness']['n']; R=tr['R'];r=len(R);m=N//32
    alpha=arb(str(data['witness']['alpha']));gamma=arb(str(data['gamma']));pi=arb.pi()
    cell=[arb(float.fromhex(x)) for x in tr['charge_exact_float_hex']]
    atoms=[arb(float.fromhex(x)) for x in tr['atom_exact_float_hex']]
    mass=sum(cell+atoms,arb(0));cell=[z/mass for z in cell];atoms=[z/mass for z in atoms]
    E0,diag=scalar_energy(cell,atoms,N,r,alpha,bits)
    print(json.dumps({'stage':'analytic_G0_energy','label':tr['label'],'value':outward(E0),'seconds':time.perf_counter()-started}),flush=True)
    spatial=[]
    for gate in range(M):
        if gate in R:
            col=R.index(gate)
            spatial.append(acb.dft([acb(cell[t*r+col]) for t in range(m)]+[acb(0)]*(N-m)))
        else:spatial.append(None)
    modal=[acb.dft([spatial[j][k] if spatial[j] is not None else acb(0) for j in range(M)]) for k in range(N)]
    del spatial
    am0=acb.dft([acb(atoms[R.index(j)]) if j in R else acb(0) for j in range(M)])
    am1=acb.dft([acb(atoms[r+R.index(j)]) if j in R else acb(0) for j in range(M)])
    rates=[1-(2*pi*j/M).cos() for j in range(M)];rates[0]=arb(0)
    Er=arb(0)
    for ell in range(1,M):
        z=modal[0][ell]+am0[ell]+am1[ell]
        Er+=(z.real*z.real+z.imag*z.imag)/(gamma*rates[ell]*M)
    Ezero=Er
    for k in range(1,K+1):
        power=arb(k)**alpha; angle=pi*k/N
        cf=acb(0,-angle).exp()*angle.sin()/angle
        phase=acb(0,-2*pi*(k%32)/32).exp()
        for ell in range(1,M):
            q=modal[k%N][ell]*cf+am0[ell]+phase*am1[ell]
            sq=q.real*q.real+q.imag*q.imag
            lam=gamma*rates[ell]
            Er-=2*sq*lam/(M*power*(power+lam))
        if k%131072==0:print(json.dumps({'stage':'gate_energy','label':tr['label'],'k':k,'seconds':time.perf_counter()-started}),flush=True)
    # Matrix operator norm <= 2 gamma k^(-2 alpha). Bound each Fourier
    # vector by source l1. All omitted correction energies are nonpositive.
    B=sum(map(abs,cell+atoms),arb(0))
    tail=4*gamma*B*B/((2*alpha-1)*arb(K)**(2*alpha-1))
    energy=E0+Er+(-tail).union(arb(0))
    rec={'witness':data['witness']['id'],'label':tr['label'],'N':N,'gamma':str(data['gamma']),
         'alpha':str(data['witness']['alpha']),'fourier_cutoff':K,'precision_bits':bits,
         'G0_energy':outward(E0),'gate_zero_mode_energy':outward(Ezero),
         'finite_gate_correction_energy':outward(Er),'gate_energy_tail_bound':outward(tail),
         'exact_continuous_trial_green_energy':outward(energy),
         'tau_variational_upper_ball':outward(M*energy),
         'source_total_variation':outward(B),'analytic_diagnostics':diag,
         'seconds':time.perf_counter()-started,'flint_version':flint.__version__,
         'trial_path':str(trial_path),'trial_sha256':hashlib.sha256(trial_path.read_bytes()).hexdigest(),
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'engine_source_filename':Path(__file__).name,
         'status':'conditional_on_Arb_ball_arithmetic_and_stated_analytic_lemma'}
    return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--witness',default='C18_0064');p.add_argument('--n',type=int,default=65536)
    p.add_argument('--kmax',type=int,default=524288);p.add_argument('--bits',type=int,default=100)
    p.add_argument('--only',choices=('A','B'));args=p.parse_args()
    path=V9/f'{args.witness}_N{args.n}_enriched_trial.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    for tr in data['charges']:
        if args.only and tr['label']!=args.only:continue
        rec=run(data,tr,args.kmax,args.bits,path)
        out=HERE/f'{args.witness}_N{args.n}_K{args.kmax}_{tr["label"]}_energy_safe_square.json'
        save(out,rec);print(json.dumps({'completed':str(out),'energy':rec['exact_continuous_trial_green_energy']}),flush=True)
if __name__=='__main__':main()
