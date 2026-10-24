"""Arb continuous MFPT enclosure for cell + real boundary-atom trials.

The common atomic Green singularity is evaluated analytically; only the
smoother gate-resolvent difference is Fourier truncated. All displayed bounds
include infinite-frequency tails and all points between samples.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path
from flint import arb,acb,ctx
import flint
from green_potential_enclosure import HERE,outward,save

def run(data,trial,K,bits):
    started=time.perf_counter();ctx.prec=bits
    N=data['N'];M=data['witness']['n'];R=trial['R'];r=len(R);m=N//32
    L=2**math.ceil(math.log2(2*K+2));h=arb(1)/L;w=arb(1)/32
    alpha=arb(str(data['witness']['alpha']));beta=alpha-1;pi=arb.pi();gamma=arb(str(data['gamma']))
    cell=[arb(float.fromhex(z)) for z in trial['charge_exact_float_hex']]
    atoms=[arb(float.fromhex(z)) for z in trial['atom_exact_float_hex']]
    mass=sum(cell+atoms,arb(0));cell=[z/mass for z in cell];atoms=[z/mass for z in atoms]
    bc=sum(map(abs,cell),arb(0));ba=sum(map(abs,atoms),arb(0))
    tv=arb(0)
    for j in range(r):
        seq=[arb(0)]+[cell[t*r+j] for t in range(m)]+[arb(0)]
        tv+=sum((abs(b-a) for a,b in zip(seq,seq[1:])),arb(0))
    spatial=[]
    for gate in range(M):
        if gate in R:
            col=R.index(gate);v=[acb(cell[j*r+col]) for j in range(m)]+[acb(0)]*(N-m)
            spatial.append(acb.dft(v))
        else:spatial.append([acb(0)]*N)
    modal=[acb.dft([spatial[j][k] for j in range(M)]) for k in range(N)];del spatial
    av0=[atoms[R.index(j)] if j in R else arb(0) for j in range(M)]
    av1=[atoms[r+R.index(j)] if j in R else arb(0) for j in range(M)]
    am0=acb.dft(av0);am1=acb.dft(av1)
    rates=[1-(2*pi*j/M).cos() for j in range(M)];rates[0]=arb(0)
    zero=[acb(0)]+[(modal[0][j]+am0[j]+am1[j])/(gamma*rates[j]) for j in range(1,M)]
    constant=acb.dft(zero,inverse=True)
    coeff=[[acb(0)]*L for _ in R];second=[arb(0) for _ in R]
    for j,gate in enumerate(R):coeff[j][0]=constant[gate]*L
    for k in range(1,K+1):
        power=arb(k)**alpha;angle=pi*k/N
        cellfactor=acb(0,-angle).exp()*angle.sin()/angle
        phase=acb(0,-2*pi*(k%32)/32).exp()
        z=[]
        for ell in range(M):
            ca=am0[ell]+phase*am1[ell]
            z.append(modal[k%N][ell]*cellfactor/(power+gamma*rates[ell])
                     -ca*(gamma*rates[ell])/(power*(power+gamma*rates[ell])))
        physical=acb.dft(z,inverse=True)
        for j,gate in enumerate(R):
            v=physical[gate];coeff[j][k]=v*L;coeff[j][L-k]=v.conjugate()*L
            second[j]+=8*pi*pi*k*k*v.abs_upper()
        if k%65536==0:print(json.dumps({'gate':trial['label'],'k':k,'seconds':time.perf_counter()-started}),flush=True)
    del modal
    # TV, rather than l1, exploits that the source is piecewise uniform:
    # |qhat_cell(k)| <= N TV(q)/(2 pi |k|).
    celltail=tv*N/(pi*alpha*arb(K)**alpha)
    atomtail=4*gamma*ba/((2*alpha-1)*arb(K)**(2*alpha-1))
    # G0(x) = C |x|^beta + sum_j p_j x^(2j), |x|<1.
    C=2*(1-alpha).gamma()*(pi*alpha/2).sin()*(2*pi)**beta
    J=8
    poly=[2*(alpha-2*j).zeta()*((-1)**j)*(2*pi)**(2*j)/arb(math.factorial(2*j)) for j in range(J+1)]
    cfe=2**(alpha+1)*pi**(alpha-1)*(pi*alpha/2).sin()
    gremainder=abs(cfe)*arb(3).zeta()*w**(2*J+2)/(1-w*w)
    psecond=sum((abs(poly[j])*(2*j)*(2*j-1)*w**(2*j-2) for j in range(1,J+1)),arb(0))
    shape=(1-beta)*beta**(beta/(1-beta))
    gdefect=abs(C)*h**beta*shape+h*h*psecond/8+2*gremainder
    gv=[]
    for t in range(L//32+1):
        x=arb(t)/L
        value=C*x**beta if t else arb(0)
        xx=x*x;mon=arb(1)
        for c in poly:value+=c*mon;mon*=xx
        value+=arb(0,gremainder.abs_upper())
        gv.append(value)
    all_lo=[];all_hi=[];records=[]
    for j,gate in enumerate(R):
        vals=acb.dft(coeff[j],inverse=True)
        target=[vals[t].real+atoms[j]*gv[t]+atoms[r+j]*gv[-1-t] for t in range(len(gv))]
        # lower()/upper() are exact dyadic point Arb objects (zero radius).
        # Compare those exactly, rather than selecting extrema via binary64.
        lo=min(x.lower() for x in target);hi=max(x.upper() for x in target)
        interp=second[j]*h*h/8+(abs(atoms[j])+abs(atoms[r+j]))*gdefect
        err=celltail+atomtail+interp
        lower=lo-err;upper=hi+err
        records.append({'gate':gate,'sample_potential_range':outward(lo.union(hi)),
                        'finite_fourier_interpolation_bound':outward(second[j]*h*h/8),
                        'atomic_interpolation_bound':outward((abs(atoms[j])+abs(atoms[r+j]))*gdefect),
                        'total_added_bound':outward(err),'potential_enclosure':outward(lower.union(upper))})
        all_lo.append(lower.lower());all_hi.append(upper.upper());coeff[j]=[]
    lower=min(all_lo);upper=max(all_hi);mean=(M*lower).union(M*upper)
    return {'label':trial['label'],'witness':data['witness']['id'],'alpha':str(data['witness']['alpha']),
            'gamma':str(data['gamma']),'source_N':N,'fourier_cutoff':K,'sample_grid_L':L,'precision_bits':bits,
            'cell_l1':outward(bc),'atom_l1':outward(ba),'cell_mass_total_variation':outward(tv),
            'cell_fourier_tail_bound':outward(celltail),'atom_gate_remainder_fourier_tail_bound':outward(atomtail),
            'gate_records':records,'continuous_mean_enclosure':outward(mean),'enclosure_ball':str(mean),
            'seconds':time.perf_counter()-started,'flint_version':flint.__version__,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'trial_sha256':hashlib.sha256((HERE/f"{data['witness']['id']}_N{N}_enriched_trial.json").read_bytes()).hexdigest(),
            'status':'conditional_on_Arb_ball_arithmetic_and_stated_analytic_lemma'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--witness',default='C18_0064');p.add_argument('--n',type=int,default=65536)
    p.add_argument('--kmax',type=int,default=262144);p.add_argument('--bits',type=int,default=100);p.add_argument('--only',choices=('A','B'))
    a=p.parse_args();data=json.loads((HERE/f'{a.witness}_N{a.n}_enriched_trial.json').read_text(encoding='utf-8'))
    for tr in data['charges']:
        if a.only and a.only!=tr['label']:continue
        rec=run(data,tr,a.kmax,a.bits)
        path=HERE/f'{a.witness}_N{a.n}_K{a.kmax}_{tr["label"]}_enriched_enclosure.json';save(path,rec)
        print(json.dumps({'completed':str(path),'mean':rec['continuous_mean_enclosure']}),flush=True)
if __name__=='__main__':main()
