"""Arb-certified continuous Green-potential range from an arbitrary charge.

The approximate CG charge is only a trial measure, never assumed exact.  Its
binary64 coefficients are reinterpreted as exact dyadic rationals and normalized
with Arb. All DFTs, powers and trigonometry in the enclosure use Arb balls.
The infinitely many omitted Fourier modes and intersample variation are bounded
analytically.  No grid extrapolation is used as an error bound.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, math, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
sys.path.insert(0,str(WORK/'reproducible'))
import numpy as np
import gate_solver as gs
import flint
from flint import arb,acb,ctx

def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def outward(x):
    """Convert an Arb enclosure to outward binary64 endpoints."""
    return [math.nextafter(float(x.lower()),-math.inf),
            math.nextafter(float(x.upper()),math.inf)]

def prepare(args):
    path=HERE/f'{args.witness}_N{args.n}_trial.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    witness=next(w for w in json.loads((WORK/'reproducible/witnesses.json').read_text(encoding='utf-8')) if w['id']==args.witness)
    gs.M=witness['n'];gs.RSETS=(tuple(witness['A']),tuple(witness['B']))
    gs.Q=gs.cycle(gs.M);gs.GATE_RATES=1-np.cos(2*np.pi*np.arange(gs.M)/gs.M);gs.GATE_RATES[0]=0
    cols,lower,alias=gs.kernels(args.n,witness['alpha'],args.gamma,'cell')
    charges=[]
    for label,R in zip(('A','B'),gs.RSETS):
        op=gs.BlockToeplitz(cols,R,args.n//32)
        q,rec=gs.solve_charge(op,lower*(1-len(R)/gs.M/32),rtol=1e-12)
        q=q/q.sum()
        charges.append({'label':label,'R':list(R),'charge_exact_float_hex':[float(x).hex() for x in q],
                        'shape':[args.n//32,len(R)],'solver_record_diagnostic_only':rec})
    data={'witness':witness,'N':args.n,'gamma':args.gamma,'width':'1/32',
          'interpretation':'Exact piecewise uniform signed trial charge, not a certified equilibrium charge.',
          'charges':charges}
    save(path,data);return data

def enclosure(data,trial,args):
    started=time.perf_counter();N=data['N'];M=data['witness']['n'];K=args.kmax;L=2**math.ceil(math.log2(2*K+2))
    alpha=arb(str(data['witness']['alpha']));gamma=arb(str(data['gamma']));pi=arb.pi()
    R=trial['R'];m=N//32
    q=[arb(float.fromhex(h)) for h in trial['charge_exact_float_hex']]
    mass=sum(q,arb(0));q=[x/mass for x in q]
    b=sum((abs(x) for x in q),arb(0))
    spatial=[]
    for gate in range(M):
        if gate in R:
            col=R.index(gate);v=[acb(q[j*len(R)+col]) for j in range(m)]+[acb(0)]*(N-m)
            spatial.append(acb.dft(v))
        else: spatial.append([acb(0)]*N)
    print(json.dumps({'phase':'spatial_dfts','gate':trial['label'],'elapsed_seconds':time.perf_counter()-started}),flush=True)
    modal=[acb.dft([spatial[j][r] for j in range(M)]) for r in range(N)]
    del spatial
    rates=[1-(2*pi*j/M).cos() for j in range(M)];rates[0]=arb(0)
    # At k=0 only the uniform spatial density remains. Centered global mode is 0.
    zz=[acb(0)]+[modal[0][j]/(gamma*rates[j]) for j in range(1,M)]
    constant=acb.dft(zz,inverse=True)
    coeff=[[acb(0) for _ in range(L)] for _ in R]
    for ri,gate in enumerate(R): coeff[ri][0]=constant[gate]*L
    derivative=[arb(0) for _ in R]
    for k in range(1,K+1):
        power=arb(k)**alpha
        zz=[modal[k%N][j]/(power+gamma*rates[j]) for j in range(M)]
        physical=acb.dft(zz,inverse=True)
        angle=pi*k/N
        cell=acb(0,-angle).exp()*(angle.sin()/angle)
        for ri,gate in enumerate(R):
            v=physical[gate]*cell
            coeff[ri][k]=v*L;coeff[ri][L-k]=v.conjugate()*L
            derivative[ri]+=4*pi*k*v.abs_upper()
        if k%8192==0:
            print(json.dumps({'phase':'coefficients','gate':trial['label'],'k':k,'elapsed_seconds':time.perf_counter()-started}),flush=True)
    del modal
    # For k > K, sinc <= N/(pi*k); the positive matrix resolvent has row sum k^-alpha.
    # An l1 bound on all component source coefficients is at most B.
    tail=2*b*N/(pi*alpha*(arb(K)**alpha))
    derivative_tail=4*b*N/((alpha-1)*(arb(K)**(alpha-1)))
    gate_records=[];all_lo=[];all_hi=[]
    for ri,gate in enumerate(R):
        values=acb.dft(coeff[ri],inverse=True)
        target=values[:L//32+1]
        lows=[x.real.lower() for x in target];highs=[x.real.upper() for x in target]
        lo=min(lows,key=float);hi=max(highs,key=float)
        deriv=derivative[ri]+derivative_tail
        between=deriv/(2*L)
        lower=lo-tail-between;upper=hi+tail+between
        gate_records.append({'gate':gate,'sampled_range':outward(lo.union(hi)),
                             'fourier_tail_bound':outward(tail),'full_derivative_bound':outward(deriv),
                             'inter_sample_bound':outward(between),
                             'potential_enclosure':outward(lower.union(upper))})
        all_lo.append(lower.lower());all_hi.append(upper.upper())
        coeff[ri]=[]
    lower=min(all_lo,key=float);upper=max(all_hi,key=float)
    mean=(lower*M).union(upper*M)
    return {'label':trial['label'],'source_N':N,'fourier_cutoff':K,'sample_grid_L':L,'precision_bits':ctx.prec,
            'trial_total_variation':outward(b),'gate_records':gate_records,
            'continuous_mean_enclosure':outward(mean),'enclosure_ball':str(mean),
            'elapsed_seconds':time.perf_counter()-started,
            'method':'equilibrium-charge positivity + continuous Green-potential extrema + Arb Fourier DFT + analytic high-mode and intersample bounds',
            'status':'conditional_on_Arb_ball_arithmetic_and_stated_analytic_lemma'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--witness',default='C18_0064');parser.add_argument('--n',type=int,default=8192)
    parser.add_argument('--kmax',type=int,default=16384);parser.add_argument('--gamma',type=float,default=2560000)
    parser.add_argument('--bits',type=int,default=100);parser.add_argument('--only',choices=('A','B'))
    args=parser.parse_args();ctx.prec=args.bits
    data=prepare(args)
    for trial in data['charges']:
        if args.only and trial['label']!=args.only:continue
        result=enclosure(data,trial,args)
        result.update(witness=args.witness,alpha=str(data['witness']['alpha']),gamma=str(args.gamma),width='1/32',
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),flint_version=flint.__version__)
        path=HERE/f'{args.witness}_N{args.n}_K{args.kmax}_{trial["label"]}_enclosure.json';save(path,result)
        print(json.dumps({'completed':str(path),'mean':result['continuous_mean_enclosure']}),flush=True)

if __name__=='__main__':main()
