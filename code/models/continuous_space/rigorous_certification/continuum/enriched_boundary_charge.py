"""Trial charge generator: add real boundary atoms to cell charges.

All results here are numerical trial coordinates. A separate Arb evaluation
certifies them as arbitrary signed probability measures; neither the mixed CG
solve nor its apparent convergence is interpreted as a continuum error bound.
"""
from __future__ import annotations
import os
for z in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[z]='1'
import argparse,json,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[2]
sys.path.insert(0,str(WORK/'reproducible'))
import numpy as np
from scipy import fft,linalg,special
from scipy.sparse.linalg import cg
import gate_solver as gs

def point_cell_values(n,alpha,rate):
    j=np.arange(1,n,dtype=float);q=j/n;r=rate/n**alpha
    z=1j*np.zeros(n);z[0]=0 if rate==0 else 1/rate
    if r>=.25:raise ValueError(r)
    val=q**(-1)/(q**alpha+r)-(1-q)**(-1)/((1-q)**alpha+r)
    terms=max(5,int(np.ceil(np.log(1e-18)/np.log(r)))) if r else 1
    for t in range(terms):
        val+=(-r)**t*(special.zeta(alpha*(t+1)+1,1+q)-special.zeta(alpha*(t+1)+1,2-q))
    z[1:]=n**(-alpha)*np.sin(np.pi*q)/np.pi*np.exp(-1j*np.pi*q)*val
    return z

def atomic_green(alpha,gamma,M,kmax=1048576):
    k=np.arange(1,kmax+1,dtype=float);ka=k**alpha
    cp=np.cos(2*np.pi*(k%32)/32)
    # Common spatial Green at separation zero and one target width.
    x=1/32;beta=alpha-1
    c=2*special.gamma(1-alpha)*np.sin(np.pi*alpha/2)*(2*np.pi)**beta
    gw=c*x**beta
    for j in range(9):gw+=2*special.zeta(alpha-2*j)*(-1)**j*(2*np.pi*x)**(2*j)/special.factorial(2*j)
    base=np.array([2*special.zeta(alpha),gw])
    spatial=[]
    for ell in range(M):
        rate=gamma*(1-np.cos(2*np.pi*ell/M))
        if ell==0:spatial.append(base);continue
        weights=2*rate/(ka*(ka+rate))
        spatial.append(base+1/rate-np.array([weights.sum(),weights@cp]))
    return fft.ifft(np.array(spatial),axis=0).real

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--witness',default='C18_0064');pa.add_argument('--n',type=int,default=65536)
    pa.add_argument('--gamma',type=float,default=2560000);args=pa.parse_args()
    start=time.perf_counter();w=next(z for z in json.loads((WORK/'reproducible/witnesses.json').read_text(encoding='utf-8')) if z['id']==args.witness)
    M=w['n'];N=args.n;m=N//32;alpha=w['alpha'];gs.M=M;gs.Q=gs.cycle(M)
    gs.GATE_RATES=1-np.cos(2*np.pi*np.arange(M)/M);gs.GATE_RATES[0]=0
    cols,lower,alias=gs.kernels(N,alpha,args.gamma,'cell')
    pc=[]
    for ell in range(M):pc.append(fft.ifft(point_cell_values(N,alpha,args.gamma*gs.GATE_RATES[ell])).real*N)
    pc=fft.ifft(np.array(pc),axis=0).real
    atomic=atomic_green(alpha,args.gamma,M)
    output={'witness':w,'N':N,'gamma':args.gamma,'width':'1/32','method':'cell plus independent real boundary Dirac masses; trial coordinates only','charges':[]}
    for label,R0 in zip(('A','B'),(w['A'],w['B'])):
        R=np.array(R0);r=len(R);op=gs.BlockToeplitz(cols,tuple(R),m)
        j=np.arange(m)[:,None,None];dist=(R[:,None]-R[None,:])%M
        P=np.concatenate([pc[dist[None,:,:],(-j)%N].reshape(m*r,r),
                          pc[dist[None,:,:],(m-j)%N].reshape(m*r,r)],axis=1)
        S=np.block([[atomic[dist,0],atomic[dist,1]],[atomic[dist,1],atomic[dist,0]]])
        base,info=cg(op.op,np.ones(m*r),M=op.prec,rtol=1e-12,atol=0,maxiter=4000)
        if info:raise ArithmeticError(('base',info))
        U=[]
        for col in range(2*r):
            sol,info=cg(op.op,P[:,col],M=op.prec,rtol=1e-12,atol=0,maxiter=4000)
            if info:raise ArithmeticError((col,info))
            U.append(sol)
        U=np.array(U).T
        Schur=S-P.T@U
        atom=linalg.solve((Schur+Schur.T)/2,np.ones(2*r)-P.T@base,assume_a='sym')
        cell=base-U@atom;mass=cell.sum()+atom.sum();cell/=mass;atom/=mass
        tau=M/mass
        rec={'label':label,'R':R0,'shape':[m,r],'charge_exact_float_hex':[float(z).hex() for z in cell],
             'atom_exact_float_hex':[float(z).hex() for z in atom],
             'atom_positions':['0']*r+['1/32']*r,'atom_gates':R0+R0,
             'numeric_mean_diagnostic':tau,'cell_total_variation':abs(cell).sum(),'atom_total_variation':abs(atom).sum(),
             'min_atom':atom.min(),'schur_min_eig':linalg.eigvalsh((Schur+Schur.T)/2)[0]}
        output['charges'].append(rec)
        print(json.dumps({'label':label,'tau_trial_diagnostic':tau,'cell_total_variation':rec['cell_total_variation'],'atom_total_variation':rec['atom_total_variation'],'elapsed':time.perf_counter()-start}),flush=True)
    path=HERE/f'{args.witness}_N{N}_enriched_trial.json';path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(str(path),flush=True)
if __name__=='__main__':main()
