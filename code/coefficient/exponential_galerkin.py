"""Whole-line continuous exponential Galerkin basis, including constant right tail.

All integrals in this finite-dimensional variational problem are analytic.
Floating assembly/solves and finite-basis errors remain: this is NOT an interval
certificate and does NOT use the Brownian-only residual bound for new trials.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy import linalg, special

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def cycle(n):
    h=np.eye(n)
    for j in range(n):h[j,(j-1)%n]=h[j,(j+1)%n]=-.5
    return h

def spatial_kernel(a,b,s,cross=False):
    """Kernel for UNNORMALIZED derivative transforms 1/(a +/- i xi)."""
    a=np.asarray(a)[:,None];b=np.asarray(b)[None,:]
    rho=2*s-1;t=rho-1;kappa=np.sin(np.pi*rho/2)
    if not cross:return (a**t+b**t)/(2*kappa*(a+b))
    aa,bb=np.broadcast_arrays(a,b)
    near=np.abs(bb-aa)<1e-11*np.maximum(aa,bb)
    ans=np.empty_like(aa)
    ans[near]=-t*((aa[near]+bb[near])/2)**(t-1)
    ans[~near]=-(aa[~near]**t)*np.expm1(t*np.log(bb[~near]/aa[~near]))/(bb[~near]-aa[~near])
    return ans/(2*kappa)

def build_basis(n,R,rates):
    h=cycle(n);closed=np.array(sorted(set(range(n))-set(R)))
    ev,vec=linalg.eigh(h);keep=ev>1e-12
    lam=np.sqrt(ev[keep]);v=vec[:,keep]
    mu2,u=linalg.eigh(h[np.ix_(closed,closed)]);mu=np.sqrt(mu2)
    rates=np.array([float(a) for a in rates if abs(float(a)-1)>1e-12])
    left_rates=np.r_[mu,rates,1.]
    right_rates=np.r_[lam,rates,1.]
    nr=len(closed);count=nr+nr*len(rates)+n*len(rates)+1
    left=np.zeros((count,len(left_rates),n));right=np.zeros((count,len(right_rates),n))
    constants=np.zeros((count,n));labels=[]
    # Complete Brownian harmonic extensions of the closed-state boundary values.
    for b,j in enumerate(closed):
        left[b][np.ix_(np.arange(nr),closed)]=(u*u[b,:]).T
        right[b,:len(lam),:]=(v*v[j,:]).T
        constants[b,:]=1/n
        labels.append('Brownian_boundary_'+str(j))
    pos=nr
    for j in closed:
        for k,a in enumerate(rates):
            left[pos,nr+k,j]=1;left[pos,-1,j]=-1
            labels.append('left_zero_trace_'+str(j)+'_'+str(a));pos+=1
    for j in range(n):
        for k,a in enumerate(rates):
            right[pos,len(lam)+k,j]=1;right[pos,-1,j]=-1
            labels.append('right_zero_trace_'+str(j)+'_'+str(a));pos+=1
    # A common positive-side plateau. H1=0 eliminates its constant gate energy.
    constants[pos,:]=1;right[pos,-1,:]=-1
    labels.append('common_constant_tail')
    continuity=float(np.max(np.abs(left.sum(axis=1)-right.sum(axis=1)-constants)))
    assert continuity<1e-12,continuity
    return h,closed,left_rates,right_rates,left,right,constants,labels,continuity

def assemble(n,R,alpha,rates):
    h,closed,a,b,left,right,constants,labels,continuity=build_basis(n,R,rates)
    s=alpha/2
    ld=left*a[None,:,None];rd=-right*b[None,:,None]
    ll=spatial_kernel(a,a,s);rr=spatial_kernel(b,b,s);lr=spatial_kernel(a,b,s,True)
    kinetic=np.zeros((len(labels),len(labels)))
    for j in range(n):
        cross=ld[:,:,j]@lr@rd[:,:,j].T
        kinetic+=ld[:,:,j]@ll@ld[:,:,j].T+rd[:,:,j]@rr@rd[:,:,j].T+cross+cross.T
    gate=np.einsum('bki,ij,clj,kl->bc',left,h,left,1/(a[:,None]+a[None,:]),optimize=True)
    gate+=np.einsum('bki,ij,clj,kl->bc',right,h,right,1/(b[:,None]+b[None,:]),optimize=True)
    # Every constant vector is exactly a multiple of 1 and hence has zero H energy.
    f=special.gamma(1+s)*np.einsum('bki,k->b',left,a**(s-1))
    matrix=kinetic+gate;matrix=(matrix+matrix.T)/2
    return matrix,f,len(closed),{'basis_dimension':len(labels),'rates':list(map(float,rates)),
           'continuity_residual':continuity,'source':'whole-line analytic integrals',
           'includes_constant_right_tail':True,'domain_truncation':False}

def solve(n,R,alpha,rates,cutoff):
    A,f,r,meta=assemble(n,R,alpha,rates)
    a0=A[:r,:r];cross=A[:r,r:];b=linalg.solve(a0,f[:r],assume_a='pos')
    residual=f[r:]-cross.T@b
    schur=A[r:,r:]-cross.T@linalg.solve(a0,cross,assume_a='pos')
    schur=(schur+schur.T)/2
    scale=1/np.sqrt(np.maximum(np.diag(schur),1e-300))
    whitened=scale[:,None]*schur*scale[None,:]
    eig,vec=linalg.eigh(whitened)
    selected=eig>cutoff*eig[-1]
    projected=vec[:,selected].T@(scale*residual)
    correction=scale*(vec[:,selected]@(projected/eig[selected]))
    coeff=np.r_[b-linalg.solve(a0,cross@correction,assume_a='pos'),correction]
    denominator=n*special.gamma(1+alpha/2)**2
    baseline=float(f[:r]@b/denominator)
    gain=float(np.sum(projected**2/eig[selected])/denominator)
    objective=float((2*f@coeff-coeff@A@coeff)/denominator)
    return dict(**meta,alpha=alpha,cutoff=cutoff,retained_correction_directions=int(selected.sum()),
                minimum_whitened_eigenvalue=float(eig[0]),maximum_whitened_eigenvalue=float(eig[-1]),
                negative_whitened_eigenvalues=int(np.sum(eig<0)),
                baseline_coefficient=baseline,correction_gain=gain,coefficient=baseline+gain,
                direct_objective=objective,objective_discrepancy=abs(objective-baseline-gain),
                projected_residual_norm=float(np.linalg.norm(A@coeff-f)),
                status='FLOATING_WHOLE_LINE_GALERKIN_NOT_CERTIFIED')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--levels',nargs='+',type=int,default=[1,2,3,4])
    ap.add_argument('--cutoffs',nargs='+',type=float,default=[1e-10,1e-12])
    args=ap.parse_args()
    witnesses=json.loads((ROOT/'reproducible/witnesses.json').read_text())
    w=next(z for z in witnesses if z['id']=='C18_0064')
    rows=[];start=time.perf_counter()
    for alpha in [2.,w['alpha']]:
        for level in args.levels:
            rates=[2.**k for k in range(-level,level+1) if k]
            for cutoff in args.cutoffs:
                t=time.perf_counter()
                a=solve(w['n'],w['A'],alpha,rates,cutoff)
                b=solve(w['n'],w['B'],alpha,rates,cutoff)
                row={'pair':w['id'],'alpha':alpha,'level':level,'cutoff':cutoff,
                     'A':a,'B':b,'gap':a['coefficient']-b['coefficient'],'seconds':time.perf_counter()-t}
                rows.append(row)
                output={'status':'NUMERICAL_WHOLE_LINE_APPROXIMATION_NOT_RESIDUAL_CERTIFICATE',
                        'rows':rows,'seconds':time.perf_counter()-start,
                        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                (HERE/'exponential_results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
                print(json.dumps({k:row[k] for k in ['alpha','level','cutoff','gap','seconds']}),flush=True)

if __name__=='__main__':main()
