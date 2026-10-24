"""Symmetric hidden-gate potential solver, with no fitted asymptotic coefficient.

Model: unit circle; D=(2*pi)**(-alpha), hence continuum eigenvalues |k|**alpha.

Detection on spatial interval of exact width w and gate states R. Initial law

is uniform product; initially absorbed mass is included. The lattice symbol

converges to the same model. Cell kernels include all continuum Fourier modes.

"""

from __future__ import annotations

import time

import numpy as np

from scipy import fft, linalg, special

from scipy.sparse.linalg import LinearOperator, cg

RSETS=((0,1,2,5,6,8),(0,1,2,4,7,8))

M=12

def cycle(n:int)->np.ndarray:

    if n<3: raise ValueError('n>=3 required')

    Q=-np.eye(n)

    for j in range(n): Q[j,(j-1)%n]=Q[j,(j+1)%n]=.5

    return Q

Q=cycle(M)

GATE_RATES=1-np.cos(2*np.pi*np.arange(M)/M)

GATE_RATES[0]=0.

def cell_values(n:int,a:float,g:float|None=None):

    # Analytically summed squared-sinc aliases of the continuous resolvent.

    j=np.arange(1,n,dtype=float); q=j/n

    ss=(np.sin(np.pi*q)/np.pi)**2

    val=np.zeros(n)

    if g is None or g==0:

        val[1:]=n**(-a)*ss*(special.zeta(a+2,q)+special.zeta(a+2,1-q))

        return val,0.

    r=g/n**a

    if r>=.25: raise ValueError(f'Alias-series ratio {r} >= .25; refine n.')

    terms=max(5,int(np.ceil(np.log(1e-17)/np.log(r))))

    val[0]=1/g

    val[1:]=ss*(q**(-2)/(g+j**a)+(1-q)**(-2)/(g+(n-j)**a))

    for t in range(terms):

        val[1:]+=n**(-a)*ss*(-r)**t*(special.zeta(a*(t+1)+2,1+q)+special.zeta(a*(t+1)+2,2-q))

    rem=2*n**(-a)/np.pi**2*r**terms*special.zeta(a*(terms+1)+2,1)/(1-r)

    return val,float(rem)

def kernels(n:int,a:float,gamma:float,method:str):

    """Return N-scaled potential columns indexed by gate distance and space."""

    if n<8 or gamma<=0 or not 1<a<=2: raise ValueError('Invalid n/alpha/gamma')

    eig=(np.sin(np.pi*np.arange(n)/n)/np.sin(np.pi/n))**a; eig[0]=0.

    space=[];minpos=np.inf;alias=0.

    for j in range(M//2+1):

        rate=GATE_RATES[j]*gamma

        if method=='cell': vals,rem=cell_values(n,a,rate)

        elif method=='lattice':

            vals=np.zeros(n)

            if j==0: vals[1:]=1/eig[1:]

            else: vals=1/(eig+rate)

            rem=0.

        else: raise ValueError(method)

        minpos=min(minpos,np.min(vals[1:] if j==0 else vals))

        alias=max(alias,rem)

        space.append(fft.ifft(vals).real*n)

    sp=np.array(space)

    full=sp[np.minimum(np.arange(M),M-np.arange(M))]

    # hidden gate Fourier inversion; real due to the symmetric cycle

    cols=fft.ifft(full,axis=0).real

    return cols,float(n*minpos),float(alias*n)

class BlockToeplitz:

    def __init__(self,cols:np.ndarray,R:tuple[int,...],m:int,precondition:bool=True):

        self.m=m;self.s=len(R);self.dim=m*self.s

        d=(np.asarray(R)[:,None]-np.asarray(R)[None,:])%M

        self.c=cols[d,:m].transpose(2,0,1).copy()

        # c(-h)=c(h), and each hidden-gate block is symmetric.

        self.length=fft.next_fast_len(2*m-1)

        emb=np.zeros((self.length,self.s,self.s))

        emb[:m]=self.c

        if m>1: emb[-(m-1):]=self.c[1:][::-1]

        self.fc=fft.rfft(emb,axis=0)

        self.op=LinearOperator((self.dim,self.dim),matvec=self.mv,dtype=float)

        self.prec=None;self.pc_min=None

        if precondition:

            pc=self.c.copy()

            if m>1:

                t=np.arange(1,m,dtype=float)/m

                pc[1:]=(1-t[:,None,None])*self.c[1:]+t[:,None,None]*self.c[1:][::-1]

            fp=fft.rfft(pc,axis=0).real

            fp=(fp+fp.transpose(0,2,1))/2

            ev,vec=np.linalg.eigh(fp)

            self.pc_min=float(ev.min())

            if ev.min()<=0: raise ArithmeticError(f'Nonpositive circulant preconditioner {ev.min()}')

            self.ifp=np.einsum('kij,kj,klj->kil',vec,1/ev,vec,optimize=True)

            self.prec=LinearOperator((self.dim,self.dim),matvec=self.pmv,dtype=float)

    def mv(self,x):

        x=np.asarray(x).reshape(self.m,self.s)

        fx=fft.rfft(x,n=self.length,axis=0)

        y=fft.irfft(np.einsum('kij,kj->ki',self.fc,fx,optimize=True),n=self.length,axis=0)

        return y[:self.m].ravel()

    def pmv(self,x):

        x=np.asarray(x).reshape(self.m,self.s)

        fx=fft.rfft(x,axis=0)

        return fft.irfft(np.einsum('kij,kj->ki',self.ifp,fx,optimize=True),n=self.m,axis=0).ravel()

    def dense(self):

        k=np.abs(np.arange(self.m)[:,None]-np.arange(self.m)[None,:])

        return self.c[k].transpose(0,2,1,3).reshape(self.dim,self.dim)

def solve_charge(op:BlockToeplitz,lower:float,rtol:float=2e-12,maxiter:int=4000):

    b=np.ones(op.dim);count=[0]

    def cb(x): count[0]+=1

    t=time.perf_counter()

    x,info=cg(op.op,b,M=op.prec,rtol=rtol,atol=0,maxiter=maxiter,callback=cb)

    r=b-op.mv(x)

    # Valid algebraic identity. Floating-point/FFT and spatial errors are separate.

    capacity_lower=float(2*x.sum()-np.dot(x,op.mv(x)))

    capacity_error=float(np.dot(r,r)/lower)

    mean_upper=M/capacity_lower

    mean_lower=M/(capacity_lower+capacity_error)

    rec={'mean':M/float(x.sum()),'cg_info':int(info),'iterations':count[0],

         'residual_l2':float(linalg.norm(r)),'residual_linf':float(np.max(abs(r))),

         'residual_relative':float(linalg.norm(r)/np.sqrt(op.dim)),

         'algebraic_mean_error_indicator':float(M*capacity_error/capacity_lower**2),

         'lambda_lower':lower,'seconds':time.perf_counter()-t,

         'capacity_quadratic_lower':capacity_lower,'capacity_error_indicator':capacity_error,

         'min_normalized_charge':float((x/x.sum()).min())}

    if info!=0: rec['status']='SOLVER_NOT_CONVERGED'

    else: rec['status']='computed_not_continuum_certified'

    return x,rec

def solve_pair(n:int,a:float,w:float,gamma:float,method:str='cell',rtol:float=2e-12):

    m=round(n*w)

    if abs(m/n-w)>1e-14 or m<2: raise ValueError('Align target width to grid')

    t=time.perf_counter(); cols,minpos,alias=kernels(n,a,gamma,method)

    xs=[];ops=[];recs=[]

    for R in RSETS:

        op=BlockToeplitz(cols,R,m);ops.append(op)

        x,rec=solve_charge(op,minpos*(1-len(R)/M*w),rtol);xs.append(x);recs.append(rec)

    # Bilinear resolvent identity avoids subtracting two nearly equal means.

    dop=BlockToeplitz.__new__(BlockToeplitz)

    dop.m=m;dop.s=len(RSETS[0]);dop.dim=m*dop.s;dop.length=ops[0].length

    # Subtract kernels BEFORE their FFT, preserving common-mode cancellation.

    diff=ops[0].c-ops[1].c

    emb=np.zeros((dop.length,dop.s,dop.s));emb[:m]=diff

    if m>1: emb[-(m-1):]=diff[1:][::-1]

    dop.fc=fft.rfft(emb,axis=0)

    qa,qb=[x/x.sum() for x in xs]

    gap=float(M*np.dot(qa,dop.mv(qb)))

    direct=recs[0]['mean']-recs[1]['mean']

    D=(2*np.pi)**(-a);epsilon=(D/gamma)**(1/a)

    return {'n':n,'m':m,'alpha':a,'width':w,'gamma':gamma,'D':D,'method':method,

            'epsilon':epsilon,'cells_per_epsilon':n*epsilon,

            'epsilon_over_width':epsilon/w,'gap':gap,'gap_direct':direct,

            'bilinear_direct_discrepancy':abs(gap-direct),'normalized_gap':gap/epsilon,

            'A':recs[0],'B':recs[1], 'alias_operator_indicator':alias,

            'wall_seconds':time.perf_counter()-t}

def direct_lattice(n:int,a:float,w:float,gamma:float):

    eig=(np.sin(np.pi*np.arange(n)/n)/np.sin(np.pi/n))**a

    H=linalg.circulant(fft.ifft(eig).real)

    joint=np.kron(H,np.eye(M))-gamma*np.kron(np.eye(n),Q)

    out=[];m=round(n*w)

    for R in RSETS:

        keep=[x*M+j for x in range(n) for j in range(M) if not(x<m and j in R)]

        A=joint[np.ix_(keep,keep)]

        u=linalg.solve(A,np.ones(len(keep)),assume_a='pos')

        out.append(float(u.sum()/(M*n)))

    return out

