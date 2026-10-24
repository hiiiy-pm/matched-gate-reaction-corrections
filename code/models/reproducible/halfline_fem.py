"""Independent real-line half-space variational experiment.

No circle Green kernels are used. Continuous piecewise-linear hat functions,

zero exterior at +/-B; exact real-line fractional stiffness and exact source

integrals. Domain truncation and mesh errors remain explicitly separate.

E_s(z)=M^-1 sum_j || |D|^s z_j ||_2^2 + M^-1 int z^T(-Q)z,

F_s(z)=Gamma(1+s)/(M Gamma(1-s)) sum_C int_0^infty x^-s z_j(-x) dx.

Candidate local coefficient = sup(2 F_s-E_s)/Gamma(1+s)^2.

"""

from pathlib import Path

import numpy as np

from scipy import fft,special,linalg

from scipy.sparse.linalg import LinearOperator,cg

import time,json,sys

from gate_solver import M,RSETS,GATE_RATES

def centered_diff_power(k,beta,order):

    k=np.asarray(k,dtype=float)

    out=np.empty_like(k);small=k<16

    x=k[small].astype(np.longdouble)

    if order==4:

        out[small]=(abs(x-2)**beta-4*abs(x-1)**beta+6*x**beta-4*(x+1)**beta+(x+2)**beta).astype(float)

    elif order==2:

        out[small]=((x+1)**beta-2*x**beta+np.maximum(x-1,0)**beta).astype(float)

    else: raise ValueError(order)

    x=k[~small]

    value=np.zeros(len(x));bc=1.

    for j in range(1,29):

        bc*= (beta-j+1)/j

        if j%2==0:

            factor=2*(2**j-4) if order==4 else 2

            if order==4 and j<4: continue

            value+=factor*bc*x**(beta-j)

    out[~small]=value

    return out

class Halfline:

    def __init__(self,a,B,h,kind="cycle"):

        if not (1<a<=2 and B>0 and h>0): raise ValueError("Require 1<alpha<=2 and positive B,h")

        if kind not in ("cycle","reset"): raise ValueError("Unsupported gate kind")

        self.a=a;self.s=a/2;self.B=B;self.h=h;self.kind=kind

        K=round(B/h)

        if abs(K*h-B)>1e-10:raise ValueError('B/h must be integer')

        self.n=2*K-1;self.ks=np.arange(-K+1,K)

        self.x=self.ks*h

        d=np.arange(self.n,dtype=float)

        if a==2:

            col=np.zeros(self.n);col[0]=2/h;col[1]=-1/h

        else:

            col=h**(1-a)*centered_diff_power(d,3-a,4)/(2*special.gamma(4-a)*np.cos(np.pi*a/2))

        self.col=col

        self.nfft=fft.next_fast_len(2*self.n-1)

        emb=np.zeros(self.nfft);emb[:self.n]=col;emb[-(self.n-1):]=col[1:][::-1]

        self.fc=fft.rfft(emb)

        self.force=np.zeros(self.n)

        ii=self.ks<=0;k=-self.ks[ii].astype(float)

        self.force[ii]=special.gamma(1+self.s)/special.gamma(3-self.s)*h**(1-self.s)*centered_diff_power(k,2-self.s,2)

        # Strang circulant SPD approximation, used only as a preconditioner.

        pc=col[np.minimum(np.arange(self.n),self.n-np.arange(self.n))]

        spec=fft.fft(pc).real

        mass=h*(2/3+np.cos(2*np.pi*np.arange(self.n)/self.n)/3)

        rates=GATE_RATES if kind=="cycle" else np.r_[0.,np.ones(M-1)]

        self.den=spec[:,None]+mass[:,None]*rates[None,:]

        # The Brownian periodic reference has a null constant mode. Only this

        # preconditioner mode is regularized; the solved operator is unchanged.

        self.den[0,0]=max(self.den[0,0],h/B**a)

        if self.den.min()<=0:raise ArithmeticError(('preconditioner',self.den.min()))

    def mv_full(self,x):

        x=x.reshape(self.n,M)

        fx=fft.rfft(x,n=self.nfft,axis=0)

        y=fft.irfft(self.fc[:,None]*fx,n=self.nfft,axis=0)[:self.n]

        hx=(x-.5*np.roll(x,1,axis=1)-.5*np.roll(x,-1,axis=1)) if self.kind=="cycle" else x-x.mean(axis=1,keepdims=True)

        y+=self.h*2/3*hx

        y[1:]+=self.h/6*hx[:-1];y[:-1]+=self.h/6*hx[1:]

        return y

    def solve(self,R,rtol=2e-10):

        active=np.ones((self.n,M),dtype=bool)

        active[np.ix_(self.ks<=0,np.array(R,dtype=int))]=False

        shape=active.shape

        def embed(v):

            z=np.zeros(shape);z[active]=v;return z

        def mv(v):return self.mv_full(embed(v))[active]

        def pmv(v):

            z=embed(v);p=fft.ifftn(fft.fftn(z)/self.den).real

            return p[active]

        C=sorted(set(range(M))-set(R));f=np.zeros(shape);f[:,C]=self.force[:,None];rhs=f[active]

        A=LinearOperator((rhs.size,rhs.size),matvec=mv,dtype=float)

        pre=LinearOperator(A.shape,matvec=pmv,dtype=float)

        it=[0]

        def cb(v):it[0]+=1

        t=time.perf_counter();v,info=cg(A,rhs,M=pre,rtol=rtol,atol=0,maxiter=3000,callback=cb)

        av=mv(v);res=rhs-av

        quadratic=(2*np.dot(rhs,v)-np.dot(v,av))/(M*special.gamma(1+self.s)**2)

        return {'coefficient_lower_galerkin':float(quadratic),'linear_estimate':float(np.dot(rhs,v)/(M*special.gamma(1+self.s)**2)),

                'iterations':it[0],'cg_info':int(info),'residual_relative':float(linalg.norm(res)/linalg.norm(rhs)),

                'residual_linf':float(np.max(abs(res))),'seconds':time.perf_counter()-t,'active_dofs':int(rhs.size)},embed(v)

def run(a,B,h,rtol=2e-10,kind="cycle"):

    t=time.perf_counter();H=Halfline(a,B,h,kind);rows=[]

    for R in RSETS:

        r,z=H.solve(R,rtol);rows.append(r)

    return {'alpha':a,'B':B,'h':h,'kind':kind,'nodes':H.n,'A':rows[0],'B_gate':rows[1],

            'coefficient_gap':rows[0]['coefficient_lower_galerkin']-rows[1]['coefficient_lower_galerkin'],

            'seconds':time.perf_counter()-t,'note':'Exact stiffness/source formulas, floating solves. Each energy is a variational lower approximation; their difference is not a rigorous signed bound on the true coefficient gap.'}

if __name__=='__main__':

    group=sys.argv[1] if len(sys.argv)>1 else 'pilot';P=Path(__file__).resolve().parents[1]

    if group=='pilot':cases=[(a,B,1/16) for a in [1.5,1.75,1.9,2.] for B in [16.,32.,64.]]

    elif group=='refine':cases=[(a,B,h) for a in [1.5,1.75,1.9,2.] for B,h in [(64.,1/32),(64.,1/64),(128.,1/32),(256.,1/32)]]

    else:raise ValueError(group)

    (P/'results'/f'halfline_{group}_config.json').write_text(json.dumps(cases,indent=2))

    for a,B,h in cases:

        r=run(a,B,h)

        with (P/'results'/f'halfline_{group}.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')

        print(json.dumps(r),flush=True)

