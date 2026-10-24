"""Conditional enclosing-arithmetic pilot for one fixed analytic trial.

Uses the original SI/interval_engine enclosing-arithmetic contract. It does
not establish a separate formal verification of mpmath interval arithmetic.
No frozen prediction, manuscript, or inherited scientific data is changed.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import mpmath as mp
from extended_source_residual_pilot import create

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'reproducible'))
import interval_engine as eng
I=mp.iv


def sm(xs):return sum(xs,I.mpf(0))
def dot(a,b):return sm(x*y for x,y in zip(a,b))
def norm(a):return I.sqrt(sm(x**2 for x in a))
def mean(a):return sm(a)/len(a)
def project(a,common):
    av=mean(a)
    return [av]*len(a) if common else [x-av for x in a]
def bounds(a):return eng.bounds(a)
def mpstr(x):return mp.nstr(mp.mpf(x),65)


def physical_trial(model,coeff,rates):
    n=model.n;nr=len(model.C)
    p=[I.mpf(0)]*n
    for j,x in zip(model.C,coeff[:nr]):p[j]=I.mpf(x)
    left=[([a*x for x in v],mu) for v,mu in zip(model.path,model.mu) for a in [dot(v,p)]]
    right=[([a*x for x in u],lam) for u,lam in zip(model.ring,model.lam) for a in [dot(u,p)]]
    constant=mean(p)
    dl=[[I.mpf(0)]*n for _ in rates];dr=[[I.mpf(0)]*n for _ in rates]
    pos=nr
    for j in model.C:
        for k in range(len(rates)):dl[k][j]=I.mpf(coeff[pos]);pos+=1
    for j in range(n):
        for k in range(len(rates)):dr[k][j]=I.mpf(coeff[pos]);pos+=1
    lref=[-sm(row[j] for row in dl) for j in range(n)]
    rref=[-sm(row[j] for row in dr) for j in range(n)]
    constant+=I.mpf(coeff[pos])
    rref=[x-I.mpf(coeff[pos]) for x in rref]
    left.extend((row,I.mpf(str(a))) for row,a in zip(dl,rates))
    right.extend((row,I.mpf(str(a))) for row,a in zip(dr,rates))
    left.append((lref,I.mpf(1)));right.append((rref,I.mpf(1)))
    return left,right,constant


def brownian_physical(model,delta_boundary=None):
    p=model.b if delta_boundary is None else delta_boundary
    left=[([a*x for x in v],mu) for v,mu in zip(model.path,model.mu) for a in [dot(v,p)]]
    right=[([a*x for x in u],lam) for u,lam in zip(model.ring,model.lam) for a in [dot(u,p)]]
    return left,right,mean(p)


def eig_project(rows,us):
    return [(tuple(dot(row,u) for u in us),rate) for row,rate in rows]


def operators(model,coeff,rates,s):
    n=model.n
    us=[[1/I.sqrt(n)]*n]+model.ring
    eig=[I.mpf(0)]+[l*l for l in model.lam]
    left,right,constant=physical_trial(model,coeff,rates)
    oldleft,oldright,oldconstant=brownian_physical(model)
    # Construct the small difference in coefficient space, not by subtracting
    # two interval Fourier functions of size O(1).
    delta_boundary=[-x for x in model.b]
    for j,x in zip(model.C,coeff[:len(model.C)]):delta_boundary[j]+=I.mpf(x)
    dl,dr,dc=brownian_physical(model,delta_boundary)
    dl.extend(left[len(model.mu):]);dr.extend(right[len(model.lam):])
    dc+=I.mpf(coeff[-1])
    oldlp=eig_project(oldleft,us);oldrp=eig_project(oldright,us)
    dlp=eig_project(dl,us);drp=eig_project(dr,us)
    oldcp=oldconstant*I.sqrt(n);dcp=dc*I.sqrt(n)
    delta=1-s;lg=I.loggamma(1+s);phase=I.mpc(0,I.pi*delta/2)
    def transform(lp,rp,cp,x,high):
        if high:
            li=[a/(a-I.j*x) for _,a in lp];ri=[-a/(a+I.j*x) for _,a in rp]
            return [(sm(row[k]*z for (row,_),z in zip(lp,li))+
                     sm(row[k]*z for (row,_),z in zip(rp,ri)))/(I.j*x) for k in range(n)]
        li=[1/(a-I.j*x) for _,a in lp];ri=[1/(a+I.j*x) for _,a in rp]
        out=[sm(row[k]*z for (row,_),z in zip(lp,li))+
             sm(row[k]*z for (row,_),z in zip(rp,ri)) for k in range(n)]
        out[0]+=cp/(I.j*x)
        return out
    def density(t,high):
        x=I.exp(t);power=I.exp(2*s*t)
        old=transform(oldlp,oldrp,oldcp,x,high)
        dz=transform(dlp,drp,dcp,x,high)
        eplus=I.expm1(I.mpc(lg+delta*t,phase.imag))
        eminus=I.expm1(I.mpc(lg-delta*t,phase.imag))
        total=I.mpf(0)
        for k,h in enumerate(eig):
            symbol=power+h
            r=(power*eplus+h*eminus)*old[k]-symbol*dz[k]
            total+=(r.real**2+r.imag**2)/symbol
        return total*x/(n*I.pi)
    return density,left,right,constant


def trial_objective(model,left,right,s):
    n=model.n;rho=2*s-1;den=2*I.sin(I.pi*rho/2)
    def same(a,b):return (a**(rho-1)+b**(rho-1))/(den*(a+b))
    def cross(a,b):
        diff=b-a
        if float(diff.a)<=0<=float(diff.b):
            hull=I.mpf([min(mp.mpf(a.a),mp.mpf(b.a)),max(mp.mpf(a.b),mp.mpf(b.b))])
            return (1-rho)*hull**(rho-2)/den
        return (a**(rho-1)-b**(rho-1))/(den*diff)
    kinetic=I.mpf(0);gate=I.mpf(0)
    def htimes(x):return [x[j]-(x[(j-1)%n]+x[(j+1)%n])/2 for j in range(n)]
    for side in (left,right):
        for i,(a,ar) in enumerate(side):
            ha=htimes(a)
            for j,(b,br) in enumerate(side[:i+1]):
                fac=1 if j==i else 2
                kinetic+=fac*dot(a,b)*ar*br*same(ar,br)
                gate+=fac*dot(ha,b)/(ar+br)
    for a,ar in left:
        for b,br in right:kinetic-=2*dot(a,b)*ar*br*cross(ar,br)
    source=I.gamma(1+s)*sm(sm(a)*ar**(s-1) for a,ar in left)
    return (2*source-kinetic-gate)/(n*I.gamma(1+s)**2)


def tails(model,left,right,constant,s,window):
    n=model.n
    def htimes(x):return [x[j]-(x[(j-1)%n]+x[(j+1)%n])/2 for j in range(n)]
    j=[sm(l*b*u[k] for l,b,u in zip(model.lam,model.beta,model.ring))+
       sm(l*a*v[k] for l,a,v in zip(model.mu,model.a,model.path)) for k in range(n)]
    for k in model.C:j[k]=I.mpf(1)
    volumes=[]
    for v,a,mu in zip(model.path,model.a,model.mu):
        hv=htimes([x*a for x in v])
        for k in model.C:hv[k]=I.mpf(0)
        volumes.append((hv,mu))
    # The exact cycle spectral gap/max values are known analytically.
    minimum=1-I.cos(2*I.pi/n)
    maximum=I.mpf(2) if n%2==0 else 1+I.cos(I.pi/n)
    low=I.exp(-window);up=I.exp(window);gam=I.gamma(1+s)
    J=norm(j);V1=sm(norm(v) for v,_ in volumes)
    D1=sm(rate*norm(a) for side in (left,right) for a,rate in side)
    high=(gam*(J+V1/up)+D1*(up**(s-1)+maximum*up**(-s-1)))**2/(n*I.pi*up)
    vp=sm(norm(project(v,True))/mu**2 for v,mu in volumes)
    zp=sm(norm(project(a,True))/rate for side in (left,right) for a,rate in side)
    zq=sm(norm(project(a,False))/rate for side in (left,right) for a,rate in side)
    fq=norm(project(j,False))+sm(norm(project(v,False))/mu for v,mu in volumes)
    cp=gam*vp*low**(1-s)+I.sqrt(n)*abs(constant)+zp*low
    cq=gam*fq/I.sqrt(minimum)+I.sqrt(low**(2*s)+maximum)*zq*low**(1-s)
    lowbound=(cp**2+cq**2)*low**(2*s-1)/(n*I.pi*(2*s-1))
    return lowbound,high


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gate',choices=['A','B'],default='A')
    ap.add_argument('--panels',type=int,nargs='+',default=[32,64,128])
    ap.add_argument('--dps',type=int,default=40);ap.add_argument('--window',type=int,default=32)
    args=ap.parse_args();mp.mp.dps=args.dps+20;I.dps=args.dps
    start=time.perf_counter()
    cfgpath=ROOT/'reproducible/results/round2/interval_C18_0064_95.json'
    cfg=json.loads(cfgpath.read_text());opened=cfg[args.gate+'_open']
    rates=[2.**k for k in range(-4,5) if k]
    frozen=HERE/f'interval_trial_coordinates_{args.gate}.json'
    if frozen.exists():coords=json.loads(frozen.read_text())
    else:
        details=create(cfg['n'],opened,1.999,rates,1e-12,True,return_details=True)
        coords={'gate':args.gate,'alpha_exact_decimal':'1.999','rates':[str(x) for x in rates],
                'coordinates_exact_decimal':[str(float(x)) for x in details['coefficients']],
                'meaning':'Frozen rational coordinates in the exact continuous cycle/run analytic basis; generated in floating arithmetic.'}
        frozen.write_text(json.dumps(coords,indent=2)+'\n',encoding='utf-8')
    model=eng.Model(opened,cfg['n'],dps=args.dps)
    s=I.mpf('1.999')/2
    density,left,right,constant=operators(model,coords['coordinates_exact_decimal'],rates,s)
    obj=trial_objective(model,left,right,s)
    low,high=tails(model,left,right,constant,s,args.window)
    rows=[]
    target=HERE/f'interval_residual_pilot_{args.gate}.json'
    for count in args.panels:
        assert count%2==0
        t0=time.perf_counter();integral=I.mpf(0)
        width=I.mpf(2*args.window)/count
        for k in range(count):
            al=I.mpf(-args.window)+k*width;be=al+width
            t=I.mpf([mp.mpf(al.a),mp.mpf(be.b)])
            value=density(t,k>=count//2)
            integral+=value*width
        full=integral+I.mpf([0,mp.mpf(low.b)])+I.mpf([0,mp.mpf(high.b)])
        normalized=full/I.gamma(1+s)**2
        row={'panels':count,'log_window':[-args.window,args.window],
             'finite_window_integral_enclosure':bounds(integral),
             'low_tail_upper':eng.hi(low),'high_tail_upper':eng.hi(high),
             'rho_squared_enclosure_with_tails':bounds(full),
             'normalized_residual_squared_enclosure':bounds(normalized),
             'normalized_coefficient_enclosure':bounds(obj+I.mpf([0,mp.mpf(normalized.b)])),
             'seconds':time.perf_counter()-t0}
        rows.append(row)
        result={'status':'CONDITIONAL_INTERVAL_RECTANGLE_PILOT',
                'contract':'Conditional on the original SI analytic assertions and enclosing-arithmetic implementation; not external formal verification.',
                'scope':'One original-parameter local coefficient; no geometry, circle discretization, or finite-gamma remainder certification.',
                'gate':args.gate,'alpha_exact_decimal':'1.999','dps':args.dps,
                'trial_objective_normalized_enclosure':bounds(obj),'rows':rows,
                'coordinates_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'seconds':time.perf_counter()-start}
        target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
