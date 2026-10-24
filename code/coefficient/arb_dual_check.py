"""Second enclosing backend (Arb) with direct interval matrix solve.

Shares analytic residual formulas with the mpmath implementation, not an
independent analytic proof. The arithmetic, solve, and adaptive grid differ.
"""
import probe_imports as p
import json,time,heapq,hashlib,inspect
from fractions import Fraction
from flint import arb,acb,ctx
import interval_engine_arb as ab
import certify_dual_point as cert
from extended_source_residual_pilot import create

class Arithmetic:
    pi=arb.pi()
    j=acb(0,1)
    def mpf(self,x):
        if isinstance(x,(list,tuple)):return self.mpf(x[0]).union(self.mpf(x[1]))
        if isinstance(x,Fraction):return arb(x.numerator)/arb(x.denominator)
        return arb(x)
    def mpc(self,r,i=0):return acb(r,i)
    def sqrt(self,x):return arb(x).sqrt()
    def cos(self,x):return arb(x).cos()
    def sin(self,x):return arb(x).sin()
    def gamma(self,x):return arb(x).gamma()
    def loggamma(self,x):return arb(x).lgamma()
    def exp(self,x):return (x if isinstance(x,(arb,acb)) else arb(x)).exp()
    def expm1(self,x):return x.expm1()

def objective(model,left,right,s):
    rho=2*s-1;den=2*(arb.pi()*rho/2).sin()
    sm=ab.isum;dot=ab.dot
    def same(a,b):return (a**(rho-1)+b**(rho-1))/(den*(a+b))
    def cross(a,b):
        diff=b-a
        if not(diff>0 or diff<0):return (1-rho)*a.union(b)**(rho-2)/den
        return (a**(rho-1)-b**(rho-1))/(den*diff)
    kinetic=arb(0);gate=arb(0);n=model.n
    def htimes(x):return [x[j]-(x[(j-1)%n]+x[(j+1)%n])/2 for j in range(n)]
    for side in (left,right):
        for i,(a,ar) in enumerate(side):
            ha=htimes(a)
            for j,(b,br) in enumerate(side[:i+1]):
                fac=1 if i==j else 2
                kinetic+=fac*dot(a,b)*ar*br*same(ar,br)
                gate+=fac*dot(ha,b)/(ar+br)
    for a,ar in left:
        for b,br in right:kinetic-=2*dot(a,b)*ar*br*cross(ar,br)
    source=(1+s).gamma()*sm(sm(a)*ar**(s-1) for a,ar in left)
    return (2*source-kinetic-gate)/(n*(1+s).gamma()**2)

def main():
    ctx.prec=250
    arithmetic=Arithmetic();arithmetic.pi=arb.pi()
    cert.I=arithmetic;cert.old.I=arithmetic
    # Arb's generic power may be indeterminate for a ball crossing zero.
    # For a squared real expression, multiplication followed by intersection
    # with the nonnegative halfline is a valid enclosure.
    def safe_norm(xs):return ab.isum(x*x for x in xs).nonnegative_part().sqrt()
    cert.old.norm=safe_norm;cert.norm=safe_norm
    namespace=dict(cert.__dict__)
    operator_source=inspect.getsource(cert.operators).replace(
        'r.real**2+r.imag**2',
        '(r.real*r.real).nonnegative_part()+(r.imag*r.imag).nonnegative_part()')
    exec(operator_source,namespace)
    cfgpath=p.ROOT/'dual_B_1p9_L8_F4_Z1.json';cfg=json.loads(cfgpath.read_text())
    model=ab.Model(p.MASKS['B'],18,prec=250);s=arb('1.9')/2
    density,left,right,constant,ps,zs=namespace['operators'](model,cfg,s)
    jb=objective(model,left,right,s)
    l0,h0=cert.old.tails(model,left,right,constant,s,32)
    lq,hq=cert.force_tails(model,ps,zs,s,32)
    low=(l0.sqrt()+lq.sqrt())**2;high=(h0.sqrt()+hq.sqrt())**2
    ma=ab.Model(p.MASKS['A'],18,prec=250)
    details=create(18,p.MASKS['A'],1.9,[float(x) for x in cfg['trial_rates_exact_decimal']],1e-12,True,return_details=True)
    acoords=[str(float(x)) for x in details['coefficients']]
    al,ar,ac=cert.old.physical_trial(ma,acoords,cfg['trial_rates_exact_decimal'])
    ja=objective(ma,al,ar,s)
    print(json.dumps({'stage':'setup','trial_gap':ab.bounds(ja-jb),'low':ab.bounds(low),'high':ab.bounds(high)}),flush=True)
    heap=[];counter=0;start=time.perf_counter()
    def cell(a,b):
        nonlocal counter
        assert a>=0 or b<=0
        t=arithmetic.mpf([a,b]);val=density(t,a>=0)*arithmetic.mpf(b-a)
        width=arb(val.upper())-arb(val.lower())
        item=(-float(width),counter,a,b,val);counter+=1
        return item
    for k in range(128):heapq.heappush(heap,cell(Fraction(-32)+Fraction(k,2),Fraction(-32)+Fraction(k+1,2)))
    upper=ab.isum(arb(z[4].upper()) for z in heap)
    target=arb((ja-jb).lower())*(1+s).gamma()**2/2
    while len(heap)<8192 and not upper<target:
        _,_,a,b,val=heapq.heappop(heap);mid=(a+b)/2
        children=[cell(a,mid),cell(mid,b)]
        for z in children:heapq.heappush(heap,z)
        upper=upper-arb(val.upper())+ab.isum(arb(z[4].upper()) for z in children)
        if len(heap)%512==0:print(json.dumps({'panels':len(heap),'integral_upper':ab.outward_float(upper,True),'seconds':time.perf_counter()-start}),flush=True)
    ordered=sorted(heap,key=lambda z:z[2])
    assert ordered[0][2]==-32 and ordered[-1][3]==32
    assert all(x[3]==y[2] for x,y in zip(ordered,ordered[1:]))
    integral=ab.isum(z[4] for z in ordered)
    residual=(integral+ab.interval_hull(0,low.upper())+ab.interval_hull(0,high.upper()))/(1+s).gamma()**2
    lower=ja-jb-arb(residual.upper())
    out={'status':'POINT_BOUND_WITH_SECOND_ENCLOSING_BACKEND','alpha_exact':'1.9',
         'continuous_alpha_range':False,'backend':'python-flint/Arb','precision_bits':250,
         'trial_A':ab.bounds(ja),'trial_B':ab.bounds(jb),'trial_gap':ab.bounds(ja-jb),
         'residual_B_squared_with_tails':ab.bounds(residual),
         'gap_lower_bound':ab.outward_float(lower),'positive_lower_bound':bool(lower>0),
         'panels':len(heap),'low_tail':ab.bounds(low),'high_tail':ab.bounds(high),
         'finite_window_integral':ab.bounds(integral),'seconds':time.perf_counter()-start,
         'exact_log_frequency_grid':[[str(z[2]),str(z[3])] for z in ordered],
         'A_coordinates_exact_decimal':acoords,'source_sha256':hashlib.sha256(p.Path(__file__).read_bytes()).hexdigest(),
         'dual_coordinates_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),
         'contract':'Shares analytic source/residual/tail formulas with mpmath implementation; second arithmetic backend, not a second analytic proof.'}
    (p.ROOT/'point_1p9_arb250_adaptive.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if 'grid' not in k and 'coordinates' not in k}),flush=True)
if __name__=='__main__':main()
