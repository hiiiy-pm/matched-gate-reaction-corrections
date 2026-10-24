"""Continuous-alpha residual certificate for Pair II, with exact input slabs.

Floating optimization supplies admissible rational coordinates only. All
acceptance decisions use Arb enclosures over a WHOLE alpha interval and the
whole frequency window, plus analytic infinite tails. No sampling inference.
"""
import probe_imports as p
import json,inspect,heapq,time,hashlib
from fractions import Fraction as F
from flint import arb,arb_series,ctx
import interval_engine_arb as ab
import certify_dual_point as cert
import arb_dual_check as point
import dual_probe
from extended_source_residual_pilot import create

def setup(bits):
    ctx.prec=bits
    api=point.Arithmetic();api.pi=arb.pi();cert.I=api;cert.old.I=api
    def safe_norm(xs):return ab.isum(x*x for x in xs).nonnegative_part().sqrt()
    cert.old.norm=safe_norm;cert.norm=safe_norm
    namespace=dict(cert.__dict__)
    source=inspect.getsource(cert.operators).replace('r.real**2+r.imag**2',
       '(r.real*r.real).nonnegative_part()+(r.imag*r.imag).nonnegative_part()')
    exec(source,namespace)
    models={k:ab.Model(v,18,prec=bits) for k,v in p.MASKS.items()}
    return api,models,namespace['operators']

def generate_coordinates(center,level=8):
    value=float(center)
    b=dual_probe.optimize(value,'B',level,4,True)
    a=create(18,p.MASKS['A'],value,[float(x) for x in b['trial_rates_exact_decimal']],1e-12,True,return_details=True)
    b['anchor_alpha_exact']=str(center)
    b['A_coordinates_exact_decimal']=[str(float(x)) for x in a['coefficients']]
    return b

def group_rows(model,rows,side,rates):
    # Exact spectral labels, not floating equality, identify equal decay rates.
    angles=model.pa if side=='left' else model.ra
    tags=[('c',str(x)) if x!=F(1,2) else ('r','1') for x in angles]
    tags += [('r',str(F(a))) for a in rates]+[('r','1')]
    assert len(tags)==len(rows)
    grouped={}
    for tag,(row,rate) in zip(tags,rows):
        if tag not in grouped:grouped[tag]=([arb(0) for _ in range(model.n)],rate)
        acc,r=grouped[tag]
        # Label equivalence proves equality even when balls have different radii.
        for j,x in enumerate(row):acc[j]+=x
    return grouped

class JointObjective:
    """Cancel A/B coefficients BEFORE evaluating alpha-dependent kernels."""
    def __init__(self,models,cfg):
        grouped={}
        for key,model in models.items():
            coords=cfg['A_coordinates_exact_decimal'] if key=='A' else cfg['trial_coordinates_exact_decimal']
            left,right,c=cert.old.physical_trial(model,coords,cfg['trial_rates_exact_decimal'])
            grouped[key]={side:group_rows(model,rows,side,cfg['trial_rates_exact_decimal']) for side,rows in [('left',left),('right',right)]}
        self.same=[];self.cross=[];self.source=[];self.gate=arb(0);self.n=18
        def H(x):return [x[j]-(x[(j-1)%18]+x[(j+1)%18])/2 for j in range(18)]
        for side in ('left','right'):
            A=grouped['A'][side];B=grouped['B'][side]
            assert set(A)==set(B)
            keys=sorted(A)
            for i,ki in enumerate(keys):
                va,ra=A[ki];vb,rb=B[ki]
                if side=='left':self.source.append((ab.isum(va)-ab.isum(vb),ra))
                for j,kj in enumerate(keys[:i+1]):
                    wa,sa=A[kj];wb,sb=B[kj];fac=1 if i==j else 2
                    self.same.append((fac*(ab.dot(va,wa)-ab.dot(vb,wb))*ra*sa,ra,sa))
                    self.gate+=fac*(ab.dot(H(va),wa)-ab.dot(H(vb),wb))/(ra+sa)
        for kl in grouped['A']['left']:
            va,ra=grouped['A']['left'][kl];vb,_=grouped['B']['left'][kl]
            for kr in grouped['A']['right']:
                wa,sa=grouped['A']['right'][kr];wb,_=grouped['B']['right'][kr]
                self.cross.append((-2*(ab.dot(va,wa)-ab.dot(vb,wb))*ra*sa,ra,sa,kl==kr))
    def __call__(self,s):
        rho=2*s-1;den=2*(arb.pi()*rho/2).sin()
        power=lambda a,t:(t*a.log()).exp()
        kinetic=ab.isum(c*(power(a,rho-1)+power(b,rho-1))/(den*(a+b)) for c,a,b in self.same)
        for c,a,b,equal in self.cross:
            z=(rho-1)*(b/a).log()
            em1=z.expm1() if isinstance(z,arb) else z.exp()-1
            kernel=((1-rho)*power(a,rho-2)/den if equal else
                    -power(a,rho-1)*em1/(den*(b-a)))
            kinetic+=c*kernel
        source=(1+s).gamma()*ab.isum(c*power(a,s-1) for c,a in self.source)
        return (2*source-kinetic-self.gate)/(self.n*(1+s).gamma()**2)

def certify_slab(lo,hi,cfg,bits=250,maxpanels=8192,target_fraction=F(3,5),grid=None):
    start=time.perf_counter();api,models,operator=setup(bits)
    lo,hi=F(lo),F(hi);assert F(19,10)<=lo<=hi<=2
    s=api.mpf([lo,hi])/2
    joint=JointObjective(models,cfg)
    # Taylor's mean-value enclosure: derivative is enclosed on the full slab.
    center=api.mpf((lo+hi)/4);offset=api.mpf([-(hi-lo)/4,(hi-lo)/4])
    derivative=joint(arb_series([s,arb(1)],2))[1]
    gap=joint(center)+offset*derivative
    model=models['B'];density,left,right,constant,ps,zs=operator(model,cfg,s)
    l0,h0=cert.old.tails(model,left,right,constant,s,32)
    lq,hq=cert.force_tails(model,ps,zs,s,32)
    low=(l0.sqrt()+lq.sqrt())**2;high=(h0.sqrt()+hq.sqrt())**2
    gam2=(1+s).gamma()**2
    target=arb(gap.lower())*arb(gam2.lower())*api.mpf(target_fraction)
    if not gap>0:
        return {'alpha_lo':str(lo),'alpha_hi':str(hi),'positive':False,'reason':'trial_gap_interval_contains_zero',
                'trial_gap':ab.bounds(gap),'seconds':time.perf_counter()-start}
    heap=[];counter=0
    def cell(a,b):
        nonlocal counter
        assert a<b and (a>=0 or b<=0)
        val=density(api.mpf([a,b]),a>=0)*api.mpf(b-a)
        width=arb(val.upper())-arb(val.lower())
        item=(-float(width),counter,a,b,val);counter+=1;return item
    if grid is not None:
        for a,b in grid:heap.append(cell(F(a),F(b)))
    else:
        for k in range(128):heapq.heappush(heap,cell(F(-32)+F(k,2),F(-32)+F(k+1,2)))
        upper=ab.isum(arb(z[4].upper()) for z in heap)
        while len(heap)<maxpanels and not upper<target:
            _,_,a,b,val=heapq.heappop(heap);mid=(a+b)/2
            children=[cell(a,mid),cell(mid,b)]
            for z in children:heapq.heappush(heap,z)
            upper=upper-arb(val.upper())+ab.isum(arb(z[4].upper()) for z in children)
    ordered=sorted(heap,key=lambda z:z[2])
    assert ordered[0][2]==-32 and ordered[-1][3]==32
    assert all(x[3]==y[2] for x,y in zip(ordered,ordered[1:]))
    integral=ab.isum(z[4] for z in ordered)
    residual=(integral+ab.interval_hull(0,low.upper())+ab.interval_hull(0,high.upper()))/gam2
    lower=gap-arb(residual.upper())
    result={'alpha_lo':str(lo),'alpha_hi':str(hi),'alpha_anchor':cfg['anchor_alpha_exact'],
            'positive':bool(lower>0),'lower_bound':ab.outward_float(lower),
            'trial_gap':ab.bounds(gap),'trial_gap_derivative_wrt_s':ab.bounds(derivative),'residual_B_squared':ab.bounds(residual),
            'low_tail':ab.bounds(low),'high_tail':ab.bounds(high),'panels':len(ordered),
            'frequency_grid':[[str(z[2]),str(z[3])] for z in ordered],
            'precision_bits':bits,'seconds':time.perf_counter()-start,
            'scope':'ALL alpha in the exact closed interval, with pointwise lower bound uniform on this interval'}
    return result

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--lo',default='1.9');ap.add_argument('--hi',default='1.901')
    ap.add_argument('--panels',type=int,default=4096);args=ap.parse_args()
    lo,hi=F(args.lo),F(args.hi);cfg=generate_coordinates((lo+hi)/2)
    dest=p.ROOT/'pilot';dest.mkdir(exist_ok=True)
    name=f'{str(lo).replace("/","_")}_{str(hi).replace("/","_")}'
    (dest/(name+'_coordinates.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    row=certify_slab(lo,hi,cfg,maxpanels=args.panels)
    (dest/(name+'_certificate.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='frequency_grid'}),flush=True)
