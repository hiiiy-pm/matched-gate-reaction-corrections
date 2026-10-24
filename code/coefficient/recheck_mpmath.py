"""Independent mpmath.iv residual integration over selected WHOLE alpha slabs.

The Taylor objective-difference bound is imported from the separately checked
Arb certificate. This replay cross-checks residual arithmetic, not the Taylor
engine, and does not pretend to be a second full-range proof.
"""
import probe_imports as p
import json,time,heapq,argparse
from fractions import Fraction as F
import mpmath as mp
import certify_dual_point as c
I=mp.iv
def iv(q):
    q=F(q);return I.mpf(q.numerator)/q.denominator

def run(index):
    mp.mp.dps=100;I.dps=70
    folder=p.ROOT.parents[1]/'data/continuous_coefficient';root=json.loads((folder/'continuous_certificate.json').read_text())
    row=root['slabs'][index];original=json.loads((folder/row['file']).read_text())
    cfg=json.loads((folder/original['coordinate_file']).read_text())
    lo,hi=map(F,(row['alpha_lo'],row['alpha_hi']))
    s=I.mpf([mp.mpf(iv(lo).a),mp.mpf(iv(hi).b)])/2
    model=c.old.eng.Model(p.MASKS['B'],18,dps=70)
    density,left,right,const,ps,zs=c.operators(model,cfg,s)
    l0,h0=c.old.tails(model,left,right,const,s,32);lq,hq=c.force_tails(model,ps,zs,s,32)
    low=(I.sqrt(l0)+I.sqrt(lq))**2;high=(I.sqrt(h0)+I.sqrt(hq))**2
    gamma=I.gamma(1+s)**2
    # Decimal serialization is an explicitly outward bound, treated exactly.
    gaplo=I.mpf(str(original['trial_gap'][0]))
    target=mp.mpf((gaplo*gamma*I.mpf('0.85')).a)
    heap=[];counter=0;start=time.perf_counter()
    def cell(a,b):
        nonlocal counter
        val=density(I.mpf([mp.mpf(iv(a).a),mp.mpf(iv(b).b)]),a>=0)*iv(b-a)
        item=(-float(val.delta),counter,a,b,val);counter+=1;return item
    for k in range(128):heapq.heappush(heap,cell(F(-32)+F(k,2),F(-32)+F(k+1,2)))
    upper=c.sm(I.mpf(z[4].b) for z in heap)
    while len(heap)<10000 and mp.mpf(upper.b)>target:
        _,_,a,b,val=heapq.heappop(heap);mid=(a+b)/2;children=[cell(a,mid),cell(mid,b)]
        for z in children:heapq.heappush(heap,z)
        upper=upper-I.mpf(val.b)+c.sm(I.mpf(z[4].b) for z in children)
        if len(heap)%1024==0:print(json.dumps({'slab':index,'panels':len(heap),'residual_upper_indicator':float((upper/gamma).b),'seconds':time.perf_counter()-start}),flush=True)
    integral=c.sm(z[4] for z in heap)
    residual=(integral+I.mpf([0,mp.mpf(low.b)])+I.mpf([0,mp.mpf(high.b)]))/gamma
    lower=gaplo-I.mpf(residual.b)
    result={'status':'PASS_SELECTED_SLAB_RESIDUAL_REPLAY' if float(lower.a)>0 else 'UNRESOLVED_SELECTED_SLAB',
      'slab_index':index,'alpha':[str(lo),str(hi)],'dps':70,'lower_bound':c.old.bounds(lower)[0],
      'residual_squared':c.old.bounds(residual),'panels':len(heap),'seconds':time.perf_counter()-start,
      'scope':'Whole alpha slab residual independently enclosed with mpmath.iv; Taylor objective difference imported from Arb.',
      'frequency_grid':[[str(z[2]),str(z[3])] for z in sorted(heap,key=lambda z:z[2])]}
    (folder/f'mpmath_slab_{index:02d}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frequency_grid'}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('indices',nargs='+',type=int);args=ap.parse_args()
    for index in args.indices:run(index)
