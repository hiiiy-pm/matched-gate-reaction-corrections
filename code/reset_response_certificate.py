"""Fresh independent-arithmetic interval bounds and explicit reset neighborhoods."""
from pathlib import Path
from decimal import Decimal
import sys,json,time,hashlib
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'code/models/reproducible'))
import interval_engine as iv
import interval_engine_arb as ab
import mpmath as mp
from flint import arb,ctx
OUT=R/'results/family_certificates';OUT.mkdir(parents=True,exist_ok=True)
ws=json.loads((R/'code/models/reproducible/witnesses.json').read_text())
specs=[('C18_0189','1.9982',360,1),('C18_0220','1.999999',100,-1)]
records=[];start=time.perf_counter()
for wid,lower,count,sign in specs:
    w=next(w for w in ws if w['id']==wid)
    backend_rows={}
    for dps in [70,95]:
        mp.mp.dps=dps+20;mp.iv.dps=dps
        A=iv.Model(w['A'],18,dps);B=iv.Model(w['B'],18,dps)
        lo=Decimal(lower);step=(Decimal(2)-lo)/count
        rows=[iv.certify(lo+j*step,lo+(j+1)*step,A,B) for j in range(count)]
        assert all(r['gap'][0]>0 if sign>0 else r['gap'][1]<0 for r in rows)
        backend_rows[str(dps)]=rows
        (OUT/f'{wid}_iv_{dps}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(wid,'iv',dps,'PASS',flush=True)
    A=ab.Model(w['A'],18,prec=250);B=ab.Model(w['B'],18,prec=250)
    lo=Decimal(lower);step=(Decimal(2)-lo)/count;rows=[];upper_sums=[]
    for j in range(count):
        l=lo+j*step;h=lo+(j+1)*step
        z=ab.certify(l,h,A,B);rows.append(z)
        s=ab.interval_hull(str(l),str(h))/2;smin=arb(str(l))/2
        upper=(A.trial(s)+A.residue_square(s,smin)+B.trial(s)+B.residue_square(s,smin))/(1+s).gamma()**2
        assert upper<1
        upper_sums.append(ab.outward_float(upper,True))
        assert z['gap'][0]>0 if sign>0 else z['gap'][1]<0
        for dps in ['70','95']:
            other=backend_rows[dps][j]['gap']
            assert max(other[0],z['gap'][0])<=min(other[1],z['gap'][1])
    (OUT/f'{wid}_arb_250.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    minabs=min((r['gap'][0] if sign>0 else -r['gap'][1]) for r in rows)
    records.append({'seed':wid,'alpha_interval':[lower,'2'],'right_endpoint_excluded':True,
        'segments':count,'sign':sign,'minimum_absolute_gap_lower':minabs,
        'coefficient_sum_upper':max(upper_sums),'coefficient_sum_less_than_one':True,
        'all_three_backends_overlap':True})
    if wid=='C18_0189':
        point=ab.certify(Decimal('1.999'),Decimal('1.999'),A,B)
        assert arb(str(point['gap'][0]))>arb('0.00005')
    print(wid,'Arb PASS',minabs,flush=True)
ctx.prec=250
spectral=1/(1-(arb.pi()/9).cos())-arb(17)/18
assert spectral<17
assert records[0]['minimum_absolute_gap_lower']>3e-6
# t(theta) <= 17 theta. Delta(theta) >= [m - t M]/(1-t^2)
# with M<1. The positive numerator is itself a conservative lower bound.
boxes=[
 {'seed':'C18_0189','alpha_interval':['1.9982','2'],'theta_interval':['0','0.0000001'],
  'base_absolute_gap_lower':'0.000003','absolute_gap_lower':'0.0000013','sign':1},
 {'seed':'C18_0189','alpha_interval':['1.999','1.999'],'theta_interval':['0','0.000002'],
  'base_absolute_gap_lower':'0.00005','absolute_gap_lower':'0.000016','sign':1}]
second=records[1]['minimum_absolute_gap_lower']
if second>1e-9:
    boxes.append({'seed':'C18_0220','alpha_interval':['1.999999','2'],
        'theta_interval':['0','0.00000000001'],'base_absolute_gap_lower':'0.000000001',
        'absolute_gap_lower':'0.00000000083','sign':-1})
record={'complete':True,'base_certificates':records,'pair_ii_point':point,
        'spectral_multiplier_upper':'17','spectral_multiplier_arb':ab.bounds(spectral),
        'reset_boxes':boxes,
        'form_bound':'|Delta_theta| >= (m-t*M)/(1-t*t), with the base sign, provided m>t*M; M bounds ell_A+ell_B.',
        'scope':'Conditional on the inherited local variational inequalities; no finite-gamma claim for perturbed gates is inferred from these coefficient bounds.',
        'elapsed_seconds':time.perf_counter()-start,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'FAMILY_RESPONSE_CERTIFICATES.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2),flush=True)
