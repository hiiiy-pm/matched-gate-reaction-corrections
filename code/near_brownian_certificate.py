from pathlib import Path
from decimal import Decimal
import sys,json,time
import interval_engine as iv
import interval_engine_arb as ab
import mpmath as mp
R=Path(__file__).resolve().parents[1];start=time.perf_counter()
ws=json.loads((R/'data/witnesses.json').read_text());results=[]
out=R/'results/segments';out.mkdir(exist_ok=True)
for wid,lo,count,sign in [('C18_0064','1.99651',698,-1),('C18_0189','1.9982',360,1),('C20_0201','1.995055',500,1)]:
    w=next(z for z in ws if z['id']==wid);previous=[]
    for precision in [70,95,250]:
        backend=ab if precision==250 else iv
        if precision==250:A=ab.Model(w['A'],w['n'],prec=250);B=ab.Model(w['B'],w['n'],prec=250)
        else:
            mp.mp.dps=precision+20;mp.iv.dps=precision
            A=iv.Model(w['A'],w['n'],precision);B=iv.Model(w['B'],w['n'],precision)
        lower=Decimal(lo);step=(Decimal(2)-lower)/count;rows=[]
        for j in range(count):
            z=backend.certify(lower+j*step,lower+(j+1)*step,A,B);assert z['gap'][1]<0 if sign<0 else z['gap'][0]>0
            for old in previous:assert max(old[j]['gap'][0],z['gap'][0])<=min(old[j]['gap'][1],z['gap'][1])
            rows.append(z)
        previous.append(rows)
        (out/f'{wid}_{precision}.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        print(wid,precision,len(rows),'PASS',flush=True)
    results.append({'pair':wid,'segments':count,'backends':['mpmath.iv 70 dps','mpmath.iv 95 dps','Arb 250 bits'],'all_overlap':True,'all_signs_certified':True})
(R/'results/SEGMENT_RERUN.json').write_text(json.dumps({'all_pass':True,'unique_segments':1558,'evaluations':4674,'rows':results,'seconds':time.perf_counter()-start},indent=2),encoding='utf8')
