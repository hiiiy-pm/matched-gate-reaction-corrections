"""Generate a gap-free closed cover; only ALL passing slabs complete the run."""
import probe_imports as p
import json,hashlib,time,contextlib,io,sys
from fractions import Fraction as F
import continuous_engine as eng

OUT=p.ROOT.parents[1]/'data/continuous_coefficient';OUT.mkdir(exist_ok=True)
sources={str(x.relative_to(p.ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()
         for x in p.ROOT.rglob('*.py') if '__pycache__' not in x.parts}
rows=[];attempts=[];start=time.perf_counter()
def name(a,b):return f'{a.numerator}_{a.denominator}__{b.numerator}_{b.denominator}'
def save(status):
    out={'status':status,'pair':'II / C18_0189','masks':p.MASKS,'alpha_requested':['19/10','2'],
         'endpoint_2_included':True,'precision_bits':250,'slabs':rows,'attempts':attempts,
         'source_sha256':sources,'seconds':time.perf_counter()-start,
         'scope':'Local coefficient at theta=0. No extension of finite-rate or reset certificates.'}
    (OUT/'continuous_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
def run(a,b,depth=0):
    key=name(a,b);coords=OUT/(key+'_coordinates.json')
    with contextlib.redirect_stdout(io.StringIO()):cfg=eng.generate_coordinates((a+b)/2)
    coords.write_text(json.dumps(cfg,indent=2)+'\n')
    result=eng.certify_slab(a,b,cfg,bits=250,maxpanels=4096)
    result['coordinate_file']=coords.name
    result['coordinate_sha256']=hashlib.sha256(coords.read_bytes()).hexdigest()
    dest=OUT/(key+'_certificate.json');dest.write_text(json.dumps(result,indent=2)+'\n')
    attempts.append({'alpha_lo':str(a),'alpha_hi':str(b),'positive':result['positive'],'file':dest.name})
    if result['positive']:
        rows.append({'alpha_lo':str(a),'alpha_hi':str(b),'lower_bound':result['lower_bound'],
                     'panels':result['panels'],'file':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
        print(json.dumps({'accepted':len(rows),'alpha':[str(a),str(b)],'lower':result['lower_bound'],
                          'panels':result['panels'],'seconds':time.perf_counter()-start}),flush=True)
        save('IN_PROGRESS_NOT_FULL_RANGE')
    else:
        print(json.dumps({'split':[str(a),str(b)],'reason':result.get('reason','residual_too_wide')}),flush=True)
        if depth>=8:save('INCOMPLETE');raise RuntimeError('unresolved interval')
        mid=(a+b)/2;run(a,mid,depth+1);run(mid,b,depth+1)

if __name__=='__main__':
    save('IN_PROGRESS_NOT_FULL_RANGE')
    for k in range(20):run(F(19,10)+F(k,200),F(19,10)+F(k+1,200))
    assert rows[0]['alpha_lo']=='19/10' and rows[-1]['alpha_hi']=='2'
    assert all(F(x['alpha_hi'])==F(y['alpha_lo']) for x,y in zip(rows,rows[1:]))
    assert all(x['lower_bound']>0 for x in rows)
    for path,expected in sources.items():assert hashlib.sha256((p.ROOT/path).read_bytes()).hexdigest()==expected
    save('PASS_COMPLETE_CONTINUOUS_RANGE')
    print(json.dumps({'status':'PASS_COMPLETE_CONTINUOUS_RANGE','slabs':len(rows),
                      'uniform_lower':min(x['lower_bound'] for x in rows),'seconds':time.perf_counter()-start}),flush=True)
