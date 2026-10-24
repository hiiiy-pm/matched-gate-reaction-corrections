"""Full 384-bit replay, exact cover/provenance audit and independent objectives."""
import probe_imports as p
import json,time,hashlib
from fractions import Fraction as F
import mpmath as mp
from flint import arb,arb_series
import continuous_engine as eng
import interval_engine_arb as ab
import arb_dual_check as point

DIR=p.ROOT.parents[1]/'data/continuous_coefficient';root=json.loads((DIR/'continuous_certificate.json').read_text())
assert root['status']=='PASS_COMPLETE_CONTINUOUS_RANGE'
for file,digest in root['source_sha256'].items():assert hashlib.sha256((p.ROOT/file).read_bytes()).hexdigest()==digest
rows=root['slabs'];assert F(rows[0]['alpha_lo'])==F('1.9') and F(rows[-1]['alpha_hi'])==2
assert all(F(x['alpha_hi'])==F(y['alpha_lo']) for x,y in zip(rows,rows[1:]))
output=[];start=time.perf_counter()
for index,row in enumerate(rows):
    file=DIR/row['file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
    original=json.loads(file.read_text());coords=DIR/original['coordinate_file']
    assert hashlib.sha256(coords.read_bytes()).hexdigest()==original['coordinate_sha256']
    cfg=json.loads(coords.read_text())
    # Exact symbolic support and zero total mass of the added force.
    assert cfg['force_reference_open_component'] in p.MASKS['B']
    for lab,c in zip(cfg['force_labels'],cfg['force_coordinates_exact_decimal']):
        assert lab['component'] in p.MASKS['B'];assert F(lab['rate_exact_decimal'])>0;F(c)
        assert lab['kind'] in ('component_difference','zero_integral')
    assert len(cfg['A_coordinates_exact_decimal'])==len(cfg['trial_coordinates_exact_decimal'])==442
    replay=eng.certify_slab(row['alpha_lo'],row['alpha_hi'],cfg,bits=384,grid=original['frequency_grid'])
    assert replay['positive'] and replay['lower_bound']>2.72e-5
    api,models,operator=eng.setup(384);joint=eng.JointObjective(models,cfg)
    s=api.mpf((F(row['alpha_lo'])+F(row['alpha_hi']))/4)
    values=[]
    for key in ('A','B'):
        cs=cfg['A_coordinates_exact_decimal'] if key=='A' else cfg['trial_coordinates_exact_decimal']
        l,r,c=eng.cert.old.physical_trial(models[key],cs,cfg['trial_rates_exact_decimal'])
        # Continuity and vanishing open-side components follow algebraically;
        # evaluate their exact-basis interval expressions as a coding check.
        assert all(v[j].is_zero() for v,a in l for j in p.MASKS[key])
        for j in range(18):
            z=ab.isum(v[j] for v,a in l)-ab.isum(v[j] for v,a in r)-c
            assert z.contains(0)
        values.append(point.objective(models[key],l,r,s))
    independent=values[0]-values[1];combined=joint(s)
    assert independent.overlaps(combined)
    # A second numerical differentiation implementation checks the AD formula
    # at a point; it is an error detector, not a whole-slab proof.
    mp.mp.dps=90
    def mpval(x):
        q=(ab.endpoint_fraction(x.lower())+ab.endpoint_fraction(x.upper()))/2
        return mp.mpf(q.numerator)/q.denominator
    same=[tuple(mpval(x) for x in term) for term in joint.same]
    cross=[(*[mpval(x) for x in term[:3]],term[3]) for term in joint.cross]
    source=[tuple(mpval(x) for x in term) for term in joint.source]
    gate=mpval(joint.gate)
    def f(v):
        rho=2*v-1;den=2*mp.sin(mp.pi*rho/2)
        kin=sum(c*(a**(rho-1)+b**(rho-1))/(den*(a+b)) for c,a,b in same)
        for c,a,b,equal in cross:
            ker=(1-rho)*a**(rho-2)/den if equal else -a**(rho-1)*mp.expm1((rho-1)*mp.log(b/a))/(den*(b-a))
            kin+=c*ker
        src=mp.gamma(1+v)*sum(c*a**(v-1) for c,a in source)
        return (2*src-kin-gate)/(18*mp.gamma(1+v)**2)
    numerical=mp.diff(f,mpval(s));automatic=joint(arb_series([s,arb(1)],2))[1]
    delta=abs(numerical-mpval(automatic));assert delta<mp.mpf('1e-65')
    info={'slab':index,'alpha':[row['alpha_lo'],row['alpha_hi']],
          'lower_bound_384':replay['lower_bound'],'lower_bound_250':original['lower_bound'],
          'panels':replay['panels'],'separate_objective_matches':True,
          'derivative_point_difference':mp.nstr(delta,12),'source_constraints':True,
          'trial_constraints':True}
    output.append(info)
    (DIR/(row['file'].replace('_certificate','_replay384'))).write_text(json.dumps(replay,indent=2)+'\n')
    (DIR/'precision_384_summary.json').write_text(json.dumps({'status':'IN_PROGRESS','rows':output},indent=2)+'\n')
    print(json.dumps(info),flush=True)
out={'status':'PASS_FULL_384_BIT_REPLAY','slabs':len(output),'rows':output,
     'uniform_lower':min(z['lower_bound_384'] for z in output),'seconds':time.perf_counter()-start,
     'exact_alpha_cover':True,'arithmetic_backend':'Arb, same analytic formulas at 384 bits',
     'mpmath_derivatives':'Pointwise numerical error-detection checks, not interval certificates',
     'source_sha256':hashlib.sha256(p.Path(__file__).read_bytes()).hexdigest()}
(DIR/'precision_384_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],flush=True)
