"""Replay every frozen whole-parameter cell without numerical optimization.

The caller may copy this package anywhere. Stored decimals are rational input
coordinates; only enclosing arithmetic, not a center-value test, accepts a cell.
"""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,time,argparse
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'code/coefficient'))
import probe_imports as p
import continuous_engine as eng

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def replay(bits):
    folder=R/'data/continuous_coefficient'
    master=json.loads((folder/'continuous_certificate.json').read_text())
    assert master['status']=='PASS_COMPLETE_CONTINUOUS_RANGE'
    assert master['pair']=='II / C18_0189' and master['masks']==p.MASKS
    for file,h in master['source_sha256'].items():
        assert digest(R/'code/coefficient'/file.replace('\\','/'))==h,file
    rows=master['slabs'];assert len(rows)==20
    for k,row in enumerate(rows):
        assert F(row['alpha_lo'])==F(19,10)+F(k,200)
        assert F(row['alpha_hi'])==F(19,10)+F(k+1,200)
    results=[];start=time.perf_counter()
    for k,row in enumerate(rows):
        f=folder/row['file'];assert digest(f)==row['sha256']
        original=json.loads(f.read_text())
        assert original['alpha_lo']==row['alpha_lo'] and original['alpha_hi']==row['alpha_hi']
        cf=folder/original['coordinate_file'];assert digest(cf)==original['coordinate_sha256']
        cfg=json.loads(cf.read_text())
        assert F(cfg['anchor_alpha_exact'])==(F(row['alpha_lo'])+F(row['alpha_hi']))/2
        assert len(cfg['A_coordinates_exact_decimal'])==len(cfg['trial_coordinates_exact_decimal'])==442
        assert cfg['force_reference_open_component'] in p.MASKS['B']
        assert len(cfg['force_labels'])==len(cfg['force_coordinates_exact_decimal'])
        for lab,c in zip(cfg['force_labels'],cfg['force_coordinates_exact_decimal']):
            assert lab['component'] in p.MASKS['B'] and F(lab['rate_exact_decimal'])>0
            assert lab['kind'] in ('component_difference','zero_integral');F(c)
        for field in ['A_coordinates_exact_decimal','trial_coordinates_exact_decimal','trial_rates_exact_decimal']:
            for v in cfg[field]:F(v)
        result=eng.certify_slab(row['alpha_lo'],row['alpha_hi'],cfg,bits=bits,grid=original['frequency_grid'])
        assert result['positive'] and F(str(result['lower_bound']))>F('0.0000272')
        result['slab_index']=k;results.append(result)
        print(json.dumps({'bits':bits,'cell':k,'alpha':[row['alpha_lo'],row['alpha_hi']],
                          'lower':result['lower_bound'],'rectangles':result['panels']}),flush=True)
    out={'status':'PASS_COMPLETE_FROZEN_REPLAY','precision_bits':bits,'closed_alpha_cover':['19/10','2'],
         'uniform_lower':min(x['lower_bound'] for x in results),'slabs':results,
         'seconds':time.perf_counter()-start,'source_hashes_verified':True,
         'scope':'Original Pair II local coefficient, theta=0. No finite-rate or reset extension.'}
    target=R/'results';target.mkdir(exist_ok=True)
    (target/f'FROZEN_REPLAY_{bits}.json').write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bits',nargs='+',type=int,default=[250,384])
    args=parser.parse_args()
    for bits in args.bits:
        assert bits>=250
        out=replay(bits);print('PASS',bits,out['uniform_lower'],flush=True)
