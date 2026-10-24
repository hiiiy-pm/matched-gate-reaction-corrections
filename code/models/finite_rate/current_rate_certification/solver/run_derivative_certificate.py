"""Construct the V9 smooth-potential coefficients and run the V10 certificate.

This is a fresh driver.  It does not edit the frozen V9 source or the already
executed V10 continuous-energy engine.  All charge and Fourier computations
use Arb/Acb; CG trial coordinates are not assumed to be exact solutions.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import flint
from flint import arb,acb,ctx
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[2]
V9=WORK/'continuous_space/rigorous_certification/continuum'
MATH=HERE.parent/'math'
sys.path.insert(0,str(MATH))

def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare_coefficients(data,trial,K,bits,progress=True):
    """p_i(k): cell full G plus atomic (G-G0), positive frequency only.

    The common atomic G0 terms are handled analytically by the certifier.
    These formulas and DFT conventions are exactly those in the formal V9
    enriched_green_enclosure.py, rather than a new spatial discretization.
    """
    started=time.perf_counter();ctx.prec=bits
    N=data['N'];M=data['witness']['n'];R=trial['R'];r=len(R);m=N//32
    alpha=arb(str(data['witness']['alpha']));gamma=arb(str(data['gamma']));pi=arb.pi()
    cell=[arb(float.fromhex(z)) for z in trial['charge_exact_float_hex']]
    atoms=[arb(float.fromhex(z)) for z in trial['atom_exact_float_hex']]
    mass=sum(cell+atoms,arb(0));cell=[z/mass for z in cell];atoms=[z/mass for z in atoms]
    spatial=[]
    for gate in range(M):
        if gate in R:
            col=R.index(gate)
            spatial.append(acb.dft([acb(cell[t*r+col]) for t in range(m)]+[acb(0)]*(N-m)))
        else:spatial.append(None)
    modal=[acb.dft([spatial[j][k] if spatial[j] is not None else acb(0) for j in range(M)]) for k in range(N)]
    del spatial
    am0=acb.dft([acb(atoms[R.index(j)]) if j in R else acb(0) for j in range(M)])
    am1=acb.dft([acb(atoms[r+R.index(j)]) if j in R else acb(0) for j in range(M)])
    rates=[1-(2*pi*j/M).cos() for j in range(M)];rates[0]=arb(0)
    coeffs=[[] for _ in R]
    for k in range(1,K+1):
        power=arb(k)**alpha;angle=pi*k/N
        cellfactor=acb(0,-angle).exp()*angle.sin()/angle
        phase=acb(0,-2*pi*(k%32)/32).exp()
        z=[]
        for ell in range(M):
            ca=am0[ell]+phase*am1[ell]
            rate=gamma*rates[ell]
            z.append(modal[k%N][ell]*cellfactor/(power+rate)-ca*rate/(power*(power+rate)))
        physical=acb.dft(z,inverse=True)
        for j,gate in enumerate(R):coeffs[j].append(physical[gate])
        if progress and k%131072==0:
            print(json.dumps({'stage':'prepare_derivative_coefficients','label':trial['label'],'k':k,
                             'seconds':time.perf_counter()-started}),flush=True)
    del modal
    return coeffs,{'coefficient_generation_seconds':time.perf_counter()-started,
                   'coefficient_count_per_gate':K,'reaction_gates':R}

def main():
    p=argparse.ArgumentParser();p.add_argument('--witness',default='C18_0064');p.add_argument('--n',type=int)
    p.add_argument('--kmax',type=int);p.add_argument('--bits',type=int,default=160)
    p.add_argument('--only',choices=('A','B'));a=p.parse_args()
    N=a.n or (262144 if a.witness=='C18_0189' else 65536)
    K=a.kmax or (1048576 if a.witness=='C18_0189' else 524288)
    trial_path=V9/f'{a.witness}_N{N}_enriched_trial.json'
    data=json.loads(trial_path.read_text(encoding='utf-8'))
    from derivative_energy import certify
    for tr in data['charges']:
        if a.only and tr['label']!=a.only:continue
        started=time.perf_counter()
        driver_hash_before=sha(Path(__file__))
        module_hash_before=sha(MATH/'derivative_energy.py')
        formal=V9/f'{a.witness}_N{N}_K{K}_{tr["label"]}_enriched_enclosure.json'
        formal_data=json.loads(formal.read_text(encoding='utf-8'))
        coeffs,diag=prepare_coefficients(data,tr,K,a.bits)
        rec=certify(data,tr,coeffs,formal_data)
        if sha(Path(__file__))!=driver_hash_before or sha(MATH/'derivative_energy.py')!=module_hash_before:
            raise ArithmeticError('Source changed during execution; this run is not released as a certificate.')
        rec['driver_provenance']={'driver_sha256':driver_hash_before,
             'derivative_module_sha256':module_hash_before,
             'input_trial':str(trial_path),'input_trial_sha256':sha(trial_path),
             'formal_potential_input':str(formal),'formal_potential_sha256':sha(formal),
             'precision_bits':a.bits,'fourier_cutoff':K,'flint_version':flint.__version__,
             'diagnostic':diag,'total_seconds':time.perf_counter()-started}
        out=HERE/f'{a.witness}_N{N}_K{K}_{tr["label"]}_derivative_certificate.json'
        save(out,rec);print(json.dumps({'completed':str(out),'seconds':time.perf_counter()-started}),flush=True)
if __name__=='__main__':main()
