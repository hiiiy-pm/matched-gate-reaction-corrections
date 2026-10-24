"""Recombine the frozen continuous-space energy and defect bounds at both rates.

This evaluates the posterior inequalities; it does not rerun the Fourier sums.
The full Fourier solvers and their exact trial inputs are in code/models.
"""
from pathlib import Path
import json,hashlib
from flint import arb,ctx
R=Path(__file__).resolve().parents[1];ctx.prec=256
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def number(x):return arb(x).union(arb(str(x))) if isinstance(x,float) else arb(str(x))
def hull(x):return number(x[0]).union(number(x[1]))
def bounds(x):return [str(x.lower()),str(x.upper())]
first={x['pair']:x for x in read(R/'data/figures/certificates/current_rate_certificate.json')['rows']}
rows=[]
for rate,base in [(2560000,'finite_rate/current_rate_certification'),(1280000,'extended_models/second_rate')]:
    b=R/'code/models'/base
    trialdir=(R/'code/models/continuous_space/rigorous_certification/continuum') if rate==2560000 else b/'continuum'
    for pair,N,K in [('C18_0064',65536,524288),('C18_0189',262144,1048576),('C20_0201',65536,524288)]:
        values={};errors={}
        trial=trialdir/f'{pair}_N{N}_enriched_trial.json';tr=read(trial)
        assert int(tr['gamma'])==rate and tr['N']==N and tr['width']=='1/32'
        for color in ['A','B']:
            en=read(b/f'solver/{pair}_N{N}_K1048576_{color}_energy_safe_square.json')
            de=read(b/f'solver/{pair}_N{N}_K{K}_{color}_derivative_certificate.json')
            assert en['witness']==de['witness']==pair and en['label']==de['label']==color
            assert int(float(en['gamma']))==int(float(de['gamma']))==rate
            assert en['N']==de['source_N']==N and en['alpha']==de['alpha']==first[pair]['alpha_exact_decimal']
            assert sha(trial)==en['trial_sha256']==de['driver_provenance']['input_trial_sha256']
            formal=trialdir/f'{pair}_N{N}_K{K}_{color}_enriched_enclosure.json'
            assert sha(formal)==de['driver_provenance']['formal_potential_sha256']
            values[color]=hull(en['exact_continuous_trial_green_energy'])
            errors[color]=number(de['green_energy_defect_upper'][1]).upper()
            assert errors[color].is_finite() and errors[color]>=0
        n=tr['witness']['n']
        lower=(n*(values['A'].lower()-values['B'].upper()-errors['A'])).lower()
        upper=(n*(values['A'].upper()-values['B'].lower()+errors['B'])).upper()
        delta=lower.union(upper);alpha=arb(first[pair]['alpha_exact_decimal'])
        eps=arb(rate)**(-1/alpha)/(2*arb.pi())
        fc=first[pair]['true_leading_amplitude_interval'];C=arb(fc['lower_decimal']).union(arb(fc['upper_decimal']))
        error=(100*abs(delta/eps/C-1)).upper()
        assert not delta.contains(0) and not C.contains(0) and error<1
        rows.append({'pair':pair,'gamma':rate,'alpha':str(alpha),'delta_tau':bounds(delta),'relative_error_percent_upper':str(error)})
        print(pair,rate,'PASS',str(error))
(R/'results').mkdir(exist_ok=True)
(R/'results/finite_rate_recombination.json').write_text(json.dumps({'scope':'Recombination of frozen continuous-space inputs; not a fresh Fourier solve.','rows':rows},indent=2))
