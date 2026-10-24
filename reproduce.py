"""Run the paper's core exact and continuous-interval calculations."""
from pathlib import Path
import argparse,subprocess,sys
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--figures',action='store_true',help='redraw Figs. 1–4, S4, S7 from supplied data')
ap.add_argument('--segments',action='store_true',help='also reevaluate all 1558 original near-Brownian cells; expensive')
a=ap.parse_args();(R/'results').mkdir(exist_ok=True)
jobs=['code/duration_identities.py','code/exact_gate_models.py','code/spatial_chain_sums.py','code/brownian_response.py','code/replay_coefficient.py']
if a.segments:jobs.append('code/near_brownian_certificate.py')
if a.figures:jobs+=['code/figures/main_figures.py','code/figures/extended_figures.py','code/figures/figure_family.py']
for p in jobs:
    print('RUN',p,flush=True)
    subprocess.run([sys.executable,'-B','-X','utf8',str(R/p)],cwd=R,check=True)
