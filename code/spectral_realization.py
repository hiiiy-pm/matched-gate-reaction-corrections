"""Exact Pair II checks and the data for the revised spectral mechanism figure.
No prior numerical result is used as input. All state-space operations use
rational arithmetic. Individual trigonometric rate allocations are evaluated
at 80 and 120 digits and checked against exact rational totals.
"""
from pathlib import Path
import hashlib,json,argparse
import sympy as s
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,default=ROOT/'results/pair_ii_spectral_allocation.json')
args=ap.parse_args()
args.out.parent.mkdir(parents=True,exist_ok=True)
n=18
masks={'A':[0,1,4,5,6,8,11,12,14],'B':[0,1,4,6,7,10,11,12,14]}
Q=-s.eye(n)
for i in range(n): Q[i,(i-1)%n]=Q[i,(i+1)%n]=s.Rational(1,2)
powers=[s.eye(n)]
for k in range(1,n): powers.append(powers[-1]*Q)
exact={}; report=[]
for label,R in masks.items():
 g=s.Matrix([int(i in R) for i in range(n)]);G=s.diag(*g)
 K=s.Matrix.hstack(*[power*g for power in powers]); piv=K.rref()[1];V=K[:,list(piv)]
 P=V*(V.T*V).inv()*V.T;F=s.eye(n)-P
 # Integer determinant certificate for K + Pi K.
 KB=s.Matrix.hstack(*[((2*Q)**j)*g for j in range(10)])
 HB=KB.row_join(G*KB)
 minor_columns=list(range(10))+list(range(11,19))
 minor=HB[:,minor_columns].det()
 assert minor==-118098
 x=G*Q*g;y=G*powers[5]*g
 leak=F*G*P
 # Closing the visible space under both Q and the observation projector.
 W=V;dims=[W.cols]
 for _ in range(n):
  expanded=s.Matrix.hstack(W,Q*W,G*W)
  W1=expanded[:,list(expanded.rref()[1])]
  dims.append(W1.cols)
  if W1.cols==W.cols: break
  W=W1
 assert P*P==P and P*Q==Q*P and V.cols==10
 assert s.trace(leak.T*leak)==2
 exact[label]={'P':P,'x':x,'y':y,'g':g,
  'total':s.factor((x.T*F*y)[0]/n),
  'weighted_total':s.factor((x.T*F*Q*F*y)[0]/n),
  'visible':s.factor((x.T*P*Q*P*y)[0]/n),
  'direct_C4_derivative':s.factor((g.T*Q*G*Q*G*powers[5]*g)[0]/n)}
 O=s.Matrix([[int(i not in R) for i in range(n)],[int(i in R) for i in range(n)]])
 report.append({'mask':label,'emission_rank':O.rank(),'state_count':n,'visible_dimension':V.cols,
 'closure_dimensions':dims,'leakage_HS_squared':str(s.trace(leak.T*leak)),
 'leakage_rank':leak.rank(),'sampled_transition_determinant':'exp(-18*h)>0 for h>0',
 'trace_Q':str(s.trace(Q)), 'integer_minor_columns_1_based':[j+1 for j in minor_columns], 'integer_minor_determinant':str(minor)})
 assert exact[label]['visible']+exact[label]['weighted_total']==exact[label]['direct_C4_derivative']
assert exact['A']['total']==exact['B']['total']
assert exact['A']['visible']==exact['B']['visible']
assert exact['A']['weighted_total']-exact['B']['weighted_total']==s.Rational(1,2304)
def run(dps):
 mp.mp.dps=dps
 def cv(M): return mp.matrix([[mp.mpf(int(M[i,j].p))/int(M[i,j].q) for j in range(M.cols)] for i in range(M.rows)])
 rows=[]
 for j in range(10):
  lam=-1+mp.cos(2*mp.pi*j/n);mult=1 if j in [0,9] else 2
  Pj=mp.matrix([[mp.mpf(mult)/n*mp.cos(2*mp.pi*j*(a-b)/n) for b in range(n)] for a in range(n)])
  row={'j':j,'multiplicity':mult,'lambda':lam}
  for label,e in exact.items():
   Fj=Pj*(mp.eye(n)-cv(e['P']))
   assert mp.norm(Fj*Fj-Fj)<mp.mpf(10)**(-dps+12)
   row[label]=(cv(e['x']).T*Fj*cv(e['y']))[0]/n
   row[label+'_weighted']=lam*row[label]
  row['delta']=row['A']-row['B'];row['delta_weighted']=lam*row['delta']
  rows.append(row)
 assert abs(sum(r['delta'] for r in rows))<mp.mpf(10)**(-dps+12)
 assert abs(sum(r['delta_weighted'] for r in rows)-mp.mpf(1)/2304)<mp.mpf(10)**(-dps+12)
 return rows
r80,r120=run(80),run(120)
for a,b in zip(r80,r120):
 for k in ['lambda','A','B','delta','delta_weighted']: assert abs(a[k]-b[k])<mp.mpf('1e-65')
rows=[]
for r in r120:
 row={k:(int(v) if k in ['j','multiplicity'] else mp.nstr(v,85)) for k,v in r.items()}
 if r['j'] in [0,9]:
  for k in ['A','B','A_weighted','B_weighted','delta','delta_weighted']:row[k]='0'
 if r['j'] in [3,6]:
  j=r['j'];Pj=s.Matrix(n,n,lambda a,b:s.Rational(2,n)*s.cos(2*s.pi*j*(a-b)/n))
  ev=-1+s.cos(2*s.pi*j/n)
  de={label:s.factor((e['x'].T*Pj*(s.eye(n)-e['P'])*e['y'])[0]/n) for label,e in exact.items()}
  dd=s.factor(de['A']-de['B'])
  row['exact_delta']=str(dd);row['exact_delta_weighted']=str(ev*dd)
  if dd==0:row['delta']=row['delta_weighted']='0'
 rows.append(row)
out={'pair':'C18_0189','masks':masks,'orders':[1,1,5],
 'definition':'d_R(mu)=<Pi Qg,F_mu Pi Q^5g>_n; U_R=sum_mu mu*d_R(mu)',
 'exact_totals':{label:{k:str(e[k]) for k in ['total','weighted_total','visible','direct_C4_derivative']} for label,e in exact.items()},
 'exact_delta_total':'0','exact_delta_weighted_total':'1/2304',
 'precisions':[80,120],'maximum_allowed_cross_precision_difference':'1e-65',
 'rows':rows,'realization_checks':report,'all_pass':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
args.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:out[k] for k in ['exact_totals','realization_checks']},indent=2))
print('delta marks in units 1/2304',[[r['j'],float(mp.mpf(r['delta'])*2304),float(mp.mpf(r['delta_weighted'])*2304)] for r in rows])

