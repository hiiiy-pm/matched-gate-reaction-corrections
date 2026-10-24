"""Exact three-duration contrast after the stated symmetric edge perturbation."""
from pathlib import Path
import json
import sympy as s
R=Path(__file__).resolve().parents[1]
w=next(x for x in json.loads((R/'data/witnesses.json').read_text()) if x['id']=='C18_0189')
n=w['n'];Q=s.zeros(n)
for i in range(n):
    Q[i,i]=-1;Q[i,(i-1)%n]=Q[i,(i+1)%n]=s.Rational(1,2)
eta=s.Rational(1,1000)
for i,j in [(4,5),(5,4)]:Q[i,j]+=eta;Q[i,i]-=eta
assert Q==Q.T and Q*s.ones(n,1)==s.zeros(n,1)
def duration(mask,color):
    parts=[sorted(mask),sorted(set(range(n))-set(mask))]
    if color==0:parts.reverse()
    killed=[Q.extract(x,x) for x in parts]
    jumps=[Q.extract(parts[i],parts[1-i]) for i in range(2)]
    nu=s.ones(len(parts[1]),1).T*jumps[1];nu/=sum(nu)
    for j,z in enumerate([1,2,3]):
        nu=nu*(z*s.eye(len(parts[j%2]))-killed[j%2]).inv()*jumps[j%2]
    return (nu*s.ones(len(parts[1]),1))[0]
rows=[]
for color in [0,1]:
    delta=duration(w['A'],color)-duration(w['B'],color)
    assert delta!=0
    rows.append({'initial_color':color,'J3_A_minus_B':str(delta)})
    print(color,delta)
(R/'results').mkdir(exist_ok=True)
(R/'results/gate_perturbation.json').write_text(json.dumps({'edge':[4,5],'increment':str(eta),'rows':rows},indent=2))
