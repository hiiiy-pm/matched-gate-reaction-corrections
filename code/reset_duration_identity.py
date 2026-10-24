"""Exact polynomial certificate for complete duration matching under reset."""
from pathlib import Path
import json,itertools,time,hashlib
import numpy as np
import sympy as s
R=Path(__file__).resolve().parents[1];OUT=R/'results'
W=json.loads((R/'code/models/reproducible/witnesses.json').read_text())
Q=s.zeros(18)
for i in range(18):
    Q[i,i]=-1;Q[i,(i-1)%18]=Q[i,(i+1)%18]=s.Rational(1,2)
def blocks(mask,L):
    sets=[mask,[i for i in range(18) if i not in mask]]
    a=[np.array(L.extract(z,z),dtype=object) for z in sets]
    b=[np.array(L.extract(sets[c],sets[1-c]),dtype=object) for c in [0,1]]
    return a,b
def triple_tensor(mask,L,c):
    A,B=blocks(mask,L);d=9
    row=np.ones(9,dtype=object)@B[1-c]
    flux=sum(row);left=[]
    for j in range(d):left.append(row.copy());row=row@A[c]
    v=B[c]@np.ones(9,dtype=object);right=[]
    for j in range(d):right.append(v.copy());v=A[c]@v
    left=np.array(left,dtype=object)@B[c]
    right=B[1-c]@np.array(right,dtype=object).T
    result=[]
    for j in range(d):
        result.extend(int(x) for x in (left@right).ravel())
        left=left@A[1-c]
    return result,int(flux)
def full_base_words(mask,c,k,degree):
    L=2*Q;A,B=blocks(mask,L);powers=[]
    for mat in A:
        p=[np.eye(9,dtype=object)]
        for j in range(1,degree):p.append(p[-1]@mat)
        powers.append(p)
    row=np.ones(9,dtype=object)@B[1-c];flux=sum(row)
    rows=[row]
    for j in range(k):
        state=(c+j)%2
        rows=[v@p@B[state] for v in rows for p in powers[state]]
    return [int(sum(v)) for v in rows],int(flux)
def laplace(mask,z,theta=s.Integer(0),c=0):
    d=1+s.Rational(17,18)*theta
    q=(Q+theta*(s.ones(18)/18-s.eye(18)))/d
    sets=[mask,[i for i in range(18) if i not in mask]]
    A=[q.extract(z,z) for z in sets]
    B=[q.extract(sets[j],sets[1-j]) for j in [0,1]]
    row=s.ones(1,9)*B[1-c];row/=sum(row)
    for j,t in enumerate(z):
        state=(c+j)%2
        row=row*(t*s.eye(9)-A[state]).inv()*B[state]
    return s.factor((row*s.ones(9,1))[0])
t0=time.perf_counter();polynomial=[]
for wid in ['C18_0189','C18_0220']:
    w=next(x for x in W if x['id']==wid)
    for theta in range(29):
        L=18*Q+theta*(s.ones(18)-18*s.eye(18))
        for c in [0,1]:
            a,fa=triple_tensor(w['A'],L,c);b,fb=triple_tensor(w['B'],L,c)
            assert fa==fb and a==b,(wid,theta,c)
        print(wid,'theta',theta,'729 derivatives per color exact',flush=True)
    # Equal characteristic polynomials follow by permuting identical killed
    # run blocks; the permutation fixes the all-one vector and hence reset.
    polynomial.append({'seed':wid,'nodes':list(range(29)),
        'orders_per_duration':list(range(9)),'starting_colors':2,
        'polynomial_degree_bound':28,'all_exact_equal':True,
        'common_killed_characteristic_polynomial_reason':'Same path-run inventory; similarity by a permutation fixing ones also preserves the reset rank-one term.',
        'implication':'Polynomial identity plus Cayley-Hamilton gives complete three-duration laws for every theta>=0, not only the sampled transforms.'})
extra=next(w for w in W if w['id']=='C18_0220')
minimal=s.Symbol('x')*(s.Symbol('x')**2-1) # shifted to L=2Q: (L+2I)((L+2I)^2-I)
for label in ['A','B']:
    A,B=blocks(extra[label],2*Q)
    for a in A:
        shifted=a+2*np.eye(9,dtype=object)
        assert not any(int(x) for x in (shifted@(shifted@shifted-np.eye(9,dtype=object))).ravel())
five=[]
for k in range(1,7):
    equal=True
    for c in [0,1]:
        x,fx=full_base_words(extra['A'],c,k,3);y,fy=full_base_words(extra['B'],c,k,3)
        assert fx==fy
        equal &= x==y
    five.append({'durations':k,'derivative_orders':list(range(3)),'both_colors_equal':equal})
    assert equal == (k<=5)
z=s.symbols('z',positive=True)
w=next(w for w in W if w['id']=='C18_0189')
J1=laplace(w['A'],[z])
delta=s.factor(laplace(w['A'],[z]*4)-laplace(w['B'],[z]*4))
expected=z**2*(4*z+7)/(40*(z+1)**2*(2*z+1)**2*(2*z+3)**2*(2*z**2+4*z+1)**2)
assert s.cancel(delta-expected)==0
lambda0=s.Rational(5,18);p=s.Rational(1,2)
H=s.factor((lambda0*J1+z*p-lambda0)/z**2)
checks=[]
for theta in [s.Rational(0),s.Rational(1,1000),s.Rational(1,100),s.Rational(1,10),s.Integer(1),s.Integer(10)]:
    a=1/(1+s.Rational(17,18)*theta);b=theta*a
    zp=(1+b)/a;lam=a*lambda0+b/4;h=H.subs(z,zp)/a
    predicted=s.factor(lambda0*a/(lam*(1+b)**2*(1-b*h)**2)*delta.subs(z,zp))
    direct=laplace(w['A'],[1]*4,theta)-laplace(w['B'],[1]*4,theta)
    assert s.factor(predicted-direct)==0 and direct>0
    checks.append({'theta':str(theta),'four_duration_gap_at_all_one':str(direct),
                   'reset_transfer_identity_exact':True})
record={'complete':True,'polynomial_certificates':polynomial,
        'second_seed_full_duration_hierarchy':five,
        'second_seed_six_duration_probe':[str(laplace(extra[c],list(range(1,7)))) for c in ['A','B']],
        'second_seed_six_duration_gap':str(laplace(extra['A'],list(range(1,7)))-laplace(extra['B'],list(range(1,7)))),
        'pair_ii_single_duration_transform':str(J1),
        'pair_ii_censored_single_transform_H':str(H),'pair_ii_four_duration_diagonal_gap':str(delta),
        'four_duration_reset_checks':checks,'elapsed_seconds':time.perf_counter()-t0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'RESET_FAMILY_EXACT.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('COMPLETE',record['elapsed_seconds'],flush=True)
