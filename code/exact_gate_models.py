from pathlib import Path
import json,sys,time,itertools
import sympy as s
import numpy as np
R=Path(__file__).resolve().parents[1];start=time.perf_counter()
ws=json.loads((R/'data/witnesses.json').read_text());records={}
def Qcycle(n):
    Q=s.zeros(n)
    for i in range(n):Q[i,i]=-1;Q[i,(i-1)%n]=Q[i,(i+1)%n]=s.Rational(1,2)
    return Q
def J(Q,mask,z,color=1):
    n=Q.rows;S=[sorted(mask),sorted(set(range(n))-set(mask))]
    if not color:S.reverse()
    A=[Q.extract(x,x) for x in S];B=[Q.extract(S[i],S[1-i]) for i in (0,1)]
    one=s.ones(len(S[0]),1);v=s.ones(len(S[1]),1).T*B[1];v/=sum(v)
    for j,x in enumerate(z):v=v*(x*s.eye(len(S[j%2]))-A[j%2]).inv()*B[j%2]
    return (v*s.ones(len(S[len(z)%2]),1))[0]
for wid in ['C18_0064','C18_0189','C20_0201','C18_0220']:
    w=next(x for x in ws if x['id']==wid);n=w['n'];Q=Qcycle(n);L=np.array(2*Q,dtype=object);mom=[]
    for mask in [w['A'],w['B']]:
        g=np.array([int(i in mask) for i in range(n)],dtype=object);v=g.copy();V=[]
        for k in range(n):V.append(v);v=L@v
        V=np.array(V,dtype=object);mom.append((V@g,(V*g)@V.T))
    assert all(np.array_equal(mom[0][j],mom[1][j]) for j in (0,1))
    assert len(w['A'])==len(w['B'])==n/2
    rec={'clock_C2_C3_all_moments':True,'open_fraction':'1/2'}
    if wid in ['C18_0064','C20_0201']:
        delta=J(Q,w['A'],[1,1])-J(Q,w['B'],[1,1]);target={'C18_0064':'2/5445','C20_0201':'13/6534'}[wid]
        assert delta==s.Rational(target);rec['adjacent_gap']=str(delta)
    elif wid=='C18_0189':
        rec['reset_checks']=[]
        for theta in [s.Rational(1,10),s.Rational(2),s.Rational(10)]:
            Qt=(Q+theta*(s.ones(n,n)/n-s.eye(n)))/(1+s.Rational(17,18)*theta)
            for color in (0,1):
                for z in [[1],[1,2],[1,2,3]]:assert J(Qt,w['A'],z,color)==J(Qt,w['B'],z,color)
                gap=J(Qt,w['A'],[1]*4,color)-J(Qt,w['B'],[1]*4,color);assert gap>0
                rec['reset_checks'].append({'theta':str(theta),'entrance':color,'k1_k2_k3_distinct_arguments':True,'D4_unit':str(gap)})
    else:
        for color in (0,1):
            seq=[]
            for mask in [w['A'],w['B']]:
                S=[sorted(mask),sorted(set(range(n))-set(mask))]
                if not color:S.reverse()
                A=[(2*Q).extract(x,x) for x in S];B=[(2*Q).extract(S[i],S[1-i]) for i in (0,1)];o=s.ones(9,1)
                for a in A:assert (a+2*s.eye(9))*((a+2*s.eye(9))**2-s.eye(9))==s.zeros(9)
                v=o.T*B[1];flux=sum(v);v/=flux;left=[v];out=[]
                for k in range(1,6):
                    left=[v*A[0]**j for j in range(3)] if k==1 else [v*B[(k-2)%2]*A[(k-1)%2]**j for v in left for j in range(3)]
                    out.append(tuple((v*B[(k-1)%2]*o)[0] for v in left))
                seq.append(out)
            assert seq[0]==seq[1]
        gap=J(Q,w['A'],[1,2,3,4,5,6])-J(Q,w['B'],[1,2,3,4,5,6]);assert gap==-s.Rational(79,318745627200)
        rec.update({'complete_duration_k1_to5_both_colors':True,'D6_distinct':str(gap)})
    records[wid]=rec;print(wid,'PASS',flush=True)
# Bind the table strings to the actual exact masks.
