"""Exact duration-preserving controls and certified Brownian coefficients.

No experimental values enter these model calculations. Source definitions are
the explicit Pair II masks. Matrix square roots are enclosed using monotone
Newton iteration and a uniform scalar error bound, without numerical eigenvectors.
"""
from pathlib import Path
import json,time,hashlib
import sympy as sp
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';OUT.mkdir(parents=True,exist_ok=True)
MASKS={'A':[0,1,4,5,6,8,11,12,14],'B':[0,1,4,6,7,10,11,12,14]}

def generators(mask):
    n=18;Q=sp.zeros(n)
    for i in range(n):Q[i,i]=-1;Q[i,(i-1)%n]=Q[i,(i+1)%n]=sp.Rational(1,2)
    C=[i for i in range(n) if i not in mask]
    e=-Q.extract(mask,mask)*sp.ones(len(mask),1);f=-Q.extract(C,C)*sp.ones(len(C),1)
    assert sum(e)==sum(f)==5
    Qren=Q.copy()
    for i,x in enumerate(mask):
        for j,y in enumerate(C):Qren[x,y]=Qren[y,x]=e[i]*f[j]/5
    assert Qren==Qren.T and Qren*sp.ones(n,1)==sp.zeros(n,1)
    assert all(Qren[i,i]==-1 for i in range(n))
    return Q,Qren,C

def A(x):
    x=sp.Rational(x);return arb(int(x.p))/arb(int(x.q))

def AM(M):return arb_mat([[A(x) for x in row] for row in M.tolist()])

def identity(n):return arb_mat([[arb(int(i==j)) for j in range(n)] for i in range(n)])

def newton_root(M,minimum,iterations=12):
    # For eigenvalues in [a,2], scalar Newton starting at 2 lies above sqrt.
    # Error <= 2 sqrt(2) q^(2^k)/(1-q^(2^k)), q=(2-sqrt(a))/(2+sqrt(a)).
    S=identity(M.nrows())*2
    for _ in range(iterations):S=(S+M*S.inv())/2
    a=A(minimum);q=(2-a.sqrt())/(2+a.sqrt());t=q**(2**iterations)
    err=2*arb(2).sqrt()*t/(1-t)
    assert err>0
    return S,err

def coeff(Q,mask,prec):
    ctx.prec=prec;n=Q.rows;C=[i for i in range(n) if i not in mask]
    H=AM(-Q);P0=arb_mat([[arb(1)/n]*n for _ in range(n)])
    # Q_eta >= (1/10) Q_cycle in Dirichlet form, lambda_2 >= 1/405.
    S,err1=newton_root(H+P0,sp.Rational(1,405))
    S=S-P0
    HC=AM((-Q).extract(C,C));SC,err2=newton_root(HC,sp.Rational(1,4))
    G=arb_mat([[S[i,j] for j in C] for i in C])+SC
    eps=err1+err2;one=arb_mat([[arb(1)] for _ in C])
    lower=(one.transpose()*G.inv()*one)[0,0]/n
    upper=(one.transpose()*(G-identity(len(C))*eps).inv()*one)[0,0]/n
    # Both individual ball calculations enclose their exact rational-iterate
    # quadratic forms. Their union encloses the true coefficient by Loewner order.
    result=lower.union(upper)
    assert result>0
    return result,{'enclosure':str(result),'lower':str(result.lower()),'upper':str(result.upper()),'midpoint':float(result.mid()),'radius':float(result.rad()),'root_error_bound':str(eps),'precision_bits':prec}

def J(Q,mask,args):
    R=sorted(mask);C=[i for i in range(Q.rows) if i not in R];sets=[R,C]
    e=-Q.extract(R,R)*sp.ones(len(R),1);v=e.T/sum(e)
    for k,z in enumerate(args):
        I=sets[k%2];K=sets[1-k%2];Rz=(z*sp.eye(len(I))-Q.extract(I,I)).inv()
        v=v*Rz
        if k==len(args)-1:v=v*(-Q.extract(I,I)*sp.ones(len(I),1))
        else:v=v*Q.extract(I,K)
    return sp.factor(v[0])

def exact_checks():
    z=sp.Symbol('z');ref=(14*z**3+37*z**2+26*z+5)/(5*(z+1)*(2*z+1)*(2*z*z+4*z+1))
    rows=[];eta=sp.Symbol('eta')
    for name,mask in MASKS.items():
        Q,N,C=generators(mask)
        assert sp.factor(J(Q,mask,[z])-ref)==0 and sp.factor(J(N,mask,[z])-ref)==0
        # Endpoint renewal factorization for distinct Laplace arguments.
        for args in [[1,2],[1,2,3],[1,2,3,4]]:
            assert J(N,mask,args)==sp.prod(ref.subs(z,a) for a in args)
        M=(1-eta)*Q+eta*N
        rows.append(J(M,mask,[1,1,1,1]))
    contrast=sp.factor(rows[0]-rows[1]);assert sp.factor(contrast-sp.Rational(11,1764000)*(1-eta)**3)==0
    moments=[]
    for name,mask in MASKS.items():
        Q,N,C=generators(mask);M=(Q+N)/2;g=sp.Matrix([int(i in mask) for i in range(18)]);v=g;mm=[]
        for k in range(9):mm.append((g.T*v)[0]/18);v=M*v
        moments.append(mm)
    mismatch=[{'order':k,'difference':str(a-b)} for k,(a,b) in enumerate(zip(*moments)) if a!=b]
    assert mismatch==[{'order':8,'difference':'-1/18432'}]
    return {'common_single_duration_transform':str(ref),'four_duration_gap_all_eta':str(contrast),'clock_time_correlation_at_eta_half_first_mismatch':mismatch,'note':'The interpolated A and B generators differ. This is a duration-preserving control, not the same-generator/full-clock-summary theorem.'}

def main():
    started=time.time();proof=exact_checks();rows=[]
    for eta in [sp.Rational(0),sp.Rational(1,4),sp.Rational(1,2),sp.Rational(3,4),sp.Rational(1)]:
        perprec=[]; saved_balls=[]
        for prec in [160,256]:
            rr={};balls={}
            for name,mask in MASKS.items():
                Q,N,C=generators(mask);v,desc=coeff((1-eta)*Q+eta*N,mask,prec);rr[name]=desc;balls[name]=v
            gap=balls['A']-balls['B'];rr['gap']={'enclosure':str(gap),'lower':str(gap.lower()),'upper':str(gap.upper()),'midpoint':float(gap.mid()),'radius':float(gap.rad())}
            assert (gap>0 if eta in (0,sp.Rational(1,4)) else gap.contains(0) if eta==1 else gap<0)
            perprec.append(rr);saved_balls.append(balls)
        assert all(saved_balls[0][name].overlaps(saved_balls[1][name]) for name in MASKS)
        rows.append({'eta':str(eta),'evaluations':perprec})
        print(eta,perprec[-1],flush=True)
    # Exact endpoint observed-law equivalence implies equal response; boxes overlap.
    ctx.prec=256
    base={name:coeff(generators(mask)[0],mask,256)[0] for name,mask in MASKS.items()}
    renewal=coeff(generators(MASKS['A'])[1],MASKS['A'],256)[0]
    ratios={name:str(100*(base[name]-renewal)/base[name]) for name in MASKS}
    ratios['pair_gap_percent_of_mean_coefficient']=str(200*(base['A']-base['B'])/(base['A']+base['B']))
    data={'complete':True,'exact':proof,'rows':rows,'coefficient_percentages':ratios,'two_precision_overlap':True,'method':'Arb monotone Newton matrix-square-root enclosures; 12 iterations; two precisions; no eigenvalue rounding or spatial mesh.','seconds':time.time()-started,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'MEMORY_ERASURE.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

if __name__=='__main__':main()
