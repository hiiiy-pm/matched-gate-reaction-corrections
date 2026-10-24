"""Exact ranks of Pair II event-density kernels, reconstructed from the masks.

No supplied numerical result is an input. Polynomial annihilation of the
killed blocks proves that the finite derivative matrices span the complete
exponential functions. A nonzero four-duration minor is a lower bound on
the dimension of a finite matrix-kernel representation, not a claim about
the minimum positive realization or the output of a specific RD algorithm.
"""
from pathlib import Path
from itertools import product
import argparse,hashlib,json
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,default=ROOT/'results/pair_ii_duration_ranks.json')
args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
Q=-s.eye(18)
for i in range(18):Q[i,(i-1)%18]=Q[i,(i+1)%18]=s.Rational(1,2)
ones=s.ones(9,1);z=s.symbols('z')
ann=(z+1)*(z+s.Rational(1,2))*(z+s.Rational(3,2))*((z+1)**2-s.Rational(1,2))

def certificate(M):
    ci=list(M.rref()[1]);ri=list(M[:,ci].T.rref()[1])
    return {'rank':len(ci),'row_indices_0_based':ri,'column_indices_0_based':ci,
            'nonzero_minor':str(M[ri,ci].det())}

def annihilates(M):
    return sum((v*M**k[0] for k,v in s.Poly(ann,z).terms()),s.zeros(9))==s.zeros(9)

qs={}
for m in (1,2,3):
    qm=-s.eye(m)
    for i in range(m-1):qm[i,i+1]=qm[i+1,i]=s.Rational(1,2)
    exit_rate=-qm*s.ones(m,1)
    qs[m]=[(qm**a*exit_rate)[0] for a in range(5)]

out={'definition':'density derivatives at zero; unit hidden gate rate',
     'killed_annihilating_polynomial':str(s.expand(ann)),'models':[],
     'rank_interpretation':'rank is for the complete kernel; no assertion of minimum positive realization',
     'common_triple_middle_rate':'3/2',
     'common_triple_middle_coefficient':'((q1(t)-q2(t))*(q1(v)-q2(v))+(q2(t)-q3(t))*(q2(v)-q3(v)))/40'}
for name,R in {'A':[0,1,4,5,6,8,11,12,14],'B':[0,1,4,6,7,10,11,12,14]}.items():
    C=[i for i in range(18) if i not in R]
    for color,X,Y in [('on',R,C),('off',C,R)]:
        A,B,D,E=Q.extract(X,X),Q.extract(X,Y),Q.extract(Y,X),Q.extract(Y,Y)
        nu=s.ones(1,9)*D/5
        assert (nu*ones)[0]==1 and annihilates(A) and annihilates(E)
        Ap=[A**a for a in range(5)];Ep=[E**a for a in range(5)]
        adjacent=s.Matrix(5,5,lambda a,b:(nu*Ap[a]*B*Ep[b]*D*ones)[0])
        same=s.Matrix(5,5,lambda a,b:(nu*Ap[a]*B*(-E).inv()*D*Ap[b]*B*ones)[0])
        pair_cert={'opposite_colors':certificate(adjacent),'same_color_with_middle_duration_integrated':certificate(same)}
        assert all(c['rank']==3 for c in pair_cert.values())
        left=s.Matrix.vstack(*[nu*Ap[a]*B*Ep[b]*D for a,b in product(range(5),repeat=2)])
        right=s.Matrix.hstack(*[Ap[c]*B*Ep[d]*D*ones for c,d in product(range(5),repeat=2)])
        H=left*right;hc=certificate(H)
        assert hc['rank']==8
        # This rational polynomial is the spectral projector at killed rate 3/2.
        P15=-8*(E+s.eye(9))*(E+s.eye(9)/2)*((E+s.eye(9))**2-s.eye(9)/2)
        assert P15*P15==P15 and E*P15==-s.Rational(3,2)*P15
        mode=s.Matrix(5,5,lambda a,c:(nu*Ap[a]*B*P15*D*Ap[c]*B*ones)[0])
        predicted=s.Matrix(5,5,lambda a,c:((qs[1][a]-qs[2][a])*(qs[1][c]-qs[2][c])+(qs[2][a]-qs[3][a])*(qs[2][c]-qs[3][c]))/40)
        assert mode==predicted and mode[0,0]==s.Rational(1,160)
        out['models'].append({'mask':name,'initial_color':color,
            'pair_density_ranks':pair_cert,'four_duration_cut':hc,
            'middle_mode_derivative_checks':25,'middle_mode_at_t0_v0':str(mode[0,0])})
out['all_pass']=True;out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
args.out.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
