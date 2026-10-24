"""Interval-certified finite-rate Brownian means from an independent ODE solution."""
from pathlib import Path
import sys,json,time,hashlib
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'code'))
import interval_engine_arb as ab
from flint import arb,arb_mat,ctx
W=next(w for w in json.loads((R/'data/witnesses.json').read_text()) if w['id']=='C18_0189')
def tanh(x):
    e=(-2*x).exp()
    return (1-e)/(1+e)
def zero(n):return [[arb(0) for j in range(n)] for i in range(n)]
def means(mask,gamma,width,prec):
    ctx.prec=prec;n=18;D=1/(2*arb.pi())**2
    w=arb(str(width));g=arb(gamma);L=(1-w)/2;c=[i for i in range(n) if i not in mask]
    m=len(c);where={j:i for i,j in enumerate(c)}
    C=zero(m);B=zero(m);F=zero(m);h=[arb(0) for i in c]
    for run in ab.runs(c,n):
        length=len(run)
        for k in range(1,length+1):
            lam=1-(arb.pi()*k/(length+1)).cos();K=(g*lam/D).sqrt()
            vec=[arb(0) for i in c]
            for pos,j in enumerate(run,1):
                vec[where[j]]=(arb(2)/(length+1)).sqrt()*(arb.pi()*k*pos/(length+1)).sin()
            vc=sum(vec,arb(0));t=tanh(K*w/2)
            for i in range(m):
                h[i]+=vec[i]*vc/(g*lam)
                for j in range(m):
                    C[i][j]+=vec[i]*vec[j]*K*t
                    B[i][j]+=vec[i]*vec[j]*t/K
    for k in range(1,n):
        lam=1-(2*arb.pi()*k/n).cos();K=(g*lam/D).sqrt();value=K*tanh(K*L)/n
        for i in range(m):
            for j in range(m):
                F[i][j]+=value*(2*arb.pi()*k*(c[i]-c[j])/n).cos()
    C=arb_mat(C);B=arb_mat(B);F=arb_mat(F);h=arb_mat([[z] for z in h])
    one=arb_mat([[arb(1)] for i in c])
    v=(C+F).inv()*(one*(L/D)+C*h)
    residual=(C+F)*v-(one*(L/D)+C*h)
    assert all(residual[i,0].contains(0) for i in range(m))
    interior=B*(v-h)
    tau=2*L/n*sum((v[i,0] for i in range(m)),arb(0))+2*L**3/(3*D)
    tau+=w/n*sum((h[i,0] for i in range(m)),arb(0))
    tau+=arb(2)/n*sum((interior[i,0] for i in range(m)),arb(0))
    assert tau>0
    return tau
rows=[];t0=time.perf_counter()
for width in [1/128,1/64,1/32,1/16,1/8,1/4]:
    for gamma in [1,10,100,1000,10000]:
        vals=[]
        for prec in [160,256]:
            A=means(W['A'],gamma,width,prec);B=means(W['B'],gamma,width,prec)
            gap=A-B;ppm=arb(1000000)*gap/((A+B)/2)
            assert gap>0
            vals.append({'prec':prec,'mean_A':ab.bounds(A),'mean_B':ab.bounds(B),
                'gap':ab.bounds(gap),'relative_ppm':ab.bounds(ppm)})
        for key in ['mean_A','mean_B','gap','relative_ppm']:
            assert max(z[key][0] for z in vals)<=min(z[key][1] for z in vals)
        rows.append({'alpha':2,'width':width,'gamma':gamma,'backends':vals,
            'relative_ppm_enclosure':vals[1]['relative_ppm'],'positive':True})
        print(width,gamma,vals[1]['relative_ppm'],flush=True)
path=R/'results/BROWNIAN_FINITE_RATE.json'
path.write_text(json.dumps({'complete':True,'method':'Piecewise constant-coefficient ODE, matching values and fluxes at target endpoints; no spatial grid. Arb balls at 160 and 256 bits.',
    'rows':rows,'elapsed_seconds':time.perf_counter()-t0,
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
