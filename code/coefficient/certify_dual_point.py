"""Enclosing-arithmetic test of a fixed primal trial and source extension.

This extends the inherited full-space dual residual inequality by a force that
annihilates the constrained test space. It is a point test, not a range proof.
All optimization output is frozen as exact rational decimal coordinates.
"""
import probe_imports as p
import argparse,json,time,hashlib,heapq
import mpmath as mp
import interval_residual_pilot as old
from extended_source_residual_pilot import create
I=mp.iv
sm=old.sm;dot=old.dot;norm=old.norm

def force_data(model,cfg):
    ps={};zs={};ref=cfg['force_reference_open_component']
    for label,c in zip(cfg['force_labels'],cfg['force_coordinates_exact_decimal']):
        a=label['rate_exact_decimal'];j=label['component'];c=I.mpf(c)
        rows=ps if label['kind']=='component_difference' else zs
        if a not in rows:rows[a]=[I.mpf(0) for _ in range(model.n)]
        rows[a][j]+=c
        if label['kind']=='component_difference':rows[a][ref]-=c
    return [(row,I.mpf(a)) for a,row in ps.items()],[(row,I.mpf(a)) for a,row in zs.items()]

def operators(model,cfg,s):
    n=model.n;coeff=cfg['trial_coordinates_exact_decimal'];rates=cfg['trial_rates_exact_decimal']
    us=[[1/I.sqrt(n)]*n]+model.ring;eig=[I.mpf(0)]+[l*l for l in model.lam]
    left,right,constant=old.physical_trial(model,coeff,rates)
    oldleft,oldright,oldconstant=old.brownian_physical(model)
    db=[-x for x in model.b]
    for j,x in zip(model.C,coeff[:len(model.C)]):db[j]+=I.mpf(x)
    dl,dr,dc=old.brownian_physical(model,db)
    dl.extend(left[len(model.mu):]);dr.extend(right[len(model.lam):]);dc+=I.mpf(coeff[-1])
    oldlp=old.eig_project(oldleft,us);oldrp=old.eig_project(oldright,us)
    dlp=old.eig_project(dl,us);drp=old.eig_project(dr,us)
    oldcp=oldconstant*I.sqrt(n);dcp=dc*I.sqrt(n)
    ps,zs=force_data(model,cfg)
    pp=[((I.mpf(0),*row[1:]),a) for row,a in old.eig_project(ps,us)]
    zp=old.eig_project(zs,us)
    delta=1-s;lg=I.loggamma(1+s);phase=I.pi*delta/2
    def transform(lp,rp,cp,x,high):
        if high:
            li=[a/(a-I.j*x) for _,a in lp];ri=[-a/(a+I.j*x) for _,a in rp]
            return [(sm(row[k]*z for (row,_),z in zip(lp,li))+
                     sm(row[k]*z for (row,_),z in zip(rp,ri)))/(I.j*x) for k in range(n)]
        li=[1/(a-I.j*x) for _,a in lp];ri=[1/(a+I.j*x) for _,a in rp]
        out=[sm(row[k]*z for (row,_),z in zip(lp,li))+sm(row[k]*z for (row,_),z in zip(rp,ri)) for k in range(n)]
        out[0]+=cp/(I.j*x)
        return out
    def density(t,high):
        x=I.exp(t);power=I.exp(2*s*t)
        o=transform(oldlp,oldrp,oldcp,x,high);dz=transform(dlp,drp,dcp,x,high)
        ep=I.expm1(I.mpc(lg+delta*t,phase));em=I.expm1(I.mpc(lg-delta*t,phase))
        factorsp=[1/(a-I.j*x) for _,a in pp]
        factorsz=[I.j*x*(1-a)/((a-I.j*x)*(1-I.j*x)) for _,a in zp]
        total=I.mpf(0)
        for k,h in enumerate(eig):
            sym=power+h
            force=sm(row[k]*v for (row,_),v in zip(pp,factorsp))+sm(row[k]*v for (row,_),v in zip(zp,factorsz))
            r=(power*ep+h*em)*o[k]-sym*dz[k]+force
            total+=(r.real**2+r.imag**2)/sym
        return total*x/(n*I.pi)
    return density,left,right,constant,ps,zs

def force_tails(model,ps,zs,s,window):
    n=model.n;lo=I.exp(-window);hi=I.exp(window);gap=1-I.cos(2*I.pi/n)
    c0=sm(norm(row)/a for row,a in ps)
    c1=sm(norm(row)*abs(1-a)/a for row,a in zs)
    cinf=sm(norm(row) for row,a in ps)+sm(norm(row)*abs(1-a) for row,a in zs)
    # P is exactly mean zero. Z-hat is O(xi) at zero, including its common mode.
    low=(c0*I.sqrt(lo/gap)+c1*I.sqrt(lo**(3-2*s)/(3-2*s)))**2/(n*I.pi)
    high=cinf**2*hi**(-1-2*s)/((1+2*s)*n*I.pi)
    return low,high

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panels',type=int,default=2048)
    ap.add_argument('--dps',type=int,default=45);ap.add_argument('--level',type=int,default=8)
    ap.add_argument('--adaptive',action='store_true')
    args=ap.parse_args();mp.mp.dps=args.dps+25;I.dps=args.dps
    src=p.ROOT/f'dual_B_1p9_L{args.level}_F4_Z1.json'
    cfg=json.loads(src.read_text());s=I.mpf('1.9')/2
    model=old.eng.Model(p.MASKS['B'],18,dps=args.dps)
    density,left,right,constant,ps,zs=operators(model,cfg,s)
    jb=old.trial_objective(model,left,right,s)
    window=32
    l0,h0=old.tails(model,left,right,constant,s,window)
    lq,hq=force_tails(model,ps,zs,s,window)
    low=(I.sqrt(l0)+I.sqrt(lq))**2;high=(I.sqrt(h0)+I.sqrt(hq))**2
    # Freeze A's independent admissible trial, only a lower bound is needed.
    details=create(18,p.MASKS['A'],1.9,[float(x) for x in cfg['trial_rates_exact_decimal']],1e-12,True,return_details=True)
    acoords=[str(float(x)) for x in details['coefficients']]
    ma=old.eng.Model(p.MASKS['A'],18,dps=args.dps)
    al,ar,ac=old.physical_trial(ma,acoords,cfg['trial_rates_exact_decimal'])
    ja=old.trial_objective(ma,al,ar,s)
    print(json.dumps({'stage':'setup','J_A':old.bounds(ja),'J_B':old.bounds(jb),
                      'trial_gap':old.bounds(ja-jb),'low_tail':old.bounds(low),'high_tail':old.bounds(high)}),flush=True)
    start=time.perf_counter();integral=I.mpf(0);width=I.mpf(2*window)/args.panels
    grid=[]
    if args.adaptive:
        heap=[];counter=0
        def cell(a,b):
            nonlocal counter
            assert a>=0 or b<=0
            val=density(I.mpf([a,b]),a>=0)*I.mpf(b-a)
            item=(-float(val.delta),counter,a,b,val);counter+=1
            return item
        for k in range(128):
            a=mp.mpf(-window)+mp.mpf(k)/2;b=a+mp.mpf('0.5')
            heapq.heappush(heap,cell(a,b))
        upper=sm(I.mpf(z[4].b) for z in heap)
        target=mp.mpf((ja-jb).a)*mp.mpf(I.gamma(1+s).a)**2*mp.mpf('0.5')
        while len(heap)<args.panels and mp.mpf(upper.b)>target:
            _,_,a,b,val=heapq.heappop(heap);mid=(a+b)/2
            children=[cell(a,mid),cell(mid,b)]
            for z in children:heapq.heappush(heap,z)
            upper=upper-I.mpf(val.b)+sm(I.mpf(z[4].b) for z in children)
            if len(heap)%256==0:
                print(json.dumps({'adaptive_panels':len(heap),'integral_upper':float(upper.b),'seconds':time.perf_counter()-start}),flush=True)
        integral=sm(z[4] for z in heap)
        grid=[[mp.nstr(z[2],60),mp.nstr(z[3],60)] for z in sorted(heap,key=lambda z:z[2])]
    else:
        for k in range(args.panels):
            al=-I.mpf(window)+k*width;be=al+width
            t=I.mpf([mp.mpf(al.a),mp.mpf(be.b)])
            integral+=density(t,k>=args.panels//2)*width
            if k%256==0:print(json.dumps({'panel':k,'seconds':time.perf_counter()-start}),flush=True)
    residual=(integral+I.mpf([0,mp.mpf(low.b)])+I.mpf([0,mp.mpf(high.b)]))/I.gamma(1+s)**2
    lower=ja-jb-I.mpf([0,mp.mpf(residual.b)])
    result={'status':'POINT_LOWER_BOUND_CONDITIONAL_ON_ANALYTIC_RESIDUAL_INEQUALITY',
       'alpha_exact':'1.9','continuous_alpha_range':False,'panels':len(grid) or args.panels,'dps':args.dps,
       'adaptive':args.adaptive,'exact_log_frequency_grid':grid,
       'trial_A':old.bounds(ja),'trial_B':old.bounds(jb),'trial_gap':old.bounds(ja-jb),
       'residual_B_squared_with_tails':old.bounds(residual),
       'gap_lower_bound':old.bounds(lower)[0],'positive_lower_bound':old.bounds(lower)[0]>0,
       'finite_window_integral':old.bounds(integral),'low_tail':old.bounds(low),'high_tail':old.bounds(high),
       'seconds':time.perf_counter()-start,'A_coordinates_exact_decimal':acoords,
       'source_sha256':hashlib.sha256(p.Path(__file__).read_bytes()).hexdigest(),
       'dual_coordinates_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
       'contract':'Inherited SI residual/source extension identities plus the annihilator and explicit tail bounds in this script. Not external formal verification.'}
    path=p.ROOT/f'point_1p9_L{args.level}_{args.panels}_{args.dps}_adaptive{int(args.adaptive)}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if 'coordinates' not in k and k!='exact_log_frequency_grid'}),flush=True)

if __name__=='__main__':main()
