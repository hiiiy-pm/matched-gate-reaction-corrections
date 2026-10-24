"""Exploratory optimization of an annihilating source extension.

Added forces are supported in open components on x<0, where every admissible
test vanishes. Component-difference forces have no common mode; the other
forces have zero spatial integral and may have a nonzero common mode away
from zero frequency. Both preserve zero total common mass. Thus the extension
represents the same restricted functional without a low-frequency divergence.
Numerical integration/optimization here is NOT a certificate.
"""
import probe_imports as p
import json,time
import numpy as np
from scipy import linalg,special
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
import exponential_galerkin as eg
from extended_source_residual_pilot import create

def make(alpha,opened,level):
    rates=[2.**k for k in range(-level,level+1) if k]
    details=create(18,opened,alpha,rates,1e-12,True,return_details=True)
    coeff=details['coefficients']
    h,closed,aa,bb,left,right,constants,labels,cont=eg.build_basis(18,opened,rates)
    ev,u=linalg.eigh(h);ev[0]=0.;u[:,0]=1/np.sqrt(18)
    ec,vc=linalg.eigh(h[np.ix_(closed,closed)])
    k=(u[:,1:]*np.sqrt(ev[1:]))@u[:,1:].T
    bc=linalg.solve(k[np.ix_(closed,closed)]+(vc*np.sqrt(ec))@vc.T,np.ones(len(closed)),assume_a='pos')
    oldcoeff=np.r_[bc,np.zeros(len(labels)-len(closed))]
    def modes(c):
        lp=np.einsum('b,bki->ki',c,left)@u
        rp=np.einsum('b,bki->ki',c,right)@u
        cp=c@constants@u;cp[1:]=0.
        return lp,rp,cp
    old=modes(oldcoeff);delta=modes(coeff-oldcoeff)
    def ztransform(args,x):
        lp,rp,cp=args
        if x>=1:
            return (np.sum(lp*aa[:,None]/(aa[:,None]-1j*x),axis=0)
                    -np.sum(rp*bb[:,None]/(bb[:,None]+1j*x),axis=0))/(1j*x)
        z=np.sum(lp/(aa[:,None]-1j*x),axis=0)+np.sum(rp/(bb[:,None]+1j*x),axis=0)
        z[0]+=cp[0]/(1j*x)
        return z
    s=alpha/2;d=1-s;lg=special.gammaln(1+s)+1j*np.pi*d/2
    def residual(t):
        x=np.exp(t);power=x**(2*s);symbol=power+ev
        r=(power*np.expm1(lg+d*t)+ev*np.expm1(lg-d*t))*ztransform(old,x)-symbol*ztransform(delta,x)
        return r,symbol
    return residual,u,details

def optimize(alpha,key,level,force_level=4,integral_zero=False):
    start=time.perf_counter();opened=p.MASKS[key]
    residual,u,details=make(alpha,opened,level)
    frates=[2.**k for k in range(-force_level,force_level+1,2)]
    # For each rate use e_j-e_ref; the Fourier zero mode is exactly zero.
    labels=[{'kind':'component_difference','component':int(j),'rate_exact_decimal':str(a)}
            for a in frates for j in opened[:-1]]
    if integral_zero:
        labels += [{'kind':'zero_integral','component':int(j),'rate_exact_decimal':str(a)}
                   for a in frates if a!=1 for j in opened]
    vectors=np.array([u[z['component']]-u[opened[-1]] if z['kind']=='component_difference'
                      else u[z['component']] for z in labels])
    iz=np.array([z['kind']=='zero_integral' for z in labels])
    vectors[~iz,0]=0.
    avec=np.array([float(z['rate_exact_decimal']) for z in labels])
    def forcing(t):
        x=np.exp(t);f=1/(avec-1j*x)
        f[iz]=1j*x*(1-avec[iz])/((avec[iz]-1j*x)*(1-1j*x))
        return vectors*f[:,None]
    gx,gw=leggauss(16);ts=[];ws=[]
    for k in range(-32,32):
        ts.extend(k+.5+.5*gx);ws.extend(.5*gw)
    matrix=[];target=[]
    for t,w in zip(ts,ws):
        r,symbol=residual(t);weights=np.sqrt(w*np.exp(t)/(18*np.pi*symbol))
        matrix.append((forcing(t)*weights).T)
        target.append(r*weights)
    M=np.concatenate(matrix);y=np.concatenate(target)
    G=(M.conj().T@M).real;b=-(M.conj().T@y).real
    scale=1/np.sqrt(np.diag(G));e,v=linalg.eigh(scale[:,None]*G*scale[None,:]);keep=e>1e-12*e[-1]
    coeff=scale*(v[:,keep]@((v[:,keep].T@(scale*b))/e[keep]))
    def density(t):
        r,symbol=residual(t);r+=coeff@forcing(t)
        return float(np.sum(abs(r)**2/symbol)*np.exp(t)/(18*np.pi))
    estimates=[]
    for window in (24,32):
        val,err=quad(density,-window,window,epsabs=1e-13,epsrel=1e-8,limit=500,points=[0])
        estimates.append({'window':window,'normalized_estimate':val/special.gamma(1+alpha/2)**2,'quad_error':err})
    row={'status':'NUMERICAL_DIAGNOSTIC_NOT_CERTIFIED','alpha':alpha,'gate':key,'trial_level':level,
         'trial_rates_exact_decimal':[str(x) for x in details['rates']],
         'trial_coordinates_exact_decimal':[str(float(x)) for x in details['coefficients']],
         'force_reference_open_component':opened[-1],
         'force_labels':labels,
         'force_coordinates_exact_decimal':[str(float(x)) for x in coeff],
         'estimates':estimates,'seconds':time.perf_counter()-start}
    path=p.ROOT/f'dual_{key}_{str(alpha).replace(".","p")}_L{level}_F{force_level}_Z{int(integral_zero)}.json'
    path.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ('alpha','gate','trial_level','estimates','seconds')}),flush=True)
    return row

if __name__=='__main__':
    for level in (8,10,12):optimize(1.9,'B',level,4,True)
