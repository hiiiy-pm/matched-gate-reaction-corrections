"""Exploratory full-space Fourier residual majorant, not certified quadrature.

The analytic inequality is derived separately. This program evaluates a finite
log-frequency window in ordinary double precision and reports its sensitivity.
It never labels the integral a rigorous numerical upper bound.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy import linalg, special
from scipy.integrate import quad

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'exponential_local'))
import exponential_galerkin as eg


def create(n,opened,alpha,rates,cutoff,new,extension='blended',return_details=False):
    h,closed,aa,bb,left,right,constants,labels,cont=eg.build_basis(n,opened,rates)
    ev,u=linalg.eigh(h)
    ev[0]=0.
    # Fix the zero eigenvector sign and value to avoid spurious mean projections.
    u[:,0]=1/np.sqrt(n)
    cc=np.ix_(closed,closed)
    ec,vc=linalg.eigh(h[cc]);mu=np.sqrt(ec)
    v=np.zeros((n,len(closed)));v[closed]=vc
    k=(u[:,1:]*np.sqrt(ev[1:]))@u[:,1:].T
    j=(vc*mu)@vc.T
    bc=linalg.solve(k[cc]+j,np.ones(len(closed)),assume_a='pos')
    b=np.zeros(n);b[closed]=bc
    trace_jump=k@b
    trace_jump[closed]+=j@bc
    # Closed entries are analytically 1, without importing their floating error.
    trace_jump[closed]=1.
    meanzero=np.full(n,-len(closed)/len(opened));meanzero[closed]=1.
    t=trace_jump-meanzero
    modal_amplitudes=(v*(v.T@b)).T
    volume=modal_amplitudes@h
    volume[:,closed]=0.
    # The three arrays below have rows corresponding to negative exponential rates.
    vp=volume@u
    hp=meanzero@u;hp[0]=0.
    tp=t@u
    # On the common mode, analytically sum(volume/mu)=-t. Enforce this identity
    # via a stable cancellation formula in the Fourier evaluation below.
    cancellation=abs(np.sum(vp[:,0]/mu)+tp[0])
    A,f,nr,meta=eg.assemble(n,opened,alpha,rates)
    coeff=np.r_[bc,np.zeros(len(labels)-nr)]
    if new:
        a0=A[:nr,:nr];cross=A[:nr,nr:]
        base=linalg.solve(a0,f[:nr],assume_a='pos')
        residual=f[nr:]-cross.T@base
        schur=A[nr:,nr:]-cross.T@linalg.solve(a0,cross,assume_a='pos')
        schur=(schur+schur.T)/2
        scale=1/np.sqrt(np.diag(schur))
        ee,vv=linalg.eigh(scale[:,None]*schur*scale[None,:])
        keep=ee>cutoff*ee[-1]
        correction=scale*(vv[:,keep]@((vv[:,keep].T@(scale*residual))/ee[keep]))
        coeff=np.r_[base-linalg.solve(a0,cross@correction,assume_a='pos'),correction]
    lp=np.einsum('b,bki->ki',coeff,left)@u
    rp=np.einsum('b,bki->ki',coeff,right)@u
    cp=coeff@constants@u
    cp[1:]=0.
    oldcoeff=np.r_[bc,np.zeros(len(labels)-nr)]
    oldlp=np.einsum('b,bki->ki',oldcoeff,left)@u
    oldrp=np.einsum('b,bki->ki',oldcoeff,right)@u
    oldcp=oldcoeff@constants@u
    oldcp[1:]=0.
    s=alpha/2
    gam=special.gamma(1+s)
    def ztransform(l,r,c,x):
        if x>=1:
            # Continuity gives zhat=ghat/(i*x); this representation avoids
            # cancellation of the 1/x terms from the two sides and the plateau.
            gh=np.sum(l*aa[:,None]/(aa[:,None]-1j*x),axis=0)
            gh-=np.sum(r*bb[:,None]/(bb[:,None]+1j*x),axis=0)
            return gh/(1j*x)
        zh=np.sum(l/(aa[:,None]-1j*x),axis=0)+np.sum(r/(bb[:,None]+1j*x),axis=0)
        zh[0]+=c[0]/(1j*x)
        return zh
    def density_log(logx):
        x=np.exp(logx)
        symbol=x**(2*s)+ev
        zhat=ztransform(lp,rp,cp,x)
        if extension=='negative_fractional_integral':
            oldzhat=ztransform(oldlp,oldrp,oldcp,x)
            delta=1-s
            ellgamma=special.gammaln(1+s)+1j*np.pi*delta/2
            # f1=(xi^2+H)z1hat, f_s=Gamma(1+s)(-i*xi)^(s-1) f1.
            # The expm1 form makes the Brownian residual exactly zero.
            multiplier=x**(2*s)*np.expm1(ellgamma+delta*logx)
            multiplier+=ev*np.expm1(ellgamma-delta*logx)
            resid=multiplier*oldzhat+symbol*(oldzhat-zhat)
            return float(np.sum(abs(resid)**2/symbol)*x/(n*np.pi))
        # Spectral components of the extension force.
        fs=gam*(hp*(-1j*x)**(s-1)+tp*(1-1j*x)**(s-1)+np.sum(vp/(mu[:,None]-1j*x),axis=0))
        fs[0]=gam*(tp[0]*np.expm1((s-1)*np.log(1-1j*x))+
                   np.sum(vp[:,0]*(1j*x)/(mu*(mu-1j*x))))
        resid=fs-symbol*zhat
        return float(np.sum(abs(resid)**2/symbol)*x/(n*np.pi))
    trial=(2*f@coeff-coeff@A@coeff)/(n*gam**2)
    if return_details:
        return {'coefficients':coeff,'labels':labels,'closed':closed,'rates':rates}
    return density_log,float(trial),float(cancellation)


def main():
    cfg=json.loads((ROOT/'reproducible/results/round2/interval_C18_0064_95.json').read_text())
    rows=[]
    rates=[2.**k for k in range(-4,5) if k]
    for alpha in (2.,1.999):
        for opened in ('A_open','B_open'):
            for new in (False,True):
                density,trial,cancel=create(cfg['n'],cfg[opened],alpha,rates,1e-12,new)
                estimates=[]
                for window in (16,24,32):
                    val,err=quad(density,-window,window,epsabs=1e-14,epsrel=1e-8,limit=500,points=[0])
                    estimates.append({'log_frequency_window':[-window,window],
                                      'finite_window_integral_estimate':val,
                                      'quad_reported_error_estimate':err})
                rows.append({'gate':opened,'alpha':alpha,'trial':'new_Galerkin' if new else 'fixed_Brownian',
                             'normalized_trial':trial,'mean_force_cancellation_defect':cancel,
                             'estimates':estimates})
    out={'status':'EXPLORATORY_FINITE_WINDOW_RESIDUAL_INTEGRAL_NOT_NUMERICALLY_CERTIFIED',
         'warning':'No certified quadrature or tail enclosure; do not use these estimates as rigorous bounds.',
         'rows':rows}
    (HERE/'extended_source_residual_pilot.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
