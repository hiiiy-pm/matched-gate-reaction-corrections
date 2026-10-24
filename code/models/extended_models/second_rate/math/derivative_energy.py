"""Arb exact finite Fourier/atom derivative integral and an H1 trace extension.

The theorem is SECOND_ORDER_GREEN_ENERGY_THEOREM.zh-CN.md. This module does
not treat a sampled derivative or a floating-point FFT as a certified norm.
Input positive-frequency coefficients contain cell full G and atom gate R;
the analytic common atomic G0 is added here. Coefficient lists are consumed.
"""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
from flint import arb,acb,ctx

HERE=Path(__file__).resolve().parent

def outward(x):
    return [math.nextafter(float(x.lower()),-math.inf),math.nextafter(float(x.upper()),math.inf)]

def phase_table():
    return [acb(0,2*arb.pi()*j/32).exp() if j else acb(1) for j in range(32)]

def moment_fractional(k,nu,w,phase,series_terms=256,parts=16):
    """Enclose integral_0^w x^(nu-1)*exp(i2pi*k*x) dx, 0<nu<1."""
    omega=2*arb.pi()*k
    if k<=256:
        z=acb(0,omega*w);term=acb(1);value=acb(1/nu)
        for j in range(1,series_terms+1):
            term*=z/j;value+=term/(nu+j)
        ratio=(omega*w)/(series_terms+2)
        assert ratio.upper()<1
        tail=term.abs_upper()*(omega*w)/(series_terms+1)/(nu+series_terms+1)/(1-ratio)
        value*=w**nu;tail*=w**nu
        return value+acb(arb(0,tail.abs_upper()),arb(0,tail.abs_upper()))
    iw=acb(0,omega)
    term=w**(nu-1)/iw;value=acb(0)
    for j in range(parts):
        value+=term
        term*=(-(nu-1-j))/(iw*w)
    value=nu.gamma()*acb(0,arb.pi()*nu/2).exp()*omega**(-nu)+phase*value
    tail=2*term.abs_upper()
    return value+acb(arb(0,tail.abs_upper()),arb(0,tail.abs_upper()))

def integral_exp(m,w,phases):
    if m==0:return acb(w)
    if m%32==0:return acb(0)
    return (phases[m%32]-1)/acb(0,2*arb.pi()*m)

def finite_fourier_square(drows,w,atoms0,atoms1):
    """Sum integral(p_i')² using positive-frequency convolutions of length2K.

    drows are consumed and replaced by empty lists. Two inverse DFTs suffice
    after summing spectral products over all reacting states.
    """
    started=time.perf_counter();K=len(drows[0]);L=2*K
    assert K>1 and K&(K-1)==0
    sum_square=[acb(0)]*L;sum_abs=[acb(0)]*L
    b0=[acb(0)]*K;b1=[acb(0)]*K
    for i,row in enumerate(drows):
        assert len(row)==K
        for k,z in enumerate(row):
            b0[k]+=atoms0[i]*z;b1[k]+=atoms1[i]*z
        transformed=acb.dft(row+[acb(0)]*K)
        drows[i]=[]
        for j,z in enumerate(transformed):
            sum_square[j]+=z*z
            sum_abs[j]+=acb(z.real*z.real+z.imag*z.imag)
        del transformed,row
        print(json.dumps({'stage':'derivative_positive_fft','reacting_state_index':i,'K':K,'seconds':time.perf_counter()-started}),flush=True)
    conv=acb.dft(sum_square,inverse=True);del sum_square
    corr=acb.dft(sum_abs,inverse=True);del sum_abs
    phases=phase_table()
    absolute_part=w*corr[0].real
    for m in range(1,K):
        if m%32:absolute_part+=2*(corr[m]*integral_exp(m,w,phases)).real
    del corr
    positive_square=arb(0)
    for j in range(2*K-1):
        m=j+2
        if m%32:positive_square+=(conv[j]*integral_exp(m,w,phases)).real
    del conv
    result=2*absolute_part+2*positive_square
    return result,b0,b1

def analytic_atom_parameters(alpha,w,J=8):
    beta=alpha-1;pi=arb.pi()
    C=2*(1-alpha).gamma()*(pi*alpha/2).sin()*(2*pi)**beta
    nus=[beta];coeff=[C*beta]
    for j in range(1,J+1):
        pj=2*(alpha-2*j).zeta()*((-1)**j)*(2*pi)**(2*j)/math.factorial(2*j)
        nus.append(arb(2*j));coeff.append(2*j*pj)
    self_int=arb(0);mirror_int=arb(0)
    for nu,c in zip(nus,coeff):
        for mu,d in zip(nus,coeff):
            factor=c*d*w**(nu+mu-1)
            self_int+=factor/(nu+mu-1)
            mirror_int+=factor*nu.gamma()*mu.gamma()/(nu+mu).gamma()
    P=abs(2**(alpha+1)*pi**(alpha-1)*(pi*alpha/2).sin())*arb(3).zeta()
    tail_derivative=2*P*w**(2*J+1)*((J+1)/(1-w*w)+w*w/(1-w*w)**2)
    return beta,nus,coeff,self_int,mirror_int,tail_derivative

def certify(data,trial,p_coeffs,formal_potential_json):
    """Return a rigorous candidate derivative and residual-energy upper bound.

    All arrays are Arb/Acb, at the caller's ctx.prec. No archive files are
    modified here. formal_potential_json is the exact V9 selected certificate.
    """
    started=time.perf_counter();K=len(p_coeffs[0]);N=data['N']
    M=data['witness']['n'];R=trial['R'];r=len(R)
    alpha=arb(str(data['witness']['alpha']));w=arb(1)/32;gamma=arb(str(data['gamma']))
    assert len(p_coeffs)==r and alpha.lower()>arb('1.5')
    assert formal_potential_json['witness']==data['witness']['id']
    assert formal_potential_json['label']==trial['label'] and formal_potential_json['source_N']==N
    cell=[arb(float.fromhex(z)) for z in trial['charge_exact_float_hex']]
    atoms=[arb(float.fromhex(z)) for z in trial['atom_exact_float_hex']]
    mass=sum(cell+atoms,arb(0));cell=[z/mass for z in cell];atoms=[z/mass for z in atoms]
    a0=atoms[:r];a1=atoms[r:]
    tv2=arb(0)
    for j in range(r):
        seq=[arb(0)]+[cell[t*r+j] for t in range(N//32)]+[arb(0)]
        tv=sum((abs(b-a) for a,b in zip(seq,seq[1:])),arb(0));tv2+=tv*tv
    ba2=sum(((abs(x)+abs(y))**2 for x,y in zip(a0,a1)),arb(0))
    del cell
    # Differentiate in-place; the exact common atomic derivative is separate.
    for row in p_coeffs:
        for k in range(K):row[k]*=acb(0,2*arb.pi()*(k+1))
    finite,b0,b1=finite_fourier_square(p_coeffs,w,a0,a1)
    beta,nus,coeff,self_int,mirror_int,poly_tail=analytic_atom_parameters(alpha,w)
    aa=sum((x*x+y*y for x,y in zip(a0,a1)),arb(0))
    ab=sum((x*y for x,y in zip(a0,a1)),arb(0))
    atomic=aa*self_int-2*ab*mirror_int
    phases=phase_table();powers=[w**n for n in range(16)]
    cross_sum=arb(0)
    for j in range(K):
        k=j+1;phase=phases[k%32];iw=acb(0,2*arb.pi()*k)
        F=moment_fractional(k,beta,w,phase)
        U=coeff[0]*F
        prev=(phase-1)/iw
        for n in range(2,17):
            prev=powers[n-1]*phase/iw-(n-1)*prev/iw
            if n%2==0:U+=coeff[n//2]*prev
        cross_sum+=(b0[j]*U-b1[j]*phase*U.conjugate()).real
        if k%65536==0:print(json.dumps({'stage':'exact_atom_fourier_cross','k':k,'K':K,'seconds':time.perf_counter()-started}),flush=True)
    del b0,b1
    cross=4*cross_sum
    Jfinite=finite+cross+atomic
    if Jfinite.upper()<0:raise ArithmeticError(('Negative squared integral',str(Jfinite)))
    finite_norm=Jfinite.upper().sqrt()
    cell_tail=(2*N*N*tv2*arb(K)**(1-2*alpha)/(2*alpha-1)).sqrt()
    atom_tail=(32*arb.pi()**2*gamma**2*ba2*arb(K)**(3-4*alpha)/(4*alpha-3)).sqrt()
    analytic_tail=(ba2*w).sqrt()*poly_tail
    derivative_upper=(finite_norm+cell_tail+atom_tail+analytic_tail)**2
    # c is a fixed exact dyadic constant. It is not asserted to be a true mean.
    c_float=trial['numeric_mean_diagnostic']/M;c=arb(c_float)
    records={z['gate']:z for z in formal_potential_json['gate_records']}
    delta2=arb(0)
    for gate in R:
        ends=records[gate]['potential_enclosure']
        enclosure=arb(ends[0]).union(arb(ends[1]))
        delta=abs(enclosure-c).upper();delta2+=delta*delta
    eta=arb(1)/8192;Z=2*delta2;V=w*delta2
    B_derivative=(derivative_upper+Z/eta)/(2*arb.pi())**2
    B_gate=2*gamma*(V+eta*Z/3)
    B=B_derivative+B_gate
    return {'status':'CONDITIONAL_ARBITRARY_TRIAL_SECOND_ORDER_RESIDUAL_BOUND',
            'witness':data['witness']['id'],'label':trial['label'],'gamma':str(data['gamma']),
            'alpha':str(data['witness']['alpha']),'source_N':N,'fourier_cutoff':K,'precision_bits':ctx.prec,
            'fixed_c_exact_binary64_hex':float(c_float).hex(),'eta_exact':'1/8192',
            'finite_fourier_derivative_square':outward(finite),'fourier_atom_cross_term':outward(cross),
            'analytic_atom_derivative_square':outward(atomic),'finite_combined_derivative_square':outward(Jfinite),
            'cell_derivative_L2_tail':outward(cell_tail),'atom_gate_derivative_L2_tail':outward(atom_tail),
            'analytic_atom_series_derivative_L2_tail':outward(analytic_tail),
            'target_derivative_square_upper':outward(derivative_upper),'target_trace_delta2_bound':outward(delta2),
            'extension_derivative_energy_bound':outward(B_derivative),'extension_gate_energy_bound':outward(B_gate),
            'green_energy_defect_upper':outward(B),'tau_variational_defect_upper':outward(M*B),
            'seconds':time.perf_counter()-started,'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'contract':'Second-order trace extension theorem, exact finite Fourier/atom integrals and explicit infinite tails; Arb enclosing-arithmetic contract. No sampled derivative approximation.'}
