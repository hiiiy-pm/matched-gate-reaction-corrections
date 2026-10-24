"""Positive-kernel interval enclosure for the periodic permanent-target trace.

No gate, circle reaction output, or V8 floating Jacobi coefficients are imported.
The proof uses a positive resolvent, an explicit first Neumann correction, and
positive rectangle bounds, rather than an unverified spectral truncation.
Conditional on mpmath.iv enclosing elementary/Gamma operations; see derivation.
"""
from pathlib import Path
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
import argparse, hashlib, json, time
import mpmath as mp

HERE = Path(__file__).resolve().parent
I = mp.iv

def exact(x):
    return I.mpf(x)

def lo(x):
    return mp.mpf(x._mpi_[0])

def hi(x):
    return mp.mpf(x._mpi_[1])

def decimal_bound(raw, upper, digits=55):
    """Serialize a binary endpoint with explicitly outward decimal rounding."""
    sign, mantissa, exponent, _ = raw
    if not mantissa:
        return '0'
    with localcontext() as ctx:
        ctx.prec = max(500, abs(exponent)+digits+50)
        value = Decimal(mantissa)*Decimal(2)**exponent
        if sign:
            value = -value
        quantum = Decimal(1).scaleb(value.adjusted()-digits+1)
        return format(value.quantize(quantum, rounding=ROUND_CEILING if upper else ROUND_FLOOR), 'f')

def upper_decimal(x):
    return decimal_bound(x._mpi_[1], True)

def endpoints(x):
    return {'lower_decimal': decimal_bound(x._mpi_[0], False),
            'upper_decimal': upper_decimal(x)}

def bracket(lower, upper):
    return I.mpf([lo(lower), hi(upper)])

def beta(a, b):
    return I.gamma(a)*I.gamma(b)/I.gamma(a+b)

def kernel_point(d, s, images):
    """K(d)=sum_{m>=1}[(m-d)^(-1-2s)+(m+d)^(-1-2s)].

    Integral-test bounds enclose every omitted image. d>=0 and d<=31/32.
    """
    p = 1+2*s
    total = exact(0)
    for m in range(1, images+1):
        total += (m-d)**(-p)+(m+d)**(-p)
    tail_lower = ((images+1-d)**(-2*s)+(images+1+d)**(-2*s))/(2*s)
    tail_upper = ((images-d)**(-2*s)+(images+d)**(-2*s))/(2*s)
    return total + bracket(tail_lower, tail_upper)

def certificate(alpha_text, panels, images, dps):
    start = time.perf_counter()
    alpha = exact(alpha_text)
    s = alpha/2
    width = exact(1)/32
    length = 1-width
    a = length/2
    D = (2*I.pi)**(-alpha)
    lam = I.gamma(1+2*s)
    c = 2*s*2**(2*s-1)*I.gamma((1+2*s)/2)/(I.sqrt(I.pi)*I.gamma(1-s))
    # Endpoint convexity reduces sup_x Wv(x) to x=a. The m=1 near
    # image uses v(y)<=length^s*(a+y)^s and an integral to infinity.
    # All other positive images are bounded by the exact v mass times
    # 1+2*zeta(1+2s) <= 3+1/s.
    v_mass = a**(2*s+1)*beta(exact('0.5'), s+1)
    M_upper = c*(length**s*width**(-s)*beta(s+1,s)
                 + v_mass*(3+1/s))
    q = bracket(exact(0), M_upper/lam)
    assert hi(q) < 1
    q_upper = exact(hi(q))
    # Positive cell masses for z in [0,1]: endpoint Martin weight
    # (1-z^2)^(s-1), and interior source weight (1-y^2)^s.
    z_mass = []
    y_mass = []
    for j in range(panels):
        left = exact(j)/panels
        right = exact(j+1)/panels
        z_basic = ((1-left)**s-(1-right)**s)/s
        z_factor = bracket((1+right)**(s-1), (1+left)**(s-1))
        z_mass.append(z_basic*z_factor)
        y_basic = ((1-left)**(s+1)-(1-right)**(s+1))/(s+1)
        y_factor = bracket((1+left)**s, (1+right)**s)
        y_mass.append(y_basic*y_factor)
    # K is even and increasing in |d|. Uniform cell endpoints mean
    # only 2*N+1 kernel values are required for all N^2 rectangles.
    kernels = [kernel_point(a*exact(j)/panels,s,images)
               for j in range(2*panels+1)]
    integral_lower = exact(0)
    integral_upper = exact(0)
    for j in range(panels):
        row_lower = exact(0)
        row_upper = exact(0)
        for k in range(panels):
            nearest_difference = max(abs(j-k)-1,0)
            farthest_difference = abs(j-k)+1
            kl = (exact(lo(kernels[nearest_difference]))
                  + exact(lo(kernels[j+k])))
            ku = (exact(hi(kernels[farthest_difference]))
                  + exact(hi(kernels[j+k+2])))
            row_lower += y_mass[k]*kl
            row_upper += y_mass[k]*ku
        integral_lower += z_mass[j]*row_lower
        integral_upper += z_mass[j]*row_upper
    integral = bracket(integral_lower, integral_upper)
    martin_mass = beta(exact('0.5'),s)/2
    qbar = c*a**(2*s+1)*integral/(lam*martin_mass)
    A0 = length**s/(D*lam)
    # g=(D*lambda)^-1 v and u=g+Tg+T^2g+..., with 0<=Tg<=q*g.
    # Hence A0(1+qbar) <= A <= A0(1+qbar/(1-q)).
    A = bracket(A0*(1+exact(lo(qbar))),
                A0*(1+exact(hi(qbar))/(1-q_upper)))
    K = 2*D*I.gamma(1+s)**2*A**2
    midpoint = (lo(K)+hi(K))/2
    radius = (hi(K)-lo(K))/2
    # This displayed relative bound uses the smallest possible true K.
    relative_midpoint_bound = radius/lo(K)
    return dict(alpha_exact_decimal=alpha_text,width_exact_rational='1/32',
                D_definition='(2*pi)^(-alpha)',panels_per_axis=panels,
                explicitly_summed_images=images,interval_dps=dps,
                lambda0=endpoints(lam),image_constant=endpoints(c),
                Wv_sup_upper_decimal=upper_decimal(M_upper),
                weighted_operator_contraction_upper_decimal=upper_decimal(q),
                first_trace_correction_relative_interval=endpoints(qbar),
                real_line_trace_interval=endpoints(A0),
                periodic_left_trace_interval=endpoints(A),
                periodic_right_trace_interval=endpoints(A),
                K_interval=endpoints(K),K_midpoint_display=mp.nstr(midpoint,18),
                K_relative_midpoint_error_bound_display=mp.nstr(relative_midpoint_bound,18),
                seconds=time.perf_counter()-start)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--alpha',nargs='+',default=['1.999','1.997'])
    ap.add_argument('--panels',type=int,nargs='+',default=[64,128,256])
    ap.add_argument('--images',type=int,default=16)
    ap.add_argument('--dps',type=int,default=40)
    ap.add_argument('--out',default='geometry_positive_certificate.json')
    args=ap.parse_args()
    mp.mp.dps=args.dps+35
    I.dps=args.dps
    start=time.perf_counter()
    rows=[]
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for alpha in args.alpha:
        for n in args.panels:
            row=certificate(alpha,n,args.images,args.dps)
            rows.append(row)
            result=dict(status='CONDITIONAL_ENCLOSING_ARITHMETIC_CERTIFICATE',
                        scope='Periodic permanently-open-target endpoint traces and K only; no gate output read.',
                        arithmetic_contract='mpmath.iv encloses elementary, power, pi, and Gamma operations at the requested precision. Not an externally formally verified implementation.',
                        source_sha256=source_hash,rows=rows,seconds=time.perf_counter()-start)
            (HERE/args.out).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(row),flush=True)

if __name__=='__main__':
    main()
