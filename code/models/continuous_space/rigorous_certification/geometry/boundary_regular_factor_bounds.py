"""Explicit global boundary-regular-factor bounds for source remainders.

Uses only analytic barrier constants, not fitted Jacobi coefficients or gate data.
The derivative inverse bound is proved with a positive Jacobi Poisson semigroup
in GEOMETRY_CERTIFICATE_DERIVATION.zh-CN.md.
"""
from pathlib import Path
import hashlib,json
import mpmath as mp
from positive_neumann_geometry_certificate import I,exact,hi,endpoints,beta,upper_decimal

HERE=Path(__file__).resolve().parent

def row(alpha_text):
    s=exact(alpha_text)/2
    w=exact(1)/32
    L=1-w
    a=L/2
    D=(2*I.pi)**(-2*s)
    lam=I.gamma(1+2*s)
    lam1=I.gamma(2+2*s)
    c=2*s*2**(2*s-1)*I.gamma((1+2*s)/2)/(I.sqrt(I.pi)*I.gamma(1-s))
    mass=a**(2*s+1)*beta(exact('.5'),s+1)
    M0=c*(L**s*w**(-s)*beta(s+1,s)+mass*(3+1/s))
    q_upper=exact(hi(M0/lam))
    C=1/(D*lam*(1-q_upper))
    # Absolute derivative of W with respect to the physical x variable.
    M1=c*(1+2*s)*(L**s*w**(-s-1)*beta(s+1,s+1)
                         +mass*(3+2/(2*s+1)))
    forcing=1/(D*(1-q_upper))
    forcing_t_derivative=a*C*M1
    regular_factor=forcing/lam
    regular_factor_t_derivative=forcing_t_derivative/lam1
    B=L**(s-1)*(regular_factor+2*regular_factor_t_derivative)
    tau_upper=C*mass
    # The finite-rate agent's explicit two-end source decomposition.
    # A<=L^s*C is a simple barrier upper bound; using it keeps this file
    # independent of the first-correction rectangle computation.
    A_upper=L**s*C
    zeta_upper=1+1/(2*s)
    rho_bound=1+D*c*(2*B*L**(1-s)/(1-s)
                    +2*A_upper*L**(-s)/s+2*zeta_upper*tau_upper)
    fields=dict(weighted_contraction_upper=q_upper,comparison_coefficient_upper=C,
                Wv_sup_upper=M0,Wv_derivative_absolute_sup_upper=M1,
                forcing_sup_upper=forcing,forcing_centered_t_derivative_sup_upper=forcing_t_derivative,
                regular_factor_sup_upper=regular_factor,
                regular_factor_centered_t_derivative_sup_upper=regular_factor_t_derivative,
                B_left_global_difference_quotient_upper=B,
                B_right_global_difference_quotient_upper=B,
                tau0_upper=tau_upper,boundary_trace_simple_upper=A_upper,
                target_source_remainder_sup_upper=rho_bound)
    return dict(alpha_exact_decimal=alpha_text,width_exact_rational='1/32',
                coordinate_convention='U(y)=y^s*(L-y)^s*q(2*y/L-1), L=31/32',
                bounds={name:upper_decimal(value) for name,value in fields.items()})

def main():
    mp.mp.dps=80;I.dps=45
    result=dict(status='CONDITIONAL_ENCLOSING_ARITHMETIC_ANALYTIC_BOUNDS',
                scope='Global regular-factor and target-side source remainder constants; no finite-gamma error assertion by itself.',
                arithmetic_contract='mpmath.iv elementary/power/Gamma enclosures; analytic positive Jacobi Poisson semigroup lemma in companion derivation.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                rows=[row(a) for a in ('1.999','1.997')])
    (HERE/'boundary_regular_factor_bounds.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
