"""Ten-minute exact check of Proposition 1. Python + SymPy; no archived outputs.
Integer derivative tensors through Cayley--Hamilton degree 8 certify the
complete one-, two- and three-duration laws, at both stationary entrances.
The four-duration contrast is checked at equal and unequal rational arguments.
"""
import itertools,time
import sympy as s
t0=time.perf_counter(); n=18
Q=s.zeros(n)
for i in range(n):
    Q[i,i]=-1;Q[i,(i-1)%n]=Q[i,(i+1)%n]=s.Rational(1,2)
masks=[[0,1,4,5,6,8,11,12,14],[0,1,4,6,7,10,11,12,14]]
def blocks(mask,color):
    a=sorted(mask if color else set(range(n))-set(mask));b=sorted(set(range(n))-set(a))
    return [Q.extract(a,a),Q.extract(b,b)],[Q.extract(a,b),Q.extract(b,a)]
def tensors(mask,color):
    A,B=blocks(mask,color);A=[2*x for x in A];B=[2*x for x in B];o=s.ones(9,1)
    nu=o.T*B[1];assert sum(nu)==10
    powers=[[a**j for j in range(9)] for a in A];out=[]
    for k in (1,2,3):
        left=[nu*p for p in powers[0]]
        for j in range(1,k):left=[v*B[(j-1)%2]*p for v in left for p in powers[j%2]]
        out.append(tuple((v*B[(k-1)%2]*o)[0] for v in left))
    return out
def J(mask,color,z):
    A,B=blocks(mask,color);o=s.ones(9,1);nu=o.T*B[1];nu/=sum(nu)
    for j,v in enumerate(z):nu=nu*(v*s.eye(9)-A[j%2]).inv()*B[j%2]
    return (nu*o)[0]
for color in (0,1):
    assert [a.charpoly().as_expr() for a in blocks(masks[0],color)[0]]==[a.charpoly().as_expr() for a in blocks(masks[1],color)[0]]
    assert tensors(masks[0],color)==tensors(masks[1],color)
    delta=J(masks[0],color,[1]*4)-J(masks[1],color,[1]*4)
    assert delta==s.Rational(11,1764000)
    z=[1,2,3,s.Rational(1,2)];mixed=J(masks[0],color,z)-J(masks[1],color,z)
    assert mixed<0
    print('Entrance',color,': complete k=1,2,3 identities PASS; D4(1,1,1,1)=',delta,'; D4(1,2,3,1/2)=',mixed)
print('ALL PASS in %.2f seconds' % (time.perf_counter()-t0))
