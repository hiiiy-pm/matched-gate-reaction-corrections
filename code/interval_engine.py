from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
from fractions import Fraction
from decimal import Decimal
from collections import defaultdict
import mpmath as mp
ROOT=Path(__file__).resolve().parent

def lo(x):return math.nextafter(float(x.a),-math.inf)
def hi(x):return math.nextafter(float(x.b),math.inf)
def upper_abs(x):return mp.iv.mpf(str(hi(abs(x))))
def bounds(x):return [lo(x),hi(x)]
def isum(xs):return sum(xs,mp.iv.mpf(0))
def dot(x,y):return isum(a*b for a,b in zip(x,y))
def runs(C,n):
 out=[];cs=set(C)
 for j in C:
  if (j-1)%n not in cs:
   r=[];k=j
   while k in cs:r.append(k);k=(k+1)%n
   out.append(r)
 assert sorted(j for r in out for j in r)==C
 return out
class Model:
 def __init__(self,R,n,dps=70):
  self.R=R;self.n=n;I=mp.iv;self.C=[j for j in range(n) if j not in R];nc=len(self.C);self.runs=runs(self.C,n)
  self.ring=[];self.lam=[];self.ra=[]
  for k in range(1,n//2+1):
   angle=Fraction(2*k,n)
   for kind in ([0] if 2*k==n else [0,1]):
    norm=1/I.sqrt(n) if 2*k==n else I.sqrt(I.mpf(2)/n);fn=I.cos if kind==0 else I.sin
    self.ring.append([norm*fn(2*I.pi*k*j/n) for j in range(n)]);self.ra.append(angle)
    self.lam.append(I.sqrt(1-I.cos(I.pi*angle.numerator/angle.denominator)))
  self.path=[];self.mu=[];self.pa=[]
  for block in self.runs:
   for k in range(1,len(block)+1):
    a=Fraction(k,len(block)+1);v=[I.mpf(0) for _ in range(n)]
    for pos,j in enumerate(block,1):v[j]=I.sqrt(I.mpf(2)/(len(block)+1))*I.sin(I.pi*k*pos/(len(block)+1))
    self.path.append(v);self.pa.append(a);self.mu.append(I.sqrt(1-I.cos(I.pi*a.numerator/a.denominator)))
  mat=[[isum(l*v[i]*v[j] for l,v in zip(self.lam,self.ring))+isum(l*v[i]*v[j] for l,v in zip(self.mu,self.path)) for j in self.C] for i in self.C]
  mid=mp.matrix([[(mp.mpf(x.a)+mp.mpf(x.b))/2 for x in row] for row in mat]);inv=mid**-1;xm=inv*mp.ones(nc,1)
  ri=[[I.mpf(mp.nstr(inv[i,j],dps)) for j in range(nc)] for i in range(nc)];xc=[I.mpf(mp.nstr(xm[i],dps)) for i in range(nc)]
  er=[[I.mpf(int(i==j))-isum(ri[i][k]*mat[k][j] for k in range(nc)) for j in range(nc)] for i in range(nc)]
  en=I.mpf(str(max(hi(isum(abs(x) for x in row)) for row in er)));assert hi(en)<1
  residual=[I.mpf(1)-dot(row,xc) for row in mat];pre=[dot(row,residual) for row in ri]
  pn=I.mpf(str(max(hi(abs(x)) for x in pre)));rad=hi(pn/(1-en))
  self.b=[I.mpf(0) for _ in range(n)]
  for j,x in zip(self.C,xc):self.b[j]=x+I.mpf([-rad,rad])
  self.beta=[dot(v,self.b) for v in self.ring];self.a=[dot(v,self.b) for v in self.path]
  self.cross=[[dot(u,v) for v in self.path] for u in self.ring]
  self.lam0=1-I.cos(I.pi/(max(map(len,self.runs))+1));self.q=I.mpf(nc)/n
  grp=defaultdict(list)
  for k,a in enumerate(self.ra):grp[('r',a)].append(k)
  for k,a in enumerate(self.pa):grp[('p',a)].append(k)
  U=I.mpf(0);V=I.mpf(0)
  for (typ,a),ks in grp.items():
   coeff=self.beta if typ=='r' else self.a;lam=self.lam if typ=='r' else self.mu
   amp=I.sqrt(isum(coeff[k]**2 for k in ks));U+=amp;V+=lam[ks[0]]*amp
  jump=[]
  for j in range(n):
   gp=-isum(l*b*v[j] for l,b,v in zip(self.lam,self.beta,self.ring));gm=isum(l*a*v[j] for l,a,v in zip(self.mu,self.a,self.path));jump.append(gp-gm)
  V+=I.sqrt(isum(x*x for x in jump));self.U=I.mpf(str(hi(U)));self.V=I.mpf(str(hi(V)))
  self.gate=(isum(b*b*l for b,l in zip(self.beta,self.lam))+isum(a*a*m for a,m in zip(self.a,self.mu)))/(2*n)
  self.brownian=isum(self.b[j] for j in self.C)/n
  self.meta={'closed_runs':self.runs,'neumann_operator_bound':hi(en),'coordinate_radius':rad,'U_grouped_bound':hi(self.U),'V_grouped_bound':hi(self.V),'brownian_enclosure':bounds(self.brownian)}
 def trial(self,s):
  I=mp.iv;rho=2*s-1
  diagonal=isum(b*b*l**rho for b,l in zip(self.beta,self.lam))+isum(a*a*m**rho for a,m in zip(self.a,self.mu));mixed=I.mpf(0)
  for i,l in enumerate(self.lam):
   for j,m in enumerate(self.mu):
    div=(1-rho)*l**(rho-2) if self.ra[i]==self.pa[j] else (l**(rho-1)-m**(rho-1))/(m-l)
    mixed+=l*m*self.beta[i]*self.a[j]*self.cross[i][j]*div
  kinetic=(diagonal-2*mixed)/(2*self.n*(-I.cos(I.pi*s)))
  source=I.gamma(1+s)*isum(isum(v)*a*m**(s-1) for v,a,m in zip(self.path,self.a,self.mu))/self.n
  return 2*source-kinetic-self.gate
 def residue_square(self,s,smin):
  I=mp.iv;delta=1-smin;ar=I.gamma(1+s)/I.gamma(2-s);Bs=1/(I.gamma(2*s)*(-I.cos(I.pi*s)))
  bu=I.mpf(str(hi(Bs)));lam=I.mpf(str(lo(self.lam0)))
  f=I.sqrt(self.q)*(upper_abs(ar-1)*(1/I.sqrt(lam)+I.sqrt(bu)/(smin+I.mpf('0.5')))+delta*I.mpf(str(hi(ar)))*(2*I.sqrt(bu)+1/I.sqrt(lam*(2*smin-1))))
  op=I.sqrt((self.U**2+self.V**2)/(self.n*I.pi)*8*delta**2/(1-4*delta**2))
  return hi((f+op)**2)
def certify(al,ah,A,B):
 I=mp.iv;s=I.mpf([str(al),str(ah)])/2;smin=I.mpf(str(al))/2;ja=A.trial(s);jb=B.trial(s)
 ra=A.residue_square(s,smin);rb=B.residue_square(s,smin);den=I.gamma(1+s)**2;d=(ja-jb+I.mpf([-rb,ra]))/den
 return {'alpha_lo':str(al),'alpha_hi':str(ah),'gap':bounds(d),'positive':bool(lo(d)>0),'negative':bool(hi(d)<0),'residual_A':ra,'residual_B':rb}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,required=True);ap.add_argument('--ra',required=True);ap.add_argument('--rb',required=True);ap.add_argument('--dps',type=int,default=70);ap.add_argument('--segments',type=int,default=400);ap.add_argument('--lower',default='1.998');ap.add_argument('--name',required=True);args=ap.parse_args()
 RA=[int(x) for x in args.ra.split(',') if x!='']; RB=[int(x) for x in args.rb.split(',') if x!='']
 mp.mp.dps=args.dps+20;mp.iv.dps=args.dps;st=time.perf_counter();A=Model(RA,args.n,dps=args.dps);B=Model(RB,args.n,dps=args.dps)
 al=Decimal(args.lower);step=(Decimal(2)-al)/args.segments;rows=[]
 for k in range(args.segments): rows.append(certify(al+k*step,al+(k+1)*step,A,B))
 pts=[certify(v,v,A,B) for v in ['2','1.999','1.998','1.997','1.995','1.99','1.9']]
 out={'name':args.name,'n':args.n,'RA':RA,'RB':RB,'dps':args.dps,'segments':args.segments,'lower':args.lower,'A':A.meta,'B':B.meta,'rows':rows,'stress_points':pts,'envelope':[min(r['gap'][0] for r in rows),max(r['gap'][1] for r in rows)],'all_positive':all(r['positive'] for r in rows),'all_negative':all(r['negative'] for r in rows),'seconds':time.perf_counter()-st,'scope':'local coefficient conditional on analytic inequalities; not full PDE proof or priority certificate'}
 fn=ROOT/'results'/f"{args.name}_interval_{args.dps}dps.json";fn.write_text(json.dumps(out,indent=2));print(json.dumps({'envelope':out['envelope'],'all_positive':out['all_positive'],'all_negative':out['all_negative'],'seconds':out['seconds']}))
if __name__=='__main__':main()
