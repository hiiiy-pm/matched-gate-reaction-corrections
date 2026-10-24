"""Exact-signature enumeration for homogeneous symmetric cycle gates.

Not random sampling. Dihedral relabeling, NOT open/closed complementation,

is removed. Brownian coefficient screening is floating point only.

"""

from __future__ import annotations

import argparse,csv,json,time

from pathlib import Path

from collections import defaultdict

import numpy as np

from scipy import linalg

P=Path(__file__).resolve().parents[1]

def rotate(x,n,k=1):

 k%=n

 return ((x<<k)|(x>>(n-k)))&((1<<n)-1) if k else x

def canonical(x,n):

 rev=int(f'{x:0{n}b}'[::-1],2)

 return min(min(rotate(x,n,k),rotate(rev,n,k)) for k in range(n))

def runs(x,n):

 bits=[(x>>j)&1 for j in range(n)]

 start=next(j for j in range(n) if bits[j]!=bits[(j-1)%n])

 rr=[[],[]];cur=bits[start];length=0

 for k in range(n):

  b=bits[(start+k)%n]

  if b==cur:length+=1

  else:rr[cur].append(length);cur=b;length=1

 rr[cur].append(length)

 return tuple(sorted(rr[1])),tuple(sorted(rr[0]))

def signature(x,n):

 return (tuple((x&rotate(x,n,k)).bit_count() for k in range(n)),)+runs(x,n)

def cycle_Q(n):

 q=-np.eye(n)

 for j in range(n):q[j,(j-1)%n]=q[j,(j+1)%n]=.5

 return q

def cycle_root(n):

 rates=1-np.cos(2*np.pi*np.arange(n)/n);rates[0]=0.

 return linalg.circulant(np.fft.ifft(np.sqrt(rates)).real)

def brownian_ell(x,n,K=None):

 if K is None:K=cycle_root(n)

 C=[j for j in range(n) if not((x>>j)&1)]

 H=-cycle_Q(n)[np.ix_(C,C)];d,v=linalg.eigh(H)

 A=K[np.ix_(C,C)]+(v*np.sqrt(d))@v.T

 return float(linalg.solve(A,np.ones(len(C)),assume_a='pos').sum()/n)

def write_csv(path,rows):

 if not rows:return

 with path.open('w',newline='') as f:

  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(max_n):

 sums=[];all_groups=[];response=[];started=time.perf_counter()

 for n in range(3,max_n+1):

  t=time.perf_counter();groups=defaultdict(list);unique=0

  for x in range(1,(1<<n)-1):

   if canonical(x,n)!=x:continue

   unique+=1;groups[signature(x,n)].append(x)

  collision=[(sig,v) for sig,v in groups.items() if len(v)>1]

  K=cycle_root(n);sep=0;npairs=0;maxgap=0.

  for local,(sig,masks) in enumerate(collision):

   gid=f'C{n}_{local:04d}';vals=[brownian_ell(x,n,K) for x in masks]

   gap=max(vals)-min(vals);sep+=gap>1e-10;maxgap=max(maxgap,gap);npairs+=len(masks)*(len(masks)-1)//2

   all_groups.append({'group_id':gid,'n':n,'masks':masks,'open_sets':[[j for j in range(n) if (x>>j)&1] for x in masks],

                      'signature':sig,'brownian_coefficients':vals,'range_float':gap})

   for x,ell in zip(masks,vals):

    response.append({'group_id':gid,'n':n,'mask_int':x,'open_set':','.join(str(j) for j in range(n) if (x>>j)&1),

                     'p':x.bit_count()/n,'ell2_float':ell,'group_range_float':gap,

                     'on_runs':','.join(map(str,sig[1])),'off_runs':','.join(map(str,sig[2]))})

  row={'n':n,'all_nonconstant_masks':(1<<n)-2,'dihedral_representatives':unique,'summary_classes':len(groups),

       'nontrivial_summary_classes':len(collision),'nontrivial_pairs':npairs,

       'classes_with_ell2_range_gt_1e-10':sep,'max_ell2_range_float':maxgap,'seconds':time.perf_counter()-t}

  sums.append(row);print(json.dumps(row),flush=True)

  write_csv(P/'results/search_summary.csv',sums)

  (P/'results/matching_classes.json').write_text(json.dumps(all_groups,indent=2))

 write_csv(P/'results/brownian_screen.csv',response)

 summary={'class_restriction':'homogeneous nearest-neighbor symmetric cycles; all nonconstant binary masks; dihedral quotient only',

          'max_n':max_n,'raw_masks_total':sum(r['all_nonconstant_masks'] for r in sums),

          'dihedral_representatives_total':sum(r['dihedral_representatives'] for r in sums),

          'nontrivial_summary_classes':len(all_groups),'first_n_with_collision':next((r['n'] for r in sums if r['nontrivial_summary_classes']),None),

          'max_gap_record':max(all_groups,key=lambda g:g['range_float']) if all_groups else None,

          'wall_seconds':time.perf_counter()-started,'screen_tolerance':1e-10,

          'limitations':['No general CTMC minimum-state conclusion.','Float screen is not a signed certificate.',

                         'Counts imply neither prevalence nor publication originality.']}

 (P/'results/search_overview.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':

 ap=argparse.ArgumentParser();ap.add_argument('--max-n',type=int,default=18);a=ap.parse_args();main(a.max_n)

