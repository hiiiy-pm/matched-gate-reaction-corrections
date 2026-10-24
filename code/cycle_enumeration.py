"""Fresh exhaustive cycle-mask enumeration; exact filters precede float screening."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib,json,time,sys,itertools
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'code/models/reproducible'))
import exact_cycle_search as original
OUT=R/'results/search'
OUT.mkdir(parents=True,exist_ok=True)
def neighborhoods(x,n):
    bits=[(x>>i)&1 for i in range(n)]
    start=next(i for i in range(n) if bits[i]!=bits[(i-1)%n])
    runs=[];i=0
    while i<n:
        c=bits[(start+i)%n];m=0
        while i<n and bits[(start+i)%n]==c:
            i+=1;m+=1
        runs.append((c,m))
    return tuple(tuple(sorted(Counter((m,*sorted((runs[(j-1)%len(runs)][1],
             runs[(j+1)%len(runs)][1]))) for j,(c,m) in enumerate(runs) if c==state).items()))
             for state in [0,1])
def clock_signature(x,n):
    g=np.array([(x>>i)&1 for i in range(n)],dtype=object)
    v=g.copy();cols=[]
    for k in range(n):
        cols.append(v.copy());v=np.roll(v,1)+np.roll(v,-1)-2*v
    V=np.array(cols,dtype=object).T
    return tuple(int(y) for y in (V.T@(g[:,None]*V)).ravel())
allrows=[];summary=[];strong=[];t0=time.perf_counter()
for n in range(3,21):
    t=time.perf_counter();seen=bytearray(1<<n);groups=defaultdict(list);reps=0
    for x in range(1,(1<<n)-1):
        if seen[x]:continue
        reps+=1
        reverse=int(f'{x:0{n}b}'[::-1],2)
        for y in (x,reverse):
            for k in range(n):seen[original.rotate(y,n,k)]=1
        groups[original.signature(x,n)].append(x)
    collisions=[(sig,xs) for sig,xs in groups.items() if len(xs)>1]
    clock_count=0;triple_count=0;screen_count=0
    K=original.cycle_root(n)
    for idx,(signature,xs) in enumerate(collisions):
        gid=f'C{n}_{idx:04d}'
        clocks={x:clock_signature(x,n) for x in xs}
        neigh={x:neighborhoods(x,n) for x in xs}
        ell={x:original.brownian_ell(x,n,K) for x in xs}
        for x,y in itertools.combinations(xs,2):
            c=clocks[x]==clocks[y];d=neigh[x]==neigh[y]
            gap=ell[x]-ell[y]
            clock_count+=c;triple_count+=c and d
            screen_count+=c and d and abs(gap)>1e-10
            row={'group':gid,'n':n,'mask_int_A':x,'mask_int_B':y,
                 'A':[i for i in range(n) if (x>>i)&1],
                 'B':[i for i in range(n) if (y>>i)&1],
                 'clock_triple_exact':c,'run_neighborhood_match':d,
                 'brownian_A_float':ell[x],'brownian_B_float':ell[y],
                 'brownian_gap_float':gap,
                 'float_separation_screen':abs(gap)>1e-10}
            allrows.append(row)
            if c and d and abs(gap)>1e-10:strong.append(row)
    row={'n':n,'all_nonconstant_masks':(1<<n)-2,'dihedral_representatives':reps,
         'signature_classes':len(groups),'collision_classes':len(collisions),
         'collision_pairs':sum(len(xs)*(len(xs)-1)//2 for _,xs in collisions),
         'clock_triple_pairs':clock_count,'clock_and_neighborhood_pairs':triple_count,
         'joint_matched_float_separation_pairs':screen_count,'seconds':time.perf_counter()-t}
    summary.append(row);print(row,flush=True)
    (OUT/'search_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'all_screened_pairs.json').write_text(json.dumps(allrows,indent=2),encoding='utf-8')
    (OUT/'joint_matched_separated_candidates.json').write_text(json.dumps(strong,indent=2),encoding='utf-8')
wits=json.loads((R/'code/models/reproducible/witnesses.json').read_text())
check=[]
for w in wits:
    n=w['n'];xs=[original.canonical(sum(1<<i for i in w[c]),n) for c in ['A','B']]
    matches=[z for z in allrows if z['n']==n and {z['mask_int_A'],z['mask_int_B']}==set(xs)]
    assert len(matches)==1,(w['id'],xs,matches)
    z=matches[0]
    check.append({'archived_id':w['id'],'recomputed_id':z['group'],
                  'same_id':w['id']==z['group'],'clock_triple_exact':z['clock_triple_exact'],
                  'run_neighborhood_match':z['run_neighborhood_match']})
    assert z['clock_triple_exact']
record={'complete':True,'domain':'All nonconstant binary masks on homogeneous nearest-neighbor cycles, 3<=n<=20; rotations/reflections removed, complement retained.',
        'order':'Exact spatial-two-deck/run inventories, exact integer clock moments, exact run-neighborhood multisets, then float Brownian screening.',
        'triple_filter_scope':'Run-neighborhood equality is a sufficient complete-duration-triple criterion; failing this filter is not a proof of unequal triple laws.',
        'screen_tolerance':1e-10,'screen_is_certificate':False,
        'new_search_not_claimed_as_original_chronology':True,
        'witness_checks':check,'seconds':time.perf_counter()-t0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_search_source_sha256':hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()}
(OUT/'SEARCH_AUDIT.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2),flush=True)
