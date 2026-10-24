"""Chronological held-out prediction of observed current-level durations.

No hidden-state kinetic fit; no use of test data to select thresholds or bins.
"""
from pathlib import Path
import json, csv, sys, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'data/experimental'
OUT.mkdir(parents=True,exist_ok=True)

def runs(y):
    edges=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1,len(y)]
    return edges[:-1],edges[1:],y[edges[:-1]].astype(int)

def extract(x,threshold,min_samples,fs):
    y=x>threshold
    # Eliminate unresolved interior excursions symmetrically for both levels.
    merged=0
    while True:
        a,b,s=runs(y); lens=b-a
        eligible=np.flatnonzero((lens<min_samples)&(np.arange(len(a))>0)&(np.arange(len(a))<len(a)-1))
        if not len(eligible):break
        j=eligible[np.argmin(lens[eligible])]
        y[a[j]:b[j]]=s[j-1]; merged+=1
    a,b,s=runs(y)
    # First and last observed runs are censored. Never bridge train/test split.
    return {'duration':(b-a)[1:-1]/fs,'state':s[1:-1],'start':a[1:-1]/fs,'end':b[1:-1]/fs,'merged_short_runs':merged,'occupancy':float(y.mean())}

def encode(d,edges):
    return np.array([np.searchsorted(edges[int(s)],t,side='right') for t,s in zip(d['duration'],d['state'])],int)

def fit_predict(train,test,edges):
    tr=encode(train,edges); te=encode(test,edges); st=train['state']; ss=test['state']
    # All orders fit exactly the same training target events.
    counts={}
    for i in range(3,len(tr)):
        for k in range(4):
            key=(int(st[i]),tuple(tr[i-k:i]) if k else ())
            if key not in counts:counts[key]=np.zeros(3,float)
            counts[key][tr[i]]+=1
    scores=[]
    for i in range(3,len(te)):
        qs=[]
        p=np.ones(3)/3
        for k in range(4):
            key=(int(ss[i]),tuple(te[i-k:i]) if k else ())
            n=counts.get(key,np.zeros(3))
            strength=1.5 if k==0 else 8.0
            p=(n+strength*p)/(n.sum()+strength)
            qs.append(np.log2(p[te[i]]))
        scores.append(qs)
    return np.array(scores).reshape(-1,4)

def interval(x,seed):
    if len(x)<2:return [None,None]
    rng=np.random.default_rng(seed); n=len(x); block=max(2,int(np.ceil(np.sqrt(n))))
    starts=rng.integers(0,n,size=(2000,int(np.ceil(n/block))))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n]
    return np.quantile(x[idx].mean(axis=1),[.025,.975]).tolist()

def main():
    rows=[]; detail=[]
    for j,p in enumerate(sorted(OUT.glob('ic*.npy'))):
        x=np.load(p); fs=75000 if '25kHz' in p.name else 150000; cut=int(.6*len(x)); xx=x[:cut]
        lo=float(np.median(xx[xx<=np.median(xx)])); hi=float(np.median(xx[xx>=np.quantile(xx,.98)]))
        for fraction in [.35,.5,.65]:
            for dead_us in [40,80,160]:
                threshold=lo+fraction*(hi-lo); nmin=int(np.ceil(dead_us*1e-6*fs-1e-10))
                tr=extract(x[:cut],threshold,nmin,fs); te=extract(x[cut:],threshold,nmin,fs)
                row={'record':p.stem,'bandwidth_kHz':fs/3000,'duration_s':len(x)/fs,'threshold_fraction':fraction,'threshold_pA':threshold,'dead_time_us':dead_us,'n_train':len(tr['duration']),'n_test':len(te['duration']),'occupancy_train':tr['occupancy'],'occupancy_test':te['occupancy'],'merged_train':tr['merged_short_runs'],'merged_test':te['merged_short_runs']}
                state_counts=[int(np.sum(tr['state']==s)) for s in [0,1]]
                if min(state_counts)>=3 and len(te['duration'])>=5:
                    edges={s:np.quantile(tr['duration'][tr['state']==s],[1/3,2/3]) for s in [0,1]}
                    row['bins_unique']=all(len(np.unique(e))==2 for e in edges.values())
                    sc=fit_predict(tr,te,edges)
                    row['eligible']=len(tr['duration'])>=60 and len(te['duration'])>=30 and row['bins_unique']
                    for name,a,b in [('gain_1v0',1,0),('gain_2v0',2,0),('gain_3v2',3,2)]:
                        delta=sc[:,a]-sc[:,b]; ci=interval(delta,20261021+j)
                        row[name]=float(delta.mean()); row[name+'_lo']=ci[0];row[name+'_hi']=ci[1]
                    if fraction==.5 and dead_us==80:
                        np.savez(OUT/(p.stem+'_primary.npz'),train_duration=tr['duration'],train_state=tr['state'],test_duration=te['duration'],test_state=te['state'],test_start=te['start'],test_end=te['end'],log_scores=sc,bin_edges=np.array([edges[0],edges[1]]))
                        detail.append({'id':p.stem,'low_level':lo,'high_level':hi,'threshold':threshold,'train_bin_edges_s':{str(s):e.tolist() for s,e in edges.items()}})
                else: row['eligible']=False
                rows.append(row)
    (OUT/'prediction_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (OUT/'primary_details.json').write_text(json.dumps(detail,indent=2),encoding='utf-8')
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (OUT/'prediction_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    primary=[r for r in rows if r['threshold_fraction']==.5 and r['dead_time_us']==80]
    print(json.dumps(primary,indent=2))

if __name__=='__main__':main()
