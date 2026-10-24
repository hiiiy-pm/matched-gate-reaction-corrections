"""Fixed chronological validation on all four experimental GlyR example records.

Observed, resolution-limited event sequences; no hidden-state kinetic fit.
Whole groups are held out, never concatenated to make history windows.
"""
from pathlib import Path
import json, csv, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'data/experimental'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'code/experimental/vendor'))
from scn_legacy_reader import read
from dcio.analysis.record import from_scn
from dcio.analysis.bursts import extract_burst_intervals

SPECS=[('A-10',10,.004),('B-30',30,1.),('C-100',100,.06),('D-1000',1000,.02)]

def get_bursts(name,dead_us):
    tc=next(s[2] for s in SPECS if s[0]==name)
    rec=read(ROOT/'data/experimental/raw/glyr'/(name+'.scn'))
    assert rec.header.record_type=='experimental'
    scr=from_scn(rec,tres=dead_us*1e-6)
    p=scr.periods
    assert np.all((p.amplitudes[::2]!=0)) and np.all(p.amplitudes[1::2]==0)
    bursts=extract_burst_intervals(p.intervals,p.amplitudes,tc,flags=p.flags)
    for b in bursts:
        assert len(b)%2==1 and np.all(np.isfinite(b)) and np.all(b>0)
        assert np.all(b[1::2]<=tc)
    return bursts

def encode(b,edges):
    return np.array([np.searchsorted(edges[1-i%2],d,side='right') for i,d in enumerate(b)],int)

def fit_counts(bursts,edges):
    counts={}
    for b in bursts:
        z=encode(b,edges)
        for i in range(3,len(z)):
            for k in range(4):
                key=(1-i%2,tuple(z[i-k:i]) if k else ())
                if key not in counts:counts[key]=np.zeros(3,float)
                counts[key][z[i]]+=1
    return counts

def score(bursts,edges,counts):
    scores=[]
    for b in bursts:
        z=encode(b,edges);row=[]
        for i in range(3,len(z)):
            p=np.ones(3)/3;values=[]
            for k in range(4):
                key=(1-i%2,tuple(z[i-k:i]) if k else ())
                n=counts.get(key,np.zeros(3));strength=1.5 if k==0 else 8.
                p=(n+strength*p)/(n.sum()+strength)
                assert np.all(p>0) and abs(p.sum()-1)<1e-12
                values.append(np.log2(p[z[i]]))
            row.append(values)
        scores.append(np.array(row).reshape(-1,4))
    return scores

def group_ci(scores,a,b,seed):
    ss=[x for x in scores if len(x)];n=np.array([len(x) for x in ss]);summed=np.array([(x[:,a]-x[:,b]).sum() for x in ss]);rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(ss),size=(2000,len(ss)))
    draws=summed[ix].sum(axis=1)/n[ix].sum(axis=1)
    return float(summed.sum()/n.sum()),np.quantile(draws,[.025,.975]).tolist()

def main():
    rows=[];groups=[];events=[];details=[]
    for j,(name,conc,tc) in enumerate(SPECS):
        for dead_us in [30,60,100]:
            bursts=get_bursts(name,dead_us);cut=int(np.floor(.6*len(bursts)));train,test=bursts[:cut],bursts[cut:]
            alltrain={s:np.concatenate([b[(1-s)::2] for b in train]) for s in [0,1]}
            edges={s:np.quantile(alltrain[s],[1/3,2/3]) for s in [0,1]}
            assert all(len(np.unique(v))==2 for v in edges.values())
            counts=fit_counts(train,edges);scores=score(test,edges,counts)
            row={'record':name,'concentration_uM':conc,'dead_time_us':dead_us,'tcrit_s':tc,'n_groups_total':len(bursts),'n_groups_train':cut,'n_groups_test':len(test),'n_groups_test_scored':sum(len(s)>0 for s in scores),'n_train_durations':sum(map(len,train)),'n_test_durations':sum(map(len,test)),'n_train_targets':sum(max(0,len(b)-3) for b in train),'n_test_targets':sum(map(len,scores)),'primary':dead_us==30}
            for label,a,b in [('gain_1v0',1,0),('gain_2v0',2,0),('gain_3v2',3,2)]:
                gain,ci=group_ci(scores,a,b,20261021+j)
                row[label]=gain;row[label+'_lo']=ci[0];row[label+'_hi']=ci[1]
            for k,(bt,sc) in enumerate(zip(test,scores)):
                if len(sc):groups.append({'record':name,'dead_time_us':dead_us,'group_index':cut+k,'n_targets':len(sc),'gain_2v0':float(np.mean(sc[:,2]-sc[:,0])),'gain_3v2':float(np.mean(sc[:,3]-sc[:,2]))})
            for s in [0,1]:
                td=np.concatenate([bt[(1-s)::2] for bt in test]);row[f'median_{s}_train_ms']=float(np.median(alltrain[s])*1e3);row[f'median_{s}_test_ms']=float(np.median(td)*1e3)
            rows.append(row)
            if dead_us==30:
                details.append({'record':name,'bin_edges_s':{str(s):e.tolist() for s,e in edges.items()},'n_train_groups':cut,'training_group_targets':[max(0,len(b)-3) for b in train],'test_group_targets':[len(s) for s in scores]})
                for k,bt in enumerate(bursts):
                    for i,d in enumerate(bt):events.append({'record':name,'group':k,'within_group_index':i,'split':'train' if k<cut else 'test','state':1-i%2,'duration_s':float(d)})
                np.savez(OUT/(name+'_glyr_primary.npz'),log_scores=np.concatenate(scores),group_index=np.concatenate([np.repeat(cut+k,len(s)) for k,s in enumerate(scores)]),bin_edges=np.array([edges[0],edges[1]]))
    for name,data in [('glyr_prediction_results',rows),('glyr_group_results',groups),('glyr_primary_events',events)]:
        keys=list(data[0])
        with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
    (OUT/'glyr_prediction_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (OUT/'glyr_primary_details.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
