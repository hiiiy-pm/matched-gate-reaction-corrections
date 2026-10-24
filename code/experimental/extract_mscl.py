from pathlib import Path
import pyreadr,numpy as np
R=Path(__file__).resolve().parents[2]
for p in (R/'data/experimental/raw/mscl/data').glob('*.rda'):
    frame=next(iter(pyreadr.read_r(str(p)).values()))
    np.save(R/'data/experimental'/ (p.stem+'.npy'),frame.iloc[:,0].to_numpy(float))
