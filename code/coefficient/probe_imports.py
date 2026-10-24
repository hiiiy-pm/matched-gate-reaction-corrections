import os,sys
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
WORK=ROOT;V8=ROOT
sys.path.insert(0,str(ROOT))
MASKS={'A':[0,1,4,5,6,8,11,12,14], 'B':[0,1,4,6,7,10,11,12,14]}
