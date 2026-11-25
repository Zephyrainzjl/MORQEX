"""Independent periodic scalar cell comparisons and analytic laminate limits."""
import json
import os
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from morqex.acceleration.transport import PeriodicTransport

def reference(phase,coeff,axis):
    ids=np.arange(phase.size).reshape(phase.shape);rows=[];cols=[];data=[];b=np.zeros(phase.size);face=[]
    values=np.asarray(coeff)[phase]
    for a in range(3):
        other=np.roll(ids,-1,2-a);v=np.roll(values,-1,2-a);g=np.divide(2*values*v,values+v,out=np.zeros_like(values,dtype=float),where=values+v>0)
        i=ids.ravel();j=other.ravel();gg=g.ravel();rows.extend([i,i,j,j]);cols.extend([i,j,i,j]);data.extend([gg,-gg,-gg,gg])
        if a==axis:np.add.at(b,i,gg);np.add.at(b,j,-gg)
        face.append((j,gg))
    K=sparse.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(phase.size,)*2).tocsr();u,info=cg(K,b,rtol=1e-11,maxiter=10000)
    assert info==0
    column=[float(np.mean(g*((1. if a==axis else 0.)+u[j]-u))) for a,(j,g) in enumerate(face)]
    return u.reshape(phase.shape),column

def main():
    rng=np.random.default_rng(1);phase=rng.integers(0,3,(8,8,8),dtype=np.uint8);coeff=[0.,1.,.2]
    u,expected=reference(phase,coeff,0);model=PeriodicTransport(phase,coeff);actual,metrics=model.solve(rtol=1e-9);cp=model.xp
    np.testing.assert_allclose(metrics['effective_column'],expected,rtol=1e-7,atol=1e-9)
    phase=np.ones((6,6,8),np.uint8);phase[:,:,:4]=2
    longitudinal=PeriodicTransport(phase,[0.,1.,.2]);_,a=longitudinal.solve(0,rtol=1e-9)
    assert abs(a['effective_column'][0]-1/(.5/1.+.5/.2))<1e-9
    _,b=longitudinal.solve(1,rtol=1e-9);assert abs(b['effective_column'][1]-.6)<1e-9
    result={'random_reference':metrics,'laminate_series':a,'laminate_parallel':b,'scope':'same periodic harmonic TPFA, independent sparse SciPy assembly and analytic limits'}
    (Path(os.environ['MORQEX_WORKSPACE'])/'reports/voxel_transport_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
