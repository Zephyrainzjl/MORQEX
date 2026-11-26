"""Quantitative baseline evidence; execute under run_local.py after build."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.sparse import coo_matrix
from morqex import Workspace
from morqex.mechanics import TetMesh
from morqex.microstructure import Volume
from morqex.top_damage import NativeConfig, MFEMScalarBackend, AT2Damage
from morqex.top_damage.assembly_cache import CachedP1ScalarBackend


def analytic_form(mesh, a, D, f):
    a=np.broadcast_to(a,(len(mesh.cells),));D=np.broadcast_to(D,(len(mesh.cells),3,3));f=np.broadcast_to(f,(len(mesh.cells),))
    if not all(np.isfinite(v).all() for v in (a,D,f)) or np.any(a<0):raise ValueError('Invalid coefficients')
    scale=max(np.max(np.abs(D)),np.finfo(float).tiny)
    if np.max(np.abs(D-D.swapaxes(1,2)))>1e-13*scale or np.min(np.linalg.eigvalsh(D)) < -1e-13*scale:raise ValueError('Invalid tensor')
    local=mesh.volumes[:,None,None]*(a[:,None,None]*(np.ones((4,4))+np.eye(4))/20+np.einsum('eai,eij,ebj->eab',mesh.gradients,D,mesh.gradients))
    rows=np.broadcast_to(mesh.cells[:,:,None],local.shape);cols=np.broadcast_to(mesh.cells[:,None,:],local.shape)
    A=coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=(len(mesh.points),)*2).tocsr()
    b=np.zeros(len(mesh.points))
    for j in range(4):np.add.at(b,mesh.cells[:,j],mesh.volumes*f/4)
    return A,b


def main():
    p=argparse.ArgumentParser();p.add_argument('--provider',choices=['vendored','pymfem-wheel'],default='vendored');args=p.parse_args()
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);backend=MFEMScalarBackend(NativeConfig(provider=args.provider),ws)
    rng=np.random.default_rng(20261126)
    mesh=TetMesh.from_volume(Volume(np.ones((4,4,4),dtype=np.uint8),spacing=(.25,)*3))
    e=len(mesh.cells);Q=rng.normal(size=(e,3,3));D=Q.swapaxes(1,2)@Q+np.eye(3);a=rng.uniform(.1,4,e);f=rng.normal(size=e)
    A,b=backend.assemble(mesh,a,D,f);exact,rhs=analytic_form(mesh,a,D,f)
    result={'provider':args.provider,'backend_identity':backend.manifest['identity'],
            'assembly_relative_frobenius_error':float(np.linalg.norm((A-exact).toarray())/np.linalg.norm(exact.toarray())),
            'rhs_relative_error':float(np.linalg.norm(b-rhs)/np.linalg.norm(rhs)),
            'reference_scope':'P1 consistent mass/full tensor diffusion; mechanical specimen, not DFN lifetime'}
    model=AT2Damage(mesh,backend,2.,.2);solution=model.solve(3.,np.zeros(len(mesh.points)))
    uniform=6*(1-model.k)/(6*(1-model.k)+10)
    result['uniform_damage_max_abs_error']=float(np.max(np.abs(solution.damage-uniform)))
    result['uniform_projected_kkt']=solution.projected_kkt
    result['crack_profile_convergence']=[]
    for n in (8,16,32,64):
        m=TetMesh.from_volume(Volume(np.ones((2,2,n),dtype=np.uint8),spacing=(.1,.1,1/n)))
        pf=AT2Damage(m,backend,1.,.2);lower=np.zeros(len(m.points));lower[m.points[:,0]==0]=1
        s=pf.solve(0.,lower);exact_d=np.cosh((1-m.points[:,0])/.2)/np.cosh(5)
        exact_energy=.5*.2*.2*np.tanh(5)
        result['crack_profile_convergence'].append({'nx':n,'tetrahedra':len(m.cells),'h_over_ell':5/n,
             'relative_nodal_L2_error':float(np.linalg.norm(s.damage-exact_d)/np.linalg.norm(exact_d)),
             'fracture_energy_relative_error':float(abs(s.fracture_energy_J/exact_energy-1)),
             'kkt':s.projected_kkt})
    start=time.perf_counter();cached=CachedP1ScalarBackend(mesh,backend);setup=time.perf_counter()-start
    cache_A,cache_b=cached.assemble(mesh,a,D,f)
    result['cached_assembly_relative_error']=float(np.linalg.norm((cache_A-A).toarray())/np.linalg.norm(A.toarray()))
    timings={'cache_setup_seconds':setup,'cache_bytes':cached.cache_bytes}
    for name,fn in [('mfem',backend.assemble),('numpy_analytic',analytic_form),('cached',cached.assemble)]:
        fn(mesh,a,D,f);samples=[]
        for _ in range(15):
            start=time.perf_counter();fn(mesh,a,D,f);samples.append(time.perf_counter()-start)
        timings[name+'_median_seconds']=float(np.median(samples))
    timings['numpy_over_mfem_ratio']=timings['numpy_analytic_median_seconds']/timings['mfem_median_seconds']
    timings['mfem_over_cache_ratio']=timings['mfem_median_seconds']/timings['cached_median_seconds']
    saving=timings['mfem_median_seconds']-timings['cached_median_seconds']
    timings['cache_break_even_assemblies']=setup/saving if saving>0 else None
    timings['scope']='Same 384 Tet4 cells, cellwise full SPD tensor, consistent mass, CSR A and b. Includes validation/conversion/mesh construction; assembly only, not a simulation speedup.'
    result['assembly_timing']=timings
    errors=[r['relative_nodal_L2_error'] for r in result['crack_profile_convergence']]
    result['passed']=bool(result['assembly_relative_frobenius_error']<1e-12 and result['cached_assembly_relative_error']<1e-12 and result['rhs_relative_error']<1e-12 and result['uniform_damage_max_abs_error']<1e-10 and all(y<x for x,y in zip(errors,errors[1:])) and errors[-1]<.01)
    destination=ws.path('reports',f'top_damage_reference_{args.provider}.json');destination.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Reference mathematical gates failed')


if __name__=='__main__':main()
