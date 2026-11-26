"""Bounded finite-strain manufactured-solution refinement and solver comparison.

The exact displacement and independent invariant stress generate the body load;
no MFEM tangent or discrete residual is used to manufacture the forcing.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from morqex.workspace import Workspace
from morqex.top_damage.backend import NativeConfig, require_local_budget
from morqex.top_damage.hyperelastic import HyperelasticBackend,Hyperelasticity
from morqex.mechanics.mesh import TetMesh
from morqex.microstructure import Volume


def exact(x):
    s=np.sin(np.pi*x);c=np.cos(np.pi*x)
    u=np.zeros(x.shape,dtype=x.dtype);u[...,0]=.04*np.prod(s,axis=-1)
    F=np.broadcast_to(np.eye(3),x.shape[:-1]+(3,3)).astype(x.dtype).copy()
    for j in range(3):F[...,0,j]+=.04*np.pi*c[...,j]*np.prod(np.delete(s,j,axis=-1),axis=-1)
    return u,F


def invariant_piola(x,mu,K):
    _,F=exact(x);J=np.linalg.det(F);I1=np.sum(F*F,axis=(-1,-2));it=np.linalg.inv(F).swapaxes(-1,-2)
    return mu*J[...,None,None]**(-2/3)*(F-I1[...,None,None]/3*it)+K*(J*(J-1))[...,None,None]*it


def forcing(x,mu,K):
    result=np.zeros_like(x)
    for j in range(3):
        perturbed=x.astype(complex);perturbed[...,j]+=1e-30j
        result-=np.imag(invariant_piola(perturbed,mu,K)[..., :,j])/1e-30
    return result


def run(n,backend,solver):
    tick=time.perf_counter();mesh=TetMesh.from_volume(Volume(np.ones((n,n,n),np.uint8),spacing=(1/n,)*3))
    m=Hyperelasticity(mesh,backend,100.,.3,linear_solver=solver)
    setup=time.perf_counter()-tick
    aa=(5+3*np.sqrt(5))/20;bb=(5-np.sqrt(5))/20
    bary=np.full((4,4),bb);np.fill_diagonal(bary,aa)
    x=np.einsum('qa,eaj->eqj',bary,mesh.points[mesh.cells])
    body=forcing(x,m.shear[0],m.bulk[0])
    local=np.einsum('qa,eqi,e->eai',bary,body,mesh.volumes)/4
    force=np.zeros_like(mesh.points);np.add.at(force,mesh.cells.ravel(),local.reshape(-1,3))
    tick=time.perf_counter();state=m.solve(force=force,tolerance=1e-9);solve=time.perf_counter()-tick
    uh=np.einsum('qa,eaj->eqj',bary,state.displacement_m[mesh.cells]);ue,Fe=exact(x)
    l2=np.sqrt(np.einsum('eqi,eqi,e->',uh-ue,uh-ue,mesh.volumes)/4)
    h1=np.sqrt(np.einsum('eqij,eqij,e->',state.deformation_gradient[:,None]-Fe,state.deformation_gradient[:,None]-Fe,mesh.volumes)/4)
    return {'n':n,'cells':len(mesh.cells),'dofs':3*len(mesh.points),'solver':solver,
            'setup_s':setup,'solve_s':solve,'l2_error':float(l2),'h1_seminorm_error':float(h1),
            'equilibrium_residual':state.equilibrium_residual,'minimum_jacobian':state.minimum_jacobian,
            'newton_iterations':state.newton_iterations,'linear_iterations':m.linear_iterations,
            'energy_J':state.elastic_energy_J},state


def main():
    p=argparse.ArgumentParser();p.add_argument('--levels',nargs='+',type=int,default=[2,4,8])
    p.add_argument('--output',default='finite_B0');args=p.parse_args()
    require_local_budget();ws=Workspace(os.environ['MORQEX_WORKSPACE']);backend=HyperelasticBackend(NativeConfig(),ws)
    directory=ws.path('results/top_damage',args.output);directory.mkdir(parents=True,exist_ok=True)
    results=[]
    for n in args.levels:
        record,state=run(n,backend,'direct');results.append(record)
        print(json.dumps(record),flush=True)
    fast,other=run(args.levels[-1],backend,'matrix-free-amg-minres')
    difference=float(np.linalg.norm(other.displacement_m-state.displacement_m)/np.linalg.norm(state.displacement_m))
    results.append(fast)
    orders=[]
    for a,b in zip(results[:len(args.levels)-1],results[1:len(args.levels)]):
        orders.append({'h_ratio':b['n']/a['n'],'l2':float(np.log(a['l2_error']/b['l2_error'])/np.log(b['n']/a['n'])),
                       'h1':float(np.log(a['h1_seminorm_error']/b['h1_seminorm_error'])/np.log(b['n']/a['n']))})
    passed=all(o['l2']>1.5 and o['h1']>.75 for o in orders) and difference<1e-7
    result={'schema':'finite-B0-1','scope':'Manufactured isotropic chemical-elastic reference, not material/lifetime validation',
            'backend':backend.manifest,'rows':results,'orders':orders,'solver_displacement_difference':difference,
            'direct_to_matrix_free_total_cost_ratio':(results[-2]['setup_s']+results[-2]['solve_s'])/(fast['setup_s']+fast['solve_s']),
            'passed':passed}
    (directory/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':passed,'output':str(directory),'orders':orders,'solver_difference':difference}),flush=True)
    if not passed:raise SystemExit('Finite-strain refinement gate failed')


if __name__=='__main__':main()
