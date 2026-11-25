"""Compile CPU/CUDA stencils and compare actual fields/energy on the same grid."""
import json
import os
from pathlib import Path
import time
import numpy as np
from morqex.acceleration.voxel import NativeStencil
from morqex.mechanics import isotropic_stiffness

def main():
    os.environ['MORQEX_CUDA_DEBUG']='1'
    rng=np.random.default_rng(104);shape=(8,8,8);phase=rng.integers(0,3,shape,dtype=np.uint8)
    # A weak phase is explicit in this first compile probe; true pores are a
    # separate test, not silently regularized by the implementation.
    C=np.stack([isotropic_stiffness(e,.25) for e in [1.,100.,10.]])
    beta=np.zeros((3,6));beta[1,:3]=1e-3;alpha=np.zeros_like(beta);alpha[:,:3]=1e-5
    cpu=NativeStencil(phase,C,beta,alpha);gpu=NativeStencil(phase,C,beta,alpha,backend='cuda');cp=gpu.xp
    u=rng.standard_normal((3,*shape));v=rng.standard_normal(u.shape);c=rng.uniform(.1,1.,shape)
    Au=cpu.apply(u);Av=cpu.apply(v)
    assert abs(np.vdot(v,Au)-np.vdot(u,Av))/max(1.,abs(np.vdot(u,Av)))<1e-12
    assert np.vdot(u,Au)>0
    with gpu.budget.batch():gu=cp.asarray(u);actual=gpu.apply(gu);difference=float(cp.max(abs(actual-cp.asarray(Au))))
    assert difference<1e-12
    x,a=cpu.solve(c,rtol=1e-8,maxiter=1500)
    gc=cp.asarray(c);gx,b=gpu.solve(gc,rtol=1e-8,maxiter=1500)
    with gpu.budget.batch():gs=gpu.stress(gx,gc);stress_error=float(cp.linalg.norm(gs-cp.asarray(cpu.stress(x,c)))/cp.linalg.norm(gs))
    assert stress_error<1e-6
    gx2,refined=gpu.solve(gc,rtol=1e-8,maxiter=1500,low_memory=True)
    with gpu.budget.batch():rs=gpu.stress(gx2,gc);refined_error=float(cp.linalg.norm(rs-cp.asarray(cpu.stress(x,c)))/cp.linalg.norm(rs))
    assert refined_error<1e-6
    checkpoint=Path.cwd()/'krylov_probe'
    _,short=gpu.solve(gc,rtol=1e-8,maxiter=5,low_memory=True,checkpoint=checkpoint)
    assert not short['converged']
    continued,restarted=gpu.solve(gc,rtol=1e-8,maxiter=1500,low_memory=True,checkpoint=checkpoint)
    assert restarted['iterations_total']==short['iterations_total']+restarted['iterations']
    np.testing.assert_allclose(cp.asnumpy(continued),cp.asnumpy(gx2),rtol=1e-4,atol=1e-9)
    gpu.preconditioner_kind='coarse'
    gx3,coarse=gpu.solve(gc,rtol=1e-8,maxiter=1500,low_memory=True)
    with gpu.budget.batch():
        rs=gpu.stress(gx3,gc);coarse_error=float(cp.linalg.norm(rs-cp.asarray(cpu.stress(x,c)))/cp.linalg.norm(rs))
    assert coarse_error<1e-6
    gpu.preconditioner_kind='phase-coarse'
    gx4,phase_coarse=gpu.solve(gc,rtol=1e-8,maxiter=1500,low_memory=True)
    with gpu.budget.batch():
        rs=gpu.stress(gx4,gc);phase_error=float(cp.linalg.norm(rs-cp.asarray(cpu.stress(x,c)))/cp.linalg.norm(rs))
    assert phase_error<1e-6
    from morqex.acceleration.geometric import GeometricMultigrid
    mg=GeometricMultigrid(gpu)
    with gpu.budget.batch():
        r1=cp.asarray(u);r2=cp.asarray(v);m1=cp.empty_like(r1,dtype='float32');m2=cp.empty_like(r2,dtype='float32');mg.apply(r1,m1);mg.apply(r2,m2)
        pair1=float(cp.vdot(r1,m2));pair2=float(cp.vdot(r2,m1))
        assert abs(pair1-pair2)/max(abs(pair1),abs(pair2),1.)<2e-5
        assert float(cp.vdot(r1,m1))>0
    del mg,r1,r2,m1,m2
    gpu.preconditioner_kind='geometric'
    gx5,geometric=gpu.solve(gc,rtol=1e-8,maxiter=1500,low_memory=True)
    with gpu.budget.batch():
        rs=gpu.stress(gx5,gc);geometric_error=float(cp.linalg.norm(rs-cp.asarray(cpu.stress(x,c)))/cp.linalg.norm(rs))
    assert geometric_error<1e-6
    # Odd dimensions and SI spacings exercise a different hierarchy and the
    # NumPy/CuPy scalar promotion that a unit-spacing power-of-two grid misses.
    odd_phase=rng.integers(0,3,(9,7,5),dtype=np.uint8);odd_C=C.copy();odd_C[0]=0
    odd_cpu=NativeStencil(odd_phase,odd_C,beta,alpha,spacing=(2e-7,3e-7,4e-7))
    odd_gpu=NativeStencil(odd_phase,odd_C,beta,alpha,spacing=(2e-7,3e-7,4e-7),backend='cuda',budget=gpu.budget,preconditioner='geometric')
    odd_c=rng.uniform(.1,1.,odd_phase.shape);odd_x,odd_ref=odd_cpu.solve(odd_c,rtol=1e-8,maxiter=1500)
    odd_gx,odd_metrics=odd_gpu.solve(cp.asarray(odd_c),rtol=1e-8,maxiter=1500,low_memory=True)
    with gpu.budget.batch():
        odd_s=odd_gpu.stress(odd_gx,cp.asarray(odd_c));odd_error=float(cp.linalg.norm(odd_s-cp.asarray(odd_cpu.stress(odd_x,odd_c)))/cp.linalg.norm(odd_s))
    assert odd_error<1e-5
    from morqex.visualization.voxel_fields import export_reference_fields
    fields=export_reference_fields(gpu,gx4,gc,Path.cwd()/'gpu_fields',chunk=77)
    np.testing.assert_allclose(np.load(fields['files']['stress_Pa']['path']),cp.asnumpy(gpu.stress(gx4,gc)),atol=1e-12)
    assert fields['relative_stationarity_energy_defect']<1e-5
    out={'cpu':a,'cuda':b,'refined':refined,'coarse':coarse,'phase_coarse':phase_coarse,'geometric':geometric,'odd_true_pore_geometric':odd_metrics,'odd_true_pore_stress_relative_L2':odd_error,'geometric_stress_relative_L2':geometric_error,'phase_coarse_stress_relative_L2':phase_error,'coarse_stress_relative_L2':coarse_error,'krylov_continuation':restarted,'refined_stress_relative_L2':refined_error,'matvec_maximum_difference':difference,'stress_relative_L2':stress_error,'streamed_fields':fields,'scope':'same compatible FD discretization; 8^3 declared weak phase and 9x7x5 true pores/SI spacing; FP64 CPU/CUDA and mixed preconditioning accepted by FP64 residual'}
    w=Path(os.environ['MORQEX_WORKSPACE']);(w/'reports/voxel_backend_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
