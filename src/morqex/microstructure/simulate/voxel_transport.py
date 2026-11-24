"""Published TauFactor GPU solver under the common GEM volume contract."""
import numpy as np

def effective_diffusion(volume,phase=0,axis='z',device='cuda',tolerance=1e-5,max_iterations=50000):
    import taufactor
    if axis not in ('x','y','z'):raise ValueError('axis=x/y/z')
    if not np.allclose(volume.spacing,volume.spacing[0],rtol=1e-10,atol=0):
        raise ValueError('TauFactor adapter requires isotropic spacing; use FEM for anisotropic voxels')
    mask=np.moveaxis(volume.data==phase,{'z':0,'y':1,'x':2}[axis],0).copy()
    from scipy.ndimage import label
    ids,_=label(mask)
    spanning=(set(np.unique(ids[0]))&set(np.unique(ids[-1])))-{0}
    if not spanning:return {'effective_diffusivity':0.,'tortuosity':float('inf'),'percolating':False}
    solver=taufactor.Solver(mask,device=device)
    solver.solve(iter_limit=max_iterations,verbose=False,conv_crit=tolerance)
    def scalar(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        value=np.asarray(value)
        if value.size!=1:raise RuntimeError('Expected a single-volume TauFactor result')
        return float(value.reshape(-1)[0])
    effective=scalar(solver.D_eff)
    tau=scalar(solver.tau)
    if not np.isfinite(effective) or effective<0 or not np.isfinite(tau):
        raise RuntimeError('TauFactor returned a nonfinite transport result')
    return {'effective_diffusivity':effective,'tortuosity':tau,'percolating':True,
            'backend':'TauFactor','convergence_criterion':tolerance,'iteration_limit':max_iterations}
