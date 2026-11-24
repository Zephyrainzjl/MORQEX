"""Directional RVE homogenization -> explicitly mapped continuum battery properties."""
from morqex.microstructure.mesh import voxel_to_tetra
from .transport import conduction

def homogenize(volume,coefficients,order=2):
    mesh = voxel_to_tetra(volume)
    results = {axis:conduction(mesh,coefficients,axis=axis,order=order) for axis in 'xyz'}
    return {'conductivity_xyz':[results[a].effective_conductivity for a in 'xyz'],
            'residual_xyz':[results[a].relative_residual for a in 'xyz'],
            'boundary_condition':'affine opposite faces; insulated lateral faces',
            'fields':results}

def tortuosity(volume,pore_phase=0,axis='x',order=2):
    """Pore-only RVE; isolated pores are removed to avoid singular FEM systems."""
    from scipy.ndimage import label
    import numpy as np
    from morqex.microstructure.volume import Volume
    mask = volume.data == pore_phase
    lab,n = label(mask)
    normal = {'x':2,'y':1,'z':0}[axis]
    spanning = (set(np.unique(np.take(lab,0,normal))) & set(np.unique(np.take(lab,-1,normal)))) - {0}
    if not spanning: return {'effective_diffusivity':0.,'tortuosity':float('inf')}
    conducting = np.isin(lab,list(spanning))
    mesh = voxel_to_tetra(Volume(conducting.astype(np.uint8),volume.spacing,volume.origin),phases=[1])
    result = conduction(mesh,{1:1.},axis=axis,order=order)
    # Selected phase may not occupy full outer tangential extent. Use original RVE area.
    original_extent = np.asarray(volume.data.shape)*volume.spacing
    selected_extent = np.ptp(mesh.points,axis=0)[::-1]
    lateral = [i for i in range(3) if i != normal]
    correction = np.prod(selected_extent[lateral])/np.prod(original_extent[lateral])
    effective = result.effective_conductivity*correction
    return {'effective_diffusivity':float(effective),
            'tortuosity':float(mask.mean()/effective),'residual':result.relative_residual}
