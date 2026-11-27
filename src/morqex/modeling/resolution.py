"""Independent surface-fidelity and interior-size controls for one voxel model."""
from dataclasses import dataclass,replace,asdict
import hashlib
import numpy as np
from .adaptive import AdaptiveMeshOptions,adaptive_mesh
from .tetgen import TetGenOptions,tetgen_mesh


@dataclass(frozen=True)
class MeshResolution:
    """Preset defaults are requests, not certified solution-error estimates.

    target_cells controls a maximum cell volume V/target_cells. It is a LOWER
    density request, never a promise of exactly that many cells. Surface shape,
    grading and quality can require many more; maximum_cells stays a hard cap.
    """
    precision:str='balanced'
    target_cells:int|None=None
    maximum_volume_m3:float|None=None
    nominal_edge_m:float|None=None
    vertex_fraction:float|None=None
    maximum_phase_volume_error:float|None=None

    def __post_init__(self):
        if self.precision not in ('coarse','balanced','fine','very_fine'):raise ValueError('Unknown resolution preset')
        if self.target_cells is not None and (type(self.target_cells) is not int or self.target_cells<1):raise ValueError('Positive integer target_cells required')
        for value in (self.maximum_volume_m3,self.nominal_edge_m):
            if value is not None and (not np.isfinite(value) or value<=0):raise ValueError('Positive finite SI mesh size required')
        if self.maximum_volume_m3 is not None and self.nominal_edge_m is not None:raise ValueError('Specify maximum_volume_m3 or nominal_edge_m, not both')
        if self.vertex_fraction is not None and not 0<self.vertex_fraction<=1:raise ValueError('vertex_fraction is in(0,1]')
        if self.maximum_phase_volume_error is not None and not 0<self.maximum_phase_volume_error<1:raise ValueError('Positive phase-volume tolerance required')

    def options(self,volume_m3,*,adaptive=None,native=None):
        counts={'coarse':4000,'balanced':16000,'fine':64000,'very_fine':256000}
        default=AdaptiveMeshOptions.precision(self.precision)
        surface=adaptive or default
        changes={k:getattr(self,k) for k in ('vertex_fraction','maximum_phase_volume_error') if getattr(self,k) is not None}
        surface=replace(surface,**changes)
        volume=self.maximum_volume_m3
        if self.nominal_edge_m is not None:
            # Volume of an equilateral Tet4 with this edge; this is NOT a
            # guaranteed maximum edge length for an arbitrary sliver.
            volume=self.nominal_edge_m**3/(6*np.sqrt(2))
        if volume is None:
            volume=(native.maximum_volume if native is not None and native.maximum_volume is not None and self.target_cells is None
                else float(volume_m3)/(self.target_cells or counts[self.precision]))
        backend=replace(native or TetGenOptions.for_adaptive(),maximum_volume=volume)
        return surface,backend


def controlled_tetgen(model,*,resolution=None,regions=None,surface_method='gemsim_adaptive',
                      adaptive_options=None,tetgen_options=None,workspace=None,maximum_facets=4000000,step_size=1):
    """Mesh identical source labels at selectable fidelity and interior size.

    Select voxel_plc to keep stair-step boundaries exactly; gemsim_adaptive
    explicitly approximates the surface. SurfaceNets is a separate shared
    interface front end. No automatic source resampling or phase deletion.
    """
    from .surface_nets import surface_nets_plc
    from .voxel import VoxelModel
    if not isinstance(model,VoxelModel):raise TypeError('VoxelModel required')
    controls=resolution or MeshResolution()
    if not isinstance(controls,MeshResolution):raise TypeError('MeshResolution required')
    if controls.vertex_fraction is not None and surface_method!='gemsim_adaptive':
        raise ValueError('vertex_fraction requires the gemsim_adaptive surface route; fixed/exact surfaces use interior size controls')
    labels=tuple(map(int,np.unique(model.regions) if regions is None else regions))
    if not labels or not set(labels)<=set(map(int,np.unique(model.regions))):raise ValueError('Select existing labels')
    checksum=hashlib.sha256(np.ascontiguousarray(model.regions).view(np.uint8)).hexdigest()
    volume=float(np.isin(model.regions,labels).sum()*np.prod(model.spacing))
    surface,native=controls.options(volume,adaptive=adaptive_options,native=tetgen_options)
    if surface_method=='gemsim_adaptive':
        result=adaptive_mesh(model,regions=labels,options=surface,tetgen_options=native,workspace=workspace)
    elif surface_method=='surface_nets':
        front=surface_nets_plc(model,regions=labels,maximum_facets=maximum_facets)
        result=tetgen_mesh(front,options=native,workspace=workspace,maximum_facets=maximum_facets)
    elif surface_method in ('voxel_plc','marching_cubes'):
        result=tetgen_mesh(model,regions=labels,surface_method=surface_method,step_size=step_size,options=native,workspace=workspace,maximum_facets=maximum_facets)
    else:raise ValueError('Unknown surface method')
    if hashlib.sha256(np.ascontiguousarray(model.regions).view(np.uint8)).hexdigest()!=checksum:
        raise ArithmeticError('Meshing changed source labels')
    actual=result.report['region_volumes_m3']
    errors={str(k):abs(actual[str(k)]/float((model.regions==k).sum()*np.prod(model.spacing))-1) for k in labels}
    if max(errors.values())>surface.maximum_phase_volume_error*(1+1e-10):
        raise ArithmeticError(f'Requested source phase-volume fidelity not met: {errors}; tolerance={surface.maximum_phase_volume_error}')
    result.report['resolution_controls']={'requested':asdict(controls),'source_voxel_sha256':checksum,
        'source_shape_zyx':list(model.regions.shape),'source_spacing_zyx_m':list(model.spacing),
        'surface_method':surface_method,'step_size':step_size,'maximum_volume_m3':native.maximum_volume,
        'region_maximum_volumes_m3':native.region_maximum_volumes,'measured_source_phase_volume_errors':errors,
        'actual_nodes':len(result.points),'actual_tetrahedra':len(result.cells),
        'scope':'size and geometric-fidelity requests; no PDE-error certification or guaranteed edge-length bound'}
    return result
