"""Public adapters to the pinned complete upstream source distributions.

MicroStructPy executes in the private Python namespace. Micro_Reconstruction
retains its MATLAB/R algorithm and runtime requirements; a Python random-field
draw is deliberately not presented as execution of that algorithm.
"""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from ..geometry import MicrostructureData
from .statistics import characterize,connectivity

PINS={'microstructpy':'6fa72185d5fc6ac9fb6b01f8cecce1f29e4f2339',
      'micro_reconstruction':'0dda2705932fb7d0f4264fd1f22bc94f0daf7e3e'}


def backend_status():
    import importlib.util
    return {'microstructpy':{'commit':PINS['microstructpy'],'python_available':True,
                'power_diagram_available':importlib.util.find_spec('pyvoro') is not None},
        'micro_reconstruction':{'commit':PINS['micro_reconstruction'],'matlab':shutil.which('matlab'),
            'Rscript':shutil.which('Rscript'),'native_qualified':False,
            'requirements':'MATLAB including Image Processing / Parallel Computing toolboxes, R and upstream R packages; original binary datasets remain external'}}


def materialize_upstream(workspace,name):
    """Unpack original pinned source plus all assets outside the source tree."""
    if name not in PINS:raise ValueError('Unknown pinned statistical backend')
    folder=workspace.path('upstream','statistical',name);archive=folder/(PINS[name]+'.zip')
    pin=json.loads((folder/'pin.json').read_text())
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    expected=pin.get('archive_sha256')
    if expected is None:
        # The audited archive checksum is also distributed with the provenance.
        source=Path(__file__).resolve().parents[3]/'provenance'/(name+'_migration.json')
        expected=json.loads(source.read_text())['archive_sha256']
    if checksum!=expected:raise ValueError('Pinned upstream archive checksum mismatch')
    target=workspace.path('upstream','materialized',name,PINS[name]);target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts=Path(member.filename).parts[1:]
            if not parts:continue
            destination=target.joinpath(*parts).resolve()
            if target not in destination.parents:raise ValueError('Unsafe upstream archive path')
            if member.is_dir():destination.mkdir(parents=True,exist_ok=True)
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(z.read(member))
    return target


def reconstruction_runtime(workspace):
    """Return a native workspace only when the original runtime exists.

    Never executes the uncontrolled original parpool/script automatically.
    Caller-supplied licensed MATLAB must first be qualified with bounded workers,
    portable Rscript path and case-correct Geode files in an external copy.
    """
    status=backend_status()['micro_reconstruction']
    if not status['matlab'] or not status['Rscript']:
        raise RuntimeError('Micro_Reconstruction source is integrated; its MATLAB/R runtime is unavailable. Use an explicitly selected Python backend, or install and qualify the native runtime.')
    return {'path':materialize_upstream(workspace,'micro_reconstruction'),'status':status}


def microstructpy_voxels(phases,phase_kinds,*,shape=(16,16,16),spacing_m=1e-6,seed=104,
                        periodic=False,maximum_seeds=1000,maximum_attempts=4,require_percolation=True):
    """Execute genuine upstream seeding -> Laguerre tessellation -> raster.

    ``phases`` is a list of upstream phase dictionaries (sphere/ellipsoid,
    fractions, SciPy random variables for size/orientation). Geometric lengths
    are specified in *voxel* units to avoid small-SI-coordinate tolerances in
    Voro++; ``spacing_m`` converts the result to SI. Fractions prescribe seed
    volume statistics; raster phase fractions are measured, never silently fixed.
    All phases are retained during rasterization, including physical pore labels.
    """
    from ..vendor import microstructpy as msp
    shape=tuple(map(int,shape))
    if len(shape)!=3 or min(shape)<2 or np.prod(shape)>16_777_216 or not np.isfinite(spacing_m) or spacing_m<=0:raise ValueError('Bounded voxel grid and positive isotropic SI spacing required')
    if set(phase_kinds)!=set(range(len(phases))):raise ValueError('Upstream phase indices require explicit contiguous semantics')
    phases=[dict(p,material_type='crystalline') for p in phases]
    domain=msp.geometry.Box(side_lengths=shape[::-1],corner=(0.,0.,0.))
    previous=np.random.get_state()
    try:
        for attempt in range(maximum_attempts):
            draw=seed+attempt
            seeds=msp.seeding.SeedList.from_info(phases,domain.volume,rng_seeds={'size':draw,'phase':draw,'fraction':draw})
            if len(seeds)>maximum_seeds:raise ValueError('Statistical seed budget exceeded before meshing')
            seeds.position(domain,rng_seed=draw,periodic=periodic,max_attempts=1000)
            poly=msp.meshing.PolyMesh.from_seeds(seeds,domain,periodic=periodic)
            raster=msp.meshing.RasterMesh.from_polymesh(poly,1.,phases)
            centres=np.asarray(raster.points)[np.asarray(raster.elements)].mean(axis=1)
            indices=np.floor(centres[:,::-1]).astype(int);labels=np.full(shape,-1,dtype=np.int32)
            attributes=np.asarray(raster.element_attributes,int)
            labels[tuple(indices.T)]=np.array([seeds[s].phase for s in attributes])
            if (labels<0).any() or set(np.unique(labels))!=set(phase_kinds):raise ValueError('Upstream raster omitted cells or a requested phase; increase grid/seeds explicitly')
            data=MicrostructureData(labels,(spacing_m,)*3,(0.,)*3,phase_kinds,
                {'provider':'MicroStructPy seeding / power tessellation / RasterMesh','commit':PINS['microstructpy'],
                 'seed':draw,'seed_count':len(seeds),'periodic_seeding':periodic,'upstream_statistical_parameters':repr(phases),
                 'scope':'stochastic tessellated microstructure; not a unique CT reconstruction or automatic fabrication model'})
            if require_percolation and not all(connectivity(data).values()):continue
            data.provenance['measured_statistics']=characterize(data)[1]
            return data
    finally:np.random.set_state(previous)
    raise ValueError('Upstream geometry failed percolation within the explicit draw budget')
