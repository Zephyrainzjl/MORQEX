"""Generated or checksum-pinned user microstructures for the same cycle API."""
from dataclasses import asdict
import hashlib,re
from pathlib import Path
from .morphology import PolycrystalSpec
from .polycrystal import PolycrystalData


def load_cycle_geometry(settings,workspace):
    if 'source_hdf5' not in settings:
        spec=PolycrystalSpec(**settings)
        data=spec.generate();metadata=asdict(spec);seed=spec.seed
    else:
        if set(settings)!={'source_hdf5','sha256','solver_seed'}:
            raise ValueError('Imported geometry requires source_hdf5, sha256 and solver_seed only')
        path=Path(settings['source_hdf5'])
        if path.is_absolute() or not (workspace.root/path).resolve().is_relative_to(workspace.root.resolve()):
            raise ValueError('Geometry source must be inside the external workspace')
        path=(workspace.root/path).resolve()
        expected=settings['sha256'];seed=settings['solver_seed']
        if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Explicit geometry SHA256 required')
        if type(seed) is not int or seed<0:raise ValueError('Nonnegative solver seed required')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=expected:raise ValueError('Imported geometry file checksum mismatch')
        data=PolycrystalData.load_hdf5(path);metadata=dict(settings)
    # This physical module declares exactly these three material laws. A new
    # phase requires a constitutive extension, not an accidental fallback.
    import numpy as np
    allowed={0:'pore',1:'active',2:'cbd'}
    if any(int(label) not in allowed or data.structure.phase_kinds[int(label)]!=allowed[int(label)]
           for label in np.unique(data.structure.labels)):
        raise ValueError('Cycle material contract requires phase 0=pore, 1=active, 2=CBD')
    metadata['fingerprint']=data.fingerprint
    return data,metadata,seed
