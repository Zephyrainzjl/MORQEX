"""Reproducible true-voxel particle geometry; every artifact stays external."""
import json,os
from pathlib import Path
import numpy as np
from morqex.generation import SpherePackSpec,sphere_pack
from morqex.visualization.particles import particle_preview

if __name__=='__main__':
    ws=Path(os.environ['MORQEX_WORKSPACE']);data,owners=sphere_pack(SpherePackSpec())
    root=ws/'data/sphere32';root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'geometry.npz',labels=data.labels,owners=owners,spacing=data.spacing_zyx_m)
    (root/'manifest.json').write_text(json.dumps(data.provenance,indent=2))
    print(particle_preview(data,owners,ws/'reports/MORQEX_sphere32_Li_geometry.html'))
