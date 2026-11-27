"""Align source voxel centres and exact Tet4 sections; never modify mesh data."""
import argparse,json,os
from pathlib import Path
import h5py,numpy as np
from morqex.modeling import VoxelModel,MeshArtifact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'))
    a=p.parse_args()
    if a.workspace is None:p.error('Specify an external workspace')
    root=a.workspace.resolve()/'nmc_complex_gallery';records=[]
    folders=[root/'nmc_tiff/exact_voxel_partitioned']+[
        root/'native'/name/'partitioned' for name in ('granular','rods','platelets','agglomerates','bicontinuous','cell_stack')]
    for folder in folders:
        source=folder.parent/'geometry.h5';meshpath=folder/'mesh.h5'
        stamps=[(path.stat().st_size,path.stat().st_mtime_ns) for path in (source,meshpath)]
        model=VoxelModel.load(source);artifact=MeshArtifact.load(meshpath)
        shape=np.asarray(model.regions.shape[::-1]);spacing=np.asarray(model.spacing[::-1]);origin=np.asarray(model.origin[::-1])
        low=np.full(3,np.inf);high=-low
        with h5py.File(meshpath) as h:
            points=h['mesh/points_m']
            for first in range(0,len(points),50000):
                block=points[first:first+50000];low=np.minimum(low,block.min(0));high=np.maximum(high,block.max(0))
        if not (np.allclose(low,origin,rtol=0,atol=spacing.min()*1e-8) and
                np.allclose(high,origin+shape*spacing,rtol=0,atol=spacing.min()*1e-8)):
            raise ArithmeticError('Cannot align independent or differently sized source and mesh: '+str(folder))
        fractions=(shape//2+.5)/shape;planes=origin+(shape//2+.5)*spacing
        model.viewer(maximum_3d_size=64,slice_origin='lower',title=folder.parent.name+' / complete original source').save_html(folder.parent/'voxels.html')
        artifact.adaptive_viewer(maximum_embedded_faces=450000,node_cache_bytes=256*2**20,
            section_fractions=fractions,title=folder.parent.name+' / complete actual TetGen volume').save_html(folder/'mesh.html')
        assert stamps==[(path.stat().st_size,path.stat().st_mtime_ns) for path in (source,meshpath)]
        records.append({'source':str(source),'mesh':str(meshpath),'section_fractions_xyz':fractions.tolist(),
            'section_planes_xyz_m':planes.tolist(),'source_and_mesh_files_unmodified':True,
            'scope':'default voxel Z plane and mesh XY plane coincide; controls remain independent'})
        print(json.dumps(records[-1]),flush=True)
    from complex_geometry_gallery import write_index
    write_index(root)
    (root/'comparison_alignment.json').write_text(json.dumps({'passed':True,'checks':records},indent=2))


if __name__=='__main__':main()
