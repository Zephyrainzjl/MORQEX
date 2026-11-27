"""Regenerate offline views and correct aggregate quality metadata, not mesh data.

Use run_local --reference-memory --large-mesh-output. This also upgrades views
created by earlier frozen workers. Numeric geometry/connectivity is untouched.
"""
import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))


def main(base):
    import h5py
    from morqex.modeling import MeshArtifact,VoxelModel
    from morqex.visualization.adaptive_mesh import AdaptiveMeshView
    records=[]
    for folder in [base/'nmc_tiff/exact_voxel_partitioned',base/'native/granular/partitioned',
                   *(base/'native'/k/'partitioned' for k in ('rods','platelets','agglomerates','bicontinuous','cell_stack'))]:
        if not (folder/'validation.json').exists():continue
        reports=[json.loads(p.read_text()) for p in sorted(folder.glob('part_[0-9][0-9].json'))]
        quality=[r['quality'] for r in reports if r.get('quality')];status=[r.get('quality_limits_satisfied') for r in reports]
        with h5py.File(folder/'mesh.h5','r+') as f:
            report=json.loads(f.attrs['report']);report['quality_limits_satisfied']=False if False in status else True if all(v is True for v in status) else None
            if quality:report['quality']={'radius_edge_max':max(r['radius_edge_max'] for r in quality),
                'minimum_dihedral_degrees':min(r['minimum_dihedral_degrees'] for r in quality),
                'scope':'exact extrema aggregated from native parts; no claim for pooled quantiles'}
            f.attrs['report']=json.dumps(report)
        validation=json.loads((folder/'validation.json').read_text())
        for k in ('quality','quality_limits_satisfied'):
            if k in report:validation[k]=report[k]
        (folder/'validation.json').write_text(json.dumps(validation,indent=2))
        artifact=MeshArtifact.load(folder/'mesh.h5')
        AdaptiveMeshView(artifact,maximum_embedded_faces=450000,node_cache_bytes=256*2**20,
            title=folder.parent.name+' / complete actual TetGen volume').save_html(folder/'mesh.html')
        records.append({'folder':str(folder),'quality_limits_satisfied':report['quality_limits_satisfied'],
            'scope':'quality metadata and visualization refreshed; numeric nodes/cells/phases not modified'})
        print(json.dumps(records[-1]),flush=True)
    ref=MeshArtifact.load(base/'reference_comsol/mesh.h5')
    AdaptiveMeshView(ref,title='Actual independent 1 m COMSOL reference').save_html(base/'reference_comsol/mesh.html')
    # Correct an older XDMF declaration for imported Int64 connectivity.
    from xml.sax.saxutils import escape
    with h5py.File(ref.hdf5) as f:
        cells=f['mesh/tetrahedra'];phase=f['mesh/phase'];points=f['mesh/points_m'];fn=escape(ref.hdf5.name)
        ref.xdmf.write_text(f'''<Xdmf Version="3.0"><Domain><Grid Name="Independent COMSOL reference">
<Topology TopologyType="Tetrahedron" NumberOfElements="{len(cells)}"><DataItem Dimensions="{len(cells)} 4" NumberType="Int" Precision="{cells.dtype.itemsize}" Format="HDF">{fn}:/mesh/tetrahedra</DataItem></Topology>
<Geometry GeometryType="XYZ"><DataItem Dimensions="{len(points)} 3" NumberType="Float" Precision="8" Format="HDF">{fn}:/mesh/points_m</DataItem></Geometry>
<Attribute Name="region" Center="Cell"><DataItem Dimensions="{len(phase)}" NumberType="Int" Precision="{phase.dtype.itemsize}" Format="HDF">{fn}:/mesh/phase</DataItem></Attribute></Grid></Domain></Xdmf>''')
    for folder in [base/'nmc_tiff',*(base/'native'/k for k in ('granular','rods','platelets','agglomerates','bicontinuous','cell_stack'))]:
        model=VoxelModel.load(folder/'geometry.h5')
        model.viewer(maximum_3d_size=64,title=folder.name+' / complete original source').save_html(folder/'voxels.html')
    sys.path.insert(0,str(ROOT/'scripts'));from complex_geometry_gallery import write_index
    write_index(base);(base/'view_revision.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(os.environ['MORQEX_WORKSPACE'])/'nmc_complex_gallery')
    main(p.parse_args().root)
