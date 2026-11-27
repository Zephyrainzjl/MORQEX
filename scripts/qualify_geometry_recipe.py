"""Regenerate source through TOML, publish and independently read a full mesh.

This explicitly records checkpoint reuse; a publication/IO timing is not an
uncached offline-build timing. All generated files stay outside source.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import h5py,numpy as np


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Complete source VoxelModel HDF5')
    p.add_argument('--tiff',type=Path,help='Optional original TIFF, using the recorded explicit axis permutation')
    p.add_argument('--mesh-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--surface-method',choices=('voxel_aligned','voxel_plc','surface_nets'),default='voxel_aligned')
    p.add_argument('--slab-depth',type=int,default=32);p.add_argument('--axis',type=int,choices=(0,1,2),default=0)
    a=p.parse_args();start=time.perf_counter()
    from morqex.modeling import VoxelModel,from_toml
    from morqex.modeling.voxel import external_path
    folder=external_path(a.output);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'custom_name.h5').exists():raise FileExistsError('Use a new publication directory')
    source=VoxelModel.load(a.source);q=json.dumps
    geometry=['[geometry]']
    if a.tiff:
        a.tiff=a.tiff.resolve()
        if source.metadata.get('source_sha256')!=digest(a.tiff):raise ArithmeticError('TIFF no longer matches recorded source provenance')
        axes=source.metadata.get('axes_to_zyx')
        if axes is None:raise ValueError('Recorded TIFF axis permutation required; no guess permitted')
        geometry+=['method="tiff"','path='+q(str(a.tiff)),'axes_to_zyx='+q(list(axes))]
    else:
        np.save(folder/'labels.npy',source.regions,allow_pickle=False)
        geometry+=['method="regions"','array="labels.npy"']
    geometry+=['spacing='+q(list(map(float,source.spacing))),'origin='+q(list(map(float,source.origin))),
        '[geometry.region_names]',*[q(str(k))+'='+q(v,ensure_ascii=False) for k,v in source.region_names.items()],
        '[meshing]','method="partitioned_tetgen"','directory='+q(str(a.mesh_directory.resolve())),
        'surface_method='+q(a.surface_method),'slab_depth='+str(a.slab_depth),'axis='+str(a.axis)]
    recipe=folder/'recipe.toml';recipe.write_text('\n'.join(geometry)+'\n')
    regenerated=from_toml(recipe)
    np.testing.assert_array_equal(regenerated.regions,source.regions)
    assert regenerated.spacing==source.spacing and regenerated.origin==source.origin and regenerated.region_names==source.region_names
    del regenerated
    cached=a.mesh_directory.resolve()/'mesh.h5';marker=a.mesh_directory.resolve()/'validation.json'
    reused=marker.exists();before=cached.stat().st_mtime_ns if reused else None
    command=[sys.executable,'-m','morqex.modeling','generate',str(recipe),'--output',str(folder/'geometry_from_toml.h5'),
        '--mesh',str(folder/'custom_name.h5'),'--mesh-preview',str(folder/'mesh.html')]
    subprocess.run(command,check=True)
    published=folder/'custom_name.h5'
    sha=digest(cached)
    if digest(published)!=sha:raise ArithmeticError('Published HDF5 differs from accepted checkpoint')
    if reused and cached.stat().st_mtime_ns!=before:raise ArithmeticError('Cache reuse changed the validated whole mesh')
    loaded=VoxelModel.load(folder/'geometry_from_toml.h5')
    np.testing.assert_array_equal(loaded.regions,source.regions)
    import meshio
    mesh=meshio.read(published.with_suffix('.xdmf'))
    blocks=[(i,block.data) for i,block in enumerate(mesh.cells) if block.type=='tetra']
    if len(blocks)!=1:raise ArithmeticError('Published Tet4 XDMF must contain one volume block')
    block_index,volume_cells=blocks[0];region_values=mesh.cell_data['region'][block_index]
    with h5py.File(published) as h:
        if not h.attrs['complete']:raise ArithmeticError('Incomplete published artifact')
        report=json.loads(h.attrs['report']);nt=len(h['mesh/tetrahedra'])
        assert len(mesh.points)==len(h['mesh/points_m']) and len(volume_cells)==nt
        for first in range(0,len(mesh.points),50000):np.testing.assert_array_equal(mesh.points[first:first+50000],h['mesh/points_m'][first:first+50000])
        for first in range(0,nt,50000):
            np.testing.assert_array_equal(volume_cells[first:first+50000],h['mesh/tetrahedra'][first:first+50000])
            np.testing.assert_array_equal(region_values[first:first+50000],h['mesh/phase'][first:first+50000])
        if 'custom_name.h5' not in published.with_suffix('.xdmf').read_text():raise ArithmeticError('XDMF does not refer to custom publication name')
    result={'passed':True,'source_shape_zyx':list(source.regions.shape),'source_voxels':int(source.regions.size),
        'source':str(a.source.resolve()),'source_sha256':digest(a.source),'source_array_unchanged':True,
        'recipe':str(recipe),'published_hdf5':str(published),'published_and_checkpoint_sha256':sha,
        'nodes':len(mesh.points),'tetrahedra':nt,'all_meshio_xdmf_arrays_equal':True,
        'reused_validated_checkpoint':reused,'checkpoint_unmodified':True if reused else None,'wall_seconds':time.perf_counter()-start,
        'scope':'TOML source regeneration, complete publication and independent XDMF IO; reuse timing excludes prior offline native construction',
        'prior_validation':str(marker),'quality_limits_satisfied':report.get('quality_limits_satisfied')}
    (folder/'public_api_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
