"""Independent topology, partition, geometry and real-adaptivity checks."""
from pathlib import Path
import numpy as np
import pytest
from scipy import ndimage
from morqex.modeling import (from_regions,from_shapes,Shape,AdaptiveMeshOptions,adaptive_surface,adaptive_mesh,
    mesh,export_comsol,read_comsol,MeshArtifact,resolved_electrode,PRESETS)


@pytest.fixture
def native():
    from morqex.modeling.acvd_backend import load_acvd
    from morqex.modeling.tetgen_backend import library_path
    try:load_acvd();library_path()
    except RuntimeError as error:pytest.skip(str(error))


def sphere():
    return from_shapes([Shape('ellipsoid',3,(16e-6,20e-6,24e-6),(10e-6,12e-6,14e-6),irregularity=.10)],
                       shape=(36,40,48),spacing=(1e-6,)*3)


def test_original_weighted_kernel_is_actually_used(native):
    from morqex.modeling.acvd_backend import clustering_class
    assert clustering_class().__module__=='morqex.microstructure.mesh.pyacvd.clustering'
    model=sphere();surface=adaptive_surface(model,regions=[3],options=AdaptiveMeshOptions(vertex_fraction=.25,maximum_phase_volume_error=.06))
    g=surface.metadata
    assert g['output_surface_vertices']<g['input_surface_vertices']*.8
    for p in g['patches']:
        assert p['before']['euler']==p['after']['euler']
        assert p['after']['boundary_edges']==p['before']['boundary_edges']==0
    grid=surface.points/1e-6
    assert np.mean(np.any(abs(2*grid-np.rint(2*grid))>1e-6,axis=1))>.5


def test_density_callback_changes_real_spatial_vertex_distribution(native):
    model=sphere();centre=24e-6
    options=AdaptiveMeshOptions(vertex_fraction=.25,maximum_phase_volume_error=.08,weight_strength=0,
        weight_floor=1,maximum_weight_ratio=16,density=lambda p:np.where(p[:,0]>centre,8.,1.))
    surface=adaptive_surface(model,regions=[3],options=options)
    # A symmetric/near-symmetric source receives substantially more real
    # vertices on its deliberately more densely weighted side.
    assert np.mean(surface.points[:,0]>centre)>.58


def test_custom_nested_material_ids_and_solid_only_selection(native,tmp_path):
    shape=(32,36,40);z,y,x=np.indices(shape)
    core=(z-15.5)**2+(y-17.5)**2+(x-19.5)**2<8**2
    coating=ndimage.distance_transform_edt(~core)<=3
    labels=np.full(shape,5,np.uint8);labels[coating]=11;labels[core]=23
    model=from_regions(labels,spacing=(2e-6,)*3,origin=(1e-4,-2e-4,3e-4),metadata={'phase_order':[5,11,23]})
    options=AdaptiveMeshOptions(vertex_fraction=.5,maximum_phase_volume_error=.08)
    for selected in ([5,11,23],[11,23]):
        result=mesh(model,method='gemsim_adaptive',regions=selected,options=options)
        assert set(result.phases)==set(selected)
        assert np.all(result.signed_volumes()>0)
        assert max(result.report['region_volume_relative_errors'].values())<1e-8
        path=tmp_path/('mesh_'+str(len(selected))+'.mphtxt');export_comsol(result,path)
        back=read_comsol(path);np.testing.assert_array_equal(back['elements']['tet']['cells'],result.cells)
        if len(selected)==3:
            np.testing.assert_allclose(result.signed_volumes().sum(),np.prod(shape)*np.prod(model.spacing),rtol=1e-8)
        artifact=result.save_hdf5(tmp_path/('mesh_'+str(len(selected))+'.h5'))
        assert MeshArtifact.load(artifact.hdf5)==artifact


def test_real_tet_sections_and_adaptive_viewer(native):
    from morqex.visualization.adaptive_mesh import section_polygons
    model=sphere();result=adaptive_mesh(model,regions=[3],options=AdaptiveMeshOptions(vertex_fraction=.3,maximum_phase_volume_error=.08))
    cut=section_polygons(result,axis=0)
    assert cut['polygons'] and set(cut['phase'])=={3}
    area=0
    for p in cut['polygons']:
        p=np.array(p);area+=abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2
    assert 0.5*np.pi*10e-6*12e-6<area<1.5*np.pi*10e-6*12e-6
    view=result.adaptive_viewer();assert 'Tet4' in view.document and 'DecompressionStream' in view.document
    assert result.report['geometry']['interface_pairs']=={'1':[-1,3]}


@pytest.mark.parametrize('name',[p[0] for p in PRESETS])
def test_resolved_generators_are_real_3d_and_all_three_phases(name):
    model=resolved_electrode(name,size=96,seed=7,minimum_feature_voxels=2,cbd_thickness_voxels=2)
    assert model.regions.shape==(96,)*3 and set(np.unique(model.regions))=={0,1,2}
    assert model.metadata['phase_order']==[0,2,1]
    # The coating is resolved: there is no direct active/pore face contact.
    for axis in range(3):
        a=np.moveaxis(model.regions,axis,0);pairs=(a[:-1]==1)&(a[1:]==0)|(a[:-1]==0)&(a[1:]==1)
        assert not pairs.any()


def test_bad_density_and_nested_order_rejected(native):
    with pytest.raises(ValueError):AdaptiveMeshOptions(maximum_weight_ratio=.5)
    with pytest.raises(ValueError):resolved_electrode('granular',size=32)
    with pytest.raises(ValueError,match='density'):
        adaptive_surface(sphere(),regions=[3],options=AdaptiveMeshOptions(density=lambda p:np.full(len(p),np.nan)))
    assert AdaptiveMeshOptions.precision('coarse').vertex_fraction<AdaptiveMeshOptions.precision('fine').vertex_fraction
    with pytest.raises(ValueError):AdaptiveMeshOptions(maximum_facets=0)
    with pytest.raises(ValueError,match='regions'):adaptive_surface(sphere(),regions=[3,3])
    with pytest.raises(ValueError,match='regions'):adaptive_surface(sphere(),regions=[])


def test_shared_triple_junction_arrays_retain_all_materials(native):
    labels=np.full((14,16,18),9,np.uint8);labels[:,:,:9]=4;labels[:7,8:,:9]=17
    model=from_regions(labels,spacing=(1e-6,2e-6,3e-6))
    result=adaptive_mesh(model,regions=[4,9,17],options=AdaptiveMeshOptions(
        surface_method='shared_interfaces',vertex_fraction=.25,levels=2,maximum_phase_volume_error=1e-8))
    assert set(result.phases)=={4,9,17}
    assert result.report['native_process_isolated']
    assert max(result.report['region_volume_relative_errors'].values())<1e-8
    np.testing.assert_allclose(result.signed_volumes().sum(),np.prod(labels.shape)*np.prod(model.spacing),rtol=1e-8)


def test_cli_toml_dispatch_region_override_and_retained_native_route(native,tmp_path):
    import subprocess,sys,h5py
    labels=np.full((6,7,8),4,np.uint8);labels[:,:,4:]=9
    np.save(tmp_path/'labels.npy',labels)
    recipe=tmp_path/'geometry.toml'
    recipe.write_text('''[geometry]
method = "regions"
array = "labels.npy"
[meshing]
method = "gemsim_adaptive"
regions = [4]
[meshing.options]
vertex_fraction = 0.4
levels = 2
maximum_phase_volume_error = 0.01
''')
    geometry=tmp_path/'geometry.h5'
    subprocess.run([sys.executable,'-m','morqex.modeling','generate',str(recipe),
        '--output',str(geometry),'--mesh',str(tmp_path/'adaptive.h5'),'--regions','all',
        '--mesh-preview',str(tmp_path/'adaptive.html'),'--comsol',str(tmp_path/'adaptive.mphtxt')],check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,'-m','morqex.modeling','mesh',str(geometry),
        '--method','tetgen','--output',str(tmp_path/'raw.h5'),'--mesh-preview',str(tmp_path/'raw.html')],
        check=True,capture_output=True,text=True)
    for name in ('adaptive','raw'):
        with h5py.File(tmp_path/(name+'.h5')) as f:assert set(f['mesh/phase'][:])=={4,9}
        assert 'DecompressionStream' in (tmp_path/(name+'.html')).read_text()


def test_flat_star_replacement_preserves_boundary_and_does_not_remove_real_shell():
    from morqex.modeling.adaptive import _clean_flat_stars,_topology
    p=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.25,.25,1e-9]])
    f=np.array([[3,0,1],[3,1,2],[3,2,0]],np.int32)
    q,g,mapping,removed=_clean_flat_stars(p,f,np.array([True,True,True,False]),1e-6)
    assert removed==1 and len(g)==1 and mapping[3]==-1
    assert _topology(p,f)['euler']==_topology(q,g)['euler']==1
    assert _topology(q,g)['boundary_edges']==3
    area=np.linalg.norm(np.cross(q[g[0,1]]-q[g[0,0]],q[g[0,2]]-q[g[0,0]]))/2
    assert area==pytest.approx(.5)
    shell=np.vstack((f,[0,2,1]))
    q,g,_,removed=_clean_flat_stars(p,shell,np.zeros(4,bool),1e-6)
    assert removed==0 and len(g)==4


def test_directed_surface_orientation_is_checked_separately_from_euler():
    from morqex.modeling.adaptive import _orient_surface
    p=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    original=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int32)
    f=original.copy();f[1]=f[1,::-1]
    expected=np.cross(p[original[:,1]]-p[original[:,0]],p[original[:,2]]-p[original[:,0]])
    corrected=_orient_surface(p,f,expected)
    edges=corrected[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    key=edges.min(axis=1)*4+edges.max(axis=1);_,inverse=np.unique(key,return_inverse=True)
    assert np.all(np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))==0)
    volume=np.einsum('ij,ij->i',p[corrected[:,0]],np.cross(p[corrected[:,1]],p[corrected[:,2]])).sum()/6
    assert volume==pytest.approx(1/6)


def test_sections_through_shared_faces_have_no_double_counting():
    from morqex.visualization.adaptive_mesh import section_polygons
    labels=np.ones((2,2,2),np.uint8);labels[:,:,1]=2
    model=from_regions(labels,spacing=(1.,1.,1.));mesh=model.tetrahedralize()
    for axis in range(3):
        section=section_polygons(mesh,axis)
        area=0
        for polygon in section['polygons']:
            p=np.asarray(polygon);area+=abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2
        assert area==pytest.approx(4.,abs=1e-10)
        if axis==0:assert set(section['phase'])=={1}


def test_sections_page_large_hdf5_node_ids_without_fancy_indexing(tmp_path):
    import h5py
    from morqex.visualization.adaptive_mesh import section_polygons
    cube=from_regions(np.ones((1,1,1),np.uint8),spacing=(1.,)*3).tetrahedralize()
    # Sparse HDF5 allocation exercises IDs beyond the48MB cache threshold
    # without writing/allocating millions of physical test nodes.
    offset=2000000;path=tmp_path/'sparse_node_table.h5'
    with h5py.File(path,'w') as f:
        f.attrs['complete']=True
        points=f.create_dataset('mesh/points_m',shape=(offset+len(cube.points),3),dtype='f8',chunks=(4096,3),fillvalue=0)
        points[offset:]=cube.points
        f.create_dataset('mesh/tetrahedra',data=cube.cells+offset)
        f.create_dataset('mesh/phase',data=cube.phases)
    artifact=MeshArtifact(path,path.with_suffix('.xdmf'),offset+len(cube.points),len(cube.cells),(1,))
    for axis in range(3):
        section=section_polygons(artifact,axis);area=0.
        for polygon in section['polygons']:
            p=np.asarray(polygon);area+=abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2
        assert area==pytest.approx(1.,abs=1e-10)


def test_cavity_components_have_material_orientation_not_just_closed_edges():
    from morqex.modeling.adaptive import _orient_against_mask
    from morqex.modeling import SurfacePLC
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    q=np.array([[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])
    f=np.concatenate((q[:,[0,1,2]],q[:,[0,2,3]]))
    points=np.vstack((corners*32,8+corners*16));faces=np.vstack((f,f+8))
    labels=np.ones((32,)*3,np.uint8);labels[8:24,8:24,8:24]=0
    model=from_regions(labels,spacing=(1.,)*3)
    corrected,flipped=_orient_against_mask(model,labels==1,points,faces,1)
    volume=np.einsum('ij,ij->i',points[corrected[:,0]],np.cross(points[corrected[:,1]],points[corrected[:,2]])).sum()/6
    assert flipped==1 and volume==pytest.approx(32**3-16**3)


def test_adaptive_tetgen_retains_a_shallow_bump_volume(native):
    from morqex.modeling import SurfacePLC,TetGenOptions,tetgen_mesh
    p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],[.5,.5,1.00001]],float)*1e-6
    faces=[[0,3,2,1],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,8],[5,6,8],[6,7,8],[7,4,8]]
    expected=(1+1e-5/3)*1e-18
    plc=SurfacePLC(p,faces,seeds={7:[[.5e-6]*3]},region_volumes={7:expected})
    mesh=tetgen_mesh(plc,options=TetGenOptions.for_adaptive())
    assert mesh.signed_volumes().sum()==pytest.approx(expected,rel=1e-9)
