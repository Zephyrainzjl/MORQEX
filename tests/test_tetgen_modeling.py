"""Geometry/partition/conservation and COMSOL native-format contract tests."""
import json,os
from pathlib import Path
import numpy as np
import pytest
from morqex.modeling import (from_regions,voxel_plc,marching_cubes_plc,SurfacePLC,
    tetgen_mesh,TetGenOptions,export_comsol,read_comsol,stream_tetrahedralize)
from morqex.microstructure.mesh.voxel import FEMesh,voxel_to_hex


@pytest.fixture
def native():
    from morqex.modeling.tetgen_backend import library_path
    try:return library_path()
    except RuntimeError as e:pytest.skip(str(e))


def test_plc_interfaces_emitted_once_and_seeds_are_inside():
    a=np.zeros((3,4,5),np.uint8);a[:,:,2:]=7
    model=from_regions(a,spacing=(2e-6,3e-6,5e-6),origin=(-2e-6,5e-6,9e-6))
    plc=voxel_plc(model)
    expected=2*(3*4+3*5+4*5)+3*4
    assert len(plc.facets)==expected
    assert sum(plc.region_volumes.values())==pytest.approx(a.size*np.prod(model.spacing))
    for k,seeds in plc.seeds.items():
        ijk=np.floor((seeds[:,::-1]-model.origin)/model.spacing).astype(int)
        assert np.all(a[tuple(ijk.T)]==k)
    np.testing.assert_allclose(plc.points.min(axis=0),np.array(model.origin)[::-1])


@pytest.mark.parametrize('case',['homogeneous','layers','cavity','disconnected'])
def test_native_regions_positive_volume_and_exact_conservation(native,case):
    a=np.ones((4,5,6),np.uint8);selected=[1]
    if case=='layers':a[:,:,3:]=7;selected=[1,7]
    if case=='cavity':a[1:3,2:4,2:4]=0
    if case=='disconnected':a[:,:,2:4]=0
    model=from_regions(a,spacing=(2e-6,3e-6,5e-6),origin=(-1e-4,2e-4,3e-5))
    mesh=tetgen_mesh(model,regions=selected)
    v=mesh.signed_volumes();assert np.all(v>0)
    for k in selected:assert v[mesh.phases==k].sum()==pytest.approx((a==k).sum()*np.prod(model.spacing),rel=1e-10)
    assert set(mesh.phases)==set(selected)
    assert mesh.report['quality']['minimum_dihedral_degrees']>0
    if case=='layers':
        # A real shared mesh face joins the two material labels at x=3*h_x.
        faces={}
        for tet,phase in zip(mesh.cells,mesh.phases):
            for local in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]:
                faces.setdefault(tuple(sorted(tet[local])),[]).append(phase)
        interfaces=[f for f,p in faces.items() if len(p)==2 and p[0]!=p[1]]
        assert interfaces
        assert np.allclose(mesh.points[np.array(interfaces),0],model.origin[2]+3*model.spacing[2])


def test_marching_cubes_voxel_centre_coordinates_and_closed_surface():
    a=np.zeros((8,10,12),np.uint8);a[2:6,3:7,4:9]=1
    plc=marching_cubes_plc(from_regions(a,spacing=(1e-6,2e-6,3e-6)),region=1)
    np.testing.assert_allclose(plc.points.min(axis=0),[12e-6,6e-6,2e-6],atol=1e-12)
    np.testing.assert_allclose(plc.points.max(axis=0),[27e-6,14e-6,6e-6],atol=1e-12)


def test_native_closed_cavity_and_refinement(native):
    a=np.ones((2,2,2),np.uint8);model=from_regions(a)
    coarse=tetgen_mesh(model)
    fine=tetgen_mesh(model,options=TetGenOptions(maximum_volume=1e-20))
    assert len(fine.cells)>len(coarse.cells)
    assert fine.signed_volumes().max()<=1e-20*(1+1e-8)


@pytest.mark.parametrize('kind',['tetra','hexahedron','streamed','tetgen'])
def test_comsol_roundtrip_material_zero_si_coordinates_and_boundaries(tmp_path,native,kind):
    a=np.zeros((2,3,4),np.uint8);a[:,:,2:]=9
    model=from_regions(a,spacing=(2e-6,3e-6,4e-6),origin=(-3e-5,2e-4,7e-5))
    if kind=='tetra':mesh=model.tetrahedralize()
    elif kind=='hexahedron':mesh=voxel_to_hex(model.volume())
    elif kind=='streamed':mesh=stream_tetrahedralize(model,tmp_path/'mesh.h5')
    else:mesh=tetgen_mesh(model)
    path=export_comsol(mesh,tmp_path/'mesh.mphtxt',chunk_size=7)
    result=read_comsol(path);mapping=json.loads(path.with_suffix('.regions.json').read_text())
    assert mapping['region_to_domain']=={'0':1,'9':2}
    if kind=='streamed':
        import h5py
        with h5py.File(mesh.hdf5) as f:points=f['mesh/points_m'][:];cells=f['mesh/tetrahedra'][:];phase=f['mesh/phase'][:]
    else:points,cells,phase=mesh.points,mesh.cells,mesh.phases
    np.testing.assert_array_equal(result['points'],points)
    key='hex' if kind=='hexahedron' else 'tet'
    expected=cells[:,[0,1,3,2,4,5,7,6]] if kind=='hexahedron' else cells
    np.testing.assert_array_equal(result['elements'][key]['cells'],expected)
    np.testing.assert_array_equal(result['elements'][key]['entities'],np.where(phase==0,1,2))
    if kind=='tetgen':
        np.testing.assert_array_equal(result['elements']['tri']['cells'],mesh.facets)
        assert result['elements']['tri']['entities'].min()==0
    with pytest.raises(FileExistsError):export_comsol(mesh,path)


def test_validation_and_budgets(native):
    model=from_regions(np.ones((3,3,3),np.uint8))
    with pytest.raises(ValueError):voxel_plc(model,maximum_facets=1)
    with pytest.raises(ValueError):tetgen_mesh(model,surface_method='marching_cubes')
    with pytest.raises(ValueError):TetGenOptions(maximum_volume=-1)
    with pytest.raises(ValueError):TetGenOptions(radius_edge_ratio=float('nan'))
    with pytest.raises(RuntimeError):tetgen_mesh(model,options=TetGenOptions(maximum_cells=1))
    with pytest.raises(ValueError):SurfacePLC(np.zeros((4,3)),[[0,1,2]],{1:[[0,0,0]]})


def test_toml_native_recipe_matches_python(native,tmp_path):
    from morqex.modeling import from_toml,mesh_from_toml,capabilities,mesh
    cfg=Path(__file__).resolve().parents[1]/'configs/geometry_tetgen.toml'
    model=from_toml(cfg);result=mesh_from_toml(model,cfg)
    assert set(result.phases)=={0,1}
    assert capabilities()['tetgen']['compiled']
    assert result.report['region_volume_relative_errors']['1']<1e-10
    # An analytic user PLC also enters the shared mesher registry.
    plc=voxel_plc(from_regions(np.full((2,2,2),3,np.uint8)))
    assert set(mesh(plc,method='tetgen').phases)=={3}


def test_quality_gate_and_unseeded_regions_are_not_silently_accepted(native):
    z,y,x=np.indices((16,)*3);a=((z-7.5)**2+(y-7.5)**2+(x-7.5)**2<6**2).astype(np.uint8)
    from morqex.modeling import TetGenMesh
    with pytest.raises(ArithmeticError,match='quality'):
        tetgen_mesh(from_regions(a),regions=[1],surface_method='marching_cubes',
                    options=TetGenOptions(radius_edge_ratio=1.0,maximum_steiner=0,require_quality=True))
    plc=voxel_plc(from_regions(np.ones((2,2,2),np.uint8)))
    plc.seeds={1:np.array([[100.,100.,100.]])}
    with pytest.raises(ValueError,match='region'):tetgen_mesh(plc)
    a=np.ones((5,5,5),np.uint8);a[2,2,2]=0
    plc=voxel_plc(from_regions(a),regions=[1]);plc.seeds={1:np.array([[2.5e-6]*3])}
    with pytest.raises(ArithmeticError,match='region'):tetgen_mesh(plc)


def test_uniform_display_preserves_actual_cells_and_source_ids(tmp_path,native):
    from morqex.visualization.notebook import MeshView
    model=from_regions(np.ones((4,5,6),np.uint8));mesh=tetgen_mesh(model)
    artifact=mesh.save_hdf5(tmp_path/'tetgen.h5')
    from morqex.modeling import MeshArtifact
    assert MeshArtifact.load(artifact.hdf5)==artifact
    for source in (mesh,artifact):
        view=MeshView(source,cell_count=13,sampling='uniform')
        # The actual payload contains globally sampled element IDs, not a fabricated mesh.
        marker='const D='
        payload=view.document.split(marker)[1].split(',$=')[0]
        data=json.loads(payload);indices=np.linspace(0,len(mesh.cells)-1,13,dtype=np.int64)
        np.testing.assert_array_equal(data['cell_indices'],indices)
        points=np.array(data['points']);cells=np.array(data['cells'])
        np.testing.assert_allclose(points[cells],mesh.points[mesh.cells[indices]])
        np.testing.assert_array_equal(data['phase'],mesh.phases[indices])
        assert data['sampling']=='uniform'
