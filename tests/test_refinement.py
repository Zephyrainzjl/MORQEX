import os
import numpy as np
import pytest
from morqex import Workspace
from morqex.microstructure import Volume
from morqex.mechanics import TetMesh
from morqex.top_damage.backend import NativeConfig
from morqex.top_damage.refinement import RefinementBackend


@pytest.fixture(scope='module')
def backend():
    return RefinementBackend(NativeConfig(),Workspace(os.environ['MORQEX_WORKSPACE']))


def test_refinement_is_conforming_volume_preserving_and_nested(backend):
    labels=np.ones((3,3,3),np.uint8);labels[1:]=2
    mesh=TetMesh.from_volume(Volume(labels,spacing=(1e-6,2e-6,3e-6),origin=(7e-6,3e-6,-2e-6)))
    result=backend.refine(mesh,np.array([0,8,23,90]))
    fine=result.mesh
    assert len(fine.cells)>len(mesh.cells) and len(fine.cells)<8*len(mesh.cells)
    np.testing.assert_allclose(np.bincount(result.parent_cells,weights=fine.volumes),mesh.volumes,rtol=1e-12)
    np.testing.assert_array_equal(fine.phases,mesh.phases[result.parent_cells])
    np.testing.assert_array_equal(fine.voxel_ids,mesh.voxel_ids[result.parent_cells])
    # Hanging interior faces would add spurious interior surface area.
    assert fine.boundary()[1].sum()==pytest.approx(mesh.boundary()[1].sum(),rel=1e-12)
    rng=np.random.default_rng(273)
    scalar=rng.random(len(mesh.points));v=rng.normal(size=(len(mesh.points),3))
    old_gradient=np.einsum('ei,eij->ej',scalar[mesh.cells],mesh.gradients)
    new_scalar=result.nodal_field(scalar)
    new_gradient=np.einsum('ei,eij->ej',new_scalar[fine.cells],fine.gradients)
    np.testing.assert_allclose(new_gradient,old_gradient[result.parent_cells],rtol=1e-11,atol=1e-8)
    np.testing.assert_allclose(result.nodal_field(mesh.points),fine.points,rtol=1e-12,atol=1e-20)
    # Consistent P1 mass, gradient and vector strain integrals are unchanged.
    def mass(m,s):
        values=s[m.cells];return m.volumes@((values.sum(axis=1)**2+(values**2).sum(axis=1))/20)
    assert mass(fine,new_scalar)==pytest.approx(mass(mesh,scalar),rel=1e-12)
    assert new_scalar.min()>=scalar.min()-1e-14 and new_scalar.max()<=scalar.max()+1e-14
    np.testing.assert_allclose(np.einsum('eai,eaj->eij',result.nodal_field(v)[fine.cells],fine.gradients),
        np.einsum('eai,eaj->eij',v[mesh.cells],mesh.gradients)[result.parent_cells],rtol=1e-10,atol=1e-8)
    history=rng.random(len(mesh.cells))
    assert fine.volumes@result.cell_field(history)==pytest.approx(mesh.volumes@history,rel=1e-12)


def test_refinement_does_not_weld_separated_corner_contacts(backend):
    labels=np.zeros((2,2,2),np.uint8);labels[0,0,0]=labels[1,1,1]=1
    mesh=TetMesh.from_volume(Volume(labels,spacing=(1e-6,)*3),solid_phases=[1],separate_point_contacts=True)
    fine=backend.refine(mesh,np.array([0,6])).mesh
    assert mesh.n_components==fine.n_components==2
    assert fine.boundary()[1].sum()==pytest.approx(mesh.boundary()[1].sum(),rel=1e-12)


def test_refinement_rejects_bad_selection(backend):
    mesh=TetMesh.from_volume(Volume(np.ones((1,1,1),np.uint8)))
    for indices in ([],[-1],[100],[1.2],[True]):
        with pytest.raises(ValueError):backend.refine(mesh,indices)
