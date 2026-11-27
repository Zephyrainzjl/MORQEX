import hashlib,json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from morqex import Workspace
from morqex.top_damage.chemical_stretch import ChemicalStretchTable
from morqex.top_damage.hyperelastic import chemical_potential


def table():
    return ChemicalStretchTable(np.array([0.,.2,.5,.8,1.]),
        np.array([[.98,.98,.985],[.985,.985,1.02],[1.,1.,1.015],[1.004,1.004,1.005],[1.01,1.01,1.]]),
        .5,'Synthetic nonmonotone lattice deformation for derivative verification; not material data')


def test_reference_frame_rotation_and_derivative():
    law=table();R=Rotation.random(3,random_state=7).as_matrix();cmax=50000.;theta=np.array([.1,.37,.9])
    F,dF=law.evaluate(theta,R,cmax)
    reference,_=law.evaluate(np.full(3,.5),R,cmax)
    np.testing.assert_allclose(reference,np.broadcast_to(np.eye(3),(3,3,3)),atol=1e-14,rtol=0)
    assert np.all(np.linalg.det(F)>0)
    eps=.01
    plus=law.evaluate(theta+eps/cmax,R,cmax)[0];minus=law.evaluate(theta-eps/cmax,R,cmax)[0]
    np.testing.assert_allclose(dF,(plus-minus)/(2*eps),atol=2e-13,rtol=2e-7)
    with pytest.raises(ValueError,match='extrapolation'):law.evaluate(np.array([1.01]),R[:1],cmax)


def test_nonlinear_chemical_force_matches_free_energy_derivative():
    law=table();R=Rotation.from_rotvec([.2,-.3,.1]).as_matrix()[None];theta=np.array([.37]);cmax=50000.
    F=np.array([[[1.012,.003,.001],[.002,1.005,.001],[.001,-.002,.998]]])
    G=30e9;lam=40e9
    def energy(c):
        Fc,_=law.evaluate(c/cmax,R,cmax);Fe=F@np.linalg.inv(Fc);logJ=np.log(np.linalg.det(Fe))
        return (G/2*(np.sum(Fe**2,axis=(1,2))-3)-G*logJ+lam/2*logJ**2)[0]
    Fc,dFc=law.evaluate(theta,R,cmax);Fe=F@np.linalg.inv(Fc);logJ=np.log(np.linalg.det(Fe))
    Pe=G*Fe+(-G+lam*logJ)[:,None,None]*np.linalg.inv(Fe).swapaxes(1,2)
    P=Pe@np.linalg.inv(Fc).swapaxes(1,2)
    exact=chemical_potential(P,F,Fc,dFc)[0];h=.1
    numerical=(energy(theta*cmax+h)-energy(theta*cmax-h))/(2*h)
    np.testing.assert_allclose(exact,numerical,rtol=2e-6,atol=1e-6)


def test_table_checksum_material_and_no_arbitrary_extrapolation(tmp_path):
    ws=Workspace(tmp_path);law=table()
    data=dict(schema='morqex-chemical-stretch-1',host='NMC811',stoichiometry=law.stoichiometry.tolist(),
        principal_stretches=law.principal_stretches.tolist(),reference_stoichiometry=.5,provenance=law.provenance)
    path=tmp_path/'stretch.json';path.write_text(json.dumps(data));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    loaded=ChemicalStretchTable.load(ws,'stretch.json',sha,'NMC811')
    np.testing.assert_array_equal(loaded.principal_stretches,law.principal_stretches)
    with pytest.raises(ValueError,match='checksum'):ChemicalStretchTable.load(ws,'stretch.json','0'*64,'NMC811')
    with pytest.raises(ValueError,match='material schema'):ChemicalStretchTable.load(ws,'stretch.json',sha,'LFP')
    with pytest.raises(ValueError,match='positive'):
        ChemicalStretchTable(np.array([0.,.5,1.]),np.zeros((3,3)),.5,'inadmissible')
