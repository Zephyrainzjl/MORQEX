import numpy as np
import pytest
from morqex.top_damage.contracts import ConductingNetwork
from morqex.top_damage.reaction_network import ReactionFaces,ReactiveDualNetwork


def series(reduce=True,exchange=.3):
    e=ConductingNetwork(3,np.array([[0,1],[1,2]]),np.array([2.,4.]),np.array([0]),'electronic','series analytical benchmark')
    i=ConductingNetwork(3,np.array([[0,1],[1,2]]),np.array([3.,5.]),np.array([2]),'ionic','series analytical benchmark')
    faces=ReactionFaces(np.array([2]),np.array([0]),np.array([.2]),np.array([4.]),np.array([exchange]),'one known physical reaction face')
    return ReactiveDualNetwork(e,i,faces,electronic_collector=0,ionic_reservoir=2,reduce=reduce)


@pytest.mark.parametrize('current',[-.2,-.02,0.,.02,.2])
def test_series_network_matches_closed_form_bv_and_resistance(current):
    reduced=series();result=reduced.solve(current)
    R=1/2+1/4+1/3+1/5
    activation=np.arcsinh(current/(2*.2*.3))/reduced.k
    np.testing.assert_allclose(result.voltage_V,4-current*R-activation,atol=1e-11,rtol=0)
    np.testing.assert_allclose(result.reaction_current_A,[-current],atol=1e-12,rtol=1e-10)
    np.testing.assert_allclose(result.electronic_dissipation_W+result.ionic_dissipation_W,current**2*R,atol=1e-12)
    np.testing.assert_allclose(result.reaction_dissipation_W,current*activation,atol=1e-12)
    assert result.relative_charge_residual<1e-9 and abs(result.power_defect_W)<1e-11
    full=series(False).solve(current)
    np.testing.assert_allclose(result.electronic_potential_V,full.electronic_potential_V,atol=1e-11)
    np.testing.assert_allclose(result.ionic_potential_V,full.ionic_potential_V,atol=1e-11)


def test_floating_reactive_particle_has_zero_net_current_but_can_redistribute():
    e=ConductingNetwork(7,np.array([[0,1],[1,2],[3,4],[4,5]]),np.array([2.,4.,1.,3.]),np.array([0]),'electronic','isolated particle plus inert gauge')
    i=ConductingNetwork(3,np.array([[0,1],[1,2]]),np.array([3.,5.]),np.array([2]),'ionic','connected electrolyte')
    f=ReactionFaces(np.array([2,3,5]),np.array([0,0,0]),np.array([.2,.1,.1]),np.array([4.,4.1,3.9]),np.array([.3,.3,.3]),'three physical reaction faces')
    states=[ReactiveDualNetwork(e,i,f,electronic_collector=0,ionic_reservoir=2,reduce=r).solve(.03) for r in (True,False)]
    one,full=states
    assert abs(one.reaction_current_A[1])>1e-3
    assert abs(one.reaction_current_A[1:].sum())<1e-11
    np.testing.assert_allclose(one.reaction_current_A[0],-.03,rtol=1e-9)
    np.testing.assert_allclose(one.reaction_current_A,full.reaction_current_A,atol=1e-11)
    np.testing.assert_allclose(one.electronic_potential_V,full.electronic_potential_V,atol=1e-10)
    assert one.reaction_dissipation_W>0


def test_no_reaction_path_is_open_circuit_not_a_finite_voltage():
    op=series(exchange=0.)
    with pytest.raises(ValueError,match='open circuit'):op.solve(.01)
    assert not op.solve(0.).voltage_defined


def changed(network,factors):
    return ConductingNetwork(network.vertex_count,network.edges,
        network.conductance_S*np.asarray(factors),network.anchors,network.channel,'controlled conductance perturbation')


@pytest.mark.parametrize('epsilon',[.001,.05,.4,.8])
@pytest.mark.parametrize('current',[-.2,.02])
def test_nonlinear_voltage_bound_contains_fresh_full_graph_solution(epsilon,current):
    old=series();state=old.solve(current)
    e=changed(old.electronic,[1-epsilon,1+epsilon/2])
    i=changed(old.ionic,[1+epsilon,1-epsilon/2])
    bound=old.changed_conductance_bound(state,e,i)
    new=ReactiveDualNetwork(e,i,old.faces,electronic_collector=0,ionic_reservoir=2,reduce=False).solve(current)
    error=abs(new.voltage_V-state.voltage_V)
    assert error>0
    assert error<=bound.voltage_absolute_bound_V+1e-12
    assert bound.relative_edge_change==pytest.approx(epsilon)


def test_voltage_bound_rejects_support_changes_and_undefined_voltage():
    old=series();state=old.solve(.02)
    with pytest.raises(ValueError,match='support'):
        old.changed_conductance_bound(state,changed(old.electronic,[0,1]),old.ionic)
    with pytest.raises(ValueError,match='>=1'):
        old.changed_conductance_bound(state,changed(old.electronic,[2.1,1]),old.ionic)
    with pytest.raises(ValueError,match='physically defined'):
        series(exchange=0).changed_conductance_bound(series(exchange=0).solve(0),old.electronic,old.ionic)


def test_high_conductance_floating_voxel_graph_has_no_line_search_stagnation():
    from morqex.microstructure import Volume
    from morqex.transport import VoxelDiffusion
    from morqex.top_damage.reaction_network import voxel_reaction_problem
    labels=np.zeros((6,6,10),np.uint8);particles=np.zeros_like(labels,dtype=np.int32)
    labels[0,:2,:]=2
    for particle,x in enumerate((1,6),1):
        labels[2:4,2:4,x:x+2]=1;particles[2:4,2:4,x:x+2]=particle
        labels[0:2,2,x]=2
    fv=VoxelDiffusion(Volume(labels,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0],particle_ids=particles)
    faces=np.ones((3,)+labels.shape);faces[0,1,2,1]=0.
    model,*_=voxel_reaction_problem(fv,np.where(labels==1,10.,np.where(labels==2,100.,0.)),
        (labels==0).astype(float),np.full(16,4.),2.8,electronic_face_factors=faces)
    e=changed(model.electronic,1+.05*np.cos(np.arange(len(model.electronic.edges))*.37))
    i=changed(model.ionic,1+.05*np.cos(np.arange(len(model.ionic.edges))*.37+2))
    states=[ReactiveDualNetwork(e,i,model.faces,electronic_collector=int(e.anchors[0]),
        ionic_reservoir=int(i.anchors[0]),reduce=r,max_ports=512).solve(1.4071455142276663e-10) for r in (False,True)]
    assert max(s.relative_charge_residual for s in states)<1e-9
    assert max(s.iterations for s in states)<20
    np.testing.assert_allclose(states[0].voltage_V,states[1].voltage_V,atol=1e-10,rtol=0)
