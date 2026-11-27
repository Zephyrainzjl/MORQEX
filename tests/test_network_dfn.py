import numpy as np
import pytest
from morqex.top_damage.network_dfn import NetworkEpochDFN
from morqex.top_damage.region_dfn import RegionEpochDFN
from morqex.top_damage.region_port import RegionPort
from morqex.top_damage.epoch_dfn import DFNFeedback
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from morqex.electrochemistry.open_particle import TransportLaw


def make(cls=NetworkEpochDFN,**kwargs):
    a=np.ones((4,4,4),np.uint8);a[0]=0;a[-1]=2
    fv=VoxelDiffusion(Volume(a,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0])
    p,_=lithium_halfcell_parameters(host='NMC811',active_fraction=.5)
    cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
    port=RegionPort.from_diffusion(fv,(fv.coordinates_zyx[:,2]>=2).astype(int),fv.reactive,.5*cmax)
    law=kwargs.pop('transport_law',TransportLaw(gradient_coefficient_J_m5_mol2=1e-14))
    return cls(port,porosity=.25,nx=3,transport_law=law,**kwargs)


def test_undegraded_limit_matches_existing_region_dfn():
    one=make();ref=make(RegionEpochDFN);f=DFNFeedback(10.,.1);mu=np.zeros((3,2))
    a=one.advance_network(10.,.5*one.capacity_Ah,f,mu,reaction_fractions=np.ones((3,2)),edge_factors=np.ones((3,1)))
    b=ref.advance_regions(10.,.5*ref.capacity_Ah,f,mu)
    np.testing.assert_allclose(one.concentrations(),ref.concentrations(),rtol=1e-8)
    assert abs(a['voltage_V']-b['voltage_V'])<1e-8


def test_fragment_retains_lithium_after_topology_event_and_restart():
    one=make();f=DFNFeedback(10.,.1);mu=np.zeros((3,2));area=np.ones((3,2));edge=np.ones((3,1))
    one.advance_network(10.,.2*one.capacity_Ah,f,mu,reaction_fractions=area,edge_factors=edge)
    before=one.concentrations().copy();area[:,0]=0;edge[:]=0
    r=one.advance_network(10.,.2*one.capacity_Ah,f,mu,reaction_fractions=area,edge_factors=edge)
    np.testing.assert_allclose(one.concentrations()[:,0],before[:,0],rtol=1e-12)
    assert np.all(one.concentrations()[:,1]>before[:,1])
    assert r['inventory_relative_defect']<1e-9 and r['relative_state_jump']<1e-10
    assert r['charge_relative_defect']<1e-7 and r['local_port_relative_defect']<1e-10
    for name in ('diffusion dissipation','reaction dissipation'):
        assert np.min(one.last[f'MORQEX positive {name} [W.m-3]'].entries)>-1e-8
    assert np.max(abs(one.last['MORQEX positive port identity residual [W.m-3]'].entries))<1e-7
    restart=make();restart.restore_state(float(one.last.t[-1]),np.asarray(one.last.last_state.y).ravel(),one.last_inputs)
    for runtime in (one,restart):
        runtime.advance_network(10.,-.2*one.capacity_Ah,f,mu,reaction_fractions=area,edge_factors=edge)
    np.testing.assert_allclose(one.concentrations(),restart.concentrations(),rtol=1e-10)
    with pytest.raises(ValueError,match='No reactive'):
        one.advance_network(1.,.2*one.capacity_Ah,f,mu,reaction_fractions=0*area,edge_factors=edge)


def test_reject_invalid_inputs_before_changing_accepted_state():
    one=make();mu=np.zeros((3,2));f=DFNFeedback(10.,.1)
    with pytest.raises(ValueError,match='area fractions'):
        one.advance_network(1.,.1,f,mu,reaction_fractions=np.full((3,2),1.1),edge_factors=np.ones((3,1)))
    assert one.last is None
    with pytest.raises(ValueError,match='Nonnegative'):
        one.advance_network(1.,.1,f,mu,reaction_fractions=np.ones((3,2)),edge_factors=-np.ones((3,1)))


def test_spatial_transport_input_average_and_state_continuity():
    one=make();mu=np.zeros((3,2));f=DFNFeedback(10.,.1)
    efficiency=np.array([.1,.15,.2]);sigma=np.array([8.,10.,12.])
    one.advance_network(2.,.1*one.capacity_Ah,f,mu,reaction_fractions=np.ones((3,2)),edge_factors=np.ones((3,1)),
        electronic_conductivity_S_m=sigma,electrolyte_efficiency=efficiency)
    np.testing.assert_allclose(one.last['Positive electrolyte transport efficiency'].entries[:,0],efficiency)
    np.testing.assert_allclose(one.last['X-averaged positive electrolyte transport efficiency'].entries,.15)


@pytest.mark.parametrize('local',[False,True])
def test_batched_physical_states_equal_public_outputs_and_local_port_identity(local):
    from morqex.electrochemistry.microstructure_dfn import key
    one=make(local_mobility=local);f=DFNFeedback(10.,.1)
    mu=np.tile([10.,-5.],(3,1))
    one.advance_network(3.,.2*one.capacity_Ah,f,mu,reaction_fractions=np.ones((3,2)),edge_factors=np.ones((3,1)))
    public=np.stack([one.last[key('positive',k,'concentration [mol.m-3]')].entries for k in range(2)],axis=1)
    np.testing.assert_allclose(one.concentration_history(one.last),public,rtol=1e-14,atol=1e-10)
    np.testing.assert_allclose(one.concentrations(),public[...,-1],rtol=1e-14)
    assert np.max(abs(one.last['MORQEX positive port identity residual [W.m-3]'].entries))<1e-7
    assert np.min(one.last['MORQEX positive diffusion dissipation [W.m-3]'].entries)>=-1e-8


def test_output_projection_preserves_endpoint_capacity_and_restart():
    raw=make();fixed=make(project_algebraic_output=True)
    f=DFNFeedback(10.,.1);mu=np.tile([10.,-5.],(3,1));area=np.ones((3,2));edge=np.ones((3,1))
    for current in [.2,-.2,.3]:
        records=[]
        for runtime in (raw,fixed):
            records.append(runtime.advance_network(3.,current*runtime.capacity_Ah,f,mu,reaction_fractions=area,edge_factors=edge))
        np.testing.assert_array_equal(raw.last.last_state.y,fixed.last.last_state.y)
        np.testing.assert_array_equal(raw.concentration_history(raw.last),fixed.concentration_history(fixed.last))
        assert records[0]['net_capacity_Ah']==records[1]['net_capacity_Ah']
        assert records[1]['algebraic_output_projection']['sample_count']==3
    restart=make(project_algebraic_output=True)
    restart.restore_state(float(fixed.last.t[-1]),np.asarray(fixed.last.last_state.y).ravel(),fixed.last_inputs)
    for runtime in (fixed,restart):
        runtime.advance_network(3.,.1*runtime.capacity_Ah,f,mu,reaction_fractions=area,edge_factors=edge)
    np.testing.assert_allclose(fixed.last.last_state.y,restart.last.last_state.y,rtol=1e-9,atol=1e-11)


def test_chemical_diffusivity_conversion_has_second_order_fick_limit():
    import pybamm as pb
    one=make(local_mobility=True,diffusivity_interpretation='chemical_diffusivity',transport_law=TransportLaw())
    one.advance_network(1.,.1*one.capacity_Ah,DFNFeedback(10.,.1),np.zeros((3,2)),
        reaction_fractions=np.ones((3,2)),edge_factors=np.ones((3,1)))
    model=one.simulation.built_model;y=np.asarray(one.last.last_state.y).ravel();inputs=one.last_inputs
    J=one._region_output.jac(pb.StateVector(slice(0,len(y)))).evaluate()
    J=J.toarray() if hasattr(J,'toarray') else np.asarray(J)
    initial=np.asarray(one._region_output.evaluate(y=y[:,None])).ravel()
    residuals=[]
    # Stay inside one PCHIP piece. At a knot the table is C1, not C2;
    # a midpoint derivative has only a first-order secant limit there.
    midpoint=.501
    for delta in (40.,20.,10.):
        target=np.r_[np.full(3,one.cmax*midpoint+delta),np.full(3,one.cmax*midpoint-delta)]
        state=y+J.T@np.linalg.solve(J@J.T,target-initial)
        fields=np.asarray(one._region_output.evaluate(y=state[:,None])).ravel()
        np.testing.assert_allclose(fields,target,rtol=1e-14)
        trial=pb.Solution(np.array([one.last.t[-1]]),state[:,None],model,inputs,check_solution=False)
        flux=np.asarray(trial['TOP-DAMAGE diffusion edge 0 1 molar transfer [mol.m-3.s-1]'].entries).ravel()
        D=one.parameters['Positive particle diffusivity [m2.s-1]']
        # Native diffusion functions take stoichiometry, not molar concentration.
        D=float(np.asarray(D(pb.Scalar(midpoint),pb.Scalar(298.15)).evaluate())) if callable(D) else float(D)
        expected=one.edge_weights[0]*D*2*delta
        relative=np.max(abs(flux-expected))/abs(expected)
        residuals.append(relative)
        assert np.all(flux>0)
    assert residuals[-1]<2e-5,residuals
    assert residuals[1]<.35*residuals[0] and residuals[2]<.35*residuals[1],residuals
    with pytest.raises(ValueError,match='local network mobility'):
        make(diffusivity_interpretation='chemical_diffusivity')
