import numpy as np
import pybamm as pb
import pytest
from morqex.top_damage.region_port import RegionPort
from morqex.top_damage.region_dfn import RegionEpochDFN
from morqex.top_damage.epoch_dfn import DFNFeedback
from morqex.microstructure import Volume
from morqex.transport import VoxelDiffusion
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from morqex.electrochemistry.open_particle import TransportLaw

def test_region_mass_local_potential_and_actual_microstates():
    a=np.ones((4,4,4),np.uint8);a[0]=0;a[-1]=2
    fv=VoxelDiffusion(Volume(a,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0])
    p,_=lithium_halfcell_parameters(host='NMC811',active_fraction=.5)
    cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
    labels=(fv.coordinates_zyx[:,2]>=2).astype(int)
    port=RegionPort.from_diffusion(fv,labels,fv.reactive,.5*cmax)
    np.testing.assert_allclose(port.restrict(port.lift([2.,3.])),[2,3])
    np.testing.assert_allclose(port.weights@port.restrict(np.arange(len(labels))),fv.weights@np.arange(len(labels)))
    model=RegionEpochDFN(port,porosity=.25,nx=3,transport_law=TransportLaw(gradient_coefficient_J_m5_mol2=1e-14))
    assert not any('radial' in v.name.lower() for v in model.model.rhs)
    assert sum('region' in v.name and 'concentration' in v.name for v in model.model.rhs)==2
    feedback=DFNFeedback(10.,.1)
    r=model.advance_regions(5.,.1*model.capacity_Ah,feedback,np.zeros((3,2)))
    assert r['inventory_relative_defect']<1e-9
    before=model.concentrations();assert before.shape==(3,2)
    r=model.advance_regions(5.,.1*model.capacity_Ah,feedback,np.array([[0,100],[0,150],[0,200]]))
    assert r['relative_state_jump']<1e-9 and r['inventory_relative_defect']<1e-9
    assert np.max(abs(model.concentrations()[:,0]-model.concentrations()[:,1]))>1e-6
    for field in ['diffusion dissipation','reaction dissipation']:
        assert np.min(model.last[f'MORQEX positive {field} [W.m-3]'].entries)>-1e-7
    assert np.max(abs(model.last['MORQEX positive port identity residual [W.m-3]'].entries))<1e-7
    assert port.field(model.concentrations()).shape==(3,4,4,4)

def test_region_ports_reject_invented_connectivity_and_double_diffusivity():
    a=np.ones((4,4,4),np.uint8);a[0]=0;a[-1]=2
    volume=Volume(a,spacing=(1e-6,)*3)
    fv=VoxelDiffusion(volume,active_phases=[1],diffusivity=1.,reactive_phases=[0])
    labels=fv.coordinates_zyx[:,2]%2
    with pytest.raises(ValueError,match='disconnected'):
        RegionPort.from_diffusion(fv,labels,fv.reactive,100.)
    physical=VoxelDiffusion(volume,active_phases=[1],diffusivity=1e-14,reactive_phases=[0])
    with pytest.raises(ValueError,match='Unit geometric'):
        RegionPort.from_diffusion(physical,np.zeros(len(physical.weights),int),physical.reactive,100.)
    with pytest.raises(ValueError,match='reactive faces'):
        RegionPort.from_diffusion(fv,np.zeros(len(fv.weights),int),np.ones_like(fv.reactive),100.)


def test_mixed_spatial_input_postprocessing_and_empty_input():
    model=pb.BaseModel();a=pb.Variable('a');v=pb.InputParameter('vector',expected_size=3)
    scalar=pb.InputParameter('scalar');model.rhs={a:-a+scalar+sum(pb.Index(v,k) for k in range(3))};model.initial_conditions={a:0}
    model.variables={'mixed input result':a+scalar+sum(pb.Index(v,k) for k in range(3))}
    sim=pb.Simulation(model,solver=pb.CasadiSolver(mode='fast'))
    sol=sim.solve([0,.1],inputs={'scalar':2.,'vector':np.array([1.,2.,3.])})
    expected=8*(1-np.exp(-.1))+8
    np.testing.assert_allclose(sol['mixed input result'].entries[-1],expected,rtol=1e-5)
    bare=pb.BaseModel();a=pb.Variable('a');bare.rhs={a:-a};bare.initial_conditions={a:1};bare.variables={'a':a}
    clean=pb.Simulation(bare).solve([0,.1]);assert clean.all_inputs_stacked[0].size==0


@pytest.mark.parametrize('host',['NMC532','NMC811','LFP','LCO'])
def test_region_material_interfaces_and_restart(host):
    a=np.ones((3,3,3),np.uint8);a[0]=0;a[-1]=2
    fv=VoxelDiffusion(Volume(a,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0])
    p,_=lithium_halfcell_parameters(host=host,active_fraction=1/3)
    cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
    port=RegionPort.from_diffusion(fv,(fv.coordinates_zyx[:,2]>0).astype(int),fv.reactive,.5*cmax)
    law=TransportLaw(gradient_coefficient_J_m5_mol2=1e-14)
    one=RegionEpochDFN(port,host=host,porosity=1/3,nx=3,transport_law=law)
    feedback=DFNFeedback(10.,.1);mu=np.zeros((3,2))
    one.advance_regions(2.,.05*one.capacity_Ah,feedback,mu)
    restart=RegionEpochDFN(port,host=host,porosity=1/3,nx=3,transport_law=law)
    restart.restore_state(float(one.last.t[-1]),np.asarray(one.last.last_state.y).ravel(),one.last_inputs)
    first=one.advance_regions(2.,.05*one.capacity_Ah,feedback,mu)
    second=restart.advance_regions(2.,.05*restart.capacity_Ah,feedback,mu)
    np.testing.assert_allclose(one.concentrations(),restart.concentrations(),rtol=1e-11)
    assert abs(first['voltage_V']-second['voltage_V'])<1e-10
    assert max(first['inventory_relative_defect'],second['inventory_relative_defect'])<1e-9
    with pytest.raises(TypeError,match='no radial'):one.radial_profile()

def test_projection_is_work_conjugate_and_failed_step_rolls_back():
    a=np.ones((4,4,4),np.uint8);a[0]=0;a[-1]=2
    fv=VoxelDiffusion(Volume(a,spacing=(1e-6,)*3),active_phases=[1],diffusivity=1.,reactive_phases=[0])
    p,_=lithium_halfcell_parameters(host='NMC811',active_fraction=.5)
    cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
    port=RegionPort.from_diffusion(fv,(fv.coordinates_zyx[:,2]>=2).astype(int),fv.reactive,.5*cmax)
    rng=np.random.default_rng(44);mu=rng.normal(size=len(fv.weights));dc=rng.normal(size=2)
    np.testing.assert_allclose((port.weights*port.project_mechanical_potential(mu))@dc,(fv.weights*mu)@port.lift(dc),atol=1e-16)
    one=RegionEpochDFN(port,porosity=.25,nx=3,transport_law=TransportLaw(gradient_coefficient_J_m5_mol2=1e-14))
    feedback=DFNFeedback(10.,.1);mu=np.zeros((3,2));current=.1*one.capacity_Ah
    one.advance_regions(2.,current,feedback,mu);before=one.last;sim_before=one.simulation._solution
    initial=one.initial_inventory;one.initial_inventory*=1.1
    with pytest.raises(RuntimeError,match='lithium ledger'):one.advance_regions(2.,current,feedback,mu)
    assert one.last is before and one.simulation._solution is sim_before
    one.initial_inventory=initial;result=one.advance_regions(2.,current,feedback,mu)
    np.testing.assert_allclose(result['time_s'],4.)

def test_region_probe_configuration_contract(tmp_path):
    from morqex.top_damage.region_probe import RegionProbeConfig
    from dataclasses import replace
    cfg=RegionProbeConfig()
    for kwargs in ({'nodes':2},{'requested_regions':0},{'output':'../outside'}, {'C_rate':0}, {'length_m':float('nan')}, {'cbd_ionic_efficiency':.9}):
        with pytest.raises(ValueError):replace(cfg,**kwargs)
    path=tmp_path/'probe.toml';path.write_text('schema_version = 1\n[probe]\nnodes = 3\nrequested_regions = 8\n')
    assert RegionProbeConfig.read(path).nodes==3
    path.write_text('schema_version = 2\n[probe]\n')
    with pytest.raises(ValueError,match='schema'):RegionProbeConfig.read(path)
