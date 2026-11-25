"""Interface, geometry, cohesive resolvent and coupled-reservoir qualification."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
pb=pytest.importorskip('pybamm')
from morqex import Workspace,MicrostructureData
from morqex.gemsim import GEMsim
from morqex.extensions import Registry,ComponentSpec,validate_partition,build_operator
from morqex.ct import SegmentedCT,ArrayGeometry
from morqex.topology import PorousCBD
from morqex.bases import GeometricBasis,PortBasis,VariationalOperator
from morqex.cohesive import build_cohesive
from morqex.degradation import (DiffusionLimitedSEI,PlatingStrippingDeadLi,EnergyDrivenDebonding,NoDamage,DegradationOptions,ReactionContext,standard_registry)
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode,phase_transport
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,build_microstructure_dfn
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.electrochemistry.aging_dfn import EnergyKernel,build_aging_dfn,age
from morqex.electrochemistry.aging_cache import save_aging_operator,load_aging_kernel,AgingLocalization
from morqex.electrochemistry.microstructure_validation import thermodynamic_observables,macro_weights
from morqex.solvers.pybamm_backend import solver

class NonconservativeSEI(DiffusionLimitedSEI):
    def evaluate(self,context):
        from dataclasses import replace
        result=super().evaluate(context)
        return replace(result,rates={name:2*value for name,value in result.rates.items()})

@pytest.fixture(scope='module')
def setup_aging(tmp_path_factory):
    ws=Workspace(tmp_path_factory.mktemp('extensions'));p,_=microstructure_parameters()
    labels=np.ones((4,4,4),dtype=np.uint8);labels[0]=0;labels[1]=2
    data=MicrostructureData(labels,(1e-6,)*3,(0.,)*3,{0:'pore',1:'active',2:'cbd'},{'source':'synthetic test'})
    full={};ops={};coh={}
    for d in ['negative','positive']:
        name=f'{d.capitalize()} electrode partial molar volume [m3.mol-1]';p.update({name:p[name]*.1})
        full[d]=MicrostructureElectrode(data,GEMsim(ws),p,d,topology=PorousCBD())
        ops[d]=build_operator(full[d],PortBasis(),VariationalOperator(),4)
        coh[d]=build_cohesive(full[d],ops[d].prolongation)
    return ws,p,data,full,ops,coh

def test_registry_custom_factory_and_duplicate_guard():
    r=standard_registry();assert isinstance(r.create('basis','port'),PortBasis)
    r.register('basis',ComponentSpec('user_geometry',GeometricBasis))
    assert isinstance(r.create('basis','user_geometry'),GeometricBasis)
    assert r.manifest()['basis']['user_geometry']
    with pytest.raises(ValueError,match='Duplicate'):r.register('basis',ComponentSpec('user_geometry',GeometricBasis))
    with pytest.raises(ValueError):r.register('unknown',ComponentSpec('x',GeometricBasis))

def test_ct_explicit_labels_axes_crop_checksum_and_no_resampling(tmp_path):
    import tifffile
    labels=np.zeros((5,6,7),np.uint8);labels[1:]=1;path=tmp_path/'segmented.tif';tifffile.imwrite(path,labels,photometric='minisblack')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    imported=SegmentedCT(path,digest,(1.,2.,3.),{0:'pore',1:'active'},axes_to_zyx=(1,2,0),crop=((1,4),(2,5),(0,3))).build()
    np.testing.assert_array_equal(imported.labels,labels.transpose(1,2,0)[1:4,2:5,0:3]);assert imported.origin_zyx_m==(1.,4.,0.)
    with pytest.raises(ValueError,match='checksum'):SegmentedCT(path,'bad',(1.,)*3,{0:'pore',1:'active'}).build()
    with pytest.raises(ValueError,match='downsampling'):SegmentedCT(path,digest,(1.,)*3,{0:'pore',1:'active'},stride=2).build()

def test_microporous_cbd_adds_ionic_volume_without_changing_phase_labels(setup_aging):
    _,_,data,full,_,_=setup_aging;f=full['negative']
    assert f.porosity==pytest.approx(.25+.25*.47)
    assert f.area.sum()>0
    assert set(np.unique(data.labels))=={0,1,2}
    with pytest.raises(ValueError):PorousCBD(microporosity=.1,ionic_efficiency=.2)

def test_ion_transport_inside_cbd_does_not_require_explicit_resolved_pores(setup_aging):
    ws,p,_,_,_,_=setup_aging;labels=np.ones((4,4,4),np.uint8);labels[0]=2
    data=MicrostructureData(labels,(1e-6,)*3,(0.,)*3,{1:'active',2:'cbd'},{})
    f=MicrostructureElectrode(data,GEMsim(ws),p,'negative',topology=PorousCBD())
    assert f.porosity==pytest.approx(.25*.47) and f.area.sum()>0

def test_phase_corner_touching_islands_do_not_create_cohesive_hinges(setup_aging):
    ws,p,_,_,_,_=setup_aging;labels=np.full((5,5,5),2,np.uint8);labels[0]=0;labels[2,2,2]=1;labels[3,3,2]=1
    data=MicrostructureData(labels,(1e-6,)*3,(0.,)*3,{0:'pore',1:'active',2:'cbd'},{})
    f=MicrostructureElectrode(data,GEMsim(ws),p,'negative',topology=PorousCBD());assert f.diffusion.n_components==2
    op=f.partition(2);c=build_cohesive(f,op.prolongation)
    assert c.metadata['direct_solve_scaled_energy_error']<1e-8

def test_frozen_cohesive_kernel_is_exact_at_declared_damage(setup_aging):
    *_,ops,coh=setup_aging
    for d in ['negative','positive']:
        kernel=EnergyKernel(ops[d].detach(),coh[d],fixed_damage=.4);D,F,_,_=kernel.parts()
        assert len(F)==0;np.testing.assert_allclose(D,coh[d].hessian(.4),rtol=1e-12,atol=1e-7)

def test_weighted_phase_transport_constant_conductivity(setup_aging):
    from morqex.microstructure import Volume
    v=Volume(np.ones((3,3,3),np.uint8),spacing=(1.,)*3)
    assert phase_transport(v,[1],phase_factors={1:.2})['efficiency']==pytest.approx(.2)

def test_partition_guard_rejects_nonconnected_regions(setup_aging):
    *_,full,ops,coh=setup_aging;f=full['negative'];labels=np.arange(f.n);labels[0]=labels[-1]
    _,labels=np.unique(labels,return_inverse=True)
    with pytest.raises(ValueError,match='connected'):validate_partition(f,labels)

def test_cohesive_spectrum_energy_derivative_and_micro_macro_work(setup_aging):
    _,_,_,full,ops,coh=setup_aging;c=coh['negative'];op=ops['negative'];q=np.r_[.001,np.arange(4)*70.,3.]
    assert c.metadata['direct_solve_scaled_energy_error']<1e-9
    for d in [0.,.3,.8]:
        H=c.hessian(d);sc=np.sqrt(np.maximum(abs(H.diagonal()),1e-30));assert np.linalg.eigvalsh(H/sc[:,None]/sc[None,:]).min()>-1e-8
        if d>0:
            h=1e-5;fd=(.5*q@c.hessian(d+h)@q-.5*q@c.hessian(d-h)@q)/(2*h)
            assert fd==pytest.approx(.5*q@c.derivative(d)@q,rel=1e-7)
        field=c.recover(q,d);mean=np.einsum('e,ei->i',c.mesh.volumes/c.mesh.rve_volume,field['stress'])
        assert mean[0]==pytest.approx((H@q)[0],rel=1e-9,abs=1e-5)
        assert -.5*q@c.derivative(d)@q>=-1e-6
    assert not np.allclose(c.recover(q,0.)['opening'],c.recover(q,.8)['opening'],rtol=1e-4,atol=1e-16)

def test_online_cache_excludes_mesh_and_field_maps(setup_aging):
    ws,_,_,_,ops,coh=setup_aging;path=save_aging_operator(ws.path('operators','test'),ops['negative'],coh['negative']);kernel=load_aging_kernel(path)
    assert not hasattr(kernel.cohesive,'mesh') and not hasattr(kernel.operator,'full')
    local=AgingLocalization(path);c=ops['negative'].reference_concentration+np.arange(4)*50.;q=np.r_[.001,c-ops['negative'].reference_concentration,2.]
    direct=coh['negative'].recover(q,.4);fields=local.recover(c,np.array(.001),np.array(2.),np.array(.4))
    for n in ['stress','strain','displacement','opening']:np.testing.assert_allclose(fields[n],direct[n],atol=1e-9)
    file=path/'cohesive_online.npz';file.write_bytes(file.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):load_aging_kernel(path)

@pytest.mark.parametrize('potential',[-.2,0.,.2,.6])
def test_side_laws_reservoir_stoichiometry_and_nonnegative_dissipation(potential):
    state={'sei':pb.Scalar(2.),'plating':pb.Scalar(1.),'dead_lithium':pb.Scalar(0.),'sei_thickness_m':pb.Scalar(5e-9)}
    c=ReactionContext(state,pb.Scalar(potential),pb.Scalar(298.15),pb.Scalar(1000.),1e5,pb.Scalar(1e5),pb.Scalar(96485.33212),pb.Scalar(8.314462618),pb.Scalar(0.),pb.Scalar(0.))
    for law in [DiffusionLimitedSEI(),PlatingStrippingDeadLi()]:
        s=law.evaluate(c);j=float(s.current.evaluate());rate=sum(float(v.evaluate()) for v in s.rates.values())
        assert rate==pytest.approx(-1e5*j/96485.33212,rel=1e-12,abs=1e-12)
        assert j*(potential-s.equilibrium_V)>=-1e-15

def test_custom_reservoir_owner_and_invalid_configs():
    with pytest.raises(ValueError,match='unique'):DegradationOptions(negative=(DiffusionLimitedSEI(),DiffusionLimitedSEI()))
    with pytest.raises(ValueError):EnergyDrivenDebonding(threshold_J_m3=-1)
    with pytest.raises(ValueError):DiffusionLimitedSEI(initial_thickness_m=0.)
    with pytest.raises(ValueError):DiffusionLimitedSEI(diffusivity_m2_s=float('nan'))

def test_seeded_lithium_is_not_duplicated_when_region_count_changes(setup_aging):
    from morqex.electrochemistry.aging_dfn import AgingStates
    _,p,_,full,_,_=setup_aging;native=pb.lithium_ion.DFN(build=False)
    for count in [3,7]:
        ops={d:f.partition(count).detach() for d,f in full.items()};stack=MicrostructureStack(ops['negative'],ops['positive'],p)
        config=DegradationOptions(negative=(PlatingStrippingDeadLi(initial_plated_mol_m3=.5),),damage=NoDamage())
        states=AgingStates(native.param,native.options,stack,config);v=states.get_fundamental_variables();states.set_initial_conditions(v)
        total=sum(float(states.initial_conditions[v[age('negative','plating [mol.m-3]',k)]].evaluate()) for k in range(count))
        assert total==pytest.approx(.5,abs=1e-14)

def test_phase_catalog_anisotropic_fields_and_porous_cbd(setup_aging):
    from morqex.materials import Phase,PhaseCatalog
    from morqex.mechanics import isotropic_stiffness
    ws,p,data,_,_,_=setup_aging;C=isotropic_stiffness(2e10,.25);C[0,0]*=1.2
    beta=(1e-7,2e-7,3e-7,0.,0.,1e-8)
    catalog=PhaseCatalog({0:Phase('electrolyte','pore',ionic_efficiency=1.,electrolyte_fraction=1.),
        1:Phase('oriented-active','active',C,chemical_strain_per_concentration=beta,thermal_strain_per_K=(1e-5,)*3+(0.,)*3,
            electronic_efficiency=.4,diffusion_factor_XYZ=(1.,2.,3.)),
        2:Phase.isotropic('microporous-carbon-binder','cbd',1e9,.3,ionic_efficiency=.05,electronic_efficiency=1.,electrolyte_fraction=.47)})
    full=MicrostructureElectrode(data,GEMsim(ws),p,'negative',materials=catalog);op=full.partition(4)
    np.testing.assert_allclose(full._chemical_tensors[full._selected],np.broadcast_to(beta,(full.n*6,6)))
    assert full.porosity==pytest.approx(.25+.25*.47)
    c=op.reference_concentration+np.arange(4)*30.;q=np.r_[.001,c-op.reference_concentration,2.]
    field=op.recover(c,.001,2.);active=full._selected;cell=full._element_cells[active]
    local=np.bincount(cell,weights=-np.einsum('ei,ei->e',full._chemical_tensors[active],field['stress'][active])*full.mesh.volumes[active],minlength=full.n)/full.diffusion.volumes
    np.testing.assert_allclose(op.restriction@local,op.chemical_potential(c,.001,2.),rtol=1e-9,atol=1e-7)
    assert op.detach().reaction_port_bound(q,np.ones(4),np.zeros(4),300.)['absolute_bound_A_m3']>=0

def test_backend_and_mixed_chemistry_fail_explicitly():
    from morqex.capabilities import REGIONAL_DFN
    from morqex.materials import Phase,PhaseCatalog
    REGIONAL_DFN.require({'porous-CBD','fixed-anisotropic-elasticity'})
    with pytest.raises(ValueError,match='does not support'):REGIONAL_DFN.require({'finite-deformation','friction'})
    labels=np.array([[[0,1,2]]],np.uint8);data=MicrostructureData(labels,(1.,)*3,(0.,)*3,{0:'pore',1:'active',2:'active'},{})
    catalog=PhaseCatalog({0:Phase('pore','pore',ionic_efficiency=1,electrolyte_fraction=1),
        1:Phase.isotropic('graphite','active',1e9,.3,chemistry='graphite'),2:Phase.isotropic('silicon','active',1e9,.3,chemistry='silicon')})
    with pytest.raises(ValueError,match='common host chemistry'):catalog.validate(data)

def test_zero_degradation_limit_matches_existing_coupled_dfn(setup_aging):
    _,p,_,_,ops,_=setup_aging;stack=MicrostructureStack(ops['negative'].detach(),ops['positive'].detach(),p);stack.update_parameters(p)
    config=DegradationOptions(negative=(),damage=NoDamage());kernels={d:EnergyKernel(op.detach()) for d,op in ops.items()};sols=[]
    for model in [build_microstructure_dfn(stack),build_aging_dfn(stack,kernels,config)]:
        sim=pb.Simulation(model,parameter_values=p,solver=solver(),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2})
        sols.append(sim.solve([0,60],t_interp=np.linspace(0,60,7)))
    for n in ['Voltage [V]','Volume-averaged cell temperature [K]','Total lithium [mol]']:
        np.testing.assert_allclose(sols[0][n].entries,sols[1][n].entries,rtol=1e-7,atol=1e-6)

def test_cohesive_degradation_coupled_mass_energy_entropy(setup_aging):
    _,p,_,_,ops,coh=setup_aging;stack=MicrostructureStack(ops['negative'].detach(),ops['positive'].detach(),p);stack.update_parameters(p)
    config=DegradationOptions(damage=EnergyDrivenDebonding(threshold_J_m3=10.,rate_s=1e-7));kernels={d:EnergyKernel(op.detach(),coh[d]) for d,op in ops.items()}
    model=build_aging_dfn(stack,kernels,config);sim=pb.Simulation(model,parameter_values=p,solver=solver(),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2})
    s=sim.solve([0,90],t_interp=np.linspace(0,90,10));audit=thermodynamic_observables(s,stack,p);side=0.;extraenergy=0.
    for law in config.negative:
        for r in law.reservoirs():
            stock=sum(s[age('negative',r.name+' [mol.m-3]',k)].entries for k in range(ops['negative'].dimension));wx=macro_weights(s[age('negative',r.name+' [mol.m-3]',0)])
            A=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Number of electrodes connected in parallel to make a cell']);n=A*stack.lengths['negative']*(wx@stock)
            side+=n;extraenergy+=r.chemical_potential_J_mol*n
    lithium=s['Total lithium [mol]'].entries+side
    assert np.ptp(lithium)/lithium[0]<1e-9
    residual=audit['energy_residual_J']+extraenergy-extraenergy[0]
    assert np.max(abs(residual))<1e-3
    assert audit['metrics']['relative_entropy_residual']<1e-5
    assert max(audit['metrics']['port_identity_relative_residual'].values())<1e-8
    assert np.min(s[age('negative','damage rate [s-1]')].entries)>=0

def test_aging_runtime_checkpoint_preserves_sei_damage_and_host_state(setup_aging):
    from morqex.runtime.aging import AgingRuntime
    ws,p,_,_,ops,coh=setup_aging;paths={d:save_aging_operator(ws.path('operators','restart',d),ops[d],coh[d]) for d in ops}
    config=DegradationOptions(damage=EnergyDrivenDebonding(rate_s=1e-7,threshold_J_m3=10.))
    a=AgingRuntime(paths,p,config);assert not a._localizers;a.advance(30.,5.);checkpoint=a.checkpoint(ws.path('checkpoints','aging'))
    b=AgingRuntime(paths,p,config).restore(checkpoint);sa=a.advance(15.,5.);sb=b.advance(15.,5.)
    assert sa['voltage_V']==pytest.approx(sb['voltage_V'],abs=1e-9)
    for name in [age('negative','sei [mol.m-3]',0),age('positive','damage')]:np.testing.assert_allclose(a.solution[name].entries[:,-1],b.solution[name].entries[:,-1],atol=1e-9)
    assert np.isfinite(a.recover('positive')['opening']).all()
    with pytest.raises(NotImplementedError):a.enrich(paths)
    with pytest.raises(NotImplementedError):a.reaction_bound('negative')

def test_custom_nonconservative_law_is_rejected_and_cannot_restore_normal_checkpoint(setup_aging):
    from morqex.runtime.aging import AgingRuntime
    ws,p,_,_,ops,coh=setup_aging;paths={d:save_aging_operator(ws.path('operators','bad-law',d),ops[d],coh[d]) for d in ops}
    config=DegradationOptions(negative=(NonconservativeSEI(),),damage=NoDamage())
    runtime=AgingRuntime(paths,p,config)
    with pytest.raises(RuntimeError,match='lithium'):runtime.advance(5.,5.)
    assert runtime.solution is None
    good=AgingRuntime(paths,p,DegradationOptions(negative=(DiffusionLimitedSEI(),),damage=NoDamage()))
    assert runtime.fingerprint!=good.fingerprint
