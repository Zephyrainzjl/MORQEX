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
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,build_microstructure_dfn,key
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

class ReversibleContactError(EnergyDrivenDebonding):
    def evaluate(self,damage,release_rate):return -self.rate_s+0*damage

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

def test_cohesive_projection_residual_bounds_and_passive_derivatives(setup_aging):
    from morqex.cohesive_reduction import compress_cohesive,cohesive_certificate
    *_,coh=setup_aging
    original=coh['negative'];reduced=compress_cohesive(original,tolerance=1e-4)
    rng=np.random.default_rng(20261126)
    scale=reduced.certificate_arrays['input_scale']
    assert len(reduced.eigenvalues)<=len(original.eigenvalues)
    for damage in [.03,.29,.61,.89,.99]:
        q=rng.normal(size=len(scale))/scale*100
        certificate=cohesive_certificate(reduced,reduced.certificate_arrays,q,damage)
        difference=reduced.hessian(damage)-original.hessian(damage)
        assert .5*q@difference@q>=-1e-7
        assert .5*q@difference@q<=certificate['energy_error_upper_J_m3']+1e-7
        assert np.all(abs(difference@q)<=certificate['generalized_force_absolute_upper']+1e-4)
        full=original.recover(q,damage)['stress'];approx=reduced.recover(q,damage)['stress']
        assert np.max(np.linalg.norm(full-approx,axis=-1))<=certificate['stress_point_upper_Pa']+1e-3
        assert -.5*q@reduced.derivative(damage)@q>=-1e-7
        h=1e-5
        fd=(reduced.hessian(damage+h)-reduced.hessian(damage-h))/(2*h)
        np.testing.assert_allclose(q@fd@q,q@reduced.derivative(damage)@q,rtol=2e-7,atol=1e-6)
    with pytest.raises(ValueError):compress_cohesive(original,tolerance=0.)
    with pytest.raises(ValueError):cohesive_certificate(reduced,reduced.certificate_arrays,q,1.1)

def test_compression_certificate_cache_is_small_and_tamper_checked(setup_aging):
    from morqex.cohesive_reduction import compress_cohesive
    from morqex.electrochemistry.aging_cache import load_cohesive_certificate
    ws,_,_,_,ops,coh=setup_aging
    reduced=compress_cohesive(coh['positive'],tolerance=1e-4)
    path=save_aging_operator(ws.path('operators','compressed'),ops['positive'],reduced)
    arrays=load_cohesive_certificate(path)
    assert 'stress_modes' not in arrays and 'points' not in arrays
    c=load_aging_kernel(path).cohesive
    np.testing.assert_allclose(c.hessian(.7),reduced.hessian(.7))
    file=path/'certificate.npz';file.write_bytes(file.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):load_cohesive_certificate(path)

def test_dopant_law_provenance_convexity_and_composition_boundaries(setup_aging):
    from morqex.doping import DopantLaw,RegionalChemistry,composition_partition,DopedOperator
    _,_,_,full,ops,_=setup_aging;f=full['positive'];fractions=np.where(f.diffusion.centers[:,0]>np.median(f.diffusion.centers[:,0]),.02,0.)
    labels,profile=composition_partition(f,ops['positive'].partition,fractions)
    for k,value in enumerate(profile):assert np.all(fractions[labels==k]==value)
    law=DopantLaw('X','synthetic unit test',offset_V_per_fraction=.5,slope_V_per_fraction=-.2)
    op=DopedOperator(fractions,law).build(f,labels)
    assert np.array_equal(op.regional_chemistry.fraction,profile)
    with pytest.raises(ValueError):DopantLaw('X','',offset_V_per_fraction=.5)
    with pytest.raises(ValueError):DopantLaw('X','test',slope_V_per_fraction=1.)
    with pytest.raises(ValueError):RegionalChemistry([-1.],law)

@pytest.mark.parametrize('family,source',[('NMC','OKane2022'),('LFP','Prada2013'),('LCO','Ai2020')])
def test_cathode_families_have_independent_sources_and_audited_energy_branch(family,source):
    from morqex.electrochemistry.cathodes import cathode_parameters
    p,audit=cathode_parameters(family)
    assert audit['cathode_family']==family and audit['source_parameter_set']==source
    assert len(audit['parameter_source_sha256'])==64
    for domain in ['negative','positive']:
        entry=audit['compression'][domain]
        assert entry['maximum_value_error_V_polynomial_extrema']<=2e-5
        assert entry['minimum_positive_negative_slope_V_polynomial_extrema']>0
        assert p[f'{domain.capitalize()} electrode OCP entropic change [V.K-1]']==0

def test_dopant_chemical_free_energy_primitive():
    from morqex.doping import DopantLaw,RegionalChemistry
    p=RegionalChemistry([0.,.03],DopantLaw('X','test',offset_V_per_fraction=.4,slope_V_per_fraction=-.2))
    c=15000.;cmax=60000.;h=.01
    fd=(p.primitive_shift(1,c+h,cmax)-p.primitive_shift(1,c-h,cmax))/(2*h)
    assert fd==pytest.approx(p.ocp_shift(1,c/cmax),rel=1e-8)

def test_dopant_material_map_changes_actual_micro_elasticity_and_eigenstrain(setup_aging):
    from morqex.materials import Phase,PhaseCatalog
    from morqex.doping import DopantLaw,composition_materials
    _,_,data,_,_,_=setup_aging
    catalog=PhaseCatalog({0:Phase('pore','pore',ionic_efficiency=1.,electrolyte_fraction=1.),
        1:Phase.isotropic('active','active',1e10,.3,chemical_strain_per_concentration=(1e-7,)*3+(0.,)*3,electronic_efficiency=1.),
        2:Phase.isotropic('CBD','cbd',1e9,.3,electronic_efficiency=1.,ionic_efficiency=.05,electrolyte_fraction=.47)})
    fractions=np.tile([0.,.02],int(np.sum(data.labels==1))//2)
    law=DopantLaw('X','synthetic material sensitivity',log_young_per_fraction=-5.,log_chemical_strain_per_fraction=-3.)
    mapped,materials=composition_materials(data,fractions,law,catalog)
    active=[v for v in materials.phases.values() if v.kind=='active']
    assert len(active)==2 and mapped.provenance['composition_parent_fingerprint']==data.fingerprint
    assert active[1].elasticity[0,0]/active[0].elasticity[0,0]==pytest.approx(np.exp(-.1))
    assert active[1].chemical_strain_per_concentration[0]/active[0].chemical_strain_per_concentration[0]==pytest.approx(np.exp(-.06))

def test_doped_cathode_changes_dfn_and_preserves_energy_inventory(setup_aging):
    from morqex.doping import DopantLaw,RegionalChemistry,attach_doping
    _,p,_,_,ops,coh=setup_aging
    law=DopantLaw('X','synthetic test only',offset_V_per_fraction=.5,slope_V_per_fraction=-.2,log_diffusion_per_fraction=-10.,log_kinetics_per_fraction=-5.)
    solutions=[]
    for fractions in [np.zeros(4),np.array([0.,.02,0.,.02])]:
        parameters=p.copy();stack=MicrostructureStack(ops['negative'].detach(),ops['positive'].detach(),parameters);stack.update_parameters(parameters)
        factories=attach_doping(stack,{'positive':RegionalChemistry(fractions,law)})
        kernels={d:EnergyKernel(op,coh[d]) for d,op in stack.electrodes.items()}
        config=DegradationOptions(negative=(),damage=NoDamage())
        model=build_aging_dfn(stack,kernels,config,factories=factories)
        sim=pb.Simulation(model,parameter_values=parameters,solver=solver(),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2})
        s=sim.solve([0,60],t_interp=np.linspace(0,60,7));audit=thermodynamic_observables(s,stack,parameters)
        assert audit['metrics']['relative_lithium_drift']<1e-9
        assert audit['metrics']['relative_internal_energy_residual']<1e-5
        assert max(audit['metrics']['port_identity_relative_residual'].values())<1e-8
        assert audit['metrics']['minimum_dissipation_W']>=-1e-7
        solutions.append(s)
    assert np.max(abs(solutions[0]['Voltage [V]'].entries-solutions[1]['Voltage [V]'].entries))>1e-4

def test_dopant_profile_roundtrip_survives_operator_cache(setup_aging):
    import copy
    from morqex.doping import DopantLaw,RegionalChemistry
    ws,_,_,_,ops,coh=setup_aging;op=copy.copy(ops['positive'])
    profile=RegionalChemistry([0.,.01,.02,.03],DopantLaw('X','synthetic cache proof'))
    op.regional_chemistry=profile;path=save_aging_operator(ws.path('operators','doped'),op,coh['positive'])
    reloaded=load_aging_kernel(path).operator
    assert reloaded.regional_chemistry.fingerprint==profile.fingerprint

def test_cbd_contact_damage_does_not_scale_intrinsic_host_diffusion(setup_aging):
    from morqex.electrochemistry.aging_dfn import AgingParticle
    _,p,_,_,ops,_=setup_aging;op=ops['positive'].detach();native=pb.lithium_ion.DFN(build=False)
    particle=AgingParticle(native.param,'positive',native.options,operator=op)
    results=[];inputs={f'c{k}':op.reference_concentration+100*k for k in range(op.dimension)}
    for damage in [0.,.6]:
        v={'Positive electrode temperature [K]':pb.Scalar(298.15),age('positive','damage'):pb.Scalar(damage),
            'Positive electrode surface potential difference [V]':pb.Scalar(4.),
            'R-averaged positive particle concentration [mol.m-3]':pb.Scalar(op.reference_concentration)}
        for name in ['side dissipation [W.m-3]','film dissipation [W.m-3]','side current [A.m-3]','reservoir power [W.m-3]']:v[age('positive',name)]=pb.Scalar(0.)
        for k in range(op.dimension):
            v[key('positive',k,'concentration [mol.m-3]')]=pb.InputParameter(f'c{k}')
            for name in ['reaction current [A.m-2]','reaction overpotential [V]','elastic chemical potential [J.mol-1]','elastic chemical potential dT [J.mol-1.K-1]']:v[key('positive',k,name)]=pb.Scalar(0.)
        particle.get_coupled_variables(v)
        results.append(np.array([float(np.asarray(p.process_symbol(v[key('positive',k,'rate [mol.m-3.s-1]')]).evaluate(inputs=inputs)).ravel()[0]) for k in range(op.dimension)]))
    assert np.linalg.norm(results[0])>0
    np.testing.assert_allclose(results[0],results[1],rtol=1e-13,atol=1e-13)

def test_nonuniform_dopant_vanishing_limit_preserves_host_diffusion_discretization(setup_aging):
    from morqex.electrochemistry.aging_dfn import AgingParticle
    from morqex.doping import RegionalChemistry,DopantLaw
    _,p,_,_,ops,_=setup_aging;native=pb.lithium_ion.DFN(build=False);results=[]
    for magnitude in [None,1e-12]:
        op=ops['positive'].detach()
        if magnitude is not None:op.regional_chemistry=RegionalChemistry([0,magnitude,0,magnitude],DopantLaw('X','limit test',offset_V_per_fraction=.5,slope_V_per_fraction=-.2,log_diffusion_per_fraction=-10.))
        particle=AgingParticle(native.param,'positive',native.options,operator=op)
        v={'Positive electrode temperature [K]':pb.Scalar(298.15),age('positive','damage'):pb.Scalar(.3),
            'Positive electrode surface potential difference [V]':pb.Scalar(4.),'R-averaged positive particle concentration [mol.m-3]':pb.Scalar(op.reference_concentration)}
        for name in ['side dissipation [W.m-3]','film dissipation [W.m-3]','side current [A.m-3]','reservoir power [W.m-3]']:v[age('positive',name)]=pb.Scalar(0.)
        inputs={f'c{k}':op.reference_concentration+5000*k for k in range(op.dimension)}
        for k in range(op.dimension):
            v[key('positive',k,'concentration [mol.m-3]')]=pb.InputParameter(f'c{k}')
            for name in ['reaction current [A.m-2]','reaction overpotential [V]','elastic chemical potential [J.mol-1]','elastic chemical potential dT [J.mol-1.K-1]']:v[key('positive',k,name)]=pb.Scalar(0.)
        particle.get_coupled_variables(v)
        results.append(np.array([float(np.asarray(p.process_symbol(v[key('positive',k,'rate [mol.m-3.s-1]')]).evaluate(inputs=inputs)).ravel()[0]) for k in range(op.dimension)]))
    np.testing.assert_allclose(results[0],results[1],rtol=1e-9,atol=1e-10)

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
    moved=a.enrich(paths)
    assert moved.transfer_audit['electrodes']['negative']['reservoir_difference_mol_m3_RVE']<1e-10
    with pytest.raises(NotImplementedError):a.reaction_bound('negative')

def test_aging_nested_enrichment_transfers_side_reservoirs_and_damage(setup_aging):
    from morqex.runtime.aging import AgingRuntime
    ws,p,_,full,ops,coh=setup_aging;oldpaths={};newpaths={}
    for d in ['negative','positive']:
        oldpaths[d]=save_aging_operator(ws.path('operators','enrich_old',d),ops[d],coh[d])
        labels=np.empty(full[d].n,int);offset=0
        for group in range(ops[d].dimension):
            ids=np.flatnonzero(ops[d].partition==group);order=np.argsort(full[d].diffusion.centers[ids,0]);ids=ids[order]
            labels[ids[:len(ids)//2]]=offset;labels[ids[len(ids)//2:]]=offset+1;offset+=2
        op=full[d].reduce(labels,'nested-test');response=build_cohesive(full[d],op.prolongation)
        newpaths[d]=save_aging_operator(ws.path('operators','enrich_new',d),op,response)
    config=DegradationOptions(damage=EnergyDrivenDebonding(rate_s=1e-7,threshold_J_m3=10.))
    a=AgingRuntime(oldpaths,p,config);a.advance(30.,5.)
    before=a.recover('positive');b=a.enrich(newpaths)
    after=b.recover('positive')
    np.testing.assert_allclose(before['concentration'],after['concentration'],atol=1e-9)
    assert before['damage']==pytest.approx(after['damage'],abs=1e-12)
    for audit in b.transfer_audit['electrodes'].values():
        assert audit['reservoir_difference_mol_m3_RVE']<1e-10
        assert audit['relative_elastic_energy_jump']<1e-7
    b.advance(5.,5.)
    checkpoint=b.checkpoint(ws.path('checkpoints','enriched-aging'))
    restart=AgingRuntime(newpaths,p,config).restore(checkpoint)
    assert restart.transfer_audit==b.transfer_audit
    with pytest.raises(ValueError,match='nested'):b.enrich(oldpaths)

def test_shared_protocol_matches_native_cc_rest_cv_cycle(setup_aging):
    from morqex.electrochemistry.protocol import shared_control,add_protocol_event,solve_protocol
    from morqex.electrochemistry.validation import check_protocol
    _,p,_,_,ops,coh=setup_aging;config=DegradationOptions(negative=(),damage=NoDamage());solutions=[]
    for shared in [False,True]:
        parameters=p.copy();stack=MicrostructureStack(ops['negative'].detach(),ops['positive'].detach(),parameters);stack.update_parameters(parameters)
        kernels={d:EnergyKernel(op,coh[d]) for d,op in stack.electrodes.items()}
        model=build_aging_dfn(stack,kernels,config,control=shared_control if shared else None)
        if shared:add_protocol_event(model)
        experiment=None if shared else pb.Experiment([('Discharge at 0.5 A until 3.5 V','Rest for 300 seconds','Charge at 0.5 A until 4.0 V','Hold at 4.0 V until 0.25 A','Rest for 300 seconds')],period='30 seconds')
        sim=pb.Simulation(model,parameter_values=parameters,solver=solver(),experiment=experiment,var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2})
        if shared:s,_=solve_protocol(sim,kind='cycles',current_A=.5)
        else:s=sim.solve(calc_esoh=False)
        check_protocol(s,cycles=1,steps_per_cycle=5,lower_V=3.5,upper_V=4.,current_A=5.,rest_s=300.)
        solutions.append(s)
    for left,right in zip(solutions[0].cycles[0].steps,solutions[1].cycles[0].steps):
        assert left.t[-1]==pytest.approx(right.t[-1],rel=3e-6,abs=.01)
        for name in ['Voltage [V]','Volume-averaged cell temperature [K]']:
            t=np.linspace(max(left.t[0],right.t[0]),min(left.t[-1],right.t[-1]),11)
            a=np.interp(t,left.t,left[name].entries);b=np.interp(t,right.t,right[name].entries)
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-5)

def test_compact_linear_port_identity_copy_broadcast_and_casadi_jacobian():
    import casadi as ca
    from morqex.electrochemistry.linear_port import linear_combination
    x=pb.InputParameter('x');y=pb.InputParameter('y')
    a=linear_combination([2.,-3.],[x,y]);b=linear_combination([3.,-2.],[x,y])
    assert a.id!=b.id
    for p in [a,a.create_copy(perform_simplifications=False)]:
        assert p.evaluate(inputs={'x':2.,'y':4.})==-8.
        assert p.diff(x).evaluate()==2.
        assert p.diff(y).evaluate()==-3.
        u=ca.MX.sym('u',2);z=p.to_casadi(inputs={'x':u[0],'y':u[1]})
        f=ca.Function('port',[u],[z,ca.jacobian(z,u)])
        result,jac=f([2.,4.]);np.testing.assert_allclose(result,-8.);np.testing.assert_allclose(jac,[[2.,-3.]])
    vector=pb.Vector(np.arange(3.)[:,None]);broadcast=linear_combination([2.,3.],[vector,pb.Scalar(1.)])
    np.testing.assert_allclose(broadcast.evaluate(),2*np.arange(3.)[:,None]+3)

def test_custom_nonconservative_law_is_rejected_and_cannot_restore_normal_checkpoint(setup_aging):
    from morqex.runtime.aging import AgingRuntime
    ws,p,_,_,ops,coh=setup_aging;paths={d:save_aging_operator(ws.path('operators','bad-law',d),ops[d],coh[d]) for d in ops}
    config=DegradationOptions(negative=(NonconservativeSEI(),),damage=NoDamage())
    runtime=AgingRuntime(paths,p,config)
    with pytest.raises(RuntimeError,match='lithium'):runtime.advance(5.,5.)
    assert runtime.solution is None
    good=AgingRuntime(paths,p,DegradationOptions(negative=(DiffusionLimitedSEI(),),damage=NoDamage()))
    assert runtime.fingerprint!=good.fingerprint

def test_runtime_rejects_custom_damage_healing_and_preserves_previous_state(setup_aging):
    from morqex.runtime.aging import AgingRuntime
    ws,p,_,_,ops,coh=setup_aging;paths={d:save_aging_operator(ws.path('operators','bad-damage',d),ops[d],coh[d]) for d in ops}
    config=DegradationOptions(negative=(),damage=ReversibleContactError(),initial_damage=.1)
    runtime=AgingRuntime(paths,p,config)
    with pytest.raises(RuntimeError,match='irreversibility'):runtime.advance(5.,.5)
    assert runtime.solution is None
    with pytest.raises(ValueError):AgingRuntime(paths,p,config,nx=2)

def test_runtime_recovery_rejects_a_replaced_operator_cache(setup_aging):
    from morqex.runtime.aging import AgingRuntime
    ws,p,_,_,ops,coh=setup_aging;paths={d:save_aging_operator(ws.path('operators','tamper-live',d),ops[d],coh[d]) for d in ops}
    runtime=AgingRuntime(paths,p,DegradationOptions(negative=(),damage=NoDamage()))
    runtime.advance(2.,.5)
    path=paths['positive']/'manifest.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='cache changed'):runtime.recover('positive')

def test_evolving_reaction_port_bound_covers_implicit_film_and_passive_side_current(setup_aging):
    from scipy.optimize import brentq
    from morqex.error_estimation.aging_port import compile_reaction_certificate,AgingReactionCertificate
    ws,_,_,full,ops,coh=setup_aging;d='negative';op=ops[d];response=coh[d]
    path=save_aging_operator(ws.path('operators','aging-port'),op,response)
    certificate=AgingReactionCertificate(compile_reaction_certificate(path),path)
    rng=np.random.default_rng(20261126);F=96485.33212;RT=8.31446261815324*298.15;alpha=F/(2*RT)
    beta=full[d]._chemical_tensors[full[d]._selected];cells=full[d]._element_cells[full[d]._selected];vol=full[d].mesh.volumes[full[d]._selected]
    for damage in [0.,.47,.85]:
        for _ in range(5):
            q=np.r_[rng.normal()*.002,rng.normal(size=op.dimension)*500.,rng.normal()*2.]
            stress=response.recover(q,damage)['stress'][full[d]._selected]
            mu_fine=np.bincount(cells,weights=-np.einsum('ei,ei->e',beta,stress)*vol,minlength=full[d].n)/full[d].diffusion.volumes
            mu_coarse=(response.hessian(damage)@q)[1:-1]/op.weights
            j0=rng.uniform(.1,3.,op.dimension);resistance=rng.uniform(0,.02,op.dimension);delta=.14;ocp=.08;side_gain=.3
            def currents(mu,k):
                def rhs(z):return 2*j0[k]*np.sinh(alpha*(delta-resistance[k]*z-ocp+mu/F))+side_gain*(delta-resistance[k]*z)
                z=brentq(lambda z:z-rhs(z),-100.,100.)
                return z,2*j0[k]*np.sinh(alpha*(delta-resistance[k]*z-ocp+mu/F))
            coarse=np.array([currents(mu,k) for k,mu in enumerate(mu_coarse)])
            eta=delta-resistance*coarse[:,0]-ocp+mu_coarse/F
            actual=np.zeros(2)
            for i,k in enumerate(op.partition):actual+=full[d].area[i]*(1-damage)*abs(np.array(currents(mu_fine[i],k))-coarse[k])
            bound=certificate.bound(q,damage,j0,eta,298.15,film_resistance=resistance)
            assert actual.max()<=bound['absolute_bound_A_m3']+1e-5
    wrong=path/'reaction_certificate/online.npz';wrong.write_bytes(wrong.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):AgingReactionCertificate(path/'reaction_certificate',path)

def test_runtime_evolving_reaction_bound_uses_only_compact_online_arrays(setup_aging):
    from morqex.runtime.aging import AgingRuntime
    from morqex.error_estimation.aging_port import compile_reaction_certificate
    ws,p,_,_,ops,coh=setup_aging;paths={d:save_aging_operator(ws.path('operators','port-live',d),ops[d],coh[d]) for d in ops}
    for path in paths.values():compile_reaction_certificate(path)
    runtime=AgingRuntime(paths,p,DegradationOptions(damage=EnergyDrivenDebonding(threshold_J_m3=10.,rate_s=1e-7)))
    runtime.advance(30.,.5)
    for d in paths:
        result=runtime.reaction_bound(d)
        assert np.isfinite(result['relative_to_1C']) and result['relative_to_1C']>=0
    assert not runtime._localizers
