"""Mass/power duality, direct FEM localization, online state and energy audits."""
from pathlib import Path
import numpy as np
import pytest
pb=pytest.importorskip('pybamm')
from morqex import Workspace
from morqex.gemsim import GEMsim
from morqex.microstructure import Volume
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode,phase_transport
from morqex.electrochemistry.microstructure_cache import OnlineMicrostructureOperator,LocalizationMap
from morqex.electrochemistry.microstructure_dfn import MicrostructureStack,build_microstructure_dfn
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.electrochemistry.microstructure_runtime import MicrostructureDFNRuntime
from morqex.electrochemistry.microstructure_validation import thermodynamic_observables,ocp_primitive
from morqex.solvers.pybamm_backend import solver


@pytest.fixture(scope='module')
def micros(tmp_path_factory):
    ws=Workspace(tmp_path_factory.mktemp('micro_dfn'));g=GEMsim(ws);p,audit=microstructure_parameters();full={};coarse={};fine={}
    for d in ['negative','positive']:
        name=f'{d.capitalize()} electrode partial molar volume [m3.mol-1]';p.update({name:p[name]*.1})
        data=g.random_field(shape=(4,4,4),fractions=(.35,.65),length_scale=(.6,)*3,seed=105,spacing=(2e-6,)*3,phase_kinds={0:'pore',1:'active'})
        full[d]=MicrostructureElectrode(data,g,p,d);coarse[d]=full[d].partition(4);fine[d]=full[d].partition(12)
    return ws,g,p,audit,full,coarse,fine


@pytest.mark.parametrize('axis',[0,1,2])
def test_unit_phase_transport_constant_cube(axis):
    out=phase_transport(Volume(np.ones((3,4,5),dtype=np.uint8),spacing=(2.,3.,4.)),[1],axis)
    assert out['efficiency']==pytest.approx(1.,rel=1e-12)
    assert out['energy_efficiency']==pytest.approx(out['efficiency'],rel=1e-12)
    assert out['relative_residual']<1e-12


def test_parallel_conducting_layers_and_blocked_direction():
    labels=np.ones((4,4,4),dtype=np.uint8);labels[1::2]=0
    v=Volume(labels,spacing=(1.,)*3)
    assert phase_transport(v,[1],0)['efficiency']==pytest.approx(.5)
    assert phase_transport(v,[1],2)['efficiency']==pytest.approx(0.,abs=1e-13)


def test_restriction_prolongation_mass_and_power_duality(micros):
    *_,full,coarse,fine=micros;op=coarse['negative'];fv=full['negative'].diffusion
    np.testing.assert_allclose(op.restriction@op.prolongation,np.eye(op.dimension),atol=1e-14)
    np.testing.assert_allclose(op.weights@op.restriction,fv.weights,atol=1e-16)
    rng=np.random.default_rng(35);mu=rng.normal(size=full['negative'].n);a=rng.normal(size=op.dimension)
    assert (fv.weights*mu)@(op.prolongation@a)==pytest.approx((op.weights*(op.restriction@mu))@a,rel=1e-13)
    assert np.linalg.norm(op.stiffness@np.ones(op.dimension))/np.linalg.norm(op.stiffness.diagonal())<1e-13
    assert np.max((op.stiffness.toarray()-np.diag(op.stiffness.diagonal())))<=0


def test_localization_matches_independent_direct_fem_and_energy_derivative(micros):
    _,_,_,_,full,coarse,_=micros;f=full['negative'];op=coarse['negative'];rng=np.random.default_rng(2026)
    concentration=op.reference_concentration+rng.normal(size=op.dimension)*200.;eps=.001;theta=2.
    field=op.recover(concentration,eps,theta);cell=f._element_cells;selected=f._selected
    eigen=np.zeros((len(cell),6));eigen[selected,:3]=f._omega/3*field['concentration'][cell[selected],None]-f._omega/3*f.reference
    eigen[:,:3]+=f._alpha*theta
    direct=f.solver.solve(eigen,macro_strain=np.r_[eps,np.zeros(5)])
    np.testing.assert_allclose(field['stress'],direct.stress,rtol=1e-10,atol=1e-6)
    np.testing.assert_allclose(field['displacement'],direct.displacement,atol=1e-14)
    assert field['elastic_energy_density']*f.mesh.rve_volume==pytest.approx(direct.elastic_energy,rel=1e-11)
    q=np.r_[eps,concentration-op.reference_concentration,theta]
    for k in range(op.dimension):
        step=np.zeros(len(q));step[k+1]=.01
        fd=(.5*(q+step)@op.H@(q+step)-.5*(q-step)@op.H@(q-step))/.02
        assert fd==pytest.approx(op.weights[k]*op.chemical_potential(concentration,eps,theta)[k],rel=1e-7,abs=1e-6)


def test_direct_regional_construction_matches_full_identity_projection(micros):
    _,_,_,_,full,coarse,_=micros;f=full['negative'];op=coarse['negative'];r=op.dimension
    transform=np.zeros((f.n+2,r+2));transform[0,0]=1;transform[1:-1,1:-1]=op.prolongation;transform[-1,-1]=1
    np.testing.assert_allclose(op.H,transform.T@f.H@transform,rtol=1e-10,atol=1e-7)


def test_nested_state_enrichment_preserves_fields_mass_energy(micros):
    _,_,_,_,_,coarse,fine=micros;a=coarse['negative'];b=fine['negative'];c=a.reference_concentration+np.arange(a.dimension)*50
    updated=a.transfer(b,c)
    np.testing.assert_allclose(a.lift(c),b.lift(updated),atol=1e-10)
    assert a.weights@c==pytest.approx(b.weights@updated,rel=1e-14)
    fa=a.recover(c,.001,3.);fb=b.recover(updated,.001,3.)
    np.testing.assert_allclose(fa['stress'],fb['stress'],rtol=1e-10,atol=1e-5)
    assert fa['elastic_energy_density']==pytest.approx(fb['elastic_energy_density'],rel=1e-12)
    with pytest.raises(ValueError,match='nested'):b.transfer(a,updated)


def test_detached_cache_roi_and_quadratic_statistics(micros):
    ws,_,_,_,full,coarse,_=micros;op=coarse['negative'];path=op.save(ws,'unit_cache');online=OnlineMicrostructureOperator.load(path);local=LocalizationMap(path)
    assert not hasattr(online,'full') and not hasattr(online,'prolongation') and not hasattr(online,'solver')
    c=op.reference_concentration+np.arange(op.dimension)*50.;field=op.recover(c,.001,2.);cached=local.recover(c,.001,2.)
    for key in ['concentration','stress','strain','displacement']:np.testing.assert_allclose(field[key],cached[key],atol=1e-10)
    roi=np.arange(0,len(op.full.mesh.cells),3);np.testing.assert_allclose(local.recover(c,.001,2.,selection=roi)['stress'],field['stress'][roi])
    stats=online.moments(np.r_[.001,c-op.reference_concentration,2.]);v=op.full.mesh.volumes;w=v/v.sum();mean=w@field['stress']
    np.testing.assert_allclose(stats['solid_conditional_mean_stress'],mean,atol=1e-6)
    np.testing.assert_allclose(stats['solid_conditional_variance'],w@((field['stress']-mean)**2),atol=1.,rtol=1e-11)
    with pytest.raises(ValueError):online.H[0,0]=0.


def test_blocked_microstructure_and_corrupt_cache_are_rejected(micros):
    ws,g,p,_,_,coarse,_=micros
    blocked=g.random_field(shape=(4,4,4),fractions=(.35,.65),length_scale=(.6,)*3,seed=104,spacing=(2e-6,)*3,phase_kinds={0:'pore',1:'active'})
    with pytest.raises(ValueError,match='percolation'):MicrostructureElectrode(blocked,g,p,'negative')
    path=coarse['negative'].save(ws,'corrupt_cache');file=path/'online.npz';file.write_bytes(file.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):OnlineMicrostructureOperator.load(path)


def test_compressed_chemistry_retains_positive_curvature_and_value_accuracy(micros):
    audit=micros[3]['compression']
    for record in audit.values():
        assert record['retained_knots']<record['source_knots']
        assert record['maximum_value_error_V_polynomial_extrema']<=2e-5
        assert record['minimum_positive_negative_slope_V_polynomial_extrema']>0


@pytest.mark.parametrize('domain',['negative','positive'])
def test_chemical_energy_primitive_includes_pinned_ocp_endpoint_terms(micros,domain):
    p=micros[2];model=pb.lithium_ion.DFN();phase=model.param.domain_params[domain].prim
    cmax=float(p[f'Maximum concentration in {domain} electrode [mol.m-3]'])
    sto=np.array([.0012,.02,.2,.8,.9988]);h=1e-7*cmax
    derivative=(ocp_primitive(p,domain,sto*cmax+h)-ocp_primitive(p,domain,sto*cmax-h))/(2*h)
    U=np.asarray(p.process_symbol(phase.U(pb.Array(sto),pb.Scalar(298.15))).evaluate()).ravel()
    np.testing.assert_allclose(derivative,U,rtol=1e-6,atol=1e-7)


def test_microstates_replace_radial_states_and_reject_nonideal_energy_claim(micros):
    _,_,p,_,_,coarse,_=micros;stack=MicrostructureStack(coarse['negative'].detach(),coarse['positive'].detach(),p);stack.update_parameters(p)
    model=build_microstructure_dfn(stack)
    assert not any('particle' in domain for v in model.rhs for domain in v.domain)
    assert sum('region' in v.name for v in model.rhs)==sum(op.dimension for op in coarse.values())
    nonideal=p.copy();nonideal.update({'Thermodynamic factor':1.2})
    with pytest.raises(ValueError,match='Ideal electrolyte'):MicrostructureStack(coarse['negative'],coarse['positive'],nonideal)


def test_local_reaction_port_bound_covers_direct_fine_currents_without_online_fine_maps(micros):
    _,_,_,_,full,coarse,_=micros;op=coarse['negative'];online=op.detach();rng=np.random.default_rng(46)
    for _ in range(30):
        dc=rng.uniform(-3000,1000,op.dimension);strain=rng.uniform(-.003,.003);theta=rng.uniform(0,20);T=298.15+theta
        q=np.r_[strain,dc,theta];field=op.recover(op.reference_concentration+dc,strain,theta)
        f=full['negative'];active=f._selected;cell=f._element_cells[active]
        mu_fine=np.bincount(cell,weights=-f._omega/3*f.mesh.volumes[active]*field['stress'][active,:3].sum(axis=1),minlength=f.n)/f.diffusion.volumes
        mu_coarse=op.chemical_potential(op.reference_concentration+dc,strain,theta)
        j0=rng.uniform(.1,3.,op.dimension);eta=rng.uniform(-.1,.1,op.dimension);RT=8.31446261815324*T
        z=96485.33212*eta/(2*RT);change=mu_fine-mu_coarse[op.partition]
        fine_current=2*j0[op.partition]*np.sinh(z[op.partition]+change/(2*RT))
        coarse_current=2*j0*np.sinh(z)
        actual=abs(f.area@fine_current-op.reaction_area@coarse_current)
        bound=online.reaction_port_bound(q,j0,eta,T)['absolute_bound_A_m3']
        assert actual<=bound*(1+1e-12)+1e-7
    assert online.reaction_port_bound(q,j0,eta,T,feedback=0.)['absolute_bound_A_m3']==0.


def test_online_runtime_restart_enrichment_and_global_thermodynamics(micros):
    ws,_,p,_,_,coarse,fine=micros
    paths={d:op.save(ws,'runtime_coarse_'+d) for d,op in coarse.items()};fine_paths={d:op.save(ws,'runtime_fine_'+d) for d,op in fine.items()}
    runtime=MicrostructureDFNRuntime(paths,p,nx=3);assert not runtime._localizers
    runtime.advance(30.,5.);path=runtime.checkpoint(ws.path('checkpoints','unit_micro'))
    bound=runtime.reaction_bound('negative');assert bound['absolute_bound_A_m3']>=0 and not runtime._localizers
    restored=MicrostructureDFNRuntime(paths,p,nx=3).restore(path)
    a=runtime.advance(20.,5.);b=restored.advance(20.,5.)
    assert a['voltage_V']==pytest.approx(b['voltage_V'],abs=1e-10)
    assert a['temperature_K']==pytest.approx(b['temperature_K'],abs=1e-10)
    audit=thermodynamic_observables(runtime.solution,runtime.stack,runtime.parameters)
    assert audit['metrics']['relative_lithium_drift']<1e-10
    assert audit['metrics']['relative_internal_energy_residual']<1e-6
    assert audit['metrics']['relative_entropy_residual']<1e-5
    assert audit['metrics']['minimum_dissipation_W']>=-1e-7
    old=runtime.recover('negative');new=runtime.enrich(fine_paths)
    np.testing.assert_allclose(old['concentration'],new.recover('negative')['concentration'],atol=1e-9)
    assert new.advance(10.,5.)['completed']
    indicator=new.estimate_defect('negative',lookahead_s=10.)
    assert indicator['relative_concentration_defect']>=0 and np.isfinite(indicator['relative_reaction_port_defect'])
    changed=p.copy();changed.update({'Total heat transfer coefficient [W.m-2.K-1]':11.})
    with pytest.raises(ValueError,match='mismatch'):MicrostructureDFNRuntime(paths,changed,nx=3).restore(path)


def test_uniform_zero_feedback_limit_matches_independent_native_dfn_uniform_particles(micros):
    _,g,source,_,_,_,_=micros;p=source.copy();ops={}
    labels=np.ones((4,4,4),dtype=np.uint8);labels[1,1,:]=0
    data=g.from_volume(Volume(labels,spacing=(2e-6,)*3),phase_kinds={0:'pore',1:'active'})
    for d in ['negative','positive']:
        f=MicrostructureElectrode(data,g,p,d)
        assert f.diffusion.n_components==1
        op=f.reduce(np.zeros(f.n,dtype=int));ops[d]=op.detach();D=d.capitalize()
        p.update({f'{D} particle radius [m]':3*op.active_fraction/op.area_density,
            f'{D} electrode Bruggeman coefficient (electrolyte)':np.log(op.transport['electrolyte']['efficiency'])/np.log(op.porosity),
            f'{D} electrode Bruggeman coefficient (electrode)':np.log(op.transport['electrode']['efficiency'])/np.log(1-op.porosity)})
    stack=MicrostructureStack(ops['negative'],ops['positive'],p);stack.update_parameters(p)
    results=[]
    for model in [build_microstructure_dfn(stack,feedback=0.,thermal=False),pb.lithium_ion.DFN(options={'particle':'uniform profile'})]:
        sim=pb.Simulation(model,parameter_values=p,solver=solver(),var_pts={**model.default_var_pts,'x_n':3,'x_s':3,'x_p':3,'r_n':2,'r_p':2})
        results.append(sim.solve([0,180],t_interp=np.linspace(0,180,10)))
    for variable in ['Voltage [V]','Electrolyte concentration [mol.m-3]','R-averaged negative particle concentration [mol.m-3]','R-averaged positive particle concentration [mol.m-3]']:
        np.testing.assert_allclose(results[0][variable].entries,results[1][variable].entries,rtol=3e-7,atol=2e-5)
