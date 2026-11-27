"""Independent discrete identities for the new monolithic particle adapter."""
import numpy as np
import pytest
pytest.importorskip('skfem')
from morqex.top_damage.paper_particle import PaperParticle, ParticleMaterial, ParticleState


def model(feedback=True,**kwargs):
    from skfem import MeshTri
    mesh=MeshTri.init_tensor(np.linspace(0,1,5),np.linspace(0,1,5))
    return PaperParticle(mesh,ParticleMaterial(length_m=1e-6),stress_feedback=feedback,**kwargs)


@pytest.mark.parametrize('form',['paper_divergence','author_helmholtz'])
def test_monolithic_full_jacobian(form):
    m=model(phase_form=form,toughness_floor=.01);rng=np.random.default_rng(441)
    u=rng.normal(0,.002,2*m.n);c=rng.uniform(.3,.7,m.n);d=rng.uniform(.1,.3,m.n)
    H=np.full(m.nq,1e-6);a=np.full(m.nq,m.threshold*2)
    old=ParticleState(u,c,d,H,a,np.zeros(m.nq),0.)
    x=np.r_[u,c,d];v=rng.normal(size=len(x));v[:2*m.n]*=.01;v/=np.linalg.norm(v)
    r,J,_=m.residual(x,old,3.,1e-6)
    h=1e-6
    rp,_=m.residual(x+h*v,old,3.,1e-6,jacobian=False)
    rm,_=m.residual(x-h*v,old,3.,1e-6,jacobian=False)
    fd=(rp-rm)/(2*h);pred=J@v
    for s in (slice(0,2*m.n),slice(2*m.n,3*m.n),slice(3*m.n,None)):
        assert np.linalg.norm(fd[s]-pred[s])/np.linalg.norm(fd[s])<2e-7


def test_conservative_no_flux_and_flux_step():
    m=model();old=m.initial_state(crack_fraction=0)
    before=[v.copy() for v in (old.u,old.c,old.d,old.history,old.fatigue)]
    same,check=m.advance(old,1.,0.)
    np.testing.assert_allclose(same.c,old.c,atol=1e-12)
    state,check=m.advance(old,1.,1e-7)
    assert check['mass_error']<1e-7 and state.time_s==1
    assert np.mean(state.c)<np.mean(old.c)
    for a,b in zip(before,(old.u,old.c,old.d,old.history,old.fatigue)):np.testing.assert_array_equal(a,b)


def test_stress_feedback_changes_transport_but_not_inventory():
    models=[model(True),model(False)];states=[]
    for m in models:
        state,check=m.advance(m.initial_state(crack_fraction=0),20.,1e-6)
        states.append(state);assert check['mass_error']<1e-7
    assert np.max(np.abs(states[0].c-states[1].c))>1e-6
    np.testing.assert_allclose(models[0].lump@states[0].c,models[1].lump@states[1].c,rtol=1e-10)


def test_invalid_material_and_step():
    with pytest.raises(ValueError):ParticleMaterial(diffusivity_m2_s=-1)
    with pytest.raises(ValueError):ParticleMaterial(poisson=.5)
    m=model()
    with pytest.raises(ValueError):m.advance(m.initial_state(crack_fraction=0),-1.,0.)


def test_surface_cutoff_records_actual_lithium_not_clipped_concentration():
    from skfem import MeshTri
    for soc,flux in ((.01,1e-5),(.99,-1e-5)):
        m=PaperParticle(MeshTri.init_tensor(np.linspace(0,1,7),np.linspace(0,1,7)),
                        ParticleMaterial(length_m=1e-6),boundary_policy='surface_limits')
        old=m.initial_state(soc=soc,crack_fraction=0)
        s,r=m.advance(old,30.,flux,tolerance=1e-10)
        assert r['surface_limited_nodes']>0
        assert 0<r['delivered_flux_ratio']<1
        assert r['mass_error']<1e-7
        assert s.c.min()>=-1e-10 and s.c.max()<=1+1e-10
        prescribed=-flux*m.surface_area*30/(m.material.cmax_mol_m3*m.material.radius_m)
        np.testing.assert_allclose(m.lump@(s.c-old.c),prescribed*r['delivered_flux_ratio'],rtol=1e-7)


def test_exact_crack_measure_includes_axisymmetric_weight():
    from skfem import MeshTri
    mesh=MeshTri.init_tensor(np.linspace(0,1,5),np.linspace(0,1,5))
    for geometry in ('sphere','cylinder'):
        m=PaperParticle(mesh,ParticleMaterial(length_m=1e-6),geometry=geometry)
        # d=r: region r>t occupies 1-t^2 in axisymmetry and 1-t in plane strain.
        for threshold in (.11,.5,.95):
            expected=1-threshold**(2 if geometry=='sphere' else 1)
            assert abs(m.cracked_fraction(mesh.p[0],threshold)-expected)<1e-13


def test_chord_preserves_coupled_solution_and_failed_trial_history():
    states=[]
    for algorithm in ('newton','chord'):
        m=model(linearization=algorithm)
        old=m.initial_state(crack_fraction=0)
        for _ in range(3):
            new,record=m.advance(old,2.,1e-6,tolerance=1e-11)
            assert record['mass_error']<1e-7
            assert np.min(new.fatigue-old.fatigue)>=0
            assert np.min(new.d-old.d)>-1e-10
            old=new
        states.append(old)
        accepted=old.fatigue.copy()
        with pytest.raises(RuntimeError):m.advance(old,1.,1e-5,max_iterations=0)
        np.testing.assert_array_equal(old.fatigue,accepted)
    for field in ('u','c','d','fatigue'):
        np.testing.assert_allclose(getattr(states[0],field),getattr(states[1],field),atol=2e-10,rtol=1e-7)


def test_torn_history_append_is_recovered_but_corruption_is_not_hidden(tmp_path):
    from morqex.top_damage.paper_particle_run import _recover_history
    path=tmp_path/'history.jsonl'
    path.write_text('{"step":1}\n{"step":2')
    assert _recover_history(path)==[{'step':1}]
    assert path.read_text().endswith('\n')
    path.write_text('{"step":1}\n{"step":2}')
    assert len(_recover_history(path))==2 and path.read_text().endswith('\n')
    path.write_text('{"step":1}\ncorrupt\n{"step":2}\n')
    with pytest.raises(ValueError,match='Corrupt'):_recover_history(path)
    path.write_text('{"step":1}\n{"step":3}\n')
    with pytest.raises(ValueError,match='Noncontiguous'):_recover_history(path)
