from dataclasses import replace
import numpy as np
import pytest
from morqex.top_damage.contracts import ConductingNetwork
from morqex.top_damage.reaction_network import ReactionFaces,ReactiveDualNetwork
from morqex.top_damage.reaction_cache import ReactionNetworkCache


def data():
    e=ConductingNetwork(4,np.array([[0,1],[1,2],[1,3]]),np.array([2.,3.,4.]),np.array([0]),'electronic','two-site series test')
    i=ConductingNetwork(3,np.array([[0,1],[0,2]]),np.array([3.,5.]),np.array([0]),'ionic','two-site series test')
    f=ReactionFaces(np.array([2,3]),np.array([1,2]),np.array([.2,.1]),np.array([4.,3.99]),np.array([.3,.4]),'two unequal physical reactions')
    return e,i,f


@pytest.mark.parametrize('factor',[.6,.95,1.001,1.3])
@pytest.mark.parametrize('current',[-.1,.02,0.])
def test_face_flux_interval_contains_independent_full_solution(factor,current):
    e,i,f=data();old=ReactiveDualNetwork(e,i,f,electronic_collector=0,ionic_reservoir=0)
    state=old.solve(.03)
    en=replace(e,conductance_S=e.conductance_S*factor)
    fn=replace(f,equilibrium_V=f.equilibrium_V+np.array([.02,-.01]),exchange_A_m2=f.exchange_A_m2*.8)
    bound=old.changed_problem_bound(state,en,i,faces=fn,temperature_K=310.,
        discharge_current_A=current,include_reaction_bounds=True)
    full=ReactiveDualNetwork(en,i,fn,electronic_collector=0,ionic_reservoir=0,temperature_K=310.,reduce=False).solve(current)
    assert np.all(abs(full.reaction_current_A-state.reaction_current_A)<=bound.reaction_current_absolute_bounds_A+1e-12)
    # The potential bound is for the TARGET OCP, not the previous overpotential.
    target_eta=state.overpotential_V+f.equilibrium_V-fn.equilibrium_V
    assert np.all(abs(full.overpotential_V-target_eta)<=bound.overpotential_absolute_bounds_V+1e-12)


def test_current_reversal_resolves_local_currents_and_keeps_passive_factorization():
    e,i,f=data();cache=ReactionNetworkCache()
    first=cache.evaluate(e,i,f,temperature_K=298.15,discharge_current_A=.03)
    passive=cache.operator.eop
    fn=replace(f,equilibrium_V=f.equilibrium_V+.02,exchange_A_m2=f.exchange_A_m2*.7)
    result=cache.evaluate(e,i,fn,temperature_K=315.,discharge_current_A=-.04)
    assert first.rebuilt and not result.rebuilt and cache.operator.eop is passive
    assert sum(result.state.reaction_current_A)==pytest.approx(.04,abs=1e-11)
    assert not np.allclose(first.state.reaction_current_A,result.state.reaction_current_A)
    assert result.relative_face_current_bound<1e-6


def test_accumulated_change_compares_to_last_factorization_not_last_query():
    e,i,f=data();cache=ReactionNetworkCache(voltage_tolerance_V=1e-4,relative_current_tolerance=.03)
    cache.evaluate(e,i,f,temperature_K=298.15,discharge_current_A=.03)
    references=[];events=[]
    for k in range(1,31):
        target=replace(e,conductance_S=e.conductance_S*(1-.001*k))
        accepted=cache.evaluate(target,i,f,temperature_K=298.15,discharge_current_A=.03)
        full=ReactiveDualNetwork(target,i,f,electronic_collector=0,ionic_reservoir=0,reduce=False).solve(.03)
        assert abs(full.voltage_V-accepted.state.voltage_V)<=1e-4+1e-12
        assert np.sum(abs(full.reaction_current_A-accepted.state.reaction_current_A))<=.03*np.sum(abs(accepted.state.reaction_current_A))+1e-12
        references.append(cache.operator.electronic.conductance_S[0]);events.append(accepted.rebuilt)
    assert any(events) and not all(events)
    assert cache.rebuilds>1 and cache.reuses>1
    assert len(np.unique(references))<len(references)


def test_edge_deletion_rebuilds_and_failed_open_circuit_rolls_back():
    e,i,f=data();cache=ReactionNetworkCache()
    cache.evaluate(e,i,f,temperature_K=298.15,discharge_current_A=.03)
    target=replace(e,conductance_S=np.array([2.,0.,4.]))
    result=cache.evaluate(target,i,f,temperature_K=298.15,discharge_current_A=.03)
    assert result.rebuilt and 'support' in result.reason
    assert abs(result.state.reaction_current_A[0])<1e-12
    accepted=cache.accepted;operator=cache.operator;counts=(cache.rebuilds,cache.reuses)
    with pytest.raises(ValueError,match='open circuit'):
        cache.evaluate(replace(e,conductance_S=np.zeros(3)),i,f,temperature_K=298.15,discharge_current_A=.03)
    assert cache.accepted is accepted and cache.operator is operator and counts==(cache.rebuilds,cache.reuses)


def test_zero_exchange_support_change_forces_rebuild_without_artificial_leakage():
    e,i,f=data();cache=ReactionNetworkCache()
    cache.evaluate(e,i,f,temperature_K=298.15,discharge_current_A=.03)
    accepted=cache.evaluate(e,i,replace(f,exchange_A_m2=np.array([0.,.4])),temperature_K=298.15,discharge_current_A=.03)
    assert accepted.rebuilt and 'reaction support' in accepted.reason
    assert accepted.state.reaction_current_A[0]==0.


def test_common_potential_mode_cannot_create_a_ground_leakage_current():
    e,i,f=data()
    e=replace(e,conductance_S=np.array([.231,.117,.619]))
    f=replace(f,equilibrium_V=np.full(2,4.))
    for reduced in (True,False):
        op=ReactiveDualNetwork(e,i,f,electronic_collector=0,ionic_reservoir=0,reduce=reduced)
        # This intentionally does not fix a gauge: a common potential produces
        # no ohmic current and no BV overpotential anywhere in the network.
        v=np.full(op.S.shape[0],.037,dtype=np.longdouble)
        fields=op._fields(v,0.)
        np.testing.assert_array_equal(fields[3],np.zeros_like(v))
        assert op.passive_nullspace_correction_relative<1e-14
