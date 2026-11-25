"""Independent derivative checks and zero-feedback/full-state integration checks."""
import numpy as np
import pytest
pb=pytest.importorskip('pybamm')
from morqex import Workspace
from morqex.electrochemistry.mechanical_closure import ElectrodeClosure,AxialStackClosure
from morqex.electrochemistry.parameters import convex_parameters
from morqex.electrochemistry.dfn import build_model
from morqex.electrochemistry.validation import check_protocol
from morqex.solvers.pybamm_backend import solver


@pytest.fixture(scope='module')
def setup_stack(tmp_path_factory):
    ws=Workspace(tmp_path_factory.mktemp('dfn'))
    p,_=convex_parameters()
    for d in ['Negative','Positive']:
        key=f'{d} electrode partial molar volume [m3.mol-1]';p.update({key:p[key]*.1})
    a=ElectrodeClosure.from_gemsim(ws,p,'negative',shape=(3,3,3));b=ElectrodeClosure.from_gemsim(ws,p,'positive',shape=(3,3,3),seed=205)
    return p,AxialStackClosure(a,b,p)


def test_condensed_stack_energy_temperature_and_chemical_cross_derivatives(setup_stack):
    p,stack=setup_stack
    dc={'negative':-1000.,'positive':900.};temperature=3.
    def energy(dc,T):
        stress=stack.traction(dc,T);w=.5*stack.separator_length*stress**2/stack.separator_C
        for d,e in stack.electrodes.items():
            h=e.dfn_H;eps=(stress-h[0,1]*dc[d]-h[0,2]*T)/h[0,0];q=np.array([eps,dc[d],T]);w+=stack.lengths[d]*.5*q@h@q
        return w
    for constraint in ['pressure','fixed-thickness']:
        stack=AxialStackClosure(stack.electrodes['negative'],stack.electrodes['positive'],p,constraint=constraint)
        dt=.01
        fdTT=(energy(dc,temperature+dt)-2*energy(dc,temperature)+energy(dc,temperature-dt))/dt**2
        assert fdTT==pytest.approx(stack.energy_TT_per_area,rel=1e-7,abs=1e-8)
        for d,e in stack.electrodes.items():
            plus=dc.copy();minus=dc.copy();plus[d]+=.1;minus[d]-=.1
            cross=(energy(plus,temperature+dt)-energy(plus,temperature-dt)-energy(minus,temperature+dt)+energy(minus,temperature-dt))/(.4*dt)
            assert cross/(stack.lengths[d]*e.dfn_active_fraction)==pytest.approx(stack.mu_T[d],rel=1e-7,abs=1e-8)


def test_zero_feedback_matches_stock_full_dfn_and_thermal_ledger(setup_stack):
    p,stack=setup_stack;results=[]
    for model in [pb.lithium_ion.DFN(options={'thermal':'lumped','heat of mixing':'true'}),build_model(stack,feedback=0.)]:
        sim=pb.Simulation(model,parameter_values=p,var_pts={**model.default_var_pts,'x_n':5,'x_s':5,'x_p':5,'r_n':8,'r_p':8},solver=solver())
        results.append(sim.solve([0.,120.],t_interp=np.linspace(0,120,21)))
    for variable in ['Voltage [V]','Volume-averaged cell temperature [K]','Total lithium [mol]']:
        np.testing.assert_allclose(results[0][variable].entries,results[1][variable].entries,rtol=2e-7,atol=1e-6)
    sol=results[1]
    expected_coefficient=stack.energy_TT_per_area*p['Number of electrodes connected in parallel to make a cell']*p['Electrode width [m]']*p['Electrode height [m]']/p['Cell volume [m3]']
    np.testing.assert_allclose(sol['MORQEX elastic heat-capacity coefficient [J.m-3.K-2]'].entries,expected_coefficient,rtol=1e-12)
    assert max(abs(sol['MORQEX thermal balance residual [J.m-3]'].entries))<.1
    assert np.ptp(sol['Total lithium [mol]'].entries)<1e-12


def test_convexification_is_explicit_and_refuses_unqualified_sets():
    _,audit=convex_parameters();assert audit['electrodes']['negative']['max_ocp_change_V']>.002
    assert audit['electrodes']['negative']['min_negative_slope_V']>0
    with pytest.raises(ValueError):convex_parameters('Ai2020')
    with pytest.raises(ValueError):convex_parameters(slope_floor_V=0)


def test_partial_solver_success_is_not_protocol_completion():
    class Incomplete:cycles=[]
    with pytest.raises(RuntimeError,match='cycle count'):
        check_protocol(Incomplete(),cycles=3,steps_per_cycle=5,lower_V=2.5,upper_V=4.2,current_A=5.)


def test_coupled_segment_restart_preserves_concentration_temperature_and_heat_states(setup_stack,tmp_path):
    import pickle
    p,stack=setup_stack
    def solve(steps,starting=None):
        model=build_model(stack)
        sim=pb.Simulation(model,parameter_values=p,experiment=pb.Experiment(steps,period='10 seconds'),
            var_pts={**model.default_var_pts,'x_n':4,'x_s':4,'x_p':4,'r_n':6,'r_p':6},solver=solver())
        return sim.solve(starting_solution=starting)
    full=solve(['Discharge at 1C for 60 seconds','Rest for 60 seconds'])
    first=solve(['Discharge at 1C for 60 seconds'])
    # Trusted local test checkpoint only; no arbitrary pickle input accepted.
    path=tmp_path/'trusted_solution.pkl';first.save(path)
    with path.open('rb') as f:restored=pickle.load(f)
    resumed=solve(['Rest for 60 seconds'],restored)
    for variable in ['Negative particle concentration [mol.m-3]','Positive particle concentration [mol.m-3]',
        'Volume-averaged cell temperature [K]','MORQEX accumulated thermal heating [J.m-3]','MORQEX accumulated thermal cooling [J.m-3]']:
        np.testing.assert_allclose(full[variable].entries[...,-1],resumed[variable].entries[...,-1],rtol=1e-7,atol=1e-6)
