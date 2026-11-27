import numpy as np
import pybamm as pb
import pytest
from morqex.top_damage.dae_projection import AlgebraicOutputProjection


def test_manufactured_constraint_reconstruction_keeps_inventory_and_events():
    model=pb.BaseModel();x=pb.Variable('inventory');z=pb.Variable('potential')
    control=pb.InputParameter('control')
    model.rhs={x:-x};model.algebraic={z:z-x*x-control}
    model.initial_conditions={x:1.,z:1.+control};model.variables={'inventory':x,'potential':z}
    model.events=[pb.Event('inventory threshold',x-.5)]
    sim=pb.Simulation(model,solver=pb.IDAKLUSolver(rtol=1e-9,atol=1e-11))
    sol=sim.solve([0.,2.],inputs={'control':.2},t_interp=np.linspace(0,2,30))
    y=np.array(sol.y);original=y.copy();y[1,1:-1]+=.01
    corrupted=pb.Solution(sol.t,y,sim.built_model,sol.all_inputs[0],
        t_event=sol.t_event,y_event=sol.y_event,termination=sol.termination,check_solution=False)
    saved_root=sim.built_model.algebraic_root_solver
    projector=AlgebraicOutputProjection()
    fixed,audit=projector(corrupted)
    assert sim.built_model.algebraic_root_solver is saved_root
    np.testing.assert_array_equal(np.asarray(fixed.y)[0],original[0])
    np.testing.assert_array_equal(np.asarray(fixed.y)[:,[0,-1]],original[:,[0,-1]])
    np.testing.assert_allclose(np.asarray(fixed.y)[1,1:-1],original[0,1:-1]**2+.2,rtol=0,atol=1e-11)
    np.testing.assert_array_equal(corrupted.y,y)
    assert fixed.termination==sol.termination
    np.testing.assert_array_equal(fixed.t_event,sol.t_event)
    np.testing.assert_array_equal(fixed.y_event,sol.y_event)
    assert audit.sample_count==len(sol.t)-2 and audit.maximum_algebraic_residual_before>.009
    again,_=projector(corrupted)
    np.testing.assert_array_equal(again.y,fixed.y)
    with pytest.raises(ValueError):AlgebraicOutputProjection(tolerance=0.)
