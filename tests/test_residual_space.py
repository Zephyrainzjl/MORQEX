import numpy as np
from morqex.acceleration.residual_space import minimum_residual_weights

def test_affine_residual_smoothing_has_independent_exact_solution():
    # Orthogonal residuals: the exact affine minimizer is inverse squared norm.
    residuals=[np.eye(7)[k]*v for k,v in enumerate([1.,2.,3.])]
    weights,norm=minimum_residual_weights(residuals,chunk=2);expected=1/np.array([1.,4.,9.]);expected/=expected.sum()
    np.testing.assert_allclose(weights,expected,atol=1e-14)
    assert abs(norm-np.linalg.norm(sum(w*r for w,r in zip(weights,residuals))))<1e-14

def test_physical_linear_solution_and_rank_deficient_snapshot_space():
    rng=np.random.default_rng(3);A=rng.normal(size=(21,21));K=A.T@A+np.eye(21);f=rng.normal(size=21);exact=np.linalg.solve(K,f)
    iterates=[exact+rng.normal(size=21)*.01 for _ in range(3)];iterates.append(iterates[0].copy());residuals=[f-K@u for u in iterates]
    weights,prediction=minimum_residual_weights(residuals,chunk=4);u=sum(w*a for w,a in zip(weights,iterates))
    assert np.linalg.norm(f-K@u)<=min(map(np.linalg.norm,residuals))*(1+1e-12)
    np.testing.assert_allclose(np.linalg.norm(f-K@u),prediction,rtol=1e-12,atol=1e-14)
