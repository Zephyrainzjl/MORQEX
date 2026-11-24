from types import SimpleNamespace
import numpy as np
import pytest
from morqex.tail import QuantileTree
from morqex.statistics import principal_tension, weighted_quantile, weighted_cvar


def synthetic_response(seed=931, n=73):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(stress_operator=rng.normal(size=(n, 6, 9)), stress_offset=rng.normal(size=(n, 6)),
        volumes=rng.lognormal(size=n), dimension=9)


@pytest.mark.parametrize("probability", [.5, .9, .99])
def test_quantile_interval_cvar_and_exceedance_cover_unequal_volume_truth(probability):
    response = synthetic_response()
    tree = QuantileTree(response)
    rng = np.random.default_rng(19)
    for _ in range(6):
        q = rng.normal(size=9)
        value = principal_tension(np.einsum("eim,m->ei", response.stress_operator, q)+response.stress_offset)
        truth = weighted_quantile(value, response.volumes, probability)
        for budget in (5, 21, 2*len(value)-1):
            result = tree.query(q, probability=probability, maximum_nodes=budget, threshold=3., absolute_tolerance=1e-8)
            assert result.lower-1e-10 <= truth <= result.upper+1e-10
            exact_cvar = weighted_cvar(value, response.volumes, probability)
            assert result.cvar_lower-1e-10 <= exact_cvar <= result.cvar_upper+1e-10
            fraction = response.volumes[value > 3.].sum()/response.volumes.sum()
            assert result.exceedance_lower-1e-12 <= fraction <= result.exceedance_upper+1e-12
            assert result.evaluated_nodes <= budget
        assert result.converged


def test_zero_load_negative_stress_and_region_normalization():
    response = synthetic_response()
    response.stress_operator[:] = 0.
    response.stress_offset[:] = [-10., -5., -2., 0., 0., 0.]
    tree = QuantileTree(response, selection=np.arange(1, 17))
    result = tree.query(np.zeros(9), absolute_tolerance=1e-8)
    assert result.converged and result.evaluated_nodes == 1
    assert result.lower <= -2. <= result.upper
    assert result.exceedance_upper == 0.


def test_budget_failure_is_visible_and_does_not_claim_convergence():
    tree = QuantileTree(synthetic_response())
    result = tree.query(np.ones(9), maximum_nodes=1, absolute_tolerance=0., relative_tolerance=0.)
    assert not result.converged and "budget exhausted" in result.reason
    assert result.upper > result.lower
