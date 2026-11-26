"""Foundation verification; fixtures are synthetic graphs, not fatigue runs."""
import hashlib
import numpy as np
import pytest
from morqex.top_damage import ConductingNetwork, OperatorStamp, PartitionTransfer, TopDamageRegistry
from morqex.extensions import ComponentSpec


def test_break_contact_keeps_disconnected_vertices():
    e = np.array([[0, 1], [1, 2], [2, 3]])
    old = ConductingNetwork(4, e, [1., 1., 1.], np.array([0]), 'electronic', 'explicit synthetic contact')
    new = ConductingNetwork(4, e, [1., 0., 1.], np.array([0]), 'electronic', 'explicit synthetic contact')
    assert old.accessible().all()
    np.testing.assert_array_equal(new.accessible(), [True, True, False, False])
    assert new.components()[0] == 2
    assert old.fingerprint != new.fingerprint
    assert new.vertex_count == old.vertex_count


def test_ionic_access_does_not_imply_electronic_access():
    e = np.array([[0, 1], [1, 2]])
    electronic = ConductingNetwork(3, e, [1., 0.], np.array([0]), 'electronic', 'test')
    ionic = ConductingNetwork(3, e, [0., 1.], np.array([2]), 'ionic', 'test')
    np.testing.assert_array_equal(electronic.accessible() & ionic.accessible(), [False, True, False])


def test_partition_split_preserves_trapped_inventory_and_bounds():
    remap = PartitionTransfer(np.array([0, 0, 1, 1]), np.array([0, 1, 2, 3]), [1e-18, 2e-18, 3e-18, 4e-18])
    r = remap.apply([10., 30.], maximum_mol_m3=50.)
    np.testing.assert_allclose(r.concentration_mol_m3, [10, 10, 30, 30])
    np.testing.assert_allclose(r.inventory_mol, [1e-17, 2e-17, 9e-17, 12e-17], rtol=1e-15, atol=0)
    assert r.after_mol == pytest.approx(24e-17, rel=1e-15, abs=0)
    # Disconnected last two regions still own their lithium; no resetting SOC.
    assert np.sum(r.inventory_mol[2:]) > 0


def test_merge_is_volume_weighted_and_constant_preserving():
    p = PartitionTransfer(np.array([0, 1, 2]), np.array([0, 0, 1]), [1., 3., 2.])
    np.testing.assert_allclose(p.apply([0, 20, 50], maximum_mol_m3=50).concentration_mol_m3, [15, 50])
    np.testing.assert_allclose(p.apply([7, 7, 7], maximum_mol_m3=50).concentration_mol_m3, [7, 7])


@pytest.mark.parametrize('c', [[-1., 2.], [1., 100.], [np.nan, 2.]])
def test_invalid_concentration_rejected(c):
    with pytest.raises(ValueError):
        PartitionTransfer(np.array([0, 1]), np.array([0, 1]), [1., 1.]).apply(c, maximum_mol_m3=10)


def test_loss_of_material_cells_is_not_a_topology_update():
    with pytest.raises(ValueError):
        PartitionTransfer(np.array([0, 0, 1]), np.array([0, 1]), [1., 1., 1.])


def test_operator_rejects_changed_boundary_and_topology():
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    kw = {k: h(k) for k in OperatorStamp.__dataclass_fields__}
    stamp = OperatorStamp(**kw)
    stamp.require_same(OperatorStamp(**kw))
    for key in ['topology', 'boundary', 'materials', 'backend']:
        other = OperatorStamp(**(kw | {key: h(key+'new')}))
        assert other.fingerprint != stamp.fingerprint
        with pytest.raises(ValueError, match='Stale'):
            stamp.require_same(other)


def test_empty_edges_and_no_anchors():
    network = ConductingNetwork(2, np.empty((0, 2), dtype=int), [], np.array([], dtype=int), 'electronic', 'isolated')
    assert network.components()[0] == 2
    assert not network.accessible().any()


def test_no_distance_or_duplicate_contact_guess():
    with pytest.raises(ValueError):
        ConductingNetwork(2, np.array([[0, 1], [1, 0]]), [1., 1.], np.array([0]), 'electronic', 'duplicate')


def test_registry_uses_existing_morqex_component_spec():
    r = TopDamageRegistry().register('state-transfer', ComponentSpec('volume-overlap', PartitionTransfer))
    p = r.create('state-transfer', 'volume-overlap', old_labels=np.array([0]), new_labels=np.array([0]), material_cell_volumes_m3=[1.])
    assert p.apply([3.], maximum_mol_m3=5.).after_mol == 3.
