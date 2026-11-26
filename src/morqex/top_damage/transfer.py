"""Conservative repartition on a fixed material reference mesh.

Cracking can disconnect lithium-bearing material; it does not delete it.
This transfer covers partition changes ONLY, not moving/remeshed geometry.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, diags
from .contracts import _owned


@dataclass(frozen=True)
class TransferResult:
    concentration_mol_m3: np.ndarray
    inventory_mol: np.ndarray
    before_mol: float
    after_mol: float
    defect_mol: float


class PartitionTransfer:
    """Volume-overlap remap: c_new=diag(V_new)^-1 O c_old.

    Both partitions cover the SAME ordered material cells, including isolated
    cells. This is a convex volume projection, not energy preservation and not
    a proof of transport closure. Cell IDs outside material are never passed.
    """
    def __init__(self, old_labels, new_labels, material_cell_volumes_m3):
        old, new = np.asarray(old_labels), np.asarray(new_labels)
        v = np.asarray(material_cell_volumes_m3, dtype=float)
        for labels in (old, new):
            if labels.ndim != 1 or not len(labels) or labels.dtype.kind not in 'iu' or np.any(labels < 0):
                raise ValueError('Nonnegative integer region per material cell required')
            if not np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))):
                raise ValueError('Regions must be contiguous and nonempty')
        if old.shape != new.shape or v.shape != old.shape or not np.isfinite(v).all() or np.any(v <= 0):
            raise ValueError('Same material cells with finite positive physical volumes required')
        self.old_volumes_m3 = _owned(np.bincount(old, weights=v))
        self.new_volumes_m3 = _owned(np.bincount(new, weights=v))
        overlap = coo_matrix((v, (new, old)), shape=(len(self.new_volumes_m3), len(self.old_volumes_m3))).tocsr()
        self._matrix = diags(1/self.new_volumes_m3) @ overlap
        self.old_labels = _owned(old, np.int64)
        self.new_labels = _owned(new, np.int64)

    def apply(self, concentration_mol_m3, *, maximum_mol_m3):
        c = np.asarray(concentration_mol_m3, dtype=float)
        maximum = float(maximum_mol_m3)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError('Finite positive concentration maximum required')
        if c.shape != self.old_volumes_m3.shape or not np.isfinite(c).all() or np.any(c < 0) or np.any(c > maximum):
            raise ValueError('Finite physically bounded intrinsic concentrations required')
        mapped = np.asarray(self._matrix @ c)
        before = float(self.old_volumes_m3 @ c)
        stocks = self.new_volumes_m3 * mapped
        after = float(stocks.sum())
        tol = 128*np.finfo(float).eps*max(abs(before), abs(after), np.finfo(float).tiny)
        if abs(after-before) > tol:
            raise ArithmeticError('Repartition changed lithium inventory')
        return TransferResult(_owned(mapped), _owned(stocks), before, after, after-before)
