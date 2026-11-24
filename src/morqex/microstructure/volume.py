"""Voxel contract: array axes (z,y,x), spacing/origin in the same order, SI metres."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

@dataclass
class Volume:
    data: np.ndarray
    spacing: tuple = (1., 1., 1.)
    origin: tuple = (0., 0., 0.)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim != 3 or 0 in self.data.shape:
            raise ValueError('Expected nonempty (z,y,x) volume')
        if len(self.spacing) != 3 or not np.all(np.isfinite(self.spacing)) or min(self.spacing) <= 0:
            raise ValueError('Spacing must contain 3 positive finite values')
        if len(self.origin) != 3 or not np.all(np.isfinite(self.origin)):
            raise ValueError('Origin must contain 3 finite values')

    @property
    def fractions(self):
        ids, counts = np.unique(self.data, return_counts=True)
        return {int(i): float(n / self.data.size) for i, n in zip(ids, counts)}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.npz':
            np.savez_compressed(path, data=self.data, spacing=self.spacing,
                                origin=self.origin, metadata=json.dumps(self.metadata))
        elif path.suffix.lower() in ('.tif', '.tiff'):
            import tifffile
            tifffile.imwrite(path, self.data, photometric='minisblack', metadata={'axes': 'ZYX'})
            path.with_suffix(path.suffix + '.json').write_text(json.dumps(
                dict(spacing=self.spacing, origin=self.origin, metadata=self.metadata)), encoding='utf8')
        else:
            raise ValueError('Use .npz or .tif')
        return path

    @classmethod
    def load(cls, path, **kwargs):
        path = Path(path)
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as f:
                return cls(f['data'], tuple(f['spacing']), tuple(f['origin']), json.loads(str(f['metadata'])))
        import tifffile
        sidecar = path.with_suffix(path.suffix + '.json')
        info = json.loads(sidecar.read_text(encoding='utf8')) if sidecar.exists() else {}
        info.update(kwargs)
        return cls(tifffile.imread(path), **info)
