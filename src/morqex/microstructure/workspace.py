from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root).expanduser().resolve())
        source = Path(__file__).resolve().parents[3]
        if self.root == source or source in self.root.parents:
            raise ValueError('Workspace must be outside the source package.')

    @classmethod
    def default(cls):
        return cls(Path(os.environ.get('MORQEX_WORKSPACE', Path(__file__).resolve().parents[4] / 'morph_workspace')))

    def path(self, category, *parts):
        if category not in ('data', 'runs', 'outputs', 'reports', 'cache', 'notebooks'):
            raise ValueError(category)
        p = (self.root / category).joinpath(*parts).resolve()
        if self.root not in p.parents:
            raise ValueError('Path escapes workspace')
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def initialize(self):
        for name in ('data', 'runs', 'outputs', 'reports', 'cache', 'notebooks'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self
