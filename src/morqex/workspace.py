"""Keep every generated artifact outside the source tree."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self):
        root = Path(self.root).expanduser().resolve()
        container = Path(__file__).resolve().parents[1]
        source = (container.parent if container.name == 'src' and
                  (container.parent / 'pyproject.toml').is_file() else container)
        if root == source or source in root.parents:
            raise ValueError("Workspace must be outside MORQEX source")
        object.__setattr__(self, "root", root)

    def path(self, *parts: str) -> Path:
        target = self.root.joinpath(*parts).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("Path escapes workspace")
        return target

