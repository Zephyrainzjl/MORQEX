"""Reproducible initial polycrystals and literal voxel inspection artifacts.

The geometry command does not manufacture damage epochs. Future accepted
evolution states must supply their own fields and topology before export.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re
import tomllib
import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage
from morqex.generation.particles import SpherePackSpec, sphere_pack
from morqex.workspace import Workspace
from .polycrystal import PolycrystalData, partition_grains


@dataclass(frozen=True)
class PolycrystalSpec:
    name: str = 'few_large_particles_32'
    shape_zyx: tuple = (32, 32, 32)
    spacing_m: float = 0.5e-6
    particle_count: int = 4
    radius_voxels: float = 6.5
    radius_cv: float = 0.04
    minimum_gap_voxels: float = 1.05
    cbd_fraction: float = 0.14
    grains_per_particle: int = 8
    seed: int = 104
    grain_seed: int = 20261126

    def __post_init__(self):
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', self.name):
            raise ValueError('A single safe output name is required')
        for name in ('particle_count', 'grains_per_particle', 'seed', 'grain_seed'):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in ('grains_per_particle', 'particle_count') else 0):
                raise ValueError('Positive counts and nonnegative integer seeds required')
        if len(self.shape_zyx) != 3 or any(type(n) is not int or n < 8 for n in self.shape_zyx):
            raise ValueError('Three integer dimensions >= 8 required')
        object.__setattr__(self, 'shape_zyx', tuple(self.shape_zyx))
        values = [self.spacing_m, self.radius_voxels, self.radius_cv,
                  self.minimum_gap_voxels, self.cbd_fraction]
        if not np.isfinite(values).all():
            raise ValueError('Finite geometry parameters required')
        self.pack_spec()  # Existing generator validates physical spacing and size.

    def pack_spec(self):
        return SpherePackSpec(shape=self.shape_zyx, particle_count=self.particle_count,
            radius_voxels=self.radius_voxels, radius_cv=self.radius_cv,
            minimum_gap_voxels=self.minimum_gap_voxels, cbd_fraction=self.cbd_fraction,
            spacing_m=(self.spacing_m,) * 3, seed=self.seed)

    @classmethod
    def from_toml(cls, path):
        data = tomllib.loads(Path(path).read_text())
        if set(data) != {'schema_version', 'microstructure'} or data['schema_version'] != 1:
            raise ValueError('Expected schema_version=1 and microstructure table')
        return cls(**data['microstructure'])

    def generate(self):
        structure, particles = sphere_pack(self.pack_spec())
        return partition_grains(structure, particles, grains_per_particle=self.grains_per_particle,
                                seed=self.grain_seed)


def measure_geometry(data):
    labels = data.structure.labels
    spacing = np.asarray(data.structure.spacing_zyx_m)
    grains = np.bincount(data.grain_ids.ravel())[1:]
    particles = np.unique(data.particle_ids[data.particle_ids > 0])
    networks = {}
    for name, kinds in [('resolved_pore', {'pore'}), ('CBD', {'cbd'}),
                        ('electronic_geometry', {'active', 'cbd'})]:
        mask = np.isin(labels, [p for p, k in data.structure.phase_kinds.items() if k in kinds])
        components, count = ndimage.label(mask)
        spans = []
        for axis in (2, 1, 0):
            a = set(np.unique(np.take(components, 0, axis))) - {0}
            b = set(np.unique(np.take(components, -1, axis))) - {0}
            spans.append(bool(a & b))
        networks[name] = {'face_connected_components': count, 'spans_xyz': spans}
    source_spheres = data.structure.provenance.get('particles', [])
    volume_errors = {}
    for particle in source_spheres:
        analytic = 4 * np.pi * particle['radius_m']**3 / 3
        measured = np.sum(data.particle_ids == particle['id']) * np.prod(spacing)
        volume_errors[str(particle['id'])] = float((measured - analytic) / analytic)
    return {'schema': 'morqex-polycrystal-audit-1', 'fingerprint': data.fingerprint,
        'shape_zyx': list(labels.shape), 'spacing_zyx_m': spacing.tolist(),
        'dimensions_xyz_um': (np.asarray(labels.shape)[::-1] * spacing[::-1] * 1e6).tolist(),
        'particle_count': len(particles), 'grain_count': len(grains),
        'minimum_grain_voxels': int(grains.min()),
        'particle_voxel_counts': {str(int(p)): int(np.sum(data.particle_ids == p)) for p in particles},
        'sphere_volume_signed_relative_errors': volume_errors,
        'phase_fractions': {k: float(np.mean(np.isin(labels, [p for p, v in data.structure.phase_kinds.items() if v == k])))
                            for k in sorted(set(data.structure.phase_kinds.values()))},
        'grain_boundary_reference_area_m2': float(data.grain_boundary_faces()['reference_area_m2'].sum()),
        'networks': networks, 'stage': 'initial_geometry_only', 'physical_time_s': 0.0,
        'scope': 'Geometric connectivity, not conductivity or calibrated transport. CBD ionic microporosity is not resolved here. No fatigue evolution, crack aperture, DFN cycle or RVE qualification.'}


def write_polycrystal_vti(data, path):
    """Exact full lattice as VTK cell data; no resampling or surface smoothing."""
    nz, ny, nx = data.structure.labels.shape
    extent = f'0 {nx} 0 {ny} 0 {nz}'
    root = ET.Element('VTKFile', type='ImageData', version='1.0', byte_order='LittleEndian')
    grid = ET.SubElement(root, 'ImageData', WholeExtent=extent,
        Origin=' '.join(map(str, data.structure.origin_zyx_m[::-1])),
        Spacing=' '.join(map(str, data.structure.spacing_zyx_m[::-1])))
    cells = ET.SubElement(ET.SubElement(grid, 'Piece', Extent=extent), 'CellData', Scalars='phase_labels')
    for name, array in [('phase_labels', data.structure.labels), ('particle_ids', data.particle_ids), ('grain_ids', data.grain_ids)]:
        a = ET.SubElement(cells, 'DataArray', type='Int64', Name=name, format='ascii')
        a.text = ' '.join(map(str, array.ravel(order='C')))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return path


def write_geometry_html(data, metrics, path):
    """Offline CPU canvas renderer of actual exposed voxel faces and a slice.

    Geometry arrays are lossless. Particle boundaries are intentionally stepped;
    a smoothed sphere would hide the resolution of the numerical geometry.
    """
    payload = {'shape': list(data.structure.labels.shape),
        'spacing': (np.asarray(data.structure.spacing_zyx_m)[::-1] * 1e6).tolist(),
        'origin': (np.asarray(data.structure.origin_zyx_m)[::-1] * 1e6).tolist(),
        'phase': data.structure.labels.ravel().tolist(),
        'particle': data.particle_ids.ravel().tolist(), 'grain': data.grain_ids.ravel().tolist(),
        'kinds': data.structure.phase_kinds, 'metrics': metrics}
    template = Path(__file__).with_name('morphology_view.html').read_text()
    content = template.replace('__MORQEX_DATA__', json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c'))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def create_polycrystal(config, workspace):
    from .backend import require_local_budget
    require_local_budget()
    spec = PolycrystalSpec.from_toml(config)
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    data = spec.generate()
    directory = ws.path('results', 'top_damage', 'microstructures', spec.name)
    identity = hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()
    if (directory / 'manifest.json').exists():
        previous = json.loads((directory / 'manifest.json').read_text())
        if previous['config_sha256'] != identity or previous['geometry_fingerprint'] != data.fingerprint:
            raise ValueError('Output contains a different geometry/config; choose a new name')
    directory.mkdir(parents=True, exist_ok=True)
    metrics = measure_geometry(data)
    data.save_hdf5(directory / 'initial.h5')
    write_polycrystal_vti(data, directory / 'initial.vti')
    write_geometry_html(data, metrics, directory / 'index.html')
    font = ws.path('data', 'report_font')
    if (font / 'NotoSansCJKsc-Regular.otf').exists() and (font / 'LICENSE').exists():
        from morqex.visualization.report_font import embed_font
        page = directory / 'index.html'
        page.write_text(embed_font(page.read_text(), font / 'NotoSansCJKsc-Regular.otf',
                                   (font / 'LICENSE').read_text()))
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (directory / 'config.toml').write_text(Path(config).read_text())
    files = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
             for name in ('initial.h5', 'initial.vti', 'index.html', 'metrics.json', 'config.toml')}
    result = {'directory': str(directory), 'stage': 'initial_geometry_only',
              'config_sha256': identity, 'geometry_fingerprint': data.fingerprint,
              'files': files, 'config': asdict(spec)}
    (directory / 'manifest.json').write_text(json.dumps(result, indent=2))
    return result
