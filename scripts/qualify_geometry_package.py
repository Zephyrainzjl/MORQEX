"""Build and test an isolated wheel from an external source snapshot.

Use scripts/run_local.py for CPU/RAM enforcement. Neither compiler products,
installed packages nor source-distribution archives are written into the repo.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile
from source_snapshot import snapshot


SMOKE = r'''
import json,sys
from pathlib import Path
import numpy as np
import morqex
from morqex.modeling import (from_regions,controlled_tetgen,MeshResolution,
    TetGenOptions,export_bundle,import_comsol,RemeshOptions,remesh)
target=Path(sys.argv[1]);output=Path(sys.argv[2]);names=json.loads(sys.argv[3])
assert Path(morqex.__file__).resolve().is_relative_to(target)
model=from_regions(np.pad(np.ones((4,4,4),np.uint8),2),spacing=(1e-6,)*3)
base=model.tetrahedralize()
bundle=export_bundle(base,output/'software',formats=['comsol','vtu','gmsh','medit','abaqus'],
    region_names=model.region_names,length_unit='um')
assert bundle['passed']
result={'imported_package':str(Path(morqex.__file__).resolve()),'exchange':bundle,'native':{}}
if 'tetgen' in names:
    from morqex.modeling.tetgen_backend import library_path
    assert library_path().resolve().is_relative_to(target)
    counts=[]
    for volume in (10e-18,.05e-18):
        native=controlled_tetgen(model,resolution=MeshResolution(maximum_volume_m3=volume),
            surface_method='voxel_plc',tetgen_options=TetGenOptions(maximum_cells=100000))
        assert native.signed_volumes().min()>0
        counts.append(len(native.cells))
    assert counts[1]>counts[0]
    result['native']['tetgen']={'library':str(library_path()),'coarse_fine_cells':counts}
if 'acvd' in names:
    from morqex.modeling.acvd_backend import load_acvd
    module=load_acvd();assert Path(module.__file__).resolve().is_relative_to(target)
    result['native']['acvd']={'library':module.__file__}
if 'mmg' in names:
    from morqex.modeling.mmg_backend import library_path
    from morqex.modeling import TetGenMesh,material_interfaces
    assert library_path().resolve().is_relative_to(target)
    f,m,pairs=material_interfaces(base.cells,base.phases)
    base=TetGenMesh(base.points,base.cells,base.phases,facets=f,facet_markers=m,
        report={'geometry':{'interface_pairs':pairs}})
    adapted=remesh(base,options=RemeshOptions(uniform_edge_m=3e-6,
        hausdorff_m=1e-8,preserve_surface=True,maximum_phase_volume_error=1e-8))
    assert adapted.signed_volumes().min()>0
    result['native']['mmg']={'library':str(library_path()),'cells':len(adapted.cells),
        'phase_volume_errors':adapted.report['phase_volume_errors_relative_to_input_mesh']}
(output/'installed_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'installed_smoke_passed':True,'backends':names}),flush=True)
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=os.environ.get('MORQEX_WORKSPACE'))
    parser.add_argument('--backends',nargs='*',choices=['tetgen','acvd','mmg'],default=[])
    parser.add_argument('--sdist',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.workspace is None:parser.error('Specify --workspace or MORQEX_WORKSPACE')
    if len(set(args.backends))!=len(args.backends):parser.error('List each backend once')
    work=args.workspace.resolve()
    if work==root or root in work.parents:raise ValueError('External workspace required')
    record=snapshot(root,work)
    folder=work/'package_qualification'/(record['tree_sha256'][:20]+'_'+('_'.join(args.backends) or 'pure')+'_'+str(time.time_ns()))
    frozen=folder/'source';frozen.mkdir(parents=True)
    with tarfile.open(record['archive']) as archive:archive.extractall(frozen,filter='data')
    wheels=folder/'wheels';wheels.mkdir();env=os.environ.copy()
    for name in ('GEOMETRY','TETGEN','ACVD','MMG'):env.pop('MORQEX_BUILD_'+name,None)
    for name in args.backends:env['MORQEX_BUILD_'+name.upper()]='1'
    subprocess.run([sys.executable,'-m','pip','wheel',str(frozen),'--no-deps','--no-build-isolation',
        '--wheel-dir',str(wheels)],env=env,check=True)
    wheel,=wheels.glob('*.whl')
    required=['morqex/modeling/vtu_stream.py','morqex/modeling/medit_stream.py',
        'morqex/visualization/adaptive_mesh.html','morqex/vendor/mmg/COPYING.LESSER',
        'morqex/vendor/mmg/src/mmg3d/libmmg3d.c','morqex/vendor/mmg/provenance.json']
    with zipfile.ZipFile(wheel) as archive:
        for name in required:
            if archive.read(name)!=(frozen/'src'/name).read_bytes():raise AssertionError('Wheel/source mismatch: '+name)
        files=archive.namelist()
        for name in args.backends:
            if not any('/_'+name+'_native.' in p and p.endswith(('.so','.pyd')) for p in files):raise AssertionError('Native backend absent: '+name)
        if 'mmg' in args.backends and not any(p.endswith(('libmmg3d.so.5','libmmg3d.5.dylib','mmg3d.dll')) for p in files):raise AssertionError('Replaceable MMG runtime absent')
    source_distribution=None
    if args.sdist:
        subprocess.run([sys.executable,'-c','import build_backend; build_backend.build_sdist('+repr(str(wheels))+')'],cwd=frozen,env=env,check=True)
        source_distribution,=wheels.glob('*.tar.gz')
        with tarfile.open(source_distribution) as archive:
            names=archive.getnames()
            for name in ('configs/native_backends.toml','scripts/run_local.py','tutorials/07_complex_ct_layers_and_exchange.md','tests/test_partition_workflow.py',
                         'provenance/integration_state.json','provenance/pybamm_migration.json','literature/references.tsv'):
                if not any(p.endswith('/'+name) for p in names):raise AssertionError('Incomplete source distribution: '+name)
            extracted=folder/'sdist_source';extracted.mkdir();archive.extractall(extracted,filter='data')
        source_root,=extracted.iterdir()
        subprocess.run([sys.executable,str(source_root/'scripts/check_integrity.py')],env=env,check=True)
    installed=folder/'installed'
    subprocess.run([sys.executable,'-m','pip','install',str(wheel),'--no-deps','--no-compile','--target',str(installed)],env=env,check=True)
    env['PYTHONPATH']=str(installed)
    for key in ('MORQEX_MMG_LIBRARY','MORQEX_TETGEN_LIBRARY'):env.pop(key,None)
    subprocess.run([sys.executable,'-c',SMOKE,str(installed),str(folder),json.dumps(args.backends)],cwd=folder,env=env,check=True)
    result={'passed':True,'source_snapshot':record,'wheel':str(wheel),
        'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),
        'sdist':str(source_distribution) if source_distribution else None,
        'installed_smoke':str(folder/'installed_smoke.json'),'backends':args.backends}
    (folder/'qualification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
