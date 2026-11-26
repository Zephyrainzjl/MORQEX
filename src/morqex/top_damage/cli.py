"""Evidence, bounded native builds and explicitly scoped reference simulations."""
import argparse
import json
import os
from morqex.workspace import Workspace


def status():
    return {
        'framework': 'MORQEX / TOP-DAMAGE',
        'stage': 'B foundation: compiled AT2 reference; full battery fatigue closure remains unqualified',
        'implemented': ['explicit physical contact graphs', 'fixed-material conservative repartition',
                        'operator dependency identities', 'MORQEX ComponentSpec registry',
                        'vendored MFEM source and TOML build API', 'irreversible AT2 subproblem',
                        'variational isotropic split mechanics', 'accepted-step fatigue history',
                        'HDF5 restart and VTU/HTML mechanical specimen output',
                        'tracked 3D particle/grain geometry and TOML voxel inspection'],
        'not_qualified': ['polycrystal fatigue reference', 'phase-field-to-aperture topology update',
                          'dynamic tensor homogenization with fracture', 'topology-changing DFN restart',
                          'full long-cycle closure', 'cycle-jump error control', 'GPU fatigue backend'],
        'neural_models': False,
        'can_predict_lifetime': False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog='morqex top-damage')
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('status')
    geometry = sub.add_parser('geometry', help='Generate tracked initial polycrystals and full voxel HTML/HDF5/VTI')
    geometry.add_argument('config')
    geometry.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    build = sub.add_parser('build', help='Compile/resume a bounded MFEM backend slice from TOML')
    build.add_argument('config')
    build.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    reference = sub.add_parser('reference', help='Bounded 3D mechanical specimen run, not DFN lifetime')
    reference.add_argument('config')
    reference.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    reference.add_argument('--steps', type=int, default=8)
    reference.add_argument('--resume', action='store_true')
    reference.add_argument('--output')
    report = sub.add_parser('report')
    report.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    args = p.parse_args(argv)
    if args.action == 'status':
        print(json.dumps(status(), indent=2, ensure_ascii=False))
    elif args.action == 'geometry':
        if not args.workspace:
            p.error('An external workspace is required')
        from .morphology import create_polycrystal
        print(json.dumps(create_polycrystal(args.config, Workspace(args.workspace)), indent=2))
    elif args.action == 'build':
        if not args.workspace:
            p.error('An external workspace is required')
        from .backend import NativeConfig, build_backend
        print(json.dumps(build_backend(NativeConfig.from_toml(args.config), Workspace(args.workspace)), indent=2))
    elif args.action == 'reference':
        if not args.workspace:
            p.error('An external workspace is required')
        from .reference_run import run_reference
        print(json.dumps(run_reference(args.config, Workspace(args.workspace), steps=args.steps,
                                      resume=args.resume, output=args.output), indent=2))
    else:
        if not args.workspace:
            p.error('An external --workspace or MORQEX_WORKSPACE is required')
        from .report import build_report
        print(build_report(Workspace(args.workspace)))


if __name__ == '__main__':
    main()
