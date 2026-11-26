"""Evidence, bounded native builds and explicitly scoped reference simulations."""
import argparse
import json
import os
from morqex.workspace import Workspace


def status():
    return {
        'framework': 'MORQEX / TOP-DAMAGE',
        'stage': 'B/E candidate: full-grid fatigue and conservative DFN port; physical topology remains unqualified',
        'implemented': ['explicit physical contact graphs', 'fixed-material conservative repartition',
                        'operator dependency identities', 'MORQEX ComponentSpec registry',
                        'vendored MFEM source and TOML build API', 'irreversible AT2 subproblem',
                        'variational isotropic split mechanics', 'accepted-step fatigue history',
                        'HDF5 restart and VTU/HTML mechanical specimen output',
                        'tracked 3D particle/grain geometry and TOML voxel inspection',
                        'AMG equilibrium and active-set CG with direct-reference checks',
                        'conservative Li-metal DFN with persistent state and cutoff reversal',
                        '32-cubed diffuse-fatigue/effective-conductivity feedback candidate',
                        'MFEM finite-strain chemical elasticity and compiled FP64 tangent action',
                        'conservative regional DFN states with spatial mechanical-potential ports',
                        'TOML per-node finite-mechanics feedback probe and field visualization',
                        'accepted-state VTI/XDMF time series and bounded-memory HTML'],
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
    region=sub.add_parser('region-probe',help='TOML full-voxel regional DFN -> finite mechanics -> spatial feedback probe')
    region.add_argument('config')
    region.add_argument('--workspace',default=os.environ.get('MORQEX_WORKSPACE'))
    region.add_argument('--stage',choices=['all','prepare','node','feedback','report'],default='all')
    region.add_argument('--node',type=int)
    geometry = sub.add_parser('geometry', help='Generate tracked initial polycrystals and full voxel HTML/HDF5/VTI')
    geometry.add_argument('config')
    geometry.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    dense = sub.add_parser('dense-cycle', help='Experimental full-grid diffuse-fatigue/DFN candidate; physical topology remains gated')
    dense.add_argument('config')
    dense.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    dense.add_argument('--environment', choices=['gentle','fast','released'], default='gentle')
    dense.add_argument('--steps', type=int, default=1)
    dense.add_argument('--output')
    dense_report = sub.add_parser('dense-report', help='Check accepted histories, optionally export VTI/XDMF, and publish HTML')
    dense_report.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    dense_report.add_argument('--cases', nargs='+', default=['gentle_voltage_v3','fast_voltage_v3','released_voltage_v3'])
    dense_report.add_argument('--output', default='MORQEX_TOP_DAMAGE_voltage_cycles.html')
    dense_report.add_argument('--export', action='store_true')
    build = sub.add_parser('build', help='Compile/resume a bounded MFEM backend slice from TOML')
    build.add_argument('config')
    build.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    finite_build = sub.add_parser('build-hyperelastic', help='Build finite-strain adapter against the verified MFEM library')
    finite_build.add_argument('config')
    finite_build.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
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
    elif args.action == 'region-probe':
        if not args.workspace:p.error('An external workspace is required')
        from .region_probe import run_probe
        print(json.dumps(run_probe(args.config,Workspace(args.workspace),stage=args.stage,node=args.node),indent=2))
    elif args.action == 'build-hyperelastic':
        if not args.workspace:p.error('An external workspace is required')
        from .backend import NativeConfig
        from .hyperelastic import build_hyperelastic
        print(json.dumps(build_hyperelastic(NativeConfig.from_toml(args.config),Workspace(args.workspace)),indent=2))
    elif args.action == 'dense-report':
        if not args.workspace:p.error('An external workspace is required')
        from .backend import require_local_budget
        from .verify_dense_history import verify_dense_history
        from .dense_export import export_dense_history
        from .dense_report import build_dense_report
        require_local_budget();ws=Workspace(args.workspace);audits=[]
        for name in args.cases:
            folder=ws.path('results/top_damage/dense_cycles',name)
            audits.append(verify_dense_history(folder))
            (folder/'numerical_validation.json').write_text(json.dumps(audits[-1],indent=2))
            if args.export:export_dense_history(folder)
        report=build_dense_report(ws,args.cases,output_name=args.output)
        audit_path=report.with_suffix('.validation.json')
        audit_path.write_text(json.dumps(audits,indent=2))
        print(json.dumps({'report':str(report),'validation':str(audit_path),'cases':audits},indent=2))
    elif args.action == 'dense-cycle':
        if not args.workspace:p.error('An external workspace is required')
        from .dense_cycle import run_dense_cycle
        print(json.dumps(run_dense_cycle(args.config,Workspace(args.workspace),environment=args.environment,steps=args.steps,output=args.output),indent=2))
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
