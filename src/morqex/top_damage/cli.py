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
                        'source-integrated resolved 3D echemAMR/HYPRE and exact field-to-fatigue bridge',
                        'accepted-state VTI/XDMF time series and bounded-memory HTML',
                        'source-aware Kron graph condensation and passive-response event certificate',
                        'dual electronic/ionic reachability with immutable disconnected lithium inventory',
                        'spatial graph/area/diffusion DFN inputs and per-node finite-fatigue cycle checkpoints',
                        'nested connected concentration aggregation and full-voxel reference option'],
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
    network_cycle=sub.add_parser('network-cycle',help='Conservative regional DFN / 3D finite fatigue / dual graph candidate')
    network_cycle.add_argument('config')
    network_cycle.add_argument('--workspace',default=os.environ.get('MORQEX_WORKSPACE'))
    network_cycle.add_argument('--steps',type=int,default=1)
    network_report=sub.add_parser('network-report',help='Audit accepted network history and render offline fields/voltage')
    network_report.add_argument('directory')
    network_report.add_argument('--workspace',default=os.environ.get('MORQEX_WORKSPACE'))
    network_report.add_argument('--output')
    network_report.add_argument('--export',action='store_true')
    resolved=sub.add_parser('build-resolved',help='Bounded source-integrated 3D echemAMR build; separate from DFN qualification')
    resolved.add_argument('config')
    resolved.add_argument('--workspace',default=os.environ.get('MORQEX_WORKSPACE'))
    resolved_pf=sub.add_parser('resolved-phase-field',help='TOML exact resolved field -> one accepted finite AT2 history state')
    resolved_pf.add_argument('config')
    resolved_pf.add_argument('--workspace',default=os.environ.get('MORQEX_WORKSPACE'))
    resolved_report=sub.add_parser('resolved-report',help='Actual full-voxel electrochemistry and finite-fatigue HTML')
    resolved_report.add_argument('directory')
    resolved_report.add_argument('--workspace',default=os.environ.get('MORQEX_WORKSPACE'))
    resolved_report.add_argument('--output',default='MORQEX_resolved_3D_phase_field.html')
    resolved_report.add_argument('--export',action='store_true')
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
    particle = sub.add_parser('paper-particle', help='Bounded monolithic Ai-2022 particle equation experiment')
    particle.add_argument('config')
    particle.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    particle.add_argument('--steps', type=int, default=20)
    particle.add_argument('--resume', action='store_true')
    particle.add_argument('--output')
    particle_report = sub.add_parser('paper-report', help='Accepted particle fields and explicit reproduction gates')
    particle_report.add_argument('directory')
    particle_report.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    particle_report.add_argument('--reference')
    particle_report.add_argument('--output', default='MORQEX_Ai2022_reproduction.html')
    paper_reference = sub.add_parser('paper-reference', help='Digitize verified published Figure 5 PDF paths')
    paper_reference.add_argument('pdf')
    paper_reference.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    paper_reference.add_argument('--output', default='ai2022_figure5.json')
    args = p.parse_args(argv)
    if args.action == 'status':
        print(json.dumps(status(), indent=2, ensure_ascii=False))
    elif args.action == 'network-cycle':
        if not args.workspace:p.error('An external workspace is required')
        from .network_cycle import run_network_cycle
        print(json.dumps(run_network_cycle(args.config,Workspace(args.workspace),steps=args.steps),indent=2))
    elif args.action == 'network-report':
        if not args.workspace:p.error('An external workspace is required')
        from .network_report import build_network_report
        if args.export:
            from .network_export import export_network_history
            export_network_history(args.directory,Workspace(args.workspace))
        print(json.dumps(build_network_report(args.directory,Workspace(args.workspace),output=args.output),indent=2))
    elif args.action == 'paper-particle':
        if not args.workspace:p.error('An external workspace is required')
        from .paper_particle_run import run_particle
        print(json.dumps(run_particle(args.config, Workspace(args.workspace), steps=args.steps, resume=args.resume, output=args.output),indent=2))
    elif args.action == 'paper-report':
        if not args.workspace:p.error('An external workspace is required')
        from .paper_report import build_paper_report
        print(json.dumps(build_paper_report(args.directory,Workspace(args.workspace),reference=args.reference,output=args.output),indent=2))
    elif args.action == 'paper-reference':
        if not args.workspace:p.error('An external workspace is required')
        from .paper_reference import extract_figure5
        target=Workspace(args.workspace).path('references',args.output)
        result=extract_figure5(args.pdf,target)
        print(json.dumps(dict(output=str(target),kind=result['kind'],doi=result['doi']),indent=2))
    elif args.action == 'build-resolved':
        if not args.workspace:p.error('An external workspace is required')
        from .resolved_echem import build_resolved
        print(json.dumps(build_resolved(args.config,Workspace(args.workspace)),indent=2))
    elif args.action == 'resolved-phase-field':
        if not args.workspace:p.error('An external workspace is required')
        import tomllib
        from pathlib import Path
        from .resolved_phase_field import ResolvedMechanicsConfig,advance_resolved_phase_field
        data=tomllib.loads(Path(args.config).read_text())
        if set(data)!={'schema_version','phase_field','mechanics'} or data['schema_version']!=1:raise ValueError('Resolved phase-field TOML schema 1 required')
        cfg=ResolvedMechanicsConfig(**data['mechanics']);a=data['phase_field'];ws=Workspace(args.workspace)
        if set(a)-{'geometry','field','output','previous'}:raise ValueError('Unknown resolved field option')
        print(json.dumps(advance_resolved_phase_field(ws.path(a['geometry']),ws.path(a['field']),ws.path(a['output']),ws,config=cfg,previous=ws.path(a['previous']) if a.get('previous') else None),indent=2))
    elif args.action == 'resolved-report':
        if not args.workspace:p.error('An external workspace is required')
        from .resolved_report import build_resolved_report,export_resolved_history
        if args.export:export_resolved_history(args.directory,Workspace(args.workspace))
        print(json.dumps(build_resolved_report(args.directory,Workspace(args.workspace),output=args.output),indent=2))
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
