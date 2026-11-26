"""Evidence/report interface. Simulation commands are intentionally unavailable."""
import argparse
import json
import os
from morqex.workspace import Workspace


def status():
    return {
        'framework': 'MORQEX / TOP-DAMAGE',
        'stage': 'A: audit, literature screening, mathematical design and API foundations',
        'implemented': ['explicit physical contact graphs', 'fixed-material conservative repartition',
                        'operator dependency identities', 'MORQEX ComponentSpec registry'],
        'not_qualified': ['polycrystal fatigue reference', 'phase-field-to-aperture topology update',
                          'dynamic tensor homogenization with fracture', 'topology-changing DFN restart',
                          'full long-cycle closure', 'cycle-jump error control', 'compiled fatigue backend'],
        'neural_models': False,
        'can_predict_lifetime': False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog='morqex top-damage')
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('status')
    report = sub.add_parser('report')
    report.add_argument('--workspace', default=os.environ.get('MORQEX_WORKSPACE'))
    args = p.parse_args(argv)
    if args.action == 'status':
        print(json.dumps(status(), indent=2, ensure_ascii=False))
    else:
        if not args.workspace:
            p.error('An external --workspace or MORQEX_WORKSPACE is required')
        from .report import build_report
        print(build_report(Workspace(args.workspace)))


if __name__ == '__main__':
    main()
