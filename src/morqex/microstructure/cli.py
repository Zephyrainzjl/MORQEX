import argparse
import json
from . import Volume, Workspace

def main():
    p = argparse.ArgumentParser(prog='morqex')
    sub = p.add_subparsers(dest='action',required=True)
    init = sub.add_parser('init');init.add_argument('workspace')
    stats = sub.add_parser('inspect');stats.add_argument('volume')
    mesh = sub.add_parser('mesh');mesh.add_argument('volume');mesh.add_argument('output')
    reconstruct=sub.add_parser('reconstruct');reconstruct.add_argument('manifest');reconstruct.add_argument('output')
    reconstruct.add_argument('--size',type=int,default=256);reconstruct.add_argument('--seed',type=int,default=42)
    reconstruct.add_argument('--device',default='cpu')
    preview=sub.add_parser('preview');preview.add_argument('volume');preview.add_argument('output')
    preview.add_argument('--reference')
    args = p.parse_args()
    if args.action == 'init': print(Workspace(args.workspace).initialize().root)
    elif args.action == 'inspect':
        from .analysis import describe
        print(json.dumps(describe(Volume.load(args.volume)),indent=2))
    elif args.action == 'mesh':
        from .mesh import voxel_to_tetra
        m = voxel_to_tetra(Volume.load(args.volume))
        print(m.save_comsol(args.output) if args.output.endswith('.mphtxt') else m.save(args.output))
    elif args.action=='reconstruct':
        from .generate import Reconstructor
        v=Reconstructor.load(args.manifest).generate(size=args.size,seed=args.seed,device=args.device)
        print(v.save(args.output))
    elif args.action=='preview':
        from .reporting import structure_html
        print(structure_html(Volume.load(args.volume),args.output,reference=args.reference))
