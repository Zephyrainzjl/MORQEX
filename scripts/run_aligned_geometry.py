"""Reproduce whole-volume exact geometry stages through serial bounded jobs.

A frozen source snapshot lets an ongoing long run retain its own exact version.
No GPU is used. Compiler products, geometry, meshes and HTML remain external.
"""
import argparse,json,os,subprocess,sys,tarfile,time
from pathlib import Path
sys.dont_write_bytecode=True
from source_snapshot import snapshot,tree
from aligned_geometry_gallery import CASES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--workspace',type=Path,default=root.parent/'morph_workspace/microelectrode_gallery')
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--stages',nargs='+',choices=('mesh','preview','detail','exchange','vtk'),default=['mesh','preview','detail','exchange'])
    p.add_argument('--formats',nargs='+',choices=('comsol','vtu','gmsh','medit','abaqus'),default=['comsol','vtu','gmsh','medit','abaqus'])
    p.add_argument('--length-unit',choices=('m','cm','mm','um','nm'),default='um')
    p.add_argument('--slab-depth',type=int,default=32)
    p.add_argument('--roi-voxels',type=int,default=48)
    a=p.parse_args();work=a.workspace.resolve()
    if work==root or root in work.parents:raise ValueError('External workspace required')
    snap=snapshot(root,work);frozen=work/'frozen_aligned_geometry'/snap['tree_sha256']
    if not frozen.exists():
        frozen.mkdir(parents=True)
        with tarfile.open(snap['archive']) as archive:archive.extractall(frozen,filter='data')
    if tree(frozen)[0]!=snap['tree_sha256']:raise ArithmeticError('Frozen source integrity mismatch')
    python=work/'runtime/bin/python'
    if not python.is_file():raise FileNotFoundError('External runtime/bin/python required')
    status=work/('aligned_controller_'+str(time.time_ns())+'.json');records=[]
    def run(label,parameters):
        command=[sys.executable,str(frozen/'scripts/run_local.py'),'--workspace',str(work),
            '--reference-memory','--large-mesh-output','--label',label,'--',str(python),
            str(frozen/'scripts/aligned_geometry_gallery.py'),*parameters]
        result=subprocess.run(command)
        records.append({'label':label,'exit_code':result.returncode,'command':command})
        status.write_text(json.dumps({'source_snapshot':snap,'runs':records},indent=2))
        if result.returncode:raise SystemExit('Bounded stage failed; checkpoint retained, no candidate accepted: '+label)
    run('aligned_gallery_index',['--stage','index'])
    for case in a.cases:
        for stage in a.stages:
            if stage=='exchange':
                for fmt in a.formats:
                    run('aligned_exchange_'+case+'_'+fmt,['--stage',stage,'--case',case,'--format',fmt,'--length-unit',a.length_unit])
            else:
                run(('aligned_whole_' if stage=='mesh' else 'aligned_'+stage+'_')+case,
                    ['--stage',stage,'--case',case,'--slab-depth',str(a.slab_depth),'--roi-voxels',str(a.roi_voxels)])
        run('aligned_gallery_index',['--stage','index'])
    print(json.dumps({'html':str(work/'nmc_complex_gallery/voxel_aligned.html'),'controller':str(status),'source_snapshot':snap}),flush=True)


if __name__=='__main__':main()
