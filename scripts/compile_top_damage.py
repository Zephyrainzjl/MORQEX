"""One-command TOML build; every compiler slice runs under the local budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2]/'morph_workspace')
    p.add_argument('--max-slices',type=int,default=30)
    args=p.parse_args()
    if args.max_slices<1:p.error('Positive slice count required')
    runner=Path(__file__).resolve().with_name('run_local.py')
    for step in range(1,args.max_slices+1):
        job=subprocess.run([sys.executable,str(runner),'--label','top_damage_source_compile',
                            '--workspace',str(args.workspace.resolve()),'--',sys.executable,
                            '-m','morqex.top_damage','build',str(args.config.resolve())],capture_output=True,text=True)
        if job.returncode:
            print(job.stdout);print(job.stderr,file=sys.stderr)
            raise SystemExit(job.returncode)
        run=json.loads(job.stdout)
        result=json.loads((Path(run['run'])/'stdout.txt').read_text())
        print(json.dumps({'slice':step,'state':result['state'],'directory':result['directory'],
                          'seconds':run['wall_seconds'],'sampled_RSS_MiB':run['peak_group_rss_bytes_sampled']/2**20}),flush=True)
        if result['state']=='complete':return
    raise SystemExit('Compilation remains pending; repeat to resume existing objects')


if __name__=='__main__':main()
