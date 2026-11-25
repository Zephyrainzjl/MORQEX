import argparse
import json
import os
from pathlib import Path
from morqex.concentration_modes.full_volume import project_volume

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--extra-nodes',type=int,default=8);a=p.parse_args()
    print(json.dumps(project_volume(Path(os.environ['MORQEX_WORKSPACE'])/'operators/full_volume'/a.dataset,extra_nodes=a.extra_nodes),indent=2))
