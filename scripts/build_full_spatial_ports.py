import argparse
import json
import os
from pathlib import Path
from morqex.concentration_modes.spatial_ports import build_spatial_ports

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--bins',type=int,default=5);a=p.parse_args()
    print(json.dumps(build_spatial_ports(Path(os.environ['MORQEX_WORKSPACE'])/'operators/full_volume'/a.dataset,bins=a.bins),indent=2))
