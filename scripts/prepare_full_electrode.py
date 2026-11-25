import argparse
import json
import os
from pathlib import Path
from morqex.microstructure.full_volume import prepare_full_volume

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);args=parser.parse_args();w=Path(os.environ['MORQEX_WORKSPACE'])
    source=json.loads((w/'data/ct/full_electrodes.json').read_text())[args.dataset]
    result=prepare_full_volume(source,w/'operators/full_volume'/args.dataset);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
