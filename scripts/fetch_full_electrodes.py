"""Fetch five distinct complete public NMC532 segmented electrode volumes."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

PAGE='https://www.nlr.gov/transportation/battery-microstructure-library-data'
SOURCES={
 'nmc1uncal':('nmc-1-uncal-withcbd-w001-binarized.tif?sfvrsn=1aed4310_1','1-UNCAL'),
 'nmc1cal':('nmc-1-cal-withcbd-w001-binarized.tif?sfvrsn=2779b020_1','1-CAL'),
 'nmc2uncal':('nmc-2-uncal-withcbd-w001-binarized.tif?sfvrsn=f7188a84_1','2-UNCAL'),
 'nmc2cal':('nmc-2-cal-withcbd-w001-binarized.tif?sfvrsn=bb0564d4_1','2-CAL'),
 'nmc3uncal':('nmc-3-uncal-withcbd-w001-binarized.tif?sfvrsn=7758667d_1','3-UNCAL'),
}
def main():
    root=Path(os.environ['MORQEX_WORKSPACE'])/'data/ct';root.mkdir(parents=True,exist_ok=True);records={}
    for name,(suffix,designation) in SOURCES.items():
        path=root/(name+'.tif');url='https://www.nlr.gov/media/docs/libraries/tsdc/caebat/'+suffix
        if not path.exists():
            temp=path.with_suffix('.part')
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MORQEX-research/0.1'}),timeout=30) as response,temp.open('wb') as stream:
                    total=0
                    while chunk:=response.read(262144):
                        total+=len(chunk)
                        if total>32*2**20:raise ValueError('Public segmented volume exceeded the declared 32 MiB download budget')
                        stream.write(chunk)
                temp.replace(path)
            finally:temp.unlink(missing_ok=True)
        records[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
            'url':url,'source_page':PAGE,'designation':designation,'composition':'NMC532','voxel_size_m':398e-9,'axes_to_zyx':[1,2,0],
            'scope':'Complete provided segmented volume, without internal crop; active phase imaged, CBD reconstructed by source authors at omega=0.01. Not the entire fabricated electrode sheet or guaranteed RVE.'}
        print(name,records[name]['bytes'],flush=True)
    (root/'full_electrodes.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
