"""Fetch small official CT-derived segmented volumes, only to external workspace."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

PAGE='https://www.nlr.gov/transportation/battery-microstructure-library-data'
SOURCES={
 'nmc1cal':('https://www.nlr.gov/media/docs/libraries/tsdc/caebat/nmc-1-cal-withcbd-w001-binarized.tif?sfvrsn=2779b020_1',398e-9,'Toda NMC532, 1-CAL, omega=0.01'),
 'graphite5cal':('https://www.nlr.gov/media/docs/libraries/tsdc/caebat/graphite-5-cal-withcbd-w001-binarized.tif?sfvrsn=9b48123a_1',126e-9,'Conoco Phillips A12 graphite, 5-CAL, omega=0.01'),
}

def main():
    root=Path(os.environ['MORQEX_WORKSPACE'])/'data/ct';root.mkdir(parents=True,exist_ok=True);records={}
    for name,(url,spacing,description) in SOURCES.items():
        path=root/(name+'.tif')
        if not path.exists():
            temporary=path.with_suffix('.part')
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MORQEX-research/0.0.8'}),timeout=35) as response,temporary.open('wb') as f:
                    total=0
                    while chunk:=response.read(256*1024):
                        total+=len(chunk)
                        if total>16*2**20:raise ValueError('CT download exceeds 16 MiB per volume')
                        f.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        records[name]={'path':str(path),'url':url,'source_page':PAGE,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes':path.stat().st_size,'voxel_size_m':spacing,'description':description,'labels':{'0':'pore','1':'active','2':'cbd'},
            'provenance_limit':'Active morphology from X-ray CT; CBD numerically reconstructed by source authors, not independently resolved by CT',
            'orientation':'Source reports in-plane1 x in-plane2 x through-plane; TIFF array axes must be recorded explicitly by the importer',
            'retrieval_date':'2026-11-25'}
    (root/'sources.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

if __name__=='__main__':main()
