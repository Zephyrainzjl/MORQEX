"""Deterministic CT SVE admissibility selection; no geometry repair or resampling."""
import json
import os
from pathlib import Path
import numpy as np
from scipy import ndimage
import tifffile
from morqex.ct import SegmentedCT
from morqex import Workspace
from morqex.gemsim import GEMsim
from morqex.electrochemistry.microstructure_parameters import microstructure_parameters
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
from morqex.cohesive import build_cohesive
from morqex.topology import PorousCBD

def percolates(mask):
    labels,_=ndimage.label(mask)
    return bool((set(np.unique(labels[:,:,0]))&set(np.unique(labels[:,:,-1])))-{0})

def main():
    ws=Workspace(Path(os.environ['MORQEX_WORKSPACE']));sources=json.loads(ws.path('data','ct','sources.json').read_text());p,_=microstructure_parameters();out={}
    for d in ['Negative','Positive']:
        k=f'{d} electrode partial molar volume [m3.mol-1]';p.update({k:p[k]*.1})
    for name,domain in [('graphite5cal','negative'),('nmc1cal','positive')]:
        source=sources[name]
        with tifffile.TiffFile(source['path']) as f:labels=f.asarray(maxworkers=1).transpose(1,2,0)
        rng=np.random.default_rng(20261125);rejected={'phase_or_fraction':0,'percolation':0,'operator':0};size=4
        for candidate in range(20000):
            start=tuple(int(rng.integers(10,n-size-10)) for n in labels.shape);crop=tuple((a,a+size) for a in start)
            block=labels[tuple(slice(a,b) for a,b in crop)]
            if set(np.unique(block))!={0,1,2} or not 8<=np.sum(block==1)<=50:rejected['phase_or_fraction']+=1;continue
            if not percolates(block!=1) or not percolates(block!=0):rejected['percolation']+=1;continue
            data=SegmentedCT(Path(source['path']),source['sha256'],(source['voxel_size_m'],)*3,{0:'pore',1:'active',2:'cbd'},
                axes_to_zyx=(1,2,0),crop=crop,source_url=source['source_page']).build(ws)
            try:
                full=MicrostructureElectrode(data,GEMsim(ws),p,domain,topology=PorousCBD());op=full.partition(max(4,full.diffusion.n_components));coh=build_cohesive(full,op.prolongation)
            except ValueError as exc:
                rejected['operator']+=1
                if rejected['operator']>=8:raise ValueError('Repeated CT operator rejection: '+str(exc)) from exc
                continue
            out[domain]={'dataset':name,'crop':crop,'axes_to_zyx':[1,2,0],'candidate':candidate,'rejected':rejected,'geometry':op.metadata(),
                'cohesive':coh.metadata,'scope':'very small admissible CT-derived SVE; not representative electrode or material calibration'}
            print(json.dumps(out[domain]),flush=True)
            ws.path('data','ct','selected_crops.json').write_text(json.dumps(out,indent=2));break
        else:raise ValueError('No admissible CT crop within fixed search budget')
    ws.path('data','ct','selected_crops.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
