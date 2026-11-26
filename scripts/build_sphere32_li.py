"""Build the real 32^3, multi-particle Li-half-cell reference operator externally."""
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from morqex import Workspace
from morqex.geometry import MicrostructureData
from morqex.gemsim import GEMsim
from morqex.topology import PorousCBD
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
from morqex.electrochemistry.open_particle import TransportLaw
from morqex.nodewise.halfcell import lithium_halfcell_parameters
from morqex.nodewise.voxel_operator import build_voxel_intact_operator
from morqex.nodewise.cache import save_node_kernel


def main():
    import os,argparse
    parser=argparse.ArgumentParser();parser.add_argument('--modes',type=int,default=64);args=parser.parse_args()
    ws=Workspace(os.environ['MORQEX_WORKSPACE']);start=time.perf_counter()
    source=ws.path('data','sphere32');meta=json.loads((source/'manifest.json').read_text())
    with np.load(source/'geometry.npz') as a:
        data=MicrostructureData(a['labels'],tuple(a['spacing']),(0.,)*3,{0:'pore',1:'active',2:'cbd'},meta)
    p,audit=lithium_halfcell_parameters(active_fraction=float(np.mean(data.labels==1)))
    full=MicrostructureElectrode(data,GEMsim(ws),p,'positive',topology=PorousCBD(),defer_mechanics=True)
    print('actual geometry',data.labels.shape,'active',full.n,'particles',full.diffusion.n_components,flush=True)
    op,response=build_voxel_intact_operator(full,args.modes)
    op.transport_law=asdict(TransportLaw(gradient_coefficient_J_m5_mol2=1e-14))
    path=ws.path('operators','sphere32_li',data.fingerprint+'_'+str(args.modes))
    save_node_kernel(path,op,response)
    record={'path':str(path),'shape':list(data.labels.shape),'particle_count':meta['particle_count'],
        'dimension':op.dimension,'response':response.metadata,'elapsed_s':time.perf_counter()-start,'parameters':audit}
    out=ws.path('results','sphere32_li','operator.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
