"""GEMsim geometry-provider binding of the qualified crossscale case engine.

Only preparation is replaced. The shared DFN/degradation/inventory/energy audit
engine is used verbatim and both driver files are included in source snapshots.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from morqex.gemsim import GEMsim
from morqex.topology import PorousCBD
from morqex.bases import GeometricBasis,PortBasis,ExactBasis,VariationalOperator
from morqex.extensions import build_operator
from morqex.cohesive import build_cohesive
from morqex.cohesive_reduction import compress_cohesive
from morqex.electrochemistry.microstructure_operator import MicrostructureElectrode
from morqex.electrochemistry.aging_cache import save_aging_operator
import study_degradation as engine

GEOMETRIES={
    'isotropic':{'seed':104,'fractions':(.30,.55,.15),'length_scale':(.6,.6,.6)},
    'directional':{'seed':205,'fractions':(.30,.55,.15),'length_scale':(1.2,.5,.5)},
    'CBD-rich':{'seed':306,'fractions':(.25,.50,.25),'length_scale':(.7,.7,.7)},
}

def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--geometry-family',choices=list(GEOMETRIES),required=True)
    extra,remaining=parser.parse_known_args();configuration=GEOMETRIES[extra.geometry_family]
    original_hashes=engine.hashes
    def snapshot():return original_hashes()|{str(Path(__file__).relative_to(engine.ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def prepare(ws,p,args):
        if args.doping!='none':raise ValueError('This geometry qualification holds composition fixed; use the composition-aware workflow for doping')
        records={};paths={};provider=GEMsim(ws)
        for domain in ['negative','positive']:
            start=time.perf_counter();rejections=[]
            for attempt in range(32):
                seed=configuration['seed']+(0 if domain=='negative' else 1)+17*attempt
                data=provider.random_field(shape=(3,3,3),fractions=configuration['fractions'],length_scale=configuration['length_scale'],seed=seed,
                    spacing=(1e-6,)*3,phase_kinds={0:'pore',1:'active',2:'cbd'})
                try:full=MicrostructureElectrode(data,provider,p,domain,topology=PorousCBD())
                except ValueError as error:
                    if not any(word in str(error) for word in ['percolation','reactive interfaces','phases']):raise
                    rejections.append({'seed':seed,'reason':str(error)});continue
                break
            else:raise RuntimeError('No admissible generated SVE within declared seed budget')
            basis={'geometry':GeometricBasis(),'port':PortBasis(),'exact':ExactBasis()}[args.basis]
            op=build_operator(full,basis,VariationalOperator(),max(args.regions,full.diffusion.n_components));cohesive=build_cohesive(full,op.prolongation)
            if args.compression:cohesive=compress_cohesive(cohesive,tolerance=args.compression,maximum_modes=args.compression_modes)
            paths[domain]=save_aging_operator(ws.path('operators','aging',args.case,domain),op,cohesive)
            records[domain]={'provider':'integrated GEMsim random field','configuration':configuration,'selected_seed':seed,'rejections':rejections,
                'shape':[3,3,3],'spacing_m':[1e-6]*3,'operator':op.metadata(),'cohesive':cohesive.metadata,'offline_seconds':time.perf_counter()-start}
            del full,op,cohesive;gc.collect()
        return paths,records
    engine.prepare=prepare;engine.hashes=snapshot;sys.argv=[sys.argv[0],*remaining]
    engine.main()
    # Correct the generic engine's historical CT wording, never its numerics.
    case=remaining[remaining.index('--case')+1];path=engine.Workspace(Path(engine.os.environ['MORQEX_WORKSPACE'])).path('results','aging',case,'result.json')
    record=json.loads(path.read_text());record['geometry_family']=extra.geometry_family
    record['scope']='Independent synthetic GEMsim 3^3 SVEs; declared geometric eligibility, fixed seed list and CBD parameters. Not CT, RVE representativity or material calibration.'
    path.write_text(json.dumps(record,indent=2,allow_nan=False))

if __name__=='__main__':main()
