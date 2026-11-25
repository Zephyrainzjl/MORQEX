"""Native spherical DFN with matched geometry transport/area: distinct physics.

No claim of same-physics ROM accuracy is made against this baseline. Native
side laws and native lumped heat differ from the MORQEX energy-port closures.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pybamm as pb
from morqex.electrochemistry.aging_cache import load_aging_kernel
from morqex.electrochemistry.cathodes import cathode_parameters
from morqex.electrochemistry.microstructure_dfn import GeometryTransport,MicrostructureStack
from morqex.electrochemistry.protocol import shared_control,add_protocol_event,solve_protocol
from morqex.electrochemistry.validation import check_protocol
from morqex.solvers.pybamm_backend import solver,provenance

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['isotropic','directional','CBD-rich'],required=True)
    parser.add_argument('--aging',action='store_true');parser.add_argument('--radial-cells',type=int,default=16);args=parser.parse_args()
    ws=Path(os.environ['MORQEX_WORKSPACE']);source=ws/'results/aging'/f'gem_{args.family}_rom8/result.json';case=json.loads(source.read_text())
    p,audit=cathode_parameters('NMC');kernels={d:load_aging_kernel(case['microstructure'][d]['kernel_path']) for d in ['negative','positive']}
    for d,k in kernels.items():p.update({f'{d.capitalize()} electrode partial molar volume [m3.mol-1]':p[f'{d.capitalize()} electrode partial molar volume [m3.mol-1]']*.1})
    stack=MicrostructureStack(kernels['negative'].operator,kernels['positive'].operator,p);stack.update_parameters(p)
    radius={d:3*k.operator.active_fraction/k.operator.area_density for d,k in kernels.items()}
    for d,r in radius.items():p.update({f'{d.capitalize()} particle radius [m]':r})
    options={'thermal':'lumped','operating mode':shared_control}
    if args.aging:options.update({'SEI':'solvent-diffusion limited','lithium plating':'partially reversible','SEI porosity change':'true','lithium plating porosity change':'true'})
    start=time.perf_counter();model=pb.lithium_ion.DFN(options,build=False)
    for component in ['Electrolyte','Electrode']:
        model.submodels[component.lower()+' transport efficiency']=GeometryTransport(model.param,component,model.options,stack=stack)
    model.build_model();add_protocol_event(model);symbolic=time.perf_counter()-start
    sim=pb.Simulation(model,parameter_values=p,solver=solver(rtol=1e-9,atol=1e-10),var_pts={**model.default_var_pts,'x_n':5,'x_s':5,'x_p':5,'r_n':args.radial_cells,'r_p':args.radial_cells})
    start=time.perf_counter();q=float(p['Nominal cell capacity [A.h]'])
    sol,driver=solve_protocol(sim,kind='cycles',cycles=3,current_A=.1*q,lower_V=3.5,upper_V=3.9,cv_stop_A=q/20)
    elapsed=time.perf_counter()-start;protocol=check_protocol(sol,cycles=3,steps_per_cycle=5,lower_V=3.5,upper_V=3.9,current_A=q,rest_s=300.)
    lithium=sol['Total lithium [mol]'].entries+sol['Total lithium lost to side reactions [mol]'].entries
    drift=float(np.max(abs(lithium-lithium[0]))/abs(lithium[0]));assert drift<1e-8
    stock={key:sol[name].entries.tolist() for key,name in {'sei':'Loss of lithium to negative SEI [mol]','plating_and_dead':'Loss of lithium to negative lithium plating [mol]'}.items()}
    record={'schema':'morqex-native-geometry-dfn-1','family':args.family,'native_aging':args.aging,'radial_cells':args.radial_cells,
        'source_case':str(source),'source_case_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'backend':provenance(),'geometry_parameter_audit':audit,'equivalent_radius_m':radius,'protocol':protocol,'protocol_driver':driver,
        'relative_lithium_drift_including_side_reservoirs':drift,'symbolic_seconds':symbolic,'build_solve_seconds':elapsed,
        'curves':{'time_s':sol.t.tolist(),'voltage_V':sol['Voltage [V]'].entries.tolist(),'current_A':sol['Current [A]'].entries.tolist(),'temperature_K':sol['Volume-averaged cell temperature [K]'].entries.tolist()},
        'reservoirs_mol':stock,'cycle_discharge_Ah':[float(np.trapezoid(c.steps[0]['Current [A]'].entries,c.steps[0].t)/3600) for c in sol.cycles],
        'scope':'Matched phase fractions, geometry transport and initial surface area; equivalent spherical radial diffusion, no microstructure mechanics/debonding. Native side/thermal laws differ; model-form comparison, not ROM error or calibrated lifetime.'}
    out=ws/'results/native_dfn'/f'{args.family}_{"aging" if args.aging else "bare"}_{args.radial_cells}.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2,allow_nan=False));print(json.dumps({'path':str(out),'drift':drift,'seconds':elapsed}))

if __name__=='__main__':main()
