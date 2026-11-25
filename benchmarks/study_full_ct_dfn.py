"""Native DFN controls driven by complete CT geometry/homogenized transport.

This is the independent spherical-particle baseline, not the proposed local
voxel concentration reconstruction or a coupled full-voxel mechanics result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pybamm as pb
from morqex.electrochemistry.open_materials import OpenMaterialCard,HOSTS
from morqex.electrochemistry.cell_design import CellDesign
from morqex.solvers.pybamm_backend import solver
from morqex.electrochemistry.validation import check_protocol

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--host',choices=list(HOSTS),default='NMC532');ap.add_argument('--rate',type=float,default=1.);ap.add_argument('--nx',type=int,default=10);ap.add_argument('--nr',type=int,default=20);ap.add_argument('--cycles',type=int,default=0);a=ap.parse_args()
    if not 0<a.rate<=20 or not 3<=a.nx<=80 or not 5<=a.nr<=100 or not 0<=a.cycles<=20:raise ValueError('Rate, resolution and cycle limits required')
    ws=Path(os.environ['MORQEX_WORKSPACE']);directory=ws/'operators/full_volume'/a.dataset;geometry=json.loads((directory/'manifest.json').read_text())
    transports={kind:json.loads((directory/(kind+'_transport.json')).read_text()) for kind in ['electrolyte','electrode']}
    for value in transports.values():
        if value['geometry_sha256']!=geometry['sha256'] or not value['metrics']['converged']:raise ValueError('Converged transport for this exact CT geometry required')
    fallback=dict(pb.ParameterValues('OKane2022').items());p,material=OpenMaterialCard.host_card(a.host,supplements=fallback,source_note='Explicit missing-key supplements from OKane2022 for baseline solver compatibility; no claim of cross-material calibration').build()
    sto=pb.lithium_ion.get_min_max_stoichiometries(p);xn0,xn1,yp1,yp0=map(float,sto)
    design=CellDesign((xn0,xn1),(yp1,yp0),policy='balance-negative-thickness',negative_to_positive_ratio=1.1)
    active=geometry['phase_fractions']['1'];p,capacity=design.apply(p,active_fractions={'positive':active})
    pore=geometry['phase_fractions']['0']+.47*geometry['phase_fractions']['2']
    # The .47 CBD liquid fraction is an explicitly declared study parameter,
    # distinct from the separately prescribed .2 CBD transport efficiency.
    p.update({'Positive electrode porosity':pore,'Positive particle radius [m]':3*active/geometry['reaction_area_density_m_inv'],
        'Positive electrode Bruggeman coefficient (electrolyte)':float(np.log(transports['electrolyte']['metrics']['effective_column'][0])/np.log(pore)),
        'Positive electrode Bruggeman coefficient (electrode)':float(np.log(transports['electrode']['metrics']['effective_column'][0])/np.log(1-pore)),
        'Initial concentration in negative electrode [mol.m-3]':xn1*float(p['Maximum concentration in negative electrode [mol.m-3]']),
        'Initial concentration in positive electrode [mol.m-3]':yp1*float(p['Maximum concentration in positive electrode [mol.m-3]'])})
    # Geometry/balancing changes the stoichiometries at the voltage endpoints.
    # Recompute equilibrium usable capacity, rather than retaining windows
    # inherited from a different electrode design under the same C-rate label.
    x0,x1,y1,y0=map(float,pb.lithium_ion.get_min_max_stoichiometries(p))
    _,operating=CellDesign((x0,x1),(y1,y0),policy='preserve').apply(p)
    operating_capacity=min(operating['electrode_usable_capacity_Ah'].values())
    p.update({'Nominal cell capacity [A.h]':operating_capacity,
        'Initial concentration in negative electrode [mol.m-3]':x1*float(p['Maximum concentration in negative electrode [mol.m-3]']),
        'Initial concentration in positive electrode [mol.m-3]':y1*float(p['Maximum concentration in positive electrode [mol.m-3]'])})
    capacity={'initial_window_balance':capacity,'voltage_endpoint_recomputed_windows':{'negative':[x0,x1],'positive':[y1,y0]},
        'nameplate_capacity_Ah':operating_capacity,'operating_electrode_capacities_Ah':operating['electrode_usable_capacity_Ah'],
        'physical_C_rate_at_0_1C_nameplate':.1,'policy':'Explicit geometry balance, then SOH re-evaluation at the actual voltage endpoints; rate-dependent delivered capacity remains an output'}
    lower=float(p['Lower voltage cut-off [V]']);upper=float(p['Upper voltage cut-off [V]']);steps=[f'Discharge at {a.rate}C until {lower} V']
    if a.cycles:steps=[(steps[0],'Rest for 300 seconds',f'Charge at {a.rate}C until {upper} V',f'Hold at {upper} V until C/20','Rest for 300 seconds')]*a.cycles
    model=pb.lithium_ion.DFN(options={'thermal':'lumped'});sim=pb.Simulation(model,parameter_values=p,solver=solver(rtol=1e-8,atol=1e-9),experiment=pb.Experiment(steps,period='60 seconds' if a.rate<.2 else '10 seconds'),
        var_pts={**model.default_var_pts,'x_n':a.nx,'x_s':a.nx,'x_p':a.nx,'r_n':a.nr,'r_p':a.nr})
    started=time.perf_counter();sol=sim.solve(calc_esoh=False);elapsed=time.perf_counter()-started
    durations=[{'termination':s.termination,'duration_s':float(s.t[-1]-s.t[0]),'voltage_end_V':float(s['Voltage [V]'].entries[-1])} for cycle in sol.cycles if cycle is not None for s in cycle.steps]
    try:
        protocol=check_protocol(sol,cycles=a.cycles or 1,steps_per_cycle=5 if a.cycles else 1,lower_V=lower,upper_V=upper,current_A=operating_capacity,rest_s=300.)
        complete=True
    except RuntimeError as error:protocol={'completed':False,'reason':str(error)};complete=False
    lithium=np.asarray(sol['Total lithium [mol]'].entries);current=np.asarray(sol['Current [A]'].entries);positive=np.asarray(sol['Average positive particle concentration [mol.m-3]'].entries)
    curves={'time_s':sol.t.tolist(),'voltage_V':np.asarray(sol['Voltage [V]'].entries).tolist(),'current_A':current.tolist(),'temperature_K':np.asarray(sol['Volume-averaged cell temperature [K]'].entries).tolist(),
        'discharge_capacity_Ah':np.asarray(sol['Discharge capacity [A.h]'].entries).tolist(),'positive_average_concentration_mol_m3':positive.tolist(),
        'positive_concentration_by_X_mol_m3':np.asarray(sol['R-averaged positive particle concentration [mol.m-3]'].entries).tolist(),
        'positive_X_nodes_m':np.asarray(sol['R-averaged positive particle concentration [mol.m-3]'].mesh.nodes).tolist()}
    discharge_capacities=[float(c.steps[0]['Discharge capacity [A.h]'].entries[-1]-c.steps[0]['Discharge capacity [A.h]'].entries[0]) for c in sol.cycles if c is not None]
    out=ws/'results/full_ct_dfn'/f'{a.dataset}_{a.host}_{a.rate:g}C_x{a.nx}_r{a.nr}_cycles{a.cycles}';out.mkdir(parents=True,exist_ok=True)
    report={'arguments':vars(a),'geometry':geometry,'transport':transports,'material':material,'capacity_design':capacity,'completed':complete,'protocol':protocol,'steps':durations,'metrics':{'solve_seconds':elapsed,'relative_lithium_drift':float(np.ptp(lithium)/lithium[0]),'delivered_discharge_Ah':discharge_capacities[-1],
        'cycle_discharge_capacities_Ah':discharge_capacities,'net_integrated_capacity_Ah':float(curves['discharge_capacity_Ah'][-1]),'end_time_s':float(sol.t[-1])},'curves':curves,
        'mapping_scope':'Complete CT scalar geometry/transport -> native DFN baseline. Radial states are not uniquely mapped to voxel fields. NMC811/LFP/LCO on NMC532 morphology are virtual material substitutions, not imaged examples of those hosts.',
        'degradation':'disabled for the independent native reference; coupled open particle aging is validated separately',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).parents[1]/'src/morqex/electrochemistry/cell_design.py']}}
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in {'curves','geometry'}},indent=2))
    if not complete:raise RuntimeError('Native reference did not complete the requested physical protocol')
if __name__=='__main__':main()
