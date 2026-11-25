"""Independent energy/entropy and reservoir audit for open PCHIP chemistry."""
import numpy as np
from scipy.special import spence
import pybamm as pb
from .open_materials import TabulatedOCP
from .microstructure_dfn import key
from .aging_dfn import age
from .microstructure_validation import macro_weights

def chemical_primitives(parameters,domain,concentration,temperature_K):
    curve=parameters[f'{domain.capitalize()} electrode OCP [V]']
    if not isinstance(curve,TabulatedOCP):raise ValueError('Open audit requires the exact TabulatedOCP used by the solver')
    cmax=float(parameters[f'Maximum concentration in {domain} electrode [mol.m-3]']);sto=np.asarray(concentration)/cmax
    if np.any(sto<curve.stoichiometry[0]) or np.any(sto>curve.stoichiometry[-1]):raise ValueError('Chemical primitive requested outside audited OCP interval')
    # Pinned PyBaMM endpoint asymptotes, identical to the previous independent
    # audit. The open particle's [.001,.999] interval avoids clipped extensions.
    a,b,shift=205.0568621937484,6910.192179565431,-7.7e-4
    barrier=a/b*(spence(1+np.exp(-b*(sto-shift)))+spence(1+np.exp(-b*(1-sto-shift))))
    F=float(pb.constants.F.value);fT=-F*cmax*curve.entropy_primitive(sto)
    free=-F*cmax*(curve.primitive(sto,temperature_K)+barrier)
    return {'free_energy_J_m3':free,'entropy_J_m3_K':-fT,'internal_energy_J_m3':free-np.asarray(temperature_K)*fT}

def open_observables(solution,stack,parameters,config,*,feedback=1.):
    p=parameters;T=np.asarray(solution['Volume-averaged cell temperature [K]'].entries);n=len(T)
    A=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Number of electrodes connected in parallel to make a cell']);V=float(p['Cell volume [m3]'])
    Cp=np.broadcast_to(solution['Volume-averaged effective heat capacity [J.K-1.m-3]'].entries,T.shape)
    if np.ptp(Cp)>1e-10*Cp.mean():raise ValueError('Constant heat capacity required by this primitive')
    chemical=np.zeros(n);chemical_entropy=np.zeros(n);gradient=np.zeros(n);side_energy=np.zeros(n);side_lithium=np.zeros(n)
    for d,op in stack.electrodes.items():
        scale=A*stack.lengths[d]
        for k,w in enumerate(op.weights):
            variable=solution[key(d,k,'concentration [mol.m-3]')];wx=macro_weights(variable);parts=chemical_primitives(p,d,variable.entries,T)
            chemical+=scale*w*(wx@parts['internal_energy_J_m3']);chemical_entropy+=scale*w*(wx@parts['entropy_J_m3_K'])
            for law in getattr(config,d):
                for reservoir in law.reservoirs():
                    stock=wx@solution[age(d,reservoir.name+' [mol.m-3]',k)].entries
                    side_lithium+=scale*stock;side_energy+=scale*float(reservoir.chemical_potential_J_mol)*stock
        variable=solution[f'MORQEX {d} gradient energy density [J.m-3]'];gradient+=scale*(macro_weights(variable)@variable.entries)
    elastic=A*(solution['MORQEX elastic energy per area [J.m-2]'].entries-T*solution['MORQEX elastic entropy force per area [J.m-2.K-1]'].entries)
    internal=Cp*V*(T-float(p['Initial temperature [K]']))+chemical+gradient+side_energy+feedback*elastic
    work=np.asarray(solution['MORQEX accumulated electrical work [J]'].entries);cooling=np.asarray(solution['MORQEX accumulated cooling work [J]'].entries)
    energy_residual=internal-internal[0]-(work-work[0])-(cooling-cooling[0])
    entropy=chemical_entropy+sum(np.asarray(solution[name].entries) for name in ['MORQEX thermal entropy [J.K-1]','MORQEX elastic entropy [J.K-1]','MORQEX electrolyte mixing entropy [J.K-1]'])
    produced=np.asarray(solution['MORQEX accumulated entropy production [J.K-1]'].entries);exchange=np.asarray(solution['MORQEX accumulated entropy heat exchange [J.K-1]'].entries)
    entropy_residual=entropy-entropy[0]-(produced-produced[0])-(exchange-exchange[0]);lithium=np.asarray(solution['Total lithium [mol]'].entries)+side_lithium
    port={d:float(np.max(abs(solution[f'MORQEX {d} port identity residual [W.m-3]'].entries))/max(1.,np.max(abs(solution[f'MORQEX {d} interface port power [W.m-3]'].entries)))) for d in stack.electrodes}
    return {'internal_energy_J':internal,'chemical_internal_energy_J':chemical,'gradient_energy_J':gradient,'reservoir_energy_J':side_energy,'entropy_J_K':entropy,
        'energy_residual_J':energy_residual,'entropy_residual_J_K':entropy_residual,'lithium_including_reservoirs_mol':lithium,
        'metrics':{'relative_lithium_drift':float(np.ptp(lithium)/lithium[0]),'relative_internal_energy_residual':float(np.max(abs(energy_residual))/max(1.,np.max(abs(work-work[0])))),
            'relative_entropy_residual':float(np.max(abs(entropy_residual))/max(1e-6,np.ptp(entropy),np.ptp(produced))),
            'minimum_dissipation_W':float(np.min(solution['MORQEX total dissipation [W]'].entries)),'port_identity_relative_residual':port}}
