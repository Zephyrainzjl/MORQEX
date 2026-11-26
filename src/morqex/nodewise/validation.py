"""Independent full spatial free-energy, entropy and inventory reconstruction."""
import numpy as np
from ..electrochemistry.open_validation import chemical_primitives
from ..electrochemistry.microstructure_validation import macro_weights
from ..electrochemistry.microstructure_dfn import key
from ..electrochemistry.aging_dfn import age


def spatial_observables(solution,stack,parameters,config,*,feedback=1.):
    p=parameters;Nt=len(solution.t);A=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Number of electrodes connected in parallel to make a cell'])
    chemical=np.zeros(Nt);Schemical=np.zeros(Nt);gradient=np.zeros(Nt);elastic=np.zeros(Nt);Selastic=np.zeros(Nt)
    thermal=np.zeros(Nt);Sthermal=np.zeros(Nt);reservoir=np.zeros(Nt);lithium=np.asarray(solution['Total lithium [mol]'].entries).copy()
    T0=float(p['Initial temperature [K]']);Tref=float(p['Reference temperature [K]']);F=96485.33212
    for domain in ['negative electrode','separator','positive electrode']:
        d=domain.split()[0];var=solution[domain.capitalize()+' temperature [K]'];T=np.asarray(var.entries);wx=macro_weights(var)
        length=stack.separator_length if d=='separator' else stack.lengths[d]
        cp=np.array([cell.mean_capacity for cell in stack.thermal_cells[d]])[:,None] if d!='separator' and hasattr(stack,'thermal_cells') else float(p[domain.capitalize()+' density [kg.m-3]'])*float(p[domain.capitalize()+' specific heat capacity [J.kg-1.K-1]'])
        scale=A*length;thermal+=scale*(wx@(cp*(T-T0)));Sthermal+=scale*(wx@(cp*np.log(T/T0)))
        if d=='separator':continue
        op=stack.electrodes[d];concentrations=np.stack([solution[key(d,k,'concentration [mol.m-3]')].entries for k in range(op.dimension)],axis=1)
        E=solution[f'MORQEX {d} axial strain'].entries;damage=solution[age(d,'damage')].entries
        for i,kernel in enumerate(op.kernels):
            local=kernel.operator;m=local.dimension;weights=local.weights
            for k,w in enumerate(weights):
                c=concentrations[i,k];parts=chemical_primitives(p,d,c,T[i])
                chemical+=scale*wx[i]*w*parts['internal_energy_J_m3'];Schemical+=scale*wx[i]*w*parts['entropy_J_m3_K']
            q=np.column_stack([E[i],(concentrations[i,:m]-local.reference_concentration).T,T[i]-Tref])
            for t in range(Nt):
                H=kernel.hessian(float(damage[i,t]));force=H@q[t]
                elastic[t]+=scale*wx[i]*(.5*q[t]@force-T[i,t]*force[-1]);Selastic[t]-=scale*wx[i]*force[-1]
        for k in range(op.dimension):
            for law in getattr(config,d):
                for r in law.reservoirs():
                    stock=wx@solution[age(d,r.name+' [mol.m-3]',k)].entries
                    lithium+=scale*stock;reservoir+=scale*float(r.chemical_potential_J_mol)*stock
        variable=solution[f'MORQEX {d} gradient energy density [J.m-3]'];gradient+=scale*(wx@variable.entries)
    sigma=solution['MORQEX stack axial stress [Pa]'].entries
    elastic+=.5*A*stack.separator_length*sigma**2/stack.separator_C
    for d in ['negative','positive']:
        D=d.capitalize();T=solution[D+' current collector temperature [K]'].entries
        length=float(p[D+' current collector thickness [m]']);cp=float(p[D+' current collector density [kg.m-3]'])*float(p[D+' current collector specific heat capacity [J.kg-1.K-1]'])
        thermal+=A*length*cp*(T-T0);Sthermal+=A*length*cp*np.log(T/T0)
    internal=thermal+chemical+gradient+reservoir+feedback*elastic
    entropy=Sthermal+Schemical+feedback*Selastic+solution['MORQEX electrolyte mixing entropy [J.K-1]'].entries
    work=solution['MORQEX accumulated electrical work [J]'].entries;cooling=solution['MORQEX accumulated cooling work [J]'].entries
    produced=solution['MORQEX accumulated entropy production [J.K-1]'].entries;exchange=solution['MORQEX accumulated entropy heat exchange [J.K-1]'].entries
    e=internal-internal[0]-(work-work[0])-(cooling-cooling[0]);s=entropy-entropy[0]-(produced-produced[0])-(exchange-exchange[0])
    flux=solution['MORQEX distributed electric power closure residual [W]'].entries;incoming=solution['MORQEX electrical incoming power [W]'].entries
    return {'energy_residual_J':e,'entropy_residual_J_K':s,'lithium_mol':lithium,
        'metrics':{'relative_internal_energy_residual':float(np.max(abs(e))/max(1.,np.max(abs(work-work[0])))),
            'relative_entropy_residual':float(np.max(abs(s))/max(1e-6,np.ptp(entropy),np.ptp(produced))),
            'relative_lithium_drift':float(np.ptp(lithium)/lithium[0]),
            'electric_flux_power_relative_defect':float(np.max(abs(flux))/max(1.,np.max(abs(incoming)))),
            'minimum_conduction_entropy_production_W_K':float(np.min(solution['MORQEX distributed conduction entropy production [W.K-1]'].entries)),
            'minimum_heat_capacity_Schur_ratio':float(np.min(solution['MORQEX distributed heat-capacity Schur ratio'].entries))}}
