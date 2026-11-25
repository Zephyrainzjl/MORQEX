"""Independent state-energy/entropy audit for the qualified ideal chemistry."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spence
import pybamm as pb
from .microstructure_dfn import key


def ocp_primitive(parameters,domain,concentration):
    """Primitive of pinned phase.U, including its endpoint asymptote terms.

    The asymptote coefficients are those of bundled PyBaMM 26.3.1. Our physical
    interval [.001,.999] is above the clipping/linear-extension thresholds.
    """
    function=parameters[f'{domain.capitalize()} electrode OCP [V]']
    if not hasattr(function,'keywords') or 'x' not in function.keywords:raise ValueError('Audited cubic chemical energy primitive required')
    cmax=float(parameters[f'Maximum concentration in {domain} electrode [mol.m-3]'])
    sto=np.asarray(concentration)/cmax;anti=CubicSpline(function.keywords['x'],function.keywords['y']).antiderivative()
    a,b,shift=205.0568621937484,6910.192179565431,-7.7e-4
    barrier=a/b*(spence(1+np.exp(-b*(sto-shift)))+spence(1+np.exp(-b*(1-sto-shift))))
    return cmax*(anti(sto)+barrier)


def macro_weights(processed_variable):
    mesh=processed_variable.mesh
    weights=np.diff(mesh.edges)
    return weights/weights.sum()


def thermodynamic_observables(solution,stack,parameters,*,feedback=1.):
    p=parameters;T=np.asarray(solution['Volume-averaged cell temperature [K]'].entries)
    Cp=np.broadcast_to(solution['Volume-averaged effective heat capacity [J.K-1.m-3]'].entries,T.shape)
    if np.ptp(Cp)>1e-10*Cp.mean():raise ValueError('Constant-Cp energy primitive not applicable')
    A=float(p['Electrode width [m]'])*float(p['Electrode height [m]'])*float(p['Number of electrodes connected in parallel to make a cell'])
    V=float(p['Cell volume [m3]']);F=float(pb.constants.F.value)
    chemical=np.zeros(len(T));mean_concentrations={}
    for d,op in stack.electrodes.items():
        cs=[]
        for k,w in enumerate(op.weights):
            variable=solution[key(d,k,'concentration [mol.m-3]')];c=np.asarray(variable.entries);wx=macro_weights(variable)
            primitive=-F*(ocp_primitive(p,d,c)-ocp_primitive(p,d,op.reference_concentration))
            chemical+=A*stack.lengths[d]*float(w)*(wx@primitive);cs.append(c)
        mean_concentrations[d]=np.einsum('k,kxt->xt',op.weights,np.asarray(cs))/op.active_fraction
    elastic=A*(solution['MORQEX elastic energy per area [J.m-2]'].entries-T*solution['MORQEX elastic entropy force per area [J.m-2.K-1]'].entries)
    internal=Cp*V*(T-float(p['Initial temperature [K]']))+chemical+feedback*elastic
    work=np.asarray(solution['MORQEX accumulated electrical work [J]'].entries)
    cooling=np.asarray(solution['MORQEX accumulated cooling work [J]'].entries)
    energy_residual=internal-internal[0]-(work-work[0])-(cooling-cooling[0])
    entropy=sum(np.asarray(solution[name].entries) for name in ['MORQEX thermal entropy [J.K-1]','MORQEX elastic entropy [J.K-1]','MORQEX electrolyte mixing entropy [J.K-1]'])
    produced=np.asarray(solution['MORQEX accumulated entropy production [J.K-1]'].entries)
    exchange=np.asarray(solution['MORQEX accumulated entropy heat exchange [J.K-1]'].entries)
    entropy_residual=entropy-entropy[0]-(produced-produced[0])-(exchange-exchange[0])
    total_lithium=np.asarray(solution['Total lithium [mol]'].entries)
    port={d:float(np.max(abs(solution[f'MORQEX {d} port identity residual [W.m-3]'].entries))/max(1.,np.max(abs(solution[f'MORQEX {d} interface port power [W.m-3]'].entries)))) for d in stack.electrodes}
    return {'internal_energy_J':internal,'chemical_energy_J':chemical,'elastic_internal_energy_J':elastic,
        'entropy_J_K':entropy,'energy_residual_J':energy_residual,'entropy_residual_J_K':entropy_residual,
        'mean_concentrations':mean_concentrations,
        'metrics':{'relative_lithium_drift':float(np.ptp(total_lithium)/total_lithium[0]),
            'relative_internal_energy_residual':float(np.max(abs(energy_residual))/max(1.,np.max(abs(work-work[0])),np.max(abs(internal-internal[0])))),
            'relative_entropy_residual':float(np.max(abs(entropy_residual))/max(1e-6,np.max(abs(entropy-entropy[0])),np.max(abs(produced-produced[0])))),
            'minimum_dissipation_W':float(np.min(solution['MORQEX total dissipation [W]'].entries)),
            'port_identity_relative_residual':port}}
