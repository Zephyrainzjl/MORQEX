"""Raw material-card defaults for nodewise numerical research (not calibration)."""
import inspect
import numpy as np
import pybamm as pb
from ..electrochemistry.open_materials import OpenMaterialCard,tabulate_parameter_ocps
from ..electrochemistry.cell_design import CellDesign
from .binding import default_reference


def cbd_parameters(workspace,host='NMC532',*,overrides=None):
    fallback=dict(pb.ParameterValues('OKane2022').items())
    p,audit=OpenMaterialCard.host_card(host,supplements=fallback,
        source_note='Missing-key supplements from OKane2022; interface/CBD/thermal/mechanical properties require independent calibration',
        overrides=overrides or {}).build()
    # Explicit ideal electrolyte contract of the port energy derivation. Raw
    # electrode OCPs, including nonmonotonic portions, are left unchanged.
    if callable(p['Cation transference number']):
        fn=p['Cation transference number'];args=[pb.Scalar(p['Initial concentration in electrolyte [mol.m-3]']),pb.Scalar(p['Reference temperature [K]'])]
        tplus=float(fn(*args[:len(inspect.signature(fn).parameters)]).evaluate())
    else:tplus=float(p['Cation transference number'])
    p.update({'Thermodynamic factor':1.,'Cation transference number':tplus})
    fractions={d:sum(float(np.mean(s.labels==phase)) for phase,kind in s.phase_kinds.items() if kind=='active')
        for d in ['negative','positive'] for s in [default_reference(workspace,d)]}
    x0,x1,y1,y0=map(float,pb.lithium_ion.get_min_max_stoichiometries(p))
    p,design=CellDesign((x0,x1),(y1,y0),policy='balance-negative-thickness').apply(p,active_fractions=fractions)
    x0,x1,y1,y0=map(float,pb.lithium_ion.get_min_max_stoichiometries(p))
    _,capacity=CellDesign((x0,x1),(y1,y0)).apply(p)
    p.update({'Nominal cell capacity [A.h]':min(capacity['electrode_usable_capacity_Ah'].values()),
        'Initial concentration in negative electrode [mol.m-3]':x1*float(p['Maximum concentration in negative electrode [mol.m-3]']),
        'Initial concentration in positive electrode [mol.m-3]':y1*float(p['Maximum concentration in positive electrode [mol.m-3]'])})
    curves,ocp_audit=tabulate_parameter_ocps(p)
    for d,curve in curves.items():p.update({f'{d.capitalize()} electrode OCP [V]':curve,f'{d.capitalize()} electrode OCP entropic change [V.K-1]':curve.entropy})
    audit.update(ocp=ocp_audit,cell_design=design,capacity=capacity,
        electrolyte_assumption={'thermodynamic_factor':1.,'constant_t_plus_at_reference':tplus},
        scope='Raw OCP numerical case; CBD/interface and supplemented mechanical parameters uncalibrated; CT CBD is source-reconstructed')
    return p,audit
