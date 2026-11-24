"""Own API around upstream PyBaMM and its compiled IDAKLU solver."""
import os

def discharge(model='DFN', parameter_set='Chen2020', c_rate=1., duration=3600.,
              points=101, updates=None):
    os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY', 'true')
    import pybamm
    import numpy as np
    models = {'DFN': pybamm.lithium_ion.DFN, 'SPM': pybamm.lithium_ion.SPM,
              'SPMe': pybamm.lithium_ion.SPMe}
    if model not in models or c_rate <= 0 or duration <= 0 or points < 2:
        raise ValueError('Invalid discharge settings')
    params = pybamm.ParameterValues(parameter_set)
    if updates: params.update(updates)
    params.update({'Current function [A]': float(c_rate)*params['Nominal cell capacity [A.h]']})
    sim = pybamm.Simulation(models[model](), parameter_values=params,
                             solver=pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-9))
    return sim.solve(t_eval=np.linspace(0, duration, points))

def microstructure_parameters(volume, pore_phase=0, electrode='Negative', tortuosity=None,
                              active_phase=None):
    """Map measured pore and active fractions consistently to a cell model.

    Supply active_phase for multiphase volumes. In a binary volume the unique
    non-pore phase is inferred. Tortuosity comes from a transport calculation.
    """
    import numpy as np
    if electrode not in ('Negative', 'Positive'):
        raise ValueError(electrode)
    porosity = float(np.mean(volume.data == pore_phase))
    if not 0 < porosity < 1:
        raise ValueError('Porosity must be in (0,1)')
    out = {f'{electrode} electrode porosity': porosity}
    if active_phase is None:
        other=np.unique(volume.data[volume.data!=pore_phase])
        if len(other)!=1:
            raise ValueError('Specify active_phase for a volume with binder or other solid phases')
        active_phase=other[0]
    if active_phase==pore_phase:
        raise ValueError('Active material and pore must have different phase IDs')
    active=float(np.mean(volume.data==active_phase))
    if active<=0 or active+porosity>1+1e-12:
        raise ValueError('Invalid active material fraction')
    out[f'{electrode} electrode active material volume fraction']=active
    if tortuosity is not None:
        if not np.isfinite(tortuosity) or tortuosity < 1: raise ValueError('Finite tortuosity >=1 required')
        # PyBaMM epsilon**b equals epsilon/tau.
        out[f'{electrode} electrode Bruggeman coefficient (electrolyte)'] = float(1 - np.log(tortuosity)/np.log(porosity))
    return out
