"""Numerical history gates, deliberately separate from physical qualification."""
import json
from pathlib import Path
import h5py
import numpy as np


def verify_dense_history(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    finite=manifest.get('mechanics_model')=='finite-strain'
    minimum_jacobian=1.
    previous = None
    maximum = {'lithium_defect':0., 'state_jump':0., 'mechanics_residual':0.,
               'damage_kkt':0., 'transport_residual':0., 'transport_energy_defect':0.,
               'capacity_integral_defect_Ah':0.}
    records = []
    prior_damage = prior_fatigue = phase = None
    with h5py.File(directory/'mesh.h5','r') as f:
        points, cells, material = f['points_m'][:], f['tetrahedra'][:], f['phases'][:]
    coordinates = points[cells]
    jac = np.swapaxes(coordinates[:,1:]-coordinates[:,:1],1,2)
    gradients = np.einsum('ij,ejk->eik',np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]]),np.linalg.inv(jac))
    maximum_strain = 0.
    phase_strain = {str(int(p)):{'maximum':0.,'maximum_P99':0.,'maximum_fraction_above_0_02':0.} for p in np.unique(material)}
    for expected, path in enumerate(sorted(directory.glob('step_*.h5'))):
        with h5py.File(path, 'r') as f:
            r = json.loads(f.attrs['record'])
            if f.attrs['identity'] != manifest['identity'] or not f.attrs['complete'] or r['step'] != expected:
                raise ValueError('Accepted checkpoint identity/order failed')
            damage, fatigue, labels = f['damage'][:], f['fatigue'][:], f['voxel/phase_labels'][:]
            u = f['displacement_m'][:]
            if not all(np.isfinite(a).all() for a in (fatigue,u,f['voxel/concentration_mol_m3'][:],f['voxel/stress_Mandel_Pa'][:])):
                raise ValueError('Nonfinite physical field')
            grad_u = np.einsum('eai,eaj->eij',u[cells],gradients)
            if finite:
                F=np.eye(3)+grad_u;J=np.linalg.det(F)
                minimum_jacobian=min(minimum_jacobian,float(J.min()))
                if np.any(J<=0):raise ValueError('Finite-strain positive determinant failed')
                if not np.allclose(F,f['finite/deformation_gradient'][:],atol=1e-12,rtol=1e-12):raise ValueError('Stored deformation gradient disagrees with displacement')
                strain=.5*(np.swapaxes(F,1,2)@F-np.eye(3))
            else:strain = (grad_u+np.swapaxes(grad_u,1,2))*.5
            maximum_strain = max(maximum_strain,float(np.linalg.norm(strain,axis=(1,2)).max()))
            norm = np.linalg.norm(strain,axis=(1,2))
            for p, stats in phase_strain.items():
                values = norm[material==int(p)]
                stats['maximum'] = max(stats['maximum'],float(values.max()))
                stats['maximum_P99'] = max(stats['maximum_P99'],float(np.quantile(values,.99)))
                stats['maximum_fraction_above_0_02'] = max(stats['maximum_fraction_above_0_02'],float(np.mean(values>.02)))
            if not np.isfinite(damage).all() or damage.min() < -1e-12 or damage.max() > 1+1e-12:
                raise ValueError('Damage bounds failed')
            if prior_damage is not None:
                if np.min(damage-prior_damage)<-1e-12 or np.min(fatigue-prior_fatigue)<-1e-9:
                    raise ValueError('Accepted damage/fatigue irreversibility failed')
                if not np.array_equal(labels,phase):raise ValueError('Unexpected material deletion in diffuse-only model')
                if r['time_s'] <= previous['time_s']:raise ValueError('Physical time did not advance')
                if r['feedback'] != previous['next_feedback']:raise ValueError('Topology/transport feedback port was bypassed')
                t, current, q = (f['curve/'+k][:] for k in ('time_s','current_A','net_capacity_Ah'))
                if np.min(np.diff(t))<0:raise ValueError('DFN curve time order failed')
                integrated = np.trapezoid(current,t)/3600
                error = abs(integrated-(q[-1]-q[0]))
                maximum['capacity_integral_defect_Ah'] = max(maximum['capacity_integral_defect_Ah'],float(error))
                if error>1e-9:raise ValueError('Capacity is not consistent with actual current integration')
            for name, key, bound in [('lithium_defect','inventory_relative_defect',1e-8),
                ('state_jump','relative_state_jump',1e-9),('mechanics_residual','equilibrium_residual',1e-8),
                ('damage_kkt','damage_kkt',1e-6),('transport_residual','transport_residual',5e-10),
                ('transport_energy_defect','transport_energy_defect',1e-6)]:
                value=r[key]
                if not np.isfinite(value) or abs(value)>bound:raise ValueError(name+' gate failed')
                maximum[name] = max(maximum[name],abs(value))
            prior_damage, prior_fatigue, phase, previous = damage, fatigue, labels, r
            records.append(r)
    if not records:raise ValueError('Empty history')
    if records[-1]['step']!=manifest.get('last_accepted_step'):raise ValueError('Manifest/checkpoint progress mismatch')
    if manifest.get('protocol_complete'):
        if manifest['config'].get('protocol')=='voltage-window':
            events=[r['protocol_phase'] for r in records if r.get('completed_phase_capacity_Ah') is not None]
            expected=['conditioning_charge']+['discharge','charge']*manifest['config']['cycles']
            if events!=expected:raise ValueError('Missing full voltage-cutoff cycle')
            if records[-1]['completed_cycles']!=manifest['config']['cycles']:raise ValueError('Cycle count mismatch')
        elif records[-1]['step']!=manifest['requested_steps']:raise ValueError('Partial-depth protocol incomplete')
    return {'directory':str(directory),'accepted_states':len(records),'numerical_history_passed':True,
        'protocol_complete':manifest.get('protocol_complete',False),'maximum_errors':maximum,
        'maximum_cell_total_strain_frobenius':maximum_strain,
        'phase_total_strain':phase_strain,
        'small_strain_screen_passed':None if finite else maximum_strain<=.02,
        'strain_measure':'Green-Lagrange' if finite else 'infinitesimal symmetric displacement gradient',
        'minimum_deformation_jacobian':minimum_jacobian if finite else None,
        'small_strain_screen_definition':'Conservative diagnostic max ||sym grad u||_F <= 0.02; not a rigorous error bound. Phase fractions are Tet4 volume fractions on this uniform voxel split.',
        'physical_topology_qualified':False,'material_validation_passed':False,
        'scope':'Discrete consistency only; not mesh/time convergence or calibrated crack geometry.'}
