"""Stream complete reference fields to external NPY files under the RSS cap."""
import hashlib
import mmap
import time
from pathlib import Path
import numpy as np

def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(2**20):digest.update(block)
    return digest.hexdigest()

def export_reference_fields(model,u,c,directory,*,macro=(0.,)*6,temperature=0.,chunk=131072):
    """Exact discrete gradients/stress; no LOD in the saved numerical fields."""
    started=time.perf_counter();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);xp=model.xp;n=model.n;shape=model.shape
    stress=model.stress(u,c,macro,temperature);flat=stress.reshape(6,-1);U=u.reshape(3,-1);c=c.ravel()
    actual_macro=model.macro_strain(u,c.reshape(shape),temperature) if hasattr(model,'macro_strain') else xp.asarray(macro,dtype='float64')
    E=actual_macro;affine=xp.stack([xp.stack([E[0],E[5]/np.sqrt(2),E[4]/np.sqrt(2)]),xp.stack([E[5]/np.sqrt(2),E[1],E[3]/np.sqrt(2)]),xp.stack([E[4]/np.sqrt(2),E[3]/np.sqrt(2),E[2]])])
    displacement_mean=xp.mean(U,axis=1)
    fields={'displacement_m':np.lib.format.open_memmap(directory/'displacement_m.npy',mode='w+',dtype='float64',shape=(3,*shape)),
        'stress_Pa':np.lib.format.open_memmap(directory/'stress_Pa.npy',mode='w+',dtype='float64',shape=(6,*shape)),
        'strain':np.lib.format.open_memmap(directory/'strain.npy',mode='w+',dtype='float64',shape=(6,*shape)),
        'elastic_energy_J_m3':np.lib.format.open_memmap(directory/'elastic_energy_J_m3.npy',mode='w+',dtype='float64',shape=shape),
        'von_mises_Pa':np.lib.format.open_memmap(directory/'von_mises_Pa.npy',mode='w+',dtype='float64',shape=shape)}
    host=lambda a:np.asarray(a) if model.backend=='cpu' else xp.asnumpy(a)
    ratio=min(model.spacing)/np.asarray(model.spacing)[::-1];stride=[1,shape[2],shape[1]*shape[2]];lengths=shape[::-1];index=np.array([[0,5,4],[5,1,3],[4,3,2]])
    energy=0.;means=np.zeros(6);source_work=0.
    def write_block(begin,end):
        nonlocal energy,means,source_work
        ids=xp.arange(begin,end);s=flat[:,begin:end];e=xp.broadcast_to(actual_macro[:,None],s.shape).copy()
        for axis in range(3):
            q=(ids//stride[axis])%lengths[axis];j=ids+xp.where(q==lengths[axis]-1,-(lengths[axis]-1)*stride[axis],stride[axis])
            for k in range(3):e[index[k,axis]]+=(U[k,j]-U[k,ids])*ratio[axis]*(1. if k==axis else 1/np.sqrt(2))
        p=model.phase.ravel()[begin:end];eigen=xp.asarray(model.beta_host)[p].T*c[begin:end]+xp.asarray(model.alpha_host)[p].T*temperature
        density=.5*xp.sum(s*(e-eigen),axis=0);hydro=xp.sum(s[:3],axis=0)/3;vm=xp.sqrt(1.5*(xp.sum((s[:3]-hydro)**2,axis=0)+xp.sum(s[3:]**2,axis=0)))
        fields['stress_Pa'].reshape(6,-1)[:,begin:end]=host(s);fields['strain'].reshape(6,-1)[:,begin:end]=host(e)
        xyz=xp.stack([(ids//stride[a])%lengths[a]*model.spacing[2-a] for a in range(3)])
        fields['displacement_m'].reshape(3,-1)[:,begin:end]=host((U[:,begin:end]-displacement_mean[:,None])*min(model.spacing)+affine@xyz)
        fields['elastic_energy_J_m3'].ravel()[begin:end]=host(density);fields['von_mises_Pa'].ravel()[begin:end]=host(vm)
        energy+=float(xp.sum(density))/n;means+=host(xp.sum(s,axis=1))/n;source_work+=float(xp.sum(s*eigen))/n
        for field in fields.values():
            field.flush();field._mmap.madvise(mmap.MADV_DONTNEED)
    for begin in range(0,n,chunk):
        end=min(n,begin+chunk)
        if model.backend=='cuda':
            with model.budget.batch():write_block(begin,end)
        else:write_block(begin,end)
    del stress
    mean_macro=host(actual_macro);stationary=.5*(float(mean_macro@means)-source_work)
    for field in fields.values():field.flush()
    paths={name:directory/(name+'.npy') for name in fields};del fields
    stats={}
    for phase in np.unique(model.phase_host):
        data=np.load(paths['von_mises_Pa'],mmap_mode='r');selected=np.asarray(data)[model.phase_host==phase]
        stats[str(int(phase))]={'volume_fraction':float(np.mean(model.phase_host==phase)),'von_mises_P95_Pa':float(np.quantile(selected,.95)),
            'von_mises_P99_Pa':float(np.quantile(selected,.99)),'maximum_von_mises_Pa':float(np.max(selected))};del selected,data
    return {'files':{name:{'path':str(path),'sha256':file_hash(path)} for name,path in paths.items()},'wall_seconds':time.perf_counter()-started,'mean_stress_Pa':means.tolist(),
        'elastic_energy_J_m3':energy,'stationarity_energy_J_m3':stationary,'relative_stationarity_energy_defect':abs(energy-stationary)/max(abs(energy),abs(stationary),1e-30),
        'phase_statistics':stats,'field_precision':'FP64','stress_strain_layout':'(Kelvin component,Z,Y,X)','macro_strain':mean_macro.tolist(),
        'displacement_gauge':'periodic fluctuation has global mean removed; total displacement includes affine strain about origin XYZ=0; disconnected components can have additional rigid-body nonuniqueness','checkpoint_to_metre_scale':min(model.spacing),
        'scope':'complete fixed-geometry reference state; phase quantiles include that phase only; not a DFN trajectory or mesh-independent fracture criterion'}
