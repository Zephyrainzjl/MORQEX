"""Orthogonal mechanical atlas from verified complete-volume reference arrays."""
import argparse
import json
import mmap
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm,Normalize
import tifffile
from morqex.visualization.voxel_fields import file_hash

def extract_plane(array,axis,index):
    key=[slice(None)]*array.ndim;key[array.ndim-3+axis]=index
    result=np.array(array[tuple(key)],copy=True)
    array._mmap.madvise(mmap.MADV_DONTNEED)
    return result

def slice_quantities(u,s,e,energy):
    """Kelvin xx,yy,zz,yz,xz,xy; tension-positive hydrostatic stress."""
    hydro=s[:3].sum(axis=0)/3
    tensor=np.empty((*hydro.shape,3,3))
    for k in range(3):tensor[...,k,k]=s[k]
    for k,(i,j) in enumerate([(1,2),(0,2),(0,1)],3):tensor[...,i,j]=tensor[...,j,i]=s[k]/np.sqrt(2)
    principal=np.linalg.eigvalsh(tensor)[...,-1]
    deviator=e.copy();deviator[:3]-=e[:3].sum(axis=0)/3
    return [np.sqrt(np.sum(u*u,axis=0))*1e9,hydro/1e6,principal/1e6,
        np.sqrt(1.5*(np.sum((s[:3]-hydro)**2,axis=0)+np.sum(s[3:]**2,axis=0)))/1e6,
        np.sqrt(2/3*np.sum(deviator*deviator,axis=0))*100,energy/1e6]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='nmc1cal');args=parser.parse_args()
    ws=Path(os.environ['MORQEX_WORKSPACE']);directory=ws/'results/full_voxel'/args.dataset/'weak_0e+00';record=json.loads((directory/'result.json').read_text())
    if not record['metrics']['converged']:raise ValueError('Require a converged reference state')
    arrays={}
    for name,meta in record['fields']['files'].items():
        if file_hash(meta['path'])!=meta['sha256']:raise ValueError('Field checksum mismatch: '+name)
        arrays[name]=np.load(meta['path'],mmap_mode='r')
    source=record['source']
    if file_hash(source['path'])!=source['sha256']:raise ValueError('CT checksum mismatch')
    phase=tifffile.imread(source['path']).transpose(source['axes_to_zyx']);h=source['voxel_size_m']*1e6;planes=[];checks=[]
    # X/Y/Z normal planes, displayed horizontal/vertical: YZ, XZ, XY.
    for axis in [2,1,0]:
        index=phase.shape[axis]//2;mask=np.take(phase,index,axis=axis)==0
        u=extract_plane(arrays['displacement_m'],axis,index);s=extract_plane(arrays['stress_Pa'],axis,index);e=extract_plane(arrays['strain'],axis,index);energy=extract_plane(arrays['elastic_energy_J_m3'],axis,index)
        values=slice_quantities(u,s,e,energy);vm=extract_plane(arrays['von_mises_Pa'],axis,index)/1e6
        np.testing.assert_allclose(values[3],vm,rtol=1e-12,atol=1e-10)
        checks.append(float(np.max(abs(values[3]-vm))));planes.append([np.ma.masked_where(mask,v) for v in values])
    labels=['Displacement magnitude [nm]','Hydrostatic stress [MPa], tension positive','Maximum principal stress [MPa]',
        'von Mises stress [MPa]','Equivalent total deviatoric strain [%]','Elastic energy density [MJ/m3]']
    figure,axes=plt.subplots(6,3,figsize=(12.5,20),layout='constrained');limits=[]
    for row,label in enumerate(labels):
        lo=min(float(p[row].min()) for p in planes);hi=max(float(p[row].max()) for p in planes)
        if row in [1,2]:
            bound=max(abs(lo),abs(hi),1e-12);norm=Normalize(-bound,bound);cmap='RdBu_r';scale='linear, symmetric'
        elif row==3:
            # Keep the complete-volume extreme visible without clipping it.
            hi=max(float(a['maximum_von_mises_Pa']) for a in record['fields']['phase_statistics'].values())/1e6
            norm=SymLogNorm(linthresh=1.,vmin=0,vmax=max(hi,1e-12),base=10);cmap='magma';scale='symlog, linear <1 MPa'
        else:norm=Normalize(0,max(hi,1e-12));cmap='viridis';scale='linear'
        limits.append({'quantity':label,'color_min':float(norm.vmin),'color_max':float(norm.vmax),'scale':scale})
        for col,(ax,horizontal,vertical) in enumerate(zip(axes[row],['Y','X','X'],['Z','Z','Y'])):
            data=planes[col][row];artist=ax.imshow(data,origin='lower',extent=(0,data.shape[1]*h,0,data.shape[0]*h),norm=norm,cmap=cmap,interpolation='nearest')
            ax.set(xlabel=horizontal+' [um]',ylabel=vertical+' [um]');ax.set_facecolor('#edf1f3')
            if row==0:ax.set_title(['X mid-plane','Y mid-plane','Z mid-plane'][col])
        figure.colorbar(artist,ax=list(axes[row]),label=label+'\n'+scale,shrink=.9)
    shape_label=' x '.join(map(str,phase.shape))
    figure.suptitle(args.dataset+' | full-volume FP64 fixed-load reference | '+shape_label+' voxels\nactive dc='+str(record['input_concentration_deviation_mol_m3'])+' mol/m3, mean strain='+str(record['input_macro_strain'])+', dT='+str(record['input_temperature_deviation_K'])+' K | pores masked',fontsize=10)
    output=ws/'reports'/f'{args.dataset}_full_mechanical_atlas';figure.savefig(output.with_suffix('.png'),dpi=180);figure.savefig(output.with_suffix('.pdf'));plt.close(figure)
    result={'dataset':args.dataset,'png':str(output.with_suffix('.png')),'pdf':str(output.with_suffix('.pdf')),'source_result':str(directory/'result.json'),
        'shape_zyx':phase.shape,'voxel_size_um':h,'domain_length_um':(np.array(phase.shape)*h).tolist(),'plane_indices_zyx':[n//2 for n in phase.shape],
        'color_limits':limits,'maximum_vm_slice_consistency_error_MPa':max(checks),
        'scope':'Unmodified complete FP64 solution, shown at three orthogonal mid-planes. Displacement gauge follows saved reference. Hydrostatic stress is tension positive; equivalent strain is total deviatoric strain, not plastic strain. Shared row color scales cover displayed slices except von Mises, which also includes the global maximum. Interface extremes need mesh convergence; not a DFN trajectory or failure criterion.'}
    (ws/'reports'/f'{args.dataset}_mechanical_atlas.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
