"""Scientific orthogonal slices from complete, converged FP64 reference fields."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import tifffile
from morqex.visualization.voxel_fields import file_hash

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='nmc1cal');args=parser.parse_args();ws=Path(os.environ['MORQEX_WORKSPACE']);directory=ws/'results/full_voxel'/args.dataset/'weak_0e+00';report=json.loads((directory/'result.json').read_text())
    if not report['metrics']['converged'] or 'fields' not in report:raise ValueError('Require converged and fully exported reference fields')
    path=Path(report['fields']['files']['von_mises_Pa']['path'])
    if file_hash(path)!=report['fields']['files']['von_mises_Pa']['sha256']:raise ValueError('Field checksum mismatch')
    raw=report['source'];phase=tifffile.imread(raw['path']).transpose(raw['axes_to_zyx']);field=np.load(path,mmap_mode='r');h=raw['voxel_size_m']*1e6;vmax=max(p['maximum_von_mises_Pa'] for p in report['fields']['phase_statistics'].values())/1e6
    figure,axes=plt.subplots(1,3,figsize=(12.5,4.1),layout='constrained')
    for axis,(ax,horizontal,vertical) in enumerate(zip(axes,['Y','X','X'],['Z','Z','Y'])):
        image=np.ma.masked_where(np.take(phase,phase.shape[2-axis]//2,axis=2-axis)==0,np.take(field,field.shape[2-axis]//2,axis=2-axis)/1e6)
        artist=ax.imshow(image,origin='lower',extent=(0,image.shape[1]*h,0,image.shape[0]*h),norm=SymLogNorm(linthresh=1.,vmin=0,vmax=vmax,base=10),cmap='magma',interpolation='nearest')
        ax.set(xlabel=horizontal+' [um]',ylabel=vertical+' [um]',title=['X mid-plane','Y mid-plane','Z mid-plane'][axis]);ax.set_facecolor('#edf1f3')
    figure.colorbar(artist,ax=axes,label='von Mises [MPa], symlog, linear below 1 MPa',shrink=.8)
    figure.suptitle(args.dataset+' | full-volume reference | active dc = 5000 mol/m3, periodic mean strain = 0',fontsize=11)
    destination=ws/'reports'/f'{args.dataset}_full_reference_stress.png';figure.savefig(destination,dpi=180);plt.close(figure)
    record={'dataset':args.dataset,'image':str(destination),'image_sha256':file_hash(destination),'field_sha256':report['fields']['files']['von_mises_Pa']['sha256'],
        'scope':'orthogonal slices of complete FP64 fixed-load reference; pores masked; common symlog scale linear below 1 MPa reaches full-volume maximum; local interface/corner peaks need mesh convergence; not DFN cycle evolution or a fracture criterion'}
    (ws/'reports/full_reference_figure.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
