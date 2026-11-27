"""Bounded raw CT inspection; no registration/segmentation claims."""
import json,os
from pathlib import Path
import numpy as np,tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from morqex import Workspace

ws=Workspace(os.environ['MORQEX_WORKSPACE']);root=ws.path('data/parks2023');out=ws.path('reports/parks2023');out.mkdir(parents=True,exist_ok=True)
records=[];images=[]
for name in ('Pristine_Merged_3DTiff.tif','Charged_Merged_3DTiff.tif'):
    with tifffile.TiffFile(root/name) as f:
        series=f.series[0];n=len(series.pages);indices=sorted(set([n//4,n//2,3*n//4]))
        sample=[]
        for index in indices:
            page=series.pages[index]
            if len(page.shape)!=2 or np.prod(page.shape)>16*1024**2:raise ValueError('Require bounded2D TIFF page reads')
            sample.append(page.asarray(maxworkers=1))
        metadata={'name':name,'shape':series.shape,'axes':series.axes,'dtype':str(series.dtype),'page_count':n,
            'compression':str(f.pages[0].compression),'sampled_page_indices':indices,
            'imagej_metadata':f.imagej_metadata,'description':(f.pages[0].description or '')[:5000],
            'XResolution':str(f.pages[0].tags.get('XResolution')),'YResolution':str(f.pages[0].tags.get('YResolution')),
            'ResolutionUnit':str(f.pages[0].tags.get('ResolutionUnit')),'scope':'Raw greyscale; voxel axis physical interpretation/registration/phase labels not yet qualified'}
        records.append(metadata);images.append(sample)
fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
values=np.concatenate([a.ravel()[::64] for row in images for a in row]);lo,hi=np.percentile(values,[.5,99.5])
for row,(record,samples) in enumerate(zip(records,images)):
    for col,(index,a) in enumerate(zip(record['sampled_page_indices'],samples)):
        axes[row,col].imshow(a,cmap='gray',vmin=lo,vmax=hi);axes[row,col].set_title(record['name'].split('_')[0]+f' raw page{index}');axes[row,col].axis('off')
fig.savefig(out/'raw_slices.png',dpi=160);plt.close(fig)
(out/'metadata.json').write_text(json.dumps(records,indent=2,default=str));print(json.dumps(records,indent=2,default=str))
import fitz
pdf=fitz.open(ws.path('references/parks2023/article.pdf'));page=pdf[7]
page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(str(out/'article_page8.png'))
for k,info in enumerate(page.get_images()):
    extracted=pdf.extract_image(info[0])
    (out/f"article_page8_image{k}.{extracted['ext']}").write_bytes(extracted['image'])
print(json.dumps({'pdf_page':8,'embedded_images':len(page.get_images()),'vector_paths':len(page.get_drawings())}))
