"""Offline 2D/3D voxel preview with notebook display-ID updates.

The original label array is retained for2D slices.3D uses an explicit bounded
display LOD; no external JavaScript, web server, GPU computation or widgets.
"""
from pathlib import Path
import base64,gzip,json,html,os
import numpy as np


class VoxelView:
    def __init__(self,model,*,maximum_3d_size=48,height=660,title='MORQEX geometry',show_region=None,slice_origin='upper'):
        if type(maximum_3d_size) is not int or not 8<=maximum_3d_size<=96:
            raise ValueError('3D display size8..96 required')
        if type(height) is not int or not 300<=height<=1400:raise ValueError('Height300..1400 required')
        if show_region is not None and type(show_region) is not int:raise ValueError('show_region must be an integer region or None')
        if slice_origin not in ('upper','lower'):raise ValueError('slice_origin must be upper or lower')
        self.slice_origin=slice_origin
        self.maximum_3d_size=maximum_3d_size;self.height=height;self.title=str(title);self.show_region=show_region
        self.handle=None;self._set(model)

    def _set(self,model):
        from morqex.modeling.voxel import VoxelModel
        if not isinstance(model,VoxelModel):raise TypeError('VoxelModel required')
        self.model=model
        if self.show_region is not None and self.show_region not in model.region_names:raise ValueError('Requested display region is absent')
        a=model.regions
        if a.max()>np.iinfo(np.uint32).max:raise ValueError('Viewer requires region IDs withinUInt32')
        stride=max(1,int(np.ceil(max(a.shape)/self.maximum_3d_size)))
        preview=a[::stride,::stride,::stride]
        def pack(values):return base64.b64encode(gzip.compress(np.asarray(values,dtype='<u4').tobytes(),compresslevel=1)).decode()
        payload={'shape':a.shape,'spacing':model.spacing,'origin':model.origin,'names':model.region_names,
            'labels':pack(a),'preview':pack(preview),'previewShape':preview.shape,'stride':stride,
            'title':self.title,'metadata':model.metadata,'slice_origin':self.slice_origin,'show_region':self.show_region if self.show_region is not None else
                next((k for k,v in model.region_names.items() if 'active' in v.lower()),None)}
        template=Path(__file__).with_name('notebook_voxels.html').read_text()
        self.document=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
        workspace=Path(os.environ.get('MORQEX_WORKSPACE',Path(__file__).resolve().parents[4]/'morph_workspace/microelectrode_gallery'))
        font=workspace/'data/report_font/NotoSansCJKsc-Regular.otf'
        if font.exists():
            try:
                from .report_font import embed_font
                self.document=embed_font(self.document,font,(font.parent/'LICENSE').read_text())
            except ImportError:
                pass  # System-font fallback when the report extra is absent.

    def _repr_html_(self):
        return '<iframe style="width:100%;border:1px solid #ccd9df;border-radius:10px" height="'+str(self.height)+'" srcdoc="'+html.escape(self.document,quote=True)+'"></iframe>'

    def display(self):
        from IPython.display import HTML,display
        self.handle=display(HTML(self._repr_html_()),display_id=True)
        return self

    def _ipython_display_(self):
        if self.handle is None:self.display()

    def update(self,model):
        """Refresh the same output; also serves as from_shapes progress callback."""
        self._set(model)
        if self.handle is not None:
            from IPython.display import HTML
            self.handle.update(HTML(self._repr_html_()))
        return self

    def save_html(self,path):
        from morqex.modeling.voxel import external_path
        target=external_path(path)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(self.document)
        return target


def show(model,**kwargs):return VoxelView(model,**kwargs).display()


class MeshView(VoxelView):
    """Actual bounded Tet4 subset from FEMesh or full streamed MeshArtifact."""
    def __init__(self,mesh,*,first_cell=0,cell_count=1000,height=620,title='Actual Tet4 / 真实四面体',sampling='contiguous'):
        if type(first_cell) is not int or first_cell<0 or type(cell_count) is not int or not 1<=cell_count<=5000:
            raise ValueError('Nonnegative first cell and1..5000 preview cells required')
        self.first_cell=first_cell;self.cell_count=cell_count;self.height=height;self.title=title;self.handle=None
        if sampling not in ('contiguous','uniform'):raise ValueError('sampling must be contiguous or uniform')
        if sampling=='uniform' and first_cell!=0:raise ValueError('Uniform display sampling covers the complete mesh; first_cell must be0')
        self.sampling=sampling
        self._set(mesh)

    def _set(self,mesh):
        from morqex.modeling.meshing import MeshArtifact
        from morqex.microstructure.mesh.voxel import FEMesh
        if isinstance(mesh,MeshArtifact):
            import h5py
            with h5py.File(mesh.hdf5) as f:
                total=len(f['mesh/tetrahedra']);end=min(total,self.first_cell+self.cell_count)
                if self.first_cell>=end:raise ValueError('Preview outside stored mesh')
                selected=np.linspace(0,total-1,min(self.cell_count,total),dtype=np.int64) if self.sampling=='uniform' else np.arange(self.first_cell,end)
                cells=f['mesh/tetrahedra'][selected];ids,inverse=np.unique(cells,return_inverse=True)
                points=f['mesh/points_m'][ids];cells=inverse.reshape(-1,4)
                phase=f['mesh/phase'][selected]
        elif isinstance(mesh,FEMesh) and mesh.cell_type=='tetra':
            total=len(mesh.cells);end=min(total,self.first_cell+self.cell_count)
            if self.first_cell>=end:raise ValueError('Preview outside mesh')
            selected=np.linspace(0,total-1,min(self.cell_count,total),dtype=np.int64) if self.sampling=='uniform' else np.arange(self.first_cell,end)
            original=mesh.cells[selected];ids,inverse=np.unique(original,return_inverse=True)
            points=mesh.points[ids];cells=inverse.reshape(-1,4);phase=mesh.phases[selected]
        else:raise TypeError('MeshArtifact or tetrahedral FEMesh required')
        self.model=mesh
        payload={'points':points.tolist(),'cells':cells.tolist(),'phase':phase.tolist(),'total':total,
            'first':self.first_cell,'title':self.title,'sampling':self.sampling,'cell_indices':selected.tolist()}
        self.document=Path(__file__).with_name('notebook_tetra.html').read_text().replace('__DATA__',json.dumps(payload,separators=(',',':')).replace('<','\\u003c'))
        workspace=Path(os.environ.get('MORQEX_WORKSPACE',Path(__file__).resolve().parents[4]/'morph_workspace/microelectrode_gallery'))
        font=workspace/'data/report_font/NotoSansCJKsc-Regular.otf'
        if font.exists():
            try:
                from .report_font import embed_font
                self.document=embed_font(self.document,font,(font.parent/'LICENSE').read_text())
            except ImportError:
                pass
