"""Actual irregular surface triangles and exact Tet4 plane intersections."""
import base64,gzip,html,json,os
from pathlib import Path
import numpy as np


def section_polygons(mesh,axis=0,fraction=.5):
    """Cut real Tet4 cells by one physical plane. No voxel/surrogate mesh."""
    from morqex.modeling.comsol import _arrays
    if axis not in (0,1,2) or not 0<fraction<1:raise ValueError('XYZ axis0..2 and fraction in(0,1)')
    polygons=[];labels=[]
    with _arrays(mesh) as (points,cells,phase,kind):
        if kind!='tetra':raise ValueError('Tet4 section required')
        # h5py fancy indexing tens of thousands of irregular node IDs can
        # dominate a section. Cache at most48MB of nodes, or gather through
        # bounded contiguous blocks. Cell connectivity stays streamed.
        node_cache=points if isinstance(points,np.ndarray) else (points[:] if len(points)<=2000000 else None)
        low=np.full(3,np.inf);high=np.full(3,-np.inf)
        for i in range(0,len(points),50000):
            block=np.asarray(points[i:i+50000]);low=np.minimum(low,block.min(axis=0));high=np.maximum(high,block.max(axis=0))
        plane=low[axis]+fraction*(high[axis]-low[axis]);other=[i for i in range(3) if i!=axis]
        tolerance=max(high-low)*1e-12
        for i in range(0,len(cells),50000):
            block=np.asarray(cells[i:i+50000])
            if node_cache is not None:p=node_cache[block]
            else:
                ids,inverse=np.unique(block,return_inverse=True);gathered=np.empty((len(ids),3),float)
                groups=ids//32768;unique,starts=np.unique(groups,return_index=True)
                ends=np.r_[starts[1:],len(ids)]
                for group,start,end in zip(unique,starts,ends):
                    base=int(group)*32768;buffer=np.asarray(points[base:base+32768])
                    gathered[start:end]=buffer[ids[start:end]-base]
                p=gathered[inverse.reshape(-1,4)]
            # Negative-side half-open ownership avoids double-counting a face
            # shared by two cells when the cutting plane passes through it.
            value=p[:,:,axis]-plane;cross=(value.min(axis=1)<-tolerance)&(value.max(axis=1)>=-tolerance)
            for vertices,signs,label in zip(p[cross],value[cross],np.asarray(phase[i:i+50000])[cross]):
                intersections=[vertices[k] for k in range(4) if abs(signs[k])<=tolerance]
                for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
                    if signs[a]*signs[b]<0:
                        t=signs[a]/(signs[a]-signs[b]);intersections.append(vertices[a]+t*(vertices[b]-vertices[a]))
                if not intersections:continue
                q=np.unique(np.rint((np.asarray(intersections)-low)/tolerance).astype(np.int64),axis=0)*tolerance+low
                if len(q)<3:continue
                xy=q[:,other];centre=xy.mean(axis=0);order=np.argsort(np.arctan2(xy[:,1]-centre[1],xy[:,0]-centre[0]))
                polygons.append(xy[order].tolist());labels.append(int(label))
    return {'axis':axis,'plane_m':float(plane),'axes':other,'polygons':polygons,'phase':labels,
            'bounds':[[float(low[i]),float(high[i])] for i in other],
            'scope':'actual Tet4-plane intersections, negative-side half-open ownership on shared coplanar faces; polygons are not a new FE discretization'}


class AdaptiveMeshView:
    def __init__(self,mesh,*,sections=True,height=820,title='GEMsim adaptive mesh / 自适应贴体网格'):
        from morqex.modeling.tetgen import SurfacePLC
        from morqex.modeling.meshing import MeshArtifact
        self.height=height;self.title=title;self.handle=None
        if isinstance(mesh,SurfacePLC):
            points=mesh.points;markers=mesh.markers;report=mesh.metadata;faces=[]
            diagnostics={'geometry':report,'scope':'surface only; no volume mesh'}
            repeated=[]
            for f,m in zip(mesh.facets,markers):
                faces.extend([[f[0],f[i],f[i+1]] for i in range(1,len(f)-1)])
                repeated.extend([m]*(len(f)-2))
            faces=np.asarray(faces,np.int32);markers=np.asarray(repeated,np.int32);cuts=[]
        elif isinstance(mesh,MeshArtifact):
            import h5py
            with h5py.File(mesh.hdf5) as f:
                points=f['mesh/points_m'][:];faces=f['mesh/facets'][:];markers=f['mesh/facet_markers'][:]
                diagnostics=json.loads(f.attrs['report']);report=diagnostics['geometry']
            cuts=[section_polygons(mesh,axis) for axis in range(3)] if sections else []
        else:
            points=mesh.points;faces=mesh.facets;markers=mesh.facet_markers;diagnostics=mesh.report;report=diagnostics['geometry']
            cuts=[section_polygons(mesh,axis) for axis in range(3)] if sections else []
        used,inverse=np.unique(faces,return_inverse=True);points=points[used];faces=inverse.reshape(-1,3)
        def pack(a,dtype):return base64.b64encode(gzip.compress(np.asarray(a,dtype=dtype).tobytes(),compresslevel=1)).decode()
        render_origin=points.min(axis=0)
        payload={'points':pack(points-render_origin,'<f4'),'origin_m':render_origin.tolist(),
            'faces':pack(faces,'<u4'),'markers':pack(markers,'<u4'),
            'pairs':report.get('interface_pairs',{}),'names':report.get('region_names',{}),
            'sections':cuts,'title':title,'report':diagnostics,'point_count':len(points),'face_count':len(faces)}
        template=Path(__file__).with_name('adaptive_mesh.html').read_text()
        self.document=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
        workspace=os.environ.get('MORQEX_WORKSPACE')
        if workspace:
            font=Path(workspace)/'data/report_font/NotoSansCJKsc-Regular.otf'
            if font.exists():
                try:
                    from .report_font import embed_font
                    self.document=embed_font(self.document,font,(font.parent/'LICENSE').read_text())
                except ImportError:pass

    def _repr_html_(self):
        return f'<iframe style="width:100%;border:0" height="{self.height}" srcdoc="{html.escape(self.document,quote=True)}"></iframe>'

    def display(self):
        from IPython.display import HTML,display
        self.handle=display(HTML(self._repr_html_()),display_id=True);return self

    def _ipython_display_(self):self.display()

    def save_html(self,path):
        from morqex.modeling.voxel import external_path
        path=external_path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(self.document);return path
