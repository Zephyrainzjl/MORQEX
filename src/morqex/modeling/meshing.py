"""Bounded-memory conforming Freudenthal Tet4 using the integrated GEM module."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import h5py
from morqex.microstructure import Volume
from morqex.microstructure.mesh.voxel import voxel_to_tetra


@dataclass(frozen=True)
class MeshArtifact:
    hdf5:Path
    xdmf:Path
    nodes:int
    tetrahedra:int
    selected_regions:tuple

    @classmethod
    def load(cls,path):
        """Open a complete voxel or TetGen HDF5 artifact without loading cells."""
        path=Path(path).expanduser().resolve()
        with h5py.File(path,'r') as f:
            if not f.attrs.get('complete',False):raise ValueError('Incomplete mesh artifact')
            points=f['mesh/points_m'];cells=f['mesh/tetrahedra']
            if points.shape[1:]!=(3,) or cells.shape[1:]!=(4,) or len(f['mesh/phase'])!=len(cells):
                raise ValueError('Invalid Tet4 artifact array shapes')
            selected=f.attrs.get('selected_regions')
            if selected is None:
                labels=set();phase=f['mesh/phase']
                for start in range(0,len(phase),65536):labels.update(map(int,np.unique(phase[start:start+65536])))
                selected=sorted(labels)
            return cls(path,path.with_suffix('.xdmf'),len(points),len(cells),tuple(map(int,selected)))

    def viewer(self,**kwargs):
        from morqex.visualization.notebook import MeshView
        return MeshView(self,**kwargs)

    def show(self,**kwargs):return self.viewer(**kwargs).display()

    def export_comsol(self,path,**kwargs):
        from .comsol import export_comsol
        return export_comsol(self,path,**kwargs)


def tetrahedralize(model,*,regions=None,maximum_voxels=262144):
    if model.regions.size>maximum_voxels:raise ValueError('Use mesh_to/stream_tetrahedralize for larger arrays')
    return voxel_to_tetra(model.volume(),phases=regions)


def stream_tetrahedralize(model,path,*,regions=None,slab_depth=2,compression=1):
    """Global compact shared nodes, exact parent labels, bounded cell slabs.

    Only mesh_to stores the complete mesh at256^3. No full global element or
    gradient matrix is allocated. Nodes touching only along an edge/corner are
    shared, following voxel_to_tetra's bonded shared-grid convention.
    """
    from xml.sax.saxutils import escape
    from .voxel import external_path
    path=external_path(path)
    a=model.regions;shape=np.array(a.shape);spacing=np.asarray(model.spacing);origin=np.asarray(model.origin)
    chosen=tuple(map(int,np.unique(a) if regions is None else regions))
    if not chosen or not set(chosen)<=set(map(int,np.unique(a))):raise ValueError('Select existing region labels explicitly')
    if type(slab_depth) is not int or not 1<=slab_depth<=16:raise ValueError('Slab depth must be1..16')
    mask=np.isin(a,chosen);nvox=int(np.count_nonzero(mask))
    if not nvox:raise ValueError('No selected cells')
    flags=np.zeros(tuple(shape+1),bool)
    for z in (0,1):
        for y in (0,1):
            for x in (0,1):flags[z:z+shape[0],y:y+shape[1],x:x+shape[2]]|=mask
    nodes=int(flags.sum());mapping=np.cumsum(flags.ravel(),dtype=np.uint32)-1
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Do not overwrite an existing mesh artifact')
    pending=path.with_suffix('.pending');total=6*nvox;volume_sum=0.;minimum=np.inf;cursor=0;blocks=0
    with h5py.File(pending,'w') as f:
        points=f.create_dataset('mesh/points_m',(nodes,3),dtype='<f8',chunks=(min(nodes,65536),3),compression='gzip',compression_opts=compression,shuffle=True)
        ids=np.flatnonzero(flags.ravel());del flags
        for start in range(0,nodes,65536):
            selected=ids[start:start+65536]
            xyz=np.column_stack(np.unravel_index(selected,tuple(shape+1)))
            points[start:start+len(selected)]=(xyz*spacing+origin)[:,::-1]
        del ids
        cells=f.create_dataset('mesh/tetrahedra',(total,4),dtype='<u4',chunks=(min(total,65536),4),compression='gzip',compression_opts=compression,shuffle=True)
        phase=f.create_dataset('mesh/phase',(total,),dtype=a.dtype,chunks=True,compression='gzip',compression_opts=compression,shuffle=True)
        parent=f.create_dataset('mesh/parent_voxel',(total,),dtype='<u4',chunks=True,compression='gzip',compression_opts=compression,shuffle=True)
        f.create_dataset('phase_labels',data=a,chunks=True,compression='gzip',compression_opts=compression,shuffle=True)
        for start in range(0,shape[0],slab_depth):
            stop=min(start+slab_depth,shape[0]);block=a[start:stop]
            local_selected=np.flatnonzero(np.isin(block,chosen))
            if not len(local_selected):continue
            local_origin=origin+np.array([start,0,0])*spacing
            mesh=voxel_to_tetra(Volume(block,tuple(spacing),tuple(local_origin)),phases=chosen)
            grid=np.rint((mesh.points[:,::-1]-origin)/spacing).astype(np.int64)
            node_ids=mapping[np.ravel_multi_index(grid.T,tuple(shape+1))]
            count=len(mesh.cells);volumes=mesh.signed_volumes()
            if np.any(volumes<=0):raise ArithmeticError('Negative/degenerate streamed Tet4')
            volume_sum+=float(volumes.sum());minimum=min(minimum,float(volumes.min()))
            global_parent=np.repeat(local_selected+start*shape[1]*shape[2],6)
            if not np.array_equal(mesh.phases,a.ravel()[global_parent]):raise ArithmeticError('Parent phase mismatch')
            cells[cursor:cursor+count]=node_ids[mesh.cells];phase[cursor:cursor+count]=mesh.phases;parent[cursor:cursor+count]=global_parent
            cursor+=count;blocks+=1
            print(json.dumps({'slab_end':int(stop),'cells_written':cursor,'total_cells':total}),flush=True)
        error=abs(volume_sum/(nvox*np.prod(spacing))-1)
        if cursor!=total or error>1e-11:raise ArithmeticError('Streamed mesh count/volume failed')
        f.attrs.update(complete=True,shape_zyx=shape,spacing_zyx_m=spacing,origin_zyx_m=origin,selected_regions=chosen,
            node_count=nodes,tetrahedra=total,minimum_signed_volume_m3=minimum,volume_relative_error=error,
            topology_policy='shared grid nodes including edge/corner contacts',blocks=blocks,
            region_names=json.dumps(model.region_names),metadata=json.dumps(model.metadata))
    pending.replace(path)
    xdmf=path.with_suffix('.xdmf');filename=escape(path.name)
    xdmf.write_text(f'''<?xml version="1.0"?><Xdmf Version="3.0"><Domain><Grid Name="MORQEX Tet4" GridType="Uniform">
<Topology TopologyType="Tetrahedron" NumberOfElements="{total}"><DataItem Dimensions="{total} 4" NumberType="UInt" Precision="4" Format="HDF">{filename}:/mesh/tetrahedra</DataItem></Topology>
<Geometry GeometryType="XYZ"><DataItem Dimensions="{nodes} 3" NumberType="Float" Precision="8" Format="HDF">{filename}:/mesh/points_m</DataItem></Geometry>
<Attribute Name="region" Center="Cell"><DataItem Dimensions="{total}" NumberType="UInt" Precision="{a.dtype.itemsize}" Format="HDF">{filename}:/mesh/phase</DataItem></Attribute>
<Attribute Name="parent_voxel" Center="Cell"><DataItem Dimensions="{total}" NumberType="UInt" Precision="4" Format="HDF">{filename}:/mesh/parent_voxel</DataItem></Attribute>
</Grid></Domain></Xdmf>''')
    return MeshArtifact(path,xdmf,nodes,total,chosen)
