import numpy as np

def slices(volume, indices=None):
    import matplotlib.pyplot as plt
    a = np.asarray(getattr(volume, 'data', volume))
    indices = indices or tuple(n // 2 for n in a.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.imshow(np.take(a, indices[i], axis=i), interpolation='nearest')
        ax.set_title(('XY', 'XZ', 'YZ')[i]); ax.set_axis_off()
    fig.tight_layout()
    return fig

def mesh(femesh, **kwargs):
    import pyvista as pv
    types = {'tetra': pv.CellType.TETRA, 'hexahedron': pv.CellType.HEXAHEDRON,
             'triangle': pv.CellType.TRIANGLE}
    cells = np.column_stack((np.full(len(femesh.cells), femesh.cells.shape[1]), femesh.cells))
    grid = pv.UnstructuredGrid(cells.ravel(), np.full(len(cells), types[femesh.cell_type]), femesh.points)
    grid.cell_data['phase'] = femesh.phases
    return grid.plot(**kwargs)

def raytrace(volume,phases=None,colors=None,start=True):
    """Optional OptiX voxel rendering without hard-coded files or scene dimensions."""
    from plotoptix import TkOptiX
    a=np.asarray(getattr(volume,'data',volume))
    spacing=np.asarray(getattr(volume,'spacing',(1.,1.,1.)))[::-1]
    origin=np.asarray(getattr(volume,'origin',(0.,0.,0.)))[::-1]
    phases=list(np.unique(a) if phases is None else phases)
    import matplotlib.pyplot as plt
    renderer=TkOptiX(start_now=False)
    renderer.set_param(min_accumulation_step=4,max_accumulation_frames=1000)
    for i,phase in enumerate(phases):
        positions=np.argwhere(a == phase)[:,::-1]*spacing+origin
        if not len(positions):continue
        color=colors[phase] if colors else plt.get_cmap('viridis')(i/max(1,len(phases)-1))[:3]
        renderer.set_data(str(phase),pos=positions,u=[spacing[0],0,0],v=[0,spacing[1],0],
                          w=[0,0,spacing[2]],geom='Parallelepipeds',mat='diffuse',c=color)
    extent=np.asarray(a.shape[::-1])*spacing
    center=origin+extent/2
    renderer.setup_camera('camera',cam_type='Pinhole',eye=center+2*extent,target=center,up=[0,0,1],fov=35)
    renderer.set_background(.15)
    renderer.setup_light('light',pos=center+3*extent,color=[10,10,10],radius=float(extent.max()))
    if start:renderer.start()
    return renderer
