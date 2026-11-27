"""Optional VTK multi-label Surface Nets front end with shared interfaces.

VTK retains its BSD license and is a compiled runtime dependency. No independent
per-material contour is overlaid, and source segmentation is never filtered.
"""
import numpy as np
from .tetgen import SurfacePLC,_component_seeds


def surface_nets_plc(model,*,regions=None,iterations=8,constraint=.2,maximum_facets=14000000,conserve_volumes=False):
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkCommonCore import vtkSMPTools
    from vtkmodules.vtkFiltersCore import vtkSurfaceNets3D
    from vtkmodules.util.numpy_support import numpy_to_vtk,vtk_to_numpy
    if type(iterations) is not int or iterations<0 or not 0<constraint<=.49:raise ValueError('Finite voxel-local smoothing bounds required')
    chosen=tuple(map(int,np.unique(model.regions) if regions is None else regions))
    present=tuple(map(int,np.unique(model.regions)))
    if not chosen or len(set(chosen))!=len(chosen) or not set(chosen)<=set(present):raise ValueError('Select existing distinct labels')
    vtkSMPTools.Initialize(1)
    # Remapping avoids collisions between arbitrary region IDs and the VTK
    # background. All source phases, including original0, are retained.
    a=np.searchsorted(present,model.regions).astype(np.int32)+1
    padded=np.pad(a,1,constant_values=0)
    image=vtkImageData();image.SetDimensions(*padded.shape[::-1]);image.SetOrigin(-.5,-.5,-.5)
    image.GetPointData().SetScalars(numpy_to_vtk(padded.ravel(),deep=True))
    contour=vtkSurfaceNets3D();contour.SetInputData(image);contour.SetBackgroundLabel(0)
    contour.SetNumberOfLabels(len(present))
    for i in range(len(present)):contour.SetLabel(i,i+1)
    contour.SetOutputMeshTypeToTriangles();contour.SmoothingOff();contour.Update()
    raw=vtk_to_numpy(contour.GetOutput().GetPoints().GetData()).copy()
    contour.SetAutomaticSmoothingConstraints(False);contour.SetConstraintDistance(constraint)
    contour.GetSmoother().SetNumberOfIterations(iterations)
    contour.SetSmoothing(iterations>0);contour.Update();output=contour.GetOutput()
    points=vtk_to_numpy(output.GetPoints().GetData()).astype(float)
    if len(points)!=len(raw):raise ArithmeticError('Surface Nets point correspondence changed during constrained smoothing')
    for axis,n in enumerate(a.shape[::-1]):
        for value in (0,n):points[np.isclose(raw[:,axis],value,atol=1e-6,rtol=0),axis]=value
    points=points*np.asarray(model.spacing)[::-1]+np.asarray(model.origin)[::-1]
    faces=vtk_to_numpy(output.GetPolys().GetConnectivityArray()).reshape(-1,3).astype(np.int32)
    if len(faces)>maximum_facets:raise ValueError('Surface Nets facet budget exceeded')
    encoded=vtk_to_numpy(output.GetCellData().GetArray('BoundaryLabels'))
    lookup=np.r_[-1,present];pairs=lookup[encoded]
    include=np.isin(pairs,chosen).any(axis=1);faces=faces[include];pairs=pairs[include]
    reverse=pairs[:,0]>pairs[:,1];faces[reverse]=faces[reverse][:,::-1]
    unique,markers=np.unique(np.sort(pairs,axis=1),axis=0,return_inverse=True)
    h=np.asarray(model.spacing);origin=np.asarray(model.origin)
    seeds={k:_component_seeds(model.regions==k,h,origin) for k in chosen}
    holes=_component_seeds(~np.isin(model.regions,chosen),h,origin,bounded=True)
    plc=SurfacePLC(points,faces,seeds,markers.astype(np.int32)+1,holes,metadata={
        'surface_method':'multi-label Surface Nets','region_names':model.region_names,
        'interface_pairs':{str(i+1):list(map(int,p)) for i,p in enumerate(unique)},
        'source_shape_zyx':list(model.regions.shape),'segmentation_changed':False,
        'smoothing_iterations':iterations,'constraint_distance_voxels':constraint,'box_planes_preserved':True,
        'topology_policy':'VTK may split nonmanifold contacts geometrically; not a proof of unchanged physical contact topology',
        'upstream':'https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html'})
    if conserve_volumes:
        from .volume_projection import conserve_phase_volumes
        conserve_phase_volumes(plc,model)
    return plc
