"""Render an unsampled physical ROI from an existing full FEM artifact."""
import argparse,json
from pathlib import Path
import numpy as np
from morqex.modeling import VoxelModel,MeshArtifact
from morqex.modeling.voxel import external_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--roi-voxels',type=int,default=48)
    p.add_argument('--center-fraction',nargs=3,type=float,default=(.5,.5,.5))
    p.add_argument('--maximum-faces',type=int,default=500000);a=p.parse_args()
    model=VoxelModel.load(a.geometry);mesh=MeshArtifact.load(a.mesh);centre=np.asarray(a.center_fraction)
    if a.roi_voxels<2 or not np.isfinite(centre).all() or np.any(centre<=0) or np.any(centre>=1):raise ValueError('Positive display width and centre fractions strictly inside0..1 required')
    shape=np.array(model.regions.shape[::-1]);spacing=np.array(model.spacing[::-1]);origin=np.array(model.origin[::-1])
    middle=origin+shape*spacing*centre;half=spacing*a.roi_voxels/2
    bounds=np.array([np.maximum(origin,middle-half),np.minimum(origin+shape*spacing,middle+half)])
    stamps=[(f.stat().st_size,f.stat().st_mtime_ns) for f in (a.geometry,a.mesh)]
    view=mesh.adaptive_viewer(sections=False,surface_bounds_m=bounds,maximum_embedded_faces=a.maximum_faces,
        title=a.geometry.parent.name+' / complete actual ROI faces; full computation retained / 真实网格局部，完整计算不裁剪')
    if view.selected_surface_faces!=view.embedded_surface_faces:raise ValueError('ROI exceeds complete-display budget; reduce display width or explicitly increase maximum-faces')
    output=external_path(a.output);view.save_html(output)
    if stamps!=[(f.stat().st_size,f.stat().st_mtime_ns) for f in (a.geometry,a.mesh)]:raise ArithmeticError('Display modified source files')
    report={'passed':True,'source_geometry':str(a.geometry.resolve()),'source_mesh':str(a.mesh.resolve()),
        'source_tetrahedra':mesh.tetrahedra,'source_boundary_faces':view.source_surface_faces,
        'complete_actual_roi_faces':view.selected_surface_faces,'embedded_actual_faces':view.embedded_surface_faces,
        'surface_bounds_xyz_m':bounds.tolist(),'source_files_unmodified':True,
        'scope':'display ROI of original FEM triangles, not a new mesh/SVE/physical solve; no artificial caps'}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
