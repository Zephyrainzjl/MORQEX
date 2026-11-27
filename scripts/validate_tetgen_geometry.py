"""Bounded native geometry qualification and offline TetGen/COMSOL inspection.

Run with scripts/run_local.py; outputs are external. No electrochemical or
mechanical-solver validation is inferred from these geometry experiments.
"""
import os,json,html,time
from pathlib import Path
import numpy as np
from morqex.modeling import (from_shapes,Shape,electrode,PRESETS,tetgen_mesh,
    TetGenOptions,export_comsol,read_comsol,from_toml,mesh_from_toml)

from morqex.modeling.voxel import external_path
work=external_path(os.environ.get('MORQEX_TETGEN_OUTPUT',str(Path(os.environ['MORQEX_WORKSPACE'])/'tetgen_validation')))
work.mkdir(parents=True,exist_ok=True);reports=[];cards=[]

def save(name,model,result):
    folder=work/name;folder.mkdir(exist_ok=True)
    artifact=result.save_hdf5(folder/'mesh.h5')
    path=export_comsol(result,folder/'mesh.mphtxt',region_names=model.region_names)
    back=read_comsol(path)
    assert np.array_equal(back['points'],result.points)
    assert np.array_equal(back['elements']['tet']['cells'],result.cells)
    mapping=json.loads(path.with_suffix('.regions.json').read_text())['region_to_domain']
    assert np.array_equal(back['elements']['tet']['entities'],[mapping[str(int(k))] for k in result.phases])
    result.viewer(cell_count=min(2500,len(result.cells)),sampling='uniform',title=name+' / Actual TetGen elements').save_html(folder/'mesh.html')
    model.viewer(maximum_3d_size=48,title=name+' / Source voxels').save_html(folder/'voxels.html')
    report={'name':name,'source_shape':model.regions.shape,'source_voxels':model.regions.size,
        'selected_voxels':int(np.isin(model.regions,result.phases).sum()),
        'native_report':result.report,'comsol_readback':True,'hdf5':str(artifact.hdf5)}
    (folder/'validation.json').write_text(json.dumps(report,indent=2))
    reports.append(report)
    cards.append(f'''<article><h2>{html.escape(name)}</h2><p>Source {model.regions.shape}: {model.regions.size:,} voxels;
    TetGen {len(result.cells):,} Tet4, {len(result.points):,} nodes.
    COMSOL read-back passed. Measured q/angle request satisfied: {result.report['quality_limits_satisfied']}.
    Geometry: {html.escape(result.report['geometry']['surface_method'])}.</p>
    <p><a href="{name}/mesh.xdmf">XDMF</a> · <a href="{name}/mesh.h5">HDF5</a> ·
    <a href="{name}/mesh.mphtxt">COMSOL mesh</a> · <a href="{name}/mesh.regions.json">Region map</a> ·
    <a href="{name}/validation.json">Measured quality</a></p>
    <button onclick="load(this,'{name}/voxels.html')">Source voxel view / 体素</button>
    <button onclick="load(this,'{name}/mesh.html')">Actual TetGen / 真实单元</button>
    <iframe title="{name}" loading="lazy" src="{name}/mesh.html"></iframe></article>''')

# All10 complete small irregular geometries: exact shared multiphase geometry,
# including pore region0. Do not turn failed solid-only manifolds into fake meshes.
for preset,_ in PRESETS:
    name='preset_'+preset
    model,_=electrode(preset,size=32,spacing=1e-6,seed=20261127,cbd_fraction=.10)
    mesh=tetgen_mesh(model,regions=None,options=TetGenOptions(maximum_cells=500000))
    save(name,model,mesh);print(json.dumps({'finished':name,'tetrahedra':len(mesh.cells)}),flush=True)

# A full-resolution256³ field of three irregular elongated particles.
# step_size1: all source voxels participate in contour extraction; no ROI/stride.
model=from_shapes([
    Shape('capsule',1,(70e-6,70e-6,125e-6),(16e-6,16e-6,75e-6),irregularity=.16),
    Shape('ellipsoid',1,(170e-6,85e-6,125e-6),(40e-6,22e-6,85e-6),irregularity=.13),
    Shape('capsule',1,(145e-6,190e-6,125e-6),(22e-6,22e-6,65e-6),irregularity=.19),
],shape=(256,)*3,spacing=(1e-6,)*3)
model.save(work/'source256.h5')
mesh=tetgen_mesh(model,regions=[1],surface_method='marching_cubes',step_size=1,
                 options=TetGenOptions(maximum_cells=1500000,maximum_steiner=300000))
save('full256_irregular',model,mesh)

config=Path(__file__).resolve().parents[1]/'configs/geometry_tetgen.toml'
model=from_toml(config);mesh=mesh_from_toml(model,config);save('toml_multiphase',model,mesh)

payload={'passed_geometry_and_export_checks':True,'cases':reports,
    'scope':'10 complete32^3 multiphase models; one complete256^3 single-phase contour at step1; one24^3 TOML recipe. Not a battery simulation.',
    'comsol_application_import':'Not run: no local COMSOL installation',
    'quality_policy':'All positive/region-volume/index/readback checks passed. Requested q limits are reported separately, not silently certified.'}
(work/'validation.json').write_text(json.dumps(payload,indent=2))
(work/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>MORQEX TetGen + COMSOL</title>
<style>body{font:16px system-ui;background:#eef3f6;color:#163047;max-width:1200px;margin:auto;padding:24px}article{background:white;padding:20px;margin:20px 0;border-radius:12px}iframe{width:100%;height:680px;border:0}button{margin:8px;padding:10px}a{color:#156380}</style>
<h1>MORQEX · TetGen / COMSOL</h1><p>真实原生 TetGen 网格与 COMSOL 原生文本导出。10个完整32³分区模型、1个完整256³全分辨率轮廓、1个TOML示例。
256³并非十个新 TetGen 模型；原256³十模型体素剖分图库保持独立。此页不代表力学/电池求解验证。</p>
<p>Native meshes passed positive-volume, exact-PLC material-volume and COMSOL numerical read-back checks.
q requests at small PLC angles are reported separately. COMSOL GUI import is not tested on this machine.</p>
<p><a href="validation.json">Full numerical report</a> · <a href="__ATLAS_LINK__">Original ten256³ voxel-split meshes</a></p>
'''.replace('__ATLAS_LINK__',html.escape(os.path.relpath(Path(os.environ['MORQEX_WORKSPACE'])/'atlas256/index.html',work)))+''.join(cards)+'''<script>function load(b,url){b.closest('article').querySelector('iframe').src=url;document.body.dataset.rendered='true'}</script>''')
print(json.dumps({'passed':True,'cases':len(reports),'html':str(work/'index.html')}))
