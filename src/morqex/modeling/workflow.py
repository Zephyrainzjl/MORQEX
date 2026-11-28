"""One reproducible, checkpointed raw → prepared → mesh → exchange workflow."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,shutil,time,tomllib
from .features import voxel_fingerprint
from .progress import ProgressBar,HTMLProgress,progress_stage


@dataclass(frozen=True)
class GeometryRun:
    directory:Path
    source:Path
    prepared:Path
    mesh:object
    html:Path|None
    report:dict


def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _index(folder,source,prepared,mesh,workspace=None):
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>MORQEX geometry preparation</title>
<style>body{font:15px system-ui;background:#edf3f6;color:#173b4b;margin:20px}main{max-width:1850px;margin:auto}section{background:white;border-radius:14px;padding:20px;margin:18px 0}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}iframe{height:960px;width:100%;border:1px solid #dce7ec}nav{display:flex;gap:20px;flex-wrap:wrap}select{padding:8px;font-size:16px}@media(max-width:1100px){.views{grid-template-columns:1fr}}</style>
<main><h1>Fixed-grid smoothing → adaptive FEM / 原尺寸平滑 → 自适应有限元</h1><nav><a href="validation.json">Measured report / 实测报告</a><a href="mesh.h5">Complete mesh / 完整网格</a><a href="prepared.h5">Complete prepared geometry / 完整平滑几何</a><a href="progress.html">Progress / 进度</a></nav>
<section><p>Smoothing retains the original grid and physical box. Resolution reduction is a different explicit API and is not applied here. / 平滑不改变原尺寸或间距；本流程没有降低分辨率。</p><label>Compare / 对照 <select id="mode"><option value="smooth">Original vs smoothed voxels / 原体素与平滑体素</option><option value="mesh">Smoothed voxels vs actual FEM / 平滑体素与真实有限元</option></select></label><div class="views" id="views"></div></section>
<section><h2>Validation scope / 验证范围</h2><p>All arrays remain complete. Whole3D displays may use explicitly labelled LOD. Mesh slices intersect actual tetrahedra. Shape and volume checks are separate from PDE convergence. / 完整数据不裁剪；显示LOD有标注；网格切面来自真实四面体；形状与体积验收不等于物理解收敛。</p></section></main>
<script>function show(){let names=document.getElementById('mode').value==='smooth'?['source_voxels.html','prepared_voxels.html']:['prepared_voxels.html','mesh.html#view=2'];document.getElementById('views').replaceChildren(...names.map(src=>{let f=document.createElement('iframe');f.src=src;return f;}));document.body.dataset.rendered='true';}document.getElementById('mode').onchange=show;show();</script></html>'''
    from morqex.visualization.report_font import embed_workspace_font
    (folder/'index.html').write_text(embed_workspace_font(doc,workspace))


def run_geometry(config,directory,*,workspace=None,progress=None,content_addressed=True,preview=True):
    """Generate/load the complete source, condition, mesh and export by TOML.

    Content addressing includes the generated source, recipe and current
    algorithm files. No manual per-geometry repair or input-file deletion is
    required when the caller changes precision or smoothing settings.
    """
    from .interfaces import from_toml,prepare_from_toml,mesh_from_toml,export_from_toml
    from .preparation import PreparedGeometry
    from .meshing import MeshArtifact
    from .voxel import external_path
    config=Path(config).resolve();raw=tomllib.loads(config.read_text());start=time.perf_counter()
    if raw.get('meshing',{}).get('method')!='implicit_adaptive':raise ValueError('run_geometry requires meshing.method=implicit_adaptive; existing routes remain separate APIs')
    estimate=max(.1,float(raw.get('geometry',{}).get('size',64))**3/2e6)
    with progress_stage('complete source generation / 完整体素生成',progress or ProgressBar(),estimated_seconds=estimate):source=from_toml(config)
    signature=hashlib.sha256(config.read_bytes()+voxel_fingerprint(source).encode())
    # Material/orientation/provenance metadata may change with identical
    # labels. Do not return a checkpoint carrying stale caller metadata.
    signature.update(json.dumps(source.metadata,sort_keys=True,separators=(',',':')).encode())
    for name in ('workflow.py','preparation.py','features.py','implicit.py','cgal_backend.py'):
        signature.update(Path(__file__).with_name(name).read_bytes())
    from .cgal_backend import identity
    signature.update(identity()['source_sha256'].encode());key=signature.hexdigest()
    folder=external_path(directory)/(key if content_addressed else '')
    folder.mkdir(parents=True,exist_ok=True);manifest=folder/'configuration.json'
    record={'configuration_sha256':key,'source_fingerprint':voxel_fingerprint(source),'original_config':str(config),
        'recipe_sha256':_sha(config),'content_addressed':content_addressed}
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if any(existing.get(k)!=record[k] for k in ('configuration_sha256','source_fingerprint')):raise ValueError('Workflow checkpoint belongs to different geometry/settings; enable content_addressed or choose a new directory')
    manifest.write_text(json.dumps(record,indent=2));shutil.copyfile(config,folder/'recipe.toml')
    live=HTMLProgress(folder/'progress.html');bar=ProgressBar()
    def callback(event):
        live(event)
        if progress:progress(event)
        else:bar(event)
    source_path=folder/'source.h5';prepared_path=folder/'prepared.h5';mesh_path=folder/'mesh.h5'
    cached_prepared=prepared_path.exists();cached_mesh=False
    try:
        if not source_path.exists():source.save(source_path)
        else:
            from .voxel import VoxelModel
            if voxel_fingerprint(VoxelModel.load(source_path))!=voxel_fingerprint(source):raise ArithmeticError('Stored source checkpoint was changed')
        if cached_prepared:prepared=PreparedGeometry.load(prepared_path)
        else:prepared=prepare_from_toml(source,config,progress=callback);prepared.save(prepared_path,progress=callback)
        if prepared.report['source_fingerprint']!=voxel_fingerprint(source):raise ArithmeticError('Prepared checkpoint/source mismatch')
        validation=folder/'validation.json';mesh_state=folder/'mesh_checkpoint.json'
        if mesh_path.exists() and mesh_state.exists():
            old=json.loads(mesh_state.read_text())
            if old.get('configuration_sha256')!=key or old.get('mesh_sha256')!=_sha(mesh_path):raise ArithmeticError('Mesh checkpoint integrity mismatch')
            artifact=MeshArtifact.load(mesh_path);mesh_report=old['mesh_report'];cached_mesh=True
        else:
            if mesh_path.exists():
                rejected=folder/'rejected'/str(time.time_ns());rejected.mkdir(parents=True)
                for p in (mesh_path,mesh_path.with_suffix('.xdmf')):
                    if p.exists():shutil.move(str(p),str(rejected/p.name))
            mesh=mesh_from_toml(prepared,config,overrides={'workspace':workspace,'progress':callback})
            artifact=mesh.save_hdf5(mesh_path);mesh_report=mesh.report
            mesh_state.write_text(json.dumps({'configuration_sha256':key,'mesh_sha256':_sha(mesh_path),'mesh_report':mesh_report},indent=2))
        bundle=None
        if 'exports' in raw:
            target=folder/'software';bundle_path=target/'bundle.json'
            if bundle_path.exists():bundle=json.loads(bundle_path.read_text())
            else:
                if target.exists() and any(target.iterdir()):
                    rejected=folder/'rejected'/f'software_{time.time_ns()}';rejected.parent.mkdir(exist_ok=True);shutil.move(str(target),str(rejected))
                with progress_stage('software export/readback / 软件导出及回读',callback):bundle=export_from_toml(artifact,prepared,config,directory=target)
            if not bundle.get('passed'):raise ArithmeticError('Software exchange checkpoint not accepted')
        if preview:
            with progress_stage('voxel/FEM HTML / 体素及网格可视化',callback):
                if not (folder/'source_voxels.html').exists():source.viewer(maximum_3d_size=64,slice_origin='lower').save_html(folder/'source_voxels.html')
                if not (folder/'prepared_voxels.html').exists():prepared.model.viewer(maximum_3d_size=64,slice_origin='lower').save_html(folder/'prepared_voxels.html')
                if not (folder/'mesh.html').exists():artifact.adaptive_viewer(maximum_embedded_faces=350000,node_cache_bytes=256*2**20).save_html(folder/'mesh.html')
            _index(folder,source,prepared,artifact,workspace)
        result={'configuration_sha256':key,'mesh_sha256':_sha(mesh_path),'source':str(source_path),'prepared':str(prepared_path),
            'mesh':str(mesh_path),'preparation_cached':cached_prepared,'mesh_cached':cached_mesh,'mesh_report':mesh_report,
            'exchange':bundle,'elapsed_seconds_this_call':time.perf_counter()-start,
            'scope':'complete geometry preparation and meshing workflow; no automatic resolution reduction; no PDE/publication certification'}
        validation.write_text(json.dumps(result,indent=2));live.finish(message='Workflow complete / 全流程结束；质量与保真度见 validation.json')
        return GeometryRun(folder,source_path,prepared_path,artifact,folder/'index.html' if preview else None,result)
    except BaseException as error:
        live.finish(message='FAILED / 失败；无合格结果: '+str(error)[:1000]);raise
