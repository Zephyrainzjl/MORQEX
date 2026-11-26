"""Self-contained node-to-voxel explorer with on-demand field localization."""
import base64
import gzip
import json
import os
from pathlib import Path
import numpy as np
from ..nodewise.cache import NodeLocalization
from ..nodewise.localization import voxel_maps


def _packed(array,quantize=False):
    a=np.asarray(array)
    if quantize:
        scale=np.max(abs(a.reshape(-1,a.shape[-1])),axis=0)/32767 if a.shape[-1] else np.empty(0)
        scale=np.where(scale>0,scale,1.)
        encoded=np.rint(a/scale).astype('<i2');dtype='i2'
    else:encoded=a.astype('<f4');scale=None;dtype='f4'
    return {'shape':list(a.shape),'dtype':dtype,'data':base64.b64encode(encoded.tobytes()).decode(),
        'scale':scale.tolist() if scale is not None else None}


def build_node_report(result_path,destination,*,title=None):
    """Embed low-dimensional histories and all-node voxel maps; no field LOD.

    The browser's map columns use signed 16-bit presentation quantization.
    Scientific FP64 maps remain checksum-verified in external operator caches.
    This is explicitly separated from solver/discretization errors.
    """
    path=Path(result_path);result=json.loads(path.read_text());geometry={};mapping={}
    for d,paths in result['paths'].items():
        mapping[d]=[]
        for cache in paths:
            local=NodeLocalization(cache);identity=local.kernel.operator.metadata()['fingerprint'];mapping[d].append(identity)
            if identity in geometry:continue
            labels=local.array('labels');spacing=local.array('spacing');P=local.array('P');active_ids=local.array('voxel_ids')
            phases=local.manifest['phase_kinds'];maps=voxel_maps(local);c=local.kernel.cohesive
            geometry[identity]={'shape':list(labels.shape),'spacing':spacing.tolist(),'labels':labels.ravel().tolist(),
                'phase_kinds':phases,'reference_c':local.kernel.operator.reference_concentration,
                'reference_T':local.kernel.operator.metadata()['reference_temperature_K'],
                'weights':local.kernel.operator.weights.tolist(),'areas':local.kernel.operator.reaction_area.tolist(),
                'P':_packed(P),'active_ids':active_ids.tolist(),'factors':c.factors.tolist(),'rho':c.eigenvalues.tolist(),'residual':c.residual,
                'maps':{name:_packed(a,quantize=True) for name,a in maps.items() if name!='concentration'},
                'source':local.manifest['source_geometry'],'qualification':local.manifest['cohesive']}
    data={'title':title or 'MORQEX · 节点—微结构演化','result':result,'geometry':geometry,'mapping':mapping,
        'live_url':os.path.relpath(path.parent/'live.json',Path(destination).parent),
        'scope':{'stress':'体素内 Tet4 应力张量的体积平均；主应力和 von Mises 在平均张量上计算',
            'potential':'本闭合模型的 DFN 节点电势；节点微结构内分相常量，未冒充已解析的微观电势涨落',
            'temperature':'双向耦合均温模型；微结构各体素共享节点温度',
            'damage':'每个微结构一个不可逆界面损伤状态；改变界面弹簧、可达反应面积和电子输运',
            'concentration':'真实几何上的守恒分区浓度状态，替代 DFN 球形径向状态；未声称已通过模式收敛',
            'presentation':'浏览器使用按列缩放的 16 位算子显示副本；FP64 科学算子和计算数据均在外部工作区'}}
    if result.get('configuration',{}).get('degradation','').startswith('disabled'):
        data['scope']['damage']='本轮为固定完整界面的线弹性参考验证，未启用演化脱粘；损伤为零不能解释为寿命预测。'
    text=json.dumps(data,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    encoded=base64.b64encode(gzip.compress(text.encode(),compresslevel=6)).decode()
    root=Path(__file__).parent;html=(root/'nodewise.html').read_text().replace('__PAYLOAD__',encoded).replace('__SCRIPT__',(root/'nodewise.js').read_text())
    if result.get('counterelectrode')=='planar lithium metal':
        start=html.index('<div class="pills">');end=html.index('</div>',start)+6
        html=html[:start]+'<div class="pills"><a href="MORQEX_sphere32_Li_geometry.html">32³ 颗粒与 CBD 几何</a><span>共享几何 · 6 个独立正极状态 · Li 平面边界</span></div>'+html[end:]
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    font=Path(os.environ.get('MORQEX_WORKSPACE',str(destination.parent.parent)))/'data/report_font'
    if (font/'NotoSansCJKsc-Regular.otf').exists():
        from .report_font import embed_font
        glyphs=''.join(sorted({c for c in text if '\u3400'<=c<='\u9fff'}))
        html=html.replace('</main>','<span hidden>'+glyphs+'</span></main>')
        html=embed_font(html,font/'NotoSansCJKsc-Regular.otf',(font/'LICENSE').read_text())
    destination.write_text(html)
    return {'path':str(destination),'bytes':destination.stat().st_size,'unique_geometries':len(geometry),'nodes':{d:len(v) for d,v in mapping.items()}}
