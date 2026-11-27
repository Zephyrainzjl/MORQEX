"""Geometry-derived previews and offline Canvas FEM atlas; no remote assets."""
import base64,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def thumbnail(path,points,attributes,size,*,title):
    yaw=-.65;pitch=.55
    rotation=np.array([[np.cos(yaw),-np.sin(yaw),0],
        [np.sin(pitch)*np.sin(yaw),np.sin(pitch)*np.cos(yaw),-np.cos(pitch)],
        [np.cos(pitch)*np.sin(yaw),np.cos(pitch)*np.cos(yaw),np.sin(pitch)]])
    selected=attributes[:,0]==1;p=(points[selected]-size/2)@rotation.T
    a=points[selected];normal=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
    visible=(normal@rotation[2])>0;p=p[visible];normal=normal[visible]
    order=np.argsort(p[:,:,2].mean(axis=1));shade=.40+.60*np.maximum(0,normal@np.array([.3,-.4,.866]))
    image=Image.new('RGB',(640,520),'#edf1f3');draw=ImageDraw.Draw(image)
    scale=4.7*64/size
    for i in order:
        polygon=[(float(320+x*scale),float(255+y*scale)) for x,y in p[i,:,:2]]
        draw.polygon(polygon,fill=tuple(int(x*shade[i]) for x in (200,128,58)))
    draw.text((24,474),title,fill='#263d4d',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18))
    image.save(path)


def write_large_gallery(root):
    """Lazy local-script loading of256³ geometry: no server or CORS required."""
    root=Path(root);records=[json.loads(p.read_text()) for p in sorted((root/'structures').glob('*/metadata.json'))]
    if len(records)!=10 or any(r['shape_zyx']!=[256]*3 for r in records):raise ValueError('Ten complete256³ models required')
    catalog=[]
    for i,r in enumerate(records):
        catalog.append({'record':r,'preview':'data:image/png;base64,'+base64.b64encode((root/f'preview_{i+1:02d}.png').read_bytes()).decode(),'displayExternal':True})
    html=Path(__file__).with_name('electrode_gallery.html').read_text().replace('__DATA__',json.dumps(catalog,ensure_ascii=False,separators=(',',':')))
    links='<section class="panel bottom"><h2>验证与Notebook / Validation and notebooks</h2><div class="downloads"><a href="validation.json">全部体单元读回检查</a><a href="probe_mapping_validation.json">显示单元与完整网格对应检查</a><a href="browser_validation.json">浏览器检查</a><a href="../notebook_example/live_geometry.html">Notebook几何预览</a><a href="../notebook_example/actual_mesh.html">Notebook真实四面体预览</a><a href="../notebook_example/geometry_workflow.executed.ipynb">已执行Notebook</a><a href="../../../MORQEX/tutorials/README.md">中英文教程</a></div></section>'
    html=html.replace('</main>',links+'</main>',1)
    font=root.parent/'data/report_font/NotoSansCJKsc-Regular.otf'
    if font.exists():
        from .report_font import embed_font
        html=embed_font(html,font,(font.parent/'LICENSE').read_text())
    (root/'index.html').write_text(html)
    montage=Image.new('RGB',(3200,1040),'#edf1f3')
    for i in range(10):
        with Image.open(root/f'preview_{i+1:02d}.png') as img:montage.paste(img,((i%5)*640,(i//5)*520))
    montage.save(root/'ten_structures.png')
    (root/'manifest.json').write_text(json.dumps({'structures':records,'total_voxels':10*256**3,
        'total_tetrahedra':sum(r['tetrahedra'] for r in records),'display':'64³ labelled LOD;256³ raw slices and exact local FEM; full mesh in HDF5/XDMF'},ensure_ascii=False,indent=2))
