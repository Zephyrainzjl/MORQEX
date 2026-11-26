"""Pin and integrate complete upstream text sources; keep datasets external."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REPOS={'microstructpy':('kip-hart','MicroStructPy','master'),'micro_reconstruction':('nagda95','Micro_Reconstruction','main')}
TEXT={'.py','.pyx','.pxd','.m','.r','.rd','.xml','.rst','.txt','.cfg','.toml','.yml','.yaml','.json','.csv','.c','.h','.cpp','.hpp','.cxx','.cmake','.in','.sh','.bib','.md','.rproj'}

def fetch(url,maximum):
    request=urllib.request.Request(url,headers={'User-Agent':'MORQEX-source-integration'})
    with urllib.request.urlopen(request,timeout=30) as response:
        data=response.read(maximum+1)
    if len(data)>maximum:raise ValueError('Upstream download exceeded declared budget')
    return data

def integrate(name,workspace):
    owner,repo,branch=REPOS[name];upstream=workspace/'upstream/statistical'/name;upstream.mkdir(parents=True,exist_ok=True)
    pin=upstream/'pin.json'
    if pin.exists():metadata=json.loads(pin.read_text())
    else:
        commit=subprocess.check_output(['git','ls-remote',f'https://github.com/{owner}/{repo}.git',f'refs/heads/{branch}'],text=True,timeout=30).split()[0]
        metadata={'repository':f'https://github.com/{owner}/{repo}','commit':commit};pin.write_text(json.dumps(metadata,indent=2))
    archive=upstream/(metadata['commit']+'.zip')
    if not archive.exists():archive.write_bytes(fetch(f'https://codeload.github.com/{owner}/{repo}/zip/{metadata["commit"]}',32*2**20))
    metadata['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    pin.write_text(json.dumps(metadata,indent=2));rows=[];assets=[]
    destination=ROOT/'src/morqex/vendor'/name;destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if entry.is_dir():continue
            relative=Path(*Path(entry.filename).parts[1:])
            if '..' in relative.parts:raise ValueError('Unsafe upstream archive path')
            data=z.read(entry);is_text=relative.suffix.lower() in TEXT or relative.name in {'LICENSE','COPYING','NOTICE','DESCRIPTION','NAMESPACE','MANIFEST.in','.gitignore','.Rbuildignore'}
            # Every source text is retained, including examples/tests/docs.
            # Package runtime sources are moved into our private namespace;
            # non-package source text is retained in upstream_source/.
            if not is_text or len(data)>2*2**20:
                assets.append({'path':str(relative),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});continue
            if name=='microstructpy' and relative.parts[:2]==('src','microstructpy'):target=destination/Path(*relative.parts[2:])
            elif name=='micro_reconstruction' and relative.parts[:1]==('src',):target=destination/'matlab'/Path(*relative.parts[1:])
            else:target=destination/'upstream_source'/relative
            if target.suffix.lower()=='.md':target=target.with_suffix('.txt')
            target.parent.mkdir(parents=True,exist_ok=True);text=data.decode('utf-8-sig',errors='strict');original=text
            if name=='microstructpy' and target.suffix=='.py':
                text=re.sub(r'(?m)^(\s*)from microstructpy(?=[ .])',r'\1from morqex.vendor.microstructpy',text)
                text=re.sub(r'(?m)^(\s*)import microstructpy as ',r'\1import morqex.vendor.microstructpy as ',text)
                text=re.sub(r'(?m)^(\s*)import microstructpy\s*$',r'\1from morqex.vendor import microstructpy',text)
                text=re.sub(r'(?m)^(\s*)import microstructpy(\.[\w.]+)\s*$',lambda m:m[1]+'from morqex.vendor import microstructpy\n'+m[1]+'import morqex.vendor.microstructpy'+m[2],text)
                if target==destination/'_misc.py':
                    text=text.replace('np.linalg.solve(A, b)','np.linalg.solve(A, b[..., None])[..., 0]').replace('np.linalg.solve(A, c)','np.linalg.solve(A, c[..., None])[..., 0]')
                if target==destination/'meshing/polymesh.py':
                    for variable in ['f','n','r']:text=text.replace('str(tuple('+variable+'))','str(tuple(map(int, '+variable+')))')
                if target==destination/'meshing/trimesh.py':
                    text=text.replace('import pygmsh as pg','try:\n    import pygmsh as pg\nexcept (ImportError, OSError):\n    # Raster/power-diagram paths do not require Gmsh or system libGLU.\n    pg = None')
                    text=text.replace('    with pg.geo.Geometry() as geom:','    if pg is None:\n        raise ImportError("Gmsh backend requires pygmsh and system Gmsh/libGLU; voxel raster remains available")\n    with pg.geo.Geometry() as geom:')
                if target==destination/'__init__.py':
                    version=re.search(r"__version__\s*=\s*(['\"])(.*?)\1",original)[2]
                    text='"""Pinned MicroStructPy; optional meshing backends load on demand."""\nfrom importlib import import_module\n__version__ = '+repr(version)+'\ndef __getattr__(name):\n    if name in {"cli","geometry","meshing","seeding","verification"}:\n        module=import_module("."+name,__name__);globals()[name]=module;return module\n    raise AttributeError(name)\n'
            target.write_text(text)
            rows.append({'upstream':str(relative),'integrated':str(target.relative_to(ROOT)),'upstream_sha256':hashlib.sha256(data).hexdigest(),'local_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'namespace_adapted':text!=original})
    if name=='micro_reconstruction':
        (destination/'__init__.py').write_text('"""Complete upstream MATLAB/R text sources; use morqex.generation APIs."""\n')
    metadata.update(files=rows,external_assets=assets,archive=str(archive),license_scope='MIT for MicroStructPy; Micro_Reconstruction parent has no declared top-level license; preserve nested pulpDistr license, private integration only, no blanket relicensing',scope='All upstream text source files retained. Datasets/binary assets remain in the pinned external archive. Native MATLAB/R execution needs those runtimes.')
    provenance=ROOT/'provenance'/f'{name}_migration.json';provenance.write_text(json.dumps(metadata,indent=2));print(json.dumps({'name':name,'commit':metadata['commit'],'source_files':len(rows),'external_assets':len(assets)},indent=2),flush=True)

if __name__=='__main__':
    ws=Path(os.environ['MORQEX_WORKSPACE']);(ROOT/'src/morqex/vendor').mkdir(exist_ok=True)
    (ROOT/'src/morqex/vendor/__init__.py').write_text('"""Pinned, attributed upstream sources integrated in private namespaces."""\n')
    for name in REPOS:integrate(name,ws)
