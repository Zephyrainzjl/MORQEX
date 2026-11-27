"""Qualify pure/TetGen/ACVD/combined pip wheels in isolated external targets."""
import os,sys,subprocess,json,zipfile,shutil,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=Path(os.environ['MORQEX_WORKSPACE'])/'adaptive_installation';out.mkdir(parents=True,exist_ok=True)
(out/'temporary').mkdir(exist_ok=True)
os.environ['TMPDIR']=str(out/'temporary');os.environ['PIP_CACHE_DIR']=str(out/'pip_cache');os.environ['PIP_DISABLE_PIP_VERSION_CHECK']='1'
frozen=out/'source';shutil.copytree(root,frozen,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','*.pyc'),dirs_exist_ok=True)
checks=[]
sdist=out/'sdist';sdist.mkdir(exist_ok=True)
sdist_env=os.environ.copy();sdist_env['MORQEX_BUILD_TETGEN']='1';sdist_env['MORQEX_BUILD_ACVD']='1';sdist_env.pop('MORQEX_BUILD_GEOMETRY',None)
subprocess.run([sys.executable,'-c','import build_backend,sys;print(build_backend.build_sdist(sys.argv[1]))',str(sdist)],
    cwd=frozen,env=sdist_env,check=True)
source_archives=list(sdist.glob('*.tar.gz'));assert len(source_archives)==1
modes=[('pure',False,False),('tetgen',True,False),('acvd',False,True),('combined',True,True),('pure_after',False,False),('sdist_combined',True,True)]
for label,tetgen,acvd in modes:
 folder=out/label;folder.mkdir(exist_ok=True)
 env=os.environ.copy();env['MORQEX_BUILD_TETGEN']=str(int(tetgen));env['MORQEX_BUILD_ACVD']=str(int(acvd));env.pop('MORQEX_BUILD_GEOMETRY',None)
 source=source_archives[0] if label=='sdist_combined' else frozen
 subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',str(folder),str(source)],env=env,check=True)
 wheels=list(folder.glob('*.whl'));assert len(wheels)==1
 with zipfile.ZipFile(wheels[0]) as z:
  files=z.namelist();t=[n for n in files if '_tetgen_native' in n and n.endswith(('.so','.pyd'))]
  c=[n for n in files if '_acvd_native' in n and n.endswith(('.so','.pyd'))]
  assert bool(t)==tetgen and bool(c)==acvd
  assert any(n.endswith('tetgen/cython/tetgen/LICENSE') for n in files)
  assert any(n.endswith('pyacvd/LICENSE') for n in files)
  assert any('licenses/THIRD_PARTY_NOTICES' in n for n in files)
 target=folder/'installed'
 if target.exists():shutil.rmtree(target)
 subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheels[0])],env=env,check=True)
 probe='''import sys,importlib.util,numpy as np
sys.path.insert(0,sys.argv[1])
tetgen,acvd=bool(int(sys.argv[2])),bool(int(sys.argv[3]))
assert bool(importlib.util.find_spec('morqex._tetgen_native'))==tetgen
assert bool(importlib.util.find_spec('morqex._acvd_native'))==acvd
from morqex.modeling import from_regions,tetgen_mesh,adaptive_surface,adaptive_mesh,AdaptiveMeshOptions
if tetgen:
 import morqex._tetgen_native as t
 assert t.TETGEN_VERSION=='1.6.0'
 mesh=tetgen_mesh(from_regions(np.ones((2,2,2),np.uint8)))
 assert mesh.signed_volumes().min()>0
if acvd:
 from morqex.modeling.acvd_backend import clustering_class
 assert clustering_class().__module__.endswith('pyacvd.clustering')
 z,y,x=np.indices((24,)*3);labels=((z-11.5)**2+(y-11.5)**2+(x-11.5)**2<8**2).astype(np.uint8)
 model=from_regions(labels);options=AdaptiveMeshOptions(vertex_fraction=.4,maximum_phase_volume_error=.1)
 surface=adaptive_surface(model,regions=[1],options=options)
 assert surface.metadata['output_surface_vertices']<surface.metadata['input_surface_vertices']
 if tetgen:
  mesh=adaptive_mesh(model,regions=[1],options=options)
  assert mesh.signed_volumes().min()>0
print('installed public API passed',tetgen,acvd)'''
 clean=env.copy();clean.pop('MORQEX_WORKSPACE',None);clean.pop('MORQEX_TETGEN_LIBRARY',None)
 subprocess.run([sys.executable,'-c',probe,str(target),str(int(tetgen)),str(int(acvd))],env=clean,check=True)
 checks.append({'mode':label,'wheel':str(wheels[0]),'bytes':wheels[0].stat().st_size,
  'sha256':hashlib.sha256(wheels[0].read_bytes()).hexdigest(),'tetgen_binary':t,'acvd_binary':c,'passed':True})
env=os.environ.copy();env['MORQEX_BUILD_TETGEN']='1';env['MORQEX_BUILD_ACVD']='1'
result=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','--target',str(out/'editable_rejected'),'-e',str(frozen)],env=env,capture_output=True,text=True)
(out/'editable_build.log').write_text(result.stdout+result.stderr)
assert result.returncode!=0 and 'Native editable builds would put binaries in src' in result.stdout+result.stderr
assert not list((frozen/'src').rglob('*.so'))
checks.append({'mode':'native_editable_rejected','passed':True})
(out/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
assert not list((root/'src').rglob('*.so'))
