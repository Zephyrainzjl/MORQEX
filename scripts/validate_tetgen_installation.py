"""Build/install pure→native→pure wheels in an external workspace and inspect ABI.

Use the bounded local runner. No dependencies are installed or replaced in the
research environment; each wheel is installed into its own --target directory.
"""
import os,sys,subprocess,json,zipfile,shutil,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=Path(os.environ['MORQEX_WORKSPACE'])/'tetgen_validation/packaging';out.mkdir(parents=True,exist_ok=True)
frozen=out/'source';shutil.copytree(root,frozen,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','*.pyc'),dirs_exist_ok=True)
checks=[]
for native in [False,True,False]:
 label='native' if native else ('pure_after' if len(checks)==2 else 'pure')
 folder=out/label;folder.mkdir(exist_ok=True)
 env=os.environ.copy();env['MORQEX_BUILD_TETGEN']='1' if native else '0';env.pop('MORQEX_BUILD_GEOMETRY',None)
 subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',str(folder),str(frozen)],env=env,check=True)
 wheels=list(folder.glob('*.whl'));assert len(wheels)==1
 with zipfile.ZipFile(wheels[0]) as z:
  binaries=[n for n in z.namelist() if '_tetgen_native' in n and n.endswith(('.so','.pyd'))]
  assert bool(binaries)==native
  assert any(n.endswith('modeling/native_tetgen/bridge.cpp') for n in z.namelist())
  assert any(n.endswith('tetgen/cython/tetgen/LICENSE') for n in z.namelist())
 target=folder/'installed'
 if target.exists():shutil.rmtree(target)
 subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheels[0])],env=env,check=True)
 probe='''import sys,numpy as np
sys.path.insert(0,sys.argv[1])
from morqex.modeling import from_regions,tetgen_mesh,mesh_from_toml,export_comsol
import morqex._tetgen_native as n
assert n.TETGEN_VERSION=='1.6.0'
m=tetgen_mesh(from_regions(np.ones((2,2,2),np.uint8)))
assert len(m.cells)>0 and np.all(m.signed_volumes()>0)
print('installed native API passed',len(m.cells))'''
 if native:
  env.pop('MORQEX_TETGEN_LIBRARY',None);env.pop('MORQEX_WORKSPACE',None)
  subprocess.run([sys.executable,'-c',probe,str(target)],env=env,check=True)
 checks.append({'mode':label,'wheel':str(wheels[0]),'bytes':wheels[0].stat().st_size,'sha256':hashlib.sha256(wheels[0].read_bytes()).hexdigest(),'binary':binaries,'passed':True})
# Native editable builds must fail before writing an in-source binary.
env=os.environ.copy();env['MORQEX_BUILD_TETGEN']='1'
result=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','--target',str(out/'editable_rejected'),'-e',str(frozen)],env=env,capture_output=True,text=True)
(out/'editable_build.log').write_text(result.stdout+result.stderr)
assert result.returncode!=0 and 'Native editable builds would put binaries in src' in result.stdout+result.stderr
assert not list((frozen/'src').rglob('*.so'))
checks.append({'mode':'native_editable_rejected','passed':True,'returncode':result.returncode})
(out/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
assert not list((root/'src').rglob('*.so'))
