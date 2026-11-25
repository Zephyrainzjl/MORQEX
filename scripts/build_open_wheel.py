"""Build from an external source snapshot; never create build files in ROOT."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile
from source_snapshot import snapshot

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];workspace=Path(os.environ['MORQEX_WORKSPACE']);record=snapshot(root,workspace)
    unpacked=workspace/'build/packaging'/record['tree_sha256'];unpacked.mkdir(parents=True,exist_ok=True)
    with tarfile.open(record['archive']) as archive:archive.extractall(unpacked,filter='data')
    destination=workspace/'dist';destination.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','pip','wheel',str(unpacked),'--no-deps','--no-build-isolation','--wheel-dir',str(destination)],check=True)
    wheels=list(destination.glob('morqex_research-0.1.0.dev0-*.whl'))
    if len(wheels)!=1:raise RuntimeError('Expected one development wheel')
    checked=0
    with zipfile.ZipFile(wheels[0]) as archive:
        names=set(archive.namelist())
        for path in (unpacked/'src').rglob('*'):
            if not path.is_file():continue
            member=str(path.relative_to(unpacked/'src'))
            if member not in names and path.suffix not in {'.py','.cu','.html','.js'}:continue
            if member not in names or archive.read(member)!=path.read_bytes():raise RuntimeError(f'Wheel/source mismatch: {member}')
            checked+=1
    result={'source_snapshot':record,'wheel':str(wheels[0]),'wheel_sha256':hashlib.sha256(wheels[0].read_bytes()).hexdigest(),'build_directory':str(unpacked),'packaged_files_compared_to_snapshot':checked}
    (workspace/'reports/open_wheel_build.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
