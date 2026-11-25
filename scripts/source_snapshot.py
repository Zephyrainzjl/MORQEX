"""Content-addressed, local-only source snapshots for numerical provenance."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

def tree(root,*,capture=False):
    root=Path(root)
    result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root,capture_output=True)
    paths=sorted(set(result.stdout.decode().split('\0'))- {''}) if result.returncode==0 else sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '.git' not in p.parts)
    records=[];contents={};digest=hashlib.sha256()
    for name in paths:
        path=root/name
        if not path.exists():continue
        if path.is_symlink():raise ValueError('Source snapshot requires ordinary in-tree files')
        if path.stat().st_size>2*2**20:raise ValueError(f'Large file in source tree: {name}')
        data=path.read_bytes();sha=hashlib.sha256(data).hexdigest();records.append({'path':name,'sha256':sha,'bytes':len(data)})
        digest.update(name.encode()+b'\0'+sha.encode()+b'\0')
        if capture:contents[name]=data
    return digest.hexdigest(),records,contents

def snapshot(root,workspace):
    signature,records,contents=tree(root,capture=True);directory=Path(workspace)/'source_snapshots';directory.mkdir(parents=True,exist_ok=True);path=directory/(signature+'.tar.gz')
    if not path.exists():
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1) as compressed,tarfile.open(fileobj=compressed,mode='w') as archive:
            for name,data in contents.items():
                info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;archive.addfile(info,io.BytesIO(data))
        temporary.replace(path)
        (directory/(signature+'.json')).write_text(json.dumps({'tree_sha256':signature,'files':records},indent=2))
    return {'tree_sha256':signature,'archive':str(path),'file_count':len(records),
        'scope':'Source captured before execution; a finish-time tree comparison reports edits during the run. Runtime is not moved to the snapshot.'}
