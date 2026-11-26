"""Pinned optional serial HYPRE dependency; all sources and binaries external."""
from pathlib import Path
import hashlib,json,shutil,signal,subprocess,sys,tarfile,urllib.request
from morqex import Workspace
from .backend import require_local_budget

VERSION='2.32.0'
ARCHIVE_SHA256='2277b6f01de4a7d0b01cfe12615255d9640eaa02268565a7ce1a769beab25fa1'
URL=f'https://codeload.github.com/hypre-space/hypre/tar.gz/refs/tags/v{VERSION}'

def build_hypre(workspace,*,slice_seconds=55):
    require_local_budget()
    if type(slice_seconds) is not int or not 5<=slice_seconds<=55:raise ValueError('Build slice must be 5..55 seconds')
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    source=ws.path('upstream',f'hypre-{VERSION}');archive=ws.path('upstream',f'hypre-{VERSION}.tar.gz')
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        temporary=archive.with_suffix('.pending')
        with urllib.request.urlopen(URL,timeout=45) as response,temporary.open('wb') as out:
            total=0
            while block:=response.read(1024*1024):
                total+=len(block)
                if total>20*1024*1024:raise ValueError('HYPRE archive exceeds pinned resource budget')
                out.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=ARCHIVE_SHA256:raise ValueError('HYPRE download checksum mismatch')
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA256:raise ValueError('HYPRE archive modified')
    if not source.exists():
        with tarfile.open(archive) as tar:tar.extractall(source.parent,filter='data')
    # Verify actual source against the approved archive, not just the download.
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile():
                path=source.parent/member.name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(tar.extractfile(member).read()).digest():
                    raise ValueError('HYPRE source changed: '+member.name)
    build=ws.path('build',f'hypre-{VERSION}-serial');build.mkdir(parents=True,exist_ok=True)
    prefix=ws.path('build',f'hypre-{VERSION}-install');ninja=str(Path(sys.executable).parent/'ninja')
    command=['cmake','-S',str(source/'src'),'-B',str(build),'-G','Ninja','-DCMAKE_MAKE_PROGRAM='+ninja,
      '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_POSITION_INDEPENDENT_CODE=ON','-DHYPRE_WITH_MPI=OFF',
      '-DHYPRE_WITH_OPENMP=OFF','-DHYPRE_BUILD_TESTS=OFF','-DCMAKE_INSTALL_PREFIX='+str(prefix)]
    with (build/'morqex_configure.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
    with (build/'morqex_build.log').open('a') as log:
        proc=subprocess.Popen([ninja,'-C',str(build),'-j1','install'],stdout=log,stderr=subprocess.STDOUT)
        try:code=proc.wait(timeout=slice_seconds)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT);proc.wait(timeout=15);state='pending'
        else:state='complete' if code==0 else 'failed'
    result={'state':state,'dependency':'HYPRE','version':VERSION,'archive_sha256':ARCHIVE_SHA256,'url':URL,'prefix':str(prefix),'license':'MIT option; source COPYRIGHT retained','jobs':1,'mpi':False,'openmp':False}
    if state=='complete':
        if '#define HYPRE_SEQUENTIAL 1' not in (prefix/'include/HYPRE_config.h').read_text():raise ValueError('HYPRE was not built serially')
        notices=prefix/'share/morqex-notices';notices.mkdir(parents=True,exist_ok=True)
        for name in ('LICENSE-MIT','COPYRIGHT'):shutil.copyfile(source/name,notices/name)
        result['library_sha256']=hashlib.sha256((prefix/'lib/libHYPRE.a').read_bytes()).hexdigest()
    (build/'morqex_dependency.json').write_text(json.dumps(result,indent=2))
    if state=='failed':raise RuntimeError('HYPRE build failed: '+str(build/'morqex_build.log'))
    return result
