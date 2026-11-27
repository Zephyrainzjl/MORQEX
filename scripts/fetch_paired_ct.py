"""Checksum-pinned Parks2023 pristine/charged NMC811 CT; streaming, <=4MiB/s.

Raw greyscale tomography, NOT segmented phases or crystallographic grains.
No scientific calculation is performed by this IO-only acquisition utility.
"""
import hashlib,json,os,resource,shutil,time,urllib.request
from pathlib import Path


def main():
    root=Path(os.environ['MORQEX_WORKSPACE']).resolve()
    if root==Path(__file__).resolve().parents[1] or root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Raw data must remain outside source')
    target=root/'data/parks2023';target.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(target).free<2*1024**3:raise RuntimeError('Reserve at least2GiB for acquisition')
    os.nice(15);os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(90,90))
    resource.setrlimit(resource.RLIMIT_FSIZE,(512*1024**2,512*1024**2))
    sources=[('Pristine_Merged_3DTiff.tif',39307304,416632590,'8b1b8fabe45486512287020ba46354d6'),
             ('Charged_Merged_3DTiff.tif',39307298,413725604,'e97798174c9bda35a0176b8696e46233')]
    records=[]
    for name,identifier,size,md5 in sources:
        path=target/name;url=f'https://ndownloader.figshare.com/files/{identifier}'
        if not path.exists():
            temporary=path.with_suffix('.part');start=time.monotonic();total=0
            try:
                with urllib.request.urlopen(url,timeout=40) as response,temporary.open('wb') as output:
                    while chunk:=response.read(1024**2):
                        total+=len(chunk)
                        if total>size:raise RuntimeError('Remote file exceeds pinned size')
                        output.write(chunk)
                        delay=total/(4*1024**2)-(time.monotonic()-start)
                        if delay>0:time.sleep(min(delay,1.))
                if total!=size:raise RuntimeError('Incomplete raw CT download')
                # Validate before the file becomes accepted data.
                digest=hashlib.md5(usedforsecurity=False);sha=hashlib.sha256()
                with temporary.open('rb') as stream:
                    while chunk:=stream.read(1024**2):digest.update(chunk);sha.update(chunk)
                if digest.hexdigest()!=md5:raise RuntimeError('Publisher MD5 mismatch')
                temporary.replace(path)
            finally:temporary.unlink(missing_ok=True)
        digest=hashlib.md5(usedforsecurity=False);sha=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(1024**2):digest.update(chunk);sha.update(chunk)
        if path.stat().st_size!=size or digest.hexdigest()!=md5:raise RuntimeError('Accepted raw CT checksum mismatch')
        records.append(dict(name=name,bytes=size,url=url,md5=md5,sha256=sha.hexdigest()))
        print(json.dumps(records[-1]),flush=True)
    result={'dataset_doi':'10.5522/04/22120061','article_doi':'10.1039/D3TA03057A',
        'license':'CC BY4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
        'attribution':'Parks et al., NMC811 4D Nanotomography Tiff Stacks, UCL Research Data Repository.',
        'scope':'Raw pristine and charged greyscale data; registration, segmentation, spacing and grain inference not yet qualified.',
        'files':records}
    (target/'manifest.json').write_text(json.dumps(result,indent=2));print('Acquisition checks passed',flush=True)

if __name__=='__main__':main()
