import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from morqex.top_damage.reference_run import run_reference, read_records

ROOT=Path(__file__).resolve().parents[1]


def test_hdf5_restart_matches_uninterrupted_and_vtu(tmp_path):
    if not os.environ.get('MORQEX_WORKSPACE'):pytest.skip('External limited workspace required')
    provider=os.environ.get('MORQEX_TEST_MFEM_PROVIDER','vendored')
    text=(ROOT/'configs/top_damage_notched_reference.toml').read_text().replace('provider = "vendored"',f'provider = "{provider}"').replace('cycles = 3','cycles = 1').replace('shape_zyx = [2, 8, 8]','shape_zyx = [2, 4, 4]')
    config=tmp_path/'case.toml';config.write_text(text)
    # The actual compiled library belongs to the main external workspace;
    # isolate outputs by unique relative run directory.
    unique='pytest_'+hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    ws=os.environ['MORQEX_WORKSPACE']
    full=run_reference(config,ws,steps=9,output=unique+'_full')
    part=run_reference(config,ws,steps=4,output=unique+'_resumed')
    assert not part['complete_specimen_protocol']
    done=run_reference(config,ws,steps=5,output=unique+'_resumed',resume=True)
    assert done['complete_specimen_protocol']
    a,b=read_records(full['directory']),read_records(done['directory'])
    for ra,rb in zip(a,b):
        for key in ('mean_damage','mean_stress_yy_MPa','elastic_energy_J','damage_kkt'):
            assert ra[key]==pytest.approx(rb[key],rel=1e-13,abs=1e-15)
        for key in ra['fields']:np.testing.assert_array_equal(ra['fields'][key],rb['fields'][key])
    mesh=ET.parse(Path(done['directory'])/'step_00008.vtu').getroot().find('./UnstructuredGrid/Piece')
    assert int(mesh.attrib['NumberOfCells'])==2*4*4*6
    assert {v.attrib['Name'] for v in mesh.findall('./PointData/DataArray')}=={'damage','displacement_m'}
    config.write_text(text.replace('maximum_yy_strain = 0.004','maximum_yy_strain = 0.005'))
    with pytest.raises(ValueError,match='identity mismatch'):run_reference(config,ws,output=unique+'_resumed',resume=True)


def test_budget_tracks_child_with_separate_process_group(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('morqex_local_runner',ROOT/'scripts/run_local.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    marker=tmp_path/'pid.txt'
    child="import os,time; x=bytearray(12*1024*1024); open("+repr(str(marker))+",'w').write(str(os.getpid())); time.sleep(20)"
    parent="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child)+"],start_new_session=True); time.sleep(20)"
    proc=subprocess.Popen([sys.executable,'-c',parent],start_new_session=True)
    tracked={}
    try:
        for _ in range(100):
            if marker.exists():break
            time.sleep(.01)
        assert marker.exists()
        pid=int(marker.read_text())
        assert os.getpgid(pid)!=proc.pid
        assert pid in module.job_processes(proc.pid,tracked)
        assert module.group_rss(proc.pid,tracked)>12*2**20
    finally:
        module.kill_job(proc.pid,tracked);proc.wait(timeout=5)
