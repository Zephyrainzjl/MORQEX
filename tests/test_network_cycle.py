import os
from pathlib import Path
import numpy as np
import h5py
from morqex.top_damage.network_cycle import NetworkCycleConfig,run_network_cycle
from morqex.top_damage.network_report import audit_network_history
import pytest


def test_controls_reject_unphysical_rates_and_boundaries():
    for arg in ({'C_rate':0.},{'charge_C_rate':-1.},{'nodes':1},{'cycles':0},{'conductance_floor':0.},{'cbd_ionic_efficiency':.8},
                {'chemical_stretch_table':'missing.json'},{'chemical_stretch_table':None}):
        with pytest.raises(ValueError):NetworkCycleConfig(**arg)


@pytest.mark.parametrize('tabulated,stack',[(False,False),(True,False),(False,True)])
def test_actual_3d_fatigue_restart_preserves_cache_and_states(tmp_path,tabulated,stack):
    if not os.environ.get('MORQEX_WORKSPACE'):pytest.skip('Bounded compiled reference workspace required')
    template=(Path(__file__).parents[1]/'configs/top_damage_network_cycle.toml').read_text()
    import hashlib
    name='restart_'+hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    if stack:template=template.replace('[cycle]','[cycle]\nmechanical_constraint = "fixed_thickness_stack"')
    if tabulated:
        import json
        x=np.linspace(0,1,5);stretches=np.exp(x[:,None]*np.array([.012,.012,.04]))
        relative='experiments/'+name+'_stretch.json'
        material=Path(os.environ['MORQEX_WORKSPACE'])/relative;material.parent.mkdir(parents=True,exist_ok=True)
        material.write_text(json.dumps(dict(schema='morqex-chemical-stretch-1',host='NMC811',stoichiometry=x.tolist(),
            principal_stretches=stretches.tolist(),reference_stoichiometry=.5,provenance='Synthetic restart test, not measured material parameters')))
        checksum=hashlib.sha256(material.read_bytes()).hexdigest()
        template=template.replace('[cycle]',f'[cycle]\nchemical_stretch_table = "{relative}"\nchemical_stretch_sha256 = "{checksum}"')
    import re
    a=tmp_path/'a.toml';b=tmp_path/'b.toml'
    a.write_text(re.sub(r'output = ".*"','output = "'+name+'_whole"',template))
    b.write_text(re.sub(r'output = ".*"','output = "'+name+'_split"',template))
    ws=os.environ['MORQEX_WORKSPACE']
    full=run_network_cycle(a,ws,steps=4)
    run_network_cycle(b,ws,steps=2);split=run_network_cycle(b,ws,steps=2)
    with h5py.File(Path(full['directory'])/'step_000003.h5') as f,h5py.File(Path(split['directory'])/'step_000003.h5') as g:
        for key in ('damage','fatigue_J_m3','concentration_regions_mol_m3','next/electronic_S_m','next/mu_J_mol'):
            np.testing.assert_allclose(f[key][:],g[key][:],rtol=2e-7,atol=1e-7)
        for node in range(3):
            np.testing.assert_allclose(f[f'cache/{node}/electronic'][:],g[f'cache/{node}/electronic'][:],rtol=1e-9)
        if stack:np.testing.assert_allclose(f['macro/axial_stretches'][:],g['macro/axial_stretches'][:],rtol=0,atol=1e-10)
    for case in (full,split):
        summary,_,_=audit_network_history(case['directory'])
        assert summary['history_consistency_passed'] and summary['max_damage']>0
        assert summary['stack_equilibrium_checked']==stack
    from morqex.top_damage.network_export import export_network_history
    import xml.etree.ElementTree as ET
    import base64,struct
    exported=export_network_history(full['directory'],ws,steps=[0,3])
    assert len(exported['files'])==6
    with h5py.File(Path(full['directory'])/'step_000003.h5') as f:
        for node in range(3):
            root=ET.parse(Path(exported['directory'])/f'node_{node:03d}_step_000003.vti')
            a=root.find('.//DataArray[@Name="concentration_mol_m3"]')
            raw=base64.b64decode(a.text);length=struct.unpack('<Q',raw[:8])[0]
            assert length==len(raw)-8
            np.testing.assert_array_equal(np.frombuffer(raw[8:],dtype='<f8'),f['voxel/concentration_mol_m3'][node].ravel())
            assert len(ET.parse(Path(exported['directory'])/f'node_{node:03d}.pvd').findall('.//DataSet'))==2
            vt=ET.parse(Path(exported['directory'])/f'node_{node:03d}_step_000003.vtu')
            exact=np.fromstring(vt.find('.//PointData/DataArray[@Name="damage"]').text,sep=' ')
            np.testing.assert_array_equal(exact,f['damage'][node])
