"""Explicit NMC/LFP/LCO host packs for the qualified common-host closure.

Published electrochemical inputs remain distinguishable from thermodynamic
projection and uncalibrated supplements. LFP here is a homogeneous effective
free energy, not phase separation, mosaic transformation or hysteresis.
"""
import hashlib
from pathlib import Path
import pybamm as pb
from .microstructure_parameters import microstructure_parameters

HOSTS={'NMC':'OKane2022','LFP':'Prada2013','LCO':'Ai2020'}

def cathode_parameters(family):
    if family not in HOSTS:raise ValueError('Choose NMC, LFP or LCO, or supply a separately qualified parameter provider')
    name=HOSTS[family]
    p,audit=microstructure_parameters(source_parameter_set=name,allow_idealization=family!='NMC')
    source=Path(pb.__file__).parent/'input/parameters/lithium_ion'/f'{name}.py'
    audit.update(cathode_family=family,parameter_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Explicit modified host pack; not calibrated CT, dopant, phase-field or experimental lifetime validation.')
    if family=='LFP':audit['material_limit']='Effective single-phase OCP branch only; two-phase transformation/hysteresis and LFP mechanical/thermal supplements uncalibrated'
    return p,audit
