"""Six-node-compatible porous cathode / planar Li-metal DFN composition.

The counterelectrode has no particle/radial/voxel concentration states. Native
Li-metal BV and optional planar SEI are retained. A finite Li inventory ledger
closes mass accounting; dendrites, moving foil geometry and Li plasticity are
outside this fixed-reference-geometry model.
"""
import numpy as np
import pybamm as pb
from .constitutive import NodewiseOperator,NodewiseCohesive
from ..electrochemistry.aging_dfn import EnergyKernel


class HalfCellStack:
    working_electrode='positive'
    def __init__(self,kernels,parameters,*,constraint='pressure',separator_modulus_Pa=1e8,metal_sei='none'):
        if constraint not in {'pressure','fixed-thickness'}:raise ValueError('Supported zero-pressure or fixed-thickness mechanical constraint required')
        if parameters['Thermodynamic factor']!=1 or not isinstance(parameters['Cation transference number'],(float,int)):raise ValueError('Ideal electrolyte with constant t+ required by this energy model')
        self.electrodes={'positive':NodewiseOperator('positive',kernels)};self.constraint=constraint;self.linear_ocp_entropy=True
        self.lengths={'positive':float(parameters['Positive electrode thickness [m]'])};self.separator_length=float(parameters['Separator thickness [m]']);self.separator_C=separator_modulus_Pa
        op=self.electrodes['positive'];self.kernels={'positive':EnergyKernel(op,NodewiseCohesive(op))}
        self.metal_sei=metal_sei
        self.energy_TT_per_area=0.;self.metal_inventory_initial=float(parameters['Electrode width [m]'])*float(parameters['Electrode height [m]'])*float(parameters['Number of electrodes connected in parallel to make a cell'])*float(parameters['Negative electrode thickness [m]'])/float(parameters['Lithium metal partial molar volume [m3.mol-1]'])
    def update_parameters(self,parameters):
        from .constitutive import field
        op=self.electrodes['positive']
        for local in op.local:
            if local.reference_concentration!=float(parameters['Initial concentration in positive electrode [mol.m-3]']):raise ValueError('Cathode concentration reference mismatch')
            if local.metadata()['reference_temperature_K']!=float(parameters['Reference temperature [K]']):raise ValueError('Cathode temperature reference mismatch')
            material=local.metadata()['material']
            for key,name in [("Positive electrode Young's modulus [Pa]",'young_Pa'),("Positive electrode Poisson's ratio",'poisson'),('Positive electrode partial molar volume [m3.mol-1]','molar_volume_m3_mol')]:
                if float(parameters[key])!=material[name]:raise ValueError('Mechanical parameter changes require a rebuilt microstructure operator')
        parameters.update({'Positive electrode porosity':(lambda *args:op.porosity) if isinstance(op.porosity,pb.Symbol) else op.porosity,
            'Positive electrode active material volume fraction':(lambda *args:op.active_fraction) if isinstance(op.active_fraction,pb.Symbol) else op.active_fraction})
        return parameters
    def var_pts(self,model,separator_nodes=5):
        return {**model.default_var_pts,'x_n':2,'x_s':separator_nodes,'x_p':self.electrodes['positive'].nodes,'r_p':2}


class MetalInventory(pb.BaseSubModel):
    def __init__(self,param,options,stack):super().__init__(param,options=options);self.stack=stack
    def get_fundamental_variables(self):
        name='MORQEX Li metal inventory [mol]'
        return {name:pb.Variable(name,scale=self.stack.metal_inventory_initial,bounds=(0,np.inf))}
    def get_coupled_variables(self,v):
        A=self.param.L_y*self.param.L_z*self.param.n_electrodes_parallel
        jsei=pb.yz_average(v['Negative electrode SEI interfacial current density [A.m-2]'])
        rate=-A*jsei/self.param.F
        v['MORQEX metal SEI lithium rate [mol.s-1]']=rate
        v['MORQEX metal side chemical power [W]']=-self.param.F*self.param.n.prim.U_sei*rate
        v['MORQEX metal SEI lithium loss [mol]']=self.param.n_electrodes_parallel*v['Loss of lithium to negative SEI [mol]']
        return v
    def set_rhs(self,v):
        A=self.param.L_y*self.param.L_z*self.param.n_electrodes_parallel
        self.rhs[v['MORQEX Li metal inventory [mol]']]=-A*pb.yz_average(v['Lithium metal plating current density [A.m-2]'])/self.param.F
    def set_initial_conditions(self,v):self.initial_conditions[v['MORQEX Li metal inventory [mol]']]=pb.Scalar(self.stack.metal_inventory_initial)
    def add_events_from(self,v):self.events.append(pb.Event('MORQEX Li metal inventory exhausted',v['MORQEX Li metal inventory [mol]']-self.stack.metal_inventory_initial*1e-6))


def lithium_halfcell_parameters(*,host='NMC532',active_fraction=.2569580078125,stoichiometry_window=(.25,.9),overrides=None):
    from ..electrochemistry.open_materials import OpenMaterialCard,tabulate_parameter_ocps
    from pybamm.input.parameters.lithium_ion.Xu2019 import li_metal_electrolyte_exchange_current_density_Xu2019
    p,audit=OpenMaterialCard.host_card(host,supplements=dict(pb.ParameterValues('OKane2022').items()),
        source_note='Explicit missing-key supplements from OKane2022; fixed-C/CBD parameters are not a material calibration').build()
    lo,hi=stoichiometry_window
    if not 0.001<lo<hi<.999 or not 0<active_fraction<1:raise ValueError('Explicit in-table cathode SOC window and phase fraction required')
    p.update({'Thermodynamic factor':1.,'Cation transference number':.3,
        'Negative electrode OCP [V]':0.,'Negative electrode OCP entropic change [V.K-1]':0.,
        'Negative electrode conductivity [S.m-1]':10776000.,'Negative electrode thickness [m]':50e-6,
        'Exchange-current density for lithium metal electrode [A.m-2]':li_metal_electrolyte_exchange_current_density_Xu2019,
        'Negative electrode charge transfer coefficient':.5,'Negative electrode double-layer capacity [F.m-2]':.2,
        'Negative electrode density [kg.m-3]':534.,'Negative electrode specific heat capacity [J.kg-1.K-1]':24.822/.00694,
        'Negative electrode thermal conductivity [W.m-1.K-1]':84.8,
        'Electrode width [m]':.01,'Electrode height [m]':.01,'Number of electrodes connected in parallel to make a cell':1.,
        'Positive electrode active material volume fraction':active_fraction,
        'Initial concentration in positive electrode [mol.m-3]':lo*float(p['Maximum concentration in positive electrode [mol.m-3]'])},check_already_exists=False)
    curves,table_audit=tabulate_parameter_ocps(p)
    # A metal reference is constant; only the cathode needs an interpolation.
    for d,curve in curves.items():p.update({f'{d.capitalize()} electrode OCP [V]':curve,f'{d.capitalize()} electrode OCP entropic change [V.K-1]':curve.entropy})
    positive=curves['positive'];U=lambda x:float(positive(pb.Scalar(x)).evaluate())
    p.update({'Upper voltage cut-off [V]':U(lo),'Lower voltage cut-off [V]':U(hi)})
    area=float(p['Electrode width [m]'])*float(p['Electrode height [m]']);length=float(p['Positive electrode thickness [m]']);cmax=float(p['Maximum concentration in positive electrode [mol.m-3]'])
    capacity=area*length*active_fraction*cmax*(hi-lo)*96485.33212/3600
    total_length=sum(float(p[key]) for key in ['Negative electrode thickness [m]','Separator thickness [m]','Positive electrode thickness [m]','Negative current collector thickness [m]','Positive current collector thickness [m]'])
    p.update({'Nominal cell capacity [A.h]':capacity,'Current function [A]':capacity,
        'Cell volume [m3]':area*total_length,'Cell cooling surface area [m2]':2*area+4*.01*total_length})
    if overrides:p.update(overrides,check_already_exists=False)
    audit.update(halfcell={'counterelectrode':'planar Li metal, native Xu2019 exchange kinetics; no negative particles',
        'stoichiometry_window':list(stoichiometry_window),'nominal_capacity_Ah':capacity,'voltage_window_V':[U(hi),U(lo)],
        'metal_Cp_source':'NIST / Cox Wagman 1984, Cp(298.15K)=24.822 J/mol/K; constant reference heat capacity',
        'metal_density_and_conductivity_source':'Arcadium Lithium technical product data / Handbook of Lithium; 534 kg/m3, 84.8 W/m/K',
        'scope':'fixed geometry; no moving Li surface, dendrites or metal plasticity; positive electrode mechanics retained'},ocp=table_audit)
    return p,audit
