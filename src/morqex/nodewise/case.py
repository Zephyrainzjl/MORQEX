"""Configurable preparation and composition API for heterogeneous DFN nodes."""
from dataclasses import dataclass,asdict
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from ..gemsim import GEMsim
from ..topology import PorousCBD
from ..sparse_cohesive import build_sparse_operator
from ..electrochemistry.microstructure_operator import MicrostructureElectrode
from ..electrochemistry.open_particle import TransportLaw,build_open_dfn
from ..electrochemistry.aging_dfn import AgingMechanics
from ..electrochemistry.open_materials import tabulate_parameter_ocps
from ..degradation import DegradationOptions,NoDamage
from .cache import save_node_kernel,load_node_kernel
from .constitutive import NodewiseStack


class ModalMechanics(AgingMechanics):
    modal_variables=True


@dataclass(frozen=True)
class Physics:
    mechanics:bool=True
    thermal:str='lumped'
    mechanical_feedback:float=1.
    constraint:str='fixed-thickness'
    def __post_init__(self):
        if self.thermal not in {'lumped','x-full','isothermal'} or self.constraint not in {'pressure','fixed-thickness'} or not 0<=self.mechanical_feedback<=1:raise ValueError('Supported thermal/constraint modes and feedback in [0,1] required')


@dataclass
class PreparedCase:
    workspace:object
    bindings:dict
    paths:dict
    recipe:dict
    def kernels(self):return {d:[load_node_kernel(p) for p in paths] for d,paths in self.paths.items()}
    def build(self,parameters,*,physics=None,degradation=None,separator_nodes=5,solver_options=None,factories=None,control=None,thermal_boundary=None,thermal_cells=None,thermal_damage_tables=None):
        import pybamm as pb
        from ..solvers.pybamm_backend import solver
        physics=physics or Physics();degradation=degradation or DegradationOptions();p=parameters.copy()
        if not physics.mechanics and degradation.damage.name!='none':raise ValueError('Disable mechanical damage explicitly for electrochemistry-only physics')
        curves,ocp_audit=tabulate_parameter_ocps(p)
        for d,curve in curves.items():p.update({f'{d.capitalize()} electrode OCP [V]':curve,f'{d.capitalize()} electrode OCP entropic change [V.K-1]':curve.entropy})
        stack=NodewiseStack(self.kernels(),p,constraint=physics.constraint,allow_linear_ocp_entropy=True)
        stack.update_parameters(p)
        if thermal_cells is not None:
            from .heat import apply_thermal_homogenization
            apply_thermal_homogenization(stack,p,thermal_cells)
        if thermal_damage_tables is not None:stack.thermal_damage_tables=thermal_damage_tables
        if physics.thermal=='x-full':
            if thermal_boundary is None:raise ValueError('Supply explicit pouch thermal boundary coefficients for x-full heat transport')
            thermal_boundary.apply(p)
        roles={'mechanics':ModalMechanics};roles.update(factories or {})
        feedback=physics.mechanical_feedback if physics.mechanics else 0.
        model=build_open_dfn(stack,stack.kernels,degradation,parameters=p,feedback=feedback,
            thermal_mode=physics.thermal,factories=roles,control=control)
        self.ocp_audit=ocp_audit
        mesh_types=dict(model.default_submesh_types)
        starts={'negative':0.,'positive':stack.lengths['negative']+stack.separator_length}
        for d,nodes in self.bindings.items():
            edges=np.array([nodes[0].selection['macro_cell_fraction'][0],*[n.selection['macro_cell_fraction'][1] for n in nodes]])
            if not np.allclose(np.diff(edges),1/len(nodes),rtol=0,atol=1e-14):
                actual=starts[d]+edges*stack.lengths[d]
                mesh_types[d+' electrode']=pb.MeshGenerator(pb.UserSupplied1DSubMesh,submesh_params={'edges':actual})
        simulation=pb.Simulation(model,parameter_values=p,solver=solver(**(solver_options or {'rtol':1e-8,'atol':1e-9})),
            var_pts=stack.var_pts(model,separator_nodes),submesh_types=mesh_types)
        return simulation,stack,p


def prepare_nodes(workspace,parameters,bindings,*,target_modes=16,transport_law=None,
                  topology=None,materials=None,thermal_expansion_per_K=1e-5,cbd_young_Pa=1e9,
                  operator_builder=None,operator_options=None):
    """Prepare each distinct geometry once; every location retains its state.

    ``bindings`` contains negative/positive NodeGeometry sequences. ``target_modes``
    accepts a scalar, per-electrode scalar, or per-node sequence. Any builder must
    return the conservative operator and a derivative-consistent energy response.
    Code/geometry/material/basis/solver fingerprints define the cache recipe.
    """
    if set(bindings)!={'negative','positive'}:raise ValueError('Bindings for both electrodes required')
    if transport_law is None:transport_law=TransportLaw()
    provider=GEMsim(workspace);topology=topology or PorousCBD();builder=operator_builder or build_sparse_operator
    code=inspect.getsourcefile(builder)
    if code is None:raise ValueError('Custom builder needs an inspectable versioned source file')
    code_hash=hashlib.sha256(Path(code).read_bytes()).hexdigest();options=operator_options or {}
    paths={};recipes={}
    for d,nodes in bindings.items():
        if len(nodes)<2 or any(n.index!=i or n.domain!=d for i,n in enumerate(nodes)):raise ValueError('Ordered, domain-matching DFN bindings with at least two nodes required')
        paths[d]=[];recipes[d]=[]
        for i,node in enumerate(nodes):
            target=target_modes[d] if isinstance(target_modes,dict) else target_modes
            if not isinstance(target,(int,np.integer)):target=target[i]
            material={name:float(parameters[name]) for name in [f'Initial concentration in {d} electrode [mol.m-3]',
                'Reference temperature [K]',f'{d.capitalize()} electrode Young\'s modulus [Pa]',f'{d.capitalize()} electrode Poisson\'s ratio',
                f'{d.capitalize()} electrode partial molar volume [m3.mol-1]']}
            record={'schema':'node-recipe-1','geometry':node.operator_geometry.fingerprint,'material':material,'target_modes':int(target),
                'transport_law':asdict(transport_law),'thermal_expansion':thermal_expansion_per_K,'CBD_modulus':cbd_young_Pa,
                'topology':topology.metadata(),'material_catalog':materials.metadata() if materials is not None else None,
                'builder':builder.__module__+'.'+builder.__qualname__,'builder_sha256':code_hash,'operator_options':options}
            record=json.loads(json.dumps(record,sort_keys=True,allow_nan=False))
            identity=hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False).encode()).hexdigest();path=workspace.path('operators','nodewise',identity)
            if (path/'manifest.json').exists():
                if json.loads((path/'recipe.json').read_text())!=record:raise ValueError('Operator cache recipe mismatch')
                load_node_kernel(path)
            else:
                full=MicrostructureElectrode(node.operator_geometry,provider,parameters,d,topology=topology,materials=materials,
                    thermal_expansion=thermal_expansion_per_K,cbd_young=cbd_young_Pa)
                op,response=builder(full,target=int(target),**options);op.transport_law=asdict(transport_law)
                save_node_kernel(path,op,response);(path/'recipe.json').write_text(json.dumps(record,indent=2))
                del op,response,full
            paths[d].append(path);recipes[d].append(record)
    return PreparedCase(workspace,bindings,paths,recipes)


def mechanical_snapshot(kernels,concentrations,temperatures,damage,parameters,*,pressure_Pa=None,thickness_change_m=None,separator_modulus_Pa=1e8):
    """Standalone mechanics for prescribed electrochemical/thermal histories.

    Zero- or nonzero-pressure mechanical snapshots need no DFN solve. A coupled
    nonzero-pressure transient additionally needs its external work in the heat
    balance; that transient is deliberately not inferred from this static API.
    """
    if (pressure_Pa is None)==(thickness_change_m is None):raise ValueError('Specify pressure or total thickness change')
    Ls=float(parameters['Separator thickness [m]']);C=Ls/separator_modulus_Pa;N=0.;local={}
    for d,bank in kernels.items():
        L=float(parameters[d.capitalize()+' electrode thickness [m]']);rows=[]
        for i,k in enumerate(bank):
            H=k.hessian(float(damage[d][i]));c=np.asarray(concentrations[d][i])-k.operator.reference_concentration;theta=temperatures[d][i]-float(parameters['Reference temperature [K]'])
            b=H[0,1:-1]@c+H[0,-1]*theta;C+=L/len(bank)/H[0,0];N+=L/len(bank)*b/H[0,0];rows.append((H,c,theta,b))
        local[d]=rows
    stress=-float(pressure_Pa) if pressure_Pa is not None else (float(thickness_change_m)+N)/C
    output={'stress_Pa':stress,'negative_end_traction_Pa':-stress,'positive_end_traction_Pa':stress,'nodes':{},'thickness_change_m':Ls*stress/separator_modulus_Pa}
    for d,rows in local.items():
        output['nodes'][d]=[]
        for kernel,(H,c,theta,b) in zip(kernels[d],rows):
            E=(stress-b)/H[0,0];q=np.r_[E,c,theta]
            output['nodes'][d].append({'strain':E,'energy_J_m3':.5*q@H@q,'chemical_potential_J_mol':(H@q)[1:-1]/kernel.operator.weights})
            output['thickness_change_m']+=float(parameters[d.capitalize()+' electrode thickness [m]'])*E/len(rows)
    return output
