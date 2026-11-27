"""Dynamic graph conductance / reactive-area inputs with persistent Li states.

Regions are material inventories, never deleted on disconnection. The graph
layout and material volumes stay fixed; positive edge weights may become zero.
No static condensation of these transient diffusion states is performed.
This is an isothermal, lagged micro/macro port, not an unconditional global
energy theorem. Intra-region fracture requires enrichment before this closure
can be qualified as an accurate representation of that fracture.
"""
import numpy as np
import pybamm as pb
from .region_dfn import RegionParticle,RegionEpochDFN,mu_input
from morqex.electrochemistry.microstructure_dfn import MicrostructureKinetics,key,dot
from .epoch_dfn import GeometryEfficiency


def area_input(k):return f'TOP-DAMAGE region {k} reactive area fraction'
def edge_input(i,j):return f'TOP-DAMAGE diffusion edge {i} {j} factor'
def spatial_input(name):
    return pb.SecondaryBroadcast(pb.InputParameter(name,domain='positive electrode'),'current collector')


class SpatialGeometryEfficiency(GeometryEfficiency):
    def get_coupled_variables(self,v):
        parts={}
        for domain in self.options.whole_cell_domains:
            if domain=='positive electrode':
                parts[domain]=(spatial_input('TOP-DAMAGE electrolyte efficiency') if self.component=='Electrolyte'
                    else pb.FullBroadcast(1.,domain,'current collector'))
            elif self.component=='Electrolyte':
                parts[domain]=v[domain.capitalize()+' porosity']**self.param.domain_params[domain.split()[0]].b_e
            else:parts[domain]=pb.FullBroadcast(0,domain,'current collector')
        v.update(self._get_standard_transport_efficiency_variables(parts));return v


def diffusion_edges(port):
    upper=np.triu(port.stiffness.toarray(),1)
    i,j=np.nonzero(upper)
    if np.any(upper[i,j]>=0):raise ValueError('Diffusion graph requires nonpositive off-diagonal entries')
    return np.column_stack([i,j]),-upper[i,j]


class NetworkKinetics(MicrostructureKinetics):
    def get_coupled_variables(self,v):
        # Standard equations use reference area A0 and an effective current
        # density j_eff=(A_accessible/A0)*j_BV. A0*j_eff is exactly A*j_BV.
        v=super().get_coupled_variables(v)
        d,D=self.domain_Domain;op=self.op;currents=[];exchange=[];eta=[]
        for k in range(op.dimension):
            factor=spatial_input(area_input(k))
            j=v[key(d,k,'reaction current [A.m-2]')]*factor
            j0=v[key(d,k,'exchange current [A.m-2]')]*factor
            v[key(d,k,'reaction current [A.m-2]')]=j
            v[key(d,k,'exchange current [A.m-2]')]=j0
            currents.append(j);exchange.append(j0);eta.append(v[key(d,k,'reaction overpotential [V]')])
        v.update(self._get_standard_interfacial_current_variables(dot(op.reaction_area,currents)/op.area_density))
        v.update(self._get_standard_exchange_current_variables(dot(op.reaction_area,exchange)/op.area_density))
        v.update(self._get_standard_volumetric_current_density_variables(v))
        v['TOP-DAMAGE accessible reactive area [m-1]']=dot(op.reaction_area,[spatial_input(area_input(k)) for k in range(op.dimension)])
        return v


class NetworkParticle(RegionParticle):
    def __init__(self,*args,local_mobility=False,diffusivity_interpretation='tracer_reference',**kwargs):
        super().__init__(*args,**kwargs);self.local_mobility=local_mobility
        self.diffusivity_interpretation=diffusivity_interpretation

    def get_coupled_variables(self,v):
        d,D=self.domain_Domain;op=self.op;phase=self.phase_param;law=self.law
        T=v[f'{D} electrode temperature [K]']
        c=[v[key(d,k,'concentration [mol.m-3]')] for k in range(op.dimension)]
        chemical=[-self.param.F*phase.U(a/phase.c_max,T) for a in c]
        # The reference gradient energy remains fixed. Changing it requires
        # accounting for its structural work; it is NOT silently rescaled.
        grad=[law.gradient_coefficient_J_m5_mol2*dot(row,c)/float(w) for row,w in zip(op.stiffness.toarray(),op.weights)]
        mu=[a+v[key(d,k,'elastic chemical potential [J.mol-1]')]+grad[k] for k,a in enumerate(chemical)]
        mean=dot(op.weights,c)/op.active_fraction
        mobility=law.mobility_multiplier*phase.D(mean,T)*phase.c_max/(self.param.R*T)
        if law.mobility=='lattice':mobility*=mean/phase.c_max*(1-mean/phase.c_max)
        diffusive=[pb.Scalar(0) for _ in c];diss_terms=[]
        edges,weights=diffusion_edges(op)
        for (i,j),g in zip(edges,weights):
            edge_mobility=mobility
            if self.local_mobility:
                # Local symmetric midpoint Onsager mobility. Besides removing
                # artificial cross-particle coefficient dependence, this keeps
                # the transient Jacobian local instead of a dense global-mean
                # rank-one update expanded through hundreds of state symbols.
                middle=(c[i]+c[j])/2
                if self.diffusivity_interpretation=='chemical_diffusivity':
                    # Only admissible after strict monotonicity verification.
                    # Include the exact temperature-dependent OCP and native
                    # asymptotic term in the chemical curvature derivative.
                    sto=middle/phase.c_max
                    curvature=-self.param.F*phase.U(sto,T).diff(sto)/phase.c_max
                    edge_mobility=law.mobility_multiplier*phase.D(middle,T)/curvature
                else:
                    edge_mobility=law.mobility_multiplier*phase.D(middle,T)*phase.c_max/(self.param.R*T)
                    if law.mobility=='lattice':edge_mobility*=middle/phase.c_max*(1-middle/phase.c_max)
            flux=float(g)*spatial_input(edge_input(i,j))*edge_mobility*(mu[i]-mu[j])
            v[f'TOP-DAMAGE diffusion edge {i} {j} molar transfer [mol.m-3.s-1]']=flux
            diffusive[i]-=flux;diffusive[j]+=flux;diss_terms.append(flux*(mu[i]-mu[j]))
        reaction_current=[v[key(d,k,'reaction current [A.m-2]')] for k in range(op.dimension)]
        massdot=[a-float(A)*b/self.param.F for a,A,b in zip(diffusive,op.reaction_area,reaction_current)]
        # Compact linear ports avoid O(number_of_edges) expression-tree depth.
        # This is the same sum, with identical analytic derivatives; no terms
        # are discarded and the Python recursion limit is not increased.
        diss=dot(np.ones(len(diss_terms)),diss_terms)
        reaction=dot(op.reaction_area,[a*v[key(d,k,'reaction overpotential [V]')] for k,a in enumerate(reaction_current)])
        power=dot(np.ones(len(mu)),[a*b for a,b in zip(massdot,mu)])
        port=v[f'{D} electrode surface potential difference [V]']*dot(op.reaction_area,reaction_current)
        for k,rate in enumerate(massdot):v[key(d,k,'rate [mol.m-3.s-1]')]=rate/float(op.weights[k])
        v.update({f'MORQEX {d} diffusion dissipation [W.m-3]':diss,
                  f'MORQEX {d} reaction dissipation [W.m-3]':reaction,
                  f'MORQEX {d} port identity residual [W.m-3]':power+diss+reaction-port})
        v.update(self._get_standard_diffusivity_variables(phase.D(mean,T)))
        v.update(self._get_standard_flux_variables(0*mean));return v


class NetworkEpochDFN(RegionEpochDFN):
    def __init__(self,port,**kwargs):
        super().__init__(port,dynamic_topology=True,**kwargs)
        self.edges,self.edge_weights=diffusion_edges(port)

    def advance_network(self,dt,current_A,feedback,mechanical_potential_J_mol,*,reaction_fractions,edge_factors,
                        electronic_conductivity_S_m=None,electrolyte_efficiency=None):
        mu=np.asarray(mechanical_potential_J_mol,float)
        area=np.asarray(reaction_fractions,float);edge=np.asarray(edge_factors,float)
        if mu.shape!=(self.nx,self.port.dimension) or not np.isfinite(mu).all():raise ValueError('Finite local mechanical potentials required')
        if area.shape!=mu.shape or not np.isfinite(area).all() or np.any((area<0)|(area>1)):
            raise ValueError('Accessible area fractions in [0,1] per region and macro node required')
        if edge.shape!=(self.nx,len(self.edges)) or not np.isfinite(edge).all() or np.any(edge<0):
            raise ValueError('Nonnegative edge factors per macro node required')
        if np.any(area@self.port.reaction_area<=0):
            raise ValueError('No reactive interface: terminate or solve the disconnected electrode state')
        if feedback.reaction_fraction!=1 or feedback.mechanical_potential_J_mol!=0:
            raise ValueError('Local network inputs replace scalar reaction/mu corrections')
        inputs=feedback.inputs(current_A)
        for name,field,upper in [('TOP-DAMAGE electronic conductivity [S.m-1]',electronic_conductivity_S_m,float('inf')),
                                 ('TOP-DAMAGE electrolyte efficiency',electrolyte_efficiency,1.)]:
            value=np.broadcast_to(inputs[name] if field is None else field,(self.nx,)).copy()
            if not np.isfinite(value).all() or np.any(value<=0) or np.any(value>upper):raise ValueError('Finite positive spatial transport inputs required')
            inputs[name]=value
        inputs.update({mu_input(k):mu[:,k] for k in range(self.port.dimension)})
        inputs.update({area_input(k):area[:,k] for k in range(self.port.dimension)})
        inputs.update({edge_input(i,j):edge[:,k] for k,(i,j) in enumerate(self.edges)})
        previous=self.simulation._solution
        try:return self._advance_regions(dt,current_A,feedback,inputs)
        except Exception:
            self.simulation._solution=previous
            raise

    def advance_regions(self,*args,**kwargs):
        raise TypeError('Dynamic topology requires explicit advance_network inputs')
