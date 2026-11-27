"""Current-controlled electronic/ionic graph with explicit nonlinear BV ports.

Only passive, source-free interior vertices are Schur-condensed. Every reaction
vertex remains a port. This is an isothermal quasi-static *working-potential*
problem for supplied concentrations/kinetics, not a diffusion time integrator or
an automatically coupled DFN closure. Equilibrium voltages and exchange currents
must come from the same chemical model used by the transient inventory solver.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve,MatrixRankWarning
import warnings
from .contracts import ConductingNetwork,_owned
from .graph_transport import KronOperator


@dataclass(frozen=True)
class ReactionFaces:
    electronic_vertices: np.ndarray
    ionic_vertices: np.ndarray
    area_m2: np.ndarray
    equilibrium_V: np.ndarray
    exchange_A_m2: np.ndarray
    provenance: str

    def __post_init__(self):
        e=np.asarray(self.electronic_vertices);i=np.asarray(self.ionic_vertices)
        a=np.asarray(self.area_m2,float);u=np.asarray(self.equilibrium_V,float);j=np.asarray(self.exchange_A_m2,float)
        if e.ndim!=1 or not len(e) or i.shape!=e.shape or e.dtype.kind not in 'iu' or i.dtype.kind not in 'iu' or np.any(e<0) or np.any(i<0):
            raise ValueError('Explicit nonnegative integer electronic/ionic face vertices required')
        if any(v.shape!=e.shape for v in (a,u,j)) or not all(np.isfinite(v).all() for v in (a,u,j)) or np.any(a<=0) or np.any(j<0) or not self.provenance.strip():
            raise ValueError('Positive physical areas, finite OCP and nonnegative exchange currents with provenance required')
        for name,value in [('electronic_vertices',e),('ionic_vertices',i),('area_m2',a),('equilibrium_V',u),('exchange_A_m2',j)]:
            object.__setattr__(self,name,_owned(value))


@dataclass(frozen=True)
class WorkingNetworkState:
    electronic_potential_V: np.ndarray
    ionic_potential_V: np.ndarray
    reaction_current_A: np.ndarray
    overpotential_V: np.ndarray
    voltage_V: float
    discharge_current_A: float
    electronic_dissipation_W: float
    ionic_dissipation_W: float
    reaction_dissipation_W: float
    relative_charge_residual: float
    power_defect_W: float
    iterations: int
    retained_electronic_vertices: int
    retained_ionic_vertices: int
    voltage_defined: bool
    potential_precision: str = 'float64'


@dataclass(frozen=True)
class ReactionPortErrorBound:
    voltage_absolute_bound_V: float
    retained_energy_error_bound: float
    relative_edge_change: float
    cached_nonlinear_residual_dual_norm: float
    scope: str = ('Strong-monotonicity voltage bound for explicitly supplied reaction states/current and '
                  'unchanged positive edge/reaction support; floating-point evaluation, not interval-certified. '
                  'Not an accumulated DFN or fatigue error bound.')


class ReactiveDualNetwork:
    """Monotone symmetric Butler--Volmer coupled through two physical graphs.

    Positive applied current denotes discharge and sum(j_face A_face)=-I.
    The only prescribed potential is the physical ionic reservoir (zero).
    The electronic collector potential is solved, not imposed or fitted.
    No artificial opposite-face homogenization terminal may remain in a graph.
    Floating but reactive particles remain present: their net reaction is zero,
    but local internal redox currents are permitted by Kirchhoff balance.
    """
    def __init__(self,electronic,ionic,faces,*,electronic_collector,ionic_reservoir,
                 temperature_K=298.15,reduce=True,max_ports=512):
        if not isinstance(electronic,ConductingNetwork) or not isinstance(ionic,ConductingNetwork) or electronic.channel!='electronic' or ionic.channel!='ionic':
            raise ValueError('Two explicitly typed physical conducting networks required')
        if not isinstance(faces,ReactionFaces):raise ValueError('Physical reaction-face data required')
        if np.max(faces.electronic_vertices)>=electronic.vertex_count or np.max(faces.ionic_vertices)>=ionic.vertex_count:
            raise ValueError('Reaction vertex outside its physical graph')
        for value,n in [(electronic_collector,electronic.vertex_count),(ionic_reservoir,ionic.vertex_count)]:
            if type(value) is not int or not 0<=value<n:raise ValueError('One valid reservoir per physical channel required')
        if not np.array_equal(electronic.anchors,[electronic_collector]) or not np.array_equal(ionic.anchors,[ionic_reservoir]):
            raise ValueError('Graphs must contain exactly the declared physical reservoirs')
        if not np.isfinite(temperature_K) or temperature_K<=0:raise ValueError('Positive absolute temperature required')
        self.electronic=electronic;self.ionic=ionic;self.faces=faces
        self.temperature_K=float(temperature_K);self.k=96485.33212/(2*8.31446261815324*self.temperature_K)
        et=np.unique(np.r_[electronic_collector,faces.electronic_vertices]) if reduce else np.arange(electronic.vertex_count)
        it=np.unique(np.r_[ionic_reservoir,faces.ionic_vertices]) if reduce else np.arange(ionic.vertex_count)
        self.eop=KronOperator(electronic,et,max_terminals=max_ports)
        self.iop=KronOperator(ionic,it,max_terminals=max_ports)
        ne=len(et);ni=len(it);self.ne=ne
        self.collector=int(np.flatnonzero(et==electronic_collector)[0])
        self.reservoir=ne+int(np.flatnonzero(it==ionic_reservoir)[0])
        ep=np.searchsorted(et,faces.electronic_vertices);ip=ne+np.searchsorted(it,faces.ionic_vertices)
        nface=len(ep);self.B=sparse.coo_matrix((np.r_[np.ones(nface),-np.ones(nface)],
            (np.r_[np.arange(nface),np.arange(nface)],np.r_[ep,ip])),shape=(nface,ne+ni)).tocsr()
        self.S=sparse.block_diag((self.eop.matrix_S,self.iop.matrix_S),format='csr')
        upper=sparse.triu(self.S,k=1).tocoo()
        self.passive_i,self.passive_j=upper.row,upper.col
        self.passive_g=-upper.data
        self.passive_row_sum=np.asarray(self.S.sum(axis=1)).ravel()
        self.amplitude=2*faces.area_m2*faces.exchange_A_m2
        self.offset=float(np.average(faces.equilibrium_V,weights=self.amplitude)) if self.amplitude.sum()>0 else float(faces.equilibrium_V.mean())
        # Gauges follow the union of *physical* passive edges and active BV
        # links, not small numerical entries in a condensed dense matrix.
        en=electronic.vertex_count;total=en+ionic.vertex_count
        ee=electronic.edges[electronic.conductance_S>0]
        ii=ionic.edges[ionic.conductance_S>0]+en
        active=self.amplitude>0
        rr=np.column_stack([faces.electronic_vertices[active],faces.ionic_vertices[active]+en])
        joined=np.concatenate([ee,ii,rr])
        adj=sparse.coo_matrix((np.ones(len(joined)),(joined[:,0],joined[:,1])),shape=(total,total)).tocsr()
        _,component=connected_components(adj,directed=False)
        retained=component[np.r_[et,it+en]]
        self.voltage_defined=bool(component[electronic_collector]==component[en+ionic_reservoir])
        fixed=[self.reservoir]
        for comp in np.unique(retained):
            if comp==retained[self.reservoir]:continue
            candidates=np.flatnonzero(retained==comp)
            ionic_candidates=candidates[candidates>=ne]
            fixed.append(int(ionic_candidates[0] if len(ionic_candidates) else candidates[0]))
        self.fixed=np.asarray(fixed,int);self.free=np.setdiff1d(np.arange(ne+ni),self.fixed)

    def with_reaction_state(self,faces,*,temperature_K=None):
        """Reuse passive factorizations while solving *new* BV/charge equations.

        Positive reaction support and incidence must be unchanged. This returns
        a distinct problem; no accepted state or old kinetics are mutated.
        It does not approximate a change in passive conductance by itself.
        """
        if (not isinstance(faces,ReactionFaces)
                or not np.array_equal(faces.electronic_vertices,self.faces.electronic_vertices)
                or not np.array_equal(faces.ionic_vertices,self.faces.ionic_vertices)):
            raise ValueError('Changed reaction-face topology requires reconstruction')
        amplitude=2*faces.area_m2*faces.exchange_A_m2
        if not np.array_equal(amplitude>0,self.amplitude>0):
            raise ValueError('Changed positive reaction support requires reconstruction')
        temperature=self.temperature_K if temperature_K is None else float(temperature_K)
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('Positive absolute temperature required')
        result=object.__new__(type(self));result.__dict__=self.__dict__.copy()
        result.faces=faces;result.amplitude=amplitude;result.temperature_K=temperature
        result.k=96485.33212/(2*8.31446261815324*temperature)
        result.offset=float(np.average(faces.equilibrium_V,weights=amplitude)) if amplitude.sum()>0 else float(faces.equilibrium_V.mean())
        return result

    def initial_from(self,state):
        """Retain previous physical potentials while applying this OCP shift."""
        if (not isinstance(state,WorkingNetworkState)
                or state.electronic_potential_V.shape!=(self.electronic.vertex_count,)
                or state.ionic_potential_V.shape!=(self.ionic.vertex_count,)):
            raise ValueError('Compatible previous physical graph potentials required')
        return np.r_[state.electronic_potential_V[self.eop.terminals]-self.offset,
                     state.ionic_potential_V[self.iop.terminals]]

    def _fields(self,v,current):
        eta=self.B@v+(self.offset-self.faces.equilibrium_V)
        z=self.k*eta
        if not np.isfinite(z).all() or np.max(abs(z),initial=0)>600:return None
        j=self.amplitude*np.sinh(z)
        dj=self.amplitude*self.k*np.cosh(z)
        # Evaluate the identical quadratic form through potential differences.
        # Direct v.T*S*v subtracts large near-equal common-mode terms when
        # ionic/electronic conductance greatly exceeds reaction conductance.
        # Retain the numerical row-sum term, rather than silently modifying S.
        dv=v[self.passive_i]-v[self.passive_j]
        current_edges=self.passive_g*dv
        if v.dtype==np.dtype(np.longdouble):
            passive=self.passive_row_sum*v
            np.add.at(passive,self.passive_i,current_edges);np.add.at(passive,self.passive_j,-current_edges)
        else:
            passive=(np.bincount(self.passive_i,weights=current_edges,minlength=len(v))
                     -np.bincount(self.passive_j,weights=current_edges,minlength=len(v))
                     +self.passive_row_sum*v)
        residual=passive+self.B.T@j
        residual[self.collector]+=current
        # Drop only a constant I*offset from the objective. Using shifted solid
        # potentials avoids cancellation of a common ~4 V in ohmic graph terms.
        passive_energy=float(self.passive_g@dv**2+self.passive_row_sum@v**2)
        objective=.5*passive_energy+float(np.sum(self.amplitude/self.k*2*np.sinh(z/2)**2))+current*v[self.collector]
        return eta,j,dj,residual,objective

    def solve(self,discharge_current_A,*,initial=None,tolerance=1e-10,max_iterations=60,diagnostics=None,extended_precision=False):
        """FP64 tangents, optionally extended-precision potentials/residuals.

        The optional mode addresses sub-ulp potential corrections in highly
        conducting, weakly reacting full graphs. It does not relax charge gates.
        Platforms without a wider longdouble type reject the requested mode.
        """
        current=float(discharge_current_A)
        if not np.isfinite(current) or not np.isfinite(tolerance) or not 0<tolerance<1e-3 or type(max_iterations) is not int or max_iterations<1:
            raise ValueError('Finite current and explicit nonlinear solver controls required')
        if current!=0 and not self.voltage_defined:raise ValueError('No physical reaction path between reservoirs: current-controlled problem is open circuit')
        if type(extended_precision) is not bool:raise ValueError('Boolean precision control required')
        if extended_precision and np.finfo(np.longdouble).eps>=np.finfo(float).eps:
            raise ValueError('Extended potential precision unavailable on this platform')
        dtype=np.longdouble if extended_precision else float
        v=np.zeros(self.S.shape[0],dtype=dtype) if initial is None else np.array(initial,dtype=dtype,copy=True)
        if v.shape!=(self.S.shape[0],) or not np.isfinite(v).all():raise ValueError('Finite retained shifted potentials required')
        v[self.fixed]=0.
        for iteration in range(max_iterations+1):
            state=self._fields(v,current)
            if state is None:raise RuntimeError('Invalid initial BV potential; no state accepted')
            eta,j,dj,residual,objective=state
            scale=max(abs(current),float(np.sum(abs(j))),1e-30)
            defect=float(np.max(abs(residual[self.free]),initial=0))/scale
            if diagnostics is not None:diagnostics.append({'iteration':iteration,'relative_residual':defect,'objective':float(objective)})
            if defect<=tolerance:break
            if iteration==max_iterations:raise RuntimeError(f'Reactive dual-network Newton failed charge balance: residual={defect:.6e}, tolerance={tolerance:.6e}, iterations={iteration}')
            H=(self.S+self.B.T@sparse.diags(np.asarray(dj,float))@self.B).tocsr()[self.free][:,self.free]
            diag=H.diagonal()
            if np.any(diag<=0) or not np.isfinite(diag).all():raise RuntimeError('Unresolved null mode in reactive dual network')
            scaling=1/np.sqrt(diag);D=sparse.diags(scaling)
            with warnings.catch_warnings():
                warnings.simplefilter('error',MatrixRankWarning)
                direction=scaling*spsolve((D@H@D).tocsc(),np.asarray(-scaling*residual[self.free],float))
            if not np.isfinite(direction).all():raise RuntimeError('Reactive graph tangent solve failed')
            slope=float(residual[self.free]@direction)
            accepted=False;step=1.
            roundoff=128*np.finfo(float).eps*max(abs(objective),float(np.sum(abs(j*eta))),abs(current)*max(abs(v[self.collector]),1e-3),1e-30)
            for _ in range(60):
                trial=v.copy();trial[self.free]+=step*direction
                evaluated=self._fields(trial,current)
                if evaluated is not None:
                    new_residual=evaluated[3];new_objective=evaluated[4]
                    descent=abs(step*slope)>roundoff and new_objective<=objective+1e-4*step*slope
                    rounding=(abs(step*slope)<=roundoff and new_objective<=objective+roundoff
                        and np.linalg.norm(new_residual[self.free])<(1-1e-4*step)*np.linalg.norm(residual[self.free]))
                    if descent or rounding:v=trial;accepted=True;break
                step*=.5
            if not accepted:raise RuntimeError('Reactive graph line search failed; no state accepted')
            if diagnostics is not None:diagnostics[-1].update(step=step,slope=slope,roundoff=roundoff)
        e=self.eop.recover(v[:self.ne]);i=self.iop.recover(v[self.ne:])
        voltage=float(v[self.collector]+self.offset)
        reaction=float(j@eta)
        power=e.dissipation_W+i.dissipation_W+reaction+current*voltage+float(j@self.faces.equilibrium_V)
        power_scale=max(e.dissipation_W+i.dissipation_W+abs(reaction),abs(current*voltage),float(np.sum(abs(j*self.faces.equilibrium_V))),1e-30)
        charge=float(max(defect,abs(j.sum()+current)/max(abs(current),float(np.sum(abs(j))),1e-30)))
        if charge>10*tolerance or reaction < -1e-12*power_scale or abs(power)>max(10*tolerance,1e-11)*power_scale:
            raise RuntimeError('Reactive graph charge/power identity failed')
        return WorkingNetworkState(_owned(e.potential_V+self.offset),i.potential_V,_owned(j),_owned(eta),voltage,current,
            e.dissipation_W,i.dissipation_W,reaction,charge,float(power),iteration,
            len(self.eop.terminals),len(self.iop.terminals),self.voltage_defined,str(v.dtype))

    def changed_conductance_bound(self,state,electronic,ionic):
        """Backward-compatible fixed-kinetics/current conductance-only bound."""
        return self.changed_problem_bound(state,electronic,ionic)

    def changed_problem_bound(self,state,electronic,ionic,*,faces=None,temperature_K=None,discharge_current_A=None):
        """Bound a cached nonlinear solution under passive conductance changes.

        No updated graph factorization or Schur matrix is constructed. With
        |g_new/g_old-1|<=eps<1 and identical support, (1-eps)S<=S_new<=
        (1+eps)S. Symmetric BV has derivative >=2*A*j0*k. On the same gauge
        space, H=(1-eps)S+B.T diag(2*A*j0*k)B is a monotonicity lower bound.
        Target OCP, area, exchange current, temperature and current may change
        when explicitly supplied. Let r*=F_old(v)+B.T*(j_new(v)-j_old(v))
        +(I_new-I_old)*e_collector, and use a_new*k_new in H. Then
        ||v-v_new||_H <= ||r*||_{H^-1}+eps/sqrt(1-eps)*sqrt(v.T*S*v).
        A collector dual solve gives its voltage bound. The target reaction
        law must keep the same face incidence and positive reaction support.
        There is no estimate of uncertainty in the supplied chemical state.

        This is a sufficient, potentially conservative local bound. It does
        not certify electrolyte/solid state errors or repeated-cycle drift.
        Support changes, unresolved gauges and uncertain voltage are rejected.
        """
        from scipy.linalg import cho_factor,cho_solve
        if not isinstance(state,WorkingNetworkState) or not state.voltage_defined:
            raise ValueError('A solved, physically defined working voltage required')
        target_faces=self.faces if faces is None else faces
        if (not isinstance(target_faces,ReactionFaces)
                or not np.array_equal(target_faces.electronic_vertices,self.faces.electronic_vertices)
                or not np.array_equal(target_faces.ionic_vertices,self.faces.ionic_vertices)):
            raise ValueError('Changed reaction-face topology requires reconstruction')
        amplitude=2*target_faces.area_m2*target_faces.exchange_A_m2
        if not np.array_equal(amplitude>0,self.amplitude>0):
            raise ValueError('Changed positive reaction support requires reconstruction')
        temperature=self.temperature_K if temperature_K is None else float(temperature_K)
        current=state.discharge_current_A if discharge_current_A is None else float(discharge_current_A)
        if not np.isfinite([temperature,current]).all() or temperature<=0:
            raise ValueError('Finite target current and positive absolute temperature required')
        k=96485.33212/(2*8.31446261815324*temperature)
        eps=0.
        for old,new in [(self.electronic,electronic),(self.ionic,ionic)]:
            if (not isinstance(new,ConductingNetwork) or new.channel!=old.channel
                    or new.vertex_count!=old.vertex_count
                    or not np.array_equal(new.edges,old.edges)
                    or not np.array_equal(new.anchors,old.anchors)):
                raise ValueError('Changed topology requires graph reconstruction')
            positive=old.conductance_S>0
            if not np.array_equal(positive,new.conductance_S>0):
                raise ValueError('Changed positive edge support requires graph reconstruction')
            eps=max(eps,float(np.max(abs(new.conductance_S[positive]/old.conductance_S[positive]-1),initial=0)))
        if eps>=1:raise ValueError('Relative conductance change >=1 requires reconstruction')
        if (state.electronic_potential_V.shape!=(self.electronic.vertex_count,)
                or state.ionic_potential_V.shape!=(self.ionic.vertex_count,)):
            raise ValueError('State belongs to a different graph')
        v=np.r_[state.electronic_potential_V[self.eop.terminals]-self.offset,
                state.ionic_potential_V[self.iop.terminals]]
        if np.max(abs(v[self.fixed]),initial=0)>1e-10:
            raise ValueError('State does not use this problem gauge')
        evaluated=self._fields(v,state.discharge_current_A)
        if evaluated is None:raise ValueError('Invalid cached state')
        eta,j,_,residual,_=evaluated
        jscale=max(float(np.max(abs(j),initial=0)),abs(state.discharge_current_A),1e-30)
        roundoff=64*np.finfo(float).eps*max(abs(self.offset),1.)*float(np.max(self.amplitude*self.k,initial=0))
        if (np.max(abs(j-state.reaction_current_A),initial=0)>1e-8*jscale+roundoff
                or not np.allclose(eta,state.overpotential_V,rtol=1e-8,atol=1e-12)):
            raise ValueError('State does not solve the declared reaction laws')
        z=k*(eta+self.faces.equilibrium_V-target_faces.equilibrium_V)
        if not np.isfinite(z).all() or np.max(abs(z),initial=0)>600:
            raise ValueError('Target reaction state exceeds finite BV evaluation range')
        target_j=amplitude*np.sinh(z)
        if not np.isfinite(target_j).all():raise ValueError('Nonfinite target reaction currents')
        residual=residual+self.B.T@(target_j-j)
        residual[self.collector]+=current-state.discharge_current_A
        H=((1-eps)*self.S+self.B.T@sparse.diags(amplitude*k)@self.B).tocsr()[self.free][:,self.free]
        diagonal=H.diagonal()
        if np.any(diagonal<=0):raise ValueError('Unresolved null space in voltage bound')
        scale=1/np.sqrt(diagonal)
        factor=cho_factor(scale[:,None]*H.toarray()*scale[None,:],lower=True,check_finite=True)
        def dual(a):
            b=scale*a
            square=float(b@cho_solve(factor,b))
            if square < -1e-13*float(b@b):raise ArithmeticError('Negative dual norm')
            return np.sqrt(max(square,0.))
        residual_norm=dual(residual[self.free])
        dv=v[self.passive_i]-v[self.passive_j]
        energy=float(self.passive_g@dv**2+self.passive_row_sum@v**2)
        if energy < -1e-12*max(abs(energy),1e-30):raise ArithmeticError('Negative passive graph energy')
        bound=residual_norm+eps/np.sqrt(1-eps)*np.sqrt(max(energy,0.))
        goal=np.zeros(len(self.free));where=np.flatnonzero(self.free==self.collector)
        if len(where)!=1:raise ValueError('Collector voltage must be a free physical unknown')
        goal[where[0]]=1.
        return ReactionPortErrorBound(float(dual(goal)*bound),float(bound),eps,float(residual_norm))


def physical_voxel_network(layout):
    """Remove the opposite *test* electrode from a voxel homogenization graph."""
    network=layout.network
    probes=np.setdiff1d(layout.boundary_terminals,network.anchors)
    keep=~np.isin(network.edges,probes).any(axis=1)
    return ConductingNetwork(network.vertex_count,network.edges[keep],network.conductance_S[keep],
        network.anchors,network.channel,'Physical voxel graph; opposite homogenization test terminal removed')


def voxel_reaction_problem(diffusion,electronic_S_m,ionic_S_m,equilibrium_V,exchange_A_m2,*,
                           temperature_K=298.15,electronic_face_factors=None,ionic_face_factors=None,
                           reduce=True,max_ports=512):
    """Build actual boundary-reaction faces on the supplied common voxel lattice.

    OCP has one value per active material cell; j0 has one value per original FV
    boundary face (or a broadcast scalar). Both electronic and ionic conductivity
    fields are in S/m, not dimensionless efficiency. Returns the problem, the
    two voxel index maps and selected original face indices for flux transfer.
    Ionic spatial concentration and diffusion-potential terms are not included
    in this ohmic benchmark; do not relabel it a concentrated-solution reference.
    """
    from .graph_transport import voxel_network
    fv=diffusion;shape=fv.volume.data.shape
    e=voxel_network(electronic_S_m,fv.volume.spacing,channel='electronic',reservoir_side=1,face_factors=electronic_face_factors)
    i=voxel_network(ionic_S_m,fv.volume.spacing,channel='ionic',reservoir_side=0,face_factors=ionic_face_factors)
    if e.shape!=shape or i.shape!=shape:raise ValueError('Same full material lattice required')
    u=np.asarray(equilibrium_V,float)
    if u.shape!=(len(fv.voxel_ids),) or not np.isfinite(u).all():raise ValueError('Equilibrium potential for every active cell required')
    exchange=np.broadcast_to(np.asarray(exchange_A_m2,float),fv.face_cells.shape)
    coordinates=fv.coordinates_zyx[fv.face_cells]+fv.face_normals[:,::-1].astype(int)
    inside=np.all((coordinates>=0)&(coordinates<np.asarray(shape)),axis=1)
    neighbor=np.zeros(len(inside),int);neighbor[inside]=np.ravel_multi_index(coordinates[inside].T,shape)
    emap=np.full(np.prod(shape),-1,int);emap[e.voxel_ids]=np.arange(len(e.voxel_ids))
    imap=np.full(np.prod(shape),-1,int);imap[i.voxel_ids]=np.arange(len(i.voxel_ids))
    ev=emap[fv.voxel_ids[fv.face_cells]];iv=imap[neighbor]
    selected=np.flatnonzero(inside&fv.reactive&(ev>=0)&(iv>=0))
    if not len(selected):raise ValueError('No explicit electronic/ionic reaction faces')
    faces=ReactionFaces(ev[selected],iv[selected],fv.face_areas[selected],u[fv.face_cells[selected]],exchange[selected],
        'Actual FV active/electrolyte boundary faces; no added crack surface or inferred wetting')
    problem=ReactiveDualNetwork(physical_voxel_network(e),physical_voxel_network(i),faces,
        electronic_collector=int(e.network.anchors[0]),ionic_reservoir=int(i.network.anchors[0]),
        temperature_K=temperature_K,reduce=reduce,max_ports=max_ports)
    return problem,e,i,selected
