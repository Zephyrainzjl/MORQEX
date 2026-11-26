"""Conservative face electrical-work flux for the DFN finite-volume mesh.

Use one potential on each shared interface and actual collector potentials at
external current-carrying faces. The telescoping flux gives the terminal power
without a mesh-dependent lumped correction or removing physical safety events.
"""
import numpy as np
from scipy import sparse
import pybamm as pb


class _Faces:
    __name__='morqex_energy_face_potential'
    def __call__(self,phi,left,right):
        import casadi as ca
        if isinstance(phi,(ca.MX,ca.SX,ca.DM)):
            return ca.vertcat(left[0],.5*(phi[:-1]+phi[1:]),right[-1])
        a=np.asarray(phi)
        if np.isnan(a).any():return np.full((a.size+1,1),np.nan)
        return np.concatenate([np.asarray(left).reshape(-1)[:1],.5*(a.ravel()[:-1]+a.ravel()[1:]),np.asarray(right).reshape(-1)[-1:]])[:,None]


class FacePotential(pb.Function):
    def __init__(self,phi,left,right):
        if not isinstance(left,pb.Symbol):left=pb.Scalar(left)
        if not isinstance(right,pb.Symbol):right=pb.Scalar(right)
        if not left.domain:left=pb.FullBroadcast(left,phi.domain,'current collector')
        elif left.domain!=phi.domain:left=pb.PrimaryBroadcast(left,phi.domain)
        if not right.domain:right=pb.FullBroadcast(right,phi.domain,'current collector')
        elif right.domain!=phi.domain:right=pb.PrimaryBroadcast(right,phi.domain)
        super().__init__(_Faces(),phi,left,right,name='function (MORQEX conservative face potential)')
    def _evaluates_on_edges(self,dimension):return dimension=='primary'
    def _evaluate_for_shape(self):
        shape=self.children[0].evaluate_for_shape().shape
        # Before discretization PyBaMM deliberately gives nodes and edges the
        # same dummy domain size. After discretization the edge has N+1 entries.
        actual=self.children[0].has_symbol_of_classes(pb.StateVector)
        return np.full((shape[0]+int(actual),1),np.nan)
    def _function_new_copy(self,children):return FacePotential(*children)
    def create_copy(self,new_children=None,perform_simplifications=True):return FacePotential(*self._children_for_copying(new_children))
    def _function_jac(self,children_jacs):
        n=self.children[0].size
        row=np.repeat(np.arange(1,n),2);col=np.column_stack([np.arange(n-1),np.arange(1,n)]).ravel()
        matrices=[sparse.coo_matrix((np.full(len(row),.5),(row,col)),shape=(n+1,n)).tocsr(),
            sparse.coo_matrix(([1.],([0],[0])),shape=(n+1,n)).tocsr(),
            sparse.coo_matrix(([1.],([n],[n-1])),shape=(n+1,n)).tocsr()]
        result=sum((pb.Matrix(matrix)@jac for matrix,jac in zip(matrices,children_jacs)),pb.Scalar(0.))
        result.clear_domains();return result


def conservative_macro_heat(v,port):
    domains=['negative electrode','separator','positive electrode']
    current=v['Electrolyte current density [A.m-2]']
    # Build ionic work on the *whole* mesh so internal faces have exactly one
    # potential and one flux. Exterior ionic current is zero in a full DFN cell.
    ionic=-pb.div(FacePotential(v['Electrolyte potential [V]'],0.,0.)*current)
    heat=[]
    for i,domain in enumerate(domains):
        if i!=1:
            d=domain.split()[0];D=d.capitalize();cc=v[D+' current collector potential [V]']
            flux=FacePotential(v[D+' electrode potential [V]'],cc,cc)*v[D+' electrode current density [A.m-2]']
            heat.append(-pb.div(flux))
        else:heat.append(0*v['Separator temperature [K]'])
    return ionic+pb.concatenation(*heat)-pb.concatenation(*port)
