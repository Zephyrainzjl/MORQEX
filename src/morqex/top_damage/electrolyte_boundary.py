"""Exact finite-volume species-flux boundary for the Li-metal half-cell.

At the metal/separator face N_Li = i/F; at the cathode collector N_Li = 0.
Prescribing a concentration-gradient BC and separately extrapolating nonlinear
D(c) does not enforce this algebraic flux identity on a coarse FV mesh.
Only exterior faces are replaced; all internal DFN fluxes remain unchanged.
"""
import numpy as np
from scipy import sparse
import pybamm as pb
from pybamm.models.submodels.electrolyte_diffusion.full_diffusion import Full


class _Flux:
    __name__='morqex_prescribed_exterior_species_flux'
    def __call__(self,flux,left,right):
        import casadi as ca
        if isinstance(flux,(ca.MX,ca.SX,ca.DM)):return ca.vertcat(left[0],flux[1:-1],right[0])
        a=np.asarray(flux)
        if np.isnan(a).any():return np.full(a.shape,np.nan)
        return np.r_[np.asarray(left).ravel()[0],a.ravel()[1:-1],np.asarray(right).ravel()[0]][:,None]


class BoundaryFlux(pb.Function):
    def __init__(self,flux,left,right):
        if not isinstance(left,pb.Symbol):left=pb.Scalar(left)
        if not isinstance(right,pb.Symbol):right=pb.Scalar(right)
        if not left.domain:left=pb.FullBroadcast(left,flux.domain,'current collector')
        elif left.domain!=flux.domain:left=pb.PrimaryBroadcast(left,flux.domain)
        if not right.domain:right=pb.FullBroadcast(right,flux.domain,'current collector')
        elif right.domain!=flux.domain:right=pb.PrimaryBroadcast(right,flux.domain)
        super().__init__(_Flux(),flux,left,right,name='function (MORQEX prescribed boundary flux)')
    def _evaluates_on_edges(self,dimension):return dimension=='primary'
    def _evaluate_for_shape(self):return np.full(self.children[0].evaluate_for_shape().shape,np.nan)
    def _function_new_copy(self,children):return BoundaryFlux(*children)
    def create_copy(self,new_children=None,perform_simplifications=True):return BoundaryFlux(*self._children_for_copying(new_children))
    def _function_jac(self,children_jacs):
        n=self.children[0].size
        matrices=[sparse.diags(np.r_[0.,np.ones(n-2),0.],format='csr'),
            sparse.coo_matrix(([1.],([0],[0])),shape=(n,self.children[1].size)).tocsr(),
            sparse.coo_matrix(([1.],([n-1],[0])),shape=(n,self.children[2].size)).tocsr()]
        result=sum((pb.Matrix(a)@jac for a,jac in zip(matrices,children_jacs)),pb.Scalar(0.));result.clear_domains();return result


class ConservativeHalfCellElectrolyte(Full):
    def get_coupled_variables(self,v):
        v=super().get_coupled_variables(v)
        if self.options.whole_cell_domains[0]!='separator':raise ValueError('This boundary model requires a planar Li-metal half-cell')
        flux=v['Current collector current density [A.m-2]']/self.param.F
        ce=v['Electrolyte concentration [mol.m-3]'];T=v['Cell temperature [K]']
        transference=pb.boundary_value(self.param.t_plus(ce,T),'left')
        v['Electrolyte diffusion flux [mol.m-2.s-1]']=BoundaryFlux(v['Electrolyte diffusion flux [mol.m-2.s-1]'],(1-transference)*flux,0.)
        v['Electrolyte flux [mol.m-2.s-1]']=BoundaryFlux(v['Electrolyte flux [mol.m-2.s-1]'],flux,0.)
        return v
