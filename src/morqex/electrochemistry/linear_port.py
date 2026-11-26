"""Exact batched linear ports with compact symbolic/CasADi representation.

Coefficients are part of the symbol identity. A plain unnamed Function would
alias distinct constitutive rows and silently give incorrect Jacobians.
"""
import hashlib
import numpy as np
import pybamm as pb
_PORT_FUNCTIONS={}


class _WeightedSum:
    __name__='morqex_weighted_sum'
    def __init__(self,weights):self.weights=np.asarray(weights,float)
    def __call__(self,*values):
        import casadi as ca
        if any(isinstance(v,(ca.MX,ca.SX,ca.DM)) for v in values):
            n=max(int(v.shape[0]) for v in values)
            values=[ca.repmat(v,n,1) if v.numel()==1 and n>1 else v for v in values]
            key=self.weights.tobytes()
            if key not in _PORT_FUNCTIONS:
                z=ca.MX.sym('local_states',len(self.weights))
                identity=hashlib.sha256(key).hexdigest()[:20]
                _PORT_FUNCTIONS[key]=ca.Function('morqex_linear_'+identity,[z],[ca.DM(self.weights).T@z],{'never_inline':True,'jac_penalty':-1})
            return _PORT_FUNCTIONS[key](ca.horzcat(*values).T).T
        return np.stack(np.broadcast_arrays(*values),axis=-1)@self.weights


class LinearPort(pb.Function):
    def __init__(self,weights,*children):
        self.weights=np.asarray(weights,float).copy();self.weights.flags.writeable=False
        identity=hashlib.sha256(self.weights.tobytes()).hexdigest()
        super().__init__(_WeightedSum(self.weights),*children,name='function (MORQEX linear '+identity+')')
    def _function_diff(self,children,idx):return pb.Scalar(float(self.weights[int(idx)]))
    def _function_new_copy(self,children):return LinearPort(self.weights,*children)
    def create_copy(self,new_children=None,perform_simplifications=True):
        return LinearPort(self.weights,*self._children_for_copying(new_children))


def linear_combination(weights,values):
    values=list(values)
    if any(isinstance(w,pb.Symbol) for w in weights):
        # Spatial material coefficients already own their electrode domain.
        # Multiplication is pointwise, so each DFN location uses its own row.
        return sum((coefficient(w)*v for w,v in zip(weights,values)),pb.Scalar(0))
    weights=np.asarray(weights,float);values=list(values)
    if len(weights)!=len(values) or not np.isfinite(weights).all():raise ValueError('Finite matching port coefficients required')
    ids=np.flatnonzero(weights)
    if not len(ids):return pb.Scalar(0.)
    if len(ids)==1:return float(weights[ids[0]])*values[ids[0]]
    return LinearPort(weights[ids],*(values[k] for k in ids))


def coefficient(value):
    """Preserve a spatial coefficient; keep legacy scalar constants scalar."""
    return value if isinstance(value,pb.Symbol) else float(value)


def reaction_mask(operator,k):
    masks=getattr(operator,'reaction_masks',None)
    return masks[k] if masks is not None else float(operator.reaction_area[k]>0)
